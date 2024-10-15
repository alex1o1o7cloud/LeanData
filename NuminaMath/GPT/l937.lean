import Mathlib

namespace NUMINAMATH_GPT_painting_methods_correct_l937_93750

def num_painting_methods : Nat := 72

theorem painting_methods_correct :
  let vertices : Fin 4 := by sorry -- Ensures there are four vertices
  let edges : Fin 4 := by sorry -- Ensures each edge has different colored endpoints
  let available_colors : Fin 4 := by sorry -- Ensures there are four available colors
  num_painting_methods = 72 :=
sorry

end NUMINAMATH_GPT_painting_methods_correct_l937_93750


namespace NUMINAMATH_GPT_jerome_time_6_hours_l937_93703

theorem jerome_time_6_hours (T: ℝ) (s_J: ℝ) (t_N: ℝ) (s_N: ℝ)
  (h1: s_J = 4) 
  (h2: t_N = 3) 
  (h3: s_N = 8): T = 6 :=
by
  -- Given s_J = 4, t_N = 3, and s_N = 8,
  -- we need to prove that T = 6.
  sorry

end NUMINAMATH_GPT_jerome_time_6_hours_l937_93703


namespace NUMINAMATH_GPT_sum_of_factors_of_30_multiplied_by_2_equals_144_l937_93757

-- We define the factors of 30
def factors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- We define the function to multiply each factor by 2 and sum them
def sum_factors_multiplied_by_2 (factors : List ℕ) : ℕ :=
  factors.foldl (λ acc x => acc + 2 * x) 0

-- The final statement to be proven
theorem sum_of_factors_of_30_multiplied_by_2_equals_144 :
  sum_factors_multiplied_by_2 factors_of_30 = 144 :=
by sorry

end NUMINAMATH_GPT_sum_of_factors_of_30_multiplied_by_2_equals_144_l937_93757


namespace NUMINAMATH_GPT_problem1_problem2_l937_93764

-- Definitions and Lean statement for Problem 1
noncomputable def curve1 (x : ℝ) : ℝ := x / (2 * x - 1)
def point1 : ℝ × ℝ := (1, 1)
noncomputable def tangent_line1 (x y : ℝ) : Prop := x + y - 2 = 0

theorem problem1 : tangent_line1 (point1.fst) (curve1 (point1.fst)) :=
sorry -- proof goes here

-- Definitions and Lean statement for Problem 2
def parabola (x : ℝ) : ℝ := x^2
def point2 : ℝ × ℝ := (2, 3)
noncomputable def tangent_line2a (x y : ℝ) : Prop := 2 * x - y - 1 = 0
noncomputable def tangent_line2b (x y : ℝ) : Prop := 6 * x - y - 9 = 0

theorem problem2 : (tangent_line2a point2.fst point2.snd ∨ tangent_line2b point2.fst point2.snd) :=
sorry -- proof goes here

end NUMINAMATH_GPT_problem1_problem2_l937_93764


namespace NUMINAMATH_GPT_fg_evaluation_l937_93766

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_evaluation : f (g 3) = 97 := by
  sorry

end NUMINAMATH_GPT_fg_evaluation_l937_93766


namespace NUMINAMATH_GPT_factor_3a3_minus_6a2_plus_3a_factor_a2_minus_b2_x_minus_y_factor_16a_plus_b_sq_minus_9a_minus_b_sq_l937_93762

-- First factorization problem
theorem factor_3a3_minus_6a2_plus_3a (a : ℝ) : 
  3 * a ^ 3 - 6 * a ^ 2 + 3 * a = 3 * a * (a - 1) ^ 2 :=
by sorry

-- Second factorization problem
theorem factor_a2_minus_b2_x_minus_y (a b x y : ℝ) : 
  a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a - b) * (a + b) :=
by sorry

-- Third factorization problem
theorem factor_16a_plus_b_sq_minus_9a_minus_b_sq (a b : ℝ) : 
  16 * (a + b) ^ 2 - 9 * (a - b) ^ 2 = (a + 7 * b) * (7 * a + b) :=
by sorry

end NUMINAMATH_GPT_factor_3a3_minus_6a2_plus_3a_factor_a2_minus_b2_x_minus_y_factor_16a_plus_b_sq_minus_9a_minus_b_sq_l937_93762


namespace NUMINAMATH_GPT_sheets_of_paper_in_each_box_l937_93765

theorem sheets_of_paper_in_each_box (S E : ℕ) 
  (h1 : S - E = 70) 
  (h2 : 4 * (E - 20) = S) : 
  S = 120 := 
by 
  sorry

end NUMINAMATH_GPT_sheets_of_paper_in_each_box_l937_93765


namespace NUMINAMATH_GPT_initial_amount_simple_interest_l937_93748

theorem initial_amount_simple_interest 
  (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hA : A = 1125)
  (hR : R = 0.10)
  (hT : T = 5) :
  A = P * (1 + R * T) → P = 750 := 
by
  sorry

end NUMINAMATH_GPT_initial_amount_simple_interest_l937_93748


namespace NUMINAMATH_GPT_compare_star_values_l937_93796

def star (A B : ℤ) : ℤ := A * B - A / B

theorem compare_star_values : star 6 (-3) < star 4 (-4) := by
  sorry

end NUMINAMATH_GPT_compare_star_values_l937_93796


namespace NUMINAMATH_GPT_jaya_amitabh_number_of_digits_l937_93722

-- Definitions
def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def digit_sum (n1 n2 : ℕ) : ℕ :=
  let (d1, d2) := (n1 % 10, n1 / 10)
  let (d3, d4) := (n2 % 10, n2 / 10)
  d1 + d2 + d3 + d4
def append_ages (j a : ℕ) : ℕ := 1000 * (j / 10) + 100 * (j % 10) + 10 * (a / 10) + (a % 10)
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Main theorem
theorem jaya_amitabh_number_of_digits 
  (j a : ℕ) 
  (hj : is_two_digit_number j)
  (ha : is_two_digit_number a)
  (h_sum : digit_sum j a = 7)
  (h_square : is_perfect_square (append_ages j a)) : 
  ∃ n : ℕ, String.length (toString (append_ages j a)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_jaya_amitabh_number_of_digits_l937_93722


namespace NUMINAMATH_GPT_perp_line_parallel_plane_perp_line_l937_93793

variable {Line : Type} {Plane : Type}
variable (a b : Line) (α β : Plane)
variable (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (parallel_lines : Line → Line → Prop)

-- Conditions
variable (non_coincident_lines : ¬(a = b))
variable (non_coincident_planes : ¬(α = β))
variable (a_perp_α : perpendicular a α)
variable (b_par_α : parallel b α)

-- Prove
theorem perp_line_parallel_plane_perp_line :
  perpendicular a α ∧ parallel b α → parallel_lines a b :=
sorry

end NUMINAMATH_GPT_perp_line_parallel_plane_perp_line_l937_93793


namespace NUMINAMATH_GPT_fraction_sum_l937_93775

theorem fraction_sum (n : ℕ) (a : ℚ) (sum_fraction : a = 1/12) (number_of_fractions : n = 450) : 
  ∀ (f : ℚ), (n * f = a) → (f = 1/5400) :=
by
  intros f H
  sorry

end NUMINAMATH_GPT_fraction_sum_l937_93775


namespace NUMINAMATH_GPT_y_expression_l937_93735

theorem y_expression (x y : ℝ) (h : 4 * x + y = 9) : y = 9 - 4 * x := 
by
  sorry

end NUMINAMATH_GPT_y_expression_l937_93735


namespace NUMINAMATH_GPT_condition_for_equation_l937_93715

theorem condition_for_equation (a b c d : ℝ) 
  (h : (a^2 + b) / (b + c^2) = (c^2 + d) / (d + a^2)) : 
  a = c ∨ a^2 + d + 2 * b = 0 :=
by
  sorry

end NUMINAMATH_GPT_condition_for_equation_l937_93715


namespace NUMINAMATH_GPT_cheesecakes_sold_l937_93713

theorem cheesecakes_sold
  (initial_display : Nat)
  (initial_fridge : Nat)
  (left_to_sell : Nat)
  (total_cheesecakes := initial_display + initial_fridge)
  (total_after_sales : Nat) :
  initial_display = 10 →
  initial_fridge = 15 →
  left_to_sell = 18 →
  total_after_sales = total_cheesecakes - left_to_sell →
  total_after_sales = 7 := sorry

end NUMINAMATH_GPT_cheesecakes_sold_l937_93713


namespace NUMINAMATH_GPT_sum_of_roots_l937_93754

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l937_93754


namespace NUMINAMATH_GPT_no_person_has_fewer_than_6_cards_l937_93786

-- Definition of the problem and conditions
def cards := 60
def people := 10
def cards_per_person := cards / people

-- Lean statement of the proof problem
theorem no_person_has_fewer_than_6_cards
  (cards_dealt : cards = 60)
  (people_count : people = 10)
  (even_distribution : cards % people = 0) :
  ∀ person, person < people → cards_per_person = 6 ∧ person < people → person = 0 := 
by 
  sorry

end NUMINAMATH_GPT_no_person_has_fewer_than_6_cards_l937_93786


namespace NUMINAMATH_GPT_parabola_equation_line_AB_fixed_point_min_area_AMBN_l937_93763

-- Prove that the equation of the parabola is y^2 = 4x given the focus (1,0) for y^2 = 2px
theorem parabola_equation (p : ℝ) (h : p > 0) (foc : (1, 0) = (1, 2*p*1/4)):
  (∀ x y: ℝ, y^2 = 4*x ↔ y^2 = 2*p*x) := sorry

-- Prove that line AB passes through fixed point T(2,0) given conditions
theorem line_AB_fixed_point (A B : ℝ × ℝ) (hA : A.2^2 = 4*A.1) 
    (hB : B.2^2 = 4*B.1) (h : A.1*B.1 + A.2*B.2 = -4) :
  ∃ T : ℝ × ℝ, T = (2, 0) := sorry

-- Prove that minimum value of area Quadrilateral AMBN is 48
theorem min_area_AMBN (T : ℝ × ℝ) (A B M N : ℝ × ℝ)
    (hT : T = (2, 0)) (hA : A.2^2 = 4*A.1) (hB : B.2^2 = 4*B.1)
    (hM : M.2^2 = 4*M.1) (hN : N.2^2 = 4*N.1)
    (line_AB : A.1 * B.1 + A.2 * B.2 = -4) :
  ∀ (m : ℝ), T.2 = -(1/m)*T.1 + 2 → 
  ((1+m^2) * (1+1/m^2)) * ((m^2 + 2) * (1/m^2 + 2)) = 256 → 
  8 * 48 = 48 := sorry

end NUMINAMATH_GPT_parabola_equation_line_AB_fixed_point_min_area_AMBN_l937_93763


namespace NUMINAMATH_GPT_angle_value_is_140_l937_93726

-- Definitions of conditions
def angle_on_straight_line_degrees (x y : ℝ) : Prop := x + y = 180

-- Main statement in Lean
theorem angle_value_is_140 (x : ℝ) (h₁ : angle_on_straight_line_degrees 40 x) : x = 140 :=
by
  -- Proof is omitted (not required as per instructions)
  sorry

end NUMINAMATH_GPT_angle_value_is_140_l937_93726


namespace NUMINAMATH_GPT_curve_C1_parametric_equiv_curve_C2_general_equiv_curve_C3_rectangular_equiv_max_distance_C2_to_C3_l937_93768

-- Definitions of the curves
def curve_C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1
def curve_C2_parametric (theta : ℝ) (x y : ℝ) : Prop := (x = 4 * Real.cos theta) ∧ (y = 3 * Real.sin theta)
def curve_C3_polar (rho theta : ℝ) : Prop := rho * (Real.cos theta - 2 * Real.sin theta) = 7

-- Proving the mathematical equivalence:
theorem curve_C1_parametric_equiv (t : ℝ) : ∃ x y, curve_C1 x y ∧ (x = 3 + Real.cos t) ∧ (y = 2 + Real.sin t) :=
by sorry

theorem curve_C2_general_equiv (x y : ℝ) : (∃ theta, curve_C2_parametric theta x y) ↔ (x^2 / 16 + y^2 / 9 = 1) :=
by sorry

theorem curve_C3_rectangular_equiv (x y : ℝ) : (∃ rho theta, x = rho * Real.cos theta ∧ y = rho * Real.sin theta ∧ curve_C3_polar rho theta) ↔ (x - 2 * y - 7 = 0) :=
by sorry

theorem max_distance_C2_to_C3 : ∃ (d : ℝ), d = (2 * Real.sqrt 65 + 7 * Real.sqrt 5) / 5 :=
by sorry

end NUMINAMATH_GPT_curve_C1_parametric_equiv_curve_C2_general_equiv_curve_C3_rectangular_equiv_max_distance_C2_to_C3_l937_93768


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l937_93704

theorem infinite_geometric_series_sum (p q : ℝ)
  (h : (∑' n : ℕ, p / q ^ (n + 1)) = 5) :
  (∑' n : ℕ, p / (p^2 + q) ^ (n + 1)) = 5 * (q - 1) / (25 * q^2 - 50 * q + 26) :=
sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l937_93704


namespace NUMINAMATH_GPT_fraction_of_females_l937_93721

def local_soccer_league_female_fraction : Prop :=
  ∃ (males_last_year females_last_year : ℕ),
    males_last_year = 30 ∧
    (1.10 * males_last_year : ℝ) = 33 ∧
    (males_last_year + females_last_year : ℝ) * 1.15 = 52 ∧
    (females_last_year : ℝ) * 1.25 = 19 ∧
    (33 + 19 = 52)

theorem fraction_of_females
  : local_soccer_league_female_fraction → 
    ∃ (females fraction : ℝ),
    females = 19 ∧ 
    fraction = 19 / 52 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_females_l937_93721


namespace NUMINAMATH_GPT_lowest_selling_price_l937_93712

/-- Define the variables and constants -/
def production_cost_per_component := 80
def shipping_cost_per_component := 7
def fixed_costs_per_month := 16500
def components_per_month := 150

/-- Define the total variable cost -/
def total_variable_cost (production_cost_per_component shipping_cost_per_component : ℕ) (components_per_month : ℕ) :=
  (production_cost_per_component + shipping_cost_per_component) * components_per_month

/-- Define the total cost -/
def total_cost (variable_cost fixed_costs_per_month : ℕ) :=
  variable_cost + fixed_costs_per_month

/-- Define the lowest price per component -/
def lowest_price_per_component (total_cost components_per_month : ℕ) :=
  total_cost / components_per_month

/-- The main theorem to prove the lowest selling price required to cover all costs -/
theorem lowest_selling_price (production_cost shipping_cost fixed_costs components : ℕ)
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 7)
  (h3 : fixed_costs = 16500)
  (h4 : components = 150) :
  lowest_price_per_component (total_cost (total_variable_cost production_cost shipping_cost components) fixed_costs) components = 197 :=
by
  sorry

end NUMINAMATH_GPT_lowest_selling_price_l937_93712


namespace NUMINAMATH_GPT_gcd_m_n_l937_93736

def m : ℕ := 333333
def n : ℕ := 888888888

theorem gcd_m_n : Nat.gcd m n = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_l937_93736


namespace NUMINAMATH_GPT_find_b_l937_93744

-- Define the slopes of the two lines derived from the given conditions
noncomputable def slope1 := -2 / 3
noncomputable def slope2 (b : ℚ) := -b / 3

-- Lean 4 statement to prove that for the lines to be perpendicular, b must be -9/2
theorem find_b (b : ℚ) (h_perpendicular: slope1 * slope2 b = -1) : b = -9 / 2 := by
  sorry

end NUMINAMATH_GPT_find_b_l937_93744


namespace NUMINAMATH_GPT_number_of_other_values_l937_93707

def orig_value : ℕ := 2 ^ (2 ^ (2 ^ 2))

def other_values : Finset ℕ :=
  {2 ^ (2 ^ (2 ^ 2)), 2 ^ ((2 ^ 2) ^ 2), ((2 ^ 2) ^ 2) ^ 2, (2 ^ (2 ^ 2)) ^ 2, (2 ^ 2) ^ (2 ^ 2)}

theorem number_of_other_values :
  other_values.erase orig_value = {256} :=
by
  sorry

end NUMINAMATH_GPT_number_of_other_values_l937_93707


namespace NUMINAMATH_GPT_mr_arevalo_change_l937_93792

-- Definitions for the costs of the food items
def cost_smoky_salmon : ℤ := 40
def cost_black_burger : ℤ := 15
def cost_chicken_katsu : ℤ := 25

-- Definitions for the service charge and tip percentages
def service_charge_percent : ℝ := 0.10
def tip_percent : ℝ := 0.05

-- Definition for the amount Mr. Arevalo pays
def amount_paid : ℤ := 100

-- Calculation for total food cost
def total_food_cost : ℤ := cost_smoky_salmon + cost_black_burger + cost_chicken_katsu

-- Calculation for service charge
def service_charge : ℝ := service_charge_percent * total_food_cost

-- Calculation for tip
def tip : ℝ := tip_percent * total_food_cost

-- Calculation for the final bill amount
def final_bill_amount : ℝ := total_food_cost + service_charge + tip

-- Calculation for the change
def change : ℝ := amount_paid - final_bill_amount

-- Proof statement
theorem mr_arevalo_change : change = 8 := by
  sorry

end NUMINAMATH_GPT_mr_arevalo_change_l937_93792


namespace NUMINAMATH_GPT_inequality_sum_l937_93739

theorem inequality_sum 
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≥ a2)
  (h2 : a2 ≥ a3)
  (h3 : a3 > 0)
  (h4 : b1 ≥ b2)
  (h5 : b2 ≥ b3)
  (h6 : b3 > 0)
  (h7 : a1 * a2 * a3 = b1 * b2 * b3)
  (h8 : a1 - a3 ≤ b1 - b3) :
  a1 + a2 + a3 ≤ 2 * (b1 + b2 + b3) := 
sorry

end NUMINAMATH_GPT_inequality_sum_l937_93739


namespace NUMINAMATH_GPT_orthogonal_circles_l937_93760

theorem orthogonal_circles (R1 R2 d : ℝ) :
  (d^2 = R1^2 + R2^2) ↔ (d^2 = R1^2 + R2^2) :=
by sorry

end NUMINAMATH_GPT_orthogonal_circles_l937_93760


namespace NUMINAMATH_GPT_contractor_absent_days_proof_l937_93770

def contractor_absent_days (x y : ℝ) : Prop :=
  x + y = 30 ∧ 25 * x - 7.5 * y = 425

theorem contractor_absent_days_proof : ∃ (y : ℝ), contractor_absent_days x y ∧ y = 10 :=
by
  sorry

end NUMINAMATH_GPT_contractor_absent_days_proof_l937_93770


namespace NUMINAMATH_GPT_test_tube_full_with_two_amoebas_l937_93746

-- Definition: Each amoeba doubles in number every minute.
def amoeba_doubling (initial : Nat) (minutes : Nat) : Nat :=
  initial * 2 ^ minutes

-- Condition: Starting with one amoeba, the test tube is filled in 60 minutes.
def time_to_fill_one_amoeba := 60

-- Theorem: If two amoebas are placed in the test tube, it takes 59 minutes to fill.
theorem test_tube_full_with_two_amoebas : amoeba_doubling 2 59 = amoeba_doubling 1 time_to_fill_one_amoeba :=
by sorry

end NUMINAMATH_GPT_test_tube_full_with_two_amoebas_l937_93746


namespace NUMINAMATH_GPT_highest_y_coordinate_l937_93753

theorem highest_y_coordinate (x y : ℝ) (h : (x^2 / 49 + (y-3)^2 / 25 = 0)) : y = 3 :=
by
  sorry

end NUMINAMATH_GPT_highest_y_coordinate_l937_93753


namespace NUMINAMATH_GPT_series_pattern_l937_93745

theorem series_pattern :
    (3 / (1 * 2) * (1 / 2) + 4 / (2 * 3) * (1 / 2^2) + 5 / (3 * 4) * (1 / 2^3) + 6 / (4 * 5) * (1 / 2^4) + 7 / (5 * 6) * (1 / 2^5)) 
    = (1 - 1 / (6 * 2^5)) :=
  sorry

end NUMINAMATH_GPT_series_pattern_l937_93745


namespace NUMINAMATH_GPT_joan_mortgage_payment_l937_93734

noncomputable def geometric_series_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (1 - r^n) / (1 - r)

theorem joan_mortgage_payment : 
  ∃ n : ℕ, geometric_series_sum 100 3 n = 109300 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_joan_mortgage_payment_l937_93734


namespace NUMINAMATH_GPT_initial_outlay_is_10000_l937_93780

theorem initial_outlay_is_10000 
  (I : ℝ)
  (manufacturing_cost_per_set : ℝ := 20)
  (selling_price_per_set : ℝ := 50)
  (num_sets : ℝ := 500)
  (profit : ℝ := 5000) :
  profit = (selling_price_per_set * num_sets) - (I + manufacturing_cost_per_set * num_sets) → I = 10000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_outlay_is_10000_l937_93780


namespace NUMINAMATH_GPT_find_pairs_l937_93740

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 1 + 5^a = 6^b → (a, b) = (1, 1) := by
  sorry

end NUMINAMATH_GPT_find_pairs_l937_93740


namespace NUMINAMATH_GPT_equal_number_of_coins_l937_93795

theorem equal_number_of_coins (x : ℕ) (hx : 1 * x + 5 * x + 10 * x + 25 * x + 100 * x = 305) : x = 2 :=
sorry

end NUMINAMATH_GPT_equal_number_of_coins_l937_93795


namespace NUMINAMATH_GPT_remainder_of_number_divided_by_39_l937_93723

theorem remainder_of_number_divided_by_39 
  (N : ℤ) 
  (k m : ℤ) 
  (h₁ : N % 195 = 79) 
  (h₂ : N % 273 = 109) : 
  N % 39 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_of_number_divided_by_39_l937_93723


namespace NUMINAMATH_GPT_f_of_10_is_20_l937_93701

theorem f_of_10_is_20 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (3 * x + 1) = x^2 + 3 * x + 2) : f 10 = 20 :=
  sorry

end NUMINAMATH_GPT_f_of_10_is_20_l937_93701


namespace NUMINAMATH_GPT_line_eq_of_midpoint_and_hyperbola_l937_93706

theorem line_eq_of_midpoint_and_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : 9 * (8 : ℝ)^2 - 16 * (3 : ℝ)^2 = 144)
    (h2 : x1 + x2 = 16) (h3 : y1 + y2 = 6) (h4 : 9 * x1^2 - 16 * y1^2 = 144) (h5 : 9 * x2^2 - 16 * y2^2 = 144) :
    3 * (8 : ℝ) - 2 * (3 : ℝ) - 18 = 0 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_line_eq_of_midpoint_and_hyperbola_l937_93706


namespace NUMINAMATH_GPT_bowlfuls_per_box_l937_93787

def clusters_per_spoonful : ℕ := 4
def spoonfuls_per_bowl : ℕ := 25
def clusters_per_box : ℕ := 500

theorem bowlfuls_per_box : clusters_per_box / (clusters_per_spoonful * spoonfuls_per_bowl) = 5 :=
by
  sorry

end NUMINAMATH_GPT_bowlfuls_per_box_l937_93787


namespace NUMINAMATH_GPT_intersection_M_N_l937_93747

def M : Set ℝ := { x | x < 2017 }
def N : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } := 
by 
  sorry

end NUMINAMATH_GPT_intersection_M_N_l937_93747


namespace NUMINAMATH_GPT_original_rice_amount_l937_93771

theorem original_rice_amount (r : ℚ) (x y : ℚ)
  (h1 : r = 3/5)
  (h2 : x + y = 10)
  (h3 : x + r * y = 7) : 
  x + y = 10 ∧ x + 3/5 * y = 7 := 
by
  sorry

end NUMINAMATH_GPT_original_rice_amount_l937_93771


namespace NUMINAMATH_GPT_resulting_chemical_percentage_l937_93755

theorem resulting_chemical_percentage 
  (init_solution_pct : ℝ) (replacement_frac : ℝ) (replacing_solution_pct : ℝ) (resulting_solution_pct : ℝ) : 
  init_solution_pct = 0.85 →
  replacement_frac = 0.8181818181818182 →
  replacing_solution_pct = 0.30 →
  resulting_solution_pct = 0.40 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_resulting_chemical_percentage_l937_93755


namespace NUMINAMATH_GPT_sequence_general_formula_l937_93759

theorem sequence_general_formula (a : ℕ → ℕ) 
  (h₁ : a 1 = 2)
  (h₂ : ∀ n, a (n + 1) = 2 * a n - 1) :
  ∀ n, a n = 1 + 2^(n - 1) := 
sorry

end NUMINAMATH_GPT_sequence_general_formula_l937_93759


namespace NUMINAMATH_GPT_volume_increased_by_3_l937_93779

theorem volume_increased_by_3 {l w h : ℝ}
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + l * h = 925)
  (h3 : l + w + h = 60) :
  (l + 3) * (w + 3) * (h + 3) = 8342 := 
by
  sorry

end NUMINAMATH_GPT_volume_increased_by_3_l937_93779


namespace NUMINAMATH_GPT_find_subtracted_value_l937_93741

theorem find_subtracted_value (N V : ℤ) (hN : N = 12) (h : 4 * N - 3 = 9 * (N - V)) : V = 7 := 
by
  sorry

end NUMINAMATH_GPT_find_subtracted_value_l937_93741


namespace NUMINAMATH_GPT_min_sum_xy_l937_93794

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (pos_x : 0 < x) (pos_y : 0 < y)
  (h : (1 : ℚ) / x + 1 / y = 1 / 12) : x + y = 49 :=
sorry

end NUMINAMATH_GPT_min_sum_xy_l937_93794


namespace NUMINAMATH_GPT_least_possible_n_l937_93778

noncomputable def d (n : ℕ) := 105 * n - 90

theorem least_possible_n :
  ∀ n : ℕ, d n > 0 → (45 - (d n + 90) / n = 150) → n ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_n_l937_93778


namespace NUMINAMATH_GPT_find_common_difference_l937_93710

section
variables (a1 a7 a8 a9 S5 S6 : ℚ) (d : ℚ)

/-- Given an arithmetic sequence with the sum of the first n terms S_n,
    if S_5 = a_8 + 5 and S_6 = a_7 + a_9 - 5, we need to find the common difference d. -/
theorem find_common_difference
  (h1 : S5 = a8 + 5)
  (h2 : S6 = a7 + a9 - 5)
  (h3 : S5 = 5 / 2 * (2 * a1 + 4 * d))
  (h4 : S6 = 6 / 2 * (2 * a1 + 5 * d))
  (h5 : a8 = a1 + 7 * d)
  (h6 : a7 = a1 + 6 * d)
  (h7 : a9 = a1 + 8 * d):
  d = -55 / 19 :=
by
  sorry
end

end NUMINAMATH_GPT_find_common_difference_l937_93710


namespace NUMINAMATH_GPT_sara_steps_l937_93782

theorem sara_steps (n : ℕ) (h : n^2 ≤ 210) : n = 14 :=
sorry

end NUMINAMATH_GPT_sara_steps_l937_93782


namespace NUMINAMATH_GPT_total_plates_used_l937_93720

-- Definitions from the conditions
def number_of_people := 6
def meals_per_day_per_person := 3
def plates_per_meal_per_person := 2
def number_of_days := 4

-- Statement of the theorem
theorem total_plates_used : number_of_people * meals_per_day_per_person * plates_per_meal_per_person * number_of_days = 144 := 
by
  sorry

end NUMINAMATH_GPT_total_plates_used_l937_93720


namespace NUMINAMATH_GPT_find_values_of_expression_l937_93719

theorem find_values_of_expression (a b : ℝ) 
  (h : (2 * a) / (a + b) + b / (a - b) = 2) : 
  (∃ x : ℝ, x = (3 * a - b) / (a + 5 * b) ∧ (x = 3 ∨ x = 1)) :=
by 
  sorry

end NUMINAMATH_GPT_find_values_of_expression_l937_93719


namespace NUMINAMATH_GPT_training_days_l937_93729

def total_minutes : ℕ := 5 * 60
def minutes_per_day : ℕ := 10 + 20

theorem training_days :
  total_minutes / minutes_per_day = 10 :=
by
  sorry

end NUMINAMATH_GPT_training_days_l937_93729


namespace NUMINAMATH_GPT_domino_covering_l937_93772

theorem domino_covering (m n : ℕ) (m_eq : (m, n) ∈ [(5, 5), (4, 6), (3, 7), (5, 6), (3, 8)]) :
  (m * n % 2 = 1) ↔ (m = 5 ∧ n = 5) ∨ (m = 3 ∧ n = 7) :=
by
  sorry

end NUMINAMATH_GPT_domino_covering_l937_93772


namespace NUMINAMATH_GPT_integer_solutions_count_l937_93773

theorem integer_solutions_count :
  let cond1 (x : ℤ) := -4 * x ≥ 2 * x + 9
  let cond2 (x : ℤ) := -3 * x ≤ 15
  let cond3 (x : ℤ) := -5 * x ≥ x + 22
  ∃ s : Finset ℤ, 
    (∀ x ∈ s, cond1 x ∧ cond2 x ∧ cond3 x) ∧
    (∀ x, cond1 x ∧ cond2 x ∧ cond3 x → x ∈ s) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_GPT_integer_solutions_count_l937_93773


namespace NUMINAMATH_GPT_company_A_profit_l937_93790

-- Define the conditions
def total_profit (x : ℝ) : ℝ := x
def company_B_share (x : ℝ) : Prop := 0.4 * x = 60000
def company_A_percentage : ℝ := 0.6

-- Define the statement to be proved
theorem company_A_profit (x : ℝ) (h : company_B_share x) : 0.6 * x = 90000 := sorry

end NUMINAMATH_GPT_company_A_profit_l937_93790


namespace NUMINAMATH_GPT_train_crossing_time_l937_93761

def train_length := 140
def train_speed_kmph := 45
def bridge_length := 235
def speed_to_mps (kmph : ℕ) : ℕ := (kmph * 1000) / 3600
def total_distance := train_length + bridge_length
def train_speed := speed_to_mps train_speed_kmph
def time_to_cross := total_distance / train_speed

theorem train_crossing_time : time_to_cross = 30 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l937_93761


namespace NUMINAMATH_GPT_simplify_expression_l937_93732

theorem simplify_expression (t : ℝ) (t_ne_zero : t ≠ 0) : (t^5 * t^3) / t^4 = t^4 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l937_93732


namespace NUMINAMATH_GPT_angle_sum_is_180_l937_93789

theorem angle_sum_is_180 (A B C : ℝ) (h_triangle : (A + B + C) = 180) (h_sum : A + B = 90) : C = 90 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_angle_sum_is_180_l937_93789


namespace NUMINAMATH_GPT_div_by_133_l937_93737

theorem div_by_133 (n : ℕ) : 133 ∣ 11^(n+2) + 12^(2*n+1) :=
by sorry

end NUMINAMATH_GPT_div_by_133_l937_93737


namespace NUMINAMATH_GPT_no_solution_exists_l937_93767

theorem no_solution_exists : 
  ¬(∃ x y : ℝ, 2 * x - 3 * y = 7 ∧ 4 * x - 6 * y = 20) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l937_93767


namespace NUMINAMATH_GPT_angle_WYZ_correct_l937_93749

-- Define the angles as constants
def angle_XYZ : ℝ := 36
def angle_XYW : ℝ := 15

-- Theorem statement asserting the solution
theorem angle_WYZ_correct :
  (angle_XYZ - angle_XYW = 21) := 
by
  -- This is where the proof would go, but we use 'sorry' as instructed
  sorry

end NUMINAMATH_GPT_angle_WYZ_correct_l937_93749


namespace NUMINAMATH_GPT_base_5_to_base_10_l937_93776

theorem base_5_to_base_10 : 
  let n : ℕ := 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4 * 5^0
  n = 194 :=
by 
  sorry

end NUMINAMATH_GPT_base_5_to_base_10_l937_93776


namespace NUMINAMATH_GPT_gcd_315_2016_l937_93708

def a : ℕ := 315
def b : ℕ := 2016

theorem gcd_315_2016 : Nat.gcd a b = 63 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_315_2016_l937_93708


namespace NUMINAMATH_GPT_convert_speed_l937_93742

-- Definitions based on the given condition
def kmh_to_mps (kmh : ℝ) : ℝ := kmh * 0.277778

-- Theorem statement
theorem convert_speed : kmh_to_mps 84 = 23.33 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_convert_speed_l937_93742


namespace NUMINAMATH_GPT_david_savings_l937_93717

def lawn_rate_monday : ℕ := 14
def lawn_rate_wednesday : ℕ := 18
def lawn_rate_friday : ℕ := 20
def hours_per_day : ℕ := 2
def weekly_earnings : ℕ := (lawn_rate_monday * hours_per_day) + (lawn_rate_wednesday * hours_per_day) + (lawn_rate_friday * hours_per_day)

def tax_rate : ℚ := 0.10
def tax_paid (earnings : ℚ) : ℚ := earnings * tax_rate

def shoe_price : ℚ := 75
def discount : ℚ := 0.15
def discounted_shoe_price : ℚ := shoe_price * (1 - discount)

def money_remaining (earnings : ℚ) (tax : ℚ) (shoes : ℚ) : ℚ := earnings - tax - shoes

def gift_rate : ℚ := 1 / 3
def money_given_to_mom (remaining : ℚ) : ℚ := remaining * gift_rate

def final_savings (remaining : ℚ) (gift : ℚ) : ℚ := remaining - gift

theorem david_savings : 
  final_savings (money_remaining weekly_earnings (tax_paid weekly_earnings) discounted_shoe_price) 
                (money_given_to_mom (money_remaining weekly_earnings (tax_paid weekly_earnings) discounted_shoe_price)) 
  = 19.90 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_david_savings_l937_93717


namespace NUMINAMATH_GPT_include_both_male_and_female_l937_93769

noncomputable def probability_includes_both_genders (total_students male_students female_students selected_students : ℕ) : ℚ :=
  let total_ways := Nat.choose total_students selected_students
  let all_female_ways := Nat.choose female_students selected_students
  (total_ways - all_female_ways) / total_ways

theorem include_both_male_and_female :
  probability_includes_both_genders 6 2 4 4 = 14 / 15 := 
by
  sorry

end NUMINAMATH_GPT_include_both_male_and_female_l937_93769


namespace NUMINAMATH_GPT_present_value_l937_93718

theorem present_value (BD TD PV : ℝ) (hBD : BD = 42) (hTD : TD = 36)
  (h : BD = TD + (TD^2 / PV)) : PV = 216 :=
sorry

end NUMINAMATH_GPT_present_value_l937_93718


namespace NUMINAMATH_GPT_find_original_faculty_count_l937_93714

variable (F : ℝ)
variable (final_count : ℝ := 195)
variable (first_year_reduction : ℝ := 0.075)
variable (second_year_increase : ℝ := 0.125)
variable (third_year_reduction : ℝ := 0.0325)
variable (fourth_year_increase : ℝ := 0.098)
variable (fifth_year_reduction : ℝ := 0.1465)

theorem find_original_faculty_count (h : F * (1 - first_year_reduction)
                                        * (1 + second_year_increase)
                                        * (1 - third_year_reduction)
                                        * (1 + fourth_year_increase)
                                        * (1 - fifth_year_reduction) = final_count) :
  F = 244 :=
by sorry

end NUMINAMATH_GPT_find_original_faculty_count_l937_93714


namespace NUMINAMATH_GPT_ratio_sharks_to_pelicans_l937_93728

-- Define the conditions given in the problem
def original_pelican_count {P : ℕ} (h : (2/3 : ℚ) * P = 20) : Prop :=
  P = 30

-- Define the final ratio we want to prove
def shark_to_pelican_ratio (sharks pelicans : ℕ) : ℚ :=
  sharks / pelicans

theorem ratio_sharks_to_pelicans
  (P : ℕ) (h : (2/3 : ℚ) * P = 20) (number_sharks : ℕ) (number_pelicans : ℕ)
  (H_sharks : number_sharks = 60) (H_pelicans : number_pelicans = P)
  (H_original_pelicans : original_pelican_count h) :
  shark_to_pelican_ratio number_sharks number_pelicans = 2 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_ratio_sharks_to_pelicans_l937_93728


namespace NUMINAMATH_GPT_find_x_range_l937_93781

theorem find_x_range : 
  {x : ℝ | (2 / (x + 2) + 4 / (x + 8) ≤ 3 / 4)} = 
  {x : ℝ | (-4 < x ∧ x ≤ -2) ∨ (4 ≤ x)} := by
  sorry

end NUMINAMATH_GPT_find_x_range_l937_93781


namespace NUMINAMATH_GPT_cleaning_time_with_doubled_an_speed_l937_93758

def A := 1 / 12  -- Anne's cleaning rate (houses per hour)
def B := 1 / 6   -- Bruce's cleaning rate (houses per hour)

theorem cleaning_time_with_doubled_an_speed :
  (A * 2 + B) * 3 = 1 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cleaning_time_with_doubled_an_speed_l937_93758


namespace NUMINAMATH_GPT_percentage_profit_l937_93702

variable (total_crates : ℕ)
variable (total_cost : ℕ)
variable (lost_crates : ℕ)
variable (sell_price_per_crate : ℕ)

theorem percentage_profit (h1 : total_crates = 10) (h2 : total_cost = 160)
  (h3 : lost_crates = 2) (h4 : sell_price_per_crate = 25) :
  (8 * sell_price_per_crate - total_cost) * 100 / total_cost = 25 :=
by
  -- Definitions and steps to prove this can be added here.
  sorry

end NUMINAMATH_GPT_percentage_profit_l937_93702


namespace NUMINAMATH_GPT_mascot_sales_growth_rate_equation_l937_93731

-- Define the conditions
def march_sales : ℝ := 100000
def may_sales : ℝ := 115000
def growth_rate (x : ℝ) : Prop := x > 0

-- Define the equation to be proven
theorem mascot_sales_growth_rate_equation (x : ℝ) (h : growth_rate x) :
    10 * (1 + x) ^ 2 = 11.5 :=
sorry

end NUMINAMATH_GPT_mascot_sales_growth_rate_equation_l937_93731


namespace NUMINAMATH_GPT_relationship_of_f_values_l937_93777

noncomputable def f : ℝ → ℝ := sorry  -- placeholder for the actual function 

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f (-x + 2)

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := a < b → f a < f b

theorem relationship_of_f_values (h1 : is_increasing f 0 2) (h2 : is_even f) :
  f (5/2) > f 1 ∧ f 1 > f (7/2) :=
sorry -- proof goes here

end NUMINAMATH_GPT_relationship_of_f_values_l937_93777


namespace NUMINAMATH_GPT_sum_of_arithmetic_series_l937_93798

def a₁ : ℕ := 9
def d : ℕ := 4
def n : ℕ := 50

noncomputable def nth_term (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d
noncomputable def sum_arithmetic_series (a₁ d n : ℕ) : ℕ := n / 2 * (a₁ + nth_term a₁ d n)

theorem sum_of_arithmetic_series :
  sum_arithmetic_series a₁ d n = 5350 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_series_l937_93798


namespace NUMINAMATH_GPT_solution_set_transformation_l937_93751

noncomputable def solution_set_of_first_inequality (a b : ℝ) : Set ℝ :=
  {x | a * x^2 - 5 * x + b > 0}

noncomputable def solution_set_of_second_inequality (a b : ℝ) : Set ℝ :=
  {x | b * x^2 - 5 * x + a > 0}

theorem solution_set_transformation (a b : ℝ)
  (h : solution_set_of_first_inequality a b = {x | -3 < x ∧ x < 2}) :
  solution_set_of_second_inequality a b = {x | x < -3 ∨ x > 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_transformation_l937_93751


namespace NUMINAMATH_GPT_simple_interest_rate_l937_93784

theorem simple_interest_rate (P : ℝ) (T : ℝ) (A : ℝ) (R : ℝ) (h : A = 3 * P) (h1 : T = 12) (h2 : A - P = (P * R * T) / 100) :
  R = 16.67 :=
by sorry

end NUMINAMATH_GPT_simple_interest_rate_l937_93784


namespace NUMINAMATH_GPT_range_of_a_l937_93743

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 ≤ 0) ↔ (-1 < a ∧ a < 3) := 
sorry

end NUMINAMATH_GPT_range_of_a_l937_93743


namespace NUMINAMATH_GPT_total_cost_of_soup_l937_93724

theorem total_cost_of_soup 
  (pounds_beef : ℕ) (pounds_veg : ℕ) (cost_veg_per_pound : ℕ) (beef_price_multiplier : ℕ)
  (h1 : pounds_beef = 4)
  (h2 : pounds_veg = 6)
  (h3 : cost_veg_per_pound = 2)
  (h4 : beef_price_multiplier = 3):
  (pounds_veg * cost_veg_per_pound + pounds_beef * (cost_veg_per_pound * beef_price_multiplier)) = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_soup_l937_93724


namespace NUMINAMATH_GPT_library_books_count_l937_93738

def students_per_day : List ℕ := [4, 5, 6, 9]
def books_per_student : ℕ := 5
def total_books_given (students : List ℕ) (books_per_student : ℕ) : ℕ :=
  students.foldl (λ acc n => acc + n * books_per_student) 0

theorem library_books_count :
  total_books_given students_per_day books_per_student = 120 :=
by
  sorry

end NUMINAMATH_GPT_library_books_count_l937_93738


namespace NUMINAMATH_GPT_inequality_range_l937_93716

theorem inequality_range (a : ℝ) : (-1 < a ∧ a ≤ 0) → ∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0 :=
by
  intro ha
  sorry

end NUMINAMATH_GPT_inequality_range_l937_93716


namespace NUMINAMATH_GPT_minimum_correct_answers_l937_93791

/-
There are a total of 20 questions. Answering correctly scores 10 points, while answering incorrectly or not answering deducts 5 points. 
To pass, one must score no less than 80 points. Xiao Ming passed the selection. Prove that the minimum number of questions Xiao Ming 
must have answered correctly is no less than 12.
-/

theorem minimum_correct_answers (total_questions correct_points incorrect_points pass_score : ℕ)
  (h1 : total_questions = 20)
  (h2 : correct_points = 10)
  (h3 : incorrect_points = 5)
  (h4 : pass_score = 80)
  (h_passed : ∃ x : ℕ, x ≤ total_questions ∧ (correct_points * x - incorrect_points * (total_questions - x)) ≥ pass_score) :
  ∃ x : ℕ, x ≥ 12 ∧ (correct_points * x - incorrect_points * (total_questions - x)) ≥ pass_score := 
sorry

end NUMINAMATH_GPT_minimum_correct_answers_l937_93791


namespace NUMINAMATH_GPT_point_reflection_y_l937_93783

def coordinates_with_respect_to_y_axis (x y : ℝ) : ℝ × ℝ :=
  (-x, y)

theorem point_reflection_y (x y : ℝ) (h : (x, y) = (-2, 3)) : coordinates_with_respect_to_y_axis x y = (2, 3) := by
  sorry

end NUMINAMATH_GPT_point_reflection_y_l937_93783


namespace NUMINAMATH_GPT_joanne_trip_l937_93733

theorem joanne_trip (a b c x : ℕ) (h1 : 1 ≤ a) (h2 : a + b + c = 9) (h3 : 100 * c + 10 * a + b - (100 * a + 10 * b + c) = 60 * x) : 
  a^2 + b^2 + c^2 = 51 :=
by
  sorry

end NUMINAMATH_GPT_joanne_trip_l937_93733


namespace NUMINAMATH_GPT_cyclist_arrives_first_l937_93727

-- Definitions based on given conditions
def speed_cyclist (v : ℕ) := v
def speed_motorist (v : ℕ) := 5 * v

def distance_total (d : ℕ) := d
def distance_half (d : ℕ) := d / 2

def time_motorist_first_half (d v : ℕ) : ℕ := distance_half d / speed_motorist v

def remaining_distance_cyclist (d v : ℕ) := d - v * time_motorist_first_half d v

def speed_motorist_walking (v : ℕ) := v / 2

def time_motorist_second_half (d v : ℕ) := distance_half d / speed_motorist_walking v
def time_cyclist_remaining (d v : ℕ) : ℕ := remaining_distance_cyclist d v / speed_cyclist v

-- Comparison to prove cyclist arrives first
theorem cyclist_arrives_first (d v : ℕ) (hv : 0 < v) (hd : 0 < d) :
  time_cyclist_remaining d v < time_motorist_second_half d v :=
by sorry

end NUMINAMATH_GPT_cyclist_arrives_first_l937_93727


namespace NUMINAMATH_GPT_scientific_notation_350_million_l937_93711

theorem scientific_notation_350_million : 350000000 = 3.5 * 10^8 := 
  sorry

end NUMINAMATH_GPT_scientific_notation_350_million_l937_93711


namespace NUMINAMATH_GPT_sum_of_squares_of_chords_in_sphere_l937_93799

-- Defining variables
variables (R PO : ℝ)

-- Define the problem statement
theorem sum_of_squares_of_chords_in_sphere
  (chord_lengths_squared : ℝ)
  (H_chord_lengths_squared : chord_lengths_squared = 3 * R^2 - 2 * PO^2) :
  chord_lengths_squared = 3 * R^2 - 2 * PO^2 :=
by
  sorry -- proof is omitted

end NUMINAMATH_GPT_sum_of_squares_of_chords_in_sphere_l937_93799


namespace NUMINAMATH_GPT_original_price_of_article_l937_93756

theorem original_price_of_article 
  (S : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h1 : S = 25)
  (h2 : gain_percent = 1.5)
  (h3 : S = P + P * gain_percent) : 
  P = 10 :=
by 
  sorry

end NUMINAMATH_GPT_original_price_of_article_l937_93756


namespace NUMINAMATH_GPT_monotonicity_condition_l937_93700

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x

theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, f a x ≥ f a 1) ↔ a ∈ Set.Ici 2 :=
by
  sorry

end NUMINAMATH_GPT_monotonicity_condition_l937_93700


namespace NUMINAMATH_GPT_Bettina_card_value_l937_93774

theorem Bettina_card_value (x : ℝ) (h₀ : 0 < x) (h₁ : x < π / 2) (h₂ : Real.tan x ≠ 1) (h₃ : Real.sin x ≠ Real.cos x) :
  ∀ {a b c : ℝ}, (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
                  (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
                  (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
                  a ≠ b → b ≠ c → a ≠ c →
                  (b = Real.cos x) → b = Real.sqrt 3 / 2 := 
  sorry

end NUMINAMATH_GPT_Bettina_card_value_l937_93774


namespace NUMINAMATH_GPT_unique_linear_eq_sol_l937_93788

theorem unique_linear_eq_sol (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (a b c : ℤ), (∀ x y : ℕ, (a * x + b * y = c ↔ x = m ∧ y = n)) :=
by
  sorry

end NUMINAMATH_GPT_unique_linear_eq_sol_l937_93788


namespace NUMINAMATH_GPT_meet_days_l937_93730

-- Definition of conditions
def person_a_days : ℕ := 5
def person_b_days : ℕ := 7
def person_b_early_departure : ℕ := 2

-- Definition of the number of days after A's start that they meet
variable {x : ℕ}

-- Statement to be proven
theorem meet_days (x : ℕ) : (x + 2 : ℚ) / person_b_days + x / person_a_days = 1 := sorry

end NUMINAMATH_GPT_meet_days_l937_93730


namespace NUMINAMATH_GPT_geometric_progression_fourth_term_eq_one_l937_93785

theorem geometric_progression_fourth_term_eq_one :
  let a₁ := (2:ℝ)^(1/4)
  let a₂ := (2:ℝ)^(1/6)
  let a₃ := (2:ℝ)^(1/12)
  let r := a₂ / a₁
  let a₄ := a₃ * r
  a₄ = 1 := by
  sorry

end NUMINAMATH_GPT_geometric_progression_fourth_term_eq_one_l937_93785


namespace NUMINAMATH_GPT_class_size_l937_93725

theorem class_size (n : ℕ) (h₁ : 60 - n > 0) (h₂ : (60 - n) / 2 = n) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_class_size_l937_93725


namespace NUMINAMATH_GPT_tire_usage_l937_93752

theorem tire_usage (total_distance : ℕ) (num_tires : ℕ) (active_tires : ℕ) 
  (h1 : total_distance = 45000) 
  (h2 : num_tires = 5) 
  (h3 : active_tires = 4) 
  (equal_usage : (total_distance * active_tires) / num_tires = 36000) : 
  (∀ tire, tire < num_tires → used_miles_per_tire = 36000) := 
by
  sorry

end NUMINAMATH_GPT_tire_usage_l937_93752


namespace NUMINAMATH_GPT_mod_residue_17_l937_93797

theorem mod_residue_17 : (513 + 3 * 68 + 9 * 289 + 2 * 34 - 10) % 17 = 7 := by
  -- We first compute the modulo 17 residue of each term given in the problem:
  -- 513 == 0 % 17
  -- 68 == 0 % 17
  -- 289 == 0 % 17
  -- 34 == 0 % 17
  -- -10 == 7 % 17
  sorry

end NUMINAMATH_GPT_mod_residue_17_l937_93797


namespace NUMINAMATH_GPT_false_statement_is_D_l937_93705

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

def is_scalene_triangle (a b c : ℝ) : Prop :=
  (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def is_right_isosceles_triangle (a b c : ℝ) : Prop :=
  is_right_triangle a b c ∧ is_isosceles_triangle a b c

-- Statements derived from conditions
def statement_A : Prop := ∀ (a b c : ℝ), is_isosceles_triangle a b c → a = b ∨ b = c ∨ c = a
def statement_B : Prop := ∀ (a b c : ℝ), is_right_triangle a b c → a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2
def statement_C : Prop := ∀ (a b c : ℝ), is_scalene_triangle a b c → a ≠ b ∧ b ≠ c ∧ c ≠ a
def statement_D : Prop := ∀ (a b c : ℝ), is_right_triangle a b c → is_isosceles_triangle a b c
def statement_E : Prop := ∀ (a b c : ℝ), is_right_isosceles_triangle a b c → ∃ (θ : ℝ), θ ≠ 90 ∧ θ = 45

-- Main theorem to be proved
theorem false_statement_is_D : statement_D = false :=
by
  sorry

end NUMINAMATH_GPT_false_statement_is_D_l937_93705


namespace NUMINAMATH_GPT_range_of_a_l937_93709

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x_0 : ℝ, x_0^2 + (a - 1) * x_0 + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l937_93709
