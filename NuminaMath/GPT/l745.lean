import Mathlib

namespace NUMINAMATH_GPT_necessary_condition_for_q_implies_m_in_range_neg_p_or_neg_q_false_implies_x_in_range_l745_74503

-- Proof Problem 1
theorem necessary_condition_for_q_implies_m_in_range (m : ℝ) (h1 : 0 < m) :
  (∀ x : ℝ, 2 - m ≤ x ∧ x ≤ 2 + m → -2 ≤ x ∧ x ≤ 6) →
  0 < m ∧ m ≤ 4 :=
by
  sorry

-- Proof Problem 2
theorem neg_p_or_neg_q_false_implies_x_in_range (m : ℝ) (x : ℝ)
  (h2 : m = 2)
  (h3 : (x + 2) * (x - 6) ≤ 0)
  (h4 : 2 - m ≤ x ∧ x ≤ 2 + m)
  (h5 : ¬ ((x + 2) * (x - 6) > 0 ∨ x < 2 - m ∨ x > 2 + m)) :
  0 ≤ x ∧ x ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_necessary_condition_for_q_implies_m_in_range_neg_p_or_neg_q_false_implies_x_in_range_l745_74503


namespace NUMINAMATH_GPT_quadratic_equation_with_given_root_l745_74562

theorem quadratic_equation_with_given_root : 
  ∃ p q : ℤ, (∀ x : ℝ, x^2 + (p : ℝ) * x + (q : ℝ) = 0 ↔ x = 2 - Real.sqrt 7 ∨ x = 2 + Real.sqrt 7) 
  ∧ (p = -4) ∧ (q = -3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_with_given_root_l745_74562


namespace NUMINAMATH_GPT_bob_investment_correct_l745_74564

noncomputable def initial_investment_fundA : ℝ := 2000
noncomputable def interest_rate_fundA : ℝ := 0.12
noncomputable def initial_investment_fundB : ℝ := 1000
noncomputable def interest_rate_fundB : ℝ := 0.30
noncomputable def fundA_after_two_years := initial_investment_fundA * (1 + interest_rate_fundA)
noncomputable def fundB_after_two_years (B : ℝ) := B * (1 + interest_rate_fundB)^2
noncomputable def extra_value : ℝ := 549.9999999999998

theorem bob_investment_correct :
  fundA_after_two_years = fundB_after_two_years initial_investment_fundB + extra_value :=
by
  sorry

end NUMINAMATH_GPT_bob_investment_correct_l745_74564


namespace NUMINAMATH_GPT_supplementary_angle_l745_74560

theorem supplementary_angle (θ : ℝ) (k : ℤ) : (θ = 10) → (∃ k, θ + 250 = k * 360 + 360) :=
by
  sorry

end NUMINAMATH_GPT_supplementary_angle_l745_74560


namespace NUMINAMATH_GPT_prism_faces_l745_74572

theorem prism_faces (E V F n : ℕ) (h1 : E + V = 30) (h2 : F + V = E + 2) (h3 : E = 3 * n) : F = 8 :=
by
  -- Actual proof omitted
  sorry

end NUMINAMATH_GPT_prism_faces_l745_74572


namespace NUMINAMATH_GPT_sum_of_squares_of_rates_l745_74584

variable (b j s : ℕ)

theorem sum_of_squares_of_rates
  (h1 : 3 * b + 2 * j + 3 * s = 82)
  (h2 : 5 * b + 3 * j + 2 * s = 99) :
  b^2 + j^2 + s^2 = 314 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_rates_l745_74584


namespace NUMINAMATH_GPT_compare_log_exp_l745_74578

theorem compare_log_exp (x y z : ℝ) 
  (hx : x = Real.log 2 / Real.log 5) 
  (hy : y = Real.log 2) 
  (hz : z = Real.sqrt 2) : 
  x < y ∧ y < z := 
sorry

end NUMINAMATH_GPT_compare_log_exp_l745_74578


namespace NUMINAMATH_GPT_R_l745_74512

variable (a d n : ℕ)

def arith_sum (k : ℕ) : ℕ :=
  k * (a + (k - 1) * d / 2)

def s1 := arith_sum n
def s2 := arith_sum (3 * n)
def s3 := arith_sum (5 * n)
def s4 := arith_sum (7 * n)

def R' := s4 - s3 - s2

theorem R'_depends_on_d_n : 
  R' = 2 * d * n^2 := 
by 
  sorry

end NUMINAMATH_GPT_R_l745_74512


namespace NUMINAMATH_GPT_simplify_expression_l745_74538

variable (i : ℂ)

-- Define the conditions

def i_squared_eq_neg_one : Prop := i^2 = -1
def i_cubed_eq_neg_i : Prop := i^3 = i * i^2 ∧ i^3 = -i
def i_fourth_eq_one : Prop := i^4 = (i^2)^2 ∧ i^4 = 1
def i_fifth_eq_i : Prop := i^5 = i * i^4 ∧ i^5 = i

-- Define the proof problem

theorem simplify_expression (h1 : i_squared_eq_neg_one i) (h2 : i_cubed_eq_neg_i i) (h3 : i_fourth_eq_one i) (h4 : i_fifth_eq_i i) : 
  i + i^2 + i^3 + i^4 + i^5 = i := 
  by sorry

end NUMINAMATH_GPT_simplify_expression_l745_74538


namespace NUMINAMATH_GPT_cannot_form_right_triangle_l745_74585

theorem cannot_form_right_triangle (a b c : ℝ) (h₁ : a = 2) (h₂ : b = 2) (h₃ : c = 3) :
  a^2 + b^2 ≠ c^2 :=
by
  rw [h₁, h₂, h₃]
  -- Next step would be to simplify and show the inequality, but we skip the proof
  -- 2^2 + 2^2 = 4 + 4 = 8 
  -- 3^2 = 9 
  -- 8 ≠ 9
  sorry

end NUMINAMATH_GPT_cannot_form_right_triangle_l745_74585


namespace NUMINAMATH_GPT_largest_lcm_l745_74588

theorem largest_lcm :
  ∀ (a b c d e f : ℕ),
  a = Nat.lcm 18 2 →
  b = Nat.lcm 18 4 →
  c = Nat.lcm 18 6 →
  d = Nat.lcm 18 9 →
  e = Nat.lcm 18 12 →
  f = Nat.lcm 18 16 →
  max (max (max (max (max a b) c) d) e) f = 144 :=
by
  intros a b c d e f ha hb hc hd he hf
  sorry

end NUMINAMATH_GPT_largest_lcm_l745_74588


namespace NUMINAMATH_GPT_find_b_and_sinA_find_sin_2A_plus_pi_over_4_l745_74566

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sinB : ℝ)

-- Conditions
def triangle_conditions :=
  (a > b) ∧
  (a = 5) ∧
  (c = 6) ∧
  (sinB = 3 / 5)

-- Question 1: Prove b = sqrt 13 and sin A = (3 * sqrt 13) / 13
theorem find_b_and_sinA (h : triangle_conditions a b c sinB) :
  b = Real.sqrt 13 ∧
  ∃ sinA : ℝ, sinA = (3 * Real.sqrt 13) / 13 :=
  sorry

-- Question 2: Prove sin (2A + π/4) = 7 * sqrt 2 / 26
theorem find_sin_2A_plus_pi_over_4 (h : triangle_conditions a b c sinB)
  (hb : b = Real.sqrt 13)
  (sinA : ℝ)
  (h_sinA : sinA = (3 * Real.sqrt 13) / 13) :
  ∃ sin2Aπ4 : ℝ, sin2Aπ4 = (7 * Real.sqrt 2) / 26 :=
  sorry

end NUMINAMATH_GPT_find_b_and_sinA_find_sin_2A_plus_pi_over_4_l745_74566


namespace NUMINAMATH_GPT_josh_initial_money_l745_74523

/--
Josh spent $1.75 on a drink, and then spent another $1.25, and has $6.00 left. 
Prove that initially Josh had $9.00.
-/
theorem josh_initial_money : 
  ∃ (initial : ℝ), (initial - 1.75 - 1.25 = 6) ∧ initial = 9 := 
sorry

end NUMINAMATH_GPT_josh_initial_money_l745_74523


namespace NUMINAMATH_GPT_sophie_total_spend_l745_74590

-- Definitions based on conditions
def cost_cupcakes : ℕ := 5 * 2
def cost_doughnuts : ℕ := 6 * 1
def cost_apple_pie : ℕ := 4 * 2
def cost_cookies : ℕ := 15 * 6 / 10 -- since 0.60 = 6/10

-- Total cost
def total_cost : ℕ := cost_cupcakes + cost_doughnuts + cost_apple_pie + cost_cookies

-- Prove the total cost
theorem sophie_total_spend : total_cost = 33 := by
  sorry

end NUMINAMATH_GPT_sophie_total_spend_l745_74590


namespace NUMINAMATH_GPT_Barbara_spent_46_22_on_different_goods_l745_74587

theorem Barbara_spent_46_22_on_different_goods :
  let tuna_cost := (5 * 2) -- Total cost of tuna
  let water_cost := (4 * 1.5) -- Total cost of water
  let total_before_discount := 56 / 0.9 -- Total before discount, derived from the final amount paid after discount
  let total_tuna_water_cost := 10 + 6 -- Total cost of tuna and water together
  let different_goods_cost := total_before_discount - total_tuna_water_cost
  different_goods_cost = 46.22 := 
sorry

end NUMINAMATH_GPT_Barbara_spent_46_22_on_different_goods_l745_74587


namespace NUMINAMATH_GPT_regular_polygon_sides_and_exterior_angle_l745_74550

theorem regular_polygon_sides_and_exterior_angle (n : ℕ) (exterior_sum : ℝ) :
  (180 * (n - 2) = 360 + exterior_sum) → (exterior_sum = 360) → n = 6 ∧ (360 / n = 60) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_and_exterior_angle_l745_74550


namespace NUMINAMATH_GPT_black_pens_count_l745_74514

variable (T B : ℕ)
variable (h1 : (3/10:ℚ) * T = 12)
variable (h2 : (1/5:ℚ) * T = B)

theorem black_pens_count (h1 : (3/10:ℚ) * T = 12) (h2 : (1/5:ℚ) * T = B) : B = 8 := by
  sorry

end NUMINAMATH_GPT_black_pens_count_l745_74514


namespace NUMINAMATH_GPT_at_least_one_woman_selected_l745_74583

noncomputable def probability_at_least_one_woman_selected (men women : ℕ) (total_selected : ℕ) : ℚ :=
  let total_people := men + women
  let prob_no_woman := (men / total_people) * ((men - 1) / (total_people - 1)) * ((men - 2) / (total_people - 2))
  1 - prob_no_woman

theorem at_least_one_woman_selected (men women : ℕ) (total_selected : ℕ) :
  men = 5 → women = 5 → total_selected = 3 → 
  probability_at_least_one_woman_selected men women total_selected = 11 / 12 := by
  intros hmen hwomen hselected
  rw [hmen, hwomen, hselected]
  unfold probability_at_least_one_woman_selected
  sorry

end NUMINAMATH_GPT_at_least_one_woman_selected_l745_74583


namespace NUMINAMATH_GPT_taxi_ride_distance_l745_74549

variable (t : ℝ) (c₀ : ℝ) (cᵢ : ℝ)

theorem taxi_ride_distance (h_t : t = 18.6) (h_c₀ : c₀ = 3.0) (h_cᵢ : cᵢ = 0.4) : 
  ∃ d : ℝ, d = 8 := 
by 
  sorry

end NUMINAMATH_GPT_taxi_ride_distance_l745_74549


namespace NUMINAMATH_GPT_find_abcde_l745_74507

theorem find_abcde (N : ℕ) (a b c d e f : ℕ) (h : a ≠ 0) 
(h1 : N % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f)
(h2 : (N^2) % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) :
    a * 10000 + b * 1000 + c * 100 + d * 10 + e = 48437 :=
by sorry

end NUMINAMATH_GPT_find_abcde_l745_74507


namespace NUMINAMATH_GPT_number_of_purple_balls_l745_74521

theorem number_of_purple_balls (k : ℕ) (h : k > 0) (E : (24 - k) / (8 + k) = 1) : k = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_purple_balls_l745_74521


namespace NUMINAMATH_GPT_molecular_weight_compound_l745_74535

theorem molecular_weight_compound :
  let weight_H := 1.008
  let weight_Cr := 51.996
  let weight_O := 15.999
  let n_H := 2
  let n_Cr := 1
  let n_O := 4
  (n_H * weight_H) + (n_Cr * weight_Cr) + (n_O * weight_O) = 118.008 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_compound_l745_74535


namespace NUMINAMATH_GPT_inequality_transformation_range_of_a_l745_74536

-- Define the given function f(x) = |x + 2|
def f (x : ℝ) : ℝ := abs (x + 2)

-- State the inequality transformation problem
theorem inequality_transformation (x : ℝ) :  (2 * abs (x + 2) < 4 - abs (x - 1)) ↔ (-7 / 3 < x ∧ x < -1) :=
by sorry

-- State the implication problem involving m, n, and a
theorem range_of_a (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (hmn : m + n = 1) (a : ℝ) :
  (∀ x : ℝ, abs (x - a) - f x ≤ 1 / m + 1 / n) → (-6 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_GPT_inequality_transformation_range_of_a_l745_74536


namespace NUMINAMATH_GPT_quotient_larger_than_dividend_l745_74528

-- Define the problem conditions
variables {a b : ℝ}

-- State the theorem corresponding to the problem
theorem quotient_larger_than_dividend (h : b ≠ 0) : ¬ (∀ a : ℝ, ∀ b : ℝ, (a / b > a) ) :=
by
  sorry

end NUMINAMATH_GPT_quotient_larger_than_dividend_l745_74528


namespace NUMINAMATH_GPT_selected_room_l745_74537

theorem selected_room (room_count interval selected initial_room : ℕ) 
  (h_init : initial_room = 5)
  (h_interval : interval = 8)
  (h_room_count : room_count = 64) : 
  ∃ (nth_room : ℕ), nth_room = initial_room + interval * 6 ∧ nth_room = 53 :=
by
  sorry

end NUMINAMATH_GPT_selected_room_l745_74537


namespace NUMINAMATH_GPT_not_equal_77_l745_74539

theorem not_equal_77 (x y : ℤ) : x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end NUMINAMATH_GPT_not_equal_77_l745_74539


namespace NUMINAMATH_GPT_max_interval_length_l745_74586

def m (x : ℝ) : ℝ := x^2 - 3 * x + 4
def n (x : ℝ) : ℝ := 2 * x - 3

def are_close_functions (m n : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |m x - n x| ≤ 1

theorem max_interval_length
  (h : are_close_functions m n 2 3) :
  3 - 2 = 1 :=
sorry

end NUMINAMATH_GPT_max_interval_length_l745_74586


namespace NUMINAMATH_GPT_CH4_reaction_with_Cl2_l745_74594

def balanced_chemical_equation (CH4 Cl2 CH3Cl HCl : ℕ) : Prop :=
  CH4 + Cl2 = CH3Cl + HCl

theorem CH4_reaction_with_Cl2
  (CH4 Cl2 CH3Cl HCl : ℕ)
  (balanced_eq : balanced_chemical_equation 1 1 1 1)
  (reaction_cl2 : Cl2 = 2) :
  CH4 = 2 :=
by
  sorry

end NUMINAMATH_GPT_CH4_reaction_with_Cl2_l745_74594


namespace NUMINAMATH_GPT_find_angle_C_l745_74515

-- Definitions based on conditions
variables (α β γ : ℝ) -- Angles of the triangle

-- Condition: Angles between the altitude and the angle bisector at vertices A and B are equal
-- This implies α = β
def angles_equal (α β : ℝ) : Prop :=
  α = β

-- Condition: Sum of the angles in a triangle is 180 degrees
def angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Condition: Angle at vertex C is greater than angles at vertices A and B
def c_greater_than_a_and_b (α γ : ℝ) : Prop :=
  γ > α

-- The proof problem: Prove γ = 120 degrees given the conditions
theorem find_angle_C (α β γ : ℝ) (h1 : angles_equal α β) (h2 : angles_sum_to_180 α β γ) (h3 : c_greater_than_a_and_b α γ) : γ = 120 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_C_l745_74515


namespace NUMINAMATH_GPT_gcd_7920_14553_l745_74541

theorem gcd_7920_14553 : Int.gcd 7920 14553 = 11 := by
  sorry

end NUMINAMATH_GPT_gcd_7920_14553_l745_74541


namespace NUMINAMATH_GPT_relationship_between_p_and_q_l745_74573

variable {a b : ℝ}

theorem relationship_between_p_and_q 
  (h_a : a > 2) 
  (h_p : p = a + 1 / (a - 2)) 
  (h_q : q = -b^2 - 2 * b + 3) : 
  p ≥ q := 
sorry

end NUMINAMATH_GPT_relationship_between_p_and_q_l745_74573


namespace NUMINAMATH_GPT_cos_sum_equals_fraction_sqrt_13_minus_1_div_4_l745_74580

noncomputable def cos_sum : ℝ :=
  (Real.cos (2 * Real.pi / 17) +
   Real.cos (6 * Real.pi / 17) +
   Real.cos (8 * Real.pi / 17))

theorem cos_sum_equals_fraction_sqrt_13_minus_1_div_4 :
  cos_sum = (Real.sqrt 13 - 1) / 4 := 
sorry

end NUMINAMATH_GPT_cos_sum_equals_fraction_sqrt_13_minus_1_div_4_l745_74580


namespace NUMINAMATH_GPT_trip_cost_is_correct_l745_74529

-- Given conditions
def bills_cost : ℕ := 3500
def save_per_month : ℕ := 500
def savings_duration_months : ℕ := 2 * 12
def savings : ℕ := save_per_month * savings_duration_months
def remaining_after_bills : ℕ := 8500

-- Prove that the cost of the trip to Paris is 3500 dollars
theorem trip_cost_is_correct : (savings - remaining_after_bills) = bills_cost :=
sorry

end NUMINAMATH_GPT_trip_cost_is_correct_l745_74529


namespace NUMINAMATH_GPT_find_ab_cd_l745_74502

variables (a b c d : ℝ)

def special_eq (x : ℝ) := 
  (min (20 * x + 19) (19 * x + 20) = (a * x + b) - abs (c * x + d))

theorem find_ab_cd (h : ∀ x : ℝ, special_eq a b c d x) :
  a * b + c * d = 380 := 
sorry

end NUMINAMATH_GPT_find_ab_cd_l745_74502


namespace NUMINAMATH_GPT_tetrahedron_volume_l745_74547

noncomputable def volume_of_tetrahedron (AB : ℝ) (area_ABC : ℝ) (area_ABD : ℝ) (angle_ABC_ABD : ℝ) : ℝ :=
  (1/3) * area_ABC * area_ABD * (Real.sin angle_ABC_ABD) * (AB / (Real.sqrt 2))

theorem tetrahedron_volume :
  let AB := 5 -- edge AB length in cm
  let area_ABC := 18 -- area of face ABC in cm^2
  let area_ABD := 24 -- area of face ABD in cm^2
  let angle_ABC_ABD := Real.pi / 4 -- 45 degrees in radians
  volume_of_tetrahedron AB area_ABC area_ABD angle_ABC_ABD = 43.2 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_l745_74547


namespace NUMINAMATH_GPT_greatest_integer_gcd_l745_74591

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end NUMINAMATH_GPT_greatest_integer_gcd_l745_74591


namespace NUMINAMATH_GPT_num_frisbees_more_than_deck_cards_l745_74530

variables (M F D x : ℕ)
variable (bought_fraction : ℝ)

theorem num_frisbees_more_than_deck_cards :
  M = 60 ∧ M = 2 * F ∧ F = D + x ∧
  M + bought_fraction * M + F + bought_fraction * F + D + bought_fraction * D = 140 ∧ bought_fraction = 2/5 →
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_num_frisbees_more_than_deck_cards_l745_74530


namespace NUMINAMATH_GPT_bridge_length_l745_74551

theorem bridge_length (length_train : ℝ) (speed_train : ℝ) (time : ℝ) (h1 : length_train = 15) (h2 : speed_train = 275) (h3 : time = 48) : 
    (speed_train / 100) * time - length_train = 117 := 
by
    -- these are the provided conditions, enabling us to skip actual proof steps with 'sorry'
    sorry

end NUMINAMATH_GPT_bridge_length_l745_74551


namespace NUMINAMATH_GPT_seeds_germination_l745_74592

theorem seeds_germination (seed_plot1 seed_plot2 : ℕ) (germ_rate2 total_germ_rate : ℝ) (germinated_total_pct : ℝ)
  (h1 : seed_plot1 = 300)
  (h2 : seed_plot2 = 200)
  (h3 : germ_rate2 = 0.35)
  (h4 : germinated_total_pct = 28.999999999999996 / 100) :
  (germinated_total_pct * (seed_plot1 + seed_plot2) - germ_rate2 * seed_plot2) / seed_plot1 * 100 = 25 :=
by sorry  -- Proof not required

end NUMINAMATH_GPT_seeds_germination_l745_74592


namespace NUMINAMATH_GPT_faye_total_crayons_l745_74574

-- Define the number of rows and the number of crayons per row as given conditions.
def num_rows : ℕ := 7
def crayons_per_row : ℕ := 30

-- State the theorem we need to prove.
theorem faye_total_crayons : (num_rows * crayons_per_row) = 210 :=
by
  sorry

end NUMINAMATH_GPT_faye_total_crayons_l745_74574


namespace NUMINAMATH_GPT_potion_kit_cost_is_18_l745_74555

def price_spellbook : ℕ := 5
def count_spellbooks : ℕ := 5
def price_owl : ℕ := 28
def count_potion_kits : ℕ := 3
def payment_total_silver : ℕ := 537
def silver_per_gold : ℕ := 9

def cost_each_potion_kit_in_silver (payment_total_silver : ℕ)
                                   (price_spellbook : ℕ)
                                   (count_spellbooks : ℕ)
                                   (price_owl : ℕ)
                                   (count_potion_kits : ℕ)
                                   (silver_per_gold : ℕ) : ℕ :=
  let total_gold := payment_total_silver / silver_per_gold
  let cost_spellbooks := count_spellbooks * price_spellbook
  let cost_remaining_gold := total_gold - cost_spellbooks - price_owl
  let cost_each_potion_kit_gold := cost_remaining_gold / count_potion_kits
  cost_each_potion_kit_gold * silver_per_gold

theorem potion_kit_cost_is_18 :
  cost_each_potion_kit_in_silver payment_total_silver
                                 price_spellbook
                                 count_spellbooks
                                 price_owl
                                 count_potion_kits
                                 silver_per_gold = 18 :=
by sorry

end NUMINAMATH_GPT_potion_kit_cost_is_18_l745_74555


namespace NUMINAMATH_GPT_tens_digit_of_13_pow_2021_l745_74577

theorem tens_digit_of_13_pow_2021 :
  let p := 2021
  let base := 13
  let mod_val := 100
  let digit := (base^p % mod_val) / 10
  digit = 1 := by
  sorry

end NUMINAMATH_GPT_tens_digit_of_13_pow_2021_l745_74577


namespace NUMINAMATH_GPT_prove_trig_values_l745_74575

/-- Given angles A and B, where both are acute angles,
  and their sine values are known,
  we aim to prove the cosine of (A + B) and the measure
  of angle C in triangle ABC. -/
theorem prove_trig_values (A B : ℝ)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2)
  (sin_A_eq : Real.sin A = (Real.sqrt 5) / 5)
  (sin_B_eq : Real.sin B = (Real.sqrt 10) / 10) :
  Real.cos (A + B) = (Real.sqrt 2) / 2 ∧ (π - (A + B)) = 3 * π / 4 := by
sorry

end NUMINAMATH_GPT_prove_trig_values_l745_74575


namespace NUMINAMATH_GPT_minimum_balls_to_draw_l745_74571

-- Defining the sizes for the different colors of balls
def red_balls : Nat := 40
def green_balls : Nat := 25
def yellow_balls : Nat := 20
def blue_balls : Nat := 15
def purple_balls : Nat := 10
def orange_balls : Nat := 5

-- Given conditions
def max_red_balls_before_18 : Nat := 17
def max_green_balls_before_18 : Nat := 17
def max_yellow_balls_before_18 : Nat := 17
def max_blue_balls_before_18 : Nat := 15
def max_purple_balls_before_18 : Nat := 10
def max_orange_balls_before_18 : Nat := 5

-- Sum of maximum balls of each color that can be drawn without ensuring 18 of any color
def max_balls_without_18 : Nat := 
  max_red_balls_before_18 + 
  max_green_balls_before_18 + 
  max_yellow_balls_before_18 + 
  max_blue_balls_before_18 + 
  max_purple_balls_before_18 + 
  max_orange_balls_before_18

theorem minimum_balls_to_draw {n : Nat} (h : n = max_balls_without_18 + 1) :
  n = 82 := by
  sorry

end NUMINAMATH_GPT_minimum_balls_to_draw_l745_74571


namespace NUMINAMATH_GPT_speed_of_stream_l745_74513

theorem speed_of_stream (v : ℝ) 
    (h1 : ∀ (v : ℝ), v ≠ 0 → (80 / (36 + v) = 40 / (36 - v))) : 
    v = 12 := 
by 
    sorry

end NUMINAMATH_GPT_speed_of_stream_l745_74513


namespace NUMINAMATH_GPT_non_consecutive_heads_probability_l745_74524

-- Define the total number of basic events (n).
def total_events : ℕ := 2^4

-- Define the number of events where heads do not appear consecutively (m).
def non_consecutive_heads_events : ℕ := 1 + (Nat.choose 4 1) + (Nat.choose 3 2)

-- Define the probability of heads not appearing consecutively.
def probability_non_consecutive_heads : ℚ := non_consecutive_heads_events / total_events

-- The theorem we seek to prove
theorem non_consecutive_heads_probability :
  probability_non_consecutive_heads = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_non_consecutive_heads_probability_l745_74524


namespace NUMINAMATH_GPT_parametric_to_standard_l745_74557

theorem parametric_to_standard (t a b x y : ℝ)
(h1 : x = (a / 2) * (t + 1 / t))
(h2 : y = (b / 2) * (t - 1 / t)) :
  (x^2 / a^2) - (y^2 / b^2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_parametric_to_standard_l745_74557


namespace NUMINAMATH_GPT_toothpicks_grid_total_l745_74519

theorem toothpicks_grid_total (L W : ℕ) (hL : L = 60) (hW : W = 32) : 
  (L + 1) * W + (W + 1) * L = 3932 := 
by 
  sorry

end NUMINAMATH_GPT_toothpicks_grid_total_l745_74519


namespace NUMINAMATH_GPT_quadratic_has_real_roots_iff_l745_74565

theorem quadratic_has_real_roots_iff (m : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + m + 5 = 0) ↔ m ≤ -1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_iff_l745_74565


namespace NUMINAMATH_GPT_find_date_behind_l745_74532

variables (x y : ℕ)
-- Conditions
def date_behind_C := x
def date_behind_A := x + 1
def date_behind_B := x + 13
def date_behind_P := x + 14

-- Statement to prove
theorem find_date_behind : (x + y = (x + 1) + (x + 13)) → (y = date_behind_P) :=
by
  sorry

end NUMINAMATH_GPT_find_date_behind_l745_74532


namespace NUMINAMATH_GPT_find_multiple_of_savings_l745_74542

variable (A K m : ℝ)

-- Conditions
def condition1 : Prop := A - 150 = (1 / 3) * K
def condition2 : Prop := A + K = 750

-- Question
def question : Prop := m * K = 3 * A

-- Proof Problem Statement
theorem find_multiple_of_savings (h1 : condition1 A K) (h2 : condition2 A K) : 
  question A K 2 :=
sorry

end NUMINAMATH_GPT_find_multiple_of_savings_l745_74542


namespace NUMINAMATH_GPT_floor_ceil_difference_l745_74534

theorem floor_ceil_difference : 
  let a := (18 / 5) * (-33 / 4)
  let b := ⌈(-33 / 4 : ℝ)⌉
  let c := (18 / 5) * (b : ℝ)
  let d := ⌈c⌉
  ⌊a⌋ - d = -2 :=
by
  sorry

end NUMINAMATH_GPT_floor_ceil_difference_l745_74534


namespace NUMINAMATH_GPT_total_amount_paid_l745_74589

theorem total_amount_paid (cost_lunch : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) (sales_tax : ℝ) (tip : ℝ) 
  (h1 : cost_lunch = 100) 
  (h2 : sales_tax_rate = 0.04) 
  (h3 : tip_rate = 0.06) 
  (h4 : sales_tax = cost_lunch * sales_tax_rate) 
  (h5 : tip = cost_lunch * tip_rate) :
  cost_lunch + sales_tax + tip = 110 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l745_74589


namespace NUMINAMATH_GPT_percent_of_x_is_y_in_terms_of_z_l745_74546

theorem percent_of_x_is_y_in_terms_of_z (x y z : ℝ) (h1 : 0.7 * (x - y) = 0.3 * (x + y))
    (h2 : 0.6 * (x + z) = 0.4 * (y - z)) : y / x = 0.4 :=
  sorry

end NUMINAMATH_GPT_percent_of_x_is_y_in_terms_of_z_l745_74546


namespace NUMINAMATH_GPT_arcsin_one_half_eq_pi_six_l745_74518

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_one_half_eq_pi_six_l745_74518


namespace NUMINAMATH_GPT_average_age_inhabitants_Campo_Verde_l745_74545

theorem average_age_inhabitants_Campo_Verde
  (H M : ℕ)
  (ratio_h_m : H / M = 2 / 3)
  (avg_age_men : ℕ := 37)
  (avg_age_women : ℕ := 42) :
  ((37 * H + 42 * M) / (H + M) : ℕ) = 40 := 
sorry

end NUMINAMATH_GPT_average_age_inhabitants_Campo_Verde_l745_74545


namespace NUMINAMATH_GPT_solution_l745_74527

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ),
    x - y = 1 ∧
    x^3 - y^3 = 2 ∧
    x^4 + y^4 = 23 / 9 ∧
    x^5 - y^5 = 29 / 9

theorem solution : problem_statement := sorry

end NUMINAMATH_GPT_solution_l745_74527


namespace NUMINAMATH_GPT_quadratic_root_value_k_l745_74582

theorem quadratic_root_value_k (k : ℝ) :
  (
    ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -4 / 3 ∧
    (∀ x : ℝ, x^2 * k - 8 * x - 18 = 0 ↔ (x = x₁ ∨ x = x₂))
  ) → k = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_value_k_l745_74582


namespace NUMINAMATH_GPT_smallest_angle_in_right_triangle_l745_74540

-- Given conditions
def angle_α := 90 -- The right-angle in degrees
def angle_β := 55 -- The given angle in degrees

-- Goal: Prove that the smallest angle is 35 degrees.
theorem smallest_angle_in_right_triangle (a b c : ℕ) (h1 : a = angle_α) (h2 : b = angle_β) (h3 : c = 180 - a - b) : c = 35 := 
by {
  -- use sorry to skip the proof steps
  sorry
}

end NUMINAMATH_GPT_smallest_angle_in_right_triangle_l745_74540


namespace NUMINAMATH_GPT_player_A_prize_received_event_A_not_low_probability_l745_74505

-- Condition Definitions
def k : ℕ := 4
def m : ℕ := 2
def n : ℕ := 1
def p : ℚ := 2 / 3
def a : ℚ := 243

-- Part 1: Player A's Prize
theorem player_A_prize_received :
  (a * (p * p + 3 * p * (1 - p) * p + 3 * (1 - p) * p * p + (1 - p) * (1 - p) * p * p)) = 216 := sorry

-- Part 2: Probability of Event A with Low Probability Conditions
def low_probability_event (prob : ℚ) : Prop := prob < 0.05

-- Probability that player B wins the entire prize
def event_A_probability (p : ℚ) : ℚ :=
  (1 - p) ^ 3 + 3 * p * (1 - p) ^ 3

theorem event_A_not_low_probability (p : ℚ) (hp : p ≥ 3 / 4) :
  ¬ low_probability_event (event_A_probability p) := sorry

end NUMINAMATH_GPT_player_A_prize_received_event_A_not_low_probability_l745_74505


namespace NUMINAMATH_GPT_planes_meet_in_50_minutes_l745_74544

noncomputable def time_to_meet (d : ℕ) (vA vB : ℕ) : ℚ :=
  d / (vA + vB : ℚ)

theorem planes_meet_in_50_minutes
  (d : ℕ) (vA vB : ℕ)
  (h_d : d = 500) (h_vA : vA = 240) (h_vB : vB = 360) :
  (time_to_meet d vA vB * 60 : ℚ) = 50 := by
  sorry

end NUMINAMATH_GPT_planes_meet_in_50_minutes_l745_74544


namespace NUMINAMATH_GPT_young_li_age_l745_74506

theorem young_li_age (x : ℝ) (old_li_age : ℝ) 
  (h1 : old_li_age = 2.5 * x)  
  (h2 : old_li_age + 10 = 2 * (x + 10)) : 
  x = 20 := 
by
  sorry

end NUMINAMATH_GPT_young_li_age_l745_74506


namespace NUMINAMATH_GPT_solve_for_x_l745_74500

theorem solve_for_x (x : ℝ) (h : 5 / (4 + 1 / x) = 1) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l745_74500


namespace NUMINAMATH_GPT_called_back_students_l745_74510

/-- Given the number of girls, boys, and students who didn't make the cut,
    this theorem proves the number of students who got called back. -/
theorem called_back_students (girls boys didnt_make_the_cut : ℕ)
    (h_girls : girls = 39)
    (h_boys : boys = 4)
    (h_didnt_make_the_cut : didnt_make_the_cut = 17) :
    girls + boys - didnt_make_the_cut = 26 := by
  sorry

end NUMINAMATH_GPT_called_back_students_l745_74510


namespace NUMINAMATH_GPT_downstream_speed_l745_74570

variable (Vu Vs Vd Vc : ℝ)

theorem downstream_speed
  (h1 : Vu = 25)
  (h2 : Vs = 32)
  (h3 : Vu = Vs - Vc)
  (h4 : Vd = Vs + Vc) :
  Vd = 39 := by
  sorry

end NUMINAMATH_GPT_downstream_speed_l745_74570


namespace NUMINAMATH_GPT_exists_b_mod_5_l745_74558

theorem exists_b_mod_5 (p q r s : ℤ) (h1 : ¬ (s % 5 = 0)) (a : ℤ) (h2 : (p * a^3 + q * a^2 + r * a + s) % 5 = 0) : 
  ∃ b : ℤ, (s * b^3 + r * b^2 + q * b + p) % 5 = 0 :=
sorry

end NUMINAMATH_GPT_exists_b_mod_5_l745_74558


namespace NUMINAMATH_GPT_find_intersection_l745_74559

variable (A : Set ℝ)
variable (B : Set ℝ := {1, 2})
variable (f : ℝ → ℝ := λ x => x^2)

theorem find_intersection (h : ∀ x, x ∈ A → f x ∈ B) : A ∩ B = ∅ ∨ A ∩ B = {1} :=
by
  sorry

end NUMINAMATH_GPT_find_intersection_l745_74559


namespace NUMINAMATH_GPT_find_f1_verify_function_l745_74568

theorem find_f1 (f : ℝ → ℝ) (h_mono : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2)
    (h1_pos : ∀ x : ℝ, 0 < x → f x > 1 / x^2)
    (h_eq : ∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3) :
    f 1 = 2 := sorry

theorem verify_function (f : ℝ → ℝ) (h_def : ∀ x : ℝ, 0 < x → f x = 2 / x^2) :
    (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2) ∧ (∀ x : ℝ, 0 < x → f x > 1 / x^2) ∧
    (∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3) := sorry

end NUMINAMATH_GPT_find_f1_verify_function_l745_74568


namespace NUMINAMATH_GPT_greatest_drop_in_price_is_May_l745_74520

def priceChangeJan := -1.25
def priceChangeFeb := 2.75
def priceChangeMar := -0.75
def priceChangeApr := 1.50
def priceChangeMay := -3.00
def priceChangeJun := -1.00

theorem greatest_drop_in_price_is_May :
  priceChangeMay < priceChangeJan ∧
  priceChangeMay < priceChangeMar ∧
  priceChangeMay < priceChangeApr ∧
  priceChangeMay < priceChangeJun ∧
  priceChangeMay < priceChangeFeb :=
by sorry

end NUMINAMATH_GPT_greatest_drop_in_price_is_May_l745_74520


namespace NUMINAMATH_GPT_time_to_hit_ground_l745_74552

theorem time_to_hit_ground : ∃ t : ℝ, 
  (y = -4.9 * t^2 + 7.2 * t + 8) → (y - (-0.6 * t) * t = 0) → t = 223/110 :=
by
  sorry

end NUMINAMATH_GPT_time_to_hit_ground_l745_74552


namespace NUMINAMATH_GPT_math_problem_l745_74567

noncomputable def problem_statement : Prop := (7^2 - 5^2)^4 = 331776

theorem math_problem : problem_statement := by
  sorry

end NUMINAMATH_GPT_math_problem_l745_74567


namespace NUMINAMATH_GPT_b3_b8_product_l745_74596

-- Definitions based on conditions
def is_arithmetic_seq (b : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

-- The problem statement
theorem b3_b8_product (b : ℕ → ℤ) (h_seq : is_arithmetic_seq b) (h4_7 : b 4 * b 7 = 24) : 
  b 3 * b 8 = 200 / 9 :=
sorry

end NUMINAMATH_GPT_b3_b8_product_l745_74596


namespace NUMINAMATH_GPT_distinct_left_views_l745_74522

/-- Consider 10 small cubes each having dimension 1 cm × 1 cm × 1 cm.
    Each pair of adjacent cubes shares at least one edge (1 cm) or one face (1 cm × 1 cm).
    The cubes must not be suspended in the air and each cube's edges should be either
    perpendicular or parallel to the horizontal lines. Prove that the number of distinct
    left views of any arrangement of these 10 cubes is 16. -/
theorem distinct_left_views (cube_count : ℕ) (dimensions : ℝ) 
  (shared_edge : (ℝ × ℝ) → Prop) (no_suspension : Prop) (alignment : Prop) :
  cube_count = 10 →
  dimensions = 1 →
  (∀ x y, shared_edge (x, y) ↔ x = y ∨ x - y = 1) →
  no_suspension →
  alignment →
  distinct_left_views_count = 16 :=
by
  sorry

end NUMINAMATH_GPT_distinct_left_views_l745_74522


namespace NUMINAMATH_GPT_model_y_completion_time_l745_74508

theorem model_y_completion_time :
  ∀ (T : ℝ), (∃ k ≥ 0, k = 20) →
  (∀ (task_completed_x_per_minute : ℝ), task_completed_x_per_minute = 1 / 60) →
  (∀ (task_completed_y_per_minute : ℝ), task_completed_y_per_minute = 1 / T) →
  (20 * (1 / 60) + 20 * (1 / T) = 1) →
  T = 30 :=
by
  sorry

end NUMINAMATH_GPT_model_y_completion_time_l745_74508


namespace NUMINAMATH_GPT_second_year_growth_rate_l745_74597

variable (initial_investment : ℝ) (first_year_growth : ℝ) (additional_investment : ℝ) (final_value : ℝ) (second_year_growth : ℝ)

def calculate_portfolio_value_after_first_year (initial_investment first_year_growth : ℝ) : ℝ :=
  initial_investment * (1 + first_year_growth)

def calculate_new_value_after_addition (value_after_first_year additional_investment : ℝ) : ℝ :=
  value_after_first_year + additional_investment

def calculate_final_value_after_second_year (new_value second_year_growth : ℝ) : ℝ :=
  new_value * (1 + second_year_growth)

theorem second_year_growth_rate 
  (h1 : initial_investment = 80) 
  (h2 : first_year_growth = 0.15) 
  (h3 : additional_investment = 28) 
  (h4 : final_value = 132) : 
  calculate_final_value_after_second_year
    (calculate_new_value_after_addition
      (calculate_portfolio_value_after_first_year initial_investment first_year_growth)
      additional_investment)
    0.1 = final_value := 
  by
  sorry

end NUMINAMATH_GPT_second_year_growth_rate_l745_74597


namespace NUMINAMATH_GPT_yard_length_l745_74548

theorem yard_length (father_step : ℝ) (son_step : ℝ) (total_footprints : ℕ) 
  (h_father_step : father_step = 0.72) 
  (h_son_step : son_step = 0.54) 
  (h_total_footprints : total_footprints = 61) : 
  ∃ length : ℝ, length = 21.6 :=
by
  sorry

end NUMINAMATH_GPT_yard_length_l745_74548


namespace NUMINAMATH_GPT_max_value_quadratic_l745_74531

theorem max_value_quadratic (r : ℝ) : 
  ∃ M, (∀ r, -3 * r^2 + 36 * r - 9 ≤ M) ∧ M = 99 :=
sorry

end NUMINAMATH_GPT_max_value_quadratic_l745_74531


namespace NUMINAMATH_GPT_small_order_peanuts_l745_74569

theorem small_order_peanuts (total_peanuts : ℕ) (large_orders : ℕ) (peanuts_per_large : ℕ) 
    (small_orders : ℕ) (peanuts_per_small : ℕ) : 
    total_peanuts = large_orders * peanuts_per_large + small_orders * peanuts_per_small → 
    total_peanuts = 800 → 
    large_orders = 3 → 
    peanuts_per_large = 200 → 
    small_orders = 4 → 
    peanuts_per_small = 50 := by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_small_order_peanuts_l745_74569


namespace NUMINAMATH_GPT_somu_fathers_age_ratio_l745_74543

noncomputable def somus_age := 16

def proof_problem (S F : ℕ) : Prop :=
  S = 16 ∧ 
  (S - 8 = (1 / 5) * (F - 8)) ∧
  (S / F = 1 / 3)

theorem somu_fathers_age_ratio (S F : ℕ) : proof_problem S F :=
by
  sorry

end NUMINAMATH_GPT_somu_fathers_age_ratio_l745_74543


namespace NUMINAMATH_GPT_book_page_count_l745_74581

def total_pages_in_book (pages_three_nights_ago pages_two_nights_ago pages_last_night pages_tonight total_pages : ℕ) : Prop :=
  pages_three_nights_ago = 15 ∧
  pages_two_nights_ago = 2 * pages_three_nights_ago ∧
  pages_last_night = pages_two_nights_ago + 5 ∧
  pages_tonight = 20 ∧
  total_pages = pages_three_nights_ago + pages_two_nights_ago + pages_last_night + pages_tonight

theorem book_page_count : total_pages_in_book 15 30 35 20 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_book_page_count_l745_74581


namespace NUMINAMATH_GPT_perfect_square_representation_l745_74595

theorem perfect_square_representation :
  29 - 12*Real.sqrt 5 = (2*Real.sqrt 5 - 3*Real.sqrt 5 / 5)^2 :=
sorry

end NUMINAMATH_GPT_perfect_square_representation_l745_74595


namespace NUMINAMATH_GPT_kenny_jumps_l745_74553

theorem kenny_jumps (M : ℕ) (h : 34 + M + 0 + 123 + 64 + 23 + 61 = 325) : M = 20 :=
by
  sorry

end NUMINAMATH_GPT_kenny_jumps_l745_74553


namespace NUMINAMATH_GPT_single_elimination_games_l745_74599

theorem single_elimination_games (n : ℕ) (h : n = 23) : 
  ∃ games : ℕ, games = n - 1 :=
by
  use 22
  sorry

end NUMINAMATH_GPT_single_elimination_games_l745_74599


namespace NUMINAMATH_GPT_domain_range_of_g_l745_74501

variable (f : ℝ → ℝ)
variable (dom_f : Set.Icc 1 3)
variable (rng_f : Set.Icc 0 1)
variable (g : ℝ → ℝ)
variable (g_eq : ∀ x, g x = 2 - f (x - 1))

theorem domain_range_of_g :
  (Set.Icc 2 4) = { x | ∃ y, x = y ∧ g y = (g y) } ∧ Set.Icc 1 2 = { z | ∃ w, z = g w} :=
  sorry

end NUMINAMATH_GPT_domain_range_of_g_l745_74501


namespace NUMINAMATH_GPT_consecutive_digits_sum_190_to_199_l745_74525

-- Define the digits sum function
def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define ten consecutive numbers starting from m
def ten_consecutive_sum (m : ℕ) : ℕ :=
  (List.range 10).map (λ i => digits_sum (m + i)) |>.sum

theorem consecutive_digits_sum_190_to_199:
  ten_consecutive_sum 190 = 145 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_digits_sum_190_to_199_l745_74525


namespace NUMINAMATH_GPT_problem_1_problem_2_l745_74511

-- (1) Conditions and proof statement
theorem problem_1 (x y m : ℝ) (P : ℝ × ℝ) (k : ℝ) :
  (x, y) = (1, 2) → m = 1 →
  ((x - 1)^2 + (y - 2)^2 = 4) →
  P = (3, -1) →
  (l : ℝ → ℝ → Prop) →
  (∀ x y, l x y ↔ x = 3 ∨ (5 * x + 12 * y - 3 = 0)) →
  l 3 (-1) →
  l (x + k * (3 - x)) (y-1) := sorry

-- (2) Conditions and proof statement
theorem problem_2 (x y m : ℝ) (line : ℝ → ℝ) :
  (x - 1)^2 + (y - 2)^2 = 5 - m →
  m < 5 →
  (2 * (5 - m - 20) ^ (1/2) = 2 * (5) ^ (1/2)) →
  m = -20 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l745_74511


namespace NUMINAMATH_GPT_Carol_max_chance_l745_74517

-- Definitions of the conditions
def Alice_random_choice (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1
def Bob_random_choice (b : ℝ) : Prop := 0.4 ≤ b ∧ b ≤ 0.6
def Carol_wins (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

-- Statement that Carol maximizes her chances by picking 0.5
theorem Carol_max_chance : ∃ c : ℝ, (∀ a b : ℝ, Alice_random_choice a → Bob_random_choice b → Carol_wins a b c) ∧ c = 0.5 := 
sorry

end NUMINAMATH_GPT_Carol_max_chance_l745_74517


namespace NUMINAMATH_GPT_homework_duration_equation_l745_74554

-- Define the initial and final durations and the rate of decrease
def initial_duration : ℝ := 100
def final_duration : ℝ := 70
def rate_of_decrease (x : ℝ) : ℝ := x

-- Statement of the proof problem
theorem homework_duration_equation (x : ℝ) :
  initial_duration * (1 - rate_of_decrease x) ^ 2 = final_duration :=
sorry

end NUMINAMATH_GPT_homework_duration_equation_l745_74554


namespace NUMINAMATH_GPT_problem1_problem2_l745_74504

-- Problem 1
theorem problem1 (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) + (b / (b - a)) = 1 := 
sorry

-- Problem 2
theorem problem2 (m : ℝ) : 
  (m^2 - 4) / (4 + 4 * m + m^2) / ((m - 2) / (2 * m - 2)) * ((m + 2) / (m - 1)) = 2 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l745_74504


namespace NUMINAMATH_GPT_sum_of_selected_terms_l745_74598

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function from natural numbers to rational numbers

noncomputable def sum_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_of_selected_terms (h₁ : sum_first_n_terms a 13 = 39) : a 6 + a 7 + a 8 = 13 :=
sorry

end NUMINAMATH_GPT_sum_of_selected_terms_l745_74598


namespace NUMINAMATH_GPT_smallest_prime_divisor_of_sum_l745_74561

theorem smallest_prime_divisor_of_sum (h1 : ∃ k : ℕ, 3^19 = 2*k + 1)
                                      (h2 : ∃ l : ℕ, 11^13 = 2*l + 1) :
  Nat.minFac (3^19 + 11^13) = 2 := 
by
  sorry

end NUMINAMATH_GPT_smallest_prime_divisor_of_sum_l745_74561


namespace NUMINAMATH_GPT_parallel_vectors_sufficiency_l745_74516

noncomputable def parallel_vectors_sufficiency_problem (a b : ℝ × ℝ) (x : ℝ) : Prop :=
a = (1, x) ∧ b = (x, 4) →
(x = 2 → ∃ k : ℝ, k • a = b) ∧ (∃ k : ℝ, k • a = b → x = 2 ∨ x = -2)

theorem parallel_vectors_sufficiency (x : ℝ) :
  parallel_vectors_sufficiency_problem (1, x) (x, 4) x :=
sorry

end NUMINAMATH_GPT_parallel_vectors_sufficiency_l745_74516


namespace NUMINAMATH_GPT_ratio_of_product_of_composites_l745_74576

theorem ratio_of_product_of_composites :
  let A := [4, 6, 8, 9, 10, 12]
  let B := [14, 15, 16, 18, 20, 21]
  (A.foldl (λ x y => x * y) 1) / (B.foldl (λ x y => x * y) 1) = 1 / 49 :=
by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_ratio_of_product_of_composites_l745_74576


namespace NUMINAMATH_GPT_range_of_a_l745_74563

-- Defining the problem conditions
def f (x : ℝ) : ℝ := sorry -- The function f : ℝ → ℝ is defined elsewhere such that its range is [0, 4]
def g (a x : ℝ) : ℝ := a * x - 1

-- Theorem to prove the range of 'a'
theorem range_of_a (a : ℝ) : (a ≥ 1/2) ∨ (a ≤ -1/2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l745_74563


namespace NUMINAMATH_GPT_siamese_cats_initially_l745_74556

theorem siamese_cats_initially (house_cats: ℕ) (cats_sold: ℕ) (cats_left: ℕ) (initial_siamese: ℕ) :
  house_cats = 5 → 
  cats_sold = 10 → 
  cats_left = 8 → 
  (initial_siamese + house_cats - cats_sold = cats_left) → 
  initial_siamese = 13 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_siamese_cats_initially_l745_74556


namespace NUMINAMATH_GPT_heartsuit_ratio_l745_74579

def k : ℝ := 3

def heartsuit (n m : ℕ) : ℝ := k * n^3 * m^2

theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 3 / 5 := 
by
  sorry

end NUMINAMATH_GPT_heartsuit_ratio_l745_74579


namespace NUMINAMATH_GPT_hexagon_arithmetic_sum_l745_74526

theorem hexagon_arithmetic_sum (a n : ℝ) (h : 6 * a + 15 * n = 720) : 2 * a + 5 * n = 240 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_arithmetic_sum_l745_74526


namespace NUMINAMATH_GPT_smallest_sum_B_c_l745_74593

theorem smallest_sum_B_c : 
  ∃ (B : ℕ) (c : ℕ), (0 ≤ B ∧ B ≤ 4) ∧ (c ≥ 6) ∧ 31 * B = 4 * (c + 1) ∧ B + c = 8 := 
sorry

end NUMINAMATH_GPT_smallest_sum_B_c_l745_74593


namespace NUMINAMATH_GPT_cost_of_12_roll_package_is_correct_l745_74533

variable (cost_per_roll_package : ℝ)
variable (individual_cost_per_roll : ℝ := 1)
variable (number_of_rolls : ℕ := 12)
variable (percent_savings : ℝ := 0.25)

-- The definition of the total cost of the package
def total_cost_package := number_of_rolls * (individual_cost_per_roll - (percent_savings * individual_cost_per_roll))

-- The goal is to prove that the total cost of the package is $9
theorem cost_of_12_roll_package_is_correct : total_cost_package = 9 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_12_roll_package_is_correct_l745_74533


namespace NUMINAMATH_GPT_total_parallelepipeds_l745_74509

theorem total_parallelepipeds (m n k : ℕ) : 
  ∃ (num : ℕ), num == (m * n * k * (m + 1) * (n + 1) * (k + 1)) / 8 :=
  sorry

end NUMINAMATH_GPT_total_parallelepipeds_l745_74509
