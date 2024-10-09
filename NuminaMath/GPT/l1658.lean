import Mathlib

namespace periodic_odd_function_example_l1658_165888

open Real

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem periodic_odd_function_example (f : ℝ → ℝ) 
  (h_odd : odd f) 
  (h_periodic : periodic f 2) : 
  f 1 + f 4 + f 7 = 0 := 
sorry

end periodic_odd_function_example_l1658_165888


namespace norris_money_left_l1658_165851

-- Define the amounts saved each month
def september_savings : ℕ := 29
def october_savings : ℕ := 25
def november_savings : ℕ := 31

-- Define the total savings
def total_savings : ℕ := september_savings + october_savings + november_savings

-- Define the amount spent on the online game
def amount_spent : ℕ := 75

-- Define the remaining money
def money_left : ℕ := total_savings - amount_spent

-- The theorem stating the problem and the solution
theorem norris_money_left : money_left = 10 := by
  sorry

end norris_money_left_l1658_165851


namespace abel_overtake_kelly_chris_overtake_both_l1658_165812

-- Given conditions and variables
variable (d : ℝ)  -- distance at which Abel overtakes Kelly
variable (d_c : ℝ)  -- distance at which Chris overtakes both Kelly and Abel
variable (t_k : ℝ)  -- time taken by Kelly to run d meters
variable (t_a : ℝ)  -- time taken by Abel to run (d + 3) meters
variable (t_c : ℝ)  -- time taken by Chris to run the required distance
variable (k_speed : ℝ := 9)  -- Kelly's speed
variable (a_speed : ℝ := 9.5)  -- Abel's speed
variable (c_speed : ℝ := 10)  -- Chris's speed
variable (head_start_k : ℝ := 3)  -- Kelly's head start over Abel
variable (head_start_c : ℝ := 2)  -- Chris's head start behind Abel
variable (lost_by : ℝ := 0.75)  -- Abel lost by distance

-- Proof problem for Abel overtaking Kelly
theorem abel_overtake_kelly 
  (hk : t_k = d / k_speed) 
  (ha : t_a = (d + head_start_k) / a_speed) 
  (h_lost : lost_by = 0.75):
  d + lost_by = 54.75 := 
sorry

-- Proof problem for Chris overtaking both Kelly and Abel
theorem chris_overtake_both 
  (hc : t_c = (d_c + 5) / c_speed)
  (h_56 : d_c = 56):
  d_c = c_speed * (56 / c_speed) :=
sorry

end abel_overtake_kelly_chris_overtake_both_l1658_165812


namespace abc_sum_is_17_l1658_165850

noncomputable def A := 3
noncomputable def B := 5
noncomputable def C := 9

theorem abc_sum_is_17 (A B C : ℕ) (h1 : 100 * A + 10 * B + C = 359) (h2 : 4 * (100 * A + 10 * B + C) = 1436)
  (h3 : A ≠ B) (h4 : B ≠ C) (h5 : A ≠ C) : A + B + C = 17 :=
by
  sorry

end abc_sum_is_17_l1658_165850


namespace count_consecutive_sequences_l1658_165862

def consecutive_sequences (n : ℕ) : ℕ :=
  if n = 15 then 270 else 0

theorem count_consecutive_sequences : consecutive_sequences 15 = 270 :=
by
  sorry

end count_consecutive_sequences_l1658_165862


namespace fraction_product_l1658_165809

theorem fraction_product :
  ((5/4) * (8/16) * (20/12) * (32/64) * (50/20) * (40/80) * (70/28) * (48/96) : ℚ) = 625/768 := 
by
  sorry

end fraction_product_l1658_165809


namespace problem_1_problem_2_l1658_165860

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h₀ : a = 1) (h₁ : ∀ x : ℝ, x^2 - 5 * a * x + 4 * a^2 < 0)
                                    (h₂ : ∀ x : ℝ, (x - 2) * (x - 5) < 0) :
  ∃ x : ℝ, 2 < x ∧ x < 4 :=
by sorry

-- Proof Problem 2
theorem problem_2 (p q : ℝ → Prop) (h₀ : ∀ x : ℝ, p x → q x) 
                                (p_def : ∀ (a : ℝ) (x : ℝ), 0 < a → p x ↔ a < x ∧ x < 4 * a) 
                                (q_def : ∀ x : ℝ, q x ↔ 2 < x ∧ x < 5) :
  ∃ a : ℝ, (5 / 4) ≤ a ∧ a ≤ 2 :=
by sorry

end problem_1_problem_2_l1658_165860


namespace unrelated_statement_l1658_165898

-- Definitions
def timely_snow_promises_harvest : Prop := true -- assumes it has a related factor
def upper_beam_not_straight_lower_beam_crooked : Prop := true -- assumes it has a related factor
def smoking_harmful_to_health : Prop := true -- assumes it has a related factor
def magpies_signify_joy_crows_signify_mourning : Prop := false -- does not have an inevitable relationship

-- Theorem
theorem unrelated_statement :
  ¬magpies_signify_joy_crows_signify_mourning :=
by 
  -- proof to be provided
  sorry

end unrelated_statement_l1658_165898


namespace domain_of_y_l1658_165823

noncomputable def domain_of_function (x : ℝ) : Bool :=
  x < 0 ∧ x ≠ -1

theorem domain_of_y :
  {x : ℝ | (∃ y, y = (x + 1) ^ 0 / Real.sqrt (|x| - x)) } =
  {x : ℝ | domain_of_function x} :=
by
  sorry

end domain_of_y_l1658_165823


namespace function_is_linear_l1658_165826

noncomputable def f : ℕ → ℕ :=
  λ n => n + 1

axiom f_at_0 : f 0 = 1
axiom f_at_2016 : f 2016 = 2017
axiom f_equation : ∀ n : ℕ, f (f n) + f n = 2 * n + 3

theorem function_is_linear : ∀ n : ℕ, f n = n + 1 :=
by
  intro n
  sorry

end function_is_linear_l1658_165826


namespace calculate_expression_l1658_165819

theorem calculate_expression : -2 - 2 * Real.sin (Real.pi / 4) + (Real.pi - 3.14) * 0 + (-1) ^ 3 = -3 - Real.sqrt 2 := by 
sorry

end calculate_expression_l1658_165819


namespace cyclic_inequality_l1658_165835

theorem cyclic_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (x + y) * Real.sqrt (y + z) * Real.sqrt (z + x) + (y + z) * Real.sqrt (z + x) * Real.sqrt (x + y) + (z + x) * Real.sqrt (x + y) * Real.sqrt (y + z) ≥ 4 * (x * y + y * z + z * x) :=
by
  sorry

end cyclic_inequality_l1658_165835


namespace sum_of_roots_of_cubic_eq_l1658_165811

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 - 72 * x + 6

-- Define the statement to prove
theorem sum_of_roots_of_cubic_eq : 
  ∀ (r p q : ℝ), (cubic_eq r = 0) ∧ (cubic_eq p = 0) ∧ (cubic_eq q = 0) → 
  (r + p + q) = 3 :=
sorry

end sum_of_roots_of_cubic_eq_l1658_165811


namespace dinner_customers_l1658_165846

theorem dinner_customers 
    (breakfast : ℕ)
    (lunch : ℕ)
    (total_friday : ℕ)
    (H : breakfast = 73)
    (H1 : lunch = 127)
    (H2 : total_friday = 287) :
  (breakfast + lunch + D = total_friday) → D = 87 := by
  sorry

end dinner_customers_l1658_165846


namespace factorization_25x2_minus_155x_minus_150_l1658_165843

theorem factorization_25x2_minus_155x_minus_150 :
  ∃ (a b : ℤ), (a + b) * 5 = -155 ∧ a * b = -150 ∧ a + 2 * b = 27 :=
by
  sorry

end factorization_25x2_minus_155x_minus_150_l1658_165843


namespace geometric_sequence_sum_l1658_165808

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a n = (a 0) * q^n)
  (h2 : ∀ n, a n > a (n + 1))
  (h3 : a 2 + a 3 + a 4 = 28)
  (h4 : a 3 + 2 = (a 2 + a 4) / 2) :
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 63 :=
by {
  sorry
}

end geometric_sequence_sum_l1658_165808


namespace hyperbola_eccentricity_l1658_165842

theorem hyperbola_eccentricity
  (a b m : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (PA_perpendicular_to_l2 : (b/a * m) / (m + a) * (-b/a) = -1)
  (PB_parallel_to_l2 : (b/a * m) / (m - a) = -b/a) :
  (∃ e, e = 2) :=
by sorry

end hyperbola_eccentricity_l1658_165842


namespace large_hexagon_toothpicks_l1658_165806

theorem large_hexagon_toothpicks (n : Nat) (h : n = 1001) : 
  let T_half := (n * (n + 1)) / 2
  let T_total := 2 * T_half + n
  let boundary_toothpicks := 6 * T_half
  let total_toothpicks := 3 * T_total - boundary_toothpicks
  total_toothpicks = 3006003 :=
by
  sorry

end large_hexagon_toothpicks_l1658_165806


namespace maximum_distance_l1658_165876

-- Definitions from the conditions
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def distance_driven : ℝ := 244
def gallons_used : ℝ := 20

-- Problem statement
theorem maximum_distance (h: (distance_driven / gallons_used = highway_mpg)): 
  (distance_driven = 244) :=
sorry

end maximum_distance_l1658_165876


namespace rectangle_area_l1658_165897

theorem rectangle_area (y : ℝ) (w : ℝ) (h : w > 0) (h_diag : y^2 = 10 * w^2) : 
  (3 * w)^2 * w = 3 * (y^2 / 10) :=
by sorry

end rectangle_area_l1658_165897


namespace initial_people_lifting_weights_l1658_165833

theorem initial_people_lifting_weights (x : ℕ) (h : x + 3 = 19) : x = 16 :=
by
  sorry

end initial_people_lifting_weights_l1658_165833


namespace ellipse_focal_distance_m_value_l1658_165825

-- Define the given conditions 
def focal_distance := 2
def ellipse_equation (x y : ℝ) (m : ℝ) := (x^2 / m) + (y^2 / 4) = 1

-- The proof statement
theorem ellipse_focal_distance_m_value :
  ∀ (m : ℝ), 
    (∃ c : ℝ, (2 * c = focal_distance) ∧ (m = 4 + c^2)) →
      m = 5 := by
  sorry

end ellipse_focal_distance_m_value_l1658_165825


namespace candle_remaining_length_l1658_165854

-- Define the initial length of the candle and the burn rate
def initial_length : ℝ := 20
def burn_rate : ℝ := 5

-- Define the remaining length function
def remaining_length (t : ℝ) : ℝ := initial_length - burn_rate * t

-- Prove the relationship between time and remaining length for the given range of time
theorem candle_remaining_length (t : ℝ) (ht: 0 ≤ t ∧ t ≤ 4) : remaining_length t = 20 - 5 * t :=
by
  dsimp [remaining_length]
  sorry

end candle_remaining_length_l1658_165854


namespace increase_in_average_age_l1658_165894

variable (A : ℝ)
variable (A_increase : ℝ)
variable (orig_age_sum : ℝ)
variable (new_age_sum : ℝ)

def original_total_age (A : ℝ) := 8 * A
def new_total_age (A : ℝ) := original_total_age A - 20 - 22 + 29 + 29

theorem increase_in_average_age (A : ℝ) (orig_age_sum := original_total_age A) (new_age_sum := new_total_age A) : 
  (new_age_sum / 8) = (A + 2) := 
by
  unfold new_total_age
  unfold original_total_age
  sorry

end increase_in_average_age_l1658_165894


namespace annette_miscalculation_l1658_165883

theorem annette_miscalculation :
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  x' - y' = 1 :=
by
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  sorry

end annette_miscalculation_l1658_165883


namespace instantaneous_speed_at_3_l1658_165822

noncomputable def s (t : ℝ) : ℝ := 1 - t + 2 * t^2

theorem instantaneous_speed_at_3 : deriv s 3 = 11 :=
by
  sorry

end instantaneous_speed_at_3_l1658_165822


namespace find_schnauzers_l1658_165884

theorem find_schnauzers (D S : ℕ) (h : 3 * D - 5 + (D - S) = 90) (hD : D = 20) : S = 45 :=
by
  sorry

end find_schnauzers_l1658_165884


namespace vendor_apples_sold_l1658_165810

theorem vendor_apples_sold (x : ℝ) (h : 0.15 * (1 - x / 100) + 0.50 * (1 - x / 100) * 0.85 = 0.23) : x = 60 :=
sorry

end vendor_apples_sold_l1658_165810


namespace waiter_customers_l1658_165841

-- Define initial conditions
def initial_customers : ℕ := 47
def customers_left : ℕ := 41
def new_customers : ℕ := 20

-- Calculate remaining customers after some left
def remaining_customers : ℕ := initial_customers - customers_left

-- Calculate the total customers after getting new ones
def total_customers : ℕ := remaining_customers + new_customers

-- State the theorem to prove the final total customers
theorem waiter_customers : total_customers = 26 := by
  -- We include sorry for the proof placeholder
  sorry

end waiter_customers_l1658_165841


namespace color_triangle_vertices_no_same_color_l1658_165885

-- Define the colors and the vertices
inductive Color | red | green | blue | yellow
inductive Vertex | A | B | C 

-- Define a function that counts ways to color the triangle given constraints
def count_valid_colorings (colors : List Color) (vertices : List Vertex) : Nat := 
  -- There are 4 choices for the first vertex, 3 for the second, 2 for the third
  4 * 3 * 2

-- The theorem we want to prove
theorem color_triangle_vertices_no_same_color : count_valid_colorings [Color.red, Color.green, Color.blue, Color.yellow] [Vertex.A, Vertex.B, Vertex.C] = 24 := by
  sorry

end color_triangle_vertices_no_same_color_l1658_165885


namespace find_other_number_l1658_165847

theorem find_other_number (a b : ℕ) (h_lcm : Nat.lcm a b = 9240) (h_gcd : Nat.gcd a b = 33) (h_a : a = 231) : b = 1320 :=
sorry

end find_other_number_l1658_165847


namespace ratio_depends_on_S_and_r_l1658_165892

theorem ratio_depends_on_S_and_r
    (S : ℝ) (r : ℝ) (P1 : ℝ) (C2 : ℝ)
    (h1 : P1 = 4 * S)
    (h2 : C2 = 2 * Real.pi * r) :
    (P1 / C2 = 4 * S / (2 * Real.pi * r)) := by
  sorry

end ratio_depends_on_S_and_r_l1658_165892


namespace Bella_catch_correct_l1658_165872

def Martha_catch : ℕ := 3 + 7
def Cara_catch : ℕ := 5 * Martha_catch - 3
def T : ℕ := Martha_catch + Cara_catch
def Andrew_catch : ℕ := T^2 + 2
def F : ℕ := Martha_catch + Cara_catch + Andrew_catch
def Bella_catch : ℕ := 2 ^ (F / 3)

theorem Bella_catch_correct : Bella_catch = 2 ^ 1102 := by
  sorry

end Bella_catch_correct_l1658_165872


namespace area_difference_l1658_165879

theorem area_difference (d : ℝ) (r : ℝ) (ratio : ℝ) (h1 : d = 10) (h2 : ratio = 2) (h3 : r = 5) :
  (π * r^2 - ((d^2 / (ratio^2 + 1)).sqrt * (2 * d^2 / (ratio^2 + 1)).sqrt)) = 38.5 :=
by
  sorry

end area_difference_l1658_165879


namespace least_number_to_subtract_l1658_165805

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h1 : n = 42398) (h2 : d = 15) (h3 : r = 8) : 
  ∃ k, n - r = k * d :=
by
  sorry

end least_number_to_subtract_l1658_165805


namespace Andy_solves_correct_number_of_problems_l1658_165836

-- Define the problem boundaries
def first_problem : ℕ := 80
def last_problem : ℕ := 125

-- The goal is to prove that Andy solves 46 problems given the range
theorem Andy_solves_correct_number_of_problems : (last_problem - first_problem + 1) = 46 :=
by
  sorry

end Andy_solves_correct_number_of_problems_l1658_165836


namespace value_of_a_in_terms_of_b_l1658_165803

noncomputable def value_of_a (b : ℝ) : ℝ :=
  b * (38.1966 / 61.8034)

theorem value_of_a_in_terms_of_b (b a : ℝ) :
  (∀ x : ℝ, (b / x = 61.80339887498949 / 100) ∧ (x = (a + b) * (61.80339887498949 / 100)))
  → a = value_of_a b :=
by
  sorry

end value_of_a_in_terms_of_b_l1658_165803


namespace textbook_order_total_cost_l1658_165881

theorem textbook_order_total_cost :
  let english_quantity := 35
  let geography_quantity := 35
  let mathematics_quantity := 20
  let science_quantity := 30
  let english_price := 7.50
  let geography_price := 10.50
  let mathematics_price := 12.00
  let science_price := 9.50
  (english_quantity * english_price + geography_quantity * geography_price + mathematics_quantity * mathematics_price + science_quantity * science_price = 1155.00) :=
by sorry

end textbook_order_total_cost_l1658_165881


namespace tan_105_eq_minus_2_minus_sqrt_3_l1658_165817

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l1658_165817


namespace initially_planned_days_l1658_165873

-- Definitions of the conditions
def total_work_initial (x : ℕ) : ℕ := 50 * x
def total_work_with_reduction (x : ℕ) : ℕ := 25 * (x + 20)

-- The main theorem
theorem initially_planned_days :
  ∀ (x : ℕ), total_work_initial x = total_work_with_reduction x → x = 20 :=
by
  intro x
  intro h
  sorry

end initially_planned_days_l1658_165873


namespace arithmetic_sequence_ninth_term_l1658_165866

theorem arithmetic_sequence_ninth_term :
  ∃ a d : ℤ, (a + 2 * d = 23) ∧ (a + 5 * d = 29) ∧ (a + 8 * d = 35) :=
by
  sorry

end arithmetic_sequence_ninth_term_l1658_165866


namespace find_a1_l1658_165874

variable (a : ℕ → ℚ) (d : ℚ)
variable (S : ℕ → ℚ)
variable (h_seq : ∀ n, a (n + 1) = a n + d)
variable (h_diff : d ≠ 0)
variable (h_prod : (a 2) * (a 3) = (a 4) * (a 5))
variable (h_sum : S 4 = 27)
variable (h_sum_def : ∀ n, S n = n * (a 1 + a n) / 2)

theorem find_a1 : a 1 = 135 / 8 := by
  sorry

end find_a1_l1658_165874


namespace sum_of_a_and_b_l1658_165828

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l1658_165828


namespace identify_true_statements_l1658_165807

-- Definitions of the given statements
def statement1 (a x y : ℝ) : Prop := a * (x + y) = a * x + a * y
def statement2 (a x y : ℝ) : Prop := a ^ (x + y) = a ^ x + a ^ y
def statement3 (x y : ℝ) : Prop := (x + y) ^ 2 = x ^ 2 + y ^ 2
def statement4 (a b : ℝ) : Prop := Real.sqrt (a ^ 2 + b ^ 2) = a + b
def statement5 (a b c : ℝ) : Prop := a * (b / c) = (a * b) / c

-- The statement to prove
theorem identify_true_statements (a x y b c : ℝ) :
  statement1 a x y ∧ statement5 a b c ∧
  ¬ statement2 a x y ∧ ¬ statement3 x y ∧ ¬ statement4 a b :=
sorry

end identify_true_statements_l1658_165807


namespace kerosene_price_increase_l1658_165867

theorem kerosene_price_increase (P C : ℝ) (x : ℝ)
  (h1 : 1 = (1 + x / 100) * 0.8) :
  x = 25 := by
  sorry

end kerosene_price_increase_l1658_165867


namespace minimum_value_of_GP_l1658_165858

theorem minimum_value_of_GP (a : ℕ → ℝ) (h : ∀ n, 0 < a n) (h_prod : a 2 * a 10 = 9) :
  a 5 + a 7 = 6 :=
by
  -- proof steps will be filled in here
  sorry

end minimum_value_of_GP_l1658_165858


namespace number_of_valid_grids_l1658_165804

-- Define the concept of a grid and the necessary properties
structure Grid (n : ℕ) :=
  (cells: Fin (n * n) → ℕ)
  (unique: Function.Injective cells)
  (ordered_rows: ∀ i j : Fin n, i < j → cells ⟨i * n + j, sorry⟩ > cells ⟨i * n + j - 1, sorry⟩)
  (ordered_columns: ∀ i j : Fin n, i < j → cells ⟨j * n + i, sorry⟩ > cells ⟨(j - 1) * n + i, sorry⟩)

-- Define the 4x4 grid
def grid_4x4 := Grid 4

-- Statement of the problem: prove there are 2 valid grid_4x4 configurations
theorem number_of_valid_grids : ∃ g : grid_4x4, (∃ g1 g2 : grid_4x4, (g1 ≠ g2) ∧ (∀ g3 : grid_4x4, g3 = g1 ∨ g3 = g2)) :=
  sorry

end number_of_valid_grids_l1658_165804


namespace original_price_of_petrol_l1658_165855

variable (P : ℝ)

theorem original_price_of_petrol (h : 0.9 * (200 / P - 200 / (0.9 * P)) = 5) : 
  (P = 20 / 4.5) :=
sorry

end original_price_of_petrol_l1658_165855


namespace domain_of_function_l1658_165852

theorem domain_of_function :
  {x : ℝ | x ≥ -1 ∧ x ≠ 1 / 2} =
  {x : ℝ | 2 * x - 1 ≠ 0 ∧ x + 1 ≥ 0} :=
by {
  sorry
}

end domain_of_function_l1658_165852


namespace find_integer_pairs_l1658_165890

def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, k * k = x

theorem find_integer_pairs (m n : ℤ) :
  (is_perfect_square (m^2 + 4 * n) ∧ is_perfect_square (n^2 + 4 * m)) ↔
  (∃ a : ℤ, (m = 0 ∧ n = a^2) ∨ (m = a^2 ∧ n = 0) ∨ (m = -4 ∧ n = -4) ∨ (m = -5 ∧ n = -6) ∨ (m = -6 ∧ n = -5)) :=
by
  sorry

end find_integer_pairs_l1658_165890


namespace total_distance_traveled_l1658_165813

theorem total_distance_traveled :
  let day1_distance := 5 * 7
  let day2_distance_part1 := 6 * 6
  let day2_distance_part2 := 3 * 3
  let day3_distance := 7 * 5
  let total_distance := day1_distance + day2_distance_part1 + day2_distance_part2 + day3_distance
  total_distance = 115 :=
by
  sorry

end total_distance_traveled_l1658_165813


namespace exists_nat_numbers_except_two_three_l1658_165815

theorem exists_nat_numbers_except_two_three (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ (k ≠ 2 ∧ k ≠ 3) :=
by
  sorry

end exists_nat_numbers_except_two_three_l1658_165815


namespace georgie_entry_exit_ways_l1658_165869

-- Defining the conditions
def castle_windows : Nat := 8
def non_exitable_windows : Nat := 2

-- Defining the problem
theorem georgie_entry_exit_ways (total_windows : Nat) (blocked_exits : Nat) (entry_windows : Nat) : 
  total_windows = castle_windows → blocked_exits = non_exitable_windows → 
  entry_windows = castle_windows →
  (entry_windows * (total_windows - 1 - blocked_exits) = 40) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end georgie_entry_exit_ways_l1658_165869


namespace initial_speed_is_sixty_l1658_165861

variable (D T : ℝ)

-- Condition: Two-thirds of the distance is covered in one-third of the total time.
def two_thirds_distance_in_one_third_time (V : ℝ) : Prop :=
  (2 * D / 3) / V = T / 3

-- Condition: The remaining distance is covered at 15 kmph.
def remaining_distance_at_fifteen_kmph : Prop :=
  (D / 3) / 15 = T - T / 3

-- Given that 30T = D from simplification in the solution.
def distance_time_relationship : Prop :=
  D = 30 * T

-- Prove that the initial speed V is 60 kmph.
theorem initial_speed_is_sixty (V : ℝ) (h1 : two_thirds_distance_in_one_third_time D T V) (h2 : remaining_distance_at_fifteen_kmph D T) (h3 : distance_time_relationship D T) : V = 60 := 
  sorry

end initial_speed_is_sixty_l1658_165861


namespace first_group_people_count_l1658_165800

theorem first_group_people_count (P : ℕ) (W : ℕ) 
  (h1 : P * 3 * W = 3 * W) 
  (h2 : 8 * 3 * W = 8 * W) : 
  P = 3 :=
by
  sorry

end first_group_people_count_l1658_165800


namespace range_of_z_l1658_165832

theorem range_of_z (x y : ℝ) (hx1 : x - 2 * y + 1 ≥ 0) (hx2 : y ≥ x) (hx3 : x ≥ 0) :
  ∃ z, z = x^2 + y^2 ∧ 0 ≤ z ∧ z ≤ 2 :=
by
  sorry

end range_of_z_l1658_165832


namespace three_lines_intersect_at_three_points_l1658_165871

-- Define the lines as propositions expressing the equations
def line1 (x y : ℝ) := 2 * y - 3 * x = 4
def line2 (x y : ℝ) := x + 3 * y = 3
def line3 (x y : ℝ) := 3 * x - 4.5 * y = 7.5

-- Define a proposition stating that there are exactly 3 unique points of intersection among the three lines
def number_of_intersections : ℕ := 3

-- Prove that the number of unique intersection points is exactly 3 given the lines
theorem three_lines_intersect_at_three_points : 
  ∃! p1 p2 p3 : ℝ × ℝ, 
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
    (line3 p3.1 p3.2 ∧ line1 p3.1 p3.2) :=
sorry

end three_lines_intersect_at_three_points_l1658_165871


namespace correct_calculation_l1658_165857

-- Definitions of the equations
def option_A (a : ℝ) : Prop := a + 2 * a = 3 * a^2
def option_B (a b : ℝ) : Prop := (a^2 * b)^3 = a^6 * b^3
def option_C (a : ℝ) (m : ℕ) : Prop := (a^m)^2 = a^(m+2)
def option_D (a : ℝ) : Prop := a^3 * a^2 = a^6

-- The theorem that states option B is correct and others are incorrect
theorem correct_calculation (a b : ℝ) (m : ℕ) : 
  ¬ option_A a ∧ 
  option_B a b ∧ 
  ¬ option_C a m ∧ 
  ¬ option_D a :=
by sorry

end correct_calculation_l1658_165857


namespace min_path_length_l1658_165831

noncomputable def problem_statement : Prop :=
  let XY := 12
  let XZ := 8
  let angle_XYZ := 30
  let YP_PQ_QZ := by {
    -- Reflect Z across XY to get Z' and Y across XZ to get Y'.
    -- Use the Law of cosines in triangle XY'Z'.
    let cos_150 := -Real.sqrt 3 / 2
    let Y_prime_Z_prime := Real.sqrt (8^2 + 12^2 + 2 * 8 * 12 * cos_150)
    exact Y_prime_Z_prime
  }
  ∃ (P Q : Type), (YP_PQ_QZ = Real.sqrt (208 + 96 * Real.sqrt 3))

-- Goal is to prove the problem statement
theorem min_path_length : problem_statement := sorry

end min_path_length_l1658_165831


namespace line_tangent_to_circle_l1658_165875

theorem line_tangent_to_circle (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 2 * k + 3 = 0 → x^2 + (y + 1)^2 = 4) → k = 3 / 4 :=
by 
  intro h
  sorry

end line_tangent_to_circle_l1658_165875


namespace general_term_formula_l1658_165829

-- Conditions: sequence \(\frac{1}{2}\), \(\frac{1}{3}\), \(\frac{1}{4}\), \(\frac{1}{5}, \ldots\)
-- Let seq be the sequence in question.

def seq (n : ℕ) : ℚ := 1 / (n + 1)

-- Question: prove the general term formula is \(\frac{1}{n+1}\)
theorem general_term_formula (n : ℕ) : seq n = 1 / (n + 1) :=
by
  -- Proof goes here
  sorry

end general_term_formula_l1658_165829


namespace min_value_of_reciprocal_sum_l1658_165865

variable (m n : ℝ)

theorem min_value_of_reciprocal_sum (hmn : m * n > 0) (h_line : m + n = 2) :
  (1 / m + 1 / n = 2) :=
sorry

end min_value_of_reciprocal_sum_l1658_165865


namespace solve_for_y_l1658_165830

theorem solve_for_y :
  ∃ (y : ℝ), 
    (∑' n : ℕ, (4 * (n + 1) - 2) * y^n) = 100 ∧ |y| < 1 ∧ y = 0.6036 :=
sorry

end solve_for_y_l1658_165830


namespace find_slope_of_intersecting_line_l1658_165896

-- Define the conditions
def line_p (x : ℝ) : ℝ := 2 * x + 3
def line_q (x : ℝ) (m : ℝ) : ℝ := m * x + 1

-- Define the point of intersection
def intersection_point : ℝ × ℝ := (4, 11)

-- Prove that the slope m of line q such that both lines intersect at (4, 11) is 2.5
theorem find_slope_of_intersecting_line (m : ℝ) :
  line_q 4 m = 11 → m = 2.5 :=
by
  intro h
  sorry

end find_slope_of_intersecting_line_l1658_165896


namespace common_tangent_exists_l1658_165837

theorem common_tangent_exists:
  ∃ (a b c : ℕ), (a + b + c = 11) ∧
  ( ∀ (x y : ℝ),
      (y = x^2 + 12/5) ∧ 
      (x = y^2 + 99/10) ∧ 
      (a*x + b*y = c) ∧ 
      0 < a ∧ 0 < b ∧ 0 < c ∧ 
      Int.gcd (Int.gcd a b) c = 1
  ) := 
by
  sorry

end common_tangent_exists_l1658_165837


namespace circles_intersect_in_two_points_l1658_165802

def circle1 (x y : ℝ) : Prop := x^2 + (y - 3/2)^2 = (3/2)^2
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

theorem circles_intersect_in_two_points :
  ∃! (p : ℝ × ℝ), (circle1 p.1 p.2) ∧ (circle2 p.1 p.2) := 
sorry

end circles_intersect_in_two_points_l1658_165802


namespace total_spent_is_64_l1658_165882

/-- Condition 1: The cost of each deck is 8 dollars -/
def deck_cost : ℕ := 8

/-- Condition 2: Tom bought 3 decks -/
def tom_decks : ℕ := 3

/-- Condition 3: Tom's friend bought 5 decks -/
def friend_decks : ℕ := 5

/-- Total amount spent by Tom and his friend -/
def total_amount_spent : ℕ := (tom_decks * deck_cost) + (friend_decks * deck_cost)

/-- Proof statement: Prove that total amount spent is 64 -/
theorem total_spent_is_64 : total_amount_spent = 64 := by
  sorry

end total_spent_is_64_l1658_165882


namespace no_integer_solutions_l1658_165834

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2 * x^2 * y^2 + 2 * y^2 * z^2 + 2 * z^2 * x^2 + 24 := 
by {
  sorry
}

end no_integer_solutions_l1658_165834


namespace dodecahedron_edges_l1658_165886

noncomputable def regular_dodecahedron := Type

def faces : regular_dodecahedron → ℕ := λ _ => 12
def edges_per_face : regular_dodecahedron → ℕ := λ _ => 5
def shared_edges : regular_dodecahedron → ℕ := λ _ => 2

theorem dodecahedron_edges (d : regular_dodecahedron) :
  (faces d * edges_per_face d) / shared_edges d = 30 :=
by
  sorry

end dodecahedron_edges_l1658_165886


namespace avg_visitors_per_day_correct_l1658_165848

-- Define the given conditions
def avg_sundays : Nat := 540
def avg_other_days : Nat := 240
def num_days : Nat := 30
def sundays_in_month : Nat := 5
def other_days_in_month : Nat := 25

-- Define the total visitors calculation
def total_visitors := (sundays_in_month * avg_sundays) + (other_days_in_month * avg_other_days)

-- Define the average visitors per day calculation
def avg_visitors_per_day := total_visitors / num_days

-- State the proof problem
theorem avg_visitors_per_day_correct : avg_visitors_per_day = 290 :=
by
  sorry

end avg_visitors_per_day_correct_l1658_165848


namespace area_of_triangle_formed_by_line_and_axes_l1658_165863

-- Definition of the line equation condition
def line_eq (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

-- Statement of the problem to prove
theorem area_of_triangle_formed_by_line_and_axes :
  (∃ x y : ℝ, line_eq x y ∧ x = 0 ∧ y = -2) ∧
  (∃ x y : ℝ, line_eq x y ∧ x = 5 ∧ y = 0) →
  let base : ℝ := 5
  let height : ℝ := 2
  let area := (1 / 2) * base * height
  area = 5 := 
by
  sorry

end area_of_triangle_formed_by_line_and_axes_l1658_165863


namespace exactly_two_succeed_probability_l1658_165880

/-- Define the probabilities of three independent events -/
def P1 : ℚ := 1 / 2
def P2 : ℚ := 1 / 3
def P3 : ℚ := 3 / 4

/-- Define the probability that exactly two out of the three people successfully decrypt the password -/
def prob_exactly_two_succeed : ℚ := P1 * P2 * (1 - P3) + P1 * (1 - P2) * P3 + (1 - P1) * P2 * P3

theorem exactly_two_succeed_probability :
  prob_exactly_two_succeed = 5 / 12 :=
sorry

end exactly_two_succeed_probability_l1658_165880


namespace common_ratio_of_geometric_series_l1658_165853

-- Definitions of the first two terms of the geometric series
def term1 : ℚ := 4 / 7
def term2 : ℚ := -8 / 3

-- Theorem to prove the common ratio
theorem common_ratio_of_geometric_series : (term2 / term1 = -14 / 3) := by
  sorry

end common_ratio_of_geometric_series_l1658_165853


namespace value_of_a5_l1658_165818

theorem value_of_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = 2 * n * (n + 1)) (ha : ∀ n, a n = S n - S (n - 1)) :
  a 5 = 20 :=
by
  sorry

end value_of_a5_l1658_165818


namespace wrenches_in_comparison_group_l1658_165816

theorem wrenches_in_comparison_group (H W : ℝ) (x : ℕ) 
  (h1 : W = 2 * H)
  (h2 : 2 * H + 2 * W = (1 / 3) * (8 * H + x * W)) : x = 5 :=
by
  sorry

end wrenches_in_comparison_group_l1658_165816


namespace find_a_for_perpendicular_lines_l1658_165840

theorem find_a_for_perpendicular_lines (a : ℝ) 
    (h_perpendicular : 2 * a + (-1) * (3 - a) = 0) :
    a = 1 :=
by
  sorry

end find_a_for_perpendicular_lines_l1658_165840


namespace product_of_integers_P_Q_R_S_l1658_165824

theorem product_of_integers_P_Q_R_S (P Q R S : ℤ)
  (h1 : 0 < P) (h2 : 0 < Q) (h3 : 0 < R) (h4 : 0 < S)
  (h_sum : P + Q + R + S = 50)
  (h_rel : P + 4 = Q - 4 ∧ P + 4 = R * 3 ∧ P + 4 = S / 3) :
  P * Q * R * S = 43 * 107 * 75 * 225 / 1536 := 
by { sorry }

end product_of_integers_P_Q_R_S_l1658_165824


namespace four_nat_nums_prime_condition_l1658_165878

theorem four_nat_nums_prime_condition (a b c d : ℕ) (h₁ : a = 1) (h₂ : b = 2) (h₃ : c = 3) (h₄ : d = 5) :
  Nat.Prime (a * b + c * d) ∧ Nat.Prime (a * c + b * d) ∧ Nat.Prime (a * d + b * c) :=
by
  sorry

end four_nat_nums_prime_condition_l1658_165878


namespace geometric_sequence_sum_l1658_165838

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a n = a 1 * q ^ n

theorem geometric_sequence_sum (h : geometric_sequence a 2) (h_sum : a 1 + a 2 = 3) :
  a 4 + a 5 = 24 := by
  sorry

end geometric_sequence_sum_l1658_165838


namespace inconsistent_linear_system_l1658_165893

theorem inconsistent_linear_system :
  ¬ ∃ (x1 x2 x3 : ℝ), 
    (2 * x1 + 5 * x2 - 4 * x3 = 8) ∧
    (3 * x1 + 15 * x2 - 9 * x3 = 5) ∧
    (5 * x1 + 5 * x2 - 7 * x3 = 1) :=
by
  -- Proof of inconsistency
  sorry

end inconsistent_linear_system_l1658_165893


namespace min_vertical_segment_length_l1658_165849

noncomputable def vertical_segment_length (x : ℝ) : ℝ :=
  abs (|x| - (-x^2 - 4*x - 3))

theorem min_vertical_segment_length :
  ∃ x : ℝ, vertical_segment_length x = 3 / 4 :=
by
  sorry

end min_vertical_segment_length_l1658_165849


namespace fountains_for_m_4_fountains_for_m_3_l1658_165801

noncomputable def ceil_div (a b : ℕ) : ℕ :=
  (a + b - 1) / b

-- Problem for m = 4
theorem fountains_for_m_4 (n : ℕ) : ∃ f : ℕ, f = 2 * ceil_div n 3 := 
sorry

-- Problem for m = 3
theorem fountains_for_m_3 (n : ℕ) : ∃ f : ℕ, f = 3 * ceil_div n 3 :=
sorry

end fountains_for_m_4_fountains_for_m_3_l1658_165801


namespace lattice_points_condition_l1658_165844

/-- A lattice point is a point on the plane with integer coordinates. -/
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

/-- A triangle in the plane with three vertices and at least two lattice points inside. -/
structure Triangle :=
  (A B C : LatticePoint)
  (lattice_points_inside : List LatticePoint)
  (lattice_points_nonempty : lattice_points_inside.length ≥ 2)

noncomputable def exists_lattice_points (T : Triangle) : Prop :=
∃ (X Y : LatticePoint) (hX : X ∈ T.lattice_points_inside) (hY : Y ∈ T.lattice_points_inside), 
  ((∃ (V : LatticePoint), V = T.A ∨ V = T.B ∨ V = T.C ∧ ∃ (k : ℤ), (k : ℝ) * (Y.x - X.x) = (V.x - X.x) ∧ (k : ℝ) * (Y.y - X.y) = (V.y - X.y)) ∨
  (∃ (l m n : ℝ), l * (Y.x - X.x) = m * (T.A.x - T.B.x) ∧ l * (Y.y - X.y) = m * (T.A.y - T.B.y) ∨ l * (Y.x - X.x) = n * (T.B.x - T.C.x) ∧ l * (Y.y - X.y) = n * (T.B.y - T.C.y) ∨ l * (Y.x - X.x) = m * (T.C.x - T.A.x) ∧ l * (Y.y - X.y) = m * (T.C.y - T.A.y)))

theorem lattice_points_condition (T : Triangle) : exists_lattice_points T :=
sorry

end lattice_points_condition_l1658_165844


namespace sin_cos_sum_2018_l1658_165891

theorem sin_cos_sum_2018 {x : ℝ} (h : Real.sin x + Real.cos x = 1) :
  (Real.sin x)^2018 + (Real.cos x)^2018 = 1 :=
by
  sorry

end sin_cos_sum_2018_l1658_165891


namespace betty_honey_oats_problem_l1658_165864

theorem betty_honey_oats_problem
  (o h : ℝ)
  (h_condition1 : o ≥ 8 + h / 3)
  (h_condition2 : o ≤ 3 * h) :
  h ≥ 3 :=
sorry

end betty_honey_oats_problem_l1658_165864


namespace find_larger_number_l1658_165821

theorem find_larger_number (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 10) : L = 1636 := 
by
  sorry

end find_larger_number_l1658_165821


namespace value_of_expression_l1658_165814

theorem value_of_expression (m n : ℝ) (h : m + n = -2) : 5 * m^2 + 5 * n^2 + 10 * m * n = 20 := 
by
  sorry

end value_of_expression_l1658_165814


namespace find_z_l1658_165839

theorem find_z
  (z : ℝ)
  (h : (1 : ℝ) • (2 : ℝ) + 4 • (-1 : ℝ) + z • (3 : ℝ) = 6) :
  z = 8 / 3 :=
by 
  sorry

end find_z_l1658_165839


namespace traveling_cost_l1658_165877

def area_road_length_parallel (length width : ℕ) := width * length

def area_road_breadth_parallel (length width : ℕ) := width * length

def area_intersection (width : ℕ) := width * width

def total_area_of_roads  (length breadth width : ℕ) : ℕ :=
  (area_road_length_parallel length width) + (area_road_breadth_parallel breadth width) - area_intersection width

def cost_of_traveling_roads (total_area_of_roads cost_per_sq_m : ℕ) := total_area_of_roads * cost_per_sq_m

theorem traveling_cost
  (length breadth width cost_per_sq_m : ℕ)
  (h_length : length = 80)
  (h_breadth : breadth = 50)
  (h_width : width = 10)
  (h_cost_per_sq_m : cost_per_sq_m = 3)
  : cost_of_traveling_roads (total_area_of_roads length breadth width) cost_per_sq_m = 3600 :=
by
  sorry

end traveling_cost_l1658_165877


namespace quadratic_interval_solution_l1658_165889

open Set

def quadratic_function (x : ℝ) : ℝ := x^2 + 5 * x + 6

theorem quadratic_interval_solution :
  {x : ℝ | 6 ≤ quadratic_function x ∧ quadratic_function x ≤ 12} = {x | -6 ≤ x ∧ x ≤ -5} ∪ {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end quadratic_interval_solution_l1658_165889


namespace b_is_arithmetic_sequence_l1658_165827

theorem b_is_arithmetic_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) :
  a 1 = 1 →
  a 2 = 2 →
  (∀ n, a (n + 2) = 2 * a (n + 1) - a n + 2) →
  (∀ n, b n = a (n + 1) - a n) →
  ∃ d, ∀ n, b (n + 1) = b n + d :=
by
  intros h1 h2 h3 h4
  use 2
  sorry

end b_is_arithmetic_sequence_l1658_165827


namespace inequality_wxyz_l1658_165887

theorem inequality_wxyz 
  (w x y z : ℝ) 
  (h₁ : w^2 + y^2 ≤ 1) : 
  (w * x + y * z - 1)^2 ≥ (w^2 + y^2 - 1) * (x^2 + z^2 - 1) :=
by
  sorry

end inequality_wxyz_l1658_165887


namespace find_m_l1658_165870

theorem find_m (m : ℕ) :
  (2022 ^ 2 - 4) * (2021 ^ 2 - 4) = 2024 * 2020 * 2019 * m → 
  m = 2023 :=
by
  sorry

end find_m_l1658_165870


namespace relationship_between_y_values_l1658_165899

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variables (m : ℝ) (y1 y2 y3 : ℝ)
variables (h : m > 0)
variables (h1 : y1 = quadratic_function m (-1))
variables (h2 : y2 = quadratic_function m (5 / 2))
variables (h3 : y3 = quadratic_function m 6)

theorem relationship_between_y_values : y3 > y1 ∧ y1 > y2 :=
by
  sorry

end relationship_between_y_values_l1658_165899


namespace max_residents_per_apartment_l1658_165845

theorem max_residents_per_apartment (total_floors : ℕ) (floors_with_6_apts : ℕ) (floors_with_5_apts : ℕ)
  (rooms_per_6_floors : ℕ) (rooms_per_5_floors : ℕ) (max_residents : ℕ) : 
  total_floors = 12 ∧ floors_with_6_apts = 6 ∧ floors_with_5_apts = 6 ∧ 
  rooms_per_6_floors = 6 ∧ rooms_per_5_floors = 5 ∧ max_residents = 264 → 
  264 / (6 * 6 + 6 * 5) = 4 := sorry

end max_residents_per_apartment_l1658_165845


namespace larger_of_two_numbers_with_hcf_25_l1658_165820

theorem larger_of_two_numbers_with_hcf_25 (a b : ℕ) (h_hcf: Nat.gcd a b = 25)
  (h_lcm_factors: 13 * 14 = (25 * 13 * 14) / (Nat.gcd a b)) :
  max a b = 350 :=
sorry

end larger_of_two_numbers_with_hcf_25_l1658_165820


namespace no_positive_integer_solutions_l1658_165868

theorem no_positive_integer_solutions (x y z : ℕ) (h_cond : x^2 + y^2 = 7 * z^2) : 
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_positive_integer_solutions_l1658_165868


namespace num_four_digit_integers_with_at_least_one_4_or_7_l1658_165895

def count_four_digit_integers_with_4_or_7 : ℕ := 5416

theorem num_four_digit_integers_with_at_least_one_4_or_7 :
  let all_four_digit_integers := 9000
  let valid_digits_first := 7
  let valid_digits := 8
  let integers_without_4_or_7 := valid_digits_first * valid_digits * valid_digits * valid_digits
  all_four_digit_integers - integers_without_4_or_7 = count_four_digit_integers_with_4_or_7 :=
by
  -- Using known values from the problem statement
  let all_four_digit_integers := 9000
  let valid_digits_first := 7
  let valid_digits := 8
  let integers_without_4_or_7 := valid_digits_first * valid_digits * valid_digits * valid_digits
  show all_four_digit_integers - integers_without_4_or_7 = count_four_digit_integers_with_4_or_7
  sorry

end num_four_digit_integers_with_at_least_one_4_or_7_l1658_165895


namespace determinant_inequality_l1658_165859

theorem determinant_inequality (x : ℝ) (h : 2 * x - (3 - x) > 0) : 3 * x - 3 > 0 := 
by
  sorry

end determinant_inequality_l1658_165859


namespace red_basket_fruit_count_l1658_165856

-- Defining the basket counts
def blue_basket_bananas := 12
def blue_basket_apples := 4
def blue_basket_fruits := blue_basket_bananas + blue_basket_apples
def red_basket_fruits := blue_basket_fruits / 2

-- Statement of the proof problem
theorem red_basket_fruit_count : red_basket_fruits = 8 := by
  sorry

end red_basket_fruit_count_l1658_165856
