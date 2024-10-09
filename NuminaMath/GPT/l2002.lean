import Mathlib

namespace max_value_of_function_cos_sin_l2002_200281

noncomputable def max_value_function (x : ℝ) : ℝ := 
  (Real.cos x)^3 + (Real.sin x)^2 - Real.cos x

theorem max_value_of_function_cos_sin : 
  ∃ x ∈ (Set.univ : Set ℝ), max_value_function x = (32 / 27) := 
sorry

end max_value_of_function_cos_sin_l2002_200281


namespace largest_odd_digit_multiple_of_11_l2002_200276

theorem largest_odd_digit_multiple_of_11 (n : ℕ) (h1 : n < 10000) (h2 : ∀ d ∈ (n.digits 10), d % 2 = 1) (h3 : 11 ∣ n) : n ≤ 9559 :=
sorry

end largest_odd_digit_multiple_of_11_l2002_200276


namespace proof_problem_l2002_200222

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

theorem proof_problem :
  (∃ A ω φ, (A = 2) ∧ (ω = 2) ∧ (φ = Real.pi / 4) ∧
  f (3 * Real.pi / 8) = 0 ∧
  f (Real.pi / 8) = 2 ∧
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≤ 2) ∧
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≥ -Real.sqrt 2) ∧
  f (-Real.pi / 4) = -Real.sqrt 2) :=
sorry

end proof_problem_l2002_200222


namespace right_triangle_ineq_l2002_200255

-- Definitions based on conditions in (a)
variables {a b c m f : ℝ}
variable (h_a : a ≥ 0)
variable (h_b : b ≥ 0)
variable (h_c : c > 0)
variable (h_a_b : a ≤ b)
variable (h_triangle : c = Real.sqrt (a^2 + b^2))
variable (h_m : m = a * b / c)
variable (h_f : f = (Real.sqrt 2 * a * b) / (a + b))

-- Proof goal based on the problem in (c)
theorem right_triangle_ineq : m + f ≤ c :=
sorry

end right_triangle_ineq_l2002_200255


namespace xyz_inequality_l2002_200251

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z + x * y + y * z + z * x = 4) : x + y + z ≥ 3 := 
by
  sorry

end xyz_inequality_l2002_200251


namespace find_range_of_a_l2002_200237

noncomputable def range_of_a (a : ℝ) (n : ℕ) : Prop :=
  1 + 1 / (n : ℝ) ≤ a ∧ a < 1 + 1 / ((n - 1) : ℝ)

theorem find_range_of_a (a : ℝ) (n : ℕ) (h1 : 1 < a) (h2 : 2 ≤ n) :
  (∃ x : ℕ, ∀ x₀ < x, (⌊a * (x₀ : ℝ)⌋ : ℝ) = x₀) ↔ range_of_a a n := by
  sorry

end find_range_of_a_l2002_200237


namespace shortest_travel_time_to_sunny_town_l2002_200284

-- Definitions based on the given conditions
def highway_length : ℕ := 12

def railway_crossing_closed (t : ℕ) : Prop :=
  ∃ k : ℕ, t = 6 * k + 0 ∨ t = 6 * k + 1 ∨ t = 6 * k + 2

def traffic_light_red (t : ℕ) : Prop :=
  ∃ k1 : ℕ, t = 5 * k1 + 0 ∨ t = 5 * k1 + 1

def initial_conditions (t : ℕ) : Prop :=
  railway_crossing_closed 0 ∧ traffic_light_red 0

def shortest_time_to_sunny_town (time : ℕ) : Prop := 
  time = 24

-- The proof statement
theorem shortest_travel_time_to_sunny_town :
  ∃ time : ℕ, shortest_time_to_sunny_town time ∧
  (∀ t : ℕ, 0 ≤ t → t ≤ time → ¬railway_crossing_closed t ∧ ¬traffic_light_red t) :=
sorry

end shortest_travel_time_to_sunny_town_l2002_200284


namespace gift_arrangement_l2002_200268

theorem gift_arrangement (n k : ℕ) (h_n : n = 5) (h_k : k = 4) : 
  (n * Nat.factorial k) = 120 :=
by
  sorry

end gift_arrangement_l2002_200268


namespace problem1_problem2_problem3_problem4_l2002_200249

-- Problem (1)
theorem problem1 : 6 - -2 + -4 - 3 = 1 :=
by sorry

-- Problem (2)
theorem problem2 : 8 / -2 * (1 / 3 : ℝ) * (-(1 + 1/2: ℝ)) = 2 :=
by sorry

-- Problem (3)
theorem problem3 : (13 + (2 / 7 - 1 / 14) * 56) / (-1 / 4) = -100 :=
by sorry

-- Problem (4)
theorem problem4 : 
  |-(5 / 6 : ℝ)| / ((-(3 + 1 / 5: ℝ)) / (-4)^2 + (-7 / 4) * (4 / 7)) = -(25 / 36) :=
by sorry

end problem1_problem2_problem3_problem4_l2002_200249


namespace park_attraction_children_count_l2002_200243

theorem park_attraction_children_count
  (C : ℕ) -- Number of children
  (entrance_fee : ℕ := 5) -- Entrance fee per person
  (kids_attr_fee : ℕ := 2) -- Attraction fee for kids
  (adults_attr_fee : ℕ := 4) -- Attraction fee for adults
  (parents : ℕ := 2) -- Number of parents
  (grandmother : ℕ := 1) -- Number of grandmothers
  (total_cost : ℕ := 55) -- Total cost paid
  (entry_eq : entrance_fee * (C + parents + grandmother) + kids_attr_fee * C + adults_attr_fee * (parents + grandmother) = total_cost) :
  C = 4 :=
by
  sorry

end park_attraction_children_count_l2002_200243


namespace marcy_sip_amount_l2002_200278

theorem marcy_sip_amount (liters : ℕ) (ml_per_liter : ℕ) (total_minutes : ℕ) (interval_minutes : ℕ) (total_ml : ℕ) (total_sips : ℕ) (ml_per_sip : ℕ) 
  (h1 : liters = 2) 
  (h2 : ml_per_liter = 1000)
  (h3 : total_minutes = 250) 
  (h4 : interval_minutes = 5)
  (h5 : total_ml = liters * ml_per_liter)
  (h6 : total_sips = total_minutes / interval_minutes)
  (h7 : ml_per_sip = total_ml / total_sips) : 
  ml_per_sip = 40 := 
by
  sorry

end marcy_sip_amount_l2002_200278


namespace Lakeview_High_School_Basketball_Team_l2002_200266

theorem Lakeview_High_School_Basketball_Team :
  ∀ (total_players taking_physics taking_both statistics: ℕ),
  total_players = 25 →
  taking_physics = 10 →
  taking_both = 5 →
  statistics = 20 :=
sorry

end Lakeview_High_School_Basketball_Team_l2002_200266


namespace hall_length_width_difference_l2002_200246

theorem hall_length_width_difference :
  ∃ (L W : ℝ), 
  (W = (1 / 2) * L) ∧
  (L * W = 288) ∧
  (L - W = 12) :=
by
  -- The mathematical proof follows from the conditions given
  sorry

end hall_length_width_difference_l2002_200246


namespace algebraic_expression_value_l2002_200285

theorem algebraic_expression_value (x y : ℝ) (h : x^4 + 6*x^2*y + 9*y^2 + 2*x^2 + 6*y + 4 = 7) :
(x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = -2) ∨ (x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = 14) :=
sorry

end algebraic_expression_value_l2002_200285


namespace max_banner_area_l2002_200208

theorem max_banner_area (x y : ℕ) (cost_constraint : 330 * x + 450 * y ≤ 10000) : x * y ≤ 165 :=
by
  sorry

end max_banner_area_l2002_200208


namespace sequence_property_l2002_200273

theorem sequence_property : 
  (∀ (a : ℕ → ℝ), a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + (2 * a n) / n) → a 200 = 40200) :=
by
  sorry

end sequence_property_l2002_200273


namespace inequality_one_inequality_two_l2002_200212

variable (a b c : ℝ)

-- Conditions given in the problem
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom positive_c : 0 < c
axiom sum_eq_one : a + b + c = 1

-- Statements to prove
theorem inequality_one : ab + bc + ac ≤ 1 / 3 :=
sorry

theorem inequality_two : a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
sorry

end inequality_one_inequality_two_l2002_200212


namespace ca_co3_to_ca_cl2_l2002_200258

theorem ca_co3_to_ca_cl2 (caCO3 HCl : ℕ) (main_reaction : caCO3 = 1 ∧ HCl = 2) : ∃ CaCl2, CaCl2 = 1 :=
by
  -- The proof of the theorem will go here.
  sorry

end ca_co3_to_ca_cl2_l2002_200258


namespace ellipse_equation_y_intercept_range_l2002_200221

noncomputable def a := 2 * Real.sqrt 2
noncomputable def b := Real.sqrt 2
noncomputable def e := Real.sqrt 3 / 2
noncomputable def c := Real.sqrt 6
def M : ℝ × ℝ := (2, 1)

-- Condition: The ellipse equation form
def ellipse (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Question 1: Proof that the ellipse equation is as given
theorem ellipse_equation :
  ellipse x y ↔ (x^2) / 8 + (y^2) / 2 = 1 := sorry

-- Condition: Line l is parallel to OM
def slope_OM := 1 / 2
def line_l (m x y : ℝ) : Prop := y = slope_OM * x + m

-- Question 2: Proof of the range for y-intercept m given the conditions
theorem y_intercept_range (m : ℝ) :
  (-Real.sqrt 2 < m ∧ m < 0 ∨ 0 < m ∧ m < Real.sqrt 2) ↔
  ∃ x1 y1 x2 y2,
    line_l m x1 y1 ∧ 
    line_l m x2 y2 ∧ 
    x1 ≠ x2 ∧ 
    y1 ≠ y2 ∧
    x1 * x2 + y1 * y2 < 0 := sorry

end ellipse_equation_y_intercept_range_l2002_200221


namespace answered_both_correctly_l2002_200235

variable (A B : Prop)
variable (P_A P_B P_not_A_and_not_B P_A_and_B : ℝ)

axiom P_A_eq : P_A = 0.75
axiom P_B_eq : P_B = 0.35
axiom P_not_A_and_not_B_eq : P_not_A_and_not_B = 0.20

theorem answered_both_correctly (h1 : P_A = 0.75) (h2 : P_B = 0.35) (h3 : P_not_A_and_not_B = 0.20) : 
  P_A_and_B = 0.30 :=
by
  sorry

end answered_both_correctly_l2002_200235


namespace arithmetic_expression_value_l2002_200231

theorem arithmetic_expression_value :
  2 - (-3) * 2 - 4 - (-5) * 2 - 6 = 8 :=
by {
  sorry
}

end arithmetic_expression_value_l2002_200231


namespace unique_solution_quadratic_eq_l2002_200291

theorem unique_solution_quadratic_eq (p : ℝ) (h_nonzero : p ≠ 0) : (∀ x : ℝ, p * x^2 - 20 * x + 4 = 0) → p = 25 :=
by
  sorry

end unique_solution_quadratic_eq_l2002_200291


namespace max_omega_is_2_l2002_200211

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem max_omega_is_2 {ω : ℝ} (h₀ : ω > 0) (h₁ : MonotoneOn (f ω) (Set.Icc (-Real.pi / 6) (Real.pi / 6))) :
  ω ≤ 2 :=
sorry

end max_omega_is_2_l2002_200211


namespace smallest_dividend_l2002_200293

   theorem smallest_dividend (b a : ℤ) (q : ℤ := 12) (r : ℤ := 3) (h : a = b * q + r) (h' : r < b) : a = 51 :=
   by
     sorry
   
end smallest_dividend_l2002_200293


namespace sum_of_fourth_powers_l2002_200250

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 18.5 :=
sorry

end sum_of_fourth_powers_l2002_200250


namespace divisibility_by_3_l2002_200280

theorem divisibility_by_3 (a b c : ℤ) (h1 : c ≠ b)
    (h2 : ∃ x : ℂ, (a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0)) :
    3 ∣ (a + b + 2 * c) :=
by
  sorry

end divisibility_by_3_l2002_200280


namespace max_area_of_rectangle_with_perimeter_40_l2002_200247

theorem max_area_of_rectangle_with_perimeter_40 :
  ∃ (A : ℝ), (A = 100) ∧ (∀ (length width : ℝ), 2 * (length + width) = 40 → length * width ≤ A) :=
by
  sorry

end max_area_of_rectangle_with_perimeter_40_l2002_200247


namespace gross_profit_value_l2002_200239

theorem gross_profit_value (sales_price : ℝ) (cost : ℝ) (gross_profit : ℝ) 
    (h1 : sales_price = 54) 
    (h2 : gross_profit = 1.25 * cost) 
    (h3 : sales_price = cost + gross_profit): gross_profit = 30 := 
  sorry

end gross_profit_value_l2002_200239


namespace evaluate_f_l2002_200216

def f (n : ℕ) : ℕ :=
  if n < 4 then n^2 - 1 else 3*n - 2

theorem evaluate_f (h : f (f (f 2)) = 22) : f (f (f 2)) = 22 :=
by
  -- we state the final result directly
  sorry

end evaluate_f_l2002_200216


namespace period_and_symmetry_of_function_l2002_200238

-- Given conditions
variables (f : ℝ → ℝ)
variable (hf_odd : ∀ x, f (-x) = -f x)
variable (hf_cond : ∀ x, f (-2 * x + 4) = -f (2 * x))
variable (hf_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 ^ x - 1)

-- Prove that 4 is a period and x=1 is a line of symmetry for the graph of f(x)
theorem period_and_symmetry_of_function :
  (∀ x, f (x + 4) = f x) ∧ (∀ x, f (x) + f (4 - x) = 0) :=
by sorry

end period_and_symmetry_of_function_l2002_200238


namespace reliability_is_correct_l2002_200262

-- Define the probabilities of each switch functioning properly.
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.7

-- Define the system reliability.
def reliability : ℝ := P_A * P_B * P_C

-- The theorem stating the reliability of the system.
theorem reliability_is_correct : reliability = 0.504 := by
  sorry

end reliability_is_correct_l2002_200262


namespace radius_of_cylinder_l2002_200288

-- Define the main parameters and conditions
def diameter_cone := 8
def radius_cone := diameter_cone / 2
def altitude_cone := 10
def height_cylinder (r : ℝ) := 2 * r

-- Assume similarity of triangles
theorem radius_of_cylinder (r : ℝ) (h_c := height_cylinder r) :
  altitude_cone - h_c / r = altitude_cone / radius_cone → r = 20 / 9 := 
by
  intro h
  sorry

end radius_of_cylinder_l2002_200288


namespace contractor_engaged_days_l2002_200232

theorem contractor_engaged_days (x y : ℕ) (earnings_per_day : ℕ) (fine_per_day : ℝ) 
    (total_earnings : ℝ) (absent_days : ℕ) 
    (h1 : earnings_per_day = 25) 
    (h2 : fine_per_day = 7.50) 
    (h3 : total_earnings = 555) 
    (h4 : absent_days = 6) 
    (h5 : total_earnings = (earnings_per_day * x : ℝ) - fine_per_day * y) 
    (h6 : y = absent_days) : 
    x = 24 := 
by
  sorry

end contractor_engaged_days_l2002_200232


namespace Harriet_sibling_product_l2002_200207

-- Definition of the family structure
def Harry : Prop := 
  let sisters := 4
  let brothers := 4
  true

-- Harriet being one of Harry's sisters and calculating her siblings
def Harriet : Prop :=
  let S := 4 - 1 -- Number of Harriet's sisters
  let B := 4 -- Number of Harriet's brothers
  S * B = 12

theorem Harriet_sibling_product : Harry → Harriet := by
  intro h
  let S := 3
  let B := 4
  have : S * B = 12 := by norm_num
  exact this

end Harriet_sibling_product_l2002_200207


namespace angle_between_AD_and_BC_l2002_200271

variables {a b c : ℝ} 
variables {θ : ℝ}
variables {α β γ δ ε ζ : ℝ} -- representing the angles

-- Conditions of the problem
def conditions (a b c : ℝ) (α β γ δ ε ζ : ℝ) : Prop :=
  (α + β + γ = 180) ∧ (δ + ε + ζ = 180) ∧ 
  (a > 0) ∧ (b > 0) ∧ (c > 0)

-- Definition of the theorem to prove the angle between AD and BC
theorem angle_between_AD_and_BC
  (a b c : ℝ) (α β γ δ ε ζ : ℝ)
  (h : conditions a b c α β γ δ ε ζ) :
  θ = Real.arccos ((|b^2 - c^2|) / a^2) :=
sorry

end angle_between_AD_and_BC_l2002_200271


namespace maple_trees_cut_down_l2002_200203

-- Define the initial number of maple trees.
def initial_maple_trees : ℝ := 9.0

-- Define the final number of maple trees after cutting.
def final_maple_trees : ℝ := 7.0

-- Define the number of maple trees cut down.
def cut_down_maple_trees : ℝ := initial_maple_trees - final_maple_trees

-- Prove that the number of cut down maple trees is 2.
theorem maple_trees_cut_down : cut_down_maple_trees = 2 := by
  sorry

end maple_trees_cut_down_l2002_200203


namespace mutually_exclusive_not_contradictory_l2002_200229

namespace BallProbability
  -- Definitions of events based on the conditions
  def at_least_two_white (outcome : Multiset (String)) : Prop := 
    Multiset.count "white" outcome ≥ 2

  def all_red (outcome : Multiset (String)) : Prop := 
    Multiset.count "red" outcome = 3

  -- Problem statement
  theorem mutually_exclusive_not_contradictory :
    ∀ outcome : Multiset (String),
    Multiset.card outcome = 3 →
    (at_least_two_white outcome → ¬all_red outcome) ∧
    ¬(∀ outcome, at_least_two_white outcome ↔ ¬all_red outcome) := 
  by
    intros
    sorry
end BallProbability

end mutually_exclusive_not_contradictory_l2002_200229


namespace sum_of_roots_l2002_200265

def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def condition (a b c : ℝ) (x : ℝ) :=
  quadratic_polynomial a b c (x^3 + x) ≥ quadratic_polynomial a b c (x^2 + 1)

theorem sum_of_roots (a b c : ℝ) (h : ∀ x : ℝ, condition a b c x) :
  b = -4 * a → -(b / a) = 4 :=
by
  sorry

end sum_of_roots_l2002_200265


namespace multiple_of_960_l2002_200244

theorem multiple_of_960 (a : ℤ) (h1 : a % 10 = 4) (h2 : ¬ (a % 4 = 0)) :
  ∃ k : ℤ, a * (a^2 - 1) * (a^2 - 4) = 960 * k :=
  sorry

end multiple_of_960_l2002_200244


namespace conditions_for_right_triangle_l2002_200282

universe u

variables {A B C : Type u}
variables [OrderedAddCommGroup A] [OrderedAddCommGroup B] [OrderedAddCommGroup C]

noncomputable def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem conditions_for_right_triangle :
  (∀ (A B C : ℝ), A + B = C → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), ( A / C = 1 / 6 ) → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), A = 90 - B → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), (A = B → B = C / 2) → is_right_triangle A B C) ∧
  ∀ (A B C : ℝ), ¬ ((A = 2 * B) ∧ B = 3 * C) 
:=
sorry

end conditions_for_right_triangle_l2002_200282


namespace red_balls_in_bag_l2002_200230

/-- Given the conditions of the ball distribution in the bag,
we need to prove the number of red balls is 9. -/
theorem red_balls_in_bag (total_balls white_balls green_balls yellow_balls purple_balls : ℕ)
  (prob_neither_red_nor_purple : ℝ) (h_total : total_balls = 100)
  (h_white : white_balls = 50) (h_green : green_balls = 30)
  (h_yellow : yellow_balls = 8) (h_purple : purple_balls = 3)
  (h_prob : prob_neither_red_nor_purple = 0.88) :
  ∃ R : ℕ, (total_balls = white_balls + green_balls + yellow_balls + purple_balls + R) ∧ R = 9 :=
by {
  sorry
}

end red_balls_in_bag_l2002_200230


namespace positive_real_solution_eq_l2002_200245

theorem positive_real_solution_eq :
  ∃ x : ℝ, 0 < x ∧ ( (1/4) * (5 * x^2 - 4) = (x^2 - 40 * x - 5) * (x^2 + 20 * x + 2) ) ∧ x = 20 + 10 * Real.sqrt 41 :=
by
  sorry

end positive_real_solution_eq_l2002_200245


namespace sum_quotient_dividend_divisor_l2002_200272

theorem sum_quotient_dividend_divisor (N : ℕ) (divisor : ℕ) (quotient : ℕ) (sum : ℕ) 
    (h₁ : N = 40) (h₂ : divisor = 2) (h₃ : quotient = N / divisor)
    (h₄ : sum = quotient + N + divisor) : sum = 62 := by
  -- proof goes here
  sorry

end sum_quotient_dividend_divisor_l2002_200272


namespace fraction_comparison_l2002_200205

theorem fraction_comparison (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a / b) > (a + 1) / (b + 1) :=
by
  sorry

end fraction_comparison_l2002_200205


namespace minimize_expression_l2002_200240

variables {x y : ℝ}

theorem minimize_expression : ∃ (x y : ℝ), 2 * x^2 + 2 * x * y + y^2 - 2 * x - 1 = -2 :=
by sorry

end minimize_expression_l2002_200240


namespace part_I_part_II_l2002_200201

noncomputable def f (x : ℝ) (a : ℝ) := x - (2 * a - 1) / x - 2 * a * Real.log x

theorem part_I (a : ℝ) (h : a = 3 / 2) : 
  (∀ x, 0 < x ∧ x < 1 → f x a < 0) ∧ (∀ x, 1 < x ∧ x < 2 → f x a > 0) ∧ (∀ x, 2 < x → f x a < 0) := sorry

theorem part_II (a : ℝ) : (∀ x, 1 ≤ x → f x a ≥ 0) → a ≤ 1 := sorry

end part_I_part_II_l2002_200201


namespace minimum_value_ineq_l2002_200277

variable {a b c : ℝ}

theorem minimum_value_ineq (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2 * a + 1) * (b^2 + 2 * b + 1) * (c^2 + 2 * c + 1) / (a * b * c) ≥ 64 :=
sorry

end minimum_value_ineq_l2002_200277


namespace largest_integer_less_than_120_with_remainder_5_div_8_l2002_200269

theorem largest_integer_less_than_120_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 120 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 120 → m % 8 = 5 → m ≤ n :=
sorry

end largest_integer_less_than_120_with_remainder_5_div_8_l2002_200269


namespace alice_leaves_30_minutes_after_bob_l2002_200296

theorem alice_leaves_30_minutes_after_bob :
  ∀ (distance : ℝ) (speed_bob : ℝ) (speed_alice : ℝ) (time_diff : ℝ),
  distance = 220 ∧ speed_bob = 40 ∧ speed_alice = 44 ∧ 
  time_diff = (distance / speed_bob) - (distance / speed_alice) →
  (time_diff * 60 = 30) := by
  intro distance speed_bob speed_alice time_diff
  intro h
  have h1 : distance = 220 := h.1
  have h2 : speed_bob = 40 := h.2.1
  have h3 : speed_alice = 44 := h.2.2.1
  have h4 : time_diff = (distance / speed_bob) - (distance / speed_alice) := h.2.2.2
  sorry

end alice_leaves_30_minutes_after_bob_l2002_200296


namespace find_x_l2002_200218

theorem find_x (x y : ℝ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
by
  sorry

end find_x_l2002_200218


namespace brother_age_in_5_years_l2002_200206

noncomputable def Nick : ℕ := 13
noncomputable def Sister : ℕ := Nick + 6
noncomputable def CombinedAge : ℕ := Nick + Sister
noncomputable def Brother : ℕ := CombinedAge / 2

theorem brother_age_in_5_years : Brother + 5 = 21 := by
  sorry

end brother_age_in_5_years_l2002_200206


namespace number_of_long_sleeved_jerseys_l2002_200227

def cost_per_long_sleeved := 15
def cost_per_striped := 10
def num_striped_jerseys := 2
def total_spent := 80

theorem number_of_long_sleeved_jerseys (x : ℕ) :
  total_spent = cost_per_long_sleeved * x + cost_per_striped * num_striped_jerseys →
  x = 4 := by
  sorry

end number_of_long_sleeved_jerseys_l2002_200227


namespace committee_formation_l2002_200290

theorem committee_formation :
  let club_size := 15
  let num_roles := 2
  let num_members := 3
  let total_ways := (15 * 14) * Nat.choose (15 - num_roles) num_members
  total_ways = 60060 := by
    let club_size := 15
    let num_roles := 2
    let num_members := 3
    let total_ways := (15 * 14) * Nat.choose (15 - num_roles) num_members
    show total_ways = 60060
    sorry

end committee_formation_l2002_200290


namespace mandy_toys_count_l2002_200215

theorem mandy_toys_count (M A Am P : ℕ) 
    (h1 : A = 3 * M) 
    (h2 : A = Am - 2) 
    (h3 : A = P / 2) 
    (h4 : M + A + Am + P = 278) : 
    M = 21 := 
by
  sorry

end mandy_toys_count_l2002_200215


namespace lana_total_pages_l2002_200233

theorem lana_total_pages (lana_initial_pages : ℕ) (duane_total_pages : ℕ) :
  lana_initial_pages = 8 ∧ duane_total_pages = 42 →
  (lana_initial_pages + duane_total_pages / 2) = 29 :=
by
  sorry

end lana_total_pages_l2002_200233


namespace arithmetic_sequence_terms_l2002_200295

variable (n : ℕ)
variable (sumOdd sumEven : ℕ)
variable (terms : ℕ)

theorem arithmetic_sequence_terms
  (h1 : sumOdd = 120)
  (h2 : sumEven = 110)
  (h3 : terms = 2 * n + 1)
  (h4 : sumOdd + sumEven = 230) :
  terms = 23 := 
sorry

end arithmetic_sequence_terms_l2002_200295


namespace exists_f_ff_eq_square_l2002_200226

open Nat

theorem exists_f_ff_eq_square : ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n ^ 2 :=
by
  -- proof to be provided
  sorry

end exists_f_ff_eq_square_l2002_200226


namespace concentration_of_acid_in_third_flask_is_correct_l2002_200204

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l2002_200204


namespace part_i_part_ii_l2002_200298

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + x - a
noncomputable def g (x a : ℝ) : ℝ := Real.sqrt (f x a)

theorem part_i (a : ℝ) :
  (∀ x ∈ Set.Icc (0:ℝ) (1:ℝ), f x a ≥ 0) ↔ (a ≤ 1) :=
by {
  -- Suppose it is already known that theorem is true.
  sorry
}

theorem part_ii (a : ℝ) :
  (∃ x0 y0 : ℝ, (x0, y0) ∈ (Set.Icc (-1) 1) ∧ y0 = Real.cos (2 * x0) ∧ g (g y0 a) a = y0) ↔ (1 ≤ a ∧ a ≤ Real.exp 1) :=
by {
  -- Suppose it is already known that theorem is true.
  sorry
}

end part_i_part_ii_l2002_200298


namespace rowing_time_ratio_l2002_200219

def V_b : ℕ := 57
def V_s : ℕ := 19
def V_up : ℕ := V_b - V_s
def V_down : ℕ := V_b + V_s

theorem rowing_time_ratio :
  ∀ (T_up T_down : ℕ), V_up * T_up = V_down * T_down → T_up = 2 * T_down :=
by
  intros T_up T_down h
  sorry

end rowing_time_ratio_l2002_200219


namespace evaluate_expression_l2002_200256

theorem evaluate_expression : 
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 :=
by
  sorry

end evaluate_expression_l2002_200256


namespace range_of_z_l2002_200289

theorem range_of_z (x y : ℝ) 
  (h1 : x + 2 ≥ y) 
  (h2 : x + 2 * y ≥ 4) 
  (h3 : y ≤ 5 - 2 * x) : 
  ∃ (z_min z_max : ℝ), 
    (z_min = 1) ∧ 
    (z_max = 2) ∧ 
    (∀ z, z = (2 * x + y - 1) / (x + 1) → z_min ≤ z ∧ z ≤ z_max) :=
by
  sorry

end range_of_z_l2002_200289


namespace part1_part2_l2002_200292

variable {α : Type*}
def A : Set ℝ := {x | 0 < x ∧ x < 9}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part (1)
theorem part1 : B 5 ∩ A = {x | 6 ≤ x ∧ x < 9} := 
sorry

-- Part (2)
theorem part2 (m : ℝ): A ∩ B m = B m ↔ m < 5 :=
sorry

end part1_part2_l2002_200292


namespace least_number_with_remainder_l2002_200257

theorem least_number_with_remainder (n : ℕ) (d₁ d₂ d₃ d₄ r : ℕ) 
  (h₁ : d₁ = 5) (h₂ : d₂ = 6) (h₃ : d₃ = 9) (h₄ : d₄ = 12) (hr : r = 184) :
  (∀ d, d ∈ [d₁, d₂, d₃, d₄] → n % d = r % d) → n = 364 := 
sorry

end least_number_with_remainder_l2002_200257


namespace at_least_one_not_less_than_l2002_200294

variables {A B C D a b c : ℝ}

theorem at_least_one_not_less_than :
  (a = A * C) →
  (b = A * D + B * C) →
  (c = B * D) →
  (a + b + c = (A + B) * (C + D)) →
  a ≥ (4 * (A + B) * (C + D) / 9) ∨ b ≥ (4 * (A + B) * (C + D) / 9) ∨ c ≥ (4 * (A + B) * (C + D) / 9) :=
by
  intro h1 h2 h3 h4
  sorry

end at_least_one_not_less_than_l2002_200294


namespace ratio_of_james_to_jacob_l2002_200202

noncomputable def MarkJumpHeight : ℕ := 6
noncomputable def LisaJumpHeight : ℕ := 2 * MarkJumpHeight
noncomputable def JacobJumpHeight : ℕ := 2 * LisaJumpHeight
noncomputable def JamesJumpHeight : ℕ := 16

theorem ratio_of_james_to_jacob : (JamesJumpHeight : ℚ) / (JacobJumpHeight : ℚ) = 2 / 3 :=
by
  sorry

end ratio_of_james_to_jacob_l2002_200202


namespace radius_of_spheres_in_cone_l2002_200286

def base_radius := 8
def cone_height := 15
def num_spheres := 3
def spheres_are_tangent := true

theorem radius_of_spheres_in_cone :
  ∃ (r : ℝ), r = (280 - 100 * Real.sqrt 3) / 121 :=
sorry

end radius_of_spheres_in_cone_l2002_200286


namespace green_blue_tile_difference_is_15_l2002_200260

def initial_blue_tiles : Nat := 13
def initial_green_tiles : Nat := 6
def second_blue_tiles : Nat := 2 * initial_blue_tiles
def second_green_tiles : Nat := 2 * initial_green_tiles
def border_green_tiles : Nat := 36
def total_blue_tiles : Nat := initial_blue_tiles + second_blue_tiles
def total_green_tiles : Nat := initial_green_tiles + second_green_tiles + border_green_tiles
def tile_difference : Nat := total_green_tiles - total_blue_tiles

theorem green_blue_tile_difference_is_15 : tile_difference = 15 := by
  sorry

end green_blue_tile_difference_is_15_l2002_200260


namespace quotient_of_larger_divided_by_smaller_l2002_200248

theorem quotient_of_larger_divided_by_smaller
  (x y : ℕ)
  (h1 : x * y = 9375)
  (h2 : x + y = 400)
  (h3 : x > y) :
  x / y = 15 :=
sorry

end quotient_of_larger_divided_by_smaller_l2002_200248


namespace math_problem_l2002_200299

theorem math_problem (x y z : ℤ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : (x + y * Complex.I)^2 - 46 * Complex.I = z) :
  x + y + z = 552 :=
by
  sorry

end math_problem_l2002_200299


namespace sequence_sum_problem_l2002_200210

theorem sequence_sum_problem :
  let seq := [72, 76, 80, 84, 88, 92, 96, 100, 104, 108]
  3 * (seq.sum) = 2700 :=
by
  sorry

end sequence_sum_problem_l2002_200210


namespace value_of_a_minus_2_b_minus_2_l2002_200200

theorem value_of_a_minus_2_b_minus_2 :
  ∀ (a b : ℝ), (a + b = -4/3 ∧ a * b = -7/3) → ((a - 2) * (b - 2) = 0) := by
  sorry

end value_of_a_minus_2_b_minus_2_l2002_200200


namespace dot_product_is_six_l2002_200252

def a : ℝ × ℝ := (-2, 4)
def b : ℝ × ℝ := (1, 2)

theorem dot_product_is_six : (a.1 * b.1 + a.2 * b.2) = 6 := 
by 
  -- definition and proof logic follows
  sorry

end dot_product_is_six_l2002_200252


namespace intersection_M_N_l2002_200261

def M : Set ℕ := {3, 5, 6, 8}
def N : Set ℕ := {4, 5, 7, 8}

theorem intersection_M_N : M ∩ N = {5, 8} :=
  sorry

end intersection_M_N_l2002_200261


namespace general_term_formula_l2002_200209

theorem general_term_formula (f : ℕ → ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ x, f x = 1 - 2^x) →
  (∀ n, f n = S n) →
  (∀ n, S n = 1 - 2^n) →
  (∀ n, n = 1 → a n = S 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (∀ n, a n = -2^(n-1)) :=
by
  sorry

end general_term_formula_l2002_200209


namespace triangle_base_length_l2002_200214

theorem triangle_base_length (A h b : ℝ) 
  (h1 : A = 30) 
  (h2 : h = 5) 
  (h3 : A = (b * h) / 2) : 
  b = 12 :=
by
  sorry

end triangle_base_length_l2002_200214


namespace original_recipe_serves_7_l2002_200283

theorem original_recipe_serves_7 (x : ℕ)
  (h1 : 2 / x = 10 / 35) :
  x = 7 := by
  sorry

end original_recipe_serves_7_l2002_200283


namespace marie_erasers_l2002_200228

-- Define the initial conditions
def initial_erasers : ℝ := 95.0
def additional_erasers : ℝ := 42.0

-- Define the target final erasers count
def final_erasers : ℝ := 137.0

-- The theorem we need to prove
theorem marie_erasers :
  initial_erasers + additional_erasers = final_erasers := by
  sorry

end marie_erasers_l2002_200228


namespace initial_volume_salt_solution_l2002_200213

theorem initial_volume_salt_solution (V : ℝ) (V1 : ℝ) (V2 : ℝ) : 
  V1 = 0.20 * V → 
  V2 = 30 →
  V1 = 0.15 * (V + V2) →
  V = 90 := 
by 
  sorry

end initial_volume_salt_solution_l2002_200213


namespace problem1_problem2_l2002_200234

-- Define the required conditions
variables {a b : ℤ}
-- Conditions
axiom h1 : a ≥ 1
axiom h2 : b ≥ 1

-- Proof statement for question 1
theorem problem1 : ¬ (a ∣ b^2 ↔ a ∣ b) := by
  sorry

-- Proof statement for question 2
theorem problem2 : (a^2 ∣ b^2 ↔ a ∣ b) := by
  sorry

end problem1_problem2_l2002_200234


namespace house_total_volume_l2002_200259

def room_volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

def bathroom_volume := room_volume 4 2 7
def bedroom_volume := room_volume 12 10 8
def livingroom_volume := room_volume 15 12 9

def total_volume := bathroom_volume + bedroom_volume + livingroom_volume

theorem house_total_volume : total_volume = 2636 := by
  sorry

end house_total_volume_l2002_200259


namespace a3_5a6_value_l2002_200270

variable {a : ℕ → ℤ}

-- Conditions: The sequence {a_n} is an arithmetic sequence, and a_4 + a_7 = 19
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

axiom a_seq_arithmetic : is_arithmetic_sequence a
axiom a4_a7_sum : a 4 + a 7 = 19

-- Problem statement: Prove that a_3 + 5a_6 = 57
theorem a3_5a6_value : a 3 + 5 * a 6 = 57 :=
by
  -- Proof goes here
  sorry

end a3_5a6_value_l2002_200270


namespace johns_last_month_savings_l2002_200224

theorem johns_last_month_savings (earnings rent dishwasher left_over : ℝ) 
  (h1 : rent = 0.40 * earnings) 
  (h2 : dishwasher = 0.70 * rent) 
  (h3 : left_over = earnings - rent - dishwasher) :
  left_over = 0.32 * earnings :=
by 
  sorry

end johns_last_month_savings_l2002_200224


namespace find_t_l2002_200264

theorem find_t (t : ℚ) : 
  ((t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 1) + 5) → t = 5 / 3 :=
by
  intro h
  sorry

end find_t_l2002_200264


namespace proof_of_greatest_sum_quotient_remainder_l2002_200236

def greatest_sum_quotient_remainder : Prop :=
  ∃ q r : ℕ, 1051 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q + r = 61

theorem proof_of_greatest_sum_quotient_remainder : greatest_sum_quotient_remainder := 
sorry

end proof_of_greatest_sum_quotient_remainder_l2002_200236


namespace hide_and_seek_l2002_200267

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l2002_200267


namespace seq_50th_term_eq_327_l2002_200297

theorem seq_50th_term_eq_327 : 
  let n := 50
  let binary_representation : List Nat := [1, 1, 0, 0, 1, 0] -- 50 in binary
  let powers_of_3 := [5, 4, 1] -- Positions of 1s in the binary representation 
  let term := List.sum (powers_of_3.map (λ k => 3^k))
  term = 327 := by
  sorry

end seq_50th_term_eq_327_l2002_200297


namespace factorize_expr1_factorize_expr2_l2002_200287

theorem factorize_expr1 (x y : ℝ) : 
  3 * (x + y) * (x - y) - (x - y)^2 = 2 * (x - y) * (x + 2 * y) :=
by
  sorry

theorem factorize_expr2 (x y : ℝ) : 
  x^2 * (y^2 - 1) + 2 * x * (y^2 - 1) = x * (y + 1) * (y - 1) * (x + 2) :=
by
  sorry

end factorize_expr1_factorize_expr2_l2002_200287


namespace find_a_plus_b_l2002_200274

theorem find_a_plus_b (a b : ℝ) 
  (h_a : a^3 - 3 * a^2 + 5 * a = 1) 
  (h_b : b^3 - 3 * b^2 + 5 * b = 5) : 
  a + b = 2 := 
sorry

end find_a_plus_b_l2002_200274


namespace books_received_l2002_200242

theorem books_received (initial_books : ℕ) (total_books : ℕ) (h1 : initial_books = 54) (h2 : total_books = 77) : (total_books - initial_books) = 23 :=
by
  sorry

end books_received_l2002_200242


namespace evaluate_expr_l2002_200241

theorem evaluate_expr : 3 * (3 * (3 * (3 * (3 * (3 * 2 * 2) * 2) * 2) * 2) * 2) * 2 = 1458 := by
  sorry

end evaluate_expr_l2002_200241


namespace eval_expression_l2002_200279

theorem eval_expression :
  (2^2003 * 3^2005) / (6^2004) = 3 / 2 := by
  sorry

end eval_expression_l2002_200279


namespace sequence_property_l2002_200263

theorem sequence_property (a : ℕ+ → ℚ)
  (h1 : ∀ p q : ℕ+, a p + a q = a (p + q))
  (h2 : a 1 = 1 / 9) :
  a 36 = 4 :=
sorry

end sequence_property_l2002_200263


namespace sum_of_roots_of_quadratic_eq_l2002_200275

theorem sum_of_roots_of_quadratic_eq (x : ℝ) :
  (x + 3) * (x - 4) = 18 → (∃ a b : ℝ, x ^ 2 + a * x + b = 0) ∧ (a = -1) ∧ (b = -30) :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l2002_200275


namespace root_diff_condition_l2002_200254

noncomputable def g (x : ℝ) : ℝ := 4^x + 2*x - 2
noncomputable def f (x : ℝ) : ℝ := 4*x - 1

theorem root_diff_condition :
  ∃ x₀, g x₀ = 0 ∧ |x₀ - 1/4| ≤ 1/4 ∧ ∃ y₀, f y₀ = 0 ∧ |y₀ - x₀| ≤ 0.25 :=
sorry

end root_diff_condition_l2002_200254


namespace fraction_habitable_surface_l2002_200217

def fraction_exposed_land : ℚ := 3 / 8
def fraction_inhabitable_land : ℚ := 2 / 3

theorem fraction_habitable_surface :
  fraction_exposed_land * fraction_inhabitable_land = 1 / 4 := by
    -- proof steps omitted
    sorry

end fraction_habitable_surface_l2002_200217


namespace brad_books_this_month_l2002_200225

-- Define the number of books William read last month
def william_books_last_month : ℕ := 6

-- Define the number of books Brad read last month
def brad_books_last_month : ℕ := 3 * william_books_last_month

-- Define the number of books Brad read this month as a variable
variable (B : ℕ)

-- Define the total number of books William read over the two months
def total_william_books (B : ℕ) : ℕ := william_books_last_month + 2 * B

-- Define the total number of books Brad read over the two months
def total_brad_books (B : ℕ) : ℕ := brad_books_last_month + B

-- State the condition that William read 4 more books than Brad
def william_read_more_books_condition (B : ℕ) : Prop := total_william_books B = total_brad_books B + 4

-- State the theorem to be proven
theorem brad_books_this_month (B : ℕ) : william_read_more_books_condition B → B = 16 :=
by
  sorry

end brad_books_this_month_l2002_200225


namespace recurring_decimal_to_fraction_l2002_200223

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l2002_200223


namespace minimum_value_l2002_200220

theorem minimum_value (x y z : ℝ) (h : x + 2 * y + z = 1) : x^2 + 4 * y^2 + z^2 ≥ 1 / 3 :=
sorry

end minimum_value_l2002_200220


namespace sachin_rahul_age_ratio_l2002_200253

theorem sachin_rahul_age_ratio :
  ∀ (Sachin_age Rahul_age: ℕ),
    Sachin_age = 49 →
    Rahul_age = Sachin_age + 14 →
    Nat.gcd Sachin_age Rahul_age = 7 →
    (Sachin_age / Nat.gcd Sachin_age Rahul_age) = 7 ∧ (Rahul_age / Nat.gcd Sachin_age Rahul_age) = 9 :=
by
  intros Sachin_age Rahul_age h1 h2 h3
  rw [h1, h2]
  sorry

end sachin_rahul_age_ratio_l2002_200253
