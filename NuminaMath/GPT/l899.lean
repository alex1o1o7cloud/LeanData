import Mathlib

namespace sin_300_eq_neg_sqrt3_div_2_l899_89955

-- Defining the problem statement as a Lean theorem
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l899_89955


namespace simplify_expr_for_a_neq_0_1_neg1_final_value_when_a_2_l899_89959

theorem simplify_expr_for_a_neq_0_1_neg1 (a : ℝ) (h1 : a ≠ 1) (h0 : a ≠ 0) (h_neg1 : a ≠ -1) :
  ( (a - 1)^2 / ((a + 1) * (a - 1)) ) / (a - (2 * a / (a + 1))) = 1 / a := by
  sorry

theorem final_value_when_a_2 :
  ( (2 - 1)^2 / ((2 + 1) * (2 - 1)) ) / (2 - (2 * 2 / (2 + 1))) = 1 / 2 := by
  sorry

end simplify_expr_for_a_neq_0_1_neg1_final_value_when_a_2_l899_89959


namespace find_value_l899_89975

noncomputable def a : ℝ := 5 - 2 * Real.sqrt 6

theorem find_value :
  a^2 - 10 * a + 1 = 0 :=
by
  -- Since we are only required to write the statement, add sorry to skip the proof.
  sorry

end find_value_l899_89975


namespace bert_same_kangaroos_as_kameron_in_40_days_l899_89973

theorem bert_same_kangaroos_as_kameron_in_40_days
  (k : ℕ := 100)
  (b : ℕ := 20)
  (r : ℕ := 2) :
  ∃ t : ℕ, t = 40 ∧ b + t * r = k := by
  sorry

end bert_same_kangaroos_as_kameron_in_40_days_l899_89973


namespace period1_period2_multiple_l899_89967

theorem period1_period2_multiple
  (students_period1 : ℕ)
  (students_period2 : ℕ)
  (h_students_period1 : students_period1 = 11)
  (h_students_period2 : students_period2 = 8)
  (M : ℕ)
  (h_condition : students_period1 = M * students_period2 - 5) :
  M = 2 :=
by
  sorry

end period1_period2_multiple_l899_89967


namespace negation_of_universal_proposition_l899_89983

theorem negation_of_universal_proposition (x : ℝ) :
  (¬ (∀ x : ℝ, |x| < 0)) ↔ (∃ x_0 : ℝ, |x_0| ≥ 0) := by
  sorry

end negation_of_universal_proposition_l899_89983


namespace crayons_given_l899_89957

theorem crayons_given (initial lost left given : ℕ)
  (h1 : initial = 1453)
  (h2 : lost = 558)
  (h3 : left = 332)
  (h4 : given = initial - left - lost) :
  given = 563 :=
by
  rw [h1, h2, h3] at h4
  exact h4

end crayons_given_l899_89957


namespace triangle_area_l899_89999

theorem triangle_area (a b c : ℕ) (h₁ : a = 7) (h₂ : b = 24) (h₃ : c = 25) (h₄ : a^2 + b^2 = c^2) : 
  ∃ A : ℕ, A = 84 ∧ A = (a * b) / 2 := by
  sorry

end triangle_area_l899_89999


namespace woman_weaves_amount_on_20th_day_l899_89916

theorem woman_weaves_amount_on_20th_day
  (a d : ℚ)
  (a2 : a + d = 17) -- second-day weaving in inches
  (S15 : 15 * a + 105 * d = 720) -- total for the first 15 days in inches
  : a + 19 * d = 108 := -- weaving on the twentieth day in inches (9 feet)
by
  sorry

end woman_weaves_amount_on_20th_day_l899_89916


namespace cos_180_degree_l899_89901

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l899_89901


namespace mitchell_pizzas_l899_89923

def pizzas_bought (slices_per_goal goals_per_game games slices_per_pizza : ℕ) : ℕ :=
  (slices_per_goal * goals_per_game * games) / slices_per_pizza

theorem mitchell_pizzas : pizzas_bought 1 9 8 12 = 6 := by
  sorry

end mitchell_pizzas_l899_89923


namespace hyperbola_eccentricity_l899_89907

/--
Given a hyperbola with the following properties:
1. Point \( P \) is on the left branch of the hyperbola \( C \): \(\frac{x^2}{a^2} - \frac{y^2}{b^2} = 1\), where \( a > 0 \) and \( b > 0 \).
2. \( F_2 \) is the right focus of the hyperbola.
3. One of the asymptotes of the hyperbola is perpendicular to the line segment \( PF_2 \).

Prove that the eccentricity \( e \) of the hyperbola is \( \sqrt{5} \).
-/
theorem hyperbola_eccentricity (a b e : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (P_on_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (F2_is_focus : True) -- Placeholder for focus-related condition
  (asymptote_perpendicular : True) -- Placeholder for asymptote perpendicular condition
  : e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l899_89907


namespace total_pay_per_week_l899_89984

variable (X Y : ℝ)
variable (hx : X = 1.2 * Y)
variable (hy : Y = 240)

theorem total_pay_per_week : X + Y = 528 := by
  sorry

end total_pay_per_week_l899_89984


namespace largest_cube_surface_area_l899_89913

theorem largest_cube_surface_area (width length height: ℕ) (h_w: width = 12) (h_l: length = 16) (h_h: height = 14) :
  (6 * (min width (min length height))^2) = 864 := by
  sorry

end largest_cube_surface_area_l899_89913


namespace dusting_days_l899_89954

theorem dusting_days 
    (vacuuming_minutes_per_day : ℕ) 
    (vacuuming_days_per_week : ℕ)
    (dusting_minutes_per_day : ℕ)
    (total_cleaning_minutes_per_week : ℕ)
    (x : ℕ) :
    vacuuming_minutes_per_day = 30 →
    vacuuming_days_per_week = 3 →
    dusting_minutes_per_day = 20 →
    total_cleaning_minutes_per_week = 130 →
    (vacuuming_minutes_per_day * vacuuming_days_per_week + dusting_minutes_per_day * x = total_cleaning_minutes_per_week) →
    x = 2 :=
by
  -- Proof steps go here
  sorry

end dusting_days_l899_89954


namespace remainders_of_65_powers_l899_89986

theorem remainders_of_65_powers (n : ℕ) :
  (65 ^ (6 * n)) % 9 = 1 ∧
  (65 ^ (6 * n + 1)) % 9 = 2 ∧
  (65 ^ (6 * n + 2)) % 9 = 4 ∧
  (65 ^ (6 * n + 3)) % 9 = 8 :=
by
  sorry

end remainders_of_65_powers_l899_89986


namespace gcd_hcf_of_36_and_84_l899_89914

theorem gcd_hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := sorry

end gcd_hcf_of_36_and_84_l899_89914


namespace allocation_methods_count_l899_89989

/-- The number of ways to allocate 24 quotas to 3 venues such that:
1. Each venue gets at least one quota.
2. Each venue gets a different number of quotas.
is equal to 222. -/
theorem allocation_methods_count : 
  ∃ n : ℕ, n = 222 ∧ 
  ∃ (a b c: ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a + b + c = 24 ∧ 
  1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c := 
sorry

end allocation_methods_count_l899_89989


namespace smallest_three_digit_multiple_of_17_l899_89933

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l899_89933


namespace halve_second_column_l899_89935

-- Definitions of given matrices
variable (f g h i : ℝ)
variable (A : Matrix (Fin 2) (Fin 2) ℝ := ![![f, g], ![h, i]])
variable (N : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, (1/2)]])

-- Proof statement to be proved
theorem halve_second_column (hf : f ≠ 0) (hh : h ≠ 0) : N * A = ![![f, (1/2) * g], ![h, (1/2) * i]] := by
  sorry

end halve_second_column_l899_89935


namespace boxes_given_away_l899_89972

def total_boxes := 12
def pieces_per_box := 6
def remaining_pieces := 30

theorem boxes_given_away : (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box = 7 :=
by
  sorry

end boxes_given_away_l899_89972


namespace Geli_pushups_and_runs_l899_89968

def initial_pushups : ℕ := 10
def increment_pushups : ℕ := 5
def workouts_per_week : ℕ := 3
def weeks_in_a_month : ℕ := 4
def pushups_per_mile_run : ℕ := 30

def workout_days_in_month : ℕ := workouts_per_week * weeks_in_a_month

def pushups_on_day (day : ℕ) : ℕ := initial_pushups + (day - 1) * increment_pushups

def total_pushups : ℕ := (workout_days_in_month / 2) * (initial_pushups + pushups_on_day workout_days_in_month)

def one_mile_runs (total_pushups : ℕ) : ℕ := total_pushups / pushups_per_mile_run

theorem Geli_pushups_and_runs :
  total_pushups = 450 ∧ one_mile_runs total_pushups = 15 :=
by
  -- Here, we should prove total_pushups = 450 and one_mile_runs total_pushups = 15.
  sorry

end Geli_pushups_and_runs_l899_89968


namespace fraction_power_mult_equality_l899_89952

-- Define the fraction and the power
def fraction := (1 : ℚ) / 3
def power : ℚ := fraction ^ 4

-- Define the multiplication
def result := 8 * power

-- Prove the equality
theorem fraction_power_mult_equality : result = 8 / 81 := by
  sorry

end fraction_power_mult_equality_l899_89952


namespace smallest_x_l899_89911

theorem smallest_x (x: ℕ) (hx: x > 0) (h: 11^2021 ∣ 5^(3*x) - 3^(4*x)) : 
  x = 11^2020 := sorry

end smallest_x_l899_89911


namespace find_f_g_2_l899_89949

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x^2 - 6

theorem find_f_g_2 : f (g 2) = 1 := 
  by
  -- Proof goes here
  sorry

end find_f_g_2_l899_89949


namespace ratio_eval_l899_89904

universe u

def a : ℕ := 121
def b : ℕ := 123
def c : ℕ := 122

theorem ratio_eval : (2 ^ a * 3 ^ b) / (6 ^ c) = (3 / 2) := by
  sorry

end ratio_eval_l899_89904


namespace rectangle_square_area_ratio_eq_one_l899_89909

theorem rectangle_square_area_ratio_eq_one (r l w s: ℝ) (h1: l = 2 * w) (h2: r ^ 2 = (l / 2) ^ 2 + w ^ 2) (h3: s ^ 2 = 2 * r ^ 2) : 
  (l * w) / (s ^ 2) = 1 :=
by
sorry

end rectangle_square_area_ratio_eq_one_l899_89909


namespace initial_pounds_of_coffee_l899_89932

variable (x : ℝ) (h1 : 0.25 * x = d₀) (h2 : 0.60 * 100 = d₁) 
          (h3 : (d₀ + d₁) / (x + 100) = 0.32)

theorem initial_pounds_of_coffee (d₀ d₁ : ℝ) : 
  x = 400 :=
by
  -- Given conditions
  have h1 : d₀ = 0.25 * x := sorry
  have h2 : d₁ = 0.60 * 100 := sorry
  have h3 : 0.32 = (d₀ + d₁) / (x + 100) := sorry
  
  -- Additional steps to solve for x
  sorry

end initial_pounds_of_coffee_l899_89932


namespace decrease_percent_in_revenue_l899_89910

-- Definitions based on the conditions
def original_tax (T : ℝ) := T
def original_consumption (C : ℝ) := C
def new_tax (T : ℝ) := 0.70 * T
def new_consumption (C : ℝ) := 1.20 * C

-- Theorem statement for the decrease percent in revenue
theorem decrease_percent_in_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) : 
  100 * ((original_tax T * original_consumption C - new_tax T * new_consumption C) / (original_tax T * original_consumption C)) = 16 :=
by
  sorry

end decrease_percent_in_revenue_l899_89910


namespace max_value_of_f_on_interval_l899_89977

noncomputable def f (x : ℝ) : ℝ := (Real.sin (4 * x)) / (2 * Real.sin ((Real.pi / 2) - 2 * x))

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 6), f x = (Real.sqrt 3) / 2 := sorry

end max_value_of_f_on_interval_l899_89977


namespace num_isosceles_triangles_is_24_l899_89997

-- Define the structure of the hexagonal prism
structure HexagonalPrism :=
  (height : ℝ)
  (side_length : ℝ)
  (num_vertices : ℕ)

-- Define the specific hexagonal prism from the problem
def prism := HexagonalPrism.mk 2 1 12

-- Function to count the number of isosceles triangles in a given hexagonal prism
noncomputable def count_isosceles_triangles (hp : HexagonalPrism) : ℕ := sorry

-- The theorem that needs to be proved
theorem num_isosceles_triangles_is_24 :
  count_isosceles_triangles prism = 24 :=
sorry

end num_isosceles_triangles_is_24_l899_89997


namespace injective_g_restricted_to_interval_l899_89943

def g (x : ℝ) : ℝ := (x + 3) ^ 2 - 10

theorem injective_g_restricted_to_interval :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Ici (-3) → x2 ∈ Set.Ici (-3) → g x1 = g x2 → x1 = x2) :=
sorry

end injective_g_restricted_to_interval_l899_89943


namespace initial_number_of_bedbugs_l899_89974

theorem initial_number_of_bedbugs (N : ℕ) 
  (h1 : ∃ N : ℕ, True)
  (h2 : ∀ (n : ℕ), (triples_daily : ℕ → ℕ) → triples_daily n = 3 * n)
  (h3 : ∀ (n : ℕ), (N * 3^4 = n) → n = 810) : 
  N = 10 :=
sorry

end initial_number_of_bedbugs_l899_89974


namespace triangle_angle_area_l899_89912

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x
variables {A B C : ℝ}
variables {BC : ℝ}
variables {S : ℝ}

theorem triangle_angle_area (hABC : A + B + C = π) (hBC : BC = 2) (h_fA : f A = 0) 
  (hA : A = π / 3) : S = Real.sqrt 3 :=
by
  -- Sorry, proof skipped
  sorry

end triangle_angle_area_l899_89912


namespace find_b_from_conditions_l899_89990

theorem find_b_from_conditions 
  (x y b : ℝ) 
  (h1 : 3 * x - 5 * y = b) 
  (h2 : x / (x + y) = 5 / 7) 
  (h3 : x - y = 3) : 
  b = 5 := 
by 
  sorry

end find_b_from_conditions_l899_89990


namespace initial_fee_l899_89998

theorem initial_fee (total_bowls : ℤ) (lost_bowls : ℤ) (broken_bowls : ℤ) (safe_fee : ℤ)
  (loss_fee : ℤ) (total_payment : ℤ) (paid_amount : ℤ) :
  total_bowls = 638 →
  lost_bowls = 12 →
  broken_bowls = 15 →
  safe_fee = 3 →
  loss_fee = 4 →
  total_payment = 1825 →
  paid_amount = total_payment - ((total_bowls - lost_bowls - broken_bowls) * safe_fee - (lost_bowls + broken_bowls) * loss_fee) →
  paid_amount = 100 :=
by
  intros _ _ _ _ _ _ _
  sorry

end initial_fee_l899_89998


namespace Olivia_money_left_l899_89982

theorem Olivia_money_left (initial_amount spend_amount : ℕ) (h1 : initial_amount = 128) 
  (h2 : spend_amount = 38) : initial_amount - spend_amount = 90 := by
  sorry

end Olivia_money_left_l899_89982


namespace price_for_two_bracelets_l899_89988

theorem price_for_two_bracelets
    (total_bracelets : ℕ)
    (price_per_bracelet : ℕ)
    (total_earned_for_single : ℕ)
    (total_earned : ℕ)
    (bracelets_sold_single : ℕ)
    (bracelets_left : ℕ)
    (remaining_earned : ℕ)
    (pairs_sold : ℕ)
    (price_per_pair : ℕ) :
    total_bracelets = 30 →
    price_per_bracelet = 5 →
    total_earned_for_single = 60 →
    total_earned = 132 →
    bracelets_sold_single = total_earned_for_single / price_per_bracelet →
    bracelets_left = total_bracelets - bracelets_sold_single →
    remaining_earned = total_earned - total_earned_for_single →
    pairs_sold = bracelets_left / 2 →
    price_per_pair = remaining_earned / pairs_sold →
    price_per_pair = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end price_for_two_bracelets_l899_89988


namespace initial_money_equals_26_l899_89930

def cost_jumper : ℕ := 9
def cost_tshirt : ℕ := 4
def cost_heels : ℕ := 5
def money_left : ℕ := 8

def total_cost_items : ℕ := cost_jumper + cost_tshirt + cost_heels

theorem initial_money_equals_26 : total_cost_items + money_left = 26 := by
  sorry

end initial_money_equals_26_l899_89930


namespace determine_a_values_l899_89926

theorem determine_a_values (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) ↔ a = 2 ∨ a = 8 :=
by
  sorry

end determine_a_values_l899_89926


namespace polynomial_average_k_l899_89970

theorem polynomial_average_k (h : ∀ x : ℕ, x * (36 / x) = 36 → (x + (36 / x) = 37 ∨ x + (36 / x) = 20 ∨ x + (36 / x) = 15 ∨ x + (36 / x) = 13 ∨ x + (36 / x) = 12)) :
  (37 + 20 + 15 + 13 + 12) / 5 = 19.4 := by
sorry

end polynomial_average_k_l899_89970


namespace lcm_is_multiple_of_230_l899_89902

theorem lcm_is_multiple_of_230 (d n : ℕ) (h1 : n = 230) (h2 : ¬ (3 ∣ n)) (h3 : ¬ (2 ∣ d)) : ∃ m : ℕ, Nat.lcm d n = 230 * m :=
by
  exists 1 -- Placeholder for demonstration purposes
  sorry

end lcm_is_multiple_of_230_l899_89902


namespace base7_to_base10_proof_l899_89951

theorem base7_to_base10_proof (c d : ℕ) (h1 : 764 = 4 * 100 + c * 10 + d) : (c * d) / 20 = 6 / 5 :=
by
  sorry

end base7_to_base10_proof_l899_89951


namespace total_cost_nancy_spends_l899_89985

def price_crystal_beads : ℝ := 12
def price_metal_beads : ℝ := 15
def sets_crystal_beads : ℕ := 3
def sets_metal_beads : ℕ := 4
def discount_crystal : ℝ := 0.10
def tax_metal : ℝ := 0.05

theorem total_cost_nancy_spends :
  sets_crystal_beads * price_crystal_beads * (1 - discount_crystal) + 
  sets_metal_beads * price_metal_beads * (1 + tax_metal) = 95.40 := 
  by sorry

end total_cost_nancy_spends_l899_89985


namespace last_two_digits_of_product_squared_l899_89924

def mod_100 (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_product_squared :
  mod_100 ((301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 2) = 76 := 
by
  sorry

end last_two_digits_of_product_squared_l899_89924


namespace steven_needs_more_seeds_l899_89948

theorem steven_needs_more_seeds :
  let total_seeds_needed := 60
  let seeds_per_apple := 6
  let seeds_per_pear := 2
  let seeds_per_grape := 3
  let apples_collected := 4
  let pears_collected := 3
  let grapes_collected := 9
  total_seeds_needed - (apples_collected * seeds_per_apple + pears_collected * seeds_per_pear + grapes_collected * seeds_per_grape) = 3 :=
by
  sorry

end steven_needs_more_seeds_l899_89948


namespace cube_volume_is_27_l899_89971

noncomputable def original_cube_edge (a : ℝ) : ℝ := a

noncomputable def original_cube_volume (a : ℝ) : ℝ := a^3

noncomputable def new_rectangular_solid_volume (a : ℝ) : ℝ := (a-2) * a * (a+2)

theorem cube_volume_is_27 (a : ℝ) (h : original_cube_volume a - new_rectangular_solid_volume a = 14) : original_cube_volume a = 27 :=
by
  sorry

end cube_volume_is_27_l899_89971


namespace weight_computation_requires_initial_weight_l899_89966

-- Let's define the conditions
variable (initial_weight : ℕ) -- The initial weight of the pet; needs to be provided
def yearly_gain := 11  -- The pet gains 11 pounds each year
def age := 8  -- The pet is 8 years old

-- Define the goal to be proved
def current_weight_computable : Prop :=
  initial_weight ≠ 0 → initial_weight + (yearly_gain * age) ≠ 0

-- State the theorem
theorem weight_computation_requires_initial_weight : ¬ ∃ current_weight, initial_weight + (yearly_gain * age) = current_weight :=
by {
  sorry
}

end weight_computation_requires_initial_weight_l899_89966


namespace function_decreasing_on_interval_l899_89947

noncomputable def g (x : ℝ) := -(1 / 3) * Real.sin (4 * x - Real.pi / 3)
noncomputable def f (x : ℝ) := -(1 / 3) * Real.sin (4 * x)

theorem function_decreasing_on_interval :
  ∀ x y : ℝ, (-Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 8) → (-Real.pi / 8 ≤ y ∧ y ≤ Real.pi / 8) → x < y → f x > f y :=
sorry

end function_decreasing_on_interval_l899_89947


namespace max_area_of_triangle_l899_89928

noncomputable def max_triangle_area (v1 v2 v3 : ℝ) (S : ℝ) : Prop :=
  2 * S + Real.sqrt 3 * (v1 * v2 + v3) = 0 ∧ v3 = Real.sqrt 3 → S ≤ Real.sqrt 3 / 4

theorem max_area_of_triangle (v1 v2 v3 S : ℝ) :
  max_triangle_area v1 v2 v3 S :=
by
  sorry

end max_area_of_triangle_l899_89928


namespace eccentricity_of_ellipse_l899_89927

theorem eccentricity_of_ellipse (m n : ℝ) (h1 : 1 / m + 2 / n = 1) (h2 : 0 < m) (h3 : 0 < n) (h4 : m * n = 8) :
  let a := n
  let b := m
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  e = Real.sqrt 3 / 2 := 
sorry

end eccentricity_of_ellipse_l899_89927


namespace f_2017_of_9_eq_8_l899_89969

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_k (k n : ℕ) : ℕ :=
  if k = 0 then n else f (f_k (k-1) n)

theorem f_2017_of_9_eq_8 : f_k 2017 9 = 8 := by
  sorry

end f_2017_of_9_eq_8_l899_89969


namespace non_congruent_squares_6x6_grid_l899_89956

theorem non_congruent_squares_6x6_grid : 
  let count_squares (n: ℕ) : ℕ := 
    let horizontal_or_vertical := (6 - n) * (6 - n)
    let diagonal := if n * n <= 6 * 6 then (6 - n + 1) * (6 - n + 1) else 0
    horizontal_or_vertical + diagonal
  (count_squares 1) + (count_squares 2) + (count_squares 3) + (count_squares 4) + (count_squares 5) = 141 :=
by
  sorry

end non_congruent_squares_6x6_grid_l899_89956


namespace george_correct_possible_change_sum_l899_89931

noncomputable def george_possible_change_sum : ℕ :=
if h : ∃ (change : ℕ), change < 100 ∧
  ((change % 25 == 7) ∨ (change % 25 == 32) ∨ (change % 25 == 57) ∨ (change % 25 == 82)) ∧
  ((change % 10 == 2) ∨ (change % 10 == 12) ∨ (change % 10 == 22) ∨
   (change % 10 == 32) ∨ (change % 10 == 42) ∨ (change % 10 == 52) ∨
   (change % 10 == 62) ∨ (change % 10 == 72) ∨ (change % 10 == 82) ∨ (change % 10 == 92)) ∧
  ((change % 5 == 9) ∨ (change % 5 == 14) ∨ (change % 5 == 19) ∨
   (change % 5 == 24) ∨ (change % 5 == 29) ∨ (change % 5 == 34) ∨
   (change % 5 == 39) ∨ (change % 5 == 44) ∨ (change % 5 == 49) ∨
   (change % 5 == 54) ∨ (change % 5 == 59) ∨ (change % 5 == 64) ∨
   (change % 5 == 69) ∨ (change % 5 == 74) ∨ (change % 5 == 79) ∨
   (change % 5 == 84) ∨ (change % 5 == 89) ∨ (change % 5 == 94) ∨ (change % 5 == 99)) then
  114
else 0

theorem george_correct_possible_change_sum :
  george_possible_change_sum = 114 :=
by
  sorry

end george_correct_possible_change_sum_l899_89931


namespace nth_equation_l899_89938

theorem nth_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * (n - 1) + 1 :=
by
  sorry

end nth_equation_l899_89938


namespace find_chemistry_marks_l899_89961

theorem find_chemistry_marks :
  (let E := 96
   let M := 95
   let P := 82
   let B := 95
   let avg := 93
   let n := 5
   let Total := avg * n
   let Chemistry_marks := Total - (E + M + P + B)
   Chemistry_marks = 97) :=
by
  let E := 96
  let M := 95
  let P := 82
  let B := 95
  let avg := 93
  let n := 5
  let Total := avg * n
  have h_total : Total = 465 := by norm_num
  let Chemistry_marks := Total - (E + M + P + B)
  have h_chemistry_marks : Chemistry_marks = 97 := by norm_num
  exact h_chemistry_marks

end find_chemistry_marks_l899_89961


namespace algebraic_expression_value_l899_89960

-- Define the given condition as a predicate
def condition (a : ℝ) := a^2 + a - 4 = 0

-- Then the goal to prove with the given condition
theorem algebraic_expression_value (a : ℝ) (h : condition a) : (a^2 - 3) * (a + 2) = -2 :=
sorry

end algebraic_expression_value_l899_89960


namespace gravel_cost_calculation_l899_89946

def cubicYardToCubicFoot : ℕ := 27
def costPerCubicFoot : ℕ := 8
def volumeInCubicYards : ℕ := 8

theorem gravel_cost_calculation : 
  (volumeInCubicYards * cubicYardToCubicFoot * costPerCubicFoot) = 1728 := 
by
  -- This is just a placeholder to ensure the statement is syntactically correct.
  sorry

end gravel_cost_calculation_l899_89946


namespace sum_of_fifth_powers_52070424_l899_89920

noncomputable def sum_of_fifth_powers (n : ℤ) : ℤ :=
  (n-1)^5 + n^5 + (n+1)^5

theorem sum_of_fifth_powers_52070424 :
  ∃ (n : ℤ), (n-1)^2 + n^2 + (n+1)^2 = 2450 ∧ sum_of_fifth_powers n = 52070424 :=
by
  sorry

end sum_of_fifth_powers_52070424_l899_89920


namespace tangent_lines_parallel_l899_89987

-- Definitions and conditions
def curve (x : ℝ) : ℝ := x^3 + x - 2
def line (x : ℝ) : ℝ := 4 * x - 1
def tangent_line_eq (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Proof statement
theorem tangent_lines_parallel (tangent_line : ℝ → ℝ) :
  (∃ x : ℝ, tangent_line_eq 4 (-1) 0 x (curve x)) ∧ 
  (∃ x : ℝ, tangent_line_eq 4 (-1) (-4) x (curve x)) :=
sorry

end tangent_lines_parallel_l899_89987


namespace julia_change_l899_89908

-- Definitions based on the problem conditions
def price_of_snickers : ℝ := 1.5
def price_of_mms : ℝ := 2 * price_of_snickers
def total_cost_of_snickers (num_snickers : ℕ) : ℝ := num_snickers * price_of_snickers
def total_cost_of_mms (num_mms : ℕ) : ℝ := num_mms * price_of_mms
def total_purchase (num_snickers num_mms : ℕ) : ℝ := total_cost_of_snickers num_snickers + total_cost_of_mms num_mms
def amount_given : ℝ := 2 * 10

-- Prove the change is $8
theorem julia_change : total_purchase 2 3 = 12 ∧ (amount_given - total_purchase 2 3) = 8 :=
by
  sorry

end julia_change_l899_89908


namespace Andy_more_white_socks_than_black_l899_89905

def num_black_socks : ℕ := 6
def initial_num_white_socks : ℕ := 4 * num_black_socks
def final_num_white_socks : ℕ := initial_num_white_socks / 2
def more_white_than_black : ℕ := final_num_white_socks - num_black_socks

theorem Andy_more_white_socks_than_black :
  more_white_than_black = 6 :=
sorry

end Andy_more_white_socks_than_black_l899_89905


namespace merchant_product_quantities_l899_89944

theorem merchant_product_quantities
  (x p1 : ℝ)
  (h1 : 4000 = x * p1)
  (h2 : 8800 = 2 * x * (p1 + 4))
  (h3 : (8800 / (2 * x)) - (4000 / x) = 4):
  x = 100 ∧ 2 * x = 200 :=
by sorry

end merchant_product_quantities_l899_89944


namespace natural_pair_prime_ratio_l899_89963

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem natural_pair_prime_ratio :
  ∃ (x y : ℕ), (x = 14 ∧ y = 2) ∧ is_prime (x * y^3 / (x + y)) :=
by
  use 14
  use 2
  sorry

end natural_pair_prime_ratio_l899_89963


namespace nina_walking_distance_l899_89950

def distance_walked_by_john : ℝ := 0.7
def distance_john_further_than_nina : ℝ := 0.3

def distance_walked_by_nina : ℝ := distance_walked_by_john - distance_john_further_than_nina

theorem nina_walking_distance :
  distance_walked_by_nina = 0.4 :=
by
  sorry

end nina_walking_distance_l899_89950


namespace relationship_between_x_and_y_l899_89996

theorem relationship_between_x_and_y (x y : ℝ) (h1 : 2 * x - y > 3 * x) (h2 : x + 2 * y < 2 * y) :
  x < 0 ∧ y > 0 :=
sorry

end relationship_between_x_and_y_l899_89996


namespace probability_at_least_one_defective_is_correct_l899_89919

noncomputable def probability_at_least_one_defective : ℚ :=
  let total_bulbs := 23
  let defective_bulbs := 4
  let non_defective_bulbs := total_bulbs - defective_bulbs
  let probability_neither_defective :=
    (non_defective_bulbs / total_bulbs) * ((non_defective_bulbs - 1) / (total_bulbs - 1))
  1 - probability_neither_defective

theorem probability_at_least_one_defective_is_correct :
  probability_at_least_one_defective = 164 / 506 :=
by
  sorry

end probability_at_least_one_defective_is_correct_l899_89919


namespace part1_part2_l899_89979

noncomputable def f (x a : ℝ) := (x - 1) * Real.exp x + a * x + 1
noncomputable def g (x : ℝ) := x * Real.exp x

-- Problem Part 1: Prove the range of a for which f(x) has two extreme points
theorem part1 (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = f x₂ a) ↔ (0 < a ∧ a < (1 / Real.exp 1)) :=
sorry

-- Problem Part 2: Prove the range of a for which f(x) ≥ 2sin(x) for x ≥ 0
theorem part2 (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f x a ≥ 2 * Real.sin x) ↔ (2 ≤ a) :=
sorry

end part1_part2_l899_89979


namespace complex_quadratic_solution_l899_89992

theorem complex_quadratic_solution (c d : ℤ) (h1 : 0 < c) (h2 : 0 < d) (h3 : (c + d * Complex.I) ^ 2 = 7 + 24 * Complex.I) :
  c + d * Complex.I = 4 + 3 * Complex.I :=
sorry

end complex_quadratic_solution_l899_89992


namespace trader_sold_40_meters_of_cloth_l899_89978

theorem trader_sold_40_meters_of_cloth 
  (total_profit_per_meter : ℕ) 
  (total_profit : ℕ) 
  (meters_sold : ℕ) 
  (h1 : total_profit_per_meter = 30) 
  (h2 : total_profit = 1200) 
  (h3 : total_profit = total_profit_per_meter * meters_sold) : 
  meters_sold = 40 := by
  sorry

end trader_sold_40_meters_of_cloth_l899_89978


namespace quadratic_solution_l899_89941

theorem quadratic_solution
  (a c : ℝ) (h : a ≠ 0) (h_passes_through : ∃ b, b = c - 9 * a) :
  ∀ (x : ℝ), (ax^2 - 2 * a * x + c = 0) ↔ (x = -1) ∨ (x = 3) :=
by
  sorry

end quadratic_solution_l899_89941


namespace no_2021_residents_possible_l899_89953

-- Definition: Each islander is either a knight or a liar
def is_knight_or_liar (i : ℕ) : Prop := true -- Placeholder definition for either being a knight or a liar

-- Definition: Knights always tell the truth
def knight_tells_truth (i : ℕ) : Prop := true -- Placeholder definition for knights telling the truth

-- Definition: Liars always lie
def liar_always_lies (i : ℕ) : Prop := true -- Placeholder definition for liars always lying

-- Definition: Even number of knights claimed by some islanders
def even_number_of_knights : Prop := true -- Placeholder definition for the claim of even number of knights

-- Definition: Odd number of liars claimed by remaining islanders
def odd_number_of_liars : Prop := true -- Placeholder definition for the claim of odd number of liars

-- Question and proof problem
theorem no_2021_residents_possible (K L : ℕ) (h1 : K + L = 2021) (h2 : ∀ i, is_knight_or_liar i) 
(h3 : ∀ k, knight_tells_truth k → even_number_of_knights) 
(h4 : ∀ l, liar_always_lies l → odd_number_of_liars) : 
  false := sorry

end no_2021_residents_possible_l899_89953


namespace find_n_l899_89917

theorem find_n (n : ℕ) (h : 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n = 3^2012) : n = 1005 :=
sorry

end find_n_l899_89917


namespace day_after_exponential_days_l899_89925

noncomputable def days_since_monday (n : ℕ) : String :=
  let days := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  days.get! (n % 7)

theorem day_after_exponential_days :
  days_since_monday (2^20) = "Friday" :=
by
  sorry

end day_after_exponential_days_l899_89925


namespace joan_seashells_l899_89962

variable (initialSeashells seashellsGiven remainingSeashells : ℕ)

theorem joan_seashells : initialSeashells = 79 ∧ seashellsGiven = 63 ∧ remainingSeashells = initialSeashells - seashellsGiven → remainingSeashells = 16 :=
by
  intros
  sorry

end joan_seashells_l899_89962


namespace lcm_condition_proof_l899_89942

theorem lcm_condition_proof (n : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 0 < a i)
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j)
  (h3 : ∀ i, 1 ≤ i → i ≤ n → a i ≤ 2 * n)
  (h4 : ∀ i j, 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → i ≠ j → Nat.lcm (a i) (a j) > 2 * n) :
  a 1 > n * 2 / 3 := 
sorry

end lcm_condition_proof_l899_89942


namespace meaningful_fraction_l899_89965

theorem meaningful_fraction {x : ℝ} : (x - 2) ≠ 0 ↔ x ≠ 2 :=
by
  sorry

end meaningful_fraction_l899_89965


namespace total_lives_correct_l899_89981

namespace VideoGame

def num_friends : ℕ := 8
def lives_each : ℕ := 8

def total_lives (n : ℕ) (l : ℕ) : ℕ := n * l 

theorem total_lives_correct : total_lives num_friends lives_each = 64 := by
  sorry

end total_lives_correct_l899_89981


namespace imaginary_part_zero_iff_a_eq_neg1_l899_89993

theorem imaginary_part_zero_iff_a_eq_neg1 (a : ℝ) (h : (Complex.I * (a + Complex.I) + a - 1).im = 0) : 
  a = -1 :=
sorry

end imaginary_part_zero_iff_a_eq_neg1_l899_89993


namespace cone_volume_l899_89906

theorem cone_volume (V_cylinder : ℝ) (V_cone : ℝ) (h : V_cylinder = 81 * Real.pi) :
  V_cone = 27 * Real.pi :=
by
  sorry

end cone_volume_l899_89906


namespace solve_linear_system_l899_89940

theorem solve_linear_system :
  ∃ x y : ℚ, (3 * x - y = 4) ∧ (6 * x - 3 * y = 10) ∧ (x = 2 / 3) ∧ (y = -2) :=
by
  sorry

end solve_linear_system_l899_89940


namespace quadratic_inequality_solution_set_empty_l899_89980

theorem quadratic_inequality_solution_set_empty
  (m : ℝ)
  (h : ∀ x : ℝ, mx^2 - mx - 1 < 0) :
  -4 < m ∧ m < 0 :=
sorry

end quadratic_inequality_solution_set_empty_l899_89980


namespace pyramid_coloring_ways_l899_89929

theorem pyramid_coloring_ways (colors : Fin 5) 
  (coloring_condition : ∀ (a b : Fin 5), a ≠ b) :
  ∃ (ways: Nat), ways = 420 :=
by
  -- Given:
  -- 1. There are 5 available colors
  -- 2. Each vertex of the pyramid is colored differently from the vertices connected by an edge
  -- Prove:
  -- There are 420 ways to color the pyramid's vertices
  sorry

end pyramid_coloring_ways_l899_89929


namespace unique_solution_condition_l899_89976

theorem unique_solution_condition (a b c : ℝ) : 
  (∃! x : ℝ, 4 * x - 7 + a = c * x + b) ↔ c ≠ 4 :=
sorry

end unique_solution_condition_l899_89976


namespace total_study_time_is_60_l899_89915

-- Define the times Elizabeth studied for each test
def science_time : ℕ := 25
def math_time : ℕ := 35

-- Define the total study time
def total_study_time : ℕ := science_time + math_time

-- Proposition that the total study time equals 60 minutes
theorem total_study_time_is_60 : total_study_time = 60 := by
  /-
  Here we would provide the proof steps, but since the task is to write the statement only,
  we add 'sorry' to indicate the missing proof.
  -/
  sorry

end total_study_time_is_60_l899_89915


namespace cannot_be_square_of_difference_formula_l899_89945

theorem cannot_be_square_of_difference_formula (x y c d a b m n : ℝ) :
  ¬ ((m - n) * (-m + n) = (x^2 - y^2) ∨ 
       (m - n) * (-m + n) = (c^2 - d^2) ∨ 
       (m - n) * (-m + n) = (a^2 - b^2)) :=
by sorry

end cannot_be_square_of_difference_formula_l899_89945


namespace tea_blend_gain_percent_l899_89921

theorem tea_blend_gain_percent :
  let cost_18 := 18
  let cost_20 := 20
  let ratio_5_to_3 := (5, 3)
  let selling_price := 21
  let total_cost := (ratio_5_to_3.1 * cost_18) + (ratio_5_to_3.2 * cost_20)
  let total_weight := ratio_5_to_3.1 + ratio_5_to_3.2
  let cost_price_per_kg := total_cost / total_weight
  let gain_percent := ((selling_price - cost_price_per_kg) / cost_price_per_kg) * 100
  gain_percent = 12 :=
by
  sorry

end tea_blend_gain_percent_l899_89921


namespace find_AD_length_l899_89900

noncomputable def triangle_AD (A B C : Type) (AB AC : ℝ) (ratio_BD_CD : ℝ) (AD : ℝ) : Prop :=
  AB = 13 ∧ AC = 20 ∧ ratio_BD_CD = 3 / 4 → AD = 8 * Real.sqrt 2

theorem find_AD_length {A B C : Type} :
  triangle_AD A B C 13 20 (3/4) (8 * Real.sqrt 2) :=
by
  sorry

end find_AD_length_l899_89900


namespace child_tickets_sold_l899_89995

-- Define variables and types
variables (A C : ℕ)

-- Main theorem to prove
theorem child_tickets_sold : A + C = 80 ∧ 12 * A + 5 * C = 519 → C = 63 :=
by
  intros
  sorry

end child_tickets_sold_l899_89995


namespace smallest_possible_QNNN_l899_89934

theorem smallest_possible_QNNN :
  ∃ (Q N : ℕ), (N = 1 ∨ N = 5 ∨ N = 6) ∧ (NN = 10 * N + N) ∧ (Q * 1000 + NN * 10 + N = NN * N) ∧ (Q * 1000 + NN * 10 + N) = 275 :=
sorry

end smallest_possible_QNNN_l899_89934


namespace sum_of_integers_is_19_l899_89939

theorem sum_of_integers_is_19
  (a b : ℕ) 
  (h1 : a > b) 
  (h2 : a - b = 5) 
  (h3 : a * b = 84) : 
  a + b = 19 :=
sorry

end sum_of_integers_is_19_l899_89939


namespace longest_side_of_enclosure_l899_89958

theorem longest_side_of_enclosure (l w : ℝ)
  (h_perimeter : 2 * l + 2 * w = 240)
  (h_area : l * w = 8 * 240) :
  max l w = 80 :=
by
  sorry

end longest_side_of_enclosure_l899_89958


namespace max_min_value_d_l899_89991

-- Definitions of the given conditions
def circle_eqn (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Definition of the distance squared from a point to a fixed point
def dist_sq (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Definition of the function d
def d (P : ℝ × ℝ) : ℝ := dist_sq P A + dist_sq P B

-- The main theorem that we need to prove
theorem max_min_value_d :
  (∀ P : ℝ × ℝ, circle_eqn P.1 P.2 → d P ≤ 74) ∧
  (∃ P : ℝ × ℝ, circle_eqn P.1 P.2 ∧ d P = 74) ∧
  (∀ P : ℝ × ℝ, circle_eqn P.1 P.2 → 34 ≤ d P) ∧
  (∃ P : ℝ × ℝ, circle_eqn P.1 P.2 ∧ d P = 34) :=
sorry

end max_min_value_d_l899_89991


namespace rectangles_in_square_rectangles_in_three_squares_l899_89936

-- Given conditions as definitions
def positive_integer (n : ℕ) : Prop := n > 0

-- Part a
theorem rectangles_in_square (n : ℕ) (h : positive_integer n) :
  (n * (n + 1) / 2) ^ 2 = (n * (n + 1) / 2) ^ 2 :=
by sorry

-- Part b
theorem rectangles_in_three_squares (n : ℕ) (h : positive_integer n) :
  n^2 * (2 * n + 1)^2 - n^4 - n^3 * (n + 1) - (n * (n + 1) / 2)^2 = 
  n^2 * (2 * n + 1)^2 - n^4 - n^3 * (n + 1) - (n * (n + 1) / 2)^2 :=
by sorry

end rectangles_in_square_rectangles_in_three_squares_l899_89936


namespace find_f_2021_l899_89918

variable (f : ℝ → ℝ)

axiom functional_equation : ∀ a b : ℝ, f ( (a + 2 * b) / 3) = (f a + 2 * f b) / 3
axiom f_one : f 1 = 1
axiom f_four : f 4 = 7

theorem find_f_2021 : f 2021 = 4041 := by
  sorry

end find_f_2021_l899_89918


namespace find_weeks_period_l899_89937

def weekly_addition : ℕ := 3
def bikes_sold : ℕ := 18
def bikes_in_stock : ℕ := 45
def initial_stock : ℕ := 51

theorem find_weeks_period (x : ℕ) :
  initial_stock + weekly_addition * x - bikes_sold = bikes_in_stock ↔ x = 4 := 
by 
  sorry

end find_weeks_period_l899_89937


namespace find_age_l899_89964

theorem find_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 9) : b = 39 := 
by 
  sorry

end find_age_l899_89964


namespace intersection_a_b_l899_89994

-- Definitions of sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 2}
def B : Set ℝ := {-2, -1, 0}

-- The proof problem
theorem intersection_a_b : A ∩ B = {-1, 0} :=
by
  sorry

end intersection_a_b_l899_89994


namespace relationship_among_abc_l899_89903

noncomputable def a : ℝ := Real.sqrt 6 + Real.sqrt 7
noncomputable def b : ℝ := Real.sqrt 5 + Real.sqrt 8
def c : ℝ := 5

theorem relationship_among_abc : c < b ∧ b < a :=
by
  sorry

end relationship_among_abc_l899_89903


namespace value_preserving_interval_of_g_l899_89922

noncomputable def g (x : ℝ) (m : ℝ) : ℝ :=
  x + m - Real.log x

theorem value_preserving_interval_of_g
  (m : ℝ)
  (h_increasing : ∀ x, x ∈ Set.Ici 2 → 1 - 1 / x > 0)
  (h_range : ∀ y, y ∈ Set.Ici 2): 
  (2 + m - Real.log 2 = 2) → 
  m = Real.log 2 :=
by 
  sorry

end value_preserving_interval_of_g_l899_89922
