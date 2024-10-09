import Mathlib

namespace ratio_of_ages_l849_84995

theorem ratio_of_ages (D R : ℕ) (h1 : D = 3) (h2 : R + 22 = 26) : R / D = 4 / 3 := by
  sorry

end ratio_of_ages_l849_84995


namespace gas_total_cost_l849_84929

theorem gas_total_cost (x : ℝ) (h : (x/3) - 11 = x/5) : x = 82.5 :=
sorry

end gas_total_cost_l849_84929


namespace rational_number_25_units_away_l849_84971

theorem rational_number_25_units_away (x : ℚ) (h : |x| = 2.5) : x = 2.5 ∨ x = -2.5 := 
by
  sorry

end rational_number_25_units_away_l849_84971


namespace system_eq_solution_l849_84909

theorem system_eq_solution (x y c d : ℝ) (hd : d ≠ 0) 
  (h1 : 4 * x - 2 * y = c) 
  (h2 : 6 * y - 12 * x = d) :
  c / d = -1 / 3 := 
by 
  sorry

end system_eq_solution_l849_84909


namespace percentage_of_class_taking_lunch_l849_84925

theorem percentage_of_class_taking_lunch 
  (total_students : ℕ)
  (boys_ratio : ℕ := 6)
  (girls_ratio : ℕ := 4)
  (boys_percentage_lunch : ℝ := 0.60)
  (girls_percentage_lunch : ℝ := 0.40) :
  total_students = 100 →
  (6 / (6 + 4) * 100) = 60 →
  (4 / (6 + 4) * 100) = 40 →
  (boys_percentage_lunch * 60 + girls_percentage_lunch * 40) = 52 →
  ℝ :=
    by
      intros
      sorry

end percentage_of_class_taking_lunch_l849_84925


namespace consecutive_integers_sum_l849_84983

theorem consecutive_integers_sum (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : a < Real.sqrt 17) (h4 : Real.sqrt 17 < b) : a + b = 9 :=
sorry

end consecutive_integers_sum_l849_84983


namespace distance_to_river_l849_84966

theorem distance_to_river (d : ℝ) (h1 : ¬ (d ≥ 8)) (h2 : ¬ (d ≤ 7)) (h3 : ¬ (d ≤ 6)) : 7 < d ∧ d < 8 :=
by
  sorry

end distance_to_river_l849_84966


namespace ice_cream_tubs_eaten_l849_84993

-- Conditions
def number_of_pans := 2
def pieces_per_pan := 16
def percentage_eaten_second_pan := 0.75
def scoops_per_tub := 8
def scoops_per_guest := 2
def guests_not_eating_ala_mode := 4

-- Questions
def tubs_of_ice_cream_eaten : Nat :=
  sorry

theorem ice_cream_tubs_eaten :
  tubs_of_ice_cream_eaten = 6 := by
  sorry

end ice_cream_tubs_eaten_l849_84993


namespace distance_between_trees_l849_84998

theorem distance_between_trees (length_yard : ℕ) (num_trees : ℕ) (dist : ℕ) :
  length_yard = 275 → num_trees = 26 → dist = length_yard / (num_trees - 1) → dist = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  assumption

end distance_between_trees_l849_84998


namespace arithmetic_sequence_problem_l849_84928

variable (a : ℕ → ℤ) -- defining the sequence {a_n}
variable (S : ℕ → ℤ) -- defining the sum of the first n terms S_n

theorem arithmetic_sequence_problem (m : ℕ) (h1 : m > 1) 
  (h2 : a (m - 1) + a (m + 1) - a m ^ 2 = 0) 
  (h3 : S (2 * m - 1) = 38) 
  : m = 10 :=
sorry

end arithmetic_sequence_problem_l849_84928


namespace minimal_APR_bank_A_l849_84937

def nominal_interest_rate_A : Float := 0.05
def nominal_interest_rate_B : Float := 0.055
def nominal_interest_rate_C : Float := 0.06

def compounding_periods_A : ℕ := 4
def compounding_periods_B : ℕ := 2
def compounding_periods_C : ℕ := 12

def effective_annual_rate (nom_rate : Float) (n : ℕ) : Float :=
  (1 + nom_rate / n.toFloat)^n.toFloat - 1

def APR_A := effective_annual_rate nominal_interest_rate_A compounding_periods_A
def APR_B := effective_annual_rate nominal_interest_rate_B compounding_periods_B
def APR_C := effective_annual_rate nominal_interest_rate_C compounding_periods_C

theorem minimal_APR_bank_A :
  APR_A < APR_B ∧ APR_A < APR_C ∧ APR_A = 0.050945 :=
by
  sorry

end minimal_APR_bank_A_l849_84937


namespace option_B_is_correct_l849_84982

-- Definitions and Conditions
variable {Line : Type} {Plane : Type}
variable (m n : Line) (α β γ : Plane)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Conditions
axiom m_perp_β : perpendicular m β
axiom m_parallel_α : parallel m α

-- Statement to prove
theorem option_B_is_correct : perpendicular_planes α β :=
by
  sorry

end option_B_is_correct_l849_84982


namespace umbrellas_problem_l849_84978

theorem umbrellas_problem :
  ∃ (b r : ℕ), b = 36 ∧ r = 27 ∧ 
  b = (45 + r) / 2 ∧ 
  r = (45 + b) / 3 :=
by sorry

end umbrellas_problem_l849_84978


namespace clock1_runs_10_months_longer_l849_84945

noncomputable def battery_a_charge (C_B : ℝ) := 6 * C_B
noncomputable def clock1_total_charge (C_B : ℝ) := 2 * battery_a_charge C_B
noncomputable def clock2_total_charge (C_B : ℝ) := 2 * C_B
noncomputable def clock2_operating_time := 2
noncomputable def clock1_operating_time (C_B : ℝ) := clock1_total_charge C_B / C_B
noncomputable def operating_time_difference (C_B : ℝ) := clock1_operating_time C_B - clock2_operating_time

theorem clock1_runs_10_months_longer (C_B : ℝ) :
  operating_time_difference C_B = 10 :=
by
  unfold operating_time_difference clock1_operating_time clock2_operating_time clock1_total_charge battery_a_charge
  sorry

end clock1_runs_10_months_longer_l849_84945


namespace sum_of_coefficients_l849_84920

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℕ) (h₁ : (1 + x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) : 
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 := by
  sorry

end sum_of_coefficients_l849_84920


namespace pastries_sold_value_l849_84934

-- Define the number of cakes sold and the relationship between cakes and pastries
def number_of_cakes_sold := 78
def pastries_sold (C : Nat) := C + 76

-- State the theorem we want to prove
theorem pastries_sold_value : pastries_sold number_of_cakes_sold = 154 := by
  sorry

end pastries_sold_value_l849_84934


namespace nested_sqrt_eq_five_l849_84990

-- Define the infinite nested square root expression
def nested_sqrt : ℝ := sorry -- we assume the definition exists
-- Define the property it satisfies
theorem nested_sqrt_eq_five : nested_sqrt = 5 := by
  sorry

end nested_sqrt_eq_five_l849_84990


namespace sequence_problem_l849_84917

open Nat

theorem sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h : ∀ n : ℕ, 0 < n → S n + a n = 2 * n) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = 7 / 4 ∧ a 4 = 15 / 8 ∧ ∀ n : ℕ, 0 < n → a n = (2^n - 1) / (2^(n-1)) :=
by
  sorry

end sequence_problem_l849_84917


namespace solve_for_product_l849_84953

theorem solve_for_product (a b c d : ℚ) (h1 : 3 * a + 4 * b + 6 * c + 8 * d = 48)
                          (h2 : 4 * (d + c) = b) 
                          (h3 : 4 * b + 2 * c = a) 
                          (h4 : c - 2 = d) : 
                          a * b * c * d = -1032192 / 1874161 := 
by 
  sorry

end solve_for_product_l849_84953


namespace solve_quad_eq1_solve_quad_eq2_solve_quad_eq3_solve_quad_eq4_l849_84904

-- Problem 1: Prove the solutions to x^2 = 2
theorem solve_quad_eq1 : ∃ x : ℝ, x^2 = 2 ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by
  sorry

-- Problem 2: Prove the solutions to 4x^2 - 1 = 0
theorem solve_quad_eq2 : ∃ x : ℝ, 4 * x^2 - 1 = 0 ∧ (x = 1/2 ∨ x = -1/2) :=
by
  sorry

-- Problem 3: Prove the solutions to (x-1)^2 - 4 = 0
theorem solve_quad_eq3 : ∃ x : ℝ, (x - 1)^2 - 4 = 0 ∧ (x = 3 ∨ x = -1) :=
by
  sorry

-- Problem 4: Prove the solutions to 12 * (3 - x)^2 - 48 = 0
theorem solve_quad_eq4 : ∃ x : ℝ, 12 * (3 - x)^2 - 48 = 0 ∧ (x = 1 ∨ x = 5) :=
by
  sorry

end solve_quad_eq1_solve_quad_eq2_solve_quad_eq3_solve_quad_eq4_l849_84904


namespace train_speed_l849_84968

theorem train_speed (length : ℤ) (time : ℤ) 
  (h_length : length = 280) (h_time : time = 14) : 
  (length * 3600) / (time * 1000) = 72 := 
by {
  -- The proof would go here, this part is omitted as per instructions
  sorry
}

end train_speed_l849_84968


namespace value_of_g_neg3_l849_84942

def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem value_of_g_neg3 : g (-3) = 4 :=
by
  sorry

end value_of_g_neg3_l849_84942


namespace spring_compression_l849_84969

theorem spring_compression (s F : ℝ) (h : F = 16 * s^2) (hF : F = 4) : s = 0.5 :=
by
  sorry

end spring_compression_l849_84969


namespace max_value_S_n_l849_84985

open Nat

noncomputable def a_n (n : ℕ) : ℤ := 20 + (n - 1) * (-2)

noncomputable def S_n (n : ℕ) : ℤ := n * 20 + (n * (n - 1)) * (-2) / 2

theorem max_value_S_n : ∃ n : ℕ, S_n n = 110 :=
by
  sorry

end max_value_S_n_l849_84985


namespace rationalize_denominator_l849_84961

theorem rationalize_denominator (sqrt_98 : Real) (h : sqrt_98 = 7 * Real.sqrt 2) : 
  7 / sqrt_98 = Real.sqrt 2 / 2 := 
by 
  sorry

end rationalize_denominator_l849_84961


namespace problem_ACD_l849_84935

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 - x

theorem problem_ACD (a : ℝ) :
  (f a 0 = (2/3) ∧
  ¬(∀ x, f a x ≥ 0 → ((a ≥ 1) ∨ (a ≤ -1))) ∧
  (∃ x1 x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ∧
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0)) :=
sorry

end problem_ACD_l849_84935


namespace solve_quadratic_eq_l849_84931

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem solve_quadratic_eq :
  ∀ a b c x1 x2 : ℝ,
  a = 2 →
  b = -2 →
  c = -1 →
  quadratic_eq a b c x1 ∧ quadratic_eq a b c x2 →
  (x1 = (1 + Real.sqrt 3) / 2 ∧ x2 = (1 - Real.sqrt 3) / 2) :=
by
  intros a b c x1 x2 ha hb hc h
  sorry

end solve_quadratic_eq_l849_84931


namespace ivans_profit_l849_84988

def price_meat_per_kg : ℕ := 500
def kg_meat_sold : ℕ := 100
def price_eggs_per_dozen : ℕ := 50
def eggs_sold : ℕ := 20000
def annual_expenses : ℕ := 100000

def revenue_meat : ℕ := kg_meat_sold * price_meat_per_kg
def revenue_eggs : ℕ := eggs_sold * (price_eggs_per_dozen / 10)
def total_revenue : ℕ := revenue_meat + revenue_eggs

def profit : ℕ := total_revenue - annual_expenses

theorem ivans_profit : profit = 50000 := by
  sorry

end ivans_profit_l849_84988


namespace find_y_square_divisible_by_three_between_50_and_120_l849_84952

theorem find_y_square_divisible_by_three_between_50_and_120 :
  ∃ (y : ℕ), y = 81 ∧ (∃ (n : ℕ), y = n^2) ∧ (3 ∣ y) ∧ (50 < y) ∧ (y < 120) :=
by
  sorry

end find_y_square_divisible_by_three_between_50_and_120_l849_84952


namespace smallest_integer_equal_costs_l849_84923

-- Definitions based directly on conditions
def decimal_cost (n : ℕ) : ℕ :=
  (n.digits 10).sum * 2

def binary_cost (n : ℕ) : ℕ :=
  (n.digits 2).sum

-- The main statement to prove
theorem smallest_integer_equal_costs : ∃ n : ℕ, n < 2000 ∧ decimal_cost n = binary_cost n ∧ n = 255 :=
by 
  sorry

end smallest_integer_equal_costs_l849_84923


namespace sqrt_eq_cond_l849_84927

theorem sqrt_eq_cond (a b c : ℕ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
    (not_perfect_square_a : ¬(∃ n : ℕ, n * n = a)) (not_perfect_square_b : ¬(∃ n : ℕ, n * n = b))
    (not_perfect_square_c : ¬(∃ n : ℕ, n * n = c)) :
    (Real.sqrt a + Real.sqrt b = Real.sqrt c) →
    (2 * Real.sqrt (a * b) = c - (a + b) ∧ (∃ k : ℕ, a * b = k * k)) :=
sorry

end sqrt_eq_cond_l849_84927


namespace initial_time_between_maintenance_checks_l849_84991

theorem initial_time_between_maintenance_checks (x : ℝ) (h1 : 1.20 * x = 30) : x = 25 := by
  sorry

end initial_time_between_maintenance_checks_l849_84991


namespace arithmetic_sequence_ratio_l849_84965

variable {a_n : ℕ → ℤ} {S_n : ℕ → ℤ}
variable (d : ℤ)
variable (a1 a3 a4 : ℤ)
variable (h_geom : a3^2 = a1 * a4)
variable (h_seq : ∀ n, a_n (n+1) = a_n n + d)
variable (h_sum : ∀ n, S_n n = (n * (2 * a1 + (n - 1) * d)) / 2)

theorem arithmetic_sequence_ratio :
  (S_n 3 - S_n 2) / (S_n 5 - S_n 3) = 2 :=
by 
  sorry

end arithmetic_sequence_ratio_l849_84965


namespace mass_percentage_Al_in_AlI3_l849_84914

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_AlI3 : ℝ := molar_mass_Al + 3 * molar_mass_I

theorem mass_percentage_Al_in_AlI3 : 
  (molar_mass_Al / molar_mass_AlI3) * 100 = 6.62 := 
  sorry

end mass_percentage_Al_in_AlI3_l849_84914


namespace value_of_f_neg1_l849_84996

def f (x : ℝ) : ℝ := x^3 + x^2 - 2 * x

theorem value_of_f_neg1 : f (-1) = 2 := by
  sorry

end value_of_f_neg1_l849_84996


namespace rod_mass_equilibrium_l849_84910

variable (g : ℝ) (m1 : ℝ) (l : ℝ) (S : ℝ)

-- Given conditions
axiom m1_value : m1 = 1
axiom l_value  : l = 0.5
axiom S_value  : S = 0.1

-- The goal is to find m2 such that the equilibrium condition holds
theorem rod_mass_equilibrium (m2 : ℝ) :
  (m1 * S = m2 * l) → m2 = 0.2 :=
by
  sorry

end rod_mass_equilibrium_l849_84910


namespace arithmetic_geometric_sequence_l849_84922

/-- Given:
  * 1, a₁, a₂, 4 form an arithmetic sequence
  * 1, b₁, b₂, b₃, 4 form a geometric sequence
Prove that:
  (a₁ + a₂) / b₂ = 5 / 2
-/
theorem arithmetic_geometric_sequence (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (h_arith : 2 * a₁ = 1 + a₂ ∧ 2 * a₂ = a₁ + 4)
  (h_geom : b₁ * b₁ = b₂ ∧ b₁ * b₂ = b₃ ∧ b₂ * b₂ = b₃ * 4) :
  (a₁ + a₂) / b₂ = 5 / 2 :=
sorry

end arithmetic_geometric_sequence_l849_84922


namespace no_fixed_points_range_l849_84939

def no_fixed_points (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 ≠ x

theorem no_fixed_points_range (a : ℝ) : no_fixed_points a ↔ -1 < a ∧ a < 3 := by
  sorry

end no_fixed_points_range_l849_84939


namespace kiran_has_105_l849_84958

theorem kiran_has_105 
  (R G K L : ℕ) 
  (ratio_rg : 6 * G = 7 * R)
  (ratio_gk : 6 * K = 15 * G)
  (R_value : R = 36) : 
  K = 105 :=
by
  sorry

end kiran_has_105_l849_84958


namespace required_equation_l849_84933

-- Define the given lines
def line1 (x y : ℝ) : Prop := 2 * x - y = 0
def line2 (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the equation to be proven for the line through the intersection point and perpendicular to perp_line
def required_line (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Define the predicate that states a point (2, 4) lies on line1 and line2
def point_intersect (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- The main theorem to be proven in Lean 4
theorem required_equation : 
  point_intersect 2 4 ∧ perp_line 2 4 → required_line 2 4 := by
  sorry

end required_equation_l849_84933


namespace largest_y_coordinate_l849_84963

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 :=
by
  intro h
  -- This is where the proofs steps would go if required.
  sorry

end largest_y_coordinate_l849_84963


namespace linen_tablecloth_cost_l849_84962

def num_tables : ℕ := 20
def cost_per_place_setting : ℕ := 10
def num_place_settings_per_table : ℕ := 4
def cost_per_rose : ℕ := 5
def num_roses_per_centerpiece : ℕ := 10
def cost_per_lily : ℕ := 4
def num_lilies_per_centerpiece : ℕ := 15
def total_decoration_cost : ℕ := 3500

theorem linen_tablecloth_cost :
  (total_decoration_cost - (num_tables * num_place_settings_per_table * cost_per_place_setting + num_tables * (num_roses_per_centerpiece * cost_per_rose + num_lilies_per_centerpiece * cost_per_lily))) / num_tables = 25 :=
  sorry

end linen_tablecloth_cost_l849_84962


namespace jack_afternoon_emails_l849_84974

theorem jack_afternoon_emails : 
  ∀ (morning_emails afternoon_emails : ℕ), 
  morning_emails = 6 → 
  afternoon_emails = morning_emails + 2 → 
  afternoon_emails = 8 := 
by
  intros morning_emails afternoon_emails hm ha
  rw [hm] at ha
  exact ha

end jack_afternoon_emails_l849_84974


namespace circle_radius_five_eq_neg_eight_l849_84960

theorem circle_radius_five_eq_neg_eight (c : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ∧ (x + 4)^2 + (y + 1)^2 = 25) → c = -8 :=
by
  sorry

end circle_radius_five_eq_neg_eight_l849_84960


namespace fractional_to_decimal_l849_84986

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l849_84986


namespace larger_gate_width_is_10_l849_84924

-- Define the conditions as constants
def garden_length : ℝ := 225
def garden_width : ℝ := 125
def small_gate_width : ℝ := 3
def total_fencing_length : ℝ := 687

-- Define the perimeter function for a rectangle
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

-- Define the width of the larger gate
def large_gate_width : ℝ :=
  let total_perimeter := perimeter garden_length garden_width
  let remaining_fencing := total_perimeter - total_fencing_length
  remaining_fencing - small_gate_width

-- State the theorem
theorem larger_gate_width_is_10 : large_gate_width = 10 := by
  -- skipping proof part
  sorry

end larger_gate_width_is_10_l849_84924


namespace tom_tickets_l849_84944

theorem tom_tickets :
  let tickets_whack_a_mole := 32
  let tickets_skee_ball := 25
  let tickets_spent_on_hat := 7
  let total_tickets := tickets_whack_a_mole + tickets_skee_ball
  let tickets_left := total_tickets - tickets_spent_on_hat
  tickets_left = 50 :=
by
  sorry

end tom_tickets_l849_84944


namespace amber_max_ounces_l849_84997

theorem amber_max_ounces :
  ∀ (money : ℝ) (candy_cost : ℝ) (candy_ounces : ℝ) (chips_cost : ℝ) (chips_ounces : ℝ),
    money = 7 →
    candy_cost = 1 →
    candy_ounces = 12 →
    chips_cost = 1.4 →
    chips_ounces = 17 →
    max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces) = 85 :=
by
  intros money candy_cost candy_ounces chips_cost chips_ounces
  intros h_money h_candy_cost h_candy_ounces h_chips_cost h_chips_ounces
  sorry

end amber_max_ounces_l849_84997


namespace fraction_of_sum_l849_84970

theorem fraction_of_sum (l : List ℝ) (hl : l.length = 51)
  (n : ℝ) (hn : n ∈ l)
  (h : n = 7 * (l.erase n).sum / 50) :
  n / l.sum = 7 / 57 := by
  sorry

end fraction_of_sum_l849_84970


namespace galya_number_l849_84994

theorem galya_number (N k : ℤ) (h : (k - N + 1 = k - 7729)) : N = 7730 := 
by
  sorry

end galya_number_l849_84994


namespace revenue_after_fall_is_correct_l849_84900

variable (originalRevenue : ℝ) (percentageDecrease : ℝ)

theorem revenue_after_fall_is_correct :
    originalRevenue = 69 ∧ percentageDecrease = 39.130434782608695 →
    originalRevenue - (originalRevenue * (percentageDecrease / 100)) = 42 := by
  intro h
  rcases h with ⟨h1, h2⟩
  sorry

end revenue_after_fall_is_correct_l849_84900


namespace find_a_g_range_l849_84940

noncomputable def f (x a : ℝ) : ℝ := x^2 + 4 * a * x + 2 * a + 6
noncomputable def g (a : ℝ) : ℝ := 2 - a * |a - 1|

theorem find_a (x a : ℝ) :
  (∀ x, f x a ≥ 0) ∧ (∀ x, f x a = 0 → x^2 + 4 * a * x + 2 * a + 6 = 0) ↔ (a = -1 ∨ a = 3 / 2) :=
  sorry

theorem g_range :
  (∀ x, f x a ≥ 0) ∧ (-1 ≤ a ∧ a ≤ 3/2) → (∀ a, (5 / 4 ≤ g a ∧ g a ≤ 4)) :=
  sorry

end find_a_g_range_l849_84940


namespace intersection_complement_correct_l849_84912

open Set

def A : Set ℕ := {x | (x + 4) * (x - 5) ≤ 0}
def B : Set ℕ := {x | x < 2}
def U : Set ℕ := { x | True }

theorem intersection_complement_correct :
  (A ∩ (U \ B)) = {x | x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5} :=
by
  sorry

end intersection_complement_correct_l849_84912


namespace calculation_l849_84956

noncomputable def seq (n : ℕ) : ℕ → ℚ := sorry

axiom cond1 : ∀ (n : ℕ), seq (n + 1) - 2 * seq n = 0
axiom cond2 : ∀ (n : ℕ), seq n ≠ 0

theorem calculation :
  (2 * seq 1 + seq 2) / (seq 3 + seq 5) = 1 / 5 :=
  sorry

end calculation_l849_84956


namespace find_triplets_l849_84946

theorem find_triplets (x y z : ℕ) (h1 : x ≤ y) (h2 : x^2 + y^2 = 3 * 2016^z + 77) :
  (x, y, z) = (4, 8, 0) ∨ (x, y, z) = (14, 77, 1) ∨ (x, y, z) = (35, 70, 1) :=
  sorry

end find_triplets_l849_84946


namespace parallel_lines_a_unique_l849_84999

theorem parallel_lines_a_unique (a : ℝ) :
  (∀ x y : ℝ, x + (a + 1) * y + (a^2 - 1) = 0 → x + 2 * y = 0 → -a / 2 = -1 / (a + 1)) →
  a = -2 :=
by
  sorry

end parallel_lines_a_unique_l849_84999


namespace conference_hall_initial_people_l849_84913

theorem conference_hall_initial_people (x : ℕ)  
  (h1 : 3 ∣ x) 
  (h2 : 4 ∣ (2 * x / 3))
  (h3 : (x / 2) = 27) : 
  x = 54 := 
by 
  sorry

end conference_hall_initial_people_l849_84913


namespace evaluate_expression_l849_84955

theorem evaluate_expression :
  (18 : ℝ) / (14 * 5.3) = (1.8 : ℝ) / 7.42 :=
by
  sorry

end evaluate_expression_l849_84955


namespace clothing_value_is_correct_l849_84981

-- Define the value of the clothing to be C and the correct answer
def value_of_clothing (C : ℝ) : Prop :=
  (C + 2) = (7 / 12) * (C + 10)

-- Statement of the problem
theorem clothing_value_is_correct :
  ∃ (C : ℝ), value_of_clothing C ∧ C = 46 / 5 :=
by {
  sorry
}

end clothing_value_is_correct_l849_84981


namespace books_sold_l849_84992

theorem books_sold {total_books sold_fraction left_fraction : ℕ} (h_total : total_books = 9900)
    (h_fraction : left_fraction = 4/6) (h_sold : sold_fraction = 1 - left_fraction) : 
  (sold_fraction * total_books) = 3300 := 
  by 
  sorry

end books_sold_l849_84992


namespace inverse_89_mod_90_l849_84989

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  sorry -- proof goes here

end inverse_89_mod_90_l849_84989


namespace min_sum_a1_a2_l849_84936

-- Define the condition predicate for the sequence
def satisfies_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n + 2009) / (1 + a (n + 1))

-- State the main problem as a theorem in Lean 4
theorem min_sum_a1_a2 (a : ℕ → ℕ) (h_seq : satisfies_seq a) (h_pos : ∀ n, a n > 0) :
  a 1 * a 2 = 2009 → a 1 + a 2 = 90 :=
sorry

end min_sum_a1_a2_l849_84936


namespace geometric_sequence_sum_l849_84951

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h₀ : q > 1)
  (h₁ : ∀ n : ℕ, a (n + 1) = a n * q)
  (h₂ : ∀ x : ℝ, 4 * x^2 - 8 * x + 3 = 0 → (x = a 2005 ∨ x = a 2006)) : 
  a 2007 + a 2008 = 18 := 
sorry

end geometric_sequence_sum_l849_84951


namespace cos_value_third_quadrant_l849_84975

theorem cos_value_third_quadrant (x : Real) (h1 : Real.sin x = -1 / 3) (h2 : π < x ∧ x < 3 * π / 2) : 
  Real.cos x = -2 * Real.sqrt 2 / 3 :=
by
  sorry

end cos_value_third_quadrant_l849_84975


namespace sum_of_squares_l849_84954

open Int

theorem sum_of_squares (p q r s t u : ℤ) (h : ∀ x : ℤ, 343 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3506 :=
sorry

end sum_of_squares_l849_84954


namespace max_profit_at_60_l849_84921

variable (x : ℕ) (y W : ℝ)

def charter_fee : ℝ := 15000
def max_group_size : ℕ := 75

def ticket_price (x : ℕ) : ℝ :=
  if x ≤ 30 then 900
  else if 30 < x ∧ x ≤ max_group_size then -10 * (x - 30) + 900
  else 0

def profit (x : ℕ) : ℝ :=
  if x ≤ 30 then 900 * x - charter_fee
  else if 30 < x ∧ x ≤ max_group_size then (-10 * x + 1200) * x - charter_fee
  else 0

theorem max_profit_at_60 : x = 60 → profit x = 21000 := by
  sorry

end max_profit_at_60_l849_84921


namespace solve_abs_equation_l849_84979

theorem solve_abs_equation (x : ℝ) (h : abs (x - 20) + abs (x - 18) = abs (2 * x - 36)) : x = 19 :=
sorry

end solve_abs_equation_l849_84979


namespace coplanar_points_scalar_eq_l849_84987

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D O : V) (k : ℝ)

theorem coplanar_points_scalar_eq:
  (3 • (A - O) - 2 • (B - O) + 5 • (C - O) + k • (D - O) = (0 : V)) →
  k = -6 :=
by sorry

end coplanar_points_scalar_eq_l849_84987


namespace sum_of_squares_of_medians_l849_84964

-- Define the components of the triangle
variables (a b c : ℝ)

-- Define the medians of the triangle
variables (s_a s_b s_c : ℝ)

-- State the theorem
theorem sum_of_squares_of_medians (h1 : s_a^2 + s_b^2 + s_c^2 = (3 / 4) * (a^2 + b^2 + c^2)) : 
  s_a^2 + s_b^2 + s_c^2 = (3 / 4) * (a^2 + b^2 + c^2) :=
by {
  -- The proof goes here
  sorry
}

end sum_of_squares_of_medians_l849_84964


namespace find_incorrect_observation_l849_84905

theorem find_incorrect_observation (n : ℕ) (initial_mean new_mean : ℝ) (correct_value incorrect_value : ℝ) (observations_count : ℕ)
  (h1 : observations_count = 50)
  (h2 : initial_mean = 36)
  (h3 : new_mean = 36.5)
  (h4 : correct_value = 44) :
  incorrect_value = 19 :=
by
  sorry

end find_incorrect_observation_l849_84905


namespace find_a_range_l849_84901

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 1 then (a + 3) * x - 5 else 2 * a / x

theorem find_a_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2) → -2 ≤ a ∧ a < 0 :=
by
  sorry

end find_a_range_l849_84901


namespace number_of_toys_sold_l849_84947

theorem number_of_toys_sold (n : ℕ) 
  (sell_price : ℕ) (gain_price : ℕ) (cost_price_per_toy : ℕ) :
  sell_price = 27300 → 
  gain_price = 3 * cost_price_per_toy → 
  cost_price_per_toy = 1300 →
  n * cost_price_per_toy + gain_price = sell_price → 
  n = 18 :=
by sorry

end number_of_toys_sold_l849_84947


namespace inequality_proof_l849_84907

open Real

-- Define the conditions
def conditions (a b c : ℝ) := (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a * b * c = 1)

-- Express the inequality we need to prove
def inequality (a b c : ℝ) :=
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1

-- Statement of the theorem
theorem inequality_proof (a b c : ℝ) (h : conditions a b c) : inequality a b c :=
by {
  sorry
}

end inequality_proof_l849_84907


namespace area_of_triangle_ABC_l849_84977

theorem area_of_triangle_ABC
  (A B C : ℝ)
  (a b c : ℝ)
  (sin_C_eq : Real.sin C = Real.sqrt 3 / 3)
  (sin_CBA_eq : Real.sin C + Real.sin (B - A) = Real.sin (2 * A))
  (a_minus_b_eq : a - b = 3 - Real.sqrt 6)
  (c_eq : c = Real.sqrt 3) :
  1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 2 / 2 := sorry

end area_of_triangle_ABC_l849_84977


namespace combined_weight_difference_l849_84984

-- Define the weights of the textbooks
def chemistry_weight : ℝ := 7.125
def geometry_weight : ℝ := 0.625
def calculus_weight : ℝ := 5.25
def biology_weight : ℝ := 3.75

-- Define the problem statement that needs to be proven
theorem combined_weight_difference :
  ((calculus_weight + biology_weight) - (chemistry_weight - geometry_weight)) = 2.5 :=
by
  sorry

end combined_weight_difference_l849_84984


namespace all_tutors_work_together_in_90_days_l849_84950

theorem all_tutors_work_together_in_90_days :
  lcm 5 (lcm 6 (lcm 9 10)) = 90 := by
  sorry

end all_tutors_work_together_in_90_days_l849_84950


namespace solution_l849_84915

def problem_statement : Prop :=
  (3025 - 2880) ^ 2 / 225 = 93

theorem solution : problem_statement :=
by {
  sorry
}

end solution_l849_84915


namespace find_A_in_terms_of_B_and_C_l849_84916

noncomputable def f (A B : ℝ) (x : ℝ) := A * x - 3 * B^2
noncomputable def g (B C : ℝ) (x : ℝ) := B * x + C

theorem find_A_in_terms_of_B_and_C (A B C : ℝ) (h : B ≠ 0) (h1 : f A B (g B C 1) = 0) : A = 3 * B^2 / (B + C) :=
by sorry

end find_A_in_terms_of_B_and_C_l849_84916


namespace polar_to_rect_l849_84948

open Real 

theorem polar_to_rect (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 3 * π / 4) : 
  (r * cos θ, r * sin θ) = (-3 / Real.sqrt 2, 3 / Real.sqrt 2) :=
by
  -- Optional step: you can introduce the variables as they have already been proved using the given conditions
  have hr : r = 3 := h_r
  have hθ : θ = 3 * π / 4 := h_θ
  -- Goal changes according to the values of r and θ derived from the conditions
  sorry

end polar_to_rect_l849_84948


namespace bus_speed_l849_84906

noncomputable def radius : ℝ := 35 / 100  -- Radius in meters
noncomputable def rpm : ℝ := 500.4549590536851

noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def distance_in_one_minute : ℝ := circumference * rpm
noncomputable def distance_in_km_per_hour : ℝ := (distance_in_one_minute / 1000) * 60

theorem bus_speed :
  distance_in_km_per_hour = 66.037 :=
by
  -- The proof is skipped here as it is not required
  sorry

end bus_speed_l849_84906


namespace percentage_of_local_arts_students_is_50_l849_84957

-- Definitions
def total_students_arts := 400
def total_students_science := 100
def total_students_commerce := 120
def percent_local_science := 25 / 100
def percent_local_commerce := 85 / 100
def total_locals := 327

-- Problem statement in Lean
theorem percentage_of_local_arts_students_is_50
  (x : ℕ) -- Percentage of local arts students as a natural number
  (h1 : percent_local_science * total_students_science = 25)
  (h2 : percent_local_commerce * total_students_commerce = 102)
  (h3 : (x / 100 : ℝ) * total_students_arts + 25 + 102 = total_locals) :
  x = 50 :=
sorry

end percentage_of_local_arts_students_is_50_l849_84957


namespace max_complete_bouquets_l849_84938

-- Definitions based on conditions
def total_roses := 20
def total_lilies := 15
def total_daisies := 10

def wilted_roses := 12
def wilted_lilies := 8
def wilted_daisies := 5

def roses_per_bouquet := 3
def lilies_per_bouquet := 2
def daisies_per_bouquet := 1

-- Calculation of remaining flowers
def remaining_roses := total_roses - wilted_roses
def remaining_lilies := total_lilies - wilted_lilies
def remaining_daisies := total_daisies - wilted_daisies

-- Proof statement
theorem max_complete_bouquets : 
  min
    (remaining_roses / roses_per_bouquet)
    (min (remaining_lilies / lilies_per_bouquet) (remaining_daisies / daisies_per_bouquet)) = 2 :=
by
  sorry

end max_complete_bouquets_l849_84938


namespace system_has_three_real_k_with_unique_solution_l849_84959

theorem system_has_three_real_k_with_unique_solution :
  (∃ (k : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = 2 * k^2 ∧ k * x - y = 2 * k) → (x, y) = (0, 0)) → 
  ∃ (k : ℝ), ∃ (x y : ℝ), (x^2 + y^2 = 2 * k^2 ∧ k * x - y = 2 * k) :=
by
  sorry

end system_has_three_real_k_with_unique_solution_l849_84959


namespace scallops_cost_l849_84932

-- define the conditions
def scallops_per_pound : ℝ := 8
def cost_per_pound : ℝ := 24
def scallops_per_person : ℝ := 2
def number_of_people : ℝ := 8

-- the question
theorem scallops_cost : (scallops_per_person * number_of_people / scallops_per_pound) * cost_per_pound = 48 := by 
  sorry

end scallops_cost_l849_84932


namespace warehouse_bins_total_l849_84967

theorem warehouse_bins_total (x : ℕ) (h1 : 12 * 20 + x * 15 = 510) : 12 + x = 30 :=
by
  sorry

end warehouse_bins_total_l849_84967


namespace alcohol_water_ratio_l849_84930

variable {r s V1 : ℝ}

theorem alcohol_water_ratio 
  (h1 : r > 0) 
  (h2 : s > 0) 
  (h3 : V1 > 0) :
  let alcohol_in_JarA := 2 * r * V1 / (r + 1) + V1
  let water_in_JarA := 2 * V1 / (r + 1)
  let alcohol_in_JarB := 3 * s * V1 / (s + 1)
  let water_in_JarB := 3 * V1 / (s + 1)
  let total_alcohol := alcohol_in_JarA + alcohol_in_JarB
  let total_water := water_in_JarA + water_in_JarB
  (total_alcohol / total_water) = 
  ((2 * r / (r + 1) + 1 + 3 * s / (s + 1)) / (2 / (r + 1) + 3 / (s + 1))) :=
by
  sorry

end alcohol_water_ratio_l849_84930


namespace min_max_values_f_l849_84949

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_f :
  ∃ (a b : ℝ), a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧ 
                ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a ∧ f x ≤ b :=
by
  sorry

end min_max_values_f_l849_84949


namespace rex_cards_left_l849_84980

-- Definitions
def nicole_cards : ℕ := 400
def cindy_cards : ℕ := 2 * nicole_cards
def combined_total : ℕ := nicole_cards + cindy_cards
def rex_cards : ℕ := combined_total / 2
def people_count : ℕ := 4
def cards_per_person : ℕ := rex_cards / people_count

-- Proof statement
theorem rex_cards_left : cards_per_person = 150 := by
  sorry

end rex_cards_left_l849_84980


namespace monotonic_increasing_on_interval_l849_84919

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - a * Real.log x

theorem monotonic_increasing_on_interval (a : ℝ) :
  (∀ x > 1, 2 * x - a / x ≥ 0) → a ≤ 2 :=
sorry

end monotonic_increasing_on_interval_l849_84919


namespace price_reduction_l849_84972

variable (x : ℝ)

theorem price_reduction :
  28 * (1 - x) * (1 - x) = 16 :=
sorry

end price_reduction_l849_84972


namespace max_value_expression_l849_84943

variable (x y z : ℝ)

theorem max_value_expression (h : x^2 + y^2 + z^2 = 4) :
  (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 ≤ 28 :=
sorry

end max_value_expression_l849_84943


namespace max_stickers_one_student_l849_84902

def total_students : ℕ := 25
def mean_stickers : ℕ := 4
def total_stickers := total_students * mean_stickers
def minimum_stickers_per_student : ℕ := 1
def minimum_stickers_taken_by_24_students := (total_students - 1) * minimum_stickers_per_student

theorem max_stickers_one_student : 
  total_stickers - minimum_stickers_taken_by_24_students = 76 := by
  sorry

end max_stickers_one_student_l849_84902


namespace exists_sum_of_two_squares_l849_84911

theorem exists_sum_of_two_squares (n : ℕ) (h₁ : n > 10000) : 
  ∃ m : ℕ, (∃ a b : ℕ, m = a^2 + b^2) ∧ 0 < m - n ∧ m - n < 3 * Real.sqrt n := 
sorry

end exists_sum_of_two_squares_l849_84911


namespace total_walking_time_l849_84918

open Nat

def walking_time (distance speed : ℕ) : ℕ :=
distance / speed

def number_of_rests (distance : ℕ) : ℕ :=
(distance / 10) - 1

def resting_time_in_minutes (rests : ℕ) : ℕ :=
rests * 5

def resting_time_in_hours (rest_time : ℕ) : ℚ :=
rest_time / 60

def total_time (walking_time resting_time : ℚ) : ℚ :=
walking_time + resting_time

theorem total_walking_time (distance speed : ℕ) (rest_per_10 : ℕ) (rest_time : ℕ) :
  speed = 10 →
  rest_per_10 = 10 →
  rest_time = 5 →
  distance = 50 →
  total_time (walking_time distance speed) (resting_time_in_hours (resting_time_in_minutes (number_of_rests distance))) = 5 + 1 / 3 :=
sorry

end total_walking_time_l849_84918


namespace random_variable_prob_l849_84903

theorem random_variable_prob (n : ℕ) (h : (3 : ℝ) / n = 0.3) : n = 10 :=
sorry

end random_variable_prob_l849_84903


namespace sqrt_operation_l849_84973

def operation (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt_operation (sqrt5 : ℝ) (h : sqrt5 = Real.sqrt 5) : 
  operation sqrt5 sqrt5 = 20 := by
  sorry

end sqrt_operation_l849_84973


namespace no_positive_integer_solutions_l849_84941

theorem no_positive_integer_solutions (x y : ℕ) (h : x > 0 ∧ y > 0) : x^2 + (x+1)^2 ≠ y^4 + (y+1)^4 :=
by
  intro h1
  sorry

end no_positive_integer_solutions_l849_84941


namespace find_polynomial_l849_84976

theorem find_polynomial (P : ℝ → ℝ) (h_poly : ∀ a b c : ℝ, ab + bc + ca = 0 → P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) :
  ∃ r s : ℝ, ∀ x : ℝ, P x = r * x^4 + s * x^2 :=
sorry

end find_polynomial_l849_84976


namespace cos_double_angle_l849_84908

theorem cos_double_angle (α : ℝ) (h : Real.cos (π - α) = -3/5) : Real.cos (2 * α) = -7/25 :=
  sorry

end cos_double_angle_l849_84908


namespace ratio_of_routes_l849_84926

-- Definitions of m and n
def m : ℕ := 2 
def n : ℕ := 6

-- Theorem statement
theorem ratio_of_routes (m_positive : m > 0) : n / m = 3 := by
  sorry

end ratio_of_routes_l849_84926
