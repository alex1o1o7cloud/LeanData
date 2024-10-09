import Mathlib

namespace f_of_x_squared_domain_l209_20916

structure FunctionDomain (f : ℝ → ℝ) :=
  (domain : Set ℝ)
  (domain_eq : domain = Set.Icc 0 1)

theorem f_of_x_squared_domain (f : ℝ → ℝ) (h : FunctionDomain f) :
  FunctionDomain (fun x => f (x ^ 2)) :=
{
  domain := Set.Icc (-1) 1,
  domain_eq := sorry
}

end f_of_x_squared_domain_l209_20916


namespace solve_for_n_l209_20919

theorem solve_for_n : ∃ n : ℤ, 3^3 - 5 = 4^2 + n ∧ n = 6 := 
by
  use 6
  sorry

end solve_for_n_l209_20919


namespace rectangle_width_l209_20928

theorem rectangle_width (side_length_square : ℕ) (length_rectangle : ℕ) (area_equal : side_length_square * side_length_square = length_rectangle * w) : w = 4 := by
  sorry

end rectangle_width_l209_20928


namespace total_games_played_l209_20917

theorem total_games_played (games_attended games_missed : ℕ) 
  (h_attended : games_attended = 395) 
  (h_missed : games_missed = 469) : 
  games_attended + games_missed = 864 := 
by
  sorry

end total_games_played_l209_20917


namespace square_side_length_s2_l209_20975

theorem square_side_length_s2 (s1 s2 s3 : ℕ)
  (h1 : s1 + s2 + s3 = 3322)
  (h2 : s1 - s2 + s3 = 2020) :
  s2 = 651 :=
by sorry

end square_side_length_s2_l209_20975


namespace construct_unit_segment_l209_20942

-- Definitions of the problem
variable (a b : ℝ)

-- Parabola definition
def parabola (x : ℝ) : ℝ := x^2 + a * x + b

-- Statement of the problem in Lean 4
theorem construct_unit_segment
  (h : ∃ x y : ℝ, parabola a b x = y) :
  ∃ (u v : ℝ), abs (u - v) = 1 :=
sorry

end construct_unit_segment_l209_20942


namespace alice_profit_l209_20946

-- Define the variables and conditions
def total_bracelets : ℕ := 52
def material_cost : ℝ := 3.00
def bracelets_given_away : ℕ := 8
def sale_price : ℝ := 0.25

-- Calculate the number of bracelets sold
def bracelets_sold : ℕ := total_bracelets - bracelets_given_away

-- Calculate the revenue from selling the bracelets
def revenue : ℝ := bracelets_sold * sale_price

-- Define the profit as revenue minus material cost
def profit : ℝ := revenue - material_cost

-- The statement to prove
theorem alice_profit : profit = 8.00 := 
by
  sorry

end alice_profit_l209_20946


namespace ann_fare_90_miles_l209_20908

-- Define the conditions as given in the problem
def fare (distance : ℕ) : ℕ := 30 + distance * 2

-- Theorem statement
theorem ann_fare_90_miles : fare 90 = 210 := by
  sorry

end ann_fare_90_miles_l209_20908


namespace max_clouds_through_planes_l209_20970

-- Define the problem parameters and conditions
def max_clouds (n : ℕ) : ℕ :=
  n + 1

-- Mathematically equivalent proof problem statement in Lean 4
theorem max_clouds_through_planes : max_clouds 10 = 11 :=
  by
    sorry  -- Proof skipped as required

end max_clouds_through_planes_l209_20970


namespace correct_answers_is_36_l209_20986

noncomputable def num_correct_answers (c w : ℕ) : Prop :=
  (c + w = 50) ∧ (4 * c - w = 130)

theorem correct_answers_is_36 (c w : ℕ) (h : num_correct_answers c w) : c = 36 :=
by
  sorry

end correct_answers_is_36_l209_20986


namespace local_minimum_f_when_k2_l209_20921

noncomputable def f (k : ℕ) (x : ℝ) : ℝ := (Real.exp x - 1) * (x - 1) ^ k

theorem local_minimum_f_when_k2 : ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f 2 x ≥ f 2 1 :=
by
  -- the question asks to prove that the function attains a local minimum at x = 1 when k = 2
  sorry

end local_minimum_f_when_k2_l209_20921


namespace wholesale_price_l209_20954

theorem wholesale_price (RP SP W : ℝ) (h1 : RP = 120)
  (h2 : SP = 0.9 * RP)
  (h3 : SP = W + 0.2 * W) : W = 90 :=
by
  sorry

end wholesale_price_l209_20954


namespace initial_men_l209_20990

/-- Initial number of men M being catered for. 
Proof that the initial number of men M is equal to 760 given the conditions. -/
theorem initial_men (M : ℕ)
  (H1 : 22 * M = 20 * M)
  (H2 : 2 * (M + 3040) = M) : M = 760 := 
sorry

end initial_men_l209_20990


namespace PedoeInequalityHolds_l209_20953

noncomputable def PedoeInequality 
  (a b c a1 b1 c1 : ℝ) (Δ Δ1 : ℝ) :
  Prop :=
  a^2 * (b1^2 + c1^2 - a1^2) + 
  b^2 * (c1^2 + a1^2 - b1^2) + 
  c^2 * (a1^2 + b1^2 - c1^2) >= 16 * Δ * Δ1 

axiom areas_triangle 
  (a b c : ℝ) : ℝ 

axiom areas_triangle1 
  (a1 b1 c1 : ℝ) : ℝ 

theorem PedoeInequalityHolds 
  (a b c a1 b1 c1 : ℝ) 
  (Δ := areas_triangle a b c) 
  (Δ1 := areas_triangle1 a1 b1 c1) :
  PedoeInequality a b c a1 b1 c1 Δ Δ1 :=
sorry

end PedoeInequalityHolds_l209_20953


namespace polygon_sides_eq_eight_l209_20924

theorem polygon_sides_eq_eight (n : ℕ) 
  (h_diff : (n - 2) * 180 - 360 = 720) :
  n = 8 := 
by 
  sorry

end polygon_sides_eq_eight_l209_20924


namespace find_ratio_l209_20927

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a n > 0 ∧ a (n+1) / a n = a 1 / a 0

def forms_arithmetic_sequence (a1 a3_half a2_times_two : ℝ) : Prop :=
  a3_half = (a1 + a2_times_two) / 2

theorem find_ratio (a : ℕ → ℝ) (h_geom : is_geometric_sequence a)
  (h_arith : forms_arithmetic_sequence (a 1) (1/2 * a 3) (2 * a 2)) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 :=
sorry

end find_ratio_l209_20927


namespace solve_for_m_l209_20911

theorem solve_for_m (m α : ℝ) (h1 : Real.tan α = m / 3) (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
by
  sorry

end solve_for_m_l209_20911


namespace distinct_factors_of_product_l209_20902

theorem distinct_factors_of_product (m a b d : ℕ) (hm : m ≥ 1) (ha : m^2 < a ∧ a < m^2 + m)
  (hb : m^2 < b ∧ b < m^2 + m) (hab : a ≠ b) (hd : d ∣ (a * b)) (hd_range: m^2 < d ∧ d < m^2 + m) :
  d = a ∨ d = b :=
sorry

end distinct_factors_of_product_l209_20902


namespace cos_triple_angle_l209_20935

theorem cos_triple_angle (x θ : ℝ) (h : x = Real.cos θ) : Real.cos (3 * θ) = 4 * x^3 - 3 * x :=
by
  sorry

end cos_triple_angle_l209_20935


namespace order_of_expressions_l209_20959

theorem order_of_expressions (k : ℕ) (hk : k > 4) : (k + 2) < (2 * k) ∧ (2 * k) < (k^2) ∧ (k^2) < (2^k) := by
  sorry

end order_of_expressions_l209_20959


namespace jade_statue_ratio_l209_20950

/-!
Nancy carves statues out of jade. A giraffe statue takes 120 grams of jade and sells for $150.
An elephant statue sells for $350. Nancy has 1920 grams of jade, and the revenue from selling all
elephant statues is $400 more than selling all giraffe statues.
Prove that the ratio of the amount of jade used for an elephant statue to the amount used for a
giraffe statue is 2.
-/

theorem jade_statue_ratio
  (g_grams : ℕ := 120) -- grams of jade for a giraffe statue
  (g_price : ℕ := 150) -- price of a giraffe statue
  (e_price : ℕ := 350) -- price of an elephant statue
  (total_jade : ℕ := 1920) -- total grams of jade Nancy has
  (additional_revenue : ℕ := 400) -- additional revenue from elephant statues
  (r : ℕ) -- ratio of jade usage of elephant to giraffe statue
  (h : total_jade / g_grams * g_price + additional_revenue = (total_jade / (g_grams * r)) * e_price) :
  r = 2 :=
sorry

end jade_statue_ratio_l209_20950


namespace machines_in_first_scenario_l209_20988

theorem machines_in_first_scenario :
  ∃ M : ℕ, (∀ (units1 units2 : ℕ) (hours1 hours2 : ℕ),
    units1 = 20 ∧ hours1 = 10 ∧ units2 = 200 ∧ hours2 = 25 ∧
    (M * units1 / hours1 = 20 * units2 / hours2)) → M = 5 :=
by
  sorry

end machines_in_first_scenario_l209_20988


namespace least_number_to_subtract_l209_20965

theorem least_number_to_subtract (n : ℕ) (h : n = 427398) : ∃ k : ℕ, (n - k) % 11 = 0 ∧ k = 4 :=
by
  sorry

end least_number_to_subtract_l209_20965


namespace ice_palace_steps_l209_20930

theorem ice_palace_steps (time_for_20_steps total_time : ℕ) (h1 : time_for_20_steps = 120) (h2 : total_time = 180) : 
  total_time * 20 / time_for_20_steps = 30 := by
  have time_per_step : ℕ := time_for_20_steps / 20
  have total_steps : ℕ := total_time / time_per_step
  sorry

end ice_palace_steps_l209_20930


namespace find_number_l209_20977

theorem find_number (a b N : ℕ) (h1 : b = 7) (h2 : b - a = 2) (h3 : a * b = 2 * (a + b) + N) : N = 11 :=
  sorry

end find_number_l209_20977


namespace max_frac_sum_l209_20901

theorem max_frac_sum (n a b c d : ℕ) (hn : 1 < n) (hab : 0 < a) (hcd : 0 < c)
    (hfrac : (a / b) + (c / d) < 1) (hsum : a + c ≤ n) :
    (∃ (b_val : ℕ), 2 ≤ b_val ∧ b_val ≤ n ∧ 
    1 - 1 / (b_val * (b_val * (n + 1 - b_val) + 1)) = 
    1 - 1 / ((2 * n / 3 + 7 / 6) * ((2 * n / 3 + 7 / 6) * (n - (2 * n / 3 + 1 / 6)) + 1))) :=
sorry

end max_frac_sum_l209_20901


namespace side_length_square_eq_6_l209_20982

theorem side_length_square_eq_6
  (width length : ℝ)
  (h_width : width = 2)
  (h_length : length = 18) :
  (∃ s : ℝ, s^2 = width * length) ∧ (∀ s : ℝ, s^2 = width * length → s = 6) :=
by
  sorry

end side_length_square_eq_6_l209_20982


namespace total_equipment_cost_l209_20956

-- Definitions of costs in USD
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80
def number_of_players : ℝ := 16

-- Statement to prove
theorem total_equipment_cost :
  number_of_players * (jersey_cost + shorts_cost + socks_cost) = 752 :=
by
  sorry

end total_equipment_cost_l209_20956


namespace program_output_is_201_l209_20983

theorem program_output_is_201 :
  ∃ x S n, x = 3 + 2 * n ∧ S = n^2 + 4 * n ∧ S ≥ 10000 ∧ x = 201 :=
by
  sorry

end program_output_is_201_l209_20983


namespace dream_star_games_l209_20923

theorem dream_star_games (x y : ℕ) 
  (h1 : x + y + 2 = 9)
  (h2 : 3 * x + y = 17) : 
  x = 5 ∧ y = 2 := 
by 
  sorry

end dream_star_games_l209_20923


namespace adam_first_year_students_l209_20974

theorem adam_first_year_students (X : ℕ) 
  (remaining_years_students : ℕ := 9 * 50)
  (total_students : ℕ := 490) 
  (total_years_students : X + remaining_years_students = total_students) : X = 40 :=
by { sorry }

end adam_first_year_students_l209_20974


namespace doughnut_machine_completion_time_l209_20939

-- Define the start time and the time when half the job is completed
def start_time := 8 * 60 -- 8:00 AM in minutes
def half_job_time := 10 * 60 + 30 -- 10:30 AM in minutes

-- Given the machine completes half of the day's job by 10:30 AM
-- Prove that the doughnut machine will complete the entire job by 1:00 PM
theorem doughnut_machine_completion_time :
  half_job_time - start_time = 150 → 
  (start_time + 2 * 150) % (24 * 60) = 13 * 60 :=
by
  sorry

end doughnut_machine_completion_time_l209_20939


namespace repeat_45_fraction_repeat_245_fraction_l209_20993

-- Define the repeating decimal 0.454545... == n / d
def repeating_45_equiv : Prop := ∃ n d : ℕ, (d ≠ 0) ∧ (0.45454545 = (n : ℚ) / (d : ℚ))

-- First problem statement: 0.4545... == 5 / 11
theorem repeat_45_fraction : 0.45454545 = (5 : ℚ) / (11 : ℚ) :=
by
  sorry

-- Define the repeating decimal 0.2454545... == n / d
def repeating_245_equiv : Prop := ∃ n d : ℕ, (d ≠ 0) ∧ (0.2454545 = (n : ℚ) / (d : ℚ))

-- Second problem statement: 0.2454545... == 27 / 110
theorem repeat_245_fraction : 0.2454545 = (27 : ℚ) / (110 : ℚ) :=
by
  sorry

end repeat_45_fraction_repeat_245_fraction_l209_20993


namespace total_amount_paid_l209_20961

variable (n : ℕ) (each_paid : ℕ)

/-- This is a statement that verifies the total amount paid given the number of friends and the amount each friend pays. -/
theorem total_amount_paid (h1 : n = 7) (h2 : each_paid = 70) : n * each_paid = 490 := by
  -- This proof will validate that the total amount paid is 490
  sorry

end total_amount_paid_l209_20961


namespace inequality_solution_l209_20909

open Set

noncomputable def solution_set := { x : ℝ | 5 - x^2 > 4 * x }

theorem inequality_solution :
  solution_set = { x : ℝ | -5 < x ∧ x < 1 } :=
by
  sorry

end inequality_solution_l209_20909


namespace move_up_4_units_l209_20952

-- Define the given points M and N
def M : ℝ × ℝ := (-1, -1)
def N : ℝ × ℝ := (-1, 3)

-- State the theorem to be proved
theorem move_up_4_units (M N : ℝ × ℝ) :
  (M = (-1, -1)) → (N = (-1, 3)) → (N = (M.1, M.2 + 4)) :=
by
  intros hM hN
  rw [hM, hN]
  sorry

end move_up_4_units_l209_20952


namespace calendars_ordered_l209_20945

theorem calendars_ordered 
  (C D : ℝ) 
  (h1 : C + D = 500) 
  (h2 : 0.75 * C + 0.50 * D = 300) 
  : C = 200 :=
by
  sorry

end calendars_ordered_l209_20945


namespace trig_simplification_l209_20997

theorem trig_simplification :
  (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180)) = 4 :=
sorry

end trig_simplification_l209_20997


namespace calc_expr_l209_20966

theorem calc_expr : 
  (1 / (1 + 1 / (2 + 1 / 5))) = 11 / 16 :=
by
  sorry

end calc_expr_l209_20966


namespace subtraction_identity_l209_20943

theorem subtraction_identity : 3.57 - 1.14 - 0.23 = 2.20 := sorry

end subtraction_identity_l209_20943


namespace complex_power_sum_eq_five_l209_20979

noncomputable def w : ℂ := sorry

theorem complex_power_sum_eq_five (h : w^3 + w^2 + 1 = 0) : 
  w^100 + w^101 + w^102 + w^103 + w^104 = 5 :=
sorry

end complex_power_sum_eq_five_l209_20979


namespace equalChargesAtFour_agencyADecisionWhenTen_l209_20905

-- Define the conditions as constants
def fullPrice : ℕ := 240
def agencyADiscount : ℕ := 50
def agencyBDiscount : ℕ := 60

-- Define the total charge function for both agencies
def totalChargeAgencyA (students: ℕ) : ℕ :=
  fullPrice * students * agencyADiscount / 100 + fullPrice

def totalChargeAgencyB (students: ℕ) : ℕ :=
  fullPrice * (students + 1) * agencyBDiscount / 100

-- Define the equivalence when the number of students is 4
theorem equalChargesAtFour : totalChargeAgencyA 4 = totalChargeAgencyB 4 := by sorry

-- Define the decision when there are 10 students
theorem agencyADecisionWhenTen : totalChargeAgencyA 10 < totalChargeAgencyB 10 := by sorry

end equalChargesAtFour_agencyADecisionWhenTen_l209_20905


namespace symmetric_slope_angle_l209_20926

-- Define the problem conditions in Lean
def slope_angle (θ : Real) : Prop :=
  0 ≤ θ ∧ θ < Real.pi

-- Statement of the theorem in Lean
theorem symmetric_slope_angle (θ : Real) (h : slope_angle θ) :
  θ = 0 ∨ θ = Real.pi - θ :=
sorry

end symmetric_slope_angle_l209_20926


namespace line_AB_bisects_segment_DE_l209_20915

variables {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
  {trapezoid : A × B × C × D} (AC CD : Prop) (BD_sym : Prop) (intersect_E : Prop)
  (line_AB : Prop) (bisects_DE : Prop)

-- Given a trapezoid ABCD
def is_trapezoid (A B C D : Type) : Prop := sorry

-- Given the diagonal AC is equal to the side CD
def diagonal_eq_leg (AC CD : Prop) : Prop := sorry

-- Given line BD is symmetric with respect to AD intersects AC at point E
def symmetric_line_intersect (BD_sym AD AC E : Prop) : Prop := sorry

-- Prove that line AB bisects segment DE
theorem line_AB_bisects_segment_DE
  (h_trapezoid : is_trapezoid A B C D)
  (h_diagonal_eq_leg : diagonal_eq_leg AC CD)
  (h_symmetric_line_intersect : symmetric_line_intersect BD_sym (sorry : Prop) AC intersect_E)
  (h_line_AB : line_AB) :
  bisects_DE := sorry

end line_AB_bisects_segment_DE_l209_20915


namespace vertex_on_xaxis_l209_20913

-- Definition of the parabola equation with vertex on the x-axis
def parabola (x m : ℝ) := x^2 - 8 * x + m

-- The problem statement: show that m = 16 given that the vertex of the parabola is on the x-axis
theorem vertex_on_xaxis (m : ℝ) : ∃ x : ℝ, parabola x m = 0 → m = 16 :=
by
  sorry

end vertex_on_xaxis_l209_20913


namespace initial_ratio_of_liquids_l209_20992

theorem initial_ratio_of_liquids (p q : ℕ) (h1 : p + q = 40) (h2 : p / (q + 15) = 5 / 6) : p / q = 5 / 3 :=
by
  sorry

end initial_ratio_of_liquids_l209_20992


namespace number_of_laborers_l209_20907

-- Definitions based on conditions in the problem
def hpd := 140   -- Earnings per day for heavy equipment operators
def gpd := 90    -- Earnings per day for general laborers
def totalPeople := 35  -- Total number of people hired
def totalPayroll := 3950  -- Total payroll in dollars

-- Variables H and L for the number of operators and laborers
variables (H L : ℕ)

-- Conditions provided in mathematical problem
axiom equation1 : H + L = totalPeople
axiom equation2 : hpd * H + gpd * L = totalPayroll

-- Theorem statement: we want to prove that L = 19
theorem number_of_laborers : L = 19 :=
sorry

end number_of_laborers_l209_20907


namespace divisibility_theorem_l209_20925

theorem divisibility_theorem (a b n : ℕ) (h : a^n ∣ b) : a^(n + 1) ∣ (a + 1)^b - 1 :=
by 
sorry

end divisibility_theorem_l209_20925


namespace abs_eq_two_iff_l209_20999

theorem abs_eq_two_iff (a : ℝ) : |a| = 2 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end abs_eq_two_iff_l209_20999


namespace task_assignments_count_l209_20964

theorem task_assignments_count (S : Finset (Fin 5)) :
  ∃ (assignments : Fin 5 → Fin 3),  
    (∀ t, assignments t ≠ t) ∧ 
    (∀ v, ∃ t, assignments t = v) ∧ 
    (∀ t, (t = 4 → assignments t = 1)) ∧ 
    S.card = 60 :=
by sorry

end task_assignments_count_l209_20964


namespace kate_change_l209_20985

def first_candy_cost : ℝ := 0.54
def second_candy_cost : ℝ := 0.35
def third_candy_cost : ℝ := 0.68
def amount_given : ℝ := 5.00

theorem kate_change : amount_given - (first_candy_cost + second_candy_cost + third_candy_cost) = 3.43 := by
  sorry

end kate_change_l209_20985


namespace range_of_a_l209_20987

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(x^2 - 2*x + 3 ≤ a^2 - 2*a - 1))
  ↔ (-1 < a ∧ a < 3) :=
by sorry

end range_of_a_l209_20987


namespace fourth_equation_general_expression_l209_20972

theorem fourth_equation :
  (10 : ℕ)^2 - 4 * (4 : ℕ)^2 = 36 := 
sorry

theorem general_expression (n : ℕ) (hn : n > 0) :
  (2 * n + 2)^2 - 4 * n^2 = 8 * n + 4 :=
sorry

end fourth_equation_general_expression_l209_20972


namespace family_reunion_people_l209_20998

theorem family_reunion_people (pasta_per_person : ℚ) (total_pasta : ℚ) (recipe_people : ℚ) : 
  pasta_per_person = 2 / 7 ∧ total_pasta = 10 -> recipe_people = 35 :=
by
  sorry

end family_reunion_people_l209_20998


namespace find_x_when_y_is_6_l209_20973

-- Condition for inverse variation
def inverse_var (k y : ℝ) (x : ℝ) : Prop := x = k / y^2

-- Given values
def given_value_x : ℝ := 1
def given_value_y : ℝ := 2
def new_value_y : ℝ := 6

-- The theorem to prove
theorem find_x_when_y_is_6 :
  ∃ k, inverse_var k given_value_y given_value_x → inverse_var k new_value_y (1/9) :=
by
  sorry

end find_x_when_y_is_6_l209_20973


namespace sum_of_selected_sections_l209_20989

-- Given volumes of a bamboo, we denote them as a1, a2, ..., a9 forming an arithmetic sequence.
-- Where the sum of the volumes of the top four sections is 3 liters, and the
-- sum of the volumes of the bottom three sections is 4 liters.

-- Definitions based on the conditions
def arith_seq (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

variables {a : ℕ → ℝ} {d : ℝ}
variable (sum_top_four : a 1 + a 2 + a 3 + a 4 = 3)
variable (sum_bottom_three : a 7 + a 8 + a 9 = 4)
variable (seq_condition : arith_seq a d)

theorem sum_of_selected_sections 
  (h1 : a 1 + a 2 + a 3 + a 4 = 3)
  (h2 : a 7 + a 8 + a 9 = 4)
  (h_seq : arith_seq a d) : 
  a 2 + a 3 + a 8 = 17 / 6 := 
sorry -- proof goes here

end sum_of_selected_sections_l209_20989


namespace maximum_students_l209_20929

theorem maximum_students (x : ℕ) (hx : x / 2 + x / 4 + x / 7 + 6 > x) : x ≤ 28 :=
by sorry

end maximum_students_l209_20929


namespace enclosure_largest_side_l209_20903

theorem enclosure_largest_side (l w : ℕ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 3600) : l = 60 :=
by
  sorry

end enclosure_largest_side_l209_20903


namespace line_slope_product_l209_20995

theorem line_slope_product (x y : ℝ) (h1 : (x, 6) = (x, 6)) (h2 : (10, y) = (10, y)) (h3 : ∀ x, y = (1 / 2) * x) : x * y = 60 :=
sorry

end line_slope_product_l209_20995


namespace mark_pond_depth_l209_20938

def depth_of_Peter_pond := 5

def depth_of_Mark_pond := 3 * depth_of_Peter_pond + 4

theorem mark_pond_depth : depth_of_Mark_pond = 19 := by
  sorry

end mark_pond_depth_l209_20938


namespace find_k_value_l209_20933

theorem find_k_value (a : ℕ → ℕ) (k : ℕ) (S : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 3 = 5) 
  (h₃ : S (k + 2) - S k = 36) : 
  k = 8 := 
by 
  sorry

end find_k_value_l209_20933


namespace cost_of_ice_cream_l209_20962

/-- Alok ordered 16 chapatis, 5 plates of rice, 7 plates of mixed vegetable, and 6 ice-cream cups. 
    The cost of each chapati is Rs. 6, that of each plate of rice is Rs. 45, and that of mixed 
    vegetable is Rs. 70. Alok paid the cashier Rs. 931. Prove the cost of each ice-cream cup is Rs. 20. -/
theorem cost_of_ice_cream (n_chapatis n_rice n_vegetable n_ice_cream : ℕ) 
    (cost_chapati cost_rice cost_vegetable total_paid : ℕ)
    (h_chapatis : n_chapatis = 16) 
    (h_rice : n_rice = 5)
    (h_vegetable : n_vegetable = 7)
    (h_ice_cream : n_ice_cream = 6)
    (h_cost_chapati : cost_chapati = 6)
    (h_cost_rice : cost_rice = 45)
    (h_cost_vegetable : cost_vegetable = 70)
    (h_total_paid : total_paid = 931) :
    (total_paid - (n_chapatis * cost_chapati + n_rice * cost_rice + n_vegetable * cost_vegetable)) / n_ice_cream = 20 := 
by
  sorry

end cost_of_ice_cream_l209_20962


namespace percentage_error_l209_20957

theorem percentage_error (x : ℝ) : ((x * 3 - x / 5) / (x * 3) * 100) = 93.33 := 
  sorry

end percentage_error_l209_20957


namespace calculate_base_length_l209_20900

variable (A b h : ℝ)

def is_parallelogram_base_length (A : ℝ) (b : ℝ) (h : ℝ) : Prop :=
  (A = b * h) ∧ (h = 2 * b)

theorem calculate_base_length (H : is_parallelogram_base_length A b h) : b = 15 := by
  -- H gives us the hypothesis that (A = b * h) and (h = 2 * b)
  have H1 : A = b * h := H.1
  have H2 : h = 2 * b := H.2
  -- Use substitution and algebra to solve for b
  sorry

end calculate_base_length_l209_20900


namespace unique_integer_solution_l209_20934

theorem unique_integer_solution (x y : ℤ) : 
  x^4 + y^4 = 3 * x^3 * y → x = 0 ∧ y = 0 :=
by
  -- This is where the proof would go
  sorry

end unique_integer_solution_l209_20934


namespace two_digit_perfect_squares_divisible_by_3_l209_20980

theorem two_digit_perfect_squares_divisible_by_3 :
  ∃! n1 n2 : ℕ, (10 ≤ n1^2 ∧ n1^2 < 100 ∧ n1^2 % 3 = 0) ∧
               (10 ≤ n2^2 ∧ n2^2 < 100 ∧ n2^2 % 3 = 0) ∧
                (n1 ≠ n2) :=
by sorry

end two_digit_perfect_squares_divisible_by_3_l209_20980


namespace total_bending_angle_l209_20996

theorem total_bending_angle (n : ℕ) (h : n > 4) (θ : ℝ) (hθ : θ = 360 / (2 * n)) : 
  ∃ α : ℝ, α = 180 :=
by
  sorry

end total_bending_angle_l209_20996


namespace find_perimeter_square3_l209_20984

-- Define the conditions: perimeter of first and second square
def perimeter_square1 := 60
def perimeter_square2 := 48

-- Calculate side lengths based on the perimeter
def side_length_square1 := perimeter_square1 / 4
def side_length_square2 := perimeter_square2 / 4

-- Calculate areas of the two squares
def area_square1 := side_length_square1 * side_length_square1
def area_square2 := side_length_square2 * side_length_square2

-- Calculate the area of the third square
def area_square3 := area_square1 - area_square2

-- Calculate the side length of the third square
def side_length_square3 := Nat.sqrt area_square3

-- Define the perimeter of the third square
def perimeter_square3 := 4 * side_length_square3

/-- Theorem: The perimeter of the third square is 36 cm -/
theorem find_perimeter_square3 : perimeter_square3 = 36 := by
  sorry

end find_perimeter_square3_l209_20984


namespace find_number_l209_20981

theorem find_number (x : ℝ) (h : 3034 - (1002 / x) = 2984) : x = 20.04 :=
by
  sorry

end find_number_l209_20981


namespace necessary_not_sufficient_l209_20968

theorem necessary_not_sufficient (a b c : ℝ) : (a < b) → (ac^2 < b * c^2) ∧ ∀a b c : ℝ, (ac^2 < b * c^2) → (a < b) :=
sorry

end necessary_not_sufficient_l209_20968


namespace solution_set_of_inequality_l209_20958

theorem solution_set_of_inequality :
  { x : ℝ | x ≠ 5 ∧ (x * (x + 1)) / ((x - 5) ^ 3) ≥ 25 } = 
  { x : ℝ | x ≤ 5 / 3 } ∪ { x : ℝ | x > 5 } := by
  sorry

end solution_set_of_inequality_l209_20958


namespace chord_property_l209_20963

noncomputable def chord_length (R r k : ℝ) : Prop :=
  k = 2 * Real.sqrt (R^2 - r^2)

theorem chord_property (P O : Point) (R k : ℝ) (hR : 0 < R) (hk : 0 < k) :
  ∃ r, r = Real.sqrt (R^2 - k^2 / 4) ∧ chord_length R r k :=
sorry

end chord_property_l209_20963


namespace inequality_solution_l209_20932

theorem inequality_solution (x : ℤ) (h : x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1) : x - 1 ≥ 0 ↔ x = 1 :=
by
  sorry

end inequality_solution_l209_20932


namespace sandwich_cost_90_cents_l209_20978

theorem sandwich_cost_90_cents :
  let cost_bread := 0.15
  let cost_ham := 0.25
  let cost_cheese := 0.35
  (2 * cost_bread + cost_ham + cost_cheese) * 100 = 90 := 
by
  sorry

end sandwich_cost_90_cents_l209_20978


namespace red_grapes_more_than_three_times_green_l209_20906

-- Definitions from conditions
variables (G R B : ℕ)
def condition1 := R = 3 * G + (R - 3 * G)
def condition2 := B = G - 5
def condition3 := R + G + B = 102
def condition4 := R = 67

-- The proof problem
theorem red_grapes_more_than_three_times_green : (R = 67) ∧ (R + G + (G - 5) = 102) ∧ (R = 3 * G + (R - 3 * G)) → R - 3 * G = 7 :=
by sorry

end red_grapes_more_than_three_times_green_l209_20906


namespace suitable_land_acres_l209_20922

theorem suitable_land_acres (new_multiplier : ℝ) (previous_acres : ℝ) (pond_acres : ℝ) :
  new_multiplier = 10 ∧ previous_acres = 2 ∧ pond_acres = 1 → 
  (new_multiplier * previous_acres - pond_acres) = 19 :=
by
  intro h
  sorry

end suitable_land_acres_l209_20922


namespace count_multiples_4_or_9_but_not_both_l209_20912

theorem count_multiples_4_or_9_but_not_both (n : ℕ) (h : n = 200) :
  let count_multiples (k : ℕ) := (n / k)
  count_multiples 4 + count_multiples 9 - 2 * count_multiples 36 = 62 :=
by
  sorry

end count_multiples_4_or_9_but_not_both_l209_20912


namespace find_N_l209_20947

-- Define the problem parameters
def certain_value : ℝ := 0
def x : ℝ := 10

-- Define the main statement to be proved
theorem find_N (N : ℝ) : 3 * x = (N - x) + certain_value → N = 40 :=
  by sorry

end find_N_l209_20947


namespace binom_divisible_by_prime_l209_20948

theorem binom_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (h1 : 1 ≤ k) (h2 : k ≤ p - 1) : p ∣ Nat.choose p k :=
sorry

end binom_divisible_by_prime_l209_20948


namespace police_officer_placement_l209_20949

-- The given problem's conditions
def intersections : Finset String := {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}

def streets : List (Finset String) := [
    {"A", "B", "C", "D"},        -- Horizontal streets
    {"E", "F", "G"},
    {"H", "I", "J", "K"},
    {"A", "E", "H"},             -- Vertical streets
    {"B", "F", "I"},
    {"D", "G", "J"},
    {"H", "F", "C"},             -- Diagonal streets
    {"C", "G", "K"}
]

def chosen_intersections : Finset String := {"B", "G", "H"}

-- Proof problem
theorem police_officer_placement :
  ∀ street ∈ streets, ∃ p ∈ chosen_intersections, p ∈ street := by
  sorry

end police_officer_placement_l209_20949


namespace pairwise_coprime_circle_l209_20976

theorem pairwise_coprime_circle :
  ∃ (circle : Fin 100 → ℕ),
    (∀ i, Nat.gcd (circle i) (Nat.gcd (circle ((i + 1) % 100)) (circle ((i - 1) % 100))) = 1) → 
    ∀ i j, i ≠ j → Nat.gcd (circle i) (circle j) = 1 :=
by
  sorry

end pairwise_coprime_circle_l209_20976


namespace count_prime_numbers_in_sequence_l209_20937

theorem count_prime_numbers_in_sequence : 
  ∀ (k : Nat), (∃ n : Nat, 47 * (10^n * k + (10^(n-1) - 1) / 9) = 47) → k = 0 :=
  sorry

end count_prime_numbers_in_sequence_l209_20937


namespace parallel_lines_a_perpendicular_lines_a_l209_20931

-- Definitions of the lines
def l1 (a x y : ℝ) := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) := x + (a - 1) * y + a^2 - 1 = 0

-- Statement for parallel lines problem
theorem parallel_lines_a (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y) → (a = -1) :=
by
  sorry

-- Statement for perpendicular lines problem
theorem perpendicular_lines_a (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y → (-a / 2) * (1 / (a - 1)) = -1) → (a = 2 / 3) :=
by
  sorry

end parallel_lines_a_perpendicular_lines_a_l209_20931


namespace solve_equation_l209_20910

theorem solve_equation (x : ℝ) (h1 : x ≠ 2 / 3) :
  (7 * x + 3) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔
  x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end solve_equation_l209_20910


namespace books_bound_l209_20969

theorem books_bound (x : ℕ) (w c : ℕ) (h₀ : w = 92) (h₁ : c = 135) 
(h₂ : 92 - x = 2 * (135 - x)) :
x = 178 :=
by
  sorry

end books_bound_l209_20969


namespace sum_first_seven_arithmetic_l209_20967

theorem sum_first_seven_arithmetic (a : ℕ) (d : ℕ) (h : a + 3 * d = 3) :
    let a1 := a
    let a2 := a + d
    let a3 := a + 2 * d
    let a4 := a + 3 * d
    let a5 := a + 4 * d
    let a6 := a + 5 * d
    let a7 := a + 6 * d
    a1 + a2 + a3 + a4 + a5 + a6 + a7 = 21 :=
by
  sorry

end sum_first_seven_arithmetic_l209_20967


namespace roots_geometric_progression_two_complex_conjugates_l209_20951

theorem roots_geometric_progression_two_complex_conjugates (a : ℝ) :
  (∃ b k : ℝ, b ≠ 0 ∧ k ≠ 0 ∧ (k + 1/ k = 2) ∧ 
    (b * (1 + k + 1/k) = 9) ∧ (b^2 * (k + 1 + 1/k) = 27) ∧ (b^3 = -a)) →
  a = -27 :=
by sorry

end roots_geometric_progression_two_complex_conjugates_l209_20951


namespace smallest_C_inequality_l209_20941

theorem smallest_C_inequality (x y z : ℝ) (h : x + y + z = -1) : 
  |x^3 + y^3 + z^3 + 1| ≤ (9/10) * |x^5 + y^5 + z^5 + 1| :=
  sorry

end smallest_C_inequality_l209_20941


namespace gold_coins_count_l209_20994

theorem gold_coins_count (G : ℕ) 
  (h1 : 50 * G + 125 + 30 = 305) :
  G = 3 := 
by
  sorry

end gold_coins_count_l209_20994


namespace range_of_f_l209_20940

noncomputable def f (x : ℕ) : ℤ := x^2 - 3 * x

def domain : Finset ℕ := {1, 2, 3}

def range : Finset ℤ := {-2, 0}

theorem range_of_f :
  Finset.image f domain = range :=
by
  sorry

end range_of_f_l209_20940


namespace calculate_triple_transform_l209_20960

def transformation (N : ℝ) : ℝ :=
  0.4 * N + 2

theorem calculate_triple_transform :
  transformation (transformation (transformation 20)) = 4.4 :=
by
  sorry

end calculate_triple_transform_l209_20960


namespace Suresh_meeting_time_l209_20920

theorem Suresh_meeting_time :
  let C := 726
  let v1 := 75
  let v2 := 62.5
  C / (v1 + v2) = 5.28 := by
  sorry

end Suresh_meeting_time_l209_20920


namespace min_lcm_value_l209_20904

-- Definitions
def gcd_77 (a b c d : ℕ) : Prop :=
  Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 77

def lcm_n (a b c d n : ℕ) : Prop :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = n

-- Problem statement
theorem min_lcm_value :
  (∃ a b c d : ℕ, gcd_77 a b c d ∧ lcm_n a b c d 27720) ∧
  (∀ n : ℕ, (∃ a b c d : ℕ, gcd_77 a b c d ∧ lcm_n a b c d n) → 27720 ≤ n) :=
sorry

end min_lcm_value_l209_20904


namespace cricketer_average_after_19_innings_l209_20914

theorem cricketer_average_after_19_innings
  (runs_19th_inning : ℕ)
  (increase_in_average : ℤ)
  (initial_average : ℤ)
  (new_average : ℤ)
  (h1 : runs_19th_inning = 95)
  (h2 : increase_in_average = 4)
  (eq1 : 18 * initial_average + 95 = 19 * (initial_average + increase_in_average))
  (eq2 : new_average = initial_average + increase_in_average) :
  new_average = 23 :=
by sorry

end cricketer_average_after_19_innings_l209_20914


namespace dots_not_visible_l209_20918

def total_dots_on_die : Nat := 21
def number_of_dice : Nat := 4
def total_dots : Nat := number_of_dice * total_dots_on_die
def visible_faces : List Nat := [1, 2, 2, 3, 3, 5, 6]
def sum_visible_faces : Nat := visible_faces.sum

theorem dots_not_visible : total_dots - sum_visible_faces = 62 := by
  sorry

end dots_not_visible_l209_20918


namespace remainder_5_to_5_to_5_to_5_mod_1000_l209_20936

theorem remainder_5_to_5_to_5_to_5_mod_1000 : (5^(5^(5^5))) % 1000 = 125 :=
by {
  sorry
}

end remainder_5_to_5_to_5_to_5_mod_1000_l209_20936


namespace divisible_by_8_l209_20955

theorem divisible_by_8 (k : ℤ) : 
  let m := 2 * k + 1 
  let n := 2 * k + 3 
  8 ∣ (7 * m^2 - 5 * n^2 - 2) :=
by 
  let m := 2 * k + 1 
  let n := 2 * k + 3 
  sorry

end divisible_by_8_l209_20955


namespace sequence_general_term_l209_20971

theorem sequence_general_term (n : ℕ) : 
  (∀ (a : ℕ → ℚ), (a 1 = 1) ∧ (a 2 = 2 / 3) ∧ (a 3 = 3 / 7) ∧ (a 4 = 4 / 15) ∧ (a 5 = 5 / 31) → a n = n / (2^n - 1)) :=
by
  sorry

end sequence_general_term_l209_20971


namespace simplify_fraction_expression_l209_20991

theorem simplify_fraction_expression (d : ℤ) :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 :=
by
  -- skip the proof by adding sorry
  sorry

end simplify_fraction_expression_l209_20991


namespace perimeter_regular_polygon_l209_20944

-- Condition definitions
def is_regular_polygon (n : ℕ) (s : ℝ) : Prop := 
  n * s > 0

def exterior_angle (E : ℝ) (n : ℕ) : Prop := 
  E = 360 / n

def side_length (s : ℝ) : Prop :=
  s = 6

-- Theorem statement to prove the perimeter is 24 units
theorem perimeter_regular_polygon 
  (n : ℕ) (s E : ℝ)
  (h1 : is_regular_polygon n s)
  (h2 : exterior_angle E n)
  (h3 : side_length s)
  (h4 : E = 90) :
  4 * s = 24 :=
by
  sorry

end perimeter_regular_polygon_l209_20944
