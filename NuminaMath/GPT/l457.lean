import Mathlib

namespace slope_of_parallel_line_l457_45775

theorem slope_of_parallel_line (a b c : ℝ) (h : a = 3 ∧ b = -6 ∧ c = 12) :
  ∃ m : ℝ, (∀ (x y : ℝ), 3 * x - 6 * y = 12 → y = m * x - 2) ∧ m = 1/2 := 
sorry

end slope_of_parallel_line_l457_45775


namespace largest_square_area_l457_45791

theorem largest_square_area (a b c : ℝ)
  (h1 : a^2 + b^2 = c^2)
  (h2 : a^2 + b^2 + c^2 = 450) :
  c^2 = 225 :=
by
  sorry

end largest_square_area_l457_45791


namespace min_value_expr_l457_45731

-- Definition of the expression given a real constant k
def expr (k : ℝ) (x y : ℝ) : ℝ := 9 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9

-- The proof problem statement
theorem min_value_expr (k : ℝ) (h : k = 2 / 9) : ∃ x y : ℝ, expr k x y = 1 ∧ ∀ x y : ℝ, expr k x y ≥ 1 :=
by
  sorry

end min_value_expr_l457_45731


namespace tennis_tournament_cycle_l457_45763

noncomputable def exists_cycle_of_three_players (P : Type) [Fintype P] (G : P → P → Bool) : Prop :=
  (∃ (a b c : P), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ G a b ∧ G b c ∧ G c a)

theorem tennis_tournament_cycle (P : Type) [Fintype P] (n : ℕ) (hp : 3 ≤ n) 
  (G : P → P → Bool) (H : ∀ a b : P, a ≠ b → (G a b ∨ G b a))
  (Hw : ∀ a : P, ∃ b : P, a ≠ b ∧ G a b) : exists_cycle_of_three_players P G :=
by 
  sorry

end tennis_tournament_cycle_l457_45763


namespace highest_vs_lowest_temp_difference_l457_45779

theorem highest_vs_lowest_temp_difference 
  (highest_temp lowest_temp : ℤ) 
  (h_highest : highest_temp = 26) 
  (h_lowest : lowest_temp = 14) : 
  highest_temp - lowest_temp = 12 := 
by 
  sorry

end highest_vs_lowest_temp_difference_l457_45779


namespace Grace_pool_water_capacity_l457_45766

theorem Grace_pool_water_capacity :
  let rate1 := 50 -- gallons per hour of the first hose
  let rate2 := 70 -- gallons per hour of the second hose
  let hours1 := 3 -- hours the first hose was used alone
  let hours2 := 2 -- hours both hoses were used together
  let water1 := rate1 * hours1 -- water from the first hose in the first period
  let water2 := rate2 * hours2 -- water from the second hose in the second period
  let water3 := rate1 * hours2 -- water from the first hose in the second period
  let total_water := water1 + water2 + water3 -- total water in the pool
  total_water = 390 :=
by
  sorry

end Grace_pool_water_capacity_l457_45766


namespace christian_age_in_years_l457_45705

theorem christian_age_in_years (B C x : ℕ) (h1 : C = 2 * B) (h2 : B + x = 40) (h3 : C + x = 72) :
    x = 8 := 
sorry

end christian_age_in_years_l457_45705


namespace positive_divisors_3k1_ge_3k_minus_1_l457_45773

theorem positive_divisors_3k1_ge_3k_minus_1 (n : ℕ) (h : 0 < n) :
  (∃ k : ℕ, (3 * k + 1) ∣ n) → (∃ k : ℕ, ¬ (3 * k - 1) ∣ n) :=
  sorry

end positive_divisors_3k1_ge_3k_minus_1_l457_45773


namespace largest_percentage_increase_l457_45792

def student_count (year: ℕ) : ℝ :=
  match year with
  | 2010 => 80
  | 2011 => 88
  | 2012 => 95
  | 2013 => 100
  | 2014 => 105
  | 2015 => 112
  | _    => 0  -- Because we only care about 2010-2015

noncomputable def percentage_increase (year1 year2 : ℕ) : ℝ :=
  ((student_count year2 - student_count year1) / student_count year1) * 100

theorem largest_percentage_increase :
  (∀ x y, percentage_increase 2010 2011 ≥ percentage_increase x y) :=
by sorry

end largest_percentage_increase_l457_45792


namespace solution_set_of_inequality_l457_45747

theorem solution_set_of_inequality :
  {x : ℝ | |x + 1| - |x - 5| < 4} = {x : ℝ | x < 4} :=
sorry

end solution_set_of_inequality_l457_45747


namespace fraction_operations_l457_45795

theorem fraction_operations :
  let a := 1 / 3
  let b := 1 / 4
  let c := 1 / 2
  (a + b = 7 / 12) ∧ ((7 / 12) / c = 7 / 6) := by
{
  sorry
}

end fraction_operations_l457_45795


namespace road_construction_equation_l457_45769

theorem road_construction_equation (x : ℝ) (hx : x > 0) :
  (9 / x) - (12 / (x + 1)) = 1 / 2 :=
sorry

end road_construction_equation_l457_45769


namespace carol_age_l457_45704

theorem carol_age (B C : ℕ) (h1 : B + C = 66) (h2 : C = 3 * B + 2) : C = 50 :=
sorry

end carol_age_l457_45704


namespace N_subset_M_l457_45744

open Set

def M : Set (ℝ × ℝ) := { p | ∃ x, p = (x, 2*x + 1) }
def N : Set (ℝ × ℝ) := { p | ∃ x, p = (x, -x^2) }

theorem N_subset_M : N ⊆ M :=
by
  sorry

end N_subset_M_l457_45744


namespace number_of_cakes_sold_l457_45774

-- Definitions based on the conditions provided
def cakes_made : ℕ := 173
def cakes_bought : ℕ := 103
def cakes_left : ℕ := 190

-- Calculate the initial total number of cakes
def initial_cakes : ℕ := cakes_made + cakes_bought

-- Calculate the number of cakes sold
def cakes_sold : ℕ := initial_cakes - cakes_left

-- The proof statement
theorem number_of_cakes_sold : cakes_sold = 86 :=
by
  unfold cakes_sold initial_cakes cakes_left cakes_bought cakes_made
  rfl

end number_of_cakes_sold_l457_45774


namespace find_function_l457_45703

def satisfies_condition (f : ℕ+ → ℕ+) :=
  ∀ a b : ℕ+, f a + b ∣ a^2 + f a * f b

theorem find_function :
  ∀ f : ℕ+ → ℕ+, satisfies_condition f → (∀ a : ℕ+, f a = a) :=
by
  intros f h
  sorry

end find_function_l457_45703


namespace k_value_l457_45752

theorem k_value (k : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |k * x - 4| ≤ 2) → k = 2 := 
by
  intros h
  sorry

end k_value_l457_45752


namespace find_x_l457_45706

theorem find_x (x : ℝ) (h : 65 + 5 * 12 / (x / 3) = 66) : x = 180 :=
by
  sorry

end find_x_l457_45706


namespace length_of_paving_stone_l457_45728

theorem length_of_paving_stone (courtyard_length courtyard_width : ℝ)
  (num_paving_stones : ℕ) (paving_stone_width : ℝ) (total_area : ℝ)
  (paving_stone_length : ℝ) : 
  courtyard_length = 70 ∧ courtyard_width = 16.5 ∧ num_paving_stones = 231 ∧ paving_stone_width = 2 ∧ total_area = courtyard_length * courtyard_width ∧ total_area = num_paving_stones * paving_stone_length * paving_stone_width → 
  paving_stone_length = 2.5 :=
by
  sorry

end length_of_paving_stone_l457_45728


namespace inequality_proof_l457_45743

theorem inequality_proof (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1)) ≤ 1 / 2 :=
sorry

end inequality_proof_l457_45743


namespace determine_real_pairs_l457_45715

theorem determine_real_pairs (a b : ℝ) :
  (∀ n : ℕ+, a * ⌊ b * n ⌋ = b * ⌊ a * n ⌋) →
  (∃ c : ℝ, (a = 0 ∧ b = c) ∨ (a = c ∧ b = 0) ∨ (a = c ∧ b = c) ∨ (∃ k l : ℤ, a = k ∧ b = l)) :=
by
  sorry

end determine_real_pairs_l457_45715


namespace smallest_sector_angle_l457_45761

def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem smallest_sector_angle 
  (a : ℕ) (d : ℕ) (n : ℕ := 15) (sum_angles : ℕ := 360) 
  (angles_arith_seq : arithmetic_sequence_sum a d n = sum_angles) 
  (h_poses : ∀ m : ℕ, arithmetic_sequence_sum a d m = sum_angles -> m = n) 
  : a = 3 := 
by 
  sorry

end smallest_sector_angle_l457_45761


namespace intersection_M_N_l457_45713

open Set

-- Definitions from conditions
def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {x | x < 1}

-- Proof statement
theorem intersection_M_N : M ∩ N = {-1} := 
by sorry

end intersection_M_N_l457_45713


namespace average_growth_rate_of_second_brand_l457_45719

theorem average_growth_rate_of_second_brand 
  (init1 : ℝ) (rate1 : ℝ) (init2 : ℝ) (t : ℝ) (r : ℝ)
  (h1 : init1 = 4.9) (h2 : rate1 = 0.275) (h3 : init2 = 2.5) (h4 : t = 5.647)
  (h_eq : init1 + rate1 * t = init2 + r * t) : 
  r = 0.7 :=
by 
  -- proof steps would go here
  sorry

end average_growth_rate_of_second_brand_l457_45719


namespace f_const_one_l457_45751

-- Mathematical Translation of the Definitions
variable (f g h : ℕ → ℕ)

-- Given conditions
axiom h_injective : Function.Injective h
axiom g_surjective : Function.Surjective g
axiom f_eq : ∀ n, f n = g n - h n + 1

-- Theorem to Prove
theorem f_const_one : ∀ n, f n = 1 :=
by
  sorry

end f_const_one_l457_45751


namespace not_perfect_square_for_n_greater_than_11_l457_45702

theorem not_perfect_square_for_n_greater_than_11 (n : ℤ) (h1 : n > 11) :
  ∀ m : ℤ, n^2 - 19 * n + 89 ≠ m^2 :=
sorry

end not_perfect_square_for_n_greater_than_11_l457_45702


namespace pos_real_ineq_l457_45733

theorem pos_real_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c)/3) :=
by 
  sorry

end pos_real_ineq_l457_45733


namespace prize_distribution_l457_45770

theorem prize_distribution : 
  ∃ (n1 n2 n3 : ℕ), -- The number of 1st, 2nd, and 3rd prize winners
  n1 + n2 + n3 = 7 ∧ -- Total number of winners is 7
  n1 * 800 + n2 * 700 + n3 * 300 = 4200 ∧ -- Total prize money distributed is $4200
  n1 = 1 ∧ -- Number of 1st prize winners
  n2 = 4 ∧ -- Number of 2nd prize winners
  n3 = 2 -- Number of 3rd prize winners
:= sorry

end prize_distribution_l457_45770


namespace solve_eq1_solve_eq2_l457_45718

-- Proof problem 1: Prove that under the condition 6x - 4 = 3x + 2, x = 2
theorem solve_eq1 : ∀ x : ℝ, 6 * x - 4 = 3 * x + 2 → x = 2 :=
by
  intro x
  intro h
  sorry

-- Proof problem 2: Prove that under the condition (x / 4) - (3 / 5) = (x + 1) / 2, x = -22/5
theorem solve_eq2 : ∀ x : ℝ, (x / 4) - (3 / 5) = (x + 1) / 2 → x = -(22 / 5) :=
by
  intro x
  intro h
  sorry

end solve_eq1_solve_eq2_l457_45718


namespace tan_plus_pi_over_4_l457_45745

variable (θ : ℝ)

-- Define the conditions
def condition_θ_interval : Prop := θ ∈ Set.Ioo (Real.pi / 2) Real.pi
def condition_sin_θ : Prop := Real.sin θ = 3 / 5

-- Define the theorem to be proved
theorem tan_plus_pi_over_4 (h1 : condition_θ_interval θ) (h2 : condition_sin_θ θ) :
  Real.tan (θ + Real.pi / 4) = 7 :=
sorry

end tan_plus_pi_over_4_l457_45745


namespace david_reading_time_l457_45742

theorem david_reading_time
  (total_time : ℕ)
  (math_time : ℕ)
  (spelling_time : ℕ)
  (reading_time : ℕ)
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18)
  (h4 : reading_time = total_time - (math_time + spelling_time)) :
  reading_time = 27 := 
by {
  sorry
}

end david_reading_time_l457_45742


namespace problem_a_problem_b_l457_45736

-- Problem (a)
theorem problem_a (n : ℕ) (hn : n > 0) :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

-- Problem (b)
theorem problem_b (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (coprime_ab : Nat.gcd a b = 1)
  (coprime_ac_or_bc : Nat.gcd c a = 1 ∨ Nat.gcd c b = 1) :
  ∃ᶠ x : ℕ in Filter.atTop, ∃ (y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^a + y^b = z^c :=
sorry

end problem_a_problem_b_l457_45736


namespace part_a_part_b_l457_45735

theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) : x + y + z ≤ 4 :=
sorry

theorem part_b : ∃ (S : Set (ℚ × ℚ × ℚ)), S.Countable ∧
  (∀ (x y z : ℚ), (x, y, z) ∈ S → 0 < x ∧ 0 < y ∧ 0 < z ∧ 16 * x * y * z = (x + y)^2 * (x + z)^2 ∧ x + y + z = 4) ∧ 
  Infinite S :=
sorry

end part_a_part_b_l457_45735


namespace initial_observations_count_l457_45793

theorem initial_observations_count (S x n : ℕ) (h1 : S = 12 * n) (h2 : S + x = 11 * (n + 1)) (h3 : x = 5) : n = 6 :=
sorry

end initial_observations_count_l457_45793


namespace machine_parts_probabilities_l457_45714

-- Define the yield rates for the two machines
def yield_rate_A : ℝ := 0.8
def yield_rate_B : ℝ := 0.9

-- Define the probabilities of defectiveness for each machine
def defective_probability_A := 1 - yield_rate_A
def defective_probability_B := 1 - yield_rate_B

theorem machine_parts_probabilities :
  (defective_probability_A * defective_probability_B = 0.02) ∧
  (((yield_rate_A * defective_probability_B) + (defective_probability_A * yield_rate_B)) = 0.26) ∧
  (defective_probability_A * defective_probability_B + (1 - (defective_probability_A * defective_probability_B)) = 1) :=
by
  sorry

end machine_parts_probabilities_l457_45714


namespace angle_AM_BN_60_degrees_area_triangle_ABP_eq_area_quadrilateral_MDNP_l457_45700

-- Definitions according to the given conditions
variables (A B C D E F M N P : Point)
  (hexagon_regular : is_regular_hexagon A B C D E F)
  (is_midpoint_M : is_midpoint M C D)
  (is_midpoint_N : is_midpoint N D E)
  (intersection_P : intersection_point P (line_through A M) (line_through B N))

-- Angle between AM and BN is 60 degrees
theorem angle_AM_BN_60_degrees 
  (h1 : hexagon_regular)
  (h2 : is_midpoint_M)
  (h3 : is_midpoint_N)
  (h4 : intersection_P) :
  angle (line_through A M) (line_through B N) = 60 := 
sorry

-- Area of triangle ABP is equal to the area of quadrilateral MDNP
theorem area_triangle_ABP_eq_area_quadrilateral_MDNP 
  (h1 : hexagon_regular)
  (h2 : is_midpoint_M)
  (h3 : is_midpoint_N)
  (h4 : intersection_P) :
  area (triangle A B P) = area (quadrilateral M D N P) := 
sorry

end angle_AM_BN_60_degrees_area_triangle_ABP_eq_area_quadrilateral_MDNP_l457_45700


namespace treasure_chest_l457_45740

theorem treasure_chest (n : ℕ) 
  (h1 : n % 8 = 2)
  (h2 : n % 7 = 6)
  (h3 : ∀ m : ℕ, (m % 8 = 2 → m % 7 = 6 → m ≥ n)) :
  n % 9 = 7 :=
sorry

end treasure_chest_l457_45740


namespace price_of_first_variety_of_oil_l457_45797

theorem price_of_first_variety_of_oil 
  (P : ℕ) 
  (x : ℕ) 
  (cost_second_variety : ℕ) 
  (volume_second_variety : ℕ)
  (cost_mixture_per_liter : ℕ) 
  : x = 160 ∧ cost_second_variety = 60 ∧ volume_second_variety = 240 ∧ cost_mixture_per_liter = 52 → P = 40 :=
by
  sorry

end price_of_first_variety_of_oil_l457_45797


namespace not_universally_better_l457_45760

-- Definitions based on the implicitly given conditions
def can_show_quantity (chart : Type) : Prop := sorry
def can_reflect_changes (chart : Type) : Prop := sorry

-- Definitions of bar charts and line charts
inductive BarChart
| mk : BarChart

inductive LineChart
| mk : LineChart

-- Assumptions based on characteristics of the charts
axiom bar_chart_shows_quantity : can_show_quantity BarChart 
axiom line_chart_shows_quantity : can_show_quantity LineChart 
axiom line_chart_reflects_changes : can_reflect_changes LineChart 

-- Proof problem statement
theorem not_universally_better : ¬(∀ (c1 c2 : Type), can_show_quantity c1 → can_reflect_changes c1 → ¬can_show_quantity c2 → ¬can_reflect_changes c2) :=
  sorry

end not_universally_better_l457_45760


namespace Jessie_lost_7_kilograms_l457_45721

def Jessie_previous_weight : ℕ := 74
def Jessie_current_weight : ℕ := 67
def Jessie_weight_lost : ℕ := Jessie_previous_weight - Jessie_current_weight

theorem Jessie_lost_7_kilograms : Jessie_weight_lost = 7 :=
by
  sorry

end Jessie_lost_7_kilograms_l457_45721


namespace polygon_eq_quadrilateral_l457_45757

theorem polygon_eq_quadrilateral (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 := 
sorry

end polygon_eq_quadrilateral_l457_45757


namespace work_done_on_gas_in_process_1_2_l457_45724

variables (V₁ V₂ V₃ V₄ A₁₂ A₃₄ T n R : ℝ)

-- Both processes 1-2 and 3-4 are isothermal.
def is_isothermal_process := true -- Placeholder

-- Volumes relationship: for any given pressure, the volume in process 1-2 is exactly twice the volume in process 3-4.
def volumes_relation (V₁ V₂ V₃ V₄ : ℝ) : Prop :=
  V₁ = 2 * V₃ ∧ V₂ = 2 * V₄

-- Work done on a gas during an isothermal process can be represented as: A = 2 * A₃₄
def work_relation (A₁₂ A₃₄ : ℝ) : Prop :=
  A₁₂ = 2 * A₃₄

theorem work_done_on_gas_in_process_1_2
  (h_iso : is_isothermal_process)
  (h_vol : volumes_relation V₁ V₂ V₃ V₄)
  (h_work : work_relation A₁₂ A₃₄) :
  A₁₂ = 2 * A₃₄ :=
by 
  sorry

end work_done_on_gas_in_process_1_2_l457_45724


namespace no_nontrivial_sum_periodic_functions_l457_45782

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

def is_nontrivial_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  periodic f p ∧ ∃ x y, x ≠ y ∧ f x ≠ f y

theorem no_nontrivial_sum_periodic_functions (g h : ℝ → ℝ) :
  is_nontrivial_periodic_function g 1 →
  is_nontrivial_periodic_function h π →
  ¬ ∃ T > 0, ∀ x, (g + h) (x + T) = (g + h) x :=
sorry

end no_nontrivial_sum_periodic_functions_l457_45782


namespace triangle_area_parallel_line_l457_45783

/-- Given line passing through (8, 2) and parallel to y = -x + 1,
    the area of the triangle formed by this line and the coordinate axes is 50. -/
theorem triangle_area_parallel_line :
  ∃ k b : ℝ, k = -1 ∧ (8 * k + b = 2) ∧ (1/2 * 10 * 10 = 50) :=
sorry

end triangle_area_parallel_line_l457_45783


namespace simplify_fraction_l457_45799

theorem simplify_fraction :
  10 * (15 / 8) * (-40 / 45) = -(50 / 3) :=
sorry

end simplify_fraction_l457_45799


namespace find_positive_integer_pair_l457_45768

noncomputable def quadratic_has_rational_solutions (d : ℤ) : Prop :=
  ∃ x : ℚ, 7 * x^2 + 13 * x + d = 0

theorem find_positive_integer_pair :
  ∃ (d1 d2 : ℕ), 
  d1 > 0 ∧ d2 > 0 ∧ 
  quadratic_has_rational_solutions d1 ∧ quadratic_has_rational_solutions d2 ∧ 
  d1 * d2 = 2 := 
sorry -- Proof left as an exercise

end find_positive_integer_pair_l457_45768


namespace expression_equals_one_l457_45746

def evaluate_expression : ℕ :=
  3 * (3 * (3 * (3 * (3 - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1

theorem expression_equals_one : evaluate_expression = 1 := by
  sorry

end expression_equals_one_l457_45746


namespace range_of_m_l457_45729

def p (m : ℝ) : Prop :=
  ∀ (x : ℝ), 2^x - m + 1 > 0

def q (m : ℝ) : Prop :=
  5 - 2 * m > 1

theorem range_of_m (m : ℝ) (hpq : p m ∧ q m) : m ≤ 1 := sorry

end range_of_m_l457_45729


namespace slope_of_line_l457_45790

theorem slope_of_line
  (m : ℝ)
  (b : ℝ)
  (h1 : b = 4)
  (h2 : ∀ x y : ℝ, y = m * x + b → (x = 199 ∧ y = 800) → True) :
  m = 4 :=
by
  sorry

end slope_of_line_l457_45790


namespace simplify_polynomial_l457_45722

theorem simplify_polynomial (y : ℝ) :
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + y ^ 10 + 2 * y ^ 9) =
  15 * y ^ 13 - y ^ 12 - 3 * y ^ 11 + 4 * y ^ 10 - 4 * y ^ 9 := 
by
  sorry

end simplify_polynomial_l457_45722


namespace find_c_l457_45734

theorem find_c (a b c : ℤ) (h1 : a + b * c = 2017) (h2 : b + c * a = 8) :
  c = -6 ∨ c = 0 ∨ c = 2 ∨ c = 8 :=
by 
  sorry

end find_c_l457_45734


namespace math_problem_proof_l457_45784

def ratio_area_BFD_square_ABCE (x : ℝ) (AF FE DE CD : ℝ) (h1 : AF = FE / 3) (h2 : CD = 3 * DE) : Prop :=
  let AE := (AF + FE)
  let area_square := (AE)^2
  let area_triangle_BFD := area_square - (1/2 * AF * (AE - FE) + 1/2 * (AE - FE) * FE + 1/2 * DE * CD)
  (area_triangle_BFD / area_square) = (1/16)
  
theorem math_problem_proof (x AF FE DE CD : ℝ) (h1 : AF = FE / 3) (h2 : CD = 3 * DE) (area_ratio : area_triangle_BFD / area_square = 1/16) : ratio_area_BFD_square_ABCE x AF FE DE CD h1 h2 :=
sorry

end math_problem_proof_l457_45784


namespace determine_a_for_quadratic_l457_45787

theorem determine_a_for_quadratic (a : ℝ) : 
  (∃ x : ℝ, 3 * x ^ (a - 1) - x = 5 ∧ a - 1 = 2) → a = 3 := 
sorry

end determine_a_for_quadratic_l457_45787


namespace distance_between_trees_l457_45778

theorem distance_between_trees 
  (rows columns : ℕ)
  (boundary_distance garden_length d : ℝ)
  (h_rows : rows = 10)
  (h_columns : columns = 12)
  (h_boundary_distance : boundary_distance = 5)
  (h_garden_length : garden_length = 32) :
  (9 * d + 2 * boundary_distance = garden_length) → 
  d = 22 / 9 := 
by 
  intros h_eq
  sorry

end distance_between_trees_l457_45778


namespace tic_tac_toe_tie_probability_l457_45739

theorem tic_tac_toe_tie_probability (john_wins martha_wins : ℚ) 
  (hj : john_wins = 4 / 9) 
  (hm : martha_wins = 5 / 12) : 
  1 - (john_wins + martha_wins) = 5 / 36 := 
by {
  /- insert proof here -/
  sorry
}

end tic_tac_toe_tie_probability_l457_45739


namespace circle_area_l457_45753

-- Condition: Given the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 10 * x + 4 * y + 20 = 0

-- Theorem: The area enclosed by the given circle equation is 9π
theorem circle_area : ∀ x y : ℝ, circle_eq x y → ∃ A : ℝ, A = 9 * Real.pi :=
by
  intros
  sorry

end circle_area_l457_45753


namespace four_digit_div_90_count_l457_45708

theorem four_digit_div_90_count :
  ∃ n : ℕ, n = 10 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 → ab % 9 = 0 → 
  (10 * ab + 90) % 90 = 0 ∧ 1000 ≤ 10 * ab + 90 ∧ 10 * ab + 90 < 10000) :=
sorry

end four_digit_div_90_count_l457_45708


namespace principal_amount_l457_45749

theorem principal_amount (P : ℝ) (SI : ℝ) (R : ℝ) (T : ℝ)
  (h1 : SI = (P * R * T) / 100)
  (h2 : SI = 640)
  (h3 : R = 8)
  (h4 : T = 2) :
  P = 4000 :=
sorry

end principal_amount_l457_45749


namespace first_number_Harold_says_l457_45716

/-
  Define each student's sequence of numbers.
  - Alice skips every 4th number.
  - Barbara says numbers that Alice didn't say, skipping every 4th in her sequence.
  - Subsequent students follow the same rule.
  - Harold picks the smallest prime number not said by any student.
-/

def is_skipped_by_Alice (n : Nat) : Prop :=
  n % 4 ≠ 0

def is_skipped_by_Barbara (n : Nat) : Prop :=
  is_skipped_by_Alice n ∧ (n / 4) % 4 ≠ 3

def is_skipped_by_Candice (n : Nat) : Prop :=
  is_skipped_by_Barbara n ∧ (n / 16) % 4 ≠ 3

def is_skipped_by_Debbie (n : Nat) : Prop :=
  is_skipped_by_Candice n ∧ (n / 64) % 4 ≠ 3

def is_skipped_by_Eliza (n : Nat) : Prop :=
  is_skipped_by_Debbie n ∧ (n / 256) % 4 ≠ 3

def is_skipped_by_Fatima (n : Nat) : Prop :=
  is_skipped_by_Eliza n ∧ (n / 1024) % 4 ≠ 3

def is_skipped_by_Grace (n : Nat) : Prop :=
  is_skipped_by_Fatima n

def is_skipped_by_anyone (n : Nat) : Prop :=
  ¬ is_skipped_by_Alice n ∨ ¬ is_skipped_by_Barbara n ∨ ¬ is_skipped_by_Candice n ∨
  ¬ is_skipped_by_Debbie n ∨ ¬ is_skipped_by_Eliza n ∨ ¬ is_skipped_by_Fatima n ∨
  ¬ is_skipped_by_Grace n

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ (m : Nat), m ∣ n → m = 1 ∨ m = n

theorem first_number_Harold_says : ∃ n : Nat, is_prime n ∧ ¬ is_skipped_by_anyone n ∧ n = 11 := by
  sorry

end first_number_Harold_says_l457_45716


namespace cost_of_each_nose_spray_l457_45720

def total_nose_sprays : ℕ := 10
def total_cost : ℝ := 15
def buy_one_get_one_free : Bool := true

theorem cost_of_each_nose_spray :
  buy_one_get_one_free = true →
  total_nose_sprays = 10 →
  total_cost = 15 →
  (total_cost / (total_nose_sprays / 2)) = 3 :=
by
  intros h1 h2 h3
  sorry

end cost_of_each_nose_spray_l457_45720


namespace area_of_triangle_given_conditions_l457_45796

noncomputable def area_triangle_ABC (a b B : ℝ) : ℝ :=
  0.5 * a * b * Real.sin B

theorem area_of_triangle_given_conditions :
  area_triangle_ABC 2 (Real.sqrt 3) (Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end area_of_triangle_given_conditions_l457_45796


namespace percent_red_prob_l457_45771

-- Define the conditions
def initial_red := 2
def initial_blue := 4
def additional_red := 2
def additional_blue := 2
def total_balloons := initial_red + initial_blue + additional_red + additional_blue
def total_red := initial_red + additional_red

-- State the theorem
theorem percent_red_prob : (total_red.toFloat / total_balloons.toFloat) * 100 = 40 :=
by
  sorry

end percent_red_prob_l457_45771


namespace smaller_angle_at_7_15_l457_45701

theorem smaller_angle_at_7_15 
  (hour_hand_rate : ℕ → ℝ)
  (minute_hand_rate : ℕ → ℝ)
  (hour_time : ℕ)
  (minute_time : ℕ)
  (top_pos : ℝ)
  (smaller_angle : ℝ) 
  (h1 : hour_hand_rate hour_time + (minute_time/60) * hour_hand_rate hour_time = 217.5)
  (h2 : minute_hand_rate minute_time = 90.0)
  (h3 : |217.5 - 90.0| = smaller_angle) :
  smaller_angle = 127.5 :=
by
  sorry

end smaller_angle_at_7_15_l457_45701


namespace gcd_calculation_l457_45750

def gcd_36_45_495 : ℕ :=
  Int.gcd 36 (Int.gcd 45 495)

theorem gcd_calculation : gcd_36_45_495 = 9 := by
  sorry

end gcd_calculation_l457_45750


namespace sasha_remaining_questions_l457_45788

theorem sasha_remaining_questions
  (qph : ℕ) (total_questions : ℕ) (hours_worked : ℕ)
  (h_qph : qph = 15) (h_total_questions : total_questions = 60) (h_hours_worked : hours_worked = 2) :
  total_questions - (qph * hours_worked) = 30 :=
by
  sorry

end sasha_remaining_questions_l457_45788


namespace circle_line_distance_difference_l457_45756

/-- We define the given circle and line and prove the difference between maximum and minimum distances
    from any point on the circle to the line is 5√2. -/
theorem circle_line_distance_difference :
  (∀ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 10 = 0) →
  (∀ (x y : ℝ), x + y - 8 = 0) →
  ∃ (d : ℝ), d = 5 * Real.sqrt 2 :=
by
  sorry

end circle_line_distance_difference_l457_45756


namespace inequality_solution_l457_45776

theorem inequality_solution (x : ℝ) :
    (x < 1 ∨ (3 < x ∧ x < 4) ∨ (4 < x ∧ x < 5) ∨ (5 < x ∧ x < 6) ∨ x > 6) ↔
    ((x - 1) * (x - 3) * (x - 4) / ((x - 2) * (x - 5) * (x - 6)) > 0) := by
  sorry

end inequality_solution_l457_45776


namespace dot_product_of_a_and_b_is_correct_l457_45732

-- Define vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -1)

-- Define dot product for ℝ × ℝ vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem statement (proof can be omitted with sorry)
theorem dot_product_of_a_and_b_is_correct : dot_product a b = -4 :=
by
  -- proof goes here, omitted for now
  sorry

end dot_product_of_a_and_b_is_correct_l457_45732


namespace find_smaller_number_l457_45762

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 18) (h2 : a * b = 45) : a = 3 ∨ b = 3 := 
by 
  -- Proof would go here
  sorry

end find_smaller_number_l457_45762


namespace linear_equation_solution_l457_45717

theorem linear_equation_solution (a : ℝ) (x y : ℝ) 
    (h : (a - 2) * x^(|a| - 1) + 3 * y = 1) 
    (h1 : ∀ (x y : ℝ), (a - 2) ≠ 0)
    (h2 : |a| - 1 = 1) : a = -2 :=
by
  sorry

end linear_equation_solution_l457_45717


namespace scientific_notation_conversion_l457_45758

theorem scientific_notation_conversion :
  216000 = 2.16 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l457_45758


namespace lisa_flight_distance_l457_45727

-- Define the given speed and time
def speed : ℝ := 32
def time : ℝ := 8

-- Define the distance formula
def distance (v : ℝ) (t : ℝ) : ℝ := v * t

-- State the theorem to be proved
theorem lisa_flight_distance : distance speed time = 256 := by
sorry

end lisa_flight_distance_l457_45727


namespace M_subset_P_l457_45748

universe u

-- Definitions of the sets
def U : Set ℝ := {x : ℝ | True}
def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x^2 > 1}

-- Proof statement
theorem M_subset_P : M ⊆ P := by
  sorry

end M_subset_P_l457_45748


namespace speed_of_train_l457_45781

theorem speed_of_train (length : ℝ) (time : ℝ) (conversion_factor : ℝ) (speed_kmh : ℝ) 
  (h1 : length = 240) (h2 : time = 16) (h3 : conversion_factor = 3.6) :
  speed_kmh = (length / time) * conversion_factor := 
sorry

end speed_of_train_l457_45781


namespace solution_set_for_inequality_l457_45725

theorem solution_set_for_inequality :
  {x : ℝ | -x^2 + 4 * x + 5 < 0} = {x : ℝ | x > 5 ∨ x < -1} :=
by
  sorry

end solution_set_for_inequality_l457_45725


namespace measure_of_angle_XPM_l457_45709

-- Definitions based on given conditions
variables (X Y Z L M N P : Type)
variables (a b c : ℝ) -- Angles are represented in degrees
variables [DecidableEq X] [DecidableEq Y] [DecidableEq Z]

-- Triangle XYZ with angle bisectors XL, YM, and ZN meeting at incenter P
-- Given angle XYZ in degrees
def angle_XYZ : ℝ := 46

-- Incenter angle properties
axiom angle_bisector_XL (angle_XYP : ℝ) : angle_XYP = angle_XYZ / 2
axiom angle_bisector_YM (angle_YXP : ℝ) : ∃ (angle_YXZ : ℝ), angle_YXP = angle_YXZ / 2

-- The proposition we need to prove
theorem measure_of_angle_XPM : ∃ (angle_XPM : ℝ), angle_XPM = 67 := 
by {
  sorry
}

end measure_of_angle_XPM_l457_45709


namespace range_of_3x_plus_2y_l457_45737

theorem range_of_3x_plus_2y (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : -1 ≤ y ∧ y ≤ 4) :
  1 ≤ 3 * x + 2 * y ∧ 3 * x + 2 * y ≤ 17 :=
sorry

end range_of_3x_plus_2y_l457_45737


namespace yeast_cells_at_10_30_l457_45789

def yeast_population (initial_population : ℕ) (intervals : ℕ) (growth_rate : ℝ) (decay_rate : ℝ) : ℝ :=
  initial_population * (growth_rate * (1 - decay_rate)) ^ intervals

theorem yeast_cells_at_10_30 :
  yeast_population 50 6 3 0.10 = 52493 := by
  sorry

end yeast_cells_at_10_30_l457_45789


namespace fraction_meaningful_l457_45777

theorem fraction_meaningful (x : ℝ) : (x - 2 ≠ 0) ↔ (x ≠ 2) :=
by 
  sorry

end fraction_meaningful_l457_45777


namespace problem_l457_45726

-- Conditions
def a_n (n : ℕ) : ℚ := (1/3)^(n-1)

def b_n (n : ℕ) : ℚ := n * (1/3)^n

-- Sums over the first n terms
def S_n (n : ℕ) : ℚ := (3/2) - (1/2) * (1/3)^n

def T_n (n : ℕ) : ℚ := (3/4) - (1/4) * (1/3)^n - (n/2) * (1/3)^n

-- Problem: Prove T_n < S_n / 2
theorem problem (n : ℕ) : T_n n < S_n n / 2 :=
by sorry

end problem_l457_45726


namespace fewer_mpg_in_city_l457_45780

def city_mpg := 14
def city_distance := 336
def highway_distance := 480

def tank_size := city_distance / city_mpg
def highway_mpg := highway_distance / tank_size
def fewer_mpg := highway_mpg - city_mpg

theorem fewer_mpg_in_city : fewer_mpg = 6 := by
  sorry

end fewer_mpg_in_city_l457_45780


namespace remaining_money_l457_45772

-- Defining the conditions
def orchids_qty := 20
def price_per_orchid := 50
def chinese_money_plants_qty := 15
def price_per_plant := 25
def worker_qty := 2
def salary_per_worker := 40
def cost_of_pots := 150

-- Earnings and expenses calculations
def earnings_from_orchids := orchids_qty * price_per_orchid
def earnings_from_plants := chinese_money_plants_qty * price_per_plant
def total_earnings := earnings_from_orchids + earnings_from_plants

def worker_expenses := worker_qty * salary_per_worker
def total_expenses := worker_expenses + cost_of_pots

-- The proof problem
theorem remaining_money : total_earnings - total_expenses = 1145 :=
by
  sorry

end remaining_money_l457_45772


namespace proof_negation_l457_45723

-- Definitions of rational and real numbers
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- Proposition stating the existence of an irrational number that is rational
def original_proposition : Prop :=
  ∃ x : ℝ, is_irrational x ∧ is_rational x

-- Negation of the original proposition
def negated_proposition : Prop :=
  ∀ x : ℝ, is_irrational x → ¬ is_rational x

theorem proof_negation : ¬ original_proposition = negated_proposition := 
sorry

end proof_negation_l457_45723


namespace busy_squirrels_count_l457_45755

variable (B : ℕ)
variable (busy_squirrel_nuts_per_day : ℕ := 30)
variable (sleepy_squirrel_nuts_per_day : ℕ := 20)
variable (days : ℕ := 40)
variable (total_nuts : ℕ := 3200)

theorem busy_squirrels_count : busy_squirrel_nuts_per_day * days * B + sleepy_squirrel_nuts_per_day * days = total_nuts → B = 2 := by
  sorry

end busy_squirrels_count_l457_45755


namespace ratio_Y_to_Z_l457_45738

variables (X Y Z : ℕ)

def population_relation1 (X Y : ℕ) : Prop := X = 3 * Y
def population_relation2 (X Z : ℕ) : Prop := X = 6 * Z

theorem ratio_Y_to_Z (h1 : population_relation1 X Y) (h2 : population_relation2 X Z) : Y / Z = 2 :=
  sorry

end ratio_Y_to_Z_l457_45738


namespace same_sign_iff_product_positive_different_sign_iff_product_negative_l457_45730

variable (a b : ℝ)

theorem same_sign_iff_product_positive :
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0)) ↔ (a * b > 0) :=
sorry

theorem different_sign_iff_product_negative :
  ((a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) ↔ (a * b < 0) :=
sorry

end same_sign_iff_product_positive_different_sign_iff_product_negative_l457_45730


namespace min_flowers_for_bouquets_l457_45741

open Classical

noncomputable def minimum_flowers (types : ℕ) (flowers_for_bouquet : ℕ) (bouquets : ℕ) : ℕ := 
  sorry

theorem min_flowers_for_bouquets : minimum_flowers 6 5 10 = 70 := 
  sorry

end min_flowers_for_bouquets_l457_45741


namespace petes_original_number_l457_45764

theorem petes_original_number (x : ℤ) (h : 5 * (3 * x - 6) = 195) : x = 15 :=
sorry

end petes_original_number_l457_45764


namespace john_finances_l457_45754

theorem john_finances :
  let total_first_year := 10000
  let tuition_percent := 0.4
  let room_board_percent := 0.35
  let textbook_transport_percent := 0.25
  let tuition_increase := 0.06
  let room_board_increase := 0.03
  let aid_first_year := 0.25
  let aid_increase := 0.02

  let tuition_first_year := total_first_year * tuition_percent
  let room_board_first_year := total_first_year * room_board_percent
  let textbook_transport_first_year := total_first_year * textbook_transport_percent

  let tuition_second_year := tuition_first_year * (1 + tuition_increase)
  let room_board_second_year := room_board_first_year * (1 + room_board_increase)
  let financial_aid_second_year := tuition_second_year * (aid_first_year + aid_increase)

  let tuition_third_year := tuition_second_year * (1 + tuition_increase)
  let room_board_third_year := room_board_second_year * (1 + room_board_increase)
  let financial_aid_third_year := tuition_third_year * (aid_first_year + 2 * aid_increase)

  let total_cost_first_year := 
      (tuition_first_year - tuition_first_year * aid_first_year) +
      room_board_first_year + 
      textbook_transport_first_year

  let total_cost_second_year :=
      (tuition_second_year - financial_aid_second_year) +
      room_board_second_year +
      textbook_transport_first_year

  let total_cost_third_year :=
      (tuition_third_year - financial_aid_third_year) +
      room_board_third_year +
      textbook_transport_first_year

  total_cost_first_year = 9000 ∧
  total_cost_second_year = 9200.20 ∧
  total_cost_third_year = 9404.17 := 
by
  sorry

end john_finances_l457_45754


namespace problem_range_of_k_l457_45798

theorem problem_range_of_k (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11 * x + (30 + k) = 0 → x > 5) → (0 < k ∧ k ≤ 1 / 4) :=
by
  sorry

end problem_range_of_k_l457_45798


namespace right_triangle_excircle_incircle_l457_45712

theorem right_triangle_excircle_incircle (a b c r r_a : ℝ) (h : a^2 + b^2 = c^2) :
  (r = (a + b - c) / 2) → (r_a = (b + c - a) / 2) → r_a = 2 * r :=
by
  intros hr hra
  sorry

end right_triangle_excircle_incircle_l457_45712


namespace combined_age_of_siblings_l457_45765

theorem combined_age_of_siblings (a s h : ℕ) (h1 : a = 15) (h2 : s = 3 * a) (h3 : h = 4 * s) : a + s + h = 240 :=
by
  sorry

end combined_age_of_siblings_l457_45765


namespace repayment_days_least_integer_l457_45711

theorem repayment_days_least_integer:
  ∀ (x : ℤ), (20 + 2 * x ≥ 60) → (x ≥ 20) :=
by
  intro x
  intro h
  sorry

end repayment_days_least_integer_l457_45711


namespace scientific_notation_of_0_0000000005_l457_45767

theorem scientific_notation_of_0_0000000005 : 0.0000000005 = 5 * 10^(-10) :=
by {
  sorry
}

end scientific_notation_of_0_0000000005_l457_45767


namespace average_difference_l457_45759

theorem average_difference (F1 L1 F2 L2 : ℤ) (H1 : F1 = 200) (H2 : L1 = 400) (H3 : F2 = 100) (H4 : L2 = 200) :
  (F1 + L1) / 2 - (F2 + L2) / 2 = 150 := 
by 
  sorry

end average_difference_l457_45759


namespace square_area_inscribed_in_parabola_l457_45710

def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

theorem square_area_inscribed_in_parabola :
  ∃ s : ℝ, s = (-1 + Real.sqrt 5) ∧ (2 * s)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end square_area_inscribed_in_parabola_l457_45710


namespace speed_boat_25_kmph_l457_45794

noncomputable def speed_of_boat_in_still_water (V_s : ℝ) (time : ℝ) (distance : ℝ) : ℝ :=
  let V_d := distance / time
  V_d - V_s

theorem speed_boat_25_kmph (h_vs : V_s = 5) (h_time : time = 4) (h_distance : distance = 120) :
  speed_of_boat_in_still_water V_s time distance = 25 :=
by
  rw [h_vs, h_time, h_distance]
  unfold speed_of_boat_in_still_water
  simp
  norm_num

end speed_boat_25_kmph_l457_45794


namespace polynomial_sum_correct_l457_45785

def f (x : ℝ) : ℝ := -4 * x^3 + 2 * x^2 - x - 5
def g (x : ℝ) : ℝ := -6 * x^3 - 7 * x^2 + 4 * x - 2
def h (x : ℝ) : ℝ := 2 * x^3 + 8 * x^2 + 6 * x + 3
def sum_polynomials (x : ℝ) : ℝ := -8 * x^3 + 3 * x^2 + 9 * x - 4

theorem polynomial_sum_correct (x : ℝ) : f x + g x + h x = sum_polynomials x :=
by sorry

end polynomial_sum_correct_l457_45785


namespace sum_infinite_series_l457_45786

theorem sum_infinite_series : 
  ∑' n : ℕ, (2 * (n + 1) + 3) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2) * ((n + 1) + 3)) = 9 / 4 := by
  sorry

end sum_infinite_series_l457_45786


namespace students_transferred_l457_45707

theorem students_transferred (initial_students : ℝ) (students_left : ℝ) (end_students : ℝ) :
  initial_students = 42.0 →
  students_left = 4.0 →
  end_students = 28.0 →
  initial_students - students_left - end_students = 10.0 :=
by
  intros
  sorry

end students_transferred_l457_45707
