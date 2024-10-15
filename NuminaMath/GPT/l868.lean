import Mathlib

namespace NUMINAMATH_GPT_percentage_decrease_in_larger_angle_l868_86839

-- Define the angles and conditions
def angles_complementary (A B : ℝ) : Prop := A + B = 90
def angle_ratio (A B : ℝ) : Prop := A / B = 1 / 2
def small_angle_increase (A A' : ℝ) : Prop := A' = A * 1.2
def large_angle_new (A' B' : ℝ) : Prop := A' + B' = 90

-- Prove percentage decrease in larger angle
theorem percentage_decrease_in_larger_angle (A B A' B' : ℝ) 
    (h1 : angles_complementary A B)
    (h2 : angle_ratio A B)
    (h3 : small_angle_increase A A')
    (h4 : large_angle_new A' B')
    : (B - B') / B * 100 = 10 :=
sorry

end NUMINAMATH_GPT_percentage_decrease_in_larger_angle_l868_86839


namespace NUMINAMATH_GPT_max_L_shaped_figures_in_5x7_rectangle_l868_86883

def L_shaped_figure : Type := ℕ

def rectangle_area := 5 * 7

def l_shape_area := 3

def max_l_shapes_in_rectangle (rect_area : ℕ) (l_area : ℕ) : ℕ := rect_area / l_area

theorem max_L_shaped_figures_in_5x7_rectangle : max_l_shapes_in_rectangle rectangle_area l_shape_area = 11 :=
by
  sorry

end NUMINAMATH_GPT_max_L_shaped_figures_in_5x7_rectangle_l868_86883


namespace NUMINAMATH_GPT_l_shaped_area_l868_86897

theorem l_shaped_area (A B C D : Type) (side_abcd: ℝ) (side_small_1: ℝ) (side_small_2: ℝ)
  (area_abcd : side_abcd = 6)
  (area_small_1 : side_small_1 = 2)
  (area_small_2 : side_small_2 = 4)
  (no_overlap : true) :
  side_abcd * side_abcd - (side_small_1 * side_small_1 + side_small_2 * side_small_2) = 16 := by
  sorry

end NUMINAMATH_GPT_l_shaped_area_l868_86897


namespace NUMINAMATH_GPT_vertex_of_parabola_l868_86848

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x + 9)^2 - 3

-- State the theorem to prove
theorem vertex_of_parabola : ∃ h k : ℝ, (h = -9 ∧ k = -3) ∧ (parabola h = k) :=
by sorry

end NUMINAMATH_GPT_vertex_of_parabola_l868_86848


namespace NUMINAMATH_GPT_bananas_oranges_equiv_l868_86831

def bananas_apples_equiv (x y : ℕ) : Prop :=
  4 * x = 3 * y

def apples_oranges_equiv (w z : ℕ) : Prop :=
  9 * w = 5 * z

theorem bananas_oranges_equiv (x y w z : ℕ) (h1 : bananas_apples_equiv x y) (h2 : apples_oranges_equiv y z) :
  bananas_apples_equiv 24 18 ∧ apples_oranges_equiv 18 10 :=
by sorry

end NUMINAMATH_GPT_bananas_oranges_equiv_l868_86831


namespace NUMINAMATH_GPT_compare_star_l868_86892

def star (m n : ℤ) : ℤ := (m + 2) * 3 - n

theorem compare_star : star 2 (-2) > star (-2) 2 := 
by sorry

end NUMINAMATH_GPT_compare_star_l868_86892


namespace NUMINAMATH_GPT_egg_laying_hens_l868_86821

theorem egg_laying_hens (total_chickens : ℕ) (roosters : ℕ) (non_laying_hens : ℕ) :
  total_chickens = 325 →
  roosters = 28 →
  non_laying_hens = 20 →
  (total_chickens - roosters - non_laying_hens = 277) :=
by
  intros
  sorry

end NUMINAMATH_GPT_egg_laying_hens_l868_86821


namespace NUMINAMATH_GPT_trapezoidal_park_no_solution_l868_86874

theorem trapezoidal_park_no_solution :
  (∃ b1 b2 : ℕ, 2 * 1800 = 40 * (b1 + b2) ∧ (∃ m : ℕ, b1 = 5 * (2 * m + 1)) ∧ (∃ n : ℕ, b2 = 2 * n)) → false :=
by
  sorry

end NUMINAMATH_GPT_trapezoidal_park_no_solution_l868_86874


namespace NUMINAMATH_GPT_number_of_wins_and_losses_l868_86836

theorem number_of_wins_and_losses (x y : ℕ) (h1 : x + y = 15) (h2 : 3 * x + y = 41) :
  x = 13 ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_number_of_wins_and_losses_l868_86836


namespace NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l868_86890

theorem arithmetic_sequence_tenth_term (a d : ℤ) 
  (h1 : a + 2 * d = 23) 
  (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l868_86890


namespace NUMINAMATH_GPT_polygon_diagonals_with_restriction_l868_86867

def num_sides := 150

def total_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

def restricted_diagonals (n : ℕ) : ℕ :=
  n * 150 / 4

def valid_diagonals (n : ℕ) : ℕ :=
  total_diagonals n - restricted_diagonals n

theorem polygon_diagonals_with_restriction : valid_diagonals num_sides = 5400 :=
by
  sorry

end NUMINAMATH_GPT_polygon_diagonals_with_restriction_l868_86867


namespace NUMINAMATH_GPT_election_winner_votes_l868_86834

-- Define the conditions and question in Lean 4
theorem election_winner_votes (V : ℝ) (h1 : V > 0) 
  (h2 : 0.54 * V - 0.46 * V = 288) : 0.54 * V = 1944 :=
by
  sorry

end NUMINAMATH_GPT_election_winner_votes_l868_86834


namespace NUMINAMATH_GPT_al_told_the_truth_l868_86854

-- Definitions of G, S, and B based on each pirate's claim
def tom_G := 10
def tom_S := 8
def tom_B := 11

def al_G := 9
def al_S := 11
def al_B := 10

def pit_G := 10
def pit_S := 10
def pit_B := 9

def jim_G := 8
def jim_S := 10
def jim_B := 11

-- Condition that the total number of coins is 30
def total_coins (G : ℕ) (S : ℕ) (B : ℕ) : Prop := G + S + B = 30

-- The assertion that only Al told the truth
theorem al_told_the_truth :
  (total_coins tom_G tom_S tom_B → false) →
  (total_coins al_G al_S al_B) →
  (total_coins pit_G pit_S pit_B → false) →
  (total_coins jim_G jim_S jim_B → false) →
  true :=
by
  intros
  sorry

end NUMINAMATH_GPT_al_told_the_truth_l868_86854


namespace NUMINAMATH_GPT_annual_interest_rate_is_12_percent_l868_86855

theorem annual_interest_rate_is_12_percent
  (P : ℕ := 750000)
  (I : ℕ := 37500)
  (t : ℕ := 5)
  (months_in_year : ℕ := 12)
  (annual_days : ℕ := 360)
  (days_per_month : ℕ := 30) :
  ∃ r : ℚ, (r * 100 * months_in_year = 12) ∧ I = P * r * t := 
sorry

end NUMINAMATH_GPT_annual_interest_rate_is_12_percent_l868_86855


namespace NUMINAMATH_GPT_sandy_final_fish_l868_86830

theorem sandy_final_fish :
  let Initial_fish := 26
  let Bought_fish := 6
  let Given_away_fish := 10
  let Babies_fish := 15
  let Final_fish := Initial_fish + Bought_fish - Given_away_fish + Babies_fish
  Final_fish = 37 :=
by
  sorry

end NUMINAMATH_GPT_sandy_final_fish_l868_86830


namespace NUMINAMATH_GPT_inequality_solution_l868_86895

theorem inequality_solution {x : ℝ} :
  ((x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x)) ↔
  ((x - 2) * (x - 3) * (x - 4) / ((x - 1) * (x - 5) * (x - 6)) > 0) := sorry

end NUMINAMATH_GPT_inequality_solution_l868_86895


namespace NUMINAMATH_GPT_range_x_f_inequality_l868_86816

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - |x|) + 1 / (x^2 + 1)

theorem range_x_f_inequality :
  (∀ x : ℝ, f (2 * x + 1) ≥ f x) ↔ x ∈ Set.Icc (-1 : ℝ) (-1 / 3) := sorry

end NUMINAMATH_GPT_range_x_f_inequality_l868_86816


namespace NUMINAMATH_GPT_average_rate_of_change_l868_86853

def f (x : ℝ) : ℝ := x^2 - 1

theorem average_rate_of_change : (f 1.1) - (f 1) / (1.1 - 1) = 2.1 :=
by
  sorry

end NUMINAMATH_GPT_average_rate_of_change_l868_86853


namespace NUMINAMATH_GPT_vasya_gift_choices_l868_86838

theorem vasya_gift_choices :
  let cars := 7
  let construction_sets := 5
  (cars * construction_sets + Nat.choose cars 2 + Nat.choose construction_sets 2) = 66 :=
by
  sorry

end NUMINAMATH_GPT_vasya_gift_choices_l868_86838


namespace NUMINAMATH_GPT_decreasing_implies_bound_l868_86888

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  - (1 / 2) * x ^ 2 + b * Real.log x

theorem decreasing_implies_bound (b : ℝ) :
  (∀ x > 2, -x + b / x ≤ 0) → b ≤ 4 :=
  sorry

end NUMINAMATH_GPT_decreasing_implies_bound_l868_86888


namespace NUMINAMATH_GPT_parabola_opens_downward_iff_l868_86828

theorem parabola_opens_downward_iff (m : ℝ) : (m - 1 < 0) ↔ (m < 1) :=
by
  sorry

end NUMINAMATH_GPT_parabola_opens_downward_iff_l868_86828


namespace NUMINAMATH_GPT_find_b_l868_86894

-- Let's define the real numbers and the conditions given.
variables (b y a : ℝ)

-- Conditions from the problem
def condition1 := abs (b - y) = b + y - a
def condition2 := abs (b + y) = b + a

-- The goal is to find the value of b
theorem find_b (h1 : condition1 b y a) (h2 : condition2 b y a) : b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l868_86894


namespace NUMINAMATH_GPT_increasing_function_shape_implies_number_l868_86891

variable {I : Set ℝ} {f : ℝ → ℝ}

theorem increasing_function_shape_implies_number (h : ∀ (x₁ x₂ : ℝ), x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ → f x₁ < f x₂) 
: ∀ (x₁ x₂ : ℝ), x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ → f x₁ < f x₂ :=
sorry

end NUMINAMATH_GPT_increasing_function_shape_implies_number_l868_86891


namespace NUMINAMATH_GPT_arable_land_decrease_max_l868_86844

theorem arable_land_decrease_max
  (A₀ : ℕ := 100000)
  (grain_yield_increase : ℝ := 1.22)
  (per_capita_increase : ℝ := 1.10)
  (pop_growth_rate : ℝ := 0.01)
  (years : ℕ := 10) :
  ∃ (max_decrease : ℕ), max_decrease = 4 := sorry

end NUMINAMATH_GPT_arable_land_decrease_max_l868_86844


namespace NUMINAMATH_GPT_total_supervisors_correct_l868_86808

-- Define the number of supervisors on each bus
def bus_supervisors : List ℕ := [4, 5, 3, 6, 7]

-- Define the total number of supervisors
def total_supervisors := bus_supervisors.sum

-- State the theorem to prove that the total number of supervisors is 25
theorem total_supervisors_correct : total_supervisors = 25 :=
by
  sorry -- Proof is to be completed

end NUMINAMATH_GPT_total_supervisors_correct_l868_86808


namespace NUMINAMATH_GPT_jimmy_fill_pool_time_l868_86807

theorem jimmy_fill_pool_time (pool_gallons : ℕ) (bucket_gallons : ℕ) (time_per_trip_sec : ℕ) (sec_per_min : ℕ) :
  pool_gallons = 84 → 
  bucket_gallons = 2 → 
  time_per_trip_sec = 20 → 
  sec_per_min = 60 → 
  (pool_gallons / bucket_gallons) * time_per_trip_sec / sec_per_min = 14 :=
by
  sorry

end NUMINAMATH_GPT_jimmy_fill_pool_time_l868_86807


namespace NUMINAMATH_GPT_find_utilities_second_l868_86850

def rent_first : ℝ := 800
def utilities_first : ℝ := 260
def distance_first : ℕ := 31
def rent_second : ℝ := 900
def distance_second : ℕ := 21
def cost_per_mile : ℝ := 0.58
def days_per_month : ℕ := 20
def cost_difference : ℝ := 76

-- Helper definitions
def driving_cost (distance : ℕ) : ℝ :=
  distance * days_per_month * cost_per_mile

def total_cost_first : ℝ :=
  rent_first + utilities_first + driving_cost distance_first

def total_cost_second_no_utilities : ℝ :=
  rent_second + driving_cost distance_second

theorem find_utilities_second :
  ∃ (utilities_second : ℝ),
  total_cost_first - total_cost_second_no_utilities = cost_difference →
  utilities_second = 200 :=
sorry

end NUMINAMATH_GPT_find_utilities_second_l868_86850


namespace NUMINAMATH_GPT_cost_per_bottle_l868_86842

theorem cost_per_bottle (cost_3_bottles cost_4_bottles : ℝ) (n_bottles : ℕ) 
  (h1 : cost_3_bottles = 1.50) (h2 : cost_4_bottles = 2) : 
  (cost_3_bottles / 3) = (cost_4_bottles / 4) ∧ (cost_3_bottles / 3) * n_bottles = 0.50 * n_bottles :=
by
  sorry

end NUMINAMATH_GPT_cost_per_bottle_l868_86842


namespace NUMINAMATH_GPT_sequence_is_decreasing_l868_86812

-- Define the sequence {a_n} using a recursive function
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ (∀ n, a (n + 1) = a n / (3 * a n + 1))

-- Define a condition ensuring the sequence a_n is decreasing
theorem sequence_is_decreasing (a : ℕ → ℝ) (h : seq a) : ∀ n, a (n + 1) < a n :=
by
  intro n
  sorry

end NUMINAMATH_GPT_sequence_is_decreasing_l868_86812


namespace NUMINAMATH_GPT_find_functions_satisfying_lcm_gcd_eq_l868_86882

noncomputable def satisfies_functional_equation (f : ℕ → ℕ) : Prop := 
  ∀ m n : ℕ, m > 0 ∧ n > 0 → f (m * n) = Nat.lcm m n * Nat.gcd (f m) (f n)

noncomputable def solution_form (f : ℕ → ℕ) : Prop := 
  ∃ k : ℕ, ∀ x : ℕ, f x = k * x

theorem find_functions_satisfying_lcm_gcd_eq (f : ℕ → ℕ) : 
  satisfies_functional_equation f ↔ solution_form f := 
sorry

end NUMINAMATH_GPT_find_functions_satisfying_lcm_gcd_eq_l868_86882


namespace NUMINAMATH_GPT_age_hence_l868_86829

theorem age_hence (A x : ℕ) (hA : A = 24) (hx : 4 * (A + x) - 4 * (A - 3) = A) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_age_hence_l868_86829


namespace NUMINAMATH_GPT_inequality_relation_l868_86823

noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log Q / Real.log 2

theorem inequality_relation : R < Q ∧ Q < P := by
  sorry

end NUMINAMATH_GPT_inequality_relation_l868_86823


namespace NUMINAMATH_GPT_campers_afternoon_l868_86875

def morning_campers : ℕ := 52
def additional_campers : ℕ := 9
def total_campers_afternoon : ℕ := morning_campers + additional_campers

theorem campers_afternoon : total_campers_afternoon = 61 :=
by
  sorry

end NUMINAMATH_GPT_campers_afternoon_l868_86875


namespace NUMINAMATH_GPT_number_of_extreme_points_zero_l868_86800

def f (x a : ℝ) : ℝ := x^3 + 3*x^2 + 4*x - a

theorem number_of_extreme_points_zero (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ∀ x, f x1 a = f x a → x = x1 ∨ x = x2) → False := 
by
  sorry

end NUMINAMATH_GPT_number_of_extreme_points_zero_l868_86800


namespace NUMINAMATH_GPT_houses_built_during_boom_l868_86865

-- Define initial and current number of houses
def initial_houses : ℕ := 1426
def current_houses : ℕ := 2000

-- Define the expected number of houses built during the boom
def expected_houses_built : ℕ := 574

-- The theorem to prove
theorem houses_built_during_boom : (current_houses - initial_houses) = expected_houses_built :=
by 
    sorry

end NUMINAMATH_GPT_houses_built_during_boom_l868_86865


namespace NUMINAMATH_GPT_small_cuboid_length_is_five_l868_86846

-- Define initial conditions
def large_cuboid_length : ℝ := 18
def large_cuboid_width : ℝ := 15
def large_cuboid_height : ℝ := 2
def num_small_cuboids : ℕ := 6
def small_cuboid_width : ℝ := 6
def small_cuboid_height : ℝ := 3

-- Theorem to prove the length of the smaller cuboid
theorem small_cuboid_length_is_five (small_cuboid_length : ℝ) 
  (h1 : large_cuboid_length * large_cuboid_width * large_cuboid_height 
          = num_small_cuboids * (small_cuboid_length * small_cuboid_width * small_cuboid_height)) :
  small_cuboid_length = 5 := by
  sorry

end NUMINAMATH_GPT_small_cuboid_length_is_five_l868_86846


namespace NUMINAMATH_GPT_min_value_cx_plus_dy_squared_l868_86878

theorem min_value_cx_plus_dy_squared
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (∃ (x y : ℝ), a * x^2 + b * y^2 = 1 ∧ ∀ (x y : ℝ), a * x^2 + b * y^2 = 1 → c * x + d * y^2 ≥ -c / a.sqrt) :=
sorry

end NUMINAMATH_GPT_min_value_cx_plus_dy_squared_l868_86878


namespace NUMINAMATH_GPT_eval_floor_expr_l868_86858

def frac_part1 : ℚ := (15 / 8)
def frac_part2 : ℚ := (11 / 3)
def square_frac1 : ℚ := frac_part1 ^ 2
def ceil_part : ℤ := ⌈square_frac1⌉
def add_frac2 : ℚ := ceil_part + frac_part2

theorem eval_floor_expr : (⌊add_frac2⌋ : ℤ) = 7 := 
sorry

end NUMINAMATH_GPT_eval_floor_expr_l868_86858


namespace NUMINAMATH_GPT_proof_problem_l868_86885

variables {a b c d e : ℝ}

theorem proof_problem (h1 : a * b^2 * c^3 * d^4 * e^5 < 0) (h2 : b^2 ≥ 0) (h3 : d^4 ≥ 0) :
  a * b^2 * c * d^4 * e < 0 :=
sorry

end NUMINAMATH_GPT_proof_problem_l868_86885


namespace NUMINAMATH_GPT_unique_tangent_circle_of_radius_2_l868_86856

noncomputable def is_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  dist c₁ c₂ = r₁ + r₂

theorem unique_tangent_circle_of_radius_2
    (C1_center C2_center C3_center : ℝ × ℝ)
    (h_C1_C2 : is_tangent C1_center C2_center 1 1)
    (h_C2_C3 : is_tangent C2_center C3_center 1 1)
    (h_C3_C1 : is_tangent C3_center C1_center 1 1):
    ∃! center : ℝ × ℝ, is_tangent center C1_center 2 1 ∧
                        is_tangent center C2_center 2 1 ∧
                        is_tangent center C3_center 2 1 := sorry

end NUMINAMATH_GPT_unique_tangent_circle_of_radius_2_l868_86856


namespace NUMINAMATH_GPT_pairs_condition_l868_86825

theorem pairs_condition (a b : ℕ) (prime_p : ∃ p, p = a^2 + b + 1 ∧ Nat.Prime p)
    (divides : ∀ p, p = a^2 + b + 1 → p ∣ (b^2 - a^3 - 1))
    (not_divides : ∀ p, p = a^2 + b + 1 → ¬ p ∣ (a + b - 1)^2) :
  ∃ x, x ≥ 2 ∧ a = 2 ^ x ∧ b = 2 ^ (2 * x) - 1 := sorry

end NUMINAMATH_GPT_pairs_condition_l868_86825


namespace NUMINAMATH_GPT_price_restoration_l868_86827

theorem price_restoration (P : Real) (hP : P > 0) :
  let new_price := 0.85 * P
  let required_increase := ((1 / 0.85) - 1) * 100
  required_increase = 17.65 :=
by 
  sorry

end NUMINAMATH_GPT_price_restoration_l868_86827


namespace NUMINAMATH_GPT_sqrt_0_54_in_terms_of_a_b_l868_86810

variable (a b : ℝ)

-- Conditions
def sqrt_two_eq_a : Prop := a = Real.sqrt 2
def sqrt_three_eq_b : Prop := b = Real.sqrt 3

-- The main statement to prove
theorem sqrt_0_54_in_terms_of_a_b (h1 : sqrt_two_eq_a a) (h2 : sqrt_three_eq_b b) :
  Real.sqrt 0.54 = 0.3 * a * b := sorry

end NUMINAMATH_GPT_sqrt_0_54_in_terms_of_a_b_l868_86810


namespace NUMINAMATH_GPT_lassis_from_12_mangoes_l868_86813

-- Conditions as definitions in Lean 4
def total_mangoes : ℕ := 12
def damaged_mango_ratio : ℕ := 1 / 6
def lassis_per_pair_mango : ℕ := 11

-- Equation to calculate the lassis
theorem lassis_from_12_mangoes : (total_mangoes - total_mangoes / 6) / 2 * lassis_per_pair_mango = 55 :=
by
  -- calculation steps should go here, but are omitted as per instructions
  sorry

end NUMINAMATH_GPT_lassis_from_12_mangoes_l868_86813


namespace NUMINAMATH_GPT_max_savings_theorem_band_members_theorem_selection_plans_theorem_l868_86806

/-- Given conditions for maximum savings calculation -/
def number_of_sets_purchased : ℕ := 75
def max_savings (cost_separate : ℕ) (cost_together : ℕ) : Prop :=
cost_separate - cost_together = 800

theorem max_savings_theorem : 
    ∃ cost_separate cost_together, 
    (cost_separate = 5600) ∧ (cost_together = 4800) → max_savings cost_separate cost_together := by
  sorry

/-- Given conditions for number of members in bands A and B -/
def conditions (x y : ℕ) : Prop :=
x + y = 75 ∧ 70 * x + 80 * y = 5600 ∧ x >= 40

theorem band_members_theorem :
    ∃ x y, conditions x y → (x = 40 ∧ y = 35) := by
  sorry

/-- Given conditions for possible selection plans for charity event -/
def heart_to_heart_activity (a b : ℕ) : Prop :=
3 * a + 5 * b = 65 ∧ a >= 5 ∧ b >= 5

theorem selection_plans_theorem :
    ∃ a b, heart_to_heart_activity a b → 
    ((a = 5 ∧ b = 10) ∨ (a = 10 ∧ b = 7)) := by
  sorry

end NUMINAMATH_GPT_max_savings_theorem_band_members_theorem_selection_plans_theorem_l868_86806


namespace NUMINAMATH_GPT_age_difference_l868_86822

theorem age_difference (A B C : ℕ) (hB : B = 14) (hBC : B = 2 * C) (hSum : A + B + C = 37) : A - B = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l868_86822


namespace NUMINAMATH_GPT_tank_capacity_l868_86887

theorem tank_capacity (V : ℝ) (initial_fraction final_fraction : ℝ) (added_water : ℝ)
  (h1 : initial_fraction = 1 / 4)
  (h2 : final_fraction = 3 / 4)
  (h3 : added_water = 208)
  (h4 : final_fraction - initial_fraction = 1 / 2)
  (h5 : (1 / 2) * V = added_water) :
  V = 416 :=
by
  -- Given: initial_fraction = 1/4, final_fraction = 3/4, added_water = 208
  -- Difference in fullness: 1/2
  -- Equation for volume: 1/2 * V = 208
  -- Hence, V = 416
  sorry

end NUMINAMATH_GPT_tank_capacity_l868_86887


namespace NUMINAMATH_GPT_emily_first_round_points_l868_86871

theorem emily_first_round_points (x : ℤ) 
  (second_round : ℤ := 33) 
  (last_round_loss : ℤ := 48) 
  (total_points_end : ℤ := 1) 
  (eqn : x + second_round - last_round_loss = total_points_end) : 
  x = 16 := 
by 
  sorry

end NUMINAMATH_GPT_emily_first_round_points_l868_86871


namespace NUMINAMATH_GPT_sum_first_7_terms_eq_105_l868_86843

variable {a : ℕ → ℤ}

-- Definitions from conditions.
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a)

def a_4_eq_15 : a 4 = 15 := sorry

-- Sum definition specific for 7 terms of an arithmetic sequence.
def sum_first_7_terms (a : ℕ → ℤ) : ℤ := (7 / 2 : ℤ) * (a 1 + a 7)

-- The theorem to prove.
theorem sum_first_7_terms_eq_105 
    (arith_seq : is_arithmetic_sequence a) 
    (a4 : a 4 = 15) : 
  sum_first_7_terms a = 105 := 
sorry

end NUMINAMATH_GPT_sum_first_7_terms_eq_105_l868_86843


namespace NUMINAMATH_GPT_adjusted_ratio_l868_86868

theorem adjusted_ratio :
  (2^2003 * 3^2005) / (6^2004) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_adjusted_ratio_l868_86868


namespace NUMINAMATH_GPT_tiling_not_possible_l868_86861

-- Definitions for the puzzle pieces
inductive Piece
| L | T | I | Z | O

-- Function to check if tiling a rectangle is possible
noncomputable def can_tile_rectangle (pieces : List Piece) : Prop :=
  ∀ (width height : ℕ), width * height % 4 = 0 → ∃ (tiling : List (Piece × ℕ × ℕ)), sorry

theorem tiling_not_possible : ¬ can_tile_rectangle [Piece.L, Piece.T, Piece.I, Piece.Z, Piece.O] :=
sorry

end NUMINAMATH_GPT_tiling_not_possible_l868_86861


namespace NUMINAMATH_GPT_train_crosses_pole_l868_86893

theorem train_crosses_pole
  (speed_kmph : ℝ)
  (train_length_meters : ℝ)
  (conversion_factor : ℝ)
  (speed_mps : ℝ)
  (time_seconds : ℝ)
  (h1 : speed_kmph = 270)
  (h2 : train_length_meters = 375.03)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_mps = speed_kmph * conversion_factor)
  (h5 : time_seconds = train_length_meters / speed_mps)
  : time_seconds = 5.0004 :=
by
  sorry

end NUMINAMATH_GPT_train_crosses_pole_l868_86893


namespace NUMINAMATH_GPT_complete_work_together_in_days_l868_86824

-- Define the work rates for John, Rose, and Michael
def johnWorkRate : ℚ := 1 / 10
def roseWorkRate : ℚ := 1 / 40
def michaelWorkRate : ℚ := 1 / 20

-- Define the combined work rate when they work together
def combinedWorkRate : ℚ := johnWorkRate + roseWorkRate + michaelWorkRate

-- Define the total work to be done
def totalWork : ℚ := 1

-- Calculate the total number of days required to complete the work together
def totalDays : ℚ := totalWork / combinedWorkRate

-- Theorem to prove the total days is 40/7
theorem complete_work_together_in_days : totalDays = 40 / 7 :=
by
  -- Following steps would be the complete proofs if required
  rw [totalDays, totalWork, combinedWorkRate, johnWorkRate, roseWorkRate, michaelWorkRate]
  sorry

end NUMINAMATH_GPT_complete_work_together_in_days_l868_86824


namespace NUMINAMATH_GPT_Sammy_has_8_bottle_caps_l868_86835

-- Definitions representing the conditions
def BilliesBottleCaps := 2
def JaninesBottleCaps := 3 * BilliesBottleCaps
def SammysBottleCaps := JaninesBottleCaps + 2

-- Goal: Prove that Sammy has 8 bottle caps
theorem Sammy_has_8_bottle_caps : 
  SammysBottleCaps = 8 := 
sorry

end NUMINAMATH_GPT_Sammy_has_8_bottle_caps_l868_86835


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l868_86889

variable {α : Type*} [LinearOrderedRing α]

theorem arithmetic_sequence_problem
  (a : ℕ → α)
  (h : ∀ n, a (n + 1) = a n + (a 1 - a 0))
  (h_seq : a 5 + a 6 + a 7 + a 8 + a 9 = 450) :
  a 3 + a 11 = 180 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l868_86889


namespace NUMINAMATH_GPT_f_of_x_plus_1_f_of_2_f_of_x_l868_86840

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_x_plus_1 (x : ℝ) : f (x + 1) = x^2 + 2 * x := sorry

theorem f_of_2 : f 2 = 3 := sorry

theorem f_of_x (x : ℝ) : f x = x^2 - 1 := sorry

end NUMINAMATH_GPT_f_of_x_plus_1_f_of_2_f_of_x_l868_86840


namespace NUMINAMATH_GPT_perpendicular_bisector_eq_l868_86881

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 2)

-- Theorem stating that the perpendicular bisector has the specified equation
theorem perpendicular_bisector_eq : ∀ (x y : ℝ), (y = -2 * x + 3) ↔ ∃ (a b : ℝ), (a, b) = A ∨ (a, b) = B ∧ (y = -2 * x + 3) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_bisector_eq_l868_86881


namespace NUMINAMATH_GPT_greatest_b_solution_l868_86857

def f (b : ℝ) : ℝ := b^2 - 10 * b + 24

theorem greatest_b_solution : ∃ (b : ℝ), (f b ≤ 0) ∧ (∀ (b' : ℝ), (f b' ≤ 0) → b' ≤ b) ∧ b = 6 :=
by
  sorry

end NUMINAMATH_GPT_greatest_b_solution_l868_86857


namespace NUMINAMATH_GPT_A_and_B_worked_together_for_5_days_before_A_left_the_job_l868_86852

noncomputable def workRate_A (W : ℝ) : ℝ := W / 20
noncomputable def workRate_B (W : ℝ) : ℝ := W / 12

noncomputable def combinedWorkRate (W : ℝ) : ℝ := workRate_A W + workRate_B W

noncomputable def workDoneTogether (x : ℝ) (W : ℝ) : ℝ := x * combinedWorkRate W
noncomputable def workDoneBy_B_Alone (W : ℝ) : ℝ := 3 * workRate_B W

theorem A_and_B_worked_together_for_5_days_before_A_left_the_job (W : ℝ) :
  ∃ x : ℝ, workDoneTogether x W + workDoneBy_B_Alone W = W ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_A_and_B_worked_together_for_5_days_before_A_left_the_job_l868_86852


namespace NUMINAMATH_GPT_total_selling_price_l868_86817

theorem total_selling_price
  (CP : ℕ) (Gain : ℕ) (TCP : ℕ)
  (h1 : CP = 1200)
  (h2 : Gain = 3 * CP)
  (h3 : TCP = 18 * CP) :
  ∃ TSP : ℕ, TSP = 25200 := 
by
  sorry

end NUMINAMATH_GPT_total_selling_price_l868_86817


namespace NUMINAMATH_GPT_remainder_zero_by_68_l868_86873

theorem remainder_zero_by_68 (N R1 Q2 : ℕ) (h1 : N = 68 * 269 + R1) (h2 : N % 67 = 1) : R1 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_zero_by_68_l868_86873


namespace NUMINAMATH_GPT_roots_imply_value_l868_86870

noncomputable def value_of_expression (a b c : ℝ) : ℝ :=
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)

theorem roots_imply_value {a b c : ℝ} 
  (h1 : a + b + c = 15) 
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 10) 
  : value_of_expression a b c = 175 / 11 :=
sorry

end NUMINAMATH_GPT_roots_imply_value_l868_86870


namespace NUMINAMATH_GPT_slab_cost_l868_86879

-- Define the conditions
def cubes_per_stick : ℕ := 4
def cubes_per_slab : ℕ := 80
def total_kabob_cost : ℕ := 50
def kabob_sticks_made : ℕ := 40
def total_cubes_needed := kabob_sticks_made * cubes_per_stick
def slabs_needed := total_cubes_needed / cubes_per_slab

-- Final proof problem statement in Lean 4
theorem slab_cost : (total_kabob_cost / slabs_needed) = 25 := by
  sorry

end NUMINAMATH_GPT_slab_cost_l868_86879


namespace NUMINAMATH_GPT_bus_ride_time_l868_86845

def walking_time : ℕ := 15
def waiting_time : ℕ := 2 * walking_time
def train_ride_time : ℕ := 360
def total_trip_time : ℕ := 8 * 60

theorem bus_ride_time : 
  (total_trip_time - (walking_time + waiting_time + train_ride_time)) = 75 := by
  sorry

end NUMINAMATH_GPT_bus_ride_time_l868_86845


namespace NUMINAMATH_GPT_parabola_points_relation_l868_86860

theorem parabola_points_relation (c y1 y2 y3 : ℝ)
  (h1 : y1 = -(-2)^2 - 2*(-2) + c)
  (h2 : y2 = -(0)^2 - 2*(0) + c)
  (h3 : y3 = -(1)^2 - 2*(1) + c) :
  y1 = y2 ∧ y2 > y3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_points_relation_l868_86860


namespace NUMINAMATH_GPT_range_mn_squared_l868_86832

-- Let's define the conditions in Lean

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f is strictly increasing
axiom h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0

-- Condition 2: f(x-1) is centrally symmetric about (1,0)
axiom h2 : ∀ x : ℝ, f (x - 1) = - f (2 - (x - 1))

-- Condition 3: Given inequality
axiom h3 : ∀ m n : ℝ, f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0

-- Prove the range for m^2 + n^2 is (9, 49)
theorem range_mn_squared : ∀ m n : ℝ, f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0 →
  9 < m^2 + n^2 ∧ m^2 + n^2 < 49 :=
sorry

end NUMINAMATH_GPT_range_mn_squared_l868_86832


namespace NUMINAMATH_GPT_find_p_l868_86815

theorem find_p (p: ℝ) (x1 x2: ℝ) (h1: p > 0) (h2: x1^2 + p * x1 + 1 = 0) (h3: x2^2 + p * x2 + 1 = 0) (h4: |x1^2 - x2^2| = p) : p = 5 :=
sorry

end NUMINAMATH_GPT_find_p_l868_86815


namespace NUMINAMATH_GPT_sequence_values_l868_86877

theorem sequence_values (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
    (h_arith : 2 + (a - 2) = a + (b - a)) (h_geom : a * a = b * (9 / b)) : a = 4 ∧ b = 6 :=
by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_sequence_values_l868_86877


namespace NUMINAMATH_GPT_perpendicular_lines_and_slope_l868_86849

theorem perpendicular_lines_and_slope (b : ℝ) : (x + 3 * y + 4 = 0) ∧ (b * x + 3 * y + 6 = 0) → b = -9 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_and_slope_l868_86849


namespace NUMINAMATH_GPT_consecutive_odd_integers_sum_l868_86876

theorem consecutive_odd_integers_sum (x : ℤ) (h : x + (x + 4) = 138) :
  x + (x + 2) + (x + 4) = 207 :=
sorry

end NUMINAMATH_GPT_consecutive_odd_integers_sum_l868_86876


namespace NUMINAMATH_GPT_find_QS_l868_86818

theorem find_QS (RS QR QS : ℕ) (h1 : RS = 13) (h2 : QR = 5) (h3 : QR * 13 = 5 * 13) :
  QS = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_QS_l868_86818


namespace NUMINAMATH_GPT_count_paths_l868_86886

-- Define the lattice points and paths
def isLatticePoint (P : ℤ × ℤ) : Prop := true
def isLatticePath (P : ℕ → ℤ × ℤ) (n : ℕ) : Prop :=
  (∀ i, 0 < i → i ≤ n → abs ((P i).1 - (P (i - 1)).1) + abs ((P i).2 - (P (i - 1)).2) = 1)

-- Define F(n) with the given constraints
def numberOfPaths (n : ℕ) : ℕ :=
  -- Placeholder for the actual complex counting logic, which is not detailed here
  sorry

-- Identify F(n) from the initial conditions and the correct result
theorem count_paths (n : ℕ) :
  numberOfPaths n = Nat.choose (2 * n) n :=
sorry

end NUMINAMATH_GPT_count_paths_l868_86886


namespace NUMINAMATH_GPT_algebra_sum_l868_86880

-- Given conditions
def letterValue (ch : Char) : Int :=
  let pos := ch.toNat - 'a'.toNat + 1
  match pos % 6 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 0 => -2
  | _ => 0  -- This case is actually unreachable.

def wordValue (w : List Char) : Int :=
  w.foldl (fun acc ch => acc + letterValue ch) 0

theorem algebra_sum : wordValue ['a', 'l', 'g', 'e', 'b', 'r', 'a'] = 0 :=
  sorry

end NUMINAMATH_GPT_algebra_sum_l868_86880


namespace NUMINAMATH_GPT_lidia_money_left_l868_86859

theorem lidia_money_left 
  (cost_per_app : ℕ := 4) 
  (num_apps : ℕ := 15) 
  (total_money : ℕ := 66) 
  (discount_rate : ℚ := 0.15) :
  total_money - (num_apps * cost_per_app - (num_apps * cost_per_app * discount_rate)) = 15 := by 
  sorry

end NUMINAMATH_GPT_lidia_money_left_l868_86859


namespace NUMINAMATH_GPT_actual_average_height_calculation_l868_86872

noncomputable def actual_average_height (incorrect_avg_height : ℚ) (number_of_boys : ℕ) (incorrect_recorded_height : ℚ) (actual_height : ℚ) : ℚ :=
  let incorrect_total_height := incorrect_avg_height * number_of_boys
  let overestimated_height := incorrect_recorded_height - actual_height
  let correct_total_height := incorrect_total_height - overestimated_height
  correct_total_height / number_of_boys

theorem actual_average_height_calculation :
  actual_average_height 182 35 166 106 = 180.29 :=
by
  -- The detailed proof is omitted here.
  sorry

end NUMINAMATH_GPT_actual_average_height_calculation_l868_86872


namespace NUMINAMATH_GPT_sum_series_eq_seven_twelve_l868_86884

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * (n:ℝ)^2 + 2 * (n:ℝ) + 1) / ((n:ℝ) * (n + 1) * (n + 2) * (n + 3)) else 0

theorem sum_series_eq_seven_twelve : sum_series = 7 / 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_eq_seven_twelve_l868_86884


namespace NUMINAMATH_GPT_direct_proportion_function_l868_86803

theorem direct_proportion_function (m : ℝ) (h1 : m^2 - 8 = 1) (h2 : m ≠ 3) : m = -3 :=
by
  sorry

end NUMINAMATH_GPT_direct_proportion_function_l868_86803


namespace NUMINAMATH_GPT_certain_event_l868_86814

-- Define the conditions for the problem
def EventA : Prop := ∃ (seat_number : ℕ), seat_number % 2 = 1
def EventB : Prop := ∃ (shooter_hits : Prop), shooter_hits
def EventC : Prop := ∃ (broadcast_news : Prop), broadcast_news
def EventD : Prop := 
  ∀ (red_ball_count white_ball_count : ℕ), (red_ball_count = 2) ∧ (white_ball_count = 1) → 
  ∀ (draw_count : ℕ), (draw_count = 2) → 
  (∃ (red_ball_drawn : Prop), red_ball_drawn)

-- Define the main statement to prove EventD is the certain event
theorem certain_event : EventA → EventB → EventC → EventD
:= 
sorry

end NUMINAMATH_GPT_certain_event_l868_86814


namespace NUMINAMATH_GPT_sale_price_is_91_percent_of_original_price_l868_86833

variable (x : ℝ)
variable (h_increase : ∀ p : ℝ, p * 1.4)
variable (h_sale : ∀ p : ℝ, p * 0.65)

/--The sale price of an item is 91% of the original price.-/
theorem sale_price_is_91_percent_of_original_price {x : ℝ} 
  (h_increase : ∀ p, p * 1.4 = 1.40 * p)
  (h_sale : ∀ p, p * 0.65 = 0.65 * p): 
  (0.65 * 1.40 * x = 0.91 * x) := 
by 
  sorry

end NUMINAMATH_GPT_sale_price_is_91_percent_of_original_price_l868_86833


namespace NUMINAMATH_GPT_solve_for_x_l868_86862

theorem solve_for_x : ∀ x : ℝ, (3 * x + 15 = (1 / 3) * (6 * x + 45)) → x = 0 := by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_for_x_l868_86862


namespace NUMINAMATH_GPT_quadratic_function_symmetry_l868_86801

theorem quadratic_function_symmetry (a b x_1 x_2: ℝ) (h_roots: x_1^2 + a * x_1 + b = 0 ∧ x_2^2 + a * x_2 + b = 0)
(h_symmetry: ∀ x, (x - 2015)^2 + a * (x - 2015) + b = (x + 2015 - 2016)^2 + a * (x + 2015 - 2016) + b):
  (x_1 + x_2) / 2 = 2015 :=
sorry

end NUMINAMATH_GPT_quadratic_function_symmetry_l868_86801


namespace NUMINAMATH_GPT_oldest_sibling_multiple_l868_86851

-- Definitions according to the conditions
def kay_age : Nat := 32
def youngest_sibling_age : Nat := kay_age / 2 - 5
def oldest_sibling_age : Nat := 44

-- The statement to prove
theorem oldest_sibling_multiple : oldest_sibling_age = 4 * youngest_sibling_age :=
by sorry

end NUMINAMATH_GPT_oldest_sibling_multiple_l868_86851


namespace NUMINAMATH_GPT_line_intersects_ellipse_max_chord_length_l868_86809

theorem line_intersects_ellipse (m : ℝ) :
  (-2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2) ↔
  ∃ (x y : ℝ), (9 * x^2 + 6 * m * x + 2 * m^2 - 8 = 0) ∧ (y = (3 / 2) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1) :=
sorry

theorem max_chord_length (m : ℝ) :
  m = 0 → (∃ (A B : ℝ × ℝ),
  ((A.1^2 / 4 + A.2^2 / 9 = 1) ∧ (A.2 = (3 / 2) * A.1 + m)) ∧
  ((B.1^2 / 4 + B.2^2 / 9 = 1) ∧ (B.2 = (3 / 2) * B.1 + m)) ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 26 / 3)) :=
sorry

end NUMINAMATH_GPT_line_intersects_ellipse_max_chord_length_l868_86809


namespace NUMINAMATH_GPT_committee_form_count_l868_86866

def numWaysToFormCommittee (departments : Fin 4 → (ℕ × ℕ)) : ℕ :=
  let waysCase1 := 6 * 81 * 81
  let waysCase2 := 6 * 9 * 9 * 2 * 9 * 9
  waysCase1 + waysCase2

theorem committee_form_count (departments : Fin 4 → (ℕ × ℕ)) 
  (h : ∀ i, departments i = (3, 3)) :
  numWaysToFormCommittee departments = 48114 := 
by
  sorry

end NUMINAMATH_GPT_committee_form_count_l868_86866


namespace NUMINAMATH_GPT_solve_for_3x_plus_9_l868_86826

theorem solve_for_3x_plus_9 :
  ∀ (x : ℝ), (5 * x - 8 = 15 * x + 18) → 3 * (x + 9) = 96 / 5 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_for_3x_plus_9_l868_86826


namespace NUMINAMATH_GPT_minimal_distance_l868_86819

noncomputable def minimum_distance_travel (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) : ℝ :=
  2 * Real.sqrt 19

theorem minimal_distance (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) :
  minimum_distance_travel a b c ha hb hc = 2 * Real.sqrt 19 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_minimal_distance_l868_86819


namespace NUMINAMATH_GPT_steve_first_stack_plastic_cups_l868_86869

theorem steve_first_stack_plastic_cups (cups_n : ℕ -> ℕ)
  (h_prop : ∀ n, cups_n (n + 1) = cups_n n + 4)
  (h_second : cups_n 2 = 21)
  (h_third : cups_n 3 = 25)
  (h_fourth : cups_n 4 = 29) :
  cups_n 1 = 17 :=
sorry

end NUMINAMATH_GPT_steve_first_stack_plastic_cups_l868_86869


namespace NUMINAMATH_GPT_correct_equation_l868_86841

theorem correct_equation :
  (2 * Real.sqrt 2) / (Real.sqrt 2) = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_correct_equation_l868_86841


namespace NUMINAMATH_GPT_find_f4_l868_86847

noncomputable def f : ℝ → ℝ := sorry

theorem find_f4 (hf_odd : ∀ x : ℝ, f (-x) = -f x)
                (hf_property : ∀ x : ℝ, f (x + 2) = -f x) :
  f 4 = 0 :=
sorry

end NUMINAMATH_GPT_find_f4_l868_86847


namespace NUMINAMATH_GPT_equivalent_prop_l868_86805

theorem equivalent_prop (x : ℝ) : (x > 1 → (x - 1) * (x + 3) > 0) ↔ ((x - 1) * (x + 3) ≤ 0 → x ≤ 1) :=
sorry

end NUMINAMATH_GPT_equivalent_prop_l868_86805


namespace NUMINAMATH_GPT_banana_nn_together_count_l868_86899

open Finset

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def arrangements_banana_with_nn_together : ℕ :=
  (factorial 4) / (factorial 3)

theorem banana_nn_together_count : arrangements_banana_with_nn_together = 4 := by
  sorry

end NUMINAMATH_GPT_banana_nn_together_count_l868_86899


namespace NUMINAMATH_GPT_james_profit_l868_86864

/--
  Prove that James's profit from buying 200 lotto tickets at $2 each, given the 
  conditions about winning tickets, is $4,830.
-/
theorem james_profit 
  (total_tickets : ℕ := 200)
  (cost_per_ticket : ℕ := 2)
  (winner_percentage : ℝ := 0.2)
  (five_dollar_win_pct : ℝ := 0.8)
  (grand_prize : ℝ := 5000)
  (average_other_wins : ℝ := 10) :
  let total_cost := total_tickets * cost_per_ticket 
  let total_winners := winner_percentage * total_tickets
  let five_dollar_winners := five_dollar_win_pct * total_winners
  let total_five_dollar := five_dollar_winners * 5
  let remaining_winners := total_winners - 1 - five_dollar_winners
  let total_remaining_winners := remaining_winners * average_other_wins
  let total_winnings := total_five_dollar + grand_prize + total_remaining_winners
  let profit := total_winnings - total_cost
  profit = 4830 :=
by
  sorry

end NUMINAMATH_GPT_james_profit_l868_86864


namespace NUMINAMATH_GPT_train_avg_speed_without_stoppages_l868_86804

/-- A train with stoppages has an average speed of 125 km/h. Given that the train stops for 30 minutes per hour,
the average speed of the train without stoppages is 250 km/h. -/
theorem train_avg_speed_without_stoppages (avg_speed_with_stoppages : ℝ) 
  (stoppage_time_per_hour : ℝ) (no_stoppage_speed : ℝ) 
  (h1 : avg_speed_with_stoppages = 125) (h2 : stoppage_time_per_hour = 0.5) : 
  no_stoppage_speed = 250 :=
sorry

end NUMINAMATH_GPT_train_avg_speed_without_stoppages_l868_86804


namespace NUMINAMATH_GPT_kopeechka_items_l868_86898

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end NUMINAMATH_GPT_kopeechka_items_l868_86898


namespace NUMINAMATH_GPT_original_price_of_boots_l868_86896

theorem original_price_of_boots (P : ℝ) (h : P * 0.80 = 72) : P = 90 :=
by 
  sorry

end NUMINAMATH_GPT_original_price_of_boots_l868_86896


namespace NUMINAMATH_GPT_correct_statements_l868_86811

-- Define the regression condition
def regression_condition (b : ℝ) : Prop := b < 0

-- Conditon ③: Event A is the complement of event B implies mutual exclusivity
def mutually_exclusive_and_complementary (A B : Prop) : Prop := 
  (A → ¬B) → (¬A ↔ B)

-- Main theorem combining the conditions and questions
theorem correct_statements: 
  (∀ b, regression_condition b ↔ (b < 0)) ∧
  (∀ A B, mutually_exclusive_and_complementary A B → (¬A ≠ B)) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l868_86811


namespace NUMINAMATH_GPT_natural_numbers_not_divisible_by_5_or_7_l868_86802

def num_not_divisible_by_5_or_7 (n : ℕ) : ℕ :=
  let num_div_5 := n / 5
  let num_div_7 := n / 7
  let num_div_35 := n / 35
  n - (num_div_5 + num_div_7 - num_div_35)

theorem natural_numbers_not_divisible_by_5_or_7 :
  num_not_divisible_by_5_or_7 999 = 686 :=
by sorry

end NUMINAMATH_GPT_natural_numbers_not_divisible_by_5_or_7_l868_86802


namespace NUMINAMATH_GPT_find_largest_number_l868_86863

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
  sorry

end NUMINAMATH_GPT_find_largest_number_l868_86863


namespace NUMINAMATH_GPT_width_of_field_l868_86820

noncomputable def field_width : ℝ := 60

theorem width_of_field (L W : ℝ) (hL : L = (7/5) * W) (hP : 288 = 2 * L + 2 * W) : W = field_width :=
by
  sorry

end NUMINAMATH_GPT_width_of_field_l868_86820


namespace NUMINAMATH_GPT_Jill_ball_difference_l868_86837

theorem Jill_ball_difference (r_packs y_packs balls_per_pack : ℕ)
  (h_r_packs : r_packs = 5) 
  (h_y_packs : y_packs = 4) 
  (h_balls_per_pack : balls_per_pack = 18) :
  (r_packs * balls_per_pack) - (y_packs * balls_per_pack) = 18 :=
by
  sorry

end NUMINAMATH_GPT_Jill_ball_difference_l868_86837
