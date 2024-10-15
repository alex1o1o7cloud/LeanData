import Mathlib

namespace NUMINAMATH_GPT_uncle_money_given_l481_48196

-- Definitions
def lizzy_mother_money : Int := 80
def lizzy_father_money : Int := 40
def candy_expense : Int := 50
def total_money_now : Int := 140

-- Theorem to prove
theorem uncle_money_given : (total_money_now - ((lizzy_mother_money + lizzy_father_money) - candy_expense)) = 70 := 
  by
    sorry

end NUMINAMATH_GPT_uncle_money_given_l481_48196


namespace NUMINAMATH_GPT_cost_of_one_shirt_l481_48179

theorem cost_of_one_shirt
  (cost_J : ℕ)  -- The cost of one pair of jeans
  (cost_S : ℕ)  -- The cost of one shirt
  (h1 : 3 * cost_J + 2 * cost_S = 69)
  (h2 : 2 * cost_J + 3 * cost_S = 81) :
  cost_S = 21 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_shirt_l481_48179


namespace NUMINAMATH_GPT_alice_number_l481_48152

theorem alice_number (n : ℕ) 
  (h1 : 243 ∣ n) 
  (h2 : 36 ∣ n) 
  (h3 : 1000 < n) 
  (h4 : n < 3000) : 
  n = 1944 ∨ n = 2916 := 
sorry

end NUMINAMATH_GPT_alice_number_l481_48152


namespace NUMINAMATH_GPT_log_equality_implies_exp_equality_l481_48106

theorem log_equality_implies_exp_equality (x y z a : ℝ) (h : (x * (y + z - x)) / (Real.log x) = (y * (x + z - y)) / (Real.log y) ∧ (y * (x + z - y)) / (Real.log y) = (z * (x + y - z)) / (Real.log z)) :
  x^y * y^x = z^x * x^z ∧ z^x * x^z = y^z * z^y :=
by
  sorry

end NUMINAMATH_GPT_log_equality_implies_exp_equality_l481_48106


namespace NUMINAMATH_GPT_greatest_sum_consecutive_integers_l481_48168

theorem greatest_sum_consecutive_integers (n : ℤ) (h : n * (n + 1) < 360) : n + (n + 1) ≤ 37 := by
  sorry

end NUMINAMATH_GPT_greatest_sum_consecutive_integers_l481_48168


namespace NUMINAMATH_GPT_parker_shorter_than_daisy_l481_48102

noncomputable def solve_height_difference : Nat :=
  let R := 60
  let D := R + 8
  let avg := 64
  ((3 * avg) - (D + R))

theorem parker_shorter_than_daisy :
  let P := solve_height_difference
  D - P = 4 := by
  sorry

end NUMINAMATH_GPT_parker_shorter_than_daisy_l481_48102


namespace NUMINAMATH_GPT_simplify_polynomial_l481_48139

theorem simplify_polynomial :
  (3 * x ^ 4 - 2 * x ^ 3 + 5 * x ^ 2 - 8 * x + 10) + (7 * x ^ 5 - 3 * x ^ 4 + x ^ 3 - 7 * x ^ 2 + 2 * x - 2)
  = 7 * x ^ 5 - x ^ 3 - 2 * x ^ 2 - 6 * x + 8 :=
by sorry

end NUMINAMATH_GPT_simplify_polynomial_l481_48139


namespace NUMINAMATH_GPT_max_groups_eq_one_l481_48174

-- Defining the conditions 
def eggs : ℕ := 16
def marbles : ℕ := 3
def rubber_bands : ℕ := 5

-- The theorem statement
theorem max_groups_eq_one
  (h1 : eggs = 16)
  (h2 : marbles = 3)
  (h3 : rubber_bands = 5) :
  ∀ g : ℕ, (g ≤ eggs ∧ g ≤ marbles ∧ g ≤ rubber_bands) →
  (eggs % g = 0) ∧ (marbles % g = 0) ∧ (rubber_bands % g = 0) →
  g = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_groups_eq_one_l481_48174


namespace NUMINAMATH_GPT_find_eccentricity_l481_48113

-- Define the hyperbola structure
structure Hyperbola where
  a : ℝ
  b : ℝ
  (a_pos : 0 < a)
  (b_pos : 0 < b)

-- Define the point P and focus F₁ F₂ relationship
structure PointsRelation (C : Hyperbola) where
  P : ℝ × ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  (distance_condition : dist P F1 = 3 * dist P F2)
  (dot_product_condition : (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = C.a^2)

noncomputable def eccentricity (C : Hyperbola) (rel : PointsRelation C) : ℝ :=
  Real.sqrt (1 + (C.b ^ 2) / (C.a ^ 2))

theorem find_eccentricity (C : Hyperbola) (rel : PointsRelation C) : eccentricity C rel = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_find_eccentricity_l481_48113


namespace NUMINAMATH_GPT_sum_of_first_3m_terms_l481_48157

variable {a : ℕ → ℝ}   -- The arithmetic sequence
variable {S : ℕ → ℝ}   -- The sum of the first n terms of the sequence

def arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : Prop :=
  S m = 30 ∧ S (2 * m) = 100 ∧ S (3 * m) = 170

theorem sum_of_first_3m_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence_sum a S m :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_3m_terms_l481_48157


namespace NUMINAMATH_GPT_max_marks_l481_48120

theorem max_marks (M : ℝ) (h1 : 0.25 * M = 185 + 25) : M = 840 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l481_48120


namespace NUMINAMATH_GPT_exponent_equivalence_l481_48142

theorem exponent_equivalence (a b : ℕ) (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) (h1 : 9 ^ m = a) (h2 : 3 ^ n = b) : 
  3 ^ (2 * m + 4 * n) = a * b ^ 4 := 
by 
  sorry

end NUMINAMATH_GPT_exponent_equivalence_l481_48142


namespace NUMINAMATH_GPT_square_pizza_area_larger_by_27_percent_l481_48154

theorem square_pizza_area_larger_by_27_percent :
  let r := 5
  let A_circle := Real.pi * r^2
  let s := 2 * r
  let A_square := s^2
  let delta_A := A_square - A_circle
  let percent_increase := (delta_A / A_circle) * 100
  Int.floor (percent_increase + 0.5) = 27 :=
by
  sorry

end NUMINAMATH_GPT_square_pizza_area_larger_by_27_percent_l481_48154


namespace NUMINAMATH_GPT_smallest_number_l481_48121

theorem smallest_number (a b c d : ℤ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -1) (h4 : d = -3) :
  d = -3 ∧ d < c ∧ d < b ∧ d < a :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_l481_48121


namespace NUMINAMATH_GPT_solve_quadratic_equation_l481_48116

theorem solve_quadratic_equation :
  ∀ x : ℝ, (10 - x) ^ 2 = 2 * x ^ 2 + 4 * x ↔ x = 3.62 ∨ x = -27.62 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l481_48116


namespace NUMINAMATH_GPT_woman_alone_days_l481_48140

theorem woman_alone_days (M W : ℝ) (h1 : (10 * M + 15 * W) * 5 = 1) (h2 : M * 100 = 1) : W * 150 = 1 :=
by
  sorry

end NUMINAMATH_GPT_woman_alone_days_l481_48140


namespace NUMINAMATH_GPT_mo_hot_chocolate_l481_48131

noncomputable def cups_of_hot_chocolate (total_drinks: ℕ) (extra_tea: ℕ) (non_rainy_days: ℕ) (tea_per_day: ℕ) : ℕ :=
  let tea_drinks := non_rainy_days * tea_per_day 
  let chocolate_drinks := total_drinks - tea_drinks 
  (extra_tea - chocolate_drinks)

theorem mo_hot_chocolate :
  cups_of_hot_chocolate 36 14 5 5 = 11 :=
by
  sorry

end NUMINAMATH_GPT_mo_hot_chocolate_l481_48131


namespace NUMINAMATH_GPT_three_consecutive_multiples_sum_l481_48128

theorem three_consecutive_multiples_sum (h1 : Int) (h2 : h1 % 3 = 0) (h3 : Int) (h4 : h3 = h1 - 3) (h5 : Int) (h6 : h5 = h1 - 6) (h7: h1 = 27) : h1 + h3 + h5 = 72 := 
by 
  -- let numbers be n, n-3, n-6 and n = 27
  -- so n + n-3 + n-6 = 27 + 24 + 21 = 72
  sorry

end NUMINAMATH_GPT_three_consecutive_multiples_sum_l481_48128


namespace NUMINAMATH_GPT_m_perpendicular_beta_l481_48176

variables {Plane : Type*} {Line : Type*}

-- Definitions of the perpendicularity and parallelism
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Given variables
variables (α β : Plane) (m : Line)

-- Conditions
axiom M_perpendicular_Alpha : perpendicular m α
axiom Alpha_parallel_Beta : parallel α β

-- Proof goal
theorem m_perpendicular_beta 
  (h1 : perpendicular m α) 
  (h2 : parallel α β) : 
  perpendicular m β := 
  sorry

end NUMINAMATH_GPT_m_perpendicular_beta_l481_48176


namespace NUMINAMATH_GPT_tan_sum_pi_over_12_l481_48141

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end NUMINAMATH_GPT_tan_sum_pi_over_12_l481_48141


namespace NUMINAMATH_GPT_Todd_ate_5_cupcakes_l481_48171

theorem Todd_ate_5_cupcakes (original_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) (remaining_cupcakes : ℕ) :
  original_cupcakes = 50 ∧ packages = 9 ∧ cupcakes_per_package = 5 ∧ remaining_cupcakes = packages * cupcakes_per_package →
  original_cupcakes - remaining_cupcakes = 5 :=
by
  sorry

end NUMINAMATH_GPT_Todd_ate_5_cupcakes_l481_48171


namespace NUMINAMATH_GPT_cricket_team_players_l481_48148

-- Define conditions 
def non_throwers (T P : ℕ) : ℕ := P - T
def left_handers (N : ℕ) : ℕ := N / 3
def right_handers_non_thrower (N : ℕ) : ℕ := 2 * N / 3
def total_right_handers (T R : ℕ) : Prop := R = T + right_handers_non_thrower (non_throwers T R)

-- Assume conditions are given
variables (P N R T : ℕ)
axiom hT : T = 37
axiom hR : R = 49
axiom hNonThrower : N = non_throwers T P
axiom hRightHanders : right_handers_non_thrower N = R - T

-- Prove the total number of players is 55
theorem cricket_team_players : P = 55 :=
by
  sorry

end NUMINAMATH_GPT_cricket_team_players_l481_48148


namespace NUMINAMATH_GPT_exists_a_perfect_power_l481_48138

def is_perfect_power (n : ℕ) : Prop :=
  ∃ b k : ℕ, b > 0 ∧ k ≥ 2 ∧ n = b^k

theorem exists_a_perfect_power :
  ∃ a > 0, ∀ n, 2015 ≤ n ∧ n ≤ 2558 → is_perfect_power (n * a) :=
sorry

end NUMINAMATH_GPT_exists_a_perfect_power_l481_48138


namespace NUMINAMATH_GPT_map_length_25_cm_represents_125_km_l481_48182

-- Define the conditions
def map_scale (cm: ℝ) : ℝ := 5 * cm

-- Define the main statement to be proved
theorem map_length_25_cm_represents_125_km : map_scale 25 = 125 := by
  sorry

end NUMINAMATH_GPT_map_length_25_cm_represents_125_km_l481_48182


namespace NUMINAMATH_GPT_percent_equivalence_l481_48162

theorem percent_equivalence (x : ℝ) : (0.6 * 0.3 * x - 0.1 * x) / x * 100 = 8 := by
  sorry

end NUMINAMATH_GPT_percent_equivalence_l481_48162


namespace NUMINAMATH_GPT_problem_solution_l481_48149

-- Given non-zero numbers x and y such that x = 1 / y,
-- prove that (2x - 1/x) * (y - 1/y) = -2x^2 + y^2.
theorem problem_solution (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x = 1 / y) :
  (2 * x - 1 / x) * (y - 1 / y) = -2 * x^2 + y^2 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l481_48149


namespace NUMINAMATH_GPT_minimum_prism_volume_l481_48156

theorem minimum_prism_volume (l m n : ℕ) (h1 : l > 0) (h2 : m > 0) (h3 : n > 0)
    (hidden_volume_condition : (l - 1) * (m - 1) * (n - 1) = 420) :
    ∃ N : ℕ, N = l * m * n ∧ N = 630 := by
  sorry

end NUMINAMATH_GPT_minimum_prism_volume_l481_48156


namespace NUMINAMATH_GPT_find_S_l481_48186

theorem find_S :
  (1/4 : ℝ) * (1/6 : ℝ) * S = (1/5 : ℝ) * (1/8 : ℝ) * 160 → S = 96 :=
by
  intro h
  -- Proof is omitted
  sorry 

end NUMINAMATH_GPT_find_S_l481_48186


namespace NUMINAMATH_GPT_ratio_of_integers_l481_48191

theorem ratio_of_integers (a b : ℤ) (h : 1996 * a + b / 96 = a + b) : a / b = 1 / 2016 ∨ b / a = 2016 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_integers_l481_48191


namespace NUMINAMATH_GPT_find_radius_of_tangent_circle_l481_48163

def tangent_circle_radius : Prop :=
  ∃ (r : ℝ), 
    (r > 0) ∧ 
    (∀ (θ : ℝ),
      (∃ (x y : ℝ),
        x = 1 + r * Real.cos θ ∧ 
        y = 1 + r * Real.sin θ ∧ 
        x + y - 1 = 0))
    → r = (Real.sqrt 2) / 2

theorem find_radius_of_tangent_circle : tangent_circle_radius :=
sorry

end NUMINAMATH_GPT_find_radius_of_tangent_circle_l481_48163


namespace NUMINAMATH_GPT_largest_possible_p_l481_48180

theorem largest_possible_p (m n p : ℕ) (h1 : m > 2) (h2 : n > 2) (h3 : p > 2) (h4 : gcd m n = 1) (h5 : gcd n p = 1) (h6 : gcd m p = 1)
  (h7 : (1/m : ℚ) + (1/n : ℚ) + (1/p : ℚ) = 1/2) : p ≤ 42 :=
by sorry

end NUMINAMATH_GPT_largest_possible_p_l481_48180


namespace NUMINAMATH_GPT_solve_for_x_l481_48161

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 = -2 * x + 10) : x = 3 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l481_48161


namespace NUMINAMATH_GPT_adults_not_wearing_blue_is_10_l481_48188

section JohnsonFamilyReunion

-- Define the number of children
def children : ℕ := 45

-- Define the ratio between adults and children
def adults : ℕ := children / 3

-- Define the ratio of adults who wore blue
def adults_wearing_blue : ℕ := adults / 3

-- Define the number of adults who did not wear blue
def adults_not_wearing_blue : ℕ := adults - adults_wearing_blue

-- Theorem stating the number of adults who did not wear blue
theorem adults_not_wearing_blue_is_10 : adults_not_wearing_blue = 10 :=
by
  -- This is a placeholder for the actual proof
  sorry

end JohnsonFamilyReunion

end NUMINAMATH_GPT_adults_not_wearing_blue_is_10_l481_48188


namespace NUMINAMATH_GPT_neither_happy_nor_sad_boys_is_5_l481_48175

-- Define the total number of children
def total_children := 60

-- Define the number of happy children
def happy_children := 30

-- Define the number of sad children
def sad_children := 10

-- Define the number of neither happy nor sad children
def neither_happy_nor_sad_children := 20

-- Define the number of boys
def boys := 17

-- Define the number of girls
def girls := 43

-- Define the number of happy boys
def happy_boys := 6

-- Define the number of sad girls
def sad_girls := 4

-- Define the number of neither happy nor sad boys
def neither_happy_nor_sad_boys := boys - (happy_boys + (sad_children - sad_girls))

theorem neither_happy_nor_sad_boys_is_5 :
  neither_happy_nor_sad_boys = 5 :=
by
  -- This skips the proof
  sorry

end NUMINAMATH_GPT_neither_happy_nor_sad_boys_is_5_l481_48175


namespace NUMINAMATH_GPT_scientific_notation_suzhou_blood_donors_l481_48144

theorem scientific_notation_suzhou_blood_donors : ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 124000 = a * 10^n ∧ a = 1.24 ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_suzhou_blood_donors_l481_48144


namespace NUMINAMATH_GPT_cuboid_height_l481_48199

/-- Given a cuboid with surface area 2400 cm², length 15 cm, and breadth 10 cm,
    prove that the height is 42 cm. -/
theorem cuboid_height (SA l w : ℝ) (h : ℝ) : 
  SA = 2400 → l = 15 → w = 10 → 2 * (l * w + l * h + w * h) = SA → h = 42 :=
by
  intros hSA hl hw hformula
  sorry

end NUMINAMATH_GPT_cuboid_height_l481_48199


namespace NUMINAMATH_GPT_length_of_goods_train_l481_48177

theorem length_of_goods_train 
  (speed_kmh : ℕ) 
  (platform_length_m : ℕ) 
  (cross_time_s : ℕ) :
  speed_kmh = 72 → platform_length_m = 280 → cross_time_s = 26 → 
  ∃ train_length_m : ℕ, train_length_m = 240 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_length_of_goods_train_l481_48177


namespace NUMINAMATH_GPT_lucy_initial_balance_l481_48160

theorem lucy_initial_balance (final_balance deposit withdrawal : Int) 
  (h_final : final_balance = 76)
  (h_deposit : deposit = 15)
  (h_withdrawal : withdrawal = 4) :
  let initial_balance := final_balance + withdrawal - deposit
  initial_balance = 65 := 
by
  sorry

end NUMINAMATH_GPT_lucy_initial_balance_l481_48160


namespace NUMINAMATH_GPT_proportion_equation_l481_48153

theorem proportion_equation (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_proportion_equation_l481_48153


namespace NUMINAMATH_GPT_hyperbola_equation_l481_48194

-- Define the conditions
def hyperbola_eq := ∀ (x y a b : ℝ), a > 0 ∧ b > 0 → x^2 / a^2 - y^2 / b^2 = 1
def parabola_eq := ∀ (x y : ℝ), y^2 = (2 / 5) * x
def intersection_point_M := ∃ (x : ℝ), ∀ (y : ℝ), y = 1 → y^2 = (2 / 5) * x
def line_intersect_N := ∀ (F₁ M N : ℝ × ℝ), 
  (N.1 = -1 / 10) ∧ (F₁.1 ≠ M.1) ∧ (N.2 = 0)

-- State the proof problem
theorem hyperbola_equation 
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (hyp_eq : hyperbola_eq)
  (par_eq : parabola_eq)
  (int_pt_M : intersection_point_M)
  (line_int_N : line_intersect_N) :
  ∀ (x y : ℝ), x^2 / 5 - y^2 / 4 = 1 :=
by sorry

end NUMINAMATH_GPT_hyperbola_equation_l481_48194


namespace NUMINAMATH_GPT_shifted_function_is_correct_l481_48187

-- Given conditions
def original_function (x : ℝ) : ℝ := -(x + 2) ^ 2 + 1

def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Resulting function after shifting 1 unit to the right
def shifted_function : ℝ → ℝ := shift_right original_function 1

-- Correct answer
def correct_function (x : ℝ) : ℝ := -(x + 1) ^ 2 + 1

-- Proof Statement
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = correct_function x := by
  sorry

end NUMINAMATH_GPT_shifted_function_is_correct_l481_48187


namespace NUMINAMATH_GPT_no_rain_five_days_l481_48193

-- Define the problem conditions and the required result.
def prob_rain := (2 / 3)
def prob_no_rain := (1 - prob_rain)
def prob_no_rain_five_days := prob_no_rain^5

theorem no_rain_five_days : 
  prob_no_rain_five_days = (1 / 243) :=
by
  sorry

end NUMINAMATH_GPT_no_rain_five_days_l481_48193


namespace NUMINAMATH_GPT_responses_needed_l481_48129

noncomputable def Q : ℝ := 461.54
noncomputable def percentage : ℝ := 0.65
noncomputable def required_responses : ℝ := percentage * Q

theorem responses_needed : required_responses = 300 := by
  sorry

end NUMINAMATH_GPT_responses_needed_l481_48129


namespace NUMINAMATH_GPT_area_calculation_l481_48147

variable (x : ℝ)

def area_large_rectangle : ℝ := (2 * x + 9) * (x + 6)
def area_rectangular_hole : ℝ := (x - 1) * (2 * x - 5)
def area_square : ℝ := (x + 3) ^ 2
def area_remaining : ℝ := area_large_rectangle x - area_rectangular_hole x - area_square x

theorem area_calculation : area_remaining x = -x^2 + 22 * x + 40 := by
  sorry

end NUMINAMATH_GPT_area_calculation_l481_48147


namespace NUMINAMATH_GPT_find_ethanol_percentage_l481_48155

noncomputable def ethanol_percentage_in_fuel_A (P_A : ℝ) (V_A : ℝ) : Prop :=
  (P_A / 100) * V_A + 0.16 * (200 - V_A) = 18

theorem find_ethanol_percentage (P_A : ℝ) (V_A : ℝ) (h₀ : V_A ≤ 200) (h₁ : 0 ≤ V_A) :
  ethanol_percentage_in_fuel_A P_A V_A :=
by
  sorry

end NUMINAMATH_GPT_find_ethanol_percentage_l481_48155


namespace NUMINAMATH_GPT_part1_max_area_part2_find_a_l481_48119

-- Part (1): Define the function and prove maximum area of the triangle
noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.exp x - 3 * a * x + 2 * Real.sin x - 1

theorem part1_max_area (a : ℝ) (h : 0 < a ∧ a < 1) : 
  let f' := a^2 - 3 * a + 2
  ∃ h_a_max, h_a_max == 3 / 8 :=
  sorry

-- Part (2): Prove that the function reaches an extremum at x = 0 and determine the value of a.
theorem part2_find_a (a : ℝ) : (a^2 - 3 * a + 2 = 0) → (a = 1 ∨ a = 2) :=
  sorry

end NUMINAMATH_GPT_part1_max_area_part2_find_a_l481_48119


namespace NUMINAMATH_GPT_problem_l481_48165

theorem problem (a b : ℕ)
  (ha : a = 2) 
  (hb : b = 121) 
  (h_minPrime : ∀ n, n < a → ¬ (∀ d, d ∣ n → d = 1 ∨ d = n))
  (h_threeDivisors : ∀ n, n < 150 → ∀ d, d ∣ n → d = 1 ∨ d = n → n = 121) :
  a + b = 123 := by
  sorry

end NUMINAMATH_GPT_problem_l481_48165


namespace NUMINAMATH_GPT_angle_A_is_30_degrees_l481_48115

theorem angle_A_is_30_degrees
    (a b : ℝ)
    (B A : ℝ)
    (a_eq_4 : a = 4)
    (b_eq_4_sqrt2 : b = 4 * Real.sqrt 2)
    (B_eq_45 : B = Real.pi / 4) : 
    A = Real.pi / 6 := 
by 
    sorry

end NUMINAMATH_GPT_angle_A_is_30_degrees_l481_48115


namespace NUMINAMATH_GPT_angle_C_is_30_degrees_l481_48122

theorem angle_C_is_30_degrees
  (A B C : ℝ)
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1)
  (A_rad: 0 ≤ A ∧ A ≤ Real.pi)
  (B_rad: 0 ≤ B ∧ B ≤ Real.pi)
  (C_rad : 0 ≤ C ∧ C ≤ Real.pi)
  (triangle_condition: A + B + C = Real.pi) :
  C = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_angle_C_is_30_degrees_l481_48122


namespace NUMINAMATH_GPT_m_divides_product_iff_composite_ne_4_l481_48114

theorem m_divides_product_iff_composite_ne_4 (m : ℕ) : 
  (m ∣ Nat.factorial (m - 1)) ↔ 
  (∃ a b : ℕ, a ≠ b ∧ 1 < a ∧ 1 < b ∧ m = a * b ∧ m ≠ 4) := 
sorry

end NUMINAMATH_GPT_m_divides_product_iff_composite_ne_4_l481_48114


namespace NUMINAMATH_GPT_total_movies_in_series_l481_48159

def book_count := 4
def total_books_read := 19
def movies_watched := 7
def movies_to_watch := 10

theorem total_movies_in_series : movies_watched + movies_to_watch = 17 := by
  sorry

end NUMINAMATH_GPT_total_movies_in_series_l481_48159


namespace NUMINAMATH_GPT_total_weight_of_containers_l481_48197

theorem total_weight_of_containers (x y z : ℕ) :
  x + y = 162 →
  y + z = 168 →
  z + x = 174 →
  x + y + z = 252 :=
by
  intros hxy hyz hzx
  -- proof skipped
  sorry

end NUMINAMATH_GPT_total_weight_of_containers_l481_48197


namespace NUMINAMATH_GPT_unique_solution_l481_48103

def is_prime (p : ℕ) : Prop := ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem unique_solution (n : ℕ) :
  (0 < n ∧ is_prime (n + 1) ∧ is_prime (n + 3) ∧
   is_prime (n + 7) ∧ is_prime (n + 9) ∧
   is_prime (n + 13) ∧ is_prime (n + 15)) ↔ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l481_48103


namespace NUMINAMATH_GPT_value_of_f_m_plus_one_l481_48166

variable (a m : ℝ)

def f (x : ℝ) : ℝ := x^2 - x + a

theorem value_of_f_m_plus_one 
  (h : f a (-m) < 0) : f a (m + 1) < 0 := by
  sorry

end NUMINAMATH_GPT_value_of_f_m_plus_one_l481_48166


namespace NUMINAMATH_GPT_moon_speed_conversion_correct_l481_48136

-- Define the conversions
def kilometers_per_second_to_miles_per_hour (kmps : ℝ) : ℝ :=
  kmps * 0.621371 * 3600

-- Condition: The moon's speed
def moon_speed_kmps : ℝ := 1.02

-- Correct answer in miles per hour
def expected_moon_speed_mph : ℝ := 2281.34

-- Theorem stating the equivalence of converted speed to expected speed
theorem moon_speed_conversion_correct :
  kilometers_per_second_to_miles_per_hour moon_speed_kmps = expected_moon_speed_mph :=
by 
  sorry

end NUMINAMATH_GPT_moon_speed_conversion_correct_l481_48136


namespace NUMINAMATH_GPT_inequality_solution_l481_48100

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ (x < 1 ∨ x > 3) ∧ (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l481_48100


namespace NUMINAMATH_GPT_triangle_length_l481_48117

theorem triangle_length (DE DF : ℝ) (Median_to_EF : ℝ) (EF : ℝ) :
  DE = 2 ∧ DF = 3 ∧ Median_to_EF = EF → EF = (13:ℝ).sqrt / 5 := by
  sorry

end NUMINAMATH_GPT_triangle_length_l481_48117


namespace NUMINAMATH_GPT_find_matrix_M_l481_48127

theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) (h : M^3 - 3 • M^2 + 4 • M = ![![6, 12], ![3, 6]]) :
  M = ![![2, 4], ![1, 2]] :=
sorry

end NUMINAMATH_GPT_find_matrix_M_l481_48127


namespace NUMINAMATH_GPT_initial_tabs_count_l481_48124

theorem initial_tabs_count (T : ℕ) (h1 : T > 0)
  (h2 : (3 / 4 : ℚ) * T - (2 / 5 : ℚ) * ((3 / 4 : ℚ) * T) > 0)
  (h3 : (9 / 20 : ℚ) * T - (1 / 2 : ℚ) * ((9 / 20 : ℚ) * T) = 90) :
  T = 400 :=
sorry

end NUMINAMATH_GPT_initial_tabs_count_l481_48124


namespace NUMINAMATH_GPT_simplify_expression_l481_48135

variables {x y : ℝ}
-- Ensure that x and y are not zero to avoid division by zero errors.
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y :=
sorry

end NUMINAMATH_GPT_simplify_expression_l481_48135


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l481_48108

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x - 2 > 0) ∧ (3 * (x - 1) - 7 < -2 * x) → 1 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l481_48108


namespace NUMINAMATH_GPT_remaining_volume_after_pours_l481_48181

-- Definitions based on the problem conditions
def initial_volume_liters : ℝ := 2
def initial_volume_milliliters : ℝ := initial_volume_liters * 1000
def pour_amount (x : ℝ) : ℝ := x

-- Statement of the problem as a theorem in Lean 4
theorem remaining_volume_after_pours (x : ℝ) : 
  ∃ remaining_volume : ℝ, remaining_volume = initial_volume_milliliters - 4 * pour_amount x :=
by
  -- To be filled with the proof
  sorry

end NUMINAMATH_GPT_remaining_volume_after_pours_l481_48181


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l481_48134

theorem quadratic_inequality_solution_set {x : ℝ} : 
  x^2 < x + 6 ↔ (-2 < x ∧ x < 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l481_48134


namespace NUMINAMATH_GPT_sum_of_integers_with_largest_proper_divisor_55_l481_48190

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def largest_proper_divisor (n d : ℕ) : Prop :=
  (d ∣ n) ∧ (d < n) ∧ ∀ e, (e ∣ n ∧ e < n ∧ e > d) → False

theorem sum_of_integers_with_largest_proper_divisor_55 : 
  (∀ n : ℕ, largest_proper_divisor n 55 → n = 110 ∨ n = 165 ∨ n = 275) →
  110 + 165 + 275 = 550 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_with_largest_proper_divisor_55_l481_48190


namespace NUMINAMATH_GPT_printing_shop_paper_boxes_l481_48101

variable (x y : ℕ) -- Assuming x and y are natural numbers since the number of boxes can't be negative.

theorem printing_shop_paper_boxes (h1 : 80 * x + 180 * y = 2660)
                                  (h2 : x = 5 * y - 3) :
    x = 22 ∧ y = 5 := sorry

end NUMINAMATH_GPT_printing_shop_paper_boxes_l481_48101


namespace NUMINAMATH_GPT_avg_difference_is_5_l481_48105

def avg (s : List ℕ) : ℕ :=
  s.sum / s.length

def set1 := [20, 40, 60]
def set2 := [20, 60, 25]

theorem avg_difference_is_5 :
  avg set1 - avg set2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_avg_difference_is_5_l481_48105


namespace NUMINAMATH_GPT_infinite_set_P_l481_48192

-- Define the condition as given in the problem
def has_property_P (P : Set ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → (∀ p : ℕ, p.Prime → p ∣ k^3 + 6 → p ∈ P)

-- State the proof problem
theorem infinite_set_P (P : Set ℕ) (h : has_property_P P) : ∃ p : ℕ, p ∉ P → false :=
by
  -- The statement asserts that the set P described by has_property_P is infinite.
  sorry

end NUMINAMATH_GPT_infinite_set_P_l481_48192


namespace NUMINAMATH_GPT_largest_number_in_L_shape_l481_48170

theorem largest_number_in_L_shape (x : ℤ) (sum : ℤ) (h : sum = 2015) : x = 676 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_in_L_shape_l481_48170


namespace NUMINAMATH_GPT_prime_square_sum_l481_48126

theorem prime_square_sum (p q m : ℕ) (hp : Prime p) (hq : Prime q) (hne : p ≠ q)
  (hp_eq : p^2 - 2001 * p + m = 0) (hq_eq : q^2 - 2001 * q + m = 0) :
  p^2 + q^2 = 3996005 :=
sorry

end NUMINAMATH_GPT_prime_square_sum_l481_48126


namespace NUMINAMATH_GPT_victor_draw_order_count_l481_48167

-- Definitions based on the problem conditions
def num_piles : ℕ := 3
def num_cards_per_pile : ℕ := 3
def total_cards : ℕ := num_piles * num_cards_per_pile

-- The cardinality of the set of valid sequences where within each pile cards must be drawn in order
def valid_sequences_count : ℕ :=
  Nat.factorial total_cards / (Nat.factorial num_cards_per_pile ^ num_piles)

-- Now we state the problem: proving the valid sequences count is 1680
theorem victor_draw_order_count :
  valid_sequences_count = 1680 :=
by
  sorry

end NUMINAMATH_GPT_victor_draw_order_count_l481_48167


namespace NUMINAMATH_GPT_equal_distribution_of_drawings_l481_48195

theorem equal_distribution_of_drawings (total_drawings : ℕ) (neighbors : ℕ) (drawings_per_neighbor : ℕ)
  (h1 : total_drawings = 54)
  (h2 : neighbors = 6)
  (h3 : total_drawings = neighbors * drawings_per_neighbor) :
  drawings_per_neighbor = 9 :=
by
  rw [h1, h2] at h3
  linarith

end NUMINAMATH_GPT_equal_distribution_of_drawings_l481_48195


namespace NUMINAMATH_GPT_total_money_is_correct_l481_48189

-- Define the values of different types of coins and the amount of each.
def gold_value : ℕ := 75
def silver_value : ℕ := 40
def bronze_value : ℕ := 20
def titanium_value : ℕ := 10

def gold_count : ℕ := 6
def silver_count : ℕ := 8
def bronze_count : ℕ := 10
def titanium_count : ℕ := 4
def cash : ℕ := 45

-- Define the total amount of money.
def total_money : ℕ :=
  (gold_count * gold_value) +
  (silver_count * silver_value) +
  (bronze_count * bronze_value) +
  (titanium_count * titanium_value) + cash

-- The proof statement
theorem total_money_is_correct : total_money = 1055 := by
  sorry

end NUMINAMATH_GPT_total_money_is_correct_l481_48189


namespace NUMINAMATH_GPT_probability_is_correct_l481_48118

-- Define the ratios for the colors: red, yellow, blue, black
def red_ratio := 6
def yellow_ratio := 2
def blue_ratio := 1
def black_ratio := 4

-- Define the total ratio
def total_ratio := red_ratio + yellow_ratio + blue_ratio + black_ratio

-- Define the ratio of red or blue regions
def red_or_blue_ratio := red_ratio + blue_ratio

-- Define the probability of landing on a red or blue region
def probability_red_or_blue := red_or_blue_ratio / total_ratio

-- State the theorem to prove
theorem probability_is_correct : probability_red_or_blue = 7 / 13 := 
by 
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_probability_is_correct_l481_48118


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l481_48172

theorem solve_equation_1 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by sorry

theorem solve_equation_2 (x : ℝ) : 2 * x^2 + 5 * x + 3 = 0 ↔ (x = -1 ∨ x = -3/2) :=
by sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l481_48172


namespace NUMINAMATH_GPT_cups_of_flour_required_l481_48185

/-- Define the number of cups of sugar and salt required by the recipe. --/
def sugar := 14
def salt := 7
/-- Define the number of cups of flour already added. --/
def flour_added := 2
/-- Define the additional requirement of flour being 3 more cups than salt. --/
def additional_flour_requirement := 3

/-- Main theorem to prove the total amount of flour the recipe calls for. --/
theorem cups_of_flour_required : total_flour = 10 :=
by
  sorry

end NUMINAMATH_GPT_cups_of_flour_required_l481_48185


namespace NUMINAMATH_GPT_part1_part2_l481_48130

-- Define the conditions
def P_condition (a x : ℝ) : Prop := 1 - a / x < 0
def Q_condition (x : ℝ) : Prop := abs (x + 2) < 3

-- First part: Given a = 3, prove the solution set P
theorem part1 (x : ℝ) : P_condition 3 x ↔ 0 < x ∧ x < 3 := by 
  sorry

-- Second part: Prove the range of values for the positive number a
theorem part2 (a : ℝ) (ha : 0 < a) : 
  (∀ x, (P_condition a x → Q_condition x)) → 0 < a ∧ a ≤ 1 := by 
  sorry

end NUMINAMATH_GPT_part1_part2_l481_48130


namespace NUMINAMATH_GPT_Miriam_gave_brother_60_marbles_l481_48169

def Miriam_current_marbles : ℕ := 30
def Miriam_initial_marbles : ℕ := 300
def brother_marbles (B : ℕ) : Prop := B = 60
def sister_marbles (B : ℕ) : ℕ := 2 * B
def friend_marbles : ℕ := 90
def total_given_away_marbles (B : ℕ) : ℕ := B + sister_marbles B + friend_marbles

theorem Miriam_gave_brother_60_marbles (B : ℕ) 
    (h1 : Miriam_current_marbles = 30) 
    (h2 : Miriam_initial_marbles = 300)
    (h3 : total_given_away_marbles B = Miriam_initial_marbles - Miriam_current_marbles) : 
    brother_marbles B :=
by 
    sorry

end NUMINAMATH_GPT_Miriam_gave_brother_60_marbles_l481_48169


namespace NUMINAMATH_GPT_amount_paid_is_correct_l481_48158

-- Define the conditions
def time_painting_house : ℕ := 8
def time_fixing_counter := 3 * time_painting_house
def time_mowing_lawn : ℕ := 6
def hourly_rate : ℕ := 15

-- Define the total time worked
def total_time_worked := time_painting_house + time_fixing_counter + time_mowing_lawn

-- Define the total amount paid
def total_amount_paid := total_time_worked * hourly_rate

-- Formalize the goal
theorem amount_paid_is_correct : total_amount_paid = 570 :=
by
  -- Proof steps to be filled in
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_amount_paid_is_correct_l481_48158


namespace NUMINAMATH_GPT_radius_of_circumcircle_of_triangle_l481_48178

theorem radius_of_circumcircle_of_triangle (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0) :
  a = 3 ∧ b = 4 ∧ c = 5 ∧ (∃ (R : ℝ), R = 2.5) :=
by {
  sorry
}

end NUMINAMATH_GPT_radius_of_circumcircle_of_triangle_l481_48178


namespace NUMINAMATH_GPT_intersection_A_B_l481_48173

-- Definitions based on conditions
variable (U : Set Int) (A B : Set Int)

#check Set

-- Given conditions
def U_def : Set Int := {-1, 3, 5, 7, 9}
def compl_U_A : Set Int := {-1, 9}
def B_def : Set Int := {3, 7, 9}

-- A is defined as the set difference of U and the complement of A in U
def A_def : Set Int := { x | x ∈ U_def ∧ ¬ (x ∈ compl_U_A) }

-- Theorem stating the intersection of A and B equals {3, 7}
theorem intersection_A_B : A_def ∩ B_def = {3, 7} :=
by
  -- Here would be the proof block, but we add 'sorry' to indicate it is unfinished.
  sorry

end NUMINAMATH_GPT_intersection_A_B_l481_48173


namespace NUMINAMATH_GPT_even_function_and_inverse_property_l481_48137

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem even_function_and_inverse_property (x : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  f (-x) = f x ∧ f (1 / x) = -f x := by
  sorry

end NUMINAMATH_GPT_even_function_and_inverse_property_l481_48137


namespace NUMINAMATH_GPT_range_of_a_l481_48146

open Set

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 3 → x^2 - a * x - a + 1 ≥ 0) ↔ a ≤ 5 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l481_48146


namespace NUMINAMATH_GPT_number_of_classes_l481_48183

theorem number_of_classes
  (p : ℕ) (s : ℕ) (t : ℕ) (c : ℕ)
  (hp : p = 2) (hs : s = 30) (ht : t = 360) :
  c = t / (p * s) :=
by
  simp [hp, hs, ht]
  sorry

end NUMINAMATH_GPT_number_of_classes_l481_48183


namespace NUMINAMATH_GPT_quadratic_b_value_l481_48112

theorem quadratic_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + b * x - 12 < 0 ↔ x < 3 ∨ x > 7) → b = 10 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_b_value_l481_48112


namespace NUMINAMATH_GPT_minimum_value_of_x_plus_y_existence_of_minimum_value_l481_48109

theorem minimum_value_of_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y + 2 * x + y = 8) :
  x + y ≥ 2 * Real.sqrt 10 - 3 :=
sorry

theorem existence_of_minimum_value (x y : ℝ) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 8 ∧ x + y = 2 * Real.sqrt 10 - 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_x_plus_y_existence_of_minimum_value_l481_48109


namespace NUMINAMATH_GPT_triangle_BC_60_l481_48104

theorem triangle_BC_60 {A B C X : Type}
    (AB AC BX CX : ℕ) (h1 : AB = 70) (h2 : AC = 80) 
    (h3 : AB^2 - BX^2 = CX*(CX + BX)) 
    (h4 : BX % 7 = 0)
    (h5 : BX + CX = (BC : ℕ)) 
    (h6 : BC = 60) :
  BC = 60 := 
sorry

end NUMINAMATH_GPT_triangle_BC_60_l481_48104


namespace NUMINAMATH_GPT_max_area_rectangle_l481_48143

/-- Given a rectangle with a perimeter of 40, the rectangle with the maximum area is a square
with sides of length 10. The maximum area is thus 100. -/
theorem max_area_rectangle (a b : ℝ) (h : a + b = 20) : a * b ≤ 100 :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l481_48143


namespace NUMINAMATH_GPT_three_integers_same_parity_l481_48164

theorem three_integers_same_parity (a b c : ℤ) : 
  (∃ i j, i ≠ j ∧ (i = a ∨ i = b ∨ i = c) ∧ (j = a ∨ j = b ∨ j = c) ∧ (i % 2 = j % 2)) :=
by
  sorry

end NUMINAMATH_GPT_three_integers_same_parity_l481_48164


namespace NUMINAMATH_GPT_obtuse_triangle_two_acute_angles_l481_48145

-- Define the angle type (could be Real between 0 and 180 in degrees).
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define an obtuse triangle using three angles α, β, γ
structure obtuse_triangle :=
(angle1 angle2 angle3 : ℝ)
(sum_angles_eq : angle1 + angle2 + angle3 = 180)
(obtuse_condition : is_obtuse angle1 ∨ is_obtuse angle2 ∨ is_obtuse angle3)

-- The theorem to prove the number of acute angles in an obtuse triangle is 2.
theorem obtuse_triangle_two_acute_angles (T : obtuse_triangle) : 
  (is_acute T.angle1 ∧ is_acute T.angle2 ∧ ¬ is_acute T.angle3) ∨ 
  (is_acute T.angle1 ∧ ¬ is_acute T.angle2 ∧ is_acute T.angle3) ∨ 
  (¬ is_acute T.angle1 ∧ is_acute T.angle2 ∧ is_acute T.angle3) :=
by sorry

end NUMINAMATH_GPT_obtuse_triangle_two_acute_angles_l481_48145


namespace NUMINAMATH_GPT_find_ab_l481_48132

theorem find_ab 
(a b : ℝ) 
(h1 : a + b = 2) 
(h2 : a * b = 1 ∨ a * b = -1) :
(a = 1 ∧ b = 1) ∨
(a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨
(a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_find_ab_l481_48132


namespace NUMINAMATH_GPT_winning_strategy_ping_pong_l481_48150

theorem winning_strategy_ping_pong:
  ∀ {n : ℕ}, n = 18 → (∀ a : ℕ, 1 ≤ a ∧ a ≤ 4 → (∀ k : ℕ, k = 3 * a → (∃ b : ℕ, 1 ≤ b ∧ b ≤ 4 ∧ n - k - b = 18 - (k + b))) → (∃ c : ℕ, c = 3)) :=
by
sorry

end NUMINAMATH_GPT_winning_strategy_ping_pong_l481_48150


namespace NUMINAMATH_GPT_maximize_revenue_l481_48198

theorem maximize_revenue (p : ℝ) (h₁ : p ≤ 30) (h₂ : p = 18.75) : 
  ∃(R : ℝ), R = p * (150 - 4 * p) :=
by
  sorry

end NUMINAMATH_GPT_maximize_revenue_l481_48198


namespace NUMINAMATH_GPT_find_polynomials_g_l481_48123

-- Define functions f and proof target is g
def f (x : ℝ) : ℝ := x ^ 2

-- g is defined as an unknown polynomial with some constraints
variable (g : ℝ → ℝ)

-- The proof problem stating that if f(g(x)) = 9x^2 + 12x + 4, 
-- then g(x) = 3x + 2 or g(x) = -3x - 2
theorem find_polynomials_g (h : ∀ x : ℝ, f (g x) = 9 * x ^ 2 + 12 * x + 4) :
  (∀ x : ℝ, g x = 3 * x + 2) ∨ (∀ x : ℝ, g x = -3 * x - 2) := 
by
  sorry

end NUMINAMATH_GPT_find_polynomials_g_l481_48123


namespace NUMINAMATH_GPT_negation_of_cos_proposition_l481_48151

variable (x : ℝ)

theorem negation_of_cos_proposition (h : ∀ x : ℝ, Real.cos x ≤ 1) : ∃ x₀ : ℝ, Real.cos x₀ > 1 :=
sorry

end NUMINAMATH_GPT_negation_of_cos_proposition_l481_48151


namespace NUMINAMATH_GPT_part1_part2_l481_48133

variables (A B C : ℝ)
variables (a b c : ℝ) -- sides of the triangle opposite to angles A, B, and C respectively

-- Part (I): Prove that c / a = 2 given b(cos A - 2 * cos C) = (2 * c - a) * cos B
theorem part1 (h1 : b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B) : c / a = 2 :=
sorry

-- Part (II): Prove that b = 2 given the results from part (I) and additional conditions
theorem part2 (h1 : c / a = 2) (h2 : Real.cos B = 1 / 4) (h3 : a + b + c = 5) : b = 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l481_48133


namespace NUMINAMATH_GPT_simplify_fraction_l481_48111

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l481_48111


namespace NUMINAMATH_GPT_last_three_digits_of_power_l481_48110

theorem last_three_digits_of_power (h : 3^400 ≡ 1 [MOD 800]) : 3^8000 ≡ 1 [MOD 800] :=
by {
  sorry
}

end NUMINAMATH_GPT_last_three_digits_of_power_l481_48110


namespace NUMINAMATH_GPT_lines_intersect_at_common_point_iff_l481_48125

theorem lines_intersect_at_common_point_iff (a b : ℝ) :
  (∃ x y : ℝ, a * x + 2 * b * y + 3 * (a + b + 1) = 0 ∧ 
               b * x + 2 * (a + b + 1) * y + 3 * a = 0 ∧ 
               (a + b + 1) * x + 2 * a * y + 3 * b = 0) ↔ 
  a + b = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_at_common_point_iff_l481_48125


namespace NUMINAMATH_GPT_value_of_f_f_f_2_l481_48184

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem value_of_f_f_f_2 : f (f (f 2)) = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_f_f_f_2_l481_48184


namespace NUMINAMATH_GPT_john_total_time_spent_l481_48107

-- Define conditions
def num_pictures : ℕ := 10
def draw_time_per_picture : ℝ := 2
def color_time_reduction : ℝ := 0.3

-- Define the actual color time per picture
def color_time_per_picture : ℝ := draw_time_per_picture * (1 - color_time_reduction)

-- Define the total time per picture
def total_time_per_picture : ℝ := draw_time_per_picture + color_time_per_picture

-- Define the total time for all pictures
def total_time_for_all_pictures : ℝ := total_time_per_picture * num_pictures

-- The theorem we need to prove
theorem john_total_time_spent : total_time_for_all_pictures = 34 :=
by
sorry

end NUMINAMATH_GPT_john_total_time_spent_l481_48107
