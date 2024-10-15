import Mathlib

namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2238_223877

-- Define that for all x in ℝ, x^2 - 4x + 2m ≥ 0
def proposition_p (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 - 4 * x + 2 * m ≥ 0

-- Main theorem statement
theorem necessary_but_not_sufficient (m : ℝ) : 
  (proposition_p m → m ≥ 2) → (m ≥ 1 → m ≥ 2) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2238_223877


namespace NUMINAMATH_GPT_problem_statement_l2238_223879

-- Define the arithmetic sequence and required terms
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
variables (a : ℕ → ℝ) (d : ℝ)
axiom seq_is_arithmetic : arithmetic_seq a d
axiom sum_of_a2_a4_a6_is_3 : a 2 + a 4 + a 6 = 3

-- Goal: Prove a1 + a3 + a5 + a7 = 4
theorem problem_statement : a 1 + a 3 + a 5 + a 7 = 4 :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l2238_223879


namespace NUMINAMATH_GPT_root_in_interval_l2238_223884

noncomputable def f (m x : ℝ) := m * 3^x - x + 3

theorem root_in_interval (m : ℝ) (h1 : m < 0) (h2 : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f m x = 0) : -3 < m ∧ m < -2/3 :=
by
  sorry

end NUMINAMATH_GPT_root_in_interval_l2238_223884


namespace NUMINAMATH_GPT_multiplication_correct_l2238_223876

theorem multiplication_correct :
  72514 * 99999 = 7250675486 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_correct_l2238_223876


namespace NUMINAMATH_GPT_linette_problem_proof_l2238_223855

def boxes_with_neither_markers_nor_stickers (total_boxes markers stickers both : ℕ) : ℕ :=
  total_boxes - (markers + stickers - both)

theorem linette_problem_proof : 
  let total_boxes := 15
  let markers := 9
  let stickers := 5
  let both := 4
  boxes_with_neither_markers_nor_stickers total_boxes markers stickers both = 5 :=
by
  sorry

end NUMINAMATH_GPT_linette_problem_proof_l2238_223855


namespace NUMINAMATH_GPT_difference_of_squares_eval_l2238_223890

-- Define the conditions
def a : ℕ := 81
def b : ℕ := 49

-- State the corresponding problem and its equivalence
theorem difference_of_squares_eval : (a^2 - b^2) = 4160 := by
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_difference_of_squares_eval_l2238_223890


namespace NUMINAMATH_GPT_rectangle_area_perimeter_l2238_223800

/-- 
Given a rectangle with positive integer sides a and b,
let A be the area and P be the perimeter.

A = a * b
P = 2 * a + 2 * b

Prove that 100 cannot be expressed as A + P - 4.
-/
theorem rectangle_area_perimeter (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (A : ℕ) (P : ℕ)
  (hA : A = a * b) (hP : P = 2 * a + 2 * b) : 
  ¬ (A + P - 4 = 100) := 
sorry

end NUMINAMATH_GPT_rectangle_area_perimeter_l2238_223800


namespace NUMINAMATH_GPT_range_a_for_inequality_l2238_223870

theorem range_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, (a-2) * x^2 - 2 * (a-2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_a_for_inequality_l2238_223870


namespace NUMINAMATH_GPT_even_odd_decomposition_exp_l2238_223822

variable (f g : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x
def decomposition (f g : ℝ → ℝ) := ∀ x, f x + g x = Real.exp x

-- Main statement to prove
theorem even_odd_decomposition_exp (hf : is_even f) (hg : is_odd g) (hfg : decomposition f g) :
  f (Real.log 2) + g (Real.log (1 / 2)) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_even_odd_decomposition_exp_l2238_223822


namespace NUMINAMATH_GPT_distance_between_city_A_and_B_is_180_l2238_223858

theorem distance_between_city_A_and_B_is_180
  (D : ℝ)
  (h1 : ∀ T_C : ℝ, T_C = D / 30)
  (h2 : ∀ T_D : ℝ, T_D = T_C - 1)
  (h3 : ∀ V_D : ℝ, V_D > 36 → T_D = D / V_D) :
  D = 180 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_city_A_and_B_is_180_l2238_223858


namespace NUMINAMATH_GPT_correct_option_D_l2238_223848

theorem correct_option_D : 
  (-3)^2 = 9 ∧ 
  - (x + y) = -x - y ∧ 
  ¬ (3 * a + 5 * b = 8 * a * b) ∧ 
  5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 :=
by { sorry }

end NUMINAMATH_GPT_correct_option_D_l2238_223848


namespace NUMINAMATH_GPT_ked_ben_eggs_ratio_l2238_223882

theorem ked_ben_eggs_ratio 
  (saly_needs_ben_weekly_ratio : ℕ)
  (weeks_in_month : ℕ := 4) 
  (total_production_month : ℕ := 124)
  (saly_needs_weekly : ℕ := 10) 
  (ben_needs_weekly : ℕ := 14)
  (ben_needs_monthly : ℕ := ben_needs_weekly * weeks_in_month)
  (saly_needs_monthly : ℕ := saly_needs_weekly * weeks_in_month)
  (total_saly_ben_monthly : ℕ := saly_needs_monthly + ben_needs_monthly)
  (ked_needs_monthly : ℕ := total_production_month - total_saly_ben_monthly)
  (ked_needs_weekly : ℕ := ked_needs_monthly / weeks_in_month) :
  ked_needs_weekly / ben_needs_weekly = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ked_ben_eggs_ratio_l2238_223882


namespace NUMINAMATH_GPT_number_of_pizzas_ordered_l2238_223801

-- Definitions from conditions
def slices_per_pizza : Nat := 2
def total_slices : Nat := 28

-- Proof that the number of pizzas ordered is 14
theorem number_of_pizzas_ordered : total_slices / slices_per_pizza = 14 := by
  sorry

end NUMINAMATH_GPT_number_of_pizzas_ordered_l2238_223801


namespace NUMINAMATH_GPT_pure_imaginary_solution_second_quadrant_solution_l2238_223867

def isPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

def isSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

def complexNumber (m : ℝ) : ℂ :=
  ⟨m^2 - 2*m - 3, m^2 + 3*m + 2⟩

theorem pure_imaginary_solution (m : ℝ) : isPureImaginary (complexNumber m) ↔ m = 3 :=
by sorry

theorem second_quadrant_solution (m : ℝ) : isSecondQuadrant (complexNumber m) ↔ (-1 < m ∧ m < 3) :=
by sorry

end NUMINAMATH_GPT_pure_imaginary_solution_second_quadrant_solution_l2238_223867


namespace NUMINAMATH_GPT_transformed_graph_equation_l2238_223823

theorem transformed_graph_equation (x y x' y' : ℝ)
  (h1 : x' = 5 * x)
  (h2 : y' = 3 * y)
  (h3 : x^2 + y^2 = 1) :
  x'^2 / 25 + y'^2 / 9 = 1 :=
by
  sorry

end NUMINAMATH_GPT_transformed_graph_equation_l2238_223823


namespace NUMINAMATH_GPT_remaining_sweet_cookies_correct_remaining_salty_cookies_correct_remaining_chocolate_cookies_correct_l2238_223857

-- Definition of initial conditions
def initial_sweet_cookies := 34
def initial_salty_cookies := 97
def initial_chocolate_cookies := 45

def sweet_cookies_eaten := 15
def salty_cookies_eaten := 56
def chocolate_cookies_given_away := 22
def chocolate_cookies_given_back := 7

-- Calculate remaining cookies
def remaining_sweet_cookies : Nat := initial_sweet_cookies - sweet_cookies_eaten
def remaining_salty_cookies : Nat := initial_salty_cookies - salty_cookies_eaten
def remaining_chocolate_cookies : Nat := (initial_chocolate_cookies - chocolate_cookies_given_away) + chocolate_cookies_given_back

-- Theorem statements
theorem remaining_sweet_cookies_correct : remaining_sweet_cookies = 19 := 
by sorry

theorem remaining_salty_cookies_correct : remaining_salty_cookies = 41 := 
by sorry

theorem remaining_chocolate_cookies_correct : remaining_chocolate_cookies = 30 := 
by sorry

end NUMINAMATH_GPT_remaining_sweet_cookies_correct_remaining_salty_cookies_correct_remaining_chocolate_cookies_correct_l2238_223857


namespace NUMINAMATH_GPT_repeating_decimal_fractional_representation_l2238_223862

theorem repeating_decimal_fractional_representation :
  (0.36 : ℝ) = (4 / 11 : ℝ) :=
sorry

end NUMINAMATH_GPT_repeating_decimal_fractional_representation_l2238_223862


namespace NUMINAMATH_GPT_evaluate_expression_l2238_223838

theorem evaluate_expression :
  (3 + 6 + 9 : ℚ) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2238_223838


namespace NUMINAMATH_GPT_remaining_area_correct_l2238_223819

noncomputable def remaining_area_ABHFGD : ℝ :=
  let area_square_ABCD := 25
  let area_square_EFGD := 16
  let side_length_ABCD := Real.sqrt area_square_ABCD
  let side_length_EFGD := Real.sqrt area_square_EFGD
  let overlap_area := 8
  area_square_ABCD + area_square_EFGD - overlap_area

theorem remaining_area_correct :
  let area := remaining_area_ABHFGD
  area = 33 :=
by
  sorry

end NUMINAMATH_GPT_remaining_area_correct_l2238_223819


namespace NUMINAMATH_GPT_gcd_91_49_l2238_223852

theorem gcd_91_49 : Int.gcd 91 49 = 7 := by
  sorry

end NUMINAMATH_GPT_gcd_91_49_l2238_223852


namespace NUMINAMATH_GPT_average_percentage_decrease_l2238_223812

theorem average_percentage_decrease
  (original_price final_price : ℕ)
  (h_original_price : original_price = 2000)
  (h_final_price : final_price = 1280) :
  (original_price - final_price) / original_price * 100 / 2 = 18 :=
by 
  sorry

end NUMINAMATH_GPT_average_percentage_decrease_l2238_223812


namespace NUMINAMATH_GPT_cos_value_of_geometric_sequence_l2238_223832

theorem cos_value_of_geometric_sequence (a : ℕ → ℝ) (r : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * r)
  (h2 : a 1 * a 13 + 2 * (a 7) ^ 2 = 5 * Real.pi) :
  Real.cos (a 2 * a 12) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_cos_value_of_geometric_sequence_l2238_223832


namespace NUMINAMATH_GPT_integer_part_inequality_l2238_223889

theorem integer_part_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
 (h_cond : (x + y + z) * ((1 / x) + (1 / y) + (1 / z)) = (91 / 10)) :
  (⌊(x^3 + y^3 + z^3) * ((1 / x^3) + (1 / y^3) + (1 / z^3))⌋) = 9 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_integer_part_inequality_l2238_223889


namespace NUMINAMATH_GPT_perimeter_of_ABC_HI_IJK_l2238_223874

theorem perimeter_of_ABC_HI_IJK (AB AC AH HI AI AK KI IJ JK : ℝ) 
(H_midpoint : H = AC / 2) (K_midpoint : K = AI / 2) 
(equil_triangle_ABC : AB = AC) (equil_triangle_AHI : AH = HI ∧ HI = AI) 
(equil_triangle_IJK : IJ = JK ∧ JK = KI) 
(AB_eq : AB = 6) : 
  AB + AC + AH + HI + IJ + JK + KI = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_ABC_HI_IJK_l2238_223874


namespace NUMINAMATH_GPT_xiao_ming_equation_l2238_223840

-- Defining the parameters of the problem
def distance : ℝ := 2000
def regular_time (x : ℝ) := x
def increased_speed := 5
def time_saved := 2

-- Problem statement to be proven in Lean 4:
theorem xiao_ming_equation (x : ℝ) (h₁ : x > 2) : 
  (distance / (x - time_saved)) - (distance / regular_time x) = increased_speed :=
by
  sorry

end NUMINAMATH_GPT_xiao_ming_equation_l2238_223840


namespace NUMINAMATH_GPT_rationalize_denominator_l2238_223824

theorem rationalize_denominator : (1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l2238_223824


namespace NUMINAMATH_GPT_CoinRun_ProcGen_ratio_l2238_223881

theorem CoinRun_ProcGen_ratio
  (greg_ppo_reward: ℝ)
  (maximum_procgen_reward: ℝ)
  (ppo_ratio: ℝ)
  (maximum_coinrun_reward: ℝ)
  (coinrun_to_procgen_ratio: ℝ)
  (greg_ppo_reward_eq: greg_ppo_reward = 108)
  (maximum_procgen_reward_eq: maximum_procgen_reward = 240)
  (ppo_ratio_eq: ppo_ratio = 0.90)
  (coinrun_equation: maximum_coinrun_reward = greg_ppo_reward / ppo_ratio)
  (ratio_definition: coinrun_to_procgen_ratio = maximum_coinrun_reward / maximum_procgen_reward) :
  coinrun_to_procgen_ratio = 0.5 :=
sorry

end NUMINAMATH_GPT_CoinRun_ProcGen_ratio_l2238_223881


namespace NUMINAMATH_GPT_topsoil_cost_l2238_223826

theorem topsoil_cost
  (cost_per_cubic_foot : ℕ)
  (volume_cubic_yards : ℕ)
  (conversion_factor : ℕ)
  (volume_cubic_feet : ℕ := volume_cubic_yards * conversion_factor)
  (total_cost : ℕ := volume_cubic_feet * cost_per_cubic_foot)
  (cost_per_cubic_foot_def : cost_per_cubic_foot = 8)
  (volume_cubic_yards_def : volume_cubic_yards = 8)
  (conversion_factor_def : conversion_factor = 27) :
  total_cost = 1728 := by
  sorry

end NUMINAMATH_GPT_topsoil_cost_l2238_223826


namespace NUMINAMATH_GPT_teacher_age_is_56_l2238_223809

theorem teacher_age_is_56 (s t : ℝ) (h1 : s = 40 * 15) (h2 : s + t = 41 * 16) : t = 56 := by
  sorry

end NUMINAMATH_GPT_teacher_age_is_56_l2238_223809


namespace NUMINAMATH_GPT_prime_iff_even_and_power_of_two_l2238_223841

theorem prime_iff_even_and_power_of_two (a n : ℕ) (h_pos_a : a > 1) (h_pos_n : n > 0) :
  Nat.Prime (a^n + 1) → (∃ k : ℕ, a = 2 * k) ∧ (∃ m : ℕ, n = 2^m) :=
by 
  sorry

end NUMINAMATH_GPT_prime_iff_even_and_power_of_two_l2238_223841


namespace NUMINAMATH_GPT_product_PA_PB_eq_nine_l2238_223896

theorem product_PA_PB_eq_nine 
  (P A B : ℝ × ℝ) 
  (hP : P = (3, 1)) 
  (h1 : A ≠ B)
  (h2 : ∃ L : ℝ × ℝ → Prop, L P ∧ L A ∧ L B) 
  (h3 : A.fst ^ 2 + A.snd ^ 2 = 1) 
  (h4 : B.fst ^ 2 + B.snd ^ 2 = 1) : 
  |((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2)| * |((P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2)| = 9 := 
sorry

end NUMINAMATH_GPT_product_PA_PB_eq_nine_l2238_223896


namespace NUMINAMATH_GPT_irrational_sum_floor_eq_iff_l2238_223811

theorem irrational_sum_floor_eq_iff (a b c d : ℝ) (h_irr_a : ¬ ∃ (q : ℚ), a = q) 
                                     (h_irr_b : ¬ ∃ (q : ℚ), b = q) 
                                     (h_irr_c : ¬ ∃ (q : ℚ), c = q) 
                                     (h_irr_d : ¬ ∃ (q : ℚ), d = q) 
                                     (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
                                     (h_pos_c : 0 < c) (h_pos_d : 0 < d)
                                     (h_sum_ab : a + b = 1) :
  (c + d = 1) ↔ (∀ (n : ℕ), ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) :=
sorry

end NUMINAMATH_GPT_irrational_sum_floor_eq_iff_l2238_223811


namespace NUMINAMATH_GPT_moles_of_magnesium_l2238_223846

-- Assuming the given conditions as hypotheses
variables (Mg CO₂ MgO C : ℕ)

-- Theorem statement
theorem moles_of_magnesium (h1 : 2 * Mg + CO₂ = 2 * MgO + C) 
                           (h2 : MgO = Mg) 
                           (h3 : CO₂ = 1) 
                           : Mg = 2 :=
by sorry  -- Proof to be provided

end NUMINAMATH_GPT_moles_of_magnesium_l2238_223846


namespace NUMINAMATH_GPT_jason_seashells_after_giving_l2238_223804

-- Define the number of seashells Jason originally found
def original_seashells : ℕ := 49

-- Define the number of seashells Jason gave to Tim
def seashells_given : ℕ := 13

-- Prove that the number of seashells Jason now has is 36
theorem jason_seashells_after_giving : original_seashells - seashells_given = 36 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_jason_seashells_after_giving_l2238_223804


namespace NUMINAMATH_GPT_find_p_l2238_223854

theorem find_p :
  ∀ r s : ℝ, (3 * r^2 + 4 * r + 2 = 0) → (3 * s^2 + 4 * s + 2 = 0) →
  (∀ p q : ℝ, (p = - (1/(r^2)) - (1/(s^2))) → (p = -1)) :=
by 
  intros r s hr hs p q hp
  sorry

end NUMINAMATH_GPT_find_p_l2238_223854


namespace NUMINAMATH_GPT_trig_identity_l2238_223883

theorem trig_identity (α : ℝ) (h : Real.sin (π + α) = 1 / 2) : Real.cos (α - 3 / 2 * π) = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_trig_identity_l2238_223883


namespace NUMINAMATH_GPT_largest_difference_l2238_223860

theorem largest_difference (P Q R S T U : ℕ) 
    (hP : P = 3 * 2003 ^ 2004)
    (hQ : Q = 2003 ^ 2004)
    (hR : R = 2002 * 2003 ^ 2003)
    (hS : S = 3 * 2003 ^ 2003)
    (hT : T = 2003 ^ 2003)
    (hU : U = 2003 ^ 2002) 
    : max (P - Q) (max (Q - R) (max (R - S) (max (S - T) (T - U)))) = P - Q :=
sorry

end NUMINAMATH_GPT_largest_difference_l2238_223860


namespace NUMINAMATH_GPT_find_breadth_l2238_223814

-- Define variables and constants
variables (SA l h w : ℝ)

-- Given conditions
axiom h1 : SA = 2400
axiom h2 : l = 15
axiom h3 : h = 16

-- Define the surface area equation for a cuboid 
def surface_area := 2 * (l * w + l * h + w * h)

-- Statement to prove
theorem find_breadth : surface_area l w h = SA → w = 30.97 := sorry

end NUMINAMATH_GPT_find_breadth_l2238_223814


namespace NUMINAMATH_GPT_change_of_b_l2238_223808

variable {t b1 b2 C C_new : ℝ}

theorem change_of_b (hC : C = t * b1^4) 
                   (hC_new : C_new = 16 * C) 
                   (hC_new_eq : C_new = t * b2^4) : 
                   b2 = 2 * b1 :=
by
  sorry

end NUMINAMATH_GPT_change_of_b_l2238_223808


namespace NUMINAMATH_GPT_largest_value_of_n_l2238_223880

theorem largest_value_of_n :
  ∃ (n : ℕ) (X Y Z : ℕ),
    n = 25 * X + 5 * Y + Z ∧
    n = 81 * Z + 9 * Y + X ∧
    X < 5 ∧ Y < 5 ∧ Z < 5 ∧
    n = 121 := by
  sorry

end NUMINAMATH_GPT_largest_value_of_n_l2238_223880


namespace NUMINAMATH_GPT_arcsin_cos_solution_l2238_223803

theorem arcsin_cos_solution (x : ℝ) (h : -π/2 ≤ x/3 ∧ x/3 ≤ π/2) :
  x = 3*π/10 ∨ x = 3*π/8 := 
sorry

end NUMINAMATH_GPT_arcsin_cos_solution_l2238_223803


namespace NUMINAMATH_GPT_smallest_relatively_prime_to_180_is_7_l2238_223843

theorem smallest_relatively_prime_to_180_is_7 :
  ∃ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 ∧ ∀ z : ℕ, z > 1 ∧ Nat.gcd z 180 = 1 → y ≤ z :=
by
  sorry

end NUMINAMATH_GPT_smallest_relatively_prime_to_180_is_7_l2238_223843


namespace NUMINAMATH_GPT_children_difference_l2238_223821

-- Define the initial number of children on the bus
def initial_children : ℕ := 5

-- Define the number of children who got off the bus
def children_off : ℕ := 63

-- Define the number of children on the bus after more got on
def final_children : ℕ := 14

-- Define the number of children who got on the bus
def children_on : ℕ := (final_children + children_off) - initial_children

-- Prove the number of children who got on minus the number of children who got off is equal to 9
theorem children_difference :
  (children_on - children_off) = 9 :=
by
  -- Direct translation from the proof steps
  sorry

end NUMINAMATH_GPT_children_difference_l2238_223821


namespace NUMINAMATH_GPT_number_of_ways_is_25_l2238_223872

-- Define the number of books
def number_of_books : ℕ := 5

-- Define the function to calculate the number of ways
def number_of_ways_to_buy_books : ℕ :=
  number_of_books * number_of_books

-- Define the theorem to be proved
theorem number_of_ways_is_25 : 
  number_of_ways_to_buy_books = 25 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_is_25_l2238_223872


namespace NUMINAMATH_GPT_rationalize_denominator_l2238_223853

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  A + B + C + D + E = 22 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l2238_223853


namespace NUMINAMATH_GPT_winner_more_votes_than_second_place_l2238_223805

theorem winner_more_votes_than_second_place :
  ∃ (W S T F : ℕ), 
    F = 199 ∧
    W = S + (W - S) ∧
    W = T + 79 ∧
    W = F + 105 ∧
    W + S + T + F = 979 ∧
    W - S = 53 :=
by
  sorry

end NUMINAMATH_GPT_winner_more_votes_than_second_place_l2238_223805


namespace NUMINAMATH_GPT_road_length_l2238_223875

theorem road_length 
  (D : ℕ) (N1 : ℕ) (t : ℕ) (d1 : ℝ) (N_extra : ℝ) 
  (h1 : D = 300) (h2 : N1 = 35) (h3 : t = 100) (h4 : d1 = 2.5) (h5 : N_extra = 52.5) : 
  ∃ L : ℝ, L = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_road_length_l2238_223875


namespace NUMINAMATH_GPT_not_even_not_odd_neither_even_nor_odd_l2238_223899

noncomputable def f (x : ℝ) : ℝ := ⌊x⌋ + 1 / 2

theorem not_even (x : ℝ) : f (-x) ≠ f x := sorry
theorem not_odd (x : ℝ) : f (0) ≠ 0 ∨ f (-x) ≠ -f x := sorry

theorem neither_even_nor_odd : ∀ x : ℝ, f (-x) ≠ f x ∧ (f (0) ≠ 0 ∨ f (-x) ≠ -f x) :=
by
  intros x
  exact ⟨not_even x, not_odd x⟩

end NUMINAMATH_GPT_not_even_not_odd_neither_even_nor_odd_l2238_223899


namespace NUMINAMATH_GPT_jia_winning_strategy_l2238_223845

variables {p q : ℝ}
def is_quadratic_real_roots (a b c : ℝ) : Prop := b ^ 2 - 4 * a * c > 0

def quadratic_with_roots (x1 x2 : ℝ) :=
  x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ is_quadratic_real_roots 1 (- (x1 + x2)) (x1 * x2)

def modify_jia (p q x1 : ℝ) : (ℝ × ℝ) := (p + 1, q - x1)

def modify_yi1 (p q : ℝ) : (ℝ × ℝ) := (p - 1, q)

def modify_yi2 (p q x2 : ℝ) : (ℝ × ℝ) := (p - 1, q + x2)

def winning_strategy_jia (x1 x2 : ℝ) : Prop :=
  ∃ n : ℕ, ∀ m ≥ n, ∀ p q, quadratic_with_roots x1 x2 → 
  (¬ is_quadratic_real_roots 1 p q) ∨ (q ≤ 0)

theorem jia_winning_strategy (x1 x2 : ℝ)
  (h: quadratic_with_roots x1 x2) : 
  winning_strategy_jia x1 x2 :=
sorry

end NUMINAMATH_GPT_jia_winning_strategy_l2238_223845


namespace NUMINAMATH_GPT_games_that_didnt_work_l2238_223863

variable (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (good_games : ℕ)

theorem games_that_didnt_work
  (h₁ : games_from_friend = 2)
  (h₂ : games_from_garage_sale = 2)
  (h₃ : good_games = 2) :
  (games_from_friend + games_from_garage_sale - good_games) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_games_that_didnt_work_l2238_223863


namespace NUMINAMATH_GPT_distance_problem_l2238_223828

noncomputable def distance_point_to_plane 
  (x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ) : ℝ :=
  -- Equation of the plane passing through three points derived using determinants
  let a := x2 - x1
  let b := y2 - y1
  let c := z2 - z1
  let d := x3 - x1
  let e := y3 - y1
  let f := z3 - z1
  let A := b*f - c*e
  let B := c*d - a*f
  let C := a*e - b*d
  let D := -(A*x1 + B*y1 + C*z1)
  -- Distance from the given point to the above plane
  (|A*x0 + B*y0 + C*z0 + D|) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_problem :
  distance_point_to_plane 
  3 6 68 
  (-3) (-5) 6 
  2 1 (-4) 
  0 (-3) (-1) 
  = Real.sqrt 573 :=
by sorry

end NUMINAMATH_GPT_distance_problem_l2238_223828


namespace NUMINAMATH_GPT_maggie_earnings_proof_l2238_223827

def rate_per_subscription : ℕ := 5
def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_neighbor1 : ℕ := 2
def subscriptions_to_neighbor2 : ℕ := 2 * subscriptions_to_neighbor1
def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_neighbor1 + subscriptions_to_neighbor2
def total_earnings : ℕ := total_subscriptions * rate_per_subscription

theorem maggie_earnings_proof : total_earnings = 55 := by
  sorry

end NUMINAMATH_GPT_maggie_earnings_proof_l2238_223827


namespace NUMINAMATH_GPT_chloe_profit_l2238_223820

def cost_per_dozen : ℕ := 50
def sell_per_half_dozen : ℕ := 30
def total_dozens_sold : ℕ := 50

def total_cost (n: ℕ) : ℕ := n * cost_per_dozen
def total_revenue (n: ℕ) : ℕ := n * (sell_per_half_dozen * 2)
def profit (cost revenue : ℕ) : ℕ := revenue - cost

theorem chloe_profit : 
  profit (total_cost total_dozens_sold) (total_revenue total_dozens_sold) = 500 := 
by
  sorry

end NUMINAMATH_GPT_chloe_profit_l2238_223820


namespace NUMINAMATH_GPT_yard_fraction_occupied_by_flowerbeds_l2238_223831

noncomputable def rectangular_yard_area (length width : ℕ) : ℕ :=
  length * width

noncomputable def triangle_area (leg_length : ℕ) : ℕ :=
  2 * (1 / 2 * leg_length ^ 2)

theorem yard_fraction_occupied_by_flowerbeds :
  let length := 30
  let width := 7
  let parallel_side_short := 20
  let parallel_side_long := 30
  let flowerbed_leg := 7
  rectangular_yard_area length width ≠ 0 ∧
  triangle_area flowerbed_leg * 2 = 49 →
  (triangle_area flowerbed_leg * 2) / rectangular_yard_area length width = 7 / 30 :=
sorry

end NUMINAMATH_GPT_yard_fraction_occupied_by_flowerbeds_l2238_223831


namespace NUMINAMATH_GPT_max_value_of_expression_l2238_223885

noncomputable def expression (x : ℝ) : ℝ :=
  x^6 / (x^10 + 3 * x^8 - 5 * x^6 + 15 * x^4 + 25)

theorem max_value_of_expression : ∃ x : ℝ, (expression x) = 1 / 17 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2238_223885


namespace NUMINAMATH_GPT_projection_is_orthocenter_l2238_223891

-- Define a structure for a point in 3D space.
structure Point (α : Type) :=
(x : α)
(y : α)
(z : α)

-- Define mutually perpendicular edges condition.
def mutually_perpendicular {α : Type} [Field α] (A B C D : Point α) :=
(A.x - D.x) * (B.x - D.x) + (A.y - D.y) * (B.y - D.y) + (A.z - D.z) * (B.z - D.z) = 0 ∧
(A.x - D.x) * (C.x - D.x) + (A.y - D.y) * (C.y - D.y) + (A.z - D.z) * (C.z - D.z) = 0 ∧
(B.x - D.x) * (C.x - D.x) + (B.y - D.y) * (C.y - D.y) + (B.z - D.z) * (C.z - D.z) = 0

-- The main theorem statement.
theorem projection_is_orthocenter {α : Type} [Field α]
    (A B C D : Point α) (h : mutually_perpendicular A B C D) :
    ∃ O : Point α, -- there exists a point O (the orthocenter)
    (O.x * (B.y - A.y) + O.y * (A.y - B.y) + O.z * (A.y - B.y)) = 0 ∧
    (O.x * (C.y - B.y) + O.y * (B.y - C.y) + O.z * (B.y - C.y)) = 0 ∧
    (O.x * (A.y - C.y) + O.y * (C.y - A.y) + O.z * (C.y - A.y)) = 0 := 
sorry

end NUMINAMATH_GPT_projection_is_orthocenter_l2238_223891


namespace NUMINAMATH_GPT_min_varphi_symmetry_l2238_223813

theorem min_varphi_symmetry (ϕ : ℝ) (hϕ : ϕ > 0) :
  (∃ k : ℤ, ϕ = (4 * Real.pi) / 3 - k * Real.pi ∧ ϕ > 0 ∧ (∀ x : ℝ, Real.cos (x - ϕ + (4 * Real.pi) / 3) = Real.cos (-x - ϕ + (4 * Real.pi) / 3))) 
  → ϕ = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_min_varphi_symmetry_l2238_223813


namespace NUMINAMATH_GPT_find_a_if_lines_perpendicular_l2238_223895

theorem find_a_if_lines_perpendicular (a : ℝ) :
  (∀ x, (y1 : ℝ) = a * x - 2 → (y2 : ℝ) = (a + 2) * x + 1 → y1 * y2 = -1) → a = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_if_lines_perpendicular_l2238_223895


namespace NUMINAMATH_GPT_divisible_by_units_digit_l2238_223806

theorem divisible_by_units_digit :
  ∃ l : List ℕ, l = [21, 22, 24, 25] ∧ l.length = 4 := 
  sorry

end NUMINAMATH_GPT_divisible_by_units_digit_l2238_223806


namespace NUMINAMATH_GPT_number_of_draws_l2238_223868

-- Definition of the competition conditions
def competition_conditions (A B C D E : ℕ) : Prop :=
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧
  (A = B ∨ B = C ∨ C = D ∨ D = E) ∧
  15 ∣ (10000 * A + 1000 * B + 100 * C + 10 * D + E)

-- The main theorem stating the number of draws
theorem number_of_draws :
  ∃ (A B C D E : ℕ), competition_conditions A B C D E ∧ 
  (∃ (draws : ℕ), draws = 3) :=
by
  sorry

end NUMINAMATH_GPT_number_of_draws_l2238_223868


namespace NUMINAMATH_GPT_absolute_value_inequality_solution_set_l2238_223844

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 * x - 1| - |x - 2| < 0} = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_absolute_value_inequality_solution_set_l2238_223844


namespace NUMINAMATH_GPT_number_of_matches_among_three_players_l2238_223825

-- Define the given conditions
variables (n r : ℕ) -- n is the number of participants, r is the number of matches among the 3 players
variables (m : ℕ := 50) -- m is the total number of matches played

-- Given assumptions
def condition1 := m = 50
def condition2 := ∃ (n: ℕ), 50 = Nat.choose (n-3) 2 + r + (6 - 2 * r)

-- The target proof
theorem number_of_matches_among_three_players (n r : ℕ) (m : ℕ := 50)
  (h1 : m = 50)
  (h2 : ∃ (n: ℕ), 50 = Nat.choose (n-3) 2 + r + (6 - 2 * r)) :
  r = 1 :=
sorry

end NUMINAMATH_GPT_number_of_matches_among_three_players_l2238_223825


namespace NUMINAMATH_GPT_product_of_integers_l2238_223833

theorem product_of_integers (x y : ℕ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 40) : x * y = 99 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_integers_l2238_223833


namespace NUMINAMATH_GPT_apples_added_l2238_223898

theorem apples_added (initial_apples added_apples final_apples : ℕ) 
  (h1 : initial_apples = 8) 
  (h2 : final_apples = 13) 
  (h3 : final_apples = initial_apples + added_apples) : 
  added_apples = 5 :=
by
  sorry

end NUMINAMATH_GPT_apples_added_l2238_223898


namespace NUMINAMATH_GPT_find_k_exists_p3_p5_no_number_has_p2_and_p4_l2238_223816

def has_prop_pk (n k : ℕ) : Prop := ∃ lst : List ℕ, (∀ x ∈ lst, x > 1) ∧ (lst.length = k) ∧ (lst.prod = n)

theorem find_k_exists_p3_p5 :
  ∃ (k : ℕ), (k = 3) ∧ ∃ (n : ℕ), has_prop_pk n k ∧ has_prop_pk n (k + 2) :=
by {
  sorry
}

theorem no_number_has_p2_and_p4 :
  ¬ ∃ (n : ℕ), has_prop_pk n 2 ∧ has_prop_pk n 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_k_exists_p3_p5_no_number_has_p2_and_p4_l2238_223816


namespace NUMINAMATH_GPT_spinner_final_direction_north_l2238_223866

def start_direction := "north"
def clockwise_revolutions := (7 : ℚ) / 2
def counterclockwise_revolutions := (5 : ℚ) / 2
def net_revolutions := clockwise_revolutions - counterclockwise_revolutions

theorem spinner_final_direction_north :
  net_revolutions = 1 → start_direction = "north" → 
  start_direction = "north" :=
by
  intro h1 h2
  -- Here you would prove that net_revolutions of 1 full cycle leads back to start
  exact h2 -- Skipping proof

end NUMINAMATH_GPT_spinner_final_direction_north_l2238_223866


namespace NUMINAMATH_GPT_amount_with_r_l2238_223865

theorem amount_with_r (p q r T : ℝ) 
  (h1 : p + q + r = 4000)
  (h2 : r = (2/3) * T)
  (h3 : T = p + q) : 
  r = 1600 := by
  sorry

end NUMINAMATH_GPT_amount_with_r_l2238_223865


namespace NUMINAMATH_GPT_range_of_x_l2238_223864

noncomputable def integerPart (x : ℝ) : ℤ := Int.floor x

theorem range_of_x (x : ℝ) (h : integerPart ((1 - 3 * x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l2238_223864


namespace NUMINAMATH_GPT_lcm_9_12_15_l2238_223869

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end NUMINAMATH_GPT_lcm_9_12_15_l2238_223869


namespace NUMINAMATH_GPT_simplify_trig_expression_l2238_223836

theorem simplify_trig_expression :
  (Real.cos (72 * Real.pi / 180) * Real.sin (78 * Real.pi / 180) +
   Real.sin (72 * Real.pi / 180) * Real.sin (12 * Real.pi / 180) = 1 / 2) :=
by sorry

end NUMINAMATH_GPT_simplify_trig_expression_l2238_223836


namespace NUMINAMATH_GPT_shaded_area_correct_l2238_223861

-- Definitions of the given conditions
def first_rectangle_length : ℕ := 8
def first_rectangle_width : ℕ := 5
def second_rectangle_length : ℕ := 4
def second_rectangle_width : ℕ := 9
def overlapping_area : ℕ := 3

def first_rectangle_area := first_rectangle_length * first_rectangle_width
def second_rectangle_area := second_rectangle_length * second_rectangle_width

-- Problem statement in Lean 4
theorem shaded_area_correct :
  first_rectangle_area + second_rectangle_area - overlapping_area = 73 :=
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l2238_223861


namespace NUMINAMATH_GPT_quarts_of_water_needed_l2238_223815

-- Definitions of conditions
def total_parts := 5 + 2 + 1
def total_gallons := 3
def quarts_per_gallon := 4
def water_parts := 5

-- Lean proof statement
theorem quarts_of_water_needed :
  (water_parts : ℚ) * ((total_gallons * quarts_per_gallon) / total_parts) = 15 / 2 :=
by sorry

end NUMINAMATH_GPT_quarts_of_water_needed_l2238_223815


namespace NUMINAMATH_GPT_factorization_problem_l2238_223856

theorem factorization_problem (x : ℝ) :
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 =
  (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + 2) * (x + 1) * (x - 1) :=
sorry

end NUMINAMATH_GPT_factorization_problem_l2238_223856


namespace NUMINAMATH_GPT_circle_radius_on_sphere_l2238_223859

theorem circle_radius_on_sphere
  (sphere_radius : ℝ)
  (circle1_radius : ℝ)
  (circle2_radius : ℝ)
  (circle3_radius : ℝ)
  (all_circle_touch_each_other : Prop)
  (smaller_circle_touches_all : Prop)
  (smaller_circle_radius : ℝ) :
  sphere_radius = 2 →
  circle1_radius = 1 →
  circle2_radius = 1 →
  circle3_radius = 1 →
  all_circle_touch_each_other →
  smaller_circle_touches_all →
  smaller_circle_radius = 1 - Real.sqrt (2 / 3) :=
by
  intros h_sphere_radius h_circle1_radius h_circle2_radius h_circle3_radius h_all_circle_touch h_smaller_circle_touch
  sorry

end NUMINAMATH_GPT_circle_radius_on_sphere_l2238_223859


namespace NUMINAMATH_GPT_bridge_length_l2238_223849

theorem bridge_length (length_of_train : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) : 
  length_of_train = 110 → train_speed_kmph = 45 → time_seconds = 30 → 
  ∃ length_of_bridge : ℕ, length_of_bridge = 265 := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_bridge_length_l2238_223849


namespace NUMINAMATH_GPT_greatest_multiple_of_four_cubed_less_than_2000_l2238_223886

theorem greatest_multiple_of_four_cubed_less_than_2000 :
  ∃ x, (x > 0) ∧ (x % 4 = 0) ∧ (x^3 < 2000) ∧ ∀ y, (y > x) ∧ (y % 4 = 0) → y^3 ≥ 2000 :=
sorry

end NUMINAMATH_GPT_greatest_multiple_of_four_cubed_less_than_2000_l2238_223886


namespace NUMINAMATH_GPT_probability_of_one_red_ball_is_one_third_l2238_223893

-- Define the number of red and black balls
def red_balls : Nat := 2
def black_balls : Nat := 4
def total_balls : Nat := red_balls + black_balls

-- Define the probability calculation
def probability_red_ball : ℚ := red_balls / (red_balls + black_balls)

-- State the theorem
theorem probability_of_one_red_ball_is_one_third :
  probability_red_ball = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_one_red_ball_is_one_third_l2238_223893


namespace NUMINAMATH_GPT_minimum_distance_l2238_223818

theorem minimum_distance (m n : ℝ) (a : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ 4) 
  (h3 : m * Real.sqrt (Real.log a - 1 / 4) + 2 * a + 1 / 2 * n = 0) : 
  Real.sqrt (m^2 + n^2) = 4 * Real.sqrt (Real.log 2) / Real.log 2 :=
sorry

end NUMINAMATH_GPT_minimum_distance_l2238_223818


namespace NUMINAMATH_GPT_f_at_10_l2238_223817

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

-- Prove that f(10) = 756
theorem f_at_10 : f 10 = 756 := by
  sorry

end NUMINAMATH_GPT_f_at_10_l2238_223817


namespace NUMINAMATH_GPT_bertha_daughters_no_daughters_l2238_223871

theorem bertha_daughters_no_daughters (daughters granddaughters: ℕ) (no_great_granddaughters: granddaughters = 5 * daughters) (total_women: 8 + granddaughters = 48) :
  8 + granddaughters = 48 :=
by {
  sorry
}

end NUMINAMATH_GPT_bertha_daughters_no_daughters_l2238_223871


namespace NUMINAMATH_GPT_sum_of_xy_eq_20_l2238_223807

theorem sum_of_xy_eq_20 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt : x < 30) (hy_lt : y < 30)
    (hxy : x + y + x * y = 119) : x + y = 20 :=
sorry

end NUMINAMATH_GPT_sum_of_xy_eq_20_l2238_223807


namespace NUMINAMATH_GPT_calc_g_x_plus_3_l2238_223830

def g (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem calc_g_x_plus_3 (x : ℝ) : g (x + 3) = x^2 + 3*x + 2 :=
by
  sorry

end NUMINAMATH_GPT_calc_g_x_plus_3_l2238_223830


namespace NUMINAMATH_GPT_find_integers_l2238_223887

-- Problem statement rewritten as a Lean 4 definition
theorem find_integers (a b c : ℤ) (H1 : a = 1) (H2 : b = 2) (H3 : c = 1) : 
  a^2 + b^2 + c^2 + 3 < a * b + 3 * b + 2 * c :=
by
  -- The proof will be presented here
  sorry

end NUMINAMATH_GPT_find_integers_l2238_223887


namespace NUMINAMATH_GPT_tan_domain_correct_l2238_223834

noncomputable def domain_tan : Set ℝ := {x | ∃ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4}

def is_domain_correct : Prop :=
  ∀ x : ℝ, x ∈ domain_tan ↔ (∃ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4)

-- Statement of the problem in Lean 4
theorem tan_domain_correct : is_domain_correct :=
  sorry

end NUMINAMATH_GPT_tan_domain_correct_l2238_223834


namespace NUMINAMATH_GPT_total_accidents_l2238_223892

-- Define the given vehicle counts for the highways
def total_vehicles_A : ℕ := 4 * 10^9
def total_vehicles_B : ℕ := 2 * 10^9
def total_vehicles_C : ℕ := 1 * 10^9

-- Define the accident ratios per highway
def accident_ratio_A : ℕ := 80
def accident_ratio_B : ℕ := 120
def accident_ratio_C : ℕ := 65

-- Define the number of vehicles in millions
def million := 10^6

-- Define the accident calculations per highway
def accidents_A : ℕ := (total_vehicles_A / (100 * million)) * accident_ratio_A
def accidents_B : ℕ := (total_vehicles_B / (200 * million)) * accident_ratio_B
def accidents_C : ℕ := (total_vehicles_C / (50 * million)) * accident_ratio_C

-- Prove the total number of accidents across all highways
theorem total_accidents : accidents_A + accidents_B + accidents_C = 5700 := by
  have : accidents_A = 3200 := by sorry
  have : accidents_B = 1200 := by sorry
  have : accidents_C = 1300 := by sorry
  sorry

end NUMINAMATH_GPT_total_accidents_l2238_223892


namespace NUMINAMATH_GPT_inverse_var_q_value_l2238_223802

theorem inverse_var_q_value (p q : ℝ) (h1 : ∀ p q, (p * q = 400))
(p_init : p = 800) (q_init : q = 0.5) (new_p : p = 400) :
  q = 1 := by
  sorry

end NUMINAMATH_GPT_inverse_var_q_value_l2238_223802


namespace NUMINAMATH_GPT_geometric_seq_sum_l2238_223839

theorem geometric_seq_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * (-1)) 
  (h_a3 : a 3 = 3) 
  (h_sum_cond : a 2016 + a 2017 = 0) : 
  S 101 = 3 := 
by
  sorry

end NUMINAMATH_GPT_geometric_seq_sum_l2238_223839


namespace NUMINAMATH_GPT_compute_expression_value_l2238_223847

theorem compute_expression_value (x y : ℝ) (hxy : x ≠ y) 
  (h : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (xy + 1)) :
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (xy + 1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_value_l2238_223847


namespace NUMINAMATH_GPT_distance_between_circle_center_and_point_l2238_223835

theorem distance_between_circle_center_and_point (x y : ℝ) (h : x^2 + y^2 = 8*x - 12*y + 40) : 
  dist (4, -6) (4, -2) = 4 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_circle_center_and_point_l2238_223835


namespace NUMINAMATH_GPT_correct_calculation_l2238_223850

theorem correct_calculation : 
  ¬(2 * Real.sqrt 3 + 3 * Real.sqrt 2 = 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  ¬(5 * Real.sqrt 3 * 5 * Real.sqrt 2 = 5 * Real.sqrt 6) ∧
  ¬(Real.sqrt (4 + 1 / 2) = 2 * Real.sqrt (1 / 2)) :=
by {
  -- Using the conditions to prove the correct option B
  sorry
}

end NUMINAMATH_GPT_correct_calculation_l2238_223850


namespace NUMINAMATH_GPT_intersection_of_sets_eq_l2238_223878

noncomputable def set_intersection (M N : Set ℝ): Set ℝ :=
  {x | x ∈ M ∧ x ∈ N}

theorem intersection_of_sets_eq :
  let M := {x : ℝ | -2 < x ∧ x < 2}
  let N := {x : ℝ | x^2 - 2 * x - 3 < 0}
  set_intersection M N = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_eq_l2238_223878


namespace NUMINAMATH_GPT_cube_root_3375_l2238_223888

theorem cube_root_3375 (c d : ℕ) (h1 : c > 0 ∧ d > 0) (h2 : c * d^3 = 3375) (h3 : ∀ k : ℕ, k > 0 → c * (d / k)^3 ≠ 3375) : 
  c + d = 16 :=
sorry

end NUMINAMATH_GPT_cube_root_3375_l2238_223888


namespace NUMINAMATH_GPT_complement_of_65_degrees_l2238_223829

def angle_complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_65_degrees : angle_complement 65 = 25 := by
  -- Proof would follow here, but it's omitted since 'sorry' is added.
  sorry

end NUMINAMATH_GPT_complement_of_65_degrees_l2238_223829


namespace NUMINAMATH_GPT_proof_problem_l2238_223837

-- Triangle and Point Definitions
variables {A B C P : Type}
variables (BC : ℝ) (a b c : ℝ) (PA PB PC : ℝ)

-- Conditions: Triangle ABC with angle A = 90 degrees and P on BC
def is_right_triangle (A B C : Type) (a b c : ℝ) (BC : ℝ) (angleA : ℝ := 90) : Prop :=
a^2 + b^2 = c^2 ∧ c = BC

def on_hypotenuse (P : Type) (BC : ℝ) (PB PC : ℝ) : Prop :=
PB + PC = BC

-- The proof problem
theorem proof_problem (A B C P : Type) 
  (BC : ℝ) (a b c : ℝ) (PA PB PC : ℝ)
  (h1 : is_right_triangle A B C a b c BC)
  (h2 : on_hypotenuse P BC PB PC) :
  (a^2 / PC + b^2 / PB) ≥ (BC^3 / (PA^2 + PB * PC)) := 
sorry

end NUMINAMATH_GPT_proof_problem_l2238_223837


namespace NUMINAMATH_GPT_find_added_number_l2238_223897

theorem find_added_number 
  (initial_number : ℕ)
  (final_result : ℕ)
  (h : initial_number = 8)
  (h_result : 3 * (2 * initial_number + final_result) = 75) : 
  final_result = 9 := by
  sorry

end NUMINAMATH_GPT_find_added_number_l2238_223897


namespace NUMINAMATH_GPT_triangle_angle_C_l2238_223810

theorem triangle_angle_C (A B C : Real) (h1 : A - B = 10) (h2 : B = A / 2) :
  C = 150 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_triangle_angle_C_l2238_223810


namespace NUMINAMATH_GPT_number_of_girls_attending_winter_festival_l2238_223894

variables (g b : ℝ)
variables (totalStudents attendFestival: ℝ)

theorem number_of_girls_attending_winter_festival
  (H1 : g + b = 1500)
  (H2 : (3/5) * g + (2/5) * b = 800) :
  (3/5 * g) = 600 :=
sorry

end NUMINAMATH_GPT_number_of_girls_attending_winter_festival_l2238_223894


namespace NUMINAMATH_GPT_find_triples_l2238_223873

theorem find_triples (k : ℕ) (hk : 0 < k) :
  ∃ (a b c : ℕ), 
    (0 < a ∧ 0 < b ∧ 0 < c) ∧
    (a + b + c = 3 * k + 1) ∧ 
    (a * b + b * c + c * a = 3 * k^2 + 2 * k) ∧ 
    (a = k + 1 ∧ b = k ∧ c = k) :=
by
  sorry

end NUMINAMATH_GPT_find_triples_l2238_223873


namespace NUMINAMATH_GPT_find_f3_minus_f4_l2238_223851

noncomputable def f : ℝ → ℝ := sorry

axiom h_odd : ∀ x : ℝ, f (-x) = - f x
axiom h_periodic : ∀ x : ℝ, f (x + 5) = f x
axiom h_f1 : f 1 = 1
axiom h_f2 : f 2 = 2

theorem find_f3_minus_f4 : f 3 - f 4 = -1 := by
  sorry

end NUMINAMATH_GPT_find_f3_minus_f4_l2238_223851


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l2238_223842

theorem area_of_triangle_ABC 
  (r : ℝ) (R : ℝ) (ACB : ℝ) 
  (hr : r = 2) 
  (hR : R = 4) 
  (hACB : ACB = 120) : 
  let s := (2 * (2 + 4 * Real.sqrt 3)) / Real.sqrt 3 
  let S := s * r 
  S = 56 / Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l2238_223842
