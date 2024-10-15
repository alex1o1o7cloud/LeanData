import Mathlib

namespace NUMINAMATH_GPT_ant_rest_position_l1402_140241

noncomputable def percent_way_B_to_C (s : ℕ) : ℕ :=
  let perimeter := 3 * s
  let distance_traveled := (42 * perimeter) / 100
  let distance_AB := s
  let remaining_distance := distance_traveled - distance_AB
  (remaining_distance * 100) / s

theorem ant_rest_position :
  ∀ (s : ℕ), percent_way_B_to_C s = 26 :=
by
  intros
  unfold percent_way_B_to_C
  sorry

end NUMINAMATH_GPT_ant_rest_position_l1402_140241


namespace NUMINAMATH_GPT_total_cubes_l1402_140229

noncomputable def original_cubes : ℕ := 2
noncomputable def additional_cubes : ℕ := 7

theorem total_cubes : original_cubes + additional_cubes = 9 := by
  sorry

end NUMINAMATH_GPT_total_cubes_l1402_140229


namespace NUMINAMATH_GPT_three_digit_square_ends_with_self_l1402_140287

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end NUMINAMATH_GPT_three_digit_square_ends_with_self_l1402_140287


namespace NUMINAMATH_GPT_solve_fraction_eq_for_x_l1402_140267

theorem solve_fraction_eq_for_x (x : ℝ) (hx : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end NUMINAMATH_GPT_solve_fraction_eq_for_x_l1402_140267


namespace NUMINAMATH_GPT_intersection_M_N_l1402_140250

def M : Set ℝ := {x | (x - 1) * (x - 4) = 0}
def N : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem intersection_M_N :
  M ∩ N = {1} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l1402_140250


namespace NUMINAMATH_GPT_pig_problem_l1402_140244

theorem pig_problem (x y : ℕ) (h₁ : y - 100 = 100 * x) (h₂ : y = 90 * x) : x = 10 ∧ y = 900 := 
by
  sorry

end NUMINAMATH_GPT_pig_problem_l1402_140244


namespace NUMINAMATH_GPT_divide_composite_products_l1402_140256

def first_eight_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List ℕ := [16, 18, 20, 21, 22, 24, 25, 26]

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem divide_composite_products :
  product first_eight_composites * 3120 = product next_eight_composites :=
by
  -- This would be the place for the proof solution
  sorry

end NUMINAMATH_GPT_divide_composite_products_l1402_140256


namespace NUMINAMATH_GPT_least_number_to_add_1054_23_l1402_140200

def least_number_to_add (n k : ℕ) : ℕ :=
  let remainder := n % k
  if remainder = 0 then 0 else k - remainder

theorem least_number_to_add_1054_23 : least_number_to_add 1054 23 = 4 :=
by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_least_number_to_add_1054_23_l1402_140200


namespace NUMINAMATH_GPT_annie_crayons_l1402_140249

def initial_crayons : ℕ := 4
def additional_crayons : ℕ := 36
def total_crayons : ℕ := initial_crayons + additional_crayons

theorem annie_crayons : total_crayons = 40 :=
by
  sorry

end NUMINAMATH_GPT_annie_crayons_l1402_140249


namespace NUMINAMATH_GPT_number_of_smaller_cubes_l1402_140265

theorem number_of_smaller_cubes 
  (volume_large_cube : ℝ)
  (volume_small_cube : ℝ)
  (surface_area_difference : ℝ)
  (h1 : volume_large_cube = 216)
  (h2 : volume_small_cube = 1)
  (h3 : surface_area_difference = 1080) :
  ∃ n : ℕ, n * 6 - 6 * (volume_large_cube^(1/3))^2 = surface_area_difference ∧ n = 216 :=
by
  sorry

end NUMINAMATH_GPT_number_of_smaller_cubes_l1402_140265


namespace NUMINAMATH_GPT_brother_and_sister_ages_l1402_140248

theorem brother_and_sister_ages :
  ∃ (b s : ℕ), (b - 3 = 7 * (s - 3)) ∧ (b - 2 = 4 * (s - 2)) ∧ (b - 1 = 3 * (s - 1)) ∧ (b = 5 / 2 * s) ∧ b = 10 ∧ s = 4 :=
by 
  sorry

end NUMINAMATH_GPT_brother_and_sister_ages_l1402_140248


namespace NUMINAMATH_GPT_total_students_l1402_140279

theorem total_students (n1 n2 : ℕ) (h1 : (158 - 140)/(n1 + 1) = 2) (h2 : (158 - 140)/(n2 + 1) = 3) :
  n1 + n2 + 2 = 15 :=
sorry

end NUMINAMATH_GPT_total_students_l1402_140279


namespace NUMINAMATH_GPT_sin_of_angle_F_l1402_140285

theorem sin_of_angle_F 
  (DE EF DF : ℝ) 
  (h : DE = 12) 
  (h0 : EF = 20) 
  (h1 : DF = Real.sqrt (DE^2 + EF^2)) : 
  Real.sin (Real.arctan (DF / EF)) = 12 / Real.sqrt (DE^2 + EF^2) := 
by 
  sorry

end NUMINAMATH_GPT_sin_of_angle_F_l1402_140285


namespace NUMINAMATH_GPT_find_fractions_l1402_140255

-- Define the numerators and denominators
def p1 := 75
def p2 := 70
def q1 := 34
def q2 := 51

-- Define the fractions
def frac1 := p1 / q1
def frac2 := p1 / q2

-- Define the greatest common divisor (gcd) condition
def gcd_condition := Nat.gcd p1 p2 = p1 - p2

-- Define the least common multiple (lcm) condition
def lcm_condition := Nat.lcm p1 p2 = 1050

-- Define the difference condition
def difference_condition := (frac1 - frac2) = (5 / 6)

-- Lean proof statement
theorem find_fractions :
  gcd_condition ∧ lcm_condition ∧ difference_condition :=
by
  sorry

end NUMINAMATH_GPT_find_fractions_l1402_140255


namespace NUMINAMATH_GPT_prices_correct_minimum_cost_correct_l1402_140270

-- Define the prices of the mustard brands
variables (x y m : ℝ)

def brandACost : ℝ := 9 * x + 6 * y
def brandBCost : ℝ := 5 * x + 8 * y

-- Conditions for prices
axiom cost_condition1 : brandACost x y = 390
axiom cost_condition2 : brandBCost x y = 310

-- Solution for prices
def priceA : ℝ := 30
def priceB : ℝ := 20

theorem prices_correct : x = priceA ∧ y = priceB :=
sorry

-- Conditions for minimizing cost
def totalCost (m : ℝ) : ℝ := 30 * m + 20 * (30 - m)
def totalPacks : ℝ := 30

-- Constraints
def constraint1 (m : ℝ) : Prop := m ≥ 5 + (30 - m)
def constraint2 (m : ℝ) : Prop := m ≤ 2 * (30 - m)

-- Minimum cost condition
def min_cost : ℝ := 780
def optimal_m : ℝ := 18

theorem minimum_cost_correct : constraint1 optimal_m ∧ constraint2 optimal_m ∧ totalCost optimal_m = min_cost :=
sorry

end NUMINAMATH_GPT_prices_correct_minimum_cost_correct_l1402_140270


namespace NUMINAMATH_GPT_verify_BG_BF_verify_FG_EG_find_x_l1402_140211

noncomputable def verify_angles (CBG GBE EBF BCF FCE : ℝ) :=
  CBG = 20 ∧ GBE = 40 ∧ EBF = 20 ∧ BCF = 50 ∧ FCE = 30

theorem verify_BG_BF (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → BG = BF :=
by
  sorry

theorem verify_FG_EG (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → FG = EG :=
by
  sorry

theorem find_x (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → x = 30 :=
by
  sorry

end NUMINAMATH_GPT_verify_BG_BF_verify_FG_EG_find_x_l1402_140211


namespace NUMINAMATH_GPT_not_both_267_and_269_non_standard_l1402_140232

def G : ℤ → ℤ := sorry

def exists_x_ne_c (G : ℤ → ℤ) : Prop :=
  ∀ c : ℤ, ∃ x : ℤ, G x ≠ c

def non_standard (G : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, G x = G (a - x)

theorem not_both_267_and_269_non_standard (G : ℤ → ℤ)
  (h1 : exists_x_ne_c G) :
  ¬ (non_standard G 267 ∧ non_standard G 269) :=
sorry

end NUMINAMATH_GPT_not_both_267_and_269_non_standard_l1402_140232


namespace NUMINAMATH_GPT_derivative_at_one_l1402_140247

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

theorem derivative_at_one :
  deriv f 1 = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_one_l1402_140247


namespace NUMINAMATH_GPT_ellipse_area_l1402_140215

theorem ellipse_area :
  ∃ a b : ℝ, 
    (∀ x y : ℝ, (x^2 - 2 * x + 9 * y^2 + 18 * y + 16 = 0) → 
    (a = 2 ∧ b = (2 / 3) ∧ (π * a * b = 4 * π / 3))) :=
sorry

end NUMINAMATH_GPT_ellipse_area_l1402_140215


namespace NUMINAMATH_GPT_cube_eq_minus_one_l1402_140251

theorem cube_eq_minus_one (x : ℝ) (h : x = -2) : (x + 1) ^ 3 = -1 :=
by
  sorry

end NUMINAMATH_GPT_cube_eq_minus_one_l1402_140251


namespace NUMINAMATH_GPT_domain_ln_x_plus_one_l1402_140277

theorem domain_ln_x_plus_one :
  ∀ (x : ℝ), ∃ (y : ℝ), y = Real.log (x + 1) ↔ x > -1 :=
by sorry

end NUMINAMATH_GPT_domain_ln_x_plus_one_l1402_140277


namespace NUMINAMATH_GPT_A_inter_B_eq_A_union_C_U_B_eq_l1402_140273

section
  -- Define the universal set U
  def U : Set ℝ := { x | x^2 - (5 / 2) * x + 1 ≥ 0 }

  -- Define set A
  def A : Set ℝ := { x | |x - 1| > 1 }

  -- Define set B
  def B : Set ℝ := { x | (x + 1) / (x - 2) ≥ 0 }

  -- Define the complement of B in U
  def C_U_B : Set ℝ := U \ B

  -- Theorem for A ∩ B
  theorem A_inter_B_eq : A ∩ B = { x | x ≤ -1 ∨ x > 2 } := sorry

  -- Theorem for A ∪ (C_U_B)
  theorem A_union_C_U_B_eq : A ∪ C_U_B = U := sorry
end

end NUMINAMATH_GPT_A_inter_B_eq_A_union_C_U_B_eq_l1402_140273


namespace NUMINAMATH_GPT_find_c_l1402_140212

theorem find_c (c : ℝ) : (∃ a : ℝ, (x : ℝ) → (x^2 + 80*x + c = (x + a)^2)) → (c = 1600) := by
  sorry

end NUMINAMATH_GPT_find_c_l1402_140212


namespace NUMINAMATH_GPT_positive_number_y_l1402_140298

theorem positive_number_y (y : ℕ) (h1 : y > 0) (h2 : y^2 / 100 = 9) : y = 30 :=
by
  sorry

end NUMINAMATH_GPT_positive_number_y_l1402_140298


namespace NUMINAMATH_GPT_equivalence_sufficient_necessary_l1402_140209

-- Definitions for conditions
variables (A B : Prop)

-- Statement to prove
theorem equivalence_sufficient_necessary :
  (A → B) ↔ (¬B → ¬A) :=
by sorry

end NUMINAMATH_GPT_equivalence_sufficient_necessary_l1402_140209


namespace NUMINAMATH_GPT_remainder_when_sum_divided_by_5_l1402_140293

/-- Reinterpreting the same conditions and question: -/
theorem remainder_when_sum_divided_by_5 (a b c : ℕ) 
    (ha : a < 5) (hb : b < 5) (hc : c < 5) 
    (h1 : a * b * c % 5 = 1) 
    (h2 : 3 * c % 5 = 2)
    (h3 : 4 * b % 5 = (3 + b) % 5): 
    (a + b + c) % 5 = 4 := 
sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_by_5_l1402_140293


namespace NUMINAMATH_GPT_inequality_solution_l1402_140201

theorem inequality_solution (x : ℝ) :
  (2 * x^2 - 4 * x - 70 > 0) ∧ (x ≠ -2) ∧ (x ≠ 0) ↔ (x < -5 ∨ x > 7) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1402_140201


namespace NUMINAMATH_GPT_fraction_of_number_l1402_140268

theorem fraction_of_number : (7 / 8) * 64 = 56 := by
  sorry

end NUMINAMATH_GPT_fraction_of_number_l1402_140268


namespace NUMINAMATH_GPT_tangent_line_of_cubic_at_l1402_140224

theorem tangent_line_of_cubic_at (x y : ℝ) (h : y = x^3) (hx : x = 1) (hy : y = 1) : 
  3 * x - y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_of_cubic_at_l1402_140224


namespace NUMINAMATH_GPT_domain_of_tan_function_l1402_140261

theorem domain_of_tan_function :
  (∀ x : ℝ, ∀ k : ℤ, 2 * x - π / 4 ≠ k * π + π / 2 ↔ x ≠ (k * π) / 2 + 3 * π / 8) :=
sorry

end NUMINAMATH_GPT_domain_of_tan_function_l1402_140261


namespace NUMINAMATH_GPT_shaded_area_percentage_l1402_140206

theorem shaded_area_percentage (side : ℕ) (total_shaded_area : ℕ) (expected_percentage : ℕ)
  (h1 : side = 5)
  (h2 : total_shaded_area = 15)
  (h3 : expected_percentage = 60) :
  ((total_shaded_area : ℚ) / (side * side) * 100) = expected_percentage :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_percentage_l1402_140206


namespace NUMINAMATH_GPT_williams_land_percentage_l1402_140219

variable (total_tax : ℕ) (williams_tax : ℕ)

theorem williams_land_percentage (h1 : total_tax = 3840) (h2 : williams_tax = 480) : 
  (williams_tax:ℚ) / (total_tax:ℚ) * 100 = 12.5 := 
  sorry

end NUMINAMATH_GPT_williams_land_percentage_l1402_140219


namespace NUMINAMATH_GPT_new_team_average_weight_is_113_l1402_140222

-- Defining the given constants and conditions
def original_players := 7
def original_average_weight := 121 
def weight_new_player1 := 110 
def weight_new_player2 := 60 

-- Definition to calculate the new average weight
def new_average_weight : ℕ :=
  let original_total_weight := original_players * original_average_weight
  let new_total_weight := original_total_weight + weight_new_player1 + weight_new_player2
  let new_total_players := original_players + 2
  new_total_weight / new_total_players

-- Statement to prove
theorem new_team_average_weight_is_113 : new_average_weight = 113 :=
sorry

end NUMINAMATH_GPT_new_team_average_weight_is_113_l1402_140222


namespace NUMINAMATH_GPT_complementSetM_l1402_140246

open Set Real

-- The universal set U is the set of all real numbers
def universalSet : Set ℝ := univ

-- The set M is defined as {x | |x - 1| ≤ 2}
def setM : Set ℝ := {x : ℝ | |x - 1| ≤ 2}

-- We need to prove that the complement of M with respect to U is {x | x < -1 ∨ x > 3}
theorem complementSetM :
  (universalSet \ setM) = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end NUMINAMATH_GPT_complementSetM_l1402_140246


namespace NUMINAMATH_GPT_robert_turns_30_after_2_years_l1402_140213

variable (P R : ℕ) -- P for Patrick's age, R for Robert's age
variable (h1 : P = 14) -- Patrick is 14 years old now
variable (h2 : P * 2 = R) -- Patrick is half the age of Robert

theorem robert_turns_30_after_2_years : R + 2 = 30 :=
by
  -- Here should be the proof, but for now we skip it with sorry
  sorry

end NUMINAMATH_GPT_robert_turns_30_after_2_years_l1402_140213


namespace NUMINAMATH_GPT_shape_is_cone_l1402_140204

-- Define spherical coordinates
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the positive constant c
def c : ℝ := sorry

-- Assume c is positive
axiom c_positive : c > 0

-- Define the shape equation in spherical coordinates
def shape_equation (p : SphericalCoordinates) : Prop :=
  p.ρ = c * Real.sin p.φ

-- The theorem statement
theorem shape_is_cone (p : SphericalCoordinates) : shape_equation p → 
  ∃ z : ℝ, (z = p.ρ * Real.cos p.φ) ∧ (p.ρ ^ 2 = (c * Real.sin p.φ) ^ 2 + z ^ 2) :=
sorry

end NUMINAMATH_GPT_shape_is_cone_l1402_140204


namespace NUMINAMATH_GPT_find_x_l1402_140214

-- Introducing the main theorem
theorem find_x (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (x : ℝ) (h_x : 0 < x) : 
  let r := (4 * a) ^ (4 * b)
  let y := x ^ 2
  r = a ^ b * y → 
  x = 16 ^ b * a ^ (1.5 * b) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1402_140214


namespace NUMINAMATH_GPT_jerry_gets_logs_l1402_140238

def logs_per_pine_tree : ℕ := 80
def logs_per_maple_tree : ℕ := 60
def logs_per_walnut_tree : ℕ := 100
def logs_per_oak_tree : ℕ := 90
def logs_per_birch_tree : ℕ := 55

def pine_trees_cut : ℕ := 8
def maple_trees_cut : ℕ := 3
def walnut_trees_cut : ℕ := 4
def oak_trees_cut : ℕ := 7
def birch_trees_cut : ℕ := 5

def total_logs : ℕ :=
  pine_trees_cut * logs_per_pine_tree +
  maple_trees_cut * logs_per_maple_tree +
  walnut_trees_cut * logs_per_walnut_tree +
  oak_trees_cut * logs_per_oak_tree +
  birch_trees_cut * logs_per_birch_tree

theorem jerry_gets_logs : total_logs = 2125 :=
by
  sorry

end NUMINAMATH_GPT_jerry_gets_logs_l1402_140238


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1402_140269

-- Definitions for costs
variables (x y : ℝ)
variables (cost_A cost_B : ℝ)

-- Conditions
def condition1 : 80 * x + 35 * y = 2250 :=
  sorry

def condition2 : x = y - 15 :=
  sorry

-- Part 1: Cost of one bottle of each disinfectant
theorem part1_solution : x = cost_A ∧ y = cost_B :=
  sorry

-- Additional conditions for part 2
variables (m : ℕ)
variables (total_bottles : ℕ := 50)
variables (budget : ℝ := 1200)

-- Conditions for part 2
def condition3 : m + (total_bottles - m) = total_bottles :=
  sorry

def condition4 : 15 * m + 30 * (total_bottles - m) ≤ budget :=
  sorry

-- Part 2: Minimum number of bottles of Class A disinfectant
theorem part2_solution : m ≥ 20 :=
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l1402_140269


namespace NUMINAMATH_GPT_even_function_f_l1402_140271

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 - x^2 else -(-x)^3 - (-x)^2

theorem even_function_f (x : ℝ) (h : ∀ x ≤ 0, f x = x^3 - x^2) :
  (∀ x, f x = f (-x)) ∧ (∀ x > 0, f x = -x^3 - x^2) :=
by
  sorry

end NUMINAMATH_GPT_even_function_f_l1402_140271


namespace NUMINAMATH_GPT_total_cost_of_tickets_l1402_140263

-- Conditions
def normal_price : ℝ := 50
def website_tickets_cost : ℝ := 2 * normal_price
def scalper_tickets_cost : ℝ := 2 * (2.4 * normal_price) - 10
def discounted_ticket_cost : ℝ := 0.6 * normal_price

-- Proof Statement
theorem total_cost_of_tickets :
  website_tickets_cost + scalper_tickets_cost + discounted_ticket_cost = 360 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_tickets_l1402_140263


namespace NUMINAMATH_GPT_algebraic_identity_l1402_140243

theorem algebraic_identity (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) :
    a^2 - b^2 = -8 := by
  sorry

end NUMINAMATH_GPT_algebraic_identity_l1402_140243


namespace NUMINAMATH_GPT_tank_third_dimension_l1402_140203

theorem tank_third_dimension (x : ℕ) (h1 : 4 * 5 = 20) (h2 : 2 * (4 * x) + 2 * (5 * x) = 18 * x) (h3 : (40 + 18 * x) * 20 = 1520) :
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_tank_third_dimension_l1402_140203


namespace NUMINAMATH_GPT_square_side_length_l1402_140216

theorem square_side_length (d s : ℝ) (h_diag : d = 2) (h_rel : d = s * Real.sqrt 2) : s = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_square_side_length_l1402_140216


namespace NUMINAMATH_GPT_eugene_initial_pencils_l1402_140227

theorem eugene_initial_pencils (P : ℕ) (h1 : P + 6 = 57) : P = 51 :=
by
  sorry

end NUMINAMATH_GPT_eugene_initial_pencils_l1402_140227


namespace NUMINAMATH_GPT_positive_solution_iff_abs_a_b_lt_one_l1402_140217

theorem positive_solution_iff_abs_a_b_lt_one
  (a b : ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 - x2 = a)
  (h2 : x3 - x4 = b)
  (h3 : x1 + x2 + x3 + x4 = 1)
  (h4 : x1 > 0)
  (h5 : x2 > 0)
  (h6 : x3 > 0)
  (h7 : x4 > 0) :
  |a| + |b| < 1 :=
sorry

end NUMINAMATH_GPT_positive_solution_iff_abs_a_b_lt_one_l1402_140217


namespace NUMINAMATH_GPT_lucille_paint_cans_needed_l1402_140257

theorem lucille_paint_cans_needed :
  let wall1_area := 3 * 2
  let wall2_area := 3 * 2
  let wall3_area := 5 * 2
  let wall4_area := 4 * 2
  let total_area := wall1_area + wall2_area + wall3_area + wall4_area
  let coverage_per_can := 2
  let cans_needed := total_area / coverage_per_can
  cans_needed = 15 := 
by 
  sorry

end NUMINAMATH_GPT_lucille_paint_cans_needed_l1402_140257


namespace NUMINAMATH_GPT_tan_shift_monotonic_interval_l1402_140208

noncomputable def monotonic_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | k * Real.pi - 3 * Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 4}

theorem tan_shift_monotonic_interval {k : ℤ} :
  ∀ x, (monotonic_interval k x) → (Real.tan (x + Real.pi / 4)) = (Real.tan x) := sorry

end NUMINAMATH_GPT_tan_shift_monotonic_interval_l1402_140208


namespace NUMINAMATH_GPT_original_number_of_turtles_l1402_140260

-- Define the problem
theorem original_number_of_turtles (T : ℕ) (h1 : 17 = (T + 3 * T - 2) / 2) : T = 9 := by
  sorry

end NUMINAMATH_GPT_original_number_of_turtles_l1402_140260


namespace NUMINAMATH_GPT_time_needed_to_gather_remaining_flowers_l1402_140221

-- conditions
def classmates : ℕ := 30
def time_per_flower : ℕ := 10
def gathering_time : ℕ := 2 * 60
def lost_flowers : ℕ := 3

-- question and proof goal
theorem time_needed_to_gather_remaining_flowers : 
  let flowers_needed := classmates - ((gathering_time / time_per_flower) - lost_flowers)
  flowers_needed * time_per_flower = 210 :=
by
  sorry

end NUMINAMATH_GPT_time_needed_to_gather_remaining_flowers_l1402_140221


namespace NUMINAMATH_GPT_prime_factor_of_difference_l1402_140237

theorem prime_factor_of_difference (A B C : ℕ) (hA : A ≠ 0) (hABC_digits : A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (hA_range : 0 ≤ A ∧ A ≤ 9) (hB_range : 0 ≤ B ∧ B ≤ 9) (hC_range : 0 ≤ C ∧ C ≤ 9) :
  11 ∣ (100 * A + 10 * B + C) - (100 * C + 10 * B + A) :=
by
  sorry

end NUMINAMATH_GPT_prime_factor_of_difference_l1402_140237


namespace NUMINAMATH_GPT_band_row_lengths_l1402_140234

theorem band_row_lengths (x y : ℕ) :
  (x * y = 90) → (5 ≤ x ∧ x ≤ 20) → (Even y) → False :=
by sorry

end NUMINAMATH_GPT_band_row_lengths_l1402_140234


namespace NUMINAMATH_GPT_arithmetic_mean_solution_l1402_140252

theorem arithmetic_mean_solution (x : ℚ) :
  (x + 10 + 20 + 3*x + 18 + 3*x + 6) / 5 = 30 → x = 96 / 7 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_arithmetic_mean_solution_l1402_140252


namespace NUMINAMATH_GPT_number_of_m_l1402_140272

theorem number_of_m (k : ℕ) : 
  (∀ m a b : ℤ, 
      (a ≠ 0 ∧ b ≠ 0) ∧ 
      (a + b = m) ∧ 
      (a * b = m + 2006) → k = 5) :=
sorry

end NUMINAMATH_GPT_number_of_m_l1402_140272


namespace NUMINAMATH_GPT_sum_lengths_DE_EF_equals_9_l1402_140235

variable (AB BC FA : ℝ)
variable (area_ABCDEF : ℝ)
variable (DE EF : ℝ)

theorem sum_lengths_DE_EF_equals_9 (h1 : area_ABCDEF = 52) (h2 : AB = 8) (h3 : BC = 9) (h4 : FA = 5)
  (h5 : AB * BC - area_ABCDEF = DE * EF) (h6 : BC - FA = DE) : DE + EF = 9 := 
by 
  sorry

end NUMINAMATH_GPT_sum_lengths_DE_EF_equals_9_l1402_140235


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l1402_140296

def point : ℝ × ℝ := (4, -3)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l1402_140296


namespace NUMINAMATH_GPT_sinA_value_triangle_area_l1402_140289

-- Definitions of the given variables
variables (A B C : ℝ)
variables (a b c : ℝ)
variables (sinA sinC cosC : ℝ)

-- Given conditions
axiom h_c : c = Real.sqrt 2
axiom h_a : a = 1
axiom h_cosC : cosC = 3 / 4
axiom h_sinC : sinC = Real.sqrt 7 / 4
axiom h_b : b = 2

-- Question 1: Prove sin A = sqrt 14 / 8
theorem sinA_value : sinA = Real.sqrt 14 / 8 :=
sorry

-- Question 2: Prove the area of triangle ABC is sqrt 7 / 4
theorem triangle_area : 1/2 * a * b * sinC = Real.sqrt 7 / 4 :=
sorry

end NUMINAMATH_GPT_sinA_value_triangle_area_l1402_140289


namespace NUMINAMATH_GPT_exists_h_not_divisible_l1402_140210

theorem exists_h_not_divisible : ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ % ⌊h * 1969^(n-1)⌋ = 0) :=
by
  sorry

end NUMINAMATH_GPT_exists_h_not_divisible_l1402_140210


namespace NUMINAMATH_GPT_area_quadrilateral_ABCDE_correct_l1402_140294

noncomputable def area_quadrilateral_ABCDE (AM NM AN BN BO OC CP CD EP DE : ℝ) : ℝ :=
  (0.5 * AM * NM * Real.sqrt 2) + (0.5 * BN * BO) + (0.5 * OC * CP * Real.sqrt 2) - (0.5 * DE * EP)

theorem area_quadrilateral_ABCDE_correct :
  ∀ (AM NM AN BN BO OC CP CD EP DE : ℝ),
    DE = 12 ∧ 
    AM = 36 ∧ 
    NM = 36 ∧ 
    AN = 36 * Real.sqrt 2 ∧
    BN = 36 * Real.sqrt 2 - 36 ∧
    BO = 36 ∧
    OC = 36 ∧
    CP = 36 * Real.sqrt 2 ∧
    CD = 24 ∧
    EP = 24
    → area_quadrilateral_ABCDE AM NM AN BN BO OC CP CD EP DE = 2311.2 * Real.sqrt 2 + 504 :=
by intro AM NM AN BN BO OC CP CD EP DE h;
   cases h;
   sorry

end NUMINAMATH_GPT_area_quadrilateral_ABCDE_correct_l1402_140294


namespace NUMINAMATH_GPT_extreme_value_f_g_gt_one_l1402_140240

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.exp x * (a * x + x * Real.cos x + 1)

theorem extreme_value_f : f 0 = 0 :=
by
  sorry

theorem g_gt_one (a : ℝ) (h : a > -1) (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : g x a > 1 :=
by
  sorry

end NUMINAMATH_GPT_extreme_value_f_g_gt_one_l1402_140240


namespace NUMINAMATH_GPT_root_in_interval_l1402_140297

def polynomial (x : ℝ) := x^3 + 3 * x^2 - x + 1

noncomputable def A : ℤ := -4
noncomputable def B : ℤ := -3

theorem root_in_interval : (∃ x : ℝ, polynomial x = 0 ∧ (A : ℝ) < x ∧ x < (B : ℝ)) :=
sorry

end NUMINAMATH_GPT_root_in_interval_l1402_140297


namespace NUMINAMATH_GPT_gcd_times_xyz_is_square_l1402_140266

theorem gcd_times_xyz_is_square (x y z : ℕ) (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) : 
  ∃ k : ℕ, (Nat.gcd x (Nat.gcd y z) * x * y * z) = k ^ 2 :=
sorry

end NUMINAMATH_GPT_gcd_times_xyz_is_square_l1402_140266


namespace NUMINAMATH_GPT_cannot_form_optionE_l1402_140283

-- Define the 4x4 tile
structure Tile4x4 :=
(matrix : Fin 4 → Fin 4 → Bool) -- Boolean to represent black or white

-- Define the condition of alternating rows and columns
def alternating_pattern (tile : Tile4x4) : Prop :=
  (∀ i, tile.matrix i 0 ≠ tile.matrix i 1 ∧
         tile.matrix i 2 ≠ tile.matrix i 3) ∧
  (∀ j, tile.matrix 0 j ≠ tile.matrix 1 j ∧
         tile.matrix 2 j ≠ tile.matrix 3 j)

-- Example tiles for options A, B, C, D, E
def optionA : Tile4x4 := sorry
def optionB : Tile4x4 := sorry
def optionC : Tile4x4 := sorry
def optionD : Tile4x4 := sorry
def optionE : Tile4x4 := sorry

-- Given pieces that can form a 4x4 alternating tile
axiom given_piece1 : Tile4x4
axiom given_piece2 : Tile4x4

-- Combining given pieces to form a 4x4 tile
def combine_pieces (p1 p2 : Tile4x4) : Tile4x4 := sorry -- Combination logic here

-- Proposition stating the problem
theorem cannot_form_optionE :
  (∀ tile, tile = optionA ∨ tile = optionB ∨ tile = optionC ∨ tile = optionD ∨ tile = optionE →
    (tile = optionA ∨ tile = optionB ∨ tile = optionC ∨ tile = optionD → alternating_pattern tile) ∧
    tile = optionE → ¬alternating_pattern tile) :=
sorry

end NUMINAMATH_GPT_cannot_form_optionE_l1402_140283


namespace NUMINAMATH_GPT_triangle_ABC_area_l1402_140286

-- definition of points A, B, and C
def A : (ℝ × ℝ) := (0, 2)
def B : (ℝ × ℝ) := (6, 0)
def C : (ℝ × ℝ) := (3, 7)

-- helper function to calculate area of triangle given vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_ABC_area :
  triangle_area A B C = 18 := by
  sorry

end NUMINAMATH_GPT_triangle_ABC_area_l1402_140286


namespace NUMINAMATH_GPT_average_computer_time_per_person_is_95_l1402_140223

def people : ℕ := 8
def computers : ℕ := 5
def work_time : ℕ := 152 -- total working day minutes

def total_computer_time : ℕ := work_time * computers
def average_time_per_person : ℕ := total_computer_time / people

theorem average_computer_time_per_person_is_95 :
  average_time_per_person = 95 := 
by
  sorry

end NUMINAMATH_GPT_average_computer_time_per_person_is_95_l1402_140223


namespace NUMINAMATH_GPT_f_not_surjective_l1402_140281

def f : ℝ → ℕ → Prop := sorry

theorem f_not_surjective (f : ℝ → ℕ) 
  (h : ∀ x y : ℝ, f (x + (1 / f y)) = f (y + (1 / f x))) : 
  ¬ (∀ n : ℕ, ∃ x : ℝ, f x = n) :=
sorry

end NUMINAMATH_GPT_f_not_surjective_l1402_140281


namespace NUMINAMATH_GPT_lucas_total_pages_l1402_140242

-- Define the variables and conditions
def lucas_read_pages : Nat :=
  let pages_first_four_days := 4 * 20
  let pages_break_day := 0
  let pages_next_four_days := 4 * 30
  let pages_last_day := 15
  pages_first_four_days + pages_break_day + pages_next_four_days + pages_last_day

-- State the theorem
theorem lucas_total_pages :
  lucas_read_pages = 215 :=
sorry

end NUMINAMATH_GPT_lucas_total_pages_l1402_140242


namespace NUMINAMATH_GPT_alcohol_mix_problem_l1402_140236

theorem alcohol_mix_problem
  (x_volume : ℕ) (y_volume : ℕ)
  (x_percentage : ℝ) (y_percentage : ℝ)
  (target_percentage : ℝ)
  (x_volume_eq : x_volume = 200)
  (x_percentage_eq : x_percentage = 0.10)
  (y_percentage_eq : y_percentage = 0.30)
  (target_percentage_eq : target_percentage = 0.14)
  (y_solution : ℝ)
  (h : y_volume = 50) :
  (20 + 0.3 * y_solution) / (200 + y_solution) = target_percentage := by sorry

end NUMINAMATH_GPT_alcohol_mix_problem_l1402_140236


namespace NUMINAMATH_GPT_no_real_roots_quadratic_l1402_140225

theorem no_real_roots_quadratic (a b c : ℝ) (h : a = 1 ∧ b = -4 ∧ c = 8) :
    (a ≠ 0) → (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_quadratic_l1402_140225


namespace NUMINAMATH_GPT_brick_height_correct_l1402_140274

-- Definitions
def wall_length : ℝ := 8
def wall_height : ℝ := 6
def wall_thickness : ℝ := 0.02 -- converted from 2 cm to meters
def brick_length : ℝ := 0.05 -- converted from 5 cm to meters
def brick_width : ℝ := 0.11 -- converted from 11 cm to meters
def brick_height : ℝ := 0.06 -- converted from 6 cm to meters
def number_of_bricks : ℝ := 2909.090909090909

-- Statement to prove
theorem brick_height_correct : brick_height = 0.06 := by
  sorry

end NUMINAMATH_GPT_brick_height_correct_l1402_140274


namespace NUMINAMATH_GPT_anne_total_bottle_caps_l1402_140254

/-- 
Anne initially has 10 bottle caps 
and then finds another 5 bottle caps.
-/
def anne_initial_bottle_caps : ℕ := 10
def anne_found_bottle_caps : ℕ := 5

/-- 
Prove that the total number of bottle caps
Anne ends with is equal to 15.
-/
theorem anne_total_bottle_caps : 
  anne_initial_bottle_caps + anne_found_bottle_caps = 15 :=
by 
  sorry

end NUMINAMATH_GPT_anne_total_bottle_caps_l1402_140254


namespace NUMINAMATH_GPT_initial_skittles_geq_16_l1402_140299

variable (S : ℕ) -- S represents the total number of Skittles Lillian had initially
variable (L : ℕ) -- L represents the number of Skittles Lillian kept as leftovers

theorem initial_skittles_geq_16 (h1 : S = 8 * 2 + L) : S ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_initial_skittles_geq_16_l1402_140299


namespace NUMINAMATH_GPT_pipe_B_fill_time_l1402_140280

theorem pipe_B_fill_time
  (rate_A : ℝ)
  (rate_B : ℝ)
  (t : ℝ)
  (h_rate_A : rate_A = 2 / 75)
  (h_rate_B : rate_B = 1 / t)
  (h_fill_total : 9 * (rate_A + rate_B) + 21 * rate_A = 1) :
  t = 45 := 
sorry

end NUMINAMATH_GPT_pipe_B_fill_time_l1402_140280


namespace NUMINAMATH_GPT_vector_parallel_solution_l1402_140245

theorem vector_parallel_solution (x : ℝ) : 
  let a := (2, 3)
  let b := (x, -9)
  (a.snd = 3) → (a.fst = 2) → (b.snd = -9) → (a.fst * b.snd = a.snd * (b.fst)) → x = -6 := 
by
  intros 
  sorry

end NUMINAMATH_GPT_vector_parallel_solution_l1402_140245


namespace NUMINAMATH_GPT_geometric_sequence_sum_of_first_four_terms_l1402_140259

theorem geometric_sequence_sum_of_first_four_terms (a r : ℝ) 
  (h1 : a + a * r = 7) 
  (h2 : a * (1 + r + r^2 + r^3 + r^4 + r^5) = 91) : 
  a * (1 + r + r^2 + r^3) = 32 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_of_first_four_terms_l1402_140259


namespace NUMINAMATH_GPT_monthly_income_of_labourer_l1402_140264

variable (I : ℕ) -- Monthly income

-- Conditions: 
def condition1 := (85 * 6) - (6 * I) -- A boolean expression depicting the labourer fell into debt
def condition2 := (60 * 4) + (85 * 6 - 6 * I) + 30 -- Total income covers debt and saving 30

-- Statement to be proven
theorem monthly_income_of_labourer : 
  ∃ I : ℕ, condition1 I = 0 ∧ condition2 I = 4 * I → I = 78 :=
by
  sorry

end NUMINAMATH_GPT_monthly_income_of_labourer_l1402_140264


namespace NUMINAMATH_GPT_daria_still_owes_l1402_140276

-- Definitions of the given conditions
def saved_amount : ℝ := 500
def couch_cost : ℝ := 750
def table_cost : ℝ := 100
def lamp_cost : ℝ := 50

-- Calculation of total cost of the furniture
def total_cost : ℝ := couch_cost + table_cost + lamp_cost

-- Calculation of the remaining amount owed
def remaining_owed : ℝ := total_cost - saved_amount

-- Proof statement that Daria still owes $400 before interest
theorem daria_still_owes : remaining_owed = 400 := by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_daria_still_owes_l1402_140276


namespace NUMINAMATH_GPT_exists_x0_f_leq_one_tenth_l1402_140207

noncomputable def f (x a : ℝ) : ℝ := x^2 + (Real.log (3*x))^2 - 2*a*x - 6*a*(Real.log (3*x)) + 10*a^2

theorem exists_x0_f_leq_one_tenth (a : ℝ) : (∃ x₀, f x₀ a ≤ 1/10) ↔ a = 1/30 := by
  sorry

end NUMINAMATH_GPT_exists_x0_f_leq_one_tenth_l1402_140207


namespace NUMINAMATH_GPT_bottom_level_legos_l1402_140205

theorem bottom_level_legos
  (x : ℕ)
  (h : x^2 + (x - 1)^2 + (x - 2)^2 = 110) :
  x = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_bottom_level_legos_l1402_140205


namespace NUMINAMATH_GPT_rate_percent_l1402_140231

theorem rate_percent (SI P T: ℝ) (h₁: SI = 250) (h₂: P = 1500) (h₃: T = 5) : 
  ∃ R : ℝ, R = (SI * 100) / (P * T) := 
by
  use (250 * 100) / (1500 * 5)
  sorry

end NUMINAMATH_GPT_rate_percent_l1402_140231


namespace NUMINAMATH_GPT_ellie_runs_8_miles_in_24_minutes_l1402_140202

theorem ellie_runs_8_miles_in_24_minutes (time_max : ℝ) (distance_max : ℝ) 
  (time_ellie_fraction : ℝ) (distance_ellie : ℝ) (distance_ellie_final : ℝ)
  (h1 : distance_max = 6) 
  (h2 : time_max = 36) 
  (h3 : time_ellie_fraction = 1/3) 
  (h4 : distance_ellie = 4) 
  (h5 : distance_ellie_final = 8) :
  ((time_ellie_fraction * time_max) / distance_ellie) * distance_ellie_final = 24 :=
by
  sorry

end NUMINAMATH_GPT_ellie_runs_8_miles_in_24_minutes_l1402_140202


namespace NUMINAMATH_GPT_lia_quadrilateral_rod_count_l1402_140230

theorem lia_quadrilateral_rod_count :
  let rods := {n : ℕ | 1 ≤ n ∧ n ≤ 40}
  let selected_rods := {5, 10, 20}
  let remaining_rods := rods \ selected_rods
  rod_count = 26 ∧ (∃ d ∈ remaining_rods, 
    (5 + 10 + 20) > d ∧ (10 + 20 + d) > 5 ∧ (5 + 20 + d) > 10 ∧ (5 + 10 + d) > 20)
:=
sorry

end NUMINAMATH_GPT_lia_quadrilateral_rod_count_l1402_140230


namespace NUMINAMATH_GPT_toy_cost_price_l1402_140239

theorem toy_cost_price (C : ℕ) (h : 18 * C + 3 * C = 25200) : C = 1200 := by
  -- The proof is not required
  sorry

end NUMINAMATH_GPT_toy_cost_price_l1402_140239


namespace NUMINAMATH_GPT_intersect_at_one_point_l1402_140228

-- Definitions of points and circles
variable (Point : Type)
variable (Circle : Type)
variable (A : Point)
variable (C1 C2 C3 C4 : Circle)

-- Definition of intersection points
variable (B12 B13 B14 B23 B24 B34 : Point)

-- Note: Assumptions around the geometry structure axioms need to be defined
-- Assuming we have a function that checks if three points are collinear:
variable (are_collinear : Point → Point → Point → Prop)
-- Assuming we have a function that checks if a point is part of a circle:
variable (on_circle : Point → Circle → Prop)

-- Axioms related to the conditions
axiom collinear_B12_B34_B (hC1 : on_circle B12 C1) (hC2 : on_circle B12 C2) (hC3 : on_circle B34 C3) (hC4 : on_circle B34 C4) : 
  ∃ P : Point, are_collinear B12 P B34 

axiom collinear_B13_B24_B (hC1 : on_circle B13 C1) (hC2 : on_circle B13 C3) (hC3 : on_circle B24 C2) (hC4 : on_circle B24 C4) : 
  ∃ P : Point, are_collinear B13 P B24 

axiom collinear_B14_B23_B (hC1 : on_circle B14 C1) (hC2 : on_circle B14 C4) (hC3 : on_circle B23 C2) (hC4 : on_circle B23 C3) : 
  ∃ P : Point, are_collinear B14 P B23 

-- The theorem to be proved
theorem intersect_at_one_point :
  ∃ P : Point, 
    are_collinear B12 P B34 ∧ are_collinear B13 P B24 ∧ are_collinear B14 P B23 := 
sorry

end NUMINAMATH_GPT_intersect_at_one_point_l1402_140228


namespace NUMINAMATH_GPT_batsman_avg_l1402_140220

variable (A : ℕ) -- The batting average in 46 innings

-- Given conditions
variables (highest lowest : ℕ)
variables (diff : ℕ) (avg_excl : ℕ) (num_excl : ℕ)

namespace cricket

-- Define the given values
def highest_score := 225
def difference := 150
def avg_excluding := 58
def num_excluding := 44

-- Calculate the lowest score
def lowest_score := highest_score - difference

-- Calculate the total runs in 44 innings excluding highest and lowest scores
def total_run_excluded := avg_excluding * num_excluding

-- Calculate the total runs in 46 innings
def total_runs := total_run_excluded + highest_score + lowest_score

-- Define the equation relating the average to everything else
def batting_avg_eq : Prop :=
  total_runs = 46 * A

-- Prove that the batting average A is 62 given the conditions
theorem batsman_avg :
  A = 62 :=
  by
    sorry

end cricket

end NUMINAMATH_GPT_batsman_avg_l1402_140220


namespace NUMINAMATH_GPT_initial_number_of_persons_l1402_140275

theorem initial_number_of_persons (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (weight_diff : ℝ)
  (h1 : avg_increase = 2.5) 
  (h2 : old_weight = 75) 
  (h3 : new_weight = 95)
  (h4 : weight_diff = new_weight - old_weight)
  (h5 : weight_diff = avg_increase * n) : n = 8 := 
sorry

end NUMINAMATH_GPT_initial_number_of_persons_l1402_140275


namespace NUMINAMATH_GPT_rahul_matches_l1402_140295

theorem rahul_matches
  (initial_avg : ℕ)
  (runs_today : ℕ)
  (final_avg : ℕ)
  (n : ℕ)
  (H1 : initial_avg = 50)
  (H2 : runs_today = 78)
  (H3 : final_avg = 54)
  (H4 : (initial_avg * n + runs_today) = final_avg * (n + 1)) :
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_rahul_matches_l1402_140295


namespace NUMINAMATH_GPT_condition_iff_absolute_value_l1402_140288

theorem condition_iff_absolute_value (a b : ℝ) : (a > b) ↔ (a * |a| > b * |b|) :=
sorry

end NUMINAMATH_GPT_condition_iff_absolute_value_l1402_140288


namespace NUMINAMATH_GPT_dice_surface_dots_l1402_140226

def total_dots_on_die := 1 + 2 + 3 + 4 + 5 + 6

def total_dots_on_seven_dice := 7 * total_dots_on_die

def hidden_dots_on_central_die := total_dots_on_die

def visible_dots_on_surface := total_dots_on_seven_dice - hidden_dots_on_central_die

theorem dice_surface_dots : visible_dots_on_surface = 105 := by
  sorry

end NUMINAMATH_GPT_dice_surface_dots_l1402_140226


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1402_140278

noncomputable def a : ℝ := (0.8 : ℝ)^(5.2 : ℝ)
noncomputable def b : ℝ := (0.8 : ℝ)^(5.5 : ℝ)
noncomputable def c : ℝ := (5.2 : ℝ)^(0.1 : ℝ)

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1402_140278


namespace NUMINAMATH_GPT_value_of_m_l1402_140284

theorem value_of_m (m : ℝ) : (3 = 2 * m + 1) → m = 1 :=
by
  intro h
  -- skipped proof due to requirement
  sorry

end NUMINAMATH_GPT_value_of_m_l1402_140284


namespace NUMINAMATH_GPT_incorrect_conclusion_C_l1402_140258

noncomputable def f (x : ℝ) := (x - 1)^2 * Real.exp x

theorem incorrect_conclusion_C : 
  ¬(∀ x, ∀ ε > 0, ∃ δ > 0, ∀ y, abs (y - x) < δ → abs (f y - f x) ≥ ε) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_conclusion_C_l1402_140258


namespace NUMINAMATH_GPT_fencing_rate_correct_l1402_140233

noncomputable def rate_per_meter (d : ℝ) (cost : ℝ) : ℝ :=
  cost / (Real.pi * d)

theorem fencing_rate_correct : rate_per_meter 26 122.52211349000194 = 1.5 := by
  sorry

end NUMINAMATH_GPT_fencing_rate_correct_l1402_140233


namespace NUMINAMATH_GPT_race_car_cost_l1402_140292

variable (R : ℝ)
variable (Mater_cost SallyMcQueen_cost : ℝ)

-- Conditions
def Mater_cost_def : Mater_cost = 0.10 * R := by sorry
def SallyMcQueen_cost_def : SallyMcQueen_cost = 3 * Mater_cost := by sorry
def SallyMcQueen_cost_val : SallyMcQueen_cost = 42000 := by sorry

-- Theorem to prove the race car cost
theorem race_car_cost : R = 140000 :=
  by
    -- Use the conditions to prove
    sorry

end NUMINAMATH_GPT_race_car_cost_l1402_140292


namespace NUMINAMATH_GPT_trailing_zeros_in_15_factorial_base_15_are_3_l1402_140282

/--
Compute the number of trailing zeros in \( 15! \) when expressed in base 15.
-/
def compute_trailing_zeros_in_factorial_base_15 : ℕ :=
  let num_factors_3 := (15 / 3) + (15 / 9)
  let num_factors_5 := (15 / 5)
  min num_factors_3 num_factors_5

theorem trailing_zeros_in_15_factorial_base_15_are_3 :
  compute_trailing_zeros_in_factorial_base_15 = 3 :=
sorry

end NUMINAMATH_GPT_trailing_zeros_in_15_factorial_base_15_are_3_l1402_140282


namespace NUMINAMATH_GPT_floor_sqrt_80_eq_8_l1402_140291

theorem floor_sqrt_80_eq_8
  (h1 : 8^2 = 64)
  (h2 : 9^2 = 81)
  (h3 : 64 < 80 ∧ 80 < 81)
  (h4 : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) : 
  Int.floor (Real.sqrt 80) = 8 := by
  sorry

end NUMINAMATH_GPT_floor_sqrt_80_eq_8_l1402_140291


namespace NUMINAMATH_GPT_solve_system_of_equations_l1402_140290

theorem solve_system_of_equations :
  ∃ (x y : ℕ), (x + 2 * y = 5) ∧ (3 * x + y = 5) ∧ (x = 1) ∧ (y = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_system_of_equations_l1402_140290


namespace NUMINAMATH_GPT_Elise_paid_23_dollars_l1402_140253

-- Definitions and conditions
def base_price := 3
def cost_per_mile := 4
def distance := 5

-- Desired conclusion (total cost)
def total_cost := base_price + cost_per_mile * distance

-- Theorem statement
theorem Elise_paid_23_dollars : total_cost = 23 := by
  sorry

end NUMINAMATH_GPT_Elise_paid_23_dollars_l1402_140253


namespace NUMINAMATH_GPT_min_x_value_l1402_140218

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 18 * x + 50 * y + 56

theorem min_x_value : 
  ∃ (x : ℝ), ∃ (y : ℝ), circle_eq x y ∧ x = 9 - Real.sqrt 762 :=
by
  sorry

end NUMINAMATH_GPT_min_x_value_l1402_140218


namespace NUMINAMATH_GPT_number_of_solutions_l1402_140262

-- Define the relevant trigonometric equation
def trig_equation (x : ℝ) : Prop := (Real.cos x)^2 + 3 * (Real.sin x)^2 = 1

-- Define the range for x
def in_range (x : ℝ) : Prop := -20 < x ∧ x < 100

-- Define the predicate that x satisfies both the trig equation and the range condition
def satisfies_conditions (x : ℝ) : Prop := trig_equation x ∧ in_range x

-- The final theorem statement (proof is omitted)
theorem number_of_solutions : 
  ∃ (count : ℕ), count = 38 ∧ ∀ (x : ℝ), satisfies_conditions x ↔ x = k * Real.pi ∧ -20 < k * Real.pi ∧ k * Real.pi < 100 := sorry

end NUMINAMATH_GPT_number_of_solutions_l1402_140262
