import Mathlib

namespace desired_interest_rate_l2212_221270

def face_value : Real := 52
def dividend_rate : Real := 0.09
def market_value : Real := 39

theorem desired_interest_rate : (dividend_rate * face_value / market_value) * 100 = 12 := by
  sorry

end desired_interest_rate_l2212_221270


namespace quadratic_factorization_l2212_221271

theorem quadratic_factorization :
  ∃ a b : ℕ, (a > b) ∧ (x^2 - 20 * x + 96 = (x - a) * (x - b)) ∧ (4 * b - a = 20) := sorry

end quadratic_factorization_l2212_221271


namespace trigonometric_identity_l2212_221267

theorem trigonometric_identity (α : ℝ) (h : Real.sin (3 * Real.pi - α) = 2 * Real.sin (Real.pi / 2 + α)) : 
  (Real.sin (Real.pi - α) ^ 3 - Real.sin (Real.pi / 2 - α)) / 
  (3 * Real.cos (Real.pi / 2 + α) + 2 * Real.cos (Real.pi + α)) = -3/40 :=
by
  sorry

end trigonometric_identity_l2212_221267


namespace evaluate_f_at_2_l2212_221269

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem evaluate_f_at_2 :
  f 2 = 125 :=
by
  sorry

end evaluate_f_at_2_l2212_221269


namespace reversed_number_increase_l2212_221204

theorem reversed_number_increase (a b c : ℕ) 
  (h1 : a + b + c = 10) 
  (h2 : b = a + c)
  (h3 : a = 2 ∧ b = 5 ∧ c = 3) :
  (c * 100 + b * 10 + a) - (a * 100 + b * 10 + c) = 99 :=
by
  sorry

end reversed_number_increase_l2212_221204


namespace speed_of_truck_l2212_221257

theorem speed_of_truck
  (v : ℝ)                         -- Let \( v \) be the speed of the truck.
  (car_speed : ℝ := 55)           -- Car speed is 55 mph.
  (start_delay : ℝ := 1)          -- Truck starts 1 hour later.
  (catchup_time : ℝ := 6.5)       -- Truck takes 6.5 hours to pass the car.
  (additional_distance_car : ℝ := car_speed * catchup_time)  -- Additional distance covered by the car in 6.5 hours.
  (total_distance_truck : ℝ := car_speed * start_delay + additional_distance_car)  -- Total distance truck must cover to pass the car.
  (truck_distance_eq : v * catchup_time = total_distance_truck)  -- Distance equation for the truck.
  : v = 63.46 :=                -- Prove the truck's speed is 63.46 mph.
by
  -- Original problem solution confirms truck's speed as 63.46 mph. 
  sorry

end speed_of_truck_l2212_221257


namespace negation_equiv_l2212_221243

theorem negation_equiv (x : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) := 
by 
  sorry

end negation_equiv_l2212_221243


namespace Karlee_initial_grapes_l2212_221227

theorem Karlee_initial_grapes (G S Remaining_Fruits : ℕ)
  (h1 : S = (3 * G) / 5)
  (h2 : Remaining_Fruits = 96)
  (h3 : Remaining_Fruits = (3 * G) / 5 + (9 * G) / 25) :
  G = 100 := by
  -- add proof here
  sorry

end Karlee_initial_grapes_l2212_221227


namespace number_of_diagonals_is_correct_sum_of_interior_angles_is_correct_l2212_221263

-- Definition for the number of sides in the polygon
def n : ℕ := 150

-- Definition of the formula for the number of diagonals
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Definition of the formula for the sum of interior angles
def sum_of_interior_angles (n : ℕ) : ℕ :=
  180 * (n - 2)

-- Theorem statements to be proved
theorem number_of_diagonals_is_correct : number_of_diagonals n = 11025 := sorry

theorem sum_of_interior_angles_is_correct : sum_of_interior_angles n = 26640 := sorry

end number_of_diagonals_is_correct_sum_of_interior_angles_is_correct_l2212_221263


namespace no_non_trivial_solution_l2212_221289

theorem no_non_trivial_solution (a b c : ℤ) (h : a^2 = 2 * b^2 + 3 * c^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end no_non_trivial_solution_l2212_221289


namespace real_number_unique_l2212_221221

variable (a x : ℝ)

theorem real_number_unique (h1 : (a + 3) * (a + 3) = x)
  (h2 : (2 * a - 9) * (2 * a - 9) = x) : x = 25 := by
  sorry

end real_number_unique_l2212_221221


namespace pigeons_in_house_l2212_221214

variable (x F c : ℝ)

theorem pigeons_in_house 
  (H1 : F = (x - 75) * 20 * c)
  (H2 : F = (x + 100) * 15 * c) :
  x = 600 := by
  sorry

end pigeons_in_house_l2212_221214


namespace cube_dihedral_angle_is_60_degrees_l2212_221295

-- Define the cube and related geometrical features
structure Point := (x y z : ℝ)
structure Cube :=
  (A B C D A₁ B₁ C₁ D₁ : Point)
  (is_cube : true) -- Placeholder for cube properties

-- Define the function to calculate dihedral angle measure
noncomputable def dihedral_angle_measure (cube: Cube) : ℝ := sorry

-- The theorem statement
theorem cube_dihedral_angle_is_60_degrees (cube : Cube) : dihedral_angle_measure cube = 60 :=
by sorry

end cube_dihedral_angle_is_60_degrees_l2212_221295


namespace weights_are_equal_l2212_221265

variable {n : ℕ}
variables {a : Fin (2 * n + 1) → ℝ}

def weights_condition
    (a : Fin (2 * n + 1) → ℝ) : Prop :=
  ∀ i : Fin (2 * n + 1), ∃ (A B : Finset (Fin (2 * n + 1))),
    A.card = n ∧ B.card = n ∧ A ∩ B = ∅ ∧
    A ∪ B = Finset.univ.erase i ∧
    (A.sum a = B.sum a)

theorem weights_are_equal
    (h : weights_condition a) :
  ∃ k : ℝ, ∀ i : Fin (2 * n + 1), a i = k :=
  sorry

end weights_are_equal_l2212_221265


namespace initial_extra_planks_l2212_221231

-- Definitions corresponding to the conditions
def charlie_planks : Nat := 10
def father_planks : Nat := 10
def total_planks : Nat := 35

-- The proof problem statement
theorem initial_extra_planks : total_planks - (charlie_planks + father_planks) = 15 := by
  sorry

end initial_extra_planks_l2212_221231


namespace darwin_final_money_l2212_221277

def initial_amount : ℕ := 600
def spent_on_gas (initial : ℕ) : ℕ := initial * 1 / 3
def remaining_after_gas (initial spent_gas : ℕ) : ℕ := initial - spent_gas
def spent_on_food (remaining : ℕ) : ℕ := remaining * 1 / 4
def final_amount (remaining spent_food : ℕ) : ℕ := remaining - spent_food

theorem darwin_final_money :
  final_amount (remaining_after_gas initial_amount (spent_on_gas initial_amount)) (spent_on_food (remaining_after_gas initial_amount (spent_on_gas initial_amount))) = 300 :=
by
  sorry

end darwin_final_money_l2212_221277


namespace carol_total_peanuts_l2212_221275

open Nat

-- Define the conditions
def peanuts_from_tree : Nat := 48
def peanuts_from_ground : Nat := 178
def bags_of_peanuts : Nat := 3
def peanuts_per_bag : Nat := 250

-- Define the total number of peanuts Carol has to prove it equals 976
def total_peanuts (peanuts_from_tree peanuts_from_ground bags_of_peanuts peanuts_per_bag : Nat) : Nat :=
  peanuts_from_tree + peanuts_from_ground + (bags_of_peanuts * peanuts_per_bag)

theorem carol_total_peanuts : total_peanuts peanuts_from_tree peanuts_from_ground bags_of_peanuts peanuts_per_bag = 976 :=
  by
    -- proof goes here
    sorry

end carol_total_peanuts_l2212_221275


namespace man_speed_l2212_221285

theorem man_speed (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ) (speed : ℝ) 
  (h1 : distance = 12)
  (h2 : time_minutes = 72)
  (h3 : time_hours = time_minutes / 60)
  (h4 : speed = distance / time_hours) : speed = 10 :=
by
  sorry

end man_speed_l2212_221285


namespace evaluate_expression_l2212_221279

theorem evaluate_expression :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
sorry

end evaluate_expression_l2212_221279


namespace solve_equation_l2212_221211

noncomputable def equation (x : ℝ) : Prop :=
  2021 * x = 2022 * x ^ (2021 / 2022) - 1

theorem solve_equation : ∀ x : ℝ, equation x ↔ x = 1 :=
by
  intro x
  sorry

end solve_equation_l2212_221211


namespace find_number_l2212_221276

theorem find_number (x : ℝ) : 4 * x - 23 = 33 → x = 14 :=
by
  intros h
  sorry

end find_number_l2212_221276


namespace min_max_ab_bc_cd_de_l2212_221245

theorem min_max_ab_bc_cd_de (a b c d e : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e) (h_sum : a + b + c + d + e = 2018) : 
  ∃ a b c d e, 
  a > 0 ∧ 
  b > 0 ∧ 
  c > 0 ∧ 
  d > 0 ∧ 
  e > 0 ∧ 
  a + b + c + d + e = 2018 ∧ 
  ∀ M, M = max (max (max (a + b) (b + c)) (max (c + d) (d + e))) ↔ M = 673  :=
sorry

end min_max_ab_bc_cd_de_l2212_221245


namespace max_f_when_a_minus_1_range_of_a_l2212_221286

noncomputable section

-- Definitions of the functions given in the problem
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := x * f a x
def h (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - (2 * a - 1) * x + (a - 1)

-- Statement (1): Proving the maximum value of f(x) when a = -1
theorem max_f_when_a_minus_1 : 
  (∀ x : ℝ, f (-1) x ≤ f (-1) 1) :=
sorry

-- Statement (2): Proving the range of a when g(x) ≤ h(x) for x ≥ 1
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → g a x ≤ h a x) → (1 ≤ a) :=
sorry

end max_f_when_a_minus_1_range_of_a_l2212_221286


namespace volume_inside_sphere_outside_cylinder_l2212_221294

noncomputable def volumeDifference (r_cylinder base_radius_sphere : ℝ) :=
  let height := 4 * Real.sqrt 5
  let V_sphere := (4/3) * Real.pi * base_radius_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * height
  V_sphere - V_cylinder

theorem volume_inside_sphere_outside_cylinder
  (base_radius_sphere r_cylinder : ℝ) (h_base_radius_sphere : base_radius_sphere = 6) (h_r_cylinder : r_cylinder = 4) :
  volumeDifference r_cylinder base_radius_sphere = (288 - 64 * Real.sqrt 5) * Real.pi := by
  sorry

end volume_inside_sphere_outside_cylinder_l2212_221294


namespace lcm_first_ten_l2212_221274

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l2212_221274


namespace total_cost_is_correct_l2212_221208

def goldfish_price := 3
def goldfish_quantity := 15
def blue_fish_price := 6
def blue_fish_quantity := 7
def neon_tetra_price := 2
def neon_tetra_quantity := 10
def angelfish_price := 8
def angelfish_quantity := 5

def total_cost := goldfish_quantity * goldfish_price 
                 + blue_fish_quantity * blue_fish_price 
                 + neon_tetra_quantity * neon_tetra_price 
                 + angelfish_quantity * angelfish_price

theorem total_cost_is_correct : total_cost = 147 :=
by
  -- Summary of the proof steps goes here
  sorry

end total_cost_is_correct_l2212_221208


namespace gcd_ab_a2b2_l2212_221235

theorem gcd_ab_a2b2 (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_coprime : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by
  sorry

end gcd_ab_a2b2_l2212_221235


namespace selling_price_l2212_221236

-- Definitions for conditions
variables (CP SP_loss SP_profit : ℝ)
variable (h1 : SP_loss = 0.8 * CP)
variable (h2 : SP_profit = 1.05 * CP)
variable (h3 : SP_profit = 11.8125)

-- Theorem statement to prove
theorem selling_price (h1 : SP_loss = 0.8 * CP) (h2 : SP_profit = 1.05 * CP) (h3 : SP_profit = 11.8125) :
  SP_loss = 9 := 
sorry

end selling_price_l2212_221236


namespace geometric_series_inequality_l2212_221248

variables {x y : ℝ}

theorem geometric_series_inequality 
  (hx : |x| < 1) 
  (hy : |y| < 1) :
  (1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x * y)) :=
sorry

end geometric_series_inequality_l2212_221248


namespace sequence_fill_l2212_221246

theorem sequence_fill (x2 x3 x4 x5 x6 x7: ℕ) : 
  (20 + x2 + x3 = 100) ∧ 
  (x2 + x3 + x4 = 100) ∧ 
  (x3 + x4 + x5 = 100) ∧ 
  (x4 + x5 + x6 = 100) ∧ 
  (x5 + x6 + 16 = 100) →
  [20, x2, x3, x4, x5, x6, 16] = [20, 16, 64, 20, 16, 64, 20, 16] :=
by
  sorry

end sequence_fill_l2212_221246


namespace sequence_term_1000_l2212_221266

open Nat

theorem sequence_term_1000 :
  (∃ b : ℕ → ℤ,
    b 1 = 3010 ∧
    b 2 = 3011 ∧
    (∀ n, 1 ≤ n → b n + b (n + 1) + b (n + 2) = n + 4) ∧
    b 1000 = 3343) :=
sorry

end sequence_term_1000_l2212_221266


namespace sum_abcd_l2212_221282

variables (a b c d : ℚ)

theorem sum_abcd :
  3 * a + 4 * b + 6 * c + 8 * d = 48 →
  4 * (d + c) = b →
  4 * b + 2 * c = a →
  c + 1 = d →
  a + b + c + d = 513 / 37 :=
by
sorry

end sum_abcd_l2212_221282


namespace prescribedDosageLessThanTypical_l2212_221240

noncomputable def prescribedDosage : ℝ := 12
noncomputable def bodyWeight : ℝ := 120
noncomputable def typicalDosagePer15Pounds : ℝ := 2
noncomputable def typicalDosage : ℝ := (bodyWeight / 15) * typicalDosagePer15Pounds
noncomputable def percentageDecrease : ℝ := ((typicalDosage - prescribedDosage) / typicalDosage) * 100

theorem prescribedDosageLessThanTypical :
  percentageDecrease = 25 :=
by
  sorry

end prescribedDosageLessThanTypical_l2212_221240


namespace sum_reciprocal_squares_roots_l2212_221284

-- Define the polynomial P(X) = X^3 - 3X - 1
noncomputable def P (X : ℂ) : ℂ := X^3 - 3 * X - 1

-- Define the roots of the polynomial
variables (r1 r2 r3 : ℂ)

-- State that r1, r2, and r3 are roots of the polynomial
variable (hroots : P r1 = 0 ∧ P r2 = 0 ∧ P r3 = 0)

-- Vieta's formulas conditions for the polynomial P
variable (hvieta : r1 + r2 + r3 = 0 ∧ r1 * r2 + r1 * r3 + r2 * r3 = -3 ∧ r1 * r2 * r3 = 1)

-- The sum of the reciprocals of the squares of the roots
theorem sum_reciprocal_squares_roots : (1 / r1^2) + (1 / r2^2) + (1 / r3^2) = 9 := 
sorry

end sum_reciprocal_squares_roots_l2212_221284


namespace inequality_solution_l2212_221200

theorem inequality_solution (x : ℝ) : 
    (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
    sorry

end inequality_solution_l2212_221200


namespace range_of_m_l2212_221260

theorem range_of_m (m : ℝ) :
  (∃ x y, (x^2 / (2*m) + y^2 / (15 - m) = 1) ∧ m > 0 ∧ (15 - m > 0) ∧ (15 - m > 2 * m))
  ∨ (∀ e, (2 < e ∧ e < 3) ∧ ∃ a b x y, (y^2 / 2 - x^2 / (3 * m) = 1) ∧ (4 < (b^2 / a^2) ∧ (b^2 / a^2) < 9)) →
  (¬ (∃ x y, (x^2 / (2*m) + y^2 / (15 - m) = 1) ∧ (∀ e, (2 < e ∧ e < 3) ∧ ∃ a b x y, (y^2 / 2 - x^2 / (3 * m) = 1) ∧ (4 < (b^2 / a^2) ∧ (b^2 / a^2) < 9)))) →
  (0 < m ∧ m ≤ 2) ∨ (5 ≤ m ∧ m < 16/3) :=
by
  sorry

end range_of_m_l2212_221260


namespace geometric_arithmetic_seq_unique_ratio_l2212_221292

variable (d : ℚ) (q : ℚ) (k : ℤ)
variable (h_d_nonzero : d ≠ 0)
variable (h_q_pos : 0 < q) (h_q_lt_one : q < 1)
variable (h_integer : 14 / (1 + q + q^2) = k)

theorem geometric_arithmetic_seq_unique_ratio :
  q = 1 / 2 :=
by
  sorry

end geometric_arithmetic_seq_unique_ratio_l2212_221292


namespace sum_of_ages_l2212_221230

theorem sum_of_ages (a b c : ℕ) (h₁ : a = 20 + b + c) (h₂ : a^2 = 2050 + (b + c)^2) : a + b + c = 80 :=
sorry

end sum_of_ages_l2212_221230


namespace map_width_l2212_221296

theorem map_width (length : ℝ) (area : ℝ) (h1 : length = 2) (h2 : area = 20) : ∃ (width : ℝ), width = 10 :=
by
  sorry

end map_width_l2212_221296


namespace sum_of_three_consecutive_even_integers_l2212_221203

theorem sum_of_three_consecutive_even_integers : 
  ∃ (n : ℤ), n * (n + 2) * (n + 4) = 480 → n + (n + 2) + (n + 4) = 24 :=
by
  sorry

end sum_of_three_consecutive_even_integers_l2212_221203


namespace greatest_possible_value_of_x_l2212_221228

theorem greatest_possible_value_of_x (x : ℝ) (h : ( (5 * x - 25) / (4 * x - 5) ) ^ 3 + ( (5 * x - 25) / (4 * x - 5) ) = 16):
  x = 5 :=
sorry

end greatest_possible_value_of_x_l2212_221228


namespace bills_needed_can_pay_groceries_l2212_221225

theorem bills_needed_can_pay_groceries 
  (cans_of_soup : ℕ := 6) (price_per_can : ℕ := 2)
  (loaves_of_bread : ℕ := 3) (price_per_loaf : ℕ := 5)
  (boxes_of_cereal : ℕ := 4) (price_per_box : ℕ := 3)
  (gallons_of_milk : ℕ := 2) (price_per_gallon : ℕ := 4)
  (apples : ℕ := 7) (price_per_apple : ℕ := 1)
  (bags_of_cookies : ℕ := 5) (price_per_bag : ℕ := 3)
  (bottles_of_olive_oil : ℕ := 1) (price_per_bottle : ℕ := 8)
  : ∃ (bills_needed : ℕ), bills_needed = 4 :=
by
  let total_cost := (cans_of_soup * price_per_can) + 
                    (loaves_of_bread * price_per_loaf) +
                    (boxes_of_cereal * price_per_box) +
                    (gallons_of_milk * price_per_gallon) +
                    (apples * price_per_apple) +
                    (bags_of_cookies * price_per_bag) +
                    (bottles_of_olive_oil * price_per_bottle)
  let bills_needed := (total_cost + 19) / 20   -- Calculating ceiling of total_cost / 20
  sorry

end bills_needed_can_pay_groceries_l2212_221225


namespace parallel_line_through_P_perpendicular_line_through_P_l2212_221253

-- Define the line equations
def line1 (x y : ℝ) : Prop := 2 * x + y - 5 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y = 0
def line_l (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Define the equations for parallel and perpendicular lines through point P
def parallel_line (x y : ℝ) : Prop := 3 * x - y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x + 3 * y - 5 = 0

-- Define the point P where the lines intersect
def point_P : (ℝ × ℝ) := (2, 1)

-- Assert the proof statements
theorem parallel_line_through_P : parallel_line point_P.1 point_P.2 :=
by 
  -- proof content skipped with sorry
  sorry
  
theorem perpendicular_line_through_P : perpendicular_line point_P.1 point_P.2 :=
by 
  -- proof content skipped with sorry
  sorry

end parallel_line_through_P_perpendicular_line_through_P_l2212_221253


namespace factorize_polynomial_l2212_221237

theorem factorize_polynomial (x y : ℝ) : x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2 := 
by 
  sorry

end factorize_polynomial_l2212_221237


namespace hyperbola_condition_l2212_221251

theorem hyperbola_condition (m : ℝ) : 
  (exists a b : ℝ, ¬ a = 0 ∧ ¬ b = 0 ∧ ( ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 )) →
  ( -2 < m ∧ m < -1 ) :=
by
  sorry

end hyperbola_condition_l2212_221251


namespace selling_price_correct_l2212_221278

noncomputable def cost_price : ℝ := 2800
noncomputable def loss_percentage : ℝ := 25
noncomputable def loss_amount (cost_price loss_percentage : ℝ) : ℝ := (loss_percentage / 100) * cost_price
noncomputable def selling_price (cost_price loss_amount : ℝ) : ℝ := cost_price - loss_amount

theorem selling_price_correct : 
  selling_price cost_price (loss_amount cost_price loss_percentage) = 2100 :=
by
  sorry

end selling_price_correct_l2212_221278


namespace product_of_two_numbers_l2212_221258

theorem product_of_two_numbers : 
  ∀ (x y : ℝ), (x + y = 60) ∧ (x - y = 10) → x * y = 875 :=
by
  intros x y h
  sorry

end product_of_two_numbers_l2212_221258


namespace expression_value_l2212_221205

theorem expression_value : (100 - (1000 - 300)) - (1000 - (300 - 100)) = -1400 := by
  sorry

end expression_value_l2212_221205


namespace fraction_meaningful_l2212_221242

theorem fraction_meaningful (a : ℝ) : (∃ x, x = 2 / (a + 1)) ↔ a ≠ -1 :=
by
  sorry

end fraction_meaningful_l2212_221242


namespace planet_not_observed_l2212_221254

theorem planet_not_observed (k : ℕ) (d : Fin (2*k+1) → Fin (2*k+1) → ℝ) 
  (h_d : ∀ i j : Fin (2*k+1), i ≠ j → d i i = 0 ∧ d i j ≠ d i i) 
  (h_astronomer : ∀ i : Fin (2*k+1), ∃ j : Fin (2*k+1), j ≠ i ∧ ∀ k : Fin (2*k+1), k ≠ i → d i j < d i k) : 
  ∃ i : Fin (2*k+1), ∀ j : Fin (2*k+1), i ≠ j → ∃ l : Fin (2*k+1), (j ≠ l ∧ d l i < d l j) → false :=
  sorry

end planet_not_observed_l2212_221254


namespace range_of_a_l2212_221216

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + (a^2 + 1) * x + a - 2

theorem range_of_a (a : ℝ) :
  (f a 1 < 0) ∧ (f a (-1) < 0) → -1 < a ∧ a < 0 :=
by
  sorry

end range_of_a_l2212_221216


namespace max_of_2x_plus_y_l2212_221212

theorem max_of_2x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y / 2 + 1 / x + 8 / y = 10) : 
  2 * x + y ≤ 18 :=
sorry

end max_of_2x_plus_y_l2212_221212


namespace emilee_earns_25_l2212_221272

-- Define the conditions
def earns_together (jermaine terrence emilee : ℕ) : Prop := 
  jermaine + terrence + emilee = 90

def jermaine_more (jermaine terrence : ℕ) : Prop :=
  jermaine = terrence + 5

def terrence_earning : ℕ := 30

-- The goal: Prove Emilee earns 25 dollars
theorem emilee_earns_25 (jermaine terrence emilee : ℕ) (h1 : earns_together jermaine terrence emilee) 
  (h2 : jermaine_more jermaine terrence) (h3 : terrence = terrence_earning) : 
  emilee = 25 := 
sorry

end emilee_earns_25_l2212_221272


namespace horizontal_asymptote_is_3_l2212_221250

-- Definitions of the polynomials
noncomputable def p (x : ℝ) : ℝ := 15 * x^5 + 10 * x^4 + 5 * x^3 + 7 * x^2 + 6 * x + 2
noncomputable def q (x : ℝ) : ℝ := 5 * x^5 + 3 * x^4 + 9 * x^3 + 4 * x^2 + 2 * x + 1

-- Statement that we need to prove
theorem horizontal_asymptote_is_3 : 
  (∃ (y : ℝ), (∀ x : ℝ, x ≠ 0 → (p x / q x) = y) ∧ y = 3) :=
  sorry -- The proof is left as an exercise.

end horizontal_asymptote_is_3_l2212_221250


namespace Cara_possible_pairs_l2212_221255

-- Define the conditions and the final goal.
theorem Cara_possible_pairs : ∃ p : Nat, p = Nat.choose 7 2 ∧ p = 21 :=
by
  sorry

end Cara_possible_pairs_l2212_221255


namespace arctan_arcsin_arccos_sum_l2212_221223

theorem arctan_arcsin_arccos_sum :
  (Real.arctan (Real.sqrt 3 / 3) + Real.arcsin (-1 / 2) + Real.arccos 1 = 0) :=
by
  sorry

end arctan_arcsin_arccos_sum_l2212_221223


namespace minimum_sum_of_areas_l2212_221290

theorem minimum_sum_of_areas (x y : ℝ) (hx : x + y = 16) (hx_nonneg : 0 ≤ x) (hy_nonneg : 0 ≤ y) : 
  (x ^ 2 / 16 + y ^ 2 / 16) / 4 ≥ 8 :=
  sorry

end minimum_sum_of_areas_l2212_221290


namespace cost_per_ton_ice_correct_l2212_221232

variables {a p n s : ℝ}

-- Define the cost per ton of ice received by enterprise A
noncomputable def cost_per_ton_ice_received (a p n s : ℝ) : ℝ :=
  (2.5 * a + p * s) * 1000 / (2000 - n * s)

-- The statement of the theorem
theorem cost_per_ton_ice_correct :
  ∀ a p n s : ℝ,
  2000 - n * s ≠ 0 →
  cost_per_ton_ice_received a p n s = (2.5 * a + p * s) * 1000 / (2000 - n * s) := by
  intros a p n s h
  unfold cost_per_ton_ice_received
  sorry

end cost_per_ton_ice_correct_l2212_221232


namespace sum_cotangents_equal_l2212_221291

theorem sum_cotangents_equal (a b c S m_a m_b m_c S' : ℝ) (cot_A cot_B cot_C cot_A' cot_B' cot_C' : ℝ)
  (h1 : cot_A + cot_B + cot_C = (a^2 + b^2 + c^2) / (4 * S))
  (h2 : m_a^2 + m_b^2 + m_c^2 = 3 * (a^2 + b^2 + c^2) / 4)
  (h3 : S' = 3 * S / 4)
  (h4 : cot_A' + cot_B' + cot_C' = (m_a^2 + m_b^2 + m_c^2) / (4 * S')) :
  cot_A + cot_B + cot_C = cot_A' + cot_B' + cot_C' :=
by
  -- Proof is needed, but omitted here
  sorry

end sum_cotangents_equal_l2212_221291


namespace marble_problem_l2212_221239

def total_marbles_originally 
  (white_marbles : ℕ := 20) 
  (blue_marbles : ℕ) 
  (red_marbles : ℕ := blue_marbles) 
  (total_left : ℕ := 40)
  (jack_removes : ℕ := 2 * (white_marbles - blue_marbles)) : ℕ :=
  white_marbles + blue_marbles + red_marbles

theorem marble_problem : 
  ∀ (white_marbles : ℕ := 20) 
    (blue_marbles red_marbles : ℕ) 
    (jack_removes total_left : ℕ),
    red_marbles = blue_marbles →
    jack_removes = 2 * (white_marbles - blue_marbles) →
    total_left = total_marbles_originally white_marbles blue_marbles red_marbles - jack_removes →
    total_left = 40 →
    total_marbles_originally white_marbles blue_marbles red_marbles = 50 :=
by
  intros white_marbles blue_marbles red_marbles jack_removes total_left h1 h2 h3 h4
  sorry

end marble_problem_l2212_221239


namespace cost_per_pound_correct_l2212_221298

noncomputable def cost_per_pound_of_coffee (initial_amount spent_amount pounds_of_coffee : ℕ) : ℚ :=
  (initial_amount - spent_amount) / pounds_of_coffee

theorem cost_per_pound_correct :
  let initial_amount := 70
  let amount_left    := 35.68
  let pounds_of_coffee := 4
  (initial_amount - amount_left) / pounds_of_coffee = 8.58 := 
by
  sorry

end cost_per_pound_correct_l2212_221298


namespace b_should_pay_l2212_221233

def TotalRent : ℕ := 725
def Cost_a : ℕ := 12 * 8 * 5
def Cost_b : ℕ := 16 * 9 * 6
def Cost_c : ℕ := 18 * 6 * 7
def Cost_d : ℕ := 20 * 4 * 4
def TotalCost : ℕ := Cost_a + Cost_b + Cost_c + Cost_d
def Payment_b (Cost_b TotalCost TotalRent : ℕ) : ℕ := (Cost_b * TotalRent) / TotalCost

theorem b_should_pay :
  Payment_b Cost_b TotalCost TotalRent = 259 := 
  by
  unfold Payment_b
  -- Leaving the proof body empty as per instructions
  sorry

end b_should_pay_l2212_221233


namespace east_bound_cyclist_speed_l2212_221222

-- Define the speeds of the cyclists and the relationship between them
def east_bound_speed (t : ℕ) (x : ℕ) : ℕ := t * x
def west_bound_speed (t : ℕ) (x : ℕ) : ℕ := t * (x + 4)

-- Condition: After 5 hours, they are 200 miles apart
def total_distance (t : ℕ) (x : ℕ) : ℕ := east_bound_speed t x + west_bound_speed t x

theorem east_bound_cyclist_speed :
  ∃ x : ℕ, total_distance 5 x = 200 ∧ x = 18 :=
by
  sorry

end east_bound_cyclist_speed_l2212_221222


namespace sum_of_coefficients_l2212_221249

-- Given polynomial
def polynomial (x : ℝ) : ℝ := (3 * x - 1) ^ 7

-- Statement
theorem sum_of_coefficients :
  (polynomial 1) = 128 := 
sorry

end sum_of_coefficients_l2212_221249


namespace max_neg_integers_l2212_221241

theorem max_neg_integers (
  a b c d e f g h : ℤ
) (h_a : a ≠ 0) (h_c : c ≠ 0) (h_e : e ≠ 0)
  (h_ineq : (a * b^2 + c * d * e^3) * (f * g^2 * h + f^3 - g^2) < 0)
  (h_abs : |d| < |f| ∧ |f| < |h|)
  : ∃ s, s = 5 ∧ ∀ (neg_count : ℕ), neg_count ≤ s := 
sorry

end max_neg_integers_l2212_221241


namespace exists_n_divides_2022n_minus_n_l2212_221244

theorem exists_n_divides_2022n_minus_n (p : ℕ) [hp : Fact (Nat.Prime p)] :
  ∃ n : ℕ, p ∣ (2022^n - n) :=
sorry

end exists_n_divides_2022n_minus_n_l2212_221244


namespace min_value_of_expression_l2212_221268

theorem min_value_of_expression 
  (x y : ℝ) 
  (h : 3 * |x - y| + |2 * x - 5| = x + 1) : 
  ∃ (x y : ℝ), 2 * x + y = 4 :=
by {
  sorry
}

end min_value_of_expression_l2212_221268


namespace expected_losses_correct_l2212_221202

def game_probabilities : List (ℕ × ℝ) := [
  (5, 0.6), (10, 0.75), (15, 0.4), (12, 0.85), (20, 0.5),
  (30, 0.2), (10, 0.9), (25, 0.7), (35, 0.65), (10, 0.8)
]

def expected_losses : ℝ :=
  (1 - 0.6) + (1 - 0.75) + (1 - 0.4) + (1 - 0.85) +
  (1 - 0.5) + (1 - 0.2) + (1 - 0.9) + (1 - 0.7) +
  (1 - 0.65) + (1 - 0.8)

theorem expected_losses_correct :
  expected_losses = 3.55 :=
by {
  -- Skipping the actual proof and inserting a sorry as instructed
  sorry
}

end expected_losses_correct_l2212_221202


namespace minValue_expression_l2212_221220

noncomputable def minValue (x y : ℝ) : ℝ :=
  4 / x^2 + 4 / (x * y) + 1 / y^2

theorem minValue_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : (x - 2 * y)^2 = (x * y)^3) :
  minValue x y = 4 * Real.sqrt 2 :=
sorry

end minValue_expression_l2212_221220


namespace num_O_atoms_l2212_221287

def compound_molecular_weight : ℕ := 62
def atomic_weight_H : ℕ := 1
def atomic_weight_C : ℕ := 12
def atomic_weight_O : ℕ := 16
def num_H_atoms : ℕ := 2
def num_C_atoms : ℕ := 1

theorem num_O_atoms (H_weight : ℕ := num_H_atoms * atomic_weight_H)
                    (C_weight : ℕ := num_C_atoms * atomic_weight_C)
                    (total_weight : ℕ := compound_molecular_weight)
                    (O_weight := atomic_weight_O) : 
    (total_weight - (H_weight + C_weight)) / O_weight = 3 :=
by
  sorry

end num_O_atoms_l2212_221287


namespace eval_x_squared_minus_y_squared_l2212_221226

theorem eval_x_squared_minus_y_squared (x y : ℝ) (h1 : 3 * x + 2 * y = 30) (h2 : 4 * x + 2 * y = 34) : x^2 - y^2 = -65 :=
by
  sorry

end eval_x_squared_minus_y_squared_l2212_221226


namespace weekend_weekday_ratio_l2212_221288

-- Defining the basic constants and conditions
def weekday_episodes : ℕ := 8
def total_episodes_in_week : ℕ := 88

-- Defining the main theorem
theorem weekend_weekday_ratio : (2 * (total_episodes_in_week - 5 * weekday_episodes)) / weekday_episodes = 3 :=
by
  sorry

end weekend_weekday_ratio_l2212_221288


namespace simplify_fraction_l2212_221234

theorem simplify_fraction (m : ℝ) (h₁: m ≠ 0) (h₂: m ≠ 1): (m - 1) / m / ((m - 1) / (m * m)) = m := by
  sorry

end simplify_fraction_l2212_221234


namespace sum_a_b_eq_five_l2212_221213

theorem sum_a_b_eq_five (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - a * x + b < 0) : a + b = 5 :=
sorry

end sum_a_b_eq_five_l2212_221213


namespace logarithm_simplification_l2212_221252

theorem logarithm_simplification :
  (1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 7 / Real.log 9 + 1)) =
  1 - (Real.log 7 / Real.log 1008) :=
sorry

end logarithm_simplification_l2212_221252


namespace find_k_l2212_221219

theorem find_k (x y k : ℝ)
  (h1 : x - 4 * y + 3 ≤ 0)
  (h2 : 3 * x + 5 * y - 25 ≤ 0)
  (h3 : x ≥ 1)
  (h4 : ∃ z, z = k * x + y ∧ z = 12)
  (h5 : ∃ z', z' = k * x + y ∧ z' = 3) :
  k = 2 :=
by sorry

end find_k_l2212_221219


namespace corresponding_angles_not_always_equal_l2212_221238

theorem corresponding_angles_not_always_equal :
  (∀ α β c : ℝ, (α = β ∧ ¬c = 0) → (∃ x1 x2 y : ℝ, α = x1 ∧ β = x2 ∧ x1 = y * c ∧ x2 = y * c)) → False :=
by
  sorry

end corresponding_angles_not_always_equal_l2212_221238


namespace triangle_angles_l2212_221293

theorem triangle_angles (α β : ℝ) (A B C : ℝ) (hA : A = 2) (hB : B = 3) (hC : C = 4) :
  2 * α + 3 * β = 180 :=
sorry

end triangle_angles_l2212_221293


namespace sqrt_two_irrational_l2212_221209

theorem sqrt_two_irrational :
  ¬ ∃ (a b : ℕ), (a.gcd b = 1) ∧ (b ≠ 0) ∧ (a^2 = 2 * b^2) :=
sorry

end sqrt_two_irrational_l2212_221209


namespace janet_spends_more_on_piano_l2212_221264

-- Condition definitions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℝ := 52

-- Calculations based on conditions
def weekly_cost_clarinet : ℝ := clarinet_hourly_rate * clarinet_hours_per_week
def weekly_cost_piano : ℝ := piano_hourly_rate * piano_hours_per_week
def weekly_difference : ℝ := weekly_cost_piano - weekly_cost_clarinet
def yearly_difference : ℝ := weekly_difference * weeks_per_year

theorem janet_spends_more_on_piano : yearly_difference = 1040 := by
  sorry 

end janet_spends_more_on_piano_l2212_221264


namespace Andrey_knows_the_secret_l2212_221273

/-- Question: Does Andrey know the secret?
    Conditions:
    - Andrey says: "I know the secret!"
    - Boris says to Andrey: "No, you don't!"
    - Victor says to Boris: "Boris, you are wrong!"
    - Gosha says to Victor: "No, you are wrong!"
    - Dima says to Gosha: "Gosha, you are lying!"
    - More than half of the kids told the truth (i.e., at least 3 out of 5). --/
theorem Andrey_knows_the_secret (Andrey Boris Victor Gosha Dima : Prop) (truth_count : ℕ)
    (h1 : Andrey)   -- Andrey says he knows the secret
    (h2 : ¬Andrey → Boris)   -- Boris says Andrey does not know the secret
    (h3 : ¬Boris → Victor)   -- Victor says Boris is wrong
    (h4 : ¬Victor → Gosha)   -- Gosha says Victor is wrong
    (h5 : ¬Gosha → Dima)   -- Dima says Gosha is lying
    (h6 : truth_count > 2)   -- More than half of the friends tell the truth (at least 3 out of 5)
    : Andrey := 
sorry

end Andrey_knows_the_secret_l2212_221273


namespace fraction_expression_l2212_221256

theorem fraction_expression : (1 / 3) ^ 3 * (1 / 8) = 1 / 216 :=
by
  sorry

end fraction_expression_l2212_221256


namespace length_MN_l2212_221283

variables {A B C D M N : Type}
variables {BC AD AB : ℝ} -- Lengths of sides
variables {a b : ℝ}

-- Given conditions
def is_trapezoid (a b BC AD AB : ℝ) : Prop :=
  BC = a ∧ AD = b ∧ AB = AD + BC

-- Given, side AB is divided into 5 equal parts and a line parallel to bases is drawn through the 3rd division point
def is_divided (AB : ℝ) : Prop := ∃ P_1 P_2 P_3 P_4, AB = P_4 + P_3 + P_2 + P_1

-- Prove the length of MN
theorem length_MN (a b : ℝ) (h_trapezoid : is_trapezoid a b BC AD AB) (h_divided : is_divided AB) : 
  MN = (2 * BC + 3 * AD) / 5 :=
sorry

end length_MN_l2212_221283


namespace ratio_of_areas_l2212_221299

theorem ratio_of_areas (s : ℝ) (h1 : s > 0) : 
  let small_square_area := s^2
  let total_small_squares_area := 4 * s^2
  let large_square_side_length := 4 * s
  let large_square_area := (4 * s)^2
  total_small_squares_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l2212_221299


namespace find_fraction_l2212_221206

-- Define the given variables and conditions
variables (x y : ℝ)
-- Assume x and y are nonzero
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
-- Assume the given condition
variable (h : (4*x + 2*y) / (2*x - 8*y) = 3)

-- Define the theorem to be proven
theorem find_fraction (h : (4*x + 2*y) / (2*x - 8*y) = 3) : (x + 4 * y) / (4 * x - y) = 1 / 3 := 
by
  sorry

end find_fraction_l2212_221206


namespace number_of_boys_l2212_221201

-- Definitions reflecting the conditions
def total_students := 1200
def sample_size := 200
def extra_boys := 10

-- Main problem statement
theorem number_of_boys (B G b g : ℕ) 
  (h_total_students : B + G = total_students)
  (h_sample_size : b + g = sample_size)
  (h_extra_boys : b = g + extra_boys)
  (h_stratified : b * G = g * B) :
  B = 660 :=
by sorry

end number_of_boys_l2212_221201


namespace number_of_b_values_l2212_221217

-- Let's define the conditions and the final proof required.
def inequations (x b : ℤ) : Prop := 
  (3 * x > 4 * x - 4) ∧
  (4 * x - b > -8) ∧
  (5 * x < b + 13)

theorem number_of_b_values :
  (∀ x : ℤ, 1 ≤ x → x ≠ 3 → ¬ inequations x b) →
  (∃ (b_values : Finset ℤ), 
      (∀ b ∈ b_values, inequations 3 b) ∧ 
      (b_values.card = 7)) :=
sorry

end number_of_b_values_l2212_221217


namespace find_a_l2212_221210

theorem find_a (a b c : ℕ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) (h_positive_c : 0 < c) (h_eq : (18 ^ a) * (9 ^ (3 * a - 1)) * (c ^ a) = (2 ^ 7) * (3 ^ b)) : a = 7 := by
  sorry

end find_a_l2212_221210


namespace total_weight_collected_l2212_221261

def GinaCollectedBags : ℕ := 8
def NeighborhoodFactor : ℕ := 120
def WeightPerBag : ℕ := 6

theorem total_weight_collected :
  (GinaCollectedBags * NeighborhoodFactor + GinaCollectedBags) * WeightPerBag = 5808 :=
by
  sorry

end total_weight_collected_l2212_221261


namespace min_sum_fraction_sqrt_l2212_221281

open Real

theorem min_sum_fraction_sqrt (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ min, min = sqrt 2 ∧ ∀ z, (z = (x / sqrt (1 - x) + y / sqrt (1 - y))) → z ≥ sqrt 2 :=
sorry

end min_sum_fraction_sqrt_l2212_221281


namespace ratio_of_boys_l2212_221215

theorem ratio_of_boys (p : ℚ) (hp : p = (3 / 4) * (1 - p)) : p = 3 / 7 :=
by
  -- Proof would be provided here
  sorry

end ratio_of_boys_l2212_221215


namespace number_of_times_each_player_plays_l2212_221280

def players : ℕ := 7
def total_games : ℕ := 42

theorem number_of_times_each_player_plays (x : ℕ) 
  (H1 : 42 = (players * (players - 1) * x) / 2) : x = 2 :=
by
  sorry

end number_of_times_each_player_plays_l2212_221280


namespace thirty_percent_of_forty_percent_of_x_l2212_221262

theorem thirty_percent_of_forty_percent_of_x (x : ℝ) (h : 0.12 * x = 24) : 0.30 * 0.40 * x = 24 :=
sorry

end thirty_percent_of_forty_percent_of_x_l2212_221262


namespace value_of_y_plus_10_l2212_221229

theorem value_of_y_plus_10 (x y : ℝ) (h1 : 3 * x = (3 / 4) * y) (h2 : x = 20) : y + 10 = 90 :=
by
  sorry

end value_of_y_plus_10_l2212_221229


namespace columbian_coffee_price_is_correct_l2212_221224

-- Definitions based on the conditions
def total_mix_weight : ℝ := 100
def brazilian_coffee_price_per_pound : ℝ := 3.75
def final_mix_price_per_pound : ℝ := 6.35
def columbian_coffee_weight : ℝ := 52

-- Let C be the price per pound of the Columbian coffee
noncomputable def columbian_coffee_price_per_pound : ℝ := sorry

-- Define the Lean 4 proof problem
theorem columbian_coffee_price_is_correct :
  columbian_coffee_price_per_pound = 8.75 :=
by
  -- Total weight and calculation based on conditions
  let brazilian_coffee_weight := total_mix_weight - columbian_coffee_weight
  let total_value_of_columbian := columbian_coffee_weight * columbian_coffee_price_per_pound
  let total_value_of_brazilian := brazilian_coffee_weight * brazilian_coffee_price_per_pound
  let total_value_of_mix := total_mix_weight * final_mix_price_per_pound
  
  -- Main equation based on the mix
  have main_eq : total_value_of_columbian + total_value_of_brazilian = total_value_of_mix :=
    by sorry

  -- Solve for C (columbian coffee price per pound)
  sorry

end columbian_coffee_price_is_correct_l2212_221224


namespace total_amount_is_24_l2212_221218

-- Define the original price of a tub of ice cream
def original_price_ice_cream : ℕ := 12

-- Define the discount per tub of ice cream
def discount_per_tub : ℕ := 2

-- Define the discounted price of a tub of ice cream
def discounted_price_ice_cream : ℕ := original_price_ice_cream - discount_per_tub

-- Define the price for 5 cans of juice
def price_per_5_cans_of_juice : ℕ := 2

-- Define the number of cans of juice bought
def cans_of_juice_bought : ℕ := 10

-- Calculate the total cost for two tubs of ice cream and 10 cans of juice
def total_cost (p1 p2 : ℕ) : ℕ := 2 * p1 + (price_per_5_cans_of_juice * (cans_of_juice_bought / 5))

-- Prove that the total cost is $24
theorem total_amount_is_24 : total_cost discounted_price_ice_cream price_per_5_cans_of_juice = 24 := by
  sorry

end total_amount_is_24_l2212_221218


namespace smallest_other_integer_l2212_221207

-- Definitions of conditions
def gcd_condition (a b : ℕ) (x : ℕ) : Prop := 
  Nat.gcd a b = x + 5

def lcm_condition (a b : ℕ) (x : ℕ) : Prop := 
  Nat.lcm a b = x * (x + 5)

def sum_condition (a b : ℕ) : Prop := 
  a + b < 100

-- Main statement incorporating all conditions
theorem smallest_other_integer {x b : ℕ} (hx_pos : x > 0)
  (h_gcd : gcd_condition 45 b x)
  (h_lcm : lcm_condition 45 b x)
  (h_sum : sum_condition 45 b) :
  b = 12 :=
sorry

end smallest_other_integer_l2212_221207


namespace remainder_2753_div_98_l2212_221297

theorem remainder_2753_div_98 : (2753 % 98) = 9 := 
by sorry

end remainder_2753_div_98_l2212_221297


namespace shopkeeper_oranges_l2212_221247

theorem shopkeeper_oranges (O : ℕ) 
  (bananas : ℕ) 
  (percent_rotten_oranges : ℕ) 
  (percent_rotten_bananas : ℕ) 
  (percent_good_condition : ℚ) 
  (h1 : bananas = 400) 
  (h2 : percent_rotten_oranges = 15) 
  (h3 : percent_rotten_bananas = 6) 
  (h4 : percent_good_condition = 88.6) : 
  O = 600 :=
by
  -- This proof needs to be filled in.
  sorry

end shopkeeper_oranges_l2212_221247


namespace find_c_l2212_221259

def conditions (c d : ℝ) : Prop :=
  -- The polynomial 6x^3 + 7cx^2 + 3dx + 2c = 0 has three distinct positive roots
  ∃ u v w : ℝ, 0 < u ∧ 0 < v ∧ 0 < w ∧ u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
  (6 * u^3 + 7 * c * u^2 + 3 * d * u + 2 * c = 0) ∧
  (6 * v^3 + 7 * c * v^2 + 3 * d * v + 2 * c = 0) ∧
  (6 * w^3 + 7 * c * w^2 + 3 * d * w + 2 * c = 0) ∧
  -- Sum of the base-2 logarithms of the roots is 6
  Real.log (u * v * w) / Real.log 2 = 6

theorem find_c (c d : ℝ) (h : conditions c d) : c = -192 :=
sorry

end find_c_l2212_221259
