import Mathlib

namespace sampling_survey_suitability_l453_45395

-- Define the conditions
def OptionA := "Understanding the effectiveness of a certain drug"
def OptionB := "Understanding the vision status of students in this class"
def OptionC := "Organizing employees of a unit to undergo physical examinations at a hospital"
def OptionD := "Inspecting components of artificial satellite"

-- Mathematical statement
theorem sampling_survey_suitability : OptionA = "Understanding the effectiveness of a certain drug" → 
  ∃ (suitable_for_sampling_survey : String), suitable_for_sampling_survey = OptionA :=
by
  sorry

end sampling_survey_suitability_l453_45395


namespace vecMA_dotProduct_vecBA_range_l453_45303

-- Define the conditions
def pointM : ℝ × ℝ := (1, 0)

def onEllipse (p : ℝ × ℝ) : Prop := (p.1^2 / 4 + p.2^2 = 1)

def vecMA (A : ℝ × ℝ) := (A.1 - pointM.1, A.2 - pointM.2)
def vecMB (B : ℝ × ℝ) := (B.1 - pointM.1, B.2 - pointM.2)
def vecBA (A B : ℝ × ℝ) := (A.1 - B.1, A.2 - B.2)

def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the statement
theorem vecMA_dotProduct_vecBA_range (A B : ℝ × ℝ) (α : ℝ) :
  onEllipse A → onEllipse B → dotProduct (vecMA A) (vecMB B) = 0 → 
  A = (2 * Real.cos α, Real.sin α) → 
  (2/3 ≤ dotProduct (vecMA A) (vecBA A B) ∧ dotProduct (vecMA A) (vecBA A B) ≤ 9) :=
sorry

end vecMA_dotProduct_vecBA_range_l453_45303


namespace range_of_a_l453_45329

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, (2 * a < x ∧ x < a + 5) → (x < 6)) ↔ (1 < a ∧ a < 5) :=
by
  sorry

end range_of_a_l453_45329


namespace avgPercentageSpentOnFoodCorrect_l453_45383

-- Definitions for given conditions
def JanuaryIncome : ℕ := 3000
def JanuaryPetrolExpenditure : ℕ := 300
def JanuaryHouseRentPercentage : ℕ := 14
def JanuaryClothingPercentage : ℕ := 10
def JanuaryUtilityBillsPercentage : ℕ := 5
def FebruaryIncome : ℕ := 4000
def FebruaryPetrolExpenditure : ℕ := 400
def FebruaryHouseRentPercentage : ℕ := 14
def FebruaryClothingPercentage : ℕ := 10
def FebruaryUtilityBillsPercentage : ℕ := 5

-- Calculate percentage spent on food over January and February
noncomputable def avgPercentageSpentOnFood : ℝ :=
  let totalIncome := (JanuaryIncome + FebruaryIncome: ℝ)
  let totalFoodExpenditure :=
    let remainingJan := (JanuaryIncome - JanuaryPetrolExpenditure: ℝ) 
                         - (JanuaryHouseRentPercentage / 100 * (JanuaryIncome - JanuaryPetrolExpenditure: ℝ))
                         - (JanuaryClothingPercentage / 100 * JanuaryIncome)
                         - (JanuaryUtilityBillsPercentage / 100 * JanuaryIncome)
    let remainingFeb := (FebruaryIncome - FebruaryPetrolExpenditure: ℝ)
                         - (FebruaryHouseRentPercentage / 100 * (FebruaryIncome - FebruaryPetrolExpenditure: ℝ))
                         - (FebruaryClothingPercentage / 100 * FebruaryIncome)
                         - (FebruaryUtilityBillsPercentage / 100 * FebruaryIncome)
    remainingJan + remainingFeb
  (totalFoodExpenditure / totalIncome) * 100

theorem avgPercentageSpentOnFoodCorrect : avgPercentageSpentOnFood = 62.4 := by
  sorry

end avgPercentageSpentOnFoodCorrect_l453_45383


namespace num_2_edge_paths_l453_45332

-- Let T be a tetrahedron with vertices connected such that each vertex has exactly 3 edges.
-- Prove that the number of distinct 2-edge paths from a starting vertex P to an ending vertex Q is 3.

def tetrahedron : Type := ℕ -- This is a simplified representation of vertices

noncomputable def edges (a b : tetrahedron) : Prop := true -- Each pair of distinct vertices is an edge in a tetrahedron

theorem num_2_edge_paths (P Q : tetrahedron) (hP : P ≠ Q) : 
  -- There are 3 distinct 2-edge paths from P to Q  
  ∃ (paths : Finset (tetrahedron × tetrahedron)), 
    paths.card = 3 ∧ 
    ∀ (p : tetrahedron × tetrahedron), p ∈ paths → 
      edges P p.1 ∧ edges p.1 p.2 ∧ p.2 = Q :=
by 
  sorry

end num_2_edge_paths_l453_45332


namespace annie_blocks_walked_l453_45365

theorem annie_blocks_walked (x : ℕ) (h1 : 7 * 2 = 14) (h2 : 2 * x + 14 = 24) : x = 5 :=
by
  sorry

end annie_blocks_walked_l453_45365


namespace slope_of_perpendicular_line_l453_45307

theorem slope_of_perpendicular_line 
  (x1 y1 x2 y2 : ℤ)
  (h : x1 = 3 ∧ y1 = -4 ∧ x2 = -6 ∧ y2 = 2) : 
∃ m : ℚ, m = 3/2 :=
by
  sorry

end slope_of_perpendicular_line_l453_45307


namespace race_distance_l453_45351

theorem race_distance (D : ℝ) (h1 : (D / 36) * 45 = D + 20) : D = 80 :=
by
  sorry

end race_distance_l453_45351


namespace solve_ff_eq_x_l453_45379

def f (x : ℝ) : ℝ := x^2 + 2 * x - 5

theorem solve_ff_eq_x :
  ∀ x : ℝ, f (f x) = x ↔ (x = ( -1 + Real.sqrt 21 ) / 2) ∨ (x = ( -1 - Real.sqrt 21 ) / 2) ∨
                          (x = ( -3 + Real.sqrt 17 ) / 2) ∨ (x = ( -3 - Real.sqrt 17 ) / 2) := 
by
  sorry

end solve_ff_eq_x_l453_45379


namespace savings_per_egg_l453_45343

def price_per_organic_egg : ℕ := 50 
def cost_of_tray : ℕ := 1200 -- in cents
def number_of_eggs_in_tray : ℕ := 30

theorem savings_per_egg : 
  price_per_organic_egg - (cost_of_tray / number_of_eggs_in_tray) = 10 := 
by
  sorry

end savings_per_egg_l453_45343


namespace quadratic_root_value_of_b_l453_45348

theorem quadratic_root_value_of_b :
  (∃ r1 r2 : ℝ, 2 * r1^2 + b * r1 - 20 = 0 ∧ r1 = -5 ∧ r1 * r2 = -10 ∧ r1 + r2 = -b / 2) → b = 6 :=
by
  intro h
  obtain ⟨r1, r2, h_eq1, h_r1, h_prod, h_sum⟩ := h
  sorry

end quadratic_root_value_of_b_l453_45348


namespace lines_in_plane_l453_45313

  -- Define the necessary objects in Lean
  structure Line (α : Type) := (equation : α → α → Prop)

  def same_plane (l1 l2 : Line ℝ) : Prop := 
  -- Here you can define what it means for l1 and l2 to be in the same plane.
  sorry

  def intersect (l1 l2 : Line ℝ) : Prop := 
  -- Define what it means for two lines to intersect.
  sorry

  def parallel (l1 l2 : Line ℝ) : Prop := 
  -- Define what it means for two lines to be parallel.
  sorry

  theorem lines_in_plane (l1 l2 : Line ℝ) (h : same_plane l1 l2) : 
    (intersect l1 l2) ∨ (parallel l1 l2) := 
  by 
      sorry
  
end lines_in_plane_l453_45313


namespace a_4_is_zero_l453_45374

def a_n (n : ℕ) : ℕ := n^2 - 2*n - 8

theorem a_4_is_zero : a_n 4 = 0 := 
by
  sorry

end a_4_is_zero_l453_45374


namespace total_customers_in_line_l453_45363

-- Definition of the number of people standing in front of the last person
def num_people_in_front : Nat := 8

-- Definition of the last person in the line
def last_person : Nat := 1

-- Statement to prove
theorem total_customers_in_line : num_people_in_front + last_person = 9 := by
  sorry

end total_customers_in_line_l453_45363


namespace cubes_sum_l453_45308

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 8) (h2 : a * b + a * c + b * c = 9) (h3 : a * b * c = -18) :
  a^3 + b^3 + c^3 = 242 :=
by
  sorry

end cubes_sum_l453_45308


namespace at_least_one_nonnegative_l453_45368

theorem at_least_one_nonnegative
  (a1 a2 a3 a4 a5 a6 a7 a8 : ℝ)
  (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (h3 : a3 ≠ 0) (h4 : a4 ≠ 0)
  (h5 : a5 ≠ 0) (h6 : a6 ≠ 0) (h7 : a7 ≠ 0) (h8 : a8 ≠ 0)
  : (a1 * a3 + a2 * a4 ≥ 0) ∨ (a1 * a5 + a2 * a6 ≥ 0) ∨ (a1 * a7 + a2 * a8 ≥ 0) ∨
    (a3 * a5 + a4 * a6 ≥ 0) ∨ (a3 * a7 + a4 * a8 ≥ 0) ∨ (a5 * a7 + a6 * a8 ≥ 0) := 
sorry

end at_least_one_nonnegative_l453_45368


namespace polynomial_approx_eq_l453_45320

theorem polynomial_approx_eq (x : ℝ) (h : x^4 - 4*x^3 + 4*x^2 + 4 = 4.999999999999999) : x = 1 :=
sorry

end polynomial_approx_eq_l453_45320


namespace quadratic_root_k_value_l453_45367

theorem quadratic_root_k_value 
  (k : ℝ) 
  (h_roots : ∀ x : ℝ, (5 * x^2 + 7 * x + k = 0) → (x = ( -7 + Real.sqrt (-191) ) / 10 ∨ x = ( -7 - Real.sqrt (-191) ) / 10)) : 
  k = 12 :=
sorry

end quadratic_root_k_value_l453_45367


namespace intersection_eq_l453_45322

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | x^2 - 1 ≥ 0}

theorem intersection_eq : A ∩ B = {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ 2)} :=
by sorry

end intersection_eq_l453_45322


namespace find_number_l453_45331

theorem find_number:
  ∃ x : ℝ, (3/4 * x + 9 = 1/5 * (x - 8 * x^(1/3))) ∧ x = -27 :=
by
  sorry

end find_number_l453_45331


namespace find_m_l453_45375

-- Definition of vectors in terms of the condition
def vec_a (m : ℝ) : ℝ × ℝ := (2 * m + 1, m)
def vec_b (m : ℝ) : ℝ × ℝ := (1, m)

-- Condition that vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2) = 0

-- Problem statement: find m such that vec_a is perpendicular to vec_b
theorem find_m (m : ℝ) (h : perpendicular (vec_a m) (vec_b m)) : m = -1 := by
  sorry

end find_m_l453_45375


namespace two_digit_numbers_equal_three_times_product_of_digits_l453_45369

theorem two_digit_numbers_equal_three_times_product_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 3 * a * b} = {15, 24} :=
by
  sorry

end two_digit_numbers_equal_three_times_product_of_digits_l453_45369


namespace intersection_AB_l453_45382

def setA : Set ℝ := { x | x^2 - 2*x - 3 < 0}
def setB : Set ℝ := { x | x > 1 }
def intersection : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem intersection_AB : setA ∩ setB = intersection :=
by
  sorry

end intersection_AB_l453_45382


namespace min_value_of_expression_l453_45378

-- positive real numbers a and b
variables (a b : ℝ)
variables (ha : 0 < a) (hb : 0 < b)
-- given condition: 1/a + 9/b = 6
variable (h : 1 / a + 9 / b = 6)

theorem min_value_of_expression : (a + 1) * (b + 9) ≥ 16 := by
  sorry

end min_value_of_expression_l453_45378


namespace Pam_current_balance_l453_45391

-- Given conditions as definitions
def initial_balance : ℕ := 400
def tripled_balance : ℕ := 3 * initial_balance
def current_balance : ℕ := tripled_balance - 250

-- The theorem to be proved
theorem Pam_current_balance : current_balance = 950 := by
  sorry

end Pam_current_balance_l453_45391


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l453_45302

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l453_45302


namespace chickens_rabbits_l453_45357

theorem chickens_rabbits (c r : ℕ) 
  (h1 : c = r - 20)
  (h2 : 4 * r = 6 * c + 10) :
  c = 35 := by
  sorry

end chickens_rabbits_l453_45357


namespace find_d_from_factor_condition_l453_45301

theorem find_d_from_factor_condition (d : ℚ) : (∀ x, x = 5 → d * x^4 + 13 * x^3 - 2 * d * x^2 - 58 * x + 65 = 0) → d = -28 / 23 :=
by
  intro h
  sorry

end find_d_from_factor_condition_l453_45301


namespace arithmetic_sequence_a7_l453_45397

theorem arithmetic_sequence_a7 (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 2) 
  (h2 : a 3 + a 4 = 9) 
  (common_diff : ∀ n, a (n + 1) = a n + d) :
  a 7 = 8 :=
by
  sorry

end arithmetic_sequence_a7_l453_45397


namespace coordinates_of_point_P_l453_45328

theorem coordinates_of_point_P {x y : ℝ} (hx : |x| = 2) (hy : y = 1 ∨ y = -1) (hxy : x < 0 ∧ y > 0) : 
  (x, y) = (-2, 1) := 
by 
  sorry

end coordinates_of_point_P_l453_45328


namespace max_three_topping_pizzas_l453_45358

-- Define the combinations function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Assert the condition and the question with the expected answer
theorem max_three_topping_pizzas : combination 8 3 = 56 :=
by
  sorry

end max_three_topping_pizzas_l453_45358


namespace pens_in_each_pack_l453_45359

-- Given the conditions
def Kendra_packs : ℕ := 4
def Tony_packs : ℕ := 2
def pens_kept_each : ℕ := 2
def friends : ℕ := 14

-- Theorem statement
theorem pens_in_each_pack : ∃ (P : ℕ), Kendra_packs * P + Tony_packs * P - pens_kept_each * 2 - friends = 0 ∧ P = 3 := by
  sorry

end pens_in_each_pack_l453_45359


namespace brick_width_l453_45317

/-- Let dimensions of the wall be 700 cm (length), 600 cm (height), and 22.5 cm (thickness).
    Let dimensions of each brick be 25 cm (length), W cm (width), and 6 cm (height).
    Given that 5600 bricks are required to build the wall, prove that the width of each brick is 11.25 cm. -/
theorem brick_width (W : ℝ)
  (h_wall_dimensions : 700 = 700) (h_wall_height : 600 = 600) (h_wall_thickness : 22.5 = 22.5)
  (h_brick_length : 25 = 25) (h_brick_height : 6 = 6) (h_num_bricks : 5600 = 5600)
  (h_wall_volume : 700 * 600 * 22.5 = 9450000)
  (h_brick_volume : 25 * W * 6 = 9450000 / 5600) :
  W = 11.25 :=
sorry

end brick_width_l453_45317


namespace intersection_of_A_and_B_l453_45384

-- Definitions representing the conditions
def setA : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def setB : Set ℝ := {x | x < 2}

-- Proof problem statement
theorem intersection_of_A_and_B : setA ∩ setB = {x | -1 < x ∧ x < 2} :=
sorry

end intersection_of_A_and_B_l453_45384


namespace no_single_two_three_digit_solution_l453_45356

theorem no_single_two_three_digit_solution :
  ¬ ∃ (x y z : ℕ),
    (1 ≤ x ∧ x ≤ 9) ∧
    (10 ≤ y ∧ y ≤ 99) ∧
    (100 ≤ z ∧ z ≤ 999) ∧
    (1/x : ℝ) = 1/y + 1/z :=
by
  sorry

end no_single_two_three_digit_solution_l453_45356


namespace distribution_methods_l453_45366

theorem distribution_methods (n m k : Nat) (h : n = 23) (h1 : m = 10) (h2 : k = 2) :
  (∃ d : Nat, d = Nat.choose m 1 + 2 * Nat.choose m 2 + Nat.choose m 3) →
  ∃ x : Nat, x = 220 :=
by
  sorry

end distribution_methods_l453_45366


namespace largest_num_blocks_l453_45360

-- Define the volume of the box
def volume_box (l₁ w₁ h₁ : ℕ) : ℕ :=
  l₁ * w₁ * h₁

-- Define the volume of the block
def volume_block (l₂ w₂ h₂ : ℕ) : ℕ :=
  l₂ * w₂ * h₂

-- Define the function to calculate maximum blocks
def max_blocks (V_box V_block : ℕ) : ℕ :=
  V_box / V_block

theorem largest_num_blocks :
  max_blocks (volume_box 5 4 6) (volume_block 3 3 2) = 6 :=
by
  sorry

end largest_num_blocks_l453_45360


namespace hat_cost_l453_45389

noncomputable def cost_of_hat (H : ℕ) : Prop :=
  let cost_shirts := 3 * 5
  let cost_jeans := 2 * 10
  let cost_hats := 4 * H
  let total_cost := 51
  cost_shirts + cost_jeans + cost_hats = total_cost

theorem hat_cost : ∃ H : ℕ, cost_of_hat H ∧ H = 4 :=
by 
  sorry

end hat_cost_l453_45389


namespace factorization_result_l453_45381

theorem factorization_result (a b : ℤ) (h1 : 25 * x^2 - 160 * x - 336 = (5 * x + a) * (5 * x + b)) :
  a + 2 * b = 20 :=
by
  sorry

end factorization_result_l453_45381


namespace bella_items_l453_45306

theorem bella_items (M F D : ℕ) 
  (h1 : M = 60)
  (h2 : M = 2 * F)
  (h3 : F = D + 20) :
  (7 * M + 7 * F + 7 * D) / 5 = 140 := 
by
  sorry

end bella_items_l453_45306


namespace loan_percentage_correct_l453_45338

-- Define the parameters and conditions of the problem
def house_initial_value : ℕ := 100000
def house_increase_percentage : ℝ := 0.25
def new_house_cost : ℕ := 500000
def loan_percentage : ℝ := 75.0

-- Define the theorem we want to prove
theorem loan_percentage_correct :
  let increase_value := house_initial_value * house_increase_percentage
  let sale_price := house_initial_value + increase_value
  let loan_amount := new_house_cost - sale_price
  let loan_percentage_computed := (loan_amount / new_house_cost) * 100
  loan_percentage_computed = loan_percentage :=
by
  -- Proof placeholder
  sorry

end loan_percentage_correct_l453_45338


namespace sum_of_squares_of_projections_constant_l453_45398

-- Define the sum of the squares of projections function
noncomputable def sum_of_squares_of_projections (a : ℝ) (α : ℝ) : ℝ :=
  let p1 := a * Real.cos α
  let p2 := a * Real.cos (Real.pi / 3 - α)
  let p3 := a * Real.cos (Real.pi / 3 + α)
  p1^2 + p2^2 + p3^2

-- Statement of the theorem
theorem sum_of_squares_of_projections_constant (a α : ℝ) : 
  sum_of_squares_of_projections a α = 3 / 2 * a^2 :=
sorry

end sum_of_squares_of_projections_constant_l453_45398


namespace continuous_function_identity_l453_45373

theorem continuous_function_identity (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_func_eq : ∀ x y : ℝ, 2 * f (x + y) = f x * f y)
  (h_f1 : f 1 = 10) :
  ∀ x : ℝ, f x = 2 * 5^x :=
by
  sorry

end continuous_function_identity_l453_45373


namespace final_pressure_of_helium_l453_45309

theorem final_pressure_of_helium
  (p v v' : ℝ) (k : ℝ)
  (h1 : p = 4)
  (h2 : v = 3)
  (h3 : v' = 6)
  (h4 : p * v = k)
  (h5 : ∀ p' : ℝ, p' * v' = k → p' = 2) :
  p' = 2 := by
  sorry

end final_pressure_of_helium_l453_45309


namespace modulo_17_residue_l453_45341

theorem modulo_17_residue : (392 + 6 * 51 + 8 * 221 + 3^2 * 23) % 17 = 11 :=
by 
  sorry

end modulo_17_residue_l453_45341


namespace solve_for_x_l453_45315

theorem solve_for_x : ∀ (x : ℕ), (y = 2 / (4 * x + 2)) → (y = 1 / 2) → (x = 1/2) :=
by
  sorry

end solve_for_x_l453_45315


namespace distance_squared_l453_45336

noncomputable def circumcircle_radius (R : ℝ) : Prop := sorry
noncomputable def excircle_radius (p : ℝ) : Prop := sorry
noncomputable def distance_between_centers (d : ℝ) (R : ℝ) (p : ℝ) : Prop := sorry

theorem distance_squared (R p d : ℝ) (h1 : circumcircle_radius R) (h2 : excircle_radius p) (h3 : distance_between_centers d R p) :
  d^2 = R^2 + 2 * R * p := sorry

end distance_squared_l453_45336


namespace find_constant_l453_45352

theorem find_constant (t : ℝ) (constant : ℝ) :
  (x = constant - 3 * t) → (y = 2 * t - 3) → (t = 0.8) → (x = y) → constant = 1 :=
by
  intros h1 h2 h3 h4
  sorry

end find_constant_l453_45352


namespace selling_price_l453_45305

theorem selling_price (cost_price profit_percentage : ℝ) (h_cost : cost_price = 250) (h_profit : profit_percentage = 0.60) :
  cost_price + profit_percentage * cost_price = 400 := sorry

end selling_price_l453_45305


namespace coprime_divisibility_l453_45387

theorem coprime_divisibility (p q r P Q R : ℕ)
  (hpq : Nat.gcd p q = 1) (hpr : Nat.gcd p r = 1) (hqr : Nat.gcd q r = 1)
  (h : ∃ k : ℤ, (P:ℤ) * (q*r) + (Q:ℤ) * (p*r) + (R:ℤ) * (p*q) = k * (p*q * r)) :
  ∃ a b c : ℤ, (P:ℤ) = a * (p:ℤ) ∧ (Q:ℤ) = b * (q:ℤ) ∧ (R:ℤ) = c * (r:ℤ) :=
by
  sorry

end coprime_divisibility_l453_45387


namespace find_x_l453_45319

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l453_45319


namespace grandmother_age_five_times_lingling_l453_45342

theorem grandmother_age_five_times_lingling (x : ℕ) :
  let lingling_age := 8
  let grandmother_age := 60
  (grandmother_age + x = 5 * (lingling_age + x)) ↔ (x = 5) := by
  sorry

end grandmother_age_five_times_lingling_l453_45342


namespace line_y_intercept_l453_45323

theorem line_y_intercept (t : ℝ) (h : ∃ (t : ℝ), ∀ (x y : ℝ), x - 2 * y + t = 0 → (x = 2 ∧ y = -1)) :
  ∃ y : ℝ, (0 - 2 * y + t = 0) ∧ y = -2 :=
by
  sorry

end line_y_intercept_l453_45323


namespace complement_eq_target_l453_45335

namespace ComplementProof

-- Define the universal set U
def U : Set ℕ := {2, 4, 6, 8, 10}

-- Define the set A
def A : Set ℕ := {2, 6, 8}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x ∈ U | x ∉ A}

-- Define the target set
def target_set : Set ℕ := {4, 10}

-- Prove that the complement of A with respect to U is equal to {4, 10}
theorem complement_eq_target :
  complement_U_A = target_set := by sorry

end ComplementProof

end complement_eq_target_l453_45335


namespace find_a_l453_45349

theorem find_a 
  (a b c : ℤ) 
  (h_vertex : ∀ x, (a * (x - 2)^2 + 5 = a * x^2 + b * x + c))
  (h_point : ∀ y, y = a * (1 - 2)^2 + 5)
  : a = -1 := by
  sorry

end find_a_l453_45349


namespace square_garden_perimeter_l453_45354

theorem square_garden_perimeter (A : ℝ) (h : A = 450) : ∃ P : ℝ, P = 60 * Real.sqrt 2 :=
  sorry

end square_garden_perimeter_l453_45354


namespace circle_equation_through_points_l453_45370

-- Line and circle definitions
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 15 = 0

-- Intersection point definition
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ circle1 x y

-- Revised circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 28 * x - 15 * y = 0

-- Proof statement
theorem circle_equation_through_points :
  (∀ x y, intersection_point x y → circle_equation x y) ∧ circle_equation 0 0 :=
sorry

end circle_equation_through_points_l453_45370


namespace find_x_from_roots_l453_45390

variable (x m : ℕ)

theorem find_x_from_roots (h1 : (m + 3)^2 = x) (h2 : (2 * m - 15)^2 = x) : x = 49 := by
  sorry

end find_x_from_roots_l453_45390


namespace overall_gain_percent_l453_45385

theorem overall_gain_percent (cp1 cp2 cp3: ℝ) (sp1 sp2 sp3: ℝ) (h1: cp1 = 840) (h2: cp2 = 1350) (h3: cp3 = 2250) (h4: sp1 = 1220) (h5: sp2 = 1550) (h6: sp3 = 2150) : 
  (sp1 + sp2 + sp3 - (cp1 + cp2 + cp3)) / (cp1 + cp2 + cp3) * 100 = 10.81 := 
by 
  sorry

end overall_gain_percent_l453_45385


namespace mul_same_base_exp_ten_pow_1000_sq_l453_45380

theorem mul_same_base_exp (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by
  sorry

-- Given specific constants for this problem
theorem ten_pow_1000_sq : (10:ℝ)^(1000) * (10)^(1000) = (10)^(2000) := by
  exact mul_same_base_exp 10 1000 1000

end mul_same_base_exp_ten_pow_1000_sq_l453_45380


namespace sandy_books_from_second_shop_l453_45318

noncomputable def books_from_second_shop (books_first: ℕ) (cost_first: ℕ) (cost_second: ℕ) (avg_price: ℕ): ℕ :=
  let total_cost := cost_first + cost_second
  let total_books := books_first + (total_cost / avg_price) - books_first
  total_cost / avg_price - books_first

theorem sandy_books_from_second_shop :
  books_from_second_shop 65 1380 900 19 = 55 :=
by
  sorry

end sandy_books_from_second_shop_l453_45318


namespace eight_div_pow_64_l453_45345

theorem eight_div_pow_64 (h : 64 = 8^2) : 8^15 / 64^7 = 8 := by
  sorry

end eight_div_pow_64_l453_45345


namespace range_of_a_decreasing_l453_45326

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a / x

def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≥ f y

theorem range_of_a_decreasing (a : ℝ) :
  (∃ a : ℝ, (1/6) ≤ a ∧ a < (1/3)) ↔ is_decreasing (f a) :=
sorry

end range_of_a_decreasing_l453_45326


namespace initial_cake_pieces_l453_45346

-- Define the initial number of cake pieces
variable (X : ℝ)

-- Define the conditions as assumptions
def cake_conditions (X : ℝ) : Prop :=
  0.60 * X + 3 * 32 = X 

theorem initial_cake_pieces (X : ℝ) (h : cake_conditions X) : X = 240 := sorry

end initial_cake_pieces_l453_45346


namespace circle_equation_l453_45327

/-- Given a circle passing through points P(4, -2) and Q(-1, 3), and with the length of the segment 
intercepted by the circle on the y-axis as 4, prove that the standard equation of the circle
is (x-1)^2 + y^2 = 13 or (x-5)^2 + (y-4)^2 = 37 -/
theorem circle_equation {P Q : ℝ × ℝ} {a b k : ℝ} :
  P = (4, -2) ∧ Q = (-1, 3) ∧ k = 4 →
  (∃ (r : ℝ), (∀ y : ℝ, (b - y)^2 = r^2) ∧
    ((a - 1)^2 + b^2 = 13 ∨ (a - 5)^2 + (b - 4)^2 = 37)
  ) :=
by
  sorry

end circle_equation_l453_45327


namespace number_tower_proof_l453_45394

theorem number_tower_proof : 123456 * 9 + 7 = 1111111 := 
  sorry

end number_tower_proof_l453_45394


namespace train_crossing_time_l453_45312

def speed_kmph : ℝ := 90
def length_train : ℝ := 225

noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)

theorem train_crossing_time : (length_train / speed_mps) = 9 := by
  sorry

end train_crossing_time_l453_45312


namespace all_children_receive_candy_l453_45339

-- Define f(x) function
def f (x n : ℕ) : ℕ := ((x * (x + 1)) / 2) % n

-- Define the problem statement: prove that all children receive at least one candy if n is a power of 2.
theorem all_children_receive_candy (n : ℕ) (h : ∃ m, n = 2^m) : 
    ∀ i : ℕ, i < n → ∃ x : ℕ, i = f x n := 
sorry

end all_children_receive_candy_l453_45339


namespace find_ratio_l453_45310

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Given conditions
axiom sum_arithmetic_a (n : ℕ) : S n = n / 2 * (a 1 + a n)
axiom sum_arithmetic_b (n : ℕ) : T n = n / 2 * (b 1 + b n)
axiom sum_ratios (n : ℕ) : S n / T n = (2 * n + 1) / (3 * n + 2)

-- The proof problem
theorem find_ratio : (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := 
sorry

end find_ratio_l453_45310


namespace parabola_vector_sum_distance_l453_45347

noncomputable def parabola_focus (x y : ℝ) : Prop := x^2 = 8 * y

noncomputable def on_parabola (x y : ℝ) : Prop := parabola_focus x y

theorem parabola_vector_sum_distance :
  ∀ (A B C : ℝ × ℝ) (F : ℝ × ℝ),
  on_parabola A.1 A.2 ∧ on_parabola B.1 B.2 ∧ on_parabola C.1 C.2 ∧
  F = (0, 2) ∧
  ((A.1 - F.1)^2 + (A.2 - F.2)^2) + ((B.1 - F.1)^2 + (B.2 - F.2)^2) + ((C.1 - F.1)^2 + (C.2 - F.2)^2) = 0
  → (abs ((A.2 + F.2)) + abs ((B.2 + F.2)) + abs ((C.2 + F.2))) = 12 :=
by sorry

end parabola_vector_sum_distance_l453_45347


namespace arcsin_one_half_eq_pi_over_six_arccos_one_half_eq_pi_over_three_l453_45300

theorem arcsin_one_half_eq_pi_over_six : Real.arcsin (1/2) = Real.pi/6 :=
by 
  sorry

theorem arccos_one_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi/3 :=
by 
  sorry

end arcsin_one_half_eq_pi_over_six_arccos_one_half_eq_pi_over_three_l453_45300


namespace geese_problem_l453_45330

theorem geese_problem 
  (G : ℕ)  -- Total number of geese in the original V formation
  (T : ℕ)  -- Number of geese that flew up from the trees to join the new V formation
  (h1 : G / 2 + T = 12)  -- Final number of geese flying in the V formation was 12 
  (h2 : T = G / 2)  -- Number of geese that flew out from the trees is the same as the number of geese that landed initially
: T = 6 := 
sorry

end geese_problem_l453_45330


namespace fraction_product_l453_45311

theorem fraction_product :
  (2 / 3) * (3 / 4) * (5 / 6) * (6 / 7) * (8 / 9) = 80 / 63 :=
by sorry

end fraction_product_l453_45311


namespace number_of_meetings_l453_45340

-- Definitions based on the given conditions
def track_circumference : ℕ := 300
def boy1_speed : ℕ := 7
def boy2_speed : ℕ := 3
def both_start_simultaneously := true

-- The theorem to prove
theorem number_of_meetings (h1 : track_circumference = 300) (h2 : boy1_speed = 7) (h3 : boy2_speed = 3) (h4 : both_start_simultaneously) : 
  ∃ n : ℕ, n = 1 := 
sorry

end number_of_meetings_l453_45340


namespace no_divisor_form_24k_20_l453_45324

theorem no_divisor_form_24k_20 (n : ℕ) : ¬ ∃ k : ℕ, 24 * k + 20 ∣ 3^n + 1 :=
sorry

end no_divisor_form_24k_20_l453_45324


namespace bisection_interval_length_l453_45353

theorem bisection_interval_length (n : ℕ) : 
  (1 / (2:ℝ)^n) ≤ 0.01 → n ≥ 7 :=
by 
  sorry

end bisection_interval_length_l453_45353


namespace ten_thousands_written_correctly_ten_thousands_truncated_correctly_l453_45350

-- Definitions to be used in the proof
def ten_thousands_description := "Three thousand nine hundred seventy-six ten thousands"
def num_written : ℕ := 39760000
def truncated_num : ℕ := 3976

-- Theorems to be proven
theorem ten_thousands_written_correctly :
  (num_written = 39760000) :=
sorry

theorem ten_thousands_truncated_correctly :
  (truncated_num = 3976) :=
sorry

end ten_thousands_written_correctly_ten_thousands_truncated_correctly_l453_45350


namespace parabola_distance_l453_45321

theorem parabola_distance (p : ℝ) : 
  (∃ p: ℝ, y^2 = 10*x ∧ 2*p = 10) → p = 5 :=
by
  sorry

end parabola_distance_l453_45321


namespace probability_in_smaller_spheres_l453_45304

theorem probability_in_smaller_spheres 
    (R r : ℝ)
    (h_eq : ∀ (R r : ℝ), R + r = 4 * r)
    (vol_eq : ∀ (R r : ℝ), (4/3) * π * r^3 * 5 = (4/3) * π * R^3 * (5/27)) :
    P = 0.2 := by
  sorry

end probability_in_smaller_spheres_l453_45304


namespace smallest_k_divides_ab_l453_45371

theorem smallest_k_divides_ab (S : Finset ℕ) (hS : S = Finset.range 51)
  (k : ℕ) : (∀ T : Finset ℕ, T ⊆ S → T.card = k → ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ (a + b) ∣ (a * b)) ↔ k = 39 :=
by
  sorry

end smallest_k_divides_ab_l453_45371


namespace greatest_cars_with_ac_not_racing_stripes_l453_45396

def total_cars : ℕ := 100
def without_ac : ℕ := 49
def at_least_racing_stripes : ℕ := 51

theorem greatest_cars_with_ac_not_racing_stripes :
  (total_cars - without_ac) - (at_least_racing_stripes - without_ac) = 49 :=
by
  unfold total_cars without_ac at_least_racing_stripes
  sorry

end greatest_cars_with_ac_not_racing_stripes_l453_45396


namespace added_number_is_6_l453_45386

theorem added_number_is_6 : ∃ x : ℤ, (∃ y : ℤ, y = 9 ∧ (2 * y + x) * 3 = 72) → x = 6 := 
by
  sorry

end added_number_is_6_l453_45386


namespace perimeter_of_8_sided_figure_l453_45399

theorem perimeter_of_8_sided_figure (n : ℕ) (len : ℕ) (h1 : n = 8) (h2 : len = 2) :
  n * len = 16 := by
  sorry

end perimeter_of_8_sided_figure_l453_45399


namespace solve_for_x_l453_45314

theorem solve_for_x (x : ℝ) (h : x + 3 * x = 500 - (4 * x + 5 * x)) : x = 500 / 13 := 
by 
  sorry

end solve_for_x_l453_45314


namespace piglet_weight_l453_45376

variable (C K P L : ℝ)

theorem piglet_weight (h1 : C = K + P) (h2 : P + C = L + K) (h3 : L = 30) : P = 15 := by
  sorry

end piglet_weight_l453_45376


namespace negation_of_proposition_l453_45392

-- Given condition
def original_statement (a : ℝ) : Prop :=
  ∃ x : ℝ, a*x^2 - 2*a*x + 1 ≤ 0

-- Correct answer (negation statement)
def negated_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - 2*a*x + 1 > 0

-- Statement to prove
theorem negation_of_proposition (a : ℝ) :
  ¬ (original_statement a) ↔ (negated_statement a) :=
by 
  sorry

end negation_of_proposition_l453_45392


namespace restaurant_vegetarian_dishes_l453_45337

theorem restaurant_vegetarian_dishes (n : ℕ) : 
    5 ≥ 2 → 200 < Nat.choose 5 2 * Nat.choose n 2 → n ≥ 7 :=
by
  intros h_combinations h_least
  sorry

end restaurant_vegetarian_dishes_l453_45337


namespace rectangle_image_l453_45344

-- A mathematically equivalent Lean 4 proof problem statement

variable (x y : ℝ)

def rectangle_OABC (x y : ℝ) : Prop :=
  (x = 0 ∧ (0 ≤ y ∧ y ≤ 3)) ∨
  (y = 0 ∧ (0 ≤ x ∧ x ≤ 2)) ∨
  (x = 2 ∧ (0 ≤ y ∧ y ≤ 3)) ∨
  (y = 3 ∧ (0 ≤ x ∧ x ≤ 2))

def transform_u (x y : ℝ) : ℝ := x^2 - y^2 + 1
def transform_v (x y : ℝ) : ℝ := x * y

theorem rectangle_image (u v : ℝ) :
  (∃ (x y : ℝ), rectangle_OABC x y ∧ u = transform_u x y ∧ v = transform_v x y) ↔
  (u, v) = (-8, 0) ∨
  (u, v) = (1, 0) ∨
  (u, v) = (5, 0) ∨
  (u, v) = (-4, 6) :=
sorry

end rectangle_image_l453_45344


namespace D_double_prime_coordinates_l453_45334

-- The coordinates of points A, B, C, D as given in the problem
def A : (ℝ × ℝ) := (3, 6)
def B : (ℝ × ℝ) := (5, 10)
def C : (ℝ × ℝ) := (7, 6)
def D : (ℝ × ℝ) := (5, 2)

-- Reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def D' : ℝ × ℝ := reflect_x D

-- Translate the point (x, y) by (dx, dy)
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

-- Reflect across the line y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Combined translation and reflection across y = x + 2
def reflect_y_eq_x_plus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p_translated := translate p 0 (-2)
  let p_reflected := reflect_y_eq_x p_translated
  translate p_reflected 0 2

def D'' : ℝ × ℝ := reflect_y_eq_x_plus_2 D'

theorem D_double_prime_coordinates : D'' = (-4, 7) := by
  sorry

end D_double_prime_coordinates_l453_45334


namespace prop_disjunction_is_true_l453_45355

variable (p q : Prop)
axiom hp : p
axiom hq : ¬q

theorem prop_disjunction_is_true (hp : p) (hq : ¬q) : p ∨ q :=
by
  sorry

end prop_disjunction_is_true_l453_45355


namespace tina_total_leftover_l453_45325

def monthly_income : ℝ := 1000

def june_savings : ℝ := 0.25 * monthly_income
def june_expenses : ℝ := 200 + 0.05 * monthly_income
def june_leftover : ℝ := monthly_income - june_savings - june_expenses

def july_savings : ℝ := 0.20 * monthly_income
def july_expenses : ℝ := 250 + 0.15 * monthly_income
def july_leftover : ℝ := monthly_income - july_savings - july_expenses

def august_savings : ℝ := 0.30 * monthly_income
def august_expenses : ℝ := 250 + 50 + 0.10 * monthly_income
def august_gift : ℝ := 50
def august_leftover : ℝ := (monthly_income - august_savings - august_expenses) + august_gift

def total_leftover : ℝ :=
  june_leftover + july_leftover + august_leftover

theorem tina_total_leftover (I : ℝ) (hI : I = 1000) :
  total_leftover = 1250 := by
  rw [←hI] at *
  show total_leftover = 1250
  sorry

end tina_total_leftover_l453_45325


namespace probability_sum_of_10_l453_45362

theorem probability_sum_of_10 (total_outcomes : ℕ) 
  (h1 : total_outcomes = 6^4) : 
  (46 / total_outcomes) = 23 / 648 := by
  sorry

end probability_sum_of_10_l453_45362


namespace books_problem_l453_45316

theorem books_problem
  (M H : ℕ)
  (h1 : M + H = 80)
  (h2 : 4 * M + 5 * H = 390) :
  M = 10 :=
by
  sorry

end books_problem_l453_45316


namespace blake_change_given_l453_45372

theorem blake_change_given :
  let oranges := 40
  let apples := 50
  let mangoes := 60
  let total_amount := 300
  let total_spent := oranges + apples + mangoes
  let change_given := total_amount - total_spent
  change_given = 150 :=
by
  sorry

end blake_change_given_l453_45372


namespace probability_inequality_up_to_99_l453_45364

theorem probability_inequality_up_to_99 :
  (∀ x : ℕ, 1 ≤ x ∧ x < 100 → (2^x / x!) > x^2) →
    (∃ n : ℕ, (1 ≤ n ∧ n < 100) ∧ (2^n / n!) > n^2) →
      ∃ p : ℚ, p = 1/99 :=
by
  sorry

end probability_inequality_up_to_99_l453_45364


namespace total_coffee_consumed_l453_45333

def Ivory_hourly_coffee := 2
def Kimberly_hourly_coffee := Ivory_hourly_coffee
def Brayan_hourly_coffee := 4
def Raul_hourly_coffee := Brayan_hourly_coffee / 2
def duration_hours := 10

theorem total_coffee_consumed :
  (Brayan_hourly_coffee * duration_hours) + 
  (Ivory_hourly_coffee * duration_hours) + 
  (Kimberly_hourly_coffee * duration_hours) + 
  (Raul_hourly_coffee * duration_hours) = 100 :=
by sorry

end total_coffee_consumed_l453_45333


namespace ratio_constant_l453_45361

theorem ratio_constant (a b c d : ℕ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d)
    (h : ∀ k : ℕ, ∃ m : ℤ, a + c * k = m * (b + d * k)) :
    ∃ m : ℤ, ∀ k : ℕ, a + c * k = m * (b + d * k) :=
    sorry

end ratio_constant_l453_45361


namespace theodore_pays_10_percent_in_taxes_l453_45388

-- Defining the quantities
def num_stone_statues : ℕ := 10
def num_wooden_statues : ℕ := 20
def price_per_stone_statue : ℕ := 20
def price_per_wooden_statue : ℕ := 5
def total_earnings_after_taxes : ℕ := 270

-- Assertion: Theodore pays 10% of his earnings in taxes
theorem theodore_pays_10_percent_in_taxes :
  (num_stone_statues * price_per_stone_statue + num_wooden_statues * price_per_wooden_statue) - total_earnings_after_taxes
  = (10 * (num_stone_statues * price_per_stone_statue + num_wooden_statues * price_per_wooden_statue)) / 100 := 
by
  sorry

end theodore_pays_10_percent_in_taxes_l453_45388


namespace min_value_f_exists_min_value_f_l453_45393

noncomputable def f (a b c : ℝ) := 1 / (b^2 + b * c) + 1 / (c^2 + c * a) + 1 / (a^2 + a * b)

theorem min_value_f (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) : f a b c ≥ 3 / 2 :=
  sorry

theorem exists_min_value_f : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧ f a b c = 3 / 2 :=
  sorry

end min_value_f_exists_min_value_f_l453_45393


namespace var_power_l453_45377

theorem var_power {a b c x y z : ℝ} (h1 : x = a * y^4) (h2 : y = b * z^(1/3)) :
  ∃ n : ℝ, x = c * z^n ∧ n = 4/3 := by
  sorry

end var_power_l453_45377
