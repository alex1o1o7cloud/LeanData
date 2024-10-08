import Mathlib

namespace patricia_candies_final_l220_220716

def initial_candies : ℕ := 764
def taken_candies : ℕ := 53
def back_candies_per_7_taken : ℕ := 19

theorem patricia_candies_final :
  let given_back_times := taken_candies / 7
  let total_given_back := given_back_times * back_candies_per_7_taken
  let final_candies := initial_candies - taken_candies + total_given_back
  final_candies = 844 :=
by
  sorry

end patricia_candies_final_l220_220716


namespace time_expression_l220_220411

theorem time_expression (h V₀ g S V t : ℝ) :
  (V = g * t + V₀) →
  (S = h + (1 / 2) * g * t^2 + V₀ * t) →
  t = (2 * (S - h)) / (V + V₀) :=
by
  intro h_eq v_eq
  sorry

end time_expression_l220_220411


namespace derivative_of_y_l220_220285

variable (x : ℝ)

def y := x^3 + 3 * x^2 + 6 * x - 10

theorem derivative_of_y : (deriv y) x = 3 * x^2 + 6 * x + 6 :=
sorry

end derivative_of_y_l220_220285


namespace mens_wages_l220_220829

-- Definitions based on the problem conditions
def equivalent_wages (M W_earn B : ℝ) : Prop :=
  (5 * M = W_earn) ∧ 
  (W_earn = 8 * B) ∧ 
  (5 * M + W_earn + 8 * B = 210)

-- Prove that the total wages of 5 men are Rs. 105 given the conditions
theorem mens_wages (M W_earn B : ℝ) (h : equivalent_wages M W_earn B) : 5 * M = 105 :=
by
  sorry

end mens_wages_l220_220829


namespace max_M_inequality_l220_220625

theorem max_M_inequality :
  ∃ M : ℝ, (∀ x y : ℝ, x + y ≥ 0 → (x^2 + y^2)^3 ≥ M * (x^3 + y^3) * (x * y - x - y)) ∧ M = 32 :=
by {
  sorry
}

end max_M_inequality_l220_220625


namespace table_can_be_zeroed_out_l220_220823

open Matrix

-- Define the dimensions of the table
def m := 8
def n := 5

-- Define the operation of doubling all elements in a row
def double_row (table : Matrix (Fin m) (Fin n) ℕ) (i : Fin m) : Matrix (Fin m) (Fin n) ℕ :=
  fun i' j => if i' = i then 2 * table i' j else table i' j

-- Define the operation of subtracting one from all elements in a column
def subtract_one_column (table : Matrix (Fin m) (Fin n) ℕ) (j : Fin n) : Matrix (Fin m) (Fin n) ℕ :=
  fun i j' => if j' = j then table i j' - 1 else table i j'

-- The main theorem stating that it is possible to transform any table to a table of all zeros
theorem table_can_be_zeroed_out (table : Matrix (Fin m) (Fin n) ℕ) : 
  ∃ (ops : List (Matrix (Fin m) (Fin n) ℕ → Matrix (Fin m) (Fin n) ℕ)), 
    (ops.foldl (fun t op => op t) table) = fun _ _ => 0 :=
sorry

end table_can_be_zeroed_out_l220_220823


namespace regular_price_of_shrimp_l220_220908

theorem regular_price_of_shrimp 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) 
  (quarter_pound_price : ℝ) 
  (full_pound_price : ℝ) 
  (price_relation : quarter_pound_price = discounted_price * (1 - discount_rate) / 4) 
  (discounted_value : quarter_pound_price = 2) 
  (given_discount_rate : discount_rate = 0.6) 
  (given_discounted_price : discounted_price = full_pound_price) 
  : full_pound_price = 20 :=
by {
  sorry
}

end regular_price_of_shrimp_l220_220908


namespace problem1_problem2_problem3_l220_220046

-- 1. Given: ∃ x ∈ ℤ, x^2 - 2x - 3 = 0
--    Show: ∀ x ∈ ℤ, x^2 - 2x - 3 ≠ 0
theorem problem1 : (∃ x : ℤ, x^2 - 2 * x - 3 = 0) ↔ (∀ x : ℤ, x^2 - 2 * x - 3 ≠ 0) := sorry

-- 2. Given: ∀ x ∈ ℝ, x^2 + 3 ≥ 2x
--    Show: ∃ x ∈ ℝ, x^2 + 3 < 2x
theorem problem2 : (∀ x : ℝ, x^2 + 3 ≥ 2 * x) ↔ (∃ x : ℝ, x^2 + 3 < 2 * x) := sorry

-- 3. Given: If x > 1 and y > 1, then x + y > 2
--    Show: If x ≤ 1 or y ≤ 1, then x + y ≤ 2
theorem problem3 : (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ↔ (∀ x y : ℝ, x ≤ 1 ∨ y ≤ 1 → x + y ≤ 2) := sorry

end problem1_problem2_problem3_l220_220046


namespace minimum_value_f_l220_220518

noncomputable def f (x : ℝ) : ℝ := (x^2 / 8) + x * (Real.cos x) + (Real.cos (2 * x))

theorem minimum_value_f : ∃ x : ℝ, f x = -1 :=
by {
  sorry
}

end minimum_value_f_l220_220518


namespace smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6_l220_220406

theorem smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6 :
  ∃ n : ℤ, n = 3323 ∧ n > (Real.sqrt 5 + Real.sqrt 3)^6 ∧ ∀ m : ℤ, m > (Real.sqrt 5 + Real.sqrt 3)^6 → n ≤ m :=
by
  sorry

end smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6_l220_220406


namespace nail_pierces_one_cardboard_only_l220_220006

/--
Seryozha cut out two identical figures from cardboard. He placed them overlapping
at the bottom of a rectangular box. The bottom turned out to be completely covered. 
A nail was driven into the center of the bottom. Prove that it is possible for the 
nail to pierce one cardboard piece without piercing the other.
-/
theorem nail_pierces_one_cardboard_only 
  (identical_cardboards : Prop)
  (overlapping : Prop)
  (fully_covered_bottom : Prop)
  (nail_center : Prop) 
  : ∃ (layout : Prop), layout ∧ nail_center → nail_pierces_one :=
sorry

end nail_pierces_one_cardboard_only_l220_220006


namespace conference_games_l220_220924

/-- 
Two divisions of 8 teams each, where each team plays 21 games within its division 
and 8 games against the teams of the other division. 
Prove total number of scheduled conference games is 232.
-/
theorem conference_games (div_teams : ℕ) (intra_div_games : ℕ) (inter_div_games : ℕ) (total_teams : ℕ) :
  div_teams = 8 →
  intra_div_games = 21 →
  inter_div_games = 8 →
  total_teams = 2 * div_teams →
  (total_teams * (intra_div_games + inter_div_games)) / 2 = 232 :=
by
  intros
  sorry


end conference_games_l220_220924


namespace loss_percentage_is_ten_l220_220988

variable (CP SP SP_new : ℝ)  -- introduce the cost price, selling price, and new selling price as variables

theorem loss_percentage_is_ten
  (h1 : CP = 2000)
  (h2 : SP_new = CP + 80)
  (h3 : SP_new = SP + 280)
  (h4 : SP = CP - (L / 100 * CP)) : L = 10 :=
by
  -- proof goes here
  sorry

end loss_percentage_is_ten_l220_220988


namespace correct_M_min_t_for_inequality_l220_220643

-- Define the set M
def M : Set ℝ := {a | 0 ≤ a ∧ a < 4}

-- Prove that M is correct given ax^2 + ax + 2 > 0 for all x ∈ ℝ implies 0 ≤ a < 4
theorem correct_M (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

-- Prove the minimum value of t given t > 0 and the inequality holds for all a ∈ M
theorem min_t_for_inequality (t : ℝ) (h : 0 < t) : 
  (∀ a ∈ M, (a^2 - 2 * a) * t ≤ t^2 + 3 * t - 46) ↔ 46 ≤ t :=
sorry

end correct_M_min_t_for_inequality_l220_220643


namespace difference_between_numbers_l220_220938

theorem difference_between_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 190) : x - y = 19 :=
by sorry

end difference_between_numbers_l220_220938


namespace value_of_expression_l220_220473

theorem value_of_expression (a : ℝ) (h : a^2 - 2 * a = 1) : 3 * a^2 - 6 * a - 4 = -1 :=
by
  sorry

end value_of_expression_l220_220473


namespace larger_number_is_26_l220_220471

theorem larger_number_is_26 {x y : ℤ} 
  (h1 : x + y = 45) 
  (h2 : x - y = 7) : 
  max x y = 26 :=
by
  sorry

end larger_number_is_26_l220_220471


namespace necessarily_negative_sum_l220_220443

theorem necessarily_negative_sum 
  (u v w : ℝ)
  (hu : -1 < u ∧ u < 0)
  (hv : 0 < v ∧ v < 1)
  (hw : -2 < w ∧ w < -1) :
  v + w < 0 :=
sorry

end necessarily_negative_sum_l220_220443


namespace divisors_of_90_l220_220274

def num_pos_divisors (n : ℕ) : ℕ :=
  let factors := if n = 90 then [(2, 1), (3, 2), (5, 1)] else []
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

theorem divisors_of_90 : num_pos_divisors 90 = 12 := by
  sorry

end divisors_of_90_l220_220274


namespace perpendicular_line_slope_l220_220137

theorem perpendicular_line_slope (m : ℝ) 
  (h1 : ∀ x y : ℝ, x - 2 * y + 5 = 0 → x = 2 * y - 5)
  (h2 : ∀ x y : ℝ, 2 * x + m * y - 6 = 0 → y = - (2 / m) * x + 6 / m)
  (h3 : (1 / 2 : ℝ) * - (2 / m) = -1) : m = 1 :=
sorry

end perpendicular_line_slope_l220_220137


namespace square_area_l220_220382

theorem square_area (XY ZQ : ℕ) (inscribed_square : Prop) : (XY = 35) → (ZQ = 65) → inscribed_square → ∃ (a : ℕ), a^2 = 2275 :=
by
  intros hXY hZQ hinscribed
  use 2275
  sorry

end square_area_l220_220382


namespace infinite_series_sum_l220_220695

theorem infinite_series_sum : 
  ∑' k : ℕ, (5^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 := 
sorry

end infinite_series_sum_l220_220695


namespace cost_of_each_shirt_l220_220623

theorem cost_of_each_shirt
  (x : ℝ) 
  (h : 3 * x + 2 * 20 = 85) : x = 15 :=
sorry

end cost_of_each_shirt_l220_220623


namespace determine_value_of_x_l220_220014

theorem determine_value_of_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 2 = 6 * y) : x = 48 :=
by
  sorry

end determine_value_of_x_l220_220014


namespace two_is_four_percent_of_fifty_l220_220259

theorem two_is_four_percent_of_fifty : (2 / 50) * 100 = 4 := 
by
  sorry

end two_is_four_percent_of_fifty_l220_220259


namespace find_theta_l220_220896

def rectangle : Type := sorry
def angle (α : ℝ) : Prop := 0 ≤ α ∧ α < 180

-- Given conditions in the problem
variables {α β γ δ θ : ℝ}

axiom angle_10 : angle 10
axiom angle_14 : angle 14
axiom angle_33 : angle 33
axiom angle_26 : angle 26

axiom zig_zag_angles (a b c d e f : ℝ) :
  a = 26 ∧ f = 10 ∧
  26 + b = 33 ∧ b = 7 ∧
  e + 10 = 14 ∧ e = 4 ∧
  c = b ∧ d = e ∧
  θ = c + d

theorem find_theta : θ = 11 :=
sorry

end find_theta_l220_220896


namespace digit_positions_in_8008_l220_220423

theorem digit_positions_in_8008 :
  (8008 % 10 = 8) ∧ (8008 / 1000 % 10 = 8) :=
by
  sorry

end digit_positions_in_8008_l220_220423


namespace ethanol_percentage_fuel_B_l220_220335

noncomputable def percentage_ethanol_in_fuel_B : ℝ :=
  let tank_capacity := 208
  let ethanol_in_fuelA := 0.12
  let total_ethanol := 30
  let volume_fuelA := 82
  let ethanol_from_fuelA := volume_fuelA * ethanol_in_fuelA
  let ethanol_from_fuelB := total_ethanol - ethanol_from_fuelA
  let volume_fuelB := tank_capacity - volume_fuelA
  (ethanol_from_fuelB / volume_fuelB) * 100

theorem ethanol_percentage_fuel_B :
  percentage_ethanol_in_fuel_B = 16 :=
by
  sorry

end ethanol_percentage_fuel_B_l220_220335


namespace sum_of_roots_l220_220238

variable {h b : ℝ}
variable {x₁ x₂ : ℝ}

-- Definition of the distinct property
def distinct (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂

-- Definition of the original equations given the conditions
def satisfies_equation (x : ℝ) (h b : ℝ) : Prop := 3 * x^2 - h * x = b

-- Main theorem statement translating the given mathematical problem
theorem sum_of_roots (h b : ℝ) (x₁ x₂ : ℝ) (h₁ : satisfies_equation x₁ h b) 
  (h₂ : satisfies_equation x₂ h b) (h₃ : distinct x₁ x₂) : x₁ + x₂ = h / 3 :=
sorry

end sum_of_roots_l220_220238


namespace Sahil_transportation_charges_l220_220244

theorem Sahil_transportation_charges
  (cost_machine : ℝ)
  (cost_repair : ℝ)
  (actual_selling_price : ℝ)
  (profit_percentage : ℝ)
  (transportation_charges : ℝ)
  (h1 : cost_machine = 12000)
  (h2 : cost_repair = 5000)
  (h3 : profit_percentage = 0.50)
  (h4 : actual_selling_price = 27000)
  (h5 : transportation_charges + (cost_machine + cost_repair) * (1 + profit_percentage) = actual_selling_price) :
  transportation_charges = 1500 :=
by
  sorry

end Sahil_transportation_charges_l220_220244


namespace triangle_inequality_l220_220129

variable {α : Type*} [LinearOrderedField α]

/-- Given a triangle ABC with sides a, b, c, circumradius R, 
exradii r_a, r_b, r_c, and given 2R ≤ r_a, we need to show that a > b, a > c, 2R > r_b, and 2R > r_c. -/
theorem triangle_inequality (a b c R r_a r_b r_c : α) (h₁ : 2 * R ≤ r_a) :
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := by
  sorry

end triangle_inequality_l220_220129


namespace find_starting_number_l220_220727

theorem find_starting_number (k m : ℕ) (hk : 67 = (m - k) / 3 + 1) (hm : m = 300) : k = 102 := by
  sorry

end find_starting_number_l220_220727


namespace card_sorting_moves_upper_bound_l220_220521

theorem card_sorting_moves_upper_bound (n : ℕ) (cells : Fin (n+1) → Fin (n+1)) (cards : Fin (n+1) → Fin (n+1)) : 
  (∃ (moves : (Fin (n+1) × Fin (n+1)) → ℕ),
    (∀ (i : Fin (n+1)), moves (i, cards i) ≤ 2 * n - 1) ∧ 
    (cards 0 = 0 → moves (0, 0) = 2 * n - 1) ∧ 
    (∃! start_pos : Fin (n+1) → Fin (n+1), 
      moves (start_pos (n), start_pos (0)) = 2 * n - 1)) := sorry

end card_sorting_moves_upper_bound_l220_220521


namespace recipe_sugar_amount_l220_220353

theorem recipe_sugar_amount (F_total F_added F_additional F_needed S : ℕ)
  (h1 : F_total = 9)
  (h2 : F_added = 2)
  (h3 : F_additional = S + 1)
  (h4 : F_needed = F_total - F_added)
  (h5 : F_needed = F_additional) :
  S = 6 := 
sorry

end recipe_sugar_amount_l220_220353


namespace part1_part2_l220_220801

-- Define the conditions that translate the quadratic equation having distinct real roots
def discriminant_condition (m : ℝ) : Prop :=
  let a := 1
  let b := -4
  let c := 3 - 2 * m
  b ^ 2 - 4 * a * c > 0

-- Define the root condition from Vieta's formulas and the additional given condition
def additional_condition (m : ℝ) : Prop :=
  let x1_plus_x2 := 4
  let x1_times_x2 := 3 - 2 * m
  x1_times_x2 + x1_plus_x2 - m^2 = 4

-- Prove the range of m for part 1
theorem part1 (m : ℝ) : discriminant_condition m → m ≥ -1/2 := by
  sorry

-- Prove the value of m for part 2 with the range condition
theorem part2 (m : ℝ) : discriminant_condition m → additional_condition m → m = 1 := by
  sorry

end part1_part2_l220_220801


namespace isosceles_triangle_perimeter_l220_220158

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_iso : a = b ∨ b = c ∨ c = a)
  (h_triangle_ineq1 : a + b > c) (h_triangle_ineq2 : b + c > a) (h_triangle_ineq3 : c + a > b)
  (h_sides : (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) :
  a + b + c = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l220_220158


namespace remainder_of_f_100_div_100_l220_220041

def pascal_triangle_row_sum (n : ℕ) : ℕ :=
  2^n - 2

theorem remainder_of_f_100_div_100 : 
  (pascal_triangle_row_sum 100) % 100 = 74 :=
by
  sorry

end remainder_of_f_100_div_100_l220_220041


namespace min_value_of_expression_l220_220624

noncomputable def minExpression (x : ℝ) : ℝ := (15 - x) * (14 - x) * (15 + x) * (14 + x)

theorem min_value_of_expression : ∀ x : ℝ, ∃ m : ℝ, (m ≤ minExpression x) ∧ (m = -142.25) :=
by
  sorry

end min_value_of_expression_l220_220624


namespace find_a_l220_220190

theorem find_a (a : ℝ) (h : ∃ x, x = -1 ∧ 4 * x^3 + 2 * a * x = 8) : a = -6 :=
sorry

end find_a_l220_220190


namespace percentage_of_alcohol_in_mixture_A_l220_220961

theorem percentage_of_alcohol_in_mixture_A (x : ℝ) :
  (10 * x / 100 + 5 * 50 / 100 = 15 * 30 / 100) → x = 20 :=
by
  intro h
  sorry

end percentage_of_alcohol_in_mixture_A_l220_220961


namespace total_fencing_l220_220004

def playground_side_length : ℕ := 27
def garden_length : ℕ := 12
def garden_width : ℕ := 9

def perimeter_square (side : ℕ) : ℕ := 4 * side
def perimeter_rectangle (length width : ℕ) : ℕ := 2 * length + 2 * width

theorem total_fencing (side playground_side_length : ℕ) (garden_length garden_width : ℕ) :
  perimeter_square playground_side_length + perimeter_rectangle garden_length garden_width = 150 :=
by
  sorry

end total_fencing_l220_220004


namespace find_x_l220_220749

theorem find_x (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1)
  (geom_seq : (x - ⌊x⌋) * x = ⌊x⌋^2) : x = 1.618 :=
by
  sorry

end find_x_l220_220749


namespace reflect_point_example_l220_220383

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def reflect_over_x_axis (P : Point3D) : Point3D :=
  { x := P.x, y := -P.y, z := -P.z }

theorem reflect_point_example :
  reflect_over_x_axis ⟨2, 3, 4⟩ = ⟨2, -3, -4⟩ :=
by
  -- Proof can be filled in here
  sorry

end reflect_point_example_l220_220383


namespace minimum_distance_PQ_l220_220434

open Real

noncomputable def minimum_distance (t : ℝ) : ℝ := 
  (|t - 1|) / (sqrt (1 + t ^ 2))

theorem minimum_distance_PQ :
  let t := sqrt 2 / 2
  let x_P := 2
  let y_P := 0
  let x_Q := -1 + t
  let y_Q := 2 + t
  let d := minimum_distance (x_Q - y_Q + 3)
  (d - 2) = (5 * sqrt 2) / 2 - 2 :=
sorry

end minimum_distance_PQ_l220_220434


namespace exists_c_gt_zero_l220_220627

theorem exists_c_gt_zero (a b : ℕ) (h_a_square_free : ¬ ∃ (k : ℕ), k^2 ∣ a)
    (h_b_square_free : ¬ ∃ (k : ℕ), k^2 ∣ b) (h_a_b_distinct : a ≠ b) :
    ∃ c > 0, ∀ n : ℕ, n > 0 →
    |(n * Real.sqrt a % 1) - (n * Real.sqrt b % 1)| > c / n^3 := sorry

end exists_c_gt_zero_l220_220627


namespace victor_total_money_l220_220334

def initial_amount : ℕ := 10
def allowance : ℕ := 8
def total_amount : ℕ := initial_amount + allowance

theorem victor_total_money : total_amount = 18 := by
  -- This is where the proof steps would go
  sorry

end victor_total_money_l220_220334


namespace correct_quadratic_opens_upwards_l220_220707

-- Define the quadratic functions
def A (x : ℝ) : ℝ := 1 - x - 6 * x^2
def B (x : ℝ) : ℝ := -8 * x + x^2 + 1
def C (x : ℝ) : ℝ := (1 - x) * (x + 5)
def D (x : ℝ) : ℝ := 2 - (5 - x)^2

-- The theorem stating that function B is the one that opens upwards
theorem correct_quadratic_opens_upwards :
  ∃ (f : ℝ → ℝ) (h : f = B), ∀ (a b c : ℝ), f x = a * x^2 + b * x + c → a > 0 :=
sorry

end correct_quadratic_opens_upwards_l220_220707


namespace parabola_line_no_intersection_l220_220934

theorem parabola_line_no_intersection (x y : ℝ) (h : y^2 < 4 * x) :
  ¬ ∃ (x' y' : ℝ), y' = y ∧ y'^2 = 4 * x' ∧ 2 * x' = x + x :=
by sorry

end parabola_line_no_intersection_l220_220934


namespace jimin_notebooks_proof_l220_220872

variable (m f o n : ℕ)

theorem jimin_notebooks_proof (hm : m = 7) (hf : f = 14) (ho : o = 33) (hn : n = o + m + f) :
  n - o = 21 := by
  sorry

end jimin_notebooks_proof_l220_220872


namespace lines_intersect_l220_220121

theorem lines_intersect (a b : ℝ) 
  (h₁ : ∃ y : ℝ, 4 = (3/4) * y + a ∧ y = 3)
  (h₂ : ∃ x : ℝ, 3 = (3/4) * x + b ∧ x = 4) :
  a + b = 7/4 :=
sorry

end lines_intersect_l220_220121


namespace total_number_of_coins_l220_220604

theorem total_number_of_coins (num_5c : Nat) (num_10c : Nat) (h1 : num_5c = 16) (h2 : num_10c = 16) : num_5c + num_10c = 32 := by
  sorry

end total_number_of_coins_l220_220604


namespace distance_after_3rd_turn_l220_220195

theorem distance_after_3rd_turn (d1 d2 d4 total_distance : ℕ) 
  (h1 : d1 = 5) 
  (h2 : d2 = 8) 
  (h4 : d4 = 0) 
  (h_total : total_distance = 23) : 
  total_distance - (d1 + d2 + d4) = 10 := 
  sorry

end distance_after_3rd_turn_l220_220195


namespace flat_path_time_l220_220373

/-- Malcolm's walking time problem -/
theorem flat_path_time (x : ℕ) (h1 : 6 + 12 + 6 = 24)
                       (h2 : 3 * x = 24 + 18) : x = 14 := 
by
  sorry

end flat_path_time_l220_220373


namespace product_of_N1_N2_l220_220433

theorem product_of_N1_N2 :
  (∃ (N1 N2 : ℤ),
    (∀ (x : ℚ),
      (47 * x - 35) * (x - 1) * (x - 2) = N1 * (x - 2) * (x - 1) + N2 * (x - 1) * (x - 2)) ∧
    N1 * N2 = -708) :=
sorry

end product_of_N1_N2_l220_220433


namespace min_value_ge_9_l220_220593

noncomputable def minValue (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 2)) : ℝ :=
  1 / (Real.sin θ) ^ 2 + 4 / (Real.cos θ) ^ 2

theorem min_value_ge_9 (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 2)) : minValue θ h ≥ 9 := 
  sorry

end min_value_ge_9_l220_220593


namespace min_socks_to_guarantee_10_pairs_l220_220479

/--
Given a drawer containing 100 red socks, 80 green socks, 60 blue socks, and 40 black socks, 
and socks are selected one at a time without seeing their color. 
The minimum number of socks that must be selected to guarantee at least 10 pairs is 23.
-/
theorem min_socks_to_guarantee_10_pairs 
  (red_socks green_socks blue_socks black_socks : ℕ) 
  (total_pairs : ℕ)
  (h_red : red_socks = 100)
  (h_green : green_socks = 80)
  (h_blue : blue_socks = 60)
  (h_black : black_socks = 40)
  (h_total_pairs : total_pairs = 10) :
  ∃ (n : ℕ), n = 23 := 
sorry

end min_socks_to_guarantee_10_pairs_l220_220479


namespace max_cars_per_div_100_is_20_l220_220176

theorem max_cars_per_div_100_is_20 :
  let m : ℕ := Nat.succ (Nat.succ 0) -- represents m going to infinity
  let car_length : ℕ := 5
  let speed_factor : ℕ := 10
  let sensor_distance_per_hour : ℕ := speed_factor * 1000 * m
  let separation_distance : ℕ := car_length * (m + 1)
  let max_cars : ℕ := (sensor_distance_per_hour / separation_distance) * m
  Nat.floor ((2 * (max_cars : ℝ)) / 100) = 20 :=
by
  sorry

end max_cars_per_div_100_is_20_l220_220176


namespace coeff_x20_greater_in_Q_l220_220332

noncomputable def coeff (f : ℕ → ℕ → ℤ) (p x : ℤ) : ℤ :=
(x ^ 20) * p

noncomputable def P (x : ℤ) := (1 - x^2 + x^3) ^ 1000
noncomputable def Q (x : ℤ) := (1 + x^2 - x^3) ^ 1000

theorem coeff_x20_greater_in_Q :
  coeff 20 (Q x) x > coeff 20 (P x) x :=
  sorry

end coeff_x20_greater_in_Q_l220_220332


namespace solve_inequality_l220_220572

theorem solve_inequality :
  {x : ℝ | 8*x^3 - 6*x^2 + 5*x - 5 < 0} = {x : ℝ | x < 1/2} :=
sorry

end solve_inequality_l220_220572


namespace total_birds_from_monday_to_wednesday_l220_220222

def birds_monday := 70
def birds_tuesday := birds_monday / 2
def birds_wednesday := birds_tuesday + 8
def total_birds := birds_monday + birds_tuesday + birds_wednesday

theorem total_birds_from_monday_to_wednesday : total_birds = 148 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end total_birds_from_monday_to_wednesday_l220_220222


namespace initially_calculated_average_height_l220_220231

theorem initially_calculated_average_height
  (A : ℝ)
  (h1 : ∀ heights : List ℝ, heights.length = 35 → (heights.sum + (106 - 166) = heights.sum) → (heights.sum / 35) = 180) :
  A = 181.71 :=
sorry

end initially_calculated_average_height_l220_220231


namespace flower_beds_fraction_l220_220196

-- Define the main problem parameters
def leg_length := (30 - 18) / 2
def triangle_area := (1 / 2) * (leg_length ^ 2)
def total_flower_bed_area := 2 * triangle_area
def yard_area := 30 * 6
def fraction_of_yard_occupied := total_flower_bed_area / yard_area

-- The theorem to be proved
theorem flower_beds_fraction :
  fraction_of_yard_occupied = 1/5 := by
  sorry

end flower_beds_fraction_l220_220196


namespace maria_earnings_l220_220161

-- Define the conditions
def costOfBrushes : ℕ := 20
def costOfCanvas : ℕ := 3 * costOfBrushes
def costPerLiterOfPaint : ℕ := 8
def litersOfPaintNeeded : ℕ := 5
def sellingPriceOfPainting : ℕ := 200

-- Define the total cost calculation
def totalCostOfMaterials : ℕ := costOfBrushes + costOfCanvas + (costPerLiterOfPaint * litersOfPaintNeeded)

-- Define the final earning calculation
def mariaEarning : ℕ := sellingPriceOfPainting - totalCostOfMaterials

-- State the theorem
theorem maria_earnings :
  mariaEarning = 80 := by
  sorry

end maria_earnings_l220_220161


namespace range_of_g_l220_220452

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + (Real.pi / 4) * (Real.arcsin (x / 3)) 
    - (Real.arcsin (x / 3))^2 + (Real.pi^2 / 16) * (x^2 + 2 * x + 3)

theorem range_of_g : 
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → 
  ∃ y, y = g x ∧ y ∈ (Set.Icc (Real.pi^2 / 4) (15 * Real.pi^2 / 16 + Real.pi / 4 * Real.arcsin 1)) :=
by
  sorry

end range_of_g_l220_220452


namespace max_b_c_plus_four_over_a_l220_220738

theorem max_b_c_plus_four_over_a (a b c : ℝ) (ha : a < 0)
  (h_quad : ∀ x : ℝ, -1 < x ∧ x < 2 → (a * x^2 + b * x + c) > 0) : 
  b - c + 4 / a ≤ -4 :=
sorry

end max_b_c_plus_four_over_a_l220_220738


namespace cube_volume_increase_l220_220368

theorem cube_volume_increase (s : ℝ) (h : s > 0) :
  let new_volume := (1.4 * s) ^ 3
  let original_volume := s ^ 3
  let increase_percentage := ((new_volume - original_volume) / original_volume) * 100
  increase_percentage = 174.4 := by
  sorry

end cube_volume_increase_l220_220368


namespace number_of_rectangles_with_one_gray_cell_l220_220893

theorem number_of_rectangles_with_one_gray_cell 
    (num_gray_cells : Nat) 
    (num_blue_cells : Nat) 
    (num_red_cells : Nat) 
    (blue_rectangles_per_cell : Nat) 
    (red_rectangles_per_cell : Nat)
    (total_gray_cells_calc : num_gray_cells = 2 * 20)
    (num_gray_cells_definition : num_gray_cells = num_blue_cells + num_red_cells)
    (blue_rect_cond : blue_rectangles_per_cell = 4)
    (red_rect_cond : red_rectangles_per_cell = 8)
    (num_blue_cells_calc : num_blue_cells = 36)
    (num_red_cells_calc : num_red_cells = 4)
  : num_blue_cells * blue_rectangles_per_cell + num_red_cells * red_rectangles_per_cell = 176 := 
  by
  sorry

end number_of_rectangles_with_one_gray_cell_l220_220893


namespace model_car_cost_l220_220683

theorem model_car_cost (x : ℕ) :
  (5 * x) + (5 * 10) + (5 * 2) = 160 → x = 20 :=
by
  intro h
  sorry

end model_car_cost_l220_220683


namespace find_sum_mod_7_l220_220651

open ZMod

-- Let a, b, and c be elements of the cyclic group modulo 7
def a : ZMod 7 := sorry
def b : ZMod 7 := sorry
def c : ZMod 7 := sorry

-- Conditions
axiom h1 : a * b * c = 1
axiom h2 : 4 * c = 5
axiom h3 : 5 * b = 4 + b

-- Goal
theorem find_sum_mod_7 : a + b + c = 2 := by
  sorry

end find_sum_mod_7_l220_220651


namespace project_completion_in_16_days_l220_220111

noncomputable def a_work_rate : ℚ := 1 / 20
noncomputable def b_work_rate : ℚ := 1 / 30
noncomputable def c_work_rate : ℚ := 1 / 40
noncomputable def days_a_works (X: ℚ) : ℚ := X - 10
noncomputable def days_b_works (X: ℚ) : ℚ := X - 5
noncomputable def days_c_works (X: ℚ) : ℚ := X

noncomputable def total_work (X: ℚ) : ℚ :=
  (a_work_rate * days_a_works X) + (b_work_rate * days_b_works X) + (c_work_rate * days_c_works X)

theorem project_completion_in_16_days : total_work 16 = 1 := by
  sorry

end project_completion_in_16_days_l220_220111


namespace gas_cost_per_gallon_l220_220347

def car_mileage : Nat := 450
def car1_mpg : Nat := 50
def car2_mpg : Nat := 10
def car3_mpg : Nat := 15
def monthly_gas_cost : Nat := 56

theorem gas_cost_per_gallon (car_mileage car1_mpg car2_mpg car3_mpg monthly_gas_cost : Nat)
  (h1 : car_mileage = 450) 
  (h2 : car1_mpg = 50) 
  (h3 : car2_mpg = 10) 
  (h4 : car3_mpg = 15) 
  (h5 : monthly_gas_cost = 56) :
  monthly_gas_cost / ((car_mileage / 3) / car1_mpg + 
                      (car_mileage / 3) / car2_mpg + 
                      (car_mileage / 3) / car3_mpg) = 2 := 
by 
  sorry

end gas_cost_per_gallon_l220_220347


namespace find_g5_l220_220557

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l220_220557


namespace eighteenth_entry_of_sequence_l220_220318

def r_7 (n : ℕ) : ℕ := n % 7

theorem eighteenth_entry_of_sequence : ∃ n : ℕ, (r_7 (4 * n) ≤ 3) ∧ (∀ m : ℕ, m < 18 → (r_7 (4 * m) ≤ 3) → m ≠ n) ∧ n = 30 := 
by 
  sorry

end eighteenth_entry_of_sequence_l220_220318


namespace no_three_partition_exists_l220_220654

/-- Define the partitioning property for three subsets -/
def partitions (A B C : Set ℤ) : Prop :=
  ∀ n : ℤ, (n ∈ A ∨ n ∈ B ∨ n ∈ C) ∧ (n ∈ A ↔ n-50 ∈ B ∧ n+1987 ∈ C) ∧ (n-50 ∈ A ∨ n-50 ∈ B ∨ n-50 ∈ C) ∧ (n-50 ∈ B ↔ n-50-50 ∈ A ∧ n-50+1987 ∈ C) ∧ (n+1987 ∈ A ∨ n+1987 ∈ B ∨ n+1987 ∈ C) ∧ (n+1987 ∈ C ↔ n+1987-50 ∈ A ∧ n+1987+1987 ∈ B)

/-- The main theorem stating that no such partition is possible -/
theorem no_three_partition_exists :
  ¬∃ A B C : Set ℤ, partitions A B C :=
sorry

end no_three_partition_exists_l220_220654


namespace mary_total_spent_l220_220768

-- The conditions given in the problem
def cost_berries : ℝ := 11.08
def cost_apples : ℝ := 14.33
def cost_peaches : ℝ := 9.31

-- The theorem to prove the total cost
theorem mary_total_spent : cost_berries + cost_apples + cost_peaches = 34.72 := 
by
  sorry

end mary_total_spent_l220_220768


namespace total_amount_distributed_l220_220206

def number_of_persons : ℕ := 22
def amount_per_person : ℕ := 1950

theorem total_amount_distributed : (number_of_persons * amount_per_person) = 42900 := by
  sorry

end total_amount_distributed_l220_220206


namespace find_number_of_boxes_l220_220879

-- Definitions and assumptions
def pieces_per_box : ℕ := 5 + 5
def total_pieces : ℕ := 60

-- The theorem to be proved
theorem find_number_of_boxes (B : ℕ) (h : total_pieces = B * pieces_per_box) :
  B = 6 :=
sorry

end find_number_of_boxes_l220_220879


namespace smallest_k_divides_polynomial_l220_220179

theorem smallest_k_divides_polynomial :
  ∃ (k : ℕ), k > 0 ∧ (∀ z : ℂ, z ≠ 0 → 
    (z ^ 11 + z ^ 9 + z ^ 7 + z ^ 6 + z ^ 5 + z ^ 2 + 1) ∣ (z ^ k - 1)) ∧ k = 11 := by
  sorry

end smallest_k_divides_polynomial_l220_220179


namespace roots_square_sum_l220_220154

theorem roots_square_sum {a b c : ℝ} (h1 : 3 * a^3 + 2 * a^2 - 3 * a - 8 = 0)
                                  (h2 : 3 * b^3 + 2 * b^2 - 3 * b - 8 = 0)
                                  (h3 : 3 * c^3 + 2 * c^2 - 3 * c - 8 = 0)
                                  (sum_roots : a + b + c = -2/3)
                                  (product_pairs : a * b + b * c + c * a = -1) : 
  a^2 + b^2 + c^2 = 22 / 9 := by
  sorry

end roots_square_sum_l220_220154


namespace average_value_correct_l220_220429

noncomputable def average_value (k z : ℝ) : ℝ :=
  (k + 2 * k * z + 4 * k * z + 8 * k * z + 16 * k * z) / 5

theorem average_value_correct (k z : ℝ) :
  average_value k z = (k * (1 + 30 * z)) / 5 := by
  sorry

end average_value_correct_l220_220429


namespace program_output_is_10_l220_220118

def final_value_of_A : ℤ :=
  let A := 2
  let A := A * 2
  let A := A + 6
  A

theorem program_output_is_10 : final_value_of_A = 10 := by
  sorry

end program_output_is_10_l220_220118


namespace obrien_hats_theorem_l220_220677

-- Define the number of hats Fire Chief Simpson has.
def simpson_hats : ℕ := 15

-- Define the number of hats Policeman O'Brien had before any hats were stolen.
def obrien_hats_before (simpson_hats : ℕ) : ℕ := 2 * simpson_hats + 5

-- Define the number of hats Policeman O'Brien has now, after x hats were stolen.
def obrien_hats_now (x : ℕ) : ℕ := obrien_hats_before simpson_hats - x

-- Define the theorem stating the problem
theorem obrien_hats_theorem (x : ℕ) : obrien_hats_now x = 35 - x :=
by
  sorry

end obrien_hats_theorem_l220_220677


namespace fraction_of_tomato_plants_in_second_garden_l220_220935

theorem fraction_of_tomato_plants_in_second_garden 
    (total_plants_first_garden : ℕ := 20)
    (percent_tomato_first_garden : ℚ := 10 / 100)
    (total_plants_second_garden : ℕ := 15)
    (percent_total_tomato_plants : ℚ := 20 / 100) :
    (15 : ℚ) * (1 / 3) = 5 :=
by
  sorry

end fraction_of_tomato_plants_in_second_garden_l220_220935


namespace squares_to_nine_l220_220484

theorem squares_to_nine (x : ℤ) : x^2 = 9 ↔ x = 3 ∨ x = -3 :=
sorry

end squares_to_nine_l220_220484


namespace quilt_shaded_fraction_l220_220399

theorem quilt_shaded_fraction (total_squares : ℕ) (fully_shaded : ℕ) (half_shaded_squares : ℕ) (half_shades_per_square: ℕ) : 
  (((fully_shaded) + (half_shaded_squares * half_shades_per_square / 2)) / total_squares) = (1 / 4) :=
by 
  let fully_shaded := 2
  let half_shaded_squares := 4
  let half_shades_per_square := 1
  let total_squares := 16
  sorry

end quilt_shaded_fraction_l220_220399


namespace division_result_l220_220035

theorem division_result (k q : ℕ) (h₁ : k % 81 = 11) (h₂ : 81 > 0) : k / 81 = q + 11 / 81 :=
  sorry

end division_result_l220_220035


namespace cost_price_percentage_l220_220949

theorem cost_price_percentage (CP SP : ℝ) (h1 : SP = 4 * CP) : (CP / SP) * 100 = 25 :=
by
  sorry

end cost_price_percentage_l220_220949


namespace find_a9_l220_220671

-- Define the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions of the problem
variables {a : ℕ → ℝ}
axiom h_geom_seq : is_geometric_sequence a
axiom h_root1 : a 3 * a 15 = 1
axiom h_root2 : a 3 + a 15 = -4

-- The proof statement
theorem find_a9 : a 9 = 1 := 
by sorry

end find_a9_l220_220671


namespace sum_of_cubes_8001_l220_220570
-- Import the entire Mathlib library

-- Define a property on integers
def approx (x y : ℝ) := abs (x - y) < 0.000000000000004

-- Define the variables a and b
variables (a b : ℤ)

-- State the theorem
theorem sum_of_cubes_8001 (h : approx (a * b : ℝ) 19.999999999999996) : a^3 + b^3 = 8001 := 
sorry

end sum_of_cubes_8001_l220_220570


namespace total_order_cost_l220_220424

theorem total_order_cost :
  let c := 2 * 30
  let w := 9 * 15
  let s := 50
  c + w + s = 245 := 
by
  linarith

end total_order_cost_l220_220424


namespace truck_speed_on_dirt_road_l220_220251

theorem truck_speed_on_dirt_road 
  (total_distance: ℝ) (time_on_dirt: ℝ) (time_on_paved: ℝ) (speed_difference: ℝ)
  (h1: total_distance = 200) (h2: time_on_dirt = 3) (h3: time_on_paved = 2) (h4: speed_difference = 20) : 
  ∃ v: ℝ, (time_on_dirt * v + time_on_paved * (v + speed_difference) = total_distance) ∧ v = 32 := 
sorry

end truck_speed_on_dirt_road_l220_220251


namespace simplify_expression_l220_220843

noncomputable def simplified_result (a b : ℝ) (i : ℂ) (hi : i * i = -1) : ℂ :=
  (a + b * i) * (a - b * i)

theorem simplify_expression (a b : ℝ) (i : ℂ) (hi : i * i = -1) :
  simplified_result a b i hi = a^2 + b^2 := by
  sorry

end simplify_expression_l220_220843


namespace georgie_ghost_ways_l220_220255

-- Define the total number of windows and locked windows
def total_windows : ℕ := 8
def locked_windows : ℕ := 2

-- Define the number of usable windows
def usable_windows : ℕ := total_windows - locked_windows

-- Define the theorem to prove the number of ways Georgie the Ghost can enter and exit
theorem georgie_ghost_ways :
  usable_windows * (usable_windows - 1) = 30 := by
  sorry

end georgie_ghost_ways_l220_220255


namespace convert_to_scientific_notation_l220_220089

def original_value : ℝ := 3462.23
def scientific_notation_value : ℝ := 3.46223 * 10^3

theorem convert_to_scientific_notation : 
  original_value = scientific_notation_value :=
sorry

end convert_to_scientific_notation_l220_220089


namespace value_of_a_l220_220756

-- Define the quadratic function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

-- Define the condition f(1) = f(2)
def condition (a b : ℝ) : Prop := f 1 a b = f 2 a b

-- The proof problem statement
theorem value_of_a (a b : ℝ) (h : condition a b) : a = -3 :=
by sorry

end value_of_a_l220_220756


namespace collinear_points_l220_220313

-- Define collinear points function
def collinear (x1 y1 z1 x2 y2 z2 x3 y3 z3: ℝ) : Prop :=
  ∀ (a b c : ℝ), a * (y2 - y1) * (z3 - z1) + b * (z2 - z1) * (x3 - x1) + c * (x2 - x1) * (y3 - y1) = 0

-- Problem statement
theorem collinear_points (a b : ℝ)
  (h : collinear 2 a b a 3 b a b 4) :
  a + b = -2 :=
sorry

end collinear_points_l220_220313


namespace y1_greater_than_y2_l220_220481

-- Definitions of the conditions.
def point1_lies_on_line (y₁ b : ℝ) : Prop := y₁ = -3 * (-2 : ℝ) + b
def point2_lies_on_line (y₂ b : ℝ) : Prop := y₂ = -3 * (-1 : ℝ) + b

-- The theorem to prove: y₁ > y₂ given the conditions.
theorem y1_greater_than_y2 (y₁ y₂ b : ℝ) (h1 : point1_lies_on_line y₁ b) (h2 : point2_lies_on_line y₂ b) : y₁ > y₂ :=
by {
  sorry
}

end y1_greater_than_y2_l220_220481


namespace number_of_children_l220_220927

namespace CurtisFamily

variables {m x : ℕ} {xy : ℕ}

/-- Given conditions for Curtis family average ages. -/
def family_average_age (m x xy : ℕ) : Prop := (m + 50 + xy) / (2 + x) = 25

def mother_children_average_age (m x xy : ℕ) : Prop := (m + xy) / (1 + x) = 20

/-- The number of children in Curtis family is 4, given the average age conditions. -/
theorem number_of_children (m xy : ℕ) (h1 : family_average_age m 4 xy) (h2 : mother_children_average_age m 4 xy) : x = 4 :=
by
  sorry

end CurtisFamily

end number_of_children_l220_220927


namespace trigonometric_identity_l220_220199

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = Real.sqrt 2) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = (5 - Real.sqrt 2) / 3 := 
by
  sorry

end trigonometric_identity_l220_220199


namespace scissor_count_l220_220266

theorem scissor_count :
  let initial_scissors := 54 
  let added_scissors := 22
  let removed_scissors := 15
  initial_scissors + added_scissors - removed_scissors = 61 := by
  sorry

end scissor_count_l220_220266


namespace percentage_of_students_passed_l220_220047

theorem percentage_of_students_passed
  (students_failed : ℕ)
  (total_students : ℕ)
  (H_failed : students_failed = 260)
  (H_total : total_students = 400)
  (passed := total_students - students_failed) :
  (passed * 100 / total_students : ℝ) = 35 := 
by
  -- proof steps would go here
  sorry

end percentage_of_students_passed_l220_220047


namespace largest_angle_of_triangle_l220_220457

theorem largest_angle_of_triangle (x : ℝ) (h : x + 3 * x + 5 * x = 180) : 5 * x = 100 :=
sorry

end largest_angle_of_triangle_l220_220457


namespace amy_balloons_l220_220510

theorem amy_balloons (james_balloons amy_balloons : ℕ) (h1 : james_balloons = 1222) (h2 : james_balloons = amy_balloons + 709) : amy_balloons = 513 :=
by
  sorry

end amy_balloons_l220_220510


namespace exists_same_color_points_at_unit_distance_l220_220336

theorem exists_same_color_points_at_unit_distance
  (color : ℝ × ℝ → ℕ)
  (coloring : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end exists_same_color_points_at_unit_distance_l220_220336


namespace part1_part2_l220_220915

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 
  Real.log x + 0.5 * m * x^2 - 2 

def perpendicular_slope_condition (m : ℝ) : Prop := 
  let k := (1 / 1 + m)
  k = -1 / 2

def inequality_condition (m : ℝ) : Prop := 
  ∀ x > 0, 
  Real.log x - 0.5 * m * x^2 + (1 - m) * x + 1 ≤ 0

theorem part1 : perpendicular_slope_condition (-3/2) :=
  sorry

theorem part2 : ∃ m : ℤ, m ≥ 2 ∧ inequality_condition m :=
  sorry

end part1_part2_l220_220915


namespace scientific_notation_of_one_point_six_million_l220_220679

-- Define the given number
def one_point_six_million : ℝ := 1.6 * 10^6

-- State the theorem to prove the equivalence
theorem scientific_notation_of_one_point_six_million :
  one_point_six_million = 1.6 * 10^6 :=
by
  sorry

end scientific_notation_of_one_point_six_million_l220_220679


namespace largest_quotient_is_25_l220_220194

def largest_quotient_set : Set ℤ := {-25, -4, -1, 1, 3, 9}

theorem largest_quotient_is_25 :
  ∃ (a b : ℤ), a ∈ largest_quotient_set ∧ b ∈ largest_quotient_set ∧ b ≠ 0 ∧ (a : ℚ) / b = 25 := by
  sorry

end largest_quotient_is_25_l220_220194


namespace total_amount_is_105_l220_220472

theorem total_amount_is_105 (x_amount y_amount z_amount : ℝ) 
  (h1 : ∀ x, y_amount = x * 0.45) 
  (h2 : ∀ x, z_amount = x * 0.30) 
  (h3 : y_amount = 27) : 
  (x_amount + y_amount + z_amount = 105) := 
sorry

end total_amount_is_105_l220_220472


namespace arccos_neg_one_eq_pi_l220_220586

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l220_220586


namespace volunteer_arrangement_l220_220225

theorem volunteer_arrangement (volunteers : Fin 5) (elderly : Fin 2) 
  (h1 : elderly.1 ≠ 0 ∧ elderly.1 ≠ 6) : 
  ∃ arrangements : ℕ, arrangements = 960 := 
sorry

end volunteer_arrangement_l220_220225


namespace constant_in_denominator_l220_220490

theorem constant_in_denominator (x y z : ℝ) (some_constant : ℝ)
  (h : ((x - y)^3 + (y - z)^3 + (z - x)^3) / (some_constant * (x - y) * (y - z) * (z - x)) = 0.2) :
  some_constant = 15 := 
sorry

end constant_in_denominator_l220_220490


namespace simplify_fraction_product_l220_220055

theorem simplify_fraction_product : 
  (270 / 24) * (7 / 210) * (6 / 4) = 4.5 :=
by
  sorry

end simplify_fraction_product_l220_220055


namespace ones_digit_of_8_pow_47_l220_220243

theorem ones_digit_of_8_pow_47 : (8^47) % 10 = 2 := 
  sorry

end ones_digit_of_8_pow_47_l220_220243


namespace non_black_cows_l220_220263

-- Define the main problem conditions
def total_cows : ℕ := 18
def black_cows : ℕ := (total_cows / 2) + 5

-- Statement to prove the number of non-black cows
theorem non_black_cows :
  total_cows - black_cows = 4 :=
by
  sorry

end non_black_cows_l220_220263


namespace sum_of_ages_l220_220329

variables (M A : ℕ)

def Maria_age_relation : Prop :=
  M = A + 8

def future_age_relation : Prop :=
  M + 10 = 3 * (A - 6)

theorem sum_of_ages (h₁ : Maria_age_relation M A) (h₂ : future_age_relation M A) : M + A = 44 :=
by
  sorry

end sum_of_ages_l220_220329


namespace ratio_of_amount_lost_l220_220282

noncomputable def amount_lost (initial_amount spent_motorcycle spent_concert after_loss : ℕ) : ℕ :=
  let remaining_after_motorcycle := initial_amount - spent_motorcycle
  let remaining_after_concert := remaining_after_motorcycle / 2
  remaining_after_concert - after_loss

noncomputable def ratio (a b : ℕ) : ℕ × ℕ :=
  let g := Nat.gcd a b
  (a / g, b / g)

theorem ratio_of_amount_lost 
  (initial_amount spent_motorcycle spent_concert after_loss : ℕ)
  (h1 : initial_amount = 5000)
  (h2 : spent_motorcycle = 2800)
  (h3 : spent_concert = (initial_amount - spent_motorcycle) / 2)
  (h4 : after_loss = 825) :
  ratio (amount_lost initial_amount spent_motorcycle spent_concert after_loss)
        spent_concert = (1, 4) := by
  sorry

end ratio_of_amount_lost_l220_220282


namespace Rachel_total_earnings_l220_220758

-- Define the constants for the conditions
def hourly_wage : ℝ := 12
def people_served : ℕ := 20
def tip_per_person : ℝ := 1.25

-- Define the problem
def total_money_made : ℝ := hourly_wage + (people_served * tip_per_person)

-- State the theorem to be proved
theorem Rachel_total_earnings : total_money_made = 37 := by
  sorry

end Rachel_total_earnings_l220_220758


namespace option_D_is_correct_l220_220192

noncomputable def correct_operation : Prop := 
  (∀ x : ℝ, x + x ≠ 2 * x^2) ∧
  (∀ y : ℝ, 2 * y^3 + 3 * y^2 ≠ 5 * y^5) ∧
  (∀ x : ℝ, 2 * x - x ≠ 1) ∧
  (∀ x y : ℝ, 4 * x^3 * y^2 - (-2)^2 * x^3 * y^2 = 0)

theorem option_D_is_correct : correct_operation :=
by {
  -- We'll complete the proofs later
  sorry
}

end option_D_is_correct_l220_220192


namespace Megan_pays_correct_amount_l220_220523

def original_price : ℝ := 22
def discount : ℝ := 6
def amount_paid := original_price - discount

theorem Megan_pays_correct_amount : amount_paid = 16 := by
  sorry

end Megan_pays_correct_amount_l220_220523


namespace sufficient_condition_l220_220330

theorem sufficient_condition (a : ℝ) (h : a ≥ 10) : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → x^2 - a ≤ 0 :=
by
  sorry

end sufficient_condition_l220_220330


namespace annual_interest_rate_l220_220054

noncomputable def compound_interest 
  (P : ℝ) (A : ℝ) (n : ℕ) (t : ℝ) (r : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem annual_interest_rate 
  (P := 140) (A := 169.40) (n := 2) (t := 1) :
  ∃ r : ℝ, compound_interest P A n t r ∧ r = 0.2 :=
sorry

end annual_interest_rate_l220_220054


namespace cube_sufficient_but_not_necessary_l220_220773

theorem cube_sufficient_but_not_necessary (x : ℝ) : (x^3 > 27 → |x| > 3) ∧ (¬(|x| > 3 → x^3 > 27)) :=
by
  sorry

end cube_sufficient_but_not_necessary_l220_220773


namespace false_statements_count_is_3_l220_220637

-- Define the statements
def statement1_false : Prop := ¬ (1 ≠ 1)     -- Not exactly one statement is false
def statement2_false : Prop := ¬ (2 ≠ 2)     -- Not exactly two statements are false
def statement3_false : Prop := ¬ (3 ≠ 3)     -- Not exactly three statements are false
def statement4_false : Prop := ¬ (4 ≠ 4)     -- Not exactly four statements are false
def statement5_false : Prop := ¬ (5 ≠ 5)     -- Not all statements are false

-- Prove that the number of false statements is 3
theorem false_statements_count_is_3 :
  (statement1_false → statement2_false →
  statement3_false → statement4_false →
  statement5_false → (3 = 3)) := by
  sorry

end false_statements_count_is_3_l220_220637


namespace league_games_count_l220_220912

theorem league_games_count :
  let num_divisions := 2
  let teams_per_division := 9
  let intra_division_games (teams_per_div : ℕ) := (teams_per_div * (teams_per_div - 1) / 2) * 3
  let inter_division_games (teams_per_div : ℕ) (num_div : ℕ) := teams_per_div * teams_per_div * 2
  intra_division_games teams_per_division * num_divisions + inter_division_games teams_per_division num_divisions = 378 :=
by
  sorry

end league_games_count_l220_220912


namespace probability_blue_given_not_red_l220_220653

theorem probability_blue_given_not_red :
  let total_balls := 20
  let red_balls := 5
  let yellow_balls := 5
  let blue_balls := 10
  let non_red_balls := yellow_balls + blue_balls
  let blue_given_not_red := (blue_balls : ℚ) / non_red_balls
  blue_given_not_red = 2 / 3 := 
by
  sorry

end probability_blue_given_not_red_l220_220653


namespace sequence_properties_l220_220183

-- Define the sequences a_n and b_n
noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def b (n : ℕ) : ℕ := sorry

-- Define the conditions
axiom h1 : a 1 = 1
axiom h2 : b 1 = 1
axiom h3 : ∀ n, b (n + 1) ^ 2 = b n * b (n + 2)
axiom h4 : 9 * (b 3) ^ 2 = b 2 * b 6
axiom h5 : ∀ n, b (n + 1) / a (n + 1) = b n / (a n + 2 * b n)

-- Define the theorem to prove
theorem sequence_properties :
  (∀ n, a n = (2 * n - 1) * 3 ^ (n - 1)) ∧
  (∀ n, (a n) / (b n) = (a (n + 1)) / (b (n + 1)) + 2) := by
  sorry

end sequence_properties_l220_220183


namespace chicago_bulls_wins_l220_220694

theorem chicago_bulls_wins (B H : ℕ) (h1 : B + H = 145) (h2 : H = B + 5) : B = 70 :=
by
  sorry

end chicago_bulls_wins_l220_220694


namespace function_monotonicity_l220_220203

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_monotonicity :
  ∀ x₁ x₂, -Real.pi / 6 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ Real.pi / 3 → f x₁ ≤ f x₂ :=
by
  sorry

end function_monotonicity_l220_220203


namespace sales_tax_difference_l220_220590

/-- The difference in sales tax calculation given the changes in rate. -/
theorem sales_tax_difference 
  (market_price : ℝ := 9000) 
  (original_rate : ℝ := 0.035) 
  (new_rate : ℝ := 0.0333) 
  (difference : ℝ := 15.3) :
  market_price * original_rate - market_price * new_rate = difference :=
by
  /- The proof is omitted as per the instructions. -/
  sorry

end sales_tax_difference_l220_220590


namespace secret_code_count_l220_220668

-- Conditions
def num_colors : ℕ := 8
def num_slots : ℕ := 5

-- The proof statement
theorem secret_code_count : (num_colors ^ num_slots) = 32768 := by
  sorry

end secret_code_count_l220_220668


namespace incorrect_observation_l220_220460

theorem incorrect_observation (n : ℕ) (mean_original mean_corrected correct_obs incorrect_obs : ℝ)
  (h1 : n = 40) 
  (h2 : mean_original = 36) 
  (h3 : mean_corrected = 36.45) 
  (h4 : correct_obs = 34) 
  (h5 : n * mean_original = 1440) 
  (h6 : n * mean_corrected = 1458) 
  (h_diff : 1458 - 1440 = 18) :
  incorrect_obs = 52 :=
by
  sorry

end incorrect_observation_l220_220460


namespace unique_solution_of_functional_equation_l220_220134

theorem unique_solution_of_functional_equation
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f (x + y)) = f x + y) :
  ∀ x : ℝ, f x = x := 
sorry

end unique_solution_of_functional_equation_l220_220134


namespace total_steps_correct_l220_220333

/-- Definition of the initial number of steps on the first day --/
def steps_first_day : Nat := 200 + 300

/-- Definition of the number of steps on the second day --/
def steps_second_day : Nat := (3 / 2) * steps_first_day -- 1.5 is expressed as 3/2

/-- Definition of the number of steps on the third day --/
def steps_third_day : Nat := 2 * steps_second_day

/-- The total number of steps Eliana walked during the three days --/
def total_steps : Nat := steps_first_day + steps_second_day + steps_third_day

theorem total_steps_correct : total_steps = 2750 :=
  by
  -- provide the proof here
  sorry

end total_steps_correct_l220_220333


namespace swimmer_speed_in_still_water_l220_220323

variable (distance : ℝ) (time : ℝ) (current_speed : ℝ) (swimmer_speed_still_water : ℝ)

-- Define the given conditions
def conditions := 
  distance = 8 ∧
  time = 5 ∧
  current_speed = 1.4 ∧
  (distance / time = swimmer_speed_still_water - current_speed)

-- The theorem we want to prove
theorem swimmer_speed_in_still_water : 
  conditions distance time current_speed swimmer_speed_still_water → 
  swimmer_speed_still_water = 3 := 
by 
  -- Skipping the actual proof
  sorry

end swimmer_speed_in_still_water_l220_220323


namespace greatest_solution_of_equation_l220_220504

theorem greatest_solution_of_equation : ∀ x : ℝ, x ≠ 9 ∧ (x^2 - x - 90) / (x - 9) = 4 / (x + 6) → x ≤ -7 :=
by
  intros x hx
  sorry

end greatest_solution_of_equation_l220_220504


namespace density_of_cone_in_mercury_l220_220067

variable {h : ℝ} -- height of the cone
variable {ρ : ℝ} -- density of the cone
variable {ρ_m : ℝ} -- density of the mercury
variable {k : ℝ} -- proportion factor

-- Archimedes' principle applied to the cone floating in mercury
theorem density_of_cone_in_mercury (stable_eq: ∀ (V V_sub: ℝ), (ρ * V) = (ρ_m * V_sub))
(h_sub: h / k = (k - 1) / k) :
  ρ = ρ_m * ((k - 1)^3 / k^3) :=
by
  sorry

end density_of_cone_in_mercury_l220_220067


namespace inequality_solution_set_l220_220113

noncomputable def solution_set := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem inequality_solution_set : {x : ℝ | (x - 1) * (3 - x) ≥ 0} = solution_set := by
  sorry

end inequality_solution_set_l220_220113


namespace cost_of_camel_l220_220338

-- Define the cost of each animal as variables
variables (C H O E : ℝ)

-- Assume the given relationships as hypotheses
def ten_camels_eq_twentyfour_horses := (10 * C = 24 * H)
def sixteens_horses_eq_four_oxen := (16 * H = 4 * O)
def six_oxen_eq_four_elephants := (6 * O = 4 * E)
def ten_elephants_eq_140000 := (10 * E = 140000)

-- The theorem that we want to prove
theorem cost_of_camel (h1 : ten_camels_eq_twentyfour_horses C H)
                      (h2 : sixteens_horses_eq_four_oxen H O)
                      (h3 : six_oxen_eq_four_elephants O E)
                      (h4 : ten_elephants_eq_140000 E) :
  C = 5600 := sorry

end cost_of_camel_l220_220338


namespace neg_p_iff_forall_l220_220250

-- Define the proposition p
def p : Prop := ∃ (x : ℝ), x > 1 ∧ x^2 - 1 > 0

-- State the negation of p as a theorem
theorem neg_p_iff_forall : ¬ p ↔ ∀ (x : ℝ), x > 1 → x^2 - 1 ≤ 0 :=
by sorry

end neg_p_iff_forall_l220_220250


namespace total_arrangements_l220_220102

theorem total_arrangements (students communities : ℕ) 
  (h_students : students = 5) 
  (h_communities : communities = 3)
  (h_conditions :
    ∀(student : Fin students) (community : Fin communities), 
      true 
  ) : 150 = 150 :=
by sorry

end total_arrangements_l220_220102


namespace polynomial_value_at_2018_l220_220672

theorem polynomial_value_at_2018 (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f (-x^2 - x - 1) = x^4 + 2*x^3 + 2022*x^2 + 2021*x + 2019) : 
  f 2018 = -2019 :=
sorry

end polynomial_value_at_2018_l220_220672


namespace find_some_number_l220_220513

theorem find_some_number :
  ∃ some_number : ℝ, (3.242 * 10 / some_number) = 0.032420000000000004 ∧ some_number = 1000 :=
by
  sorry

end find_some_number_l220_220513


namespace expression_is_odd_l220_220483

-- Define positive integers
def is_positive (n : ℕ) := n > 0

-- Define odd integer
def is_odd (n : ℕ) := n % 2 = 1

-- Define multiple of 3
def is_multiple_of_3 (n : ℕ) := ∃ k : ℕ, n = 3 * k

-- The Lean 4 statement to prove the problem
theorem expression_is_odd (a b c : ℕ)
  (ha : is_positive a) (hb : is_positive b) (hc : is_positive c)
  (h_odd_a : is_odd a) (h_odd_b : is_odd b) (h_mult_3_c : is_multiple_of_3 c) :
  is_odd (5^a + (b-1)^2 * c) :=
by
  sorry

end expression_is_odd_l220_220483


namespace cindy_arrival_speed_l220_220001

def cindy_speed (d t1 t2 t3: ℕ) : Prop :=
  (d = 20 * t1) ∧ 
  (d = 10 * (t2 + 3 / 4)) ∧
  (t3 = t1 + 1 / 2) ∧
  (20 * t1 = 10 * (t2 + 3 / 4)) -> 
  (d / (t3) = 12)

theorem cindy_arrival_speed (t1 t2: ℕ) (h₁: t2 = t1 + 3 / 4) (d: ℕ) (h2: d = 20 * t1) (h3: t3 = t1 + 1 / 2) :
  cindy_speed d t1 t2 t3 := by
  sorry

end cindy_arrival_speed_l220_220001


namespace buddy_met_boy_students_l220_220892

theorem buddy_met_boy_students (total_students : ℕ) (girl_students : ℕ) (boy_students : ℕ) (h1 : total_students = 123) (h2 : girl_students = 57) : boy_students = 66 :=
by
  sorry

end buddy_met_boy_students_l220_220892


namespace factorization_of_M_l220_220317

theorem factorization_of_M :
  ∀ (x y z : ℝ), x^3 * (y - z) + y^3 * (z - x) + z^3 * (x - y) = 
  (x + y + z) * (x - y) * (y - z) * (z - x) := by
  sorry

end factorization_of_M_l220_220317


namespace problem1_problem2_l220_220240

-- Problem 1: Prove that \(\sqrt{27}+3\sqrt{\frac{1}{3}}-\sqrt{24} \times \sqrt{2} = 0\)
theorem problem1 : Real.sqrt 27 + 3 * Real.sqrt (1 / 3) - Real.sqrt 24 * Real.sqrt 2 = 0 := 
by sorry

-- Problem 2: Prove that \((\sqrt{5}-2)(2+\sqrt{5})-{(\sqrt{3}-1)}^{2} = -3 + 2\sqrt{3}\)
theorem problem2 : (Real.sqrt 5 - 2) * (2 + Real.sqrt 5) - (Real.sqrt 3 - 1) ^ 2 = -3 + 2 * Real.sqrt 3 := 
by sorry

end problem1_problem2_l220_220240


namespace factorial_divides_exponential_difference_l220_220261

theorem factorial_divides_exponential_difference (n : ℕ) : n! ∣ 2^(2 * n!) - 2^n! :=
by
  sorry

end factorial_divides_exponential_difference_l220_220261


namespace transform_expression_to_product_l220_220224

open Real

noncomputable def transform_expression (α : ℝ) : ℝ :=
  4.66 * sin (5 * π / 2 + 4 * α) - (sin (5 * π / 2 + 2 * α)) ^ 6 + (cos (7 * π / 2 - 2 * α)) ^ 6

theorem transform_expression_to_product (α : ℝ) :
  transform_expression α = (1 / 8) * sin (4 * α) * sin (8 * α) :=
by
  sorry

end transform_expression_to_product_l220_220224


namespace rod_length_of_weight_l220_220956

theorem rod_length_of_weight (w10 : ℝ) (wL : ℝ) (L : ℝ) (h1 : w10 = 23.4) (h2 : wL = 14.04) : L = 6 :=
by
  sorry

end rod_length_of_weight_l220_220956


namespace lisa_total_distance_l220_220124

-- Definitions for distances and counts of trips
def plane_distance : ℝ := 256.0
def train_distance : ℝ := 120.5
def bus_distance : ℝ := 35.2

def plane_trips : ℕ := 32
def train_trips : ℕ := 16
def bus_trips : ℕ := 42

-- Definition of total distance traveled
def total_distance_traveled : ℝ :=
  (plane_distance * plane_trips)
  + (train_distance * train_trips)
  + (bus_distance * bus_trips)

-- The statement to be proven
theorem lisa_total_distance :
  total_distance_traveled = 11598.4 := by
  sorry

end lisa_total_distance_l220_220124


namespace melissa_total_cost_l220_220230

-- Definitions based on conditions
def daily_rental_rate : ℝ := 15
def mileage_rate : ℝ := 0.10
def number_of_days : ℕ := 3
def number_of_miles : ℕ := 300

-- Theorem statement to prove the total cost
theorem melissa_total_cost : daily_rental_rate * number_of_days + mileage_rate * number_of_miles = 75 := 
by 
  sorry

end melissa_total_cost_l220_220230


namespace project_completion_l220_220947

theorem project_completion (x : ℕ) :
  (21 - x) * (1 / 12 : ℚ) + x * (1 / 30 : ℚ) = 1 → x = 15 :=
by
  sorry

end project_completion_l220_220947


namespace james_calories_ratio_l220_220160

theorem james_calories_ratio:
  ∀ (dancing_sessions_per_day : ℕ) (hours_per_session : ℕ) 
  (days_per_week : ℕ) (calories_per_hour_walking : ℕ) 
  (total_calories_dancing_per_week : ℕ),
  dancing_sessions_per_day = 2 →
  hours_per_session = 1/2 →
  days_per_week = 4 →
  calories_per_hour_walking = 300 →
  total_calories_dancing_per_week = 2400 →
  300 * 2 = 600 →
  (total_calories_dancing_per_week / (dancing_sessions_per_day * hours_per_session * days_per_week)) / calories_per_hour_walking = 2 :=
by
  sorry

end james_calories_ratio_l220_220160


namespace virginia_ends_up_with_93_eggs_l220_220219

-- Define the initial and subtracted number of eggs as conditions
def initial_eggs : ℕ := 96
def taken_eggs : ℕ := 3

-- The theorem we want to prove
theorem virginia_ends_up_with_93_eggs : (initial_eggs - taken_eggs) = 93 :=
by
  sorry

end virginia_ends_up_with_93_eggs_l220_220219


namespace correct_avg_and_mode_l220_220784

-- Define the conditions and correct answers
def avgIncorrect : ℚ := 13.5
def medianIncorrect : ℚ := 12
def modeCorrect : ℚ := 16
def totalNumbers : ℕ := 25
def incorrectNums : List ℚ := [33.5, 47.75, 58.5, 19/2]
def correctNums : List ℚ := [43.5, 56.25, 68.5, 21/2]

noncomputable def correctSum : ℚ := (avgIncorrect * totalNumbers) + (correctNums.sum - incorrectNums.sum)
noncomputable def correctAvg : ℚ := correctSum / totalNumbers

theorem correct_avg_and_mode :
  correctAvg = 367 / 25 ∧ modeCorrect = 16 :=
by
  sorry

end correct_avg_and_mode_l220_220784


namespace rightmost_three_digits_of_7_pow_1987_l220_220776

theorem rightmost_three_digits_of_7_pow_1987 :
  (7^1987 : ℕ) % 1000 = 643 := 
by 
  sorry

end rightmost_three_digits_of_7_pow_1987_l220_220776


namespace range_of_k_l220_220676

theorem range_of_k (k : ℝ) (H : ∀ x : ℤ, |(x : ℝ) - 1| < k * x ↔ x ∈ ({1, 2, 3} : Set ℤ)) : 
  (2 / 3 : ℝ) < k ∧ k ≤ (3 / 4 : ℝ) :=
by
  sorry

end range_of_k_l220_220676


namespace find_original_number_l220_220362

theorem find_original_number (r : ℝ) (h1 : r * 1.125 - r * 0.75 = 30) : r = 80 :=
by
  sorry

end find_original_number_l220_220362


namespace parabola_circle_intersection_l220_220401

theorem parabola_circle_intersection :
  (∃ x y : ℝ, y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) →
  (∃ r : ℝ, ∀ x y : ℝ, (y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) →
    (x - 5/2)^2 + (y + 3/2)^2 = r^2 ∧ r^2 = 3/2) :=
by
  intros
  sorry

end parabola_circle_intersection_l220_220401


namespace ratio_traditionalists_progressives_l220_220372

variables (T P C : ℝ)

-- Conditions from the problem
-- There are 6 provinces and each province has the same number of traditionalists
-- The fraction of the country that is traditionalist is 0.6
def country_conditions (T P C : ℝ) :=
  (6 * T = 0.6 * C) ∧
  (C = P + 6 * T)

-- Theorem that needs to be proven
theorem ratio_traditionalists_progressives (T P C : ℝ) (h : country_conditions T P C) :
  T / P = 1 / 4 :=
by
  -- Setup conditions from the hypothesis h
  rcases h with ⟨h1, h2⟩
  -- Start the proof (Proof content is not required as per instructions)
  sorry

end ratio_traditionalists_progressives_l220_220372


namespace length_of_garden_l220_220071

theorem length_of_garden (P B : ℕ) (hP : P = 1800) (hB : B = 400) : 
  ∃ L : ℕ, L = 500 ∧ P = 2 * (L + B) :=
by
  sorry

end length_of_garden_l220_220071


namespace reverse_digits_difference_l220_220031

theorem reverse_digits_difference (q r : ℕ) (x y : ℕ) 
  (hq : q = 10 * x + y)
  (hr : r = 10 * y + x)
  (hq_r_pos : q > r)
  (h_diff_lt_20 : q - r < 20)
  (h_max_diff : q - r = 18) :
  x - y = 2 := 
by
  sorry

end reverse_digits_difference_l220_220031


namespace correct_combined_average_l220_220754

noncomputable def average_marks : ℝ :=
  let num_students : ℕ := 100
  let avg_math_marks : ℝ := 85
  let avg_science_marks : ℝ := 89
  let incorrect_math_marks : List ℝ := [76, 80, 95, 70, 90]
  let correct_math_marks : List ℝ := [86, 70, 75, 90, 100]
  let incorrect_science_marks : List ℝ := [105, 60, 80, 92, 78]
  let correct_science_marks : List ℝ := [95, 70, 90, 82, 88]

  let total_incorrect_math := incorrect_math_marks.sum
  let total_correct_math := correct_math_marks.sum
  let diff_math := total_correct_math - total_incorrect_math

  let total_incorrect_science := incorrect_science_marks.sum
  let total_correct_science := correct_science_marks.sum
  let diff_science := total_correct_science - total_incorrect_science

  let incorrect_total_math := avg_math_marks * num_students
  let correct_total_math := incorrect_total_math + diff_math

  let incorrect_total_science := avg_science_marks * num_students
  let correct_total_science := incorrect_total_science + diff_science

  let combined_total := correct_total_math + correct_total_science
  combined_total / (num_students * 2)

theorem correct_combined_average :
  average_marks = 87.1 :=
by
  sorry

end correct_combined_average_l220_220754


namespace fitted_bowling_ball_volume_correct_l220_220050

noncomputable def volume_of_fitted_bowling_ball : ℝ :=
  let ball_radius := 12
  let ball_volume := (4/3) * Real.pi * ball_radius^3
  let hole1_radius := 1
  let hole1_volume := Real.pi * hole1_radius^2 * 6
  let hole2_radius := 1.25
  let hole2_volume := Real.pi * hole2_radius^2 * 6
  let hole3_radius := 2
  let hole3_volume := Real.pi * hole3_radius^2 * 6
  ball_volume - (hole1_volume + hole2_volume + hole3_volume)

theorem fitted_bowling_ball_volume_correct :
  volume_of_fitted_bowling_ball = 2264.625 * Real.pi := by
  -- proof would go here
  sorry

end fitted_bowling_ball_volume_correct_l220_220050


namespace pythagorean_theorem_l220_220761

theorem pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : c^2 = a^2 + b^2 :=
sorry

end pythagorean_theorem_l220_220761


namespace cos_triple_sum_div_l220_220828

theorem cos_triple_sum_div {A B C : ℝ} (h : Real.cos A + Real.cos B + Real.cos C = 0) : 
  (Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C)) / (Real.cos A * Real.cos B * Real.cos C) = 12 :=
by
  sorry

end cos_triple_sum_div_l220_220828


namespace contrapositive_false_of_implication_false_l220_220016

variable (p q : Prop)

-- The statement we need to prove: If "if p then q" is false, 
-- then "if not q then not p" must be false.
theorem contrapositive_false_of_implication_false (h : ¬ (p → q)) : ¬ (¬ q → ¬ p) :=
by
sorry

end contrapositive_false_of_implication_false_l220_220016


namespace no_adjacent_numbers_differ_by_10_or_multiple_10_l220_220390

theorem no_adjacent_numbers_differ_by_10_or_multiple_10 :
  ¬ ∃ (f : Fin 25 → Fin 25),
    (∀ n : Fin 25, f (n + 1) - f n = 10 ∨ (f (n + 1) - f n) % 10 = 0) :=
by
  sorry

end no_adjacent_numbers_differ_by_10_or_multiple_10_l220_220390


namespace johnsonville_max_band_members_l220_220451

def max_band_members :=
  ∃ m : ℤ, 30 * m % 34 = 2 ∧ 30 * m < 1500 ∧
  ∀ n : ℤ, (30 * n % 34 = 2 ∧ 30 * n < 1500) → 30 * n ≤ 30 * m

theorem johnsonville_max_band_members : ∃ m : ℤ, 30 * m % 34 = 2 ∧ 30 * m < 1500 ∧
                                           30 * m = 1260 :=
by 
  sorry

end johnsonville_max_band_members_l220_220451


namespace permutation_and_combination_results_l220_220345

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

def C (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem permutation_and_combination_results :
  A 5 2 = 20 ∧ C 6 3 + C 6 4 = 35 := by
  sorry

end permutation_and_combination_results_l220_220345


namespace suit_price_after_discount_l220_220648

-- Define the original price of the suit.
def original_price : ℝ := 150

-- Define the increase rate and the discount rate.
def increase_rate : ℝ := 0.20
def discount_rate : ℝ := 0.20

-- Define the increased price after the 20% increase.
def increased_price : ℝ := original_price * (1 + increase_rate)

-- Define the final price after applying the 20% discount.
def final_price : ℝ := increased_price * (1 - discount_rate)

-- Prove that the final price is $144.
theorem suit_price_after_discount : final_price = 144 := by
  sorry  -- Proof to be completed

end suit_price_after_discount_l220_220648


namespace max_books_per_student_l220_220408

-- Define the variables and conditions
variables (students : ℕ) (not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 : ℕ)
variables (avg_books_per_student : ℕ)
variables (remaining_books : ℕ) (max_books : ℕ)

-- Assume given conditions
def conditions : Prop :=
  students = 100 ∧ 
  not_borrowed5 = 5 ∧ 
  borrowed1_20 = 20 ∧ 
  borrowed2_25 = 25 ∧ 
  borrowed3_30 = 30 ∧ 
  borrowed5_20 = 20 ∧ 
  avg_books_per_student = 3

-- Prove the maximum number of books any single student could have borrowed is 50
theorem max_books_per_student (students not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 avg_books_per_student : ℕ) (max_books : ℕ) :
  conditions students not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 avg_books_per_student →
  max_books = 50 :=
by
  sorry

end max_books_per_student_l220_220408


namespace sum_of_roots_l220_220539

theorem sum_of_roots (a1 a2 a3 a4 a5 : ℤ)
  (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧
                a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧
                a3 ≠ a4 ∧ a3 ≠ a5 ∧
                a4 ≠ a5)
  (h_poly : (104 - a1) * (104 - a2) * (104 - a3) * (104 - a4) * (104 - a5) = 2012) :
  a1 + a2 + a3 + a4 + a5 = 17 := by
  sorry

end sum_of_roots_l220_220539


namespace herd_total_cows_l220_220678

theorem herd_total_cows (n : ℕ) : 
  let first_son := 1 / 3 * n
  let second_son := 1 / 6 * n
  let third_son := 1 / 8 * n
  let remaining := n - (first_son + second_son + third_son)
  remaining = 9 ↔ n = 24 := 
by
  -- Skipping proof, placeholder
  sorry

end herd_total_cows_l220_220678


namespace calculate_expression_l220_220309

theorem calculate_expression (a b c : ℕ) (h1 : a = 2011) (h2 : b = 2012) (h3 : c = 2013) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end calculate_expression_l220_220309


namespace least_possible_value_of_m_plus_n_l220_220530

theorem least_possible_value_of_m_plus_n 
(m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) 
(hgcd : Nat.gcd (m + n) 210 = 1) 
(hdiv : ∃ k, m^m = k * n^n)
(hnotdiv : ¬ ∃ k, m = k * n) : 
  m + n = 407 := 
sorry

end least_possible_value_of_m_plus_n_l220_220530


namespace polygon_sides_l220_220556

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 1620) : n = 11 := 
by 
  sorry

end polygon_sides_l220_220556


namespace evaluate_product_l220_220548

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 :=
by 
  sorry

end evaluate_product_l220_220548


namespace base_8_to_base_10_4652_l220_220970

def convert_base_8_to_base_10 (n : ℕ) : ℕ :=
  (4 * 8^3) + (6 * 8^2) + (5 * 8^1) + (2 * 8^0)

theorem base_8_to_base_10_4652 :
  convert_base_8_to_base_10 4652 = 2474 :=
by
  -- Skipping the proof steps
  sorry

end base_8_to_base_10_4652_l220_220970


namespace geometric_sequence_value_of_a_l220_220851

noncomputable def a : ℝ :=
sorry

theorem geometric_sequence_value_of_a
  (is_geometric_seq : ∀ (x y z : ℝ), z / y = y / x)
  (first_term : ℝ)
  (second_term : ℝ)
  (third_term : ℝ)
  (h1 : first_term = 140)
  (h2 : second_term = a)
  (h3 : third_term = 45 / 28)
  (pos_a : a > 0):
  a = 15 :=
sorry

end geometric_sequence_value_of_a_l220_220851


namespace ratio_of_intercepts_l220_220747

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end ratio_of_intercepts_l220_220747


namespace simplify_expression_l220_220558

theorem simplify_expression (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) =
  (3 * (x - (7 + Real.sqrt 37) / 6) * (x - (7 - Real.sqrt 37) / 6)) / ((x - 3) * (x + 2)) :=
by
  sorry

end simplify_expression_l220_220558


namespace second_hand_distance_l220_220876

theorem second_hand_distance (r : ℝ) (minutes : ℝ) : r = 8 → minutes = 45 → (2 * π * r * minutes) = 720 * π :=
by
  intros r_eq minutes_eq
  simp only [r_eq, minutes_eq, mul_assoc, mul_comm π 8, mul_mul_mul_comm]
  sorry

end second_hand_distance_l220_220876


namespace inequality_solution_l220_220939

-- Define the variable x as a real number
variable (x : ℝ)

-- Define the given condition that x is positive
def is_positive (x : ℝ) := x > 0

-- Define the condition that x satisfies the inequality sqrt(9x) < 3x^2
def satisfies_inequality (x : ℝ) := Real.sqrt (9 * x) < 3 * x^2

-- The statement we need to prove
theorem inequality_solution (x : ℝ) (h : is_positive x) : satisfies_inequality x ↔ x > 1 :=
sorry

end inequality_solution_l220_220939


namespace boat_speed_in_still_water_l220_220946

theorem boat_speed_in_still_water (v s : ℝ) (h1 : v + s = 15) (h2 : v - s = 7) : v = 11 := 
by
  sorry

end boat_speed_in_still_water_l220_220946


namespace intersection_of_A_and_B_l220_220555

def setA : Set ℝ := { x | x ≤ 4 }
def setB : Set ℝ := { x | x ≥ 1/2 }

theorem intersection_of_A_and_B : setA ∩ setB = { x | 1/2 ≤ x ∧ x ≤ 4 } := by
  sorry

end intersection_of_A_and_B_l220_220555


namespace number_of_chickens_l220_220374

theorem number_of_chickens (c k : ℕ) (h1 : c + k = 120) (h2 : 2 * c + 4 * k = 350) : c = 65 :=
by sorry

end number_of_chickens_l220_220374


namespace real_solutions_iff_a_geq_3_4_l220_220468

theorem real_solutions_iff_a_geq_3_4:
  (∃ (x y : ℝ), x + y^2 = a ∧ y + x^2 = a) ↔ a ≥ 3 / 4 := sorry

end real_solutions_iff_a_geq_3_4_l220_220468


namespace multimedia_sets_max_profit_l220_220465

-- Definitions of conditions:
def cost_A : ℝ := 3
def cost_B : ℝ := 2.4
def price_A : ℝ := 3.3
def price_B : ℝ := 2.8
def total_sets : ℕ := 50
def total_cost : ℝ := 132
def min_m : ℕ := 11

-- Problem 1: Prove the number of sets based on equations
theorem multimedia_sets (x y : ℕ) (h1 : x + y = total_sets) (h2 : cost_A * x + cost_B * y = total_cost) :
  x = 20 ∧ y = 30 :=
by sorry

-- Problem 2: Prove the maximum profit within a given range
theorem max_profit (m : ℕ) (h_m : 10 < m ∧ m < 20) :
  (-(0.1 : ℝ) * m + 20 = 18.9) ↔ m = min_m :=
by sorry

end multimedia_sets_max_profit_l220_220465


namespace value_of_a_l220_220084

noncomputable def function_f (x a : ℝ) : ℝ := (x - a) ^ 2 + (Real.log x ^ 2 - 2 * a) ^ 2

theorem value_of_a (x0 : ℝ) (a : ℝ) (h1 : x0 > 0) (h2 : function_f x0 a ≤ 4 / 5) : a = 1 / 5 :=
sorry

end value_of_a_l220_220084


namespace age_of_15th_person_l220_220913

variable (avg_age_20 : ℕ) (avg_age_5 : ℕ) (avg_age_9 : ℕ) (A : ℕ)
variable (num_20 : ℕ) (num_5 : ℕ) (num_9 : ℕ)

theorem age_of_15th_person (h1 : avg_age_20 = 15) (h2 : avg_age_5 = 14) (h3 : avg_age_9 = 16)
  (h4 : num_20 = 20) (h5 : num_5 = 5) (h6 : num_9 = 9) :
  (num_20 * avg_age_20) = (num_5 * avg_age_5) + (num_9 * avg_age_9) + A → A = 86 :=
by
  sorry

end age_of_15th_person_l220_220913


namespace vitamin_supplement_problem_l220_220297

theorem vitamin_supplement_problem :
  let packA := 7
  let packD := 17
  (∀ n : ℕ, n ≠ 0 → (packA * n = packD * n)) → n = 119 :=
by
  sorry

end vitamin_supplement_problem_l220_220297


namespace combined_age_of_siblings_l220_220910

-- We are given Aaron's age
def aaronAge : ℕ := 15

-- Henry's sister's age is three times Aaron's age
def henrysSisterAge : ℕ := 3 * aaronAge

-- Henry's age is four times his sister's age
def henryAge : ℕ := 4 * henrysSisterAge

-- The combined age of the siblings
def combinedAge : ℕ := aaronAge + henrysSisterAge + henryAge

theorem combined_age_of_siblings : combinedAge = 240 := by
  sorry

end combined_age_of_siblings_l220_220910


namespace initial_total_toys_l220_220818

-- Definitions based on the conditions
def initial_red_toys (R : ℕ) : Prop := R - 2 = 88
def twice_as_many_red_toys (R W : ℕ) : Prop := R - 2 = 2 * W

-- The proof statement: show that initially there were 134 toys in the box
theorem initial_total_toys (R W : ℕ) (hR : initial_red_toys R) (hW : twice_as_many_red_toys R W) : R + W = 134 := 
by sorry

end initial_total_toys_l220_220818


namespace scarlett_initial_oil_amount_l220_220478

theorem scarlett_initial_oil_amount (x : ℝ) (h : x + 0.67 = 0.84) : x = 0.17 :=
by sorry

end scarlett_initial_oil_amount_l220_220478


namespace find_percentage_l220_220325

theorem find_percentage (P N : ℝ) (h1 : (P / 100) * N = 60) (h2 : 0.80 * N = 240) : P = 20 :=
sorry

end find_percentage_l220_220325


namespace volume_ratio_l220_220919

theorem volume_ratio (V1 V2 M1 M2 : ℝ)
  (h1 : M1 / (V1 - M1) = 1 / 2)
  (h2 : M2 / (V2 - M2) = 3 / 2)
  (h3 : (M1 + M2) / (V1 - M1 + V2 - M2) = 1) :
  V1 / V2 = 9 / 5 :=
by
  sorry

end volume_ratio_l220_220919


namespace unique_solution_a_l220_220999

theorem unique_solution_a (a : ℚ) : 
  (∃ x : ℚ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0 ∧ 
  ∀ y : ℚ, (y ≠ x → (a^2 - 1) * y^2 + (a + 1) * y + 1 ≠ 0)) ↔ a = 1 ∨ a = 5/3 := 
sorry

end unique_solution_a_l220_220999


namespace circumcircle_trilinear_eq_incircle_trilinear_eq_excircle_trilinear_eq_l220_220127

-- Define the variables
variables {a b c : ℝ} {x y z : ℝ}
variables {α β γ : ℝ}

-- Circumcircle equation
theorem circumcircle_trilinear_eq :
  a * y * z + b * x * z + c * x * y = 0 :=
sorry

-- Incircle equation
theorem incircle_trilinear_eq :
  (Real.cos (α / 2) * Real.sqrt x) + 
  (Real.cos (β / 2) * Real.sqrt y) + 
  (Real.cos (γ / 2) * Real.sqrt z) = 0 :=
sorry

-- Excircle equation
theorem excircle_trilinear_eq :
  (Real.cos (α / 2) * Real.sqrt (-x)) + 
  (Real.cos (β / 2) * Real.sqrt y) + 
  (Real.cos (γ / 2) * Real.sqrt z) = 0 :=
sorry

end circumcircle_trilinear_eq_incircle_trilinear_eq_excircle_trilinear_eq_l220_220127


namespace parabola_value_l220_220315

theorem parabola_value (b c : ℝ) (h : 3 = -(-2) ^ 2 + b * -2 + c) : 2 * c - 4 * b - 9 = 5 := by
  sorry

end parabola_value_l220_220315


namespace solve_first_system_solve_second_system_l220_220563

-- Define the first system of equations
def first_system (x y : ℝ) : Prop := (3 * x + 2 * y = 5) ∧ (y = 2 * x - 8)

-- Define the solution to the first system
def solution1 (x y : ℝ) : Prop := (x = 3) ∧ (y = -2)

-- Define the second system of equations
def second_system (x y : ℝ) : Prop := (2 * x - y = 10) ∧ (2 * x + 3 * y = 2)

-- Define the solution to the second system
def solution2 (x y : ℝ) : Prop := (x = 4) ∧ (y = -2)

-- Define the problem statement in Lean
theorem solve_first_system : ∃ x y : ℝ, first_system x y ↔ solution1 x y :=
by
  sorry

theorem solve_second_system : ∃ x y : ℝ, second_system x y ↔ solution2 x y :=
by
  sorry

end solve_first_system_solve_second_system_l220_220563


namespace arithmetic_mean_of_roots_l220_220622

-- Definitions corresponding to the conditions
def quadratic_eqn (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- The term statement for the quadratic equation mean
theorem arithmetic_mean_of_roots : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 1 → (∃ (x1 x2 : ℝ), quadratic_eqn a b c x1 ∧ quadratic_eqn a b c x2 ∧ -4 / 2 = -2) :=
by
  -- skip the proof
  sorry

end arithmetic_mean_of_roots_l220_220622


namespace marching_band_members_l220_220985

theorem marching_band_members :
  ∃ (n : ℕ), 100 < n ∧ n < 200 ∧
             n % 4 = 1 ∧
             n % 5 = 2 ∧
             n % 7 = 3 :=
  by sorry

end marching_band_members_l220_220985


namespace candy_eaten_l220_220034

theorem candy_eaten 
  {initial_pieces remaining_pieces eaten_pieces : ℕ} 
  (h₁ : initial_pieces = 12) 
  (h₂ : remaining_pieces = 3) 
  (h₃ : eaten_pieces = initial_pieces - remaining_pieces) 
  : eaten_pieces = 9 := 
by 
  sorry

end candy_eaten_l220_220034


namespace simplify_expression_l220_220092

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = 2 * x⁻¹ * y⁻¹ * z⁻¹ :=
by
  sorry

end simplify_expression_l220_220092


namespace necessary_and_sufficient_condition_l220_220326

-- Sum of the first n terms of the sequence
noncomputable def S_n (n : ℕ) (c : ℤ) : ℤ := (n + 1) * (n + 1) + c

-- The nth term of the sequence
noncomputable def a_n (n : ℕ) (c : ℤ) : ℤ := S_n n c - (S_n (n - 1) c)

-- Define the sequence being arithmetic
noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) - a n = d

theorem necessary_and_sufficient_condition (c : ℤ) :
  (∀ n ≥ 1, a_n n c - a_n (n-1) c = 2) ↔ (c = -1) :=
by
  sorry

end necessary_and_sufficient_condition_l220_220326


namespace min_value_exp_sum_eq_4sqrt2_l220_220769

theorem min_value_exp_sum_eq_4sqrt2 {a b : ℝ} (h : a + b = 3) : 2^a + 2^b ≥ 4 * Real.sqrt 2 :=
by
  sorry

end min_value_exp_sum_eq_4sqrt2_l220_220769


namespace chocolate_squares_remaining_l220_220996

theorem chocolate_squares_remaining (m : ℕ) : m * 6 - 21 = 45 :=
by
  sorry

end chocolate_squares_remaining_l220_220996


namespace eval_expression_l220_220207

def square_avg (a b : ℚ) : ℚ := (a^2 + b^2) / 2
def custom_avg (a b c : ℚ) : ℚ := (a + b + 2 * c) / 3

theorem eval_expression : 
  custom_avg (custom_avg 2 (-1) 1) (square_avg 2 3) 1 = 19 / 6 :=
by
  sorry

end eval_expression_l220_220207


namespace enrollment_inversely_proportional_l220_220636

theorem enrollment_inversely_proportional :
  ∃ k : ℝ, (40 * 2000 = k) → (s * 2500 = k) → s = 32 :=
by
  sorry

end enrollment_inversely_proportional_l220_220636


namespace only_A_forms_triangle_l220_220597

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_A_forms_triangle :
  (triangle_inequality 5 6 10) ∧ ¬(triangle_inequality 5 2 9) ∧ ¬(triangle_inequality 5 7 12) ∧ ¬(triangle_inequality 3 4 8) :=
by
  sorry

end only_A_forms_triangle_l220_220597


namespace arithmetic_geometric_sequence_l220_220930

theorem arithmetic_geometric_sequence
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (S : ℕ → ℝ)
  (f : ℕ → ℝ)
  (h₁ : a 1 = 3)
  (h₂ : b 1 = 1)
  (h₃ : b 2 * S 2 = 64)
  (h₄ : b 3 * S 3 = 960)
  : (∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 8^(n - 1)) ∧ 
    (∀ n, f n = (a n - 1) / (S n + 100)) ∧ 
    (∃ n, f n = 1 / 11 ∧ n = 10) := 
sorry

end arithmetic_geometric_sequence_l220_220930


namespace value_of_a_l220_220143

theorem value_of_a (a : ℝ) :
  (∀ x y : ℝ, x + a^2 * y + 6 = 0 → (a-2) * x + 3 * a * y + 2 * a = 0) →
  (a = 0 ∨ a = -1) :=
by
  sorry

end value_of_a_l220_220143


namespace derivative_at_one_l220_220720

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + x^2) : f' 1 = -2 :=
by
  sorry

end derivative_at_one_l220_220720


namespace alpha_beta_sum_equal_two_l220_220894

theorem alpha_beta_sum_equal_two (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0) 
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := 
sorry

end alpha_beta_sum_equal_two_l220_220894


namespace squares_ratio_l220_220616

noncomputable def inscribed_squares_ratio :=
  let x := 60 / 17
  let y := 780 / 169
  (x / y : ℚ)

theorem squares_ratio (x y : ℚ) (h₁ : x = 60 / 17) (h₂ : y = 780 / 169) :
  x / y = 169 / 220 := by
  rw [h₁, h₂]
  -- Here we would perform calculations to show equality, omitted for brevity.
  sorry

end squares_ratio_l220_220616


namespace relative_frequency_defective_books_l220_220626

theorem relative_frequency_defective_books 
  (N_defective : ℤ) (N_total : ℤ)
  (h_defective : N_defective = 5)
  (h_total : N_total = 100) :
  (N_defective : ℚ) / N_total = 0.05 := by
  sorry

end relative_frequency_defective_books_l220_220626


namespace coefficient_fifth_term_expansion_l220_220381

theorem coefficient_fifth_term_expansion :
  let a := (2 : ℝ)
  let b := -(1 : ℝ)
  let n := 6
  let k := 4
  Nat.choose n k * (a ^ (n - k)) * (b ^ k) = 60 := by
  -- We can assume x to be any nonzero real, but it is not needed in the theorem itself.
  sorry

end coefficient_fifth_term_expansion_l220_220381


namespace min_english_score_l220_220968

theorem min_english_score (A B : ℕ) (h_avg_AB : (A + B) / 2 = 90) : 
  ∀ E : ℕ, ((A + B + E) / 3 ≥ 92) ↔ E ≥ 96 := by
  sorry

end min_english_score_l220_220968


namespace remainder_modulus_9_l220_220628

theorem remainder_modulus_9 : (9 * 7^18 + 2^18) % 9 = 1 := 
by sorry

end remainder_modulus_9_l220_220628


namespace find_const_functions_l220_220608

theorem find_const_functions
  (f g : ℝ → ℝ)
  (hf : ∀ x y : ℝ, 0 < x → 0 < y → f (x^2 + y^2) = g (x * y)) :
  ∃ c : ℝ, (∀ x, 0 < x → f x = c) ∧ (∀ x, 0 < x → g x = c) :=
sorry

end find_const_functions_l220_220608


namespace ron_tickets_sold_l220_220444

theorem ron_tickets_sold 
  (R K : ℕ) 
  (h1 : R + K = 20) 
  (h2 : 2 * R + 9 / 2 * K = 60) : 
  R = 12 := 
by 
  sorry

end ron_tickets_sold_l220_220444


namespace base_area_of_cuboid_l220_220278

theorem base_area_of_cuboid (V h : ℝ) (hv : V = 144) (hh : h = 8) : ∃ A : ℝ, A = 18 := by
  sorry

end base_area_of_cuboid_l220_220278


namespace divisibility_3804_l220_220103

theorem divisibility_3804 (n : ℕ) (h : 0 < n) :
    3804 ∣ ((n ^ 3 - n) * (5 ^ (8 * n + 4) + 3 ^ (4 * n + 2))) :=
sorry

end divisibility_3804_l220_220103


namespace Mary_younger_by_14_l220_220331

variable (Betty_age : ℕ) (Albert_age : ℕ) (Mary_age : ℕ)

theorem Mary_younger_by_14 :
  (Betty_age = 7) →
  (Albert_age = 4 * Betty_age) →
  (Albert_age = 2 * Mary_age) →
  (Albert_age - Mary_age = 14) :=
by
  intros
  sorry

end Mary_younger_by_14_l220_220331


namespace skilled_picker_capacity_minimize_costs_l220_220348

theorem skilled_picker_capacity (x : ℕ) (h1 : ∀ x : ℕ, ∀ s : ℕ, s = 3 * x) (h2 : 450 * 25 = 3 * x * 25 + 600) :
  s = 30 :=
by
  sorry

theorem minimize_costs (s n m : ℕ)
(h1 : s ≤ 20)
(h2 : n ≤ 15)
(h3 : 600 = s * 30 + n * 10)
(h4 : ∀ y, y = s * 300 + n * 80) :
  m = 15 ∧ s = 15 :=
by
  sorry

end skilled_picker_capacity_minimize_costs_l220_220348


namespace less_than_its_reciprocal_l220_220740

-- Define the numbers as constants
def a := -1/3
def b := -3/2
def c := 1/4
def d := 3/4
def e := 4/3 

-- Define the proposition that needs to be proved
theorem less_than_its_reciprocal (n : ℚ) :
  (n = -3/2 ∨ n = 1/4) ↔ (n < 1/n) :=
by
  sorry

end less_than_its_reciprocal_l220_220740


namespace average_increase_l220_220377

theorem average_increase (A A' : ℕ) (runs_in_17th : ℕ) (total_innings : ℕ) (new_avg : ℕ) 
(h1 : total_innings = 17)
(h2 : runs_in_17th = 87)
(h3 : new_avg = 39)
(h4 : A' = new_avg)
(h5 : 16 * A + runs_in_17th = total_innings * new_avg) 
: A' - A = 3 := by
  sorry

end average_increase_l220_220377


namespace eric_time_ratio_l220_220798

-- Defining the problem context
def eric_runs : ℕ := 20
def eric_jogs : ℕ := 10
def eric_return_time : ℕ := 90

-- The ratio is represented as a fraction
def ratio (a b : ℕ) := a / b

-- Stating the theorem
theorem eric_time_ratio :
  ratio eric_return_time (eric_runs + eric_jogs) = 3 :=
by
  sorry

end eric_time_ratio_l220_220798


namespace min_tiles_l220_220167

theorem min_tiles (x y : ℕ) (h1 : 25 * x + 9 * y = 2014) (h2 : ∀ a b, 25 * a + 9 * b = 2014 -> (a + b) >= (x + y)) : x + y = 94 :=
  sorry

end min_tiles_l220_220167


namespace percent_greater_than_l220_220711

theorem percent_greater_than (M N : ℝ) (hN : N ≠ 0) : (M - N) / N * 100 = 100 * (M - N) / N :=
by sorry

end percent_greater_than_l220_220711


namespace contradiction_proof_l220_220842

theorem contradiction_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h1 : a + 1/b < 2) (h2 : b + 1/c < 2) (h3 : c + 1/a < 2) : 
  ¬ (a + 1/b ≥ 2 ∨ b + 1/c ≥ 2 ∨ c + 1/a ≥ 2) :=
by
  sorry

end contradiction_proof_l220_220842


namespace chickens_increased_l220_220130

-- Definitions and conditions
def initial_chickens := 45
def chickens_bought_day1 := 18
def chickens_bought_day2 := 12
def total_chickens_bought := chickens_bought_day1 + chickens_bought_day2

-- Proof statement
theorem chickens_increased :
  total_chickens_bought = 30 :=
by
  sorry

end chickens_increased_l220_220130


namespace non_degenerate_ellipse_l220_220974

theorem non_degenerate_ellipse (k : ℝ) : (∃ (x y : ℝ), x^2 + 4*y^2 - 10*x + 56*y = k) ↔ k > -221 :=
sorry

end non_degenerate_ellipse_l220_220974


namespace gallons_in_pond_after_50_days_l220_220272

def initial_amount : ℕ := 500
def evaporation_rate : ℕ := 1
def days_passed : ℕ := 50
def total_evaporation : ℕ := days_passed * evaporation_rate
def final_amount : ℕ := initial_amount - total_evaporation

theorem gallons_in_pond_after_50_days : final_amount = 450 := by
  sorry

end gallons_in_pond_after_50_days_l220_220272


namespace inequality_holds_l220_220617

theorem inequality_holds (c : ℝ) (X Y : ℝ) (h1 : X^2 - c * X - c = 0) (h2 : Y^2 - c * Y - c = 0) :
    X^3 + Y^3 + (X * Y)^3 ≥ 0 :=
sorry

end inequality_holds_l220_220617


namespace sum_of_first_n_terms_l220_220824

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def forms_geometric_sequence (a2 a4 a8 : ℤ) : Prop :=
  a4^2 = a2 * a8

def arithmetic_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (n : ℕ) : Prop :=
  S n = n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1)

theorem sum_of_first_n_terms
  (d : ℤ) (n : ℕ)
  (h_nonzero : d ≠ 0)
  (h_arithmetic : is_arithmetic_sequence a d)
  (h_initial : a 1 = 1)
  (h_geom : forms_geometric_sequence (a 2) (a 4) (a 8)) :
  S n = n * (n + 1) / 2 := 
sorry

end sum_of_first_n_terms_l220_220824


namespace tangerine_and_orange_percentage_l220_220964

-- Given conditions
def initial_apples := 9
def initial_oranges := 5
def initial_tangerines := 17
def initial_grapes := 12
def initial_kiwis := 7

def removed_oranges := 2
def removed_tangerines := 10
def removed_grapes := 4
def removed_kiwis := 3

def added_oranges := 3
def added_tangerines := 6

-- Computed values based on the initial conditions and changes
def remaining_apples := initial_apples
def remaining_oranges := initial_oranges - removed_oranges + added_oranges
def remaining_tangerines := initial_tangerines - removed_tangerines + added_tangerines
def remaining_grapes := initial_grapes - removed_grapes
def remaining_kiwis := initial_kiwis - removed_kiwis

def total_remaining_fruits := remaining_apples + remaining_oranges + remaining_tangerines + remaining_grapes + remaining_kiwis
def total_citrus_fruits := remaining_oranges + remaining_tangerines

-- Statement to prove
def citrus_percentage := (total_citrus_fruits : ℚ) / total_remaining_fruits * 100

theorem tangerine_and_orange_percentage : citrus_percentage = 47.5 := by
  sorry

end tangerine_and_orange_percentage_l220_220964


namespace igor_min_score_needed_l220_220440

theorem igor_min_score_needed
  (scores : List ℕ)
  (goal : ℚ)
  (next_test_score : ℕ)
  (h_scores : scores = [88, 92, 75, 83, 90])
  (h_goal : goal = 87)
  (h_solution : next_test_score = 94)
  : 
  let current_sum := scores.sum
  let current_tests := scores.length
  let required_total := (goal * (current_tests + 1))
  let next_test_needed := required_total - current_sum
  next_test_needed ≤ next_test_score := 
by 
  sorry

end igor_min_score_needed_l220_220440


namespace min_sum_abc_l220_220873

theorem min_sum_abc (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1716) :
  a + b + c = 31 :=
sorry

end min_sum_abc_l220_220873


namespace factorial_division_l220_220878

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division : (factorial 15) / ((factorial 6) * (factorial 9)) = 5005 :=
by
  sorry

end factorial_division_l220_220878


namespace excess_percentage_l220_220687

theorem excess_percentage (x : ℝ) 
  (L W : ℝ) (hL : L > 0) (hW : W > 0) 
  (h1 : L * (1 + x / 100) * W * 0.96 = L * W * 1.008) : 
  x = 5 :=
by sorry

end excess_percentage_l220_220687


namespace solve_for_x_l220_220977

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 2984) : x = 20.04 :=
by
  sorry

end solve_for_x_l220_220977


namespace carla_receives_correct_amount_l220_220863

theorem carla_receives_correct_amount (L B C X : ℝ) : 
  (L + B + C + X) / 3 - (C + X) = (L + B - 2 * C - 2 * X) / 3 :=
by
  sorry

end carla_receives_correct_amount_l220_220863


namespace ratio_sheep_horses_l220_220522

theorem ratio_sheep_horses (amount_food_per_horse : ℕ) (total_food_per_day : ℕ) (num_sheep : ℕ) (num_horses : ℕ) :
  amount_food_per_horse = 230 ∧ total_food_per_day = 12880 ∧ num_sheep = 24 ∧ num_horses = total_food_per_day / amount_food_per_horse →
  num_sheep / num_horses = 3 / 7 :=
by
  sorry

end ratio_sheep_horses_l220_220522


namespace compare_polynomials_l220_220610

theorem compare_polynomials (x : ℝ) : 2 * x^2 - 2 * x + 1 > x^2 - 2 * x := 
by
  sorry

end compare_polynomials_l220_220610


namespace example_of_four_three_digit_numbers_sum_2012_two_digits_exists_l220_220984

-- Define what it means to be a three-digit number using only two distinct digits
def two_digit_natural (d1 d2 : ℕ) (n : ℕ) : Prop :=
  (∀ (d : ℕ), d ∈ n.digits 10 → d = d1 ∨ d = d2) ∧ 100 ≤ n ∧ n < 1000

-- State the main theorem
theorem example_of_four_three_digit_numbers_sum_2012_two_digits_exists :
  ∃ a b c d : ℕ, 
    two_digit_natural 3 5 a ∧
    two_digit_natural 3 5 b ∧
    two_digit_natural 3 5 c ∧
    two_digit_natural 3 5 d ∧
    a + b + c + d = 2012 :=
by
  sorry

end example_of_four_three_digit_numbers_sum_2012_two_digits_exists_l220_220984


namespace fraction_identity_l220_220114

open Real

theorem fraction_identity
  (p q r : ℝ)
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) = 8) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) = 2.2 :=
  sorry

end fraction_identity_l220_220114


namespace repetend_of_4_div_17_l220_220858

theorem repetend_of_4_div_17 :
  ∃ (r : String), (∀ (n : ℕ), (∃ (k : ℕ), (0 < k) ∧ (∃ (q : ℤ), (4 : ℤ) * 10 ^ (n + 12 * k) / 17 % 10 ^ 12 = q)) ∧ r = "235294117647") :=
sorry

end repetend_of_4_div_17_l220_220858


namespace last_number_nth_row_sum_of_nth_row_position_of_2008_l220_220550

theorem last_number_nth_row (n : ℕ) : 
  ∃ last_number, last_number = 2^n - 1 := by
  sorry

theorem sum_of_nth_row (n : ℕ) : 
  ∃ sum_nth_row, sum_nth_row = 2^(2*n-2) + 2^(2*n-3) - 2^(n-2) := by
  sorry

theorem position_of_2008 : 
  ∃ (row : ℕ) (position : ℕ), row = 11 ∧ position = 2008 - 2^10 + 1 :=
  by sorry

end last_number_nth_row_sum_of_nth_row_position_of_2008_l220_220550


namespace cost_per_adult_meal_l220_220685

-- Definitions and given conditions
def total_people : ℕ := 13
def num_kids : ℕ := 9
def total_cost : ℕ := 28

-- Question translated into a proof statement
theorem cost_per_adult_meal : (total_cost / (total_people - num_kids)) = 7 := 
by
  sorry

end cost_per_adult_meal_l220_220685


namespace coin_problem_l220_220459

theorem coin_problem :
  ∃ (p n d q : ℕ), p + n + d + q = 11 ∧ 
                   1 * p + 5 * n + 10 * d + 25 * q = 132 ∧
                   p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧ 
                   q = 3 :=
by
  sorry

end coin_problem_l220_220459


namespace vasya_birthday_day_l220_220696

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l220_220696


namespace sphere_surface_area_l220_220902

noncomputable def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem sphere_surface_area (r_circle r_distance : ℝ) :
  (Real.pi * r_circle^2 = 16 * Real.pi) →
  (r_distance = 3) →
  (surface_area_of_sphere (Real.sqrt (r_distance^2 + r_circle^2)) = 100 * Real.pi) := by
sorry

end sphere_surface_area_l220_220902


namespace triangle_table_distinct_lines_l220_220191

theorem triangle_table_distinct_lines (a : ℕ) (h : a > 1) : 
  ∀ (n : ℕ) (line : ℕ → ℕ), 
  (line 0 = a) → 
  (∀ k, line (2*k + 1) = line k ^ 2 ∧ line (2*k + 2) = line k + 1) → 
  ∀ i j, i < 2^n → j < 2^n → (i ≠ j → line i ≠ line j) := 
by {
  sorry
}

end triangle_table_distinct_lines_l220_220191


namespace problem_proof_l220_220959

-- Assume definitions for lines and planes, and their relationships like parallel and perpendicular exist.

variables (m n : Line) (α β : Plane)

-- Define conditions
def line_is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_is_parallel_to_line (l1 l2 : Line) : Prop := sorry
def planes_are_perpendicular (p1 p2 : Plane) : Prop := sorry

-- Problem statement
theorem problem_proof :
  (line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n α) → 
  (line_is_parallel_to_line m n) ∧
  ((line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n β) ∧ (line_is_perpendicular_to_plane m n) → 
  (planes_are_perpendicular α β)) := 
sorry

end problem_proof_l220_220959


namespace find_valid_pairs_l220_220365

theorem find_valid_pairs :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 12 →
  (∃ C : ℤ, ∀ (n : ℕ), 0 < n → (a^n + b^(n+9)) % 13 = C % 13) ↔
  (a, b) = (1, 1) ∨ (a, b) = (4, 4) ∨ (a, b) = (10, 10) ∨ (a, b) = (12, 12) := 
by
  sorry

end find_valid_pairs_l220_220365


namespace jessica_marbles_62_l220_220253

-- Definitions based on conditions
def marbles_kurt (marbles_dennis : ℕ) : ℕ := marbles_dennis - 45
def marbles_laurie (marbles_kurt : ℕ) : ℕ := marbles_kurt + 12
def marbles_jessica (marbles_laurie : ℕ) : ℕ := marbles_laurie + 25

-- Given marbles for Dennis
def marbles_dennis : ℕ := 70

-- Proof statement: Prove that Jessica has 62 marbles given the conditions
theorem jessica_marbles_62 : marbles_jessica (marbles_laurie (marbles_kurt marbles_dennis)) = 62 := 
by
  sorry

end jessica_marbles_62_l220_220253


namespace election_majority_l220_220535

theorem election_majority (V : ℝ) 
  (h1 : ∃ w l : ℝ, w = 0.70 * V ∧ l = 0.30 * V ∧ w - l = 174) : 
  V = 435 :=
by
  sorry

end election_majority_l220_220535


namespace solution_set_of_inequality_l220_220422

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2 * a * x - 3 * a^2 < 0} = {x : ℝ | 3 * a < x ∧ x < -a} :=
by
  sorry

end solution_set_of_inequality_l220_220422


namespace find_a_l220_220943

noncomputable def f (a x : ℝ) := (x - 1)^2 + a * x + Real.cos x

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → 
  a = 2 :=
by
  sorry

end find_a_l220_220943


namespace find_x_value_l220_220249

open Real

theorem find_x_value (a : ℝ) (x : ℝ) (h : a > 0) (h_eq : 10^x = log (10 * a) + log (a⁻¹)) : x = 0 :=
by
  sorry

end find_x_value_l220_220249


namespace cereal_difference_l220_220789

-- Variables to represent the amounts of cereal in each box
variable (A B C : ℕ)

-- Define the conditions given in the problem
def problem_conditions : Prop :=
  A = 14 ∧
  B = A / 2 ∧
  A + B + C = 33

-- Prove the desired conclusion under these conditions
theorem cereal_difference
  (h : problem_conditions A B C) :
  C - B = 5 :=
sorry

end cereal_difference_l220_220789


namespace joe_money_left_l220_220010

theorem joe_money_left
  (initial_money : ℕ) (notebook_cost : ℕ) (notebooks : ℕ)
  (book_cost : ℕ) (books : ℕ) (pen_cost : ℕ) (pens : ℕ)
  (sticker_pack_cost : ℕ) (sticker_packs : ℕ) (charity : ℕ)
  (remaining_money : ℕ) :
  initial_money = 150 →
  notebook_cost = 4 →
  notebooks = 7 →
  book_cost = 12 →
  books = 2 →
  pen_cost = 2 →
  pens = 5 →
  sticker_pack_cost = 6 →
  sticker_packs = 3 →
  charity = 10 →
  remaining_money = 60 →
  remaining_money = 
    initial_money - 
    ((notebooks * notebook_cost) + 
     (books * book_cost) + 
     (pens * pen_cost) + 
     (sticker_packs * sticker_pack_cost) + 
     charity) := 
by
  intros; sorry

end joe_money_left_l220_220010


namespace tips_fraction_l220_220834

theorem tips_fraction (S T I : ℝ) (hT : T = 9 / 4 * S) (hI : I = S + T) : 
  T / I = 9 / 13 := 
by 
  sorry

end tips_fraction_l220_220834


namespace tree_height_at_year_3_l220_220410

theorem tree_height_at_year_3 :
  ∃ h₃ : ℕ, h₃ = 27 ∧
  (∃ h₇ h₆ h₅ h₄ : ℕ,
   h₇ = 648 ∧
   h₆ = h₇ / 2 ∧
   h₅ = h₆ / 2 ∧
   h₄ = h₅ / 2 ∧
   h₄ = 3 * h₃) :=
by
  sorry

end tree_height_at_year_3_l220_220410


namespace cakes_served_during_lunch_today_l220_220663

theorem cakes_served_during_lunch_today (L : ℕ) 
  (h_total : L + 6 + 3 = 14) : 
  L = 5 :=
sorry

end cakes_served_during_lunch_today_l220_220663


namespace max_true_statements_l220_220030

theorem max_true_statements (x : ℝ) :
  (∀ x, -- given the conditions
    (0 < x^2 ∧ x^2 < 1) →
    (x^2 > 1) →
    (-1 < x ∧ x < 0) →
    (0 < x ∧ x < 1) →
    (0 < x - x^2 ∧ x - x^2 < 1)) →
  -- Prove the maximum number of these statements that can be true is 3
  (∃ (count : ℕ), count = 3) :=
sorry

end max_true_statements_l220_220030


namespace value_of_expression_is_one_l220_220245

theorem value_of_expression_is_one : 
  ∃ (a b c d : ℚ), (a = 1) ∧ (b = -1) ∧ (c = 0) ∧ (d = 1 ∨ d = -1) ∧ (a - b + c^2 - |d| = 1) :=
by
  sorry

end value_of_expression_is_one_l220_220245


namespace dvaneft_percentage_bounds_l220_220795

theorem dvaneft_percentage_bounds (x y z : ℝ) (n m : ℕ) 
  (h1 : x * n + y * m = z * (m + n))
  (h2 : 3 * x * n = y * m)
  (h3_1 : 10 ≤ y - x)
  (h3_2 : y - x ≤ 18)
  (h4_1 : 18 ≤ z)
  (h4_2 : z ≤ 42)
  : (15 ≤ (n:ℝ) / (2 * (n + m)) * 100) ∧ ((n:ℝ) / (2 * (n + m)) * 100 ≤ 25) :=
by
  sorry

end dvaneft_percentage_bounds_l220_220795


namespace chloe_profit_l220_220997

theorem chloe_profit 
  (cost_per_dozen : ℕ)
  (selling_price_per_half_dozen : ℕ)
  (dozens_sold : ℕ)
  (h1 : cost_per_dozen = 50)
  (h2 : selling_price_per_half_dozen = 30)
  (h3 : dozens_sold = 50) : 
  (selling_price_per_half_dozen - cost_per_dozen / 2) * (dozens_sold * 2) = 500 :=
by 
  sorry

end chloe_profit_l220_220997


namespace speed_of_second_train_is_16_l220_220038

def speed_second_train (v : ℝ) : Prop :=
  ∃ t : ℝ, 
    (20 * t = v * t + 70) ∧ -- Condition: the first train traveled 70 km more than the second train
    (20 * t + v * t = 630)  -- Condition: total distance between stations

theorem speed_of_second_train_is_16 : speed_second_train 16 :=
by
  sorry

end speed_of_second_train_is_16_l220_220038


namespace parabola_x_intercepts_incorrect_l220_220260

-- Define the given quadratic function
noncomputable def f (x : ℝ) : ℝ := -1 / 2 * (x - 1)^2 + 2

-- The Lean statement for the problem
theorem parabola_x_intercepts_incorrect :
  ¬ ((f 3 = 0) ∧ (f (-3) = 0)) :=
by
  sorry

end parabola_x_intercepts_incorrect_l220_220260


namespace KarenParagraphCount_l220_220745

theorem KarenParagraphCount :
  ∀ (num_essays num_short_ans num_paragraphs total_time essay_time short_ans_time paragraph_time : ℕ),
    (num_essays = 2) →
    (num_short_ans = 15) →
    (total_time = 240) →
    (essay_time = 60) →
    (short_ans_time = 3) →
    (paragraph_time = 15) →
    (total_time = num_essays * essay_time + num_short_ans * short_ans_time + num_paragraphs * paragraph_time) →
    num_paragraphs = 5 :=
by
  sorry

end KarenParagraphCount_l220_220745


namespace length_AD_of_circle_l220_220403

def circle_radius : ℝ := 8
def p_A : Prop := True  -- stand-in for the point A on the circle
def p_B : Prop := True  -- stand-in for the point B on the circle
def dist_AB : ℝ := 10
def p_D : Prop := True  -- stand-in for point D opposite B

theorem length_AD_of_circle 
  (r : ℝ := circle_radius)
  (A B D : Prop)
  (h_AB : dist_AB = 10)
  (h_radius : r = 8)
  (h_opposite : D)
  : ∃ AD : ℝ, AD = Real.sqrt 252.75 :=
sorry

end length_AD_of_circle_l220_220403


namespace horse_food_per_day_l220_220508

theorem horse_food_per_day
  (total_horse_food_per_day : ℕ)
  (sheep_count : ℕ)
  (sheep_to_horse_ratio : ℕ)
  (horse_to_sheep_ratio : ℕ)
  (horse_food_per_horse_per_day : ℕ) :
  sheep_to_horse_ratio * horse_food_per_horse_per_day = total_horse_food_per_day / (sheep_count / sheep_to_horse_ratio * horse_to_sheep_ratio) :=
by
  -- Given
  let total_horse_food_per_day := 12880
  let sheep_count := 24
  let sheep_to_horse_ratio := 3
  let horse_to_sheep_ratio := 7

  -- We need to show that horse_food_per_horse_per_day = 230
  have horse_count : ℕ := (sheep_count / sheep_to_horse_ratio) * horse_to_sheep_ratio
  have horse_food_per_horse_per_day : ℕ := total_horse_food_per_day / horse_count

  -- Desired proof statement
  sorry

end horse_food_per_day_l220_220508


namespace initial_value_calculation_l220_220722

theorem initial_value_calculation (P : ℝ) (h1 : ∀ n : ℕ, 0 ≤ n →
                                (P:ℝ) * (1 + 1/8) ^ n = 78468.75 → n = 2) :
  P = 61952 :=
sorry

end initial_value_calculation_l220_220722


namespace integer_multiplication_for_ones_l220_220975

theorem integer_multiplication_for_ones :
  ∃ x : ℤ, (10^9 - 1) * x = (10^81 - 1) / 9 :=
by
  sorry

end integer_multiplication_for_ones_l220_220975


namespace total_books_l220_220666

variable (Sandy_books Benny_books Tim_books : ℕ)
variable (h_Sandy : Sandy_books = 10)
variable (h_Benny : Benny_books = 24)
variable (h_Tim : Tim_books = 33)

theorem total_books :
  Sandy_books + Benny_books + Tim_books = 67 :=
by sorry

end total_books_l220_220666


namespace adults_in_each_group_l220_220013

theorem adults_in_each_group (A : ℕ) :
  (∃ n : ℕ, n >= 17 ∧ n * 15 = 255) →
  (∃ m : ℕ, m * A = 255 ∧ m >= 17) →
  A = 15 :=
by
  intros h_child_groups h_adult_groups
  -- Use sorry to skip the proof
  sorry

end adults_in_each_group_l220_220013


namespace domain_of_function_l220_220454

theorem domain_of_function : 
  {x : ℝ | x + 1 ≥ 0 ∧ x ≠ 1} = {x : ℝ | -1 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x} :=
by 
  sorry

end domain_of_function_l220_220454


namespace triangle_interior_angle_contradiction_l220_220972

theorem triangle_interior_angle_contradiction :
  (∀ (A B C : ℝ), A + B + C = 180 ∧ A > 60 ∧ B > 60 ∧ C > 60 → false) :=
by
  sorry

end triangle_interior_angle_contradiction_l220_220972


namespace quadratic_root_condition_l220_220126

theorem quadratic_root_condition (k : ℝ) :
  (∀ (x : ℝ), x^2 + k * x + 4 * k^2 - 3 = 0 → ∃ x1 x2 : ℝ, x1 + x2 = (-k) ∧ x1 * x2 = 4 * k^2 - 3 ∧ x1 + x2 = x1 * x2) →
  k = 3 / 4 :=
by
  sorry

end quadratic_root_condition_l220_220126


namespace seating_arrangement_l220_220027

def num_ways_seated (total_passengers : ℕ) (window_seats : ℕ) : ℕ :=
  window_seats * (total_passengers - 1) * (total_passengers - 2) * (total_passengers - 3)

theorem seating_arrangement (passengers_seats taxi_window_seats : ℕ)
  (h1 : passengers_seats = 4) (h2 : taxi_window_seats = 2) :
  num_ways_seated passengers_seats taxi_window_seats = 12 :=
by
  -- proof will go here
  sorry

end seating_arrangement_l220_220027


namespace integer_square_root_35_consecutive_l220_220948

theorem integer_square_root_35_consecutive : 
  ∃ n : ℕ, ∀ k : ℕ, n^2 ≤ k ∧ k < (n+1)^2 ∧ ((n + 1)^2 - n^2 = 35) ∧ (n = 17) := by 
  sorry

end integer_square_root_35_consecutive_l220_220948


namespace trapezoid_midsegment_l220_220436

-- Define the problem conditions and question
theorem trapezoid_midsegment (b h x : ℝ) (h_nonzero : h ≠ 0) (hx : x = b + 75)
  (equal_areas : (1 / 2) * (h / 2) * (b + (b + 75)) = (1 / 2) * (h / 2) * ((b + 75) + (b + 150))) :
  ∃ n : ℤ, n = ⌊x^2 / 120⌋ ∧ n = 3000 := 
by 
  sorry

end trapezoid_midsegment_l220_220436


namespace pablo_mother_pays_each_page_l220_220847

-- Definitions based on the conditions in the problem
def pages_per_book := 150
def number_books_read := 12
def candy_cost := 15
def money_leftover := 3
def total_money := candy_cost + money_leftover
def total_pages := number_books_read * pages_per_book
def amount_paid_per_page := total_money / total_pages

-- The theorem to be proven
theorem pablo_mother_pays_each_page
    (pages_per_book : ℝ)
    (number_books_read : ℝ)
    (candy_cost : ℝ)
    (money_leftover : ℝ)
    (total_money := candy_cost + money_leftover)
    (total_pages := number_books_read * pages_per_book)
    (amount_paid_per_page := total_money / total_pages) :
    amount_paid_per_page = 0.01 :=
by
  sorry

end pablo_mother_pays_each_page_l220_220847


namespace range_of_x_plus_y_l220_220797

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 + 2 * x * y + 4 * y^2 = 1) : 0 < x + y ∧ x + y < 1 :=
by
  sorry

end range_of_x_plus_y_l220_220797


namespace decreasing_function_l220_220273

theorem decreasing_function (m : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → (m + 3) * x1 - 2 > (m + 3) * x2 - 2) ↔ m < -3 :=
by
  sorry

end decreasing_function_l220_220273


namespace remainder_of_sum_l220_220954

theorem remainder_of_sum :
  (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 20 = 18 :=
by
  sorry

end remainder_of_sum_l220_220954


namespace smallest_value_fraction_l220_220551

theorem smallest_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ k : ℝ, (∀ (x y : ℝ), (-6 ≤ x ∧ x ≤ -3) → (3 ≤ y ∧ y ≤ 6) → k ≤ (x + y) / x) ∧ k = 0 :=
by
  sorry

end smallest_value_fraction_l220_220551


namespace problem1_problem2_l220_220463

-- Problem (1)
theorem problem1 (α : ℝ) (h1 : Real.tan α = 2) : 
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 := 
by 
  sorry

-- Problem (2)
theorem problem2 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.tan α = -4 / 3 := 
by
  sorry

end problem1_problem2_l220_220463


namespace tens_digit_23_pow_1987_l220_220087

def tens_digit_of_power (a b n : ℕ) : ℕ :=
  ((a^b % n) / 10) % 10

theorem tens_digit_23_pow_1987 : tens_digit_of_power 23 1987 100 = 4 := by
  sorry

end tens_digit_23_pow_1987_l220_220087


namespace tangent_line_eq_l220_220577

theorem tangent_line_eq {f : ℝ → ℝ} (hf : ∀ x, f x = x - 2 * Real.log x) :
  ∃ m b, (m = -1) ∧ (b = 2) ∧ (∀ x, f x = m * x + b) :=
by
  sorry

end tangent_line_eq_l220_220577


namespace molecular_weight_compound_l220_220579

/-- Definition of atomic weights for elements H, Cr, and O in AMU (Atomic Mass Units) --/
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cr : ℝ := 51.996
def atomic_weight_O : ℝ := 15.999

/-- Proof statement to calculate the molecular weight of a compound with 2 H, 1 Cr, and 4 O --/
theorem molecular_weight_compound :
  2 * atomic_weight_H + 1 * atomic_weight_Cr + 4 * atomic_weight_O = 118.008 :=
by
  sorry

end molecular_weight_compound_l220_220579


namespace number_of_members_l220_220189

noncomputable def club_members (n O N : ℕ) : Prop :=
  (3 * n = O - N) ∧ (O - N = 15)

theorem number_of_members (n O N : ℕ) (h : club_members n O N) : n = 5 :=
  by
    sorry

end number_of_members_l220_220189


namespace find_m_l220_220159

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem find_m (a b m : ℝ) (h1 : f m a b = 0) (h2 : 3 * m^2 + 2 * a * m + b = 0)
  (h3 : f (m / 3) a b = 1 / 2) (h4 : m ≠ 0) : m = 3 / 2 :=
  sorry

end find_m_l220_220159


namespace number_of_books_l220_220009

-- Define the given conditions as variables
def movies_in_series : Nat := 62
def books_read : Nat := 4
def books_yet_to_read : Nat := 15

-- State the proposition we need to prove
theorem number_of_books : (books_read + books_yet_to_read) = 19 :=
by
  sorry

end number_of_books_l220_220009


namespace verify_parabola_D_l220_220486

def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

def parabola_vertex (y : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, y x = vertex_form (-1) h k x

-- Given conditions
def h : ℝ := 2
def k : ℝ := 3

-- Possible expressions
def parabola_A (x : ℝ) : ℝ := -((x + 2)^2) - 3
def parabola_B (x : ℝ) : ℝ := -((x - 2)^2) - 3
def parabola_C (x : ℝ) : ℝ := -((x + 2)^2) + 3
def parabola_D (x : ℝ) : ℝ := -((x - 2)^2) + 3

theorem verify_parabola_D : parabola_vertex parabola_D 2 3 :=
by
  -- Placeholder for the proof
  sorry

end verify_parabola_D_l220_220486


namespace min_value_expr_l220_220487

open Real

theorem min_value_expr (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) :
  3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ ≥ (3 : ℝ) * (12 * sqrt 2)^((1 : ℝ) / (3 : ℝ)) := sorry

end min_value_expr_l220_220487


namespace num_integers_between_cubed_values_l220_220941

theorem num_integers_between_cubed_values : 
  let a : ℝ := 10.5
  let b : ℝ := 10.7
  let c1 := a^3
  let c2 := b^3
  let first_integer := Int.ceil c1
  let last_integer := Int.floor c2
  first_integer ≤ last_integer → 
  last_integer - first_integer + 1 = 67 :=
by
  sorry

end num_integers_between_cubed_values_l220_220941


namespace simple_interest_rate_l220_220733

theorem simple_interest_rate
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (h1 : SI = 400)
  (h2 : P = 800)
  (h3 : T = 2) :
  R = 25 :=
by
  sorry

end simple_interest_rate_l220_220733


namespace loss_percentage_on_book_sold_at_loss_l220_220519

theorem loss_percentage_on_book_sold_at_loss :
  ∀ (total_cost cost1 : ℝ) (gain_percent : ℝ),
    total_cost = 420 → cost1 = 245 → gain_percent = 0.19 →
    (∀ (cost2 SP : ℝ), cost2 = total_cost - cost1 →
                       SP = cost2 * (1 + gain_percent) →
                       SP = 208.25 →
                       ((cost1 - SP) / cost1 * 100) = 15) :=
by
  intros total_cost cost1 gain_percent h_total_cost h_cost1 h_gain_percent cost2 SP h_cost2 h_SP h_SP_value
  sorry

end loss_percentage_on_book_sold_at_loss_l220_220519


namespace jason_books_l220_220598

theorem jason_books (books_per_shelf : ℕ) (num_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 45 → num_shelves = 7 → total_books = books_per_shelf * num_shelves → total_books = 315 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jason_books_l220_220598


namespace solve_eq_sqrt_exp_l220_220559

theorem solve_eq_sqrt_exp :
  (∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) → (x = 2 ∨ x = -1)) :=
by
  -- Prove that the solutions are x = 2 and x = -1
  sorry

end solve_eq_sqrt_exp_l220_220559


namespace correct_option_l220_220288

-- Definitions representing the conditions
variable (a b c : Line) -- Define the lines a, b, and c

-- Conditions for the problem
def is_parallel (x y : Line) : Prop := -- Define parallel property
  sorry

def is_perpendicular (x y : Line) : Prop := -- Define perpendicular property
  sorry

noncomputable def proof_statement : Prop :=
  is_parallel a b → is_perpendicular a c → is_perpendicular b c

-- Lean statement of the proof problem
theorem correct_option (h1 : is_parallel a b) (h2 : is_perpendicular a c) : is_perpendicular b c :=
  sorry

end correct_option_l220_220288


namespace relationship_in_size_l220_220587

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 2.1
noncomputable def c : ℝ := Real.log (1.5) / Real.log (2)

theorem relationship_in_size : b > a ∧ a > c := by
  sorry

end relationship_in_size_l220_220587


namespace sum_S6_l220_220591

variable (a_n : ℕ → ℚ)
variable (d : ℚ)
variable (S : ℕ → ℚ)
variable (a1 : ℚ)

/-- Define arithmetic sequence with common difference -/
def arithmetic_seq (n : ℕ) := a1 + n * d

/-- Define the sum of the first n terms of the sequence -/
def sum_of_arith_seq (n : ℕ) := n * a1 + (n * (n - 1) / 2) * d

/-- The given conditions -/
axiom h1 : d = 5
axiom h2 : (a_n 1 = a1) ∧ (a_n 2 = a1 + d) ∧ (a_n 5 = a1 + 4 * d)
axiom geom_seq : (a1 + d)^2 = a1 * (a1 + 4 * d)

theorem sum_S6 : S 6 = 90 := by
  sorry

end sum_S6_l220_220591


namespace distance_travelled_l220_220018

theorem distance_travelled (speed time distance : ℕ) 
  (h1 : speed = 25)
  (h2 : time = 5)
  (h3 : distance = speed * time) : 
  distance = 125 :=
by
  sorry

end distance_travelled_l220_220018


namespace last_four_digits_of_power_of_5_2017_l220_220032

theorem last_four_digits_of_power_of_5_2017 :
  (5 ^ 2017 % 10000) = 3125 :=
by
  sorry

end last_four_digits_of_power_of_5_2017_l220_220032


namespace range_of_a_l220_220647

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, -1 ≤ x → f a x ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by sorry

end range_of_a_l220_220647


namespace sector_area_l220_220494

theorem sector_area (r : ℝ) (θ : ℝ) (h_r : r = 12) (h_θ : θ = 40) : (θ / 360) * π * r^2 = 16 * π :=
by
  rw [h_r, h_θ]
  sorry

end sector_area_l220_220494


namespace compute_y_geometric_series_l220_220354

theorem compute_y_geometric_series :
  let S1 := (∑' n : ℕ, (1 / 3)^n)
  let S2 := (∑' n : ℕ, (-1)^n * (1 / 3)^n)
  (S1 * S2 = ∑' n : ℕ, (1 / 9)^n) → 
  S1 = 3 / 2 →
  S2 = 3 / 4 →
  (∑' n : ℕ, (1 / y)^n) = 9 / 8 →
  y = 9 := 
by
  intros S1 S2 h₁ h₂ h₃ h₄
  sorry

end compute_y_geometric_series_l220_220354


namespace minimum_distance_midpoint_l220_220025

theorem minimum_distance_midpoint 
    (θ : ℝ)
    (P : ℝ × ℝ := (-4, 4))
    (C1_standard : ∀ (x y : ℝ), (x + 4)^2 + (y - 3)^2 = 1)
    (C2_standard : ∀ (x y : ℝ), x^2 / 64 + y^2 / 9 = 1)
    (Q : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ))
    (M : ℝ × ℝ := (-2 + 4 * Real.cos θ, 2 + 3 / 2 * Real.sin θ))
    (C3_standard : ∀ (x y : ℝ), x - 2*y - 7 = 0) :
    ∃ (θ : ℝ), θ = Real.arcsin (-3/5) ∧ (θ = Real.arccos 4/5) ∧
    (∀ (d : ℝ), d = abs (5 * Real.sin (Real.arctan (4 / 3) - θ) - 13) / Real.sqrt 5 ∧ 
    d = 8 * Real.sqrt 5 / 5) :=
sorry

end minimum_distance_midpoint_l220_220025


namespace sphere_radius_eq_cylinder_radius_l220_220308

theorem sphere_radius_eq_cylinder_radius
  (r h d : ℝ) (h_eq_d : h = 16) (d_eq_h : d = 16)
  (sphere_surface_area_eq_cylinder : 4 * Real.pi * r^2 = 2 * Real.pi * (d / 2) * h) : 
  r = 8 :=
by
  sorry

end sphere_radius_eq_cylinder_radius_l220_220308


namespace fish_buckets_last_l220_220777

theorem fish_buckets_last (buckets_sharks : ℕ) (buckets_total : ℕ) 
  (h1 : buckets_sharks = 4)
  (h2 : ∀ (buckets_dolphins : ℕ), buckets_dolphins = buckets_sharks / 2)
  (h3 : ∀ (buckets_other : ℕ), buckets_other = 5 * buckets_sharks)
  (h4 : buckets_total = 546)
  : 546 / ((buckets_sharks + (buckets_sharks / 2) + (5 * buckets_sharks)) * 7) = 3 :=
by
  -- Calculation steps skipped for brevity
  sorry

end fish_buckets_last_l220_220777


namespace sum_third_column_l220_220265

variable (a b c d e f g h i : ℕ)

theorem sum_third_column :
  (a + b + c = 24) →
  (d + e + f = 26) →
  (g + h + i = 40) →
  (a + d + g = 27) →
  (b + e + h = 20) →
  (c + f + i = 43) :=
by
  intros
  sorry

end sum_third_column_l220_220265


namespace bill_head_circumference_l220_220407

theorem bill_head_circumference (jack_head_circumference charlie_head_circumference bill_head_circumference : ℝ) :
  jack_head_circumference = 12 →
  charlie_head_circumference = (1 / 2 * jack_head_circumference) + 9 →
  bill_head_circumference = (2 / 3 * charlie_head_circumference) →
  bill_head_circumference = 10 :=
by
  intro hj hc hb
  sorry

end bill_head_circumference_l220_220407


namespace striped_to_total_ratio_l220_220497

theorem striped_to_total_ratio (total_students shorts_checkered_diff striped_shorts_diff : ℕ)
    (h_total : total_students = 81)
    (h_shorts_checkered : ∃ checkered, shorts_checkered_diff = checkered + 19)
    (h_striped_shorts : ∃ shorts, striped_shorts_diff = shorts + 8) :
    (striped_shorts_diff : ℚ) / total_students = 2 / 3 :=
by sorry

end striped_to_total_ratio_l220_220497


namespace smallest_number_of_marbles_l220_220015

theorem smallest_number_of_marbles (M : ℕ) (h1 : M ≡ 2 [MOD 5]) (h2 : M ≡ 2 [MOD 6]) (h3 : M ≡ 2 [MOD 7]) (h4 : 1 < M) : M = 212 :=
by sorry

end smallest_number_of_marbles_l220_220015


namespace response_rate_percentage_50_l220_220638

def questionnaire_response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℕ :=
  (responses_needed * 100) / questionnaires_mailed

theorem response_rate_percentage_50 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 300) 
  (h2 : questionnaires_mailed = 600) : 
  questionnaire_response_rate_percentage responses_needed questionnaires_mailed = 50 :=
by 
  rw [h1, h2]
  norm_num
  sorry

end response_rate_percentage_50_l220_220638


namespace car_speed_proof_l220_220148

noncomputable def car_speed_in_kmh (rpm : ℕ) (circumference : ℕ) : ℕ :=
  (rpm * circumference * 60) / 1000

theorem car_speed_proof : 
  car_speed_in_kmh 400 1 = 24 := 
by
  sorry

end car_speed_proof_l220_220148


namespace smallest_c_d_sum_l220_220816

theorem smallest_c_d_sum : ∃ (c d : ℕ), 2^12 * 7^6 = c^d ∧  (∀ (c' d' : ℕ), 2^12 * 7^6 = c'^d'  → (c + d) ≤ (c' + d')) ∧ c + d = 21954 := by
  sorry

end smallest_c_d_sum_l220_220816


namespace football_game_spectators_l220_220078

theorem football_game_spectators (total_wristbands wristbands_per_person : ℕ) (h1 : total_wristbands = 234) (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 117 := by
  sorry

end football_game_spectators_l220_220078


namespace quadrilateral_is_parallelogram_l220_220897

theorem quadrilateral_is_parallelogram
  (A B C D : Type)
  (angle_DAB angle_ABC angle_BAD angle_DCB : ℝ)
  (h1 : angle_DAB = 135)
  (h2 : angle_ABC = 45)
  (h3 : angle_BAD = 45)
  (h4 : angle_DCB = 45) :
  (A B C D : Type) → Prop :=
by
  -- Definitions and conditions are given.
  sorry

end quadrilateral_is_parallelogram_l220_220897


namespace circle_center_l220_220104

theorem circle_center (x y : ℝ) :
  4 * x^2 - 16 * x + 4 * y^2 + 8 * y - 12 = 0 →
  (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = 8 ∧ h = 2 ∧ k = -1) :=
sorry

end circle_center_l220_220104


namespace weekly_allowance_l220_220506

theorem weekly_allowance (A : ℝ) (H1 : A - (3/5) * A = (2/5) * A)
(H2 : (2/5) * A - (1/3) * ((2/5) * A) = (4/15) * A)
(H3 : (4/15) * A = 0.96) : A = 3.6 := 
sorry

end weekly_allowance_l220_220506


namespace find_f_1991_l220_220786

namespace FunctionProof

-- Defining the given conditions as statements in Lean
def func_f (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f (f n)) = -f (f (m + 1)) - n

def poly_g (f g : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, g n = g (f n)

-- Statement of the problem
theorem find_f_1991 
  (f g : ℤ → ℤ)
  (Hf : func_f f)
  (Hg : poly_g f g) :
  f 1991 = -1992 := 
sorry

end FunctionProof

end find_f_1991_l220_220786


namespace range_of_m_l220_220659
-- Import the entire math library

-- Defining the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0 
def q (x m : ℝ) : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0 

-- Main theorem statement
theorem range_of_m (m : ℝ) (h1 : 0 < m) 
(hsuff : ∀ x : ℝ, p x → q x m) 
(hnsuff : ¬ (∀ x : ℝ, q x m → p x)) : m ≥ 9 := 
sorry

end range_of_m_l220_220659


namespace football_goals_l220_220871

theorem football_goals :
  (exists A B C : ℕ,
    (A = 3 ∧ B ≠ 1 ∧ (C = 5 ∧ V = 6 ∧ A ≠ 2 ∧ V = 5)) ∨
    (A ≠ 3 ∧ B = 1 ∧ (C ≠ 5 ∧ V = 6 ∧ A = 2 ∧ V ≠ 5))) →
  A + B + C ≠ 10 :=
by {
  sorry
}

end football_goals_l220_220871


namespace vacation_days_in_march_l220_220721

theorem vacation_days_in_march 
  (days_worked : ℕ) 
  (days_worked_to_vacation_days : ℕ) 
  (vacation_days_left : ℕ) 
  (days_in_march : ℕ) 
  (days_in_september : ℕ)
  (h1 : days_worked = 300)
  (h2 : days_worked_to_vacation_days = 10)
  (h3 : vacation_days_left = 15)
  (h4 : days_in_september = 2 * days_in_march)
  (h5 : days_worked / days_worked_to_vacation_days - (days_in_march + days_in_september) = vacation_days_left) 
  : days_in_march = 5 := 
by
  sorry

end vacation_days_in_march_l220_220721


namespace problem_statement_l220_220813

def is_ideal_circle (circle : ℝ × ℝ → ℝ) (l : ℝ × ℝ → ℝ) : Prop :=
  ∃ P Q : ℝ × ℝ, (circle P = 0 ∧ circle Q = 0) ∧ (abs (l P) = 1 ∧ abs (l Q) = 1)

noncomputable def line_l (p : ℝ × ℝ) : ℝ := 3 * p.1 + 4 * p.2 - 12

noncomputable def circle_D (p : ℝ × ℝ) : ℝ := (p.1 - 4) ^ 2 + (p.2 - 4) ^ 2 - 16

theorem problem_statement : is_ideal_circle circle_D line_l :=
sorry  -- The proof would go here

end problem_statement_l220_220813


namespace siamese_cats_initial_l220_220042

theorem siamese_cats_initial (S : ℕ) : S + 25 - 45 = 18 -> S = 38 :=
by
  intro h
  sorry

end siamese_cats_initial_l220_220042


namespace tim_campaign_total_l220_220467

theorem tim_campaign_total (amount_max : ℕ) (num_max : ℕ) (num_half : ℕ) (total_donations : ℕ) (total_raised : ℕ)
  (H1 : amount_max = 1200)
  (H2 : num_max = 500)
  (H3 : num_half = 3 * num_max)
  (H4 : total_donations = num_max * amount_max + num_half * (amount_max / 2))
  (H5 : total_donations = 40 * total_raised / 100) :
  total_raised = 3750000 :=
by
  -- Proof is omitted
  sorry

end tim_campaign_total_l220_220467


namespace parallel_lines_implies_m_opposite_sides_implies_m_range_l220_220073

-- Definitions of the given lines and points
def l1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (m : ℝ) : ℝ × ℝ := (m, 4)

-- Problem Part (I)
theorem parallel_lines_implies_m (m : ℝ) : 
  (∀ (x y : ℝ), l1 x y → false) ∧ (∀ (x2 y2 : ℝ), (x2, y2) = A m ∨ (x2, y2) = B m → false) →
  (∃ m, 2 * m + 3 = 0 ∧ m + 5 = 0) :=
sorry

-- Problem Part (II)
theorem opposite_sides_implies_m_range (m : ℝ) :
  ((2 * (-2) + m - 1) * (2 * m + 4 - 1) < 0) →
  m ∈ Set.Ioo (-3/2 : ℝ) (5 : ℝ) :=
sorry

end parallel_lines_implies_m_opposite_sides_implies_m_range_l220_220073


namespace find_constants_l220_220889

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, (8 * x + 1) / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) → 
  A = 33 / 4 ∧ B = -19 / 4 ∧ C = -17 / 2 :=
by 
  intro h
  sorry

end find_constants_l220_220889


namespace cost_of_jacket_is_60_l220_220356

/-- Define the constants from the problem --/
def cost_of_shirt : ℕ := 8
def cost_of_pants : ℕ := 18
def shirts_bought : ℕ := 4
def pants_bought : ℕ := 2
def jackets_bought : ℕ := 2
def carrie_paid : ℕ := 94

/-- Define the problem statement --/
theorem cost_of_jacket_is_60 (total_cost jackets_cost : ℕ) 
    (H1 : total_cost = (shirts_bought * cost_of_shirt) + (pants_bought * cost_of_pants) + jackets_cost)
    (H2 : carrie_paid = total_cost / 2)
    : jackets_cost / jackets_bought = 60 := 
sorry

end cost_of_jacket_is_60_l220_220356


namespace product_of_digits_of_non_divisible_number_l220_220165

theorem product_of_digits_of_non_divisible_number:
  (¬ (3641 % 4 = 0)) →
  ((3641 % 10) * ((3641 / 10) % 10)) = 4 :=
by
  intro h
  sorry

end product_of_digits_of_non_divisible_number_l220_220165


namespace abs_condition_iff_range_l220_220187

theorem abs_condition_iff_range (x : ℝ) : 
  (|x-1| + |x+2| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 2) := 
sorry

end abs_condition_iff_range_l220_220187


namespace inequality_and_equality_condition_l220_220072

variable {x y : ℝ}

theorem inequality_and_equality_condition
  (hx : 0 < x) (hy : 0 < y) :
  (x + y^2 / x ≥ 2 * y) ∧ (x + y^2 / x = 2 * y ↔ x = y) := sorry

end inequality_and_equality_condition_l220_220072


namespace value_of_a_m_minus_3n_l220_220785

theorem value_of_a_m_minus_3n (a : ℝ) (m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) : a^(m - 3 * n) = 1 :=
sorry

end value_of_a_m_minus_3n_l220_220785


namespace intersection_is_N_l220_220682

-- Define the sets M and N as given in the problem
def M := {x : ℝ | x > 0}
def N := {x : ℝ | Real.log x > 0}

-- State the theorem for the intersection of M and N
theorem intersection_is_N : (M ∩ N) = N := 
  by 
    sorry

end intersection_is_N_l220_220682


namespace fraction_simplification_l220_220340

noncomputable def x : ℚ := 0.714714714 -- Repeating decimal representation for x
noncomputable def y : ℚ := 2.857857857 -- Repeating decimal representation for y

theorem fraction_simplification :
  (x / y) = (714 / 2855) :=
by
  sorry

end fraction_simplification_l220_220340


namespace PetrovFamilySavings_l220_220227

def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

def total_income : ℕ := parents_salary + grandmothers_pension + sons_scholarship
def total_expenses : ℕ := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses

def surplus : ℕ := total_income - total_expenses
def deposit : ℕ := surplus / 10

def amount_set_aside : ℕ := surplus - deposit

theorem PetrovFamilySavings : amount_set_aside = 16740 := by
  sorry

end PetrovFamilySavings_l220_220227


namespace coefficient_of_x8y2_l220_220344

theorem coefficient_of_x8y2 :
  let term1 := (1 / x^2)
  let term2 := (3 / y)
  let expansion := (x^2 - y)^7
  let coeff1 := 21 * (x ^ 10) * (y ^ 2) * (-1)
  let coeff2 := 35 * (3 / y) * (x ^ 8) * (y ^ 3)
  let comb := coeff1 + coeff2
  comb = -84 * x ^ 8 * y ^ 2 := by
  sorry

end coefficient_of_x8y2_l220_220344


namespace total_tiles_l220_220901

theorem total_tiles (s : ℕ) (h1 : true) (h2 : true) (h3 : true) (h4 : true) (h5 : 4 * s - 4 = 100): s * s = 676 :=
by
  sorry

end total_tiles_l220_220901


namespace compound_interest_two_years_l220_220283

/-- Given the initial amount, and year-wise interest rates, 
     we want to find the amount in 2 years and prove it equals to a specific value. -/
theorem compound_interest_two_years 
  (P : ℝ) (R1 : ℝ) (R2 : ℝ) (T1 : ℝ) (T2 : ℝ) 
  (initial_amount : P = 7644) 
  (interest_rate_first_year : R1 = 0.04) 
  (interest_rate_second_year : R2 = 0.05) 
  (time_first_year : T1 = 1) 
  (time_second_year : T2 = 1) : 
  (P + (P * R1 * T1) + ((P + (P * R1 * T1)) * R2 * T2) = 8347.248) := 
by 
  sorry

end compound_interest_two_years_l220_220283


namespace T_0_2006_correct_T_1_2006_correct_T_2_2006_correct_l220_220241

def T (r n : ℕ) : ℕ :=
  sorry -- Define the function T_r(n) according to the problem's condition

-- Specific cases given in the problem statement
noncomputable def T_0_2006 : ℕ := T 0 2006
noncomputable def T_1_2006 : ℕ := T 1 2006
noncomputable def T_2_2006 : ℕ := T 2 2006

-- Theorems stating the result
theorem T_0_2006_correct : T_0_2006 = 1764 := sorry
theorem T_1_2006_correct : T_1_2006 = 122 := sorry
theorem T_2_2006_correct : T_2_2006 = 121 := sorry

end T_0_2006_correct_T_1_2006_correct_T_2_2006_correct_l220_220241


namespace labor_arrangement_count_l220_220942

theorem labor_arrangement_count (volunteers : ℕ) (choose_one_day : ℕ) (days : ℕ) 
    (h_volunteers : volunteers = 7) 
    (h_choose_one_day : choose_one_day = 3) 
    (h_days : days = 2) : 
    (Nat.choose volunteers choose_one_day) * (Nat.choose (volunteers - choose_one_day) choose_one_day) = 140 := 
by
  sorry

end labor_arrangement_count_l220_220942


namespace rent_increase_percentage_l220_220302

theorem rent_increase_percentage (a x: ℝ) (h1: a ≠ 0) (h2: (9 / 10) * a = (4 / 5) * a * (1 + x / 100)) : x = 12.5 :=
sorry

end rent_increase_percentage_l220_220302


namespace num_adults_on_field_trip_l220_220351

-- Definitions of the conditions
def num_vans : Nat := 6
def people_per_van : Nat := 9
def num_students : Nat := 40

-- The theorem to prove
theorem num_adults_on_field_trip : (num_vans * people_per_van) - num_students = 14 := by
  sorry

end num_adults_on_field_trip_l220_220351


namespace sum_divisible_by_100_l220_220198

theorem sum_divisible_by_100 (S : Finset ℤ) (hS : S.card = 200) : 
  ∃ T : Finset ℤ, T ⊆ S ∧ T.card = 100 ∧ (T.sum id) % 100 = 0 := 
  sorry

end sum_divisible_by_100_l220_220198


namespace problem_S_equal_102_l220_220321

-- Define the values in Lean
def S : ℕ := 1 * 3^1 + 2 * 3^2 + 3 * 3^3

-- Theorem to prove that S is equal to 102
theorem problem_S_equal_102 : S = 102 :=
by
  sorry

end problem_S_equal_102_l220_220321


namespace grid_labelings_count_l220_220514

theorem grid_labelings_count :
  ∃ (labeling_count : ℕ), 
    labeling_count = 2448 ∧ 
    (∀ (grid : Matrix (Fin 3) (Fin 3) ℕ),
      grid 0 0 = 1 ∧ 
      grid 2 2 = 2009 ∧ 
      (∀ (i j : Fin 3), j < 2 → grid i j ∣ grid i (j + 1)) ∧ 
      (∀ (i j : Fin 3), i < 2 → grid i j ∣ grid (i + 1) j)) :=
sorry

end grid_labelings_count_l220_220514


namespace find_lines_through_p_and_intersecting_circle_l220_220639

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25

noncomputable def passes_through (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = l P.1

noncomputable def chord_length (c p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

theorem find_lines_through_p_and_intersecting_circle :
  ∃ l : ℝ → ℝ, (passes_through l (-2, 3)) ∧
  (∃ p1 p2 : ℝ × ℝ, trajectory_equation p1.1 p1.2 ∧ trajectory_equation p2.1 p2.2 ∧
  chord_length (1, 2) p1 p2 = 8^2) :=
by
  sorry

end find_lines_through_p_and_intersecting_circle_l220_220639


namespace angle_of_inclination_of_line_l220_220446

theorem angle_of_inclination_of_line (θ : ℝ) (m : ℝ) (h : |m| = 1) :
  θ = 45 ∨ θ = 135 :=
sorry

end angle_of_inclination_of_line_l220_220446


namespace subtraction_of_negatives_l220_220574

theorem subtraction_of_negatives : (-1) - (-4) = 3 :=
by
  -- Proof goes here.
  sorry

end subtraction_of_negatives_l220_220574


namespace set_intersection_l220_220450

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {1, 2, 5}
noncomputable def B : Set ℕ := {x ∈ U | (3 / (2 - x) + 1 ≤ 0)}
noncomputable def C_U_B : Set ℕ := U \ B

theorem set_intersection : A ∩ C_U_B = {1, 2} :=
by {
  sorry
}

end set_intersection_l220_220450


namespace age_ratio_4_years_hence_4_years_ago_l220_220096

-- Definitions based on the conditions
def current_age_ratio (A B : ℕ) := 5 * B = 3 * A
def age_ratio_4_years_ago_4_years_hence (A B : ℕ) := A - 4 = B + 4

-- The main theorem to prove
theorem age_ratio_4_years_hence_4_years_ago (A B : ℕ) 
  (h1 : current_age_ratio A B) 
  (h2 : age_ratio_4_years_ago_4_years_hence A B) : 
  A + 4 = 3 * (B - 4) := 
sorry

end age_ratio_4_years_hence_4_years_ago_l220_220096


namespace oaks_not_adjacent_probability_l220_220853

theorem oaks_not_adjacent_probability :
  let total_trees := 13
  let oaks := 5
  let other_trees := total_trees - oaks
  let possible_slots := other_trees + 1
  let combinations := Nat.choose possible_slots oaks
  let total_arrangements := Nat.factorial total_trees / (Nat.factorial oaks * Nat.factorial (total_trees - oaks))
  let probability := combinations / total_arrangements
  probability = 1 / 220 :=
by
  sorry

end oaks_not_adjacent_probability_l220_220853


namespace required_C6H6_for_C6H5CH3_and_H2_l220_220453

-- Define the necessary molecular structures and stoichiometry
def C6H6 : Type := ℕ -- Benzene
def CH4 : Type := ℕ -- Methane
def C6H5CH3 : Type := ℕ -- Toluene
def H2 : Type := ℕ -- Hydrogen

-- Balanced equation condition
def balanced_reaction (x : C6H6) (y : CH4) (z : C6H5CH3) (w : H2) : Prop :=
  x = y ∧ x = z ∧ x = w

-- Given conditions
def condition (m : ℕ) : Prop :=
  balanced_reaction m m m m

theorem required_C6H6_for_C6H5CH3_and_H2 :
  ∀ (n : ℕ), condition n → n = 3 → n = 3 :=
by
  intros n h hn
  exact hn

end required_C6H6_for_C6H5CH3_and_H2_l220_220453


namespace area_of_rectangle_l220_220485

theorem area_of_rectangle (P : ℝ) (w : ℝ) (h : ℝ) (A : ℝ) 
  (hP : P = 28) 
  (hw : w = 6) 
  (hP_formula : P = 2 * (h + w)) 
  (hA_formula : A = h * w) : 
  A = 48 :=
by
  sorry

end area_of_rectangle_l220_220485


namespace cyclists_meeting_l220_220442

-- Define the velocities of the cyclists and the time variable
variables (v₁ v₂ t : ℝ)

-- Define the conditions for the problem
def condition1 : Prop := v₁ * t = v₂ * (2/3)
def condition2 : Prop := v₂ * t = v₁ * 1.5

-- Define the main theorem to be proven
theorem cyclists_meeting (h1 : condition1 v₁ v₂ t) (h2 : condition2 v₁ v₂ t) :
  t = 1 ∧ (v₁ / v₂ = 3 / 2) :=
by sorry

end cyclists_meeting_l220_220442


namespace average_age_of_women_l220_220080

theorem average_age_of_women (A : ℝ) (W1 W2 : ℝ)
  (cond1 : 10 * (A + 6) - 10 * A = 60)
  (cond2 : W1 + W2 = 60 + 40) :
  (W1 + W2) / 2 = 50 := 
by
  sorry

end average_age_of_women_l220_220080


namespace double_chess_first_player_can_draw_l220_220133

-- Define the basic structure and rules of double chess
structure Game :=
  (state : Type)
  (move : state → state)
  (turn : ℕ → state → state)

-- Define the concept of double move
def double_move (g : Game) (s : g.state) : g.state :=
  g.move (g.move s)

-- Define a condition stating that the first player can at least force a draw
theorem double_chess_first_player_can_draw
  (game : Game)
  (initial_state : game.state)
  (double_move_valid : ∀ s : game.state, ∃ s' : game.state, s' = double_move game s) :
  ∃ draw : game.state, ∀ second_player_strategy : game.state → game.state, 
    double_move game initial_state = draw :=
  sorry

end double_chess_first_player_can_draw_l220_220133


namespace figure_perimeter_l220_220380

theorem figure_perimeter 
  (side_length : ℕ)
  (inner_large_square_sides : ℕ)
  (shared_edge_length : ℕ)
  (rectangle_dimension_1 : ℕ)
  (rectangle_dimension_2 : ℕ) 
  (h1 : side_length = 2)
  (h2 : inner_large_square_sides = 4)
  (h3 : shared_edge_length = 2)
  (h4 : rectangle_dimension_1 = 2)
  (h5 : rectangle_dimension_2 = 1) : 
  let large_square_perimeter := inner_large_square_sides * side_length
  let horizontal_perimeter := large_square_perimeter - shared_edge_length + rectangle_dimension_1 + rectangle_dimension_2
  let vertical_perimeter := large_square_perimeter
  horizontal_perimeter + vertical_perimeter = 33 := 
by
  sorry

end figure_perimeter_l220_220380


namespace completion_time_workshop_3_l220_220416

-- Define the times for workshops
def time_in_workshop_3 : ℝ := 8
def time_in_workshop_1 : ℝ := time_in_workshop_3 + 10
def time_in_workshop_2 : ℝ := (time_in_workshop_3 + 10) - 3.6

-- Define the combined work equation
def combined_work_eq := (1 / time_in_workshop_1) + (1 / time_in_workshop_2) = (1 / time_in_workshop_3)

-- Final theorem statement
theorem completion_time_workshop_3 (h : combined_work_eq) : time_in_workshop_3 - 7 = 1 :=
by
  sorry

end completion_time_workshop_3_l220_220416


namespace centers_distance_ABC_l220_220952

-- Define triangle ABC with the given properties
structure RightTriangle (ABC : Type) :=
(angle_A : ℝ)
(angle_C : ℝ)
(shorter_leg : ℝ)

-- Given: angle A is 30 degrees, angle C is 90 degrees, and shorter leg AC is 1
def triangle_ABC : RightTriangle ℝ := {
  angle_A := 30,
  angle_C := 90,
  shorter_leg := 1
}

-- Define the distance between the centers of the inscribed circles of triangles ACD and BCD
noncomputable def distance_between_centers (ABC : RightTriangle ℝ): ℝ :=
  sorry  -- placeholder for the actual proof

-- Example problem statement
theorem centers_distance_ABC (ABC : RightTriangle ℝ) (h_ABC : ABC = triangle_ABC) :
  distance_between_centers ABC = (Real.sqrt 3 - 1) / Real.sqrt 2 :=
sorry

end centers_distance_ABC_l220_220952


namespace profit_percentage_is_4_l220_220592

-- Define the cost price and selling price
def cost_price : Nat := 600
def selling_price : Nat := 624

-- Calculate profit in dollars
def profit_dollars : Nat := selling_price - cost_price

-- Calculate profit percentage
def profit_percentage : Nat := (profit_dollars * 100) / cost_price

-- Prove that the profit percentage is 4%
theorem profit_percentage_is_4 : profit_percentage = 4 := by
  sorry

end profit_percentage_is_4_l220_220592


namespace daughters_meet_days_count_l220_220099

noncomputable def days_elder_returns := 5
noncomputable def days_second_returns := 4
noncomputable def days_youngest_returns := 3

noncomputable def total_days := 100

-- Defining the count of individual and combined visits
noncomputable def count_individual_visits (period : ℕ) : ℕ := total_days / period
noncomputable def count_combined_visits (period1 : ℕ) (period2 : ℕ) : ℕ := total_days / Nat.lcm period1 period2
noncomputable def count_all_together_visits (periods : List ℕ) : ℕ := total_days / periods.foldr Nat.lcm 1

-- Specific counts
noncomputable def count_youngest_visits : ℕ := count_individual_visits days_youngest_returns
noncomputable def count_second_visits : ℕ := count_individual_visits days_second_returns
noncomputable def count_elder_visits : ℕ := count_individual_visits days_elder_returns

noncomputable def count_youngest_and_second : ℕ := count_combined_visits days_youngest_returns days_second_returns
noncomputable def count_youngest_and_elder : ℕ := count_combined_visits days_youngest_returns days_elder_returns
noncomputable def count_second_and_elder : ℕ := count_combined_visits days_second_returns days_elder_returns

noncomputable def count_all_three : ℕ := count_all_together_visits [days_youngest_returns, days_second_returns, days_elder_returns]

-- Final Inclusion-Exclusion principle application
noncomputable def days_at_least_one_returns : ℕ := 
  count_youngest_visits + count_second_visits + count_elder_visits
  - count_youngest_and_second
  - count_youngest_and_elder
  - count_second_and_elder
  + count_all_three

theorem daughters_meet_days_count : days_at_least_one_returns = 60 := by
  sorry

end daughters_meet_days_count_l220_220099


namespace arithmetic_progression_x_value_l220_220247

theorem arithmetic_progression_x_value :
  ∀ (x : ℝ), (3 * x + 2) - (2 * x - 4) = (5 * x - 1) - (3 * x + 2) → x = 9 :=
by
  intros x h
  sorry

end arithmetic_progression_x_value_l220_220247


namespace jelly_beans_problem_l220_220136

/-- Mrs. Wonderful's jelly beans problem -/
theorem jelly_beans_problem : ∃ n_girls n_boys : ℕ, 
  (n_boys = n_girls + 2) ∧
  ((n_girls ^ 2) + ((n_girls + 2) ^ 2) = 394) ∧
  (n_girls + n_boys = 28) :=
by
  sorry

end jelly_beans_problem_l220_220136


namespace joseph_vs_kyle_emily_vs_joseph_emily_vs_kyle_l220_220925

noncomputable def distance_joseph : ℝ := 48 * 2.5 + 60 * 1.5
noncomputable def distance_kyle : ℝ := 70 * 2 + 63 * 2.5
noncomputable def distance_emily : ℝ := 65 * 3

theorem joseph_vs_kyle : distance_joseph - distance_kyle = -87.5 := by
  unfold distance_joseph
  unfold distance_kyle
  sorry

theorem emily_vs_joseph : distance_emily - distance_joseph = -15 := by
  unfold distance_emily
  unfold distance_joseph
  sorry

theorem emily_vs_kyle : distance_emily - distance_kyle = -102.5 := by
  unfold distance_emily
  unfold distance_kyle
  sorry

end joseph_vs_kyle_emily_vs_joseph_emily_vs_kyle_l220_220925


namespace find_pqr_abs_l220_220012

variables {p q r : ℝ}

-- Conditions as hypotheses
def conditions (p q r : ℝ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
  (p^2 + 2/q = q^2 + 2/r) ∧ (q^2 + 2/r = r^2 + 2/p)

-- Statement of the theorem
theorem find_pqr_abs (h : conditions p q r) : |p * q * r| = 2 :=
sorry

end find_pqr_abs_l220_220012


namespace tangerine_boxes_l220_220447

theorem tangerine_boxes
  (num_boxes_apples : ℕ)
  (apples_per_box : ℕ)
  (num_boxes_tangerines : ℕ)
  (tangerines_per_box : ℕ)
  (total_fruits : ℕ)
  (h1 : num_boxes_apples = 19)
  (h2 : apples_per_box = 46)
  (h3 : tangerines_per_box = 170)
  (h4 : total_fruits = 1894)
  : num_boxes_tangerines = 6 := 
  sorry

end tangerine_boxes_l220_220447


namespace average_speed_ratio_l220_220350

theorem average_speed_ratio (t_E t_F : ℝ) (d_B d_C : ℝ) (htE : t_E = 3) (htF : t_F = 4) (hdB : d_B = 450) (hdC : d_C = 300) :
  (d_B / t_E) / (d_C / t_F) = 2 :=
by
  sorry

end average_speed_ratio_l220_220350


namespace slices_left_for_lunch_tomorrow_l220_220762

-- Definitions according to conditions
def initial_slices : ℕ := 12
def slices_eaten_for_lunch := initial_slices / 2
def remaining_slices_after_lunch := initial_slices - slices_eaten_for_lunch
def slices_eaten_for_dinner := 1 / 3 * remaining_slices_after_lunch
def remaining_slices_after_dinner := remaining_slices_after_lunch - slices_eaten_for_dinner
def slices_shared_with_friend := 1 / 4 * remaining_slices_after_dinner
def remaining_slices_after_sharing := remaining_slices_after_dinner - slices_shared_with_friend
def slices_eaten_by_sibling := if (1 / 5 * remaining_slices_after_sharing < 1) then 0 else 1 / 5 * remaining_slices_after_sharing
def remaining_slices_after_sibling := remaining_slices_after_sharing - slices_eaten_by_sibling

-- Lean statement of the proof problem
theorem slices_left_for_lunch_tomorrow : remaining_slices_after_sibling = 3 := by
  sorry

end slices_left_for_lunch_tomorrow_l220_220762


namespace bob_salary_is_14400_l220_220427

variables (mario_salary_current : ℝ) (mario_salary_last_year : ℝ) (bob_salary_last_year : ℝ) (bob_salary_current : ℝ)

-- Given Conditions
axiom mario_salary_increase : mario_salary_current = 4000
axiom mario_salary_equation : 1.40 * mario_salary_last_year = mario_salary_current
axiom bob_salary_last_year_equation : bob_salary_last_year = 3 * mario_salary_current
axiom bob_salary_increase : bob_salary_current = bob_salary_last_year + 0.20 * bob_salary_last_year

-- Theorem to prove
theorem bob_salary_is_14400 
    (mario_salary_last_year_eq : mario_salary_last_year = 4000 / 1.40)
    (bob_salary_last_year_eq : bob_salary_last_year = 3 * 4000)
    (bob_salary_current_eq : bob_salary_current = 12000 + 0.20 * 12000) :
    bob_salary_current = 14400 := 
by
  sorry

end bob_salary_is_14400_l220_220427


namespace policeman_catches_thief_l220_220093

/-
  From a police station situated on a straight road infinite in both directions, a thief has stolen a police car.
  Its maximal speed equals 90% of the maximal speed of a police cruiser. When the theft is discovered some time
  later, a policeman starts to pursue the thief on a cruiser. However, the policeman does not know in which direction
  along the road the thief has gone, nor does he know how long ago the car has been stolen. The goal is to prove
  that it is possible for the policeman to catch the thief.
-/
theorem policeman_catches_thief (v : ℝ) (T₀ : ℝ) (o₀ : ℝ) :
  (0 < v) →
  (0 < T₀) →
  ∃ T p, T₀ ≤ T ∧ p ≤ v * T :=
sorry

end policeman_catches_thief_l220_220093


namespace passing_marks_required_l220_220388

theorem passing_marks_required (T : ℝ)
  (h1 : 0.30 * T + 60 = 0.40 * T)
  (h2 : 0.40 * T = passing_mark)
  (h3 : 0.50 * T - 40 = passing_mark) :
  passing_mark = 240 := by
  sorry

end passing_marks_required_l220_220388


namespace last_digit_11_power_11_last_digit_9_power_9_last_digit_9219_power_9219_last_digit_2014_power_2014_l220_220612

-- Definition of function to calculate the last digit of a number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Proof statements
theorem last_digit_11_power_11 : last_digit (11 ^ 11) = 1 := sorry

theorem last_digit_9_power_9 : last_digit (9 ^ 9) = 9 := sorry

theorem last_digit_9219_power_9219 : last_digit (9219 ^ 9219) = 9 := sorry

theorem last_digit_2014_power_2014 : last_digit (2014 ^ 2014) = 6 := sorry

end last_digit_11_power_11_last_digit_9_power_9_last_digit_9219_power_9219_last_digit_2014_power_2014_l220_220612


namespace find_x_angle_l220_220903

-- Define the conditions
def angles_around_point (a b c d : ℝ) : Prop :=
  a + b + c + d = 360

-- The given problem implies:
-- 120 + x + x + 2x = 360
-- We need to find x such that the above equation holds.
theorem find_x_angle :
  angles_around_point 120 x x (2 * x) → x = 60 :=
by
  sorry

end find_x_angle_l220_220903


namespace greatest_drop_in_price_l220_220724

def jan_change : ℝ := -0.75
def feb_change : ℝ := 1.50
def mar_change : ℝ := -3.00
def apr_change : ℝ := 2.50
def may_change : ℝ := -0.25
def jun_change : ℝ := 0.80
def jul_change : ℝ := -2.75
def aug_change : ℝ := -1.20

theorem greatest_drop_in_price : 
  mar_change = min (min (min (min (min (min jan_change jul_change) aug_change) may_change) feb_change) apr_change) jun_change :=
by
  -- This statement is where the proof would go.
  sorry

end greatest_drop_in_price_l220_220724


namespace convert_1623_to_base7_l220_220665

theorem convert_1623_to_base7 :
  ∃ a b c d : ℕ, 1623 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
  a = 4 ∧ b = 5 ∧ c = 0 ∧ d = 6 :=
by
  sorry

end convert_1623_to_base7_l220_220665


namespace green_ball_probability_l220_220967

-- Defining the number of red and green balls in each container
def containerI_red : ℕ := 10
def containerI_green : ℕ := 5

def containerII_red : ℕ := 3
def containerII_green : ℕ := 6

def containerIII_red : ℕ := 3
def containerIII_green : ℕ := 6

-- Probability of selecting any container
def prob_container : ℚ := 1 / 3

-- Defining the probabilities of drawing a green ball from each container
def prob_green_I : ℚ := containerI_green / (containerI_red + containerI_green)
def prob_green_II : ℚ := containerII_green / (containerII_red + containerII_green)
def prob_green_III : ℚ := containerIII_green / (containerIII_red + containerIII_green)

-- Law of total probability
def prob_green_total : ℚ :=
  prob_container * prob_green_I +
  prob_container * prob_green_II +
  prob_container * prob_green_III

-- The mathematical statement to be proven
theorem green_ball_probability :
  prob_green_total = 5 / 9 := by
  sorry

end green_ball_probability_l220_220967


namespace lollipop_ratio_l220_220885

/-- Sarah bought 12 lollipops for a total of 3 dollars. Julie gave Sarah 75 cents to pay for the shared lollipops.
Prove that the ratio of the number of lollipops shared to the total number of lollipops bought is 1:4. -/
theorem lollipop_ratio
  (h1 : 12 = lollipops_bought)
  (h2 : 3 = total_cost_dollars)
  (h3 : 75 = amount_paid_cents)
  : (75 / 25) / lollipops_bought = 1/4 :=
sorry

end lollipop_ratio_l220_220885


namespace regular_octagon_interior_angle_l220_220109

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l220_220109


namespace fraction_sum_equals_mixed_number_l220_220223

theorem fraction_sum_equals_mixed_number :
  (3 / 5 : ℚ) + (2 / 3) + (16 / 15) = (7 / 3) :=
by sorry

end fraction_sum_equals_mixed_number_l220_220223


namespace gretchen_total_earnings_l220_220509

-- Define the conditions
def price_per_drawing : ℝ := 20.0
def caricatures_sold_saturday : ℕ := 24
def caricatures_sold_sunday : ℕ := 16

-- The total caricatures sold
def total_caricatures_sold : ℕ := caricatures_sold_saturday + caricatures_sold_sunday

-- The total amount of money made
def total_money_made : ℝ := total_caricatures_sold * price_per_drawing

-- The theorem to be proven
theorem gretchen_total_earnings : total_money_made = 800.0 := by
  sorry

end gretchen_total_earnings_l220_220509


namespace perpendicular_vectors_x_value_l220_220236

theorem perpendicular_vectors_x_value 
  (x : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (x, -1)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : x = 2 :=
by
  sorry

end perpendicular_vectors_x_value_l220_220236


namespace painted_surface_area_is_33_l220_220003

/-- 
Problem conditions:
    1. We have 14 unit cubes each with side length 1 meter.
    2. The cubes are arranged in a rectangular formation with dimensions 3x3x1.
The question:
    Prove that the total painted surface area is 33 square meters.
-/
def total_painted_surface_area (cubes : ℕ) (dim_x dim_y dim_z : ℕ) : ℕ :=
  let top_area := dim_x * dim_y
  let side_area := 2 * (dim_x * dim_z + dim_y * dim_z + (dim_z - 1) * dim_x)
  top_area + side_area

theorem painted_surface_area_is_33 :
  total_painted_surface_area 14 3 3 1 = 33 :=
by
  -- Proof would go here
  sorry

end painted_surface_area_is_33_l220_220003


namespace part_a_part_b_l220_220719

-- Part (a)
theorem part_a (ABC : Type) (M: ABC) (R_a R_b R_c r : ℝ):
  ∀ (ABC : Type) (A B C : ABC) (M : ABC), 
  R_a + R_b + R_c ≥ 6 * r := sorry

-- Part (b)
theorem part_b (ABC : Type) (M: ABC) (R_a R_b R_c r : ℝ):
  ∀ (ABC : Type) (A B C : ABC) (M : ABC), 
  R_a^2 + R_b^2 + R_c^2 ≥ 12 * r^2 := sorry

end part_a_part_b_l220_220719


namespace parabola_equation_l220_220033

open Real

theorem parabola_equation (vertex focus : ℝ × ℝ) (h_vertex : vertex = (0, 0)) (h_focus : focus = (0, 3)) :
  ∃ a : ℝ, x^2 = 12 * y := by
  sorry

end parabola_equation_l220_220033


namespace solution_l220_220734

theorem solution :
  ∀ (x : ℝ), x ≠ 0 → (9 * x) ^ 18 = (27 * x) ^ 9 → x = 1 / 3 :=
by
  intro x
  intro h
  intro h_eq
  sorry

end solution_l220_220734


namespace quadratic_roots_x_no_real_solution_y_l220_220021

theorem quadratic_roots_x (x : ℝ) : 
  x^2 - 4*x + 3 = 0 ↔ (x = 3 ∨ x = 1) := sorry

theorem no_real_solution_y (y : ℝ) : 
  ¬∃ y : ℝ, 4*y^2 - 3*y + 2 = 0 := sorry

end quadratic_roots_x_no_real_solution_y_l220_220021


namespace total_time_spent_l220_220827

-- Define the total time for one shoe
def time_per_shoe (time_buckle: ℕ) (time_heel: ℕ) : ℕ :=
  time_buckle + time_heel

-- Conditions
def time_buckle : ℕ := 5
def time_heel : ℕ := 10
def number_of_shoes : ℕ := 2

-- The proof problem statement
theorem total_time_spent :
  (time_per_shoe time_buckle time_heel) * number_of_shoes = 30 :=
by
  sorry

end total_time_spent_l220_220827


namespace nonneg_sets_property_l220_220732

open Set Nat

theorem nonneg_sets_property (A : Set ℕ) :
  (∀ m n : ℕ, m + n ∈ A → m * n ∈ A) ↔
  (A = ∅ ∨ A = {0} ∨ A = {0, 1} ∨ A = {0, 1, 2} ∨ A = {0, 1, 2, 3} ∨ A = {0, 1, 2, 3, 4} ∨ A = { n | 0 ≤ n }) :=
sorry

end nonneg_sets_property_l220_220732


namespace remainder_of_p_div_10_is_6_l220_220417

-- Define the problem
def a : ℕ := sorry -- a is a positive integer and a multiple of 2

-- Define p based on a
def p : ℕ := 4^a

-- The main goal is to prove the remainder when p is divided by 10 is 6
theorem remainder_of_p_div_10_is_6 (ha : a > 0 ∧ a % 2 = 0) : p % 10 = 6 := by
  sorry

end remainder_of_p_div_10_is_6_l220_220417


namespace number_of_whole_numbers_between_sqrts_l220_220922

noncomputable def count_whole_numbers_between_sqrts : ℕ :=
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let min_int := Int.ceil lower_bound
  let max_int := Int.floor upper_bound
  Int.natAbs (max_int - min_int + 1)

theorem number_of_whole_numbers_between_sqrts :
  count_whole_numbers_between_sqrts = 7 :=
by
  sorry

end number_of_whole_numbers_between_sqrts_l220_220922


namespace find_smallest_even_number_l220_220360

theorem find_smallest_even_number (n : ℕ) (h : n + (n + 2) + (n + 4) = 162) : n = 52 :=
by
  sorry

end find_smallest_even_number_l220_220360


namespace chord_slope_range_l220_220599

theorem chord_slope_range (x1 y1 x2 y2 x0 y0 : ℝ) (h1 : x1^2 + (y1^2)/4 = 1) (h2 : x2^2 + (y2^2)/4 = 1)
  (h3 : x0 = (x1 + x2) / 2) (h4 : y0 = (y1 + y2) / 2)
  (h5 : x0 = 1/2) (h6 : 1/2 ≤ y0 ∧ y0 ≤ 1) :
  -4 ≤ (-2 / y0) ∧ -2 ≤ (-2 / y0) :=
by
  sorry

end chord_slope_range_l220_220599


namespace greatest_number_is_2040_l220_220469

theorem greatest_number_is_2040 (certain_number : ℕ) : 
  (∀ d : ℕ, d ∣ certain_number ∧ d ∣ 2037 → d ≤ 1) ∧ 
  (certain_number % 1 = 10) ∧ 
  (2037 % 1 = 7) → 
  certain_number = 2040 :=
by
  sorry

end greatest_number_is_2040_l220_220469


namespace coefficient_x2y6_expansion_l220_220049

theorem coefficient_x2y6_expansion :
  let x : ℤ := 1
  let y : ℤ := 1
  ∃ a : ℤ, a = -28 ∧ (a • x ^ 2 * y ^ 6) = (1 - y / x) * (x + y) ^ 8 :=
by
  sorry

end coefficient_x2y6_expansion_l220_220049


namespace quadratic_inequality_solution_set_l220_220162

variable (a b c : ℝ) (α β : ℝ)

theorem quadratic_inequality_solution_set
  (hαβ : α < β)
  (hα_lt_0 : α < 0) 
  (hβ_lt_0 : β < 0)
  (h_sol_set : ∀ x : ℝ, a * x^2 + b * x + c < 0 ↔ (x < α ∨ x > β)) :
  (∀ x : ℝ, c * x^2 - b * x + a > 0 ↔ (-(1 / α) < x ∧ x < -(1 / β))) :=
  sorry

end quadratic_inequality_solution_set_l220_220162


namespace consecutive_odd_numbers_square_difference_l220_220782

theorem consecutive_odd_numbers_square_difference (a b : ℤ) :
  (a - b = 2 ∨ b - a = 2) → (a^2 - b^2 = 2000) → (a = 501 ∧ b = 499 ∨ a = -501 ∧ b = -499) :=
by 
  intros h1 h2
  sorry

end consecutive_odd_numbers_square_difference_l220_220782


namespace largest_integer_condition_l220_220202

theorem largest_integer_condition (m a b : ℤ) 
  (h1 : m < 150) 
  (h2 : m > 50) 
  (h3 : m = 9 * a - 2) 
  (h4 : m = 6 * b - 4) : 
  m = 106 := 
sorry

end largest_integer_condition_l220_220202


namespace trig_identity_l220_220898

theorem trig_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 :=
by
  sorry

end trig_identity_l220_220898


namespace red_pairs_count_l220_220275

theorem red_pairs_count (students_green : ℕ) (students_red : ℕ) (total_students : ℕ) (total_pairs : ℕ)
(pairs_green_green : ℕ) : 
students_green = 63 →
students_red = 69 →
total_students = 132 →
total_pairs = 66 →
pairs_green_green = 21 →
∃ (pairs_red_red : ℕ), pairs_red_red = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end red_pairs_count_l220_220275


namespace find_angle_degree_l220_220226

theorem find_angle_degree (x : ℝ) (h : 90 - x = (1 / 3) * (180 - x) + 20) : x = 75 := by
    sorry

end find_angle_degree_l220_220226


namespace increasing_sequence_range_l220_220565

theorem increasing_sequence_range (a : ℝ) (a_seq : ℕ → ℝ)
  (h₁ : ∀ (n : ℕ), n ≤ 5 → a_seq n = (5 - a) * n - 11)
  (h₂ : ∀ (n : ℕ), n > 5 → a_seq n = a ^ (n - 4))
  (h₃ : ∀ (n : ℕ), a_seq n < a_seq (n + 1)) :
  2 < a ∧ a < 5 := 
sorry

end increasing_sequence_range_l220_220565


namespace simplify_expression_l220_220611

theorem simplify_expression (x : ℝ) 
  (h1 : x^2 - 4*x + 3 = (x-3)*(x-1))
  (h2 : x^2 - 6*x + 9 = (x-3)^2)
  (h3 : x^2 - 6*x + 8 = (x-2)*(x-4))
  (h4 : x^2 - 8*x + 15 = (x-3)*(x-5)) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = (x-1)*(x-5) / ((x-2)*(x-4)) :=
by
  sorry

end simplify_expression_l220_220611


namespace car_speed_l220_220208

theorem car_speed (distance time speed : ℝ)
  (h_const_speed : ∀ t : ℝ, t = time → speed = distance / t)
  (h_distance : distance = 48)
  (h_time : time = 8) :
  speed = 6 :=
by
  sorry

end car_speed_l220_220208


namespace fraction_zero_when_x_eq_3_l220_220002

theorem fraction_zero_when_x_eq_3 : ∀ x : ℝ, x = 3 → (x^6 - 54 * x^3 + 729) / (x^3 - 27) = 0 :=
by
  intro x hx
  rw [hx]
  sorry

end fraction_zero_when_x_eq_3_l220_220002


namespace line_equation_of_projection_l220_220475

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_v2 := v.1 * v.1 + v.2 * v.2
  (dot_uv / norm_v2 * v.1, dot_uv / norm_v2 * v.2)

theorem line_equation_of_projection (x y : ℝ) :
  proj (x, y) (3, -4) = (9 / 5, -12 / 5) ↔ y = (3 / 4) * x - 15 / 4 :=
sorry

end line_equation_of_projection_l220_220475


namespace consecutive_even_numbers_sum_is_3_l220_220986

-- Definitions from the conditions provided
def consecutive_even_numbers := [80, 82, 84]
def sum_of_numbers := 246

-- The problem is to prove that there are 3 consecutive even numbers summing up to 246
theorem consecutive_even_numbers_sum_is_3 :
  (consecutive_even_numbers.sum = sum_of_numbers) → consecutive_even_numbers.length = 3 :=
by
  sorry

end consecutive_even_numbers_sum_is_3_l220_220986


namespace find_income_4_l220_220882

noncomputable def income_4 (income_1 income_2 income_3 income_5 average_income num_days : ℕ) : ℕ :=
  average_income * num_days - (income_1 + income_2 + income_3 + income_5)

theorem find_income_4
  (income_1 : ℕ := 200)
  (income_2 : ℕ := 150)
  (income_3 : ℕ := 750)
  (income_5 : ℕ := 500)
  (average_income : ℕ := 400)
  (num_days : ℕ := 5) :
  income_4 income_1 income_2 income_3 income_5 average_income num_days = 400 :=
by
  unfold income_4
  sorry

end find_income_4_l220_220882


namespace right_triangle_BD_length_l220_220850

theorem right_triangle_BD_length (BC AC AD BD : ℝ ) (h_bc: BC = 1) (h_ac: AC = b) (h_ad: AD = 2) :
  BD = Real.sqrt (b^2 - 3) :=
by
  sorry

end right_triangle_BD_length_l220_220850


namespace number_of_a_values_l220_220357

theorem number_of_a_values (a : ℝ) : 
  (∃ a : ℝ, ∃ b : ℝ, a = 0 ∨ a = 1) := sorry

end number_of_a_values_l220_220357


namespace probability_of_xiao_li_l220_220005

def total_students : ℕ := 5
def xiao_li : ℕ := 1

noncomputable def probability_xiao_li_chosen : ℚ :=
  (xiao_li : ℚ) / (total_students : ℚ)

theorem probability_of_xiao_li : probability_xiao_li_chosen = 1 / 5 :=
sorry

end probability_of_xiao_li_l220_220005


namespace g_at_five_l220_220618

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_five :
  (∀ x : ℝ, g (3 * x - 7) = 4 * x + 6) →
  g (5) = 22 :=
by
  intros h
  sorry

end g_at_five_l220_220618


namespace expand_polynomial_l220_220371

theorem expand_polynomial (x : ℝ) : 
  3 * (x - 2) * (x^2 + x + 1) = 3 * x^3 - 3 * x^2 - 3 * x - 6 :=
by
  sorry

end expand_polynomial_l220_220371


namespace max_a2b3c4_l220_220045

noncomputable def maximum_value (a b c : ℝ) : ℝ := a^2 * b^3 * c^4

theorem max_a2b3c4 (a b c : ℝ) (h₁ : a + b + c = 2) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  maximum_value a b c ≤ 143327232 / 386989855 := sorry

end max_a2b3c4_l220_220045


namespace x_is_36_percent_of_z_l220_220792

variable (x y z : ℝ)

theorem x_is_36_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.30 * z) : x = 0.36 * z :=
by
  sorry

end x_is_36_percent_of_z_l220_220792


namespace min_value_of_expression_l220_220367

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  (1 / (2 * a) + 2 / b) = 8 :=
sorry

end min_value_of_expression_l220_220367


namespace projection_magnitude_of_a_onto_b_equals_neg_three_l220_220234

variables {a b : ℝ}

def vector_magnitude (v : ℝ) : ℝ := abs v

def dot_product (a b : ℝ) : ℝ := a * b

noncomputable def projection (a b : ℝ) : ℝ := (dot_product a b) / (vector_magnitude b)

theorem projection_magnitude_of_a_onto_b_equals_neg_three
  (ha : vector_magnitude a = 5)
  (hb : vector_magnitude b = 3)
  (hab : dot_product a b = -9) :
  projection a b = -3 :=
by sorry

end projection_magnitude_of_a_onto_b_equals_neg_three_l220_220234


namespace daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l220_220605

-- Part (1)
theorem daily_sales_volume_and_profit (x : ℝ) :
  let increase_in_sales := 2 * x
  let profit_per_piece := 40 - x
  increase_in_sales = 2 * x ∧ profit_per_piece = 40 - x :=
by
  sorry

-- Part (2)
theorem profit_for_1200_yuan (x : ℝ) (h1 : (40 - x) * (20 + 2 * x) = 1200) :
  x = 10 ∨ x = 20 :=
by
  sorry

-- Part (3)
theorem profit_impossible_for_1800_yuan :
  ¬ ∃ y : ℝ, (40 - y) * (20 + 2 * y) = 1800 :=
by
  sorry

end daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l220_220605


namespace probability_of_at_least_one_pair_of_women_l220_220836

/--
Theorem: Calculate the probability that at least one pair consists of two young women from a group of 6 young men and 6 young women paired up randomly is 0.93.
-/
theorem probability_of_at_least_one_pair_of_women 
  (men_women_group : Finset (Fin 12))
  (pairs : Finset (Finset (Fin 12)))
  (h_pairs : pairs.card = 6)
  (h_men_women : ∀ pair ∈ pairs, pair.card = 2)
  (h_distinct : ∀ (x y : Finset (Fin 12)), x ≠ y → x ∩ y = ∅):
  ∃ (p : ℝ), p = 0.93 := 
sorry

end probability_of_at_least_one_pair_of_women_l220_220836


namespace average_speed_l220_220861

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem average_speed (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a * b) / (a + b) = (2 * b * a) / (a + b) :=
by
  sorry

end average_speed_l220_220861


namespace angle_BDC_is_30_l220_220057

theorem angle_BDC_is_30 
    (A E C B D : ℝ) 
    (hA : A = 50) 
    (hE : E = 60) 
    (hC : C = 40) : 
    BDC = 30 :=
by
  sorry

end angle_BDC_is_30_l220_220057


namespace pot_holds_three_liters_l220_220849

theorem pot_holds_three_liters (drips_per_minute : ℕ) (ml_per_drop : ℕ) (minutes : ℕ) (full_pot_volume : ℕ) :
  drips_per_minute = 3 → ml_per_drop = 20 → minutes = 50 → full_pot_volume = (drips_per_minute * ml_per_drop * minutes) / 1000 →
  full_pot_volume = 3 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end pot_holds_three_liters_l220_220849


namespace product_of_abc_l220_220289

variable (a b c m : ℚ)

-- Conditions
def condition1 : Prop := a + b + c = 200
def condition2 : Prop := 8 * a = m
def condition3 : Prop := m = b - 10
def condition4 : Prop := m = c + 10

-- The theorem to prove
theorem product_of_abc :
  a + b + c = 200 ∧ 8 * a = m ∧ m = b - 10 ∧ m = c + 10 →
  a * b * c = 505860000 / 4913 :=
by
  sorry

end product_of_abc_l220_220289


namespace gain_percentage_for_40_clocks_is_10_l220_220656

-- Condition: Cost price per clock
def cost_price := 79.99999999999773

-- Condition: Selling price of 50 clocks at a gain of 20%
def selling_price_50 := 50 * cost_price * 1.20

-- Uniform profit condition
def uniform_profit_total := 90 * cost_price * 1.15

-- Given total revenue difference Rs. 40
def total_revenue := uniform_profit_total + 40

-- Question: Prove that selling price of 40 clocks leads to 10% gain
theorem gain_percentage_for_40_clocks_is_10 :
    40 * cost_price * 1.10 = total_revenue - selling_price_50 :=
by
  sorry

end gain_percentage_for_40_clocks_is_10_l220_220656


namespace survived_more_than_died_l220_220299

-- Define the given conditions
def total_trees : ℕ := 13
def trees_died : ℕ := 6
def trees_survived : ℕ := total_trees - trees_died

-- The proof statement
theorem survived_more_than_died :
  trees_survived - trees_died = 1 := 
by
  -- This is where the proof would go
  sorry

end survived_more_than_died_l220_220299


namespace wedding_chairs_total_l220_220852

theorem wedding_chairs_total :
  let first_section_rows := 5
  let first_section_chairs_per_row := 10
  let first_section_late_people := 15
  let first_section_extra_chairs_per_late := 2
  
  let second_section_rows := 8
  let second_section_chairs_per_row := 12
  let second_section_late_people := 25
  let second_section_extra_chairs_per_late := 3
  
  let third_section_rows := 4
  let third_section_chairs_per_row := 15
  let third_section_late_people := 8
  let third_section_extra_chairs_per_late := 1

  let fourth_section_rows := 6
  let fourth_section_chairs_per_row := 9
  let fourth_section_late_people := 12
  let fourth_section_extra_chairs_per_late := 1
  
  let total_original_chairs := 
    (first_section_rows * first_section_chairs_per_row) + 
    (second_section_rows * second_section_chairs_per_row) + 
    (third_section_rows * third_section_chairs_per_row) + 
    (fourth_section_rows * fourth_section_chairs_per_row)
  
  let total_extra_chairs :=
    (first_section_late_people * first_section_extra_chairs_per_late) + 
    (second_section_late_people * second_section_extra_chairs_per_late) + 
    (third_section_late_people * third_section_extra_chairs_per_late) + 
    (fourth_section_late_people * fourth_section_extra_chairs_per_late)
  
  total_original_chairs + total_extra_chairs = 385 :=
by
  sorry

end wedding_chairs_total_l220_220852


namespace solve_system_of_equations_l220_220536

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) :=
by
  sorry

end solve_system_of_equations_l220_220536


namespace polynomial_remainder_l220_220737

theorem polynomial_remainder (x : ℂ) : 
  (3 * x ^ 1010 + x ^ 1000) % (x ^ 2 + 1) * (x - 1) = 3 * x ^ 2 + 1 := 
sorry

end polynomial_remainder_l220_220737


namespace town_population_l220_220588

variable (P₀ P₁ P₂ : ℝ)

def population_two_years_ago (P₀ : ℝ) : Prop := P₀ = 800

def first_year_increase (P₀ P₁ : ℝ) : Prop := P₁ = P₀ * 1.25

def second_year_increase (P₁ P₂ : ℝ) : Prop := P₂ = P₁ * 1.15

theorem town_population 
  (h₀ : population_two_years_ago P₀)
  (h₁ : first_year_increase P₀ P₁)
  (h₂ : second_year_increase P₁ P₂) : 
  P₂ = 1150 := 
sorry

end town_population_l220_220588


namespace weight_loss_challenge_l220_220537

noncomputable def percentage_weight_loss (W : ℝ) : ℝ :=
  ((W - (0.918 * W)) / W) * 100

theorem weight_loss_challenge (W : ℝ) (h : W > 0) :
  percentage_weight_loss W = 8.2 :=
by
  sorry

end weight_loss_challenge_l220_220537


namespace percentage_problem_l220_220692

noncomputable def percentage_of_value (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
  (y / x) * 100

theorem percentage_problem :
  percentage_of_value 2348 (528.0642570281125 * 4.98) = 112 := 
by
  sorry

end percentage_problem_l220_220692


namespace height_of_taller_tree_l220_220614

theorem height_of_taller_tree 
  (h : ℝ) 
  (ratio_condition : (h - 20) / h = 2 / 3) : 
  h = 60 := 
by 
  sorry

end height_of_taller_tree_l220_220614


namespace down_payment_amount_l220_220008

-- Define the monthly savings per person
def monthly_savings_per_person : ℤ := 1500

-- Define the number of people
def number_of_people : ℤ := 2

-- Define the total monthly savings
def total_monthly_savings : ℤ := monthly_savings_per_person * number_of_people

-- Define the number of years they will save
def years_saving : ℤ := 3

-- Define the number of months in a year
def months_in_year : ℤ := 12

-- Define the total number of months
def total_months : ℤ := years_saving * months_in_year

-- Define the total savings needed for the down payment
def total_savings_needed : ℤ := total_monthly_savings * total_months

-- Prove that the total amount needed for the down payment is $108,000
theorem down_payment_amount : total_savings_needed = 108000 := by
  -- This part requires a proof, which we skip with sorry
  sorry

end down_payment_amount_l220_220008


namespace pi_bounds_l220_220888

theorem pi_bounds : 
  3.14 < Real.pi ∧ Real.pi < 3.142 ∧
  9.86 < Real.pi ^ 2 ∧ Real.pi ^ 2 < 9.87 := sorry

end pi_bounds_l220_220888


namespace fractions_arith_l220_220053

theorem fractions_arith : (3 / 50) + (2 / 25) - (5 / 1000) = 0.135 := by
  sorry

end fractions_arith_l220_220053


namespace complex_power_identity_l220_220895

theorem complex_power_identity (z : ℂ) (i : ℂ) 
  (h1 : z = (1 + i) / Real.sqrt 2) 
  (h2 : z^2 = i) : 
  z^100 = -1 := 
  sorry

end complex_power_identity_l220_220895


namespace odd_blue_faces_in_cubes_l220_220698

noncomputable def count_odd_blue_faces (length width height : ℕ) : ℕ :=
if length = 6 ∧ width = 4 ∧ height = 2 then 16 else 0

theorem odd_blue_faces_in_cubes : count_odd_blue_faces 6 4 2 = 16 := 
by
  -- The proof would involve calculating the corners, edges, etc.
  sorry

end odd_blue_faces_in_cubes_l220_220698


namespace find_f_2005_1000_l220_220293

-- Define the real-valued function and its properties
def f (x y : ℝ) : ℝ := sorry

-- The condition given in the problem
axiom condition :
  ∀ x y z : ℝ, f x y = f x z - 2 * f y z - 2 * z

-- The target we need to prove
theorem find_f_2005_1000 : f 2005 1000 = 5 := 
by 
  -- all necessary logical steps (detailed in solution) would go here
  sorry

end find_f_2005_1000_l220_220293


namespace cr_inequality_l220_220327

theorem cr_inequality 
  (a b : ℝ) (r : ℝ)
  (cr : ℝ := if r < 1 then 1 else 2^(r - 1)) 
  (h0 : r ≥ 0) : 
  |a + b|^r ≤ cr * (|a|^r + |b|^r) :=
by 
  sorry

end cr_inequality_l220_220327


namespace glucose_solution_volume_l220_220911

theorem glucose_solution_volume
  (h1 : 6.75 / 45 = 15 / x) :
  x = 100 :=
by
  sorry

end glucose_solution_volume_l220_220911


namespace probability_right_triangle_in_3x3_grid_l220_220066

theorem probability_right_triangle_in_3x3_grid : 
  let vertices := (3 + 1) * (3 + 1)
  let total_combinations := Nat.choose vertices 3
  let right_triangles_on_gridlines := 144
  let right_triangles_off_gridlines := 24 + 32
  let total_right_triangles := right_triangles_on_gridlines + right_triangles_off_gridlines
  (total_right_triangles : ℚ) / total_combinations = 5 / 14 :=
by 
  sorry

end probability_right_triangle_in_3x3_grid_l220_220066


namespace xy_sq_is_37_over_36_l220_220069

theorem xy_sq_is_37_over_36 (x y : ℚ) (h : 2002 * (x - 1)^2 + |x - 12 * y + 1| = 0) : x^2 + y^2 = 37 / 36 :=
sorry

end xy_sq_is_37_over_36_l220_220069


namespace number_of_intersections_l220_220086

   -- Definitions corresponding to conditions
   def C1 (x y : ℝ) : Prop := x^2 - y^2 + 4*y - 3 = 0
   def C2 (a x y : ℝ) : Prop := y = a*x^2
   def positive_real (a : ℝ) : Prop := a > 0

   -- Final statement converting the question, conditions, and correct answer into Lean code
   theorem number_of_intersections (a : ℝ) (ha : positive_real a) :
     ∃ (count : ℕ), (count = 4) ∧
     (∀ x y : ℝ, C1 x y → C2 a x y → True) := sorry
   
end number_of_intersections_l220_220086


namespace solution_set_of_inequality_l220_220601

variable {α : Type*} [LinearOrder α]

def is_decreasing (f : α → α) : Prop :=
  ∀ ⦃x y⦄, x < y → f y < f x

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_decreasing : is_decreasing f)
  (domain_cond : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → x ∈ Set.Ioo (-2 : ℝ) 2)
  : { x | x > 0 ∧ x < 1 } = { x | f x > f (2 - x) } :=
by {
  sorry
}

end solution_set_of_inequality_l220_220601


namespace abc_sum_l220_220428

theorem abc_sum
  (a b c : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 17 * x + 70 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 - 19 * x + 84 = (x - b) * (x - c)) :
  a + b + c = 29 := by
  sorry

end abc_sum_l220_220428


namespace calculate_bankers_discount_l220_220524

noncomputable def present_worth : ℝ := 800
noncomputable def true_discount : ℝ := 36
noncomputable def face_value : ℝ := present_worth + true_discount
noncomputable def bankers_discount : ℝ := (face_value * true_discount) / (face_value - true_discount)

theorem calculate_bankers_discount :
  bankers_discount = 37.62 := 
sorry

end calculate_bankers_discount_l220_220524


namespace area_of_rectangle_l220_220529

-- Definitions from the conditions
def breadth (b : ℝ) : Prop := b > 0
def length (l b : ℝ) : Prop := l = 3 * b
def perimeter (P l b : ℝ) : Prop := P = 2 * (l + b)

-- The main theorem we are proving
theorem area_of_rectangle (b l : ℝ) (P : ℝ) (h1 : breadth b) (h2 : length l b) (h3 : perimeter P l b) (h4 : P = 96) : l * b = 432 := 
by
  -- Proof steps will go here
  sorry

end area_of_rectangle_l220_220529


namespace age_twice_of_father_l220_220979

theorem age_twice_of_father (S M Y : ℕ) (h₁ : S = 22) (h₂ : M = S + 24) (h₃ : M + Y = 2 * (S + Y)) : Y = 2 := by
  sorry

end age_twice_of_father_l220_220979


namespace increasing_interval_l220_220726

noncomputable def f (x k : ℝ) : ℝ := (x^2 / 2) - k * (Real.log x)

theorem increasing_interval (k : ℝ) (h₀ : 0 < k) : 
  ∃ (a : ℝ), (a = Real.sqrt k) ∧ 
  ∀ (x : ℝ), (x > a) → (∃ ε > 0, ∀ y, (x < y) → (f y k > f x k)) :=
sorry

end increasing_interval_l220_220726


namespace tan_2beta_l220_220512

theorem tan_2beta {α β : ℝ} 
  (h₁ : Real.tan (α + β) = 2) 
  (h₂ : Real.tan (α - β) = 3) : 
  Real.tan (2 * β) = -1 / 7 :=
by 
  sorry

end tan_2beta_l220_220512


namespace fraction_evaluation_l220_220455

theorem fraction_evaluation (x z : ℚ) (hx : x = 4/7) (hz : z = 8/11) :
  (7 * x + 10 * z) / (56 * x * z) = 31 / 176 := by
  sorry

end fraction_evaluation_l220_220455


namespace find_XY_XZ_l220_220705

open Set

variable (P Q R X Y Z : Type) [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited X] [Inhabited Y] [Inhabited Z]
variable (length : (P → P → Real) → (Q → Q → Real) → (R → R → Real) → (X → X → Real) → (Y → Y → Real) → (Z → Z → Real) )


-- Definitions based on the conditions
def similar_triangles (PQ QR PR XY XZ YZ : Real) : Prop :=
  QR / YZ = PQ / XY ∧ QR / YZ = PR / XZ

def PQ : Real := 8
def QR : Real := 16
def YZ : Real := 32

-- We need to prove (XY = 16 ∧ XZ = 32) given the conditions of similarity
theorem find_XY_XZ (XY XZ : Real) (h_sim : similar_triangles PQ QR PQ XY XZ YZ) : XY = 16 ∧ XZ = 32 :=
by
  sorry

end find_XY_XZ_l220_220705


namespace candy_per_bag_correct_l220_220395

def total_candy : ℕ := 648
def sister_candy : ℕ := 48
def friends : ℕ := 3
def bags : ℕ := 8

def remaining_candy (total candy_kept : ℕ) : ℕ := total - candy_kept
def candy_per_person (remaining people : ℕ) : ℕ := remaining / people
def candy_per_bag (per_person bags : ℕ) : ℕ := per_person / bags

theorem candy_per_bag_correct :
  candy_per_bag (candy_per_person (remaining_candy total_candy sister_candy) (friends + 1)) bags = 18 :=
by
  sorry

end candy_per_bag_correct_l220_220395


namespace fraction_pattern_l220_220549

theorem fraction_pattern (n m k : ℕ) (h : n / m = k * n / (k * m)) : (n + m) / m = (k * n + k * m) / (k * m) := by
  sorry

end fraction_pattern_l220_220549


namespace roots_not_in_interval_l220_220751

theorem roots_not_in_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2 * a) → (x < -1 ∨ x > 1) :=
by
  sorry

end roots_not_in_interval_l220_220751


namespace min_adjacent_seat_occupation_l220_220689

def minOccupiedSeats (n : ℕ) : ℕ :=
  n / 3

theorem min_adjacent_seat_occupation (n : ℕ) (h : n = 150) :
  minOccupiedSeats n = 50 :=
by
  -- Placeholder for proof
  sorry

end min_adjacent_seat_occupation_l220_220689


namespace cube_vertices_count_l220_220516

-- Defining the conditions of the problem
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def euler_formula (V E F : ℕ) : Prop := V - E + F = 2

-- Stating the proof problem
theorem cube_vertices_count : ∃ V : ℕ, euler_formula V num_edges num_faces ∧ V = 8 :=
by
  sorry

end cube_vertices_count_l220_220516


namespace engineer_formula_updated_l220_220857

theorem engineer_formula_updated (T H : ℕ) (hT : T = 5) (hH : H = 10) :
  (30 * T^5) / (H^3 : ℚ) = 375 / 4 := by
  sorry

end engineer_formula_updated_l220_220857


namespace angle_measure_supplement_complement_l220_220216

theorem angle_measure_supplement_complement (x : ℝ) 
    (h1 : 180 - x = 7 * (90 - x)) : 
    x = 75 := by
  sorry

end angle_measure_supplement_complement_l220_220216


namespace coefficient_of_x_in_expansion_l220_220438

theorem coefficient_of_x_in_expansion : 
  (Polynomial.coeff (((X ^ 2 + 3 * X + 2) ^ 6) : Polynomial ℤ) 1) = 576 := 
by 
  sorry

end coefficient_of_x_in_expansion_l220_220438


namespace smallest_integer_x_l220_220430

theorem smallest_integer_x (x : ℤ) : 
  ( ∀ x : ℤ, ( 2 * (x : ℚ) / 5 + 3 / 4 > 7 / 5 → 2 ≤ x )) :=
by
  intro x
  sorry

end smallest_integer_x_l220_220430


namespace number_of_triangles_l220_220492

theorem number_of_triangles (points_AB points_BC points_AC : ℕ)
                            (hAB : points_AB = 12)
                            (hBC : points_BC = 9)
                            (hAC : points_AC = 10) :
    let total_points := points_AB + points_BC + points_AC
    let total_combinations := Nat.choose total_points 3
    let degenerate_AB := Nat.choose points_AB 3
    let degenerate_BC := Nat.choose points_BC 3
    let degenerate_AC := Nat.choose points_AC 3
    let valid_triangles := total_combinations - (degenerate_AB + degenerate_BC + degenerate_AC)
    valid_triangles = 4071 :=
by
  sorry

end number_of_triangles_l220_220492


namespace Elina_garden_area_l220_220188

theorem Elina_garden_area :
  ∀ (L W: ℝ),
    (30 * L = 1500) →
    (12 * (2 * (L + W)) = 1500) →
    (L * W = 625) :=
by
  intros L W h1 h2
  sorry

end Elina_garden_area_l220_220188


namespace range_of_a_if_p_is_false_l220_220855

theorem range_of_a_if_p_is_false (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_if_p_is_false_l220_220855


namespace find_g_zero_l220_220098

variable {g : ℝ → ℝ}

theorem find_g_zero (h : ∀ x y : ℝ, g (x + y) = g x + g y - 1) : g 0 = 1 :=
sorry

end find_g_zero_l220_220098


namespace photo_students_count_l220_220474

theorem photo_students_count (n m : ℕ) 
  (h1 : m - 1 = n + 4) 
  (h2 : m - 2 = n) : 
  n * m = 24 := 
by 
  sorry

end photo_students_count_l220_220474


namespace total_tickets_sold_l220_220583

def price_adult_ticket : ℕ := 7
def price_child_ticket : ℕ := 4
def total_revenue : ℕ := 5100
def adult_tickets_sold : ℕ := 500

theorem total_tickets_sold : 
  ∃ (child_tickets_sold : ℕ), 
    price_adult_ticket * adult_tickets_sold + price_child_ticket * child_tickets_sold = total_revenue ∧
    adult_tickets_sold + child_tickets_sold = 900 :=
by
  sorry

end total_tickets_sold_l220_220583


namespace total_oranges_and_apples_l220_220166

-- Given conditions as definitions
def bags_with_5_oranges_and_7_apples (m : ℕ) : ℕ × ℕ :=
  (5 * m + 1, 7 * m)

def bags_with_9_oranges_and_7_apples (n : ℕ) : ℕ × ℕ :=
  (9 * n, 7 * n + 21)

theorem total_oranges_and_apples (m n : ℕ) (k : ℕ) 
  (h1 : (5 * m + 1, 7 * m) = (9 * n, 7 * n + 21)) 
  (h2 : 4 * n ≡ 1 [MOD 5]) : 85 = 36 + 49 :=
by
  sorry

end total_oranges_and_apples_l220_220166


namespace intersection_of_A_and_B_l220_220152

def A := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1) / Real.log 2}
def B := {x : ℝ | x < 2}

theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l220_220152


namespace granger_buys_3_jars_of_peanut_butter_l220_220339

theorem granger_buys_3_jars_of_peanut_butter :
  ∀ (spam_cost peanut_butter_cost bread_cost total_cost spam_count loaf_count peanut_butter_count: ℕ),
    spam_cost = 3 → peanut_butter_cost = 5 → bread_cost = 2 →
    spam_count = 12 → loaf_count = 4 → total_cost = 59 →
    spam_cost * spam_count + bread_cost * loaf_count + peanut_butter_cost * peanut_butter_count = total_cost →
    peanut_butter_count = 3 :=
by
  intros spam_cost peanut_butter_cost bread_cost total_cost spam_count loaf_count peanut_butter_count
  intros hspam_cost hpeanut_butter_cost hbread_cost hspam_count hloaf_count htotal_cost htotal
  sorry  -- The proof step is omitted as requested.

end granger_buys_3_jars_of_peanut_butter_l220_220339


namespace distance_between_trees_l220_220900

theorem distance_between_trees (trees : ℕ) (total_length : ℝ) (n : trees = 26) (l : total_length = 500) :
  ∃ d : ℝ, d = total_length / (trees - 1) ∧ d = 20 :=
by
  sorry

end distance_between_trees_l220_220900


namespace emily_speed_l220_220421

theorem emily_speed (distance time : ℝ) (h1 : distance = 10) (h2 : time = 2) : (distance / time) = 5 := 
by sorry

end emily_speed_l220_220421


namespace remaining_bottles_after_2_days_l220_220918

-- Definitions based on the conditions:
def initial_bottles : ℕ := 24
def fraction_first_day : ℚ := 1 / 3
def fraction_second_day : ℚ := 1 / 2

-- Theorem statement proving the remaining number of bottles after 2 days
theorem remaining_bottles_after_2_days : 
    (initial_bottles - initial_bottles * fraction_first_day) - 
    ((initial_bottles - initial_bottles * fraction_first_day) * fraction_second_day) = 8 := 
by 
    -- Skipping the proof
    sorry

end remaining_bottles_after_2_days_l220_220918


namespace cricket_target_l220_220674

theorem cricket_target (run_rate_first_10overs run_rate_next_40overs : ℝ) (overs_first_10 next_40_overs : ℕ)
    (h_first : run_rate_first_10overs = 3.2) 
    (h_next : run_rate_next_40overs = 6.25) 
    (h_overs_first : overs_first_10 = 10) 
    (h_overs_next : next_40_overs = 40) 
    : (overs_first_10 * run_rate_first_10overs + next_40_overs * run_rate_next_40overs) = 282 :=
by
  sorry

end cricket_target_l220_220674


namespace find_cost_price_l220_220082

variable (C : ℝ)

theorem find_cost_price (h : 56 - C = C - 42) : C = 49 :=
by
  sorry

end find_cost_price_l220_220082


namespace expected_total_rainfall_over_week_l220_220554

noncomputable def daily_rain_expectation : ℝ :=
  (0.5 * 0) + (0.2 * 2) + (0.3 * 5)

noncomputable def total_rain_expectation (days: ℕ) : ℝ :=
  days * daily_rain_expectation

theorem expected_total_rainfall_over_week : total_rain_expectation 7 = 13.3 :=
by 
  -- calculation of expected value here
  -- daily_rain_expectation = 1.9
  -- total_rain_expectation 7 = 7 * 1.9 = 13.3
  sorry

end expected_total_rainfall_over_week_l220_220554


namespace find_a_l220_220235

-- Define the polynomial expansion term conditions
def binomial_coefficient (n k : ℕ) := Nat.choose n k

def fourth_term_coefficient (x a : ℝ) : ℝ :=
  binomial_coefficient 9 3 * x^6 * a^3

theorem find_a (a : ℝ) (x : ℝ) (h : fourth_term_coefficient x a = 84) : a = 1 :=
by
  unfold fourth_term_coefficient at h
  sorry

end find_a_l220_220235


namespace carpet_dimensions_l220_220507

theorem carpet_dimensions
  (x y q : ℕ)
  (h_dim : y = 2 * x)
  (h_room1 : ((q^2 + 50^2) = (q * 2 - 50)^2 + (50 * 2 - q)^2))
  (h_room2 : ((q^2 + 38^2) = (q * 2 - 38)^2 + (38 * 2 - q)^2)) :
  x = 25 ∧ y = 50 :=
sorry

end carpet_dimensions_l220_220507


namespace perpendicular_line_through_A_l220_220584

variable (m : ℝ)

-- Conditions
def line1 (x y : ℝ) : Prop := x + (1 + m) * y + m - 2 = 0
def line2 (x y : ℝ) : Prop := m * x + 2 * y + 8 = 0
def pointA : ℝ × ℝ := (3, 2)

-- Question and proof
theorem perpendicular_line_through_A (h_parallel : ∃ x y, line1 m x y ∧ line2 m x y) :
  ∃ (t : ℝ), ∀ (x y : ℝ), (y = 2 * x + t) ↔ (2 * x - y - 4 = 0) :=
by
  sorry

end perpendicular_line_through_A_l220_220584


namespace p_necessary_not_sufficient_for_q_l220_220752

def condition_p (x : ℝ) : Prop := x > 2
def condition_q (x : ℝ) : Prop := x > 3

theorem p_necessary_not_sufficient_for_q (x : ℝ) :
  (∀ (x : ℝ), condition_q x → condition_p x) ∧ ¬(∀ (x : ℝ), condition_p x → condition_q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l220_220752


namespace banana_group_size_l220_220992

theorem banana_group_size (bananas groups : ℕ) (h1 : bananas = 407) (h2 : groups = 11) : bananas / groups = 37 :=
by sorry

end banana_group_size_l220_220992


namespace expand_fraction_product_l220_220562

theorem expand_fraction_product (x : ℝ) (hx : x ≠ 0) : 
  (3 / 4) * (8 / x^2 - 5 * x^3) = 6 / x^2 - 15 * x^3 / 4 := 
by 
  sorry

end expand_fraction_product_l220_220562


namespace expression_for_f_l220_220746

noncomputable def f (x : ℝ) : ℝ := sorry

theorem expression_for_f (x : ℝ) :
  (∀ x, f (x - 1) = x^2) → f x = x^2 + 2 * x + 1 :=
by
  intro h
  sorry

end expression_for_f_l220_220746


namespace prove_a_21022_le_1_l220_220790

-- Define the sequence a_n
variable (a : ℕ → ℝ)

-- Conditions for the sequence
axiom seq_condition {n : ℕ} (hn : n ≥ 1) :
  (a (n + 1))^2 + a n * a (n + 2) ≤ a n + a (n + 2)

-- Positive real numbers condition
axiom seq_positive {n : ℕ} (hn : n ≥ 1) :
  a n > 0

-- The main theorem to prove
theorem prove_a_21022_le_1 :
  a 21022 ≤ 1 :=
sorry

end prove_a_21022_le_1_l220_220790


namespace parabola_vertex_parabola_point_condition_l220_220218

-- Define the parabola function 
def parabola (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

-- 1. Prove the vertex of the parabola
theorem parabola_vertex (m : ℝ) : ∃ x y, (∀ x m, parabola x m = (x - m)^2 - 1) ∧ (x = m ∧ y = -1) :=
by
  sorry

-- 2. Prove the range of values for m given the conditions on points A and B
theorem parabola_point_condition (m : ℝ) (y1 y2 : ℝ) :
  (y1 > y2) ∧ 
  (parabola (1 - 2*m) m = y1) ∧ 
  (parabola (m + 1) m = y2) → m < 0 ∨ m > 2/3 :=
by
  sorry

end parabola_vertex_parabola_point_condition_l220_220218


namespace sum_of_first_five_terms_l220_220525

theorem sum_of_first_five_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q) -- geometric sequence definition
  (h3 : a 2 + a 5 = 2 * (a 4 + 2)) : 
  S 5 = 62 :=
by
  -- lean tactics would go here to provide the proof
  sorry

end sum_of_first_five_terms_l220_220525


namespace extremum_value_of_a_g_monotonicity_l220_220780

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + x ^ 2

theorem extremum_value_of_a (a : ℝ) (h : (3 * a * (-4 / 3) ^ 2 + 2 * (-4 / 3) = 0)) : a = 1 / 2 :=
by
  -- We need to prove that a = 1 / 2 given the extremum condition.
  sorry

noncomputable def g (x : ℝ) : ℝ := (1 / 2 * x ^ 3 + x ^ 2) * Real.exp x

theorem g_monotonicity :
  (∀ x < -4, deriv g x < 0) ∧
  (∀ x, -4 < x ∧ x < -1 → deriv g x > 0) ∧
  (∀ x, -1 < x ∧ x < 0 → deriv g x < 0) ∧
  (∀ x > 0, deriv g x > 0) :=
by
  -- We need to prove the monotonicity of the function g in the specified intervals.
  sorry

end extremum_value_of_a_g_monotonicity_l220_220780


namespace correct_statement_about_CH3COOK_l220_220397

def molar_mass_CH3COOK : ℝ := 98  -- in g/mol

def avogadro_number : ℝ := 6.02 * 10^23  -- molecules per mole

def hydrogen_atoms_in_CH3COOK (mol_CH3COOK : ℝ) : ℝ :=
  3 * mol_CH3COOK * avogadro_number

theorem correct_statement_about_CH3COOK (mol_CH3COOK : ℝ) (h: mol_CH3COOK = 1) :
  hydrogen_atoms_in_CH3COOK mol_CH3COOK = 3 * avogadro_number :=
by
  sorry

end correct_statement_about_CH3COOK_l220_220397


namespace partner_q_investment_time_l220_220177

theorem partner_q_investment_time 
  (P Q R : ℝ)
  (Profit_p Profit_q Profit_r : ℝ)
  (Tp Tq Tr : ℝ)
  (h1 : P / Q = 7 / 5)
  (h2 : Q / R = 5 / 3)
  (h3 : Profit_p / Profit_q = 7 / 14)
  (h4 : Profit_q / Profit_r = 14 / 9)
  (h5 : Tp = 5)
  (h6 : Tr = 9) :
  Tq = 14 :=
by
  sorry

end partner_q_investment_time_l220_220177


namespace vertical_angles_eq_l220_220702

theorem vertical_angles_eq (A B : Type) (are_vertical : A = B) :
  A = B := 
by
  exact are_vertical

end vertical_angles_eq_l220_220702


namespace tangents_and_fraction_l220_220932

theorem tangents_and_fraction
  (α β : ℝ)
  (tan_diff : Real.tan (α - β) = 2)
  (tan_beta : Real.tan β = 4) :
  (7 * Real.sin α - Real.cos α) / (7 * Real.sin α + Real.cos α) = 7 / 5 :=
sorry

end tangents_and_fraction_l220_220932


namespace proof_problem_l220_220531

variables {a b c : Real}

theorem proof_problem (h1 : a < 0) (h2 : |a| < |b|) (h3 : |b| < |c|) (h4 : b < 0) :
  (|a * b| < |b * c|) ∧ (a * c < |b * c|) ∧ (|a + b| < |b + c|) :=
by
  sorry

end proof_problem_l220_220531


namespace height_is_centimeters_weight_is_kilograms_book_length_is_centimeters_book_thickness_is_millimeters_cargo_capacity_is_tons_sleep_time_is_hours_tree_height_is_meters_l220_220290

-- Definitions
def Height (x : ℕ) : Prop := x = 140
def Weight (x : ℕ) : Prop := x = 23
def BookLength (x : ℕ) : Prop := x = 20
def BookThickness (x : ℕ) : Prop := x = 7
def CargoCapacity (x : ℕ) : Prop := x = 4
def SleepTime (x : ℕ) : Prop := x = 9
def TreeHeight (x : ℕ) : Prop := x = 12

-- Propositions
def XiaohongHeightUnit := "centimeters"
def XiaohongWeightUnit := "kilograms"
def MathBookLengthUnit := "centimeters"
def MathBookThicknessUnit := "millimeters"
def TruckCargoCapacityUnit := "tons"
def ChildrenSleepTimeUnit := "hours"
def BigTreeHeightUnit := "meters"

theorem height_is_centimeters (x : ℕ) (h : Height x) : XiaohongHeightUnit = "centimeters" := sorry
theorem weight_is_kilograms (x : ℕ) (w : Weight x) : XiaohongWeightUnit = "kilograms" := sorry
theorem book_length_is_centimeters (x : ℕ) (l : BookLength x) : MathBookLengthUnit = "centimeters" := sorry
theorem book_thickness_is_millimeters (x : ℕ) (t : BookThickness x) : MathBookThicknessUnit = "millimeters" := sorry
theorem cargo_capacity_is_tons (x : ℕ) (c : CargoCapacity x) : TruckCargoCapacityUnit = "tons" := sorry
theorem sleep_time_is_hours (x : ℕ) (s : SleepTime x) : ChildrenSleepTimeUnit = "hours" := sorry
theorem tree_height_is_meters (x : ℕ) (th : TreeHeight x) : BigTreeHeightUnit = "meters" := sorry

end height_is_centimeters_weight_is_kilograms_book_length_is_centimeters_book_thickness_is_millimeters_cargo_capacity_is_tons_sleep_time_is_hours_tree_height_is_meters_l220_220290


namespace total_cans_needed_l220_220547

-- Definitions
def cans_per_box : ℕ := 4
def number_of_boxes : ℕ := 203

-- Statement of the problem
theorem total_cans_needed : cans_per_box * number_of_boxes = 812 := 
by
  -- skipping the proof
  sorry

end total_cans_needed_l220_220547


namespace box_white_balls_count_l220_220070

/--
A box has exactly 100 balls, and each ball is either red, blue, or white.
Given that the box has 12 more blue balls than white balls,
and twice as many red balls as blue balls,
prove that the number of white balls is 16.
-/
theorem box_white_balls_count (W B R : ℕ) 
  (h1 : B = W + 12) 
  (h2 : R = 2 * B) 
  (h3 : W + B + R = 100) : 
  W = 16 := 
sorry

end box_white_balls_count_l220_220070


namespace combined_cost_is_107_l220_220304

def wallet_cost : ℕ := 22
def purse_cost (wallet_price : ℕ) : ℕ := 4 * wallet_price - 3
def combined_cost (wallet_price : ℕ) (purse_price : ℕ) : ℕ := wallet_price + purse_price

theorem combined_cost_is_107 : combined_cost wallet_cost (purse_cost wallet_cost) = 107 := 
by 
  -- Proof
  sorry

end combined_cost_is_107_l220_220304


namespace sum_of_arith_geo_progression_l220_220303

noncomputable def sum_two_numbers (a b : ℝ) : ℝ :=
  a + b

theorem sum_of_arith_geo_progression : 
  ∃ (a b : ℝ), (∃ d : ℝ, a = 4 + d ∧ b = 4 + 2 * d) ∧ 
  (∃ r : ℝ, a * r = b ∧ b * r = 16) ∧ 
  sum_two_numbers a b = 8 + 6 * Real.sqrt 3 :=
by
  sorry

end sum_of_arith_geo_progression_l220_220303


namespace degree_to_radian_l220_220783

theorem degree_to_radian : (855 : ℝ) * (Real.pi / 180) = (59 / 12) * Real.pi :=
by
  sorry

end degree_to_radian_l220_220783


namespace sum_of_inverses_A_B_C_eq_300_l220_220441

theorem sum_of_inverses_A_B_C_eq_300 
  (p q r : ℝ)
  (hroots : ∀ x, (x^3 - 30*x^2 + 105*x - 114 = 0) → (x = p ∨ x = q ∨ x = r))
  (A B C : ℝ)
  (hdecomp : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    (1 / (s^3 - 30*s^2 + 105*s - 114) = A/(s - p) + B/(s - q) + C/(s - r))) :
  (1 / A) + (1 / B) + (1 / C) = 300 :=
sorry

end sum_of_inverses_A_B_C_eq_300_l220_220441


namespace average_students_present_l220_220832

-- Define the total number of students
def total_students : ℝ := 50

-- Define the absent rates for each day
def absent_rate_mon : ℝ := 0.10
def absent_rate_tue : ℝ := 0.12
def absent_rate_wed : ℝ := 0.15
def absent_rate_thu : ℝ := 0.08
def absent_rate_fri : ℝ := 0.05

-- Define the number of students present each day
def present_mon := (1 - absent_rate_mon) * total_students
def present_tue := (1 - absent_rate_tue) * total_students
def present_wed := (1 - absent_rate_wed) * total_students
def present_thu := (1 - absent_rate_thu) * total_students
def present_fri := (1 - absent_rate_fri) * total_students

-- Define the statement to prove
theorem average_students_present : 
  (present_mon + present_tue + present_wed + present_thu + present_fri) / 5 = 45 :=
by 
  -- The proof would go here
  sorry

end average_students_present_l220_220832


namespace sqrt_81_eq_pm_9_l220_220271

theorem sqrt_81_eq_pm_9 (x : ℤ) (hx : x^2 = 81) : x = 9 ∨ x = -9 :=
by
  sorry

end sqrt_81_eq_pm_9_l220_220271


namespace g_diff_l220_220392

def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 2 * x - 1

theorem g_diff (x h : ℝ) : g (x + h) - g x = h * (6 * x^2 + 6 * x * h + 2 * h^2 + 10 * x + 5 * h - 2) := 
by
  sorry

end g_diff_l220_220392


namespace sum_of_midpoints_y_coordinates_l220_220809

theorem sum_of_midpoints_y_coordinates (d e f : ℝ) (h : d + e + f = 15) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_y_coordinates_l220_220809


namespace three_distinct_numbers_l220_220544

theorem three_distinct_numbers (s : ℕ) (A : Finset ℕ) (S : Finset ℕ) (hA : A = Finset.range (4 * s + 1) \ Finset.range 1)
  (hS : S ⊆ A) (hcard: S.card = 2 * s + 2) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x + y = 2 * z :=
by
  sorry

end three_distinct_numbers_l220_220544


namespace find_a_plus_b_l220_220989

noncomputable def f (a b x : ℝ) : ℝ := (a * x^3) / 3 - b * x^2 + a^2 * x - 1 / 3
noncomputable def f_prime (a b x : ℝ) : ℝ := a * x^2 - 2 * b * x + a^2

theorem find_a_plus_b 
  (a b : ℝ)
  (h_deriv : f_prime a b 1 = 0)
  (h_extreme : f a b 1 = 0) :
  a + b = -7 / 9 := 
sorry

end find_a_plus_b_l220_220989


namespace cost_per_item_l220_220215

theorem cost_per_item (total_cost : ℝ) (num_items : ℕ) (cost_per_item : ℝ) 
                      (h1 : total_cost = 26) (h2 : num_items = 8) : 
                      cost_per_item = total_cost / num_items := 
by
  sorry

end cost_per_item_l220_220215


namespace cosine_product_inequality_l220_220237

theorem cosine_product_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  8 * Real.cos A * Real.cos B * Real.cos C ≤ 1 := 
sorry

end cosine_product_inequality_l220_220237


namespace fraction_of_bread_slices_eaten_l220_220184

theorem fraction_of_bread_slices_eaten
    (total_slices : ℕ)
    (slices_used_for_sandwich : ℕ)
    (remaining_slices : ℕ)
    (slices_eaten_for_breakfast : ℕ)
    (h1 : total_slices = 12)
    (h2 : slices_used_for_sandwich = 2)
    (h3 : remaining_slices = 6)
    (h4 : total_slices - slices_used_for_sandwich - remaining_slices = slices_eaten_for_breakfast) :
    slices_eaten_for_breakfast / total_slices = 1 / 3 :=
sorry

end fraction_of_bread_slices_eaten_l220_220184


namespace girls_doctors_percentage_l220_220546

-- Define the total number of students in the class
variables (total_students : ℕ)

-- Define the proportions given in the problem
def proportion_boys : ℚ := 3 / 5
def proportion_boys_who_want_to_be_doctors : ℚ := 1 / 3
def proportion_doctors_who_are_boys : ℚ := 2 / 5

-- Compute the proportion of boys in the class who want to be doctors
def proportion_boys_as_doctors := proportion_boys * proportion_boys_who_want_to_be_doctors

-- Compute the proportion of girls in the class
def proportion_girls := 1 - proportion_boys

-- Compute the number of girls who want to be doctors compared to boys
def proportion_girls_as_doctors := (1 - proportion_doctors_who_are_boys) / proportion_doctors_who_are_boys * proportion_boys_as_doctors

-- Compute the proportion of girls who want to be doctors
def proportion_girls_who_want_to_be_doctors := proportion_girls_as_doctors / proportion_girls

-- Define the expected percentage of girls who want to be doctors
def expected_percentage_girls_who_want_to_be_doctors : ℚ := 75 / 100

-- The theorem we need to prove
theorem girls_doctors_percentage : proportion_girls_who_want_to_be_doctors * 100 = expected_percentage_girls_who_want_to_be_doctors :=
sorry

end girls_doctors_percentage_l220_220546


namespace find_range_of_m_l220_220971

theorem find_range_of_m:
  (∀ x: ℝ, ¬ ∃ x: ℝ, x^2 + (m - 3) * x + 1 = 0) →
  (∀ y: ℝ, ¬ ∀ y: ℝ, x^2 + y^2 / (m - 1) = 1) → 
  1 < m ∧ m ≤ 2 :=
by
  sorry

end find_range_of_m_l220_220971


namespace triangle_angles_l220_220396

theorem triangle_angles (second_angle first_angle third_angle : ℝ) 
  (h1 : first_angle = 2 * second_angle)
  (h2 : third_angle = second_angle + 30)
  (h3 : second_angle + first_angle + third_angle = 180) :
  second_angle = 37.5 ∧ first_angle = 75 ∧ third_angle = 67.5 :=
sorry

end triangle_angles_l220_220396


namespace poster_width_l220_220661
   
   theorem poster_width (h : ℕ) (A : ℕ) (w : ℕ) (h_eq : h = 7) (A_eq : A = 28) (area_eq : w * h = A) : w = 4 :=
   by
   sorry
   
end poster_width_l220_220661


namespace dihedral_minus_solid_equals_expression_l220_220081

-- Definitions based on the conditions provided.
noncomputable def sumDihedralAngles (P : Polyhedron) : ℝ := sorry
noncomputable def sumSolidAngles (P : Polyhedron) : ℝ := sorry
def numFaces (P : Polyhedron) : ℕ := sorry

-- Theorem statement we want to prove.
theorem dihedral_minus_solid_equals_expression (P : Polyhedron) :
  sumDihedralAngles P - sumSolidAngles P = 2 * Real.pi * (numFaces P - 2) :=
sorry

end dihedral_minus_solid_equals_expression_l220_220081


namespace smallest_positive_period_tan_l220_220973

noncomputable def max_value (a b x : ℝ) := b + a * Real.sin x = -1
noncomputable def min_value (a b x : ℝ) := b - a * Real.sin x = -5
noncomputable def a_negative (a : ℝ) := a < 0

theorem smallest_positive_period_tan :
  ∃ (a b : ℝ), (max_value a b 0) ∧ (min_value a b 0) ∧ (a_negative a) →
  (1 / |3 * a + b|) * Real.pi = Real.pi / 9 :=
by
  sorry

end smallest_positive_period_tan_l220_220973


namespace simplify_expression_l220_220232

open Real

-- Assuming lg refers to the common logarithm log base 10
noncomputable def problem_expression : ℝ :=
  log 4 + 2 * log 5 + 4^(-1/2:ℝ)

theorem simplify_expression : problem_expression = 5 / 2 :=
by
  -- Placeholder proof, actual steps not required
  sorry

end simplify_expression_l220_220232


namespace ms_warren_walking_time_l220_220928

/-- 
Ms. Warren ran at 6 mph for 20 minutes. After the run, 
she walked at 2 mph for a certain amount of time. 
She ran and walked a total of 3 miles.
-/
def time_spent_walking (running_speed walking_speed : ℕ) (running_time_minutes : ℕ) (total_distance : ℕ) : ℕ := 
  let running_time_hours := running_time_minutes / 60;
  let distance_ran := running_speed * running_time_hours;
  let distance_walked := total_distance - distance_ran;
  let time_walked_hours := distance_walked / walking_speed;
  time_walked_hours * 60

theorem ms_warren_walking_time :
  time_spent_walking 6 2 20 3 = 30 :=
by
  sorry

end ms_warren_walking_time_l220_220928


namespace full_price_ticket_revenue_l220_220619

theorem full_price_ticket_revenue (f t : ℕ) (p : ℝ) 
  (h1 : f + t = 160) 
  (h2 : f * p + t * (p / 3) = 2500) 
  (h3 : p = 30) :
  f * p = 1350 := 
by sorry

end full_price_ticket_revenue_l220_220619


namespace greatest_percentage_l220_220420

theorem greatest_percentage (pA : ℝ) (pB : ℝ) (wA : ℝ) (wB : ℝ) (sA : ℝ) (sB : ℝ) :
  pA = 0.4 → pB = 0.6 → wA = 0.8 → wB = 0.1 → sA = 0.9 → sB = 0.5 →
  pA * min wA sA + pB * min wB sB = 0.38 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Here you would continue with the proof by leveraging the conditions
  sorry

end greatest_percentage_l220_220420


namespace numeral_of_place_face_value_difference_l220_220532

theorem numeral_of_place_face_value_difference (P F : ℕ) (H : P - F = 63) (Hface : F = 7) : P = 70 :=
sorry

end numeral_of_place_face_value_difference_l220_220532


namespace blackjack_payout_ratio_l220_220958

theorem blackjack_payout_ratio (total_payout original_bet : ℝ) (h1 : total_payout = 60) (h2 : original_bet = 40):
  total_payout - original_bet = (1 / 2) * original_bet :=
by
  sorry

end blackjack_payout_ratio_l220_220958


namespace find_value_of_x2_div_y2_l220_220561

theorem find_value_of_x2_div_y2 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : x ≠ z)
    (h7 : (y^2 / (x^2 - z^2) = (x^2 + y^2) / z^2))
    (h8 : (x^2 + y^2) / z^2 = x^2 / y^2) : x^2 / y^2 = 2 := by
  sorry

end find_value_of_x2_div_y2_l220_220561


namespace part_A_part_B_part_D_l220_220553

variables (c d : ℤ)

def multiple_of_5 (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k
def multiple_of_10 (x : ℤ) : Prop := ∃ k : ℤ, x = 10 * k

-- Given conditions
axiom h1 : multiple_of_5 c
axiom h2 : multiple_of_10 d

-- Problems to prove
theorem part_A : multiple_of_5 d := by sorry
theorem part_B : multiple_of_5 (c - d) := by sorry
theorem part_D : multiple_of_5 (c + d) := by sorry

end part_A_part_B_part_D_l220_220553


namespace frank_change_l220_220252

theorem frank_change (n_c n_b money_given c_c c_b : ℕ) 
  (h1 : n_c = 5) 
  (h2 : n_b = 2) 
  (h3 : money_given = 20) 
  (h4 : c_c = 2) 
  (h5 : c_b = 3) : 
  money_given - (n_c * c_c + n_b * c_b) = 4 := 
by
  sorry

end frank_change_l220_220252


namespace sum_of_intercepts_modulo_13_l220_220866

theorem sum_of_intercepts_modulo_13 :
  ∃ (x0 y0 : ℤ), 0 ≤ x0 ∧ x0 < 13 ∧ 0 ≤ y0 ∧ y0 < 13 ∧
    (4 * x0 ≡ 1 [ZMOD 13]) ∧ (3 * y0 ≡ 12 [ZMOD 13]) ∧ (x0 + y0 = 14) := 
sorry

end sum_of_intercepts_modulo_13_l220_220866


namespace find_expression_l220_220595

-- Definitions based on the conditions provided
def prop_rel (y x : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 2)

def prop_value_k (k : ℝ) : Prop :=
  k = -4

def prop_value_y (y x : ℝ) : Prop :=
  y = -4 * x + 8

theorem find_expression (y x k : ℝ) : 
  (prop_rel y x k) → 
  (x = 3) → 
  (y = -4) → 
  (prop_value_k k) → 
  (prop_value_y y x) :=
by
  intros h1 h2 h3 h4
  subst h4
  subst h3
  subst h2
  sorry

end find_expression_l220_220595


namespace polynomial_value_at_minus_two_l220_220200

def f (x : ℝ) : ℝ := x^5 + 4*x^4 + x^2 + 20*x + 16

theorem polynomial_value_at_minus_two : f (-2) = 12 := by 
  sorry

end polynomial_value_at_minus_two_l220_220200


namespace min_value_of_expression_l220_220993

theorem min_value_of_expression {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : (a + b) * (a + c) = 4) : 2 * a + b + c ≥ 4 :=
sorry

end min_value_of_expression_l220_220993


namespace total_supermarkets_FGH_chain_l220_220320

variable (US_supermarkets : ℕ) (Canada_supermarkets : ℕ)
variable (total_supermarkets : ℕ)

-- Conditions
def condition1 := US_supermarkets = 37
def condition2 := US_supermarkets = Canada_supermarkets + 14

-- Goal
theorem total_supermarkets_FGH_chain
    (h1 : condition1 US_supermarkets)
    (h2 : condition2 US_supermarkets Canada_supermarkets) :
    total_supermarkets = US_supermarkets + Canada_supermarkets :=
sorry

end total_supermarkets_FGH_chain_l220_220320


namespace axis_of_symmetry_l220_220923

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = f (4 - x)) : ∀ y, f 2 = y ↔ f 2 = y := 
by
  sorry

end axis_of_symmetry_l220_220923


namespace pyramid_base_side_length_l220_220048

theorem pyramid_base_side_length
  (area : ℝ)
  (slant_height : ℝ)
  (h : area = 90)
  (sh : slant_height = 15) :
  ∃ (s : ℝ), 90 = 1 / 2 * s * 15 ∧ s = 12 :=
by
  sorry

end pyramid_base_side_length_l220_220048


namespace blueprint_conversion_proof_l220_220693

-- Let inch_to_feet be the conversion factor from blueprint inches to actual feet.
def inch_to_feet : ℝ := 500

-- Let line_segment_inch be the length of the line segment on the blueprint in inches.
def line_segment_inch : ℝ := 6.5

-- Then, line_segment_feet is the actual length of the line segment in feet.
def line_segment_feet : ℝ := line_segment_inch * inch_to_feet

-- Theorem statement to prove
theorem blueprint_conversion_proof : line_segment_feet = 3250 := by
  -- Proof goes here
  sorry

end blueprint_conversion_proof_l220_220693


namespace perpendicular_vectors_m_eq_0_or_neg2_l220_220543

theorem perpendicular_vectors_m_eq_0_or_neg2
  (m : ℝ)
  (a : ℝ × ℝ := (m, 1))
  (b : ℝ × ℝ := (1, m - 1))
  (h : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) :
  m = 0 ∨ m = -2 := sorry

end perpendicular_vectors_m_eq_0_or_neg2_l220_220543


namespace geometric_sequence_sum_l220_220464

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

noncomputable def sum_geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum 
  (a₁ : ℝ) (q : ℝ) 
  (h_q : q = 1 / 2) 
  (h_a₂ : geometric_sequence a₁ q 2 = 2) : 
  sum_geometric_sequence a₁ q 6 = 63 / 8 :=
by
  -- The proof is skipped here
  sorry

end geometric_sequence_sum_l220_220464


namespace joan_bought_72_eggs_l220_220644

def dozen := 12
def joan_eggs (dozens: Nat) := dozens * dozen

theorem joan_bought_72_eggs : joan_eggs 6 = 72 :=
by
  sorry

end joan_bought_72_eggs_l220_220644


namespace apples_per_basket_holds_15_l220_220766

-- Conditions as Definitions
def trees := 10
def total_apples := 3000
def baskets_per_tree := 20

-- Definition for apples per tree (from the given total apples and number of trees)
def apples_per_tree : ℕ := total_apples / trees

-- Definition for apples per basket (from apples per tree and baskets per tree)
def apples_per_basket : ℕ := apples_per_tree / baskets_per_tree

-- The statement to prove the equivalent mathematical problem
theorem apples_per_basket_holds_15 
  (H1 : trees = 10)
  (H2 : total_apples = 3000)
  (H3 : baskets_per_tree = 20) :
  apples_per_basket = 15 :=
by 
  sorry

end apples_per_basket_holds_15_l220_220766


namespace number_of_solutions_abs_eq_l220_220314

theorem number_of_solutions_abs_eq (f : ℝ → ℝ) (g : ℝ → ℝ) : 
  (∀ x : ℝ, f x = |3 * x| ∧ g x = |x - 2| ∧ (f x + g x = 4) → 
  ∃! x1 x2 : ℝ, 
    ((0 < x1 ∧ x1 < 2 ∧ f x1 + g x1 = 4 ) ∨ 
    (x2 < 0 ∧ f x2 + g x2 = 4) ∧ x1 ≠ x2)) :=
by
  sorry

end number_of_solutions_abs_eq_l220_220314


namespace min_value_f_a_neg3_max_value_g_ge_7_l220_220277

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x) * (x^2 + a * x + 1)

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := 2 * x^3 + 3 * (b + 1) * x^2 + 6 * b * x + 6

theorem min_value_f_a_neg3 (h : -3 ≤ -1) : 
  (∀ x : ℝ, f x (-3) ≥ -Real.exp 2) := 
sorry

theorem max_value_g_ge_7 (a : ℝ) (h : a ≤ -1) (b : ℝ) (h_b : b = a + 1) :
  ∃ m : ℝ, (∀ x : ℝ, g x b ≤ m) ∧ (m ≥ 7) := 
sorry

end min_value_f_a_neg3_max_value_g_ge_7_l220_220277


namespace smallest_sum_of_three_integers_l220_220713

theorem smallest_sum_of_three_integers (a b c : ℕ) (h1: a ≠ b) (h2: b ≠ c) (h3: a ≠ c) (h4: a * b * c = 72) :
  a + b + c = 13 :=
sorry

end smallest_sum_of_three_integers_l220_220713


namespace find_triples_l220_220151

def is_solution (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

theorem find_triples :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ is_solution a b c :=
sorry

end find_triples_l220_220151


namespace katrina_cookies_left_l220_220630

def initial_cookies : ℕ := 120
def morning_sales : ℕ := 3 * 12
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16
def total_sales : ℕ := morning_sales + lunch_sales + afternoon_sales
def cookies_left_to_take_home (initial: ℕ) (sold: ℕ) : ℕ := initial - sold

theorem katrina_cookies_left :
  cookies_left_to_take_home initial_cookies total_sales = 11 :=
by sorry

end katrina_cookies_left_l220_220630


namespace arbitrary_large_sum_of_digits_l220_220891

noncomputable def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem arbitrary_large_sum_of_digits (a : Nat) (h1 : 2 ≤ a) (h2 : ¬ (2 ∣ a)) (h3 : ¬ (5 ∣ a)) :
  ∃ m : Nat, sum_of_digits (a^m) > m :=
by
  sorry

end arbitrary_large_sum_of_digits_l220_220891


namespace fifteen_times_number_eq_150_l220_220830

theorem fifteen_times_number_eq_150 (n : ℕ) (h : 15 * n = 150) : n = 10 :=
sorry

end fifteen_times_number_eq_150_l220_220830


namespace calculate_difference_l220_220091

variable (σ : ℝ) -- Let \square be represented by a real number σ
def correct_answer := 4 * (σ - 3)
def incorrect_answer := 4 * σ - 3
def difference := correct_answer σ - incorrect_answer σ

theorem calculate_difference : difference σ = -9 := by
  sorry

end calculate_difference_l220_220091


namespace red_yellow_flowers_l220_220171

theorem red_yellow_flowers
  (total : ℕ)
  (yellow_white : ℕ)
  (red_white : ℕ)
  (extra_red_over_white : ℕ)
  (H1 : total = 44)
  (H2 : yellow_white = 13)
  (H3 : red_white = 14)
  (H4 : extra_red_over_white = 4) :
  ∃ (red_yellow : ℕ), red_yellow = 17 := by
  sorry

end red_yellow_flowers_l220_220171


namespace arithmetic_sequence_abs_sum_l220_220146

theorem arithmetic_sequence_abs_sum :
  ∀ (a : ℕ → ℤ), (∀ n, a (n + 1) - a n = 2) → a 1 = -5 → 
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 18) :=
by
  sorry

end arithmetic_sequence_abs_sum_l220_220146


namespace compute_expression_l220_220139

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l220_220139


namespace steve_total_time_on_roads_l220_220717

variables (d : ℝ) (v_back : ℝ) (v_to_work : ℝ)

-- Constants from the problem statement
def distance := 10 -- The distance from Steve's house to work is 10 km
def speed_back := 5 -- Steve's speed on the way back from work is 5 km/h

-- Given conditions
def speed_to_work := speed_back / 2 -- On the way back, Steve drives twice as fast as he did on the way to work

-- Define the time to get to work and back
def time_to_work := distance / speed_to_work
def time_back_home := distance / speed_back

-- Total time on roads
def total_time := time_to_work + time_back_home

-- The theorem to prove
theorem steve_total_time_on_roads : total_time = 6 := by
  -- Proof here
  sorry

end steve_total_time_on_roads_l220_220717


namespace relay_race_total_time_l220_220425

theorem relay_race_total_time :
  let athlete1 := 55
  let athlete2 := athlete1 + 10
  let athlete3 := athlete2 - 15
  let athlete4 := athlete1 - 25
  athlete1 + athlete2 + athlete3 + athlete4 = 200 :=
by
  let athlete1 := 55
  let athlete2 := athlete1 + 10
  let athlete3 := athlete2 - 15
  let athlete4 := athlete1 - 25
  show athlete1 + athlete2 + athlete3 + athlete4 = 200
  sorry

end relay_race_total_time_l220_220425


namespace ball_bounce_height_l220_220119

theorem ball_bounce_height :
  ∃ k : ℕ, (500 * (2 / 3:ℝ)^k < 10) ∧ (∀ m : ℕ, m < k → ¬(500 * (2 / 3:ℝ)^m < 10)) :=
sorry

end ball_bounce_height_l220_220119


namespace g_neither_even_nor_odd_l220_220193

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 2)) + 1

theorem g_neither_even_nor_odd : ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g x = -g (-x)) := 
by sorry

end g_neither_even_nor_odd_l220_220193


namespace sum_of_series_l220_220185

noncomputable def infinite_series_sum : ℚ :=
∑' n : ℕ, (3 * (n + 1) - 2) / (((n + 1) : ℚ) * ((n + 1) + 1) * ((n + 1) + 3))

theorem sum_of_series : infinite_series_sum = 11 / 24 := by
  sorry

end sum_of_series_l220_220185


namespace divide_triangle_in_half_l220_220502

def triangle_vertices : Prop :=
  let A := (0, 2)
  let B := (0, 0)
  let C := (10, 0)
  let base := 10
  let height := 2
  let total_area := (1 / 2) * base * height

  ∀ (a : ℝ),
  (1 / 2) * a * height = total_area / 2 → a = 5

theorem divide_triangle_in_half : triangle_vertices := 
  sorry

end divide_triangle_in_half_l220_220502


namespace pirate_total_dollar_amount_l220_220774

def base_5_to_base_10 (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.reverse.enum.map (λ ⟨p, d⟩ => d * base^p) |>.sum

def jewelry_base5 := [3, 1, 2, 4]
def gold_coins_base5 := [3, 1, 2, 2]
def alcohol_base5 := [1, 2, 4]

def jewelry_base10 := base_5_to_base_10 jewelry_base5 5
def gold_coins_base10 := base_5_to_base_10 gold_coins_base5 5
def alcohol_base10 := base_5_to_base_10 alcohol_base5 5

def total_base10 := jewelry_base10 + gold_coins_base10 + alcohol_base10

theorem pirate_total_dollar_amount :
  total_base10 = 865 :=
by
  unfold total_base10 jewelry_base10 gold_coins_base10 alcohol_base10 base_5_to_base_10
  simp
  sorry

end pirate_total_dollar_amount_l220_220774


namespace hyperbola_transverse_axis_l220_220864

noncomputable def hyperbola_transverse_axis_length (a b : ℝ) : ℝ :=
  2 * a

theorem hyperbola_transverse_axis {a b : ℝ} (h : a > 0) (h_b : b > 0) 
  (eccentricity_cond : Real.sqrt 2 = Real.sqrt (1 + b^2 / a^2))
  (area_cond : ∃ x y : ℝ, x^2 = -4 * Real.sqrt 3 * y ∧ y * y / a^2 - x^2 / b^2 = 1 ∧ 
                 Real.sqrt 3 = 1 / 2 * (2 * Real.sqrt (3 - a^2)) * Real.sqrt 3) :
  hyperbola_transverse_axis_length a b = 2 * Real.sqrt 2 :=
by
  sorry

end hyperbola_transverse_axis_l220_220864


namespace burn_down_village_in_1920_seconds_l220_220405

-- Definitions of the initial conditions
def initial_cottages : Nat := 90
def burn_interval_seconds : Nat := 480
def burn_time_per_unit : Nat := 5
def max_burns_per_interval : Nat := burn_interval_seconds / burn_time_per_unit

-- Recurrence relation for the number of cottages after n intervals
def cottages_remaining (n : Nat) : Nat :=
if n = 0 then initial_cottages
else 2 * cottages_remaining (n - 1) - max_burns_per_interval

-- Time taken to burn all cottages is when cottages_remaining(n) becomes 0
def total_burn_time_seconds (intervals : Nat) : Nat :=
intervals * burn_interval_seconds

-- Main theorem statement
theorem burn_down_village_in_1920_seconds :
  ∃ n, cottages_remaining n = 0 ∧ total_burn_time_seconds n = 1920 := by
  sorry

end burn_down_village_in_1920_seconds_l220_220405


namespace correct_inequality_l220_220363

variable (a b c d : ℝ)
variable (h₁ : a > b)
variable (h₂ : b > 0)
variable (h₃ : 0 > c)
variable (h₄ : c > d)

theorem correct_inequality :
  (c / a) - (d / b) > 0 :=
by sorry

end correct_inequality_l220_220363


namespace find_expression_for_a_n_l220_220501

-- Definitions for conditions in the problem
variable (a : ℕ → ℝ) -- Sequence is of positive real numbers
variable (S : ℕ → ℝ) -- Sum of the first n terms of the sequence

-- Condition that all terms in the sequence a_n are positive and indexed by natural numbers starting from 1
axiom pos_seq : ∀ n : ℕ, 0 < a (n + 1)
-- Condition for the sum of the terms: 4S_n = a_n^2 + 2a_n for n ∈ ℕ*
axiom sum_condition : ∀ n : ℕ, 4 * S (n + 1) = (a (n + 1))^2 + 2 * a (n + 1)

-- Theorem stating that sequence a_n = 2n given the above conditions
theorem find_expression_for_a_n : ∀ n : ℕ, a (n + 1) = 2 * (n + 1) := by
  sorry

end find_expression_for_a_n_l220_220501


namespace big_SUV_wash_ratio_l220_220359

-- Defining constants for time taken for various parts of the car
def time_windows : ℕ := 4
def time_body : ℕ := 7
def time_tires : ℕ := 4
def time_waxing : ℕ := 9

-- Time taken to wash one normal car
def time_normal_car : ℕ := time_windows + time_body + time_tires + time_waxing

-- Given total time William spent washing all vehicles
def total_time : ℕ := 96

-- Time taken for two normal cars
def time_two_normal_cars : ℕ := 2 * time_normal_car

-- Time taken for the big SUV
def time_big_SUV : ℕ := total_time - time_two_normal_cars

-- Ratio of time taken to wash the big SUV to the time taken to wash a normal car
def time_ratio : ℕ := time_big_SUV / time_normal_car

theorem big_SUV_wash_ratio : time_ratio = 2 := by
  sorry

end big_SUV_wash_ratio_l220_220359


namespace intersection_M_N_l220_220802

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | |x| > 1}

theorem intersection_M_N : M ∩ N = {2} := by
  sorry

end intersection_M_N_l220_220802


namespace range_of_m_for_second_quadrant_l220_220969

theorem range_of_m_for_second_quadrant (m : ℝ) :
  (P : ℝ × ℝ) → P = (1 + m, 3) → P.fst < 0 → m < -1 :=
by
  intro P hP hQ
  sorry

end range_of_m_for_second_quadrant_l220_220969


namespace coat_price_calculation_l220_220564

noncomputable def effective_price (initial_price : ℝ) 
  (reduction1 reduction2 reduction3 : ℝ) 
  (tax1 tax2 tax3 : ℝ) : ℝ :=
  let price_after_first_month := initial_price * (1 - reduction1 / 100) * (1 + tax1 / 100)
  let price_after_second_month := price_after_first_month * (1 - reduction2 / 100) * (1 + tax2 / 100)
  let price_after_third_month := price_after_second_month * (1 - reduction3 / 100) * (1 + tax3 / 100)
  price_after_third_month

noncomputable def total_percent_reduction (initial_price final_price : ℝ) : ℝ :=
  (initial_price - final_price) / initial_price * 100

theorem coat_price_calculation :
  let original_price := 500
  let price_final := effective_price original_price 10 15 20 5 8 6
  let reduction_percentage := total_percent_reduction original_price price_final
  price_final = 367.824 ∧ reduction_percentage = 26.44 :=
by
  sorry

end coat_price_calculation_l220_220564


namespace cuboid_edge_lengths_l220_220036

theorem cuboid_edge_lengths (
  a b c : ℕ
) (h_volume : a * b * c + a * b + b * c + c * a + a + b + c = 2000) :
  (a = 28 ∧ b = 22 ∧ c = 2) ∨ 
  (a = 28 ∧ b = 2 ∧ c = 22) ∨
  (a = 22 ∧ b = 28 ∧ c = 2) ∨
  (a = 22 ∧ b = 2 ∧ c = 28) ∨
  (a = 2 ∧ b = 28 ∧ c = 22) ∨
  (a = 2 ∧ b = 22 ∧ c = 28) :=
sorry

end cuboid_edge_lengths_l220_220036


namespace negative_expression_l220_220352

noncomputable def U : ℝ := -2.5
noncomputable def V : ℝ := -0.8
noncomputable def W : ℝ := 0.4
noncomputable def X : ℝ := 1.0
noncomputable def Y : ℝ := 2.2

theorem negative_expression :
  (U - V < 0) ∧ ¬(U * V < 0) ∧ ¬((X / V) * U < 0) ∧ ¬(W / (U * V) < 0) ∧ ¬((X + Y) / W < 0) :=
by
  sorry

end negative_expression_l220_220352


namespace hyperbola_asymptote_slopes_l220_220931

theorem hyperbola_asymptote_slopes :
  ∀ (x y : ℝ), 2 * (y^2 / 16) - 2 * (x^2 / 9) = 1 → (∃ m : ℝ, y = m * x ∨ y = -m * x) ∧ m = (Real.sqrt 80) / 3 :=
by
  sorry

end hyperbola_asymptote_slopes_l220_220931


namespace sum_of_distinct_squares_l220_220787

theorem sum_of_distinct_squares:
  ∀ (a b c : ℕ),
  a + b + c = 23 ∧ Nat.gcd a b + Nat.gcd b c + Nat.gcd c a = 9 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 + c^2 = 179 ∨ a^2 + b^2 + c^2 = 259 →
  a^2 + b^2 + c^2 = 438 :=
by
  sorry

end sum_of_distinct_squares_l220_220787


namespace correct_calculation_l220_220600

theorem correct_calculation (x : ℤ) (h : 7 * (x + 24) / 5 = 70) :
  (5 * x + 24) / 7 = 22 :=
sorry

end correct_calculation_l220_220600


namespace candy_bar_cost_l220_220023

-- Definitions of conditions
def soft_drink_cost : ℕ := 4
def num_soft_drinks : ℕ := 2
def num_candy_bars : ℕ := 5
def total_cost : ℕ := 28

-- Proof Statement
theorem candy_bar_cost : (total_cost - num_soft_drinks * soft_drink_cost) / num_candy_bars = 4 := by
  sorry

end candy_bar_cost_l220_220023


namespace largest_constant_D_l220_220364

theorem largest_constant_D (D : ℝ) 
  (h : ∀ (x y : ℝ), x^2 + y^2 + 4 ≥ D * (x + y)) : 
  D ≤ 2 * Real.sqrt 2 :=
sorry

end largest_constant_D_l220_220364


namespace find_a_l220_220122

theorem find_a (a : ℝ) (x₁ x₂ : ℝ) (h1 : a > 0) (h2 : x₁ = -2 * a) (h3 : x₂ = 4 * a) (h4 : x₂ - x₁ = 15) : a = 5 / 2 :=
by 
  sorry

end find_a_l220_220122


namespace total_people_in_boats_l220_220493

theorem total_people_in_boats (bo_num : ℝ) (avg_people : ℝ) (bo_num_eq : bo_num = 3.0) (avg_people_eq : avg_people = 1.66666666699999) : ∃ total_people : ℕ, total_people = 6 := 
by
  sorry

end total_people_in_boats_l220_220493


namespace percentage_increase_l220_220204

theorem percentage_increase (x : ℝ) : 
  (1 + x / 100)^2 = 1.1025 → x = 5.024 := 
sorry

end percentage_increase_l220_220204


namespace find_angle_sum_l220_220256

theorem find_angle_sum (c d : ℝ) (hc : 0 < c ∧ c < π/2) (hd : 0 < d ∧ d < π/2)
    (h1 : 4 * (Real.cos c)^2 + 3 * (Real.sin d)^2 = 1)
    (h2 : 4 * Real.sin (2 * c) = 3 * Real.cos (2 * d)) :
    2 * c + 3 * d = π / 2 :=
by
  sorry

end find_angle_sum_l220_220256


namespace virus_affected_computers_l220_220296

theorem virus_affected_computers (m n : ℕ) (h1 : 5 * m + 2 * n = 52) : m = 8 :=
by
  sorry

end virus_affected_computers_l220_220296


namespace sin_sides_of_triangle_l220_220346

theorem sin_sides_of_triangle {a b c : ℝ} 
  (habc: a + b > c) (hbac: a + c > b) (hcbc: b + c > a) (h_sum: a + b + c ≤ 2 * Real.pi) :
  a > 0 ∧ a < Real.pi ∧ b > 0 ∧ b < Real.pi ∧ c > 0 ∧ c < Real.pi ∧ 
  (Real.sin a + Real.sin b > Real.sin c) ∧ 
  (Real.sin a + Real.sin c > Real.sin b) ∧ 
  (Real.sin b + Real.sin c > Real.sin a) :=
by
  sorry

end sin_sides_of_triangle_l220_220346


namespace jogging_path_diameter_l220_220633

theorem jogging_path_diameter 
  (d_pond : ℝ)
  (w_flowerbed : ℝ)
  (w_jogging_path : ℝ)
  (h_pond : d_pond = 20)
  (h_flowerbed : w_flowerbed = 10)
  (h_jogging_path : w_jogging_path = 12) :
  2 * (d_pond / 2 + w_flowerbed + w_jogging_path) = 64 :=
by
  sorry

end jogging_path_diameter_l220_220633


namespace work_completion_days_l220_220884

theorem work_completion_days (A B C : ℝ) (h1 : A + B + C = 1/4) (h2 : B = 1/18) (h3 : C = 1/6) : A = 1/36 :=
by
  sorry

end work_completion_days_l220_220884


namespace batsman_average_after_12th_innings_l220_220061

-- Defining the conditions
def before_12th_innings_average (A : ℕ) : Prop :=
11 * A + 80 = 12 * (A + 2)

-- Defining the question and expected answer
def after_12th_innings_average : ℕ := 58

-- Proving the equivalence
theorem batsman_average_after_12th_innings (A : ℕ) (h : before_12th_innings_average A) : after_12th_innings_average = 58 :=
by
sorry

end batsman_average_after_12th_innings_l220_220061


namespace simplify_trig_l220_220831

open Real

theorem simplify_trig : 
  (sin (30 * pi / 180) + sin (60 * pi / 180)) / (cos (30 * pi / 180) + cos (60 * pi / 180)) = tan (45 * pi / 180) :=
by
  sorry

end simplify_trig_l220_220831


namespace sin_45_eq_1_div_sqrt_2_l220_220517

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l220_220517


namespace constructible_angles_l220_220714

def is_constructible (θ : ℝ) : Prop :=
  -- Define that θ is constructible if it can be constructed using compass and straightedge.
  sorry

theorem constructible_angles (α : ℝ) (β : ℝ) (k n : ℤ) (hβ : is_constructible β) :
  is_constructible (k * α / 2^n + β) :=
sorry

end constructible_angles_l220_220714


namespace find_value_of_a_l220_220867

variable (a b : ℝ)

def varies_inversely (a : ℝ) (b_minus_one_sq : ℝ) : ℝ :=
  a * b_minus_one_sq

theorem find_value_of_a 
  (h₁ : ∀ b : ℝ, varies_inversely a ((b - 1) ^ 2) = 64)
  (h₂ : b = 5) : a = 4 :=
by sorry

end find_value_of_a_l220_220867


namespace largest_odd_not_sum_of_three_distinct_composites_l220_220670

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem largest_odd_not_sum_of_three_distinct_composites :
  ∀ n : ℕ, is_odd n → (¬ ∃ (a b c : ℕ), is_composite a ∧ is_composite b ∧ is_composite c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a + b + c) → n ≤ 17 :=
by
  sorry

end largest_odd_not_sum_of_three_distinct_composites_l220_220670


namespace one_way_ticket_cost_l220_220887

theorem one_way_ticket_cost (x : ℝ) (h : 50 / 26 < x) : x >= 2 :=
by sorry

end one_way_ticket_cost_l220_220887


namespace moles_of_HC2H3O2_needed_l220_220810

theorem moles_of_HC2H3O2_needed :
  (∀ (HC2H3O2 NaHCO3 H2O : ℕ), 
    (HC2H3O2 + NaHCO3 = NaC2H3O2 + H2O + CO2) → 
    (H2O = 3) → 
    (NaHCO3 = 3) → 
    HC2H3O2 = 3) :=
by
  intros HC2H3O2 NaHCO3 H2O h_eq h_H2O h_NaHCO3
  -- Hint: You can use the balanced chemical equation to derive that HC2H3O2 must be 3
  sorry

end moles_of_HC2H3O2_needed_l220_220810


namespace arithmetic_seq_a2_l220_220439

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n m : ℕ, a m = a (n + 1) + d * (m - (n + 1))

theorem arithmetic_seq_a2 
  (a : ℕ → ℤ) (d a1 : ℤ)
  (h_arith: ∀ n : ℕ, a n = a1 + n * d)
  (h_sum: a 3 + a 11 = 50)
  (h_a4: a 4 = 13) :
  a 2 = 5 :=
sorry

end arithmetic_seq_a2_l220_220439


namespace calculate_polynomial_value_l220_220095

theorem calculate_polynomial_value (a a1 a2 a3 a4 a5 : ℝ) : 
  (∀ x : ℝ, (1 - x)^2 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) → 
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := 
by 
  intro h
  sorry

end calculate_polynomial_value_l220_220095


namespace system_of_equations_correct_l220_220538

-- Define the problem conditions
variable (x y : ℝ) -- Define the productivity of large and small harvesters

-- Define the correct system of equations as per the problem
def system_correct : Prop := (2 * (2 * x + 5 * y) = 3.6) ∧ (5 * (3 * x + 2 * y) = 8)

-- State the theorem to prove the correctness of the system of equations under given conditions
theorem system_of_equations_correct (x y : ℝ) : (2 * (2 * x + 5 * y) = 3.6) ∧ (5 * (3 * x + 2 * y) = 8) :=
by
  sorry

end system_of_equations_correct_l220_220538


namespace neg_or_false_of_or_true_l220_220870

variable {p q : Prop}

theorem neg_or_false_of_or_true (h : ¬ (p ∨ q) = false) : p ∨ q :=
by {
  sorry
}

end neg_or_false_of_or_true_l220_220870


namespace modulus_of_z_l220_220294

open Complex

theorem modulus_of_z (z : ℂ) (h : z * ⟨0, 1⟩ = ⟨2, 1⟩) : abs z = Real.sqrt 5 :=
by
  sorry

end modulus_of_z_l220_220294


namespace fixed_point_line_passes_through_range_of_t_l220_220324

-- Definition for first condition: Line with slope k (k ≠ 0)
variables {k : ℝ} (hk : k ≠ 0)

-- Definition for second condition: Ellipse C
def ellipse_C (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Third condition: Intersections M and N
variables (M N : ℝ × ℝ)
variables (intersection_M : ellipse_C M.1 M.2)
variables (intersection_N : ellipse_C N.1 N.2)

-- Fourth condition: Slopes are k1 and k2
variables {k1 k2 : ℝ}
variables (hk1 : k1 = M.2 / M.1)
variables (hk2 : k2 = N.2 / N.1)

-- Fifth condition: Given equation 3(k1 + k2) = 8k
variables (h_eq : 3 * (k1 + k2) = 8 * k)

-- Proof for question 1: Line passes through a fixed point
theorem fixed_point_line_passes_through 
    (h_eq : 3 * (k1 + k2) = 8 * k) : 
    ∃ n : ℝ, n = 1/2 ∨ n = -1/2 := sorry

-- Additional conditions for question 2
variables {D : ℝ × ℝ} (hD : D = (1, 0))
variables (t : ℝ)
variables (area_ratio : (M.2 / N.2) = t)
variables (h_ineq : k^2 < 5 / 12)

-- Proof for question 2: Range for t
theorem range_of_t
    (hD : D = (1, 0))
    (area_ratio : (M.2 / N.2) = t)
    (h_ineq : k^2 < 5 / 12) : 
    2 < t ∧ t < 3 ∨ 1 / 3 < t ∧ t < 1 / 2 := sorry

end fixed_point_line_passes_through_range_of_t_l220_220324


namespace impossible_to_place_50_pieces_on_torus_grid_l220_220621

theorem impossible_to_place_50_pieces_on_torus_grid :
  ¬ (∃ (a b c x y z : ℕ),
    a + b + c = 50 ∧
    2 * a ≤ x ∧ x ≤ 2 * b ∧
    2 * b ≤ y ∧ y ≤ 2 * c ∧
    2 * c ≤ z ∧ z ≤ 2 * a) :=
by
  sorry

end impossible_to_place_50_pieces_on_torus_grid_l220_220621


namespace problem_1_problem_2_problem_3_l220_220862

-- Problem 1
theorem problem_1 (m n : ℝ) : 
  3 * (m - n) ^ 2 - 4 * (m - n) ^ 2 + 3 * (m - n) ^ 2 = 2 * (m - n) ^ 2 := 
by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h : x^2 + 2 * y = 4) : 
  3 * x^2 + 6 * y - 2 = 10 := 
by
  sorry

-- Problem 3
theorem problem_3 (x y : ℝ) 
  (h1 : x^2 + x * y = 2) 
  (h2 : 2 * y^2 + 3 * x * y = 5) : 
  2 * x^2 + 11 * x * y + 6 * y^2 = 19 := 
by
  sorry

end problem_1_problem_2_problem_3_l220_220862


namespace kevin_bucket_size_l220_220796

def rate_of_leakage (r : ℝ) : Prop := r = 1.5
def time_away (t : ℝ) : Prop := t = 12
def bucket_size (b : ℝ) (r t : ℝ) : Prop := b = 2 * r * t

theorem kevin_bucket_size
  (r t b : ℝ)
  (H1 : rate_of_leakage r)
  (H2 : time_away t) :
  bucket_size b r t :=
by
  simp [rate_of_leakage, time_away, bucket_size] at *
  sorry

end kevin_bucket_size_l220_220796


namespace initial_avg_production_is_50_l220_220980

-- Define the initial conditions and parameters
variables (A : ℝ) (n : ℕ := 10) (today_prod : ℝ := 105) (new_avg : ℝ := 55)

-- State that the initial total production over n days
def initial_total_production (A : ℝ) (n : ℕ) : ℝ := A * n

-- State the total production after today's production is added
def post_total_production (A : ℝ) (n : ℕ) (today_prod : ℝ) : ℝ := initial_total_production A n + today_prod

-- State the new average production calculation
def new_avg_production (n : ℕ) (new_avg : ℝ) : ℝ := new_avg * (n + 1)

-- State the main claim: Prove that the initial average daily production was 50 units per day
theorem initial_avg_production_is_50 (A : ℝ) (n : ℕ := 10) (today_prod : ℝ := 105) (new_avg : ℝ := 55) 
  (h : post_total_production A n today_prod = new_avg_production n new_avg) : 
  A = 50 := 
by {
  -- Preliminary setups (we don't need detailed proof steps here)
  sorry
}

end initial_avg_production_is_50_l220_220980


namespace quadratic_real_roots_range_l220_220641

theorem quadratic_real_roots_range (m : ℝ) : (∃ x : ℝ, x^2 - 2 * x - m = 0) → -1 ≤ m := 
sorry

end quadratic_real_roots_range_l220_220641


namespace solution_to_quadratic_inequality_l220_220667

theorem solution_to_quadratic_inequality 
  (a : ℝ)
  (h : ∀ x : ℝ, x^2 - a * x + 1 < 0 ↔ (1 / 2 : ℝ) < x ∧ x < 2) :
  a = 5 / 2 :=
sorry

end solution_to_quadratic_inequality_l220_220667


namespace equilateral_triangle_path_l220_220874

noncomputable def equilateral_triangle_path_length (side_length_triangle side_length_square : ℝ) : ℝ :=
  let radius := side_length_triangle
  let rotational_path_length := 4 * 3 * 2 * Real.pi
  let diagonal_length := (Real.sqrt (side_length_square^2 + side_length_square^2))
  let linear_path_length := 2 * diagonal_length
  rotational_path_length + linear_path_length

theorem equilateral_triangle_path (side_length_triangle side_length_square : ℝ) 
  (h_triangle : side_length_triangle = 3) (h_square : side_length_square = 6) :
  equilateral_triangle_path_length side_length_triangle side_length_square = 24 * Real.pi + 12 * Real.sqrt 2 :=
by
  rw [h_triangle, h_square]
  unfold equilateral_triangle_path_length
  sorry

end equilateral_triangle_path_l220_220874


namespace matrix_determinant_eq_9_l220_220228

theorem matrix_determinant_eq_9 (x : ℝ) :
  let a := x - 1
  let b := 2
  let c := 3
  let d := -5
  (a * d - b * c = 9) → x = -2 :=
by 
  let a := x - 1
  let b := 2
  let c := 3
  let d := -5
  sorry

end matrix_determinant_eq_9_l220_220228


namespace range_of_a_l220_220953

-- Problem statement and conditions definition
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def Q (a : ℝ) : Prop := (5 - 2 * a) > 1

-- Proof problem statement
theorem range_of_a (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ -2 :=
sorry

end range_of_a_l220_220953


namespace line_does_not_pass_through_third_quadrant_l220_220112

-- Define the Cartesian equation of the line
def line_eq (x y : ℝ) : Prop :=
  x + 2 * y = 1

-- Define the property that a point (x, y) belongs to the third quadrant
def in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- State the theorem
theorem line_does_not_pass_through_third_quadrant :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ in_third_quadrant x y :=
by
  sorry

end line_does_not_pass_through_third_quadrant_l220_220112


namespace find_divisor_l220_220890

open Nat

theorem find_divisor 
  (d n : ℕ)
  (h1 : n % d = 3)
  (h2 : 2 * n % d = 2) : 
  d = 4 := 
sorry

end find_divisor_l220_220890


namespace rectangle_perimeter_l220_220051

-- We first define the side lengths of the squares and their relationships
def b1 : ℕ := 3
def b2 : ℕ := 9
def b3 := b1 + b2
def b4 := 2 * b1 + b2
def b5 := 3 * b1 + 2 * b2
def b6 := 3 * b1 + 3 * b2
def b7 := 4 * b1 + 3 * b2

-- Dimensions of the rectangle
def L := 37
def W := 52

-- Theorem to prove the perimeter of the rectangle
theorem rectangle_perimeter : 2 * L + 2 * W = 178 := by
  -- Proof will be provided here
  sorry

end rectangle_perimeter_l220_220051


namespace solve_system_of_equations_in_nat_numbers_l220_220470

theorem solve_system_of_equations_in_nat_numbers :
  ∃ a b c d : ℕ, a * b = c + d ∧ c * d = a + b ∧ a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 :=
by
  sorry

end solve_system_of_equations_in_nat_numbers_l220_220470


namespace sum_of_two_numbers_l220_220414

theorem sum_of_two_numbers :
  (∃ x y : ℕ, y = 2 * x - 43 ∧ y = 31 ∧ x + y = 68) :=
sorry

end sum_of_two_numbers_l220_220414


namespace repeating_decimal_427_diff_l220_220765

theorem repeating_decimal_427_diff :
  let G := 0.427427427427
  let num := 427
  let denom := 999
  num.gcd denom = 1 →
  denom - num = 572 :=
by
  intros G num denom gcd_condition
  sorry

end repeating_decimal_427_diff_l220_220765


namespace initial_crayons_count_l220_220580

theorem initial_crayons_count (C : ℕ) :
  (3 / 8) * C = 18 → C = 48 :=
by
  sorry

end initial_crayons_count_l220_220580


namespace quadratic_roots_range_quadratic_root_condition_l220_220811

-- Problem 1: Prove that the range of real number \(k\) for which the quadratic 
-- equation \(x^{2} + (2k + 1)x + k^{2} + 1 = 0\) has two distinct real roots is \(k > \frac{3}{4}\). 
theorem quadratic_roots_range (k : ℝ) : 
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x^2 + (2*k+1)*x + k^2 + 1 = 0) ↔ (k > 3/4) := 
sorry

-- Problem 2: Given \(k > \frac{3}{4}\), prove that if the roots \(x₁\) and \(x₂\) of 
-- the equation satisfy \( |x₁| + |x₂| = x₁ \cdot x₂ \), then \( k = 2 \).
theorem quadratic_root_condition (k : ℝ) 
    (hk : k > 3 / 4)
    (x₁ x₂ : ℝ)
    (h₁ : x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0)
    (h₂ : x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0)
    (h3 : |x₁| + |x₂| = x₁ * x₂) : 
    k = 2 := 
sorry

end quadratic_roots_range_quadratic_root_condition_l220_220811


namespace ratio_of_incomes_l220_220145

variable {I1 I2 E1 E2 S1 S2 : ℝ}

theorem ratio_of_incomes
  (h1 : I1 = 4000)
  (h2 : E1 / E2 = 3 / 2)
  (h3 : S1 = 1600)
  (h4 : S2 = 1600)
  (h5 : S1 = I1 - E1)
  (h6 : S2 = I2 - E2) :
  I1 / I2 = 5 / 4 :=
by
  sorry

end ratio_of_incomes_l220_220145


namespace x_square_minus_5x_is_necessary_not_sufficient_l220_220432

theorem x_square_minus_5x_is_necessary_not_sufficient (x : ℝ) :
  (x^2 - 5 * x < 0) → (|x - 1| < 1) → (x^2 - 5 * x < 0 ∧ ∃ y : ℝ, (0 < y ∧ y < 2) → x = y) :=
by
  sorry

end x_square_minus_5x_is_necessary_not_sufficient_l220_220432


namespace mike_ride_equals_42_l220_220239

-- Define the costs as per the conditions
def cost_mike (M : ℕ) : ℝ := 2.50 + 0.25 * M
def cost_annie : ℝ := 2.50 + 5.00 + 0.25 * 22

-- State the theorem that needs to be proved
theorem mike_ride_equals_42 : ∃ M : ℕ, cost_mike M = cost_annie ∧ M = 42 :=
by
  sorry

end mike_ride_equals_42_l220_220239


namespace natural_pairs_prime_l220_220022

theorem natural_pairs_prime (x y : ℕ) (p : ℕ) (hp : Nat.Prime p) (h_eq : p = xy^2 / (x + y))
  : (x, y) = (2, 2) ∨ (x, y) = (6, 2) :=
sorry

end natural_pairs_prime_l220_220022


namespace resulting_polygon_sides_l220_220757

/-
Problem statement: 

Construct a regular pentagon on one side of a regular heptagon.
On one non-adjacent side of the pentagon, construct a regular hexagon.
On a non-adjacent side of the hexagon, construct an octagon.
Continue to construct regular polygons in the same way, until you construct a nonagon.
How many sides does the resulting polygon have?

Given facts:
1. Start with a heptagon (7 sides).
2. Construct a pentagon (5 sides) on one side of the heptagon.
3. Construct a hexagon (6 sides) on a non-adjacent side of the pentagon.
4. Construct an octagon (8 sides) on a non-adjacent side of the hexagon.
5. Construct a nonagon (9 sides) on a non-adjacent side of the octagon.
-/

def heptagon_sides : ℕ := 7
def pentagon_sides : ℕ := 5
def hexagon_sides : ℕ := 6
def octagon_sides : ℕ := 8
def nonagon_sides : ℕ := 9

theorem resulting_polygon_sides : 
  (heptagon_sides + nonagon_sides - 2 * 1) + (pentagon_sides + hexagon_sides + octagon_sides - 3 * 2) = 27 := by
  sorry

end resulting_polygon_sides_l220_220757


namespace polynomial_simplification_l220_220646

theorem polynomial_simplification (p : ℤ) :
  (5 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 2) + (-3 * p^4 + 4 * p^3 + 8 * p^2 - 2 * p + 6) = 
  2 * p^4 + 6 * p^3 + p^2 + p + 4 :=
by
  sorry

end polynomial_simplification_l220_220646


namespace y_coordinate_of_second_point_l220_220706

theorem y_coordinate_of_second_point
  (m n : ℝ)
  (h₁ : m = 2 * n + 3)
  (h₂ : m + 2 = 2 * (n + 1) + 3) :
  (n + 1) = n + 1 :=
by
  -- proof to be provided
  sorry

end y_coordinate_of_second_point_l220_220706


namespace year_2022_form_l220_220437

theorem year_2022_form :
  ∃ (a b c d e f g h i j : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10 ∧
    2001 ≤ (a + b * c * d * e) / (f + g * h * i * j) ∧ (a + b * c * d * e) / (f + g * h * i * j) ≤ 2100 ∧
    (a + b * c * d * e) / (f + g * h * i * j) = 2022 :=
sorry

end year_2022_form_l220_220437


namespace largest_s_for_angle_ratio_l220_220921

theorem largest_s_for_angle_ratio (r s : ℕ) (hr : r ≥ 3) (hs : s ≥ 3) (h_angle_ratio : (130 * (r - 2)) * s = (131 * (s - 2)) * r) :
  s ≤ 260 :=
by 
  sorry

end largest_s_for_angle_ratio_l220_220921


namespace length_of_A_l220_220138

structure Point := (x : ℝ) (y : ℝ)

noncomputable def length (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

theorem length_of_A'B' (A A' B B' C : Point) 
    (hA : A = ⟨0, 6⟩)
    (hB : B = ⟨0, 10⟩)
    (hC : C = ⟨3, 6⟩)
    (hA'_line : A'.y = A'.x)
    (hB'_line : B'.y = B'.x) 
    (hA'C : ∃ m b, ((C.y = m * C.x + b) ∧ (C.y = b) ∧ (A.y = b))) 
    (hB'C : ∃ m b, ((C.y = m * C.x + b) ∧ (B.y = m * B.x + b)))
    : length A' B' = (12 / 7) * Real.sqrt 2 :=
by
  sorry

end length_of_A_l220_220138


namespace percent_profit_l220_220065

theorem percent_profit (CP LP SP Profit : ℝ) 
  (hCP : CP = 100) 
  (hLP : LP = CP + 0.30 * CP)
  (hSP : SP = LP - 0.10 * LP) 
  (hProfit : Profit = SP - CP) : 
  (Profit / CP) * 100 = 17 :=
by
  sorry

end percent_profit_l220_220065


namespace downstream_distance_l220_220094

-- Define the speeds and distances as constants or variables
def speed_boat := 30 -- speed in kmph
def speed_stream := 10 -- speed in kmph
def distance_upstream := 40 -- distance in km
def time_upstream := distance_upstream / (speed_boat - speed_stream) -- time in hours

-- Define the variable for the downstream distance
variable {D : ℝ}

-- The Lean 4 statement to prove that the downstream distance is the specified value
theorem downstream_distance : 
  (time_upstream = D / (speed_boat + speed_stream)) → D = 80 :=
by
  sorry

end downstream_distance_l220_220094


namespace shape_of_phi_eq_d_in_spherical_coordinates_l220_220541

theorem shape_of_phi_eq_d_in_spherical_coordinates (d : ℝ) : 
  (∃ (ρ θ : ℝ), ∀ (φ : ℝ), φ = d) ↔ ( ∃ cone_vertex : ℝ × ℝ × ℝ, ∃ opening_angle : ℝ, cone_vertex = (0, 0, 0) ∧ opening_angle = d) :=
sorry

end shape_of_phi_eq_d_in_spherical_coordinates_l220_220541


namespace james_needs_more_marbles_l220_220690

def number_of_additional_marbles (friends marbles : Nat) : Nat :=
  let required_marbles := (friends * (friends + 1)) / 2
  (if marbles < required_marbles then required_marbles - marbles else 0)

theorem james_needs_more_marbles :
  number_of_additional_marbles 15 80 = 40 := by
  sorry

end james_needs_more_marbles_l220_220690


namespace liam_birthday_next_monday_2018_l220_220174

-- Define year advancement rules
def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

-- Define function to calculate next weekday
def next_weekday (current_day : ℕ) (years_elapsed : ℕ) : ℕ :=
  let advance := (years_elapsed / 4) * 2 + (years_elapsed % 4)
  (current_day + advance) % 7

theorem liam_birthday_next_monday_2018 :
  (next_weekday 4 3 = 0) :=
sorry

end liam_birthday_next_monday_2018_l220_220174


namespace minimum_function_value_l220_220435

theorem minimum_function_value :
  ∃ (x y : ℕ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 3 ∧
  (∀ x' y', 0 ≤ x' ∧ x' ≤ 2 → 0 ≤ y' ∧ y' ≤ 3 →
  (x^2 * y^2 : ℝ) / ((x^2 + y^2)^2 : ℝ) ≤ (x'^2 * y'^2 : ℝ) / ((x'^2 + y'^2)^2 : ℝ)) ∧
  (x = 0 ∨ y = 0) ∧ ((x^2 * y^2 : ℝ) / ((x^2 + y^2)^2 : ℝ) = 0) :=
by
  --; Implementation of the theorem would follow
  sorry

end minimum_function_value_l220_220435


namespace fraction_1790s_l220_220052

def total_states : ℕ := 30
def states_1790s : ℕ := 16

theorem fraction_1790s : (states_1790s / total_states : ℚ) = 8 / 15 :=
by
  -- We claim that the fraction of states admitted during the 1790s is exactly 8/15
  sorry

end fraction_1790s_l220_220052


namespace range_independent_variable_l220_220341

theorem range_independent_variable (x : ℝ) (h : x + 1 > 0) : x > -1 :=
sorry

end range_independent_variable_l220_220341


namespace find_k_l220_220688

theorem find_k (m : ℝ) (h : ∃ A B : ℝ, (m^3 - 24*m + 16) = (m^2 - 8*m) * (A*m + B) ∧ A - 8 = -k ∧ -8*B = -24) : k = 5 :=
sorry

end find_k_l220_220688


namespace total_length_of_sticks_l220_220540

theorem total_length_of_sticks :
  ∃ (s1 s2 s3 : ℝ), s1 = 3 ∧ s2 = 2 * s1 ∧ s3 = s2 - 1 ∧ (s1 + s2 + s3 = 14) := by
  sorry

end total_length_of_sticks_l220_220540


namespace net_rate_of_pay_equals_39_dollars_per_hour_l220_220182

-- Definitions of the conditions
def hours_travelled : ℕ := 3
def speed_per_hour : ℕ := 60
def car_consumption_rate : ℕ := 30
def earnings_per_mile : ℕ := 75  -- expressing $0.75 as 75 cents to avoid floating-point
def gasoline_cost_per_gallon : ℕ := 300  -- expressing $3.00 as 300 cents to avoid floating-point

-- Proof statement
theorem net_rate_of_pay_equals_39_dollars_per_hour : 
  (earnings_per_mile * (speed_per_hour * hours_travelled) - gasoline_cost_per_gallon * ((speed_per_hour * hours_travelled) / car_consumption_rate)) / hours_travelled = 3900 := 
by 
  -- The statement below essentially expresses 39 dollars per hour in cents (i.e., 3900 cents per hour).
  sorry

end net_rate_of_pay_equals_39_dollars_per_hour_l220_220182


namespace simplify_expression_l220_220735

variables (x y z : ℝ)

theorem simplify_expression (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) : 
  ((x - 2) / (4 - z)) * ((y - 3) / (2 - x)) * ((z - 4) / (3 - y)) = -1 :=
by sorry

end simplify_expression_l220_220735


namespace surveyDSuitableForComprehensiveSurvey_l220_220280

inductive Survey where
| A : Survey
| B : Survey
| C : Survey
| D : Survey

def isComprehensiveSurvey (s : Survey) : Prop :=
  match s with
  | Survey.A => False
  | Survey.B => False
  | Survey.C => False
  | Survey.D => True

theorem surveyDSuitableForComprehensiveSurvey : isComprehensiveSurvey Survey.D :=
by
  sorry

end surveyDSuitableForComprehensiveSurvey_l220_220280


namespace histogram_groups_l220_220755

theorem histogram_groups 
  (max_height : ℕ)
  (min_height : ℕ)
  (class_interval : ℕ)
  (h_max : max_height = 176)
  (h_min : min_height = 136)
  (h_interval : class_interval = 6) :
  Nat.ceil ((max_height - min_height) / class_interval) = 7 :=
by
  sorry

end histogram_groups_l220_220755


namespace find_x_l220_220875

theorem find_x (x : ℝ) : (3 / 4 * 1 / 2 * 2 / 5) * x = 765.0000000000001 → x = 5100.000000000001 :=
by
  intro h
  sorry

end find_x_l220_220875


namespace find_n_l220_220311

theorem find_n (n : ℕ) (M N : ℕ) (hM : M = 4 ^ n) (hN : N = 2 ^ n) (h : M - N = 992) : n = 5 :=
sorry

end find_n_l220_220311


namespace initial_meals_for_adults_l220_220880

theorem initial_meals_for_adults (C A : ℕ) (h1 : C = 90) (h2 : 14 * C / A = 72) : A = 18 :=
by
  sorry

end initial_meals_for_adults_l220_220880


namespace proof_2_in_M_l220_220496

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l220_220496


namespace AM_GM_inequality_example_l220_220515

open Real

theorem AM_GM_inequality_example 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ((a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2)) ≥ 9 * (a^2 * b^2 * c^2) :=
sorry

end AM_GM_inequality_example_l220_220515


namespace intersection_A_B_l220_220978

-- Define the sets A and the function f
def A : Set ℤ := {-2, 0, 2}
def f (x : ℤ) : ℤ := |x|

-- Define the set B as the image of A under the function f
def B : Set ℤ := {b | ∃ a ∈ A, f a = b}

-- State the property that every element in B has a pre-image in A
axiom B_has_preimage : ∀ b ∈ B, ∃ a ∈ A, f a = b

-- The theorem we want to prove
theorem intersection_A_B : A ∩ B = {0, 2} :=
by sorry

end intersection_A_B_l220_220978


namespace oranges_in_first_bucket_l220_220772

theorem oranges_in_first_bucket
  (x : ℕ) -- number of oranges in the first bucket
  (h1 : ∃ n, n = x) -- condition: There are some oranges in the first bucket
  (h2 : ∃ y, y = x + 17) -- condition: The second bucket has 17 more oranges than the first bucket
  (h3 : ∃ z, z = x + 6) -- condition: The third bucket has 11 fewer oranges than the second bucket
  (h4 : x + (x + 17) + (x + 6) = 89) -- condition: There are 89 oranges in all the buckets
  : x = 22 := -- conclusion: number of oranges in the first bucket is 22
sorry

end oranges_in_first_bucket_l220_220772


namespace distance_walked_by_man_l220_220917

theorem distance_walked_by_man (x t : ℝ) (h1 : d = (x + 0.5) * (4 / 5) * t) (h2 : d = (x - 0.5) * (t + 2.5)) : d = 15 :=
by
  sorry

end distance_walked_by_man_l220_220917


namespace candies_bought_l220_220571

theorem candies_bought :
  ∃ (S C : ℕ), S + C = 8 ∧ 300 * S + 500 * C = 3000 ∧ C = 3 :=
by
  sorry

end candies_bought_l220_220571


namespace bees_hatch_every_day_l220_220211

   /-- 
   Given:
   - The queen loses 900 bees every day.
   - The initial number of bees is 12500.
   - After 7 days, the total number of bees is 27201.
   
   Prove:
   - The number of bees hatching from the queen's eggs every day is 3001.
   -/
   
   theorem bees_hatch_every_day :
     ∃ x : ℕ, 12500 + 7 * (x - 900) = 27201 → x = 3001 :=
   sorry
   
end bees_hatch_every_day_l220_220211


namespace polynomial_multiplication_correct_l220_220634

noncomputable def polynomial_expansion : Polynomial ℤ :=
  (Polynomial.C (3 : ℤ) * Polynomial.X ^ 3 + Polynomial.C (4 : ℤ) * Polynomial.X ^ 2 - Polynomial.C (8 : ℤ) * Polynomial.X - Polynomial.C (5 : ℤ)) *
  (Polynomial.C (2 : ℤ) * Polynomial.X ^ 4 - Polynomial.C (3 : ℤ) * Polynomial.X ^ 2 + Polynomial.C (1 : ℤ))

theorem polynomial_multiplication_correct :
  polynomial_expansion = Polynomial.C (6 : ℤ) * Polynomial.X ^ 7 +
                         Polynomial.C (12 : ℤ) * Polynomial.X ^ 6 -
                         Polynomial.C (25 : ℤ) * Polynomial.X ^ 5 -
                         Polynomial.C (20 : ℤ) * Polynomial.X ^ 4 +
                         Polynomial.C (34 : ℤ) * Polynomial.X ^ 2 -
                         Polynomial.C (8 : ℤ) * Polynomial.X -
                         Polynomial.C (5 : ℤ) :=
by
  sorry

end polynomial_multiplication_correct_l220_220634


namespace find_a_l220_220916

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb 2 (x^2 + a)

theorem find_a (a : ℝ) : f 3 a = 1 → a = -7 :=
by
  intro h
  unfold f at h
  sorry

end find_a_l220_220916


namespace triangle_inequality_sum_2_l220_220615

theorem triangle_inequality_sum_2 (a b c : ℝ) (h_triangle : a + b + c = 2) (h_side_ineq : a + c > b ∧ a + b > c ∧ b + c > a):
  1 ≤ a * b + b * c + c * a - a * b * c ∧ a * b + b * c + c * a - a * b * c ≤ 1 + 1 / 27 :=
by
  sorry

end triangle_inequality_sum_2_l220_220615


namespace Roshesmina_pennies_l220_220068

theorem Roshesmina_pennies :
  (∀ compartments : ℕ, compartments = 12 → 
   (∀ initial_pennies : ℕ, initial_pennies = 2 → 
   (∀ additional_pennies : ℕ, additional_pennies = 6 → 
   (compartments * (initial_pennies + additional_pennies) = 96)))) :=
by
  sorry

end Roshesmina_pennies_l220_220068


namespace tangent_point_value_l220_220300

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l220_220300


namespace cloth_sales_value_l220_220426

theorem cloth_sales_value (commission_rate : ℝ) (commission : ℝ) (total_sales : ℝ) 
  (h1: commission_rate = 2.5)
  (h2: commission = 18)
  (h3: total_sales = commission / (commission_rate / 100)):
  total_sales = 720 := by
  sorry

end cloth_sales_value_l220_220426


namespace polar_coordinate_conversion_l220_220060

theorem polar_coordinate_conversion :
  ∃ (r θ : ℝ), (r = 2) ∧ (θ = 11 * Real.pi / 8) ∧ 
    ∀ (r1 θ1 : ℝ), (r1 = -2) ∧ (θ1 = 3 * Real.pi / 8) →
      (abs r1 = r) ∧ (θ1 + Real.pi = θ) :=
by
  sorry

end polar_coordinate_conversion_l220_220060


namespace california_more_license_plates_l220_220295

theorem california_more_license_plates :
  let CA_format := 26^4 * 10^2
  let NY_format := 26^3 * 10^3
  CA_format - NY_format = 28121600 := by
  let CA_format : Nat := 26^4 * 10^2
  let NY_format : Nat := 26^3 * 10^3
  have CA_plates : CA_format = 45697600 := by sorry
  have NY_plates : NY_format = 17576000 := by sorry
  calc
    CA_format - NY_format = 45697600 - 17576000 := by rw [CA_plates, NY_plates]
                    _ = 28121600 := by norm_num

end california_more_license_plates_l220_220295


namespace midpoint_trajectory_of_circle_l220_220534

theorem midpoint_trajectory_of_circle 
  (M P : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hx : B = (3, 0))
  (hp : ∃(a b : ℝ), (P = (2 * a - 3, 2 * b)) ∧ (a^2 + b^2 = 1))
  (hm : M = ((P.1 + B.1) / 2, (P.2 + B.2) / 2)) :
  M.1^2 + M.2^2 - 3 * M.1 + 2 = 0 :=
by {
  -- Proof goes here
  sorry
}

end midpoint_trajectory_of_circle_l220_220534


namespace Billy_weighs_more_l220_220527

-- Variables and assumptions
variable (Billy Brad Carl : ℕ)
variable (b_weight : Billy = 159)
variable (c_weight : Carl = 145)
variable (brad_formula : Brad = Carl + 5)

-- Theorem statement to prove the required condition
theorem Billy_weighs_more :
  Billy - Brad = 9 :=
by
  -- Here we put the proof steps, but it's omitted as per instructions.
  sorry

end Billy_weighs_more_l220_220527


namespace adapted_bowling_ball_volume_l220_220431

noncomputable def volume_adapted_bowling_ball : ℝ :=
  let volume_sphere := (4/3) * Real.pi * (20 ^ 3)
  let volume_hole1 := Real.pi * (1 ^ 2) * 10
  let volume_hole2 := Real.pi * (1.5 ^ 2) * 10
  let volume_hole3 := Real.pi * (2 ^ 2) * 10
  volume_sphere - (volume_hole1 + volume_hole2 + volume_hole3)

theorem adapted_bowling_ball_volume :
  volume_adapted_bowling_ball = 10594.17 * Real.pi :=
sorry

end adapted_bowling_ball_volume_l220_220431


namespace blueberry_pancakes_count_l220_220748

-- Definitions of the conditions
def total_pancakes : ℕ := 67
def banana_pancakes : ℕ := 24
def plain_pancakes : ℕ := 23

-- Statement of the problem
theorem blueberry_pancakes_count :
  total_pancakes - banana_pancakes - plain_pancakes = 20 := by
  sorry

end blueberry_pancakes_count_l220_220748


namespace solution_l220_220007

noncomputable def problem_statement (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

theorem solution (f : ℝ → ℝ) (h : problem_statement f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b * x^2 := by
  sorry

end solution_l220_220007


namespace calculate_expression_l220_220840

theorem calculate_expression :
  ( (128^2 - 5^2) / (72^2 - 13^2) * ((72 - 13) * (72 + 13)) / ((128 - 5) * (128 + 5)) * (128 + 5) / (72 + 13) )
  = (133 / 85) :=
by
  -- placeholder for the proof
  sorry

end calculate_expression_l220_220840


namespace tangent_line_at_2_number_of_zeros_l220_220778

noncomputable def f (x : ℝ) := 3 * Real.log x + (1/2) * x^2 - 4 * x + 1

theorem tangent_line_at_2 :
  let x := 2
  ∃ k b : ℝ, (∀ y : ℝ, y = k * x + b) ∧ (k = -1/2) ∧ (b = 3 * Real.log 2 - 5) ∧ (∀ x y : ℝ, (y - (3 * Real.log 2 - 5) = -1/2 * (x - 2)) ↔ (x + 2 * y - 6 * Real.log 2 + 8 = 0)) :=
by
  sorry

noncomputable def g (x : ℝ) (m : ℝ) := f x - m

theorem number_of_zeros (m : ℝ) :
  let g := g
  (m > -5/2 ∨ m < 3 * Real.log 3 - 13/2 → ∃ x : ℝ, g x = 0) ∧ 
  (m = -5/2 ∨ m = 3 * Real.log 3 - 13/2 → ∃ x y : ℝ, g x = 0 ∧ g y = 0 ∧ x ≠ y) ∧
  (3 * Real.log 3 - 13/2 < m ∧ m < -5/2 → ∃ x y z : ℝ, g x = 0 ∧ g y = 0 ∧ g z = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) :=
by
  sorry

end tangent_line_at_2_number_of_zeros_l220_220778


namespace solution_set_inequality_l220_220322

theorem solution_set_inequality (x : ℝ) : 4 * x^2 - 3 * x > 5 ↔ x < -5/4 ∨ x > 1 :=
by
  sorry

end solution_set_inequality_l220_220322


namespace arrangement_problem_l220_220585
noncomputable def num_arrangements : ℕ := 144

theorem arrangement_problem (A B C D E F : ℕ) 
  (adjacent_easy : A = B) 
  (not_adjacent_difficult : E ≠ F) : num_arrangements = 144 :=
by sorry

end arrangement_problem_l220_220585


namespace older_brother_catches_up_l220_220664

theorem older_brother_catches_up :
  ∃ (x : ℝ), 0 ≤ x ∧ 6 * x = 2 + 2 * x ∧ x + 1 < 1.75 :=
by
  sorry

end older_brother_catches_up_l220_220664


namespace line_equation_through_point_slope_l220_220269

theorem line_equation_through_point_slope :
  ∃ (a b c : ℝ), (a, b) ≠ (0, 0) ∧ (a * 1 + b * 3 + c = 0) ∧ (y = -4 * x → k = -4 / 9) ∧ (∀ (x y : ℝ), y - 3 = k * (x - 1) → 4 * x + 3 * y - 13 = 0) :=
sorry

end line_equation_through_point_slope_l220_220269


namespace percentage_discount_four_friends_l220_220209

theorem percentage_discount_four_friends 
  (num_friends : ℕ)
  (original_price : ℝ)
  (total_spent : ℝ)
  (item_per_friend : ℕ)
  (total_items : ℕ)
  (each_spent : ℝ)
  (discount_percentage : ℝ):
  num_friends = 4 →
  original_price = 20 →
  total_spent = 40 →
  item_per_friend = 1 →
  total_items = num_friends * item_per_friend →
  each_spent = total_spent / num_friends →
  discount_percentage = ((original_price - each_spent) / original_price) * 100 →
  discount_percentage = 50 :=
by
  sorry

end percentage_discount_four_friends_l220_220209


namespace intersection_value_l220_220044

theorem intersection_value (x0 : ℝ) (h1 : -x0 = Real.tan x0) (h2 : x0 ≠ 0) :
  (x0^2 + 1) * (1 + Real.cos (2 * x0)) = 2 := 
  sorry

end intersection_value_l220_220044


namespace line_intersects_x_axis_at_3_0_l220_220125

theorem line_intersects_x_axis_at_3_0 : ∃ (x : ℝ), ∃ (y : ℝ), 2 * y + 5 * x = 15 ∧ y = 0 ∧ (x, y) = (3, 0) :=
by
  sorry

end line_intersects_x_axis_at_3_0_l220_220125


namespace reservoir_fullness_before_storm_l220_220178

-- Definition of the conditions as Lean definitions
def storm_deposits : ℝ := 120 -- in billion gallons
def reservoir_percentage_after_storm : ℝ := 85 -- percentage
def original_contents : ℝ := 220 -- in billion gallons

-- The proof statement
theorem reservoir_fullness_before_storm (storm_deposits reservoir_percentage_after_storm original_contents : ℝ) : 
    (169 / 340) * 100 = 49.7 := 
  sorry

end reservoir_fullness_before_storm_l220_220178


namespace line_equation_l220_220106

theorem line_equation
  (x y : ℝ)
  (h1 : 2 * x + y + 2 = 0)
  (h2 : 2 * x - y + 2 = 0)
  (h3 : ∀ x y, x + y = 0 → x - 1 = y): 
  x - y + 1 = 0 :=
sorry

end line_equation_l220_220106


namespace extreme_points_of_f_range_of_a_for_f_le_g_l220_220859

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x + (1 / 2) * x^2 + a * x

noncomputable def g (x : ℝ) : ℝ :=
  Real.exp x + (3 / 2) * x^2

theorem extreme_points_of_f (a : ℝ) :
  (∃ (x1 x2 : ℝ), x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0)
    ↔ a < -2 :=
sorry

theorem range_of_a_for_f_le_g :
  (∀ x : ℝ, x > 0 → f x a ≤ g x) ↔ a ≤ Real.exp 1 + 1 :=
sorry

end extreme_points_of_f_range_of_a_for_f_le_g_l220_220859


namespace meaningful_expression_l220_220886

theorem meaningful_expression (x : ℝ) : 
    (x + 2 > 0 ∧ x - 1 ≠ 0) ↔ (x > -2 ∧ x ≠ 1) :=
by
  sorry

end meaningful_expression_l220_220886


namespace reggie_free_throws_l220_220753

namespace BasketballShootingContest

-- Define the number of points for different shots
def points (layups free_throws long_shots : ℕ) : ℕ :=
  1 * layups + 2 * free_throws + 3 * long_shots

-- Conditions given in the problem
def Reggie_points (F: ℕ) : ℕ := 
  points 3 F 1

def Brother_points : ℕ := 
  points 0 0 4

-- The given condition that Reggie loses by 2 points
theorem reggie_free_throws:
  ∃ F : ℕ, Reggie_points F + 2 = Brother_points :=
sorry

end BasketballShootingContest

end reggie_free_throws_l220_220753


namespace grace_clyde_ratio_l220_220907

theorem grace_clyde_ratio (C G : ℕ) (h1 : G = C + 35) (h2 : G = 40) : G / C = 8 :=
by sorry

end grace_clyde_ratio_l220_220907


namespace weekly_earnings_before_rent_l220_220781

theorem weekly_earnings_before_rent (EarningsAfterRent : ℝ) (weeks : ℕ) (rentPerWeek : ℝ) :
  EarningsAfterRent = 93899 → weeks = 233 → rentPerWeek = 49 →
  ((EarningsAfterRent + rentPerWeek * weeks) / weeks) = 451.99 :=
by
  intros H1 H2 H3
  -- convert the assumptions to the required form
  rw [H1, H2, H3]
  -- provide the objective statement
  change ((93899 + 49 * 233) / 233) = 451.99
  -- leave the final proof details as a sorry for now
  sorry

end weekly_earnings_before_rent_l220_220781


namespace estimated_germination_probability_l220_220511

-- This definition represents the conditions of the problem in Lean.
def germination_data : List (ℕ × ℕ × Real) :=
  [(2, 2, 1.000), (5, 4, 0.800), (10, 9, 0.900), (50, 44, 0.880), (100, 92, 0.920),
   (500, 463, 0.926), (1000, 928, 0.928), (1500, 1396, 0.931), (2000, 1866, 0.933), (3000, 2794, 0.931)]

-- The theorem states that the germination probability is approximately 0.93.
theorem estimated_germination_probability (data : List (ℕ × ℕ × Real)) (h : data = germination_data) :
  ∃ p : Real, p = 0.93 ∧ ∀ n m r, (n, m, r) ∈ data → |r - p| < 0.01 :=
by
  -- Placeholder for proof
  sorry

end estimated_germination_probability_l220_220511


namespace analysis_duration_unknown_l220_220355

-- Definitions based on the given conditions
def number_of_bones : Nat := 206
def analysis_duration_per_bone (bone: Nat) : Nat := 5  -- assumed fixed for simplicity
-- Time spent analyzing all bones (which needs more information to be accurately known)
def total_analysis_time (bones_analyzed: Nat) (hours_per_bone: Nat) : Nat := bones_analyzed * hours_per_bone

-- Given the number of bones and duration per bone, there isn't enough information to determine the total analysis duration
theorem analysis_duration_unknown (total_bones : Nat) (duration_per_bone : Nat) (bones_remaining: Nat) (analysis_already_done : Nat) :
  total_bones = number_of_bones →
  (∀ bone, analysis_duration_per_bone bone = duration_per_bone) →
  analysis_already_done ≠ (total_bones - bones_remaining) ->
  ∃ hours_needed, hours_needed = total_analysis_time (total_bones - bones_remaining) duration_per_bone :=
by
  intros
  sorry

end analysis_duration_unknown_l220_220355


namespace no_line_normal_to_both_curves_l220_220366

theorem no_line_normal_to_both_curves :
  ¬ ∃ a b : ℝ, ∃ (l : ℝ → ℝ),
    -- normal to y = cosh x at x = a
    (∀ x : ℝ, l x = -1 / (Real.sinh a) * (x - a) + Real.cosh a) ∧
    -- normal to y = sinh x at x = b
    (∀ x : ℝ, l x = -1 / (Real.cosh b) * (x - b) + Real.sinh b) := 
  sorry

end no_line_normal_to_both_curves_l220_220366


namespace proportion_of_ones_l220_220144

theorem proportion_of_ones (m n : ℕ) (h : Nat.gcd m n = 1) : 
  m + n = 275 :=
  sorry

end proportion_of_ones_l220_220144


namespace alice_walks_miles_each_morning_l220_220744

theorem alice_walks_miles_each_morning (x : ℕ) :
  (5 * x + 5 * 12 = 110) → x = 10 :=
by
  intro h
  -- Proof omitted
  sorry

end alice_walks_miles_each_morning_l220_220744


namespace prize_interval_l220_220881

theorem prize_interval (prize1 prize2 prize3 prize4 prize5 interval : ℝ) (h1 : prize1 = 5000) 
  (h2 : prize2 = 5000 - interval) (h3 : prize3 = 5000 - 2 * interval) 
  (h4 : prize4 = 5000 - 3 * interval) (h5 : prize5 = 5000 - 4 * interval) 
  (h_total : prize1 + prize2 + prize3 + prize4 + prize5 = 15000) : 
  interval = 1000 := 
by
  sorry

end prize_interval_l220_220881


namespace bob_total_earnings_l220_220990

def hourly_rate_regular := 5
def hourly_rate_overtime := 6
def regular_hours_per_week := 40

def hours_worked_week1 := 44
def hours_worked_week2 := 48

def earnings_week1 : ℕ :=
  let regular_hours := regular_hours_per_week
  let overtime_hours := hours_worked_week1 - regular_hours_per_week
  (regular_hours * hourly_rate_regular) + (overtime_hours * hourly_rate_overtime)

def earnings_week2 : ℕ :=
  let regular_hours := regular_hours_per_week
  let overtime_hours := hours_worked_week2 - regular_hours_per_week
  (regular_hours * hourly_rate_regular) + (overtime_hours * hourly_rate_overtime)

def total_earnings : ℕ := earnings_week1 + earnings_week2

theorem bob_total_earnings : total_earnings = 472 := by
  sorry

end bob_total_earnings_l220_220990


namespace no_equal_partition_of_173_ones_and_neg_ones_l220_220865

theorem no_equal_partition_of_173_ones_and_neg_ones
  (L : List ℤ) (h1 : L.length = 173) (h2 : ∀ x ∈ L, x = 1 ∨ x = -1) :
  ¬ (∃ (L1 L2 : List ℤ), L = L1 ++ L2 ∧ L1.sum = L2.sum) :=
by
  sorry

end no_equal_partition_of_173_ones_and_neg_ones_l220_220865


namespace height_percentage_difference_l220_220950

theorem height_percentage_difference (H_A H_B : ℝ) (h : H_B = H_A * 1.5384615384615385) :
  (H_B - H_A) / H_B * 100 = 35 := 
sorry

end height_percentage_difference_l220_220950


namespace tan_of_geometric_sequence_is_negative_sqrt_3_l220_220759

variable {a : ℕ → ℝ} 

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q, m + n = p + q → a m * a n = a p * a q

theorem tan_of_geometric_sequence_is_negative_sqrt_3 
  (hgeo : is_geometric_sequence a)
  (hcond : a 2 * a 3 * a 4 = - a 7 ^ 2 ∧ a 7 ^ 2 = 64) :
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = - Real.sqrt 3 :=
sorry

end tan_of_geometric_sequence_is_negative_sqrt_3_l220_220759


namespace find_x_intercept_of_line_through_points_l220_220149

-- Definitions based on the conditions
def point1 : ℝ × ℝ := (-1, 1)
def point2 : ℝ × ℝ := (0, 3)

-- Statement: The x-intercept of the line passing through the given points is -3/2
theorem find_x_intercept_of_line_through_points :
  let x1 := point1.1
  let y1 := point1.2
  let x2 := point2.1
  let y2 := point2.2
  ∃ x_intercept : ℝ, x_intercept = -3 / 2 ∧ 
    (∀ x, ∀ y, (x2 - x1) * (y - y1) = (y2 - y1) * (x - x1) → y = 0 → x = x_intercept) :=
by
  sorry

end find_x_intercept_of_line_through_points_l220_220149


namespace find_f3_value_l220_220292

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * Real.tan x - b * x^5 + c * x - 3

theorem find_f3_value (a b c : ℝ) (h : f (-3) a b c = 7) : f 3 a b c = -13 := 
by 
  sorry

end find_f3_value_l220_220292


namespace last_digit_322_power_111569_l220_220807

theorem last_digit_322_power_111569 : (322 ^ 111569) % 10 = 2 := 
by {
  sorry
}

end last_digit_322_power_111569_l220_220807


namespace abc_inequality_l220_220869

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
  sorry

end abc_inequality_l220_220869


namespace four_m_plus_one_2013_eq_neg_one_l220_220291

theorem four_m_plus_one_2013_eq_neg_one (m : ℝ) (h : |m| = m + 1) : (4 * m + 1) ^ 2013 = -1 := 
sorry

end four_m_plus_one_2013_eq_neg_one_l220_220291


namespace geometric_sum_S_40_l220_220028

variable (S : ℕ → ℝ)

-- Conditions
axiom sum_S_10 : S 10 = 18
axiom sum_S_20 : S 20 = 24

-- Proof statement
theorem geometric_sum_S_40 : S 40 = 80 / 3 :=
by
  sorry

end geometric_sum_S_40_l220_220028


namespace greater_expected_area_l220_220904

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l220_220904


namespace square_of_any_real_number_not_always_greater_than_zero_l220_220039

theorem square_of_any_real_number_not_always_greater_than_zero (a : ℝ) : 
    (∀ x : ℝ, x^2 ≥ 0) ∧ (exists x : ℝ, x = 0 ∧ x^2 = 0) :=
by {
  sorry
}

end square_of_any_real_number_not_always_greater_than_zero_l220_220039


namespace gdp_scientific_notation_l220_220173

theorem gdp_scientific_notation (gdp : ℝ) (h : gdp = 338.8 * 10^9) : gdp = 3.388 * 10^10 :=
by sorry

end gdp_scientific_notation_l220_220173


namespace continuity_of_f_at_3_l220_220684

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 3 then 3*x^2 + 2*x - 4 else b*x + 7

theorem continuity_of_f_at_3 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x b - f 3 b) < ε) ↔ b = 22 / 3 :=
by
  sorry

end continuity_of_f_at_3_l220_220684


namespace total_revenue_correct_l220_220955

noncomputable def total_ticket_revenue : ℕ :=
  let revenue_2pm := 180 * 6 + 20 * 5 + 60 * 4 + 20 * 3 + 20 * 5
  let revenue_5pm := 95 * 8 + 30 * 7 + 110 * 5 + 15 * 6
  let revenue_8pm := 122 * 10 + 74 * 7 + 29 * 8
  revenue_2pm + revenue_5pm + revenue_8pm

theorem total_revenue_correct : total_ticket_revenue = 5160 := by
  sorry

end total_revenue_correct_l220_220955


namespace second_order_derivative_l220_220220

-- Define the parameterized functions x and y
noncomputable def x (t : ℝ) : ℝ := 1 / t
noncomputable def y (t : ℝ) : ℝ := 1 / (1 + t ^ 2)

-- Define the second-order derivative of y with respect to x
noncomputable def d2y_dx2 (t : ℝ) : ℝ := (2 * (t^2 - 3) * t^4) / (1 + t^2) ^ 3

-- Prove the relationship based on given conditions
theorem second_order_derivative :
  ∀ t : ℝ, (∃ x y : ℝ, x = 1 / t ∧ y = 1 / (1 + t ^ 2)) → 
    (d2y_dx2 t) = (2 * (t^2 - 3) * t^4) / (1 + t^2) ^ 3 :=
by
  intros t ht
  -- Proof omitted
  sorry

end second_order_derivative_l220_220220


namespace shaded_area_floor_l220_220568

noncomputable def area_of_white_quarter_circle : ℝ := Real.pi / 4

noncomputable def area_of_white_per_tile : ℝ := 4 * area_of_white_quarter_circle

noncomputable def area_of_tile : ℝ := 4

noncomputable def shaded_area_per_tile : ℝ := area_of_tile - area_of_white_per_tile

noncomputable def number_of_tiles : ℕ := by
  have floor_area : ℝ := 12 * 15
  have tile_area : ℝ := 2 * 2
  exact Nat.floor (floor_area / tile_area)

noncomputable def total_shaded_area (num_tiles : ℕ) : ℝ := num_tiles * shaded_area_per_tile

theorem shaded_area_floor : total_shaded_area number_of_tiles = 180 - 45 * Real.pi := by
  sorry

end shaded_area_floor_l220_220568


namespace average_weight_of_dogs_is_5_l220_220788

def weight_of_brown_dog (B : ℝ) : ℝ := B
def weight_of_black_dog (B : ℝ) : ℝ := B + 1
def weight_of_white_dog (B : ℝ) : ℝ := 2 * B
def weight_of_grey_dog (B : ℝ) : ℝ := B - 1

theorem average_weight_of_dogs_is_5 (B : ℝ) (h : (weight_of_brown_dog B + weight_of_black_dog B + weight_of_white_dog B + weight_of_grey_dog B) / 4 = 5) :
  5 = 5 :=
by sorry

end average_weight_of_dogs_is_5_l220_220788


namespace largest_base_6_five_digits_l220_220815

-- Define the base-6 number 55555 in base 10
def base_6_to_base_10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 10000) % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0 * 6^0

theorem largest_base_6_five_digits : base_6_to_base_10 55555 = 7775 := by
  sorry

end largest_base_6_five_digits_l220_220815


namespace fruit_seller_apples_l220_220835

theorem fruit_seller_apples : 
  ∃ (x : ℝ), (x * 0.6 = 420) → x = 700 :=
sorry

end fruit_seller_apples_l220_220835


namespace frank_bought_2_bags_of_chips_l220_220662

theorem frank_bought_2_bags_of_chips
  (cost_choco_bar : ℕ)
  (num_choco_bar : ℕ)
  (total_money : ℕ)
  (change : ℕ)
  (cost_bag_chip : ℕ)
  (num_bags_chip : ℕ)
  (h1 : cost_choco_bar = 2)
  (h2 : num_choco_bar = 5)
  (h3 : total_money = 20)
  (h4 : change = 4)
  (h5 : cost_bag_chip = 3)
  (h6 : total_money - change = (cost_choco_bar * num_choco_bar) + (cost_bag_chip * num_bags_chip)) :
  num_bags_chip = 2 := by
  sorry

end frank_bought_2_bags_of_chips_l220_220662


namespace temperature_drop_l220_220074

-- Define the initial temperature and the drop in temperature
def initial_temperature : ℤ := -6
def drop : ℤ := 5

-- Define the resulting temperature after the drop
def resulting_temperature : ℤ := initial_temperature - drop

-- The theorem to be proved
theorem temperature_drop : resulting_temperature = -11 :=
by
  sorry

end temperature_drop_l220_220074


namespace new_edition_pages_less_l220_220819

theorem new_edition_pages_less :
  let new_edition_pages := 450
  let old_edition_pages := 340
  (2 * old_edition_pages - new_edition_pages) = 230 :=
by
  let new_edition_pages := 450
  let old_edition_pages := 340
  sorry

end new_edition_pages_less_l220_220819


namespace exist_coprime_integers_l220_220415

theorem exist_coprime_integers:
  ∀ (a b p : ℤ), ∃ (k l : ℤ), Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
by
  sorry

end exist_coprime_integers_l220_220415


namespace proof_of_problem_l220_220108

noncomputable def problem_statement : Prop :=
  ∃ (x y z m : ℝ), (x > 0 ∧ y > 0 ∧ z > 0 ∧ x^3 * y^2 * z = 1 ∧ m = x + 2*y + 3*z ∧ m^3 = 72)

theorem proof_of_problem : problem_statement :=
sorry

end proof_of_problem_l220_220108


namespace map_distance_l220_220825

noncomputable def map_scale_distance (actual_distance_km : ℕ) (scale : ℕ) : ℕ :=
  let actual_distance_cm := actual_distance_km * 100000;  -- conversion from kilometers to centimeters
  actual_distance_cm / scale

theorem map_distance (d_km : ℕ) (scale : ℕ) (h1 : d_km = 500) (h2 : scale = 8000000) :
  map_scale_distance d_km scale = 625 :=
by
  rw [h1, h2]
  dsimp [map_scale_distance]
  norm_num
  sorry

end map_distance_l220_220825


namespace calc_expr_correct_l220_220499

noncomputable def eval_expr : ℚ :=
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 12.5

theorem calc_expr_correct : eval_expr = 12.5 :=
by
  sorry

end calc_expr_correct_l220_220499


namespace exists_1998_distinct_natural_numbers_l220_220594

noncomputable def exists_1998_distinct_numbers : Prop :=
  ∃ (s : Finset ℕ), s.card = 1998 ∧
    (∀ {x y : ℕ}, x ∈ s → y ∈ s → x ≠ y → (x * y) % ((x - y) ^ 2) = 0)

theorem exists_1998_distinct_natural_numbers : exists_1998_distinct_numbers :=
by
  sorry

end exists_1998_distinct_natural_numbers_l220_220594


namespace total_chickens_and_ducks_l220_220750

-- Definitions based on conditions
def num_chickens : Nat := 45
def more_chickens_than_ducks : Nat := 8
def num_ducks : Nat := num_chickens - more_chickens_than_ducks

-- The proof statement
theorem total_chickens_and_ducks : num_chickens + num_ducks = 82 := by
  -- The actual proof is omitted, only the statement is required
  sorry

end total_chickens_and_ducks_l220_220750


namespace most_likely_number_of_cars_l220_220877

theorem most_likely_number_of_cars 
  (total_time_seconds : ℕ)
  (rate_cars_per_second : ℚ)
  (h1 : total_time_seconds = 180)
  (h2 : rate_cars_per_second = 8 / 15) : 
  ∃ (n : ℕ), n = 100 :=
by
  sorry

end most_likely_number_of_cars_l220_220877


namespace common_ratio_of_geometric_sequence_is_4_l220_220655

theorem common_ratio_of_geometric_sequence_is_4 
  (a_n : ℕ → ℝ) 
  (b_n : ℕ → ℝ) 
  (d : ℝ) 
  (h₁ : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h₂ : d ≠ 0)
  (h₃ : (a_n 3)^2 = (a_n 2) * (a_n 7)) :
  b_n 2 / b_n 1 = 4 :=
sorry

end common_ratio_of_geometric_sequence_is_4_l220_220655


namespace correct_option_l220_220723

-- Conditions as definitions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def optionC (a : ℝ) : Prop := (-2 * a^2)^3 = -8 * a^6
def optionD (a : ℝ) : Prop := a^6 / a^2 = a^3

-- The statement to prove
theorem correct_option (a : ℝ) : optionC a :=
by 
  unfold optionC
  sorry

end correct_option_l220_220723


namespace intersection_P_Q_l220_220607

def P : Set ℝ := {x | |x| > 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {x | -2 ≤ x ∧ x < -1 ∨ 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_P_Q_l220_220607


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l220_220393

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l220_220393


namespace catch_up_time_l220_220456

noncomputable def speed_ratios (v : ℝ) : Prop :=
  let a_speed := (4 / 5) * v
  let b_speed := (2 / 5) * v
  a_speed = 2 * b_speed

theorem catch_up_time (v t : ℝ) (a_speed b_speed : ℝ)
  (h1 : a_speed = (4 / 5) * v)
  (h2 : b_speed = (2 / 5) * v)
  (h3 : a_speed = 2 * b_speed) :
  (t = 11) := by
  sorry

end catch_up_time_l220_220456


namespace tea_bags_count_l220_220799

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l220_220799


namespace infinite_solutions_of_linear_eq_l220_220217

theorem infinite_solutions_of_linear_eq (a b : ℝ) : 
  (∃ b : ℝ, ∃ a : ℝ, 5 * a - 11 * b = 21) := sorry

end infinite_solutions_of_linear_eq_l220_220217


namespace students_play_both_sports_l220_220312

theorem students_play_both_sports 
  (total_students : ℕ) (students_play_football : ℕ) 
  (students_play_cricket : ℕ) (students_play_neither : ℕ) :
  total_students = 470 → students_play_football = 325 → 
  students_play_cricket = 175 → students_play_neither = 50 → 
  (students_play_football + students_play_cricket - 
    (total_students - students_play_neither)) = 80 :=
by
  intros h_total h_football h_cricket h_neither
  sorry

end students_play_both_sports_l220_220312


namespace r_and_s_earns_per_day_l220_220940

variable (P Q R S : Real)

-- Conditions as given in the problem
axiom cond1 : P + Q + R + S = 2380 / 9
axiom cond2 : P + R = 600 / 5
axiom cond3 : Q + S = 800 / 6
axiom cond4 : Q + R = 910 / 7
axiom cond5 : P = 150 / 3

theorem r_and_s_earns_per_day : R + S = 143.33 := by
  sorry

end r_and_s_earns_per_day_l220_220940


namespace find_x_l220_220575

open Nat

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x : ℕ) (hx : x > 0) (hprime : is_prime (x^5 + x + 1)) : x = 1 := 
by 
  sorry

end find_x_l220_220575


namespace cooking_time_eq_80_l220_220242

-- Define the conditions
def hushpuppies_per_guest : Nat := 5
def number_of_guests : Nat := 20
def hushpuppies_per_batch : Nat := 10
def time_per_batch : Nat := 8

-- Calculate total number of hushpuppies needed
def total_hushpuppies : Nat := hushpuppies_per_guest * number_of_guests

-- Calculate number of batches needed
def number_of_batches : Nat := total_hushpuppies / hushpuppies_per_batch

-- Calculate total time needed
def total_time_needed : Nat := number_of_batches * time_per_batch

-- Statement to prove the correctness
theorem cooking_time_eq_80 : total_time_needed = 80 := by
  sorry

end cooking_time_eq_80_l220_220242


namespace initial_nickels_proof_l220_220378

def initial_nickels (N : ℕ) (D : ℕ) (total_value : ℝ) : Prop :=
  D = 3 * N ∧
  total_value = (N + 2 * N) * 0.05 + 3 * N * 0.10 ∧
  total_value = 9

theorem initial_nickels_proof : ∃ N, ∃ D, (initial_nickels N D 9) → (N = 20) :=
by
  sorry

end initial_nickels_proof_l220_220378


namespace find_value_of_expression_l220_220404

theorem find_value_of_expression (x y : ℝ) 
  (h1 : 4 * x + 2 * y = 20)
  (h2 : 2 * x + 4 * y = 16) : 
  4 * x ^ 2 + 12 * x * y + 12 * y ^ 2 = 292 :=
by
  sorry

end find_value_of_expression_l220_220404


namespace group_selection_l220_220741

theorem group_selection (m k n : ℕ) (h_m : m = 6) (h_k : k = 7) 
  (groups : ℕ → ℕ) (h_groups : groups k = n) : 
  n % 10 = (m + k) % 10 :=
by
  sorry

end group_selection_l220_220741


namespace point_in_second_quadrant_coordinates_l220_220596

variable (x y : ℝ)
variable (P : ℝ × ℝ)
variable (h1 : P.1 = x)
variable (h2 : P.2 = y)

def isInSecondQuadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def distanceToXAxis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distanceToYAxis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem point_in_second_quadrant_coordinates (h1 : isInSecondQuadrant P)
    (h2 : distanceToXAxis P = 2)
    (h3 : distanceToYAxis P = 1) :
    P = (-1, 2) :=
by 
  sorry

end point_in_second_quadrant_coordinates_l220_220596


namespace product_xy_eq_3_l220_220860

variable {x y : ℝ}
variables (h₀ : x ≠ y) (h₁ : x ≠ 0) (h₂ : y ≠ 0)
variable (h₃ : x + (3 / x) = y + (3 / y))

theorem product_xy_eq_3 : x * y = 3 := by
  sorry

end product_xy_eq_3_l220_220860


namespace cornflowers_count_l220_220083

theorem cornflowers_count
  (n k : ℕ)
  (total_flowers : 9 * n + 17 * k = 70)
  (equal_dandelions_daisies : 5 * n = 7 * k) :
  (9 * n - 20 - 14 = 2) ∧ (17 * k - 20 - 14 = 0) :=
by
  sorry

end cornflowers_count_l220_220083


namespace polynomial_operation_correct_l220_220201

theorem polynomial_operation_correct :
    ∀ (s t : ℝ), (s * t + 0.25 * s * t = 0) :=
by
  intros s t
  sorry

end polynomial_operation_correct_l220_220201


namespace range_of_c_extreme_values_l220_220123

noncomputable def f (c x : ℝ) : ℝ := x^3 - 2 * c * x^2 + x

theorem range_of_c_extreme_values 
  (c : ℝ) 
  (h : ∃ a b : ℝ, a ≠ b ∧ (3 * a^2 - 4 * c * a + 1 = 0) ∧ (3 * b^2 - 4 * c * b + 1 = 0)) :
  c < - (Real.sqrt 3 / 2) ∨ c > (Real.sqrt 3 / 2) :=
by sorry

end range_of_c_extreme_values_l220_220123


namespace largest_constant_C_l220_220246

theorem largest_constant_C :
  ∃ C : ℝ, 
    (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z - 1)) 
      ∧ (∀ D : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ D * (x + y + z - 1)) → C ≥ D)
    ∧ C = (2 + 2 * Real.sqrt 7) / 3 :=
sorry

end largest_constant_C_l220_220246


namespace compare_abc_l220_220116

noncomputable def a : ℝ := - Real.logb 2 (1/5)
noncomputable def b : ℝ := Real.logb 8 27
noncomputable def c : ℝ := Real.exp (-3)

theorem compare_abc : a = Real.logb 2 5 ∧ 1 < b ∧ b < 2 ∧ c = Real.exp (-3) → a > b ∧ b > c :=
by
  sorry

end compare_abc_l220_220116


namespace two_sum_fourth_power_square_l220_220708

-- Define the condition
def sum_zero (x y z : ℤ) : Prop := x + y + z = 0

-- The theorem to be proven
theorem two_sum_fourth_power_square (x y z : ℤ) (h : sum_zero x y z) : ∃ k : ℤ, 2 * (x^4 + y^4 + z^4) = k^2 :=
by
  -- skipping the proof
  sorry

end two_sum_fourth_power_square_l220_220708


namespace gcd_288_123_l220_220062

theorem gcd_288_123 : gcd 288 123 = 3 :=
by
  sorry

end gcd_288_123_l220_220062


namespace maximize_profit_l220_220342

theorem maximize_profit (x : ℤ) (hx : 20 ≤ x ∧ x ≤ 30) :
  (∀ y, 20 ≤ y ∧ y ≤ 30 → ((y - 20) * (30 - y)) ≤ ((25 - 20) * (30 - 25))) := 
sorry

end maximize_profit_l220_220342


namespace longest_chord_length_of_circle_l220_220770

theorem longest_chord_length_of_circle (r : ℝ) (h : r = 5) : ∃ d, d = 10 :=
by
  sorry

end longest_chord_length_of_circle_l220_220770


namespace lcm_48_180_l220_220578

theorem lcm_48_180 : Int.lcm 48 180 = 720 := by
  sorry

end lcm_48_180_l220_220578


namespace friends_count_l220_220029

-- Define the given conditions
def initial_chicken_wings := 2
def additional_chicken_wings := 25
def chicken_wings_per_person := 3

-- Define the total number of chicken wings
def total_chicken_wings := initial_chicken_wings + additional_chicken_wings

-- Define the target number of friends in the group
def number_of_friends := total_chicken_wings / chicken_wings_per_person

-- The theorem stating that the number of friends is 9
theorem friends_count : number_of_friends = 9 := by
  sorry

end friends_count_l220_220029


namespace toothpick_250_stage_l220_220141

-- Define the arithmetic sequence for number of toothpicks at each stage
def toothpicks (n : ℕ) : ℕ := 5 + (n - 1) * 4

-- The proof statement for the 250th stage
theorem toothpick_250_stage : toothpicks 250 = 1001 :=
  by
  sorry

end toothpick_250_stage_l220_220141


namespace tangent_line_to_ellipse_l220_220264

theorem tangent_line_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 1 → x^2 + 4 * y^2 = 1 → (x^2 + 4 * (m * x + 1)^2 = 1)) →
  m^2 = 3 / 4 :=
by
  sorry

end tangent_line_to_ellipse_l220_220264


namespace total_people_going_to_museum_l220_220603

def number_of_people_on_first_bus := 12
def number_of_people_on_second_bus := 2 * number_of_people_on_first_bus
def number_of_people_on_third_bus := number_of_people_on_second_bus - 6
def number_of_people_on_fourth_bus := number_of_people_on_first_bus + 9

theorem total_people_going_to_museum :
  number_of_people_on_first_bus + number_of_people_on_second_bus + number_of_people_on_third_bus + number_of_people_on_fourth_bus = 75 :=
by
  sorry

end total_people_going_to_museum_l220_220603


namespace hyperbola_min_value_l220_220319

def hyperbola_condition : Prop :=
  ∀ (m : ℝ), ∀ (x y : ℝ), (4 * x + 3 * y + m = 0 → (x^2 / 9 - y^2 / 16 = 1) → false)

noncomputable def minimum_value : ℝ :=
  2 * Real.sqrt 37 - 6

theorem hyperbola_min_value :
  hyperbola_condition → minimum_value =  2 * Real.sqrt 37 - 6 :=
by
  intro h
  sorry

end hyperbola_min_value_l220_220319


namespace people_stools_chairs_l220_220488

def total_legs (x y z : ℕ) : ℕ := 2 * x + 3 * y + 4 * z 

theorem people_stools_chairs (x y z : ℕ) : 
  (x > y) → (x > z) → (x < y + z) → (total_legs x y z = 32) → 
  (x = 5 ∧ y = 2 ∧ z = 4) :=
by
  intro h1 h2 h3 h4
  sorry

end people_stools_chairs_l220_220488


namespace find_constants_l220_220631

theorem find_constants (c d : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
     (r^3 + c*r^2 + 17*r + 10 = 0) ∧ (s^3 + c*s^2 + 17*s + 10 = 0) ∧
     (r^3 + d*r^2 + 22*r + 14 = 0) ∧ (s^3 + d*s^2 + 22*s + 14 = 0)) →
  (c = 8 ∧ d = 9) :=
by
  sorry

end find_constants_l220_220631


namespace waiting_for_stocker_proof_l220_220856

-- Definitions for the conditions
def waiting_for_cart := 3
def waiting_for_employee := 13
def waiting_in_line := 18
def total_shopping_trip_time := 90
def time_shopping := 42

-- Calculate the total waiting time
def total_waiting_time := total_shopping_trip_time - time_shopping

-- Calculate the total known waiting time
def total_known_waiting_time := waiting_for_cart + waiting_for_employee + waiting_in_line

-- Calculate the waiting time for the stocker
def waiting_for_stocker := total_waiting_time - total_known_waiting_time

-- Prove that the waiting time for the stocker is 14 minutes
theorem waiting_for_stocker_proof : waiting_for_stocker = 14 := by
  -- Here the proof steps would normally be included
  sorry

end waiting_for_stocker_proof_l220_220856


namespace solve_inequality_l220_220838

namespace InequalityProof

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem solve_inequality (x : ℝ) : cube_root x + 3 / (cube_root x + 4) ≤ 0 ↔ x ∈ Set.Icc (-27 : ℝ) (-1 : ℝ) :=
by
  have y_eq := cube_root x
  sorry

end InequalityProof

end solve_inequality_l220_220838


namespace problem_proof_l220_220552

theorem problem_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = ab → a + 4 * b = 9) ∧
  (a + b = 1 → ∀ a b,  2^a + 2^(b + 1) ≥ 4) ∧
  (a + b = ab → 1 / a^2 + 2 / b^2 = 2 / 3) ∧
  (a + b = 1 → ∀ a b,  2 * a / (a + b^2) + b / (a^2 + b) = (2 * Real.sqrt 3 / 3) + 1) :=
by
  sorry

end problem_proof_l220_220552


namespace average_time_relay_race_l220_220814

theorem average_time_relay_race :
  let dawson_time := 38
  let henry_time := 7
  let total_legs := 2
  (dawson_time + henry_time) / total_legs = 22.5 :=
by
  sorry

end average_time_relay_race_l220_220814


namespace tan_double_angle_l220_220489

theorem tan_double_angle (α : ℝ) (h1 : α > 0) (h2 : α < Real.pi)
  (h3 : Real.cos α + Real.sin α = -1 / 5) : Real.tan (2 * α) = -24 / 7 :=
by
  sorry

end tan_double_angle_l220_220489


namespace part1_part2_l220_220505

theorem part1 (a : ℝ) (h1 : ∀ x y, y = a * x + 1 → 3 * x^2 - y^2 = 1) (h2 : ∃ x1 y1 x2 y2 : ℝ, y1 = a * x1 + 1 ∧ y2 = a * x2 + 1 ∧ 3 * x1 * x1 - y1 * y1 = 1 ∧ 3 * x2 * x2 - y2 * y2 = 1 ∧ x1 * x2 + (a * x1 + 1) * (a * x2 + 1) = 0) : a = 1 ∨ a = -1 := sorry

theorem part2 (h : ∀ x y, y = a * x + 1 → 3 * x^2 - y^2 = 1) (a : ℝ) (h2 : ∃ x1 y1 x2 y2 : ℝ, y1 = a * x1 + 1 ∧ y2 = a * x2 + 1 ∧ 3 * x1 * x1 - y1 * y1 = 1 ∧ 3 * x2 * x2 - y2 * y2 = 1 ∧ (y1 + y2) / 2 = (1 / 2) * (x1 + x2) / 2 ∧ (y1 - y2) / (x1 - x2) = -2) : false := sorry

end part1_part2_l220_220505


namespace expected_value_coin_flip_l220_220306

-- Define the conditions
def probability_heads := 2 / 3
def probability_tails := 1 / 3
def gain_heads := 5
def loss_tails := -10

-- Define the expected value calculation
def expected_value := (probability_heads * gain_heads) + (probability_tails * loss_tails)

-- Prove that the expected value is 0.00
theorem expected_value_coin_flip : expected_value = 0 := 
by sorry

end expected_value_coin_flip_l220_220306


namespace sin_30_eq_one_half_cos_11pi_over_4_eq_neg_sqrt2_over_2_l220_220476

theorem sin_30_eq_one_half : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
by 
  -- This is the statement only, the proof will be here
  sorry

theorem cos_11pi_over_4_eq_neg_sqrt2_over_2 : Real.cos (11 * Real.pi / 4) = - Real.sqrt 2 / 2 :=
by 
  -- This is the statement only, the proof will be here
  sorry

end sin_30_eq_one_half_cos_11pi_over_4_eq_neg_sqrt2_over_2_l220_220476


namespace total_amount_l220_220398

theorem total_amount (x y z : ℝ) 
  (hy : y = 0.45 * x) 
  (hz : z = 0.30 * x) 
  (hy_value : y = 54) : 
  x + y + z = 210 := 
by
  sorry

end total_amount_l220_220398


namespace percentage_of_students_choose_harvard_l220_220026

theorem percentage_of_students_choose_harvard
  (total_applicants : ℕ)
  (acceptance_rate : ℝ)
  (students_attend_harvard : ℕ)
  (students_attend_other : ℝ)
  (percentage_attended_harvard : ℝ) :
  total_applicants = 20000 →
  acceptance_rate = 0.05 →
  students_attend_harvard = 900 →
  students_attend_other = 0.10 →
  percentage_attended_harvard = ((students_attend_harvard / (total_applicants * acceptance_rate)) * 100) →
  percentage_attended_harvard = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_of_students_choose_harvard_l220_220026


namespace total_frogs_seen_by_hunter_l220_220495

/-- Hunter saw 5 frogs sitting on lily pads in the pond. -/
def initial_frogs : ℕ := 5

/-- Three more frogs climbed out of the water onto logs floating in the pond. -/
def frogs_on_logs : ℕ := 3

/-- Two dozen baby frogs (24 frogs) hopped onto a big rock jutting out from the pond. -/
def baby_frogs : ℕ := 24

/--
The total number of frogs Hunter saw in the pond.
-/
theorem total_frogs_seen_by_hunter : initial_frogs + frogs_on_logs + baby_frogs = 32 := by
sorry

end total_frogs_seen_by_hunter_l220_220495


namespace num_math_books_l220_220358

theorem num_math_books (total_books total_cost math_book_cost history_book_cost : ℕ) (M H : ℕ)
  (h1 : total_books = 80)
  (h2 : math_book_cost = 4)
  (h3 : history_book_cost = 5)
  (h4 : total_cost = 368)
  (h5 : M + H = total_books)
  (h6 : math_book_cost * M + history_book_cost * H = total_cost) :
  M = 32 :=
by
  sorry

end num_math_books_l220_220358


namespace books_combination_l220_220328

theorem books_combination : (Nat.choose 15 3) = 455 := by
  sorry

end books_combination_l220_220328


namespace xiaoming_additional_games_l220_220660

variable (total_games games_won target_percentage : ℕ)

theorem xiaoming_additional_games :
  total_games = 20 →
  games_won = 95 * total_games / 100 →
  target_percentage = 96 →
  ∃ additional_games, additional_games = 5 ∧
    (games_won + additional_games) / (total_games + additional_games) = target_percentage / 100 :=
by
  sorry

end xiaoming_additional_games_l220_220660


namespace min_value_of_trig_expression_l220_220197

open Real

theorem min_value_of_trig_expression (α : ℝ) (h₁ : sin α ≠ 0) (h₂ : cos α ≠ 0) : 
  (9 / (sin α)^2 + 1 / (cos α)^2) ≥ 16 :=
  sorry

end min_value_of_trig_expression_l220_220197


namespace ratio_condition_l220_220833

theorem ratio_condition (x y a b : ℝ) (h1 : 8 * x - 6 * y = a) 
  (h2 : 9 * y - 12 * x = b) (hx : x ≠ 0) (hy : y ≠ 0) (hb : b ≠ 0) : 
  a / b = -2 / 3 := 
by
  sorry

end ratio_condition_l220_220833


namespace value_of_a_l220_220370

theorem value_of_a (a x : ℝ) (h : x = 4) (h_eq : x^2 - 3 * x = a^2) : a = 2 ∨ a = -2 :=
by
  -- The proof is omitted, but the theorem statement adheres to the problem conditions and expected result.
  sorry

end value_of_a_l220_220370


namespace maximum_rectangle_area_l220_220914

theorem maximum_rectangle_area (P : ℝ) (hP : P = 36) :
  ∃ (A : ℝ), A = (P / 4) * (P / 4) :=
by
  use 81
  sorry

end maximum_rectangle_area_l220_220914


namespace men_build_wall_l220_220933

theorem men_build_wall (k : ℕ) (h1 : 20 * 6 = k) : ∃ d : ℝ, (30 * d = k) ∧ d = 4.0 := by
  sorry

end men_build_wall_l220_220933


namespace second_machine_equation_l220_220899

-- Let p1_rate and p2_rate be the rates of printing for machine 1 and 2 respectively.
-- Let x be the unknown time for the second machine to print 500 envelopes.

theorem second_machine_equation (x : ℝ) :
    (500 / 8) + (500 / x) = (500 / 2) :=
  sorry

end second_machine_equation_l220_220899


namespace find_integer_solutions_l220_220729

theorem find_integer_solutions (n : ℕ) (h1 : ∃ b : ℤ, 8 * n - 7 = b^2) (h2 : ∃ a : ℤ, 18 * n - 35 = a^2) : 
  n = 2 ∨ n = 22 := 
sorry

end find_integer_solutions_l220_220729


namespace part1_problem_part2_problem_l220_220110

/-- Given initial conditions and price adjustment, prove the expected number of helmets sold and the monthly profit. -/
theorem part1_problem (initial_price : ℕ) (initial_sales : ℕ) 
(price_reduction : ℕ) (sales_per_reduction : ℕ) (cost_price : ℕ) : 
  initial_price = 80 → initial_sales = 200 → price_reduction = 10 → 
  sales_per_reduction = 20 → cost_price = 50 → 
  (initial_sales + price_reduction * sales_per_reduction = 400) ∧ 
  ((initial_price - price_reduction - cost_price) * 
  (initial_sales + price_reduction * sales_per_reduction) = 8000) :=
by
  intros
  sorry

/-- Given initial conditions and profit target, prove the expected selling price of helmets. -/
theorem part2_problem (initial_price : ℕ) (initial_sales : ℕ) 
(cost_price : ℕ) (profit_target : ℕ) (x : ℕ) :
  initial_price = 80 → initial_sales = 200 → cost_price = 50 → 
  profit_target = 7500 → (x = 15) → 
  (initial_price - x = 65) :=
by
  intros
  sorry

end part1_problem_part2_problem_l220_220110


namespace circles_intersect_l220_220389

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*x - 4*y - 1 = 0

theorem circles_intersect : 
  (∃ (x y : ℝ), circle1 x y ∧ circle2 x y) := 
sorry

end circles_intersect_l220_220389


namespace electricity_usage_A_B_l220_220376

def electricity_cost (x : ℕ) : ℝ :=
  if h₁ : 0 ≤ x ∧ x ≤ 24 then 4.2 * x
  else if h₂ : 24 < x ∧ x ≤ 60 then 5.2 * x - 24
  else if h₃ : 60 < x ∧ x ≤ 100 then 6.6 * x - 108
  else if h₄ : 100 < x ∧ x ≤ 150 then 7.6 * x - 208
  else if h₅ : 150 < x ∧ x ≤ 250 then 8 * x - 268
  else 8.4 * x - 368

theorem electricity_usage_A_B (x : ℕ) (h : electricity_cost x = 486) :
  60 < x ∧ x ≤ 100 ∧ 5 * x = 450 ∧ 2 * x = 180 :=
by
  sorry

end electricity_usage_A_B_l220_220376


namespace Dorothy_found_57_pieces_l220_220576

def total_pieces_Dorothy_found 
  (B_green B_red R_red R_blue : ℕ)
  (D_red_factor D_blue_factor : ℕ)
  (H1 : B_green = 12)
  (H2 : B_red = 3)
  (H3 : R_red = 9)
  (H4 : R_blue = 11)
  (H5 : D_red_factor = 2)
  (H6 : D_blue_factor = 3) : ℕ := 
  let D_red := D_red_factor * (B_red + R_red)
  let D_blue := D_blue_factor * R_blue
  D_red + D_blue

theorem Dorothy_found_57_pieces 
  (B_green B_red R_red R_blue : ℕ)
  (D_red_factor D_blue_factor : ℕ)
  (H1 : B_green = 12)
  (H2 : B_red = 3)
  (H3 : R_red = 9)
  (H4 : R_blue = 11)
  (H5 : D_red_factor = 2)
  (H6 : D_blue_factor = 3) :
  total_pieces_Dorothy_found B_green B_red R_red R_blue D_red_factor D_blue_factor H1 H2 H3 H4 H5 H6 = 57 := by
    sorry

end Dorothy_found_57_pieces_l220_220576


namespace line_equation_l220_220700

theorem line_equation (x y : ℝ) : 
  (3 * x + y = 0) ∧ (x + y - 2 = 0) ∧ 
  ∃ m : ℝ, -2 = -(1 / m) ∧ 
  (∃ b : ℝ, (y = m * x + b) ∧ (3 = m * (-1) + b)) ∧ 
  x - 2 * y + 7 = 0 :=
sorry

end line_equation_l220_220700


namespace river_depth_mid_July_l220_220965

theorem river_depth_mid_July :
  let d_May := 5
  let d_June := d_May + 10
  let d_July := 3 * d_June
  d_July = 45 :=
by
  sorry

end river_depth_mid_July_l220_220965


namespace cost_of_fencing_is_8750_rsquare_l220_220221

variable (l w : ℝ)
variable (area : ℝ := 7500)
variable (cost_per_meter : ℝ := 0.25)
variable (ratio_lw : ℝ := 4/3)

theorem cost_of_fencing_is_8750_rsquare :
  (l / w = ratio_lw) → 
  (l * w = area) → 
  (2 * (l + w) * cost_per_meter = 87.50) :=
by 
  intros h1 h2
  sorry

end cost_of_fencing_is_8750_rsquare_l220_220221


namespace compare_expression_l220_220609

variable (m x : ℝ)

theorem compare_expression : x^2 - x + 1 > -2 * m^2 - 2 * m * x := 
sorry

end compare_expression_l220_220609


namespace middle_term_arithmetic_sequence_l220_220461

theorem middle_term_arithmetic_sequence (m : ℝ) (h : 2 * m = 1 + 5) : m = 3 :=
by
  sorry

end middle_term_arithmetic_sequence_l220_220461


namespace sum_of_possible_values_of_G_F_l220_220936

theorem sum_of_possible_values_of_G_F (G F : ℕ) (hG : 0 ≤ G ∧ G ≤ 9) (hF : 0 ≤ F ∧ F ≤ 9)
  (hdiv : (G + 2 + 4 + 3 + F + 1 + 6) % 9 = 0) : G + F = 2 ∨ G + F = 11 → 2 + 11 = 13 :=
by { sorry }

end sum_of_possible_values_of_G_F_l220_220936


namespace units_digit_odd_product_l220_220270

theorem units_digit_odd_product (l : List ℕ) (h_odds : ∀ n ∈ l, n % 2 = 1) :
  (∀ x ∈ l, x % 10 = 5) ↔ (5 ∈ l) := by
  sorry

end units_digit_odd_product_l220_220270


namespace p_sq_plus_q_sq_l220_220542

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 := by
  sorry

end p_sq_plus_q_sq_l220_220542


namespace triangle_area_difference_l220_220632

-- Definitions per conditions
def right_angle (A B C : Type) (angle_EAB : Prop) : Prop := angle_EAB
def angle_ABC_eq_30 (A B C : Type) (angle_ABC : ℝ) : Prop := angle_ABC = 30
def length_AB_eq_5 (A B : Type) (AB : ℝ) : Prop := AB = 5
def length_BC_eq_7 (B C : Type) (BC : ℝ) : Prop := BC = 7
def length_AE_eq_10 (A E : Type) (AE : ℝ) : Prop := AE = 10
def lines_intersect_at_D (A B C E D : Type) (intersects : Prop) : Prop := intersects

-- Main theorem statement
theorem triangle_area_difference
  (A B C E D : Type)
  (angle_EAB : Prop)
  (right_EAB : right_angle A E B angle_EAB)
  (angle_ABC : ℝ)
  (angle_ABC_is_30 : angle_ABC_eq_30 A B C angle_ABC)
  (AB : ℝ)
  (AB_is_5 : length_AB_eq_5 A B AB)
  (BC : ℝ)
  (BC_is_7 : length_BC_eq_7 B C BC)
  (AE : ℝ)
  (AE_is_10 : length_AE_eq_10 A E AE)
  (intersects : Prop)
  (intersects_at_D : lines_intersect_at_D A B C E D intersects) :
  (area_ADE - area_BDC) = 16.25 := sorry

end triangle_area_difference_l220_220632


namespace selection_methods_count_l220_220181

-- Define a function to compute combinations (n choose r)
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem statement
theorem selection_methods_count :
  combination 5 2 * combination 3 1 * combination 2 1 = 60 :=
by
  sorry

end selection_methods_count_l220_220181


namespace negation_of_existence_l220_220402

theorem negation_of_existence (h : ¬ (∃ x : ℝ, x^2 - x - 1 > 0)) : ∀ x : ℝ, x^2 - x - 1 ≤ 0 :=
sorry

end negation_of_existence_l220_220402


namespace quadratic_distinct_roots_l220_220254

theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0 ∧ x ≠ y) ↔ k > -1 ∧ k ≠ 0 := 
sorry

end quadratic_distinct_roots_l220_220254


namespace find_real_solutions_l220_220100

theorem find_real_solutions : 
  ∀ x : ℝ, 1 / ((x - 2) * (x - 3)) 
         + 1 / ((x - 3) * (x - 4)) 
         + 1 / ((x - 4) * (x - 5)) 
         = 1 / 8 ↔ x = 7 ∨ x = -2 :=
by
  intro x
  sorry

end find_real_solutions_l220_220100


namespace convert_mixed_decimals_to_fractions_l220_220673

theorem convert_mixed_decimals_to_fractions :
  (4.26 = 4 + 13/50) ∧
  (1.15 = 1 + 3/20) ∧
  (3.08 = 3 + 2/25) ∧
  (2.37 = 2 + 37/100) :=
by
  -- Proof omitted
  sorry

end convert_mixed_decimals_to_fractions_l220_220673


namespace number_of_plains_routes_is_81_l220_220846

-- Define the number of cities in each region
def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70

-- Define the number of routes established over three years
def total_routes : ℕ := 150
def routes_per_year : ℕ := 50

-- Define the number of routes connecting pairs of mountainous cities
def mountainous_routes : ℕ := 21

-- Define a function to calculate the number of routes connecting pairs of plains cities
def plains_routes : ℕ :=
  let total_endpoints := total_routes * 2
  let mountainous_endpoints := mountainous_cities * 3
  let plains_endpoints := plains_cities * 3
  let mountainous_pair_endpoints := mountainous_routes * 2
  let mountain_plain_routes := (mountainous_endpoints - mountainous_pair_endpoints) / 2
  let plain_only_endpoints := plains_endpoints - mountain_plain_routes
  plain_only_endpoints / 2

theorem number_of_plains_routes_is_81 : plains_routes = 81 := 
  sorry

end number_of_plains_routes_is_81_l220_220846


namespace x_squared_plus_y_squared_l220_220090

theorem x_squared_plus_y_squared (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
(h3 : x * y + x + y = 71)
(h4 : x^2 * y + x * y^2 = 880) :
x^2 + y^2 = 146 :=
sorry

end x_squared_plus_y_squared_l220_220090


namespace value_of_3a_plus_6b_l220_220775

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2 * b = 1) : 3 * a + 6 * b = 3 :=
sorry

end value_of_3a_plus_6b_l220_220775


namespace quadratic_difference_square_l220_220681

theorem quadratic_difference_square (α β : ℝ) (h : α ≠ β) (hα : α^2 - 3 * α + 1 = 0) (hβ : β^2 - 3 * β + 1 = 0) : (α - β)^2 = 5 := by
  sorry

end quadratic_difference_square_l220_220681


namespace value_of_X_l220_220276

theorem value_of_X (X : ℝ) (h : ((X + 0.064)^2 - (X - 0.064)^2) / (X * 0.064) = 4.000000000000002) : X ≠ 0 :=
sorry

end value_of_X_l220_220276


namespace calculation_power_l220_220715

theorem calculation_power :
  (0.125 : ℝ) ^ 2012 * (2 ^ 2012) ^ 3 = 1 :=
sorry

end calculation_power_l220_220715


namespace quadratic_real_roots_l220_220019

theorem quadratic_real_roots (k : ℝ) (h : ∀ x : ℝ, k * x^2 - 4 * x + 1 = 0) : k ≤ 4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_real_roots_l220_220019


namespace lee_charged_per_action_figure_l220_220995

theorem lee_charged_per_action_figure :
  ∀ (sneakers_cost savings action_figures leftovers price_per_fig),
    sneakers_cost = 90 →
    savings = 15 →
    action_figures = 10 →
    leftovers = 25 →
    price_per_fig = 10 →
    (savings + action_figures * price_per_fig) - sneakers_cost = leftovers → price_per_fig = 10 :=
by
  intros sneakers_cost savings action_figures leftovers price_per_fig
  intros h_sneakers_cost h_savings h_action_figures h_leftovers h_price_per_fig
  intros h_total
  sorry

end lee_charged_per_action_figure_l220_220995


namespace option_C_sets_same_l220_220020

-- Define the sets for each option
def option_A_set_M : Set (ℕ × ℕ) := {(3, 2)}
def option_A_set_N : Set (ℕ × ℕ) := {(2, 3)}

def option_B_set_M : Set (ℕ × ℕ) := {p | p.1 + p.2 = 1}
def option_B_set_N : Set ℕ := { y | ∃ x, x + y = 1 }

def option_C_set_M : Set ℕ := {4, 5}
def option_C_set_N : Set ℕ := {5, 4}

def option_D_set_M : Set ℕ := {1, 2}
def option_D_set_N : Set (ℕ × ℕ) := {(1, 2)}

-- Prove that option C sets represent the same set
theorem option_C_sets_same : option_C_set_M = option_C_set_N := by
  sorry

end option_C_sets_same_l220_220020


namespace soda_difference_l220_220991

theorem soda_difference :
  let Julio_orange_bottles := 4
  let Julio_grape_bottles := 7
  let Mateo_orange_bottles := 1
  let Mateo_grape_bottles := 3
  let liters_per_bottle := 2
  let Julio_total_liters := Julio_orange_bottles * liters_per_bottle + Julio_grape_bottles * liters_per_bottle
  let Mateo_total_liters := Mateo_orange_bottles * liters_per_bottle + Mateo_grape_bottles * liters_per_bottle
  Julio_total_liters - Mateo_total_liters = 14 := by
    sorry

end soda_difference_l220_220991


namespace angle_BDC_proof_l220_220545

noncomputable def angle_sum_triangle (angle_A angle_B angle_C : ℝ) : Prop :=
  angle_A + angle_B + angle_C = 180

-- Given conditions
def angle_A : ℝ := 70
def angle_E : ℝ := 50
def angle_C : ℝ := 40

-- The problem of proving that angle_BDC = 20 degrees
theorem angle_BDC_proof (A E C BDC : ℝ) 
  (hA : A = angle_A)
  (hE : E = angle_E)
  (hC : C = angle_C) :
  BDC = 20 :=
  sorry

end angle_BDC_proof_l220_220545


namespace grill_burns_fifteen_coals_in_twenty_minutes_l220_220117

-- Define the problem conditions
def total_coals (bags : ℕ) (coals_per_bag : ℕ) : ℕ :=
  bags * coals_per_bag

def burning_ratio (total_coals : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / total_coals

-- Given conditions
def bags := 3
def coals_per_bag := 60
def total_minutes := 240
def fifteen_coals := 15

-- Problem statement
theorem grill_burns_fifteen_coals_in_twenty_minutes :
  total_minutes / total_coals bags coals_per_bag * fifteen_coals = 20 :=
by
  sorry

end grill_burns_fifteen_coals_in_twenty_minutes_l220_220117


namespace sides_of_figures_intersection_l220_220212

theorem sides_of_figures_intersection (n p q : ℕ) (h1 : p ≠ 0) (h2 : q ≠ 0) :
  p + q ≤ n + 4 :=
by sorry

end sides_of_figures_intersection_l220_220212


namespace isosceles_trapezoid_perimeter_l220_220957

/-- In an isosceles trapezoid ABCD with bases AB = 10 units and CD = 18 units, 
and height from AB to CD is 4 units, the perimeter of ABCD is 28 + 8 * sqrt(2) units. -/
theorem isosceles_trapezoid_perimeter :
  ∃ (A B C D : Type) (AB CD AD BC h : ℝ), 
      AB = 10 ∧ 
      CD = 18 ∧ 
      AD = BC ∧ 
      h = 4 →
      ∀ (P : ℝ), P = AB + BC + CD + DA → 
      P = 28 + 8 * Real.sqrt 2 :=
by
  sorry

end isosceles_trapezoid_perimeter_l220_220957


namespace find_f_10_l220_220000

def f (x : Int) : Int := sorry

axiom condition_1 : f 1 + 1 > 0
axiom condition_2 : ∀ x y : Int, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom condition_3 : ∀ x : Int, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := by
  sorry

end find_f_10_l220_220000


namespace second_cat_weight_l220_220287

theorem second_cat_weight :
  ∀ (w1 w2 w3 w_total : ℕ), 
    w1 = 2 ∧ w3 = 4 ∧ w_total = 13 → 
    w_total = w1 + w2 + w3 → 
    w2 = 7 :=
by
  sorry

end second_cat_weight_l220_220287


namespace calculation_correct_l220_220906

-- Defining the initial values
def a : ℕ := 20 ^ 10
def b : ℕ := 20 ^ 9
def c : ℕ := 10 ^ 6
def d : ℕ := 2 ^ 12

-- The expression we need to prove
theorem calculation_correct : ((a / b) ^ 3 * c) / d = 1953125 :=
by
  sorry

end calculation_correct_l220_220906


namespace find_value_of_abc_cubed_l220_220760

-- Variables and conditions
variables {a b c : ℝ}
variables (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = a^4 + b^4 + c^4)

-- The statement
theorem find_value_of_abc_cubed (ha : a ≠ 0) (hb: b ≠ 0) (hc: c ≠ 0) :
  a^3 + b^3 + c^3 = -3 * a * b * (a + b) :=
by
  sorry

end find_value_of_abc_cubed_l220_220760


namespace middle_group_frequency_l220_220718

theorem middle_group_frequency (sample_size : ℕ) (num_rectangles : ℕ)
  (A_middle : ℝ) (other_area_sum : ℝ)
  (h1 : sample_size = 300)
  (h2 : num_rectangles = 9)
  (h3 : A_middle = 1 / 5 * other_area_sum)
  (h4 : other_area_sum + A_middle = 1) :
  sample_size * A_middle = 50 :=
by
  sorry

end middle_group_frequency_l220_220718


namespace smallest_product_of_digits_l220_220076

theorem smallest_product_of_digits : 
  ∃ (a b c d : ℕ), 
  (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6) ∧ 
  (∃ x y : ℕ, (x = a * 10 + c ∧ y = b * 10 + d) ∨ (x = a * 10 + d ∧ y = b * 10 + c) ∨ (x = b * 10 + c ∧ y = a * 10 + d) ∨ (x = b * 10 + d ∧ y = a * 10 + c)) ∧
  (∀ x1 y1 x2 y2 : ℕ, ((x1 = 34 ∧ y1 = 56 ∨ x1 = 35 ∧ y1 = 46) ∧ (x2 = 34 ∧ y2 = 56 ∨ x2 = 35 ∧ y2 = 46)) → x1 * y1 ≥ x2 * y2) ∧
  35 * 46 = 1610 :=
sorry

end smallest_product_of_digits_l220_220076


namespace find_y_given_x_l220_220566

-- Let x and y be real numbers
variables (x y : ℝ)

-- Assume x and y are inversely proportional, so their product is a constant C
variable (C : ℝ)

-- Additional conditions from the problem statement
variable (h1 : x + y = 40) (h2 : x - y = 10) (hx : x = 7)

-- Define the goal: y = 375 / 7
theorem find_y_given_x : y = 375 / 7 :=
sorry

end find_y_given_x_l220_220566


namespace distance_between_locations_l220_220172

theorem distance_between_locations
  (d_AC d_BC : ℚ)
  (d : ℚ)
  (meet_C : d_AC + d_BC = d)
  (travel_A_B : 150 + 150 + 540 = 840)
  (distance_ratio : 840 / 540 = 14 / 9)
  (distance_ratios : d_AC / d_BC = 14 / 9)
  (C_D : 540 = 5 * d / 23) :
  d = 2484 :=
by
  sorry

end distance_between_locations_l220_220172


namespace bee_fraction_remaining_l220_220097

theorem bee_fraction_remaining (N : ℕ) (L : ℕ) (D : ℕ) (hN : N = 80000) (hL : L = 1200) (hD : D = 50) :
  (N - (L * D)) / N = 1 / 4 :=
by
  sorry

end bee_fraction_remaining_l220_220097


namespace compute_expr_l220_220883

theorem compute_expr : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expr_l220_220883


namespace point_not_in_region_l220_220024

theorem point_not_in_region (A B C D : ℝ × ℝ) :
  (A = (0, 0) ∧ 3 * A.1 + 2 * A.2 < 6) ∧
  (B = (1, 1) ∧ 3 * B.1 + 2 * B.2 < 6) ∧
  (C = (0, 2) ∧ 3 * C.1 + 2 * C.2 < 6) ∧
  (D = (2, 0) ∧ ¬ ( 3 * D.1 + 2 * D.2 < 6 )) :=
by {
  sorry
}

end point_not_in_region_l220_220024


namespace max_red_dominated_rows_plus_blue_dominated_columns_l220_220155

-- Definitions of the problem conditions and statement
theorem max_red_dominated_rows_plus_blue_dominated_columns (m n : ℕ)
  (h1 : Odd m) (h2 : Odd n) (h3 : 0 < m ∧ 0 < n) :
  ∃ A : Finset (Fin m) × Finset (Fin n),
  (A.1.card + A.2.card = m + n - 2) :=
sorry

end max_red_dominated_rows_plus_blue_dominated_columns_l220_220155


namespace exists_triangle_perimeter_lt_1cm_circumradius_gt_1km_l220_220498

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem exists_triangle_perimeter_lt_1cm_circumradius_gt_1km :
  ∃ (A B C : ℝ) (a b c : ℝ), a + b + c < 0.01 ∧ circumradius a b c > 1000 :=
by
  sorry

end exists_triangle_perimeter_lt_1cm_circumradius_gt_1km_l220_220498


namespace solve_equation_nat_numbers_l220_220998

theorem solve_equation_nat_numbers (a b : ℕ) (h : (a, b) = (11, 170) ∨ (a, b) = (22, 158) ∨ (a, b) = (33, 146) ∨
                                    (a, b) = (44, 134) ∨ (a, b) = (55, 122) ∨ (a, b) = (66, 110) ∨
                                    (a, b) = (77, 98) ∨ (a, b) = (88, 86) ∨ (a, b) = (99, 74) ∨
                                    (a, b) = (110, 62) ∨ (a, b) = (121, 50) ∨ (a, b) = (132, 38) ∨
                                    (a, b) = (143, 26) ∨ (a, b) = (154, 14) ∨ (a, b) = (165, 2)) :
  12 * a + 11 * b = 2002 :=
by
  sorry

end solve_equation_nat_numbers_l220_220998


namespace problem_statement_l220_220343

def complex_number (m : ℂ) : ℂ :=
  (m^2 - 3*m - 4) + (m^2 - 5*m - 6) * Complex.I

theorem problem_statement (m : ℂ) :
  (complex_number m).im = m^2 - 5*m - 6 →
  (complex_number m).re = 0 →
  m ≠ -1 ∧ m ≠ 6 :=
by
  sorry

end problem_statement_l220_220343


namespace find_p_l220_220966

theorem find_p (m n p : ℝ)
  (h1 : m = 5 * n + 5)
  (h2 : m + 2 = 5 * (n + p) + 5) :
  p = 2 / 5 :=
by
  sorry

end find_p_l220_220966


namespace decorations_left_to_put_up_l220_220743

variable (S B W P C T : Nat)
variable (h₁ : S = 12)
variable (h₂ : B = 4)
variable (h₃ : W = 12)
variable (h₄ : P = 2 * W)
variable (h₅ : C = 1)
variable (h₆ : T = 83)

theorem decorations_left_to_put_up (h₁ : S = 12) (h₂ : B = 4) (h₃ : W = 12) (h₄ : P = 2 * W) (h₅ : C = 1) (h₆ : T = 83) :
  T - (S + B + W + P + C) = 30 := sorry

end decorations_left_to_put_up_l220_220743


namespace correct_option_D_l220_220728

theorem correct_option_D (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by sorry

end correct_option_D_l220_220728


namespace arrangement_possible_l220_220379

noncomputable def exists_a_b : Prop :=
  ∃ a b : ℝ, a + 2*b > 0 ∧ 7*a + 13*b < 0

theorem arrangement_possible : exists_a_b := by
  sorry

end arrangement_possible_l220_220379


namespace base3_to_base10_equiv_l220_220140

theorem base3_to_base10_equiv : 
  let repr := 1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  repr = 142 :=
by
  sorry

end base3_to_base10_equiv_l220_220140


namespace hyperbola_eccentricity_l220_220581

theorem hyperbola_eccentricity
    (a b e : ℝ)
    (ha : a > 0)
    (hb : b > 0)
    (h_hyperbola : ∀ x y, x ^ 2 / a^2 - y^2 / b^2 = 1)
    (h_circle : ∀ x y, (x - 2) ^ 2 + y ^ 2 = 4)
    (h_chord_length : ∀ x y, (x ^ 2 + y ^ 2)^(1/2) = 2) :
    e = 2 := 
sorry

end hyperbola_eccentricity_l220_220581


namespace find_m_pure_imaginary_l220_220079

noncomputable def find_m (m : ℝ) : ℝ := m

theorem find_m_pure_imaginary (m : ℝ) (h : (m^2 - 5 * m + 6 : ℂ) = 0) :
  find_m m = 2 :=
by
  sorry

end find_m_pure_imaginary_l220_220079


namespace decreasing_function_range_l220_220413

theorem decreasing_function_range {f : ℝ → ℝ} (h_decreasing : ∀ x y : ℝ, x < y → f x > f y) :
  {x : ℝ | f (x^2 - 3 * x - 3) < f 1} = {x : ℝ | x < -1 ∨ x > 4} :=
by
  sorry

end decreasing_function_range_l220_220413


namespace gumball_machine_l220_220063

variable (R B G Y O : ℕ)

theorem gumball_machine : 
  (B = (1 / 2) * R) ∧
  (G = 4 * B) ∧
  (Y = (7 / 2) * B) ∧
  (O = (2 / 3) * (R + B)) ∧
  (R = (3 / 2) * Y) ∧
  (Y = 24) →
  (R + B + G + Y + O = 186) :=
sorry

end gumball_machine_l220_220063


namespace ratio_of_green_to_yellow_l220_220180

def envelopes_problem (B Y G X : ℕ) : Prop :=
  B = 14 ∧
  Y = B - 6 ∧
  G = X * Y ∧
  B + Y + G = 46 ∧
  G / Y = 3

theorem ratio_of_green_to_yellow :
  ∃ B Y G X : ℕ, envelopes_problem B Y G X :=
by
  sorry

end ratio_of_green_to_yellow_l220_220180


namespace c_is_younger_l220_220337

variables (a b c d : ℕ) -- assuming ages as natural numbers

-- Conditions
axiom cond1 : a + b = b + c + 12
axiom cond2 : b + d = c + d + 8
axiom cond3 : d = a + 5

-- Question
theorem c_is_younger : c = a - 12 :=
sorry

end c_is_younger_l220_220337


namespace calculate_v3_l220_220147

def f (x : ℤ) : ℤ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def v0 : ℤ := 2
def v1 (x : ℤ) : ℤ := v0 * x + 5
def v2 (x : ℤ) : ℤ := v1 x * x + 6
def v3 (x : ℤ) : ℤ := v2 x * x + 23

theorem calculate_v3 : v3 (-4) = -49 :=
by
sorry

end calculate_v3_l220_220147


namespace sum_of_abc_l220_220691

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (eq1 : a^2 + b * c = 115) (eq2 : b^2 + a * c = 127) (eq3 : c^2 + a * b = 115) :
  a + b + c = 22 := by
  sorry

end sum_of_abc_l220_220691


namespace tank_capacity_l220_220153

theorem tank_capacity (T : ℝ) (h1 : T * (4 / 5) - T * (5 / 8) = 15) : T = 86 :=
by
  sorry

end tank_capacity_l220_220153


namespace ones_digit_of_4567_times_3_is_1_l220_220248

theorem ones_digit_of_4567_times_3_is_1 :
  let n := 4567
  let m := 3
  (n * m) % 10 = 1 :=
by
  let n := 4567
  let m := 3
  have h : (n * m) % 10 = ((4567 * 3) % 10) := by rfl -- simplifying the product
  sorry -- this is where the proof would go, if required

end ones_digit_of_4567_times_3_is_1_l220_220248


namespace cost_price_per_meter_l220_220650

theorem cost_price_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (total_cost_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : total_meters = 400)
  (h2 : selling_price = 18000)
  (h3 : loss_per_meter = 5)
  (h4 : total_cost_price = selling_price + total_meters * loss_per_meter)
  (h5 : cost_price_per_meter = total_cost_price / total_meters) :
  cost_price_per_meter = 50 :=
by
  sorry

end cost_price_per_meter_l220_220650


namespace money_left_in_wallet_l220_220400

def initial_amount := 106
def spent_supermarket := 31
def spent_showroom := 49

theorem money_left_in_wallet : initial_amount - spent_supermarket - spent_showroom = 26 := by
  sorry

end money_left_in_wallet_l220_220400


namespace daily_rental_cost_l220_220854

theorem daily_rental_cost
  (daily_rent : ℝ)
  (cost_per_mile : ℝ)
  (max_budget : ℝ)
  (miles : ℝ)
  (H1 : cost_per_mile = 0.18)
  (H2 : max_budget = 75)
  (H3 : miles = 250)
  (H4 : daily_rent + (cost_per_mile * miles) = max_budget) : daily_rent = 30 :=
by sorry

end daily_rental_cost_l220_220854


namespace value_of_card_l220_220077

/-- For this problem: 
    1. Matt has 8 baseball cards worth $6 each.
    2. He trades two of them to Jane in exchange for 3 $2 cards and a card of certain value.
    3. He makes a profit of $3.
    We need to prove that the value of the card that Jane gave to Matt apart from the $2 cards is $9. -/
theorem value_of_card (value_per_card traded_cards received_dollar_cards profit received_total_value : ℤ)
  (h1 : value_per_card = 6)
  (h2 : traded_cards = 2)
  (h3 : received_dollar_cards = 6)
  (h4 : profit = 3)
  (h5 : received_total_value = 15) :
  received_total_value - received_dollar_cards = 9 :=
by {
  -- This is just left as a placeholder to signal that the proof needs to be provided.
  sorry
}

end value_of_card_l220_220077


namespace p_iff_q_l220_220310

variable (a b : ℝ)

def p := a > 2 ∧ b > 3

def q := a + b > 5 ∧ (a - 2) * (b - 3) > 0

theorem p_iff_q : p a b ↔ q a b := by
  sorry

end p_iff_q_l220_220310


namespace weighted_average_yield_l220_220602

-- Define the conditions
def face_value_A : ℝ := 1000
def market_price_A : ℝ := 1200
def yield_A : ℝ := 0.18

def face_value_B : ℝ := 1000
def market_price_B : ℝ := 800
def yield_B : ℝ := 0.22

def face_value_C : ℝ := 1000
def market_price_C : ℝ := 1000
def yield_C : ℝ := 0.15

def investment_A : ℝ := 5000
def investment_B : ℝ := 3000
def investment_C : ℝ := 2000

-- Prove the weighted average yield
theorem weighted_average_yield :
  (investment_A + investment_B + investment_C) = 10000 →
  ((investment_A / (investment_A + investment_B + investment_C)) * yield_A +
   (investment_B / (investment_A + investment_B + investment_C)) * yield_B +
   (investment_C / (investment_A + investment_B + investment_C)) * yield_C) = 0.186 :=
by
  sorry

end weighted_average_yield_l220_220602


namespace find_c_l220_220763

variable {a b c : ℝ} 
variable (h_perpendicular : (a / 3) * (-3 / b) = -1)
variable (h_intersect1 : 2 * a + 9 = c)
variable (h_intersect2 : 6 - 3 * b = -c)
variable (h_ab_equal : a = b)

theorem find_c : c = 39 := 
by
  sorry

end find_c_l220_220763


namespace complement_union_l220_220701

namespace SetComplement

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {4, 5}
def B : Set ℕ := {3, 4}

theorem complement_union :
  U \ (A ∪ B) = {1, 2, 6} := by
  sorry

end SetComplement

end complement_union_l220_220701


namespace consecutive_integers_sum_l220_220316

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l220_220316


namespace fourth_term_of_geometric_progression_l220_220157

theorem fourth_term_of_geometric_progression (x : ℝ) (r : ℝ) 
  (h1 : (2 * x + 5) = r * x) 
  (h2 : (3 * x + 10) = r * (2 * x + 5)) : 
  (3 * x + 10) * r = -5 :=
by
  sorry

end fourth_term_of_geometric_progression_l220_220157


namespace initial_welders_count_l220_220043

theorem initial_welders_count
  (W : ℕ)
  (complete_in_5_days : W * 5 = 1)
  (leave_after_1_day : 12 ≤ W) 
  (remaining_complete_in_6_days : (W - 12) * 6 = 1) : 
  W = 72 :=
by
  -- proof steps here
  sorry

end initial_welders_count_l220_220043


namespace Marissa_sister_height_l220_220567

theorem Marissa_sister_height (sunflower_height_feet : ℕ) (height_difference_inches : ℕ) :
  sunflower_height_feet = 6 -> height_difference_inches = 21 -> 
  let sunflower_height_inches := sunflower_height_feet * 12
  let sister_height_inches := sunflower_height_inches - height_difference_inches
  let sister_height_feet := sister_height_inches / 12
  let sister_height_remainder_inches := sister_height_inches % 12
  sister_height_feet = 4 ∧ sister_height_remainder_inches = 3 :=
by
  intros
  sorry

end Marissa_sister_height_l220_220567


namespace time_to_cross_bridge_l220_220606

-- Defining the given conditions
def length_of_train : ℕ := 110
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 140

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

-- Calculating the speed in m/s
def speed_of_train_ms : ℚ := kmh_to_ms speed_of_train_kmh

-- Calculating total distance to be covered
def total_distance : ℕ := length_of_train + length_of_bridge

-- Expected time to cross the bridge
def expected_time : ℚ := total_distance / speed_of_train_ms

-- The proof statement
theorem time_to_cross_bridge :
  expected_time = 12.5 := by
  sorry

end time_to_cross_bridge_l220_220606


namespace not_square_a2_b2_ab_l220_220528

theorem not_square_a2_b2_ab (n : ℕ) (h_n : n > 2) (a : ℕ) (b : ℕ) (h_b : b = 2^(2^n))
  (h_a_odd : a % 2 = 1) (h_a_le_b : a ≤ b) (h_b_le_2a : b ≤ 2 * a) :
  ¬ ∃ k : ℕ, a^2 + b^2 - a * b = k^2 :=
by
  sorry

end not_square_a2_b2_ab_l220_220528


namespace steven_total_seeds_l220_220812

-- Definitions based on the conditions
def apple_seed_count := 6
def pear_seed_count := 2
def grape_seed_count := 3

def apples_set_aside := 4
def pears_set_aside := 3
def grapes_set_aside := 9

def additional_seeds_needed := 3

-- The total seeds Steven already has
def total_seeds_from_fruits : ℕ :=
  apples_set_aside * apple_seed_count +
  pears_set_aside * pear_seed_count +
  grapes_set_aside * grape_seed_count

-- The total number of seeds Steven needs to collect, as given by the problem's solution
def total_seeds_needed : ℕ :=
  total_seeds_from_fruits + additional_seeds_needed

-- The actual proof statement
theorem steven_total_seeds : total_seeds_needed = 60 :=
  by
    sorry

end steven_total_seeds_l220_220812


namespace two_bacteria_fill_time_l220_220307

-- Define the conditions
def one_bacterium_fills_bottle_in (a : Nat) (t : Nat) : Prop :=
  (2^t = 2^a)

def two_bacteria_fill_bottle_in (a : Nat) (x : Nat) : Prop :=
  (2 * 2^x = 2^a)

-- State the theorem
theorem two_bacteria_fill_time (a : Nat) : ∃ x, two_bacteria_fill_bottle_in a x ∧ x = a - 1 :=
by
  -- Use the given conditions
  sorry

end two_bacteria_fill_time_l220_220307


namespace sum_of_nine_consecutive_quotients_multiple_of_9_l220_220085

def a (i : ℕ) : ℕ := (10^(2 * i) - 1) / 9
def q (i : ℕ) : ℕ := a i / 11
def s (i : ℕ) : ℕ := q i + q (i + 1) + q (i + 2) + q (i + 3) + q (i + 4) + q (i + 5) + q (i + 6) + q (i + 7) + q (i + 8)

theorem sum_of_nine_consecutive_quotients_multiple_of_9 (i n : ℕ) (h : n > 8) 
  (h2 : i ≤ n - 8) : s i % 9 = 0 :=
sorry

end sum_of_nine_consecutive_quotients_multiple_of_9_l220_220085


namespace gcd_problem_l220_220712

theorem gcd_problem : Nat.gcd 12740 220 - 10 = 10 :=
by
  sorry

end gcd_problem_l220_220712


namespace subset_S_A_inter_B_nonempty_l220_220983

open Finset

-- Definitions of sets A and B
def A : Finset ℕ := {1, 2, 3, 4, 5, 6}
def B : Finset ℕ := {4, 5, 6, 7, 8}

-- Definition of the subset S and its condition
def S : Finset ℕ := {5, 6}

-- The statement to be proved
theorem subset_S_A_inter_B_nonempty : S ⊆ A ∧ S ∩ B ≠ ∅ :=
by {
  sorry -- proof to be provided
}

end subset_S_A_inter_B_nonempty_l220_220983


namespace smallest_square_area_l220_220131

theorem smallest_square_area (a b c d : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 4) (h4 : d = 5) :
  ∃ s : ℕ, s * s = 64 ∧ (a + c <= s ∧ max b d <= s) ∨ (max a c <= s ∧ b + d <= s) :=
sorry

end smallest_square_area_l220_220131


namespace exterior_angle_BAC_l220_220629

theorem exterior_angle_BAC (square_octagon_coplanar : Prop) (common_side_AD : Prop) : 
    angle_BAC = 135 :=
by
  sorry

end exterior_angle_BAC_l220_220629


namespace sum_of_numbers_l220_220135

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 :=
by
  sorry

end sum_of_numbers_l220_220135


namespace science_book_multiple_l220_220448

theorem science_book_multiple (history_pages novel_pages science_pages : ℕ)
  (H1 : history_pages = 300)
  (H2 : novel_pages = history_pages / 2)
  (H3 : science_pages = 600) :
  science_pages / novel_pages = 4 := 
by
  -- Proof will be filled out here
  sorry

end science_book_multiple_l220_220448


namespace subset_range_l220_220569

open Set

-- Definitions of sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- The statement of the problem
theorem subset_range (a : ℝ) (h : A ⊆ B a) : 2 ≤ a :=
sorry -- Skipping the proof

end subset_range_l220_220569


namespace harkamal_payment_l220_220268

theorem harkamal_payment :
  let grapes_kg := 9
  let grape_rate_per_kg := 70
  let mangoes_kg := 9
  let mango_rate_per_kg := 55
  let cost_of_grapes := grapes_kg * grape_rate_per_kg
  let cost_of_mangoes := mangoes_kg * mango_rate_per_kg
  let total_payment := cost_of_grapes + cost_of_mangoes
  total_payment = 1125 :=
by
  let grapes_kg := 9
  let grape_rate_per_kg := 70
  let mangoes_kg := 9
  let mango_rate_per_kg := 55
  let cost_of_grapes := grapes_kg * grape_rate_per_kg
  let cost_of_mangoes := mangoes_kg * mango_rate_per_kg
  let total_payment := cost_of_grapes + cost_of_mangoes
  sorry

end harkamal_payment_l220_220268


namespace ellipse_k_range_ellipse_k_eccentricity_l220_220697

theorem ellipse_k_range (k : ℝ) : 
  (∃ x y : ℝ, x^2/(9 - k) + y^2/(k - 1) = 1) ↔ (1 < k ∧ k < 5 ∨ 5 < k ∧ k < 9) := 
sorry

theorem ellipse_k_eccentricity (k : ℝ) (h : ∃ x y : ℝ, x^2/(9 - k) + y^2/(k - 1) = 1) : 
  eccentricity = Real.sqrt (6/7) → (k = 2 ∨ k = 8) := 
sorry

end ellipse_k_range_ellipse_k_eccentricity_l220_220697


namespace translated_line_expression_l220_220088

theorem translated_line_expression (x y : ℝ) (b : ℝ) :
  (∀ x y, y = 2 * x + 3 ∧ (5, 1).2 = 2 * (5, 1).1 + b) → y = 2 * x - 9 :=
by
  sorry

end translated_line_expression_l220_220088


namespace negation_P_eq_Q_l220_220963

-- Define the proposition P: For any x ∈ ℝ, x^2 - 2x - 3 ≤ 0
def P : Prop := ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0

-- Define its negation which is the proposition Q
def Q : Prop := ∃ x : ℝ, x^2 - 2*x - 3 > 0

-- Prove that the negation of P is equivalent to Q
theorem negation_P_eq_Q : ¬P = Q :=
  by
  sorry

end negation_P_eq_Q_l220_220963


namespace find_x_l220_220589

theorem find_x (x : ℕ) (h₁ : 3 * (Nat.factorial 8) / (Nat.factorial (8 - x)) = 4 * (Nat.factorial 9) / (Nat.factorial (9 - (x - 1)))) : x = 6 :=
sorry

end find_x_l220_220589


namespace alley_width_l220_220791

theorem alley_width (L w : ℝ) (k h : ℝ)
    (h1 : k = L / 2)
    (h2 : h = L * (Real.sqrt 3) / 2)
    (h3 : w^2 + (L / 2)^2 = L^2)
    (h4 : w^2 + (L * (Real.sqrt 3) / 2)^2 = L^2):
    w = (Real.sqrt 3) * L / 2 := 
sorry

end alley_width_l220_220791


namespace sum_of_decimals_as_fraction_l220_220037

theorem sum_of_decimals_as_fraction :
  let x := (0 : ℝ) + 1 / 3;
  let y := (0 : ℝ) + 2 / 3;
  let z := (0 : ℝ) + 2 / 5;
  x + y + z = 7 / 5 :=
by
  let x := (0 : ℝ) + 1 / 3
  let y := (0 : ℝ) + 2 / 3
  let z := (0 : ℝ) + 2 / 5
  show x + y + z = 7 / 5
  sorry

end sum_of_decimals_as_fraction_l220_220037


namespace largest_real_root_range_l220_220503

theorem largest_real_root_range (b0 b1 b2 b3 : ℝ) (h0 : |b0| ≤ 1) (h1 : |b1| ≤ 1) (h2 : |b2| ≤ 1) (h3 : |b3| ≤ 1) :
  ∀ r : ℝ, (Polynomial.eval r (Polynomial.C (1:ℝ) + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C b0) = 0) → (5 / 2) < r ∧ r < 3 :=
by
  sorry

end largest_real_root_range_l220_220503


namespace skill_testing_question_l220_220817

theorem skill_testing_question : (5 * (10 - 6) / 2) = 10 := by
  sorry

end skill_testing_question_l220_220817


namespace distinct_dress_designs_l220_220056

theorem distinct_dress_designs : 
  let num_colors := 5
  let num_patterns := 6
  num_colors * num_patterns = 30 :=
by
  sorry

end distinct_dress_designs_l220_220056


namespace min_value_of_sum_of_squares_l220_220981

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x + 2 * y + z = 1) : 
    x^2 + y^2 + z^2 ≥ (1 / 6) := 
  sorry

noncomputable def min_val_xy2z (x y z : ℝ) (h : x + 2 * y + z = 1) : ℝ :=
  if h_sq : x^2 + y^2 + z^2 = 1 / 6 then (x^2 + y^2 + z^2) else if x = 1 / 6 ∧ z = 1 / 6 ∧ y = 1 / 3 then 1 / 6 else (1 / 6)

example (x y z : ℝ) (h : x + 2 * y + z = 1) : x^2 + y^2 + z^2 = min_val_xy2z x y z h :=
  sorry

end min_value_of_sum_of_squares_l220_220981


namespace max_value_of_determinant_l220_220286

noncomputable def determinant_of_matrix (θ : ℝ) : ℝ :=
  Matrix.det ![
    ![1, 1, 1],
    ![1, 1 + Real.sin (2 * θ), 1],
    ![1, 1, 1 + Real.cos (2 * θ)]
  ]

theorem max_value_of_determinant : 
  ∃ θ : ℝ, (∀ θ : ℝ, determinant_of_matrix θ ≤ (1 / 2)) ∧ determinant_of_matrix (θ_at_maximum) = (1 / 2) :=
sorry

end max_value_of_determinant_l220_220286


namespace marilyn_bottle_caps_start_l220_220976

-- Definitions based on the conditions
def initial_bottle_caps (X : ℕ) := X  -- Number of bottle caps Marilyn started with
def shared_bottle_caps := 36           -- Number of bottle caps shared with Nancy
def remaining_bottle_caps := 15        -- Number of bottle caps left after sharing

-- Theorem statement: Given the conditions, show that Marilyn started with 51 bottle caps
theorem marilyn_bottle_caps_start (X : ℕ) 
  (h1 : initial_bottle_caps X - shared_bottle_caps = remaining_bottle_caps) : 
  X = 51 := 
sorry  -- Proof omitted

end marilyn_bottle_caps_start_l220_220976


namespace percentage_increase_l220_220839

theorem percentage_increase (Z Y X : ℝ) (h1 : Y = 1.20 * Z) (h2 : Z = 250) (h3 : X + Y + Z = 925) :
  ((X - Y) / Y) * 100 = 25 :=
by
  sorry

end percentage_increase_l220_220839


namespace johns_original_earnings_l220_220361

theorem johns_original_earnings (x : ℝ) (h1 : x + 0.5 * x = 90) : x = 60 := 
by
  -- sorry indicates the proof steps are omitted
  sorry

end johns_original_earnings_l220_220361


namespace number_of_integer_solutions_l220_220142

theorem number_of_integer_solutions (h : ∀ n : ℤ, (2020 - n) ^ 2 / (2020 - n ^ 2) ≥ 0) :
  ∃! (m : ℤ), m = 90 := 
sorry

end number_of_integer_solutions_l220_220142


namespace x1_x2_eq_e2_l220_220837

variable (x1 x2 : ℝ)

-- Conditions
def condition1 : Prop := x1 * Real.exp x1 = Real.exp 2
def condition2 : Prop := x2 * Real.log x2 = Real.exp 2

-- The proof problem
theorem x1_x2_eq_e2 (hx1 : condition1 x1) (hx2 : condition2 x2) : x1 * x2 = Real.exp 2 := 
sorry

end x1_x2_eq_e2_l220_220837


namespace arcsin_one_half_l220_220305

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l220_220305


namespace find_x_of_parallel_vectors_l220_220909

theorem find_x_of_parallel_vectors
  (x : ℝ)
  (p : ℝ × ℝ := (2, -3))
  (q : ℝ × ℝ := (x, 6))
  (h : ∃ k : ℝ, q = k • p) :
  x = -4 :=
sorry

end find_x_of_parallel_vectors_l220_220909


namespace geometric_seq_l220_220533

def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ (∀ n : ℕ, S (n + 1) + a n = S n + 5 * 4 ^ n)

theorem geometric_seq (a S : ℕ → ℝ) (h : seq a S) :
  ∃ r : ℝ, ∃ a1 : ℝ, (∀ n : ℕ, (a (n + 1) - 4 ^ (n + 1)) = r * (a n - 4 ^ n)) :=
by
  sorry

end geometric_seq_l220_220533


namespace sarah_score_l220_220704

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end sarah_score_l220_220704


namespace converse_not_true_without_negatives_l220_220175

theorem converse_not_true_without_negatives (a b c d : ℕ) (h : a + d = b + c) : ¬(a - c = b - d) :=
by
  sorry

end converse_not_true_without_negatives_l220_220175


namespace problem_statement_l220_220951

theorem problem_statement (g : ℝ → ℝ) :
  (∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - 2 * x * y - x + 2) →
  (∃ m t : ℝ, m = 1 ∧ t = 3 ∧ m * t = 3) :=
sorry

end problem_statement_l220_220951


namespace average_payment_l220_220573

theorem average_payment (total_payments : ℕ) (first_n_payments : ℕ)  (first_payment_amt : ℕ) (remaining_payment_amt : ℕ) 
  (H1 : total_payments = 104)
  (H2 : first_n_payments = 24)
  (H3 : first_payment_amt = 520)
  (H4 : remaining_payment_amt = 615)
  :
  (24 * 520 + 80 * 615) / 104 = 593.08 := 
  by 
    sorry

end average_payment_l220_220573


namespace g_sqrt_45_l220_220645

noncomputable def g (x : ℝ) : ℝ :=
if x % 1 = 0 then 7 * x + 6 else ⌊x⌋ + 7

theorem g_sqrt_45 : g (Real.sqrt 45) = 13 := by
  sorry

end g_sqrt_45_l220_220645


namespace compare_a_b_c_compare_explicitly_defined_a_b_c_l220_220059

theorem compare_a_b_c (a b c : ℕ) (ha : a = 81^31) (hb : b = 27^41) (hc : c = 9^61) : a > b ∧ b > c := 
by
  sorry

-- Noncomputable definitions if necessary
noncomputable def a := 81^31
noncomputable def b := 27^41
noncomputable def c := 9^61

theorem compare_explicitly_defined_a_b_c : a > b ∧ b > c := 
by
  sorry

end compare_a_b_c_compare_explicitly_defined_a_b_c_l220_220059


namespace min_value_expression_l220_220848

noncomputable def log (base : ℝ) (num : ℝ) := Real.log num / Real.log base

theorem min_value_expression (a b : ℝ) (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * log a b + 6 * log b a = 11) : 
  a^3 + (2 / (b - 1)) ≥ 2 * Real.sqrt 2 + 1 :=
by
  sorry

end min_value_expression_l220_220848


namespace manager_salary_l220_220115

theorem manager_salary :
  let avg_salary_employees := 1500
  let num_employees := 20
  let new_avg_salary := 2000
  (new_avg_salary * (num_employees + 1) - avg_salary_employees * num_employees = 12000) :=
by
  sorry

end manager_salary_l220_220115


namespace lowest_score_dropped_l220_220868

-- Conditions definitions
def total_sum_of_scores (A B C D : ℕ) := A + B + C + D = 240
def total_sum_after_dropping_lowest (A B C : ℕ) := A + B + C = 195

-- Theorem statement
theorem lowest_score_dropped (A B C D : ℕ) (h1 : total_sum_of_scores A B C D) (h2 : total_sum_after_dropping_lowest A B C) : D = 45 := 
sorry

end lowest_score_dropped_l220_220868


namespace david_more_push_ups_than_zachary_l220_220820

def zachary_push_ups : ℕ := 53
def zachary_crunches : ℕ := 14
def zachary_total : ℕ := 67
def david_crunches : ℕ := zachary_crunches - 10
def david_push_ups : ℕ := zachary_total - david_crunches

theorem david_more_push_ups_than_zachary : david_push_ups - zachary_push_ups = 10 := by
  sorry  -- Proof is not required as per instructions

end david_more_push_ups_than_zachary_l220_220820


namespace car_capacities_rental_plans_l220_220920

-- Define the capacities for part 1
def capacity_A : ℕ := 3
def capacity_B : ℕ := 4

theorem car_capacities (x y : ℕ) (h₁ : 2 * x + y = 10) (h₂ : x + 2 * y = 11) : 
  x = capacity_A ∧ y = capacity_B := by
  sorry

-- Define the valid rental plans for part 2
def valid_rental_plan (a b : ℕ) : Prop :=
  3 * a + 4 * b = 31

theorem rental_plans (a b : ℕ) (h : valid_rental_plan a b) : 
  (a = 1 ∧ b = 7) ∨ (a = 5 ∧ b = 4) ∨ (a = 9 ∧ b = 1) := by
  sorry

end car_capacities_rental_plans_l220_220920


namespace distance_between_trees_l220_220017

theorem distance_between_trees
  (yard_length : ℕ)
  (num_trees : ℕ)
  (h_yard_length : yard_length = 441)
  (h_num_trees : num_trees = 22) :
  (yard_length / (num_trees - 1)) = 21 :=
by
  sorry

end distance_between_trees_l220_220017


namespace find_x_l220_220210

theorem find_x (k : ℝ) (x : ℝ) (h : x ≠ 4) :
  (x = (1 - k) / 2) ↔ ((x^2 - 3 * x - 4) / (x - 4) = 3 * x + k) :=
by sorry

end find_x_l220_220210


namespace fraction_value_l220_220386

theorem fraction_value :
  (2015^2 : ℤ) / (2014^2 + 2016^2 - 2) = (1 : ℚ) / 2 :=
by
  sorry

end fraction_value_l220_220386


namespace modular_inverse_28_mod_29_l220_220384

theorem modular_inverse_28_mod_29 :
  28 * 28 ≡ 1 [MOD 29] :=
by
  sorry

end modular_inverse_28_mod_29_l220_220384


namespace no_function_satisfies_condition_l220_220156

theorem no_function_satisfies_condition :
  ¬ ∃ (f: ℕ → ℕ), ∀ (n: ℕ), f (f n) = n + 2017 :=
by
  -- Proof details are omitted
  sorry

end no_function_satisfies_condition_l220_220156


namespace problem_statement_l220_220526

theorem problem_statement (k : ℕ) (h : 35^k ∣ 1575320897) : 7^k - k^7 = 1 := by
  sorry

end problem_statement_l220_220526


namespace eight_natural_numbers_exist_l220_220128

theorem eight_natural_numbers_exist :
  ∃ (n : Fin 8 → ℕ), (∀ i j : Fin 8, i ≠ j → ¬(n i ∣ n j)) ∧ (∀ i j : Fin 8, i ≠ j → n i ∣ (n j * n j)) :=
by 
  sorry

end eight_natural_numbers_exist_l220_220128


namespace failed_in_english_l220_220491

/- Lean definitions and statement -/

def total_percentage := 100
def failed_H := 32
def failed_H_and_E := 12
def passed_H_or_E := 24

theorem failed_in_english (total_percentage failed_H failed_H_and_E passed_H_or_E : ℕ) (h1 : total_percentage = 100) (h2 : failed_H = 32) (h3 : failed_H_and_E = 12) (h4 : passed_H_or_E = 24) :
  total_percentage - (failed_H + (total_percentage - passed_H_or_E - failed_H_and_E)) = 56 :=
by sorry

end failed_in_english_l220_220491


namespace y_increase_for_x_increase_l220_220703

theorem y_increase_for_x_increase (x y : ℝ) (h : 4 * y = 9) : 12 * y = 27 :=
by
  sorry

end y_increase_for_x_increase_l220_220703


namespace graph_of_equation_is_two_intersecting_lines_l220_220170

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ x y : ℝ, (x + 3 * y) ^ 3 = x ^ 3 + 9 * y ^ 3 ↔ (x = 0 ∨ y = 0 ∨ x + 3 * y = 0) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l220_220170


namespace units_digit_35_pow_35_mul_17_pow_17_l220_220462

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_35_pow_35_mul_17_pow_17:
  units_digit (35 ^ (35 * 17 ^ 17)) = 5 := 
by {
  -- Here we're skipping the proof.
  sorry
}

end units_digit_35_pow_35_mul_17_pow_17_l220_220462


namespace Eva_numbers_l220_220987

theorem Eva_numbers : ∃ (a b : ℕ), a + b = 43 ∧ a - b = 15 ∧ a = 29 ∧ b = 14 :=
by
  sorry

end Eva_numbers_l220_220987


namespace cylinder_surface_area_minimization_l220_220349

theorem cylinder_surface_area_minimization (S V r h : ℝ) (h₁ : π * r^2 * h = V) (h₂ : r^2 + (h / 2)^2 = S^2) : (h / r) = 2 :=
sorry

end cylinder_surface_area_minimization_l220_220349


namespace alex_sweaters_l220_220841

def num_items (shirts : ℕ) (pants : ℕ) (jeans : ℕ) (total_cycle_time_minutes : ℕ)
  (cycle_time_minutes : ℕ) (max_items_per_cycle : ℕ) : ℕ :=
  total_cycle_time_minutes / cycle_time_minutes * max_items_per_cycle

def num_sweaters_to_wash (total_items : ℕ) (non_sweater_items : ℕ) : ℕ :=
  total_items - non_sweater_items

theorem alex_sweaters :
  ∀ (shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle : ℕ),
  shirts = 18 →
  pants = 12 →
  jeans = 13 →
  total_cycle_time_minutes = 180 →
  cycle_time_minutes = 45 →
  max_items_per_cycle = 15 →
  num_sweaters_to_wash
    (num_items shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle)
    (shirts + pants + jeans) = 17 :=
by
  intros shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle
    h_shirts h_pants h_jeans h_total_cycle_time_minutes h_cycle_time_minutes h_max_items_per_cycle
  
  sorry

end alex_sweaters_l220_220841


namespace bear_meat_needs_l220_220445

theorem bear_meat_needs (B_total : ℕ) (cubs : ℕ) (w_cub : ℚ) 
  (h1 : B_total = 210)
  (h2 : cubs = 4)
  (h3 : w_cub = B_total / cubs) : 
  w_cub = 52.5 :=
by 
  sorry

end bear_meat_needs_l220_220445


namespace monochromatic_rectangle_l220_220412

theorem monochromatic_rectangle (n : ℕ) (coloring : ℕ × ℕ → Fin n) :
  ∃ (a b c d : ℕ × ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  (coloring a = coloring b) ∧ (coloring b = coloring c) ∧ (coloring c = coloring d) :=
sorry

end monochromatic_rectangle_l220_220412


namespace square_side_length_same_area_l220_220742

theorem square_side_length_same_area (length width : ℕ) (l_eq : length = 72) (w_eq : width = 18) : 
  ∃ side_length : ℕ, side_length * side_length = length * width ∧ side_length = 36 :=
by
  sorry

end square_side_length_same_area_l220_220742


namespace correct_system_of_equations_l220_220120

-- Definitions based on the conditions
def rope_exceeds (x y : ℝ) : Prop := x - y = 4.5
def rope_half_falls_short (x y : ℝ) : Prop := (1/2) * x + 1 = y

-- Proof statement
theorem correct_system_of_equations (x y : ℝ) :
  rope_exceeds x y → rope_half_falls_short x y → 
  (x - y = 4.5 ∧ (1/2 * x + 1 = y)) := 
by 
  sorry

end correct_system_of_equations_l220_220120


namespace smallest_x_y_sum_l220_220764

theorem smallest_x_y_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hne : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 24) :
  x + y = 100 :=
sorry

end smallest_x_y_sum_l220_220764


namespace problem_f_val_l220_220064

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_val (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (-x) = -f x)
  (h2 : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (h3 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^3) :
  f 2015 = -1 :=
  sorry

end problem_f_val_l220_220064


namespace division_of_mixed_numbers_l220_220229

noncomputable def mixed_to_improper (n : ℕ) (a b : ℕ) : ℚ :=
  n + (a / b)

theorem division_of_mixed_numbers : 
  (mixed_to_improper 7 1 3) / (mixed_to_improper 2 1 2) = 44 / 15 :=
by
  sorry

end division_of_mixed_numbers_l220_220229


namespace jim_age_is_55_l220_220945

-- Definitions of the conditions
def jim_age (t : ℕ) : ℕ := 3 * t + 10

def sum_ages (j t : ℕ) : Prop := j + t = 70

-- Statement of the proof problem
theorem jim_age_is_55 : ∃ t : ℕ, jim_age t = 55 ∧ sum_ages (jim_age t) t :=
by
  sorry

end jim_age_is_55_l220_220945


namespace cost_of_iphone_l220_220214

def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80
def weeks_worked : ℕ := 7
def total_earnings := weekly_earnings * weeks_worked
def total_money := total_earnings + trade_in_value
def new_iphone_cost : ℕ := 800

theorem cost_of_iphone :
  total_money = new_iphone_cost := by
  sorry

end cost_of_iphone_l220_220214


namespace students_received_B_l220_220771

theorem students_received_B (x : ℕ) 
  (h1 : (0.8 * x : ℝ) + x + (1.2 * x : ℝ) = 28) : 
  x = 9 := 
by
  sorry

end students_received_B_l220_220771


namespace expand_polynomial_l220_220944

theorem expand_polynomial (N : ℕ) :
  (∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ (a + b + c + d + 1)^N = 715) ↔ N = 13 := by
  sorry -- Replace with the actual proof when ready

end expand_polynomial_l220_220944


namespace rectangular_area_l220_220680

theorem rectangular_area (length width : ℝ) (h₁ : length = 0.4) (h₂ : width = 0.22) :
  (length * width = 0.088) :=
by sorry

end rectangular_area_l220_220680


namespace total_canoes_built_l220_220844

-- Given conditions as definitions
def a1 : ℕ := 10
def r : ℕ := 3

-- Define the geometric series sum for first four terms
noncomputable def sum_of_geometric_series (a1 r : ℕ) (n : ℕ) : ℕ :=
  a1 * ((r^n - 1) / (r - 1))

-- Prove that the total number of canoes built by the end of April is 400
theorem total_canoes_built (a1 r : ℕ) (n : ℕ) : sum_of_geometric_series a1 r n = 400 :=
  sorry

end total_canoes_built_l220_220844


namespace find_solution_l220_220669

-- Definitions for the problem
def is_solution (x y z t : ℕ) : Prop := (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ (2^y + 2^z * 5^t - 5^x = 1))

-- Statement of the theorem
theorem find_solution : ∀ x y z t : ℕ, is_solution x y z t → (x, y, z, t) = (2, 4, 1, 1) := by
  sorry

end find_solution_l220_220669


namespace andrew_donuts_l220_220613

/--
Andrew originally asked for 3 donuts for each of his 2 friends, Brian and Samuel. 
Then invited 2 more friends and asked for the same amount of donuts for them. 
Andrew’s mother wants to buy one more donut for each of Andrew’s friends. 
Andrew's mother is also going to buy the same amount of donuts for Andrew as everybody else.
Given these conditions, the total number of donuts Andrew’s mother needs to buy is 20.
-/
theorem andrew_donuts : (3 * 2) + (3 * 2) + 4 + 4 = 20 :=
by
  -- Given:
  -- 1. Andrew asked for 3 donuts for each of his two friends, Brian and Samuel.
  -- 2. He later invited 2 more friends and asked for the same amount of donuts for them.
  -- 3. Andrew’s mother wants to buy one more donut for each of Andrew’s friends.
  -- 4. Andrew’s mother is going to buy the same amount of donuts for Andrew as everybody else.
  -- Prove: The total number of donuts Andrew’s mother needs to buy is 20.
  sorry

end andrew_donuts_l220_220613


namespace investment_ratio_l220_220101

theorem investment_ratio (A_invest B_invest C_invest : ℝ) (F : ℝ) (total_profit B_share : ℝ)
  (h1 : A_invest = 3 * B_invest)
  (h2 : B_invest = F * C_invest)
  (h3 : total_profit = 7700)
  (h4 : B_share = 1400)
  (h5 : (B_invest / (A_invest + B_invest + C_invest)) * total_profit = B_share) :
  (B_invest / C_invest) = 2 / 3 := 
by
  sorry

end investment_ratio_l220_220101


namespace speed_of_stream_l220_220960

variable (v : ℝ)

theorem speed_of_stream (h : (64 / (24 + v)) = (32 / (24 - v))) : v = 8 := 
by
  sorry

end speed_of_stream_l220_220960


namespace product_of_fractions_l220_220767

open BigOperators

theorem product_of_fractions :
  (∏ n in Finset.range 9, (n + 2)^3 - 1) / (∏ n in Finset.range 9, (n + 2)^3 + 1) = 74 / 55 :=
by
  sorry

end product_of_fractions_l220_220767


namespace area_of_rectangle_l220_220649

-- Define the given conditions
def side_length_of_square (s : ℝ) (ABCD : ℝ) : Prop :=
  ABCD = 4 * s^2

def perimeter_of_rectangle (s : ℝ) (perimeter : ℝ): Prop :=
  perimeter = 8 * s

-- Statement of the proof problem
theorem area_of_rectangle (s perimeter_area : ℝ) (h_perimeter : perimeter_of_rectangle s 160) :
  side_length_of_square s 1600 :=
by
  sorry

end area_of_rectangle_l220_220649


namespace parabola_equation_l220_220686

theorem parabola_equation (a b c d e f : ℤ)
  (h1 : a = 0 )    -- The equation should have no x^2 term
  (h2 : b = 0 )    -- The equation should have no xy term
  (h3 : c > 0)     -- The coefficient of y^2 should be positive
  (h4 : d = -2)    -- The coefficient of x in the final form should be -2
  (h5 : e = -8)    -- The coefficient of y in the final form should be -8
  (h6 : f = 16)    -- The constant term in the final form should be 16
  (pass_through : (2 : ℤ) = k * (6 - 4) ^ 2)
  (vertex : (0 : ℤ) = k * (sym_axis - 4) ^ 2)
  (symmetry_axis_parallel_x : True)
  (vertex_on_y_axis : True):
  ax^2 + bxy + cy^2 + dx + ey + f = 0 :=
by
  sorry

end parabola_equation_l220_220686


namespace unw_touchable_area_l220_220794

-- Define the conditions
def ball_radius : ℝ := 1
def container_edge_length : ℝ := 5

-- Define the surface area that the ball can never touch
theorem unw_touchable_area : (ball_radius = 1) ∧ (container_edge_length = 5) → 
  let total_unreachable_area := 120
  let overlapping_area := 24
  let unreachable_area := total_unreachable_area - overlapping_area
  unreachable_area = 96 :=
by
  intros
  sorry

end unw_touchable_area_l220_220794


namespace line_equation_in_slope_intercept_form_l220_220375

variable {x y : ℝ}

theorem line_equation_in_slope_intercept_form :
  (3 * (x - 2) - 4 * (y - 8) = 0) → (y = (3 / 4) * x + 6.5) :=
by
  intro h
  sorry

end line_equation_in_slope_intercept_form_l220_220375


namespace calc_g_inv_sum_l220_220779

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x * x

noncomputable def g_inv (y : ℝ) : ℝ := 
  if y = -4 then 4
  else if y = 0 then 3
  else if y = 4 then -1
  else 0

theorem calc_g_inv_sum : g_inv (-4) + g_inv 0 + g_inv 4 = 6 :=
by
  sorry

end calc_g_inv_sum_l220_220779


namespace intersection_distance_l220_220466

open Real

-- Definition of the curve C in standard coordinates
def curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Definition of the line l in parametric form
def line_l (x y t : ℝ) : Prop :=
  x = 1 + t ∧ y = -1 + t

-- The length of the intersection points A and B of curve C and line l
theorem intersection_distance : ∃ t1 t2 : ℝ, (curve_C (1 + t1) (-1 + t1) ∧ curve_C (1 + t2) (-1 + t2)) ∧ (abs (t1 - t2) = 4 * sqrt 6) :=
sorry

end intersection_distance_l220_220466


namespace problem_solution_l220_220822

noncomputable def problem_statement : Prop :=
  ∀ (α β : ℝ), 
    (0 < α ∧ α < Real.pi / 2) →
    (0 < β ∧ β < Real.pi / 2) →
    (Real.sin α = 4 / 5) →
    (Real.cos (α + β) = 5 / 13) →
    (Real.cos β = 63 / 65 ∧ (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos (2 * α) - 1) = -5 / 4)
    
theorem problem_solution : problem_statement :=
by
  sorry

end problem_solution_l220_220822


namespace brett_blue_marbles_more_l220_220105

theorem brett_blue_marbles_more (r b : ℕ) (hr : r = 6) (hb : b = 5 * r) : b - r = 24 := by
  rw [hr, hb]
  norm_num
  sorry

end brett_blue_marbles_more_l220_220105


namespace rate_of_dividend_is_12_l220_220699

-- Defining the conditions
def total_investment : ℝ := 4455
def price_per_share : ℝ := 8.25
def annual_income : ℝ := 648
def face_value_per_share : ℝ := 10

-- Expected rate of dividend
def expected_rate_of_dividend : ℝ := 12

-- The proof problem statement: Prove that the rate of dividend is 12% given the conditions.
theorem rate_of_dividend_is_12 :
  ∃ (r : ℝ), r = 12 ∧ annual_income = 
    (total_investment / price_per_share) * (r / 100) * face_value_per_share :=
by 
  use 12
  sorry

end rate_of_dividend_is_12_l220_220699


namespace perimeter_of_triangle_l220_220164

noncomputable def ellipse_perimeter (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1) : ℝ :=
  let a := 2
  let c := 1
  2 * a + 2 * c

theorem perimeter_of_triangle (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1) :
  ellipse_perimeter x y h = 6 :=
by 
  sorry

end perimeter_of_triangle_l220_220164


namespace polygon_sides_equation_l220_220040

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end polygon_sides_equation_l220_220040


namespace time_to_decorate_l220_220477

variable (mia_rate billy_rate total_eggs : ℕ)

theorem time_to_decorate (h_mia : mia_rate = 24) (h_billy : billy_rate = 10) (h_total : total_eggs = 170) :
  total_eggs / (mia_rate + billy_rate) = 5 :=
by
  sorry

end time_to_decorate_l220_220477


namespace textile_firm_looms_l220_220163

theorem textile_firm_looms
  (sales_val : ℝ)
  (manu_exp : ℝ)
  (estab_charges : ℝ)
  (profit_decrease : ℝ)
  (L : ℝ)
  (h_sales : sales_val = 500000)
  (h_manu_exp : manu_exp = 150000)
  (h_estab_charges : estab_charges = 75000)
  (h_profit_decrease : profit_decrease = 7000)
  (hem_equal_contrib : ∀ l : ℝ, l > 0 →
    (l = sales_val / (sales_val / L) - manu_exp / (manu_exp / L)))
  : L = 50 := 
by
  sorry

end textile_firm_looms_l220_220163


namespace initial_volume_solution_l220_220793

variable (V : ℝ)

theorem initial_volume_solution
  (h1 : 0.35 * V + 1.8 = 0.50 * (V + 1.8)) :
  V = 6 :=
by
  sorry

end initial_volume_solution_l220_220793


namespace inequality_log_equality_log_l220_220262

theorem inequality_log (x : ℝ) (hx : x < 0 ∨ x > 0) :
  max 0 (Real.log (|x|)) ≥ 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) := 
sorry

theorem equality_log (x : ℝ) :
  (max 0 (Real.log (|x|)) = 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2)) ↔ 
  (x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2) := 
sorry

end inequality_log_equality_log_l220_220262


namespace fraction_invariant_l220_220267

variable {R : Type*} [Field R]
variables (x y : R)

theorem fraction_invariant : (2 * x) / (3 * x - y) = (6 * x) / (9 * x - 3 * y) :=
by
  sorry

end fraction_invariant_l220_220267


namespace find_q_of_quadratic_with_roots_ratio_l220_220169

theorem find_q_of_quadratic_with_roots_ratio {q : ℝ} :
  (∃ r1 r2 : ℝ, r1 ≠ 0 ∧ r2 ≠ 0 ∧ r1 / r2 = 3 / 1 ∧ r1 + r2 = -10 ∧ r1 * r2 = q) →
  q = 18.75 :=
by
  sorry

end find_q_of_quadratic_with_roots_ratio_l220_220169


namespace polynomial_coefficients_sum_l220_220560

theorem polynomial_coefficients_sum :
  let a := -15
  let b := 69
  let c := -81
  let d := 27
  10 * a + 5 * b + 2 * c + d = 60 :=
by
  let a := -15
  let b := 69
  let c := -81
  let d := 27
  sorry

end polynomial_coefficients_sum_l220_220560


namespace solve_recursive_fn_eq_l220_220419

-- Define the recursive function
def recursive_fn (x : ℝ) : ℝ :=
  2 * (2 * (2 * (2 * (2 * x - 1) - 1) - 1) - 1) - 1

-- State the theorem we need to prove
theorem solve_recursive_fn_eq (x : ℝ) : recursive_fn x = x → x = 1 :=
by
  sorry

end solve_recursive_fn_eq_l220_220419


namespace weight_of_B_l220_220132

theorem weight_of_B (A B C : ℝ)
(h1 : (A + B + C) / 3 = 45)
(h2 : (A + B) / 2 = 40)
(h3 : (B + C) / 2 = 41)
(h4 : 2 * A = 3 * B ∧ 5 * C = 3 * B)
(h5 : A + B + C = 144) :
B = 43.2 :=
sorry

end weight_of_B_l220_220132


namespace sum_geom_seq_l220_220391

theorem sum_geom_seq (S : ℕ → ℝ) (a_n : ℕ → ℝ) (h1 : S 4 ≠ 0) 
  (h2 : S 8 / S 4 = 4) 
  (h3 : ∀ n : ℕ, S n = a_n 0 * (1 - (a_n 1 / a_n 0)^n) / (1 - a_n 1 / a_n 0)) :
  S 12 / S 4 = 13 :=
sorry

end sum_geom_seq_l220_220391


namespace series_sum_l220_220449

variable {c d : ℝ}

theorem series_sum (h : ∑' n : ℕ, c / d ^ ((3 : ℝ) ^ n) = 9) :
  ∑' n : ℕ, c / (c + 2 * d) ^ (n + 1) = 9 / 11 :=
by
  -- The code that follows will include the steps and proof to reach the conclusion
  sorry

end series_sum_l220_220449


namespace yards_green_correct_l220_220257

-- Define the conditions
def total_yards_silk := 111421
def yards_pink := 49500

-- Define the question as a theorem statement
theorem yards_green_correct :
  (total_yards_silk - yards_pink = 61921) :=
by
  sorry

end yards_green_correct_l220_220257


namespace anna_lemonade_difference_l220_220520

variables (x y p s : ℝ)

theorem anna_lemonade_difference (h : x * p = 1.5 * (y * s)) : (x * p) - (y * s) = 0.5 * (y * s) :=
by
  -- Insert proof here
  sorry

end anna_lemonade_difference_l220_220520


namespace rationalize_denominator_eqn_l220_220642

theorem rationalize_denominator_eqn : 
  let expr := (3 + Real.sqrt 2) / (2 - Real.sqrt 5)
  let rationalized := -6 - 3 * Real.sqrt 5 - 2 * Real.sqrt 2 - Real.sqrt 10
  let A := -6
  let B := -2
  let C := 2
  expr = rationalized ∧ A * B * C = -24 :=
by
  sorry

end rationalize_denominator_eqn_l220_220642


namespace simplify_expression_l220_220107

noncomputable def sin_30 := 1 / 2
noncomputable def cos_30 := Real.sqrt 3 / 2

theorem simplify_expression :
  (sin_30 ^ 3 + cos_30 ^ 3) / (sin_30 + cos_30) = 1 - Real.sqrt 3 / 4 := sorry

end simplify_expression_l220_220107


namespace exists_f_satisfying_iteration_l220_220385

-- Mathematically equivalent problem statement in Lean 4
theorem exists_f_satisfying_iteration :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, (f^[1995] n) = 2 * n :=
by
  -- Fill in proof here
  sorry

end exists_f_satisfying_iteration_l220_220385


namespace range_of_a_l220_220640

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) : Prop :=
  a > 1

-- Translate the problem to a Lean 4 statement
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → a ∈ Set.Icc (-2 : ℝ) 1 ∪ Set.Ici 2 :=
by
  sorry

end range_of_a_l220_220640


namespace eccentricity_of_ellipse_l220_220937

open Real

noncomputable def eccentricity_min (m : ℝ) (h₁ : m > 0) (h₂ : m ≥ 2) : ℝ :=
  if h : m = 2 then (sqrt 6)/3 else 0

theorem eccentricity_of_ellipse (m : ℝ) (h₁ : m > 0) (h₂ : m ≥ 2) :
    eccentricity_min m h₁ h₂ = (sqrt 6)/3 := by
  sorry

end eccentricity_of_ellipse_l220_220937


namespace problem_inequality_l220_220058

theorem problem_inequality (x y z : ℝ) (h1 : x + y + z = 0) (h2 : |x| + |y| + |z| ≤ 1) :
  x + (y / 2) + (z / 3) ≤ 1 / 3 :=
sorry

end problem_inequality_l220_220058


namespace max_value_fraction_l220_220150

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 1 = 0

theorem max_value_fraction (a b : ℝ) (H : circle_eq a b) :
  ∃ t : ℝ, -1/2 ≤ t ∧ t ≤ 1/2 ∧ b = t * (a - 3) ∧ t = 1 / 2 :=
by sorry

end max_value_fraction_l220_220150


namespace evaluate_expression_at_minus_one_l220_220805

theorem evaluate_expression_at_minus_one :
  ((-1 + 1) * (-1 - 2) + 2 * (-1 + 4) * (-1 - 4)) = -30 := by
  sorry

end evaluate_expression_at_minus_one_l220_220805


namespace wire_ratio_l220_220213

theorem wire_ratio (a b : ℝ) (h : (a / 4) ^ 2 = (b / (2 * Real.pi)) ^ 2 * Real.pi) : a / b = 2 / Real.sqrt Real.pi := by
  sorry

end wire_ratio_l220_220213


namespace find_c_in_terms_of_a_and_b_l220_220826

theorem find_c_in_terms_of_a_and_b (a b : ℝ) :
  (∃ α β : ℝ, (α + β = -a) ∧ (α * β = b)) →
  (∃ c d : ℝ, (∃ α β : ℝ, (α^3 + β^3 = -c) ∧ (α^3 * β^3 = d))) →
  c = a^3 - 3 * a * b :=
by
  intros h1 h2
  sorry

end find_c_in_terms_of_a_and_b_l220_220826


namespace simple_interest_rate_l220_220418

theorem simple_interest_rate (P R: ℝ) (T : ℝ) (hT : T = 8) (h : 2 * P = P + (P * R * T) / 100) : R = 12.5 :=
by
  -- Placeholder for proof steps
  sorry

end simple_interest_rate_l220_220418


namespace calculate_total_travel_time_l220_220658

/-- The total travel time, including stops, from the first station to the last station. -/
def total_travel_time (d1 d2 d3 : ℕ) (s1 s2 s3 : ℕ) (t1 t2 : ℕ) : ℚ :=
  let leg1_time := d1 / s1
  let stop1_time := t1 / 60
  let leg2_time := d2 / s2
  let stop2_time := t2 / 60
  let leg3_time := d3 / s3
  leg1_time + stop1_time + leg2_time + stop2_time + leg3_time

/-- Proof that total travel time is 2 hours and 22.5 minutes. -/
theorem calculate_total_travel_time :
  total_travel_time 30 40 50 60 40 80 10 5 = 2.375 :=
by
  sorry

end calculate_total_travel_time_l220_220658


namespace quadratic_roots_distinct_real_l220_220409

theorem quadratic_roots_distinct_real (a b c : ℝ) (h_eq : 2 * a = 2 ∧ 2 * b + -3 = b ∧ 2 * c + 1 = c) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ x : ℝ, (2 * x^2 + (-3) * x + 1 = 0) ↔ (x = x1 ∨ x = x2)) :=
by
  sorry

end quadratic_roots_distinct_real_l220_220409


namespace sum_integer_solutions_correct_l220_220482

noncomputable def sum_of_integer_solutions (m : ℝ) : ℝ :=
  if (3 ≤ m ∧ m < 6) ∨ (-6 ≤ m ∧ m < -3) then -9 else 0

theorem sum_integer_solutions_correct (m : ℝ) :
  (∀ x : ℝ, (3 * x + m < 0 ∧ x > -5) → (∃ s : ℝ, s = sum_of_integer_solutions m ∧ s = -9)) :=
by
  sorry

end sum_integer_solutions_correct_l220_220482


namespace remainder_of_product_mod_7_l220_220739

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l220_220739


namespace point_P_coordinates_l220_220994

theorem point_P_coordinates :
  ∃ (x y : ℝ), (y = (x^3 - 10 * x + 3)) ∧ (x < 0) ∧ (3 * x^2 - 10 = 2) ∧ (x = -2 ∧ y = 15) := by
sorry

end point_P_coordinates_l220_220994


namespace find_length_of_second_train_l220_220301

def length_of_second_train (L : ℝ) : Prop :=
  let speed_first_train := 33.33 -- Speed in m/s
  let speed_second_train := 22.22 -- Speed in m/s
  let relative_speed := speed_first_train + speed_second_train -- Relative speed in m/s
  let time_to_cross := 9 -- time in seconds
  let length_first_train := 260 -- Length in meters
  length_first_train + L = relative_speed * time_to_cross

theorem find_length_of_second_train : length_of_second_train 239.95 :=
by
  admit -- To be completed (proof)

end find_length_of_second_train_l220_220301


namespace geometric_sequence_x_l220_220168

theorem geometric_sequence_x (x : ℝ) (h : 1 * 9 = x^2) : x = 3 ∨ x = -3 :=
by
  sorry

end geometric_sequence_x_l220_220168


namespace candy_bar_cost_l220_220458

variable (C : ℕ)

theorem candy_bar_cost
  (soft_drink_cost : ℕ)
  (num_candy_bars : ℕ)
  (total_spent : ℕ)
  (h1 : soft_drink_cost = 2)
  (h2 : num_candy_bars = 5)
  (h3 : total_spent = 27) :
  num_candy_bars * C + soft_drink_cost = total_spent → C = 5 := by
  sorry

end candy_bar_cost_l220_220458


namespace find_q_l220_220298

theorem find_q (p : ℝ) (q : ℝ) (h1 : p ≠ 0) (h2 : p = 4) (h3 : q ≠ 0) (avg_speed_eq : (2 * p * 3) / (p + 3) = 24 / q) : q = 7 := 
 by
  sorry

end find_q_l220_220298


namespace positive_difference_solutions_abs_l220_220011

theorem positive_difference_solutions_abs (x1 x2 : ℝ) 
  (h1 : 2 * x1 - 3 = 18 ∨ 2 * x1 - 3 = -18) 
  (h2 : 2 * x2 - 3 = 18 ∨ 2 * x2 - 3 = -18) : 
  |x1 - x2| = 18 :=
sorry

end positive_difference_solutions_abs_l220_220011


namespace certain_number_divisibility_l220_220821

theorem certain_number_divisibility :
  ∃ k : ℕ, 3150 = 1050 * k :=
sorry

end certain_number_divisibility_l220_220821


namespace bus_trip_distance_l220_220284

-- Defining the problem variables
variables (x D : ℝ) -- x: speed in mph, D: total distance in miles

-- Main theorem stating the problem
theorem bus_trip_distance
  (h1 : 0 < x) -- speed of the bus is positive
  (h2 : (2 * x + 3 * (D - 2 * x) / (2 / 3 * x) / 2 + 0.75) - 2 - 4 = 0)
  -- The first scenario summarising the travel and delays
  (h3 : ((2 * x + 120) / x + 3 * (D - (2 * x + 120)) / (2 / 3 * x) / 2 + 0.75) - 3 = 0)
  -- The second scenario summarising the travel and delays; accident 120 miles further down
  : D = 720 := sorry

end bus_trip_distance_l220_220284


namespace num_pos_int_solutions_2a_plus_3b_eq_15_l220_220075

theorem num_pos_int_solutions_2a_plus_3b_eq_15 : 
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 2 * a + 3 * b = 15) ∧ 
  (∀ (a1 a2 b1 b2 : ℕ), 0 < a1 ∧ 0 < a2 ∧ 0 < b1 ∧ 0 < b2 ∧ 
  (2 * a1 + 3 * b1 = 15) ∧ (2 * a2 + 3 * b2 = 15) → 
  ((a1 = 3 ∧ b1 = 3 ∨ a1 = 6 ∧ b1 = 1) ∧ (a2 = 3 ∧ b2 = 3 ∨ a2 = 6 ∧ b2 = 1))) := 
  sorry

end num_pos_int_solutions_2a_plus_3b_eq_15_l220_220075


namespace part_one_part_two_l220_220675

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + a * x + 6

theorem part_one (x : ℝ) : ∀ a, a = 5 → f x a < 0 ↔ -3 < x ∧ x < -2 :=
by
  sorry

theorem part_two : ∀ a, (∀ x, f x a > 0) ↔ - 2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 :=
by
  sorry

end part_one_part_two_l220_220675


namespace pipes_fill_tank_in_one_hour_l220_220186

theorem pipes_fill_tank_in_one_hour (p q r s : ℝ) (hp : p = 1/2) (hq : q = 1/4) (hr : r = 1/12) (hs : s = 1/6) :
  1 / (p + q + r + s) = 1 :=
by
  sorry

end pipes_fill_tank_in_one_hour_l220_220186


namespace triangle_side_lengths_l220_220657

theorem triangle_side_lengths (a : ℝ) :
  (∃ (b c : ℝ), b = 1 - 2 * a ∧ c = 8 ∧ (3 + b > c ∧ 3 + c > b ∧ b + c > 3)) ↔ (-5 < a ∧ a < -2) :=
sorry

end triangle_side_lengths_l220_220657


namespace find_C_coordinates_l220_220480

-- Define the points A, B, and the vector relationship
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)
def C : ℝ × ℝ := (-3, 9)

-- The condition stating vector AC is twice vector AB
def vector_condition (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1, C.2 - A.2) = (2 * (B.1 - A.1), 2 * (B.2 - A.2))

-- The theorem we need to prove
theorem find_C_coordinates (A B C : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (-1, 5))
  (hCondition : vector_condition A B C) : C = (-3, 9) :=
by
  rw [hA, hB] at hCondition
  -- sorry here skips the proof
  sorry

end find_C_coordinates_l220_220480


namespace second_part_shorter_l220_220800

def length_wire : ℕ := 180
def length_part1 : ℕ := 106
def length_part2 : ℕ := length_wire - length_part1
def length_difference : ℕ := length_part1 - length_part2

theorem second_part_shorter :
  length_difference = 32 :=
by
  sorry

end second_part_shorter_l220_220800


namespace probability_neither_defective_l220_220806

def total_pens : ℕ := 8
def defective_pens : ℕ := 3
def non_defective_pens : ℕ := total_pens - defective_pens
def draw_count : ℕ := 2

def probability_of_non_defective (total : ℕ) (defective : ℕ) (draws : ℕ) : ℚ :=
  let non_defective := total - defective
  (non_defective / total) * ((non_defective - 1) / (total - 1))

theorem probability_neither_defective :
  probability_of_non_defective total_pens defective_pens draw_count = 5 / 14 :=
by sorry

end probability_neither_defective_l220_220806


namespace graph_of_eqn_is_pair_of_lines_l220_220369

theorem graph_of_eqn_is_pair_of_lines : 
  ∃ (l₁ l₂ : ℝ × ℝ → Prop), 
  (∀ x y, l₁ (x, y) ↔ x = 2 * y) ∧ 
  (∀ x y, l₂ (x, y) ↔ x = -2 * y) ∧ 
  (∀ x y, (x^2 - 4 * y^2 = 0) ↔ (l₁ (x, y) ∨ l₂ (x, y))) :=
by
  sorry

end graph_of_eqn_is_pair_of_lines_l220_220369


namespace quadrants_contain_points_l220_220725

def satisfy_inequalities (x y : ℝ) : Prop :=
  y > -3 * x ∧ y > x + 2

def in_quadrant_I (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def in_quadrant_II (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem quadrants_contain_points (x y : ℝ) :
  satisfy_inequalities x y → (in_quadrant_I x y ∨ in_quadrant_II x y) :=
sorry

end quadrants_contain_points_l220_220725


namespace expression_equals_5_l220_220620

theorem expression_equals_5 : (3^2 - 2^2) = 5 := by
  calc
    (3^2 - 2^2) = 5 := by sorry

end expression_equals_5_l220_220620


namespace project_completion_l220_220281

theorem project_completion (a b c d e : ℕ) 
  (h₁ : 1 / (a : ℝ) + 1 / b + 1 / c + 1 / d = 1 / 6)
  (h₂ : 1 / (b : ℝ) + 1 / c + 1 / d + 1 / e = 1 / 8)
  (h₃ : 1 / (a : ℝ) + 1 / e = 1 / 12) : 
  e = 48 :=
sorry

end project_completion_l220_220281


namespace cost_of_letter_is_0_37_l220_220929

-- Definitions based on the conditions
def total_cost : ℝ := 4.49
def package_cost : ℝ := 0.88
def num_letters : ℕ := 5
def num_packages : ℕ := 3
def letter_cost (L : ℝ) : ℝ := 5 * L
def package_total_cost : ℝ := num_packages * package_cost

-- Theorem that encapsulates the mathematical proof problem
theorem cost_of_letter_is_0_37 (L : ℝ) (h : letter_cost L + package_total_cost = total_cost) : L = 0.37 :=
by sorry

end cost_of_letter_is_0_37_l220_220929


namespace binom_60_3_eq_34220_l220_220258

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l220_220258


namespace calculate_kevin_training_time_l220_220709

theorem calculate_kevin_training_time : 
  ∀ (laps : ℕ) 
    (track_length : ℕ) 
    (run1_distance : ℕ) 
    (run1_speed : ℕ) 
    (walk_distance : ℕ) 
    (walk_speed : Real) 
    (run2_distance : ℕ) 
    (run2_speed : ℕ) 
    (minutes : ℕ) 
    (seconds : Real),
    laps = 8 →
    track_length = 500 →
    run1_distance = 200 →
    run1_speed = 3 →
    walk_distance = 100 →
    walk_speed = 1.5 →
    run2_distance = 200 →
    run2_speed = 4 →
    minutes = 24 →
    seconds = 27 →
    (∀ (t1 t2 t3 t_total t_training : Real),
      t1 = run1_distance / run1_speed →
      t2 = walk_distance / walk_speed →
      t3 = run2_distance / run2_speed →
      t_total = t1 + t2 + t3 →
      t_training = laps * t_total →
      t_training = (minutes * 60 + seconds)) := 
by
  intros laps track_length run1_distance run1_speed walk_distance walk_speed run2_distance run2_speed minutes seconds
  intros h_laps h_track_length h_run1_distance h_run1_speed h_walk_distance h_walk_speed h_run2_distance h_run2_speed h_minutes h_seconds
  intros t1 t2 t3 t_total t_training
  intros h_t1 h_t2 h_t3 h_t_total h_t_training
  sorry

end calculate_kevin_training_time_l220_220709


namespace part_one_solution_set_part_two_range_a_l220_220845

noncomputable def f (x a : ℝ) := |x - a| + x

theorem part_one_solution_set (x : ℝ) :
  f x 3 ≥ x + 4 ↔ (x ≤ -1 ∨ x ≥ 7) :=
by sorry

theorem part_two_range_a (a : ℝ) :
  (∀ x, (1 ≤ x ∧ x ≤ 3) → f x a ≥ 2 * a^2) ↔ (-1 ≤ a ∧ a ≤ 1/2) :=
by sorry

end part_one_solution_set_part_two_range_a_l220_220845


namespace simplify_expression_l220_220233

theorem simplify_expression (x : ℝ) : (2 * x)^5 - (5 * x) * (x^4) = 27 * x^5 :=
by
  sorry

end simplify_expression_l220_220233


namespace binomial_multiplication_subtract_240_l220_220982

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_multiplication_subtract_240 :
  binom 10 3 * binom 8 3 - 240 = 6480 :=
by
  sorry

end binomial_multiplication_subtract_240_l220_220982


namespace m_is_perfect_square_l220_220731

-- Given definitions and conditions
def is_odd (k : ℤ) : Prop := ∃ n : ℤ, k = 2 * n + 1

def is_perfect_square (m : ℕ) : Prop := ∃ a : ℕ, m = a * a

theorem m_is_perfect_square (k m n : ℕ) (h1 : (2 + Real.sqrt 3) ^ k = 1 + m + n * Real.sqrt 3)
  (h2 : 0 < m) (h3 : 0 < n) (h4 : 0 < k) (h5 : is_odd k) : is_perfect_square m := 
sorry

end m_is_perfect_square_l220_220731


namespace simplify_fraction_l220_220205

theorem simplify_fraction (a b : ℕ) (h : a = 150) (hb : b = 450) : a / b = 1 / 3 := by
  sorry

end simplify_fraction_l220_220205


namespace units_digit_diff_l220_220905

theorem units_digit_diff (p : ℕ) (hp : p > 0) (even_p : p % 2 = 0) (units_p1_7 : (p + 1) % 10 = 7) : (p^3 % 10) = (p^2 % 10) :=
by
  sorry

end units_digit_diff_l220_220905


namespace interval_length_implies_difference_l220_220387

theorem interval_length_implies_difference (a b : ℝ) (h : (b - 5) / 3 - (a - 5) / 3 = 15) : b - a = 45 := by
  sorry

end interval_length_implies_difference_l220_220387


namespace cans_to_paint_35_rooms_l220_220394

/-- Paula the painter initially had enough paint for 45 identically sized rooms.
    Unfortunately, she lost five cans of paint, leaving her with only enough paint for 35 rooms.
    Prove that she now uses 18 cans of paint to paint the 35 rooms. -/
theorem cans_to_paint_35_rooms :
  ∀ (cans_per_room : ℕ) (total_cans : ℕ) (lost_cans : ℕ) (rooms_before : ℕ) (rooms_after : ℕ),
  rooms_before = 45 →
  lost_cans = 5 →
  rooms_after = 35 →
  rooms_before - rooms_after = cans_per_room * lost_cans →
  (cans_per_room * rooms_after) / rooms_after = 18 :=
by
  intros
  sorry

end cans_to_paint_35_rooms_l220_220394


namespace members_of_groups_l220_220926

variable {x y : ℕ}

theorem members_of_groups (h1 : x = y + 10) (h2 : x - 1 = 2 * (y + 1)) :
  x = 17 ∧ y = 7 :=
by
  sorry

end members_of_groups_l220_220926


namespace interchangeable_statements_l220_220804

-- Modeled conditions and relationships
def perpendicular (l p: Type) : Prop := sorry -- Definition of perpendicularity between a line and a plane
def parallel (a b: Type) : Prop := sorry -- Definition of parallelism between two objects (lines or planes)

-- Original Statements
def statement_1 := ∀ (l₁ l₂ p: Type), (perpendicular l₁ p) ∧ (perpendicular l₂ p) → parallel l₁ l₂
def statement_2 := ∀ (p₁ p₂ p: Type), (perpendicular p₁ p) ∧ (perpendicular p₂ p) → parallel p₁ p₂
def statement_3 := ∀ (l₁ l₂ l: Type), (parallel l₁ l) ∧ (parallel l₂ l) → parallel l₁ l₂
def statement_4 := ∀ (l₁ l₂ p: Type), (parallel l₁ p) ∧ (parallel l₂ p) → parallel l₁ l₂

-- Swapped Statements
def swapped_1 := ∀ (p₁ p₂ l: Type), (perpendicular p₁ l) ∧ (perpendicular p₂ l) → parallel p₁ p₂
def swapped_2 := ∀ (l₁ l₂ l: Type), (perpendicular l₁ l) ∧ (perpendicular l₂ l) → parallel l₁ l₂
def swapped_3 := ∀ (p₁ p₂ p: Type), (parallel p₁ p) ∧ (parallel p₂ p) → parallel p₁ p₂
def swapped_4 := ∀ (p₁ p₂ l: Type), (parallel p₁ l) ∧ (parallel p₂ l) → parallel p₁ p₂

-- Proof Problem: Verify which statements are interchangeable
theorem interchangeable_statements :
  (statement_1 ↔ swapped_1) ∧
  (statement_2 ↔ swapped_2) ∧
  (statement_3 ↔ swapped_3) ∧
  (statement_4 ↔ swapped_4) :=
sorry

end interchangeable_statements_l220_220804


namespace locus_of_feet_of_perpendiculars_from_focus_l220_220962

def parabola_locus (p : ℝ) : Prop :=
  ∀ x y : ℝ, (y^2 = (p / 2) * x)

theorem locus_of_feet_of_perpendiculars_from_focus (p : ℝ) :
    parabola_locus p :=
by
  sorry

end locus_of_feet_of_perpendiculars_from_focus_l220_220962


namespace shares_owned_l220_220652

theorem shares_owned (expected_earnings dividend_ratio additional_per_10c actual_earnings total_dividend : ℝ)
  ( h1 : expected_earnings = 0.80 )
  ( h2 : dividend_ratio = 0.50 )
  ( h3 : additional_per_10c = 0.04 )
  ( h4 : actual_earnings = 1.10 )
  ( h5 : total_dividend = 156.0 ) :
  ∃ shares : ℝ, shares = total_dividend / (expected_earnings * dividend_ratio + (max ((actual_earnings - expected_earnings) / 0.10) 0) * additional_per_10c) ∧ shares = 300 := 
sorry

end shares_owned_l220_220652


namespace reflection_y_axis_correct_l220_220500

-- Define the coordinates and reflection across the y-axis
def reflect_y_axis (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (-p.1, p.2)

-- Define the original point M
def M : (ℝ × ℝ) := (3, 2)

-- State the theorem we want to prove
theorem reflection_y_axis_correct : reflect_y_axis M = (-3, 2) :=
by
  -- The proof would go here, but it is omitted as per the instructions
  sorry

end reflection_y_axis_correct_l220_220500


namespace area_of_rectangular_field_l220_220710

theorem area_of_rectangular_field (length width perimeter : ℕ) 
  (h_perimeter : perimeter = 2 * (length + width)) 
  (h_length : length = 15) 
  (h_perimeter_value : perimeter = 70) : 
  (length * width = 300) :=
by
  sorry

end area_of_rectangular_field_l220_220710


namespace sum_of_trinomials_1_l220_220803

theorem sum_of_trinomials_1 (p q : ℝ) :
  (p + q = 0 ∨ p + q = 8) →
  (2 * (1 : ℝ)^2 + (p + q) * 1 + (p + q) = 2 ∨ 2 * (1 : ℝ)^2 + (p + q) * 1 + (p + q) = 18) :=
by sorry

end sum_of_trinomials_1_l220_220803


namespace yura_finishes_problems_by_sept_12_l220_220582

def total_problems := 91
def initial_date := 6 -- September 6
def problems_left_date := 8 -- September 8
def remaining_problems := 46
def decreasing_rate := 1

def problems_solved (z : ℕ) (day : ℕ) : ℕ :=
if day = 6 then z + 1 else if day = 7 then z else if day = 8 then z - 1 else z - (day - 7)

theorem yura_finishes_problems_by_sept_12 :
  ∃ z : ℕ, (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 = total_problems - remaining_problems) ∧
           (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 + problems_solved z 9 + problems_solved z 10 + problems_solved z 11 + problems_solved z 12 = total_problems) :=
sorry

end yura_finishes_problems_by_sept_12_l220_220582


namespace intersection_of_A_and_B_l220_220736

def A : Set ℝ := { x | x > 2 ∨ x < -1 }
def B : Set ℝ := { x | (x + 1) * (4 - x) < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | x > 3 ∨ x < -1 } := sorry

end intersection_of_A_and_B_l220_220736


namespace ratio_of_sums_l220_220808

noncomputable def first_sum : Nat := 
  let sequence := (List.range' 1 15)
  let differences := (List.range' 2 30).map (fun x => 2 * x)
  let sequence_sum := sequence.zip differences |>.map (λ ⟨a, d⟩ => 10 / 2 * (2 * a + 9 * d))
  5 * (20 * (sequence_sum.sum))

noncomputable def second_sum : Nat :=
  let sequence := (List.range' 1 15)
  let differences := (List.range' 1 29).filterMap (fun x => if x % 2 = 1 then some x else none)
  let sequence_sum := sequence.zip differences |>.map (λ ⟨a, d⟩ => 10 / 2 * (2 * a + 9 * d))
  5 * (20 * (sequence_sum.sum) - 135)

theorem ratio_of_sums : (first_sum / second_sum : Rat) = (160 / 151 : Rat) :=
  sorry

end ratio_of_sums_l220_220808


namespace largest_value_l220_220635

theorem largest_value :
  let A := 1/2
  let B := 1/3 + 1/4
  let C := 1/4 + 1/5 + 1/6
  let D := 1/5 + 1/6 + 1/7 + 1/8
  let E := 1/6 + 1/7 + 1/8 + 1/9 + 1/10
  E > A ∧ E > B ∧ E > C ∧ E > D := by
sorry

end largest_value_l220_220635


namespace least_five_digit_congruent_l220_220279

theorem least_five_digit_congruent (x : ℕ) (h1 : x ≥ 10000) (h2 : x < 100000) (h3 : x % 17 = 8) : x = 10004 :=
by {
  sorry
}

end least_five_digit_congruent_l220_220279


namespace simplify_expression_l220_220730

variable (q : Int) -- condition that q is an integer

theorem simplify_expression (q : Int) : 
  ((7 * q + 3) - 3 * q * 2) * 4 + (5 - 2 / 4) * (8 * q - 12) = 40 * q - 42 :=
  by
  sorry

end simplify_expression_l220_220730
