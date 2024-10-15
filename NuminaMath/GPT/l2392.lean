import Mathlib

namespace NUMINAMATH_GPT_locus_of_moving_point_l2392_239275

open Real

theorem locus_of_moving_point
  (M N P Q T E : ℝ × ℝ)
  (a b : ℝ)
  (h_ellipse_M : M.1^2 / 48 + M.2^2 / 16 = 1)
  (h_P : P = (-M.1, M.2))
  (h_Q : Q = (-M.1, -M.2))
  (h_T : T = (M.1, -M.2))
  (h_ellipse_N : N.1^2 / 48 + N.2^2 / 16 = 1)
  (h_perp : (M.1 - N.1) * (M.1 + N.1) + (M.2 - N.2) * (M.2 + N.2) = 0)
  (h_intersection : ∃ x y : ℝ, (y - Q.2) = (N.2 - Q.2)/(N.1 - Q.1) * (x - Q.1) ∧ (y - P.2) = (T.2 - P.2)/(T.1 - P.1) * (x - P.1) ∧ E = (x, y)) : 
  (E.1^2 / 12 + E.2^2 / 4 = 1) :=
  sorry

end NUMINAMATH_GPT_locus_of_moving_point_l2392_239275


namespace NUMINAMATH_GPT_problem_inequality_l2392_239203

def f (x : ℝ) : ℝ := abs (x - 1)

def A := {x : ℝ | -1 < x ∧ x < 1}

theorem problem_inequality (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) : 
  f (a * b) > f a - f b := by
  sorry

end NUMINAMATH_GPT_problem_inequality_l2392_239203


namespace NUMINAMATH_GPT_generatrix_length_of_cone_l2392_239232

theorem generatrix_length_of_cone (r : ℝ) (l : ℝ) (h1 : r = 4) (h2 : (2 * Real.pi * r) = (Real.pi / 2) * l) : l = 16 := 
by
  sorry

end NUMINAMATH_GPT_generatrix_length_of_cone_l2392_239232


namespace NUMINAMATH_GPT_new_light_wattage_is_143_l2392_239236

-- Define the original wattage and the percentage increase
def original_wattage : ℕ := 110
def percentage_increase : ℕ := 30

-- Compute the increase in wattage
noncomputable def increase : ℕ := (percentage_increase * original_wattage) / 100

-- The new wattage should be the original wattage plus the increase
noncomputable def new_wattage : ℕ := original_wattage + increase

-- State the theorem that proves the new wattage is 143 watts
theorem new_light_wattage_is_143 : new_wattage = 143 := by
  unfold new_wattage
  unfold increase
  sorry

end NUMINAMATH_GPT_new_light_wattage_is_143_l2392_239236


namespace NUMINAMATH_GPT_volume_of_rectangular_box_l2392_239282

theorem volume_of_rectangular_box (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 20) 
  (h3 : x * z = 12) : 
  x * y * z = 60 := 
sorry

end NUMINAMATH_GPT_volume_of_rectangular_box_l2392_239282


namespace NUMINAMATH_GPT_rectangle_perimeter_l2392_239265

theorem rectangle_perimeter (x y : ℝ) (h1 : 2 * x + y = 44) (h2 : x + 2 * y = 40) : 2 * (x + y) = 56 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l2392_239265


namespace NUMINAMATH_GPT_sum_powers_divisible_by_5_iff_l2392_239297

theorem sum_powers_divisible_by_5_iff (n : ℕ) (h_pos : n > 0) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end NUMINAMATH_GPT_sum_powers_divisible_by_5_iff_l2392_239297


namespace NUMINAMATH_GPT_distance_P_to_y_axis_l2392_239291

-- Define the Point structure
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Condition: Point P with coordinates (-3, 5)
def P : Point := ⟨-3, 5⟩

-- Definition of distance from a point to the y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  abs p.x

-- Proof problem statement
theorem distance_P_to_y_axis : distance_to_y_axis P = 3 := 
  sorry

end NUMINAMATH_GPT_distance_P_to_y_axis_l2392_239291


namespace NUMINAMATH_GPT_total_hours_played_l2392_239243

-- Definitions based on conditions
def Nathan_hours_per_day : ℕ := 3
def Nathan_weeks : ℕ := 2
def days_per_week : ℕ := 7

def Tobias_hours_per_day : ℕ := 5
def Tobias_weeks : ℕ := 1

-- Calculating total hours
def Nathan_total_hours := Nathan_hours_per_day * days_per_week * Nathan_weeks
def Tobias_total_hours := Tobias_hours_per_day * days_per_week * Tobias_weeks

-- Theorem statement
theorem total_hours_played : Nathan_total_hours + Tobias_total_hours = 77 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_total_hours_played_l2392_239243


namespace NUMINAMATH_GPT_toms_restaurant_bill_l2392_239267

theorem toms_restaurant_bill (num_adults num_children : ℕ) (meal_cost : ℕ) (total_meals : ℕ) (bill : ℕ) :
  num_adults = 2 ∧ num_children = 5 ∧ meal_cost = 8 ∧ total_meals = num_adults + num_children ∧ bill = total_meals * meal_cost → bill = 56 :=
by sorry

end NUMINAMATH_GPT_toms_restaurant_bill_l2392_239267


namespace NUMINAMATH_GPT_auditorium_shared_days_l2392_239269

theorem auditorium_shared_days :
  let drama_club_days := 3
  let choir_days := 5
  let debate_team_days := 7
  Nat.lcm (Nat.lcm drama_club_days choir_days) debate_team_days = 105 :=
by
  let drama_club_days := 3
  let choir_days := 5
  let debate_team_days := 7
  sorry

end NUMINAMATH_GPT_auditorium_shared_days_l2392_239269


namespace NUMINAMATH_GPT_range_of_m_value_of_m_l2392_239220

-- Defining the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- The condition for having real roots
def has_real_roots (a b c : ℝ) := b^2 - 4 * a * c ≥ 0

-- First part: Range of values for m
theorem range_of_m (m : ℝ) : has_real_roots 1 (-2) (m - 1) ↔ m ≤ 2 := sorry

-- Second part: Finding m when x₁² + x₂² = 6x₁x₂
theorem value_of_m 
  (x₁ x₂ m : ℝ) (h₁ : quadratic_eq 1 (-2) (m - 1) x₁) (h₂ : quadratic_eq 1 (-2) (m - 1) x₂) 
  (h_sum : x₁ + x₂ = 2) (h_prod : x₁ * x₂ = m - 1) (h_condition : x₁^2 + x₂^2 = 6 * (x₁ * x₂)) : 
  m = 3 / 2 := sorry

end NUMINAMATH_GPT_range_of_m_value_of_m_l2392_239220


namespace NUMINAMATH_GPT_find_n_l2392_239231

open Nat

theorem find_n (d : ℕ → ℕ) (n : ℕ) (h1 : ∀ j, d (j + 1) > d j) (h2 : n = d 13 + d 14 + d 15) (h3 : (d 5 + 1)^3 = d 15 + 1) : 
  n = 1998 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2392_239231


namespace NUMINAMATH_GPT_square_garden_perimeter_l2392_239226

theorem square_garden_perimeter (q p : ℝ) (h : q = 2 * p + 20) : p = 40 :=
sorry

end NUMINAMATH_GPT_square_garden_perimeter_l2392_239226


namespace NUMINAMATH_GPT_xy_identity_l2392_239215

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end NUMINAMATH_GPT_xy_identity_l2392_239215


namespace NUMINAMATH_GPT_problem2_l2392_239274

noncomputable def problem1 (a b c : ℝ) (A B C : ℝ) (h1 : 2 * (Real.sin A)^2 + (Real.sin B)^2 = (Real.sin C)^2)
    (h2 : b = 2 * a) (h3 : a = 2) : (1 / 2) * a * b * Real.sin C = Real.sqrt 15 :=
by
  sorry

theorem problem2 (a b c : ℝ) (h : 2 * a^2 + b^2 = c^2) :
  ∃ m : ℝ, (m = 2 * Real.sqrt 2) ∧ (∀ x y z : ℝ, 2 * x^2 + y^2 = z^2 → (z^2 / (x * y)) ≥ m) ∧ ((c / a) = 2) :=
by
  sorry

end NUMINAMATH_GPT_problem2_l2392_239274


namespace NUMINAMATH_GPT_joe_list_possibilities_l2392_239239

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end NUMINAMATH_GPT_joe_list_possibilities_l2392_239239


namespace NUMINAMATH_GPT_problem_l2392_239278

def f (x : ℝ) : ℝ := x^2 - 3 * x + 7
def g (x : ℝ) : ℝ := 2 * x + 4
theorem problem : f (g 5) - g (f 5) = 123 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l2392_239278


namespace NUMINAMATH_GPT_median_of_first_15_integers_l2392_239295

theorem median_of_first_15_integers :
  150 * (8 / 100 : ℝ) = 12.0 :=
by
  sorry

end NUMINAMATH_GPT_median_of_first_15_integers_l2392_239295


namespace NUMINAMATH_GPT_tomatoes_left_l2392_239238

theorem tomatoes_left (initial_tomatoes : ℕ) (fraction_eaten : ℚ) (eaters : ℕ) (final_tomatoes : ℕ)  
  (h_initial : initial_tomatoes = 21)
  (h_fraction : fraction_eaten = 1 / 3)
  (h_eaters : eaters = 2)
  (h_final : final_tomatoes = initial_tomatoes - initial_tomatoes * fraction_eaten) :
  final_tomatoes = 14 := by
  sorry

end NUMINAMATH_GPT_tomatoes_left_l2392_239238


namespace NUMINAMATH_GPT_root_expression_value_l2392_239217

theorem root_expression_value
  (r s : ℝ)
  (h1 : 3 * r^2 - 4 * r - 8 = 0)
  (h2 : 3 * s^2 - 4 * s - 8 = 0) :
  (9 * r^3 - 9 * s^3) * (r - s)⁻¹ = 40 := 
sorry

end NUMINAMATH_GPT_root_expression_value_l2392_239217


namespace NUMINAMATH_GPT_original_price_l2392_239253

variable (a : ℝ)

theorem original_price (h : 0.6 * x = a) : x = (5 / 3) * a :=
sorry

end NUMINAMATH_GPT_original_price_l2392_239253


namespace NUMINAMATH_GPT_sin_symmetry_value_l2392_239201

theorem sin_symmetry_value (ϕ : ℝ) (hϕ₀ : 0 < ϕ) (hϕ₁ : ϕ < π / 2) :
  ϕ = 5 * π / 12 :=
sorry

end NUMINAMATH_GPT_sin_symmetry_value_l2392_239201


namespace NUMINAMATH_GPT_age_of_15th_student_l2392_239218

theorem age_of_15th_student (avg_age_15_students avg_age_5_students avg_age_9_students : ℕ)
  (total_students total_age_15_students total_age_5_students total_age_9_students : ℕ)
  (h1 : total_students = 15)
  (h2 : avg_age_15_students = 15)
  (h3 : avg_age_5_students = 14)
  (h4 : avg_age_9_students = 16)
  (h5 : total_age_15_students = total_students * avg_age_15_students)
  (h6 : total_age_5_students = 5 * avg_age_5_students)
  (h7 : total_age_9_students = 9 * avg_age_9_students):
  total_age_15_students = total_age_5_students + total_age_9_students + 11 :=
by
  sorry

end NUMINAMATH_GPT_age_of_15th_student_l2392_239218


namespace NUMINAMATH_GPT_length_RS_14_l2392_239299

-- Definitions of conditions
def edges : List ℕ := [8, 14, 19, 28, 37, 42]
def PQ_length : ℕ := 42

-- Problem statement
theorem length_RS_14 (edges : List ℕ) (PQ_length : ℕ) (h : PQ_length = 42) (h_edges : edges = [8, 14, 19, 28, 37, 42]) :
  ∃ RS_length : ℕ, RS_length ∈ edges ∧ RS_length = 14 :=
by
  sorry

end NUMINAMATH_GPT_length_RS_14_l2392_239299


namespace NUMINAMATH_GPT_find_positive_integer_l2392_239233

theorem find_positive_integer (x : ℕ) (h1 : (10 * x + 4) % (x + 4) = 0) (h2 : (10 * x + 4) / (x + 4) = x - 23) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integer_l2392_239233


namespace NUMINAMATH_GPT_number_of_zeros_of_g_l2392_239204

noncomputable def f (x a : ℝ) := Real.exp x * (x + a)

noncomputable def g (x a : ℝ) := f (x - a) a - x^2

theorem number_of_zeros_of_g (a : ℝ) :
  (if a < 1 then ∃! x, g x a = 0
   else if a = 1 then ∃! x1 x2, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0
   else ∃! x1 x2 x3, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0) := sorry

end NUMINAMATH_GPT_number_of_zeros_of_g_l2392_239204


namespace NUMINAMATH_GPT_not_partition_1985_1987_partition_1987_1989_l2392_239229

-- Define the number of squares in an L-shape
def squares_in_lshape : ℕ := 3

-- Question 1: Can 1985 x 1987 be partitioned into L-shapes?
def partition_1985_1987 (m n : ℕ) (L_shape_size : ℕ) : Prop :=
  ∃ k : ℕ, m * n = k * L_shape_size ∧ (m % L_shape_size = 0 ∨ n % L_shape_size = 0)

theorem not_partition_1985_1987 :
  ¬ partition_1985_1987 1985 1987 squares_in_lshape :=
by {
  -- Proof omitted
  sorry
}

-- Question 2: Can 1987 x 1989 be partitioned into L-shapes?
theorem partition_1987_1989 :
  partition_1985_1987 1987 1989 squares_in_lshape :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_not_partition_1985_1987_partition_1987_1989_l2392_239229


namespace NUMINAMATH_GPT_cone_volume_increase_l2392_239246

theorem cone_volume_increase (r h : ℝ) (k : ℝ) :
  let V := (1/3) * π * r^2 * h
  let h' := 2.60 * h
  let r' := r * (1 + k / 100)
  let V' := (1/3) * π * (r')^2 * h'
  let percentage_increase := ((V' / V) - 1) * 100
  percentage_increase = ((1 + k / 100)^2 * 2.60 - 1) * 100 :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_increase_l2392_239246


namespace NUMINAMATH_GPT_algebra_correct_option_B_l2392_239251

theorem algebra_correct_option_B (a b c : ℝ) (h : b * (c^2 + 1) ≠ 0) : 
  (a * (c^2 + 1)) / (b * (c^2 + 1)) = a / b := 
by
  -- Skipping the proof to focus on the statement
  sorry

end NUMINAMATH_GPT_algebra_correct_option_B_l2392_239251


namespace NUMINAMATH_GPT_pond_volume_l2392_239293

theorem pond_volume (L W H : ℝ) (hL : L = 20) (hW : W = 10) (hH : H = 5) : 
  L * W * H = 1000 :=
by
  rw [hL, hW, hH]
  norm_num

end NUMINAMATH_GPT_pond_volume_l2392_239293


namespace NUMINAMATH_GPT_fraction_meaningful_l2392_239283

theorem fraction_meaningful (a : ℝ) : (∃ b, b = 2 / (a + 1)) → a ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l2392_239283


namespace NUMINAMATH_GPT_jerry_liters_of_mustard_oil_l2392_239252

-- Definitions
def cost_per_liter_mustard_oil : ℕ := 13
def cost_per_pound_penne_pasta : ℕ := 4
def cost_per_pound_pasta_sauce : ℕ := 5
def total_money_jerry_had : ℕ := 50
def money_left_with_jerry : ℕ := 7
def pounds_of_penne_pasta : ℕ := 3
def pounds_of_pasta_sauce : ℕ := 1

-- Our goal is to calculate how many liters of mustard oil Jerry bought
theorem jerry_liters_of_mustard_oil : ℕ :=
  let cost_of_penne_pasta := pounds_of_penne_pasta * cost_per_pound_penne_pasta
  let cost_of_pasta_sauce := pounds_of_pasta_sauce * cost_per_pound_pasta_sauce
  let total_spent := total_money_jerry_had - money_left_with_jerry
  let spent_on_pasta_and_sauce := cost_of_penne_pasta + cost_of_pasta_sauce
  let spent_on_mustard_oil := total_spent - spent_on_pasta_and_sauce
  spent_on_mustard_oil / cost_per_liter_mustard_oil

example : jerry_liters_of_mustard_oil = 2 := by
  unfold jerry_liters_of_mustard_oil
  simp
  sorry

end NUMINAMATH_GPT_jerry_liters_of_mustard_oil_l2392_239252


namespace NUMINAMATH_GPT_simplify_expression_l2392_239286

theorem simplify_expression (a b : ℝ) (h1 : 2 * b - a < 3) (h2 : 2 * a - b < 5) : 
  -abs (2 * b - a - 7) - abs (b - 2 * a + 8) + abs (a + b - 9) = -6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2392_239286


namespace NUMINAMATH_GPT_percentage_markup_l2392_239240

theorem percentage_markup (selling_price cost_price : ℚ)
  (h_selling_price : selling_price = 8325)
  (h_cost_price : cost_price = 7239.13) :
  ((selling_price - cost_price) / cost_price) * 100 = 15 := 
sorry

end NUMINAMATH_GPT_percentage_markup_l2392_239240


namespace NUMINAMATH_GPT_clothing_store_earnings_l2392_239221

-- Definitions for the given conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def cost_per_shirt : ℕ := 10
def cost_per_jeans : ℕ := 2 * cost_per_shirt

-- Theorem statement
theorem clothing_store_earnings : 
  (num_shirts * cost_per_shirt + num_jeans * cost_per_jeans = 400) := 
sorry

end NUMINAMATH_GPT_clothing_store_earnings_l2392_239221


namespace NUMINAMATH_GPT_find_percent_l2392_239280

theorem find_percent (x y z : ℝ) (h1 : z * (x - y) = 0.15 * (x + y)) (h2 : y = 0.25 * x) : 
  z = 0.25 := 
sorry

end NUMINAMATH_GPT_find_percent_l2392_239280


namespace NUMINAMATH_GPT_quadratic_inequality_l2392_239298

noncomputable def quadratic_inequality_solution : Set ℝ :=
  {x | x < 2} ∪ {x | x > 4}

theorem quadratic_inequality (x : ℝ) : (x^2 - 6 * x + 8 > 0) ↔ (x ∈ quadratic_inequality_solution) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l2392_239298


namespace NUMINAMATH_GPT_ten_row_geometric_figure_has_286_pieces_l2392_239273

noncomputable def rods (rows : ℕ) : ℕ := 3 * rows * (rows + 1) / 2
noncomputable def connectors (rows : ℕ) : ℕ := (rows +1) * (rows + 2) / 2
noncomputable def squares (rows : ℕ) : ℕ := rows * (rows + 1) / 2

theorem ten_row_geometric_figure_has_286_pieces :
    rods 10 + connectors 10 + squares 10 = 286 := by
  sorry

end NUMINAMATH_GPT_ten_row_geometric_figure_has_286_pieces_l2392_239273


namespace NUMINAMATH_GPT_f_is_periodic_f_nat_exact_l2392_239287

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_eq (x y : ℝ) : f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom f_0_nonzero : f 0 ≠ 0
axiom f_1_zero : f 1 = 0

theorem f_is_periodic : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
  by
    use 4
    sorry

theorem f_nat_exact (n : ℕ) : f n = Real.cos (n * Real.pi / 2) :=
  by
    sorry

end NUMINAMATH_GPT_f_is_periodic_f_nat_exact_l2392_239287


namespace NUMINAMATH_GPT_sequence_an_sum_sequence_Tn_l2392_239212

theorem sequence_an (k c : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n, S n = k * c ^ n - k) (ha2 : a 2 = 4) (ha6 : a 6 = 8 * a 3) :
  ∀ n, a n = 2 ^ n :=
by
  -- Proof is assumed to be given
  sorry

theorem sum_sequence_Tn (a : ℕ → ℝ) (T : ℕ → ℝ)
  (ha : ∀ n, a n = 2 ^ n) :
  ∀ n, T n = (n - 1) * 2 ^ (n + 1) + 2 :=
by
  -- Proof is assumed to be given
  sorry

end NUMINAMATH_GPT_sequence_an_sum_sequence_Tn_l2392_239212


namespace NUMINAMATH_GPT_find_f2_l2392_239241

noncomputable def f (x : ℝ) : ℝ := (4*x + 2/x + 3) / 3

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x - f (1 / x) = 2 * x + 1) : f 2 = 4 :=
  by
  sorry

end NUMINAMATH_GPT_find_f2_l2392_239241


namespace NUMINAMATH_GPT_logical_equivalence_l2392_239228

variable (R S T : Prop)

theorem logical_equivalence :
  (R → ¬S ∧ ¬T) ↔ ((S ∨ T) → ¬R) :=
by
  sorry

end NUMINAMATH_GPT_logical_equivalence_l2392_239228


namespace NUMINAMATH_GPT_sandra_remaining_money_l2392_239230

def sandra_savings : ℝ := 10
def mother_contribution : ℝ := 4
def father_contribution : ℝ := 2 * mother_contribution
def candy_cost : ℝ := 0.5
def jelly_bean_cost : ℝ := 0.2
def num_candies : ℝ := 14
def num_jelly_beans : ℝ := 20

theorem sandra_remaining_money : (sandra_savings + mother_contribution + father_contribution) - (num_candies * candy_cost + num_jelly_beans * jelly_bean_cost) = 11 :=
by
  sorry

end NUMINAMATH_GPT_sandra_remaining_money_l2392_239230


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2392_239292

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ), (∀ n : ℕ, a (n+1) - a n = 2) → a 2 = 5 → (a 0 + a 1 + a 2 + a 3) = 24 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2392_239292


namespace NUMINAMATH_GPT_fraction_of_fraction_l2392_239272

theorem fraction_of_fraction:
  let a := (3:ℚ) / 4
  let b := (5:ℚ) / 12
  b / a = (5:ℚ) / 9 := by
  sorry

end NUMINAMATH_GPT_fraction_of_fraction_l2392_239272


namespace NUMINAMATH_GPT_dogwood_trees_l2392_239262

/-- There are 7 dogwood trees currently in the park. 
Park workers will plant 5 dogwood trees today. 
The park will have 16 dogwood trees when the workers are finished.
Prove that 4 dogwood trees will be planted tomorrow. --/
theorem dogwood_trees (x : ℕ) : 7 + 5 + x = 16 → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_dogwood_trees_l2392_239262


namespace NUMINAMATH_GPT_garden_furniture_costs_l2392_239202

theorem garden_furniture_costs (B T U : ℝ)
    (h1 : T + B + U = 765)
    (h2 : T = 2 * B)
    (h3 : U = 3 * B) :
    B = 127.5 ∧ T = 255 ∧ U = 382.5 :=
by
  sorry

end NUMINAMATH_GPT_garden_furniture_costs_l2392_239202


namespace NUMINAMATH_GPT_complement_of_A_l2392_239254

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}

theorem complement_of_A : U \ A = {2, 4, 6} := 
by 
  sorry

end NUMINAMATH_GPT_complement_of_A_l2392_239254


namespace NUMINAMATH_GPT_lengths_equal_l2392_239270

-- a rhombus AFCE inscribed in a rectangle ABCD
variables {A B C D E F : Type}
variables {width length perimeter side_BF side_DE : ℝ}
variables {AF CE FC AF_side FC_side : ℝ}
variables {h1 : width = 20} {h2 : length = 25} {h3 : perimeter = 82}
variables {h4 : side_BF = (82 / 4 - 20)} {h5 : side_DE = (82 / 4 - 20)} 

-- prove that the lengths of BF and DE are equal
theorem lengths_equal :
  side_BF = side_DE :=
by
  sorry

end NUMINAMATH_GPT_lengths_equal_l2392_239270


namespace NUMINAMATH_GPT_find_ratio_AF_FB_l2392_239208

-- Define the vector space over reals
variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions of points A, B, C, D, F, P
variables (a b c d f p : V)

-- Given conditions as hypotheses
variables (h1 : (p = 2 / 5 • a + 3 / 5 • d))
variables (h2 : (p = 5 / 7 • f + 2 / 7 • c))
variables (hd : (d = 1 / 3 • b + 2 / 3 • c))
variables (hf : (f = 1 / 4 • a + 3 / 4 • b))

-- Theorem statement
theorem find_ratio_AF_FB : (41 : ℝ) / 15 = (41 : ℝ) / 15 := 
by sorry

end NUMINAMATH_GPT_find_ratio_AF_FB_l2392_239208


namespace NUMINAMATH_GPT_quadratic_function_expr_value_of_b_minimum_value_of_m_l2392_239234

-- Problem 1: Proving the quadratic function expression
theorem quadratic_function_expr (x : ℝ) (b c : ℝ)
  (h1 : (0:ℝ) = x^2 + b * 0 + c)
  (h2 : -b / 2 = (1:ℝ)) :
  x^2 - 2 * x + 4 = x^2 + b * x + c := sorry

-- Problem 2: Proving specific values of b
theorem value_of_b (b c : ℝ)
  (h1 : b^2 - c = 0)
  (h2 : ∀ x : ℝ, (b - 3 ≤ x ∧ x ≤ b → (x^2 + b * x + c ≥ 21))) :
  b = -Real.sqrt 7 ∨ b = 4 := sorry

-- Problem 3: Proving the minimum value of m
theorem minimum_value_of_m (x : ℝ) (m : ℝ)
  (h1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x^2 + x + m ≥ x^2 - 2 * x + 4) :
  m = 4 := sorry

end NUMINAMATH_GPT_quadratic_function_expr_value_of_b_minimum_value_of_m_l2392_239234


namespace NUMINAMATH_GPT_james_carrot_sticks_l2392_239250

theorem james_carrot_sticks (carrots_before : ℕ) (carrots_after : ℕ) 
(h_before : carrots_before = 22) (h_after : carrots_after = 15) : 
carrots_before + carrots_after = 37 := 
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_james_carrot_sticks_l2392_239250


namespace NUMINAMATH_GPT_bucket_p_fill_time_l2392_239266

theorem bucket_p_fill_time (capacity_P capacity_Q drum_capacity turns : ℕ)
  (h1 : capacity_P = 3 * capacity_Q)
  (h2 : drum_capacity = 45 * (capacity_P + capacity_Q))
  (h3 : bucket_fill_turns = drum_capacity / capacity_P) :
  bucket_fill_turns = 60 :=
by
  sorry

end NUMINAMATH_GPT_bucket_p_fill_time_l2392_239266


namespace NUMINAMATH_GPT_surface_area_hemisphere_radius_1_l2392_239288

noncomputable def surface_area_hemisphere (r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + Real.pi * r^2

theorem surface_area_hemisphere_radius_1 :
  surface_area_hemisphere 1 = 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_surface_area_hemisphere_radius_1_l2392_239288


namespace NUMINAMATH_GPT_jerry_needs_money_l2392_239223

theorem jerry_needs_money
  (jerry_has : ℕ := 7)
  (total_needed : ℕ := 16)
  (cost_per_figure : ℕ := 8) :
  (total_needed - jerry_has) * cost_per_figure = 72 :=
by
  sorry

end NUMINAMATH_GPT_jerry_needs_money_l2392_239223


namespace NUMINAMATH_GPT_daria_needs_to_earn_more_money_l2392_239222

noncomputable def moneyNeeded (ticket_cost : ℕ) (discount : ℕ) (gift_card : ℕ) 
  (transport_cost : ℕ) (parking_cost : ℕ) (tshirt_cost : ℕ) (current_money : ℕ) (tickets : ℕ) : ℕ :=
  let discounted_ticket_price := ticket_cost - (ticket_cost * discount / 100)
  let total_ticket_cost := discounted_ticket_price * tickets
  let ticket_cost_after_gift_card := total_ticket_cost - gift_card
  let total_cost := ticket_cost_after_gift_card + transport_cost + parking_cost + tshirt_cost
  total_cost - current_money

theorem daria_needs_to_earn_more_money :
  moneyNeeded 90 10 50 20 10 25 189 6 = 302 :=
by
  sorry

end NUMINAMATH_GPT_daria_needs_to_earn_more_money_l2392_239222


namespace NUMINAMATH_GPT_jesse_started_with_l2392_239206

-- Define the conditions
variables (g e : ℕ)

-- Theorem stating that given the conditions, Jesse started with 78 pencils
theorem jesse_started_with (g e : ℕ) (h1 : g = 44) (h2 : e = 34) : e + g = 78 :=
by sorry

end NUMINAMATH_GPT_jesse_started_with_l2392_239206


namespace NUMINAMATH_GPT_sasha_made_an_error_l2392_239284

theorem sasha_made_an_error :
  ∀ (f : ℕ → ℤ), 
  (∀ n, 1 ≤ n → n ≤ 9 → f n = n ∨ f n = -n) →
  (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 21) →
  (f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 = 20) →
  false :=
by
  intros f h_cons h_volodya_sum h_sasha_sum
  sorry

end NUMINAMATH_GPT_sasha_made_an_error_l2392_239284


namespace NUMINAMATH_GPT_tangent_line_parallel_points_l2392_239209

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Prove the points where the derivative equals 4
theorem tangent_line_parallel_points :
  ∃ (P0 : ℝ × ℝ), P0 = (1, 0) ∨ P0 = (-1, -4) ∧ (f' P0.fst = 4) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_parallel_points_l2392_239209


namespace NUMINAMATH_GPT_sqrt_domain_l2392_239255

theorem sqrt_domain (x : ℝ) : 3 * x - 6 ≥ 0 ↔ x ≥ 2 := sorry

end NUMINAMATH_GPT_sqrt_domain_l2392_239255


namespace NUMINAMATH_GPT_action_figures_more_than_books_proof_l2392_239205

-- Definitions for the conditions
def books := 3
def action_figures_initial := 4
def action_figures_added := 2

-- Definition for the total action figures
def action_figures_total := action_figures_initial + action_figures_added

-- Definition for the number difference
def action_figures_more_than_books := action_figures_total - books

-- Proof statement
theorem action_figures_more_than_books_proof : action_figures_more_than_books = 3 :=
by
  sorry

end NUMINAMATH_GPT_action_figures_more_than_books_proof_l2392_239205


namespace NUMINAMATH_GPT_truck_distance_l2392_239242

theorem truck_distance (V_t : ℝ) (D : ℝ) (h1 : D = V_t * 8) (h2 : D = (V_t + 18) * 5) : D = 240 :=
by
  sorry

end NUMINAMATH_GPT_truck_distance_l2392_239242


namespace NUMINAMATH_GPT_matrix_expression_solution_l2392_239281

theorem matrix_expression_solution (x : ℝ) :
  let a := 3 * x + 1
  let b := x + 1
  let c := 2
  let d := 2 * x
  ab - cd = 5 :=
by
  sorry

end NUMINAMATH_GPT_matrix_expression_solution_l2392_239281


namespace NUMINAMATH_GPT_circle_line_intersection_l2392_239261

theorem circle_line_intersection (x y a : ℝ) (A B C O : ℝ × ℝ) :
  (x + y = 1) ∧ ((x^2 + y^2) = a) ∧ 
  (O = (0, 0)) ∧ 
  (x^2 + y^2 = a ∧ (A.1^2 + A.2^2 = a) ∧ (B.1^2 + B.2^2 = a) ∧ (C.1^2 + C.2^2 = a) ∧ 
  (A.1 + B.1 = C.1) ∧ (A.2 + B.2 = C.2)) -> 
  a = 2 := 
sorry

end NUMINAMATH_GPT_circle_line_intersection_l2392_239261


namespace NUMINAMATH_GPT_number_of_jars_good_for_sale_l2392_239249

def numberOfGoodJars (initialCartons : Nat) (cartonsNotDelivered : Nat) (jarsPerCarton : Nat)
  (damagedJarsPerCarton : Nat) (numberOfDamagedCartons : Nat) (oneTotallyDamagedCarton : Nat) : Nat := 
  let deliveredCartons := initialCartons - cartonsNotDelivered
  let totalJars := deliveredCartons * jarsPerCarton
  let damagedJars := (damagedJarsPerCarton * numberOfDamagedCartons) + oneTotallyDamagedCarton
  totalJars - damagedJars

theorem number_of_jars_good_for_sale : 
  numberOfGoodJars 50 20 20 3 5 20 = 565 :=
by
  sorry

end NUMINAMATH_GPT_number_of_jars_good_for_sale_l2392_239249


namespace NUMINAMATH_GPT_ratio_of_speeds_l2392_239259

def eddy_time := 3
def eddy_distance := 480
def freddy_time := 4
def freddy_distance := 300

def eddy_speed := eddy_distance / eddy_time
def freddy_speed := freddy_distance / freddy_time

theorem ratio_of_speeds : (eddy_speed / freddy_speed) = 32 / 15 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l2392_239259


namespace NUMINAMATH_GPT_find_greater_solution_of_quadratic_l2392_239210

theorem find_greater_solution_of_quadratic:
  (x^2 + 14 * x - 88 = 0 → x = -22 ∨ x = 4) → (∀ x₁ x₂, (x₁ = -22 ∨ x₁ = 4) ∧ (x₂ = -22 ∨ x₂ = 4) → max x₁ x₂ = 4) :=
by
  intros h x₁ x₂ hx1x2
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_greater_solution_of_quadratic_l2392_239210


namespace NUMINAMATH_GPT_trace_bag_weight_l2392_239276

-- Definitions for the given problem
def weight_gordon_bag1 := 3
def weight_gordon_bag2 := 7
def total_weight_gordon := weight_gordon_bag1 + weight_gordon_bag2

noncomputable def weight_trace_one_bag : ℕ :=
  sorry

-- Theorem for what we need to prove
theorem trace_bag_weight :
  total_weight_gordon = 10 ∧
  weight_trace_one_bag = total_weight_gordon / 5 :=
sorry

end NUMINAMATH_GPT_trace_bag_weight_l2392_239276


namespace NUMINAMATH_GPT_gcf_75_90_l2392_239248

theorem gcf_75_90 : Nat.gcd 75 90 = 15 :=
by
  sorry

end NUMINAMATH_GPT_gcf_75_90_l2392_239248


namespace NUMINAMATH_GPT_students_accommodated_l2392_239235

theorem students_accommodated 
  (total_students : ℕ)
  (total_workstations : ℕ)
  (workstations_accommodating_x_students : ℕ)
  (x : ℕ)
  (workstations_accommodating_3_students : ℕ)
  (workstation_capacity_10 : ℕ)
  (workstation_capacity_6 : ℕ) :
  total_students = 38 → 
  total_workstations = 16 → 
  workstations_accommodating_x_students = 10 → 
  workstations_accommodating_3_students = 6 → 
  workstation_capacity_10 = 10 * x → 
  workstation_capacity_6 = 6 * 3 → 
  10 * x + 18 = 38 → 
  10 * 2 = 20 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_students_accommodated_l2392_239235


namespace NUMINAMATH_GPT_George_says_365_l2392_239211

-- Definitions based on conditions
def skips_Alice (n : Nat) : Prop :=
  ∃ k, n = 3 * k - 1

def skips_Barbara (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * k - 1) - 1
  
def skips_Candice (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * k - 1) - 1) - 1

def skips_Debbie (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1

def skips_Eliza (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1

def skips_Fatima (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1) - 1

def numbers_said_by_students (n : Nat) : Prop :=
  skips_Alice n ∨ skips_Barbara n ∨ skips_Candice n ∨ skips_Debbie n ∨ skips_Eliza n ∨ skips_Fatima n

-- The proof statement
theorem George_says_365 : ¬numbers_said_by_students 365 :=
sorry

end NUMINAMATH_GPT_George_says_365_l2392_239211


namespace NUMINAMATH_GPT_iron_wire_square_rectangle_l2392_239256

theorem iron_wire_square_rectangle 
  (total_length : ℕ) 
  (rect_length : ℕ) 
  (h1 : total_length = 28) 
  (h2 : rect_length = 12) :
  (total_length / 4 = 7) ∧
  ((total_length / 2) - rect_length = 2) :=
by 
  sorry

end NUMINAMATH_GPT_iron_wire_square_rectangle_l2392_239256


namespace NUMINAMATH_GPT_complex_expression_evaluation_l2392_239294

theorem complex_expression_evaluation (i : ℂ) (h : i^2 = -1) : i^3 * (1 - i)^2 = -2 :=
by
  -- Placeholder for the actual proof which is skipped here
  sorry

end NUMINAMATH_GPT_complex_expression_evaluation_l2392_239294


namespace NUMINAMATH_GPT_matthew_more_strawberries_than_betty_l2392_239216

noncomputable def B : ℕ := 16

theorem matthew_more_strawberries_than_betty (M N : ℕ) 
  (h1 : M > B)
  (h2 : M = 2 * N) 
  (h3 : B + M + N = 70) : M - B = 20 :=
by
  sorry

end NUMINAMATH_GPT_matthew_more_strawberries_than_betty_l2392_239216


namespace NUMINAMATH_GPT_expected_value_l2392_239271

noncomputable def p : ℝ := 0.25
noncomputable def P_xi_1 : ℝ := 0.24
noncomputable def P_black_bag_b : ℝ := 0.8
noncomputable def P_xi_0 : ℝ := (1 - p) * (1 - P_black_bag_b) * (1 - P_black_bag_b)
noncomputable def P_xi_2 : ℝ := p * (1 - P_black_bag_b) * (1 - P_black_bag_b) + (1 - p) * P_black_bag_b * P_black_bag_b
noncomputable def P_xi_3 : ℝ := p * P_black_bag_b + p * (1 - P_black_bag_b) * P_black_bag_b
noncomputable def E_xi : ℝ := 0 * P_xi_0 + 1 * P_xi_1 + 2 * P_xi_2 + 3 * P_xi_3

theorem expected_value : E_xi = 1.94 := by
  sorry

end NUMINAMATH_GPT_expected_value_l2392_239271


namespace NUMINAMATH_GPT_rectangles_in_grid_squares_in_grid_l2392_239247

theorem rectangles_in_grid (h_lines : ℕ) (v_lines : ℕ) : h_lines = 31 → v_lines = 31 → 
  (∃ rect_count : ℕ, rect_count = 216225) :=
by
  intros h_lines_eq v_lines_eq
  sorry

theorem squares_in_grid (n : ℕ) : n = 31 → (∃ square_count : ℕ, square_count = 6975) :=
by
  intros n_eq
  sorry

end NUMINAMATH_GPT_rectangles_in_grid_squares_in_grid_l2392_239247


namespace NUMINAMATH_GPT_horner_v3_at_2_l2392_239264

-- Defining the polynomial f(x).
def f (x : ℝ) := 2 * x^5 + 3 * x^3 - 2 * x^2 + x - 1

-- Defining the Horner's method evaluation up to v3 at x = 2.
def horner_eval (x : ℝ) := (((2 * x + 0) * x + 3) * x - 2) * x + 1

-- The proof statement we need to show.
theorem horner_v3_at_2 : horner_eval 2 = 20 := sorry

end NUMINAMATH_GPT_horner_v3_at_2_l2392_239264


namespace NUMINAMATH_GPT_no_solution_for_floor_x_plus_x_eq_15_point_3_l2392_239257

theorem no_solution_for_floor_x_plus_x_eq_15_point_3 : ¬ ∃ (x : ℝ), (⌊x⌋ : ℝ) + x = 15.3 := by
  sorry

end NUMINAMATH_GPT_no_solution_for_floor_x_plus_x_eq_15_point_3_l2392_239257


namespace NUMINAMATH_GPT_value_fraction_eq_three_l2392_239207

namespace Problem

variable {R : Type} [Field R]

theorem value_fraction_eq_three (a b c : R) (h : a / 2 = b / 3 ∧ b / 3 = c / 4) :
  (a + b + c) / (2 * a + b - c) = 3 := by
  sorry

end Problem

end NUMINAMATH_GPT_value_fraction_eq_three_l2392_239207


namespace NUMINAMATH_GPT_length_FJ_is_35_l2392_239244

noncomputable def length_of_FJ (h : ℝ) : ℝ :=
  let FG := 50
  let HI := 20
  let trapezium_area := (1 / 2) * (FG + HI) * h
  let half_trapezium_area := trapezium_area / 2
  let JI_area := (1 / 2) * 35 * h
  35

theorem length_FJ_is_35 (h : ℝ) : length_of_FJ h = 35 :=
  sorry

end NUMINAMATH_GPT_length_FJ_is_35_l2392_239244


namespace NUMINAMATH_GPT_negation_of_not_both_are_not_even_l2392_239263

variables {a b : ℕ}

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem negation_of_not_both_are_not_even :
  ¬ (¬ is_even a ∧ ¬ is_even b) ↔ (is_even a ∨ is_even b) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_not_both_are_not_even_l2392_239263


namespace NUMINAMATH_GPT_linear_inequality_solution_l2392_239213

theorem linear_inequality_solution (a b : ℝ)
  (h₁ : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ (x < -3 ∨ x > 1)) :
  ∀ x : ℝ, a * x + b < 0 ↔ x < 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_linear_inequality_solution_l2392_239213


namespace NUMINAMATH_GPT_find_x_value_l2392_239289

theorem find_x_value (b x : ℝ) (hb : b > 1) (hx : x > 0) 
    (heq: (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) : 
    x = 1 / 5 := 
sorry

end NUMINAMATH_GPT_find_x_value_l2392_239289


namespace NUMINAMATH_GPT_trains_cross_time_l2392_239268

def L : ℕ := 120 -- Length of each train in meters

def t1 : ℕ := 10 -- Time for the first train to cross the telegraph post in seconds
def t2 : ℕ := 12 -- Time for the second train to cross the telegraph post in seconds

def V1 : ℕ := L / t1 -- Speed of the first train (in m/s)
def V2 : ℕ := L / t2 -- Speed of the second train (in m/s)

def Vr : ℕ := V1 + V2 -- Relative speed when traveling in opposite directions

def TotalDistance : ℕ := 2 * L -- Total distance when both trains cross each other

def T : ℚ := TotalDistance / Vr -- Time for the trains to cross each other

theorem trains_cross_time : T = 11 := sorry

end NUMINAMATH_GPT_trains_cross_time_l2392_239268


namespace NUMINAMATH_GPT_new_class_average_l2392_239227

theorem new_class_average (total_students : ℕ) (students_group1 : ℕ) (avg1 : ℝ) (students_group2 : ℕ) (avg2 : ℝ) : 
  total_students = 40 → students_group1 = 28 → avg1 = 68 → students_group2 = 12 → avg2 = 77 → 
  ((students_group1 * avg1 + students_group2 * avg2) / total_students) = 70.7 :=
by
  sorry

end NUMINAMATH_GPT_new_class_average_l2392_239227


namespace NUMINAMATH_GPT_sum_of_first_ten_terms_l2392_239258

theorem sum_of_first_ten_terms (a1 d : ℝ) (h1 : 3 * (a1 + d) = 15) 
  (h2 : (a1 + d - 1) ^ 2 = (a1 - 1) * (a1 + 2 * d + 1)) : 
  (10 / 2) * (2 * a1 + (10 - 1) * d) = 120 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_first_ten_terms_l2392_239258


namespace NUMINAMATH_GPT_evaluate_expression_l2392_239224

theorem evaluate_expression : (36 + 12) / (6 - (2 + 1)) = 16 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2392_239224


namespace NUMINAMATH_GPT_find_alpha_after_five_operations_l2392_239245

def returns_to_starting_point_after_operations (α : Real) (n : Nat) : Prop :=
  (n * α) % 360 = 0

theorem find_alpha_after_five_operations (α : Real) 
  (hα1 : 0 < α)
  (hα2 : α < 180)
  (h_return : returns_to_starting_point_after_operations α 5) :
  α = 72 ∨ α = 144 :=
sorry

end NUMINAMATH_GPT_find_alpha_after_five_operations_l2392_239245


namespace NUMINAMATH_GPT_diet_soda_bottles_l2392_239200

theorem diet_soda_bottles (r d l t : Nat) (h1 : r = 49) (h2 : l = 6) (h3 : t = 89) (h4 : t = r + d) : d = 40 :=
by
  sorry

end NUMINAMATH_GPT_diet_soda_bottles_l2392_239200


namespace NUMINAMATH_GPT_minimize_distance_l2392_239285

noncomputable def f (x : ℝ) := 9 * x^3
noncomputable def g (x : ℝ) := Real.log x

theorem minimize_distance :
  ∃ m > 0, (∀ x > 0, |f m - g m| ≤ |f x - g x|) ∧ m = 1/3 :=
sorry

end NUMINAMATH_GPT_minimize_distance_l2392_239285


namespace NUMINAMATH_GPT_rowing_upstream_speed_l2392_239279

theorem rowing_upstream_speed 
  (V_m : ℝ) (V_downstream : ℝ) (V_upstream : ℝ)
  (hyp1 : V_m = 30)
  (hyp2 : V_downstream = 35) :
  V_upstream = V_m - (V_downstream - V_m) := 
  sorry

end NUMINAMATH_GPT_rowing_upstream_speed_l2392_239279


namespace NUMINAMATH_GPT_solve_trig_eq_l2392_239296

open Real

theorem solve_trig_eq (n : ℤ) (x : ℝ) : 
  (sin x) ^ 4 + (cos x) ^ 4 = (sin (2 * x)) ^ 4 + (cos (2 * x)) ^ 4 ↔ x = (n : ℝ) * π / 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_trig_eq_l2392_239296


namespace NUMINAMATH_GPT_white_roses_per_bouquet_l2392_239290

/-- Mrs. Dunbar needs to make 5 bouquets and 7 table decorations. -/
def number_of_bouquets : ℕ := 5
def number_of_table_decorations : ℕ := 7
/-- She uses 12 white roses in each table decoration. -/
def white_roses_per_table_decoration : ℕ := 12
/-- She needs a total of 109 white roses to complete all bouquets and table decorations. -/
def total_white_roses_needed : ℕ := 109

/-- Prove that the number of white roses used in each bouquet is 5. -/
theorem white_roses_per_bouquet : ∃ (white_roses_per_bouquet : ℕ),
  number_of_bouquets * white_roses_per_bouquet + number_of_table_decorations * white_roses_per_table_decoration = total_white_roses_needed
  ∧ white_roses_per_bouquet = 5 := 
by
  sorry

end NUMINAMATH_GPT_white_roses_per_bouquet_l2392_239290


namespace NUMINAMATH_GPT_bounce_height_less_than_two_l2392_239277

theorem bounce_height_less_than_two (k : ℕ) (h₀ : ℝ) (r : ℝ) (ε : ℝ) 
    (h₀_pos : h₀ = 20) (r_pos : r = 1/2) (ε_pos : ε = 2): 
  (h₀ * (r ^ k) < ε) ↔ k >= 4 := by
  sorry

end NUMINAMATH_GPT_bounce_height_less_than_two_l2392_239277


namespace NUMINAMATH_GPT_largest_multiple_of_8_less_than_100_l2392_239214

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end NUMINAMATH_GPT_largest_multiple_of_8_less_than_100_l2392_239214


namespace NUMINAMATH_GPT_prudence_nap_is_4_hours_l2392_239219

def prudence_nap_length (total_sleep : ℕ) (weekdays_sleep : ℕ) (weekend_sleep : ℕ) (weeks : ℕ) (total_weeks : ℕ) : ℕ :=
  (total_sleep - (weekdays_sleep + weekend_sleep) * total_weeks) / (2 * total_weeks)

theorem prudence_nap_is_4_hours
  (total_sleep weekdays_sleep weekend_sleep total_weeks : ℕ) :
  total_sleep = 200 ∧ weekdays_sleep = 5 * 6 ∧ weekend_sleep = 2 * 9 ∧ total_weeks = 4 →
  prudence_nap_length total_sleep weekdays_sleep weekend_sleep total_weeks total_weeks = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_prudence_nap_is_4_hours_l2392_239219


namespace NUMINAMATH_GPT_total_fish_l2392_239225

theorem total_fish (fish_Lilly fish_Rosy : ℕ) (hL : fish_Lilly = 10) (hR : fish_Rosy = 8) : fish_Lilly + fish_Rosy = 18 := 
by 
  sorry

end NUMINAMATH_GPT_total_fish_l2392_239225


namespace NUMINAMATH_GPT_pizzasServedDuringDinner_l2392_239260

-- Definitions based on the conditions
def pizzasServedDuringLunch : ℕ := 9
def totalPizzasServedToday : ℕ := 15

-- Theorem statement
theorem pizzasServedDuringDinner : 
  totalPizzasServedToday - pizzasServedDuringLunch = 6 := 
  by 
    sorry

end NUMINAMATH_GPT_pizzasServedDuringDinner_l2392_239260


namespace NUMINAMATH_GPT_prism_edges_l2392_239237

theorem prism_edges (n : ℕ) (h1 : n > 310) (h2 : n < 320) (h3 : n % 2 = 1) : n = 315 := by
  sorry

end NUMINAMATH_GPT_prism_edges_l2392_239237
