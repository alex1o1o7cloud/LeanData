import Mathlib

namespace NUMINAMATH_GPT_line_ellipse_common_point_l962_96224

theorem line_ellipse_common_point (k : ℝ) (m : ℝ) :
  (∀ (x y : ℝ), y = k * x + 1 →
    (y^2 / m + x^2 / 5 ≤ 1)) ↔ (m ≥ 1 ∧ m ≠ 5) :=
by sorry

end NUMINAMATH_GPT_line_ellipse_common_point_l962_96224


namespace NUMINAMATH_GPT_geometric_sequence_sum_l962_96204

theorem geometric_sequence_sum (a : ℕ → ℤ)
  (h1 : a 0 = 1)
  (h_q : ∀ n, a (n + 1) = a n * -2) :
  a 0 + |a 1| + a 2 + |a 3| = 15 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l962_96204


namespace NUMINAMATH_GPT_price_difference_l962_96225

noncomputable def originalPriceStrawberries (s : ℝ) (sale_revenue_s : ℝ) := sale_revenue_s / (0.70 * s)
noncomputable def originalPriceBlueberries (b : ℝ) (sale_revenue_b : ℝ) := sale_revenue_b / (0.80 * b)

theorem price_difference
    (s : ℝ) (sale_revenue_s : ℝ)
    (b : ℝ) (sale_revenue_b : ℝ)
    (h1 : sale_revenue_s = 70 * (0.70 * s))
    (h2 : sale_revenue_b = 50 * (0.80 * b)) :
    originalPriceStrawberries (sale_revenue_s / 49) sale_revenue_s - originalPriceBlueberries (sale_revenue_b / 40) sale_revenue_b = 0.71 :=
by
  sorry

end NUMINAMATH_GPT_price_difference_l962_96225


namespace NUMINAMATH_GPT_greatest_number_of_roses_l962_96215

noncomputable def individual_rose_price: ℝ := 2.30
noncomputable def dozen_rose_price: ℝ := 36
noncomputable def two_dozen_rose_price: ℝ := 50
noncomputable def budget: ℝ := 680

theorem greatest_number_of_roses (P: ℝ → ℝ → ℝ → ℝ → ℕ) :
  P individual_rose_price dozen_rose_price two_dozen_rose_price budget = 325 :=
sorry

end NUMINAMATH_GPT_greatest_number_of_roses_l962_96215


namespace NUMINAMATH_GPT_juhye_initial_money_l962_96251

theorem juhye_initial_money
  (M : ℝ)
  (h1 : M - (1 / 4) * M - (2 / 3) * ((3 / 4) * M) = 2500) :
  M = 10000 := by
  sorry

end NUMINAMATH_GPT_juhye_initial_money_l962_96251


namespace NUMINAMATH_GPT_calculate_value_l962_96291

theorem calculate_value :
  let X := (354 * 28) ^ 2
  let Y := (48 * 14) ^ 2
  (X * 9) / (Y * 2) = 2255688 :=
by
  sorry

end NUMINAMATH_GPT_calculate_value_l962_96291


namespace NUMINAMATH_GPT_find_value_of_xy_plus_yz_plus_xz_l962_96294

variable (x y z : ℝ)

-- Conditions
def cond1 : Prop := x^2 + x * y + y^2 = 108
def cond2 : Prop := y^2 + y * z + z^2 = 64
def cond3 : Prop := z^2 + x * z + x^2 = 172

-- Theorem statement
theorem find_value_of_xy_plus_yz_plus_xz (hx : cond1 x y) (hy : cond2 y z) (hz : cond3 z x) : 
  x * y + y * z + x * z = 96 :=
sorry

end NUMINAMATH_GPT_find_value_of_xy_plus_yz_plus_xz_l962_96294


namespace NUMINAMATH_GPT_find_remainder_q_neg2_l962_96236

-- Define q(x)
def q (x : ℝ) (D E F : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 6

-- The given conditions in the problem
variable {D E F : ℝ}
variable (h_q_2 : q 2 D E F = 14)

-- The statement we aim to prove
theorem find_remainder_q_neg2 (h_q_2 : q 2 D E F = 14) : q (-2) D E F = 14 :=
sorry

end NUMINAMATH_GPT_find_remainder_q_neg2_l962_96236


namespace NUMINAMATH_GPT_poster_distance_from_wall_end_l962_96246

theorem poster_distance_from_wall_end (w_wall w_poster : ℝ) (h1 : w_wall = 25) (h2 : w_poster = 4) (h3 : 2 * x + w_poster = w_wall) : x = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_poster_distance_from_wall_end_l962_96246


namespace NUMINAMATH_GPT_min_value_a4b3c2_l962_96269

theorem min_value_a4b3c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : 1/a + 1/b + 1/c = 9) : (∀ a b c : ℝ, a^4 * b^3 * c^2 ≥ 1/(9^9)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_a4b3c2_l962_96269


namespace NUMINAMATH_GPT_find_flights_of_stairs_l962_96257

def t_flight : ℕ := 11
def t_bomb : ℕ := 72
def t_spent : ℕ := 165
def t_diffuse : ℕ := 17

def total_time_running : ℕ := t_spent + (t_bomb - t_diffuse)
def flights_of_stairs (t_run: ℕ) (time_per_flight: ℕ) : ℕ := t_run / time_per_flight

theorem find_flights_of_stairs :
  flights_of_stairs total_time_running t_flight = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_flights_of_stairs_l962_96257


namespace NUMINAMATH_GPT_first_part_lending_years_l962_96285

-- Definitions and conditions from the problem
def total_sum : ℕ := 2691
def second_part : ℕ := 1656
def rate_first_part : ℚ := 3 / 100
def rate_second_part : ℚ := 5 / 100
def time_second_part : ℕ := 3

-- Calculated first part
def first_part : ℕ := total_sum - second_part

-- Prove that the number of years (n) the first part is lent is 8
theorem first_part_lending_years : 
  ∃ n : ℕ, (first_part : ℚ) * rate_first_part * n = (second_part : ℚ) * rate_second_part * time_second_part ∧ n = 8 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_first_part_lending_years_l962_96285


namespace NUMINAMATH_GPT_five_fridays_in_september_l962_96256

theorem five_fridays_in_september (year : ℕ) :
  (∃ (july_wednesdays : ℕ × ℕ × ℕ × ℕ × ℕ), 
     (july_wednesdays = (1, 8, 15, 22, 29) ∨ 
      july_wednesdays = (2, 9, 16, 23, 30) ∨ 
      july_wednesdays = (3, 10, 17, 24, 31)) ∧ 
      september_days = 30) → 
  ∃ (september_fridays : ℕ × ℕ × ℕ × ℕ × ℕ), 
  (september_fridays = (1, 8, 15, 22, 29)) :=
by
  sorry

end NUMINAMATH_GPT_five_fridays_in_september_l962_96256


namespace NUMINAMATH_GPT_find_number_l962_96229

theorem find_number (N x : ℕ) (h1 : 3 * x = (N - x) + 26) (h2 : x = 22) : N = 62 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l962_96229


namespace NUMINAMATH_GPT_comparison_abc_l962_96208

variable (f : Real → Real)
variable (a b c : Real)
variable (x : Real)
variable (h_even : ∀ x, f (-x + 1) = f (x + 1))
variable (h_periodic : ∀ x, f (x + 2) = f x)
variable (h_mono : ∀ x y, 0 < x ∧ y < 1 ∧ x < y → f x < f y)
variable (h_f0 : f 0 = 0)
variable (a_def : a = f (Real.log 2))
variable (b_def : b = f (Real.log 3))
variable (c_def : c = f 0.5)

theorem comparison_abc : b > a ∧ a > c :=
sorry

end NUMINAMATH_GPT_comparison_abc_l962_96208


namespace NUMINAMATH_GPT_exists_subset_sum_mod_p_l962_96279

theorem exists_subset_sum_mod_p (p : ℕ) (hp : Nat.Prime p) (A : Finset ℕ)
  (hA_card : A.card = p - 1) (hA : ∀ a ∈ A, a % p ≠ 0) : 
  ∀ n : ℕ, n < p → ∃ B ⊆ A, (B.sum id) % p = n :=
by
  sorry

end NUMINAMATH_GPT_exists_subset_sum_mod_p_l962_96279


namespace NUMINAMATH_GPT_length_chord_AB_l962_96244

-- Given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- Prove the length of the chord AB
theorem length_chord_AB : 
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧ A ≠ B) →
  (∃ (length : ℝ), length = 2*Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_length_chord_AB_l962_96244


namespace NUMINAMATH_GPT_number_drawn_from_first_group_l962_96205

theorem number_drawn_from_first_group (n: ℕ) (groups: ℕ) (interval: ℕ) (fourth_group_number: ℕ) (total_bags: ℕ) 
    (h1: total_bags = 50) (h2: groups = 5) (h3: interval = total_bags / groups)
    (h4: interval = 10) (h5: fourth_group_number = 36) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_drawn_from_first_group_l962_96205


namespace NUMINAMATH_GPT_function_relationship_selling_price_for_profit_max_profit_l962_96280

-- Step (1): Prove the function relationship between y and x
theorem function_relationship (x y: ℝ) (h1 : ∀ x, y = -2*x + 80)
  (h2 : x = 22 ∧ y = 36 ∨ x = 24 ∧ y = 32) :
  y = -2*x + 80 := by
  sorry

-- Step (2): Selling price per book for a 150 yuan profit per week
theorem selling_price_for_profit (x: ℝ) (hx : 20 ≤ x ∧ x ≤ 28) (profit : ℝ)
  (h_profit : profit = (x - 20) * (-2*x + 80)) (h2 : profit = 150) : 
  x = 25 := by
  sorry

-- Step (3): Maximizing the weekly profit
theorem max_profit (x w: ℝ) (hx : 20 ≤ x ∧ x ≤ 28) 
  (profit : ∀ x, w = (x - 20) * (-2*x + 80)) :
  w = 192 ∧ x = 28 := by
  sorry

end NUMINAMATH_GPT_function_relationship_selling_price_for_profit_max_profit_l962_96280


namespace NUMINAMATH_GPT_max_product_of_triangle_sides_l962_96223

theorem max_product_of_triangle_sides (a c : ℝ) (ha : a ≥ 0) (hc : c ≥ 0) :
  ∃ b : ℝ, b = 4 ∧ ∃ B : ℝ, B = 60 * (π / 180) ∧ a^2 + c^2 - a * c = b^2 ∧ a * c ≤ 16 :=
by
  sorry

end NUMINAMATH_GPT_max_product_of_triangle_sides_l962_96223


namespace NUMINAMATH_GPT_total_students_l962_96207

-- Defining the conditions
variable (H : ℕ) -- Number of students who ordered hot dogs
variable (students_ordered_burgers : ℕ) -- Number of students who ordered burgers

-- Given conditions
def burger_condition := students_ordered_burgers = 30
def hotdog_condition := students_ordered_burgers = 2 * H

-- Theorem to prove the total number of students
theorem total_students (H : ℕ) (students_ordered_burgers : ℕ) 
  (h1 : burger_condition students_ordered_burgers) 
  (h2 : hotdog_condition students_ordered_burgers H) : 
  students_ordered_burgers + H = 45 := 
by
  sorry

end NUMINAMATH_GPT_total_students_l962_96207


namespace NUMINAMATH_GPT_estimate_points_in_interval_l962_96200

-- Define the conditions
def total_data_points : ℕ := 1000
def frequency_interval : ℝ := 0.16
def interval_estimation : ℝ := total_data_points * frequency_interval

-- Lean theorem statement
theorem estimate_points_in_interval : interval_estimation = 160 :=
by
  sorry

end NUMINAMATH_GPT_estimate_points_in_interval_l962_96200


namespace NUMINAMATH_GPT_fraction_simplification_l962_96220

theorem fraction_simplification :
  (2/5 + 3/4) / (4/9 + 1/6) = (207/110) := by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l962_96220


namespace NUMINAMATH_GPT_decreasing_cubic_function_l962_96249

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^3 - x

-- Define the condition that f is decreasing on (-∞, ∞)
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

-- The main theorem that needs to be proven
theorem decreasing_cubic_function (m : ℝ) : is_decreasing (f m) → m < 0 := 
by
  sorry

end NUMINAMATH_GPT_decreasing_cubic_function_l962_96249


namespace NUMINAMATH_GPT_remaining_savings_after_purchase_l962_96293

-- Definitions of the conditions
def cost_per_sweater : ℕ := 30
def num_sweaters : ℕ := 6
def cost_per_scarf : ℕ := 20
def num_scarves : ℕ := 6
def initial_savings : ℕ := 500

-- Theorem stating the remaining savings
theorem remaining_savings_after_purchase : initial_savings - ((cost_per_sweater * num_sweaters) + (cost_per_scarf * num_scarves)) = 200 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_remaining_savings_after_purchase_l962_96293


namespace NUMINAMATH_GPT_ratio_of_Victoria_to_Beacon_l962_96283

def Richmond_population : ℕ := 3000
def Beacon_population : ℕ := 500
def Victoria_population : ℕ := Richmond_population - 1000
def ratio_Victoria_Beacon : ℕ := Victoria_population / Beacon_population

theorem ratio_of_Victoria_to_Beacon : ratio_Victoria_Beacon = 4 := 
by
  unfold ratio_Victoria_Beacon Victoria_population Richmond_population Beacon_population
  sorry

end NUMINAMATH_GPT_ratio_of_Victoria_to_Beacon_l962_96283


namespace NUMINAMATH_GPT_find_x_l962_96263

-- Definitions used in conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (4, x)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Main statement of the problem to be proved
theorem find_x (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = -1) : x = -1 / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l962_96263


namespace NUMINAMATH_GPT_impossible_configuration_l962_96230

-- Define the initial state of stones in boxes
def stones_in_box (n : ℕ) : ℕ :=
  if n ≥ 1 ∧ n ≤ 100 then n else 0

-- Define the condition for moving stones between boxes
def can_move_stones (box1 box2 : ℕ) : Prop :=
  stones_in_box box1 + stones_in_box box2 = 101

-- The proposition: it is impossible to achieve the desired configuration
theorem impossible_configuration :
  ¬ ∃ boxes : ℕ → ℕ, 
    (boxes 70 = 69) ∧ 
    (boxes 50 = 51) ∧ 
    (∀ n, n ≠ 70 → n ≠ 50 → boxes n = stones_in_box n) ∧
    (∀ n1 n2, can_move_stones n1 n2 → (boxes n1 + boxes n2 = 101)) :=
sorry

end NUMINAMATH_GPT_impossible_configuration_l962_96230


namespace NUMINAMATH_GPT_line_intersects_ellipse_with_conditions_l962_96247

theorem line_intersects_ellipse_with_conditions :
  ∃ l : ℝ → ℝ, (∃ A B : ℝ × ℝ, 
  (A.fst^2/6 + A.snd^2/3 = 1 ∧ B.fst^2/6 + B.snd^2/3 = 1) ∧
  A.fst > 0 ∧ A.snd > 0 ∧ B.fst > 0 ∧ B.snd > 0 ∧
  (∃ M N : ℝ × ℝ, 
    M.snd = 0 ∧ N.fst = 0 ∧
    M.fst^2 + N.snd^2 = (2 * Real.sqrt 3)^2 ∧
    (M.snd - A.snd)^2 + (M.fst - A.fst)^2 = (N.fst - B.fst)^2 + (N.snd - B.snd)^2) ∧
    (∀ x, l x + Real.sqrt 2 * x - 2 * Real.sqrt 2 = 0)
) :=
sorry

end NUMINAMATH_GPT_line_intersects_ellipse_with_conditions_l962_96247


namespace NUMINAMATH_GPT_unique_solution_value_k_l962_96288

theorem unique_solution_value_k (k : ℚ) :
  (∀ x : ℚ, (x + 3) / (k * x - 2) = x → x = -2) ↔ k = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_value_k_l962_96288


namespace NUMINAMATH_GPT_binomial_equality_l962_96268

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end NUMINAMATH_GPT_binomial_equality_l962_96268


namespace NUMINAMATH_GPT_expected_winnings_is_correct_l962_96235

variable (prob_1 prob_23 prob_456 : ℚ)
variable (win_1 win_23 loss_456 : ℚ)

theorem expected_winnings_is_correct :
  prob_1 = 1/4 → 
  prob_23 = 1/2 → 
  prob_456 = 1/4 → 
  win_1 = 2 → 
  win_23 = 4 → 
  loss_456 = -3 → 
  (prob_1 * win_1 + prob_23 * win_23 + prob_456 * loss_456 = 1.75) :=
by
  intros
  sorry

end NUMINAMATH_GPT_expected_winnings_is_correct_l962_96235


namespace NUMINAMATH_GPT_min_value_of_b1_plus_b2_l962_96206

theorem min_value_of_b1_plus_b2 (b : ℕ → ℕ) (h1 : ∀ n ≥ 1, b (n + 2) = (b n + 4030) / (1 + b (n + 1)))
  (h2 : ∀ n, b n > 0) : ∃ b1 b2, b1 * b2 = 4030 ∧ b1 + b2 = 127 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_of_b1_plus_b2_l962_96206


namespace NUMINAMATH_GPT_find_n_l962_96240

theorem find_n (n : ℤ) (h : 8 + 6 = n + 8) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l962_96240


namespace NUMINAMATH_GPT_common_difference_l962_96211

theorem common_difference (a1 d : ℕ) (S3 : ℕ) (h1 : S3 = 6) (h2 : a1 = 1)
  (h3 : S3 = 3 * (2 * a1 + 2 * d) / 2) : d = 1 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_l962_96211


namespace NUMINAMATH_GPT_tan_identity_example_l962_96221

theorem tan_identity_example (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_identity_example_l962_96221


namespace NUMINAMATH_GPT_rectangle_width_length_ratio_l962_96227

theorem rectangle_width_length_ratio (w : ℕ) (h : w + 10 = 15) : w / 10 = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_rectangle_width_length_ratio_l962_96227


namespace NUMINAMATH_GPT_remainder_problem_l962_96252

theorem remainder_problem (d r : ℤ) (h1 : 1237 % d = r)
    (h2 : 1694 % d = r) (h3 : 2791 % d = r) (hd : d > 1) :
    d - r = 134 := sorry

end NUMINAMATH_GPT_remainder_problem_l962_96252


namespace NUMINAMATH_GPT_noah_total_wattage_l962_96271

def bedroom_wattage := 6
def office_wattage := 3 * bedroom_wattage
def living_room_wattage := 4 * bedroom_wattage
def hours_on := 2

theorem noah_total_wattage : 
  bedroom_wattage * hours_on + 
  office_wattage * hours_on + 
  living_room_wattage * hours_on = 96 := by
  sorry

end NUMINAMATH_GPT_noah_total_wattage_l962_96271


namespace NUMINAMATH_GPT_black_lambs_count_l962_96282

-- Definitions based on the conditions given
def total_lambs : ℕ := 6048
def white_lambs : ℕ := 193

-- Theorem statement
theorem black_lambs_count : total_lambs - white_lambs = 5855 :=
by 
  -- the proof would be provided here
  sorry

end NUMINAMATH_GPT_black_lambs_count_l962_96282


namespace NUMINAMATH_GPT_no_real_roots_of_quad_eq_l962_96278

theorem no_real_roots_of_quad_eq (k : ℝ) :
  ∀ x : ℝ, ¬ (x^2 - 2*x - k = 0) ↔ k < -1 :=
by sorry

end NUMINAMATH_GPT_no_real_roots_of_quad_eq_l962_96278


namespace NUMINAMATH_GPT_nicolai_ate_6_pounds_of_peaches_l962_96241

noncomputable def total_weight_pounds : ℝ := 8
noncomputable def pound_to_ounce : ℝ := 16
noncomputable def mario_weight_ounces : ℝ := 8
noncomputable def lydia_weight_ounces : ℝ := 24

theorem nicolai_ate_6_pounds_of_peaches :
  (total_weight_pounds * pound_to_ounce - (mario_weight_ounces + lydia_weight_ounces)) / pound_to_ounce = 6 :=
by
  sorry

end NUMINAMATH_GPT_nicolai_ate_6_pounds_of_peaches_l962_96241


namespace NUMINAMATH_GPT_rectangle_area_l962_96243

theorem rectangle_area (x y : ℝ) (L W : ℝ) (h_diagonal : (L ^ 2 + W ^ 2) ^ (1 / 2) = x + y) (h_ratio : L / W = 3 / 2) : 
  L * W = (6 * (x + y) ^ 2) / 13 := 
sorry

end NUMINAMATH_GPT_rectangle_area_l962_96243


namespace NUMINAMATH_GPT_tan_half_angle_lt_l962_96239

theorem tan_half_angle_lt (x : ℝ) (h : 0 < x ∧ x ≤ π / 2) : 
  Real.tan (x / 2) < x := 
by
  sorry

end NUMINAMATH_GPT_tan_half_angle_lt_l962_96239


namespace NUMINAMATH_GPT_accident_rate_is_100_million_l962_96201

theorem accident_rate_is_100_million (X : ℕ) (h1 : 96 * 3000000000 = 2880 * X) : X = 100000000 :=
by
  sorry

end NUMINAMATH_GPT_accident_rate_is_100_million_l962_96201


namespace NUMINAMATH_GPT_g_of_g_of_2_l962_96276

def g (x : ℝ) : ℝ := 4 * x^2 - 3

theorem g_of_g_of_2 : g (g 2) = 673 := 
by 
  sorry

end NUMINAMATH_GPT_g_of_g_of_2_l962_96276


namespace NUMINAMATH_GPT_cube_paint_problem_l962_96292

theorem cube_paint_problem : 
  ∀ (n : ℕ),
  n = 6 →
  (∃ k : ℕ, 216 = k^3 ∧ k = n) →
  ∀ (faces inner_faces total_cubelets : ℕ),
  faces = 6 →
  inner_faces = 4 →
  total_cubelets = faces * (inner_faces * inner_faces) →
  total_cubelets = 96 :=
by 
  intros n hn hc faces hfaces inner_faces hinner_faces total_cubelets htotal_cubelets
  sorry

end NUMINAMATH_GPT_cube_paint_problem_l962_96292


namespace NUMINAMATH_GPT_shirt_cost_l962_96209

variables (S : ℝ)

theorem shirt_cost (h : 2 * S + (S + 3) + (1/2) * (2 * S + S + 3) = 36) : S = 7.88 :=
sorry

end NUMINAMATH_GPT_shirt_cost_l962_96209


namespace NUMINAMATH_GPT_find_a7_a8_l962_96245

noncomputable def geometric_sequence_property (a : ℕ → ℝ) (r : ℝ) :=
∀ n, a (n + 1) = r * a n

theorem find_a7_a8
  (a : ℕ → ℝ)
  (r : ℝ)
  (hs : geometric_sequence_property a r)
  (h1 : a 1 + a 2 = 40)
  (h2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end NUMINAMATH_GPT_find_a7_a8_l962_96245


namespace NUMINAMATH_GPT_sports_probability_boy_given_sports_probability_l962_96266

variable (x : ℝ) -- Number of girls

def number_of_boys := 1.5 * x
def boys_liking_sports := 0.4 * number_of_boys x
def girls_liking_sports := 0.2 * x
def total_students := x + number_of_boys x
def total_students_liking_sports := boys_liking_sports x + girls_liking_sports x

theorem sports_probability : (total_students_liking_sports x) / (total_students x) = 8 / 25 := 
sorry

theorem boy_given_sports_probability :
  (boys_liking_sports x) / (total_students_liking_sports x) = 3 / 4 := 
sorry

end NUMINAMATH_GPT_sports_probability_boy_given_sports_probability_l962_96266


namespace NUMINAMATH_GPT_sale_price_l962_96297

def original_price : ℝ := 100
def discount_rate : ℝ := 0.80

theorem sale_price (original_price discount_rate : ℝ) : original_price * (1 - discount_rate) = 20 := by
  sorry

end NUMINAMATH_GPT_sale_price_l962_96297


namespace NUMINAMATH_GPT_remainder_of_product_mod_seven_l962_96250

-- Definitions derived from the conditions
def seq : List ℕ := [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

-- The main statement to prove
theorem remainder_of_product_mod_seven : 
  (seq.foldl (λ acc x => acc * x) 1) % 7 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_product_mod_seven_l962_96250


namespace NUMINAMATH_GPT_pencils_are_left_l962_96262

-- Define the conditions
def original_pencils : ℕ := 87
def removed_pencils : ℕ := 4

-- Define the expected outcome
def pencils_left : ℕ := original_pencils - removed_pencils

-- Prove that the number of pencils left in the jar is 83
theorem pencils_are_left : pencils_left = 83 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_pencils_are_left_l962_96262


namespace NUMINAMATH_GPT_linear_function_expression_l962_96226

theorem linear_function_expression (k b : ℝ) (h : ∀ x : ℝ, (1 ≤ x ∧ x ≤ 4 → 3 ≤ k * x + b ∧ k * x + b ≤ 6)) :
  (k = 1 ∧ b = 2) ∨ (k = -1 ∧ b = 7) :=
by
  sorry

end NUMINAMATH_GPT_linear_function_expression_l962_96226


namespace NUMINAMATH_GPT_find_weight_difference_l962_96298

variables (W_A W_B W_C W_D W_E : ℝ)

-- Definitions of the conditions
def average_weight_abc := (W_A + W_B + W_C) / 3 = 84
def average_weight_abcd := (W_A + W_B + W_C + W_D) / 4 = 80
def average_weight_bcde := (W_B + W_C + W_D + W_E) / 4 = 79
def weight_a := W_A = 77

-- The theorem statement
theorem find_weight_difference (h1 : average_weight_abc W_A W_B W_C)
                               (h2 : average_weight_abcd W_A W_B W_C W_D)
                               (h3 : average_weight_bcde W_B W_C W_D W_E)
                               (h4 : weight_a W_A) :
  W_E - W_D = 5 :=
sorry

end NUMINAMATH_GPT_find_weight_difference_l962_96298


namespace NUMINAMATH_GPT_negation_of_prop_l962_96260

open Classical

theorem negation_of_prop (h : ∀ x : ℝ, x^2 + x + 1 > 0) : ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_prop_l962_96260


namespace NUMINAMATH_GPT_complex_number_in_first_quadrant_l962_96265

-- Definition of the imaginary unit
def i : ℂ := Complex.I

-- Definition of the complex number z
def z : ℂ := i * (1 - i)

-- Coordinates of the complex number z
def z_coords : ℝ × ℝ := (z.re, z.im)

-- Statement asserting that the point corresponding to z lies in the first quadrant
theorem complex_number_in_first_quadrant : z_coords.fst > 0 ∧ z_coords.snd > 0 := 
by
  sorry

end NUMINAMATH_GPT_complex_number_in_first_quadrant_l962_96265


namespace NUMINAMATH_GPT_minimum_detectors_203_l962_96281

def minimum_detectors (length : ℕ) : ℕ :=
  length / 3 * 2 -- This models the generalization for 1 × (3k + 2)

theorem minimum_detectors_203 : minimum_detectors 203 = 134 :=
by
  -- Length is 203, k = 67 which follows from the floor division
  -- Therefore, minimum detectors = 2 * 67 = 134
  sorry

end NUMINAMATH_GPT_minimum_detectors_203_l962_96281


namespace NUMINAMATH_GPT_family_ate_doughnuts_l962_96274

variable (box_initial : ℕ) (box_left : ℕ) (dozen : ℕ)

-- Define the initial and remaining conditions
def dozen_value : ℕ := 12
def box_initial_value : ℕ := 2 * dozen_value
def doughnuts_left_value : ℕ := 16

theorem family_ate_doughnuts (h1 : box_initial = box_initial_value) (h2 : box_left = doughnuts_left_value) :
  box_initial - box_left = 8 := by
  -- h1 says the box initially contains 2 dozen, which is 24.
  -- h2 says that there are 16 doughnuts left.
  sorry

end NUMINAMATH_GPT_family_ate_doughnuts_l962_96274


namespace NUMINAMATH_GPT_height_of_taller_tree_l962_96218

-- Define the conditions as hypotheses:
variables (h₁ h₂ : ℝ)
-- The top of one tree is 24 feet higher than the top of another tree
variables (h_difference : h₁ = h₂ + 24)
-- The heights of the two trees are in the ratio 2:3
variables (h_ratio : h₂ / h₁ = 2 / 3)

theorem height_of_taller_tree : h₁ = 72 :=
by
  -- This is the place where the solution steps would be applied
  sorry

end NUMINAMATH_GPT_height_of_taller_tree_l962_96218


namespace NUMINAMATH_GPT_meaningful_sqrt_range_l962_96275

theorem meaningful_sqrt_range (x : ℝ) : (3 * x - 6 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end NUMINAMATH_GPT_meaningful_sqrt_range_l962_96275


namespace NUMINAMATH_GPT_andrew_game_night_expenses_l962_96255

theorem andrew_game_night_expenses : 
  let cost_per_game := 9 
  let number_of_games := 5 
  total_money_spent = cost_per_game * number_of_games 
→ total_money_spent = 45 := 
by
  intro cost_per_game number_of_games total_money_spent
  sorry

end NUMINAMATH_GPT_andrew_game_night_expenses_l962_96255


namespace NUMINAMATH_GPT_correct_option_for_sentence_completion_l962_96232

-- Define the mathematical formalization of the problem
def sentence_completion_problem : String × (List String) := 
    ("One of the most important questions they had to consider was _ of public health.", 
     ["what", "this", "that", "which"])

-- Define the correct answer
def correct_answer : String := "that"

-- The formal statement of the problem in Lean 4
theorem correct_option_for_sentence_completion 
    (problem : String × (List String)) (answer : String) :
    answer = "that" :=
by
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_correct_option_for_sentence_completion_l962_96232


namespace NUMINAMATH_GPT_possible_values_of_X_l962_96258

-- Define the conditions and the problem
def defective_products_total := 3
def total_products := 10
def selected_products := 2

-- Define the random variable X
def X (n : ℕ) : ℕ := n / selected_products

-- Now the statement to prove is that X can only take the values {0, 1, 2}
theorem possible_values_of_X :
  ∀ (X : ℕ → ℕ), ∃ (vals : Set ℕ), (vals = {0, 1, 2} ∧ ∀ (n : ℕ), X n ∈ vals) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_X_l962_96258


namespace NUMINAMATH_GPT_gcd_power_diff_l962_96237

theorem gcd_power_diff (n m : ℕ) (h₁ : n = 2025) (h₂ : m = 2007) :
  (Nat.gcd (2^n - 1) (2^m - 1)) = 2^18 - 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_power_diff_l962_96237


namespace NUMINAMATH_GPT_expression_bounds_l962_96213

theorem expression_bounds (p q r s : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) (hr : 0 ≤ r ∧ r ≤ 1) (hs : 0 ≤ s ∧ s ≤ 1) :
  2 * Real.sqrt 2 ≤ (Real.sqrt (p^2 + (1 - q)^2) + Real.sqrt (q^2 + (1 - r)^2) + Real.sqrt (r^2 + (1 - s)^2) + Real.sqrt (s^2 + (1 - p)^2)) ∧
  (Real.sqrt (p^2 + (1 - q)^2) + Real.sqrt (q^2 + (1 - r)^2) + Real.sqrt (r^2 + (1 - s)^2) + Real.sqrt (s^2 + (1 - p)^2)) ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_expression_bounds_l962_96213


namespace NUMINAMATH_GPT_derivative_of_f_l962_96272

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_of_f (x : ℝ) (h : 0 < x) :
    deriv f x = (1 - Real.log x) / (x ^ 2) := 
sorry

end NUMINAMATH_GPT_derivative_of_f_l962_96272


namespace NUMINAMATH_GPT_sum_of_squares_divisibility_l962_96277

theorem sum_of_squares_divisibility
  (p : ℕ) (hp : Nat.Prime p)
  (x y z : ℕ)
  (hx : 0 < x) (hxy : x < y) (hyz : y < z) (hzp : z < p)
  (hmod_eq : ∀ a b c : ℕ, a^3 % p = b^3 % p → b^3 % p = c^3 % p → a^3 % p = c^3 % p) :
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_divisibility_l962_96277


namespace NUMINAMATH_GPT_ratio_percent_l962_96248

theorem ratio_percent (x : ℕ) (h : (15 / x : ℚ) = 60 / 100) : x = 25 := 
sorry

end NUMINAMATH_GPT_ratio_percent_l962_96248


namespace NUMINAMATH_GPT_two_digit_number_as_expression_l962_96296

-- Define the conditions of the problem
variables (a : ℕ)

-- Statement to be proved
theorem two_digit_number_as_expression (h : 0 ≤ a ∧ a ≤ 9) : 10 * a + 1 = 10 * a + 1 := by
  sorry

end NUMINAMATH_GPT_two_digit_number_as_expression_l962_96296


namespace NUMINAMATH_GPT_find_roots_l962_96234

theorem find_roots : 
  (∃ x : ℝ, (x-1) * (x-2) * (x+1) * (x-5) = 0) ↔ 
  x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 5 :=
by sorry

end NUMINAMATH_GPT_find_roots_l962_96234


namespace NUMINAMATH_GPT_initial_people_count_l962_96217

-- Definitions from conditions
def initial_people (W : ℕ) : ℕ := W
def net_increase : ℕ := 5 - 2
def current_people : ℕ := 19

-- Theorem to prove: initial_people == 16 given conditions
theorem initial_people_count (W : ℕ) (h1 : W + net_increase = current_people) : initial_people W = 16 :=
by
  sorry

end NUMINAMATH_GPT_initial_people_count_l962_96217


namespace NUMINAMATH_GPT_dormouse_stole_flour_l962_96284

-- Define the suspects
inductive Suspect 
| MarchHare 
| MadHatter 
| Dormouse 

open Suspect 

-- Condition 1: Only one of three suspects stole the flour
def only_one_thief (s : Suspect) : Prop := 
  s = MarchHare ∨ s = MadHatter ∨ s = Dormouse

-- Condition 2: Only the person who stole the flour gave a truthful testimony
def truthful (thief : Suspect) (testimony : Suspect → Prop) : Prop :=
  testimony thief

-- Condition 3: The March Hare testified that the Mad Hatter stole the flour
def marchHare_testimony (s : Suspect) : Prop := 
  s = MadHatter

-- The theorem to prove: Dormouse stole the flour
theorem dormouse_stole_flour : 
  ∃ thief : Suspect, only_one_thief thief ∧ 
    (∀ s : Suspect, (s = thief ↔ truthful s marchHare_testimony) → thief = Dormouse) :=
by
  sorry

end NUMINAMATH_GPT_dormouse_stole_flour_l962_96284


namespace NUMINAMATH_GPT_total_area_of_plots_l962_96299

theorem total_area_of_plots (n : ℕ) (side_length : ℕ) (area_one_plot : ℕ) (total_plots : ℕ) (total_area : ℕ)
  (h1 : n = 9)
  (h2 : side_length = 6)
  (h3 : area_one_plot = side_length * side_length)
  (h4 : total_plots = n)
  (h5 : total_area = area_one_plot * total_plots) :
  total_area = 324 := 
by
  sorry

end NUMINAMATH_GPT_total_area_of_plots_l962_96299


namespace NUMINAMATH_GPT_find_x_values_l962_96242

-- Defining the given condition as a function
def equation (x : ℝ) : Prop :=
  (4 / (Real.sqrt (x + 5) - 7)) +
  (3 / (Real.sqrt (x + 5) - 2)) +
  (6 / (Real.sqrt (x + 5) + 2)) +
  (9 / (Real.sqrt (x + 5) + 7)) = 0

-- Statement of the theorem in Lean
theorem find_x_values :
  equation ( -796 / 169) ∨ equation (383 / 22) :=
sorry

end NUMINAMATH_GPT_find_x_values_l962_96242


namespace NUMINAMATH_GPT_cora_reading_ratio_l962_96210

variable (P : Nat) 
variable (M T W Th F : Nat)

-- Conditions
def conditions (P M T W Th F : Nat) : Prop := 
  P = 158 ∧ 
  M = 23 ∧ 
  T = 38 ∧ 
  W = 61 ∧ 
  Th = 12 ∧ 
  F = Th

-- The theorem statement
theorem cora_reading_ratio (h : conditions P M T W Th F) : F / Th = 1 / 1 :=
by
  -- We use the conditions to apply the proof
  obtain ⟨hp, hm, ht, hw, hth, hf⟩ := h
  rw [hf]
  norm_num
  sorry

end NUMINAMATH_GPT_cora_reading_ratio_l962_96210


namespace NUMINAMATH_GPT_find_profit_range_l962_96219

noncomputable def profit_range (x : ℝ) : Prop :=
  0 < x → 0.15 * (1 + 0.25 * x) * (100000 - x) ≥ 0.15 * 100000

theorem find_profit_range (x : ℝ) : profit_range x → 0 < x ∧ x ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_find_profit_range_l962_96219


namespace NUMINAMATH_GPT_range_of_a_l962_96222

variables (a x : ℝ) -- Define real number variables a and x

-- Define proposition p
def p : Prop := (a - 2) * x * x + 2 * (a - 2) * x - 4 < 0 -- Inequality condition for any real x

-- Define proposition q
def q : Prop := 0 < a ∧ a < 1 -- Condition for logarithmic function to be strictly decreasing

-- Lean 4 statement for the proof problem
theorem range_of_a (Hpq : (p a x ∨ q a) ∧ ¬ (p a x ∧ q a)) :
  (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0) :=
sorry

end NUMINAMATH_GPT_range_of_a_l962_96222


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l962_96214

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l962_96214


namespace NUMINAMATH_GPT_james_found_bills_l962_96233

def initial_money : ℝ := 75
def final_money : ℝ := 135
def bill_value : ℝ := 20

theorem james_found_bills :
  (final_money - initial_money) / bill_value = 3 :=
by
  sorry

end NUMINAMATH_GPT_james_found_bills_l962_96233


namespace NUMINAMATH_GPT_six_coins_not_sum_to_14_l962_96259

def coin_values : Set ℕ := {1, 5, 10, 25}

theorem six_coins_not_sum_to_14 (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 ∈ coin_values) (h2 : a2 ∈ coin_values) (h3 : a3 ∈ coin_values) (h4 : a4 ∈ coin_values) (h5 : a5 ∈ coin_values) (h6 : a6 ∈ coin_values) : a1 + a2 + a3 + a4 + a5 + a6 ≠ 14 := 
sorry

end NUMINAMATH_GPT_six_coins_not_sum_to_14_l962_96259


namespace NUMINAMATH_GPT_cupcakes_leftover_l962_96254

theorem cupcakes_leftover {total_cupcakes nutty_cupcakes gluten_free_cupcakes children children_no_nuts child_only_gf leftover_nutty leftover_regular : Nat} :
  total_cupcakes = 84 →
  children = 7 →
  nutty_cupcakes = 18 →
  gluten_free_cupcakes = 25 →
  children_no_nuts = 2 →
  child_only_gf = 1 →
  leftover_nutty = 3 →
  leftover_regular = 2 →
  leftover_nutty + leftover_regular = 5 :=
by
  sorry

end NUMINAMATH_GPT_cupcakes_leftover_l962_96254


namespace NUMINAMATH_GPT_poles_inside_base_l962_96270

theorem poles_inside_base :
  ∃ n : ℕ, 2015 + n ≡ 0 [MOD 36] ∧ n = 1 :=
sorry

end NUMINAMATH_GPT_poles_inside_base_l962_96270


namespace NUMINAMATH_GPT_integral_of_2x2_cos3x_l962_96238

theorem integral_of_2x2_cos3x :
  ∫ x in (0 : ℝ)..(2 * Real.pi), (2 * x ^ 2 - 15) * Real.cos (3 * x) = (8 * Real.pi) / 9 :=
by
  sorry

end NUMINAMATH_GPT_integral_of_2x2_cos3x_l962_96238


namespace NUMINAMATH_GPT_faye_books_l962_96216

theorem faye_books (initial_books given_away final_books books_bought: ℕ) 
  (h1 : initial_books = 34) 
  (h2 : given_away = 3) 
  (h3 : final_books = 79) 
  (h4 : final_books = initial_books - given_away + books_bought) : 
  books_bought = 48 := 
by 
  sorry

end NUMINAMATH_GPT_faye_books_l962_96216


namespace NUMINAMATH_GPT_correct_total_l962_96202

-- Define the conditions in Lean
variables (y : ℕ) -- y is a natural number (non-negative integer)

-- Define the values of the different coins in cents
def value_of_quarter := 25
def value_of_dollar := 100
def value_of_nickel := 5
def value_of_dime := 10

-- Define the errors in terms of y
def error_due_to_quarters := y * (value_of_dollar - value_of_quarter) -- 75y
def error_due_to_nickels := y * (value_of_dime - value_of_nickel) -- 5y

-- Net error calculation
def net_error := error_due_to_quarters - error_due_to_nickels -- 70y

-- Math proof problem statement
theorem correct_total (h : error_due_to_quarters = 75 * y ∧ error_due_to_nickels = 5 * y) :
  net_error = 70 * y :=
by sorry

end NUMINAMATH_GPT_correct_total_l962_96202


namespace NUMINAMATH_GPT_union_A_B_l962_96264

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | x > 2}

theorem union_A_B :
  A ∪ B = {x : ℝ | 1 ≤ x} := sorry

end NUMINAMATH_GPT_union_A_B_l962_96264


namespace NUMINAMATH_GPT_carol_has_35_nickels_l962_96261

def problem_statement : Prop :=
  ∃ (n d : ℕ), 5 * n + 10 * d = 455 ∧ n = d + 7 ∧ n = 35

theorem carol_has_35_nickels : problem_statement := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_carol_has_35_nickels_l962_96261


namespace NUMINAMATH_GPT_f_geq_expression_l962_96287

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 * a - 1 / a) * x - Real.log x

theorem f_geq_expression (a x : ℝ) (h : a < 0) : f x a ≥ (1 - 2 * a) * (a + 1) := 
  sorry

end NUMINAMATH_GPT_f_geq_expression_l962_96287


namespace NUMINAMATH_GPT_circle_regions_l962_96295

def regions_divided_by_chords (n : ℕ) : ℕ :=
  (n^4 - 6 * n^3 + 23 * n^2 - 18 * n + 24) / 24

theorem circle_regions (n : ℕ) : 
  regions_divided_by_chords n = (n^4 - 6 * n^3 + 23 * n^2 - 18 * n + 24) / 24 := 
  by 
  sorry

end NUMINAMATH_GPT_circle_regions_l962_96295


namespace NUMINAMATH_GPT_probability_seven_chairs_probability_n_chairs_l962_96289
-- Importing necessary library to ensure our Lean code can be built successfully

-- Definition for case where n = 7
theorem probability_seven_chairs : 
  let total_seating := 7 * 6 * 5 / 6 
  let favorable_seating := 1 
  let probability := favorable_seating / total_seating 
  probability = 1 / 35 := 
by 
  sorry

-- Definition for general case where n ≥ 6
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) : 
  let total_seating := (n - 1) * (n - 2) / 2 
  let favorable_seating := (n - 4) * (n - 5) / 2 
  let probability := favorable_seating / total_seating 
  probability = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := 
by 
  sorry

end NUMINAMATH_GPT_probability_seven_chairs_probability_n_chairs_l962_96289


namespace NUMINAMATH_GPT_solve_squares_and_circles_l962_96212

theorem solve_squares_and_circles (x y : ℝ) :
  (5 * x + 2 * y = 39) ∧ (3 * x + 3 * y = 27) → (x = 7) ∧ (y = 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_squares_and_circles_l962_96212


namespace NUMINAMATH_GPT_equidistant_points_eq_two_l962_96253

noncomputable def number_of_equidistant_points (O : Point) (r d : ℝ) 
  (h1 : d > r) : ℕ := 
2

theorem equidistant_points_eq_two (O : Point) (r d : ℝ) 
  (h1 : d > r) : number_of_equidistant_points O r d h1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_equidistant_points_eq_two_l962_96253


namespace NUMINAMATH_GPT_JeremyTotalExpenses_l962_96267

noncomputable def JeremyExpenses : ℝ :=
  let motherGift := 400
  let fatherGift := 280
  let sisterGift := 100
  let brotherGift := 60
  let friendGift := 50
  let giftWrappingRate := 0.07
  let taxRate := 0.09
  let miscExpenses := 40
  let wrappingCost := motherGift * giftWrappingRate
                  + fatherGift * giftWrappingRate
                  + sisterGift * giftWrappingRate
                  + brotherGift * giftWrappingRate
                  + friendGift * giftWrappingRate
  let totalGiftCost := motherGift + fatherGift + sisterGift + brotherGift + friendGift
  let totalTax := totalGiftCost * taxRate
  wrappingCost + totalTax + miscExpenses

theorem JeremyTotalExpenses : JeremyExpenses = 182.40 := by
  sorry

end NUMINAMATH_GPT_JeremyTotalExpenses_l962_96267


namespace NUMINAMATH_GPT_sin_alpha_pi_over_3_plus_sin_alpha_l962_96203

-- Defining the problem with the given conditions
variable (α : ℝ)
variable (hcos : Real.cos (α + (2 / 3) * Real.pi) = 4 / 5)
variable (hα : -Real.pi / 2 < α ∧ α < 0)

-- Statement to prove
theorem sin_alpha_pi_over_3_plus_sin_alpha :
  Real.sin (α + Real.pi / 3) + Real.sin α = -4 * Real.sqrt 3 / 5 :=
sorry

end NUMINAMATH_GPT_sin_alpha_pi_over_3_plus_sin_alpha_l962_96203


namespace NUMINAMATH_GPT_fraction_division_l962_96273

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end NUMINAMATH_GPT_fraction_division_l962_96273


namespace NUMINAMATH_GPT_balls_removal_l962_96286

theorem balls_removal (total_balls : ℕ) (percent_green initial_green initial_yellow remaining_percent : ℝ)
    (h_percent_green : percent_green = 0.7)
    (h_total_balls : total_balls = 600)
    (h_initial_green : initial_green = percent_green * total_balls)
    (h_initial_yellow : initial_yellow = total_balls - initial_green)
    (h_remaining_percent : remaining_percent = 0.6) :
    ∃ x : ℝ, (initial_green - x) / (total_balls - x) = remaining_percent ∧ x = 150 := 
by 
  sorry

end NUMINAMATH_GPT_balls_removal_l962_96286


namespace NUMINAMATH_GPT_trig_identity_l962_96290

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (π + α)) / (Real.sin (π / 2 - α)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l962_96290


namespace NUMINAMATH_GPT_percentage_deposit_paid_l962_96231

theorem percentage_deposit_paid (D R T : ℝ) (hd : D = 105) (hr : R = 945) (ht : T = D + R) : (D / T) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_deposit_paid_l962_96231


namespace NUMINAMATH_GPT_given_condition_l962_96228

variable (a : ℝ)

theorem given_condition
  (h1 : (a + 1/a)^2 = 5) :
  a^2 + 1/a^2 + a^3 + 1/a^3 = 3 + 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_given_condition_l962_96228
