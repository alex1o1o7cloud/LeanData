import Mathlib

namespace NUMINAMATH_GPT_luke_total_points_l573_57312

-- Definitions based on conditions
def points_per_round : ℕ := 3
def rounds_played : ℕ := 26

-- Theorem stating the question and correct answer
theorem luke_total_points : points_per_round * rounds_played = 78 := 
by 
  sorry

end NUMINAMATH_GPT_luke_total_points_l573_57312


namespace NUMINAMATH_GPT_find_c_for_min_value_zero_l573_57359

theorem find_c_for_min_value_zero :
  ∃ c : ℝ, c = 1 ∧ (∀ x y : ℝ, 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 6 * x - 6 * y + 9 ≥ 0) ∧
  (∀ x y : ℝ, 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 6 * x - 6 * y + 9 = 0 → c = 1) :=
by
  use 1
  sorry

end NUMINAMATH_GPT_find_c_for_min_value_zero_l573_57359


namespace NUMINAMATH_GPT_two_pow_n_minus_one_div_by_seven_iff_two_pow_n_plus_one_not_div_by_seven_l573_57358

theorem two_pow_n_minus_one_div_by_seven_iff (n : ℕ) : (7 ∣ 2^n - 1) ↔ ∃ k : ℕ, n = 3 * k :=
by sorry

theorem two_pow_n_plus_one_not_div_by_seven (n : ℕ) : n > 0 → ¬(7 ∣ 2^n + 1) :=
by sorry

end NUMINAMATH_GPT_two_pow_n_minus_one_div_by_seven_iff_two_pow_n_plus_one_not_div_by_seven_l573_57358


namespace NUMINAMATH_GPT_factorization_problem_l573_57366

theorem factorization_problem (p q : ℝ) :
  (∃ a b c : ℝ, 
    x^4 + p * x^2 + q = (x^2 + 2 * x + 5) * (a * x^2 + b * x + c)) ↔
  p = 6 ∧ q = 25 := 
sorry

end NUMINAMATH_GPT_factorization_problem_l573_57366


namespace NUMINAMATH_GPT_intersection_A_B_l573_57322

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem intersection_A_B : (A ∩ B) = { x | 0 < x ∧ x ≤ 1 } := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l573_57322


namespace NUMINAMATH_GPT_value_of_3k_squared_minus_1_l573_57381

theorem value_of_3k_squared_minus_1 (x k : ℤ)
  (h1 : 7 * x + 2 = 3 * x - 6)
  (h2 : x + 1 = k)
  : 3 * k^2 - 1 = 2 := 
by
  sorry

end NUMINAMATH_GPT_value_of_3k_squared_minus_1_l573_57381


namespace NUMINAMATH_GPT_circle_range_of_m_l573_57347

theorem circle_range_of_m (m : ℝ) :
  (∃ h k r : ℝ, (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2 ↔ x ^ 2 + y ^ 2 - x + y + m = 0)) ↔ (m < 1/2) :=
by
  sorry

end NUMINAMATH_GPT_circle_range_of_m_l573_57347


namespace NUMINAMATH_GPT_ratio_xy_l573_57334

theorem ratio_xy (x y : ℝ) (h : 2*y - 5*x = 0) : x / y = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_ratio_xy_l573_57334


namespace NUMINAMATH_GPT_restaurant_sodas_l573_57309

theorem restaurant_sodas (M : ℕ) (h1 : M + 19 = 96) : M = 77 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_sodas_l573_57309


namespace NUMINAMATH_GPT_alcohol_water_ratio_l573_57316

theorem alcohol_water_ratio (a b : ℚ) (h₀ : a > 0) (h₁ : b > 0) :
  (3 * a / (a + 2) + 8 / (4 + b)) / (6 / (a + 2) + 2 * b / (4 + b)) = (3 * a + 8) / (6 + 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_alcohol_water_ratio_l573_57316


namespace NUMINAMATH_GPT_position_of_99_l573_57351

-- Define a function that describes the position of an odd number in the 5-column table.
def position_in_columns (n : ℕ) : ℕ := sorry  -- position in columns is defined by some rule

-- Now, state the theorem regarding the position of 99.
theorem position_of_99 : position_in_columns 99 = 3 := 
by 
  sorry  -- Proof goes here

end NUMINAMATH_GPT_position_of_99_l573_57351


namespace NUMINAMATH_GPT_gen_formula_is_arith_seq_l573_57313

-- Given: The sum of the first n terms of the sequence {a_n} is S_n = n^2 + 2n
def sum_seq (S : ℕ → ℕ) := ∀ n : ℕ, S n = n^2 + 2 * n

-- The general formula for {a_n} is a_n = 2n + 1
theorem gen_formula (S : ℕ → ℕ) (h : sum_seq S) : ∀ n : ℕ,  n > 0 → (∃ a : ℕ → ℕ, a n = 2 * n + 1 ∧ ∀ m : ℕ, m < n → a m = S (m + 1) - S m) :=
by sorry

-- The sequence {a_n} defined by a_n = 2n + 1 is an arithmetic sequence
theorem is_arith_seq : ∀ n : ℕ, n > 0 → (∀ a : ℕ → ℕ, (∀ k, k > 0 → a k = 2 * k + 1) → ∃ d : ℕ, d = 2 ∧ ∀ j > 0, a j - a (j - 1) = d) :=
by sorry

end NUMINAMATH_GPT_gen_formula_is_arith_seq_l573_57313


namespace NUMINAMATH_GPT_hair_ratio_l573_57325

theorem hair_ratio (washed : ℕ) (grow_back : ℕ) (brushed : ℕ) (n : ℕ)
  (hwashed : washed = 32)
  (hgrow_back : grow_back = 49)
  (heq : washed + brushed + 1 = grow_back) :
  (brushed : ℚ) / washed = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_hair_ratio_l573_57325


namespace NUMINAMATH_GPT_floor_of_pi_l573_57341

noncomputable def floor_of_pi_eq_three : Prop :=
  ⌊Real.pi⌋ = 3

theorem floor_of_pi : floor_of_pi_eq_three :=
  sorry

end NUMINAMATH_GPT_floor_of_pi_l573_57341


namespace NUMINAMATH_GPT_nick_charges_l573_57305

theorem nick_charges (y : ℕ) :
  let travel_cost := 7
  let hourly_rate := 10
  10 * y + 7 = travel_cost + hourly_rate * y :=
by sorry

end NUMINAMATH_GPT_nick_charges_l573_57305


namespace NUMINAMATH_GPT_chocolate_pieces_l573_57398

theorem chocolate_pieces (total_pieces : ℕ) (michael_portion : ℕ) (paige_portion : ℕ) (mandy_portion : ℕ) 
  (h_total : total_pieces = 60) 
  (h_michael : michael_portion = total_pieces / 2) 
  (h_paige : paige_portion = (total_pieces - michael_portion) / 2) 
  (h_mandy : mandy_portion = total_pieces - (michael_portion + paige_portion)) : 
  mandy_portion = 15 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_pieces_l573_57398


namespace NUMINAMATH_GPT_min_value_of_a_l573_57323

noncomputable def f (x a : ℝ) : ℝ :=
  Real.exp x * (x + (3 / x) - 3) - (a / x)

noncomputable def g (x : ℝ) : ℝ :=
  (x^2 - 3 * x + 3) * Real.exp x

theorem min_value_of_a (a : ℝ) :
  (∃ x > 0, f x a ≤ 0) → a ≥ Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_l573_57323


namespace NUMINAMATH_GPT_simplify_expression_l573_57353

theorem simplify_expression :
  let a := 7
  let b := 2
  (a^5 + b^8) * (b^3 - (-b)^3)^7 = 0 := by
  let a := 7
  let b := 2
  sorry

end NUMINAMATH_GPT_simplify_expression_l573_57353


namespace NUMINAMATH_GPT_solve_for_y_l573_57370

theorem solve_for_y {y : ℝ} : 
  (2012 + y)^2 = 2 * y^2 ↔ y = 2012 * (Real.sqrt 2 + 1) ∨ y = -2012 * (Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_GPT_solve_for_y_l573_57370


namespace NUMINAMATH_GPT_nancy_total_money_l573_57399

theorem nancy_total_money (n : ℕ) (d : ℕ) (h1 : n = 9) (h2 : d = 5) : n * d = 45 := 
by
  sorry

end NUMINAMATH_GPT_nancy_total_money_l573_57399


namespace NUMINAMATH_GPT_total_pics_uploaded_l573_57394

-- Definitions of conditions
def pic_in_first_album : Nat := 14
def albums_with_7_pics : Nat := 3
def pics_per_album : Nat := 7

-- Theorem statement
theorem total_pics_uploaded :
  pic_in_first_album + albums_with_7_pics * pics_per_album = 35 := by
  sorry

end NUMINAMATH_GPT_total_pics_uploaded_l573_57394


namespace NUMINAMATH_GPT_part1_part2_l573_57320

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin ((1 / 3) * x - (Real.pi / 6))

theorem part1 : f (5 * Real.pi / 4) = Real.sqrt 2 :=
by sorry

theorem part2 (α β : ℝ) (hαβ : 0 ≤ α ∧ α ≤ Real.pi / 2 ∧ 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h1: f (3 * α + Real.pi / 2) = 10 / 13) (h2: f (3 * β + 2 * Real.pi) = 6 / 5) :
  Real.cos (α + β) = 16 / 65 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l573_57320


namespace NUMINAMATH_GPT_bruces_son_age_l573_57337

variable (Bruce_age : ℕ) (son_age : ℕ)
variable (h1 : Bruce_age = 36)
variable (h2 : Bruce_age + 6 = 3 * (son_age + 6))

theorem bruces_son_age :
  son_age = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_bruces_son_age_l573_57337


namespace NUMINAMATH_GPT_find_b_l573_57388

open Real

theorem find_b (b : ℝ) : 
  (∀ x y : ℝ, 4 * y - 3 * x - 2 = 0 -> 6 * y + b * x + 1 = 0 -> 
   exists m₁ m₂ : ℝ, 
   ((y = m₁ * x + _1 / 2) -> m₁ = 3 / 4) ∧ ((y = m₂ * x - 1 / 6) -> m₂ = -b / 6)) -> 
  b = -4.5 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l573_57388


namespace NUMINAMATH_GPT_percentage_reduction_in_price_l573_57392

-- Definitions based on conditions
def original_price (P : ℝ) (X : ℝ) := P * X
def reduced_price (R : ℝ) (X : ℝ) := R * (X + 5)

-- Theorem statement based on the problem to prove
theorem percentage_reduction_in_price
  (R : ℝ) (H1 : R = 55)
  (H2 : original_price P X = 1100)
  (H3 : reduced_price R X = 1100) :
  ((P - R) / P) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_reduction_in_price_l573_57392


namespace NUMINAMATH_GPT_length_of_AB_area_of_ΔABF1_l573_57338

theorem length_of_AB (A B : ℝ × ℝ) (x1 x2 y1 y2 : ℝ) :
  (y1 = x1 - 2) ∧ (y2 = x2 - 2) ∧ ((x1 = 0) ∧ (y1 = -2)) ∧ ((x2 = 8/3) ∧ (y2 = 2/3)) →
  |((x1 - x2)^2 + (y1 - y2)^2)^(1/2)| = (8 / 3) * (2)^(1/2) :=
by sorry

theorem area_of_ΔABF1 (A B F1 : ℝ × ℝ) (x1 x2 y1 y2 : ℝ) :
  (F1 = (0, -2)) ∧ ((y1 = x1 - 2) ∧ (y2 = x2 - 2) ∧ ((x1 = 0) ∧ (y1 = -2)) ∧ ((x2 = 8/3) ∧ (y2 = 2/3))) →
  (1/2) * (((x1 - x2)^2 + (y1 - y2)^2)^(1/2)) * (|(-2-2)/((2)^(1/2))|) = 16 / 3 :=
by sorry

end NUMINAMATH_GPT_length_of_AB_area_of_ΔABF1_l573_57338


namespace NUMINAMATH_GPT_john_reaching_floor_pushups_l573_57345

-- Definitions based on conditions
def john_train_days_per_week : ℕ := 5
def reps_to_progress : ℕ := 20
def variations : ℕ := 3  -- wall, incline, knee

-- Mathematical statement
theorem john_reaching_floor_pushups : 
  (reps_to_progress * variations) / john_train_days_per_week = 12 := 
by
  sorry

end NUMINAMATH_GPT_john_reaching_floor_pushups_l573_57345


namespace NUMINAMATH_GPT_tangent_line_sum_l573_57395

theorem tangent_line_sum (a b : ℝ) :
  (∃ x₀ : ℝ, (e^(x₀ - 1) = 1) ∧ (x₀ + a = e^(x₀-1) * (1 - x₀) - b + 1)) → a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_sum_l573_57395


namespace NUMINAMATH_GPT_parallel_lines_b_value_l573_57378

-- Define the first line equation in slope-intercept form.
def line1_slope (b : ℝ) : ℝ :=
  3

-- Define the second line equation in slope-intercept form.
def line2_slope (b : ℝ) : ℝ :=
  b + 10

-- Theorem stating that if the lines are parallel, the value of b is -7.
theorem parallel_lines_b_value :
  ∀ b : ℝ, line1_slope b = line2_slope b → b = -7 :=
by
  intro b
  intro h
  sorry

end NUMINAMATH_GPT_parallel_lines_b_value_l573_57378


namespace NUMINAMATH_GPT_solve_xyz_system_l573_57377

theorem solve_xyz_system :
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 
    (x * (6 - y) = 9) ∧ 
    (y * (6 - z) = 9) ∧ 
    (z * (6 - x) = 9) ∧ 
    x = 3 ∧ y = 3 ∧ z = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_xyz_system_l573_57377


namespace NUMINAMATH_GPT_find_angle_NCB_l573_57328

def triangle_ABC_with_point_N (A B C N : Point) : Prop :=
  ∃ (angle_ABC angle_ACB angle_NAB angle_NBC : ℝ),
    angle_ABC = 50 ∧
    angle_ACB = 20 ∧
    angle_NAB = 40 ∧
    angle_NBC = 30 

theorem find_angle_NCB (A B C N : Point) 
  (h : triangle_ABC_with_point_N A B C N) :
  ∃ (angle_NCB : ℝ), 
  angle_NCB = 10 :=
sorry

end NUMINAMATH_GPT_find_angle_NCB_l573_57328


namespace NUMINAMATH_GPT_telescope_visual_range_increase_l573_57352

theorem telescope_visual_range_increase (original_range : ℝ) (increase_percent : ℝ) 
(h1 : original_range = 100) (h2 : increase_percent = 0.50) : 
original_range + (increase_percent * original_range) = 150 := 
sorry

end NUMINAMATH_GPT_telescope_visual_range_increase_l573_57352


namespace NUMINAMATH_GPT_find_g3_l573_57361

variable {α : Type*} [Field α]

-- Define the function g
noncomputable def g (x : α) : α := sorry

-- Define the condition as a hypothesis
axiom condition (x : α) (hx : x ≠ 0) : 2 * g (1 / x) + 3 * g x / x = 2 * x ^ 2

-- State what needs to be proven
theorem find_g3 : g 3 = 242 / 15 := by
  sorry

end NUMINAMATH_GPT_find_g3_l573_57361


namespace NUMINAMATH_GPT_divides_b_n_minus_n_l573_57380

theorem divides_b_n_minus_n (a b : ℕ) (h_a : a > 0) (h_b : b > 0) :
  ∃ n : ℕ, n > 0 ∧ a ∣ (b^n - n) :=
by
  sorry

end NUMINAMATH_GPT_divides_b_n_minus_n_l573_57380


namespace NUMINAMATH_GPT_factorize_expression_l573_57339

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l573_57339


namespace NUMINAMATH_GPT_tan_neg_405_eq_one_l573_57330

theorem tan_neg_405_eq_one : Real.tan (-(405 * Real.pi / 180)) = 1 :=
by
-- Proof omitted
sorry

end NUMINAMATH_GPT_tan_neg_405_eq_one_l573_57330


namespace NUMINAMATH_GPT_joan_games_l573_57386

theorem joan_games (last_year_games this_year_games total_games : ℕ)
  (h1 : last_year_games = 9)
  (h2 : total_games = 13)
  : this_year_games = total_games - last_year_games → this_year_games = 4 := 
by
  intros h
  rw [h1, h2] at h
  exact h

end NUMINAMATH_GPT_joan_games_l573_57386


namespace NUMINAMATH_GPT_remainder_of_n_l573_57365

theorem remainder_of_n (n : ℕ) (h1 : n^2 ≡ 9 [MOD 11]) (h2 : n^3 ≡ 5 [MOD 11]) : n ≡ 3 [MOD 11] :=
sorry

end NUMINAMATH_GPT_remainder_of_n_l573_57365


namespace NUMINAMATH_GPT_molecular_weight_correct_l573_57303

-- Atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 15.999
def atomic_weight_H : ℝ := 1.008

-- Number of each type of atom in the compound
def num_Al : ℕ := 1
def num_O : ℕ := 3
def num_H : ℕ := 3

-- Molecular weight calculation
def molecular_weight : ℝ :=
  (num_Al * atomic_weight_Al) + (num_O * atomic_weight_O) + (num_H * atomic_weight_H)

theorem molecular_weight_correct : molecular_weight = 78.001 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_correct_l573_57303


namespace NUMINAMATH_GPT_area_enclosed_by_circle_l573_57318

theorem area_enclosed_by_circle :
  let center := (3, -10)
  let radius := 3
  let equation := ∀ (x y : ℝ), (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2
  ∃ enclosed_area : ℝ, enclosed_area = 9 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_circle_l573_57318


namespace NUMINAMATH_GPT_each_child_ate_3_jellybeans_l573_57304

-- Define the given conditions
def total_jellybeans : ℕ := 100
def total_kids : ℕ := 24
def sick_kids : ℕ := 2
def leftover_jellybeans : ℕ := 34

-- Calculate the number of kids who attended
def attending_kids : ℕ := total_kids - sick_kids

-- Calculate the total jellybeans eaten
def total_jellybeans_eaten : ℕ := total_jellybeans - leftover_jellybeans

-- Calculate the number of jellybeans each child ate
def jellybeans_per_child : ℕ := total_jellybeans_eaten / attending_kids

theorem each_child_ate_3_jellybeans : jellybeans_per_child = 3 :=
by sorry

end NUMINAMATH_GPT_each_child_ate_3_jellybeans_l573_57304


namespace NUMINAMATH_GPT_tory_earns_more_than_bert_l573_57391

-- Define the initial prices of the toys
def initial_price_phones : ℝ := 18
def initial_price_guns : ℝ := 20

-- Define the quantities sold by Bert and Tory
def quantity_phones : ℕ := 10
def quantity_guns : ℕ := 15

-- Define the discounts
def discount_phones : ℝ := 0.15
def discounted_phones_quantity : ℕ := 3

def discount_guns : ℝ := 0.10
def discounted_guns_quantity : ℕ := 7

-- Define the tax
def tax_rate : ℝ := 0.05

noncomputable def bert_initial_earnings : ℝ := initial_price_phones * quantity_phones

noncomputable def tory_initial_earnings : ℝ := initial_price_guns * quantity_guns

noncomputable def bert_discount : ℝ := discount_phones * initial_price_phones * discounted_phones_quantity

noncomputable def tory_discount : ℝ := discount_guns * initial_price_guns * discounted_guns_quantity

noncomputable def bert_earnings_after_discount : ℝ := bert_initial_earnings - bert_discount

noncomputable def tory_earnings_after_discount : ℝ := tory_initial_earnings - tory_discount

noncomputable def bert_tax : ℝ := tax_rate * bert_earnings_after_discount

noncomputable def tory_tax : ℝ := tax_rate * tory_earnings_after_discount

noncomputable def bert_final_earnings : ℝ := bert_earnings_after_discount + bert_tax

noncomputable def tory_final_earnings : ℝ := tory_earnings_after_discount + tory_tax

noncomputable def earning_difference : ℝ := tory_final_earnings - bert_final_earnings

theorem tory_earns_more_than_bert : earning_difference = 119.805 := by
  sorry

end NUMINAMATH_GPT_tory_earns_more_than_bert_l573_57391


namespace NUMINAMATH_GPT_sum_of_faces_l573_57382

theorem sum_of_faces (n_side_faces_per_prism : ℕ) (n_non_side_faces_per_prism : ℕ)
  (num_prisms : ℕ) (h1 : n_side_faces_per_prism = 3) (h2 : n_non_side_faces_per_prism = 2) 
  (h3 : num_prisms = 3) : 
  n_side_faces_per_prism * num_prisms + n_non_side_faces_per_prism * num_prisms = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_faces_l573_57382


namespace NUMINAMATH_GPT_fraction_subtraction_result_l573_57321

theorem fraction_subtraction_result :
  (3 * 5 + 5 * 7 + 7 * 9) / (2 * 4 + 4 * 6 + 6 * 8) - (2 * 4 + 4 * 6 + 6 * 8) / (3 * 5 + 5 * 7 + 7 * 9) = 74 / 119 :=
by sorry

end NUMINAMATH_GPT_fraction_subtraction_result_l573_57321


namespace NUMINAMATH_GPT_min_text_length_l573_57355

theorem min_text_length : ∃ (L : ℕ), (∀ x : ℕ, 0.105 * (L : ℝ) < (x : ℝ) ∧ (x : ℝ) < 0.11 * (L : ℝ)) → L = 19 :=
by
  sorry

end NUMINAMATH_GPT_min_text_length_l573_57355


namespace NUMINAMATH_GPT_union_complement_eq_l573_57324

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

theorem union_complement_eq :
  (complement U A ∪ B) = {2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_union_complement_eq_l573_57324


namespace NUMINAMATH_GPT_equal_number_of_boys_and_girls_l573_57379

theorem equal_number_of_boys_and_girls
    (num_classrooms : ℕ) (girls : ℕ) (total_per_classroom : ℕ)
    (equal_boys_and_girls : ∀ (c : ℕ), c ≤ num_classrooms → (girls + boys) = total_per_classroom):
    num_classrooms = 4 → girls = 44 → total_per_classroom = 25 → boys = 44 :=
by
  sorry

end NUMINAMATH_GPT_equal_number_of_boys_and_girls_l573_57379


namespace NUMINAMATH_GPT_differential_system_solution_l573_57326

noncomputable def x (t : ℝ) := 1 - t - Real.exp (-6 * t) * Real.cos t
noncomputable def y (t : ℝ) := 1 - 7 * t + Real.exp (-6 * t) * Real.cos t + Real.exp (-6 * t) * Real.sin t

theorem differential_system_solution :
  (∀ t : ℝ, (deriv x t) = -7 * x t + y t + 5) ∧
  (∀ t : ℝ, (deriv y t) = -2 * x t - 5 * y t - 37 * t) ∧
  (x 0 = 0) ∧
  (y 0 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_differential_system_solution_l573_57326


namespace NUMINAMATH_GPT_base_r_representation_26_eq_32_l573_57367

theorem base_r_representation_26_eq_32 (r : ℕ) : 
  26 = 3 * r + 6 → r = 8 :=
by
  sorry

end NUMINAMATH_GPT_base_r_representation_26_eq_32_l573_57367


namespace NUMINAMATH_GPT_prob_xi_ge_2_eq_one_third_l573_57344

noncomputable def pmf (c k : ℝ) : ℝ := c / (k * (k + 1))

theorem prob_xi_ge_2_eq_one_third 
  (c : ℝ) 
  (h₁ : pmf c 1 + pmf c 2 + pmf c 3 = 1) :
  pmf c 2 + pmf c 3 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_prob_xi_ge_2_eq_one_third_l573_57344


namespace NUMINAMATH_GPT_initial_mean_of_observations_l573_57396

-- Definitions of the given conditions and proof of the correct initial mean
theorem initial_mean_of_observations 
  (M : ℝ) -- Mean of 50 observations
  (initial_sum := 50 * M) -- Initial sum of observations
  (wrong_observation : ℝ := 23) -- Wrong observation
  (correct_observation : ℝ := 45) -- Correct observation
  (understated_by := correct_observation - wrong_observation) -- Amount of understatement
  (correct_sum := initial_sum + understated_by) -- Corrected sum
  (corrected_mean : ℝ := 36.5) -- Corrected new mean
  (eq1 : correct_sum = 50 * corrected_mean) -- Equation from condition of corrected mean
  (eq2 : initial_sum = 50 * corrected_mean - understated_by) -- Restating in terms of initial sum
  : M = 36.06 := -- The initial mean of observations
  sorry -- Proof omitted

end NUMINAMATH_GPT_initial_mean_of_observations_l573_57396


namespace NUMINAMATH_GPT_find_p0_over_q0_l573_57333

-- Definitions

def p (x : ℝ) := 3 * (x - 4) * (x - 2)
def q (x : ℝ) := (x - 4) * (x + 3)

theorem find_p0_over_q0 : (p 0) / (q 0) = -2 :=
by
  -- Prove the equality given the conditions
  sorry

end NUMINAMATH_GPT_find_p0_over_q0_l573_57333


namespace NUMINAMATH_GPT_sum_a_b_is_95_l573_57315

-- Define the conditions
def product_condition (a b : ℕ) : Prop :=
  (a : ℤ) / 3 = 16 ∧ b = a - 1

-- Define the theorem to be proven
theorem sum_a_b_is_95 (a b : ℕ) (h : product_condition a b) : a + b = 95 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_b_is_95_l573_57315


namespace NUMINAMATH_GPT_combined_population_lake_bright_and_sunshine_hills_l573_57314

theorem combined_population_lake_bright_and_sunshine_hills
  (p_toadon p_gordonia p_lake_bright p_riverbank p_sunshine_hills : ℕ)
  (h1 : p_toadon + p_gordonia + p_lake_bright + p_riverbank + p_sunshine_hills = 120000)
  (h2 : p_gordonia = 1 / 3 * 120000)
  (h3 : p_toadon = 3 / 4 * p_gordonia)
  (h4 : p_riverbank = p_toadon + 2 / 5 * p_toadon) :
  p_lake_bright + p_sunshine_hills = 8000 :=
by
  sorry

end NUMINAMATH_GPT_combined_population_lake_bright_and_sunshine_hills_l573_57314


namespace NUMINAMATH_GPT_num_men_employed_l573_57332

noncomputable def original_number_of_men (M : ℕ) : Prop :=
  let total_work_original := M * 5
  let total_work_actual := (M - 8) * 15
  total_work_original = total_work_actual

theorem num_men_employed (M : ℕ) (h : original_number_of_men M) : M = 12 :=
by sorry

end NUMINAMATH_GPT_num_men_employed_l573_57332


namespace NUMINAMATH_GPT_no_four_points_with_all_odd_distances_l573_57372

theorem no_four_points_with_all_odd_distances :
  ∀ (A B C D : ℝ × ℝ),
    (∃ (x y z p q r : ℕ),
      (x = dist A B ∧ x % 2 = 1) ∧
      (y = dist B C ∧ y % 2 = 1) ∧
      (z = dist C D ∧ z % 2 = 1) ∧
      (p = dist D A ∧ p % 2 = 1) ∧
      (q = dist A C ∧ q % 2 = 1) ∧
      (r = dist B D ∧ r % 2 = 1))
    → false :=
by
  sorry

end NUMINAMATH_GPT_no_four_points_with_all_odd_distances_l573_57372


namespace NUMINAMATH_GPT_sin_double_angle_l573_57317

theorem sin_double_angle (α : ℝ) (h_tan : Real.tan α < 0) (h_sin : Real.sin α = - (Real.sqrt 3) / 3) :
  Real.sin (2 * α) = - (2 * Real.sqrt 2) / 3 := 
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l573_57317


namespace NUMINAMATH_GPT_vacation_cost_l573_57397

theorem vacation_cost (C P : ℕ) 
    (h1 : C = 5 * P)
    (h2 : C = 7 * (P - 40))
    (h3 : C = 8 * (P - 60)) : C = 700 := 
by 
    sorry

end NUMINAMATH_GPT_vacation_cost_l573_57397


namespace NUMINAMATH_GPT_percentage_food_given_out_l573_57371

theorem percentage_food_given_out 
  (first_week_donations : ℕ)
  (second_week_donations : ℕ)
  (total_amount_donated : ℕ)
  (remaining_food : ℕ)
  (amount_given_out : ℕ)
  (percentage_given_out : ℕ) : 
  (first_week_donations = 40) →
  (second_week_donations = 2 * first_week_donations) →
  (total_amount_donated = first_week_donations + second_week_donations) →
  (remaining_food = 36) →
  (amount_given_out = total_amount_donated - remaining_food) →
  (percentage_given_out = (amount_given_out * 100) / total_amount_donated) →
  percentage_given_out = 70 :=
by sorry

end NUMINAMATH_GPT_percentage_food_given_out_l573_57371


namespace NUMINAMATH_GPT_prove_inequality_l573_57373

noncomputable def problem_statement (p q r : ℝ) (n : ℕ) (h_pqr : p * q * r = 1) : Prop :=
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1

theorem prove_inequality (p q r : ℝ) (n : ℕ) (h_pqr : p * q * r = 1) (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r) : 
  problem_statement p q r n h_pqr :=
by
  sorry

end NUMINAMATH_GPT_prove_inequality_l573_57373


namespace NUMINAMATH_GPT_merchant_marked_price_percent_l573_57342

theorem merchant_marked_price_percent (L : ℝ) (hL : L = 100) (purchase_price : ℝ) (h1 : purchase_price = L * 0.70) (x : ℝ)
  (selling_price : ℝ) (h2 : selling_price = x * 0.75) :
  (selling_price - purchase_price) / selling_price = 0.30 → x = 133.33 :=
by
  sorry

end NUMINAMATH_GPT_merchant_marked_price_percent_l573_57342


namespace NUMINAMATH_GPT_parabola_intercepts_sum_l573_57390

theorem parabola_intercepts_sum (a b c : ℝ)
  (h₁ : a = 5)
  (h₂ : b = (9 + Real.sqrt 21) / 6)
  (h₃ : c = (9 - Real.sqrt 21) / 6) :
  a + b + c = 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_intercepts_sum_l573_57390


namespace NUMINAMATH_GPT_chris_money_left_l573_57389

def video_game_cost : ℕ := 60
def candy_cost : ℕ := 5
def babysitting_rate : ℕ := 8
def hours_worked : ℕ := 9
def earnings : ℕ := babysitting_rate * hours_worked
def total_cost : ℕ := video_game_cost + candy_cost
def money_left : ℕ := earnings - total_cost

theorem chris_money_left
  (h1 : video_game_cost = 60)
  (h2 : candy_cost = 5)
  (h3 : babysitting_rate = 8)
  (h4 : hours_worked = 9) :
  money_left = 7 :=
by
  -- The detailed proof is omitted.
  sorry

end NUMINAMATH_GPT_chris_money_left_l573_57389


namespace NUMINAMATH_GPT_max_value_of_x1_squared_plus_x2_squared_l573_57329

theorem max_value_of_x1_squared_plus_x2_squared :
  ∀ (k : ℝ), -4 ≤ k ∧ k ≤ -4 / 3 → (∃ x1 x2 : ℝ, x1^2 + x2^2 = 18) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_x1_squared_plus_x2_squared_l573_57329


namespace NUMINAMATH_GPT_number_of_people_adopting_cats_l573_57311

theorem number_of_people_adopting_cats 
    (initial_cats : ℕ)
    (monday_kittens : ℕ)
    (tuesday_injured_cat : ℕ)
    (final_cats : ℕ)
    (cats_per_person_adopting : ℕ)
    (h_initial : initial_cats = 20)
    (h_monday : monday_kittens = 2)
    (h_tuesday : tuesday_injured_cat = 1)
    (h_final: final_cats = 17)
    (h_cats_per_person: cats_per_person_adopting = 2) :
    ∃ (people_adopting : ℕ), people_adopting = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_adopting_cats_l573_57311


namespace NUMINAMATH_GPT_neg_3_14_gt_neg_pi_l573_57302

theorem neg_3_14_gt_neg_pi (π : ℝ) (h : 0 < π) : -3.14 > -π := 
sorry

end NUMINAMATH_GPT_neg_3_14_gt_neg_pi_l573_57302


namespace NUMINAMATH_GPT_intersection_A_B_l573_57348

def A : Set ℝ := { x | x > -1 }
def B : Set ℝ := { y | (y - 2) * (y + 3) < 0 }

theorem intersection_A_B : A ∩ B = Set.Ioo (-1) 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l573_57348


namespace NUMINAMATH_GPT_smallest_number_divisible_remainders_l573_57354

theorem smallest_number_divisible_remainders :
  ∃ n : ℕ,
    (n % 10 = 9) ∧
    (n % 9 = 8) ∧
    (n % 8 = 7) ∧
    (n % 7 = 6) ∧
    (n % 6 = 5) ∧
    (n % 5 = 4) ∧
    (n % 4 = 3) ∧
    (n % 3 = 2) ∧
    (n % 2 = 1) ∧
    n = 2519 :=
sorry

end NUMINAMATH_GPT_smallest_number_divisible_remainders_l573_57354


namespace NUMINAMATH_GPT_rank_matrix_sum_l573_57383

theorem rank_matrix_sum (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (h : ∀ i j, A i j = ↑i + ↑j) : Matrix.rank A = 2 := by
  sorry

end NUMINAMATH_GPT_rank_matrix_sum_l573_57383


namespace NUMINAMATH_GPT_tomas_first_month_distance_l573_57369

theorem tomas_first_month_distance 
  (distance_n_5 : ℝ := 26.3)
  (double_distance_each_month : ∀ (n : ℕ), n ≥ 1 → (distance_n : ℝ) = distance_n_5 / (2 ^ (5 - n)))
  : distance_n_5 / (2 ^ (5 - 1)) = 1.64375 :=
by
  sorry

end NUMINAMATH_GPT_tomas_first_month_distance_l573_57369


namespace NUMINAMATH_GPT_circle_passing_given_points_l573_57385

theorem circle_passing_given_points :
  ∃ (D E F : ℚ), (F = 0) ∧ (E = - (9 / 5)) ∧ (D = 19 / 5) ∧
  (∀ (x y : ℚ), x^2 + y^2 + D * x + E * y + F = 0 ↔ (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 3) ∨ (x = -4 ∧ y = 1)) :=
by
  sorry

end NUMINAMATH_GPT_circle_passing_given_points_l573_57385


namespace NUMINAMATH_GPT_complement_of_angle_correct_l573_57336

noncomputable def complement_of_angle (α : ℝ) : ℝ := 90 - α

theorem complement_of_angle_correct (α : ℝ) (h : complement_of_angle α = 125 + 12 / 60) :
  complement_of_angle α = 35 + 12 / 60 :=
by
  sorry

end NUMINAMATH_GPT_complement_of_angle_correct_l573_57336


namespace NUMINAMATH_GPT_num_socks_in_machine_l573_57363

-- Definition of the number of people who played the match
def num_players : ℕ := 11

-- Definition of the number of socks per player
def socks_per_player : ℕ := 2

-- The goal is to prove that the total number of socks in the washing machine is 22
theorem num_socks_in_machine : num_players * socks_per_player = 22 :=
by
  sorry

end NUMINAMATH_GPT_num_socks_in_machine_l573_57363


namespace NUMINAMATH_GPT_perimeters_equal_l573_57376

noncomputable def side_length_square := 15 -- cm
noncomputable def length_rectangle := 18 -- cm
noncomputable def area_rectangle := 216 -- cm²

theorem perimeters_equal :
  let perimeter_square := 4 * side_length_square
  let width_rectangle := area_rectangle / length_rectangle
  let perimeter_rectangle := 2 * (length_rectangle + width_rectangle)
  perimeter_square = perimeter_rectangle :=
by
  sorry

end NUMINAMATH_GPT_perimeters_equal_l573_57376


namespace NUMINAMATH_GPT_vegetables_sold_mass_correct_l573_57356

-- Definitions based on the problem's conditions
def mass_carrots : ℕ := 15
def mass_zucchini : ℕ := 13
def mass_broccoli : ℕ := 8
def total_mass_vegetables := mass_carrots + mass_zucchini + mass_broccoli
def mass_of_vegetables_sold := total_mass_vegetables / 2

-- Theorem to be proved
theorem vegetables_sold_mass_correct : mass_of_vegetables_sold = 18 := by 
  sorry

end NUMINAMATH_GPT_vegetables_sold_mass_correct_l573_57356


namespace NUMINAMATH_GPT_solve_for_a_l573_57300

-- Defining the equation and given solution
theorem solve_for_a (x a : ℝ) (h : 2 * x - 5 * a = 3 * a + 22) (hx : x = 3) : a = -2 := by
  sorry

end NUMINAMATH_GPT_solve_for_a_l573_57300


namespace NUMINAMATH_GPT_min_a5_of_geom_seq_l573_57306

-- Definition of geometric sequence positivity and difference condition.
def geom_seq_pos_diff (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (a 3 - a 1 = 2)

-- The main theorem stating that the minimum value of a_5 is 8.
theorem min_a5_of_geom_seq {a : ℕ → ℝ} {q : ℝ} (h : geom_seq_pos_diff a q) :
  a 5 ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_a5_of_geom_seq_l573_57306


namespace NUMINAMATH_GPT_birdhouse_distance_l573_57349

theorem birdhouse_distance (car_distance : ℕ) (lawnchair_distance : ℕ) (birdhouse_distance : ℕ) 
  (h1 : car_distance = 200) 
  (h2 : lawnchair_distance = 2 * car_distance) 
  (h3 : birdhouse_distance = 3 * lawnchair_distance) : 
  birdhouse_distance = 1200 :=
by
  sorry

end NUMINAMATH_GPT_birdhouse_distance_l573_57349


namespace NUMINAMATH_GPT_determine_x_l573_57393

theorem determine_x (x : ℝ) (h : 9 * x^2 + 2 * x^2 + 3 * x^2 / 2 = 300) : x = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_GPT_determine_x_l573_57393


namespace NUMINAMATH_GPT_average_goals_l573_57375

theorem average_goals (c s j : ℕ) (h1 : c = 4) (h2 : s = c / 2) (h3 : j = 2 * s - 3) :
  c + s + j = 7 :=
sorry

end NUMINAMATH_GPT_average_goals_l573_57375


namespace NUMINAMATH_GPT_B_knit_time_l573_57368

theorem B_knit_time (x : ℕ) (hA : 3 > 0) (h_combined_rate : 1/3 + 1/x = 1/2) : x = 6 := sorry

end NUMINAMATH_GPT_B_knit_time_l573_57368


namespace NUMINAMATH_GPT_number_is_125_l573_57374

/-- Let x be a real number such that the difference between x and 3/5 of x is 50. -/
def problem_statement (x : ℝ) : Prop :=
  x - (3 / 5) * x = 50

/-- Prove that the only number that satisfies the above condition is 125. -/
theorem number_is_125 (x : ℝ) (h : problem_statement x) : x = 125 :=
by
  sorry

end NUMINAMATH_GPT_number_is_125_l573_57374


namespace NUMINAMATH_GPT_sales_difference_greatest_in_june_l573_57350

def percentage_difference (D B : ℕ) : ℚ :=
  if B = 0 then 0 else (↑(max D B - min D B) / ↑(min D B)) * 100

def january : ℕ × ℕ := (8, 5)
def february : ℕ × ℕ := (10, 5)
def march : ℕ × ℕ := (8, 8)
def april : ℕ × ℕ := (4, 8)
def may : ℕ × ℕ := (5, 10)
def june : ℕ × ℕ := (3, 9)

noncomputable
def greatest_percentage_difference_month : String :=
  let jan_diff := percentage_difference january.1 january.2
  let feb_diff := percentage_difference february.1 february.2
  let mar_diff := percentage_difference march.1 march.2
  let apr_diff := percentage_difference april.1 april.2
  let may_diff := percentage_difference may.1 may.2
  let jun_diff := percentage_difference june.1 june.2
  if max jan_diff (max feb_diff (max mar_diff (max apr_diff (max may_diff jun_diff)))) == jun_diff
  then "June" else "Not June"
  
theorem sales_difference_greatest_in_june : greatest_percentage_difference_month = "June" :=
  by sorry

end NUMINAMATH_GPT_sales_difference_greatest_in_june_l573_57350


namespace NUMINAMATH_GPT_find_power_l573_57301

noncomputable def x : Real := 14.500000000000002
noncomputable def target : Real := 126.15

theorem find_power (n : Real) (h : (3/5) * x^n = target) : n = 2 :=
sorry

end NUMINAMATH_GPT_find_power_l573_57301


namespace NUMINAMATH_GPT_arithmetic_to_geometric_progression_l573_57343

theorem arithmetic_to_geometric_progression (d : ℝ) (h : ∀ d, (4 + d) * (4 + d) = 7 * (22 + 2 * d)) :
  ∃ d, 7 + 2 * d = 3.752 :=
sorry

end NUMINAMATH_GPT_arithmetic_to_geometric_progression_l573_57343


namespace NUMINAMATH_GPT_maximum_triangle_area_within_circles_l573_57331

noncomputable def radius1 : ℕ := 71
noncomputable def radius2 : ℕ := 100
noncomputable def largest_triangle_area : ℕ := 24200

theorem maximum_triangle_area_within_circles : 
  ∃ (L : ℕ), L = largest_triangle_area ∧ 
             ∀ (r1 r2 : ℕ), r1 = radius1 → 
                             r2 = radius2 → 
                             L ≥ (r1 * r1 + 2 * r1 * r2) :=
by
  sorry

end NUMINAMATH_GPT_maximum_triangle_area_within_circles_l573_57331


namespace NUMINAMATH_GPT_polygon_perimeter_is_35_l573_57340

-- Define the concept of a regular polygon with given side length and exterior angle
def regular_polygon_perimeter (n : ℕ) (side_length : ℕ) : ℕ := 
  n * side_length

theorem polygon_perimeter_is_35 (side_length : ℕ) (exterior_angle : ℕ) (n : ℕ)
  (h1 : side_length = 7) (h2 : exterior_angle = 72) (h3 : 360 / exterior_angle = n) :
  regular_polygon_perimeter n side_length = 35 :=
by
  -- We skip the proof body as only the statement is required
  sorry

end NUMINAMATH_GPT_polygon_perimeter_is_35_l573_57340


namespace NUMINAMATH_GPT_angles_on_x_axis_eq_l573_57335

open Set

def S1 : Set ℝ := { β | ∃ k : ℤ, β = k * 360 }
def S2 : Set ℝ := { β | ∃ k : ℤ, β = 180 + k * 360 }
def S_total : Set ℝ := S1 ∪ S2
def S_target : Set ℝ := { β | ∃ n : ℤ, β = n * 180 }

theorem angles_on_x_axis_eq : S_total = S_target := 
by 
  sorry

end NUMINAMATH_GPT_angles_on_x_axis_eq_l573_57335


namespace NUMINAMATH_GPT_peter_total_books_is_20_l573_57357

noncomputable def total_books_peter_has (B : ℝ) : Prop :=
  let Peter_Books_Read := 0.40 * B
  let Brother_Books_Read := 0.10 * B
  Peter_Books_Read = Brother_Books_Read + 6

theorem peter_total_books_is_20 :
  ∃ B : ℝ, total_books_peter_has B ∧ B = 20 := 
by
  sorry

end NUMINAMATH_GPT_peter_total_books_is_20_l573_57357


namespace NUMINAMATH_GPT_min_value_abc_l573_57360

open Real

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/10368 :=
sorry

end NUMINAMATH_GPT_min_value_abc_l573_57360


namespace NUMINAMATH_GPT_burn_5_sticks_per_hour_l573_57364

-- Define the number of sticks each type of furniture makes
def sticks_per_chair := 6
def sticks_per_table := 9
def sticks_per_stool := 2

-- Define the number of each furniture Mary chopped up
def chairs_chopped := 18
def tables_chopped := 6
def stools_chopped := 4

-- Define the total number of hours Mary can keep warm
def hours_warm := 34

-- Calculate the total number of sticks of wood from each type of furniture
def total_sticks_chairs := chairs_chopped * sticks_per_chair
def total_sticks_tables := tables_chopped * sticks_per_table
def total_sticks_stools := stools_chopped * sticks_per_stool

-- Calculate the total number of sticks of wood
def total_sticks := total_sticks_chairs + total_sticks_tables + total_sticks_stools

-- The number of sticks of wood Mary needs to burn per hour
def sticks_per_hour := total_sticks / hours_warm

-- Prove that Mary needs to burn 5 sticks per hour to stay warm
theorem burn_5_sticks_per_hour : sticks_per_hour = 5 := sorry

end NUMINAMATH_GPT_burn_5_sticks_per_hour_l573_57364


namespace NUMINAMATH_GPT_solve_for_z_l573_57327

theorem solve_for_z (a b s z : ℝ) (h1 : z ≠ 0) (h2 : 1 - 6 * s ≠ 0) (h3 : z = a^3 * b^2 + 6 * z * s - 9 * s^2) :
  z = (a^3 * b^2 - 9 * s^2) / (1 - 6 * s) := 
 by
  sorry

end NUMINAMATH_GPT_solve_for_z_l573_57327


namespace NUMINAMATH_GPT_no_real_roots_f_of_f_x_eq_x_l573_57319

theorem no_real_roots_f_of_f_x_eq_x (a b c : ℝ) (h: (b - 1)^2 - 4 * a * c < 0) : 
  ¬(∃ x : ℝ, (a * (a * x^2 + b * x + c)^2 + b * (a * x^2 + b * x + c) + c = x)) := 
by
  sorry

end NUMINAMATH_GPT_no_real_roots_f_of_f_x_eq_x_l573_57319


namespace NUMINAMATH_GPT_x_minus_y_values_l573_57308

theorem x_minus_y_values (x y : ℝ) (h₁ : |x + 1| = 4) (h₂ : (y + 2)^2 = 4) (h₃ : x + y ≥ -5) :
  x - y = -5 ∨ x - y = 3 ∨ x - y = 7 :=
by
  sorry

end NUMINAMATH_GPT_x_minus_y_values_l573_57308


namespace NUMINAMATH_GPT_different_kinds_of_hamburgers_l573_57362

theorem different_kinds_of_hamburgers 
  (n_condiments : ℕ) 
  (condiment_choices : ℕ)
  (meat_patty_choices : ℕ)
  (h1 : n_condiments = 8)
  (h2 : condiment_choices = 2 ^ n_condiments)
  (h3 : meat_patty_choices = 3)
  : condiment_choices * meat_patty_choices = 768 := 
by
  sorry

end NUMINAMATH_GPT_different_kinds_of_hamburgers_l573_57362


namespace NUMINAMATH_GPT_find_x_equals_4_l573_57384

noncomputable def repeatingExpr (x : ℝ) : ℝ :=
2 + 4 / (1 + 4 / (2 + 4 / (1 + 4 / x)))

theorem find_x_equals_4 :
  ∃ x : ℝ, x = repeatingExpr x ∧ x = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_find_x_equals_4_l573_57384


namespace NUMINAMATH_GPT_symmetric_point_l573_57346

-- Definitions
def P : ℝ × ℝ := (5, -2)
def line (x y : ℝ) : Prop := x - y + 5 = 0

-- Statement 
theorem symmetric_point (a b : ℝ) 
  (symmetric_condition1 : ∀ x y, line x y → (b + 2)/(a - 5) * 1 = -1)
  (symmetric_condition2 : ∀ x y, line x y → (a + 5)/2 - (b - 2)/2 + 5 = 0) :
  (a, b) = (-7, 10) :=
sorry

end NUMINAMATH_GPT_symmetric_point_l573_57346


namespace NUMINAMATH_GPT_minimum_value_f_1_minimum_value_f_e_minimum_value_f_m_minus_1_range_of_m_l573_57307

noncomputable def f (x m : ℝ) : ℝ := x - m * Real.log x - (m - 1) / x
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem minimum_value_f_1 (m : ℝ) : (m ≤ 2) → f 1 m = 2 - m := sorry

theorem minimum_value_f_e (m : ℝ) : (m ≥ Real.exp 1 + 1) → f (Real.exp 1) m = Real.exp 1 - m - (m - 1) / Real.exp 1 := sorry

theorem minimum_value_f_m_minus_1 (m : ℝ) : (2 < m ∧ m < Real.exp 1 + 1) → 
  f (m - 1) m = m - 2 - m * Real.log (m - 1) := sorry

theorem range_of_m (m : ℝ) : 
  (m ≤ 2) → 
  (∃ x1 ∈ Set.Icc (Real.exp 1) (Real.exp 1 ^ 2), ∀ x2 ∈ Set.Icc (-2 : ℝ) 0, f x1 m ≤ g x2) → 
  Real.exp 1 - m - (m - 1) / Real.exp 1 ≤ 1 → 
  (m ≥ (Real.exp 1 ^ 2 - Real.exp 1 + 1) / (Real.exp 1 + 1) ∧ m ≤ 2) := sorry

end NUMINAMATH_GPT_minimum_value_f_1_minimum_value_f_e_minimum_value_f_m_minus_1_range_of_m_l573_57307


namespace NUMINAMATH_GPT_map_length_conversion_l573_57310

-- Define the given condition: 12 cm on the map represents 72 km in reality.
def length_on_map := 12 -- in cm
def distance_in_reality := 72 -- in km

-- Define the length in cm we want to find the real-world distance for.
def query_length := 17 -- in cm

-- State the proof problem.
theorem map_length_conversion :
  (distance_in_reality / length_on_map) * query_length = 102 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_map_length_conversion_l573_57310


namespace NUMINAMATH_GPT_Janet_initial_crayons_l573_57387

variable (Michelle_initial Janet_initial Michelle_final : ℕ)

theorem Janet_initial_crayons (h1 : Michelle_initial = 2) (h2 : Michelle_final = 4) (h3 : Michelle_final = Michelle_initial + Janet_initial) :
  Janet_initial = 2 :=
by
  sorry

end NUMINAMATH_GPT_Janet_initial_crayons_l573_57387
