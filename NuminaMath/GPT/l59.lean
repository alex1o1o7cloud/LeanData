import Mathlib

namespace nicolai_ate_6_pounds_of_peaches_l59_59820

noncomputable def total_weight_pounds : ℝ := 8
noncomputable def pound_to_ounce : ℝ := 16
noncomputable def mario_weight_ounces : ℝ := 8
noncomputable def lydia_weight_ounces : ℝ := 24

theorem nicolai_ate_6_pounds_of_peaches :
  (total_weight_pounds * pound_to_ounce - (mario_weight_ounces + lydia_weight_ounces)) / pound_to_ounce = 6 :=
by
  sorry

end nicolai_ate_6_pounds_of_peaches_l59_59820


namespace find_CD_l59_59300

theorem find_CD (C D : ℚ) :
  (∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 → (7 * x - 4) / (x ^ 2 - 9 * x - 36) = C / (x - 12) + D / (x + 3))
  → C = 16 / 3 ∧ D = 5 / 3 :=
by
  sorry

end find_CD_l59_59300


namespace problem_statement_l59_59150

variables {AB CD BC DA : ℝ} (E : ℝ) (midpoint_E : E = BC / 2) (ins_ABC : circle_inscribable AB ED)
  (ins_AEC : circle_inscribable AE CD) (a b c d : ℝ) (h_AB : AB = a) (h_BC : BC = b) (h_CD : CD = c)
  (h_DA : DA = d)

theorem problem_statement :
  a + c = b / 3 + d ∧ (1 / a + 1 / c = 3 / b) :=
by
  sorry

end problem_statement_l59_59150


namespace tank_capacity_l59_59979

theorem tank_capacity (x : ℝ) (h : (5/12) * x = 150) : x = 360 :=
by
  sorry

end tank_capacity_l59_59979


namespace stacy_berries_multiple_l59_59777

theorem stacy_berries_multiple (Skylar_berries : ℕ) (Stacy_berries : ℕ) (Steve_berries : ℕ) (m : ℕ)
  (h1 : Skylar_berries = 20)
  (h2 : Steve_berries = Skylar_berries / 2)
  (h3 : Stacy_berries = m * Steve_berries + 2)
  (h4 : Stacy_berries = 32) :
  m = 3 :=
by
  sorry

end stacy_berries_multiple_l59_59777


namespace fraction_arithmetic_l59_59549

theorem fraction_arithmetic : 
  (2 / 5 + 3 / 7) / (4 / 9 * 1 / 8) = 522 / 35 := by
  sorry

end fraction_arithmetic_l59_59549


namespace pencil_distribution_l59_59272

theorem pencil_distribution (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) 
  (h1 : total_pens = 1001) (h2 : total_pencils = 910) (h3 : max_students = 91) : 
  total_pencils / max_students = 10 :=
by
  sorry

end pencil_distribution_l59_59272


namespace infinite_div_by_100_l59_59599

theorem infinite_div_by_100 : ∀ k : ℕ, ∃ n : ℕ, n > 0 ∧ (2 ^ n + n ^ 2) % 100 = 0 :=
by
  sorry

end infinite_div_by_100_l59_59599


namespace percentage_wearing_blue_shirts_l59_59880

theorem percentage_wearing_blue_shirts (total_students : ℕ) (red_percentage green_percentage : ℕ) 
  (other_students : ℕ) (H1 : total_students = 900) (H2 : red_percentage = 28) 
  (H3 : green_percentage = 10) (H4 : other_students = 162) : 
  (44 : ℕ) = 100 - (red_percentage + green_percentage + (other_students * 100 / total_students)) :=
by
  sorry

end percentage_wearing_blue_shirts_l59_59880


namespace circle_radius_tangent_to_semicircles_and_sides_l59_59337

noncomputable def side_length_of_square : ℝ := 4
noncomputable def side_length_of_smaller_square : ℝ := side_length_of_square / 2
noncomputable def radius_of_semicircle : ℝ := side_length_of_smaller_square / 2
noncomputable def distance_from_center_to_tangent_point : ℝ := Real.sqrt (side_length_of_smaller_square^2 + radius_of_semicircle^2)

theorem circle_radius_tangent_to_semicircles_and_sides : 
  ∃ (r : ℝ), r = (Real.sqrt 5 - 1) / 2 :=
by
  have r : ℝ := (Real.sqrt 5 - 1) / 2
  use r
  sorry -- Proof omitted

end circle_radius_tangent_to_semicircles_and_sides_l59_59337


namespace toms_age_l59_59555

variable (T J : ℕ)

theorem toms_age :
  (J - 6 = 3 * (T - 6)) ∧ (J + 4 = 2 * (T + 4)) → T = 16 :=
by
  intros h
  sorry

end toms_age_l59_59555


namespace ratio_female_to_male_l59_59323

variable (m f : ℕ)

-- Average ages given in the conditions
def avg_female_age : ℕ := 35
def avg_male_age : ℕ := 45
def avg_total_age : ℕ := 40

-- Total ages based on number of members
def total_female_age (f : ℕ) : ℕ := avg_female_age * f
def total_male_age (m : ℕ) : ℕ := avg_male_age * m
def total_age (f m : ℕ) : ℕ := total_female_age f + total_male_age m

-- Equation based on average age of all members
def avg_age_eq (f m : ℕ) : Prop :=
  total_age f m / (f + m) = avg_total_age

theorem ratio_female_to_male : avg_age_eq f m → f = m :=
by
  sorry

end ratio_female_to_male_l59_59323


namespace arithmetic_seq_a4_l59_59799

-- Definition of an arithmetic sequence with the first three terms given.
def arithmetic_seq (a : ℕ → ℕ) :=
  a 0 = 2 ∧ a 1 = 4 ∧ a 2 = 6 ∧ ∃ d, ∀ n, a (n + 1) = a n + d

-- The actual proof goal.
theorem arithmetic_seq_a4 : ∃ a : ℕ → ℕ, arithmetic_seq a ∧ a 3 = 8 :=
by
  sorry

end arithmetic_seq_a4_l59_59799


namespace obtuse_angles_at_intersection_l59_59158

theorem obtuse_angles_at_intersection (lines_intersect_x_at_diff_points : Prop) (lines_not_perpendicular : Prop) 
(lines_form_obtuse_angle_at_intersection : Prop) : 
(lines_intersect_x_at_diff_points ∧ lines_not_perpendicular ∧ lines_form_obtuse_angle_at_intersection) → 
  ∃ (n : ℕ), n = 2 :=
by 
  sorry

end obtuse_angles_at_intersection_l59_59158


namespace sahil_purchase_price_l59_59397

def purchase_price (P : ℝ) : Prop :=
  let repair_cost := 5000
  let transportation_charges := 1000
  let total_cost := repair_cost + transportation_charges
  let selling_price := 27000
  let profit_factor := 1.5
  profit_factor * (P + total_cost) = selling_price

theorem sahil_purchase_price : ∃ P : ℝ, purchase_price P ∧ P = 12000 :=
by
  use 12000
  unfold purchase_price
  simp
  sorry

end sahil_purchase_price_l59_59397


namespace find_x_l59_59028

theorem find_x :
  (x : ℝ) →
  (0.40 * 2 = 0.25 * (0.30 * 15 + x)) →
  x = -1.3 :=
by
  intros x h
  sorry

end find_x_l59_59028


namespace maximize_farmer_profit_l59_59349

theorem maximize_farmer_profit :
  ∃ x y : ℝ, x + y ≤ 2 ∧ 3 * x + y ≤ 5 ∧ x ≥ 0 ∧ y ≥ 0 ∧ x = 1.5 ∧ y = 0.5 ∧ 
  (∀ x' y' : ℝ, x' + y' ≤ 2 ∧ 3 * x' + y' ≤ 5 ∧ x' ≥ 0 ∧ y' ≥ 0 → 14400 * x + 6300 * y ≥ 14400 * x' + 6300 * y') :=
by
  sorry

end maximize_farmer_profit_l59_59349


namespace find_angle_A_find_range_expression_l59_59939

-- Define the variables and conditions in a way consistent with Lean's syntax
variables {α β γ : Type}
variables (a b c : ℝ) (A B C : ℝ)

-- The mathematical conditions translated to Lean
def triangle_condition (a b c A B C : ℝ) : Prop := (b + c) / a = Real.cos B + Real.cos C

-- Statement for Proof 1: Prove that A = π/2 given the conditions
theorem find_angle_A (h : triangle_condition a b c A B C) : A = Real.pi / 2 :=
sorry

-- Statement for Proof 2: Prove the range of the given expression under the given conditions
theorem find_range_expression (h : triangle_condition a b c A B C) (hA : A = Real.pi / 2) :
  ∃ (l u : ℝ), l = Real.sqrt 3 + 2 ∧ u = Real.sqrt 3 + 3 ∧ (2 * Real.cos (B / 2) ^ 2 + 2 * Real.sqrt 3 * Real.cos (C / 2) ^ 2) ∈ Set.Ioc l u :=
sorry

end find_angle_A_find_range_expression_l59_59939


namespace sqrt_of_neg_five_squared_l59_59788

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : ℝ)^2) = 5 ∨ Real.sqrt ((-5 : ℝ)^2) = -5 :=
by
  sorry

end sqrt_of_neg_five_squared_l59_59788


namespace train_b_speed_l59_59084

theorem train_b_speed (v : ℝ) (t : ℝ) (d : ℝ) (sA : ℝ := 30) (start_time_diff : ℝ := 2) :
  (d = 180) -> (60 + sA*t = d) -> (v * t = d) -> v = 45 := by 
  sorry

end train_b_speed_l59_59084


namespace area_relationship_l59_59948

theorem area_relationship (P Q R : ℝ) (h_square : 10 * 10 = 100)
  (h_triangle1 : P + R = 50)
  (h_triangle2 : Q + R = 50) :
  P - Q = 0 :=
by
  sorry

end area_relationship_l59_59948


namespace circle_chord_segments_l59_59710

theorem circle_chord_segments (r : ℝ) (ch : ℝ) (a : ℝ) :
  (r = 8) ∧ (ch = 12) ∧ (r^2 - a^2 = 36) →
  a = 2 * Real.sqrt 7 → ∃ (ak bk : ℝ), ak = 8 - 2 * Real.sqrt 7 ∧ bk = 8 + 2 * Real.sqrt 7 :=
by
  sorry

end circle_chord_segments_l59_59710


namespace k_zero_only_solution_l59_59301

noncomputable def polynomial_factorable (k : ℤ) : Prop :=
  ∃ (A B C D E F : ℤ), (A * D = 1) ∧ (B * E = 4) ∧ (A * E + B * D = k) ∧ (A * F + C * D = 1) ∧ (C * F = -k)

theorem k_zero_only_solution : ∀ k : ℤ, polynomial_factorable k ↔ k = 0 :=
by 
  sorry

end k_zero_only_solution_l59_59301


namespace costPerUse_l59_59520

-- Definitions based on conditions
def heatingPadCost : ℝ := 30
def usesPerWeek : ℕ := 3
def totalWeeks : ℕ := 2

-- Calculate the total number of uses
def totalUses : ℕ := usesPerWeek * totalWeeks

-- The amount spent per use
theorem costPerUse : heatingPadCost / totalUses = 5 := by
  sorry

end costPerUse_l59_59520


namespace rank_from_right_l59_59708

theorem rank_from_right (rank_from_left total_students : ℕ) (h1 : rank_from_left = 5) (h2 : total_students = 10) :
  total_students - rank_from_left + 1 = 6 :=
by 
  -- Placeholder for the actual proof.
  sorry

end rank_from_right_l59_59708


namespace meaningful_sqrt_range_l59_59863

theorem meaningful_sqrt_range (x : ℝ) : (3 * x - 6 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end meaningful_sqrt_range_l59_59863


namespace correct_operation_l59_59500

theorem correct_operation (a b : ℝ) : 
  ¬(a^2 + a^3 = a^5) ∧ ¬((a^2)^3 = a^8) ∧ (a^3 / a^2 = a) ∧ ¬((a - b)^2 = a^2 - b^2) := 
by {
  sorry
}

end correct_operation_l59_59500


namespace equilateral_sector_area_l59_59980

noncomputable def area_of_equilateral_sector (r : ℝ) : ℝ :=
  if h : r = r then (1/2) * r^2 * 1 else 0

theorem equilateral_sector_area (r : ℝ) : r = 2 → area_of_equilateral_sector r = 2 :=
by
  intros hr
  rw [hr]
  unfold area_of_equilateral_sector
  split_ifs
  · norm_num
  · contradiction

end equilateral_sector_area_l59_59980


namespace polynomial_sum_of_squares_is_23456_l59_59903

theorem polynomial_sum_of_squares_is_23456 (p q r s t u : ℤ) :
  (∀ x, 1728 * x ^ 3 + 64 = (p * x ^ 2 + q * x + r) * (s * x ^ 2 + t * x + u)) →
  p ^ 2 + q ^ 2 + r ^ 2 + s ^ 2 + t ^ 2 + u ^ 2 = 23456 :=
by
  sorry

end polynomial_sum_of_squares_is_23456_l59_59903


namespace yolanda_walking_rate_l59_59371

theorem yolanda_walking_rate 
  (d_xy : ℕ) (bob_start_after_yolanda : ℕ) (bob_distance_walked : ℕ) 
  (bob_rate : ℕ) (y : ℕ) 
  (bob_distance_to_time : bob_rate ≠ 0 ∧ bob_distance_walked / bob_rate = 2) 
  (yolanda_distance_walked : d_xy - bob_distance_walked = 9 ∧ y = 9 / 3) : 
  y = 3 :=
by 
  sorry

end yolanda_walking_rate_l59_59371


namespace total_worksheets_l59_59468

theorem total_worksheets (x : ℕ) (h1 : 7 * (x - 8) = 63) : x = 17 := 
by {
  sorry
}

end total_worksheets_l59_59468


namespace suit_price_the_day_after_sale_l59_59888

def originalPrice : ℕ := 300
def increaseRate : ℚ := 0.20
def couponDiscount : ℚ := 0.30
def additionalReduction : ℚ := 0.10

def increasedPrice := originalPrice * (1 + increaseRate)
def priceAfterCoupon := increasedPrice * (1 - couponDiscount)
def finalPrice := increasedPrice * (1 - additionalReduction)

theorem suit_price_the_day_after_sale 
  (op : ℕ := originalPrice) 
  (ir : ℚ := increaseRate) 
  (cd : ℚ := couponDiscount) 
  (ar : ℚ := additionalReduction) :
  finalPrice = 324 := 
sorry

end suit_price_the_day_after_sale_l59_59888


namespace functional_expression_result_l59_59800

theorem functional_expression_result {f : ℝ → ℝ} (h : ∀ x y : ℝ, f (2 * x - 3 * y) - f (x + y) = -2 * x + 8 * y) :
  ∀ t : ℝ, (f (4 * t) - f t) / (f (3 * t) - f (2 * t)) = 3 :=
sorry

end functional_expression_result_l59_59800


namespace find_analytical_expression_of_f_l59_59756

-- Define the function f and the condition it needs to satisfy
variable (f : ℝ → ℝ)
variable (hf : ∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1))

-- State the objective to prove
theorem find_analytical_expression_of_f : 
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x / (1 + x) := by
  sorry

end find_analytical_expression_of_f_l59_59756


namespace remainder_problem_l59_59807

theorem remainder_problem (d r : ℤ) (h1 : 1237 % d = r)
    (h2 : 1694 % d = r) (h3 : 2791 % d = r) (hd : d > 1) :
    d - r = 134 := sorry

end remainder_problem_l59_59807


namespace ratio_problem_l59_59399

theorem ratio_problem 
  (a b c d : ℚ)
  (h₁ : a / b = 8)
  (h₂ : c / b = 5)
  (h₃ : c / d = 1 / 3) : 
  d / a = 15 / 8 := 
by 
  sorry

end ratio_problem_l59_59399


namespace five_point_eight_one_million_in_scientific_notation_l59_59002

theorem five_point_eight_one_million_in_scientific_notation :
  5.81 * 10^6 = 5.81e6 :=
sorry

end five_point_eight_one_million_in_scientific_notation_l59_59002


namespace trees_planted_l59_59093

-- Definitions for the quantities of lindens (x) and birches (y)
variables (x y : ℕ)

-- Definitions matching the given problem conditions
def condition1 := x + y > 14
def condition2 := y + 18 > 2 * x
def condition3 := x > 2 * y

-- The theorem stating that if the conditions hold, then x = 11 and y = 5
theorem trees_planted (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : 
  x = 11 ∧ y = 5 := 
sorry

end trees_planted_l59_59093


namespace minimum_value_of_f_range_of_x_l59_59243

noncomputable def f (x : ℝ) := |2*x + 1| + |2*x - 1|

-- Problem 1
theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 2 :=
by
  intro x
  sorry

-- Problem 2
theorem range_of_x (a b : ℝ) (h : |2*a + b| + |a| - (1/2) * |a + b| * f x ≥ 0) : 
  - (1/2) ≤ x ∧ x ≤ 1/2 :=
by
  sorry

end minimum_value_of_f_range_of_x_l59_59243


namespace calculation_is_correct_l59_59285

theorem calculation_is_correct : 450 / (6 * 5 - 10 / 2) = 18 :=
by {
  -- Let me provide an outline for solving this problem
  -- (6 * 5 - 10 / 2) must be determined first
  -- After that substituted into the fraction
  sorry
}

end calculation_is_correct_l59_59285


namespace exists_close_ratios_l59_59904

theorem exists_close_ratios (S : Finset ℝ) (h : S.card = 2000) :
  ∃ (a b c d : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧
  abs ((a - b) / (c - d) - 1) < 1 / 100000 :=
sorry

end exists_close_ratios_l59_59904


namespace total_students_l59_59635

-- Lean statement: Prove the number of students given the conditions.
theorem total_students (num_classrooms : ℕ) (num_buses : ℕ) (seats_per_bus : ℕ) 
  (students : ℕ) (h1 : num_classrooms = 87) (h2 : num_buses = 29) 
  (h3 : seats_per_bus = 2) (h4 : students = num_classrooms * num_buses * seats_per_bus) :
  students = 5046 :=
by
  sorry

end total_students_l59_59635


namespace cost_of_slices_eaten_by_dog_is_correct_l59_59796

noncomputable def total_cost_before_tax : ℝ :=
  2 * 3 + 1 * 2 + 1 * 5 + 3 * 0.5 + 0.25 + 1.5 + 1.25

noncomputable def sales_tax_rate : ℝ := 0.06

noncomputable def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax

noncomputable def slices : ℝ := 8

noncomputable def cost_per_slice : ℝ := total_cost_after_tax / slices

noncomputable def slices_eaten_by_dog : ℝ := 8 - 3

noncomputable def cost_of_slices_eaten_by_dog : ℝ := cost_per_slice * slices_eaten_by_dog

theorem cost_of_slices_eaten_by_dog_is_correct : 
  cost_of_slices_eaten_by_dog = 11.59 := by
    sorry

end cost_of_slices_eaten_by_dog_is_correct_l59_59796


namespace problem_l59_59489

theorem problem : 
  let b := 2 ^ 51
  let c := 4 ^ 25
  b > c :=
by 
  let b := 2 ^ 51
  let c := 4 ^ 25
  sorry

end problem_l59_59489


namespace sum_of_three_squares_power_l59_59586

theorem sum_of_three_squares_power (n a b c k : ℕ) (h : n = a^2 + b^2 + c^2) (h_pos : n > 0) (k_pos : k > 0) :
  ∃ A B C : ℕ, n^(2*k) = A^2 + B^2 + C^2 :=
by
  sorry

end sum_of_three_squares_power_l59_59586


namespace option_C_correct_l59_59370

theorem option_C_correct (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := 
by 
  sorry

end option_C_correct_l59_59370


namespace largest_power_dividing_factorial_l59_59348

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2015) : ∃ k : ℕ, (2015^k ∣ n!) ∧ k = 67 :=
by
  sorry

end largest_power_dividing_factorial_l59_59348


namespace average_of_four_digits_l59_59573

theorem average_of_four_digits (sum9 : ℤ) (avg9 : ℤ) (avg5 : ℤ) (sum4 : ℤ) (n : ℤ) :
  avg9 = 18 →
  n = 9 →
  sum9 = avg9 * n →
  avg5 = 26 →
  sum4 = sum9 - (avg5 * 5) →
  avg4 = sum4 / 4 →
  avg4 = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_of_four_digits_l59_59573


namespace factor_expression_l59_59935

theorem factor_expression (m n x y : ℝ) :
  m * (x - y) + n * (y - x) = (x - y) * (m - n) := by
  sorry

end factor_expression_l59_59935


namespace problem_1_problem_2_l59_59970

-- Definitions for sets A and B
def A (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 6
def B (x : ℝ) (m : ℝ) : Prop := (x - 1 + m) * (x - 1 - m) ≤ 0

-- Problem (1): What is A ∩ B when m = 3
theorem problem_1 : ∀ (x : ℝ), A x → B x 3 → (-1 ≤ x ∧ x ≤ 4) := by
  intro x hA hB
  sorry

-- Problem (2): What is the range of m if A ⊆ B and m > 0
theorem problem_2 (m : ℝ) : m > 0 → (∀ x, A x → B x m) → (m ≥ 5) := by
  intros hm hAB
  sorry

end problem_1_problem_2_l59_59970


namespace arithmetic_sequence_general_formula_l59_59316

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom a2 : a 2 = 6
axiom S5 : S 5 = 40

-- Prove the general formulas
theorem arithmetic_sequence_general_formula (n : ℕ)
  (h1 : ∃ d a1, ∀ n, a n = a1 + (n - 1) * d)
  (h2 : ∃ d a1, ∀ n, S n = n * ((2 * a1) + (n - 1) * d) / 2) :
  (a n = 2 * n + 2) ∧ (S n = n * (n + 3)) := by
  sorry

end arithmetic_sequence_general_formula_l59_59316


namespace Anna_needs_308_tulips_l59_59361

-- Define conditions as assertions or definitions
def number_of_eyes := 2
def red_tulips_per_eye := 8 
def number_of_eyebrows := 2
def purple_tulips_per_eyebrow := 5
def red_tulips_for_nose := 12
def red_tulips_for_smile := 18
def yellow_tulips_background := 9 * red_tulips_for_smile
def additional_purple_tulips_eyebrows := 4 * number_of_eyes * red_tulips_per_eye - number_of_eyebrows * purple_tulips_per_eyebrow
def yellow_tulips_for_nose := 3 * red_tulips_for_nose

-- Define total number of tulips for each color
def total_red_tulips := number_of_eyes * red_tulips_per_eye + red_tulips_for_nose + red_tulips_for_smile
def total_purple_tulips := number_of_eyebrows * purple_tulips_per_eyebrow + additional_purple_tulips_eyebrows
def total_yellow_tulips := yellow_tulips_background + yellow_tulips_for_nose

-- Define the total number of tulips
def total_tulips := total_red_tulips + total_purple_tulips + total_yellow_tulips

theorem Anna_needs_308_tulips :
  total_tulips = 308 :=
sorry

end Anna_needs_308_tulips_l59_59361


namespace tank_fill_time_l59_59669

theorem tank_fill_time :
  let fill_rate_A := 1 / 8
  let empty_rate_B := 1 / 24
  let combined_rate := fill_rate_A - empty_rate_B
  let time_with_both_pipes := 66
  let partial_fill := time_with_both_pipes * combined_rate
  let remaining_fill := 1 - (partial_fill % 1)
  let additional_time_A := remaining_fill / fill_rate_A
  time_with_both_pipes + additional_time_A = 70 :=
by
  let fill_rate_A := 1 / 8
  let empty_rate_B := 1 / 24
  let combined_rate := fill_rate_A - empty_rate_B
  let time_with_both_pipes := 66
  let partial_fill := time_with_both_pipes * combined_rate
  let remaining_fill := 1 - (partial_fill % 1)
  let additional_time_A := remaining_fill / fill_rate_A
  have h : time_with_both_pipes + additional_time_A = 70 := sorry
  exact h

end tank_fill_time_l59_59669


namespace total_possible_arrangements_l59_59524

-- Define the subjects
inductive Subject : Type
| PoliticalScience
| Chinese
| Mathematics
| English
| PhysicalEducation
| Physics

open Subject

-- Define the condition that the first period cannot be Chinese
def first_period_cannot_be_chinese (schedule : Fin 6 → Subject) : Prop :=
  schedule 0 ≠ Chinese

-- Define the condition that the fifth period cannot be English
def fifth_period_cannot_be_english (schedule : Fin 6 → Subject) : Prop :=
  schedule 4 ≠ English

-- Define the schedule includes six unique subjects
def schedule_includes_all_subjects (schedule : Fin 6 → Subject) : Prop :=
  ∀ s : Subject, ∃ i : Fin 6, schedule i = s

-- Define the main theorem to prove the total number of possible arrangements
theorem total_possible_arrangements : 
  ∃ (schedules : List (Fin 6 → Subject)), 
  (∀ schedule, schedule ∈ schedules → 
    first_period_cannot_be_chinese schedule ∧ 
    fifth_period_cannot_be_english schedule ∧ 
    schedule_includes_all_subjects schedule) ∧ 
  schedules.length = 600 :=
sorry

end total_possible_arrangements_l59_59524


namespace least_subtract_for_divisibility_l59_59654

theorem least_subtract_for_divisibility (n : ℕ) (hn : n = 427398) : 
  (∃ m : ℕ, n - m % 10 = 0 ∧ m = 2) :=
by
  sorry

end least_subtract_for_divisibility_l59_59654


namespace janice_typing_proof_l59_59950

noncomputable def janice_typing : Prop :=
  let initial_speed := 6
  let error_speed := 8
  let corrected_speed := 5
  let typing_duration_initial := 20
  let typing_duration_corrected := 15
  let erased_sentences := 40
  let typing_duration_after_lunch := 18
  let total_sentences_end_of_day := 536

  let sentences_initial_typing := typing_duration_initial * error_speed
  let sentences_post_error_typing := typing_duration_corrected * initial_speed
  let sentences_final_typing := typing_duration_after_lunch * corrected_speed

  let sentences_total_typed := sentences_initial_typing + sentences_post_error_typing - erased_sentences + sentences_final_typing

  let sentences_started_with := total_sentences_end_of_day - sentences_total_typed

  sentences_started_with = 236

theorem janice_typing_proof : janice_typing := by
  sorry

end janice_typing_proof_l59_59950


namespace intersection_A_B_l59_59276

open Set

variable (x : ℝ)

def setA : Set ℝ := {x | x^2 - 3 * x ≤ 0}
def setB : Set ℝ := {1, 2}

theorem intersection_A_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_A_B_l59_59276


namespace determine_k_l59_59193

variable (x y z k : ℝ)

theorem determine_k
  (h1 : 9 / (x - y) = 16 / (z + y))
  (h2 : k / (x + z) = 16 / (z + y)) :
  k = 25 := by
  sorry

end determine_k_l59_59193


namespace find_a7_a8_l59_59843

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

end find_a7_a8_l59_59843


namespace quadrilateral_area_l59_59885

def diagonal : ℝ := 15
def offset1 : ℝ := 6
def offset2 : ℝ := 4

theorem quadrilateral_area :
  (1/2) * diagonal * (offset1 + offset2) = 75 :=
by 
  sorry

end quadrilateral_area_l59_59885


namespace no_three_segments_form_triangle_l59_59720

theorem no_three_segments_form_triangle :
  ∃ (a : Fin 10 → ℕ), ∀ {i j k : Fin 10}, i < j → j < k → a i + a j ≤ a k :=
by
  sorry

end no_three_segments_form_triangle_l59_59720


namespace find_x_value_l59_59971

theorem find_x_value (x : ℝ) (h1 : x^2 + x = 6) (h2 : x^2 - 2 = 1) : x = 2 := sorry

end find_x_value_l59_59971


namespace prime_transformation_l59_59665

theorem prime_transformation (p : ℕ) (prime_p : Nat.Prime p) (h : p = 3) : ∃ q : ℕ, q = 13 * p + 2 ∧ Nat.Prime q :=
by
  use 41
  sorry

end prime_transformation_l59_59665


namespace cos_identity_of_angle_l59_59444

open Real

theorem cos_identity_of_angle (α : ℝ) :
  sin (π / 6 + α) = sqrt 3 / 3 → cos (π / 3 - α) = sqrt 3 / 3 :=
by
  intro h
  sorry

end cos_identity_of_angle_l59_59444


namespace total_surface_area_of_three_face_painted_cubes_l59_59114

def cube_side_length : ℕ := 9
def small_cube_side_length : ℕ := 1
def num_small_cubes_with_three_faces_painted : ℕ := 8
def surface_area_of_each_painted_face : ℕ := 6

theorem total_surface_area_of_three_face_painted_cubes :
  num_small_cubes_with_three_faces_painted * surface_area_of_each_painted_face = 48 := by
  sorry

end total_surface_area_of_three_face_painted_cubes_l59_59114


namespace number_of_newspapers_l59_59086

theorem number_of_newspapers (total_reading_materials magazines_sold: ℕ) (h_total: total_reading_materials = 700) (h_magazines: magazines_sold = 425) : 
  ∃ newspapers_sold : ℕ, newspapers_sold + magazines_sold = total_reading_materials ∧ newspapers_sold = 275 :=
by
  sorry

end number_of_newspapers_l59_59086


namespace ruby_shares_with_9_friends_l59_59545

theorem ruby_shares_with_9_friends
    (total_candies : ℕ) (candies_per_friend : ℕ)
    (h1 : total_candies = 36) (h2 : candies_per_friend = 4) :
    total_candies / candies_per_friend = 9 := by
  sorry

end ruby_shares_with_9_friends_l59_59545


namespace average_age_of_class_l59_59925

theorem average_age_of_class 
  (avg_age_8 : ℕ → ℕ)
  (avg_age_6 : ℕ → ℕ)
  (age_15th : ℕ)
  (A : ℕ)
  (h1 : avg_age_8 8 = 112)
  (h2 : avg_age_6 6 = 96)
  (h3 : age_15th = 17)
  (h4 : 15 * A = (avg_age_8 8) + (avg_age_6 6) + age_15th)
  : A = 15 :=
by
  sorry

end average_age_of_class_l59_59925


namespace probability_symmetric_line_l59_59120

theorem probability_symmetric_line (P : (ℕ × ℕ) := (5, 5))
    (n : ℕ := 10) (total_points remaining_points symmetric_points : ℕ) 
    (probability : ℚ) :
  total_points = n * n →
  remaining_points = total_points - 1 →
  symmetric_points = 4 * (n - 1) →
  probability = (symmetric_points : ℚ) / (remaining_points : ℚ) →
  probability = 32 / 99 :=
by
  sorry

end probability_symmetric_line_l59_59120


namespace total_number_of_books_l59_59667

theorem total_number_of_books (history_books geography_books math_books : ℕ)
  (h1 : history_books = 32) (h2 : geography_books = 25) (h3 : math_books = 43) :
  history_books + geography_books + math_books = 100 :=
by
  -- the proof would go here but we use sorry to skip it
  sorry

end total_number_of_books_l59_59667


namespace brooke_initial_l59_59356

variable (B : ℕ)

def brooke_balloons_initially (B : ℕ) :=
  let brooke_balloons := B + 8
  let tracy_balloons_initial := 6
  let tracy_added_balloons := 24
  let tracy_balloons := tracy_balloons_initial + tracy_added_balloons
  let tracy_popped_balloons := tracy_balloons / 2 -- Tracy having half her balloons popped.
  (brooke_balloons + tracy_popped_balloons = 35)

theorem brooke_initial (h : brooke_balloons_initially B) : B = 12 :=
  sorry

end brooke_initial_l59_59356


namespace percentage_of_masters_l59_59998

-- Definition of given conditions
def average_points_juniors := 22
def average_points_masters := 47
def overall_average_points := 41

-- Problem statement
theorem percentage_of_masters (x y : ℕ) (hx : x ≥ 0) (hy : y ≥ 0) 
    (h_avg_juniors : 22 * x = average_points_juniors * x)
    (h_avg_masters : 47 * y = average_points_masters * y)
    (h_overall_average : 22 * x + 47 * y = overall_average_points * (x + y)) : 
    (y : ℚ) / (x + y) * 100 = 76 := 
sorry

end percentage_of_masters_l59_59998


namespace smallest_possible_S_l59_59774

/-- Define the maximum possible sum for n dice --/
def max_sum (n : ℕ) : ℕ := 6 * n

/-- Define the transformation of the dice sum when each result is transformed to 7 - d_i --/
def transformed_sum (n R : ℕ) : ℕ := 7 * n - R

/-- Determine the smallest possible S under given conditions --/
theorem smallest_possible_S :
  ∃ n : ℕ, max_sum n ≥ 2001 ∧ transformed_sum n 2001 = 337 :=
by
  -- TODO: Complete the proof
  sorry

end smallest_possible_S_l59_59774


namespace milk_removal_replacement_l59_59244

theorem milk_removal_replacement (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 45) :
  (45 - x) * (45 - x) / 45 = 28.8 → x = 9 :=
by
  -- skipping the proof for now
  sorry

end milk_removal_replacement_l59_59244


namespace find_u_value_l59_59182

theorem find_u_value (h : ∃ n : ℕ, n = 2012) : ∃ u : ℕ, u = 2015 := 
by
  sorry

end find_u_value_l59_59182


namespace solve_for_x_l59_59501

theorem solve_for_x (x : ℝ) (h : 3 * x = 16 - x + 4) : x = 5 := 
by
  sorry

end solve_for_x_l59_59501


namespace mother_reaches_timothy_l59_59763

/--
Timothy leaves home for school, riding his bicycle at a rate of 6 miles per hour.
Fifteen minutes after he leaves, his mother sees Timothy's math homework lying on his bed and immediately leaves home to bring it to him.
If his mother drives at 36 miles per hour, prove that she must drive 1.8 miles to reach Timothy.
-/
theorem mother_reaches_timothy
  (timothy_speed : ℕ)
  (mother_speed : ℕ)
  (delay_minutes : ℕ)
  (distance_must_drive : ℕ)
  (h_speed_t : timothy_speed = 6)
  (h_speed_m : mother_speed = 36)
  (h_delay : delay_minutes = 15)
  (h_distance : distance_must_drive = 18 / 10 ) :
  ∃ t : ℚ, (timothy_speed * (delay_minutes / 60) + timothy_speed * t) = (mother_speed * t) := sorry

end mother_reaches_timothy_l59_59763


namespace tan_theta_l59_59518

theorem tan_theta (θ : ℝ) (x y : ℝ) (hx : x = - (Real.sqrt 3) / 2) (hy : y = 1 / 2) (h_terminal : True) : 
  Real.tan θ = - (Real.sqrt 3) / 3 :=
sorry

end tan_theta_l59_59518


namespace grazing_months_l59_59668

theorem grazing_months
    (total_rent : ℝ)
    (c_rent : ℝ)
    (a_oxen : ℕ)
    (a_months : ℕ)
    (b_oxen : ℕ)
    (c_oxen : ℕ)
    (c_months : ℕ)
    (b_months : ℝ)
    (total_oxen_months : ℝ) :
    total_rent = 140 ∧
    c_rent = 36 ∧
    a_oxen = 10 ∧
    a_months = 7 ∧
    b_oxen = 12 ∧
    c_oxen = 15 ∧
    c_months = 3 ∧
    c_rent / total_rent = (c_oxen * c_months) / total_oxen_months ∧
    total_oxen_months = (a_oxen * a_months) + (b_oxen * b_months) + (c_oxen * c_months)
    → b_months = 5 := by
    sorry

end grazing_months_l59_59668


namespace blankets_first_day_l59_59239

-- Definition of the conditions
def num_people := 15
def blankets_day_three := 22
def total_blankets := 142

-- The problem statement
theorem blankets_first_day (B : ℕ) : 
  (num_people * B) + (3 * (num_people * B)) + blankets_day_three = total_blankets → 
  B = 2 :=
by sorry

end blankets_first_day_l59_59239


namespace given_condition_l59_59850

variable (a : ℝ)

theorem given_condition
  (h1 : (a + 1/a)^2 = 5) :
  a^2 + 1/a^2 + a^3 + 1/a^3 = 3 + 2 * Real.sqrt 5 :=
sorry

end given_condition_l59_59850


namespace remaining_numbers_l59_59087

theorem remaining_numbers (S S3 S2 N : ℕ) (h1 : S / 5 = 8) (h2 : S3 / 3 = 4) (h3 : S2 / N = 14) 
(hS  : S = 5 * 8) (hS3 : S3 = 3 * 4) (hS2 : S2 = S - S3) : N = 2 := by
  sorry

end remaining_numbers_l59_59087


namespace arithmetic_sequence_ninth_term_l59_59436

theorem arithmetic_sequence_ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end arithmetic_sequence_ninth_term_l59_59436


namespace height_of_taller_tree_l59_59845

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

end height_of_taller_tree_l59_59845


namespace line_equation_cartesian_circle_equation_cartesian_l59_59264

theorem line_equation_cartesian (t : ℝ) (x y : ℝ) : 
  (x = 3 - (Real.sqrt 2 / 2) * t ∧ y = Real.sqrt 5 + (Real.sqrt 2 / 2) * t) -> 
  y = -2 * x + 6 + Real.sqrt 5 :=
sorry

theorem circle_equation_cartesian (ρ θ x y : ℝ) : 
  (ρ = 2 * Real.sqrt 5 * Real.sin θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) -> 
  x^2 = 0 :=
sorry

end line_equation_cartesian_circle_equation_cartesian_l59_59264


namespace number_of_labelings_l59_59877

-- Define the concept of a truncated chessboard with 8 squares
structure TruncatedChessboard :=
(square_labels : Fin 8 → ℕ)
(condition : ∀ i j, i ≠ j → square_labels i ≠ square_labels j)

-- Assuming a wider adjacency matrix for "connected" (has at least one common vertex)
def connected (i j : Fin 8) : Prop := sorry

-- Define the non-consecutiveness condition
def non_consecutive (board : TruncatedChessboard) :=
  ∀ i j, connected i j → (board.square_labels i ≠ board.square_labels j + 1 ∧
                          board.square_labels i ≠ board.square_labels j - 1)

-- Theorem statement
theorem number_of_labelings : ∃ c : Fin 8 → ℕ, ∀ b : TruncatedChessboard, non_consecutive b → 
  (b.square_labels = c) := sorry

end number_of_labelings_l59_59877


namespace mechanism_completion_times_l59_59426

theorem mechanism_completion_times :
  ∃ (x y : ℝ), (1 / x + 1 / y = 1 / 30) ∧ (6 * (1 / x + 1 / y) + 40 * (1 / y) = 1) ∧ x = 75 ∧ y = 50 :=
by {
  sorry
}

end mechanism_completion_times_l59_59426


namespace problem_1_problem_2_l59_59652

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
axiom h1 : ∀ n : ℕ, 2 * S n = a (n + 1) - 2^(n + 1) + 1
axiom h2 : a 2 + 5 = a 1 + (a 3 - a 2)

-- Problem 1: Prove the value of a₁
theorem problem_1 : a 1 = 1 := sorry

-- Problem 2: Find the general term formula for the sequence {aₙ}
theorem problem_2 : ∀ n : ℕ, a n = 3^n - 2^n := sorry

end problem_1_problem_2_l59_59652


namespace solve_for_x_l59_59218

theorem solve_for_x (x : ℤ) (h : 3 * x + 36 = 48) : x = 4 := by
  sorry

end solve_for_x_l59_59218


namespace plane_split_four_regions_l59_59528

theorem plane_split_four_regions :
  (∀ x y : ℝ, y = 3 * x ∨ x = 3 * y) → (exists regions : ℕ, regions = 4) :=
by
  sorry

end plane_split_four_regions_l59_59528


namespace range_a_range_b_l59_59231

def set_A : Set ℝ := {x | Real.log x / Real.log 2 > 2}
def set_B (a : ℝ) : Set ℝ := {x | x > a}
def set_C (b : ℝ) : Set ℝ := {x | b + 1 < x ∧ x < 2 * b + 1}

-- Part (1)
theorem range_a (a : ℝ) : (∀ x, x ∈ set_A → x ∈ set_B a) ↔ a ∈ Set.Iic 4 := sorry

-- Part (2)
theorem range_b (b : ℝ) : (set_A ∪ set_C b = set_A) ↔ b ∈ Set.Iic 0 ∪ Set.Ici 3 := sorry

end range_a_range_b_l59_59231


namespace arithmetic_seq_general_term_geometric_seq_general_term_sequence_sum_l59_59059

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 2^n

def seq_sum (n : ℕ) (seq : ℕ → ℕ) : ℕ :=
  (Finset.range n).sum seq

noncomputable def T_n (n : ℕ) : ℕ :=
  seq_sum n (λ i => (a_n (i + 1) + 1) * b_n (i + 1))

theorem arithmetic_seq_general_term (n : ℕ) : a_n n = 2 * n - 1 := by
  sorry

theorem geometric_seq_general_term (n : ℕ) : b_n n = 2^n := by
  sorry

theorem sequence_sum (n : ℕ) : T_n n = (n - 1) * 2^(n+2) + 4 := by
  sorry

end arithmetic_seq_general_term_geometric_seq_general_term_sequence_sum_l59_59059


namespace rectangle_area_l59_59464

theorem rectangle_area (b : ℕ) (side radius length : ℕ) 
    (h1 : side * side = 1296)
    (h2 : radius = side)
    (h3 : length = radius / 6) :
    length * b = 6 * b :=
by
  sorry

end rectangle_area_l59_59464


namespace simplify_tangent_sum_l59_59598

theorem simplify_tangent_sum :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 := by
  have h1 : Real.tan (45 * Real.pi / 180) = Real.tan ((10 + 35) * Real.pi / 180) := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h3 : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y) := by sorry
  sorry

end simplify_tangent_sum_l59_59598


namespace sum_of_solutions_eq_zero_l59_59780

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) + 4 * abs x

theorem sum_of_solutions_eq_zero : 
  (∃ x : ℝ, f x = 20) ∧ (∃ y : ℝ, f y = 20 ∧ x = -y) → 
  x + y = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l59_59780


namespace find_y_value_l59_59344

theorem find_y_value : (15^3 * 7^4) / 5670 = 1428.75 := by
  sorry

end find_y_value_l59_59344


namespace tina_made_more_140_dollars_l59_59614

def candy_bars_cost : ℕ := 2
def marvin_candy_bars : ℕ := 35
def tina_candy_bars : ℕ := 3 * marvin_candy_bars
def marvin_money : ℕ := marvin_candy_bars * candy_bars_cost
def tina_money : ℕ := tina_candy_bars * candy_bars_cost
def tina_extra_money : ℕ := tina_money - marvin_money

theorem tina_made_more_140_dollars :
  tina_extra_money = 140 := by
  sorry

end tina_made_more_140_dollars_l59_59614


namespace steven_has_19_peaches_l59_59266

-- Conditions
def jill_peaches : ℕ := 6
def steven_peaches : ℕ := jill_peaches + 13

-- Statement to prove
theorem steven_has_19_peaches : steven_peaches = 19 :=
by {
    -- Proof steps would go here
    sorry
}

end steven_has_19_peaches_l59_59266


namespace no_non_integer_point_exists_l59_59265

variable (b0 b1 b2 b3 b4 b5 u v : ℝ)

def q (x y : ℝ) : ℝ := b0 + b1 * x + b2 * y + b3 * x^2 + b4 * x * y + b5 * y^2

theorem no_non_integer_point_exists
    (h₀ : q b0 b1 b2 b3 b4 b5 0 0 = 0)
    (h₁ : q b0 b1 b2 b3 b4 b5 1 0 = 0)
    (h₂ : q b0 b1 b2 b3 b4 b5 (-1) 0 = 0)
    (h₃ : q b0 b1 b2 b3 b4 b5 0 1 = 0)
    (h₄ : q b0 b1 b2 b3 b4 b5 0 (-1) = 0)
    (h₅ : q b0 b1 b2 b3 b4 b5 1 1 = 0) :
  ∀ u v : ℝ, (¬ ∃ (n m : ℤ), u = n ∧ v = m) → q b0 b1 b2 b3 b4 b5 u v ≠ 0 :=
by
  sorry

end no_non_integer_point_exists_l59_59265


namespace machines_produce_x_units_l59_59597

variable (x : ℕ) (d : ℕ)

-- Define the conditions
def four_machines_produce_in_d_days (x : ℕ) (d : ℕ) : Prop := 
  4 * (x / d) = x / d

def twelve_machines_produce_three_x_in_d_days (x : ℕ) (d : ℕ) : Prop := 
  12 * (x / d) = 3 * (x / d)

-- Given the conditions, prove the number of days for 4 machines to produce x units
theorem machines_produce_x_units (x : ℕ) (d : ℕ) 
  (H1 : four_machines_produce_in_d_days x d)
  (H2 : twelve_machines_produce_three_x_in_d_days x d) : 
  x / d = x / d := 
by 
  sorry

end machines_produce_x_units_l59_59597


namespace waynes_son_time_to_shovel_l59_59401

-- Definitions based on the conditions
variables (S W : ℝ) (son_rate : S = 1 / 21) (wayne_rate : W = 6 * S) (together_rate : 3 * (S + W) = 1)

theorem waynes_son_time_to_shovel : 
  1 / S = 21 :=
by
  -- Proof will be provided later
  sorry

end waynes_son_time_to_shovel_l59_59401


namespace full_day_students_l59_59698

def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

theorem full_day_students : 
  (total_students - (total_students * percentage_half_day_students / 100)) = 60 := by
  sorry

end full_day_students_l59_59698


namespace family_ate_doughnuts_l59_59837

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

end family_ate_doughnuts_l59_59837


namespace compute_sum_of_squares_roots_l59_59124

-- p, q, and r are roots of 3*x^3 - 2*x^2 + 6*x + 15 = 0.
def P (x : ℝ) : Prop := 3*x^3 - 2*x^2 + 6*x + 15 = 0

theorem compute_sum_of_squares_roots :
  ∀ p q r : ℝ, P p ∧ P q ∧ P r → p^2 + q^2 + r^2 = -32 / 9 :=
by
  intros p q r h
  sorry

end compute_sum_of_squares_roots_l59_59124


namespace exists_nat_with_digit_sum_l59_59642

-- Definitions of the necessary functions
def digit_sum (n : ℕ) : ℕ := sorry -- Assume this is the sum of the digits of n

theorem exists_nat_with_digit_sum :
  ∃ n : ℕ, digit_sum n = 1000 ∧ digit_sum (n^2) = 1000000 :=
by
  sorry

end exists_nat_with_digit_sum_l59_59642


namespace jogging_track_circumference_l59_59152

def speed_Suresh_km_hr : ℝ := 4.5
def speed_wife_km_hr : ℝ := 3.75
def meet_time_min : ℝ := 5.28

theorem jogging_track_circumference : 
  let speed_Suresh_km_min := speed_Suresh_km_hr / 60
  let speed_wife_km_min := speed_wife_km_hr / 60
  let distance_Suresh_km := speed_Suresh_km_min * meet_time_min
  let distance_wife_km := speed_wife_km_min * meet_time_min
  let total_distance_km := distance_Suresh_km + distance_wife_km
  total_distance_km = 0.726 :=
by sorry

end jogging_track_circumference_l59_59152


namespace monotonicity_and_zeros_l59_59547

open Real

noncomputable def f (x k : ℝ) : ℝ := exp x - k * x + k

theorem monotonicity_and_zeros
  (k : ℝ)
  (h₁ : k > exp 2)
  (x₁ x₂ : ℝ)
  (h₂ : f x₁ k = 0)
  (h₃ : f x₂ k = 0)
  (h₄ : x₁ ≠ x₂) :
  x₁ + x₂ > 4 := 
sorry

end monotonicity_and_zeros_l59_59547


namespace james_found_bills_l59_59857

def initial_money : ℝ := 75
def final_money : ℝ := 135
def bill_value : ℝ := 20

theorem james_found_bills :
  (final_money - initial_money) / bill_value = 3 :=
by
  sorry

end james_found_bills_l59_59857


namespace James_future_age_when_Thomas_reaches_James_current_age_l59_59611

-- Defining the given conditions
def Thomas_age := 6
def Shay_age := Thomas_age + 13
def James_age := Shay_age + 5

-- Goal: Proving James's age when Thomas reaches James's current age
theorem James_future_age_when_Thomas_reaches_James_current_age :
  let years_until_Thomas_is_James_current_age := James_age - Thomas_age
  let James_future_age := James_age + years_until_Thomas_is_James_current_age
  James_future_age = 42 :=
by
  sorry

end James_future_age_when_Thomas_reaches_James_current_age_l59_59611


namespace min_distance_to_water_all_trees_l59_59116

/-- Proof that the minimum distance Xiao Zhang must walk to water all 10 trees is 410 meters -/
def minimum_distance_to_water_trees (num_trees : ℕ) (distance_between_trees : ℕ) : ℕ := 
  (sorry) -- implementation to calculate the minimum distance

theorem min_distance_to_water_all_trees (num_trees distance_between_trees : ℕ) :
  num_trees = 10 → 
  distance_between_trees = 10 →
  minimum_distance_to_water_trees num_trees distance_between_trees = 410 :=
by
  intros h_num_trees h_distance_between_trees
  rw [h_num_trees, h_distance_between_trees]
  -- Add proof here that the distance is 410
  sorry

end min_distance_to_water_all_trees_l59_59116


namespace baking_completion_time_l59_59945

theorem baking_completion_time (start_time : ℕ) (partial_bake_time : ℕ) (fraction_baked : ℕ) :
  start_time = 9 → partial_bake_time = 3 → fraction_baked = 4 →
  (start_time + (partial_bake_time * fraction_baked)) = 21 :=
by
  intros h_start h_partial h_fraction
  sorry

end baking_completion_time_l59_59945


namespace greatest_divisor_l59_59076

theorem greatest_divisor (d : ℕ) (h1 : 4351 % d = 8) (h2 : 5161 % d = 10) : d = 1 :=
by
  -- Proof goes here
  sorry

end greatest_divisor_l59_59076


namespace num_play_both_l59_59613

-- Definitions based on the conditions
def total_members : ℕ := 30
def play_badminton : ℕ := 17
def play_tennis : ℕ := 19
def play_neither : ℕ := 2

-- The statement we want to prove
theorem num_play_both :
  play_badminton + play_tennis - 8 = total_members - play_neither := by
  -- Omitted proof
  sorry

end num_play_both_l59_59613


namespace sets_equal_l59_59740

def M := { u | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l }
def N := { u | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r }

theorem sets_equal : M = N :=
by sorry

end sets_equal_l59_59740


namespace total_patients_in_a_year_l59_59040

-- Define conditions from the problem
def patients_per_day_first : ℕ := 20
def percent_increase_second : ℕ := 20
def working_days_per_week : ℕ := 5
def working_weeks_per_year : ℕ := 50

-- Lean statement for the problem
theorem total_patients_in_a_year (patients_per_day_first : ℕ) (percent_increase_second : ℕ) (working_days_per_week : ℕ) (working_weeks_per_year : ℕ) :
  (patients_per_day_first + ((patients_per_day_first * percent_increase_second) / 100)) * working_days_per_week * working_weeks_per_year = 11000 :=
by
  sorry

end total_patients_in_a_year_l59_59040


namespace profit_no_discount_l59_59942

theorem profit_no_discount (CP SP ASP : ℝ) (discount profit : ℝ) (h1 : discount = 4 / 100) (h2 : profit = 38 / 100) (h3 : SP = CP + CP * profit) (h4 : ASP = SP - SP * discount) :
  ((SP - CP) / CP) * 100 = 38 :=
by
  sorry

end profit_no_discount_l59_59942


namespace arithmetic_geometric_seq_l59_59563

theorem arithmetic_geometric_seq (a : ℕ → ℝ) (d a_1 : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_nonzero : d ≠ 0) (h_geom : (a 0, a 1, a 4) = (a_1, a_1 + d, a_1 + 4 * d) ∧ (a 1)^2 = a 0 * a 4)
  (h_sum : a 0 + a 1 + a 4 > 13) : a_1 > 1 :=
by sorry

end arithmetic_geometric_seq_l59_59563


namespace earliest_year_exceeds_target_l59_59342

/-- Define the initial deposit and annual interest rate -/
def initial_deposit : ℝ := 100000
def annual_interest_rate : ℝ := 0.10

/-- Define the amount in the account after n years -/
def amount_after_years (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

/-- Define the target amount to exceed -/
def target_amount : ℝ := 150100

/-- Define the year the initial deposit is made -/
def initial_year : ℕ := 2021

/-- Prove that the earliest year the amount exceeds the target is 2026 -/
theorem earliest_year_exceeds_target :
  ∃ n : ℕ, n > 0 ∧ amount_after_years initial_deposit annual_interest_rate n > target_amount ∧ (initial_year + n) = 2026 :=
by
  sorry

end earliest_year_exceeds_target_l59_59342


namespace barrel_to_cask_ratio_l59_59313

theorem barrel_to_cask_ratio
  (k : ℕ) -- k is the multiple
  (B C : ℕ) -- B is the amount a barrel can store, C is the amount a cask can store
  (h1 : C = 20) -- C stores 20 gallons
  (h2 : B = k * C + 3) -- A barrel stores 3 gallons more than k times the amount a cask stores
  (h3 : 4 * B + C = 172) -- The total storage capacity is 172 gallons
  : B / C = 19 / 10 :=
sorry

end barrel_to_cask_ratio_l59_59313


namespace area_of_hall_l59_59129

-- Define the conditions
def length := 25
def breadth := length - 5

-- Define the area calculation
def area := length * breadth

-- The statement to prove
theorem area_of_hall : area = 500 :=
by
  sorry

end area_of_hall_l59_59129


namespace problem_statement_l59_59020

open BigOperators

-- Defining the arithmetic sequence
def a (n : ℕ) : ℕ := n - 1

-- Defining the sequence b_n
def b (n : ℕ) : ℕ :=
if n % 2 = 1 then
  a n + 1
else
  2 ^ a n

-- Defining T_2n as the sum of the first 2n terms of b
def T (n : ℕ) : ℕ :=
(∑ i in Finset.range n, b (2 * i + 1)) +
(∑ i in Finset.range n, b (2 * i + 2))

-- The theorem to be proven
theorem problem_statement (n : ℕ) : 
  a 2 * (a 4 + 1) = a 3 ^ 2 ∧
  T n = n^2 + (2^(2*n+1) - 2) / 3 :=
by
  sorry

end problem_statement_l59_59020


namespace solve_squares_and_circles_l59_59745

theorem solve_squares_and_circles (x y : ℝ) :
  (5 * x + 2 * y = 39) ∧ (3 * x + 3 * y = 27) → (x = 7) ∧ (y = 2) :=
by
  intro h
  sorry

end solve_squares_and_circles_l59_59745


namespace value_of_trig_expr_l59_59025

theorem value_of_trig_expr : 2 * Real.cos (Real.pi / 12) ^ 2 + 1 = 2 + Real.sqrt 3 / 2 :=
by
  sorry

end value_of_trig_expr_l59_59025


namespace expression_simplification_l59_59691

theorem expression_simplification :
  (2 ^ 2 / 3 + (-(3 ^ 2) + 5) + (-(3) ^ 2) * ((2 / 3) ^ 2)) = 4 / 3 :=
sorry

end expression_simplification_l59_59691


namespace Mr_Pendearly_optimal_speed_l59_59171

noncomputable def optimal_speed (d t : ℝ) : ℝ := d / t

theorem Mr_Pendearly_optimal_speed :
  ∀ (d t : ℝ),
  (d = 45 * (t + 1/15)) →
  (d = 75 * (t - 1/15)) →
  optimal_speed d t = 56.25 :=
by
  intros d t h1 h2
  have h_d_eq_45 := h1
  have h_d_eq_75 := h2
  sorry

end Mr_Pendearly_optimal_speed_l59_59171


namespace find_flights_of_stairs_l59_59822

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

end find_flights_of_stairs_l59_59822


namespace multiplication_as_sum_of_squares_l59_59715

theorem multiplication_as_sum_of_squares :
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 := by
  sorry

end multiplication_as_sum_of_squares_l59_59715


namespace roots_expression_eval_l59_59574

theorem roots_expression_eval (p q r : ℝ) 
  (h1 : p + q + r = 2)
  (h2 : p * q + q * r + r * p = -1)
  (h3 : p * q * r = -2)
  (hp : p^3 - 2 * p^2 - p + 2 = 0)
  (hq : q^3 - 2 * q^2 - q + 2 = 0)
  (hr : r^3 - 2 * r^2 - r + 2 = 0) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = 16 :=
sorry

end roots_expression_eval_l59_59574


namespace largest_among_four_theorem_l59_59512

noncomputable def largest_among_four (a b : ℝ) (h1 : 0 < a ∧ a < b) (h2 : a + b = 1) : Prop :=
  (a^2 + b^2 > 1) ∧ (a^2 + b^2 > 2 * a * b) ∧ (a^2 + b^2 > a)

theorem largest_among_four_theorem (a b : ℝ) (h1 : 0 < a ∧ a < b) (h2 : a + b = 1) :
  largest_among_four a b h1 h2 :=
sorry

end largest_among_four_theorem_l59_59512


namespace transaction_result_l59_59079

theorem transaction_result
  (house_selling_price store_selling_price : ℝ)
  (house_loss_perc : ℝ)
  (store_gain_perc : ℝ)
  (house_selling_price_eq : house_selling_price = 15000)
  (store_selling_price_eq : store_selling_price = 15000)
  (house_loss_perc_eq : house_loss_perc = 0.1)
  (store_gain_perc_eq : store_gain_perc = 0.3) :
  (store_selling_price + house_selling_price - ((house_selling_price / (1 - house_loss_perc)) + (store_selling_price / (1 + store_gain_perc)))) = 1795 :=
by
  sorry

end transaction_result_l59_59079


namespace part1_part2_l59_59992

def set_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def set_B (a : ℝ) := {x : ℝ | (x - a) * (x - a - 1) < 0}

theorem part1 (a : ℝ) : (1 ∈ set_B a) → 0 < a ∧ a < 1 := by
  sorry

theorem part2 (a : ℝ) : (∀ x, x ∈ set_B a → x ∈ set_A) ∧ (∃ x, x ∉ set_B a ∧ x ∈ set_A) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l59_59992


namespace complex_number_in_first_quadrant_l59_59804

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

end complex_number_in_first_quadrant_l59_59804


namespace container_capacity_l59_59378

variable (C : ℝ)
variable (h1 : 0.30 * C + 27 = (3/4) * C)

theorem container_capacity : C = 60 := by
  sorry

end container_capacity_l59_59378


namespace find_m_l59_59638

variable {α : Type*} [DecidableEq α]

-- Definitions and conditions
def A (m : ℤ) : Set ℤ := {-1, 3, m ^ 2}
def B : Set ℤ := {3, 4}

theorem find_m (m : ℤ) (h : B ⊆ A m) : m = 2 ∨ m = -2 := by
  sorry

end find_m_l59_59638


namespace find_profit_range_l59_59846

noncomputable def profit_range (x : ℝ) : Prop :=
  0 < x → 0.15 * (1 + 0.25 * x) * (100000 - x) ≥ 0.15 * 100000

theorem find_profit_range (x : ℝ) : profit_range x → 0 < x ∧ x ≤ 6 :=
by
  sorry

end find_profit_range_l59_59846


namespace translated_vector_ab_l59_59045

-- Define points A and B, and vector a
def A : ℝ × ℝ := (3, 7)
def B : ℝ × ℝ := (5, 2)
def a : ℝ × ℝ := (1, 2)

-- Define the vector AB
def vectorAB : ℝ × ℝ :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  (Bx - Ax, By - Ay)

-- Prove that after translating vector AB by vector a, the result remains (2, -5)
theorem translated_vector_ab :
  vectorAB = (2, -5) := by
  sorry

end translated_vector_ab_l59_59045


namespace total_income_per_minute_l59_59207

theorem total_income_per_minute :
  let black_shirt_price := 30
  let black_shirt_quantity := 250
  let white_shirt_price := 25
  let white_shirt_quantity := 200
  let red_shirt_price := 28
  let red_shirt_quantity := 100
  let blue_shirt_price := 25
  let blue_shirt_quantity := 50

  let black_discount := 0.05
  let white_discount := 0.08
  let red_discount := 0.10

  let total_black_income_before_discount := black_shirt_quantity * black_shirt_price
  let total_white_income_before_discount := white_shirt_quantity * white_shirt_price
  let total_red_income_before_discount := red_shirt_quantity * red_shirt_price
  let total_blue_income_before_discount := blue_shirt_quantity * blue_shirt_price

  let total_income_before_discount :=
    total_black_income_before_discount + total_white_income_before_discount + total_red_income_before_discount + total_blue_income_before_discount

  let total_black_discount := black_discount * total_black_income_before_discount
  let total_white_discount := white_discount * total_white_income_before_discount
  let total_red_discount := red_discount * total_red_income_before_discount

  let total_discount :=
    total_black_discount + total_white_discount + total_red_discount

  let total_income_after_discount :=
    total_income_before_discount - total_discount

  let total_minutes := 40
  let total_income_per_minute := total_income_after_discount / total_minutes

  total_income_per_minute = 387.38 := by
  sorry

end total_income_per_minute_l59_59207


namespace largest_common_number_in_sequences_from_1_to_200_l59_59693

theorem largest_common_number_in_sequences_from_1_to_200 :
  ∃ a, a ≤ 200 ∧ a % 8 = 3 ∧ a % 9 = 5 ∧ ∀ b, (b ≤ 200 ∧ b % 8 = 3 ∧ b % 9 = 5) → b ≤ a :=
sorry

end largest_common_number_in_sequences_from_1_to_200_l59_59693


namespace find_n_l59_59855

theorem find_n (n : ℤ) (h : 8 + 6 = n + 8) : n = 6 :=
by
  sorry

end find_n_l59_59855


namespace subtract_complex_eq_l59_59786

noncomputable def subtract_complex (a b : ℂ) : ℂ := a - b

theorem subtract_complex_eq (i : ℂ) (h_i : i^2 = -1) :
  subtract_complex (5 - 3 * i) (7 - 7 * i) = -2 + 4 * i :=
by
  sorry

end subtract_complex_eq_l59_59786


namespace target_hit_prob_l59_59994

-- Probability definitions for A, B, and C
def prob_A := 1 / 2
def prob_B := 1 / 3
def prob_C := 1 / 4

-- Theorem to prove the probability of the target being hit
theorem target_hit_prob :
  (1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)) = 3 / 4 :=
by
  sorry

end target_hit_prob_l59_59994


namespace max_rooks_in_cube_l59_59766

def non_attacking_rooks (n : ℕ) (cube : ℕ × ℕ × ℕ) : ℕ :=
  if cube = (8, 8, 8) then 64 else 0

theorem max_rooks_in_cube:
  non_attacking_rooks 64 (8, 8, 8) = 64 :=
by
  -- proof by logical steps matching the provided solution, if necessary, start with sorry for placeholder
  sorry

end max_rooks_in_cube_l59_59766


namespace part_a_part_b_l59_59460

variable {α β γ δ AB CD : ℝ}
variable {A B C D : Point}
variable {A_obtuse B_obtuse : Prop}
variable {α_gt_δ β_gt_γ : Prop}

-- Definition of a convex quadrilateral
def convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Conditions for part (a)
axiom angle_A_obtuse : A_obtuse
axiom angle_B_obtuse : B_obtuse

-- Conditions for part (b)
axiom angle_α_gt_δ : α_gt_δ
axiom angle_β_gt_γ : β_gt_γ

-- Part (a) statement: Given angles A and B are obtuse, AB ≤ CD
theorem part_a {A B C D : Point} (h_convex : convex_quadrilateral A B C D) 
    (h_A_obtuse : A_obtuse) (h_B_obtuse : B_obtuse) : AB ≤ CD :=
sorry

-- Part (b) statement: Given angle A > angle D and angle B > angle C, AB < CD
theorem part_b {A B C D : Point} (h_convex : convex_quadrilateral A B C D) 
    (h_angle_α_gt_δ : α_gt_δ) (h_angle_β_gt_γ : β_gt_γ) : AB < CD :=
sorry

end part_a_part_b_l59_59460


namespace whales_last_year_eq_4000_l59_59610

variable (W : ℕ) (last_year this_year next_year : ℕ)

theorem whales_last_year_eq_4000
    (h1 : this_year = 2 * last_year)
    (h2 : next_year = this_year + 800)
    (h3 : next_year = 8800) :
    last_year = 4000 := by
  sorry

end whales_last_year_eq_4000_l59_59610


namespace find_roots_l59_59813

theorem find_roots : 
  (∃ x : ℝ, (x-1) * (x-2) * (x+1) * (x-5) = 0) ↔ 
  x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 5 :=
by sorry

end find_roots_l59_59813


namespace find_third_polygon_sides_l59_59536

def interior_angle (n : ℕ) : ℚ :=
  (n - 2) * 180 / n

theorem find_third_polygon_sides :
  let square_angle := interior_angle 4
  let pentagon_angle := interior_angle 5
  let third_polygon_angle := 360 - (square_angle + pentagon_angle)
  ∃ (m : ℕ), interior_angle m = third_polygon_angle ∧ m = 20 :=
by
  let square_angle := interior_angle 4
  let pentagon_angle := interior_angle 5
  let third_polygon_angle := 360 - (square_angle + pentagon_angle)
  use 20
  sorry

end find_third_polygon_sides_l59_59536


namespace time_to_fill_pot_l59_59625

def pot_volume : ℕ := 3000  -- in ml
def rate_of_entry : ℕ := 60 -- in ml/minute

-- Statement: Prove that the time required for the pot to be full is 50 minutes.
theorem time_to_fill_pot : (pot_volume / rate_of_entry) = 50 := by
  sorry

end time_to_fill_pot_l59_59625


namespace numeric_puzzle_AB_eq_B_pow_V_l59_59216

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l59_59216


namespace n_square_divisible_by_144_l59_59385

theorem n_square_divisible_by_144 (n : ℤ) (hn : n > 0)
  (hw : ∃ k : ℤ, n = 12 * k) : ∃ m : ℤ, n^2 = 144 * m :=
by {
  sorry
}

end n_square_divisible_by_144_l59_59385


namespace num_integers_n_with_properties_l59_59915

theorem num_integers_n_with_properties :
  ∃ (N : Finset ℕ), N.card = 50 ∧
  ∀ n ∈ N, n < 150 ∧
    ∃ (m : ℕ), (∃ k, n = 2*k + 1 ∧ m = k*(k+1)) ∧ ¬ (3 ∣ m) :=
sorry

end num_integers_n_with_properties_l59_59915


namespace final_weight_of_box_l59_59553

theorem final_weight_of_box (w1 w2 w3 w4 : ℝ) (h1 : w1 = 2) (h2 : w2 = 3 * w1) (h3 : w3 = w2 + 2) (h4 : w4 = 2 * w3) : w4 = 16 :=
by
  sorry

end final_weight_of_box_l59_59553


namespace black_king_eventually_in_check_l59_59173

theorem black_king_eventually_in_check 
  (n : ℕ) (h1 : n = 1000) (r : ℕ) (h2 : r = 499)
  (rooks : Fin r → (ℕ × ℕ)) (king : ℕ × ℕ)
  (take_not_allowed : ∀ rk : Fin r, rooks rk ≠ king) :
  ∃ m : ℕ, m ≤ 1000 ∧ (∃ t : Fin r, rooks t = king) :=
by
  sorry

end black_king_eventually_in_check_l59_59173


namespace impossible_configuration_l59_59827

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

end impossible_configuration_l59_59827


namespace eggs_per_snake_l59_59522

-- Define the conditions
def num_snakes : ℕ := 3
def price_regular : ℕ := 250
def price_super_rare : ℕ := 1000
def total_revenue : ℕ := 2250

-- Prove for the number of eggs each snake lays
theorem eggs_per_snake (E : ℕ) 
  (h1 : E * (num_snakes - 1) * price_regular + E * price_super_rare = total_revenue) : 
  E = 2 :=
sorry

end eggs_per_snake_l59_59522


namespace isosceles_triangle_l59_59281

def triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) : Prop :=
  a^2 + b^2 - (a * Real.cos B + b * Real.cos A)^2 = 2 * a * b * Real.cos B → (B = C)

theorem isosceles_triangle (a b c A B C : ℝ) (h : a^2 + b^2 - (a * Real.cos B + b * Real.cos A)^2 = 2 * a * b * Real.cos B) : B = C :=
  sorry

end isosceles_triangle_l59_59281


namespace triangle_ab_length_triangle_roots_quadratic_l59_59632

open Real

noncomputable def right_angled_triangle_length_ab (p s : ℝ) : ℝ :=
  (p / 2) - sqrt ((p / 2)^2 - 2 * s)

noncomputable def right_angled_triangle_quadratic (p s : ℝ) : Polynomial ℝ :=
  Polynomial.X^2 - Polynomial.C ((p / 2) + sqrt ((p / 2)^2 - 2 * s)) * Polynomial.X
    + Polynomial.C (2 * s)

theorem triangle_ab_length (p s : ℝ) :
  ∃ (AB : ℝ), AB = right_angled_triangle_length_ab p s ∧
    ∃ (AC BC : ℝ), (AC + BC + AB = p) ∧ (1 / 2 * BC * AC = s) :=
by
  use right_angled_triangle_length_ab p s
  sorry

theorem triangle_roots_quadratic (p s : ℝ) :
  ∃ (AC BC : ℝ), AC + BC = (p / 2) + sqrt ((p / 2)^2 - 2 * s) ∧
    AC * BC = 2 * s ∧
    (Polynomial.aeval AC (right_angled_triangle_quadratic p s) = 0) ∧
    (Polynomial.aeval BC (right_angled_triangle_quadratic p s) = 0) :=
by
  sorry

end triangle_ab_length_triangle_roots_quadratic_l59_59632


namespace ray_inequality_l59_59533

theorem ray_inequality (a : ℝ) :
  (∀ x : ℝ, x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3 ≥ 0 ↔ x ≥ 1)
  ∨ (∀ x : ℝ, x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3 ≥ 0 ↔ x ≥ -1) :=
sorry

end ray_inequality_l59_59533


namespace cylindrical_pipe_height_l59_59525

theorem cylindrical_pipe_height (r_outer r_inner : ℝ) (SA : ℝ) (h : ℝ) 
  (h_outer : r_outer = 5)
  (h_inner : r_inner = 3)
  (h_SA : SA = 50 * Real.pi)
  (surface_area_eq: SA = 2 * Real.pi * (r_outer + r_inner) * h) : 
  h = 25 / 8 := 
by
  {
    sorry
  }

end cylindrical_pipe_height_l59_59525


namespace divisor_five_l59_59902

theorem divisor_five {D : ℝ} (h : 95 / D + 23 = 42) : D = 5 := by
  sorry

end divisor_five_l59_59902


namespace max_teams_participation_l59_59900

theorem max_teams_participation (n : ℕ) (H : 9 * n * (n - 1) / 2 ≤ 200) : n ≤ 7 := by
  -- Proof to be filled in
  sorry

end max_teams_participation_l59_59900


namespace combined_loss_l59_59360

variable (initial : ℕ) (donation : ℕ) (prize : ℕ) (final : ℕ) (lottery_winning : ℕ) (X : ℕ)

theorem combined_loss (h1 : initial = 10) (h2 : donation = 4) (h3 : prize = 90) 
                      (h4 : final = 94) (h5 : lottery_winning = 65) :
                      (initial - donation + prize - X + lottery_winning = final) ↔ (X = 67) :=
by
  -- proof steps will go here
  sorry

end combined_loss_l59_59360


namespace fraction_eq_four_l59_59585

theorem fraction_eq_four (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 3 * b = 2 * a) : 
  (2 * a + b) / b = 4 := 
by 
  sorry

end fraction_eq_four_l59_59585


namespace RiverJoe_popcorn_shrimp_price_l59_59212

theorem RiverJoe_popcorn_shrimp_price
  (price_catfish : ℝ)
  (total_orders : ℕ)
  (total_revenue : ℝ)
  (orders_popcorn_shrimp : ℕ)
  (catfish_revenue : ℝ)
  (popcorn_shrimp_price : ℝ) :
  price_catfish = 6.00 →
  total_orders = 26 →
  total_revenue = 133.50 →
  orders_popcorn_shrimp = 9 →
  catfish_revenue = (total_orders - orders_popcorn_shrimp) * price_catfish →
  catfish_revenue + orders_popcorn_shrimp * popcorn_shrimp_price = total_revenue →
  popcorn_shrimp_price = 3.50 :=
by
  intros price_catfish_eq total_orders_eq total_revenue_eq orders_popcorn_shrimp_eq catfish_revenue_eq revenue_eq
  sorry

end RiverJoe_popcorn_shrimp_price_l59_59212


namespace cricket_innings_l59_59256

theorem cricket_innings (n : ℕ) (h1 : (36 * n) / n = 36) (h2 : (36 * n + 80) / (n + 1) = 40) : n = 10 := by
  -- The proof goes here
  sorry

end cricket_innings_l59_59256


namespace smaller_of_two_numbers_l59_59007

theorem smaller_of_two_numbers 
  (a b d : ℝ) (h : 0 < a ∧ a < b) (u v : ℝ) 
  (huv : u / v = b / a) (sum_uv : u + v = d) : 
  min u v = (a * d) / (a + b) :=
by
  sorry

end smaller_of_two_numbers_l59_59007


namespace area_is_12_l59_59476

-- Definitions based on conditions
def isosceles_triangle (a b m : ℝ) : Prop :=
  a = b ∧ m > 0 ∧ a > 0

def median (height base_length : ℝ) : Prop :=
  height > 0 ∧ base_length > 0

noncomputable def area_of_isosceles_triangle_with_given_median (a m : ℝ) : ℝ :=
  let base_half := Real.sqrt (a^2 - m^2)
  let base := 2 * base_half
  (1 / 2) * base * m

-- Prove that the area of the isosceles triangle is correct given conditions
theorem area_is_12 :
  ∀ (a m : ℝ), isosceles_triangle a a m → median m (2 * Real.sqrt (a^2 - m^2)) → area_of_isosceles_triangle_with_given_median a m = 12 := 
by
  intros a m hiso hmed
  sorry  -- Proof steps are omitted

end area_is_12_l59_59476


namespace perimeter_triangle_PQR_is_24_l59_59033

noncomputable def perimeter_triangle_PQR (QR PR : ℝ) : ℝ :=
  let PQ := Real.sqrt (QR^2 + PR^2)
  PQ + QR + PR

theorem perimeter_triangle_PQR_is_24 :
  perimeter_triangle_PQR 8 6 = 24 := by
  sorry

end perimeter_triangle_PQR_is_24_l59_59033


namespace find_integer_of_divisors_l59_59990

theorem find_integer_of_divisors:
  ∃ (N : ℕ), (∀ (l m n : ℕ), N = (2^l) * (3^m) * (5^n) → 
  (2^120) * (3^60) * (5^90) = (2^l * 3^m * 5^n)^( ((l+1)*(m+1)*(n+1)) / 2 ) ) → 
  N = 18000 :=
sorry

end find_integer_of_divisors_l59_59990


namespace pages_read_per_day_l59_59485

-- Define the total number of pages in the book
def total_pages := 96

-- Define the number of days it took to finish the book
def number_of_days := 12

-- Define pages read per day for Charles
def pages_per_day := total_pages / number_of_days

-- Prove that the number of pages read per day is equal to 8
theorem pages_read_per_day : pages_per_day = 8 :=
by
  sorry

end pages_read_per_day_l59_59485


namespace five_fridays_in_september_l59_59805

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

end five_fridays_in_september_l59_59805


namespace days_y_worked_l59_59017

theorem days_y_worked 
  (W : ℝ) 
  (x_days : ℝ) (h1 : x_days = 36)
  (y_days : ℝ) (h2 : y_days = 24)
  (x_remaining_days : ℝ) (h3 : x_remaining_days = 18)
  (d : ℝ) :
  d * (W / y_days) + x_remaining_days * (W / x_days) = W → d = 12 :=
by
  -- Mathematical proof goes here
  sorry

end days_y_worked_l59_59017


namespace total_spent_l59_59338

theorem total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ)
  (h1 : deck_price = 8)
  (h2 : victor_decks = 6)
  (h3 : friend_decks = 2) :
  deck_price * victor_decks + deck_price * friend_decks = 64 :=
by
  sorry

end total_spent_l59_59338


namespace seats_in_hall_l59_59583

theorem seats_in_hall (S : ℝ) (h1 : 0.50 * S = 300) : S = 600 :=
by
  sorry

end seats_in_hall_l59_59583


namespace maria_nickels_l59_59629

theorem maria_nickels (dimes quarters_initial quarters_additional : ℕ) (total_amount : ℚ) 
  (Hd : dimes = 4) (Hqi : quarters_initial = 4) (Hqa : quarters_additional = 5) (Htotal : total_amount = 3) : 
  (dimes * 0.10 + quarters_initial * 0.25 + quarters_additional * 0.25 + n/20) = total_amount → n = 7 :=
  sorry

end maria_nickels_l59_59629


namespace tangent_line_parallel_curve_l59_59014

def curve (x : ℝ) : ℝ := x^4

def line_parallel_to_curve (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x0 y0 : ℝ, l x0 y0 ∧ curve x0 = y0 ∧ ∀ (x : ℝ), l x (curve x)

theorem tangent_line_parallel_curve :
  ∃ (l : ℝ → ℝ → Prop), line_parallel_to_curve l ∧ ∀ x y, l x y ↔ 8 * x + 16 * y + 3 = 0 :=
by
  sorry

end tangent_line_parallel_curve_l59_59014


namespace length_of_BC_l59_59873

theorem length_of_BC (x : ℝ) (h1 : (20 * x^2) / 3 - (400 * x) / 3 = 140) :
  ∃ (BC : ℝ), BC = 29 := 
by
  sorry

end length_of_BC_l59_59873


namespace abs_neg_one_third_l59_59060

theorem abs_neg_one_third : abs (-1/3) = 1/3 := by
  sorry

end abs_neg_one_third_l59_59060


namespace abs_add_three_eq_two_l59_59617

theorem abs_add_three_eq_two (a : ℝ) (h : a = -1) : |a + 3| = 2 :=
by
  rw [h]
  sorry

end abs_add_three_eq_two_l59_59617


namespace blue_tshirts_in_pack_l59_59229

theorem blue_tshirts_in_pack
  (packs_white : ℕ := 2) 
  (white_per_pack : ℕ := 5) 
  (packs_blue : ℕ := 4)
  (cost_per_tshirt : ℕ := 3)
  (total_cost : ℕ := 66)
  (B : ℕ := 3) :
  (packs_white * white_per_pack * cost_per_tshirt) + (packs_blue * B * cost_per_tshirt) = total_cost := 
by
  sorry

end blue_tshirts_in_pack_l59_59229


namespace garden_length_l59_59071

theorem garden_length (P : ℕ) (breadth : ℕ) (length : ℕ) 
  (h1 : P = 600) (h2 : breadth = 95) (h3 : P = 2 * (length + breadth)) : 
  length = 205 :=
by
  sorry

end garden_length_l59_59071


namespace value_of_b_minus_d_squared_l59_59169

theorem value_of_b_minus_d_squared (a b c d : ℤ) 
  (h1 : a - b - c + d = 18) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := 
by 
  sorry

end value_of_b_minus_d_squared_l59_59169


namespace find_lost_card_number_l59_59686

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l59_59686


namespace balls_removal_l59_59812

theorem balls_removal (total_balls : ℕ) (percent_green initial_green initial_yellow remaining_percent : ℝ)
    (h_percent_green : percent_green = 0.7)
    (h_total_balls : total_balls = 600)
    (h_initial_green : initial_green = percent_green * total_balls)
    (h_initial_yellow : initial_yellow = total_balls - initial_green)
    (h_remaining_percent : remaining_percent = 0.6) :
    ∃ x : ℝ, (initial_green - x) / (total_balls - x) = remaining_percent ∧ x = 150 := 
by 
  sorry

end balls_removal_l59_59812


namespace longest_boat_length_l59_59368

theorem longest_boat_length (a : ℝ) (c : ℝ) 
  (parallel_banks : ∀ x y : ℝ, (x = y) ∨ (x = -y)) 
  (right_angle_bend : ∃ b : ℝ, b = a) :
  c = 2 * a * Real.sqrt 2 := by
  sorry

end longest_boat_length_l59_59368


namespace num_pos_integers_congruent_to_4_mod_7_l59_59080

theorem num_pos_integers_congruent_to_4_mod_7 (n : ℕ) (h1 : n < 500) (h2 : ∃ k : ℕ, n = 7 * k + 4) : 
  ∃ total : ℕ, total = 71 :=
sorry

end num_pos_integers_congruent_to_4_mod_7_l59_59080


namespace calculation_expression_solve_system_of_equations_l59_59185

-- Part 1: Prove the calculation
theorem calculation_expression :
  (6 - 2 * Real.sqrt 3) * Real.sqrt 3 - Real.sqrt ((2 - Real.sqrt 2) ^ 2) + 1 / Real.sqrt 2 = 
  6 * Real.sqrt 3 - 8 + 3 * Real.sqrt 2 / 2 :=
by
  -- proof will be here
  sorry

-- Part 2: Prove the solution of the system of equations
theorem solve_system_of_equations (x y : ℝ) :
  (5 * x - y = -9) ∧ (3 * x + y = 1) → (x = -1 ∧ y = 4) :=
by
  -- proof will be here
  sorry

end calculation_expression_solve_system_of_equations_l59_59185


namespace minimum_value_of_quadratic_function_l59_59270

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 2

theorem minimum_value_of_quadratic_function :
  ∃ m : ℝ, (∀ x : ℝ, quadratic_function x ≥ m) ∧ (∀ ε > 0, ∃ x : ℝ, quadratic_function x < m + ε) ∧ m = 2 :=
by
  sorry

end minimum_value_of_quadratic_function_l59_59270


namespace range_of_m_l59_59575

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) ↔ (m ≤ 2) :=
by
  sorry

end range_of_m_l59_59575


namespace abs_inequality_l59_59662

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_inequality_l59_59662


namespace find_n_l59_59556

def binomial_coefficient_sum (n : ℕ) (a b : ℝ) : ℝ :=
  (a + b) ^ n

def expanded_coefficient_sum (n : ℕ) (a b : ℝ) : ℝ :=
  (a + 3 * b) ^ n

theorem find_n (n : ℕ) :
  (expanded_coefficient_sum n 1 1) / (binomial_coefficient_sum n 1 1) = 64 → n = 6 :=
by 
  sorry

end find_n_l59_59556


namespace probability_of_sequence_123456_l59_59011

theorem probability_of_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1     -- Total number of sequences
  let specific_sequences := 1 * 3 * 5 * 7 * 9 * 11        -- Sequences leading to 123456
  specific_sequences / total_sequences = 1 / 720 := by
  let total_sequences := 74919600
  let specific_sequences := 10395
  sorry

end probability_of_sequence_123456_l59_59011


namespace sandy_final_position_and_distance_l59_59908

-- Define the conditions as statements
def walked_south (distance : ℕ) := distance = 20
def turned_left_facing_east := true
def walked_east (distance : ℕ) := distance = 20
def turned_left_facing_north := true
def walked_north (distance : ℕ) := distance = 20
def turned_right_facing_east := true
def walked_east_again (distance : ℕ) := distance = 20

-- Final position computation as a proof statement
theorem sandy_final_position_and_distance :
  ∃ (d : ℕ) (dir : String), walked_south 20 → turned_left_facing_east → walked_east 20 →
  turned_left_facing_north → walked_north 20 →
  turned_right_facing_east → walked_east_again 20 ∧ d = 40 ∧ dir = "east" :=
by
  sorry

end sandy_final_position_and_distance_l59_59908


namespace city_population_divided_l59_59744

theorem city_population_divided (total_population : ℕ) (parts : ℕ) (male_parts : ℕ) 
  (h1 : total_population = 1000) (h2 : parts = 5) (h3 : male_parts = 2) : 
  ∃ males : ℕ, males = 400 :=
by
  sorry

end city_population_divided_l59_59744


namespace unique_midpoints_are_25_l59_59474

/-- Define the properties of a parallelogram with marked points such as vertices, midpoints of sides, and intersection point of diagonals --/
structure Parallelogram :=
(vertices : Set ℝ)
(midpoints : Set ℝ)
(diagonal_intersection : ℝ)

def congruent_parallelograms (P P' : Parallelogram) : Prop :=
  P.vertices = P'.vertices ∧ P.midpoints = P'.midpoints ∧ P.diagonal_intersection = P'.diagonal_intersection

def unique_midpoints_count (P P' : Parallelogram) : ℕ := sorry

theorem unique_midpoints_are_25
  (P P' : Parallelogram)
  (h_congruent : congruent_parallelograms P P') :
  unique_midpoints_count P P' = 25 := sorry

end unique_midpoints_are_25_l59_59474


namespace shirt_cost_l59_59783

variables (S : ℝ)

theorem shirt_cost (h : 2 * S + (S + 3) + (1/2) * (2 * S + S + 3) = 36) : S = 7.88 :=
sorry

end shirt_cost_l59_59783


namespace solution_fractional_equation_l59_59204

noncomputable def solve_fractional_equation : Prop :=
  ∀ x : ℝ, (4/(x-2) = 2/x) ↔ x = -2

theorem solution_fractional_equation :
  solve_fractional_equation :=
by
  sorry

end solution_fractional_equation_l59_59204


namespace correct_transformation_of_95_sq_l59_59757

theorem correct_transformation_of_95_sq : 95^2 = 100^2 - 2 * 100 * 5 + 5^2 := by
  sorry

end correct_transformation_of_95_sq_l59_59757


namespace product_of_possible_values_l59_59960

theorem product_of_possible_values (b : ℝ) (side_length : ℝ) (square_condition : (b - 2) = side_length ∨ (2 - b) = side_length) : 
  (b = -3 ∨ b = 7) → (-3 * 7 = -21) :=
by
  intro h
  sorry

end product_of_possible_values_l59_59960


namespace gcd_values_count_l59_59125

theorem gcd_values_count (a b : ℕ) (h : a * b = 3600) : ∃ n, n = 29 ∧ ∀ d, d ∣ a ∧ d ∣ b → d = gcd a b → n = 29 :=
by { sorry }

end gcd_values_count_l59_59125


namespace first_part_lending_years_l59_59847

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

end first_part_lending_years_l59_59847


namespace harry_morning_routine_l59_59142

theorem harry_morning_routine : 
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  buy_time + read_and_eat_time = 45 :=
by
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  show buy_time + read_and_eat_time = 45
  sorry

end harry_morning_routine_l59_59142


namespace problem_I_problem_II_problem_III_l59_59054

variables {pA pB : ℝ}

-- Given conditions
def probability_A : ℝ := 0.7
def probability_B : ℝ := 0.6

-- Questions reformulated as proof goals
theorem problem_I : 
  sorry := 
 sorry

theorem problem_II : 
  -- Find: Probability that at least one of A or B succeeds on the first attempt
  sorry := 
 sorry

theorem problem_III : 
  -- Find: Probability that A succeeds exactly one more time than B in two attempts each
  sorry := 
 sorry

end problem_I_problem_II_problem_III_l59_59054


namespace line_ellipse_common_point_l59_59871

theorem line_ellipse_common_point (k : ℝ) (m : ℝ) :
  (∀ (x y : ℝ), y = k * x + 1 →
    (y^2 / m + x^2 / 5 ≤ 1)) ↔ (m ≥ 1 ∧ m ≠ 5) :=
by sorry

end line_ellipse_common_point_l59_59871


namespace yvettes_final_bill_l59_59608

theorem yvettes_final_bill :
  let alicia : ℝ := 7.5
  let brant : ℝ := 10
  let josh : ℝ := 8.5
  let yvette : ℝ := 9
  let tip_percentage : ℝ := 0.2
  ∃ final_bill : ℝ, final_bill = (alicia + brant + josh + yvette) * (1 + tip_percentage) ∧ final_bill = 42 :=
by
  sorry

end yvettes_final_bill_l59_59608


namespace base8_subtraction_l59_59922

theorem base8_subtraction : (325 : Nat) - (237 : Nat) = 66 :=
by 
  sorry

end base8_subtraction_l59_59922


namespace six_units_away_has_two_solutions_l59_59092

-- Define point A and its position on the number line
def A_position : ℤ := -3

-- Define the condition for a point x being 6 units away from point A
def is_6_units_away (x : ℤ) : Prop := abs (x + 3) = 6

-- The theorem stating that if x is 6 units away from -3, then x must be either 3 or -9
theorem six_units_away_has_two_solutions (x : ℤ) (h : is_6_units_away x) : x = 3 ∨ x = -9 := by
  sorry

end six_units_away_has_two_solutions_l59_59092


namespace find_g_inverse_75_l59_59666

noncomputable def g (x : ℝ) : ℝ := 3 * x^3 - 6

theorem find_g_inverse_75 : g⁻¹ 75 = 3 := sorry

end find_g_inverse_75_l59_59666


namespace average_bowling_score_l59_59336

theorem average_bowling_score 
    (gretchen_score : ℕ) (mitzi_score : ℕ) (beth_score : ℕ)
    (gretchen_eq : gretchen_score = 120)
    (mitzi_eq : mitzi_score = 113)
    (beth_eq : beth_score = 85) :
    (gretchen_score + mitzi_score + beth_score) / 3 = 106 := 
by
  sorry

end average_bowling_score_l59_59336


namespace temperature_lower_than_minus_three_l59_59305

theorem temperature_lower_than_minus_three (a b : ℤ) (hx : a = -3) (hy : b = -6) : a + b = -9 :=
by
  sorry

end temperature_lower_than_minus_three_l59_59305


namespace area_ratio_proof_l59_59907

open Real

noncomputable def area_ratio (FE AF DE CD ABCE : ℝ) :=
  (AF = 3 * FE) ∧ (CD = 3 * DE) ∧ (ABCE = 16 * FE^2) →
  (10 * FE^2 / ABCE = (5 / 8))

theorem area_ratio_proof (FE AF DE CD ABCE : ℝ) :
  AF = 3 * FE → CD = 3 * DE → ABCE = 16 * FE^2 →
  10 * FE^2 / ABCE = 5 / 8 :=
by
  intro hAF hCD hABCE
  sorry

end area_ratio_proof_l59_59907


namespace find_C_when_F_10_l59_59603

theorem find_C_when_F_10 : (∃ C : ℚ, ∀ F : ℚ, F = 10 → F = (9 / 5 : ℚ) * C + 32 → C = -110 / 9) :=
by
  sorry

end find_C_when_F_10_l59_59603


namespace find_smallest_angle_l59_59735

open Real

theorem find_smallest_angle :
  ∃ x : ℝ, (x > 0 ∧ sin (4 * x * (π / 180)) * sin (6 * x * (π / 180)) = cos (4 * x * (π / 180)) * cos (6 * x * (π / 180))) ∧ x = 9 :=
by
  sorry

end find_smallest_angle_l59_59735


namespace initial_bales_l59_59417

theorem initial_bales (B : ℕ) (cond1 : B + 35 = 82) : B = 47 :=
by
  sorry

end initial_bales_l59_59417


namespace ticket_is_five_times_soda_l59_59689

variable (p_i p_r : ℝ)

theorem ticket_is_five_times_soda
  (h1 : 6 * p_i + 20 * p_r = 50)
  (h2 : 6 * p_r = p_i + p_r) : p_i = 5 * p_r :=
sorry

end ticket_is_five_times_soda_l59_59689


namespace smallest_b_value_is_6_l59_59353

noncomputable def smallest_b_value (a b c : ℝ) (h_arith : a + c = 2 * b) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 216) : ℝ :=
b

theorem smallest_b_value_is_6 (a b c : ℝ) (h_arith : a + c = 2 * b) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 216) : 
  smallest_b_value a b c h_arith h_pos h_prod = 6 :=
sorry

end smallest_b_value_is_6_l59_59353


namespace each_person_bid_count_l59_59637

-- Define the conditions and initial values
noncomputable def auctioneer_price_increase : ℕ := 5
noncomputable def initial_price : ℕ := 15
noncomputable def final_price : ℕ := 65
noncomputable def number_of_bidders : ℕ := 2

-- Define the proof statement
theorem each_person_bid_count : 
  (final_price - initial_price) / auctioneer_price_increase / number_of_bidders = 5 :=
by sorry

end each_person_bid_count_l59_59637


namespace distinct_solutions_abs_eq_l59_59284

theorem distinct_solutions_abs_eq (x : ℝ) : 
  (|x - 3| = |x + 5|) → x = -1 :=
by
  sorry

end distinct_solutions_abs_eq_l59_59284


namespace calculate_speed_of_stream_l59_59591

noncomputable def speed_of_stream (boat_speed : ℕ) (downstream_distance : ℕ) (upstream_distance : ℕ) : ℕ :=
  let x := (downstream_distance * boat_speed - boat_speed * upstream_distance) / (downstream_distance + upstream_distance)
  x

theorem calculate_speed_of_stream :
  speed_of_stream 20 26 14 = 6 := by
  sorry

end calculate_speed_of_stream_l59_59591


namespace find_t_l59_59981

theorem find_t
  (x y t : ℝ)
  (h1 : 2 ^ x = t)
  (h2 : 5 ^ y = t)
  (h3 : 1 / x + 1 / y = 2)
  (h4 : t ≠ 1) : 
  t = Real.sqrt 10 := 
by
  sorry

end find_t_l59_59981


namespace sheets_in_stack_l59_59379

theorem sheets_in_stack (sheets : ℕ) (thickness : ℝ) (h1 : sheets = 400) (h2 : thickness = 4) :
    let thickness_per_sheet := thickness / sheets
    let stack_height := 6
    (stack_height / thickness_per_sheet = 600) :=
by
  sorry

end sheets_in_stack_l59_59379


namespace area_of_quadrilateral_l59_59031

theorem area_of_quadrilateral (d a b : ℝ) (h₀ : d = 28) (h₁ : a = 9) (h₂ : b = 6) :
  (1 / 2 * d * a) + (1 / 2 * d * b) = 210 :=
by
  -- Provided proof steps are skipped
  sorry

end area_of_quadrilateral_l59_59031


namespace xy_squared_sum_l59_59278

theorem xy_squared_sum {x y : ℝ} (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 :=
by
  sorry

end xy_squared_sum_l59_59278


namespace height_of_carton_is_70_l59_59422

def carton_dimensions : ℕ × ℕ := (25, 42)
def soap_box_dimensions : ℕ × ℕ × ℕ := (7, 6, 5)
def max_soap_boxes : ℕ := 300

theorem height_of_carton_is_70 :
  let (carton_length, carton_width) := carton_dimensions
  let (soap_box_length, soap_box_width, soap_box_height) := soap_box_dimensions
  let boxes_per_layer := (carton_length / soap_box_length) * (carton_width / soap_box_width)
  let num_layers := max_soap_boxes / boxes_per_layer
  (num_layers * soap_box_height) = 70 :=
by
  have carton_length := 25
  have carton_width := 42
  have soap_box_length := 7
  have soap_box_width := 6
  have soap_box_height := 5
  have max_soap_boxes := 300
  have boxes_per_layer := (25 / 7) * (42 / 6)
  have num_layers := max_soap_boxes / boxes_per_layer
  sorry

end height_of_carton_is_70_l59_59422


namespace function_relationship_selling_price_for_profit_max_profit_l59_59816

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

end function_relationship_selling_price_for_profit_max_profit_l59_59816


namespace arithmetic_seq_a6_l59_59526

open Real

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, (a (n + m) = a n + a m - a 0)

-- Given conditions
def condition_1 (a : ℕ → ℝ) : Prop :=
  a 2 = 4

def condition_2 (a : ℕ → ℝ) : Prop :=
  a 4 = 2

-- Mathematical statement
theorem arithmetic_seq_a6 
  (a : ℕ → ℝ)
  (h_seq: arithmetic_sequence a)
  (h_cond1 : condition_1 a)
  (h_cond2 : condition_2 a) : 
  a 6 = 0 := 
sorry

end arithmetic_seq_a6_l59_59526


namespace solve_for_pure_imaginary_l59_59168

theorem solve_for_pure_imaginary (x : ℝ) 
  (h1 : x^2 - 1 = 0) 
  (h2 : x - 1 ≠ 0) 
  : x = -1 :=
sorry

end solve_for_pure_imaginary_l59_59168


namespace circle_equation_midpoint_trajectory_l59_59797

-- Definition for the circle equation proof
theorem circle_equation (x y : ℝ) (h : (x - 3)^2 + (y - 2)^2 = 13)
  (hx : x = 3) (hy : y = 2) : 
  (x - 3)^2 + (y - 2)^2 = 13 := by
  sorry -- Placeholder for proof

-- Definition for the midpoint trajectory proof
theorem midpoint_trajectory (x y : ℝ) (hx : x = (2 * x - 11) / 2)
  (hy : y = (2 * y - 2) / 2) (h : (2 * x - 11)^2 + (2 * y - 2)^2 = 13) :
  (x - 11 / 2)^2 + (y - 1)^2 = 13 / 4 := by
  sorry -- Placeholder for proof

end circle_equation_midpoint_trajectory_l59_59797


namespace series_proof_l59_59567

theorem series_proof (a b : ℝ) (h : (∑' n : ℕ, (-1)^n * a / b^(n+1)) = 6) : 
  (∑' n : ℕ, (-1)^n * a / (a - b)^(n+1)) = 6 / 7 := 
sorry

end series_proof_l59_59567


namespace prime_factor_of_sum_l59_59495

theorem prime_factor_of_sum (n : ℤ) : ∃ p : ℕ, Nat.Prime p ∧ p = 2 ∧ (2 * n + 1 + 2 * n + 3 + 2 * n + 5 + 2 * n + 7) % p = 0 :=
by
  sorry

end prime_factor_of_sum_l59_59495


namespace part1_exists_rectangle_B_part2_no_rectangle_B_general_exists_rectangle_B_l59_59128

-- Part 1: Prove existence of rectangle B with sides 2 + sqrt(2)/2 and 2 - sqrt(2)/2
theorem part1_exists_rectangle_B : 
  ∃ (x y : ℝ), (x + y = 4) ∧ (x * y = 7 / 2) :=
by
  sorry

-- Part 2: Prove non-existence of rectangle B for given sides of the known rectangle
theorem part2_no_rectangle_B : 
  ¬ ∃ (x y : ℝ), (x + y = 5 / 2) ∧ (x * y = 2) :=
by
  sorry

-- Part 3: General proof for any given sides of the known rectangle
theorem general_exists_rectangle_B (m n : ℝ) : 
  ∃ (x y : ℝ), (x + y = 3 * (m + n)) ∧ (x * y = 3 * m * n) :=
by
  sorry

end part1_exists_rectangle_B_part2_no_rectangle_B_general_exists_rectangle_B_l59_59128


namespace no_xy_term_implies_k_eq_4_l59_59382

theorem no_xy_term_implies_k_eq_4 (k : ℝ) :
  (∀ x y : ℝ, (x + 2 * y) * (2 * x - k * y - 1) = 2 * x^2 + (4 - k) * x * y - x - 2 * k * y^2 - 2 * y) →
  ((4 - k) = 0) →
  k = 4 := 
by
  intros h1 h2
  sorry

end no_xy_term_implies_k_eq_4_l59_59382


namespace intersection_of_A_and_B_l59_59527

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
  sorry

end intersection_of_A_and_B_l59_59527


namespace waiter_date_trick_l59_59237

theorem waiter_date_trick :
  ∃ d₂ : ℕ, ∃ x : ℝ, 
  (∀ d₁ : ℕ, ∀ x : ℝ, x + d₁ = 168) ∧
  3 * x + d₂ = 486 ∧
  3 * (x + d₂) = 516 ∧
  d₂ = 15 :=
by
  sorry

end waiter_date_trick_l59_59237


namespace mean_equality_l59_59161

theorem mean_equality (x : ℚ) : 
  (3 + 7 + 15) / 3 = (x + 10) / 2 → x = 20 / 3 := 
by 
  sorry

end mean_equality_l59_59161


namespace cupcakes_leftover_l59_59806

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

end cupcakes_leftover_l59_59806


namespace part1_part2_l59_59986

-- Part (1)
theorem part1 (a : ℝ) (A B : Set ℝ) 
  (hA : A = { x : ℝ | x^2 - 3 * x + 2 = 0 }) 
  (hB : B = { x : ℝ | x^2 - a * x + a - 1 = 0 }) 
  (hUnion : A ∪ B = A) : 
  a = 2 ∨ a = 3 := 
sorry

-- Part (2)
theorem part2 (m : ℝ) (A C : Set ℝ) 
  (hA : A = { x : ℝ | x^2 - 3 * x + 2 = 0 }) 
  (hC : C = { x : ℝ | x^2 + 2 * (m + 1) * x + m^2 - 5 = 0 }) 
  (hInter : A ∩ C = C) : 
  m ∈ Set.Iic (-3) := 
sorry

end part1_part2_l59_59986


namespace total_time_watching_videos_l59_59392

theorem total_time_watching_videos 
  (cat_video_length : ℕ)
  (dog_video_length : ℕ)
  (gorilla_video_length : ℕ)
  (h1 : cat_video_length = 4)
  (h2 : dog_video_length = 2 * cat_video_length)
  (h3 : gorilla_video_length = 2 * (cat_video_length + dog_video_length)) :
  cat_video_length + dog_video_length + gorilla_video_length = 36 :=
  by
  sorry

end total_time_watching_videos_l59_59392


namespace math_problem_l59_59729

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end math_problem_l59_59729


namespace complement_A_in_U_l59_59471

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | x^2 + x - 2 < 0}

theorem complement_A_in_U :
  (U \ A) = {-2, 1, 2} :=
by 
  -- proof will be done here
  sorry

end complement_A_in_U_l59_59471


namespace intercepts_of_line_l59_59781

theorem intercepts_of_line (x y : ℝ) : 
  (x + 6 * y + 2 = 0) → (x = -2) ∧ (y = -1 / 3) :=
by
  sorry

end intercepts_of_line_l59_59781


namespace ratio_karen_beatrice_l59_59443

noncomputable def karen_crayons : ℕ := 128
noncomputable def judah_crayons : ℕ := 8
noncomputable def gilbert_crayons : ℕ := 4 * judah_crayons
noncomputable def beatrice_crayons : ℕ := 2 * gilbert_crayons

theorem ratio_karen_beatrice :
  karen_crayons / beatrice_crayons = 2 := by
sorry

end ratio_karen_beatrice_l59_59443


namespace min_time_to_complete_tasks_l59_59723

-- Define the conditions as individual time durations for each task in minutes
def bed_making_time : ℕ := 3
def teeth_washing_time : ℕ := 4
def water_boiling_time : ℕ := 10
def breakfast_time : ℕ := 7
def dish_washing_time : ℕ := 1
def backpack_organizing_time : ℕ := 2
def milk_making_time : ℕ := 1

-- Define the total minimum time required to complete all tasks
def min_completion_time : ℕ := 18

-- A theorem stating that given the times for each task, the minimum completion time is 18 minutes
theorem min_time_to_complete_tasks :
  bed_making_time + teeth_washing_time + water_boiling_time + 
  breakfast_time + dish_washing_time + backpack_organizing_time + milk_making_time - 
  (bed_making_time + teeth_washing_time + backpack_organizing_time + milk_making_time) <=
  min_completion_time := by
  sorry

end min_time_to_complete_tasks_l59_59723


namespace trig_identity_l59_59146

-- Define the angle alpha with the given condition tan(alpha) = 2
variables (α : ℝ) (h : Real.tan α = 2)

-- State the theorem
theorem trig_identity : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
by
  sorry

end trig_identity_l59_59146


namespace f_le_g_for_a_eq_neg1_l59_59226

noncomputable def f (a : ℝ) (b : ℝ) (x : ℝ) : ℝ :=
  (a * x + b) * Real.exp x

noncomputable def g (t : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * x - Real.log x + t

theorem f_le_g_for_a_eq_neg1 (t : ℝ) :
  let b := 3
  ∃ x ∈ Set.Ioi 0, f (-1) b x ≤ g t x ↔ t ≤ Real.exp 2 - 1 / 2 :=
by
  sorry

end f_le_g_for_a_eq_neg1_l59_59226


namespace total_pages_in_book_l59_59398

theorem total_pages_in_book (P : ℕ) 
  (h1 : 7 / 13 * P = P - 96 - 5 / 9 * (P - 7 / 13 * P))
  (h2 : 96 = 4 / 9 * (P - 7 / 13 * P)) : 
  P = 468 :=
 by 
    sorry

end total_pages_in_book_l59_59398


namespace rectangle_width_length_ratio_l59_59849

theorem rectangle_width_length_ratio (w : ℕ) (h : w + 10 = 15) : w / 10 = 1 / 2 :=
by sorry

end rectangle_width_length_ratio_l59_59849


namespace nursing_home_milk_l59_59053

theorem nursing_home_milk :
  ∃ x y : ℕ, (2 * x + 16 = y) ∧ (4 * x - 12 = y) ∧ (x = 14) ∧ (y = 44) :=
by
  sorry

end nursing_home_milk_l59_59053


namespace compare_squares_l59_59753

theorem compare_squares (a : ℝ) : (a + 1)^2 > a^2 + 2 * a := by
  -- the proof would go here, but we skip it according to the instruction
  sorry

end compare_squares_l59_59753


namespace range_of_a_l59_59914

theorem range_of_a (a : ℝ) 
  (p : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a ≥ Real.exp x) 
  (q : ∃ x : ℝ, x^2 - 4 * x + a ≤ 0) : 
  e ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l59_59914


namespace theater_ticket_cost_l59_59540

theorem theater_ticket_cost
  (O B : ℕ)
  (h1 : O + B = 370)
  (h2 : B = O + 190) 
  : 12 * O + 8 * B = 3320 :=
by
  sorry

end theater_ticket_cost_l59_59540


namespace cagr_decline_l59_59701

theorem cagr_decline 
  (EV BV : ℝ) (n : ℕ) 
  (h_ev : EV = 52)
  (h_bv : BV = 89)
  (h_n : n = 3)
: ((EV / BV) ^ (1 / n) - 1) = -0.1678 := 
by
  rw [h_ev, h_bv, h_n]
  sorry

end cagr_decline_l59_59701


namespace temperature_difference_in_fahrenheit_l59_59123

-- Define the conversion formula from Celsius to Fahrenheit as a function
def celsius_to_fahrenheit (C : ℝ) : ℝ := 1.8 * C + 32

-- Define the temperatures in Boston and New York
variables (C_B C_N : ℝ)

-- Condition: New York is 10 degrees Celsius warmer than Boston
axiom temp_difference : C_N = C_B + 10

-- Goal: The temperature difference in Fahrenheit
theorem temperature_difference_in_fahrenheit : celsius_to_fahrenheit C_N - celsius_to_fahrenheit C_B = 18 :=
by sorry

end temperature_difference_in_fahrenheit_l59_59123


namespace price_reduction_l59_59005

theorem price_reduction (x y : ℕ) (h1 : (13 - x) * y = 781) (h2 : y ≤ 100) : x = 2 :=
sorry

end price_reduction_l59_59005


namespace smallest_value_of_y_l59_59721

theorem smallest_value_of_y (x y z d : ℝ) (h1 : x = y - d) (h2 : z = y + d) (h3 : x * y * z = 125) (h4 : 0 < x ∧ 0 < y ∧ 0 < z) : y ≥ 5 :=
by
  -- Officially, the user should navigate through the proof, but we conclude with 'sorry' as placeholder
  sorry

end smallest_value_of_y_l59_59721


namespace find_k_values_l59_59985

theorem find_k_values :
    ∀ (k : ℚ),
    (∀ (a b : ℚ), (5 * a^2 + 7 * a + k = 0) ∧ (5 * b^2 + 7 * b + k = 0) ∧ |a - b| = a^2 + b^2 → k = 21 / 25 ∨ k = -21 / 25) :=
by
  sorry

end find_k_values_l59_59985


namespace negation_of_existential_prop_l59_59252

open Real

theorem negation_of_existential_prop :
  ¬ (∃ x, x ≥ π / 2 ∧ sin x > 1) ↔ ∀ x, x < π / 2 → sin x ≤ 1 :=
by
  sorry

end negation_of_existential_prop_l59_59252


namespace evaluate_expression_l59_59312

theorem evaluate_expression : 
  (2 ^ 2003 * 3 ^ 2002 * 5) / (6 ^ 2003) = (5 / 3) :=
by sorry

end evaluate_expression_l59_59312


namespace fruit_seller_price_l59_59609

theorem fruit_seller_price 
  (CP SP SP_profit : ℝ)
  (h1 : SP = CP * 0.88)
  (h2 : SP_profit = CP * 1.20)
  (h3 : SP_profit = 21.818181818181817) :
  SP = 16 := 
by 
  sorry

end fruit_seller_price_l59_59609


namespace smaller_part_area_l59_59057

theorem smaller_part_area (x y : ℝ) (h1 : x + y = 500) (h2 : y - x = (1 / 5) * ((x + y) / 2)) : x = 225 :=
by
  sorry

end smaller_part_area_l59_59057


namespace min_value_x2_y2_z2_l59_59623

theorem min_value_x2_y2_z2 (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + 2 * y + 3 * z = 2) : 
  x^2 + y^2 + z^2 ≥ 2 / 7 :=
sorry

end min_value_x2_y2_z2_l59_59623


namespace count_numbers_with_remainder_7_dividing_65_l59_59627

theorem count_numbers_with_remainder_7_dividing_65 : 
  (∃ n : ℕ, n > 7 ∧ n ∣ 58 ∧ 65 % n = 7) ∧ 
  (∀ m : ℕ, m > 7 ∧ m ∣ 58 ∧ 65 % m = 7 → m = 29 ∨ m = 58) :=
sorry

end count_numbers_with_remainder_7_dividing_65_l59_59627


namespace negation_of_existence_l59_59027

theorem negation_of_existence : 
  (¬ ∃ x_0 : ℝ, (x_0 + 1 < 0) ∨ (x_0^2 - x_0 > 0)) ↔ ∀ x : ℝ, (x + 1 ≥ 0) ∧ (x^2 - x ≤ 0) := 
by
  sorry

end negation_of_existence_l59_59027


namespace simplify_and_evaluate_l59_59389

theorem simplify_and_evaluate (x : ℝ) (h₁ : x ≠ 0) (h₂ : x = 2) : 
  (1 + 1 / x) / ((x^2 - 1) / x) = 1 := 
by 
  sorry

end simplify_and_evaluate_l59_59389


namespace solve_inequality_l59_59938

theorem solve_inequality : {x : ℝ | 9 * x^2 + 6 * x + 1 ≤ 0} = {(-1 : ℝ) / 3} :=
by
  sorry

end solve_inequality_l59_59938


namespace point_coordinates_l59_59634

-- We assume that the point P has coordinates (2, 4) and prove that the coordinates with respect to the origin in Cartesian system are indeed (2, 4).
theorem point_coordinates (x y : ℝ) (h : x = 2 ∧ y = 4) : (x, y) = (2, 4) :=
by
  sorry

end point_coordinates_l59_59634


namespace lines_intersect_lines_parallel_lines_coincident_l59_59465

-- Define line equations
def l1 (m x y : ℝ) := (m + 2) * x + (m + 3) * y - 5 = 0
def l2 (m x y : ℝ) := 6 * x + (2 * m - 1) * y - 5 = 0

-- Prove conditions for intersection
theorem lines_intersect (m : ℝ) : ¬(m = -5 / 2 ∨ m = 4) ↔
  ∃ x y : ℝ, l1 m x y ∧ l2 m x y := sorry

-- Prove conditions for parallel lines
theorem lines_parallel (m : ℝ) : m = -5 / 2 ↔
  ∀ x y : ℝ, l1 m x y ∧ l2 m x y → l1 m x y → l2 m x y := sorry

-- Prove conditions for coincident lines
theorem lines_coincident (m : ℝ) : m = 4 ↔
  ∀ x y : ℝ, l1 m x y ↔ l2 m x y := sorry

end lines_intersect_lines_parallel_lines_coincident_l59_59465


namespace point_on_line_l59_59343

theorem point_on_line (s : ℝ) : 
  (∃ b : ℝ, ∀ x y : ℝ, (y = 3 * x + b) → 
    ((2 = x ∧ y = 8) ∨ (4 = x ∧ y = 14) ∨ (6 = x ∧ y = 20) ∨ (35 = x ∧ y = s))) → s = 107 :=
by
  sorry

end point_on_line_l59_59343


namespace magician_card_pairs_l59_59479

theorem magician_card_pairs:
  ∃ (f : Fin 65 → Fin 65 × Fin 65), 
  (∀ m n : Fin 65, ∃ k l : Fin 65, (f m = (k, l) ∧ f n = (l, k))) := 
sorry

end magician_card_pairs_l59_59479


namespace intersection_of_sets_l59_59706

open Set Real

theorem intersection_of_sets :
  let A := {x : ℝ | x^2 - 2*x - 3 < 0}
  let B := {y : ℝ | ∃ (x : ℝ), y = sin x}
  A ∩ B = Ioc (-1) 1 := by
  sorry

end intersection_of_sets_l59_59706


namespace find_factors_of_224_l59_59172

theorem find_factors_of_224 : ∃ (a b c : ℕ), a * b * c = 224 ∧ c = 2 * a ∧ a ≠ b ∧ b ≠ c :=
by
  -- Prove that the factors meeting the criteria exist
  sorry

end find_factors_of_224_l59_59172


namespace anatoliy_handshakes_l59_59113

-- Define the total number of handshakes
def total_handshakes := 197

-- Define friends excluding Anatoliy
def handshake_func (n : Nat) : Nat :=
  n * (n - 1) / 2

-- Define the target problem stating that Anatoliy made 7 handshakes
theorem anatoliy_handshakes (n k : Nat) (h : handshake_func n + k = total_handshakes) : k = 7 :=
by sorry

end anatoliy_handshakes_l59_59113


namespace restaurant_problem_l59_59375

theorem restaurant_problem (A K : ℕ) (h1 : A + K = 11) (h2 : 8 * A = 72) : K = 2 :=
by
  sorry

end restaurant_problem_l59_59375


namespace smallest_common_multiple_l59_59321

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l59_59321


namespace circumcircle_radius_l59_59184

open Real

theorem circumcircle_radius (a b c A B C S R : ℝ) 
  (h1 : S = (1/2) * sin A * sin B * sin C)
  (h2 : S = (1/2) * a * b * sin C)
  (h3 : ∀ x y, x = y → x * cos 0 = y * cos 0):
  R = (1/2) :=
by
  sorry

end circumcircle_radius_l59_59184


namespace eggs_in_each_basket_is_15_l59_59246
open Nat

theorem eggs_in_each_basket_is_15 :
  ∃ n : ℕ, (n ∣ 30) ∧ (n ∣ 45) ∧ (n ≥ 5) ∧ (n = 15) :=
sorry

end eggs_in_each_basket_is_15_l59_59246


namespace pipe_r_fill_time_l59_59801

theorem pipe_r_fill_time (x : ℝ) : 
  (1 / 3 + 1 / 9 + 1 / x = 1 / 2) → 
  x = 18 :=
by 
  sorry

end pipe_r_fill_time_l59_59801


namespace election_result_l59_59427

def votes_A : ℕ := 12
def votes_B : ℕ := 3
def votes_C : ℕ := 15

def is_class_president (candidate_votes : ℕ) : Prop :=
  candidate_votes = max (max votes_A votes_B) votes_C

theorem election_result : is_class_president votes_C :=
by
  unfold is_class_president
  rw [votes_A, votes_B, votes_C]
  sorry

end election_result_l59_59427


namespace negative_expression_b_negative_expression_c_negative_expression_e_l59_59372

theorem negative_expression_b:
  3 * Real.sqrt 11 - 10 < 0 := 
sorry

theorem negative_expression_c:
  18 - 5 * Real.sqrt 13 < 0 := 
sorry

theorem negative_expression_e:
  10 * Real.sqrt 26 - 51 < 0 := 
sorry

end negative_expression_b_negative_expression_c_negative_expression_e_l59_59372


namespace exponentiation_problem_l59_59157

theorem exponentiation_problem 
(a b : ℝ) 
(h : a ^ b = 1 / 8) : a ^ (-3 * b) = 512 := 
sorry

end exponentiation_problem_l59_59157


namespace julie_savings_fraction_l59_59250

variables (S : ℝ) (x : ℝ)
theorem julie_savings_fraction (h : 12 * S * x = 4 * S * (1 - x)) : 1 - x = 3 / 4 :=
sorry

end julie_savings_fraction_l59_59250


namespace complement_union_eq_l59_59677

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem complement_union_eq : (U \ (S ∪ T)) = {2, 4, 7, 8} :=
by {
  sorry
}

end complement_union_eq_l59_59677


namespace total_handshakes_l59_59496

section Handshakes

-- Define the total number of players
def total_players : ℕ := 4 + 6

-- Define the number of players in 2 and 3 player teams
def num_2player_teams : ℕ := 2
def num_3player_teams : ℕ := 2

-- Define the number of players per 2 player team and 3 player team
def players_per_2player_team : ℕ := 2
def players_per_3player_team : ℕ := 3

-- Define the total number of players in 2 player teams and in 3 player teams
def total_2player_team_players : ℕ := num_2player_teams * players_per_2player_team
def total_3player_team_players : ℕ := num_3player_teams * players_per_3player_team

-- Calculate handshakes
def handshakes (total_2player : ℕ) (total_3player : ℕ) : ℕ :=
  let h1 := total_2player * (total_players - players_per_2player_team) / 2
  let h2 := total_3player * (total_players - players_per_3player_team) / 2
  h1 + h2

-- Prove the total number of handshakes
theorem total_handshakes : handshakes total_2player_team_players total_3player_team_players = 37 :=
by
  have h1 := total_2player_team_players * (total_players - players_per_2player_team) / 2
  have h2 := total_3player_team_players * (total_players - players_per_3player_team) / 2
  have h_total := h1 + h2
  sorry

end Handshakes

end total_handshakes_l59_59496


namespace derivative_of_f_l59_59840

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_of_f (x : ℝ) (h : 0 < x) :
    deriv f x = (1 - Real.log x) / (x ^ 2) := 
sorry

end derivative_of_f_l59_59840


namespace smaller_integer_l59_59050

noncomputable def m : ℕ := 1
noncomputable def n : ℕ := 1998 * m

lemma two_digit_number (m: ℕ) : 10 ≤ m ∧ m < 100 := by sorry
lemma three_digit_number (n: ℕ) : 100 ≤ n ∧ n < 1000 := by sorry

theorem smaller_integer 
  (two_digit_m: 10 ≤ m ∧ m < 100)
  (three_digit_n: 100 ≤ n ∧ n < 1000)
  (avg_eq_decimal: (m + n) / 2 = m + n / 1000)
  : m = 1 := by 
  sorry

end smaller_integer_l59_59050


namespace line_through_origin_and_intersection_of_lines_l59_59199

theorem line_through_origin_and_intersection_of_lines 
  (x y : ℝ)
  (h1 : x - 3 * y + 4 = 0)
  (h2 : 2 * x + y + 5 = 0) :
  3 * x + 19 * y = 0 :=
sorry

end line_through_origin_and_intersection_of_lines_l59_59199


namespace investment_rate_l59_59776

theorem investment_rate (P_total P_7000 P_15000 I_total : ℝ)
  (h_investment : P_total = 22000)
  (h_investment_7000 : P_7000 = 7000)
  (h_investment_15000 : P_15000 = P_total - P_7000)
  (R_7000 : ℝ)
  (h_rate_7000 : R_7000 = 0.18)
  (I_7000 : ℝ)
  (h_interest_7000 : I_7000 = P_7000 * R_7000)
  (h_total_interest : I_total = 3360) :
  ∃ (R_15000 : ℝ), (I_total - I_7000) = P_15000 * R_15000 ∧ R_15000 = 0.14 := 
by
  sorry

end investment_rate_l59_59776


namespace best_fitting_model_l59_59506

/-- A type representing the coefficient of determination of different models -/
def r_squared (m : ℕ) : ℝ :=
  match m with
  | 1 => 0.98
  | 2 => 0.80
  | 3 => 0.50
  | 4 => 0.25
  | _ => 0 -- An auxiliary value for invalid model numbers

/-- The best fitting model is the one with the highest r_squared value --/
theorem best_fitting_model : r_squared 1 = max (r_squared 1) (max (r_squared 2) (max (r_squared 3) (r_squared 4))) :=
by
  sorry

end best_fitting_model_l59_59506


namespace fraction_meaningful_l59_59510

theorem fraction_meaningful (x : ℝ) : (x-5) ≠ 0 ↔ (1 / (x - 5)) = (1 / (x - 5)) := 
by 
  sorry

end fraction_meaningful_l59_59510


namespace ammeter_sum_l59_59576

variable (A1 A2 A3 A4 A5 : ℝ)
variable (I2 : ℝ)
variable (h1 : I2 = 4)
variable (h2 : A1 = I2)
variable (h3 : A3 = 2 * A1)
variable (h4 : A5 = A3 + A1)
variable (h5 : A4 = (5 / 3) * A5)

theorem ammeter_sum (A1 A2 A3 A4 A5 I2 : ℝ) (h1 : I2 = 4) (h2 : A1 = I2) (h3 : A3 = 2 * A1)
                   (h4 : A5 = A3 + A1) (h5 : A4 = (5 / 3) * A5) :
  A1 + I2 + A3 + A4 + A5 = 48 := 
sorry

end ammeter_sum_l59_59576


namespace original_number_is_3199_l59_59174

theorem original_number_is_3199 (n : ℕ) (k : ℕ) (h1 : k = 3200) (h2 : (n + k) % 8 = 0) : n = 3199 :=
sorry

end original_number_is_3199_l59_59174


namespace sin_alpha_pi_over_3_plus_sin_alpha_l59_59791

-- Defining the problem with the given conditions
variable (α : ℝ)
variable (hcos : Real.cos (α + (2 / 3) * Real.pi) = 4 / 5)
variable (hα : -Real.pi / 2 < α ∧ α < 0)

-- Statement to prove
theorem sin_alpha_pi_over_3_plus_sin_alpha :
  Real.sin (α + Real.pi / 3) + Real.sin α = -4 * Real.sqrt 3 / 5 :=
sorry

end sin_alpha_pi_over_3_plus_sin_alpha_l59_59791


namespace no_integer_solution_l59_59681

theorem no_integer_solution (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by
  -- Proof omitted
  sorry

end no_integer_solution_l59_59681


namespace total_peaches_l59_59222

variable (numberOfBaskets : ℕ)
variable (redPeachesPerBasket : ℕ)
variable (greenPeachesPerBasket : ℕ)

theorem total_peaches (h1 : numberOfBaskets = 1) 
                      (h2 : redPeachesPerBasket = 4)
                      (h3 : greenPeachesPerBasket = 3) :
  numberOfBaskets * (redPeachesPerBasket + greenPeachesPerBasket) = 7 := 
by
  sorry

end total_peaches_l59_59222


namespace minimize_S_l59_59196

theorem minimize_S (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 23) : n = 7 ↔ ∃ (m : ℕ), (∀ k ≤ m, a k <= 0) ∧ m = 7 :=
by
  sorry

end minimize_S_l59_59196


namespace min_value_is_3_plus_2_sqrt_2_l59_59170

noncomputable def minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = a*b) : ℝ :=
a + b

theorem min_value_is_3_plus_2_sqrt_2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = a*b) :
  minimum_value a b h1 h2 h3 = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_is_3_plus_2_sqrt_2_l59_59170


namespace airsickness_related_to_gender_l59_59310

def a : ℕ := 28
def b : ℕ := 28
def c : ℕ := 28
def d : ℕ := 56
def n : ℕ := 140

def contingency_relation (a b c d n K2 : ℕ) : Prop := 
  let numerator := n * (a * d - b * c)^2
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  K2 > 3841 / 1000

-- Goal statement for the proof
theorem airsickness_related_to_gender :
  contingency_relation a b c d n 3888 :=
  sorry

end airsickness_related_to_gender_l59_59310


namespace set_intersection_complement_l59_59978
open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}
def comp_T : Set ℕ := U \ T

theorem set_intersection_complement :
  S ∩ comp_T = {1, 5} := by
  sorry

end set_intersection_complement_l59_59978


namespace third_derivative_y_l59_59000

noncomputable def y (x : ℝ) : ℝ := x * Real.cos (x^2)

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = (8 * x^4 - 6) * Real.sin (x^2) - 24 * x^2 * Real.cos (x^2) :=
by
  sorry

end third_derivative_y_l59_59000


namespace tommy_needs_to_save_l59_59191

theorem tommy_needs_to_save (books : ℕ) (cost_per_book : ℕ) (money_he_has : ℕ) 
  (total_cost : ℕ) (money_needed : ℕ) 
  (h1 : books = 8)
  (h2 : cost_per_book = 5)
  (h3 : money_he_has = 13)
  (h4 : total_cost = books * cost_per_book) :
  money_needed = total_cost - money_he_has ∧ money_needed = 27 :=
by 
  sorry

end tommy_needs_to_save_l59_59191


namespace classes_after_drop_remaining_hours_of_classes_per_day_l59_59944

def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_classes : ℕ := 1

theorem classes_after_drop 
  (initial_classes : ℕ)
  (hours_per_class : ℕ)
  (dropped_classes : ℕ) :
  initial_classes - dropped_classes = 3 :=
by
  -- We are skipping the proof and using sorry for now.
  sorry

theorem remaining_hours_of_classes_per_day
  (initial_classes : ℕ)
  (hours_per_class : ℕ)
  (dropped_classes : ℕ)
  (h : initial_classes - dropped_classes = 3) :
  hours_per_class * (initial_classes - dropped_classes) = 6 :=
by
  -- We are skipping the proof and using sorry for now.
  sorry

end classes_after_drop_remaining_hours_of_classes_per_day_l59_59944


namespace domain_log_function_l59_59148

open Real

def quadratic_term (x : ℝ) : ℝ := 4 - 3 * x - x^2

def valid_argument (x : ℝ) : Prop := quadratic_term x > 0

theorem domain_log_function : { x : ℝ | valid_argument x } = Set.Ioo (-4 : ℝ) (1 : ℝ) :=
by
  sorry

end domain_log_function_l59_59148


namespace part1_part2_l59_59934

open Real

noncomputable def part1_statement (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 - 2 * m * x - 1 < 0

noncomputable def part2_statement (x : ℝ) : Prop := 
  ∀ (m : ℝ), |m| ≤ 1 → (m * x^2 - 2 * m * x - 1 < 0)

theorem part1 : part1_statement m ↔ (-1 < m ∧ m ≤ 0) :=
sorry

theorem part2 : part2_statement x ↔ ((1 - sqrt 2 < x ∧ x < 1) ∨ (1 < x ∧ x < 1 + sqrt 2)) :=
sorry

end part1_part2_l59_59934


namespace all_integers_equal_l59_59235

theorem all_integers_equal (k : ℕ) (a : Fin (2 * k + 1) → ℤ)
(h : ∀ b : Fin (2 * k + 1) → ℤ,
  (∀ i : Fin (2 * k + 1), b i = (a ((i : ℕ) % (2 * k + 1)) + a ((i + 1) % (2 * k + 1))) / 2) →
  ∀ i : Fin (2 * k + 1), ↑(b i) % 2 = 0) :
∀ i j : Fin (2 * k + 1), a i = a j :=
by
  sorry

end all_integers_equal_l59_59235


namespace exponentiation_properties_l59_59628

theorem exponentiation_properties:
  (10^6) * (10^2)^3 / 10^4 = 10^8 :=
by
  sorry

end exponentiation_properties_l59_59628


namespace sum_difference_even_odd_l59_59004

theorem sum_difference_even_odd :
  let x := (100 / 2) * (2 + 200)
  let y := (100 / 2) * (1 + 199)
  x - y = 100 :=
by
  sorry

end sum_difference_even_odd_l59_59004


namespace find_a_l59_59612

variable (a : ℤ) -- We assume a is an integer for simplicity

def point_on_x_axis (P : Nat × ℤ) : Prop :=
  P.snd = 0

theorem find_a (h : point_on_x_axis (4, 2 * a + 6)) : a = -3 :=
by
  sorry

end find_a_l59_59612


namespace billy_piles_l59_59475

theorem billy_piles (Q D : ℕ) (h : 2 * Q + 3 * D = 20) :
  Q = 4 ∧ D = 4 :=
sorry

end billy_piles_l59_59475


namespace senior_employee_bonus_l59_59748

theorem senior_employee_bonus (J S : ℝ) 
  (h1 : S = J + 1200)
  (h2 : J + S = 5000) : 
  S = 3100 :=
sorry

end senior_employee_bonus_l59_59748


namespace mark_fewer_than_susan_l59_59768

variable (apples_total : ℕ) (greg_apples : ℕ) (susan_apples : ℕ) (mark_apples : ℕ) (mom_apples : ℕ)

def evenly_split (total : ℕ) : ℕ := total / 2

theorem mark_fewer_than_susan
    (h1 : apples_total = 18)
    (h2 : greg_apples = evenly_split apples_total)
    (h3 : susan_apples = 2 * greg_apples)
    (h4 : mom_apples = 40 + 9)
    (h5 : mark_apples = mom_apples - susan_apples) :
    susan_apples - mark_apples = 13 := 
sorry

end mark_fewer_than_susan_l59_59768


namespace cylinder_height_l59_59554

   theorem cylinder_height (r h : ℝ) (SA : ℝ) (π : ℝ) :
     r = 3 → SA = 30 * π → SA = 2 * π * r^2 + 2 * π * r * h → h = 2 :=
   by
     intros hr hSA hSA_formula
     rw [hr] at hSA_formula
     rw [hSA] at hSA_formula
     sorry
   
end cylinder_height_l59_59554


namespace sum_x_coordinates_common_points_l59_59319

-- Definition of the equivalence relation modulo 9
def equiv_mod (a b n : ℤ) : Prop := ∃ k : ℤ, a = b + n * k

-- Definitions of the given conditions
def graph1 (x y : ℤ) : Prop := equiv_mod y (3 * x + 6) 9
def graph2 (x y : ℤ) : Prop := equiv_mod y (7 * x + 3) 9

-- Definition of when two graphs intersect
def points_in_common (x y : ℤ) : Prop := graph1 x y ∧ graph2 x y

-- Proof that the sum of the x-coordinates of the points in common is 3
theorem sum_x_coordinates_common_points : 
  ∃ x y, points_in_common x y ∧ (x = 3) := 
sorry

end sum_x_coordinates_common_points_l59_59319


namespace remainder_of_product_mod_seven_l59_59851

-- Definitions derived from the conditions
def seq : List ℕ := [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

-- The main statement to prove
theorem remainder_of_product_mod_seven : 
  (seq.foldl (λ acc x => acc * x) 1) % 7 = 0 := by
  sorry

end remainder_of_product_mod_seven_l59_59851


namespace Rohit_is_to_the_east_of_starting_point_l59_59523

-- Define the conditions and the problem statement.
def Rohit's_movements_proof
  (distance_south : ℕ) (distance_first_left : ℕ) (distance_second_left : ℕ) (distance_right : ℕ)
  (final_distance : ℕ) : Prop :=
  distance_south = 25 ∧
  distance_first_left = 20 ∧
  distance_second_left = 25 ∧
  distance_right = 15 ∧
  final_distance = 35 →
  (direction : String) → (distance : ℕ) →
  direction = "east" ∧ distance = final_distance

-- We can now state the theorem
theorem Rohit_is_to_the_east_of_starting_point :
  Rohit's_movements_proof 25 20 25 15 35 :=
by
  sorry

end Rohit_is_to_the_east_of_starting_point_l59_59523


namespace first_number_in_sum_l59_59241

theorem first_number_in_sum (a b c : ℝ) (h : a + b + c = 3.622) : a = 3.15 :=
by
  -- Assume the given values of b and c
  have hb : b = 0.014 := sorry
  have hc : c = 0.458 := sorry
  -- From the assumption h and hb, hc, we deduce a = 3.15
  sorry

end first_number_in_sum_l59_59241


namespace condition_for_a_l59_59100

theorem condition_for_a (a : ℝ) :
  (∀ x : ℤ, (x < 0 → (x + a) / 2 ≥ 1) → (x = -1 ∨ x = -2)) ↔ 4 ≤ a ∧ a < 5 :=
by
  sorry

end condition_for_a_l59_59100


namespace triangle_lengths_relationship_l59_59268

-- Given data
variables {a b c f_a f_b f_c t_a t_b t_c : ℝ}
-- Conditions/assumptions
variables (h1 : f_a * t_a = b * c)
variables (h2 : f_b * t_b = a * c)
variables (h3 : f_c * t_c = a * b)

-- Theorem to prove
theorem triangle_lengths_relationship :
  a^2 * b^2 * c^2 = f_a * f_b * f_c * t_a * t_b * t_c :=
by sorry

end triangle_lengths_relationship_l59_59268


namespace correct_operation_c_l59_59329

theorem correct_operation_c (a b : ℝ) :
  ¬ (a^2 + a^2 = 2 * a^4)
  ∧ ¬ ((-3 * a * b^2)^2 = -6 * a^2 * b^4)
  ∧ a^6 / (-a)^2 = a^4
  ∧ ¬ ((a - b)^2 = a^2 - b^2) :=
by
  sorry

end correct_operation_c_l59_59329


namespace true_proposition_l59_59898

variable (p : Prop) (q : Prop)

-- Introduce the propositions as Lean variables
def prop_p : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2
def prop_q : Prop := ∀ a b : ℝ, ((a > 1 ∧ b > 1) → a * b > 1) ∧ ((a * b > 1) ∧ (¬ (a > 1 ∧ b > 1)))

-- Rewrite the main goal as a Lean statement
theorem true_proposition : ¬ prop_p ∧ prop_q := 
  sorry

end true_proposition_l59_59898


namespace determine_var_phi_l59_59431

open Real

theorem determine_var_phi (φ : ℝ) (h₀ : 0 ≤ φ ∧ φ ≤ 2 * π) :
  (∀ x, sin (x + φ) = sin (x - π / 6)) → φ = 11 * π / 6 :=
by
  sorry

end determine_var_phi_l59_59431


namespace second_valve_emits_more_l59_59159

noncomputable def V1 : ℝ := 12000 / 120 -- Rate of first valve (100 cubic meters/minute)
noncomputable def V2 : ℝ := 12000 / 48 - V1 -- Rate of second valve

theorem second_valve_emits_more : V2 - V1 = 50 :=
by
  sorry

end second_valve_emits_more_l59_59159


namespace area_of_rectangle_l59_59177

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l59_59177


namespace work_done_by_first_group_l59_59587

theorem work_done_by_first_group :
  (6 * 8 * 5 : ℝ) / W = (4 * 3 * 8 : ℝ) / 30 →
  W = 75 :=
by
  sorry

end work_done_by_first_group_l59_59587


namespace value_of_f_two_l59_59949

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_two :
  (∀ x : ℝ, f (1 / x) = 1 / (x + 1)) → f 2 = 2 / 3 := by
  intro h
  -- The proof would go here
  sorry

end value_of_f_two_l59_59949


namespace probability_square_product_l59_59412

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_favorable_outcomes : ℕ :=
  List.length [(1, 1), (1, 4), (2, 2), (4, 1), (3, 3), (4, 4), (2, 8), (8, 2), (5, 5), (4, 9), (6, 6), (7, 7), (8, 8), (9, 9)]

def total_outcomes : ℕ := 12 * 8

theorem probability_square_product :
  (count_favorable_outcomes : ℚ) / (total_outcomes : ℚ) = (7 : ℚ) / (48 : ℚ) := 
by 
  sorry

end probability_square_product_l59_59412


namespace product_of_translated_roots_l59_59178

noncomputable def roots (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem product_of_translated_roots
  {d e : ℝ}
  (h_d : roots 3 4 (-7) d)
  (h_e : roots 3 4 (-7) e)
  (sum_roots : d + e = -4 / 3)
  (product_roots : d * e = -7 / 3) :
  (d - 1) * (e - 1) = 1 :=
by
  sorry

end product_of_translated_roots_l59_59178


namespace total_questions_needed_l59_59633

def m_total : ℕ := 35
def p_total : ℕ := 15
def t_total : ℕ := 20

def m_written : ℕ := (3 * m_total) / 7
def p_written : ℕ := p_total / 5
def t_written : ℕ := t_total / 4

def m_remaining : ℕ := m_total - m_written
def p_remaining : ℕ := p_total - p_written
def t_remaining : ℕ := t_total - t_written

def total_remaining : ℕ := m_remaining + p_remaining + t_remaining

theorem total_questions_needed : total_remaining = 47 := by
  sorry

end total_questions_needed_l59_59633


namespace simplify_eq_l59_59126

theorem simplify_eq {x y z : ℕ} (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  9 * (x : ℝ) - ((10 / (2 * y) / 3 + 7 * z) * Real.pi) =
  9 * (x : ℝ) - (5 * Real.pi / (3 * y) + 7 * z * Real.pi) := by
  sorry

end simplify_eq_l59_59126


namespace fourth_machine_works_for_12_hours_daily_l59_59704

noncomputable def hours_fourth_machine_works (m1_hours m1_production_rate: ℕ) (m2_hours m2_production_rate: ℕ) (price_per_kg: ℕ) (total_earning: ℕ) :=
  let m1_total_production := m1_hours * m1_production_rate
  let m1_total_output := 3 * m1_total_production
  let m1_revenue := m1_total_output * price_per_kg
  let remaining_revenue := total_earning - m1_revenue
  let m2_total_production := remaining_revenue / price_per_kg
  m2_total_production / m2_production_rate

theorem fourth_machine_works_for_12_hours_daily : hours_fourth_machine_works 23 2 (sorry) (sorry) 50 8100 = 12 := by
  sorry

end fourth_machine_works_for_12_hours_daily_l59_59704


namespace problem_I_problem_II_problem_III_l59_59163

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 3

theorem problem_I (a b : ℝ) (h_a : a = 0) :
  (b ≥ 0 → ∀ x : ℝ, 3 * x^2 + b ≥ 0) ∧
  (b < 0 → 
    ∀ x : ℝ, (x < -Real.sqrt (-b / 3) ∨ x > Real.sqrt (-b / 3)) → 
      3 * x^2 + b > 0) := sorry

theorem problem_II (b : ℝ) :
  ∃ x0 : ℝ, f x0 0 b = x0 ∧ (3 * x0^2 + b = 0) ↔ b = -3 := sorry

theorem problem_III :
  ∀ a b : ℝ, ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧
    (3 * x1^2 + 2 * a * x1 + b = 0) ∧
    (3 * x2^2 + 2 * a * x2 + b = 0) ∧
    (f x1 a b = x1) ∧
    (f x2 a b = x2)) := sorry

end problem_I_problem_II_problem_III_l59_59163


namespace fractions_comparison_l59_59707

theorem fractions_comparison : 
  (99 / 100 < 100 / 101) ∧ (100 / 101 > 199 / 201) ∧ (99 / 100 < 199 / 201) :=
by sorry

end fractions_comparison_l59_59707


namespace avg_growth_rate_first_brand_eq_l59_59139

noncomputable def avg_growth_rate_first_brand : ℝ :=
  let t := 5.647
  let first_brand_households_2001 := 4.9
  let second_brand_households_2001 := 2.5
  let second_brand_growth_rate := 0.7
  let equalization_time := t
  (second_brand_households_2001 + second_brand_growth_rate * equalization_time - first_brand_households_2001) / equalization_time

theorem avg_growth_rate_first_brand_eq :
  avg_growth_rate_first_brand = 0.275 := by
  sorry

end avg_growth_rate_first_brand_eq_l59_59139


namespace tangent_slope_is_4_l59_59690

theorem tangent_slope_is_4 (x y : ℝ) (h_curve : y = x^4) (h_slope : (deriv (fun x => x^4) x) = 4) :
    (x, y) = (1, 1) :=
by
  -- Place proof here
  sorry

end tangent_slope_is_4_l59_59690


namespace four_digit_arithmetic_sequence_l59_59564

theorem four_digit_arithmetic_sequence :
  ∃ (a b c d : ℕ), 1000 * a + 100 * b + 10 * c + d = 5555 ∨ 1000 * a + 100 * b + 10 * c + d = 2468 ∧
  (a + d = 10) ∧ (b + c = 10) ∧ (2 * b = a + c) ∧ (c - b = b - a) ∧ (d - c = c - b) ∧
  (1000 * d + 100 * c + 10 * b + a + 1000 * a + 100 * b + 10 * c + d = 11110) :=
sorry

end four_digit_arithmetic_sequence_l59_59564


namespace students_more_than_pets_l59_59695

theorem students_more_than_pets
  (students_per_classroom : ℕ)
  (rabbits_per_classroom : ℕ)
  (birds_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (total_students : ℕ)
  (total_rabbits : ℕ)
  (total_birds : ℕ)
  (total_pets : ℕ)
  (difference : ℕ)
  : students_per_classroom = 22 → 
    rabbits_per_classroom = 3 → 
    birds_per_classroom = 2 → 
    number_of_classrooms = 5 → 
    total_students = students_per_classroom * number_of_classrooms → 
    total_rabbits = rabbits_per_classroom * number_of_classrooms → 
    total_birds = birds_per_classroom * number_of_classrooms → 
    total_pets = total_rabbits + total_birds → 
    difference = total_students - total_pets →
    difference = 85 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end students_more_than_pets_l59_59695


namespace sampling_methods_correct_l59_59600

def first_method_sampling : String :=
  "Simple random sampling"

def second_method_sampling : String :=
  "Systematic sampling"

theorem sampling_methods_correct :
  first_method_sampling = "Simple random sampling" ∧ second_method_sampling = "Systematic sampling" :=
by
  sorry

end sampling_methods_correct_l59_59600


namespace jeremy_school_distance_l59_59560

theorem jeremy_school_distance (d : ℝ) (v : ℝ) :
  (d = v * 0.5) ∧
  (d = (v + 15) * 0.3) ∧
  (d = (v - 10) * (2 / 3)) →
  d = 15 :=
by 
  sorry

end jeremy_school_distance_l59_59560


namespace car_overtakes_buses_l59_59156

/-- 
  Buses leave the airport every 3 minutes. 
  A bus takes 60 minutes to travel from the airport to the city center. 
  A car takes 35 minutes to travel from the airport to the city center. 
  Prove that the car overtakes 8 buses on its way to the city center excluding the bus it left with.
--/
theorem car_overtakes_buses (arr_bus : ℕ) (arr_car : ℕ) (interval : ℕ) (diff : ℕ) : 
  interval = 3 → arr_bus = 60 → arr_car = 35 → diff = arr_bus - arr_car →
  ∃ n : ℕ, n = diff / interval ∧ n = 8 := by
  sorry

end car_overtakes_buses_l59_59156


namespace people_per_table_l59_59132

def total_people_invited : ℕ := 68
def people_who_didn't_show_up : ℕ := 50
def number_of_tables_needed : ℕ := 6

theorem people_per_table (total_people_invited people_who_didn't_show_up number_of_tables_needed : ℕ) : 
  total_people_invited - people_who_didn't_show_up = 18 ∧
  (total_people_invited - people_who_didn't_show_up) / number_of_tables_needed = 3 :=
by
  sorry

end people_per_table_l59_59132


namespace solve_for_a_l59_59062

-- Given conditions
def x : ℕ := 2
def y : ℕ := 2
def equation (a : ℚ) : Prop := a * x + y = 5

-- Our goal is to prove that "a = 3/2" given the conditions
theorem solve_for_a : ∃ a : ℚ, equation a ∧ a = 3 / 2 :=
by
  sorry

end solve_for_a_l59_59062


namespace max_min_x2_sub_xy_add_y2_l59_59433

/-- Given a point \((x, y)\) on the curve defined by \( |5x + y| + |5x - y| = 20 \), prove that the maximum value of \(x^2 - xy + y^2\) is 124 and the minimum value is 3. -/
theorem max_min_x2_sub_xy_add_y2 (x y : ℝ) (h : abs (5 * x + y) + abs (5 * x - y) = 20) :
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end max_min_x2_sub_xy_add_y2_l59_59433


namespace tony_average_time_to_store_l59_59254

-- Definitions and conditions
def speed_walking := 2  -- MPH
def speed_running := 10  -- MPH
def distance_to_store := 4  -- miles
def days_walking := 1  -- Sunday
def days_running := 2  -- Tuesday, Thursday
def total_days := days_walking + days_running

-- Proof statement
theorem tony_average_time_to_store :
  let time_walking := (distance_to_store / speed_walking) * 60
  let time_running := (distance_to_store / speed_running) * 60
  let total_time := time_walking * days_walking + time_running * days_running
  let average_time := total_time / total_days
  average_time = 56 :=
by
  sorry  -- Proof to be completed

end tony_average_time_to_store_l59_59254


namespace point_in_third_quadrant_l59_59089

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 :=
by
  sorry

end point_in_third_quadrant_l59_59089


namespace exists_t_perpendicular_min_dot_product_coordinates_l59_59492

-- Definitions of points
def OA : ℝ × ℝ := (5, 1)
def OB : ℝ × ℝ := (1, 7)
def OC : ℝ × ℝ := (4, 2)

-- Definition of vector OM depending on t
def OM (t : ℝ) : ℝ × ℝ := (4 * t, 2 * t)

-- Definition of vector MA and MB
def MA (t : ℝ) : ℝ × ℝ := (5 - 4 * t, 1 - 2 * t)
def MB (t : ℝ) : ℝ × ℝ := (1 - 4 * t, 7 - 2 * t)

-- Dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Proof that there exists a t such that MA ⊥ MB
theorem exists_t_perpendicular : ∃ t : ℝ, dot_product (MA t) (MB t) = 0 :=
by 
  sorry

-- Proof that coordinates of M minimizing MA ⋅ MB is (4, 2)
theorem min_dot_product_coordinates : ∃ t : ℝ, t = 1 ∧ (OM t) = (4, 2) :=
by
  sorry

end exists_t_perpendicular_min_dot_product_coordinates_l59_59492


namespace polynomial_roots_l59_59201

theorem polynomial_roots :
  (∃ x : ℝ, x^4 - 16*x^3 + 91*x^2 - 216*x + 180 = 0) ↔ (x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 6) := 
sorry

end polynomial_roots_l59_59201


namespace charlie_more_apples_than_bella_l59_59347

variable (D : ℝ) 

theorem charlie_more_apples_than_bella 
    (hC : C = 1.75 * D)
    (hB : B = 1.50 * D) :
    (C - B) / B = 0.1667 := 
by
  sorry

end charlie_more_apples_than_bella_l59_59347


namespace range_of_m_l59_59649

theorem range_of_m {m : ℝ} (h : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ x^2 + 2 * x - m = 0) : 8 < m ∧ m < 15 :=
sorry

end range_of_m_l59_59649


namespace greatest_multiple_of_four_l59_59165

theorem greatest_multiple_of_four (x : ℕ) (hx : x > 0) (h4 : x % 4 = 0) (hcube : x^3 < 800) : x ≤ 8 :=
by {
  sorry
}

end greatest_multiple_of_four_l59_59165


namespace num_sheets_in_stack_l59_59742

-- Definitions coming directly from the conditions
def thickness_ream := 4 -- cm
def num_sheets_ream := 400
def height_stack := 10 -- cm

-- The final proof statement
theorem num_sheets_in_stack : (height_stack / (thickness_ream / num_sheets_ream)) = 1000 :=
by
  sorry

end num_sheets_in_stack_l59_59742


namespace mean_second_set_l59_59685

theorem mean_second_set (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 62) :
  (48 + 62 + 98 + 124 + x) / 5 = 78 :=
sorry

end mean_second_set_l59_59685


namespace find_f_of_3_l59_59876

theorem find_f_of_3 (a b c : ℝ) (f : ℝ → ℝ) (h1 : f 1 = 7) (h2 : f 2 = 12) (h3 : ∀ x, f x = ax + bx + c) : f 3 = 17 :=
by
  sorry

end find_f_of_3_l59_59876


namespace small_triangle_perimeter_l59_59743

theorem small_triangle_perimeter (P : ℕ) (P₁ : ℕ) (P₂ : ℕ) (P₃ : ℕ)
  (h₁ : P = 11) (h₂ : P₁ = 5) (h₃ : P₂ = 7) (h₄ : P₃ = 9) :
  (P₁ + P₂ + P₃) - P = 10 :=
by
  sorry

end small_triangle_perimeter_l59_59743


namespace sum_is_1716_l59_59308

-- Given conditions:
variables (a b c d : ℤ)
variable (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
variable (h_roots1 : ∀ t, t * t - 12 * a * t - 13 * b = 0 ↔ t = c ∨ t = d)
variable (h_roots2 : ∀ t, t * t - 12 * c * t - 13 * d = 0 ↔ t = a ∨ t = b)

-- Prove the desired sum of the constants:
theorem sum_is_1716 : a + b + c + d = 1716 :=
by
  sorry

end sum_is_1716_l59_59308


namespace initial_average_mark_l59_59008

theorem initial_average_mark (A : ℕ) (A_excluded : ℕ := 20) (A_remaining : ℕ := 90) (n_total : ℕ := 14) (n_excluded : ℕ := 5) :
    (n_total * A = n_excluded * A_excluded + (n_total - n_excluded) * A_remaining) → A = 65 :=
by 
  intros h
  sorry

end initial_average_mark_l59_59008


namespace range_of_a_l59_59592

-- Definitions of the sets U and A
def U := {x : ℝ | 0 < x ∧ x < 9}
def A (a : ℝ) := {x : ℝ | 1 < x ∧ x < a}

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) (H_non_empty : A a ≠ ∅) (H_not_subset : ¬ ∀ x, x ∈ A a → x ∈ U) : 
  1 < a ∧ a ≤ 9 :=
sorry

end range_of_a_l59_59592


namespace cubic_roots_equal_l59_59135

theorem cubic_roots_equal (k : ℚ) (h1 : k > 0)
  (h2 : ∃ a b : ℚ, a ≠ b ∧ (a + a + b = -3) ∧ (2 * a * b + a^2 = -54) ∧ (3 * x^3 + 9 * x^2 - 162 * x + k = 0)) : 
  k = 7983 / 125 :=
sorry

end cubic_roots_equal_l59_59135


namespace product_eq_sum_l59_59326

variables {x y : ℝ}

theorem product_eq_sum (h : x * y = x + y) (h_ne : y ≠ 1) : x = y / (y - 1) :=
sorry

end product_eq_sum_l59_59326


namespace tangent_curve_line_l59_59930

/-- Given the line y = x + 1 and the curve y = ln(x + a) are tangent, prove that the value of a is 2. -/
theorem tangent_curve_line (a : ℝ) :
  (∃ x₀ y₀, y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧ (1 / (x₀ + a) = 1)) → a = 2 :=
by
  sorry

end tangent_curve_line_l59_59930


namespace men_work_equivalence_l59_59697

theorem men_work_equivalence : 
  ∀ (M : ℕ) (m w : ℕ),
  (3 * w = 2 * m) ∧ 
  (M * 21 * 8 * m = 21 * 60 * 3 * w) →
  M = 15 := by
  intro M m w
  intro h
  sorry

end men_work_equivalence_l59_59697


namespace equation_of_parallel_line_l59_59589

theorem equation_of_parallel_line (x y : ℝ) :
  (∀ b : ℝ, 2 * x + y + b = 0 → x = -1 → y = 2 → b = 0) →
  (∀ x y b: ℝ, 2 * x + y + b = 0 → x = -1 → y = 2 → 2 * x + y = 0) :=
by
  sorry

end equation_of_parallel_line_l59_59589


namespace distance_to_campground_l59_59303

-- definitions for speeds and times
def speed1 : ℤ := 50
def time1 : ℤ := 3
def speed2 : ℤ := 60
def time2 : ℤ := 2
def speed3 : ℤ := 55
def time3 : ℤ := 1
def speed4 : ℤ := 65
def time4 : ℤ := 2

-- definitions for calculating the distances
def distance1 : ℤ := speed1 * time1
def distance2 : ℤ := speed2 * time2
def distance3 : ℤ := speed3 * time3
def distance4 : ℤ := speed4 * time4

-- definition for the total distance
def total_distance : ℤ := distance1 + distance2 + distance3 + distance4

-- proof statement
theorem distance_to_campground : total_distance = 455 := by
  sorry -- proof omitted

end distance_to_campground_l59_59303


namespace minimum_value_of_f_l59_59894

noncomputable def f (x a : ℝ) := (1/3) * x^3 + (a-1) * x^2 - 4 * a * x + a

theorem minimum_value_of_f (a : ℝ) (h : a < -1) :
  (if -3/2 < a then ∀ (x : ℝ), 2 ≤ x ∧ x ≤ 3 → f x a ≥ f (-2*a) a
   else ∀ (x : ℝ), 2 ≤ x ∧ x ≤ 3 → f x a ≥ f 3 a) :=
sorry

end minimum_value_of_f_l59_59894


namespace trig_identity_l59_59963

theorem trig_identity (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 2 / 3) : 
  Real.cos (2 * α + Real.pi / 3) = -1 / 9 :=
by
  sorry

end trig_identity_l59_59963


namespace jane_last_day_vases_l59_59557

def vasesPerDay : Nat := 16
def totalVases : Nat := 248

theorem jane_last_day_vases : totalVases % vasesPerDay = 8 := by
  sorry

end jane_last_day_vases_l59_59557


namespace race_outcomes_l59_59959

-- Definition of participants
inductive Participant
| Abe 
| Bobby
| Charles
| Devin
| Edwin
| Frank
deriving DecidableEq

open Participant

def num_participants : ℕ := 6

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Proving the number of different 1st-2nd-3rd outcomes
theorem race_outcomes : factorial 6 / factorial 3 = 120 := by
  sorry

end race_outcomes_l59_59959


namespace ratio_of_speeds_is_two_l59_59006

noncomputable def joe_speed : ℝ := 0.266666666667
noncomputable def time : ℝ := 40
noncomputable def total_distance : ℝ := 16

noncomputable def joe_distance : ℝ := joe_speed * time
noncomputable def pete_distance : ℝ := total_distance - joe_distance
noncomputable def pete_speed : ℝ := pete_distance / time

theorem ratio_of_speeds_is_two :
  joe_speed / pete_speed = 2 := by
  sorry

end ratio_of_speeds_is_two_l59_59006


namespace units_digit_6_pow_4_l59_59430

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Define the main theorem to prove
theorem units_digit_6_pow_4 : units_digit (6 ^ 4) = 6 := 
by
  sorry

end units_digit_6_pow_4_l59_59430


namespace all_integers_appear_exactly_once_l59_59351

noncomputable def sequence_of_integers (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ m : ℕ, a m > 0 ∧ ∃ m' : ℕ, a m' < 0

noncomputable def distinct_modulo_n (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, (∀ i j : ℕ, i < j ∧ j < n → a i % n ≠ a j % n)

theorem all_integers_appear_exactly_once
  (a : ℕ → ℤ)
  (h_seq : sequence_of_integers a)
  (h_distinct : distinct_modulo_n a) :
  ∀ x : ℤ, ∃! i : ℕ, a i = x := 
sorry

end all_integers_appear_exactly_once_l59_59351


namespace pete_miles_walked_l59_59918

-- Define the conditions
def maxSteps := 99999
def numFlips := 50
def finalReading := 25000
def stepsPerMile := 1500

-- Proof statement that Pete walked 3350 miles
theorem pete_miles_walked : 
  (numFlips * (maxSteps + 1) + finalReading) / stepsPerMile = 3350 := 
by 
  sorry

end pete_miles_walked_l59_59918


namespace length_of_base_of_vessel_l59_59121

noncomputable def volume_of_cube (edge : ℝ) := edge ^ 3

theorem length_of_base_of_vessel 
  (cube_edge : ℝ)
  (vessel_width : ℝ)
  (rise_in_water_level : ℝ)
  (volume_cube : ℝ)
  (h1 : cube_edge = 15)
  (h2 : vessel_width = 15)
  (h3 : rise_in_water_level = 11.25)
  (h4 : volume_cube = volume_of_cube cube_edge)
  : ∃ L : ℝ, L = volume_cube / (vessel_width * rise_in_water_level) ∧ L = 20 :=
by
  sorry

end length_of_base_of_vessel_l59_59121


namespace pants_cost_l59_59411

def total_cost (P : ℕ) : ℕ := 4 * 8 + 2 * 60 + 2 * P

theorem pants_cost :
  (∃ P : ℕ, total_cost P = 188) →
  ∃ P : ℕ, P = 18 :=
by
  intro h
  sorry

end pants_cost_l59_59411


namespace number_of_ways_to_choose_bases_l59_59653

theorem number_of_ways_to_choose_bases : ∀ (students bases : ℕ), students = 4 → bases = 4 → (bases^students) = 256 :=
by
  intros students bases h_students h_bases
  rw [h_students, h_bases]
  exact pow_succ' 4 3

end number_of_ways_to_choose_bases_l59_59653


namespace inequality_4th_power_l59_59253

theorem inequality_4th_power (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_ineq : a ≥ b) :
  (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 :=
sorry

end inequality_4th_power_l59_59253


namespace number_drawn_from_first_group_l59_59762

theorem number_drawn_from_first_group (n: ℕ) (groups: ℕ) (interval: ℕ) (fourth_group_number: ℕ) (total_bags: ℕ) 
    (h1: total_bags = 50) (h2: groups = 5) (h3: interval = total_bags / groups)
    (h4: interval = 10) (h5: fourth_group_number = 36) : n = 6 :=
by
  sorry

end number_drawn_from_first_group_l59_59762


namespace ticket_cost_calculation_l59_59187

theorem ticket_cost_calculation :
  let adult_price := 12
  let child_price := 10
  let num_adults := 3
  let num_children := 3
  let total_cost := (num_adults * adult_price) + (num_children * child_price)
  total_cost = 66 := 
by
  rfl -- or add sorry to skip proof

end ticket_cost_calculation_l59_59187


namespace romanov_family_savings_l59_59661

theorem romanov_family_savings :
  let cost_multi_tariff_meter := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let night_consumption := 230
  let day_consumption := monthly_consumption - night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let yearly_cost_multi_tariff :=
    (night_consumption * night_rate * 12) +
    (day_consumption * day_rate * 12)
  let total_cost_multi_tariff :=
    cost_multi_tariff_meter + installation_cost + (yearly_cost_multi_tariff * 3)
  let yearly_cost_standard :=
    monthly_consumption * standard_rate * 12
  let total_cost_standard :=
    yearly_cost_standard * 3
  total_cost_standard - total_cost_multi_tariff = 3824 := 
by {
  sorry -- Proof goes here
}

end romanov_family_savings_l59_59661


namespace minimum_discount_l59_59288

variable (C P : ℝ) (r x : ℝ)

def microwave_conditions := 
  C = 1000 ∧ 
  P = 1500 ∧ 
  r = 0.02 ∧ 
  P * (x / 10) ≥ C * (1 + r)

theorem minimum_discount : ∃ x, microwave_conditions C P r x ∧ x ≥ 6.8 :=
by 
  sorry

end minimum_discount_l59_59288


namespace total_amount_spent_l59_59075

-- Define the prices related to John's Star Wars toy collection
def other_toys_cost : ℕ := 1000
def lightsaber_cost : ℕ := 2 * other_toys_cost

-- Problem statement in Lean: Prove the total amount spent is $3000
theorem total_amount_spent : (other_toys_cost + lightsaber_cost) = 3000 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end total_amount_spent_l59_59075


namespace installation_quantities_l59_59670

theorem installation_quantities :
  ∃ x1 x2 x3 : ℕ, x1 = 22 ∧ x2 = 88 ∧ x3 = 22 ∧
  (x1 + x2 + x3 ≥ 100) ∧
  (x2 = 4 * x1) ∧
  (∃ k : ℕ, x3 = k * x1) ∧
  (5 * x3 = x2 + 22) :=
  by {
    -- We are simply stating the equivalence and supporting conditions.
    -- Here, we will use 'sorry' as a placeholder.
    sorry
  }

end installation_quantities_l59_59670


namespace john_has_388_pennies_l59_59175

theorem john_has_388_pennies (k : ℕ) (j : ℕ) (hk : k = 223) (hj : j = k + 165) : j = 388 := by
  sorry

end john_has_388_pennies_l59_59175


namespace find_digits_sum_l59_59339

theorem find_digits_sum (A B : ℕ) (h1 : A < 10) (h2 : B < 10) 
  (h3 : (A = 6) ∧ (B = 6))
  (h4 : (100 * A + 44610 + B) % 72 = 0) : A + B = 12 := 
by
  sorry

end find_digits_sum_l59_59339


namespace find_sum_of_min_area_ks_l59_59195

def point := ℝ × ℝ

def A : point := (2, 9)
def B : point := (14, 18)

def is_int (k : ℝ) : Prop := ∃ (n : ℤ), k = n

def min_triangle_area (P Q R : point) : ℝ := sorry
-- Placeholder for the area formula of a triangle given three points

def valid_ks (k : ℝ) : Prop :=
  is_int k ∧ min_triangle_area A B (6, k) ≠ 0

theorem find_sum_of_min_area_ks :
  (∃ k1 k2 : ℤ, valid_ks k1 ∧ valid_ks k2 ∧ (k1 + k2) = 31) :=
sorry

end find_sum_of_min_area_ks_l59_59195


namespace calculate_expr_l59_59260

theorem calculate_expr : (2023^0 + (-1/3) = 2/3) := by
  sorry

end calculate_expr_l59_59260


namespace building_height_l59_59702

theorem building_height (flagpole_height : ℝ) (flagpole_shadow : ℝ) 
  (building_shadow : ℝ) (building_height : ℝ)
  (h_flagpole : flagpole_height = 18)
  (s_flagpole : flagpole_shadow = 45)
  (s_building : building_shadow = 70)
  (ratio_eq : flagpole_height / flagpole_shadow = building_height / building_shadow) :
  building_height = 28 :=
by
  have h_flagpole_shadow := ratio_eq ▸ h_flagpole ▸ s_flagpole ▸ s_building
  sorry

end building_height_l59_59702


namespace third_student_gold_stickers_l59_59449

theorem third_student_gold_stickers:
  ∃ (n : ℕ), n = 41 ∧ 
  (∃ (a1 a2 a4 a5 a6 : ℕ), 
    a1 = 29 ∧ 
    a2 = 35 ∧ 
    a4 = 47 ∧ 
    a5 = 53 ∧ 
    a6 = 59 ∧ 
    a2 - a1 = 6 ∧ 
    a5 - a4 = 6 ∧ 
    ∀ k, k = 3 → n = a2 + 6) := 
sorry

end third_student_gold_stickers_l59_59449


namespace total_blankets_collected_l59_59508

theorem total_blankets_collected : 
  let original_members := 15
  let new_members := 5
  let blankets_per_original_member_first_day := 2
  let blankets_per_original_member_second_day := 2
  let blankets_per_new_member_second_day := 4
  let tripled_first_day_total := 3
  let blankets_school_third_day := 22
  let blankets_online_third_day := 30
  let first_day_blankets := original_members * blankets_per_original_member_first_day
  let second_day_original_members_blankets := original_members * blankets_per_original_member_second_day
  let second_day_new_members_blankets := new_members * blankets_per_new_member_second_day
  let second_day_additional_blankets := tripled_first_day_total * first_day_blankets
  let second_day_blankets := second_day_original_members_blankets + second_day_new_members_blankets + second_day_additional_blankets
  let third_day_blankets := blankets_school_third_day + blankets_online_third_day
  let total_blankets := first_day_blankets + second_day_blankets + third_day_blankets
  -- Prove that
  total_blankets = 222 :=
by 
  sorry

end total_blankets_collected_l59_59508


namespace algae_cells_count_10_days_l59_59456

-- Define the initial condition where the pond starts with one algae cell.
def initial_algae_cells : ℕ := 1

-- Define the daily splitting of each cell into 3 new cells.
def daily_split (cells : ℕ) : ℕ := cells * 3

-- Define the function to compute the number of algae cells after n days.
def algae_cells_after_days (n : ℕ) : ℕ :=
  initial_algae_cells * (3 ^ n)

-- State the theorem to be proved.
theorem algae_cells_count_10_days : algae_cells_after_days 10 = 59049 :=
by {
  sorry
}

end algae_cells_count_10_days_l59_59456


namespace common_difference_l59_59755

theorem common_difference (a1 d : ℕ) (S3 : ℕ) (h1 : S3 = 6) (h2 : a1 = 1)
  (h3 : S3 = 3 * (2 * a1 + 2 * d) / 2) : d = 1 :=
by
  sorry

end common_difference_l59_59755


namespace largest_number_in_ratio_l59_59558

theorem largest_number_in_ratio (x : ℕ) (h : ((4 * x + 5 * x + 6 * x) / 3 : ℝ) = 20) : 6 * x = 24 := 
by 
  sorry

end largest_number_in_ratio_l59_59558


namespace norma_total_cards_l59_59090

theorem norma_total_cards (initial_cards : ℝ) (additional_cards : ℝ) (total_cards : ℝ) 
  (h1 : initial_cards = 88) (h2 : additional_cards = 70) : total_cards = 158 :=
by
  sorry

end norma_total_cards_l59_59090


namespace gcf_of_294_and_108_l59_59287

theorem gcf_of_294_and_108 : Nat.gcd 294 108 = 6 :=
by
  -- We are given numbers 294 and 108
  -- Their prime factorizations are 294 = 2 * 3 * 7^2 and 108 = 2^2 * 3^3
  -- The minimum power of the common prime factors are 2^1 and 3^1
  -- Thus, the GCF by multiplying these factors is 2^1 * 3^1 = 6
  sorry

end gcf_of_294_and_108_l59_59287


namespace bob_total_distance_traveled_over_six_days_l59_59481

theorem bob_total_distance_traveled_over_six_days (x : ℤ) (hx1 : 3 ≤ x) (hx2 : x % 3 = 0):
  (90 / x + 90 / (x + 3) + 90 / (x + 6) + 90 / (x + 9) + 90 / (x + 12) + 90 / (x + 15) : ℝ) = 73.5 :=
by
  sorry

end bob_total_distance_traveled_over_six_days_l59_59481


namespace magnitude_of_difference_between_roots_l59_59696

variable (α β m : ℝ)

theorem magnitude_of_difference_between_roots
    (hαβ_root : ∀ x, x^2 - 2 * m * x + m^2 - 4 = 0 → (x = α ∨ x = β)) :
    |α - β| = 4 := by
  sorry

end magnitude_of_difference_between_roots_l59_59696


namespace speed_conversion_l59_59530

theorem speed_conversion (s : ℝ) (h1 : s = 1 / 3) : s * 3.6 = 1.2 := by
  -- Proof follows from the conditions given
  sorry

end speed_conversion_l59_59530


namespace exists_subset_sum_mod_p_l59_59818

theorem exists_subset_sum_mod_p (p : ℕ) (hp : Nat.Prime p) (A : Finset ℕ)
  (hA_card : A.card = p - 1) (hA : ∀ a ∈ A, a % p ≠ 0) : 
  ∀ n : ℕ, n < p → ∃ B ⊆ A, (B.sum id) % p = n :=
by
  sorry

end exists_subset_sum_mod_p_l59_59818


namespace min_boat_trips_l59_59240
-- Import Mathlib to include necessary libraries

-- Define the problem using noncomputable theory if necessary
theorem min_boat_trips (students boat_capacity : ℕ) (h1 : students = 37) (h2 : boat_capacity = 5) : ∃ x : ℕ, x ≥ 9 :=
by
  -- Here we need to prove the assumption and goal, hence adding sorry
  sorry

end min_boat_trips_l59_59240


namespace six_coins_not_sum_to_14_l59_59848

def coin_values : Set ℕ := {1, 5, 10, 25}

theorem six_coins_not_sum_to_14 (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 ∈ coin_values) (h2 : a2 ∈ coin_values) (h3 : a3 ∈ coin_values) (h4 : a4 ∈ coin_values) (h5 : a5 ∈ coin_values) (h6 : a6 ∈ coin_values) : a1 + a2 + a3 + a4 + a5 + a6 ≠ 14 := 
sorry

end six_coins_not_sum_to_14_l59_59848


namespace grill_run_time_l59_59133

-- Definitions of conditions
def coals_burned_per_minute : ℕ := 15
def minutes_per_coal_burned : ℕ := 20
def coals_per_bag : ℕ := 60
def bags_burned : ℕ := 3

-- Theorems to prove the question
theorem grill_run_time (coals_burned_per_minute: ℕ) (minutes_per_coal_burned: ℕ) (coals_per_bag: ℕ) (bags_burned: ℕ): (coals_burned_per_minute * (minutes_per_coal_burned * bags_burned * coals_per_bag / (coals_burned_per_minute * coals_per_bag))) / 60 = 4 := 
by 
  -- Lean statement skips detailed proof steps for conciseness
  sorry

end grill_run_time_l59_59133


namespace poster_distance_from_wall_end_l59_59866

theorem poster_distance_from_wall_end (w_wall w_poster : ℝ) (h1 : w_wall = 25) (h2 : w_poster = 4) (h3 : 2 * x + w_poster = w_wall) : x = 10.5 :=
by
  sorry

end poster_distance_from_wall_end_l59_59866


namespace greg_distance_work_to_market_l59_59457

-- Given conditions translated into definitions
def total_distance : ℝ := 40
def time_from_market_to_home : ℝ := 0.5  -- in hours
def speed_from_market_to_home : ℝ := 20  -- in miles per hour

-- Distance calculation from farmer's market to home
def distance_from_market_to_home := speed_from_market_to_home * time_from_market_to_home

-- Definition for the distance from workplace to the farmer's market
def distance_from_work_to_market := total_distance - distance_from_market_to_home

-- The theorem to be proved
theorem greg_distance_work_to_market : distance_from_work_to_market = 30 := by
  -- Skipping the detailed proof
  sorry

end greg_distance_work_to_market_l59_59457


namespace initial_parking_hours_proof_l59_59673

noncomputable def initial_parking_hours (total_cost : ℝ) (excess_hourly_rate : ℝ) (average_cost : ℝ) (total_hours : ℕ) : ℝ :=
  let h := (total_hours * average_cost - total_cost) / excess_hourly_rate
  h

theorem initial_parking_hours_proof : initial_parking_hours 21.25 1.75 2.361111111111111 9 = 2 :=
by
  sorry

end initial_parking_hours_proof_l59_59673


namespace area_of_quadrilateral_centroids_l59_59183

noncomputable def square_side_length : ℝ := 40
noncomputable def point_Q_XQ : ℝ := 15
noncomputable def point_Q_YQ : ℝ := 35

theorem area_of_quadrilateral_centroids (h1 : square_side_length = 40)
    (h2 : point_Q_XQ = 15)
    (h3 : point_Q_YQ = 35) :
    ∃ (area : ℝ), area = 800 / 9 :=
by
  sorry

end area_of_quadrilateral_centroids_l59_59183


namespace power_eval_l59_59897

theorem power_eval : (9^6 * 3^4) / (27^5) = 3 := by
  sorry

end power_eval_l59_59897


namespace ratio_evaluation_l59_59447

theorem ratio_evaluation :
  (10 ^ 2003 + 10 ^ 2001) / (2 * 10 ^ 2002) = 101 / 20 := 
by sorry

end ratio_evaluation_l59_59447


namespace lamp_probability_l59_59607

theorem lamp_probability (rope_length : ℝ) (pole_distance : ℝ) (h_pole_distance : pole_distance = 8) :
  let lamp_range := 2
  let favorable_segment_length := 4
  let total_rope_length := rope_length
  let probability := (favorable_segment_length / total_rope_length)
  rope_length = 8 → probability = 1 / 2 :=
by
  intros
  sorry

end lamp_probability_l59_59607


namespace expression_incorrect_l59_59416

theorem expression_incorrect (x : ℝ) : 5 * (x + 7) ≠ 5 * x + 7 := 
by 
  sorry

end expression_incorrect_l59_59416


namespace tangerines_count_l59_59428

theorem tangerines_count (apples pears tangerines : ℕ)
  (h1 : apples = 45)
  (h2 : pears = apples - 21)
  (h3 : tangerines = pears + 18) :
  tangerines = 42 :=
by
  sorry

end tangerines_count_l59_59428


namespace monomial_addition_l59_59391

-- Definition of a monomial in Lean
def isMonomial (p : ℕ → ℝ) : Prop := ∃ c n, ∀ x, p x = c * x^n

theorem monomial_addition (A : ℕ → ℝ) :
  (isMonomial (fun x => -3 * x + A x)) → isMonomial A :=
sorry

end monomial_addition_l59_59391


namespace lewis_weekly_earning_l59_59016

theorem lewis_weekly_earning
  (weeks : ℕ)
  (weekly_rent : ℤ)
  (total_savings : ℤ)
  (h1 : weeks = 1181)
  (h2 : weekly_rent = 216)
  (h3 : total_savings = 324775)
  : ∃ (E : ℤ), E = 49075 / 100 :=
by
  let E := 49075 / 100
  use E
  sorry -- The proof would go here

end lewis_weekly_earning_l59_59016


namespace prove_value_of_expression_l59_59550

theorem prove_value_of_expression (x : ℝ) (h : 10000 * x + 2 = 4) : 5000 * x + 1 = 2 :=
by 
  sorry

end prove_value_of_expression_l59_59550


namespace sum_of_first_five_primes_with_units_digit_3_l59_59957

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l59_59957


namespace minimum_value_l59_59961

variable (a b : ℝ)
variable (ab_nonzero : a ≠ 0 ∧ b ≠ 0)
variable (circle1 : ∀ x y, x^2 + y^2 + 2 * a * x + a^2 - 9 = 0)
variable (circle2 : ∀ x y, x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0)
variable (centers_distance : a^2 + 4 * b^2 = 16)

theorem minimum_value :
  (4 / a^2 + 1 / b^2) = 1 := sorry

end minimum_value_l59_59961


namespace part1_part2_l59_59421

-- Definitions for the sets A and B
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Proof statement for the first part
theorem part1 (m : ℝ) (h : m = 4) : A ∪ B m = { x | -2 ≤ x ∧ x ≤ 7 } :=
sorry

-- Proof statement for the second part
theorem part2 (h : ∀ {m : ℝ}, B m ⊆ A) : ∀ m : ℝ, m ∈ Set.Iic 3 :=
sorry

end part1_part2_l59_59421


namespace solve_inequality_l59_59315

open Real

theorem solve_inequality (a : ℝ) :
  ((a < 0 ∨ a > 1) → (∀ x, a < x ∧ x < a^2 ↔ (x - a) * (x - a^2) < 0)) ∧
  ((0 < a ∧ a < 1) → (∀ x, a^2 < x ∧ x < a ↔ (x - a) * (x - a^2) < 0)) ∧
  ((a = 0 ∨ a = 1) → (∀ x, ¬((x - a) * (x - a^2) < 0))) :=
by
  sorry

end solve_inequality_l59_59315


namespace YaoMing_stride_impossible_l59_59295

-- Defining the conditions as Lean definitions.
def XiaoMing_14_years_old (current_year : ℕ) : Prop := current_year = 14
def sum_of_triangle_angles (angles : ℕ) : Prop := angles = 180
def CCTV5_broadcasting_basketball_game : Prop := ∃ t : ℕ, true -- Random event placeholder
def YaoMing_stride (stride_length : ℕ) : Prop := stride_length = 10

-- The main statement: Prove that Yao Ming cannot step 10 meters in one stride.
theorem YaoMing_stride_impossible (h1: ∃ y : ℕ, XiaoMing_14_years_old y) 
                                  (h2: ∃ a : ℕ, sum_of_triangle_angles a) 
                                  (h3: CCTV5_broadcasting_basketball_game) 
: ¬ ∃ s : ℕ, YaoMing_stride s := sorry

end YaoMing_stride_impossible_l59_59295


namespace distinct_digit_numbers_count_l59_59784

def numDistinctDigitNumbers : Nat := 
  let first_digit_choices := 10
  let second_digit_choices := 9
  let third_digit_choices := 8
  let fourth_digit_choices := 7
  first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices

theorem distinct_digit_numbers_count : numDistinctDigitNumbers = 5040 :=
by
  sorry

end distinct_digit_numbers_count_l59_59784


namespace problem1_problem2_l59_59967

-- Definitions of sets A and B
def A : Set ℝ := { x | x > 1 }
def B (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

-- Problem 1:
theorem problem1 (a : ℝ) : B a ⊆ A → 1 ≤ a :=
  sorry

-- Problem 2:
theorem problem2 (a : ℝ) : (A ∩ B a).Nonempty → 0 < a :=
  sorry

end problem1_problem2_l59_59967


namespace binomial_equality_l59_59819

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l59_59819


namespace expression_value_at_neg3_l59_59455

theorem expression_value_at_neg3 (p q : ℤ) (h : 27 * p + 3 * q = 14) :
  (p * (-3)^3 + q * (-3) - 1) = -15 :=
sorry

end expression_value_at_neg3_l59_59455


namespace coo_coo_count_correct_l59_59929

theorem coo_coo_count_correct :
  let monday_coos := 89
  let tuesday_coos := 179
  let wednesday_coos := 21
  let total_coos := monday_coos + tuesday_coos + wednesday_coos
  total_coos = 289 :=
by
  sorry

end coo_coo_count_correct_l59_59929


namespace calc_6_4_3_199_plus_100_l59_59277

theorem calc_6_4_3_199_plus_100 (a b : ℕ) (h_a : a = 199) (h_b : b = 100) :
  6 * a + 4 * a + 3 * a + a + b = 2886 :=
by
  sorry

end calc_6_4_3_199_plus_100_l59_59277


namespace value_of_x_l59_59197

theorem value_of_x (u w z y x : ℤ) (h1 : u = 95) (h2 : w = u + 10) (h3 : z = w + 25) (h4 : y = z + 15) (h5 : x = y + 12) : x = 157 := by
  sorry

end value_of_x_l59_59197


namespace total_boys_across_grades_is_692_l59_59912

theorem total_boys_across_grades_is_692 (ga_girls gb_girls gc_girls : ℕ) (ga_boys : ℕ) :
  ga_girls = 256 →
  ga_girls = ga_boys + 52 →
  gb_girls = 360 →
  gb_boys = gb_girls - 40 →
  gc_girls = 168 →
  gc_girls = gc_boys →
  ga_boys + gb_boys + gc_boys = 692 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_boys_across_grades_is_692_l59_59912


namespace arithmetic_sequence_product_l59_59359

theorem arithmetic_sequence_product 
  (a d : ℤ)
  (h1 : a + 6 * d = 20)
  (h2 : d = 2) : 
  a * (a + d) * (a + 2 * d) = 960 := 
by
  -- proof goes here
  sorry

end arithmetic_sequence_product_l59_59359


namespace min_tan_of_acute_angle_l59_59327

def is_ocular_ray (u : ℚ) (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 ∧ u = x / y

def acute_angle_tangent (u v : ℚ) : ℚ :=
  |(u - v) / (1 + u * v)|

theorem min_tan_of_acute_angle :
  ∃ θ : ℚ, (∀ u v : ℚ, (∃ x1 y1 x2 y2 : ℕ, is_ocular_ray u x1 y1 ∧ is_ocular_ray v x2 y2 ∧ u ≠ v) 
  → acute_angle_tangent u v ≥ θ) ∧ θ = 1 / 722 :=
sorry

end min_tan_of_acute_angle_l59_59327


namespace yoongi_stacked_higher_by_one_cm_l59_59451

def height_box_A : ℝ := 3
def height_box_B : ℝ := 3.5
def boxes_stacked_by_Taehyung : ℕ := 16
def boxes_stacked_by_Yoongi : ℕ := 14
def height_Taehyung_stack : ℝ := height_box_A * boxes_stacked_by_Taehyung
def height_Yoongi_stack : ℝ := height_box_B * boxes_stacked_by_Yoongi

theorem yoongi_stacked_higher_by_one_cm :
  height_Yoongi_stack = height_Taehyung_stack + 1 :=
by
  sorry

end yoongi_stacked_higher_by_one_cm_l59_59451


namespace initial_erasers_count_l59_59626

noncomputable def erasers_lost := 42
noncomputable def erasers_ended_up_with := 53

theorem initial_erasers_count (initial_erasers : ℕ) : 
  initial_erasers_ended_up_with = initial_erasers - erasers_lost → initial_erasers = 95 :=
by
  sorry

end initial_erasers_count_l59_59626


namespace glove_selection_l59_59717

theorem glove_selection :
  let n := 6                -- Number of pairs
  let k := 4                -- Number of selected gloves
  let m := 1                -- Number of matching pairs
  let total_ways := n * 10 * 8 / 2  -- Calculation based on solution steps
  total_ways = 240 := by
  sorry

end glove_selection_l59_59717


namespace value_of_otimes_difference_l59_59400

def otimes (a b : ℚ) : ℚ := (a^3) / (b^2)

theorem value_of_otimes_difference :
  otimes (otimes 2 3) 4 - otimes 2 (otimes 3 4) = - 1184 / 243 := 
by
  sorry

end value_of_otimes_difference_l59_59400


namespace correct_statements_l59_59134

open Classical

variables {α l m n p : Type*}
variables (is_perpendicular_to : α → α → Prop) (is_parallel_to : α → α → Prop)
variables (is_in_plane : α → α → Prop)

noncomputable def problem_statement (l : α) (α : α) : Prop :=
  (∀ m, is_perpendicular_to m l → is_parallel_to m α) ∧
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α)

theorem correct_statements (l : α) (α : α) (h_l_α : is_perpendicular_to l α) :
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α) :=
sorry

end correct_statements_l59_59134


namespace attendees_on_monday_is_10_l59_59345

-- Define the given conditions
def attendees_tuesday : ℕ := 15
def attendees_wed_thru_fri : ℕ := 10
def days_wed_thru_fri : ℕ := 3
def average_attendance : ℕ := 11
def total_days : ℕ := 5

-- Define the number of people who attended class on Monday
def attendees_tuesday_to_friday : ℕ := attendees_tuesday + attendees_wed_thru_fri * days_wed_thru_fri
def total_attendance : ℕ := average_attendance * total_days
def attendees_monday : ℕ := total_attendance - attendees_tuesday_to_friday

-- State the theorem
theorem attendees_on_monday_is_10 : attendees_monday = 10 :=
by
  -- Proof omitted
  sorry

end attendees_on_monday_is_10_l59_59345


namespace find_angle_B_l59_59022

-- Given definitions and conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variable (h1 : (a + b + c) * (a - b + c) = a * c )

-- Statement of the proof problem
theorem find_angle_B (h1 : (a + b + c) * (a - b + c) = a * c) :
  B = 2 * π / 3 :=
sorry

end find_angle_B_l59_59022


namespace initial_red_martians_l59_59209

/-- Red Martians always tell the truth, while Blue Martians lie and then turn red.
    In a group of 2018 Martians, they answered in the sequence 1, 2, 3, ..., 2018 to the question
    of how many of them were red at that moment. Prove that the initial number of red Martians was 0 or 1. -/
theorem initial_red_martians (N : ℕ) (answers : Fin (N+1) → ℕ) :
  (∀ i : Fin (N+1), answers i = i.succ) → N = 2018 → (initial_red_martians_count = 0 ∨ initial_red_martians_count = 1)
:= sorry

end initial_red_martians_l59_59209


namespace range_of_t_l59_59463

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → f y < f x

variable {f : ℝ → ℝ}

theorem range_of_t (h_odd : odd_function f) 
  (h_decreasing : decreasing_function f)
  (h_ineq : ∀ t, -1 < t → t < 1 → f (1 - t) + f (1 - t^2) < 0) 
  : ∀ t, 0 < t → t < 1 :=
by sorry

end range_of_t_l59_59463


namespace sample_size_is_200_l59_59410
-- Define the total number of students and the number of students surveyed
def total_students : ℕ := 3600
def students_surveyed : ℕ := 200

-- Define the sample size
def sample_size := students_surveyed

-- Prove the sample size is 200
theorem sample_size_is_200 : sample_size = 200 :=
by
  -- Placeholder for the actual proof
  sorry

end sample_size_is_200_l59_59410


namespace repeating_decimal_to_fraction_l59_59910

theorem repeating_decimal_to_fraction :
  (2 + (35 / 99 : ℚ)) = (233 / 99) := 
sorry

end repeating_decimal_to_fraction_l59_59910


namespace one_fourth_one_third_two_fifths_l59_59179

theorem one_fourth_one_third_two_fifths (N : ℝ)
  (h₁ : 0.40 * N = 300) :
  (1/4) * (1/3) * (2/5) * N = 25 := 
sorry

end one_fourth_one_third_two_fifths_l59_59179


namespace mixture_replacement_l59_59958

theorem mixture_replacement:
  ∀ (A B x : ℝ),
    A = 64 →
    B = A / 4 →
    (A - (4/5) * x) / (B + (4/5) * x) = 2 / 3 →
    x = 40 :=
by
  intros A B x hA hB hRatio
  sorry

end mixture_replacement_l59_59958


namespace comparison_abc_l59_59761

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

end comparison_abc_l59_59761


namespace directrix_of_parabola_l59_59263

theorem directrix_of_parabola (h : ∀ x : ℝ, y = -3 * x ^ 2 + 6 * x - 5) : ∃ y : ℝ, y = -25 / 12 :=
by
  sorry

end directrix_of_parabola_l59_59263


namespace import_tax_amount_in_excess_l59_59425

theorem import_tax_amount_in_excess (X : ℝ) 
  (h1 : 0.07 * (2590 - X) = 111.30) : 
  X = 1000 :=
by
  sorry

end import_tax_amount_in_excess_l59_59425


namespace intersection_complement_l59_59570

universe u

-- Define the universal set U, and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement (U A : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The main theorem to be proved
theorem intersection_complement :
  B ∩ (complement U A) = {3, 4} := by
  sorry

end intersection_complement_l59_59570


namespace frosting_need_l59_59130

theorem frosting_need : 
  (let layer_cake_frosting := 1
   let single_cake_frosting := 0.5
   let brownie_frosting := 0.5
   let dozen_cupcakes_frosting := 0.5
   let num_layer_cakes := 3
   let num_dozen_cupcakes := 6
   let num_single_cakes := 12
   let num_pans_brownies := 18
   
   let total_frosting := 
     (num_layer_cakes * layer_cake_frosting) + 
     (num_dozen_cupcakes * dozen_cupcakes_frosting) + 
     (num_single_cakes * single_cake_frosting) + 
     (num_pans_brownies * brownie_frosting)
   
   total_frosting = 21) :=
  by
    sorry

end frosting_need_l59_59130


namespace solve_equation1_solve_equation2_l59_59258

-- Define the two equations
def equation1 (x : ℝ) := 3 * x - 4 = -2 * (x - 1)
def equation2 (x : ℝ) := 1 + (2 * x + 1) / 3 = (3 * x - 2) / 2

-- The statements to prove
theorem solve_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1.2 :=
by
  sorry

theorem solve_equation2 : ∃ x : ℝ, equation2 x ∧ x = 2.8 :=
by
  sorry

end solve_equation1_solve_equation2_l59_59258


namespace typist_current_salary_l59_59889

def original_salary : ℝ := 4000.0000000000005
def increased_salary (os : ℝ) : ℝ := os + (os * 0.1)
def decreased_salary (is : ℝ) : ℝ := is - (is * 0.05)

theorem typist_current_salary : decreased_salary (increased_salary original_salary) = 4180 :=
by
  sorry

end typist_current_salary_l59_59889


namespace problem1_problem2_l59_59373

-- Problem 1: Prove that 2023 * 2023 - 2024 * 2022 = 1
theorem problem1 : 2023 * 2023 - 2024 * 2022 = 1 := 
by 
  sorry

-- Problem 2: Prove that (-4 * x * y^3) * (1/2 * x * y) + (-3 * x * y^2)^2 = 7 * x^2 * y^4
theorem problem2 (x y : ℝ) : (-4 * x * y^3) * ((1/2) * x * y) + (-3 * x * y^2)^2 = 7 * x^2 * y^4 := 
by 
  sorry

end problem1_problem2_l59_59373


namespace best_marksman_score_l59_59160

def team_size : ℕ := 6
def total_points : ℕ := 497
def hypothetical_best_score : ℕ := 92
def hypothetical_average : ℕ := 84

theorem best_marksman_score :
  let total_with_hypothetical_best := team_size * hypothetical_average
  let difference := total_with_hypothetical_best - total_points
  let actual_best_score := hypothetical_best_score - difference
  actual_best_score = 85 := 
by
  -- Definitions in Lean are correctly set up
  intro total_with_hypothetical_best difference actual_best_score
  sorry

end best_marksman_score_l59_59160


namespace degrees_to_radians_300_l59_59030

theorem degrees_to_radians_300:
  (300 * (Real.pi / 180) = 5 * Real.pi / 3) := 
by
  repeat { sorry }

end degrees_to_radians_300_l59_59030


namespace estimate_points_in_interval_l59_59794

-- Define the conditions
def total_data_points : ℕ := 1000
def frequency_interval : ℝ := 0.16
def interval_estimation : ℝ := total_data_points * frequency_interval

-- Lean theorem statement
theorem estimate_points_in_interval : interval_estimation = 160 :=
by
  sorry

end estimate_points_in_interval_l59_59794


namespace chord_length_of_larger_circle_tangent_to_smaller_circle_l59_59081

theorem chord_length_of_larger_circle_tangent_to_smaller_circle :
  ∀ (A B C : ℝ), B = 5 → π * (A ^ 2 - B ^ 2) = 50 * π → (C / 2) ^ 2 + B ^ 2 = A ^ 2 → C = 10 * Real.sqrt 2 :=
by
  intros A B C hB hArea hChord
  sorry

end chord_length_of_larger_circle_tangent_to_smaller_circle_l59_59081


namespace range_of_k_l59_59987

theorem range_of_k :
  ∀ k : ℝ, (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0) :=
by
  sorry

end range_of_k_l59_59987


namespace percentage_taxed_l59_59289

theorem percentage_taxed (T : ℝ) (H1 : 3840 = T * (P : ℝ)) (H2 : 480 = 0.25 * T * (P : ℝ)) : P = 0.5 := 
by
  sorry

end percentage_taxed_l59_59289


namespace scientific_notation_example_l59_59596

theorem scientific_notation_example :
  (0.000000007: ℝ) = 7 * 10^(-9 : ℝ) :=
sorry

end scientific_notation_example_l59_59596


namespace antonio_correct_answers_l59_59461

theorem antonio_correct_answers :
  ∃ c w : ℕ, c + w = 15 ∧ 6 * c - 3 * w = 36 ∧ c = 9 :=
by
  sorry

end antonio_correct_answers_l59_59461


namespace expression_bounds_l59_59746

theorem expression_bounds (p q r s : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) (hr : 0 ≤ r ∧ r ≤ 1) (hs : 0 ≤ s ∧ s ≤ 1) :
  2 * Real.sqrt 2 ≤ (Real.sqrt (p^2 + (1 - q)^2) + Real.sqrt (q^2 + (1 - r)^2) + Real.sqrt (r^2 + (1 - s)^2) + Real.sqrt (s^2 + (1 - p)^2)) ∧
  (Real.sqrt (p^2 + (1 - q)^2) + Real.sqrt (q^2 + (1 - r)^2) + Real.sqrt (r^2 + (1 - s)^2) + Real.sqrt (s^2 + (1 - p)^2)) ≤ 4 :=
by
  sorry

end expression_bounds_l59_59746


namespace pages_per_chapter_l59_59716

-- Definitions based on conditions
def chapters_in_book : ℕ := 2
def days_to_finish : ℕ := 664
def chapters_per_day : ℕ := 332
def total_chapters_read : ℕ := chapters_per_day * days_to_finish

-- Theorem that states the problem
theorem pages_per_chapter : total_chapters_read / chapters_in_book = 110224 :=
by
  -- Proof is omitted
  sorry

end pages_per_chapter_l59_59716


namespace cora_reading_ratio_l59_59754

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

end cora_reading_ratio_l59_59754


namespace smallest_solution_of_quartic_l59_59292

theorem smallest_solution_of_quartic :
  ∃ x : ℝ, x^4 - 40*x^2 + 144 = 0 ∧ ∀ y : ℝ, (y^4 - 40*y^2 + 144 = 0) → x ≤ y :=
sorry

end smallest_solution_of_quartic_l59_59292


namespace num_points_satisfying_inequalities_l59_59622

theorem num_points_satisfying_inequalities :
  ∃ (n : ℕ), n = 2551 ∧
  ∀ (x y : ℤ), (y ≤ 3 * x) ∧ (y ≥ x / 3) ∧ (x + y ≤ 100) → 
              ∃ (p : ℕ), p = n := 
by
  sorry

end num_points_satisfying_inequalities_l59_59622


namespace correct_total_l59_59759

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

end correct_total_l59_59759


namespace solution_sum_l59_59891

theorem solution_sum (m n : ℝ) (h₀ : m ≠ 0) (h₁ : m^2 + m * n - m = 0) : m + n = 1 := 
by 
  sorry

end solution_sum_l59_59891


namespace min_value_correct_l59_59131

noncomputable def min_value (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m + n = 1) : ℝ :=
(1 / m) + (2 / n)

theorem min_value_correct :
  ∃ m n : ℝ, ∃ h₁ : m > 0, ∃ h₂ : n > 0, ∃ h₃ : m + n = 1,
  min_value m n h₁ h₂ h₃ = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_correct_l59_59131


namespace max_value_of_y_over_x_l59_59684

theorem max_value_of_y_over_x {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 2 * y = 3) :
  y / x ≤ 9 / 8 :=
sorry

end max_value_of_y_over_x_l59_59684


namespace A_share_of_profit_l59_59324

-- Define necessary financial terms and operations
def initial_investment_A := 3000
def initial_investment_B := 4000

def withdrawal_A := 1000
def advanced_B := 1000

def duration_initial := 8
def duration_remaining := 4

def total_profit := 630

-- Calculate the equivalent investment duration for A and B
def investment_months_A_first := initial_investment_A * duration_initial
def investment_months_A_remaining := (initial_investment_A - withdrawal_A) * duration_remaining
def investment_months_A := investment_months_A_first + investment_months_A_remaining

def investment_months_B_first := initial_investment_B * duration_initial
def investment_months_B_remaining := (initial_investment_B + advanced_B) * duration_remaining
def investment_months_B := investment_months_B_first + investment_months_B_remaining

-- Prove that A's share of the profit is Rs. 240
theorem A_share_of_profit : 
  let ratio_A : ℚ := 4
  let ratio_B : ℚ := 6.5
  let total_ratio : ℚ := ratio_A + ratio_B
  let a_share : ℚ := (total_profit * ratio_A) / total_ratio
  a_share = 240 := 
by
  sorry

end A_share_of_profit_l59_59324


namespace original_triangle_area_l59_59590

theorem original_triangle_area (new_area : ℝ) (scaling_factor : ℝ) (area_ratio : ℝ) : 
  new_area = 32 → scaling_factor = 2 → 
  area_ratio = scaling_factor ^ 2 → 
  new_area / area_ratio = 8 := 
by
  intros
  -- insert your proof logic here
  sorry

end original_triangle_area_l59_59590


namespace math_problem_l59_59641

theorem math_problem (p q : ℕ) (hp : p % 13 = 7) (hq : q % 13 = 7) (hp_lower : 1000 ≤ p) (hp_upper : p < 10000) (hq_lower : 10000 ≤ q) (min_p : ∀ n, n % 13 = 7 → 1000 ≤ n → n < 10000 → p ≤ n) (min_q : ∀ n, n % 13 = 7 → 10000 ≤ n → q ≤ n) : 
  q - p = 8996 := 
sorry

end math_problem_l59_59641


namespace intersect_xz_plane_at_point_l59_59664

-- Define points and vectors in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the points A and B
def A : Point3D := ⟨2, -1, 3⟩
def B : Point3D := ⟨6, 7, -2⟩

-- Define the direction vector as the difference between points A and B
def direction_vector (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

-- Function to parameterize the line given a point and direction vector
def parametric_line (P : Point3D) (v : Point3D) (t : ℝ) : Point3D :=
  ⟨P.x + t * v.x, P.y + t * v.y, P.z + t * v.z⟩

-- Define the xz-plane intersection condition (y coordinate should be 0)
def intersects_xz_plane (P : Point3D) (v : Point3D) (t : ℝ) : Prop :=
  (parametric_line P v t).y = 0

-- Define the intersection point as a Point3D
def intersection_point : Point3D := ⟨2.5, 0, 2.375⟩

-- Statement to prove the intersection
theorem intersect_xz_plane_at_point : 
  ∃ t : ℝ, intersects_xz_plane A (direction_vector A B) t ∧ parametric_line A (direction_vector A B) t = intersection_point :=
by
  sorry

end intersect_xz_plane_at_point_l59_59664


namespace problem1_xy_xplusy_l59_59765

theorem problem1_xy_xplusy (x y: ℝ) (h1: x * y = 5) (h2: x + y = 6) : x - y = 4 ∨ x - y = -4 := 
sorry

end problem1_xy_xplusy_l59_59765


namespace ratio_adidas_skechers_l59_59473

-- Conditions
def total_expenditure : ℤ := 8000
def expenditure_adidas : ℤ := 600
def expenditure_clothes : ℤ := 2600
def expenditure_nike := 3 * expenditure_adidas

-- Calculation for sneakers
def total_sneakers := total_expenditure - expenditure_clothes
def expenditure_nike_adidas := expenditure_nike + expenditure_adidas
def expenditure_skechers := total_sneakers - expenditure_nike_adidas

-- Prove the ratio
theorem ratio_adidas_skechers (H1 : total_expenditure = 8000)
                              (H2 : expenditure_adidas = 600)
                              (H3 : expenditure_nike = 3 * expenditure_adidas)
                              (H4 : expenditure_clothes = 2600) :
  expenditure_adidas / expenditure_skechers = 1 / 5 :=
by
  sorry

end ratio_adidas_skechers_l59_59473


namespace find_tuesday_temperature_l59_59110

variable (T W Th F : ℝ)

def average_temperature_1 : Prop := (T + W + Th) / 3 = 52
def average_temperature_2 : Prop := (W + Th + F) / 3 = 54
def friday_temperature : Prop := F = 53

theorem find_tuesday_temperature (h1 : average_temperature_1 T W Th) (h2 : average_temperature_2 W Th F) (h3 : friday_temperature F) :
  T = 47 :=
by
  sorry

end find_tuesday_temperature_l59_59110


namespace sum_series_l59_59509

noncomputable def b : ℕ → ℝ
| 0     => 2
| 1     => 2
| (n+2) => b (n+1) + b n

theorem sum_series : (∑' n, b n / 3^(n+1)) = 1 / 3 := by
  sorry

end sum_series_l59_59509


namespace frequency_of_largest_rectangle_area_l59_59511

theorem frequency_of_largest_rectangle_area (a : ℕ → ℝ) (sample_size : ℕ)
    (h_geom : ∀ n, a (n + 1) = 2 * a n) (h_sum : a 0 + a 1 + a 2 + a 3 = 1)
    (h_sample : sample_size = 300) : 
    sample_size * a 3 = 160 := by
  sorry

end frequency_of_largest_rectangle_area_l59_59511


namespace A_50_correct_l59_59542

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := 
  ![![3, 2], 
    ![-8, -5]]

-- The theorem to prove
theorem A_50_correct : A^50 = ![![(-199 : ℤ), -100], 
                                 ![400, 201]] := 
by
  sorry

end A_50_correct_l59_59542


namespace sin_110_correct_tan_945_correct_cos_25pi_over_4_correct_l59_59355

noncomputable def sin_110_degrees : ℝ := Real.sin (110 * Real.pi / 180)
noncomputable def tan_945_degrees_reduction : ℝ := Real.tan (945 * Real.pi / 180 - 5 * Real.pi)
noncomputable def cos_25pi_over_4_reduction : ℝ := Real.cos (25 * Real.pi / 4 - 6 * 2 * Real.pi)

theorem sin_110_correct : sin_110_degrees = Real.sin (110 * Real.pi / 180) :=
by
  sorry

theorem tan_945_correct : tan_945_degrees_reduction = 1 :=
by 
  sorry

theorem cos_25pi_over_4_correct : cos_25pi_over_4_reduction = Real.cos (Real.pi / 4) :=
by 
  sorry

end sin_110_correct_tan_945_correct_cos_25pi_over_4_correct_l59_59355


namespace slope_and_y_intercept_l59_59789

def line_equation (x y : ℝ) : Prop := 4 * y = 6 * x - 12

theorem slope_and_y_intercept (x y : ℝ) (h : line_equation x y) : 
  ∃ m b : ℝ, (m = 3/2) ∧ (b = -3) ∧ (y = m * x + b) :=
  sorry

end slope_and_y_intercept_l59_59789


namespace KarenEggRolls_l59_59388

-- Definitions based on conditions
def OmarEggRolls : ℕ := 219
def TotalEggRolls : ℕ := 448

-- The statement to be proved
theorem KarenEggRolls : (TotalEggRolls - OmarEggRolls = 229) :=
by {
    -- Proof step goes here
    sorry
}

end KarenEggRolls_l59_59388


namespace simplify_polynomial_l59_59069

/-- Simplification of the polynomial expression -/
theorem simplify_polynomial (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^2 + 15) - (x^6 + 4 * x^5 - 2 * x^3 + 20) = x^6 - x^5 + 2 * x^3 - 5 :=
by {
  sorry
}

end simplify_polynomial_l59_59069


namespace part1_solution_set_m1_part2_find_m_l59_59882

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (m+1) * x^2 - m * x + m - 1

theorem part1_solution_set_m1 :
  { x : ℝ | f x 1 > 0 } = { x : ℝ | x < 0 } ∪ { x : ℝ | x > 0.5 } :=
by
  sorry

theorem part2_find_m :
  (∀ x : ℝ, f x m + 1 > 0 ↔ x > 1.5 ∧ x < 3) → m = -9/7 :=
by
  sorry

end part1_solution_set_m1_part2_find_m_l59_59882


namespace cindy_correct_method_l59_59318

theorem cindy_correct_method (x : ℝ) (h : (x - 7) / 5 = 15) : (x - 5) / 7 = 11 := 
by
  sorry

end cindy_correct_method_l59_59318


namespace water_usage_correct_l59_59593

variable (y : ℝ) (C₁ : ℝ) (C₂ : ℝ) (x : ℝ)

noncomputable def water_bill : ℝ :=
  if x ≤ 4 then C₁ * x else 4 * C₁ + C₂ * (x - 4)

theorem water_usage_correct (h1 : y = 12.8) (h2 : C₁ = 1.2) (h3 : C₂ = 1.6) : x = 9 :=
by
  have h4 : x > 4 := sorry
  sorry

end water_usage_correct_l59_59593


namespace passenger_difference_l59_59102

theorem passenger_difference {x : ℕ} :
  (30 + x = 3 * x + 14) →
  6 = 3 * x - x - 16 :=
by
  sorry

end passenger_difference_l59_59102


namespace possible_values_of_X_l59_59842

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

end possible_values_of_X_l59_59842


namespace total_miles_Wednesday_l59_59602

-- The pilot flew 1134 miles on Tuesday and 1475 miles on Thursday.
def miles_flown_Tuesday : ℕ := 1134
def miles_flown_Thursday : ℕ := 1475

-- The miles flown on Wednesday is denoted as "x".
variable (x : ℕ)

-- The period is 4 weeks.
def weeks : ℕ := 4

-- We need to prove that the total miles flown on Wednesdays during this 4-week period is 4 * x.
theorem total_miles_Wednesday : 4 * x = 4 * x := by sorry

end total_miles_Wednesday_l59_59602


namespace log_product_l59_59725

open Real

theorem log_product : log 9 / log 2 * (log 5 / log 3) * (log 8 / log (sqrt 5)) = 12 :=
by
  sorry

end log_product_l59_59725


namespace x_eq_sum_of_squares_of_two_consecutive_integers_l59_59387

noncomputable def x_seq (n : ℕ) : ℝ :=
  1 / 4 * ((2 + Real.sqrt 3) ^ (2 * n - 1) + (2 - Real.sqrt 3) ^ (2 * n - 1))

theorem x_eq_sum_of_squares_of_two_consecutive_integers (n : ℕ) : 
  ∃ y : ℤ, x_seq n = (y:ℝ)^2 + (y + 1)^2 :=
sorry

end x_eq_sum_of_squares_of_two_consecutive_integers_l59_59387


namespace poles_inside_base_l59_59835

theorem poles_inside_base :
  ∃ n : ℕ, 2015 + n ≡ 0 [MOD 36] ∧ n = 1 :=
sorry

end poles_inside_base_l59_59835


namespace sally_received_quarters_l59_59679

theorem sally_received_quarters : 
  ∀ (original_quarters total_quarters received_quarters : ℕ), 
  original_quarters = 760 → 
  total_quarters = 1178 → 
  received_quarters = total_quarters - original_quarters → 
  received_quarters = 418 :=
by 
  intros original_quarters total_quarters received_quarters h_original h_total h_received
  rw [h_original, h_total] at h_received
  exact h_received

end sally_received_quarters_l59_59679


namespace correct_option_is_d_l59_59290

theorem correct_option_is_d (x : ℚ) : -x^3 = (-x)^3 :=
sorry

end correct_option_is_d_l59_59290


namespace second_customer_payment_l59_59621

def price_of_headphones : ℕ := 30
def total_cost_first_customer (P H : ℕ) : ℕ := 5 * P + 8 * H
def total_cost_second_customer (P H : ℕ) : ℕ := 3 * P + 4 * H

theorem second_customer_payment
  (P : ℕ)
  (H_eq : H = price_of_headphones)
  (first_customer_eq : total_cost_first_customer P H = 840) :
  total_cost_second_customer P H = 480 :=
by
  -- Proof to be filled in later
  sorry

end second_customer_payment_l59_59621


namespace car_return_speed_l59_59234

theorem car_return_speed (d : ℕ) (r : ℕ) (h₁ : d = 180) (h₂ : (2 * d) / ((d / 90) + (d / r)) = 60) : r = 45 :=
by
  rw [h₁] at h₂
  have h3 : 2 * 180 / ((180 / 90) + (180 / r)) = 60 := h₂
  -- The rest of the proof involves solving for r, but here we only need the statement
  sorry

end car_return_speed_l59_59234


namespace bowls_total_marbles_l59_59962

theorem bowls_total_marbles :
  let C2 := 600
  let C1 := (3 / 4 : ℝ) * C2
  let C3 := (1 / 2 : ℝ) * C1
  C1 = 450 ∧ C3 = 225 ∧ (C1 + C2 + C3 = 1275) := 
by
  let C2 := 600
  let C1 := (3 / 4 : ℝ) * C2
  let C3 := (1 / 2 : ℝ) * C1
  have hC1 : C1 = 450 := by norm_num
  have hC3 : C3 = 225 := by norm_num
  have hTotal : C1 + C2 + C3 = 1275 := by norm_num
  exact ⟨hC1, hC3, hTotal⟩

end bowls_total_marbles_l59_59962


namespace max_volume_cylinder_l59_59299

theorem max_volume_cylinder (x : ℝ) (h1 : x > 0) (h2 : x < 10) : 
  (∀ x, 0 < x ∧ x < 10 → ∃ max_v, max_v = (4 * (10^3) * Real.pi) / 27) ∧ 
  ∃ x, x = 20/3 := 
by
  sorry

end max_volume_cylinder_l59_59299


namespace find_values_of_a_and_b_l59_59584

theorem find_values_of_a_and_b
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (x : ℝ) (hx : x > 1)
  (h : 9 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 17)
  (h2 : (Real.log b / Real.log a) * (Real.log a / Real.log b) = 2) :
  a = 10 ^ Real.sqrt 2 ∧ b = 10 := by
sorry

end find_values_of_a_and_b_l59_59584


namespace max_profit_thousand_rubles_l59_59951

theorem max_profit_thousand_rubles :
  ∃ x y : ℕ, 
    (80 * x + 100 * y = 2180) ∧ 
    (10 * x + 70 * y ≤ 700) ∧ 
    (23 * x + 40 * y ≤ 642) := 
by
  -- proof goes here
  sorry

end max_profit_thousand_rubles_l59_59951


namespace sufficient_but_not_necessary_condition_still_holds_when_not_positive_l59_59331

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (a > 0 ∧ b > 0) → (b / a + a / b ≥ 2) :=
by 
  sorry

theorem still_holds_when_not_positive (a b : ℝ) (h1 : a ≤ 0 ∨ b ≤ 0) :
  (b / a + a / b ≥ 2) :=
by
  sorry

end sufficient_but_not_necessary_condition_still_holds_when_not_positive_l59_59331


namespace evaluate_three_squared_raised_four_l59_59232

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end evaluate_three_squared_raised_four_l59_59232


namespace extreme_values_l59_59731

-- Define the function f(x) with symbolic constants a and b
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x

-- Given conditions
def intersects_at_1_0 (a b : ℝ) : Prop := (f 1 a b = 0)
def derivative_at_1_0 (a b : ℝ) : Prop := (3 - 2 * a - b = 0)

-- Main theorem statement
theorem extreme_values (a b : ℝ) (h1 : intersects_at_1_0 a b) (h2 : derivative_at_1_0 a b) :
  (∀ x, f x a b ≤ 4 / 27) ∧ (∀ x, 0 ≤ f x a b) :=
sorry

end extreme_values_l59_59731


namespace sachin_is_younger_by_8_years_l59_59205

variable (S R : ℕ)

-- Conditions
axiom age_of_sachin : S = 28
axiom ratio_of_ages : S * 9 = R * 7

-- Goal
theorem sachin_is_younger_by_8_years (S R : ℕ) (h1 : S = 28) (h2 : S * 9 = R * 7) : R - S = 8 :=
by
  sorry

end sachin_is_younger_by_8_years_l59_59205


namespace flowers_sold_difference_l59_59275

def number_of_daisies_sold_on_second_day (d2 : ℕ) (d3 : ℕ) (d_sum : ℕ) : Prop :=
  d3 = 2 * d2 - 10 ∧
  d_sum = 45 + d2 + d3 + 120

theorem flowers_sold_difference (d2 : ℕ) (d3 : ℕ) (d_sum : ℕ) 
  (h : number_of_daisies_sold_on_second_day d2 d3 d_sum) :
  45 + d2 + d3 + 120 = 350 → 
  d2 - 45 = 20 := 
by
  sorry

end flowers_sold_difference_l59_59275


namespace class_size_is_44_l59_59122

theorem class_size_is_44 (n : ℕ) : 
  (n - 1) % 2 = 1 ∧ (n - 1) % 7 = 1 → n = 44 := 
by 
  sorry

end class_size_is_44_l59_59122


namespace compute_nested_f_l59_59091

def f(x : ℤ) : ℤ := x^2 - 4 * x + 3

theorem compute_nested_f : f (f (f (f (f (f 2))))) = f 1179395 := 
  sorry

end compute_nested_f_l59_59091


namespace no_absolute_winner_prob_l59_59332

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end no_absolute_winner_prob_l59_59332


namespace width_of_deck_l59_59728

noncomputable def length : ℝ := 30
noncomputable def cost_per_sqft_construction : ℝ := 3
noncomputable def cost_per_sqft_sealant : ℝ := 1
noncomputable def total_cost : ℝ := 4800
noncomputable def total_cost_per_sqft : ℝ := cost_per_sqft_construction + cost_per_sqft_sealant

theorem width_of_deck (w : ℝ) 
  (h1 : length * w * total_cost_per_sqft = total_cost) : 
  w = 40 := 
sorry

end width_of_deck_l59_59728


namespace dormouse_stole_flour_l59_59824

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

end dormouse_stole_flour_l59_59824


namespace probability_adjacent_vertices_in_decagon_l59_59143

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l59_59143


namespace find_f_log_3_54_l59_59749

noncomputable def f : ℝ → ℝ := sorry  -- Since we have to define a function and we do not need the exact implementation.

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_property : ∀ x : ℝ, f (x + 2) = - 1 / f x
axiom interval_property : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 3 ^ x

theorem find_f_log_3_54 : f (Real.log 54 / Real.log 3) = -3 / 2 :=
by
  sorry


end find_f_log_3_54_l59_59749


namespace circle_range_of_a_l59_59192

theorem circle_range_of_a (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + a = 0) → a < 5 := by
  sorry

end circle_range_of_a_l59_59192


namespace george_score_l59_59424

theorem george_score (avg_without_george avg_with_george : ℕ) (num_students : ℕ) 
(h1 : avg_without_george = 75) (h2 : avg_with_george = 76) (h3 : num_students = 20) :
  (num_students * avg_with_george) - ((num_students - 1) * avg_without_george) = 95 :=
by 
  sorry

end george_score_l59_59424


namespace number_of_4_digit_numbers_divisible_by_9_l59_59019

theorem number_of_4_digit_numbers_divisible_by_9 :
  ∃ n : ℕ, (∀ k : ℕ, k ∈ Finset.range n → 1008 + k * 9 ≤ 9999) ∧
           (1008 + (n - 1) * 9 = 9999) ∧
           n = 1000 :=
by
  sorry

end number_of_4_digit_numbers_divisible_by_9_l59_59019


namespace sum_of_two_numbers_l59_59905

theorem sum_of_two_numbers (x y : ℝ) (h1 : 0.5 * x + 0.3333 * y = 11)
(h2 : max x y = y) (h3 : y = 15) : x + y = 27 :=
by
  -- Skip the proof and add sorry
  sorry

end sum_of_two_numbers_l59_59905


namespace steps_from_center_to_square_l59_59726

-- Define the conditions and question in Lean 4
def steps_to_center := 354
def total_steps := 582

-- Prove that the steps from Rockefeller Center to Times Square is 228
theorem steps_from_center_to_square : (total_steps - steps_to_center) = 228 := by
  sorry

end steps_from_center_to_square_l59_59726


namespace solve_fraction_equation_l59_59975

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 5 * x - 14) - 1 / (x^2 - 15 * x - 18) = 0) →
  x = 2 ∨ x = -9 ∨ x = 6 ∨ x = -3 :=
sorry

end solve_fraction_equation_l59_59975


namespace complex_modulus_problem_l59_59023

noncomputable def imaginary_unit : ℂ := Complex.I

theorem complex_modulus_problem (z : ℂ) (h : (1 + Real.sqrt 3 * imaginary_unit)^2 * z = 1 - imaginary_unit^3) :
  Complex.abs z = Real.sqrt 2 / 4 :=
by
  sorry

end complex_modulus_problem_l59_59023


namespace exists_root_in_interval_l59_59604

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

-- Conditions given in the problem
variables {a b c : ℝ}
variable  (h_a_nonzero : a ≠ 0)
variable  (h_neg_value : quadratic a b c 3.24 = -0.02)
variable  (h_pos_value : quadratic a b c 3.25 = 0.01)

-- Problem statement to be proved
theorem exists_root_in_interval : ∃ x : ℝ, 3.24 < x ∧ x < 3.25 ∧ quadratic a b c x = 0 :=
sorry

end exists_root_in_interval_l59_59604


namespace washing_machine_heavy_washes_l59_59214

theorem washing_machine_heavy_washes
  (H : ℕ)                                  -- The number of heavy washes
  (heavy_wash_gallons : ℕ := 20)            -- Gallons of water for a heavy wash
  (regular_wash_gallons : ℕ := 10)          -- Gallons of water for a regular wash
  (light_wash_gallons : ℕ := 2)             -- Gallons of water for a light wash
  (num_regular_washes : ℕ := 3)             -- Number of regular washes
  (num_light_washes : ℕ := 1)               -- Number of light washes
  (num_bleach_rinses : ℕ := 2)              -- Number of bleach rinses (extra light washes)
  (total_water_needed : ℕ := 76)            -- Total gallons of water needed
  (h_regular_wash_water : num_regular_washes * regular_wash_gallons = 30)
  (h_light_wash_water : num_light_washes * light_wash_gallons = 2)
  (h_bleach_rinse_water : num_bleach_rinses * light_wash_gallons = 4) :
  20 * H + 30 + 2 + 4 = 76 → H = 2 :=
by
  intros
  sorry

end washing_machine_heavy_washes_l59_59214


namespace days_to_cover_half_lake_l59_59273

-- Define the problem conditions in Lean
def doubles_every_day (size: ℕ → ℝ) : Prop :=
  ∀ n : ℕ, size (n + 1) = 2 * size n

def takes_25_days_to_cover_lake (size: ℕ → ℝ) (lake_size: ℝ) : Prop :=
  size 25 = lake_size

-- Define the main theorem
theorem days_to_cover_half_lake (size: ℕ → ℝ) (lake_size: ℝ) 
  (h1: doubles_every_day size) (h2: takes_25_days_to_cover_lake size lake_size) : 
  size 24 = lake_size / 2 :=
sorry

end days_to_cover_half_lake_l59_59273


namespace mike_needs_percentage_to_pass_l59_59752

theorem mike_needs_percentage_to_pass :
  ∀ (mike_score marks_short max_marks : ℕ),
  mike_score = 212 → marks_short = 22 → max_marks = 780 →
  ((mike_score + marks_short : ℕ) / (max_marks : ℕ) : ℚ) * 100 = 30 :=
by
  intros mike_score marks_short max_marks Hmike Hshort Hmax
  rw [Hmike, Hshort, Hmax]
  -- Proof will be filled out here
  sorry

end mike_needs_percentage_to_pass_l59_59752


namespace sum_c_d_eq_neg11_l59_59552

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := (x + 6) / (x^2 + c * x + d)

theorem sum_c_d_eq_neg11 (c d : ℝ) 
    (h₀ : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = 3 ∨ x = -4)) :
    c + d = -11 := 
sorry

end sum_c_d_eq_neg11_l59_59552


namespace sum_of_x_coords_Q3_l59_59189

-- Definitions
def Q1_vertices_sum_x (S : ℝ) := S = 1050

def Q2_vertices_sum_x (S' : ℝ) (S : ℝ) := S' = S

def Q3_vertices_sum_x (S'' : ℝ) (S' : ℝ) := S'' = S'

-- Lean 4 statement
theorem sum_of_x_coords_Q3 (S : ℝ) (S' : ℝ) (S'' : ℝ) :
  Q1_vertices_sum_x S →
  Q2_vertices_sum_x S' S →
  Q3_vertices_sum_x S'' S' →
  S'' = 1050 :=
by
  sorry

end sum_of_x_coords_Q3_l59_59189


namespace radian_measure_of_negative_150_degree_l59_59070

theorem radian_measure_of_negative_150_degree  : (-150 : ℝ) * (Real.pi / 180) = - (5 * Real.pi / 6) := by
  sorry

end radian_measure_of_negative_150_degree_l59_59070


namespace negation_of_existential_proposition_l59_59565

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end negation_of_existential_proposition_l59_59565


namespace chord_count_l59_59645

theorem chord_count {n : ℕ} (h : n = 2024) : 
  ∃ k : ℕ, k ≥ 1024732 ∧ ∀ (i j : ℕ), (i < n → j < n → i ≠ j → true) := sorry

end chord_count_l59_59645


namespace intersection_A_B_l59_59491

-- Definition of sets A and B
def A := {x : ℝ | x > 2}
def B := { x : ℝ | (x - 1) * (x - 3) < 0 }

-- Claim that A ∩ B = {x : ℝ | 2 < x < 3}
theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l59_59491


namespace molecular_weight_of_7_moles_KBrO3_l59_59893

def potassium_atomic_weight : ℝ := 39.10
def bromine_atomic_weight : ℝ := 79.90
def oxygen_atomic_weight : ℝ := 16.00
def oxygen_atoms_in_KBrO3 : ℝ := 3

def KBrO3_molecular_weight : ℝ := 
  potassium_atomic_weight + bromine_atomic_weight + (oxygen_atomic_weight * oxygen_atoms_in_KBrO3)

def moles := 7

theorem molecular_weight_of_7_moles_KBrO3 : KBrO3_molecular_weight * moles = 1169.00 := 
by {
  -- The proof would be here, but it is omitted as instructed.
  sorry
}

end molecular_weight_of_7_moles_KBrO3_l59_59893


namespace ian_number_is_1021_l59_59381

-- Define the sequences each student skips
def alice_skips (n : ℕ) := ∃ k : ℕ, n = 4 * k
def barbara_skips (n : ℕ) := ∃ k : ℕ, n = 16 * (k + 1)
def candice_skips (n : ℕ) := ∃ k : ℕ, n = 64 * (k + 1)
-- Similar definitions for Debbie, Eliza, Fatima, Greg, and Helen

-- Define the condition under which Ian says a number
def ian_says (n : ℕ) :=
  ¬(alice_skips n) ∧ ¬(barbara_skips n) ∧ ¬(candice_skips n) -- and so on for Debbie, Eliza, Fatima, Greg, Helen

theorem ian_number_is_1021 : ian_says 1021 :=
by
  sorry

end ian_number_is_1021_l59_59381


namespace real_solutions_l59_59458

open Real

theorem real_solutions (x : ℝ) : (x - 2) ^ 4 + (2 - x) ^ 4 = 50 ↔ 
  x = 2 + sqrt (-12 + 3 * sqrt 17) ∨ x = 2 - sqrt (-12 + 3 * sqrt 17) :=
by
  sorry

end real_solutions_l59_59458


namespace find_y_l59_59176

theorem find_y (x y : ℕ) (h1 : x = 2407) (h2 : x^y + y^x = 2408) : y = 1 :=
sorry

end find_y_l59_59176


namespace min_ω_value_l59_59539

def min_ω (ω : Real) : Prop :=
  ω > 0 ∧ (∃ k : Int, ω = 2 * k + 2 / 3)

theorem min_ω_value : ∃ ω : Real, min_ω ω ∧ ω = 2 / 3 := by
  sorry

end min_ω_value_l59_59539


namespace min_value_of_a_l59_59202

theorem min_value_of_a (a : ℝ) (h : a > 0) (h₁ : ∀ x : ℝ, |x - a| + |1 - x| ≥ 1) : a ≥ 2 := 
sorry

end min_value_of_a_l59_59202


namespace plums_total_correct_l59_59972

-- Define the number of plums picked by Melanie, Dan, and Sally
def plums_melanie : ℕ := 4
def plums_dan : ℕ := 9
def plums_sally : ℕ := 3

-- Define the total number of plums picked
def total_plums : ℕ := plums_melanie + plums_dan + plums_sally

-- Theorem stating the total number of plums picked
theorem plums_total_correct : total_plums = 16 := by
  sorry

end plums_total_correct_l59_59972


namespace solution_sets_equiv_solve_l59_59999

theorem solution_sets_equiv_solve (a b : ℝ) :
  (∀ x : ℝ, (4 * x + 1) / (x + 2) < 0 ↔ -2 < x ∧ x < -1 / 4) →
  (∀ x : ℝ, a * x^2 + b * x - 2 > 0 ↔ -2 < x ∧ x < -1 / 4) →
  a = -4 ∧ b = -9 := by
  sorry

end solution_sets_equiv_solve_l59_59999


namespace eight_in_M_nine_in_M_ten_not_in_M_l59_59658

def M (a : ℤ) : Prop := ∃ b c : ℤ, a = b^2 - c^2

theorem eight_in_M : M 8 := by
  sorry

theorem nine_in_M : M 9 := by
  sorry

theorem ten_not_in_M : ¬ M 10 := by
  sorry

end eight_in_M_nine_in_M_ten_not_in_M_l59_59658


namespace initial_percentage_of_water_l59_59769

theorem initial_percentage_of_water (P : ℕ) : 
  (P / 100) * 120 + 54 = (3 / 4) * 120 → P = 30 :=
by 
  intro h
  sorry

end initial_percentage_of_water_l59_59769


namespace value_of_expression_l59_59478

theorem value_of_expression : (5^2 - 4^2 + 3^2) = 18 := 
by
  sorry

end value_of_expression_l59_59478


namespace find_k_value_l59_59671

theorem find_k_value (x y k : ℝ) 
  (h1 : x - 3 * y = k + 2) 
  (h2 : x - y = 4) 
  (h3 : 3 * x + y = -8) : 
  k = 12 := 
  by {
    sorry
  }

end find_k_value_l59_59671


namespace area_of_region_l59_59067

theorem area_of_region : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*x - 6*y = 1) → (∃ (A : ℝ), A = 14 * Real.pi) := 
by
  sorry

end area_of_region_l59_59067


namespace ordered_triples_count_l59_59616

theorem ordered_triples_count : 
  let b := 3003
  let side_length_squared := b * b
  let num_divisors := (2 + 1) * (2 + 1) * (2 + 1) * (2 + 1)
  let half_divisors := num_divisors / 2
  half_divisors = 40 := by
  sorry

end ordered_triples_count_l59_59616


namespace rectangle_area_l59_59838

theorem rectangle_area (x y : ℝ) (L W : ℝ) (h_diagonal : (L ^ 2 + W ^ 2) ^ (1 / 2) = x + y) (h_ratio : L / W = 3 / 2) : 
  L * W = (6 * (x + y) ^ 2) / 13 := 
sorry

end rectangle_area_l59_59838


namespace triangle_side_y_values_l59_59376

theorem triangle_side_y_values (y : ℕ) : (4 < y^2 ∧ y^2 < 20) ↔ (y = 3 ∨ y = 4) :=
by
  sorry

end triangle_side_y_values_l59_59376


namespace sum_even_integers_602_to_700_l59_59233

-- Definitions based on the conditions and the problem statement
def sum_first_50_even_integers := 2550
def n_even_602_700 := 50
def first_term_602_to_700 := 602
def last_term_602_to_700 := 700

-- Theorem statement
theorem sum_even_integers_602_to_700 : 
  sum_first_50_even_integers = 2550 → 
  n_even_602_700 = 50 →
  (n_even_602_700 / 2) * (first_term_602_to_700 + last_term_602_to_700) = 32550 :=
by
  sorry

end sum_even_integers_602_to_700_l59_59233


namespace lemonade_percentage_l59_59529

theorem lemonade_percentage (V : ℝ) (L : ℝ) :
  (0.80 * 0.40 * V + (100 - L) / 100 * 0.60 * V = 0.65 * V) →
  L = 99.45 :=
by
  intro h
  -- The proof would go here
  sorry

end lemonade_percentage_l59_59529


namespace probability_AC_adjacent_l59_59350

noncomputable def probability_AC_adjacent_given_AB_adjacent : ℚ :=
  let total_permutations_with_AB_adjacent := 48
  let permutations_with_ABC_adjacent := 12
  permutations_with_ABC_adjacent / total_permutations_with_AB_adjacent

theorem probability_AC_adjacent :  
  probability_AC_adjacent_given_AB_adjacent = 1 / 4 :=
by
  sorry

end probability_AC_adjacent_l59_59350


namespace soap_bars_problem_l59_59988

theorem soap_bars_problem :
  ∃ (N : ℤ), 200 < N ∧ N < 300 ∧ 2007 % N = 5 :=
sorry

end soap_bars_problem_l59_59988


namespace candidate_fails_by_50_marks_l59_59588

theorem candidate_fails_by_50_marks (T : ℝ) (pass_mark : ℝ) (h1 : pass_mark = 199.99999999999997)
    (h2 : 0.45 * T - 25 = 199.99999999999997) :
    199.99999999999997 - 0.30 * T = 50 :=
by
  sorry

end candidate_fails_by_50_marks_l59_59588


namespace q_compound_l59_59056

def q (x y : ℤ) : ℤ :=
  if x ≥ 1 ∧ y ≥ 1 then 2 * x + 3 * y
  else if x < 0 ∧ y < 0 then x + y^2
  else 4 * x - 2 * y

theorem q_compound : q (q 2 (-2)) (q 0 0) = 48 := 
by 
  sorry

end q_compound_l59_59056


namespace range_of_t_l59_59021

theorem range_of_t (t : ℝ) (x : ℝ) : (1 < x ∧ x ≤ 4) → (|x - t| < 1 ↔ 2 ≤ t ∧ t ≤ 3) :=
by
  sorry

end range_of_t_l59_59021


namespace percent_research_and_development_is_9_l59_59630

-- Define given percentages
def percent_transportation := 20
def percent_utilities := 5
def percent_equipment := 4
def percent_supplies := 2

-- Define degree representation and calculate percent for salaries
def degrees_in_circle := 360
def degrees_salaries := 216
def percent_salaries := (degrees_salaries * 100) / degrees_in_circle

-- Define the total percentage representation
def total_percent := 100
def known_percent := percent_transportation + percent_utilities + percent_equipment + percent_supplies + percent_salaries

-- Calculate the percent for research and development
def percent_research_and_development := total_percent - known_percent

-- Theorem statement
theorem percent_research_and_development_is_9 : percent_research_and_development = 9 :=
by 
  -- Placeholder for actual proof
  sorry

end percent_research_and_development_is_9_l59_59630


namespace find_angle_F_l59_59472

-- Define the angles of the triangle
variables (D E F : ℝ)

-- Define the conditions given in the problem
def angle_conditions (D E F : ℝ) : Prop :=
  (D = 3 * E) ∧ (E = 18) ∧ (D + E + F = 180)

-- The theorem to prove that angle F is 108 degrees
theorem find_angle_F (D E F : ℝ) (h : angle_conditions D E F) : 
  F = 108 :=
by
  -- The proof body is omitted
  sorry

end find_angle_F_l59_59472


namespace melissa_gave_x_books_l59_59357

-- Define the initial conditions as constants
def initial_melissa_books : ℝ := 123
def initial_jordan_books : ℝ := 27
def final_melissa_books (x : ℝ) : ℝ := initial_melissa_books - x
def final_jordan_books (x : ℝ) : ℝ := initial_jordan_books + x

-- The main theorem to prove how many books Melissa gave to Jordan
theorem melissa_gave_x_books : ∃ x : ℝ, final_melissa_books x = 3 * final_jordan_books x ∧ x = 10.5 :=
sorry

end melissa_gave_x_books_l59_59357


namespace initial_people_count_l59_59772

-- Definitions from conditions
def initial_people (W : ℕ) : ℕ := W
def net_increase : ℕ := 5 - 2
def current_people : ℕ := 19

-- Theorem to prove: initial_people == 16 given conditions
theorem initial_people_count (W : ℕ) (h1 : W + net_increase = current_people) : initial_people W = 16 :=
by
  sorry

end initial_people_count_l59_59772


namespace exists_X_Y_l59_59911

theorem exists_X_Y {A n : ℤ} (h_coprime : Int.gcd A n = 1) :
  ∃ X Y : ℤ, |X| < Int.sqrt n ∧ |Y| < Int.sqrt n ∧ n ∣ (A * X - Y) :=
sorry

end exists_X_Y_l59_59911


namespace sqrt_72_eq_6_sqrt_2_l59_59448

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := 
by
  sorry

end sqrt_72_eq_6_sqrt_2_l59_59448


namespace time_b_started_walking_l59_59561

/-- A's speed is 7 kmph, B's speed is 7.555555555555555 kmph, and B overtakes A after 1.8 hours. -/
theorem time_b_started_walking (t : ℝ) (A_speed : ℝ) (B_speed : ℝ) (overtake_time : ℝ)
    (hA : A_speed = 7) (hB : B_speed = 7.555555555555555) (hOvertake : overtake_time = 1.8) 
    (distance_A : ℝ) (distance_B : ℝ)
    (hDistanceA : distance_A = (t + overtake_time) * A_speed)
    (hDistanceB : distance_B = B_speed * overtake_time) :
  t = 8.57 / 60 := by
  sorry

end time_b_started_walking_l59_59561


namespace probability_sum_less_than_product_l59_59672

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l59_59672


namespace range_of_b_l59_59144

noncomputable def set_A : Set ℝ := {x | -2 < x ∧ x < 1/3}
noncomputable def set_B (b : ℝ) : Set ℝ := {x | x^2 - 4*b*x + 3*b^2 < 0}

theorem range_of_b (b : ℝ) : 
  (set_A ∩ set_B b = ∅) ↔ (b = 0 ∨ b ≥ 1/3 ∨ b ≤ -2) :=
sorry

end range_of_b_l59_59144


namespace union_A_B_l59_59858

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | x > 2}

theorem union_A_B :
  A ∪ B = {x : ℝ | 1 ≤ x} := sorry

end union_A_B_l59_59858


namespace sum_of_ages_is_42_l59_59875

-- Define the variables for present ages of the son (S) and the father (F)
variables (S F : ℕ)

-- Define the conditions:
-- 1. 6 years ago, the father's age was 4 times the son's age.
-- 2. After 6 years, the son's age will be 18 years.

def son_age_condition := S + 6 = 18
def father_age_6_years_ago_condition := F - 6 = 4 * (S - 6)

-- Theorem statement to prove:
theorem sum_of_ages_is_42 (S F : ℕ)
  (h1 : son_age_condition S)
  (h2 : father_age_6_years_ago_condition F S) :
  S + F = 42 :=
sorry

end sum_of_ages_is_42_l59_59875


namespace minimum_workers_needed_l59_59618

noncomputable def units_per_first_worker : Nat := 48
noncomputable def units_per_second_worker : Nat := 32
noncomputable def units_per_third_worker : Nat := 28

def minimum_workers_first_process : Nat := 14
def minimum_workers_second_process : Nat := 21
def minimum_workers_third_process : Nat := 24

def lcm_3_nat (a b c : Nat) : Nat :=
  Nat.lcm (Nat.lcm a b) c

theorem minimum_workers_needed (a b c : Nat) (w1 w2 w3 : Nat)
  (h1 : a = 48) (h2 : b = 32) (h3 : c = 28)
  (hw1 : w1 = minimum_workers_first_process )
  (hw2 : w2 = minimum_workers_second_process )
  (hw3 : w3 = minimum_workers_third_process ) :
  lcm_3_nat a b c / a = w1 ∧ lcm_3_nat a b c / b = w2 ∧ lcm_3_nat a b c / c = w3 :=
by
  sorry

end minimum_workers_needed_l59_59618


namespace new_supervisor_salary_l59_59149

namespace FactorySalaries

variables (W S2 : ℝ)

def old_supervisor_salary : ℝ := 870
def old_average_salary : ℝ := 430
def new_average_salary : ℝ := 440

theorem new_supervisor_salary :
  (W + old_supervisor_salary) / 9 = old_average_salary →
  (W + S2) / 9 = new_average_salary →
  S2 = 960 :=
by
  intros h1 h2
  -- Proof steps would go here
  sorry

end FactorySalaries

end new_supervisor_salary_l59_59149


namespace orthocenter_of_triangle_l59_59778

theorem orthocenter_of_triangle :
  ∀ (A B C H : ℝ × ℝ × ℝ),
    A = (2, 3, 4) → 
    B = (6, 4, 2) → 
    C = (4, 5, 6) → 
    H = (17/53, 152/53, 725/53) → 
    true :=
by sorry

end orthocenter_of_triangle_l59_59778


namespace complement_union_eq_l59_59405

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l59_59405


namespace kylie_total_beads_used_l59_59996

noncomputable def beads_monday_necklaces : ℕ := 10 * 20
noncomputable def beads_tuesday_necklaces : ℕ := 2 * 20
noncomputable def beads_wednesday_bracelets : ℕ := 5 * 10
noncomputable def beads_thursday_earrings : ℕ := 3 * 5
noncomputable def beads_friday_anklets : ℕ := 4 * 8
noncomputable def beads_friday_rings : ℕ := 6 * 7

noncomputable def total_beads_used : ℕ :=
  beads_monday_necklaces +
  beads_tuesday_necklaces +
  beads_wednesday_bracelets +
  beads_thursday_earrings +
  beads_friday_anklets +
  beads_friday_rings

theorem kylie_total_beads_used : total_beads_used = 379 := by
  sorry

end kylie_total_beads_used_l59_59996


namespace parallel_line_dividing_triangle_l59_59714

theorem parallel_line_dividing_triangle (base : ℝ) (length_parallel_line : ℝ) 
    (h_base : base = 24) 
    (h_parallel : (length_parallel_line / base)^2 = 1/2) : 
    length_parallel_line = 12 * Real.sqrt 2 :=
sorry

end parallel_line_dividing_triangle_l59_59714


namespace simplify_abs_expression_l59_59055

/-- Simplify the expression: |-4^3 + 5^2 - 6| and prove the result is equal to 45 -/
theorem simplify_abs_expression :
  |(- 4 ^ 3 + 5 ^ 2 - 6)| = 45 :=
by
  sorry

end simplify_abs_expression_l59_59055


namespace number_of_customers_l59_59162

theorem number_of_customers (offices_sandwiches : Nat)
                            (group_per_person_sandwiches : Nat)
                            (total_sandwiches : Nat)
                            (half_group : Nat) :
  (offices_sandwiches = 3 * 10) →
  (total_sandwiches = 54) →
  (half_group * group_per_person_sandwiches = total_sandwiches - offices_sandwiches) →
  (2 * half_group = 12) := 
by
  sorry

end number_of_customers_l59_59162


namespace third_side_length_l59_59402

/-- Given two sides of a triangle with lengths 4cm and 9cm, prove that the valid length of the third side must be 9cm. -/
theorem third_side_length (a b c : ℝ) (h₀ : a = 4) (h₁ : b = 9) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → (c = 9) :=
by {
  sorry
}

end third_side_length_l59_59402


namespace average_rate_of_change_l59_59976

variable {α : Type*} [LinearOrderedField α]
variable (f : α → α)
variable (x x₁ : α)
variable (h₁ : x ≠ x₁)

theorem average_rate_of_change : 
  (f x₁ - f x) / (x₁ - x) = (f x₁ - f x) / (x₁ - x) :=
by
  sorry

end average_rate_of_change_l59_59976


namespace find_k_inverse_proportion_l59_59544

theorem find_k_inverse_proportion :
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, ∀ y : ℝ, (x = 1 ∧ y = 3) → (y = k / x)) ∧ k = 3 :=
by
  sorry

end find_k_inverse_proportion_l59_59544


namespace number_of_valid_ns_l59_59887

theorem number_of_valid_ns :
  ∃ (n : ℝ), (n = 8 ∨ n = 1/2) ∧ ∀ n₁ n₂, (n₁ = 8 ∨ n₁ = 1/2) ∧ (n₂ = 8 ∨ n₂ = 1/2) → n₁ = n₂ :=
sorry

end number_of_valid_ns_l59_59887


namespace ratio_of_Victoria_to_Beacon_l59_59823

def Richmond_population : ℕ := 3000
def Beacon_population : ℕ := 500
def Victoria_population : ℕ := Richmond_population - 1000
def ratio_Victoria_Beacon : ℕ := Victoria_population / Beacon_population

theorem ratio_of_Victoria_to_Beacon : ratio_Victoria_Beacon = 4 := 
by
  unfold ratio_Victoria_Beacon Victoria_population Richmond_population Beacon_population
  sorry

end ratio_of_Victoria_to_Beacon_l59_59823


namespace fraction_of_girls_l59_59989

variable (total_students : ℕ) (number_of_boys : ℕ)

theorem fraction_of_girls (h1 : total_students = 160) (h2 : number_of_boys = 60) :
    (total_students - number_of_boys) / total_students = 5 / 8 := by
  sorry

end fraction_of_girls_l59_59989


namespace solve_geometric_sequence_product_l59_59414

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r

theorem solve_geometric_sequence_product (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h_a35 : a 3 * a 5 = 4) : 
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 = 128 :=
sorry

end solve_geometric_sequence_product_l59_59414


namespace task_assignment_l59_59883

theorem task_assignment (volunteers : ℕ) (tasks : ℕ) (selected : ℕ) (h_volunteers : volunteers = 6) (h_tasks : tasks = 4) (h_selected : selected = 4) :
  ((Nat.factorial volunteers) / (Nat.factorial (volunteers - selected))) = 360 :=
by
  rw [h_volunteers, h_selected]
  norm_num
  sorry

end task_assignment_l59_59883


namespace expected_winnings_is_correct_l59_59814

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

end expected_winnings_is_correct_l59_59814


namespace similar_triangles_perimeter_l59_59543

open Real

-- Defining the similar triangles and their associated conditions
noncomputable def triangle1 := (4, 6, 8)
noncomputable def side2 := 2

-- Define the possible perimeters of the other triangle
theorem similar_triangles_perimeter (h : True) :
  (∃ x, x = 4.5 ∨ x = 6 ∨ x = 9) :=
sorry

end similar_triangles_perimeter_l59_59543


namespace smallest_prime_perimeter_l59_59705

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_scalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_triple_prime (a b c : ℕ) : Prop := is_prime a ∧ is_prime b ∧ is_prime c

def is_triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

theorem smallest_prime_perimeter :
  ∃ a b c : ℕ, is_scalene a b c ∧ is_triple_prime a b c ∧ is_prime (a + b + c) ∧ a + b + c = 23 :=
sorry

end smallest_prime_perimeter_l59_59705


namespace andrew_game_night_expenses_l59_59834

theorem andrew_game_night_expenses : 
  let cost_per_game := 9 
  let number_of_games := 5 
  total_money_spent = cost_per_game * number_of_games 
→ total_money_spent = 45 := 
by
  intro cost_per_game number_of_games total_money_spent
  sorry

end andrew_game_night_expenses_l59_59834


namespace not_p_and_pq_false_not_necessarily_p_or_q_l59_59437

theorem not_p_and_pq_false_not_necessarily_p_or_q (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : ¬(p ∨ q) ∨ (p ∨ q) := by
  sorry

end not_p_and_pq_false_not_necessarily_p_or_q_l59_59437


namespace objective_function_range_l59_59541

theorem objective_function_range:
  (∃ x y : ℝ, x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1) ∧
  (∀ x y : ℝ, (x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1) →
  (3*x + y ≥ (19:ℝ) / 9 ∧ 3*x + y ≤ 6)) :=
sorry

-- We have defined the conditions, the objective function, and the assertion in Lean 4.

end objective_function_range_l59_59541


namespace total_points_earned_l59_59188

def defeated_enemies := 15
def points_per_enemy := 12
def level_completion_points := 20
def special_challenges_completed := 5
def points_per_special_challenge := 10

theorem total_points_earned :
  defeated_enemies * points_per_enemy
  + level_completion_points
  + special_challenges_completed * points_per_special_challenge = 250 :=
by
  -- The proof would be developed here.
  sorry

end total_points_earned_l59_59188


namespace effective_annual_interest_rate_is_correct_l59_59452

noncomputable def quarterly_interest_rate : ℝ := 0.02

noncomputable def annual_interest_rate (quarterly_rate : ℝ) : ℝ :=
  ((1 + quarterly_rate) ^ 4 - 1) * 100

theorem effective_annual_interest_rate_is_correct :
  annual_interest_rate quarterly_interest_rate = 8.24 :=
by
  sorry

end effective_annual_interest_rate_is_correct_l59_59452


namespace smallest_number_of_eggs_over_150_l59_59579

theorem smallest_number_of_eggs_over_150 
  (d : ℕ) 
  (h1: 12 * d - 3 > 150) 
  (h2: ∀ k < d, 12 * k - 3 ≤ 150) :
  12 * d - 3 = 153 :=
by
  sorry

end smallest_number_of_eggs_over_150_l59_59579


namespace solve_for_x_l59_59220

variable (x y z a b w : ℝ)
variable (angle_DEB : ℝ)

def angle_sum_D (x y z angle_DEB : ℝ) : Prop := x + y + z + angle_DEB = 360
def angle_sum_E (a b w angle_DEB : ℝ) : Prop := a + b + w + angle_DEB = 360

theorem solve_for_x 
  (h1 : angle_sum_D x y z angle_DEB) 
  (h2 : angle_sum_E a b w angle_DEB) : 
  x = a + b + w - y - z :=
by
  -- Proof not required
  sorry

end solve_for_x_l59_59220


namespace inequality_solution_l59_59325

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20) ↔
  (1 < x ∧ x < 2 ∨ 3 < x ∧ x < 6) :=
by
  sorry

end inequality_solution_l59_59325


namespace area_of_right_triangle_l59_59255

-- Define the conditions
def hypotenuse : ℝ := 9
def angle : ℝ := 30

-- Define the Lean statement for the proof problem
theorem area_of_right_triangle : 
  ∃ (area : ℝ), area = 10.125 * Real.sqrt 3 ∧
  ∃ (shorter_leg : ℝ) (longer_leg : ℝ),
    shorter_leg = hypotenuse / 2 ∧
    longer_leg = shorter_leg * Real.sqrt 3 ∧
    area = (shorter_leg * longer_leg) / 2 :=
by {
  -- The proof would go here, but we only need to state the problem for this task.
  sorry
}

end area_of_right_triangle_l59_59255


namespace combustion_moles_l59_59660

-- Chemical reaction definitions
def balanced_equation : Prop :=
  ∀ (CH4 Cl2 O2 CO2 HCl H2O : ℝ),
  1 * CH4 + 4 * Cl2 + 4 * O2 = 1 * CO2 + 4 * HCl + 2 * H2O

-- Moles of substances
def moles_CH4 := 24
def moles_Cl2 := 48
def moles_O2 := 96
def moles_CO2 := 24
def moles_HCl := 48
def moles_H2O := 48

-- Prove the conditions based on the balanced equation
theorem combustion_moles :
  balanced_equation →
  (moles_O2 = 4 * moles_CH4) ∧
  (moles_H2O = 2 * moles_CH4) :=
by {
  sorry
}

end combustion_moles_l59_59660


namespace problem_statement_l59_59718

def P := {x : ℤ | ∃ k : ℤ, x = 2 * k - 1}
def Q := {y : ℤ | ∃ n : ℤ, y = 2 * n}

theorem problem_statement (x y : ℤ) (hx : x ∈ P) (hy : y ∈ Q) :
  (x + y ∈ P) ∧ (x * y ∈ Q) :=
by
  sorry

end problem_statement_l59_59718


namespace aira_fewer_bands_than_joe_l59_59269

-- Define initial conditions
variables (samantha_bands aira_bands joe_bands : ℕ)
variables (shares_each : ℕ) (total_bands: ℕ)

-- Conditions from the problem
axiom h1 : shares_each = 6
axiom h2 : samantha_bands = aira_bands + 5
axiom h3 : total_bands = shares_each * 3
axiom h4 : aira_bands = 4
axiom h5 : samantha_bands + aira_bands + joe_bands = total_bands

-- The statement to be proven
theorem aira_fewer_bands_than_joe : joe_bands - aira_bands = 1 :=
sorry

end aira_fewer_bands_than_joe_l59_59269


namespace find_distance_BC_l59_59953

variables {d_AB d_AC d_BC : ℝ}

theorem find_distance_BC
  (h1 : d_AB = d_AC + d_BC - 200)
  (h2 : d_AC = d_AB + d_BC - 300) :
  d_BC = 250 := 
sorry

end find_distance_BC_l59_59953


namespace juhye_initial_money_l59_59859

theorem juhye_initial_money
  (M : ℝ)
  (h1 : M - (1 / 4) * M - (2 / 3) * ((3 / 4) * M) = 2500) :
  M = 10000 := by
  sorry

end juhye_initial_money_l59_59859


namespace actual_total_area_in_acres_l59_59803

-- Define the conditions
def base_cm : ℝ := 20
def height_cm : ℝ := 12
def rect_length_cm : ℝ := 20
def rect_width_cm : ℝ := 5
def scale_cm_to_miles : ℝ := 3
def sq_mile_to_acres : ℝ := 640

-- Define the total area in acres calculation
def total_area_cm_squared : ℝ := 120 + 100
def total_area_miles_squared : ℝ := total_area_cm_squared * (scale_cm_to_miles ^ 2)
def total_area_acres : ℝ := total_area_miles_squared * sq_mile_to_acres

-- The theorem statement
theorem actual_total_area_in_acres : total_area_acres = 1267200 :=
by
  sorry

end actual_total_area_in_acres_l59_59803


namespace number_of_real_roots_eq_3_eq_m_l59_59699

theorem number_of_real_roots_eq_3_eq_m {x m : ℝ} (h : ∀ x, x^2 - 2 * |x| + 2 = m) : m = 2 :=
sorry

end number_of_real_roots_eq_3_eq_m_l59_59699


namespace intersection_complement_l59_59663

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement : A ∩ (U \ B) = {1, 3} :=
by {
  sorry
}

end intersection_complement_l59_59663


namespace smallest_perfect_cube_divisor_l59_59283

theorem smallest_perfect_cube_divisor (p q r : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] [hr : Fact (Nat.Prime r)] (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ k : ℕ, (k = (p * q * r^2)^3) ∧ (∃ n, n = p * q^3 * r^4 ∧ n ∣ k) := 
sorry

end smallest_perfect_cube_divisor_l59_59283


namespace marbles_in_larger_bottle_l59_59514

theorem marbles_in_larger_bottle 
  (small_bottle_volume : ℕ := 20)
  (small_bottle_marbles : ℕ := 40)
  (larger_bottle_volume : ℕ := 60) :
  (small_bottle_marbles / small_bottle_volume) * larger_bottle_volume = 120 := 
by
  sorry

end marbles_in_larger_bottle_l59_59514


namespace lines_perpendicular_l59_59001

theorem lines_perpendicular (A1 B1 C1 A2 B2 C2 : ℝ) (h : A1 * A2 + B1 * B2 = 0) :
  ∃(x y : ℝ), A1 * x + B1 * y + C1 = 0 ∧ A2 * x + B2 * y + C2 = 0 → A1 * A2 + B1 * B2 = 0 :=
by
  sorry

end lines_perpendicular_l59_59001


namespace simplify_expression_l59_59879

theorem simplify_expression (x y : ℤ) (h1 : x = -2) (h2 : y = 3) :
  (x + 2 * y)^2 - (x + y) * (2 * x - y) = 23 :=
by
  sorry

end simplify_expression_l59_59879


namespace months_rent_in_advance_required_l59_59432

def janet_savings : ℕ := 2225
def rent_per_month : ℕ := 1250
def deposit : ℕ := 500
def additional_needed : ℕ := 775

theorem months_rent_in_advance_required : 
  (janet_savings + additional_needed - deposit) / rent_per_month = 2 :=
by
  sorry

end months_rent_in_advance_required_l59_59432


namespace sqrt_x_div_sqrt_y_l59_59682

theorem sqrt_x_div_sqrt_y (x y : ℝ)
  (h : ( ( (2/3)^2 + (1/6)^2 ) / ( (1/2)^2 + (1/7)^2 ) ) = 28 * x / (25 * y)) :
  (Real.sqrt x) / (Real.sqrt y) = 5 / 2 :=
sorry

end sqrt_x_div_sqrt_y_l59_59682


namespace max_tickets_l59_59386

theorem max_tickets (ticket_price normal_discounted_price budget : ℕ) (h1 : ticket_price = 15) (h2 : normal_discounted_price = 13) (h3 : budget = 180) :
  ∃ n : ℕ, ((n ≤ 10 → ticket_price * n ≤ budget) ∧ (n > 10 → normal_discounted_price * n ≤ budget)) ∧ ∀ m : ℕ, ((m ≤ 10 → ticket_price * m ≤ budget) ∧ (m > 10 → normal_discounted_price * m ≤ budget)) → m ≤ 13 :=
by
  sorry

end max_tickets_l59_59386


namespace perfect_square_l59_59700

-- Define natural numbers m and n and the condition mn ∣ m^2 + n^2 + m
variables (m n : ℕ)

-- Define the condition as a hypothesis
def condition (m n : ℕ) : Prop := (m * n) ∣ (m ^ 2 + n ^ 2 + m)

-- The main theorem statement: if the condition holds, then m is a perfect square
theorem perfect_square (m n : ℕ) (h : condition m n) : ∃ k : ℕ, m = k ^ 2 :=
sorry

end perfect_square_l59_59700


namespace find_a_tangent_line_at_minus_one_l59_59073

-- Define the function f with variable a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f with variable a
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Given conditions
def condition_1 : Prop := f' 1 = 1
def condition_2 : Prop := f' 2 (1 : ℝ) = 1

-- Prove that a = 2 given f'(1) = 1
theorem find_a : f' 2 (1 : ℝ) = 1 → 2 = 2 := by
  sorry

-- Given a = 2, find the tangent line equation at x = -1
def tangent_line_equation (x y : ℝ) : Prop := 9*x - y + 3 = 0

-- Define the coordinates of the point on the curve at x = -1
def point_on_curve : Prop := f 2 (-1) = -6

-- Prove the tangent line equation at x = -1 given a = 2
theorem tangent_line_at_minus_one (h : true) : tangent_line_equation 9 (f' 2 (-1)) := by
  sorry

end find_a_tangent_line_at_minus_one_l59_59073


namespace value_of_place_ratio_l59_59383

theorem value_of_place_ratio :
  let d8_pos := 10000
  let d6_pos := 0.1
  d8_pos = 100000 * d6_pos :=
by
  let d8_pos := 10000
  let d6_pos := 0.1
  sorry

end value_of_place_ratio_l59_59383


namespace at_least_one_not_less_than_2_l59_59307

theorem at_least_one_not_less_than_2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) :=
sorry

end at_least_one_not_less_than_2_l59_59307


namespace grandma_red_bacon_bits_l59_59210

theorem grandma_red_bacon_bits:
  ∀ (mushrooms cherryTomatoes pickles baconBits redBaconBits : ℕ),
    mushrooms = 3 →
    cherryTomatoes = 2 * mushrooms →
    pickles = 4 * cherryTomatoes →
    baconBits = 4 * pickles →
    redBaconBits = 1 / 3 * baconBits →
    redBaconBits = 32 := 
by
  intros mushrooms cherryTomatoes pickles baconBits redBaconBits
  intros h1 h2 h3 h4 h5
  sorry

end grandma_red_bacon_bits_l59_59210


namespace total_amount_divided_into_two_parts_l59_59036

theorem total_amount_divided_into_two_parts (P1 P2 : ℝ) (annual_income : ℝ) :
  P1 = 1500.0000000000007 →
  annual_income = 135 →
  (P1 * 0.05 + P2 * 0.06 = annual_income) →
  P1 + P2 = 2500.000000000000 :=
by
  intros hP1 hIncome hInterest
  sorry

end total_amount_divided_into_two_parts_l59_59036


namespace line_intersects_ellipse_with_conditions_l59_59825

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

end line_intersects_ellipse_with_conditions_l59_59825


namespace sequence_general_term_l59_59166

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 4 else 4 * (-1 / 3)^(n - 1) 

theorem sequence_general_term (n : ℕ) (hn : n ≥ 1) 
  (hrec : ∀ n, 3 * a_n (n + 1) + a_n n = 0)
  (hinit : a_n 2 = -4 / 3) :
  a_n n = 4 * (-1 / 3)^(n - 1) := by
  sorry

end sequence_general_term_l59_59166


namespace noah_total_wattage_l59_59836

def bedroom_wattage := 6
def office_wattage := 3 * bedroom_wattage
def living_room_wattage := 4 * bedroom_wattage
def hours_on := 2

theorem noah_total_wattage : 
  bedroom_wattage * hours_on + 
  office_wattage * hours_on + 
  living_room_wattage * hours_on = 96 := by
  sorry

end noah_total_wattage_l59_59836


namespace ratio_percent_l59_59832

theorem ratio_percent (x : ℕ) (h : (15 / x : ℚ) = 60 / 100) : x = 25 := 
sorry

end ratio_percent_l59_59832


namespace range_of_a_l59_59709

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 2) * x + 6 * a - 1 else a ^ x

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (3 / 8 ≤ a ∧ a < 2 / 3) :=
by
  sorry

end range_of_a_l59_59709


namespace max_value_of_quadratic_l59_59993

-- Define the quadratic function
def f (x : ℝ) : ℝ := 12 * x - 4 * x^2 + 2

-- State the main theorem of finding the maximum value
theorem max_value_of_quadratic : ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = 11 := sorry

end max_value_of_quadratic_l59_59993


namespace right_triangle_angle_ratio_l59_59577

theorem right_triangle_angle_ratio
  (a b : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) 
  (h : a / b = 5 / 4)
  (h3 : a + b = 90) :
  (a = 50) ∧ (b = 40) :=
by
  sorry

end right_triangle_angle_ratio_l59_59577


namespace intersect_at_one_point_l59_59578

-- Define the equations as given in the conditions
def equation1 (b : ℝ) (x : ℝ) : ℝ := b * x ^ 2 + 2 * x + 2
def equation2 (x : ℝ) : ℝ := -2 * x - 2

-- Statement of the theorem
theorem intersect_at_one_point (b : ℝ) :
  (∀ x : ℝ, equation1 b x = equation2 x → x = 1) ↔ b = 1 := sorry

end intersect_at_one_point_l59_59578


namespace milk_revenue_l59_59048

theorem milk_revenue :
  let yesterday_morning := 68
  let yesterday_evening := 82
  let this_morning := yesterday_morning - 18
  let total_milk_before_selling := yesterday_morning + yesterday_evening + this_morning
  let milk_left := 24
  let milk_sold := total_milk_before_selling - milk_left
  let cost_per_gallon := 3.50
  let revenue := milk_sold * cost_per_gallon
  revenue = 616 := by {
    sorry
}

end milk_revenue_l59_59048


namespace slope_of_line_l59_59415

-- Definitions of the conditions in the problem
def line_eq (a : ℝ) (x y : ℝ) : Prop := x + a * y + 1 = 0

def y_intercept (l : ℝ → ℝ → Prop) (b : ℝ) : Prop :=
  l 0 b

-- The statement of the proof problem
theorem slope_of_line (a : ℝ) (h : y_intercept (line_eq a) (-2)) : 
  ∃ (m : ℝ), m = -2 :=
sorry

end slope_of_line_l59_59415


namespace problem_solution_l59_59155

variable (α β : ℝ)

-- Conditions
variable (h1 : 3 * Real.sin α - Real.cos α = 0)
variable (h2 : 7 * Real.sin β + Real.cos β = 0)
variable (h3 : 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π)

theorem problem_solution : 2 * α - β = - (3 * π / 4) := by
  sorry

end problem_solution_l59_59155


namespace minimum_area_sum_l59_59200

-- Define the coordinates and the conditions
variable {x1 y1 x2 y2 : ℝ}
variable (on_parabola_A : y1^2 = x1)
variable (on_parabola_B : y2^2 = x2)
variable (y1_pos : y1 > 0)
variable (y2_neg : y2 < 0)
variable (dot_product : x1 * x2 + y1 * y2 = 2)

-- Define the function to calculate areas
noncomputable def area_sum (y1 y2 x1 x2 : ℝ) : ℝ :=
  1/2 * 2 * (y1 - y2) + 1/2 * 1/4 * y1

theorem minimum_area_sum :
  ∃ y1 y2 x1 x2, y1^2 = x1 ∧ y2^2 = x2 ∧ y1 > 0 ∧ y2 < 0 ∧ x1 * x2 + y1 * y2 = 2 ∧
  (area_sum y1 y2 x1 x2 = 3) := sorry

end minimum_area_sum_l59_59200


namespace classify_triangle_l59_59154

theorem classify_triangle (m : ℕ) (h₁ : m > 1) (h₂ : 3 * m + 3 = 180) :
  (m < 60) ∧ (m + 1 < 90) ∧ (m + 2 < 90) :=
by
  sorry

end classify_triangle_l59_59154


namespace difference_of_profit_share_l59_59906

theorem difference_of_profit_share (a b c : ℕ) (pa pb pc : ℕ) (profit_b : ℕ) 
  (a_capital : a = 8000) (b_capital : b = 10000) (c_capital : c = 12000) 
  (b_profit_share : profit_b = 1600)
  (investment_ratio : pa / 4 = pb / 5 ∧ pb / 5 = pc / 6) :
  pa - pc = 640 := 
sorry

end difference_of_profit_share_l59_59906


namespace rowing_time_l59_59738

theorem rowing_time (rowing_speed : ℕ) (current_speed : ℕ) (distance : ℕ) 
  (h_rowing_speed : rowing_speed = 10)
  (h_current_speed : current_speed = 2)
  (h_distance : distance = 24) : 
  2 * distance / (rowing_speed + current_speed) + 2 * distance / (rowing_speed - current_speed) = 5 :=
by
  rw [h_rowing_speed, h_current_speed, h_distance]
  norm_num
  sorry

end rowing_time_l59_59738


namespace avg_height_students_l59_59956

theorem avg_height_students 
  (x : ℕ)  -- number of students in the first group
  (avg_height_first_group : ℕ)  -- average height of the first group
  (avg_height_second_group : ℕ)  -- average height of the second group
  (avg_height_combined_group : ℕ)  -- average height of the combined group
  (h1 : avg_height_first_group = 20)
  (h2 : avg_height_second_group = 20)
  (h3 : avg_height_combined_group = 20)
  (h4 : 20*x + 20*11 = 20*31) :
  x = 20 := 
  by {
    sorry
  }

end avg_height_students_l59_59956


namespace find_function_and_max_profit_l59_59038

noncomputable def profit_function (x : ℝ) : ℝ := -50 * x^2 + 1200 * x - 6400

theorem find_function_and_max_profit :
  (∀ (x : ℝ), (x = 10 → (-50 * x + 800 = 300)) ∧ (x = 13 → (-50 * x + 800 = 150))) ∧
  (∃ (x : ℝ), x = 12 ∧ profit_function x = 800) :=
by
  sorry

end find_function_and_max_profit_l59_59038


namespace constant_term_in_first_equation_l59_59594

/-- Given the system of equations:
  1. 5x + y = C
  2. x + 3y = 1
  3. 3x + 2y = 10
  Prove that the constant term C is 19.
-/
theorem constant_term_in_first_equation
  (x y C : ℝ)
  (h1 : 5 * x + y = C)
  (h2 : x + 3 * y = 1)
  (h3 : 3 * x + 2 * y = 10) :
  C = 19 :=
by
  sorry

end constant_term_in_first_equation_l59_59594


namespace cube_edge_length_surface_area_equals_volume_l59_59739

theorem cube_edge_length_surface_area_equals_volume (a : ℝ) (h : 6 * a ^ 2 = a ^ 3) : a = 6 := 
by {
  sorry
}

end cube_edge_length_surface_area_equals_volume_l59_59739


namespace ratio_sheep_to_horses_l59_59724

theorem ratio_sheep_to_horses 
  (horse_food_per_day : ℕ) 
  (total_horse_food : ℕ) 
  (num_sheep : ℕ) 
  (H1 : horse_food_per_day = 230) 
  (H2 : total_horse_food = 12880) 
  (H3 : num_sheep = 48) 
  : (num_sheep : ℚ) / (total_horse_food / horse_food_per_day : ℚ) = 6 / 7
  :=
by
  sorry

end ratio_sheep_to_horses_l59_59724


namespace find_r_l59_59044

theorem find_r (k r : ℝ) : 
  5 = k * 3^r ∧ 45 = k * 9^r → r = 2 :=
by 
  sorry

end find_r_l59_59044


namespace minimum_detectors_203_l59_59810

def minimum_detectors (length : ℕ) : ℕ :=
  length / 3 * 2 -- This models the generalization for 1 × (3k + 2)

theorem minimum_detectors_203 : minimum_detectors 203 = 134 :=
by
  -- Length is 203, k = 67 which follows from the floor division
  -- Therefore, minimum detectors = 2 * 67 = 134
  sorry

end minimum_detectors_203_l59_59810


namespace geostationary_orbit_distance_l59_59764

noncomputable def distance_between_stations (earth_radius : ℝ) (orbit_altitude : ℝ) (num_stations : ℕ) : ℝ :=
  let θ : ℝ := 360 / num_stations
  let R : ℝ := earth_radius + orbit_altitude
  let sin_18 := (Real.sqrt 5 - 1) / 4
  2 * R * sin_18

theorem geostationary_orbit_distance :
  distance_between_stations 3960 22236 10 = -13098 + 13098 * Real.sqrt 5 :=
by
  sorry

end geostationary_orbit_distance_l59_59764


namespace exists_odd_integers_l59_59118

theorem exists_odd_integers (n : ℕ) (hn : n ≥ 3) : 
  ∃ x y : ℤ, x % 2 = 1 ∧ y % 2 = 1 ∧ x^2 + 7 * y^2 = 2^n :=
sorry

end exists_odd_integers_l59_59118


namespace maximum_candy_leftover_l59_59639

theorem maximum_candy_leftover (x : ℕ) 
  (h1 : ∀ (bags : ℕ), bags = 12 → x ≥ bags * 10)
  (h2 : ∃ (leftover : ℕ), leftover < 12 ∧ leftover = (x - 120) % 12) : 
  ∃ (leftover : ℕ), leftover = 11 :=
by
  sorry

end maximum_candy_leftover_l59_59639


namespace quadratic_roots_ratio_l59_59580

theorem quadratic_roots_ratio (p x1 x2 : ℝ) (h_eq : x1^2 + p * x1 - 16 = 0) (h_ratio : x1 / x2 = -4) :
  p = 6 ∨ p = -6 :=
by {
  sorry
}

end quadratic_roots_ratio_l59_59580


namespace sqrt_expression_eval_l59_59103

theorem sqrt_expression_eval :
  (Real.sqrt 8) + (Real.sqrt (1 / 2)) + (Real.sqrt 3 - 1) ^ 2 + (Real.sqrt 6 / (1 / 2 * Real.sqrt 2)) = (5 / 2) * Real.sqrt 2 + 4 := 
by
  sorry

end sqrt_expression_eval_l59_59103


namespace max_product_of_triangle_sides_l59_59861

theorem max_product_of_triangle_sides (a c : ℝ) (ha : a ≥ 0) (hc : c ≥ 0) :
  ∃ b : ℝ, b = 4 ∧ ∃ B : ℝ, B = 60 * (π / 180) ∧ a^2 + c^2 - a * c = b^2 ∧ a * c ≤ 16 :=
by
  sorry

end max_product_of_triangle_sides_l59_59861


namespace ellipse_equation_midpoint_coordinates_l59_59646

noncomputable def ellipse_c := {x : ℝ × ℝ | (x.1^2 / 25) + (x.2^2 / 16) = 1}

theorem ellipse_equation (a b : ℝ) (h1 : a = 5) (h2 : b = 4) :
    ∀ x y : ℝ, x = 0 → y = 4 → (y^2 / b^2 = 1) ∧ (e = 3 / 5) → 
      (a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) := 
sorry

theorem midpoint_coordinates (a b : ℝ) (h1 : a = 5) (h2 : b = 4) :
    ∀ x y x1 x2 y1 y2 : ℝ, 
    (y = 4 / 5 * (x - 3)) → 
    (y1 = 4 / 5 * (x1 - 3)) ∧ (y2 = 4 / 5 * (x2 - 3)) ∧ 
    (x1^2 / a^2) + ((y1 - 3)^2 / b^2) = 1 ∧ (x2^2 / a^2) + ((y2 - 3)^2 / b^2) = 1 ∧ 
    (x1 + x2 = 3) → 
    ((x1 + x2) / 2 = 3 / 2) ∧ ((y1 + y2) / 2 = -6 / 5) := 
sorry

end ellipse_equation_midpoint_coordinates_l59_59646


namespace parallelepiped_intersection_l59_59097

/-- Given a parallelepiped A B C D A₁ B₁ C₁ D₁.
    Point X is chosen on edge A₁ D₁, and point Y is chosen on edge B C.
    It is known that A₁ X = 5, B Y = 3, and B₁ C₁ = 14.
    The plane C₁ X Y intersects ray D A at point Z.
    Prove that D Z = 20. -/
theorem parallelepiped_intersection
  (A B C D A₁ B₁ C₁ D₁ X Y Z : ℝ)
  (h₁: A₁ - X = 5)
  (h₂: B - Y = 3)
  (h₃: B₁ - C₁ = 14) :
  D - Z = 20 :=
sorry

end parallelepiped_intersection_l59_59097


namespace correct_operation_l59_59787

theorem correct_operation (a b : ℝ) : 
  (-a^3 * b)^2 = a^6 * b^2 :=
by
  sorry

end correct_operation_l59_59787


namespace find_four_numbers_l59_59138

theorem find_four_numbers 
    (a b c d : ℕ) 
    (h1 : b - a = c - b)  -- first three numbers form an arithmetic sequence
    (h2 : d / c = c / (b - a + b))  -- last three numbers form a geometric sequence
    (h3 : a + d = 16)  -- sum of first and last numbers is 16
    (h4 : b + (12 - b) = 12)  -- sum of the two middle numbers is 12
    : (a = 15 ∧ b = 9 ∧ c = 3 ∧ d = 1) ∨ (a = 0 ∧ b = 4 ∧ c = 8 ∧ d = 16) :=
by
  -- Proof will be provided here
  sorry

end find_four_numbers_l59_59138


namespace number_of_ways_to_make_78_rubles_l59_59245

theorem number_of_ways_to_make_78_rubles : ∃ n, n = 5 ∧ ∃ x y : ℕ, 78 = 5 * x + 3 * y := sorry

end number_of_ways_to_make_78_rubles_l59_59245


namespace find_f_when_x_lt_0_l59_59106

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_defined (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 2 * x

theorem find_f_when_x_lt_0 (f : ℝ → ℝ) (h_odd : odd_function f) (h_defined : f_defined f) :
  ∀ x < 0, f x = -x^2 - 2 * x :=
by
  sorry

end find_f_when_x_lt_0_l59_59106


namespace eccentricity_of_hyperbola_l59_59631

theorem eccentricity_of_hyperbola {a b c e : ℝ} (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : b = 2 * a)
  (h₄ : c^2 = a^2 + b^2) :
  e = Real.sqrt 5 :=
by
  sorry

end eccentricity_of_hyperbola_l59_59631


namespace prop_2_prop_3_l59_59088

variables {a b c : ℝ}

-- Proposition 2: a > |b| -> a^2 > b^2
theorem prop_2 (h : a > |b|) : a^2 > b^2 := sorry

-- Proposition 3: a > b -> a^3 > b^3
theorem prop_3 (h : a > b) : a^3 > b^3 := sorry

end prop_2_prop_3_l59_59088


namespace brinley_animal_count_l59_59051

def snakes : ℕ := 100
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters : ℕ := 12 * leopards
def cheetahs : ℕ := snakes / 3  -- rounding down implicitly considered
def alligators : ℕ := 2 * (arctic_foxes + leopards)
def total_animals : ℕ := snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

theorem brinley_animal_count : total_animals = 673 :=
by
  -- Mathematical proof would go here.
  sorry

end brinley_animal_count_l59_59051


namespace equidistant_points_eq_two_l59_59862

noncomputable def number_of_equidistant_points (O : Point) (r d : ℝ) 
  (h1 : d > r) : ℕ := 
2

theorem equidistant_points_eq_two (O : Point) (r d : ℝ) 
  (h1 : d > r) : number_of_equidistant_points O r d h1 = 2 :=
by
  sorry

end equidistant_points_eq_two_l59_59862


namespace ravi_overall_profit_l59_59330

-- Define the cost price of the refrigerator and the mobile phone
def cost_price_refrigerator : ℝ := 15000
def cost_price_mobile_phone : ℝ := 8000

-- Define the loss percentage for the refrigerator and the profit percentage for the mobile phone
def loss_percentage_refrigerator : ℝ := 0.05
def profit_percentage_mobile_phone : ℝ := 0.10

-- Calculate the loss amount and the selling price of the refrigerator
def loss_amount_refrigerator : ℝ := loss_percentage_refrigerator * cost_price_refrigerator
def selling_price_refrigerator : ℝ := cost_price_refrigerator - loss_amount_refrigerator

-- Calculate the profit amount and the selling price of the mobile phone
def profit_amount_mobile_phone : ℝ := profit_percentage_mobile_phone * cost_price_mobile_phone
def selling_price_mobile_phone : ℝ := cost_price_mobile_phone + profit_amount_mobile_phone

-- Calculate the total cost price and the total selling price
def total_cost_price : ℝ := cost_price_refrigerator + cost_price_mobile_phone
def total_selling_price : ℝ := selling_price_refrigerator + selling_price_mobile_phone

-- Calculate the overall profit or loss
def overall_profit_or_loss : ℝ := total_selling_price - total_cost_price

theorem ravi_overall_profit : overall_profit_or_loss = 50 := 
by
  sorry

end ravi_overall_profit_l59_59330


namespace gifts_left_l59_59470

variable (initial_gifts : ℕ)
variable (gifts_sent : ℕ)

theorem gifts_left (h_initial : initial_gifts = 77) (h_sent : gifts_sent = 66) : initial_gifts - gifts_sent = 11 := by
  sorry

end gifts_left_l59_59470


namespace age_difference_between_brother_and_cousin_is_five_l59_59937

variable (Lexie_age brother_age sister_age uncle_age grandma_age cousin_age : ℕ)

-- Conditions
axiom lexie_age_def : Lexie_age = 8
axiom grandma_age_def : grandma_age = 68
axiom lexie_brother_condition : Lexie_age = brother_age + 6
axiom lexie_sister_condition : sister_age = 2 * Lexie_age
axiom uncle_grandma_condition : uncle_age = grandma_age - 12
axiom cousin_brother_condition : cousin_age = brother_age + 5

-- Goal
theorem age_difference_between_brother_and_cousin_is_five : 
  Lexie_age = 8 → grandma_age = 68 → brother_age = Lexie_age - 6 → cousin_age = brother_age + 5 → cousin_age - brother_age = 5 :=
by sorry

end age_difference_between_brother_and_cousin_is_five_l59_59937


namespace math_problem_l59_59066

theorem math_problem :
  101 * 102^2 - 101 * 98^2 = 80800 :=
by
  sorry

end math_problem_l59_59066


namespace find_n_l59_59094

def exp (m n : ℕ) : ℕ := m ^ n

-- Now we restate the problem formally
theorem find_n 
  (m n : ℕ) 
  (h1 : exp 10 m = n * 22) : 
  n = 10^m / 22 := 
sorry

end find_n_l59_59094


namespace non_shaded_region_perimeter_l59_59012

def outer_rectangle_length : ℕ := 12
def outer_rectangle_width : ℕ := 10
def inner_rectangle_length : ℕ := 6
def inner_rectangle_width : ℕ := 2
def shaded_area : ℕ := 116

theorem non_shaded_region_perimeter :
  let total_area := outer_rectangle_length * outer_rectangle_width
  let inner_area := inner_rectangle_length * inner_rectangle_width
  let non_shaded_area := total_area - shaded_area
  non_shaded_area = 4 →
  ∃ width height, width * height = non_shaded_area ∧ 2 * (width + height) = 10 :=
by intros
   sorry

end non_shaded_region_perimeter_l59_59012


namespace is_quadratic_l59_59037

theorem is_quadratic (A B C D : Prop) :
  (A = (∀ x : ℝ, x + (1 / x) = 0)) ∧
  (B = (∀ x y : ℝ, x + x * y + 1 = 0)) ∧
  (C = (∀ x : ℝ, 3 * x + 2 = 0)) ∧
  (D = (∀ x : ℝ, x^2 + 2 * x = 1)) →
  D := 
by
  sorry

end is_quadratic_l59_59037


namespace ticket_price_divisor_l59_59507

def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

def GCD (a b : ℕ) := Nat.gcd a b

theorem ticket_price_divisor :
  let total7 := 70
  let total8 := 98
  let y := 4
  is_divisor (GCD total7 total8) y :=
by
  sorry

end ticket_price_divisor_l59_59507


namespace total_bill_is_correct_l59_59434

-- Given conditions
def hourly_rate := 45
def parts_cost := 225
def hours_worked := 5

-- Total bill calculation
def labor_cost := hourly_rate * hours_worked
def total_bill := labor_cost + parts_cost

-- Prove that the total bill is equal to 450 dollars
theorem total_bill_is_correct : total_bill = 450 := by
  sorry

end total_bill_is_correct_l59_59434


namespace boxes_in_carton_of_pencils_l59_59532

def cost_per_box_pencil : ℕ := 2
def cost_per_box_marker : ℕ := 4
def boxes_per_carton_marker : ℕ := 5
def cartons_of_pencils : ℕ := 20
def cartons_of_markers : ℕ := 10
def total_spent : ℕ := 600

theorem boxes_in_carton_of_pencils : ∃ x : ℕ, 20 * (2 * x) + 10 * (5 * 4) = 600 :=
by
  sorry

end boxes_in_carton_of_pencils_l59_59532


namespace ben_is_10_l59_59537

-- Define the ages of the cousins
def ages : List ℕ := [6, 8, 10, 12, 14]

-- Define the conditions
def wentToPark (x y : ℕ) : Prop := x + y = 18
def wentToLibrary (x y : ℕ) : Prop := x + y < 20
def stayedHome (ben young : ℕ) : Prop := young = 6 ∧ ben ∈ ages ∧ ben ≠ 6 ∧ ben ≠ 12

-- The main theorem stating Ben's age
theorem ben_is_10 : ∃ ben, stayedHome ben 6 ∧ 
  (∃ x y, wentToPark x y ∧ x ∈ ages ∧ y ∈ ages ∧ x ≠ y ∧ x ≠ ben ∧ y ≠ ben) ∧
  (∃ x y, wentToLibrary x y ∧ x ∈ ages ∧ y ∈ ages ∧ x ≠ y ∧ x ≠ ben ∧ y ≠ ben) :=
by
  use 10
  -- Proof steps would go here
  sorry

end ben_is_10_l59_59537


namespace ab_fraction_inequality_l59_59127

theorem ab_fraction_inequality (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (hb1 : b < 1) :
  (a * b * (1 - a) * (1 - b)) / ((1 - a * b) ^ 2) < 1 / 4 :=
by
  sorry

end ab_fraction_inequality_l59_59127


namespace solve_equation_1_solve_equation_2_solve_equation_3_l59_59703

theorem solve_equation_1 (x : ℝ) : (x^2 - 3 * x = 0) ↔ (x = 0 ∨ x = 3) := sorry

theorem solve_equation_2 (x : ℝ) : (4 * x^2 - x - 5 = 0) ↔ (x = 5/4 ∨ x = -1) := sorry

theorem solve_equation_3 (x : ℝ) : (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2/3) := sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l59_59703


namespace marble_count_l59_59562

theorem marble_count (p y v : ℝ) (h1 : y + v = 10) (h2 : p + v = 12) (h3 : p + y = 5) :
  p + y + v = 13.5 :=
sorry

end marble_count_l59_59562


namespace infinite_non_prime_numbers_l59_59782

theorem infinite_non_prime_numbers : ∀ (n : ℕ), ∃ (m : ℕ), m ≥ n ∧ (¬(Nat.Prime (2 ^ (2 ^ m) + 1) ∨ ¬Nat.Prime (2018 ^ (2 ^ m) + 1))) := sorry

end infinite_non_prime_numbers_l59_59782


namespace age_discrepancy_l59_59943

theorem age_discrepancy (R G M F A : ℕ)
  (hR : R = 12)
  (hG : G = 7 * R)
  (hM : M = G / 2)
  (hF : F = M + 5)
  (hA : A = G - 8)
  (hDiff : A - F = 10) :
  false :=
by
  -- proofs and calculations leading to contradiction go here
  sorry

end age_discrepancy_l59_59943


namespace reciprocal_roots_condition_l59_59291

theorem reciprocal_roots_condition (a b c : ℝ) (h : a ≠ 0) (roots_reciprocal : ∃ r s : ℝ, r * s = 1 ∧ r + s = -b/a ∧ r * s = c/a) : c = a :=
by
  sorry

end reciprocal_roots_condition_l59_59291


namespace total_weight_is_40_l59_59450

def marco_strawberries_weight : ℕ := 8
def dad_strawberries_weight : ℕ := 32
def total_strawberries_weight := marco_strawberries_weight + dad_strawberries_weight

theorem total_weight_is_40 : total_strawberries_weight = 40 := by
  sorry

end total_weight_is_40_l59_59450


namespace product_b2_b7_l59_59896

def is_increasing_arithmetic_sequence (bs : ℕ → ℤ) :=
  ∀ n m : ℕ, n < m → bs n < bs m

def arithmetic_sequence (bs : ℕ → ℤ) (d : ℤ) :=
  ∀ n : ℕ, bs (n + 1) - bs n = d

theorem product_b2_b7 (bs : ℕ → ℤ) (d : ℤ) (h_incr : is_increasing_arithmetic_sequence bs)
    (h_arith : arithmetic_sequence bs d)
    (h_prod : bs 4 * bs 5 = 10) :
    bs 2 * bs 7 = -224 ∨ bs 2 * bs 7 = -44 :=
by
  sorry

end product_b2_b7_l59_59896


namespace shaded_area_l59_59311

noncomputable def area_of_shaded_region (AB : ℝ) (pi_approx : ℝ) : ℝ :=
  let R := AB / 2
  let r := R / 2
  let A_large := (1/2) * pi_approx * R^2
  let A_small := (1/2) * pi_approx * r^2
  2 * A_large - 4 * A_small

theorem shaded_area (h : area_of_shaded_region 40 3.14 = 628) : true :=
  sorry

end shaded_area_l59_59311


namespace field_trip_classrooms_count_l59_59991

variable (students : ℕ) (seats_per_bus : ℕ) (number_of_buses : ℕ) (total_classrooms : ℕ)

def fieldTrip 
    (students := 58)
    (seats_per_bus := 2)
    (number_of_buses := 29)
    (total_classrooms := 2) : Prop :=
  students = seats_per_bus * number_of_buses  ∧ total_classrooms = students / (students / total_classrooms)

theorem field_trip_classrooms_count : fieldTrip := by
  -- Proof goes here
  sorry

end field_trip_classrooms_count_l59_59991


namespace blocks_per_box_l59_59083

theorem blocks_per_box (total_blocks : ℕ) (boxes : ℕ) (h1 : total_blocks = 16) (h2 : boxes = 8) : total_blocks / boxes = 2 :=
by
  sorry

end blocks_per_box_l59_59083


namespace tan_alpha_value_l59_59712

theorem tan_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α * Real.cos α = 1 / 4) :
  Real.tan α = 2 - Real.sqrt 3 ∨ Real.tan α = 2 + Real.sqrt 3 :=
sorry

end tan_alpha_value_l59_59712


namespace download_time_is_2_hours_l59_59013

theorem download_time_is_2_hours (internet_speed : ℕ) (f1 f2 f3 : ℕ) (total_size : ℕ)
  (total_min : ℕ) (hours : ℕ) :
  internet_speed = 2 ∧ f1 = 80 ∧ f2 = 90 ∧ f3 = 70 ∧ total_size = f1 + f2 + f3
  ∧ total_min = total_size / internet_speed ∧ hours = total_min / 60 → hours = 2 :=
by
  sorry

end download_time_is_2_hours_l59_59013


namespace menkara_index_card_area_l59_59954

theorem menkara_index_card_area :
  ∀ (length width: ℕ), 
  length = 5 → width = 7 → (length - 2) * width = 21 → 
  (length * (width - 2) = 25) :=
by
  intros length width h_length h_width h_area
  sorry

end menkara_index_card_area_l59_59954


namespace find_p_q_l59_59049

theorem find_p_q (D : ℝ) (p q : ℝ) (h_roots : ∀ x, x^2 + p * x + q = 0 → (x = D ∨ x = 1 - D))
  (h_discriminant : D = p^2 - 4 * q) :
  (p = -1 ∧ q = 0) ∨ (p = -1 ∧ q = 3 / 16) :=
by
  sorry

end find_p_q_l59_59049


namespace emma_possible_lists_l59_59137

-- Define the number of balls
def number_of_balls : ℕ := 24

-- Define the number of draws Emma repeats independently
def number_of_draws : ℕ := 4

-- Define the calculation for the total number of different lists
def total_number_of_lists : ℕ := number_of_balls ^ number_of_draws

theorem emma_possible_lists : total_number_of_lists = 331776 := by
  sorry

end emma_possible_lists_l59_59137


namespace bug_crawl_distance_l59_59140

-- Define the positions visited by the bug
def start_position := -3
def first_stop := 0
def second_stop := -8
def final_stop := 10

-- Define the function to calculate the total distance crawled by the bug
def total_distance : ℤ :=
  abs (first_stop - start_position) + abs (second_stop - first_stop) + abs (final_stop - second_stop)

-- Prove that the total distance is 29 units
theorem bug_crawl_distance : total_distance = 29 :=
by
  -- Definitions are used here to validate the statement
  sorry

end bug_crawl_distance_l59_59140


namespace transform_roots_to_quadratic_l59_59559

noncomputable def quadratic_formula (p q : ℝ) (x : ℝ) : ℝ :=
  x^2 + p * x + q

theorem transform_roots_to_quadratic (x₁ x₂ y₁ y₂ p q : ℝ)
  (h₁ : quadratic_formula p q x₁ = 0)
  (h₂ : quadratic_formula p q x₂ = 0)
  (h₃ : x₁ ≠ 1)
  (h₄ : x₂ ≠ 1)
  (hy₁ : y₁ = (x₁ + 1) / (x₁ - 1))
  (hy₂ : y₂ = (x₂ + 1) / (x₂ - 1)) :
  (1 + p + q) * y₁^2 + 2 * (1 - q) * y₁ + (1 - p + q) = 0 ∧
  (1 + p + q) * y₂^2 + 2 * (1 - q) * y₂ + (1 - p + q) = 0 := 
sorry

end transform_roots_to_quadratic_l59_59559


namespace exist_ordering_rectangles_l59_59936

open Function

structure Rectangle :=
  (left_bot : ℝ × ℝ)  -- Bottom-left corner
  (right_top : ℝ × ℝ)  -- Top-right corner

def below (R1 R2 : Rectangle) : Prop :=
  ∃ g : ℝ, (∀ (x y : ℝ), R1.left_bot.1 ≤ x ∧ x ≤ R1.right_top.1 ∧ R1.left_bot.2 ≤ y ∧ y ≤ R1.right_top.2 → y < g) ∧
           (∀ (x y : ℝ), R2.left_bot.1 <= x ∧ x <= R2.right_top.1 ∧ R2.left_bot.2 <= y ∧ y <= R2.right_top.2 → y > g)

def to_right_of (R1 R2 : Rectangle) : Prop :=
  ∃ h : ℝ, (∀ (x y : ℝ), R1.left_bot.1 ≤ x ∧ x ≤ R1.right_top.1 ∧ R1.left_bot.2 ≤ y ∧ y ≤ R1.right_top.2 → x > h) ∧
           (∀ (x y : ℝ), R2.left_bot.1 <= x ∧ x <= R2.right_top.1 ∧ R2.left_bot.2 <= y ∧ y <= R2.right_top.2 → x < h)

def disjoint (R1 R2 : Rectangle) : Prop :=
  ¬ ((R1.left_bot.1 < R2.right_top.1) ∧ (R1.right_top.1 > R2.left_bot.1) ∧
     (R1.left_bot.2 < R2.right_top.2) ∧ (R1.right_top.2 > R2.left_bot.2))

theorem exist_ordering_rectangles (n : ℕ) (rectangles : Fin n → Rectangle)
  (h_disjoint : ∀ i j, i ≠ j → disjoint (rectangles i) (rectangles j)) :
  ∃ f : Fin n → Fin n, ∀ i j : Fin n, i < j → 
    (to_right_of (rectangles (f i)) (rectangles (f j)) ∨ 
    below (rectangles (f i)) (rectangles (f j))) := 
sorry

end exist_ordering_rectangles_l59_59936


namespace minimize_expression_l59_59335

theorem minimize_expression :
  ∀ n : ℕ, 0 < n → (n = 6 ↔ ∀ m : ℕ, 0 < m → (n ≤ (2 * (m + 9))/(m))) := 
by
  sorry

end minimize_expression_l59_59335


namespace razorback_tshirt_profit_l59_59497

theorem razorback_tshirt_profit
  (total_tshirts_sold : ℕ)
  (tshirts_sold_arkansas_game : ℕ)
  (money_made_arkansas_game : ℕ) :
  total_tshirts_sold = 163 →
  tshirts_sold_arkansas_game = 89 →
  money_made_arkansas_game = 8722 →
  money_made_arkansas_game / tshirts_sold_arkansas_game = 98 :=
by 
  intros _ _ _
  sorry

end razorback_tshirt_profit_l59_59497


namespace negation_of_prop_l59_59808

open Classical

theorem negation_of_prop (h : ∀ x : ℝ, x^2 + x + 1 > 0) : ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
sorry

end negation_of_prop_l59_59808


namespace evaluate_expression_l59_59377

theorem evaluate_expression :
  let a := Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
  let b := - Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
  let c := Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
  let d := - Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
  (1/a + 1/b + 1/c + 1/d)^2 = 5 :=
by
  sorry

end evaluate_expression_l59_59377


namespace divide_fractions_l59_59211

theorem divide_fractions : (3 / 8) / (1 / 4) = 3 / 2 :=
by sorry

end divide_fractions_l59_59211


namespace no_real_roots_of_quad_eq_l59_59817

theorem no_real_roots_of_quad_eq (k : ℝ) :
  ∀ x : ℝ, ¬ (x^2 - 2*x - k = 0) ↔ k < -1 :=
by sorry

end no_real_roots_of_quad_eq_l59_59817


namespace line_through_center_eq_line_chord_len_eq_l59_59909

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

noncomputable def point_P : ℝ × ℝ := (2, 2)

def line_through_center (x y : ℝ) : Prop := 2 * x - y - 2 = 0

def line_chord_len (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0 ∨ x = 2

theorem line_through_center_eq (x y : ℝ) (hC : circle_eq x y) :
  line_through_center x y :=
sorry

theorem line_chord_len_eq (x y : ℝ) (hC : circle_eq x y) (hP : x = 2 ∧ y = 2 ∧ (line_through_center x y)) :
  line_chord_len x y :=
sorry

end line_through_center_eq_line_chord_len_eq_l59_59909


namespace proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l59_59104

open Classical

variable (a b x y : ℝ)

theorem proposition_A_correct (h : a > 1) : (1 / a < 1) ∧ ¬((1 / a < 1) → (a > 1)) :=
sorry

theorem proposition_B_incorrect (h_neg : ¬(x < 1 → x^2 < 1)) : ¬(∃ x, x ≥ 1 ∧ x^2 ≥ 1) :=
sorry

theorem proposition_C_incorrect (h_xy : x ≥ 2 ∧ y ≥ 2) : ¬((x ≥ 2 ∧ y ≥ 2) → x^2 + y^2 ≥ 4) :=
sorry

theorem proposition_D_correct (h_a : a ≠ 0) : (a * b ≠ 0) ∧ ¬((a * b ≠ 0) → (a ≠ 0)) :=
sorry

end proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l59_59104


namespace mask_distribution_l59_59358

theorem mask_distribution (x : ℕ) (total_masks_3 : ℕ) (total_masks_4 : ℕ)
    (h1 : total_masks_3 = 3 * x + 20)
    (h2 : total_masks_4 = 4 * x - 25) :
    3 * x + 20 = 4 * x - 25 :=
by
  sorry

end mask_distribution_l59_59358


namespace pencils_are_left_l59_59867

-- Define the conditions
def original_pencils : ℕ := 87
def removed_pencils : ℕ := 4

-- Define the expected outcome
def pencils_left : ℕ := original_pencils - removed_pencils

-- Prove that the number of pencils left in the jar is 83
theorem pencils_are_left : pencils_left = 83 := by
  -- Placeholder for the proof
  sorry

end pencils_are_left_l59_59867


namespace a_4_is_4_l59_59261

-- Define the general term formula of the sequence
def a (n : ℕ) : ℤ := (-1)^n * n

-- State the desired proof goal
theorem a_4_is_4 : a 4 = 4 :=
by
  -- Proof to be provided here,
  -- adding 'sorry' as we are only defining the statement, not solving it
  sorry

end a_4_is_4_l59_59261


namespace sticker_sum_mod_problem_l59_59034

theorem sticker_sum_mod_problem :
  ∃ N < 100, (N % 6 = 5) ∧ (N % 8 = 6) ∧ (N = 47 ∨ N = 95) ∧ (47 + 95 = 142) :=
by
  sorry

end sticker_sum_mod_problem_l59_59034


namespace number_of_ordered_pairs_l59_59298

noncomputable def count_valid_ordered_pairs (a b: ℝ) : Prop :=
  ∃ (x y : ℤ), a * (x : ℝ) + b * (y : ℝ) = 2 ∧ x^2 + y^2 = 65

theorem number_of_ordered_pairs : ∃ s : Finset (ℝ × ℝ), s.card = 128 ∧ ∀ (p : ℝ × ℝ), p ∈ s ↔ count_valid_ordered_pairs p.1 p.2 :=
by
  sorry

end number_of_ordered_pairs_l59_59298


namespace reciprocal_of_sum_of_fractions_l59_59480

theorem reciprocal_of_sum_of_fractions :
  (1 / (1 / 4 + 1 / 6)) = 12 / 5 :=
by
  sorry

end reciprocal_of_sum_of_fractions_l59_59480


namespace gcd_7392_15015_l59_59403

-- Define the two numbers
def num1 : ℕ := 7392
def num2 : ℕ := 15015

-- State the theorem and use sorry to omit the proof
theorem gcd_7392_15015 : Nat.gcd num1 num2 = 1 := 
  by sorry

end gcd_7392_15015_l59_59403


namespace cosine_triangle_ABC_l59_59722

noncomputable def triangle_cosine_proof (a b : ℝ) (A : ℝ) (cosB : ℝ) : Prop :=
  let sinA := Real.sin A
  let sinB := b * sinA / a
  let cosB_expr := Real.sqrt (1 - sinB^2)
  cosB = cosB_expr

theorem cosine_triangle_ABC : triangle_cosine_proof (Real.sqrt 7) 2 (Real.pi / 4) (Real.sqrt 35 / 7) :=
by
  sorry

end cosine_triangle_ABC_l59_59722


namespace carol_has_35_nickels_l59_59809

def problem_statement : Prop :=
  ∃ (n d : ℕ), 5 * n + 10 * d = 455 ∧ n = d + 7 ∧ n = 35

theorem carol_has_35_nickels : problem_statement := by
  -- Proof goes here
  sorry

end carol_has_35_nickels_l59_59809


namespace Sophie_Spends_72_80_l59_59248

noncomputable def SophieTotalCost : ℝ :=
  let cupcakesCost := 5 * 2
  let doughnutsCost := 6 * 1
  let applePieCost := 4 * 2
  let cookiesCost := 15 * 0.60
  let chocolateBarsCost := 8 * 1.50
  let sodaCost := 12 * 1.20
  let gumCost := 3 * 0.80
  let chipsCost := 10 * 1.10
  cupcakesCost + doughnutsCost + applePieCost + cookiesCost + chocolateBarsCost + sodaCost + gumCost + chipsCost

theorem Sophie_Spends_72_80 : SophieTotalCost = 72.80 :=
by
  sorry

end Sophie_Spends_72_80_l59_59248


namespace ratio_of_dolls_l59_59615

-- Definitions used in Lean 4 statement directly appear in the conditions
variable (I : ℕ) -- the number of dolls Ivy has
variable (Dina_dolls : ℕ := 60) -- Dina has 60 dolls
variable (Ivy_collectors : ℕ := 20) -- Ivy has 20 collector edition dolls

-- Condition based on given problem
axiom Ivy_collectors_condition : (2 / 3 : ℚ) * I = 20

-- Lean 4 statement for the proof problem
theorem ratio_of_dolls (h : 3 * Ivy_collectors = 2 * I) : Dina_dolls / I = 2 := by
  sorry

end ratio_of_dolls_l59_59615


namespace missing_fraction_l59_59657

-- Defining all the given fractions
def f1 : ℚ := 1 / 3
def f2 : ℚ := 1 / 2
def f3 : ℚ := 1 / 5
def f4 : ℚ := 1 / 4
def f5 : ℚ := -9 / 20
def f6 : ℚ := -5 / 6

-- Defining the total sum in decimal form
def total_sum : ℚ := 5 / 6  -- Since 0.8333333333333334 is equivalent to 5/6

-- Defining the sum of the given fractions
def given_sum : ℚ := f1 + f2 + f3 + f4 + f5 + f6

-- The Lean 4 statement to prove the missing fraction
theorem missing_fraction : ∃ x : ℚ, (given_sum + x = total_sum) ∧ x = 5 / 6 :=
by
  use 5 / 6
  constructor
  . sorry
  . rfl

end missing_fraction_l59_59657


namespace parametric_to_line_segment_l59_59215

theorem parametric_to_line_segment :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 5 →
  ∃ x y : ℝ, x = 3 * t^2 + 2 ∧ y = t^2 - 1 ∧ (x - 3 * y = 5) ∧ (-1 ≤ y ∧ y ≤ 24) :=
by
  sorry

end parametric_to_line_segment_l59_59215


namespace monotonic_decreasing_interval_l59_59136

noncomputable def f (x : ℝ) : ℝ :=
  x / 4 + 5 / (4 * x) - Real.log x

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), (a = 0) ∧ (b = 5) ∧ (∀ x, 0 < x ∧ x < 5 → (deriv f x < 0)) :=
by
  sorry

end monotonic_decreasing_interval_l59_59136


namespace internal_angles_and_area_of_grey_triangle_l59_59732

/-- Given three identical grey triangles, 
    three identical squares, and an equilateral 
    center triangle with area 2 cm^2,
    the internal angles of the grey triangles 
    are 120 degrees and 30 degrees, and the 
    total grey area is 6 cm^2. -/
theorem internal_angles_and_area_of_grey_triangle 
  (triangle_area : ℝ)
  (α β : ℝ)
  (grey_area : ℝ) :
  triangle_area = 2 →  
  α = 120 ∧ β = 30 ∧ grey_area = 6 :=
by
  sorry

end internal_angles_and_area_of_grey_triangle_l59_59732


namespace find_angle_C_l59_59236

theorem find_angle_C (a b c : ℝ) (h : a ^ 2 + b ^ 2 - c ^ 2 + a * b = 0) : 
  C = 2 * pi / 3 := 
sorry

end find_angle_C_l59_59236


namespace fraction_given_to_cousin_l59_59015

theorem fraction_given_to_cousin
  (initial_candies : ℕ)
  (brother_share sister_share : ℕ)
  (eaten_candies left_candies : ℕ)
  (remaining_candies : ℕ)
  (given_to_cousin : ℕ)
  (fraction : ℚ)
  (h1 : initial_candies = 50)
  (h2 : brother_share = 5)
  (h3 : sister_share = 5)
  (h4 : eaten_candies = 12)
  (h5 : left_candies = 18)
  (h6 : initial_candies - brother_share - sister_share = remaining_candies)
  (h7 : remaining_candies - given_to_cousin - eaten_candies = left_candies)
  (h8 : fraction = (given_to_cousin : ℚ) / (remaining_candies : ℚ))
  : fraction = 1 / 4 := 
sorry

end fraction_given_to_cousin_l59_59015


namespace accident_rate_is_100_million_l59_59750

theorem accident_rate_is_100_million (X : ℕ) (h1 : 96 * 3000000000 = 2880 * X) : X = 100000000 :=
by
  sorry

end accident_rate_is_100_million_l59_59750


namespace discriminant_negative_of_positive_parabola_l59_59775

variable (a b c : ℝ)

theorem discriminant_negative_of_positive_parabola (h1 : ∀ x : ℝ, a * x^2 + b * x + c > 0) (h2 : a > 0) : b^2 - 4*a*c < 0 := 
sorry

end discriminant_negative_of_positive_parabola_l59_59775


namespace scale_reading_l59_59535

theorem scale_reading (a b c : ℝ) (h₁ : 10.15 < a ∧ a < 10.4) (h₂ : 10.275 = (10.15 + 10.4) / 2) : a = 10.3 := 
by 
  sorry

end scale_reading_l59_59535


namespace correct_operation_B_l59_59534

theorem correct_operation_B (a b : ℝ) : - (a - b) = -a + b := 
by sorry

end correct_operation_B_l59_59534


namespace JeremyTotalExpenses_l59_59870

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

end JeremyTotalExpenses_l59_59870


namespace Erica_Ice_Cream_Spend_l59_59041

theorem Erica_Ice_Cream_Spend :
  (6 * ((3 * 2.00) + (2 * 1.50) + (2 * 3.00))) = 90 := sorry

end Erica_Ice_Cream_Spend_l59_59041


namespace second_smallest_odd_number_l59_59680

-- Define the conditions
def four_consecutive_odd_numbers_sum (n : ℕ) : Prop := 
  n % 2 = 1 ∧ (n + (n + 2) + (n + 4) + (n + 6) = 112)

-- State the theorem
theorem second_smallest_odd_number (n : ℕ) (h : four_consecutive_odd_numbers_sum n) : n + 2 = 27 :=
sorry

end second_smallest_odd_number_l59_59680


namespace product_of_numerator_and_denominator_l59_59741

-- Defining the repeating decimal as a fraction in lowest terms
def repeating_decimal_as_fraction_in_lowest_terms : ℚ :=
  1 / 37

-- Theorem to prove the product of the numerator and the denominator
theorem product_of_numerator_and_denominator :
  (repeating_decimal_as_fraction_in_lowest_terms.num.natAbs *
   repeating_decimal_as_fraction_in_lowest_terms.den) = 37 :=
by
  -- declaration of the needed fact and its direct consequence
  sorry

end product_of_numerator_and_denominator_l59_59741


namespace jordan_rectangle_width_l59_59751

noncomputable def carol_length : ℝ := 4.5
noncomputable def carol_width : ℝ := 19.25
noncomputable def jordan_length : ℝ := 3.75

noncomputable def carol_area : ℝ := carol_length * carol_width
noncomputable def jordan_width : ℝ := carol_area / jordan_length

theorem jordan_rectangle_width : jordan_width = 23.1 := by
  -- proof will go here
  sorry

end jordan_rectangle_width_l59_59751


namespace subtracted_value_l59_59940

theorem subtracted_value (N V : ℕ) (h1 : N = 800) (h2 : N / 5 - V = 6) : V = 154 :=
by
  sorry

end subtracted_value_l59_59940


namespace median_mean_l59_59785

theorem median_mean (n : ℕ) (h : n + 4 = 8) : (4 + 6 + 8 + 14 + 16) / 5 = 9.6 := by
  sorry

end median_mean_l59_59785


namespace percentage_not_speak_french_l59_59112

open Nat

theorem percentage_not_speak_french (students_surveyed : ℕ)
  (speak_french_and_english : ℕ) (speak_only_french : ℕ) :
  students_surveyed = 200 →
  speak_french_and_english = 25 →
  speak_only_french = 65 →
  ((students_surveyed - (speak_french_and_english + speak_only_french)) * 100 / students_surveyed) = 55 :=
by
  intros h1 h2 h3
  sorry

end percentage_not_speak_french_l59_59112


namespace integral_of_2x2_cos3x_l59_59856

theorem integral_of_2x2_cos3x :
  ∫ x in (0 : ℝ)..(2 * Real.pi), (2 * x ^ 2 - 15) * Real.cos (3 * x) = (8 * Real.pi) / 9 :=
by
  sorry

end integral_of_2x2_cos3x_l59_59856


namespace total_ticket_sales_l59_59566

-- Define the parameters and the theorem to be proven.
theorem total_ticket_sales (total_people : ℕ) (kids : ℕ) (adult_ticket_price : ℕ) (kid_ticket_price : ℕ) 
  (adult_tickets := total_people - kids) 
  (adult_ticket_sales := adult_tickets * adult_ticket_price) 
  (kid_ticket_sales := kids * kid_ticket_price) : 
  total_people = 254 → kids = 203 → adult_ticket_price = 28 → kid_ticket_price = 12 → 
  adult_ticket_sales + kid_ticket_sales = 3864 := 
by
  intros h1 h2 h3 h4
  sorry

end total_ticket_sales_l59_59566


namespace tan_identity_example_l59_59844

theorem tan_identity_example (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 :=
by
  sorry

end tan_identity_example_l59_59844


namespace cos_angle_between_vectors_l59_59737

theorem cos_angle_between_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (1, 3)
  let dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  let magnitude (x : ℝ × ℝ) : ℝ := Real.sqrt (x.1 ^ 2 + x.2 ^ 2)
  let cos_theta := dot_product a b / (magnitude a * magnitude b)
  cos_theta = -Real.sqrt 2 / 10 :=
by
  sorry

end cos_angle_between_vectors_l59_59737


namespace pizza_topping_cost_l59_59513

/- 
   Given:
   1. Ruby ordered 3 pizzas.
   2. Each pizza costs $10.00.
   3. The total number of toppings were 4.
   4. Ruby added a $5.00 tip to the order.
   5. The total cost of the order, including tip, was $39.00.

   Prove: The cost per topping is $1.00.
-/
theorem pizza_topping_cost (cost_per_pizza : ℝ) (total_pizzas : ℕ) (tip : ℝ) (total_cost : ℝ) 
    (total_toppings : ℕ) (x : ℝ) : 
    cost_per_pizza = 10 → total_pizzas = 3 → tip = 5 → total_cost = 39 → total_toppings = 4 → 
    total_cost = cost_per_pizza * total_pizzas + x * total_toppings + tip →
    x = 1 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end pizza_topping_cost_l59_59513


namespace seymour_fertilizer_requirement_l59_59340

theorem seymour_fertilizer_requirement :
  let flats_petunias := 4
  let petunias_per_flat := 8
  let flats_roses := 3
  let roses_per_flat := 6
  let venus_flytraps := 2
  let fert_per_petunia := 8
  let fert_per_rose := 3
  let fert_per_venus_flytrap := 2

  let total_petunias := flats_petunias * petunias_per_flat
  let total_roses := flats_roses * roses_per_flat
  let fert_petunias := total_petunias * fert_per_petunia
  let fert_roses := total_roses * fert_per_rose
  let fert_venus_flytraps := venus_flytraps * fert_per_venus_flytrap

  let total_fertilizer := fert_petunias + fert_roses + fert_venus_flytraps
  total_fertilizer = 314 := sorry

end seymour_fertilizer_requirement_l59_59340


namespace intersection_eq_l59_59369

-- defining the set A
def A := {x : ℝ | x^2 + 2*x - 3 ≤ 0}

-- defining the set B
def B := {y : ℝ | ∃ x ∈ A, y = x^2 + 4*x + 3}

-- The proof problem statement: prove that A ∩ B = [-1, 1]
theorem intersection_eq : A ∩ B = {y : ℝ | -1 ≤ y ∧ y ≤ 1} :=
by sorry

end intersection_eq_l59_59369


namespace only_book_A_l59_59933

theorem only_book_A (purchasedBoth : ℕ) (purchasedOnlyB : ℕ) (purchasedA : ℕ) (purchasedB : ℕ) 
  (h1 : purchasedBoth = 500)
  (h2 : 2 * purchasedOnlyB = purchasedBoth)
  (h3 : purchasedA = 2 * purchasedB)
  (h4 : purchasedB = purchasedOnlyB + purchasedBoth) :
  purchasedA - purchasedBoth = 1000 :=
by
  sorry

end only_book_A_l59_59933


namespace find_seating_capacity_l59_59320

noncomputable def seating_capacity (buses : ℕ) (students_left : ℤ) : ℤ :=
  buses * 40 + students_left

theorem find_seating_capacity :
  (seating_capacity 4 30) = (seating_capacity 5 (-10)) :=
by
  -- Proof is not required, hence omitted.
  sorry

end find_seating_capacity_l59_59320


namespace maximum_area_of_right_triangle_l59_59047

theorem maximum_area_of_right_triangle
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b)
  (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = 2) : 
  ∃ S, S ≤ (3 - 2 * Real.sqrt 2) ∧ S = (1/2) * a * b :=
by
  sorry

end maximum_area_of_right_triangle_l59_59047


namespace sqrt_sum_eq_nine_l59_59790

theorem sqrt_sum_eq_nine (x : ℝ) (h : Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) :
  (7 + x) * (28 - x) = 529 :=
sorry

end sqrt_sum_eq_nine_l59_59790


namespace range_of_a_l59_59913

open Set

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) :
  a ∈ Iic (-1 / 2) ∪ Ici 2 :=
by
  sorry

end range_of_a_l59_59913


namespace possible_values_for_a_l59_59459

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) - x - 1

theorem possible_values_for_a (a : ℝ) (h: a ≠ 0) : 
  (∀ x : ℝ, f a x ≥ 0) ↔ a = 1 :=
by
  sorry

end possible_values_for_a_l59_59459


namespace total_students_l59_59760

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

end total_students_l59_59760


namespace book_distribution_l59_59581

theorem book_distribution (x : ℕ) (h1 : 9 * x + 7 < 11 * x) : 
  9 * x + 7 = totalBooks - 9 * x ∧ totalBooks - 9 * x = 7 :=
by
  sorry

end book_distribution_l59_59581


namespace grain_demand_l59_59551

variable (F : ℝ)
def S0 : ℝ := 1800000 -- base supply value

theorem grain_demand : ∃ D : ℝ, S = 0.75 * D ∧ S = S0 * (1 + F) ∧ D = (1800000 * (1 + F) / 0.75) :=
by
  sorry

end grain_demand_l59_59551


namespace tan_half_angle_lt_l59_59854

theorem tan_half_angle_lt (x : ℝ) (h : 0 < x ∧ x ≤ π / 2) : 
  Real.tan (x / 2) < x := 
by
  sorry

end tan_half_angle_lt_l59_59854


namespace parabola_properties_l59_59035

theorem parabola_properties (p m k1 k2 k3 : ℝ)
  (parabola_eq : ∀ x y, y^2 = 2 * p * x ↔ y = m)
  (parabola_passes_through : m^2 = 2 * p)
  (point_distance : ((1 + p / 2)^2 + m^2 = 8) ∨ ((1 + p / 2)^2 + m^2 = 8))
  (p_gt_zero : p > 0)
  (point_P : (1, 2) ∈ { (x, y) | y^2 = 4 * x })
  (slope_eq : k3 = (k1 * k2) / (k1 + k2 - k1 * k2)) :
  (y^2 = 4 * x) ∧ (1/k1 + 1/k2 - 1/k3 = 1) := sorry

end parabola_properties_l59_59035


namespace gcd_1151_3079_l59_59643

def a : ℕ := 1151
def b : ℕ := 3079

theorem gcd_1151_3079 : gcd a b = 1 := by
  sorry

end gcd_1151_3079_l59_59643


namespace coral_remaining_pages_l59_59145

def pages_after_week1 (total_pages : ℕ) : ℕ :=
  total_pages / 2

def pages_after_week2 (remaining_pages_week1 : ℕ) : ℕ :=
  remaining_pages_week1 - (3 * remaining_pages_week1 / 10)

def pages_after_week3 (remaining_pages_week2 : ℕ) (reading_hours : ℕ) (reading_speed : ℕ) : ℕ :=
  remaining_pages_week2 - (reading_hours * reading_speed)

theorem coral_remaining_pages (total_pages remaining_pages_week1 remaining_pages_week2 remaining_pages_week3 : ℕ) 
  (reading_hours reading_speed unread_pages : ℕ)
  (h1 : total_pages = 600)
  (h2 : remaining_pages_week1 = pages_after_week1 total_pages)
  (h3 : remaining_pages_week2 = pages_after_week2 remaining_pages_week1)
  (h4 : reading_hours = 10)
  (h5 : reading_speed = 15)
  (h6 : remaining_pages_week3 = pages_after_week3 remaining_pages_week2 reading_hours reading_speed)
  (h7 : unread_pages = remaining_pages_week3) :
  unread_pages = 60 :=
by
  sorry

end coral_remaining_pages_l59_59145


namespace pushing_car_effort_l59_59656

theorem pushing_car_effort (effort constant : ℕ) (people1 people2 : ℕ) 
  (h1 : constant = people1 * effort)
  (h2 : people1 = 4)
  (h3 : effort = 120)
  (h4 : people2 = 6) :
  effort * people1 = constant → constant = people2 * 80 :=
by
  sorry

end pushing_car_effort_l59_59656


namespace intersection_P_complement_Q_l59_59931

-- Defining the sets P and Q
def R := Set ℝ
def P : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def Q : Set ℝ := {x | Real.log x < 1}
def complement_R_Q : Set ℝ := {x | x ≤ 0 ∨ x ≥ Real.exp 1}
def intersection := {x | x ∈ P ∧ x ∈ complement_R_Q}

-- Statement of the theorem
theorem intersection_P_complement_Q : 
  intersection = {-3} :=
by
  sorry

end intersection_P_complement_Q_l59_59931


namespace find_remainder_l59_59224

theorem find_remainder :
  ∀ (D d q r : ℕ), 
    D = 18972 → 
    d = 526 → 
    q = 36 → 
    D = d * q + r → 
    r = 36 :=
by 
  intros D d q r hD hd hq hEq
  sorry

end find_remainder_l59_59224


namespace infinite_natural_numbers_with_factored_polynomial_l59_59406

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l59_59406


namespace mowing_lawn_each_week_l59_59767

-- Definitions based on the conditions
def riding_speed : ℝ := 2 -- acres per hour with riding mower
def push_speed : ℝ := 1 -- acre per hour with push mower
def total_hours : ℝ := 5 -- total hours

-- The problem we want to prove
theorem mowing_lawn_each_week (A : ℝ) :
  (3 / 4) * A / riding_speed + (1 / 4) * A / push_speed = total_hours → 
  A = 15 :=
by
  sorry

end mowing_lawn_each_week_l59_59767


namespace units_digit_7_pow_6_l59_59052

theorem units_digit_7_pow_6 : (7 ^ 6) % 10 = 9 := by
  sorry

end units_digit_7_pow_6_l59_59052


namespace lucy_total_packs_l59_59225

-- Define the number of packs of cookies Lucy bought
def packs_of_cookies : ℕ := 12

-- Define the number of packs of noodles Lucy bought
def packs_of_noodles : ℕ := 16

-- Define the total number of packs of groceries Lucy bought
def total_packs_of_groceries : ℕ := packs_of_cookies + packs_of_noodles

-- Proof statement: The total number of packs of groceries Lucy bought is 28
theorem lucy_total_packs : total_packs_of_groceries = 28 := by
  sorry

end lucy_total_packs_l59_59225


namespace beth_gave_away_54_crayons_l59_59483

-- Define the initial number of crayons
def initialCrayons : ℕ := 106

-- Define the number of crayons left
def remainingCrayons : ℕ := 52

-- Define the number of crayons given away
def crayonsGiven (initial remaining: ℕ) : ℕ := initial - remaining

-- The goal is to prove that Beth gave away 54 crayons
theorem beth_gave_away_54_crayons : crayonsGiven initialCrayons remainingCrayons = 54 :=
by
  sorry

end beth_gave_away_54_crayons_l59_59483


namespace work_done_in_11_days_l59_59445

-- Given conditions as definitions
def a_days := 24
def b_days := 30
def c_days := 40
def combined_work_rate := (1 / a_days) + (1 / b_days) + (1 / c_days)
def days_c_leaves_before_completion := 4

-- Statement of the problem to be proved
theorem work_done_in_11_days :
  ∃ (D : ℕ), D = 11 ∧ ((D - days_c_leaves_before_completion) * combined_work_rate) + 
  (days_c_leaves_before_completion * ((1 / a_days) + (1 / b_days))) = 1 :=
sorry

end work_done_in_11_days_l59_59445


namespace isosceles_triangle_base_length_l59_59747

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l59_59747


namespace sum_of_two_numbers_l59_59009

theorem sum_of_two_numbers (a b : ℝ) (h1 : a * b = 16) (h2 : (1 / a) = 3 * (1 / b)) (ha : 0 < a) (hb : 0 < b) :
  a + b = 16 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end sum_of_two_numbers_l59_59009


namespace price_difference_l59_59852

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

end price_difference_l59_59852


namespace serum_prevents_colds_l59_59420

noncomputable def hypothesis_preventive_effect (H : Prop) : Prop :=
  let K2 := 3.918
  let critical_value := 3.841
  let P_threshold := 0.05
  K2 >= critical_value ∧ P_threshold = 0.05 → H

theorem serum_prevents_colds (H : Prop) : hypothesis_preventive_effect H → H :=
by
  -- Proof will be added here
  sorry

end serum_prevents_colds_l59_59420


namespace cost_per_meal_is_8_l59_59546

-- Define the conditions
def number_of_adults := 2
def number_of_children := 5
def total_bill := 56
def total_people := number_of_adults + number_of_children

-- Define the cost per meal
def cost_per_meal := total_bill / total_people

-- State the theorem we want to prove
theorem cost_per_meal_is_8 : cost_per_meal = 8 := 
by
  -- The proof would go here, but we'll use sorry to skip it
  sorry

end cost_per_meal_is_8_l59_59546


namespace dart_lands_in_center_hexagon_l59_59881

noncomputable def area_regular_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

theorem dart_lands_in_center_hexagon {s : ℝ} (h : s > 0) :
  let A_outer := area_regular_hexagon s
  let A_inner := area_regular_hexagon (s / 2)
  (A_inner / A_outer) = 1 / 4 :=
by
  let A_outer := area_regular_hexagon s
  let A_inner := area_regular_hexagon (s / 2)
  sorry

end dart_lands_in_center_hexagon_l59_59881


namespace g_of_g_of_2_l59_59830

def g (x : ℝ) : ℝ := 4 * x^2 - 3

theorem g_of_g_of_2 : g (g 2) = 673 := 
by 
  sorry

end g_of_g_of_2_l59_59830


namespace game_is_not_fair_l59_59519

noncomputable def expected_winnings : ℚ := 
  let p_1 := 1 / 8
  let p_2 := 7 / 8
  let gain_case_1 := 2
  let loss_case_2 := -1 / 7
  (p_1 * gain_case_1) + (p_2 * loss_case_2)

theorem game_is_not_fair : expected_winnings = 1 / 8 :=
sorry

end game_is_not_fair_l59_59519


namespace factor_expression_l59_59467

theorem factor_expression (
  x y z : ℝ
) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3) = 
  (x + y) * (y + z) * (z + x) := 
sorry

end factor_expression_l59_59467


namespace competition_score_l59_59674

theorem competition_score
    (x : ℕ)
    (h1 : 20 ≥ x)
    (h2 : 5 * x - (20 - x) = 70) :
    x = 15 :=
sorry

end competition_score_l59_59674


namespace total_profit_equals_254000_l59_59297

-- Definitions
def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 6000
def investment_D : ℕ := 10000

def time_A : ℕ := 12
def time_B : ℕ := 8
def time_C : ℕ := 6
def time_D : ℕ := 9

def capital_months (investment : ℕ) (time : ℕ) : ℕ := investment * time

-- Given conditions
def A_capital_months := capital_months investment_A time_A
def B_capital_months := capital_months investment_B time_B
def C_capital_months := capital_months investment_C time_C
def D_capital_months := capital_months investment_D time_D

def total_capital_months : ℕ := A_capital_months + B_capital_months + C_capital_months + D_capital_months

def C_profit : ℕ := 36000

-- Proportion equation
def total_profit (C_capital_months : ℕ) (total_capital_months : ℕ) (C_profit : ℕ) : ℕ :=
  (C_profit * total_capital_months) / C_capital_months

-- Theorem statement
theorem total_profit_equals_254000 : total_profit C_capital_months total_capital_months C_profit = 254000 := by
  sorry

end total_profit_equals_254000_l59_59297


namespace area_of_unpainted_section_l59_59601

-- Define the conditions
def board1_width : ℝ := 5
def board2_width : ℝ := 7
def cross_angle : ℝ := 45
def negligible_holes : Prop := true

-- The main statement
theorem area_of_unpainted_section (h1 : board1_width = 5) (h2 : board2_width = 7) (h3 : cross_angle = 45) (h4 : negligible_holes) : 
  ∃ (area : ℝ), area = 35 := 
sorry

end area_of_unpainted_section_l59_59601


namespace employee_pay_per_week_l59_59917

theorem employee_pay_per_week (total_pay : ℝ) (ratio : ℝ) (pay_b : ℝ)
  (h1 : total_pay = 570)
  (h2 : ratio = 1.5)
  (h3 : total_pay = pay_b * (ratio + 1)) :
  pay_b = 228 :=
sorry

end employee_pay_per_week_l59_59917


namespace fourth_root_12960000_eq_60_l59_59974

theorem fourth_root_12960000_eq_60 :
  (6^4 = 1296) →
  (10^4 = 10000) →
  (60^4 = 12960000) →
  (Real.sqrt (Real.sqrt 12960000) = 60) := 
by
  intros h1 h2 h3
  sorry

end fourth_root_12960000_eq_60_l59_59974


namespace children_marbles_problem_l59_59895

theorem children_marbles_problem (n x N : ℕ) 
  (h1 : N = n * x)
  (h2 : 1 + (N - 1) / 10 = x) :
  n = 9 ∧ x = 9 :=
by
  sorry

end children_marbles_problem_l59_59895


namespace find_pages_revised_twice_l59_59793

def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_time : ℕ) (cost_revised_once : ℕ) (cost_revised_twice : ℕ) (total_cost : ℕ) :=
  ∃ (x : ℕ), 
    (total_pages - pages_revised_once - x) * cost_first_time
    + pages_revised_once * (cost_first_time + cost_revised_once)
    + x * (cost_first_time + cost_revised_once + cost_revised_once) = total_cost 

theorem find_pages_revised_twice :
  pages_revised_twice 100 35 6 4 4 860 ↔ ∃ x, x = 15 :=
by
  sorry

end find_pages_revised_twice_l59_59793


namespace max_value_of_quadratic_l59_59624

theorem max_value_of_quadratic :
  ∃ x_max : ℝ, x_max = 1.5 ∧
  ∀ x : ℝ, -3 * x^2 + 9 * x + 24 ≤ -3 * (1.5)^2 + 9 * 1.5 + 24 := by
  sorry

end max_value_of_quadratic_l59_59624


namespace range_of_a_l59_59043

noncomputable def f (a x : ℝ) :=
  if x < 0 then
    9 * x + a^2 / x + 7
  else
    9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) → a ≤ -8 / 7 :=
  sorry

end range_of_a_l59_59043


namespace num_remainders_prime_squares_mod_210_l59_59230

theorem num_remainders_prime_squares_mod_210 :
  (∃ (p : ℕ) (hp : p > 7) (hprime : Prime p), 
    ∀ r : Finset ℕ, 
      (∀ q ∈ r, (∃ (k : ℕ), p = 210 * k + q)) 
      → r.card = 8) :=
sorry

end num_remainders_prime_squares_mod_210_l59_59230


namespace gcd_power_diff_l59_59815

theorem gcd_power_diff (n m : ℕ) (h₁ : n = 2025) (h₂ : m = 2007) :
  (Nat.gcd (2^n - 1) (2^m - 1)) = 2^18 - 1 :=
by
  sorry

end gcd_power_diff_l59_59815


namespace small_mold_radius_l59_59504

theorem small_mold_radius (r : ℝ) (n : ℝ) (s : ℝ) :
    r = 2 ∧ n = 8 ∧ (1 / 2) * (2 / 3) * Real.pi * r^3 = (8 * (2 / 3) * Real.pi * s^3) → s = 1 :=
by
  sorry

end small_mold_radius_l59_59504


namespace oldest_child_age_l59_59249

theorem oldest_child_age 
  (avg_age : ℕ) (child1 : ℕ) (child2 : ℕ) (child3 : ℕ) (child4 : ℕ)
  (h_avg : avg_age = 8) 
  (h_child1 : child1 = 5) 
  (h_child2 : child2 = 7) 
  (h_child3 : child3 = 10)
  (h_avg_eq : (child1 + child2 + child3 + child4) / 4 = avg_age) :
  child4 = 10 := 
by 
  sorry

end oldest_child_age_l59_59249


namespace gcd_of_g_y_and_y_l59_59390

theorem gcd_of_g_y_and_y (y : ℤ) (h : 9240 ∣ y) : Int.gcd ((5 * y + 3) * (11 * y + 2) * (17 * y + 8) * (4 * y + 7)) y = 168 := by
  sorry

end gcd_of_g_y_and_y_l59_59390


namespace largest_divisible_n_l59_59418

/-- Largest positive integer n for which n^3 + 10 is divisible by n + 1 --/
theorem largest_divisible_n (n : ℕ) :
  n = 0 ↔ ∀ m : ℕ, (m > n) → ¬ ((m^3 + 10) % (m + 1) = 0) :=
by
  sorry

end largest_divisible_n_l59_59418


namespace problem_statement_l59_59505

theorem problem_statement (x θ : ℝ) (h : Real.logb 2 x + Real.cos θ = 2) : |x - 8| + |x + 2| = 10 :=
sorry

end problem_statement_l59_59505


namespace product_of_two_numbers_l59_59438

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 48) (h_gcd : gcd a b = 8) : a * b = 384 :=
by sorry

end product_of_two_numbers_l59_59438


namespace find_x0_l59_59309

-- Defining the function f
def f (a c x : ℝ) : ℝ := a * x^2 + c

-- Defining the integral condition
def integral_condition (a c x0 : ℝ) : Prop :=
  (∫ x in (0 : ℝ)..(1 : ℝ), f a c x) = f a c x0

-- Proving the main statement
theorem find_x0 (a c x0 : ℝ) (h : a ≠ 0) (h_range : 0 ≤ x0 ∧ x0 ≤ 1) (h_integral : integral_condition a c x0) :
  x0 = Real.sqrt (1 / 3) :=
by
  sorry

end find_x0_l59_59309


namespace sam_initial_pennies_l59_59109

def initial_pennies_spent (spent: Nat) (left: Nat) : Nat :=
  spent + left

theorem sam_initial_pennies (spent: Nat) (left: Nat) : spent = 93 ∧ left = 5 → initial_pennies_spent spent left = 98 :=
by
  sorry

end sam_initial_pennies_l59_59109


namespace heptagon_diagonals_l59_59111

theorem heptagon_diagonals (n : ℕ) (h : n = 7) : (n * (n - 3)) / 2 = 14 := by
  sorry

end heptagon_diagonals_l59_59111


namespace find_number_of_even_numbers_l59_59074

-- Define the average of the first n even numbers
def average_of_first_n_even (n : ℕ) : ℕ :=
  (n * (1 + n)) / n

-- The given condition: The average is 21
def average_is_21 (n : ℕ) : Prop :=
  average_of_first_n_even n = 21

-- The theorem to prove: If the average is 21, then n = 20
theorem find_number_of_even_numbers (n : ℕ) (h : average_is_21 n) : n = 20 :=
  sorry

end find_number_of_even_numbers_l59_59074


namespace two_leq_one_add_one_div_n_pow_n_lt_three_l59_59644

theorem two_leq_one_add_one_div_n_pow_n_lt_three :
  ∀ (n : ℕ), 2 ≤ (1 + (1 : ℝ) / n) ^ n ∧ (1 + (1 : ℝ) / n) ^ n < 3 := 
by 
  sorry

end two_leq_one_add_one_div_n_pow_n_lt_three_l59_59644


namespace find_C_share_l59_59955

-- Definitions
variable (A B C : ℝ)
variable (H1 : A + B + C = 585)
variable (H2 : 4 * A = 6 * B)
variable (H3 : 6 * B = 3 * C)

-- Problem statement
theorem find_C_share (A B C : ℝ) (H1 : A + B + C = 585) (H2 : 4 * A = 6 * B) (H3 : 6 * B = 3 * C) : C = 260 :=
by
  sorry

end find_C_share_l59_59955


namespace probability_white_ball_l59_59462

def num_white_balls : ℕ := 5
def num_black_balls : ℕ := 6
def total_balls : ℕ := num_white_balls + num_black_balls

theorem probability_white_ball : (num_white_balls : ℚ) / total_balls = 5 / 11 := by
  sorry

end probability_white_ball_l59_59462


namespace probability_of_two_points_is_three_sevenths_l59_59928

/-- Define the problem's conditions and statement. -/
def num_choices (n : ℕ) : ℕ :=
  match n with
  | 1 => 4  -- choose 1 option from 4
  | 2 => 6  -- choose 2 options from 4 (binomial coefficient)
  | 3 => 4  -- choose 3 options from 4 (binomial coefficient)
  | _ => 0

def total_ways : ℕ := 14  -- Total combinations of choosing 1 to 3 options from 4

def two_points_ways : ℕ := 6  -- 3 ways for 1 correct, 3 ways for 2 correct (B, C, D combinations)

def probability_two_points : ℚ :=
  (two_points_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_two_points_is_three_sevenths :
  probability_two_points = (3 / 7 : ℚ) :=
sorry

end probability_of_two_points_is_three_sevenths_l59_59928


namespace rightmost_three_digits_seven_pow_1983_add_123_l59_59548

theorem rightmost_three_digits_seven_pow_1983_add_123 :
  (7 ^ 1983 + 123) % 1000 = 466 := 
by 
  -- Proof steps are omitted
  sorry 

end rightmost_three_digits_seven_pow_1983_add_123_l59_59548


namespace no_such_set_exists_l59_59640

open Nat Set

theorem no_such_set_exists (M : Set ℕ) : 
  (∀ m : ℕ, m > 1 → ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ a + b = m) →
  (∀ a b c d : ℕ, a ∈ M → b ∈ M → c ∈ M → d ∈ M → 
    a > 10 → b > 10 → c > 10 → d > 10 → a + b = c + d → a = c ∨ a = d) → 
  False := by
  sorry

end no_such_set_exists_l59_59640


namespace fourth_throw_probability_l59_59115

-- Define a fair dice where each face has an equal probability.
def fair_dice (n : ℕ) : Prop := (n >= 1 ∧ n <= 6)

-- Define the probability of rolling a 6 on a fair dice.
noncomputable def probability_of_6 : ℝ := 1 / 6

/-- 
  Prove that the probability of getting a "6" on the 4th throw is 1/6 
  given that the dice is fair and the first three throws result in "6".
-/
theorem fourth_throw_probability : 
  (∀ (n1 n2 n3 : ℕ), fair_dice n1 ∧ fair_dice n2 ∧ fair_dice n3 ∧ n1 = 6 ∧ n2 = 6 ∧ n3 = 6) 
  → (probability_of_6 = 1 / 6) :=
by 
  sorry

end fourth_throw_probability_l59_59115


namespace min_frac_sum_pos_real_l59_59151

variable {x y z w : ℝ}

theorem min_frac_sum_pos_real (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) (h_sum : x + y + z + w = 1) : 
  (x + y + z) / (x * y * z * w) ≥ 144 := 
sorry

end min_frac_sum_pos_real_l59_59151


namespace manager_salary_is_correct_l59_59926

noncomputable def manager_salary (avg_salary_50_employees : ℝ) (increase_in_avg : ℝ) : ℝ :=
  let total_salary_50_employees := 50 * avg_salary_50_employees
  let new_avg_salary := avg_salary_50_employees + increase_in_avg
  let total_salary_51_people := 51 * new_avg_salary
  let manager_salary := total_salary_51_people - total_salary_50_employees
  manager_salary

theorem manager_salary_is_correct :
  manager_salary 2500 1500 = 79000 :=
by
  sorry

end manager_salary_is_correct_l59_59926


namespace expansion_of_product_l59_59259

theorem expansion_of_product (x : ℝ) :
  (7 * x + 3) * (5 * x^2 + 2 * x + 4) = 35 * x^3 + 29 * x^2 + 34 * x + 12 := 
by
  sorry

end expansion_of_product_l59_59259


namespace establish_model_steps_correct_l59_59676

-- Define each step as a unique identifier
inductive Step : Type
| observe_pose_questions
| propose_assumptions
| express_properties
| test_or_revise

open Step

-- The sequence of steps to establish a mathematical model for population change
def correct_model_steps : List Step :=
  [observe_pose_questions, propose_assumptions, express_properties, test_or_revise]

-- The correct answer is the sequence of steps in the correct order
theorem establish_model_steps_correct :
  correct_model_steps = [observe_pose_questions, propose_assumptions, express_properties, test_or_revise] :=
  by sorry

end establish_model_steps_correct_l59_59676


namespace initial_balance_before_check_deposit_l59_59795

theorem initial_balance_before_check_deposit (new_balance : ℝ) (initial_balance : ℝ) : 
  (50 = 1 / 4 * new_balance) → (initial_balance = new_balance - 50) → initial_balance = 150 :=
by
  sorry

end initial_balance_before_check_deposit_l59_59795


namespace xy_yz_zx_nonzero_l59_59219

theorem xy_yz_zx_nonzero (x y z : ℝ)
  (h1 : 1 / |x^2 + 2 * y * z| + 1 / |y^2 + 2 * z * x| > 1 / |z^2 + 2 * x * y|)
  (h2 : 1 / |y^2 + 2 * z * x| + 1 / |z^2 + 2 * x * y| > 1 / |x^2 + 2 * y * z|)
  (h3 : 1 / |z^2 + 2 * x * y| + 1 / |x^2 + 2 * y * z| > 1 / |y^2 + 2 * z * x|) :
  x * y + y * z + z * x ≠ 0 := by
  sorry

end xy_yz_zx_nonzero_l59_59219


namespace squares_perimeter_and_rectangle_area_l59_59498

theorem squares_perimeter_and_rectangle_area (x y : ℝ) (hx : x^2 + y^2 = 145) (hy : x^2 - y^2 = 105) : 
  (4 * x + 4 * y = 28 * Real.sqrt 5) ∧ ((x + y) * x = 175) := 
by 
  sorry

end squares_perimeter_and_rectangle_area_l59_59498


namespace remainder_2457634_div_8_l59_59107

theorem remainder_2457634_div_8 : 2457634 % 8 = 2 := by
  sorry

end remainder_2457634_div_8_l59_59107


namespace best_choice_for_square_formula_l59_59396

theorem best_choice_for_square_formula : 
  (89.8^2 = (90 - 0.2)^2) :=
by sorry

end best_choice_for_square_formula_l59_59396


namespace inequality_solution_set_l59_59441

theorem inequality_solution_set (a : ℤ) : 
  (∀ x : ℤ, (1 + a) * x > 1 + a → x < 1) → a < -1 :=
sorry

end inequality_solution_set_l59_59441


namespace temple_shop_total_cost_l59_59228

theorem temple_shop_total_cost :
  let price_per_object := 11
  let num_people := 5
  let items_per_person := 4
  let extra_items := 4
  let total_objects := num_people * items_per_person + extra_items
  let total_cost := total_objects * price_per_object
  total_cost = 374 :=
by
  let price_per_object := 11
  let num_people := 5
  let items_per_person := 4
  let extra_items := 4
  let total_objects := num_people * items_per_person + extra_items
  let total_cost := total_objects * price_per_object
  show total_cost = 374
  sorry

end temple_shop_total_cost_l59_59228


namespace sum_series_eq_4_div_9_l59_59892

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l59_59892


namespace range_of_a_l59_59860

variables (a x : ℝ) -- Define real number variables a and x

-- Define proposition p
def p : Prop := (a - 2) * x * x + 2 * (a - 2) * x - 4 < 0 -- Inequality condition for any real x

-- Define proposition q
def q : Prop := 0 < a ∧ a < 1 -- Condition for logarithmic function to be strictly decreasing

-- Lean 4 statement for the proof problem
theorem range_of_a (Hpq : (p a x ∨ q a) ∧ ¬ (p a x ∧ q a)) :
  (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0) :=
sorry

end range_of_a_l59_59860


namespace exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987_l59_59404

theorem exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987 :
  ∃ n : ℕ, n ^ n + (n + 1) ^ n ≡ 0 [MOD 1987] := sorry

end exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987_l59_59404


namespace find_y_l59_59317

def operation (x y : ℝ) : ℝ := 5 * x - 4 * y + 3 * x * y

theorem find_y : ∃ y : ℝ, operation 4 y = 21 ∧ y = 1 / 8 := by
  sorry

end find_y_l59_59317


namespace distance_between_foci_l59_59213

-- Defining the given ellipse equation 
def ellipse_eq (x y : ℝ) : Prop := 25 * x^2 - 150 * x + 4 * y^2 + 8 * y + 9 = 0

-- Statement to prove the distance between the foci
theorem distance_between_foci (x y : ℝ) (h : ellipse_eq x y) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 46.2 := 
sorry

end distance_between_foci_l59_59213


namespace total_salmon_now_l59_59487

def initial_salmon : ℕ := 500

def increase_factor : ℕ := 10

theorem total_salmon_now : initial_salmon * increase_factor = 5000 := by
  sorry

end total_salmon_now_l59_59487


namespace constant_term_in_binomial_expansion_is_40_l59_59294

-- Define the binomial coefficient C(n, k)
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression for the binomial expansion of (x^2 + 2/x^3)^5
def term (r : ℕ) : ℕ := binom 5 r * 2^r

theorem constant_term_in_binomial_expansion_is_40 
  (x : ℝ) (h : x ≠ 0) : 
  (∃ r : ℕ, 10 - 5 * r = 0) ∧ term 2 = 40 :=
by 
  sorry

end constant_term_in_binomial_expansion_is_40_l59_59294


namespace max_value_g_l59_59257

def g : ℕ → ℤ
| n => if n < 5 then n + 10 else g (n - 3)

theorem max_value_g : ∃ x, (∀ n : ℕ, g n ≤ x) ∧ (∃ y, g y = x) ∧ x = 14 := 
by
  sorry

end max_value_g_l59_59257


namespace man_gets_dividend_l59_59064

    -- Definitions based on conditions
    noncomputable def investment : ℝ := 14400
    noncomputable def premium_rate : ℝ := 0.20
    noncomputable def face_value : ℝ := 100
    noncomputable def dividend_rate : ℝ := 0.07

    -- Calculate the price per share with premium
    noncomputable def price_per_share : ℝ := face_value * (1 + premium_rate)

    -- Calculate the number of shares bought
    noncomputable def number_of_shares : ℝ := investment / price_per_share

    -- Calculate the dividend per share
    noncomputable def dividend_per_share : ℝ := face_value * dividend_rate

    -- Calculate the total dividend
    noncomputable def total_dividend : ℝ := dividend_per_share * number_of_shares

    -- The proof statement
    theorem man_gets_dividend : total_dividend = 840 := by
        sorry
    
end man_gets_dividend_l59_59064


namespace determine_x_l59_59515

theorem determine_x (x : ℝ) (A B : Set ℝ) (H1 : A = {-1, 0}) (H2 : B = {0, 1, x + 2}) (H3 : A ⊆ B) : x = -3 :=
sorry

end determine_x_l59_59515


namespace solution_correct_l59_59605

-- Define the conditions
def abs_inequality (x : ℝ) : Prop := abs (x - 3) + abs (x + 4) < 8
def quadratic_eq (x : ℝ) : Prop := x^2 - x - 12 = 0

-- Define the main statement to prove
theorem solution_correct : ∃ (x : ℝ), abs_inequality x ∧ quadratic_eq x ∧ x = -3 := sorry

end solution_correct_l59_59605


namespace factor_1024_into_three_factors_l59_59384

theorem factor_1024_into_three_factors :
  ∃ (factors : Finset (Finset ℕ)), factors.card = 14 ∧
  ∀ f ∈ factors, ∃ a b c : ℕ, a + b + c = 10 ∧ a ≥ b ∧ b ≥ c ∧ (2 ^ a) * (2 ^ b) * (2 ^ c) = 1024 :=
sorry

end factor_1024_into_three_factors_l59_59384


namespace opposite_z_is_E_l59_59147

noncomputable def cube_faces := ["A", "B", "C", "D", "E", "z"]

def opposite_face (net : List String) (face : String) : String :=
  if face = "z" then "E" else sorry  -- generalize this function as needed

theorem opposite_z_is_E :
  opposite_face cube_faces "z" = "E" :=
by
  sorry

end opposite_z_is_E_l59_59147


namespace find_A_minus_B_l59_59758

def A : ℕ := (55 * 100) + (19 * 10)
def B : ℕ := 173 + (5 * 224)

theorem find_A_minus_B : A - B = 4397 := by
  sorry

end find_A_minus_B_l59_59758


namespace cube_surface_area_l59_59394

theorem cube_surface_area (s : ℝ) (h : s = 8) : 6 * s^2 = 384 :=
by
  sorry

end cube_surface_area_l59_59394


namespace water_consumption_150_litres_per_household_4_months_6000_litres_l59_59242

def number_of_households (household_water_use_per_month : ℕ) (water_supply : ℕ) (duration_months : ℕ) : ℕ :=
  water_supply / (household_water_use_per_month * duration_months)

theorem water_consumption_150_litres_per_household_4_months_6000_litres : 
  number_of_households 150 6000 4 = 10 :=
by
  sorry

end water_consumption_150_litres_per_household_4_months_6000_litres_l59_59242


namespace find_x_l59_59965

/-- Let r be the result of doubling both the base and exponent of a^b, 
and b does not equal to 0. If r equals the product of a^b by x^b,
then x equals 4a. -/
theorem find_x (a b x: ℝ) (h₁ : b ≠ 0) (h₂ : (2*a)^(2*b) = a^b * x^b) : x = 4*a := 
  sorry

end find_x_l59_59965


namespace water_percentage_in_fresh_grapes_l59_59997

theorem water_percentage_in_fresh_grapes 
  (P : ℝ) -- the percentage of water in fresh grapes
  (fresh_grapes_weight : ℝ := 40) -- weight of fresh grapes in kg
  (dry_grapes_weight : ℝ := 5) -- weight of dry grapes in kg
  (dried_grapes_water_percentage : ℝ := 20) -- percentage of water in dried grapes
  (solid_content : ℝ := 4) -- solid content in both fresh and dried grapes in kg
  : P = 90 :=
by
  sorry

end water_percentage_in_fresh_grapes_l59_59997


namespace decreasing_cubic_function_l59_59833

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^3 - x

-- Define the condition that f is decreasing on (-∞, ∞)
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

-- The main theorem that needs to be proven
theorem decreasing_cubic_function (m : ℝ) : is_decreasing (f m) → m < 0 := 
by
  sorry

end decreasing_cubic_function_l59_59833


namespace turtles_order_l59_59262

-- Define variables for each turtle as real numbers representing their positions
variables (O P S E R : ℝ)

-- Define the conditions given in the problem
def condition1 := S = O - 10
def condition2 := S = R + 25
def condition3 := R = E - 5
def condition4 := E = P - 25

-- Define the order of arrival
def order_of_arrival (O P S E R : ℝ) := 
     O = 0 ∧ 
     P = -5 ∧
     S = -10 ∧
     E = -30 ∧
     R = -35

-- Theorem to show the given conditions imply the order of arrival
theorem turtles_order (h1 : condition1 S O)
                     (h2 : condition2 S R)
                     (h3 : condition3 R E)
                     (h4 : condition4 E P) :
  order_of_arrival O P S E R :=
by sorry

end turtles_order_l59_59262


namespace function_d_has_no_boundary_point_l59_59251

def is_boundary_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  (∃ x₁ < x₀, f x₁ = 0) ∧ (∃ x₂ > x₀, f x₂ = 0)

def f_a (b : ℝ) (x : ℝ) : ℝ := x^2 + b * x - 2
def f_b (x : ℝ) : ℝ := abs (x^2 - 3)
def f_c (x : ℝ) : ℝ := 1 - abs (x - 2)
def f_d (x : ℝ) : ℝ := x^3 + x

theorem function_d_has_no_boundary_point :
  ¬ ∃ x₀ : ℝ, is_boundary_point f_d x₀ :=
sorry

end function_d_has_no_boundary_point_l59_59251


namespace ratio_of_areas_l59_59181

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l59_59181


namespace min_filtration_cycles_l59_59409

theorem min_filtration_cycles {c₀ : ℝ} (initial_concentration : c₀ = 225)
  (max_concentration : ℝ := 7.5) (reduction_factor : ℝ := 1 / 3)
  (log2 : ℝ := 0.3010) (log3 : ℝ := 0.4771) :
  ∃ n : ℕ, (c₀ * (reduction_factor ^ n) ≤ max_concentration ∧ n ≥ 9) :=
sorry

end min_filtration_cycles_l59_59409


namespace polynomial_divisible_exists_l59_59730

theorem polynomial_divisible_exists (p : Polynomial ℤ) (a : ℕ → ℤ) (k : ℕ) 
  (h_inc : ∀ i j, i < j → a i < a j) (h_nonzero : ∀ i, i < k → p.eval (a i) ≠ 0) :
  ∃ a_0 : ℤ, ∀ i, i < k → p.eval (a i) ∣ p.eval a_0 := 
by
  sorry

end polynomial_divisible_exists_l59_59730


namespace quadrilateral_offset_l59_59650

theorem quadrilateral_offset (d A h₂ x : ℝ)
  (h_da: d = 40)
  (h_A: A = 400)
  (h_h2 : h₂ = 9)
  (h_area : A = 1/2 * d * (x + h₂)) : 
  x = 11 :=
by sorry

end quadrilateral_offset_l59_59650


namespace money_distribution_problem_l59_59063

theorem money_distribution_problem :
  ∃ n : ℕ, (3 * n + n * (n - 1) / 2 = 100 * n) ∧ n = 195 :=
by {
  use 195,
  sorry
}

end money_distribution_problem_l59_59063


namespace find_x_values_l59_59821

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

end find_x_values_l59_59821


namespace company_bought_gravel_l59_59363

def weight_of_gravel (total_weight_of_materials : ℝ) (weight_of_sand : ℝ) : ℝ :=
  total_weight_of_materials - weight_of_sand

theorem company_bought_gravel :
  weight_of_gravel 14.02 8.11 = 5.91 := 
by
  sorry

end company_bought_gravel_l59_59363


namespace no_prime_satisfies_polynomial_l59_59042

theorem no_prime_satisfies_polynomial :
  ∀ p : ℕ, p.Prime → p^3 - 6*p^2 - 3*p + 14 ≠ 0 := by
  sorry

end no_prime_satisfies_polynomial_l59_59042


namespace linda_fraction_savings_l59_59314

theorem linda_fraction_savings (savings tv_cost : ℝ) (f : ℝ) 
  (h1 : savings = 800) 
  (h2 : tv_cost = 200) 
  (h3 : f * savings + tv_cost = savings) : 
  f = 3 / 4 := 
sorry

end linda_fraction_savings_l59_59314


namespace solve_digits_l59_59941

theorem solve_digits : ∃ A B C : ℕ, (A = 1 ∧ B = 0 ∧ (C = 9 ∨ C = 1)) ∧ 
  (∃ (X : ℕ), X ≥ 2 ∧ (C = X - 1 ∨ C = 1)) ∧ 
  (A * 1000 + B * 100 + B * 10 + C) * (C * 100 + C * 10 + A) = C * 100000 + C * 10000 + C * 1000 + C * 100 + A * 10 + C :=
by sorry

end solve_digits_l59_59941


namespace student_solves_exactly_20_problems_l59_59024

theorem student_solves_exactly_20_problems :
  (∀ n, 1 ≤ (a : ℕ → ℕ) n) ∧ (∀ k, a (k + 7) ≤ a k + 12) ∧ a 77 ≤ 132 →
  ∃ i j, i < j ∧ a j - a i = 20 := sorry

end student_solves_exactly_20_problems_l59_59024


namespace winnieKeepsBalloons_l59_59423

-- Given conditions
def redBalloons : Nat := 24
def whiteBalloons : Nat := 39
def greenBalloons : Nat := 72
def chartreuseBalloons : Nat := 91
def totalFriends : Nat := 11

-- Total balloons
def totalBalloons : Nat := redBalloons + whiteBalloons + greenBalloons + chartreuseBalloons

-- Theorem: Prove the number of balloons Winnie keeps for herself
theorem winnieKeepsBalloons :
  totalBalloons % totalFriends = 6 :=
by
  -- Placeholder for the proof
  sorry

end winnieKeepsBalloons_l59_59423


namespace negation_of_prop_p_l59_59010

open Classical

variable (p : Prop)

def prop_p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_prop_p : ¬prop_p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end negation_of_prop_p_l59_59010


namespace solve_x_l59_59039

def otimes (a b : ℝ) : ℝ := a - 3 * b

theorem solve_x : ∃ x : ℝ, otimes x 1 + otimes 2 x = 1 ∧ x = -1 :=
by
  use -1
  rw [otimes, otimes]
  sorry

end solve_x_l59_59039


namespace albert_horses_l59_59221

variable {H C : ℝ}

theorem albert_horses :
  (2000 * H + 9 * C = 13400) ∧ (200 * H + 0.20 * 9 * C = 1880) ∧ (∀ x : ℝ, x = 2000) → H = 4 := 
by
  sorry

end albert_horses_l59_59221


namespace ellipse_focal_length_l59_59798

theorem ellipse_focal_length :
  ∀ a b c : ℝ, (a^2 = 11) → (b^2 = 3) → (c^2 = a^2 - b^2) → (2 * c = 4 * Real.sqrt 2) :=
by
  sorry

end ellipse_focal_length_l59_59798


namespace binomial_divisible_by_prime_l59_59493

-- Define the conditions: p is prime and 0 < k < p
variables (p k : ℕ)
variable (hp : Nat.Prime p)
variable (hk : 0 < k ∧ k < p)

-- State that the binomial coefficient \(\binom{p}{k}\) is divisible by \( p \)
theorem binomial_divisible_by_prime
  (p k : ℕ) (hp : Nat.Prime p) (hk : 0 < k ∧ k < p) :
  p ∣ Nat.choose p k :=
by
  sorry

end binomial_divisible_by_prime_l59_59493


namespace pyramid_property_l59_59477

-- Define the areas of the faces of the right-angled triangular pyramid.
variables (S_ABC S_ACD S_ADB S_BCD : ℝ)

-- Define the condition that the areas correspond to a right-angled triangular pyramid.
def right_angled_triangular_pyramid (S_ABC S_ACD S_ADB S_BCD : ℝ) : Prop :=
  S_BCD^2 = S_ABC^2 + S_ACD^2 + S_ADB^2

-- State the theorem to be proven.
theorem pyramid_property : right_angled_triangular_pyramid S_ABC S_ACD S_ADB S_BCD :=
sorry

end pyramid_property_l59_59477


namespace star_running_back_yardage_l59_59964

-- Definitions
def total_yardage : ℕ := 150
def catching_passes_yardage : ℕ := 60
def running_yardage (total_yardage catching_passes_yardage : ℕ) : ℕ :=
  total_yardage - catching_passes_yardage

-- Statement to prove
theorem star_running_back_yardage :
  running_yardage total_yardage catching_passes_yardage = 90 := 
sorry

end star_running_back_yardage_l59_59964


namespace find_value_of_p_l59_59440

-- Definition of the parabola and ellipse
def parabola (p : ℝ) : Set (ℝ × ℝ) := {xy | xy.1 ^ 2 = 2 * p * xy.2}
def ellipse : Set (ℝ × ℝ) := {xy | xy.1 ^ 2 / 6 + xy.2 ^ 2 / 4 = 1}

-- Hypotheses
variables (p : ℝ) (h_pos : p > 0)

-- Latus rectum tangent to the ellipse
theorem find_value_of_p (h_tangent : ∃ (x y : ℝ),
  (parabola p (x, y) ∧ ellipse (x, y) ∧ y = -p / 2)) : p = 4 := sorry

end find_value_of_p_l59_59440


namespace relative_error_approximation_l59_59982

theorem relative_error_approximation (y : ℝ) (h : |y| < 1) :
  (1 / (1 + y) - (1 - y)) / (1 / (1 + y)) = y^2 :=
by
  sorry

end relative_error_approximation_l59_59982


namespace find_x_l59_59868

-- Definitions used in conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (4, x)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Main statement of the problem to be proved
theorem find_x (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = -1) : x = -1 / 5 :=
by {
  sorry
}

end find_x_l59_59868


namespace root_equation_value_l59_59367

theorem root_equation_value (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : 2026 - m^2 + 2 * m = 2023 :=
sorry

end root_equation_value_l59_59367


namespace remainder_polynomial_l59_59874

theorem remainder_polynomial (x : ℤ) : (1 + x) ^ 2010 % (1 + x + x^2) = 1 := 
  sorry

end remainder_polynomial_l59_59874


namespace maximize_profit_l59_59920

-- Define the price of the book
variables (p : ℝ) (p_max : ℝ)
-- Define the revenue function
def R (p : ℝ) : ℝ := p * (150 - 4 * p)
-- Define the profit function accounting for fixed costs of $200
def P (p : ℝ) := R p - 200
-- Set the maximum feasible price
def max_price_condition := p_max = 30
-- Define the price that maximizes the profit
def optimal_price := 18.75

-- The theorem to be proved
theorem maximize_profit : p_max = 30 → p = 18.75 → P p = 2612.5 :=
by {
  sorry
}

end maximize_profit_l59_59920


namespace find_b_l59_59206

theorem find_b (a b c : ℕ) (h1 : a * b + b * c - c * a = 0) (h2 : a - c = 101) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) : b = 2550 :=
sorry

end find_b_l59_59206


namespace range_of_set_l59_59736

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l59_59736


namespace brad_zip_code_l59_59488

theorem brad_zip_code (x y : ℕ) (h1 : x + x + 0 + 2 * x + y = 10) : 2 * x + y = 8 :=
by 
  sorry

end brad_zip_code_l59_59488


namespace sum_of_three_numbers_l59_59365

theorem sum_of_three_numbers :
  ((3 : ℝ) / 8) + 0.125 + 9.51 = 10.01 :=
sorry

end sum_of_three_numbers_l59_59365


namespace number_is_correct_l59_59293

theorem number_is_correct (x : ℝ) (h : 0.35 * x = 0.25 * 50) : x = 35.7143 :=
by 
  sorry

end number_is_correct_l59_59293


namespace regular_polygon_sides_l59_59620

theorem regular_polygon_sides (n : ℕ) (h : ∀ n, (n > 2) → (360 / n = 20)) : n = 18 := sorry

end regular_polygon_sides_l59_59620


namespace instructors_teach_together_in_360_days_l59_59364

def Felicia_teaches_every := 5
def Greg_teaches_every := 3
def Hannah_teaches_every := 9
def Ian_teaches_every := 2
def Joy_teaches_every := 8

def lcm_multiple (a b c d e : ℕ) : ℕ := Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e)))

theorem instructors_teach_together_in_360_days :
  lcm_multiple Felicia_teaches_every
               Greg_teaches_every
               Hannah_teaches_every
               Ian_teaches_every
               Joy_teaches_every = 360 :=
by
  -- Since the real proof is omitted, we close with sorry
  sorry

end instructors_teach_together_in_360_days_l59_59364


namespace jiwoo_magnets_two_digit_count_l59_59119

def num_magnets : List ℕ := [1, 2, 7]

theorem jiwoo_magnets_two_digit_count : 
  (∀ (x y : ℕ), x ≠ y → x ∈ num_magnets → y ∈ num_magnets → 2 * 3 = 6) := 
by {
  sorry
}

end jiwoo_magnets_two_digit_count_l59_59119


namespace sports_probability_boy_given_sports_probability_l59_59869

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

end sports_probability_boy_given_sports_probability_l59_59869


namespace hexagon_sequences_l59_59648

theorem hexagon_sequences : ∃ n : ℕ, n = 7 ∧ 
  ∀ (x d : ℕ), 6 * x + 15 * d = 720 ∧ (2 * x + 5 * d = 240) ∧ 
  (x + 5 * d < 160) ∧ (0 < x) ∧ (0 < d) ∧ (d % 2 = 0) ↔ (∃ k < n, (∃ x, ∃ d, x = 85 - 2*k ∧ d = 2 + 2*k)) :=
by
  sorry

end hexagon_sequences_l59_59648


namespace necessary_condition_for_abs_ab_l59_59446

theorem necessary_condition_for_abs_ab {a b : ℝ} (h : |a - b| = |a| - |b|) : ab ≥ 0 :=
sorry

end necessary_condition_for_abs_ab_l59_59446


namespace probability_divisible_by_5_l59_59279

def is_three_digit_integer (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

theorem probability_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit_integer N) 
  (h2 : ends_with_five N) : 
  ∃ (p : ℚ), p = 1 := 
sorry

end probability_divisible_by_5_l59_59279


namespace star_5_3_eq_31_l59_59569

def star (a b : ℤ) : ℤ := a^2 + a * b - b^2

theorem star_5_3_eq_31 : star 5 3 = 31 :=
by
  sorry

end star_5_3_eq_31_l59_59569


namespace find_a_b_l59_59247

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x < -2 ∨ x > 1) → (x^2 + a * x + b > 0)) →
  (a = 1 ∧ b = -2) :=
by
  sorry

end find_a_b_l59_59247


namespace linear_function_expression_l59_59853

theorem linear_function_expression (k b : ℝ) (h : ∀ x : ℝ, (1 ≤ x ∧ x ≤ 4 → 3 ≤ k * x + b ∧ k * x + b ≤ 6)) :
  (k = 1 ∧ b = 2) ∨ (k = -1 ∧ b = 7) :=
by
  sorry

end linear_function_expression_l59_59853


namespace number_of_leap_years_l59_59651

noncomputable def is_leap_year (year : ℕ) : Prop :=
  (year % 1300 = 300 ∨ year % 1300 = 700) ∧ 2000 ≤ year ∧ year ≤ 5000

noncomputable def leap_years : List ℕ :=
  [2900, 4200, 3300, 4600]

theorem number_of_leap_years : leap_years.length = 4 ∧ ∀ y ∈ leap_years, is_leap_year y := by
  sorry

end number_of_leap_years_l59_59651


namespace parabola_directrix_tangent_circle_l59_59675

theorem parabola_directrix_tangent_circle (p : ℝ) (h_pos : 0 < p) (h_tangent: ∃ x : ℝ, (x = p/2) ∧ (x-5)^2 + (0:ℝ)^2 = 25) : p = 20 :=
sorry

end parabola_directrix_tangent_circle_l59_59675


namespace num_positive_terms_arithmetic_seq_l59_59932

theorem num_positive_terms_arithmetic_seq :
  (∃ k : ℕ+, (∀ n : ℕ, n ≤ k → (90 - 2 * n) > 0)) → (k = 44) :=
sorry

end num_positive_terms_arithmetic_seq_l59_59932


namespace average_mark_of_excluded_students_l59_59946

theorem average_mark_of_excluded_students (N A E A_R A_E : ℝ) 
  (hN : N = 25) 
  (hA : A = 80) 
  (hE : E = 5) 
  (hAR : A_R = 90) 
  (h_eq : N * A - E * A_E = (N - E) * A_R) : 
  A_E = 40 := 
by 
  sorry

end average_mark_of_excluded_students_l59_59946


namespace solve_for_y_l59_59413

theorem solve_for_y (y : ℕ) : (1000^4 = 10^y) → y = 12 :=
by {
  sorry
}

end solve_for_y_l59_59413


namespace num_supermarkets_us_l59_59026

noncomputable def num_supermarkets_total : ℕ := 84

noncomputable def us_canada_relationship (C : ℕ) : Prop := C + (C + 10) = num_supermarkets_total

theorem num_supermarkets_us (C : ℕ) (h : us_canada_relationship C) : C + 10 = 47 :=
sorry

end num_supermarkets_us_l59_59026


namespace solution_set_of_quadratic_inequality_l59_59032

theorem solution_set_of_quadratic_inequality (x : ℝ) : 
  x^2 + 3*x - 4 < 0 ↔ -4 < x ∧ x < 1 :=
sorry

end solution_set_of_quadratic_inequality_l59_59032


namespace bryden_amount_correct_l59_59408

-- Each state quarter has a face value of $0.25.
def face_value (q : ℕ) : ℝ := 0.25 * q

-- The collector offers to buy the state quarters for 1500% of their face value.
def collector_multiplier : ℝ := 15

-- Bryden has 10 state quarters.
def bryden_quarters : ℕ := 10

-- Calculate the amount Bryden will get for his 10 state quarters.
def amount_received : ℝ := collector_multiplier * face_value bryden_quarters

-- Prove that the amount received by Bryden equals $37.5.
theorem bryden_amount_correct : amount_received = 37.5 :=
by
  sorry

end bryden_amount_correct_l59_59408


namespace find_m_in_hyperbola_l59_59984

-- Define the problem in Lean 4
theorem find_m_in_hyperbola (m : ℝ) (x y : ℝ) (e : ℝ) (a_sq : ℝ := 9) (h_eq : e = 2) (h_hyperbola : x^2 / a_sq - y^2 / m = 1) : m = 27 :=
sorry

end find_m_in_hyperbola_l59_59984


namespace mean_temperature_l59_59096

theorem mean_temperature
  (temps : List ℤ) 
  (h_temps : temps = [-8, -3, -7, -6, 0, 4, 6, 5, -1, 2]) :
  (temps.sum: ℚ) / temps.length = -0.8 := 
by
  sorry

end mean_temperature_l59_59096


namespace pieces_equality_l59_59966

-- Define the pieces of chocolate and their areas.
def piece1_area : ℝ := 6 -- Area of triangle EBC
def piece2_area : ℝ := 6 -- Area of triangle AEC
def piece3_area : ℝ := 6 -- Area of polygon AHGFD
def piece4_area : ℝ := 6 -- Area of polygon CFGH

-- State the problem: proving the equality of the areas.
theorem pieces_equality : piece1_area = piece2_area ∧ piece2_area = piece3_area ∧ piece3_area = piece4_area :=
by
  sorry

end pieces_equality_l59_59966


namespace sum_of_roots_l59_59802

theorem sum_of_roots :
  let a := (6 : ℝ) + 3 * Real.sqrt 3
  let b := (3 : ℝ) + Real.sqrt 3
  let c := -(3 : ℝ)
  let root_sum := -b / a
  root_sum = -1 + Real.sqrt 3 / 3 := sorry

end sum_of_roots_l59_59802


namespace non_monotonic_piecewise_l59_59117

theorem non_monotonic_piecewise (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ (x t : ℝ),
    (f x = if x ≤ t then (4 * a - 3) * x + (2 * a - 4) else (2 * x^3 - 6 * x)))
  : a ≤ 3 / 4 := 
sorry

end non_monotonic_piecewise_l59_59117


namespace determine_g_l59_59516

variable {R : Type*} [CommRing R]

theorem determine_g (g : R → R) (x : R) :
  (4 * x^5 + 3 * x^3 - 2 * x + 1 + g x = 7 * x^3 - 5 * x^2 + 4 * x - 3) →
  g x = -4 * x^5 + 4 * x^3 - 5 * x^2 + 6 * x - 4 :=
by
  sorry

end determine_g_l59_59516


namespace isosceles_triangle_angle_l59_59058

-- Definition of required angles and the given geometric context
variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]
variables (angleBAC : ℝ) (angleBCA : ℝ)

-- Given: shared vertex A, with angle BAC of pentagon
axiom angleBAC_def : angleBAC = 108

-- To Prove: determining the measure of angle BCA in the isosceles triangle
theorem isosceles_triangle_angle (h : 180 > 2 * angleBAC) : angleBCA = (180 - angleBAC) / 2 :=
  sorry

end isosceles_triangle_angle_l59_59058


namespace ones_digit_of_power_l59_59164

theorem ones_digit_of_power (n : ℕ) : 
  (13 ^ (13 * (12 ^ 12)) % 10) = 9 :=
by
  sorry

end ones_digit_of_power_l59_59164


namespace my_op_identity_l59_59380

def my_op (a b : ℕ) : ℕ := a + b + a * b

theorem my_op_identity (a : ℕ) : my_op (my_op a 1) 2 = 6 * a + 5 :=
by
  sorry

end my_op_identity_l59_59380


namespace rectangle_area_increase_l59_59302

variable (L B : ℝ)

theorem rectangle_area_increase :
  let L_new := 1.30 * L
  let B_new := 1.45 * B
  let A_original := L * B
  let A_new := L_new * B_new
  let A_increase := A_new - A_original
  let percentage_increase := (A_increase / A_original) * 100
  percentage_increase = 88.5 := by
    sorry

end rectangle_area_increase_l59_59302


namespace triangle_inequality_l59_59238

variables {α β γ a b c : ℝ}
variable {n : ℕ}

theorem triangle_inequality (h_sum_angles : α + β + γ = Real.pi) (h_pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  (Real.pi / 3) ^ n ≤ (a * α ^ n + b * β ^ n + c * γ ^ n) / (a + b + c) ∧ 
  (a * α ^ n + b * β ^ n + c * γ ^ n) / (a + b + c) < (Real.pi ^ n / 2) :=
by
  sorry

end triangle_inequality_l59_59238


namespace parallel_lines_eq_l59_59395

theorem parallel_lines_eq {a x y : ℝ} :
  (∀ x y : ℝ, x + a * y = 2 * a + 2) ∧ (∀ x y : ℝ, a * x + y = a + 1) →
  a = 1 :=
by
  sorry

end parallel_lines_eq_l59_59395


namespace problem_sol_l59_59341

-- Assume g is an invertible function
variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
variable (h_invertible : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y)

-- Define p and q such that g(p) = 3 and g(q) = 5
variable (p q : ℝ)
variable (h1 : g p = 3) (h2 : g q = 5)

-- Goal to prove that p - q = 2
theorem problem_sol : p - q = 2 :=
by
  sorry

end problem_sol_l59_59341


namespace karen_wins_in_race_l59_59711

theorem karen_wins_in_race (w : ℝ) (h1 : w / 45 > 1 / 15) 
    (h2 : 60 * (w / 45 - 1 / 15) = w + 4) : 
    w = 8 / 3 := 
sorry

end karen_wins_in_race_l59_59711


namespace james_shirts_l59_59153

theorem james_shirts (S P : ℕ) (h1 : P = S / 2) (h2 : 6 * S + 8 * P = 100) : S = 10 :=
sorry

end james_shirts_l59_59153


namespace no_real_solutions_for_equation_l59_59333

theorem no_real_solutions_for_equation : ¬ (∃ x : ℝ, x + Real.sqrt (2 * x - 6) = 5) :=
sorry

end no_real_solutions_for_equation_l59_59333


namespace optimal_washing_effect_l59_59655

noncomputable def total_capacity : ℝ := 20 -- kilograms
noncomputable def weight_clothes : ℝ := 5 -- kilograms
noncomputable def weight_detergent_existing : ℝ := 2 * 0.02 -- kilograms
noncomputable def optimal_concentration : ℝ := 0.004 -- kilograms per kilogram of water

theorem optimal_washing_effect :
  ∃ (additional_detergent additional_water : ℝ),
    additional_detergent = 0.02 ∧ additional_water = 14.94 ∧
    weight_clothes + additional_water + weight_detergent_existing + additional_detergent = total_capacity ∧
    weight_detergent_existing + additional_detergent = optimal_concentration * additional_water :=
by
  sorry

end optimal_washing_effect_l59_59655


namespace geometric_sequence_sum_is_120_l59_59517

noncomputable def sum_first_four_geometric_seq (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4

theorem geometric_sequence_sum_is_120 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_pos_geometric : 0 < q ∧ q < 1)
  (h_a3_a5 : a 3 + a 5 = 20)
  (h_a3_a5_product : a 3 * a 5 = 64) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) :
  sum_first_four_geometric_seq a q = 120 :=
sorry

end geometric_sequence_sum_is_120_l59_59517


namespace relationship_among_abc_l59_59334

theorem relationship_among_abc (e1 e2 : ℝ) (h1 : 0 ≤ e1) (h2 : e1 < 1) (h3 : e2 > 1) :
  let a := 3 ^ e1
  let b := 2 ^ (-e2)
  let c := Real.sqrt 5
  b < c ∧ c < a := by
  sorry

end relationship_among_abc_l59_59334


namespace range_of_a_l59_59901

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ)
  (h_cond : ∀ (n : ℕ), n > 0 → (a_seq n = if n ≤ 4 then 2^n - 1 else -n^2 + (a - 1) * n))
  (h_max_a5 : ∀ (n : ℕ), n > 0 → a_seq n ≤ a_seq 5) :
  9 ≤ a ∧ a ≤ 12 := 
by
  sorry

end range_of_a_l59_59901


namespace find_remainder_q_neg2_l59_59864

-- Define q(x)
def q (x : ℝ) (D E F : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 6

-- The given conditions in the problem
variable {D E F : ℝ}
variable (h_q_2 : q 2 D E F = 14)

-- The statement we aim to prove
theorem find_remainder_q_neg2 (h_q_2 : q 2 D E F = 14) : q (-2) D E F = 14 :=
sorry

end find_remainder_q_neg2_l59_59864


namespace max_value_fraction_l59_59619

theorem max_value_fraction (x y : ℝ) (hx : 1 / 3 ≤ x ∧ x ≤ 3 / 5) (hy : 1 / 4 ≤ y ∧ y ≤ 1 / 2) :
  (∃ x y, (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) ∧ (xy / (x^2 + y^2) = 6 / 13)) :=
by
  sorry

end max_value_fraction_l59_59619


namespace interest_rate_l59_59595

theorem interest_rate (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) (diff : ℝ) 
    (hP : P = 1500)
    (ht : t = 2)
    (hdiff : diff = 15)
    (hCI : CI = P * (1 + r / 100)^t - P)
    (hSI : SI = P * r * t / 100)
    (hCI_SI_diff : CI - SI = diff) :
    r = 1 := 
by
  sorry -- proof goes here


end interest_rate_l59_59595


namespace sum_of_abc_eq_11_l59_59582

theorem sum_of_abc_eq_11 (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_order : a < b ∧ b < c)
  (h_inv_sum : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1) : a + b + c = 11 :=
  sorry

end sum_of_abc_eq_11_l59_59582


namespace seventeen_divides_l59_59296

theorem seventeen_divides (a b : ℤ) (h : 17 ∣ (2 * a + 3 * b)) : 17 ∣ (9 * a + 5 * b) :=
sorry

end seventeen_divides_l59_59296


namespace dave_spent_on_books_l59_59694

-- Define the cost of books in each category without any discounts or taxes
def cost_animal_books : ℝ := 8 * 10
def cost_outer_space_books : ℝ := 6 * 12
def cost_train_books : ℝ := 9 * 8
def cost_history_books : ℝ := 4 * 15
def cost_science_books : ℝ := 5 * 18

-- Define the discount and tax rates
def discount_animal_books : ℝ := 0.10
def tax_science_books : ℝ := 0.15

-- Apply the discount to animal books
def discounted_cost_animal_books : ℝ := cost_animal_books * (1 - discount_animal_books)

-- Apply the tax to science books
def final_cost_science_books : ℝ := cost_science_books * (1 + tax_science_books)

-- Calculate the total cost of all books after discounts and taxes
def total_cost : ℝ := discounted_cost_animal_books 
                  + cost_outer_space_books
                  + cost_train_books
                  + cost_history_books
                  + final_cost_science_books

theorem dave_spent_on_books : total_cost = 379.5 := by
  sorry

end dave_spent_on_books_l59_59694


namespace ab_value_l59_59890

theorem ab_value (a b : ℚ) (h1 : 3 * a - 8 = 0) (h2 : b = 3) : a * b = 8 :=
by
  sorry

end ab_value_l59_59890


namespace xiao_ming_error_step_l59_59223

theorem xiao_ming_error_step (x : ℝ) :
  (1 / (x + 1) = (2 * x) / (3 * x + 3) - 1) → 
  3 = 2 * x - (3 * x + 3) → 
  (3 = 2 * x - 3 * x + 3) ↔ false := by
  sorry

end xiao_ming_error_step_l59_59223


namespace time_to_pass_tree_l59_59606

-- Define the conditions given in the problem
def train_length : ℕ := 1200
def platform_length : ℕ := 700
def time_to_pass_platform : ℕ := 190

-- Calculate the total distance covered while passing the platform
def distance_passed_platform : ℕ := train_length + platform_length

-- The main theorem we need to prove
theorem time_to_pass_tree : (distance_passed_platform / time_to_pass_platform) * train_length = 120 := 
by
  sorry

end time_to_pass_tree_l59_59606


namespace triangle_side_length_l59_59227

theorem triangle_side_length (a b p : ℝ) (H_perimeter : a + b + 10 = p) (H_a : a = 7) (H_b : b = 15) (H_p : p = 32) : 10 = 10 :=
by
  sorry

end triangle_side_length_l59_59227


namespace trigonometric_identity_l59_59208

open Real

theorem trigonometric_identity (α φ : ℝ) :
  cos α ^ 2 + cos φ ^ 2 + cos (α + φ) ^ 2 - 2 * cos α * cos φ * cos (α + φ) = 1 :=
sorry

end trigonometric_identity_l59_59208


namespace subtraction_like_terms_l59_59082

variable (a : ℝ)

theorem subtraction_like_terms : 3 * a ^ 2 - 2 * a ^ 2 = a ^ 2 :=
by
  sorry

end subtraction_like_terms_l59_59082


namespace roots_cubic_polynomial_l59_59968

theorem roots_cubic_polynomial (a b c : ℝ) 
  (h1 : a^3 - 2*a - 2 = 0) 
  (h2 : b^3 - 2*b - 2 = 0) 
  (h3 : c^3 - 2*c - 2 = 0) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -18 :=
by
  sorry

end roots_cubic_polynomial_l59_59968


namespace sum_of_variables_is_233_l59_59362

-- Define A, B, C, D, E, F with their corresponding values.
def A : ℤ := 13
def B : ℤ := 9
def C : ℤ := -3
def D : ℤ := -2
def E : ℕ := 165
def F : ℕ := 51

-- Define the main theorem to prove the sum of A, B, C, D, E, F equals 233.
theorem sum_of_variables_is_233 : A + B + C + D + E + F = 233 := 
by {
  -- Proof is not required according to problem statement, hence using sorry.
  sorry
}

end sum_of_variables_is_233_l59_59362


namespace count_divisible_2_3_or_5_lt_100_l59_59995
-- We need the Mathlib library for general mathematical functions

-- The main theorem statement
theorem count_divisible_2_3_or_5_lt_100 : 
  let A2 := Nat.floor (100 / 2)
  let A3 := Nat.floor (100 / 3)
  let A5 := Nat.floor (100 / 5)
  let A23 := Nat.floor (100 / 6)
  let A25 := Nat.floor (100 / 10)
  let A35 := Nat.floor (100 / 15)
  let A235 := Nat.floor (100 / 30)
  (A2 + A3 + A5 - A23 - A25 - A35 + A235) = 74 :=
by
  sorry

end count_divisible_2_3_or_5_lt_100_l59_59995


namespace faye_pencils_l59_59429

theorem faye_pencils (rows crayons : ℕ) (pencils_per_row : ℕ) (h1 : rows = 7) (h2 : pencils_per_row = 5) : 
  (rows * pencils_per_row) = 35 :=
by {
  sorry
}

end faye_pencils_l59_59429


namespace power_identity_l59_59322

theorem power_identity (a b : ℕ) (R S : ℕ) (hR : R = 2^a) (hS : S = 5^b) : 
    20^(a * b) = R^(2 * b) * S^a := 
by 
    -- Insert the proof here
    sorry

end power_identity_l59_59322


namespace sara_quarters_final_l59_59018

def initial_quarters : ℕ := 21
def quarters_from_dad : ℕ := 49
def quarters_spent_at_arcade : ℕ := 15
def dollar_bills_from_mom : ℕ := 2
def quarters_per_dollar : ℕ := 4

theorem sara_quarters_final :
  (initial_quarters + quarters_from_dad - quarters_spent_at_arcade + dollar_bills_from_mom * quarters_per_dollar) = 63 :=
by
  sorry

end sara_quarters_final_l59_59018


namespace asymptotes_of_hyperbola_l59_59442

-- Definitions
variables (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)

-- Theorem: Equation of the asymptotes of the given hyperbola
theorem asymptotes_of_hyperbola (h_equiv : b = 2 * a) :
  ∀ x y : ℝ, 
    (x ≠ 0 ∧ y ≠ 0 ∧ (y = (2 : ℝ) * x ∨ y = - (2 : ℝ) * x)) ↔ (x, y) ∈ {p : ℝ × ℝ | (x^2 / a^2) - (y^2 / b^2) = 1} := 
sorry

end asymptotes_of_hyperbola_l59_59442


namespace quadratic_solutions_l59_59713

theorem quadratic_solutions : ∀ x : ℝ, x^2 - 25 = 0 → (x = 5 ∨ x = -5) :=
by
  sorry

end quadratic_solutions_l59_59713


namespace sum_of_solutions_l59_59502

theorem sum_of_solutions (y : ℤ) (x1 x2 : ℤ) (h1 : y = 8) (h2 : x1^2 + y^2 = 145) (h3 : x2^2 + y^2 = 145) : x1 + x2 = 0 := by
  sorry

end sum_of_solutions_l59_59502


namespace power_seven_evaluation_l59_59484

theorem power_seven_evaluation (a b : ℝ) (h : a = (7 : ℝ)^(1/4) ∧ b = (7 : ℝ)^(1/7)) : 
  a / b = (7 : ℝ)^(3/28) :=
  sorry

end power_seven_evaluation_l59_59484


namespace volume_s_l59_59439

def condition1 (x y : ℝ) : Prop := |9 - x| + y ≤ 12
def condition2 (x y : ℝ) : Prop := 3 * y - x ≥ 18
def S (x y : ℝ) : Prop := condition1 x y ∧ condition2 x y

def is_volume_correct (m n : ℕ) (p : ℕ) :=
  (m + n + p = 153) ∧ (m = 135) ∧ (n = 8) ∧ (p = 10)

theorem volume_s (m n p : ℕ) :
  (∀ x y : ℝ, S x y) → is_volume_correct m n p :=
by 
  sorry

end volume_s_l59_59439


namespace choir_robe_costs_l59_59203

theorem choir_robe_costs:
  ∀ (total_robes needed_robes total_cost robe_cost : ℕ),
  total_robes = 30 →
  needed_robes = 30 - 12 →
  total_cost = 36 →
  total_cost = needed_robes * robe_cost →
  robe_cost = 2 :=
by
  intros total_robes needed_robes total_cost robe_cost
  intro h_total_robes h_needed_robes h_total_cost h_cost_eq
  sorry

end choir_robe_costs_l59_59203


namespace factor_expr_l59_59921

variable (x : ℝ)

def expr : ℝ := (20 * x ^ 3 + 100 * x - 10) - (-5 * x ^ 3 + 5 * x - 10)

theorem factor_expr :
  expr x = 5 * x * (5 * x ^ 2 + 19) :=
by
  sorry

end factor_expr_l59_59921


namespace total_bus_capacity_l59_59924

def left_seats : ℕ := 15
def right_seats : ℕ := left_seats - 3
def people_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 8

theorem total_bus_capacity :
  (left_seats + right_seats) * people_per_seat + back_seat_capacity = 89 := by
  sorry

end total_bus_capacity_l59_59924


namespace part1_solution_l59_59531

def f (x m : ℝ) := |x + m| + |2 * x + 1|

theorem part1_solution (x : ℝ) : f x (-1) ≤ 3 → -1 ≤ x ∧ x ≤ 1 := 
sorry

end part1_solution_l59_59531


namespace cucumber_to_tomato_ratio_l59_59571

variable (total_rows : ℕ) (space_per_row_tomato : ℕ) (tomatoes_per_plant : ℕ) (total_tomatoes : ℕ)

/-- Aubrey's Garden -/
theorem cucumber_to_tomato_ratio (total_rows_eq : total_rows = 15)
  (space_per_row_tomato_eq : space_per_row_tomato = 8)
  (tomatoes_per_plant_eq : tomatoes_per_plant = 3)
  (total_tomatoes_eq : total_tomatoes = 120) :
  let total_tomato_plants := total_tomatoes / tomatoes_per_plant
  let rows_tomato := total_tomato_plants / space_per_row_tomato
  let rows_cucumber := total_rows - rows_tomato
  (2 * rows_tomato = rows_cucumber)
:=
by
  sorry

end cucumber_to_tomato_ratio_l59_59571


namespace equivalence_a_gt_b_and_inv_a_lt_inv_b_l59_59521

variable {a b : ℝ}

theorem equivalence_a_gt_b_and_inv_a_lt_inv_b (h : a * b > 0) : 
  (a > b) ↔ (1 / a < 1 / b) := 
sorry

end equivalence_a_gt_b_and_inv_a_lt_inv_b_l59_59521


namespace average_price_over_3_months_l59_59198

theorem average_price_over_3_months (dMay : ℕ) 
  (pApril pMay pJune : ℝ) 
  (h1 : pApril = 1.20) 
  (h2 : pMay = 1.20) 
  (h3 : pJune = 3.00) 
  (h4 : dApril = 2 / 3 * dMay) 
  (h5 : dJune = 2 * dApril) :
  ((dApril * pApril + dMay * pMay + dJune * pJune) / (dApril + dMay + dJune) = 2) := 
by sorry

end average_price_over_3_months_l59_59198


namespace percentage_deposit_paid_l59_59828

theorem percentage_deposit_paid (D R T : ℝ) (hd : D = 105) (hr : R = 945) (ht : T = D + R) : (D / T) * 100 = 10 := by
  sorry

end percentage_deposit_paid_l59_59828


namespace fraction_division_l59_59841

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l59_59841


namespace number_of_ordered_triples_l59_59482

/-- 
Prove the number of ordered triples (x, y, z) of positive integers that satisfy 
  lcm(x, y) = 180, lcm(x, z) = 210, and lcm(y, z) = 420 is 2.
-/
theorem number_of_ordered_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h₁ : Nat.lcm x y = 180) (h₂ : Nat.lcm x z = 210) (h₃ : Nat.lcm y z = 420) : 
  ∃ (n : ℕ), n = 2 := 
sorry

end number_of_ordered_triples_l59_59482


namespace solution_is_correct_l59_59886

noncomputable def solve_system_of_inequalities : Prop :=
  ∃ x y : ℝ, 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧ 
    (2 * x - 4 * y ≤ -3) ∧ 
    (x = -1/3) ∧ 
    (y = 2/3)

theorem solution_is_correct : solve_system_of_inequalities :=
sorry

end solution_is_correct_l59_59886


namespace sum_of_squares_divisibility_l59_59831

theorem sum_of_squares_divisibility
  (p : ℕ) (hp : Nat.Prime p)
  (x y z : ℕ)
  (hx : 0 < x) (hxy : x < y) (hyz : y < z) (hzp : z < p)
  (hmod_eq : ∀ a b c : ℕ, a^3 % p = b^3 % p → b^3 % p = c^3 % p → a^3 % p = c^3 % p) :
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
  sorry

end sum_of_squares_divisibility_l59_59831


namespace greatest_number_of_roses_l59_59770

noncomputable def individual_rose_price: ℝ := 2.30
noncomputable def dozen_rose_price: ℝ := 36
noncomputable def two_dozen_rose_price: ℝ := 50
noncomputable def budget: ℝ := 680

theorem greatest_number_of_roses (P: ℝ → ℝ → ℝ → ℝ → ℕ) :
  P individual_rose_price dozen_rose_price two_dozen_rose_price budget = 325 :=
sorry

end greatest_number_of_roses_l59_59770


namespace sum_of_coefficients_256_l59_59354

theorem sum_of_coefficients_256 (n : ℕ) (h : (3 + 1)^n = 256) : n = 4 :=
sorry

end sum_of_coefficients_256_l59_59354


namespace arithmetic_seq_sum_2017_l59_59683

theorem arithmetic_seq_sum_2017 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (a1 : a 1 = -2017) 
  (h1 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1))
  (h2 : (S 2014) / 2014 - (S 2008) / 2008 = 6) : 
  S 2017 = -2017 :=
by
  sorry

end arithmetic_seq_sum_2017_l59_59683


namespace fraction_simplification_l59_59872

theorem fraction_simplification :
  (2/5 + 3/4) / (4/9 + 1/6) = (207/110) := by
  sorry

end fraction_simplification_l59_59872


namespace distance_from_P_to_x_axis_l59_59727

-- Define the point P with coordinates (4, -3)
def P : ℝ × ℝ := (4, -3)

-- Define the distance from a point to the x-axis as the absolute value of the y-coordinate
def distance_to_x_axis (point : ℝ × ℝ) : ℝ :=
  abs point.snd

-- State the theorem to be proved
theorem distance_from_P_to_x_axis : distance_to_x_axis P = 3 :=
by
  -- The proof is not required; we can use sorry to skip it
  sorry

end distance_from_P_to_x_axis_l59_59727


namespace solve_inequality1_solve_inequality2_l59_59271

-- Problem 1: Solve the inequality (1)
theorem solve_inequality1 (x : ℝ) (h : x ≠ -4) : 
  (2 - x) / (x + 4) ≤ 0 ↔ (x ≥ 2 ∨ x < -4) := sorry

-- Problem 2: Solve the inequality (2) for different cases of a
theorem solve_inequality2 (x a : ℝ) : 
  (x^2 - 3 * a * x + 2 * a^2 ≥ 0) ↔
  (if a > 0 then (x ≥ 2 * a ∨ x ≤ a) 
   else if a < 0 then (x ≥ a ∨ x ≤ 2 * a) 
   else true) := sorry

end solve_inequality1_solve_inequality2_l59_59271


namespace find_number_l59_59878

theorem find_number (x : ℤ) (h : 3 * x - 4 = 5) : x = 3 :=
sorry

end find_number_l59_59878


namespace counts_duel_with_marquises_l59_59916

theorem counts_duel_with_marquises (x y z k : ℕ) (h1 : 3 * x = 2 * y) (h2 : 6 * y = 3 * z)
    (h3 : ∀ c : ℕ, c = x → ∃ m : ℕ, m = k) : k = 6 :=
by
  sorry

end counts_duel_with_marquises_l59_59916


namespace sufficient_condition_for_increasing_l59_59490

theorem sufficient_condition_for_increasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^y < a^x) →
  (∀ x y : ℝ, x < y → (2 - a) * y ^ 3 > (2 - a) * x ^ 3) :=
sorry

end sufficient_condition_for_increasing_l59_59490


namespace evaluate_expressions_for_pos_x_l59_59973

theorem evaluate_expressions_for_pos_x :
  (∀ x : ℝ, x > 0 → 6^x * x^3 = 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → (3 * x)^(3 * x) ≠ 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → 3^x * x^6 ≠ 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → (6 * x)^x ≠ 6^x * x^3) →
  ∃ n : ℕ, n = 1 := 
by
  sorry

end evaluate_expressions_for_pos_x_l59_59973


namespace find_A_l59_59923

def U : Set ℕ := {1, 2, 3, 4, 5}

def compl_U (A : Set ℕ) : Set ℕ := U \ A

theorem find_A (A : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
  (h_compl_U : compl_U A = {2, 3}) : A = {1, 4, 5} :=
by
  sorry

end find_A_l59_59923


namespace diane_owes_money_l59_59267

theorem diane_owes_money (initial_amount winnings total_losses : ℤ) (h_initial : initial_amount = 100) (h_winnings : winnings = 65) (h_losses : total_losses = 215) : 
  initial_amount + winnings - total_losses = -50 := by
  sorry

end diane_owes_money_l59_59267


namespace sum_of_money_is_6000_l59_59927

noncomputable def original_interest (P R : ℝ) := (P * R * 3) / 100
noncomputable def new_interest (P R : ℝ) := (P * (R + 2) * 3) / 100

theorem sum_of_money_is_6000 (P R : ℝ) (h : new_interest P R - original_interest P R = 360) : P = 6000 :=
by
  sorry

end sum_of_money_is_6000_l59_59927


namespace solve_inequality_l59_59101

theorem solve_inequality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2)
  (h3 : (x^2 + 3*x - 1) / (4 - x^2) < 1)
  (h4 : (x^2 + 3*x - 1) / (4 - x^2) ≥ -1) :
  x < -5 / 2 ∨ (-1 ≤ x ∧ x < 1) :=
by sorry

end solve_inequality_l59_59101


namespace brenda_has_eight_l59_59687

-- Define the amounts each friend has
def emma_money : ℕ := 8
def daya_money : ℕ := emma_money + (emma_money / 4)
def jeff_money : ℕ := (2 * daya_money) / 5
def brenda_money : ℕ := jeff_money + 4

-- Define the theorem to prove Brenda's money is 8
theorem brenda_has_eight : brenda_money = 8 := by
  sorry

end brenda_has_eight_l59_59687


namespace problem_I_problem_II_l59_59186

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}
def B : Set ℝ := {x : ℝ | x ≥ 1}

-- Define the complement of A in the universal set U which is ℝ
def complement_U_A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

-- Define the union of complement_U_A and B
def union_complement_U_A_B : Set ℝ := complement_U_A ∪ B

-- Proof Problem I: Prove that the set A is as specified
theorem problem_I : A = {x : ℝ | x ≤ -1 ∨ x ≥ 3} := sorry

-- Proof Problem II: Prove that the union of the complement of A and B is as specified
theorem problem_II : union_complement_U_A_B = {x : ℝ | x > -1} := sorry

end problem_I_problem_II_l59_59186


namespace find_central_angle_l59_59077

variable (L : ℝ) (r : ℝ) (α : ℝ)

-- Given conditions
def arc_length_condition : Prop := L = 200
def radius_condition : Prop := r = 2
def arc_length_formula : Prop := L = r * α

-- Theorem statement
theorem find_central_angle 
  (hL : arc_length_condition L) 
  (hr : radius_condition r) 
  (hf : arc_length_formula L r α) : 
  α = 100 := by
  -- Proof goes here
  sorry

end find_central_angle_l59_59077


namespace back_wheel_revolutions_l59_59029

-- Defining relevant distances and conditions
def front_wheel_radius : ℝ := 3 -- radius in feet
def back_wheel_radius : ℝ := 0.5 -- radius in feet
def front_wheel_revolutions : ℕ := 120

-- The target theorem
theorem back_wheel_revolutions :
  let front_wheel_circumference := 2 * Real.pi * front_wheel_radius
  let total_distance := front_wheel_circumference * (front_wheel_revolutions : ℝ)
  let back_wheel_circumference := 2 * Real.pi * back_wheel_radius
  let back_wheel_revs := total_distance / back_wheel_circumference
  back_wheel_revs = 720 :=
by
  sorry

end back_wheel_revolutions_l59_59029


namespace rolls_sold_to_uncle_l59_59046

theorem rolls_sold_to_uncle (total_rolls needed_rolls rolls_to_grandmother rolls_to_neighbor rolls_to_uncle : ℕ)
  (h1 : total_rolls = 45)
  (h2 : needed_rolls = 28)
  (h3 : rolls_to_grandmother = 1)
  (h4 : rolls_to_neighbor = 6)
  (h5 : rolls_to_uncle + rolls_to_grandmother + rolls_to_neighbor + needed_rolls = total_rolls) :
  rolls_to_uncle = 10 :=
by {
  sorry
}

end rolls_sold_to_uncle_l59_59046


namespace additional_grassy_ground_l59_59773

theorem additional_grassy_ground (r1 r2 : ℝ) (h1: r1 = 16) (h2: r2 = 23) :
  (π * r2 ^ 2) - (π * r1 ^ 2) = 273 * π :=
by
  sorry

end additional_grassy_ground_l59_59773


namespace min_value_a4b3c2_l59_59865

theorem min_value_a4b3c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : 1/a + 1/b + 1/c = 9) : (∀ a b c : ℝ, a^4 * b^3 * c^2 ≥ 1/(9^9)) :=
by
  sorry

end min_value_a4b3c2_l59_59865


namespace scientific_notation_proof_l59_59374

-- Given number is 657,000
def number : ℕ := 657000

-- Scientific notation of the given number
def scientific_notation (n : ℕ) : Prop :=
    n = 657000 ∧ (6.57 : ℝ) * (10 : ℝ)^5 = 657000

theorem scientific_notation_proof : scientific_notation number :=
by 
  sorry

end scientific_notation_proof_l59_59374


namespace minimize_travel_time_l59_59969

-- Definitions and conditions
def grid_size : ℕ := 7
def mid_point : ℕ := (grid_size + 1) / 2
def is_meeting_point (p : ℕ × ℕ) : Prop := 
  p = (mid_point, mid_point)

-- Main theorem statement to be proven
theorem minimize_travel_time : 
  ∃ (p : ℕ × ℕ), is_meeting_point p ∧
  (∀ (q : ℕ × ℕ), is_meeting_point q → p = q) :=
sorry

end minimize_travel_time_l59_59969


namespace faye_homework_problems_left_l59_59095

-- Defining the problem conditions
def M : ℕ := 46
def S : ℕ := 9
def A : ℕ := 40

-- The statement to prove
theorem faye_homework_problems_left : M + S - A = 15 := by
  sorry

end faye_homework_problems_left_l59_59095


namespace trip_is_400_miles_l59_59180

def fuel_per_mile_empty_plane := 20
def fuel_increase_per_person := 3
def fuel_increase_per_bag := 2
def number_of_passengers := 30
def number_of_crew := 5
def bags_per_person := 2
def total_fuel_needed := 106000

def fuel_consumption_per_mile :=
  fuel_per_mile_empty_plane +
  (number_of_passengers + number_of_crew) * fuel_increase_per_person +
  (number_of_passengers + number_of_crew) * bags_per_person * fuel_increase_per_bag

def trip_length := total_fuel_needed / fuel_consumption_per_mile

theorem trip_is_400_miles : trip_length = 400 := 
by sorry

end trip_is_400_miles_l59_59180


namespace owen_profit_l59_59078

theorem owen_profit
  (num_boxes : ℕ)
  (cost_per_box : ℕ)
  (pieces_per_box : ℕ)
  (sold_boxes : ℕ)
  (price_per_25_pieces : ℕ)
  (remaining_pieces : ℕ)
  (price_per_10_pieces : ℕ) :
  num_boxes = 12 →
  cost_per_box = 9 →
  pieces_per_box = 50 →
  sold_boxes = 6 →
  price_per_25_pieces = 5 →
  remaining_pieces = 300 →
  price_per_10_pieces = 3 →
  sold_boxes * 2 * price_per_25_pieces + (remaining_pieces / 10) * price_per_10_pieces - num_boxes * cost_per_box = 42 :=
by
  intros h_num h_cost h_pieces h_sold h_price_25 h_remain h_price_10
  sorry

end owen_profit_l59_59078


namespace apples_in_blue_basket_l59_59190

-- Define the number of bananas in the blue basket
def bananas := 12

-- Define the total number of fruits in the blue basket
def totalFruits := 20

-- Define the number of apples as total fruits minus bananas
def apples := totalFruits - bananas

-- Prove that the number of apples in the blue basket is 8
theorem apples_in_blue_basket : apples = 8 := by
  sorry

end apples_in_blue_basket_l59_59190


namespace valid_triangle_side_l59_59167

theorem valid_triangle_side (x : ℝ) (h1 : 2 + x > 6) (h2 : 2 + 6 > x) (h3 : x + 6 > 2) : x = 6 :=
by
  sorry

end valid_triangle_side_l59_59167


namespace tan_neg_two_sin_cos_sum_l59_59453

theorem tan_neg_two_sin_cos_sum (θ : ℝ) (h : Real.tan θ = -2) : 
  Real.sin (2 * θ) + Real.cos (2 * θ) = -7 / 5 :=
by
  sorry

end tan_neg_two_sin_cos_sum_l59_59453


namespace pq_square_sum_l59_59435

theorem pq_square_sum (p q : ℝ) (h1 : p * q = 9) (h2 : p + q = 6) : p^2 + q^2 = 18 := 
by
  sorry

end pq_square_sum_l59_59435


namespace factorize_expression_l59_59503

theorem factorize_expression (m n : ℝ) :
  2 * m^3 * n - 32 * m * n = 2 * m * n * (m + 4) * (m - 4) :=
by
  sorry

end factorize_expression_l59_59503


namespace canoe_vs_kayak_l59_59280

theorem canoe_vs_kayak (
  C K : ℕ 
) (h1 : 14 * C + 15 * K = 288) 
  (h2 : C = (3 * K) / 2) : 
  C - K = 4 := 
sorry

end canoe_vs_kayak_l59_59280


namespace smallest_positive_integer_linear_combination_l59_59454

theorem smallest_positive_integer_linear_combination : ∃ m n : ℤ, 3003 * m + 55555 * n = 1 :=
by
  sorry

end smallest_positive_integer_linear_combination_l59_59454


namespace gcd_example_l59_59636

-- Define the two numbers
def a : ℕ := 102
def b : ℕ := 238

-- Define the GCD of a and b
def gcd_ab : ℕ :=
  Nat.gcd a b

-- The expected result of the GCD
def expected_gcd : ℕ := 34

-- Prove that the GCD of a and b is equal to the expected GCD
theorem gcd_example : gcd_ab = expected_gcd := by
  sorry

end gcd_example_l59_59636


namespace students_chose_apples_l59_59346

theorem students_chose_apples (total students choosing_bananas : ℕ) (h1 : students_choosing_bananas = 168) 
  (h2 : 3 * total = 4 * students_choosing_bananas) : (total / 4) = 56 :=
  by
  sorry

end students_chose_apples_l59_59346


namespace largest_consecutive_odd_numbers_l59_59719

theorem largest_consecutive_odd_numbers (x : ℤ)
  (h : (x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) : 
  x + 6 = 27 :=
  sorry

end largest_consecutive_odd_numbers_l59_59719


namespace regular_polygon_perimeter_is_28_l59_59952

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l59_59952


namespace sum_not_complete_residue_system_l59_59085

theorem sum_not_complete_residue_system {n : ℕ} (hn_even : Even n)
    (a b : Fin n → ℕ) (ha : ∀ k, a k < n) (hb : ∀ k, b k < n) 
    (h_complete_a : ∀ x : Fin n, ∃ k : Fin n, a k = x) 
    (h_complete_b : ∀ y : Fin n, ∃ k : Fin n, b k = y) :
    ¬ (∀ z : Fin n, ∃ k : Fin n, ∃ l : Fin n, z = (a k + b l) % n) :=
by
  sorry

end sum_not_complete_residue_system_l59_59085


namespace expected_rolls_in_non_leap_year_l59_59098

-- Define the conditions and the expected value
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def stops_rolling (n : ℕ) : Prop := is_prime n ∨ is_multiple_of_4 n

def expected_rolls_one_day : ℚ := 6 / 7

def non_leap_year_days : ℕ := 365

def expected_rolls_one_year := expected_rolls_one_day * non_leap_year_days

theorem expected_rolls_in_non_leap_year : expected_rolls_one_year = 314 :=
by
  -- Verification of the mathematical model
  sorry

end expected_rolls_in_non_leap_year_l59_59098


namespace find_other_number_l59_59105

theorem find_other_number (A B : ℕ) (LCM HCF : ℕ) (h1 : LCM = 2310) (h2 : HCF = 83) (h3 : A = 210) (h4 : LCM * HCF = A * B) : B = 913 :=
by
  sorry

end find_other_number_l59_59105


namespace length_chord_AB_l59_59839

-- Given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- Prove the length of the chord AB
theorem length_chord_AB : 
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧ A ≠ B) →
  (∃ (length : ℝ), length = 2*Real.sqrt 2) :=
by
  sorry

end length_chord_AB_l59_59839


namespace mary_brought_stickers_l59_59494

theorem mary_brought_stickers (friends_stickers : Nat) (other_stickers : Nat) (left_stickers : Nat) 
                              (total_students : Nat) (num_friends : Nat) (stickers_per_friend : Nat) 
                              (stickers_per_other_student : Nat) :
  friends_stickers = num_friends * stickers_per_friend →
  left_stickers = 8 →
  total_students = 17 →
  num_friends = 5 →
  stickers_per_friend = 4 →
  stickers_per_other_student = 2 →
  other_stickers = (total_students - 1 - num_friends) * stickers_per_other_student →
  (friends_stickers + other_stickers + left_stickers) = 50 :=
by
  intros
  sorry

end mary_brought_stickers_l59_59494


namespace correct_option_for_sentence_completion_l59_59829

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

end correct_option_for_sentence_completion_l59_59829


namespace geometric_sequence_a5_l59_59194

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 * a 5 = 16) (h2 : a 4 = 8) (h3 : ∀ n, a n > 0) : a 5 = 16 := 
by
  sorry

end geometric_sequence_a5_l59_59194


namespace radar_coverage_correct_l59_59899

noncomputable def radar_coverage (r : ℝ) (width : ℝ) : ℝ × ℝ :=
  let θ := Real.pi / 7
  let distance := 40 / Real.sin θ
  let area := 1440 * Real.pi / Real.tan θ
  (distance, area)

theorem radar_coverage_correct : radar_coverage 41 18 = 
  (40 / Real.sin (Real.pi / 7), 1440 * Real.pi / Real.tan (Real.pi / 7)) :=
by
  sorry

end radar_coverage_correct_l59_59899


namespace max_value_of_reciprocals_l59_59068

noncomputable def quadratic (x t q : ℝ) : ℝ := x^2 - t * x + q

theorem max_value_of_reciprocals (α β t q : ℝ) (h1 : α + β = α^2 + β^2)
                                               (h2 : α + β = α^3 + β^3)
                                               (h3 : ∀ n, 1 ≤ n ∧ n ≤ 2010 → α^n + β^n = α + β)
                                               (h4 : α * β = q)
                                               (h5 : α + β = t) :
  ∃ (α β : ℝ), (1 / α^2012 + 1 / β^2012) = 2 := 
sorry

end max_value_of_reciprocals_l59_59068


namespace paint_left_l59_59919

-- Define the conditions
def total_paint_needed : ℕ := 333
def paint_needed_to_buy : ℕ := 176

-- State the theorem
theorem paint_left : total_paint_needed - paint_needed_to_buy = 157 := 
by 
  sorry

end paint_left_l59_59919


namespace speed_of_boat_in_still_water_l59_59947

variable (V_b V_s t_up t_down : ℝ)

theorem speed_of_boat_in_still_water (h1 : t_up = 2 * t_down)
  (h2 : V_s = 18) 
  (h3 : ∀ d : ℝ, d = (V_b - V_s) * t_up ∧ d = (V_b + V_s) * t_down) : V_b = 54 :=
sorry

end speed_of_boat_in_still_water_l59_59947


namespace expansion_coeff_l59_59688

theorem expansion_coeff (a b : ℝ) (x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x^2 + a^5 * x^5) :
  b = 40 :=
sorry

end expansion_coeff_l59_59688


namespace grayson_vs_rudy_distance_l59_59217

-- Definitions based on the conditions
def grayson_first_part_distance : Real := 25 * 1
def grayson_second_part_distance : Real := 20 * 0.5
def total_grayson_distance : Real := grayson_first_part_distance + grayson_second_part_distance
def rudy_distance : Real := 10 * 3

-- Theorem stating the problem to be proved
theorem grayson_vs_rudy_distance : total_grayson_distance - rudy_distance = 5 := by
  -- Proof would go here
  sorry

end grayson_vs_rudy_distance_l59_59217


namespace black_lambs_count_l59_59811

-- Definitions based on the conditions given
def total_lambs : ℕ := 6048
def white_lambs : ℕ := 193

-- Theorem statement
theorem black_lambs_count : total_lambs - white_lambs = 5855 :=
by 
  -- the proof would be provided here
  sorry

end black_lambs_count_l59_59811


namespace find_number_l59_59826

theorem find_number (N x : ℕ) (h1 : 3 * x = (N - x) + 26) (h2 : x = 22) : N = 62 :=
by
  sorry

end find_number_l59_59826


namespace find_sum_of_a_and_b_l59_59466

variable (a b w y z S : ℕ)

-- Conditions based on problem statement
axiom condition1 : 19 + w + 23 = S
axiom condition2 : 22 + y + a = S
axiom condition3 : b + 18 + z = S
axiom condition4 : 19 + 22 + b = S
axiom condition5 : w + y + 18 = S
axiom condition6 : 23 + a + z = S
axiom condition7 : 19 + y + z = S
axiom condition8 : 23 + y + b = S

theorem find_sum_of_a_and_b : a + b = 23 :=
by
  sorry  -- To be provided with the actual proof later

end find_sum_of_a_and_b_l59_59466


namespace original_price_of_sweater_l59_59328

theorem original_price_of_sweater (sold_price : ℝ) (discount : ℝ) (original_price : ℝ) 
    (h1 : sold_price = 120) (h2 : discount = 0.40) (h3: (1 - discount) * original_price = sold_price) : 
    original_price = 200 := by 
  sorry

end original_price_of_sweater_l59_59328


namespace jasmine_coffee_beans_purchase_l59_59538

theorem jasmine_coffee_beans_purchase (x : ℝ) (coffee_cost per_pound milk_cost per_gallon total_cost : ℝ)
  (h1 : coffee_cost = 2.50)
  (h2 : milk_cost = 3.50)
  (h3 : total_cost = 17)
  (h4 : milk_purchased = 2)
  (h_equation : coffee_cost * x + milk_cost * milk_purchased = total_cost) :
  x = 4 :=
by
  sorry

end jasmine_coffee_beans_purchase_l59_59538


namespace hazel_drank_one_cup_l59_59003

theorem hazel_drank_one_cup (total_cups made_to_crew bike_sold friends_given remaining_cups : ℕ) 
  (H1 : total_cups = 56)
  (H2 : made_to_crew = total_cups / 2)
  (H3 : bike_sold = 18)
  (H4 : friends_given = bike_sold / 2)
  (H5 : remaining_cups = total_cups - (made_to_crew + bike_sold + friends_given)) :
  remaining_cups = 1 := 
sorry

end hazel_drank_one_cup_l59_59003


namespace probability_x_plus_2y_lt_6_l59_59678

noncomputable def prob_x_plus_2y_lt_6 : ℚ :=
  let rect_area : ℚ := (4 : ℚ) * 3
  let quad_area : ℚ := (4 : ℚ) * 1 + (1 / 2 : ℚ) * 4 * 2
  quad_area / rect_area

theorem probability_x_plus_2y_lt_6 :
  prob_x_plus_2y_lt_6 = 2 / 3 :=
by
  sorry

end probability_x_plus_2y_lt_6_l59_59678


namespace num_pairs_mod_eq_l59_59108

theorem num_pairs_mod_eq (k : ℕ) (h : k ≥ 7) :
  ∃ n : ℕ, n = 2^(k+5) ∧
  (∀ x y : ℕ, 0 ≤ x ∧ x < 2^k ∧ 0 ≤ y ∧ y < 2^k → (73^(73^x) ≡ 9^(9^y) [MOD 2^k]) → true) :=
sorry

end num_pairs_mod_eq_l59_59108


namespace base_b_digits_l59_59407

theorem base_b_digits (b : ℕ) : b^4 ≤ 500 ∧ 500 < b^5 → b = 4 := by
  intro h
  sorry

end base_b_digits_l59_59407


namespace number_of_sarees_l59_59274

-- Define variables representing the prices of one saree and one shirt
variables (X S T : ℕ)

-- Define the conditions 
def condition1 := X * S + 4 * T = 1600
def condition2 := S + 6 * T = 1600
def condition3 := 12 * T = 2400

-- The proof problem (statement only, without proof)
theorem number_of_sarees (X S T : ℕ) (h1 : condition1 X S T) (h2 : condition2 S T) (h3 : condition3 T) : X = 2 := by
  sorry

end number_of_sarees_l59_59274


namespace proof1_proof2_proof3_l59_59366

variables (x m n : ℝ)

theorem proof1 (x : ℝ) : (-3 * x - 5) * (5 - 3 * x) = 9 * x^2 - 25 :=
sorry

theorem proof2 (x : ℝ) : (-3 * x - 5) * (5 + 3 * x) = - (3 * x + 5) ^ 2 :=
sorry

theorem proof3 (m n : ℝ) : (2 * m - 3 * n + 1) * (2 * m + 1 + 3 * n) = (2 * m + 1) ^ 2 - (3 * n) ^ 2 :=
sorry

end proof1_proof2_proof3_l59_59366


namespace simple_interest_amount_l59_59983

noncomputable def simple_interest (P r t : ℝ) : ℝ := (P * r * t) / 100
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r / 100)^t - P

theorem simple_interest_amount:
  ∀ (P : ℝ), compound_interest P 5 2 = 51.25 → simple_interest P 5 2 = 50 :=
by
  intros P h
  -- this is where the proof would go
  sorry

end simple_interest_amount_l59_59983


namespace sequence_even_numbers_sequence_odd_numbers_sequence_square_numbers_sequence_arithmetic_progression_l59_59061

-- Problem 1: Prove the general formula for the sequence of all positive even numbers
theorem sequence_even_numbers (n : ℕ) : ∃ a_n, a_n = 2 * n := by 
  sorry

-- Problem 2: Prove the general formula for the sequence of all positive odd numbers
theorem sequence_odd_numbers (n : ℕ) : ∃ b_n, b_n = 2 * n - 1 := by 
  sorry

-- Problem 3: Prove the general formula for the sequence 1, 4, 9, 16, ...
theorem sequence_square_numbers (n : ℕ) : ∃ a_n, a_n = n^2 := by
  sorry

-- Problem 4: Prove the general formula for the sequence -4, -1, 2, 5, ...
theorem sequence_arithmetic_progression (n : ℕ) : ∃ a_n, a_n = 3 * n - 7 := by
  sorry

end sequence_even_numbers_sequence_odd_numbers_sequence_square_numbers_sequence_arithmetic_progression_l59_59061


namespace charity_dinner_cost_l59_59659

def cost_of_rice_per_plate : ℝ := 0.10
def cost_of_chicken_per_plate : ℝ := 0.40
def number_of_plates : ℕ := 100

theorem charity_dinner_cost : 
  cost_of_rice_per_plate + cost_of_chicken_per_plate * number_of_plates = 50 :=
by
  sorry

end charity_dinner_cost_l59_59659


namespace number_of_terms_in_arithmetic_sequence_l59_59572

theorem number_of_terms_in_arithmetic_sequence : 
  ∀ (a d l : ℕ), a = 20 → d = 5 → l = 150 → 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 27 :=
by
  intros a d l ha hd hl
  use 27
  rw [ha, hd, hl]
  sorry

end number_of_terms_in_arithmetic_sequence_l59_59572


namespace math_scores_population_l59_59099

/-- 
   Suppose there are 50,000 students who took the high school entrance exam.
   The education department randomly selected 2,000 students' math scores 
   for statistical analysis. Prove that the math scores of the 50,000 students 
   are the population.
-/
theorem math_scores_population (students : ℕ) (selected : ℕ) 
    (students_eq : students = 50000) (selected_eq : selected = 2000) : 
    true :=
by
  sorry

end math_scores_population_l59_59099


namespace angle_between_line_and_plane_l59_59568

noncomputable def vector_angle (m n : ℝ) : ℝ := 120

theorem angle_between_line_and_plane (m n : ℝ) : 
  (vector_angle m n = 120) → (90 - (vector_angle m n - 90) = 30) :=
by sorry

end angle_between_line_and_plane_l59_59568


namespace geometric_sequence_sum_l59_59792

theorem geometric_sequence_sum (a : ℕ → ℤ)
  (h1 : a 0 = 1)
  (h_q : ∀ n, a (n + 1) = a n * -2) :
  a 0 + |a 1| + a 2 + |a 3| = 15 := by
  sorry

end geometric_sequence_sum_l59_59792


namespace significant_digits_of_square_side_l59_59304

theorem significant_digits_of_square_side (A : ℝ) (s : ℝ) (h : A = 0.6400) (hs : s^2 = A) : 
  s = 0.8000 :=
sorry

end significant_digits_of_square_side_l59_59304


namespace sam_investment_time_l59_59393

theorem sam_investment_time (P r : ℝ) (n A t : ℕ) (hP : P = 8000) (hr : r = 0.10) (hn : n = 2) (hA : A = 8820) :
  A = P * (1 + r / n) ^ (n * t) → t = 1 :=
by
  sorry

end sam_investment_time_l59_59393


namespace rhombus_side_length_l59_59141

theorem rhombus_side_length (s : ℝ) (h : 4 * s = 32) : s = 8 :=
by
  sorry

end rhombus_side_length_l59_59141


namespace percentage_boy_scouts_l59_59065

theorem percentage_boy_scouts (S B G : ℝ) (h1 : B + G = S)
  (h2 : 0.60 * S = 0.50 * B + 0.6818 * G) : (B / S) * 100 = 45 := by
  sorry

end percentage_boy_scouts_l59_59065


namespace remainder_of_product_l59_59352

theorem remainder_of_product (a b n : ℕ) (h1 : a = 2431) (h2 : b = 1587) (h3 : n = 800) : 
  (a * b) % n = 397 := 
by
  sorry

end remainder_of_product_l59_59352


namespace nuts_consumed_range_l59_59733

def diet_day_nuts : Nat := 1
def normal_day_nuts : Nat := diet_day_nuts + 2

def total_nuts_consumed (start_with_diet_day : Bool) : Nat :=
  if start_with_diet_day then
    (10 * diet_day_nuts) + (9 * normal_day_nuts)
  else
    (10 * normal_day_nuts) + (9 * diet_day_nuts)

def min_nuts_consumed : Nat :=
  Nat.min (total_nuts_consumed true) (total_nuts_consumed false)

def max_nuts_consumed : Nat :=
  Nat.max (total_nuts_consumed true) (total_nuts_consumed false)

theorem nuts_consumed_range :
  min_nuts_consumed = 37 ∧ max_nuts_consumed = 39 := by
  sorry

end nuts_consumed_range_l59_59733


namespace total_tickets_used_l59_59419

theorem total_tickets_used :
  let shooting_game_cost := 5
  let carousel_cost := 3
  let jen_games := 2
  let russel_rides := 3
  let jen_total := shooting_game_cost * jen_games
  let russel_total := carousel_cost * russel_rides
  jen_total + russel_total = 19 :=
by
  -- proof goes here
  sorry

end total_tickets_used_l59_59419


namespace sequence_general_term_l59_59977

theorem sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (hS : ∀ n, S n = 2^n - 1) :
  ∀ n, a n = S n - S (n-1) :=
by
  -- The proof will be filled in here
  sorry

end sequence_general_term_l59_59977


namespace tan_alpha_value_l59_59282

noncomputable def f (x : ℝ) := 3 * Real.sin x + 4 * Real.cos x

theorem tan_alpha_value (α : ℝ) (h : ∀ x : ℝ, f x ≥ f α) : Real.tan α = 3 / 4 := 
sorry

end tan_alpha_value_l59_59282


namespace quadratic_roots_bounds_l59_59072

theorem quadratic_roots_bounds (m x1 x2 : ℝ) (h : m < 0)
  (hx : x1 < x2) 
  (hr : ∀ x, x^2 - x - 6 = m → x = x1 ∨ x = x2) :
  -2 < x1 ∧ x2 < 3 :=
by
  sorry

end quadratic_roots_bounds_l59_59072


namespace min_value_of_b1_plus_b2_l59_59779

theorem min_value_of_b1_plus_b2 (b : ℕ → ℕ) (h1 : ∀ n ≥ 1, b (n + 2) = (b n + 4030) / (1 + b (n + 1)))
  (h2 : ∀ n, b n > 0) : ∃ b1 b2, b1 * b2 = 4030 ∧ b1 + b2 = 127 :=
by {
  sorry
}

end min_value_of_b1_plus_b2_l59_59779


namespace div_sqrt_81_by_3_is_3_l59_59486

-- Definitions based on conditions
def sqrt_81 := Nat.sqrt 81
def number_3 := 3

-- Problem statement
theorem div_sqrt_81_by_3_is_3 : sqrt_81 / number_3 = 3 := by
  sorry

end div_sqrt_81_by_3_is_3_l59_59486


namespace maximum_cards_l59_59499

def total_budget : ℝ := 15
def card_cost : ℝ := 1.25
def transaction_fee : ℝ := 2
def desired_savings : ℝ := 3

theorem maximum_cards : ∃ n : ℕ, n ≤ 8 ∧ (card_cost * (n : ℝ) + transaction_fee ≤ total_budget - desired_savings) :=
by sorry

end maximum_cards_l59_59499


namespace gcd_of_repeated_six_digit_integers_l59_59734

-- Given condition
def is_repeated_six_digit_integer (n : ℕ) : Prop :=
  100 ≤ n / 1000 ∧ n / 1000 < 1000 ∧ n = 1001 * (n / 1000)

-- Theorem to prove
theorem gcd_of_repeated_six_digit_integers :
  ∀ n : ℕ, is_repeated_six_digit_integer n → gcd n 1001 = 1001 :=
by sorry

end gcd_of_repeated_six_digit_integers_l59_59734


namespace total_students_in_Lansing_l59_59306

theorem total_students_in_Lansing:
  (number_of_schools : Nat) → 
  (students_per_school : Nat) → 
  (total_students : Nat) →
  number_of_schools = 25 → 
  students_per_school = 247 → 
  total_students = number_of_schools * students_per_school → 
  total_students = 6175 :=
by
  intros number_of_schools students_per_school total_students h_schools h_students h_total
  rw [h_schools, h_students] at h_total
  exact h_total

end total_students_in_Lansing_l59_59306


namespace oil_amount_to_add_l59_59469

variable (a b : ℝ)
variable (h1 : a = 0.16666666666666666)
variable (h2 : b = 0.8333333333333334)

theorem oil_amount_to_add (a b : ℝ) (h1 : a = 0.16666666666666666) (h2 : b = 0.8333333333333334) : 
  b - a = 0.6666666666666667 := by
  rw [h1, h2]
  norm_num
  sorry

end oil_amount_to_add_l59_59469


namespace length_side_AB_is_4_l59_59286

-- Defining a triangle ABC with area 6
variables {A B C K L Q : Type*}
variables {side_AB : Float} {ratio_K : Float} {ratio_L : Float} {dist_Q : Float}
variables (area_ABC : ℝ := 6) (ratio_AK_BK : ℝ := 2 / 3) (ratio_AL_LC : ℝ := 5 / 3)
variables (dist_Q_to_AB : ℝ := 1.5)

theorem length_side_AB_is_4 : 
  side_AB = 4 → 
  (area_ABC = 6 ∧ ratio_AK_BK = 2 / 3 ∧ ratio_AL_LC = 5 / 3 ∧ dist_Q_to_AB = 1.5) :=
by
  sorry

end length_side_AB_is_4_l59_59286


namespace total_profit_l59_59692

theorem total_profit (P Q R : ℝ) (profit : ℝ) 
  (h1 : 4 * P = 6 * Q) 
  (h2 : 6 * Q = 10 * R) 
  (h3 : R = 840 / 6) : 
  profit = 4340 :=
sorry

end total_profit_l59_59692


namespace allie_carl_product_points_l59_59884

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 2, 5]
def carl_rolls : List ℕ := [1, 4, 3, 6, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.foldr (λ x acc => g x + acc) 0

theorem allie_carl_product_points : (total_points allie_rolls) * (total_points carl_rolls) = 594 :=
  sorry

end allie_carl_product_points_l59_59884


namespace faye_books_l59_59771

theorem faye_books (initial_books given_away final_books books_bought: ℕ) 
  (h1 : initial_books = 34) 
  (h2 : given_away = 3) 
  (h3 : final_books = 79) 
  (h4 : final_books = initial_books - given_away + books_bought) : 
  books_bought = 48 := 
by 
  sorry

end faye_books_l59_59771


namespace set_intersection_complement_l59_59647

def setA : Set ℝ := {-2, -1, 0, 1, 2}
def setB : Set ℝ := { x : ℝ | x^2 + 2*x < 0 }
def complementB : Set ℝ := { x : ℝ | x ≥ 0 ∨ x ≤ -2 }

theorem set_intersection_complement :
  setA ∩ complementB = {-2, 0, 1, 2} :=
by
  sorry

end set_intersection_complement_l59_59647
