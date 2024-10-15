import Mathlib

namespace NUMINAMATH_GPT_dave_age_l941_94156

theorem dave_age (C D E : ℝ) (h1 : C = 4 * D) (h2 : E = D + 5) (h3 : C = E) : D = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_dave_age_l941_94156


namespace NUMINAMATH_GPT_diameter_of_outer_circle_l941_94158

theorem diameter_of_outer_circle (D d : ℝ) 
  (h1 : d = 24) 
  (h2 : π * (D / 2) ^ 2 - π * (d / 2) ^ 2 = 0.36 * π * (D / 2) ^ 2) : D = 30 := 
by 
  sorry

end NUMINAMATH_GPT_diameter_of_outer_circle_l941_94158


namespace NUMINAMATH_GPT_intersection_of_S_and_complement_of_T_in_U_l941_94105

def U : Set ℕ := { x | 0 ≤ x ∧ x ≤ 8 }
def S : Set ℕ := { 1, 2, 4, 5 }
def T : Set ℕ := { 3, 5, 7 }
def C_U_T : Set ℕ := { x | x ∈ U ∧ x ∉ T }

theorem intersection_of_S_and_complement_of_T_in_U :
  S ∩ C_U_T = { 1, 2, 4 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_S_and_complement_of_T_in_U_l941_94105


namespace NUMINAMATH_GPT_men_women_arrangement_l941_94124

theorem men_women_arrangement :
  let men := 2
  let women := 4
  let slots := 5
  (Nat.choose slots women) * women.factorial * men.factorial = 240 :=
by
  sorry

end NUMINAMATH_GPT_men_women_arrangement_l941_94124


namespace NUMINAMATH_GPT_area_ratio_trapezoid_l941_94194

/--
In trapezoid PQRS, the lengths of the bases PQ and RS are 10 and 21 respectively.
The legs of the trapezoid are extended beyond P and Q to meet at point T.
Prove that the ratio of the area of triangle TPQ to the area of trapezoid PQRS is 100/341.
-/
theorem area_ratio_trapezoid (PQ RS TPQ PQRS : ℝ) (hPQ : PQ = 10) (hRS : RS = 21) :
  let area_TPQ := TPQ
  let area_PQRS := PQRS
  area_TPQ / area_PQRS = 100 / 341 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_trapezoid_l941_94194


namespace NUMINAMATH_GPT_jason_cards_l941_94157

theorem jason_cards :
  (initial_cards - bought_cards = remaining_cards) →
  initial_cards = 676 →
  bought_cards = 224 →
  remaining_cards = 452 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_jason_cards_l941_94157


namespace NUMINAMATH_GPT_ratio_triangle_square_l941_94112

noncomputable def square_area (s : ℝ) : ℝ := s * s

noncomputable def triangle_PTU_area (s : ℝ) : ℝ := 1 / 2 * (s / 2) * (s / 2)

theorem ratio_triangle_square (s : ℝ) (h : s > 0) : 
  triangle_PTU_area s / square_area s = 1 / 8 := 
sorry

end NUMINAMATH_GPT_ratio_triangle_square_l941_94112


namespace NUMINAMATH_GPT_x_intercept_of_perpendicular_line_is_16_over_3_l941_94186

theorem x_intercept_of_perpendicular_line_is_16_over_3 :
  (∃ x : ℚ, (∃ y : ℚ, (4 * x - 3 * y = 12))
    ∧ (∃ x y : ℚ, (y = - (3 / 4) * x + 4 ∧ y = 0) ∧ x = 16 / 3)) :=
by {
  sorry
}

end NUMINAMATH_GPT_x_intercept_of_perpendicular_line_is_16_over_3_l941_94186


namespace NUMINAMATH_GPT_volume_of_rectangular_solid_l941_94159

theorem volume_of_rectangular_solid (x y z : ℝ) 
  (h1 : x * y = 18) 
  (h2 : y * z = 15) 
  (h3 : z * x = 10) : 
  x * y * z = 30 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_volume_of_rectangular_solid_l941_94159


namespace NUMINAMATH_GPT_total_dots_not_visible_l941_94130

theorem total_dots_not_visible
    (num_dice : ℕ)
    (dots_per_die : ℕ)
    (visible_faces : ℕ → ℕ)
    (visible_faces_count : ℕ)
    (total_dots : ℕ)
    (dots_visible : ℕ) :
    num_dice = 4 →
    dots_per_die = 21 →
    visible_faces 0 = 1 →
    visible_faces 1 = 2 →
    visible_faces 2 = 2 →
    visible_faces 3 = 3 →
    visible_faces 4 = 4 →
    visible_faces 5 = 5 →
    visible_faces 6 = 6 →
    visible_faces 7 = 6 →
    visible_faces_count = 8 →
    total_dots = num_dice * dots_per_die →
    dots_visible = visible_faces 0 + visible_faces 1 + visible_faces 2 + visible_faces 3 + visible_faces 4 + visible_faces 5 + visible_faces 6 + visible_faces 7 →
    total_dots - dots_visible = 55 := by
  sorry

end NUMINAMATH_GPT_total_dots_not_visible_l941_94130


namespace NUMINAMATH_GPT_inequality_true_l941_94154

theorem inequality_true (a b : ℝ) (h : a^2 + b^2 > 1) : |a| + |b| > 1 :=
sorry

end NUMINAMATH_GPT_inequality_true_l941_94154


namespace NUMINAMATH_GPT_derivative_y_l941_94173

noncomputable def y (a α x : ℝ) :=
  (Real.exp (a * x)) * (3 * Real.sin (3 * x) - α * Real.cos (3 * x)) / (a ^ 2 + 9)

theorem derivative_y (a α x : ℝ) :
  (deriv (y a α) x) =
    (Real.exp (a * x)) * ((3 * a + 3 * α) * Real.sin (3 * x) + (9 - a * α) * Real.cos (3 * x)) / (a ^ 2 + 9) := 
sorry

end NUMINAMATH_GPT_derivative_y_l941_94173


namespace NUMINAMATH_GPT_construct_segment_AB_l941_94121

-- Define the two points A and B and assume the distance between them is greater than 1 meter
variables {A B : Point} (dist_AB_gt_1m : Distance A B > 1)

-- Define the ruler length as 10 cm
def ruler_length : ℝ := 0.1

theorem construct_segment_AB 
  (h : dist_AB_gt_1m) 
  (ruler : ℝ := ruler_length) : ∃ (AB : Segment), Distance A B = AB.length ∧ AB.length > 1 :=
sorry

end NUMINAMATH_GPT_construct_segment_AB_l941_94121


namespace NUMINAMATH_GPT_molecular_weight_K3AlC2O4_3_l941_94160

noncomputable def molecularWeightOfCompound : ℝ :=
  let potassium_weight : ℝ := 39.10
  let aluminum_weight  : ℝ := 26.98
  let carbon_weight    : ℝ := 12.01
  let oxygen_weight    : ℝ := 16.00
  let total_potassium_weight : ℝ := 3 * potassium_weight
  let total_aluminum_weight  : ℝ := aluminum_weight
  let total_carbon_weight    : ℝ := 3 * 2 * carbon_weight
  let total_oxygen_weight    : ℝ := 3 * 4 * oxygen_weight
  total_potassium_weight + total_aluminum_weight + total_carbon_weight + total_oxygen_weight

theorem molecular_weight_K3AlC2O4_3 : molecularWeightOfCompound = 408.34 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_K3AlC2O4_3_l941_94160


namespace NUMINAMATH_GPT_equal_area_bisecting_line_slope_l941_94146

theorem equal_area_bisecting_line_slope 
  (circle1_center circle2_center : ℝ × ℝ) 
  (radius : ℝ) 
  (line_point : ℝ × ℝ) 
  (h1 : circle1_center = (20, 100))
  (h2 : circle2_center = (25, 90))
  (h3 : radius = 4)
  (h4 : line_point = (20, 90))
  : ∃ (m : ℝ), |m| = 2 :=
by
  sorry

end NUMINAMATH_GPT_equal_area_bisecting_line_slope_l941_94146


namespace NUMINAMATH_GPT_linearly_dependent_k_l941_94185

theorem linearly_dependent_k (k : ℝ) : 
  (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨1, k⟩ : ℝ × ℝ) = (0, 0)) ↔ k = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_linearly_dependent_k_l941_94185


namespace NUMINAMATH_GPT_loaned_out_books_is_50_l941_94137

-- Define the conditions
def initial_books : ℕ := 75
def end_books : ℕ := 60
def percent_returned : ℝ := 0.70

-- Define the variable to represent the number of books loaned out
noncomputable def loaned_out_books := (15:ℝ) / (1 - percent_returned)

-- The target theorem statement we need to prove
theorem loaned_out_books_is_50 : loaned_out_books = 50 :=
by
  sorry

end NUMINAMATH_GPT_loaned_out_books_is_50_l941_94137


namespace NUMINAMATH_GPT_martha_to_doris_ratio_l941_94199

-- Define the amounts involved
def initial_amount : ℕ := 21
def doris_spent : ℕ := 6
def remaining_after_doris : ℕ := initial_amount - doris_spent
def final_amount : ℕ := 12
def martha_spent : ℕ := remaining_after_doris - final_amount

-- State the theorem about the ratio
theorem martha_to_doris_ratio : martha_spent * 2 = doris_spent :=
by
  -- Detailed proof is skipped
  sorry

end NUMINAMATH_GPT_martha_to_doris_ratio_l941_94199


namespace NUMINAMATH_GPT_horatio_sonnets_count_l941_94131

-- Each sonnet consists of 14 lines
def lines_per_sonnet : ℕ := 14

-- The number of sonnets his lady fair heard
def heard_sonnets : ℕ := 7

-- The total number of unheard lines
def unheard_lines : ℕ := 70

-- Calculate sonnets Horatio wrote by the heard and unheard components
def total_sonnets : ℕ := heard_sonnets + (unheard_lines / lines_per_sonnet)

-- Prove the total number of sonnets horatio wrote
theorem horatio_sonnets_count : total_sonnets = 12 := 
by sorry

end NUMINAMATH_GPT_horatio_sonnets_count_l941_94131


namespace NUMINAMATH_GPT_quarters_total_l941_94118

def initial_quarters : ℕ := 21
def additional_quarters : ℕ := 49
def total_quarters : ℕ := initial_quarters + additional_quarters

theorem quarters_total : total_quarters = 70 := by
  sorry

end NUMINAMATH_GPT_quarters_total_l941_94118


namespace NUMINAMATH_GPT_minimum_value_expression_l941_94145

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ k, k = 729 ∧ ∀ x y z, 0 < x → 0 < y → 0 < z → k ≤ (x^2 + 4*x + 4) * (y^2 + 4*y + 4) * (z^2 + 4*z + 4) / (x * y * z) :=
by 
  use 729
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l941_94145


namespace NUMINAMATH_GPT_remainder_of_82460_div_8_l941_94162

theorem remainder_of_82460_div_8 :
  82460 % 8 = 4 :=
sorry

end NUMINAMATH_GPT_remainder_of_82460_div_8_l941_94162


namespace NUMINAMATH_GPT_simplify_expression_l941_94132

theorem simplify_expression (w : ℕ) : 
  4 * w + 6 * w + 8 * w + 10 * w + 12 * w + 14 * w + 16 = 54 * w + 16 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l941_94132


namespace NUMINAMATH_GPT_final_value_of_x_l941_94155

noncomputable def initial_x : ℝ := 52 * 1.2
noncomputable def decreased_x : ℝ := initial_x * 0.9
noncomputable def final_x : ℝ := decreased_x * 1.15

theorem final_value_of_x : final_x = 64.584 := by
  sorry

end NUMINAMATH_GPT_final_value_of_x_l941_94155


namespace NUMINAMATH_GPT_problem_inequality_I_problem_inequality_II_l941_94197

theorem problem_inequality_I (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1):
  1 / a + 1 / b ≥ 4 := sorry

theorem problem_inequality_II (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1):
  (a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2 := sorry

end NUMINAMATH_GPT_problem_inequality_I_problem_inequality_II_l941_94197


namespace NUMINAMATH_GPT_second_player_wins_12_petals_second_player_wins_11_petals_l941_94184

def daisy_game (n : Nat) : Prop :=
  ∀ (p1_move p2_move : Nat → Nat → Prop), n % 2 = 0 → (∃ k, p1_move n k = false) ∧ (∃ ℓ, p2_move n ℓ = true)

theorem second_player_wins_12_petals : daisy_game 12 := sorry
theorem second_player_wins_11_petals : daisy_game 11 := sorry

end NUMINAMATH_GPT_second_player_wins_12_petals_second_player_wins_11_petals_l941_94184


namespace NUMINAMATH_GPT_find_x_l941_94161

theorem find_x (x y : ℝ) :
  (x / (x - 1) = (y^2 + 3*y + 2) / (y^2 + 3*y - 1)) →
  x = (y^2 + 3*y + 2) / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l941_94161


namespace NUMINAMATH_GPT_reducible_fraction_l941_94100

theorem reducible_fraction (l : ℤ) : ∃ k : ℤ, l = 13 * k + 4 ↔ (∃ d > 1, d ∣ (5 * l + 6) ∧ d ∣ (8 * l + 7)) :=
sorry

end NUMINAMATH_GPT_reducible_fraction_l941_94100


namespace NUMINAMATH_GPT_face_value_of_share_l941_94135

theorem face_value_of_share 
  (F : ℝ)
  (dividend_rate : ℝ := 0.09)
  (desired_return_rate : ℝ := 0.12)
  (market_value : ℝ := 15) 
  (h_eq : (dividend_rate * F) / market_value = desired_return_rate) : 
  F = 20 := 
by 
  sorry

end NUMINAMATH_GPT_face_value_of_share_l941_94135


namespace NUMINAMATH_GPT_radius_of_circle_l941_94119

open Complex

theorem radius_of_circle (z : ℂ) (h : (z + 2)^4 = 16 * z^4) : abs z = 2 / Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_radius_of_circle_l941_94119


namespace NUMINAMATH_GPT_compute_division_l941_94138

theorem compute_division : 0.182 / 0.0021 = 86 + 14 / 21 :=
by
  sorry

end NUMINAMATH_GPT_compute_division_l941_94138


namespace NUMINAMATH_GPT_integer_ratio_value_l941_94139

theorem integer_ratio_value {x y : ℝ} (h1 : 3 < (x^2 - y^2) / (x^2 + y^2)) (h2 : (x^2 - y^2) / (x^2 + y^2) < 4) (h3 : ∃ t : ℤ, x = t * y) : ∃ t : ℤ, t = 2 :=
by
  sorry

end NUMINAMATH_GPT_integer_ratio_value_l941_94139


namespace NUMINAMATH_GPT_find_common_ratio_geometric_l941_94169

variable {α : Type*} [Field α] {a : ℕ → α} {S : ℕ → α} {q : α} (h₁ : a 3 = 2 * S 2 + 1) (h₂ : a 4 = 2 * S 3 + 1)

def common_ratio_geometric : α := 3

theorem find_common_ratio_geometric (ha₃ : a 3 = 2 * S 2 + 1) (ha₄ : a 4 = 2 * S 3 + 1) :
  q = common_ratio_geometric := 
  sorry

end NUMINAMATH_GPT_find_common_ratio_geometric_l941_94169


namespace NUMINAMATH_GPT_john_payment_l941_94110

noncomputable def amount_paid_by_john := (3 * 12) / 2

theorem john_payment : amount_paid_by_john = 18 :=
by
  sorry

end NUMINAMATH_GPT_john_payment_l941_94110


namespace NUMINAMATH_GPT_points_lie_on_hyperbola_l941_94193

noncomputable def point_on_hyperbola (t : ℝ) : Prop :=
  let x := 2 * (Real.exp t + Real.exp (-t))
  let y := 4 * (Real.exp t - Real.exp (-t))
  (x^2 / 16) - (y^2 / 64) = 1

theorem points_lie_on_hyperbola (t : ℝ) : point_on_hyperbola t := 
by
  sorry

end NUMINAMATH_GPT_points_lie_on_hyperbola_l941_94193


namespace NUMINAMATH_GPT_grasshopper_jump_is_31_l941_94181

def frog_jump : ℕ := 35
def total_jump : ℕ := 66
def grasshopper_jump := total_jump - frog_jump

theorem grasshopper_jump_is_31 : grasshopper_jump = 31 := 
by
  unfold grasshopper_jump
  sorry

end NUMINAMATH_GPT_grasshopper_jump_is_31_l941_94181


namespace NUMINAMATH_GPT_solution_to_trig_equation_l941_94191

theorem solution_to_trig_equation (x : ℝ) (k : ℤ) :
  (1 - 2 * Real.sin (x / 2) * Real.cos (x / 2) = 
  (Real.sin (x / 2) - Real.cos (x / 2)) / Real.cos (x / 2)) →
  (Real.sin (x / 2) = Real.cos (x / 2)) →
  (∃ k : ℤ, x = (π / 2) + 2 * π * ↑k) :=
by sorry

end NUMINAMATH_GPT_solution_to_trig_equation_l941_94191


namespace NUMINAMATH_GPT_right_triangle_count_l941_94122

theorem right_triangle_count (a b : ℕ) (h1 : b < 100) (h2 : a^2 + b^2 = (b + 2)^2) : 
∃ n, n = 10 :=
by sorry

end NUMINAMATH_GPT_right_triangle_count_l941_94122


namespace NUMINAMATH_GPT_compare_values_l941_94133

-- Define that f(x) is an even function, periodic and satisfies decrease and increase conditions as given
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

noncomputable def f : ℝ → ℝ := sorry -- the exact definition of f is unknown, so we use sorry for now

-- The conditions of the problem
axiom f_even : is_even_function f
axiom f_period : periodic_function f 2
axiom f_decreasing : decreasing_on_interval f (-1) 0
axiom f_transformation : ∀ x, f (x + 1) = 1 / f x

-- Prove the comparison between a, b, and c under the given conditions
theorem compare_values (a b c : ℝ) (h1 : a = f (Real.log 2 / Real.log 5)) (h2 : b = f (Real.log 4 / Real.log 2)) (h3 : c = f (Real.sqrt 2)) :
  a > c ∧ c > b :=
by
  sorry

end NUMINAMATH_GPT_compare_values_l941_94133


namespace NUMINAMATH_GPT_difference_of_numbers_l941_94165

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : abs (x - y) = 7 :=
sorry

end NUMINAMATH_GPT_difference_of_numbers_l941_94165


namespace NUMINAMATH_GPT_number_of_apples_remaining_l941_94141

def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples_before_giving_away : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℕ := total_apples_before_giving_away / 5
def apples_remaining : ℕ := total_apples_before_giving_away - apples_given_to_son

theorem number_of_apples_remaining : apples_remaining = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_apples_remaining_l941_94141


namespace NUMINAMATH_GPT_only_one_passes_prob_l941_94172

variable (P_A P_B P_C : ℚ)
variable (only_one_passes : ℚ)

def prob_A := 4 / 5 
def prob_B := 3 / 5
def prob_C := 7 / 10

def prob_only_A := prob_A * (1 - prob_B) * (1 - prob_C)
def prob_only_B := (1 - prob_A) * prob_B * (1 - prob_C)
def prob_only_C := (1 - prob_A) * (1 - prob_B) * prob_C

def prob_sum : ℚ := prob_only_A + prob_only_B + prob_only_C

theorem only_one_passes_prob : prob_sum = 47 / 250 := 
by sorry

end NUMINAMATH_GPT_only_one_passes_prob_l941_94172


namespace NUMINAMATH_GPT_scooter_gain_percent_l941_94140

theorem scooter_gain_percent 
  (purchase_price : ℕ) 
  (repair_costs : ℕ) 
  (selling_price : ℕ)
  (h1 : purchase_price = 900)
  (h2 : repair_costs = 300)
  (h3 : selling_price = 1320) : 
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_scooter_gain_percent_l941_94140


namespace NUMINAMATH_GPT_sum_of_c_and_d_l941_94143

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := (x - 3) / (x^2 + c * x + d)

theorem sum_of_c_and_d (c d : ℝ) (h_asymptote1 : (2:ℝ)^2 + c * 2 + d = 0) (h_asymptote2 : (-1:ℝ)^2 - c + d = 0) :
  c + d = -3 :=
by
-- theorem body (proof omitted)
sorry

end NUMINAMATH_GPT_sum_of_c_and_d_l941_94143


namespace NUMINAMATH_GPT_michael_initial_money_l941_94149

theorem michael_initial_money 
  (M B_initial B_left B_spent : ℕ) 
  (h_split : M / 2 = B_initial - B_left + B_spent): 
  (M / 2 + B_left = 17 + 35) → M = 152 :=
by
  sorry

end NUMINAMATH_GPT_michael_initial_money_l941_94149


namespace NUMINAMATH_GPT_sequence_property_l941_94179

def Sn (n : ℕ) (a : ℕ → ℕ) : ℕ := (Finset.range (n + 1)).sum a

theorem sequence_property (a : ℕ → ℕ) (h : ∀ n : ℕ, Sn (n + 1) a = 2 * a n + 1) : a 3 = 2 :=
sorry

end NUMINAMATH_GPT_sequence_property_l941_94179


namespace NUMINAMATH_GPT_range_of_a_l941_94134

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * x

theorem range_of_a (a : ℝ) (h : f (2 * a) < f (a - 1)) : a < -1 :=
by
  -- Steps of the proof would be placed here, but we're skipping them for now
  sorry

end NUMINAMATH_GPT_range_of_a_l941_94134


namespace NUMINAMATH_GPT_Oliver_ferris_wheel_rides_l941_94142

theorem Oliver_ferris_wheel_rides :
  ∃ (F : ℕ), (4 * 7 + F * 7 = 63) ∧ (F = 5) :=
by
  sorry

end NUMINAMATH_GPT_Oliver_ferris_wheel_rides_l941_94142


namespace NUMINAMATH_GPT_prob_A_two_qualified_l941_94164

noncomputable def prob_qualified (p : ℝ) : ℝ := p * p

def qualified_rate : ℝ := 0.8

theorem prob_A_two_qualified : prob_qualified qualified_rate = 0.64 :=
by
  sorry

end NUMINAMATH_GPT_prob_A_two_qualified_l941_94164


namespace NUMINAMATH_GPT_shem_wage_multiple_kem_l941_94111

-- Define the hourly wages and conditions
def kem_hourly_wage : ℝ := 4
def shem_daily_wage : ℝ := 80
def shem_workday_hours : ℝ := 8

-- Prove the multiple of Shem's hourly wage compared to Kem's hourly wage
theorem shem_wage_multiple_kem : (shem_daily_wage / shem_workday_hours) / kem_hourly_wage = 2.5 := by
  sorry

end NUMINAMATH_GPT_shem_wage_multiple_kem_l941_94111


namespace NUMINAMATH_GPT_inequality_holds_l941_94103

theorem inequality_holds 
  (a b c : ℝ) 
  (h1 : a > 0)
  (h2 : b < 0) 
  (h3 : b > c) : 
  (a / (c^2)) > (b / (c^2)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l941_94103


namespace NUMINAMATH_GPT_libraryRoomNumber_l941_94196

-- Define the conditions
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def isPrime (n : ℕ) : Prop := Nat.Prime n
def isEven (n : ℕ) : Prop := n % 2 = 0
def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0
def hasDigit7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

-- Main theorem
theorem libraryRoomNumber (n : ℕ) (h1 : isTwoDigit n)
  (h2 : (isPrime n ∧ isEven n ∧ isDivisibleBy5 n ∧ hasDigit7 n) ↔ false)
  : n % 10 = 0 := 
sorry

end NUMINAMATH_GPT_libraryRoomNumber_l941_94196


namespace NUMINAMATH_GPT_total_water_hold_l941_94127

variables
  (first : ℕ := 100)
  (second : ℕ := 150)
  (third : ℕ := 75)
  (total : ℕ := 325)

theorem total_water_hold :
  first + second + third = total := by
  sorry

end NUMINAMATH_GPT_total_water_hold_l941_94127


namespace NUMINAMATH_GPT_loop_until_correct_l941_94170

-- Define the conditions
def num_iterations := 20

-- Define the loop condition
def loop_condition (i : Nat) : Prop := i > num_iterations

-- Theorem: Proof that the loop should continue until the counter i exceeds 20
theorem loop_until_correct (i : Nat) : loop_condition i := by
  sorry

end NUMINAMATH_GPT_loop_until_correct_l941_94170


namespace NUMINAMATH_GPT_right_triangle_sum_of_legs_l941_94180

theorem right_triangle_sum_of_legs (a b : ℝ) (h₁ : a^2 + b^2 = 2500) (h₂ : (1 / 2) * a * b = 600) : a + b = 70 :=
sorry

end NUMINAMATH_GPT_right_triangle_sum_of_legs_l941_94180


namespace NUMINAMATH_GPT_overall_percentage_support_l941_94174

theorem overall_percentage_support (p_men : ℕ) (p_women : ℕ) (n_men : ℕ) (n_women : ℕ) : 
  (p_men = 55) → (p_women = 80) → (n_men = 200) → (n_women = 800) → 
  (p_men * n_men + p_women * n_women) / (n_men + n_women) = 75 :=
by
  sorry

end NUMINAMATH_GPT_overall_percentage_support_l941_94174


namespace NUMINAMATH_GPT_abs_expression_value_l941_94104

theorem abs_expression_value (x : ℤ) (h : x = -2023) :
  abs (2 * abs (abs x - x) - abs x) - x = 8092 :=
by {
  -- Proof will be provided here
  sorry
}

end NUMINAMATH_GPT_abs_expression_value_l941_94104


namespace NUMINAMATH_GPT_domino_swap_correct_multiplication_l941_94115

theorem domino_swap_correct_multiplication :
  ∃ (a b c d e f : ℕ), 
    a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 3 ∧ e = 12 ∧ f = 3 ∧ 
    a * b = 6 ∧ c * d = 3 ∧ e * f = 36 ∧
    ∃ (x y : ℕ), x * y = 36 := sorry

end NUMINAMATH_GPT_domino_swap_correct_multiplication_l941_94115


namespace NUMINAMATH_GPT_find_f_2_l941_94198

def f (x : ℕ) : ℤ := sorry

axiom func_def : ∀ x : ℕ, f (x + 1) = x^2 - x

theorem find_f_2 : f 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2_l941_94198


namespace NUMINAMATH_GPT_smaller_angle_at_6_30_l941_94102
-- Import the Mathlib library

-- Define the conditions as a structure
structure ClockAngleConditions where
  hours_on_clock : ℕ
  degrees_per_hour : ℕ
  minute_hand_position : ℕ
  hour_hand_position : ℕ

-- Initialize the conditions for 6:30
def conditions : ClockAngleConditions := {
  hours_on_clock := 12,
  degrees_per_hour := 30,
  minute_hand_position := 180,
  hour_hand_position := 195
}

-- Define the theorem to be proven
theorem smaller_angle_at_6_30 (c : ClockAngleConditions) : 
  c.hour_hand_position - c.minute_hand_position = 15 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_smaller_angle_at_6_30_l941_94102


namespace NUMINAMATH_GPT_remainder_of_sum_mod_13_l941_94166

theorem remainder_of_sum_mod_13 (a b c d e : ℕ) 
  (h1: a % 13 = 3) (h2: b % 13 = 5) (h3: c % 13 = 7) (h4: d % 13 = 9) (h5: e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_sum_mod_13_l941_94166


namespace NUMINAMATH_GPT_red_balls_in_box_l941_94129

theorem red_balls_in_box (initial_red_balls added_red_balls : ℕ) (initial_blue_balls : ℕ) 
  (h_initial : initial_red_balls = 5) (h_added : added_red_balls = 2) : 
  initial_red_balls + added_red_balls = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_red_balls_in_box_l941_94129


namespace NUMINAMATH_GPT_solution_set_l941_94187

theorem solution_set (x : ℝ) : 
  (-2 * x ≤ 6) ∧ (x + 1 < 0) ↔ (-3 ≤ x) ∧ (x < -1) := by
  sorry

end NUMINAMATH_GPT_solution_set_l941_94187


namespace NUMINAMATH_GPT_multiple_of_4_and_6_sum_even_l941_94195

theorem multiple_of_4_and_6_sum_even (a b : ℤ) (h₁ : ∃ m : ℤ, a = 4 * m) (h₂ : ∃ n : ℤ, b = 6 * n) : ∃ k : ℤ, (a + b) = 2 * k :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_4_and_6_sum_even_l941_94195


namespace NUMINAMATH_GPT_age_of_15th_student_is_15_l941_94106

-- Define the total number of students
def total_students : Nat := 15

-- Define the average age of all 15 students together
def avg_age_all_students : Nat := 15

-- Define the average age of the first group of 7 students
def avg_age_first_group : Nat := 14

-- Define the average age of the second group of 7 students
def avg_age_second_group : Nat := 16

-- Define the total age based on the average age and number of students
def total_age_all_students : Nat := total_students * avg_age_all_students
def total_age_first_group : Nat := 7 * avg_age_first_group
def total_age_second_group : Nat := 7 * avg_age_second_group

-- Define the age of the 15th student
def age_of_15th_student : Nat := total_age_all_students - (total_age_first_group + total_age_second_group)

-- Theorem: prove that the age of the 15th student is 15 years
theorem age_of_15th_student_is_15 : age_of_15th_student = 15 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_age_of_15th_student_is_15_l941_94106


namespace NUMINAMATH_GPT_fruit_seller_l941_94117

theorem fruit_seller (A P : ℝ) (h1 : A = 700) (h2 : A * (100 - P) / 100 = 420) : P = 40 :=
sorry

end NUMINAMATH_GPT_fruit_seller_l941_94117


namespace NUMINAMATH_GPT_max_marks_l941_94144

theorem max_marks (M : ℝ) (score passing shortfall : ℝ)
  (h_score : score = 212)
  (h_shortfall : shortfall = 44)
  (h_passing : passing = score + shortfall)
  (h_pass_cond : passing = 0.4 * M) :
  M = 640 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l941_94144


namespace NUMINAMATH_GPT_find_a_l941_94190

theorem find_a (a : ℝ) (h : ∃ x, x = 3 ∧ x^2 + a * x + a - 1 = 0) : a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_l941_94190


namespace NUMINAMATH_GPT_cost_price_computer_table_l941_94152

theorem cost_price_computer_table (CP SP : ℝ) (h1 : SP = 1.15 * CP) (h2 : SP = 6400) : CP = 5565.22 :=
by sorry

end NUMINAMATH_GPT_cost_price_computer_table_l941_94152


namespace NUMINAMATH_GPT_find_possible_values_of_y_l941_94177

theorem find_possible_values_of_y (x : ℝ) (h : x^2 + 9 * (3 * x / (x - 3))^2 = 90) :
  y = (x - 3)^3 * (x + 2) / (2 * x - 4) → y = 28 / 3 ∨ y = 169 :=
by
  sorry

end NUMINAMATH_GPT_find_possible_values_of_y_l941_94177


namespace NUMINAMATH_GPT_range_of_a_minus_abs_b_l941_94126

theorem range_of_a_minus_abs_b (a b : ℝ) (h1 : 1 < a ∧ a < 8) (h2 : -4 < b ∧ b < 2) : 
  -3 < a - |b| ∧ a - |b| < 8 :=
sorry

end NUMINAMATH_GPT_range_of_a_minus_abs_b_l941_94126


namespace NUMINAMATH_GPT_other_number_is_36_l941_94176

theorem other_number_is_36 (hcf lcm given_number other_number : ℕ) 
  (hcf_val : hcf = 16) (lcm_val : lcm = 396) (given_number_val : given_number = 176) 
  (relation : hcf * lcm = given_number * other_number) : 
  other_number = 36 := 
by 
  sorry

end NUMINAMATH_GPT_other_number_is_36_l941_94176


namespace NUMINAMATH_GPT_third_motorcyclist_speed_l941_94116

theorem third_motorcyclist_speed 
  (t₁ t₂ : ℝ)
  (x : ℝ)
  (h1 : t₁ - t₂ = 1.25)
  (h2 : 80 * t₁ = x * (t₁ - 0.5))
  (h3 : 60 * t₂ = x * (t₂ - 0.5))
  (h4 : x ≠ 60)
  (h5 : x ≠ 80):
  x = 100 :=
by
  sorry

end NUMINAMATH_GPT_third_motorcyclist_speed_l941_94116


namespace NUMINAMATH_GPT_number_of_cars_in_second_box_is_31_l941_94163

-- Define the total number of toy cars, and the number of toy cars in the first and third boxes
def total_toy_cars : ℕ := 71
def cars_in_first_box : ℕ := 21
def cars_in_third_box : ℕ := 19

-- Define the number of toy cars in the second box
def cars_in_second_box : ℕ := total_toy_cars - cars_in_first_box - cars_in_third_box

-- Theorem stating that the number of toy cars in the second box is 31
theorem number_of_cars_in_second_box_is_31 : cars_in_second_box = 31 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cars_in_second_box_is_31_l941_94163


namespace NUMINAMATH_GPT_isosceles_triangle_angles_l941_94171

theorem isosceles_triangle_angles (a b : ℝ) (h₁ : a = 80 ∨ b = 80) (h₂ : a + b + c = 180) (h_iso : a = b ∨ a = c ∨ b = c) :
  (a = 80 ∧ b = 20 ∧ c = 80)
  ∨ (a = 80 ∧ b = 80 ∧ c = 20)
  ∨ (a = 50 ∧ b = 50 ∧ c = 80) :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_angles_l941_94171


namespace NUMINAMATH_GPT_range_alpha_div_three_l941_94109

open Real

theorem range_alpha_div_three (α : ℝ) (k : ℤ) :
  sin α > 0 → cos α < 0 → sin (α / 3) > cos (α / 3) →
  ∃ k : ℤ,
    (2 * k * π + π / 4 < α / 3 ∧ α / 3 < 2 * k * π + π / 3) ∨
    (2 * k * π + 5 * π / 6 < α / 3 ∧ α / 3 < 2 * k * π + π) :=
by
  intros
  sorry

end NUMINAMATH_GPT_range_alpha_div_three_l941_94109


namespace NUMINAMATH_GPT_intersection_point_exists_l941_94150

theorem intersection_point_exists :
  ∃ t u x y : ℚ,
    (x = 2 + 3 * t) ∧ (y = 3 - 4 * t) ∧
    (x = 4 + 5 * u) ∧ (y = -6 + u) ∧
    (x = 175 / 23) ∧ (y = 19 / 23) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_exists_l941_94150


namespace NUMINAMATH_GPT_find_number_l941_94108

theorem find_number (x : ℝ) 
  (h1 : 0.15 * 40 = 6) 
  (h2 : 6 = 0.25 * x + 2) : 
  x = 16 := 
sorry

end NUMINAMATH_GPT_find_number_l941_94108


namespace NUMINAMATH_GPT_boiling_point_C_l941_94153

-- Water boils at 212 °F
def water_boiling_point_F : ℝ := 212
-- Ice melts at 32 °F
def ice_melting_point_F : ℝ := 32
-- Ice melts at 0 °C
def ice_melting_point_C : ℝ := 0
-- The temperature of a pot of water in °C
def pot_water_temp_C : ℝ := 40
-- The temperature of the pot of water in °F
def pot_water_temp_F : ℝ := 104

-- The boiling point of water in Celsius is 100 °C.
theorem boiling_point_C : water_boiling_point_F = 212 ∧ ice_melting_point_F = 32 ∧ ice_melting_point_C = 0 ∧ pot_water_temp_C = 40 ∧ pot_water_temp_F = 104 → exists bp_C : ℝ, bp_C = 100 :=
by
  sorry

end NUMINAMATH_GPT_boiling_point_C_l941_94153


namespace NUMINAMATH_GPT_f_iterate_result_l941_94101

def f (n : ℕ) : ℕ :=
if n < 3 then n^2 + 1 else 4*n - 3

theorem f_iterate_result : f (f (f 1)) = 17 :=
by
  sorry

end NUMINAMATH_GPT_f_iterate_result_l941_94101


namespace NUMINAMATH_GPT_shopkeeper_total_cards_l941_94167

-- Definition of the number of cards in standard, Uno, and tarot decks.
def std_deck := 52
def uno_deck := 108
def tarot_deck := 78

-- Number of complete decks and additional cards.
def std_decks := 4
def uno_decks := 3
def tarot_decks := 5
def additional_std := 12
def additional_uno := 7
def additional_tarot := 9

-- Calculate the total number of cards.
def total_standard_cards := (std_decks * std_deck) + additional_std
def total_uno_cards := (uno_decks * uno_deck) + additional_uno
def total_tarot_cards := (tarot_decks * tarot_deck) + additional_tarot

def total_cards := total_standard_cards + total_uno_cards + total_tarot_cards

theorem shopkeeper_total_cards : total_cards = 950 := by
  sorry

end NUMINAMATH_GPT_shopkeeper_total_cards_l941_94167


namespace NUMINAMATH_GPT_value_of_expression_l941_94113

variables (m n c d : ℝ)
variables (h1 : m = -n) (h2 : c * d = 1)

theorem value_of_expression : m + n + 3 * c * d - 10 = -7 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l941_94113


namespace NUMINAMATH_GPT_find_digits_l941_94120

def five_digit_subtraction (a b c d e : ℕ) : Prop :=
    let n1 := 10000 * a + 1000 * b + 100 * c + 10 * d + e
    let n2 := 10000 * e + 1000 * d + 100 * c + 10 * b + a
    (n1 - n2) % 10 = 2 ∧ (((n1 - n2) / 10) % 10) = 7 ∧ a > e ∧ a - e = 2 ∧ b - a = 7

theorem find_digits 
    (a b c d e : ℕ) 
    (h : five_digit_subtraction a b c d e) :
    a = 9 ∧ e = 7 :=
by 
    sorry

end NUMINAMATH_GPT_find_digits_l941_94120


namespace NUMINAMATH_GPT_real_and_imaginary_parts_of_z_l941_94107

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i^2 + i

-- State the theorem
theorem real_and_imaginary_parts_of_z :
  z.re = -1 ∧ z.im = 1 :=
by
  -- Provide the proof or placeholder
  sorry

end NUMINAMATH_GPT_real_and_imaginary_parts_of_z_l941_94107


namespace NUMINAMATH_GPT_vanya_meets_mother_opposite_dir_every_4_minutes_l941_94183

-- Define the parameters
def lake_perimeter : ℝ := sorry  -- Length of the lake's perimeter, denoted as l
def mother_time_lap : ℝ := 12    -- Time taken by the mother to complete one lap (in minutes)
def vanya_time_overtake : ℝ := 12 -- Time taken by Vanya to overtake the mother (in minutes)

-- Define speeds
noncomputable def mother_speed : ℝ := lake_perimeter / mother_time_lap
noncomputable def vanya_speed : ℝ := 2 * lake_perimeter / vanya_time_overtake

-- Define their relative speed when moving in opposite directions
noncomputable def relative_speed : ℝ := mother_speed + vanya_speed

-- Prove that the meeting interval is 4 minutes
theorem vanya_meets_mother_opposite_dir_every_4_minutes :
  (lake_perimeter / relative_speed) = 4 := by
  sorry

end NUMINAMATH_GPT_vanya_meets_mother_opposite_dir_every_4_minutes_l941_94183


namespace NUMINAMATH_GPT_possible_value_of_a_eq_neg1_l941_94168

theorem possible_value_of_a_eq_neg1 (a : ℝ) : (-6 * a ^ 2 = 3 * (4 * a + 2)) → (a = -1) :=
by
  intro h
  have H : a^2 + 2*a + 1 = 0
  · sorry
  show a = -1
  · sorry

end NUMINAMATH_GPT_possible_value_of_a_eq_neg1_l941_94168


namespace NUMINAMATH_GPT_cistern_height_l941_94148

theorem cistern_height (l w A : ℝ) (h : ℝ) (hl : l = 8) (hw : w = 6) (hA : 48 + 2 * (l * h) + 2 * (w * h) = 99.8) : h = 1.85 := by
  sorry

end NUMINAMATH_GPT_cistern_height_l941_94148


namespace NUMINAMATH_GPT_fill_tank_time_l941_94182

-- Define the rates at which the pipes fill the tank
noncomputable def rate_A := (1:ℝ)/50
noncomputable def rate_B := (1:ℝ)/75

-- Define the combined rate of both pipes
noncomputable def combined_rate := rate_A + rate_B

-- Define the time to fill the tank at the combined rate
noncomputable def time_to_fill := 1 / combined_rate

-- The theorem that states the time taken to fill the tank is 30 hours
theorem fill_tank_time : time_to_fill = 30 := sorry

end NUMINAMATH_GPT_fill_tank_time_l941_94182


namespace NUMINAMATH_GPT_remainder_when_divided_by_24_l941_94125

theorem remainder_when_divided_by_24 (m k : ℤ) (h : m = 288 * k + 47) : m % 24 = 23 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_24_l941_94125


namespace NUMINAMATH_GPT_spilled_wax_amount_l941_94147

-- Definitions based on conditions
def car_wax := 3
def suv_wax := 4
def total_wax := 11
def remaining_wax := 2

-- The theorem to be proved
theorem spilled_wax_amount : car_wax + suv_wax + (total_wax - remaining_wax - (car_wax + suv_wax)) = total_wax - remaining_wax :=
by
  sorry


end NUMINAMATH_GPT_spilled_wax_amount_l941_94147


namespace NUMINAMATH_GPT_lcm_144_132_eq_1584_l941_94178

theorem lcm_144_132_eq_1584 :
  Nat.lcm 144 132 = 1584 :=
by
  sorry

end NUMINAMATH_GPT_lcm_144_132_eq_1584_l941_94178


namespace NUMINAMATH_GPT_regular_polygon_perimeter_l941_94114

theorem regular_polygon_perimeter
  (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : side_length = 8)
  (h2 : exterior_angle = 90)
  (h3 : n = 360 / exterior_angle) :
  n * side_length = 32 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_perimeter_l941_94114


namespace NUMINAMATH_GPT_problem_solution_l941_94175

-- Define the arithmetic sequence and its sum
def arith_seq_sum (n : ℕ) (a1 d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Define the specific condition for our problem
def a1_a5_equal_six (a1 d : ℕ) : Prop :=
  a1 + (a1 + 4 * d) = 6

-- The target value of S5 that we want to prove
def S5 (a1 d : ℕ) : ℕ :=
  arith_seq_sum 5 a1 d

theorem problem_solution (a1 d : ℕ) (h : a1_a5_equal_six a1 d) : S5 a1 d = 15 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l941_94175


namespace NUMINAMATH_GPT_solve_inequality_l941_94151

open Real

noncomputable def expression (x : ℝ) : ℝ :=
  (sqrt (x^2 - 4*x + 3) + 1) * log x / (log 2 * 5) + (1 / x) * (sqrt (8 * x - 2 * x^2 - 6) + 1)

theorem solve_inequality :
  ∃ x : ℝ, x = 1 ∧
    (x > 0) ∧
    (x^2 - 4 * x + 3 ≥ 0) ∧
    (8 * x - 2 * x^2 - 6 ≥ 0) ∧
    expression x ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l941_94151


namespace NUMINAMATH_GPT_ten_percent_of_fifty_percent_of_five_hundred_l941_94192

theorem ten_percent_of_fifty_percent_of_five_hundred :
  0.10 * (0.50 * 500) = 25 :=
by
  sorry

end NUMINAMATH_GPT_ten_percent_of_fifty_percent_of_five_hundred_l941_94192


namespace NUMINAMATH_GPT_shanghai_team_score_l941_94123

variables (S B : ℕ)

-- Conditions
def yao_ming_points : ℕ := 30
def point_margin : ℕ := 10
def total_points_minus_10 : ℕ := 5 * yao_ming_points - 10
def combined_total_points : ℕ := total_points_minus_10

-- The system of equations as conditions
axiom condition1 : S - B = point_margin
axiom condition2 : S + B = combined_total_points

-- The proof statement
theorem shanghai_team_score : S = 75 :=
by
  sorry

end NUMINAMATH_GPT_shanghai_team_score_l941_94123


namespace NUMINAMATH_GPT_smallest_positive_integer_k_l941_94136

-- Define the conditions
def y : ℕ := 2^3 * 3^4 * (2^2)^5 * 5^6 * (2*3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Define the question statement
theorem smallest_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (y * k) = m^2) ∧ k = 30 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_k_l941_94136


namespace NUMINAMATH_GPT_pears_in_basket_l941_94128

def TaniaFruits (b1 b2 b3 b4 b5 : ℕ) : Prop :=
  b1 = 18 ∧ b2 = 12 ∧ b3 = 9 ∧ b4 = b3 ∧ b5 + b1 + b2 + b3 + b4 = 58

theorem pears_in_basket {b1 b2 b3 b4 b5 : ℕ} (h : TaniaFruits b1 b2 b3 b4 b5) : b5 = 10 :=
by 
  sorry

end NUMINAMATH_GPT_pears_in_basket_l941_94128


namespace NUMINAMATH_GPT_quadratic_floor_eq_more_than_100_roots_l941_94189

open Int

theorem quadratic_floor_eq_more_than_100_roots (p q : ℤ) (h : p ≠ 0) :
  ∃ (S : Finset ℤ), S.card > 100 ∧ ∀ x ∈ S, ⌊(x : ℝ) ^ 2⌋ + p * x + q = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_floor_eq_more_than_100_roots_l941_94189


namespace NUMINAMATH_GPT_remainder_of_4_pow_a_div_10_l941_94188

theorem remainder_of_4_pow_a_div_10 (a : ℕ) (h1 : a > 0) (h2 : a % 2 = 0) :
  (4 ^ a) % 10 = 6 :=
by sorry

end NUMINAMATH_GPT_remainder_of_4_pow_a_div_10_l941_94188
