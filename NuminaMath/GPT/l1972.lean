import Mathlib

namespace base_area_cone_l1972_197294

theorem base_area_cone (V h : ℝ) (s_cylinder s_cone : ℝ) 
  (cylinder_volume : V = s_cylinder * h) 
  (cone_volume : V = (1 / 3) * s_cone * h) 
  (s_cylinder_val : s_cylinder = 15) : s_cone = 45 := 
by 
  sorry

end base_area_cone_l1972_197294


namespace tank_emptying_time_l1972_197261

theorem tank_emptying_time
  (initial_volume : ℝ)
  (filling_rate : ℝ)
  (emptying_rate : ℝ)
  (initial_fraction_full : initial_volume = 1 / 5)
  (pipe_a_rate : filling_rate = 1 / 10)
  (pipe_b_rate : emptying_rate = 1 / 6) :
  (initial_volume / (filling_rate - emptying_rate) = 3) :=
by
  sorry

end tank_emptying_time_l1972_197261


namespace card_at_42_is_8_spade_l1972_197234

-- Conditions Definition
def cards_sequence : List String := 
  ["A♥", "A♠", "2♥", "2♠", "3♥", "3♠", "4♥", "4♠", "5♥", "5♠", "6♥", "6♠", "7♥", "7♠", "8♥", "8♠",
   "9♥", "9♠", "10♥", "10♠", "J♥", "J♠", "Q♥", "Q♠", "K♥", "K♠"]

-- Proposition to be proved
theorem card_at_42_is_8_spade :
  cards_sequence[(41 % 26)] = "8♠" :=
by sorry

end card_at_42_is_8_spade_l1972_197234


namespace sum_of_sequences_l1972_197291

def sequence1 := [2, 14, 26, 38, 50]
def sequence2 := [12, 24, 36, 48, 60]
def sequence3 := [5, 15, 25, 35, 45]

theorem sum_of_sequences :
  (sequence1.sum + sequence2.sum + sequence3.sum) = 435 := 
by 
  sorry

end sum_of_sequences_l1972_197291


namespace solution_set_of_inequality_l1972_197205

theorem solution_set_of_inequality (x : ℝ) :
  (x + 1) * (2 - x) < 0 ↔ (x > 2 ∨ x < -1) :=
by
  sorry

end solution_set_of_inequality_l1972_197205


namespace ellipse_equation_standard_form_l1972_197250

theorem ellipse_equation_standard_form :
  ∃ (a b : ℝ) (h k : ℝ), 
    a = (Real.sqrt 146 + Real.sqrt 242) / 2 ∧ 
    b = Real.sqrt ((Real.sqrt 146 + Real.sqrt 242) / 2)^2 - 9 ∧ 
    h = 1 ∧ 
    k = 4 ∧ 
    (∀ x y : ℝ, (x, y) = (12, -4) → 
      ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)) :=
  sorry

end ellipse_equation_standard_form_l1972_197250


namespace abs_diff_p_q_l1972_197275

theorem abs_diff_p_q (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 :=
by 
  sorry

end abs_diff_p_q_l1972_197275


namespace flower_bed_can_fit_l1972_197287

noncomputable def flower_bed_fits_in_yard : Prop :=
  let yard_side := 70
  let yard_area := yard_side ^ 2
  let building1 := (20 * 10)
  let building2 := (25 * 15)
  let building3 := (30 * 30)
  let tank_radius := 10 / 2
  let tank_area := Real.pi * tank_radius^2
  let total_occupied_area := building1 + building2 + building3 + 2*tank_area
  let available_area := yard_area - total_occupied_area
  let flower_bed_radius := 10 / 2
  let flower_bed_area := Real.pi * flower_bed_radius^2
  let buffer_area := (yard_side - 2 * flower_bed_radius)^2
  available_area >= flower_bed_area ∧ buffer_area >= flower_bed_area

theorem flower_bed_can_fit : flower_bed_fits_in_yard := 
  sorry

end flower_bed_can_fit_l1972_197287


namespace problem_statement_l1972_197240

def binary_op (a b : ℚ) : ℚ := (a^2 + b^2) / (a^2 - b^2)

theorem problem_statement : binary_op (binary_op 8 6) 2 = 821 / 429 := 
by sorry

end problem_statement_l1972_197240


namespace actual_distance_l1972_197282

theorem actual_distance (d_map : ℝ) (scale_inches : ℝ) (scale_miles : ℝ) (H1 : d_map = 20)
    (H2 : scale_inches = 0.5) (H3 : scale_miles = 10) : 
    d_map * (scale_miles / scale_inches) = 400 := 
by
  sorry

end actual_distance_l1972_197282


namespace joseph_drives_more_l1972_197246

def joseph_speed : ℝ := 50
def joseph_time : ℝ := 2.5
def kyle_speed : ℝ := 62
def kyle_time : ℝ := 2

def joseph_distance : ℝ := joseph_speed * joseph_time
def kyle_distance : ℝ := kyle_speed * kyle_time

theorem joseph_drives_more : (joseph_distance - kyle_distance) = 1 := by
  sorry

end joseph_drives_more_l1972_197246


namespace general_formula_sequence_l1972_197256

variable {a : ℕ → ℝ}

-- Definitions and assumptions
def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n - 2 * a (n + 1) + a (n + 2) = 0

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 2 = 4

-- The proof problem
theorem general_formula_sequence (a : ℕ → ℝ)
  (h1 : recurrence_relation a)
  (h2 : initial_conditions a) :
  ∀ n : ℕ, a n = 2 * n :=

sorry

end general_formula_sequence_l1972_197256


namespace one_third_way_l1972_197296

theorem one_third_way (x₁ x₂ : ℚ) (w₁ w₂ : ℕ) (h₁ : x₁ = 1/4) (h₂ : x₂ = 3/4) (h₃ : w₁ = 2) (h₄ : w₂ = 1) : 
  (w₁ * x₁ + w₂ * x₂) / (w₁ + w₂) = 5 / 12 :=
by 
  rw [h₁, h₂, h₃, h₄]
  -- Simplification of the weighted average to get 5/12
  sorry

end one_third_way_l1972_197296


namespace sodium_thiosulfate_properties_l1972_197289

def thiosulfate_structure : Type := sorry
-- Define the structure of S2O3^{2-} with S-S bond
def has_s_s_bond (ion : thiosulfate_structure) : Prop := sorry
-- Define the formation reaction
def formed_by_sulfite_reaction (ion : thiosulfate_structure) : Prop := sorry

theorem sodium_thiosulfate_properties :
  ∃ (ion : thiosulfate_structure),
    has_s_s_bond ion ∧ formed_by_sulfite_reaction ion :=
by
  sorry

end sodium_thiosulfate_properties_l1972_197289


namespace total_payment_l1972_197206

def cement_bags := 500
def cost_per_bag := 10
def lorries := 20
def tons_per_lorry := 10
def cost_per_ton := 40

theorem total_payment : cement_bags * cost_per_bag + lorries * tons_per_lorry * cost_per_ton = 13000 := by
  sorry

end total_payment_l1972_197206


namespace sequence_expression_l1972_197268

theorem sequence_expression {a : ℕ → ℝ} (h1 : ∀ n, a (n + 1) ^ 2 = a n ^ 2 + 4)
  (h2 : a 1 = 1) (h3 : ∀ n, a n > 0) : ∀ n, a n = Real.sqrt (4 * n - 3) := by
  sorry

end sequence_expression_l1972_197268


namespace collinear_vectors_m_n_sum_l1972_197297

theorem collinear_vectors_m_n_sum (m n : ℕ)
  (h1 : (2, 3, m) = (2 * n, 6, 8)) :
  m + n = 6 :=
sorry

end collinear_vectors_m_n_sum_l1972_197297


namespace curve_of_constant_width_l1972_197293

structure Curve :=
  (is_convex : Prop)

structure Point := 
  (x : ℝ) 
  (y : ℝ)

def rotate_180 (K : Curve) (O : Point) : Curve := sorry

def sum_curves (K1 K2 : Curve) : Curve := sorry

def is_circle_with_radius (K : Curve) (r : ℝ) : Prop := sorry

def constant_width (K : Curve) (w : ℝ) : Prop := sorry

theorem curve_of_constant_width {K : Curve} {O : Point} {h : ℝ} :
  K.is_convex →
  (K' : Curve) → K' = rotate_180 K O →
  is_circle_with_radius (sum_curves K K') h →
  constant_width K h :=
by 
  sorry

end curve_of_constant_width_l1972_197293


namespace total_people_bought_tickets_l1972_197209

-- Definitions based on the conditions from step a)
def num_adults := 375
def num_children := 3 * num_adults
def total_revenue := 7 * num_adults + 3 * num_children

-- Statement of the theorem based on the question in step a)
theorem total_people_bought_tickets : (num_adults + num_children) = 1500 :=
by
  -- The proof is omitted, but we're ensuring the correctness of the theorem statement.
  sorry

end total_people_bought_tickets_l1972_197209


namespace age_of_b_l1972_197216

-- Define the conditions as per the problem statement
variables (A B C D E : ℚ)

axiom cond1 : A = B + 2
axiom cond2 : B = 2 * C
axiom cond3 : D = A - 3
axiom cond4 : E = D / 2 + 3
axiom cond5 : A + B + C + D + E = 70

theorem age_of_b : B = 16.625 :=
by {
  -- Placeholder for the proof
  sorry
}

end age_of_b_l1972_197216


namespace smallest_x_l1972_197203

theorem smallest_x (x : ℚ) (h : 7 * (4 * x^2 + 4 * x + 5) = x * (4 * x - 35)) : 
  x = -5/3 ∨ x = -7/8 := by
  sorry

end smallest_x_l1972_197203


namespace log_expression_value_l1972_197235

theorem log_expression_value : 
  let log4_3 := (Real.log 3) / (Real.log 4)
  let log8_3 := (Real.log 3) / (Real.log 8)
  let log3_2 := (Real.log 2) / (Real.log 3)
  let log9_2 := (Real.log 2) / (Real.log 9)
  (log4_3 + log8_3) * (log3_2 + log9_2) = 5 / 4 := 
by
  sorry

end log_expression_value_l1972_197235


namespace constantin_mother_deposit_return_l1972_197258

theorem constantin_mother_deposit_return :
  (10000 : ℝ) * 58.15 = 581500 :=
by
  sorry

end constantin_mother_deposit_return_l1972_197258


namespace correct_statements_l1972_197259

-- Definitions
noncomputable def f (x b c : ℝ) : ℝ := abs x * x + b * x + c

-- Proof statements
theorem correct_statements (b c : ℝ) :
  (b > 0 → ∀ x y : ℝ, x ≤ y → f x b c ≤ f y b c) ∧
  (b < 0 → ¬ (∀ x : ℝ, ∃ m : ℝ, f x b c = m)) ∧
  (b = 0 → ∀ x : ℝ, f (x) b c = f (-x) b c) ∧
  (∃ x1 x2 x3 : ℝ, f x1 b c = 0 ∧ f x2 b c = 0 ∧ f x3 b c = 0) :=
sorry

end correct_statements_l1972_197259


namespace triangle_inequality_l1972_197247

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) : 
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7 / 3 :=
by
  sorry

end triangle_inequality_l1972_197247


namespace price_of_first_doughnut_l1972_197227

theorem price_of_first_doughnut 
  (P : ℕ)  -- Price of the first doughnut
  (total_doughnuts : ℕ := 48)  -- Total number of doughnuts
  (price_per_dozen : ℕ := 6)  -- Price per dozen of additional doughnuts
  (total_cost : ℕ := 24)  -- Total cost spent
  (doughnuts_left : ℕ := total_doughnuts - 1)  -- Doughnuts left after the first one
  (dozens : ℕ := doughnuts_left / 12)  -- Number of whole dozens
  (cost_of_dozens : ℕ := dozens * price_per_dozen)  -- Cost of the dozens of doughnuts
  (cost_after_first : ℕ := total_cost - cost_of_dozens)  -- Remaining cost after dozens
  : P = 6 := 
by
  -- Proof to be filled in
  sorry

end price_of_first_doughnut_l1972_197227


namespace geometric_sequence_x_l1972_197204

theorem geometric_sequence_x (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 := by
  sorry

end geometric_sequence_x_l1972_197204


namespace triangle_sine_cosine_l1972_197221

theorem triangle_sine_cosine (a b A : ℝ) (B C : ℝ) (c : ℝ) 
  (ha : a = Real.sqrt 7) 
  (hb : b = 2) 
  (hA : A = 60 * Real.pi / 180) 
  (hsinB : Real.sin B = Real.sin B := by sorry)
  (hc : c = 3 := by sorry) :
  (Real.sin B = Real.sqrt 21 / 7) ∧ (c = 3) := 
sorry

end triangle_sine_cosine_l1972_197221


namespace bob_deli_total_cost_l1972_197232

-- Definitions based on the problem's conditions
def sandwich_cost : ℕ := 5
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 10
def discount_threshold : ℕ := 50
def discount_amount : ℕ := 10

-- The total initial cost without discount
def initial_total_cost : ℕ :=
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)

-- The final cost after applying discount if applicable
def final_cost : ℕ :=
  if initial_total_cost > discount_threshold then
    initial_total_cost - discount_amount
  else
    initial_total_cost

-- Statement to prove
theorem bob_deli_total_cost : final_cost = 55 := by
  sorry

end bob_deli_total_cost_l1972_197232


namespace train_pass_time_is_38_seconds_l1972_197298

noncomputable def speed_of_jogger_kmhr : ℝ := 9
noncomputable def speed_of_train_kmhr : ℝ := 45
noncomputable def lead_distance_m : ℝ := 260
noncomputable def train_length_m : ℝ := 120

noncomputable def speed_of_jogger_ms : ℝ := speed_of_jogger_kmhr * (1000 / 3600)
noncomputable def speed_of_train_ms : ℝ := speed_of_train_kmhr * (1000 / 3600)

noncomputable def relative_speed_ms : ℝ := speed_of_train_ms - speed_of_jogger_ms
noncomputable def total_distance_m : ℝ := lead_distance_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ := total_distance_m / relative_speed_ms

theorem train_pass_time_is_38_seconds :
  time_to_pass_jogger_s = 38 := 
sorry

end train_pass_time_is_38_seconds_l1972_197298


namespace probability_of_same_number_l1972_197299

theorem probability_of_same_number (m n : ℕ) 
  (hb : m < 250 ∧ m % 20 = 0) 
  (bb : n < 250 ∧ n % 30 = 0) : 
  (∀ (b : ℕ), b < 250 ∧ b % 60 = 0 → ∃ (m n : ℕ), ((m < 250 ∧ m % 20 = 0) ∧ (n < 250 ∧ n % 30 = 0)) → (m = n)) :=
sorry

end probability_of_same_number_l1972_197299


namespace sum_of_legs_of_right_triangle_l1972_197215

theorem sum_of_legs_of_right_triangle
  (a b : ℕ)
  (h1 : a % 2 = 0)
  (h2 : b = a + 2)
  (h3 : a^2 + b^2 = 50^2) :
  a + b = 70 := by
  sorry

end sum_of_legs_of_right_triangle_l1972_197215


namespace eq_solutions_of_equation_l1972_197238

open Int

theorem eq_solutions_of_equation (x y : ℤ) :
  ((x, y) = (0, -4) ∨ (x, y) = (0, 8) ∨
   (x, y) = (-2, 0) ∨ (x, y) = (-4, 8) ∨
   (x, y) = (-2, 0) ∨ (x, y) = (-6, 6) ∨
   (x, y) = (0, 0) ∨ (x, y) = (-10, 4)) ↔
  (x - y) * (x - y) = (x - y + 6) * (x + y) :=
sorry

end eq_solutions_of_equation_l1972_197238


namespace total_number_of_items_l1972_197266

-- Definitions based on the problem conditions
def number_of_notebooks : ℕ := 40
def pens_more_than_notebooks : ℕ := 80
def pencils_more_than_notebooks : ℕ := 45

-- Total items calculation based on the conditions
def number_of_pens : ℕ := number_of_notebooks + pens_more_than_notebooks
def number_of_pencils : ℕ := number_of_notebooks + pencils_more_than_notebooks
def total_items : ℕ := number_of_notebooks + number_of_pens + number_of_pencils

-- Statement to be proved
theorem total_number_of_items : total_items = 245 := 
by 
  sorry

end total_number_of_items_l1972_197266


namespace rate_per_kg_first_batch_l1972_197244

/-- This theorem proves the rate per kg of the first batch of wheat. -/
theorem rate_per_kg_first_batch (x : ℝ) 
  (h1 : 30 * x + 20 * 14.25 = 285 + 30 * x) 
  (h2 : (30 * x + 285) * 1.3 = 819) : 
  x = 11.5 := 
sorry

end rate_per_kg_first_batch_l1972_197244


namespace overlapping_area_of_congruent_isosceles_triangles_l1972_197200

noncomputable def isosceles_right_triangle (hypotenuse : ℝ) := 
  {l : ℝ // l = hypotenuse / Real.sqrt 2}

theorem overlapping_area_of_congruent_isosceles_triangles (hypotenuse : ℝ) 
  (A₁ A₂ : isosceles_right_triangle hypotenuse) (h_congruent : A₁ = A₂) :
  hypotenuse = 10 → 
  let leg := hypotenuse / Real.sqrt 2 
  let area := (leg * leg) / 2 
  let shared_area := area / 2 
  shared_area = 12.5 :=
by
  sorry

end overlapping_area_of_congruent_isosceles_triangles_l1972_197200


namespace number_division_l1972_197285

theorem number_division (n : ℕ) (h1 : n / 25 = 5) (h2 : n % 25 = 2) : n = 127 :=
by
  sorry

end number_division_l1972_197285


namespace area_of_L_shaped_figure_l1972_197231

theorem area_of_L_shaped_figure :
  let large_rect_area := 10 * 7
  let small_rect_area := 4 * 3
  large_rect_area - small_rect_area = 58 := by
  sorry

end area_of_L_shaped_figure_l1972_197231


namespace cans_increment_l1972_197286

/--
If there are 9 rows of cans in a triangular display, where each successive row increases 
by a certain number of cans \( x \) compared to the row above it, with the seventh row having 
19 cans, and the total number of cans being fewer than 120, then 
each row has 4 more cans than the row above it.
-/
theorem cans_increment (x : ℕ) : 
  9 * 19 - 16 * x < 120 → x > 51 / 16 → x = 4 :=
by
  intros h1 h2
  sorry

end cans_increment_l1972_197286


namespace rectangle_perimeter_l1972_197270

theorem rectangle_perimeter {a b c width : ℕ} (h₁: a = 15) (h₂: b = 20) (h₃: c = 25) (w : ℕ) (h₄: w = 5) :
  let area_triangle := (a * b) / 2
  let length := area_triangle / w
  let perimeter := 2 * (length + w)
  perimeter = 70 :=
by
  sorry

end rectangle_perimeter_l1972_197270


namespace division_expression_is_7_l1972_197295

noncomputable def evaluate_expression : ℝ :=
  1 / 2 / 3 / 4 / 5 / (6 / 7 / 8 / 9 / 10)

theorem division_expression_is_7 : evaluate_expression = 7 :=
by
  sorry

end division_expression_is_7_l1972_197295


namespace journey_time_ratio_l1972_197257

theorem journey_time_ratio (D : ℝ) (h₁ : D > 0) :
  let T1 := D / 48
  let T2 := D / 32
  (T2 / T1) = 3 / 2 :=
by
  sorry

end journey_time_ratio_l1972_197257


namespace eval_neg64_pow_two_thirds_l1972_197229

theorem eval_neg64_pow_two_thirds : (-64 : Real)^(2/3) = 16 := 
by 
  sorry

end eval_neg64_pow_two_thirds_l1972_197229


namespace f_odd_f_shift_f_in_range_find_f_7_5_l1972_197267

def f : ℝ → ℝ := sorry  -- We define the function f (implementation is not needed here)

theorem f_odd (x : ℝ) : f (-x) = -f x := sorry

theorem f_shift (x : ℝ) : f (x + 2) = -f x := sorry

theorem f_in_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x = x := sorry

theorem find_f_7_5 : f 7.5 = 0.5 :=
by
  sorry

end f_odd_f_shift_f_in_range_find_f_7_5_l1972_197267


namespace find_natural_solution_l1972_197260

theorem find_natural_solution (x y : ℕ) (h : y^6 + 2 * y^3 - y^2 + 1 = x^3) : x = 1 ∧ y = 0 :=
by
  sorry

end find_natural_solution_l1972_197260


namespace range_of_sum_coords_on_ellipse_l1972_197202

theorem range_of_sum_coords_on_ellipse (x y : ℝ) 
  (h : x^2 / 144 + y^2 / 25 = 1) : 
  -13 ≤ x + y ∧ x + y ≤ 13 := 
sorry

end range_of_sum_coords_on_ellipse_l1972_197202


namespace proof_problem_l1972_197252

noncomputable def f (x y k : ℝ) : ℝ := k * x + (1 / y)

theorem proof_problem
  (a b k : ℝ) (h1 : f a b k = f b a k) (h2 : a ≠ b) :
  f (a * b) 1 k = 0 :=
sorry

end proof_problem_l1972_197252


namespace negation_of_p_l1972_197280

variable (x y : ℝ)

def proposition_p := ∀ x y : ℝ, x^2 + y^2 - 1 > 0 

theorem negation_of_p : (¬ proposition_p) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) := by
  sorry

end negation_of_p_l1972_197280


namespace winning_percentage_l1972_197213

theorem winning_percentage (total_votes majority : ℕ) (h1 : total_votes = 455) (h2 : majority = 182) :
  ∃ P : ℕ, P = 70 ∧ (P / 100) * total_votes - ((100 - P) / 100) * total_votes = majority := 
sorry

end winning_percentage_l1972_197213


namespace simplify_expression_l1972_197262

theorem simplify_expression :
  (8 * 10^12) / (4 * 10^4) + 2 * 10^3 = 200002000 := 
by
  sorry

end simplify_expression_l1972_197262


namespace heather_lighter_than_combined_weights_l1972_197224

noncomputable def heather_weight : ℝ := 87.5
noncomputable def emily_weight : ℝ := 45.3
noncomputable def elizabeth_weight : ℝ := 38.7
noncomputable def george_weight : ℝ := 56.9

theorem heather_lighter_than_combined_weights :
  heather_weight - (emily_weight + elizabeth_weight + george_weight) = -53.4 :=
by 
  sorry

end heather_lighter_than_combined_weights_l1972_197224


namespace custom_mul_2021_1999_l1972_197283

axiom custom_mul : ℕ → ℕ → ℕ

axiom custom_mul_id1 : ∀ (A : ℕ), custom_mul A A = 0
axiom custom_mul_id2 : ∀ (A B C : ℕ), custom_mul A (custom_mul B C) = custom_mul A B + C

theorem custom_mul_2021_1999 : custom_mul 2021 1999 = 22 := by
  sorry

end custom_mul_2021_1999_l1972_197283


namespace dagger_evaluation_l1972_197230

def dagger (a b : ℚ) : ℚ :=
match a, b with
| ⟨m, n, _, _⟩, ⟨p, q, _, _⟩ => (m * p : ℚ) * (q / n : ℚ)

theorem dagger_evaluation : dagger (3/7) (11/4) = 132/7 := by
  sorry

end dagger_evaluation_l1972_197230


namespace area_not_covered_by_small_squares_l1972_197239

def large_square_side_length : ℕ := 10
def small_square_side_length : ℕ := 4
def large_square_area : ℕ := large_square_side_length ^ 2
def small_square_area : ℕ := small_square_side_length ^ 2
def uncovered_area : ℕ := large_square_area - small_square_area

theorem area_not_covered_by_small_squares :
  uncovered_area = 84 := by
  sorry

end area_not_covered_by_small_squares_l1972_197239


namespace simplify_and_find_ratio_l1972_197248

theorem simplify_and_find_ratio (m : ℤ) (c d : ℤ) (h : (5 * m + 15) / 5 = c * m + d) : d / c = 3 := by
  sorry

end simplify_and_find_ratio_l1972_197248


namespace fraction_value_l1972_197279

theorem fraction_value : (1998 - 998) / 1000 = 1 :=
by
  sorry

end fraction_value_l1972_197279


namespace solve_inequality1_solve_inequality2_l1972_197217

-- Proof problem 1
theorem solve_inequality1 (x : ℝ) : 
  2 < |2 * x - 5| → |2 * x - 5| ≤ 7 → -1 ≤ x ∧ x < (3 / 2) ∨ (7 / 2) < x ∧ x ≤ 6 :=
sorry

-- Proof problem 2
theorem solve_inequality2 (x : ℝ) : 
  (1 / (x - 1)) > (x + 1) → x < - Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2) :=
sorry

end solve_inequality1_solve_inequality2_l1972_197217


namespace arrangement_exists_l1972_197219

-- Definitions of pairwise coprimeness and gcd
def pairwise_coprime (a b c d : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1

def common_divisor (x y : ℕ) : Prop := ∃ d > 1, d ∣ x ∧ d ∣ y

def relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

-- Main theorem statement
theorem arrangement_exists :
  ∃ a b c d ab cd ad bc abcd : ℕ,
    pairwise_coprime a b c d ∧
    ab = a * b ∧ cd = c * d ∧ ad = a * d ∧ bc = b * c ∧ abcd = a * b * c * d ∧
    (common_divisor ab abcd ∧ common_divisor cd abcd ∧ common_divisor ad abcd ∧ common_divisor bc abcd) ∧
    (common_divisor ab ad ∧ common_divisor ab bc ∧ common_divisor cd ad ∧ common_divisor cd bc) ∧
    (relatively_prime ab cd ∧ relatively_prime ad bc) :=
by
  -- The proof will be filled here
  sorry

end arrangement_exists_l1972_197219


namespace geometric_sequence_a5_l1972_197255

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) 
  (h₀ : ∀ n, a n + a (n + 1) = 3 * (1 / 2) ^ n)
  (h₁ : ∀ n, a (n + 1) = a n * q)
  (h₂ : q = 1 / 2) :
  a 5 = 1 / 16 :=
sorry

end geometric_sequence_a5_l1972_197255


namespace xy_range_l1972_197228

theorem xy_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 1/x + y + 1/y = 5) :
  1/4 ≤ x * y ∧ x * y ≤ 4 :=
sorry

end xy_range_l1972_197228


namespace abscissa_range_of_point_P_l1972_197292

-- Definitions based on the conditions from the problem
def y_function (x : ℝ) : ℝ := 4 - 3 * x
def point_P (x y : ℝ) : Prop := y = y_function x
def ordinate_greater_than_negative_five (y : ℝ) : Prop := y > -5

-- Theorem statement combining the above definitions
theorem abscissa_range_of_point_P (x y : ℝ) :
  point_P x y →
  ordinate_greater_than_negative_five y →
  x < 3 :=
sorry

end abscissa_range_of_point_P_l1972_197292


namespace Martha_should_buy_84oz_of_apples_l1972_197276

theorem Martha_should_buy_84oz_of_apples 
  (apple_weight : ℕ)
  (orange_weight : ℕ)
  (bag_capacity : ℕ)
  (num_bags : ℕ)
  (equal_fruits : Prop) 
  (total_weight : ℕ :=
    num_bags * bag_capacity)
  (pair_weight : ℕ :=
    apple_weight + orange_weight)
  (num_pairs : ℕ :=
    total_weight / pair_weight)
  (total_apple_weight : ℕ :=
    num_pairs * apple_weight) :
  apple_weight = 4 → 
  orange_weight = 3 → 
  bag_capacity = 49 → 
  num_bags = 3 → 
  equal_fruits → 
  total_apple_weight = 84 := 
by sorry

end Martha_should_buy_84oz_of_apples_l1972_197276


namespace sin_y_gt_half_x_l1972_197288

theorem sin_y_gt_half_x (x y : ℝ) (hx : x ≤ 90) (h : Real.sin y = (3 / 4) * Real.sin x) : y > x / 2 :=
by
  sorry

end sin_y_gt_half_x_l1972_197288


namespace locus_of_midpoint_l1972_197269

open Real

noncomputable def circumcircle_eq (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := 1
  let b := 3
  let r2 := 5
  (a, b, r2)

theorem locus_of_midpoint (A B C N : ℝ × ℝ) :
  N = (6, 2) ∧ A = (0, 1) ∧ B = (2, 1) ∧ C = (3, 4) → 
  let P := (7 / 2, 5 / 2)
  let r2 := 5 / 4
  ∃ x y : ℝ, 
  (x, y) = P ∧ (x - 7 / 2)^2 + (y - 5 / 2)^2 = r2 :=
by sorry

end locus_of_midpoint_l1972_197269


namespace sum_coordinates_is_60_l1972_197225

theorem sum_coordinates_is_60 :
  let points := [(5 + Real.sqrt 91, 13), (5 - Real.sqrt 91, 13), (5 + Real.sqrt 91, 7), (5 - Real.sqrt 91, 7)]
  let x_coords_sum := (5 + Real.sqrt 91) + (5 - Real.sqrt 91) + (5 + Real.sqrt 91) + (5 - Real.sqrt 91)
  let y_coords_sum := 13 + 13 + 7 + 7
  x_coords_sum + y_coords_sum = 60 :=
by
  sorry

end sum_coordinates_is_60_l1972_197225


namespace range_of_a_l1972_197201

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := {x : ℝ | (a * x - 1) / (x - a) < 0}

theorem range_of_a (h1 : 2 ∈ A a) (h2 : 3 ∉ A a) : (1 / 3 : ℝ) ≤ a ∧ a < 1 / 2 ∨ 2 < a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l1972_197201


namespace part1_part2_l1972_197290

section
variable (x a : ℝ)
def p (x a : ℝ) : Prop := (x - 3 * a) * (x - a) < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem part1 (h : a = 1) (hq : q x) (hp : p x a) : 2 < x ∧ x < 3 := by
  sorry

theorem part2 (h : ∀ x, q x → p x a) : 1 ≤ a ∧ a ≤ 2 := by
  sorry
end

end part1_part2_l1972_197290


namespace simplify_rationalize_expr_l1972_197212

theorem simplify_rationalize_expr : 
  (1 / (2 + 1 / (Real.sqrt 5 - 2))) = (4 - Real.sqrt 5) / 11 := 
by 
  sorry

end simplify_rationalize_expr_l1972_197212


namespace circle_equation_AB_diameter_l1972_197249

theorem circle_equation_AB_diameter (A B : ℝ × ℝ) :
  A = (1, -4) → B = (-5, 4) →
  ∃ C : ℝ × ℝ, C = (-2, 0) ∧ ∃ r : ℝ, r = 5 ∧ (∀ x y : ℝ, (x + 2)^2 + y^2 = 25) :=
by intros h1 h2; sorry

end circle_equation_AB_diameter_l1972_197249


namespace sine_angle_greater_implies_angle_greater_l1972_197222

noncomputable def triangle := {ABC : Type* // Π A B C : ℕ, 
  A + B + C = 180 ∧ 0 < A ∧ A < 180 ∧ 0 < B ∧ B < 180 ∧ 0 < C ∧ C < 180}

variables {A B C : ℕ} (T : triangle)

theorem sine_angle_greater_implies_angle_greater (h1 : 0 < A ∧ A < 180) (h2 : 0 < B ∧ B < 180)
  (h3 : 0 < C ∧ C < 180) (h_sum : A + B + C = 180) (h_sine : Real.sin A > Real.sin B) :
  A > B := 
sorry

end sine_angle_greater_implies_angle_greater_l1972_197222


namespace survey_method_correct_l1972_197245

/-- Definitions to represent the options in the survey method problem. -/
inductive SurveyMethod
| A
| B
| C
| D

/-- The function to determine the correct survey method. -/
def appropriate_survey_method : SurveyMethod :=
  SurveyMethod.C

/-- The theorem stating that the appropriate survey method is indeed option C. -/
theorem survey_method_correct : appropriate_survey_method = SurveyMethod.C :=
by
  /- The actual proof is omitted as per instruction. -/
  sorry

end survey_method_correct_l1972_197245


namespace maximum_absolute_sum_l1972_197284

theorem maximum_absolute_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : |x| + |y| + |z| ≤ 2 :=
sorry

end maximum_absolute_sum_l1972_197284


namespace pyramid_volume_and_base_edge_l1972_197220

theorem pyramid_volume_and_base_edge:
  ∀ (r: ℝ) (h: ℝ) (_: r = 5) (_: h = 10), 
  ∃ s V: ℝ,
    s = (10 * Real.sqrt 6) / 3 ∧ 
    V = (2000 / 9) :=
by
    sorry

end pyramid_volume_and_base_edge_l1972_197220


namespace convert_quadratic_to_general_form_l1972_197271

theorem convert_quadratic_to_general_form
  (x : ℝ)
  (h : 3 * x * (x - 3) = 4) :
  3 * x ^ 2 - 9 * x - 4 = 0 :=
by
  sorry

end convert_quadratic_to_general_form_l1972_197271


namespace number_of_BMWs_sold_l1972_197263

theorem number_of_BMWs_sold (total_cars_sold : ℕ)
  (percent_Ford percent_Nissan percent_Chevrolet : ℕ)
  (h_total : total_cars_sold = 300)
  (h_percent_Ford : percent_Ford = 18)
  (h_percent_Nissan : percent_Nissan = 25)
  (h_percent_Chevrolet : percent_Chevrolet = 20) :
  (300 * (100 - (percent_Ford + percent_Nissan + percent_Chevrolet)) / 100) = 111 :=
by
  -- We assert that the calculated number of BMWs is 111
  sorry

end number_of_BMWs_sold_l1972_197263


namespace rectangle_area_increase_l1972_197226

-- Definitions to match the conditions
variables {l w : ℝ}

-- The statement 
theorem rectangle_area_increase (h1 : l > 0) (h2 : w > 0) :
  (((1.15 * l) * (1.2 * w) - (l * w)) / (l * w)) * 100 = 38 :=
by
  sorry

end rectangle_area_increase_l1972_197226


namespace geese_percentage_l1972_197223

noncomputable def percentage_of_geese_among_non_swans (geese swans herons ducks : ℝ) : ℝ :=
  (geese / (100 - swans)) * 100

theorem geese_percentage (geese swans herons ducks : ℝ)
  (h1 : geese = 40)
  (h2 : swans = 20)
  (h3 : herons = 15)
  (h4 : ducks = 25) :
  percentage_of_geese_among_non_swans geese swans herons ducks = 50 :=
by
  simp [percentage_of_geese_among_non_swans, h1, h2, h3, h4]
  sorry

end geese_percentage_l1972_197223


namespace sets_satisfy_union_l1972_197214

theorem sets_satisfy_union (A : Set Int) : (A ∪ {-1, 1} = {-1, 0, 1}) → 
  (∃ (X : Finset (Set Int)), X.card = 4 ∧ ∀ B ∈ X, A = B) :=
  sorry

end sets_satisfy_union_l1972_197214


namespace Nicole_fewer_questions_l1972_197264

-- Definitions based on the given conditions
def Nicole_correct : ℕ := 22
def Cherry_correct : ℕ := 17
def Kim_correct : ℕ := Cherry_correct + 8

-- Theorem to prove the number of fewer questions Nicole answered compared to Kim
theorem Nicole_fewer_questions : Kim_correct - Nicole_correct = 3 :=
by
  -- We set up the definitions
  let Nicole_correct := 22
  let Cherry_correct := 17
  let Kim_correct := Cherry_correct + 8
  -- The proof will be filled in here. 
  -- The goal theorem statement is filled with 'sorry' to bypass the actual proof.
  have : Kim_correct - Nicole_correct = 3 := sorry
  exact this

end Nicole_fewer_questions_l1972_197264


namespace chord_length_l1972_197210

theorem chord_length (x y : ℝ) :
  (x^2 + y^2 - 2 * x - 4 * y = 0) →
  (x + 2 * y - 5 + Real.sqrt 5 = 0) →
  ∃ l, l = 4 :=
by
  intros h_circle h_line
  sorry

end chord_length_l1972_197210


namespace smallest_n_for_terminating_decimal_l1972_197243

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), 0 < n ∧ ∀ m : ℕ, (0 < m ∧ m < n+53) → (∃ a b : ℕ, n + 53 = 2^a * 5^b) → n = 11 :=
by
  sorry

end smallest_n_for_terminating_decimal_l1972_197243


namespace fencing_cost_proof_l1972_197236

noncomputable def totalCostOfFencing (length : ℕ) (breadth : ℕ) (costPerMeter : ℚ) : ℚ :=
  2 * (length + breadth) * costPerMeter

theorem fencing_cost_proof : totalCostOfFencing 56 (56 - 12) 26.50 = 5300 := by
  sorry

end fencing_cost_proof_l1972_197236


namespace visited_both_countries_l1972_197251

theorem visited_both_countries (total_people visited_Iceland visited_Norway visited_neither : ℕ) 
(h_total: total_people = 60)
(h_visited_Iceland: visited_Iceland = 35)
(h_visited_Norway: visited_Norway = 23)
(h_visited_neither: visited_neither = 33) : 
total_people - visited_neither = visited_Iceland + visited_Norway - (visited_Iceland + visited_Norway - (total_people - visited_neither)) :=
by sorry

end visited_both_countries_l1972_197251


namespace simplify_expression_eq_l1972_197242

noncomputable def simplified_expression (b : ℝ) : ℝ :=
  (Real.rpow (Real.rpow (b ^ 16) (1 / 8)) (1 / 4)) ^ 3 *
  (Real.rpow (Real.rpow (b ^ 16) (1 / 4)) (1 / 8)) ^ 3

theorem simplify_expression_eq (b : ℝ) (hb : 0 < b) :
  simplified_expression b = b ^ 3 :=
by sorry

end simplify_expression_eq_l1972_197242


namespace fraction_equal_decimal_l1972_197278

theorem fraction_equal_decimal : (1 / 4) = 0.25 :=
sorry

end fraction_equal_decimal_l1972_197278


namespace compute_expression_l1972_197277

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l1972_197277


namespace cakes_remain_l1972_197208

def initial_cakes := 110
def sold_cakes := 75
def new_cakes := 76

theorem cakes_remain : (initial_cakes - sold_cakes) + new_cakes = 111 :=
by
  sorry

end cakes_remain_l1972_197208


namespace sin_600_eq_l1972_197281

theorem sin_600_eq : Real.sin (600 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_600_eq_l1972_197281


namespace isosceles_triangle_height_ratio_l1972_197253

theorem isosceles_triangle_height_ratio (a b : ℝ) (h₁ : b = (4 / 3) * a) :
  ∃ m n : ℝ, b / 2 = m + n ∧ m = (2 / 3) * a ∧ n = (1 / 3) * a ∧ (m / n) = 2 :=
by
  sorry

end isosceles_triangle_height_ratio_l1972_197253


namespace correct_calculation_l1972_197265

theorem correct_calculation (a b m : ℤ) : 
  (¬((a^3)^2 = a^5)) ∧ ((-2 * m^3)^2 = 4 * m^6) ∧ (¬(a^6 / a^2 = a^3)) ∧ (¬((a + b)^2 = a^2 + b^2)) := 
by
  sorry

end correct_calculation_l1972_197265


namespace complement_of_A_in_U_l1972_197207

-- Define the universal set U
def U : Set ℕ := {2, 3, 4}

-- Define set A
def A : Set ℕ := {2, 3}

-- Define the complement of A in U
def C_U_A : Set ℕ := U \ A

-- Prove the complement of A in U is {4}
theorem complement_of_A_in_U : C_U_A = {4} := 
  by 
  sorry

end complement_of_A_in_U_l1972_197207


namespace exists_gcd_one_l1972_197218

theorem exists_gcd_one (p q r : ℤ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : Int.gcd p (Int.gcd q r) = 1) : ∃ a : ℤ, Int.gcd p (q + a * r) = 1 :=
sorry

end exists_gcd_one_l1972_197218


namespace OddPrimeDivisorCondition_l1972_197272

theorem OddPrimeDivisorCondition (n : ℕ) (h_pos : 0 < n) (h_div : ∀ d : ℕ, d ∣ n → d + 1 ∣ n + 1) : 
  ∃ p : ℕ, Prime p ∧ n = p ∧ ¬ Even p :=
sorry

end OddPrimeDivisorCondition_l1972_197272


namespace coin_tosses_l1972_197241

theorem coin_tosses (n : ℤ) (h : (1/2 : ℝ)^n = 0.125) : n = 3 :=
by
  sorry

end coin_tosses_l1972_197241


namespace problem_statement_l1972_197273

variable (a : ℕ → ℝ)

-- Defining sequences {b_n} and {c_n}
def b (n : ℕ) := a n - a (n + 2)
def c (n : ℕ) := a n + 2 * a (n + 1) + 3 * a (n + 2)

-- Defining that a sequence is arithmetic
def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) - seq n = d

-- Problem statement
theorem problem_statement :
  is_arithmetic a ↔ (is_arithmetic (c a) ∧ ∀ n, b a n ≤ b a (n + 1)) :=
sorry

end problem_statement_l1972_197273


namespace initial_positions_2048_l1972_197233

noncomputable def number_of_initial_positions (n : ℕ) : ℤ :=
  2 ^ n - 2

theorem initial_positions_2048 : number_of_initial_positions 2048 = 2 ^ 2048 - 2 :=
by
  sorry

end initial_positions_2048_l1972_197233


namespace find_central_angle_l1972_197237

-- We define the given conditions.
def radius : ℝ := 2
def area : ℝ := 8

-- We state the theorem that we need to prove.
theorem find_central_angle (R : ℝ) (A : ℝ) (hR : R = radius) (hA : A = area) :
  ∃ α : ℝ, α = 4 :=
by
  sorry

end find_central_angle_l1972_197237


namespace equation_of_parallel_line_l1972_197211

theorem equation_of_parallel_line : 
  ∃ l : ℝ, (∀ x y : ℝ, 2 * x - 3 * y + 8 = 0 ↔ l = 2 * x - 3 * y + 8) :=
sorry

end equation_of_parallel_line_l1972_197211


namespace consecutive_integers_equality_l1972_197274

theorem consecutive_integers_equality (n : ℕ) (h_eq : (n - 3) + (n - 2) + (n - 1) + n = (n + 1) + (n + 2) + (n + 3)) : n = 12 :=
by {
  sorry
}

end consecutive_integers_equality_l1972_197274


namespace swim_distance_downstream_l1972_197254

theorem swim_distance_downstream 
  (V_m V_s : ℕ) 
  (t d : ℕ) 
  (h1 : V_m = 9) 
  (h2 : t = 3) 
  (h3 : 3 * (V_m - V_s) = 18) : 
  t * (V_m + V_s) = 36 := 
by 
  sorry

end swim_distance_downstream_l1972_197254
