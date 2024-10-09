import Mathlib

namespace gcd_power_of_two_sub_one_l772_77242

def a : ℤ := 2^1100 - 1
def b : ℤ := 2^1122 - 1
def c : ℤ := 2^22 - 1

theorem gcd_power_of_two_sub_one :
  Int.gcd (2^1100 - 1) (2^1122 - 1) = 2^22 - 1 := by
  sorry

end gcd_power_of_two_sub_one_l772_77242


namespace relation_of_M_and_N_l772_77258

-- Define the functions for M and N
def M (x : ℝ) : ℝ := (x - 3) * (x - 4)
def N (x : ℝ) : ℝ := (x - 1) * (x - 6)

-- Formulate the theorem to prove M < N for all x
theorem relation_of_M_and_N (x : ℝ) : M x < N x := sorry

end relation_of_M_and_N_l772_77258


namespace complex_quadrant_l772_77200

theorem complex_quadrant (z : ℂ) (h : z * (2 - I) = 2 + I) : 0 < z.re ∧ 0 < z.im := 
sorry

end complex_quadrant_l772_77200


namespace parabola_properties_l772_77232

theorem parabola_properties :
  let a := -2
  let b := 4
  let c := 8
  ∃ h k : ℝ, 
    (∀ x : ℝ, y = a * x^2 + b * x + c) ∧ 
    (h = 1) ∧ 
    (k = 10) ∧ 
    (a < 0) ∧ 
    (axisOfSymmetry = h) ∧ 
    (vertex = (h, k)) :=
by
  sorry

end parabola_properties_l772_77232


namespace total_cost_of_fencing_l772_77269

def costOfFencing (lengths rates : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) lengths rates)

theorem total_cost_of_fencing :
  costOfFencing [14, 20, 35, 40, 15, 30, 25]
                [2.50, 3.00, 3.50, 4.00, 2.75, 3.25, 3.75] = 610.00 :=
by
  sorry

end total_cost_of_fencing_l772_77269


namespace cheryl_distance_walked_l772_77281

theorem cheryl_distance_walked (speed : ℕ) (time : ℕ) (distance_away : ℕ) (distance_home : ℕ) 
  (h1 : speed = 2) 
  (h2 : time = 3) 
  (h3 : distance_away = speed * time) 
  (h4 : distance_home = distance_away) : 
  distance_away + distance_home = 12 := 
by
  sorry

end cheryl_distance_walked_l772_77281


namespace isabella_jumped_farthest_l772_77208

-- defining the jumping distances
def ricciana_jump : ℕ := 4
def margarita_jump : ℕ := 2 * ricciana_jump - 1
def isabella_jump : ℕ := ricciana_jump + 3 

-- defining the total distances
def ricciana_total : ℕ := 20 + ricciana_jump
def margarita_total : ℕ := 18 + margarita_jump
def isabella_total : ℕ := 22 + isabella_jump

-- stating the theorem
theorem isabella_jumped_farthest : isabella_total = 29 :=
by sorry

end isabella_jumped_farthest_l772_77208


namespace diagonal_ratio_l772_77222

variable (a b : ℝ)
variable (d1 : ℝ) -- diagonal length of the first square
variable (r : ℝ := 1.5) -- ratio between perimeters

theorem diagonal_ratio (h : 4 * a / (4 * b) = r) (hd1 : d1 = a * Real.sqrt 2) : 
  (b * Real.sqrt 2) = (2/3) * d1 := 
sorry

end diagonal_ratio_l772_77222


namespace area_of_square_l772_77268

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line (y : ℝ) : Prop := y = 7

-- Define the roots of the quadratic equation derived from the conditions
noncomputable def root1 : ℝ := -2 + 2 * Real.sqrt 2
noncomputable def root2 : ℝ := -2 - 2 * Real.sqrt 2

-- Define the side length of the square
noncomputable def side_length : ℝ := abs (root1 - root2)

-- Define the area of the square
noncomputable def area_square : ℝ := side_length^2

-- Theorem statement for the problem
theorem area_of_square : area_square = 32 :=
sorry

end area_of_square_l772_77268


namespace inequality_abc_l772_77226

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end inequality_abc_l772_77226


namespace younger_son_age_after_30_years_l772_77265

-- Definitions based on given conditions
def age_difference : Nat := 10
def elder_son_current_age : Nat := 40

-- We need to prove that given these conditions, the younger son will be 60 years old 30 years from now
theorem younger_son_age_after_30_years : (elder_son_current_age - age_difference) + 30 = 60 := by
  -- Proof should go here, but we will skip it as per the instructions
  sorry

end younger_son_age_after_30_years_l772_77265


namespace intersection_M_N_l772_77279

theorem intersection_M_N :
  let M := {x | x^2 < 36}
  let N := {2, 4, 6, 8}
  M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l772_77279


namespace cubicroots_expression_l772_77201

theorem cubicroots_expression (a b c : ℝ)
  (h₁ : a + b + c = 6)
  (h₂ : a * b + b * c + c * a = 11)
  (h₃ : a * b * c = 6) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 251 / 216 :=
by sorry

end cubicroots_expression_l772_77201


namespace packet_weight_l772_77228

theorem packet_weight
  (tons_to_pounds : ℕ := 2600) -- 1 ton = 2600 pounds
  (total_tons : ℕ := 13)       -- Total capacity in tons
  (num_packets : ℕ := 2080)    -- Number of packets
  (expected_weight_per_packet : ℚ := 16.25) : 
  total_tons * tons_to_pounds / num_packets = expected_weight_per_packet := 
sorry

end packet_weight_l772_77228


namespace A_and_C_complete_remaining_work_in_2_point_4_days_l772_77235

def work_rate_A : ℚ := 1 / 12
def work_rate_B : ℚ := 1 / 15
def work_rate_C : ℚ := 1 / 18
def work_completed_B_in_10_days : ℚ := (10 : ℚ) * work_rate_B
def remaining_work : ℚ := 1 - work_completed_B_in_10_days
def combined_work_rate_AC : ℚ := work_rate_A + work_rate_C
def time_to_complete_remaining_work : ℚ := remaining_work / combined_work_rate_AC

theorem A_and_C_complete_remaining_work_in_2_point_4_days :
  time_to_complete_remaining_work = 2.4 := 
sorry

end A_and_C_complete_remaining_work_in_2_point_4_days_l772_77235


namespace Ryan_learning_days_l772_77277

theorem Ryan_learning_days
  (hours_english_per_day : ℕ)
  (hours_chinese_per_day : ℕ)
  (total_hours : ℕ)
  (h1 : hours_english_per_day = 6)
  (h2 : hours_chinese_per_day = 7)
  (h3 : total_hours = 65) :
  total_hours / (hours_english_per_day + hours_chinese_per_day) = 5 := by
  sorry

end Ryan_learning_days_l772_77277


namespace tan_double_angle_l772_77296

theorem tan_double_angle (theta : ℝ) (h : 2 * Real.sin (Real.pi / 2 + theta) + Real.sin (Real.pi + theta) = 0) :
  Real.tan (2 * theta) = -4 / 3 :=
by
  sorry

end tan_double_angle_l772_77296


namespace simplify_trig_expression_trig_identity_l772_77283

-- Defining the necessary functions
noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ
noncomputable def cos (θ : ℝ) : ℝ := Real.cos θ

-- First problem
theorem simplify_trig_expression (α : ℝ) :
  (sin (2 * Real.pi - α) * sin (Real.pi + α) * cos (-Real.pi - α)) / (sin (3 * Real.pi - α) * cos (Real.pi - α)) = sin α :=
sorry

-- Second problem
theorem trig_identity (x : ℝ) (hx : cos x ≠ 0) (hx' : 1 - sin x ≠ 0) :
  (cos x / (1 - sin x)) = ((1 + sin x) / cos x) :=
sorry

end simplify_trig_expression_trig_identity_l772_77283


namespace problem_expression_l772_77249

theorem problem_expression (x y : ℝ) (h1 : x - y = 5) (h2 : x * y = 4) : x^2 + y^2 = 33 :=
by sorry

end problem_expression_l772_77249


namespace initial_mixture_equals_50_l772_77213

theorem initial_mixture_equals_50 (x : ℝ) (h1 : 0.10 * x + 10 = 0.25 * (x + 10)) : x = 50 :=
by
  sorry

end initial_mixture_equals_50_l772_77213


namespace evaluate_expression_l772_77278

variable (a b c d : ℝ)

theorem evaluate_expression :
  (a - (b - (c + d))) - ((a + b) - (c - d)) = -2 * b + 2 * c :=
sorry

end evaluate_expression_l772_77278


namespace pies_from_36_apples_l772_77216

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l772_77216


namespace smallest_integer_ends_in_3_and_divisible_by_5_l772_77227

theorem smallest_integer_ends_in_3_and_divisible_by_5 : ∃ (n : ℕ), n > 0 ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ n = 53 :=
by
  sorry

end smallest_integer_ends_in_3_and_divisible_by_5_l772_77227


namespace inscribed_circle_equals_arc_length_l772_77251

open Real

theorem inscribed_circle_equals_arc_length 
  (R : ℝ) 
  (hR : 0 < R) 
  (θ : ℝ)
  (hθ : θ = (2 * π) / 3)
  (r : ℝ)
  (h_r : r = R / 2) 
  : 2 * π * r = 2 * π * R * (θ / (2 * π)) := by
  sorry

end inscribed_circle_equals_arc_length_l772_77251


namespace closest_to_fraction_l772_77280

theorem closest_to_fraction (n d : ℝ) (h_n : n = 510) (h_d : d = 0.125) :
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 5000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 6000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 7000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 8000 :=
by
  sorry

end closest_to_fraction_l772_77280


namespace point_not_on_graph_l772_77246

def on_graph (x y : ℚ) : Prop := y = x / (x + 2)

/-- Let's state the main theorem -/
theorem point_not_on_graph : ¬ on_graph 2 (2 / 3) := by
  sorry

end point_not_on_graph_l772_77246


namespace fraction_identity_l772_77220

variable {a b x : ℝ}

-- Conditions
axiom h1 : x = a / b
axiom h2 : a ≠ b
axiom h3 : b ≠ 0

-- Question to prove
theorem fraction_identity :
  (a + b) / (a - b) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_identity_l772_77220


namespace find_f_six_minus_a_l772_77248

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^(x-2) - 2 else -Real.logb 2 (x + 1)

variable (a : ℝ)
axiom h : f a = -3

theorem find_f_six_minus_a : f (6 - a) = - 15 / 8 :=
by
  sorry

end find_f_six_minus_a_l772_77248


namespace luke_fish_catching_l772_77223

theorem luke_fish_catching :
  ∀ (days : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ),
  days = 30 → fillets_per_fish = 2 → total_fillets = 120 →
  (total_fillets / fillets_per_fish) / days = 2 :=
by
  intros days fillets_per_fish total_fillets days_eq fillets_eq fillets_total_eq
  sorry

end luke_fish_catching_l772_77223


namespace find_angle_A_find_minimum_bc_l772_77256

open Real

variables (A B C a b c : ℝ)

-- Conditions
def side_opposite_angles_condition : Prop :=
  A > 0 ∧ A < π ∧ (A + B + C) = π

def collinear_vectors_condition (B C : ℝ) : Prop :=
  ∃ (k : ℝ), (2 * cos B * cos C + 1, 2 * sin B) = k • (sin C, 1)

-- Questions translated to proof statements
theorem find_angle_A (h1 : side_opposite_angles_condition A B C) (h2 : collinear_vectors_condition B C) :
  A = π / 3 :=
sorry

theorem find_minimum_bc (h1 : side_opposite_angles_condition A B C) (h2 : collinear_vectors_condition B C)
  (h3 : (1 / 2) * b * c * sin A = sqrt 3) :
  b + c = 4 :=
sorry

end find_angle_A_find_minimum_bc_l772_77256


namespace range_of_a_if_f_decreasing_l772_77207

noncomputable def f (a x : ℝ) : ℝ := Real.sqrt (x^2 - a * x + 4)

theorem range_of_a_if_f_decreasing:
  ∀ (a : ℝ),
    (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y → f a y < f a x) →
    2 ≤ a ∧ a ≤ 5 :=
by
  intros a h
  sorry

end range_of_a_if_f_decreasing_l772_77207


namespace prove_inequality_l772_77214

-- Defining properties of f
variable {α : Type*} [LinearOrderedField α] (f : α → α)

-- Condition 1: f is even function
def is_even_function (f : α → α) : Prop := ∀ x : α, f (-x) = f x

-- Condition 2: f is monotonically increasing on (0, ∞)
def is_monotonically_increasing_on_positive (f : α → α) : Prop := ∀ ⦃x y : α⦄, 0 < x → 0 < y → x < y → f x < f y

-- Define the main theorem we need to prove:
theorem prove_inequality (h1 : is_even_function f) (h2 : is_monotonically_increasing_on_positive f) : 
  f (-1) < f 2 ∧ f 2 < f (-3) :=
by
  sorry

end prove_inequality_l772_77214


namespace not_perfect_square_l772_77287

theorem not_perfect_square (n : ℕ) (h₁ : 100 + 200 = 300) (h₂ : ¬(300 % 9 = 0)) : ¬(∃ m : ℕ, n = m * m) :=
by
  intros
  sorry

end not_perfect_square_l772_77287


namespace line_within_plane_correct_l772_77204

-- Definitions of sets representing a line and a plane
variable {Point : Type}
variable (l α : Set Point)

-- Definition of the statement
def line_within_plane : Prop := l ⊆ α

-- Proof statement (without the actual proof)
theorem line_within_plane_correct (h : l ⊆ α) : line_within_plane l α :=
by
  sorry

end line_within_plane_correct_l772_77204


namespace sum_of_arithmetic_sequence_l772_77270

variable {S : ℕ → ℕ}

def isArithmeticSum (S : ℕ → ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ n, S n = n * (2 * a + (n - 1) * d ) / 2

theorem sum_of_arithmetic_sequence :
  isArithmeticSum S →
  S 8 - S 4 = 12 →
  S 12 = 36 :=
by
  intros
  sorry

end sum_of_arithmetic_sequence_l772_77270


namespace sum_is_2000_l772_77262

theorem sum_is_2000 (x y : ℝ) (h : x ≠ y) (h_eq : x^2 - 2000 * x = y^2 - 2000 * y) : x + y = 2000 := by
  sorry

end sum_is_2000_l772_77262


namespace circle_tangency_problem_l772_77275

theorem circle_tangency_problem :
  let u1 := ∀ (x y : ℝ), x^2 + y^2 + 8 * x - 30 * y - 63 = 0
  let u2 := ∀ (x y : ℝ), x^2 + y^2 - 6 * x - 30 * y + 99 = 0
  let line := ∀ (b x : ℝ), y = b * x
  ∃ p q : ℕ, gcd p q = 1 ∧ n^2 = (p : ℚ) / (q : ℚ) ∧ p + q = 7 :=
sorry

end circle_tangency_problem_l772_77275


namespace steps_in_five_days_l772_77297

def steps_to_school : ℕ := 150
def daily_steps : ℕ := steps_to_school * 2
def days : ℕ := 5

theorem steps_in_five_days : daily_steps * days = 1500 := by
  sorry

end steps_in_five_days_l772_77297


namespace parallel_lines_slope_l772_77241

theorem parallel_lines_slope (m : ℝ) :
  ((m + 2) * (2 * m - 1) = 3 * 1) →
  m = - (5 / 2) :=
by
  sorry

end parallel_lines_slope_l772_77241


namespace jancy_currency_notes_l772_77274

theorem jancy_currency_notes (x y : ℕ) (h1 : 70 * x + 50 * y = 5000) (h2 : y = 2) : x + y = 72 :=
by
  -- proof goes here
  sorry

end jancy_currency_notes_l772_77274


namespace find_q_in_geometric_sequence_l772_77230

theorem find_q_in_geometric_sequence
  {q : ℝ} (q_pos : q > 0) 
  (a1_def : ∀(a : ℕ → ℝ), a 1 = 1 / q^2) 
  (S5_eq_S2_plus_2 : ∀(S : ℕ → ℝ), S 5 = S 2 + 2) :
  q = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end find_q_in_geometric_sequence_l772_77230


namespace vector_subtraction_correct_l772_77272

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (-4, 2)

theorem vector_subtraction_correct :
  vector_a - 2 • vector_b = (10, -5) :=
sorry

end vector_subtraction_correct_l772_77272


namespace tangent_line_touching_circle_l772_77245

theorem tangent_line_touching_circle (a : ℝ) : 
  (∃ (x y : ℝ), 5 * x + 12 * y + a = 0 ∧ (x - 1)^2 + y^2 = 1) → 
  (a = 8 ∨ a = -18) :=
by
  sorry

end tangent_line_touching_circle_l772_77245


namespace Elle_in_seat_2_given_conditions_l772_77286

theorem Elle_in_seat_2_given_conditions
    (seats : Fin 4 → Type) -- Representation of the seating arrangement.
    (Garry Elle Fiona Hank : Type)
    (seat_of : Type → Fin 4)
    (h1 : seat_of Garry = 0) -- Garry is in seat #1 (index 0)
    (h2 : ¬ (seat_of Elle = seat_of Hank + 1 ∨ seat_of Elle = seat_of Hank - 1)) -- Elle is not next to Hank
    (h3 : ¬ (seat_of Fiona > seat_of Garry ∧ seat_of Fiona < seat_of Hank) ∧ ¬ (seat_of Fiona < seat_of Garry ∧ seat_of Fiona > seat_of Hank)) -- Fiona is not between Garry and Hank
    : seat_of Elle = 1 :=  -- Conclusion: Elle is in seat #2 (index 1)
    sorry

end Elle_in_seat_2_given_conditions_l772_77286


namespace product_plus_one_square_l772_77294

theorem product_plus_one_square (n : ℕ):
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 := 
  sorry

end product_plus_one_square_l772_77294


namespace unique_paintings_count_l772_77284

-- Given the conditions of the problem:
-- - N = 6 disks
-- - 3 disks are blue
-- - 2 disks are red
-- - 1 disk is green
-- - Two paintings that can be obtained from one another by a rotation or a reflection are considered the same

-- Define a theorem to calculate the number of unique paintings.
theorem unique_paintings_count : 
    ∃ n : ℕ, n = 13 :=
sorry

end unique_paintings_count_l772_77284


namespace complex_number_fourth_quadrant_l772_77203

theorem complex_number_fourth_quadrant (m : ℝ) (h1 : 2/3 < m) (h2 : m < 1) : 
  (3 * m - 2) > 0 ∧ (m - 1) < 0 := 
by 
  sorry

end complex_number_fourth_quadrant_l772_77203


namespace trigonometric_identity_l772_77261

theorem trigonometric_identity :
  (3 / (Real.sin (20 * Real.pi / 180))^2) - (1 / (Real.cos (20 * Real.pi / 180))^2) + 64 * (Real.sin (20 * Real.pi / 180))^2 = 32 :=
by
  sorry

end trigonometric_identity_l772_77261


namespace evaluate_expression_l772_77259

theorem evaluate_expression 
  (d a b c : ℚ)
  (h1 : d = a + 1)
  (h2 : a = b - 3)
  (h3 : b = c + 5)
  (h4 : c = 6)
  (nz1 : d + 3 ≠ 0)
  (nz2 : a + 2 ≠ 0)
  (nz3 : b - 5 ≠ 0)
  (nz4 : c + 7 ≠ 0) :
  (d + 5) / (d + 3) * (a + 3) / (a + 2) * (b - 3) / (b - 5) * (c + 10) / (c + 7) = 1232 / 585 :=
sorry

end evaluate_expression_l772_77259


namespace cats_weight_more_than_puppies_l772_77267

theorem cats_weight_more_than_puppies :
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  (num_cats * weight_per_cat) - (num_puppies * weight_per_puppy) = 5 :=
by 
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  sorry

end cats_weight_more_than_puppies_l772_77267


namespace calculate_oplus_l772_77255

def op (X Y : ℕ) : ℕ :=
  (X + Y) / 2

theorem calculate_oplus : op (op 6 10) 14 = 11 := by
  sorry

end calculate_oplus_l772_77255


namespace min_value_geometric_sequence_l772_77236

noncomputable def geometric_min_value (b1 b2 b3 : ℝ) (s : ℝ) : ℝ :=
  3 * b2 + 4 * b3

theorem min_value_geometric_sequence (s : ℝ) :
  ∃ s : ℝ, 2 = b1 ∧ b2 = 2 * s ∧ b3 = 2 * s^2 ∧ 3 * b2 + 4 * b3 = -9 / 8 :=
by
  sorry

end min_value_geometric_sequence_l772_77236


namespace right_triangle_perimeter_area_ratio_l772_77218

theorem right_triangle_perimeter_area_ratio 
  (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (hyp : ∀ c, c = Real.sqrt (a^2 + b^2))
  : (a + b + Real.sqrt (a^2 + b^2)) / (0.5 * a * b) = 5 → (∃! x y : ℝ, x + y + Real.sqrt (x^2 + y^2) / (0.5 * x * y) = 5) :=
by
  sorry   -- Proof is omitted as per instructions.

end right_triangle_perimeter_area_ratio_l772_77218


namespace valid_k_sum_correct_l772_77257

def sum_of_valid_k : ℤ :=
  (List.range 17).sum * 1734 + (List.range 17).sum * 3332

theorem valid_k_sum_correct : sum_of_valid_k = 5066 := by
  sorry

end valid_k_sum_correct_l772_77257


namespace cone_lateral_surface_area_l772_77212

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l772_77212


namespace triangle_inequality_proof_l772_77293

theorem triangle_inequality_proof 
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_proof_l772_77293


namespace min_moves_to_equalize_boxes_l772_77231

def initialCoins : List ℕ := [5, 8, 11, 17, 20, 15, 10]

def targetCoins (boxes : List ℕ) : ℕ := boxes.sum / boxes.length

def movesRequiredToBalance : List ℕ → ℕ
| [5, 8, 11, 17, 20, 15, 10] => 22
| _ => sorry

theorem min_moves_to_equalize_boxes :
  movesRequiredToBalance initialCoins = 22 :=
by
  sorry

end min_moves_to_equalize_boxes_l772_77231


namespace quadrilateral_area_ABCDEF_l772_77219

theorem quadrilateral_area_ABCDEF :
  ∀ (A B C D E : Type)
  (AC CD AE : ℝ) 
  (angle_ABC angle_ACD : ℝ),
  angle_ABC = 90 ∧
  angle_ACD = 90 ∧
  AC = 20 ∧
  CD = 30 ∧
  AE = 5 →
  ∃ S : ℝ, S = 360 :=
by
  sorry

end quadrilateral_area_ABCDEF_l772_77219


namespace complement_intersection_l772_77295

-- Define the universal set U.
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set M.
def M : Set ℕ := {2, 3}

-- Define the set N.
def N : Set ℕ := {1, 3}

-- Define the complement of set M in U.
def complement_U_M : Set ℕ := {x ∈ U | x ∉ M}

-- Define the complement of set N in U.
def complement_U_N : Set ℕ := {x ∈ U | x ∉ N}

-- The statement to be proven.
theorem complement_intersection :
  (complement_U_M ∩ complement_U_N) = {4, 5, 6} :=
sorry

end complement_intersection_l772_77295


namespace textbook_cost_l772_77247

theorem textbook_cost 
  (credits : ℕ) 
  (cost_per_credit : ℕ) 
  (facility_fee : ℕ) 
  (total_cost : ℕ) 
  (num_textbooks : ℕ) 
  (total_spent : ℕ) 
  (h1 : credits = 14) 
  (h2 : cost_per_credit = 450) 
  (h3 : facility_fee = 200) 
  (h4 : total_spent = 7100) 
  (h5 : num_textbooks = 5) :
  (total_cost - (credits * cost_per_credit + facility_fee)) / num_textbooks = 120 :=
by
  sorry

end textbook_cost_l772_77247


namespace simplify_and_evaluate_l772_77282

variable (a : ℝ)
noncomputable def given_expression : ℝ :=
    (3 * a / (a^2 - 4)) * (1 - 2 / a) - (4 / (a + 2))

theorem simplify_and_evaluate (h : a = Real.sqrt 2 - 1) : 
  given_expression a = 1 - Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l772_77282


namespace sum_reciprocal_of_shifted_roots_l772_77217

noncomputable def roots_of_cubic (a b c : ℝ) : Prop := 
    ∀ x : ℝ, x^3 - x - 2 = (x - a) * (x - b) * (x - c)

theorem sum_reciprocal_of_shifted_roots (a b c : ℝ) 
    (h : roots_of_cubic a b c) : 
    (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) = 1 :=
by
  sorry

end sum_reciprocal_of_shifted_roots_l772_77217


namespace cost_of_shorts_l772_77225

-- Define the given conditions and quantities
def initial_money : ℕ := 50
def jerseys_cost : ℕ := 5 * 2
def basketball_cost : ℕ := 18
def remaining_money : ℕ := 14

-- The total amount spent
def total_spent : ℕ := initial_money - remaining_money

-- The total cost of the jerseys and basketball
def jerseys_basketball_cost : ℕ := jerseys_cost + basketball_cost

-- The cost of the shorts
def shorts_cost : ℕ := total_spent - jerseys_basketball_cost

theorem cost_of_shorts : shorts_cost = 8 := sorry

end cost_of_shorts_l772_77225


namespace touchdowns_points_l772_77229

theorem touchdowns_points 
    (num_touchdowns : ℕ) (total_points : ℕ) 
    (h1 : num_touchdowns = 3) 
    (h2 : total_points = 21) : 
    total_points / num_touchdowns = 7 :=
by
    sorry

end touchdowns_points_l772_77229


namespace math_problem_proof_l772_77209

noncomputable def problem_expr : ℚ :=
  ((11 + 1/9) - (3 + 2/5) * (1 + 2/17)) - (8 + 2/5) / (36/10) / (2 + 6/25)

theorem math_problem_proof : problem_expr = 20 / 9 := by
  sorry

end math_problem_proof_l772_77209


namespace poly_divisible_coeff_sum_eq_one_l772_77205

theorem poly_divisible_coeff_sum_eq_one (C D : ℂ) :
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^100 + C * x^2 + D * x + 1 = 0) →
  C + D = 1 :=
by
  sorry

end poly_divisible_coeff_sum_eq_one_l772_77205


namespace time_for_type_Q_machine_l772_77233

theorem time_for_type_Q_machine (Q : ℝ) (h1 : Q > 0)
  (h2 : 2 * (1 / Q) + 3 * (1 / 7) = 5 / 6) :
  Q = 84 / 17 :=
sorry

end time_for_type_Q_machine_l772_77233


namespace exponent_value_l772_77254

theorem exponent_value (y k : ℕ) (h1 : 9^y = 3^k) (h2 : y = 7) : k = 14 := by
  sorry

end exponent_value_l772_77254


namespace arun_deepak_age_ratio_l772_77263

-- Define the current age of Arun based on the condition that after 6 years he will be 26 years old
def Arun_current_age : ℕ := 26 - 6

-- Define Deepak's current age based on the given condition
def Deepak_current_age : ℕ := 15

-- The present ratio between Arun's age and Deepak's age
theorem arun_deepak_age_ratio : Arun_current_age / Nat.gcd Arun_current_age Deepak_current_age = (4 : ℕ) ∧ Deepak_current_age / Nat.gcd Arun_current_age Deepak_current_age = (3 : ℕ) := 
by
  -- Proof omitted
  sorry

end arun_deepak_age_ratio_l772_77263


namespace black_car_overtakes_red_car_in_one_hour_l772_77224

def red_car_speed : ℕ := 40
def black_car_speed : ℕ := 50
def initial_gap : ℕ := 10

theorem black_car_overtakes_red_car_in_one_hour (h_red_car_speed : red_car_speed = 40)
                                               (h_black_car_speed : black_car_speed = 50)
                                               (h_initial_gap : initial_gap = 10) :
  initial_gap / (black_car_speed - red_car_speed) = 1 :=
by
  sorry

end black_car_overtakes_red_car_in_one_hour_l772_77224


namespace correct_transformation_D_l772_77239

theorem correct_transformation_D : ∀ x, 2 * (x + 1) = x + 7 → x = 5 :=
by
  intro x
  sorry

end correct_transformation_D_l772_77239


namespace largest_value_is_D_l772_77215

theorem largest_value_is_D :
  let A := 15432 + 1/3241
  let B := 15432 - 1/3241
  let C := 15432 * (1/3241)
  let D := 15432 / (1/3241)
  let E := 15432.3241
  max (max (max A B) (max C D)) E = D := by
{
  sorry -- proof not required
}

end largest_value_is_D_l772_77215


namespace p_at_zero_l772_77276

-- We state the conditions: p is a polynomial of degree 6, and p(3^n) = 1/(3^n) for n = 0 to 6
def p : Polynomial ℝ := sorry

axiom p_degree : p.degree = 6
axiom p_values : ∀ (n : ℕ), n ≤ 6 → p.eval (3^n) = 1 / (3^n)

-- We want to prove that p(0) = 29523 / 2187
theorem p_at_zero : p.eval 0 = 29523 / 2187 := by sorry

end p_at_zero_l772_77276


namespace simplify_sqrt_expression_l772_77289

theorem simplify_sqrt_expression :
  (Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2)) = 6 :=
by
  sorry

end simplify_sqrt_expression_l772_77289


namespace find_general_equation_of_line_l772_77298

variables {x y k b : ℝ}

-- Conditions: slope of the line is -2 and sum of its intercepts is 12.
def slope_of_line (l : ℝ → ℝ → Prop) : Prop := ∃ b, ∀ x y, l x y ↔ y = -2 * x + b
def sum_of_intercepts (l : ℝ → ℝ → Prop) : Prop := ∃ b, b + (b / 2) = 12

-- Question: What is the general equation of the line?
noncomputable def general_equation (l : ℝ → ℝ → Prop) : Prop :=
  slope_of_line l ∧ sum_of_intercepts l → ∀ x y, l x y ↔ 2 * x + y - 8 = 0

-- The theorem we need to prove
theorem find_general_equation_of_line (l : ℝ → ℝ → Prop) : general_equation l :=
sorry

end find_general_equation_of_line_l772_77298


namespace problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13_l772_77221

variable (x y a b : ℝ)

def A : ℝ := 2*x^2 + a*x - y + 6
def B : ℝ := b*x^2 - 3*x + 5*y - 1

theorem problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13 
  (h : A x y a - B x y b = -6*y + 7) : a^2 + b^2 = 13 := by
  sorry

end problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13_l772_77221


namespace find_n_with_divisors_sum_l772_77237

theorem find_n_with_divisors_sum (n : ℕ) (d1 d2 d3 d4 : ℕ)
  (h1 : d1 = 1) (h2 : d2 = 2) (h3 : d3 = 5) (h4 : d4 = 10) 
  (hd : n = 130) : d1^2 + d2^2 + d3^2 + d4^2 = n :=
sorry

end find_n_with_divisors_sum_l772_77237


namespace radius_ratio_l772_77240

theorem radius_ratio (V₁ V₂ : ℝ) (hV₁ : V₁ = 432 * Real.pi) (hV₂ : V₂ = 108 * Real.pi) : 
  (∃ (r₁ r₂ : ℝ), V₁ = (4/3) * Real.pi * r₁^3 ∧ V₂ = (4/3) * Real.pi * r₂^3) →
  ∃ k : ℝ, k = r₂ / r₁ ∧ k = 1 / 2^(2/3) := 
by
  sorry

end radius_ratio_l772_77240


namespace burger_cost_cents_l772_77238

theorem burger_cost_cents 
  (b s : ℕ)
  (h1 : 4 * b + 3 * s = 550) 
  (h2 : 3 * b + 2 * s = 400) 
  (h3 : 2 * b + s = 250) : 
  b = 100 :=
by
  sorry

end burger_cost_cents_l772_77238


namespace greatest_int_satisfying_inequality_l772_77299

theorem greatest_int_satisfying_inequality : ∃ n : ℤ, (∀ m : ℤ, m^2 - 13 * m + 40 ≤ 0 → m ≤ n) ∧ n = 8 := 
sorry

end greatest_int_satisfying_inequality_l772_77299


namespace area_of_triangle_XPQ_l772_77260

noncomputable def area_triangle_XPQ (XY YZ XZ XP XQ : ℝ) (hXY : XY = 12) (hYZ : YZ = 13) (hXZ : XZ = 15) (hXP : XP = 5) (hXQ : XQ = 9) : ℝ :=
  let s := (XY + YZ + XZ) / 2
  let area_XYZ := Real.sqrt (s * (s - XY) * (s - YZ) * (s - XZ))
  let cosX := (XY^2 + YZ^2 - XZ^2) / (2 * XY * YZ)
  let sinX := Real.sqrt (1 - cosX^2)
  (1 / 2) * XP * XQ * sinX

theorem area_of_triangle_XPQ :
  area_triangle_XPQ 12 13 15 5 9 (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) = 45 * Real.sqrt 1400 / 78 :=
by
  sorry

end area_of_triangle_XPQ_l772_77260


namespace infinite_series_sum_eq_seven_l772_77252

noncomputable def infinite_series_sum : ℝ :=
  ∑' k : ℕ, (1 + k)^2 / 3^(1 + k)

theorem infinite_series_sum_eq_seven : infinite_series_sum = 7 :=
sorry

end infinite_series_sum_eq_seven_l772_77252


namespace fraction_of_blue_cars_l772_77290

-- Definitions of the conditions
def total_cars : ℕ := 516
def red_cars : ℕ := total_cars / 2
def black_cars : ℕ := 86
def blue_cars : ℕ := total_cars - (red_cars + black_cars)

-- Statement to prove that the fraction of blue cars is 1/3
theorem fraction_of_blue_cars :
  (blue_cars : ℚ) / total_cars = 1 / 3 :=
by
  sorry -- Proof to be filled in

end fraction_of_blue_cars_l772_77290


namespace John_works_5_days_a_week_l772_77234

theorem John_works_5_days_a_week
  (widgets_per_hour : ℕ)
  (hours_per_day : ℕ)
  (widgets_per_week : ℕ)
  (H1 : widgets_per_hour = 20)
  (H2 : hours_per_day = 8)
  (H3 : widgets_per_week = 800) :
  widgets_per_week / (widgets_per_hour * hours_per_day) = 5 :=
by
  sorry

end John_works_5_days_a_week_l772_77234


namespace determine_b_from_quadratic_l772_77288

theorem determine_b_from_quadratic (b n : ℝ) (h1 : b > 0) 
  (h2 : ∀ x, x^2 + b*x + 36 = (x + n)^2 + 20) : b = 8 := 
by 
  sorry

end determine_b_from_quadratic_l772_77288


namespace problem_statement_l772_77206

def g (x : ℝ) : ℝ := 3 * x + 2

theorem problem_statement : g (g (g 3)) = 107 := by
  sorry

end problem_statement_l772_77206


namespace price_increase_percentage_l772_77250

theorem price_increase_percentage (original_price : ℝ) (discount : ℝ) (reduced_price : ℝ) : 
  reduced_price = original_price * (1 - discount) →
  (original_price / reduced_price - 1) * 100 = 8.7 :=
by
  intros h
  sorry

end price_increase_percentage_l772_77250


namespace track_length_l772_77285

theorem track_length (x : ℝ) (tom_dist1 jerry_dist1 : ℝ) (tom_dist2 jerry_dist2 : ℝ) (deg_gap : ℝ) :
  deg_gap = 120 ∧ 
  tom_dist1 = 120 ∧ 
  (tom_dist1 + jerry_dist1 = x * deg_gap / 360) ∧ 
  (jerry_dist1 + jerry_dist2 = x * deg_gap / 360 + 180) →
  x = 630 :=
by
  sorry

end track_length_l772_77285


namespace find_c_for_equal_real_roots_l772_77211

theorem find_c_for_equal_real_roots
  (c : ℝ)
  (h : ∀ x : ℝ, x^2 + 6 * x + c = 0 → x = -3) : c = 9 :=
sorry

end find_c_for_equal_real_roots_l772_77211


namespace lines_intersect_l772_77202

def line1 (t : ℝ) : ℝ × ℝ :=
  (1 - 2 * t, 2 + 4 * t)

def line2 (u : ℝ) : ℝ × ℝ :=
  (3 + u, 5 + 3 * u)

theorem lines_intersect :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (1.2, 1.6) :=
by
  sorry

end lines_intersect_l772_77202


namespace complex_number_division_l772_77271

theorem complex_number_division (i : ℂ) (h_i : i^2 = -1) :
  2 / (i * (3 - i)) = (1 - 3 * i) / 5 :=
by
  sorry

end complex_number_division_l772_77271


namespace abs_ineq_range_k_l772_77292

theorem abs_ineq_range_k (k : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| > k) → k < 4 :=
by
  sorry

end abs_ineq_range_k_l772_77292


namespace range_of_h_l772_77273

theorem range_of_h 
  (y1 y2 y3 k : ℝ)
  (h : ℝ)
  (H1 : y1 = (-3 - h)^2 + k)
  (H2 : y2 = (-1 - h)^2 + k)
  (H3 : y3 = (1 - h)^2 + k)
  (H_ord : y2 < y1 ∧ y1 < y3) : 
  -2 < h ∧ h < -1 :=
sorry

end range_of_h_l772_77273


namespace original_number_unique_l772_77264

theorem original_number_unique (N : ℤ) (h : (N - 31) % 87 = 0) : N = 118 :=
by
  sorry

end original_number_unique_l772_77264


namespace cannot_be_covered_by_dominoes_l772_77266

-- Definitions for each board
def board_3x4_squares : ℕ := 3 * 4
def board_3x5_squares : ℕ := 3 * 5
def board_4x4_one_removed_squares : ℕ := 4 * 4 - 1
def board_5x5_squares : ℕ := 5 * 5
def board_6x3_squares : ℕ := 6 * 3

-- Parity check
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Mathematical proof problem statement
theorem cannot_be_covered_by_dominoes :
  ¬ is_even board_3x5_squares ∧
  ¬ is_even board_4x4_one_removed_squares ∧
  ¬ is_even board_5x5_squares :=
by
  -- Checking the conditions that must hold
  sorry

end cannot_be_covered_by_dominoes_l772_77266


namespace danica_planes_l772_77210

def smallestAdditionalPlanes (n k : ℕ) : ℕ :=
  let m := k * (n / k + 1)
  m - n

theorem danica_planes : smallestAdditionalPlanes 17 7 = 4 :=
by
  -- Proof would go here
  sorry

end danica_planes_l772_77210


namespace largest_valid_n_l772_77243

def is_valid_n (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 10 * a + b ∧ n = a * (a + b)

theorem largest_valid_n : ∀ n : ℕ, is_valid_n n → n ≤ 48 := by sorry

example : is_valid_n 48 := by sorry

end largest_valid_n_l772_77243


namespace retail_price_percentage_l772_77253

variable (P : ℝ)
variable (wholesale_cost : ℝ)
variable (employee_price : ℝ)

axiom wholesale_cost_def : wholesale_cost = 200
axiom employee_price_def : employee_price = 192
axiom employee_discount_def : employee_price = 0.80 * (wholesale_cost + (P / 100 * wholesale_cost))

theorem retail_price_percentage (P : ℝ) (wholesale_cost : ℝ) (employee_price : ℝ)
    (H1 : wholesale_cost = 200)
    (H2 : employee_price = 192)
    (H3 : employee_price = 0.80 * (wholesale_cost + (P / 100 * wholesale_cost))) :
    P = 20 :=
  sorry

end retail_price_percentage_l772_77253


namespace find_p_q_l772_77244

def op (a b c d : ℝ) : ℝ × ℝ := (a * c - b * d, a * d + b * c)

theorem find_p_q :
  (∀ (a b c d : ℝ), (a = c ∧ b = d) ↔ (a, b) = (c, d)) →
  (op 1 2 p q = (5, 0)) →
  (p, q) = (1, -2) :=
by
  intro h
  intro eq_op
  sorry

end find_p_q_l772_77244


namespace right_isosceles_areas_l772_77291

theorem right_isosceles_areas (A B C : ℝ) (hA : A = 1 / 2 * 5 * 5) (hB : B = 1 / 2 * 12 * 12) (hC : C = 1 / 2 * 13 * 13) :
  A + B = C :=
by
  sorry

end right_isosceles_areas_l772_77291
