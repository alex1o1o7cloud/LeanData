import Mathlib

namespace NUMINAMATH_GPT_runner_injury_point_l1504_150426

-- Define the initial setup conditions
def total_distance := 40
def second_half_time := 10
def first_half_additional_time := 5

-- Prove that given the conditions, the runner injured her foot at 20 miles.
theorem runner_injury_point : 
  ∃ (d v : ℝ), (d = 5 * v) ∧ (total_distance - d = 5 * v) ∧ (10 = second_half_time) ∧ (first_half_additional_time = 5) ∧ (d = 20) :=
by
  sorry

end NUMINAMATH_GPT_runner_injury_point_l1504_150426


namespace NUMINAMATH_GPT_figure_perimeter_equals_26_l1504_150436

noncomputable def rectangle_perimeter : ℕ := 26

def figure_arrangement (width height : ℕ) : Prop :=
width = 2 ∧ height = 1

theorem figure_perimeter_equals_26 {width height : ℕ} (h : figure_arrangement width height) :
  rectangle_perimeter = 26 :=
by
  sorry

end NUMINAMATH_GPT_figure_perimeter_equals_26_l1504_150436


namespace NUMINAMATH_GPT_correct_population_l1504_150484

variable (P : ℕ) (S : ℕ)
variable (math_scores : ℕ → Type)

-- Assume P is the total number of students who took the exam.
-- Let math_scores(P) represent the math scores of P students.

def population_data (P : ℕ) : Prop := 
  P = 50000

def sample_data (S : ℕ) : Prop :=
  S = 2000

theorem correct_population (P : ℕ) (S : ℕ) (math_scores : ℕ → Type)
  (hP : population_data P) (hS : sample_data S) : 
  math_scores P = math_scores 50000 :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_population_l1504_150484


namespace NUMINAMATH_GPT_foil_covered_prism_width_l1504_150447

def inner_prism_dimensions (l w h : ℕ) : Prop :=
  w = 2 * l ∧ w = 2 * h ∧ l * w * h = 128

def outer_prism_width (l w h outer_width : ℕ) : Prop :=
  inner_prism_dimensions l w h ∧ outer_width = w + 2

theorem foil_covered_prism_width (l w h outer_width : ℕ) (h_inner_prism : inner_prism_dimensions l w h) :
  outer_prism_width l w h outer_width → outer_width = 10 :=
by
  intro h_outer_prism
  obtain ⟨h_w_eq, h_w_eq_2, h_volume_eq⟩ := h_inner_prism
  obtain ⟨_, h_outer_width_eq⟩ := h_outer_prism
  sorry

end NUMINAMATH_GPT_foil_covered_prism_width_l1504_150447


namespace NUMINAMATH_GPT_maximum_sin_C_in_triangle_l1504_150438

theorem maximum_sin_C_in_triangle 
  (A B C : ℝ)
  (h1 : A + B + C = π) 
  (h2 : 1 / Real.tan A + 1 / Real.tan B = 6 / Real.tan C) : 
  Real.sin C = Real.sqrt 15 / 4 :=
sorry

end NUMINAMATH_GPT_maximum_sin_C_in_triangle_l1504_150438


namespace NUMINAMATH_GPT_triangle_inequality_l1504_150480

-- Define the lengths of the existing sticks
def a := 4
def b := 7

-- Define the list of potential third sticks
def potential_sticks := [3, 6, 11, 12]

-- Define the triangle inequality conditions
def valid_length (c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Prove that the valid length satisfying these conditions is 6
theorem triangle_inequality : ∃ c ∈ potential_sticks, valid_length c ∧ c = 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1504_150480


namespace NUMINAMATH_GPT_maximum_xy_l1504_150446

theorem maximum_xy (x y : ℝ) (h : x^2 + 2 * y^2 - 2 * x * y = 4) : 
  xy ≤ 2 * (Float.sqrt 2) + 2 :=
sorry

end NUMINAMATH_GPT_maximum_xy_l1504_150446


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1504_150445

theorem sum_of_squares_of_roots :
  ∀ r1 r2 : ℝ, (r1 + r2 = 14) ∧ (r1 * r2 = 8) → (r1^2 + r2^2 = 180) := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1504_150445


namespace NUMINAMATH_GPT_avg_abc_l1504_150419

variable (A B C : ℕ)

-- Conditions
def avg_ac : Prop := (A + C) / 2 = 29
def age_b : Prop := B = 26

-- Theorem stating the average age of a, b, and c
theorem avg_abc (h1 : avg_ac A C) (h2 : age_b B) : (A + B + C) / 3 = 28 := by
  sorry

end NUMINAMATH_GPT_avg_abc_l1504_150419


namespace NUMINAMATH_GPT_perfect_squares_diff_consecutive_l1504_150439

theorem perfect_squares_diff_consecutive (h1 : ∀ a : ℕ, a^2 < 1000000 → ∃ b : ℕ, a^2 = (b + 1)^2 - b^2) : 
  (∃ n : ℕ, n = 500) := 
by 
  sorry

end NUMINAMATH_GPT_perfect_squares_diff_consecutive_l1504_150439


namespace NUMINAMATH_GPT_olivia_probability_l1504_150443

noncomputable def total_outcomes (n m : ℕ) : ℕ := Nat.choose n m

noncomputable def favorable_outcomes : ℕ :=
  let choose_three_colors := total_outcomes 4 3
  let choose_one_for_pair := total_outcomes 3 1
  let choose_socks :=
    (total_outcomes 3 2) * (total_outcomes 3 1) * (total_outcomes 3 1)
  choose_three_colors * choose_one_for_pair * choose_socks

def probability (n m : ℕ) : ℚ := n / m

theorem olivia_probability :
  probability favorable_outcomes (total_outcomes 12 5) = 9 / 22 :=
by
  sorry

end NUMINAMATH_GPT_olivia_probability_l1504_150443


namespace NUMINAMATH_GPT_probability_of_purple_l1504_150451

def total_faces := 10
def purple_faces := 3

theorem probability_of_purple : (purple_faces : ℚ) / (total_faces : ℚ) = 3 / 10 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_purple_l1504_150451


namespace NUMINAMATH_GPT_find_x_pow_y_l1504_150461

theorem find_x_pow_y (x y : ℝ) : |x + 2| + (y - 3)^2 = 0 → x ^ y = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_x_pow_y_l1504_150461


namespace NUMINAMATH_GPT_books_not_sold_l1504_150470

-- Definitions capturing the conditions
variable (B : ℕ)
variable (books_price : ℝ := 3.50)
variable (total_received : ℝ := 252)

-- Lean statement to capture the proof problem
theorem books_not_sold (h : (2 / 3 : ℝ) * B * books_price = total_received) :
  B / 3 = 36 :=
by
  sorry

end NUMINAMATH_GPT_books_not_sold_l1504_150470


namespace NUMINAMATH_GPT_probability_of_defective_on_second_draw_l1504_150458

-- Define the conditions
variable (batch_size : ℕ) (defective_items : ℕ) (good_items : ℕ)
variable (first_draw_good : Prop)
variable (without_replacement : Prop)

-- Given conditions
def batch_conditions : Prop :=
  batch_size = 10 ∧ defective_items = 3 ∧ good_items = 7 ∧ first_draw_good ∧ without_replacement

-- The desired probability as a proof
theorem probability_of_defective_on_second_draw
  (h : batch_conditions batch_size defective_items good_items first_draw_good without_replacement) : 
  (3 / 9 : ℝ) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_of_defective_on_second_draw_l1504_150458


namespace NUMINAMATH_GPT_ratio_of_boys_l1504_150488

theorem ratio_of_boys 
  (p : ℚ) 
  (h : p = (3/4) * (1 - p)) : 
  p = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_l1504_150488


namespace NUMINAMATH_GPT_bananas_per_box_l1504_150466

theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) (h1 : total_bananas = 40) (h2 : num_boxes = 8) :
  total_bananas / num_boxes = 5 := by
  sorry

end NUMINAMATH_GPT_bananas_per_box_l1504_150466


namespace NUMINAMATH_GPT_new_energy_vehicles_l1504_150401

-- Given conditions
def conditions (a b : ℕ) : Prop :=
  3 * a + 2 * b = 95 ∧ 4 * a + 1 * b = 110

-- Given prices
def purchase_prices : Prop :=
  ∃ a b, conditions a b ∧ a = 25 ∧ b = 10

-- Total value condition for different purchasing plans
def purchase_plans (m n : ℕ) : Prop :=
  25 * m + 10 * n = 250 ∧ m > 0 ∧ n > 0

-- Number of different purchasing plans
def num_purchase_plans : Prop :=
  ∃ num_plans, num_plans = 4

-- Profit calculation for a given plan
def profit (m n : ℕ) : ℕ :=
  12 * m + 8 * n

-- Maximum profit condition
def max_profit : Prop :=
  ∃ max_profit, max_profit = 184 ∧ ∀ (m n : ℕ), purchase_plans m n → profit m n ≤ 184

-- Main theorem
theorem new_energy_vehicles : purchase_prices ∧ num_purchase_plans ∧ max_profit :=
  sorry

end NUMINAMATH_GPT_new_energy_vehicles_l1504_150401


namespace NUMINAMATH_GPT_triangle_isosceles_or_right_angled_l1504_150413

theorem triangle_isosceles_or_right_angled
  (β γ : ℝ)
  (h : Real.tan β * Real.sin γ ^ 2 = Real.tan γ * Real.sin β ^ 2) :
  (β = γ ∨ β + γ = Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_triangle_isosceles_or_right_angled_l1504_150413


namespace NUMINAMATH_GPT_vanya_faster_speed_l1504_150433

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end NUMINAMATH_GPT_vanya_faster_speed_l1504_150433


namespace NUMINAMATH_GPT_gnomes_telling_the_truth_l1504_150412

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end NUMINAMATH_GPT_gnomes_telling_the_truth_l1504_150412


namespace NUMINAMATH_GPT_books_sold_to_used_bookstore_l1504_150440

-- Conditions
def initial_books := 72
def books_from_club := 1 * 12
def books_from_bookstore := 5
def books_from_yardsales := 2
def books_from_daughter := 1
def books_from_mother := 4
def books_donated := 12
def books_end_of_year := 81

-- Proof problem
theorem books_sold_to_used_bookstore :
  initial_books
  + books_from_club
  + books_from_bookstore
  + books_from_yardsales
  + books_from_daughter
  + books_from_mother
  - books_donated
  - books_end_of_year
  = 3 := by
  -- calculation omitted
  sorry

end NUMINAMATH_GPT_books_sold_to_used_bookstore_l1504_150440


namespace NUMINAMATH_GPT_vector_equation_l1504_150496

variable {V : Type} [AddCommGroup V]

variables (A B C : V)

theorem vector_equation :
  (B - A) - 2 • (C - A) + (C - B) = (A - C) :=
by
  sorry

end NUMINAMATH_GPT_vector_equation_l1504_150496


namespace NUMINAMATH_GPT_seeds_planted_l1504_150472

theorem seeds_planted (seeds_per_bed : ℕ) (beds : ℕ) (total_seeds : ℕ) :
  seeds_per_bed = 10 → beds = 6 → total_seeds = seeds_per_bed * beds → total_seeds = 60 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_seeds_planted_l1504_150472


namespace NUMINAMATH_GPT_robins_hair_length_l1504_150457

-- Conditions:
-- Robin cut off 4 inches of his hair.
-- After cutting, his hair is now 13 inches long.
-- Question: How long was Robin's hair before he cut it? Answer: 17 inches

theorem robins_hair_length (current_length : ℕ) (cut_length : ℕ) (initial_length : ℕ) 
  (h_cut_length : cut_length = 4) 
  (h_current_length : current_length = 13) 
  (h_initial : initial_length = current_length + cut_length) :
  initial_length = 17 :=
sorry

end NUMINAMATH_GPT_robins_hair_length_l1504_150457


namespace NUMINAMATH_GPT_find_k_value_l1504_150495

theorem find_k_value (k : ℝ) : 
  5 + ∑' n : ℕ, (5 + k + n) / 5^(n+1) = 12 → k = 18.2 :=
by 
  sorry

end NUMINAMATH_GPT_find_k_value_l1504_150495


namespace NUMINAMATH_GPT_translate_triangle_vertex_l1504_150483

theorem translate_triangle_vertex 
    (a b : ℤ) 
    (hA : (-3, a) = (-1, 2) + (-2, a - 2)) 
    (hB : (b, 3) = (1, -1) + (b - 1, 4)) :
    (2 + (-3 - (-1)), 1 + (3 - (-1))) = (0, 5) :=
by 
  -- proof is omitted as instructed
  sorry

end NUMINAMATH_GPT_translate_triangle_vertex_l1504_150483


namespace NUMINAMATH_GPT_machines_job_completion_time_l1504_150404

theorem machines_job_completion_time (t : ℕ) 
  (hR_rate : ∀ t, 1 / t = 1 / 216) 
  (hS_rate : ∀ t, 1 / t = 1 / 216) 
  (same_num_machines : ∀ R S, R = 9 ∧ S = 9) 
  (total_time : 12 = 12) 
  (jobs_completed : 1 = (18 / t) * 12) : 
  t = 216 := 
sorry

end NUMINAMATH_GPT_machines_job_completion_time_l1504_150404


namespace NUMINAMATH_GPT_find_x_for_sin_cos_l1504_150453

theorem find_x_for_sin_cos (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x + Real.cos x = Real.sqrt 2) : x = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_find_x_for_sin_cos_l1504_150453


namespace NUMINAMATH_GPT_sock_pairs_l1504_150402

open Nat

theorem sock_pairs (r g y : ℕ) (hr : r = 5) (hg : g = 6) (hy : y = 4) :
  (choose r 2) + (choose g 2) + (choose y 2) = 31 :=
by
  rw [hr, hg, hy]
  norm_num
  sorry

end NUMINAMATH_GPT_sock_pairs_l1504_150402


namespace NUMINAMATH_GPT_max_value_x_plus_y_max_value_x_plus_y_achieved_l1504_150471

theorem max_value_x_plus_y (x y : ℝ) (h1: x^2 + y^2 = 100) (h2: x * y = 40) : x + y ≤ 6 * Real.sqrt 5 :=
by
  sorry

theorem max_value_x_plus_y_achieved (x y : ℝ) (h1: x^2 + y^2 = 100) (h2: x * y = 40) : ∃ x y, x + y = 6 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_max_value_x_plus_y_max_value_x_plus_y_achieved_l1504_150471


namespace NUMINAMATH_GPT_cats_on_edges_l1504_150428

variables {W1 W2 B1 B2 : ℕ}  -- representing positions of cats on a line

def distance_from_white_to_black_sum_1 (a1 a2 : ℕ) : Prop := a1 + a2 = 4
def distance_from_white_to_black_sum_2 (b1 b2 : ℕ) : Prop := b1 + b2 = 8
def distance_from_black_to_white_sum_1 (b1 a1 : ℕ) : Prop := b1 + a1 = 9
def distance_from_black_to_white_sum_2 (b2 a2 : ℕ) : Prop := b2 + a2 = 3

theorem cats_on_edges
  (a1 a2 b1 b2 : ℕ)
  (h1 : distance_from_white_to_black_sum_1 a1 a2)
  (h2 : distance_from_white_to_black_sum_2 b1 b2)
  (h3 : distance_from_black_to_white_sum_1 b1 a1)
  (h4 : distance_from_black_to_white_sum_2 b2 a2) :
  (a1 = 2) ∧ (a2 = 2) ∧ (b1 = 7) ∧ (b2 = 1) ∧ (W1 = min W1 W2) ∧ (B2 = max B1 B2) :=
sorry

end NUMINAMATH_GPT_cats_on_edges_l1504_150428


namespace NUMINAMATH_GPT_sasha_made_50_muffins_l1504_150423

/-- 
Sasha made some chocolate muffins for her school bake sale fundraiser. Melissa made 4 times as many 
muffins as Sasha, and Tiffany made half of Sasha and Melissa's total number of muffins. They 
contributed $900 to the fundraiser by selling muffins at $4 each. Prove that Sasha made 50 muffins.
-/
theorem sasha_made_50_muffins 
  (S : ℕ)
  (Melissa_made : ℕ := 4 * S)
  (Tiffany_made : ℕ := (1 / 2) * (S + Melissa_made))
  (Total_muffins : ℕ := S + Melissa_made + Tiffany_made)
  (total_income : ℕ := 900)
  (price_per_muffin : ℕ := 4)
  (muffins_sold : ℕ := total_income / price_per_muffin)
  (eq_muffins_sold : Total_muffins = muffins_sold) : 
  S = 50 := 
by sorry

end NUMINAMATH_GPT_sasha_made_50_muffins_l1504_150423


namespace NUMINAMATH_GPT_James_balloons_correct_l1504_150411

def Amy_balloons : ℕ := 101
def diff_balloons : ℕ := 131
def James_balloons (a : ℕ) (d : ℕ) : ℕ := a + d

theorem James_balloons_correct : James_balloons Amy_balloons diff_balloons = 232 :=
by
  sorry

end NUMINAMATH_GPT_James_balloons_correct_l1504_150411


namespace NUMINAMATH_GPT_valid_ATM_passwords_l1504_150449

theorem valid_ATM_passwords : 
  let total_passwords := 10^4
  let restricted_passwords := 10
  total_passwords - restricted_passwords = 9990 :=
by
  sorry

end NUMINAMATH_GPT_valid_ATM_passwords_l1504_150449


namespace NUMINAMATH_GPT_smallest_d_for_inverse_l1504_150444

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 1

theorem smallest_d_for_inverse :
  ∃ d : ℝ, (∀ x1 x2 : ℝ, x1 ≠ x2 → (d ≤ x1) → (d ≤ x2) → g x1 ≠ g x2) ∧ d = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_d_for_inverse_l1504_150444


namespace NUMINAMATH_GPT_tax_rate_as_percent_l1504_150431

def TaxAmount (amount : ℝ) : Prop := amount = 82
def BaseAmount (amount : ℝ) : Prop := amount = 100

theorem tax_rate_as_percent {tax_amt base_amt : ℝ} 
  (h_tax : TaxAmount tax_amt) (h_base : BaseAmount base_amt) : 
  (tax_amt / base_amt) * 100 = 82 := 
by 
  sorry

end NUMINAMATH_GPT_tax_rate_as_percent_l1504_150431


namespace NUMINAMATH_GPT_vertical_asymptote_condition_l1504_150489

theorem vertical_asymptote_condition (c : ℝ) :
  (∀ x : ℝ, (x = 3 ∨ x = -6) → (x^2 - x + c = 0)) → 
  (c = -6 ∨ c = -42) :=
by
  sorry

end NUMINAMATH_GPT_vertical_asymptote_condition_l1504_150489


namespace NUMINAMATH_GPT_parallelogram_area_l1504_150462

open Matrix

noncomputable def u : Fin 2 → ℝ := ![7, -4]
noncomputable def z : Fin 2 → ℝ := ![8, -1]

theorem parallelogram_area :
  let matrix := ![u, z]
  |det (of fun (i j : Fin 2) => (matrix i) j)| = 25 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1504_150462


namespace NUMINAMATH_GPT_pentagon_rectangle_ratio_l1504_150452

theorem pentagon_rectangle_ratio (p w : ℝ) 
    (pentagon_perimeter : 5 * p = 30) 
    (rectangle_perimeter : ∃ l, 2 * w + 2 * l = 30 ∧ l = 2 * w) : 
    p / w = 6 / 5 := 
by
  sorry

end NUMINAMATH_GPT_pentagon_rectangle_ratio_l1504_150452


namespace NUMINAMATH_GPT_rectangle_width_decrease_proof_l1504_150430

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_width_decrease_proof_l1504_150430


namespace NUMINAMATH_GPT_integer_values_sides_triangle_l1504_150415

theorem integer_values_sides_triangle (x : ℝ) (hx_pos : x > 0) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) : 
    (∃ (n : ℤ), ∃ (hn : 0 < n) (hn1 : (n : ℝ) = x) (hn2 : 26 ≤ n) (hn3 : n ≤ 54), 
    ∀ (y : ℤ), (26 ≤ y ∧ y ≤ 54) → (∃ (m : ℤ), y = 26 + m ∧ m < 29 ∧ m ≥ 0)) := 
sorry

end NUMINAMATH_GPT_integer_values_sides_triangle_l1504_150415


namespace NUMINAMATH_GPT_problem_l1504_150469

noncomputable def f (x : ℝ) := Real.log x + (x + 1) / x

noncomputable def g (x : ℝ) := x - 1/x - 2 * Real.log x

theorem problem 
  (x : ℝ) (hx : x > 0) (hxn1 : x ≠ 1) :
  f x > (x + 1) * Real.log x / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1504_150469


namespace NUMINAMATH_GPT_monthly_income_of_P_l1504_150491

-- Define variables and assumptions
variables (P Q R : ℝ)
axiom avg_P_Q : (P + Q) / 2 = 5050
axiom avg_Q_R : (Q + R) / 2 = 6250
axiom avg_P_R : (P + R) / 2 = 5200

-- Prove that the monthly income of P is 4000
theorem monthly_income_of_P : P = 4000 :=
by
  sorry

end NUMINAMATH_GPT_monthly_income_of_P_l1504_150491


namespace NUMINAMATH_GPT_EF_side_length_l1504_150467

def square_side_length (n : ℝ) : Prop := n = 10

def distance_parallel_line (d : ℝ) : Prop := d = 6.5

def area_difference (a : ℝ) : Prop := a = 13.8

theorem EF_side_length :
  ∃ (x : ℝ), square_side_length 10 ∧ distance_parallel_line 6.5 ∧ area_difference 13.8 ∧ x = 5.4 :=
sorry

end NUMINAMATH_GPT_EF_side_length_l1504_150467


namespace NUMINAMATH_GPT_not_enough_evidence_to_show_relationship_l1504_150418

noncomputable def isEvidenceToShowRelationship (table : Array (Array Nat)) : Prop :=
  ∃ evidence : Bool, ¬evidence

theorem not_enough_evidence_to_show_relationship :
  isEvidenceToShowRelationship #[#[5, 15, 20], #[40, 10, 50], #[45, 25, 70]] :=
sorry 

end NUMINAMATH_GPT_not_enough_evidence_to_show_relationship_l1504_150418


namespace NUMINAMATH_GPT_man_age_difference_l1504_150421

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 24 :=
by sorry

end NUMINAMATH_GPT_man_age_difference_l1504_150421


namespace NUMINAMATH_GPT_cos2_plus_sin2_given_tan_l1504_150459

noncomputable def problem_cos2_plus_sin2_given_tan : Prop :=
  ∀ (α : ℝ), Real.tan α = 2 → Real.cos α ^ 2 + Real.sin (2 * α) = 1

-- Proof is omitted
theorem cos2_plus_sin2_given_tan : problem_cos2_plus_sin2_given_tan := sorry

end NUMINAMATH_GPT_cos2_plus_sin2_given_tan_l1504_150459


namespace NUMINAMATH_GPT_marble_problem_l1504_150410

theorem marble_problem (a : ℚ) (total : ℚ) 
  (h1 : total = a + 2 * a + 6 * a + 42 * a) :
  a = 42 / 17 :=
by 
  sorry

end NUMINAMATH_GPT_marble_problem_l1504_150410


namespace NUMINAMATH_GPT_determine_continuous_function_l1504_150400

open Real

theorem determine_continuous_function (f : ℝ → ℝ) 
  (h_continuous : Continuous f)
  (h_initial : f 0 = 1)
  (h_inequality : ∀ x y : ℝ, f (x + y) ≥ f x * f y) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = exp (k * x) :=
sorry

end NUMINAMATH_GPT_determine_continuous_function_l1504_150400


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l1504_150403

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 4|
def g (x : ℝ) : ℝ := |2 * x + 1|

-- Problem 1
theorem problem1_solution :
  {x : ℝ | f x < g x} = {x : ℝ | x < -5 ∨ x > 1} :=
sorry

-- Problem 2
theorem problem2_solution :
  ∀ (a : ℝ), (∀ x : ℝ, 2 * f x + g x > a * x) ↔ -4 ≤ a ∧ a < 9 / 4 :=
sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l1504_150403


namespace NUMINAMATH_GPT_johns_out_of_pocket_l1504_150493

noncomputable def total_cost_after_discounts (computer_cost gaming_chair_cost accessories_cost : ℝ) 
  (comp_discount gaming_discount : ℝ) (tax : ℝ) : ℝ :=
  let comp_price := computer_cost * (1 - comp_discount)
  let chair_price := gaming_chair_cost * (1 - gaming_discount)
  let pre_tax_total := comp_price + chair_price + accessories_cost
  pre_tax_total * (1 + tax)

noncomputable def total_selling_price (playstation_value playstation_discount bicycle_price : ℝ) (exchange_rate : ℝ) : ℝ :=
  let playstation_price := playstation_value * (1 - playstation_discount)
  (playstation_price * exchange_rate) / exchange_rate + bicycle_price

theorem johns_out_of_pocket (computer_cost gaming_chair_cost accessories_cost comp_discount gaming_discount tax 
  playstation_value playstation_discount bicycle_price exchange_rate : ℝ) :
  computer_cost = 1500 →
  gaming_chair_cost = 400 →
  accessories_cost = 300 →
  comp_discount = 0.2 →
  gaming_discount = 0.1 →
  tax = 0.05 →
  playstation_value = 600 →
  playstation_discount = 0.2 →
  bicycle_price = 200 →
  exchange_rate = 100 →
  total_cost_after_discounts computer_cost gaming_chair_cost accessories_cost comp_discount gaming_discount tax -
  total_selling_price playstation_value playstation_discount bicycle_price exchange_rate = 1273 := by
  intros
  sorry

end NUMINAMATH_GPT_johns_out_of_pocket_l1504_150493


namespace NUMINAMATH_GPT_gift_distribution_l1504_150494

noncomputable section

structure Recipients :=
  (ondra : String)
  (matej : String)
  (kuba : String)

structure PetrStatements :=
  (ondra_fire_truck : Bool)
  (kuba_no_fire_truck : Bool)
  (matej_no_merkur : Bool)

def exactly_one_statement_true (s : PetrStatements) : Prop :=
  (s.ondra_fire_truck && ¬s.kuba_no_fire_truck && ¬s.matej_no_merkur)
  ∨ (¬s.ondra_fire_truck && s.kuba_no_fire_truck && ¬s.matej_no_merkur)
  ∨ (¬s.ondra_fire_truck && ¬s.kuba_no_fire_truck && s.matej_no_merkur)

def correct_recipients (r : Recipients) : Prop :=
  r.kuba = "fire truck" ∧ r.matej = "helicopter" ∧ r.ondra = "Merkur"

theorem gift_distribution
  (r : Recipients)
  (s : PetrStatements)
  (h : exactly_one_statement_true s)
  (h0 : ¬exactly_one_statement_true ⟨r.ondra = "fire truck", r.kuba ≠ "fire truck", r.matej ≠ "Merkur"⟩)
  (h1 : ¬exactly_one_statement_true ⟨r.ondra ≠ "fire truck", r.kuba ≠ "fire truck", r.matej ≠ "Merkur"⟩)
  : correct_recipients r := by
  -- Proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_gift_distribution_l1504_150494


namespace NUMINAMATH_GPT_lines_passing_through_neg1_0_l1504_150499

theorem lines_passing_through_neg1_0 (k : ℝ) :
  ∀ x y : ℝ, (y = k * (x + 1)) ↔ (x = -1 → y = 0 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_lines_passing_through_neg1_0_l1504_150499


namespace NUMINAMATH_GPT_project_completion_equation_l1504_150485

variables (x : ℕ)

-- Project completion conditions
def person_A_time : ℕ := 12
def person_B_time : ℕ := 8
def A_initial_work_days : ℕ := 3

-- Work done by Person A when working alone for 3 days
def work_A_initial := (A_initial_work_days:ℚ) / person_A_time

-- Work done by Person A and B after the initial 3 days until completion
def combined_work_remaining := 
  (λ x:ℕ => ((x - A_initial_work_days):ℚ) * (1/person_A_time + 1/person_B_time))

-- The equation representing the total work done equals 1
theorem project_completion_equation (x : ℕ) : 
  (x:ℚ) / person_A_time + (x - A_initial_work_days:ℚ) / person_B_time = 1 :=
sorry

end NUMINAMATH_GPT_project_completion_equation_l1504_150485


namespace NUMINAMATH_GPT_a_gt_b_l1504_150424

noncomputable def a (R : Type*) [OrderedRing R] := {x : R // 0 < x ∧ x ^ 3 = x + 1}
noncomputable def b (R : Type*) [OrderedRing R] (a : R) := {y : R // 0 < y ∧ y ^ 6 = y + 3 * a}

theorem a_gt_b (R : Type*) [OrderedRing R] (a_pos_real : a R) (b_pos_real : b R (a_pos_real.val)) : a_pos_real.val > b_pos_real.val :=
sorry

end NUMINAMATH_GPT_a_gt_b_l1504_150424


namespace NUMINAMATH_GPT_factor_poly_l1504_150450

theorem factor_poly (a b : ℤ) (h : 3*(y^2) - y - 24 = (3*y + a)*(y + b)) : a - b = 11 :=
sorry

end NUMINAMATH_GPT_factor_poly_l1504_150450


namespace NUMINAMATH_GPT_fg_of_5_eq_140_l1504_150474

def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 10

theorem fg_of_5_eq_140 : f (g 5) = 140 := by
  sorry

end NUMINAMATH_GPT_fg_of_5_eq_140_l1504_150474


namespace NUMINAMATH_GPT_remaining_oil_quantity_check_remaining_oil_quantity_l1504_150475

def initial_oil_quantity : Real := 40
def outflow_rate : Real := 0.2

theorem remaining_oil_quantity (t : Real) : Real :=
  initial_oil_quantity - outflow_rate * t

theorem check_remaining_oil_quantity (t : Real) : remaining_oil_quantity t = 40 - 0.2 * t := 
by 
  sorry

end NUMINAMATH_GPT_remaining_oil_quantity_check_remaining_oil_quantity_l1504_150475


namespace NUMINAMATH_GPT_tie_rate_correct_l1504_150441

-- Define the fractions indicating win rates for Amy, Lily, and John
def AmyWinRate : ℚ := 4 / 9
def LilyWinRate : ℚ := 1 / 3
def JohnWinRate : ℚ := 1 / 6

-- Define the fraction they tie
def TieRate : ℚ := 1 / 18

-- The theorem for proving the tie rate
theorem tie_rate_correct : AmyWinRate + LilyWinRate + JohnWinRate = 17 / 18 → (1 : ℚ) - (17 / 18) = TieRate :=
by
  sorry -- Proof is omitted

-- Define the win rate sums and tie rate equivalence
example : (AmyWinRate + LilyWinRate + JohnWinRate = 17 / 18) ∧ (TieRate = 1 - 17 / 18) :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_tie_rate_correct_l1504_150441


namespace NUMINAMATH_GPT_smaller_of_two_digit_numbers_with_product_2210_l1504_150414

theorem smaller_of_two_digit_numbers_with_product_2210 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2210 ∧ a ≤ b ∧ a = 26 :=
by
  sorry

end NUMINAMATH_GPT_smaller_of_two_digit_numbers_with_product_2210_l1504_150414


namespace NUMINAMATH_GPT_no_multiple_of_2310_in_2_j_minus_2_i_l1504_150481

theorem no_multiple_of_2310_in_2_j_minus_2_i (i j : ℕ) (h₀ : 0 ≤ i) (h₁ : i < j) (h₂ : j ≤ 50) :
  ¬ ∃ k : ℕ, 2^j - 2^i = 2310 * k :=
by 
  sorry

end NUMINAMATH_GPT_no_multiple_of_2310_in_2_j_minus_2_i_l1504_150481


namespace NUMINAMATH_GPT_total_income_by_nth_year_max_m_and_k_range_l1504_150408

noncomputable def total_income (a : ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  (6 - (n + 6) * 0.1 ^ n) * a

theorem total_income_by_nth_year (a : ℝ) (n : ℕ) :
  total_income a 0.1 n = (6 - (n + 6) * 0.1 ^ n) * a :=
sorry

theorem max_m_and_k_range (a : ℝ) (m : ℕ) :
  (m = 4 ∧ 1 ≤ 1) ∧ (∀ k, k ≥ 1 → m = 4) :=
sorry

end NUMINAMATH_GPT_total_income_by_nth_year_max_m_and_k_range_l1504_150408


namespace NUMINAMATH_GPT_max_value_expression_l1504_150420

theorem max_value_expression (p : ℝ) (q : ℝ) (h : q = p - 2) :
  ∃ M : ℝ, M = -70 + 96.66666666666667 ∧ (∀ p : ℝ, -3 * p^2 + 24 * p - 50 + 10 * q ≤ M) :=
sorry

end NUMINAMATH_GPT_max_value_expression_l1504_150420


namespace NUMINAMATH_GPT_time_since_production_approximate_l1504_150490

noncomputable def solve_time (N N₀ : ℝ) (t : ℝ) : Prop :=
  N = N₀ * (1 / 2) ^ (t / 5730) ∧
  N / N₀ = 3 / 8 ∧
  t = 8138

theorem time_since_production_approximate
  (N N₀ : ℝ)
  (h_decay : N = N₀ * (1 / 2) ^ (t / 5730))
  (h_ratio : N / N₀ = 3 / 8) :
  t = 8138 := 
sorry

end NUMINAMATH_GPT_time_since_production_approximate_l1504_150490


namespace NUMINAMATH_GPT_solution_l1504_150463

noncomputable def determine_numbers (x y : ℚ) : Prop :=
  x^2 + y^2 = 45 / 4 ∧ x - y = x * y

theorem solution (x y : ℚ) :
  determine_numbers x y → (x = -3 ∧ y = 3/2) ∨ (x = -3/2 ∧ y = 3) :=
-- We state the main theorem that relates the determine_numbers predicate to the specific pairs of numbers
sorry

end NUMINAMATH_GPT_solution_l1504_150463


namespace NUMINAMATH_GPT_population_in_2001_l1504_150477

-- Define the populations at specific years
def pop_2000 := 50
def pop_2002 := 146
def pop_2003 := 350

-- Define the population difference condition
def pop_condition (n : ℕ) (pop : ℕ → ℕ) :=
  pop (n + 3) - pop n = 3 * pop (n + 2)

-- Given that the population condition holds, and specific populations are known,
-- the population in the year 2001 is 100
theorem population_in_2001 :
  (∃ (pop : ℕ → ℕ), pop 2000 = pop_2000 ∧ pop 2002 = pop_2002 ∧ pop 2003 = pop_2003 ∧ 
    pop_condition 2000 pop) → ∃ (pop : ℕ → ℕ), pop 2001 = 100 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_population_in_2001_l1504_150477


namespace NUMINAMATH_GPT_questionnaires_drawn_from_unit_D_l1504_150464

theorem questionnaires_drawn_from_unit_D 
  (arith_seq_collected : ∃ a1 d : ℕ, [a1, a1 + d, a1 + 2 * d, a1 + 3 * d] = [aA, aB, aC, aD] ∧ aA + aB + aC + aD = 1000)
  (stratified_sample : [30 - d, 30, 30 + d, 30 + 2 * d] = [sA, sB, sC, sD] ∧ sA + sB + sC + sD = 150)
  (B_drawn : 30 = sB) :
  sD = 60 := 
by {
  sorry
}

end NUMINAMATH_GPT_questionnaires_drawn_from_unit_D_l1504_150464


namespace NUMINAMATH_GPT_tidy_up_time_l1504_150427

theorem tidy_up_time (A B C : ℕ) (tidyA : A = 5 * 3600) (tidyB : B = 5 * 60) (tidyC : C = 5) :
  B < A ∧ B > C :=
by
  sorry

end NUMINAMATH_GPT_tidy_up_time_l1504_150427


namespace NUMINAMATH_GPT_compound_interest_principal_l1504_150468

theorem compound_interest_principal (CI t : ℝ) (r n : ℝ) (P : ℝ) : CI = 630 ∧ t = 2 ∧ r = 0.10 ∧ n = 1 → P = 3000 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_compound_interest_principal_l1504_150468


namespace NUMINAMATH_GPT_finish_11th_l1504_150478

noncomputable def place_in_race (place: Fin 15) := ℕ

variables (Dana Ethan Alice Bob Chris Flora : Fin 15)

def conditions := 
  Dana.val + 3 = Ethan.val ∧
  Alice.val = Bob.val - 2 ∧
  Chris.val = Flora.val - 5 ∧
  Flora.val = Dana.val + 2 ∧
  Ethan.val = Alice.val - 3 ∧
  Bob.val = 6

theorem finish_11th (h : conditions Dana Ethan Alice Bob Chris Flora) : Flora.val = 10 :=
  by sorry

end NUMINAMATH_GPT_finish_11th_l1504_150478


namespace NUMINAMATH_GPT_michael_twenty_dollar_bills_l1504_150429

theorem michael_twenty_dollar_bills (total_amount : ℕ) (denomination : ℕ) 
  (h_total : total_amount = 280) (h_denom : denomination = 20) : 
  total_amount / denomination = 14 := by
  sorry

end NUMINAMATH_GPT_michael_twenty_dollar_bills_l1504_150429


namespace NUMINAMATH_GPT_perimeter_after_growth_operations_perimeter_after_four_growth_operations_l1504_150422

theorem perimeter_after_growth_operations (initial_perimeter : ℝ) (growth_factor : ℝ) (growth_steps : ℕ):
  initial_perimeter = 27 ∧ growth_factor = 4/3 ∧ growth_steps = 2 → 
    initial_perimeter * growth_factor^growth_steps = 48 :=
by
  sorry

theorem perimeter_after_four_growth_operations (initial_perimeter : ℝ) (growth_factor : ℝ) (growth_steps : ℕ):
  initial_perimeter = 27 ∧ growth_factor = 4/3 ∧ growth_steps = 4 → 
    initial_perimeter * growth_factor^growth_steps = 256/3 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_after_growth_operations_perimeter_after_four_growth_operations_l1504_150422


namespace NUMINAMATH_GPT_math_problem_l1504_150486

noncomputable def alpha_condition (α : ℝ) : Prop :=
  4 * Real.cos α - 2 * Real.sin α = 0

theorem math_problem (α : ℝ) (h : alpha_condition α) :
  (Real.sin α)^3 + (Real.cos α)^3 / (Real.sin α - Real.cos α) = 9 / 5 :=
  sorry

end NUMINAMATH_GPT_math_problem_l1504_150486


namespace NUMINAMATH_GPT_solution_volume_l1504_150407

theorem solution_volume (x : ℝ) (h1 : (0.16 * x) / (x + 13) = 0.0733333333333333) : x = 11 :=
by sorry

end NUMINAMATH_GPT_solution_volume_l1504_150407


namespace NUMINAMATH_GPT_max_abs_z_2_2i_l1504_150434

open Complex

theorem max_abs_z_2_2i (z : ℂ) (h : abs (z + 2 - 2 * I) = 1) : 
  ∃ w : ℂ, abs (w - 2 - 2 * I) = 5 :=
sorry

end NUMINAMATH_GPT_max_abs_z_2_2i_l1504_150434


namespace NUMINAMATH_GPT_perfect_play_winner_l1504_150417

theorem perfect_play_winner (A B : ℕ) :
    (A = B → (∃ f : ℕ → ℕ, ∀ n, 0 < f n ∧ f n ≤ B ∧ f n = B - A → false)) ∧
    (A ≠ B → (∃ g : ℕ → ℕ, ∀ n, 0 < g n ∧ g n ≤ B ∧ g n = A - B → false)) :=
sorry

end NUMINAMATH_GPT_perfect_play_winner_l1504_150417


namespace NUMINAMATH_GPT_hexagon_points_fourth_layer_l1504_150406

theorem hexagon_points_fourth_layer :
  ∃ (h : ℕ → ℕ), h 1 = 1 ∧ (∀ n ≥ 2, h n = h (n - 1) + 6 * (n - 1)) ∧ h 4 = 37 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_points_fourth_layer_l1504_150406


namespace NUMINAMATH_GPT_Carver_school_earnings_l1504_150498

noncomputable def total_earnings_Carver_school : ℝ :=
  let base_payment := 20
  let total_payment := 900
  let Allen_days := 7 * 3
  let Balboa_days := 5 * 6
  let Carver_days := 4 * 10
  let total_student_days := Allen_days + Balboa_days + Carver_days
  let adjusted_total_payment := total_payment - 3 * base_payment
  let daily_wage := adjusted_total_payment / total_student_days
  daily_wage * Carver_days

theorem Carver_school_earnings : 
  total_earnings_Carver_school = 369.6 := 
by 
  sorry

end NUMINAMATH_GPT_Carver_school_earnings_l1504_150498


namespace NUMINAMATH_GPT_coin_difference_l1504_150460

noncomputable def max_value (p n d : ℕ) : ℕ := p + 5 * n + 10 * d
noncomputable def min_value (p n d : ℕ) : ℕ := p + 5 * n + 10 * d

theorem coin_difference (p n d : ℕ) (h₁ : p + n + d = 3030) (h₂ : 10 ≤ p) (h₃ : 10 ≤ n) (h₄ : 10 ≤ d) :
  max_value 10 10 3010 - min_value 3010 10 10 = 27000 := by
  sorry

end NUMINAMATH_GPT_coin_difference_l1504_150460


namespace NUMINAMATH_GPT_k_value_l1504_150437

noncomputable def find_k : ℚ := 49 / 15

theorem k_value :
  ∀ (a b : ℚ), (3 * a^2 + 7 * a + find_k = 0) ∧ (3 * b^2 + 7 * b + find_k = 0) →
                (a^2 + b^2 = 3 * a * b) →
                find_k = 49 / 15 :=
by
  intros a b h_eq_root h_rel
  sorry

end NUMINAMATH_GPT_k_value_l1504_150437


namespace NUMINAMATH_GPT_soda_relationship_l1504_150425

theorem soda_relationship (J : ℝ) (L : ℝ) (A : ℝ) (hL : L = 1.75 * J) (hA : A = 1.20 * J) : 
  (L - A) / A = 0.46 := 
by
  sorry

end NUMINAMATH_GPT_soda_relationship_l1504_150425


namespace NUMINAMATH_GPT_intersection_is_expected_l1504_150479

open Set

def setA : Set ℝ := { x | (x + 1) / (x - 2) ≤ 0 }
def setB : Set ℝ := { x | x^2 - 4 * x + 3 ≤ 0 }
def expectedIntersection : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem intersection_is_expected :
  (setA ∩ setB) = expectedIntersection := by
  sorry

end NUMINAMATH_GPT_intersection_is_expected_l1504_150479


namespace NUMINAMATH_GPT_count_divisible_neither_5_nor_7_below_500_l1504_150442

def count_divisible_by (n k : ℕ) : ℕ := (n - 1) / k

def count_divisible_by_5_or_7_below (n : ℕ) : ℕ :=
  let count_5 := count_divisible_by n 5
  let count_7 := count_divisible_by n 7
  let count_35 := count_divisible_by n 35
  count_5 + count_7 - count_35

def count_divisible_neither_5_nor_7_below (n : ℕ) : ℕ :=
  n - 1 - count_divisible_by_5_or_7_below n

theorem count_divisible_neither_5_nor_7_below_500 : count_divisible_neither_5_nor_7_below 500 = 343 :=
by
  sorry

end NUMINAMATH_GPT_count_divisible_neither_5_nor_7_below_500_l1504_150442


namespace NUMINAMATH_GPT_solve_quadratic_l1504_150432

theorem solve_quadratic {x : ℝ} (h : 2 * (x - 1)^2 = x - 1) : x = 1 ∨ x = 3 / 2 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_l1504_150432


namespace NUMINAMATH_GPT_avg_tickets_per_member_is_66_l1504_150476

-- Definitions based on the problem's conditions
def avg_female_tickets : ℕ := 70
def male_to_female_ratio : ℕ := 2
def avg_male_tickets : ℕ := 58

-- Let the number of male members be M and number of female members be F
variables (M : ℕ) (F : ℕ)
def num_female_members : ℕ := male_to_female_ratio * M

-- Total tickets sold by males
def total_male_tickets : ℕ := avg_male_tickets * M

-- Total tickets sold by females
def total_female_tickets : ℕ := avg_female_tickets * num_female_members M

-- Total tickets sold by all members
def total_tickets_sold : ℕ := total_male_tickets M + total_female_tickets M

-- Total number of members
def total_members : ℕ := M + num_female_members M

-- Statement to prove: the average number of tickets sold per member is 66
theorem avg_tickets_per_member_is_66 : total_tickets_sold M / total_members M = 66 :=
by 
  sorry

end NUMINAMATH_GPT_avg_tickets_per_member_is_66_l1504_150476


namespace NUMINAMATH_GPT_mr_hernandez_tax_l1504_150492

theorem mr_hernandez_tax :
  let taxable_income := 42500
  let resident_months := 9
  let standard_deduction := if resident_months > 6 then 5000 else 0
  let adjusted_income := taxable_income - standard_deduction
  let tax_bracket_1 := min adjusted_income 10000 * 0.01
  let tax_bracket_2 := min (max (adjusted_income - 10000) 0) 20000 * 0.03
  let tax_bracket_3 := min (max (adjusted_income - 30000) 0) 30000 * 0.05
  let total_tax_before_credit := tax_bracket_1 + tax_bracket_2 + tax_bracket_3
  let tax_credit := if resident_months < 10 then 500 else 0
  total_tax_before_credit - tax_credit = 575 := 
by
  sorry
  
end NUMINAMATH_GPT_mr_hernandez_tax_l1504_150492


namespace NUMINAMATH_GPT_math_problem_l1504_150416

variable (x y : ℚ)

theorem math_problem (h : 1.5 * x = 0.04 * y) : (y - x) / (y + x) = 73 / 77 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1504_150416


namespace NUMINAMATH_GPT_remaining_hard_hats_l1504_150487

theorem remaining_hard_hats 
  (pink_initial : ℕ)
  (green_initial : ℕ)
  (yellow_initial : ℕ)
  (carl_takes_pink : ℕ)
  (john_takes_pink : ℕ)
  (john_takes_green : ℕ) :
  john_takes_green = 2 * john_takes_pink →
  pink_initial = 26 →
  green_initial = 15 →
  yellow_initial = 24 →
  carl_takes_pink = 4 →
  john_takes_pink = 6 →
  ∃ pink_remaining green_remaining yellow_remaining total_remaining, 
    pink_remaining = pink_initial - carl_takes_pink - john_takes_pink ∧
    green_remaining = green_initial - john_takes_green ∧
    yellow_remaining = yellow_initial ∧
    total_remaining = pink_remaining + green_remaining + yellow_remaining ∧
    total_remaining = 43 :=
by
  sorry

end NUMINAMATH_GPT_remaining_hard_hats_l1504_150487


namespace NUMINAMATH_GPT_integer_exponentiation_l1504_150409

theorem integer_exponentiation
  (a b x y : ℕ)
  (h_gcd : a.gcd b = 1)
  (h_pos_a : 1 < a)
  (h_pos_b : 1 < b)
  (h_pos_x : 1 < x)
  (h_pos_y : 1 < y)
  (h_eq : x^a = y^b) :
  ∃ n : ℕ, 1 < n ∧ x = n^b ∧ y = n^a :=
by sorry

end NUMINAMATH_GPT_integer_exponentiation_l1504_150409


namespace NUMINAMATH_GPT_modular_inverse_example_l1504_150454

open Int

theorem modular_inverse_example :
  ∃ b : ℤ, 0 ≤ b ∧ b < 120 ∧ (7 * b) % 120 = 1 ∧ b = 103 :=
by
  sorry

end NUMINAMATH_GPT_modular_inverse_example_l1504_150454


namespace NUMINAMATH_GPT_diagonal_length_not_possible_l1504_150448

-- Define the side lengths of the parallelogram
def sides_of_parallelogram : ℕ × ℕ := (6, 8)

-- Define the length of a diagonal that cannot exist
def invalid_diagonal_length : ℕ := 15

-- Statement: Prove that a diagonal of length 15 cannot exist for such a parallelogram.
theorem diagonal_length_not_possible (a b d : ℕ) 
  (h₁ : sides_of_parallelogram = (a, b)) 
  (h₂ : d = invalid_diagonal_length) 
  : d ≥ a + b := 
sorry

end NUMINAMATH_GPT_diagonal_length_not_possible_l1504_150448


namespace NUMINAMATH_GPT_simplify_fraction_l1504_150497

theorem simplify_fraction (d : ℝ) : (6 - 5 * d) / 9 - 3 = (-21 - 5 * d) / 9 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1504_150497


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l1504_150435

theorem min_value_reciprocal_sum 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 2) : 
  ∃ x, x = 2 ∧ (∀ y, y = (1 / a) + (1 / b) → x ≤ y) := 
sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l1504_150435


namespace NUMINAMATH_GPT_no_solution_m_l1504_150455

theorem no_solution_m {
  m : ℚ
  } (h : ∀ x : ℚ, x ≠ 3 → (3 - 2 * x) / (x - 3) - (m * x - 2) / (3 - x) ≠ -1) : 
  m = 1 ∨ m = 5 / 3 :=
sorry

end NUMINAMATH_GPT_no_solution_m_l1504_150455


namespace NUMINAMATH_GPT_decomposition_of_5_to_4_eq_125_l1504_150405

theorem decomposition_of_5_to_4_eq_125 :
  (∃ a b c : ℕ, (5^4 = a + b + c) ∧ 
                (a = 121) ∧ 
                (b = 123) ∧ 
                (c = 125)) := by 
sorry

end NUMINAMATH_GPT_decomposition_of_5_to_4_eq_125_l1504_150405


namespace NUMINAMATH_GPT_laser_beam_total_distance_l1504_150456

theorem laser_beam_total_distance :
  let A := (4, 7)
  let B := (-4, 7)
  let C := (-4, -7)
  let D := (4, -7)
  let E := (9, 7)
  let dist (p1 p2 : (ℤ × ℤ)) : ℝ := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  dist A B + dist B C + dist C D + dist D E = 30 + Real.sqrt 221 :=
by
  sorry

end NUMINAMATH_GPT_laser_beam_total_distance_l1504_150456


namespace NUMINAMATH_GPT_students_not_enrolled_in_either_course_l1504_150465

theorem students_not_enrolled_in_either_course 
  (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h_total : total = 87) (h_french : french = 41) (h_german : german = 22) (h_both : both = 9) : 
  ∃ (not_enrolled : ℕ), not_enrolled = (total - (french + german - both)) ∧ not_enrolled = 33 := by
  have h_french_or_german : ℕ := french + german - both
  have h_not_enrolled : ℕ := total - h_french_or_german
  use h_not_enrolled
  sorry

end NUMINAMATH_GPT_students_not_enrolled_in_either_course_l1504_150465


namespace NUMINAMATH_GPT_smallest_value_of_x_l1504_150482

theorem smallest_value_of_x (x : ℝ) (h : |x - 3| = 8) : x = -5 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_x_l1504_150482


namespace NUMINAMATH_GPT_solve_equation_l1504_150473

theorem solve_equation (x : ℝ) : 
  (x - 1)^2 + 2 * x * (x - 1) = 0 ↔ x = 1 ∨ x = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1504_150473
