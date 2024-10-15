import Mathlib

namespace NUMINAMATH_GPT_division_of_sums_and_products_l1542_154258

theorem division_of_sums_and_products (a b c : ℕ) (h_a : a = 7) (h_b : b = 5) (h_c : c = 3) :
  (a^3 + b^3 + c^3) / (a^2 - a * b + b^2 - b * c + c^2) = 15 := by
  -- proofs go here
  sorry

end NUMINAMATH_GPT_division_of_sums_and_products_l1542_154258


namespace NUMINAMATH_GPT_ratio_longer_to_shorter_side_l1542_154263

-- Definitions of the problem
variables (l s : ℝ)
def rect_sheet_fold : Prop :=
  l = Real.sqrt (s^2 + (s^2 / l)^2)

-- The to-be-proved theorem
theorem ratio_longer_to_shorter_side (h : rect_sheet_fold l s) :
  l / s = Real.sqrt ((2 : ℝ) / (Real.sqrt 5 - 1)) :=
sorry

end NUMINAMATH_GPT_ratio_longer_to_shorter_side_l1542_154263


namespace NUMINAMATH_GPT_point_distance_to_focus_of_parabola_with_focus_distance_l1542_154224

def parabola_with_focus_distance (focus_distance : ℝ) (p : ℝ × ℝ) : Prop :=
  let f := (0, focus_distance)
  let directrix := -focus_distance
  let (x, y) := p
  let distance_to_focus := Real.sqrt ((x - 0)^2 + (y - focus_distance)^2)
  let distance_to_directrix := abs (y - directrix)
  distance_to_focus = distance_to_directrix

theorem point_distance_to_focus_of_parabola_with_focus_distance 
  (focus_distance : ℝ) (y_axis_distance : ℝ) (p : ℝ × ℝ)
  (h_focus_distance : focus_distance = 4)
  (h_y_axis_distance : abs (p.1) = 1) :
  parabola_with_focus_distance focus_distance p →
  Real.sqrt ((p.1 - 0)^2 + (p.2 - focus_distance)^2) = 5 :=
by
  sorry

end NUMINAMATH_GPT_point_distance_to_focus_of_parabola_with_focus_distance_l1542_154224


namespace NUMINAMATH_GPT_Heesu_has_greatest_sum_l1542_154268

-- Define the numbers collected by each individual
def Sora_collected : (Nat × Nat) := (4, 6)
def Heesu_collected : (Nat × Nat) := (7, 5)
def Jiyeon_collected : (Nat × Nat) := (3, 8)

-- Calculate the sums
def Sora_sum : Nat := Sora_collected.1 + Sora_collected.2
def Heesu_sum : Nat := Heesu_collected.1 + Heesu_collected.2
def Jiyeon_sum : Nat := Jiyeon_collected.1 + Jiyeon_collected.2

-- The theorem to prove that Heesu has the greatest sum
theorem Heesu_has_greatest_sum :
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  sorry

end NUMINAMATH_GPT_Heesu_has_greatest_sum_l1542_154268


namespace NUMINAMATH_GPT_strictly_increasing_function_exists_l1542_154296

noncomputable def exists_strictly_increasing_function (f : ℕ → ℕ) :=
  (∀ n : ℕ, n = 1 → f n = 2) ∧
  (∀ n : ℕ, f (f n) = f n + n) ∧
  (∀ m n : ℕ, m < n → f m < f n)

theorem strictly_increasing_function_exists : 
  ∃ f : ℕ → ℕ,
  exists_strictly_increasing_function f :=
sorry

end NUMINAMATH_GPT_strictly_increasing_function_exists_l1542_154296


namespace NUMINAMATH_GPT_distance_walked_by_friend_P_l1542_154248

def trail_length : ℝ := 33
def speed_ratio : ℝ := 1.20

theorem distance_walked_by_friend_P (v t d_P : ℝ) 
  (h1 : t = 33 / (2.20 * v)) 
  (h2 : d_P = 1.20 * v * t) 
  : d_P = 18 := by
  sorry

end NUMINAMATH_GPT_distance_walked_by_friend_P_l1542_154248


namespace NUMINAMATH_GPT_alexandra_magazines_l1542_154201

noncomputable def magazines (bought_on_friday : ℕ) (bought_on_saturday : ℕ) (times_friday : ℕ) (chewed_up : ℕ) : ℕ :=
  bought_on_friday + bought_on_saturday + times_friday * bought_on_friday - chewed_up

theorem alexandra_magazines :
  ∀ (bought_on_friday bought_on_saturday times_friday chewed_up : ℕ),
      bought_on_friday = 8 → 
      bought_on_saturday = 12 → 
      times_friday = 4 → 
      chewed_up = 4 →
      magazines bought_on_friday bought_on_saturday times_friday chewed_up = 48 :=
by
  intros
  sorry

end NUMINAMATH_GPT_alexandra_magazines_l1542_154201


namespace NUMINAMATH_GPT_only_D_is_quadratic_l1542_154276

-- Conditions
def eq_A (x : ℝ) : Prop := x^2 + 1/x - 1 = 0
def eq_B (x : ℝ) : Prop := (2*x + 1) + x = 0
def eq_C (m x : ℝ) : Prop := 2*m^2 + x = 3
def eq_D (x : ℝ) : Prop := x^2 - x = 0

-- Proof statement
theorem only_D_is_quadratic :
  ∃ (x : ℝ), eq_D x ∧ 
  (¬(∃ x : ℝ, eq_A x) ∧ ¬(∃ x : ℝ, eq_B x) ∧ ¬(∃ (m x : ℝ), eq_C m x)) :=
by
  sorry

end NUMINAMATH_GPT_only_D_is_quadratic_l1542_154276


namespace NUMINAMATH_GPT_mul_582964_99999_l1542_154236

theorem mul_582964_99999 : 582964 * 99999 = 58295817036 := by
  sorry

end NUMINAMATH_GPT_mul_582964_99999_l1542_154236


namespace NUMINAMATH_GPT_sqrt_sum_eq_eight_l1542_154255

theorem sqrt_sum_eq_eight :
  Real.sqrt (24 - 8 * Real.sqrt 3) + Real.sqrt (24 + 8 * Real.sqrt 3) = 8 := by
  sorry

end NUMINAMATH_GPT_sqrt_sum_eq_eight_l1542_154255


namespace NUMINAMATH_GPT_circumradius_geq_3_times_inradius_l1542_154250

-- Define the variables representing the circumradius and inradius
variables {R r : ℝ}

-- Assume the conditions that R is the circumradius and r is the inradius of a tetrahedron
def tetrahedron_circumradius (R : ℝ) : Prop := true
def tetrahedron_inradius (r : ℝ) : Prop := true

-- State the theorem
theorem circumradius_geq_3_times_inradius (hR : tetrahedron_circumradius R) (hr : tetrahedron_inradius r) : R ≥ 3 * r :=
sorry

end NUMINAMATH_GPT_circumradius_geq_3_times_inradius_l1542_154250


namespace NUMINAMATH_GPT_arithmetic_sequence_middle_term_l1542_154237

theorem arithmetic_sequence_middle_term 
  (a b c d e : ℕ) 
  (h_seq : a = 23 ∧ e = 53 ∧ (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d)) :
  c = 38 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_middle_term_l1542_154237


namespace NUMINAMATH_GPT_sum_of_possible_k_l1542_154206

theorem sum_of_possible_k (a b c k : ℂ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a / (2 - b) = k) (h5 : b / (3 - c) = k) (h6 : c / (4 - a) = k) :
  k = 1 ∨ k = -1 ∨ k = -2 → k = 1 + (-1) + (-2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_k_l1542_154206


namespace NUMINAMATH_GPT_difference_sum_first_100_odds_evens_l1542_154205

def sum_first_n_odds (n : ℕ) : ℕ :=
  n^2

def sum_first_n_evens (n : ℕ) : ℕ :=
  n * (n-1)

theorem difference_sum_first_100_odds_evens :
  sum_first_n_odds 100 - sum_first_n_evens 100 = 100 := by
  sorry

end NUMINAMATH_GPT_difference_sum_first_100_odds_evens_l1542_154205


namespace NUMINAMATH_GPT_negation_proposition_l1542_154280

open Classical

theorem negation_proposition :
  ¬ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ ∀ x : ℝ, x^2 - 2*x + 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1542_154280


namespace NUMINAMATH_GPT_ladder_of_twos_l1542_154222

theorem ladder_of_twos (n : ℕ) (h : n ≥ 3) : 
  ∃ N_n : ℕ, N_n = 2 ^ (n - 3) :=
by
  sorry

end NUMINAMATH_GPT_ladder_of_twos_l1542_154222


namespace NUMINAMATH_GPT_rectangle_area_l1542_154247

-- Definitions from conditions:
def side_length : ℕ := 16 / 4
def area_B : ℕ := side_length * side_length
def probability_not_within_B : ℝ := 0.4666666666666667

-- Main statement to prove
theorem rectangle_area (A : ℝ) (h1 : side_length = 4)
 (h2 : area_B = 16)
 (h3 : probability_not_within_B = 0.4666666666666667) :
   A * 0.5333333333333333 = 16 → A = 30 :=
by
  intros h
  sorry


end NUMINAMATH_GPT_rectangle_area_l1542_154247


namespace NUMINAMATH_GPT_exponent_combination_l1542_154228

theorem exponent_combination (a : ℝ) (m n : ℕ) (h₁ : a^m = 3) (h₂ : a^n = 4) :
  a^(2 * m + 3 * n) = 576 :=
by
  sorry

end NUMINAMATH_GPT_exponent_combination_l1542_154228


namespace NUMINAMATH_GPT_december_sales_fraction_l1542_154257

noncomputable def average_sales (A : ℝ) := 11 * A
noncomputable def december_sales (A : ℝ) := 3 * A
noncomputable def total_sales (A : ℝ) := average_sales A + december_sales A

theorem december_sales_fraction (A : ℝ) (h1 : december_sales A = 3 * A)
  (h2 : average_sales A = 11 * A) :
  december_sales A / total_sales A = 3 / 14 :=
by
  sorry

end NUMINAMATH_GPT_december_sales_fraction_l1542_154257


namespace NUMINAMATH_GPT_cuberoot_condition_l1542_154283

/-- If \(\sqrt[3]{x-1}=3\), then \((x-1)^2 = 729\). -/
theorem cuberoot_condition (x : ℝ) (h : (x - 1)^(1/3) = 3) : (x - 1)^2 = 729 := 
  sorry

end NUMINAMATH_GPT_cuberoot_condition_l1542_154283


namespace NUMINAMATH_GPT_cos_x_when_sin_x_is_given_l1542_154207

theorem cos_x_when_sin_x_is_given (x : ℝ) (h : Real.sin x = (Real.sqrt 5) / 5) :
  Real.cos x = -(Real.sqrt 20) / 5 :=
sorry

end NUMINAMATH_GPT_cos_x_when_sin_x_is_given_l1542_154207


namespace NUMINAMATH_GPT_find_a_b_find_range_of_x_l1542_154288

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (Real.log x / Real.log 2)^2 - 2 * a * (Real.log x / Real.log 2) + b

theorem find_a_b (a b : ℝ) :
  (f (1/4) a b = -1) → (a = -2 ∧ b = 3) :=
by
  sorry

theorem find_range_of_x (a b : ℝ) :
  a = -2 → b = 3 →
  ∀ x : ℝ, (f x a b < 0) → (1/8 < x ∧ x < 1/2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_find_range_of_x_l1542_154288


namespace NUMINAMATH_GPT_larinjaitis_age_l1542_154282

theorem larinjaitis_age : 
  ∀ (birth_year : ℤ) (death_year : ℤ), birth_year = -30 → death_year = 30 → (death_year - birth_year + 1) = 1 :=
by
  intros birth_year death_year h_birth h_death
  sorry

end NUMINAMATH_GPT_larinjaitis_age_l1542_154282


namespace NUMINAMATH_GPT_chessboard_not_divisible_by_10_l1542_154293

theorem chessboard_not_divisible_by_10 :
  ∀ (B : ℕ × ℕ → ℕ), 
  (∀ x y, B (x, y) < 10) ∧ 
  (∀ x y, x ≥ 0 ∧ x < 8 ∧ y ≥ 0 ∧ y < 8) →
  ¬ ( ∃ k : ℕ, ∀ x y, (B (x, y) + k) % 10 = 0 ) :=
by
  intros
  sorry

end NUMINAMATH_GPT_chessboard_not_divisible_by_10_l1542_154293


namespace NUMINAMATH_GPT_prob_below_8_correct_l1542_154212

-- Defining the probabilities of hitting the 10, 9, and 8 rings
def prob_10 : ℝ := 0.20
def prob_9 : ℝ := 0.30
def prob_8 : ℝ := 0.10

-- Defining the event of scoring below 8
def prob_below_8 : ℝ := 1 - (prob_10 + prob_9 + prob_8)

-- The main theorem to prove: the probability of scoring below 8 is 0.40
theorem prob_below_8_correct : prob_below_8 = 0.40 :=
by 
  -- We need to show this proof in a separate proof phase
  sorry

end NUMINAMATH_GPT_prob_below_8_correct_l1542_154212


namespace NUMINAMATH_GPT_triangle_has_side_property_l1542_154275

theorem triangle_has_side_property (a b c : ℝ) (A B C : ℝ) 
  (h₀ : 3 * b * Real.cos C + 3 * c * Real.cos B = a^2)
  (h₁ : A + B + C = Real.pi)
  (h₂ : a = 3) :
  a = 3 := 
sorry

end NUMINAMATH_GPT_triangle_has_side_property_l1542_154275


namespace NUMINAMATH_GPT_minimize_total_resistance_l1542_154214

variable (a1 a2 a3 a4 a5 a6 : ℝ)
variable (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧ a5 > a6)

/-- Theorem: Given resistances a1, a2, a3, a4, a5, a6 such that a1 > a2 > a3 > a4 > a5 > a6, 
arranging them in the sequence a1 > a2 > a3 > a4 > a5 > a6 minimizes the total resistance
for the assembled component. -/
theorem minimize_total_resistance : 
  True := 
sorry

end NUMINAMATH_GPT_minimize_total_resistance_l1542_154214


namespace NUMINAMATH_GPT_average_side_length_of_squares_l1542_154209

theorem average_side_length_of_squares (a b c : ℕ) (h₁ : a = 36) (h₂ : b = 64) (h₃ : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 26 / 3 := by
  sorry

end NUMINAMATH_GPT_average_side_length_of_squares_l1542_154209


namespace NUMINAMATH_GPT_parallel_lines_k_l1542_154211

theorem parallel_lines_k (k : ℝ) :
  (∃ (x y : ℝ), (k-3) * x + (4-k) * y + 1 = 0 ∧ 2 * (k-3) * x - 2 * y + 3 = 0) →
  (k = 3 ∨ k = 5) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_k_l1542_154211


namespace NUMINAMATH_GPT_number_of_shelves_l1542_154231

theorem number_of_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h_total_books : total_books = 14240) (h_books_per_shelf : books_per_shelf = 8) : total_books / books_per_shelf = 1780 :=
by 
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_number_of_shelves_l1542_154231


namespace NUMINAMATH_GPT_total_volume_of_cubes_l1542_154265

theorem total_volume_of_cubes (s : ℕ) (n : ℕ) (h_s : s = 5) (h_n : n = 4) : 
  n * s^3 = 500 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_cubes_l1542_154265


namespace NUMINAMATH_GPT_weight_of_raisins_proof_l1542_154285

-- Define the conditions
def weight_of_peanuts : ℝ := 0.1
def total_weight_of_snacks : ℝ := 0.5

-- Theorem to prove that the weight of raisins equals 0.4 pounds
theorem weight_of_raisins_proof : total_weight_of_snacks - weight_of_peanuts = 0.4 := by
  sorry

end NUMINAMATH_GPT_weight_of_raisins_proof_l1542_154285


namespace NUMINAMATH_GPT_first_player_wins_if_not_power_of_two_l1542_154277

/-- 
  Prove that the first player can guarantee a win if and only if $n$ is not a power of two, under the given conditions. 
-/
theorem first_player_wins_if_not_power_of_two
  (n : ℕ) (h : n > 1) :
  (∃ k : ℕ, n = 2^k) ↔ false :=
sorry

end NUMINAMATH_GPT_first_player_wins_if_not_power_of_two_l1542_154277


namespace NUMINAMATH_GPT_bellas_goal_product_l1542_154203

theorem bellas_goal_product (g1 g2 g3 g4 g5 g6 : ℕ) (g7 g8 : ℕ) 
  (h1 : g1 = 5) 
  (h2 : g2 = 3) 
  (h3 : g3 = 2) 
  (h4 : g4 = 4)
  (h5 : g5 = 1) 
  (h6 : g6 = 6)
  (h7 : g7 < 10)
  (h8 : (g1 + g2 + g3 + g4 + g5 + g6 + g7) % 7 = 0) 
  (h9 : g8 < 10)
  (h10 : (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) % 8 = 0) :
  g7 * g8 = 28 :=
by 
  sorry

end NUMINAMATH_GPT_bellas_goal_product_l1542_154203


namespace NUMINAMATH_GPT_balls_per_color_l1542_154292

theorem balls_per_color (total_balls : ℕ) (total_colors : ℕ)
  (h1 : total_balls = 350) (h2 : total_colors = 10) : 
  total_balls / total_colors = 35 :=
by
  sorry

end NUMINAMATH_GPT_balls_per_color_l1542_154292


namespace NUMINAMATH_GPT_mike_spent_on_car_parts_l1542_154229

-- Define the costs as constants
def cost_speakers : ℝ := 118.54
def cost_tires : ℝ := 106.33
def cost_cds : ℝ := 4.58

-- Define the total cost of car parts excluding the CDs
def total_cost_car_parts : ℝ := cost_speakers + cost_tires

-- The theorem we want to prove
theorem mike_spent_on_car_parts :
  total_cost_car_parts = 224.87 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_mike_spent_on_car_parts_l1542_154229


namespace NUMINAMATH_GPT_answer_is_p_and_q_l1542_154240

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end NUMINAMATH_GPT_answer_is_p_and_q_l1542_154240


namespace NUMINAMATH_GPT_certain_number_divides_expression_l1542_154238

theorem certain_number_divides_expression : 
  ∃ m : ℕ, (∃ n : ℕ, n = 6 ∧ m ∣ (11 * n - 1)) ∧ m = 65 := 
by
  sorry

end NUMINAMATH_GPT_certain_number_divides_expression_l1542_154238


namespace NUMINAMATH_GPT_find_value_of_D_l1542_154223

theorem find_value_of_D (C : ℕ) (D : ℕ) (k : ℕ) (h : C = (10^D) * k) (hD : k % 10 ≠ 0) : D = 69 := by
  sorry

end NUMINAMATH_GPT_find_value_of_D_l1542_154223


namespace NUMINAMATH_GPT_hexahedron_has_six_faces_l1542_154291

-- Definition based on the condition
def is_hexahedron (P : Type) := 
  ∃ (f : P → ℕ), ∀ (x : P), f x = 6

-- Theorem statement based on the question and correct answer
theorem hexahedron_has_six_faces (P : Type) (h : is_hexahedron P) : 
  ∀ (x : P), ∃ (f : P → ℕ), f x = 6 :=
by 
  sorry

end NUMINAMATH_GPT_hexahedron_has_six_faces_l1542_154291


namespace NUMINAMATH_GPT_totalCarsProduced_is_29621_l1542_154219

def numSedansNA    := 3884
def numSUVsNA      := 2943
def numPickupsNA   := 1568

def numSedansEU    := 2871
def numSUVsEU      := 2145
def numPickupsEU   := 643

def numSedansASIA  := 5273
def numSUVsASIA    := 3881
def numPickupsASIA := 2338

def numSedansSA    := 1945
def numSUVsSA      := 1365
def numPickupsSA   := 765

def totalCarsProduced : Nat :=
  numSedansNA + numSUVsNA + numPickupsNA +
  numSedansEU + numSUVsEU + numPickupsEU +
  numSedansASIA + numSUVsASIA + numPickupsASIA +
  numSedansSA + numSUVsSA + numPickupsSA

theorem totalCarsProduced_is_29621 : totalCarsProduced = 29621 :=
by
  sorry

end NUMINAMATH_GPT_totalCarsProduced_is_29621_l1542_154219


namespace NUMINAMATH_GPT_distance_NYC_to_DC_l1542_154215

noncomputable def horse_speed := 10 -- miles per hour
noncomputable def travel_time := 24 -- hours

theorem distance_NYC_to_DC : horse_speed * travel_time = 240 := by
  sorry

end NUMINAMATH_GPT_distance_NYC_to_DC_l1542_154215


namespace NUMINAMATH_GPT_probability_reach_3_1_in_8_steps_l1542_154227

theorem probability_reach_3_1_in_8_steps :
  let m := 35
  let n := 2048
  let q := m / n
  ∃ (m n : ℕ), (Nat.gcd m n = 1) ∧ (q = 35 / 2048) ∧ (m + n = 2083) := by
  sorry

end NUMINAMATH_GPT_probability_reach_3_1_in_8_steps_l1542_154227


namespace NUMINAMATH_GPT_factory_fills_boxes_per_hour_l1542_154230

theorem factory_fills_boxes_per_hour
  (colors_per_box : ℕ)
  (crayons_per_color : ℕ)
  (total_crayons : ℕ)
  (hours : ℕ)
  (crayons_per_hour := total_crayons / hours)
  (crayons_per_box := colors_per_box * crayons_per_color)
  (boxes_per_hour := crayons_per_hour / crayons_per_box) :
  colors_per_box = 4 →
  crayons_per_color = 2 →
  total_crayons = 160 →
  hours = 4 →
  boxes_per_hour = 5 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_factory_fills_boxes_per_hour_l1542_154230


namespace NUMINAMATH_GPT_expected_value_is_correct_l1542_154242

noncomputable def expected_winnings : ℚ :=
  (5/12 : ℚ) * 2 + (1/3 : ℚ) * 0 + (1/6 : ℚ) * (-2) + (1/12 : ℚ) * 10

theorem expected_value_is_correct : expected_winnings = 4 / 3 := 
by 
  -- Complex calculations skipped for brevity
  sorry

end NUMINAMATH_GPT_expected_value_is_correct_l1542_154242


namespace NUMINAMATH_GPT_volleyballs_basketballs_difference_l1542_154204

variable (V B : ℕ)

theorem volleyballs_basketballs_difference :
  (V + B = 14) →
  (4 * V + 5 * B = 60) →
  V - B = 6 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_volleyballs_basketballs_difference_l1542_154204


namespace NUMINAMATH_GPT_quadratic_transformation_l1542_154278

theorem quadratic_transformation :
  ∀ x : ℝ, (x^2 - 6 * x - 5 = 0) → ((x - 3)^2 = 14) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_quadratic_transformation_l1542_154278


namespace NUMINAMATH_GPT_alternating_students_count_l1542_154252

theorem alternating_students_count :
  let num_male := 4
  let num_female := 5
  let arrangements := Nat.factorial num_female * Nat.factorial num_male
  arrangements = 2880 :=
by
  sorry

end NUMINAMATH_GPT_alternating_students_count_l1542_154252


namespace NUMINAMATH_GPT_eval_expression_l1542_154289

theorem eval_expression : 
  (8^5) / (4 * 2^5 + 16) = 2^11 / 9 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1542_154289


namespace NUMINAMATH_GPT_oil_bill_january_l1542_154202

theorem oil_bill_january (F J : ℝ)
  (h1 : F / J = 5 / 4)
  (h2 : (F + 30) / J = 3 / 2) :
  J = 120 :=
sorry

end NUMINAMATH_GPT_oil_bill_january_l1542_154202


namespace NUMINAMATH_GPT_inscribed_circle_diameter_l1542_154295

theorem inscribed_circle_diameter (PQ PR QR : ℝ) (h₁ : PQ = 13) (h₂ : PR = 14) (h₃ : QR = 15) :
  ∃ d : ℝ, d = 8 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_diameter_l1542_154295


namespace NUMINAMATH_GPT_ninth_term_arith_seq_l1542_154249

theorem ninth_term_arith_seq (a d : ℤ) (h1 : a + 2 * d = 25) (h2 : a + 5 * d = 31) : a + 8 * d = 37 :=
sorry

end NUMINAMATH_GPT_ninth_term_arith_seq_l1542_154249


namespace NUMINAMATH_GPT_probability_neither_event_l1542_154279

theorem probability_neither_event (P_A P_B P_A_and_B : ℝ)
  (h1 : P_A = 0.25)
  (h2 : P_B = 0.40)
  (h3 : P_A_and_B = 0.20) :
  1 - (P_A + P_B - P_A_and_B) = 0.55 :=
by
  sorry

end NUMINAMATH_GPT_probability_neither_event_l1542_154279


namespace NUMINAMATH_GPT_largest_multiple_l1542_154260

theorem largest_multiple (a b limit : ℕ) (ha : a = 3) (hb : b = 5) (h_limit : limit = 800) : 
  ∃ (n : ℕ), (lcm a b) * n < limit ∧ (lcm a b) * (n + 1) ≥ limit ∧ (lcm a b) * n = 795 := 
by 
  sorry

end NUMINAMATH_GPT_largest_multiple_l1542_154260


namespace NUMINAMATH_GPT_consecutive_negative_integers_sum_l1542_154234

theorem consecutive_negative_integers_sum (n : ℤ) (hn : n < 0) (hn1 : n + 1 < 0) (hprod : n * (n + 1) = 2550) : n + (n + 1) = -101 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_negative_integers_sum_l1542_154234


namespace NUMINAMATH_GPT_jenny_distance_from_school_l1542_154243

-- Definitions based on the given conditions.
def kernels_per_feet : ℕ := 1
def feet_per_kernel : ℕ := 25
def squirrel_fraction_eaten : ℚ := 1/4
def remaining_kernels : ℕ := 150

-- Problem statement in Lean 4.
theorem jenny_distance_from_school : 
  ∀ (P : ℕ), (3/4:ℚ) * P = 150 → P * feet_per_kernel = 5000 :=
by
  intros P h
  sorry

end NUMINAMATH_GPT_jenny_distance_from_school_l1542_154243


namespace NUMINAMATH_GPT_trapezoid_QR_length_l1542_154232

noncomputable def length_QR (PQ RS area altitude : ℕ) : ℝ :=
  24 - Real.sqrt 11 - 2 * Real.sqrt 24

theorem trapezoid_QR_length :
  ∀ (PQ RS area altitude : ℕ), 
  area = 240 → altitude = 10 → PQ = 12 → RS = 22 →
  length_QR PQ RS area altitude = 24 - Real.sqrt 11 - 2 * Real.sqrt 24 :=
by
  intros PQ RS area altitude h_area h_altitude h_PQ h_RS
  unfold length_QR
  sorry

end NUMINAMATH_GPT_trapezoid_QR_length_l1542_154232


namespace NUMINAMATH_GPT_larger_number_l1542_154239

theorem larger_number (x y : ℕ) (h₁ : x + y = 27) (h₂ : x - y = 5) : x = 16 :=
by sorry

end NUMINAMATH_GPT_larger_number_l1542_154239


namespace NUMINAMATH_GPT_fraction_of_difference_l1542_154244

theorem fraction_of_difference (A_s A_l : ℝ) (h_total : A_s + A_l = 500) (h_smaller : A_s = 225) :
  (A_l - A_s) / ((A_s + A_l) / 2) = 1 / 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fraction_of_difference_l1542_154244


namespace NUMINAMATH_GPT_proof_problem_l1542_154294

variables (α : ℝ)

-- Condition: tan(α) = 2
def tan_condition : Prop := Real.tan α = 2

-- First expression: (sin α + 2 cos α) / (4 cos α - sin α) = 2
def expression1 : Prop := (Real.sin α + 2 * Real.cos α) / (4 * Real.cos α - Real.sin α) = 2

-- Second expression: sqrt(2) * sin(2α + π/4) + 1 = 6/5
def expression2 : Prop := Real.sqrt 2 * Real.sin (2 * α + Real.pi / 4) + 1 = 6 / 5

-- Theorem: Prove the expressions given the condition
theorem proof_problem :
  tan_condition α → expression1 α ∧ expression2 α :=
by
  intro tan_cond
  have h1 : expression1 α := sorry
  have h2 : expression2 α := sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_proof_problem_l1542_154294


namespace NUMINAMATH_GPT_walking_time_l1542_154266

noncomputable def time_to_reach_destination (mr_harris_speed : ℝ) (mr_harris_time_to_store : ℝ) (your_speed : ℝ) (distance_factor : ℝ) : ℝ :=
  let store_distance := mr_harris_speed * mr_harris_time_to_store
  let your_destination_distance := distance_factor * store_distance
  your_destination_distance / your_speed

theorem walking_time (mr_harris_speed your_speed : ℝ) (mr_harris_time_to_store : ℝ) (distance_factor : ℝ) (h_speed : your_speed = 2 * mr_harris_speed) (h_time : mr_harris_time_to_store = 2) (h_factor : distance_factor = 3) :
  time_to_reach_destination mr_harris_speed mr_harris_time_to_store your_speed distance_factor = 3 :=
by
  rw [h_time, h_speed, h_factor]
  -- calculations based on given conditions
  sorry

end NUMINAMATH_GPT_walking_time_l1542_154266


namespace NUMINAMATH_GPT_coordinates_of_B_l1542_154220
open Real

-- Define the conditions given in the problem
def A : ℝ × ℝ := (1, 6)
def d : ℝ := 4

-- Define the properties of the solution given the conditions
theorem coordinates_of_B (B : ℝ × ℝ) :
  (B = (-3, 6) ∨ B = (5, 6)) ↔
  (B.2 = A.2 ∧ (B.1 = A.1 - d ∨ B.1 = A.1 + d)) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_B_l1542_154220


namespace NUMINAMATH_GPT_trig_identity_solution_l1542_154281

theorem trig_identity_solution
  (x : ℝ)
  (h : Real.sin (x + Real.pi / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_solution_l1542_154281


namespace NUMINAMATH_GPT_opposite_of_2021_l1542_154264

theorem opposite_of_2021 : ∃ y : ℝ, 2021 + y = 0 ∧ y = -2021 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2021_l1542_154264


namespace NUMINAMATH_GPT_investment_amount_correct_l1542_154272

-- Lean statement definitions based on conditions
def cost_per_tshirt : ℕ := 3
def selling_price_per_tshirt : ℕ := 20
def tshirts_sold : ℕ := 83
def total_revenue : ℕ := tshirts_sold * selling_price_per_tshirt
def total_cost_of_tshirts : ℕ := tshirts_sold * cost_per_tshirt
def investment_in_equipment : ℕ := total_revenue - total_cost_of_tshirts

-- Theorem statement
theorem investment_amount_correct : investment_in_equipment = 1411 := by
  sorry

end NUMINAMATH_GPT_investment_amount_correct_l1542_154272


namespace NUMINAMATH_GPT_movies_watched_total_l1542_154218

theorem movies_watched_total :
  ∀ (Timothy2009 Theresa2009 Timothy2010 Theresa2010 total : ℕ),
    Timothy2009 = 24 →
    Timothy2010 = Timothy2009 + 7 →
    Theresa2010 = 2 * Timothy2010 →
    Theresa2009 = Timothy2009 / 2 →
    total = Timothy2009 + Timothy2010 + Theresa2009 + Theresa2010 →
    total = 129 :=
by
  intros Timothy2009 Theresa2009 Timothy2010 Theresa2010 total
  sorry

end NUMINAMATH_GPT_movies_watched_total_l1542_154218


namespace NUMINAMATH_GPT_find_m_l1542_154235

-- Definitions based on conditions in the problem
def f (x : ℝ) := 4 * x + 7

-- Theorem statement to prove m = 3/4 given the conditions
theorem find_m (m : ℝ) :
  (∀ x : ℝ, f (1/2 * x - 1) = 2 * x + 3) →
  f (m - 1) = 6 →
  m = 3 / 4 :=
by
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_find_m_l1542_154235


namespace NUMINAMATH_GPT_inequality_proof_l1542_154208

variable {a b c d : ℝ}

theorem inequality_proof
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_pos_d : 0 < d)
  (h_inequality : a / b < c / d) :
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1542_154208


namespace NUMINAMATH_GPT_present_age_of_son_l1542_154267

/-- A man is 46 years older than his son and in two years, the man's age will be twice the age of his son. Prove that the present age of the son is 44. -/
theorem present_age_of_son (M S : ℕ) (h1 : M = S + 46) (h2 : M + 2 = 2 * (S + 2)) : S = 44 :=
by {
  sorry
}

end NUMINAMATH_GPT_present_age_of_son_l1542_154267


namespace NUMINAMATH_GPT_cost_difference_proof_l1542_154241

-- Define the cost per copy at print shop X
def cost_per_copy_X : ℝ := 1.25

-- Define the cost per copy at print shop Y
def cost_per_copy_Y : ℝ := 2.75

-- Define the number of copies
def number_of_copies : ℝ := 60

-- Define the total cost at print shop X
def total_cost_X : ℝ := cost_per_copy_X * number_of_copies

-- Define the total cost at print shop Y
def total_cost_Y : ℝ := cost_per_copy_Y * number_of_copies

-- Define the difference in cost between print shop Y and print shop X
def cost_difference : ℝ := total_cost_Y - total_cost_X

-- The theorem statement proving the cost difference is $90
theorem cost_difference_proof : cost_difference = 90 := by
  sorry

end NUMINAMATH_GPT_cost_difference_proof_l1542_154241


namespace NUMINAMATH_GPT_complex_number_solution_l1542_154216

open Complex

theorem complex_number_solution (z : ℂ) (h : (z - 2 * I) * (2 - I) = 5) : z = 2 + 3 * I :=
  sorry

end NUMINAMATH_GPT_complex_number_solution_l1542_154216


namespace NUMINAMATH_GPT_williams_farm_tax_l1542_154270

variables (T : ℝ)
variables (tax_collected : ℝ := 3840)
variables (percentage_williams_land : ℝ := 0.5)
variables (percentage_taxable_land : ℝ := 0.25)

theorem williams_farm_tax : (percentage_williams_land * tax_collected) = 1920 := by
  sorry

end NUMINAMATH_GPT_williams_farm_tax_l1542_154270


namespace NUMINAMATH_GPT_parabola_hyperbola_focus_l1542_154274

/-- 
Proof problem: If the focus of the parabola y^2 = 2px coincides with the right focus of the hyperbola x^2/3 - y^2/1 = 1, then p = 2.
-/
theorem parabola_hyperbola_focus (p : ℝ) :
    ∀ (focus_parabola : ℝ × ℝ) (focus_hyperbola : ℝ × ℝ),
      (focus_parabola = (p, 0)) →
      (focus_hyperbola = (2, 0)) →
      (focus_parabola = focus_hyperbola) →
        p = 2 :=
by
  intros focus_parabola focus_hyperbola h1 h2 h3
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_focus_l1542_154274


namespace NUMINAMATH_GPT_locus_of_circle_centers_l1542_154246

theorem locus_of_circle_centers (a : ℝ) (x0 y0 : ℝ) :
  { (α, β) | (x0 - α)^2 + (y0 - β)^2 = a^2 } = 
  { (x, y) | (x - x0)^2 + (y - y0)^2 = a^2 } :=
by
  sorry

end NUMINAMATH_GPT_locus_of_circle_centers_l1542_154246


namespace NUMINAMATH_GPT_prism_volume_l1542_154259

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : a * c = 84) : a * b * c = 1572 :=
by
  sorry

end NUMINAMATH_GPT_prism_volume_l1542_154259


namespace NUMINAMATH_GPT_melanie_average_speed_l1542_154225

theorem melanie_average_speed
  (bike_distance run_distance total_time : ℝ)
  (h_bike : bike_distance = 15)
  (h_run : run_distance = 5)
  (h_time : total_time = 4) :
  (bike_distance + run_distance) / total_time = 5 :=
by
  sorry

end NUMINAMATH_GPT_melanie_average_speed_l1542_154225


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_l1542_154286

theorem arithmetic_sequence_a3 :
  ∃ (a : ℕ → ℝ) (d : ℝ), 
    (∀ n, a n = 2 + (n - 1) * d) ∧
    (a 1 = 2) ∧
    (a 5 = a 4 + 2) →
    a 3 = 6 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a3_l1542_154286


namespace NUMINAMATH_GPT_abs_neg_three_l1542_154271

theorem abs_neg_three : |(-3 : ℝ)| = 3 := 
by
  -- The proof would go here, but we skip it for this exercise.
  sorry

end NUMINAMATH_GPT_abs_neg_three_l1542_154271


namespace NUMINAMATH_GPT_women_fraction_half_l1542_154287

theorem women_fraction_half
  (total_people : ℕ)
  (married_fraction : ℝ)
  (max_unmarried_women : ℕ)
  (total_people_eq : total_people = 80)
  (married_fraction_eq : married_fraction = 1 / 2)
  (max_unmarried_women_eq : max_unmarried_women = 32) :
  (∃ (women_fraction : ℝ), women_fraction = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_women_fraction_half_l1542_154287


namespace NUMINAMATH_GPT_sum_series_eq_l1542_154262

theorem sum_series_eq : 
  ∑' n : ℕ, (n + 1) * (1 / 3 : ℝ)^n = 9 / 4 :=
by sorry

end NUMINAMATH_GPT_sum_series_eq_l1542_154262


namespace NUMINAMATH_GPT_solve_Q1_l1542_154298

noncomputable def Q1 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y + y * f x) = f x + f y + x * f y

theorem solve_Q1 :
  ∀ f : ℝ → ℝ, Q1 f → f = (id : ℝ → ℝ) :=
  by sorry

end NUMINAMATH_GPT_solve_Q1_l1542_154298


namespace NUMINAMATH_GPT_total_money_earned_l1542_154254

def earning_per_question : ℝ := 0.2
def questions_per_survey : ℕ := 10
def surveys_on_monday : ℕ := 3
def surveys_on_tuesday : ℕ := 4

theorem total_money_earned :
  earning_per_question * (questions_per_survey * (surveys_on_monday + surveys_on_tuesday)) = 14 := by
  sorry

end NUMINAMATH_GPT_total_money_earned_l1542_154254


namespace NUMINAMATH_GPT_total_toys_per_week_l1542_154256

def toys_per_day := 1100
def working_days_per_week := 5

theorem total_toys_per_week : toys_per_day * working_days_per_week = 5500 :=
by
  sorry

end NUMINAMATH_GPT_total_toys_per_week_l1542_154256


namespace NUMINAMATH_GPT_egg_processing_l1542_154253

theorem egg_processing (E : ℕ) 
  (h1 : (24 / 25) * E + 12 = (99 / 100) * E) : 
  E = 400 :=
sorry

end NUMINAMATH_GPT_egg_processing_l1542_154253


namespace NUMINAMATH_GPT_average_age_of_women_is_37_33_l1542_154251

noncomputable def women_average_age (A : ℝ) : ℝ :=
  let total_age_men := 12 * A
  let removed_men_age := (25 : ℝ) + 15 + 30
  let new_average := A + 3.5
  let total_age_with_women := 12 * new_average
  let total_age_women := total_age_with_women -  (total_age_men - removed_men_age)
  total_age_women / 3

theorem average_age_of_women_is_37_33 (A : ℝ) (h_avg : women_average_age A = 37.33) :
  true :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_women_is_37_33_l1542_154251


namespace NUMINAMATH_GPT_distance_between_A_and_B_l1542_154221

def rowing_speed_still_water : ℝ := 10
def round_trip_time : ℝ := 5
def stream_speed : ℝ := 2

theorem distance_between_A_and_B : 
  ∃ x : ℝ, 
    (x / (rowing_speed_still_water - stream_speed) + x / (rowing_speed_still_water + stream_speed) = round_trip_time) 
    ∧ x = 24 :=
sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l1542_154221


namespace NUMINAMATH_GPT_cistern_total_wet_surface_area_l1542_154245

/-- Given a cistern with length 6 meters, width 4 meters, and water depth 1.25 meters,
    the total area of the wet surface is 49 square meters. -/
theorem cistern_total_wet_surface_area
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (h_length : length = 6) (h_width : width = 4) (h_depth : depth = 1.25) :
  (length * width) + 2 * (length * depth) + 2 * (width * depth) = 49 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_cistern_total_wet_surface_area_l1542_154245


namespace NUMINAMATH_GPT_contest_B_third_place_4_competitions_l1542_154210

/-- Given conditions:
1. There are three contestants: A, B, and C.
2. Scores for the first three places in each knowledge competition are \(a\), \(b\), and \(c\) where \(a > b > c\) and \(a, b, c ∈ ℕ^*\).
3. The final score of A is 26 points.
4. The final scores of both B and C are 11 points.
5. Contestant B won first place in one of the competitions.
Prove that Contestant B won third place in four competitions.
-/
theorem contest_B_third_place_4_competitions
  (a b c : ℕ)
  (ha : a > b)
  (hb : b > c)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hc_pos : 0 < c)
  (hA_score : a + a + a + a + b + c = 26)
  (hB_score : a + c + c + c + c + b = 11)
  (hC_score : b + b + b + b + c + c = 11) :
  ∃ n1 n3 : ℕ,
    n1 = 1 ∧ n3 = 4 ∧
    ∃ k m l p1 p2 : ℕ,
      n1 * a + k * a + l * a + m * a + p1 * a + p2 * a + p1 * b + k * b + p2 * b + n3 * c = 11 := sorry

end NUMINAMATH_GPT_contest_B_third_place_4_competitions_l1542_154210


namespace NUMINAMATH_GPT_negate_statement_l1542_154273

variable (Students Teachers : Type)
variable (Patient : Students → Prop)
variable (PatientT : Teachers → Prop)
variable (a : ∀ t : Teachers, PatientT t)
variable (b : ∃ t : Teachers, PatientT t)
variable (c : ∀ s : Students, ¬ Patient s)
variable (d : ∀ s : Students, ¬ Patient s)
variable (e : ∃ s : Students, ¬ Patient s)
variable (f : ∀ s : Students, Patient s)

theorem negate_statement : (∃ s : Students, ¬ Patient s) ↔ ¬ (∀ s : Students, Patient s) :=
by sorry

end NUMINAMATH_GPT_negate_statement_l1542_154273


namespace NUMINAMATH_GPT_max_value_on_interval_l1542_154217

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_on_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f x ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_max_value_on_interval_l1542_154217


namespace NUMINAMATH_GPT_max_value_of_e_n_l1542_154213

def b (n : ℕ) : ℕ := (8^n - 1) / 7
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem max_value_of_e_n : ∀ n : ℕ, e n = 1 := 
by
  sorry

end NUMINAMATH_GPT_max_value_of_e_n_l1542_154213


namespace NUMINAMATH_GPT_total_donation_l1542_154261

theorem total_donation : 2 + 6 + 2 + 8 = 18 := 
by sorry

end NUMINAMATH_GPT_total_donation_l1542_154261


namespace NUMINAMATH_GPT_pairs_of_real_numbers_l1542_154226

theorem pairs_of_real_numbers (a b : ℝ) (h : ∀ (n : ℕ), n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)) :
  a = 0 ∨ b = 0 ∨ a = b ∨ (∃ m n : ℤ, a = (m : ℝ) ∧ b = (n : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_pairs_of_real_numbers_l1542_154226


namespace NUMINAMATH_GPT_tiles_needed_l1542_154233

/--
A rectangular swimming pool is 20m long, 8m wide, and 1.5m deep. 
Each tile used to cover the pool has a side length of 2dm. 
We need to prove the number of tiles required to cover the bottom and all four sides of the pool.
-/
theorem tiles_needed (pool_length pool_width pool_depth : ℝ) (tile_side : ℝ) 
  (h1 : pool_length = 20) (h2 : pool_width = 8) (h3 : pool_depth = 1.5) 
  (h4 : tile_side = 0.2) : 
  (pool_length * pool_width + 2 * pool_length * pool_depth + 2 * pool_width * pool_depth) / (tile_side * tile_side) = 6100 :=
by
  sorry

end NUMINAMATH_GPT_tiles_needed_l1542_154233


namespace NUMINAMATH_GPT_flagpole_break_height_l1542_154297

theorem flagpole_break_height (h h_break distance : ℝ) (h_pos : 0 < h) (h_break_pos : 0 < h_break)
  (h_flagpole : h = 8) (d_distance : distance = 3) (h_relationship : (h_break ^ 2 + distance^2) = (h - h_break)^2) :
  h_break = Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_flagpole_break_height_l1542_154297


namespace NUMINAMATH_GPT_square_side_length_l1542_154290

theorem square_side_length {s : ℝ} (h1 : 4 * s = 60) : s = 15 := 
by
  linarith

end NUMINAMATH_GPT_square_side_length_l1542_154290


namespace NUMINAMATH_GPT_remaining_time_for_P_l1542_154269

theorem remaining_time_for_P 
  (P_rate : ℝ) (Q_rate : ℝ) (together_time : ℝ) (remaining_time_minutes : ℝ)
  (hP_rate : P_rate = 1 / 3) 
  (hQ_rate : Q_rate = 1 / 18) 
  (h_together_time : together_time = 2) 
  (h_remaining_time_minutes : remaining_time_minutes = 40) :
  (((P_rate + Q_rate) * together_time) + P_rate * (remaining_time_minutes / 60)) = 1 :=
by  rw [hP_rate, hQ_rate, h_together_time, h_remaining_time_minutes]
    admit

end NUMINAMATH_GPT_remaining_time_for_P_l1542_154269


namespace NUMINAMATH_GPT_marla_colors_green_squares_l1542_154284

-- Condition 1: Grid dimensions
def num_rows : ℕ := 10
def num_cols : ℕ := 15

-- Condition 2: Red squares
def red_rows : ℕ := 4
def red_squares_per_row : ℕ := 6
def red_squares : ℕ := red_rows * red_squares_per_row

-- Condition 3: Blue rows (first 2 and last 2)
def blue_rows : ℕ := 2 + 2
def blue_squares_per_row : ℕ := num_cols
def blue_squares : ℕ := blue_rows * blue_squares_per_row

-- Derived information
def total_squares : ℕ := num_rows * num_cols
def non_green_squares : ℕ := red_squares + blue_squares

-- The Lemma to prove
theorem marla_colors_green_squares : total_squares - non_green_squares = 66 := by
  sorry

end NUMINAMATH_GPT_marla_colors_green_squares_l1542_154284


namespace NUMINAMATH_GPT_quotient_multiple_of_y_l1542_154299

theorem quotient_multiple_of_y (x y m : ℤ) (h1 : x = 11 * y + 4) (h2 : 2 * x = 8 * m * y + 3) (h3 : 13 * y - x = 1) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_quotient_multiple_of_y_l1542_154299


namespace NUMINAMATH_GPT_Shekar_weighted_average_l1542_154200

def score_weighted_sum (scores_weights : List (ℕ × ℚ)) : ℚ :=
  scores_weights.foldl (fun acc sw => acc + (sw.1 * sw.2 : ℚ)) 0

def Shekar_scores_weights : List (ℕ × ℚ) :=
  [(76, 0.20), (65, 0.15), (82, 0.10), (67, 0.15), (55, 0.10), (89, 0.05), (74, 0.05),
   (63, 0.10), (78, 0.05), (71, 0.05)]

theorem Shekar_weighted_average : score_weighted_sum Shekar_scores_weights = 70.55 := by
  sorry

end NUMINAMATH_GPT_Shekar_weighted_average_l1542_154200
