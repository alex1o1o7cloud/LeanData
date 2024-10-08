import Mathlib

namespace certain_number_is_two_l135_135179

theorem certain_number_is_two (n : ℕ) 
  (h1 : 1 = 62) 
  (h2 : 363 = 3634) 
  (h3 : 3634 = n) 
  (h4 : n = 365) 
  (h5 : 36 = 2) : 
  n = 2 := 
by 
  sorry

end certain_number_is_two_l135_135179


namespace cube_diagonal_length_l135_135817

theorem cube_diagonal_length
  (side_length : ℝ)
  (h_side_length : side_length = 15) :
  ∃ d : ℝ, d = side_length * Real.sqrt 3 :=
by
  sorry

end cube_diagonal_length_l135_135817


namespace eight_mul_eleven_and_one_fourth_l135_135262

theorem eight_mul_eleven_and_one_fourth : 8 * (11 + (1 / 4 : ℝ)) = 90 := by
  sorry

end eight_mul_eleven_and_one_fourth_l135_135262


namespace value_of_x_l135_135017

theorem value_of_x (x y : ℝ) (h₁ : x = y - 0.10 * y) (h₂ : y = 125 + 0.10 * 125) : x = 123.75 := 
by
  sorry

end value_of_x_l135_135017


namespace tea_to_cheese_ratio_l135_135694

-- Definitions based on conditions
def total_cost : ℝ := 21
def tea_cost : ℝ := 10
def butter_to_cheese_ratio : ℝ := 0.8
def bread_to_butter_ratio : ℝ := 0.5

-- Main theorem statement
theorem tea_to_cheese_ratio (B C Br : ℝ) (hBr : Br = B * bread_to_butter_ratio) (hB : B = butter_to_cheese_ratio * C) (hTotal : B + Br + C + tea_cost = total_cost) :
  10 / C = 2 :=
  sorry

end tea_to_cheese_ratio_l135_135694


namespace probability_B_and_C_exactly_two_out_of_A_B_C_l135_135950

variables (A B C : Prop)
noncomputable def P : Prop → ℚ := sorry

axiom hA : P A = 3 / 4
axiom hAC : P (¬ A ∧ ¬ C) = 1 / 12
axiom hBC : P (B ∧ C) = 1 / 4

theorem probability_B_and_C : P B = 3 / 8 ∧ P C = 2 / 3 :=
sorry

theorem exactly_two_out_of_A_B_C : 
  P (A ∧ B ∧ ¬ C) + P (A ∧ ¬ B ∧ C) + P (¬ A ∧ B ∧ C) = 15 / 32 :=
sorry

end probability_B_and_C_exactly_two_out_of_A_B_C_l135_135950


namespace percentage_increase_l135_135293

theorem percentage_increase (X Y Z : ℝ) (h1 : X = 1.25 * Y) (h2 : Z = 100) (h3 : X + Y + Z = 370) :
  ((Y - Z) / Z) * 100 = 20 :=
by
  sorry

end percentage_increase_l135_135293


namespace ratio_S3_S9_l135_135335

noncomputable def Sn (a r : ℝ) (n : ℕ) : ℝ := (a * (1 - r ^ n)) / (1 - r)

theorem ratio_S3_S9 (a r : ℝ) (h1 : r ≠ 1) (h2 : Sn a r 6 = 3 * Sn a r 3) :
  Sn a r 3 / Sn a r 9 = 1 / 7 :=
by
  sorry

end ratio_S3_S9_l135_135335


namespace squirrel_spiral_distance_l135_135593

/-- The squirrel runs up a cylindrical post in a perfect spiral path, making one circuit for each rise of 4 feet.
Given the post is 16 feet tall and 3 feet in circumference, the total distance traveled by the squirrel is 20 feet. -/
theorem squirrel_spiral_distance :
  let height : ℝ := 16
  let circumference : ℝ := 3
  let rise_per_circuit : ℝ := 4
  let number_of_circuits := height / rise_per_circuit
  let distance_per_circuit := (circumference^2 + rise_per_circuit^2).sqrt
  number_of_circuits * distance_per_circuit = 20 := by
  sorry

end squirrel_spiral_distance_l135_135593


namespace negation_proposition_of_cube_of_odd_is_odd_l135_135756

def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem negation_proposition_of_cube_of_odd_is_odd :
  (¬ ∀ n : ℤ, odd n → odd (n^3)) ↔ (∃ n : ℤ, odd n ∧ ¬ odd (n^3)) :=
by
  sorry

end negation_proposition_of_cube_of_odd_is_odd_l135_135756


namespace convex_polygon_num_sides_l135_135101

theorem convex_polygon_num_sides (n : ℕ) 
  (h1 : ∀ (i : ℕ), i < n → 120 + i * 5 < 180) 
  (h2 : (n - 2) * 180 = n * (240 + (n - 1) * 5) / 2) : 
  n = 9 :=
sorry

end convex_polygon_num_sides_l135_135101


namespace problem_part1_problem_part2_area_height_l135_135063

theorem problem_part1 (x y : ℝ) (h : abs (x - 4 - 2 * Real.sqrt 2) + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0) : 
  x * y ^ 2 - x ^ 2 * y = -32 * Real.sqrt 2 := 
  sorry

theorem problem_part2_area_height (x y : ℝ) (h : abs (x - 4 - 2 * Real.sqrt 2) + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0) :
  let side_length := Real.sqrt 12
  let area := (1 / 2) * x * y
  let height := area / side_length
  area = 4 ∧ height = (2 * Real.sqrt 3) / 3 := 
  sorry

end problem_part1_problem_part2_area_height_l135_135063


namespace train_length_l135_135223

noncomputable section

-- Define the variables involved in the problem.
def train_length_cross_signal (V : ℝ) : ℝ := V * 18
def train_speed_cross_platform (L : ℝ) (platform_length : ℝ) : ℝ := (L + platform_length) / 40

-- Define the main theorem to prove the length of the train.
theorem train_length (V L : ℝ) (platform_length : ℝ) (h1 : L = V * 18)
(h2 : L + platform_length = V * 40) (h3 : platform_length = 366.67) :
L = 300 := 
by
  sorry

end train_length_l135_135223


namespace hyperbola_real_axis_length_l135_135491

theorem hyperbola_real_axis_length (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : ∀ x y : ℝ, x = 1 → y = 2 → (x^2 / (a^2)) - (y^2 / (b^2)) = 1)
  (h_parabola : ∀ y : ℝ, y = 2 → (y^2) = 4 * 1)
  (h_focus : (1, 2) = (1, 2))
  (h_eq : a^2 + b^2 = 1) :
  2 * a = 2 * (Real.sqrt 2 - 1) :=
by 
-- Skipping the proof part
sorry

end hyperbola_real_axis_length_l135_135491


namespace megs_cat_weight_l135_135605

/-- The ratio of the weight of Meg's cat to Anne's cat is 5:7 and Anne's cat weighs 8 kg more than Meg's cat. Prove that the weight of Meg's cat is 20 kg. -/
theorem megs_cat_weight
  (M A : ℝ)
  (h1 : M / A = 5 / 7)
  (h2 : A = M + 8) :
  M = 20 :=
sorry

end megs_cat_weight_l135_135605


namespace chairs_carried_per_trip_l135_135921

theorem chairs_carried_per_trip (x : ℕ) (friends : ℕ) (trips : ℕ) (total_chairs : ℕ) 
  (h1 : friends = 4) (h2 : trips = 10) (h3 : total_chairs = 250) 
  (h4 : 5 * (trips * x) = total_chairs) : x = 5 :=
by sorry

end chairs_carried_per_trip_l135_135921


namespace tangent_line_eq_a1_max_value_f_a_gt_1_div_5_l135_135078

noncomputable def f (a b x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + b * x + 1)

noncomputable def f_prime (a b x : ℝ) : ℝ := 
  -Real.exp (-x) * (a * x^2 + b * x + 1) + Real.exp (-x) * (2 * a * x + b)

theorem tangent_line_eq_a1 (b : ℝ) (h : f_prime 1 b (-1) = 0) : 
  ∃ m q, m = 1 ∧ q = 1 ∧ ∀ y, y = f 1 b 0 + m * y := sorry

theorem max_value_f_a_gt_1_div_5 (a b : ℝ) 
  (h_gt : a > 1/5) 
  (h_fp_eq : f_prime a b (-1) = 0)
  (h_max : ∀ x, -1 ≤ x ∧ x ≤ 1 → f a b x ≤ 4 * Real.exp 1) : 
  a = (24 * Real.exp 2 - 9) / 15 ∧ b = (12 * Real.exp 2 - 2) / 5 := sorry

end tangent_line_eq_a1_max_value_f_a_gt_1_div_5_l135_135078


namespace find_a_plus_b_l135_135574

theorem find_a_plus_b (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : a^2 - b^4 = 2009) : a + b = 47 :=
by
  sorry

end find_a_plus_b_l135_135574


namespace find_number_of_sides_l135_135838

theorem find_number_of_sides (n : ℕ) (h : n - (n * (n - 3)) / 2 = 3) : n = 3 := 
sorry

end find_number_of_sides_l135_135838


namespace probability_green_or_yellow_l135_135032

def total_marbles (green yellow red blue : Nat) : Nat :=
  green + yellow + red + blue

def marble_probability (green yellow red blue : Nat) : Rat :=
  (green + yellow) / (total_marbles green yellow red blue)

theorem probability_green_or_yellow :
  let green := 4
  let yellow := 3
  let red := 4
  let blue := 2
  marble_probability green yellow red blue = 7 / 13 := by
  sorry

end probability_green_or_yellow_l135_135032


namespace find_a_l135_135628

-- Define the polynomial f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 - 0.8

-- Define the intermediate values v_0, v_1, and v_2 using Horner's method
def v_0 (a : ℝ) : ℝ := a
def v_1 (a : ℝ) (x : ℝ) : ℝ := v_0 a * x + 2
def v_2 (a : ℝ) (x : ℝ) : ℝ := v_1 a x * x + 3.5 * x - 2.6 * x + 13.5

-- The condition for v_2 when x = 5
axiom v2_value (a : ℝ) : v_2 a 5 = 123.5

-- Prove that a = 4
theorem find_a : ∃ a : ℝ, v_2 a 5 = 123.5 ∧ a = 4 := by
  sorry

end find_a_l135_135628


namespace direction_vector_of_line_l135_135253

theorem direction_vector_of_line : ∃ Δx Δy : ℚ, y = - (1/2) * x + 1 → Δx = 2 ∧ Δy = -1 :=
sorry

end direction_vector_of_line_l135_135253


namespace fruit_vendor_total_l135_135870

theorem fruit_vendor_total (lemons_dozen avocados_dozen : ℝ) (dozen_size : ℝ) 
  (lemons : ℝ) (avocados : ℝ) (total_fruits : ℝ) 
  (h1 : lemons_dozen = 2.5) (h2 : avocados_dozen = 5) 
  (h3 : dozen_size = 12) (h4 : lemons = lemons_dozen * dozen_size) 
  (h5 : avocados = avocados_dozen * dozen_size) 
  (h6 : total_fruits = lemons + avocados) : 
  total_fruits = 90 := 
sorry

end fruit_vendor_total_l135_135870


namespace miss_adamson_num_classes_l135_135046

theorem miss_adamson_num_classes
  (students_per_class : ℕ)
  (sheets_per_student : ℕ)
  (total_sheets : ℕ)
  (h1 : students_per_class = 20)
  (h2 : sheets_per_student = 5)
  (h3 : total_sheets = 400) :
  let sheets_per_class := sheets_per_student * students_per_class
  let num_classes := total_sheets / sheets_per_class
  num_classes = 4 :=
by
  sorry

end miss_adamson_num_classes_l135_135046


namespace bob_paid_correctly_l135_135461

-- Define the variables involved
def alice_acorns : ℕ := 3600
def price_per_acorn : ℕ := 15
def multiplier : ℕ := 9
def total_amount_alice_paid : ℕ := alice_acorns * price_per_acorn

-- Define Bob's payment amount
def bob_payment : ℕ := total_amount_alice_paid / multiplier

-- The main theorem
theorem bob_paid_correctly : bob_payment = 6000 := by
  sorry

end bob_paid_correctly_l135_135461


namespace find_a_b_and_tangent_line_l135_135288

noncomputable def f (a b x : ℝ) := x^3 + 2 * a * x^2 + b * x + a
noncomputable def g (x : ℝ) := x^2 - 3 * x + 2
noncomputable def f' (a b x : ℝ) := 3 * x^2 + 4 * a * x + b
noncomputable def g' (x : ℝ) := 2 * x - 3

theorem find_a_b_and_tangent_line (a b : ℝ) :
  f a b 2 = 0 ∧ g 2 = 0 ∧ f' a b 2 = 1 ∧ g' 2 = 1 → (a = -2 ∧ b = 5 ∧ ∀ x y : ℝ, y = x - 2 ↔ x - y - 2 = 0) :=
by
  intro h
  sorry

end find_a_b_and_tangent_line_l135_135288


namespace simplification_l135_135966

theorem simplification (a b c : ℤ) :
  (12 * a + 35 * b + 17 * c) + (13 * a - 15 * b + 8 * c) - (8 * a + 28 * b - 25 * c) = 17 * a - 8 * b + 50 * c :=
by
  sorry

end simplification_l135_135966


namespace factor_expression_l135_135275

theorem factor_expression (a : ℝ) : 
  49 * a ^ 3 + 245 * a ^ 2 + 588 * a = 49 * a * (a ^ 2 + 5 * a + 12) :=
by
  sorry

end factor_expression_l135_135275


namespace largest_y_coordinate_ellipse_l135_135414

theorem largest_y_coordinate_ellipse:
  (∀ x y : ℝ, (x^2 / 49) + ((y + 3)^2 / 25) = 1 → y ≤ 2)  ∧ 
  (∃ x : ℝ, (x^2 / 49) + ((2 + 3)^2 / 25) = 1) := sorry

end largest_y_coordinate_ellipse_l135_135414


namespace solution_set_inequality_l135_135513

theorem solution_set_inequality (x : ℝ) : (-2 * x + 3 < 0) ↔ (x > 3 / 2) := by 
  sorry

end solution_set_inequality_l135_135513


namespace find_original_number_l135_135787

theorem find_original_number :
  ∃ x : ℚ, (5 * (3 * x + 15) = 245) ∧ x = 34 / 3 := by
  sorry

end find_original_number_l135_135787


namespace water_consumption_per_hour_l135_135791

theorem water_consumption_per_hour 
  (W : ℝ) 
  (initial_water : ℝ := 20) 
  (initial_food : ℝ := 10) 
  (initial_gear : ℝ := 20) 
  (food_consumption_rate : ℝ := 1 / 3) 
  (hours : ℝ := 6) 
  (remaining_weight : ℝ := 34)
  (initial_weight := initial_water + initial_food + initial_gear)
  (consumed_water := W * hours)
  (consumed_food := food_consumption_rate * W * hours)
  (consumed_weight := consumed_water + consumed_food)
  (final_equation := initial_weight - consumed_weight)
  (correct_answer := 2) :
  final_equation = remaining_weight → W = correct_answer := 
by 
  sorry

end water_consumption_per_hour_l135_135791


namespace product_of_roots_in_range_l135_135967

noncomputable def f (x : ℝ) : ℝ := abs (abs (x - 1) - 1)

theorem product_of_roots_in_range (m : ℝ) (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∃ x1 x2 x3 x4 : ℝ, 
        f x1 = m ∧ 
        f x2 = m ∧ 
        f x3 = m ∧ 
        f x4 = m ∧ 
        x1 ≠ x2 ∧ 
        x1 ≠ x3 ∧ 
        x1 ≠ x4 ∧ 
        x2 ≠ x3 ∧ 
        x2 ≠ x4 ∧ 
        x3 ≠ x4) :
  ∃ p : ℝ, p = (m * (2 - m) * (m + 2) * (-m)) ∧ -3 < p ∧ p < 0 :=
sorry

end product_of_roots_in_range_l135_135967


namespace cost_per_ball_correct_l135_135906

-- Define the values given in the conditions
def total_amount_paid : ℝ := 4.62
def number_of_balls : ℝ := 3.0

-- Define the expected cost per ball according to the problem statement
def expected_cost_per_ball : ℝ := 1.54

-- Statement to prove that the cost per ball is as expected
theorem cost_per_ball_correct : (total_amount_paid / number_of_balls) = expected_cost_per_ball := 
sorry

end cost_per_ball_correct_l135_135906


namespace train_length_is_250_l135_135364

noncomputable def train_length (V₁ V₂ V₃ : ℕ) (T₁ T₂ T₃ : ℕ) : ℕ :=
  let S₁ := (V₁ * (5/18) * T₁)
  let S₂ := (V₂ * (5/18)* T₂)
  let S₃ := (V₃ * (5/18) * T₃)
  if S₁ = S₂ ∧ S₂ = S₃ then S₁ else 0

theorem train_length_is_250 :
  train_length 50 60 70 18 20 22 = 250 := by
  -- proof omitted
  sorry

end train_length_is_250_l135_135364


namespace chip_placement_count_l135_135154

def grid := Fin 4 × Fin 3

def grid_positions (n : Nat) := {s : Finset grid // s.card = n}

def no_direct_adjacency (positions : Finset grid) : Prop :=
  ∀ (x y : grid), x ∈ positions → y ∈ positions →
  (x.fst ≠ y.fst ∨ x.snd ≠ y.snd)

noncomputable def count_valid_placements : Nat :=
  -- Function to count valid placements
  sorry

theorem chip_placement_count :
  count_valid_placements = 4 :=
  sorry

end chip_placement_count_l135_135154


namespace sum_of_cubes_divisible_l135_135732

theorem sum_of_cubes_divisible (a b c : ℤ) (h : (a + b + c) % 3 = 0) : 
  (a^3 + b^3 + c^3) % 3 = 0 := 
by sorry

end sum_of_cubes_divisible_l135_135732


namespace max_sector_area_l135_135012

theorem max_sector_area (r l : ℝ) (hp : 2 * r + l = 40) : (1 / 2) * l * r ≤ 100 := 
by
  sorry

end max_sector_area_l135_135012


namespace number_of_red_dresses_l135_135531

-- Define context for Jane's dress shop problem
def dresses_problem (R B : Nat) : Prop :=
  R + B = 200 ∧ B = R + 34

-- Prove that the number of red dresses (R) should be 83
theorem number_of_red_dresses : ∃ R B : Nat, dresses_problem R B ∧ R = 83 :=
by
  sorry

end number_of_red_dresses_l135_135531


namespace james_planted_60_percent_l135_135954

theorem james_planted_60_percent :
  let total_trees := 2
  let plants_per_tree := 20
  let seeds_per_plant := 1
  let total_seeds := total_trees * plants_per_tree * seeds_per_plant
  let planted_trees := 24
  (planted_trees / total_seeds) * 100 = 60 := 
by
  sorry

end james_planted_60_percent_l135_135954


namespace youngest_child_cakes_l135_135264

theorem youngest_child_cakes : 
  let total_cakes := 60
  let oldest_cakes := (1 / 4 : ℚ) * total_cakes
  let second_oldest_cakes := (3 / 10 : ℚ) * total_cakes
  let middle_cakes := (1 / 6 : ℚ) * total_cakes
  let second_youngest_cakes := (1 / 5 : ℚ) * total_cakes
  let distributed_cakes := oldest_cakes + second_oldest_cakes + middle_cakes + second_youngest_cakes
  let youngest_cakes := total_cakes - distributed_cakes
  youngest_cakes = 5 := 
by
  exact sorry

end youngest_child_cakes_l135_135264


namespace sequence_term_is_100th_term_l135_135172

theorem sequence_term_is_100th_term (a : ℕ → ℝ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, a (n + 1) = 2 * a n / (a n + 2)) :
  (∃ n : ℕ, a n = 2 / 101) ∧ ((∃ n : ℕ, a n = 2 / 101) → n = 100) :=
by
  sorry

end sequence_term_is_100th_term_l135_135172


namespace julia_total_kids_l135_135631

def kidsMonday : ℕ := 7
def kidsTuesday : ℕ := 13
def kidsThursday : ℕ := 18
def kidsWednesdayCards : ℕ := 20
def kidsWednesdayHideAndSeek : ℕ := 11
def kidsWednesdayPuzzle : ℕ := 9
def kidsFridayBoardGame : ℕ := 15
def kidsFridayDrawingCompetition : ℕ := 12

theorem julia_total_kids : 
  kidsMonday + kidsTuesday + kidsThursday + kidsWednesdayCards + kidsWednesdayHideAndSeek + kidsWednesdayPuzzle + kidsFridayBoardGame + kidsFridayDrawingCompetition = 105 :=
by
  sorry

end julia_total_kids_l135_135631


namespace simplify_expression_l135_135410

theorem simplify_expression (x y : ℝ) (h : x = -3) : 
  x * (x - 4) * (x + 4) - (x + 3) * (x^2 - 6 * x + 9) + 5 * x^3 * y^2 / (x^2 * y^2) = -66 :=
by
  sorry

end simplify_expression_l135_135410


namespace power_of_three_l135_135398

theorem power_of_three (a b : ℕ) (h1 : 360 = (2^3) * (3^2) * (5^1))
  (h2 : 2^a ∣ 360 ∧ ∀ n, 2^n ∣ 360 → n ≤ a)
  (h3 : 5^b ∣ 360 ∧ ∀ n, 5^n ∣ 360 → n ≤ b) :
  (1/3 : ℝ)^(b - a) = 9 :=
by sorry

end power_of_three_l135_135398


namespace line_slope_translation_l135_135697

theorem line_slope_translation (k : ℝ) (b : ℝ) :
  (∀ x y : ℝ, y = k * x + b → y = k * (x - 3) + (b + 2)) → k = 2 / 3 :=
by
  intro h
  sorry

end line_slope_translation_l135_135697


namespace solve_equation_l135_135607

-- Define the equation to be solved
def equation (x : ℝ) : Prop := (x + 2)^4 + (x - 4)^4 = 272

-- State the theorem we want to prove
theorem solve_equation : ∃ x : ℝ, equation x :=
  sorry

end solve_equation_l135_135607


namespace intersection_M_N_l135_135405

open Set

def M : Set ℝ := { x | x^2 - 2*x - 3 < 0 }
def N : Set ℝ := { x | x >= 1 }

theorem intersection_M_N : M ∩ N = { x | 1 <= x ∧ x < 3 } :=
by
  sorry

end intersection_M_N_l135_135405


namespace find_some_number_l135_135186

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l135_135186


namespace problem_min_ineq_range_l135_135741

theorem problem_min_ineq_range (a b : ℝ) (x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x, 1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) ∧ (1 / a + 4 / b = 9) ∧ (-7 ≤ x ∧ x ≤ 11) :=
sorry

end problem_min_ineq_range_l135_135741


namespace problem_statement_l135_135341

-- Defining the propositions p and q as Boolean variables
variables (p q : Prop)

-- Assume the given conditions
theorem problem_statement (hnp : ¬¬p) (hnpq : ¬(p ∧ q)) : p ∧ ¬q :=
by {
  -- Derived steps to satisfy the conditions are implicit within this scope
  sorry
}

end problem_statement_l135_135341


namespace area_of_polygon_l135_135076

-- Define the conditions
variables (n : ℕ) (s : ℝ)
-- Given that polygon has 32 sides.
def sides := 32
-- Each side is congruent, and the total perimeter is 64 units.
def perimeter := 64
-- Side length of each side
def side_length := perimeter / sides

-- Area of the polygon we need to prove
def target_area := 96

theorem area_of_polygon : side_length * side_length * sides = target_area := 
by {
  -- Here proof would come in reality, we'll skip it by sorry for now.
  sorry
}

end area_of_polygon_l135_135076


namespace remainder_problem_l135_135665

theorem remainder_problem :
  ((98 * 103 + 7) % 12) = 1 :=
by
  sorry

end remainder_problem_l135_135665


namespace smallest_of_seven_consecutive_even_numbers_l135_135303

theorem smallest_of_seven_consecutive_even_numbers (n : ℤ) :
  (n - 6) + (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 448 → 
  (n - 6) = 58 :=
by
  sorry

end smallest_of_seven_consecutive_even_numbers_l135_135303


namespace find_a_l135_135742

-- Define point
structure Point where
  x : ℝ
  y : ℝ

-- Define curves
def C1 (a x : ℝ) : ℝ := a * x^3 + 1
def C2 (P : Point) : Prop := P.x^2 + P.y^2 = 5 / 2

-- Define the tangent slope function for curve C1
def tangent_slope_C1 (a x : ℝ) : ℝ := 3 * a * x^2

-- State the problem that we need to prove
theorem find_a (a x₀ y₀ : ℝ) (h1 : y₀ = C1 a x₀) (h2 : C2 ⟨x₀, y₀⟩) (h3 : y₀ = 3 * a * x₀^3) 
  (ha_pos : 0 < a) : a = 4 := 
  by
    sorry

end find_a_l135_135742


namespace students_neither_play_football_nor_cricket_l135_135551

theorem students_neither_play_football_nor_cricket
  (total_students football_players cricket_players both_players : ℕ)
  (h_total : total_students = 470)
  (h_football : football_players = 325)
  (h_cricket : cricket_players = 175)
  (h_both : both_players = 80) :
  (total_students - (football_players + cricket_players - both_players)) = 50 :=
by
  sorry

end students_neither_play_football_nor_cricket_l135_135551


namespace lighting_effect_improves_l135_135587

theorem lighting_effect_improves (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
    (a + m) / (b + m) > a / b := 
sorry

end lighting_effect_improves_l135_135587


namespace polynomial_simplification_l135_135891

theorem polynomial_simplification (w : ℝ) : 
  3 * w + 4 - 6 * w - 5 + 7 * w + 8 - 9 * w - 10 + 2 * w ^ 2 = 2 * w ^ 2 - 5 * w - 3 :=
by
  sorry

end polynomial_simplification_l135_135891


namespace tan_angle_equiv_tan_1230_l135_135231

theorem tan_angle_equiv_tan_1230 : ∃ n : ℤ, -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1230 * Real.pi / 180) :=
sorry

end tan_angle_equiv_tan_1230_l135_135231


namespace evaluate_expression_l135_135941

theorem evaluate_expression : (1023 * 1023) - (1022 * 1024) = 1 := by
  sorry

end evaluate_expression_l135_135941


namespace units_digit_6_power_l135_135360

theorem units_digit_6_power (n : ℕ) : (6^n % 10) = 6 :=
sorry

end units_digit_6_power_l135_135360


namespace green_folder_stickers_l135_135199

theorem green_folder_stickers (total_stickers red_sheets blue_sheets : ℕ) (red_sticker_per_sheet blue_sticker_per_sheet green_stickers_needed green_sheets : ℕ) :
  total_stickers = 60 →
  red_sticker_per_sheet = 3 →
  blue_sticker_per_sheet = 1 →
  red_sheets = 10 →
  blue_sheets = 10 →
  green_sheets = 10 →
  let red_stickers_total := red_sticker_per_sheet * red_sheets
  let blue_stickers_total := blue_sticker_per_sheet * blue_sheets
  let green_stickers_total := total_stickers - (red_stickers_total + blue_stickers_total)
  green_sticker_per_sheet = green_stickers_total / green_sheets →
  green_sticker_per_sheet = 2 := 
sorry

end green_folder_stickers_l135_135199


namespace distance_covered_l135_135245

theorem distance_covered (t : ℝ) (s_kmph : ℝ) (distance : ℝ) (h1 : t = 180) (h2 : s_kmph = 18) : 
  distance = 900 :=
by 
  sorry

end distance_covered_l135_135245


namespace original_number_of_employees_l135_135287

theorem original_number_of_employees (x : ℕ) 
  (h1 : 0.77 * (x : ℝ) = 328) : x = 427 :=
sorry

end original_number_of_employees_l135_135287


namespace trapezium_area_l135_135454

theorem trapezium_area (a b h : ℝ) (h_a : a = 20) (h_b : b = 18) (h_h : h = 10) : 
  (1 / 2) * (a + b) * h = 190 := 
by
  -- We provide the conditions:
  rw [h_a, h_b, h_h]
  -- The proof steps will be skipped using 'sorry'
  sorry

end trapezium_area_l135_135454


namespace mountain_height_correct_l135_135679

noncomputable def height_of_mountain : ℝ :=
  15 / (1 / Real.tan (Real.pi * 10 / 180) + 1 / Real.tan (Real.pi * 12 / 180))

theorem mountain_height_correct :
  abs (height_of_mountain - 1.445) < 0.001 :=
sorry

end mountain_height_correct_l135_135679


namespace alan_glasses_drank_l135_135999

-- Definition for the rate of drinking water
def glass_per_minutes := 1 / 20

-- Definition for the total time in minutes
def total_minutes := 5 * 60

-- Theorem stating the number of glasses Alan will drink in the given time
theorem alan_glasses_drank : (glass_per_minutes * total_minutes) = 15 :=
by 
  sorry

end alan_glasses_drank_l135_135999


namespace amount_of_second_alloy_used_l135_135309

variable (x : ℝ)

-- Conditions
def chromium_in_first_alloy : ℝ := 0.10 * 15
def chromium_in_second_alloy (x : ℝ) : ℝ := 0.06 * x
def total_weight (x : ℝ) : ℝ := 15 + x
def chromium_in_third_alloy (x : ℝ) : ℝ := 0.072 * (15 + x)

-- Proof statement
theorem amount_of_second_alloy_used :
  1.5 + 0.06 * x = 0.072 * (15 + x) → x = 35 := by
  sorry

end amount_of_second_alloy_used_l135_135309


namespace correct_reasoning_methods_l135_135915

-- Definitions based on conditions
def reasoning_1 : String := "Inductive reasoning"
def reasoning_2 : String := "Deductive reasoning"
def reasoning_3 : String := "Analogical reasoning"

-- Proposition stating that the correct answer is D
theorem correct_reasoning_methods :
  (reasoning_1 = "Inductive reasoning") ∧
  (reasoning_2 = "Deductive reasoning") ∧
  (reasoning_3 = "Analogical reasoning") ↔
  (choice = "D") :=
by sorry

end correct_reasoning_methods_l135_135915


namespace find_ab_l135_135184

theorem find_ab (a b : ℝ) 
  (period_cond : (π / b) = (2 * π / 5)) 
  (point_cond : a * Real.tan (5 * (π / 10) / 2) = 1) :
  a * b = 5 / 2 :=
sorry

end find_ab_l135_135184


namespace exam_items_count_l135_135027

theorem exam_items_count (x : ℝ) (hLiza : Liza_correct = 0.9 * x) (hRoseCorrect : Rose_correct = 0.9 * x + 2) (hRoseTotal : Rose_total = x) (hRoseIncorrect : Rose_incorrect = x - (0.9 * x + 2) ):
    Liza_correct + Rose_incorrect = Rose_total :=
by
    sorry

end exam_items_count_l135_135027


namespace num_5_digit_even_div_by_5_l135_135980

theorem num_5_digit_even_div_by_5 : ∃! (n : ℕ), n = 500 ∧ ∀ (d : ℕ), 
  10000 ≤ d ∧ d ≤ 99999 → 
  (∀ i, i ∈ [0, 1, 2, 3, 4] → ((d / 10^i) % 10) % 2 = 0) ∧
  (d % 10 = 0) → 
  n = 500 := sorry

end num_5_digit_even_div_by_5_l135_135980


namespace janet_spending_difference_l135_135633

-- Definitions for the conditions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℕ := 52

-- The theorem to be proven
theorem janet_spending_difference :
  (piano_hourly_rate * piano_hours_per_week * weeks_per_year - clarinet_hourly_rate * clarinet_hours_per_week * weeks_per_year) = 1040 :=
by
  sorry

end janet_spending_difference_l135_135633


namespace division_into_rectangles_l135_135003

theorem division_into_rectangles (figure : Type) (valid_division : figure → Prop) : (∃ ways, ways = 8) :=
by {
  -- assume given conditions related to valid_division using "figure"
  sorry
}

end division_into_rectangles_l135_135003


namespace f_value_neg_five_half_one_l135_135971

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom interval_definition : ∀ x, 0 < x ∧ x < 1 → f x = (4:ℝ) ^ x

-- The statement to prove
theorem f_value_neg_five_half_one : f (-5/2) + f 1 = -2 :=
by
  sorry

end f_value_neg_five_half_one_l135_135971


namespace largest_perimeter_regular_polygons_l135_135824

theorem largest_perimeter_regular_polygons :
  ∃ (p q r : ℕ), 
    (p ≥ 3 ∧ q ≥ 3 ∧ r >= 3) ∧
    (p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧
    (180 * (p - 2)/p + 180 * (q - 2)/q + 180 * (r - 2)/r = 360) ∧
    ((p + q + r - 6) = 9) :=
sorry

end largest_perimeter_regular_polygons_l135_135824


namespace scientific_notation_of_3395000_l135_135429

theorem scientific_notation_of_3395000 :
  3395000 = 3.395 * 10^6 :=
sorry

end scientific_notation_of_3395000_l135_135429


namespace abs_neg_three_l135_135650

theorem abs_neg_three : abs (-3) = 3 := 
by 
  -- Skipping proof with sorry
  sorry

end abs_neg_three_l135_135650


namespace oz_words_lost_l135_135083

theorem oz_words_lost (letters : Fin 64) (forbidden_letter : Fin 64) (h_forbidden : forbidden_letter.val = 6) : 
  let one_letter_words := 64 
  let two_letter_words := 64 * 64
  let one_letter_lost := if letters = forbidden_letter then 1 else 0
  let two_letter_lost := (if letters = forbidden_letter then 64 else 0) + (if letters = forbidden_letter then 64 else 0) 
  1 + two_letter_lost = 129 :=
by
  sorry

end oz_words_lost_l135_135083


namespace distance_traveled_l135_135516

-- Definition of the velocity function
def velocity (t : ℝ) : ℝ := 2 * t - 3

-- Prove the integral statement
theorem distance_traveled : 
  (∫ t in (0 : ℝ)..(5 : ℝ), abs (velocity t)) = 29 / 2 := by 
{ sorry }

end distance_traveled_l135_135516


namespace log_base_30_of_8_l135_135367

theorem log_base_30_of_8 (a b : ℝ) (h1 : Real.log 5 = a) (h2 : Real.log 3 = b) : 
  Real.logb 30 8 = (3 * (1 - a)) / (1 + b) :=
by
  sorry

end log_base_30_of_8_l135_135367


namespace sanghyeon_questions_l135_135065

variable (S : ℕ)

theorem sanghyeon_questions (h1 : S + (S + 5) = 43) : S = 19 :=
by
    sorry

end sanghyeon_questions_l135_135065


namespace find_a_l135_135800

theorem find_a (a : ℝ) (extreme_at_neg_2 : ∀ x : ℝ, (3 * a * x^2 + 2 * x) = 0 → x = -2) :
    a = 1 / 3 :=
sorry

end find_a_l135_135800


namespace cake_area_l135_135015

theorem cake_area (n : ℕ) (a area_per_piece : ℕ) 
  (h1 : n = 25) 
  (h2 : a = 16) 
  (h3 : area_per_piece = 4 * 4) 
  (h4 : a = area_per_piece) : 
  n * a = 400 := 
by
  sorry

end cake_area_l135_135015


namespace graph_symmetric_about_x_eq_pi_div_8_l135_135423

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem graph_symmetric_about_x_eq_pi_div_8 :
  ∀ x, f (π / 8 - x) = f (π / 8 + x) :=
sorry

end graph_symmetric_about_x_eq_pi_div_8_l135_135423


namespace find_positive_integers_l135_135959

noncomputable def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

theorem find_positive_integers (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
  (hab_c : is_power_of_two (a * b - c))
  (hbc_a : is_power_of_two (b * c - a))
  (hca_b : is_power_of_two (c * a - b)) :
  (a = 2 ∧ b = 2 ∧ c = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 3) ∨
  (a = 3 ∧ b = 5 ∧ c = 7) ∨
  (a = 2 ∧ b = 6 ∧ c = 11) :=
sorry

end find_positive_integers_l135_135959


namespace maximal_s_value_l135_135719

noncomputable def max_tiles_sum (a b c : ℕ) : ℕ := a + c

theorem maximal_s_value :
  ∃ s : ℕ, 
    ∃ a b c : ℕ, 
      4 * a + 4 * c + 5 * b = 3986000 ∧ 
      s = max_tiles_sum a b c ∧ 
      s = 996500 := 
    sorry

end maximal_s_value_l135_135719


namespace a_plus_c_eq_neg800_l135_135859

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem a_plus_c_eq_neg800 (a b c d : ℝ) (h1 : g (-a / 2) c d = 0)
  (h2 : f (-c / 2) a b = 0) (h3 : ∀ x, f x a b ≥ f (-a / 2) a b)
  (h4 : ∀ x, g x c d ≥ g (-c / 2) c d) (h5 : f (-a / 2) a b = g (-c / 2) c d)
  (h6 : f 200 a b = -200) (h7 : g 200 c d = -200) :
  a + c = -800 := sorry

end a_plus_c_eq_neg800_l135_135859


namespace train_speed_length_l135_135359

theorem train_speed_length (t1 t2 s : ℕ) (p : ℕ)
  (h1 : t1 = 7) 
  (h2 : t2 = 25) 
  (h3 : p = 378)
  (h4 : t2 - t1 = 18)
  (h5 : p / (t2 - t1) = 21) 
  (h6 : (p / (t2 - t1)) * t1 = 147) :
  (21, 147) = (21, 147) :=
by {
  sorry
}

end train_speed_length_l135_135359


namespace amount_saved_percent_l135_135813

variable (S : ℝ)

theorem amount_saved_percent :
  (0.165 * S) / (0.10 * S) * 100 = 165 := sorry

end amount_saved_percent_l135_135813


namespace trigonometric_identities_l135_135931

noncomputable def tan (θ : ℝ) : ℝ := Real.tan θ
noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ
noncomputable def cos (θ : ℝ) : ℝ := Real.cos θ

theorem trigonometric_identities (θ : ℝ) (h_tan : tan θ = 2) (h_identity : sin θ ^ 2 + cos θ ^ 2 = 1) :
    ((sin θ = 2 * Real.sqrt 5 / 5 ∧ cos θ = Real.sqrt 5 / 5) ∨ (sin θ = -2 * Real.sqrt 5 / 5 ∧ cos θ = -Real.sqrt 5 / 5)) ∧
    ((4 * sin θ - 3 * cos θ) / (6 * cos θ + 2 * sin θ) = 1 / 2) :=
by
  sorry

end trigonometric_identities_l135_135931


namespace incorrect_statement_D_l135_135408

/-
Define the conditions for the problem:
- A prism intersected by a plane.
- The intersection of a sphere and a plane when the plane is less than the radius.
- The intersection of a plane parallel to the base of a circular cone.
- The geometric solid formed by rotating a right triangle around one of its sides.
- The incorrectness of statement D.
-/

noncomputable def intersect_prism_with_plane (prism : Type) (plane : Type) : Prop := sorry

noncomputable def sphere_intersection (sphere_radius : ℝ) (distance_to_plane : ℝ) : Type := sorry

noncomputable def cone_intersection (cone : Type) (plane : Type) : Type := sorry

noncomputable def rotation_result (triangle : Type) (side : Type) : Type := sorry

theorem incorrect_statement_D :
  ¬(rotation_result RightTriangle Side = Cone) :=
sorry

end incorrect_statement_D_l135_135408


namespace ladder_length_difference_l135_135764

theorem ladder_length_difference :
  ∀ (flights : ℕ) (flight_height rope ladder_total_height : ℕ),
    flights = 3 →
    flight_height = 10 →
    rope = (flights * flight_height) / 2 →
    ladder_total_height = 70 →
    ladder_total_height - (flights * flight_height + rope) = 25 →
    ladder_total_height - (flights * flight_height) - rope = 10 :=
by
  intros
  sorry

end ladder_length_difference_l135_135764


namespace min_frac_sum_l135_135117

theorem min_frac_sum (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_eq : 2 * a + b = 1) : 
  (3 / b + 2 / a) = 7 + 4 * Real.sqrt 3 := 
sorry

end min_frac_sum_l135_135117


namespace weight_of_new_person_l135_135200

theorem weight_of_new_person
  (avg_increase : ℝ)
  (num_persons : ℕ)
  (replaced_weight : ℝ)
  (weight_increase_total : ℝ)
  (W : ℝ)
  (h1 : avg_increase = 4.5)
  (h2 : num_persons = 8)
  (h3 : replaced_weight = 65)
  (h4 : weight_increase_total = 8 * 4.5)
  (h5 : W = replaced_weight + weight_increase_total) :
  W = 101 :=
by
  sorry

end weight_of_new_person_l135_135200


namespace board_transformation_l135_135562

def transformation_possible (a b : ℕ) : Prop :=
  6 ∣ (a * b)

theorem board_transformation (a b : ℕ) (h₁ : 2 ≤ a) (h₂ : 2 ≤ b) : 
  transformation_possible a b ↔ 6 ∣ (a * b) := by
  sorry

end board_transformation_l135_135562


namespace perimeter_of_ABCD_l135_135395

theorem perimeter_of_ABCD
  (AD BC AB CD : ℕ)
  (hAD : AD = 4)
  (hAB : AB = 5)
  (hBC : BC = 10)
  (hCD : CD = 7)
  (hAD_lt_BC : AD < BC) :
  AD + AB + BC + CD = 26 :=
by
  -- Proof will be provided here.
  sorry

end perimeter_of_ABCD_l135_135395


namespace find_k_l135_135445

-- Definitions based on the problem conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 4)

-- Property of parallel vectors
def parallel (u v : ℝ × ℝ) : Prop := ∃ c : ℝ, u.1 = c * v.1 ∧ u.2 = c * v.2

-- Theorem statement equivalent to the problem
theorem find_k (k : ℝ) (h : parallel vector_a (vector_b k)) : k = -2 :=
sorry

end find_k_l135_135445


namespace initial_position_is_minus_one_l135_135021

def initial_position_of_A (A B C : ℤ) : Prop :=
  B = A - 3 ∧ C = B + 5 ∧ C = 1 ∧ A = -1

theorem initial_position_is_minus_one (A B C : ℤ) (h1 : B = A - 3) (h2 : C = B + 5) (h3 : C = 1) : A = -1 :=
  by sorry

end initial_position_is_minus_one_l135_135021


namespace part1_part2_l135_135312

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem part1 (m : ℝ) : (∃ x, deriv f x = 2 ∧ f x = 2 * x + m) → m = -Real.exp 1 :=
sorry

theorem part2 : ∀ x > 0, -1 / Real.exp 1 ≤ f x ∧ f x < Real.exp x / (2 * x) :=
sorry

end part1_part2_l135_135312


namespace linear_substitution_correct_l135_135176

theorem linear_substitution_correct (x y : ℝ) 
  (h1 : y = x - 1) 
  (h2 : x + 2 * y = 7) : 
  x + 2 * x - 2 = 7 := 
by
  sorry

end linear_substitution_correct_l135_135176


namespace relationship_y1_y2_y3_l135_135155

noncomputable def parabola_value (x m : ℝ) : ℝ := -x^2 - 4 * x + m

variable (m y1 y2 y3 : ℝ)

def point_A_on_parabola : Prop := y1 = parabola_value (-3) m
def point_B_on_parabola : Prop := y2 = parabola_value (-2) m
def point_C_on_parabola : Prop := y3 = parabola_value 1 m


theorem relationship_y1_y2_y3 (hA : point_A_on_parabola y1 m)
                              (hB : point_B_on_parabola y2 m)
                              (hC : point_C_on_parabola y3 m) :
  y2 > y1 ∧ y1 > y3 := 
  sorry

end relationship_y1_y2_y3_l135_135155


namespace principal_equivalence_l135_135901

-- Define the conditions
def SI : ℝ := 4020.75
def R : ℝ := 9
def T : ℝ := 5

-- Define the principal calculation
noncomputable def P := SI / (R * T / 100)

-- Prove that the principal P equals 8935
theorem principal_equivalence : P = 8935 := by
  sorry

end principal_equivalence_l135_135901


namespace no_integer_a_exists_l135_135616

theorem no_integer_a_exists (a x : ℤ)
  (h : x^3 - a * x^2 - 6 * a * x + a^2 - 3 = 0)
  (unique_sol : ∀ y : ℤ, (y^3 - a * y^2 - 6 * a * y + a^2 - 3 = 0 → y = x)) :
  false :=
by 
  sorry

end no_integer_a_exists_l135_135616


namespace train_speed_l135_135598

theorem train_speed
  (length_of_train : ℕ)
  (time_to_cross_bridge : ℕ)
  (length_of_bridge : ℕ)
  (speed_conversion_factor : ℕ)
  (H1 : length_of_train = 120)
  (H2 : time_to_cross_bridge = 30)
  (H3 : length_of_bridge = 255)
  (H4 : speed_conversion_factor = 36) : 
  (length_of_train + length_of_bridge) / (time_to_cross_bridge / speed_conversion_factor) = 45 :=
by
  sorry

end train_speed_l135_135598


namespace savings_fraction_l135_135449

theorem savings_fraction 
(P : ℝ) 
(f : ℝ) 
(h1 : P > 0) 
(h2 : 12 * f * P = 5 * (1 - f) * P) : 
    f = 5 / 17 :=
by
  sorry

end savings_fraction_l135_135449


namespace total_value_of_bills_in_cash_drawer_l135_135878

-- Definitions based on conditions
def total_bills := 54
def five_dollar_bills := 20
def twenty_dollar_bills := total_bills - five_dollar_bills
def value_of_five_dollar_bills := 5
def value_of_twenty_dollar_bills := 20
def total_value_of_five_dollar_bills := five_dollar_bills * value_of_five_dollar_bills
def total_value_of_twenty_dollar_bills := twenty_dollar_bills * value_of_twenty_dollar_bills

-- Statement to prove
theorem total_value_of_bills_in_cash_drawer :
  total_value_of_five_dollar_bills + total_value_of_twenty_dollar_bills = 780 :=
by
  -- Proof goes here
  sorry

end total_value_of_bills_in_cash_drawer_l135_135878


namespace find_g_of_2_l135_135601

-- Define the assumptions
variables (g : ℝ → ℝ)
axiom condition : ∀ x : ℝ, x ≠ 0 → 5 * g (1 / x) + (3 * g x) / x = Real.sqrt x

-- State the theorem to prove
theorem find_g_of_2 : g 2 = -(Real.sqrt 2) / 16 :=
by
  sorry

end find_g_of_2_l135_135601


namespace total_rubber_bands_l135_135888

theorem total_rubber_bands (harper_bands : ℕ) (brother_bands: ℕ):
  harper_bands = 15 →
  brother_bands = harper_bands - 6 →
  harper_bands + brother_bands = 24 :=
by
  intros h1 h2
  sorry

end total_rubber_bands_l135_135888


namespace probability_factor_less_than_eight_l135_135956

theorem probability_factor_less_than_eight (n : ℕ) (h72 : n = 72) :
  (∃ k < 8, k ∣ n) →
  (∃ p q, p/q = 5/12) :=
by
  sorry

end probability_factor_less_than_eight_l135_135956


namespace speed_with_stream_l135_135504

variable (V_m V_s : ℝ)

def against_speed : Prop := V_m - V_s = 13
def still_water_rate : Prop := V_m = 6

theorem speed_with_stream (h1 : against_speed V_m V_s) (h2 : still_water_rate V_m) : V_m + V_s = 13 := 
sorry

end speed_with_stream_l135_135504


namespace pain_subsided_days_l135_135559

-- Define the problem conditions in Lean
variable (x : ℕ) -- the number of days it takes for the pain to subside

-- Condition 1: The injury takes 5 times the pain subsiding period to fully heal
def injury_healing_days := 5 * x

-- Condition 2: James waits an additional 3 days after the injury is fully healed
def workout_waiting_days := injury_healing_days + 3

-- Condition 3: James waits another 3 weeks (21 days) before lifting heavy
def total_days_until_lifting_heavy := workout_waiting_days + 21

-- Given the total days until James can lift heavy is 39 days, prove x = 3
theorem pain_subsided_days : 
    total_days_until_lifting_heavy x = 39 → x = 3 := by
  sorry

end pain_subsided_days_l135_135559


namespace students_with_both_l135_135760

-- Define the problem conditions as given in a)
def total_students : ℕ := 50
def students_with_bike : ℕ := 28
def students_with_scooter : ℕ := 35

-- State the theorem
theorem students_with_both :
  ∃ (n : ℕ), n = 13 ∧ total_students = students_with_bike + students_with_scooter - n := by
  sorry

end students_with_both_l135_135760


namespace selling_prices_for_10_percent_profit_l135_135642

theorem selling_prices_for_10_percent_profit
    (cost1 cost2 cost3 : ℝ)
    (cost1_eq : cost1 = 200)
    (cost2_eq : cost2 = 300)
    (cost3_eq : cost3 = 500)
    (profit_percent : ℝ)
    (profit_percent_eq : profit_percent = 0.10):
    ∃ s1 s2 s3 : ℝ,
      s1 = cost1 + 33.33 ∧
      s2 = cost2 + 33.33 ∧
      s3 = cost3 + 33.33 ∧
      s1 + s2 + s3 = 1100 :=
by
  sorry

end selling_prices_for_10_percent_profit_l135_135642


namespace factor_expression_l135_135204

theorem factor_expression (x : ℝ) : 5 * x^2 * (x - 2) - 9 * (x - 2) = (x - 2) * (5 * x^2 - 9) :=
sorry

end factor_expression_l135_135204


namespace doughnuts_per_person_l135_135976

-- Define the number of dozens bought by Samuel
def samuel_dozens : ℕ := 2

-- Define the number of dozens bought by Cathy
def cathy_dozens : ℕ := 3

-- Define the number of doughnuts in one dozen
def dozen : ℕ := 12

-- Define the total number of people
def total_people : ℕ := 10

-- Prove that each person receives 6 doughnuts
theorem doughnuts_per_person :
  ((samuel_dozens * dozen) + (cathy_dozens * dozen)) / total_people = 6 :=
by
  sorry

end doughnuts_per_person_l135_135976


namespace trenton_commission_rate_l135_135484

noncomputable def commission_rate (fixed_earnings : ℕ) (goal : ℕ) (sales : ℕ) : ℚ :=
  ((goal - fixed_earnings : ℤ) / (sales : ℤ)) * 100

theorem trenton_commission_rate :
  commission_rate 190 500 7750 = 4 := 
  by
  sorry

end trenton_commission_rate_l135_135484


namespace box_volume_l135_135294

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_l135_135294


namespace negation_example_l135_135841

theorem negation_example :
  (¬ (∀ a : ℕ, a > 0 → 2^a ≥ a^2)) ↔ (∃ a : ℕ, a > 0 ∧ 2^a < a^2) :=
by sorry

end negation_example_l135_135841


namespace smallest_sum_of_squares_l135_135131

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l135_135131


namespace fat_content_whole_milk_l135_135828

open Real

theorem fat_content_whole_milk :
  ∃ (s w : ℝ), 0 < s ∧ 0 < w ∧
  3 / 100 = 0.75 * s / 100 ∧
  s / 100 = 0.8 * w / 100 ∧
  w = 5 :=
by
  sorry

end fat_content_whole_milk_l135_135828


namespace negation_exists_or_l135_135673

theorem negation_exists_or (x : ℝ) :
  ¬ (∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 :=
by sorry

end negation_exists_or_l135_135673


namespace john_savings_percentage_l135_135050

theorem john_savings_percentage :
  ∀ (savings discounted_price total_price original_price : ℝ),
  savings = 4.5 →
  total_price = 49.5 →
  total_price = discounted_price * 1.10 →
  original_price = discounted_price + savings →
  (savings / original_price) * 100 = 9 := by
  intros
  sorry

end john_savings_percentage_l135_135050


namespace total_pizza_order_cost_l135_135052

def pizza_cost_per_pizza := 10
def topping_cost_per_topping := 1
def tip_amount := 5
def number_of_pizzas := 3
def number_of_toppings := 4

theorem total_pizza_order_cost : 
  (pizza_cost_per_pizza * number_of_pizzas + topping_cost_per_topping * number_of_toppings + tip_amount) = 39 := by
  sorry

end total_pizza_order_cost_l135_135052


namespace find_functional_equation_solutions_l135_135883

theorem find_functional_equation_solutions :
  (∀ f : ℝ → ℝ, (∀ x y : ℝ, x > 0 → y > 0 → f x * f (y * f x) = f (x + y)) →
    (∃ a > 0, ∀ x > 0, f x = 1 / (1 + a * x) ∨ ∀ x > 0, f x = 1)) :=
by
  sorry

end find_functional_equation_solutions_l135_135883


namespace inequality_a_b_c_l135_135192

theorem inequality_a_b_c 
  (a b c : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_c : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3 / 2 :=
by 
  sorry

end inequality_a_b_c_l135_135192


namespace remainder_of_x_pow_105_div_x_sq_sub_4x_add_3_l135_135292

theorem remainder_of_x_pow_105_div_x_sq_sub_4x_add_3 :
  ∀ (x : ℤ), (x^105) % (x^2 - 4*x + 3) = (3^105 * (x-1) - (x-2)) / 2 :=
by sorry

end remainder_of_x_pow_105_div_x_sq_sub_4x_add_3_l135_135292


namespace cistern_emptying_time_l135_135162

theorem cistern_emptying_time (R L : ℝ) (hR : R = 1 / 6) (hL : L = 1 / 6 - 1 / 8) :
    1 / L = 24 := by
  -- The proof is omitted
  sorry

end cistern_emptying_time_l135_135162


namespace exception_to_roots_l135_135782

theorem exception_to_roots (x : ℝ) :
    ¬ (∃ x₀, (x₀ ∈ ({x | x = x} ∩ {x | x = x - 2}))) :=
by sorry

end exception_to_roots_l135_135782


namespace six_power_six_div_two_l135_135008

theorem six_power_six_div_two : 6 ^ (6 / 2) = 216 := by
  sorry

end six_power_six_div_two_l135_135008


namespace percentage_of_boys_l135_135831

theorem percentage_of_boys (total_students : ℕ) (ratio_boys_to_girls : ℕ) (ratio_girls_to_boys : ℕ) 
  (h_ratio : ratio_boys_to_girls = 3 ∧ ratio_girls_to_boys = 4 ∧ total_students = 42) : 
  (18 / 42) * 100 = 42.857 := 
by 
  sorry

end percentage_of_boys_l135_135831


namespace james_can_lift_546_pounds_l135_135072

def initial_lift_20m : ℝ := 300
def increase_10m : ℝ := 0.30
def strap_increase : ℝ := 0.20
def additional_weight_20m : ℝ := 50
def final_lift_10m_with_straps : ℝ := 546

theorem james_can_lift_546_pounds :
  let initial_lift_10m := initial_lift_20m * (1 + increase_10m)
  let updated_lift_20m := initial_lift_20m + additional_weight_20m
  let ratio := initial_lift_10m / initial_lift_20m
  let updated_lift_10m := updated_lift_20m * ratio
  let lift_with_straps := updated_lift_10m * (1 + strap_increase)
  lift_with_straps = final_lift_10m_with_straps :=
by
  sorry

end james_can_lift_546_pounds_l135_135072


namespace intersection_M_N_l135_135196

def M : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N :
  M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l135_135196


namespace cube_plus_eleven_mul_divisible_by_six_l135_135582

theorem cube_plus_eleven_mul_divisible_by_six (a : ℤ) : 6 ∣ (a^3 + 11 * a) := 
by sorry

end cube_plus_eleven_mul_divisible_by_six_l135_135582


namespace simplify_and_evaluate_l135_135612

theorem simplify_and_evaluate (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + 3 * b) * (2 * a - b) - 2 * (a - b)^2 = -23 := by
  sorry

end simplify_and_evaluate_l135_135612


namespace typing_page_percentage_l135_135106

/--
Given:
- Original sheet dimensions are 20 cm by 30 cm.
- Margins are 2 cm on each side (left and right), and 3 cm on the top and bottom.
Prove that the percentage of the page used by the typist is 64%.
-/
theorem typing_page_percentage (width height margin_lr margin_tb : ℝ)
  (h1 : width = 20) 
  (h2 : height = 30) 
  (h3 : margin_lr = 2) 
  (h4 : margin_tb = 3) : 
  (width - 2 * margin_lr) * (height - 2 * margin_tb) / (width * height) * 100 = 64 :=
by
  sorry

end typing_page_percentage_l135_135106


namespace triangle_area_is_correct_l135_135281

noncomputable def triangle_area (a b c B : ℝ) : ℝ := 
  0.5 * a * c * Real.sin B

theorem triangle_area_is_correct :
  let a := Real.sqrt 2
  let c := Real.sqrt 2
  let b := Real.sqrt 6
  let B := 2 * Real.pi / 3 -- 120 degrees in radians
  triangle_area a b c B = Real.sqrt 3 / 2 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end triangle_area_is_correct_l135_135281


namespace valid_five_letter_words_l135_135185

def num_valid_words : Nat :=
  let total_words := 3^5
  let invalid_3_consec := 5 * 2^3 * 1^2
  let invalid_4_consec := 2 * 2^4 * 1
  let invalid_5_consec := 2^5
  total_words - (invalid_3_consec + invalid_4_consec + invalid_5_consec)

theorem valid_five_letter_words : num_valid_words = 139 := by
  sorry

end valid_five_letter_words_l135_135185


namespace yellow_balls_count_l135_135334

theorem yellow_balls_count (purple blue total_needed : ℕ) 
  (h_purple : purple = 7) 
  (h_blue : blue = 5) 
  (h_total : total_needed = 19) : 
  ∃ (yellow : ℕ), yellow = 6 :=
by
  sorry

end yellow_balls_count_l135_135334


namespace max_distinct_rectangles_l135_135994

theorem max_distinct_rectangles : 
  ∃ (rectangles : Finset ℕ), (∀ n ∈ rectangles, n > 0) ∧ rectangles.sum id = 100 ∧ rectangles.card = 14 :=
by 
  sorry

end max_distinct_rectangles_l135_135994


namespace cost_of_72_tulips_is_115_20_l135_135524

/-
Conditions:
1. A package containing 18 tulips costs $36.
2. The price of a package is directly proportional to the number of tulips it contains.
3. There is a 20% discount applied for packages containing more than 50 tulips.
Question:
What is the cost of 72 tulips?

Correct answer:
$115.20
-/

def costOfTulips (numTulips : ℕ)  : ℚ :=
  if numTulips ≤ 50 then
    36 * numTulips / 18
  else
    (36 * numTulips / 18) * 0.8 -- apply 20% discount for more than 50 tulips

theorem cost_of_72_tulips_is_115_20 :
  costOfTulips 72 = 115.2 := 
sorry

end cost_of_72_tulips_is_115_20_l135_135524


namespace mike_investment_l135_135087

-- Define the given conditions and the conclusion we want to prove
theorem mike_investment (profit : ℝ) (mary_investment : ℝ) (mike_gets_more : ℝ) (total_profit_made : ℝ) :
  profit = 7500 → 
  mary_investment = 600 →
  mike_gets_more = 1000 →
  total_profit_made = 7500 →
  ∃ (mike_investment : ℝ), 
  ((1 / 3) * profit / 2 + (mary_investment / (mary_investment + mike_investment)) * ((2 / 3) * profit) = 
  (1 / 3) * profit / 2 + (mike_investment / (mary_investment + mike_investment)) * ((2 / 3) * profit) + mike_gets_more) →
  mike_investment = 400 :=
sorry

end mike_investment_l135_135087


namespace player_A_wins_l135_135387

theorem player_A_wins (n : ℕ) : ∃ m, (m > 2 * n^2) ∧ (∀ S : Finset (ℕ × ℕ), S.card = m → ∃ (r c : Finset ℕ), r.card = n ∧ c.card = n ∧ ∀ rc ∈ r.product c, rc ∈ S → false) :=
by sorry

end player_A_wins_l135_135387


namespace part1_part2_l135_135835

def A : Set ℝ := {x | (x + 4) * (x - 2) > 0}
def B : Set ℝ := {y | ∃ x : ℝ, y = (x - 1)^2 + 1}
def C (a : ℝ) : Set ℝ := {x | -4 ≤ x ∧ x ≤ a}

theorem part1 : A ∩ B = {x : ℝ | x > 2} := 
by sorry

theorem part2 (a : ℝ) (h : (C a \ A) ⊆ C a) : 2 ≤ a :=
by sorry

end part1_part2_l135_135835


namespace P_plus_Q_eq_14_l135_135040

variable (P Q : Nat)

-- Conditions:
axiom single_digit_P : P < 10
axiom single_digit_Q : Q < 10
axiom three_P_ends_7 : 3 * P % 10 = 7
axiom two_Q_ends_0 : 2 * Q % 10 = 0

theorem P_plus_Q_eq_14 : P + Q = 14 :=
by
  sorry

end P_plus_Q_eq_14_l135_135040


namespace max_non_equivalent_100_digit_numbers_l135_135215

noncomputable def maxPairwiseNonEquivalentNumbers : ℕ := 21^5

theorem max_non_equivalent_100_digit_numbers :
  (∀ (n : ℕ), 0 < n ∧ n < 100 → (∀ (digit : Fin n → Fin 2), 
  ∃ (max_num : ℕ), max_num = maxPairwiseNonEquivalentNumbers)) :=
by sorry

end max_non_equivalent_100_digit_numbers_l135_135215


namespace container_capacity_l135_135740

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 36 = 0.75 * C) : 
  C = 80 :=
sorry

end container_capacity_l135_135740


namespace kamala_overestimation_l135_135389

theorem kamala_overestimation : 
  let p := 150
  let q := 50
  let k := 2
  let d := 3
  let p_approx := 160
  let q_approx := 45
  let k_approx := 1
  let d_approx := 4
  let true_value := (p / q) - k + d
  let approx_value := (p_approx / q_approx) - k_approx + d_approx
  approx_value > true_value := 
  by 
  -- Skipping the detailed proof steps.
  sorry

end kamala_overestimation_l135_135389


namespace sum_of_digits_in_base_7_l135_135872

theorem sum_of_digits_in_base_7 (A B C : ℕ) (hA : A > 0) (hB : B > 0) (hC : C > 0) (hA7 : A < 7) (hB7 : B < 7) (hC7 : C < 7)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_eqn : A * 49 + B * 7 + C + (B * 7 + C) = A * 49 + C * 7 + A) : 
  (A + B + C) = 14 := by
  sorry

end sum_of_digits_in_base_7_l135_135872


namespace total_dolls_count_l135_135908

-- Define the conditions
def big_box_dolls : Nat := 7
def small_box_dolls : Nat := 4
def num_big_boxes : Nat := 5
def num_small_boxes : Nat := 9

-- State the theorem that needs to be proved
theorem total_dolls_count : 
  big_box_dolls * num_big_boxes + small_box_dolls * num_small_boxes = 71 := 
by
  sorry

end total_dolls_count_l135_135908


namespace quadratic_inequality_solution_l135_135042

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : 0 > a) 
(h2 : ∀ x : ℝ, (1 < x ∧ x < 2) ↔ (0 < ax^2 + bx + c)) : 
(∀ x : ℝ, (x < 1/2 ∨ 1 < x) ↔ (0 < 2*a*x^2 - 3*a*x + a)) :=
sorry

end quadratic_inequality_solution_l135_135042


namespace sum_divisible_by_10_l135_135970

theorem sum_divisible_by_10 :
    (111 ^ 111 + 112 ^ 112 + 113 ^ 113) % 10 = 0 :=
by
  sorry

end sum_divisible_by_10_l135_135970


namespace domain_f_1_minus_2x_is_0_to_half_l135_135936

-- Define the domain of f(x) as a set.
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- Define the domain condition for f(1 - 2*x).
def domain_f_1_minus_2x (x : ℝ) : Prop := 0 ≤ 1 - 2 * x ∧ 1 - 2 * x ≤ 1

-- State the theorem: If x is in the domain of f(1 - 2*x), then x is in [0, 1/2].
theorem domain_f_1_minus_2x_is_0_to_half :
  ∀ x : ℝ, domain_f_1_minus_2x x ↔ (0 ≤ x ∧ x ≤ 1 / 2) := by
  sorry

end domain_f_1_minus_2x_is_0_to_half_l135_135936


namespace opposite_of_neg_three_l135_135669

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_l135_135669


namespace sufficient_but_not_necessary_condition_l135_135702

theorem sufficient_but_not_necessary_condition (x : ℝ) (p : Prop) (q : Prop)
  (h₁ : p ↔ (x^2 - 1 > 0)) (h₂ : q ↔ (x < -2)) :
  (¬p → ¬q) ∧ ¬(¬q → ¬p) := 
by
  sorry

end sufficient_but_not_necessary_condition_l135_135702


namespace tricycle_total_spokes_l135_135540

noncomputable def front : ℕ := 20
noncomputable def middle : ℕ := 2 * front
noncomputable def back : ℝ := 20 * Real.sqrt 2
noncomputable def total_spokes : ℝ := front + middle + back

theorem tricycle_total_spokes : total_spokes = 88 :=
by
  sorry

end tricycle_total_spokes_l135_135540


namespace student_B_incorrect_l135_135652

-- Define the quadratic function and the non-zero condition on 'a'
def quadratic (a b x : ℝ) : ℝ := a * x^2 + b * x - 6

-- Conditions stated by the students
def student_A_condition (a b : ℝ) : Prop := -b / (2 * a) = 1
def student_B_condition (a b : ℝ) : Prop := quadratic a b 3 = -6
def student_C_condition (a b : ℝ) : Prop := (4 * a * (-6) - b^2) / (4 * a) = -8
def student_D_condition (a b : ℝ) : Prop := quadratic a b 3 = 0

-- The proof problem: Student B's conclusion is incorrect
theorem student_B_incorrect : 
  ∀ (a b : ℝ), 
  a ≠ 0 → 
  student_A_condition a b ∧ 
  student_C_condition a b ∧ 
  student_D_condition a b → 
  ¬ student_B_condition a b :=
by 
  -- problem converted to Lean problem format 
  -- based on the conditions provided
  sorry

end student_B_incorrect_l135_135652


namespace Simplify_division_l135_135907

theorem Simplify_division :
  (5 * 10^9) / (2 * 10^5 * 5) = 5000 := sorry

end Simplify_division_l135_135907


namespace find_m_n_value_l135_135541

theorem find_m_n_value (x m n : ℝ) 
  (h1 : x - 3 * m < 0) 
  (h2 : n - 2 * x < 0) 
  (h3 : -1 < x)
  (h4 : x < 3) 
  : (m + n) ^ 2023 = -1 :=
sorry

end find_m_n_value_l135_135541


namespace speed_of_stream_l135_135568

theorem speed_of_stream (D v : ℝ) (h1 : ∀ D, D / (54 - v) = 2 * (D / (54 + v))) : v = 18 := 
sorry

end speed_of_stream_l135_135568


namespace square_area_from_wire_bent_as_circle_l135_135194

theorem square_area_from_wire_bent_as_circle 
  (radius : ℝ) 
  (h_radius : radius = 56)
  (π_ineq : π > 3.1415) : 
  ∃ (A : ℝ), A = 784 * π^2 := 
by 
  sorry

end square_area_from_wire_bent_as_circle_l135_135194


namespace number_of_integers_l135_135761

theorem number_of_integers (n : ℕ) (h₁ : 300 < n^2) (h₂ : n^2 < 1200) : ∃ k, k = 17 :=
by
  sorry

end number_of_integers_l135_135761


namespace johnny_marble_choice_l135_135442

/-- Johnny has 9 different colored marbles and always chooses 1 specific red marble.
    Prove that the number of ways to choose four marbles from his bag is 56. -/
theorem johnny_marble_choice : (Nat.choose 8 3) = 56 := 
by
  sorry

end johnny_marble_choice_l135_135442


namespace train_passenger_count_l135_135974

theorem train_passenger_count (P : ℕ) (total_passengers : ℕ) (r : ℕ)
  (h1 : r = 60)
  (h2 : total_passengers = P + r + 3 * (P + r))
  (h3 : total_passengers = 640) :
  P = 100 :=
by
  sorry

end train_passenger_count_l135_135974


namespace pos_diff_of_solutions_abs_eq_20_l135_135550

theorem pos_diff_of_solutions_abs_eq_20 : ∀ (x1 x2 : ℝ), (|x1 + 5| = 20 ∧ |x2 + 5| = 20) → x1 - x2 = 40 :=
  by
    intros x1 x2 h
    sorry

end pos_diff_of_solutions_abs_eq_20_l135_135550


namespace greatest_divisor_same_remainder_l135_135219

theorem greatest_divisor_same_remainder (a b c : ℕ) (h₁ : a = 54) (h₂ : b = 87) (h₃ : c = 172) : 
  ∃ d, (d ∣ (b - a)) ∧ (d ∣ (c - b)) ∧ (d ∣ (c - a)) ∧ (∀ e, (e ∣ (b - a)) ∧ (e ∣ (c - b)) ∧ (e ∣ (c - a)) → e ≤ d) ∧ d = 1 := 
by 
  sorry

end greatest_divisor_same_remainder_l135_135219


namespace find_integer_n_l135_135007

theorem find_integer_n : ∃ (n : ℤ), 0 ≤ n ∧ n < 23 ∧ 54126 % 23 = n :=
by
  use 13
  sorry

end find_integer_n_l135_135007


namespace interest_rate_first_year_l135_135727

theorem interest_rate_first_year (R : ℚ)
  (principal : ℚ := 7000)
  (final_amount : ℚ := 7644)
  (time_period_first_year : ℚ := 1)
  (time_period_second_year : ℚ := 1)
  (rate_second_year : ℚ := 5) :
  principal + (principal * R * time_period_first_year / 100) + 
  ((principal + (principal * R * time_period_first_year / 100)) * rate_second_year * time_period_second_year / 100) = final_amount →
  R = 4 := 
by {
  sorry
}

end interest_rate_first_year_l135_135727


namespace geometric_series_ratio_l135_135104

theorem geometric_series_ratio (a_1 a_2 S q : ℝ) (hq : |q| < 1)
  (hS : S = a_1 / (1 - q))
  (ha2 : a_2 = a_1 * q) :
  S / (S - a_1) = a_1 / a_2 := 
sorry

end geometric_series_ratio_l135_135104


namespace breadth_of_rectangular_plot_l135_135608

variable (b l : ℕ)

def length_eq_thrice_breadth (b : ℕ) : ℕ := 3 * b

def area_of_rectangle_eq_2700 (b l : ℕ) : Prop := l * b = 2700

theorem breadth_of_rectangular_plot (h1 : l = 3 * b) (h2 : l * b = 2700) : b = 30 :=
by
  sorry

end breadth_of_rectangular_plot_l135_135608


namespace find_eagle_feathers_times_l135_135611

theorem find_eagle_feathers_times (x : ℕ) (hawk_feathers : ℕ) (total_feathers_before_give : ℕ) (total_feathers : ℕ) (left_after_selling : ℕ) :
  hawk_feathers = 6 →
  total_feathers_before_give = 6 + 6 * x →
  total_feathers = total_feathers_before_give - 10 →
  left_after_selling = total_feathers / 2 →
  left_after_selling = 49 →
  x = 17 :=
by
  intros h_hawk h_total_before_give h_total h_left h_after_selling
  sorry

end find_eagle_feathers_times_l135_135611


namespace factor_polynomial_l135_135088

theorem factor_polynomial (x y z : ℝ) : 
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (-(x * y + x * z + y * z)) :=
by
  sorry

end factor_polynomial_l135_135088


namespace general_term_formula_l135_135580

variable (a : ℕ → ℤ) -- A sequence of integers 
variable (d : ℤ) -- The common difference 

-- Conditions provided
axiom h1 : a 1 = 6
axiom h2 : a 3 + a 5 = 0
axiom h_arithmetic : ∀ n, a (n + 1) = a n + d -- Arithmetic progression condition

-- The general term formula we need to prove
theorem general_term_formula : ∀ n, a n = 8 - 2 * n := 
by 
  sorry -- Proof goes here


end general_term_formula_l135_135580


namespace initial_volume_of_mixture_l135_135795

theorem initial_volume_of_mixture (M W : ℕ) (h1 : 2 * M = 3 * W) (h2 : 4 * M = 3 * (W + 46)) : M + W = 115 := 
sorry

end initial_volume_of_mixture_l135_135795


namespace reflect_point_across_x_axis_l135_135942

theorem reflect_point_across_x_axis {x y : ℝ} (h : (x, y) = (2, 3)) : (x, -y) = (2, -3) :=
by
  sorry

end reflect_point_across_x_axis_l135_135942


namespace marble_group_l135_135518

theorem marble_group (x : ℕ) (h1 : 144 % x = 0) (h2 : 144 % (x + 2) = (144 / x) - 1) : x = 16 :=
sorry

end marble_group_l135_135518


namespace monotonic_intervals_and_extreme_values_of_f_f_g_inequality_sum_of_x1_x2_l135_135045

noncomputable def f (x : ℝ) := Real.exp x - (1 / 2) * x^2 - x - 1
noncomputable def f' (x : ℝ) := Real.exp x - x - 1
noncomputable def f'' (x : ℝ) := Real.exp x - 1
noncomputable def g (x : ℝ) := -f (-x)

-- Proof of (I)
theorem monotonic_intervals_and_extreme_values_of_f' :
  f' 0 = 0 ∧ (∀ x < 0, f'' x < 0 ∧ f' x > f' 0) ∧ (∀ x > 0, f'' x > 0 ∧ f' x > f' 0) := 
sorry

-- Proof of (II)
theorem f_g_inequality (x : ℝ) (hx : x > 0) : f x > g x :=
sorry

-- Proof of (III)
theorem sum_of_x1_x2 (x1 x2 : ℝ) (h : f x1 + f x2 = 0) (hne : x1 ≠ x2) : x1 + x2 < 0 := 
sorry

end monotonic_intervals_and_extreme_values_of_f_f_g_inequality_sum_of_x1_x2_l135_135045


namespace nate_current_age_l135_135236

open Real

variables (E N : ℝ)

/-- Ember is half as old as Nate, so E = 1/2 * N. -/
def ember_half_nate (h1 : E = 1/2 * N) : Prop := True

/-- The age difference of 7 years remains constant, so 21 - 14 = N - E. -/
def age_diff_constant (h2 : 7 = N - E) : Prop := True

/-- Prove that Nate is currently 14 years old given the conditions. -/
theorem nate_current_age (h1 : E = 1/2 * N) (h2 : 7 = N - E) : N = 14 :=
by sorry

end nate_current_age_l135_135236


namespace minimum_positive_period_of_f_l135_135896

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem minimum_positive_period_of_f : 
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧ 
  ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ Real.pi := 
sorry

end minimum_positive_period_of_f_l135_135896


namespace find_m_correct_l135_135571

structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  XY_length : dist X Y = 80
  XZ_length : dist X Z = 100
  YZ_length : dist Y Z = 120

noncomputable def find_m (t : Triangle) : ℝ :=
  let s := (80 + 100 + 120) / 2
  let A := 1 / 2 * 80 * 100
  let r1 := A / s
  let r2 := r1 / 2
  let r3 := r1 / 4
  let O2 := ((40 / 3), 50 + (40 / 3))
  let O3 := (40 + (20 / 3), (20 / 3))
  let O2O3 := dist O2 O3
  let m := (O2O3^2) / 10
  m

theorem find_m_correct (t : Triangle) : find_m t = 610 := sorry

end find_m_correct_l135_135571


namespace solution_set_l135_135549

noncomputable def domain := Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)
def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x, x ∈ domain → x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)
axiom f_odd : ∀ x, f x + f (-x) = 0
def f' : ℝ → ℝ := sorry
axiom derivative_condition : ∀ x, 0 < x ∧ x < Real.pi / 2 → f' x * Real.cos x + f x * Real.sin x < 0

theorem solution_set :
  {x | f x < Real.sqrt 2 * f (Real.pi / 4) * Real.cos x} = {x | Real.pi / 4 < x ∧ x < Real.pi / 2} :=
sorry

end solution_set_l135_135549


namespace average_of_eight_twelve_and_N_is_12_l135_135321

theorem average_of_eight_twelve_and_N_is_12 (N : ℝ) (hN : 11 < N ∧ N < 19) : (8 + 12 + N) / 3 = 12 :=
by
  -- Place the complete proof step here
  sorry

end average_of_eight_twelve_and_N_is_12_l135_135321


namespace fourth_metal_mass_approx_l135_135922

noncomputable def mass_of_fourth_metal 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 = 1.5 * x2)
  (h2 : x3 = 4 / 3 * x2)
  (h3 : x4 = 6 / 5 * x3)
  (h4 : x1 + x2 + x3 + x4 = 25) : ℝ :=
  x4

theorem fourth_metal_mass_approx 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 = 1.5 * x2)
  (h2 : x3 = 4 / 3 * x2)
  (h3 : x4 = 6 / 5 * x3)
  (h4 : x1 + x2 + x3 + x4 = 25) : 
  abs (mass_of_fourth_metal x1 x2 x3 x4 h1 h2 h3 h4 - 7.36) < 0.01 :=
by
  sorry

end fourth_metal_mass_approx_l135_135922


namespace correct_statement_c_l135_135458

theorem correct_statement_c (five_boys_two_girls : Nat := 7) (select_three : Nat := 3) :
  (∃ boys girls : Nat, boys + girls = five_boys_two_girls ∧ boys = 5 ∧ girls = 2) →
  (∃ selected_boys selected_girls : Nat, selected_boys + selected_girls = select_three ∧ selected_boys > 0) :=
by
  sorry

end correct_statement_c_l135_135458


namespace rect_area_correct_l135_135062

-- Defining the function to calculate the area of a rectangle given the coordinates of its vertices
noncomputable def rect_area (x1 y1 x2 y2 x3 y3 x4 y4 : ℤ) : ℤ :=
  let length := abs (x2 - x1)
  let width := abs (y1 - y3)
  length * width

-- The vertices of the rectangle
def x1 : ℤ := -8
def y1 : ℤ := 1
def x2 : ℤ := 1
def y2 : ℤ := 1
def x3 : ℤ := 1
def y3 : ℤ := -7
def x4 : ℤ := -8
def y4 : ℤ := -7

-- Proving that the area of the rectangle is 72 square units
theorem rect_area_correct : rect_area x1 y1 x2 y2 x3 y3 x4 y4 = 72 := by
  sorry

end rect_area_correct_l135_135062


namespace books_bought_l135_135519

noncomputable def totalCost : ℤ :=
  let numFilms := 9
  let costFilm := 5
  let numCDs := 6
  let costCD := 3
  let costBook := 4
  let totalSpent := 79
  totalSpent - (numFilms * costFilm + numCDs * costCD)

theorem books_bought : ∃ B : ℤ, B * 4 = totalCost := by
  sorry

end books_bought_l135_135519


namespace ratio_apples_peaches_l135_135758

theorem ratio_apples_peaches (total_fruits oranges peaches apples : ℕ)
  (h_total : total_fruits = 56)
  (h_oranges : oranges = total_fruits / 4)
  (h_peaches : peaches = oranges / 2)
  (h_apples : apples = 35) : apples / peaches = 5 := 
by
  sorry

end ratio_apples_peaches_l135_135758


namespace audrey_not_dreaming_fraction_l135_135710

theorem audrey_not_dreaming_fraction :
  let cycle1_not_dreaming := 3 / 4
  let cycle2_not_dreaming := 5 / 7
  let cycle3_not_dreaming := 2 / 3
  let cycle4_not_dreaming := 4 / 7
  cycle1_not_dreaming + cycle2_not_dreaming + cycle3_not_dreaming + cycle4_not_dreaming = 227 / 84 :=
by
  let cycle1_not_dreaming := 3 / 4
  let cycle2_not_dreaming := 5 / 7
  let cycle3_not_dreaming := 2 / 3
  let cycle4_not_dreaming := 4 / 7
  sorry

end audrey_not_dreaming_fraction_l135_135710


namespace solution_set_of_inequality_l135_135001

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_tangent : ∀ x₀ y₀, y₀ = f x₀ → (∀ x, f x = y₀ + (3*x₀^2 - 6*x₀)*(x - x₀)))
  (h_at_3 : f 3 = 0) :
  {x : ℝ | ((x - 1) / f x) ≥ 0} = {x : ℝ | x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 1} ∪ {x : ℝ | x > 3} :=
sorry

end solution_set_of_inequality_l135_135001


namespace no_real_solution_of_fraction_eq_l135_135947

theorem no_real_solution_of_fraction_eq (m : ℝ) :
  (∀ x : ℝ, (x - 1) / (x + 4) ≠ m / (x + 4)) → m = -5 :=
sorry

end no_real_solution_of_fraction_eq_l135_135947


namespace smallest_n_is_60_l135_135234

def smallest_n (n : ℕ) : Prop :=
  ∃ (n : ℕ), (n > 0) ∧ (24 ∣ n^2) ∧ (450 ∣ n^3) ∧ ∀ m : ℕ, 24 ∣ m^2 → 450 ∣ m^3 → m ≥ n

theorem smallest_n_is_60 : smallest_n 60 :=
  sorry

end smallest_n_is_60_l135_135234


namespace quadratic_roots_l135_135578

variable {a b c : ℝ}

theorem quadratic_roots (h₁ : a > 0) (h₂ : b > 0) (h₃ : c < 0) : 
  ∃ x₁ x₂ : ℝ, (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ 
  (x₁ ≠ x₂) ∧ (x₁ > 0) ∧ (x₂ < 0) ∧ (|x₂| > |x₁|) := 
sorry

end quadratic_roots_l135_135578


namespace region_area_l135_135472

noncomputable def area_of_region_outside_hexagon_inside_semicircles (s : ℝ) : ℝ :=
  let area_hexagon := (3 * Real.sqrt 3 / 2) * s^2
  let area_semicircle := (1/2) * Real.pi * (s/2)^2
  let total_area_semicircles := 6 * area_semicircle
  let total_area_circles := 6 * Real.pi * (s/2)^2
  total_area_circles - area_hexagon

theorem region_area (s := 2) : area_of_region_outside_hexagon_inside_semicircles s = (6 * Real.pi - 6 * Real.sqrt 3) :=
by
  sorry  -- Proof is skipped.

end region_area_l135_135472


namespace parallel_lines_l135_135933

theorem parallel_lines (m : ℝ) :
    (∀ x y : ℝ, x + (m+1) * y - 1 = 0 → mx + 2 * y - 1 = 0 → (m = 1 → False)) → m = -2 :=
by
  sorry

end parallel_lines_l135_135933


namespace calculate_expression_l135_135494

theorem calculate_expression : 
  (3.242 * (14 + 6) - 7.234 * 7) / 20 = 0.7101 :=
by
  sorry

end calculate_expression_l135_135494


namespace tan_sub_eq_one_third_l135_135863

theorem tan_sub_eq_one_third (α β : Real) (hα : Real.tan α = 3) (hβ : Real.tan β = 4/3) : 
  Real.tan (α - β) = 1/3 := by
  sorry

end tan_sub_eq_one_third_l135_135863


namespace parabola_vertex_sum_l135_135530

theorem parabola_vertex_sum 
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + c) = (a * (x + 3)^2 + 4))
  (h2 : (a * 49 + 4) = -2)
  : a + b + c = 100 / 49 :=
by
  sorry

end parabola_vertex_sum_l135_135530


namespace min_value_75_l135_135671

def min_value (x y z : ℝ) := x^2 + y^2 + z^2

theorem min_value_75 
  (x y z : ℝ) 
  (h1 : (x + 5) * (y - 5) = 0) 
  (h2 : (y + 5) * (z - 5) = 0) 
  (h3 : (z + 5) * (x - 5) = 0) :
  min_value x y z = 75 := 
sorry

end min_value_75_l135_135671


namespace max_principals_and_assistant_principals_l135_135019

theorem max_principals_and_assistant_principals : 
  ∀ (years term_principal term_assistant), (years = 10) ∧ (term_principal = 3) ∧ (term_assistant = 2) 
  → ∃ n, n = 9 :=
by
  sorry

end max_principals_and_assistant_principals_l135_135019


namespace sum_of_c_and_d_l135_135854

theorem sum_of_c_and_d (c d : ℝ) :
  (∀ x : ℝ, x ≠ 2 → x ≠ -3 → (x - 2) * (x + 3) = x^2 + c * x + d) →
  c + d = -5 :=
by
  intros h
  sorry

end sum_of_c_and_d_l135_135854


namespace inequality_proof_l135_135058

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b + c) / (2 * a) + (c + a) / (2 * b) + (a + b) / (2 * c) ≥ (2 * a) / (b + c) + (2 * b) / (c + a) + (2 * c) / (a + b) :=
by
  sorry

end inequality_proof_l135_135058


namespace no_such_a_exists_l135_135569

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {1, 5*a - 5, -1/2*a^2 + 3/2*a + 4, a^3 + a^2 + 3*a + 7}

theorem no_such_a_exists (a : ℝ) : ¬(A a ∩ B a = {2, 5}) :=
by
  sorry

end no_such_a_exists_l135_135569


namespace solution_to_axb_eq_0_l135_135648

theorem solution_to_axb_eq_0 (a b x : ℝ) (h₀ : a ≠ 0) (h₁ : (0, 4) ∈ {p : ℝ × ℝ | p.snd = a * p.fst + b}) (h₂ : (-3, 0) ∈ {p : ℝ × ℝ | p.snd = a * p.fst + b}) :
  x = -3 :=
by
  sorry

end solution_to_axb_eq_0_l135_135648


namespace sqrt_four_eq_two_or_neg_two_l135_135546

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 → (x = 2 ∨ x = -2) :=
sorry

end sqrt_four_eq_two_or_neg_two_l135_135546


namespace household_count_correct_l135_135924

def num_buildings : ℕ := 4
def floors_per_building : ℕ := 6
def households_first_floor : ℕ := 2
def households_other_floors : ℕ := 3
def total_households : ℕ := 68

theorem household_count_correct :
  num_buildings * (households_first_floor + (floors_per_building - 1) * households_other_floors) = total_households :=
by
  sorry

end household_count_correct_l135_135924


namespace total_profit_l135_135337

theorem total_profit (a_cap b_cap : ℝ) (a_profit : ℝ) (a_share b_share : ℝ) (P : ℝ) :
  a_cap = 15000 ∧ b_cap = 25000 ∧ a_share = 0.10 ∧ a_profit = 4200 →
  a_share * P + (a_cap / (a_cap + b_cap)) * (1 - a_share) * P = a_profit →
  P = 9600 :=
by
  intros h1 h2
  have h3 : a_share * P + (a_cap / (a_cap + b_cap)) * (1 - a_share) * P = a_profit := h2
  sorry

end total_profit_l135_135337


namespace find_x_l135_135041

def angle_sum_condition (x : ℝ) := 6 * x + 3 * x + x + x + 4 * x = 360

theorem find_x (x : ℝ) (h : angle_sum_condition x) : x = 24 := 
by {
  sorry
}

end find_x_l135_135041


namespace smallest_multiple_of_36_with_digit_product_divisible_by_9_l135_135018

theorem smallest_multiple_of_36_with_digit_product_divisible_by_9 :
  ∃ n : ℕ, n > 0 ∧ n % 36 = 0 ∧ (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ (d1 * d2 * d3) % 9 = 0) ∧ n = 936 := 
by
  sorry

end smallest_multiple_of_36_with_digit_product_divisible_by_9_l135_135018


namespace workers_together_time_l135_135483

-- Definition of the times taken by each worker to complete the job
def timeA : ℚ := 8
def timeB : ℚ := 10
def timeC : ℚ := 12

-- Definition of the rates based on the times
def rateA : ℚ := 1 / timeA
def rateB : ℚ := 1 / timeB
def rateC : ℚ := 1 / timeC

-- Definition of the total rate when working together
def total_rate : ℚ := rateA + rateB + rateC

-- Definition of the total time taken to complete the job when working together
def total_time : ℚ := 1 / total_rate

-- The final theorem we need to prove
theorem workers_together_time : total_time = 120 / 37 :=
by {
  -- structure of the proof will go here, but it is not required as per the instructions
  sorry
}

end workers_together_time_l135_135483


namespace total_cubes_in_stack_l135_135469

theorem total_cubes_in_stack :
  let bottom_layer := 4
  let middle_layer := 2
  let top_layer := 1
  bottom_layer + middle_layer + top_layer = 7 :=
by
  sorry

end total_cubes_in_stack_l135_135469


namespace p_sufficient_but_not_necessary_for_q_l135_135493

-- Definitions
variable {p q : Prop}

-- The condition: ¬p is a necessary but not sufficient condition for ¬q
def necessary_but_not_sufficient (p q : Prop) : Prop :=
  (∀ q, ¬q → ¬p) ∧ (∃ q, ¬q ∧ p)

-- The theorem stating the problem
theorem p_sufficient_but_not_necessary_for_q 
  (h : necessary_but_not_sufficient (¬p) (¬q)) : 
  (∀ p, p → q) ∧ (∃ p, p ∧ ¬q) :=
sorry

end p_sufficient_but_not_necessary_for_q_l135_135493


namespace sqrt_360000_eq_600_l135_135415

theorem sqrt_360000_eq_600 : Real.sqrt 360000 = 600 := 
sorry

end sqrt_360000_eq_600_l135_135415


namespace ab_range_l135_135998

theorem ab_range (a b : ℝ) (h : a * b = a + b + 3) : a * b ≤ 1 ∨ a * b ≥ 9 := by
  sorry

end ab_range_l135_135998


namespace ratio_of_areas_of_circles_l135_135283

theorem ratio_of_areas_of_circles (C_A C_B C_C : ℝ) (h1 : (60 / 360) * C_A = (40 / 360) * C_B) (h2 : (30 / 360) * C_B = (90 / 360) * C_C) : 
  (C_A / (2 * Real.pi))^2 / (C_C / (2 * Real.pi))^2 = 2 :=
by
  sorry

end ratio_of_areas_of_circles_l135_135283


namespace sam_bought_cards_l135_135558

-- Define the initial number of baseball cards Dan had.
def dan_initial_cards : ℕ := 97

-- Define the number of baseball cards Dan has after selling some to Sam.
def dan_remaining_cards : ℕ := 82

-- Prove that the number of baseball cards Sam bought is 15.
theorem sam_bought_cards : (dan_initial_cards - dan_remaining_cards) = 15 :=
by
  sorry

end sam_bought_cards_l135_135558


namespace fran_speed_calculation_l135_135475

noncomputable def fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) : ℝ :=
  joann_speed * joann_time / fran_time

theorem fran_speed_calculation : 
  fran_speed 15 3 2.5 = 18 := 
by
  -- Remember to write down the proof steps if needed, currently we use sorry as placeholder
  sorry

end fran_speed_calculation_l135_135475


namespace mirror_area_correct_l135_135570

-- Given conditions
def outer_length : ℕ := 80
def outer_width : ℕ := 60
def frame_width : ℕ := 10

-- Deriving the dimensions of the mirror
def mirror_length : ℕ := outer_length - 2 * frame_width
def mirror_width : ℕ := outer_width - 2 * frame_width

-- Statement: Prove that the area of the mirror is 2400 cm^2
theorem mirror_area_correct : mirror_length * mirror_width = 2400 := by
  -- Proof should go here
  sorry

end mirror_area_correct_l135_135570


namespace sum_a_b_max_power_l135_135986

theorem sum_a_b_max_power (a b : ℕ) (h_pos : 0 < a) (h_b_gt_1 : 1 < b) (h_lt_600 : a ^ b < 600) : a + b = 26 :=
sorry

end sum_a_b_max_power_l135_135986


namespace sum_non_solutions_eq_neg21_l135_135380

theorem sum_non_solutions_eq_neg21
  (A B C : ℝ)
  (h1 : ∀ x, ∃ k : ℝ, (x ≠ -C) ∧ (x ≠ -9) → (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9))
  (h2 : ∃ A B C, ∀ x, (x ≠ -C) ∧ (x ≠ -9) → (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9))
  (h3 : ∃! x, (x + C) * (x + 9) = 0)
   :
  -9 + -12 = -21 := by sorry

end sum_non_solutions_eq_neg21_l135_135380


namespace no_five_consecutive_divisible_by_2025_l135_135969

def seq (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2025 : 
  ¬ ∃ (a : ℕ), (∀ (i : ℕ), i < 5 → 2025 ∣ seq (a + i)) := 
sorry

end no_five_consecutive_divisible_by_2025_l135_135969


namespace compute_ab_l135_135067

theorem compute_ab (a b : ℝ)
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 867.75 := 
by
  sorry

end compute_ab_l135_135067


namespace minimum_components_needed_l135_135871

-- Define the parameters of the problem
def production_cost_per_component : ℝ := 80
def shipping_cost_per_component : ℝ := 7
def fixed_monthly_cost : ℝ := 16500
def selling_price_per_component : ℝ := 198.33

-- Define the total cost as a function of the number of components
def total_cost (x : ℝ) : ℝ :=
  fixed_monthly_cost + (production_cost_per_component + shipping_cost_per_component) * x

-- Define the revenue as a function of the number of components
def revenue (x : ℝ) : ℝ :=
  selling_price_per_component * x

-- Define the theorem to be proved
theorem minimum_components_needed (x : ℝ) : x = 149 ↔ total_cost x ≤ revenue x := sorry

end minimum_components_needed_l135_135871


namespace clotheslines_per_house_l135_135070

/-- There are a total of 11 children and 20 adults.
Each child has 4 items of clothing on the clotheslines.
Each adult has 3 items of clothing on the clotheslines.
Each clothesline can hold 2 items of clothing.
All of the clotheslines are full.
There are 26 houses on the street.
Show that the number of clotheslines per house is 2. -/
theorem clotheslines_per_house :
  (11 * 4 + 20 * 3) / 2 / 26 = 2 :=
by
  sorry

end clotheslines_per_house_l135_135070


namespace kopecks_payment_l135_135157

theorem kopecks_payment (n : ℕ) (h : n ≥ 8) : ∃ (a b : ℕ), n = 3 * a + 5 * b :=
sorry

end kopecks_payment_l135_135157


namespace least_number_of_marbles_divisible_by_2_3_4_5_6_7_l135_135210

theorem least_number_of_marbles_divisible_by_2_3_4_5_6_7 : 
  ∃ n : ℕ, (∀ k ∈ [2, 3, 4, 5, 6, 7], k ∣ n) ∧ n = 420 :=
  by sorry

end least_number_of_marbles_divisible_by_2_3_4_5_6_7_l135_135210


namespace sock_combination_count_l135_135716

noncomputable def numSockCombinations : Nat :=
  let striped := 4
  let solid := 4
  let checkered := 4
  let striped_and_solid := striped * solid
  let striped_and_checkered := striped * checkered
  striped_and_solid + striped_and_checkered

theorem sock_combination_count :
  numSockCombinations = 32 :=
by
  unfold numSockCombinations
  sorry

end sock_combination_count_l135_135716


namespace max_balls_in_cube_l135_135724

noncomputable def volume_of_cube : ℝ := (5 : ℝ)^3

noncomputable def volume_of_ball : ℝ := (4 / 3) * Real.pi * (1 : ℝ)^3

theorem max_balls_in_cube (c_length : ℝ) (b_radius : ℝ) (h1 : c_length = 5)
  (h2 : b_radius = 1) : 
  ⌊volume_of_cube / volume_of_ball⌋ = 29 := 
by
  sorry

end max_balls_in_cube_l135_135724


namespace symmetry_center_on_line_l135_135495

def symmetry_center_curve :=
  ∃ θ : ℝ, (∃ x y : ℝ, (x = -1 + Real.cos θ ∧ y = 2 + Real.sin θ))

-- The main theorem to prove
theorem symmetry_center_on_line : 
  (∃ cx cy : ℝ, (symmetry_center_curve ∧ (cy = -2 * cx))) :=
sorry

end symmetry_center_on_line_l135_135495


namespace lemons_left_l135_135874

/--
Prove that Cristine has 9 lemons left, given that she initially bought 12 lemons and gave away 1/4 of them.
-/
theorem lemons_left {initial_lemons : ℕ} (h1 : initial_lemons = 12) (fraction_given : ℚ) (h2 : fraction_given = 1 / 4) : initial_lemons - initial_lemons * fraction_given = 9 := by
  sorry

end lemons_left_l135_135874


namespace circle_symmetric_point_l135_135788

theorem circle_symmetric_point (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a * x - 2 * y + b = 0 → x = 2 ∧ y = 1) ∧
  (∀ x y : ℝ, (x, y) ∈ { (px, py) | px = 2 ∧ py = 1 ∨ x + y - 1 = 0 } → x^2 + y^2 + a * x - 2 * y + b = 0) →
  a = 0 ∧ b = -3 := 
by {
  sorry
}

end circle_symmetric_point_l135_135788


namespace min_value_l135_135425

theorem min_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (m n : ℝ) (h3 : m > 0) (h4 : n > 0) 
(h5 : m + 4 * n = 1) : 
  1 / m + 4 / n ≥ 25 :=
by
  sorry

end min_value_l135_135425


namespace citric_acid_molecular_weight_l135_135053

noncomputable def molecularWeightOfCitricAcid : ℝ :=
  let weight_C := 12.01
  let weight_H := 1.008
  let weight_O := 16.00
  let num_C := 6
  let num_H := 8
  let num_O := 7
  (num_C * weight_C) + (num_H * weight_H) + (num_O * weight_O)

theorem citric_acid_molecular_weight :
  molecularWeightOfCitricAcid = 192.124 :=
by
  -- the step-by-step proof will go here
  sorry

end citric_acid_molecular_weight_l135_135053


namespace complete_residue_system_l135_135731

theorem complete_residue_system {m n : ℕ} {a : ℕ → ℕ} {b : ℕ → ℕ}
  (h₁ : ∀ i j, 1 ≤ i → i ≤ m → 1 ≤ j → j ≤ n → (a i) * (b j) % (m * n) ≠ (a i) * (b j)) :
  (∀ i₁ i₂, 1 ≤ i₁ → i₁ ≤ m → 1 ≤ i₂ → i₂ ≤ m → i₁ ≠ i₂ → (a i₁ % m ≠ a i₂ % m)) ∧ 
  (∀ j₁ j₂, 1 ≤ j₁ → j₁ ≤ n → 1 ≤ j₂ → j₂ ≤ n → j₁ ≠ j₂ → (b j₁ % n ≠ b j₂ % n)) := sorry

end complete_residue_system_l135_135731


namespace father_age_is_30_l135_135435

theorem father_age_is_30 {M F : ℝ} 
  (h1 : M = (2 / 5) * F) 
  (h2 : M + 6 = (1 / 2) * (F + 6)) :
  F = 30 :=
sorry

end father_age_is_30_l135_135435


namespace probability_of_triangle_with_nonagon_side_l135_135937

-- Definitions based on the given conditions
def num_vertices : ℕ := 9

def total_triangles : ℕ := Nat.choose num_vertices 3

def favorable_outcomes : ℕ :=
  let one_side_is_side_of_nonagon := num_vertices * 5
  let two_sides_are_sides_of_nonagon := num_vertices
  one_side_is_side_of_nonagon + two_sides_are_sides_of_nonagon

def probability : ℚ := favorable_outcomes / total_triangles

-- Lean 4 statement to prove the equivalence of the probability calculation
theorem probability_of_triangle_with_nonagon_side :
  probability = 9 / 14 :=
by
  sorry

end probability_of_triangle_with_nonagon_side_l135_135937


namespace remainder_sum_mod9_l135_135909

def a1 := 8243
def a2 := 8244
def a3 := 8245
def a4 := 8246

theorem remainder_sum_mod9 : ((a1 + a2 + a3 + a4) % 9) = 7 :=
by
  sorry

end remainder_sum_mod9_l135_135909


namespace quadratic_roots_distinct_l135_135945

variable (a b c : ℤ)

theorem quadratic_roots_distinct (h_eq : 3 * a^2 - 3 * a - 4 = 0) : ∃ (x y : ℝ), x ≠ y ∧ (3 * x^2 - 3 * x - 4 = 0) ∧ (3 * y^2 - 3 * y - 4 = 0) := 
  sorry

end quadratic_roots_distinct_l135_135945


namespace max_piles_660_stones_l135_135818

-- Define the conditions in Lean
def initial_stones := 660

def valid_pile_sizes (piles : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → a ≤ b → b < 2 * a

-- Define the goal statement in Lean
theorem max_piles_660_stones :
  ∃ (piles : List ℕ), (piles.length = 30) ∧ (piles.sum = initial_stones) ∧ valid_pile_sizes piles :=
sorry

end max_piles_660_stones_l135_135818


namespace no_pos_reals_floor_prime_l135_135082

open Real
open Nat

theorem no_pos_reals_floor_prime : 
  ∀ (a b : ℝ), (0 < a) → (0 < b) → ∃ n : ℕ, ¬ Prime (⌊a * n + b⌋) :=
by
  intro a b a_pos b_pos
  sorry

end no_pos_reals_floor_prime_l135_135082


namespace cube_sum_l135_135249

theorem cube_sum (a b : ℝ) (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end cube_sum_l135_135249


namespace prob_at_least_one_2_in_two_8_sided_dice_l135_135803

/-- Probability of getting at least one 2 when rolling two 8-sided dice -/
theorem prob_at_least_one_2_in_two_8_sided_dice : 
  let total_outcomes := 64
  let favorable_outcomes := 15
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 15 / 64 :=
by
  sorry

end prob_at_least_one_2_in_two_8_sided_dice_l135_135803


namespace correct_multiplication_result_l135_135591

theorem correct_multiplication_result :
  ∃ x : ℕ, (x * 9 = 153) ∧ (x * 6 = 102) :=
by
  sorry

end correct_multiplication_result_l135_135591


namespace find_natural_numbers_l135_135595

theorem find_natural_numbers (x : ℕ) : (x % 7 = 3) ∧ (x % 9 = 4) ∧ (x < 100) ↔ (x = 31) ∨ (x = 94) := 
by sorry

end find_natural_numbers_l135_135595


namespace simplify_fractional_equation_l135_135485

theorem simplify_fractional_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 2) : (x / (x - 2) - 2 = 3 / (2 - x)) → (x - 2 * (x - 2) = -3) :=
by
  sorry

end simplify_fractional_equation_l135_135485


namespace jacoby_lottery_expense_l135_135384

-- Definitions based on the conditions:
def jacoby_trip_fund_needed : ℕ := 5000
def jacoby_hourly_wage : ℕ := 20
def jacoby_work_hours : ℕ := 10
def cookies_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def num_sisters : ℕ := 2
def money_still_needed : ℕ := 3214

-- The statement to prove:
theorem jacoby_lottery_expense : 
  (jacoby_hourly_wage * jacoby_work_hours) + (cookies_price * cookies_sold) +
  lottery_winnings + (sister_gift * num_sisters) 
  - (jacoby_trip_fund_needed - money_still_needed) = 10 :=
by {
  sorry
}

end jacoby_lottery_expense_l135_135384


namespace negation_prob1_negation_prob2_negation_prob3_l135_135073

-- Definitions and Conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def defines_const_func (f : ℝ → ℝ) (y : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, f x1 = f x2

-- Problem 1
theorem negation_prob1 : 
  (∃ n : ℕ, ∀ p : ℕ, is_prime p → p ≤ n) ↔ 
  ¬(∀ n : ℕ, ∃ p : ℕ, is_prime p ∧ n ≤ p) :=
sorry

-- Problem 2
theorem negation_prob2 : 
  (∃ n : ℤ, ∀ p : ℤ, n + p ≠ 0) ↔ 
  ¬(∀ n : ℤ, ∃! p : ℤ, n + p = 0) :=
sorry

-- Problem 3
theorem negation_prob3 : 
  (∀ y : ℝ, ¬defines_const_func (λ x => x * y) y) ↔ 
  ¬(∃ y : ℝ, defines_const_func (λ x => x * y) y) :=
sorry

end negation_prob1_negation_prob2_negation_prob3_l135_135073


namespace luke_money_last_weeks_l135_135746

theorem luke_money_last_weeks (earnings_mowing : ℕ) (earnings_weed_eating : ℕ) (weekly_spending : ℕ) 
  (h1 : earnings_mowing = 9) (h2 : earnings_weed_eating = 18) (h3 : weekly_spending = 3) :
  (earnings_mowing + earnings_weed_eating) / weekly_spending = 9 :=
by sorry

end luke_money_last_weeks_l135_135746


namespace six_digit_number_divisible_by_eleven_l135_135796

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def reverse_digits (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a

def concatenate_reverse (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a

theorem six_digit_number_divisible_by_eleven (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9)
  (h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : 0 ≤ c) (h₆ : c ≤ 9) :
  11 ∣ concatenate_reverse a b c :=
by
  sorry

end six_digit_number_divisible_by_eleven_l135_135796


namespace line_slope_l135_135876

theorem line_slope : 
  (∀ (x y : ℝ), (x / 4 - y / 3 = -2) → (y = -3/4 * x - 6)) ∧ (∀ (x : ℝ), ∃ y : ℝ, (x / 4 - y / 3 = -2)) :=
by
  sorry

end line_slope_l135_135876


namespace nat_square_iff_divisibility_l135_135201

theorem nat_square_iff_divisibility (A : ℕ) :
  (∃ k : ℕ, A = k^2) ↔ (∀ n : ℕ, ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ n ∣ ((A + i) * (A + i) - A)) :=
sorry

end nat_square_iff_divisibility_l135_135201


namespace initial_population_l135_135134

/-- The population of a town decreases annually at the rate of 20% p.a.
    Given that the population of the town after 2 years is 19200,
    prove that the initial population of the town was 30,000. -/
theorem initial_population (P : ℝ) (h : 0.64 * P = 19200) : P = 30000 :=
sorry

end initial_population_l135_135134


namespace high_speed_train_equation_l135_135033

theorem high_speed_train_equation (x : ℝ) (h1 : x > 0) : 
  700 / x - 700 / (2.8 * x) = 3.6 :=
by
  sorry

end high_speed_train_equation_l135_135033


namespace cone_height_ratio_l135_135654

theorem cone_height_ratio (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) 
  (rolls_19_times : 19 * 2 * Real.pi * r = 2 * Real.pi * Real.sqrt (r^2 + h^2)) :
  h / r = 6 * Real.sqrt 10 :=
by
  -- problem setup and mathematical manipulations
  sorry

end cone_height_ratio_l135_135654


namespace seunghyo_daily_dosage_l135_135563

theorem seunghyo_daily_dosage (total_medicine : ℝ) (daily_fraction : ℝ) (correct_dosage : ℝ) :
  total_medicine = 426 → daily_fraction = 0.06 → correct_dosage = 25.56 →
  total_medicine * daily_fraction = correct_dosage :=
by
  intros ht hf hc
  simp [ht, hf, hc]
  sorry

end seunghyo_daily_dosage_l135_135563


namespace initial_value_subtract_perfect_square_l135_135460

theorem initial_value_subtract_perfect_square :
  ∃ n : ℕ, n^2 = 308 - 139 :=
by
  sorry

end initial_value_subtract_perfect_square_l135_135460


namespace worst_ranking_l135_135695

theorem worst_ranking (teams : Fin 25 → Nat) (A : Fin 25)
  (round_robin : ∀ i j, i ≠ j → teams i + teams j ≤ 4)
  (most_goals : ∀ i, i ≠ A → teams A > teams i)
  (fewest_goals : ∀ i, i ≠ A → teams i > teams A) :
  ∃ ranking : Fin 25 → Fin 25, ranking A = 24 :=
by
  sorry

end worst_ranking_l135_135695


namespace relationship_of_a_b_c_l135_135352

noncomputable def a : ℝ := Real.log 3 / Real.log 2  -- a = log2(1/3)
noncomputable def b : ℝ := Real.exp (1 / 3)  -- b = e^(1/3)
noncomputable def c : ℝ := 1 / 3  -- c = e^ln(1/3) = 1/3

theorem relationship_of_a_b_c : b > c ∧ c > a :=
by
  -- Proof would go here
  sorry

end relationship_of_a_b_c_l135_135352


namespace question1_question2_l135_135124

/-
In ΔABC, the sides opposite to angles A, B, and C are respectively a, b, and c.
It is given that b + c = 2 * a * cos B.

(1) Prove that A = 2B;
(2) If the area of ΔABC is S = a^2 / 4, find the magnitude of angle A.
-/

variables {A B C a b c : ℝ}
variables {S : ℝ}

-- Condition given in the problem
axiom h1 : b + c = 2 * a * Real.cos B
axiom h2 : 1 / 2 * b * c * Real.sin A = a^2 / 4

-- Question 1: Prove that A = 2 * B
theorem question1 (h1 : b + c = 2 * a * Real.cos B) : A = 2 * B := sorry

-- Question 2: Find the magnitude of angle A
theorem question2 (h2 : 1 / 2 * b * c * Real.sin A = a^2 / 4) : A = 90 ∨ A = 45 := sorry

end question1_question2_l135_135124


namespace prime_iff_sum_four_distinct_products_l135_135962

variable (n : ℕ) (a b c d : ℕ)

theorem prime_iff_sum_four_distinct_products (h : n ≥ 5) :
  (Prime n ↔ ∀ (a b c d : ℕ), n = a + b + c + d → a > 0 → b > 0 → c > 0 → d > 0 → ab ≠ cd) :=
sorry

end prime_iff_sum_four_distinct_products_l135_135962


namespace find_f_3_l135_135102

def f (x : ℝ) : ℝ := x + 3  -- define the function as per the condition

theorem find_f_3 : f (3) = 7 := by
  sorry

end find_f_3_l135_135102


namespace find_higher_selling_price_l135_135393

def cost_price := 200
def selling_price_low := 340
def gain_low := selling_price_low - cost_price
def gain_high := gain_low + (5 / 100) * gain_low
def higher_selling_price := cost_price + gain_high

theorem find_higher_selling_price : higher_selling_price = 347 := 
by 
  sorry

end find_higher_selling_price_l135_135393


namespace solve_grape_rate_l135_135137

noncomputable def grape_rate (G : ℝ) : Prop :=
  11 * G + 7 * 50 = 1428

theorem solve_grape_rate : ∃ G : ℝ, grape_rate G ∧ G = 98 :=
by
  exists 98
  sorry

end solve_grape_rate_l135_135137


namespace trevor_brother_age_l135_135707

theorem trevor_brother_age :
  ∃ B : ℕ, Trevor_current_age = 11 ∧
           Trevor_future_age = 24 ∧
           Brother_future_age = 3 * Trevor_current_age ∧
           B = Brother_future_age - (Trevor_future_age - Trevor_current_age) :=
sorry

end trevor_brother_age_l135_135707


namespace proof1_proof2_l135_135893

variable (a : ℝ) (m n : ℝ)
axiom am_eq_two : a^m = 2
axiom an_eq_three : a^n = 3

theorem proof1 : a^(4 * m + 3 * n) = 432 := by
  sorry

theorem proof2 : a^(5 * m - 2 * n) = 32 / 9 := by
  sorry

end proof1_proof2_l135_135893


namespace transformation_of_95_squared_l135_135564

theorem transformation_of_95_squared :
  (9.5 : ℝ) ^ 2 = (10 : ℝ) ^ 2 - 2 * (10 : ℝ) * (0.5 : ℝ) + (0.5 : ℝ) ^ 2 :=
by
  sorry

end transformation_of_95_squared_l135_135564


namespace eq_infinite_solutions_function_satisfies_identity_l135_135506

-- First Part: Proving the equation has infinitely many positive integer solutions
theorem eq_infinite_solutions : ∃ (x y z t : ℕ), ∀ n : ℕ, x^2 + 2 * y^2 = z^2 + 2 * t^2 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 := 
sorry

-- Second Part: Finding and proving the function f
def f (n : ℕ) : ℕ := n

theorem function_satisfies_identity (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (f n^2 + 2 * f m^2) = n^2 + 2 * m^2) : ∀ k : ℕ, f k = k :=
sorry

end eq_infinite_solutions_function_satisfies_identity_l135_135506


namespace grace_pennies_l135_135396

theorem grace_pennies :
  let dime_value := 10
  let coin_value := 5
  let dimes := 10
  let coins := 10
  dimes * dime_value + coins * coin_value = 150 :=
by
  let dime_value := 10
  let coin_value := 5
  let dimes := 10
  let coins := 10
  sorry

end grace_pennies_l135_135396


namespace least_three_digit_divisible_3_4_7_is_168_l135_135256

-- Define the function that checks the conditions
def is_least_three_digit_divisible_by_3_4_7 (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ x % 3 = 0 ∧ x % 4 = 0 ∧ x % 7 = 0

-- Define the target value
def least_three_digit_number_divisible_by_3_4_7 : ℕ := 168

-- The theorem we want to prove
theorem least_three_digit_divisible_3_4_7_is_168 :
  ∃ x : ℕ, is_least_three_digit_divisible_by_3_4_7 x ∧ x = least_three_digit_number_divisible_by_3_4_7 := by
  sorry

end least_three_digit_divisible_3_4_7_is_168_l135_135256


namespace total_cookies_prepared_l135_135319

-- Definition of conditions
def cookies_per_guest : ℕ := 19
def number_of_guests : ℕ := 2

-- Theorem statement
theorem total_cookies_prepared : (cookies_per_guest * number_of_guests) = 38 :=
by
  sorry

end total_cookies_prepared_l135_135319


namespace tile_equations_correct_l135_135203

theorem tile_equations_correct (x y : ℕ) (h1 : 24 * x + 12 * y = 2220) (h2 : y = 2 * x - 15) : 
    (24 * x + 12 * y = 2220) ∧ (y = 2 * x - 15) :=
by
  exact ⟨h1, h2⟩

end tile_equations_correct_l135_135203


namespace fraction_problem_l135_135416

-- Definitions translated from conditions
variables (m n p q : ℚ)
axiom h1 : m / n = 20
axiom h2 : p / n = 5
axiom h3 : p / q = 1 / 15

-- Statement to prove
theorem fraction_problem : m / q = 4 / 15 :=
by
  sorry

end fraction_problem_l135_135416


namespace total_cups_l135_135139

theorem total_cups (t1 t2 : ℕ) (h1 : t2 = 240) (h2 : t2 = t1 - 20) : t1 + t2 = 500 := by
  sorry

end total_cups_l135_135139


namespace seq_max_value_l135_135025

theorem seq_max_value {a_n : ℕ → ℝ} (h : ∀ n, a_n n = (↑n + 2) * (3 / 4) ^ n) : 
  ∃ n, a_n n = max (a_n 1) (a_n 2) → (n = 1 ∨ n = 2) :=
by 
  sorry

end seq_max_value_l135_135025


namespace nuts_per_box_l135_135961

theorem nuts_per_box (N : ℕ)  
  (h1 : ∀ (boxes bolts_per_box : ℕ), boxes = 7 ∧ bolts_per_box = 11 → boxes * bolts_per_box = 77)
  (h2 : ∀ (boxes: ℕ), boxes = 3 → boxes * N = 3 * N)
  (h3 : ∀ (used_bolts purchased_bolts remaining_bolts : ℕ), purchased_bolts = 77 ∧ remaining_bolts = 3 → used_bolts = purchased_bolts - remaining_bolts)
  (h4 : ∀ (used_nuts purchased_nuts remaining_nuts : ℕ), purchased_nuts = 3 * N ∧ remaining_nuts = 6 → used_nuts = purchased_nuts - remaining_nuts)
  (h5 : ∀ (used_bolts used_nuts total_used : ℕ), used_bolts = 74 ∧ used_nuts = 3 * N - 6 → total_used = used_bolts + used_nuts)
  (h6 : total_used_bolts_and_nuts = 113) :
  N = 15 :=
by
  sorry

end nuts_per_box_l135_135961


namespace sum_of_interior_angles_divisible_by_360_l135_135819

theorem sum_of_interior_angles_divisible_by_360
  (n : ℕ)
  (h : n > 0) :
  ∃ k : ℤ, ((2 * n - 2) * 180) = 360 * k :=
by
  sorry

end sum_of_interior_angles_divisible_by_360_l135_135819


namespace hypotenuse_of_right_triangle_l135_135277

theorem hypotenuse_of_right_triangle (a b : ℕ) (h : ℕ)
  (h1 : a = 15) (h2 : b = 36) (right_triangle : a^2 + b^2 = h^2) : h = 39 :=
by
  sorry

end hypotenuse_of_right_triangle_l135_135277


namespace sufficient_but_not_necessary_l135_135804

theorem sufficient_but_not_necessary (a : ℝ) : (a = 1 → a^2 = 1) ∧ ¬(a^2 = 1 → a = 1) :=
by
  sorry

end sufficient_but_not_necessary_l135_135804


namespace average_score_of_male_students_l135_135675

theorem average_score_of_male_students
  (female_students : ℕ) (male_students : ℕ) (female_avg_score : ℕ) (class_avg_score : ℕ)
  (h_female_students : female_students = 20)
  (h_male_students : male_students = 30)
  (h_female_avg_score : female_avg_score = 75)
  (h_class_avg_score : class_avg_score = 72) :
  (30 * (((class_avg_score * (female_students + male_students)) - (female_avg_score * female_students)) / male_students) = 70) :=
by
  -- Sorry for the proof
  sorry

end average_score_of_male_students_l135_135675


namespace first_term_of_geometric_series_l135_135806

theorem first_term_of_geometric_series (r a S : ℚ) (h_common_ratio : r = -1/5) (h_sum : S = 16) :
  a = 96 / 5 :=
by
  sorry

end first_term_of_geometric_series_l135_135806


namespace train_speed_correct_l135_135553

def train_length : ℝ := 110
def bridge_length : ℝ := 142
def crossing_time : ℝ := 12.598992080633549
def expected_speed : ℝ := 20.002

theorem train_speed_correct :
  (train_length + bridge_length) / crossing_time = expected_speed :=
by
  sorry

end train_speed_correct_l135_135553


namespace bananas_per_box_l135_135771

def total_bananas : ℕ := 40
def number_of_boxes : ℕ := 10

theorem bananas_per_box : total_bananas / number_of_boxes = 4 := by
  sorry

end bananas_per_box_l135_135771


namespace machine_C_time_l135_135544

theorem machine_C_time (T_c : ℝ) :
  (1 / 4 + 1 / 2 + 1 / T_c = 11 / 12) → T_c = 6 :=
by
  sorry

end machine_C_time_l135_135544


namespace lines_region_division_l135_135814

theorem lines_region_division (f : ℕ → ℕ) (k : ℕ) (h : k ≥ 2) : 
  (∀ m, f m = m * (m + 1) / 2 + 1) → f (k + 1) = f k + (k + 1) :=
by
  intro h_f
  have h_base : f 1 = 2 := by sorry
  have h_ih : ∀ n, n ≥ 2 → f (n + 1) = f n + (n + 1) := by sorry
  exact h_ih k h

end lines_region_division_l135_135814


namespace television_combinations_l135_135676

def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem television_combinations :
  ∃ (combinations : ℕ), 
  ∀ (A B total : ℕ), A = 4 → B = 5 → total = 3 →
  combinations = (combination 4 2 * combination 5 1 + combination 4 1 * combination 5 2) →
  combinations = 70 :=
sorry

end television_combinations_l135_135676


namespace train_speed_l135_135451

noncomputable def original_speed_of_train (v d : ℝ) : Prop :=
  (120 ≤ v / (5/7)) ∧
  (2 * d) / (5 * v) = 65 / 60 ∧
  (2 * (d - 42)) / (5 * v) = 45 / 60

theorem train_speed (v d : ℝ) (h : original_speed_of_train v d) : v = 50.4 :=
by sorry

end train_speed_l135_135451


namespace log2_of_fraction_l135_135889

theorem log2_of_fraction : Real.logb 2 0.03125 = -5 := by
  sorry

end log2_of_fraction_l135_135889


namespace probability_of_event_A_l135_135592

noncomputable def probability_both_pieces_no_less_than_three_meters (L : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  if h : L = a + b 
  then (if a ≥ 3 ∧ b ≥ 3 then (L - 2 * 3) / L else 0)
  else 0

theorem probability_of_event_A : 
  probability_both_pieces_no_less_than_three_meters 11 6 5 = 5 / 11 :=
by
  -- Additional context to ensure proper definition of the problem
  sorry

end probability_of_event_A_l135_135592


namespace range_of_m_l135_135010

theorem range_of_m (m : ℝ) : 
  (∀ x, x^2 + 2 * x - m > 0 ↔ (x = 1 → x^2 + 2 * x - m ≤ 0) ∧ (x = 2 → x^2 + 2 * x - m > 0)) ↔ (3 ≤ m ∧ m < 8) := 
sorry

end range_of_m_l135_135010


namespace water_speed_l135_135678

theorem water_speed (v : ℝ) 
  (still_water_speed : ℝ := 4)
  (distance : ℝ := 10)
  (time : ℝ := 5)
  (effective_speed : ℝ := distance / time) 
  (h : still_water_speed - v = effective_speed) :
  v = 2 :=
by
  sorry

end water_speed_l135_135678


namespace apples_per_pie_l135_135657

theorem apples_per_pie
  (total_apples : ℕ) (apples_handed_out : ℕ) (remaining_apples : ℕ) (number_of_pies : ℕ)
  (h1 : total_apples = 96)
  (h2 : apples_handed_out = 42)
  (h3 : remaining_apples = total_apples - apples_handed_out)
  (h4 : remaining_apples = 54)
  (h5 : number_of_pies = 9) :
  remaining_apples / number_of_pies = 6 := by
  sorry

end apples_per_pie_l135_135657


namespace value_of_x_l135_135526

theorem value_of_x (x : ℝ) (h1 : |x| - 1 = 0) (h2 : x - 1 ≠ 0) : x = -1 := 
sorry

end value_of_x_l135_135526


namespace find_g_neg_one_l135_135543

theorem find_g_neg_one (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 1 / 2 → g x + g ((x + 2) / (2 - 4 * x)) = 3 * x) : 
  g (-1) = - 3 / 2 := 
sorry

end find_g_neg_one_l135_135543


namespace sara_peaches_l135_135497

theorem sara_peaches (initial_peaches : ℕ) (picked_peaches : ℕ) (total_peaches : ℕ) 
  (h1 : initial_peaches = 24) (h2 : picked_peaches = 37) : 
  total_peaches = 61 :=
by
  sorry

end sara_peaches_l135_135497


namespace exists_infinitely_many_n_odd_floor_l135_135968

def even (n : ℤ) := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem exists_infinitely_many_n_odd_floor (α : ℝ) : 
  ∃ᶠ n in at_top, odd ⌊n^2 * α⌋ := sorry

end exists_infinitely_many_n_odd_floor_l135_135968


namespace overall_gain_is_10_percent_l135_135925

noncomputable def total_cost_price : ℝ := 700 + 500 + 300
noncomputable def total_gain : ℝ := 70 + 50 + 30
noncomputable def overall_gain_percentage : ℝ := (total_gain / total_cost_price) * 100

theorem overall_gain_is_10_percent :
  overall_gain_percentage = 10 :=
by
  sorry

end overall_gain_is_10_percent_l135_135925


namespace point_in_third_quadrant_l135_135090

open Complex

-- Define that i is the imaginary unit
def imaginary_unit : ℂ := Complex.I

-- Define the condition i * z = 1 - 2i
def condition (z : ℂ) : Prop := imaginary_unit * z = (1 : ℂ) - 2 * imaginary_unit

-- Prove that the point corresponding to the complex number z is located in the third quadrant
theorem point_in_third_quadrant (z : ℂ) (h : condition z) : z.re < 0 ∧ z.im < 0 := sorry

end point_in_third_quadrant_l135_135090


namespace part1_part1_monotonicity_intervals_part2_l135_135511

noncomputable def f (x a : ℝ) := x * Real.log x - a * (x - 1)^2 - x + 1

-- Part 1: Monotonicity and Extreme values when a = 0
theorem part1 (x : ℝ) : f x 0 = x * Real.log x - x + 1 := sorry

theorem part1_monotonicity_intervals (x : ℝ) :
  (∀ (x : ℝ), 0 < x ∧ x < 1 → f x 0 < f 1 0) ∧
  (∀ (x : ℝ), x > 1 → f 1 0 < f x 0) ∧ 
  (f 1 0 = 0) := sorry

-- Part 2: f(x) < 0 for x > 1 and a >= 1/2
theorem part2 (x a : ℝ) (hx : x > 1) (ha : a ≥ 1/2) : f x a < 0 := sorry

end part1_part1_monotonicity_intervals_part2_l135_135511


namespace largest_4_digit_congruent_to_7_mod_19_l135_135209

theorem largest_4_digit_congruent_to_7_mod_19 : 
  ∃ x, (x % 19 = 7) ∧ 1000 ≤ x ∧ x < 10000 ∧ x = 9982 :=
by
  sorry

end largest_4_digit_congruent_to_7_mod_19_l135_135209


namespace min_max_sums_l135_135597

theorem min_max_sums (a b c d e f g : ℝ) 
    (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c)
    (h3 : 0 ≤ d) (h4 : 0 ≤ e) (h5 : 0 ≤ f) 
    (h6 : 0 ≤ g) (h_sum : a + b + c + d + e + f + g = 1) :
    (min (max (a + b + c) 
              (max (b + c + d) 
                   (max (c + d + e) 
                        (max (d + e + f) 
                             (e + f + g))))) = 1 / 3) :=
sorry

end min_max_sums_l135_135597


namespace cube_sum_decomposition_l135_135291

theorem cube_sum_decomposition : 
  (∃ (a b c d e : ℤ), (1000 * x^3 + 27) = (a * x + b) * (c * x^2 + d * x + e) ∧ (a + b + c + d + e = 92)) :=
by
  sorry

end cube_sum_decomposition_l135_135291


namespace seventeen_divides_9x_plus_5y_l135_135346

theorem seventeen_divides_9x_plus_5y (x y : ℤ) (h : 17 ∣ (2 * x + 3 * y)) : 17 ∣ (9 * x + 5 * y) :=
sorry

end seventeen_divides_9x_plus_5y_l135_135346


namespace division_and_subtraction_l135_135195

theorem division_and_subtraction :
  (12 : ℚ) / (1 / 6) - (1 / 3) = 215 / 3 :=
by
  sorry

end division_and_subtraction_l135_135195


namespace cost_price_percentage_of_marked_price_l135_135930

theorem cost_price_percentage_of_marked_price (MP CP : ℝ) (discount gain_percent : ℝ) 
  (h_discount : discount = 0.12) (h_gain_percent : gain_percent = 0.375) 
  (h_SP_def : SP = MP * (1 - discount))
  (h_SP_gain : SP = CP * (1 + gain_percent)) :
  CP / MP = 0.64 :=
by
  sorry

end cost_price_percentage_of_marked_price_l135_135930


namespace largest_integer_le_zero_of_f_l135_135977

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem largest_integer_le_zero_of_f :
  ∃ x₀ : ℝ, (f x₀ = 0) ∧ 2 ≤ x₀ ∧ x₀ < 3 ∧ (∀ k : ℤ, k ≤ x₀ → k = 2 ∨ k < 2) :=
by
  sorry

end largest_integer_le_zero_of_f_l135_135977


namespace subset_zero_in_A_l135_135807

def A := { x : ℝ | x > -1 }

theorem subset_zero_in_A : {0} ⊆ A :=
by sorry

end subset_zero_in_A_l135_135807


namespace cosine_identity_l135_135148

theorem cosine_identity
  (α : ℝ)
  (h : Real.sin (π / 6 + α) = (Real.sqrt 3) / 3) :
  Real.cos (π / 3 - α) = (Real.sqrt 3) / 2 :=
by
  sorry

end cosine_identity_l135_135148


namespace students_doing_at_least_one_hour_of_homework_l135_135225

theorem students_doing_at_least_one_hour_of_homework (total_angle : ℝ) (less_than_one_hour_angle : ℝ) 
  (h1 : total_angle = 360) (h2 : less_than_one_hour_angle = 90) :
  let less_than_one_hour_fraction := less_than_one_hour_angle / total_angle
  let less_than_one_hour_percentage := less_than_one_hour_fraction * 100
  let at_least_one_hour_percentage := 100 - less_than_one_hour_percentage
  at_least_one_hour_percentage = 75 :=
by
  let less_than_one_hour_fraction := less_than_one_hour_angle / total_angle
  let less_than_one_hour_percentage := less_than_one_hour_fraction * 100
  let at_least_one_hour_percentage := 100 - less_than_one_hour_percentage
  sorry

end students_doing_at_least_one_hour_of_homework_l135_135225


namespace find_value_of_c_l135_135205

-- Given: The transformed linear regression equation and the definition of z
theorem find_value_of_c (z : ℝ) (y : ℝ) (x : ℝ) (c : ℝ) (k : ℝ) (h1 : z = 0.4 * x + 2) (h2 : z = Real.log y) (h3 : y = c * Real.exp (k * x)) : 
  c = Real.exp 2 :=
by
  sorry

end find_value_of_c_l135_135205


namespace find_denomination_of_oliver_bills_l135_135691

-- Definitions based on conditions
def denomination (x : ℕ) : Prop :=
  let oliver_total := 10 * x + 3 * 5
  let william_total := 15 * 10 + 4 * 5
  oliver_total = william_total + 45

-- The Lean theorem statement
theorem find_denomination_of_oliver_bills (x : ℕ) : denomination x → x = 20 := by
  sorry

end find_denomination_of_oliver_bills_l135_135691


namespace initial_number_of_men_l135_135296

theorem initial_number_of_men (M A : ℕ) 
  (h1 : ((M * A) - 22 + 42 = M * (A + 2))) : M = 10 :=
by
  sorry

end initial_number_of_men_l135_135296


namespace factorize_eq_l135_135693

theorem factorize_eq (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2) := 
by
  sorry

end factorize_eq_l135_135693


namespace total_marks_scored_l135_135686

theorem total_marks_scored :
  let Keith_score := 3.5
  let Larry_score := Keith_score * 3.2
  let Danny_score := Larry_score + 5.7
  let Emma_score := (Danny_score * 2) - 1.2
  let Fiona_score := (Keith_score + Larry_score + Danny_score + Emma_score) / 4
  Keith_score + Larry_score + Danny_score + Emma_score + Fiona_score = 80.25 :=
by
  sorry

end total_marks_scored_l135_135686


namespace geometric_series_sum_l135_135418

-- Conditions
def is_geometric_series (a r : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a * r ^ n

-- The problem statement translated into Lean: Proving the sum of the series
theorem geometric_series_sum : ∃ S : ℕ → ℝ, is_geometric_series 1 (1/4) S ∧ ∑' n, S n = 4/3 :=
by
  sorry

end geometric_series_sum_l135_135418


namespace unique_digit_10D4_count_unique_digit_10D4_l135_135667

theorem unique_digit_10D4 (D : ℕ) (hD : D < 10) : 
  (5 + D) % 3 = 0 ∧ (10 * D + 4) % 4 = 0 ↔ D = 4 :=
by
  sorry

theorem count_unique_digit_10D4 :
  ∃! D, (D < 10 ∧ (5 + D) % 3 = 0 ∧ (10 * D + 4) % 4 = 0) :=
by
  use 4
  simp [unique_digit_10D4]
  sorry

end unique_digit_10D4_count_unique_digit_10D4_l135_135667


namespace intersect_point_sum_l135_135573

theorem intersect_point_sum (a' b' : ℝ) (x y : ℝ) 
    (h1 : x = (1 / 3) * y + a')
    (h2 : y = (1 / 3) * x + b')
    (h3 : x = 2)
    (h4 : y = 4) : 
    a' + b' = 4 :=
by
  sorry

end intersect_point_sum_l135_135573


namespace smallest_possible_value_of_n_l135_135875

theorem smallest_possible_value_of_n (n : ℕ) (h : lcm 60 n / gcd 60 n = 45) : n = 1080 :=
by
  sorry

end smallest_possible_value_of_n_l135_135875


namespace interval_second_bell_l135_135290

theorem interval_second_bell 
  (T : ℕ)
  (h1 : ∀ n : ℕ, n ≠ 0 → 630 % n = 0)
  (h2 : gcd T 630 = T)
  (h3 : lcm 9 (lcm 14 18) = lcm 9 (lcm 14 18))
  (h4 : 630 % lcm 9 (lcm 14 18) = 0) : 
  T = 5 :=
sorry

end interval_second_bell_l135_135290


namespace prime_p_q_r_condition_l135_135107

theorem prime_p_q_r_condition (p q r : ℕ) (hp : Nat.Prime p) (hq_pos : 0 < q) (hr_pos : 0 < r)
    (hp_not_dvd_q : ¬ (p ∣ q)) (h3_not_dvd_q : ¬ (3 ∣ q)) (eqn : p^3 = r^3 - q^2) : 
    p = 7 := sorry

end prime_p_q_r_condition_l135_135107


namespace round_trip_time_correct_l135_135781

variables (river_current_speed boat_speed_still_water distance_upstream_distance : ℕ)

def upstream_speed := boat_speed_still_water - river_current_speed
def downstream_speed := boat_speed_still_water + river_current_speed

def time_upstream := distance_upstream_distance / upstream_speed
def time_downstream := distance_upstream_distance / downstream_speed

def round_trip_time := time_upstream + time_downstream

theorem round_trip_time_correct :
  river_current_speed = 10 →
  boat_speed_still_water = 50 →
  distance_upstream_distance = 120 →
  round_trip_time river_current_speed boat_speed_still_water distance_upstream_distance = 5 :=
by
  intros rc bs d
  sorry

end round_trip_time_correct_l135_135781


namespace candy_bar_cost_l135_135097

-- Define the conditions
def cost_gum_over_candy_bar (C G : ℝ) : Prop :=
  G = (1/2) * C

def total_cost (C G : ℝ) : Prop :=
  2 * G + 3 * C = 6

-- Define the proof problem
theorem candy_bar_cost (C G : ℝ) (h1 : cost_gum_over_candy_bar C G) (h2 : total_cost C G) : C = 1.5 :=
by
  sorry

end candy_bar_cost_l135_135097


namespace triangular_number_30_l135_135265

theorem triangular_number_30 : 
  (∃ (T : ℕ), T = 30 * (30 + 1) / 2 ∧ T = 465) :=
by 
  sorry

end triangular_number_30_l135_135265


namespace factor_expression_l135_135836

theorem factor_expression (x : ℝ) : 45 * x^3 + 135 * x^2 = 45 * x^2 * (x + 3) :=
  by
    sorry

end factor_expression_l135_135836


namespace females_in_orchestra_not_in_band_l135_135946

theorem females_in_orchestra_not_in_band 
  (females_in_band : ℤ) 
  (males_in_band : ℤ) 
  (females_in_orchestra : ℤ) 
  (males_in_orchestra : ℤ) 
  (females_in_both : ℤ) 
  (total_members : ℤ) 
  (h1 : females_in_band = 120) 
  (h2 : males_in_band = 100) 
  (h3 : females_in_orchestra = 100) 
  (h4 : males_in_orchestra = 120) 
  (h5 : females_in_both = 80) 
  (h6 : total_members = 260) : 
  (females_in_orchestra - females_in_both = 20) := 
  sorry

end females_in_orchestra_not_in_band_l135_135946


namespace log_positive_interval_l135_135618

noncomputable def f (a x : ℝ) : ℝ := Real.log (2 * x - a) / Real.log a

theorem log_positive_interval (a : ℝ) :
  (∀ x, x ∈ Set.Icc (1 / 2) (2 / 3) → f a x > 0) ↔ (1 / 3 < a ∧ a < 1) := by
  sorry

end log_positive_interval_l135_135618


namespace smallest_positive_multiple_of_32_l135_135114

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ ∃ k : ℕ, k > 0 ∧ n = 32 * k ∧ n = 32 := by
  use 32
  constructor
  · exact Nat.zero_lt_succ 31
  · use 1
    constructor
    · exact Nat.zero_lt_succ 0
    · constructor
      · rfl
      · rfl

end smallest_positive_multiple_of_32_l135_135114


namespace reservoir_capacity_l135_135539

theorem reservoir_capacity (x : ℝ) (h1 : (3 / 8) * x - (1 / 4) * x = 100) : x = 800 :=
by
  sorry

end reservoir_capacity_l135_135539


namespace ball_first_bounce_less_than_30_l135_135160

theorem ball_first_bounce_less_than_30 (b : ℕ) :
  (243 * ((2: ℝ) / 3) ^ b < 30) ↔ (b ≥ 6) :=
sorry

end ball_first_bounce_less_than_30_l135_135160


namespace new_person_weight_increase_avg_l135_135866

theorem new_person_weight_increase_avg
  (W : ℝ) -- total weight of the original 20 people
  (new_person_weight : ℝ) -- weight of the new person
  (h1 : (W - 80 + new_person_weight) = W + 20 * 15) -- condition given in the problem
  : new_person_weight = 380 := 
sorry

end new_person_weight_increase_avg_l135_135866


namespace lcm_12_18_l135_135339

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l135_135339


namespace quadratic_has_two_distinct_real_roots_l135_135668

-- Given the discriminant condition Δ = b^2 - 4ac > 0
theorem quadratic_has_two_distinct_real_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) := 
  sorry

end quadratic_has_two_distinct_real_roots_l135_135668


namespace no_integer_solutions_3a2_eq_b2_plus_1_l135_135487

theorem no_integer_solutions_3a2_eq_b2_plus_1 : 
  ¬ ∃ a b : ℤ, 3 * a^2 = b^2 + 1 :=
by
  intro h
  obtain ⟨a, b, hab⟩ := h
  sorry

end no_integer_solutions_3a2_eq_b2_plus_1_l135_135487


namespace smallest_collection_l135_135840

def Yoongi_collected : ℕ := 4
def Jungkook_collected : ℕ := 6 * 3
def Yuna_collected : ℕ := 5

theorem smallest_collection : Yoongi_collected = 4 ∧ Yoongi_collected ≤ Jungkook_collected ∧ Yoongi_collected ≤ Yuna_collected := by
  sorry

end smallest_collection_l135_135840


namespace find_min_value_x_l135_135056

theorem find_min_value_x (x y z : ℝ) (h1 : x + y + z = 6) (h2 : xy + xz + yz = 10) : 
  ∃ (x_min : ℝ), (∀ (x' : ℝ), (∀ y' z', x' + y' + z' = 6 ∧ x' * y' + x' * z' + y' * z' = 10 → x' ≥ x_min)) ∧ x_min = 2 / 3 :=
sorry

end find_min_value_x_l135_135056


namespace general_solution_linear_diophantine_l135_135610

theorem general_solution_linear_diophantine (a b c : ℤ) (h_coprime : Int.gcd a b = 1)
    (x1 y1 : ℤ) (h_particular_solution : a * x1 + b * y1 = c) :
    ∃ (t : ℤ), (∃ (x y : ℤ), x = x1 + b * t ∧ y = y1 - a * t ∧ a * x + b * y = c) ∧
               (∃ (x' y' : ℤ), x' = x1 - b * t ∧ y' = y1 + a * t ∧ a * x' + b * y' = c) :=
by
  sorry

end general_solution_linear_diophantine_l135_135610


namespace expression_parity_l135_135232

variable (o n c : ℕ)

def is_odd (x : ℕ) : Prop := ∃ k, x = 2 * k + 1

theorem expression_parity (ho : is_odd o) (hc : is_odd c) : 
  (o^2 + n * o + c) % 2 = 0 :=
  sorry

end expression_parity_l135_135232


namespace total_hours_proof_l135_135248

-- Conditions
def half_hour_show_episodes : ℕ := 24
def one_hour_show_episodes : ℕ := 12
def half_hour_per_episode : ℝ := 0.5
def one_hour_per_episode : ℝ := 1.0

-- Define the total hours Tim watched
def total_hours_watched : ℝ :=
  half_hour_show_episodes * half_hour_per_episode + one_hour_show_episodes * one_hour_per_episode

-- Prove that the total hours watched is 24
theorem total_hours_proof : total_hours_watched = 24 := by
  sorry

end total_hours_proof_l135_135248


namespace system_inequalities_1_system_inequalities_2_l135_135892

theorem system_inequalities_1 (x: ℝ):
  (4 * (x + 1) ≤ 7 * x + 10) → (x - 5 < (x - 8)/3) → (-2 ≤ x ∧ x < 7 / 2) :=
by
  intros h1 h2
  sorry

theorem system_inequalities_2 (x: ℝ):
  (x - 3 * (x - 2) ≥ 4) → ((2 * x - 1) / 5 ≥ (x + 1) / 2) → (x ≤ -7) :=
by
  intros h1 h2
  sorry

end system_inequalities_1_system_inequalities_2_l135_135892


namespace combined_forgotten_angles_l135_135336

-- Define primary conditions
def initial_angle_sum : ℝ := 2873
def correct_angle_sum : ℝ := 16 * 180

-- The theorem to prove
theorem combined_forgotten_angles : correct_angle_sum - initial_angle_sum = 7 :=
by sorry

end combined_forgotten_angles_l135_135336


namespace find_ratio_b_c_l135_135488

variable {a b c A B C : Real}

theorem find_ratio_b_c
  (h1 : a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C)
  (h2 : Real.cos A = -1 / 4) :
  b / c = 6 :=
sorry

end find_ratio_b_c_l135_135488


namespace range_of_b_l135_135371

noncomputable def a_n (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem range_of_b (b : ℝ) : (∀ n : ℕ, 0 < n → a_n (n+1) b > a_n n b) ↔ (-3 < b) :=
by
    sorry

end range_of_b_l135_135371


namespace highest_score_batsman_l135_135344

variable (H L : ℕ)

theorem highest_score_batsman :
  (60 * 46) = (58 * 44 + H + L) ∧ (H - L = 190) → H = 199 :=
by
  intros h
  sorry

end highest_score_batsman_l135_135344


namespace find_a_l135_135672

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x0 a : ℝ) (h : f x0 a - g x0 a = 3) : a = -1 - Real.log 2 := sorry

end find_a_l135_135672


namespace cos_pi_div_four_minus_alpha_l135_135951

theorem cos_pi_div_four_minus_alpha (α : ℝ) (h : Real.sin (π / 4 + α) = 2 / 3) : 
    Real.cos (π / 4 - α) = -Real.sqrt 5 / 3 :=
sorry

end cos_pi_div_four_minus_alpha_l135_135951


namespace investment_duration_l135_135329

noncomputable def log (x : ℝ) := Real.log x

theorem investment_duration 
  (P A : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (t : ℝ) 
  (hP : P = 3000) 
  (hA : A = 3630) 
  (hr : r = 0.10) 
  (hn : n = 1) 
  (ht : A = P * (1 + r / n) ^ (n * t)) :
  t = 2 :=
by
  sorry

end investment_duration_l135_135329


namespace focus_of_parabola_l135_135914

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l135_135914


namespace rectangle_measurement_error_l135_135706

theorem rectangle_measurement_error 
  (L W : ℝ)
  (measured_length : ℝ := 1.05 * L)
  (measured_width : ℝ := 0.96 * W)
  (actual_area : ℝ := L * W)
  (calculated_area : ℝ := measured_length * measured_width)
  (error : ℝ := calculated_area - actual_area) :
  ((error / actual_area) * 100) = 0.8 :=
sorry

end rectangle_measurement_error_l135_135706


namespace total_earnings_correct_l135_135561

noncomputable def total_earnings : ℝ :=
  let earnings1 := 12 * (2 + 15 / 60)
  let earnings2 := 15 * (1 + 40 / 60)
  let earnings3 := 10 * (3 + 10 / 60)
  earnings1 + earnings2 + earnings3

theorem total_earnings_correct : total_earnings = 83.75 := by
  sorry

end total_earnings_correct_l135_135561


namespace gandalf_reachability_l135_135738

theorem gandalf_reachability :
  ∀ (k : ℕ), ∃ (s : ℕ → ℕ) (m : ℕ), (s 0 = 1) ∧ (s m = k) ∧ (∀ i < m, s (i + 1) = 2 * s i ∨ s (i + 1) = 3 * s i + 1) := 
by
  sorry

end gandalf_reachability_l135_135738


namespace sum_first_11_terms_l135_135030

variable {a : ℕ → ℕ} -- a is the arithmetic sequence

-- Condition: a_4 + a_8 = 26
axiom condition : a 4 + a 8 = 26

-- Definition of arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of the sum of the first 11 terms
def S_11 (a : ℕ → ℕ) : ℕ := (11 * (a 1 + a 11)) / 2

-- The proof problem statement
theorem sum_first_11_terms (h : is_arithmetic_sequence a) : S_11 a = 143 := 
by 
  sorry

end sum_first_11_terms_l135_135030


namespace distance_between_foci_l135_135720

-- Define the conditions
def is_asymptote (y x : ℝ) (slope intercept : ℝ) : Prop := y = slope * x + intercept

def passes_through_point (x y x0 y0 : ℝ) : Prop := x = x0 ∧ y = y0

-- The hyperbola conditions
axiom asymptote1 : ∀ x y : ℝ, is_asymptote y x 2 3
axiom asymptote2 : ∀ x y : ℝ, is_asymptote y x (-2) 5
axiom hyperbola_passes : passes_through_point 2 9 2 9

-- The proof problem statement: distance between the foci
theorem distance_between_foci : ∀ {a b c : ℝ}, ∃ c, (c^2 = 22.75 + 22.75) → 2 * c = 2 * Real.sqrt 45.5 :=
by
  sorry

end distance_between_foci_l135_135720


namespace squared_sum_inverse_l135_135431

theorem squared_sum_inverse (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 :=
by
  sorry

end squared_sum_inverse_l135_135431


namespace mr_smith_payment_l135_135189

theorem mr_smith_payment {balance : ℝ} {percentage : ℝ} 
  (h_bal : balance = 150) (h_percent : percentage = 0.02) :
  (balance + balance * percentage) = 153 :=
by
  sorry

end mr_smith_payment_l135_135189


namespace hortense_flower_production_l135_135188

-- Define the initial conditions
def daisy_seeds : ℕ := 25
def sunflower_seeds : ℕ := 25
def daisy_germination_rate : ℚ := 0.60
def sunflower_germination_rate : ℚ := 0.80
def flower_production_rate : ℚ := 0.80

-- Prove the number of plants that produce flowers
theorem hortense_flower_production :
  (daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) * flower_production_rate = 28 :=
by sorry

end hortense_flower_production_l135_135188


namespace remainder_142_to_14_l135_135873

theorem remainder_142_to_14 (N k : ℤ) 
  (h : N = 142 * k + 110) : N % 14 = 8 :=
sorry

end remainder_142_to_14_l135_135873


namespace arithmetic_geometric_sequence_product_l135_135115

theorem arithmetic_geometric_sequence_product :
  (∀ n : ℕ, ∃ d : ℝ, ∀ m : ℕ, a_n = a_1 + m * d) →
  (∀ n : ℕ, ∃ q : ℝ, ∀ m : ℕ, b_n = b_1 * q ^ m) →
  a_1 = 1 → a_2 = 2 →
  b_1 = 1 → b_2 = 2 →
  a_5 * b_5 = 80 :=
by
  sorry

end arithmetic_geometric_sequence_product_l135_135115


namespace q_implies_not_p_l135_135555

-- Define the conditions p and q
def p (x : ℝ) := x < -1
def q (x : ℝ) := x^2 - x - 2 > 0

-- Prove that q implies ¬p
theorem q_implies_not_p (x : ℝ) : q x → ¬ p x := by
  intros hq hp
  -- Provide the steps of logic here
  sorry

end q_implies_not_p_l135_135555


namespace no_real_pairs_arithmetic_prog_l135_135910

theorem no_real_pairs_arithmetic_prog :
  ¬ ∃ a b : ℝ, (a = (1 / 2) * (8 + b)) ∧ (a + a * b = 2 * b) := by
sorry

end no_real_pairs_arithmetic_prog_l135_135910


namespace percentage_of_girls_with_dogs_l135_135548

theorem percentage_of_girls_with_dogs (students total_students : ℕ)
(h_total_students : total_students = 100)
(girls boys : ℕ)
(h_half_students : girls = total_students / 2 ∧ boys = total_students / 2)
(boys_with_dogs : ℕ)
(h_boys_with_dogs : boys_with_dogs = boys / 10)
(total_with_dogs : ℕ)
(h_total_with_dogs : total_with_dogs = 15)
(girls_with_dogs : ℕ)
(h_girls_with_dogs : girls_with_dogs = total_with_dogs - boys_with_dogs)
: (girls_with_dogs * 100 / girls = 20) :=
by
  sorry

end percentage_of_girls_with_dogs_l135_135548


namespace equivalent_representations_l135_135632

theorem equivalent_representations :
  (16 / 20 = 24 / 30) ∧
  (80 / 100 = 4 / 5) ∧
  (4 / 5 = 0.8) :=
by 
  sorry

end equivalent_representations_l135_135632


namespace intersection_points_3_l135_135920

def eq1 (x y : ℝ) : Prop := (x - y + 3) * (2 * x + 3 * y - 9) = 0
def eq2 (x y : ℝ) : Prop := (2 * x - y + 2) * (x + 3 * y - 6) = 0

theorem intersection_points_3 :
  (∃ x y : ℝ, eq1 x y ∧ eq2 x y) ∧
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    eq1 x1 y1 ∧ eq2 x1 y1 ∧ 
    eq1 x2 y2 ∧ eq2 x2 y2 ∧ 
    eq1 x3 y3 ∧ eq2 x3 y3 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x2, y2) ≠ (x3, y3)) :=
sorry

end intersection_points_3_l135_135920


namespace constant_term_q_l135_135043

theorem constant_term_q (p q r : Polynomial ℝ) 
  (hp_const : p.coeff 0 = 6) 
  (hr_const : (p * q).coeff 0 = -18) : q.coeff 0 = -3 :=
sorry

end constant_term_q_l135_135043


namespace arithmetic_sequence_a10_l135_135022

variable {a : ℕ → ℝ}

-- Given the sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, (n < m) → (a (m + 1) - a m = a (n + 1) - a n)

-- Conditions
theorem arithmetic_sequence_a10 (h_arith : is_arithmetic_sequence a) 
                                (h1 : a 6 + a 8 = 16)
                                (h2 : a 4 = 1) :
  a 10 = 15 :=
sorry

end arithmetic_sequence_a10_l135_135022


namespace eq_neg2_multi_l135_135903

theorem eq_neg2_multi {m n : ℝ} (h : m = n) : -2 * m = -2 * n :=
by sorry

end eq_neg2_multi_l135_135903


namespace find_second_number_l135_135772

theorem find_second_number
  (x y z : ℚ)
  (h1 : x + y + z = 120)
  (h2 : x = (3 : ℚ) / 4 * y)
  (h3 : z = (7 : ℚ) / 5 * y) :
  y = 800 / 21 :=
by
  sorry

end find_second_number_l135_135772


namespace probability_distribution_correct_l135_135774

noncomputable def X_possible_scores : Set ℤ := {-90, -30, 30, 90}

def prob_correct : ℚ := 0.8
def prob_incorrect : ℚ := 1 - prob_correct

def P_X_neg90 : ℚ := prob_incorrect ^ 3
def P_X_neg30 : ℚ := 3 * prob_correct * prob_incorrect ^ 2
def P_X_30 : ℚ := 3 * prob_correct ^ 2 * prob_incorrect
def P_X_90 : ℚ := prob_correct ^ 3

def P_advance : ℚ := P_X_30 + P_X_90

theorem probability_distribution_correct :
  (P_X_neg90 = (1/125) ∧ P_X_neg30 = (12/125) ∧ P_X_30 = (48/125) ∧ P_X_90 = (64/125)) ∧ 
  P_advance = (112/125) := 
by
  sorry

end probability_distribution_correct_l135_135774


namespace range_of_m_l135_135687

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ 0 < m ∧ m < 8 :=
sorry

end range_of_m_l135_135687


namespace GCF_of_48_180_98_l135_135120

theorem GCF_of_48_180_98 : Nat.gcd (Nat.gcd 48 180) 98 = 2 :=
by
  sorry

end GCF_of_48_180_98_l135_135120


namespace min_value_of_reciprocal_squares_l135_135926

variable (a b : ℝ)

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x + a^2 - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0

-- The condition that the two circles are externally tangent and have three common tangents
def externallyTangent (a b : ℝ) : Prop :=
  -- From the derivation in the solution, we must have:
  (a^2 + 4 * b^2 = 9)

-- Ensure a and b are non-zero
def nonzero (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0

-- State the main theorem to prove
theorem min_value_of_reciprocal_squares (h1 : externallyTangent a b) (h2 : nonzero a b) :
  (1 / a^2) + (1 / b^2) = 1 := 
sorry

end min_value_of_reciprocal_squares_l135_135926


namespace triangle_is_isosceles_l135_135457

theorem triangle_is_isosceles (A B C : ℝ)
  (h : Real.log (Real.sin A) - Real.log (Real.cos B) - Real.log (Real.sin C) = Real.log 2) :
  ∃ a b c : ℝ, a = b ∨ b = c ∨ a = c := 
sorry

end triangle_is_isosceles_l135_135457


namespace sum_eighth_row_l135_135712

-- Definitions based on the conditions
def sum_of_interior_numbers (n : ℕ) : ℕ := 2^(n-1) - 2

axiom sum_fifth_row : sum_of_interior_numbers 5 = 14
axiom sum_sixth_row : sum_of_interior_numbers 6 = 30

-- The proof problem statement
theorem sum_eighth_row : sum_of_interior_numbers 8 = 126 :=
by {
  sorry
}

end sum_eighth_row_l135_135712


namespace sum_of_areas_of_squares_l135_135354

theorem sum_of_areas_of_squares (a b x : ℕ) 
  (h_overlapping_min : 9 ≤ (min a b) ^ 2)
  (h_overlapping_max : (min a b) ^ 2 ≤ 25)
  (h_sum_of_sides : a + b + x = 23) :
  a^2 + b^2 + x^2 = 189 := 
sorry

end sum_of_areas_of_squares_l135_135354


namespace stone_123_is_12_l135_135217

/-- Definitions: 
  1. Fifteen stones arranged in a circle counted in a specific pattern: clockwise and counterclockwise.
  2. The sequence of stones enumerated from 1 to 123
  3. The repeating pattern occurs every 28 stones
-/
def stones_counted (n : Nat) : Nat :=
  if n % 28 <= 15 then (n % 28) else (28 - (n % 28) + 1)

theorem stone_123_is_12 : stones_counted 123 = 12 :=
by
  sorry

end stone_123_is_12_l135_135217


namespace polygon_interior_angles_540_implies_pentagon_l135_135420

theorem polygon_interior_angles_540_implies_pentagon
  (n : ℕ) (H: 180 * (n - 2) = 540) : n = 5 :=
sorry

end polygon_interior_angles_540_implies_pentagon_l135_135420


namespace eval_ceil_sqrt_sum_l135_135085

theorem eval_ceil_sqrt_sum :
  (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉) = 27 :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by sorry
  have h2 : 5 < Real.sqrt 33 ∧ Real.sqrt 33 < 6 := by sorry
  have h3 : 18 < Real.sqrt 333 ∧ Real.sqrt 333 < 19 := by sorry
  sorry

end eval_ceil_sqrt_sum_l135_135085


namespace f_neg_two_l135_135057

noncomputable def f : ℝ → ℝ := sorry

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

variables (f_odd : is_odd_function f)
variables (f_two : f 2 = 2)

theorem f_neg_two : f (-2) = -2 :=
by
  -- Given that f is an odd function and f(2) = 2
  sorry

end f_neg_two_l135_135057


namespace solution_is_consecutive_even_integers_l135_135852

def consecutive_even_integers_solution_exists : Prop :=
  ∃ (x y z w : ℕ), (x + y + z + w = 68) ∧ 
                   (y = x + 2) ∧ (z = x + 4) ∧ (w = x + 6) ∧
                   (x % 2 = 0) ∧ (y % 2 = 0) ∧ (z % 2 = 0) ∧ (w % 2 = 0)

theorem solution_is_consecutive_even_integers : consecutive_even_integers_solution_exists :=
sorry

end solution_is_consecutive_even_integers_l135_135852


namespace zeros_of_f_l135_135527

noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem zeros_of_f (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (∃ x, a < x ∧ x < b ∧ f a b c x = 0) ∧ (∃ y, b < y ∧ y < c ∧ f a b c y = 0) :=
by
  sorry

end zeros_of_f_l135_135527


namespace total_money_shared_l135_135779

-- Let us define the conditions
def ratio (a b c : ℕ) : Prop := ∃ k : ℕ, (2 * k = a) ∧ (3 * k = b) ∧ (8 * k = c)

def olivia_share := 30

-- Our goal is to prove the total amount of money shared
theorem total_money_shared (a b c : ℕ) (h_ratio : ratio a b c) (h_olivia : a = olivia_share) :
    a + b + c = 195 :=
by
  sorry

end total_money_shared_l135_135779


namespace smaller_number_is_17_l135_135858

theorem smaller_number_is_17 (x y : ℕ) (h1 : x * y = 323) (h2 : x - y = 2) : y = 17 :=
sorry

end smaller_number_is_17_l135_135858


namespace third_consecutive_even_sum_52_l135_135848

theorem third_consecutive_even_sum_52
  (x : ℤ)
  (h : x + (x + 2) + (x + 4) + (x + 6) = 52) :
  x + 4 = 14 :=
by
  sorry

end third_consecutive_even_sum_52_l135_135848


namespace fraction_operation_correct_l135_135411

theorem fraction_operation_correct (a b : ℝ) (h : 0.2 * a + 0.5 * b ≠ 0) : 
  (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) :=
sorry

end fraction_operation_correct_l135_135411


namespace cos_equation_solution_l135_135330

open Real

theorem cos_equation_solution (m : ℝ) :
  (∀ x : ℝ, 4 * cos x - cos x^2 + m - 3 = 0) ↔ (0 ≤ m ∧ m ≤ 8) := by
  sorry

end cos_equation_solution_l135_135330


namespace vector_addition_parallel_l135_135943

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem vector_addition_parallel:
  ∀ x : ℝ, parallel (2, 1) (x, -2) → a + b x = ((-2 : ℝ), -1) :=
by
  intros x h
  sorry

end vector_addition_parallel_l135_135943


namespace paula_aunt_money_l135_135251

theorem paula_aunt_money
  (shirts_cost : ℕ := 2 * 11)
  (pants_cost : ℕ := 13)
  (money_left : ℕ := 74) : 
  shirts_cost + pants_cost + money_left = 109 :=
by
  sorry

end paula_aunt_money_l135_135251


namespace equivalence_of_statements_l135_135430

-- Variables used in the statements
variable (P Q : Prop)

-- Proof problem statement
theorem equivalence_of_statements : (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by sorry

end equivalence_of_statements_l135_135430


namespace solve_for_square_l135_135799

theorem solve_for_square (x : ℝ) 
  (h : 10 + 9 + 8 * 7 / x + 6 - 5 * 4 - 3 * 2 = 1) : 
  x = 28 := 
by 
  sorry

end solve_for_square_l135_135799


namespace expression_value_l135_135156

theorem expression_value (x y : ℝ) (h : x + y = -1) : 
  x^4 + 5 * x^3 * y + x^2 * y + 8 * x^2 * y^2 + x * y^2 + 5 * x * y^3 + y^4 = 1 :=
by
  sorry

end expression_value_l135_135156


namespace bag_ratio_l135_135332

noncomputable def ratio_of_costs : ℚ := 1 / 2

theorem bag_ratio :
  ∃ (shirt_cost shoes_cost total_cost bag_cost : ℚ),
    shirt_cost = 7 ∧
    shoes_cost = shirt_cost + 3 ∧
    total_cost = 2 * shirt_cost + shoes_cost ∧
    bag_cost = 36 - total_cost ∧
    bag_cost / total_cost = ratio_of_costs :=
sorry

end bag_ratio_l135_135332


namespace length_of_segment_AC_l135_135662

theorem length_of_segment_AC :
  ∀ (a b h: ℝ),
    (a = b) →
    (h = a * Real.sqrt 2) →
    (4 = (a + b - h) / 2) →
    a = 4 * Real.sqrt 2 + 8 :=
by
  sorry

end length_of_segment_AC_l135_135662


namespace geometric_sequence_common_ratio_l135_135428

theorem geometric_sequence_common_ratio (a : ℕ → ℚ) (q : ℚ) :
  (∀ n, a n = a 2 * q ^ (n - 2)) ∧ a 2 = 2 ∧ a 6 = 1 / 8 →
  (q = 1 / 2 ∨ q = -1 / 2) :=
by
  sorry

end geometric_sequence_common_ratio_l135_135428


namespace hypotenuse_not_5_cm_l135_135557

theorem hypotenuse_not_5_cm (a b c : ℝ) (h₀ : a + b = 8) (h₁ : a^2 + b^2 = c^2) : c ≠ 5 := by
  sorry

end hypotenuse_not_5_cm_l135_135557


namespace value_of_b_l135_135141

theorem value_of_b (y b : ℝ) (hy : y > 0) (h : (4 * y) / b + (3 * y) / 10 = 0.5 * y) : b = 20 :=
by
  -- Proof omitted for brevity
  sorry

end value_of_b_l135_135141


namespace find_third_month_sale_l135_135596

theorem find_third_month_sale
  (sale_1 sale_2 sale_3 sale_4 sale_5 sale_6 : ℕ)
  (h1 : sale_1 = 800)
  (h2 : sale_2 = 900)
  (h4 : sale_4 = 700)
  (h5 : sale_5 = 800)
  (h6 : sale_6 = 900)
  (h_avg : (sale_1 + sale_2 + sale_3 + sale_4 + sale_5 + sale_6) / 6 = 850) : 
  sale_3 = 1000 :=
by
  sorry

end find_third_month_sale_l135_135596


namespace real_part_of_i_squared_times_1_plus_i_l135_135274

noncomputable def imaginary_unit : ℂ := Complex.I

theorem real_part_of_i_squared_times_1_plus_i :
  (Complex.re (imaginary_unit^2 * (1 + imaginary_unit))) = -1 :=
by
  sorry

end real_part_of_i_squared_times_1_plus_i_l135_135274


namespace find_rate_of_current_l135_135753

noncomputable def rate_of_current : ℝ := 
  let speed_still_water := 42
  let distance_downstream := 33.733333333333334
  let time_hours := 44 / 60
  (distance_downstream / time_hours) - speed_still_water

theorem find_rate_of_current : rate_of_current = 4 :=
by sorry

end find_rate_of_current_l135_135753


namespace max_omega_for_increasing_l135_135086

noncomputable def sin_function (ω : ℕ) (x : ℝ) := Real.sin (ω * x + Real.pi / 6)

def is_monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem max_omega_for_increasing : ∀ (ω : ℕ), (0 < ω) →
  is_monotonically_increasing_on (sin_function ω) (Real.pi / 6) (Real.pi / 4) ↔ ω ≤ 9 :=
sorry

end max_omega_for_increasing_l135_135086


namespace boat_speed_still_water_l135_135000

theorem boat_speed_still_water (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 16) (h2 : upstream_speed = 9) : 
  (downstream_speed + upstream_speed) / 2 = 12.5 := 
by
  -- conditions explicitly stated above
  sorry

end boat_speed_still_water_l135_135000


namespace g_g_2_equals_226_l135_135500

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 4

theorem g_g_2_equals_226 : g (g 2) = 226 := by
  sorry

end g_g_2_equals_226_l135_135500


namespace molly_total_swim_l135_135532

variable (meters_saturday : ℕ) (meters_sunday : ℕ)

theorem molly_total_swim (h1 : meters_saturday = 45) (h2 : meters_sunday = 28) : meters_saturday + meters_sunday = 73 := by
  sorry

end molly_total_swim_l135_135532


namespace cubes_difference_l135_135640

theorem cubes_difference 
  (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
  sorry

end cubes_difference_l135_135640


namespace problem_l135_135743

variable (x y : ℝ)

theorem problem (h1 : x + 3 * y = 6) (h2 : x * y = -12) : x^2 + 9 * y^2 = 108 :=
sorry

end problem_l135_135743


namespace quotient_is_76_l135_135709

def original_number : ℕ := 12401
def divisor : ℕ := 163
def remainder : ℕ := 13

theorem quotient_is_76 : (original_number - remainder) / divisor = 76 :=
by
  sorry

end quotient_is_76_l135_135709


namespace valentines_left_l135_135213

def initial_valentines : ℕ := 60
def valentines_given_away : ℕ := 16
def valentines_received : ℕ := 5

theorem valentines_left : (initial_valentines - valentines_given_away + valentines_received) = 49 :=
by sorry

end valentines_left_l135_135213


namespace compute_f_1_g_3_l135_135480

def f (x : ℝ) := 3 * x - 5
def g (x : ℝ) := x + 1

theorem compute_f_1_g_3 : f (1 + g 3) = 10 := by
  sorry

end compute_f_1_g_3_l135_135480


namespace general_term_of_sequence_l135_135577

theorem general_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hSn : ∀ n, S n = 3 * n^2 - n + 1) :
  (∀ n, a n = if n = 1 then 3 else 6 * n - 4) :=
by
  sorry

end general_term_of_sequence_l135_135577


namespace sufficient_but_not_necessary_condition_l135_135258

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x > 0, (m^2 - m - 1) * x^(m - 1) > 0 → m = 2) →
  (|m - 2| < 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l135_135258


namespace three_different_suits_probability_l135_135602

def probability_three_different_suits := (39 / 51) * (35 / 50) = 91 / 170

theorem three_different_suits_probability (deck : Finset (Fin 52)) (h : deck.card = 52) :
  probability_three_different_suits :=
sorry

end three_different_suits_probability_l135_135602


namespace minimum_value_a_plus_b_plus_c_l135_135350

theorem minimum_value_a_plus_b_plus_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 4 * b + 7 * c ≤ 2 * a * b * c) : a + b + c ≥ 15 / 2 :=
by
  sorry

end minimum_value_a_plus_b_plus_c_l135_135350


namespace ball_hits_ground_l135_135939

theorem ball_hits_ground :
  ∃ t : ℝ, -16 * t^2 + 20 * t + 100 = 0 ∧ t = (5 + Real.sqrt 425) / 8 :=
by
  sorry

end ball_hits_ground_l135_135939


namespace not_pass_first_quadrant_l135_135103

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  (1/5)^(x + 1) + m

theorem not_pass_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -(1/5) :=
  by
  sorry

end not_pass_first_quadrant_l135_135103


namespace lisa_earns_more_than_tommy_l135_135358

theorem lisa_earns_more_than_tommy {total_earnings : ℤ} (h1 : total_earnings = 60) :
  let lisa_earnings := total_earnings / 2
  let tommy_earnings := lisa_earnings / 2
  lisa_earnings - tommy_earnings = 15 :=
by
  sorry

end lisa_earns_more_than_tommy_l135_135358


namespace f_log3_54_l135_135177

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x ∧ x < 1 then 3^x else sorry

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f (x)
def functional_equation (f : ℝ → ℝ) := ∀ x, f (x + 2) = -1 / f (x)

-- Hypotheses based on conditions
variable (f : ℝ → ℝ)
axiom f_is_odd : odd_function f
axiom f_is_periodic : periodic_function f 4
axiom f_functional : functional_equation f

-- Main goal
theorem f_log3_54 : f (Real.log 54 / Real.log 3) = -3 / 2 := by
  sorry

end f_log3_54_l135_135177


namespace angle_measure_l135_135584

theorem angle_measure (x y : ℝ) 
  (h1 : y = 3 * x + 10) 
  (h2 : x + y = 180) : x = 42.5 :=
by
  -- Proof goes here
  sorry

end angle_measure_l135_135584


namespace taxi_division_number_of_ways_to_divide_six_people_l135_135166

theorem taxi_division (people : Finset ℕ) (h : people.card = 6) (taxi1 taxi2 : Finset ℕ) 
  (h1 : taxi1.card ≤ 4) (h2 : taxi2.card ≤ 4) (h_union : people = taxi1 ∪ taxi2) (h_disjoint : Disjoint taxi1 taxi2) :
  (taxi1.card = 3 ∧ taxi2.card = 3) ∨ 
  (taxi1.card = 4 ∧ taxi2.card = 2) :=
sorry

theorem number_of_ways_to_divide_six_people : 
  ∃ n : ℕ, n = 50 :=
sorry

end taxi_division_number_of_ways_to_divide_six_people_l135_135166


namespace sum_of_non_solutions_l135_135680

theorem sum_of_non_solutions (A B C x: ℝ) 
  (h1 : A = 2) 
  (h2 : B = C / 2) 
  (h3 : C = 28) 
  (eq_inf_solutions : ∀ x, (x ≠ -C ∧ x ≠ -14) → 
  (x + B) * (A * x + 56) = 2 * ((x + C) * (x + 14))) : 
  (-14 + -28) = -42 :=
by
  sorry

end sum_of_non_solutions_l135_135680


namespace percentage_less_than_l135_135261

theorem percentage_less_than (p j t : ℝ) (h1 : j = 0.75 * p) (h2 : j = 0.80 * t) : 
  t = (1 - 0.0625) * p := 
by 
  sorry

end percentage_less_than_l135_135261


namespace two_times_x_equals_two_l135_135879

theorem two_times_x_equals_two (x : ℝ) (h : x = 1) : 2 * x = 2 := by
  sorry

end two_times_x_equals_two_l135_135879


namespace max_chickens_ducks_l135_135279

theorem max_chickens_ducks (x y : ℕ) 
  (h1 : ∀ (k : ℕ), k = 6 → x + y - 6 ≥ 2) 
  (h2 : ∀ (k : ℕ), k = 9 → y ≥ 1) : 
  x + y ≤ 12 :=
sorry

end max_chickens_ducks_l135_135279


namespace unique_pyramid_formation_l135_135479

theorem unique_pyramid_formation:
  ∀ (positions: Finset ℕ)
    (is_position_valid: ℕ → Prop),
    (positions.card = 5) → 
    (∀ n ∈ positions, n < 5) → 
    (∃! n, is_position_valid n) :=
by
  sorry

end unique_pyramid_formation_l135_135479


namespace range_of_4a_minus_2b_l135_135744

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b)
  (h2 : a - b ≤ 2)
  (h3 : 2 ≤ a + b)
  (h4 : a + b ≤ 4) : 
  5 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 10 :=
by
  sorry

end range_of_4a_minus_2b_l135_135744


namespace total_clouds_count_l135_135991

-- Definitions based on the conditions
def carson_clouds := 12
def little_brother_clouds := 5 * carson_clouds
def older_sister_clouds := carson_clouds / 2

-- The theorem statement that needs to be proved
theorem total_clouds_count : carson_clouds + little_brother_clouds + older_sister_clouds = 78 := by
  -- Definitions
  have h1 : carson_clouds = 12 := rfl
  have h2 : little_brother_clouds = 5 * 12 := rfl
  have h3 : older_sister_clouds = 12 / 2 := rfl
  sorry

end total_clouds_count_l135_135991


namespace moneyEarnedDuringHarvest_l135_135048

-- Define the weekly earnings, duration of harvest, and weekly rent.
def weeklyEarnings : ℕ := 403
def durationOfHarvest : ℕ := 233
def weeklyRent : ℕ := 49

-- Define total earnings and total rent.
def totalEarnings : ℕ := weeklyEarnings * durationOfHarvest
def totalRent : ℕ := weeklyRent * durationOfHarvest

-- Calculate the money earned after rent.
def moneyEarnedAfterRent : ℕ := totalEarnings - totalRent

-- The theorem to prove.
theorem moneyEarnedDuringHarvest : moneyEarnedAfterRent = 82482 :=
  by
  sorry

end moneyEarnedDuringHarvest_l135_135048


namespace proof_problem_l135_135164

variable (x : Int) (y : Int) (m : Real)

theorem proof_problem :
  ((x = -6 ∧ y = 1 ∧ m = 7.5) ∨ (x = -1 ∧ y = 2 ∧ m = 4)) ↔
  (-2 * x + 3 * y = 2 * m ∧ x - 5 * y = -11 ∧ x < 0 ∧ y > 0)
:= sorry

end proof_problem_l135_135164


namespace weight_of_each_bag_of_food_l135_135534

theorem weight_of_each_bag_of_food
  (horses : ℕ)
  (feedings_per_day : ℕ)
  (pounds_per_feeding : ℕ)
  (days : ℕ)
  (bags : ℕ)
  (total_food_in_pounds : ℕ)
  (h1 : horses = 25)
  (h2 : feedings_per_day = 2)
  (h3 : pounds_per_feeding = 20)
  (h4 : days = 60)
  (h5 : bags = 60)
  (h6 : total_food_in_pounds = horses * (feedings_per_day * pounds_per_feeding) * days) :
  total_food_in_pounds / bags = 1000 :=
by
  sorry

end weight_of_each_bag_of_food_l135_135534


namespace dot_product_vec_a_vec_b_l135_135826

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 1)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem dot_product_vec_a_vec_b : dot_product vec_a vec_b = 1 := by
  sorry

end dot_product_vec_a_vec_b_l135_135826


namespace mark_total_payment_l135_135635

def total_cost (work_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  work_hours * hourly_rate + part_cost

theorem mark_total_payment :
  total_cost 2 75 150 = 300 :=
by
  -- Proof omitted, sorry used to skip the proof
  sorry

end mark_total_payment_l135_135635


namespace combined_weight_of_olivers_bags_l135_135698

theorem combined_weight_of_olivers_bags (w_james : ℕ) (w_oliver : ℕ) (w_combined : ℕ) 
  (h1 : w_james = 18) 
  (h2 : w_oliver = w_james / 6) 
  (h3 : w_combined = 2 * w_oliver) : 
  w_combined = 6 := 
by
  sorry

end combined_weight_of_olivers_bags_l135_135698


namespace profit_without_discount_l135_135314

theorem profit_without_discount (CP SP_original SP_discount : ℝ) (h1 : CP > 0) (h2 : SP_discount = CP * 1.14) (h3 : SP_discount = SP_original * 0.95) :
  (SP_original - CP) / CP * 100 = 20 :=
by
  have h4 : SP_original = SP_discount / 0.95 := by sorry
  have h5 : SP_original = CP * 1.2 := by sorry
  have h6 : (SP_original - CP) / CP * 100 = (CP * 1.2 - CP) / CP * 100 := by sorry
  have h7 : (SP_original - CP) / CP * 100 = 20 := by sorry
  exact h7

end profit_without_discount_l135_135314


namespace linoleum_cut_rearrange_l135_135700

def linoleum : Type := sorry -- placeholder for the specific type of the linoleum piece

def A : linoleum := sorry -- define piece A
def B : linoleum := sorry -- define piece B

def cut_and_rearrange (L : linoleum) (A B : linoleum) : Prop :=
  -- Define the proposition that pieces A and B can be rearranged into an 8x8 square
  sorry

theorem linoleum_cut_rearrange (L : linoleum) (A B : linoleum) :
  cut_and_rearrange L A B :=
sorry

end linoleum_cut_rearrange_l135_135700


namespace average_percentage_25_students_l135_135365

theorem average_percentage_25_students (s1 s2 : ℕ) (p1 p2 : ℕ) (n : ℕ)
  (h1 : s1 = 15) (h2 : p1 = 75) (h3 : s2 = 10) (h4 : p2 = 95) (h5 : n = 25) :
  ((s1 * p1 + s2 * p2) / n) = 83 := 
by
  sorry

end average_percentage_25_students_l135_135365


namespace roses_in_vase_l135_135109

/-- There were initially 16 roses and 3 orchids in the vase.
    Jessica cut 8 roses and 8 orchids from her garden.
    There are now 7 orchids in the vase.
    Prove that the number of roses in the vase now is 24. -/
theorem roses_in_vase
  (initial_roses initial_orchids : ℕ)
  (cut_roses cut_orchids remaining_orchids final_roses : ℕ)
  (h_initial: initial_roses = 16)
  (h_initial_orchids: initial_orchids = 3)
  (h_cut: cut_roses = 8 ∧ cut_orchids = 8)
  (h_remaining_orchids: remaining_orchids = 7)
  (h_orchids_relation: initial_orchids + cut_orchids = remaining_orchids + cut_orchids - 4)
  : final_roses = initial_roses + cut_roses := by
  sorry

end roses_in_vase_l135_135109


namespace bus_stops_duration_per_hour_l135_135110

def speed_without_stoppages : ℝ := 90
def speed_with_stoppages : ℝ := 84
def distance_covered_lost := speed_without_stoppages - speed_with_stoppages

theorem bus_stops_duration_per_hour :
  distance_covered_lost / speed_without_stoppages * 60 = 4 :=
by
  sorry

end bus_stops_duration_per_hour_l135_135110


namespace quilt_percentage_shaded_l135_135851

theorem quilt_percentage_shaded :
  ∀ (total_squares full_shaded half_shaded quarter_shaded : ℕ),
    total_squares = 25 →
    full_shaded = 4 →
    half_shaded = 8 →
    quarter_shaded = 4 →
    ((full_shaded + half_shaded * 1 / 2 + quarter_shaded * 1 / 2) / total_squares * 100 = 40) :=
by
  intros
  sorry

end quilt_percentage_shaded_l135_135851


namespace find_fourth_term_in_sequence_l135_135855

theorem find_fourth_term_in_sequence (x: ℤ) (h1: 86 - 8 = 78) (h2: 2 - 86 = -84) (h3: x - 2 = -90) (h4: -12 - x = 76):
  x = -88 :=
sorry

end find_fourth_term_in_sequence_l135_135855


namespace ratio_PR_QS_l135_135419

/-- Given points P, Q, R, and S on a straight line in that order with
    distances PQ = 3 units, QR = 7 units, and PS = 20 units,
    the ratio of PR to QS is 1. -/
theorem ratio_PR_QS (P Q R S : ℝ) (PQ QR PS : ℝ) (hPQ : PQ = 3) (hQR : QR = 7) (hPS : PS = 20) :
  let PR := PQ + QR
  let QS := PS - PQ - QR
  PR / QS = 1 :=
by
  -- Definitions from conditions
  let PR := PQ + QR
  let QS := PS - PQ - QR
  -- Proof not required, hence sorry
  sorry

end ratio_PR_QS_l135_135419


namespace sufficient_but_not_necessary_condition_l135_135845

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, 0 < x → x < 4 → x^2 - 4 * x < 0) ∧ ¬ (∀ x : ℝ, x^2 - 4 * x < 0 → 0 < x ∧ x < 5) :=
sorry

end sufficient_but_not_necessary_condition_l135_135845


namespace donny_total_spending_l135_135266

noncomputable def total_saving_mon : ℕ := 15
noncomputable def total_saving_tue : ℕ := 28
noncomputable def total_saving_wed : ℕ := 13
noncomputable def total_saving_fri : ℕ := 22

noncomputable def total_savings_mon_to_wed : ℕ := total_saving_mon + total_saving_tue + total_saving_wed
noncomputable def thursday_spending : ℕ := total_savings_mon_to_wed / 2
noncomputable def remaining_savings_after_thursday : ℕ := total_savings_mon_to_wed - thursday_spending
noncomputable def total_savings_before_sat : ℕ := remaining_savings_after_thursday + total_saving_fri
noncomputable def saturday_spending : ℕ := total_savings_before_sat * 40 / 100

theorem donny_total_spending : thursday_spending + saturday_spending = 48 := by sorry

end donny_total_spending_l135_135266


namespace B_values_for_divisibility_l135_135615

theorem B_values_for_divisibility (B : ℕ) (h : 4 + B + B + B + 2 ≡ 0 [MOD 9]) : B = 1 ∨ B = 4 ∨ B = 7 :=
by sorry

end B_values_for_divisibility_l135_135615


namespace geometric_sequence_properties_l135_135683

theorem geometric_sequence_properties (a : ℕ → ℝ) (q : ℝ) :
  a 1 = 1 / 2 ∧ a 4 = -4 → q = -2 ∧ (∀ n, a n = 1 / 2 * q ^ (n - 1)) :=
by
  intro h
  sorry

end geometric_sequence_properties_l135_135683


namespace numberOfRottweilers_l135_135935

-- Define the grooming times in minutes for each type of dog
def groomingTimeRottweiler := 20
def groomingTimeCollie := 10
def groomingTimeChihuahua := 45

-- Define the number of each type of dog groomed
def numberOfCollies := 9
def numberOfChihuahuas := 1

-- Define the total grooming time in minutes
def totalGroomingTime := 255

-- Compute the time spent on grooming Collies
def timeSpentOnCollies := numberOfCollies * groomingTimeCollie

-- Compute the time spent on grooming Chihuahuas
def timeSpentOnChihuahuas := numberOfChihuahuas * groomingTimeChihuahua

-- Compute the time spent on grooming Rottweilers
def timeSpentOnRottweilers := totalGroomingTime - timeSpentOnCollies - timeSpentOnChihuahuas

-- The main theorem statement
theorem numberOfRottweilers :
  timeSpentOnRottweilers / groomingTimeRottweiler = 6 :=
by
  -- Proof placeholder
  sorry

end numberOfRottweilers_l135_135935


namespace triangle_side_length_difference_l135_135080

theorem triangle_side_length_difference :
  (∃ x : ℤ, 3 ≤ x ∧ x ≤ 17 ∧ ∀ a b c : ℤ, x + 8 > 10 ∧ x + 10 > 8 ∧ 8 + 10 > x) →
  (17 - 3 = 14) :=
by
  intros
  sorry

end triangle_side_length_difference_l135_135080


namespace increasing_on_1_to_infty_min_value_on_1_to_e_l135_135453

noncomputable def f (x : ℝ) (a : ℝ) := x^2 - a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) := (2 * x^2 - a) / x

-- Proof that f(x) is increasing on (1, +∞) when a = 2
theorem increasing_on_1_to_infty (x : ℝ) (h : x > 1) : f' x 2 > 0 := 
  sorry

-- Proof for minimum value of f(x) on [1, e]
theorem min_value_on_1_to_e (a : ℝ) :
  if a ≤ 2 then ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = 1
  else if 2 < a ∧ a < 2 * Real.exp 2 then 
    ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = a / 2 - (a / 2) * Real.log (a / 2)
  else if a ≥ 2 * Real.exp 2 then 
    ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = Real.exp 2 - a
  else False := 
  sorry

end increasing_on_1_to_infty_min_value_on_1_to_e_l135_135453


namespace complement_intersection_l135_135190

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection :
  ((U \ A) ∩ B) = {0} :=
by
  sorry

end complement_intersection_l135_135190


namespace area_triangle_PCB_correct_l135_135670

noncomputable def area_of_triangle_PCB (ABCD : Type) (A B C D P : ABCD)
  (AB_parallel_CD : ∀ (l m : ABCD → ABCD → Prop), l B A = m D C)
  (diagonals_intersect_P : ∀ (a b c d : ABCD → ABCD → ABCD → Prop), a A C P = b B D P)
  (area_APB : ℝ) (area_CPD : ℝ) : ℝ :=
  6

theorem area_triangle_PCB_correct (ABCD : Type) (A B C D P : ABCD)
  (AB_parallel_CD : ∀ (l m : ABCD → ABCD → Prop), l B A = m D C)
  (diagonals_intersect_P : ∀ (a b c d : ABCD → ABCD → ABCD → Prop), a A C P = b B D P)
  (area_APB : ℝ) (area_CPD : ℝ) :
  area_APB = 4 ∧ area_CPD = 9 → area_of_triangle_PCB ABCD A B C D P AB_parallel_CD diagonals_intersect_P area_APB area_CPD = 6 :=
by
  sorry

end area_triangle_PCB_correct_l135_135670


namespace geometric_series_common_ratio_l135_135121

theorem geometric_series_common_ratio (r : ℚ) : 
  (∃ (a : ℚ), a = 4 / 7 ∧ a * r = 16 / 21) → r = 4 / 3 :=
by
  sorry

end geometric_series_common_ratio_l135_135121


namespace jose_is_21_l135_135441

-- Define the ages of the individuals based on the conditions
def age_of_inez := 12
def age_of_zack := age_of_inez + 4
def age_of_jose := age_of_zack + 5

-- State the proposition we want to prove
theorem jose_is_21 : age_of_jose = 21 := 
by 
  sorry

end jose_is_21_l135_135441


namespace solve_fractional_equation_l135_135777

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) : 
  (2 * x) / (x - 1) = x / (3 * (x - 1)) + 1 ↔ x = -3 / 2 :=
by sorry

end solve_fractional_equation_l135_135777


namespace largest_house_number_l135_135432

theorem largest_house_number (house_num : ℕ) : 
  house_num ≤ 981 :=
  sorry

end largest_house_number_l135_135432


namespace greatest_divisor_l135_135739

theorem greatest_divisor (n : ℕ) (h1 : 1657 % n = 6) (h2 : 2037 % n = 5) : n = 127 :=
by
  sorry

end greatest_divisor_l135_135739


namespace find_f3_l135_135272

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f3 (h1 : ∀ x : ℝ, f (x + 1) = f (-x - 1))
                (h2 : ∀ x : ℝ, f (2 - x) = -f x) :
  f 3 = 0 := 
sorry

end find_f3_l135_135272


namespace simplify_and_evaluate_l135_135512

noncomputable def x : ℕ := 2023
noncomputable def y : ℕ := 2

theorem simplify_and_evaluate :
  (x + 2 * y)^2 - ((x^3 + 4 * x^2 * y) / x) = 16 :=
by
  sorry

end simplify_and_evaluate_l135_135512


namespace rectangle_area_l135_135376

-- Conditions: 
-- 1. The length of the rectangle is three times its width.
-- 2. The diagonal length of the rectangle is x.

theorem rectangle_area (x : ℝ) (w l : ℝ) (h1 : w * 3 = l) (h2 : w^2 + l^2 = x^2) :
  l * w = (3 / 10) * x^2 :=
by
  sorry

end rectangle_area_l135_135376


namespace Oliver_total_workout_hours_l135_135315

-- Define the working hours for each day
def Monday_hours : ℕ := 4
def Tuesday_hours : ℕ := Monday_hours - 2
def Wednesday_hours : ℕ := 2 * Monday_hours
def Thursday_hours : ℕ := 2 * Tuesday_hours

-- Prove that the total hours Oliver worked out adds up to 18
theorem Oliver_total_workout_hours : Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours = 18 := by
  sorry

end Oliver_total_workout_hours_l135_135315


namespace no_divisibility_condition_by_all_others_l135_135404

theorem no_divisibility_condition_by_all_others 
  {p : ℕ → ℕ} 
  (h_distinct_odd_primes : ∀ i j, i ≠ j → Nat.Prime (p i) ∧ Nat.Prime (p j) ∧ p i ≠ p j ∧ p i % 2 = 1 ∧ p j % 2 = 1)
  (h_ordered : ∀ i j, i < j → p i < p j) :
  ¬ ∀ i j, i ≠ j → (∀ k ≠ i, k ≠ j → p k ∣ (p i ^ 8 - p j ^ 8)) :=
by
  sorry

end no_divisibility_condition_by_all_others_l135_135404


namespace right_triangle_hypotenuse_l135_135348

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l135_135348


namespace max_expression_value_l135_135533

theorem max_expression_value (a b c d e f g h k : ℤ)
  (ha : (a = 1 ∨ a = -1)) (hb : (b = 1 ∨ b = -1))
  (hc : (c = 1 ∨ c = -1)) (hd : (d = 1 ∨ d = -1))
  (he : (e = 1 ∨ e = -1)) (hf : (f = 1 ∨ f = -1))
  (hg : (g = 1 ∨ g = -1)) (hh : (h = 1 ∨ h = -1))
  (hk : (k = 1 ∨ k = -1)) :
  a * e * k - a * f * h + b * f * g - b * d * k + c * d * h - c * e * g ≤ 4 := sorry

end max_expression_value_l135_135533


namespace vector_projection_line_l135_135979

theorem vector_projection_line (v : ℝ × ℝ) 
  (h : ∃ (x y : ℝ), v = (x, y) ∧ 
       (3 * x + 4 * y) / (3 ^ 2 + 4 ^ 2) = 1) :
  ∃ (x y : ℝ), v = (x, y) ∧ y = -3 / 4 * x + 25 / 4 :=
by
  sorry

end vector_projection_line_l135_135979


namespace minimum_soldiers_to_add_l135_135136

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l135_135136


namespace ages_correct_l135_135295

-- Definitions of the given conditions
def john_age : ℕ := 42
def tim_age : ℕ := 79
def james_age : ℕ := 30
def lisa_age : ℚ := 54.5
def kate_age : ℕ := 34
def michael_age : ℚ := 61.5
def anna_age : ℚ := 54.5

-- Mathematically equivalent proof problem
theorem ages_correct :
  (james_age = 30) ∧
  (lisa_age = 54.5) ∧
  (kate_age = 34) ∧
  (michael_age = 61.5) ∧
  (anna_age = 54.5) :=
by {
  sorry  -- Proof to be filled in
}

end ages_correct_l135_135295


namespace largest_multiple_of_seven_smaller_than_neg_85_l135_135884

theorem largest_multiple_of_seven_smaller_than_neg_85 
  : ∃ k : ℤ, (k * 7 < -85) ∧ (∀ m : ℤ, (m * 7 < -85) → (m * 7 ≤ k * 7)) ∧ (k = -13) 
  := sorry

end largest_multiple_of_seven_smaller_than_neg_85_l135_135884


namespace average_value_f_l135_135424

def f (x : ℝ) : ℝ := (1 + x)^3

theorem average_value_f : (1 / (4 - 2)) * (∫ x in (2:ℝ)..(4:ℝ), f x) = 68 :=
by
  sorry

end average_value_f_l135_135424


namespace find_d_value_l135_135079

open Nat

variable {PA BC PB : ℕ}
noncomputable def d (PA BC PB : ℕ) := PB

theorem find_d_value (h₁ : PA = 6) (h₂ : BC = 9) (h₃ : PB = d PA BC PB) : d PA BC PB = 3 := by
  sorry

end find_d_value_l135_135079


namespace number_of_adult_tickets_l135_135474

-- Define the parameters of the problem
def price_adult_ticket : ℝ := 5.50
def price_child_ticket : ℝ := 3.50
def total_tickets : ℕ := 21
def total_cost : ℝ := 83.50

-- Define the main theorem to be proven
theorem number_of_adult_tickets : 
  ∃ (A C : ℕ), A + C = total_tickets ∧ 
                (price_adult_ticket * A + price_child_ticket * C = total_cost) ∧ 
                 A = 5 :=
by
  -- The proof content will be filled in later
  sorry

end number_of_adult_tickets_l135_135474


namespace correct_relation_is_identity_l135_135140

theorem correct_relation_is_identity : 0 = 0 :=
by {
  -- Skipping proof steps as only statement is required
  sorry
}

end correct_relation_is_identity_l135_135140


namespace sum_of_solutions_l135_135175

theorem sum_of_solutions (S : Finset ℝ) (h : ∀ x ∈ S, |x^2 - 10 * x + 29| = 3) : S.sum id = 0 :=
sorry

end sum_of_solutions_l135_135175


namespace can_be_divided_into_6_triangles_l135_135378

-- Define the initial rectangle dimensions
def initial_rectangle_length := 6
def initial_rectangle_width := 5

-- Define the cut-out rectangle dimensions
def cutout_rectangle_length := 2
def cutout_rectangle_width := 1

-- Total area before the cut-out
def total_area : Nat := initial_rectangle_length * initial_rectangle_width

-- Cut-out area
def cutout_area : Nat := cutout_rectangle_length * cutout_rectangle_width

-- Remaining area after the cut-out
def remaining_area : Nat := total_area - cutout_area

-- The statement to be proved
theorem can_be_divided_into_6_triangles :
  remaining_area = 28 → (∃ (triangles : List (Nat × Nat × Nat)), triangles.length = 6) :=
by 
  intros h
  sorry

end can_be_divided_into_6_triangles_l135_135378


namespace Nellie_legos_l135_135054

def initial_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_legos : ℕ := 24

def remaining_legos : ℕ := initial_legos - lost_legos - given_legos

theorem Nellie_legos : remaining_legos = 299 := by
  sorry

end Nellie_legos_l135_135054


namespace find_principal_l135_135715

-- Problem conditions
variables (SI : ℚ := 4016.25) 
variables (R : ℚ := 0.08) 
variables (T : ℚ := 5)

-- The simple interest formula to find Principal
noncomputable def principal (SI : ℚ) (R : ℚ) (T : ℚ) : ℚ := SI * 100 / (R * T)

-- Lean statement to prove
theorem find_principal : principal SI R T = 10040.625 := by
  sorry

end find_principal_l135_135715


namespace fractional_part_lawn_remainder_l135_135233

def mary_mowing_time := 3 -- Mary can mow the lawn in 3 hours
def tom_mowing_time := 6  -- Tom can mow the lawn in 6 hours
def mary_working_hours := 1 -- Mary works for 1 hour alone

theorem fractional_part_lawn_remainder : 
  (1 - mary_working_hours / mary_mowing_time) = 2 / 3 := 
by
  sorry

end fractional_part_lawn_remainder_l135_135233


namespace max_value_of_quadratic_l135_135656

theorem max_value_of_quadratic :
  ∃ y : ℚ, ∀ x : ℚ, -x^2 - 3 * x + 4 ≤ y :=
sorry

end max_value_of_quadratic_l135_135656


namespace triangle_perimeter_l135_135780

/-- Given the lengths of two sides of a triangle are 1 and 4,
    and the length of the third side is an integer, 
    prove that the perimeter of the triangle is 9 -/
theorem triangle_perimeter
  (a b : ℕ)
  (c : ℤ)
  (h₁ : a = 1)
  (h₂ : b = 4)
  (h₃ : 3 < c ∧ c < 5) :
  a + b + c = 9 :=
by sorry

end triangle_perimeter_l135_135780


namespace paper_cranes_l135_135313

theorem paper_cranes (B C A : ℕ) (h1 : A + B + C = 1000)
  (h2 : A = 3 * B - 100)
  (h3 : C = A - 67) : A = 443 := by
  sorry

end paper_cranes_l135_135313


namespace problem4_l135_135711

theorem problem4 (a : ℝ) : (a-1)^2 = a^3 - 2*a + 1 ↔ a = 0 ∨ a = 1 := 
by sorry

end problem4_l135_135711


namespace find_n_l135_135142

theorem find_n (x : ℝ) (h1 : x = 4.0) (h2 : 3 * x + n = 48) : n = 36 := by
  sorry

end find_n_l135_135142


namespace find_x_l135_135786

theorem find_x (x : ℝ) (h : (2 * x) / 16 = 25) : x = 200 :=
sorry

end find_x_l135_135786


namespace sandy_correct_sums_l135_135146

/-- 
Sandy gets 3 marks for each correct sum and loses 2 marks for each incorrect sum.
Sandy attempts 50 sums and obtains 100 marks within a 45-minute time constraint.
If Sandy receives a 1-mark penalty for each sum not completed within the time limit,
prove that the number of correct sums Sandy got is 25.
-/
theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 50) (h2 : 3 * c - 2 * i - (50 - c) = 100) : c = 25 :=
by
  sorry

end sandy_correct_sums_l135_135146


namespace general_term_sequence_l135_135750

theorem general_term_sequence (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2)
  (h2 : ∀ (m : ℕ), m ≥ 2 → a m - a (m - 1) + 1 = 0) : 
  a n = 3 - n :=
sorry

end general_term_sequence_l135_135750


namespace problem1_l135_135880

theorem problem1 (x : ℝ) : abs (2 * x - 3) < 1 ↔ 1 < x ∧ x < 2 := sorry

end problem1_l135_135880


namespace object_speed_approx_l135_135837

theorem object_speed_approx :
  ∃ (speed : ℝ), abs (speed - 27.27) < 0.01 ∧
  (∀ (d : ℝ) (t : ℝ)
    (m : ℝ), 
    d = 80 ∧ t = 2 ∧ m = 5280 →
    speed = (d / m) / (t / 3600)) :=
by 
  sorry

end object_speed_approx_l135_135837


namespace rented_apartment_years_l135_135881

-- Given conditions
def months_in_year := 12
def payment_first_3_years_per_month := 300
def payment_remaining_years_per_month := 350
def total_paid := 19200
def first_period_years := 3

-- Define the total payment calculation
def total_payment (additional_years: ℕ): ℕ :=
  (first_period_years * months_in_year * payment_first_3_years_per_month) + 
  (additional_years * months_in_year * payment_remaining_years_per_month)

-- Main theorem statement
theorem rented_apartment_years (additional_years: ℕ) :
  total_payment additional_years = total_paid → (first_period_years + additional_years) = 5 :=
by
  intros h
  -- This skips the proof
  sorry

end rented_apartment_years_l135_135881


namespace LindasTrip_l135_135812

theorem LindasTrip (x : ℝ) :
    (1 / 4) * x + 30 + (1 / 6) * x = x →
    x = 360 / 7 :=
by
  intros h
  sorry

end LindasTrip_l135_135812


namespace aria_spent_on_cookies_in_march_l135_135207

/-- Aria purchased 4 cookies each day for the entire month of March,
    each cookie costs 19 dollars, and March has 31 days.
    Prove that the total amount Aria spent on cookies in March is 2356 dollars. -/
theorem aria_spent_on_cookies_in_march :
  (4 * 31) * 19 = 2356 := 
by 
  sorry

end aria_spent_on_cookies_in_march_l135_135207


namespace shaded_rectangle_ratio_l135_135626

variable (a : ℝ) (h : 0 < a)  -- side length of the square is 'a' and it is positive

theorem shaded_rectangle_ratio :
  (∃ l w : ℝ, (l = a / 2 ∧ w = a / 3 ∧ (l * w = a^2 / 6) ∧ (a^2 / 6 = a * a / 6))) → (l / w = 1.5) :=
by {
  -- Proof is to be provided
  sorry
}

end shaded_rectangle_ratio_l135_135626


namespace apr_sales_is_75_l135_135685

-- Definitions based on conditions
def sales_jan : ℕ := 90
def sales_feb : ℕ := 50
def sales_mar : ℕ := 70
def avg_sales : ℕ := 72

-- Total sales of first three months
def total_sales_jan_to_mar : ℕ := sales_jan + sales_feb + sales_mar

-- Total sales considering average sales over 5 months
def total_sales : ℕ := avg_sales * 5

-- Defining April sales
def sales_apr (sales_may : ℕ) : ℕ := total_sales - total_sales_jan_to_mar - sales_may

theorem apr_sales_is_75 (sales_may : ℕ) : sales_apr sales_may = 75 :=
by
  unfold sales_apr total_sales total_sales_jan_to_mar avg_sales sales_jan sales_feb sales_mar
  -- Here we could insert more steps if needed to directly connect to the proof
  sorry


end apr_sales_is_75_l135_135685


namespace draw_at_least_one_even_ball_l135_135816

theorem draw_at_least_one_even_ball:
  -- Let the total number of ordered draws of 4 balls from 15 balls
  let total_draws := 15 * 14 * 13 * 12
  -- Let the total number of ordered draws of 4 balls where all balls are odd (balls 1, 3, ..., 15)
  let odd_draws := 8 * 7 * 6 * 5
  -- The number of valid draws containing at least one even ball
  total_draws - odd_draws = 31080 :=
by
  sorry

end draw_at_least_one_even_ball_l135_135816


namespace connie_s_problem_l135_135362

theorem connie_s_problem (y : ℕ) (h : 3 * y = 90) : y / 3 = 10 :=
by
  sorry

end connie_s_problem_l135_135362


namespace equal_playing_time_l135_135894

-- Given conditions
def total_minutes : Nat := 120
def number_of_children : Nat := 6
def children_playing_at_a_time : Nat := 2

-- Proof problem statement
theorem equal_playing_time :
  (children_playing_at_a_time * total_minutes) / number_of_children = 40 :=
by
  sorry

end equal_playing_time_l135_135894


namespace line_segment_no_intersection_l135_135895

theorem line_segment_no_intersection (a : ℝ) :
  (¬ ∃ t : ℝ, (0 ≤ t ∧ t ≤ 1 ∧ (1 - t) * (3 : ℝ) + t * (1 : ℝ) = 2 ∧ (1 - t) * (1 : ℝ) + t * (2 : ℝ) = (2 - (1 - t) * (3 : ℝ)) / a)) ->
  (a < -1 ∨ a > 0.5) :=
by
  sorry

end line_segment_no_intersection_l135_135895


namespace find_three_numbers_l135_135603

theorem find_three_numbers (x : ℤ) (a b c : ℤ) :
  a + b + c = (x + 1)^2 ∧ a + b = x^2 ∧ b + c = (x - 1)^2 ∧
  a = 80 ∧ b = 320 ∧ c = 41 :=
by {
  sorry
}

end find_three_numbers_l135_135603


namespace required_run_rate_l135_135005

theorem required_run_rate
  (run_rate_first_10_overs : ℝ)
  (target_runs : ℝ)
  (overs_first : ℕ)
  (overs_remaining : ℕ)
  (H_run_rate_10_overs : run_rate_first_10_overs = 3.2)
  (H_target_runs : target_runs = 222)
  (H_overs_first : overs_first = 10)
  (H_overs_remaining : overs_remaining = 40) :
  ((target_runs - run_rate_first_10_overs * overs_first) / overs_remaining) = 4.75 := 
by
  sorry

end required_run_rate_l135_135005


namespace value_of_n_l135_135663

-- Define required conditions
variables (n : ℕ) (f : ℕ → ℕ → ℕ)

-- Conditions
axiom cond1 : n > 7
axiom cond2 : ∀ m k : ℕ, f m k = 2^(n - m) * Nat.choose m k

-- Given condition
axiom after_seventh_round : f 7 5 = 42

-- Theorem to prove
theorem value_of_n : n = 8 :=
by
  -- Proof goes here
  sorry

end value_of_n_l135_135663


namespace sufficient_not_necessary_condition_l135_135170

theorem sufficient_not_necessary_condition (a b : ℝ) (h1 : a > 1) (h2 : b > 2) : a + b > 3 :=
by
  sorry

end sufficient_not_necessary_condition_l135_135170


namespace initial_students_proof_l135_135572

def initial_students (e : ℝ) (transferred : ℝ) (left : ℝ) : ℝ :=
  e + transferred + left

theorem initial_students_proof : initial_students 28 10 4 = 42 :=
  by
    -- This is where the proof would go, but we use 'sorry' to skip it.
    sorry

end initial_students_proof_l135_135572


namespace Mark_has_23_kangaroos_l135_135382

theorem Mark_has_23_kangaroos :
  ∃ K G : ℕ, G = 3 * K ∧ 2 * K + 4 * G = 322 ∧ K = 23 :=
by
  sorry

end Mark_has_23_kangaroos_l135_135382


namespace M_value_l135_135773

noncomputable def x : ℝ := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2)

noncomputable def y : ℝ := Real.sqrt (4 - 2 * Real.sqrt 3)

noncomputable def M : ℝ := x - y

theorem M_value :
  M = (5 / 2) * Real.sqrt 2 - Real.sqrt 3 + (3 / 2) :=
sorry

end M_value_l135_135773


namespace part1_part2_l135_135427

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | 1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def H (a : ℝ) : Set ℝ := {x | abs (x - a) <= 2}

def symdiff (A B : Set ℝ) : Set ℝ := A ∩ (U \ B)

theorem part1 :
  symdiff M N = {x | 1 < x ∧ x < 2} ∧
  symdiff N M = {x | 3 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem part2 (a : ℝ) :
  symdiff (symdiff N M) (H a) =
    if a ≥ 4 ∨ a ≤ -1 then {x | 1 < x ∧ x < 2}
    else if 3 < a ∧ a < 4 then {x | 1 < x ∧ x < a - 2}
    else if -1 < a ∧ a < 0 then {x | a + 2 < x ∧ x < 2}
    else ∅ :=
by
  sorry

end part1_part2_l135_135427


namespace equivalent_proposition_l135_135150

theorem equivalent_proposition (H : Prop) (P : Prop) (Q : Prop) (hpq : H → P → ¬ Q) : (H → ¬ Q → ¬ P) :=
by
  intro h nq np
  sorry

end equivalent_proposition_l135_135150


namespace compound_interest_principal_l135_135809

theorem compound_interest_principal 
    (CI : Real)
    (r : Real)
    (n : Nat)
    (t : Nat)
    (A : Real)
    (P : Real) :
  CI = 945.0000000000009 →
  r = 0.10 →
  n = 1 →
  t = 2 →
  A = P * (1 + r / n) ^ (n * t) →
  CI = A - P →
  P = 4500.0000000000045 :=
by intros
   sorry

end compound_interest_principal_l135_135809


namespace find_m_l135_135514

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def point_on_x_axis_distance (x y : ℝ) : Prop :=
  y = 14

def point_distance_from_fixed_point (x y : ℝ) : Prop :=
  distance (x, y) (3, 8) = 8

def x_coordinate_condition (x : ℝ) : Prop :=
  x > 3

def m_distance (x y m : ℝ) : Prop :=
  distance (x, y) (0, 0) = m

theorem find_m (x y m : ℝ) 
  (h1 : point_on_x_axis_distance x y) 
  (h2 : point_distance_from_fixed_point x y) 
  (h3 : x_coordinate_condition x) :
  m_distance x y m → 
  m = Real.sqrt (233 + 12 * Real.sqrt 7) := by
  sorry

end find_m_l135_135514


namespace solve_equation_l135_135452

theorem solve_equation :
  ∃ x : ℝ, (20 / (x^2 - 9) - 3 / (x + 3) = 1) ↔ (x = -8) ∨ (x = 5) :=
by
  sorry

end solve_equation_l135_135452


namespace ideal_sleep_hours_l135_135066

open Nat

theorem ideal_sleep_hours 
  (weeknight_sleep : Nat)
  (weekend_sleep : Nat)
  (sleep_deficit : Nat)
  (num_weeknights : Nat := 5)
  (num_weekend_nights : Nat := 2)
  (total_nights : Nat := 7) :
  weeknight_sleep = 5 →
  weekend_sleep = 6 →
  sleep_deficit = 19 →
  ((num_weeknights * weeknight_sleep + num_weekend_nights * weekend_sleep) + sleep_deficit) / total_nights = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end ideal_sleep_hours_l135_135066


namespace student_marks_l135_135965

variable (M P C : ℕ)

theorem student_marks (h1 : C = P + 20) (h2 : (M + C) / 2 = 20) : M + P = 20 :=
by
  sorry

end student_marks_l135_135965


namespace smallest_area_of_ellipse_l135_135911

theorem smallest_area_of_ellipse 
    (a b : ℝ)
    (h1 : ∀ x y, (x - 2)^2 + y^2 < 4 → (x / a)^2 + (y / b)^2 < 1)
    (h2 : ∀ x y, (x + 2)^2 + y^2 < 4 → (x / a)^2 + (y / b)^2 < 1) :
    π * a * b = π :=
sorry

end smallest_area_of_ellipse_l135_135911


namespace part1_part2_l135_135381

noncomputable def f (a : ℝ) (a_pos : a > 1) (x : ℝ) : ℝ :=
  a^x + (x - 2) / (x + 1)

-- Statement for part 1
theorem part1 (a : ℝ) (a_pos : a > 1) : ∀ x : ℝ, -1 < x → f a a_pos x ≤ f a a_pos (x + ε) → 0 < ε := sorry

-- Statement for part 2
theorem part2 (a : ℝ) (a_pos : a > 1) : ¬ ∃ x : ℝ, x < 0 ∧ f a a_pos x = 0 := sorry

end part1_part2_l135_135381


namespace remainder_2_pow_2015_mod_20_l135_135794

/-- 
  Given that powers of 2 modulo 20 follow a repeating cycle every 4 terms:
  2, 4, 8, 16, 12
  
  Prove that the remainder when \(2^{2015}\) is divided by 20 is 8.
-/
theorem remainder_2_pow_2015_mod_20 : (2 ^ 2015) % 20 = 8 :=
by
  -- The proof is to be filled in.
  sorry

end remainder_2_pow_2015_mod_20_l135_135794


namespace evaluate_fraction_l135_135918

theorem evaluate_fraction : 
  (7/3) / (8/15) = 35/8 :=
by
  -- we don't need to provide the proof as per instructions
  sorry

end evaluate_fraction_l135_135918


namespace fraction_value_l135_135634

theorem fraction_value
  (a b c d : ℚ)
  (h1 : a / b = 1 / 4)
  (h2 : c / d = 1 / 4)
  (h3 : b ≠ 0)
  (h4 : d ≠ 0)
  (h5 : b + d ≠ 0) :
  (a + 2 * c) / (2 * b + 4 * d) = 1 / 8 :=
sorry

end fraction_value_l135_135634


namespace find_integer_k_l135_135821

theorem find_integer_k (k x : ℤ) (h : (k^2 - 1) * x^2 - 6 * (3 * k - 1) * x + 72 = 0) (hx : x > 0) :
  k = 1 ∨ k = 2 ∨ k = 3 :=
sorry

end find_integer_k_l135_135821


namespace car_more_miles_per_tank_after_modification_l135_135490

theorem car_more_miles_per_tank_after_modification (mpg_old : ℕ) (efficiency_factor : ℝ) (gallons : ℕ) :
  mpg_old = 33 →
  efficiency_factor = 1.25 →
  gallons = 16 →
  (efficiency_factor * mpg_old * gallons - mpg_old * gallons) = 132 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry  -- Proof omitted

end car_more_miles_per_tank_after_modification_l135_135490


namespace value_of_b_l135_135132

variable (a b c y1 y2 : ℝ)

def equation1 := (y1 = 4 * a + 2 * b + c)
def equation2 := (y2 = 4 * a - 2 * b + c)
def difference := (y1 - y2 = 8)

theorem value_of_b 
  (h1 : equation1 a b c y1)
  (h2 : equation2 a b c y2)
  (h3 : difference y1 y2) : 
  b = 2 := 
by 
  sorry

end value_of_b_l135_135132


namespace raspberry_pie_degrees_l135_135268

def total_students : ℕ := 48
def chocolate_preference : ℕ := 18
def apple_preference : ℕ := 10
def blueberry_preference : ℕ := 8
def remaining_students : ℕ := total_students - chocolate_preference - apple_preference - blueberry_preference
def raspberry_preference : ℕ := remaining_students / 2
def pie_chart_degrees : ℕ := (raspberry_preference * 360) / total_students

theorem raspberry_pie_degrees :
  pie_chart_degrees = 45 := by
  sorry

end raspberry_pie_degrees_l135_135268


namespace art_of_passing_through_walls_l135_135808

theorem art_of_passing_through_walls (n : ℕ) :
  (2 * Real.sqrt (2 / 3) = Real.sqrt (2 * (2 / 3))) ∧
  (3 * Real.sqrt (3 / 8) = Real.sqrt (3 * (3 / 8))) ∧
  (4 * Real.sqrt (4 / 15) = Real.sqrt (4 * (4 / 15))) ∧
  (5 * Real.sqrt (5 / 24) = Real.sqrt (5 * (5 / 24))) →
  8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n)) →
  n = 63 :=
by
  sorry

end art_of_passing_through_walls_l135_135808


namespace circle_area_ratio_l135_135206

theorem circle_area_ratio (r R : ℝ) (h : π * R^2 - π * r^2 = (3/4) * π * r^2) :
  R / r = Real.sqrt 7 / 2 :=
by
  sorry

end circle_area_ratio_l135_135206


namespace factorial_ends_with_base_8_zeroes_l135_135300

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l135_135300


namespace smaller_number_between_5_and_8_l135_135623

theorem smaller_number_between_5_and_8 :
  min 5 8 = 5 :=
by
  sorry

end smaller_number_between_5_and_8_l135_135623


namespace net_progress_l135_135161

-- Definitions based on conditions in the problem
def loss := 5
def gain := 9

-- Theorem: Proving the team's net progress
theorem net_progress : (gain - loss) = 4 :=
by
  -- Placeholder for proof
  sorry

end net_progress_l135_135161


namespace geometric_progression_exists_l135_135305

theorem geometric_progression_exists :
  ∃ (b_1 b_2 b_3 b_4 q : ℚ), 
    b_1 - b_2 = 35 ∧ 
    b_3 - b_4 = 560 ∧ 
    b_2 = b_1 * q ∧ 
    b_3 = b_1 * q^2 ∧ 
    b_4 = b_1 * q^3 ∧ 
    ((b_1 = 7 ∧ q = -4 ∧ b_2 = -28 ∧ b_3 = 112 ∧ b_4 = -448) ∨ 
    (b_1 = -35/3 ∧ q = 4 ∧ b_2 = -140/3 ∧ b_3 = -560/3 ∧ b_4 = -2240/3)) :=
by
  sorry

end geometric_progression_exists_l135_135305


namespace fill_boxes_l135_135239

theorem fill_boxes (a b c d e f g : ℤ) 
  (h1 : a + (-1) + 2 = 4)
  (h2 : 2 + 1 + b = 3)
  (h3 : c + (-4) + (-3) = -2)
  (h4 : b - 5 - 4 = -9)
  (h5 : f = d - 3)
  (h6 : g = d + 3)
  (h7 : -8 = 4 + 3 - 9 - 2 + (d - 3) + (d + 3)) : 
  a = 3 ∧ b = 0 ∧ c = 5 ∧ d = -2 ∧ e = -9 ∧ f = -5 ∧ g = 1 :=
by {
  sorry
}

end fill_boxes_l135_135239


namespace smallest_period_of_f_is_pi_div_2_l135_135228

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 4 + (Real.sin x) ^ 2

theorem smallest_period_of_f_is_pi_div_2 : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi / 2 :=
sorry

end smallest_period_of_f_is_pi_div_2_l135_135228


namespace determine_Q_l135_135964

def P : Set ℕ := {1, 2}

def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem determine_Q : Q = {2, 3, 4} :=
by
  sorry

end determine_Q_l135_135964


namespace square_area_l135_135830

def edge1 (x : ℝ) := 5 * x - 18
def edge2 (x : ℝ) := 27 - 4 * x
def x_val : ℝ := 5

theorem square_area : edge1 x_val = edge2 x_val → (edge1 x_val) ^ 2 = 49 :=
by
  intro h
  -- Proof required here
  sorry

end square_area_l135_135830


namespace rectangle_area_is_30_l135_135368

def Point := (ℤ × ℤ)

def vertices : List Point := [(-5, 1), (1, 1), (1, -4), (-5, -4)]

theorem rectangle_area_is_30 :
  let length := (vertices[1].1 - vertices[0].1).natAbs
  let width := (vertices[0].2 - vertices[2].2).natAbs
  length * width = 30 := by
  sorry

end rectangle_area_is_30_l135_135368


namespace find_some_number_l135_135776

theorem find_some_number : 
  ∃ (some_number : ℝ), (∃ (n : ℝ), n = 54 ∧ (n / some_number) * (n / 162) = 1) → some_number = 18 :=
by
  sorry

end find_some_number_l135_135776


namespace solution1_solution2_l135_135644

noncomputable def Problem1 : ℝ :=
  4 + (-2)^3 * 5 - (-0.28) / 4

theorem solution1 : Problem1 = -35.93 := by
  sorry

noncomputable def Problem2 : ℚ :=
  -1^4 - (1/6) * (2 - (-3)^2)

theorem solution2 : Problem2 = 1/6 := by
  sorry

end solution1_solution2_l135_135644


namespace unique_elements_set_l135_135822

theorem unique_elements_set (x : ℝ) : x ≠ 3 ∧ x ≠ -1 ∧ x ≠ 0 ↔ 3 ≠ x ∧ x ≠ (x ^ 2 - 2 * x) ∧ (x ^ 2 - 2 * x) ≠ 3 := by
  sorry

end unique_elements_set_l135_135822


namespace necessary_and_sufficient_l135_135682

variable (α β : ℝ)
variable (p : Prop := α > β)
variable (q : Prop := α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α)

theorem necessary_and_sufficient : (p ↔ q) :=
by
  sorry

end necessary_and_sufficient_l135_135682


namespace y_relationship_l135_135244

theorem y_relationship :
  ∀ (y1 y2 y3 : ℝ), 
  (y1 = (-2)^2 - 4*(-2) - 3) ∧ 
  (y2 = 1^2 - 4*1 - 3) ∧ 
  (y3 = 4^2 - 4*4 - 3) → 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end y_relationship_l135_135244


namespace cheenu_speed_difference_l135_135084

theorem cheenu_speed_difference :
  let cycling_time := 120 -- minutes
  let cycling_distance := 24 -- miles
  let jogging_time := 180 -- minutes
  let jogging_distance := 18 -- miles
  let cycling_speed := cycling_time / cycling_distance -- minutes per mile
  let jogging_speed := jogging_time / jogging_distance -- minutes per mile
  let speed_difference := jogging_speed - cycling_speed -- minutes per mile
  speed_difference = 5 := by sorry

end cheenu_speed_difference_l135_135084


namespace sticks_left_in_yard_l135_135116

def number_of_sticks_picked_up : Nat := 14
def difference_between_picked_and_left : Nat := 10

theorem sticks_left_in_yard 
  (picked_up : Nat := number_of_sticks_picked_up)
  (difference : Nat := difference_between_picked_and_left) 
  : Nat :=
  picked_up - difference

example : sticks_left_in_yard = 4 := by 
  sorry

end sticks_left_in_yard_l135_135116


namespace six_digit_number_condition_l135_135178

theorem six_digit_number_condition (a b c : ℕ) (h : 1 ≤ a ∧ a ≤ 9) (hb : b < 10) (hc : c < 10) : 
  ∃ k : ℕ, 100000 * a + 10000 * b + 1000 * c + 100 * (2 * a) + 10 * (2 * b) + 2 * c = 2 * k := 
by
  sorry

end six_digit_number_condition_l135_135178


namespace remaining_credit_l135_135125

noncomputable def initial_balance : ℝ := 30
noncomputable def call_rate : ℝ := 0.16
noncomputable def call_duration : ℝ := 22

theorem remaining_credit : initial_balance - (call_rate * call_duration) = 26.48 :=
by
  -- Definitions for readability
  let total_cost := call_rate * call_duration
  let remaining_balance := initial_balance - total_cost
  have h : total_cost = 3.52 := sorry
  have h₂ : remaining_balance = 26.48 := sorry
  exact h₂

end remaining_credit_l135_135125


namespace possible_marks_l135_135439

theorem possible_marks (n : ℕ) : n = 3 ∨ n = 6 ↔
  ∃ (m : ℕ), n = (m * (m - 1)) / 2 ∧ (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → ∃ (i j : ℕ), i < j ∧ j - i = k ∧ (∀ (x y : ℕ), x < y → x ≠ i ∨ y ≠ j)) :=
by sorry

end possible_marks_l135_135439


namespace lucca_bread_fraction_l135_135599

theorem lucca_bread_fraction 
  (total_bread : ℕ)
  (initial_fraction_eaten : ℚ)
  (final_pieces : ℕ)
  (bread_first_day : ℚ)
  (bread_second_day : ℚ)
  (bread_third_day : ℚ)
  (remaining_pieces_after_first_day : ℕ)
  (remaining_pieces_after_second_day : ℕ)
  (remaining_pieces_after_third_day : ℕ) :
  total_bread = 200 →
  initial_fraction_eaten = 1/4 →
  bread_first_day = initial_fraction_eaten * total_bread →
  remaining_pieces_after_first_day = total_bread - bread_first_day →
  bread_second_day = (remaining_pieces_after_first_day * bread_second_day) →
  remaining_pieces_after_second_day = remaining_pieces_after_first_day - bread_second_day →
  bread_third_day = 1/2 * remaining_pieces_after_second_day →
  remaining_pieces_after_third_day = remaining_pieces_after_second_day - bread_third_day →
  remaining_pieces_after_third_day = 45 →
  bread_second_day = 2/5 :=
by
  sorry

end lucca_bread_fraction_l135_135599


namespace expected_value_of_winnings_is_4_l135_135099

noncomputable def expected_value_of_winnings : ℕ := 
  let outcomes := [7, 6, 5, 4, 4, 3, 2, 1]
  let total_winnings := outcomes.sum
  total_winnings / 8

theorem expected_value_of_winnings_is_4 :
  expected_value_of_winnings = 4 :=
by
  sorry

end expected_value_of_winnings_is_4_l135_135099


namespace sharmila_hourly_wage_l135_135793

-- Sharmila works 10 hours per day on Monday, Wednesday, and Friday.
def hours_worked_mwf : ℕ := 3 * 10

-- Sharmila works 8 hours per day on Tuesday and Thursday.
def hours_worked_tt : ℕ := 2 * 8

-- Total hours worked in a week.
def total_hours_worked : ℕ := hours_worked_mwf + hours_worked_tt

-- Sharmila earns $460 per week.
def weekly_earnings : ℕ := 460

-- Calculate and prove her hourly wage is $10 per hour.
theorem sharmila_hourly_wage : (weekly_earnings / total_hours_worked) = 10 :=
by sorry

end sharmila_hourly_wage_l135_135793


namespace sufficient_but_not_necessary_condition_for_hyperbola_l135_135975

theorem sufficient_but_not_necessary_condition_for_hyperbola (k : ℝ) :
  (∃ k : ℝ, k > 3 ∧ (∃ x y : ℝ, (x^2) / (k - 3) - (y^2) / (k + 3) = 1)) ∧ 
  (∃ k : ℝ, k < -3 ∧ (∃ x y : ℝ, (x^2) / (k - 3) - (y^2) / (k + 3) = 1)) :=
    sorry

end sufficient_but_not_necessary_condition_for_hyperbola_l135_135975


namespace three_dice_probability_even_l135_135036

/-- A die is represented by numbers from 1 to 6. -/
def die := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

/-- Define an event where three dice are thrown, and we check if their sum is even. -/
def three_dice_sum_even (d1 d2 d3 : die) : Prop :=
  (d1.val + d2.val + d3.val) % 2 = 0

/-- Define the probability that a single die shows an odd number. -/
def prob_odd := 1 / 2

/-- Define the probability that a single die shows an even number. -/
def prob_even := 1 / 2

/-- Define the total probability for the sum of three dice to be even. -/
def prob_sum_even : ℚ :=
  prob_even ^ 3 + (3 * prob_odd ^ 2 * prob_even)

theorem three_dice_probability_even :
  prob_sum_even = 1 / 2 :=
by
  sorry

end three_dice_probability_even_l135_135036


namespace jess_double_cards_l135_135476

theorem jess_double_cards (rob_total_cards jess_doubles : ℕ) 
    (one_third_rob_cards_doubles : rob_total_cards / 3 = rob_total_cards / 3)
    (jess_times_rob_doubles : jess_doubles = 5 * (rob_total_cards / 3)) :
    rob_total_cards = 24 → jess_doubles = 40 :=
  by
  sorry

end jess_double_cards_l135_135476


namespace turtles_received_l135_135013

theorem turtles_received (martha_turtles : ℕ) (marion_turtles : ℕ) (h1 : martha_turtles = 40) 
    (h2 : marion_turtles = martha_turtles + 20) : martha_turtles + marion_turtles = 100 := 
by {
    sorry
}

end turtles_received_l135_135013


namespace problem_solution_l135_135327

theorem problem_solution (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 14) (h2 : a = b + c) : ab - bc + ac = 7 :=
  sorry

end problem_solution_l135_135327


namespace minimum_degree_g_l135_135438

-- Define the degree function for polynomials
noncomputable def degree (p : Polynomial ℤ) : ℕ := p.natDegree

-- Declare the variables and conditions for the proof
variables (f g h : Polynomial ℤ)
variables (deg_f : degree f = 10) (deg_h : degree h = 12)
variable (eqn : 2 * f + 5 * g = h)

-- State the main theorem for the problem
theorem minimum_degree_g : degree g ≥ 12 :=
    by sorry -- Proof to be provided

end minimum_degree_g_l135_135438


namespace kiwi_count_l135_135323

theorem kiwi_count (o a b k : ℕ) (h1 : o + a + b + k = 540) (h2 : a = 3 * o) (h3 : b = 4 * a) (h4 : k = 5 * b) : k = 420 :=
sorry

end kiwi_count_l135_135323


namespace viola_final_jump_l135_135470

variable (n : ℕ) (T : ℝ) (x : ℝ)

theorem viola_final_jump (h1 : T = 3.80 * n)
                        (h2 : (T + 3.99) / (n + 1) = 3.81)
                        (h3 : T + 3.99 + x = 3.82 * (n + 2)) : 
                        x = 4.01 :=
sorry

end viola_final_jump_l135_135470


namespace quadratic_has_real_root_l135_135310

theorem quadratic_has_real_root (a b : ℝ) : 
  (¬(∀ x : ℝ, x^2 + a * x + b ≠ 0)) → (∃ x : ℝ, x^2 + a * x + b = 0) := 
by
  intro h
  sorry

end quadratic_has_real_root_l135_135310


namespace horizontal_length_of_rectangle_l135_135609

theorem horizontal_length_of_rectangle
  (P : ℕ)
  (h v : ℕ)
  (hP : P = 54)
  (hv : v = h - 3) :
  2*h + 2*v = 54 → h = 15 :=
by sorry

end horizontal_length_of_rectangle_l135_135609


namespace chef_makes_10_cakes_l135_135783

def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

theorem chef_makes_10_cakes :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 := by
  sorry

end chef_makes_10_cakes_l135_135783


namespace complement_P_l135_135604

def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_P :
  (U \ P) = Set.Iio (-1) ∪ Set.Ioi (1) :=
by
  sorry

end complement_P_l135_135604


namespace arithmetic_and_geometric_mean_l135_135369

theorem arithmetic_and_geometric_mean (x y : ℝ) (h₁ : (x + y) / 2 = 20) (h₂ : Real.sqrt (x * y) = Real.sqrt 150) : x^2 + y^2 = 1300 :=
by
  sorry

end arithmetic_and_geometric_mean_l135_135369


namespace meal_cost_l135_135270

theorem meal_cost (total_paid change tip_rate : ℝ)
  (h_total_paid : total_paid = 20 - change)
  (h_change : change = 5)
  (h_tip_rate : tip_rate = 0.2) :
  ∃ x, x + tip_rate * x = total_paid ∧ x = 12.5 := 
by
  sorry

end meal_cost_l135_135270


namespace solve_a_b_c_d_l135_135029

theorem solve_a_b_c_d (n a b c d : ℕ) (h0 : 0 ≤ a) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : 2^n = a^2 + b^2 + c^2 + d^2) : 
  (a, b, c, d) ∈ {p | p = (↑0, ↑0, ↑0, 2^n.div (↑4)) ∨
                  p = (↑0, ↑0, 2^n.div (↑4), 2^n.div (↑4)) ∨
                  p = (2^n.div (↑4), 2^n.div (↑4), 2^n.div (↑4), 2^n.div (↑4)) ∨
                  p = (2^n.div (↑4), 0, 0, 0) ∨
                  p = (0, 2^n.div (↑4), 0, 0) ∨
                  p = (0, 0, 2^n.div (↑4), 0) ∨
                  p = (0, 0, 0, 2^n.div (↑4))} :=
sorry

end solve_a_b_c_d_l135_135029


namespace evaluate_expression_l135_135844

theorem evaluate_expression (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 :=
by
  sorry

end evaluate_expression_l135_135844


namespace tan_of_angle_through_point_l135_135754

theorem tan_of_angle_through_point (α : ℝ) (hα : ∃ x y : ℝ, (x = 1) ∧ (y = 2) ∧ (y/x = (Real.sin α) / (Real.cos α))) :
  Real.tan α = 2 :=
sorry

end tan_of_angle_through_point_l135_135754


namespace roots_quadratic_solution_l135_135759

theorem roots_quadratic_solution (α β : ℝ) (hα : α^2 - 3*α - 2 = 0) (hβ : β^2 - 3*β - 2 = 0) :
  3*α^3 + 8*β^4 = 1229 := by
  sorry

end roots_quadratic_solution_l135_135759


namespace number_of_divisors_23232_l135_135257

theorem number_of_divisors_23232 : ∀ (n : ℕ), 
    n = 23232 → 
    (∃ k : ℕ, k = 42 ∧ (∀ d : ℕ, (d > 0 ∧ d ∣ n) → (↑d < k + 1))) :=
by
  sorry

end number_of_divisors_23232_l135_135257


namespace no_descending_multiple_of_111_l135_135002

theorem no_descending_multiple_of_111 :
  ¬ ∃ (n : ℕ), (∀ (i j : ℕ), (i < j ∧ (n / 10^i % 10) < (n / 10^j % 10)) ∨ (i = j)) ∧ 111 ∣ n :=
by
  sorry

end no_descending_multiple_of_111_l135_135002


namespace students_behind_yoongi_l135_135123

theorem students_behind_yoongi (total_students : ℕ) (jungkook_position : ℕ) (yoongi_position : ℕ) (behind_students : ℕ)
  (h1 : total_students = 20)
  (h2 : jungkook_position = 3)
  (h3 : yoongi_position = jungkook_position + 1)
  (h4 : behind_students = total_students - yoongi_position) :
  behind_students = 16 :=
by
  sorry

end students_behind_yoongi_l135_135123


namespace spears_per_sapling_l135_135386

/-- Given that a log can produce 9 spears and 6 saplings plus a log produce 27 spears,
prove that a single sapling can produce 3 spears (S = 3). -/
theorem spears_per_sapling (L S : ℕ) (hL : L = 9) (h: 6 * S + L = 27) : S = 3 :=
by
  sorry

end spears_per_sapling_l135_135386


namespace a7_is_1_S2022_is_4718_l135_135588

def harmonious_progressive (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, p > 0 → q > 0 → a p = a q → a (p + 1) = a (q + 1)

variables (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom harmonious_seq : harmonious_progressive a
axiom a1 : a 1 = 1
axiom a2 : a 2 = 2
axiom a4 : a 4 = 1
axiom a6_plus_a8 : a 6 + a 8 = 6

theorem a7_is_1 : a 7 = 1 := sorry

theorem S2022_is_4718 : S 2022 = 4718 := sorry

end a7_is_1_S2022_is_4718_l135_135588


namespace minimum_value_of_linear_expression_l135_135897

theorem minimum_value_of_linear_expression :
  ∀ (x y : ℝ), |y| ≤ 2 - x ∧ x ≥ -1 → 2 * x + y ≥ -5 :=
by
  sorry

end minimum_value_of_linear_expression_l135_135897


namespace spadesuit_eval_l135_135865

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_eval : spadesuit 2 (spadesuit 3 (spadesuit 1 2)) = 4 := 
by
  sorry

end spadesuit_eval_l135_135865


namespace inequality_holds_equality_cases_l135_135973

noncomputable def posReal : Type := { x : ℝ // 0 < x }

variables (a b c d : posReal)

theorem inequality_holds (a b c d : posReal) :
  (a.1 - b.1) * (a.1 - c.1) / (a.1 + b.1 + c.1) +
  (b.1 - c.1) * (b.1 - d.1) / (b.1 + c.1 + d.1) +
  (c.1 - d.1) * (c.1 - a.1) / (c.1 + d.1 + a.1) +
  (d.1 - a.1) * (d.1 - b.1) / (d.1 + a.1 + b.1) ≥ 0 :=
sorry

theorem equality_cases (a b c d : posReal) :
  (a.1 - b.1) * (a.1 - c.1) / (a.1 + b.1 + c.1) +
  (b.1 - c.1) * (b.1 - d.1) / (b.1 + c.1 + d.1) +
  (c.1 - d.1) * (c.1 - a.1) / (c.1 + d.1 + a.1) +
  (d.1 - a.1) * (d.1 - b.1) / (d.1 + a.1 + b.1) = 0 ↔
  (a.1 = c.1 ∧ b.1 = d.1) :=
sorry

end inequality_holds_equality_cases_l135_135973


namespace solve_ticket_problem_l135_135276

def ticket_problem : Prop :=
  ∃ S N : ℕ, S + N = 2000 ∧ 9 * S + 11 * N = 20960 ∧ S = 520

theorem solve_ticket_problem : ticket_problem :=
sorry

end solve_ticket_problem_l135_135276


namespace solve_system_l135_135839

theorem solve_system (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y = 4 * z) (h2 : x / y = 81) (h3 : x * z = 36) :
  x = 36 ∧ y = 4 / 9 ∧ z = 1 :=
by
  sorry

end solve_system_l135_135839


namespace range_of_m_l135_135792

variable {R : Type*} [LinearOrderedField R]

def discriminant (a b c : R) := b * b - 4 * a * c

theorem range_of_m (m : R) : (∀ x : R, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by sorry

end range_of_m_l135_135792


namespace fraction_to_decimal_l135_135842

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l135_135842


namespace christine_savings_l135_135224

/-- Christine's commission rate as a percentage. -/
def commissionRate : ℝ := 0.12

/-- Total sales made by Christine this month in dollars. -/
def totalSales : ℝ := 24000

/-- Percentage of commission allocated to personal needs. -/
def personalNeedsRate : ℝ := 0.60

/-- The amount Christine saved this month. -/
def amountSaved : ℝ := 1152

/--
Given the commission rate, total sales, and personal needs rate,
prove the amount saved is correctly calculated.
-/
theorem christine_savings :
  (1 - personalNeedsRate) * (commissionRate * totalSales) = amountSaved :=
by
  sorry

end christine_savings_l135_135224


namespace geometric_progression_x_value_l135_135422

noncomputable def geometric_progression_solution (x : ℝ) : Prop :=
  let a := -30 + x
  let b := -10 + x
  let c := 40 + x
  b^2 = a * c

theorem geometric_progression_x_value :
  ∃ x : ℝ, geometric_progression_solution x ∧ x = 130 / 3 :=
by
  sorry

end geometric_progression_x_value_l135_135422


namespace one_third_of_1206_is_300_percent_of_134_l135_135508

theorem one_third_of_1206_is_300_percent_of_134 :
  let number := 1206
  let fraction := 1 / 3
  let computed_one_third := fraction * number
  let whole := 134
  let expected_percent := 300
  let percent := (computed_one_third / whole) * 100
  percent = expected_percent := by
  let number := 1206
  let fraction := 1 / 3
  have computed_one_third : ℝ := fraction * number
  let whole := 134
  let expected_percent := 300
  have percent : ℝ := (computed_one_third / whole) * 100
  exact sorry

end one_third_of_1206_is_300_percent_of_134_l135_135508


namespace volume_is_correct_l135_135255

noncomputable def volume_of_rectangular_parallelepiped (a b : ℝ) (h_diag : (2 * a^2 + b^2 = 1)) (h_surface_area : (4 * a * b + 2 * a^2 = 1)) : ℝ :=
  a^2 * b

theorem volume_is_correct (a b : ℝ)
  (h_diag : 2 * a^2 + b^2 = 1)
  (h_surface_area : 4 * a * b + 2 * a^2 = 1) :
  volume_of_rectangular_parallelepiped a b h_diag h_surface_area = (Real.sqrt 2) / 27 :=
sorry

end volume_is_correct_l135_135255


namespace division_by_fraction_l135_135478

theorem division_by_fraction : 5 / (1 / 5) = 25 := by
  sorry

end division_by_fraction_l135_135478


namespace Claire_photos_l135_135112

variable (C : ℕ)

def Lisa_photos := 3 * C
def Robert_photos := C + 28

theorem Claire_photos :
  Lisa_photos C = Robert_photos C → C = 14 :=
by
  sorry

end Claire_photos_l135_135112


namespace journey_length_l135_135388

theorem journey_length (speed time : ℝ) (portions_covered total_portions : ℕ)
  (h_speed : speed = 40) (h_time : time = 0.7) (h_portions_covered : portions_covered = 4) (h_total_portions : total_portions = 5) :
  (speed * time / portions_covered) * total_portions = 35 :=
by
  sorry

end journey_length_l135_135388


namespace infinite_primes_of_form_2px_plus_1_l135_135260

theorem infinite_primes_of_form_2px_plus_1 (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) : 
  ∃ᶠ (n : ℕ) in at_top, Nat.Prime (2 * p * n + 1) :=
sorry

end infinite_primes_of_form_2px_plus_1_l135_135260


namespace solid_id_views_not_cylinder_l135_135320

theorem solid_id_views_not_cylinder :
  ∀ (solid : Type),
  (∃ (shape1 shape2 shape3 : solid),
    shape1 = shape2 ∧ shape2 = shape3) →
  solid ≠ cylinder :=
by 
  sorry

end solid_id_views_not_cylinder_l135_135320


namespace parameter_values_for_roots_l135_135576

theorem parameter_values_for_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 = 5 * x2 ∧ a * x1^2 - (2 * a + 5) * x1 + 10 = 0 ∧ a * x2^2 - (2 * a + 5) * x2 + 10 = 0)
  ↔ (a = 5 / 3 ∨ a = 5) := 
sorry

end parameter_values_for_roots_l135_135576


namespace sector_properties_l135_135252

variables (r : ℝ) (alpha l S : ℝ)

noncomputable def arc_length (r alpha : ℝ) : ℝ := alpha * r
noncomputable def sector_area (l r : ℝ) : ℝ := (1/2) * l * r

theorem sector_properties
  (h_r : r = 2)
  (h_alpha : alpha = π / 6) :
  arc_length r alpha = π / 3 ∧ sector_area (arc_length r alpha) r = π / 3 :=
by
  sorry

end sector_properties_l135_135252


namespace calculate_grand_total_profit_l135_135554

-- Definitions based on conditions
def cost_per_type_A : ℕ := 8 * 10
def sell_price_type_A : ℕ := 125
def cost_per_type_B : ℕ := 12 * 18
def sell_price_type_B : ℕ := 280
def cost_per_type_C : ℕ := 15 * 12
def sell_price_type_C : ℕ := 350

def num_sold_type_A : ℕ := 45
def num_sold_type_B : ℕ := 35
def num_sold_type_C : ℕ := 25

-- Definition of profit calculations
def profit_per_type_A : ℕ := sell_price_type_A - cost_per_type_A
def profit_per_type_B : ℕ := sell_price_type_B - cost_per_type_B
def profit_per_type_C : ℕ := sell_price_type_C - cost_per_type_C

def total_profit_type_A : ℕ := num_sold_type_A * profit_per_type_A
def total_profit_type_B : ℕ := num_sold_type_B * profit_per_type_B
def total_profit_type_C : ℕ := num_sold_type_C * profit_per_type_C

def grand_total_profit : ℕ := total_profit_type_A + total_profit_type_B + total_profit_type_C

-- Statement to be proved
theorem calculate_grand_total_profit : grand_total_profit = 8515 := by
  sorry

end calculate_grand_total_profit_l135_135554


namespace spokes_ratio_l135_135902

theorem spokes_ratio (B : ℕ) (front_spokes : ℕ) (total_spokes : ℕ) 
  (h1 : front_spokes = 20) 
  (h2 : total_spokes = 60) 
  (h3 : front_spokes + B = total_spokes) : 
  B / front_spokes = 2 :=
by 
  sorry

end spokes_ratio_l135_135902


namespace cos_ratio_l135_135820

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (angle_A angle_B angle_C : ℝ)
variable (bc_coeff : 2 * c = 3 * b)
variable (sin_coeff : Real.sin angle_A = 2 * Real.sin angle_B)

theorem cos_ratio :
  (2 * c = 3 * b) →
  (Real.sin angle_A = 2 * Real.sin angle_B) →
  let cos_A := (b^2 + c^2 - a^2) / (2 * b * c)
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  (Real.cos angle_A / Real.cos angle_B = -2 / 7) :=
by
  intros bc_coeff sin_coeff
  sorry

end cos_ratio_l135_135820


namespace hours_per_day_l135_135259

theorem hours_per_day
  (num_warehouse : ℕ := 4)
  (num_managers : ℕ := 2)
  (rate_warehouse : ℝ := 15)
  (rate_manager : ℝ := 20)
  (tax_rate : ℝ := 0.10)
  (days_worked : ℕ := 25)
  (total_cost : ℝ := 22000) :
  ∃ h : ℝ, 6 * h * days_worked * (rate_warehouse + rate_manager) * (1 + tax_rate) = total_cost ∧ h = 8 :=
by
  sorry

end hours_per_day_l135_135259


namespace integer_roots_k_values_y1_y2_squared_sum_k_0_y1_y2_squared_sum_k_neg1_l135_135629

theorem integer_roots_k_values (k : ℤ) :
  (∀ x : ℤ, k * x ^ 2 + (2 * k - 1) * x + k - 1 = 0) →
  k = 0 ∨ k = -1 :=
sorry

theorem y1_y2_squared_sum_k_0 (m y1 y2: ℝ) :
  (m > -2) →
  (k = 0) →
  ((k - 1) * y1 ^ 2 - 3 * y1 + m = 0) →
  ((k - 1) * y2 ^ 2 - 3 * y2 + m = 0) →
  y1^2 + y2^2 = 9 + 2 * m :=
sorry

theorem y1_y2_squared_sum_k_neg1 (m y1 y2: ℝ) :
  (m > -2) →
  (k = -1) →
  ((k - 1) * y1 ^ 2 - 3 * y1 + m = 0) →
  ((k - 1) * y2 ^ 2 - 3 * y2 + m = 0) →
  y1^2 + y2^2 = 9 / 4 + m :=
sorry

end integer_roots_k_values_y1_y2_squared_sum_k_0_y1_y2_squared_sum_k_neg1_l135_135629


namespace vector_expression_evaluation_l135_135734

theorem vector_expression_evaluation (θ : ℝ) :
  let a := (2 * Real.cos θ, Real.sin θ)
  let b := (1, -6)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (2 * Real.cos θ + Real.sin θ) / (Real.cos θ + 3 * Real.sin θ) = 7 / 6 :=
by
  intros a b h
  sorry

end vector_expression_evaluation_l135_135734


namespace highest_score_l135_135222

variables (H L : ℕ) (average_46 : ℕ := 61) (innings_46 : ℕ := 46) 
                (difference : ℕ := 150) (average_44 : ℕ := 58) (innings_44 : ℕ := 44)

theorem highest_score:
  (H - L = difference) →
  (average_46 * innings_46 = average_44 * innings_44 + H + L) →
  H = 202 :=
by
  intros h_diff total_runs_eq
  sorry

end highest_score_l135_135222


namespace time_for_B_alone_l135_135565

theorem time_for_B_alone (r_A r_B r_C : ℚ)
  (h1 : r_A + r_B = 1/3)
  (h2 : r_B + r_C = 2/7)
  (h3 : r_A + r_C = 1/4) :
  1/r_B = 168/31 :=
by
  sorry

end time_for_B_alone_l135_135565


namespace final_silver_tokens_l135_135978

structure TokenCounts :=
  (red : ℕ)
  (blue : ℕ)

def initial_tokens : TokenCounts := { red := 100, blue := 50 }

def exchange_booth1 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red - 3, blue := tokens.blue + 2 }

def exchange_booth2 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red + 1, blue := tokens.blue - 3 }

noncomputable def max_exchanges (initial : TokenCounts) : ℕ × ℕ :=
  let x := 48
  let y := 47
  (x, y)

noncomputable def silver_tokens (x y : ℕ) : ℕ := x + y

theorem final_silver_tokens (x y : ℕ) (tokens : TokenCounts) 
  (hx : tokens.red = initial_tokens.red - 3 * x + y)
  (hy : tokens.blue = initial_tokens.blue + 2 * x - 3 * y) 
  (hx_le : tokens.red >= 3 → false)
  (hy_le : tokens.blue >= 3 → false) : 
  silver_tokens x y = 95 :=
by {
  sorry
}

end final_silver_tokens_l135_135978


namespace greatest_value_NNM_l135_135351

theorem greatest_value_NNM :
  ∃ (M : ℕ), (M * M % 10 = M) ∧ (∃ (MM : ℕ), MM = 11 * M ∧ (MM * M = 396)) :=
by
  sorry

end greatest_value_NNM_l135_135351


namespace johns_raise_percent_increase_l135_135492

theorem johns_raise_percent_increase (original_earnings new_earnings : ℝ) 
  (h₀ : original_earnings = 60) (h₁ : new_earnings = 110) : 
  ((new_earnings - original_earnings) / original_earnings) * 100 = 83.33 :=
by
  sorry

end johns_raise_percent_increase_l135_135492


namespace math_problem_l135_135433

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem math_problem
  (omega phi : ℝ)
  (h1 : omega > 0)
  (h2 : |phi| < Real.pi / 2)
  (h3 : ∀ x, f x = Real.sin (omega * x + phi))
  (h4 : ∀ k : ℤ, f (k * Real.pi) = f 0) 
  (h5 : f 0 = 1 / 2) :
  (omega = 2) ∧
  (∀ x, f (x + Real.pi / 6) = f (-x + Real.pi / 6)) ∧
  (∀ k : ℤ, 
    ∀ x, x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    ∀ y, y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    x < y → f x ≤ f y) :=
by
  sorry

end math_problem_l135_135433


namespace complex_expression_equality_l135_135846

theorem complex_expression_equality (i : ℂ) (h : i^2 = -1) : (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end complex_expression_equality_l135_135846


namespace marbles_total_l135_135349

-- Conditions
variables (T : ℕ) -- Total number of marbles
variables (h_red : T ≥ 12) -- At least 12 red marbles
variables (h_blue : T ≥ 8) -- At least 8 blue marbles
variables (h_prob : (T - 12 : ℚ) / T = (3 / 4 : ℚ)) -- Probability condition

-- Proof statement
theorem marbles_total : T = 48 :=
by
  -- Proof here
  sorry

end marbles_total_l135_135349


namespace units_digit_8421_1287_l135_135247

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_8421_1287 :
  units_digit (8421 ^ 1287) = 1 := 
by
  sorry

end units_digit_8421_1287_l135_135247


namespace hua_luogeng_optimal_selection_method_l135_135173

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l135_135173


namespace donna_card_shop_hourly_wage_correct_l135_135725

noncomputable def donna_hourly_wage_at_card_shop : ℝ := 
  let total_earnings := 305.0
  let earnings_dog_walking := 2 * 10.0 * 5
  let earnings_babysitting := 4 * 10.0
  let earnings_card_shop := total_earnings - (earnings_dog_walking + earnings_babysitting)
  let hours_card_shop := 5 * 2
  earnings_card_shop / hours_card_shop

theorem donna_card_shop_hourly_wage_correct : donna_hourly_wage_at_card_shop = 16.50 :=
by 
  -- Skipping proof steps for the implementation
  sorry

end donna_card_shop_hourly_wage_correct_l135_135725


namespace mark_sold_9_boxes_less_than_n_l135_135638

theorem mark_sold_9_boxes_less_than_n :
  ∀ (n M A : ℕ),
  n = 10 →
  M < n →
  A = n - 2 →
  M + A < n →
  M ≥ 1 →
  A ≥ 1 →
  M = 1 ∧ n - M = 9 :=
by
  intros n M A h_n h_M_lt_n h_A h_MA_lt_n h_M_ge_1 h_A_ge_1
  rw [h_n, h_A] at *
  sorry

end mark_sold_9_boxes_less_than_n_l135_135638


namespace Martiza_study_time_l135_135748

theorem Martiza_study_time :
  ∀ (x : ℕ),
  (30 * x + 30 * 25 = 20 * 60) →
  x = 15 :=
by
  intros x h
  sorry

end Martiza_study_time_l135_135748


namespace quadratic_root_q_value_l135_135867

theorem quadratic_root_q_value
  (p q : ℝ)
  (h1 : ∃ r : ℝ, r = -3 ∧ 3 * r^2 + p * r + q = 0)
  (h2 : ∃ s : ℝ, -3 + s = -2) :
  q = -9 :=
sorry

end quadratic_root_q_value_l135_135867


namespace youseff_time_difference_l135_135446

noncomputable def walking_time (blocks : ℕ) (time_per_block : ℕ) : ℕ := blocks * time_per_block
noncomputable def biking_time (blocks : ℕ) (time_per_block_seconds : ℕ) : ℕ := (blocks * time_per_block_seconds) / 60

theorem youseff_time_difference : walking_time 6 1 - biking_time 6 20 = 4 := by
  sorry

end youseff_time_difference_l135_135446


namespace sum_reciprocals_squares_l135_135034

theorem sum_reciprocals_squares {a b : ℕ} (h1 : Nat.gcd a b = 1) (h2 : a * b = 11) :
  (1 / (a: ℚ)^2) + (1 / (b: ℚ)^2) = 122 / 121 := 
sorry

end sum_reciprocals_squares_l135_135034


namespace all_div_by_25_form_no_div_by_35_l135_135699

noncomputable def exists_div_by_25 (M : ℕ) : Prop :=
∃ (M N : ℕ) (n : ℕ), M = 6 * 10 ^ (n - 1) + N ∧ M = 25 * N ∧ 4 * N = 10 ^ (n - 1)

theorem all_div_by_25_form :
  ∀ M, exists_div_by_25 M → (∃ k : ℕ, M = 625 * 10 ^ k) :=
by
  intro M
  intro h
  sorry

noncomputable def not_exists_div_by_35 (M : ℕ) : Prop :=
∀ (M N : ℕ) (n : ℕ), M ≠ 6 * 10 ^ (n - 1) + N ∨ M ≠ 35 * N

theorem no_div_by_35 :
  ∀ M, not_exists_div_by_35 M :=
by
  intro M
  intro h
  sorry

end all_div_by_25_form_no_div_by_35_l135_135699


namespace exists_close_pair_in_interval_l135_135853

theorem exists_close_pair_in_interval (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 < 1) (h2 : 0 ≤ x2 ∧ x2 < 1) (h3 : 0 ≤ x3 ∧ x3 < 1) :
  ∃ a b, (a = x1 ∨ a = x2 ∨ a = x3) ∧ (b = x1 ∨ b = x2 ∨ b = x3) ∧ a ≠ b ∧ |b - a| < 1 / 2 :=
sorry

end exists_close_pair_in_interval_l135_135853


namespace amount_each_person_needs_to_raise_l135_135646

theorem amount_each_person_needs_to_raise (Total_goal Already_collected Number_of_people : ℝ) 
(h1 : Total_goal = 2400) (h2 : Already_collected = 300) (h3 : Number_of_people = 8) : 
    (Total_goal - Already_collected) / Number_of_people = 262.5 := 
by
  sorry

end amount_each_person_needs_to_raise_l135_135646


namespace find_hcf_l135_135216

-- Defining the conditions given in the problem
def hcf_of_two_numbers_is_H (A B H : ℕ) : Prop := Nat.gcd A B = H
def lcm_of_A_B (A B : ℕ) (H : ℕ) : Prop := Nat.lcm A B = H * 21 * 23
def larger_number_is_460 (A : ℕ) : Prop := A = 460

-- The propositional goal to prove that H = 20 given the above conditions
theorem find_hcf (A B H : ℕ) (hcf_cond : hcf_of_two_numbers_is_H A B H)
  (lcm_cond : lcm_of_A_B A B H) (larger_cond : larger_number_is_460 A) : H = 20 :=
sorry

end find_hcf_l135_135216


namespace additional_books_acquired_l135_135471

def original_stock : ℝ := 40.0
def shelves_used : ℕ := 15
def books_per_shelf : ℝ := 4.0

theorem additional_books_acquired :
  (shelves_used * books_per_shelf) - original_stock = 20.0 :=
by
  sorry

end additional_books_acquired_l135_135471


namespace part1_part2_l135_135536

-- Part (1): Solution set of the inequality
theorem part1 (x : ℝ) : (|x - 1| + |x + 1| ≤ 8 - x^2) ↔ (-2 ≤ x) ∧ (x ≤ 2) :=
by
  sorry

-- Part (2): Range of real number t
theorem part2 (t : ℝ) (m n : ℝ) (x : ℝ) (h1 : m + n = 4) (h2 : m > 0) (h3 : n > 0) :  
  |x-t| + |x+t| = (4 * m^2 + n) / (m * n) → t ≥ 9 / 8 ∨ t ≤ -9 / 8 :=
by
  sorry

end part1_part2_l135_135536


namespace values_of_n_eq_100_l135_135586

theorem values_of_n_eq_100 :
  ∃ (n_count : ℕ), n_count = 100 ∧
    ∀ (a b c : ℕ),
      a + 11 * b + 111 * c = 900 →
      (∀ (a : ℕ), a ≥ 0) →
      (∃ (n : ℕ), n = a + 2 * b + 3 * c ∧ n_count = 100) :=
sorry

end values_of_n_eq_100_l135_135586


namespace find_initial_balance_l135_135528

-- Define the initial balance (X)
def initial_balance (X : ℝ) := 
  ∃ (X : ℝ), (X / 2 + 30 + 50 - 20 = 160)

theorem find_initial_balance (X : ℝ) (h : initial_balance X) : 
  X = 200 :=
sorry

end find_initial_balance_l135_135528


namespace common_ratio_of_geometric_sequence_l135_135850

variable (a : ℕ → ℝ) -- The geometric sequence {a_n}
variable (q : ℝ)     -- The common ratio

-- Conditions
axiom h1 : a 2 = 18
axiom h2 : a 4 = 8

theorem common_ratio_of_geometric_sequence :
  (∀ n : ℕ, a (n + 1) = a n * q) ∧ q^2 = 4/9 → q = 2/3 ∨ q = -2/3 := by
  sorry

end common_ratio_of_geometric_sequence_l135_135850


namespace football_game_wristbands_l135_135501

theorem football_game_wristbands (total_wristbands wristbands_per_person : Nat) (h1 : total_wristbands = 290) (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 145 :=
by
  sorry

end football_game_wristbands_l135_135501


namespace consecutive_integers_sum_l135_135763

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 504) : n + n+1 + n+2 = 24 :=
sorry

end consecutive_integers_sum_l135_135763


namespace equal_games_per_month_l135_135198

-- Define the given conditions
def total_games : ℕ := 27
def months : ℕ := 3
def games_per_month := total_games / months

-- Proposition that needs to be proven
theorem equal_games_per_month : games_per_month = 9 := 
by
  sorry

end equal_games_per_month_l135_135198


namespace brothers_travel_distance_l135_135271

theorem brothers_travel_distance
  (x : ℝ)
  (hb_x : (120 : ℝ) / (x : ℝ) - 4 = (120 : ℝ) / (x + 40))
  (total_time : 2 = 2) :
  x = 20 ∧ (x + 40) = 60 :=
by
  -- we need to prove the distances
  sorry

end brothers_travel_distance_l135_135271


namespace cylindrical_log_distance_l135_135191

def cylinder_radius := 3
def R₁ := 104
def R₂ := 64
def R₃ := 84
def straight_segment := 100

theorem cylindrical_log_distance :
  let adjusted_radius₁ := R₁ - cylinder_radius
  let adjusted_radius₂ := R₂ + cylinder_radius
  let adjusted_radius₃ := R₃ - cylinder_radius
  let arc_distance₁ := π * adjusted_radius₁
  let arc_distance₂ := π * adjusted_radius₂
  let arc_distance₃ := π * adjusted_radius₃
  let total_distance := arc_distance₁ + arc_distance₂ + arc_distance₃ + straight_segment
  total_distance = 249 * π + 100 :=
sorry

end cylindrical_log_distance_l135_135191


namespace number_of_basketball_cards_l135_135049

theorem number_of_basketball_cards 
  (B : ℕ) -- Number of basketball cards in each box
  (H1 : 4 * B = 40) -- Given condition from equation 4B = 40
  
  (H2 : 4 * B + 40 - 58 = 22) -- Given condition from the total number of cards

: B = 10 := 
by 
  sorry

end number_of_basketball_cards_l135_135049


namespace find_t_from_tan_conditions_l135_135677

theorem find_t_from_tan_conditions 
  (α t : ℝ)
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + Real.pi / 4) = 4 / t)
  (h3 : Real.tan (α + Real.pi / 4) = (Real.tan (Real.pi / 4) + Real.tan α) / (1 - Real.tan (Real.pi / 4) * Real.tan α)) :
  t = 2 := 
  by
  sorry

end find_t_from_tan_conditions_l135_135677


namespace inequality_solution_l135_135044

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -(4 / 3) ∨ x > -(5 / 3)) := 
by
  sorry

end inequality_solution_l135_135044


namespace NineChaptersProblem_l135_135751

-- Conditions: Assign the given conditions to variables
variables (x y : Int)
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Proof problem: Prove that the system of equations is consistent with the given conditions
theorem NineChaptersProblem : condition1 x y ∧ condition2 x y := sorry

end NineChaptersProblem_l135_135751


namespace find_dividend_l135_135324

-- Define the given conditions
def quotient : ℝ := 0.0012000000000000001
def divisor : ℝ := 17

-- State the problem: Prove that the dividend is the product of the quotient and the divisor
theorem find_dividend (q : ℝ) (d : ℝ) (hq : q = 0.0012000000000000001) (hd : d = 17) : 
  q * d = 0.0204000000000000027 :=
sorry

end find_dividend_l135_135324


namespace yield_percentage_is_correct_l135_135688

-- Defining the conditions and question
def market_value := 70
def face_value := 100
def dividend_percentage := 7
def annual_dividend := (dividend_percentage * face_value) / 100

-- Lean statement to prove the yield percentage
theorem yield_percentage_is_correct (market_value: ℕ) (annual_dividend: ℝ) : 
  ((annual_dividend / market_value) * 100) = 10 := 
by
  -- conditions from a)
  have market_value := 70
  have face_value := 100
  have dividend_percentage := 7
  have annual_dividend := (dividend_percentage * face_value) / 100
  
  -- proof will go here
  sorry

end yield_percentage_is_correct_l135_135688


namespace opposite_quotient_l135_135713

theorem opposite_quotient {a b : ℝ} (h1 : a ≠ b) (h2 : a = -b) : a / b = -1 := 
sorry

end opposite_quotient_l135_135713


namespace scientific_notation_470000000_l135_135407

theorem scientific_notation_470000000 : 470000000 = 4.7 * 10^8 :=
by
  sorry

end scientific_notation_470000000_l135_135407


namespace sum_of_digits_eq_4_l135_135535

theorem sum_of_digits_eq_4 (A B C D X Y : ℕ) (h1 : A + B + C + D = 22) (h2 : B + D = 9) (h3 : X = 1) (h4 : Y = 3) :
    X + Y = 4 :=
by
  sorry

end sum_of_digits_eq_4_l135_135535


namespace wilted_flowers_are_18_l135_135409

def picked_flowers := 53
def flowers_per_bouquet := 7
def bouquets_after_wilted := 5

def flowers_left := bouquets_after_wilted * flowers_per_bouquet
def flowers_wilted : ℕ := picked_flowers - flowers_left

theorem wilted_flowers_are_18 : flowers_wilted = 18 := by
  sorry

end wilted_flowers_are_18_l135_135409


namespace number_of_division_games_l135_135718

theorem number_of_division_games (N M : ℕ) (h1 : N > 2 * M) (h2 : M > 5) (h3 : 4 * N + 5 * M = 100) :
  4 * N = 60 :=
by
  sorry

end number_of_division_games_l135_135718


namespace greatest_possible_value_of_y_l135_135333

-- Definitions according to problem conditions
variables {x y : ℤ}

-- The theorem statement to prove
theorem greatest_possible_value_of_y (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
sorry

end greatest_possible_value_of_y_l135_135333


namespace range_of_a_l135_135163

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l135_135163


namespace sally_lost_orange_balloons_l135_135655

theorem sally_lost_orange_balloons :
  ∀ (initial_orange_balloons lost_orange_balloons current_orange_balloons : ℕ),
  initial_orange_balloons = 9 →
  current_orange_balloons = 7 →
  lost_orange_balloons = initial_orange_balloons - current_orange_balloons →
  lost_orange_balloons = 2 :=
by
  intros initial_orange_balloons lost_orange_balloons current_orange_balloons
  intros h_init h_current h_lost
  rw [h_init, h_current] at h_lost
  exact h_lost

end sally_lost_orange_balloons_l135_135655


namespace neg_p_necessary_not_sufficient_neg_q_l135_135829

def p (x : ℝ) := abs x < 1
def q (x : ℝ) := x^2 + x - 6 < 0

theorem neg_p_necessary_not_sufficient_neg_q :
  (¬ (∃ x, p x)) → (¬ (∃ x, q x)) ∧ ¬ ((¬ (∃ x, p x)) → (¬ (∃ x, q x))) :=
by
  sorry

end neg_p_necessary_not_sufficient_neg_q_l135_135829


namespace hyperbola_properties_l135_135928

-- Define the conditions and the final statements we need to prove
theorem hyperbola_properties (a : ℝ) (ha : a > 2) (E : ℝ → ℝ → Prop)
  (hE : ∀ x y, E x y ↔ (x^2 / a^2 - y^2 / (a^2 - 4) = 1))
  (e : ℝ) (he : e = (Real.sqrt (a^2 + (a^2 - 4))) / a) :
  (∃ E' : ℝ → ℝ → Prop,
   ∀ x y, E' x y ↔ (x^2 / 9 - y^2 / 5 = 1)) ∧
  (∃ foci line: ℝ → ℝ → Prop,
   (∀ P : ℝ × ℝ, (E P.1 P.2) →
    (∃ Q : ℝ × ℝ, (P.1 - Q.1) * (P.1 + (Real.sqrt (2*a^2-4))) = 0 ∧ Q.2=0 ∧ 
     line (P.1) (P.2) ↔ P.1 - P.2 = 2))) :=
by
  sorry

end hyperbola_properties_l135_135928


namespace find_larger_number_l135_135306

theorem find_larger_number (x y : ℕ) 
  (h1 : 4 * y = 5 * x) 
  (h2 : x + y = 54) : 
  y = 30 :=
sorry

end find_larger_number_l135_135306


namespace net_profit_correct_l135_135218

-- Define the conditions
def unit_price : ℝ := 1.25
def selling_price : ℝ := 12
def num_patches : ℕ := 100

-- Define the required total cost
def total_cost : ℝ := num_patches * unit_price

-- Define the required total revenue
def total_revenue : ℝ := num_patches * selling_price

-- Define the net profit calculation
def net_profit : ℝ := total_revenue - total_cost

-- The theorem we need to prove
theorem net_profit_correct : net_profit = 1075 := by
    sorry

end net_profit_correct_l135_135218


namespace sam_initial_money_l135_135211

theorem sam_initial_money (num_books cost_per_book money_left initial_money : ℤ) 
  (h1 : num_books = 9) 
  (h2 : cost_per_book = 7) 
  (h3 : money_left = 16)
  (h4 : initial_money = num_books * cost_per_book + money_left) :
  initial_money = 79 := 
by
  -- Proof is not required, hence we use sorry to complete the statement.
  sorry

end sam_initial_money_l135_135211


namespace solve_for_a_l135_135373

theorem solve_for_a {f : ℝ → ℝ} (h1 : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 2) (h2 : f a = 7) : a = 7 :=
sorry

end solve_for_a_l135_135373


namespace average_of_remaining_numbers_l135_135798

theorem average_of_remaining_numbers 
  (S S' : ℝ)
  (h1 : S / 12 = 90)
  (h2 : S' = S - 80 - 82) :
  S' / 10 = 91.8 :=
sorry

end average_of_remaining_numbers_l135_135798


namespace num_squares_in_6_by_6_grid_l135_135340

def squares_in_grid (m n : ℕ) : ℕ :=
  (m - 1) * (m - 1) + (m - 2) * (m - 2) + 
  (m - 3) * (m - 3) + (m - 4) * (m - 4) + 
  (m - 5) * (m - 5)

theorem num_squares_in_6_by_6_grid : squares_in_grid 6 6 = 55 := 
by 
  sorry

end num_squares_in_6_by_6_grid_l135_135340


namespace f_monotonically_increasing_on_1_to_infinity_l135_135322

noncomputable def f (x : ℝ) : ℝ := x + 1/x

theorem f_monotonically_increasing_on_1_to_infinity :
  ∀ x y : ℝ, 1 < x → x < y → f x < f y := 
sorry

end f_monotonically_increasing_on_1_to_infinity_l135_135322


namespace average_monthly_growth_rate_l135_135230

-- Define the conditions
variables (P : ℝ) (r : ℝ)
-- The condition that output in December is P times that of January
axiom growth_rate_condition : (1 + r)^11 = P

-- Define the goal to prove the average monthly growth rate
theorem average_monthly_growth_rate : r = (P^(1/11) - 1) :=
by
  sorry

end average_monthly_growth_rate_l135_135230


namespace ratio_accepted_rejected_l135_135138

-- Definitions for the conditions given
def eggs_per_day : ℕ := 400
def ratio_accepted_to_rejected : ℕ × ℕ := (96, 4)
def additional_accepted_eggs : ℕ := 12

/-- The ratio of accepted eggs to rejected eggs on that particular day is 99:1. -/
theorem ratio_accepted_rejected (a r : ℕ) (h1 : ratio_accepted_to_rejected = (a, r)) 
  (h2 : (a + r) * (eggs_per_day / (a + r)) = eggs_per_day) 
  (h3 : additional_accepted_eggs = 12) :
  (a + additional_accepted_eggs) / r = 99 :=
  sorry

end ratio_accepted_rejected_l135_135138


namespace simplify_expression_l135_135581

theorem simplify_expression (a : ℝ) (h : a = Real.sqrt 3 - 3) : 
  (a^2 - 4 * a + 4) / (a^2 - 4) / ((a - 2) / (a^2 + 2 * a)) + 3 = Real.sqrt 3 :=
by
  sorry

end simplify_expression_l135_135581


namespace min_employees_needed_l135_135202

theorem min_employees_needed
  (W A S : Finset ℕ)
  (hW : W.card = 120)
  (hA : A.card = 150)
  (hS : S.card = 100)
  (hWA : (W ∩ A).card = 50)
  (hAS : (A ∩ S).card = 30)
  (hWS : (W ∩ S).card = 20)
  (hWAS : (W ∩ A ∩ S).card = 10) :
  (W ∪ A ∪ S).card = 280 :=
by
  sorry

end min_employees_needed_l135_135202


namespace cone_height_l135_135755

theorem cone_height (S h H Vcone Vcylinder : ℝ)
  (hcylinder_height : H = 9)
  (hvolumes : Vcone = Vcylinder)
  (hbase_areas : S = S)
  (hV_cone : Vcone = (1 / 3) * S * h)
  (hV_cylinder : Vcylinder = S * H) : h = 27 :=
by
  -- sorry is used here to indicate missing proof steps which are predefined as unnecessary
  sorry

end cone_height_l135_135755


namespace solve_expression_l135_135948

theorem solve_expression : (0.76 ^ 3 - 0.008) / (0.76 ^ 2 + 0.76 * 0.2 + 0.04) = 0.560 := 
by
  sorry

end solve_expression_l135_135948


namespace ball_bounce_height_l135_135092

theorem ball_bounce_height :
  ∃ k : ℕ, k = 4 ∧ 45 * (1 / 3 : ℝ) ^ k < 2 :=
by 
  use 4
  sorry

end ball_bounce_height_l135_135092


namespace range_of_m_l135_135745

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + 5 < 4 * x - 1 ∧ x > m → x > 2) → m ≤ 2 :=
by
  intro h
  have h₁ := h 2
  sorry

end range_of_m_l135_135745


namespace digit_B_identification_l135_135129

theorem digit_B_identification (B : ℕ) 
  (hB_range : 0 ≤ B ∧ B < 10) 
  (h_units_digit : (5 * B % 10) = 5) 
  (h_product : (10 * B + 5) * (90 + B) = 9045) : 
  B = 9 :=
sorry

end digit_B_identification_l135_135129


namespace find_13x2_22xy_13y2_l135_135284

variable (x y : ℝ)

theorem find_13x2_22xy_13y2 
  (h1 : 3 * x + 2 * y = 8) 
  (h2 : 2 * x + 3 * y = 11) 
: 13 * x^2 + 22 * x * y + 13 * y^2 = 184 := 
sorry

end find_13x2_22xy_13y2_l135_135284


namespace locus_of_point_P_l135_135111

theorem locus_of_point_P (x y : ℝ) :
  let M := (-2, 0)
  let N := (2, 0)
  (x^2 + y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2) ↔ 
  ((x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16 ∧ x ≠ 2 ∧ x ≠ -2) :=
by
  sorry 

end locus_of_point_P_l135_135111


namespace initial_number_proof_l135_135061

-- Definitions for the given problem
def to_add : ℝ := 342.00000000007276
def multiple_of_412 (n : ℤ) : ℝ := 412 * n

-- The initial number
def initial_number : ℝ := 412 - to_add

-- The proof problem statement
theorem initial_number_proof (n : ℤ) (h : multiple_of_412 n = initial_number + to_add) : 
  ∃ x : ℝ, initial_number = x := 
sorry

end initial_number_proof_l135_135061


namespace num_divisors_1215_l135_135278

theorem num_divisors_1215 : (Finset.filter (λ d => 1215 % d = 0) (Finset.range (1215 + 1))).card = 12 :=
by
  sorry

end num_divisors_1215_l135_135278


namespace tv_weight_difference_l135_135343

-- Definitions for the given conditions
def bill_tv_length : ℕ := 48
def bill_tv_width : ℕ := 100
def bob_tv_length : ℕ := 70
def bob_tv_width : ℕ := 60
def weight_per_square_inch : ℕ := 4
def ounces_per_pound : ℕ := 16

-- The statement to prove
theorem tv_weight_difference : (bill_tv_length * bill_tv_width * weight_per_square_inch)
                               - (bob_tv_length * bob_tv_width * weight_per_square_inch)
                               = 150 * ounces_per_pound := by
  sorry

end tv_weight_difference_l135_135343


namespace triangle_inequality_inequality_l135_135068

theorem triangle_inequality_inequality {a b c p q r : ℝ}
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b)
  (h4 : p + q + r = 0) :
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 :=
sorry

end triangle_inequality_inequality_l135_135068


namespace primes_solution_l135_135775

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_solution (p : ℕ) (hp : is_prime p) :
  is_prime (p^2 + 2007 * p - 1) ↔ p = 3 :=
by
  sorry

end primes_solution_l135_135775


namespace absent_children_l135_135834

-- Definitions
def total_children := 840
def bananas_per_child_present := 4
def bananas_per_child_if_all_present := 2
def total_bananas_if_all_present := total_children * bananas_per_child_if_all_present

-- The theorem to prove
theorem absent_children (A : ℕ) (P : ℕ) :
  P = total_children - A →
  total_bananas_if_all_present = P * bananas_per_child_present →
  A = 420 :=
by
  sorry

end absent_children_l135_135834


namespace natalia_apartment_number_unit_digit_l135_135701

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def true_statements (n : ℕ) : Prop :=
  (n % 3 = 0 → true) ∧   -- Statement (1): divisible by 3
  (∃ k : ℕ, k^2 = n → true) ∧  -- Statement (2): square number
  (n % 2 = 1 → true) ∧   -- Statement (3): odd
  (n % 10 = 4 → true)     -- Statement (4): ends in 4

def three_out_of_four_true (n : ℕ) : Prop :=
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 = 1 ∧ n % 10 ≠ 4) ∨
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 ≠ 1 ∧ n % 10 = 4) ∨
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 ≠ n) ∧ n % 2 = 1 ∧ n % 10 = 4) ∨
  (n % 3 ≠ 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 = 1 ∧ n % 10 = 4)

theorem natalia_apartment_number_unit_digit :
  ∀ n : ℕ, two_digit_number n → three_out_of_four_true n → n % 10 = 1 :=
by sorry

end natalia_apartment_number_unit_digit_l135_135701


namespace find_pairs_l135_135250

theorem find_pairs (n k : ℕ) : (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) := by
  sorry

end find_pairs_l135_135250


namespace rectangle_length_to_width_ratio_l135_135499

-- Define the side length of the square
def s : ℝ := 1 -- Since we only need the ratio, the actual length does not matter

-- Define the length and width of the large rectangle
def length_of_large_rectangle : ℝ := 3 * s
def width_of_large_rectangle : ℝ := 3 * s

-- Define the dimensions of the small rectangle
def length_of_rectangle : ℝ := 3 * s
def width_of_rectangle : ℝ := s

-- Proving that the length of the rectangle is 3 times its width
theorem rectangle_length_to_width_ratio : length_of_rectangle = 3 * width_of_rectangle := 
by
  -- The proof is omitted
  sorry

end rectangle_length_to_width_ratio_l135_135499


namespace bracelet_price_l135_135811

theorem bracelet_price 
  (B : ℝ) -- price of each bracelet
  (H1 : B > 0) 
  (H2 : 3 * B + 2 * 10 + 20 = 100 - 15) : 
  B = 15 :=
by
  sorry

end bracelet_price_l135_135811


namespace simplify_fraction_l135_135366

theorem simplify_fraction : (5^3 + 5^5) / (5^4 - 5^2) = 65 / 12 := 
by 
  sorry

end simplify_fraction_l135_135366


namespace cubed_identity_l135_135630

theorem cubed_identity (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := 
by 
  sorry

end cubed_identity_l135_135630


namespace bees_count_l135_135289

-- Definitions of the conditions
def day1_bees (x : ℕ) := x  -- Number of bees on the first day
def day2_bees (x : ℕ) := 3 * day1_bees x  -- Number of bees on the second day is 3 times that on the first day

theorem bees_count (x : ℕ) (h : day2_bees x = 432) : day1_bees x = 144 :=
by
  dsimp [day1_bees, day2_bees] at h
  have h1 : 3 * x = 432 := h
  sorry

end bees_count_l135_135289


namespace absolute_difference_volumes_l135_135757

/-- The absolute difference in volumes of the cylindrical tubes formed by Amy and Carlos' papers. -/
theorem absolute_difference_volumes :
  let h_A := 12
  let C_A := 10
  let r_A := C_A / (2 * Real.pi)
  let V_A := Real.pi * r_A^2 * h_A
  let h_C := 8
  let C_C := 14
  let r_C := C_C / (2 * Real.pi)
  let V_C := Real.pi * r_C^2 * h_C
  abs (V_C - V_A) = 92 / Real.pi :=
by
  sorry

end absolute_difference_volumes_l135_135757


namespace sqrt_simplify_l135_135126

theorem sqrt_simplify (a b x : ℝ) (h : a < b) (hx1 : x + b ≥ 0) (hx2 : x + a ≤ 0) :
  Real.sqrt (-(x + a)^3 * (x + b)) = -(x + a) * (Real.sqrt (-(x + a) * (x + b))) :=
by
  sorry

end sqrt_simplify_l135_135126


namespace harish_ganpat_paint_wall_together_l135_135246

theorem harish_ganpat_paint_wall_together :
  let r_h := 1 / 3 -- Harish's rate of work (walls per hour)
  let r_g := 1 / 6 -- Ganpat's rate of work (walls per hour)
  let combined_rate := r_h + r_g -- Combined rate of work when both work together
  let time_to_paint_one_wall := 1 / combined_rate -- Time to paint one wall together
  time_to_paint_one_wall = 2 :=
by
  sorry

end harish_ganpat_paint_wall_together_l135_135246


namespace sock_pairs_proof_l135_135118

noncomputable def numPairsOfSocks : ℕ :=
  let n : ℕ := sorry
  n

theorem sock_pairs_proof : numPairsOfSocks = 6 := by
  sorry

end sock_pairs_proof_l135_135118


namespace find_y_l135_135940

theorem find_y (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_rem : x % y = 3) (h_div : (x:ℝ) / y = 96.15) : y = 20 :=
by
  sorry

end find_y_l135_135940


namespace number_of_interior_diagonals_of_dodecahedron_l135_135717

-- Definitions based on conditions
def dodecahedron_vertices := 20
def faces_per_vertex := 3
def vertices_per_face := 5
def shared_edges_per_vertex := faces_per_vertex
def total_faces := 12
def total_vertices := 20

-- Property of the dodecahedron
def potential_diagonals_per_vertex := dodecahedron_vertices - 1 - shared_edges_per_vertex - (vertices_per_face - 1)
def total_potential_diagonals := potential_diagonals_per_vertex * total_vertices

-- Proof statement:
theorem number_of_interior_diagonals_of_dodecahedron :
  total_potential_diagonals / 2 = 90 :=
by
  -- This is where the proof would go.
  sorry

end number_of_interior_diagonals_of_dodecahedron_l135_135717


namespace incorrect_statement_l135_135437

-- Define the relationship between the length of the spring and the mass of the object
def spring_length (mass : ℝ) : ℝ := 2.5 * mass + 10

-- Formalize statements A, B, C, and D
def statementA : Prop := spring_length 0 = 10

def statementB : Prop :=
  ¬ ∃ (length : ℝ) (mass : ℝ), (spring_length mass = length ∧ mass = (length - 10) / 2.5)

def statementC : Prop :=
  ∀ m : ℝ, spring_length (m + 1) = spring_length m + 2.5

def statementD : Prop := spring_length 4 = 20

-- The Lean statement to prove that statement B is incorrect
theorem incorrect_statement (hA : statementA) (hC : statementC) (hD : statementD) : ¬ statementB := by
  sorry

end incorrect_statement_l135_135437


namespace geometric_sequence_a6_l135_135567

theorem geometric_sequence_a6 : 
  ∀ (a : ℕ → ℚ), (∀ n, a n ≠ 0) → a 1 = 3 → (∀ n, 2 * a (n+1) - a n = 0) → a 6 = 3 / 32 :=
by
  intros a h1 h2 h3
  sorry

end geometric_sequence_a6_l135_135567


namespace general_term_b_l135_135868

noncomputable def S (n : ℕ) : ℚ := sorry -- Define the sum of the first n terms sequence S_n
noncomputable def a (n : ℕ) : ℚ := sorry -- Define the sequence a_n
noncomputable def b (n : ℕ) : ℤ := Int.log 3 (|a n|) -- Define the sequence b_n using log base 3

-- Theorem stating the general formula for the sequence b_n
theorem general_term_b (n : ℕ) (h : 0 < n) :
  b n = -n :=
sorry -- We skip the proof, focusing on statement declaration

end general_term_b_l135_135868


namespace work_done_time_l135_135357

/-
  Question: How many days does it take for \(a\) to do the work alone?

  Conditions:
  - \(b\) can do the work in 20 days.
  - \(c\) can do the work in 55 days.
  - \(a\) is assisted by \(b\) and \(c\) on alternate days, and the work can be done in 8 days.
  
  Correct Answer:
  - \(x = 8.8\)
-/

theorem work_done_time (x : ℝ) (h : 8 * x⁻¹ + 1 /  5 + 4 / 55 = 1): x = 8.8 :=
by sorry

end work_done_time_l135_135357


namespace bono_jelly_beans_l135_135023

variable (t A B C : ℤ)

theorem bono_jelly_beans (h₁ : A + B = 6 * t + 3) 
                         (h₂ : A + C = 4 * t + 5) 
                         (h₃ : B + C = 6 * t) : 
                         B = 4 * t - 1 := by
  sorry

end bono_jelly_beans_l135_135023


namespace sequence_to_geometric_l135_135399

variable (a : ℕ → ℝ)

def seq_geom (a : ℕ → ℝ) : Prop :=
∀ m n, a (m + n) = a m * a n

def condition (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 2) = a n * a (n + 1)

theorem sequence_to_geometric (a1 a2 : ℝ) (h1 : a 1 = a1) (h2 : a 2 = a2) (h : ∀ n, a (n + 2) = a n * a (n + 1)) :
  a1 = 1 → a2 = 1 → seq_geom a :=
by
  intros ha1 ha2
  have h_seq : ∀ n, a n = 1 := sorry
  intros m n
  sorry

end sequence_to_geometric_l135_135399


namespace q_simplified_l135_135462

noncomputable def q (a b c x : ℝ) : ℝ :=
  (x + a)^4 / ((a - b) * (a - c)) +
  (x + b)^4 / ((b - a) * (b - c)) +
  (x + c)^4 / ((c - a) * (c - b)) - 3 * x * (
      1 / ((a - b) * (a - c)) + 
      1 / ((b - a) * (b - c)) +
      1 / ((c - a) * (c - b))
  )

theorem q_simplified (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  q a b c x = a^2 + b^2 + c^2 + 4*x^2 - 4*(a + b + c)*x + 12*x :=
sorry

end q_simplified_l135_135462


namespace laptop_weight_l135_135468

-- Defining the weights
variables (B U L P : ℝ)
-- Karen's tote weight
def K := 8

-- Conditions from the problem
axiom tote_eq_two_briefcase : K = 2 * B
axiom umbrella_eq_half_briefcase : U = B / 2
axiom full_briefcase_eq_double_tote : B + L + P + U = 2 * K
axiom papers_eq_sixth_full_briefcase : P = (B + L + P) / 6

-- Theorem stating the weight of Kevin's laptop is 7.67 pounds
theorem laptop_weight (hB : B = 4) (hU : U = 2) (hL : L = 7.67) : 
  L - K = -0.33 :=
by
  sorry

end laptop_weight_l135_135468


namespace students_who_like_both_channels_l135_135949

theorem students_who_like_both_channels (total_students : ℕ) 
    (sports_channel : ℕ) (arts_channel : ℕ) (neither_channel : ℕ)
    (h_total : total_students = 100) (h_sports : sports_channel = 68) 
    (h_arts : arts_channel = 55) (h_neither : neither_channel = 3) :
    ∃ x, (x = 26) :=
by
  have h_at_least_one := total_students - neither_channel
  have h_A_union_B := sports_channel + arts_channel - h_at_least_one
  use h_A_union_B
  sorry

end students_who_like_both_channels_l135_135949


namespace stock_value_order_l135_135992

-- Define the initial investment and yearly changes
def initialInvestment : Float := 100
def firstYearChangeA : Float := 1.30
def firstYearChangeB : Float := 0.70
def firstYearChangeG : Float := 1.10
def firstYearChangeD : Float := 1.00 -- unchanged

def secondYearChangeA : Float := 0.90
def secondYearChangeB : Float := 1.35
def secondYearChangeG : Float := 1.05
def secondYearChangeD : Float := 1.10

-- Calculate the final values after two years
def finalValueA : Float := initialInvestment * firstYearChangeA * secondYearChangeA
def finalValueB : Float := initialInvestment * firstYearChangeB * secondYearChangeB
def finalValueG : Float := initialInvestment * firstYearChangeG * secondYearChangeG
def finalValueD : Float := initialInvestment * firstYearChangeD * secondYearChangeD

-- Theorem statement - Prove that the final order of the values is B < D < G < A
theorem stock_value_order : finalValueB < finalValueD ∧ finalValueD < finalValueG ∧ finalValueG < finalValueA := by
  sorry

end stock_value_order_l135_135992


namespace determine_hyperbola_eq_l135_135538

def hyperbola_eq (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1

def asymptote_condition (a b : ℝ) : Prop :=
  b / a = 3 / 4

def focus_condition (a b : ℝ) : Prop :=
  a^2 + b^2 = 25

theorem determine_hyperbola_eq : 
  ∃ a b : ℝ, 
  (a > 0) ∧ (b > 0) ∧ asymptote_condition a b ∧ focus_condition a b ∧ hyperbola_eq 4 3 :=
sorry

end determine_hyperbola_eq_l135_135538


namespace quadratic_range_l135_135613

theorem quadratic_range (x y : ℝ) 
    (h1 : y = (x - 1)^2 + 1)
    (h2 : 2 ≤ y ∧ y < 5) : 
    (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) :=
by
  sorry

end quadratic_range_l135_135613


namespace three_digit_numbers_satisfy_condition_l135_135768

theorem three_digit_numbers_satisfy_condition : 
  ∃ (x y z : ℕ), 
    1 ≤ x ∧ x ≤ 9 ∧ 
    0 ≤ y ∧ y ≤ 9 ∧ 
    0 ≤ z ∧ z ≤ 9 ∧ 
    x + y + z = (10 * x + y) - (10 * y + z) ∧ 
    (100 * x + 10 * y + z = 209 ∨ 
     100 * x + 10 * y + z = 428 ∨ 
     100 * x + 10 * y + z = 647 ∨ 
     100 * x + 10 * y + z = 866 ∨ 
     100 * x + 10 * y + z = 214 ∨ 
     100 * x + 10 * y + z = 433 ∨ 
     100 * x + 10 * y + z = 652 ∨ 
     100 * x + 10 * y + z = 871) := sorry

end three_digit_numbers_satisfy_condition_l135_135768


namespace induction_step_l135_135113

theorem induction_step 
  (k : ℕ) 
  (hk : ∃ m: ℕ, 5^k - 2^k = 3 * m) : 
  ∃ n: ℕ, 5^(k+1) - 2^(k+1) = 5 * (5^k - 2^k) + 3 * 2^k :=
by
  sorry

end induction_step_l135_135113


namespace mia_receives_chocolate_l135_135074

-- Given conditions
def total_chocolate : ℚ := 72 / 7
def piles : ℕ := 6
def piles_to_Mia : ℕ := 2

-- Weight of one pile
def weight_of_one_pile (total_chocolate : ℚ) (piles : ℕ) := total_chocolate / piles

-- Total weight Mia receives
def mia_chocolate (weight_of_one_pile : ℚ) (piles_to_Mia : ℕ) := piles_to_Mia * weight_of_one_pile

theorem mia_receives_chocolate : mia_chocolate (weight_of_one_pile total_chocolate piles) piles_to_Mia = 24 / 7 :=
by
  sorry

end mia_receives_chocolate_l135_135074


namespace difference_between_extrema_l135_135639

noncomputable def f (x a b : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x

theorem difference_between_extrema (a b : ℝ)
  (h1 : 3 * (2 : ℝ)^2 + 6 * a * (2 : ℝ) + 3 * b = 0)
  (h2 : 3 * (1 : ℝ)^2 + 6 * a * (1 : ℝ) + 3 * b = -3) :
  f 0 a b - f 2 a b = 4 :=
by
  sorry

end difference_between_extrema_l135_135639


namespace value_of_a_l135_135459

theorem value_of_a :
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - -1)^2 + (y - 1)^2 = 4) := sorry

end value_of_a_l135_135459


namespace distance_BC_l135_135952

variable (AC AB : ℝ) (angleACB : ℝ)
  (hAC : AC = 2)
  (hAB : AB = 3)
  (hAngle : angleACB = 120)

theorem distance_BC (BC : ℝ) : BC = Real.sqrt 6 - 1 :=
by
  sorry

end distance_BC_l135_135952


namespace price_of_third_variety_l135_135594

-- Define the given conditions
def price1 : ℝ := 126
def price2 : ℝ := 135
def average_price : ℝ := 153
def ratio1 : ℝ := 1
def ratio2 : ℝ := 1
def ratio3 : ℝ := 2

-- Define the total ratio
def total_ratio : ℝ := ratio1 + ratio2 + ratio3

-- Define the equation based on the given conditions
def weighted_avg_price (P : ℝ) : Prop :=
  (ratio1 * price1 + ratio2 * price2 + ratio3 * P) / total_ratio = average_price

-- Statement of the proof
theorem price_of_third_variety :
  ∃ P : ℝ, weighted_avg_price P ∧ P = 175.5 :=
by {
  -- Proof omitted
  sorry
}

end price_of_third_variety_l135_135594


namespace volume_water_needed_l135_135566

noncomputable def radius_sphere : ℝ := 0.5
noncomputable def radius_cylinder : ℝ := 1
noncomputable def height_cylinder : ℝ := 2

theorem volume_water_needed :
  let volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3)
  let total_volume_spheres := 4 * volume_sphere
  let volume_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder
  volume_cylinder - total_volume_spheres = (4 * Real.pi) / 3 :=
by
  let volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3)
  let total_volume_spheres := 4 * volume_sphere
  let volume_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder
  have h : volume_cylinder - total_volume_spheres = (4 * Real.pi) / 3 := sorry
  exact h

end volume_water_needed_l135_135566


namespace find_price_of_100_apples_l135_135714

noncomputable def price_of_100_apples (P : ℕ) : Prop :=
  (12000 / P) - (12000 / (P + 4)) = 5

theorem find_price_of_100_apples : price_of_100_apples 96 :=
by
  sorry

end find_price_of_100_apples_l135_135714


namespace even_function_k_value_l135_135958

theorem even_function_k_value (k : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = k * x^2 + (k - 1) * x + 2)
  (even_f : ∀ x : ℝ, f x = f (-x)) : k = 1 :=
by
  -- Proof would go here
  sorry

end even_function_k_value_l135_135958


namespace no_solution_l135_135801

noncomputable def problem_statement : Prop :=
  ∀ x : ℝ, ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x)))

theorem no_solution : problem_statement :=
by
  intro x
  have h₁ : ¬(85 + x = 3.5 * (15 + x)) -> ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x))) := sorry
  have h₂ : ¬(55 + x = 2 * (15 + x)) -> ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x))) := sorry
  exact sorry

end no_solution_l135_135801


namespace goldfish_initial_count_l135_135011

theorem goldfish_initial_count (catsfish : ℕ) (fish_left : ℕ) (fish_disappeared : ℕ) (goldfish_initial : ℕ) :
  catsfish = 12 →
  fish_left = 15 →
  fish_disappeared = 4 →
  goldfish_initial = (fish_left + fish_disappeared) - catsfish →
  goldfish_initial = 7 :=
by
  intros h1 h2 h3 h4
  rw [h2, h3, h1] at h4
  exact h4

end goldfish_initial_count_l135_135011


namespace max_abs_diff_f_l135_135789

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 * Real.exp x

theorem max_abs_diff_f (k : ℝ) (h₁ : -3 ≤ k) (h₂ : k ≤ -1) (x₁ x₂ : ℝ) (h₃ : k ≤ x₁) (h₄ : x₁ ≤ k + 2) (h₅ : k ≤ x₂) (h₆ : x₂ ≤ k + 2) :
  |f x₁ - f x₂| ≤ 4 * Real.exp 1 := sorry

end max_abs_diff_f_l135_135789


namespace area_larger_sphere_l135_135944

noncomputable def sphere_area_relation (A1: ℝ) (R1 R2: ℝ) := R2^2 / R1^2 * A1

-- Given Conditions
def radius_smaller_sphere : ℝ := 4.0  -- R1
def radius_larger_sphere : ℝ := 6.0    -- R2
def area_smaller_sphere : ℝ := 17.0    -- A1

-- Target Area Calculation based on Proportional Relationship
theorem area_larger_sphere :
  sphere_area_relation area_smaller_sphere radius_smaller_sphere radius_larger_sphere = 38.25 :=
by
  sorry

end area_larger_sphere_l135_135944


namespace coastal_village_population_l135_135347

variable (N : ℕ) (k : ℕ) (parts_for_males : ℕ) (total_males : ℕ)

theorem coastal_village_population 
  (h_total_population : N = 540)
  (h_division : k = 4)
  (h_parts_for_males : parts_for_males = 2)
  (h_total_males : total_males = (N / k) * parts_for_males) :
  total_males = 270 := 
by
  sorry

end coastal_village_population_l135_135347


namespace tan_seven_pi_over_four_l135_135443

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 := 
by
  -- In this case, we are proving a specific trigonometric identity
  sorry

end tan_seven_pi_over_four_l135_135443


namespace intersection_M_N_l135_135379

def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | (x ∈ U) ∧ ¬(x ∈ complement_U_N)}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
sorry

end intersection_M_N_l135_135379


namespace dk_is_odd_l135_135993

def NTypePermutations (k : ℕ) (x : Fin (3 * k + 1) → ℕ) : Prop :=
  (∀ i j : Fin (k + 1), i < j → x i < x j) ∧
  (∀ i j : Fin (k + 1), i < j → x (k + 1 + i) > x (k + 1 + j)) ∧
  (∀ i j : Fin (k + 1), i < j → x (2 * k + 1 + i) < x (2 * k + 1 + j))

def countNTypePermutations (k : ℕ) : ℕ :=
  sorry -- This would be the count of all N-type permutations, use advanced combinatorics or algorithms

theorem dk_is_odd (k : ℕ) (h : 0 < k) : ∃ d : ℕ, countNTypePermutations k = 2 * d + 1 :=
  sorry

end dk_is_odd_l135_135993


namespace problem_solution_l135_135537

def satisfies_conditions (x y : ℚ) : Prop :=
  (3 * x + y = 6) ∧ (x + 3 * y = 6)

theorem problem_solution :
  ∃ (x y : ℚ), satisfies_conditions x y ∧ 3 * x^2 + 5 * x * y + 3 * y^2 = 24.75 :=
by
  sorry

end problem_solution_l135_135537


namespace fraction_of_90_l135_135241

theorem fraction_of_90 : (1 / 2) * (1 / 3) * (1 / 6) * (90 : ℝ) = (5 / 2) := by
  sorry

end fraction_of_90_l135_135241


namespace total_repairs_cost_eq_l135_135983

-- Assume the initial cost of the scooter is represented by a real number C.
variable (C : ℝ)

-- Given conditions
def spent_on_first_repair := 0.05 * C
def spent_on_second_repair := 0.10 * C
def spent_on_third_repair := 0.07 * C

-- Total repairs expenditure
def total_repairs := spent_on_first_repair C + spent_on_second_repair C + spent_on_third_repair C

-- Selling price and profit
def selling_price := 1.25 * C
def profit := 1500
def profit_calc := selling_price C - (C + total_repairs C)

-- Statement to be proved: The total repairs is equal to $11,000.
theorem total_repairs_cost_eq : total_repairs 50000 = 11000 := by
  sorry

end total_repairs_cost_eq_l135_135983


namespace solve_equation_l135_135273

theorem solve_equation : ∀ x : ℝ, 2 * (3 * x - 1) = 7 - (x - 5) → x = 2 :=
by
  intro x h
  sorry

end solve_equation_l135_135273


namespace find_weight_A_l135_135436

noncomputable def weight_of_A (a b c d e : ℕ) : Prop :=
  (a + b + c) / 3 = 84 ∧
  (a + b + c + d) / 4 = 80 ∧
  e = d + 5 ∧
  (b + c + d + e) / 4 = 79 →
  a = 77

theorem find_weight_A (a b c d e : ℕ) : weight_of_A a b c d e :=
by
  sorry

end find_weight_A_l135_135436


namespace inequality_solution_l135_135158

theorem inequality_solution (x : ℝ) :
  (3 / 20 + |x - 13 / 60| < 7 / 30) ↔ (2 / 15 < x ∧ x < 3 / 10) :=
sorry

end inequality_solution_l135_135158


namespace river_depth_difference_l135_135919

theorem river_depth_difference
  (mid_may_depth : ℕ)
  (mid_july_depth : ℕ)
  (mid_june_depth : ℕ)
  (H1 : mid_july_depth = 45)
  (H2 : mid_may_depth = 5)
  (H3 : 3 * mid_june_depth = mid_july_depth) :
  mid_june_depth - mid_may_depth = 10 := 
sorry

end river_depth_difference_l135_135919


namespace complex_number_imaginary_axis_l135_135318

theorem complex_number_imaginary_axis (a : ℝ) : (a^2 - 2 * a = 0) → (a = 0 ∨ a = 2) :=
by
  sorry

end complex_number_imaginary_axis_l135_135318


namespace smallest_four_digit_divisible_by_53_l135_135105

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l135_135105


namespace calculate_expression_l135_135765

theorem calculate_expression :
  500 * 996 * 0.0996 * 20 + 5000 = 997016 :=
by
  sorry

end calculate_expression_l135_135765


namespace jill_speed_downhill_l135_135481

theorem jill_speed_downhill 
  (up_speed : ℕ) (total_time : ℕ) (hill_distance : ℕ) 
  (up_time : ℕ) (down_time : ℕ) (down_speed : ℕ) 
  (h1 : up_speed = 9)
  (h2 : total_time = 175)
  (h3 : hill_distance = 900)
  (h4 : up_time = hill_distance / up_speed)
  (h5 : down_time = total_time - up_time)
  (h6 : down_speed = hill_distance / down_time) :
  down_speed = 12 := 
  by
    sorry

end jill_speed_downhill_l135_135481


namespace math_proof_problem_l135_135985

theorem math_proof_problem
  (a b c : ℝ)
  (h : a ≠ b)
  (h1 : b ≠ c)
  (h2 : c ≠ a)
  (h3 : (a / (2 * (b - c))) + (b / (2 * (c - a))) + (c / (2 * (a - b))) = 0) :
  (a / (b - c)^3) + (b / (c - a)^3) + (c / (a - b)^3) = 0 := 
by
  sorry

end math_proof_problem_l135_135985


namespace find_value_of_expression_l135_135372

theorem find_value_of_expression
  (a b c m : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = m)
  (h5 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2 * a)^2 + b * (m - 2 * b)^2 + c * (m - 2 * c)^2) / (a * b * c) = 12 := 
sorry

end find_value_of_expression_l135_135372


namespace squares_difference_l135_135784

theorem squares_difference (n : ℕ) (h : n > 0) : (2 * n + 1)^2 - (2 * n - 1)^2 = 8 * n := 
by 
  sorry

end squares_difference_l135_135784


namespace point_translation_l135_135197

theorem point_translation :
  ∃ (x_old y_old x_new y_new : ℤ),
  (x_old = 1 ∧ y_old = -2) ∧
  (x_new = x_old + 2) ∧
  (y_new = y_old + 3) ∧
  (x_new = 3) ∧
  (y_new = 1) :=
sorry

end point_translation_l135_135197


namespace product_of_0_5_and_0_8_l135_135127

theorem product_of_0_5_and_0_8 : (0.5 * 0.8) = 0.4 := by
  sorry

end product_of_0_5_and_0_8_l135_135127


namespace compare_abc_l135_135515

noncomputable def a : ℝ := 2 * Real.log (21 / 20)
noncomputable def b : ℝ := Real.log (11 / 10)
noncomputable def c : ℝ := Real.sqrt 1.2 - 1

theorem compare_abc : a > b ∧ b < c ∧ a > c :=
by {
  sorry
}

end compare_abc_l135_135515


namespace solve_quadratic1_solve_quadratic2_l135_135165

theorem solve_quadratic1 (x : ℝ) :
  x^2 - 4 * x - 7 = 0 →
  (x = 2 - Real.sqrt 11) ∨ (x = 2 + Real.sqrt 11) :=
by
  sorry

theorem solve_quadratic2 (x : ℝ) :
  (x - 3)^2 + 2 * (x - 3) = 0 →
  (x = 3) ∨ (x = 1) :=
by
  sorry

end solve_quadratic1_solve_quadratic2_l135_135165


namespace find_side_length_l135_135397

theorem find_side_length
  (a b : ℝ)
  (S : ℝ)
  (h1 : a = 4)
  (h2 : b = 5)
  (h3 : S = 5 * Real.sqrt 3) :
  ∃ c : ℝ, c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
by
  sorry

end find_side_length_l135_135397


namespace seats_in_16th_row_l135_135860

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem seats_in_16th_row : arithmetic_sequence 5 2 16 = 35 := by
  sorry

end seats_in_16th_row_l135_135860


namespace existence_of_solution_largest_unsolvable_n_l135_135509

-- Definitions based on the conditions provided in the problem
def equation (x y z n : ℕ) : Prop := 28 * x + 30 * y + 31 * z = n

-- There exist positive integers x, y, z such that 28x + 30y + 31z = 365
theorem existence_of_solution : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z 365 :=
by
  sorry

-- The largest positive integer n such that 28x + 30y + 31z = n cannot be solved in positive integers x, y, z is 370
theorem largest_unsolvable_n : ∀ (n : ℕ), (∀ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 → n ≠ 370) → ∀ (n' : ℕ), n' > 370 → (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z n') :=
by
  sorry

end existence_of_solution_largest_unsolvable_n_l135_135509


namespace melanie_marbles_l135_135621

noncomputable def melanie_blue_marbles : ℕ :=
  let sandy_dozen_marbles := 56
  let dozen := 12
  let sandy_marbles := sandy_dozen_marbles * dozen
  let ratio := 8
  sandy_marbles / ratio

theorem melanie_marbles (h1 : ∀ sandy_dozen_marbles dozen ratio, 56 = sandy_dozen_marbles ∧ sandy_dozen_marbles * dozen = 672 ∧ ratio = 8) : melanie_blue_marbles = 84 := by
  sorry

end melanie_marbles_l135_135621


namespace box_growth_factor_l135_135093

/-
Problem: When a large box in the shape of a cuboid measuring 6 centimeters (cm) wide,
4 centimeters (cm) long, and 1 centimeters (cm) high became larger into a volume of
30 centimeters (cm) wide, 20 centimeters (cm) long, and 5 centimeters (cm) high,
find how many times it has grown.
-/

def original_box_volume (w l h : ℕ) : ℕ := w * l * h
def larger_box_volume (w l h : ℕ) : ℕ := w * l * h

theorem box_growth_factor :
  original_box_volume 6 4 1 * 125 = larger_box_volume 30 20 5 :=
by
  -- Proof goes here
  sorry

end box_growth_factor_l135_135093


namespace find_k_values_l135_135722

theorem find_k_values (k : ℝ) : 
  ((2 * 1 + 3 * k = 0) ∨
   (1 * 2 + (3 - k) * 3 = 0) ∨
   (1 * 1 + (3 - k) * k = 0)) →
   (k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 3)/2 ∨ k = (3 - Real.sqrt 3)/2) := 
by
  sorry

end find_k_values_l135_135722


namespace angle_C_length_CD_area_range_l135_135708

-- 1. Prove C = π / 3 given (2a - b)cos C = c cos B
theorem angle_C (a b c : ℝ) (A B C : ℝ) (h : (2 * a - b) * Real.cos C = c * Real.cos B) : 
  C = Real.pi / 3 := sorry

-- 2. Prove the length of CD is 6√3 / 5 given a = 2, b = 3, and CD is the angle bisector of angle C
theorem length_CD (a b x : ℝ) (C D : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : x = (6 * Real.sqrt 3) / 5) : 
  x = (6 * Real.sqrt 3) / 5 := sorry

-- 3. Prove the range of values for the area of acute triangle ABC is (8√3 / 3, 4√3] given a cos B + b cos A = 4
theorem area_range (a b : ℝ) (A B C : ℝ) (S : Set ℝ) (h1 : a * Real.cos B + b * Real.cos A = 4) 
  (h2 : S = Set.Ioc (8 * Real.sqrt 3 / 3) (4 * Real.sqrt 3)) : 
  S = Set.Ioc (8 * Real.sqrt 3 / 3) (4 * Real.sqrt 3) := sorry

end angle_C_length_CD_area_range_l135_135708


namespace collinear_probability_correct_l135_135035

def number_of_dots := 25

def number_of_four_dot_combinations := Nat.choose number_of_dots 4

-- Calculate the different possibilities for collinear sets:
def horizontal_sets := 5 * 5
def vertical_sets := 5 * 5
def diagonal_sets := 2 + 2

def total_collinear_sets := horizontal_sets + vertical_sets + diagonal_sets

noncomputable def probability_collinear : ℚ :=
  total_collinear_sets / number_of_four_dot_combinations

theorem collinear_probability_correct :
  probability_collinear = 6 / 1415 :=
sorry

end collinear_probability_correct_l135_135035


namespace team_C_has_most_uniform_height_l135_135674

theorem team_C_has_most_uniform_height
  (S_A S_B S_C S_D : ℝ)
  (h_A : S_A = 0.13)
  (h_B : S_B = 0.11)
  (h_C : S_C = 0.09)
  (h_D : S_D = 0.15)
  (h_same_num_members : ∀ (a b c d : ℕ), a = b ∧ b = c ∧ c = d) 
  : S_C = min S_A (min S_B (min S_C S_D)) :=
by
  sorry

end team_C_has_most_uniform_height_l135_135674


namespace arithmetic_sequence_sum_l135_135426

noncomputable def first_21_sum (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ) : ℝ :=
  let a1 := a 1
  let a21 := a 21
  21 * (a1 + a21) / 2

theorem arithmetic_sequence_sum
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_symmetry : ∀ x, f (x + 1) = f (-(x + 1)))
  (h_monotonic : ∀ x y, 1 < x → x < y → f x < f y)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_f_eq : f (a 4) = f (a 18))
  (h_non_zero_diff : d ≠ 0) :
  first_21_sum f a d = 21 := by
  sorry

end arithmetic_sequence_sum_l135_135426


namespace a_eq_zero_l135_135620

noncomputable def f (x a : ℝ) := x^2 - abs (x + a)

theorem a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end a_eq_zero_l135_135620


namespace gcf_factor_l135_135916

theorem gcf_factor (x y : ℕ) : gcd (6 * x ^ 3 * y ^ 2) (3 * x ^ 2 * y ^ 3) = 3 * x ^ 2 * y ^ 2 :=
by
  sorry

end gcf_factor_l135_135916


namespace min_correct_answers_l135_135060

theorem min_correct_answers (x : ℕ) : 
  (∃ x, 0 ≤ x ∧ x ≤ 20 ∧ 5 * x - (20 - x) ≥ 88) :=
sorry

end min_correct_answers_l135_135060


namespace no_solutions_for_divisibility_by_3_l135_135342

theorem no_solutions_for_divisibility_by_3 (x y : ℤ) : ¬ (x^2 + y^2 + x + y ∣ 3) :=
sorry

end no_solutions_for_divisibility_by_3_l135_135342


namespace larry_win_probability_correct_l135_135243

/-- Define the probabilities of knocking off the bottle for both players in the first four turns. -/
structure GameProb (turns : ℕ) :=
  (larry_prob : ℚ)
  (julius_prob : ℚ)

/-- Define the probabilities of knocking off the bottle for both players from the fifth turn onwards. -/
def subsequent_turns_prob : ℚ := 1 / 2
/-- Initial probabilities for the first four turns -/
def initial_prob : GameProb 4 := { larry_prob := 2 / 3, julius_prob := 1 / 3 }
/-- The probability that Larry wins the game -/
def larry_wins (prob : GameProb 4) (subsequent_prob : ℚ) : ℚ :=
  -- Calculation logic goes here resulting in the final probability
  379 / 648

theorem larry_win_probability_correct :
  larry_wins initial_prob subsequent_turns_prob = 379 / 648 :=
sorry

end larry_win_probability_correct_l135_135243


namespace carol_rectangle_length_l135_135496

theorem carol_rectangle_length (lCarol : ℝ) :
    (∃ (wCarol : ℝ), wCarol = 20 ∧ lCarol * wCarol = 300) ↔ lCarol = 15 :=
by
  have jordan_area : 6 * 50 = 300 := by norm_num
  sorry

end carol_rectangle_length_l135_135496


namespace meal_cost_l135_135345

theorem meal_cost (x : ℝ) (h1 : ∀ (x : ℝ), (x / 4) - 6 = x / 9) : 
  x = 43.2 :=
by
  have h : (∀ (x : ℝ), (x / 4) - (x / 9) = 6) := sorry
  exact sorry

end meal_cost_l135_135345


namespace average_marks_in_6_subjects_l135_135024

/-- The average marks Ashok secured in 6 subjects is 72
Given:
1. The average of marks in 5 subjects is 74.
2. Ashok secured 62 marks in the 6th subject.
-/
theorem average_marks_in_6_subjects (avg_5 : ℕ) (marks_6th : ℕ) (h_avg_5 : avg_5 = 74) (h_marks_6th : marks_6th = 62) : 
  ((avg_5 * 5 + marks_6th) / 6) = 72 :=
  by
  sorry

end average_marks_in_6_subjects_l135_135024


namespace arithmetic_sequence_common_difference_l135_135282

theorem arithmetic_sequence_common_difference :
  ∃ d : ℝ, (∀ (a_n : ℕ → ℝ), a_n 1 = 3 ∧ a_n 3 = 7 ∧ (∀ n, a_n n = 3 + (n - 1) * d)) → d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l135_135282


namespace ellipse_equation_fixed_point_l135_135280

/-- Given an ellipse with equation x^2 / a^2 + y^2 / b^2 = 1 where a > b > 0 and eccentricity e = 1/2,
    prove that the equation of the ellipse is x^2 / 4 + y^2 / 3 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a^2 = b^2 + (a / 2)^2) :
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
by sorry

/-- Given an ellipse with equation x^2 / 4 + y^2 / 3 = 1,
    if a line l: y = kx + m intersects the ellipse at two points A and B (which are not the left and right vertices),
    and a circle passing through the right vertex of the ellipse has AB as its diameter,
    prove that the line passes through a fixed point and find its coordinates -/
theorem fixed_point (k m : ℝ) :
  (∃ x y, (x = 2 / 7 ∧ y = 0)) :=
by sorry

end ellipse_equation_fixed_point_l135_135280


namespace prove_2x_plus_y_le_sqrt_11_l135_135689

variable (x y : ℝ)
variable (h : 3 * x^2 + 2 * y^2 ≤ 6)

theorem prove_2x_plus_y_le_sqrt_11 : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end prove_2x_plus_y_le_sqrt_11_l135_135689


namespace cos_value_l135_135464

theorem cos_value (α : ℝ) 
  (h1 : Real.sin (α + Real.pi / 12) = 1 / 3) : 
  Real.cos (α + 7 * Real.pi / 12) = -(1 + Real.sqrt 24) / 6 :=
sorry

end cos_value_l135_135464


namespace ski_helmet_final_price_l135_135660

variables (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ)
def final_price_after_discounts (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  let after_first_discount := initial_price * (1 - discount1)
  let after_second_discount := after_first_discount * (1 - discount2)
  after_second_discount

theorem ski_helmet_final_price :
  final_price_after_discounts 120 0.40 0.20 = 57.60 := 
  sorry

end ski_helmet_final_price_l135_135660


namespace positive_difference_of_solutions_is_14_l135_135778

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 5 * x + 15 = x + 55

-- Define the positive difference between solutions of the quadratic equation
def positive_difference (a b : ℝ) : ℝ := |a - b|

-- State the theorem
theorem positive_difference_of_solutions_is_14 : 
  ∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ positive_difference a b = 14 :=
by
  sorry

end positive_difference_of_solutions_is_14_l135_135778


namespace find_x_l135_135402

noncomputable def approx_equal (a b : ℝ) (ε : ℝ) : Prop := abs (a - b) < ε

theorem find_x :
  ∃ x : ℝ, x + Real.sqrt 68 = 24 ∧ approx_equal x 15.753788749 0.0001 :=
sorry

end find_x_l135_135402


namespace bobby_candy_left_l135_135887

theorem bobby_candy_left (initial_candies := 21) (first_eaten := 5) (second_eaten := 9) : 
  initial_candies - first_eaten - second_eaten = 7 :=
by
  -- Proof goes here
  sorry

end bobby_candy_left_l135_135887


namespace number_of_cars_on_street_l135_135790

-- Definitions based on conditions
def cars_equally_spaced (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 5.5

def distance_between_first_and_last_car (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 242

def distance_between_cars (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 5.5

-- Given all conditions, prove n = 45
theorem number_of_cars_on_street (n : ℕ) :
  cars_equally_spaced n →
  distance_between_first_and_last_car n →
  distance_between_cars n →
  n = 45 :=
sorry

end number_of_cars_on_street_l135_135790


namespace find_evening_tickets_l135_135723

noncomputable def matinee_price : ℕ := 5
noncomputable def evening_price : ℕ := 12
noncomputable def threeD_price : ℕ := 20
noncomputable def matinee_tickets : ℕ := 200
noncomputable def threeD_tickets : ℕ := 100
noncomputable def total_revenue : ℕ := 6600

theorem find_evening_tickets (E : ℕ) (hE : total_revenue = matinee_tickets * matinee_price + E * evening_price + threeD_tickets * threeD_price) :
  E = 300 :=
by
  sorry

end find_evening_tickets_l135_135723


namespace problem_statement_l135_135960

variable {F : Type*} [Field F]

theorem problem_statement (m : F) (h : m + 1 / m = 6) : m^2 + 1 / m^2 + 4 = 38 :=
by
  sorry

end problem_statement_l135_135960


namespace inverse_function_log_l135_135267

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x

theorem inverse_function_log (a : ℝ) (g : ℝ → ℝ) (x : ℝ) (y : ℝ) :
  (a > 0) → (a ≠ 1) → 
  (f 2 a = 4) → 
  (f y a = x) → 
  (g x = y) → 
  g x = Real.logb 2 x := 
by
  intros ha hn hfx hfy hg
  sorry

end inverse_function_log_l135_135267


namespace remainder_86592_8_remainder_8741_13_l135_135383

theorem remainder_86592_8 :
  86592 % 8 = 0 :=
by
  sorry

theorem remainder_8741_13 :
  8741 % 13 = 5 :=
by
  sorry

end remainder_86592_8_remainder_8741_13_l135_135383


namespace no_integer_solution_for_expression_l135_135498

theorem no_integer_solution_for_expression (x y z : ℤ) :
  x^4 + y^4 + z^4 - 2 * x^2 * y^2 - 2 * y^2 * z^2 - 2 * z^2 * x^2 ≠ 2000 :=
by sorry

end no_integer_solution_for_expression_l135_135498


namespace find_value_of_m_l135_135269

theorem find_value_of_m : ∃ m : ℤ, 2^4 - 3 = 5^2 + m ∧ m = -12 :=
by
  use -12
  sorry

end find_value_of_m_l135_135269


namespace pants_cost_correct_l135_135617

-- Define the conditions as variables
def initial_money : ℕ := 71
def shirt_cost : ℕ := 5
def num_shirts : ℕ := 5
def remaining_money : ℕ := 20

-- Define intermediates necessary to show the connection between conditions and the question
def money_spent_on_shirts : ℕ := num_shirts * shirt_cost
def money_left_after_shirts : ℕ := initial_money - money_spent_on_shirts
def pants_cost : ℕ := money_left_after_shirts - remaining_money

-- The main theorem to prove the question is equal to the correct answer
theorem pants_cost_correct : pants_cost = 26 :=
by
  sorry

end pants_cost_correct_l135_135617


namespace total_kilometers_ridden_l135_135450

theorem total_kilometers_ridden :
  ∀ (d1 d2 d3 d4 : ℕ),
    d1 = 40 →
    d2 = 50 →
    d3 = d2 - d2 / 2 →
    d4 = d1 + d3 →
    d1 + d2 + d3 + d4 = 180 :=
by 
  intros d1 d2 d3 d4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_kilometers_ridden_l135_135450


namespace seq_arithmetic_l135_135133

noncomputable def f (x n : ℝ) : ℝ := (x - 1)^2 + n

def a_n (n : ℝ) : ℝ := n
def b_n (n : ℝ) : ℝ := n + 4
def c_n (n : ℝ) : ℝ := (b_n n)^2 - (a_n n) * (b_n n)

theorem seq_arithmetic (n : ℕ) (hn : 0 < n) :
  ∃ d, d ≠ 0 ∧ ∀ n, c_n (↑n : ℝ) = c_n (↑n + 1 : ℝ) - d := 
sorry

end seq_arithmetic_l135_135133


namespace cannot_be_zero_l135_135982

noncomputable def P (x : ℝ) (a b c d e : ℝ) := x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem cannot_be_zero (a b c d e : ℝ) (p q r s : ℝ) :
  e = 0 ∧ c = 0 ∧ (∀ x, P x a b c d e = x * (x - p) * (x - q) * (x - r) * (x - s)) ∧ 
  (p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  d ≠ 0 := 
by {
  sorry
}

end cannot_be_zero_l135_135982


namespace a_seq_formula_T_seq_sum_l135_135861

-- Definition of the sequence \( \{a_n\} \)
def a_seq (n : ℕ) (p : ℤ) : ℤ := 2 * n + 5

-- Condition: Sum of the first n terms \( s_n = n^2 + pn \)
def s_seq (n : ℕ) (p : ℤ) : ℤ := n^2 + p * n

-- Condition: \( \{a_2, a_5, a_{10}\} \) form a geometric sequence
def is_geometric (a2 a5 a10 : ℤ) : Prop :=
  a2 * a10 = a5 * a5

-- Definition of the sequence \( \{b_n\} \)
def b_seq (n : ℕ) (p : ℤ) : ℚ := 1 + 5 / (a_seq n p * a_seq (n + 1) p)

-- Function to find the sum of first n terms of \( \{b_n\} \)
def T_seq (n : ℕ) (p : ℤ) : ℚ :=
  n + 5 * (1 / (7 : ℚ) - 1 / (2 * n + 7 : ℚ)) + n / (14 * n + 49 : ℚ)

theorem a_seq_formula (p : ℤ) : ∀ n, a_seq n p = 2 * n + 5 :=
by
  sorry

theorem T_seq_sum (p : ℤ) : ∀ n, T_seq n p = (14 * n^2 + 54 * n) / (14 * n + 49) :=
by
  sorry

end a_seq_formula_T_seq_sum_l135_135861


namespace complement_union_l135_135917

def universal_set : Set ℝ := { x : ℝ | true }
def M : Set ℝ := { x : ℝ | x ≤ 0 }
def N : Set ℝ := { x : ℝ | x > 2 }

theorem complement_union (x : ℝ) :
  x ∈ compl (M ∪ N) ↔ (0 < x ∧ x ≤ 2) := by
  sorry

end complement_union_l135_135917


namespace find_common_ratio_l135_135649

variable {a : ℕ → ℝ}
variable (q : ℝ)

-- Definition of geometric sequence condition
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given conditions
def conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 2 + a 4 = 20) ∧ (a 3 + a 5 = 40)

-- Proposition to be proved
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geo : is_geometric_sequence a q) (h_cond : conditions a q) : q = 2 :=
by 
  sorry

end find_common_ratio_l135_135649


namespace gcf_7fact_8fact_l135_135847

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l135_135847


namespace ratio_female_to_male_l135_135135

namespace DeltaSportsClub

variables (f m : ℕ) -- number of female and male members
-- Sum of ages of female and male members respectively
def sum_ages_females := 35 * f
def sum_ages_males := 30 * m
-- Total sum of ages
def total_sum_ages := sum_ages_females f + sum_ages_males m
-- Total number of members
def total_members := f + m

-- Given condition on the average age of all members
def average_age_condition := (total_sum_ages f m) / (total_members f m) = 32

-- The target theorem to prove the ratio of female to male members
theorem ratio_female_to_male (h : average_age_condition f m) : f/m = 2/3 :=
by sorry

end DeltaSportsClub

end ratio_female_to_male_l135_135135


namespace total_selling_price_l135_135331

theorem total_selling_price (cost_price_per_metre profit_per_metre : ℝ)
  (total_metres_sold : ℕ) :
  cost_price_per_metre = 58.02564102564102 → 
  profit_per_metre = 29 → 
  total_metres_sold = 78 →
  (cost_price_per_metre + profit_per_metre) * total_metres_sold = 6788 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  -- backend calculation, checking computation level;
  sorry

end total_selling_price_l135_135331


namespace right_triangle_sides_l135_135229

theorem right_triangle_sides {a b c : ℕ} (h1 : a * (b + 2) = 150) (h2 : a^2 + b^2 = c^2) (h3 : a + (1 / 2 : ℤ) * (a * b) = 75) :
  (a = 6 ∧ b = 23 ∧ c = 25) ∨ (a = 15 ∧ b = 8 ∧ c = 17) :=
sorry

end right_triangle_sides_l135_135229


namespace min_n_of_inequality_l135_135904

theorem min_n_of_inequality : 
  ∀ (n : ℕ), (1 ≤ n) → (1 / n - 1 / (n + 1) < 1 / 10) → (n = 3 ∨ ∃ (k : ℕ), k ≥ 3 ∧ n = k) :=
by
  sorry

end min_n_of_inequality_l135_135904


namespace subset_M_P_N_l135_135955

def setM : Set (ℝ × ℝ) := {p | |p.1| + |p.2| < 1}

def setN : Set (ℝ × ℝ) := 
  {p | (Real.sqrt ((p.1 - 1 / 2) ^ 2 + (p.2 + 1 / 2) ^ 2) + Real.sqrt ((p.1 + 1 / 2) ^ 2 + (p.2 - 1 / 2) ^ 2)) < 2 * Real.sqrt 2}

def setP : Set (ℝ × ℝ) := 
  {p | |p.1 + p.2| < 1 ∧ |p.1| < 1 ∧ |p.2| < 1}

theorem subset_M_P_N : setM ⊆ setP ∧ setP ⊆ setN := by
  sorry

end subset_M_P_N_l135_135955


namespace factorize_difference_of_squares_l135_135653

theorem factorize_difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) :=
sorry

end factorize_difference_of_squares_l135_135653


namespace selection_methods_l135_135827

/-- Type definition for the workers -/
inductive Worker
  | PliersOnly  : Worker
  | CarOnly     : Worker
  | Both        : Worker

/-- Conditions -/
def num_workers : ℕ := 11
def num_pliers_only : ℕ := 5
def num_car_only : ℕ := 4
def num_both : ℕ := 2
def pliers_needed : ℕ := 4
def car_needed : ℕ := 4

/-- Main statement -/
theorem selection_methods : 
  (num_pliers_only + num_car_only + num_both = num_workers) → 
  (num_pliers_only = 5) → 
  (num_car_only = 4) → 
  (num_both = 2) → 
  (pliers_needed = 4) → 
  (car_needed = 4) → 
  ∃ n : ℕ, n = 185 := 
by 
  sorry -- Proof Skipped

end selection_methods_l135_135827


namespace evaluate_polynomial_l135_135625

noncomputable def polynomial_evaluation : Prop :=
∀ (x : ℝ), x^2 - 3*x - 9 = 0 ∧ 0 < x → (x^4 - 3*x^3 - 9*x^2 + 27*x - 8) = (65 + 81*(Real.sqrt 5))/2

theorem evaluate_polynomial : polynomial_evaluation :=
sorry

end evaluate_polynomial_l135_135625


namespace soda_consumption_l135_135988

theorem soda_consumption 
    (dozens : ℕ)
    (people_per_dozen : ℕ)
    (cost_per_box : ℕ)
    (cans_per_box : ℕ)
    (family_members : ℕ)
    (payment_per_member : ℕ)
    (dozens_eq : dozens = 5)
    (people_per_dozen_eq : people_per_dozen = 12)
    (cost_per_box_eq : cost_per_box = 2)
    (cans_per_box_eq : cans_per_box = 10)
    (family_members_eq : family_members = 6)
    (payment_per_member_eq : payment_per_member = 4) :
  (60 * (cans_per_box)) / 60 = 2 :=
by
  -- proof would go here eventually
  sorry

end soda_consumption_l135_135988


namespace poly_remainder_l135_135658

theorem poly_remainder (x : ℤ) :
  (x^1012) % (x^3 - x^2 + x - 1) = 1 := by
  sorry

end poly_remainder_l135_135658


namespace power_addition_l135_135466

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l135_135466


namespace expand_polynomial_l135_135805

theorem expand_polynomial :
  (2 * t^2 - 3 * t + 2) * (-3 * t^2 + t - 5) = -6 * t^4 + 11 * t^3 - 19 * t^2 + 17 * t - 10 :=
by sorry

end expand_polynomial_l135_135805


namespace prime_iff_totient_divisor_sum_l135_135647

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def euler_totient (n : ℕ) : ℕ := sorry  -- we assume implementation of Euler's Totient function
def divisor_sum (n : ℕ) : ℕ := sorry  -- we assume implementation of Divisor sum function

theorem prime_iff_totient_divisor_sum (n : ℕ) :
  (2 ≤ n) → (euler_totient n ∣ (n - 1)) → (n + 1 ∣ divisor_sum n) → is_prime n :=
  sorry

end prime_iff_totient_divisor_sum_l135_135647


namespace evaluate_f_of_composed_g_l135_135864

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x + 2

theorem evaluate_f_of_composed_g :
  f (2 + g 3) = 17 :=
by
  sorry

end evaluate_f_of_composed_g_l135_135864


namespace colbert_materials_needed_l135_135736

def wooden_planks_needed (total_needed quarter_in_stock : ℕ) : ℕ :=
  let total_purchased := total_needed - quarter_in_stock / 4
  (total_purchased + 7) / 8 -- ceil division by 8

def iron_nails_needed (total_needed thirty_percent_provided : ℕ) : ℕ :=
  let total_purchased := total_needed - total_needed * thirty_percent_provided / 100
  (total_purchased + 24) / 25 -- ceil division by 25

def fabric_needed (total_needed third_provided : ℚ) : ℚ :=
  total_needed - total_needed / third_provided

def metal_brackets_needed (total_needed in_stock multiple : ℕ) : ℕ :=
  let total_purchased := total_needed - in_stock
  (total_purchased + multiple - 1) / multiple * multiple -- ceil to next multiple of 5

theorem colbert_materials_needed :
  wooden_planks_needed 250 62 = 24 ∧
  iron_nails_needed 500 30 = 14 ∧
  fabric_needed 10 3 = 6.67 ∧
  metal_brackets_needed 40 10 5 = 30 :=
by sorry

end colbert_materials_needed_l135_135736


namespace intersection_eq_l135_135529

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≤ 2}

theorem intersection_eq : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_eq_l135_135529


namespace sequence_8th_term_is_sqrt23_l135_135762

noncomputable def sequence_term (n : ℕ) : ℝ := Real.sqrt (2 + 3 * (n - 1))

theorem sequence_8th_term_is_sqrt23 : sequence_term 8 = Real.sqrt 23 :=
by
  sorry

end sequence_8th_term_is_sqrt23_l135_135762


namespace binomial_22_5_computation_l135_135542

theorem binomial_22_5_computation (h1 : Nat.choose 20 3 = 1140) (h2 : Nat.choose 20 4 = 4845) (h3 : Nat.choose 20 5 = 15504) :
    Nat.choose 22 5 = 26334 := by
  sorry

end binomial_22_5_computation_l135_135542


namespace B_contains_only_one_element_l135_135096

def setA := { x | (x - 1/2) * (x - 3) = 0 }

def setB (a : ℝ) := { x | Real.log (x^2 + a * x + a + 9 / 4) = 0 }

theorem B_contains_only_one_element (a : ℝ) :
  (∃ x, setB a x ∧ ∀ y, setB a y → y = x) →
  (a = 5 ∨ a = -1) :=
by
  intro h
  -- Proof would go here
  sorry

end B_contains_only_one_element_l135_135096


namespace f_is_even_l135_135020

-- Given an odd function g
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g (x)

-- Define the function f as given by the problem
def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  abs (g (x^2))

-- The theorem stating that f is an even function
theorem f_is_even (g : ℝ → ℝ) (h_odd : is_odd_function g) : ∀ x, f g x = f g (-x) :=
by
  sorry

end f_is_even_l135_135020


namespace range_of_a_l135_135729

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x + x^2 - 2 * a * x

theorem range_of_a (a : ℝ) (h₀ : a > 0) 
  (h₁ h₂ : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ a - f x₂ a ≥ (3/2) - 2 * Real.log 2) : 
  a ≥ (3/2) * Real.sqrt 2 :=
sorry

end range_of_a_l135_135729


namespace grasshopper_flea_adjacency_l135_135168

-- Define the types of cells
inductive CellColor
| Red
| White

-- Define the infinite grid as a function from ℤ × ℤ to CellColor
def InfiniteGrid : Type := ℤ × ℤ → CellColor

-- Define the positions of the grasshopper and the flea
variables (g_start f_start : ℤ × ℤ)

-- The conditions for the grid and movement rules
axiom grid_conditions (grid : InfiniteGrid) :
  ∃ g_pos f_pos : ℤ × ℤ, 
  (g_pos = g_start ∧ f_pos = f_start) ∧
  (∀ x y : ℤ × ℤ, grid x = CellColor.Red ∨ grid x = CellColor.White) ∧
  (∀ x y : ℤ × ℤ, grid y = CellColor.Red ∨ grid y = CellColor.White)

-- Define the movement conditions for grasshopper and flea
axiom grasshopper_jumps (grid : InfiniteGrid) (start : ℤ × ℤ) :
  ∃ end_pos : ℤ × ℤ, grid end_pos = CellColor.Red ∧ ((end_pos.1 = start.1 ∨ end_pos.2 = start.2) ∧ abs (end_pos.1 - start.1) ≤ 1 ∧ abs (end_pos.2 - start.2) ≤ 1)

axiom flea_jumps (grid : InfiniteGrid) (start : ℤ × ℤ) :
  ∃ end_pos : ℤ × ℤ, grid end_pos = CellColor.White ∧ ((end_pos.1 = start.1 ∨ end_pos.2 = start.2) ∧ abs (end_pos.1 - start.1) ≤ 1 ∧ abs (end_pos.2 - start.2) ≤ 1)

-- The main theorem statement
theorem grasshopper_flea_adjacency (grid : InfiniteGrid)
    (g_start f_start : ℤ × ℤ) :
    ∃ pos1 pos2 pos3 : ℤ × ℤ,
    (pos1 = g_start ∨ pos1 = f_start) ∧ 
    (pos2 = g_start ∨ pos2 = f_start) ∧ 
    (abs (pos3.1 - g_start.1) + abs (pos3.2 - g_start.2) ≤ 1 ∧ 
    abs (pos3.1 - f_start.1) + abs (pos3.2 - f_start.2) ≤ 1) :=
sorry

end grasshopper_flea_adjacency_l135_135168


namespace rachel_reading_homework_l135_135051

theorem rachel_reading_homework (math_hw : ℕ) (additional_reading_hw : ℕ) (total_reading_hw : ℕ) 
  (h1 : math_hw = 8) (h2 : additional_reading_hw = 6) (h3 : total_reading_hw = math_hw + additional_reading_hw) :
  total_reading_hw = 14 :=
sorry

end rachel_reading_homework_l135_135051


namespace tate_total_years_proof_l135_135455

def highSchoolYears: ℕ := 4 - 1
def gapYear: ℕ := 2
def bachelorYears (highSchoolYears: ℕ): ℕ := 2 * highSchoolYears
def workExperience: ℕ := 1
def phdYears (highSchoolYears: ℕ) (bachelorYears: ℕ): ℕ := 3 * (highSchoolYears + bachelorYears)
def totalYears (highSchoolYears: ℕ) (gapYear: ℕ) (bachelorYears: ℕ) (workExperience: ℕ) (phdYears: ℕ): ℕ :=
  highSchoolYears + gapYear + bachelorYears + workExperience + phdYears

theorem tate_total_years_proof : totalYears highSchoolYears gapYear (bachelorYears highSchoolYears) workExperience (phdYears highSchoolYears (bachelorYears highSchoolYears)) = 39 := by
  sorry

end tate_total_years_proof_l135_135455


namespace find_years_in_future_l135_135984

theorem find_years_in_future 
  (S F : ℕ)
  (h1 : F = 4 * S + 4)
  (h2 : F = 44) :
  ∃ x : ℕ, F + x = 2 * (S + x) + 20 ∧ x = 4 :=
by 
  sorry

end find_years_in_future_l135_135984


namespace red_peaches_l135_135444

theorem red_peaches (R G : ℕ) (h1 : G = 11) (h2 : G = R + 6) : R = 5 :=
by {
  sorry
}

end red_peaches_l135_135444


namespace grape_juice_percentage_after_addition_l135_135308

def initial_mixture_volume : ℝ := 40
def initial_grape_juice_percentage : ℝ := 0.10
def added_grape_juice_volume : ℝ := 10

theorem grape_juice_percentage_after_addition :
  ((initial_mixture_volume * initial_grape_juice_percentage + added_grape_juice_volume) /
  (initial_mixture_volume + added_grape_juice_volume)) * 100 = 28 :=
by 
  sorry

end grape_juice_percentage_after_addition_l135_135308


namespace student_A_more_stable_l135_135785

-- Define the variances for students A and B
def variance_A : ℝ := 0.05
def variance_B : ℝ := 0.06

-- The theorem to prove that student A has more stable performance
theorem student_A_more_stable : variance_A < variance_B :=
by {
  -- proof goes here
  sorry
}

end student_A_more_stable_l135_135785


namespace smallest_n_condition_l135_135356

open Nat

-- Define the sum of squares formula
noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- Define the condition for being a square number
def is_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- The proof problem statement
theorem smallest_n_condition : 
  ∃ n : ℕ, n > 1 ∧ is_square (sum_of_squares n / n) ∧ (∀ m : ℕ, m > 1 ∧ is_square (sum_of_squares m / m) → n ≤ m) :=
sorry

end smallest_n_condition_l135_135356


namespace hanna_gives_roses_l135_135055

-- Conditions as Lean definitions
def initial_budget : ℕ := 300
def price_jenna : ℕ := 2
def price_imma : ℕ := 3
def price_ravi : ℕ := 4
def price_leila : ℕ := 5

def roses_for_jenna (budget : ℕ) : ℕ :=
  budget / price_jenna * 1 / 3

def roses_for_imma (budget : ℕ) : ℕ :=
  budget / price_imma * 1 / 4

def roses_for_ravi (budget : ℕ) : ℕ :=
  budget / price_ravi * 1 / 6

def roses_for_leila (budget : ℕ) : ℕ :=
  budget / price_leila

-- Calculations based on conditions
def roses_jenna : ℕ := Nat.floor (50 * 1/3)
def roses_imma : ℕ := Nat.floor ((100 / price_imma) * 1 / 4)
def roses_ravi : ℕ := Nat.floor ((50 / price_ravi) * 1 / 6)
def roses_leila : ℕ := 50 / price_leila

-- Final statement to be proven
theorem hanna_gives_roses :
  roses_jenna + roses_imma + roses_ravi + roses_leila = 36 := by
  sorry

end hanna_gives_roses_l135_135055


namespace intersection_M_N_l135_135767

def M := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def N := { y : ℝ | y > 0 }

theorem intersection_M_N : (M ∩ N) = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l135_135767


namespace orlando_weight_gain_l135_135606

def weight_gain_statement (x J F : ℝ) : Prop :=
  J = 2 * x + 2 ∧ F = 1/2 * J - 3 ∧ x + J + F = 20

theorem orlando_weight_gain :
  ∃ x J F : ℝ, weight_gain_statement x J F ∧ x = 5 :=
by {
  sorry
}

end orlando_weight_gain_l135_135606


namespace rational_relation_l135_135128

variable {a b : ℚ}

theorem rational_relation (h1 : a > 0) (h2 : b < 0) (h3 : |a| > |b|) : -a < -b ∧ -b < b ∧ b < a :=
by
  sorry

end rational_relation_l135_135128


namespace maria_trip_distance_l135_135747

theorem maria_trip_distance
  (D : ℝ)
  (h1 : D/2 = D/8 + 210) :
  D = 560 :=
sorry

end maria_trip_distance_l135_135747


namespace number_of_levels_l135_135286

theorem number_of_levels (total_capacity : ℕ) (additional_cars : ℕ) (already_parked_cars : ℕ) (n : ℕ) :
  total_capacity = 425 →
  additional_cars = 62 →
  already_parked_cars = 23 →
  n = total_capacity / (already_parked_cars + additional_cars) →
  n = 5 :=
by
  intros
  sorry

end number_of_levels_l135_135286


namespace blue_to_red_marble_ratio_l135_135440

-- Define the given conditions and the result.
theorem blue_to_red_marble_ratio (total_marble yellow_marble : ℕ) 
  (h1 : total_marble = 19)
  (h2 : yellow_marble = 5)
  (red_marble : ℕ)
  (h3 : red_marble = yellow_marble + 3) : 
  ∃ blue_marble : ℕ, (blue_marble = total_marble - (yellow_marble + red_marble)) 
  ∧ (blue_marble / (gcd blue_marble red_marble)) = 3 
  ∧ (red_marble / (gcd blue_marble red_marble)) = 4 :=
by {
  --existence of blue_marble and the ratio
  sorry
}

end blue_to_red_marble_ratio_l135_135440


namespace find_initial_number_l135_135681

-- Define the initial equation
def initial_equation (x : ℤ) : Prop := x - 12 * 3 * 2 = 9938

-- Prove that the initial number x is equal to 10010 given initial_equation
theorem find_initial_number (x : ℤ) (h : initial_equation x) : x = 10010 :=
sorry

end find_initial_number_l135_135681


namespace folded_segment_square_length_eq_225_div_4_l135_135153

noncomputable def square_of_fold_length : ℝ :=
  let side_length := 15
  let distance_from_B := 5
  (side_length ^ 2 - distance_from_B * (2 * side_length - distance_from_B)) / 4

theorem folded_segment_square_length_eq_225_div_4 :
  square_of_fold_length = 225 / 4 :=
by
  sorry

end folded_segment_square_length_eq_225_div_4_l135_135153


namespace officers_count_l135_135473

theorem officers_count (average_salary_all : ℝ) (average_salary_officers : ℝ) 
    (average_salary_non_officers : ℝ) (num_non_officers : ℝ) (total_salary : ℝ) : 
    average_salary_all = 120 → 
    average_salary_officers = 470 →  
    average_salary_non_officers = 110 → 
    num_non_officers = 525 → 
    total_salary = average_salary_all * (num_non_officers + O) → 
    total_salary = average_salary_officers * O + average_salary_non_officers * num_non_officers → 
    O = 15 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end officers_count_l135_135473


namespace elizabeth_initial_bottles_l135_135193

theorem elizabeth_initial_bottles (B : ℕ) (H1 : B - 2 - 1 = (3 * X) → 3 * (B - 3) = 21) : B = 10 :=
by
  sorry

end elizabeth_initial_bottles_l135_135193


namespace prob_not_lose_money_proof_min_purchase_price_proof_l135_135089

noncomputable def prob_not_lose_money : ℚ :=
  let pr_normal_rain := (2 : ℚ) / 3
  let pr_less_rain := (1 : ℚ) / 3
  let pr_price_6_normal := (1 : ℚ) / 4
  let pr_price_6_less := (2 : ℚ) / 3
  pr_normal_rain * pr_price_6_normal + pr_less_rain * pr_price_6_less

theorem prob_not_lose_money_proof : prob_not_lose_money = 7 / 18 := sorry

noncomputable def min_purchase_price : ℚ :=
  let old_exp_income := 500
  let new_yield := 2500
  let cost_increase := 1000
  (7000 + 1500 + cost_increase) / new_yield
  
theorem min_purchase_price_proof : min_purchase_price = 3.4 := sorry

end prob_not_lose_money_proof_min_purchase_price_proof_l135_135089


namespace effective_simple_interest_rate_proof_l135_135769

noncomputable def effective_simple_interest_rate : ℝ :=
  let P := 1
  let r1 := 0.10 / 2 -- Half-yearly rate for year 1
  let t1 := 2 -- number of compounding periods semi-annual
  let A1 := P * (1 + r1) ^ t1

  let r2 := 0.12 / 2 -- Half-yearly rate for year 2
  let t2 := 2
  let A2 := A1 * (1 + r2) ^ t2

  let r3 := 0.14 / 2 -- Half-yearly rate for year 3
  let t3 := 2
  let A3 := A2 * (1 + r3) ^ t3

  let r4 := 0.16 / 2 -- Half-yearly rate for year 4
  let t4 := 2
  let A4 := A3 * (1 + r4) ^ t4

  let CI := 993
  let P_actual := CI / (A4 - P)
  let effective_simple_interest := (CI / P_actual) * 100
  effective_simple_interest

theorem effective_simple_interest_rate_proof :
  effective_simple_interest_rate = 65.48 := by
  sorry

end effective_simple_interest_rate_proof_l135_135769


namespace find_n_value_l135_135556

theorem find_n_value (x y : ℕ) : x = 3 → y = 1 → n = x - y^(x - y) → x > y → n + x * y = 5 := by sorry

end find_n_value_l135_135556


namespace arc_length_l135_135862

-- Define the conditions
def radius (r : ℝ) := 2 * r + 2 * r = 8
def central_angle (θ : ℝ) := θ = 2 -- Given the central angle

-- Define the length of the arc
def length_of_arc (l r : ℝ) := l = r * 2

-- Theorem stating that given the sector conditions, the length of the arc is 4 cm
theorem arc_length (r l : ℝ) (h1 : central_angle 2) (h2 : radius r) (h3 : length_of_arc l r) : l = 4 :=
by
  sorry

end arc_length_l135_135862


namespace sin_seventeen_pi_over_four_l135_135900

theorem sin_seventeen_pi_over_four : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := sorry

end sin_seventeen_pi_over_four_l135_135900


namespace square_area_increase_l135_135684

theorem square_area_increase (s : ℝ) (h : s > 0) :
  ((1.15 * s) ^ 2 - s ^ 2) / s ^ 2 * 100 = 32.25 :=
by
  sorry

end square_area_increase_l135_135684


namespace initial_cows_l135_135523

theorem initial_cows (x : ℕ) (h : (3 / 4 : ℝ) * (x + 5) = 42) : x = 51 :=
by
  sorry

end initial_cows_l135_135523


namespace total_money_spent_l135_135317

def total_cost (blades_cost : Nat) (string_cost : Nat) : Nat :=
  blades_cost + string_cost

theorem total_money_spent 
  (num_blades : Nat)
  (cost_per_blade : Nat)
  (string_cost : Nat)
  (h1 : num_blades = 4)
  (h2 : cost_per_blade = 8)
  (h3 : string_cost = 7) :
  total_cost (num_blades * cost_per_blade) string_cost = 39 :=
by
  sorry

end total_money_spent_l135_135317


namespace sin_double_angle_l135_135304

theorem sin_double_angle (x : ℝ) (h : Real.sin (x - π / 4) = 3 / 5) : Real.sin (2 * x) = 7 / 25 :=
by
  sorry

end sin_double_angle_l135_135304


namespace probability_is_two_thirds_l135_135997

noncomputable def probabilityOfEvent : ℚ :=
  let Ω := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 }
  let A := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 ∧ 2 * p.1 - p.2 + 2 ≥ 0 }
  let area_Ω := (2 - 0) * (6 - 0)
  let area_A := area_Ω - (1 / 2) * 2 * 4
  (area_A / area_Ω : ℚ)

theorem probability_is_two_thirds : probabilityOfEvent = (2 / 3 : ℚ) :=
  sorry

end probability_is_two_thirds_l135_135997


namespace spade_5_7_8_l135_135026

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_5_7_8 : spade 5 (spade 7 8) = -200 :=
by
  sorry

end spade_5_7_8_l135_135026


namespace factor_expression_l135_135661

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end factor_expression_l135_135661


namespace proof_problem_l135_135766

-- Define the problem:
def problem := ∀ (a : Fin 100 → ℝ), 
  (∀ i j, i ≠ j → a i ≠ a j) →  -- All numbers are distinct
  ∃ i : Fin 100, a i + a (⟨i.val + 3, sorry⟩) > a (⟨i.val + 1, sorry⟩) + a (⟨i.val + 2, sorry⟩)
-- Summarize: there exists four consecutive points on the circle such that 
-- the sum of the numbers at the ends is greater than the sum of the numbers in the middle.

theorem proof_problem : problem := sorry

end proof_problem_l135_135766


namespace number_line_steps_l135_135666

theorem number_line_steps (n : ℕ) (total_distance : ℕ) (steps_to_x : ℕ) (x : ℕ)
  (h1 : total_distance = 32)
  (h2 : n = 8)
  (h3 : steps_to_x = 6)
  (h4 : x = (total_distance / n) * steps_to_x) :
  x = 24 := 
sorry

end number_line_steps_l135_135666


namespace cylinder_volume_transformation_l135_135645

theorem cylinder_volume_transformation (π : ℝ) (r h : ℝ) (V : ℝ) (V_new : ℝ)
  (hV : V = π * r^2 * h) (hV_initial : V = 20) : V_new = π * (3 * r)^2 * (4 * h) :=
by
sorry

end cylinder_volume_transformation_l135_135645


namespace seq_bounded_l135_135077

def digit_product (n : ℕ) : ℕ :=
  n.digits 10 |>.prod

def a_seq (a : ℕ → ℕ) (m : ℕ) : Prop :=
  a 0 = m ∧ (∀ n, a (n + 1) = a n + digit_product (a n))

theorem seq_bounded (m : ℕ) : ∃ B, ∀ n, a_seq a m → a n < B :=
by sorry

end seq_bounded_l135_135077


namespace question_implies_answer_l135_135301

theorem question_implies_answer (x y : ℝ) (h : y^2 - x^2 < x) :
  (x ≥ 0 ∨ x ≤ -1) ∧ (-Real.sqrt (x^2 + x) < y ∧ y < Real.sqrt (x^2 + x)) :=
sorry

end question_implies_answer_l135_135301


namespace statement_A_correct_statement_C_correct_l135_135905

open Nat

def combinations (n r : ℕ) : ℕ := n.choose r

theorem statement_A_correct : combinations 5 3 = combinations 5 2 := sorry

theorem statement_C_correct : combinations 6 3 - combinations 4 1 = combinations 6 3 - 4 := sorry

end statement_A_correct_statement_C_correct_l135_135905


namespace noah_ate_burgers_l135_135651

theorem noah_ate_burgers :
  ∀ (weight_hotdog weight_burger weight_pie : ℕ) 
    (mason_hotdog_weight : ℕ) 
    (jacob_pies noah_burgers mason_hotdogs : ℕ),
    weight_hotdog = 2 →
    weight_burger = 5 →
    weight_pie = 10 →
    (jacob_pies + 3 = noah_burgers) →
    (mason_hotdogs = 3 * jacob_pies) →
    (mason_hotdog_weight = 30) →
    (mason_hotdog_weight / weight_hotdog = mason_hotdogs) →
    noah_burgers = 8 :=
by
  intros weight_hotdog weight_burger weight_pie mason_hotdog_weight
         jacob_pies noah_burgers mason_hotdogs
         h1 h2 h3 h4 h5 h6 h7
  sorry

end noah_ate_burgers_l135_135651


namespace units_digit_of_product_of_seven_consecutive_integers_is_zero_l135_135486

/-- Define seven consecutive positive integers and show the units digit of their product is 0 -/
theorem units_digit_of_product_of_seven_consecutive_integers_is_zero (n : ℕ) :
  ∃ (k : ℕ), k = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) % 10 ∧ k = 0 :=
by {
  -- We state that the units digit k of the product of seven consecutive integers
  -- starting from n is 0
  sorry
}

end units_digit_of_product_of_seven_consecutive_integers_is_zero_l135_135486


namespace remainder_cd_42_l135_135525

theorem remainder_cd_42 (c d : ℕ) (p q : ℕ) (hc : c = 84 * p + 76) (hd : d = 126 * q + 117) : 
  (c + d) % 42 = 25 :=
by
  sorry

end remainder_cd_42_l135_135525


namespace triangle_area_ratio_l135_135981

theorem triangle_area_ratio {A B C : ℝ} {a b c : ℝ} 
  (h : 2 * Real.sin A * Real.cos (B - C) + Real.sin (2 * A) = 2 / 3) 
  (S1 : ℝ) (S2 : ℝ) :
  S1 / S2 = 1 / (3 * Real.pi) :=
sorry

end triangle_area_ratio_l135_135981


namespace quadratic_has_two_distinct_real_roots_l135_135094

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ p : ℝ, (p = x1 ∨ p = x2) → (p ^ 2 + (4 * m + 1) * p + m = 0)) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l135_135094


namespace analytical_expression_of_C3_l135_135095

def C1 (x : ℝ) : ℝ := x^2 - 2*x + 3
def C2 (x : ℝ) : ℝ := C1 (x + 1)
def C3 (x : ℝ) : ℝ := C2 (-x)

theorem analytical_expression_of_C3 :
  ∀ x, C3 x = x^2 + 2 := by
  sorry

end analytical_expression_of_C3_l135_135095


namespace g_of_neg2_l135_135953

def g (x : ℤ) : ℤ := x^3 - x^2 + x

theorem g_of_neg2 : g (-2) = -14 := 
by
  sorry

end g_of_neg2_l135_135953


namespace length_of_AB_l135_135627

variables {A B P Q : ℝ}
variables (x y : ℝ)

-- Conditions
axiom h1 : A < P ∧ P < Q ∧ Q < B
axiom h2 : P - A = 3 * x
axiom h3 : B - P = 5 * x
axiom h4 : Q - A = 2 * y
axiom h5 : B - Q = 3 * y
axiom h6 : Q - P = 3

-- Theorem statement
theorem length_of_AB : B - A = 120 :=
by
  sorry

end length_of_AB_l135_135627


namespace stratified_sampling_group_l135_135913

-- Definitions of conditions
def female_students : ℕ := 24
def male_students : ℕ := 36
def selected_females : ℕ := 8
def selected_males : ℕ := 12

-- Total number of ways to select the group
def total_combinations : ℕ := Nat.choose female_students selected_females * Nat.choose male_students selected_males

-- Proof of the problem
theorem stratified_sampling_group :
  (total_combinations = Nat.choose 24 8 * Nat.choose 36 12) :=
by
  sorry

end stratified_sampling_group_l135_135913


namespace sqrt_infinite_nested_problem_l135_135987

theorem sqrt_infinite_nested_problem :
  ∃ m : ℝ, m = Real.sqrt (6 + m) ∧ m = 3 :=
by
  sorry

end sqrt_infinite_nested_problem_l135_135987


namespace socks_total_is_51_l135_135521

-- Define initial conditions for John and Mary
def john_initial_socks : Nat := 33
def john_thrown_away_socks : Nat := 19
def john_new_socks : Nat := 13

def mary_initial_socks : Nat := 20
def mary_thrown_away_socks : Nat := 6
def mary_new_socks : Nat := 10

-- Define the total socks function
def total_socks (john_initial john_thrown john_new mary_initial mary_thrown mary_new : Nat) : Nat :=
  (john_initial - john_thrown + john_new) + (mary_initial - mary_thrown + mary_new)

-- Statement to prove
theorem socks_total_is_51 : 
  total_socks john_initial_socks john_thrown_away_socks john_new_socks 
              mary_initial_socks mary_thrown_away_socks mary_new_socks = 51 := 
by
  sorry

end socks_total_is_51_l135_135521


namespace quadratic_always_positive_if_and_only_if_l135_135547

theorem quadratic_always_positive_if_and_only_if :
  (∀ x : ℝ, x^2 + m * x + m + 3 > 0) ↔ (-2 < m ∧ m < 6) :=
by sorry

end quadratic_always_positive_if_and_only_if_l135_135547


namespace difference_of_results_l135_135664

theorem difference_of_results (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (h_diff: a ≠ b) :
  (70 * a - 7 * a) - (70 * b - 7 * b) = 0 :=
by
  sorry

end difference_of_results_l135_135664


namespace world_cup_teams_count_l135_135169

/-- In the world cup inauguration event, captains and vice-captains of all the teams are invited and awarded welcome gifts. There are some teams participating in the world cup, and 14 gifts are needed for this event. If each team has a captain and a vice-captain, and thus receives 2 gifts, then the number of teams participating is 7. -/
theorem world_cup_teams_count (total_gifts : ℕ) (gifts_per_team : ℕ) (teams : ℕ) 
  (h1 : total_gifts = 14) 
  (h2 : gifts_per_team = 2) 
  (h3 : total_gifts = teams * gifts_per_team) 
: teams = 7 :=
by sorry

end world_cup_teams_count_l135_135169


namespace expand_expression_l135_135857

theorem expand_expression : 
  ∀ (x : ℝ), (7 * x^3 - 5 * x + 2) * 4 * x^2 = 28 * x^5 - 20 * x^3 + 8 * x^2 :=
by
  intros x
  sorry

end expand_expression_l135_135857


namespace profit_per_meter_l135_135187

theorem profit_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (total_cost_price : ℕ := cost_price_per_meter * total_meters)
  (total_profit : ℕ := selling_price - total_cost_price)
  (profit_per_meter : ℕ := total_profit / total_meters) :
  total_meters = 75 ∧ selling_price = 4950 ∧ cost_price_per_meter = 51 → profit_per_meter = 15 :=
by
  intros h
  sorry

end profit_per_meter_l135_135187


namespace total_veg_eaters_l135_135810

def people_eat_only_veg : ℕ := 16
def people_eat_only_nonveg : ℕ := 9
def people_eat_both_veg_and_nonveg : ℕ := 12

theorem total_veg_eaters : people_eat_only_veg + people_eat_both_veg_and_nonveg = 28 := 
by
  sorry

end total_veg_eaters_l135_135810


namespace divisors_72_l135_135522

theorem divisors_72 : 
  { d | d ∣ 72 ∧ 0 < d } = {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72} := 
sorry

end divisors_72_l135_135522


namespace largest_positive_integer_divisible_l135_135091

theorem largest_positive_integer_divisible (n : ℕ) :
  (n + 20 ∣ n^3 - 100) ↔ n = 2080 :=
sorry

end largest_positive_integer_divisible_l135_135091


namespace coefficients_sum_l135_135394

theorem coefficients_sum (a0 a1 a2 a3 a4 : ℝ) (h : (1 - 2*x)^4 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) : 
  a0 + a4 = 17 :=
by
  sorry

end coefficients_sum_l135_135394


namespace probability_of_selection_of_X_l135_135238

theorem probability_of_selection_of_X 
  (P_Y : ℝ)
  (P_X_and_Y : ℝ) :
  P_Y = 2 / 7 →
  P_X_and_Y = 0.05714285714285714 →
  ∃ P_X : ℝ, P_X = 0.2 :=
by
  intro hY hXY
  sorry

end probability_of_selection_of_X_l135_135238


namespace train_length_l135_135298

theorem train_length (v_kmh : ℝ) (p_len : ℝ) (t_sec : ℝ) (l_train : ℝ) 
  (h_v : v_kmh = 72) (h_p : p_len = 250) (h_t : t_sec = 26) :
  l_train = 270 :=
by
  sorry

end train_length_l135_135298


namespace james_total_cost_l135_135995

def subscription_cost (base_cost : ℕ) (free_hours : ℕ) (extra_hour_cost : ℕ) (movie_rental_cost : ℝ) (streamed_hours : ℕ) (rented_movies : ℕ) : ℝ :=
  let extra_hours := max (streamed_hours - free_hours) 0
  base_cost + extra_hours * extra_hour_cost + rented_movies * movie_rental_cost

theorem james_total_cost 
  (base_cost : ℕ)
  (free_hours : ℕ)
  (extra_hour_cost : ℕ)
  (movie_rental_cost : ℝ)
  (streamed_hours : ℕ)
  (rented_movies : ℕ)
  (h_base_cost : base_cost = 15)
  (h_free_hours : free_hours = 50)
  (h_extra_hour_cost : extra_hour_cost = 2)
  (h_movie_rental_cost : movie_rental_cost = 0.10)
  (h_streamed_hours : streamed_hours = 53)
  (h_rented_movies : rented_movies = 30) :
  subscription_cost base_cost free_hours extra_hour_cost movie_rental_cost streamed_hours rented_movies = 24 := 
by {
  sorry
}

end james_total_cost_l135_135995


namespace ratio_shorter_to_longer_l135_135220

-- Constants for the problem
def total_length : ℝ := 49
def shorter_piece_length : ℝ := 14

-- Definition of longer piece length based on the given conditions
def longer_piece_length : ℝ := total_length - shorter_piece_length

-- The theorem to be proved
theorem ratio_shorter_to_longer : 
  shorter_piece_length / longer_piece_length = 2 / 5 :=
by
  -- This is where the proof would go
  sorry

end ratio_shorter_to_longer_l135_135220


namespace outer_squares_equal_three_times_inner_squares_l135_135832

theorem outer_squares_equal_three_times_inner_squares
  (a b c m_a m_b m_c : ℝ) 
  (h : m_a^2 + m_b^2 + m_c^2 = 3 / 4 * (a^2 + b^2 + c^2)) :
  a^2 + b^2 + c^2 = 3 * (m_a^2 + m_b^2 + m_c^2) := 
by 
  sorry

end outer_squares_equal_three_times_inner_squares_l135_135832


namespace chef_cherries_l135_135923

theorem chef_cherries :
  ∀ (total_cherries used_cherries remaining_cherries : ℕ),
    total_cherries = 77 →
    used_cherries = 60 →
    remaining_cherries = total_cherries - used_cherries →
    remaining_cherries = 17 :=
by
  sorry

end chef_cherries_l135_135923


namespace even_x_satisfies_remainder_l135_135374

theorem even_x_satisfies_remainder 
  (z : ℕ) 
  (hz : z % 4 = 0) : 
  ∃ (x : ℕ), x % 2 = 0 ∧ (z * (2 + x + z) + 3) % 2 = 1 := 
by
  sorry

end even_x_satisfies_remainder_l135_135374


namespace jim_gave_away_675_cards_l135_135338

def total_cards_gave_away
  (cards_per_set : ℕ)
  (sets_to_brother sets_to_sister sets_to_friend : ℕ)
  : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend) * cards_per_set

theorem jim_gave_away_675_cards
  (cards_per_set : ℕ)
  (sets_to_brother sets_to_sister sets_to_friend : ℕ)
  (h_brother : sets_to_brother = 15)
  (h_sister : sets_to_sister = 8)
  (h_friend : sets_to_friend = 4)
  (h_cards_per_set : cards_per_set = 25)
  : total_cards_gave_away cards_per_set sets_to_brother sets_to_sister sets_to_friend = 675 :=
by
  sorry

end jim_gave_away_675_cards_l135_135338


namespace paul_bought_150_books_l135_135302

theorem paul_bought_150_books (initial_books sold_books books_now : ℤ)
  (h1 : initial_books = 2)
  (h2 : sold_books = 94)
  (h3 : books_now = 58) :
  initial_books - sold_books + books_now = 150 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end paul_bought_150_books_l135_135302


namespace exists_a_b_not_multiple_p_l135_135849

theorem exists_a_b_not_multiple_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ a b : ℤ, ∀ m : ℤ, ¬ (m^3 + 2017 * a * m + b) ∣ (p : ℤ) :=
sorry

end exists_a_b_not_multiple_p_l135_135849


namespace estimated_students_in_sport_A_correct_l135_135733

noncomputable def total_students_surveyed : ℕ := 80
noncomputable def students_in_sport_A_surveyed : ℕ := 30
noncomputable def total_school_population : ℕ := 800
noncomputable def proportion_sport_A : ℚ := students_in_sport_A_surveyed / total_students_surveyed
noncomputable def estimated_students_in_sport_A : ℚ := total_school_population * proportion_sport_A

theorem estimated_students_in_sport_A_correct :
  estimated_students_in_sport_A = 300 :=
by
  sorry

end estimated_students_in_sport_A_correct_l135_135733


namespace geom_mean_4_16_l135_135877

theorem geom_mean_4_16 (x : ℝ) (h : x^2 = 4 * 16) : x = 8 ∨ x = -8 :=
by
  sorry

end geom_mean_4_16_l135_135877


namespace find_middle_number_l135_135108

theorem find_middle_number (a : Fin 11 → ℝ)
  (h1 : ∀ i : Fin 9, a i + a (⟨i.1 + 1, by linarith [i.2]⟩) + a (⟨i.1 + 2, by linarith [i.2]⟩) = 18)
  (h2 : (Finset.univ.sum a) = 64) :
  a 5 = 8 := 
by
  sorry

end find_middle_number_l135_135108


namespace simplify_sqrt_power_l135_135214

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l135_135214


namespace power_mod_equiv_l135_135899

theorem power_mod_equiv : 7^150 % 12 = 1 := 
  by
  sorry

end power_mod_equiv_l135_135899


namespace most_reasonable_plan_l135_135004

-- Defining the conditions as a type
inductive SurveyPlans
| A -- Surveying students in the second grade of School B
| C -- Randomly surveying 150 teachers
| B -- Surveying 600 students randomly selected from School C
| D -- Randomly surveying 150 students from each of the four schools

-- Define the main theorem asserting that the most reasonable plan is Option D
theorem most_reasonable_plan : SurveyPlans.D = SurveyPlans.D :=
by
  sorry

end most_reasonable_plan_l135_135004


namespace units_digit_of_k_squared_plus_2_k_l135_135583

def k := 2008^2 + 2^2008

theorem units_digit_of_k_squared_plus_2_k : 
  (k^2 + 2^k) % 10 = 7 :=
by {
  -- The proof will be inserted here
  sorry
}

end units_digit_of_k_squared_plus_2_k_l135_135583


namespace circles_intersect_l135_135989

def C1 (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1
def C2 (x y a : ℝ) : Prop := (x-a)^2 + (y-1)^2 = 16

theorem circles_intersect (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, C1 x y → ∃ x' y' : ℝ, C2 x' y' a) ↔ 3 < a ∧ a < 4 :=
sorry

end circles_intersect_l135_135989


namespace problem_l135_135815

theorem problem (x y z : ℝ) 
  (h1 : (x - 4)^2 + (y - 3)^2 + (z - 2)^2 = 0)
  (h2 : 3 * x + 2 * y - z = 12) :
  x + y + z = 9 := 
  sorry

end problem_l135_135815


namespace incorrect_contrapositive_l135_135180

theorem incorrect_contrapositive (x : ℝ) : (x ≠ 1 → ¬ (x^2 - 1 = 0)) ↔ ¬ (x^2 - 1 = 0 → x^2 = 1) := by
  sorry

end incorrect_contrapositive_l135_135180


namespace train_passing_time_l135_135039

noncomputable def train_length : ℝ := 180
noncomputable def train_speed_km_hr : ℝ := 36
noncomputable def train_speed_m_s : ℝ := train_speed_km_hr * (1000 / 3600)

theorem train_passing_time : train_length / train_speed_m_s = 18 := by
  sorry

end train_passing_time_l135_135039


namespace complex_division_l135_135622

def i : ℂ := Complex.I

theorem complex_division :
  (i^3 / (1 + i)) = -1/2 - 1/2 * i := 
by sorry

end complex_division_l135_135622


namespace solution_set_for_x_l135_135456

theorem solution_set_for_x (x : ℝ) (h : ⌊x⌋ + ⌈x⌉ = 7) : 3 < x ∧ x < 4 :=
sorry

end solution_set_for_x_l135_135456


namespace inequality_proof_equality_condition_l135_135912

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) ≥ (4 * a) / (a + b) := 
by
  sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) = (4 * a) / (a + b) ↔ a = b ∧ b = c :=
by
  sorry

end inequality_proof_equality_condition_l135_135912


namespace abc_inequality_l135_135728

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a^2 < 16 * b * c) (h2 : b^2 < 16 * c * a) (h3 : c^2 < 16 * a * b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) :=
by sorry

end abc_inequality_l135_135728


namespace squared_difference_of_roots_l135_135064

theorem squared_difference_of_roots:
  ∀ (Φ φ : ℝ), (∀ x : ℝ, x^2 = 2*x + 1 ↔ (x = Φ ∨ x = φ)) ∧ Φ ≠ φ → (Φ - φ)^2 = 8 :=
by
  intros Φ φ h
  sorry

end squared_difference_of_roots_l135_135064


namespace quadratic_condition_l135_135690

theorem quadratic_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * x + 3 = 0) → a ≠ 0 :=
by 
  intro h
  -- Proof will be here
  sorry

end quadratic_condition_l135_135690


namespace mans_speed_upstream_l135_135208

-- Define the conditions
def V_downstream : ℝ := 15  -- Speed with the current (downstream)
def V_current : ℝ := 2.5    -- Speed of the current

-- Calculate the man's speed against the current (upstream)
theorem mans_speed_upstream : V_downstream - 2 * V_current = 10 :=
by
  sorry

end mans_speed_upstream_l135_135208


namespace minimum_value_of_PA_PF_l135_135517

noncomputable def ellipse_min_distance : ℝ :=
  let F := (1, 0)
  let A := (1, 1)
  let a : ℝ := 3
  let F1 := (-1, 0)
  let d_A_F1 : ℝ := Real.sqrt ((-1 - 1)^2 + (0 - 1)^2)
  6 - d_A_F1

theorem minimum_value_of_PA_PF :
  ellipse_min_distance = 6 - Real.sqrt 5 :=
by
  sorry

end minimum_value_of_PA_PF_l135_135517


namespace simple_interest_rate_l135_135006

def principal : ℕ := 600
def amount : ℕ := 950
def time : ℕ := 5
def expected_rate : ℚ := 11.67

theorem simple_interest_rate (P A T : ℕ) (R : ℚ) :
  P = principal → A = amount → T = time → R = expected_rate →
  (A = P + P * R * T / 100) :=
by
  intros hP hA hT hR
  sorry

end simple_interest_rate_l135_135006


namespace Katya_possible_numbers_l135_135144

def divisible_by (n m : ℕ) : Prop := m % n = 0

def possible_numbers (n : ℕ) : Prop :=
  let condition1 := divisible_by 7 n  -- Alyona's condition
  let condition2 := divisible_by 5 n  -- Lena's condition
  let condition3 := n < 9             -- Rita's condition
  (condition1 ∨ condition2) ∧ condition3 ∧ 
  ((condition1 ∧ condition3 ∧ ¬condition2) ∨ (condition2 ∧ condition3 ∧ ¬condition1))

theorem Katya_possible_numbers :
  ∀ n : ℕ, 
    (possible_numbers n) ↔ (n = 5 ∨ n = 7) :=
sorry

end Katya_possible_numbers_l135_135144


namespace geometric_sequence_a6_l135_135353

variable {a : ℕ → ℝ} (h_geo : ∀ n, a (n+1) / a n = a (n+2) / a (n+1))

theorem geometric_sequence_a6 (h5 : a 5 = 2) (h7 : a 7 = 8) : a 6 = 4 ∨ a 6 = -4 :=
by
  sorry

end geometric_sequence_a6_l135_135353


namespace find_value_of_expression_l135_135575

theorem find_value_of_expression
  (x y z : ℝ)
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x + 2 * y - 7 * z = 0)
  (hz : z ≠ 0) :
  (x^2 - 2 * x * y) / (y^2 + 4 * z^2) = -0.252 := 
sorry

end find_value_of_expression_l135_135575


namespace range_of_m_l135_135145

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y > m^2 + 2 * m)) → -4 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l135_135145


namespace sin_double_angle_l135_135412

theorem sin_double_angle (θ : Real) (h : Real.sin θ = 3/5) : Real.sin (2*θ) = 24/25 :=
by
  sorry

end sin_double_angle_l135_135412


namespace calculate_value_l135_135752

theorem calculate_value 
  (a : Int) (b : Int) (c : Real) (d : Real)
  (h1 : a = -1)
  (h2 : b = 2)
  (h3 : c * d = 1) :
  a + b - c * d = 0 := 
by
  sorry

end calculate_value_l135_135752


namespace work_completion_time_l135_135221

-- Define work rates for workers p, q, and r
def work_rate_p : ℚ := 1 / 12
def work_rate_q : ℚ := 1 / 9
def work_rate_r : ℚ := 1 / 18

-- Define time they work in respective phases
def time_p : ℚ := 2
def time_pq : ℚ := 3

-- Define the total time taken to complete the work
def total_time : ℚ := 6

-- Prove that the total time to complete the work is 6 days
theorem work_completion_time :
  (work_rate_p * time_p + (work_rate_p + work_rate_q) * time_pq + (1 - (work_rate_p * time_p + (work_rate_p + work_rate_q) * time_pq)) / (work_rate_p + work_rate_q + work_rate_r)) = total_time :=
by sorry

end work_completion_time_l135_135221


namespace compute_abc_l135_135843

theorem compute_abc (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h_sum : a + b + c = 30) (h_frac : (1 : ℚ) / a + 1 / b + 1 / c + 450 / (a * b * c) = 1) : a * b * c = 1920 :=
by sorry

end compute_abc_l135_135843


namespace cube_difference_l135_135552

theorem cube_difference (x y : ℕ) (h₁ : x + y = 64) (h₂ : x - y = 16) : x^3 - y^3 = 50176 := by
  sorry

end cube_difference_l135_135552


namespace simplify_expression_l135_135390

theorem simplify_expression (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^3 + 2) / x * (y^3 + 2) / y) - ((x^3 - 2) / y * (y^3 - 2) / x) = 4 * (x^2 / y + y^2 / x) :=
by sorry

end simplify_expression_l135_135390


namespace find_five_digit_number_l135_135600

theorem find_five_digit_number (x : ℕ) (hx : 10000 ≤ x ∧ x < 100000)
  (h : 10 * x + 1 = 3 * (100000 + x) ∨ 3 * (10 * x + 1) = 100000 + x) :
  x = 42857 :=
sorry

end find_five_digit_number_l135_135600


namespace volume_tetrahedron_PXYZ_l135_135869

noncomputable def volume_of_tetrahedron_PXYZ (x y z : ℝ) : ℝ :=
  (1 / 6) * x * y * z

theorem volume_tetrahedron_PXYZ :
  ∃ (x y z : ℝ), (x^2 + y^2 = 49) ∧ (y^2 + z^2 = 64) ∧ (z^2 + x^2 = 81) ∧
  volume_of_tetrahedron_PXYZ (Real.sqrt x) (Real.sqrt y) (Real.sqrt z) = 4 * Real.sqrt 11 := 
by {
  sorry
}

end volume_tetrahedron_PXYZ_l135_135869


namespace harry_apples_l135_135130

theorem harry_apples (martha_apples : ℕ) (tim_apples : ℕ) (harry_apples : ℕ)
  (h1 : martha_apples = 68)
  (h2 : tim_apples = martha_apples - 30)
  (h3 : harry_apples = tim_apples / 2) :
  harry_apples = 19 := 
by sorry

end harry_apples_l135_135130


namespace solve_f_neg_a_l135_135212

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem solve_f_neg_a (h : f a = 8) : f (-a) = -6 := by
  sorry

end solve_f_neg_a_l135_135212


namespace acres_of_flax_l135_135226

-- Let F be the number of acres of flax
variable (F : ℕ)

-- Condition: The total farm size is 240 acres
def total_farm_size (F : ℕ) := F + (F + 80) = 240

-- Proof statement
theorem acres_of_flax (h : total_farm_size F) : F = 80 :=
sorry

end acres_of_flax_l135_135226


namespace mary_money_left_l135_135014

def initial_amount : Float := 150
def game_cost : Float := 60
def discount_percent : Float := 15 / 100
def remaining_percent_for_goggles : Float := 20 / 100
def tax_on_goggles : Float := 8 / 100

def money_left_after_shopping_trip (initial_amount : Float) (game_cost : Float) (discount_percent : Float) (remaining_percent_for_goggles : Float) (tax_on_goggles : Float) : Float :=
  let discount := game_cost * discount_percent
  let discounted_price := game_cost - discount
  let remainder_after_game := initial_amount - discounted_price
  let goggles_cost_before_tax := remainder_after_game * remaining_percent_for_goggles
  let tax := goggles_cost_before_tax * tax_on_goggles
  let final_goggles_cost := goggles_cost_before_tax + tax
  let remainder_after_goggles := remainder_after_game - final_goggles_cost
  remainder_after_goggles

#eval money_left_after_shopping_trip initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles -- expected: 77.62

theorem mary_money_left (initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles : Float) : 
  money_left_after_shopping_trip initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles = 77.62 :=
by sorry

end mary_money_left_l135_135014


namespace min_value_m_n_l135_135463

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem min_value_m_n 
  (a : ℝ) (m n : ℝ)
  (h_a_pos : a > 0) (h_a_ne1 : a ≠ 1)
  (h_mn_pos : m > 0 ∧ n > 0)
  (h_line_eq : 2 * m + n = 1) :
  m + n = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_m_n_l135_135463


namespace prank_combinations_l135_135151

-- Conditions stated as definitions
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 3
def wednesday_choices : ℕ := 5
def thursday_choices : ℕ := 6
def friday_choices : ℕ := 2

-- Theorem to prove
theorem prank_combinations :
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 180 :=
by
  sorry

end prank_combinations_l135_135151


namespace shares_total_amount_l135_135181

theorem shares_total_amount (Nina_portion : ℕ) (m n o : ℕ) (m_ratio n_ratio o_ratio : ℕ)
  (h_ratio : m_ratio = 2 ∧ n_ratio = 3 ∧ o_ratio = 9)
  (h_Nina : Nina_portion = 60)
  (hk := Nina_portion / n_ratio)
  (h_shares : m = m_ratio * hk ∧ n = n_ratio * hk ∧ o = o_ratio * hk) :
  m + n + o = 280 :=
by 
  sorry

end shares_total_amount_l135_135181


namespace nell_initial_cards_l135_135447

theorem nell_initial_cards (cards_given cards_left total_cards : ℕ)
  (h1 : cards_given = 301)
  (h2 : cards_left = 154)
  (h3 : total_cards = cards_given + cards_left) :
  total_cards = 455 := by
  rw [h1, h2] at h3
  exact h3

end nell_initial_cards_l135_135447


namespace negation_of_proposition_l135_135503

variable (x : ℝ)
variable (p : Prop)

def proposition : Prop := ∀ x > 0, (x + 1) * Real.exp x > 1

theorem negation_of_proposition : ¬ proposition ↔ ∃ x > 0, (x + 1) * Real.exp x ≤ 1 :=
by
  sorry

end negation_of_proposition_l135_135503


namespace sum_of_roots_eq_three_l135_135147

theorem sum_of_roots_eq_three (x1 x2 : ℝ) 
  (h1 : x1^2 - 3*x1 + 2 = 0)
  (h2 : x2^2 - 3*x2 + 2 = 0) 
  (h3 : x1 ≠ x2) : 
  x1 + x2 = 3 := 
sorry

end sum_of_roots_eq_three_l135_135147


namespace daliah_garbage_l135_135417

theorem daliah_garbage (D : ℝ) (h1 : 4 * (D - 2) = 62) : D = 17.5 :=
by
  sorry

end daliah_garbage_l135_135417


namespace parabolas_vertex_condition_l135_135159

theorem parabolas_vertex_condition (p q x₁ x₂ y₁ y₂ : ℝ) (h1: y₂ = p * (x₂ - x₁)^2 + y₁) (h2: y₁ = q * (x₁ - x₂)^2 + y₂) (h3: x₁ ≠ x₂) : p + q = 0 :=
sorry

end parabolas_vertex_condition_l135_135159


namespace sum_of_extreme_T_l135_135721

theorem sum_of_extreme_T (B M T : ℝ) 
  (h1 : B^2 + M^2 + T^2 = 2022)
  (h2 : B + M + T = 72) :
  ∃ Tmin Tmax, Tmin + Tmax = 48 ∧ Tmin ≤ T ∧ T ≤ Tmax :=
by
  sorry

end sum_of_extreme_T_l135_135721


namespace mary_investment_l135_135465

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem mary_investment :
  ∃ (P : ℝ), P = 51346 ∧ compound_interest P 0.10 12 7 = 100000 :=
by
  sorry

end mary_investment_l135_135465


namespace point_between_circles_l135_135641

theorem point_between_circles 
  (a b c x1 x2 : ℝ)
  (ellipse_eq : ∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (quad_eq : a * x1^2 + b * x1 - c = 0)
  (quad_eq2 : a * x2^2 + b * x2 - c = 0)
  (sum_roots : x1 + x2 = -b / a)
  (prod_roots : x1 * x2 = -c / a) :
  1 < x1^2 + x2^2 ∧ x1^2 + x2^2 < 2 :=
sorry

end point_between_circles_l135_135641


namespace yellow_more_than_purple_l135_135505
-- Import math library for necessary definitions.

-- Define the problem conditions in Lean
def num_purple_candies : ℕ := 10
def num_total_candies : ℕ := 36

axiom exists_yellow_and_green_candies 
  (Y G : ℕ) 
  (h1 : G = Y - 2) 
  (h2 : 10 + Y + G = 36) : True

-- The theorem to prove
theorem yellow_more_than_purple 
  (Y : ℕ) 
  (hY : exists (G : ℕ), G = Y - 2 ∧ 10 + Y + G = 36) : Y - num_purple_candies = 4 :=
by {
  sorry -- proof is not required
}

end yellow_more_than_purple_l135_135505


namespace inequality_solution_set_l135_135825

theorem inequality_solution_set :
  { x : ℝ | (10 * x^2 + 20 * x - 68) / ((2 * x - 3) * (x + 4) * (x - 2)) < 3 } =
  { x : ℝ | (-4 < x ∧ x < -2) ∨ (-1 / 3 < x ∧ x < 3 / 2) } :=
by
  sorry

end inequality_solution_set_l135_135825


namespace value_of_w_l135_135297

theorem value_of_w (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := 
sorry

end value_of_w_l135_135297


namespace maximal_subset_with_property_A_l135_135403

-- Define property A for a subset S ⊆ {0, 1, 2, ..., 99}
def has_property_A (S : Finset ℕ) : Prop := 
  ∀ a b c : ℕ, (a * 10 + b ∈ S) → (b * 10 + c ∈ S) → False

-- Define the set of integers {0, 1, 2, ..., 99}
def numbers_set := Finset.range 100

-- The main statement to be proven
theorem maximal_subset_with_property_A :
  ∃ S : Finset ℕ, S ⊆ numbers_set ∧ has_property_A S ∧ S.card = 25 := 
sorry

end maximal_subset_with_property_A_l135_135403


namespace inequality_proof_l135_135059

noncomputable def a : ℝ := 1 + Real.tan (-0.2)
noncomputable def b : ℝ := Real.log (0.8 * Real.exp 1)
noncomputable def c : ℝ := 1 / Real.exp 0.2

theorem inequality_proof : c > a ∧ a > b := by
  sorry

end inequality_proof_l135_135059


namespace range_of_a_l135_135996

noncomputable def A (a : ℝ) : Set ℝ := { x | 3 + a ≤ x ∧ x ≤ 4 + 3 * a }
noncomputable def B : Set ℝ := { x | -4 ≤ x ∧ x < 5 }

theorem range_of_a (a : ℝ) : A a ⊆ B ↔ -1/2 ≤ a ∧ a < 1/3 :=
  sorry

end range_of_a_l135_135996


namespace vec_addition_l135_135385

namespace VectorCalculation

open Real

def v1 : ℤ × ℤ := (3, -8)
def v2 : ℤ × ℤ := (2, -6)
def scalar : ℤ := 5

def scaled_v2 : ℤ × ℤ := (scalar * v2.1, scalar * v2.2)
def result : ℤ × ℤ := (v1.1 + scaled_v2.1, v1.2 + scaled_v2.2)

theorem vec_addition : result = (13, -38) := by
  sorry

end VectorCalculation

end vec_addition_l135_135385


namespace multiples_of_7_between_15_and_200_l135_135885

theorem multiples_of_7_between_15_and_200 : ∃ n : ℕ, n = 26 ∧ ∃ (a₁ a_n d : ℕ), 
  a₁ = 21 ∧ a_n = 196 ∧ d = 7 ∧ (a₁ + (n - 1) * d = a_n) := 
by
  sorry

end multiples_of_7_between_15_and_200_l135_135885


namespace tomato_price_per_kilo_l135_135467

theorem tomato_price_per_kilo 
  (initial_money: ℝ) (money_left: ℝ)
  (potato_price_per_kilo: ℝ) (potato_kilos: ℝ)
  (cucumber_price_per_kilo: ℝ) (cucumber_kilos: ℝ)
  (banana_price_per_kilo: ℝ) (banana_kilos: ℝ)
  (tomato_kilos: ℝ)
  (spent_on_potatoes: initial_money - money_left = potato_price_per_kilo * potato_kilos)
  (spent_on_cucumbers: initial_money - money_left = cucumber_price_per_kilo * cucumber_kilos)
  (spent_on_bananas: initial_money - money_left = banana_price_per_kilo * banana_kilos)
  (total_spent: initial_money - money_left = 74)
  : (74 - (potato_price_per_kilo * potato_kilos + cucumber_price_per_kilo * cucumber_kilos + banana_price_per_kilo * banana_kilos)) / tomato_kilos = 3 := 
sorry

end tomato_price_per_kilo_l135_135467


namespace rayden_has_more_birds_l135_135328

-- Definitions based on given conditions
def ducks_lily := 20
def geese_lily := 10
def chickens_lily := 5
def pigeons_lily := 30

def ducks_rayden := 3 * ducks_lily
def geese_rayden := 4 * geese_lily
def chickens_rayden := 5 * chickens_lily
def pigeons_rayden := pigeons_lily / 2

def more_ducks := ducks_rayden - ducks_lily
def more_geese := geese_rayden - geese_lily
def more_chickens := chickens_rayden - chickens_lily
def fewer_pigeons := pigeons_rayden - pigeons_lily

def total_more_birds := more_ducks + more_geese + more_chickens - fewer_pigeons

-- Statement to prove that Rayden has 75 more birds in total than Lily
theorem rayden_has_more_birds : total_more_birds = 75 := by
    sorry

end rayden_has_more_birds_l135_135328


namespace craig_total_commission_correct_l135_135391

-- Define the commission structures
def refrigerator_commission (price : ℝ) : ℝ := 75 + 0.08 * price
def washing_machine_commission (price : ℝ) : ℝ := 50 + 0.10 * price
def oven_commission (price : ℝ) : ℝ := 60 + 0.12 * price

-- Define total sales
def total_refrigerator_sales : ℝ := 5280
def total_washing_machine_sales : ℝ := 2140
def total_oven_sales : ℝ := 4620

-- Define number of appliances sold
def number_of_refrigerators : ℝ := 3
def number_of_washing_machines : ℝ := 4
def number_of_ovens : ℝ := 5

-- Calculate total commissions for each appliance category
def total_refrigerator_commission : ℝ := number_of_refrigerators * refrigerator_commission total_refrigerator_sales
def total_washing_machine_commission : ℝ := number_of_washing_machines * washing_machine_commission total_washing_machine_sales
def total_oven_commission : ℝ := number_of_ovens * oven_commission total_oven_sales

-- Calculate total commission for the week
def total_commission : ℝ := total_refrigerator_commission + total_washing_machine_commission + total_oven_commission

-- Prove that the total commission is as expected
theorem craig_total_commission_correct : total_commission = 5620.20 := 
by
  sorry

end craig_total_commission_correct_l135_135391


namespace distinct_connected_stamps_l135_135614

theorem distinct_connected_stamps (n : ℕ) : 
  ∃ d : ℕ → ℝ, 
    d (n+1) = 1 / 4 * (1 + Real.sqrt 2)^(n + 3) + 1 / 4 * (1 - Real.sqrt 2)^(n + 3) - 2 * n - 7 / 2 :=
sorry

end distinct_connected_stamps_l135_135614


namespace total_number_of_animals_l135_135031

-- Prove that the total number of animals is 300 given the conditions described.
theorem total_number_of_animals (A : ℕ) (H₁ : 4 * (A / 3) = 400) : A = 300 :=
sorry

end total_number_of_animals_l135_135031


namespace sum_of_odd_function_at_points_l135_135502

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem sum_of_odd_function_at_points (f : ℝ → ℝ) (h : is_odd_function f) : 
  f (-2) + f (-1) + f 0 + f 1 + f 2 = 0 :=
by
  sorry

end sum_of_odd_function_at_points_l135_135502


namespace solve_system_eqns_l135_135311

noncomputable def eq1 (x y z : ℚ) : Prop := x^2 + 2 * y * z = x
noncomputable def eq2 (x y z : ℚ) : Prop := y^2 + 2 * z * x = y
noncomputable def eq3 (x y z : ℚ) : Prop := z^2 + 2 * x * y = z

theorem solve_system_eqns (x y z : ℚ) :
  (eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) ↔
  ((x, y, z) = (0, 0, 0) ∨
   (x, y, z) = (1/3, 1/3, 1/3) ∨
   (x, y, z) = (1, 0, 0) ∨
   (x, y, z) = (0, 1, 0) ∨
   (x, y, z) = (0, 0, 1) ∨
   (x, y, z) = (2/3, -1/3, -1/3) ∨
   (x, y, z) = (-1/3, 2/3, -1/3) ∨
   (x, y, z) = (-1/3, -1/3, 2/3)) :=
by sorry

end solve_system_eqns_l135_135311


namespace fraction_sum_equals_zero_l135_135448

theorem fraction_sum_equals_zero :
  (1 / 12) + (2 / 12) + (3 / 12) + (4 / 12) + (5 / 12) + (6 / 12) + (7 / 12) + (8 / 12) + (9 / 12) - (45 / 12) = 0 :=
by
  sorry

end fraction_sum_equals_zero_l135_135448


namespace problem_statement_l135_135227

noncomputable def f (m x : ℝ) := (m-1) * Real.log x + m * x^2 + 1

theorem problem_statement (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ → x₂ > 0 → f m x₁ - f m x₂ > 2 * (x₁ - x₂)) ↔ 
  m ≥ (1 + Real.sqrt 3) / 2 :=
sorry

end problem_statement_l135_135227


namespace avg_student_headcount_l135_135016

def student_headcount (yr1 yr2 yr3 yr4 : ℕ) : ℕ :=
  (yr1 + yr2 + yr3 + yr4) / 4

theorem avg_student_headcount :
  student_headcount 10600 10800 10500 10400 = 10825 :=
by
  sorry

end avg_student_headcount_l135_135016


namespace geometric_arithmetic_sequence_l135_135589

theorem geometric_arithmetic_sequence (a q : ℝ) 
    (h₁ : a + a * q + a * q ^ 2 = 19) 
    (h₂ : a * (q - 1) = -1) : 
  (a = 4 ∧ q = 1.5) ∨ (a = 9 ∧ q = 2/3) :=
by
  sorry

end geometric_arithmetic_sequence_l135_135589


namespace set_union_example_l135_135285

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem set_union_example : M ∪ N = {1, 2, 3, 4} := by
  sorry

end set_union_example_l135_135285


namespace arithmetic_calculation_l135_135972

theorem arithmetic_calculation : 3 - (-5) + 7 = 15 := by
  sorry

end arithmetic_calculation_l135_135972


namespace arithmetic_mean_of_18_27_45_l135_135152

theorem arithmetic_mean_of_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_of_18_27_45_l135_135152


namespace find_divisor_l135_135938

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) 
  (h1 : dividend = 62976) 
  (h2 : quotient = 123) 
  (h3 : dividend = divisor * quotient) 
  : divisor = 512 := 
by
  sorry

end find_divisor_l135_135938


namespace Marie_finish_time_l135_135934

def Time := Nat × Nat -- Represents time as (hours, minutes)

def start_time : Time := (9, 0)
def finish_two_tasks_time : Time := (11, 20)
def total_tasks : Nat := 4

def minutes_since_start (t : Time) : Nat :=
  let (h, m) := t
  (h - 9) * 60 + m

def calculate_finish_time (start: Time) (two_tasks_finish: Time) (total_tasks: Nat) : Time :=
  let duration_two_tasks := minutes_since_start two_tasks_finish
  let duration_each_task := duration_two_tasks / 2
  let total_time := duration_each_task * total_tasks
  let total_minutes_after_start := total_time + minutes_since_start start
  let finish_hour := 9 + total_minutes_after_start / 60
  let finish_minute := total_minutes_after_start % 60
  (finish_hour, finish_minute)

theorem Marie_finish_time :
  calculate_finish_time start_time finish_two_tasks_time total_tasks = (13, 40) :=
by
  sorry

end Marie_finish_time_l135_135934


namespace permutation_combination_example_l135_135254

-- Definition of permutation (A) and combination (C) in Lean
def permutation (n k : ℕ): ℕ := Nat.factorial n / Nat.factorial (n - k)
def combination (n k : ℕ): ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The Lean statement of the proof problem
theorem permutation_combination_example : 
3 * permutation 3 2 + 2 * combination 4 2 = 30 := 
by 
  sorry

end permutation_combination_example_l135_135254


namespace contrapositive_true_l135_135071

theorem contrapositive_true (q p : Prop) (h : q → p) : ¬p → ¬q :=
by sorry

end contrapositive_true_l135_135071


namespace sum_of_consecutive_numbers_LCM_168_l135_135375

theorem sum_of_consecutive_numbers_LCM_168
  (x y z : ℕ)
  (h1 : y = x + 1)
  (h2 : z = y + 1)
  (h3 : Nat.lcm (Nat.lcm x y) z = 168) :
  x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_numbers_LCM_168_l135_135375


namespace total_ages_l135_135749

variable (Craig_age Mother_age : ℕ)

theorem total_ages (h1 : Craig_age = 16) (h2 : Mother_age = Craig_age + 24) : Craig_age + Mother_age = 56 := by
  sorry

end total_ages_l135_135749


namespace common_ratio_of_geometric_series_l135_135886

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l135_135886


namespace exhibit_special_13_digit_integer_l135_135171

open Nat 

def thirteenDigitInteger (N : ℕ) : Prop :=
  N ≥ 10^12 ∧ N < 10^13

def isMultipleOf8192 (N : ℕ) : Prop :=
  8192 ∣ N

def hasOnlyEightOrNineDigits (N : ℕ) : Prop :=
  ∀ d ∈ digits 10 N, d = 8 ∨ d = 9

theorem exhibit_special_13_digit_integer : 
  ∃ N : ℕ, thirteenDigitInteger N ∧ isMultipleOf8192 N ∧ hasOnlyEightOrNineDigits N ∧ N = 8888888888888 := 
by
  sorry 

end exhibit_special_13_digit_integer_l135_135171


namespace find_total_quantities_l135_135624

theorem find_total_quantities (n S S_3 S_2 : ℕ) (h1 : S = 8 * n) (h2 : S_3 = 4 * 3) (h3 : S_2 = 14 * 2) (h4 : S = S_3 + S_2) : n = 5 :=
by
  sorry

end find_total_quantities_l135_135624


namespace necessary_not_sufficient_condition_l135_135363

theorem necessary_not_sufficient_condition (m : ℝ) 
  (h : 2 < m ∧ m < 6) :
  (∃ (x y : ℝ), (x^2 / (m - 2) + y^2 / (6 - m) = 1)) ∧ (∀ m', 2 < m' ∧ m' < 6 → ∃ (x' y' : ℝ), (x'^2 / (m' - 2) + y'^2 / (6 - m') = 1) ∧ m' ≠ 4) :=
by
  sorry

end necessary_not_sufficient_condition_l135_135363


namespace trader_loss_percentage_l135_135929

def profit_loss_percentage (SP1 SP2 CP1 CP2 : ℚ) : ℚ :=
  ((SP1 + SP2) - (CP1 + CP2)) / (CP1 + CP2) * 100

theorem trader_loss_percentage :
  let SP1 := 325475
  let SP2 := 325475
  let CP1 := SP1 / (1 + 0.10)
  let CP2 := SP2 / (1 - 0.10)
  profit_loss_percentage SP1 SP2 CP1 CP2 = -1 := by
  sorry

end trader_loss_percentage_l135_135929


namespace gcd_1729_78945_is_1_l135_135326

theorem gcd_1729_78945_is_1 :
  ∃ m n : ℤ, 1729 * m + 78945 * n = 1 := sorry

end gcd_1729_78945_is_1_l135_135326


namespace tiered_water_pricing_l135_135240

theorem tiered_water_pricing (x : ℝ) (y : ℝ) : 
  (∀ z, 0 ≤ z ∧ z ≤ 12 → y = 3 * z ∨
        12 < z ∧ z ≤ 18 → y = 36 + 6 * (z - 12) ∨
        18 < z → y = 72 + 9 * (z - 18)) → 
  y = 54 → 
  x = 15 :=
by
  sorry

end tiered_water_pricing_l135_135240


namespace a4_minus_1_divisible_5_l135_135119

theorem a4_minus_1_divisible_5 (a : ℤ) (h : ¬ (∃ k : ℤ, a = 5 * k)) : 
  (a^4 - 1) % 5 = 0 :=
by
  sorry

end a4_minus_1_divisible_5_l135_135119


namespace find_pots_l135_135325

def num_pots := 46
def cost_green_lily := 9
def cost_spider_plant := 6
def total_cost := 390

theorem find_pots (x y : ℕ) (h1 : x + y = num_pots) (h2 : cost_green_lily * x + cost_spider_plant * y = total_cost) :
  x = 38 ∧ y = 8 :=
by
  sorry

end find_pots_l135_135325


namespace average_of_remaining_three_numbers_l135_135421

noncomputable def avg_remaining_three_numbers (avg_12 : ℝ) (avg_4 : ℝ) (avg_3 : ℝ) (avg_2 : ℝ) : ℝ :=
  let sum_12 := 12 * avg_12
  let sum_4 := 4 * avg_4
  let sum_3 := 3 * avg_3
  let sum_2 := 2 * avg_2
  let sum_9 := sum_4 + sum_3 + sum_2
  let sum_remaining_3 := sum_12 - sum_9
  sum_remaining_3 / 3

theorem average_of_remaining_three_numbers :
  avg_remaining_three_numbers 6.30 5.60 4.90 7.25 = 8 :=
by {
  sorry
}

end average_of_remaining_three_numbers_l135_135421


namespace proctoring_arrangements_l135_135307

/-- Consider 4 teachers A, B, C, D each teaching their respective classes a, b, c, d.
    Each teacher must not proctor their own class.
    Prove that there are exactly 9 ways to arrange the proctoring as required. -/
theorem proctoring_arrangements : 
  ∃ (arrangements : Finset ((Fin 4) → (Fin 4))), 
    (∀ (f : (Fin 4) → (Fin 4)), f ∈ arrangements → ∀ i : Fin 4, f i ≠ i) 
    ∧ arrangements.card = 9 :=
sorry

end proctoring_arrangements_l135_135307


namespace man_walking_rate_l135_135797

theorem man_walking_rate (x : ℝ) 
  (woman_rate : ℝ := 15)
  (woman_time_after_passing : ℝ := 2 / 60)
  (man_time_to_catch_up : ℝ := 4 / 60)
  (distance_woman : ℝ := woman_rate * woman_time_after_passing)
  (distance_man : ℝ := x * man_time_to_catch_up)
  (h : distance_man = distance_woman) :
  x = 7.5 :=
sorry

end man_walking_rate_l135_135797


namespace number_of_lines_passing_through_point_and_forming_given_area_l135_135507

theorem number_of_lines_passing_through_point_and_forming_given_area :
  ∃ l : ℝ → ℝ, (∀ x y : ℝ, l 1 = 1) ∧ (∃ (a b : ℝ), abs ((1/2) * a * b) = 2)
  → (∃ n : ℕ, n = 4) :=
by
  sorry

end number_of_lines_passing_through_point_and_forming_given_area_l135_135507


namespace exists_primes_sum_2024_with_one_gt_1000_l135_135400

open Nat

-- Definition of primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Conditions given in the problem
def sum_primes_eq_2024 (p q : ℕ) : Prop :=
  p + q = 2024 ∧ is_prime p ∧ is_prime q

def at_least_one_gt_1000 (p q : ℕ) : Prop :=
  p > 1000 ∨ q > 1000

-- The theorem to be proved
theorem exists_primes_sum_2024_with_one_gt_1000 :
  ∃ (p q : ℕ), sum_primes_eq_2024 p q ∧ at_least_one_gt_1000 p q :=
sorry

end exists_primes_sum_2024_with_one_gt_1000_l135_135400


namespace concentration_of_concentrated_kola_is_correct_l135_135401

noncomputable def concentration_of_concentrated_kola_added 
  (initial_volume : ℝ) (initial_pct_sugar : ℝ)
  (sugar_added : ℝ) (water_added : ℝ)
  (required_pct_sugar : ℝ) (new_sugar_volume : ℝ) : ℝ :=
  let initial_sugar := initial_volume * initial_pct_sugar / 100
  let total_sugar := initial_sugar + sugar_added
  let new_total_volume := initial_volume + sugar_added + water_added
  let total_volume_with_kola := new_total_volume + (new_sugar_volume / required_pct_sugar * 100 - total_sugar) / (100 / required_pct_sugar - 1)
  total_volume_with_kola - new_total_volume

noncomputable def problem_kola : ℝ :=
  concentration_of_concentrated_kola_added 340 7 3.2 10 7.5 27

theorem concentration_of_concentrated_kola_is_correct : 
  problem_kola = 6.8 :=
by
  unfold problem_kola concentration_of_concentrated_kola_added
  sorry

end concentration_of_concentrated_kola_is_correct_l135_135401


namespace photo_area_with_frame_l135_135705

-- Define the areas and dimensions given in the conditions
def paper_length : ℕ := 12
def paper_width : ℕ := 8
def frame_width : ℕ := 2

-- Define the dimensions of the photo including the frame
def photo_length_with_frame : ℕ := paper_length + 2 * frame_width
def photo_width_with_frame : ℕ := paper_width + 2 * frame_width

-- The theorem statement proving the area of the wall photo including the frame
theorem photo_area_with_frame :
  (photo_length_with_frame * photo_width_with_frame) = 192 := by
  sorry

end photo_area_with_frame_l135_135705


namespace fourier_series_decomposition_l135_135735

open Real

noncomputable def f : ℝ → ℝ :=
  λ x => if (x < 0) then -1 else (if (0 < x) then 1/2 else 0)

theorem fourier_series_decomposition :
    ∀ x, -π ≤ x ∧ x ≤ π →
         f x = -1/4 + (3/π) * ∑' k, (sin ((2*k+1)*x)) / (2*k+1) :=
by
  sorry

end fourier_series_decomposition_l135_135735


namespace employee_payment_correct_l135_135704

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the percentage markup for retail price
def markup_percentage : ℝ := 0.20

-- Define the retail_price based on wholesale cost and markup percentage
def retail_price : ℝ := wholesale_cost + (markup_percentage * wholesale_cost)

-- Define the employee discount percentage
def discount_percentage : ℝ := 0.20

-- Define the discount amount based on retail price and discount percentage
def discount_amount : ℝ := retail_price * discount_percentage

-- Define the final price the employee pays after applying the discount
def employee_price : ℝ := retail_price - discount_amount

-- State the theorem to prove
theorem employee_payment_correct :
  employee_price = 192 :=
  by
    sorry

end employee_payment_correct_l135_135704


namespace speed_of_man_in_still_water_l135_135098

def upstream_speed : ℝ := 32
def downstream_speed : ℝ := 48

theorem speed_of_man_in_still_water :
  (upstream_speed + (downstream_speed - upstream_speed) / 2) = 40 :=
by
  sorry

end speed_of_man_in_still_water_l135_135098


namespace geometric_sequence_first_term_l135_135361

-- Define factorial values for convenience
def fact (n : ℕ) : ℕ := Nat.factorial n
#eval fact 6 -- This should give us 720
#eval fact 7 -- This should give us 5040

-- State the hypotheses and the goal
theorem geometric_sequence_first_term (a r : ℝ)
  (h1 : a * r^2 = 720)
  (h2 : a * r^5 = 5040) :
  a = 720 / (7^(2/3 : ℝ)) :=
by
  sorry

end geometric_sequence_first_term_l135_135361


namespace students_present_l135_135520

theorem students_present (absent_students male_students female_student_diff : ℕ) 
  (h1 : absent_students = 18) 
  (h2 : male_students = 848) 
  (h3 : female_student_diff = 49) : 
  (male_students + (male_students - female_student_diff) - absent_students = 1629) := 

by 
  sorry

end students_present_l135_135520


namespace initial_elephants_l135_135028

theorem initial_elephants (E : ℕ) :
  (E + 35 + 135 + 125 = 315) → (5 * 35 / 7 = 25) → (5 * 25 = 125) → (135 = 125 + 10) →
  E = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_elephants_l135_135028


namespace calc_correct_operation_l135_135737

theorem calc_correct_operation (a : ℕ) :
  (2 : ℕ) * a + (3 : ℕ) * a = (5 : ℕ) * a :=
by
  -- Proof
  sorry

end calc_correct_operation_l135_135737


namespace Penny_total_species_identified_l135_135482

/-- Penny identified 35 species of sharks, 15 species of eels, and 5 species of whales.
    Prove that the total number of species identified is 55. -/
theorem Penny_total_species_identified :
  let sharks_species := 35
  let eels_species := 15
  let whales_species := 5
  sharks_species + eels_species + whales_species = 55 :=
by
  sorry

end Penny_total_species_identified_l135_135482


namespace expand_product_l135_135963

theorem expand_product (x : ℝ) : 4 * (x + 3) * (x + 6) = 4 * x^2 + 36 * x + 72 :=
by
  sorry

end expand_product_l135_135963


namespace breadth_of_rectangular_plot_l135_135636

theorem breadth_of_rectangular_plot (b l A : ℕ) (h1 : A = 20 * b) (h2 : l = b + 10) 
    (h3 : A = l * b) : b = 10 := by
  sorry

end breadth_of_rectangular_plot_l135_135636


namespace solve_equation_l135_135990

theorem solve_equation (x : ℝ) (h₀ : x ≠ -3) (h₁ : (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 2) : x = 9 :=
by
  sorry

end solve_equation_l135_135990


namespace find_x_l135_135143

def balanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem find_x (x : ℝ) : 
  (∀ a b c d : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔ x ≥ 3 / 2 :=
sorry

end find_x_l135_135143


namespace interest_rate_part2_l135_135183

noncomputable def total_investment : ℝ := 3400
noncomputable def part1 : ℝ := 1300
noncomputable def part2 : ℝ := total_investment - part1
noncomputable def rate1 : ℝ := 0.03
noncomputable def total_interest : ℝ := 144
noncomputable def interest1 : ℝ := part1 * rate1
noncomputable def interest2 : ℝ := total_interest - interest1
noncomputable def rate2 : ℝ := interest2 / part2

theorem interest_rate_part2 : rate2 = 0.05 := sorry

end interest_rate_part2_l135_135183


namespace range_of_k_l135_135730

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x > 0 → (k+4) * x < 0) → k < -4 :=
by
  sorry

end range_of_k_l135_135730


namespace fish_kept_l135_135009

theorem fish_kept (Leo_caught Agrey_more Sierra_more Leo_fish Returned : ℕ) 
                  (Agrey_caught : Agrey_more = 20) 
                  (Sierra_caught : Sierra_more = 15) 
                  (Leo_caught_cond : Leo_fish = 40) 
                  (Returned_cond : Returned = 30) : 
                  (Leo_fish + (Leo_fish + Agrey_more) + ((Leo_fish + Agrey_more) + Sierra_more) - Returned) = 145 :=
by
  sorry

end fish_kept_l135_135009


namespace document_total_characters_l135_135100

theorem document_total_characters (T : ℕ) : 
  (∃ (t_1 t_2 t_3 : ℕ) (v_A v_B : ℕ),
      v_A = 100 ∧ v_B = 200 ∧
      t_1 = T / 600 ∧
      v_A * t_1 = T / 6 ∧
      v_B * t_1 = T / 3 ∧
      v_A * 3 * 5 = 1500 ∧
      t_2 = (T / 2 - 1500) / 500 ∧
      (v_A * 3 * t_2 + 1500 + v_A * t_1 = v_B * t_1 + v_B * t_2) ∧
      (v_A * 3 * (T - 3000) / 1000 + 1500 + v_A * T / 6 =
       v_B * 2 * (T - 3000) / 10 + v_B * T / 3)) →
  T = 18000 := by
  sorry

end document_total_characters_l135_135100


namespace exists_real_m_l135_135882

noncomputable def f (a : ℝ) (x : ℝ) := 4 * x + a * x ^ 2 - (2 / 3) * x ^ 3
noncomputable def g (x : ℝ) := 2 * x + (1 / 3) * x ^ 3

theorem exists_real_m (a : ℝ) (t : ℝ) (x1 x2 : ℝ) :
  (-1 : ℝ) ≤ a ∧ a ≤ 1 →
  (-1 : ℝ) ≤ t ∧ t ≤ 1 →
  f a x1 = g x1 ∧ f a x2 = g x2 →
  x1 ≠ 0 ∧ x2 ≠ 0 →
  x1 ≠ x2 →
  ∃ m : ℝ, (m ≥ 2 ∨ m ≤ -2) ∧ m^2 + t * m + 1 ≥ |x1 - x2| :=
sorry

end exists_real_m_l135_135882


namespace unique_xy_exists_l135_135579

theorem unique_xy_exists (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y) ^ 2 + 3 * x + y) / 2 := 
sorry

end unique_xy_exists_l135_135579


namespace fractional_eq_solution_l135_135392

theorem fractional_eq_solution (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 2) :
  (1 / (x - 1) = 2 / (x - 2)) → (x = 2) :=
by
  sorry

end fractional_eq_solution_l135_135392


namespace coefficient_and_degree_of_monomial_l135_135770

variable (x y : ℝ)

def monomial : ℝ := -2 * x * y^3

theorem coefficient_and_degree_of_monomial :
  ( ∃ c : ℝ, ∃ d : ℤ, monomial x y = c * x * y^d ∧ c = -2 ∧ d = 4 ) :=
by
  sorry

end coefficient_and_degree_of_monomial_l135_135770


namespace jury_deliberation_days_l135_135692

theorem jury_deliberation_days
  (jury_selection_days trial_times jury_duty_days deliberation_hours_per_day hours_in_day : ℕ)
  (h1 : jury_selection_days = 2)
  (h2 : trial_times = 4)
  (h3 : jury_duty_days = 19)
  (h4 : deliberation_hours_per_day = 16)
  (h5 : hours_in_day = 24) :
  (jury_duty_days - jury_selection_days - (trial_times * jury_selection_days)) * deliberation_hours_per_day / hours_in_day = 6 := 
by
  sorry

end jury_deliberation_days_l135_135692


namespace problem_divisibility_l135_135237

theorem problem_divisibility 
  (m n : ℕ) 
  (a : Fin (mn + 1) → ℕ)
  (h_pos : ∀ i, 0 < a i)
  (h_order : ∀ i j, i < j → a i < a j) :
  (∃ (b : Fin (m + 1) → Fin (mn + 1)), ∀ i j, i ≠ j → ¬(a (b i) ∣ a (b j))) ∨
  (∃ (c : Fin (n + 1) → Fin (mn + 1)), ∀ i, i < n → a (c i) ∣ a (c i.succ)) :=
sorry

end problem_divisibility_l135_135237


namespace athletes_meet_second_time_at_l135_135069

-- Define the conditions given in the problem
def distance_AB : ℕ := 110

def man_uphill_speed : ℕ := 3
def man_downhill_speed : ℕ := 5

def woman_uphill_speed : ℕ := 2
def woman_downhill_speed : ℕ := 3

-- Define the times for the athletes' round trips
def man_round_trip_time : ℚ := (distance_AB / man_uphill_speed) + (distance_AB / man_downhill_speed)
def woman_round_trip_time : ℚ := (distance_AB / woman_uphill_speed) + (distance_AB / woman_downhill_speed)

-- Lean statement for the proof
theorem athletes_meet_second_time_at :
  ∀ (t : ℚ), t = lcm (man_round_trip_time) (woman_round_trip_time) →
  ∃ d : ℚ, d = 330 / 7 := 
by sorry

end athletes_meet_second_time_at_l135_135069


namespace convert_to_rectangular_form_l135_135081

noncomputable def rectangular_form (z : ℂ) : ℂ :=
  let e := Complex.exp (13 * Real.pi * Complex.I / 6)
  3 * e

theorem convert_to_rectangular_form :
  rectangular_form (3 * Complex.exp (13 * Real.pi * Complex.I / 6)) = (3 * (Complex.cos (Real.pi / 6)) + 3 * Complex.I * (Complex.sin (Real.pi / 6))) :=
by
  sorry

end convert_to_rectangular_form_l135_135081


namespace mark_collects_money_l135_135235

variable (households_per_day : Nat)
variable (days : Nat)
variable (pair_amount : Nat)
variable (half_factor : Nat)

theorem mark_collects_money
  (h1 : households_per_day = 20)
  (h2 : days = 5)
  (h3 : pair_amount = 40)
  (h4 : half_factor = 2) :
  (households_per_day * days / half_factor) * pair_amount = 2000 :=
by
  sorry

end mark_collects_money_l135_135235


namespace two_digit_numbers_l135_135585

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem two_digit_numbers (a b : ℕ) (h1 : is_digit a) (h2 : is_digit b) 
  (h3 : a ≠ b) (h4 : (a + b) = 11) : 
  (∃ n m : ℕ, (n = 10 * a + b) ∧ (m = 10 * b + a) ∧ (∃ k : ℕ, (10 * a + b)^2 - (10 * b + a)^2 = k^2)) := 
sorry

end two_digit_numbers_l135_135585


namespace triangle_shape_l135_135377

theorem triangle_shape (a b : ℝ) (A B : ℝ)
  (h1 : a ≠ 0) (h2 : b ≠ 0) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π)
  (hTriangle : A + B + (π - A - B) = π)
  (h : a * Real.cos A = b * Real.cos B) : 
  (A = B ∨ A + B = π / 2) := sorry

end triangle_shape_l135_135377


namespace interest_for_1_rs_l135_135957

theorem interest_for_1_rs (I₅₀₀₀ : ℝ) (P : ℝ) (h : I₅₀₀₀ = 200) (hP : P = 5000) : I₅₀₀₀ / P = 0.04 :=
by
  rw [h, hP]
  norm_num

end interest_for_1_rs_l135_135957


namespace line_l_passes_fixed_point_line_l_perpendicular_value_a_l135_135696

variable (a : ℝ)

def line_l (a : ℝ) : ℝ × ℝ → Prop :=
  λ p => (a + 1) * p.1 + p.2 + 2 - a = 0

def perpendicular_line : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - 3 * p.2 + 4 = 0

theorem line_l_passes_fixed_point :
  line_l a (1, -3) :=
by
  sorry

theorem line_l_perpendicular_value_a (a : ℝ) :
  (∀ p : ℝ × ℝ, perpendicular_line p → line_l a p) → 
  a = 1 / 2 :=
by
  sorry

end line_l_passes_fixed_point_line_l_perpendicular_value_a_l135_135696


namespace megatek_manufacturing_percentage_l135_135477

theorem megatek_manufacturing_percentage :
  ∀ (total_degrees manufacturing_degrees total_percentage : ℝ),
  total_degrees = 360 → manufacturing_degrees = 216 → total_percentage = 100 →
  (manufacturing_degrees / total_degrees) * total_percentage = 60 :=
by
  intros total_degrees manufacturing_degrees total_percentage H1 H2 H3
  rw [H1, H2, H3]
  sorry

end megatek_manufacturing_percentage_l135_135477


namespace ratio_traditionalists_progressives_l135_135075

-- Define the given conditions
variables (T P C : ℝ)
variables (h1 : C = P + 4 * T)
variables (h2 : 4 * T = 0.75 * C)

-- State the theorem
theorem ratio_traditionalists_progressives (h1 : C = P + 4 * T) (h2 : 4 * T = 0.75 * C) : T / P = 3 / 4 :=
by {
  sorry
}

end ratio_traditionalists_progressives_l135_135075


namespace mean_value_of_pentagon_interior_angles_l135_135823

theorem mean_value_of_pentagon_interior_angles :
  let n := 5
  let sum_of_interior_angles := (n - 2) * 180
  let mean_value := sum_of_interior_angles / n
  mean_value = 108 :=
by
  sorry

end mean_value_of_pentagon_interior_angles_l135_135823


namespace rectangular_plot_width_l135_135927

/-- Theorem: The width of a rectangular plot where the length is thrice its width and the area is 432 sq meters is 12 meters. -/
theorem rectangular_plot_width (w l : ℝ) (h₁ : l = 3 * w) (h₂ : l * w = 432) : w = 12 :=
by
  sorry

end rectangular_plot_width_l135_135927


namespace part_I_part_II_l135_135833

noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) - (2 * x) / (x + 2)
noncomputable def g (x : ℝ) : ℝ := f x - (4 / (x + 2))

theorem part_I (x : ℝ) (h₀ : 0 < x) : f x > 0 := sorry

theorem part_II (a : ℝ) (h : ∀ x, g x < x + a) : -2 < a := sorry

end part_I_part_II_l135_135833


namespace arccos_gt_arctan_on_interval_l135_135619

noncomputable def c : ℝ := sorry -- placeholder for the numerical solution of arccos x = arctan x

theorem arccos_gt_arctan_on_interval (x : ℝ) (hx : -1 ≤ x ∧ x < c) :
  Real.arccos x > Real.arctan x := 
sorry

end arccos_gt_arctan_on_interval_l135_135619


namespace sequence_a7_l135_135590

theorem sequence_a7 (a b : ℕ) (h1 : a1 = a) (h2 : a2 = b) {a3 a4 a5 a6 a7 : ℕ}
  (h3 : a_3 = a + b)
  (h4 : a_4 = a + 2 * b)
  (h5 : a_5 = 2 * a + 3 * b)
  (h6 : a_6 = 3 * a + 5 * b)
  (h_a6 : a_6 = 50) :
  a_7 = 5 * a + 8 * b :=
by
  sorry

end sequence_a7_l135_135590


namespace amount_each_student_should_pay_l135_135643

noncomputable def total_rental_fee_per_book_per_half_hour : ℕ := 4000 
noncomputable def total_books : ℕ := 4
noncomputable def total_students : ℕ := 6
noncomputable def total_hours : ℕ := 3
noncomputable def total_half_hours : ℕ := total_hours * 2

noncomputable def total_fee_one_book : ℕ := total_rental_fee_per_book_per_half_hour * total_half_hours
noncomputable def total_fee_all_books : ℕ := total_fee_one_book * total_books

theorem amount_each_student_should_pay : total_fee_all_books / total_students = 16000 := by
  sorry

end amount_each_student_should_pay_l135_135643


namespace cost_difference_is_360_l135_135316

def sailboat_cost_per_day : ℕ := 60
def ski_boat_cost_per_hour : ℕ := 80
def ken_days : ℕ := 2
def aldrich_hours_per_day : ℕ := 3
def aldrich_days : ℕ := 2

theorem cost_difference_is_360 :
  let ken_total_cost := sailboat_cost_per_day * ken_days
  let aldrich_total_cost_per_day := ski_boat_cost_per_hour * aldrich_hours_per_day
  let aldrich_total_cost := aldrich_total_cost_per_day * aldrich_days
  let cost_diff := aldrich_total_cost - ken_total_cost
  cost_diff = 360 :=
by
  sorry

end cost_difference_is_360_l135_135316


namespace pencils_left_l135_135434

def initial_pencils : Nat := 127
def pencils_from_joyce : Nat := 14
def pencils_per_friend : Nat := 7

theorem pencils_left : ((initial_pencils + pencils_from_joyce) % pencils_per_friend) = 1 := by
  sorry

end pencils_left_l135_135434


namespace golf_tournament_percentage_increase_l135_135856

theorem golf_tournament_percentage_increase:
  let electricity_bill := 800
  let cell_phone_expenses := electricity_bill + 400
  let golf_tournament_cost := 1440
  (golf_tournament_cost - cell_phone_expenses) / cell_phone_expenses * 100 = 20 :=
by
  sorry

end golf_tournament_percentage_increase_l135_135856


namespace factor_expression_l135_135489

theorem factor_expression (c : ℝ) : 180 * c ^ 2 + 36 * c = 36 * c * (5 * c + 1) := 
by
  sorry

end factor_expression_l135_135489


namespace number_of_terms_in_ap_is_eight_l135_135726

theorem number_of_terms_in_ap_is_eight
  (n : ℕ) (a d : ℝ)
  (even : n % 2 = 0)
  (sum_odd : (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 24)
  (sum_even : (n / 2 : ℝ) * (2 * a + n * d) = 30)
  (last_exceeds_first : (n - 1) * d = 10.5) :
  n = 8 :=
by sorry

end number_of_terms_in_ap_is_eight_l135_135726


namespace floor_square_of_sqrt_50_eq_49_l135_135637

theorem floor_square_of_sqrt_50_eq_49 : (Int.floor (Real.sqrt 50))^2 = 49 := 
by
  sorry

end floor_square_of_sqrt_50_eq_49_l135_135637


namespace gcd_factorial_l135_135355

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l135_135355


namespace find_c_d_l135_135047

theorem find_c_d (C D : ℤ) (h1 : 3 * C - 4 * D = 18) (h2 : C = 2 * D - 5) :
  C = 28 ∧ D = 33 / 2 := by
sorry

end find_c_d_l135_135047


namespace mirasol_initial_amount_l135_135370

/-- 
Mirasol had some money in her account. She spent $10 on coffee beans and $30 on a tumbler. She has $10 left in her account.
Prove that the initial amount of money Mirasol had in her account is $50.
-/
theorem mirasol_initial_amount (spent_coffee : ℕ) (spent_tumbler : ℕ) (left_in_account : ℕ) :
  spent_coffee = 10 → spent_tumbler = 30 → left_in_account = 10 → 
  spent_coffee + spent_tumbler + left_in_account = 50 := 
by
  sorry

end mirasol_initial_amount_l135_135370


namespace interest_difference_correct_l135_135406

-- Define the basic parameters and constants
def principal : ℝ := 147.69
def rate : ℝ := 0.15
def time1 : ℝ := 3.5
def time2 : ℝ := 10
def interest1 : ℝ := principal * rate * time1
def interest2 : ℝ := principal * rate * time2
def difference : ℝ := 143.998

-- Theorem statement: The difference between the interests is approximately Rs. 143.998
theorem interest_difference_correct :
  interest2 - interest1 = difference := sorry

end interest_difference_correct_l135_135406


namespace average_sitting_time_per_student_l135_135560

def total_travel_time_in_minutes : ℕ := 152
def number_of_seats : ℕ := 5
def number_of_students : ℕ := 8

theorem average_sitting_time_per_student :
  (total_travel_time_in_minutes * number_of_seats) / number_of_students = 95 := 
by
  sorry

end average_sitting_time_per_student_l135_135560


namespace cost_price_of_article_l135_135038

noncomputable def cost_price (M : ℝ) : ℝ := 98.68 / 1.25

theorem cost_price_of_article (M : ℝ)
    (h1 : 0.95 * M = 98.68)
    (h2 : 98.68 = 1.25 * cost_price M) :
    cost_price M = 78.944 :=
by sorry

end cost_price_of_article_l135_135038


namespace camel_height_in_feet_l135_135510

theorem camel_height_in_feet (h_ht_14 : ℕ) (ratio : ℕ) (inch_to_ft : ℕ) : ℕ :=
  let hare_height := 14
  let camel_height_in_inches := hare_height * 24
  let camel_height_in_feet := camel_height_in_inches / 12
  camel_height_in_feet
#print camel_height_in_feet

example : camel_height_in_feet 14 24 12 = 28 := by sorry

end camel_height_in_feet_l135_135510


namespace number_of_technicians_l135_135299

theorem number_of_technicians
  (total_workers : ℕ)
  (avg_salary_all : ℝ)
  (avg_salary_techs : ℝ)
  (avg_salary_rest : ℝ)
  (num_techs num_rest : ℕ)
  (h_total_workers : total_workers = 56)
  (h_avg_salary_all : avg_salary_all = 6750)
  (h_avg_salary_techs : avg_salary_techs = 12000)
  (h_avg_salary_rest : avg_salary_rest = 6000)
  (h_eq_workers : num_techs + num_rest = total_workers)
  (h_eq_salaries : (num_techs * avg_salary_techs + num_rest * avg_salary_rest) = total_workers * avg_salary_all) :
  num_techs = 7 := sorry

end number_of_technicians_l135_135299


namespace isosceles_triangle_perimeter_l135_135659

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4) (h2 : b = 8) (h3 : ∃ p q r, p = b ∧ q = b ∧ r = a ∧ p + q > r) : 
  a + b + b = 20 := 
by 
  sorry

end isosceles_triangle_perimeter_l135_135659


namespace ratio_A_to_B_l135_135413

/--
Proof problem statement:
Given that A and B together can finish the work in 4 days,
and B alone can finish the work in 24 days,
prove that the ratio of the time A takes to finish the work to the time B takes to finish the work is 1:5.
-/
theorem ratio_A_to_B
  (A_time B_time working_together_time : ℝ) 
  (h1 : working_together_time = 4)
  (h2 : B_time = 24)
  (h3 : 1 / A_time + 1 / B_time = 1 / working_together_time) :
  A_time / B_time = 1 / 5 :=
sorry

end ratio_A_to_B_l135_135413


namespace heartsuit_calc_l135_135802

def heartsuit (u v : ℝ) : ℝ := (u + 2*v) * (u - v)

theorem heartsuit_calc : heartsuit 2 (heartsuit 3 4) = -260 := by
  sorry

end heartsuit_calc_l135_135802


namespace max_det_bound_l135_135890

noncomputable def max_det_estimate : ℕ := 327680 * 2^16

theorem max_det_bound (M : Matrix (Fin 17) (Fin 17) ℤ)
  (h : ∀ i j, M i j = 1 ∨ M i j = -1) :
  abs (Matrix.det M) ≤ max_det_estimate :=
sorry

end max_det_bound_l135_135890


namespace area_ratio_correct_l135_135149

noncomputable def area_ratio_of_ABC_and_GHJ : ℝ :=
  let side_length_ABC := 12
  let BD := 5
  let CE := 5
  let AF := 8
  let area_ABC := (Real.sqrt 3 / 4) * side_length_ABC ^ 2
  (1 / 74338) * area_ABC / area_ABC

theorem area_ratio_correct : area_ratio_of_ABC_and_GHJ = 1 / 74338 := by
  sorry

end area_ratio_correct_l135_135149


namespace sum_of_three_numbers_l135_135898

theorem sum_of_three_numbers :
  ∃ (S1 S2 S3 : ℕ), 
    S2 = 72 ∧
    S1 = 2 * S2 ∧
    S3 = S1 / 3 ∧
    S1 + S2 + S3 = 264 := 
by
  sorry

end sum_of_three_numbers_l135_135898


namespace original_number_of_men_l135_135122

theorem original_number_of_men 
  (x : ℕ)
  (H : 15 * 18 * x = 15 * 18 * (x - 8) + 8 * 15 * 18)
  (h_pos : x > 8) :
  x = 40 :=
sorry

end original_number_of_men_l135_135122


namespace range_of_m_l135_135182

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x-1)^2
  else if x > 0 then -(x+1)^2
  else 0

theorem range_of_m (m : ℝ) (h : f (m^2 + 2*m) + f m > 0) : -3 < m ∧ m < 0 := 
by {
  sorry
}

end range_of_m_l135_135182


namespace max_c_for_range_l135_135932

theorem max_c_for_range (c : ℝ) :
  (∃ x : ℝ, (x^2 - 7*x + c = 2)) → c ≤ 57 / 4 :=
by
  sorry

end max_c_for_range_l135_135932


namespace parallel_lines_a_eq_3_div_2_l135_135174

theorem parallel_lines_a_eq_3_div_2 (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → a = 3 / 2 :=
by sorry

end parallel_lines_a_eq_3_div_2_l135_135174


namespace verify_value_of_2a10_minus_a12_l135_135037

-- Define the arithmetic sequence and the sum condition
variable {a : ℕ → ℝ}  -- arithmetic sequence
variable {a1 : ℝ}     -- the first term of the sequence
variable {d : ℝ}      -- the common difference of the sequence

-- Assume that the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + n * d

-- Assume the sum condition
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- The goal is to prove that 2 * a 10 - a 12 = 24
theorem verify_value_of_2a10_minus_a12 (h_arith : arithmetic_sequence a a1 d) (h_sum : sum_condition a) :
  2 * a 10 - a 12 = 24 :=
  sorry

end verify_value_of_2a10_minus_a12_l135_135037


namespace pentagon_area_l135_135242

-- Definitions of the vertices of the pentagon
def vertices : List (ℝ × ℝ) :=
  [(0, 0), (1, 2), (3, 3), (4, 1), (2, 0)]

-- Definition of the number of interior points
def interior_points : ℕ := 7

-- Definition of the number of boundary points
def boundary_points : ℕ := 5

-- Pick's theorem: Area = Interior points + Boundary points / 2 - 1
noncomputable def area : ℝ :=
  interior_points + boundary_points / 2 - 1

-- Theorem to be proved
theorem pentagon_area :
  area = 8.5 :=
by
  sorry

end pentagon_area_l135_135242


namespace solve_for_x_l135_135703

theorem solve_for_x
  (x y : ℝ)
  (h1 : x + 2 * y = 100)
  (h2 : y = 25) :
  x = 50 :=
by
  sorry

end solve_for_x_l135_135703


namespace evaluate_expression_l135_135263

theorem evaluate_expression : (2^3001 * 3^3003) / 6^3002 = 3 / 2 :=
by
  sorry

end evaluate_expression_l135_135263


namespace duty_pairing_impossible_l135_135545

theorem duty_pairing_impossible :
  ∀ (m n : ℕ), 29 * m + 32 * n ≠ 29 * 32 := 
by 
  sorry

end duty_pairing_impossible_l135_135545


namespace probability_of_drawing_red_ball_l135_135167

theorem probability_of_drawing_red_ball :
  let red_balls := 7
  let black_balls := 3
  let total_balls := red_balls + black_balls
  let probability_red := (red_balls : ℚ) / total_balls
  probability_red = 7 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l135_135167
