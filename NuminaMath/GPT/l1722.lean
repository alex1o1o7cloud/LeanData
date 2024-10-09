import Mathlib

namespace alex_initial_jelly_beans_l1722_172264

variable (initial : ℕ)
variable (eaten : ℕ := 6)
variable (pile_weight : ℕ := 10)
variable (piles : ℕ := 3)

theorem alex_initial_jelly_beans :
  (initial - eaten = pile_weight * piles) → initial = 36 :=
by
  -- proof will be provided here
  sorry

end alex_initial_jelly_beans_l1722_172264


namespace shaded_fraction_in_fifth_diagram_l1722_172278

-- Definitions for conditions
def geometric_sequence (a₀ r n : ℕ) : ℕ := a₀ * r^n

def total_triangles (n : ℕ) : ℕ := n^2

-- Lean theorem statement
theorem shaded_fraction_in_fifth_diagram 
  (a₀ r n : ℕ) 
  (h_geometric : a₀ = 1) 
  (h_ratio : r = 2)
  (h_step_number : n = 4):
  (geometric_sequence a₀ r n) / (total_triangles (n + 1)) = 16 / 25 :=
by
  sorry

end shaded_fraction_in_fifth_diagram_l1722_172278


namespace triangle_area_proof_l1722_172284

-- Define the triangle sides and median
variables (AB BC BD AC : ℝ)

-- Assume given values
def AB_value : AB = 1 := by sorry 
def BC_value : BC = Real.sqrt 15 := by sorry
def BD_value : BD = 2 := by sorry

-- Assume AC calculated from problem
def AC_value : AC = 4 := by sorry

-- Final proof statement
theorem triangle_area_proof 
  (hAB : AB = 1)
  (hBC : BC = Real.sqrt 15)
  (hBD : BD = 2)
  (hAC : AC = 4) :
  (1 / 2) * AB * BC = (Real.sqrt 15) / 2 := 
sorry

end triangle_area_proof_l1722_172284


namespace ratio_of_fifth_to_second_l1722_172294

-- Definitions based on the conditions
def first_stack := 7
def second_stack := first_stack + 3
def third_stack := second_stack - 6
def fourth_stack := third_stack + 10

def total_blocks := 55

-- The number of blocks in the fifth stack
def fifth_stack := total_blocks - (first_stack + second_stack + third_stack + fourth_stack)

-- The ratio of the fifth stack to the second stack
def ratio := fifth_stack / second_stack

-- The theorem we want to prove
theorem ratio_of_fifth_to_second: ratio = 2 := by
  sorry

end ratio_of_fifth_to_second_l1722_172294


namespace probability_region_D_l1722_172255

noncomputable def P_A : ℝ := 1 / 4
noncomputable def P_B : ℝ := 1 / 3
noncomputable def P_C : ℝ := 1 / 6

theorem probability_region_D (P_D : ℝ) (h : P_A + P_B + P_C + P_D = 1) : P_D = 1 / 4 :=
by
  sorry

end probability_region_D_l1722_172255


namespace infinite_solutions_exists_l1722_172268

theorem infinite_solutions_exists : 
  ∃ (S : Set (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ S → 2 * a^2 - 3 * a + 1 = 3 * b^2 + b) 
  ∧ Set.Infinite S :=
sorry

end infinite_solutions_exists_l1722_172268


namespace probability_of_C_l1722_172258

theorem probability_of_C (P : ℕ → ℚ) (P_total : P 1 + P 2 + P 3 = 1)
  (P_A : P 1 = 1/3) (P_B : P 2 = 1/2) : P 3 = 1/6 :=
by
  sorry

end probability_of_C_l1722_172258


namespace evaluate_series_l1722_172279

-- Define the series S
noncomputable def S : ℝ := ∑' n : ℕ, (n + 1) / (3 ^ (n + 1))

-- Lean statement to show the evaluated series
theorem evaluate_series : (3:ℝ)^S = (3:ℝ)^(3 / 4) :=
by
  -- The proof is omitted
  sorry

end evaluate_series_l1722_172279


namespace geometric_sequence_first_term_and_ratio_l1722_172210

theorem geometric_sequence_first_term_and_ratio (b : ℕ → ℚ) 
  (hb2 : b 2 = 37 + 1/3) 
  (hb6 : b 6 = 2 + 1/3) : 
  ∃ (b1 q : ℚ), b 1 = b1 ∧ (∀ n, b n = b1 * q^(n-1)) ∧ b1 = 224 / 3 ∧ q = 1 / 2 :=
by 
  sorry

end geometric_sequence_first_term_and_ratio_l1722_172210


namespace small_boxes_in_big_box_l1722_172285

theorem small_boxes_in_big_box (total_candles : ℕ) (candles_per_small : ℕ) (total_big_boxes : ℕ) 
  (h1 : total_candles = 8000) 
  (h2 : candles_per_small = 40) 
  (h3 : total_big_boxes = 50) :
  (total_candles / candles_per_small) / total_big_boxes = 4 :=
by
  sorry

end small_boxes_in_big_box_l1722_172285


namespace plot_length_l1722_172263

theorem plot_length (b : ℕ) (cost_per_meter total_cost : ℕ)
  (h1 : cost_per_meter = 2650 / 100)  -- Since Lean works with integers, use 2650 instead of 26.50
  (h2 : total_cost = 5300)
  (h3 : 2 * (b + 16) + 2 * b = total_cost / cost_per_meter) :
  b + 16 = 58 :=
by
  -- Above theorem aims to prove the length of the plot is 58 meters, given the conditions.
  sorry

end plot_length_l1722_172263


namespace total_matches_round_robin_l1722_172216

/-- A round-robin chess tournament is organized in two groups with different numbers of players. 
Group A consists of 6 players, and Group B consists of 5 players. 
Each player in each group plays every other player in the same group exactly once. 
Prove that the total number of matches is 25. -/
theorem total_matches_round_robin 
  (nA : ℕ) (nB : ℕ) 
  (hA : nA = 6) (hB : nB = 5) : 
  (nA * (nA - 1) / 2) + (nB * (nB - 1) / 2) = 25 := 
  by
    sorry

end total_matches_round_robin_l1722_172216


namespace circular_garden_area_l1722_172297

theorem circular_garden_area
  (r : ℝ) (h_r : r = 16)
  (C A : ℝ) (h_C : C = 2 * Real.pi * r) (h_A : A = Real.pi * r^2)
  (fence_cond : C = 1 / 8 * A) :
  A = 256 * Real.pi := by
  sorry

end circular_garden_area_l1722_172297


namespace range_of_z_in_parallelogram_l1722_172225

-- Define the points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := {x := -1, y := 2}
def B : Point := {x := 3, y := 4}
def C : Point := {x := 4, y := -2}

-- Define the condition for point (x, y) to be inside the parallelogram (including boundary)
def isInsideParallelogram (p : Point) : Prop := sorry -- Placeholder for actual geometric condition

-- Statement of the problem
theorem range_of_z_in_parallelogram (p : Point) (h : isInsideParallelogram p) : 
  -14 ≤ 2 * p.x - 5 * p.y ∧ 2 * p.x - 5 * p.y ≤ 20 :=
sorry

end range_of_z_in_parallelogram_l1722_172225


namespace winnie_keeps_10_lollipops_l1722_172283

def winnie_keep_lollipops : Prop :=
  let cherry := 72
  let wintergreen := 89
  let grape := 23
  let shrimp_cocktail := 316
  let total_lollipops := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 14
  let lollipops_per_friend := total_lollipops / friends
  let winnie_keeps := total_lollipops % friends
  winnie_keeps = 10

theorem winnie_keeps_10_lollipops : winnie_keep_lollipops := by
  sorry

end winnie_keeps_10_lollipops_l1722_172283


namespace lcm_of_pack_sizes_l1722_172296

theorem lcm_of_pack_sizes :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 13 19) 8) 11) 17) 23 = 772616 := by
  sorry

end lcm_of_pack_sizes_l1722_172296


namespace seating_arrangement_l1722_172266

theorem seating_arrangement (n x : ℕ) (h1 : 7 * x + 6 * (n - x) = 53) : x = 5 :=
sorry

end seating_arrangement_l1722_172266


namespace overall_gain_percentage_l1722_172209

def cost_of_A : ℝ := 100
def selling_price_of_A : ℝ := 125
def cost_of_B : ℝ := 200
def selling_price_of_B : ℝ := 250
def cost_of_C : ℝ := 150
def selling_price_of_C : ℝ := 180

theorem overall_gain_percentage :
  ((selling_price_of_A + selling_price_of_B + selling_price_of_C) - (cost_of_A + cost_of_B + cost_of_C)) / (cost_of_A + cost_of_B + cost_of_C) * 100 = 23.33 := 
by
  sorry

end overall_gain_percentage_l1722_172209


namespace cos_B_in_triangle_l1722_172240

theorem cos_B_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : a = (Real.sqrt 5 / 2) * b)
  (h2 : A = 2 * B)
  (h_triangle: A + B + C = Real.pi) : 
  Real.cos B = Real.sqrt 5 / 4 :=
sorry

end cos_B_in_triangle_l1722_172240


namespace max_sum_mult_table_l1722_172298

def isEven (n : ℕ) : Prop := n % 2 = 0
def isOdd (n : ℕ) : Prop := ¬ isEven n
def entries : List ℕ := [3, 4, 6, 8, 9, 12]
def sumOfList (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem max_sum_mult_table :
  ∃ (a b c d e f : ℕ), 
    a ∈ entries ∧ b ∈ entries ∧ c ∈ entries ∧ 
    d ∈ entries ∧ e ∈ entries ∧ f ∈ entries ∧ 
    (isEven a ∧ isEven b ∧ isOdd c ∨ isEven a ∧ isOdd b ∧ isOdd c ∨ isOdd a ∧ isEven b ∧ isEven c ∨ isOdd a ∧ isOdd b ∧ isOdd c ∨ isOdd a ∧ isEven b ∧ isOdd c ∨ isEven a ∧ isOdd b ∧ isEven c) ∧ 
    (isEven d ∧ isEven e ∧ isOdd f ∨ isEven d ∧ isOdd e ∧ isOdd f ∨ isOdd d ∧ isEven e ∧ isEven f ∨ isOdd d ∧ isOdd e ∧ isOdd f ∨ isOdd d ∧ isEven e ∧ isOdd f ∨ isEven d ∧ isOdd e ∧ isEven f) ∧ 
    (sumOfList [a, b, c] * sumOfList [d, e, f] = 425) := 
by
    sorry  -- Skipping the proof as instructed.

end max_sum_mult_table_l1722_172298


namespace complete_square_l1722_172253

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) ↔ ((x - 1)^2 = 6) := 
by
  sorry

end complete_square_l1722_172253


namespace unique_three_digit_base_g_l1722_172235

theorem unique_three_digit_base_g (g : ℤ) (h : ℤ) (a b c : ℤ) 
  (hg : g > 2) 
  (h_h : h = g + 1 ∨ h = g - 1) 
  (habc_g : a * g^2 + b * g + c = c * h^2 + b * h + a) : 
  a = (g + 1) / 2 ∧ b = (g - 1) / 2 ∧ c = (g - 1) / 2 :=
  sorry

end unique_three_digit_base_g_l1722_172235


namespace find_k_find_a_l1722_172245

noncomputable def f (a k : ℝ) (x : ℝ) := a ^ x + k * a ^ (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

theorem find_k (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : is_odd_function (f a k)) : k = -1 :=
sorry

theorem find_a (k : ℝ) (h₃ : k = -1) (h₄ : f 1 = 3 / 2) (h₅ : is_monotonic_increasing (f 2 k)) : a = 2 :=
sorry

end find_k_find_a_l1722_172245


namespace necessary_and_sufficient_l1722_172265

def point_on_curve (P : ℝ × ℝ) (f : ℝ × ℝ → ℝ) : Prop :=
  f P = 0

theorem necessary_and_sufficient (P : ℝ × ℝ) (f : ℝ × ℝ → ℝ) :
  (point_on_curve P f ↔ f P = 0) :=
by
  sorry

end necessary_and_sufficient_l1722_172265


namespace part1_range_a_part2_range_a_l1722_172260

-- Definitions of the propositions
def p (a : ℝ) := ∃ x : ℝ, x^2 + a * x + 2 = 0

def q (a : ℝ) := ∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - a < 0

-- Part 1: If p is true, find the range of values for a
theorem part1_range_a (a : ℝ) :
  p a → (a ≤ -2*Real.sqrt 2 ∨ a ≥ 2*Real.sqrt 2) := sorry

-- Part 2: If one of p or q is true and the other is false, find the range of values for a
theorem part2_range_a (a : ℝ) :
  (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a) →
  (a ≤ -2*Real.sqrt 2 ∨ (1 ≤ a ∧ a < 2*Real.sqrt 2)) := sorry

end part1_range_a_part2_range_a_l1722_172260


namespace cos_two_pi_over_three_l1722_172256

theorem cos_two_pi_over_three : Real.cos (2 * Real.pi / 3) = -1 / 2 :=
by sorry

end cos_two_pi_over_three_l1722_172256


namespace percentage_goods_lost_l1722_172293

theorem percentage_goods_lost
    (cost_price selling_price loss_price : ℝ)
    (profit_percent loss_percent : ℝ)
    (h_profit : selling_price = cost_price * (1 + profit_percent / 100))
    (h_loss_value : loss_price = selling_price * (loss_percent / 100))
    (cost_price_assumption : cost_price = 100)
    (profit_percent_assumption : profit_percent = 10)
    (loss_percent_assumption : loss_percent = 45) :
    (loss_price / cost_price * 100) = 49.5 :=
sorry

end percentage_goods_lost_l1722_172293


namespace problem1_problem2_problem3_l1722_172213

theorem problem1 : (-3) - (-5) - 6 + (-4) = -8 := by sorry

theorem problem2 : ((1 / 9) + (1 / 6) - (1 / 2)) / (-1 / 18) = 4 := by sorry

theorem problem3 : -1^4 + abs (3 - 6) - 2 * (-2) ^ 2 = -6 := by sorry

end problem1_problem2_problem3_l1722_172213


namespace patty_fraction_3mph_l1722_172277

noncomputable def fraction_time_at_3mph (t3 t6 : ℝ) (h : 3 * t3 + 6 * t6 = 5 * (t3 + t6)) : ℝ :=
  t3 / (t3 + t6)

theorem patty_fraction_3mph (t3 t6 : ℝ) (h : 3 * t3 + 6 * t6 = 5 * (t3 + t6)) :
  fraction_time_at_3mph t3 t6 h = 1 / 3 :=
by
  sorry

end patty_fraction_3mph_l1722_172277


namespace compute_fraction_l1722_172286

theorem compute_fraction :
  (1 * 2 + 2 * 4 - 3 * 8 + 4 * 16 + 5 * 32 - 6 * 64) /
  (2 * 4 + 4 * 8 - 6 * 16 + 8 * 32 + 10 * 64 - 12 * 128) =
  1 / 4 :=
by
  -- Proof will go here
  sorry

end compute_fraction_l1722_172286


namespace village_duration_l1722_172252

theorem village_duration (vampire_drain : ℕ) (werewolf_eat : ℕ) (village_population : ℕ)
  (hv : vampire_drain = 3) (hw : werewolf_eat = 5) (hp : village_population = 72) :
  village_population / (vampire_drain + werewolf_eat) = 9 :=
by
  sorry

end village_duration_l1722_172252


namespace first_problem_second_problem_l1722_172226

variable (x : ℝ)

-- Proof for the first problem
theorem first_problem : 6 * x^3 / (-3 * x^2) = -2 * x := by
sorry

-- Proof for the second problem
theorem second_problem : (2 * x + 3) * (2 * x - 3) - 4 * (x - 2)^2 = 16 * x - 25 := by
sorry

end first_problem_second_problem_l1722_172226


namespace Amanda_family_paint_walls_l1722_172250

theorem Amanda_family_paint_walls :
  let num_people := 5
  let rooms_with_4_walls := 5
  let rooms_with_5_walls := 4
  let walls_per_room_4 := 4
  let walls_per_room_5 := 5
  let total_walls := (rooms_with_4_walls * walls_per_room_4) + (rooms_with_5_walls * walls_per_room_5)
  total_walls / num_people = 8 :=
by
  -- We add a sorry to skip proof
  sorry

end Amanda_family_paint_walls_l1722_172250


namespace percentage_of_part_over_whole_l1722_172203

theorem percentage_of_part_over_whole (Part Whole : ℕ) (h1 : Part = 120) (h2 : Whole = 50) :
  (Part / Whole : ℚ) * 100 = 240 := by
  sorry

end percentage_of_part_over_whole_l1722_172203


namespace average_production_last_5_days_l1722_172230

theorem average_production_last_5_days (tv_per_day_25 : ℕ) (total_tv_30 : ℕ) :
  tv_per_day_25 = 63 →
  total_tv_30 = 58 * 30 →
  (total_tv_30 - tv_per_day_25 * 25) / 5 = 33 :=
by
  intros h1 h2
  sorry

end average_production_last_5_days_l1722_172230


namespace max_value_expression_l1722_172248

theorem max_value_expression (x y : ℝ) (h : x * y > 0) : 
  ∃ (m : ℝ), (∀ x y : ℝ, x * y > 0 → 
  m ≥ (x / (x + y) + 2 * y / (x + 2 * y))) ∧ 
  m = 4 - 2 * Real.sqrt 2 := 
sorry

end max_value_expression_l1722_172248


namespace percentage_problem_l1722_172287

theorem percentage_problem
  (a b c : ℚ) :
  (8 = (2 / 100) * a) →
  (2 = (8 / 100) * b) →
  (c = b / a) →
  c = 1 / 16 :=
by
  sorry

end percentage_problem_l1722_172287


namespace boy_completion_time_l1722_172218

theorem boy_completion_time (M W B : ℝ) (h1 : M + W + B = 1/3) (h2 : M = 1/6) (h3 : W = 1/18) : B = 1/9 :=
sorry

end boy_completion_time_l1722_172218


namespace tan_585_eq_1_l1722_172295

theorem tan_585_eq_1 : Real.tan (585 * Real.pi / 180) = 1 := 
by
  sorry

end tan_585_eq_1_l1722_172295


namespace find_geometric_sequence_first_term_and_ratio_l1722_172232

theorem find_geometric_sequence_first_term_and_ratio 
  (a1 a2 a3 a4 a5 : ℕ) 
  (h : a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5)
  (geo_seq : a2 = a1 * 3 / 2 ∧ a3 = a2 * 3 / 2 ∧ a4 = a3 * 3 / 2 ∧ a5 = a4 * 3 / 2)
  (sum_cond : a1 + a2 + a3 + a4 + a5 = 211) :
  (a1 = 16) ∧ (3 / 2 = 3 / 2) := 
by {
  sorry
}

end find_geometric_sequence_first_term_and_ratio_l1722_172232


namespace overlapping_triangle_area_l1722_172208

/-- Given a rectangle with length 8 and width 4, folded along its diagonal, 
    the area of the overlapping part (grey triangle) is 10. --/
theorem overlapping_triangle_area : 
  let length := 8 
  let width := 4 
  let diagonal := (length^2 + width^2)^(1/2) 
  let base := (length^2 / (width^2 + length^2))^(1/2) * width 
  let height := width
  1 / 2 * base * height = 10 := by 
  sorry

end overlapping_triangle_area_l1722_172208


namespace area_of_park_l1722_172241

variable (length breadth speed time perimeter area : ℕ)

axiom ratio_length_breadth : length = breadth / 4
axiom speed_kmh : speed = 12 * 1000 / 60 -- speed in m/min
axiom time_taken : time = 8 -- time in minutes
axiom perimeter_eq : perimeter = speed * time -- perimeter in meters
axiom length_breadth_relation : perimeter = 2 * (length + breadth)

theorem area_of_park : ∃ length breadth, (length = 160 ∧ breadth = 640 ∧ area = length * breadth ∧ area = 102400) :=
by
  sorry

end area_of_park_l1722_172241


namespace neg_P_l1722_172259

-- Define the proposition P
def P : Prop := ∃ x : ℝ, Real.exp x ≤ 0

-- State the negation of P
theorem neg_P : ¬P ↔ ∀ x : ℝ, Real.exp x > 0 := 
by 
  sorry

end neg_P_l1722_172259


namespace angle_sum_at_F_l1722_172201

theorem angle_sum_at_F (x y z w v : ℝ) (h : x + y + z + w + v = 360) : 
  x = 360 - y - z - w - v := by
  sorry

end angle_sum_at_F_l1722_172201


namespace line_points_product_l1722_172238

theorem line_points_product (x y : ℝ) (h1 : 8 = (1/4 : ℝ) * x) (h2 : y = (1/4 : ℝ) * 20) : x * y = 160 := 
by
  sorry

end line_points_product_l1722_172238


namespace arithmetic_sequence_unique_a_l1722_172227

theorem arithmetic_sequence_unique_a (a : ℝ) (b : ℕ → ℝ) (a_seq : ℕ → ℝ)
  (h1 : a_seq 1 = a) (h2 : a > 0)
  (h3 : b 1 - a_seq 1 = 1) (h4 : b 2 - a_seq 2 = 2)
  (h5 : b 3 - a_seq 3 = 3)
  (unique_a : ∀ (a' : ℝ), (a_seq 1 = a' ∧ a' > 0 ∧ b 1 - a' = 1 ∧ b 2 - a_seq 2 = 2 ∧ b 3 - a_seq 3 = 3) → a' = a) :
  a = 1 / 3 :=
by
  sorry

end arithmetic_sequence_unique_a_l1722_172227


namespace cos_identity_l1722_172275

theorem cos_identity 
  (x : ℝ) 
  (h : Real.sin (x - π / 3) = 3 / 5) : 
  Real.cos (x + π / 6) = -3 / 5 := 
by 
  sorry

end cos_identity_l1722_172275


namespace find_B_intersection_point_l1722_172219

theorem find_B_intersection_point (k1 k2 : ℝ) (hA1 : 1 ≠ 0) 
  (hA2 : k1 = -2) (hA3 : k2 = -2) : 
  (-1, 2) ∈ {p : ℝ × ℝ | ∃ k1 k2, p.2 = k1 * p.1 ∧ p.2 = k2 / p.1} :=
sorry

end find_B_intersection_point_l1722_172219


namespace maximize_revenue_l1722_172281

-- Define the conditions
def total_time_condition (x y : ℝ) : Prop := x + y ≤ 300
def total_cost_condition (x y : ℝ) : Prop := 2.5 * x + y ≤ 4500
def non_negative_condition (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0

-- Define the revenue function
def revenue (x y : ℝ) : ℝ := 0.3 * x + 0.2 * y

-- The proof statement
theorem maximize_revenue : 
  ∃ x y, total_time_condition x y ∧ total_cost_condition x y ∧ non_negative_condition x y ∧ 
  revenue x y = 70 := 
by
  sorry

end maximize_revenue_l1722_172281


namespace moles_of_HCl_formed_l1722_172261

-- Define the reaction as given in conditions
def reaction (C2H6 Cl2 C2H4Cl2 HCl : ℝ) := C2H6 + Cl2 = C2H4Cl2 + 2 * HCl

-- Define the initial moles of reactants
def moles_C2H6 : ℝ := 2
def moles_Cl2 : ℝ := 2

-- State the expected moles of HCl produced
def expected_moles_HCl : ℝ := 4

-- The theorem stating the problem to prove
theorem moles_of_HCl_formed : ∃ HCl : ℝ, reaction moles_C2H6 moles_Cl2 0 HCl ∧ HCl = expected_moles_HCl :=
by
  -- Skipping detailed proof with sorry
  sorry

end moles_of_HCl_formed_l1722_172261


namespace numerator_greater_denominator_l1722_172282

theorem numerator_greater_denominator (x : ℝ) (h1 : -3 ≤ x) (h2 : x ≤ 3) (h3 : 5 * x + 3 > 8 - 3 * x) : (5 / 8) < x ∧ x ≤ 3 :=
by
  sorry

end numerator_greater_denominator_l1722_172282


namespace flower_beds_and_circular_path_fraction_l1722_172251

noncomputable def occupied_fraction 
  (yard_length : ℕ)
  (yard_width : ℕ)
  (side1 : ℕ)
  (side2 : ℕ)
  (triangle_leg : ℕ)
  (circle_radius : ℕ) : ℝ :=
  let flower_bed_area := 2 * (1 / 2 : ℝ) * triangle_leg^2
  let circular_path_area := Real.pi * circle_radius ^ 2
  let occupied_area := flower_bed_area + circular_path_area
  occupied_area / (yard_length * yard_width)

theorem flower_beds_and_circular_path_fraction
  (yard_length : ℕ)
  (yard_width : ℕ)
  (side1 : ℕ)
  (side2 : ℕ)
  (triangle_leg : ℕ)
  (circle_radius : ℕ)
  (h1 : side1 = 20)
  (h2 : side2 = 30)
  (h3 : triangle_leg = (side2 - side1) / 2)
  (h4 : yard_length = 30)
  (h5 : yard_width = 5)
  (h6 : circle_radius = 2) :
  occupied_fraction yard_length yard_width side1 side2 triangle_leg circle_radius = (25 + 4 * Real.pi) / 150 :=
by sorry

end flower_beds_and_circular_path_fraction_l1722_172251


namespace degenerate_ellipse_b_value_l1722_172257

theorem degenerate_ellipse_b_value :
  ∃ b : ℝ, (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + b = 0 → x = -1 ∧ y = 3) ↔ b = 12 :=
by
  sorry

end degenerate_ellipse_b_value_l1722_172257


namespace tan_inequality_solution_l1722_172249

variable (x : ℝ)
variable (k : ℤ)

theorem tan_inequality_solution (hx : Real.tan (2 * x - Real.pi / 4) ≤ 1) :
  ∃ k : ℤ,
  (k * Real.pi / 2 - Real.pi / 8 < x) ∧ (x ≤ k * Real.pi / 2 + Real.pi / 4) :=
sorry

end tan_inequality_solution_l1722_172249


namespace velocity_of_current_correct_l1722_172202

-- Definitions based on the conditions in the problem
def rowing_speed_in_still_water : ℝ := 10
def distance_to_place : ℝ := 24
def total_time_round_trip : ℝ := 5

-- Define the velocity of the current
def velocity_of_current : ℝ := 2

-- Main theorem statement
theorem velocity_of_current_correct :
  ∃ (v : ℝ), (v = 2) ∧ 
  (total_time_round_trip = (distance_to_place / (rowing_speed_in_still_water + v) + 
                            distance_to_place / (rowing_speed_in_still_water - v))) :=
by {
  sorry
}

end velocity_of_current_correct_l1722_172202


namespace chris_first_day_breath_l1722_172243

theorem chris_first_day_breath (x : ℕ) (h1 : x + 10 = 20) : x = 10 :=
by
  sorry

end chris_first_day_breath_l1722_172243


namespace equal_split_payment_l1722_172270

variable (L M N : ℝ)

theorem equal_split_payment (h1 : L < N) (h2 : L > M) : 
  (L + M + N) / 3 - L = (M + N - 2 * L) / 3 :=
by sorry

end equal_split_payment_l1722_172270


namespace hyperbola_asymptote_l1722_172246

theorem hyperbola_asymptote (x y : ℝ) : 
  (∀ x y : ℝ, (x^2 / 25 - y^2 / 16 = 1) → (y = (4 / 5) * x ∨ y = -(4 / 5) * x)) := 
by 
  sorry

end hyperbola_asymptote_l1722_172246


namespace greatest_sum_x_y_l1722_172290

theorem greatest_sum_x_y (x y : ℤ) (h : x^2 + y^2 = 36) : (x + y ≤ 9) := sorry

end greatest_sum_x_y_l1722_172290


namespace cube_edge_length_l1722_172280

theorem cube_edge_length (V : ℝ) (a : ℝ)
  (hV : V = (4 / 3) * Real.pi * (Real.sqrt 3 * a / 2) ^ 3)
  (hVolume : V = (9 * Real.pi) / 2) :
  a = Real.sqrt 3 :=
by
  sorry

end cube_edge_length_l1722_172280


namespace minimal_connections_correct_l1722_172244

-- Define a Lean structure to encapsulate the conditions
structure IslandsProblem where
  islands : ℕ
  towns : ℕ
  min_towns_per_island : ℕ
  condition_islands : islands = 13
  condition_towns : towns = 25
  condition_min_towns : min_towns_per_island = 1

-- Define a function to represent the minimal number of ferry connections
def minimalFerryConnections (p : IslandsProblem) : ℕ :=
  222

-- Define the statement to be proved
theorem minimal_connections_correct (p : IslandsProblem) : 
  p.islands = 13 → 
  p.towns = 25 → 
  p.min_towns_per_island = 1 → 
  minimalFerryConnections p = 222 :=
by
  intros
  sorry

end minimal_connections_correct_l1722_172244


namespace gcd_840_1764_l1722_172288

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l1722_172288


namespace walking_rate_ratio_l1722_172274

theorem walking_rate_ratio (R R' : ℚ) (D : ℚ) (h1: D = R * 14) (h2: D = R' * 12) : R' / R = 7 / 6 :=
by 
  sorry

end walking_rate_ratio_l1722_172274


namespace number_of_rows_of_red_notes_l1722_172214

theorem number_of_rows_of_red_notes (R : ℕ) :
  let red_notes_in_each_row := 6
  let blue_notes_per_red_note := 2
  let additional_blue_notes := 10
  let total_notes := 100
  (6 * R + 12 * R + 10 = 100) → R = 5 :=
by
  intros
  sorry

end number_of_rows_of_red_notes_l1722_172214


namespace hiking_trip_distance_l1722_172269

open Real

-- Define the given conditions
def distance_north : ℝ := 10
def distance_south : ℝ := 7
def distance_east1 : ℝ := 17
def distance_east2 : ℝ := 8

-- Define the net displacement conditions
def net_distance_north : ℝ := distance_north - distance_south
def net_distance_east : ℝ := distance_east1 + distance_east2

-- Prove the distance from the starting point
theorem hiking_trip_distance :
  sqrt ((net_distance_north)^2 + (net_distance_east)^2) = sqrt 634 := by
  sorry

end hiking_trip_distance_l1722_172269


namespace find_value_of_a_l1722_172212

theorem find_value_of_a (a : ℝ) (h : ( (-2 - (2 * a - 1)) / (3 - (-2)) = -1 )) : a = 2 :=
sorry

end find_value_of_a_l1722_172212


namespace ellipse_standard_equation_l1722_172271

theorem ellipse_standard_equation (a b c : ℝ) (h1 : 2 * a = 8) (h2 : c / a = 3 / 4) (h3 : b^2 = a^2 - c^2) :
  (x y : ℝ) →
  (x^2 / a^2 + y^2 / b^2 = 1 ∨ x^2 / b^2 + y^2 / a^2 = 1) :=
by
  sorry

end ellipse_standard_equation_l1722_172271


namespace f_max_iff_l1722_172205

noncomputable def f : ℚ → ℝ := sorry

axiom f_zero : f 0 = 0
axiom f_pos (a : ℚ) (h : a ≠ 0) : f a > 0
axiom f_mul (a b : ℚ) : f (a * b) = f a * f b
axiom f_add_le (a b : ℚ) : f (a + b) ≤ f a + f b
axiom f_bound (m : ℤ) : f m ≤ 1989

theorem f_max_iff (a b : ℚ) (h : f a ≠ f b) : f (a + b) = max (f a) (f b) := 
sorry

end f_max_iff_l1722_172205


namespace S_40_value_l1722_172224

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

axiom h1 : S 10 = 10
axiom h2 : S 30 = 70

theorem S_40_value : S 40 = 150 :=
by
  -- Conditions
  have h1 : S 10 = 10 := h1
  have h2 : S 30 = 70 := h2
  -- Start proof here
  sorry

end S_40_value_l1722_172224


namespace midpoint_locus_l1722_172254

theorem midpoint_locus (c : ℝ) (H : 0 < c ∧ c ≤ Real.sqrt 2) :
  ∃ L, L = "curvilinear quadrilateral with arcs forming transitions" :=
sorry

end midpoint_locus_l1722_172254


namespace percent_of_a_is_b_l1722_172272

variable {a b c : ℝ}

theorem percent_of_a_is_b (h1 : c = 0.25 * a) (h2 : c = 0.10 * b) : b = 2.5 * a :=
by sorry

end percent_of_a_is_b_l1722_172272


namespace solve_equation_l1722_172217

theorem solve_equation 
  (x : ℝ) 
  (h : (2 * x - 1)^2 - (1 - 3 * x)^2 = 5 * (1 - x) * (x + 1)) : 
  x = 5 / 2 := 
sorry

end solve_equation_l1722_172217


namespace rectangle_area_in_inscribed_triangle_l1722_172222

theorem rectangle_area_in_inscribed_triangle (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hxh : x < h) :
  ∃ (y : ℝ), y = (b * (h - x)) / h ∧ (x * y) = (b * x * (h - x)) / h :=
by
  sorry

end rectangle_area_in_inscribed_triangle_l1722_172222


namespace triangle_angle_and_area_l1722_172200

section Geometry

variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def triangle_sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop := 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

def vectors_parallel (a b : ℝ) (A B : ℝ) : Prop := 
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A

-- Problem statement
theorem triangle_angle_and_area (A B C a b c : ℝ) : 
  triangle_sides_opposite_angles a b c A B C ∧ vectors_parallel a b A B ∧ a = Real.sqrt 7 ∧ b = 2 ∧ A = Real.pi / 3
  → A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
sorry

end Geometry

end triangle_angle_and_area_l1722_172200


namespace total_five_digit_odd_and_multiples_of_5_l1722_172207

def count_odd_five_digit_numbers : ℕ :=
  let choices := 9 * 10 * 10 * 10 * 5
  choices

def count_multiples_of_5_five_digit_numbers : ℕ :=
  let choices := 9 * 10 * 10 * 10 * 2
  choices

theorem total_five_digit_odd_and_multiples_of_5 : count_odd_five_digit_numbers + count_multiples_of_5_five_digit_numbers = 63000 :=
by
  -- Proof Placeholder
  sorry

end total_five_digit_odd_and_multiples_of_5_l1722_172207


namespace brother_more_lambs_than_merry_l1722_172231

theorem brother_more_lambs_than_merry
  (merry_lambs : ℕ) (total_lambs : ℕ) (more_than_merry : ℕ)
  (h1 : merry_lambs = 10) 
  (h2 : total_lambs = 23)
  (h3 : more_than_merry + merry_lambs + merry_lambs = total_lambs) :
  more_than_merry = 3 :=
by
  sorry

end brother_more_lambs_than_merry_l1722_172231


namespace certain_person_current_age_l1722_172262

-- Define Sandys's current age and the certain person's current age
variable (S P : ℤ)

-- Conditions from the problem
def sandy_phone_bill_condition := 10 * S = 340
def sandy_age_relation := S + 2 = 3 * P

theorem certain_person_current_age (h1 : sandy_phone_bill_condition S) (h2 : sandy_age_relation S P) : P - 2 = 10 :=
by
  sorry

end certain_person_current_age_l1722_172262


namespace john_ratio_amounts_l1722_172276

/-- John gets $30 from his grandpa and some multiple of that amount from his grandma. 
He got $120 from the two grandparents. What is the ratio of the amount he got from 
his grandma to the amount he got from his grandpa? --/
theorem john_ratio_amounts (amount_grandpa amount_total : ℝ) (multiple : ℝ) :
  amount_grandpa = 30 → amount_total = 120 →
  amount_total = amount_grandpa + multiple * amount_grandpa →
  multiple = 3 :=
by
  intros h1 h2 h3
  sorry

end john_ratio_amounts_l1722_172276


namespace polygon_interior_equals_exterior_sum_eq_360_l1722_172239

theorem polygon_interior_equals_exterior_sum_eq_360 (n : ℕ) :
  (n - 2) * 180 = 360 → n = 6 :=
by
  intro h
  sorry

end polygon_interior_equals_exterior_sum_eq_360_l1722_172239


namespace min_houses_needed_l1722_172236

theorem min_houses_needed (n : ℕ) (x : ℕ) (h : n > 0) : (x ≤ n ∧ (x: ℚ)/n < 0.06) → n ≥ 20 :=
sorry

end min_houses_needed_l1722_172236


namespace school_student_ratio_l1722_172289

theorem school_student_ratio :
  ∀ (F S T : ℕ), (T = 200) → (S = T + 40) → (F + S + T = 920) → (F : ℚ) / (S : ℚ) = 2 / 1 :=
by
  intros F S T hT hS hSum
  sorry

end school_student_ratio_l1722_172289


namespace T_expansion_l1722_172220

def T (x : ℝ) : ℝ := (x - 2)^5 + 5 * (x - 2)^4 + 10 * (x - 2)^3 + 10 * (x - 2)^2 + 5 * (x - 2) + 1

theorem T_expansion (x : ℝ) : T x = (x - 1)^5 := by
  sorry

end T_expansion_l1722_172220


namespace probability_face_cards_l1722_172291

theorem probability_face_cards :
  let first_card_hearts_face := 3 / 52
  let second_card_clubs_face_after_hearts := 3 / 51
  let combined_probability := first_card_hearts_face * second_card_clubs_face_after_hearts
  combined_probability = 1 / 294 :=
by 
  sorry

end probability_face_cards_l1722_172291


namespace age_of_B_l1722_172233

variables (A B : ℕ)

-- Conditions
def condition1 := A + 10 = 2 * (B - 10)
def condition2 := A = B + 7

-- Theorem stating the present age of B
theorem age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 37 :=
by
  sorry

end age_of_B_l1722_172233


namespace probability_of_drawing_green_ball_l1722_172229

variable (total_balls green_balls : ℕ)
variable (total_balls_eq : total_balls = 10)
variable (green_balls_eq : green_balls = 4)

theorem probability_of_drawing_green_ball (h_total : total_balls = 10) (h_green : green_balls = 4) :
  (green_balls : ℚ) / total_balls = 2 / 5 := by
  sorry

end probability_of_drawing_green_ball_l1722_172229


namespace EF_squared_correct_l1722_172267

-- Define the problem setup and the proof goal.
theorem EF_squared_correct :
  ∀ (A B C D E F : Type)
  (side : ℝ)
  (h1 : side = 10)
  (BE DF AE CF : ℝ)
  (h2 : BE = 7)
  (h3 : DF = 7)
  (h4 : AE = 15)
  (h5 : CF = 15)
  (EF_squared : ℝ),
  EF_squared = 548 :=
by
  sorry

end EF_squared_correct_l1722_172267


namespace minimize_sum_of_squares_at_mean_l1722_172234

-- Definitions of the conditions
def P1 (x1 : ℝ) : ℝ := x1
def P2 (x2 : ℝ) : ℝ := x2
def P3 (x3 : ℝ) : ℝ := x3
def P4 (x4 : ℝ) : ℝ := x4
def P5 (x5 : ℝ) : ℝ := x5

-- Definition of the function we want to minimize
def s (P : ℝ) (x1 x2 x3 x4 x5 : ℝ) : ℝ :=
  (P - x1)^2 + (P - x2)^2 + (P - x3)^2 + (P - x4)^2 + (P - x5)^2

-- Proof statement
theorem minimize_sum_of_squares_at_mean (x1 x2 x3 x4 x5 : ℝ) :
  ∃ P : ℝ, P = (x1 + x2 + x3 + x4 + x5) / 5 ∧ 
           ∀ x : ℝ, s P x1 x2 x3 x4 x5 ≤ s x x1 x2 x3 x4 x5 := 
by
  sorry

end minimize_sum_of_squares_at_mean_l1722_172234


namespace taxi_fare_max_distance_l1722_172237

-- Setting up the conditions
def starting_price : ℝ := 7
def additional_fare_per_km : ℝ := 2.4
def max_base_distance_km : ℝ := 3
def total_fare : ℝ := 19

-- Defining the maximum distance based on the given conditions
def max_distance : ℝ := 8

-- The theorem is to prove that the maximum distance is indeed 8 kilometers
theorem taxi_fare_max_distance :
  ∀ (x : ℝ), total_fare = starting_price + additional_fare_per_km * (x - max_base_distance_km) → x ≤ max_distance :=
by
  intros x h
  sorry

end taxi_fare_max_distance_l1722_172237


namespace ratio_x_to_y_l1722_172204

theorem ratio_x_to_y (x y : ℤ) (h : (10*x - 3*y) / (13*x - 2*y) = 3 / 5) : x / y = 9 / 11 := 
by sorry

end ratio_x_to_y_l1722_172204


namespace beckys_age_ratio_l1722_172228

theorem beckys_age_ratio (Eddie_age : ℕ) (Irene_age : ℕ)
  (becky_age: ℕ)
  (H1 : Eddie_age = 92)
  (H2 : Irene_age = 46)
  (H3 : Irene_age = 2 * becky_age) :
  becky_age / Eddie_age = 1 / 4 :=
by
  sorry

end beckys_age_ratio_l1722_172228


namespace train_speed_l1722_172247

def length_of_train : ℝ := 160
def time_to_cross : ℝ := 18
def speed_in_kmh : ℝ := 32

theorem train_speed :
  (length_of_train / time_to_cross) * 3.6 = speed_in_kmh :=
by
  sorry

end train_speed_l1722_172247


namespace find_x_l1722_172206

theorem find_x (c d : ℝ) (y z x : ℝ) 
  (h1 : y^2 = c * z^2) 
  (h2 : y = d / x)
  (h3 : y = 3) 
  (h4 : x = 4) 
  (h5 : z = 6) 
  (h6 : y = 2) 
  (h7 : z = 12) 
  : x = 6 := 
by
  sorry

end find_x_l1722_172206


namespace maximum_value_of_expression_l1722_172242

theorem maximum_value_of_expression
  (a b c : ℝ)
  (h1 : 0 ≤ a)
  (h2 : 0 ≤ b)
  (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + 2 * c^2 = 1) :
  ab * Real.sqrt 3 + 3 * bc ≤ Real.sqrt 7 :=
sorry

end maximum_value_of_expression_l1722_172242


namespace proof_min_max_expected_wasted_minutes_l1722_172292

/-- The conditions given:
    - There are 8 people in the queue.
    - 5 people perform simple operations that take 1 minute each.
    - 3 people perform lengthy operations that take 5 minutes each.
--/
structure QueueStatus where
  total_people : Nat := 8
  simple_operations_people : Nat := 5
  lengthy_operations_people : Nat := 3
  simple_operation_time : Nat := 1
  lengthy_operation_time : Nat := 5

/-- Propositions to be proven:
    - Minimum possible total number of wasted person-minutes is 40.
    - Maximum possible total number of wasted person-minutes is 100.
    - Expected total number of wasted person-minutes in random order is 72.5.
--/
def min_wasted_person_minutes (qs: QueueStatus) : Nat := 40
def max_wasted_person_minutes (qs: QueueStatus) : Nat := 100
def expected_wasted_person_minutes (qs: QueueStatus) : Real := 72.5

theorem proof_min_max_expected_wasted_minutes (qs: QueueStatus) :
  min_wasted_person_minutes qs = 40 ∧ 
  max_wasted_person_minutes qs = 100 ∧ 
  expected_wasted_person_minutes qs = 72.5 := by
  sorry

end proof_min_max_expected_wasted_minutes_l1722_172292


namespace positive_integer_M_l1722_172215

theorem positive_integer_M (M : ℕ) (h : 14^2 * 35^2 = 70^2 * M^2) : M = 7 :=
sorry

end positive_integer_M_l1722_172215


namespace fourth_circle_radius_l1722_172273

theorem fourth_circle_radius (c : ℝ) (h : c > 0) :
  let r := c / 5
  let fourth_radius := (3 * c) / 10
  fourth_radius = (c / 2) - r :=
by
  let r := c / 5
  let fourth_radius := (3 * c) / 10
  sorry

end fourth_circle_radius_l1722_172273


namespace number_of_real_pairs_l1722_172211

theorem number_of_real_pairs :
  ∃! (x y : ℝ), 11 * x^2 + 2 * x * y + 9 * y^2 + 8 * x - 12 * y + 6 = 0 :=
sorry

end number_of_real_pairs_l1722_172211


namespace smallest_n_condition_l1722_172221

theorem smallest_n_condition (n : ℕ) : 25 * n - 3 ≡ 0 [MOD 16] → n ≡ 11 [MOD 16] :=
by
  sorry

end smallest_n_condition_l1722_172221


namespace photographs_taken_l1722_172223

theorem photographs_taken (P : ℝ) (h : P + 0.80 * P = 180) : P = 100 :=
by sorry

end photographs_taken_l1722_172223


namespace max_value_of_f_l1722_172299

noncomputable def f (ω a x : ℝ) : ℝ := Real.sin (ω * x) + a * Real.cos (ω * x)

theorem max_value_of_f 
  (ω a : ℝ) 
  (h1 : 0 < ω) 
  (h2 : (2 * Real.pi / ω) = Real.pi) 
  (h3 : ∃ k : ℤ, ω * (Real.pi / 12) + (k : ℝ) * Real.pi + Real.pi / 3 = Real.pi / 2 + (k : ℝ) * Real.pi) :
  ∃ x : ℝ, f ω a x = 2 := by
  sorry

end max_value_of_f_l1722_172299
