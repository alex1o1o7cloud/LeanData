import Mathlib

namespace total_cost_of_fencing_l14_1434

def P : ℤ := 42 + 35 + 52 + 66 + 40
def cost_per_meter : ℤ := 3
def total_cost : ℤ := P * cost_per_meter

theorem total_cost_of_fencing : total_cost = 705 := by
  sorry

end total_cost_of_fencing_l14_1434


namespace tan_sum_angle_l14_1409

theorem tan_sum_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (π / 4 + α) = -3 := 
by sorry

end tan_sum_angle_l14_1409


namespace fill_blank_1_fill_blank_2_l14_1447

theorem fill_blank_1 (x : ℤ) (h : 1 + x = -10) : x = -11 := sorry

theorem fill_blank_2 (y : ℝ) (h : y - 4.5 = -4.5) : y = 0 := sorry

end fill_blank_1_fill_blank_2_l14_1447


namespace proper_subset_singleton_l14_1415

theorem proper_subset_singleton : ∀ (P : Set ℕ), P = {0} → (∃ S, S ⊂ P ∧ S = ∅) :=
by
  sorry

end proper_subset_singleton_l14_1415


namespace intersection_M_N_eq_set_l14_1446

universe u

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {y | ∃ x, x ∈ M ∧ y = 2 * x + 1}

-- Prove the intersection M ∩ N = {-1, 1}
theorem intersection_M_N_eq_set : M ∩ N = {-1, 1} :=
by
  simp [Set.ext_iff, M, N]
  sorry

end intersection_M_N_eq_set_l14_1446


namespace harry_worked_32_hours_l14_1464

variable (x y : ℝ)
variable (harry_pay james_pay : ℝ)

-- Definitions based on conditions
def harry_weekly_pay (h : ℝ) := 30*x + (h - 30)*y
def james_weekly_pay := 40*x + 1*y

-- Condition: Harry and James were paid the same last week
axiom harry_james_same_pay : ∀ (h : ℝ), harry_weekly_pay x y h = james_weekly_pay x y

-- Prove: Harry worked 32 hours
theorem harry_worked_32_hours : ∃ h : ℝ, h = 32 ∧ harry_weekly_pay x y h = james_weekly_pay x y := by
  sorry

end harry_worked_32_hours_l14_1464


namespace part1_proof_part2_proof_part3_proof_l14_1476

-- Definitions and conditions for part 1
def P (a : ℤ) : ℤ × ℤ := (-3 * a - 4, 2 + a)
def part1_condition (a : ℤ) : Prop := (2 + a = 0)
def part1_answer : ℤ × ℤ := (2, 0)

-- Definitions and conditions for part 2
def Q : ℤ × ℤ := (5, 8)
def part2_condition (a : ℤ) : Prop := (-3 * a - 4 = 5)
def part2_answer : ℤ × ℤ := (5, -1)

-- Definitions and conditions for part 3
def part3_condition (a : ℤ) : Prop := 
  (-3 * a - 4 + 2 + a = 0) ∧ (-3 * a - 4 < 0 ∧ 2 + a > 0) -- Second quadrant
def part3_answer (a : ℤ) : ℤ := (a ^ 2023 + 2023)

-- Lean statements for proofs

theorem part1_proof (a : ℤ) (h : part1_condition a) : P a = part1_answer :=
by sorry

theorem part2_proof (a : ℤ) (h : part2_condition a) : P a = part2_answer :=
by sorry

theorem part3_proof (a : ℤ) (h : part3_condition a) : part3_answer a = 2022 :=
by sorry

end part1_proof_part2_proof_part3_proof_l14_1476


namespace find_M_pos_int_l14_1437

theorem find_M_pos_int (M : ℕ) (hM : 33^2 * 66^2 = 15^2 * M^2) :
    M = 726 :=
by
  -- Sorry, skipping the proof.
  sorry

end find_M_pos_int_l14_1437


namespace tangent_parallel_l14_1436

noncomputable def f (x: ℝ) : ℝ := x^4 - x
noncomputable def f' (x: ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_parallel
  (P : ℝ × ℝ)
  (hp : P = (1, 0))
  (tangent_parallel : ∀ x, f' x = 3 ↔ x = 1)
  : P = (1, 0) := 
by 
  sorry

end tangent_parallel_l14_1436


namespace purchase_price_l14_1484

theorem purchase_price (marked_price : ℝ) (discount_rate profit_rate x : ℝ)
  (h1 : marked_price = 126)
  (h2 : discount_rate = 0.05)
  (h3 : profit_rate = 0.05)
  (h4 : marked_price * (1 - discount_rate) - x = x * profit_rate) : 
  x = 114 :=
by 
  sorry

end purchase_price_l14_1484


namespace q_0_plus_q_5_l14_1416

-- Define the properties of the polynomial q(x)
variable (q : ℝ → ℝ)
variable (monic_q : ∀ x, ∃ a b c d e f, a = 1 ∧ q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f)
variable (deg_q : ∀ x, degree q = 5)
variable (q_1 : q 1 = 26)
variable (q_2 : q 2 = 52)
variable (q_3 : q 3 = 78)

-- State the theorem to find q(0) + q(5)
theorem q_0_plus_q_5 : q 0 + q 5 = 58 :=
sorry

end q_0_plus_q_5_l14_1416


namespace min_value_x_y_l14_1408

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : x + y ≥ 16 :=
sorry

end min_value_x_y_l14_1408


namespace geometric_sequence_a5_eq_2_l14_1403

-- Define geometric sequence and the properties
noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

-- Given conditions
variables {a : ℕ → ℝ} {q : ℝ}

-- Roots of given quadratic equation
variables (h1 : a 3 = 1 ∨ a 3 = 4 / 1) (h2 : a 7 = 4 / a 3)
variables (h3 : q > 0) (h4 : geometric_seq a q)

-- Prove that a5 = 2
theorem geometric_sequence_a5_eq_2 : a 5 = 2 :=
sorry

end geometric_sequence_a5_eq_2_l14_1403


namespace barbara_removed_total_sheets_l14_1450

theorem barbara_removed_total_sheets :
  let bundles_colored := 3
  let bunches_white := 2
  let heaps_scrap := 5
  let sheets_per_bunch := 4
  let sheets_per_bundle := 2
  let sheets_per_heap := 20
  bundles_colored * sheets_per_bundle + bunches_white * sheets_per_bunch + heaps_scrap * sheets_per_heap = 114 :=
by
  sorry

end barbara_removed_total_sheets_l14_1450


namespace cone_volume_not_product_base_height_l14_1421

noncomputable def cone_volume (S h : ℝ) := (1/3) * S * h

theorem cone_volume_not_product_base_height (S h : ℝ) :
  cone_volume S h ≠ S * h :=
by sorry

end cone_volume_not_product_base_height_l14_1421


namespace intersection_A_B_l14_1462

/-- Definitions for the sets A and B --/
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4, 5}

-- Theorem statement regarding the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {1} :=
by sorry

end intersection_A_B_l14_1462


namespace water_charging_standard_l14_1418

theorem water_charging_standard
  (x y : ℝ)
  (h1 : 10 * x + 5 * y = 35)
  (h2 : 10 * x + 8 * y = 44) : 
  x = 2 ∧ y = 3 :=
by
  sorry

end water_charging_standard_l14_1418


namespace cos_2_alpha_plus_beta_eq_l14_1414

variable (α β : ℝ)

def tan_roots_of_quadratic (x : ℝ) : Prop := x^2 + 5 * x - 6 = 0

theorem cos_2_alpha_plus_beta_eq :
  ∀ α β : ℝ, tan_roots_of_quadratic (Real.tan α) ∧ tan_roots_of_quadratic (Real.tan β) →
  Real.cos (2 * (α + β)) = 12 / 37 :=
by
  intros
  sorry

end cos_2_alpha_plus_beta_eq_l14_1414


namespace first_number_remainder_one_l14_1486

theorem first_number_remainder_one (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 2023) :
  (∀ (a b c : ℕ), a < b ∧ b < c ∧ b = a + 1 ∧ c = a + 2 → (a % 3 ≠ b % 3 ∧ a % 3 ≠ c % 3 ∧ b % 3 ≠ c % 3))
  → (n % 3 = 1) :=
sorry

end first_number_remainder_one_l14_1486


namespace probability_miss_at_least_once_l14_1461
-- Importing the entirety of Mathlib

-- Defining the conditions and question
variable (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1)

-- The main statement for the proof problem
theorem probability_miss_at_least_once (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1) : P ≤ 1 → 0 ≤ P ∧ 1 - P^3 ≥ 0 := 
by
  sorry

end probability_miss_at_least_once_l14_1461


namespace experimental_fertilizer_height_is_correct_l14_1488

/-- Define the static heights and percentages for each plant's growth conditions. -/
def control_plant_height : ℝ := 36
def bone_meal_multiplier : ℝ := 1.25
def cow_manure_multiplier : ℝ := 2
def experimental_fertilizer_multiplier : ℝ := 1.5

/-- Define each plant's height based on the given multipliers and conditions. -/
def bone_meal_plant_height : ℝ := bone_meal_multiplier * control_plant_height
def cow_manure_plant_height : ℝ := cow_manure_multiplier * bone_meal_plant_height
def experimental_fertilizer_plant_height : ℝ := experimental_fertilizer_multiplier * cow_manure_plant_height

/-- Proof that the height of the experimental fertilizer plant is 135 inches. -/
theorem experimental_fertilizer_height_is_correct :
  experimental_fertilizer_plant_height = 135 := by
    sorry

end experimental_fertilizer_height_is_correct_l14_1488


namespace opposite_of_five_l14_1478

theorem opposite_of_five : ∃ y : ℤ, 5 + y = 0 ∧ y = -5 := by
  use -5
  constructor
  . exact rfl
  . sorry

end opposite_of_five_l14_1478


namespace triangle_perimeter_l14_1441

theorem triangle_perimeter (MN NP MP : ℝ)
  (h1 : MN - NP = 18)
  (h2 : MP = 40)
  (h3 : MN / NP = 28 / 12) : 
  MN + NP + MP = 85 :=
by
  -- Proof is omitted
  sorry

end triangle_perimeter_l14_1441


namespace find_s_l14_1483

theorem find_s (s : ℝ) (m : ℤ) (d : ℝ) (h_floor : ⌊s⌋ = m) (h_decompose : s = m + d) (h_fractional : 0 ≤ d ∧ d < 1) (h_equation : ⌊s⌋ - s = -10.3) : s = -9.7 :=
by
  sorry

end find_s_l14_1483


namespace det_example_l14_1426

theorem det_example : (1 * 4 - 2 * 3) = -2 :=
by
  -- Skip the proof with sorry
  sorry

end det_example_l14_1426


namespace cyclic_identity_l14_1453

theorem cyclic_identity (a b c : ℝ) :
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) =
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) ∧
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) =
  c * (c - b)^2 + a * (a - b)^2 - (c - b) * (a - b) * (c + a - b) := by
sorry

end cyclic_identity_l14_1453


namespace math_problem_l14_1442

noncomputable def compute_value (a b c : ℝ) : ℝ :=
  (b / (a + b)) + (c / (b + c)) + (a / (c + a))

theorem math_problem (a b c : ℝ)
  (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -12)
  (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 15) :
  compute_value a b c = 6 :=
sorry

end math_problem_l14_1442


namespace tangent_of_inclination_of_OP_l14_1494

noncomputable def point_P_x (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def point_P_y (φ : ℝ) : ℝ := 2 * Real.sin φ

theorem tangent_of_inclination_of_OP (φ : ℝ) (h: φ = Real.pi / 6) :
  (point_P_y φ / point_P_x φ) = 2 * Real.sqrt 3 / 9 :=
by
  have h1 : point_P_x φ = 3 * (Real.sqrt 3 / 2) := by sorry
  have h2 : point_P_y φ = 1 := by sorry
  sorry

end tangent_of_inclination_of_OP_l14_1494


namespace remainder_2015_div_28_l14_1406

theorem remainder_2015_div_28 : 2015 % 28 = 17 :=
by
  sorry

end remainder_2015_div_28_l14_1406


namespace students_taking_both_languages_l14_1417

theorem students_taking_both_languages (total_students students_neither students_french students_german : ℕ) (h1 : total_students = 69)
  (h2 : students_neither = 15) (h3 : students_french = 41) (h4 : students_german = 22) :
  (students_french + students_german - (total_students - students_neither) = 9) :=
by
  sorry

end students_taking_both_languages_l14_1417


namespace dress_total_price_correct_l14_1433

-- Define constants and variables
def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

-- Function to calculate sale price after discount
def sale_price (op : ℝ) (dr : ℝ) : ℝ := op - (op * dr)

-- Function to calculate total price including tax
def total_selling_price (sp : ℝ) (tr : ℝ) : ℝ := sp + (sp * tr)

-- The proof statement to be proven
theorem dress_total_price_correct :
  total_selling_price (sale_price original_price discount_rate) tax_rate = 96.6 :=
  by sorry

end dress_total_price_correct_l14_1433


namespace algebra_problem_l14_1457

variable (a : ℝ)

-- Condition: Given (a + 1/a)^3 = 4
def condition : Prop := (a + 1/a)^3 = 4

-- Statement: Prove a^4 + 1/a^4 = -158/81
theorem algebra_problem (h : condition a) : a^4 + 1/a^4 = -158/81 := 
sorry

end algebra_problem_l14_1457


namespace consecutive_integer_sets_l14_1489

theorem consecutive_integer_sets (S : ℕ) (hS : S = 180) : 
  ∃ n_values : Finset ℕ, 
  (∀ n ∈ n_values, (∃ a : ℕ, n * (2 * a + n - 1) = 2 * S) ∧ n >= 2) ∧ 
  n_values.card = 4 :=
by
  sorry

end consecutive_integer_sets_l14_1489


namespace range_of_m_l14_1439

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x - 2 < 0) → -8 < m ∧ m ≤ 0 :=
sorry

end range_of_m_l14_1439


namespace initial_volume_of_solution_l14_1420

variable (V : ℝ)

theorem initial_volume_of_solution :
  (0.05 * V + 5.5 = 0.15 * (V + 10)) → (V = 40) :=
by
  intro h
  sorry

end initial_volume_of_solution_l14_1420


namespace find_y_l14_1458

theorem find_y (y : ℚ) (h : ⌊y⌋ + y = 5) : y = 7 / 3 :=
sorry

end find_y_l14_1458


namespace coin_flips_probability_l14_1480

section 

-- Definition for the probability of heads in a single flip
def prob_heads : ℚ := 1 / 2

-- Definition for flipping the coin 5 times and getting heads on the first 4 flips and tails on the last flip
def prob_specific_sequence (n : ℕ) (k : ℕ) : ℚ := (prob_heads) ^ k * (prob_heads) ^ (n - k)

-- The main theorem which states the probability of the desired outcome
theorem coin_flips_probability : 
  prob_specific_sequence 5 4 = 1 / 32 :=
sorry

end

end coin_flips_probability_l14_1480


namespace count_colorings_l14_1467

-- Define the number of disks
def num_disks : ℕ := 6

-- Define colorings with constraints: 2 black, 2 white, 2 blue considering rotations and reflections as equivalent
def valid_colorings : ℕ :=
  18  -- This is the result obtained using Burnside's Lemma as shown in the solution

theorem count_colorings : valid_colorings = 18 := by
  sorry

end count_colorings_l14_1467


namespace angle_ABC_bisector_l14_1469

theorem angle_ABC_bisector (θ : ℝ) (h : θ / 2 = (1 / 3) * (180 - θ)) : θ = 72 :=
by
  sorry

end angle_ABC_bisector_l14_1469


namespace circle_condition_l14_1412

theorem circle_condition (f : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 4*x + 6*y + f = 0) ↔ f < 13 :=
by
  sorry

end circle_condition_l14_1412


namespace samantha_exam_score_l14_1459

theorem samantha_exam_score :
  ∀ (q1 q2 q3 : ℕ) (s1 s2 s3 : ℚ),
  q1 = 30 → q2 = 50 → q3 = 20 →
  s1 = 0.75 → s2 = 0.8 → s3 = 0.65 →
  (22.5 + 40 + 2 * (0.65 * 20)) / (30 + 50 + 2 * 20) = 0.7375 :=
by
  intros q1 q2 q3 s1 s2 s3 hq1 hq2 hq3 hs1 hs2 hs3
  sorry

end samantha_exam_score_l14_1459


namespace base_any_number_l14_1419

theorem base_any_number (base : ℝ) (x y : ℝ) (h1 : 3^x * base^y = 19683) (h2 : x - y = 9) (h3 : x = 9) : true :=
by
  sorry

end base_any_number_l14_1419


namespace total_amount_received_l14_1499

theorem total_amount_received (CI : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ) (A : ℝ) 
  (hCI : CI = P * ((1 + r / n) ^ (n * t) - 1))
  (hCI_value : CI = 370.80)
  (hr : r = 0.06)
  (hn : n = 1)
  (ht : t = 2)
  (hP : P = 3000)
  (hP_value : P = CI / 0.1236) :
  A = P + CI := 
by 
sorry

end total_amount_received_l14_1499


namespace relationship_between_y_l14_1440

theorem relationship_between_y
  (m y₁ y₂ y₃ : ℝ)
  (hA : y₁ = -(-1)^2 + 2 * -1 + m)
  (hB : y₂ = -(1)^2 + 2 * 1 + m)
  (hC : y₃ = -(2)^2 + 2 * 2 + m) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end relationship_between_y_l14_1440


namespace complement_of_A_in_U_l14_1465

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set A
def A : Set ℤ := {x | x ∈ Set.univ ∧ x^2 + x - 2 < 0}

-- State the theorem about the complement of A in U
theorem complement_of_A_in_U :
  (U \ A) = {-2, 1, 2} :=
sorry

end complement_of_A_in_U_l14_1465


namespace solve_for_n_l14_1444

theorem solve_for_n (n : ℤ) (h : (5/4 : ℚ) * n + (5/4 : ℚ) = n) : n = -5 := by
    sorry

end solve_for_n_l14_1444


namespace polynomial_roots_l14_1454

theorem polynomial_roots :
  ∀ (x : ℝ), (x^3 - x^2 - 6 * x + 8 = 0) ↔ (x = 2 ∨ x = (-1 + Real.sqrt 17) / 2 ∨ x = (-1 - Real.sqrt 17) / 2) :=
by
  sorry

end polynomial_roots_l14_1454


namespace compacted_space_of_all_cans_l14_1479

def compacted_space_per_can (original_space: ℕ) (compaction_rate: ℕ) : ℕ :=
  original_space * compaction_rate / 100

def total_compacted_space (num_cans: ℕ) (compacted_space: ℕ) : ℕ :=
  num_cans * compacted_space

theorem compacted_space_of_all_cans :
  ∀ (num_cans original_space compaction_rate : ℕ),
  num_cans = 100 →
  original_space = 30 →
  compaction_rate = 35 →
  total_compacted_space num_cans (compacted_space_per_can original_space compaction_rate) = 1050 :=
by
  intros num_cans original_space compaction_rate h1 h2 h3
  rw [h1, h2, h3]
  dsimp [compacted_space_per_can, total_compacted_space]
  norm_num
  sorry

end compacted_space_of_all_cans_l14_1479


namespace no_such_class_exists_l14_1471

theorem no_such_class_exists : ¬ ∃ (b g : ℕ), (3 * b = 5 * g) ∧ (32 < b + g) ∧ (b + g < 40) :=
by {
  -- Proof goes here
  sorry
}

end no_such_class_exists_l14_1471


namespace percentage_increase_l14_1460

theorem percentage_increase (original final : ℝ) (h1 : original = 90) (h2 : final = 135) : ((final - original) / original) * 100 = 50 := 
by
  sorry

end percentage_increase_l14_1460


namespace B_pow_five_l14_1487

def B : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![2, 3], ![4, 6]]
  
theorem B_pow_five : 
  B^5 = (4096 : ℝ) • B + (0 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end B_pow_five_l14_1487


namespace flight_duration_l14_1491

theorem flight_duration (takeoff landing : ℕ) (h : ℕ) (m : ℕ)
  (h0 : takeoff = 11 * 60 + 7)
  (h1 : landing = 2 * 60 + 49 + 12 * 60)
  (h2 : 0 < m) (h3 : m < 60) :
  h + m = 45 := 
sorry

end flight_duration_l14_1491


namespace distance_to_place_l14_1490

theorem distance_to_place 
  (row_speed_still_water : ℝ) 
  (current_speed : ℝ) 
  (headwind_speed : ℝ) 
  (tailwind_speed : ℝ) 
  (total_trip_time : ℝ) 
  (htotal_trip_time : total_trip_time = 15) 
  (hrow_speed_still_water : row_speed_still_water = 10) 
  (hcurrent_speed : current_speed = 2) 
  (hheadwind_speed : headwind_speed = 4) 
  (htailwind_speed : tailwind_speed = 4) :
  ∃ (D : ℝ), D = 48 :=
by
  sorry

end distance_to_place_l14_1490


namespace speed_ratio_l14_1485

-- Definitions of the conditions in the problem
variables (v_A v_B : ℝ) -- speeds of A and B

-- Condition 1: positions after 3 minutes are equidistant from O
def equidistant_3min : Prop := 3 * v_A = |(-300 + 3 * v_B)|

-- Condition 2: positions after 12 minutes are equidistant from O
def equidistant_12min : Prop := 12 * v_A = |(-300 + 12 * v_B)|

-- Statement to prove
theorem speed_ratio (h1 : equidistant_3min v_A v_B) (h2 : equidistant_12min v_A v_B) :
  v_A / v_B = 4 / 5 := sorry

end speed_ratio_l14_1485


namespace complex_number_solution_l14_1482

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i * z = 1) : z = -i :=
by sorry

end complex_number_solution_l14_1482


namespace percentage_of_children_who_speak_only_english_l14_1445

theorem percentage_of_children_who_speak_only_english :
  (∃ (total_children both_languages hindi_speaking only_english : ℝ),
    total_children = 60 ∧
    both_languages = 0.20 * total_children ∧
    hindi_speaking = 42 ∧
    only_english = total_children - (hindi_speaking - both_languages + both_languages) ∧
    (only_english / total_children) * 100 = 30) :=
  sorry

end percentage_of_children_who_speak_only_english_l14_1445


namespace weighted_avg_sales_increase_l14_1456

section SalesIncrease

/-- Define the weightages for each category last year. -/
def w_e : ℝ := 0.4
def w_c : ℝ := 0.3
def w_g : ℝ := 0.3

/-- Define the percent increases for each category this year. -/
def p_e : ℝ := 0.15
def p_c : ℝ := 0.25
def p_g : ℝ := 0.35

/-- Prove that the weighted average percent increase in sales this year is 0.24 or 24%. -/
theorem weighted_avg_sales_increase :
  ((w_e * p_e) + (w_c * p_c) + (w_g * p_g)) / (w_e + w_c + w_g) = 0.24 := 
by
  sorry

end SalesIncrease

end weighted_avg_sales_increase_l14_1456


namespace density_is_not_vector_l14_1449

/-- Conditions definition -/
def is_vector (quantity : String) : Prop :=
quantity = "Buoyancy" ∨ quantity = "Wind speed" ∨ quantity = "Displacement"

/-- Problem statement -/
theorem density_is_not_vector : ¬ is_vector "Density" := 
by 
sorry

end density_is_not_vector_l14_1449


namespace cost_of_fencing_theorem_l14_1455

noncomputable def cost_of_fencing (area : ℝ) (ratio_length_width : ℝ) (cost_per_meter_paise : ℝ) : ℝ :=
  let width := (area / (ratio_length_width * 2 * ratio_length_width * 3)).sqrt
  let length := ratio_length_width * 3 * width
  let perimeter := 2 * (length + width)
  let cost_per_meter_rupees := cost_per_meter_paise / 100
  perimeter * cost_per_meter_rupees

theorem cost_of_fencing_theorem :
  cost_of_fencing 3750 3 50 = 125 :=
by
  sorry

end cost_of_fencing_theorem_l14_1455


namespace find_B_l14_1475

def A (a : ℝ) : Set ℝ := {3, Real.log a / Real.log 2}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem find_B (a b : ℝ) (hA : A a = {3, 2}) (hB : B a b = {a, b}) (h : (A a) ∩ (B a b) = {2}) :
  B a b = {2, 4} :=
sorry

end find_B_l14_1475


namespace negation_proposition_false_l14_1425

variable {R : Type} [LinearOrderedField R]

theorem negation_proposition_false (x y : R) :
  ¬ (x > 2 ∧ y > 3 → x + y > 5) = false := by
sorry

end negation_proposition_false_l14_1425


namespace particle_position_at_2004_seconds_l14_1407

structure ParticleState where
  position : ℕ × ℕ

def initialState : ParticleState :=
  { position := (0, 0) }

def moveParticle (state : ParticleState) (time : ℕ) : ParticleState :=
  if time = 0 then initialState
  else if (time - 1) % 4 < 2 then
    { state with position := (state.position.fst + 1, state.position.snd) }
  else
    { state with position := (state.position.fst, state.position.snd + 1) }

def particlePositionAfterTime (time : ℕ) : ParticleState :=
  (List.range time).foldl moveParticle initialState

/-- The position of the particle after 2004 seconds is (20, 44) -/
theorem particle_position_at_2004_seconds :
  (particlePositionAfterTime 2004).position = (20, 44) :=
  sorry

end particle_position_at_2004_seconds_l14_1407


namespace length_more_than_breadth_l14_1432

theorem length_more_than_breadth (b x : ℝ) (h1 : b + x = 61) (h2 : 26.50 * (4 * b + 2 * x) = 5300) : x = 22 :=
by
  sorry

end length_more_than_breadth_l14_1432


namespace a_seq_gt_one_l14_1428

noncomputable def a_seq (a : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 1 + a
  else (1 / a_seq a (n - 1)) + a

theorem a_seq_gt_one (a : ℝ) (h : 0 < a ∧ a < 1) : ∀ n : ℕ, 1 < a_seq a n :=
by {
  sorry
}

end a_seq_gt_one_l14_1428


namespace probability_and_relationship_l14_1495

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end probability_and_relationship_l14_1495


namespace evaluate_expression_l14_1473

theorem evaluate_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2 * x + 2) / x) * ((y^2 + 2 * y + 2) / y) + ((x^2 - 3 * x + 2) / y) * ((y^2 - 3 * y + 2) / x) 
  = 2 * x * y - (x / y) - (y / x) + 13 + 10 / x + 4 / y + 8 / (x * y) :=
by
  sorry

end evaluate_expression_l14_1473


namespace math_problem_l14_1497

theorem math_problem 
  (m n : ℕ) 
  (h1 : (m^2 - n) ∣ (m + n^2))
  (h2 : (n^2 - m) ∣ (m^2 + n)) : 
  (m, n) = (2, 2) ∨ (m, n) = (3, 3) ∨ (m, n) = (1, 2) ∨ (m, n) = (2, 1) ∨ (m, n) = (2, 3) ∨ (m, n) = (3, 2) := 
sorry

end math_problem_l14_1497


namespace intersection_point_l14_1411

noncomputable def line1 (x : ℚ) : ℚ := 3 * x
noncomputable def line2 (x : ℚ) : ℚ := -9 * x - 6

theorem intersection_point : ∃ (x y : ℚ), line1 x = y ∧ line2 x = y ∧ x = -1/2 ∧ y = -3/2 :=
by
  -- skipping the actual proof steps
  sorry

end intersection_point_l14_1411


namespace find_N_l14_1452

variable (N : ℚ)
variable (p : ℚ)

def ball_probability_same_color 
  (green1 : ℚ) (total1 : ℚ) 
  (green2 : ℚ) (blue2 : ℚ) 
  (p : ℚ) : Prop :=
  (green1/total1) * (green2 / (green2 + blue2)) + 
  ((total1 - green1) / total1) * (blue2 / (green2 + blue2)) = p

theorem find_N :
  p = 0.65 → 
  ball_probability_same_color 5 12 20 N p → 
  N = 280 / 311 := 
by
  sorry

end find_N_l14_1452


namespace pens_count_l14_1468

theorem pens_count (N P : ℕ) (h1 : N = 40) (h2 : P / N = 5 / 4) : P = 50 :=
by
  sorry

end pens_count_l14_1468


namespace sum_of_x_and_y_l14_1493

theorem sum_of_x_and_y (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hprod : x * y = 555) : x + y = 52 :=
by
  sorry

end sum_of_x_and_y_l14_1493


namespace total_candies_needed_l14_1405

def candies_per_box : ℕ := 156
def number_of_children : ℕ := 20

theorem total_candies_needed : candies_per_box * number_of_children = 3120 := by
  sorry

end total_candies_needed_l14_1405


namespace radius_range_l14_1463

-- Conditions:
-- r1 is the radius of circle O1
-- r2 is the radius of circle O2
-- d is the distance between centers of circles O1 and O2
-- PO1 is the distance from a point P on circle O2 to the center of circle O1

variables (r1 r2 d PO1 : ℝ)

-- Given r1 = 1, d = 5, PO1 = 2
axiom r1_def : r1 = 1
axiom d_def : d = 5
axiom PO1_def : PO1 = 2

-- To prove: 3 ≤ r2 ≤ 7
theorem radius_range (r2 : ℝ) (h : d = 5 ∧ r1 = 1 ∧ PO1 = 2 ∧ (∃ P : ℝ, P = r2)) : 3 ≤ r2 ∧ r2 ≤ 7 :=
by {
  sorry
}

end radius_range_l14_1463


namespace number_of_zeros_of_quadratic_function_l14_1477

-- Given the quadratic function y = x^2 + x - 1
def quadratic_function (x : ℝ) : ℝ := x^2 + x - 1

-- Prove that the number of zeros of the quadratic function y = x^2 + x - 1 is 2
theorem number_of_zeros_of_quadratic_function : 
  ∃ x1 x2 : ℝ, quadratic_function x1 = 0 ∧ quadratic_function x2 = 0 ∧ x1 ≠ x2 :=
by
  sorry

end number_of_zeros_of_quadratic_function_l14_1477


namespace vertical_asymptotes_l14_1402

noncomputable def f (x : ℝ) := (x^3 + 3*x^2 + 2*x + 12) / (x^2 - 5*x + 6)

theorem vertical_asymptotes (x : ℝ) : 
  (x^2 - 5*x + 6 = 0) ∧ (x^3 + 3*x^2 + 2*x + 12 ≠ 0) ↔ (x = 2 ∨ x = 3) :=
by
  sorry

end vertical_asymptotes_l14_1402


namespace lemonade_second_intermission_l14_1438

theorem lemonade_second_intermission (first_intermission third_intermission total_lemonade second_intermission : ℝ) 
  (h1 : first_intermission = 0.25) 
  (h2 : third_intermission = 0.25) 
  (h3 : total_lemonade = 0.92) 
  (h4 : second_intermission = total_lemonade - (first_intermission + third_intermission)) : 
  second_intermission = 0.42 := 
by 
  sorry

end lemonade_second_intermission_l14_1438


namespace money_spent_on_ferris_wheel_l14_1435

-- Conditions
def initial_tickets : ℕ := 6
def remaining_tickets : ℕ := 3
def ticket_cost : ℕ := 9

-- Prove that the money spent during the ferris wheel ride is 27 dollars
theorem money_spent_on_ferris_wheel : (initial_tickets - remaining_tickets) * ticket_cost = 27 := by
  sorry

end money_spent_on_ferris_wheel_l14_1435


namespace day_of_20th_is_Thursday_l14_1431

noncomputable def day_of_week (d : ℕ) : String :=
  match d % 7 with
  | 0 => "Saturday"
  | 1 => "Sunday"
  | 2 => "Monday"
  | 3 => "Tuesday"
  | 4 => "Wednesday"
  | 5 => "Thursday"
  | 6 => "Friday"
  | _ => "Unknown"

theorem day_of_20th_is_Thursday (s1 s2 s3: ℕ) (h1: 2 ≤ s1) (h2: s1 ≤ 30) (h3: s2 = s1 + 14) (h4: s3 = s2 + 14) (h5: s3 ≤ 30) (h6: day_of_week s1 = "Sunday") : 
  day_of_week 20 = "Thursday" :=
by
  sorry

end day_of_20th_is_Thursday_l14_1431


namespace sam_travel_time_l14_1400

theorem sam_travel_time (d_AC d_CB : ℕ) (v_sam : ℕ) 
  (h1 : d_AC = 600) (h2 : d_CB = 400) (h3 : v_sam = 50) : 
  (d_AC + d_CB) / v_sam = 20 := 
by
  sorry

end sam_travel_time_l14_1400


namespace hockey_league_games_l14_1474

theorem hockey_league_games (n : ℕ) (k : ℕ) (h_n : n = 25) (h_k : k = 15) : 
  (n * (n - 1) / 2) * k = 4500 := by
  sorry

end hockey_league_games_l14_1474


namespace initial_average_weight_l14_1498

theorem initial_average_weight (a b c d e : ℝ) (A : ℝ) 
    (h1 : (a + b + c) / 3 = A) 
    (h2 : (a + b + c + d) / 4 = 80) 
    (h3 : e = d + 3) 
    (h4 : (b + c + d + e) / 4 = 79) 
    (h5 : a = 75) : A = 84 :=
sorry

end initial_average_weight_l14_1498


namespace range_of_a_l14_1496

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 :=
by {
  sorry -- Proof is not required as per instructions.
}

end range_of_a_l14_1496


namespace quadratic_function_min_value_l14_1448

theorem quadratic_function_min_value (x : ℝ) (y : ℝ) :
  (y = x^2 - 2 * x + 6) →
  (∃ x_min, x_min = 1 ∧ y = (1 : ℝ)^2 - 2 * (1 : ℝ) + 6 ∧ (∀ x, y ≥ x^2 - 2 * x + 6)) :=
by
  sorry

end quadratic_function_min_value_l14_1448


namespace geometric_sequence_a2_value_l14_1413

theorem geometric_sequence_a2_value
    (a : ℕ → ℝ)
    (h1 : a 1 = 1/5)
    (h3 : a 3 = 5)
    (geometric : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) :
    a 2 = 1 ∨ a 2 = -1 := by
  sorry

end geometric_sequence_a2_value_l14_1413


namespace computation_of_difference_of_squares_l14_1424

theorem computation_of_difference_of_squares : (65^2 - 35^2) = 3000 := sorry

end computation_of_difference_of_squares_l14_1424


namespace rowing_speed_in_still_water_l14_1404

noncomputable def speedInStillWater (distance_m : ℝ) (time_s : ℝ) (speed_current : ℝ) : ℝ :=
  let distance_km := distance_m / 1000
  let time_h := time_s / 3600
  let speed_downstream := distance_km / time_h
  speed_downstream - speed_current

theorem rowing_speed_in_still_water :
  speedInStillWater 45.5 9.099272058235341 8.5 = 9.5 :=
by
  sorry

end rowing_speed_in_still_water_l14_1404


namespace arnold_danny_age_l14_1443

theorem arnold_danny_age (x : ℕ) : (x + 1) * (x + 1) = x * x + 9 → x = 4 :=
by
  intro h
  sorry

end arnold_danny_age_l14_1443


namespace larger_number_of_two_integers_l14_1470

theorem larger_number_of_two_integers (x y : ℤ) (h1 : x * y = 30) (h2 : x + y = 13) : (max x y = 10) :=
by
  sorry

end larger_number_of_two_integers_l14_1470


namespace sum_of_variables_l14_1410

theorem sum_of_variables (x y z : ℝ) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z)
  (hxy : x * y = 30) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 :=
by
  sorry

end sum_of_variables_l14_1410


namespace width_of_box_l14_1427

theorem width_of_box 
(length depth num_cubes : ℕ)
(h_length : length = 49)
(h_depth : depth = 14)
(h_num_cubes : num_cubes = 84)
: ∃ width : ℕ, width = 42 := 
sorry

end width_of_box_l14_1427


namespace nancy_carrots_l14_1401

theorem nancy_carrots (picked_day_1 threw_out total_left total_final picked_next_day : ℕ)
  (h1 : picked_day_1 = 12)
  (h2 : threw_out = 2)
  (h3 : total_final = 31)
  (h4 : total_left = picked_day_1 - threw_out)
  (h5 : total_final = total_left + picked_next_day) :
  picked_next_day = 21 :=
by
  sorry

end nancy_carrots_l14_1401


namespace rectangular_prism_faces_l14_1423

theorem rectangular_prism_faces (n : ℕ) (h1 : ∀ z : ℕ, z > 0 → z^3 = 2 * n^3) 
  (h2 : n > 0) :
  (∃ f : ℕ, f = (1 / 6 : ℚ) * (6 * 2 * n^3) ∧ 
    f = 10 * n^2) ↔ n = 5 := by
sorry

end rectangular_prism_faces_l14_1423


namespace negation_example_l14_1492

theorem negation_example :
  (¬ ∀ x y : ℝ, |x + y| > 3) ↔ (∃ x y : ℝ, |x + y| ≤ 3) :=
by
  sorry

end negation_example_l14_1492


namespace odd_function_behavior_l14_1430

theorem odd_function_behavior (f : ℝ → ℝ)
  (h_odd: ∀ x, f (-x) = -f x)
  (h_increasing: ∀ x y, 3 ≤ x → x ≤ 7 → 3 ≤ y → y ≤ 7 → x < y → f x < f y)
  (h_max: ∀ x, 3 ≤ x → x ≤ 7 → f x ≤ 5) :
  (∀ x, -7 ≤ x → x ≤ -3 → f x ≥ -5) ∧ (∀ x y, -7 ≤ x → x ≤ -3 → -7 ≤ y → y ≤ -3 → x < y → f x < f y) :=
sorry

end odd_function_behavior_l14_1430


namespace train_stop_time_l14_1481

theorem train_stop_time : 
  let speed_exc_stoppages := 45.0
  let speed_inc_stoppages := 31.0
  let speed_diff := speed_exc_stoppages - speed_inc_stoppages
  let km_per_minute := speed_exc_stoppages / 60.0
  let stop_time := speed_diff / km_per_minute
  stop_time = 18.67 :=
  by
    sorry

end train_stop_time_l14_1481


namespace cone_volume_l14_1422

theorem cone_volume (r l: ℝ) (r_eq : r = 2) (l_eq : l = 4) (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) :
  (1 / 3) * π * r^2 * h = (8 * Real.sqrt 3 * π) / 3 :=
by
  -- Sorry to skip the proof
  sorry

end cone_volume_l14_1422


namespace find_initial_mangoes_l14_1466

-- Define the initial conditions
def initial_apples : Nat := 7
def initial_oranges : Nat := 8
def apples_taken : Nat := 2
def oranges_taken : Nat := 2 * apples_taken
def remaining_fruits : Nat := 14
def mangoes_remaining (M : Nat) : Nat := M / 3

-- Define the problem statement
theorem find_initial_mangoes (M : Nat) (hM : 7 - apples_taken + 8 - oranges_taken + mangoes_remaining M = remaining_fruits) : M = 15 :=
by
  sorry

end find_initial_mangoes_l14_1466


namespace prob_green_second_given_first_green_l14_1472

def total_balls : Nat := 14
def green_balls : Nat := 8
def red_balls : Nat := 6

def prob_green_first_draw : ℚ := green_balls / total_balls

theorem prob_green_second_given_first_green :
  prob_green_first_draw = (8 / 14) → (green_balls / total_balls) = (4 / 7) :=
by
  sorry

end prob_green_second_given_first_green_l14_1472


namespace union_sets_eq_l14_1451

-- Definitions of the given sets
def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2}

-- The theorem to prove the union of sets A and B equals \{0, 1, 2\}
theorem union_sets_eq : (A ∪ B) = {0, 1, 2} := by
  sorry

end union_sets_eq_l14_1451


namespace compute_expression_l14_1429

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l14_1429
