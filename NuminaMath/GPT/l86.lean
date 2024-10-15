import Mathlib

namespace NUMINAMATH_GPT_evaluate_fractions_l86_8670

-- Define the fractions
def frac1 := 7 / 12
def frac2 := 8 / 15
def frac3 := 2 / 5

-- Prove that the sum and difference is as specified
theorem evaluate_fractions :
  frac1 + frac2 - frac3 = 43 / 60 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fractions_l86_8670


namespace NUMINAMATH_GPT_find_common_difference_l86_8632

variable {a : ℕ → ℤ}  -- Define a sequence indexed by natural numbers, returning integers
variable (d : ℤ)  -- Define the common difference as an integer

-- The conditions: sequence is arithmetic, a_2 = 14, a_5 = 5
axiom arithmetic_sequence (n : ℕ) : a n = a 0 + n * d
axiom a_2_eq_14 : a 2 = 14
axiom a_5_eq_5 : a 5 = 5

-- The proof statement
theorem find_common_difference : d = -3 :=
by sorry

end NUMINAMATH_GPT_find_common_difference_l86_8632


namespace NUMINAMATH_GPT_find_number_l86_8636

theorem find_number (x : ℤ) (h : x - 27 = 49) : x = 76 := by
  sorry

end NUMINAMATH_GPT_find_number_l86_8636


namespace NUMINAMATH_GPT_sum_ratio_l86_8695

variable {α : Type _} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0       => a₁
| (n + 1) => (geometric_sequence a₁ q n) * q

noncomputable def sum_geometric (a₁ q : α) (n : ℕ) : α :=
  if q = 1 then a₁ * (n + 1)
  else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_ratio {a₁ q : α} (h : 8 * (geometric_sequence a₁ q 1) + (geometric_sequence a₁ q 4) = 0) :
  (sum_geometric a₁ q 4) / (sum_geometric a₁ q 1) = -11 :=
sorry

end NUMINAMATH_GPT_sum_ratio_l86_8695


namespace NUMINAMATH_GPT_harry_total_cost_l86_8624

-- Define the price of each type of seed packet
def pumpkin_price : ℝ := 2.50
def tomato_price : ℝ := 1.50
def chili_pepper_price : ℝ := 0.90
def zucchini_price : ℝ := 1.20
def eggplant_price : ℝ := 1.80

-- Define the quantities Harry wants to buy
def pumpkin_qty : ℕ := 4
def tomato_qty : ℕ := 6
def chili_pepper_qty : ℕ := 7
def zucchini_qty : ℕ := 3
def eggplant_qty : ℕ := 5

-- Calculate the total cost
def total_cost : ℝ :=
  pumpkin_qty * pumpkin_price +
  tomato_qty * tomato_price +
  chili_pepper_qty * chili_pepper_price +
  zucchini_qty * zucchini_price +
  eggplant_qty * eggplant_price

-- The proof problem
theorem harry_total_cost : total_cost = 38.90 := by
  sorry

end NUMINAMATH_GPT_harry_total_cost_l86_8624


namespace NUMINAMATH_GPT_pears_more_than_apples_l86_8639

theorem pears_more_than_apples (red_apples green_apples pears : ℕ) (h1 : red_apples = 15) (h2 : green_apples = 8) (h3 : pears = 32) : (pears - (red_apples + green_apples) = 9) :=
by
  sorry

end NUMINAMATH_GPT_pears_more_than_apples_l86_8639


namespace NUMINAMATH_GPT_A_intersection_B_eq_C_l86_8637

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B := {x : ℝ | 0 < x ∧ x < 3}
def C := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem A_intersection_B_eq_C : A ∩ B = C := 
by sorry

end NUMINAMATH_GPT_A_intersection_B_eq_C_l86_8637


namespace NUMINAMATH_GPT_ratio_of_dinner_to_lunch_l86_8600

theorem ratio_of_dinner_to_lunch
  (dinner: ℕ) (lunch: ℕ) (breakfast: ℕ) (k: ℕ)
  (h1: dinner = 240)
  (h2: dinner = k * lunch)
  (h3: dinner = 6 * breakfast)
  (h4: breakfast + lunch + dinner = 310) :
  dinner / lunch = 8 :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_ratio_of_dinner_to_lunch_l86_8600


namespace NUMINAMATH_GPT_length_of_second_race_l86_8674

theorem length_of_second_race :
  ∀ (V_A V_B V_C T T' L : ℝ),
  (V_A * T = 200) →
  (V_B * T = 180) →
  (V_C * T = 162) →
  (V_B * T' = L) →
  (V_C * T' = L - 60) →
  (L = 600) :=
by
  intros V_A V_B V_C T T' L h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_length_of_second_race_l86_8674


namespace NUMINAMATH_GPT_monotone_decreasing_sequence_monotone_increasing_sequence_l86_8679

theorem monotone_decreasing_sequence (f : ℝ → ℝ) (a : ℕ → ℝ) (c : ℝ) :
  (∀ n : ℕ, a (n + 1) = f (a n)) →
  (a 1 = 0) →
  (∀ x : ℝ, f x = f (1 - x)) →
  (∀ x : ℝ, f x = -x^2 + x + c) →
  (∀ n : ℕ, a (n + 1) < a n) ↔ c < 0 :=
by sorry

theorem monotone_increasing_sequence (f : ℝ → ℝ) (a : ℕ → ℝ) (c : ℝ) :
  (∀ n : ℕ, a (n + 1) = f (a n)) →
  (a 1 = 0) →
  (∀ x : ℝ, f x = f (1 - x)) →
  (∀ x : ℝ, f x = -x^2 + x + c) →
  (∀ n : ℕ, a (n + 1) > a n) ↔ c > 1/4 :=
by sorry

end NUMINAMATH_GPT_monotone_decreasing_sequence_monotone_increasing_sequence_l86_8679


namespace NUMINAMATH_GPT_sin_double_angle_neg_one_l86_8602

theorem sin_double_angle_neg_one (α : ℝ) (a b : ℝ × ℝ) (h₁ : a = (1, Real.cos α)) (h₂ : b = (Real.sin α, 1)) (h₃ : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.sin (2 * α) = -1 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_neg_one_l86_8602


namespace NUMINAMATH_GPT_triangle_height_l86_8626

theorem triangle_height (base height : ℝ) (area : ℝ) (h1 : base = 2) (h2 : area = 3) (area_formula : area = (base * height) / 2) : height = 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_height_l86_8626


namespace NUMINAMATH_GPT_bernardo_wins_at_5_l86_8671

theorem bernardo_wins_at_5 :
  ∃ N : ℕ, 0 ≤ N ∧ N ≤ 499 ∧ 27 * N + 360 < 500 ∧ ∀ M : ℕ, (0 ≤ M ∧ M ≤ 499 ∧ 27 * M + 360 < 500 → N ≤ M) :=
by
  sorry

end NUMINAMATH_GPT_bernardo_wins_at_5_l86_8671


namespace NUMINAMATH_GPT_cars_without_features_l86_8633

theorem cars_without_features (total_cars cars_with_air_bags cars_with_power_windows cars_with_sunroofs 
                               cars_with_air_bags_and_power_windows cars_with_air_bags_and_sunroofs 
                               cars_with_power_windows_and_sunroofs cars_with_all_features: ℕ)
                               (h1 : total_cars = 80)
                               (h2 : cars_with_air_bags = 45)
                               (h3 : cars_with_power_windows = 40)
                               (h4 : cars_with_sunroofs = 25)
                               (h5 : cars_with_air_bags_and_power_windows = 20)
                               (h6 : cars_with_air_bags_and_sunroofs = 15)
                               (h7 : cars_with_power_windows_and_sunroofs = 10)
                               (h8 : cars_with_all_features = 8) : 
    total_cars - (cars_with_air_bags + cars_with_power_windows + cars_with_sunroofs 
                 - cars_with_air_bags_and_power_windows - cars_with_air_bags_and_sunroofs 
                 - cars_with_power_windows_and_sunroofs + cars_with_all_features) = 7 :=
by sorry

end NUMINAMATH_GPT_cars_without_features_l86_8633


namespace NUMINAMATH_GPT_quadratic_root_value_l86_8662

theorem quadratic_root_value (a b : ℤ) (h : 2 * a - b = -3) : 6 * a - 3 * b + 6 = -3 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_root_value_l86_8662


namespace NUMINAMATH_GPT_banana_distinct_arrangements_l86_8672

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end NUMINAMATH_GPT_banana_distinct_arrangements_l86_8672


namespace NUMINAMATH_GPT_Jennifer_more_boxes_l86_8681

-- Definitions based on conditions
def Kim_boxes : ℕ := 54
def Jennifer_boxes : ℕ := 71

-- Proof statement (no actual proof needed, just the statement)
theorem Jennifer_more_boxes : Jennifer_boxes - Kim_boxes = 17 := by
  sorry

end NUMINAMATH_GPT_Jennifer_more_boxes_l86_8681


namespace NUMINAMATH_GPT_find_f_l86_8645

theorem find_f
  (d e f : ℝ)
  (vertex_x vertex_y : ℝ)
  (p_x p_y : ℝ)
  (vertex_cond : vertex_x = 3 ∧ vertex_y = -1)
  (point_cond : p_x = 5 ∧ p_y = 1)
  (equation : ∀ y : ℝ, ∃ x : ℝ, x = d * y^2 + e * y + f) :
  f = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_l86_8645


namespace NUMINAMATH_GPT_brianna_fraction_left_l86_8666

theorem brianna_fraction_left (m n c : ℕ) (h : (1 : ℚ) / 4 * m = 1 / 2 * n * c) : 
  (m - (n * c) - (1 / 10 * m)) / m = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_brianna_fraction_left_l86_8666


namespace NUMINAMATH_GPT_greatest_product_l86_8678

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end NUMINAMATH_GPT_greatest_product_l86_8678


namespace NUMINAMATH_GPT_min_value_of_expr_l86_8625

theorem min_value_of_expr (a b c : ℝ) (h1 : 0 < a ∧ a ≤ b ∧ b ≤ c) (h2 : a * b * c = 1) :
    (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (a + c)) + 1 / (c ^ 3 * (a + b))) ≥ 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_expr_l86_8625


namespace NUMINAMATH_GPT_remainder_when_divided_by_7_l86_8619

-- Definitions based on conditions
def k_condition (k : ℕ) : Prop :=
(k % 5 = 2) ∧ (k % 6 = 5) ∧ (k < 38)

-- Theorem based on the question and correct answer
theorem remainder_when_divided_by_7 {k : ℕ} (h : k_condition k) : k % 7 = 3 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_7_l86_8619


namespace NUMINAMATH_GPT_edge_length_increase_l86_8617

theorem edge_length_increase (e e' : ℝ) (A : ℝ) (hA : ∀ e, A = 6 * e^2)
  (hA' : 2.25 * A = 6 * e'^2) :
  (e' - e) / e * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_edge_length_increase_l86_8617


namespace NUMINAMATH_GPT_cos_of_angle_B_l86_8684

theorem cos_of_angle_B (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : 6 * Real.sin A = 4 * Real.sin B) (h3 : 4 * Real.sin B = 3 * Real.sin C) : 
  Real.cos B = Real.sqrt 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_angle_B_l86_8684


namespace NUMINAMATH_GPT_area_of_rectangle_l86_8629

noncomputable def rectangle_area : ℚ :=
  let side1 : ℚ := 73 / 10
  let side2 : ℚ := 94 / 10
  let side3 : ℚ := 113 / 10
  let perimeter_triangle : ℚ := side1 + side2 + side3
  let width : ℚ := perimeter_triangle / 6
  let length : ℚ := 2 * width
  length * width

theorem area_of_rectangle : rectangle_area = 392 / 9 :=
  by 
  let side1 : ℚ := 73 / 10
  let side2 : ℚ := 94 / 10
  let side3 : ℚ := 113 / 10
  let perimeter_triangle : ℚ := side1 + side2 + side3
  let width : ℚ := perimeter_triangle / 6
  let length : ℚ := 2 * width
  have : length * width = 392 / 9 := sorry
  exact this

end NUMINAMATH_GPT_area_of_rectangle_l86_8629


namespace NUMINAMATH_GPT_binom_10_3_l86_8635

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end NUMINAMATH_GPT_binom_10_3_l86_8635


namespace NUMINAMATH_GPT_sum_series_eq_two_l86_8605

theorem sum_series_eq_two :
  ∑' k : Nat, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end NUMINAMATH_GPT_sum_series_eq_two_l86_8605


namespace NUMINAMATH_GPT_axis_of_symmetry_l86_8657

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (4 - x)) :
  ∀ y : ℝ, (∃ x₁ x₂ : ℝ, y = f x₁ ∧ y = f x₂ ∧ (x₁ + x₂) / 2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l86_8657


namespace NUMINAMATH_GPT_simplify_expression_d_l86_8656

variable (a b c : ℝ)

theorem simplify_expression_d : a - (b - c) = a - b + c :=
  sorry

end NUMINAMATH_GPT_simplify_expression_d_l86_8656


namespace NUMINAMATH_GPT_tom_initial_money_l86_8606

theorem tom_initial_money (spent_on_game : ℕ) (toy_cost : ℕ) (number_of_toys : ℕ)
    (total_spent : ℕ) (h1 : spent_on_game = 49) (h2 : toy_cost = 4)
    (h3 : number_of_toys = 2) (h4 : total_spent = spent_on_game + number_of_toys * toy_cost) :
  total_spent = 57 := by
  sorry

end NUMINAMATH_GPT_tom_initial_money_l86_8606


namespace NUMINAMATH_GPT_largest_digit_change_l86_8611

-- Definitions
def initial_number : ℝ := 0.12345

def change_digit (k : Fin 5) : ℝ :=
  match k with
  | 0 => 0.92345
  | 1 => 0.19345
  | 2 => 0.12945
  | 3 => 0.12395
  | 4 => 0.12349

theorem largest_digit_change :
  ∀ k : Fin 5, k ≠ 0 → change_digit 0 > change_digit k :=
by
  intros k hk
  sorry

end NUMINAMATH_GPT_largest_digit_change_l86_8611


namespace NUMINAMATH_GPT_train_passes_jogger_in_39_seconds_l86_8683

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def jogger_head_start : ℝ := 270
noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmph : ℝ := 45

noncomputable def to_meters_per_second (kmph : ℝ) : ℝ :=
  kmph * 1000 / 3600

noncomputable def jogger_speed_mps : ℝ :=
  to_meters_per_second jogger_speed_kmph

noncomputable def train_speed_mps : ℝ :=
  to_meters_per_second train_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps - jogger_speed_mps

noncomputable def total_distance : ℝ :=
  jogger_head_start + train_length

noncomputable def time_to_pass_jogger : ℝ :=
  total_distance / relative_speed_mps

theorem train_passes_jogger_in_39_seconds :
  time_to_pass_jogger = 39 := by
  sorry

end NUMINAMATH_GPT_train_passes_jogger_in_39_seconds_l86_8683


namespace NUMINAMATH_GPT_find_n_l86_8673

variable (a b c n : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n > 0)

theorem find_n (h1 : (a + b) / a = 3)
  (h2 : (b + c) / b = 4)
  (h3 : (c + a) / c = n) :
  n = 7 / 6 := 
sorry

end NUMINAMATH_GPT_find_n_l86_8673


namespace NUMINAMATH_GPT_cookies_per_batch_l86_8642

def family_size := 4
def chips_per_person := 18
def chips_per_cookie := 2
def batches := 3

theorem cookies_per_batch : (family_size * chips_per_person) / chips_per_cookie / batches = 12 := 
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_cookies_per_batch_l86_8642


namespace NUMINAMATH_GPT_olivia_race_time_l86_8661

theorem olivia_race_time (total_time : ℕ) (time_difference : ℕ) (olivia_time : ℕ)
  (h1 : total_time = 112) (h2 : time_difference = 4) (h3 : olivia_time + (olivia_time - time_difference) = total_time) :
  olivia_time = 58 :=
by
  sorry

end NUMINAMATH_GPT_olivia_race_time_l86_8661


namespace NUMINAMATH_GPT_matrix_power_identity_l86_8687

-- Define the matrix B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![0, 2]]

-- Define the identity matrix I
def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

-- Prove that B^15 - 3 * B^14 is equal to the given matrix
theorem matrix_power_identity :
  B ^ 15 - 3 • (B ^ 14) = ![![0, 4], ![0, -1]] :=
by
  -- Sorry is used here so the Lean code is syntactically correct
  sorry

end NUMINAMATH_GPT_matrix_power_identity_l86_8687


namespace NUMINAMATH_GPT_slope_of_PQ_l86_8692

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt x * (x / 3 + 1)

theorem slope_of_PQ :
  ∃ P Q : ℝ × ℝ,
    P = (0, 0) ∧ Q = (1, 8 / 3) ∧
    (∃ m : ℝ,
      m = 2 * Real.cos 0 ∧
      m = Real.sqrt 1 + 1 / Real.sqrt 1) ∧
    (Q.snd - P.snd) / (Q.fst - P.fst) = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_PQ_l86_8692


namespace NUMINAMATH_GPT_reflection_line_sum_l86_8621

-- Prove that the sum of m and b is 10 given the reflection conditions

theorem reflection_line_sum
    (m b : ℚ)
    (H : ∀ (x y : ℚ), (2, 2) = (x, y) → (8, 6) = (2 * (5 - (3 / 2) * (2 - x)), 2 + m * (y - 2)) ∧ y = m * x + b) :
  m + b = 10 :=
sorry

end NUMINAMATH_GPT_reflection_line_sum_l86_8621


namespace NUMINAMATH_GPT_largest_vertex_sum_l86_8668

def parabola_vertex_sum (a T : ℤ) (hT : T ≠ 0) : ℤ :=
  let x_vertex := T
  let y_vertex := a * T^2 - 2 * a * T^2
  x_vertex + y_vertex

theorem largest_vertex_sum (a T : ℤ) (hT : T ≠ 0)
  (hA : 0 = a * 0^2 + 0 * 0 + 0)
  (hB : 0 = a * (2 * T)^2 + (2 * T) * (2 * -T))
  (hC : 36 = a * (2 * T + 1)^2 + (2 * T - 2 * T * (2 * T + 1)))
  : parabola_vertex_sum a T hT ≤ -14 :=
sorry

end NUMINAMATH_GPT_largest_vertex_sum_l86_8668


namespace NUMINAMATH_GPT_katherine_has_5_bananas_l86_8694

theorem katherine_has_5_bananas
  (apples : ℕ) (pears : ℕ) (bananas : ℕ) (total_fruits : ℕ)
  (h1 : apples = 4)
  (h2 : pears = 3 * apples)
  (h3 : total_fruits = apples + pears + bananas)
  (h4 : total_fruits = 21) :
  bananas = 5 :=
by
  sorry

end NUMINAMATH_GPT_katherine_has_5_bananas_l86_8694


namespace NUMINAMATH_GPT_ways_to_write_1800_as_sum_of_4s_and_5s_l86_8640

theorem ways_to_write_1800_as_sum_of_4s_and_5s : 
  ∃ S : Finset (ℕ × ℕ), S.card = 91 ∧ ∀ (nm : ℕ × ℕ), nm ∈ S ↔ 4 * nm.1 + 5 * nm.2 = 1800 ∧ nm.1 ≥ 0 ∧ nm.2 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_ways_to_write_1800_as_sum_of_4s_and_5s_l86_8640


namespace NUMINAMATH_GPT_smallest_k_for_min_period_15_l86_8689

/-- Rational number with minimal period -/
def is_minimal_period (r : ℚ) (n : ℕ) : Prop :=
  ∃ m : ℤ, r = m / (10^n - 1)

variables (a b : ℚ)

-- Conditions for a and b
axiom ha : is_minimal_period a 30
axiom hb : is_minimal_period b 30

-- Condition for a - b
axiom hab_min_period : is_minimal_period (a - b) 15

-- Conclusion
theorem smallest_k_for_min_period_15 : ∃ k : ℕ, k = 6 ∧ is_minimal_period (a + k * b) 15 :=
by sorry

end NUMINAMATH_GPT_smallest_k_for_min_period_15_l86_8689


namespace NUMINAMATH_GPT_problem_equivalent_l86_8630

def modified_op (a b : ℝ) : ℝ := (a + b) ^ 2

theorem problem_equivalent (x y : ℝ) : 
  modified_op ((x + y) ^ 2) ((y + x) ^ 2) = 4 * (x + y) ^ 4 := 
by 
  sorry

end NUMINAMATH_GPT_problem_equivalent_l86_8630


namespace NUMINAMATH_GPT_willie_final_stickers_l86_8653

-- Definitions of initial stickers and given stickers
def willie_initial_stickers : ℝ := 36.0
def emily_gives : ℝ := 7.0

-- The statement to prove
theorem willie_final_stickers : willie_initial_stickers + emily_gives = 43.0 := by
  sorry

end NUMINAMATH_GPT_willie_final_stickers_l86_8653


namespace NUMINAMATH_GPT_sum_of_first_9_terms_l86_8676

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- a_n is the nth term of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of first n terms of the arithmetic sequence
def sum_seq (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- Hypotheses
axiom h1 : 2 * a 8 = 6 + a 11
axiom h2 : arithmetic_seq a
axiom h3 : sum_seq S a

-- The theorem we want to prove
theorem sum_of_first_9_terms : S 9 = 54 :=
sorry

end NUMINAMATH_GPT_sum_of_first_9_terms_l86_8676


namespace NUMINAMATH_GPT_calculation_result_l86_8607

theorem calculation_result :
  1500 * 451 * 0.0451 * 25 = 7627537500 :=
by
  -- Simply state without proof as instructed
  sorry

end NUMINAMATH_GPT_calculation_result_l86_8607


namespace NUMINAMATH_GPT_ratio_of_riding_to_total_l86_8663

-- Define the primary conditions from the problem
variables (H R W : ℕ)
variables (legs_on_ground : ℕ := 50)
variables (total_owners : ℕ := 10)
variables (legs_per_horse : ℕ := 4)
variables (legs_per_owner : ℕ := 2)

-- Express the conditions
def conditions : Prop :=
  (legs_on_ground = 6 * W) ∧
  (total_owners = H) ∧
  (H = R + W) ∧
  (H = 10)

-- Define the theorem with the given conditions and prove the required ratio
theorem ratio_of_riding_to_total (H R W : ℕ) (h : conditions H R W) : R / 10 = 1 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_of_riding_to_total_l86_8663


namespace NUMINAMATH_GPT_bread_last_days_l86_8667

def total_consumption_per_member_breakfast : ℕ := 4
def total_consumption_per_member_snacks : ℕ := 3
def total_consumption_per_member : ℕ := total_consumption_per_member_breakfast + total_consumption_per_member_snacks
def family_members : ℕ := 6
def daily_family_consumption : ℕ := family_members * total_consumption_per_member
def slices_per_loaf : ℕ := 10
def total_loaves : ℕ := 5
def total_bread_slices : ℕ := total_loaves * slices_per_loaf

theorem bread_last_days : total_bread_slices / daily_family_consumption = 1 :=
by
  sorry

end NUMINAMATH_GPT_bread_last_days_l86_8667


namespace NUMINAMATH_GPT_at_least_4_stayed_l86_8614

-- We define the number of people and their respective probabilities of staying.
def numPeople : ℕ := 8
def numCertain : ℕ := 5
def numUncertain : ℕ := 3
def probUncertainStay : ℚ := 1 / 3

-- We state the problem formally:
theorem at_least_4_stayed :
  (probUncertainStay ^ 3 * 3 + (probUncertainStay ^ 2 * (2 / 3) * 3) + (probUncertainStay * (2 / 3)^2 * 3)) = 19 / 27 :=
by
  sorry

end NUMINAMATH_GPT_at_least_4_stayed_l86_8614


namespace NUMINAMATH_GPT_green_hat_cost_l86_8616

theorem green_hat_cost (G : ℝ) (total_hats : ℕ) (blue_hats : ℕ) (green_hats : ℕ) (blue_cost : ℝ) (total_cost : ℝ) 
    (h₁ : blue_hats = 85) (h₂ : blue_cost = 6) (h₃ : green_hats = 90) (h₄ : total_cost = 600) 
    (h₅ : total_hats = blue_hats + green_hats) 
    (h₆ : total_cost = blue_hats * blue_cost + green_hats * G) : 
    G = 1 := by
  sorry

end NUMINAMATH_GPT_green_hat_cost_l86_8616


namespace NUMINAMATH_GPT_five_pow_10000_mod_1000_l86_8604

theorem five_pow_10000_mod_1000 (h : 5^500 ≡ 1 [MOD 1000]) : 5^10000 ≡ 1 [MOD 1000] := sorry

end NUMINAMATH_GPT_five_pow_10000_mod_1000_l86_8604


namespace NUMINAMATH_GPT_total_nuts_correct_l86_8651

-- Definitions for conditions
def w : ℝ := 0.25
def a : ℝ := 0.25
def p : ℝ := 0.15
def c : ℝ := 0.40

-- The theorem to be proven
theorem total_nuts_correct : w + a + p + c = 1.05 := by
  sorry

end NUMINAMATH_GPT_total_nuts_correct_l86_8651


namespace NUMINAMATH_GPT_calculate_shaded_area_l86_8610

noncomputable def square_shaded_area : ℝ := 
  let a := 10 -- side length of the square
  let s := a / 2 -- half side length, used for midpoints
  let total_area := a * a / 2 -- total area of a right triangle with legs a and a
  let triangle_DMA := total_area / 2 -- area of triangle DAM
  let triangle_DNG := triangle_DMA / 5 -- area of triangle DNG
  let triangle_CDM := total_area -- area of triangle CDM
  let shaded_area := triangle_CDM + triangle_DNG - triangle_DMA -- area of shaded region
  shaded_area

theorem calculate_shaded_area : square_shaded_area = 35 := 
by 
sorry

end NUMINAMATH_GPT_calculate_shaded_area_l86_8610


namespace NUMINAMATH_GPT_problem_l86_8647

variable (x y z w : ℚ)

theorem problem
  (h1 : x / y = 7)
  (h2 : z / y = 5)
  (h3 : z / w = 3 / 4) :
  w / x = 20 / 21 :=
by sorry

end NUMINAMATH_GPT_problem_l86_8647


namespace NUMINAMATH_GPT_p_is_sufficient_but_not_necessary_for_q_l86_8646

-- Definitions and conditions
def p (x : ℝ) : Prop := (x = 1)
def q (x : ℝ) : Prop := (x^2 - 3 * x + 2 = 0)

-- Theorem statement
theorem p_is_sufficient_but_not_necessary_for_q : ∀ x : ℝ, (p x → q x) ∧ (¬ (q x → p x)) :=
by
  sorry

end NUMINAMATH_GPT_p_is_sufficient_but_not_necessary_for_q_l86_8646


namespace NUMINAMATH_GPT_sin_alpha_value_l86_8675

theorem sin_alpha_value (α : ℝ) (h1 : Real.tan α = 2) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_value_l86_8675


namespace NUMINAMATH_GPT_tigers_home_games_l86_8652

-- Definitions based on the conditions
def losses : ℕ := 12
def ties : ℕ := losses / 2
def wins : ℕ := 38

-- Statement to prove
theorem tigers_home_games : losses + ties + wins = 56 := by
  sorry

end NUMINAMATH_GPT_tigers_home_games_l86_8652


namespace NUMINAMATH_GPT_classify_triangles_by_angles_l86_8627

-- Define the basic types and properties for triangles and their angle classifications
def acute_triangle (α β γ : ℝ) : Prop :=
  α < 90 ∧ β < 90 ∧ γ < 90

def right_triangle (α β γ : ℝ) : Prop :=
  α = 90 ∨ β = 90 ∨ γ = 90

def obtuse_triangle (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

-- Problem: Classify triangles by angles and prove that the correct classification is as per option A
theorem classify_triangles_by_angles :
  (∀ (α β γ : ℝ), acute_triangle α β γ ∨ right_triangle α β γ ∨ obtuse_triangle α β γ) :=
sorry

end NUMINAMATH_GPT_classify_triangles_by_angles_l86_8627


namespace NUMINAMATH_GPT_stratified_leader_selection_probability_of_mixed_leaders_l86_8655

theorem stratified_leader_selection :
  let num_first_grade := 150
  let num_second_grade := 100
  let total_leaders := 5
  let leaders_first_grade := (total_leaders * num_first_grade) / (num_first_grade + num_second_grade)
  let leaders_second_grade := (total_leaders * num_second_grade) / (num_first_grade + num_second_grade)
  leaders_first_grade = 3 ∧ leaders_second_grade = 2 :=
by
  sorry

theorem probability_of_mixed_leaders :
  let num_first_grade_leaders := 3
  let num_second_grade_leaders := 2
  let total_leaders := 5
  let total_ways := 10
  let favorable_ways := 6
  (favorable_ways / total_ways) = (3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_stratified_leader_selection_probability_of_mixed_leaders_l86_8655


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l86_8628

/-- 
  Given the ratio of the sum of the first n terms of two arithmetic sequences,
  prove the ratio of the 11th terms of these sequences.
-/
theorem arithmetic_sequence_ratio (S T : ℕ → ℚ) 
  (h : ∀ n, S n / T n = (7 * n + 1 : ℚ) / (4 * n + 2)) : 
  S 21 / T 21 = 74 / 43 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l86_8628


namespace NUMINAMATH_GPT_domain_of_f_2x_minus_1_l86_8697

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → (f x ≠ 0)) →
  (∀ y, 0 ≤ y ∧ y ≤ 1 ↔ exists x, (2 * x - 1 = y) ∧ (0 ≤ x ∧ x ≤ 1)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_2x_minus_1_l86_8697


namespace NUMINAMATH_GPT_tan_sum_pi_div_12_l86_8603

theorem tan_sum_pi_div_12 (h1 : Real.tan (Real.pi / 12) ≠ 0) (h2 : Real.tan (5 * Real.pi / 12) ≠ 0) :
  Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 := 
by
  sorry

end NUMINAMATH_GPT_tan_sum_pi_div_12_l86_8603


namespace NUMINAMATH_GPT_graph_of_equation_l86_8693

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_l86_8693


namespace NUMINAMATH_GPT_arithmetic_square_root_of_4_l86_8680

theorem arithmetic_square_root_of_4 : ∃ y : ℝ, y^2 = 4 ∧ y = 2 := 
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_4_l86_8680


namespace NUMINAMATH_GPT_technicians_count_l86_8643

noncomputable def total_salary := 8000 * 21
noncomputable def average_salary_all := 8000
noncomputable def average_salary_technicians := 12000
noncomputable def average_salary_rest := 6000
noncomputable def total_workers := 21

theorem technicians_count :
  ∃ (T R : ℕ),
  T + R = total_workers ∧
  average_salary_technicians * T + average_salary_rest * R = total_salary ∧
  T = 7 :=
by
  sorry

end NUMINAMATH_GPT_technicians_count_l86_8643


namespace NUMINAMATH_GPT_find_workers_l86_8658

def total_workers := 20
def male_work_days := 2
def female_work_days := 3

theorem find_workers (X Y : ℕ) 
  (h1 : X + Y = total_workers)
  (h2 : X / male_work_days + Y / female_work_days = 1) : 
  X = 12 ∧ Y = 8 :=
sorry

end NUMINAMATH_GPT_find_workers_l86_8658


namespace NUMINAMATH_GPT_line_intersects_hyperbola_left_branch_l86_8641

noncomputable def problem_statement (k : ℝ) : Prop :=
  ∀ (x y : ℝ), y = k * x - 1 ∧ x^2 - y^2 = 1 ∧ y < 0 → 
  k ∈ Set.Ioo (-Real.sqrt 2) (-1)

theorem line_intersects_hyperbola_left_branch (k : ℝ) :
  problem_statement k :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_hyperbola_left_branch_l86_8641


namespace NUMINAMATH_GPT_shaded_area_l86_8608

theorem shaded_area (r : ℝ) (sector_area : ℝ) (h1 : r = 4) (h2 : sector_area = 2 * Real.pi) : 
  sector_area - (1 / 2 * (r * Real.sqrt 2) * (r * Real.sqrt 2)) = 2 * Real.pi - 4 :=
by 
  -- Lean proof follows
  sorry

end NUMINAMATH_GPT_shaded_area_l86_8608


namespace NUMINAMATH_GPT_keiko_speed_l86_8669

theorem keiko_speed (a b s : ℝ) 
  (width : ℝ := 8) 
  (radius_inner := b) 
  (radius_outer := b + width)
  (time_difference := 48) 
  (L_inner := 2 * a + 2 * Real.pi * radius_inner)
  (L_outer := 2 * a + 2 * Real.pi * radius_outer) :
  (L_outer / s = L_inner / s + time_difference) → 
  s = Real.pi / 3 :=
by 
  sorry

end NUMINAMATH_GPT_keiko_speed_l86_8669


namespace NUMINAMATH_GPT_absolute_diff_half_l86_8648

theorem absolute_diff_half (x y : ℝ) 
  (h : ((x + y = x - y ∧ x - y = x * y) ∨ 
       (x + y = x * y ∧ x * y = x / y) ∨ 
       (x - y = x * y ∧ x * y = x / y))
       ∧ x ≠ 0 ∧ y ≠ 0) : 
     |y| - |x| = 1 / 2 := 
sorry

end NUMINAMATH_GPT_absolute_diff_half_l86_8648


namespace NUMINAMATH_GPT_count_two_digit_or_less_numbers_l86_8618

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end NUMINAMATH_GPT_count_two_digit_or_less_numbers_l86_8618


namespace NUMINAMATH_GPT_sum_mod_13_l86_8698

theorem sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_mod_13_l86_8698


namespace NUMINAMATH_GPT_smallest_x_correct_l86_8631

noncomputable def smallest_x (K : ℤ) : ℤ := 135000

theorem smallest_x_correct (K : ℤ) :
  (∃ x : ℤ, 180 * x = K ^ 5 ∧ x > 0) → smallest_x K = 135000 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_correct_l86_8631


namespace NUMINAMATH_GPT_julia_investment_l86_8634

-- Define the total investment and the relationship between the investments
theorem julia_investment:
  ∀ (m : ℕ), 
  m + 6 * m = 200000 → 6 * m = 171428 := 
by
  sorry

end NUMINAMATH_GPT_julia_investment_l86_8634


namespace NUMINAMATH_GPT_power_function_decreasing_m_l86_8664

theorem power_function_decreasing_m :
  ∀ (m : ℝ), (m^2 - 5*m - 5) * (2*m + 1) < 0 → m = -1 :=
by
  sorry

end NUMINAMATH_GPT_power_function_decreasing_m_l86_8664


namespace NUMINAMATH_GPT_smallest_n_divisible_by_5_l86_8622

def is_not_divisible_by_5 (x : ℤ) : Prop :=
  ¬ (x % 5 = 0)

def avg_is_integer (xs : List ℤ) : Prop :=
  (List.sum xs) % 5 = 0

theorem smallest_n_divisible_by_5 (n : ℕ) (h1 : n > 1980)
  (h2 : ∀ x ∈ List.range n, is_not_divisible_by_5 x)
  : n = 1985 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_smallest_n_divisible_by_5_l86_8622


namespace NUMINAMATH_GPT_inequality_system_solution_l86_8665

theorem inequality_system_solution (a b x : ℝ) 
  (h1 : x - a > 2)
  (h2 : x + 1 < b)
  (h3 : -1 < x)
  (h4 : x < 1) :
  (a + b) ^ 2023 = -1 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l86_8665


namespace NUMINAMATH_GPT_cos_beta_value_l86_8650

theorem cos_beta_value (α β : ℝ) (hα1 : 0 < α ∧ α < π/2) (hβ1 : 0 < β ∧ β < π/2) 
  (h1 : Real.sin α = 4/5) (h2 : Real.cos (α + β) = -12/13) : 
  Real.cos β = -16/65 := 
by 
  sorry

end NUMINAMATH_GPT_cos_beta_value_l86_8650


namespace NUMINAMATH_GPT_third_number_lcm_l86_8691

theorem third_number_lcm (n : ℕ) :
  n ∣ 360 ∧ lcm (lcm 24 36) n = 360 →
  n = 5 :=
by sorry

end NUMINAMATH_GPT_third_number_lcm_l86_8691


namespace NUMINAMATH_GPT_solution_set_of_inequality_l86_8609

theorem solution_set_of_inequality (a x : ℝ) (h : a > 0) : 
  (x^2 - (a + 1/a + 1) * x + a + 1/a < 0) ↔ (1 < x ∧ x < a + 1/a) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l86_8609


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l86_8699

-- Conditions
variables (x y : ℕ) -- Let x be the number of parcels each person sorts manually per hour,
                     -- y be the number of machines needed

def machine_efficiency : ℕ := 20 * x
def time_machines (parcels : ℕ) (machines : ℕ) : ℕ := parcels / (machines * machine_efficiency x)
def time_people (parcels : ℕ) (people : ℕ) : ℕ := parcels / (people * x)
def parcels_per_day : ℕ := 100000

-- Problem 1: Find x
axiom problem1 : (time_people 6000 20) - (time_machines 6000 5) = 4

-- Problem 2: Find y to sort 100000 parcels in a day with machines working 16 hours/day
axiom problem2 : 16 * machine_efficiency x * y ≥ parcels_per_day

-- Correct answers:
theorem part1_solution : x = 60 := by sorry
theorem part2_solution : y = 6 := by sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l86_8699


namespace NUMINAMATH_GPT_quadrilateral_perpendicular_diagonals_l86_8677

theorem quadrilateral_perpendicular_diagonals
  (AB BC CD DA : ℝ)
  (m n : ℝ)
  (hAB : AB = 6)
  (hBC : BC = m)
  (hCD : CD = 8)
  (hDA : DA = n)
  (h_diagonals_perpendicular : true)
  : m^2 + n^2 = 100 := 
by
  sorry

end NUMINAMATH_GPT_quadrilateral_perpendicular_diagonals_l86_8677


namespace NUMINAMATH_GPT_smallest_y_value_l86_8682

noncomputable def f (y : ℝ) : ℝ := 3 * y ^ 2 + 27 * y - 90
noncomputable def g (y : ℝ) : ℝ := y * (y + 15)

theorem smallest_y_value (y : ℝ) : (∀ y, f y = g y → y ≠ -9) → false := by
  sorry

end NUMINAMATH_GPT_smallest_y_value_l86_8682


namespace NUMINAMATH_GPT_find_t_find_s_find_a_find_c_l86_8615

-- Proof Problem I4.1
theorem find_t (p q r t : ℝ) (h1 : (p + q + r) / 3 = 12) (h2 : (p + q + r + t + 2 * t) / 5 = 15) : t = 13 :=
sorry

-- Proof Problem I4.2
theorem find_s (k t s : ℝ) (hk : k ≠ 0) (h1 : k^4 + (1 / k^4) = t + 1) (h2 : t = 13) (h_s : s = k^2 + (1 / k^2)) : s = 4 :=
sorry

-- Proof Problem I4.3
theorem find_a (s a b : ℝ) (hxₘ : 1 ≠ 11) (hyₘ : 2 ≠ 7) (h1 : (a, b) = ((1 * 11 + s * 1) / (1 + s), (1 * 7 + s * 2) / (1 + s))) (h_s : s = 4) : a = 3 :=
sorry

-- Proof Problem I4.4
theorem find_c (a c : ℝ) (h1 : ∀ x, a * x^2 + 12 * x + c = 0 → (a*x^2 + 12 * x + c = 0)) (h2 : ∃ x, a * x^2 + 12 * x + c = 0) : c = 36 / a :=
sorry

end NUMINAMATH_GPT_find_t_find_s_find_a_find_c_l86_8615


namespace NUMINAMATH_GPT_chris_eats_donuts_l86_8612

def daily_donuts := 10
def days := 12
def donuts_eaten_per_day := 1
def boxes_filled := 10
def donuts_per_box := 10

-- Define the total number of donuts made.
def total_donuts := daily_donuts * days

-- Define the total number of donuts Jeff eats.
def jeff_total_eats := donuts_eaten_per_day * days

-- Define the remaining donuts after Jeff eats his share.
def remaining_donuts := total_donuts - jeff_total_eats

-- Define the total number of donuts in the boxes.
def donuts_in_boxes := boxes_filled * donuts_per_box

-- The proof problem:
theorem chris_eats_donuts : remaining_donuts - donuts_in_boxes = 8 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_chris_eats_donuts_l86_8612


namespace NUMINAMATH_GPT_girls_in_club_l86_8644

/-
A soccer club has 30 members. For a recent team meeting, only 18 members could attend:
one-third of the girls attended but all of the boys attended. Prove that the number of 
girls in the soccer club is 18.
-/

variables (B G : ℕ)

-- Conditions
def total_members (B G : ℕ) := B + G = 30
def meeting_attendance (B G : ℕ) := (1/3 : ℚ) * G + B = 18

theorem girls_in_club (B G : ℕ) (h1 : total_members B G) (h2 : meeting_attendance B G) : G = 18 :=
  sorry

end NUMINAMATH_GPT_girls_in_club_l86_8644


namespace NUMINAMATH_GPT_possible_values_of_AC_l86_8685

theorem possible_values_of_AC (AB CD AC : ℝ) (m n : ℝ) (h1 : AB = 16) (h2 : CD = 4)
  (h3 : Set.Ioo m n = {x : ℝ | 4 < x ∧ x < 16}) : m + n = 20 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_AC_l86_8685


namespace NUMINAMATH_GPT_correct_number_of_arrangements_l86_8649

def arrangements_with_conditions (n : ℕ) : ℕ := 
  if n = 6 then
    let case1 := 120  -- when B is at the far right
    let case2 := 96   -- when A is at the far right
    case1 + case2
  else 0

theorem correct_number_of_arrangements : arrangements_with_conditions 6 = 216 :=
by {
  -- The detailed proof is omitted here
  sorry
}

end NUMINAMATH_GPT_correct_number_of_arrangements_l86_8649


namespace NUMINAMATH_GPT_unit_digit_product_zero_l86_8696

def unit_digit (n : ℕ) : ℕ := n % 10

theorem unit_digit_product_zero :
  let a := 785846
  let b := 1086432
  let c := 4582735
  let d := 9783284
  let e := 5167953
  let f := 3821759
  let g := 7594683
  unit_digit (a * b * c * d * e * f * g) = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_unit_digit_product_zero_l86_8696


namespace NUMINAMATH_GPT_possible_values_of_b_l86_8613

-- Set up the basic definitions and conditions
variable (a b c : ℝ)
variable (A B C : ℝ)

-- Assuming the conditions provided in the problem
axiom cond1 : a * (1 - Real.cos B) = b * Real.cos A
axiom cond2 : c = 3
axiom cond3 : 1 / 2 * a * c * Real.sin B = 2 * Real.sqrt 2

-- The theorem expressing the question and the correct answer
theorem possible_values_of_b : b = 2 ∨ b = 4 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_possible_values_of_b_l86_8613


namespace NUMINAMATH_GPT_graduation_messages_total_l86_8688

/-- Define the number of students in the class -/
def num_students : ℕ := 40

/-- Define the combination formula C(n, 2) for choosing 2 out of n -/
def combination (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Prove that the total number of graduation messages written is 1560 -/
theorem graduation_messages_total : combination num_students = 1560 :=
by
  sorry

end NUMINAMATH_GPT_graduation_messages_total_l86_8688


namespace NUMINAMATH_GPT_base_b_for_three_digits_l86_8620

theorem base_b_for_three_digits (b : ℕ) : b = 7 ↔ b^2 ≤ 256 ∧ 256 < b^3 := by
  sorry

end NUMINAMATH_GPT_base_b_for_three_digits_l86_8620


namespace NUMINAMATH_GPT_fraction_r_over_b_l86_8660

-- Definition of the conditions
def initial_expression (k : ℝ) : ℝ := 8 * k^2 - 12 * k + 20

-- Proposition statement
theorem fraction_r_over_b : ∃ a b r : ℝ, 
  (∀ k : ℝ, initial_expression k = a * (k + b)^2 + r) ∧ 
  r / b = -47.33 :=
sorry

end NUMINAMATH_GPT_fraction_r_over_b_l86_8660


namespace NUMINAMATH_GPT_division_of_15_by_neg_5_l86_8654

theorem division_of_15_by_neg_5 : 15 / (-5) = -3 :=
by
  sorry

end NUMINAMATH_GPT_division_of_15_by_neg_5_l86_8654


namespace NUMINAMATH_GPT_diagonal_of_rectangular_prism_l86_8623

noncomputable def diagonal_length (a b c : ℕ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem diagonal_of_rectangular_prism :
  diagonal_length 12 18 15 = 3 * Real.sqrt 77 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_of_rectangular_prism_l86_8623


namespace NUMINAMATH_GPT_covered_ratio_battonya_covered_ratio_sopron_l86_8601

noncomputable def angular_diameter_sun : ℝ := 1899 / 2
noncomputable def angular_diameter_moon : ℝ := 1866 / 2

def max_phase_battonya : ℝ := 0.766
def max_phase_sopron : ℝ := 0.678

def center_distance (R_M R_S f : ℝ) : ℝ :=
  R_M - (2 * f - 1) * R_S

-- Defining the hypothetical calculation (details omitted for brevity)
def covered_ratio (R_S R_M d : ℝ) : ℝ := 
  -- Placeholder for the actual calculation logic
  sorry

theorem covered_ratio_battonya :
  covered_ratio angular_diameter_sun angular_diameter_moon (center_distance angular_diameter_moon angular_diameter_sun max_phase_battonya) = 0.70 :=
  sorry

theorem covered_ratio_sopron :
  covered_ratio angular_diameter_sun angular_diameter_moon (center_distance angular_diameter_moon angular_diameter_sun max_phase_sopron) = 0.59 :=
  sorry

end NUMINAMATH_GPT_covered_ratio_battonya_covered_ratio_sopron_l86_8601


namespace NUMINAMATH_GPT_perimeter_of_ABCD_is_35_2_l86_8659

-- Definitions of geometrical properties and distances
variable (AB BC DC : ℝ)
variable (AB_perp_BC : ∃P, is_perpendicular AB BC)
variable (DC_parallel_AB : ∃Q, is_parallel DC AB)
variable (AB_length : AB = 7)
variable (BC_length : BC = 10)
variable (DC_length : DC = 6)

-- Target statement to be proved
theorem perimeter_of_ABCD_is_35_2
  (h1 : AB_perp_BC)
  (h2 : DC_parallel_AB)
  (h3 : AB_length)
  (h4 : BC_length)
  (h5 : DC_length) :
  ∃ P : ℝ, P = 35.2 :=
sorry

end NUMINAMATH_GPT_perimeter_of_ABCD_is_35_2_l86_8659


namespace NUMINAMATH_GPT_max_volume_of_hollow_cube_l86_8638

/-- 
We have 1000 solid cubes with edge lengths of 1 unit each. 
The small cubes can be glued together but not cut. 
The cube to be created is hollow with a wall thickness of 1 unit.
Prove that the maximum external volume of the cube we can create is 2197 cubic units.
--/

theorem max_volume_of_hollow_cube :
  ∃ x : ℕ, 6 * x^2 - 12 * x + 8 ≤ 1000 ∧ x^3 = 2197 :=
sorry

end NUMINAMATH_GPT_max_volume_of_hollow_cube_l86_8638


namespace NUMINAMATH_GPT_area_of_circle_l86_8690

-- Given condition as a Lean definition
def circle_eq (x y : ℝ) : Prop := 3 * x^2 + 3 * y^2 + 9 * x - 12 * y - 27 = 0

-- Theorem stating the goal
theorem area_of_circle : ∀ (x y : ℝ), circle_eq x y → ∃ r : ℝ, r = 15.25 ∧ ∃ a : ℝ, a = π * r := 
sorry

end NUMINAMATH_GPT_area_of_circle_l86_8690


namespace NUMINAMATH_GPT_quadratic_root_is_zero_then_m_neg_one_l86_8686

theorem quadratic_root_is_zero_then_m_neg_one (m : ℝ) (h_eq : (m-1) * 0^2 + 2 * 0 + m^2 - 1 = 0) : m = -1 := by
  sorry

end NUMINAMATH_GPT_quadratic_root_is_zero_then_m_neg_one_l86_8686
