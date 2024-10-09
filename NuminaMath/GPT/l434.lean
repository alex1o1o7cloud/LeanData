import Mathlib

namespace average_mileage_first_car_l434_43442

theorem average_mileage_first_car (X Y : ℝ) 
  (h1 : X + Y = 75) 
  (h2 : 25 * X + 35 * Y = 2275) : 
  X = 35 :=
by 
  sorry

end average_mileage_first_car_l434_43442


namespace solve_trig_eq_l434_43427

open Real

theorem solve_trig_eq (x a : ℝ) (hx1 : 0 < x) (hx2 : x < 2 * π) (ha : a > 0) :
    (sin (3 * x) + a * sin (2 * x) + 2 * sin x = 0) →
    (0 < a ∧ a < 2 → x = 0 ∨ x = π) ∧ 
    (a > 5 / 2 → ∃ α, (x = α ∨ x = 2 * π - α)) :=
by sorry

end solve_trig_eq_l434_43427


namespace triangle_area_bounded_by_lines_l434_43488

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l434_43488


namespace problem1_problem2_l434_43445

theorem problem1 :
  0.064 ^ (-1 / 3) - (-7 / 8) ^ 0 + 16 ^ 0.75 + 0.01 ^ (1 / 2) = 48 / 5 :=
by sorry

theorem problem2 :
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 
  - 25 ^ (Real.log 3 / Real.log 5) = -7 :=
by sorry

end problem1_problem2_l434_43445


namespace circle_tangent_to_yaxis_and_line_l434_43417

theorem circle_tangent_to_yaxis_and_line :
  (∃ C : ℝ → ℝ → Prop, 
    (∀ x y r : ℝ, C x y ↔ (x - 3) ^ 2 + (y - 2) ^ 2 = 9 ∨ (x + 1 / 3) ^ 2 + (y - 2) ^ 2 = 1 / 9) ∧ 
    (∀ y : ℝ, C 0 y → y = 2) ∧ 
    (∀ x y: ℝ, C x y → (∃ x1 : ℝ, 4 * x - 3 * y + 9 = 0 → 4 * x1 + 3 = 0))) :=
sorry

end circle_tangent_to_yaxis_and_line_l434_43417


namespace sqrt_expression_identity_l434_43406

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt 17 - 4

theorem sqrt_expression_identity : Real.sqrt ((-a)^3 + (b + 4)^2) = 4 :=
by
  -- Prove the statement

  sorry

end sqrt_expression_identity_l434_43406


namespace find_a_from_roots_l434_43455

theorem find_a_from_roots (θ : ℝ) (a : ℝ) (h1 : ∀ x : ℝ, 4 * x^2 + 2 * a * x + a = 0 → (x = Real.sin θ ∨ x = Real.cos θ)) :
  a = 1 - Real.sqrt 5 :=
by
  sorry

end find_a_from_roots_l434_43455


namespace find_x_collinear_l434_43481

-- Given vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (1, -3)
def vec_c (x : ℝ) : ℝ × ℝ := (-2, x)

-- Definition of vectors being collinear
def collinear (v₁ v₂ : ℝ × ℝ) : Prop :=
∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

-- Question: What is the value of x such that vec_a + vec_b is collinear with vec_c(x)?
theorem find_x_collinear : ∃ x : ℝ, collinear (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_c x) ∧ x = 1 :=
by
  sorry

end find_x_collinear_l434_43481


namespace plus_signs_count_l434_43419

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l434_43419


namespace expression_evaluation_l434_43403

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end expression_evaluation_l434_43403


namespace Angela_is_295_cm_l434_43482

noncomputable def Angela_height (Carl_height : ℕ) : ℕ :=
  let Becky_height := 2 * Carl_height
  let Amy_height := Becky_height + Becky_height / 5  -- 20% taller than Becky
  let Helen_height := Amy_height + 3
  let Angela_height := Helen_height + 4
  Angela_height

theorem Angela_is_295_cm : Angela_height 120 = 295 := 
by 
  sorry

end Angela_is_295_cm_l434_43482


namespace value_of_N_l434_43416

theorem value_of_N (N : ℕ) (h : Nat.choose N 5 = 231) : N = 11 := sorry

end value_of_N_l434_43416


namespace same_color_pair_exists_l434_43439

-- Define the coloring of a point on a plane
def is_colored (x y : ℝ) : Type := ℕ  -- Assume ℕ represents two colors 0 and 1

-- Prove there exists two points of the same color such that the distance between them is 2006 meters
theorem same_color_pair_exists (colored : ℝ → ℝ → ℕ) :
  (∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ colored x1 y1 = colored x2 y2 ∧ (x2 - x1)^2 + (y2 - y1)^2 = 2006^2) :=
sorry

end same_color_pair_exists_l434_43439


namespace final_bug_population_is_zero_l434_43467

def initial_population := 400
def spiders := 12
def spider_consumption := 7
def ladybugs := 5
def ladybug_consumption := 6
def mantises := 8
def mantis_consumption := 4

def day1_population := initial_population * 80 / 100

def predators_consumption_day := (spiders * spider_consumption) +
                                 (ladybugs * ladybug_consumption) +
                                 (mantises * mantis_consumption)

def day2_population := day1_population - predators_consumption_day
def day3_population := day2_population - predators_consumption_day
def day4_population := max 0 (day3_population - predators_consumption_day)
def day5_population := max 0 (day4_population - predators_consumption_day)
def day6_population := max 0 (day5_population - predators_consumption_day)

def day7_population := day6_population * 70 / 100

theorem final_bug_population_is_zero: 
  day7_population = 0 :=
  by
  sorry

end final_bug_population_is_zero_l434_43467


namespace ratio_c_div_d_l434_43454

theorem ratio_c_div_d (a b d : ℝ) (h1 : 8 = 0.02 * a) (h2 : 2 = 0.08 * b) (h3 : d = 0.05 * a) (c : ℝ) (h4 : c = b / a) : c / d = 1 / 320 := 
sorry

end ratio_c_div_d_l434_43454


namespace tan_value_l434_43499

theorem tan_value (α : ℝ) 
  (h : (2 * Real.cos α ^ 2 + Real.cos (π / 2 + 2 * α) - 1) / (Real.sqrt 2 * Real.sin (2 * α + π / 4)) = 4) : 
  Real.tan (2 * α + π / 4) = 1 / 4 :=
by
  sorry

end tan_value_l434_43499


namespace speed_of_goods_train_l434_43479

theorem speed_of_goods_train
  (length_train : ℕ)
  (length_platform : ℕ)
  (time_crossing : ℕ)
  (h_length_train : length_train = 240)
  (h_length_platform : length_platform = 280)
  (h_time_crossing : time_crossing = 26)
  : (length_train + length_platform) / time_crossing * (3600 / 1000) = 72 := 
by sorry

end speed_of_goods_train_l434_43479


namespace each_friend_pays_18_l434_43476

theorem each_friend_pays_18 (total_bill : ℝ) (silas_share : ℝ) (tip_fraction : ℝ) (num_friends : ℕ) (silas : ℕ) (remaining_friends : ℕ) :
  total_bill = 150 →
  silas_share = total_bill / 2 →
  tip_fraction = 0.1 →
  num_friends = 6 →
  remaining_friends = num_friends - 1 →
  silas = 1 →
  (total_bill - silas_share + tip_fraction * total_bill) / remaining_friends = 18 :=
by
  intros
  sorry

end each_friend_pays_18_l434_43476


namespace find_investment_sum_l434_43412

variable (P : ℝ)

def simple_interest (rate time : ℝ) (principal : ℝ) : ℝ :=
  principal * rate * time

theorem find_investment_sum (h : simple_interest 0.18 2 P - simple_interest 0.12 2 P = 240) :
  P = 2000 :=
by
  sorry

end find_investment_sum_l434_43412


namespace taxi_ride_cost_l434_43474

namespace TaxiFare

def baseFare : ℝ := 2.00
def costPerMile : ℝ := 0.30
def taxRate : ℝ := 0.10
def distance : ℝ := 8.0

theorem taxi_ride_cost :
  let fare_without_tax := baseFare + (costPerMile * distance)
  let tax := taxRate * fare_without_tax
  let total_fare := fare_without_tax + tax
  total_fare = 4.84 := by
  let fare_without_tax := baseFare + (costPerMile * distance)
  let tax := taxRate * fare_without_tax
  let total_fare := fare_without_tax + tax
  sorry

end TaxiFare

end taxi_ride_cost_l434_43474


namespace sum_of_first_2015_digits_l434_43487

noncomputable def repeating_decimal : List ℕ := [1, 4, 2, 8, 5, 7]

def sum_first_n_digits (digits : List ℕ) (n : ℕ) : ℕ :=
  let repeat_length := digits.length
  let full_cycles := n / repeat_length
  let remaining_digits := n % repeat_length
  full_cycles * (digits.sum) + (digits.take remaining_digits).sum

theorem sum_of_first_2015_digits :
  sum_first_n_digits repeating_decimal 2015 = 9065 :=
by
  sorry

end sum_of_first_2015_digits_l434_43487


namespace distance_from_point_to_line_condition_l434_43405

theorem distance_from_point_to_line_condition (a : ℝ) : (|a - 2| = 3) ↔ (a = 5 ∨ a = -1) :=
by
  sorry

end distance_from_point_to_line_condition_l434_43405


namespace jessie_weight_before_jogging_l434_43485

-- Definitions: conditions from the problem statement
variables (lost_weight current_weight : ℤ)
-- Conditions
def condition_lost_weight : Prop := lost_weight = 126
def condition_current_weight : Prop := current_weight = 66

-- Proposition to be proved
theorem jessie_weight_before_jogging (W_before_jogging : ℤ) :
  condition_lost_weight lost_weight → condition_current_weight current_weight →
  W_before_jogging = current_weight + lost_weight → W_before_jogging = 192 :=
by
  intros
  sorry

end jessie_weight_before_jogging_l434_43485


namespace population_correct_individual_correct_sample_correct_sample_size_correct_l434_43483

-- Definitions based on the problem conditions
def Population : Type := {s : String // s = "all seventh-grade students in the city"}
def Individual : Type := {s : String // s = "each seventh-grade student in the city"}
def Sample : Type := {s : String // s = "the 500 students that were drawn"}
def SampleSize : ℕ := 500

-- Prove given conditions
theorem population_correct (p : Population) : p.1 = "all seventh-grade students in the city" :=
by sorry

theorem individual_correct (i : Individual) : i.1 = "each seventh-grade student in the city" :=
by sorry

theorem sample_correct (s : Sample) : s.1 = "the 500 students that were drawn" :=
by sorry

theorem sample_size_correct : SampleSize = 500 :=
by sorry

end population_correct_individual_correct_sample_correct_sample_size_correct_l434_43483


namespace tank_depth_l434_43413

theorem tank_depth (d : ℝ)
    (field_length : ℝ) (field_breadth : ℝ)
    (tank_length : ℝ) (tank_breadth : ℝ)
    (remaining_field_area : ℝ)
    (rise_in_field_level : ℝ)
    (field_area_eq : field_length * field_breadth = 4500)
    (tank_area_eq : tank_length * tank_breadth = 500)
    (remaining_field_area_eq : remaining_field_area = 4500 - 500)
    (earth_volume_spread_eq : remaining_field_area * rise_in_field_level = 2000)
    (volume_eq : tank_length * tank_breadth * d = 2000)
  : d = 4 := by
  sorry

end tank_depth_l434_43413


namespace fraction_simplification_l434_43428

variable {x y z : ℝ}
variable (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z - z / x ≠ 0)

theorem fraction_simplification :
  (x^2 - 1 / y^2) / (z - z / x) = x / z :=
by
  sorry

end fraction_simplification_l434_43428


namespace same_bill_at_300_minutes_l434_43408

def monthlyBillA (x : ℕ) : ℝ := 15 + 0.1 * x
def monthlyBillB (x : ℕ) : ℝ := 0.15 * x

theorem same_bill_at_300_minutes : monthlyBillA 300 = monthlyBillB 300 := 
by
  sorry

end same_bill_at_300_minutes_l434_43408


namespace angle_bisector_inequality_l434_43459

theorem angle_bisector_inequality
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h_perimeter : (x + y + z) = 6) :
  (1 / x^2) + (1 / y^2) + (1 / z^2) ≥ 1 := by
  sorry

end angle_bisector_inequality_l434_43459


namespace profit_percent_l434_43446

theorem profit_percent (CP SP : ℕ) (h : CP * 5 = SP * 4) : 100 * (SP - CP) = 25 * CP :=
by
  sorry

end profit_percent_l434_43446


namespace days_c_worked_l434_43415

theorem days_c_worked 
    (days_a : ℕ) (days_b : ℕ) (wage_ratio_a : ℚ) (wage_ratio_b : ℚ) (wage_ratio_c : ℚ)
    (total_earnings : ℚ) (wage_c : ℚ) :
    days_a = 16 →
    days_b = 9 →
    wage_ratio_a = 3 →
    wage_ratio_b = 4 →
    wage_ratio_c = 5 →
    wage_c = 71.15384615384615 →
    total_earnings = 1480 →
    ∃ days_c : ℕ, (total_earnings = (wage_ratio_a / wage_ratio_c * wage_c * days_a) + 
                                 (wage_ratio_b / wage_ratio_c * wage_c * days_b) + 
                                 (wage_c * days_c)) ∧ days_c = 4 :=
by
  intros
  sorry

end days_c_worked_l434_43415


namespace rectangular_park_length_l434_43472

noncomputable def length_of_rectangular_park
  (P : ℕ) (B : ℕ) (L : ℕ) : Prop :=
  (P = 1000) ∧ (B = 200) ∧ (P = 2 * (L + B)) → (L = 300)

theorem rectangular_park_length : length_of_rectangular_park 1000 200 300 :=
by {
  sorry
}

end rectangular_park_length_l434_43472


namespace number_is_10_l434_43461

theorem number_is_10 (x : ℕ) (h : x * 15 = 150) : x = 10 :=
sorry

end number_is_10_l434_43461


namespace denom_asymptotes_sum_l434_43423

theorem denom_asymptotes_sum (A B C : ℤ)
  (h_denom : ∀ x, (x = -1 ∨ x = 3 ∨ x = 4) → x^3 + A * x^2 + B * x + C = 0) :
  A + B + C = 11 := 
sorry

end denom_asymptotes_sum_l434_43423


namespace no_four_distinct_sum_mod_20_l434_43424

theorem no_four_distinct_sum_mod_20 (R : Fin 9 → ℕ) (h : ∀ i, R i < 19) :
  ¬ ∃ (a b c d : Fin 9), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (R a + R b) % 20 = (R c + R d) % 20 := sorry

end no_four_distinct_sum_mod_20_l434_43424


namespace james_units_per_semester_l434_43432

theorem james_units_per_semester
  (cost_per_unit : ℕ)
  (total_cost : ℕ)
  (num_semesters : ℕ)
  (payment_per_semester : ℕ)
  (units_per_semester : ℕ)
  (H1 : cost_per_unit = 50)
  (H2 : total_cost = 2000)
  (H3 : num_semesters = 2)
  (H4 : payment_per_semester = total_cost / num_semesters)
  (H5 : units_per_semester = payment_per_semester / cost_per_unit) :
  units_per_semester = 20 :=
sorry

end james_units_per_semester_l434_43432


namespace points_in_rectangle_distance_l434_43407

/-- In a 3x4 rectangle, if 4 points are randomly located, 
    then the distance between at least two of them is at most 25/8. -/
theorem points_in_rectangle_distance (a b : ℝ) (h₁ : a = 3) (h₂ : b = 4)
  {points : Fin 4 → ℝ × ℝ}
  (h₃ : ∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ a)
  (h₄ : ∀ i, 0 ≤ (points i).2 ∧ (points i).2 ≤ b) :
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 25 / 8 := 
by
  sorry

end points_in_rectangle_distance_l434_43407


namespace readers_in_group_l434_43494

theorem readers_in_group (S L B T : ℕ) (hS : S = 120) (hL : L = 90) (hB : B = 60) :
  T = S + L - B → T = 150 :=
by
  intro h₁
  rw [hS, hL, hB] at h₁
  linarith

end readers_in_group_l434_43494


namespace arithmetic_sequence_common_difference_l434_43402

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℕ)
  (d : ℚ)
  (h_arith_seq : ∀ (n m : ℕ), (n > 0) → (m > 0) → (a n) / n - (a m) / m = (n - m) * d)
  (h_a3 : a 3 = 2)
  (h_a9 : a 9 = 12) :
  d = 1 / 9 ∧ a 12 = 20 :=
by 
  sorry

end arithmetic_sequence_common_difference_l434_43402


namespace price_increase_decrease_l434_43421

theorem price_increase_decrease (P : ℝ) (h : 0.84 * P = P * (1 - (x / 100)^2)) : x = 40 := by
  sorry

end price_increase_decrease_l434_43421


namespace geometric_series_sum_frac_l434_43410

open BigOperators

theorem geometric_series_sum_frac (q : ℚ) (a1 : ℚ) (a_list: List ℚ) (h_theta : q = 1 / 2) 
(h_a_list : a_list ⊆ [-4, -3, -2, 0, 1, 23, 4]) : 
  a1 * (1 + q^5) / (1 - q) = 33 / 4 := by
  sorry

end geometric_series_sum_frac_l434_43410


namespace find_x_l434_43404

theorem find_x (number x : ℝ) (h1 : 24 * number = 173 * x) (h2 : 24 * number = 1730) : x = 10 :=
by
  sorry

end find_x_l434_43404


namespace Thabo_owns_25_hardcover_nonfiction_books_l434_43471

variable (H P F : ℕ)

-- Conditions
def condition1 := P = H + 20
def condition2 := F = 2 * P
def condition3 := H + P + F = 160

-- Goal
theorem Thabo_owns_25_hardcover_nonfiction_books (H P F : ℕ) (h1 : condition1 H P) (h2 : condition2 P F) (h3 : condition3 H P F) : H = 25 :=
by
  sorry

end Thabo_owns_25_hardcover_nonfiction_books_l434_43471


namespace equation_has_real_root_l434_43477

theorem equation_has_real_root (x : ℝ) : (x^3 + 3 = 0) ↔ (x = -((3:ℝ)^(1/3))) :=
sorry

end equation_has_real_root_l434_43477


namespace smallest_area_of_square_containing_rectangles_l434_43457

noncomputable def smallest_area_square : ℕ :=
  let side1 := 3
  let side2 := 5
  let side3 := 4
  let side4 := 6
  let smallest_side := side1 + side3
  let square_area := smallest_side * smallest_side
  square_area

theorem smallest_area_of_square_containing_rectangles : smallest_area_square = 49 :=
by
  sorry

end smallest_area_of_square_containing_rectangles_l434_43457


namespace smallest_possible_value_of_d_l434_43463

theorem smallest_possible_value_of_d (c d : ℝ) (hc : 1 < c) (hd : c < d)
  (h_triangle1 : ¬(1 + c > d ∧ c + d > 1 ∧ 1 + d > c))
  (h_triangle2 : ¬(1 / c + 1 / d > 1 ∧ 1 / d + 1 > 1 / c ∧ 1 / c + 1 > 1 / d)) :
  d = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end smallest_possible_value_of_d_l434_43463


namespace calculate_share_A_l434_43450

-- Defining the investments
def investment_A : ℕ := 7000
def investment_B : ℕ := 11000
def investment_C : ℕ := 18000
def investment_D : ℕ := 13000
def investment_E : ℕ := 21000
def investment_F : ℕ := 15000
def investment_G : ℕ := 9000

-- Defining B's share
def share_B : ℚ := 3600

-- Function to calculate total investment
def total_investment : ℕ :=
  investment_A + investment_B + investment_C + investment_D + investment_E + investment_F + investment_G

-- Ratio of B's investment to total investment
def ratio_B : ℚ :=
  investment_B / total_investment

-- Calculate total profit using B's share and ratio
def total_profit : ℚ :=
  share_B / ratio_B

-- Ratio of A's investment to total investment
def ratio_A : ℚ :=
  investment_A / total_investment

-- Calculate A's share based on the total profit
def share_A : ℚ :=
  total_profit * ratio_A

-- The theorem to prove the share of A is approximately $2292.34
theorem calculate_share_A : 
  abs (share_A - 2292.34) < 0.01 :=
by
  sorry

end calculate_share_A_l434_43450


namespace abc_divisibility_l434_43462

theorem abc_divisibility (a b c : Nat) (h1 : a^3 ∣ b) (h2 : b^3 ∣ c) (h3 : c^3 ∣ a) :
  ∃ k : Nat, (a + b + c)^13 = k * a * b * c :=
by
  sorry

end abc_divisibility_l434_43462


namespace tan_value_l434_43458

open Real

noncomputable def geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

noncomputable def arithmetic_seq (b : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem tan_value
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : geometric_seq a)
  (hb : arithmetic_seq b)
  (h_geom : a 0 * a 5 * a 10 = -3 * sqrt 3)
  (h_arith : b 0 + b 5 + b 10 = 7 * π) :
  tan ((b 2 + b 8) / (1 - a 3 * a 7)) = -sqrt 3 :=
sorry

end tan_value_l434_43458


namespace second_layer_ratio_l434_43436

theorem second_layer_ratio
  (first_layer_sugar third_layer_sugar : ℕ)
  (third_layer_factor : ℕ)
  (h1 : first_layer_sugar = 2)
  (h2 : third_layer_sugar = 12)
  (h3 : third_layer_factor = 3) :
  third_layer_sugar = third_layer_factor * (2 * first_layer_sugar) →
  second_layer_factor = 2 :=
by
  sorry

end second_layer_ratio_l434_43436


namespace douglas_percent_votes_l434_43440

def percentageOfTotalVotesWon (votes_X votes_Y: ℕ) (percent_X percent_Y: ℕ) : ℕ :=
  let total_votes_Douglas : ℕ := (percent_X * 2 * votes_X + percent_Y * votes_Y)
  let total_votes_cast : ℕ := 3 * votes_Y
  (total_votes_Douglas * 100 / total_votes_cast)

theorem douglas_percent_votes (votes_X votes_Y : ℕ) (h_ratio : 2 * votes_X = votes_Y)
  (h_perc_X : percent_X = 64)
  (h_perc_Y : percent_Y = 46) :
  percentageOfTotalVotesWon votes_X votes_Y 64 46 = 58 := by
    sorry

end douglas_percent_votes_l434_43440


namespace functional_eq_implies_odd_l434_43484

variable (f : ℝ → ℝ)

def condition (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (x * f y) = y * f x

theorem functional_eq_implies_odd (h : condition f) : ∀ x : ℝ, f (-x) = -f x :=
sorry

end functional_eq_implies_odd_l434_43484


namespace greatest_number_of_bouquets_l434_43422

def sara_red_flowers : ℕ := 16
def sara_yellow_flowers : ℕ := 24

theorem greatest_number_of_bouquets : Nat.gcd sara_red_flowers sara_yellow_flowers = 8 := by
  rfl

end greatest_number_of_bouquets_l434_43422


namespace true_proposition_p_and_q_l434_43465

-- Define the proposition p
def p : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- Define the proposition q
def q : Prop := ∃ x : ℝ, x^3 = 1 - x^2

-- Statement to prove the conjunction p ∧ q
theorem true_proposition_p_and_q : p ∧ q := 
by 
    sorry

end true_proposition_p_and_q_l434_43465


namespace correct_calculation_is_A_l434_43475

theorem correct_calculation_is_A : (1 + (-2)) = -1 :=
by 
  sorry

end correct_calculation_is_A_l434_43475


namespace quadrilateral_angles_arith_prog_l434_43431

theorem quadrilateral_angles_arith_prog {x a b c : ℕ} (d : ℝ):
  (x^2 = 8^2 + 7^2 + 2 * 8 * 7 * Real.sin (3 * d)) →
  x = a + Real.sqrt b + Real.sqrt c →
  x = Real.sqrt 113 →
  a + b + c = 113 :=
by
  sorry

end quadrilateral_angles_arith_prog_l434_43431


namespace common_property_rhombus_rectangle_diagonals_l434_43418

-- Define a structure for Rhombus and its property
structure Rhombus (R : Type) :=
  (diagonals_perpendicular : Prop)
  (diagonals_bisect : Prop)

-- Define a structure for Rectangle and its property
structure Rectangle (R : Type) :=
  (diagonals_equal_length : Prop)
  (diagonals_bisect : Prop)

-- Define the theorem that states the common property between diagonals of both shapes
theorem common_property_rhombus_rectangle_diagonals (R : Type) 
  (rhombus_properties : Rhombus R) 
  (rectangle_properties : Rectangle R) :
  rhombus_properties.diagonals_bisect ∧ rectangle_properties.diagonals_bisect :=
by {
  -- Since the solution steps are not to be included, we conclude the proof with 'sorry'
  sorry
}

end common_property_rhombus_rectangle_diagonals_l434_43418


namespace sqrt_10_integer_decimal_partition_l434_43420

theorem sqrt_10_integer_decimal_partition:
  let a := Int.floor (Real.sqrt 10)
  let b := Real.sqrt 10 - a
  (Real.sqrt 10 + a) * b = 1 :=
by
  sorry

end sqrt_10_integer_decimal_partition_l434_43420


namespace find_four_real_numbers_l434_43491

theorem find_four_real_numbers
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 * x3 * x4 = 2)
  (h2 : x2 + x1 * x3 * x4 = 2)
  (h3 : x3 + x1 * x2 * x4 = 2)
  (h4 : x4 + x1 * x2 * x3 = 2) :
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) :=
sorry

end find_four_real_numbers_l434_43491


namespace zero_product_property_l434_43429

theorem zero_product_property {a b : ℝ} (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end zero_product_property_l434_43429


namespace pizza_boxes_sold_l434_43466

variables (P : ℕ) -- Representing the number of pizza boxes sold

def pizza_price : ℝ := 12
def fries_price : ℝ := 0.30
def soda_price : ℝ := 2

def fries_sold : ℕ := 40
def soda_sold : ℕ := 25

def goal_amount : ℝ := 500
def more_needed : ℝ := 258
def current_amount : ℝ := goal_amount - more_needed

-- Total earnings calculation
def total_earnings : ℝ := (P : ℝ) * pizza_price + fries_sold * fries_price + soda_sold * soda_price

theorem pizza_boxes_sold (h : total_earnings P = current_amount) : P = 15 := 
by
  sorry

end pizza_boxes_sold_l434_43466


namespace joe_probability_select_counsel_l434_43441

theorem joe_probability_select_counsel :
  let CANOE := ['C', 'A', 'N', 'O', 'E']
  let SHRUB := ['S', 'H', 'R', 'U', 'B']
  let FLOW := ['F', 'L', 'O', 'W']
  let COUNSEL := ['C', 'O', 'U', 'N', 'S', 'E', 'L']
  -- Probability of selecting C and O from CANOE
  let p_CANOE := 1 / (Nat.choose 5 2)
  -- Probability of selecting U, S, and E from SHRUB
  let comb_SHRUB := Nat.choose 5 3
  let count_USE := 3  -- Determined from the solution
  let p_SHRUB := count_USE / comb_SHRUB
  -- Probability of selecting L, O, W, F from FLOW
  let p_FLOW := 1 / 1
  -- Total probability
  let total_prob := p_CANOE * p_SHRUB * p_FLOW
  total_prob = 3 / 100 := by
    sorry

end joe_probability_select_counsel_l434_43441


namespace number_of_zeros_l434_43464

-- Definitions based on the conditions
def five_thousand := 5 * 10 ^ 3
def one_hundred := 10 ^ 2

-- The main theorem that we want to prove
theorem number_of_zeros : (five_thousand ^ 50) * (one_hundred ^ 2) = 10 ^ 154 * 5 ^ 50 := 
by sorry

end number_of_zeros_l434_43464


namespace fraction_equality_l434_43490

noncomputable def x := (4 : ℚ) / 6
noncomputable def y := (8 : ℚ) / 12

theorem fraction_equality : (6 * x + 8 * y) / (48 * x * y) = (7 : ℚ) / 16 := 
by 
  sorry

end fraction_equality_l434_43490


namespace product_of_b_values_is_neg_12_l434_43444

theorem product_of_b_values_is_neg_12 (b : ℝ) (y1 y2 x1 : ℝ) (h1 : y1 = 3) (h2 : y2 = 7) (h3 : x1 = 2) (h4 : y2 - y1 = 4) (h5 : ∃ b1 b2, b1 = x1 - 4 ∧ b2 = x1 + 4) : 
  (b1 * b2 = -12) :=
by
  sorry

end product_of_b_values_is_neg_12_l434_43444


namespace train_crosses_platform_in_25_002_seconds_l434_43489

noncomputable def time_to_cross_platform 
  (length_train : ℝ) 
  (length_platform : ℝ) 
  (speed_kmph : ℝ) : ℝ := 
  let total_distance := length_train + length_platform
  let speed_mps := speed_kmph * (1000 / 3600)
  total_distance / speed_mps

theorem train_crosses_platform_in_25_002_seconds :
  time_to_cross_platform 225 400.05 90 = 25.002 := by
  sorry

end train_crosses_platform_in_25_002_seconds_l434_43489


namespace area_of_sector_equals_13_75_cm2_l434_43434

noncomputable def radius : ℝ := 5 -- radius in cm
noncomputable def arc_length : ℝ := 5.5 -- arc length in cm
noncomputable def circumference : ℝ := 2 * Real.pi * radius -- circumference of the circle
noncomputable def area_of_circle : ℝ := Real.pi * radius^2 -- area of the entire circle

theorem area_of_sector_equals_13_75_cm2 :
  (arc_length / circumference) * area_of_circle = 13.75 :=
by sorry

end area_of_sector_equals_13_75_cm2_l434_43434


namespace S4k_eq_32_l434_43451

-- Definition of the problem conditions
variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)
variable (k : ℕ)

-- Conditions: Arithmetic sequence sum properties
axiom sum_arithmetic_sequence : ∀ {n : ℕ}, S n = n * (a 1 + a n) / 2

-- Given conditions
axiom Sk_eq_2 : S k = 2
axiom S3k_eq_18 : S (3 * k) = 18

-- Prove the required statement
theorem S4k_eq_32 : S (4 * k) = 32 :=
by
  sorry

end S4k_eq_32_l434_43451


namespace length_CF_is_7_l434_43437

noncomputable def CF_length
  (ABCD_rectangle : Prop)
  (triangle_ABE_right : Prop)
  (triangle_CDF_right : Prop)
  (area_triangle_ABE : ℝ)
  (length_AE length_DF : ℝ)
  (h1 : ABCD_rectangle)
  (h2 : triangle_ABE_right)
  (h3 : triangle_CDF_right)
  (h4 : area_triangle_ABE = 150)
  (h5 : length_AE = 15)
  (h6 : length_DF = 24) :
  ℝ :=
7

theorem length_CF_is_7
  (ABCD_rectangle : Prop)
  (triangle_ABE_right : Prop)
  (triangle_CDF_right : Prop)
  (area_triangle_ABE : ℝ)
  (length_AE length_DF : ℝ)
  (h1 : ABCD_rectangle)
  (h2 : triangle_ABE_right)
  (h3 : triangle_CDF_right)
  (h4 : area_triangle_ABE = 150)
  (h5 : length_AE = 15)
  (h6 : length_DF = 24) :
  CF_length ABCD_rectangle triangle_ABE_right triangle_CDF_right area_triangle_ABE length_AE length_DF h1 h2 h3 h4 h5 h6 = 7 :=
by
  sorry

end length_CF_is_7_l434_43437


namespace find_other_man_age_l434_43480

variable (avg_age_men inc_age_man other_man_age avg_age_women total_age_increase : ℕ)

theorem find_other_man_age 
    (h1 : inc_age_man = 2) 
    (h2 : ∀ m, m = 8 * (avg_age_men + inc_age_man))
    (h3 : ∃ y, y = 22) 
    (h4 : ∀ w, w = 29) 
    (h5 : total_age_increase = 2 * avg_age_women - (22 + other_man_age)) :
  total_age_increase = 16 → other_man_age = 20 :=
by
  intros
  sorry

end find_other_man_age_l434_43480


namespace sum_as_common_fraction_l434_43400

/-- The sum of 0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 as a common fraction -/
theorem sum_as_common_fraction : (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010) = (12345 / 160000) := by
  sorry

end sum_as_common_fraction_l434_43400


namespace solution_l434_43452

theorem solution (t : ℝ) :
  let x := 3 * t
  let y := t
  let z := 0
  x^2 - 9 * y^2 = z^2 :=
by
  sorry

end solution_l434_43452


namespace bianca_made_after_selling_l434_43409

def bianca_initial_cupcakes : ℕ := 14
def bianca_sold_cupcakes : ℕ := 6
def bianca_final_cupcakes : ℕ := 25

theorem bianca_made_after_selling :
  (bianca_initial_cupcakes - bianca_sold_cupcakes) + (bianca_final_cupcakes - (bianca_initial_cupcakes - bianca_sold_cupcakes)) = bianca_final_cupcakes :=
by
  sorry

end bianca_made_after_selling_l434_43409


namespace distance_between_foci_of_hyperbola_l434_43449

theorem distance_between_foci_of_hyperbola :
  ∀ x y : ℝ, (x^2 - 8 * x - 16 * y^2 - 16 * y = 48) → (∃ c : ℝ, 2 * c = 2 * Real.sqrt 63.75) :=
by
  sorry

end distance_between_foci_of_hyperbola_l434_43449


namespace grace_age_l434_43411

/-- Grace's age calculation based on given family ages. -/
theorem grace_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ)
  (h1 : mother_age = 80)
  (h2 : grandmother_age = 2 * mother_age)
  (h3 : grace_age = (3 * grandmother_age) / 8) : grace_age = 60 :=
by
  sorry

end grace_age_l434_43411


namespace sheep_count_l434_43497

theorem sheep_count (S H : ℕ) (h1 : S / H = 2 / 7) (h2 : H * 230 = 12880) : S = 16 :=
by 
  -- Lean proof goes here
  sorry

end sheep_count_l434_43497


namespace systematic_sampling_missiles_l434_43473

theorem systematic_sampling_missiles (S : Set ℕ) (hS : S = {n | 1 ≤ n ∧ n ≤ 50}) :
  (∃ seq : Fin 5 → ℕ, (∀ i : Fin 4, seq (Fin.succ i) - seq i = 10) ∧ seq 0 = 3)
  → (∃ seq : Fin 5 → ℕ, seq = ![3, 13, 23, 33, 43]) :=
by
  sorry

end systematic_sampling_missiles_l434_43473


namespace distribute_items_among_people_l434_43447

theorem distribute_items_among_people :
  (Nat.choose (10 + 3 - 1) 3) = 220 := 
by sorry

end distribute_items_among_people_l434_43447


namespace smaller_cubes_total_l434_43448

theorem smaller_cubes_total (n : ℕ) (painted_edges_cubes : ℕ) 
  (h1 : ∀ (a b : ℕ), a ^ 3 = n) 
  (h2 : ∀ (c : ℕ), painted_edges_cubes = 12) 
  (h3 : ∀ (d e : ℕ), 12 <= 2 * d * e) 
  : n = 27 :=
by
  sorry

end smaller_cubes_total_l434_43448


namespace initial_machines_l434_43478

theorem initial_machines (n x : ℕ) (hx : x > 0) (h : x / (4 * n) = x / 20) : n = 5 :=
by sorry

end initial_machines_l434_43478


namespace eval_complex_div_l434_43495

theorem eval_complex_div : 
  (i / (Real.sqrt 7 + 3 * I) = (3 / 16) + (Real.sqrt 7 / 16) * I) := 
by 
  sorry

end eval_complex_div_l434_43495


namespace smallest_b_for_34b_perfect_square_is_4_l434_43496

theorem smallest_b_for_34b_perfect_square_is_4 :
  ∃ n : ℕ, ∀ b : ℤ, b > 3 → (3 * b + 4 = n * n → b = 4) :=
by
  existsi 4
  intros b hb
  intro h
  sorry

end smallest_b_for_34b_perfect_square_is_4_l434_43496


namespace neither_necessary_nor_sufficient_l434_43469

theorem neither_necessary_nor_sufficient (x : ℝ) : 
  ¬ ((x = 0) ↔ (x^2 - 2 * x = 0) ∧ (x ≠ 0 → x^2 - 2 * x ≠ 0) ∧ (x = 0 → x^2 - 2 * x = 0)) := 
sorry

end neither_necessary_nor_sufficient_l434_43469


namespace time_elephants_l434_43492

def total_time := 130
def time_seals := 13
def time_penguins := 8 * time_seals

theorem time_elephants : total_time - (time_seals + time_penguins) = 13 :=
by
  sorry

end time_elephants_l434_43492


namespace boat_speed_in_still_water_l434_43433

theorem boat_speed_in_still_water  (b s : ℝ) (h1 : b + s = 13) (h2 : b - s = 9) : b = 11 :=
sorry

end boat_speed_in_still_water_l434_43433


namespace badges_before_exchange_l434_43401

theorem badges_before_exchange (V T : ℕ) (h1 : V = T + 5) (h2 : 76 * V + 20 * T = 80 * T + 24 * V - 100) :
  V = 50 ∧ T = 45 :=
by
  sorry

end badges_before_exchange_l434_43401


namespace solve_equation_l434_43438

theorem solve_equation (x : ℝ) : 
  (x ^ (Real.log x / Real.log 2) = x^5 / 32) ↔ (x = 2^((5 + Real.sqrt 5) / 2) ∨ x = 2^((5 - Real.sqrt 5) / 2)) := 
by 
  sorry

end solve_equation_l434_43438


namespace no_int_satisfies_both_congruences_l434_43414

theorem no_int_satisfies_both_congruences :
  ¬ ∃ n : ℤ, (n ≡ 5 [ZMOD 6]) ∧ (n ≡ 1 [ZMOD 21]) :=
sorry

end no_int_satisfies_both_congruences_l434_43414


namespace ratio_students_sent_home_to_remaining_l434_43468

theorem ratio_students_sent_home_to_remaining (total_students : ℕ) (students_taken_to_beach : ℕ)
    (students_still_in_school : ℕ) (students_sent_home : ℕ) 
    (h1 : total_students = 1000) (h2 : students_taken_to_beach = total_students / 2)
    (h3 : students_still_in_school = 250) 
    (h4 : students_sent_home = total_students / 2 - students_still_in_school) :
    (students_sent_home / students_still_in_school) = 1 := 
by
    sorry

end ratio_students_sent_home_to_remaining_l434_43468


namespace original_number_l434_43493

theorem original_number (n : ℕ) (h1 : 100000 ≤ n ∧ n < 1000000) (h2 : n / 100000 = 7) (h3 : (n % 100000) * 10 + 7 = n / 5) : n = 714285 :=
sorry

end original_number_l434_43493


namespace solution_set_of_inequality_system_l434_43498

theorem solution_set_of_inequality_system (x : ℝ) : (x + 1 > 0) ∧ (-2 * x ≤ 6) ↔ (x > -1) := 
by 
  sorry

end solution_set_of_inequality_system_l434_43498


namespace consecutive_integers_eq_l434_43430

theorem consecutive_integers_eq (a b c d e: ℕ) (h1: b = a + 1) (h2: c = a + 2) (h3: d = a + 3) (h4: e = a + 4) (h5: a^2 + b^2 + c^2 = d^2 + e^2) : a = 10 :=
by
  sorry

end consecutive_integers_eq_l434_43430


namespace min_cost_yogurt_l434_43460

theorem min_cost_yogurt (cost_per_box : ℕ) (boxes : ℕ) (promotion : ℕ → ℕ) (cost : ℕ) :
  cost_per_box = 4 → 
  boxes = 10 → 
  promotion 3 = 2 → 
  cost = 28 := 
by {
  -- The proof will go here
  sorry
}

end min_cost_yogurt_l434_43460


namespace arithmetic_sequence_general_term_l434_43486

noncomputable def an (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + (n - 1) * d
def bn (a_n : ℤ) : ℚ := (1 / 2)^a_n

theorem arithmetic_sequence_general_term
  (a_n : ℕ → ℤ)
  (b_1 b_2 b_3 : ℚ)
  (a_1 d : ℤ)
  (h_seq : ∀ n, a_n n = a_1 + (n - 1) * d)
  (h_b1 : b_1 = (1 / 2)^(a_n 1))
  (h_b2 : b_2 = (1 / 2)^(a_n 2))
  (h_b3 : b_3 = (1 / 2)^(a_n 3))
  (h_sum : b_1 + b_2 + b_3 = 21 / 8)
  (h_prod : b_1 * b_2 * b_3 = 1 / 8)
  : (∀ n, a_n n = 2 * n - 3) ∨ (∀ n, a_n n = 5 - 2 * n) :=
sorry

end arithmetic_sequence_general_term_l434_43486


namespace night_crew_worker_fraction_l434_43456

noncomputable def box_fraction_day : ℝ := 5/7

theorem night_crew_worker_fraction
  (D N : ℝ) -- Number of workers in day and night crew
  (B : ℝ)  -- Number of boxes each worker in the day crew loads
  (H1 : ∀ day_boxes_loaded : ℝ, day_boxes_loaded = D * B)
  (H2 : ∀ night_boxes_loaded : ℝ, night_boxes_loaded = N * (B / 2))
  (H3 : (D * B) / ((D * B) + (N * (B / 2))) = box_fraction_day) :
  N / D = 4/5 := 
sorry

end night_crew_worker_fraction_l434_43456


namespace evaluate_expression_l434_43443

theorem evaluate_expression : 
  (10^8 / (2.5 * 10^5) * 3) = 1200 :=
by
  sorry

end evaluate_expression_l434_43443


namespace max_AMC_expression_l434_43453

theorem max_AMC_expression (A M C : ℕ) (h : A + M + C = 15) : A * M * C + A * M + M * C + C * A ≤ 200 :=
by
  sorry

end max_AMC_expression_l434_43453


namespace solution_set_f_lt_g_l434_43426

noncomputable def f : ℝ → ℝ := sorry -- Assume f exists according to the given conditions

lemma f_at_one : f 1 = -2 := sorry

lemma f_derivative_neg (x : ℝ) : (deriv f x) < 0 := sorry

def g (x : ℝ) : ℝ := x - 3

lemma g_at_one : g 1 = -2 := sorry

theorem solution_set_f_lt_g :
  {x : ℝ | f x < g x} = {x : ℝ | 1 < x} :=
sorry

end solution_set_f_lt_g_l434_43426


namespace factor_sum_l434_43425

theorem factor_sum (P Q R : ℤ) (h : ∃ (b c : ℤ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + R*x + Q) : 
  P + Q + R = 11*P - 1 := 
sorry

end factor_sum_l434_43425


namespace subtract_value_l434_43435

theorem subtract_value (N x : ℤ) (h1 : (N - x) / 7 = 7) (h2 : (N - 6) / 8 = 6) : x = 5 := 
by 
  sorry

end subtract_value_l434_43435


namespace part1_x_values_part2_m_value_l434_43470

/-- 
Part 1: Given \(2x^2 + 3x - 5\) and \(-2x + 2\) are opposite numbers, 
prove that \(x = -\frac{3}{2}\) or \(x = 1\).
-/
theorem part1_x_values (x : ℝ)
  (hyp : 2 * x ^ 2 + 3 * x - 5 = -(-2 * x + 2)) :
  2 * x ^ 2 + 5 * x - 7 = 0 → (x = -3 / 2 ∨ x = 1) :=
by
  sorry

/-- 
Part 2: If \(\sqrt{m^2 - 6}\) and \(\sqrt{6m + 1}\) are of the same type, 
prove that \(m = 7\).
-/
theorem part2_m_value (m : ℝ)
  (hyp : m ^ 2 - 6 = 6 * m + 1) :
  7 ^ 2 - 6 = 6 * 7 + 1 → m = 7 :=
by
  sorry

end part1_x_values_part2_m_value_l434_43470
