import Mathlib

namespace NUMINAMATH_GPT_exists_nat_n_l1524_152454

theorem exists_nat_n (l : ℕ) (hl : l > 0) : ∃ n : ℕ, n^n + 47 ≡ 0 [MOD 2^l] := by
  sorry

end NUMINAMATH_GPT_exists_nat_n_l1524_152454


namespace NUMINAMATH_GPT_find_C_equation_l1524_152469

def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 2]]
def N : Matrix (Fin 2) (Fin 2) ℝ := ![![0, -1], ![1, 0]]

def C2_equation (x y : ℝ) : Prop := y = (1/8) * x^2

theorem find_C_equation (x y : ℝ) :
  (C2_equation (x) y) → (y^2 = 2 * x) := 
sorry

end NUMINAMATH_GPT_find_C_equation_l1524_152469


namespace NUMINAMATH_GPT_reunion_handshakes_l1524_152447

-- Condition: Number of boys in total
def total_boys : ℕ := 12

-- Condition: Number of left-handed boys
def left_handed_boys : ℕ := 4

-- Condition: Number of right-handed (not exclusively left-handed) boys
def right_handed_boys : ℕ := total_boys - left_handed_boys

-- Function to calculate combinations n choose 2 (number of handshakes in a group)
def combinations (n : ℕ) : ℕ := n * (n - 1) / 2

-- Condition: Number of handshakes among left-handed boys
def handshakes_left (n : ℕ) : ℕ := combinations left_handed_boys

-- Condition: Number of handshakes among right-handed boys
def handshakes_right (n : ℕ) : ℕ := combinations right_handed_boys

-- Problem statement: total number of handshakes
def total_handshakes (total_boys left_handed_boys right_handed_boys : ℕ) : ℕ :=
  handshakes_left left_handed_boys + handshakes_right right_handed_boys

theorem reunion_handshakes : total_handshakes total_boys left_handed_boys right_handed_boys = 34 :=
by sorry

end NUMINAMATH_GPT_reunion_handshakes_l1524_152447


namespace NUMINAMATH_GPT_distance_between_tangent_and_parallel_line_l1524_152452

noncomputable def distance_between_parallel_lines 
  (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ) 
  (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop) : ℝ :=
sorry

variable (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ)
variable (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop)

axiom tangent_line_at_point (M : ℝ × ℝ) (C : Set (ℝ × ℝ)) : (ℝ × ℝ → Prop)

theorem distance_between_tangent_and_parallel_line
  (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ)
  (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop) :
  C = { p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 } →
  M = (-2, 4) →
  l = tangent_line_at_point M C →
  l1 = { p | a * p.1 + 3 * p.2 + 2 * a = 0 } →
  distance_between_parallel_lines C center r M l a l1 = 12/5 :=
by
  intros hC hM hl hl1
  sorry

end NUMINAMATH_GPT_distance_between_tangent_and_parallel_line_l1524_152452


namespace NUMINAMATH_GPT_remainder_of_product_mod_10_l1524_152490

-- Definitions as conditions given in part a
def n1 := 2468
def n2 := 7531
def n3 := 92045

-- The problem expressed as a proof statement
theorem remainder_of_product_mod_10 :
  ((n1 * n2 * n3) % 10) = 0 :=
  by
    -- Sorry is used to skip the proof
    sorry

end NUMINAMATH_GPT_remainder_of_product_mod_10_l1524_152490


namespace NUMINAMATH_GPT_find_n_value_l1524_152427

theorem find_n_value (x y : ℕ) : x = 3 → y = 1 → n = x - y^(x - y) → x > y → n + x * y = 5 := by sorry

end NUMINAMATH_GPT_find_n_value_l1524_152427


namespace NUMINAMATH_GPT_calculate_grand_total_profit_l1524_152435

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

end NUMINAMATH_GPT_calculate_grand_total_profit_l1524_152435


namespace NUMINAMATH_GPT_binomial_22_5_computation_l1524_152418

theorem binomial_22_5_computation (h1 : Nat.choose 20 3 = 1140) (h2 : Nat.choose 20 4 = 4845) (h3 : Nat.choose 20 5 = 15504) :
    Nat.choose 22 5 = 26334 := by
  sorry

end NUMINAMATH_GPT_binomial_22_5_computation_l1524_152418


namespace NUMINAMATH_GPT_imaginary_part_div_l1524_152497

open Complex

theorem imaginary_part_div (z1 z2 : ℂ) (h1 : z1 = 1 + I) (h2 : z2 = I) :
  Complex.im (z1 / z2) = -1 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_div_l1524_152497


namespace NUMINAMATH_GPT_symmetry_xOz_A_l1524_152486

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetry_xOz (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y , z := p.z }

theorem symmetry_xOz_A :
  let A := Point3D.mk 2 (-3) 1
  symmetry_xOz A = Point3D.mk 2 3 1 :=
by
  sorry

end NUMINAMATH_GPT_symmetry_xOz_A_l1524_152486


namespace NUMINAMATH_GPT_parabola_vertex_sum_l1524_152409

theorem parabola_vertex_sum 
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + c) = (a * (x + 3)^2 + 4))
  (h2 : (a * 49 + 4) = -2)
  : a + b + c = 100 / 49 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_sum_l1524_152409


namespace NUMINAMATH_GPT_weight_of_each_bag_of_food_l1524_152436

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

end NUMINAMATH_GPT_weight_of_each_bag_of_food_l1524_152436


namespace NUMINAMATH_GPT_geometric_sequence_a6_l1524_152402

theorem geometric_sequence_a6 : 
  ∀ (a : ℕ → ℚ), (∀ n, a n ≠ 0) → a 1 = 3 → (∀ n, 2 * a (n+1) - a n = 0) → a 6 = 3 / 32 :=
by
  intros a h1 h2 h3
  sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l1524_152402


namespace NUMINAMATH_GPT_students_neither_play_football_nor_cricket_l1524_152408

theorem students_neither_play_football_nor_cricket
  (total_students football_players cricket_players both_players : ℕ)
  (h_total : total_students = 470)
  (h_football : football_players = 325)
  (h_cricket : cricket_players = 175)
  (h_both : both_players = 80) :
  (total_students - (football_players + cricket_players - both_players)) = 50 :=
by
  sorry

end NUMINAMATH_GPT_students_neither_play_football_nor_cricket_l1524_152408


namespace NUMINAMATH_GPT_find_xy_l1524_152466

theorem find_xy : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x^4 = y^2 + 71 ∧ x = 6 ∧ y = 35 :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l1524_152466


namespace NUMINAMATH_GPT_machine_C_time_l1524_152413

theorem machine_C_time (T_c : ℝ) :
  (1 / 4 + 1 / 2 + 1 / T_c = 11 / 12) → T_c = 6 :=
by
  sorry

end NUMINAMATH_GPT_machine_C_time_l1524_152413


namespace NUMINAMATH_GPT_inscribed_quadrilateral_exists_l1524_152438

theorem inscribed_quadrilateral_exists (a b c d : ℝ) (h1: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ∃ (p q : ℝ),
    p = Real.sqrt ((a * c + b * d) * (a * d + b * c) / (a * b + c * d)) ∧
    q = Real.sqrt ((a * b + c * d) * (a * d + b * c) / (a * c + b * d)) ∧
    a * c + b * d = p * q :=
by
  sorry

end NUMINAMATH_GPT_inscribed_quadrilateral_exists_l1524_152438


namespace NUMINAMATH_GPT_mariel_dogs_count_l1524_152465

theorem mariel_dogs_count (total_legs : ℤ) (num_dog_walkers : ℤ) (legs_per_walker : ℤ) 
  (other_dogs_count : ℤ) (legs_per_dog : ℤ) (mariel_dogs : ℤ) :
  total_legs = 36 →
  num_dog_walkers = 2 →
  legs_per_walker = 2 →
  other_dogs_count = 3 →
  legs_per_dog = 4 →
  mariel_dogs = (total_legs - (num_dog_walkers * legs_per_walker + other_dogs_count * legs_per_dog)) / legs_per_dog →
  mariel_dogs = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mariel_dogs_count_l1524_152465


namespace NUMINAMATH_GPT_remainder_cd_42_l1524_152423

theorem remainder_cd_42 (c d : ℕ) (p q : ℕ) (hc : c = 84 * p + 76) (hd : d = 126 * q + 117) : 
  (c + d) % 42 = 25 :=
by
  sorry

end NUMINAMATH_GPT_remainder_cd_42_l1524_152423


namespace NUMINAMATH_GPT_weight_of_replaced_oarsman_l1524_152441

noncomputable def average_weight (W : ℝ) : ℝ := W / 20

theorem weight_of_replaced_oarsman (W : ℝ) (W_avg : ℝ) (H1 : average_weight W = W_avg) (H2 : average_weight (W + 40) = W_avg + 2) : W = 40 :=
by sorry

end NUMINAMATH_GPT_weight_of_replaced_oarsman_l1524_152441


namespace NUMINAMATH_GPT_scatter_plot_correlation_l1524_152455

noncomputable def correlation_coefficient (points : List (ℝ × ℝ)) : ℝ := sorry

theorem scatter_plot_correlation {points : List (ℝ × ℝ)} 
  (h : ∃ (m : ℝ) (b : ℝ), m ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ points → y = m * x + b) :
  correlation_coefficient points = 1 := 
sorry

end NUMINAMATH_GPT_scatter_plot_correlation_l1524_152455


namespace NUMINAMATH_GPT_find_d_l1524_152476

theorem find_d (d : ℝ) (h1 : 0 < d) (h2 : d < 90) (h3 : Real.cos 16 = Real.sin 14 + Real.sin d) : d = 46 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1524_152476


namespace NUMINAMATH_GPT_total_weight_of_rings_l1524_152449

theorem total_weight_of_rings :
  let orange_ring := 0.08333333333333333
  let purple_ring := 0.3333333333333333
  let white_ring := 0.4166666666666667
  orange_ring + purple_ring + white_ring = 0.8333333333333333 :=
by
  let orange_ring := 0.08333333333333333
  let purple_ring := 0.3333333333333333
  let white_ring := 0.4166666666666667
  sorry

end NUMINAMATH_GPT_total_weight_of_rings_l1524_152449


namespace NUMINAMATH_GPT_find_value_of_expression_l1524_152446

theorem find_value_of_expression
  (a b : ℝ)
  (h₁ : a = 4 + Real.sqrt 15)
  (h₂ : b = 4 - Real.sqrt 15)
  (h₃ : ∀ x : ℝ, (x^3 - 9 * x^2 + 9 * x = 1) → (x = a ∨ x = b ∨ x = 1))
  : (a / b) + (b / a) = 62 := sorry

end NUMINAMATH_GPT_find_value_of_expression_l1524_152446


namespace NUMINAMATH_GPT_percentage_of_girls_with_dogs_l1524_152406

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

end NUMINAMATH_GPT_percentage_of_girls_with_dogs_l1524_152406


namespace NUMINAMATH_GPT_right_triangle_area_l1524_152494

theorem right_triangle_area (a b c : ℝ) (h₀ : a = 24) (h₁ : c = 30) (h2 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 216 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1524_152494


namespace NUMINAMATH_GPT_reservoir_capacity_l1524_152424

theorem reservoir_capacity (x : ℝ) (h1 : (3 / 8) * x - (1 / 4) * x = 100) : x = 800 :=
by
  sorry

end NUMINAMATH_GPT_reservoir_capacity_l1524_152424


namespace NUMINAMATH_GPT_range_of_a_l1524_152495

def A (x : ℝ) : Prop := x^2 - 4 * x + 3 ≤ 0
def B (x : ℝ) (a : ℝ) : Prop := x^2 - a * x < x - a

theorem range_of_a (a : ℝ) :
  (∀ x, A x → B x a) ∧ ∃ x, ¬ (A x → B x a) ↔ 1 ≤ a ∧ a ≤ 3 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1524_152495


namespace NUMINAMATH_GPT_battery_lasts_12_hours_more_l1524_152451

-- Define the battery consumption rates
def standby_consumption_rate : ℚ := 1 / 36
def active_consumption_rate : ℚ := 1 / 4

-- Define the usage times
def total_time_hours : ℚ := 12
def active_use_time_hours : ℚ := 1.5
def standby_time_hours : ℚ := total_time_hours - active_use_time_hours

-- Define the total battery used during standby and active use
def standby_battery_used : ℚ := standby_time_hours * standby_consumption_rate
def active_battery_used : ℚ := active_use_time_hours * active_consumption_rate
def total_battery_used : ℚ := standby_battery_used + active_battery_used

-- Define the remaining battery
def remaining_battery : ℚ := 1 - total_battery_used

-- Define how long the remaining battery will last on standby
def remaining_standby_time : ℚ := remaining_battery / standby_consumption_rate

-- Theorem stating the correct answer
theorem battery_lasts_12_hours_more :
  remaining_standby_time = 12 := 
sorry

end NUMINAMATH_GPT_battery_lasts_12_hours_more_l1524_152451


namespace NUMINAMATH_GPT_find_m_n_value_l1524_152417

theorem find_m_n_value (x m n : ℝ) 
  (h1 : x - 3 * m < 0) 
  (h2 : n - 2 * x < 0) 
  (h3 : -1 < x)
  (h4 : x < 3) 
  : (m + n) ^ 2023 = -1 :=
sorry

end NUMINAMATH_GPT_find_m_n_value_l1524_152417


namespace NUMINAMATH_GPT_equivalent_statements_l1524_152442

theorem equivalent_statements (P Q : Prop) : (¬P → Q) ↔ (¬Q → P) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_statements_l1524_152442


namespace NUMINAMATH_GPT_min_class_size_l1524_152473

theorem min_class_size (x : ℕ) (h : 50 ≤ 5 * x + 2) : 52 ≤ 5 * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_min_class_size_l1524_152473


namespace NUMINAMATH_GPT_translated_point_B_coords_l1524_152448

-- Define the initial point A
def point_A : ℝ × ℝ := (-2, 2)

-- Define the translation operations
def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - d)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

-- Define the translation of point A to point B
def point_B :=
  translate_right (translate_down point_A 4) 3

-- The proof statement
theorem translated_point_B_coords : point_B = (1, -2) :=
  by sorry

end NUMINAMATH_GPT_translated_point_B_coords_l1524_152448


namespace NUMINAMATH_GPT_isosceles_triangle_angle_sum_l1524_152444

theorem isosceles_triangle_angle_sum 
  (A B C : Type) 
  [Inhabited A] [Inhabited B] [Inhabited C]
  (AC AB : ℝ) 
  (angle_ABC : ℝ)
  (isosceles : AC = AB)
  (angle_A : angle_ABC = 70) :
  (∃ angle_B : ℝ, angle_B = 55) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_sum_l1524_152444


namespace NUMINAMATH_GPT_equation_not_expression_with_unknowns_l1524_152496

def is_equation (expr : String) : Prop :=
  expr = "equation"

def contains_unknowns (expr : String) : Prop :=
  expr = "contains unknowns"

theorem equation_not_expression_with_unknowns (expr : String) (h1 : is_equation expr) (h2 : contains_unknowns expr) : 
  (is_equation expr) = False := 
sorry

end NUMINAMATH_GPT_equation_not_expression_with_unknowns_l1524_152496


namespace NUMINAMATH_GPT_solve_for_y_l1524_152474

theorem solve_for_y (y : ℤ) (h : (y ≠ 2) → ((y^2 - 10*y + 24)/(y-2) + (4*y^2 + 8*y - 48)/(4*y - 8) = 0)) : y = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1524_152474


namespace NUMINAMATH_GPT_find_coordinates_C_find_range_t_l1524_152487

-- required definitions to handle the given points and vectors
structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

-- Given points
def A : Point := ⟨0, 4⟩
def B : Point := ⟨2, 0⟩

-- Proof for coordinates of point C
theorem find_coordinates_C :
  ∃ C : Point, vector A B = {x := 2 * (C.x - B.x), y := 2 * C.y} ∧ C = ⟨3, -2⟩ :=
by
  sorry

-- Proof for range of t
theorem find_range_t (t : ℝ) :
  let P := Point.mk 3 t
  let PA := vector P A
  let PB := vector P B
  (dot_product PA PB < 0 ∧ -3 * t ≠ -1 * (4 - t)) → 1 < t ∧ t < 3 :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_C_find_range_t_l1524_152487


namespace NUMINAMATH_GPT_average_sitting_time_per_student_l1524_152433

def total_travel_time_in_minutes : ℕ := 152
def number_of_seats : ℕ := 5
def number_of_students : ℕ := 8

theorem average_sitting_time_per_student :
  (total_travel_time_in_minutes * number_of_seats) / number_of_students = 95 := 
by
  sorry

end NUMINAMATH_GPT_average_sitting_time_per_student_l1524_152433


namespace NUMINAMATH_GPT_possible_values_of_sum_l1524_152491

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end NUMINAMATH_GPT_possible_values_of_sum_l1524_152491


namespace NUMINAMATH_GPT_train_speed_correct_l1524_152422

def train_length : ℝ := 110
def bridge_length : ℝ := 142
def crossing_time : ℝ := 12.598992080633549
def expected_speed : ℝ := 20.002

theorem train_speed_correct :
  (train_length + bridge_length) / crossing_time = expected_speed :=
by
  sorry

end NUMINAMATH_GPT_train_speed_correct_l1524_152422


namespace NUMINAMATH_GPT_dorothy_and_jemma_sales_l1524_152498

theorem dorothy_and_jemma_sales :
  ∀ (frames_sold_by_jemma price_per_frame_jemma : ℕ)
  (price_per_frame_dorothy frames_sold_by_dorothy : ℚ)
  (total_sales_jemma total_sales_dorothy total_sales : ℚ),
  price_per_frame_jemma = 5 →
  frames_sold_by_jemma = 400 →
  price_per_frame_dorothy = price_per_frame_jemma / 2 →
  frames_sold_by_jemma = 2 * frames_sold_by_dorothy →
  total_sales_jemma = frames_sold_by_jemma * price_per_frame_jemma →
  total_sales_dorothy = frames_sold_by_dorothy * price_per_frame_dorothy →
  total_sales = total_sales_jemma + total_sales_dorothy →
  total_sales = 2500 := by
  sorry

end NUMINAMATH_GPT_dorothy_and_jemma_sales_l1524_152498


namespace NUMINAMATH_GPT_scientific_notation_of_274000000_l1524_152457

theorem scientific_notation_of_274000000 :
  274000000 = 2.74 * 10^8 := by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_274000000_l1524_152457


namespace NUMINAMATH_GPT_tricycle_total_spokes_l1524_152425

noncomputable def front : ℕ := 20
noncomputable def middle : ℕ := 2 * front
noncomputable def back : ℝ := 20 * Real.sqrt 2
noncomputable def total_spokes : ℝ := front + middle + back

theorem tricycle_total_spokes : total_spokes = 88 :=
by
  sorry

end NUMINAMATH_GPT_tricycle_total_spokes_l1524_152425


namespace NUMINAMATH_GPT_determine_hyperbola_eq_l1524_152430

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

end NUMINAMATH_GPT_determine_hyperbola_eq_l1524_152430


namespace NUMINAMATH_GPT_flour_amount_indeterminable_l1524_152461

variable (flour_required : ℕ)
variable (sugar_required : ℕ := 11)
variable (sugar_added : ℕ := 10)
variable (flour_added : ℕ := 12)
variable (sugar_to_add : ℕ := 1)

theorem flour_amount_indeterminable :
  ¬ ∃ (flour_required : ℕ), flour_additional = flour_required - flour_added :=
by
  sorry

end NUMINAMATH_GPT_flour_amount_indeterminable_l1524_152461


namespace NUMINAMATH_GPT_three_digit_largest_fill_four_digit_smallest_fill_l1524_152477

theorem three_digit_largest_fill (n : ℕ) (h1 : n * 1000 + 28 * 4 < 1000) : n ≤ 2 := sorry

theorem four_digit_smallest_fill (n : ℕ) (h2 : n * 1000 + 28 * 4 ≥ 1000) : 3 ≤ n := sorry

end NUMINAMATH_GPT_three_digit_largest_fill_four_digit_smallest_fill_l1524_152477


namespace NUMINAMATH_GPT_mirror_area_correct_l1524_152428

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

end NUMINAMATH_GPT_mirror_area_correct_l1524_152428


namespace NUMINAMATH_GPT_find_x_given_inverse_relationship_l1524_152468

variable {x y : ℝ}

theorem find_x_given_inverse_relationship 
  (h₀ : x > 0) 
  (h₁ : y > 0) 
  (initial_condition : 3^2 * 25 = 225)
  (inversion_condition : x^2 * y = 225)
  (query : y = 1200) :
  x = Real.sqrt (3 / 16) :=
by
  sorry

end NUMINAMATH_GPT_find_x_given_inverse_relationship_l1524_152468


namespace NUMINAMATH_GPT_find_g_neg_one_l1524_152419

theorem find_g_neg_one (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 1 / 2 → g x + g ((x + 2) / (2 - 4 * x)) = 3 * x) : 
  g (-1) = - 3 / 2 := 
sorry

end NUMINAMATH_GPT_find_g_neg_one_l1524_152419


namespace NUMINAMATH_GPT_value_of_x2_minus_y2_l1524_152480

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the conditions
def condition1 : Prop := (x + y) / 2 = 5
def condition2 : Prop := (x - y) / 2 = 2

-- State the theorem to prove
theorem value_of_x2_minus_y2 (h1 : condition1 x y) (h2 : condition2 x y) : x^2 - y^2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x2_minus_y2_l1524_152480


namespace NUMINAMATH_GPT_neg_p_l1524_152456

open Nat -- Opening natural number namespace

-- Definition of the proposition p
def p := ∃ (m : ℕ), ∃ (k : ℕ), k * k = m * m + 1

-- Theorem statement for the negation of proposition p
theorem neg_p : ¬p ↔ ∀ (m : ℕ), ¬ ∃ (k : ℕ), k * k = m * m + 1 :=
by {
  -- Provide the proof here
  sorry
}

end NUMINAMATH_GPT_neg_p_l1524_152456


namespace NUMINAMATH_GPT_speed_of_stream_l1524_152412

theorem speed_of_stream (D v : ℝ) (h1 : ∀ D, D / (54 - v) = 2 * (D / (54 + v))) : v = 18 := 
sorry

end NUMINAMATH_GPT_speed_of_stream_l1524_152412


namespace NUMINAMATH_GPT_right_triangle_leg_length_l1524_152437

theorem right_triangle_leg_length (a c b : ℝ) (h : a = 4) (h₁ : c = 5) (h₂ : a^2 + b^2 = c^2) : b = 3 := 
by
  -- by is used for the proof, which we are skipping using sorry.
  sorry

end NUMINAMATH_GPT_right_triangle_leg_length_l1524_152437


namespace NUMINAMATH_GPT_johns_initial_bench_press_weight_l1524_152493

noncomputable def initialBenchPressWeight (currentWeight: ℝ) (injuryPercentage: ℝ) (trainingFactor: ℝ) :=
  (currentWeight / (injuryPercentage / 100 * trainingFactor))

theorem johns_initial_bench_press_weight:
  (initialBenchPressWeight 300 80 3) = 500 :=
by
  sorry

end NUMINAMATH_GPT_johns_initial_bench_press_weight_l1524_152493


namespace NUMINAMATH_GPT_evaluate_expression_l1524_152462

theorem evaluate_expression : 
  3 * (-3)^4 + 2 * (-3)^3 + (-3)^2 + 3^2 + 2 * 3^3 + 3 * 3^4 = 504 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1524_152462


namespace NUMINAMATH_GPT_sine_difference_l1524_152450

noncomputable def perpendicular_vectors (θ : ℝ) : Prop :=
  let a := (Real.cos θ, -Real.sqrt 3)
  let b := (1, 1 + Real.sin θ)
  a.1 * b.1 + a.2 * b.2 = 0

theorem sine_difference (θ : ℝ) (h : perpendicular_vectors θ) : Real.sin (Real.pi / 6 - θ) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sine_difference_l1524_152450


namespace NUMINAMATH_GPT_right_triangle_set_l1524_152472

def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem right_triangle_set :
  (is_right_triangle 3 4 2 = false) ∧
  (is_right_triangle 5 12 15 = false) ∧
  (is_right_triangle 8 15 17 = true) ∧
  (is_right_triangle (3^2) (4^2) (5^2) = false) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_set_l1524_152472


namespace NUMINAMATH_GPT_value_of_x_l1524_152420

theorem value_of_x (x : ℝ) (h1 : |x| - 1 = 0) (h2 : x - 1 ≠ 0) : x = -1 := 
sorry

end NUMINAMATH_GPT_value_of_x_l1524_152420


namespace NUMINAMATH_GPT_find_m_correct_l1524_152404

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

end NUMINAMATH_GPT_find_m_correct_l1524_152404


namespace NUMINAMATH_GPT_transformation_of_95_squared_l1524_152400

theorem transformation_of_95_squared :
  (9.5 : ℝ) ^ 2 = (10 : ℝ) ^ 2 - 2 * (10 : ℝ) * (0.5 : ℝ) + (0.5 : ℝ) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_transformation_of_95_squared_l1524_152400


namespace NUMINAMATH_GPT_initial_students_proof_l1524_152405

def initial_students (e : ℝ) (transferred : ℝ) (left : ℝ) : ℝ :=
  e + transferred + left

theorem initial_students_proof : initial_students 28 10 4 = 42 :=
  by
    -- This is where the proof would go, but we use 'sorry' to skip it.
    sorry

end NUMINAMATH_GPT_initial_students_proof_l1524_152405


namespace NUMINAMATH_GPT_pain_subsided_days_l1524_152426

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

end NUMINAMATH_GPT_pain_subsided_days_l1524_152426


namespace NUMINAMATH_GPT_find_natural_numbers_l1524_152460

noncomputable def valid_n (n : ℕ) : Prop :=
  2 ^ n % 7 = n ^ 2 % 7

theorem find_natural_numbers :
  {n : ℕ | valid_n n} = {n : ℕ | n % 21 = 2 ∨ n % 21 = 4 ∨ n % 21 = 5 ∨ n % 21 = 6 ∨ n % 21 = 10 ∨ n % 21 = 15} :=
sorry

end NUMINAMATH_GPT_find_natural_numbers_l1524_152460


namespace NUMINAMATH_GPT_total_earnings_correct_l1524_152434

noncomputable def total_earnings : ℝ :=
  let earnings1 := 12 * (2 + 15 / 60)
  let earnings2 := 15 * (1 + 40 / 60)
  let earnings3 := 10 * (3 + 10 / 60)
  earnings1 + earnings2 + earnings3

theorem total_earnings_correct : total_earnings = 83.75 := by
  sorry

end NUMINAMATH_GPT_total_earnings_correct_l1524_152434


namespace NUMINAMATH_GPT_small_planters_needed_l1524_152445

-- This states the conditions for the problem
def Oshea_seeds := 200
def large_planters := 4
def large_planter_capacity := 20
def small_planter_capacity := 4
def remaining_seeds := Oshea_seeds - (large_planters * large_planter_capacity) 

-- The target we aim to prove: the number of small planters required
theorem small_planters_needed :
  remaining_seeds / small_planter_capacity = 30 := by
  sorry

end NUMINAMATH_GPT_small_planters_needed_l1524_152445


namespace NUMINAMATH_GPT_find_initial_balance_l1524_152432

-- Define the initial balance (X)
def initial_balance (X : ℝ) := 
  ∃ (X : ℝ), (X / 2 + 30 + 50 - 20 = 160)

theorem find_initial_balance (X : ℝ) (h : initial_balance X) : 
  X = 200 :=
sorry

end NUMINAMATH_GPT_find_initial_balance_l1524_152432


namespace NUMINAMATH_GPT_remainder_8_digit_non_decreasing_integers_mod_1000_l1524_152464

noncomputable def M : ℕ :=
  Nat.choose 17 8

theorem remainder_8_digit_non_decreasing_integers_mod_1000 :
  M % 1000 = 310 :=
by
  sorry

end NUMINAMATH_GPT_remainder_8_digit_non_decreasing_integers_mod_1000_l1524_152464


namespace NUMINAMATH_GPT_max_rectangle_area_l1524_152440

noncomputable def curve_parametric_equation (θ : ℝ) :
    ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

theorem max_rectangle_area :
  ∃ (θ : ℝ), (θ ∈ Set.Icc 0 (2 * Real.pi)) ∧
  ∀ (x y : ℝ), (x, y) = curve_parametric_equation θ →
  |(1 + 2 * Real.cos θ) * (1 + 2 * Real.sin θ)| = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_rectangle_area_l1524_152440


namespace NUMINAMATH_GPT_time_for_B_alone_l1524_152421

theorem time_for_B_alone (r_A r_B r_C : ℚ)
  (h1 : r_A + r_B = 1/3)
  (h2 : r_B + r_C = 2/7)
  (h3 : r_A + r_C = 1/4) :
  1/r_B = 168/31 :=
by
  sorry

end NUMINAMATH_GPT_time_for_B_alone_l1524_152421


namespace NUMINAMATH_GPT_volume_water_needed_l1524_152401

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

end NUMINAMATH_GPT_volume_water_needed_l1524_152401


namespace NUMINAMATH_GPT_socks_total_is_51_l1524_152411

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

end NUMINAMATH_GPT_socks_total_is_51_l1524_152411


namespace NUMINAMATH_GPT_overall_average_marks_l1524_152471

theorem overall_average_marks (n P : ℕ) (P_avg F_avg : ℕ) (H_n : n = 120) (H_P : P = 100) (H_P_avg : P_avg = 39) (H_F_avg : F_avg = 15) :
  (P_avg * P + F_avg * (n - P)) / n = 35 := 
by
  sorry

end NUMINAMATH_GPT_overall_average_marks_l1524_152471


namespace NUMINAMATH_GPT_money_combination_l1524_152439

variable (Raquel Tom Nataly Sam : ℝ)

-- Given Conditions 
def condition1 : Prop := Tom = (1 / 4) * Nataly
def condition2 : Prop := Nataly = 3 * Raquel
def condition3 : Prop := Sam = 2 * Nataly
def condition4 : Prop := Nataly = (5 / 3) * Sam
def condition5 : Prop := Raquel = 40

-- Proving this combined total
def combined_total : Prop := Tom + Raquel + Nataly + Sam = 262

theorem money_combination (h1: condition1 Tom Nataly) 
                          (h2: condition2 Nataly Raquel) 
                          (h3: condition3 Sam Nataly) 
                          (h4: condition4 Nataly Sam) 
                          (h5: condition5 Raquel) 
                          : combined_total Tom Raquel Nataly Sam :=
sorry

end NUMINAMATH_GPT_money_combination_l1524_152439


namespace NUMINAMATH_GPT_find_aroon_pin_l1524_152467

theorem find_aroon_pin (a b : ℕ) (PIN : ℕ) 
  (h0 : 0 ≤ a ∧ a ≤ 9)
  (h1 : 0 ≤ b ∧ b < 1000)
  (h2 : PIN = 1000 * a + b)
  (h3 : 10 * b + a = 3 * PIN - 6) : 
  PIN = 2856 := 
sorry

end NUMINAMATH_GPT_find_aroon_pin_l1524_152467


namespace NUMINAMATH_GPT_find_x_from_angles_l1524_152479

theorem find_x_from_angles : ∀ (x : ℝ), (6 * x + 3 * x + 2 * x + x = 360) → x = 30 := by
  sorry

end NUMINAMATH_GPT_find_x_from_angles_l1524_152479


namespace NUMINAMATH_GPT_dealership_sedan_sales_l1524_152475

-- Definitions based on conditions:
def sports_cars_ratio : ℕ := 3
def sedans_ratio : ℕ := 5
def anticipated_sports_cars : ℕ := 36

-- Proof problem statement
theorem dealership_sedan_sales :
    (anticipated_sports_cars * sedans_ratio) / sports_cars_ratio = 60 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_dealership_sedan_sales_l1524_152475


namespace NUMINAMATH_GPT_instantaneous_velocity_at_2_l1524_152478

def displacement (t : ℝ) : ℝ := 2 * t^3

theorem instantaneous_velocity_at_2 :
  let velocity := deriv displacement
  velocity 2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_2_l1524_152478


namespace NUMINAMATH_GPT_nine_sided_polygon_diagonals_l1524_152484

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end NUMINAMATH_GPT_nine_sided_polygon_diagonals_l1524_152484


namespace NUMINAMATH_GPT_find_overtime_hours_l1524_152443

theorem find_overtime_hours
  (pay_rate_ordinary : ℝ := 0.60)
  (pay_rate_overtime : ℝ := 0.90)
  (total_pay : ℝ := 32.40)
  (total_hours : ℕ := 50) :
  ∃ y : ℕ, pay_rate_ordinary * (total_hours - y) + pay_rate_overtime * y = total_pay ∧ y = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_overtime_hours_l1524_152443


namespace NUMINAMATH_GPT_valid_selling_price_l1524_152492

-- Define the initial conditions
def cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def sales_increase_per_dollar_decrease : ℝ := 4
def max_profit : ℝ := 13600
def min_selling_price : ℝ := 150

-- Define x as the price reduction per item
variable (x : ℝ)

-- Define the function relationship of the daily sales volume y with respect to x
def sales_volume (x : ℝ) := 100 + 4 * x

-- Define the selling price based on the price reduction
def selling_price (x : ℝ) := 200 - x

-- Calculate the profit based on the selling price and sales volume
def profit (x : ℝ) := (selling_price x - cost_price) * (sales_volume x)

-- Lean theorem statement to prove the given conditions lead to the valid selling price
theorem valid_selling_price (x : ℝ) 
  (h1 : profit x = 13600)
  (h2 : selling_price x ≥ 150) : 
  selling_price x = 185 :=
sorry

end NUMINAMATH_GPT_valid_selling_price_l1524_152492


namespace NUMINAMATH_GPT_duty_pairing_impossible_l1524_152414

theorem duty_pairing_impossible :
  ∀ (m n : ℕ), 29 * m + 32 * n ≠ 29 * 32 := 
by 
  sorry

end NUMINAMATH_GPT_duty_pairing_impossible_l1524_152414


namespace NUMINAMATH_GPT_combination_of_10_choose_3_l1524_152458

theorem combination_of_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_GPT_combination_of_10_choose_3_l1524_152458


namespace NUMINAMATH_GPT_n_squared_plus_inverse_squared_plus_four_eq_102_l1524_152453

theorem n_squared_plus_inverse_squared_plus_four_eq_102 (n : ℝ) (h : n + 1 / n = 10) :
    n^2 + 1 / n^2 + 4 = 102 :=
by sorry

end NUMINAMATH_GPT_n_squared_plus_inverse_squared_plus_four_eq_102_l1524_152453


namespace NUMINAMATH_GPT_machine_shirt_rate_l1524_152483

theorem machine_shirt_rate (S : ℕ) 
  (worked_yesterday : ℕ) (worked_today : ℕ) (shirts_today : ℕ) 
  (h1 : worked_yesterday = 5)
  (h2 : worked_today = 12)
  (h3 : shirts_today = 72)
  (h4 : worked_today * S = shirts_today) : 
  S = 6 := 
by 
  sorry

end NUMINAMATH_GPT_machine_shirt_rate_l1524_152483


namespace NUMINAMATH_GPT_sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5_l1524_152470

theorem sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5 : Real.sqrt 3 - Real.sqrt 2 > Real.sqrt 6 - Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5_l1524_152470


namespace NUMINAMATH_GPT_find_number_l1524_152489

theorem find_number (n p q : ℝ) (h1 : n / p = 6) (h2 : n / q = 15) (h3 : p - q = 0.3) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1524_152489


namespace NUMINAMATH_GPT_more_campers_afternoon_than_morning_l1524_152463

def campers_morning : ℕ := 52
def campers_afternoon : ℕ := 61

theorem more_campers_afternoon_than_morning : campers_afternoon - campers_morning = 9 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_more_campers_afternoon_than_morning_l1524_152463


namespace NUMINAMATH_GPT_correct_operation_l1524_152485

theorem correct_operation (a b : ℝ) : 
  (2 * a^2 + a^2 = 3 * a^2) ∧ 
  (a^3 * a^3 ≠ 2 * a^3) ∧ 
  (a^9 / a^3 ≠ a^3) ∧ 
  (¬(7 * a * b - 5 * a = 2)) :=
by 
  sorry

end NUMINAMATH_GPT_correct_operation_l1524_152485


namespace NUMINAMATH_GPT_summer_has_150_degrees_l1524_152459

-- Define the condition that Summer has five more degrees than Jolly,
-- and the combined number of degrees they both have is 295.
theorem summer_has_150_degrees (S J : ℕ) (h1 : S = J + 5) (h2 : S + J = 295) : S = 150 :=
by sorry

end NUMINAMATH_GPT_summer_has_150_degrees_l1524_152459


namespace NUMINAMATH_GPT_no_such_a_exists_l1524_152403

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {1, 5*a - 5, -1/2*a^2 + 3/2*a + 4, a^3 + a^2 + 3*a + 7}

theorem no_such_a_exists (a : ℝ) : ¬(A a ∩ B a = {2, 5}) :=
by
  sorry

end NUMINAMATH_GPT_no_such_a_exists_l1524_152403


namespace NUMINAMATH_GPT_zeros_of_f_l1524_152431

noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem zeros_of_f (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (∃ x, a < x ∧ x < b ∧ f a b c x = 0) ∧ (∃ y, b < y ∧ y < c ∧ f a b c y = 0) :=
by
  sorry

end NUMINAMATH_GPT_zeros_of_f_l1524_152431


namespace NUMINAMATH_GPT_arithmetic_mean_of_reciprocals_of_first_four_primes_l1524_152481

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_reciprocals_of_first_four_primes_l1524_152481


namespace NUMINAMATH_GPT_hypotenuse_not_5_cm_l1524_152415

theorem hypotenuse_not_5_cm (a b c : ℝ) (h₀ : a + b = 8) (h₁ : a^2 + b^2 = c^2) : c ≠ 5 := by
  sorry

end NUMINAMATH_GPT_hypotenuse_not_5_cm_l1524_152415


namespace NUMINAMATH_GPT_solution_set_l1524_152416

noncomputable def domain := Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)
def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x, x ∈ domain → x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)
axiom f_odd : ∀ x, f x + f (-x) = 0
def f' : ℝ → ℝ := sorry
axiom derivative_condition : ∀ x, 0 < x ∧ x < Real.pi / 2 → f' x * Real.cos x + f x * Real.sin x < 0

theorem solution_set :
  {x | f x < Real.sqrt 2 * f (Real.pi / 4) * Real.cos x} = {x | Real.pi / 4 < x ∧ x < Real.pi / 2} :=
sorry

end NUMINAMATH_GPT_solution_set_l1524_152416


namespace NUMINAMATH_GPT_pos_diff_of_solutions_abs_eq_20_l1524_152407

theorem pos_diff_of_solutions_abs_eq_20 : ∀ (x1 x2 : ℝ), (|x1 + 5| = 20 ∧ |x2 + 5| = 20) → x1 - x2 = 40 :=
  by
    intros x1 x2 h
    sorry

end NUMINAMATH_GPT_pos_diff_of_solutions_abs_eq_20_l1524_152407


namespace NUMINAMATH_GPT_price_reduction_equation_l1524_152488

theorem price_reduction_equation (x : ℝ) :
  63800 * (1 - x)^2 = 3900 :=
sorry

end NUMINAMATH_GPT_price_reduction_equation_l1524_152488


namespace NUMINAMATH_GPT_max_expression_value_l1524_152410

theorem max_expression_value (a b c d e f g h k : ℤ)
  (ha : (a = 1 ∨ a = -1)) (hb : (b = 1 ∨ b = -1))
  (hc : (c = 1 ∨ c = -1)) (hd : (d = 1 ∨ d = -1))
  (he : (e = 1 ∨ e = -1)) (hf : (f = 1 ∨ f = -1))
  (hg : (g = 1 ∨ g = -1)) (hh : (h = 1 ∨ h = -1))
  (hk : (k = 1 ∨ k = -1)) :
  a * e * k - a * f * h + b * f * g - b * d * k + c * d * h - c * e * g ≤ 4 := sorry

end NUMINAMATH_GPT_max_expression_value_l1524_152410


namespace NUMINAMATH_GPT_strawberry_pancakes_l1524_152499

theorem strawberry_pancakes (total blueberry banana chocolate : ℕ) (h_total : total = 150) (h_blueberry : blueberry = 45) (h_banana : banana = 60) (h_chocolate : chocolate = 25) :
  total - (blueberry + banana + chocolate) = 20 :=
by
  sorry

end NUMINAMATH_GPT_strawberry_pancakes_l1524_152499


namespace NUMINAMATH_GPT_intersection_eq_l1524_152429

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≤ 2}

theorem intersection_eq : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1524_152429


namespace NUMINAMATH_GPT_total_lives_l1524_152482

noncomputable def C : ℝ := 9.5
noncomputable def D : ℝ := C - 3.25
noncomputable def M : ℝ := D + 7.75
noncomputable def E : ℝ := 2 * C - 5.5
noncomputable def F : ℝ := 2/3 * E

theorem total_lives : C + D + M + E + F = 52.25 :=
by
  sorry

end NUMINAMATH_GPT_total_lives_l1524_152482
