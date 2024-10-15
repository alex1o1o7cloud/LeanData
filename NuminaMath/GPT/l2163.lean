import Mathlib

namespace NUMINAMATH_GPT_unique_b_for_quadratic_l2163_216371

theorem unique_b_for_quadratic (c : ℝ) (h_c : c ≠ 0) : (∃! b : ℝ, b > 0 ∧ (2*b + 2/b)^2 - 4*c = 0) → c = 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_b_for_quadratic_l2163_216371


namespace NUMINAMATH_GPT_flowers_per_row_correct_l2163_216350

/-- Definition for the number of each type of flower. -/
def num_yellow_flowers : ℕ := 12
def num_green_flowers : ℕ := 2 * num_yellow_flowers -- Given that green flowers are twice the yellow flowers.
def num_red_flowers : ℕ := 42

/-- Total number of flowers. -/
def total_flowers : ℕ := num_yellow_flowers + num_green_flowers + num_red_flowers

/-- Number of rows in the garden. -/
def num_rows : ℕ := 6

/-- The number of flowers per row Wilma's garden has. -/
def flowers_per_row : ℕ := total_flowers / num_rows

/-- Proof statement: flowers per row should be 13. -/
theorem flowers_per_row_correct : flowers_per_row = 13 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_flowers_per_row_correct_l2163_216350


namespace NUMINAMATH_GPT_find_ratio_l2163_216369

def given_conditions (a b c x y z : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 25 ∧ x^2 + y^2 + z^2 = 36 ∧ a * x + b * y + c * z = 30

theorem find_ratio (a b c x y z : ℝ)
  (h : given_conditions a b c x y z) :
  (a + b + c) / (x + y + z) = 5 / 6 :=
sorry

end NUMINAMATH_GPT_find_ratio_l2163_216369


namespace NUMINAMATH_GPT_correct_divisor_l2163_216311

noncomputable def dividend := 12 * 35

theorem correct_divisor (x : ℕ) : (x * 20 = dividend) → x = 21 :=
sorry

end NUMINAMATH_GPT_correct_divisor_l2163_216311


namespace NUMINAMATH_GPT_min_ge_n_l2163_216319

theorem min_ge_n (x y z n : ℕ) (h : x^n + y^n = z^n) : min x y ≥ n :=
sorry

end NUMINAMATH_GPT_min_ge_n_l2163_216319


namespace NUMINAMATH_GPT_triangle_side_length_l2163_216325

theorem triangle_side_length 
  (r : ℝ)                    -- radius of the inscribed circle
  (h_cos_ABC : ℝ)            -- cosine of angle ABC
  (h_midline : Bool)         -- the circle touches the midline parallel to AC
  (h_r : r = 1)              -- given radius is 1
  (h_cos : h_cos_ABC = 0.8)  -- given cos(ABC) = 0.8
  (h_touch : h_midline = true)  -- given that circle touches the midline
  : AC = 3 := 
sorry

end NUMINAMATH_GPT_triangle_side_length_l2163_216325


namespace NUMINAMATH_GPT_hyperbola_focus_coordinates_l2163_216328

theorem hyperbola_focus_coordinates : 
  ∃ (x y : ℝ), -2 * x^2 + 3 * y^2 + 8 * x - 18 * y - 8 = 0 ∧ (x, y) = (2, 7.5) :=
sorry

end NUMINAMATH_GPT_hyperbola_focus_coordinates_l2163_216328


namespace NUMINAMATH_GPT_range_of_m_l2163_216386

noncomputable def f (x m : ℝ) : ℝ :=
  x^2 - 2 * m * x + m + 2

theorem range_of_m
  (m : ℝ)
  (h1 : ∃ a b : ℝ, f a m = 0 ∧ f b m = 0 ∧ a ≠ b)
  (h2 : ∀ x : ℝ, x ≥ 1 → 2*x - 2*m ≥ 0) :
  m < -1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2163_216386


namespace NUMINAMATH_GPT_call_cost_inequalities_min_call_cost_correct_l2163_216366

noncomputable def call_cost_before (x : ℝ) : ℝ :=
  if x ≤ 3 then 0.2 else 0.4

noncomputable def call_cost_after (x : ℝ) : ℝ :=
  if x ≤ 3 then 0.2
  else if x ≤ 4 then 0.2 + 0.1 * (x - 3)
  else 0.3 + 0.1 * (x - 4)

theorem call_cost_inequalities : 
  (call_cost_before 4 = 0.4 ∧ call_cost_after 4 = 0.3) ∧
  (call_cost_before 4.3 = 0.4 ∧ call_cost_after 4.3 = 0.4) ∧
  (call_cost_before 5.8 = 0.4 ∧ call_cost_after 5.8 = 0.5) ∧
  (∀ x, (0 < x ∧ x ≤ 3) ∨ x > 4 → call_cost_before x ≤ call_cost_after x) :=
by
  sorry

noncomputable def min_call_cost_plan (m : ℝ) (n : ℕ) : ℝ :=
  if 3 * n - 1 < m ∧ m ≤ 3 * n then 0.2 * n
  else if 3 * n < m ∧ m ≤ 3 * n + 1 then 0.2 * n + 0.1
  else if 3 * n + 1 < m ∧ m ≤ 3 * n + 2 then 0.2 * n + 0.2
  else 0.0  -- Fallback, though not necessary as per the conditions

theorem min_call_cost_correct (m : ℝ) (n : ℕ) (h : m > 5) :
  (3 * n - 1 < m ∧ m ≤ 3 * n → min_call_cost_plan m n = 0.2 * n) ∧
  (3 * n < m ∧ m ≤ 3 * n + 1 → min_call_cost_plan m n = 0.2 * n + 0.1) ∧
  (3 * n + 1 < m ∧ m ≤ 3 * n + 2 → min_call_cost_plan m n = 0.2 * n + 0.2) :=
by
  sorry

end NUMINAMATH_GPT_call_cost_inequalities_min_call_cost_correct_l2163_216366


namespace NUMINAMATH_GPT_initial_boys_count_l2163_216323

theorem initial_boys_count (B : ℕ) (boys girls : ℕ)
  (h1 : boys = 3 * B)                             -- The ratio of boys to girls is 3:4
  (h2 : girls = 4 * B)                            -- The ratio of boys to girls is 3:4
  (h3 : boys - 10 = 4 * (girls - 20))             -- The final ratio after transfer is 4:5
  : boys = 90 :=                                  -- Prove initial boys count was 90
by 
  sorry

end NUMINAMATH_GPT_initial_boys_count_l2163_216323


namespace NUMINAMATH_GPT_contrapositive_example_l2163_216365

variable (a b : ℝ)

theorem contrapositive_example
  (h₁ : a > 0)
  (h₃ : a + b < 0) :
  b < 0 := 
sorry

end NUMINAMATH_GPT_contrapositive_example_l2163_216365


namespace NUMINAMATH_GPT_calculator_to_protractors_l2163_216395

def calculator_to_rulers (c: ℕ) : ℕ := 100 * c
def rulers_to_compasses (r: ℕ) : ℕ := (r * 30) / 10
def compasses_to_protractors (p: ℕ) : ℕ := (p * 50) / 25

theorem calculator_to_protractors (c: ℕ) : compasses_to_protractors (rulers_to_compasses (calculator_to_rulers c)) = 600 * c :=
by
  sorry

end NUMINAMATH_GPT_calculator_to_protractors_l2163_216395


namespace NUMINAMATH_GPT_dress_designs_count_l2163_216334

inductive Color
| red | green | blue | yellow

inductive Pattern
| stripes | polka_dots | floral | geometric | plain

def patterns_for_color (c : Color) : List Pattern :=
  match c with
  | Color.red    => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.geometric, Pattern.plain]
  | Color.green  => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.plain]
  | Color.blue   => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.geometric, Pattern.plain]
  | Color.yellow => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.plain]

noncomputable def number_of_dress_designs : ℕ :=
  (patterns_for_color Color.red).length +
  (patterns_for_color Color.green).length +
  (patterns_for_color Color.blue).length +
  (patterns_for_color Color.yellow).length

theorem dress_designs_count : number_of_dress_designs = 18 :=
  by
  sorry

end NUMINAMATH_GPT_dress_designs_count_l2163_216334


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l2163_216374

-- Problem for Equation (1)
theorem solve_equation1 (x : ℝ) : x * (x - 6) = 2 * (x - 8) → x = 4 := by
  sorry

-- Problem for Equation (2)
theorem solve_equation2 (x : ℝ) : (2 * x - 1)^2 + 3 * (2 * x - 1) + 2 = 0 → x = 0 ∨ x = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l2163_216374


namespace NUMINAMATH_GPT_ducks_problem_l2163_216382

theorem ducks_problem :
  ∃ (adelaide ephraim kolton : ℕ),
    adelaide = 30 ∧
    adelaide = 2 * ephraim ∧
    ephraim + 45 = kolton ∧
    (adelaide + ephraim + kolton) % 9 = 0 ∧
    1 ≤ adelaide ∧
    1 ≤ ephraim ∧
    1 ≤ kolton ∧
    adelaide + ephraim + kolton = 108 ∧
    (adelaide + ephraim + kolton) / 3 = 36 :=
by
  sorry

end NUMINAMATH_GPT_ducks_problem_l2163_216382


namespace NUMINAMATH_GPT_tangent_line_at_point_l2163_216303

theorem tangent_line_at_point :
  ∀ (x y : ℝ) (h : y = x^3 - 2 * x + 1),
    ∃ (m b : ℝ), (1, 0) = (x, y) → (m = 1) ∧ (b = -1) ∧ (∀ (z : ℝ), z = m * x + b) := sorry

end NUMINAMATH_GPT_tangent_line_at_point_l2163_216303


namespace NUMINAMATH_GPT_closest_point_on_line_l2163_216385

theorem closest_point_on_line 
  (t : ℚ)
  (x y z : ℚ)
  (x_eq : x = 3 + t)
  (y_eq : y = 2 - 3 * t)
  (z_eq : z = -1 + 2 * t)
  (x_ortho_eq : (1 + t) = 0)
  (y_ortho_eq : (3 - 3 * t) = 0)
  (z_ortho_eq : (-3 + 2 * t) = 0) :
  (45/14, 16/14, -1/7) = (x, y, z) := by
  sorry

end NUMINAMATH_GPT_closest_point_on_line_l2163_216385


namespace NUMINAMATH_GPT_sequence_an_l2163_216358

theorem sequence_an (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * (a n - 1)) : a 2 = 4 := 
by
  sorry

end NUMINAMATH_GPT_sequence_an_l2163_216358


namespace NUMINAMATH_GPT_parabola_example_l2163_216344

theorem parabola_example (p : ℝ) (hp : p > 0)
    (h_intersect : ∀ x y : ℝ, y = x - p / 2 ∧ y^2 = 2 * p * x → ((x - p / 2)^2 = 2 * p * x))
    (h_AB : ∀ A B : ℝ × ℝ, A.2 = A.1 - p / 2 ∧ B.2 = B.1 - p / 2 ∧ |A.1 - B.1| = 8) :
    p = 2 := 
sorry

end NUMINAMATH_GPT_parabola_example_l2163_216344


namespace NUMINAMATH_GPT_operation_on_b_l2163_216304

theorem operation_on_b (t b0 b1 : ℝ) (h : t * b1^4 = 16 * t * b0^4) : b1 = 2 * b0 :=
by
  sorry

end NUMINAMATH_GPT_operation_on_b_l2163_216304


namespace NUMINAMATH_GPT_price_of_peaches_is_2_l2163_216376

noncomputable def price_per_pound_peaches (total_spent: ℝ) (price_per_pound_other: ℝ) (total_weight_peaches: ℝ) (total_weight_apples: ℝ) (total_weight_blueberries: ℝ) : ℝ :=
  (total_spent - (total_weight_apples + total_weight_blueberries) * price_per_pound_other) / total_weight_peaches

theorem price_of_peaches_is_2 
  (total_spent: ℝ := 51)
  (price_per_pound_other: ℝ := 1)
  (num_peach_pies: ℕ := 5)
  (num_apple_pies: ℕ := 4)
  (num_blueberry_pies: ℕ := 3)
  (weight_per_pie: ℝ := 3):
  price_per_pound_peaches total_spent price_per_pound_other 
                          (num_peach_pies * weight_per_pie) 
                          (num_apple_pies * weight_per_pie) 
                          (num_blueberry_pies * weight_per_pie) = 2 := 
by
  sorry

end NUMINAMATH_GPT_price_of_peaches_is_2_l2163_216376


namespace NUMINAMATH_GPT_distance_between_A_and_B_l2163_216331

-- Define speeds, times, and distances as real numbers
def speed_A_to_B := 42.5
def time_travelled := 1.5
def remaining_to_midpoint := 26.0

-- Define the total distance between A and B as a variable
def distance_A_to_B : ℝ := 179.5

-- Prove that the distance between locations A and B is 179.5 kilometers given the conditions
theorem distance_between_A_and_B : (42.5 * 1.5 + 26) * 2 = 179.5 :=
by 
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l2163_216331


namespace NUMINAMATH_GPT_tangent_line_perpendicular_l2163_216326

noncomputable def f (x k : ℝ) : ℝ := x^3 - (k^2 - 1) * x^2 - k^2 + 2

theorem tangent_line_perpendicular (k : ℝ) (b : ℝ) (a : ℝ)
  (h1 : ∀ (x : ℝ), f x k = x^3 - (k^2 - 1) * x^2 - k^2 + 2)
  (h2 : (3 - 2 * (k^2 - 1)) = -1) :
  a = -2 := sorry

end NUMINAMATH_GPT_tangent_line_perpendicular_l2163_216326


namespace NUMINAMATH_GPT_tanker_filling_rate_l2163_216343

theorem tanker_filling_rate :
  let barrels_per_minute := 5
  let liters_per_barrel := 159
  let minutes_per_hour := 60
  let liters_per_cubic_meter := 1000
  (barrels_per_minute * liters_per_barrel * minutes_per_hour) / 
  liters_per_cubic_meter = 47.7 :=
by
  sorry

end NUMINAMATH_GPT_tanker_filling_rate_l2163_216343


namespace NUMINAMATH_GPT_compare_f_l2163_216353

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem compare_f (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x1 + x2 = 0) : 
  f x1 < f x2 :=
by sorry

end NUMINAMATH_GPT_compare_f_l2163_216353


namespace NUMINAMATH_GPT_range_of_f_l2163_216329

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 2

theorem range_of_f : Set.range f = Set.Ici 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l2163_216329


namespace NUMINAMATH_GPT_find_y_l2163_216367

-- Definitions for the given conditions
variable (p y : ℕ) (h : p > 30)  -- Natural numbers, noting p > 30 condition

-- The initial amount of acid in ounces
def initial_acid_amount : ℕ := p * p / 100

-- The amount of acid after adding y ounces of water
def final_acid_amount : ℕ := (p - 15) * (p + y) / 100

-- Lean statement to prove y = 15p/(p-15)
theorem find_y (h1 : p > 30) (h2 : initial_acid_amount p = final_acid_amount p y) :
  y = 15 * p / (p - 15) :=
sorry

end NUMINAMATH_GPT_find_y_l2163_216367


namespace NUMINAMATH_GPT_cube_volume_surface_area_l2163_216357

-- Define volume and surface area conditions
def volume_condition (x : ℝ) (s : ℝ) : Prop := s^3 = 3 * x
def surface_area_condition (x : ℝ) (s : ℝ) : Prop := 6 * s^2 = x

-- The main theorem statement
theorem cube_volume_surface_area (x : ℝ) (s : ℝ) :
  volume_condition x s → surface_area_condition x s → x = 5832 :=
by
  intros h_volume h_area
  sorry

end NUMINAMATH_GPT_cube_volume_surface_area_l2163_216357


namespace NUMINAMATH_GPT_maximum_marks_l2163_216348

theorem maximum_marks (M : ℝ) (h1 : 0.45 * M = 180) : M = 400 := 
by sorry

end NUMINAMATH_GPT_maximum_marks_l2163_216348


namespace NUMINAMATH_GPT_monotonic_sufficient_not_necessary_maximum_l2163_216315

noncomputable def f (x : ℝ) : ℝ := sorry  -- Placeholder for the function f
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop := (∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x)
def has_max_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∃ M, ∀ x, a ≤ x → x ≤ b → f x ≤ M

theorem monotonic_sufficient_not_necessary_maximum : 
  ∀ f : ℝ → ℝ,
  ∀ a b : ℝ,
  a ≤ b →
  monotonic_on f a b → 
  has_max_on f a b :=
sorry  -- Proof is omitted

end NUMINAMATH_GPT_monotonic_sufficient_not_necessary_maximum_l2163_216315


namespace NUMINAMATH_GPT_find_z_percentage_of_1000_l2163_216384

noncomputable def x := (3 / 5) * 4864
noncomputable def y := (2 / 3) * 9720
noncomputable def z := (1 / 4) * 800

theorem find_z_percentage_of_1000 :
  (2 / 3) * x + (1 / 2) * y = z → (z / 1000) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_z_percentage_of_1000_l2163_216384


namespace NUMINAMATH_GPT_Sara_taller_than_Joe_l2163_216356

noncomputable def Roy_height := 36

noncomputable def Joe_height := Roy_height + 3

noncomputable def Sara_height := 45

theorem Sara_taller_than_Joe : Sara_height - Joe_height = 6 :=
by
  sorry

end NUMINAMATH_GPT_Sara_taller_than_Joe_l2163_216356


namespace NUMINAMATH_GPT_polygon_n_sides_l2163_216332

theorem polygon_n_sides (n : ℕ) (h : (n - 2) * 180 - x = 2000) : n = 14 :=
sorry

end NUMINAMATH_GPT_polygon_n_sides_l2163_216332


namespace NUMINAMATH_GPT_no_prime_number_between_30_and_40_mod_9_eq_7_l2163_216333

theorem no_prime_number_between_30_and_40_mod_9_eq_7 : ¬ ∃ n : ℕ, 30 ≤ n ∧ n ≤ 40 ∧ Nat.Prime n ∧ n % 9 = 7 :=
by
  sorry

end NUMINAMATH_GPT_no_prime_number_between_30_and_40_mod_9_eq_7_l2163_216333


namespace NUMINAMATH_GPT_percent_motorists_receive_tickets_l2163_216352

theorem percent_motorists_receive_tickets (n : ℕ) (h1 : (25 : ℕ) % 100 = 25) (h2 : (20 : ℕ) % 100 = 20) :
  (75 * n / 100) = (20 * n / 100) :=
by
  sorry

end NUMINAMATH_GPT_percent_motorists_receive_tickets_l2163_216352


namespace NUMINAMATH_GPT_find_m_of_quadratic_root_l2163_216346

theorem find_m_of_quadratic_root
  (m : ℤ) 
  (h : ∃ x : ℤ, x^2 - (m+3)*x + m + 2 = 0 ∧ x = 81) : 
  m = 79 :=
by
  sorry

end NUMINAMATH_GPT_find_m_of_quadratic_root_l2163_216346


namespace NUMINAMATH_GPT_longest_line_segment_l2163_216361

theorem longest_line_segment (total_length_cm : ℕ) (h : total_length_cm = 3000) :
  ∃ n : ℕ, 2 * (n * (n + 1) / 2) ≤ total_length_cm ∧ n = 54 :=
by
  use 54
  sorry

end NUMINAMATH_GPT_longest_line_segment_l2163_216361


namespace NUMINAMATH_GPT_canoe_upstream_speed_l2163_216340

theorem canoe_upstream_speed (C : ℝ) (stream_speed downstream_speed : ℝ) 
  (h_stream : stream_speed = 2) (h_downstream : downstream_speed = 12) 
  (h_equation : C + stream_speed = downstream_speed) :
  C - stream_speed = 8 := 
by 
  sorry

end NUMINAMATH_GPT_canoe_upstream_speed_l2163_216340


namespace NUMINAMATH_GPT_ratio_of_a_to_b_l2163_216379

theorem ratio_of_a_to_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
    (h_x : x = 1.25 * a) (h_m : m = 0.40 * b) (h_ratio : m / x = 0.4) 
    : (a / b) = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_of_a_to_b_l2163_216379


namespace NUMINAMATH_GPT_jessica_games_attended_l2163_216305

def total_games : ℕ := 6
def games_missed_by_jessica : ℕ := 4

theorem jessica_games_attended : total_games - games_missed_by_jessica = 2 := by
  sorry

end NUMINAMATH_GPT_jessica_games_attended_l2163_216305


namespace NUMINAMATH_GPT_proof_problem_l2163_216300

theorem proof_problem (k m : ℕ) (hk_pos : 0 < k) (hm_pos : 0 < m) (hkm : k > m)
  (hdiv : (k * m * (k ^ 2 - m ^ 2)) ∣ (k ^ 3 - m ^ 3)) :
  (k - m) ^ 3 > 3 * k * m :=
sorry

end NUMINAMATH_GPT_proof_problem_l2163_216300


namespace NUMINAMATH_GPT_nonneg_integer_solutions_otimes_l2163_216373

noncomputable def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

theorem nonneg_integer_solutions_otimes :
  {x : ℕ | otimes 2 x ≥ 3} = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_nonneg_integer_solutions_otimes_l2163_216373


namespace NUMINAMATH_GPT_count_triangles_in_figure_l2163_216380

noncomputable def triangles_in_figure : ℕ := 53

theorem count_triangles_in_figure : triangles_in_figure = 53 := 
by sorry

end NUMINAMATH_GPT_count_triangles_in_figure_l2163_216380


namespace NUMINAMATH_GPT_angle_addition_l2163_216359

open Real

theorem angle_addition (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : tan α = 1 / 3) (h₄ : cos β = 3 / 5) : α + 3 * β = 3 * π / 4 :=
by
  sorry

end NUMINAMATH_GPT_angle_addition_l2163_216359


namespace NUMINAMATH_GPT_sum_of_divisors_143_l2163_216349

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end NUMINAMATH_GPT_sum_of_divisors_143_l2163_216349


namespace NUMINAMATH_GPT_least_element_of_S_is_4_l2163_216338

theorem least_element_of_S_is_4 :
  ∃ S : Finset ℕ, S.card = 7 ∧ (S ⊆ Finset.range 16) ∧
  (∀ {a b : ℕ}, a ∈ S → b ∈ S → a < b → ¬ (b % a = 0)) ∧
  (∀ T : Finset ℕ, T.card = 7 → (T ⊆ Finset.range 16) →
  (∀ {a b : ℕ}, a ∈ T → b ∈ T → a < b → ¬ (b % a = 0)) →
  ∃ x : ℕ, x ∈ T ∧ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_least_element_of_S_is_4_l2163_216338


namespace NUMINAMATH_GPT_area_ratio_is_four_l2163_216389

-- Definitions based on the given conditions
variables (k a b c d : ℝ)
variables (ka kb kc kd : ℝ)

-- Equations from the conditions
def eq1 : a = k * ka := sorry
def eq2 : b = k * kb := sorry
def eq3 : c = k * kc := sorry
def eq4 : d = k * kd := sorry

-- Ratios provided in the problem
def ratio1 : ka / kc = 2 / 5 := sorry
def ratio2 : kb / kd = 2 / 5 := sorry

-- The theorem to prove the ratio of areas is 4:1
theorem area_ratio_is_four : (k * ka * k * kb) / (k * kc * k * kd) = 4 :=
by sorry

end NUMINAMATH_GPT_area_ratio_is_four_l2163_216389


namespace NUMINAMATH_GPT_double_angle_value_l2163_216320

theorem double_angle_value : 2 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_double_angle_value_l2163_216320


namespace NUMINAMATH_GPT_dealers_profit_percentage_l2163_216396

theorem dealers_profit_percentage 
  (articles_purchased : ℕ)
  (total_cost_price : ℝ)
  (articles_sold : ℕ)
  (total_selling_price : ℝ)
  (CP_per_article : ℝ := total_cost_price / articles_purchased)
  (SP_per_article : ℝ := total_selling_price / articles_sold)
  (profit_per_article : ℝ := SP_per_article - CP_per_article)
  (profit_percentage : ℝ := (profit_per_article / CP_per_article) * 100) :
  articles_purchased = 15 →
  total_cost_price = 25 →
  articles_sold = 12 →
  total_selling_price = 32 →
  profit_percentage = 60 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_dealers_profit_percentage_l2163_216396


namespace NUMINAMATH_GPT_jesse_money_left_after_mall_l2163_216316

theorem jesse_money_left_after_mall :
  ∀ (initial_amount novel_cost lunch_cost total_spent remaining_amount : ℕ),
    initial_amount = 50 →
    novel_cost = 7 →
    lunch_cost = 2 * novel_cost →
    total_spent = novel_cost + lunch_cost →
    remaining_amount = initial_amount - total_spent →
    remaining_amount = 29 :=
by
  intros initial_amount novel_cost lunch_cost total_spent remaining_amount
  sorry

end NUMINAMATH_GPT_jesse_money_left_after_mall_l2163_216316


namespace NUMINAMATH_GPT_find_smaller_number_l2163_216330

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end NUMINAMATH_GPT_find_smaller_number_l2163_216330


namespace NUMINAMATH_GPT_balloons_left_l2163_216335

theorem balloons_left (yellow blue pink violet friends : ℕ) (total_balloons remainder : ℕ) 
  (hy : yellow = 20) (hb : blue = 24) (hp : pink = 50) (hv : violet = 102) (hf : friends = 9)
  (ht : total_balloons = yellow + blue + pink + violet) (hr : total_balloons % friends = remainder) : 
  remainder = 7 :=
by
  sorry

end NUMINAMATH_GPT_balloons_left_l2163_216335


namespace NUMINAMATH_GPT_hamburgers_sold_last_week_l2163_216336

theorem hamburgers_sold_last_week (avg_hamburgers_per_day : ℕ) (days_in_week : ℕ) 
    (h_avg : avg_hamburgers_per_day = 9) (h_days : days_in_week = 7) : 
    avg_hamburgers_per_day * days_in_week = 63 :=
by
  -- Avg hamburgers per day times number of days
  sorry

end NUMINAMATH_GPT_hamburgers_sold_last_week_l2163_216336


namespace NUMINAMATH_GPT_chickens_in_coop_l2163_216388

theorem chickens_in_coop (C : ℕ)
  (H1 : ∃ C : ℕ, ∀ R : ℕ, R = 2 * C)
  (H2 : ∃ R : ℕ, ∀ F : ℕ, F = 2 * R - 4)
  (H3 : ∃ F : ℕ, F = 52) :
  C = 14 :=
by sorry

end NUMINAMATH_GPT_chickens_in_coop_l2163_216388


namespace NUMINAMATH_GPT_three_buses_interval_l2163_216309

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end NUMINAMATH_GPT_three_buses_interval_l2163_216309


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2163_216337

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ), (b = 5) → (c = 3) → (c^2 = a^2 + b) → (a > 0) →
  (a + c = 3) → (e = c / a) → (e = 3 / 2) :=
by
  intros a b c hb hc hc2 ha hac he
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2163_216337


namespace NUMINAMATH_GPT_oil_leakage_during_repair_l2163_216322

variables (initial_leak: ℚ) (initial_hours: ℚ) (repair_hours: ℚ) (reduction: ℚ) (total_leak: ℚ)

theorem oil_leakage_during_repair
    (h1 : initial_leak = 2475)
    (h2 : initial_hours = 7)
    (h3 : repair_hours = 5)
    (h4 : reduction = 0.75)
    (h5 : total_leak = 6206) :
    (total_leak - initial_leak = 3731) :=
by
  sorry

end NUMINAMATH_GPT_oil_leakage_during_repair_l2163_216322


namespace NUMINAMATH_GPT_ratio_of_buckets_l2163_216394

theorem ratio_of_buckets 
  (shark_feed_per_day : ℕ := 4)
  (dolphin_feed_per_day : ℕ := shark_feed_per_day / 2)
  (total_buckets : ℕ := 546)
  (days_in_weeks : ℕ := 3 * 7)
  (ratio_R : ℕ) :
  (total_buckets = days_in_weeks * (shark_feed_per_day + dolphin_feed_per_day + (ratio_R * shark_feed_per_day)) → ratio_R = 5) := sorry

end NUMINAMATH_GPT_ratio_of_buckets_l2163_216394


namespace NUMINAMATH_GPT_angle_in_first_quadrant_l2163_216345

def angle := -999 - 30 / 60 -- defining the angle as -999°30'
def coterminal (θ : Real) : Real := θ + 3 * 360 -- function to compute a coterminal angle

theorem angle_in_first_quadrant : 
  let θ := coterminal angle
  0 <= θ ∧ θ < 90 :=
by
  -- Exact proof steps would go here, but they are omitted as per instructions.
  sorry

end NUMINAMATH_GPT_angle_in_first_quadrant_l2163_216345


namespace NUMINAMATH_GPT_mass_of_man_proof_l2163_216307

def volume_displaced (L B h : ℝ) : ℝ :=
  L * B * h

def mass_of_man (V ρ : ℝ) : ℝ :=
  ρ * V

theorem mass_of_man_proof :
  ∀ (L B h ρ : ℝ), L = 9 → B = 3 → h = 0.01 → ρ = 1000 →
  mass_of_man (volume_displaced L B h) ρ = 270 :=
by
  intros L B h ρ L_eq B_eq h_eq ρ_eq
  rw [L_eq, B_eq, h_eq, ρ_eq]
  unfold volume_displaced
  unfold mass_of_man
  simp
  sorry

end NUMINAMATH_GPT_mass_of_man_proof_l2163_216307


namespace NUMINAMATH_GPT_total_kids_on_soccer_field_l2163_216383

theorem total_kids_on_soccer_field (initial_kids : ℕ) (joining_kids : ℕ) (total_kids : ℕ)
  (h₁ : initial_kids = 14)
  (h₂ : joining_kids = 22)
  (h₃ : total_kids = initial_kids + joining_kids) :
  total_kids = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_kids_on_soccer_field_l2163_216383


namespace NUMINAMATH_GPT_doses_A_correct_doses_B_correct_doses_C_correct_l2163_216363

def days_in_july : ℕ := 31

def daily_dose_A : ℕ := 1
def daily_dose_B : ℕ := 2
def daily_dose_C : ℕ := 3

def missed_days_A : ℕ := 3
def missed_days_B_morning : ℕ := 5
def missed_days_C_all : ℕ := 2

def total_doses_A : ℕ := days_in_july * daily_dose_A
def total_doses_B : ℕ := days_in_july * daily_dose_B
def total_doses_C : ℕ := days_in_july * daily_dose_C

def missed_doses_A : ℕ := missed_days_A * daily_dose_A
def missed_doses_B : ℕ := missed_days_B_morning
def missed_doses_C : ℕ := missed_days_C_all * daily_dose_C

def doses_consumed_A := total_doses_A - missed_doses_A
def doses_consumed_B := total_doses_B - missed_doses_B
def doses_consumed_C := total_doses_C - missed_doses_C

theorem doses_A_correct : doses_consumed_A = 28 := by sorry
theorem doses_B_correct : doses_consumed_B = 57 := by sorry
theorem doses_C_correct : doses_consumed_C = 87 := by sorry

end NUMINAMATH_GPT_doses_A_correct_doses_B_correct_doses_C_correct_l2163_216363


namespace NUMINAMATH_GPT_no_real_roots_iff_range_m_l2163_216381

open Real

theorem no_real_roots_iff_range_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + (m + 3) ≠ 0) ↔ (-2 < m ∧ m < 6) :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_iff_range_m_l2163_216381


namespace NUMINAMATH_GPT_correct_commutative_property_usage_l2163_216347

-- Definitions for the transformations
def transformA := 3 + (-2) = 2 + 3
def transformB := 4 + (-6) + 3 = (-6) + 4 + 3
def transformC := (5 + (-2)) + 4 = (5 + (-4)) + 2
def transformD := (1 / 6) + (-1) + (5 / 6) = ((1 / 6) + (5 / 6)) + 1

-- The theorem stating that transformB uses the commutative property correctly
theorem correct_commutative_property_usage : transformB :=
by
  sorry

end NUMINAMATH_GPT_correct_commutative_property_usage_l2163_216347


namespace NUMINAMATH_GPT_number_of_members_l2163_216301

theorem number_of_members (n : ℕ) (h : n^2 = 5929) : n = 77 :=
sorry

end NUMINAMATH_GPT_number_of_members_l2163_216301


namespace NUMINAMATH_GPT_Turner_Catapult_rides_l2163_216355

def tickets_needed (rollercoaster_rides Ferris_wheel_rides Catapult_rides : ℕ) : ℕ :=
  4 * rollercoaster_rides + 1 * Ferris_wheel_rides + 4 * Catapult_rides

theorem Turner_Catapult_rides :
  ∀ (x : ℕ), tickets_needed 3 1 x = 21 → x = 2 := by
  intros x h
  sorry

end NUMINAMATH_GPT_Turner_Catapult_rides_l2163_216355


namespace NUMINAMATH_GPT_Dan_has_five_limes_l2163_216342

-- Define the initial condition of limes Dan had
def initial_limes : Nat := 9

-- Define the limes Dan gave to Sara
def limes_given : Nat := 4

-- Define the remaining limes Dan has
def remaining_limes : Nat := initial_limes - limes_given

-- The theorem we need to prove, i.e., the remaining limes Dan has is 5
theorem Dan_has_five_limes : remaining_limes = 5 := by
  sorry

end NUMINAMATH_GPT_Dan_has_five_limes_l2163_216342


namespace NUMINAMATH_GPT_max_rectangle_area_l2163_216362

-- Definitions based on conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_rectangle_area
  (l w : ℕ)
  (h_perim : perimeter l w = 50)
  (h_prime : is_prime l)
  (h_composite : is_composite w) :
  l * w = 156 :=
sorry

end NUMINAMATH_GPT_max_rectangle_area_l2163_216362


namespace NUMINAMATH_GPT_problem1_problem2_l2163_216306

namespace MathProofs

theorem problem1 : (-3 - (-8) + (-6) + 10) = 9 :=
by
  sorry

theorem problem2 : (-12 * ((1 : ℚ) / 6 - (1 : ℚ) / 3 - 3 / 4)) = 11 :=
by
  sorry

end MathProofs

end NUMINAMATH_GPT_problem1_problem2_l2163_216306


namespace NUMINAMATH_GPT_plane_ticket_price_l2163_216341

theorem plane_ticket_price :
  ∀ (P : ℕ),
  (20 * 155) + 2900 = 30 * P →
  P = 200 := 
by
  sorry

end NUMINAMATH_GPT_plane_ticket_price_l2163_216341


namespace NUMINAMATH_GPT_original_price_before_discounts_l2163_216375

theorem original_price_before_discounts (P : ℝ) (h : 0.684 * P = 6840) : P = 10000 :=
by
  sorry

end NUMINAMATH_GPT_original_price_before_discounts_l2163_216375


namespace NUMINAMATH_GPT_solve_for_x_l2163_216324

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2163_216324


namespace NUMINAMATH_GPT_value_of_g_at_3_l2163_216391

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_of_g_at_3 : g 3 = 3 := by
  sorry

end NUMINAMATH_GPT_value_of_g_at_3_l2163_216391


namespace NUMINAMATH_GPT_simplify_expression_l2163_216302

theorem simplify_expression (x : ℝ) : 
  (12 * x ^ 12 - 3 * x ^ 10 + 5 * x ^ 9) + (-1 * x ^ 12 + 2 * x ^ 10 + x ^ 9 + 4 * x ^ 4 + 6 * x ^ 2 + 9) =
  11 * x ^ 12 - x ^ 10 + 6 * x ^ 9 + 4 * x ^ 4 + 6 * x ^ 2 + 9 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2163_216302


namespace NUMINAMATH_GPT_Diego_half_block_time_l2163_216372

def problem_conditions_and_solution : Prop :=
  ∃ (D : ℕ), (3 * 60 + D * 60) / 2 = 240 ∧ D = 5

theorem Diego_half_block_time :
  problem_conditions_and_solution :=
by
  sorry

end NUMINAMATH_GPT_Diego_half_block_time_l2163_216372


namespace NUMINAMATH_GPT_total_students_in_class_l2163_216399

theorem total_students_in_class
    (students_in_front : ℕ)
    (students_in_back : ℕ)
    (lines : ℕ)
    (total_students_line : ℕ)
    (total_class : ℕ)
    (h_front: students_in_front = 2)
    (h_back: students_in_back = 5)
    (h_lines: lines = 3)
    (h_students_line : total_students_line = students_in_front + 1 + students_in_back)
    (h_total_class : total_class = lines * total_students_line) :
  total_class = 24 := by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l2163_216399


namespace NUMINAMATH_GPT_minimum_value_of_x_minus_y_l2163_216312

variable (x y : ℝ)
open Real

theorem minimum_value_of_x_minus_y (hx : x > 0) (hy : y < 0) 
  (h : (1 / (x + 2)) + (1 / (1 - y)) = 1 / 6) : 
  x - y = 21 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_x_minus_y_l2163_216312


namespace NUMINAMATH_GPT_time_difference_l2163_216387

def joey_time : ℕ :=
  let uphill := 12 / 6 * 60
  let downhill := 10 / 25 * 60
  let flat := 20 / 15 * 60
  uphill + downhill + flat

def sue_time : ℕ :=
  let downhill := 10 / 35 * 60
  let uphill := 12 / 12 * 60
  let flat := 20 / 25 * 60
  downhill + uphill + flat

theorem time_difference : joey_time - sue_time = 99 := by
  -- calculation steps skipped
  sorry

end NUMINAMATH_GPT_time_difference_l2163_216387


namespace NUMINAMATH_GPT_giant_spider_weight_ratio_l2163_216368

theorem giant_spider_weight_ratio 
    (W_previous : ℝ)
    (A_leg : ℝ)
    (P : ℝ)
    (n : ℕ)
    (W_previous_eq : W_previous = 6.4)
    (A_leg_eq : A_leg = 0.5)
    (P_eq : P = 4)
    (n_eq : n = 8):
    (P * A_leg * n) / W_previous = 2.5 := by
  sorry

end NUMINAMATH_GPT_giant_spider_weight_ratio_l2163_216368


namespace NUMINAMATH_GPT_percentage_of_first_to_second_l2163_216364

theorem percentage_of_first_to_second (X : ℝ) (first second : ℝ) (h1 : first = (7 / 100) * X) (h2 : second = (14 / 100) * X) : 
(first / second) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_of_first_to_second_l2163_216364


namespace NUMINAMATH_GPT_quadratic_expression_rewrite_l2163_216313

theorem quadratic_expression_rewrite (a b c : ℝ) :
  (∀ x : ℝ, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) → a + b + c = 171 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_rewrite_l2163_216313


namespace NUMINAMATH_GPT_monotonic_intervals_extreme_value_closer_l2163_216327

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * (x - 1)

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧
  (a > 0 → (∀ x : ℝ, x < Real.log a → f x a > f (x + 1) a) ∧ (∀ x : ℝ, x > Real.log a → f x a < f (x + 1) a)) :=
sorry

theorem extreme_value_closer (a : ℝ) :
  a > e - 1 →
  ∀ x : ℝ, x ≥ 1 → |Real.exp 1/x - Real.log x| < |Real.exp (x - 1) + a - Real.log x| :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_extreme_value_closer_l2163_216327


namespace NUMINAMATH_GPT_largest_lucky_number_l2163_216318

theorem largest_lucky_number (n : ℕ) (h₀ : n = 160) (h₁ : ∀ k, 160 > k → k > 0) (h₂ : ∀ k, k ≡ 7 [MOD 16] → k ≤ 160) : 
  ∃ k, k = 151 := 
sorry

end NUMINAMATH_GPT_largest_lucky_number_l2163_216318


namespace NUMINAMATH_GPT_units_digit_G_n_for_n_eq_3_l2163_216370

def G (n : ℕ) : ℕ := 2 ^ 2 ^ 2 ^ n + 1

theorem units_digit_G_n_for_n_eq_3 : (G 3) % 10 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_G_n_for_n_eq_3_l2163_216370


namespace NUMINAMATH_GPT_div_mult_result_l2163_216390

theorem div_mult_result : 150 / (30 / 3) * 2 = 30 :=
by sorry

end NUMINAMATH_GPT_div_mult_result_l2163_216390


namespace NUMINAMATH_GPT_magician_card_trick_l2163_216378

-- Definitions and proof goal
theorem magician_card_trick :
  let n := 12
  let total_cards := n ^ 2
  let duplicate_cards := n
  let non_duplicate_cards := total_cards - duplicate_cards - (n - 1) - (n - 1)
  let total_ways_with_two_duplicates := Nat.choose duplicate_cards 2
  let total_ways_with_one_duplicate :=
    duplicate_cards * non_duplicate_cards
  (total_ways_with_two_duplicates + total_ways_with_one_duplicate) = 1386 :=
by
  let n := 12
  let total_cards := n ^ 2
  let duplicate_cards := n
  let non_duplicate_cards := total_cards - duplicate_cards - (n - 1) - (n - 1)
  let total_ways_with_two_duplicates := Nat.choose duplicate_cards 2
  let total_ways_with_one_duplicate :=
    duplicate_cards * non_duplicate_cards
  have h : (total_ways_with_two_duplicates + total_ways_with_one_duplicate) = 1386 := sorry
  exact h

end NUMINAMATH_GPT_magician_card_trick_l2163_216378


namespace NUMINAMATH_GPT_quad_area_l2163_216377

theorem quad_area (a b : Int) (h1 : a > b) (h2 : b > 0) (h3 : 2 * |a - b| * |a + b| = 50) : a + b = 15 :=
by
  sorry

end NUMINAMATH_GPT_quad_area_l2163_216377


namespace NUMINAMATH_GPT_Peter_speed_is_correct_l2163_216351

variable (Peter_speed : ℝ)

def Juan_speed : ℝ := Peter_speed + 3

def distance_Peter_in_1_5_hours : ℝ := 1.5 * Peter_speed

def distance_Juan_in_1_5_hours : ℝ := 1.5 * Juan_speed Peter_speed

theorem Peter_speed_is_correct (h : distance_Peter_in_1_5_hours Peter_speed + distance_Juan_in_1_5_hours Peter_speed = 19.5) : Peter_speed = 5 :=
by
  sorry

end NUMINAMATH_GPT_Peter_speed_is_correct_l2163_216351


namespace NUMINAMATH_GPT_diff_of_two_numbers_l2163_216314

theorem diff_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end NUMINAMATH_GPT_diff_of_two_numbers_l2163_216314


namespace NUMINAMATH_GPT_converse_false_inverse_false_l2163_216310

-- Definitions of the conditions
def is_rhombus (Q : Type) : Prop := -- definition of a rhombus
  sorry

def is_parallelogram (Q : Type) : Prop := -- definition of a parallelogram
  sorry

variable {Q : Type}

-- Initial statement: If a quadrilateral is a rhombus, then it is a parallelogram.
axiom initial_statement : is_rhombus Q → is_parallelogram Q

-- Goals: Prove both the converse and inverse are false
theorem converse_false : ¬ ((is_parallelogram Q) → (is_rhombus Q)) :=
sorry

theorem inverse_false : ¬ (¬ (is_rhombus Q) → ¬ (is_parallelogram Q)) :=
    sorry

end NUMINAMATH_GPT_converse_false_inverse_false_l2163_216310


namespace NUMINAMATH_GPT_circle_condition_l2163_216321

theorem circle_condition (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - x + y + m = 0 → (m < 1 / 2)) :=
by {
-- Skipping the proof here
sorry
}

end NUMINAMATH_GPT_circle_condition_l2163_216321


namespace NUMINAMATH_GPT_captain_co_captain_selection_l2163_216339

theorem captain_co_captain_selection 
  (men women : ℕ)
  (h_men : men = 12) 
  (h_women : women = 12) : 
  (men * (men - 1) + women * (women - 1)) = 264 := 
by
  -- Since we are skipping the proof here, we use sorry.
  sorry

end NUMINAMATH_GPT_captain_co_captain_selection_l2163_216339


namespace NUMINAMATH_GPT_marble_prob_l2163_216398

theorem marble_prob
  (a b x y m n : ℕ)
  (h1 : a + b = 30)
  (h2 : (x : ℚ) / a * (y : ℚ) / b = 4 / 9)
  (h3 : x * y = 36)
  (h4 : Nat.gcd m n = 1)
  (h5 : (a - x : ℚ) / a * (b - y) / b = m / n) :
  m + n = 29 := 
sorry

end NUMINAMATH_GPT_marble_prob_l2163_216398


namespace NUMINAMATH_GPT_compute_expression_l2163_216397

theorem compute_expression : 3 * 3^3 - 9^50 / 9^48 = 0 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l2163_216397


namespace NUMINAMATH_GPT_line_equation_l2163_216392

variable (θ : ℝ) (b : ℝ) (y x : ℝ)

-- Conditions: 
-- Slope angle θ = 45°
def slope_angle_condition : θ = 45 := by
  sorry

-- Y-intercept b = 2
def y_intercept_condition : b = 2 := by
  sorry

-- Given these conditions, we want to prove the line equation
theorem line_equation (x : ℝ) (θ : ℝ) (b : ℝ) :
  θ = 45 → b = 2 → y = x + 2 := by
  sorry

end NUMINAMATH_GPT_line_equation_l2163_216392


namespace NUMINAMATH_GPT_find_c_plus_d_l2163_216354

-- Conditions as definitions
variables {P A C : Point }
variables {O₁ O₂ : Point}
variables {AB AP CP : ℝ}
variables {c d : ℕ}

-- Given conditions
def Point_on_diagonal (P A C : Point) : Prop := true -- We need to code the detailed properties of being on the diagonal
def circumcenter_of_triangle (P Q R O : Point) : Prop := true -- We need to code the properties of being a circumcenter
def AP_greater_than_CP (AP CP : ℝ) : Prop := AP > CP
def angle_right (A B O : Point) : Prop := true -- Define the right angle property

-- Main statement to prove
theorem find_c_plus_d : 
  Point_on_diagonal P A C ∧
  circumcenter_of_triangle A B P O₁ ∧ 
  circumcenter_of_triangle C D P O₂ ∧ 
  AP_greater_than_CP AP CP ∧
  AB = 10 ∧
  angle_right O₁ P O₂ ∧
  (AP = Real.sqrt c + Real.sqrt d) →
  (c + d = 100) :=
by
  sorry

end NUMINAMATH_GPT_find_c_plus_d_l2163_216354


namespace NUMINAMATH_GPT_sequence_properties_l2163_216393

variable {a : ℕ → ℤ}

-- Conditions
axiom seq_add : ∀ (p q : ℕ), 1 ≤ p → 1 ≤ q → a (p + q) = a p + a q
axiom a2_neg4 : a 2 = -4

-- Theorem statement: We need to prove a6 = -12 and a_n = -2n for all n
theorem sequence_properties :
  (a 6 = -12) ∧ ∀ n : ℕ, 1 ≤ n → a n = -2 * n :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l2163_216393


namespace NUMINAMATH_GPT_set_intersection_set_union_set_complement_l2163_216317

open Set

variable (U : Set ℝ) (A B : Set ℝ)
noncomputable def setA : Set ℝ := {x | x^2 - 3*x - 4 ≥ 0}
noncomputable def setB : Set ℝ := {x | x < 5}

theorem set_intersection : (U = univ) -> (A = setA) -> (B = setB) -> A ∩ B = Ico 4 5 := by
  intros
  sorry

theorem set_union : (U = univ) -> (A = setA) -> (B = setB) -> A ∪ B = univ := by
  intros
  sorry

theorem set_complement : (U = univ) -> (A = setA) -> U \ A = Ioo (-1 : ℝ) 4 := by
  intros
  sorry

end NUMINAMATH_GPT_set_intersection_set_union_set_complement_l2163_216317


namespace NUMINAMATH_GPT_smallest_y_in_geometric_sequence_l2163_216308

theorem smallest_y_in_geometric_sequence (x y z r : ℕ) (h1 : y = x * r) (h2 : z = x * r^2) (h3 : xyz = 125) : y = 5 :=
by sorry

end NUMINAMATH_GPT_smallest_y_in_geometric_sequence_l2163_216308


namespace NUMINAMATH_GPT_circle_radius_l2163_216360

theorem circle_radius (A r : ℝ) (h1 : A = 64 * Real.pi) (h2 : A = Real.pi * r^2) : r = 8 := 
by
  sorry

end NUMINAMATH_GPT_circle_radius_l2163_216360
