import Mathlib

namespace NUMINAMATH_GPT_football_goals_l404_40411

variable (A : ℚ) (G : ℚ)

theorem football_goals (A G : ℚ) 
    (h1 : G = 14 * A)
    (h2 : G + 3 = (A + 0.08) * 15) :
    G = 25.2 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_football_goals_l404_40411


namespace NUMINAMATH_GPT_min_max_calculation_l404_40418

theorem min_max_calculation
  (p q r s : ℝ)
  (h1 : p + q + r + s = 8)
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  -32 ≤ 5 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) ∧
  5 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) ≤ 12 :=
sorry

end NUMINAMATH_GPT_min_max_calculation_l404_40418


namespace NUMINAMATH_GPT_standard_equation_of_circle_l404_40485

theorem standard_equation_of_circle
  (r : ℝ) (h_radius : r = 1)
  (h_center : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (x, y) = (a, b))
  (h_tangent_line : ∃ (a : ℝ), 1 = |4 * a - 3| / 5)
  (h_tangent_x_axis : ∃ (a : ℝ), a = 1) :
  (∃ (a b : ℝ), (x-2)^2 + (y-1)^2 = 1) :=
sorry

end NUMINAMATH_GPT_standard_equation_of_circle_l404_40485


namespace NUMINAMATH_GPT_maggi_ate_5_cupcakes_l404_40499

theorem maggi_ate_5_cupcakes
  (packages : ℕ)
  (cupcakes_per_package : ℕ)
  (left_cupcakes : ℕ)
  (total_cupcakes : ℕ := packages * cupcakes_per_package)
  (eaten_cupcakes : ℕ := total_cupcakes - left_cupcakes)
  (h1 : packages = 3)
  (h2 : cupcakes_per_package = 4)
  (h3 : left_cupcakes = 7) :
  eaten_cupcakes = 5 :=
by
  sorry

end NUMINAMATH_GPT_maggi_ate_5_cupcakes_l404_40499


namespace NUMINAMATH_GPT_amy_remaining_money_l404_40455

-- Define initial amount and purchases
def initial_amount : ℝ := 15
def stuffed_toy_cost : ℝ := 2
def hot_dog_cost : ℝ := 3.5
def candy_apple_cost : ℝ := 1.5
def discount_rate : ℝ := 0.5

-- Define the discounted hot_dog_cost
def discounted_hot_dog_cost := hot_dog_cost * discount_rate

-- Define the total spent
def total_spent := stuffed_toy_cost + discounted_hot_dog_cost + candy_apple_cost

-- Define the remaining amount
def remaining_amount := initial_amount - total_spent

theorem amy_remaining_money : remaining_amount = 9.75 := by
  sorry

end NUMINAMATH_GPT_amy_remaining_money_l404_40455


namespace NUMINAMATH_GPT_average_value_of_T_l404_40417

noncomputable def expected_value_T : ℕ := 22

theorem average_value_of_T (boys girls : ℕ) (boy_pair girl_pair : Prop) (T : ℕ) :
  boys = 9 → girls = 15 →
  boy_pair ∧ girl_pair →
  T = expected_value_T :=
by
  intros h_boys h_girls h_pairs
  sorry

end NUMINAMATH_GPT_average_value_of_T_l404_40417


namespace NUMINAMATH_GPT_total_cost_is_100_l404_40466

def shirts : ℕ := 10
def pants : ℕ := shirts / 2
def cost_shirt : ℕ := 6
def cost_pant : ℕ := 8

theorem total_cost_is_100 :
  shirts * cost_shirt + pants * cost_pant = 100 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_100_l404_40466


namespace NUMINAMATH_GPT_smallest_positive_n_l404_40477

theorem smallest_positive_n (n : ℕ) (h : 1023 * n % 30 = 2147 * n % 30) : n = 15 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_n_l404_40477


namespace NUMINAMATH_GPT_additional_amount_needed_l404_40453

-- Define the amounts spent on shampoo, conditioner, and lotion
def shampoo_cost : ℝ := 10.00
def conditioner_cost : ℝ := 10.00
def lotion_cost_per_bottle : ℝ := 6.00
def lotion_quantity : ℕ := 3

-- Define the amount required for free shipping
def free_shipping_threshold : ℝ := 50.00

-- Calculate the total amount spent
def total_spent : ℝ := shampoo_cost + conditioner_cost + (lotion_quantity * lotion_cost_per_bottle)

-- Define the additional amount needed for free shipping
def additional_needed_for_shipping : ℝ := free_shipping_threshold - total_spent

-- The final goal to prove
theorem additional_amount_needed : additional_needed_for_shipping = 12.00 :=
by
  sorry

end NUMINAMATH_GPT_additional_amount_needed_l404_40453


namespace NUMINAMATH_GPT_measure_Z_is_19_6_l404_40409

def measure_angle_X : ℝ := 72
def measure_Y (measure_Z : ℝ) : ℝ := 4 * measure_Z + 10
def angle_sum_condition (measure_Z : ℝ) : Prop :=
  measure_angle_X + (measure_Y measure_Z) + measure_Z = 180

theorem measure_Z_is_19_6 :
  ∃ measure_Z : ℝ, measure_Z = 19.6 ∧ angle_sum_condition measure_Z :=
by
  sorry

end NUMINAMATH_GPT_measure_Z_is_19_6_l404_40409


namespace NUMINAMATH_GPT_son_l404_40400

theorem son's_present_age
  (S F : ℤ)
  (h1 : F = S + 45)
  (h2 : F + 10 = 4 * (S + 10))
  (h3 : S + 15 = 2 * S) :
  S = 15 :=
by
  sorry

end NUMINAMATH_GPT_son_l404_40400


namespace NUMINAMATH_GPT_find_theta_2phi_l404_40456

-- Given
variables {θ φ : ℝ}
variables (hθ_acute : 0 < θ ∧ θ < π / 2)
variables (hφ_acute : 0 < φ ∧ φ < π / 2)
variables (h_tanθ : Real.tan θ = 3 / 11)
variables (h_sinφ : Real.sin φ = 1 / 3)

-- To prove
theorem find_theta_2phi : 
  ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ Real.tan x = (21 + 6 * Real.sqrt 2) / (77 - 6 * Real.sqrt 2) ∧ x = θ + 2 * φ := 
sorry

end NUMINAMATH_GPT_find_theta_2phi_l404_40456


namespace NUMINAMATH_GPT_eval_expr_l404_40497

theorem eval_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  (x^(2 * y) * y^(3 * x) / (y^(2 * y) * x^(3 * x))) = x^(2 * y - 3 * x) * y^(3 * x - 2 * y) :=
by
  sorry

end NUMINAMATH_GPT_eval_expr_l404_40497


namespace NUMINAMATH_GPT_exponents_mod_7_l404_40498

theorem exponents_mod_7 : (2222 ^ 5555 + 5555 ^ 2222) % 7 = 0 := 
by 
  -- sorries here because no proof is needed as stated
  sorry

end NUMINAMATH_GPT_exponents_mod_7_l404_40498


namespace NUMINAMATH_GPT_negative_integer_is_minus_21_l404_40443

variable (n : ℤ) (hn : n < 0) (h : n * (-3) + 2 = 65)

theorem negative_integer_is_minus_21 : n = -21 :=
by
  sorry

end NUMINAMATH_GPT_negative_integer_is_minus_21_l404_40443


namespace NUMINAMATH_GPT_jessica_exam_time_l404_40489

theorem jessica_exam_time (total_questions : ℕ) (answered_questions : ℕ) (used_minutes : ℕ)
    (total_time : ℕ) (remaining_time : ℕ) (rate : ℚ) :
    total_questions = 80 ∧ answered_questions = 16 ∧ used_minutes = 12 ∧ total_time = 60 ∧ rate = (answered_questions : ℚ) / used_minutes →
    remaining_time = total_time - used_minutes →
    remaining_time = 48 :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_jessica_exam_time_l404_40489


namespace NUMINAMATH_GPT_smallest_positive_period_intervals_monotonic_increase_max_min_values_l404_40480

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2

theorem smallest_positive_period (x : ℝ) : (f (x + π)) = f x :=
sorry

theorem intervals_monotonic_increase (k : ℤ) (x : ℝ) : (k * π - π/3) ≤ x ∧ x ≤ (k * π + π/6) → ∃ a b : ℝ, a < b ∧ ∀ x : ℝ, (a ≤ x ∧ x ≤ b) →
  (f x < f (x + 1)) :=
sorry

theorem max_min_values (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ π/4) : (∃ y : ℝ, y = max (f 0) (f (π/6)) ∧ y = 1) ∧ (∃ z : ℝ, z = min (f 0) (f (π/6)) ∧ z = 0) :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_intervals_monotonic_increase_max_min_values_l404_40480


namespace NUMINAMATH_GPT_runner_time_difference_l404_40459

theorem runner_time_difference 
  (v : ℝ)  -- runner's initial speed (miles per hour)
  (H1 : 0 < v)  -- speed is positive
  (d : ℝ)  -- total distance
  (H2 : d = 40)  -- total distance condition
  (t2 : ℝ)  -- time taken for the second half
  (H3 : t2 = 10)  -- second half time condition
  (H4 : v ≠ 0)  -- initial speed cannot be zero
  (H5: 20 = 10 * (v / 2))  -- equation derived from the second half conditions
  : (t2 - (20 / v)) = 5 := 
by
  sorry

end NUMINAMATH_GPT_runner_time_difference_l404_40459


namespace NUMINAMATH_GPT_solve_eqn_l404_40479

theorem solve_eqn (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 56) : x + y = 2 := by
  sorry

end NUMINAMATH_GPT_solve_eqn_l404_40479


namespace NUMINAMATH_GPT_range_of_m_l404_40428

open Real

def f (x m: ℝ) : ℝ := x^2 - 2 * x + m^2 + 3 * m - 3

def p (m: ℝ) : Prop := ∃ x, f x m < 0

def q (m: ℝ) : Prop := (5 * m - 1 > 0) ∧ (m - 2 > 0)

theorem range_of_m (m : ℝ) : ¬ (p m ∨ q m) ∧ ¬ (p m ∧ q m) → (m ≤ -4 ∨ m ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l404_40428


namespace NUMINAMATH_GPT_represent_sum_and_product_eq_231_l404_40449

theorem represent_sum_and_product_eq_231 :
  ∃ (x y z w : ℕ), x = 3 ∧ y = 7 ∧ z = 11 ∧ w = 210 ∧ (231 = x + y + z + w) ∧ (231 = x * y * z) :=
by
  -- The proof is omitted here.
  sorry

end NUMINAMATH_GPT_represent_sum_and_product_eq_231_l404_40449


namespace NUMINAMATH_GPT_fluctuations_B_greater_than_A_l404_40491

variable (A B : Type)
variable (mean_A mean_B : ℝ)
variable (var_A var_B : ℝ)

-- Given conditions
axiom avg_A : mean_A = 5
axiom avg_B : mean_B = 5
axiom variance_A : var_A = 0.1
axiom variance_B : var_B = 0.2

-- The proof problem statement
theorem fluctuations_B_greater_than_A : var_A < var_B :=
by sorry

end NUMINAMATH_GPT_fluctuations_B_greater_than_A_l404_40491


namespace NUMINAMATH_GPT_time_to_paint_one_room_l404_40492

variables (rooms_total rooms_painted : ℕ) (hours_to_paint_remaining : ℕ)

-- The conditions
def painter_conditions : Prop :=
  rooms_total = 10 ∧ rooms_painted = 8 ∧ hours_to_paint_remaining = 16

-- The goal is to find out the hours to paint one room
theorem time_to_paint_one_room (h : painter_conditions rooms_total rooms_painted hours_to_paint_remaining) : 
  let rooms_remaining := rooms_total - rooms_painted
  let hours_per_room := hours_to_paint_remaining / rooms_remaining
  hours_per_room = 8 :=
by sorry

end NUMINAMATH_GPT_time_to_paint_one_room_l404_40492


namespace NUMINAMATH_GPT_circle_chord_intersect_zero_l404_40405

noncomputable def circle_product (r : ℝ) : ℝ :=
  let O := (0, 0)
  let A := (r, 0)
  let B := (-r, 0)
  let C := (0, r)
  let D := (0, -r)
  let P := (0, 0)
  (dist A P) * (dist P B)

theorem circle_chord_intersect_zero (r : ℝ) :
  let A := (r, 0)
  let B := (-r, 0)
  let C := (0, r)
  let D := (0, -r)
  let P := (0, 0)
  (dist A P) * (dist P B) = 0 :=
by sorry

end NUMINAMATH_GPT_circle_chord_intersect_zero_l404_40405


namespace NUMINAMATH_GPT_christmas_day_december_25_l404_40430

-- Define the conditions
def is_thursday (d: ℕ) : Prop := d % 7 = 4
def thanksgiving := 26
def december_christmas := 25

-- Define the problem as a proof problem
theorem christmas_day_december_25 :
  is_thursday (thanksgiving) → thanksgiving = 26 →
  december_christmas = 25 → 
  30 - 26 + 25 = 28 → 
  is_thursday (30 - 26 + 25) :=
by
  intro h_thursday h_thanksgiving h_christmas h_days
  -- skipped proof
  sorry

end NUMINAMATH_GPT_christmas_day_december_25_l404_40430


namespace NUMINAMATH_GPT_parabola_vertex_l404_40487

theorem parabola_vertex (y x : ℝ) : y^2 - 4*y + 3*x + 7 = 0 → (x = -1 ∧ y = 2) := 
sorry

end NUMINAMATH_GPT_parabola_vertex_l404_40487


namespace NUMINAMATH_GPT_odd_function_value_l404_40495

def f (a x : ℝ) : ℝ := -x^3 + (a-2)*x^2 + x

-- Test that f(x) is an odd function:
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem odd_function_value (a : ℝ) (h : is_odd_function (f a)) : f a a = -6 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_value_l404_40495


namespace NUMINAMATH_GPT_general_term_l404_40420

noncomputable def seq (n : ℕ) : ℤ :=
  if n = 0 then 0 else
  if n = 1 then -1 else
  if n % 2 = 0 then (2 * 2 ^ (n / 2 - 1) - 1) / 3 else 
  (-2)^(n - n / 2) / 3 - 1

-- Conditions
def condition1 : Prop := seq 1 = -1
def condition2 : Prop := seq 2 > seq 1
def condition3 (n : ℕ) : Prop := |seq (n + 1) - seq n| = 2^n
def condition4 : Prop := ∀ m, seq (2*m + 1) > seq (2*m - 1)
def condition5 : Prop := ∀ m, seq (2*m) < seq (2*m + 2)

-- The theorem stating the general term of the sequence
theorem general_term (n : ℕ) :
  condition1 →
  condition2 →
  (∀ n, condition3 n) →
  condition4 →
  condition5 →
  seq n = ( (-2)^n - 1) / 3 :=
by
  sorry

end NUMINAMATH_GPT_general_term_l404_40420


namespace NUMINAMATH_GPT_Theresa_video_games_l404_40461

variable (Tory Julia Theresa : ℕ)

def condition1 : Prop := Tory = 6
def condition2 : Prop := Julia = Tory / 3
def condition3 : Prop := Theresa = (Julia * 3) + 5

theorem Theresa_video_games : condition1 Tory → condition2 Tory Julia → condition3 Julia Theresa → Theresa = 11 := by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end NUMINAMATH_GPT_Theresa_video_games_l404_40461


namespace NUMINAMATH_GPT_max_b_value_l404_40425

theorem max_b_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a * b = (2 * a - b) / (2 * a + 3 * b)) : b ≤ 1 / 3 :=
  sorry

end NUMINAMATH_GPT_max_b_value_l404_40425


namespace NUMINAMATH_GPT_minimal_bananas_l404_40482

noncomputable def total_min_bananas : ℕ :=
  let b1 := 72
  let b2 := 72
  let b3 := 216
  let b4 := 72
  b1 + b2 + b3 + b4

theorem minimal_bananas (total_bananas : ℕ) (ratio1 ratio2 ratio3 ratio4 : ℕ) 
  (b1 b2 b3 b4 : ℕ) 
  (h_ratio : ratio1 = 4 ∧ ratio2 = 3 ∧ ratio3 = 2 ∧ ratio4 = 1) 
  (h_div_constraints : ∀ n m : ℕ, (n % m = 0 ∨ m % n = 0) ∧ n ≥ ratio1 * ratio2 * ratio3 * ratio4) 
  (h_bananas : b1 = 72 ∧ b2 = 72 ∧ b3 = 216 ∧ b4 = 72 ∧ 
              4 * (b1 / 2 + b2 / 6 + b3 / 9 + 7 * b4 / 72) = 3 * (b1 / 6 + b2 / 3 + b3 / 9 + 7 * b4 / 72) ∧ 
              2 * (b1 / 6 + b2 / 6 + b3 / 6 + 7 * b4 / 72) = (b1 / 6 + b2 / 6 + b3 / 9 + b4 / 8)) : 
  total_bananas = 432 := by
  sorry

end NUMINAMATH_GPT_minimal_bananas_l404_40482


namespace NUMINAMATH_GPT_number_of_added_groups_l404_40448

-- Define the total number of students in the class
def total_students : ℕ := 47

-- Define the number of students per table and the number of tables
def students_per_table : ℕ := 3
def number_of_tables : ℕ := 6

-- Define the number of girls in the bathroom and the multiplier for students in the canteen
def girls_in_bathroom : ℕ := 3
def canteen_multiplier : ℕ := 3

-- Define the number of foreign exchange students from each country
def foreign_exchange_germany : ℕ := 3
def foreign_exchange_france : ℕ := 3
def foreign_exchange_norway : ℕ := 3

-- Define the number of students per recently added group
def students_per_group : ℕ := 4

-- Calculate the number of students currently in the classroom
def students_in_classroom := number_of_tables * students_per_table

-- Calculate the number of students temporarily absent
def students_in_canteen := girls_in_bathroom * canteen_multiplier
def temporarily_absent := girls_in_bathroom + students_in_canteen

-- Calculate the number of foreign exchange students missing
def foreign_exchange_missing := foreign_exchange_germany + foreign_exchange_france + foreign_exchange_norway

-- Calculate the total number of students accounted for
def student_accounted_for := students_in_classroom + temporarily_absent + foreign_exchange_missing

-- The proof statement (main goal)
theorem number_of_added_groups : (total_students - student_accounted_for) / students_per_group = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_added_groups_l404_40448


namespace NUMINAMATH_GPT_consecutive_sums_permutations_iff_odd_l404_40404

theorem consecutive_sums_permutations_iff_odd (n : ℕ) (h : n ≥ 2) :
  (∃ (a b : Fin n → ℕ), (∀ i, 1 ≤ a i ∧ a i ≤ n) ∧ (∀ i, 1 ≤ b i ∧ b i ≤ n) ∧
    ∃ N, ∀ i, a i + b i = N + i) ↔ (Odd n) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_sums_permutations_iff_odd_l404_40404


namespace NUMINAMATH_GPT_alpha_in_third_quadrant_l404_40445

theorem alpha_in_third_quadrant (k : ℤ) (α : ℝ) :
  (4 * k + 1) * 180 < α ∧ α < (4 * k + 1) * 180 + 60 → 180 < α ∧ α < 240 :=
  sorry

end NUMINAMATH_GPT_alpha_in_third_quadrant_l404_40445


namespace NUMINAMATH_GPT_chess_tournament_games_l404_40483

theorem chess_tournament_games (n : ℕ) (h : 2 * 404 = n * (n - 4)) : False :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l404_40483


namespace NUMINAMATH_GPT_hannah_late_times_l404_40401

variable (hourly_rate : ℝ)
variable (hours_worked : ℝ)
variable (dock_per_late : ℝ)
variable (actual_pay : ℝ)

theorem hannah_late_times (h1 : hourly_rate = 30)
                          (h2 : hours_worked = 18)
                          (h3 : dock_per_late = 5)
                          (h4 : actual_pay = 525) :
  ((hourly_rate * hours_worked - actual_pay) / dock_per_late) = 3 := 
by
  sorry

end NUMINAMATH_GPT_hannah_late_times_l404_40401


namespace NUMINAMATH_GPT_geometric_sequence_general_term_l404_40451

theorem geometric_sequence_general_term :
  ∀ (n : ℕ), (n > 0) →
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ (k : ℕ), k > 0 → a (k+1) = 2 * a k) ∧ a n = 2^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_l404_40451


namespace NUMINAMATH_GPT_complement_union_l404_40447

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end NUMINAMATH_GPT_complement_union_l404_40447


namespace NUMINAMATH_GPT_complete_square_solution_l404_40429

theorem complete_square_solution :
  ∀ x : ℝ, x^2 - 4 * x - 22 = 0 → (x - 2)^2 = 26 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_complete_square_solution_l404_40429


namespace NUMINAMATH_GPT_ratio_of_shorts_to_pants_is_half_l404_40474

-- Define the parameters
def shirts := 4
def pants := 2 * shirts
def total_clothes := 16

-- Define the number of shorts
def shorts := total_clothes - (shirts + pants)

-- Define the ratio
def ratio := shorts / pants

-- Prove the ratio is 1/2
theorem ratio_of_shorts_to_pants_is_half : ratio = 1 / 2 :=
by
  -- Start the proof, but leave it as sorry
  sorry

end NUMINAMATH_GPT_ratio_of_shorts_to_pants_is_half_l404_40474


namespace NUMINAMATH_GPT_quadratic_min_value_l404_40446

theorem quadratic_min_value :
  ∃ x : ℝ, (∀ y : ℝ, y = x^2 - 4 * x + 7 → y ≥ 3) ∧ (x = 2 → (x^2 - 4 * x + 7 = 3)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_min_value_l404_40446


namespace NUMINAMATH_GPT_ratio_of_milk_to_water_l404_40440

namespace MixtureProblem

def initial_milk (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) : ℕ :=
  (milk_ratio * total_volume) / (milk_ratio + water_ratio)

def initial_water (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) : ℕ :=
  (water_ratio * total_volume) / (milk_ratio + water_ratio)

theorem ratio_of_milk_to_water (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) (added_water : ℕ) :
  milk_ratio = 4 → water_ratio = 1 → total_volume = 45 → added_water = 21 → 
  (initial_milk total_volume milk_ratio water_ratio) = 36 →
  (initial_water total_volume milk_ratio water_ratio + added_water) = 30 →
  (36 / 30 : ℚ) = 6 / 5 :=
by
  intros
  sorry

end MixtureProblem

end NUMINAMATH_GPT_ratio_of_milk_to_water_l404_40440


namespace NUMINAMATH_GPT_select_team_of_5_l404_40465

def boys : ℕ := 7
def girls : ℕ := 9
def total_students : ℕ := boys + girls

theorem select_team_of_5 (n : ℕ := total_students) (k : ℕ := 5) :
  (Nat.choose n k) = 4368 :=
by
  sorry

end NUMINAMATH_GPT_select_team_of_5_l404_40465


namespace NUMINAMATH_GPT_group_discount_l404_40438

theorem group_discount (P : ℝ) (D : ℝ) :
  4 * (P - (D / 100) * P) = 3 * P → D = 25 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_group_discount_l404_40438


namespace NUMINAMATH_GPT_highlighter_difference_l404_40424

theorem highlighter_difference :
  ∃ (P : ℕ), 7 + P + (P + 5) = 40 ∧ P - 7 = 7 :=
by
  sorry

end NUMINAMATH_GPT_highlighter_difference_l404_40424


namespace NUMINAMATH_GPT_range_of_f_l404_40493

theorem range_of_f (x : ℝ) (h : x ∈ Set.Icc (-3 : ℝ) 3) : 
  ∃ y, y ∈ Set.Icc (0 : ℝ) 25 ∧ ∀ z, z = (x^2 - 4*x + 4) → y = z :=
sorry

end NUMINAMATH_GPT_range_of_f_l404_40493


namespace NUMINAMATH_GPT_decimal_to_vulgar_fraction_l404_40473

theorem decimal_to_vulgar_fraction (d : ℚ) (h : d = 0.36) : d = 9 / 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_decimal_to_vulgar_fraction_l404_40473


namespace NUMINAMATH_GPT_dinner_cost_l404_40464

theorem dinner_cost (tax_rate tip_rate total_cost : ℝ) (h_tax : tax_rate = 0.12) (h_tip : tip_rate = 0.20) (h_total : total_cost = 30.60) :
  let meal_cost := total_cost / (1 + tax_rate + tip_rate)
  meal_cost = 23.18 :=
by
  sorry

end NUMINAMATH_GPT_dinner_cost_l404_40464


namespace NUMINAMATH_GPT_sum_of_numbers_l404_40427

theorem sum_of_numbers {a b c : ℝ} (h1 : b = 7) (h2 : (a + b + c) / 3 = a + 8) (h3 : (a + b + c) / 3 = c - 20) : a + b + c = 57 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l404_40427


namespace NUMINAMATH_GPT_volume_in_cubic_yards_l404_40432

-- Define the conditions given in the problem
def volume_in_cubic_feet : ℕ := 216
def cubic_feet_per_cubic_yard : ℕ := 27

-- Define the theorem that needs to be proven
theorem volume_in_cubic_yards :
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 8 :=
by
  sorry

end NUMINAMATH_GPT_volume_in_cubic_yards_l404_40432


namespace NUMINAMATH_GPT_degrees_to_radians_l404_40439

theorem degrees_to_radians (degrees : ℝ) (pi : ℝ) : 
  degrees * (pi / 180) = pi / 15 ↔ degrees = 12 :=
by 
  sorry

end NUMINAMATH_GPT_degrees_to_radians_l404_40439


namespace NUMINAMATH_GPT_size_ratio_l404_40433

variable {U : ℝ} (h1 : C = 1.5 * U) (h2 : R = 4 / 3 * C)

theorem size_ratio : R = 8 / 3 * U :=
by
  sorry

end NUMINAMATH_GPT_size_ratio_l404_40433


namespace NUMINAMATH_GPT_pirate_treasure_probability_l404_40469

theorem pirate_treasure_probability :
  let p_treasure := 1 / 5
  let p_trap_no_treasure := 1 / 10
  let p_notreasure_notrap := 7 / 10
  let combinatorial_factor := Nat.choose 8 4
  let probability := (combinatorial_factor * (p_treasure ^ 4) * (p_notreasure_notrap ^ 4))
  probability = 33614 / 1250000 :=
by
  sorry

end NUMINAMATH_GPT_pirate_treasure_probability_l404_40469


namespace NUMINAMATH_GPT_arithmetic_mean_of_arithmetic_progression_l404_40476

variable (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ)

/-- General term of an arithmetic progression -/
def arithmetic_progression (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

theorem arithmetic_mean_of_arithmetic_progression (k p : ℕ) (hk : 1 < k) :
  a k = (a (k - p) + a (k + p)) / 2 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_arithmetic_progression_l404_40476


namespace NUMINAMATH_GPT_root_proof_l404_40450

noncomputable def p : ℝ := (-5 + Real.sqrt 21) / 2
noncomputable def q : ℝ := (-5 - Real.sqrt 21) / 2

theorem root_proof :
  (∃ (p q : ℝ), (∀ x : ℝ, x^3 + 6 * x^2 + 6 * x + 1 = 0 → (x = p ∨ x = q ∨ x = -1)) ∧ 
                 ((p = (-5 + Real.sqrt 21) / 2) ∧ (q = (-5 - Real.sqrt 21) / 2))) →
  (p / q + q / p = 23) :=
by
  sorry

end NUMINAMATH_GPT_root_proof_l404_40450


namespace NUMINAMATH_GPT_num_teacher_volunteers_l404_40426

theorem num_teacher_volunteers (total_needed volunteers_from_classes extra_needed teacher_volunteers : ℕ)
  (h1 : teacher_volunteers + extra_needed + volunteers_from_classes = total_needed) 
  (h2 : total_needed = 50)
  (h3 : volunteers_from_classes = 6 * 5)
  (h4 : extra_needed = 7) :
  teacher_volunteers = 13 :=
by
  sorry

end NUMINAMATH_GPT_num_teacher_volunteers_l404_40426


namespace NUMINAMATH_GPT_intersection_M_N_l404_40472

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1} :=
  by sorry

end NUMINAMATH_GPT_intersection_M_N_l404_40472


namespace NUMINAMATH_GPT_gcd_45_75_l404_40423

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_GPT_gcd_45_75_l404_40423


namespace NUMINAMATH_GPT_rope_length_91_4_l404_40435

noncomputable def total_rope_length (n : ℕ) (d : ℕ) (pi_val : Real) : Real :=
  let linear_segments := 6 * d
  let arc_length := (d * pi_val / 3) * 6
  let total_length_per_tie := linear_segments + arc_length
  total_length_per_tie * 2

theorem rope_length_91_4 :
  total_rope_length 7 5 3.14 = 91.4 :=
by
  sorry

end NUMINAMATH_GPT_rope_length_91_4_l404_40435


namespace NUMINAMATH_GPT_combined_savings_after_four_weeks_l404_40419

-- Definitions based on problem conditions
def hourly_wage : ℕ := 10
def daily_hours : ℕ := 10
def days_per_week : ℕ := 5
def weeks : ℕ := 4

def robby_saving_ratio : ℚ := 2/5
def jaylene_saving_ratio : ℚ := 3/5
def miranda_saving_ratio : ℚ := 1/2

-- Definitions derived from the conditions
def daily_earnings : ℕ := hourly_wage * daily_hours
def total_working_days : ℕ := days_per_week * weeks
def monthly_earnings : ℕ := daily_earnings * total_working_days

def robby_savings : ℚ := robby_saving_ratio * monthly_earnings
def jaylene_savings : ℚ := jaylene_saving_ratio * monthly_earnings
def miranda_savings : ℚ := miranda_saving_ratio * monthly_earnings

def total_savings : ℚ := robby_savings + jaylene_savings + miranda_savings

-- The main theorem to prove
theorem combined_savings_after_four_weeks :
  total_savings = 3000 := by sorry

end NUMINAMATH_GPT_combined_savings_after_four_weeks_l404_40419


namespace NUMINAMATH_GPT_length_of_ln_l404_40471

theorem length_of_ln (sin_N_eq : Real.sin angle_N = 3 / 5) (LM_eq : length_LM = 15) :
  length_LN = 25 :=
sorry

end NUMINAMATH_GPT_length_of_ln_l404_40471


namespace NUMINAMATH_GPT_percentage_increase_is_20_percent_l404_40462

noncomputable def SP : ℝ := 8600
noncomputable def CP : ℝ := 7166.67
noncomputable def percentageIncrease : ℝ := ((SP - CP) / CP) * 100

theorem percentage_increase_is_20_percent : percentageIncrease = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_is_20_percent_l404_40462


namespace NUMINAMATH_GPT_function_periodicity_l404_40442

variable {R : Type*} [Ring R]

def periodic_function (f : R → R) (k : R) : Prop :=
  ∀ x : R, f (x + 4*k) = f x

theorem function_periodicity {f : ℝ → ℝ} {k : ℝ} (h : ∀ x, f (x + k) * (1 - f x) = 1 + f x) (hk : k ≠ 0) : 
  periodic_function f k :=
sorry

end NUMINAMATH_GPT_function_periodicity_l404_40442


namespace NUMINAMATH_GPT_alpha_plus_beta_eq_118_l404_40403

theorem alpha_plus_beta_eq_118 (α β : ℝ) (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96 * x + 2209) / (x^2 + 63 * x - 3969)) : α + β = 118 :=
by
  sorry

end NUMINAMATH_GPT_alpha_plus_beta_eq_118_l404_40403


namespace NUMINAMATH_GPT_proof_problem_l404_40457

theorem proof_problem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y * z)) + (y / (1 + z * x)) + (z / (1 + x * y)) ∧ 
  (x / (1 + y * z)) + (y / (1 + z * x)) + (z / (1 + x * y)) ≤ Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l404_40457


namespace NUMINAMATH_GPT_range_of_m_l404_40431

theorem range_of_m (m : ℝ) : 
  (m - 1 < 0 ∧ 4 * m - 3 > 0) → (3 / 4 < m ∧ m < 1) := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l404_40431


namespace NUMINAMATH_GPT_find_subtracted_number_l404_40484

theorem find_subtracted_number (x y : ℕ) (h1 : 6 * x - 5 * x = 5) (h2 : (30 - y) * 4 = (25 - y) * 5) : y = 5 :=
sorry

end NUMINAMATH_GPT_find_subtracted_number_l404_40484


namespace NUMINAMATH_GPT_exists_bijection_l404_40415

-- Define the non-negative integers set
def N_0 := {n : ℕ // n ≥ 0}

-- Translation of the equivalent proof statement into Lean
theorem exists_bijection (f : N_0 → N_0) :
  (∀ m n : N_0, f ⟨3 * m.val * n.val + m.val + n.val, sorry⟩ = 
   ⟨4 * (f m).val * (f n).val + (f m).val + (f n).val, sorry⟩) :=
sorry

end NUMINAMATH_GPT_exists_bijection_l404_40415


namespace NUMINAMATH_GPT_parallel_lines_m_l404_40402

theorem parallel_lines_m (m : ℝ) :
  (∀ (x y : ℝ), 2 * x + (m + 1) * y + 4 = 0) ∧ (∀ (x y : ℝ), m * x + 3 * y - 2 = 0) →
  (m = -3 ∨ m = 2) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_m_l404_40402


namespace NUMINAMATH_GPT_total_workers_l404_40410

-- Definitions for the conditions in the problem
def avg_salary_all : ℝ := 8000
def num_technicians : ℕ := 7
def avg_salary_technicians : ℝ := 18000
def avg_salary_non_technicians : ℝ := 6000

-- Main theorem stating the total number of workers
theorem total_workers (W : ℕ) :
  (7 * avg_salary_technicians + (W - 7) * avg_salary_non_technicians = W * avg_salary_all) → W = 42 :=
by
  sorry

end NUMINAMATH_GPT_total_workers_l404_40410


namespace NUMINAMATH_GPT_range_of_expression_l404_40486

theorem range_of_expression (a b : ℝ) (h : a^2 + b^2 = 9) :
  -21 ≤ a^2 + b^2 - 6*a - 8*b ∧ a^2 + b^2 - 6*a - 8*b ≤ 39 :=
by
  sorry

end NUMINAMATH_GPT_range_of_expression_l404_40486


namespace NUMINAMATH_GPT_cost_per_meter_of_fencing_l404_40490

/-- The sides of the rectangular field -/
def sides_ratio (length width : ℕ) : Prop := 3 * width = 4 * length

/-- The area of the rectangular field -/
def area (length width area : ℕ) : Prop := length * width = area

/-- The cost per meter of fencing -/
def cost_per_meter (total_cost perimeter : ℕ) : ℕ := total_cost * 100 / perimeter

/-- Prove that the cost per meter of fencing the field in paise is 25 given:
 1) The sides of a rectangular field are in the ratio 3:4.
 2) The area of the field is 8112 sq. m.
 3) The total cost of fencing the field is 91 rupees. -/
theorem cost_per_meter_of_fencing
  (length width perimeter : ℕ) 
  (h1 : sides_ratio length width)
  (h2 : area length width 8112)
  (h3 : perimeter = 2 * (length + width))
  (total_cost : ℕ)
  (h4 : total_cost = 91) :
  cost_per_meter total_cost perimeter = 25 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_meter_of_fencing_l404_40490


namespace NUMINAMATH_GPT_max_length_MN_l404_40412

theorem max_length_MN (p : ℝ) (h a b c r : ℝ)
  (h_perimeter : a + b + c = 2 * p)
  (h_tangent : r = (a * h) / (2 * p))
  (h_parallel : ∀ h r : ℝ, ∃ k : ℝ, MN = k * (1 - 2 * r / h)) :
  ∀ k : ℝ, MN = (p / 4) :=
sorry

end NUMINAMATH_GPT_max_length_MN_l404_40412


namespace NUMINAMATH_GPT_binomial_product_l404_40470

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end NUMINAMATH_GPT_binomial_product_l404_40470


namespace NUMINAMATH_GPT_find_x_values_l404_40413

def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem find_x_values :
  {x : ℝ | f (f x) = f x} = {0, 4, 5, -1} :=
by
  sorry

end NUMINAMATH_GPT_find_x_values_l404_40413


namespace NUMINAMATH_GPT_Helen_taller_than_Amy_l404_40494

-- Definitions from conditions
def Angela_height : ℕ := 157
def Amy_height : ℕ := 150
def Helen_height := Angela_height - 4

-- Question as a theorem
theorem Helen_taller_than_Amy : Helen_height - Amy_height = 3 := by
  sorry

end NUMINAMATH_GPT_Helen_taller_than_Amy_l404_40494


namespace NUMINAMATH_GPT_system_solution_equation_solution_l404_40406

-- Proof problem for the first system of equations
theorem system_solution (x y : ℝ) : 
  (2 * x + 3 * y = 8) ∧ (3 * x - 5 * y = -7) → (x = 1 ∧ y = 2) :=
by sorry

-- Proof problem for the second equation
theorem equation_solution (x : ℝ) : 
  ((x - 2) / (x + 2) - 12 / (x^2 - 4) = 1) → (x = -1) :=
by sorry

end NUMINAMATH_GPT_system_solution_equation_solution_l404_40406


namespace NUMINAMATH_GPT_smallest_m_n_sum_l404_40475

theorem smallest_m_n_sum (m n : ℕ) (hmn : m > n) (div_condition : 4900 ∣ (2023 ^ m - 2023 ^ n)) : m + n = 24 :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_n_sum_l404_40475


namespace NUMINAMATH_GPT_solve_fractional_eq_l404_40467

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) : (4 / (x - 2) = 2 / x) → (x = -2) :=
by 
  sorry

end NUMINAMATH_GPT_solve_fractional_eq_l404_40467


namespace NUMINAMATH_GPT_kimmie_earnings_l404_40416

theorem kimmie_earnings (K : ℚ) (h : (1/2 : ℚ) * K + (1/3 : ℚ) * K = 375) : K = 450 := 
by
  sorry

end NUMINAMATH_GPT_kimmie_earnings_l404_40416


namespace NUMINAMATH_GPT_problem_l404_40478

def binom (n k : ℕ) : ℕ := n.choose k

def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem problem : binom 10 3 * perm 8 2 = 6720 := by
  sorry

end NUMINAMATH_GPT_problem_l404_40478


namespace NUMINAMATH_GPT_sum_tripled_numbers_l404_40496

theorem sum_tripled_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_tripled_numbers_l404_40496


namespace NUMINAMATH_GPT_weng_total_earnings_l404_40441

noncomputable def weng_earnings_usd : ℝ :=
  let usd_per_hr_job1 : ℝ := 12
  let eur_per_hr_job2 : ℝ := 13
  let gbp_per_hr_job3 : ℝ := 9
  let hr_job1 : ℝ := 2 + 15 / 60
  let hr_job2 : ℝ := 1 + 40 / 60
  let hr_job3 : ℝ := 3 + 10 / 60
  let usd_to_eur : ℝ := 0.85
  let usd_to_gbp : ℝ := 0.76
  let eur_to_usd : ℝ := 1.18
  let gbp_to_usd : ℝ := 1.32
  let earnings_job1 : ℝ := usd_per_hr_job1 * hr_job1
  let earnings_job2_eur : ℝ := eur_per_hr_job2 * hr_job2
  let earnings_job2_usd : ℝ := earnings_job2_eur * eur_to_usd
  let earnings_job3_gbp : ℝ := gbp_per_hr_job3 * hr_job3
  let earnings_job3_usd : ℝ := earnings_job3_gbp * gbp_to_usd
  earnings_job1 + earnings_job2_usd + earnings_job3_usd

theorem weng_total_earnings : weng_earnings_usd = 90.19 :=
by
  sorry

end NUMINAMATH_GPT_weng_total_earnings_l404_40441


namespace NUMINAMATH_GPT_area_of_three_layer_cover_l404_40444

-- Define the hall dimensions
def hall_width : ℕ := 10
def hall_height : ℕ := 10

-- Define the dimensions of the carpets
def carpet1_width : ℕ := 6
def carpet1_height : ℕ := 8
def carpet2_width : ℕ := 6
def carpet2_height : ℕ := 6
def carpet3_width : ℕ := 5
def carpet3_height : ℕ := 7

-- Theorem to prove area covered by the carpets in three layers
theorem area_of_three_layer_cover : 
  ∀ (w1 w2 w3 h1 h2 h3 : ℕ), w1 = carpet1_width → h1 = carpet1_height → w2 = carpet2_width → h2 = carpet2_height → w3 = carpet3_width → h3 = carpet3_height → 
  ∃ (area : ℕ), area = 6 :=
by
  intros w1 w2 w3 h1 h2 h3 hw1 hw2 hw3 hh1 hh2 hh3
  exact ⟨6, rfl⟩

#check area_of_three_layer_cover

end NUMINAMATH_GPT_area_of_three_layer_cover_l404_40444


namespace NUMINAMATH_GPT_number_of_outcomes_l404_40407

-- Define the conditions
def students : Nat := 4
def events : Nat := 3

-- Define the problem: number of possible outcomes for the champions
theorem number_of_outcomes : students ^ events = 64 :=
by sorry

end NUMINAMATH_GPT_number_of_outcomes_l404_40407


namespace NUMINAMATH_GPT_rachel_math_homework_pages_l404_40414

theorem rachel_math_homework_pages (M : ℕ) 
  (h1 : 23 = M + (M + 3)) : M = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_rachel_math_homework_pages_l404_40414


namespace NUMINAMATH_GPT_slightly_used_crayons_correct_l404_40454

def total_crayons : ℕ := 120
def new_crayons : ℕ := total_crayons / 3
def broken_crayons : ℕ := (total_crayons * 20) / 100
def slightly_used_crayons : ℕ := total_crayons - new_crayons - broken_crayons

theorem slightly_used_crayons_correct : slightly_used_crayons = 56 := sorry

end NUMINAMATH_GPT_slightly_used_crayons_correct_l404_40454


namespace NUMINAMATH_GPT_calculate_square_difference_l404_40460

theorem calculate_square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_calculate_square_difference_l404_40460


namespace NUMINAMATH_GPT_weighted_average_correct_l404_40408

noncomputable def weightedAverage := 
  (5 * (3/5 : ℝ) + 3 * (4/9 : ℝ) + 8 * 0.45 + 4 * 0.067) / (5 + 3 + 8 + 4)

theorem weighted_average_correct :
  weightedAverage = 0.41 :=
by
  sorry

end NUMINAMATH_GPT_weighted_average_correct_l404_40408


namespace NUMINAMATH_GPT_hotel_made_correct_revenue_l404_40458

noncomputable def hotelRevenue : ℕ :=
  let totalRooms := 260
  let doubleRooms := 196
  let singleRoomCost := 35
  let doubleRoomCost := 60
  let singleRooms := totalRooms - doubleRooms
  let revenueSingleRooms := singleRooms * singleRoomCost
  let revenueDoubleRooms := doubleRooms * doubleRoomCost
  revenueSingleRooms + revenueDoubleRooms

theorem hotel_made_correct_revenue :
  hotelRevenue = 14000 := by
  sorry

end NUMINAMATH_GPT_hotel_made_correct_revenue_l404_40458


namespace NUMINAMATH_GPT_total_cost_is_80_l404_40437

-- Conditions
def cost_flour := 3 * 3
def cost_eggs := 3 * 10
def cost_milk := 7 * 5
def cost_baking_soda := 2 * 3

-- Question and proof requirement
theorem total_cost_is_80 : cost_flour + cost_eggs + cost_milk + cost_baking_soda = 80 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_80_l404_40437


namespace NUMINAMATH_GPT_problem_statement_l404_40452

-- Define the operation #
def op_hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- The main theorem statement
theorem problem_statement (a b : ℕ) (h1 : op_hash a b = 100) : (a + b) + 6 = 11 := 
sorry

end NUMINAMATH_GPT_problem_statement_l404_40452


namespace NUMINAMATH_GPT_license_plate_count_l404_40488

-- Define the conditions
def num_digits : ℕ := 5
def num_letters : ℕ := 2
def digit_choices : ℕ := 10
def letter_choices : ℕ := 26

-- Define the statement to prove the total number of distinct licenses plates
theorem license_plate_count : 
  (digit_choices ^ num_digits) * (letter_choices ^ num_letters) * 2 = 2704000 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_count_l404_40488


namespace NUMINAMATH_GPT_find_scooters_l404_40421

variables (b t s : ℕ)

theorem find_scooters (h1 : b + t + s = 13) (h2 : 2 * b + 3 * t + 2 * s = 30) : s = 9 :=
sorry

end NUMINAMATH_GPT_find_scooters_l404_40421


namespace NUMINAMATH_GPT_find_difference_square_l404_40436

theorem find_difference_square (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 6) :
  (x - y)^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_square_l404_40436


namespace NUMINAMATH_GPT_integer_between_sqrt3_add1_and_sqrt11_l404_40422

theorem integer_between_sqrt3_add1_and_sqrt11 :
  (∀ x, (1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2) ∧ (3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4) → (2 < Real.sqrt 3 + 1 ∧ Real.sqrt 3 + 1 < 3) ∧ (3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4) ∧ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_integer_between_sqrt3_add1_and_sqrt11_l404_40422


namespace NUMINAMATH_GPT_soda_costs_94_cents_l404_40463

theorem soda_costs_94_cents (b s: ℤ) (h1 : 4 * b + 3 * s = 500) (h2 : 3 * b + 4 * s = 540) : s = 94 := 
by
  sorry

end NUMINAMATH_GPT_soda_costs_94_cents_l404_40463


namespace NUMINAMATH_GPT_ab_proof_l404_40434

theorem ab_proof (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 90 < a + b) (h4 : a + b < 99) 
  (h5 : 0.9 < (a : ℝ) / b) (h6 : (a : ℝ) / b < 0.91) : a * b = 2346 :=
sorry

end NUMINAMATH_GPT_ab_proof_l404_40434


namespace NUMINAMATH_GPT_smallest_sum_l404_40468

theorem smallest_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_neq : x ≠ y)
  (h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 18) : x + y = 75 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_l404_40468


namespace NUMINAMATH_GPT_remainder_of_4123_div_by_32_l404_40481

theorem remainder_of_4123_div_by_32 : 
  ∃ r, 0 ≤ r ∧ r < 32 ∧ 4123 = 32 * (4123 / 32) + r ∧ r = 27 := by
  sorry

end NUMINAMATH_GPT_remainder_of_4123_div_by_32_l404_40481
