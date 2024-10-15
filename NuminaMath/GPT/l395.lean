import Mathlib

namespace NUMINAMATH_GPT_solution_set_l395_39546

variable {f : ℝ → ℝ}

-- Define that f is an odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define that f is decreasing on positive reals
def decreasing_on_pos_reals (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f y < f x

-- Given conditions
axiom f_odd : odd_function f
axiom f_decreasing : decreasing_on_pos_reals f
axiom f_at_two_zero : f 2 = 0

-- Main theorem statement
theorem solution_set : { x : ℝ | (x - 1) * f (x - 1) > 0 } = { x | x < -1 } ∪ { x | x > 3 } :=
sorry

end NUMINAMATH_GPT_solution_set_l395_39546


namespace NUMINAMATH_GPT_find_bc_l395_39532

theorem find_bc (b c : ℤ) (h : ∀ x : ℝ, x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = 1 ∨ x = 2) :
  b = -3 ∧ c = 2 := by
  sorry

end NUMINAMATH_GPT_find_bc_l395_39532


namespace NUMINAMATH_GPT_smallest_range_l395_39593

-- Define the conditions
def estate (A B C : ℝ) : Prop :=
  A = 20000 ∧
  abs (A - B) > 0.3 * A ∧
  abs (A - C) > 0.3 * A ∧
  abs (B - C) > 0.3 * A

-- Define the statement to prove
theorem smallest_range (A B C : ℝ) (h : estate A B C) : 
  ∃ r : ℝ, r = 12000 :=
sorry

end NUMINAMATH_GPT_smallest_range_l395_39593


namespace NUMINAMATH_GPT_problem1_problem2_l395_39542

-- Define the sets P and Q
def set_P : Set ℝ := {x | 2 * x^2 - 5 * x - 3 < 0}
def set_Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Problem (1): P ∩ Q = Q implies a ∈ (-1/2, 2)
theorem problem1 (a : ℝ) : (set_Q a) ⊆ set_P → -1/2 < a ∧ a < 2 :=
by 
  sorry

-- Problem (2): P ∩ Q = ∅ implies a ∈ (-∞, -3/2] ∪ [3, ∞)
theorem problem2 (a : ℝ) : (set_Q a) ∩ set_P = ∅ → a ≤ -3/2 ∨ a ≥ 3 :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l395_39542


namespace NUMINAMATH_GPT_melanie_attended_games_l395_39543

-- Define the total number of football games and the number of games missed by Melanie.
def total_games := 7
def missed_games := 4

-- Define what we need to prove: the number of games attended by Melanie.
theorem melanie_attended_games : total_games - missed_games = 3 := 
by
  sorry

end NUMINAMATH_GPT_melanie_attended_games_l395_39543


namespace NUMINAMATH_GPT_bears_total_l395_39547

-- Define the number of each type of bear
def brown_bears : ℕ := 15
def white_bears : ℕ := 24
def black_bears : ℕ := 27
def polar_bears : ℕ := 12
def grizzly_bears : ℕ := 18

-- Define the total number of bears
def total_bears : ℕ := brown_bears + white_bears + black_bears + polar_bears + grizzly_bears

-- The theorem stating the total number of bears is 96
theorem bears_total : total_bears = 96 :=
by
  -- The proof is omitted here
  sorry

end NUMINAMATH_GPT_bears_total_l395_39547


namespace NUMINAMATH_GPT_determine_n_l395_39526

open Function

noncomputable def coeff_3 (n : ℕ) : ℕ :=
  2^(n-2) * Nat.choose n 2

noncomputable def coeff_4 (n : ℕ) : ℕ :=
  2^(n-3) * Nat.choose n 3

theorem determine_n (n : ℕ) (b3_eq_2b4 : coeff_3 n = 2 * coeff_4 n) : n = 5 :=
  sorry

end NUMINAMATH_GPT_determine_n_l395_39526


namespace NUMINAMATH_GPT_range_of_a_l395_39578

open Set

variable {a : ℝ} 

def M (a : ℝ) : Set ℝ := {x : ℝ | -4 * x + 4 * a < 0 }

theorem range_of_a (hM : 2 ∉ M a) : a ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l395_39578


namespace NUMINAMATH_GPT_correct_weights_l395_39500

def weight (item : String) : Nat :=
  match item with
  | "Banana" => 140
  | "Pear" => 120
  | "Melon" => 1500
  | "Tomato" => 150
  | "Apple" => 170
  | _ => 0

theorem correct_weights :
  weight "Banana" = 140 ∧
  weight "Pear" = 120 ∧
  weight "Melon" = 1500 ∧
  weight "Tomato" = 150 ∧
  weight "Apple" = 170 ∧
  (weight "Melon" > weight "Pear") ∧
  (weight "Melon" < weight "Tomato") :=
by
  sorry

end NUMINAMATH_GPT_correct_weights_l395_39500


namespace NUMINAMATH_GPT_count_square_free_integers_l395_39531

def square_free_in_range_2_to_199 : Nat :=
  91

theorem count_square_free_integers :
  ∃ n : Nat, n = 91 ∧
  ∀ m : Nat, 2 ≤ m ∧ m < 200 →
  (∀ k : Nat, k^2 ∣ m → k^2 = 1) :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_count_square_free_integers_l395_39531


namespace NUMINAMATH_GPT_ilya_incorrect_l395_39580

theorem ilya_incorrect (s t : ℝ) : ¬ (s + t = s * t ∧ s * t = s / t) :=
by
  sorry

end NUMINAMATH_GPT_ilya_incorrect_l395_39580


namespace NUMINAMATH_GPT_planting_area_correct_l395_39570

def garden_area : ℕ := 18 * 14
def pond_area : ℕ := 4 * 2
def flower_bed_area : ℕ := (1 / 2) * 3 * 2
def planting_area : ℕ := garden_area - pond_area - flower_bed_area

theorem planting_area_correct : planting_area = 241 := by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_planting_area_correct_l395_39570


namespace NUMINAMATH_GPT_matrix_determinant_l395_39595

variable {a b c d : ℝ}
variable (h : a * d - b * c = 4)

theorem matrix_determinant :
  (a * (7 * c + 3 * d) - c * (7 * a + 3 * b)) = 12 := by
  sorry

end NUMINAMATH_GPT_matrix_determinant_l395_39595


namespace NUMINAMATH_GPT_solution_exists_l395_39579

variable (x y : ℝ)

noncomputable def condition (x y : ℝ) : Prop :=
  (3 + 5 * x = -4 + 6 * y) ∧ (2 + (-6) * x = 6 + 8 * y)

theorem solution_exists : ∃ (x y : ℝ), condition x y ∧ x = -20 / 19 ∧ y = 11 / 38 := 
  by
  sorry

end NUMINAMATH_GPT_solution_exists_l395_39579


namespace NUMINAMATH_GPT_cyclist_problem_l395_39521

theorem cyclist_problem (MP NP : ℝ) (h1 : NP = MP + 30) (h2 : ∀ t : ℝ, t*MP = 10*t) 
  (h3 : ∀ t : ℝ, t*NP = 10*t) 
  (h4 : ∀ t : ℝ, t*MP = 42 → t*(MP + 30) = t*42 - 1/3) : 
  MP = 180 := 
sorry

end NUMINAMATH_GPT_cyclist_problem_l395_39521


namespace NUMINAMATH_GPT_intersect_condition_l395_39510

theorem intersect_condition (m : ℕ) (h : m ≠ 0) : 
  (∃ x y : ℝ, (3 * x - 2 * y = 0) ∧ ((x - m)^2 + y^2 = 1)) → m = 1 :=
by 
  sorry

end NUMINAMATH_GPT_intersect_condition_l395_39510


namespace NUMINAMATH_GPT_radius_of_inner_circle_l395_39587

theorem radius_of_inner_circle (R a x : ℝ) (hR : 0 < R) (ha : 0 ≤ a) (haR : a < R) :
  (a ≠ R ∧ a ≠ 0) → x = (R^2 - a^2) / (2 * R) :=
by
  sorry

end NUMINAMATH_GPT_radius_of_inner_circle_l395_39587


namespace NUMINAMATH_GPT_regular_price_of_one_tire_l395_39527

theorem regular_price_of_one_tire
  (x : ℝ) -- Define the variable \( x \) as the regular price of one tire
  (h1 : 3 * x + 10 = 250) -- Set up the equation based on the condition

  : x = 80 := 
sorry

end NUMINAMATH_GPT_regular_price_of_one_tire_l395_39527


namespace NUMINAMATH_GPT_value_of_k_l395_39507

theorem value_of_k (k : ℝ) : 
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ p/q = 3/2 ∧ p + q = -10 ∧ p * q = k) → k = 24 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_k_l395_39507


namespace NUMINAMATH_GPT_find_original_price_l395_39558

theorem find_original_price (x y : ℝ) 
  (h1 : 60 * x + 75 * y = 2700)
  (h2 : 60 * 0.85 * x + 75 * 0.90 * y = 2370) : 
  x = 20 ∧ y = 20 :=
sorry

end NUMINAMATH_GPT_find_original_price_l395_39558


namespace NUMINAMATH_GPT_sum_a_c_eq_13_l395_39550

noncomputable def conditions (a b c d k : ℤ) :=
  d = a * b * c ∧
  1 < a ∧ a < b ∧ b < c ∧
  233 = d * k + 79

theorem sum_a_c_eq_13 (a b c d k : ℤ) (h : conditions a b c d k) : a + c = 13 := by
  sorry

end NUMINAMATH_GPT_sum_a_c_eq_13_l395_39550


namespace NUMINAMATH_GPT_greatest_common_divisor_of_three_common_divisors_l395_39596

theorem greatest_common_divisor_of_three_common_divisors (m : ℕ) :
  (∀ d, d ∣ 126 ∧ d ∣ m → d = 1 ∨ d = 3 ∨ d = 9) →
  gcd 126 m = 9 := 
sorry

end NUMINAMATH_GPT_greatest_common_divisor_of_three_common_divisors_l395_39596


namespace NUMINAMATH_GPT_fouad_age_l395_39538

theorem fouad_age (F : ℕ) (Ahmed_current_age : ℕ) (H : Ahmed_current_age = 11) (H2 : F + 4 = 2 * Ahmed_current_age) : F = 18 :=
by
  -- We do not need to write the proof steps, just a placeholder.
  sorry

end NUMINAMATH_GPT_fouad_age_l395_39538


namespace NUMINAMATH_GPT_f_1988_eq_1988_l395_39529

noncomputable def f (n : ℕ) : ℕ := sorry

axiom f_f_eq_add (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (f m + f n) = m + n

theorem f_1988_eq_1988 : f 1988 = 1988 := 
by
  sorry

end NUMINAMATH_GPT_f_1988_eq_1988_l395_39529


namespace NUMINAMATH_GPT_parabola_vertex_sum_l395_39566

variable (a b c : ℝ)

def parabola_eq (x y : ℝ) : Prop :=
  x = a * y^2 + b * y + c

def vertex (v : ℝ × ℝ) : Prop :=
  v = (-3, 2)

def passes_through (p : ℝ × ℝ) : Prop :=
  p = (-1, 0)

theorem parabola_vertex_sum :
  ∀ (a b c : ℝ),
  (∃ v : ℝ × ℝ, vertex v) ∧
  (∃ p : ℝ × ℝ, passes_through p) →
  a + b + c = -7/2 :=
by
  intros a b c
  intro conditions
  sorry

end NUMINAMATH_GPT_parabola_vertex_sum_l395_39566


namespace NUMINAMATH_GPT_square_floor_tile_count_l395_39535

theorem square_floor_tile_count (n : ℕ) (h1 : 2 * n - 1 = 25) : n^2 = 169 :=
by
  sorry

end NUMINAMATH_GPT_square_floor_tile_count_l395_39535


namespace NUMINAMATH_GPT_transform_uniform_random_l395_39577

theorem transform_uniform_random (a_1 : ℝ) (h : 0 ≤ a_1 ∧ a_1 ≤ 1) : -2 ≤ a_1 * 8 - 2 ∧ a_1 * 8 - 2 ≤ 6 :=
by sorry

end NUMINAMATH_GPT_transform_uniform_random_l395_39577


namespace NUMINAMATH_GPT_number_of_bowls_l395_39519

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_number_of_bowls_l395_39519


namespace NUMINAMATH_GPT_gasoline_used_by_car_l395_39534

noncomputable def total_gasoline_used (gasoline_per_km : ℝ) (duration_hours : ℝ) (speed_kmh : ℝ) : ℝ :=
  gasoline_per_km * duration_hours * speed_kmh

theorem gasoline_used_by_car :
  total_gasoline_used 0.14 (2 + 0.5) 93.6 = 32.76 := sorry

end NUMINAMATH_GPT_gasoline_used_by_car_l395_39534


namespace NUMINAMATH_GPT_expand_binomial_square_l395_39559

variables (x : ℝ)

theorem expand_binomial_square (x : ℝ) : (2 - x) ^ 2 = 4 - 4 * x + x ^ 2 := 
sorry

end NUMINAMATH_GPT_expand_binomial_square_l395_39559


namespace NUMINAMATH_GPT_value_A_minus_B_l395_39554

-- Conditions definitions
def A : ℕ := (1 * 1000) + (16 * 100) + (28 * 10)
def B : ℕ := 355 + 245 * 3

-- Theorem statement
theorem value_A_minus_B : A - B = 1790 := by
  sorry

end NUMINAMATH_GPT_value_A_minus_B_l395_39554


namespace NUMINAMATH_GPT_Option_C_correct_l395_39591

theorem Option_C_correct (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = - x * y^2 :=
by
  sorry

end NUMINAMATH_GPT_Option_C_correct_l395_39591


namespace NUMINAMATH_GPT_problem_statement_l395_39564

noncomputable def f (x : ℝ) : ℝ := x - Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log x / x

theorem problem_statement (x : ℝ) (h : 0 < x ∧ x ≤ Real.exp 1) : 
  f x > g x + 1/2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l395_39564


namespace NUMINAMATH_GPT_y_intercept_with_z_3_l395_39590

theorem y_intercept_with_z_3 : 
  ∀ x y : ℝ, (4 * x + 6 * y - 2 * 3 = 24) → (x = 0) → y = 5 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_y_intercept_with_z_3_l395_39590


namespace NUMINAMATH_GPT_ned_defuse_time_l395_39539

theorem ned_defuse_time (flights_total time_per_flight bomb_time time_spent : ℕ) (h1 : flights_total = 20) (h2 : time_per_flight = 11) (h3 : bomb_time = 72) (h4 : time_spent = 165) :
  bomb_time - (flights_total * time_per_flight - time_spent) / time_per_flight * time_per_flight = 17 := by
  sorry

end NUMINAMATH_GPT_ned_defuse_time_l395_39539


namespace NUMINAMATH_GPT_subtraction_addition_example_l395_39544

theorem subtraction_addition_example :
  1500000000000 - 877888888888 + 123456789012 = 745567900124 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_addition_example_l395_39544


namespace NUMINAMATH_GPT_find_x_l395_39536

theorem find_x (x : ℤ) (h : 5 * x - 28 = 232) : x = 52 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l395_39536


namespace NUMINAMATH_GPT_lines_perpendicular_if_one_perpendicular_and_one_parallel_l395_39516

def Line : Type := sorry  -- Define the type representing lines
def Plane : Type := sorry  -- Define the type representing planes

def is_perpendicular_to_plane (a : Line) (α : Plane) : Prop := sorry  -- Definition for a line being perpendicular to a plane
def is_parallel_to_plane (b : Line) (α : Plane) : Prop := sorry  -- Definition for a line being parallel to a plane
def is_perpendicular (a b : Line) : Prop := sorry  -- Definition for a line being perpendicular to another line

theorem lines_perpendicular_if_one_perpendicular_and_one_parallel 
  (a b : Line) (α : Plane) 
  (h1 : is_perpendicular_to_plane a α) 
  (h2 : is_parallel_to_plane b α) : 
  is_perpendicular a b := 
sorry

end NUMINAMATH_GPT_lines_perpendicular_if_one_perpendicular_and_one_parallel_l395_39516


namespace NUMINAMATH_GPT_combined_volume_of_all_cubes_l395_39594

/-- Lily has 4 cubes each with side length 3, Mark has 3 cubes each with side length 4,
    and Zoe has 2 cubes each with side length 5. Prove that the combined volume of all
    the cubes is 550. -/
theorem combined_volume_of_all_cubes 
  (lily_cubes : ℕ := 4) (lily_side_length : ℕ := 3)
  (mark_cubes : ℕ := 3) (mark_side_length : ℕ := 4)
  (zoe_cubes : ℕ := 2) (zoe_side_length : ℕ := 5) :
  (lily_cubes * lily_side_length ^ 3) + 
  (mark_cubes * mark_side_length ^ 3) + 
  (zoe_cubes * zoe_side_length ^ 3) = 550 :=
by
  have lily_volume : ℕ := lily_cubes * lily_side_length ^ 3
  have mark_volume : ℕ := mark_cubes * mark_side_length ^ 3
  have zoe_volume : ℕ := zoe_cubes * zoe_side_length ^ 3
  have total_volume : ℕ := lily_volume + mark_volume + zoe_volume
  sorry

end NUMINAMATH_GPT_combined_volume_of_all_cubes_l395_39594


namespace NUMINAMATH_GPT_angle_in_second_quadrant_l395_39581

theorem angle_in_second_quadrant (x : ℝ) (hx1 : Real.tan x < 0) (hx2 : Real.sin x - Real.cos x > 0) : 
  (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 2 ∨ x = 2 * k * Real.pi + 3 * Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_angle_in_second_quadrant_l395_39581


namespace NUMINAMATH_GPT_find_k_l395_39568

def S (n : ℕ) : ℤ := n^2 - 9 * n

def a (n : ℕ) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem find_k (k : ℕ) (h1 : 5 < a k) (h2 : a k < 8) : k = 8 := by
  sorry

end NUMINAMATH_GPT_find_k_l395_39568


namespace NUMINAMATH_GPT_elise_initial_dog_food_l395_39571

variable (initial_dog_food : ℤ)
variable (bought_first_bag : ℤ := 15)
variable (bought_second_bag : ℤ := 10)
variable (final_dog_food : ℤ := 40)

theorem elise_initial_dog_food :
  initial_dog_food + bought_first_bag + bought_second_bag = final_dog_food →
  initial_dog_food = 15 :=
by
  sorry

end NUMINAMATH_GPT_elise_initial_dog_food_l395_39571


namespace NUMINAMATH_GPT_product_of_first_nine_terms_l395_39557

-- Declare the geometric sequence and given condition
variable {α : Type*} [Field α]
variable {a : ℕ → α}
variable (r : α) (a1 : α)

-- Define that the sequence is geometric
def is_geometric_sequence (a : ℕ → α) (r : α) (a1 : α) : Prop :=
  ∀ n : ℕ, a n = a1 * r ^ n

-- Given a_5 = -2 in the sequence
def geometric_sequence_with_a5 (a : ℕ → α) (r : α) (a1 : α) : Prop :=
  is_geometric_sequence a r a1 ∧ a 5 = -2

-- Prove that the product of the first 9 terms is -512
theorem product_of_first_nine_terms 
  (a : ℕ → α) 
  (r : α) 
  (a₁ : α) 
  (h : geometric_sequence_with_a5 a r a₁) : 
  (a 0) * (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) = -512 := 
by
  sorry

end NUMINAMATH_GPT_product_of_first_nine_terms_l395_39557


namespace NUMINAMATH_GPT_solve_integer_divisibility_l395_39540

theorem solve_integer_divisibility :
  {n : ℕ | n < 589 ∧ 589 ∣ (n^2 + n + 1)} = {49, 216, 315, 482} :=
by
  sorry

end NUMINAMATH_GPT_solve_integer_divisibility_l395_39540


namespace NUMINAMATH_GPT_flowers_per_day_l395_39551

-- Definitions for conditions
def total_flowers := 360
def days := 6

-- Proof that the number of flowers Miriam can take care of in one day is 60
theorem flowers_per_day : total_flowers / days = 60 := by
  sorry

end NUMINAMATH_GPT_flowers_per_day_l395_39551


namespace NUMINAMATH_GPT_total_male_students_combined_l395_39599

/-- The number of first-year students is 695, of which 329 are female students. 
If the number of male second-year students is 254, prove that the number of male students in the first-year and second-year combined is 620. -/
theorem total_male_students_combined (first_year_students : ℕ) (female_first_year_students : ℕ) (male_second_year_students : ℕ) :
  first_year_students = 695 →
  female_first_year_students = 329 →
  male_second_year_students = 254 →
  (first_year_students - female_first_year_students + male_second_year_students) = 620 := by
  sorry

end NUMINAMATH_GPT_total_male_students_combined_l395_39599


namespace NUMINAMATH_GPT_mr_bird_speed_to_work_l395_39506

theorem mr_bird_speed_to_work (
  d t : ℝ
) (h1 : d = 45 * (t + 4 / 60)) 
  (h2 : d = 55 * (t - 2 / 60))
  (h3 : t = 29 / 60)
  (d_eq : d = 24.75) :
  (24.75 / (29 / 60)) = 51.207 := 
sorry

end NUMINAMATH_GPT_mr_bird_speed_to_work_l395_39506


namespace NUMINAMATH_GPT_train_passing_time_l395_39567

noncomputable def speed_in_m_per_s : ℝ := (60 * 1000) / 3600

variable (L : ℝ) (S : ℝ)
variable (train_length : L = 500)
variable (train_speed : S = speed_in_m_per_s)

theorem train_passing_time : L / S = 30 := by
  sorry

end NUMINAMATH_GPT_train_passing_time_l395_39567


namespace NUMINAMATH_GPT_flood_damage_conversion_l395_39560

-- Define the conversion rate and the damage in Indian Rupees as given
def rupees_to_pounds (rupees : ℕ) : ℕ := rupees / 75
def damage_in_rupees : ℕ := 45000000

-- Define the expected damage in British Pounds
def expected_damage_in_pounds : ℕ := 600000

-- The theorem to prove that the damage in British Pounds is as expected, given the conditions.
theorem flood_damage_conversion :
  rupees_to_pounds damage_in_rupees = expected_damage_in_pounds :=
by
  -- The proof goes here, but we'll use sorry to skip it as instructed.
  sorry

end NUMINAMATH_GPT_flood_damage_conversion_l395_39560


namespace NUMINAMATH_GPT_set_equality_l395_39545

open Set

variable (A : Set ℕ)

theorem set_equality (h1 : {1, 3} ⊆ A) (h2 : {1, 3} ∪ A = {1, 3, 5}) : A = {1, 3, 5} :=
sorry

end NUMINAMATH_GPT_set_equality_l395_39545


namespace NUMINAMATH_GPT_find_a11_times_a55_l395_39563

noncomputable def a_ij (i j : ℕ) : ℝ := 
  if i = 4 ∧ j = 1 then -2 else
  if i = 4 ∧ j = 3 then 10 else
  if i = 2 ∧ j = 4 then 4 else sorry

theorem find_a11_times_a55 
  (arithmetic_first_row : ∀ j, a_ij 1 (j + 1) = a_ij 1 1 + (j * 6))
  (geometric_columns : ∀ i j, a_ij (i + 1) j = a_ij 1 j * (2 ^ i) ∨ a_ij (i + 1) j = a_ij 1 j * ((-2) ^ i))
  (a24_eq_4 : a_ij 2 4 = 4)
  (a41_eq_neg2 : a_ij 4 1 = -2)
  (a43_eq_10 : a_ij 4 3 = 10) :
  a_ij 1 1 * a_ij 5 5 = -11 :=
by sorry

end NUMINAMATH_GPT_find_a11_times_a55_l395_39563


namespace NUMINAMATH_GPT_num_arithmetic_sequences_l395_39515

-- Definitions of the arithmetic sequence conditions
def is_arithmetic_sequence (a d n : ℕ) : Prop :=
  0 ≤ a ∧ 0 ≤ d ∧ n ≥ 3 ∧ 
  (∃ k : ℕ, k = 97 ∧ 
  (n * (2 * a + (n - 1) * d) = 2 * k ^ 2)) 

-- Prove that there are exactly 4 such sequences
theorem num_arithmetic_sequences : 
  ∃ (n : ℕ) (a d : ℕ), 
  is_arithmetic_sequence a d n ∧ 
  (n * (2 * a + (n - 1) * d) = 2 * 97^2) ∧ (
    (n = 97 ∧ ((a = 97 ∧ d = 0) ∨ (a = 49 ∧ d = 1) ∨ (a = 1 ∧ d = 2))) ∨
    (n = 97^2 ∧ a = 1 ∧ d = 0)
  ) :=
sorry

end NUMINAMATH_GPT_num_arithmetic_sequences_l395_39515


namespace NUMINAMATH_GPT_crayons_initial_total_l395_39589

theorem crayons_initial_total 
  (lost_given : ℕ) (left : ℕ) (initial : ℕ) 
  (h1 : lost_given = 70) (h2 : left = 183) : 
  initial = lost_given + left := 
by
  sorry

end NUMINAMATH_GPT_crayons_initial_total_l395_39589


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_17_with_condition_l395_39508

theorem smallest_positive_multiple_of_17_with_condition :
  ∃ k : ℕ, k > 0 ∧ (k % 17 = 0) ∧ (k - 3) % 101 = 0 ∧ k = 306 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_17_with_condition_l395_39508


namespace NUMINAMATH_GPT_value_of_x_l395_39502

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l395_39502


namespace NUMINAMATH_GPT_mn_plus_one_unequal_pos_integers_l395_39514

theorem mn_plus_one_unequal_pos_integers (m n : ℕ) 
  (S : Finset ℕ) (h_card : S.card = m * n + 1) :
  (∃ (b : Fin (m + 1) → ℕ), (∀ i j : Fin (m + 1), i ≠ j → ¬(b i ∣ b j)) ∧ (∀ i : Fin (m + 1), b i ∈ S)) ∨ 
  (∃ (a : Fin (n + 1) → ℕ), (∀ i : Fin n, a i ∣ a (i + 1)) ∧ (∀ i : Fin (n + 1), a i ∈ S)) :=
sorry

end NUMINAMATH_GPT_mn_plus_one_unequal_pos_integers_l395_39514


namespace NUMINAMATH_GPT_length_of_living_room_l395_39569

theorem length_of_living_room (width area : ℝ) (h_width : width = 14) (h_area : area = 215.6) :
  ∃ length : ℝ, length = 15.4 ∧ area = length * width :=
by
  sorry

end NUMINAMATH_GPT_length_of_living_room_l395_39569


namespace NUMINAMATH_GPT_obtuse_triangle_count_l395_39509

-- Definitions based on conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  a * a + b * b < c * c ∨ b * b + c * c < a * a ∨ c * c + a * a < b * b

-- Main conjecture to prove
theorem obtuse_triangle_count :
  ∃ (n : ℕ), n = 157 ∧
    ∀ (a b c : ℕ), 
      a <= 50 ∧ b <= 50 ∧ c <= 50 ∧ 
      is_arithmetic_sequence a b c ∧ 
      is_triangle a b c ∧ 
      is_obtuse_triangle a b c → 
    true := sorry

end NUMINAMATH_GPT_obtuse_triangle_count_l395_39509


namespace NUMINAMATH_GPT_minimum_sum_sequence_l395_39522

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S_n (n : ℕ) : ℤ := (n * (a_n 1 + a_n n)) / 2

theorem minimum_sum_sequence : ∃ n : ℕ, S_n n = (n - 24) * (n - 24) - 24 * 24 ∧ (∀ m : ℕ, S_n m ≥ S_n n) ∧ n = 24 := 
by {
  sorry -- Proof omitted
}

end NUMINAMATH_GPT_minimum_sum_sequence_l395_39522


namespace NUMINAMATH_GPT_total_amount_collected_in_paise_total_amount_collected_in_rupees_l395_39530

-- Definitions and conditions
def num_members : ℕ := 96
def contribution_per_member : ℕ := 96
def total_paise_collected : ℕ := num_members * contribution_per_member
def total_rupees_collected : ℚ := total_paise_collected / 100

-- Theorem stating the total amount collected
theorem total_amount_collected_in_paise :
  total_paise_collected = 9216 := by sorry

theorem total_amount_collected_in_rupees :
  total_rupees_collected = 92.16 := by sorry

end NUMINAMATH_GPT_total_amount_collected_in_paise_total_amount_collected_in_rupees_l395_39530


namespace NUMINAMATH_GPT_f_decreasing_l395_39572

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / x^2 + 3

theorem f_decreasing (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h : x1 < x2) : f x1 > f x2 := 
by
  sorry

end NUMINAMATH_GPT_f_decreasing_l395_39572


namespace NUMINAMATH_GPT_smallest_integer_ending_in_6_divisible_by_13_l395_39574

theorem smallest_integer_ending_in_6_divisible_by_13 (n : ℤ) (h1 : ∃ n : ℤ, 10 * n + 6 = x) (h2 : x % 13 = 0) : x = 26 :=
  sorry

end NUMINAMATH_GPT_smallest_integer_ending_in_6_divisible_by_13_l395_39574


namespace NUMINAMATH_GPT_symmetric_point_y_axis_l395_39562

def M : ℝ × ℝ := (-5, 2)
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
theorem symmetric_point_y_axis :
  symmetric_point M = (5, 2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_y_axis_l395_39562


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l395_39523

theorem equation1_solution : ∀ x : ℚ, x - 0.4 * x = 120 → x = 200 := by
  sorry

theorem equation2_solution : ∀ x : ℚ, 5 * x - 5/6 = 5/4 → x = 5/12 := by
  sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l395_39523


namespace NUMINAMATH_GPT_tom_seashells_left_l395_39573

def initial_seashells : ℕ := 5
def given_away_seashells : ℕ := 2

theorem tom_seashells_left : (initial_seashells - given_away_seashells) = 3 :=
by
  sorry

end NUMINAMATH_GPT_tom_seashells_left_l395_39573


namespace NUMINAMATH_GPT_monica_milk_l395_39504

theorem monica_milk (don_milk : ℚ) (rachel_fraction : ℚ) (monica_fraction : ℚ) (h_don : don_milk = 3 / 4)
  (h_rachel : rachel_fraction = 1 / 2) (h_monica : monica_fraction = 1 / 3) :
  monica_fraction * (rachel_fraction * don_milk) = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_monica_milk_l395_39504


namespace NUMINAMATH_GPT_determine_b_l395_39525

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 1 / (3 * x + b)
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_b (b : ℝ) :
    (∀ x : ℝ, f_inv (f x b) = x) ↔ b = -3 :=
by
  sorry

end NUMINAMATH_GPT_determine_b_l395_39525


namespace NUMINAMATH_GPT_polio_cases_in_1990_l395_39503

theorem polio_cases_in_1990 (c_1970 c_2000 : ℕ) (T : ℕ) (linear_decrease : ∀ t, c_1970 - (c_2000 * t) / T > 0):
  (c_1970 = 300000) → (c_2000 = 600) → (T = 30) → ∃ c_1990, c_1990 = 100400 :=
by
  intros
  sorry

end NUMINAMATH_GPT_polio_cases_in_1990_l395_39503


namespace NUMINAMATH_GPT_find_quadratic_function_l395_39553

def quad_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_quadratic_function : ∃ (a b c : ℝ), 
  (∀ x : ℝ, quad_function a b c x = 2 * x^2 + 4 * x - 1) ∧ 
  (quad_function a b c (-1) = -3) ∧ 
  (quad_function a b c 1 = 5) :=
sorry

end NUMINAMATH_GPT_find_quadratic_function_l395_39553


namespace NUMINAMATH_GPT_probability_red_bean_l395_39556

section ProbabilityRedBean

-- Initially, there are 5 red beans and 9 black beans in a bag.
def initial_red_beans : ℕ := 5
def initial_black_beans : ℕ := 9
def initial_total_beans : ℕ := initial_red_beans + initial_black_beans

-- Then, 3 red beans and 3 black beans are added to the bag.
def added_red_beans : ℕ := 3
def added_black_beans : ℕ := 3
def final_red_beans : ℕ := initial_red_beans + added_red_beans
def final_black_beans : ℕ := initial_black_beans + added_black_beans
def final_total_beans : ℕ := final_red_beans + final_black_beans

-- The probability of drawing a red bean should be 2/5
theorem probability_red_bean :
  (final_red_beans : ℚ) / final_total_beans = 2 / 5 := by
  sorry

end ProbabilityRedBean

end NUMINAMATH_GPT_probability_red_bean_l395_39556


namespace NUMINAMATH_GPT_moles_of_KOH_combined_l395_39511

theorem moles_of_KOH_combined 
  (moles_NH4Cl : ℕ)
  (moles_KCl : ℕ)
  (balanced_reaction : ℕ → ℕ → ℕ)
  (h_NH4Cl : moles_NH4Cl = 3)
  (h_KCl : moles_KCl = 3)
  (reaction_ratio : ∀ n, balanced_reaction n n = n) :
  balanced_reaction moles_NH4Cl moles_KCl = 3 * balanced_reaction 1 1 := 
by
  sorry

end NUMINAMATH_GPT_moles_of_KOH_combined_l395_39511


namespace NUMINAMATH_GPT_sequence_solution_l395_39586

theorem sequence_solution (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2) (h_rec : ∀ n > 0, a (n + 1) = a n ^ 2) : 
  a n = 2 ^ 2 ^ (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_solution_l395_39586


namespace NUMINAMATH_GPT_problem_l395_39561

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * a * x^2 - (x - 1) * Real.exp x

theorem problem (a : ℝ) :
  (∀ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ x3 ∧ x3 ≤ 1 →
                  f a x1 + f a x2 ≥ f a x3) →
  1 ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_GPT_problem_l395_39561


namespace NUMINAMATH_GPT_preston_charges_5_dollars_l395_39548

def cost_per_sandwich (x : Real) : Prop :=
  let number_of_sandwiches := 18
  let delivery_fee := 20
  let tip_percentage := 0.10
  let total_received := 121
  let total_cost := number_of_sandwiches * x + delivery_fee
  let tip := tip_percentage * total_cost
  let final_amount := total_cost + tip
  final_amount = total_received

theorem preston_charges_5_dollars :
  ∀ x : Real, cost_per_sandwich x → x = 5 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_preston_charges_5_dollars_l395_39548


namespace NUMINAMATH_GPT_find_two_digit_divisors_l395_39575

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_remainder (a b r : ℕ) : Prop := a = b * (a / b) + r

theorem find_two_digit_divisors (n : ℕ) (h1 : is_two_digit n) (h2 : has_remainder 723 n 30) :
  n = 33 ∨ n = 63 ∨ n = 77 ∨ n = 99 :=
sorry

end NUMINAMATH_GPT_find_two_digit_divisors_l395_39575


namespace NUMINAMATH_GPT_ali_total_money_l395_39565

-- Definitions based on conditions
def bills_of_5_dollars : ℕ := 7
def bills_of_10_dollars : ℕ := 1
def value_of_5_dollar_bill : ℕ := 5
def value_of_10_dollar_bill : ℕ := 10

-- Prove that Ali's total amount of money is $45
theorem ali_total_money : (bills_of_5_dollars * value_of_5_dollar_bill) + (bills_of_10_dollars * value_of_10_dollar_bill) = 45 := 
by
  sorry

end NUMINAMATH_GPT_ali_total_money_l395_39565


namespace NUMINAMATH_GPT_angle_F_measure_l395_39524

theorem angle_F_measure (D E F : ℝ) (h₁ : D = 80) (h₂ : E = 2 * F + 24) (h₃ : D + E + F = 180) : F = 76 / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_F_measure_l395_39524


namespace NUMINAMATH_GPT_find_product_of_two_numbers_l395_39501

theorem find_product_of_two_numbers (a b : ℚ) (h1 : a + b = 7) (h2 : a - b = 2) : 
  a * b = 11 + 1/4 := 
by 
  sorry

end NUMINAMATH_GPT_find_product_of_two_numbers_l395_39501


namespace NUMINAMATH_GPT_bill_took_six_naps_l395_39588

def total_hours (days : Nat) : Nat := days * 24

def hours_left (total : Nat) (worked : Nat) : Nat := total - worked

def naps_taken (remaining : Nat) (duration : Nat) : Nat := remaining / duration

theorem bill_took_six_naps :
  let days := 4
  let hours_worked := 54
  let nap_duration := 7
  naps_taken (hours_left (total_hours days) hours_worked) nap_duration = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_bill_took_six_naps_l395_39588


namespace NUMINAMATH_GPT_unique_rational_solution_l395_39598

theorem unique_rational_solution (x y z : ℚ) (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_unique_rational_solution_l395_39598


namespace NUMINAMATH_GPT_complex_multiplication_l395_39520

-- Define the imaginary unit i
def i := Complex.I

-- Define the theorem we need to prove
theorem complex_multiplication : 
  (3 - 7 * i) * (-6 + 2 * i) = -4 + 48 * i := 
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_complex_multiplication_l395_39520


namespace NUMINAMATH_GPT_students_present_each_day_l395_39505
open BigOperators

namespace Absenteeism

def absenteeism_rate : ℕ → ℝ 
| 0 => 14
| n+1 => absenteeism_rate n + 2

def present_rate (n : ℕ) : ℝ := 100 - absenteeism_rate n

theorem students_present_each_day :
  present_rate 0 = 86 ∧
  present_rate 1 = 84 ∧
  present_rate 2 = 82 ∧
  present_rate 3 = 80 ∧
  present_rate 4 = 78 := 
by
  -- Placeholder for the proof steps
  sorry

end Absenteeism

end NUMINAMATH_GPT_students_present_each_day_l395_39505


namespace NUMINAMATH_GPT_convert_to_scientific_notation_l395_39549

theorem convert_to_scientific_notation :
  40.25 * 10^9 = 4.025 * 10^9 :=
by
  -- Sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_convert_to_scientific_notation_l395_39549


namespace NUMINAMATH_GPT_orchestra_members_l395_39597

theorem orchestra_members :
  ∃ (n : ℕ), 
    150 < n ∧ n < 250 ∧ 
    n % 4 = 2 ∧ 
    n % 5 = 3 ∧ 
    n % 7 = 4 :=
by
  use 158
  repeat {split};
  sorry

end NUMINAMATH_GPT_orchestra_members_l395_39597


namespace NUMINAMATH_GPT_logarithmic_inequality_l395_39576

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log (1 / 3) / Real.log 4

theorem logarithmic_inequality :
  Real.log a < (1 / 2)^b := by
  sorry

end NUMINAMATH_GPT_logarithmic_inequality_l395_39576


namespace NUMINAMATH_GPT_area_of_square_l395_39592

theorem area_of_square (r s l b : ℝ) (h1 : l = (2/5) * r)
                               (h2 : r = s)
                               (h3 : b = 10)
                               (h4 : l * b = 220) :
  s^2 = 3025 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_area_of_square_l395_39592


namespace NUMINAMATH_GPT_find_a_l395_39533

-- Define the sets A and B based on the conditions
def A (a : ℝ) : Set ℝ := {a ^ 2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, a ^ 2 + 1, 2 * a - 1}

-- Statement: Prove that a = -1 satisfies the condition A ∩ B = {-3}
theorem find_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l395_39533


namespace NUMINAMATH_GPT_temperature_on_Friday_l395_39541

-- Define the temperatures for each day
variables (M T W Th F : ℕ)

-- Declare the given conditions as assumptions
axiom cond1 : (M + T + W + Th) / 4 = 48
axiom cond2 : (T + W + Th + F) / 4 = 46
axiom cond3 : M = 40

-- State the theorem
theorem temperature_on_Friday : F = 32 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end NUMINAMATH_GPT_temperature_on_Friday_l395_39541


namespace NUMINAMATH_GPT_range_of_x_l395_39513

def y_function (x : ℝ) : ℝ := x

def y_translated (x : ℝ) : ℝ := x + 2

theorem range_of_x {x : ℝ} (h : y_translated x > 0) : x > -2 := 
by {
  sorry
}

end NUMINAMATH_GPT_range_of_x_l395_39513


namespace NUMINAMATH_GPT_find_temperature_l395_39518

theorem find_temperature 
  (temps : List ℤ)
  (h_len : temps.length = 8)
  (h_mean : (temps.sum / 8 : ℝ) = -0.5)
  (h_temps : temps = [-6, -3, x, -6, 2, 4, 3, 0]) : 
  x = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_temperature_l395_39518


namespace NUMINAMATH_GPT_exists_integer_point_touching_x_axis_l395_39552

-- Define the context for the problem
variable {p q : ℤ}

-- Condition: The quadratic trinomial touches x-axis, i.e., discriminant is zero.
axiom discriminant_zero (p q : ℤ) : p^2 - 4 * q = 0

-- Theorem statement: Proving the existence of such an integer point.
theorem exists_integer_point_touching_x_axis :
  ∃ a b : ℤ, (a = -p ∧ b = q) ∧ (∀ (x : ℝ), x^2 + a * x + b = 0 → (a * a - 4 * b) = 0) :=
sorry

end NUMINAMATH_GPT_exists_integer_point_touching_x_axis_l395_39552


namespace NUMINAMATH_GPT_dennis_years_of_teaching_l395_39555

variable (V A D E N : ℕ)

def combined_years_taught : Prop :=
  V + A + D + E + N = 225

def virginia_adrienne_relation : Prop :=
  V = A + 9

def virginia_dennis_relation : Prop :=
  V = D - 15

def elijah_adrienne_relation : Prop :=
  E = A - 3

def elijah_nadine_relation : Prop :=
  E = N + 7

theorem dennis_years_of_teaching 
  (h1 : combined_years_taught V A D E N) 
  (h2 : virginia_adrienne_relation V A)
  (h3 : virginia_dennis_relation V D)
  (h4 : elijah_adrienne_relation E A) 
  (h5 : elijah_nadine_relation E N) : 
  D = 65 :=
  sorry

end NUMINAMATH_GPT_dennis_years_of_teaching_l395_39555


namespace NUMINAMATH_GPT_hypotenuse_length_l395_39582

theorem hypotenuse_length (x y : ℝ) 
  (h1 : (1/3) * Real.pi * y^2 * x = 1080 * Real.pi) 
  (h2 : (1/3) * Real.pi * x^2 * y = 2430 * Real.pi) : 
  Real.sqrt (x^2 + y^2) = 6 * Real.sqrt 13 := 
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l395_39582


namespace NUMINAMATH_GPT_probability_at_least_one_l395_39584

theorem probability_at_least_one (p1 p2 : ℝ) (hp1 : 0 ≤ p1) (hp2 : 0 ≤ p2) (hp1p2 : p1 ≤ 1) (hp2p2 : p2 ≤ 1)
  (h0 : 0 ≤ 1 - p1) (h1 : 0 ≤ 1 - p2) (h2 : 1 - (1 - p1) ≥ 0) (h3 : 1 - (1 - p2) ≥ 0) :
  1 - (1 - p1) * (1 - p2) = 1 - (1 - p1) * (1 - p2) := by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_l395_39584


namespace NUMINAMATH_GPT_attendance_rate_correct_l395_39512

def total_students : ℕ := 50
def students_on_leave : ℕ := 2
def given_attendance_rate : ℝ := 96

theorem attendance_rate_correct :
  ((total_students - students_on_leave) / total_students * 100 : ℝ) = given_attendance_rate := sorry

end NUMINAMATH_GPT_attendance_rate_correct_l395_39512


namespace NUMINAMATH_GPT_b_3_value_S_m_formula_l395_39585

-- Definition of the sequences a_n and b_n
def a_n (n : ℕ) : ℕ := if n = 0 then 0 else 3 ^ n
def b_m (m : ℕ) : ℕ := a_n (3 * m)

-- Given b_m = 3^(2m) for m in ℕ*
lemma b_m_formula (m : ℕ) (h : m > 0) : b_m m = 3 ^ (2 * m) :=
by sorry -- (This proof step will later ensure that b_m m is defined as required)

-- Prove b_3 = 729
theorem b_3_value : b_m 3 = 729 :=
by sorry

-- Sum of the first m terms of the sequence b_n
def S_m (m : ℕ) : ℕ := (Finset.range m).sum (λ i => if i = 0 then 0 else b_m (i + 1))

-- Prove S_m = (3/8)(9^m - 1)
theorem S_m_formula (m : ℕ) : S_m m = (3 / 8) * (9 ^ m - 1) :=
by sorry

end NUMINAMATH_GPT_b_3_value_S_m_formula_l395_39585


namespace NUMINAMATH_GPT_find_c_l395_39583

variable (a b c : ℕ)

theorem find_c (h1 : a = 9) (h2 : b = 2) (h3 : Odd c) (h4 : a + b > c) (h5 : a - b < c) (h6 : b + c > a) (h7 : b - c < a) : c = 9 :=
sorry

end NUMINAMATH_GPT_find_c_l395_39583


namespace NUMINAMATH_GPT_outOfPocketCost_l395_39528

noncomputable def visitCost : ℝ := 300
noncomputable def castCost : ℝ := 200
noncomputable def insuranceCoverage : ℝ := 0.60

theorem outOfPocketCost : (visitCost + castCost - (visitCost + castCost) * insuranceCoverage) = 200 := by
  sorry

end NUMINAMATH_GPT_outOfPocketCost_l395_39528


namespace NUMINAMATH_GPT_symmetric_coords_l395_39517

-- Define the initial point and the line equation
def initial_point : ℝ × ℝ := (-1, 1)
def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

-- Define what it means for one point to be symmetric to another point with respect to a line
def symmetric_point (p q : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), line_eq m p.1 ∧ line_eq m q.1 ∧ 
             p.1 + q.1 = 2 * m ∧
             p.2 + q.2 = 2 * m

-- The theorem we want to prove
theorem symmetric_coords : ∃ (symmetric : ℝ × ℝ), symmetric_point initial_point symmetric ∧ symmetric = (2, -2) :=
sorry

end NUMINAMATH_GPT_symmetric_coords_l395_39517


namespace NUMINAMATH_GPT_cost_price_of_computer_table_l395_39537

/-- The owner of a furniture shop charges 20% more than the cost price. 
    Given that the customer paid Rs. 3000 for the computer table, 
    prove that the cost price of the computer table was Rs. 2500. -/
theorem cost_price_of_computer_table (CP SP : ℝ) (h1 : SP = CP + 0.20 * CP) (h2 : SP = 3000) : CP = 2500 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_price_of_computer_table_l395_39537
