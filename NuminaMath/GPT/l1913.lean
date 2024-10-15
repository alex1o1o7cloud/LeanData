import Mathlib

namespace NUMINAMATH_GPT_functional_equation_solution_l1913_191332

open Nat

theorem functional_equation_solution (f : ℕ+ → ℕ+) 
  (H : ∀ (m n : ℕ+), f (f (f m) * f (f m) + 2 * f (f n) * f (f n)) = m * m + 2 * n * n) : 
  ∀ n : ℕ+, f n = n := 
sorry

end NUMINAMATH_GPT_functional_equation_solution_l1913_191332


namespace NUMINAMATH_GPT_adi_change_l1913_191367

theorem adi_change : 
  let pencil := 0.35
  let notebook := 1.50
  let colored_pencils := 2.75
  let discount := 0.05
  let tax := 0.10
  let payment := 20.00
  let total_cost_before_discount := pencil + notebook + colored_pencils
  let discount_amount := discount * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let tax_amount := tax * total_cost_after_discount
  let total_cost := total_cost_after_discount + tax_amount
  let change := payment - total_cost
  change = 15.19 :=
by
  sorry

end NUMINAMATH_GPT_adi_change_l1913_191367


namespace NUMINAMATH_GPT_find_m_value_l1913_191321

theorem find_m_value (m : ℝ) : (∃ A B : ℝ × ℝ, A = (-2, m) ∧ B = (m, 4) ∧ (∃ k : ℝ, k = (4 - m) / (m + 2) ∧ k = -2) ∧ (∃ l : ℝ, l = -2 ∧ 2 * l + l - 1 = 0)) → m = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l1913_191321


namespace NUMINAMATH_GPT_trigonometric_identity_l1913_191307

theorem trigonometric_identity :
  Real.tan (70 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) * (Real.sqrt 3 * Real.tan (20 * Real.pi / 180) - 1) = -1 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1913_191307


namespace NUMINAMATH_GPT_sector_area_correct_l1913_191306

noncomputable def sector_area (r θ : ℝ) : ℝ := 0.5 * θ * r^2

theorem sector_area_correct (r θ : ℝ) (hr : r = 2) (hθ : θ = 2 * Real.pi / 3) :
  sector_area r θ = 4 * Real.pi / 3 :=
by
  subst hr
  subst hθ
  sorry

end NUMINAMATH_GPT_sector_area_correct_l1913_191306


namespace NUMINAMATH_GPT_remaining_area_correct_l1913_191353

-- Define the side lengths of the large rectangle
def large_rectangle_length1 (x : ℝ) := 2 * x + 5
def large_rectangle_length2 (x : ℝ) := x + 8

-- Define the side lengths of the rectangular hole
def hole_length1 (x : ℝ) := 3 * x - 2
def hole_length2 (x : ℝ) := x + 1

-- Define the area of the large rectangle
def large_rectangle_area (x : ℝ) := (large_rectangle_length1 x) * (large_rectangle_length2 x)

-- Define the area of the hole
def hole_area (x : ℝ) := (hole_length1 x) * (hole_length2 x)

-- Prove the remaining area after accounting for the hole
theorem remaining_area_correct (x : ℝ) : 
  large_rectangle_area x - hole_area x = -x^2 + 20 * x + 42 := 
  by 
    sorry

end NUMINAMATH_GPT_remaining_area_correct_l1913_191353


namespace NUMINAMATH_GPT_find_complement_l1913_191329

-- Define predicate for a specific universal set U and set A
def universal_set (a : ℤ) (x : ℤ) : Prop :=
  x = a^2 - 2 ∨ x = 2 ∨ x = 1

def set_A (a : ℤ) (x : ℤ) : Prop :=
  x = a ∨ x = 1

-- Define complement of A with respect to U
def complement_U_A (a : ℤ) (x : ℤ) : Prop :=
  universal_set a x ∧ ¬ set_A a x

-- Main theorem statement
theorem find_complement (a : ℤ) (h : a ≠ 2) : { x | complement_U_A a x } = {2} :=
by
  sorry

end NUMINAMATH_GPT_find_complement_l1913_191329


namespace NUMINAMATH_GPT_range_of_a_l1913_191334

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1913_191334


namespace NUMINAMATH_GPT_common_rational_root_l1913_191399

theorem common_rational_root (a b c d e f g : ℚ) (p : ℚ) :
  (48 * p^4 + a * p^3 + b * p^2 + c * p + 16 = 0) ∧
  (16 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 48 = 0) ∧
  (∃ m n : ℤ, p = m / n ∧ Int.gcd m n = 1 ∧ n ≠ 1 ∧ p < 0 ∧ n > 0) →
  p = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_common_rational_root_l1913_191399


namespace NUMINAMATH_GPT_car_travel_distance_l1913_191387

theorem car_travel_distance:
  (∃ r, r = 3 / 4 ∧ ∀ t, t = 2 → ((r * 60) * t = 90)) :=
by
  sorry

end NUMINAMATH_GPT_car_travel_distance_l1913_191387


namespace NUMINAMATH_GPT_repair_cost_l1913_191345

theorem repair_cost (C : ℝ) (repair_cost : ℝ) (profit : ℝ) (selling_price : ℝ)
  (h1 : repair_cost = 0.10 * C)
  (h2 : profit = 1100)
  (h3 : selling_price = 1.20 * C)
  (h4 : profit = selling_price - C) :
  repair_cost = 550 :=
by
  sorry

end NUMINAMATH_GPT_repair_cost_l1913_191345


namespace NUMINAMATH_GPT_triangle_is_obtuse_l1913_191366

def is_obtuse_triangle (a b c : ℕ) : Prop := a^2 + b^2 < c^2

theorem triangle_is_obtuse :
    is_obtuse_triangle 4 6 8 :=
by
    sorry

end NUMINAMATH_GPT_triangle_is_obtuse_l1913_191366


namespace NUMINAMATH_GPT_simplification_of_expression_l1913_191323

theorem simplification_of_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  ( (x - 2) / (x^2 - 2 * x + 1) / (x / (x - 1)) + 1 / (x^2 - x) ) = 1 / x := 
by 
  sorry

end NUMINAMATH_GPT_simplification_of_expression_l1913_191323


namespace NUMINAMATH_GPT_common_denominator_step1_error_in_step3_simplified_expression_l1913_191346

theorem common_denominator_step1 (x : ℝ) (h1: x ≠ 2) (h2: x ≠ -2):
  (3 * x / (x - 2) - x / (x + 2)) = (3 * x * (x + 2)) / ((x - 2) * (x + 2)) - (x * (x - 2)) / ((x - 2) * (x + 2)) :=
sorry

theorem error_in_step3 (x : ℝ) (h1: x ≠ 2) (h2: x ≠ -2) :
  (3 * x^2 + 6 * x - (x^2 - 2 * x)) / ((x - 2) * (x + 2)) ≠ (3 * x^2 + 6 * x * (x^2 - 2 * x)) / ((x - 2) * (x + 2)) :=
sorry

theorem simplified_expression (x : ℝ) (h1: x ≠ 0) (h2: x ≠ 2) (h3: x ≠ -2) :
  ((3 * x / (x - 2) - x / (x + 2)) * (x^2 - 4) / x) = 2 * x + 8 :=
sorry

end NUMINAMATH_GPT_common_denominator_step1_error_in_step3_simplified_expression_l1913_191346


namespace NUMINAMATH_GPT_find_n_l1913_191380

theorem find_n (n : ℕ) (k : ℕ) (x : ℝ) (h1 : k = 1) (h2 : x = 180 - 360 / n) (h3 : 1.5 * x = 180 - 360 / (n + 1)) :
    n = 3 :=
by
  -- proof steps will be provided here
  sorry

end NUMINAMATH_GPT_find_n_l1913_191380


namespace NUMINAMATH_GPT_total_students_in_class_l1913_191351

theorem total_students_in_class (S R : ℕ)
  (h1 : S = 2 + 12 + 4 + R)
  (h2 : 0 * 2 + 1 * 12 + 2 * 4 + 3 * R = 2 * S) : S = 34 :=
by { sorry }

end NUMINAMATH_GPT_total_students_in_class_l1913_191351


namespace NUMINAMATH_GPT_solve_fractional_equation_l1913_191325

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1913_191325


namespace NUMINAMATH_GPT_area_of_square_l1913_191309

theorem area_of_square (A_circle : ℝ) (hA_circle : A_circle = 39424) (cm_to_inch : ℝ) (hcm_to_inch : cm_to_inch = 2.54) :
  ∃ (A_square : ℝ), A_square = 121.44 := 
by
  sorry

end NUMINAMATH_GPT_area_of_square_l1913_191309


namespace NUMINAMATH_GPT_triangular_array_sum_digits_l1913_191378

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 2080) : 
  (N.digits 10).sum = 10 :=
sorry

end NUMINAMATH_GPT_triangular_array_sum_digits_l1913_191378


namespace NUMINAMATH_GPT_find_tan_α_l1913_191347

variable (α : ℝ) (h1 : Real.sin (α - Real.pi / 3) = 3 / 5)
variable (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2)

theorem find_tan_α (h1 : Real.sin (α - Real.pi / 3) = 3 / 5) (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2) : 
  Real.tan α = - (48 + 25 * Real.sqrt 3) / 11 :=
sorry

end NUMINAMATH_GPT_find_tan_α_l1913_191347


namespace NUMINAMATH_GPT_decagonal_pyramid_volume_l1913_191397

noncomputable def volume_of_decagonal_pyramid (m : ℝ) (apex_angle : ℝ) : ℝ :=
  let sin18 := Real.sin (18 * Real.pi / 180)
  let sin36 := Real.sin (36 * Real.pi / 180)
  let cos18 := Real.cos (18 * Real.pi / 180)
  (5 * m^3 * sin36) / (3 * (1 + 2 * cos18))

theorem decagonal_pyramid_volume : volume_of_decagonal_pyramid 39 (18 * Real.pi / 180) = 20023 :=
  sorry

end NUMINAMATH_GPT_decagonal_pyramid_volume_l1913_191397


namespace NUMINAMATH_GPT_find_x_value_l1913_191304

theorem find_x_value (x : ℝ) (h : (55 + 113 / x) * x = 4403) : x = 78 :=
sorry

end NUMINAMATH_GPT_find_x_value_l1913_191304


namespace NUMINAMATH_GPT_complement_U_A_union_B_is_1_and_9_l1913_191360

-- Define the universe set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define set A according to the given condition
def is_elem_of_A (x : ℕ) : Prop := 2 < x ∧ x ≤ 6
def A : Set ℕ := {x | is_elem_of_A x}

-- Define set B explicitly
def B : Set ℕ := {0, 2, 4, 5, 7, 8}

-- Define the union A ∪ B
def A_union_B : Set ℕ := A ∪ B

-- Define the complement of A ∪ B in U
def complement_U_A_union_B : Set ℕ := {x ∈ U | x ∉ A_union_B}

-- State the theorem
theorem complement_U_A_union_B_is_1_and_9 :
  complement_U_A_union_B = {1, 9} :=
by
  sorry

end NUMINAMATH_GPT_complement_U_A_union_B_is_1_and_9_l1913_191360


namespace NUMINAMATH_GPT_simplify_equation_l1913_191389

theorem simplify_equation (x : ℝ) : 
  (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) -> 
  (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by 
  sorry

end NUMINAMATH_GPT_simplify_equation_l1913_191389


namespace NUMINAMATH_GPT_sum_of_values_l1913_191358

def r (x : ℝ) : ℝ := abs (x + 1) - 3
def s (x : ℝ) : ℝ := -(abs (x + 2))

theorem sum_of_values :
  (s (r (-5)) + s (r (-4)) + s (r (-3)) + s (r (-2)) + s (r (-1)) + s (r (0)) + s (r (1)) + s (r (2)) + s (r (3))) = -37 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_values_l1913_191358


namespace NUMINAMATH_GPT_find_number_l1913_191357

theorem find_number : ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 51 ∧ x = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1913_191357


namespace NUMINAMATH_GPT_sum_of_roots_l1913_191335

open Real

theorem sum_of_roots (x1 x2 k c : ℝ) (h1 : 4 * x1^2 - k * x1 = c) (h2 : 4 * x2^2 - k * x2 = c) (h3 : x1 ≠ x2) :
  x1 + x2 = k / 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1913_191335


namespace NUMINAMATH_GPT_find_S_l1913_191376

variable {R k : ℝ}

theorem find_S (h : |k + R| / |R| = 0) : S = 1 :=
by
  let S := |k + 2*R| / |2*k + R|
  have h1 : k + R = 0 := by sorry
  have h2 : k = -R := by sorry
  sorry

end NUMINAMATH_GPT_find_S_l1913_191376


namespace NUMINAMATH_GPT_child_ticket_cost_is_2_l1913_191375

-- Define the conditions
def adult_ticket_cost : ℕ := 5
def total_tickets_sold : ℕ := 85
def total_revenue : ℕ := 275
def adult_tickets_sold : ℕ := 35

-- Define the function to calculate child ticket cost
noncomputable def child_ticket_cost (adult_ticket_cost : ℕ) (total_tickets_sold : ℕ) (total_revenue : ℕ) (adult_tickets_sold : ℕ) : ℕ :=
  let total_adult_revenue := adult_tickets_sold * adult_ticket_cost
  let total_child_revenue := total_revenue - total_adult_revenue
  let child_tickets_sold := total_tickets_sold - adult_tickets_sold
  total_child_revenue / child_tickets_sold

theorem child_ticket_cost_is_2 : child_ticket_cost adult_ticket_cost total_tickets_sold total_revenue adult_tickets_sold = 2 := 
by
  -- This is a placeholder for the actual proof which we can fill in separately.
  sorry

end NUMINAMATH_GPT_child_ticket_cost_is_2_l1913_191375


namespace NUMINAMATH_GPT_lemonade_calories_is_correct_l1913_191315

def lemon_juice_content := 150
def sugar_content := 150
def water_content := 450

def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 400
def water_calories_per_100g := 0

def total_weight := lemon_juice_content + sugar_content + water_content
def caloric_density :=
  (lemon_juice_content * lemon_juice_calories_per_100g / 100) +
  (sugar_content * sugar_calories_per_100g / 100) +
  (water_content * water_calories_per_100g / 100)
def calories_per_gram := caloric_density / total_weight

def calories_in_300_grams := 300 * calories_per_gram

theorem lemonade_calories_is_correct : calories_in_300_grams = 258 := by
  sorry

end NUMINAMATH_GPT_lemonade_calories_is_correct_l1913_191315


namespace NUMINAMATH_GPT_balls_in_boxes_l1913_191350

theorem balls_in_boxes :
  let balls := 5
  let boxes := 4
  boxes ^ balls = 1024 :=
by
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l1913_191350


namespace NUMINAMATH_GPT_problem_C_l1913_191326

theorem problem_C (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a * b :=
by sorry

end NUMINAMATH_GPT_problem_C_l1913_191326


namespace NUMINAMATH_GPT_beka_distance_l1913_191359

theorem beka_distance (jackson_distance : ℕ) (beka_more_than_jackson : ℕ) :
  jackson_distance = 563 → beka_more_than_jackson = 310 → 
  (jackson_distance + beka_more_than_jackson = 873) :=
by
  sorry

end NUMINAMATH_GPT_beka_distance_l1913_191359


namespace NUMINAMATH_GPT_no_possible_seating_arrangement_l1913_191361

theorem no_possible_seating_arrangement : 
  ¬(∃ (students : Fin 11 → Fin 4),
    ∀ (i : Fin 11),
    ∃ (s1 s2 s3 s4 s5 : Fin 11),
      s1 = i ∧ 
      (s2 = (i + 1) % 11) ∧ 
      (s3 = (i + 2) % 11) ∧ 
      (s4 = (i + 3) % 11) ∧ 
      (s5 = (i + 4) % 11) ∧
      ∃ (g1 g2 g3 g4 : Fin 4),
        (students s1 = g1) ∧ 
        (students s2 = g2) ∧ 
        (students s3 = g3) ∧ 
        (students s4 = g4) ∧ 
        (students s5).val ≠ (students s1).val ∧ 
        (students s5).val ≠ (students s2).val ∧ 
        (students s5).val ≠ (students s3).val ∧ 
        (students s5).val ≠ (students s4).val) :=
sorry

end NUMINAMATH_GPT_no_possible_seating_arrangement_l1913_191361


namespace NUMINAMATH_GPT_impossible_equal_sums_3x3_l1913_191318

theorem impossible_equal_sums_3x3 (a b c d e f g h i : ℕ) :
  a + b + c = 13 ∨ a + b + c = 14 ∨ a + b + c = 15 ∨ a + b + c = 16 ∨ a + b + c = 17 ∨ a + b + c = 18 ∨ a + b + c = 19 ∨ a + b + c = 20 →
  (a + d + g) = 13 ∨ (a + d + g) = 14 ∨ (a + d + g) = 15 ∨ (a + d + g) = 16 ∨ (a + d + g) = 17 ∨ (a + d + g) = 18 ∨ (a + d + g) = 19 ∨ (a + d + g) = 20 →
  (a + e + i) = 13 ∨ (a + e + i) = 14 ∨ (a + e + i) = 15 ∨ (a + e + i) = 16 ∨ (a + e + i) = 17 ∨ (a + e + i) = 18 ∨ (a + e + i) = 19 ∨ (a + e + i) = 20 →
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧ 1 ≤ f ∧ f ≤ 9 ∧ 1 ≤ g ∧ g ≤ 9 ∧ 1 ≤ h ∧ h ≤ 9 ∧ 1 ≤ i ∧ i ≤ 9 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ g ≠ h ∧ g ≠ i ∧ h ≠ i →
  false :=
sorry

end NUMINAMATH_GPT_impossible_equal_sums_3x3_l1913_191318


namespace NUMINAMATH_GPT_number_of_three_cell_shapes_l1913_191330

theorem number_of_three_cell_shapes (x y : ℕ) (h : 3 * x + 4 * y = 22) : x = 6 :=
sorry

end NUMINAMATH_GPT_number_of_three_cell_shapes_l1913_191330


namespace NUMINAMATH_GPT_q_joins_after_2_days_l1913_191396

-- Define the conditions
def work_rate_p := 1 / 10
def work_rate_q := 1 / 6
def total_days := 5

-- Define the proof problem
theorem q_joins_after_2_days (a b : ℝ) (t x : ℕ) : 
  a = work_rate_p → b = work_rate_q → t = total_days →
  x * a + (t - x) * (a + b) = 1 → 
  x = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_q_joins_after_2_days_l1913_191396


namespace NUMINAMATH_GPT_least_number_to_subtract_l1913_191386

theorem least_number_to_subtract :
  ∃ k : ℕ, k = 45 ∧ (568219 - k) % 89 = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l1913_191386


namespace NUMINAMATH_GPT_people_own_only_cats_and_dogs_l1913_191363

-- Define the given conditions
def total_people : ℕ := 59
def only_dogs : ℕ := 15
def only_cats : ℕ := 10
def cats_dogs_snakes : ℕ := 3
def total_snakes : ℕ := 29

-- Define the proof problem
theorem people_own_only_cats_and_dogs : ∃ x : ℕ, 15 + 10 + x + 3 + (29 - 3) = 59 ∧ x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_people_own_only_cats_and_dogs_l1913_191363


namespace NUMINAMATH_GPT_adam_and_simon_time_to_be_80_miles_apart_l1913_191383

theorem adam_and_simon_time_to_be_80_miles_apart :
  ∃ x : ℝ, (10 * x)^2 + (8 * x)^2 = 80^2 ∧ x = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_adam_and_simon_time_to_be_80_miles_apart_l1913_191383


namespace NUMINAMATH_GPT_boxes_of_chocolates_l1913_191303

theorem boxes_of_chocolates (total_pieces : ℕ) (pieces_per_box : ℕ) (h_total : total_pieces = 3000) (h_each : pieces_per_box = 500) : total_pieces / pieces_per_box = 6 :=
by
  sorry

end NUMINAMATH_GPT_boxes_of_chocolates_l1913_191303


namespace NUMINAMATH_GPT_percentage_of_adult_men_l1913_191316

theorem percentage_of_adult_men (total_members : ℕ) (children : ℕ) (p : ℕ) :
  total_members = 2000 → children = 200 → 
  (∀ adult_men_percentage : ℕ, adult_women_percentage = 2 * adult_men_percentage) → 
  (100 - p) = 3 * (p - 10) →  p = 30 :=
by sorry

end NUMINAMATH_GPT_percentage_of_adult_men_l1913_191316


namespace NUMINAMATH_GPT_rose_bushes_in_park_l1913_191300

theorem rose_bushes_in_park (current_rose_bushes total_new_rose_bushes total_rose_bushes : ℕ) 
(h1 : total_new_rose_bushes = 4)
(h2 : total_rose_bushes = 6) :
current_rose_bushes + total_new_rose_bushes = total_rose_bushes → current_rose_bushes = 2 := 
by 
  sorry

end NUMINAMATH_GPT_rose_bushes_in_park_l1913_191300


namespace NUMINAMATH_GPT_sum_of_factors_1656_l1913_191343

theorem sum_of_factors_1656 : ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 1656 ∧ a + b = 110 := by
  sorry

end NUMINAMATH_GPT_sum_of_factors_1656_l1913_191343


namespace NUMINAMATH_GPT_simplify_exponent_l1913_191352

variable {x : ℝ} {m n : ℕ}

theorem simplify_exponent (x : ℝ) : (3 * x ^ 5) * (4 * x ^ 3) = 12 * x ^ 8 := by
  sorry

end NUMINAMATH_GPT_simplify_exponent_l1913_191352


namespace NUMINAMATH_GPT_surface_area_of_segmented_part_l1913_191394

theorem surface_area_of_segmented_part (h_prism : ∀ (base_height prism_height : ℝ), base_height = 9 ∧ prism_height = 20)
  (isosceles_triangle : ∀ (a b c : ℝ), a = 18 ∧ b = 15 ∧ c = 15 ∧ b = c)
  (midpoints : ∀ (X Y Z : ℝ), X = 9 ∧ Y = 10 ∧ Z = 9) 
  : let triangle_CZX_area := 45
    let triangle_CZY_area := 45
    let triangle_CXY_area := 9
    let triangle_XYZ_area := 9
    (triangle_CZX_area + triangle_CZY_area + triangle_CXY_area + triangle_XYZ_area = 108) :=
sorry

end NUMINAMATH_GPT_surface_area_of_segmented_part_l1913_191394


namespace NUMINAMATH_GPT_nicky_profit_l1913_191322

theorem nicky_profit (value_traded_away value_received : ℤ)
  (h1 : value_traded_away = 2 * 8)
  (h2 : value_received = 21) :
  value_received - value_traded_away = 5 :=
by
  sorry

end NUMINAMATH_GPT_nicky_profit_l1913_191322


namespace NUMINAMATH_GPT_find_measure_angle_AOD_l1913_191313

-- Definitions of angles in the problem
def angle_COA := 150
def angle_BOD := 120

-- Definition of the relationship between angles
def angle_AOD_eq_four_times_angle_BOC (x : ℝ) : Prop :=
  4 * x = 360

-- Proof Problem Lean Statement
theorem find_measure_angle_AOD (x : ℝ) (h1 : 180 - 30 = angle_COA) (h2 : 180 - 60 = angle_BOD) (h3 : angle_AOD_eq_four_times_angle_BOC x) : 
  4 * x = 360 :=
  by 
  -- Insert necessary steps here
  sorry

end NUMINAMATH_GPT_find_measure_angle_AOD_l1913_191313


namespace NUMINAMATH_GPT_min_y_value_l1913_191373

theorem min_y_value (x : ℝ) : 
  ∃ y : ℝ, y = 4 * x^2 + 8 * x + 12 ∧ ∀ z, (z = 4 * x^2 + 8 * x + 12) → y ≤ z := sorry

end NUMINAMATH_GPT_min_y_value_l1913_191373


namespace NUMINAMATH_GPT_short_pencil_cost_l1913_191314

theorem short_pencil_cost (x : ℝ)
  (h1 : 200 * 0.8 + 40 * 0.5 + 35 * x = 194) : x = 0.4 :=
by {
  sorry
}

end NUMINAMATH_GPT_short_pencil_cost_l1913_191314


namespace NUMINAMATH_GPT_find_larger_number_l1913_191370

theorem find_larger_number 
  (L S : ℕ) 
  (h1 : L - S = 2342) 
  (h2 : L = 9 * S + 23) : 
  L = 2624 := 
sorry

end NUMINAMATH_GPT_find_larger_number_l1913_191370


namespace NUMINAMATH_GPT_average_of_seven_consecutive_l1913_191371

variable (a : ℕ) 

def average_of_consecutive_integers (x : ℕ) : ℕ :=
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) / 7

theorem average_of_seven_consecutive (a : ℕ) :
  average_of_consecutive_integers (average_of_consecutive_integers a) = a + 6 :=
by
  sorry

end NUMINAMATH_GPT_average_of_seven_consecutive_l1913_191371


namespace NUMINAMATH_GPT_cream_cheese_volume_l1913_191338

theorem cream_cheese_volume
  (raw_spinach : ℕ)
  (spinach_reduction : ℕ)
  (eggs_volume : ℕ)
  (total_volume : ℕ)
  (cooked_spinach : ℕ)
  (cream_cheese : ℕ) :
  raw_spinach = 40 →
  spinach_reduction = 20 →
  eggs_volume = 4 →
  total_volume = 18 →
  cooked_spinach = raw_spinach * spinach_reduction / 100 →
  cream_cheese = total_volume - cooked_spinach - eggs_volume →
  cream_cheese = 6 :=
by
  intros h_raw_spinach h_spinach_reduction h_eggs_volume h_total_volume h_cooked_spinach h_cream_cheese
  sorry

end NUMINAMATH_GPT_cream_cheese_volume_l1913_191338


namespace NUMINAMATH_GPT_find_unknown_rate_of_blankets_l1913_191398

theorem find_unknown_rate_of_blankets (x : ℕ) 
  (h1 : 3 * 100 = 300) 
  (h2 : 5 * 150 = 750)
  (h3 : 3 + 5 + 2 = 10) 
  (h4 : 10 * 160 = 1600) 
  (h5 : 300 + 750 + 2 * x = 1600) : 
  x = 275 := 
sorry

end NUMINAMATH_GPT_find_unknown_rate_of_blankets_l1913_191398


namespace NUMINAMATH_GPT_part1_combined_time_part2_copier_A_insufficient_part3_combined_after_repair_l1913_191349

-- Definitions for times needed by copiers A and B
def time_A : ℕ := 90
def time_B : ℕ := 60

-- (1) Combined time for both copiers
theorem part1_combined_time : 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 36 = 1 := 
by sorry

-- (2) Time left for copier A alone
theorem part2_copier_A_insufficient (mins_combined : ℕ) (time_left : ℕ) : 
  mins_combined = 30 → time_left = 13 → 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 30 + time_left / (time_A : ℝ) ≠ 1 := 
by sorry

-- (3) Combined time with B after repair is sufficient
theorem part3_combined_after_repair (mins_combined : ℕ) (mins_repair_B : ℕ) (time_left : ℕ) : 
  mins_combined = 30 → mins_repair_B = 9 → time_left = 13 →
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 30 + 9 / (time_A : ℝ) + 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 2.4 = 1 := 
by sorry

end NUMINAMATH_GPT_part1_combined_time_part2_copier_A_insufficient_part3_combined_after_repair_l1913_191349


namespace NUMINAMATH_GPT_R2_perfect_fit_l1913_191395

variables {n : ℕ} (x y : Fin n → ℝ) (b a : ℝ)

-- Condition: Observations \( (x_i, y_i) \) such that \( y_i = bx_i + a \)
def observations (i : Fin n) : Prop :=
  y i = b * x i + a

-- Condition: \( e_i = 0 \) for all \( i \)
def no_error (i : Fin n) : Prop := (b * x i + a + 0 = y i)

theorem R2_perfect_fit (h_obs: ∀ i, observations x y b a i)
                       (h_no_error: ∀ i, no_error x y b a i) : R_squared = 1 := by
  sorry

end NUMINAMATH_GPT_R2_perfect_fit_l1913_191395


namespace NUMINAMATH_GPT_value_of_expression_l1913_191392

theorem value_of_expression (x : ℝ) (h : x = 1 + Real.sqrt 2) : x^4 - 4 * x^3 + 4 * x^2 + 4 = 5 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1913_191392


namespace NUMINAMATH_GPT_infinite_B_l1913_191388

open Set Function

variable (A B : Type) 

theorem infinite_B (hA_inf : Infinite A) (f : A → B) : Infinite B :=
by
  sorry

end NUMINAMATH_GPT_infinite_B_l1913_191388


namespace NUMINAMATH_GPT_triangle_angles_l1913_191342

-- Define the problem and the conditions as Lean statements.
theorem triangle_angles (x y z : ℝ) 
  (h1 : y + 150 + 160 = 360)
  (h2 : z + 150 + 160 = 360)
  (h3 : x + y + z = 180) : 
  x = 80 ∧ y = 50 ∧ z = 50 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_angles_l1913_191342


namespace NUMINAMATH_GPT_min_distinct_sums_l1913_191308

theorem min_distinct_sums (n : ℕ) (hn : n ≥ 5) (s : Finset ℕ) 
  (hs : s.card = n) : 
  ∃ (t : Finset ℕ), (∀ (x y : ℕ), x ∈ s → y ∈ s → x < y → (x + y) ∈ t) ∧ t.card = 2 * n - 3 :=
by
  sorry

end NUMINAMATH_GPT_min_distinct_sums_l1913_191308


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l1913_191328

theorem sum_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith_seq : ∀ n, S (n + 1) - S n = a n)
  (h_S2 : S 2 = 4) 
  (h_S4 : S 4 = 16) 
: a 5 + a 6 = 20 :=
sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l1913_191328


namespace NUMINAMATH_GPT_smallest_solution_of_quartic_equation_l1913_191393

theorem smallest_solution_of_quartic_equation :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 625 = 0) ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := sorry

end NUMINAMATH_GPT_smallest_solution_of_quartic_equation_l1913_191393


namespace NUMINAMATH_GPT_daphne_two_visits_in_365_days_l1913_191390

def visits_in_days (d1 d2 : ℕ) (days : ℕ) : ℕ :=
  days / Nat.lcm d1 d2

theorem daphne_two_visits_in_365_days :
  let days := 365
  let lcm_all := Nat.lcm 4 (Nat.lcm 6 (Nat.lcm 8 10))
  (visits_in_days 4 6 lcm_all + 
   visits_in_days 4 8 lcm_all + 
   visits_in_days 4 10 lcm_all + 
   visits_in_days 6 8 lcm_all + 
   visits_in_days 6 10 lcm_all + 
   visits_in_days 8 10 lcm_all) * 
   (days / lcm_all) = 129 :=
by
  sorry

end NUMINAMATH_GPT_daphne_two_visits_in_365_days_l1913_191390


namespace NUMINAMATH_GPT_typing_speed_ratio_l1913_191324

theorem typing_speed_ratio (T t : ℝ) (h1 : T + t = 12) (h2 : T + 1.25 * t = 14) : t / T = 2 :=
by
  sorry

end NUMINAMATH_GPT_typing_speed_ratio_l1913_191324


namespace NUMINAMATH_GPT_true_statement_count_l1913_191301

def n_star (n : ℕ) : ℚ := 1 / n

theorem true_statement_count :
  let s1 := (n_star 4 + n_star 8 = n_star 12)
  let s2 := (n_star 9 - n_star 1 = n_star 8)
  let s3 := (n_star 5 * n_star 3 = n_star 15)
  let s4 := (n_star 16 - n_star 4 = n_star 12)
  (if s1 then 1 else 0) +
  (if s2 then 1 else 0) +
  (if s3 then 1 else 0) +
  (if s4 then 1 else 0) = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_true_statement_count_l1913_191301


namespace NUMINAMATH_GPT_projection_matrix_determinant_l1913_191362

theorem projection_matrix_determinant (a c : ℚ) (h : (a^2 + (20 / 49 : ℚ) * c = a) ∧ ((20 / 49 : ℚ) * a + 580 / 2401 = 20 / 49) ∧ (a * c + (29 / 49 : ℚ) * c = c) ∧ ((20 / 49 : ℚ) * c + 841 / 2401 = 29 / 49)) :
  (a = 41 / 49) ∧ (c = 204 / 1225) := 
by {
  sorry
}

end NUMINAMATH_GPT_projection_matrix_determinant_l1913_191362


namespace NUMINAMATH_GPT_complete_the_square_l1913_191319

theorem complete_the_square (d e f : ℤ) (h1 : d > 0)
  (h2 : 25 * d * d = 25)
  (h3 : 10 * d * e = 30)
  (h4 : 25 * d * d * (d * x + e) * (d * x + e) = 25 * x * x * 25 + 30 * x * 25 * d + 25 * e * e - 9)
  : d + e + f = 41 := 
  sorry

end NUMINAMATH_GPT_complete_the_square_l1913_191319


namespace NUMINAMATH_GPT_hoseok_divides_number_l1913_191379

theorem hoseok_divides_number (x : ℕ) (h : x / 6 = 11) : x = 66 := by
  sorry

end NUMINAMATH_GPT_hoseok_divides_number_l1913_191379


namespace NUMINAMATH_GPT_remaining_quantities_count_l1913_191331

theorem remaining_quantities_count 
  (S : ℕ) (S3 : ℕ) (S2 : ℕ) (n : ℕ) 
  (h1 : S / 5 = 10) 
  (h2 : S3 / 3 = 4) 
  (h3 : S = 50) 
  (h4 : S3 = 12) 
  (h5 : S2 = S - S3) 
  (h6 : S2 / n = 19) 
  : n = 2 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_quantities_count_l1913_191331


namespace NUMINAMATH_GPT_sum_of_interior_angles_l1913_191339

theorem sum_of_interior_angles (n : ℕ) (h : 180 * (n - 2) = 1800) : 180 * ((n - 3) - 2) = 1260 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_l1913_191339


namespace NUMINAMATH_GPT_divide_and_add_l1913_191344

theorem divide_and_add (x : ℤ) (h1 : x = 95) : (x / 5) + 23 = 42 := by
  sorry

end NUMINAMATH_GPT_divide_and_add_l1913_191344


namespace NUMINAMATH_GPT_water_level_height_l1913_191369

/-- Problem: An inverted frustum with a bottom diameter of 12 and height of 18, filled with water, 
    is emptied into another cylindrical container with a bottom diameter of 24. Assuming the 
    cylindrical container is sufficiently tall, the height of the water level in the cylindrical container -/
theorem water_level_height
  (V_cone : ℝ := (1 / 3) * π * (12 / 2) ^ 2 * 18)
  (R_cyl : ℝ := 24 / 2)
  (H_cyl : ℝ) :
  V_cone = π * R_cyl ^ 2 * H_cyl →
  H_cyl = 1.5 :=
by 
  sorry

end NUMINAMATH_GPT_water_level_height_l1913_191369


namespace NUMINAMATH_GPT_fraction_subtraction_l1913_191364

theorem fraction_subtraction : (1 / 6 : ℚ) - (5 / 12) = -1 / 4 := 
by sorry

end NUMINAMATH_GPT_fraction_subtraction_l1913_191364


namespace NUMINAMATH_GPT_find_a_b_range_of_a_l1913_191381

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + 2

-- Problem 1
theorem find_a_b (a b : ℝ) :
  f a 1 = 0 ∧ f a b = 0 ∧ (∀ x, f a x > 0 ↔ x < 1 ∨ x > b) → a = 1 ∧ b = 2 := sorry

-- Problem 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x > 0) → (0 ≤ a ∧ a < 8/9) := sorry

end NUMINAMATH_GPT_find_a_b_range_of_a_l1913_191381


namespace NUMINAMATH_GPT_minimal_inverse_presses_l1913_191354

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem minimal_inverse_presses (x : ℚ) (h : x = 50) : 
  ∃ n, n = 2 ∧ (reciprocal^[n] x = x) :=
by
  sorry

end NUMINAMATH_GPT_minimal_inverse_presses_l1913_191354


namespace NUMINAMATH_GPT_S_7_is_28_l1913_191348

-- Define the arithmetic sequence and sum of first n terms
def a : ℕ → ℝ := sorry  -- placeholder for arithmetic sequence
def S (n : ℕ) : ℝ := sorry  -- placeholder for the sum of first n terms

-- Given conditions
def a_3 : ℝ := 3
def a_10 : ℝ := 10

-- Define properties of the arithmetic sequence
axiom a_n_property (n : ℕ) : a n = a 1 + (n - 1) * (a 10 - a 3) / (10 - 3)

-- Define the sum of first n terms
axiom sum_property (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Given specific elements of the sequence
axiom a_3_property : a 3 = 3
axiom a_10_property : a 10 = 10

-- The statement to prove
theorem S_7_is_28 : S 7 = 28 :=
sorry

end NUMINAMATH_GPT_S_7_is_28_l1913_191348


namespace NUMINAMATH_GPT_find_second_number_l1913_191327

theorem find_second_number (x : ℕ) (h1 : ∀ d : ℕ, d ∣ 60 → d ∣ x → d ∣ 18) 
                           (h2 : 60 % 18 = 6) (h3 : x % 18 = 10) 
                           (h4 : x > 60) : 
  x = 64 := 
by
  sorry

end NUMINAMATH_GPT_find_second_number_l1913_191327


namespace NUMINAMATH_GPT_point_B_coordinates_l1913_191336

/-
Problem Statement:
Given a point A(2, 4) which is symmetric to point B with respect to the origin,
we need to prove the coordinates of point B.
-/

structure Point where
  x : ℝ
  y : ℝ

def symmetric_wrt_origin (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = -A.y

noncomputable def point_A : Point := ⟨2, 4⟩
noncomputable def point_B : Point := ⟨-2, -4⟩

theorem point_B_coordinates : symmetric_wrt_origin point_A point_B :=
  by
    -- Proof is omitted
    sorry

end NUMINAMATH_GPT_point_B_coordinates_l1913_191336


namespace NUMINAMATH_GPT_carl_weight_l1913_191374

variable (C R B : ℕ)

theorem carl_weight (h1 : B = R + 9) (h2 : R = C + 5) (h3 : B = 159) : C = 145 :=
by
  sorry

end NUMINAMATH_GPT_carl_weight_l1913_191374


namespace NUMINAMATH_GPT_tangent_line_parabola_d_l1913_191312

theorem tangent_line_parabola_d (d : ℝ) :
  (∀ x y : ℝ, (y = 3 * x + d) → (y^2 = 12 * x) → ∃! x, 9 * x^2 + (6 * d - 12) * x + d^2 = 0) → d = 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_parabola_d_l1913_191312


namespace NUMINAMATH_GPT_value_of_m_l1913_191305

noncomputable def TV_sales_volume_function (x : ℕ) : ℚ :=
  10 * x + 540

theorem value_of_m : ∀ (m : ℚ),
  (3200 * (1 + m / 100) * 9 / 10) * (600 * (1 - 2 * m / 100) + 220) = 3200 * 600 * (1 + 15.5 / 100) →
  m = 10 :=
by sorry

end NUMINAMATH_GPT_value_of_m_l1913_191305


namespace NUMINAMATH_GPT_fraction_power_l1913_191372

theorem fraction_power (a b : ℕ) (ha : a = 5) (hb : b = 6) : (a / b : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end NUMINAMATH_GPT_fraction_power_l1913_191372


namespace NUMINAMATH_GPT_dice_probability_green_l1913_191391

theorem dice_probability_green :
  let total_faces := 6
  let green_faces := 3
  let probability := green_faces / total_faces
  probability = 1 / 2 :=
by
  let total_faces := 6
  let green_faces := 3
  let probability := green_faces / total_faces
  have h : probability = 1 / 2 := by sorry
  exact h

end NUMINAMATH_GPT_dice_probability_green_l1913_191391


namespace NUMINAMATH_GPT_ratio_mets_redsox_l1913_191317

theorem ratio_mets_redsox 
    (Y M R : ℕ) 
    (h1 : Y = 3 * (M / 2))
    (h2 : M = 88)
    (h3 : Y + M + R = 330) : 
    M / R = 4 / 5 := 
by 
    sorry

end NUMINAMATH_GPT_ratio_mets_redsox_l1913_191317


namespace NUMINAMATH_GPT_ratio_man_to_son_in_two_years_l1913_191385

-- Define the conditions
def son_current_age : ℕ := 32
def man_current_age : ℕ := son_current_age + 34

-- Define the ages in two years
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- The theorem to prove the ratio in two years
theorem ratio_man_to_son_in_two_years : 
  (man_age_in_two_years : ℚ) / son_age_in_two_years = 2 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_ratio_man_to_son_in_two_years_l1913_191385


namespace NUMINAMATH_GPT_increase_function_a_seq_increasing_b_seq_decreasing_seq_relation_l1913_191384

open Real

-- Defining the sequences
noncomputable def a_seq (n : ℕ) : ℝ := (1 + (1 : ℝ) / n) ^ n
noncomputable def b_seq (n : ℕ) : ℝ := (1 + (1 : ℝ) / n) ^ (n + 1)

theorem increase_function (x : ℝ) (hx : 0 < x) : 
  ((1:ℝ) + 1 / x) ^ x < (1 + 1 / (x + 1)) ^ (x + 1) := sorry

theorem a_seq_increasing (n : ℕ) (hn : 0 < n) : 
  a_seq n < a_seq (n + 1) := sorry

theorem b_seq_decreasing (n : ℕ) (hn : 0 < n) : 
  b_seq (n + 1) < b_seq n := sorry

theorem seq_relation (n : ℕ) (hn : 0 < n) : 
  a_seq n < b_seq n := sorry

end NUMINAMATH_GPT_increase_function_a_seq_increasing_b_seq_decreasing_seq_relation_l1913_191384


namespace NUMINAMATH_GPT_f_recurrence_l1913_191341

noncomputable def f (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem f_recurrence (n : ℕ) : f (n + 1) - f (n - 1) = (3 * Real.sqrt 7 / 14) * f n := 
  sorry

end NUMINAMATH_GPT_f_recurrence_l1913_191341


namespace NUMINAMATH_GPT_value_range_of_m_for_equation_l1913_191311

theorem value_range_of_m_for_equation 
    (x : ℝ) 
    (cos_x : ℝ) 
    (h1: cos_x = Real.cos x) :
    ∃ (m : ℝ), (0 ≤ m ∧ m ≤ 8) ∧ (4 * cos_x + Real.sin x ^ 2 + m - 4 = 0) := sorry

end NUMINAMATH_GPT_value_range_of_m_for_equation_l1913_191311


namespace NUMINAMATH_GPT_compare_f_values_l1913_191355

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem compare_f_values (a : ℝ) (h_pos : 0 < a) :
  (a > 2 * Real.sqrt 2 → f a > f (a / 2) * f (a / 2)) ∧
  (a = 2 * Real.sqrt 2 → f a = f (a / 2) * f (a / 2)) ∧
  (0 < a ∧ a < 2 * Real.sqrt 2 → f a < f (a / 2) * f (a / 2)) :=
by
  sorry

end NUMINAMATH_GPT_compare_f_values_l1913_191355


namespace NUMINAMATH_GPT_downstream_speed_l1913_191310

-- Define the given conditions
def V_m : ℝ := 40 -- speed of the man in still water in kmph
def V_up : ℝ := 32 -- speed of the man upstream in kmph

-- Question to be proved as a statement
theorem downstream_speed : 
  ∃ (V_c V_down : ℝ), V_c = V_m - V_up ∧ V_down = V_m + V_c ∧ V_down = 48 :=
by
  -- Provide statement without proof as specified
  sorry

end NUMINAMATH_GPT_downstream_speed_l1913_191310


namespace NUMINAMATH_GPT_simplify_fraction_l1913_191365

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1913_191365


namespace NUMINAMATH_GPT_correct_transformation_l1913_191382

theorem correct_transformation (a b m : ℝ) (h : m ≠ 0) : (am / bm) = (a / b) :=
by sorry

end NUMINAMATH_GPT_correct_transformation_l1913_191382


namespace NUMINAMATH_GPT_boundary_of_shadow_of_sphere_l1913_191356

theorem boundary_of_shadow_of_sphere (x y : ℝ) :
  let O := (0, 0, 2)
  let P := (1, -2, 3)
  let r := 2
  (∃ T : ℝ × ℝ × ℝ,
    T = (0, -2, 2) ∧
    (∃ g : ℝ → ℝ,
      y = g x ∧
      g x = (x^2 - 2 * x - 11) / 6)) → 
  y = (x^2 - 2 * x - 11) / 6 :=
by
  sorry

end NUMINAMATH_GPT_boundary_of_shadow_of_sphere_l1913_191356


namespace NUMINAMATH_GPT_geometric_progression_x_l1913_191368

theorem geometric_progression_x :
  ∃ x : ℝ, (70 + x) ^ 2 = (30 + x) * (150 + x) ∧ x = 10 :=
by sorry

end NUMINAMATH_GPT_geometric_progression_x_l1913_191368


namespace NUMINAMATH_GPT_delaney_left_home_at_7_50_l1913_191377

theorem delaney_left_home_at_7_50 :
  (bus_time = 8 * 60 ∧ travel_time = 30 ∧ miss_time = 20) →
  (delaney_leave_time = bus_time + miss_time - travel_time) →
  delaney_leave_time = 7 * 60 + 50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_delaney_left_home_at_7_50_l1913_191377


namespace NUMINAMATH_GPT_sheela_deposit_amount_l1913_191340

theorem sheela_deposit_amount (monthly_income : ℕ) (deposit_percentage : ℕ) :
  monthly_income = 25000 → deposit_percentage = 20 → (deposit_percentage / 100 * monthly_income) = 5000 :=
  by
    intros h_income h_percentage
    rw [h_income, h_percentage]
    sorry

end NUMINAMATH_GPT_sheela_deposit_amount_l1913_191340


namespace NUMINAMATH_GPT_smallest_portion_quantity_l1913_191320

-- Define the conditions for the problem
def conditions (a1 a2 a3 a4 a5 d : ℚ) : Prop :=
  a2 = a1 + d ∧
  a3 = a1 + 2 * d ∧
  a4 = a1 + 3 * d ∧
  a5 = a1 + 4 * d ∧
  5 * a1 + 10 * d = 100 ∧
  (a3 + a4 + a5) = (1/7) * (a1 + a2)

-- Lean theorem statement
theorem smallest_portion_quantity : 
  ∃ (a1 a2 a3 a4 a5 d : ℚ), conditions a1 a2 a3 a4 a5 d ∧ a1 = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_portion_quantity_l1913_191320


namespace NUMINAMATH_GPT_percentage_increase_l1913_191302

theorem percentage_increase (new_wage original_wage : ℝ) (h₁ : new_wage = 42) (h₂ : original_wage = 28) :
  ((new_wage - original_wage) / original_wage) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1913_191302


namespace NUMINAMATH_GPT_parallelogram_angle_bisector_l1913_191337

theorem parallelogram_angle_bisector (a b S Q : ℝ) (α : ℝ) 
  (hS : S = a * b * Real.sin α)
  (hQ : Q = (1 / 2) * (a - b) ^ 2 * Real.sin α) :
  (2 * a * b) / (a - b) ^ 2 = (S + Q + Real.sqrt (Q ^ 2 + 2 * Q * S)) / S :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_angle_bisector_l1913_191337


namespace NUMINAMATH_GPT_range_of_values_for_a_l1913_191333

theorem range_of_values_for_a (a : ℝ) :
  (∀ x : ℝ, (x + 2) / 3 - x / 2 > 1 → 2 * (x - a) ≤ 0) → a ≥ -2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_values_for_a_l1913_191333
