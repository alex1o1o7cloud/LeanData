import Mathlib

namespace NUMINAMATH_GPT_number_of_rows_containing_53_l153_15396

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_rows_containing_53_l153_15396


namespace NUMINAMATH_GPT_right_triangle_of_altitude_ratios_l153_15385

theorem right_triangle_of_altitude_ratios
  (h1 h2 h3 : ℝ) 
  (h1_pos : h1 > 0) 
  (h2_pos : h2 > 0) 
  (h3_pos : h3 > 0) 
  (H : (h1 / h2)^2 + (h1 / h3)^2 = 1) : 
  ∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ h1 = 1 / a ∧ h2 = 1 / b ∧ h3 = 1 / c :=
sorry

end NUMINAMATH_GPT_right_triangle_of_altitude_ratios_l153_15385


namespace NUMINAMATH_GPT_tan_of_11pi_over_4_l153_15343

theorem tan_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 := by
  sorry

end NUMINAMATH_GPT_tan_of_11pi_over_4_l153_15343


namespace NUMINAMATH_GPT_number_of_other_workers_l153_15351

theorem number_of_other_workers (N : ℕ) (h1 : N ≥ 2) (h2 : 1 / ((N * (N - 1)) / 2) = 1 / 6) : N - 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_other_workers_l153_15351


namespace NUMINAMATH_GPT_apple_cost_l153_15337

theorem apple_cost (A : ℝ) (h_discount : ∃ (n : ℕ), 15 = (5 * (5: ℝ) * A + 3 * 2 + 2 * 3 - n)) : A = 1 :=
by
  sorry

end NUMINAMATH_GPT_apple_cost_l153_15337


namespace NUMINAMATH_GPT_train_crossing_time_l153_15363

theorem train_crossing_time (length_of_train : ℝ) (speed_kmh : ℝ) :
  length_of_train = 180 →
  speed_kmh = 72 →
  (180 / (72 * (1000 / 3600))) = 9 :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_train_crossing_time_l153_15363


namespace NUMINAMATH_GPT_truck_capacity_l153_15360

theorem truck_capacity (x y : ℝ)
  (h1 : 3 * x + 4 * y = 22)
  (h2 : 5 * x + 2 * y = 25) :
  4 * x + 3 * y = 23.5 :=
sorry

end NUMINAMATH_GPT_truck_capacity_l153_15360


namespace NUMINAMATH_GPT_fractional_exponent_calculation_l153_15370

variables (a b : ℝ) -- Define a and b as real numbers
variable (ha : a > 0) -- Condition a > 0
variable (hb : b > 0) -- Condition b > 0

theorem fractional_exponent_calculation :
  (a^(2 * b^(1/4)) / (a * b^(1/2))^(1/2)) = a^(1/2) :=
by
  sorry -- Proof is not required, skip with sorry

end NUMINAMATH_GPT_fractional_exponent_calculation_l153_15370


namespace NUMINAMATH_GPT_range_of_m_l153_15362

theorem range_of_m (m n : ℝ) (h1 : n = 2 / m) (h2 : n ≥ -1) :
  m ≤ -2 ∨ m > 0 := 
sorry

end NUMINAMATH_GPT_range_of_m_l153_15362


namespace NUMINAMATH_GPT_coordinates_with_respect_to_origin_l153_15310

def point_coordinates (x y : ℤ) : ℤ × ℤ :=
  (x, y)

def origin : ℤ × ℤ :=
  (0, 0)

theorem coordinates_with_respect_to_origin :
  point_coordinates 2 (-3) = (2, -3) := by
  -- placeholder proof
  sorry

end NUMINAMATH_GPT_coordinates_with_respect_to_origin_l153_15310


namespace NUMINAMATH_GPT_train_speed_is_54_kmh_l153_15395

noncomputable def train_length_m : ℝ := 285
noncomputable def train_length_km : ℝ := train_length_m / 1000
noncomputable def time_seconds : ℝ := 19
noncomputable def time_hours : ℝ := time_seconds / 3600
noncomputable def speed : ℝ := train_length_km / time_hours

theorem train_speed_is_54_kmh :
  speed = 54 := by
sorry

end NUMINAMATH_GPT_train_speed_is_54_kmh_l153_15395


namespace NUMINAMATH_GPT_compare_compound_interest_l153_15334

noncomputable def compound_annually (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ t

noncomputable def compound_monthly (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ (12 * t)

theorem compare_compound_interest :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  compound_monthly P r_monthly t > compound_annually P r_annual t :=
by
  sorry

end NUMINAMATH_GPT_compare_compound_interest_l153_15334


namespace NUMINAMATH_GPT_mason_ate_15_hotdogs_l153_15308

structure EatingContest where
  hotdogWeight : ℕ
  burgerWeight : ℕ
  pieWeight : ℕ
  noahBurgers : ℕ
  jacobPiesLess : ℕ
  masonHotdogsWeight : ℕ

theorem mason_ate_15_hotdogs (data : EatingContest)
    (h1 : data.hotdogWeight = 2)
    (h2 : data.burgerWeight = 5)
    (h3 : data.pieWeight = 10)
    (h4 : data.noahBurgers = 8)
    (h5 : data.jacobPiesLess = 3)
    (h6 : data.masonHotdogsWeight = 30) :
    (data.masonHotdogsWeight / data.hotdogWeight) = 15 :=
by
  sorry

end NUMINAMATH_GPT_mason_ate_15_hotdogs_l153_15308


namespace NUMINAMATH_GPT_circle_center_and_radius_l153_15354

-- Define a circle in the plane according to the given equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x = 0

-- Define the center of the circle
def center (x : ℝ) (y : ℝ) : Prop := x = -2 ∧ y = 0

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 2

-- The theorem statement
theorem circle_center_and_radius :
  (∀ x y, circle_eq x y → center x y) ∧ radius 2 :=
sorry

end NUMINAMATH_GPT_circle_center_and_radius_l153_15354


namespace NUMINAMATH_GPT_total_cost_charlotte_l153_15346

noncomputable def regular_rate : ℝ := 40.00
noncomputable def discount_rate : ℝ := 0.25
noncomputable def number_of_people : ℕ := 5

theorem total_cost_charlotte :
  number_of_people * (regular_rate * (1 - discount_rate)) = 150.00 := by
  sorry

end NUMINAMATH_GPT_total_cost_charlotte_l153_15346


namespace NUMINAMATH_GPT_find_x2_plus_y2_l153_15366

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + 2 * x + 2 * y = 88) :
    x^2 + y^2 = 304 / 9 := sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l153_15366


namespace NUMINAMATH_GPT_ratio_c_d_l153_15394

theorem ratio_c_d (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) : 
  c / d = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_c_d_l153_15394


namespace NUMINAMATH_GPT_distance_between_adjacent_symmetry_axes_l153_15303

noncomputable def f (x : ℝ) : ℝ := (Real.cos (3 * x))^2 - 1/2

theorem distance_between_adjacent_symmetry_axes :
  (∃ x : ℝ, f x = f (x + π / 3)) → (∃ d : ℝ, d = π / 6) :=
by
  -- Prove the distance is π / 6 based on the properties of f(x).
  sorry

end NUMINAMATH_GPT_distance_between_adjacent_symmetry_axes_l153_15303


namespace NUMINAMATH_GPT_angle_halving_quadrant_l153_15373

theorem angle_halving_quadrant (k : ℤ) (α : ℝ) 
  (h : k * 360 + 180 < α ∧ α < k * 360 + 270) : 
  k * 180 + 90 < α / 2 ∧ α / 2 < k * 180 + 135 :=
sorry

end NUMINAMATH_GPT_angle_halving_quadrant_l153_15373


namespace NUMINAMATH_GPT_sum_of_interior_diagonals_of_box_l153_15399

theorem sum_of_interior_diagonals_of_box (a b c : ℝ) 
  (h_edges : 4 * (a + b + c) = 60)
  (h_surface_area : 2 * (a * b + b * c + c * a) = 150) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 20 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_diagonals_of_box_l153_15399


namespace NUMINAMATH_GPT_cost_of_one_pencil_and_one_pen_l153_15365

variables (x y : ℝ)

def eq1 := 4 * x + 3 * y = 3.70
def eq2 := 3 * x + 4 * y = 4.20

theorem cost_of_one_pencil_and_one_pen (h₁ : eq1 x y) (h₂ : eq2 x y) :
  x + y = 1.1286 :=
sorry

end NUMINAMATH_GPT_cost_of_one_pencil_and_one_pen_l153_15365


namespace NUMINAMATH_GPT_total_shirts_made_l153_15397

def shirtsPerMinute := 6
def minutesWorkedYesterday := 12
def shirtsMadeToday := 14

theorem total_shirts_made : shirtsPerMinute * minutesWorkedYesterday + shirtsMadeToday = 86 := by
  sorry

end NUMINAMATH_GPT_total_shirts_made_l153_15397


namespace NUMINAMATH_GPT_books_left_over_l153_15339

theorem books_left_over 
  (n_boxes : ℕ) (books_per_box : ℕ) (books_per_new_box : ℕ)
  (total_books : ℕ) (full_boxes : ℕ) (books_left : ℕ) : 
  n_boxes = 1421 → 
  books_per_box = 27 → 
  books_per_new_box = 35 →
  total_books = n_boxes * books_per_box →
  full_boxes = total_books / books_per_new_box →
  books_left = total_books % books_per_new_box →
  books_left = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_books_left_over_l153_15339


namespace NUMINAMATH_GPT_product_of_solutions_eq_neg_35_l153_15361

theorem product_of_solutions_eq_neg_35 :
  ∀ (x : ℝ), -35 = -x^2 - 2 * x → ∃ (p : ℝ), p = -35 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_product_of_solutions_eq_neg_35_l153_15361


namespace NUMINAMATH_GPT_inequality_not_always_hold_l153_15348

theorem inequality_not_always_hold (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) : ¬(∀ a b, a^3 + b^3 ≥ 2 * a * b^2) :=
sorry

end NUMINAMATH_GPT_inequality_not_always_hold_l153_15348


namespace NUMINAMATH_GPT_find_number_l153_15316

theorem find_number : ∃ x : ℝ, 0 < x ∧ x + 17 = 60 * (1 / x) ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l153_15316


namespace NUMINAMATH_GPT_lizette_has_813_stamps_l153_15314

def minervas_stamps : ℕ := 688
def additional_stamps : ℕ := 125
def lizettes_stamps : ℕ := minervas_stamps + additional_stamps

theorem lizette_has_813_stamps : lizettes_stamps = 813 := by
  sorry

end NUMINAMATH_GPT_lizette_has_813_stamps_l153_15314


namespace NUMINAMATH_GPT_perpendicular_lines_with_foot_l153_15341

theorem perpendicular_lines_with_foot (n : ℝ) : 
  (∀ x y, 10 * x + 4 * y - 2 = 0 ↔ 2 * x - 5 * y + n = 0) ∧
  (2 * 1 - 5 * (-2) + n = 0) → n = -12 := 
by sorry

end NUMINAMATH_GPT_perpendicular_lines_with_foot_l153_15341


namespace NUMINAMATH_GPT_sum_solutions_eq_16_l153_15368

theorem sum_solutions_eq_16 (x y : ℝ) 
  (h1 : |x - 5| = |y - 11|)
  (h2 : |x - 11| = 2 * |y - 5|)
  (h3 : x + y = 16) :
  x + y = 16 :=
by
  sorry

end NUMINAMATH_GPT_sum_solutions_eq_16_l153_15368


namespace NUMINAMATH_GPT_digit_B_divisibility_l153_15378

theorem digit_B_divisibility (B : ℕ) (h : 4 * 1000 + B * 100 + B * 10 + 6 % 11 = 0) : B = 5 :=
sorry

end NUMINAMATH_GPT_digit_B_divisibility_l153_15378


namespace NUMINAMATH_GPT_find_k_l153_15324

/--
Given a system of linear equations:
1) x + 2 * y = -a + 1
2) x - 3 * y = 4 * a + 6
If the expression k * x - y remains unchanged regardless of the value of the constant a, 
show that k = -1.
-/
theorem find_k 
  (a x y k : ℝ) 
  (h1 : x + 2 * y = -a + 1) 
  (h2 : x - 3 * y = 4 * a + 6)
  (h3 : ∀ a₁ a₂ x₁ x₂ y₁ y₂, (x₁ + 2 * y₁ = -a₁ + 1) → (x₁ - 3 * y₁ = 4 * a₁ + 6) → 
                               (x₂ + 2 * y₂ = -a₂ + 1) → (x₂ - 3 * y₂ = 4 * a₂ + 6) → 
                               (k * x₁ - y₁ = k * x₂ - y₂)) : 
  k = -1 :=
  sorry

end NUMINAMATH_GPT_find_k_l153_15324


namespace NUMINAMATH_GPT_product_of_modified_numbers_less_l153_15333

theorem product_of_modified_numbers_less
  {a b c : ℝ}
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c := 
by {
   sorry
}

end NUMINAMATH_GPT_product_of_modified_numbers_less_l153_15333


namespace NUMINAMATH_GPT_distance_to_destination_l153_15313

theorem distance_to_destination (x : ℕ) 
    (condition_1 : True)  -- Manex is a tour bus driver. Ignore in the proof.
    (condition_2 : True)  -- Ignores the fact that the return trip is using a different path.
    (condition_3 : x / 30 + (x + 10) / 30 + 2 = 6) : 
    x = 55 :=
sorry

end NUMINAMATH_GPT_distance_to_destination_l153_15313


namespace NUMINAMATH_GPT_problem_statement_l153_15398

theorem problem_statement (x y z t : ℝ) (h : (x + y) / (y + z) = (z + t) / (t + x)) : x * (z + t + y) = z * (x + y + t) :=
sorry

end NUMINAMATH_GPT_problem_statement_l153_15398


namespace NUMINAMATH_GPT_find_range_of_a_l153_15392

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^3) / 3 - (a / 2) * x^2 + x + 1

def is_monotonically_decreasing_in (a : ℝ) (x : ℝ) : Prop := 
  ∀ s t : ℝ, (s ∈ Set.Ioo (3 / 2) 4) ∧ (t ∈ Set.Ioo (3 / 2) 4) ∧ s < t → 
  f a t ≤ f a s

theorem find_range_of_a :
  ∀ a : ℝ, is_monotonically_decreasing_in a x → 
  a ∈ Set.Ici (17/4)
:= sorry

end NUMINAMATH_GPT_find_range_of_a_l153_15392


namespace NUMINAMATH_GPT_range_of_m_l153_15321

noncomputable def common_points (k : ℝ) (m : ℝ) := 
  ∃ x y : ℝ, (y = k * x + 1) ∧ ((x^2 / 5) + (y^2 / m) = 1)

theorem range_of_m (k : ℝ) (m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, (y = k * x + 1) ∧ ((x^2 / 5) + (y^2 / m) = 1)) ↔ 
  (m ∈ (Set.Ioo 1 5 ∪ Set.Ioi 5)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l153_15321


namespace NUMINAMATH_GPT_solution_in_quadrant_IV_l153_15391

theorem solution_in_quadrant_IV (k : ℝ) :
  (∃ x y : ℝ, x + 2 * y = 4 ∧ k * x - y = 1 ∧ x > 0 ∧ y < 0) ↔ (-1 / 2 < k ∧ k < 2) :=
by
  sorry

end NUMINAMATH_GPT_solution_in_quadrant_IV_l153_15391


namespace NUMINAMATH_GPT_find_a_2013_l153_15335

def sequence_a (n : ℕ) : ℤ :=
  if n = 0 then 2
  else if n = 1 then 5
  else sequence_a (n - 1) - sequence_a (n - 2)

theorem find_a_2013 :
  sequence_a 2013 = 3 :=
sorry

end NUMINAMATH_GPT_find_a_2013_l153_15335


namespace NUMINAMATH_GPT_georges_final_score_l153_15355

theorem georges_final_score :
  (6 + 4) * 3 = 30 := 
by
  sorry

end NUMINAMATH_GPT_georges_final_score_l153_15355


namespace NUMINAMATH_GPT_find_x_plus_y_l153_15302

theorem find_x_plus_y (x y : ℝ)
  (h1 : (x - 1)^3 + 2015 * (x - 1) = -1)
  (h2 : (y - 1)^3 + 2015 * (y - 1) = 1)
  : x + y = 2 :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l153_15302


namespace NUMINAMATH_GPT_sum_x_y_650_l153_15349

theorem sum_x_y_650 (x y : ℤ) (h1 : x - y = 200) (h2 : y = 225) : x + y = 650 :=
by
  sorry

end NUMINAMATH_GPT_sum_x_y_650_l153_15349


namespace NUMINAMATH_GPT_solution_set_l153_15301

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

theorem solution_set:
  ∀ x : ℝ, x > -1 ∧ x < 1/3 → f (2*x + 1) < f x := 
by
  sorry

end NUMINAMATH_GPT_solution_set_l153_15301


namespace NUMINAMATH_GPT_arithmetic_seq_general_formula_l153_15322

-- Definitions based on given conditions
def f (x : ℝ) := x^2 - 2*x + 4
def a (n : ℕ) (d : ℝ) := f (d + n - 1) 

-- The general term formula for the arithmetic sequence
theorem arithmetic_seq_general_formula (d : ℝ) :
  (a 1 d = f (d - 1)) →
  (a 3 d = f (d + 1)) →
  (∀ n : ℕ, a n d = 2*n + 1) :=
by
  intros h1 h3
  sorry

end NUMINAMATH_GPT_arithmetic_seq_general_formula_l153_15322


namespace NUMINAMATH_GPT_exists_equal_mod_p_l153_15323

theorem exists_equal_mod_p (p : ℕ) [hp_prime : Fact p.Prime] 
  (m : Fin p → ℕ) 
  (h_consecutive : ∀ i j : Fin p, (i : ℕ) < j → m i + 1 = m j) 
  (sigma : Equiv (Fin p) (Fin p)) :
  ∃ (k l : Fin p), k ≠ l ∧ (m k * m (sigma k) - m l * m (sigma l)) % p = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_equal_mod_p_l153_15323


namespace NUMINAMATH_GPT_ratio_of_areas_l153_15319

-- Define the conditions
variable (s : ℝ) (h_pos : s > 0)
-- The total perimeter of four small square pens is reused for one large square pen
def total_fencing_length := 16 * s
def large_square_side_length := 4 * s

-- Define the areas
def small_squares_total_area := 4 * s^2
def large_square_area := (4 * s)^2

-- The statement to prove
theorem ratio_of_areas : small_squares_total_area / large_square_area = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l153_15319


namespace NUMINAMATH_GPT_correlation_is_1_3_4_l153_15345

def relationship1 := "The relationship between a person's age and their wealth"
def relationship2 := "The relationship between a point on a curve and its coordinates"
def relationship3 := "The relationship between apple production and climate"
def relationship4 := "The relationship between the diameter of the cross-section and the height of the same type of tree in a forest"

def isCorrelation (rel: String) : Bool :=
  if rel == relationship1 ∨ rel == relationship3 ∨ rel == relationship4 then true else false

theorem correlation_is_1_3_4 :
  {relationship1, relationship3, relationship4} = {r | isCorrelation r = true} := 
by
  sorry

end NUMINAMATH_GPT_correlation_is_1_3_4_l153_15345


namespace NUMINAMATH_GPT_probability_one_each_item_l153_15352

theorem probability_one_each_item :
  let num_items := 32
  let total_ways := Nat.choose num_items 4
  let favorable_outcomes := 8 * 8 * 8 * 8
  total_ways = 35960 →
  let probability := favorable_outcomes / total_ways
  probability = (128 : ℚ) / 1125 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_each_item_l153_15352


namespace NUMINAMATH_GPT_intersection_range_l153_15387

noncomputable def function1 (x : ℝ) : ℝ := abs (x^2 - 1) / (x - 1)
noncomputable def function2 (k x : ℝ) : ℝ := k * x - 2

theorem intersection_range (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ function1 x₁ = function2 k x₁ ∧ function1 x₂ = function2 k x₂) ↔ 
  (0 < k ∧ k < 1) ∨ (1 < k ∧ k < 4) := 
sorry

end NUMINAMATH_GPT_intersection_range_l153_15387


namespace NUMINAMATH_GPT_range_of_target_function_l153_15338

noncomputable def target_function (x : ℝ) : ℝ :=
  1 - 1 / (x^2 - 1)

theorem range_of_target_function :
  ∀ y : ℝ, ∃ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ target_function x = y ↔ y ∈ (Set.Iio 1 ∪ Set.Ici 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_target_function_l153_15338


namespace NUMINAMATH_GPT_total_trees_planted_total_trees_when_a_100_l153_15369

-- Define the number of trees planted by each team based on 'a'
def trees_first_team (a : ℕ) : ℕ := a
def trees_second_team (a : ℕ) : ℕ := 2 * a + 8
def trees_third_team (a : ℕ) : ℕ := (2 * a + 8) / 2 - 6

-- Define the total number of trees
def total_trees (a : ℕ) : ℕ := 
  trees_first_team a + trees_second_team a + trees_third_team a

-- The main theorem
theorem total_trees_planted (a : ℕ) : total_trees a = 4 * a + 6 :=
by
  sorry

-- The specific calculation when a = 100
theorem total_trees_when_a_100 : total_trees 100 = 406 :=
by
  sorry

end NUMINAMATH_GPT_total_trees_planted_total_trees_when_a_100_l153_15369


namespace NUMINAMATH_GPT_oldest_person_Jane_babysat_age_l153_15311

def Jane_current_age : ℕ := 32
def Jane_stop_babysitting_age : ℕ := 22 -- 32 - 10
def max_child_age_when_Jane_babysat : ℕ := Jane_stop_babysitting_age / 2  -- 22 / 2
def years_since_Jane_stopped : ℕ := Jane_current_age - Jane_stop_babysitting_age -- 32 - 22

theorem oldest_person_Jane_babysat_age :
  max_child_age_when_Jane_babysat + years_since_Jane_stopped = 21 :=
by
  sorry

end NUMINAMATH_GPT_oldest_person_Jane_babysat_age_l153_15311


namespace NUMINAMATH_GPT_opposite_negative_nine_l153_15379

theorem opposite_negative_nine : 
  (∃ (y : ℤ), -9 + y = 0 ∧ y = 9) :=
by sorry

end NUMINAMATH_GPT_opposite_negative_nine_l153_15379


namespace NUMINAMATH_GPT_sequence_satisfies_conditions_l153_15347

theorem sequence_satisfies_conditions : 
  let seq1 := [4, 1, 3, 1, 2, 4, 3, 2]
  let seq2 := [2, 3, 4, 2, 1, 3, 1, 4]
  (seq1[0] = 4 ∧ seq1[1] = 1 ∧ seq1[2] = 3 ∧ seq1[3] = 1 ∧ seq1[4] = 2 ∧ seq1[5] = 4 ∧ seq1[6] = 3 ∧ seq1[7] = 2)
  ∨ (seq2[0] = 2 ∧ seq2[1] = 3 ∧ seq2[2] = 4 ∧ seq2[3] = 2 ∧ seq2[4] = 1 ∧ seq2[5] = 3 ∧ seq2[6] = 1 ∧ seq2[7] = 4)
  ∧ (seq1[1] = 1 ∧ seq1[3] - seq1[1] = 2 ∧ seq1[4] - seq1[2] = 3 ∧ seq1[5] - seq1[2] = 4) := 
  sorry

end NUMINAMATH_GPT_sequence_satisfies_conditions_l153_15347


namespace NUMINAMATH_GPT_best_play_wins_probability_best_play_wins_with_certainty_l153_15300

-- Define the conditions

variables (n : ℕ)

-- Part (a): Probability that the best play wins
theorem best_play_wins_probability (hn_pos : 0 < n) : 
  1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) = 1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) :=
  by sorry

-- Part (b): With more than two plays, the best play wins with certainty
theorem best_play_wins_with_certainty (s : ℕ) (hs : 2 < s) : 
  1 = 1 :=
  by sorry

end NUMINAMATH_GPT_best_play_wins_probability_best_play_wins_with_certainty_l153_15300


namespace NUMINAMATH_GPT_roots_value_l153_15326

theorem roots_value (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X ^ 2) = 0) (h2 : Polynomial.eval n (Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X ^ 2) = 0) : m^2 + 4 * m + n = -2 := 
sorry

end NUMINAMATH_GPT_roots_value_l153_15326


namespace NUMINAMATH_GPT_boys_from_other_communities_l153_15372

theorem boys_from_other_communities :
  ∀ (total_boys : ℕ) (percentage_muslims percentage_hindus percentage_sikhs : ℕ),
  total_boys = 400 →
  percentage_muslims = 44 →
  percentage_hindus = 28 →
  percentage_sikhs = 10 →
  (total_boys * (100 - (percentage_muslims + percentage_hindus + percentage_sikhs)) / 100) = 72 := 
by
  intros total_boys percentage_muslims percentage_hindus percentage_sikhs h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_boys_from_other_communities_l153_15372


namespace NUMINAMATH_GPT_intersection_points_count_l153_15388

variables {R : Type*} [LinearOrderedField R]

def line1 (x y : R) : Prop := 3 * y - 2 * x = 1
def line2 (x y : R) : Prop := x + 2 * y = 2
def line3 (x y : R) : Prop := 4 * x - 6 * y = 5

theorem intersection_points_count : 
  ∃ p1 p2 : R × R, 
   (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
   (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
   p1 ≠ p2 ∧ 
   (∀ p : R × R, (line1 p.1 p.2 ∧ line3 p.1 p.2) → False) := 
sorry

end NUMINAMATH_GPT_intersection_points_count_l153_15388


namespace NUMINAMATH_GPT_moles_of_MgCO3_formed_l153_15304

theorem moles_of_MgCO3_formed 
  (moles_MgO : ℕ) (moles_CO2 : ℕ)
  (h_eq : moles_MgO = 3 ∧ moles_CO2 = 3)
  (balanced_eq : ∀ n : ℕ, n * MgO + n * CO2 = n * MgCO3) : 
  moles_MgCO3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_MgCO3_formed_l153_15304


namespace NUMINAMATH_GPT_range_of_a_l153_15342

open Classical

noncomputable def parabola_line_common_point_range (a : ℝ) : Prop :=
  ∃ (k : ℝ), ∃ (x : ℝ), ∃ (y : ℝ), 
  (y = a * x ^ 2) ∧ ((y + 2 = k * (x - 1)) ∨ (y + 2 = - (1 / k) * (x - 1)))

theorem range_of_a (a : ℝ) : 
  (∃ k : ℝ, ∃ x : ℝ, ∃ y : ℝ, 
    y = a * x ^ 2 ∧ (y + 2 = k * (x - 1) ∨ y + 2 = - (1 / k) * (x - 1))) ↔ 
  0 < a ∧ a <= 1 / 8 :=
sorry

end NUMINAMATH_GPT_range_of_a_l153_15342


namespace NUMINAMATH_GPT_plane_divided_into_four_regions_l153_15390

-- Definition of the conditions
def line1 (x y : ℝ) : Prop := y = 3 * x
def line2 (x y : ℝ) : Prop := y = (1 / 3) * x

-- Proof statement
theorem plane_divided_into_four_regions :
  (∃ f g : ℝ → ℝ, ∀ x, line1 x (f x) ∧ line2 x (g x)) →
  ∃ n : ℕ, n = 4 :=
by sorry

end NUMINAMATH_GPT_plane_divided_into_four_regions_l153_15390


namespace NUMINAMATH_GPT_unique_intersection_point_l153_15374

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (Real.log 3 / Real.log x)
noncomputable def h (x : ℝ) : ℝ := 3 - (1 / Real.sqrt (Real.log 3 / Real.log x))

theorem unique_intersection_point : (∃! (x : ℝ), (x > 0) ∧ (f x = g x ∨ f x = h x ∨ g x = h x)) :=
sorry

end NUMINAMATH_GPT_unique_intersection_point_l153_15374


namespace NUMINAMATH_GPT_slopes_product_l153_15315

variables {a b c x0 y0 alpha beta : ℝ}
variables {P Q : ℝ × ℝ}
variables (M : ℝ × ℝ) (kPQ kOM : ℝ)

-- Conditions: a, b are positive real numbers
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Condition: b^2 = a c
axiom b_squared_eq_a_mul_c : b^2 = a * c

-- Condition: P and Q lie on the hyperbola
axiom P_on_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1
axiom Q_on_hyperbola : (Q.1^2 / a^2) - (Q.2^2 / b^2) = 1

-- Condition: M is the midpoint of P and Q
axiom M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Condition: Slopes kPQ and kOM exist
axiom kOM_def : kOM = y0 / x0
axiom kPQ_def : kPQ = beta / alpha

-- Theorem: Value of the product of the slopes
theorem slopes_product : kPQ * kOM = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_slopes_product_l153_15315


namespace NUMINAMATH_GPT_quadrilateral_inequality_l153_15377

variable (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)

def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2)^2 + (y1 - y2)^2

-- Given a convex quadrilateral ABCD with vertices at (x1, y1), (x2, y2), (x3, y3), and (x4, y4), prove:
theorem quadrilateral_inequality :
  (distance_squared x1 y1 x3 y3 + distance_squared x2 y2 x4 y4) ≤
  (distance_squared x1 y1 x2 y2 + distance_squared x2 y2 x3 y3 +
   distance_squared x3 y3 x4 y4 + distance_squared x4 y4 x1 y1) :=
sorry

end NUMINAMATH_GPT_quadrilateral_inequality_l153_15377


namespace NUMINAMATH_GPT_count_not_divisible_by_2_3_5_l153_15309

theorem count_not_divisible_by_2_3_5 : 
  let count_div_2 := (100 / 2)
  let count_div_3 := (100 / 3)
  let count_div_5 := (100 / 5)
  let count_div_6 := (100 / 6)
  let count_div_10 := (100 / 10)
  let count_div_15 := (100 / 15)
  let count_div_30 := (100 / 30)
  100 - (count_div_2 + count_div_3 + count_div_5) 
      + (count_div_6 + count_div_10 + count_div_15) 
      - count_div_30 = 26 :=
by
  let count_div_2 := 50
  let count_div_3 := 33
  let count_div_5 := 20
  let count_div_6 := 16
  let count_div_10 := 10
  let count_div_15 := 6
  let count_div_30 := 3
  sorry

end NUMINAMATH_GPT_count_not_divisible_by_2_3_5_l153_15309


namespace NUMINAMATH_GPT_relationship_among_abcd_l153_15340

theorem relationship_among_abcd (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := 
by
  sorry

end NUMINAMATH_GPT_relationship_among_abcd_l153_15340


namespace NUMINAMATH_GPT_range_of_a3_l153_15364

theorem range_of_a3 (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) + a n = 4 * n + 3)
  (h2 : ∀ n : ℕ, n > 0 → a n + 2 * n^2 ≥ 0) 
  : 2 ≤ a 3 ∧ a 3 ≤ 19 := 
sorry

end NUMINAMATH_GPT_range_of_a3_l153_15364


namespace NUMINAMATH_GPT_find_ratio_l153_15356

variable (x y : ℝ)

-- Hypotheses: x and y are distinct real numbers and the given equation holds
variable (h₁ : x ≠ y)
variable (h₂ : x / y + (x + 15 * y) / (y + 15 * x) = 3)

-- We aim to prove that x / y = 0.8
theorem find_ratio (h₁ : x ≠ y) (h₂ : x / y + (x + 15 * y) / (y + 15 * x) = 3) : x / y = 0.8 :=
sorry

end NUMINAMATH_GPT_find_ratio_l153_15356


namespace NUMINAMATH_GPT_num_cars_can_be_parked_l153_15381

theorem num_cars_can_be_parked (length width : ℝ) (useable_percentage : ℝ) (area_per_car : ℝ) 
  (h_length : length = 400) (h_width : width = 500) (h_useable_percentage : useable_percentage = 0.80) 
  (h_area_per_car : area_per_car = 10) : 
  length * width * useable_percentage / area_per_car = 16000 := 
by 
  sorry

end NUMINAMATH_GPT_num_cars_can_be_parked_l153_15381


namespace NUMINAMATH_GPT_eq_zero_l153_15317

variable {x y z : ℤ}

theorem eq_zero (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_GPT_eq_zero_l153_15317


namespace NUMINAMATH_GPT_weight_of_new_person_l153_15307

/-- 
The average weight of 10 persons increases by 6.3 kg when a new person replaces one of them. 
The weight of the replaced person is 65 kg. 
Prove that the weight of the new person is 128 kg. 
-/
theorem weight_of_new_person (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : 
  (avg_increase = 6.3) → 
  (old_weight = 65) → 
  (new_weight = old_weight + 10 * avg_increase) → 
  new_weight = 128 := 
by
  intros
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l153_15307


namespace NUMINAMATH_GPT_smallest_y_exists_l153_15327

theorem smallest_y_exists (M : ℤ) (y : ℕ) (h : 2520 * y = M ^ 3) : y = 3675 :=
by
  have h_factorization : 2520 = 2^3 * 3^2 * 5 * 7 := sorry
  sorry

end NUMINAMATH_GPT_smallest_y_exists_l153_15327


namespace NUMINAMATH_GPT_find_other_number_l153_15320

/-- Given HCF(A, B), LCM(A, B), and a known A, proves the value of B. -/
theorem find_other_number (A B : ℕ) 
  (hcf : Nat.gcd A B = 16) 
  (lcm : Nat.lcm A B = 396) 
  (a_val : A = 36) : B = 176 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l153_15320


namespace NUMINAMATH_GPT_numerical_puzzle_solution_l153_15367

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end NUMINAMATH_GPT_numerical_puzzle_solution_l153_15367


namespace NUMINAMATH_GPT_abs_abc_eq_abs_k_l153_15329

variable {a b c k : ℝ}

noncomputable def distinct_nonzero (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem abs_abc_eq_abs_k (h_distinct : distinct_nonzero a b c)
                          (h_nonzero_k : k ≠ 0)
                          (h_eq : a + k / b = b + k / c ∧ b + k / c = c + k / a) :
  |a * b * c| = |k| :=
by
  sorry

end NUMINAMATH_GPT_abs_abc_eq_abs_k_l153_15329


namespace NUMINAMATH_GPT_volume_eq_three_times_other_two_l153_15318

-- declare the given ratio of the radii
def r1 : ℝ := 1
def r2 : ℝ := 2
def r3 : ℝ := 3

-- calculate the volumes based on the given radii
noncomputable def V (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- defining the volumes of the three spheres
noncomputable def V1 : ℝ := V r1
noncomputable def V2 : ℝ := V r2
noncomputable def V3 : ℝ := V r3

theorem volume_eq_three_times_other_two : V3 = 3 * (V1 + V2) := 
by
  sorry

end NUMINAMATH_GPT_volume_eq_three_times_other_two_l153_15318


namespace NUMINAMATH_GPT_find_k_l153_15312

-- Define the arithmetic sequence and the sum of the first n terms
def a (n : ℕ) : ℤ := 2 * n + 2
def S (n : ℕ) : ℤ := n^2 + 3 * n

-- The main assertion
theorem find_k : ∃ (k : ℕ), k > 0 ∧ (S k - a (k + 5) = 44) ∧ k = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l153_15312


namespace NUMINAMATH_GPT_find_original_price_l153_15331

theorem find_original_price (SP GP : ℝ) (h_SP : SP = 1150) (h_GP : GP = 27.77777777777778) :
  ∃ CP : ℝ, CP = 900 :=
by
  sorry

end NUMINAMATH_GPT_find_original_price_l153_15331


namespace NUMINAMATH_GPT_binary_addition_l153_15305

def bin_to_dec1 := 511  -- 111111111_2 in decimal
def bin_to_dec2 := 127  -- 1111111_2 in decimal

theorem binary_addition : bin_to_dec1 + bin_to_dec2 = 638 := by
  sorry

end NUMINAMATH_GPT_binary_addition_l153_15305


namespace NUMINAMATH_GPT_geom_seq_product_l153_15344

noncomputable def geom_seq (a : ℕ → ℝ) := 
∀ n m: ℕ, ∃ r : ℝ, a (n + m) = a n * r ^ m

theorem geom_seq_product (a : ℕ → ℝ) 
  (h_seq : geom_seq a) 
  (h_pos : ∀ n, 0 < a n) 
  (h_log_sum : Real.log (a 3) + Real.log (a 6) + Real.log (a 9) = 3) : 
  a 1 * a 11 = 100 := 
sorry

end NUMINAMATH_GPT_geom_seq_product_l153_15344


namespace NUMINAMATH_GPT_triangle_side_possible_values_l153_15325

theorem triangle_side_possible_values (m : ℝ) (h1 : 1 < m) (h2 : m < 7) : 
  m = 5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_possible_values_l153_15325


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l153_15380

-- Problem 1
theorem system1_solution (x z : ℤ) (h1 : 3 * x - 5 * z = 6) (h2 : x + 4 * z = -15) : x = -3 ∧ z = -3 :=
by
  sorry

-- Problem 2
theorem system2_solution (x y : ℚ) 
 (h1 : ((2 * x - 1) / 5) + ((3 * y - 2) / 4) = 2) 
 (h2 : ((3 * x + 1) / 5) - ((3 * y + 2) / 4) = 0) : x = 3 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l153_15380


namespace NUMINAMATH_GPT_quadratic_real_roots_range_find_k_l153_15306

theorem quadratic_real_roots_range (k : ℝ) (h : ∃ x1 x2 : ℝ, x^2 - 2 * (k - 1) * x + k^2 = 0):
  k ≤ 1/2 :=
  sorry

theorem find_k (k : ℝ) (x1 x2 : ℝ) (h₁ : x^2 - 2 * (k - 1) * x + k^2 = 0)
  (h₂ : x₁ * x₂ + x₁ + x₂ - 1 = 0) (h_range : k ≤ 1/2) :
    k = -3 :=
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_find_k_l153_15306


namespace NUMINAMATH_GPT_number_of_students_l153_15393

variables (m d r : ℕ) (k : ℕ)

theorem number_of_students :
  (30 < m + d ∧ m + d < 40) → (r = 3 * m) → (r = 5 * d) → m + d = 32 :=
by 
  -- The proof body is not necessary here according to instructions.
  sorry

end NUMINAMATH_GPT_number_of_students_l153_15393


namespace NUMINAMATH_GPT_magnitude_of_a_l153_15375

open Real

-- Assuming the standard inner product space for vectors in Euclidean space

variables (a b : ℝ) -- Vectors in R^n (could be general but simplified to real numbers for this example)
variable (θ : ℝ)    -- Angle between vectors
axiom angle_ab : θ = 60 -- Given angle between vectors

-- Conditions:
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0
axiom norm_b : abs b = 1
axiom norm_2a_minus_b : abs (2 * a - b) = 1

-- To prove:
theorem magnitude_of_a : abs a = 1 / 2 :=
sorry

end NUMINAMATH_GPT_magnitude_of_a_l153_15375


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l153_15376

-- Define the conditions given in the problem
def asymptote_equation_related (a b : ℝ) : Prop := a / b = 3 / 4
def hyperbola_eccentricity_relation (a c : ℝ) : Prop := c^2 / a^2 = 25 / 9

-- Define the proof problem
theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : asymptote_equation_related a b)
  (h2 : hyperbola_eccentricity_relation a c)
  (he : e = c / a) :
  e = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l153_15376


namespace NUMINAMATH_GPT_calc_x_equals_condition_l153_15353

theorem calc_x_equals_condition (m n p q x : ℝ) :
  x^2 + (2 * m * p + 2 * n * q) ^ 2 + (2 * m * q - 2 * n * p) ^ 2 = (m ^ 2 + n ^ 2 + p ^ 2 + q ^ 2) ^ 2 →
  x = m ^ 2 + n ^ 2 - p ^ 2 - q ^ 2 ∨ x = - m ^ 2 - n ^ 2 + p ^ 2 + q ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_calc_x_equals_condition_l153_15353


namespace NUMINAMATH_GPT_cake_remaining_l153_15336

theorem cake_remaining (T J: ℝ) (h1: T = 0.60) (h2: J = 0.25) :
  (1 - ((1 - T) * J + T)) = 0.30 :=
by
  sorry

end NUMINAMATH_GPT_cake_remaining_l153_15336


namespace NUMINAMATH_GPT_intersection_A_B_l153_15371

def A (x : ℝ) : Prop := 0 < x ∧ x < 2
def B (x : ℝ) : Prop := -1 < x ∧ x < 1
def C (x : ℝ) : Prop := 0 < x ∧ x < 1

theorem intersection_A_B : ∀ x, A x ∧ B x ↔ C x := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l153_15371


namespace NUMINAMATH_GPT_decimal_to_fraction_equiv_l153_15359

theorem decimal_to_fraction_equiv : (0.38 : ℝ) = 19 / 50 :=
by
  sorry

end NUMINAMATH_GPT_decimal_to_fraction_equiv_l153_15359


namespace NUMINAMATH_GPT_line_through_point_equal_intercepts_l153_15383

theorem line_through_point_equal_intercepts (P : ℝ × ℝ) (hP : P = (1, 1)) :
  (∀ x y : ℝ, (x - y = 0 ∨ x + y - 2 = 0) → ∃ k : ℝ, k = 1 ∧ k = 2) :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_equal_intercepts_l153_15383


namespace NUMINAMATH_GPT_abs_diff_probs_l153_15358

def numRedMarbles := 1000
def numBlackMarbles := 1002
def totalMarbles := numRedMarbles + numBlackMarbles

def probSame : ℚ := 
  ((numRedMarbles * (numRedMarbles - 1)) / 2 + (numBlackMarbles * (numBlackMarbles - 1)) / 2) / (totalMarbles * (totalMarbles - 1) / 2)

def probDiff : ℚ :=
  (numRedMarbles * numBlackMarbles) / (totalMarbles * (totalMarbles - 1) / 2)

theorem abs_diff_probs : |probSame - probDiff| = 999 / 2003001 := 
by {
  sorry
}

end NUMINAMATH_GPT_abs_diff_probs_l153_15358


namespace NUMINAMATH_GPT_cause_of_polarization_by_electronegativity_l153_15384

-- Definition of the problem conditions as hypotheses
def strong_polarization_of_CH_bond (C_H_bond : Prop) (electronegativity : Prop) : Prop 
  := C_H_bond ∧ electronegativity

-- Given conditions: Carbon atom is in sp hybridization and C-H bond shows strong polarization
axiom carbon_sp_hybridized : Prop
axiom CH_bond_strong_polarization : Prop

-- Question: The cause of strong polarization of the C-H bond at the carbon atom in sp hybridization in alkynes
def cause_of_strong_polarization (sp_hybridization : Prop) : Prop 
  := true  -- This definition will hold as a placeholder, to indicate there is a causal connection

-- Correct answer: high electronegativity of the carbon atom in sp-hybrid state causes strong polarization
theorem cause_of_polarization_by_electronegativity 
  (high_electronegativity : Prop) 
  (sp_hybridized : Prop) 
  (polarized : Prop) 
  (H : strong_polarization_of_CH_bond polarized high_electronegativity) 
  : sp_hybridized ∧ polarized := 
  sorry

end NUMINAMATH_GPT_cause_of_polarization_by_electronegativity_l153_15384


namespace NUMINAMATH_GPT_intersection_of_intervals_l153_15330

theorem intersection_of_intervals :
  let A := {x : ℝ | x < -3}
  let B := {x : ℝ | x > -4}
  A ∩ B = {x : ℝ | -4 < x ∧ x < -3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_intervals_l153_15330


namespace NUMINAMATH_GPT_box_volume_l153_15332

theorem box_volume (x : ℕ) (h_ratio : (x > 0)) (V : ℕ) (h_volume : V = 20 * x^3) : V = 160 :=
by
  sorry

end NUMINAMATH_GPT_box_volume_l153_15332


namespace NUMINAMATH_GPT_reciprocal_geometric_sum_l153_15389

/-- The sum of the new geometric progression formed by taking the reciprocal of each term in the original progression,
    where the original progression has 10 terms, the first term is 2, and the common ratio is 3, is \( \frac{29524}{59049} \). -/
theorem reciprocal_geometric_sum :
  let a := 2
  let r := 3
  let n := 10
  let sn := (2 * (1 - r^n)) / (1 - r)
  let sn_reciprocal := (1 / a) * (1 - (1/r)^n) / (1 - 1/r)
  (sn_reciprocal = 29524 / 59049) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_geometric_sum_l153_15389


namespace NUMINAMATH_GPT_find_original_comic_books_l153_15357

def comic_books (X : ℕ) : Prop :=
  X / 2 + 6 = 13

theorem find_original_comic_books (X : ℕ) (h : comic_books X) : X = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_original_comic_books_l153_15357


namespace NUMINAMATH_GPT_joan_jogged_3563_miles_l153_15350

noncomputable def steps_per_mile : ℕ := 1200

noncomputable def flips_per_year : ℕ := 28

noncomputable def steps_per_full_flip : ℕ := 150000

noncomputable def final_day_steps : ℕ := 75000

noncomputable def total_steps_in_year := flips_per_year * steps_per_full_flip + final_day_steps

noncomputable def miles_jogged := total_steps_in_year / steps_per_mile

theorem joan_jogged_3563_miles :
  miles_jogged = 3563 :=
by
  sorry

end NUMINAMATH_GPT_joan_jogged_3563_miles_l153_15350


namespace NUMINAMATH_GPT_triangle_inequality_cubed_l153_15382

theorem triangle_inequality_cubed
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  (a^3 / c^3) + (b^3 / c^3) + (3 * a * b / c^2) > 1 := 
sorry

end NUMINAMATH_GPT_triangle_inequality_cubed_l153_15382


namespace NUMINAMATH_GPT_find_sum_of_abc_l153_15328

variable (a b c : ℝ)

-- Given conditions
axiom h1 : a^2 + a * b + b^2 = 1
axiom h2 : b^2 + b * c + c^2 = 3
axiom h3 : c^2 + c * a + a^2 = 4

-- Positivity constraints
axiom ha : a > 0
axiom hb : b > 0
axiom hc : c > 0

theorem find_sum_of_abc : a + b + c = Real.sqrt 7 := 
by
  sorry

end NUMINAMATH_GPT_find_sum_of_abc_l153_15328


namespace NUMINAMATH_GPT_total_cows_in_ranch_l153_15386

theorem total_cows_in_ranch :
  ∀ (WTP_cows : ℕ) (HGHF_cows : ℕ), WTP_cows = 17 → HGHF_cows = 3 * WTP_cows + 2 → (HGHF_cows + WTP_cows) = 70 :=
by 
  intros WTP_cows HGHF_cows WTP_cows_def HGHF_cows_def
  rw [WTP_cows_def, HGHF_cows_def]
  sorry

end NUMINAMATH_GPT_total_cows_in_ranch_l153_15386
