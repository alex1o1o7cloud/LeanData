import Mathlib

namespace NUMINAMATH_GPT_inequality_solution_set_l1168_116828

theorem inequality_solution_set (a : ℝ) : 
    (a = 0 → (∃ x : ℝ, x > 1 ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a < 0 → (∃ x : ℝ, (x < 2/a ∨ x > 1) ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (0 < a ∧ a < 2 → (∃ x : ℝ, (1 < x ∧ x < 2/a) ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a = 2 → ¬(∃ x : ℝ, ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a > 2 → (∃ x : ℝ, (2/a < x ∧ x < 1) ∧ ax^2 - (a + 2) * x + 2 < 0)) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l1168_116828


namespace NUMINAMATH_GPT_value_of_y_l1168_116897

theorem value_of_y (x y : ℝ) (hx : x = 3) (h : x^(3 * y) = 9) : y = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_value_of_y_l1168_116897


namespace NUMINAMATH_GPT_ellipse_foci_y_axis_range_l1168_116868

theorem ellipse_foci_y_axis_range (k : ℝ) : 
  (2*k - 1 > 2 - k) → (2 - k > 0) → (1 < k ∧ k < 2) := 
by 
  intros h1 h2
  -- We use the assumptions to derive the target statement.
  sorry

end NUMINAMATH_GPT_ellipse_foci_y_axis_range_l1168_116868


namespace NUMINAMATH_GPT_M_intersect_N_l1168_116885

-- Definition of the sets M and N
def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≤ x}

-- Proposition to be proved
theorem M_intersect_N : M ∩ N = {0, 1} := 
by 
  sorry

end NUMINAMATH_GPT_M_intersect_N_l1168_116885


namespace NUMINAMATH_GPT_proof_problem_l1168_116870

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

theorem proof_problem (h_even : even_function f)
                      (h_period : ∀ x, f (x + 2) = -f x)
                      (h_incr : increasing_on f (-2) 0) :
                      periodic_function f 4 ∧ symmetric_about f 2 :=
by { sorry }

end NUMINAMATH_GPT_proof_problem_l1168_116870


namespace NUMINAMATH_GPT_find_m_l1168_116821

variable (m : ℝ)
def vector_a : ℝ × ℝ := (1, 3)
def vector_b : ℝ × ℝ := (m, -2)

theorem find_m (h : (1 + m) + 3 = 0) : m = -4 := by
  sorry

end NUMINAMATH_GPT_find_m_l1168_116821


namespace NUMINAMATH_GPT_difference_of_squares_is_149_l1168_116889

-- Definitions of the conditions
def are_consecutive (n m : ℤ) : Prop := m = n + 1
def sum_less_than_150 (n : ℤ) : Prop := (n + (n + 1)) < 150

-- The difference of their squares
def difference_of_squares (n m : ℤ) : ℤ := (m * m) - (n * n)

-- Stating the problem where the answer expected is 149
theorem difference_of_squares_is_149 :
  ∀ n : ℤ, 
  ∀ m : ℤ,
  are_consecutive n m →
  sum_less_than_150 n →
  difference_of_squares n m = 149 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_is_149_l1168_116889


namespace NUMINAMATH_GPT_implicit_derivative_l1168_116804

noncomputable section

open Real

section ImplicitDifferentiation

variable {x : ℝ} {y : ℝ → ℝ}

def f (x y : ℝ) : ℝ := y^2 + x^2 - 1

theorem implicit_derivative (h : f x (y x) = 0) :
  deriv y x = -x / y x :=
  sorry

end ImplicitDifferentiation

end NUMINAMATH_GPT_implicit_derivative_l1168_116804


namespace NUMINAMATH_GPT_find_k_l1168_116830

noncomputable def line1_slope : ℝ := -1
noncomputable def line2_slope (k : ℝ) : ℝ := -k / 3

theorem find_k (k : ℝ) : 
  (line2_slope k) * line1_slope = -1 → k = -3 := 
by
  sorry

end NUMINAMATH_GPT_find_k_l1168_116830


namespace NUMINAMATH_GPT_find_b_in_triangle_l1168_116893

theorem find_b_in_triangle
  (a b c A B C : ℝ)
  (cos_A : ℝ) (cos_C : ℝ)
  (ha : a = 1)
  (hcos_A : cos_A = 4 / 5)
  (hcos_C : cos_C = 5 / 13) :
  b = 21 / 13 :=
by
  sorry

end NUMINAMATH_GPT_find_b_in_triangle_l1168_116893


namespace NUMINAMATH_GPT_range_of_m_for_distance_l1168_116843

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (|x1 - x2|) + 2 * (|y1 - y2|)

theorem range_of_m_for_distance (m : ℝ) : 
  distance 2 1 (-1) m ≤ 5 ↔ 0 ≤ m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_distance_l1168_116843


namespace NUMINAMATH_GPT_alex_charge_per_trip_l1168_116891

theorem alex_charge_per_trip (x : ℝ)
  (savings_needed : ℝ) (n_trips : ℝ) (worth_groceries : ℝ) (charge_per_grocery_percent : ℝ) :
  savings_needed = 100 → 
  n_trips = 40 →
  worth_groceries = 800 →
  charge_per_grocery_percent = 0.05 →
  n_trips * x + charge_per_grocery_percent * worth_groceries = savings_needed →
  x = 1.5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end NUMINAMATH_GPT_alex_charge_per_trip_l1168_116891


namespace NUMINAMATH_GPT_initial_value_exists_l1168_116811

theorem initial_value_exists (x : ℕ) (h : ∃ k : ℕ, x + 7 = k * 456) : x = 449 :=
sorry

end NUMINAMATH_GPT_initial_value_exists_l1168_116811


namespace NUMINAMATH_GPT_side_length_a_l1168_116864

theorem side_length_a (a b c : ℝ) (B : ℝ) (h1 : a = c - 2 * a * Real.cos B) (h2 : c = 5) (h3 : 3 * a = 2 * b) :
  a = 4 := by
  sorry

end NUMINAMATH_GPT_side_length_a_l1168_116864


namespace NUMINAMATH_GPT_missing_number_l1168_116806

theorem missing_number (n : ℝ) (h : (0.0088 * 4.5) / (0.05 * n * 0.008) = 990) : n = 0.1 :=
sorry

end NUMINAMATH_GPT_missing_number_l1168_116806


namespace NUMINAMATH_GPT_least_num_to_divisible_l1168_116879

theorem least_num_to_divisible (n : ℕ) : (1056 + n) % 27 = 0 → n = 24 :=
by
  sorry

end NUMINAMATH_GPT_least_num_to_divisible_l1168_116879


namespace NUMINAMATH_GPT_intersection_in_first_quadrant_l1168_116898

theorem intersection_in_first_quadrant (a : ℝ) :
  (∃ x y : ℝ, ax - y + 2 = 0 ∧ x + y - a = 0 ∧ x > 0 ∧ y > 0) ↔ a > 2 := 
by
  sorry

end NUMINAMATH_GPT_intersection_in_first_quadrant_l1168_116898


namespace NUMINAMATH_GPT_floor_diff_l1168_116841

theorem floor_diff {x : ℝ} (h : x = 12.7) : 
  (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * (⌊x⌋ : ℤ) = 17 :=
by
  have h1 : x = 12.7 := h
  have hx2 : x^2 = 161.29 := by sorry
  have hfloor : ⌊x⌋ = 12 := by sorry
  have hfloor2 : ⌊x^2⌋ = 161 := by sorry
  sorry

end NUMINAMATH_GPT_floor_diff_l1168_116841


namespace NUMINAMATH_GPT_jellybean_addition_l1168_116853

-- Definitions related to the problem
def initial_jellybeans : ℕ := 37
def removed_jellybeans_initial : ℕ := 15
def added_jellybeans (x : ℕ) : ℕ := x
def removed_jellybeans_again : ℕ := 4
def final_jellybeans : ℕ := 23

-- Prove that the number of jellybeans added back (x) is 5
theorem jellybean_addition (x : ℕ) 
  (h1 : initial_jellybeans - removed_jellybeans_initial + added_jellybeans x - removed_jellybeans_again = final_jellybeans) : 
  x = 5 :=
sorry

end NUMINAMATH_GPT_jellybean_addition_l1168_116853


namespace NUMINAMATH_GPT_triangle_inequality_product_l1168_116839

theorem triangle_inequality_product (x y z : ℝ) (h1 : x + y > z) (h2 : x + z > y) (h3 : y + z > x) :
  (x + y + z) * (x + y - z) * (x + z - y) * (z + y - x) ≤ 4 * x^2 * y^2 := 
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_product_l1168_116839


namespace NUMINAMATH_GPT_mod_neg_result_l1168_116823

-- Define the hypothesis as the residue equivalence and positive range constraint.
theorem mod_neg_result : 
  ∀ (a b : ℤ), (-1277 : ℤ) % 32 = 3 := by
  sorry

end NUMINAMATH_GPT_mod_neg_result_l1168_116823


namespace NUMINAMATH_GPT_abs_diff_simplification_l1168_116829

theorem abs_diff_simplification (a b : ℝ) (h1 : a < 0) (h2 : a * b < 0) : |b - a + 1| - |a - b - 5| = -4 :=
  sorry

end NUMINAMATH_GPT_abs_diff_simplification_l1168_116829


namespace NUMINAMATH_GPT_simplify_expression_l1168_116850

theorem simplify_expression (x : ℝ) :
  (3 * x)^5 + (4 * x^2) * (3 * x^2) = 243 * x^5 + 12 * x^4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1168_116850


namespace NUMINAMATH_GPT_tangent_line_equation_l1168_116803

noncomputable def circle_eq1 (x y : ℝ) := x^2 + (y - 2)^2 - 4
noncomputable def circle_eq2 (x y : ℝ) := (x - 3)^2 + (y + 2)^2 - 21
noncomputable def line_eq (x y : ℝ) := 3*x - 4*y - 4

theorem tangent_line_equation :
  ∀ (x y : ℝ), (circle_eq1 x y = 0 ∧ circle_eq2 x y = 0) ↔ line_eq x y = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_equation_l1168_116803


namespace NUMINAMATH_GPT_primes_satisfying_equation_l1168_116886

theorem primes_satisfying_equation :
  ∀ (p q : ℕ), p.Prime ∧ q.Prime → 
    (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ 
    (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
by
  sorry

end NUMINAMATH_GPT_primes_satisfying_equation_l1168_116886


namespace NUMINAMATH_GPT_original_three_digit_number_a_original_three_digit_number_b_l1168_116899

section ProblemA

variables {x y z : ℕ}

/-- In a three-digit number, the first digit on the left was erased. Then, the resulting
  two-digit number was multiplied by 7, and the original three-digit number was obtained. -/
theorem original_three_digit_number_a (h : ∃ (N : ℕ), N = 100 * x + 10 * y + z ∧ 
  N = 7 * (10 * y + z)) : ∃ (N : ℕ), N = 350 :=
sorry

end ProblemA

section ProblemB

variables {x y z : ℕ}

/-- In a three-digit number, the middle digit was erased, and the resulting number 
  is 6 times smaller than the original. --/
theorem original_three_digit_number_b (h : ∃ (N : ℕ), N = 100 * x + 10 * y + z ∧ 
  6 * (10 * x + z) = N) : ∃ (N : ℕ), N = 108 :=
sorry

end ProblemB

end NUMINAMATH_GPT_original_three_digit_number_a_original_three_digit_number_b_l1168_116899


namespace NUMINAMATH_GPT_customers_served_total_l1168_116869

theorem customers_served_total :
  let Ann_hours := 8
  let Ann_rate := 7
  let Becky_hours := 7
  let Becky_rate := 8
  let Julia_hours := 6
  let Julia_rate := 6
  let lunch_break := 0.5
  let Ann_customers := (Ann_hours - lunch_break) * Ann_rate
  let Becky_customers := (Becky_hours - lunch_break) * Becky_rate
  let Julia_customers := (Julia_hours - lunch_break) * Julia_rate
  Ann_customers + Becky_customers + Julia_customers = 137 := by
  sorry

end NUMINAMATH_GPT_customers_served_total_l1168_116869


namespace NUMINAMATH_GPT_minimum_value_of_sum_of_squares_l1168_116808

variable {x y : ℝ}

theorem minimum_value_of_sum_of_squares (h : x^2 + 2*x*y - y^2 = 7) : 
  x^2 + y^2 ≥ 7 * Real.sqrt 2 / 2 := by 
    sorry

end NUMINAMATH_GPT_minimum_value_of_sum_of_squares_l1168_116808


namespace NUMINAMATH_GPT_negation_universal_proposition_l1168_116867

theorem negation_universal_proposition :
  ¬ (∀ x : ℝ, |x| + x^4 ≥ 0) ↔ ∃ x₀ : ℝ, |x₀| + x₀^4 < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_universal_proposition_l1168_116867


namespace NUMINAMATH_GPT_units_digit_1_to_99_is_5_l1168_116834

noncomputable def units_digit_of_product_of_odds : ℕ :=
  let seq := List.range' 1 99;
  (seq.filter (λ n => n % 2 = 1)).prod % 10

theorem units_digit_1_to_99_is_5 : units_digit_of_product_of_odds = 5 :=
by sorry

end NUMINAMATH_GPT_units_digit_1_to_99_is_5_l1168_116834


namespace NUMINAMATH_GPT_max_planes_determined_l1168_116894

-- Definitions for conditions
variables (Point Line Plane : Type)
variables (l : Line) (A B C : Point)
variables (contains : Point → Line → Prop)
variables (plane_contains_points : Plane → Point → Point → Point → Prop)
variables (plane_contains_line_and_point : Plane → Line → Point → Prop)
variables (non_collinear : Point → Point → Point → Prop)
variables (not_on_line : Point → Line → Prop)

-- Hypotheses based on the conditions
axiom three_non_collinear_points : non_collinear A B C
axiom point_not_on_line (P : Point) : not_on_line P l

-- Goal: Prove that the number of planes is 4
theorem max_planes_determined : 
  ∃ total_planes : ℕ, total_planes = 4 :=
sorry

end NUMINAMATH_GPT_max_planes_determined_l1168_116894


namespace NUMINAMATH_GPT_quadratic_eq_a_val_l1168_116807

theorem quadratic_eq_a_val (a : ℝ) (h : a - 6 = 0) :
  a = 6 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_a_val_l1168_116807


namespace NUMINAMATH_GPT_cookies_left_at_end_of_week_l1168_116892

def trays_baked_each_day : List Nat := [2, 3, 4, 5, 3, 4, 4]
def cookies_per_tray : Nat := 12
def cookies_eaten_by_frank : Nat := 2 * 7
def cookies_eaten_by_ted : Nat := 3 + 5
def cookies_eaten_by_jan : Nat := 5
def cookies_eaten_by_tom : Nat := 8
def cookies_eaten_by_neighbours_kids : Nat := 20

def total_cookies_baked : Nat :=
  (trays_baked_each_day.map (λ trays => trays * cookies_per_tray)).sum

def total_cookies_eaten : Nat :=
  cookies_eaten_by_frank + cookies_eaten_by_ted + cookies_eaten_by_jan +
  cookies_eaten_by_tom + cookies_eaten_by_neighbours_kids

def cookies_left : Nat := total_cookies_baked - total_cookies_eaten

theorem cookies_left_at_end_of_week : cookies_left = 245 :=
by
  sorry

end NUMINAMATH_GPT_cookies_left_at_end_of_week_l1168_116892


namespace NUMINAMATH_GPT_remainder_of_x_divided_by_30_l1168_116871

theorem remainder_of_x_divided_by_30:
  ∀ x : ℤ,
    (4 + x ≡ 9 [ZMOD 8]) ∧ 
    (6 + x ≡ 8 [ZMOD 27]) ∧ 
    (8 + x ≡ 49 [ZMOD 125]) ->
    (x ≡ 17 [ZMOD 30]) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_remainder_of_x_divided_by_30_l1168_116871


namespace NUMINAMATH_GPT_part_a_part_b_l1168_116818

noncomputable def same_start_digit (n x : ℕ) : Prop :=
  ∃ d : ℕ, ∀ k : ℕ, (k ≤ n) → (x * 10^(k-1) ≤ d * 10^(k-1) + 10^(k-1) - 1) ∧ ((d * 10^(k-1)) < x * 10^(k-1))

theorem part_a (x : ℕ) : 
  (same_start_digit 3 x) → ¬(∃ d : ℕ, d = 1) → false :=
  sorry

theorem part_b (x : ℕ) : 
  (same_start_digit 2015 x) → ¬(∃ d : ℕ, d = 1) → false :=
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1168_116818


namespace NUMINAMATH_GPT_find_x_l1168_116859

-- Definitions of the vectors and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vector_a_minus_b (x : ℝ) : ℝ × ℝ := ((1 - x), (4))

-- The dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The given condition of perpendicular vectors
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

-- The theorem to prove
theorem find_x : ∃ x : ℝ, is_perpendicular vector_a (vector_a_minus_b x) ∧ x = 9 :=
by {
  -- Sorry statement used to skip proof
  sorry
}

end NUMINAMATH_GPT_find_x_l1168_116859


namespace NUMINAMATH_GPT_find_vector_c_l1168_116844

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (1, 2)
def c : ℝ × ℝ := (2, 1)

def perp (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, w = (k * v.1, k * v.2)

theorem find_vector_c : 
  perp (c.1 + b.1, c.2 + b.2) a ∧ parallel (c.1 - a.1, c.2 + a.2) b :=
by 
  sorry

end NUMINAMATH_GPT_find_vector_c_l1168_116844


namespace NUMINAMATH_GPT_find_daily_rate_of_first_company_l1168_116812

-- Define the daily rate of the first car rental company
def daily_rate_first_company (x : ℝ) : ℝ :=
  x + 0.18 * 48.0

-- Define the total cost for City Rentals
def total_cost_city_rentals : ℝ :=
  18.95 + 0.16 * 48.0

-- Prove the daily rate of the first car rental company
theorem find_daily_rate_of_first_company (x : ℝ) (h : daily_rate_first_company x = total_cost_city_rentals) : 
  x = 17.99 := 
by
  sorry

end NUMINAMATH_GPT_find_daily_rate_of_first_company_l1168_116812


namespace NUMINAMATH_GPT_discount_store_purchase_l1168_116873

theorem discount_store_purchase (n x y : ℕ) (hn : 2 * n + (x + y) = 2 * n) 
(h1 : 8 * x + 9 * y = 172) (hx : 0 ≤ x) (hy : 0 ≤ y): 
x = 8 ∧ y = 12 :=
sorry

end NUMINAMATH_GPT_discount_store_purchase_l1168_116873


namespace NUMINAMATH_GPT_octahedron_cut_area_l1168_116819

theorem octahedron_cut_area:
  let a := 9
  let b := 3
  let c := 8
  a + b + c = 20 :=
by
  sorry

end NUMINAMATH_GPT_octahedron_cut_area_l1168_116819


namespace NUMINAMATH_GPT_sacksPerSectionDaily_l1168_116887

variable (totalSacks : ℕ) (sections : ℕ) (sacksPerSection : ℕ)

-- Conditions from the problem
variables (h1 : totalSacks = 360) (h2 : sections = 8)

-- The theorem statement
theorem sacksPerSectionDaily : sacksPerSection = 45 :=
by
  have h3 : totalSacks / sections = 45 := by sorry
  have h4 : sacksPerSection = totalSacks / sections := by sorry
  exact Eq.trans h4 h3

end NUMINAMATH_GPT_sacksPerSectionDaily_l1168_116887


namespace NUMINAMATH_GPT_annie_total_distance_traveled_l1168_116845

-- Definitions of conditions
def walk_distance : ℕ := 5
def bus_distance : ℕ := 7
def total_distance_one_way : ℕ := walk_distance + bus_distance
def total_distance_round_trip : ℕ := total_distance_one_way * 2

-- Theorem statement to prove the total number of blocks traveled
theorem annie_total_distance_traveled : total_distance_round_trip = 24 :=
by
  sorry

end NUMINAMATH_GPT_annie_total_distance_traveled_l1168_116845


namespace NUMINAMATH_GPT_minimum_spend_on_boxes_l1168_116816

def box_dimensions : ℕ × ℕ × ℕ := (20, 20, 12)
def cost_per_box : ℝ := 0.40
def total_volume : ℕ := 2400000
def volume_of_box (l w h : ℕ) : ℕ := l * w * h
def number_of_boxes (total_vol vol_per_box : ℕ) : ℕ := total_vol / vol_per_box
def total_cost (num_boxes : ℕ) (cost_box : ℝ) : ℝ := num_boxes * cost_box

theorem minimum_spend_on_boxes : total_cost (number_of_boxes total_volume (volume_of_box 20 20 12)) cost_per_box = 200 := by
  sorry

end NUMINAMATH_GPT_minimum_spend_on_boxes_l1168_116816


namespace NUMINAMATH_GPT_library_books_l1168_116824

theorem library_books (A : Prop) (B : Prop) (C : Prop) (D : Prop) :
  (¬A) → (B ∧ D) :=
by
  -- Assume the statement "All books in this library are available for lending." is represented by A.
  -- A is false.
  intro h_notA
  -- Show that statement II ("There is some book in this library not available for lending.")
  -- and statement IV ("Not all books in this library are available for lending.") are both true.
  -- These are represented as B and D, respectively.
  sorry

end NUMINAMATH_GPT_library_books_l1168_116824


namespace NUMINAMATH_GPT_expected_value_of_geometric_variance_of_geometric_l1168_116884

noncomputable def expected_value (p : ℝ) : ℝ :=
  1 / p

noncomputable def variance (p : ℝ) : ℝ :=
  (1 - p) / (p ^ 2)

theorem expected_value_of_geometric (p : ℝ) (hp : 0 < p ∧ p < 1) :
  ∑' n, (n + 1 : ℝ) * (1 - p) ^ n * p = expected_value p := by
  sorry

theorem variance_of_geometric (p : ℝ) (hp : 0 < p ∧ p < 1) :
  ∑' n, ((n + 1 : ℝ) ^ 2) * (1 - p) ^ n * p - (expected_value p) ^ 2 = variance p := by
  sorry

end NUMINAMATH_GPT_expected_value_of_geometric_variance_of_geometric_l1168_116884


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1168_116800

theorem sufficient_but_not_necessary (a : ℝ) : ((a = 2) → ((a - 1) * (a - 2) = 0)) ∧ (¬(((a - 1) * (a - 2) = 0) → (a = 2))) := 
by 
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1168_116800


namespace NUMINAMATH_GPT_number_of_valid_three_digit_numbers_l1168_116863

def valid_three_digit_numbers : Nat :=
  -- Proving this will be the task: showing that there are precisely 24 such numbers
  24

theorem number_of_valid_three_digit_numbers : valid_three_digit_numbers = 24 :=
by
  -- Proof would go here.
  sorry

end NUMINAMATH_GPT_number_of_valid_three_digit_numbers_l1168_116863


namespace NUMINAMATH_GPT_find_abc_l1168_116854

theorem find_abc :
  ∃ a b c : ℝ, (∀ x : ℝ, (x < -6 ∨ |x - 30| ≤ 2) ↔ ((x - a) * (x - b) / (x - c) ≤ 0)) ∧ a < b ∧ a + 2 * b + 3 * c = 74 :=
by
  sorry

end NUMINAMATH_GPT_find_abc_l1168_116854


namespace NUMINAMATH_GPT_solve_for_x_l1168_116877

theorem solve_for_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1168_116877


namespace NUMINAMATH_GPT_rhombus_shorter_diagonal_l1168_116876

variable (d1 d2 : ℝ) (Area : ℝ)

def is_rhombus (Area : ℝ) (d1 d2 : ℝ) : Prop := Area = (d1 * d2) / 2

theorem rhombus_shorter_diagonal
  (h_d2 : d2 = 20)
  (h_Area : Area = 110)
  (h_rhombus : is_rhombus Area d1 d2) :
  d1 = 11 := by
  sorry

end NUMINAMATH_GPT_rhombus_shorter_diagonal_l1168_116876


namespace NUMINAMATH_GPT_find_a_from_function_l1168_116801

theorem find_a_from_function (f : ℝ → ℝ) (h_f : ∀ x, f x = Real.sqrt (2 * x + 1)) (a : ℝ) (h_a : f a = 5) : a = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_a_from_function_l1168_116801


namespace NUMINAMATH_GPT_calories_consumed_in_week_l1168_116890

-- Define the calorie content of each type of burger
def calorie_A := 350
def calorie_B := 450
def calorie_C := 550

-- Define Dimitri's burger consumption over the 7 days
def consumption_day1 := (2 * calorie_A) + (1 * calorie_B)
def consumption_day2 := (1 * calorie_A) + (2 * calorie_B) + (1 * calorie_C)
def consumption_day3 := (1 * calorie_A) + (1 * calorie_B) + (2 * calorie_C)
def consumption_day4 := (3 * calorie_B)
def consumption_day5 := (1 * calorie_A) + (1 * calorie_B) + (1 * calorie_C)
def consumption_day6 := (2 * calorie_A) + (3 * calorie_C)
def consumption_day7 := (1 * calorie_B) + (2 * calorie_C)

-- Define the total weekly calorie consumption
def total_weekly_calories :=
  consumption_day1 + consumption_day2 + consumption_day3 +
  consumption_day4 + consumption_day5 + consumption_day6 + consumption_day7

-- State and prove the main theorem
theorem calories_consumed_in_week :
  total_weekly_calories = 11450 := 
by
  sorry

end NUMINAMATH_GPT_calories_consumed_in_week_l1168_116890


namespace NUMINAMATH_GPT_a1_greater_than_500_l1168_116855

-- Set up conditions
variables (a : ℕ → ℕ) (h1 : ∀ n, 0 < a n ∧ a n < 20000)
variables (h2 : ∀ i j, i < j → gcd (a i) (a j) < a i)
variables (h3 : ∀ i j, i < j ∧ 1 ≤ i ∧ j ≤ 10000 → a i < a j)

/-- Statement to prove / lean concept as per mathematical problem  --/
theorem a1_greater_than_500 : 500 < a 1 :=
sorry

end NUMINAMATH_GPT_a1_greater_than_500_l1168_116855


namespace NUMINAMATH_GPT_smallest_integer_solution_l1168_116832

theorem smallest_integer_solution : ∃ x : ℤ, (x^2 = 3 * x + 78) ∧ x = -6 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_integer_solution_l1168_116832


namespace NUMINAMATH_GPT_meet_time_correct_l1168_116842

variable (circumference : ℕ) (speed_yeonjeong speed_donghun : ℕ)

def meet_time (circumference speed_yeonjeong speed_donghun : ℕ) : ℕ :=
  circumference / (speed_yeonjeong + speed_donghun)

theorem meet_time_correct
  (h_circumference : circumference = 3000)
  (h_speed_yeonjeong : speed_yeonjeong = 100)
  (h_speed_donghun : speed_donghun = 150) :
  meet_time circumference speed_yeonjeong speed_donghun = 12 :=
by
  rw [h_circumference, h_speed_yeonjeong, h_speed_donghun]
  norm_num
  sorry

end NUMINAMATH_GPT_meet_time_correct_l1168_116842


namespace NUMINAMATH_GPT_symmetric_coordinates_l1168_116815

-- Define the point A as a tuple of its coordinates
def A : Prod ℤ ℤ := (-1, 2)

-- Define what it means for point A' to be symmetric to the origin
def symmetric_to_origin (p : Prod ℤ ℤ) : Prod ℤ ℤ :=
  (-p.1, -p.2)

-- The theorem we need to prove
theorem symmetric_coordinates :
  symmetric_to_origin A = (1, -2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_coordinates_l1168_116815


namespace NUMINAMATH_GPT_indeterminate_C_l1168_116837

variable (m n C : ℝ)

theorem indeterminate_C (h1 : m = 8 * n + C)
                      (h2 : m + 2 = 8 * (n + 0.25) + C) : 
                      False :=
by
  sorry

end NUMINAMATH_GPT_indeterminate_C_l1168_116837


namespace NUMINAMATH_GPT_find_y_l1168_116852

noncomputable def x : Real := 2.6666666666666665

theorem find_y (y : Real) (h : (x * y) / 3 = x^2) : y = 8 :=
sorry

end NUMINAMATH_GPT_find_y_l1168_116852


namespace NUMINAMATH_GPT_rod_division_segments_l1168_116817

theorem rod_division_segments (L : ℕ) (K : ℕ) (hL : L = 72 * K) :
  let red_divisions := 7
  let blue_divisions := 11
  let black_divisions := 17
  let overlap_9_6 := 4
  let overlap_6_4 := 6
  let overlap_9_4 := 2
  let overlap_all := 2
  let total_segments := red_divisions + blue_divisions + black_divisions - overlap_9_6 - overlap_6_4 - overlap_9_4 + overlap_all
  (total_segments = 28) ∧ ((L / 72) = K)
:=
by
  sorry

end NUMINAMATH_GPT_rod_division_segments_l1168_116817


namespace NUMINAMATH_GPT_second_sheet_width_l1168_116888

theorem second_sheet_width :
  ∃ w : ℝ, (286 = 22 * w + 100) ∧ w = 8.5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_second_sheet_width_l1168_116888


namespace NUMINAMATH_GPT_negation_of_existence_statement_l1168_116851

theorem negation_of_existence_statement :
  (¬ ∃ x_0 : ℝ, x_0^2 - x_0 + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_existence_statement_l1168_116851


namespace NUMINAMATH_GPT_inequality_proof_l1168_116896

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  (x / (y + z)) + (y / (z + x)) + (z / (x + y)) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1168_116896


namespace NUMINAMATH_GPT_gcd_8_10_l1168_116827

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_8_10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := 
by
  sorry

end NUMINAMATH_GPT_gcd_8_10_l1168_116827


namespace NUMINAMATH_GPT_trash_picked_outside_l1168_116805

theorem trash_picked_outside (T_tot : ℕ) (C1 C2 C3 C4 C5 C6 C7 C8 : ℕ)
  (hT_tot : T_tot = 1576)
  (hC1 : C1 = 124) (hC2 : C2 = 98) (hC3 : C3 = 176) (hC4 : C4 = 212)
  (hC5 : C5 = 89) (hC6 : C6 = 241) (hC7 : C7 = 121) (hC8 : C8 = 102) :
  T_tot - (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8) = 413 :=
by sorry

end NUMINAMATH_GPT_trash_picked_outside_l1168_116805


namespace NUMINAMATH_GPT_ceil_neg_sqrt_frac_l1168_116847

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := by
  sorry

end NUMINAMATH_GPT_ceil_neg_sqrt_frac_l1168_116847


namespace NUMINAMATH_GPT_evaluate_at_10_l1168_116875

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

theorem evaluate_at_10 : f 10 = 756 := by
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_evaluate_at_10_l1168_116875


namespace NUMINAMATH_GPT_number_of_questions_per_survey_is_10_l1168_116883

variable {Q : ℕ}  -- Q: Number of questions in each survey

def money_per_question : ℝ := 0.2
def surveys_on_monday : ℕ := 3
def surveys_on_tuesday : ℕ := 4
def total_money_earned : ℝ := 14

theorem number_of_questions_per_survey_is_10 :
    (surveys_on_monday + surveys_on_tuesday) * Q * money_per_question = total_money_earned → Q = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_questions_per_survey_is_10_l1168_116883


namespace NUMINAMATH_GPT_minimum_quotient_value_l1168_116862

-- Helper definition to represent the quotient 
def quotient (a b c d : ℕ) : ℚ := (1000 * a + 100 * b + 10 * c + d) / (a + b + c + d)

-- Conditions: digits are distinct and non-zero 
def distinct_and_nonzero (a b c d : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem minimum_quotient_value :
  ∀ (a b c d : ℕ), distinct_and_nonzero a b c d → quotient a b c d = 71.9 :=
by sorry

end NUMINAMATH_GPT_minimum_quotient_value_l1168_116862


namespace NUMINAMATH_GPT_optimal_order_for_ostap_l1168_116878

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end NUMINAMATH_GPT_optimal_order_for_ostap_l1168_116878


namespace NUMINAMATH_GPT_tileable_if_and_only_if_l1168_116833

def is_tileable (n : ℕ) : Prop :=
  ∃ k : ℕ, 15 * n = 4 * k

theorem tileable_if_and_only_if (n : ℕ) :
  (n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7) ↔ is_tileable n :=
sorry

end NUMINAMATH_GPT_tileable_if_and_only_if_l1168_116833


namespace NUMINAMATH_GPT_average_price_of_pen_l1168_116848

theorem average_price_of_pen (c_total : ℝ) (n_pens n_pencils : ℕ) (p_pencil : ℝ)
  (h1 : c_total = 450) (h2 : n_pens = 30) (h3 : n_pencils = 75) (h4 : p_pencil = 2) :
  (c_total - (n_pencils * p_pencil)) / n_pens = 10 :=
by
  sorry

end NUMINAMATH_GPT_average_price_of_pen_l1168_116848


namespace NUMINAMATH_GPT_order_a_c_b_l1168_116835

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log 8 / Real.log 5

theorem order_a_c_b : a > c ∧ c > b := 
by {
  sorry
}

end NUMINAMATH_GPT_order_a_c_b_l1168_116835


namespace NUMINAMATH_GPT_inequality_proof_l1168_116849

variable (a b c d : ℝ)
variable (habcda : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ ab + bc + cd + da = 1)

theorem inequality_proof :
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧ (ab + bc + cd + da = 1) →
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l1168_116849


namespace NUMINAMATH_GPT_task1_on_time_task2_not_on_time_prob_l1168_116872

def task1_on_time_prob : ℚ := 3 / 8
def task2_on_time_prob : ℚ := 3 / 5

theorem task1_on_time_task2_not_on_time_prob :
  task1_on_time_prob * (1 - task2_on_time_prob) = 3 / 20 := by
  sorry

end NUMINAMATH_GPT_task1_on_time_task2_not_on_time_prob_l1168_116872


namespace NUMINAMATH_GPT_power_mod_l1168_116820

theorem power_mod (h1: 5^2 % 17 = 8) (h2: 5^4 % 17 = 13) (h3: 5^8 % 17 = 16) (h4: 5^16 % 17 = 1):
  5^2024 % 17 = 16 :=
by
  sorry

end NUMINAMATH_GPT_power_mod_l1168_116820


namespace NUMINAMATH_GPT_problem1_problem2_l1168_116865

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - 6 * x - 72 ≤ 0) ∧ (x^2 + x - 6 > 0)

-- Problem 1: Proving the range of x
theorem problem1 (x : ℝ) (h₁ : a = -1) (h₂ : ∀ (x : ℝ), p x a → q x) : 
  x ∈ {x : ℝ | -6 ≤ x ∧ x < -3} ∨ x ∈ {x : ℝ | 1 < x ∧ x ≤ 12} := sorry

-- Problem 2: Proving the range of a
theorem problem2 (a : ℝ) (h₃ : (∀ x, q x → p x a) ∧ ¬ (∀ x, ¬q x → ¬p x a)) : 
  -4 ≤ a ∧ a ≤ -2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1168_116865


namespace NUMINAMATH_GPT_minimum_value_expression_l1168_116895

theorem minimum_value_expression 
  (a b c : ℝ) 
  (h1 : 3 * a + 2 * b + c = 5) 
  (h2 : 2 * a + b - 3 * c = 1) 
  (h3 : 0 ≤ a) 
  (h4 : 0 ≤ b) 
  (h5 : 0 ≤ c) : 
  ∃(c : ℝ), (c ≥ 3/7 ∧ c ≤ 7/11) ∧ (3 * a + b - 7 * c = -5/7) :=
sorry 

end NUMINAMATH_GPT_minimum_value_expression_l1168_116895


namespace NUMINAMATH_GPT_total_cost_of_aquarium_l1168_116874

variable (original_price discount_rate sales_tax_rate : ℝ)
variable (original_cost : original_price = 120)
variable (discount : discount_rate = 0.5)
variable (tax : sales_tax_rate = 0.05)

theorem total_cost_of_aquarium : 
  (original_price * (1 - discount_rate) * (1 + sales_tax_rate) = 63) :=
by
  rw [original_cost, discount, tax]
  sorry

end NUMINAMATH_GPT_total_cost_of_aquarium_l1168_116874


namespace NUMINAMATH_GPT_negation_of_proposition_l1168_116858

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x^2 + 1 ≥ 1) ↔ ∃ x : ℝ, x^2 + 1 < 1 :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l1168_116858


namespace NUMINAMATH_GPT_consumer_installment_credit_l1168_116866

theorem consumer_installment_credit (A C : ℝ) 
  (h1 : A = 0.36 * C) 
  (h2 : 57 = 1 / 3 * A) : 
  C = 475 := 
by 
  sorry

end NUMINAMATH_GPT_consumer_installment_credit_l1168_116866


namespace NUMINAMATH_GPT_factor_expression_l1168_116831

theorem factor_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) + 2 * (y + 2) = (y + 2) * (5 * y + 11) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l1168_116831


namespace NUMINAMATH_GPT_total_training_hours_l1168_116838

-- Define Thomas's training conditions
def hours_per_day := 5
def days_initial := 30
def days_additional := 12
def total_days := days_initial + days_additional

-- State the theorem to be proved
theorem total_training_hours : total_days * hours_per_day = 210 :=
by
  sorry

end NUMINAMATH_GPT_total_training_hours_l1168_116838


namespace NUMINAMATH_GPT_B_work_days_l1168_116881

-- Define work rates and conditions
def A_work_rate : ℚ := 1 / 18
def B_work_rate : ℚ := 1 / 15
def A_days_after_B_left : ℚ := 6
def total_work : ℚ := 1

-- Theorem statement
theorem B_work_days : ∃ x : ℚ, (x * B_work_rate + A_days_after_B_left * A_work_rate = total_work) → x = 10 := by
  sorry

end NUMINAMATH_GPT_B_work_days_l1168_116881


namespace NUMINAMATH_GPT_joan_seashells_initially_l1168_116840

variable (mikeGave joanTotal : ℕ)

theorem joan_seashells_initially (h : mikeGave = 63) (t : joanTotal = 142) : joanTotal - mikeGave = 79 := 
by
  sorry

end NUMINAMATH_GPT_joan_seashells_initially_l1168_116840


namespace NUMINAMATH_GPT_total_rainfall_in_2011_l1168_116822

-- Define the given conditions
def avg_monthly_rainfall_2010 : ℝ := 36.8
def increase_2011 : ℝ := 3.5

-- Define the resulting average monthly rainfall in 2011
def avg_monthly_rainfall_2011 : ℝ := avg_monthly_rainfall_2010 + increase_2011

-- Calculate the total annual rainfall
def total_rainfall_2011 : ℝ := avg_monthly_rainfall_2011 * 12

-- State the proof problem
theorem total_rainfall_in_2011 :
  total_rainfall_2011 = 483.6 := by
  sorry

end NUMINAMATH_GPT_total_rainfall_in_2011_l1168_116822


namespace NUMINAMATH_GPT_students_without_glasses_l1168_116802

theorem students_without_glasses (total_students : ℕ) (perc_glasses : ℕ) (students_with_glasses students_without_glasses : ℕ) 
  (h1 : total_students = 325) (h2 : perc_glasses = 40) (h3 : students_with_glasses = perc_glasses * total_students / 100)
  (h4 : students_without_glasses = total_students - students_with_glasses) : students_without_glasses = 195 := 
by
  sorry

end NUMINAMATH_GPT_students_without_glasses_l1168_116802


namespace NUMINAMATH_GPT_find_number_of_students_l1168_116860

open Nat

theorem find_number_of_students :
  ∃ n : ℕ, 35 < n ∧ n < 70 ∧ n % 6 = 3 ∧ n % 8 = 1 ∧ n = 57 :=
by
  use 57
  sorry

end NUMINAMATH_GPT_find_number_of_students_l1168_116860


namespace NUMINAMATH_GPT_solve_equation_l1168_116882

theorem solve_equation (x : ℤ) : x * (x + 2) + 1 = 36 ↔ x = 5 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1168_116882


namespace NUMINAMATH_GPT_range_of_x_satisfying_inequality_l1168_116861

def otimes (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x_satisfying_inequality :
  { x : ℝ | otimes x (x - 2) < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_satisfying_inequality_l1168_116861


namespace NUMINAMATH_GPT_jogger_distance_ahead_l1168_116825

noncomputable def jogger_speed_kmph : ℤ := 9
noncomputable def train_speed_kmph : ℤ := 45
noncomputable def train_length_m : ℤ := 120
noncomputable def time_to_pass_seconds : ℤ := 38

theorem jogger_distance_ahead
  (jogger_speed_kmph : ℤ)
  (train_speed_kmph : ℤ)
  (train_length_m : ℤ)
  (time_to_pass_seconds : ℤ) :
  jogger_speed_kmph = 9 →
  train_speed_kmph = 45 →
  train_length_m = 120 →
  time_to_pass_seconds = 38 →
  ∃ distance_ahead : ℤ, distance_ahead = 260 :=
by 
  -- the proof would go here
  sorry  

end NUMINAMATH_GPT_jogger_distance_ahead_l1168_116825


namespace NUMINAMATH_GPT_sachin_is_younger_than_rahul_by_18_years_l1168_116809

-- Definitions based on conditions
def sachin_age : ℕ := 63
def ratio_of_ages : ℚ := 7 / 9

-- Assertion that based on the given conditions, Sachin is 18 years younger than Rahul
theorem sachin_is_younger_than_rahul_by_18_years (R : ℕ) (h1 : (sachin_age : ℚ) / R = ratio_of_ages) : R - sachin_age = 18 :=
by
  sorry

end NUMINAMATH_GPT_sachin_is_younger_than_rahul_by_18_years_l1168_116809


namespace NUMINAMATH_GPT_minimum_positive_period_of_f_l1168_116880

noncomputable def f (x : ℝ) : ℝ := (1 + (Real.sqrt 3) * Real.tan x) * Real.cos x

theorem minimum_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

end NUMINAMATH_GPT_minimum_positive_period_of_f_l1168_116880


namespace NUMINAMATH_GPT_johns_number_l1168_116813

theorem johns_number (n : ℕ) :
  64 ∣ n ∧ 45 ∣ n ∧ 1000 < n ∧ n < 3000 -> n = 2880 :=
by
  sorry

end NUMINAMATH_GPT_johns_number_l1168_116813


namespace NUMINAMATH_GPT_number_of_small_companies_l1168_116810

theorem number_of_small_companies
  (large_companies : ℕ)
  (medium_companies : ℕ)
  (inspected_companies : ℕ)
  (inspected_medium_companies : ℕ)
  (total_inspected_companies : ℕ)
  (small_companies : ℕ)
  (inspection_fraction : ℕ → ℚ)
  (proportion : inspection_fraction 20 = 1 / 4)
  (H1 : large_companies = 4)
  (H2 : medium_companies = 20)
  (H3 : inspected_medium_companies = 5)
  (H4 : total_inspected_companies = 40)
  (H5 : inspected_companies = total_inspected_companies - large_companies - inspected_medium_companies)
  (H6 : small_companies = inspected_companies * 4)
  (correct_result : small_companies = 136) :
  small_companies = 136 :=
by sorry

end NUMINAMATH_GPT_number_of_small_companies_l1168_116810


namespace NUMINAMATH_GPT_find_f_of_three_l1168_116846

variable {f : ℝ → ℝ}

theorem find_f_of_three (h : ∀ x : ℝ, f (1 - 2 * x) = x^2 + x) : f 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_three_l1168_116846


namespace NUMINAMATH_GPT_find_D_l1168_116836

noncomputable def point := (ℝ × ℝ)

def vector_add (u v : point) : point := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : point) : point := (u.1 - v.1, u.2 - v.2)
def scalar_multiplication (k : ℝ) (u : point) : point := (k * u.1, k * u.2)

namespace GeometryProblem

def A : point := (2, 3)
def B : point := (-1, 5)

def D : point := 
  let AB := vector_sub B A
  vector_add A (scalar_multiplication 3 AB)

theorem find_D : D = (-7, 9) := by
  sorry

end GeometryProblem

end NUMINAMATH_GPT_find_D_l1168_116836


namespace NUMINAMATH_GPT_find_mass_of_water_vapor_l1168_116826

noncomputable def heat_balance_problem : Prop :=
  ∃ (m_s : ℝ), m_s * 536 + m_s * 80 = 
  (50 * 80 + 50 * 20 + 300 * 20 + 100 * 0.5 * 20)
  ∧ m_s = 19.48

theorem find_mass_of_water_vapor : heat_balance_problem := by
  sorry

end NUMINAMATH_GPT_find_mass_of_water_vapor_l1168_116826


namespace NUMINAMATH_GPT_money_given_to_last_set_l1168_116856

theorem money_given_to_last_set (total first second third fourth last : ℝ) 
  (h_total : total = 4500) 
  (h_first : first = 725) 
  (h_second : second = 1100) 
  (h_third : third = 950) 
  (h_fourth : fourth = 815) 
  (h_sum: total = first + second + third + fourth + last) : 
  last = 910 :=
sorry

end NUMINAMATH_GPT_money_given_to_last_set_l1168_116856


namespace NUMINAMATH_GPT_garden_ratio_l1168_116857

theorem garden_ratio (L W : ℕ) (h1 : L = 50) (h2 : 2 * L + 2 * W = 150) : L / W = 2 :=
by
  sorry

end NUMINAMATH_GPT_garden_ratio_l1168_116857


namespace NUMINAMATH_GPT_max_area_rectangle_min_area_rectangle_l1168_116814

theorem max_area_rectangle (n : ℕ) (x y : ℕ → ℕ)
  (S : ℕ → ℕ) (H1 : ∀ k, S k = 2^(2*k)) 
  (H2 : ∀ k, (1 ≤ k ∧ k ≤ n) → x k * y k = S k) 
  : (n - 1 + 2^(2*n)) * (4 * 2^(2*(n-1)) - 1/3) = 1/3 * (4^n - 1) * (4^n + n - 1) := sorry

theorem min_area_rectangle (n : ℕ) (x y : ℕ → ℕ)
  (S : ℕ → ℕ) (H1 : ∀ k, S k = 2^(2*k)) 
  (H2 : ∀ k, (1 ≤ k ∧ k ≤ n) → x k * y k = S k)
  : (2^n - 1)^2 = 4 * (2^n - 1)^2 := sorry

end NUMINAMATH_GPT_max_area_rectangle_min_area_rectangle_l1168_116814
