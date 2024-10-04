import Mathlib

namespace find_a_l56_56811

theorem find_a (a b : ℤ) (h : ∀ x, x^2 - x - 1 = 0 → ax^18 + bx^17 + 1 = 0) : a = 1597 :=
sorry

end find_a_l56_56811


namespace parallel_vectors_xy_sum_l56_56047

theorem parallel_vectors_xy_sum (x y : ℚ) (k : ℚ) 
  (h1 : (2, 4, -5) = (2 * k, 4 * k, -5 * k)) 
  (h2 : (3, x, y) = (2 * k, 4 * k, -5 * k)) 
  (h3 : 3 = 2 * k) : 
  x + y = -3 / 2 :=
by
  sorry

end parallel_vectors_xy_sum_l56_56047


namespace brenda_skittles_l56_56641

theorem brenda_skittles (initial additional : ℕ) (h1 : initial = 7) (h2 : additional = 8) :
  initial + additional = 15 :=
by {
  -- Proof would go here
  sorry
}

end brenda_skittles_l56_56641


namespace wall_length_l56_56917

theorem wall_length (side_mirror : ℝ) (width_wall : ℝ) (length_wall : ℝ) 
  (h_mirror: side_mirror = 18) 
  (h_width: width_wall = 32)
  (h_area: (side_mirror ^ 2) * 2 = width_wall * length_wall):
  length_wall = 20.25 := 
by 
  -- The following 'sorry' is a placeholder for the proof
  sorry

end wall_length_l56_56917


namespace positive_difference_of_sums_l56_56127

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l56_56127


namespace division_result_l56_56800

def expr := 180 / (12 + 13 * 2)

theorem division_result : expr = 90 / 19 := by
  sorry

end division_result_l56_56800


namespace original_class_size_l56_56921

theorem original_class_size
  (N : ℕ)
  (h1 : 40 * N = T)
  (h2 : T + 15 * 32 = 36 * (N + 15)) :
  N = 15 := by
  sorry

end original_class_size_l56_56921


namespace minimum_value_of_f_l56_56894

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2) + 2 * x

theorem minimum_value_of_f (h : ∀ x > 0, f x ≥ 3) : ∃ x, x > 0 ∧ f x = 3 :=
by
  sorry

end minimum_value_of_f_l56_56894


namespace fraction_girls_on_trip_l56_56940

theorem fraction_girls_on_trip (b g : ℕ) (hb : g = 2 * b) 
  (f_g_on_trip : ℚ := 5/6) (f_b_on_trip : ℚ := 1/2) :
  (f_g_on_trip * g) / ((f_g_on_trip * g) + (f_b_on_trip * b)) = 10/13 :=
by
  sorry

end fraction_girls_on_trip_l56_56940


namespace area_of_shaded_region_l56_56695

theorem area_of_shaded_region :
  let width := 10
  let height := 5
  let base_triangle := 3
  let height_triangle := 2
  let top_base_trapezoid := 3
  let bottom_base_trapezoid := 6
  let height_trapezoid := 3
  let area_rectangle := width * height
  let area_triangle := (1 / 2 : ℝ) * base_triangle * height_triangle
  let area_trapezoid := (1 / 2 : ℝ) * (top_base_trapezoid + bottom_base_trapezoid) * height_trapezoid
  let area_shaded := area_rectangle - area_triangle - area_trapezoid
  area_shaded = 33.5 :=
by
  sorry

end area_of_shaded_region_l56_56695


namespace vector_expression_simplification_l56_56024

variable (a b : Type)
variable (α : Type) [Field α]
variable [AddCommGroup a] [Module α a]

theorem vector_expression_simplification
  (vector_a vector_b : a) :
  (1/3 : α) • (vector_a - (2 : α) • vector_b) + vector_b = (1/3 : α) • vector_a + (1/3 : α) • vector_b :=
by
  sorry

end vector_expression_simplification_l56_56024


namespace quadrilateral_area_is_two_l56_56860

def A : (Int × Int) := (0, 0)
def B : (Int × Int) := (2, 0)
def C : (Int × Int) := (2, 3)
def D : (Int × Int) := (0, 2)

noncomputable def area (p1 p2 p3 p4 : (Int × Int)) : ℚ :=
  (1 / 2 : ℚ) * (abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2) - 
                      (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p4.1 + p4.2 * p1.1)))

theorem quadrilateral_area_is_two : 
  area A B C D = 2 := by
  sorry

end quadrilateral_area_is_two_l56_56860


namespace positive_difference_sums_l56_56174

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l56_56174


namespace Qing_Dynasty_Problem_l56_56279

variable {x y : ℕ}

theorem Qing_Dynasty_Problem (h1 : 4 * x + 6 * y = 48) (h2 : 2 * x + 5 * y = 38) :
  (4 * x + 6 * y = 48) ∧ (2 * x + 5 * y = 38) := by
  exact ⟨h1, h2⟩

end Qing_Dynasty_Problem_l56_56279


namespace exist_column_remove_keeps_rows_distinct_l56_56507

theorem exist_column_remove_keeps_rows_distinct 
    (n : ℕ) 
    (table : Fin n → Fin n → Char) 
    (h_diff_rows : ∀ i j : Fin n, i ≠ j → ∃ k : Fin n, table i k ≠ table j k) 
    : ∃ col_to_remove : Fin n, ∀ i j : Fin n, i ≠ j → (table i ≠ table j) :=
sorry

end exist_column_remove_keeps_rows_distinct_l56_56507


namespace quadratic_passing_point_l56_56099

theorem quadratic_passing_point :
  ∃ (m : ℝ), (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = 8 → x = 0) →
  (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = -10 → x = -1) →
  (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = m → x = 5) →
  m = 638 := by
  sorry

end quadratic_passing_point_l56_56099


namespace sets_are_equal_l56_56287

def setA : Set ℤ := { n | ∃ x y : ℤ, n = x^2 + 2 * y^2 }
def setB : Set ℤ := { n | ∃ x y : ℤ, n = x^2 - 6 * x * y + 11 * y^2 }

theorem sets_are_equal : setA = setB := 
by
  sorry

end sets_are_equal_l56_56287


namespace sets_are_equal_l56_56286

def setA : Set ℤ := { n | ∃ x y : ℤ, n = x^2 + 2 * y^2 }
def setB : Set ℤ := { n | ∃ x y : ℤ, n = x^2 - 6 * x * y + 11 * y^2 }

theorem sets_are_equal : setA = setB := 
by
  sorry

end sets_are_equal_l56_56286


namespace derivative_at_0_eq_6_l56_56370

-- Definition of the function
def f (x : ℝ) : ℝ := (2 * x + 1)^3

-- Theorem statement indicating the derivative at x = 0 is 6
theorem derivative_at_0_eq_6 : (deriv f 0) = 6 := 
by 
  sorry -- The proof is omitted as per the instructions

end derivative_at_0_eq_6_l56_56370


namespace quadratic_roots_l56_56585

theorem quadratic_roots (x : ℝ) : (x^2 - 8 * x - 2 = 0) ↔ (x = 4 + 3 * Real.sqrt 2) ∨ (x = 4 - 3 * Real.sqrt 2) := by
  sorry

end quadratic_roots_l56_56585


namespace inequality_of_power_sums_l56_56819

variable (a b c : ℝ)

theorem inequality_of_power_sums (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a < b + c) (h5 : b < c + a) (h6 : c < a + b) :
  a^4 + b^4 + c^4 < 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) := sorry

end inequality_of_power_sums_l56_56819


namespace arithmetic_sequence_example_l56_56450

theorem arithmetic_sequence_example 
    (a : ℕ → ℤ) 
    (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) 
    (h2 : a 1 + a 4 + a 7 = 45) 
    (h3 : a 2 + a 5 + a 8 = 39) :
    a 3 + a 6 + a 9 = 33 :=
sorry

end arithmetic_sequence_example_l56_56450


namespace constant_sums_l56_56958

theorem constant_sums (n : ℕ) 
  (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : x * y * z = 1) 
  : (x^n + y^n + z^n = 0 ∨ x^n + y^n + z^n = 3) ↔ (n = 1 ∨ n = 3) :=
by sorry

end constant_sums_l56_56958


namespace lee_can_make_36_cookies_l56_56699

-- Conditions
def initial_cups_of_flour : ℕ := 2
def initial_cookies_made : ℕ := 18
def initial_total_flour : ℕ := 5
def spilled_flour : ℕ := 1

-- Define the remaining cups of flour after spilling
def remaining_flour := initial_total_flour - spilled_flour

-- Define the proportion to solve for the number of cookies made with remaining_flour
def cookies_with_remaining_flour (c : ℕ) : Prop :=
  (initial_cookies_made / initial_cups_of_flour) = (c / remaining_flour)

-- The statement to prove
theorem lee_can_make_36_cookies : cookies_with_remaining_flour 36 :=
  sorry

end lee_can_make_36_cookies_l56_56699


namespace initial_speed_of_car_l56_56632

-- Definition of conditions
def distance_from_A_to_B := 100  -- km
def time_remaining_first_reduction := 30 / 60  -- hours
def speed_reduction_first := 10  -- km/h
def time_remaining_second_reduction := 20 / 60  -- hours
def speed_reduction_second := 10  -- km/h
def additional_time_reduced_speeds := 5 / 60  -- hours

-- Variables for initial speed and intermediate distances
variables (v x : ℝ)

-- Proposition to prove the initial speed
theorem initial_speed_of_car :
  (100 - (v / 2 + x + 20)) / v + 
  (v / 2) / (v - 10) + 
  20 / (v - 20) - 
  20 / (v - 10) 
  = 5 / 60 →
  v = 100 :=
by
  sorry

end initial_speed_of_car_l56_56632


namespace range_of_b_l56_56395

open Real

theorem range_of_b {b x x1 x2 : ℝ} 
  (h1 : ∀ x : ℝ, x^2 - b * x + 1 > 0 ↔ x < x1 ∨ x > x2)
  (h2 : x1 < 1)
  (h3 : x2 > 1) : 
  b > 2 := sorry

end range_of_b_l56_56395


namespace hexagon_area_l56_56592

theorem hexagon_area (ABCDEF : Type) (l : ℕ) (h : l = 3) (p q : ℕ)
  (area_hexagon : ℝ) (area_formula : area_hexagon = Real.sqrt p + Real.sqrt q) :
  p + q = 54 := by
  sorry

end hexagon_area_l56_56592


namespace find_x_l56_56962

def is_mean_twice_mode (l : List ℕ) (mean eq_mode : ℕ) : Prop :=
  l.sum / l.length = eq_mode * 2

theorem find_x (x : ℕ) (h1 : x > 0) (h2 : x ≤ 100)
  (h3 : is_mean_twice_mode [20, x, x, x, x] x (x * 2)) : x = 10 :=
sorry

end find_x_l56_56962


namespace melting_point_of_ice_in_Celsius_l56_56907

theorem melting_point_of_ice_in_Celsius :
  ∀ (boiling_point_F boiling_point_C melting_point_F temperature_C temperature_F : ℤ),
    (boiling_point_F = 212) →
    (boiling_point_C = 100) →
    (melting_point_F = 32) →
    (temperature_C = 60) →
    (temperature_F = 140) →
    (5 * melting_point_F = 9 * 0 + 160) →         -- Using the given equation F = (9/5)C + 32 and C = 0
    melting_point_F = 32 ∧ 0 = 0 :=
by
  intros
  sorry

end melting_point_of_ice_in_Celsius_l56_56907


namespace positive_difference_of_sums_l56_56123

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l56_56123


namespace no_infinite_sequence_of_positive_integers_l56_56343

theorem no_infinite_sequence_of_positive_integers (a : ℕ → ℕ) (H : ∀ n, a n > 0) :
  ¬(∀ n, (a (n+1))^2 ≥ 2 * (a n) * (a (n+2))) :=
sorry

end no_infinite_sequence_of_positive_integers_l56_56343


namespace cricketer_new_average_l56_56470

variable (A : ℕ) (runs_19th_inning : ℕ) (avg_increase : ℕ)
variable (total_runs_after_18 : ℕ)

theorem cricketer_new_average
  (h1 : runs_19th_inning = 98)
  (h2 : avg_increase = 4)
  (h3 : total_runs_after_18 = 18 * A)
  (h4 : 18 * A + 98 = 19 * (A + 4)) :
  A + 4 = 26 :=
by sorry

end cricketer_new_average_l56_56470


namespace machines_needed_l56_56339

variables (R x m N : ℕ) (h1 : 4 * R * 6 = x)
           (h2 : N * R * 6 = m * x)

theorem machines_needed : N = m * 4 :=
by sorry

end machines_needed_l56_56339


namespace number_of_buckets_after_reduction_l56_56899

def initial_buckets : ℕ := 25
def reduction_factor : ℚ := 2 / 5

theorem number_of_buckets_after_reduction :
  (initial_buckets : ℚ) * (1 / reduction_factor) = 63 := by
  sorry

end number_of_buckets_after_reduction_l56_56899


namespace candidate_fails_by_50_marks_l56_56479

theorem candidate_fails_by_50_marks (T : ℝ) (pass_mark : ℝ) (h1 : pass_mark = 199.99999999999997)
    (h2 : 0.45 * T - 25 = 199.99999999999997) :
    199.99999999999997 - 0.30 * T = 50 :=
by
  sorry

end candidate_fails_by_50_marks_l56_56479


namespace difference_in_speeds_is_ten_l56_56564

-- Definitions of given conditions
def distance : ℝ := 200
def time_heavy_traffic : ℝ := 5
def time_no_traffic : ℝ := 4
def speed_heavy_traffic : ℝ := distance / time_heavy_traffic
def speed_no_traffic : ℝ := distance / time_no_traffic
def difference_in_speed : ℝ := speed_no_traffic - speed_heavy_traffic

-- The theorem to prove the questioned statement
theorem difference_in_speeds_is_ten : difference_in_speed = 10 := by
  -- Prove the theorem here
  sorry

end difference_in_speeds_is_ten_l56_56564


namespace positive_difference_sums_even_odd_l56_56141

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l56_56141


namespace multiple_of_tickletoe_nails_l56_56326

def violet_nails := 27
def total_nails := 39
def difference := 3

theorem multiple_of_tickletoe_nails : ∃ (M T : ℕ), violet_nails = M * T + difference ∧ total_nails = violet_nails + T ∧ (M = 2) :=
by
  sorry

end multiple_of_tickletoe_nails_l56_56326


namespace infinite_solutions_a_l56_56667

theorem infinite_solutions_a (a : ℝ) :
  (∀ x : ℝ, 3 * (2 * x - a) = 2 * (3 * x + 12)) ↔ a = -8 :=
by
  sorry

end infinite_solutions_a_l56_56667


namespace perfect_square_eq_m_val_l56_56835

theorem perfect_square_eq_m_val (m : ℝ) (h : ∃ a : ℝ, x^2 - m * x + 49 = (x - a)^2) : m = 14 ∨ m = -14 :=
by
  sorry

end perfect_square_eq_m_val_l56_56835


namespace part1_tangent_line_at_x1_part2_a_range_l56_56248

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * x

theorem part1_tangent_line_at_x1 (a : ℝ) (h1 : a = 1) : 
  let f' (x : ℝ) : ℝ := (x + 1) * Real.exp x - 1
  (2 * Real.exp 1 - 1) * 1 - (f 1 1) = Real.exp 1 :=
by 
  sorry

theorem part2_a_range (a : ℝ) (h2 : ∀ x > 0, f x a ≥ Real.log x - x + 1) : 
  0 < a ∧ a ≤ 2 :=
by 
  sorry

end part1_tangent_line_at_x1_part2_a_range_l56_56248


namespace consecutive_integer_sum_l56_56596

theorem consecutive_integer_sum (n : ℕ) (h1 : n * (n + 1) = 2720) : n + (n + 1) = 103 :=
sorry

end consecutive_integer_sum_l56_56596


namespace real_and_equal_roots_l56_56375

theorem real_and_equal_roots (k : ℝ) : 
  (∃ x : ℝ, (3 * x^2 - k * x + 2 * x + 10) = 0 ∧ 
  (3 * x^2 - k * x + 2 * x + 10) = 0) → 
  (k = 2 - 2 * Real.sqrt 30 ∨ k = -2 - 2 * Real.sqrt 30) := 
by
  sorry

end real_and_equal_roots_l56_56375


namespace pqrs_product_l56_56089

noncomputable def product_of_area_and_perimeter :=
  let P := (1, 3)
  let Q := (4, 4)
  let R := (3, 1)
  let S := (0, 0)
  let side_length := Real.sqrt ((1 - 0)^2 * 4 + (3 - 0)^2 * 4)
  let area := side_length ^ 2
  let perimeter := 4 * side_length
  area * perimeter

theorem pqrs_product : product_of_area_and_perimeter = 208 * Real.sqrt 52 := 
  by 
    sorry

end pqrs_product_l56_56089


namespace sin_double_angle_l56_56247

theorem sin_double_angle (θ : ℝ) (h₁ : 3 * (Real.cos θ)^2 = Real.tan θ + 3) (h₂ : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.sin (2 * (Real.pi - θ)) = 2/3 := 
sorry

end sin_double_angle_l56_56247


namespace marble_ratio_l56_56220

theorem marble_ratio (A V X : ℕ) 
  (h1 : A + 5 = V - 5)
  (h2 : V + X = (A - X) + 30) : X / 5 = 2 :=
by
  sorry

end marble_ratio_l56_56220


namespace positive_difference_of_sums_l56_56137

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l56_56137


namespace market_value_of_stock_l56_56469

def face_value : ℝ := 100
def dividend_per_share : ℝ := 0.10 * face_value
def yield : ℝ := 0.08

theorem market_value_of_stock : (dividend_per_share / yield) = 125 := by
  -- Proof not required
  sorry

end market_value_of_stock_l56_56469


namespace golden_apples_per_pint_l56_56263

-- Data definitions based on given conditions and question
def farmhands : ℕ := 6
def apples_per_hour : ℕ := 240
def hours : ℕ := 5
def ratio_golden_to_pink : ℕ × ℕ := (1, 2)
def pints_of_cider : ℕ := 120
def pink_lady_per_pint : ℕ := 40

-- Total apples picked by farmhands in 5 hours
def total_apples_picked : ℕ := farmhands * apples_per_hour * hours

-- Total pink lady apples picked
def total_pink_lady_apples : ℕ := (total_apples_picked * ratio_golden_to_pink.2) / (ratio_golden_to_pink.1 + ratio_golden_to_pink.2)

-- Total golden delicious apples picked
def total_golden_delicious_apples : ℕ := (total_apples_picked * ratio_golden_to_pink.1) / (ratio_golden_to_pink.1 + ratio_golden_to_pink.2)

-- Total pink lady apples used for 120 pints of cider
def pink_lady_apples_used : ℕ := pints_of_cider * pink_lady_per_pint

-- Number of golden delicious apples used per pint of cider
def golden_delicious_apples_per_pint : ℕ := total_golden_delicious_apples / pints_of_cider

-- Main theorem to prove
theorem golden_apples_per_pint : golden_delicious_apples_per_pint = 20 := by
  -- Start proof (proof body is omitted)
  sorry

end golden_apples_per_pint_l56_56263


namespace parry_secretary_or_treasurer_probability_l56_56777

theorem parry_secretary_or_treasurer_probability (members : Finset ℕ) 
  (h_card : members.card = 10) :
  let parry := 0 in   -- assume Parry's identifier is 0
  ∃ (P : members → ℚ), 
    (P secretary + P treasurer = 19 / 90) := 
by
  let parry := 0
  have h_president := card_pred_SymDiff.card_eq_coe members
  sorry -- proof goes here

end parry_secretary_or_treasurer_probability_l56_56777


namespace Megan_not_lead_actress_l56_56422

-- Define the conditions: total number of plays and lead actress percentage
def totalPlays : ℕ := 100
def leadActressPercentage : ℕ := 80

-- Define what we need to prove: the number of times Megan was not the lead actress
theorem Megan_not_lead_actress (totalPlays: ℕ) (leadActressPercentage: ℕ) : 
  (totalPlays * (100 - leadActressPercentage)) / 100 = 20 :=
by
  -- proof omitted
  sorry

end Megan_not_lead_actress_l56_56422


namespace cos_sin_fraction_l56_56820

theorem cos_sin_fraction (α β : ℝ) (h1 : Real.tan (α + β) = 2 / 5) 
                         (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := 
  sorry

end cos_sin_fraction_l56_56820


namespace length_of_second_race_l56_56692

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

end length_of_second_race_l56_56692


namespace train_speed_l56_56932

theorem train_speed
  (train_length : ℝ)
  (cross_time : ℝ)
  (man_speed_kmh : ℝ)
  (train_speed_kmh : ℝ) :
  (train_length = 150) →
  (cross_time = 6) →
  (man_speed_kmh = 5) →
  (man_speed_kmh * 1000 / 3600 + (train_speed_kmh * 1000 / 3600)) * cross_time = train_length →
  train_speed_kmh = 85 :=
by
  intros htl hct hmk hs
  sorry

end train_speed_l56_56932


namespace triangle_perimeter_l56_56051

def ellipse (x y : ℝ) := x^2 / 4 + y^2 / 2 = 1

def foci_distance (c : ℝ) := c = Real.sqrt 2

theorem triangle_perimeter {x y : ℝ} (A : ellipse x y) (F1 F2 : ℝ)
  (hF1 : F1 = -Real.sqrt 2) (hF2 : F2 = Real.sqrt 2) :
  |(x - F1)| + |(x - F2)| = 4 + 2 * Real.sqrt 2 :=
sorry

end triangle_perimeter_l56_56051


namespace matilda_father_chocolates_l56_56713

theorem matilda_father_chocolates 
  (total_chocolates : ℕ) 
  (total_people : ℕ) 
  (give_up_fraction : ℚ) 
  (mother_chocolates : ℕ) 
  (father_eats : ℕ) 
  (father_left : ℕ) :
  total_chocolates = 20 →
  total_people = 5 →
  give_up_fraction = 1 / 2 →
  mother_chocolates = 3 →
  father_eats = 2 →
  father_left = 5 →
  let chocolates_per_person := total_chocolates / total_people,
      father_chocolates := (chocolates_per_person * total_people * give_up_fraction).nat_abs - mother_chocolates - father_eats
  in father_chocolates = father_left := by
  intros h1 h2 h3 h4 h5 h6
  have h_chocolates_per_person : total_chocolates / total_people = 4 := by sorry
  have h_chocolates_given_up : (chocolates_per_person * total_people * give_up_fraction).nat_abs = 10 := by sorry
  have h_father_chocolates : 10 - mother_chocolates - father_eats = 5 := by sorry
  exact h_father_chocolates

end matilda_father_chocolates_l56_56713


namespace find_triangle_sides_l56_56896

variable (a b c : ℕ)
variable (P : ℕ)
variable (R : ℚ := 65 / 8)
variable (r : ℕ := 4)

theorem find_triangle_sides (h1 : R = 65 / 8) (h2 : r = 4) (h3 : P = a + b + c) : 
  a = 13 ∧ b = 14 ∧ c = 15 :=
  sorry

end find_triangle_sides_l56_56896


namespace first_range_is_30_l56_56206

theorem first_range_is_30 
  (R2 R3 : ℕ)
  (h1 : R2 = 26)
  (h2 : R3 = 32)
  (h3 : min 26 (min 30 32) = 30) : 
  ∃ R1 : ℕ, R1 = 30 :=
  sorry

end first_range_is_30_l56_56206


namespace relationship_among_abc_l56_56668

-- Define a, b, c
def a : ℕ := 22 ^ 55
def b : ℕ := 33 ^ 44
def c : ℕ := 55 ^ 33

-- State the theorem regarding the relationship among a, b, and c
theorem relationship_among_abc : a > b ∧ b > c := 
by
  -- Placeholder for the proof, not required for this task
  sorry

end relationship_among_abc_l56_56668


namespace exists_xy_binom_eq_l56_56285

theorem exists_xy_binom_eq (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (x + y).choose 2 = a * x + b * y :=
by
  sorry

end exists_xy_binom_eq_l56_56285


namespace largest_quotient_is_25_l56_56329

def largest_quotient_set : Set ℤ := {-25, -4, -1, 1, 3, 9}

theorem largest_quotient_is_25 :
  ∃ (a b : ℤ), a ∈ largest_quotient_set ∧ b ∈ largest_quotient_set ∧ b ≠ 0 ∧ (a : ℚ) / b = 25 := by
  sorry

end largest_quotient_is_25_l56_56329


namespace coin_flip_probability_difference_l56_56742

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l56_56742


namespace persons_in_first_group_l56_56721

-- Define the given conditions
def first_group_work_done (P : ℕ) : ℕ := P * 12 * 10
def second_group_work_done : ℕ := 30 * 26 * 6

-- Define the proof problem statement
theorem persons_in_first_group (P : ℕ) (h : first_group_work_done P = second_group_work_done) : P = 39 :=
by
  unfold first_group_work_done second_group_work_done at h
  sorry

end persons_in_first_group_l56_56721


namespace find_m_l56_56532

noncomputable def inverse_proportion (x : ℝ) : ℝ := 4 / x

theorem find_m (m n : ℝ) (h1 : ∀ x, -4 ≤ x ∧ x ≤ m → inverse_proportion x = 4 / x ∧ n ≤ inverse_proportion x ∧ inverse_proportion x ≤ n + 3) :
  m = -1 :=
by
  sorry

end find_m_l56_56532


namespace factorize_expr_l56_56657

theorem factorize_expr (a b : ℝ) : a * b^2 - 8 * a * b + 16 * a = a * (b - 4)^2 := 
by
  sorry

end factorize_expr_l56_56657


namespace positive_difference_l56_56152

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l56_56152


namespace fraction_of_white_surface_area_l56_56780

/-- A cube has edges of 4 inches and is constructed using 64 smaller cubes, each with edges of 1 inch.
Out of these smaller cubes, 56 are white and 8 are black. The 8 black cubes fully cover one face of the larger cube.
Prove that the fraction of the surface area of the larger cube that is white is 5/6. -/
theorem fraction_of_white_surface_area 
  (total_cubes : ℕ := 64)
  (white_cubes : ℕ := 56)
  (black_cubes : ℕ := 8)
  (total_surface_area : ℕ := 96)
  (black_face_area : ℕ := 16)
  (white_surface_area : ℕ := 80) :
  white_surface_area / total_surface_area = 5 / 6 :=
sorry

end fraction_of_white_surface_area_l56_56780


namespace positive_diff_even_odd_sums_l56_56133

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l56_56133


namespace num_prime_divisors_of_50_factorial_l56_56540

-- We need to define the relevant concepts and the condition.

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Set of primes less than or equal to 50
def primes_leq_50 : list ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

-- Main theorem to prove
theorem num_prime_divisors_of_50_factorial : primes_leq_50.length = 15 :=
by
  sorry

end num_prime_divisors_of_50_factorial_l56_56540


namespace scientist_prob_rain_l56_56237

theorem scientist_prob_rain (x : ℝ) (p0 p1 : ℝ)
  (h0 : p0 + p1 = 1)
  (h1 : ∀ x : ℝ, x = (p0 * x^2 + p0 * (1 - x) * x + p1 * (1 - x) * x) / x + (1 - x) - x^2 / (x + 1))
  (h2 : (x + p0 / (x + 1) - x^2 / (x + 1)) = 0.2) :
  x = 1/9 := 
sorry

end scientist_prob_rain_l56_56237


namespace positive_difference_eq_250_l56_56121

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l56_56121


namespace tangent_and_normal_are_correct_at_point_l56_56005

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - 2*x*y + 3*y^2 - 2*y - 16 = 0

def tangent_line (x y : ℝ) : Prop :=
  2*x - 7*y + 19 = 0

def normal_line (x y : ℝ) : Prop :=
  7*x + 2*y - 13 = 0

theorem tangent_and_normal_are_correct_at_point
  (hx : point_on_curve 1 3) :
  tangent_line 1 3 ∧ normal_line 1 3 :=
by
  sorry

end tangent_and_normal_are_correct_at_point_l56_56005


namespace value_of_ratios_l56_56569

variable (x y z : ℝ)

-- Conditions
def geometric_sequence : Prop :=
  4 * y / (3 * x) = 5 * z / (4 * y)

def arithmetic_sequence : Prop :=
  2 / y = 1 / x + 1 / z

-- Theorem/Proof Statement
theorem value_of_ratios (h1 : geometric_sequence x y z) (h2 : arithmetic_sequence x y z) :
  (x / z) + (z / x) = 34 / 15 :=
by
  sorry

end value_of_ratios_l56_56569


namespace express_repeating_decimal_as_fraction_l56_56034

noncomputable def repeating_decimal : ℚ := 7 + 123 / 999
#r "3033"
theorem express_repeating_decimal_as_fraction :
  repeating_decimal = 593 / 111 :=
sorry

end express_repeating_decimal_as_fraction_l56_56034


namespace coplanar_points_l56_56231

theorem coplanar_points (a : ℝ) :
  ∀ (V : ℝ), V = 2 + a^3 → V = 0 → a = -((2:ℝ)^(1/3)) :=
by
  sorry

end coplanar_points_l56_56231


namespace solve_eq1_solve_eq2_l56_56300

-- Prove the solution of the first equation
theorem solve_eq1 (x : ℝ) : 3 * x - (x - 1) = 7 ↔ x = 3 :=
by
  sorry

-- Prove the solution of the second equation
theorem solve_eq2 (x : ℝ) : (2 * x - 1) / 3 - (x - 3) / 6 = 1 ↔ x = (5 : ℝ) / 3 :=
by
  sorry

end solve_eq1_solve_eq2_l56_56300


namespace shaniqua_haircuts_l56_56719

theorem shaniqua_haircuts
  (H : ℕ) -- number of haircuts
  (haircut_income : ℕ) (style_income : ℕ)
  (total_styles : ℕ) (total_income : ℕ)
  (haircut_income_eq : haircut_income = 12)
  (style_income_eq : style_income = 25)
  (total_styles_eq : total_styles = 5)
  (total_income_eq : total_income = 221)
  (income_from_styles : ℕ := total_styles * style_income)
  (income_from_haircuts : ℕ := total_income - income_from_styles) :
  H = income_from_haircuts / haircut_income :=
sorry

end shaniqua_haircuts_l56_56719


namespace numberOfAntiPalindromes_l56_56649

-- Define what it means for a number to be an anti-palindrome in base 3
def isAntiPalindrome (n : ℕ) : Prop :=
  ∀ (a b : ℕ), a + b = 2 → a ≠ b

-- Define the constraint of no two consecutive digits being the same
def noConsecutiveDigits (digits : List ℕ) : Prop :=
  ∀ (i : ℕ), i < digits.length - 1 → digits.nthLe i sorry ≠ digits.nthLe (i + 1) sorry

-- We want to find the number of anti-palindromes less than 3^12 fulfilling both conditions
def countAntiPalindromes (m : ℕ) (base : ℕ) : ℕ :=
  sorry -- Placeholder definition for the count, to be implemented

-- The main theorem to prove
theorem numberOfAntiPalindromes : countAntiPalindromes (3^12) 3 = 126 :=
  sorry -- Proof to be filled

end numberOfAntiPalindromes_l56_56649


namespace tim_initial_books_l56_56435

def books_problem : Prop :=
  ∃ T : ℕ, 10 + T - 24 = 19 ∧ T = 33

theorem tim_initial_books : books_problem :=
  sorry

end tim_initial_books_l56_56435


namespace reflection_line_coordinates_sum_l56_56892

theorem reflection_line_coordinates_sum (m b : ℝ)
  (h : ∀ (x y x' y' : ℝ), (x, y) = (-4, 2) → (x', y') = (2, 6) → 
  ∃ (m b : ℝ), y = m * x + b ∧ y' = m * x' + b ∧ ∀ (p q : ℝ), 
  (p, q) = ((x+x')/2, (y+y')/2) → p = ((-4 + 2)/2) ∧ q = ((2 + 6)/2)) :
  m + b = 1 :=
by
  sorry

end reflection_line_coordinates_sum_l56_56892


namespace rectangle_area_l56_56622

theorem rectangle_area (w l: ℝ) (h1: l = 2 * w) (h2: 2 * l + 2 * w = 4) : l * w = 8 / 9 := by
  sorry

end rectangle_area_l56_56622


namespace dividend_expression_l56_56946

theorem dividend_expression 
  (D d q r P : ℕ)
  (hq_square : ∃ k, q = k^2)
  (hd_expr1 : d = 3 * r + 2)
  (hd_expr2 : d = 5 * q)
  (hr_val : r = 6)
  (hD_expr : D = d * q + r)
  (hP_prime : Prime P)
  (hP_div_D : P ∣ D)
  (hP_factor : P = 2 ∨ P = 43) :
  D = 86 := 
sorry

end dividend_expression_l56_56946


namespace find_a_l56_56673

theorem find_a (x a : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) :
  A = (7, 1) ∧ B = (1, 4) ∧ C = (x, a * x) ∧ 
  (x - 7, a * x - 1) = (2 * (1 - x), 2 * (4 - a * x)) → 
  a = 1 :=
sorry

end find_a_l56_56673


namespace boudin_hormel_ratio_l56_56376

noncomputable def ratio_boudin_hormel : Prop :=
  let foster_chickens := 45
  let american_bottles := 2 * foster_chickens
  let hormel_chickens := 3 * foster_chickens
  let del_monte_bottles := american_bottles - 30
  let total_items := 375
  ∃ (boudin_chickens : ℕ), 
    foster_chickens + american_bottles + hormel_chickens + boudin_chickens + del_monte_bottles = total_items ∧
    boudin_chickens / hormel_chickens = 1 / 3

theorem boudin_hormel_ratio : ratio_boudin_hormel :=
sorry

end boudin_hormel_ratio_l56_56376


namespace remainder_7325_mod_11_l56_56462

theorem remainder_7325_mod_11 : 7325 % 11 = 6 := sorry

end remainder_7325_mod_11_l56_56462


namespace isosceles_triangle_area_l56_56386

theorem isosceles_triangle_area
  (a b : ℝ) -- sides of the triangle
  (inradius : ℝ) (perimeter : ℝ)
  (angle : ℝ) -- angle in degrees
  (h_perimeter : 2 * a + b = perimeter)
  (h_inradius : inradius = 2.5)
  (h_angle : angle = 40)
  (h_perimeter_value : perimeter = 20)
  (h_semiperimeter : (perimeter / 2) = 10) :
  (inradius * (perimeter / 2) = 25) :=
by
  sorry

end isosceles_triangle_area_l56_56386


namespace area_problem_a_area_problem_b_area_problem_c_area_problem_d_l56_56360

-- Problem a
theorem area_problem_a : 
  (let 𝛺 := {p : ℝ × ℝ | 3 * p.1 ^ 2 = 25 * p.2 ∧ 5 * p.2 ^ 2 = 9 * p.1} in 
  measure_theory.measure_space.volume.1 𝛺) = 7 :=
sorry

-- Problem b
theorem area_problem_b : 
  (let 𝛺 := {p : ℝ × ℝ | p.1 * p.2 = 4 ∧ p.1 + p.2 = 5} in 
  measure_theory.measure_space.volume.1 𝛺) = 15 / 2 - 8 * Real.log 2 :=
sorry

-- Problem c
theorem area_problem_c :
  (let 𝛺 := {p : ℝ × ℝ | exp p.1 ≤ p.2 ∧ p.2 ≤ exp (2 * p.1) ∧ p.1 ≤ 1} in 
  measure_theory.measure_space.volume.1 𝛺) = (Real.exp 1 - 1) ^ 2 / 2 :=
sorry

-- Problem d
theorem area_problem_d :
  (let 𝛺 := {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ p.1 + 3 * p.2 = 1 ∧ p.1 = p.2 ∧ p.1 + 2 * p.2 = 2} in
  measure_theory.measure_space.volume.1 𝛺) = 11 / 12 :=
sorry

end area_problem_a_area_problem_b_area_problem_c_area_problem_d_l56_56360


namespace find_integer_pairs_l56_56240

def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, k * k = x

theorem find_integer_pairs (m n : ℤ) :
  (is_perfect_square (m^2 + 4 * n) ∧ is_perfect_square (n^2 + 4 * m)) ↔
  (∃ a : ℤ, (m = 0 ∧ n = a^2) ∨ (m = a^2 ∧ n = 0) ∨ (m = -4 ∧ n = -4) ∨ (m = -5 ∧ n = -6) ∨ (m = -6 ∧ n = -5)) :=
by
  sorry

end find_integer_pairs_l56_56240


namespace problem_l56_56318

variable {w z : ℝ}

theorem problem (hw : w = 8) (hz : z = 3) (h : ∀ z w, z * (w^(1/3)) = 6) : w = 1 :=
by
  sorry

end problem_l56_56318


namespace retirement_year_l56_56630

-- Define the basic conditions
def rule_of_70 (age: ℕ) (years_of_employment: ℕ) : Prop :=
  age + years_of_employment ≥ 70

def age_in_hiring_year : ℕ := 32
def hiring_year : ℕ := 1987

theorem retirement_year : ∃ y: ℕ, rule_of_70 (age_in_hiring_year + y) y ∧ (hiring_year + y = 2006) :=
  sorry

end retirement_year_l56_56630


namespace baker_cakes_l56_56791

theorem baker_cakes : (62.5 + 149.25 - 144.75 = 67) :=
by
  sorry

end baker_cakes_l56_56791


namespace find_constants_to_satisfy_equation_l56_56040

-- Define the condition
def equation_condition (x : ℝ) (A B C : ℝ) :=
  -2 * x^2 + 5 * x - 6 = A * (x^2 + 1) + (B * x + C) * x

-- Define the proof problem as a Lean 4 statement
theorem find_constants_to_satisfy_equation (A B C : ℝ) :
  A = -6 ∧ B = 4 ∧ C = 5 ↔ ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 → equation_condition x A B C := 
by
  sorry

end find_constants_to_satisfy_equation_l56_56040


namespace least_possible_z_minus_x_l56_56555

theorem least_possible_z_minus_x (x y z : ℕ) 
  (hx_prime : Nat.Prime x) (hy_prime : Nat.Prime y) (hz_prime : Nat.Prime z)
  (hxy : x < y) (hyz : y < z) (hyx_gt_3: y - x > 3)
  (hx_even : x % 2 = 0) (hy_odd : y % 2 = 1) (hz_odd : z % 2 = 1) :
  z - x = 9 :=
sorry

end least_possible_z_minus_x_l56_56555


namespace trig_inequality_l56_56702
open Real

theorem trig_inequality (α β γ x y z : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : x + y + z = 0) :
  y * z * (sin α)^2 + z * x * (sin β)^2 + x * y * (sin γ)^2 ≤ 0 := 
sorry

end trig_inequality_l56_56702


namespace value_of_b_minus_a_l56_56684

theorem value_of_b_minus_a (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 3) (h3 : a < b) : b - a = 2 ∨ b - a = 4 :=
sorry

end value_of_b_minus_a_l56_56684


namespace sector_angle_l56_56973

theorem sector_angle (r : ℝ) (S : ℝ) (α : ℝ) (h₁ : r = 10) (h₂ : S = 50 * π / 3) (h₃ : S = 1 / 2 * r^2 * α) : 
  α = π / 3 :=
by
  sorry

end sector_angle_l56_56973


namespace white_line_longer_l56_56955

-- Define the lengths of the white and blue lines
def white_line_length : ℝ := 7.678934
def blue_line_length : ℝ := 3.33457689

-- State the main theorem
theorem white_line_longer :
  white_line_length - blue_line_length = 4.34435711 :=
by
  sorry

end white_line_longer_l56_56955


namespace num_prime_divisors_50_factorial_eq_15_l56_56544

theorem num_prime_divisors_50_factorial_eq_15 :
  (Finset.filter Nat.Prime (Finset.range 51)).card = 15 :=
by
  sorry

end num_prime_divisors_50_factorial_eq_15_l56_56544


namespace max_b_value_l56_56735

theorem max_b_value (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 10 :=
sorry

end max_b_value_l56_56735


namespace positive_difference_even_odd_sums_l56_56181

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l56_56181


namespace smallest_positive_multiple_of_37_l56_56463

theorem smallest_positive_multiple_of_37 (a : ℕ) (h1 : 37 * a ≡ 3 [MOD 101]) (h2 : ∀ b : ℕ, 0 < b ∧ (37 * b ≡ 3 [MOD 101]) → a ≤ b) : 37 * a = 1628 :=
sorry

end smallest_positive_multiple_of_37_l56_56463


namespace ratio_of_spinsters_to_cats_l56_56316

-- Definitions for the conditions given:
def S : ℕ := 12 -- 12 spinsters
def C : ℕ := S + 42 -- 42 more cats than spinsters
def ratio (a b : ℕ) : ℚ := a / b -- Ratio definition

-- The theorem stating the required equivalence:
theorem ratio_of_spinsters_to_cats :
  ratio S C = 2 / 9 :=
by
  -- This proof has been omitted for the purpose of this exercise.
  sorry

end ratio_of_spinsters_to_cats_l56_56316


namespace polynomial_remainder_l56_56663

theorem polynomial_remainder :
  ∀ (x : ℝ), (x^4 + 2 * x^3 - 3 * x^2 + 4 * x - 5) % (x^2 - 3 * x + 2) = (24 * x - 25) :=
by
  sorry

end polynomial_remainder_l56_56663


namespace dice_impossible_divisible_by_10_l56_56758

theorem dice_impossible_divisible_by_10 :
  ¬ ∃ n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ), n % 10 = 0 :=
by
  sorry

end dice_impossible_divisible_by_10_l56_56758


namespace simplify_fraction_l56_56868

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end simplify_fraction_l56_56868


namespace age_difference_l56_56404

theorem age_difference (john_age father_age mother_age : ℕ) 
    (h1 : john_age * 2 = father_age) 
    (h2 : father_age = mother_age + 4) 
    (h3 : father_age = 40) :
    mother_age - john_age = 16 :=
by
  sorry

end age_difference_l56_56404


namespace max_people_in_crowd_l56_56198

theorem max_people_in_crowd : ∃ n : ℕ, n ≤ 37 ∧ 
    (⟨1 / 2 * n⟩ + ⟨1 / 3 * n⟩ + ⟨1 / 5 * n⟩ = n) :=
sorry

end max_people_in_crowd_l56_56198


namespace books_per_shelf_l56_56025

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ) 
    (h1 : mystery_shelves = 5) (h2 : picture_shelves = 4) (h3 : total_books = 54) : 
    total_books / (mystery_shelves + picture_shelves) = 6 := 
by
  -- necessary preliminary steps and full proof will go here
  sorry

end books_per_shelf_l56_56025


namespace initial_ratio_of_milk_to_water_l56_56847

theorem initial_ratio_of_milk_to_water (M W : ℕ) (h1 : M + W = 60) (h2 : 2 * M = W + 60) : M / W = 2 :=
by
  sorry

end initial_ratio_of_milk_to_water_l56_56847


namespace eccentricity_of_hyperbola_range_dot_product_OQ_no_point_P_l56_56531

open Real

-- Define the hyperbola
def hyperbola := {P : ℝ × ℝ // P.1^2 - P.2^2 / 3 = 1}

-- Define the foci F1 and F2
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define point P on the right branch
def is_on_right_branch (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P ∈ hyperbola

-- Problem 1: Prove the eccentricity of the hyperbola is 2
theorem eccentricity_of_hyperbola : 
  (let a := 1 in let b := sqrt 3 in sqrt (a^2 + b^2)) = 2 := 
by sorry

-- Problem 2: Prove the range of ∠OP · ∠OQ is (-∞, -5]
theorem range_dot_product_OQ (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  is_on_right_branch P → ∃ Q ∈ hyperbola, ((P.1 * Q.1 + P.2 * Q.2) ≤ -5) := 
by sorry

-- Problem 3: Prove no point P satisfies |PM| + |PN| = √2
theorem no_point_P (P : ℝ × ℝ) :
  is_on_right_branch P →
  ¬ (∃ M N : ℝ × ℝ, 
     let asymptote1 := {P | P.2 = sqrt 3 * P.1} in
     let asymptote2 := {P | P.2 = -sqrt 3 * P.1} in
     let distance_to_asymptote(a : set (ℝ × ℝ), P : ℝ × ℝ) := abs ((sqrt 3 * P.1 - P.2) / sqrt 4) in
     distance_to_asymptote asymptote1 P + distance_to_asymptote asymptote2 P = sqrt 2 ) := 
by sorry

end eccentricity_of_hyperbola_range_dot_product_OQ_no_point_P_l56_56531


namespace positive_difference_l56_56156

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l56_56156


namespace ball_distribution_l56_56551

theorem ball_distribution (n m : Nat) (h_n : n = 6) (h_m : m = 2) : 
  ∃ ways, 
    (ways = 2 ^ n - (1 + n)) ∧ ways = 57 :=
by
  sorry

end ball_distribution_l56_56551


namespace simplify_fraction_l56_56871

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end simplify_fraction_l56_56871


namespace height_difference_between_crates_l56_56322

theorem height_difference_between_crates 
  (n : ℕ) (diameter : ℝ) 
  (height_A : ℝ) (height_B : ℝ) :
  n = 200 →
  diameter = 12 →
  height_A = n / 10 * diameter →
  height_B = n / 20 * (diameter + 6 * Real.sqrt 3) →
  height_A - height_B = 120 - 60 * Real.sqrt 3 :=
sorry

end height_difference_between_crates_l56_56322


namespace bisection_method_root_exists_bisection_method_next_calculation_l56_56627

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_method_root_exists :
  (f 0 < 0) → (f 0.5 > 0) → ∃ x0 : ℝ, 0 < x0 ∧ x0 < 0.5 ∧ f x0 = 0 :=
by
  intro h0 h05
  sorry

theorem bisection_method_next_calculation :
  f 0.25 = (0.25)^3 + 3 * 0.25 - 1 :=
by
  calc
    f 0.25 = 0.25^3 + 3 * 0.25 - 1 := rfl

end bisection_method_root_exists_bisection_method_next_calculation_l56_56627


namespace max_additional_spheres_in_cone_l56_56933

-- Definition of spheres O_{1} and O_{2} properties
def O₁_radius : ℝ := 2
def O₂_radius : ℝ := 3
def height_cone : ℝ := 8

-- Conditions:
def O₁_on_axis (h : ℝ) := height_cone > 0 ∧ h = O₁_radius
def O₁_tangent_top_base := height_cone = O₁_radius + O₁_radius
def O₂_tangent_O₁ := O₁_radius + O₂_radius = 5
def O₂_on_base := O₂_radius = 3

-- Lean theorem stating mathematically equivalent proof problem
theorem max_additional_spheres_in_cone (h : ℝ) :
  O₁_on_axis h → O₁_tangent_top_base →
  O₂_tangent_O₁ → O₂_on_base →
  ∃ n : ℕ, n = 2 :=
by
  sorry

end max_additional_spheres_in_cone_l56_56933


namespace inequality_solution_set_l56_56681

theorem inequality_solution_set (a c x : ℝ) 
  (h1 : -1/3 < x ∧ x < 1/2 → 0 < a * x^2 + 2 * x + c) :
  -2 < x ∧ x < 3 ↔ -c * x^2 + 2 * x - a > 0 :=
by sorry

end inequality_solution_set_l56_56681


namespace total_cost_price_of_items_l56_56229

/-- 
  Definition of the selling prices of the items A, B, and C.
  Definition of the profit percentages of the items A, B, and C.
  The statement is the total cost price calculation.
-/
def ItemA_SP : ℝ := 800
def ItemA_Profit : ℝ := 0.25
def ItemB_SP : ℝ := 1200
def ItemB_Profit : ℝ := 0.20
def ItemC_SP : ℝ := 1500
def ItemC_Profit : ℝ := 0.30

theorem total_cost_price_of_items :
  let CP_A := ItemA_SP / (1 + ItemA_Profit)
  let CP_B := ItemB_SP / (1 + ItemB_Profit)
  let CP_C := ItemC_SP / (1 + ItemC_Profit)
  CP_A + CP_B + CP_C = 2793.85 :=
by
  sorry

end total_cost_price_of_items_l56_56229


namespace average_price_per_dvd_l56_56031

-- Define the conditions
def num_movies_box1 : ℕ := 10
def price_per_movie_box1 : ℕ := 2
def num_movies_box2 : ℕ := 5
def price_per_movie_box2 : ℕ := 5

-- Define total calculations based on conditions
def total_cost_box1 : ℕ := num_movies_box1 * price_per_movie_box1
def total_cost_box2 : ℕ := num_movies_box2 * price_per_movie_box2

def total_cost : ℕ := total_cost_box1 + total_cost_box2
def total_movies : ℕ := num_movies_box1 + num_movies_box2

-- Define the average price per DVD and prove it to be 3
theorem average_price_per_dvd : total_cost / total_movies = 3 := by
  sorry

end average_price_per_dvd_l56_56031


namespace find_weight_of_second_square_l56_56635

-- Define given conditions
def side_length1 : ℝ := 4
def weight1 : ℝ := 16
def side_length2 : ℝ := 6

-- Define the uniform density and thickness condition
def uniform_density (a₁ a₂ : ℝ) (w₁ w₂ : ℝ) : Prop :=
  (a₁ * w₂ = a₂ * w₁)

-- Problem statement:
theorem find_weight_of_second_square : 
  uniform_density (side_length1 ^ 2) (side_length2 ^ 2) weight1 w₂ → 
  w₂ = 36 :=
by
  sorry

end find_weight_of_second_square_l56_56635


namespace cubic_binomial_expansion_l56_56607

theorem cubic_binomial_expansion :
  49^3 + 3 * 49^2 + 3 * 49 + 1 = 125000 :=
by
  sorry

end cubic_binomial_expansion_l56_56607


namespace calc_theoretical_yield_l56_56598
-- Importing all necessary libraries

-- Define the molar masses
def molar_mass_NaNO3 : ℝ := 85

-- Define the initial moles
def initial_moles_NH4NO3 : ℝ := 2
def initial_moles_NaOH : ℝ := 2

-- Define the final yield percentage
def yield_percentage : ℝ := 0.85

-- State the proof problem
theorem calc_theoretical_yield :
  let moles_NaNO3 := (2 : ℝ) * 2 * yield_percentage
  let grams_NaNO3 := moles_NaNO3 * molar_mass_NaNO3
  grams_NaNO3 = 289 :=
by 
  sorry

end calc_theoretical_yield_l56_56598


namespace total_length_of_intervals_l56_56372

theorem total_length_of_intervals :
  (∀ (x : ℝ), |x| < 1 → Real.tan (Real.log x / Real.log 5) < 0) →
  ∃ (length : ℝ), length = (2 * (5 ^ (Real.pi / 2))) / (1 + (5 ^ (Real.pi / 2))) :=
sorry

end total_length_of_intervals_l56_56372


namespace people_who_speak_French_l56_56275

theorem people_who_speak_French (T L N B : ℕ) (hT : T = 25) (hL : L = 13) (hN : N = 6) (hB : B = 9) : 
  ∃ F : ℕ, F = 15 := 
by 
  sorry

end people_who_speak_French_l56_56275


namespace intersection_of_M_and_N_l56_56683

open Set

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2 - |x|}

theorem intersection_of_M_and_N : M ∩ N = {y | 0 ≤ y ∧ y ≤ 2} :=
by sorry

end intersection_of_M_and_N_l56_56683


namespace Carla_servings_l56_56644

-- Define the volumes involved
def volume_watermelon : ℕ := 500
def volume_cream : ℕ := 100
def volume_per_serving : ℕ := 150

-- The total volume is the sum of the watermelon and cream volumes
def total_volume : ℕ := volume_watermelon + volume_cream

-- The number of servings is the total volume divided by the volume per serving
def n_servings : ℕ := total_volume / volume_per_serving

-- The theorem to prove that Carla can make 4 servings of smoothies
theorem Carla_servings : n_servings = 4 := by
  sorry

end Carla_servings_l56_56644


namespace sum_of_monomials_same_type_l56_56843

theorem sum_of_monomials_same_type 
  (x y : ℝ) 
  (m n : ℕ) 
  (h1 : m = 1) 
  (h2 : 3 = n + 1) : 
  (2 * x ^ m * y ^ 3) + (-5 * x * y ^ (n + 1)) = -3 * x * y ^ 3 := 
by 
  sorry

end sum_of_monomials_same_type_l56_56843


namespace self_descriptive_7_digit_first_digit_is_one_l56_56009

theorem self_descriptive_7_digit_first_digit_is_one
  (A B C D E F G : ℕ)
  (h_total : A + B + C + D + E + F + G = 7)
  (h_B : B = 2)
  (h_C : C = 1)
  (h_D : D = 1)
  (h_E : E = 0)
  (h_A_zeroes : A = (if E = 0 then 1 else 0)) :
  A = 1 :=
by
  sorry

end self_descriptive_7_digit_first_digit_is_one_l56_56009


namespace edge_length_of_cube_l56_56685

theorem edge_length_of_cube (V : ℝ) (e : ℝ) (h1 : V = 2744) (h2 : V = e^3) : e = 14 := 
by 
  sorry

end edge_length_of_cube_l56_56685


namespace problem1_problem2_l56_56498

-- Problem 1
theorem problem1 (a b : ℝ) : (a + 2 * b)^2 - a * (a + 4 * b) = 4 * b^2 :=
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) (h : m ≠ 1) : 
  (2 / (m - 1) + 1) / (2 * (m + 1) / (m^2 - 2 * m + 1)) = (m - 1) / 2 :=
by
  sorry

end problem1_problem2_l56_56498


namespace positive_difference_sums_even_odd_l56_56145

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l56_56145


namespace reach_one_from_45_reach_one_from_345_reach_one_from_any_nat_l56_56918

theorem reach_one_from_45 : ∃ (n : ℕ), n = 1 :=
by
  -- Start from 45 and follow the given steps to reach 1.
  sorry

theorem reach_one_from_345 : ∃ (n : ℕ), n = 1 :=
by
  -- Start from 345 and follow the given steps to reach 1.
  sorry

theorem reach_one_from_any_nat (n : ℕ) (h : n ≠ 0) : ∃ (k : ℕ), k = 1 :=
by
  -- Prove that starting from any non-zero natural number, you can reach 1.
  sorry

end reach_one_from_45_reach_one_from_345_reach_one_from_any_nat_l56_56918


namespace base_number_in_exponent_l56_56188

theorem base_number_in_exponent (x : ℝ) (k : ℕ) (h₁ : k = 8) (h₂ : 64^k > x^22) : 
  x = 2^(24/11) :=
sorry

end base_number_in_exponent_l56_56188


namespace positive_difference_even_odd_sums_l56_56178

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l56_56178


namespace enrique_speed_l56_56952

theorem enrique_speed (distance : ℝ) (time : ℝ) (speed_diff : ℝ) (E : ℝ) :
  distance = 200 ∧ time = 8 ∧ speed_diff = 7 ∧ 
  (2 * E + speed_diff) * time = distance → 
  E = 9 :=
by
  sorry

end enrique_speed_l56_56952


namespace prob_at_least_7_is_1_over_9_l56_56985

open BigOperators -- Open namespace for big operators

-- Given definitions
def certain_people: ℕ := 4
def uncertain_people: ℕ := 4
def probability_stay: ℚ := 1 / 3

-- Probability of exactly k out of n people with certain probability staying
def binomial_prob (n k: ℕ) (p: ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1-p)^(n-k))

-- Probability that at least 7 people stayed
def prob_at_least_7 := 
  binomial_prob 4 3 probability_stay + binomial_prob 4 4 probability_stay

-- Theorem: The probability that at least 7 people stayed the entire time is 1/9
theorem prob_at_least_7_is_1_over_9 : 
  prob_at_least_7 = 1 / 9 := by sorry

end prob_at_least_7_is_1_over_9_l56_56985


namespace math_or_sci_but_not_both_l56_56957

-- Definitions of the conditions
variable (students_math_and_sci : ℕ := 15)
variable (students_math : ℕ := 30)
variable (students_only_sci : ℕ := 18)

-- The theorem to prove
theorem math_or_sci_but_not_both :
  (students_math - students_math_and_sci) + students_only_sci = 33 := by
  -- Proof is omitted.
  sorry

end math_or_sci_but_not_both_l56_56957


namespace speed_ratio_correct_l56_56345

noncomputable def boat_speed_still_water := 12 -- Boat's speed in still water (in mph)
noncomputable def current_speed := 4 -- Current speed of the river (in mph)

-- Calculate the downstream speed
noncomputable def downstream_speed := boat_speed_still_water + current_speed

-- Calculate the upstream speed
noncomputable def upstream_speed := boat_speed_still_water - current_speed

-- Assume a distance for the trip (1 mile each up and down)
noncomputable def distance := 1

-- Calculate time for downstream
noncomputable def time_downstream := distance / downstream_speed

-- Calculate time for upstream
noncomputable def time_upstream := distance / upstream_speed

-- Calculate total time for the round trip
noncomputable def total_time := time_downstream + time_upstream

-- Calculate total distance for the round trip
noncomputable def total_distance := 2 * distance

-- Calculate the average speed for the round trip
noncomputable def avg_speed_trip := total_distance / total_time

-- Calculate the ratio of average speed to speed in still water
noncomputable def speed_ratio := avg_speed_trip / boat_speed_still_water

theorem speed_ratio_correct : speed_ratio = 8/9 := by
  sorry

end speed_ratio_correct_l56_56345


namespace sin_of_5pi_over_6_l56_56236

theorem sin_of_5pi_over_6 : Real.sin (5 * Real.pi / 6) = 1 / 2 := 
by
  sorry

end sin_of_5pi_over_6_l56_56236


namespace multiplication_with_negative_l56_56362

theorem multiplication_with_negative (a b : Int) (h1 : a = 3) (h2 : b = -2) : a * b = -6 :=
by
  sorry

end multiplication_with_negative_l56_56362


namespace total_number_of_dresses_l56_56295

theorem total_number_of_dresses (ana_dresses lisa_more_dresses : ℕ) (h_condition : ana_dresses = 15) (h_more : lisa_more_dresses = ana_dresses + 18) : ana_dresses + lisa_more_dresses = 48 :=
by
  sorry

end total_number_of_dresses_l56_56295


namespace math_problem_correct_l56_56881

noncomputable def math_problem : Prop :=
  (1 / ((3 / (Real.sqrt 5 + 2)) - (1 / (Real.sqrt 4 + 1)))) = ((27 * Real.sqrt 5 + 57) / 40)

theorem math_problem_correct : math_problem := by
  sorry

end math_problem_correct_l56_56881


namespace find_angles_l56_56511

open Real Set

def match_sets (s1 s2 : Set ℝ) : Prop :=
  ∀ x, x ∈ s1 ↔ x ∈ s2

theorem find_angles (α : ℝ) :
  match_sets {sin α, sin (2 * α), sin (3 * α)} {cos α, cos (2 * α), cos (3 * α)} ↔ 
  ∃ k : ℤ, α = (π / 3) + (π * k / 2) :=
sorry

end find_angles_l56_56511


namespace John_total_amount_l56_56403

theorem John_total_amount (x : ℝ)
  (h1 : ∃ x : ℝ, (3 * x * 5 * 3 * x) = 300):
  (x + 3 * x + 15 * x) = 380 := by
  sorry

end John_total_amount_l56_56403


namespace fraction_divisible_by_1963_l56_56583

theorem fraction_divisible_by_1963 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ,
    13 * 733^n + 1950 * 582^n = 1963 * k ∧
    ∃ m : ℤ,
      333^n - 733^n - 1068^n + 431^n = 1963 * m :=
by
  sorry

end fraction_divisible_by_1963_l56_56583


namespace simplify_fraction_l56_56873

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end simplify_fraction_l56_56873


namespace positive_difference_sums_l56_56170

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l56_56170


namespace time_interval_between_recordings_is_5_seconds_l56_56851

theorem time_interval_between_recordings_is_5_seconds
  (instances_per_hour : ℕ)
  (seconds_per_hour : ℕ)
  (h1 : instances_per_hour = 720)
  (h2 : seconds_per_hour = 3600) :
  seconds_per_hour / instances_per_hour = 5 :=
by
  -- proof omitted
  sorry

end time_interval_between_recordings_is_5_seconds_l56_56851


namespace factor_correct_l56_56956

noncomputable def factor_expr (x : ℝ) : ℝ :=
  75 * x^3 - 225 * x^10
  
noncomputable def factored_form (x : ℝ) : ℝ :=
  75 * x^3 * (1 - 3 * x^7)

theorem factor_correct (x : ℝ): 
  factor_expr x = factored_form x :=
by
  -- Proof omitted
  sorry

end factor_correct_l56_56956


namespace mrs_smith_class_boys_girls_ratio_l56_56845

theorem mrs_smith_class_boys_girls_ratio (total_students boys girls : ℕ) (h1 : boys / girls = 3 / 4) (h2 : boys + girls = 42) : girls = boys + 6 :=
by
  sorry

end mrs_smith_class_boys_girls_ratio_l56_56845


namespace positive_difference_of_sums_l56_56122

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l56_56122


namespace sports_field_perimeter_l56_56219

noncomputable def perimeter_of_sports_field (a b : ℝ) (h1 : a^2 + b^2 = 400) (h2 : a * b = 120) : ℝ :=
  2 * (a + b)

theorem sports_field_perimeter {a b : ℝ} (h1 : a^2 + b^2 = 400) (h2 : a * b = 120) :
  perimeter_of_sports_field a b h1 h2 = 51 := by
  sorry

end sports_field_perimeter_l56_56219


namespace graph_transformation_matches_B_l56_56891

noncomputable def f (x : ℝ) : ℝ :=
  if (-3 : ℝ) ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0 -- Define this part to handle cases outside the given range.

noncomputable def g (x : ℝ) : ℝ :=
  f ((1 - x) / 2)

theorem graph_transformation_matches_B :
  g = some_graph_function_B := 
sorry

end graph_transformation_matches_B_l56_56891


namespace largest_operation_result_is_div_l56_56039

noncomputable def max_operation_result : ℚ :=
  max (max (-1 + (-1 / 2)) (-1 - (-1 / 2)))
      (max (-1 * (-1 / 2)) (-1 / (-1 / 2)))

theorem largest_operation_result_is_div :
  max_operation_result = 2 := by
  sorry

end largest_operation_result_is_div_l56_56039


namespace find_linear_equation_l56_56610

def is_linear_eq (eq : String) : Prop :=
  eq = "2x = 0"

theorem find_linear_equation :
  is_linear_eq "2x = 0" :=
by
  sorry

end find_linear_equation_l56_56610


namespace positive_diff_even_odd_sums_l56_56130

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l56_56130


namespace min_double_rooms_needed_min_triple_rooms_needed_with_discount_l56_56482

-- Define the conditions 
def double_room_price : ℕ := 200
def triple_room_price : ℕ := 250
def total_students : ℕ := 50
def male_students : ℕ := 27
def female_students : ℕ := 23
def discount : ℚ := 0.2
def max_double_rooms : ℕ := 15

-- Define the property for part (1)
theorem min_double_rooms_needed (d : ℕ) (t : ℕ) : 
  2 * d + 3 * t = total_students ∧
  2 * (d - 1) + 3 * t ≠ total_students :=
sorry

-- Define the property for part (2)
theorem min_triple_rooms_needed_with_discount (d : ℕ) (t : ℕ) : 
  d + t = total_students ∧
  d ≤ max_double_rooms ∧
  2 * d + 3 * t = total_students ∧
  (1* (d - 1) + 3 * t ≠ total_students) :=
sorry

end min_double_rooms_needed_min_triple_rooms_needed_with_discount_l56_56482


namespace jamie_cherry_pies_l56_56075

theorem jamie_cherry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 36) (h_ratio : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4) : 
  (cherry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio) = 144 / 11 := 
by {
  sorry
}

end jamie_cherry_pies_l56_56075


namespace new_paint_intensity_l56_56886

def I1 : ℝ := 0.50
def I2 : ℝ := 0.25
def F : ℝ := 0.2

theorem new_paint_intensity : (1 - F) * I1 + F * I2 = 0.45 := by
  sorry

end new_paint_intensity_l56_56886


namespace correct_calculation_l56_56912

theorem correct_calculation (m n : ℝ) : 4 * m + 2 * n - (n - m) = 5 * m + n :=
by sorry

end correct_calculation_l56_56912


namespace neg_one_third_squared_l56_56226

theorem neg_one_third_squared :
  (-(1/3))^2 = 1/9 :=
sorry

end neg_one_third_squared_l56_56226


namespace combined_mixture_nuts_l56_56723

def sue_percentage_nuts : ℝ := 0.30
def sue_percentage_dried_fruit : ℝ := 0.70

def jane_percentage_nuts : ℝ := 0.60
def combined_percentage_dried_fruit : ℝ := 0.35

theorem combined_mixture_nuts :
  let sue_contribution := 100.0
  let jane_contribution := 100.0
  let sue_nuts := sue_contribution * sue_percentage_nuts
  let jane_nuts := jane_contribution * jane_percentage_nuts
  let combined_nuts := sue_nuts + jane_nuts
  let total_weight := sue_contribution + jane_contribution
  (combined_nuts / total_weight) * 100 = 45 :=
by
  sorry

end combined_mixture_nuts_l56_56723


namespace probability_one_left_one_right_l56_56033

/-- Define the conditions: 12 left-handed gloves, 10 right-handed gloves. -/
def num_left_handed_gloves : ℕ := 12

def num_right_handed_gloves : ℕ := 10

/-- Total number of gloves is 22. -/
def total_gloves : ℕ := num_left_handed_gloves + num_right_handed_gloves

/-- Total number of ways to pick any two gloves from 22 gloves. -/
def total_pick_two_ways : ℕ := (total_gloves * (total_gloves - 1)) / 2

/-- Number of favorable outcomes picking one left-handed and one right-handed glove. -/
def favorable_outcomes : ℕ := num_left_handed_gloves * num_right_handed_gloves

/-- Define the probability as favorable outcomes divided by total outcomes. 
 It should yield 40/77. -/
theorem probability_one_left_one_right : 
  (favorable_outcomes : ℚ) / total_pick_two_ways = 40 / 77 :=
by
  -- Skip the proof.
  sorry

end probability_one_left_one_right_l56_56033


namespace determine_d_l56_56652

-- Given conditions
def equation (d x : ℝ) : Prop := 3 * (5 + d * x) = 15 * x + 15

-- Proof statement
theorem determine_d (d : ℝ) : (∀ x : ℝ, equation d x) ↔ d = 5 :=
by
  sorry

end determine_d_l56_56652


namespace total_dollars_l56_56705

def mark_dollars : ℚ := 4 / 5
def carolyn_dollars : ℚ := 2 / 5
def jack_dollars : ℚ := 1 / 2

theorem total_dollars :
  mark_dollars + carolyn_dollars + jack_dollars = 1.7 := 
sorry

end total_dollars_l56_56705


namespace divisible_by_9_l56_56861

theorem divisible_by_9 (n : ℕ) : 9 ∣ (4^n + 15 * n - 1) :=
by
  sorry

end divisible_by_9_l56_56861


namespace number_of_prime_divisors_of_factorial_l56_56538

theorem number_of_prime_divisors_of_factorial :
  let primes_leq_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] in
  ∃ (n : ℕ), n = primes_leq_50.length ∧ n = 15 :=
by
  sorry

end number_of_prime_divisors_of_factorial_l56_56538


namespace smallest_sum_of_exterior_angles_l56_56453

open Real

theorem smallest_sum_of_exterior_angles 
  (p q r : ℕ) 
  (hp : p > 2) 
  (hq : q > 2) 
  (hr : r > 2) 
  (hpq : p ≠ q) 
  (hqr : q ≠ r) 
  (hrp : r ≠ p) 
  : (360 / p + 360 / q + 360 / r) ≥ 282 ∧ 
    (360 / p + 360 / q + 360 / r) = 282 → 
    360 / p = 120 ∧ 360 / q = 90 ∧ 360 / r = 72 := 
sorry

end smallest_sum_of_exterior_angles_l56_56453


namespace feet_more_than_heads_l56_56846

def num_hens := 50
def num_goats := 45
def num_camels := 8
def num_keepers := 15

def feet_per_hen := 2
def feet_per_goat := 4
def feet_per_camel := 4
def feet_per_keeper := 2

def total_heads := num_hens + num_goats + num_camels + num_keepers
def total_feet := (num_hens * feet_per_hen) + (num_goats * feet_per_goat) + (num_camels * feet_per_camel) + (num_keepers * feet_per_keeper)

-- Theorem to prove:
theorem feet_more_than_heads : total_feet - total_heads = 224 := by
  -- proof goes here
  sorry

end feet_more_than_heads_l56_56846


namespace work_problem_l56_56478

theorem work_problem (B_rate : ℝ) (C_rate : ℝ) (A_rate : ℝ) :
  (B_rate = 1/12) →
  (B_rate + C_rate = 1/3) →
  (A_rate + C_rate = 1/2) →
  (A_rate = 1/4) :=
by
  intros h1 h2 h3
  sorry

end work_problem_l56_56478


namespace exists_graph_with_clique_smaller_than_chromatic_l56_56366

open Classical -- To use classical existence

-- Define a graph C_5
noncomputable def C_5 : SimpleGraph (Fin 5) := {
  adj := λ i j, (i.val + 1) % 5 = j.val ∨ (i.val + 4) % 5 = j.val,
  sym := by { intros i j h, cases h; { simp [h] }},
  loopless := by { intro i, simp }
}

-- Define chromatic number function (assuming such a function exists)
noncomputable def chromaticNumber (G : SimpleGraph V) : ℕ :=
sorry -- The actual implementation is omitted

-- Define clique number function (assuming such a function exists)
noncomputable def cliqueNumber (G : SimpleGraph V) : ℕ :=
sorry -- The actual implementation is omitted

theorem exists_graph_with_clique_smaller_than_chromatic :
  ∃ G : SimpleGraph (Fin 5), chromaticNumber C_5 = 3 ∧ cliqueNumber C_5 = 2 :=
begin
  use C_5,
  split,
  { sorry }, -- Proof that chromaticNumber C_5 = 3
  { sorry }  -- Proof that cliqueNumber C_5 = 2
end

end exists_graph_with_clique_smaller_than_chromatic_l56_56366


namespace incorrect_ratio_implies_l56_56036

variable {a b c d : ℝ} (h : a * d = b * c) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

theorem incorrect_ratio_implies :
  ¬ (c / b = a / d) :=
sorry

end incorrect_ratio_implies_l56_56036


namespace line_through_ellipse_and_midpoint_l56_56349

theorem line_through_ellipse_and_midpoint :
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ (x + y) = 0) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      (x₁ + x₂ = 2 ∧ y₁ + y₂ = 1) ∧
      (x₁^2 / 2 + y₁^2 = 1 ∧ x₂^2 / 2 + y₂^2 = 1) ∧
      l x₁ y₁ ∧ l x₂ y₂ ∧
      ∀ (mx my : ℝ), (mx, my) = (1, 0.5) → (mx = (x₁ + x₂) / 2 ∧ my = (y₁ + y₂) / 2))
  := sorry

end line_through_ellipse_and_midpoint_l56_56349


namespace total_bill_l56_56021

def num_adults := 2
def num_children := 5
def cost_per_meal := 3

theorem total_bill : (num_adults + num_children) * cost_per_meal = 21 := 
by 
  sorry

end total_bill_l56_56021


namespace correct_overestimation_l56_56949

theorem correct_overestimation (y : ℕ) : 
  25 * y + 4 * y = 29 * y := 
by 
  sorry

end correct_overestimation_l56_56949


namespace ratio_of_brownies_l56_56854

def total_brownies : ℕ := 15
def eaten_on_monday : ℕ := 5
def eaten_on_tuesday : ℕ := total_brownies - eaten_on_monday

theorem ratio_of_brownies : eaten_on_tuesday / eaten_on_monday = 2 := 
by
  sorry

end ratio_of_brownies_l56_56854


namespace complex_expression_l56_56516

theorem complex_expression (z : ℂ) (i : ℂ) (h1 : z^2 + 1 = 0) (h2 : i^2 = -1) : 
  (z^4 + i) * (z^4 - i) = 0 :=
sorry

end complex_expression_l56_56516


namespace required_run_rate_per_batsman_l56_56280

variable (initial_run_rate : ℝ) (overs_played : ℕ) (remaining_overs : ℕ)
variable (remaining_wickets : ℕ) (total_target : ℕ) 

theorem required_run_rate_per_batsman 
  (h_initial_run_rate : initial_run_rate = 3.4)
  (h_overs_played : overs_played = 10)
  (h_remaining_overs  : remaining_overs = 40)
  (h_remaining_wickets : remaining_wickets = 7)
  (h_total_target : total_target = 282) :
  (total_target - initial_run_rate * overs_played) / remaining_overs = 6.2 :=
by
  sorry

end required_run_rate_per_batsman_l56_56280


namespace greatest_value_of_x_l56_56310

theorem greatest_value_of_x (x : ℕ) : (Nat.lcm (Nat.lcm x 12) 18 = 180) → x ≤ 180 :=
by
  sorry

end greatest_value_of_x_l56_56310


namespace logical_impossibility_of_thoughts_l56_56608

variable (K Q : Prop)

/-- Assume that King and Queen are sane (sane is represented by them not believing they're insane) -/
def sane (p : Prop) : Prop :=
  ¬(p = true)

/-- Define the nested thoughts -/
def KingThinksQueenThinksKingThinksQueenOutOfMind (K Q : Prop) :=
  K ∧ Q ∧ K ∧ Q = ¬sane Q

/-- The main proposition -/
theorem logical_impossibility_of_thoughts (hK : sane K) (hQ : sane Q) : 
  ¬KingThinksQueenThinksKingThinksQueenOutOfMind K Q :=
by sorry

end logical_impossibility_of_thoughts_l56_56608


namespace sum_digits_largest_N_l56_56412

-- Define the conditions
def is_multiple_of_six (N : ℕ) : Prop := N % 6 = 0

def P (N : ℕ) : ℚ := 
  let favorable_positions := (N + 1) *
    (⌊(1:ℚ) / 3 * N⌋ + 1 + (N - ⌈(2:ℚ) / 3 * N⌉ + 1))
  favorable_positions / (N + 1)

axiom P_6_equals_1 : P 6 = 1
axiom P_large_N : ∀ ε > 0, ∃ N > 0, is_multiple_of_six N ∧ P N ≥ (5/6) - ε

-- Main theorem statement
theorem sum_digits_largest_N : 
  ∃ N : ℕ, is_multiple_of_six N ∧ P N > 3/4 ∧ (N.digits 10).sum = 6 :=
sorry

end sum_digits_largest_N_l56_56412


namespace probability_interval_normal_distribution_l56_56888

noncomputable def normal_distribution (μ σ : ℝ) := ProbDensityFunction.mk (fun x => (1 / (σ * (2 * Real.pi)^(1/2))) * Real.exp (-(x - μ)^2 / (2 * σ^2))) sorry sorry

theorem probability_interval_normal_distribution (μ σ : ℝ)
  (h1 : Prob (event {x | x > 5}) (normal_distribution μ σ) = 0.2)
  (h2 : Prob (event {x | x < -1}) (normal_distribution μ σ) = 0.2) : 
  Prob (event {x | 2 < x ∧ x < 5}) (normal_distribution μ σ) = 0.3 :=
sorry

end probability_interval_normal_distribution_l56_56888


namespace impossibility_of_arrangement_l56_56767

-- Definitions based on identified conditions in the problem
def isValidArrangement (arr : List ℕ) : Prop :=
  arr.length = 300 ∧
  (∀ i, i < 300 - 1 → arr.get i = |arr.get (i - 1) - arr.get (i + 1)|) ∧
  (arr.all (λ x => x > 0))

theorem impossibility_of_arrangement :
  ¬ (∃ arr : List ℕ, isValidArrangement arr) :=
sorry

end impossibility_of_arrangement_l56_56767


namespace positive_diff_even_odd_sums_l56_56132

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l56_56132


namespace impossible_300_numbers_l56_56768

theorem impossible_300_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) (hp : ∀ i, 0 < a i)
(hdiff : ∃ k, ∀ i ≠ k, a i = a ((i + 1) % n) - a ((i - 1 + n) % n)) 
: false :=
by {
  sorry
}

end impossible_300_numbers_l56_56768


namespace part1_part2_l56_56678

open Real

def f (x : ℝ) (a : ℝ) : ℝ := |x - 2| + |3 * x + a|

theorem part1 (a : ℝ) (h : a = 1) :
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} := by
  sorry

theorem part2 (h : ∃ x_0 : ℝ, f x_0 (a := a) + 2 * |x_0 - 2| < 3) : -9 < a ∧ a < -3 := by
  sorry

end part1_part2_l56_56678


namespace positive_difference_l56_56155

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l56_56155


namespace probability_bijection_l56_56975

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2, 3, 4, 5}

theorem probability_bijection : 
  let total_mappings := 5^4
  let bijections := 5 * 4 * 3 * 2
  let probability := bijections / total_mappings
  probability = 24 / 125 := 
by
  sorry

end probability_bijection_l56_56975


namespace simplify_fraction_l56_56870

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end simplify_fraction_l56_56870


namespace angle_problem_l56_56241

theorem angle_problem (θ : ℝ) (h1 : 90 - θ = 0.4 * (180 - θ)) (h2 : 180 - θ = 2 * θ) : θ = 30 :=
by
  sorry

end angle_problem_l56_56241


namespace axis_of_symmetry_l56_56688

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (4 - x)) :
  ∀ y : ℝ, (∃ x₁ x₂ : ℝ, y = f x₁ ∧ y = f x₂ ∧ (x₁ + x₂) / 2 = 2) :=
by
  sorry

end axis_of_symmetry_l56_56688


namespace value_of_x_y_mn_l56_56527

variables (x y m n : ℝ)

-- Conditions for arithmetic sequence 2, x, y, 3
def arithmetic_sequence_condition_1 : Prop := 2 * x = 2 + y
def arithmetic_sequence_condition_2 : Prop := 2 * y = 3 + x

-- Conditions for geometric sequence 2, m, n, 3
def geometric_sequence_condition_1 : Prop := m^2 = 2 * n
def geometric_sequence_condition_2 : Prop := n^2 = 3 * m

theorem value_of_x_y_mn (h1 : arithmetic_sequence_condition_1 x y) 
                        (h2 : arithmetic_sequence_condition_2 x y) 
                        (h3 : geometric_sequence_condition_1 m n)
                        (h4 : geometric_sequence_condition_2 m n) : 
  x + y + m * n = 11 :=
sorry

end value_of_x_y_mn_l56_56527


namespace mary_initial_baseball_cards_l56_56296

theorem mary_initial_baseball_cards (X : ℕ) :
  (X - 8 + 26 + 40 = 84) → (X = 26) :=
by
  sorry

end mary_initial_baseball_cards_l56_56296


namespace danny_marks_in_math_l56_56230

theorem danny_marks_in_math
  (english_marks : ℕ := 76)
  (physics_marks : ℕ := 82)
  (chemistry_marks : ℕ := 67)
  (biology_marks : ℕ := 75)
  (average_marks : ℕ := 73)
  (num_subjects : ℕ := 5) :
  ∃ (math_marks : ℕ), math_marks = 65 :=
by
  let total_marks := average_marks * num_subjects
  let other_subjects_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  have math_marks := total_marks - other_subjects_marks
  use math_marks
  sorry

end danny_marks_in_math_l56_56230


namespace at_least_n_minus_one_linear_indep_derivatives_l56_56798

variables {n : ℕ} {f : Fin n → ℝ → ℝ}
variables (hf : ∀ i : Fin n, Differentiable ℝ (f i))
variables (h_lin_indep : LinearIndependence ℝ (fun i : Fin n => f i))

theorem at_least_n_minus_one_linear_indep_derivatives :
  ∃ g : Fin (n - 1) → ℝ → ℝ, LinearIndependence ℝ (fun i => Deriv (f i)) :=
sorry

end at_least_n_minus_one_linear_indep_derivatives_l56_56798


namespace simplify_expression_l56_56466

theorem simplify_expression : 
  (Real.sqrt 2 * 2^(1/2) * 2) + (18 / 3 * 2) - (8^(1/2) * 4) = 16 - 8 * Real.sqrt 2 :=
by 
  sorry  -- proof omitted

end simplify_expression_l56_56466


namespace samantha_sleep_hours_l56_56093

def time_in_hours (hours minutes : ℕ) : ℕ :=
  hours + (minutes / 60)

def hours_slept (bed_time wake_up_time : ℕ) : ℕ :=
  if bed_time < wake_up_time then wake_up_time - bed_time + 12 else 24 - bed_time + wake_up_time

theorem samantha_sleep_hours : hours_slept 7 11 = 16 := by
  sorry

end samantha_sleep_hours_l56_56093


namespace compute_b_c_sum_l56_56080

def polynomial_decomposition (Q : ℝ[X]) (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ) : Prop :=
  ∀ x : ℝ, Q.eval x = (x^2 + (b1*x) + c1) * (x^2 + (b2*x) + c2) * (x^2 + (b3*x) + c3) * (x^2 + (b4*x) + c4)

theorem compute_b_c_sum (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : polynomial_decomposition (polynomial.mk [1, -1, 1, -1, 1, -1, 1, -1, 1]) b1 b2 b3 b4 c1 c2 c3 c4)
  (c1_eq : c1 = 1) (c2_eq : c2 = 1) (c3_eq : c3 = 1) (c4_eq : c4 = 1) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 := 
sorry

end compute_b_c_sum_l56_56080


namespace correct_calculation_l56_56913

theorem correct_calculation (m n : ℝ) : 4 * m + 2 * n - (n - m) = 5 * m + n :=
by sorry

end correct_calculation_l56_56913


namespace largest_difference_l56_56290

noncomputable def A := 3 * (2010: ℕ) ^ 2011
noncomputable def B := (2010: ℕ) ^ 2011
noncomputable def C := 2009 * (2010: ℕ) ^ 2010
noncomputable def D := 3 * (2010: ℕ) ^ 2010
noncomputable def E := (2010: ℕ) ^ 2010
noncomputable def F := (2010: ℕ) ^ 2009

theorem largest_difference :
  (A - B) > (B - C) ∧ (A - B) > (C - D) ∧ (A - B) > (D - E) ∧ (A - B) > (E - F) :=
by
  sorry

end largest_difference_l56_56290


namespace proof_sets_l56_56420

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 3, 6}
def complement (s : Set ℕ) : Set ℕ := {x | x ∈ I ∧ x ∉ s}

theorem proof_sets :
  M ∩ (complement N) = {4, 5} ∧ {2, 7, 8} = complement (M ∪ N) :=
by
  sorry

end proof_sets_l56_56420


namespace count_prime_divisors_50_factorial_l56_56545

/-- The count of prime divisors of 50! is 15 -/
theorem count_prime_divisors_50_factorial : 
  (set.primes.filter (λ p, p ≤ 50)).card = 15 :=
by
  sorry

end count_prime_divisors_50_factorial_l56_56545


namespace value_of_x0_l56_56986

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x
noncomputable def f_deriv (x : ℝ) : ℝ := ((x - 1) * Real.exp x) / (x * x)

theorem value_of_x0 (x0 : ℝ) (h : f_deriv x0 = -f x0) : x0 = 1 / 2 := by
  sorry

end value_of_x0_l56_56986


namespace red_balls_count_is_correct_l56_56011

-- Define conditions
def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def purple_balls : ℕ := 3
def non_red_purple_prob : ℝ := 0.9

-- Define the number of red balls
def number_of_red_balls (red_balls : ℕ) : Prop :=
  total_balls - (white_balls + green_balls + yellow_balls + purple_balls) = red_balls
  
-- The proof statement
theorem red_balls_count_is_correct : number_of_red_balls 7 := by
  sorry

end red_balls_count_is_correct_l56_56011


namespace find_c_value_l56_56838

def f (c : ℝ) (x : ℝ) : ℝ := c * x^4 + (c^2 - 3) * x^2 + 1

theorem find_c_value (c : ℝ) :
  (∀ x < -1, deriv (f c) x < 0) ∧ 
  (∀ x, -1 < x → x < 0 → deriv (f c) x > 0) → 
  c = 1 :=
by 
  sorry

end find_c_value_l56_56838


namespace intersection_of_sets_l56_56970

variable (x : ℝ)
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets 
  (hA : ∀ x, x ∈ A ↔ -2 < x ∧ x ≤ 1)
  (hB : ∀ x, x ∈ B ↔ 0 < x ∧ x ≤ 1) :
  ∀ x, (x ∈ A ∩ B) ↔ (0 < x ∧ x ≤ 1) := 
by
  sorry

end intersection_of_sets_l56_56970


namespace original_number_is_85_l56_56786

theorem original_number_is_85
  (x : ℤ) (h_sum : 10 ≤ x ∧ x < 100) 
  (h_condition1 : (x / 10) + (x % 10) = 13)
  (h_condition2 : 10 * (x % 10) + (x / 10) = x - 27) :
  x = 85 :=
by
  sorry

end original_number_is_85_l56_56786


namespace alice_has_ball_after_three_turns_l56_56355

open ProbabilityMassFunction

/-- Alice and Bob's ball game probability problem -/
def alice_bob_game (P : Population (Fin 4)) : ProbabilityMassFunction (Fin 2) :=
  let first_turn := if P = 0 then PMF.ofMultiset [(0, 1 / 3), (1, 2 / 3)] else PMF.ofMultiset [(1, 2 / 3), (0, 1 / 3)]
  let second_turn := if first_turn = 0 then PMF.ofMultiset [(0, 1 / 3), (1, 2 / 3)] else PMF.ofMultiset [(1, 2 / 3), (0, 1 / 3)]
  let third_turn := if second_turn = 0 then PMF.ofMultiset [(0, 1 / 3), (1, 2 / 3)] else PMF.ofMultiset [(1, 2 / 3), (0, 1 / 3)]
  third_turn

/-- Probability of Alice having the ball after three turns is 5/9 given the game rules -/
theorem alice_has_ball_after_three_turns:
  alice_bob_game 0.V.ProbabilityTree.stashing 3 0 = (5 / 9) := sorry

end alice_has_ball_after_three_turns_l56_56355


namespace weight_difference_calc_l56_56495

-- Define the weights in pounds
def Anne_weight : ℕ := 67
def Douglas_weight : ℕ := 52
def Maria_weight : ℕ := 48

-- Define the combined weight of Douglas and Maria
def combined_weight_DM : ℕ := Douglas_weight + Maria_weight

-- Define the weight difference
def weight_difference : ℤ := Anne_weight - combined_weight_DM

-- The theorem stating the difference
theorem weight_difference_calc : weight_difference = -33 := by
  -- The proof will go here
  sorry

end weight_difference_calc_l56_56495


namespace positive_difference_even_odd_sums_l56_56176

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l56_56176


namespace positive_difference_even_odd_sums_l56_56158

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l56_56158


namespace matilda_father_chocolates_l56_56712

theorem matilda_father_chocolates 
  (total_chocolates : ℕ) 
  (total_people : ℕ) 
  (give_up_fraction : ℚ) 
  (mother_chocolates : ℕ) 
  (father_eats : ℕ) 
  (father_left : ℕ) :
  total_chocolates = 20 →
  total_people = 5 →
  give_up_fraction = 1 / 2 →
  mother_chocolates = 3 →
  father_eats = 2 →
  father_left = 5 →
  let chocolates_per_person := total_chocolates / total_people,
      father_chocolates := (chocolates_per_person * total_people * give_up_fraction).nat_abs - mother_chocolates - father_eats
  in father_chocolates = father_left := by
  intros h1 h2 h3 h4 h5 h6
  have h_chocolates_per_person : total_chocolates / total_people = 4 := by sorry
  have h_chocolates_given_up : (chocolates_per_person * total_people * give_up_fraction).nat_abs = 10 := by sorry
  have h_father_chocolates : 10 - mother_chocolates - father_eats = 5 := by sorry
  exact h_father_chocolates

end matilda_father_chocolates_l56_56712


namespace other_number_remainder_l56_56911

theorem other_number_remainder (x : ℕ) (k n : ℤ) (hx : x > 0) (hk : 200 = k * x + 2) (hnk : n ≠ k) : ∃ m : ℤ, (n * ↑x + 2) = m * ↑x + 2 ∧ (n * ↑x + 2) % x = 2 := 
by
  sorry

end other_number_remainder_l56_56911


namespace increase_by_percentage_l56_56628

def initial_value : ℕ := 550
def percentage_increase : ℚ := 0.35
def final_value : ℚ := 742.5

theorem increase_by_percentage :
  (initial_value : ℚ) * (1 + percentage_increase) = final_value := by
  sorry

end increase_by_percentage_l56_56628


namespace mean_problem_l56_56763

theorem mean_problem (x : ℝ) (h : (12 + x + 42 + 78 + 104) / 5 = 62) :
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 :=
by
  sorry

end mean_problem_l56_56763


namespace proportional_function_decreases_l56_56057

theorem proportional_function_decreases
  (k : ℝ) (h : k ≠ 0) (h_point : ∃ k, (-4 : ℝ) = k * 2) :
  ∀ x1 x2 : ℝ, x1 < x2 → (k * x1) > (k * x2) :=
by
  sorry

end proportional_function_decreases_l56_56057


namespace solve_for_x_l56_56438

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 8) * x = 12) : x = 168 := by
  sorry

end solve_for_x_l56_56438


namespace peter_stamps_l56_56432

theorem peter_stamps (M : ℕ) (h1 : M % 5 = 2) (h2 : M % 11 = 2) (h3 : M % 13 = 2) (h4 : M > 1) : M = 717 :=
by
  -- proof will be filled in
  sorry

end peter_stamps_l56_56432


namespace remainder_div_30_l56_56419

-- Define the conditions as Lean definitions
variables (x y z p q : ℕ)

-- Hypotheses based on the conditions
def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- assuming the conditions
axiom x_div_by_4 : is_divisible_by x 4
axiom y_div_by_5 : is_divisible_by y 5
axiom z_div_by_6 : is_divisible_by z 6
axiom p_div_by_7 : is_divisible_by p 7
axiom q_div_by_3 : is_divisible_by q 3

-- Statement to be proved
theorem remainder_div_30 : ((x^3) * (y^2) * (z * p * q + (x + y)^3) - 10) % 30 = 20 :=
by {
  sorry -- the proof will go here
}

end remainder_div_30_l56_56419


namespace instantaneous_velocity_at_3_l56_56486

noncomputable def s (t : ℝ) : ℝ := t^2 + 10

theorem instantaneous_velocity_at_3 :
  deriv s 3 = 6 :=
by {
  -- proof goes here
  sorry
}

end instantaneous_velocity_at_3_l56_56486


namespace intersection_of_M_and_N_l56_56256

-- Define sets M and N as given in the conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- The theorem statement to prove the intersection of M and N is {2, 3}
theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by sorry  -- The proof is skipped with 'sorry'

end intersection_of_M_and_N_l56_56256


namespace trapezoid_sides_l56_56013

theorem trapezoid_sides (r kl: ℝ) (h1 : r = 5) (h2 : kl = 8) :
  ∃ (ab cd bc_ad : ℝ), ab = 5 ∧ cd = 20 ∧ bc_ad = 12.5 :=
by
  sorry

end trapezoid_sides_l56_56013


namespace list_price_of_article_l56_56102

theorem list_price_of_article 
(paid_price : ℝ) 
(first_discount second_discount : ℝ)
(list_price : ℝ)
(h_paid_price : paid_price = 59.22)
(h_first_discount : first_discount = 0.10)
(h_second_discount : second_discount = 0.06000000000000002)
(h_final_price : paid_price = (1 - first_discount) * (1 - second_discount) * list_price) :
  list_price = 70 := 
by
  sorry

end list_price_of_article_l56_56102


namespace positive_difference_of_sums_l56_56124

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l56_56124


namespace odd_expression_l56_56413

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem odd_expression (k m : ℤ) (o := 2 * k + 3) (n := 2 * m) :
  is_odd (o^2 + n * o) :=
by sorry

end odd_expression_l56_56413


namespace intersection_eq_l56_56259

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof Problem
theorem intersection_eq : M ∩ N = {2, 3} := 
by
  sorry

end intersection_eq_l56_56259


namespace intersection_m_n_l56_56828

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := { -1, 0, 1, 2, 3 }

theorem intersection_m_n : M ∩ N = {0, 1, 2} := 
sorry

end intersection_m_n_l56_56828


namespace range_of_a_l56_56682

theorem range_of_a (a : ℝ) : (1 ∉ {x : ℝ | (x - a) / (x + a) < 0}) → ( -1 ≤ a ∧ a ≤ 1 ) := 
by
  intro h
  sorry

end range_of_a_l56_56682


namespace john_horizontal_distance_l56_56284

theorem john_horizontal_distance
  (vertical_distance_ratio horizontal_distance_ratio : ℕ)
  (initial_elevation final_elevation : ℕ)
  (h_ratio : vertical_distance_ratio = 1)
  (h_dist_ratio : horizontal_distance_ratio = 3)
  (h_initial : initial_elevation = 500)
  (h_final : final_elevation = 3450) :
  (final_elevation - initial_elevation) * horizontal_distance_ratio = 8850 := 
by {
  sorry
}

end john_horizontal_distance_l56_56284


namespace sum_possible_m_continuous_l56_56418

noncomputable def g (x m : ℝ) : ℝ :=
if x < m then x^2 + 4 * x + 3 else 3 * x + 9

theorem sum_possible_m_continuous :
  let m₁ := -3
  let m₂ := 2
  m₁ + m₂ = -1 :=
by
  sorry

end sum_possible_m_continuous_l56_56418


namespace find_number_l56_56617

theorem find_number (N : ℝ) (h : 0.1 * 0.3 * 0.5 * N = 90) : N = 6000 :=
by
  sorry

end find_number_l56_56617


namespace prime_divisors_50_num_prime_divisors_50_l56_56542

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.range' 1 n.succ)

def primes_up_to (n : ℕ) : List ℕ :=
  List.filter is_prime (List.range' 2 (n+1))

def prime_divisors (n : ℕ) : List ℕ :=
  List.filter (λ p, p ∣ factorial n) (primes_up_to n)

theorem prime_divisors_50 :
  prime_divisors 50 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] :=
by
  sorry

theorem num_prime_divisors_50 :
  List.length (prime_divisors 50) = 15 :=
by
  sorry

end prime_divisors_50_num_prime_divisors_50_l56_56542


namespace value_of_m_squared_plus_2m_minus_3_l56_56824

theorem value_of_m_squared_plus_2m_minus_3 (m : ℤ) : 
  (∀ x : ℤ, 4 * (x - 1) - m * x + 6 = 8 → x = 3) →
  m^2 + 2 * m - 3 = 5 :=
by
  sorry

end value_of_m_squared_plus_2m_minus_3_l56_56824


namespace rope_cut_into_pieces_l56_56344

theorem rope_cut_into_pieces (length_of_rope_cm : ℕ) (num_equal_pieces : ℕ) (length_equal_piece_mm : ℕ) (length_remaining_piece_mm : ℕ) 
  (h1 : length_of_rope_cm = 1165) (h2 : num_equal_pieces = 150) (h3 : length_equal_piece_mm = 75) (h4 : length_remaining_piece_mm = 100) :
  (num_equal_pieces * length_equal_piece_mm + (11650 - num_equal_pieces * length_equal_piece_mm) / length_remaining_piece_mm = 154) :=
by
  sorry

end rope_cut_into_pieces_l56_56344


namespace hare_height_l56_56559

theorem hare_height (camel_height_ft : ℕ) (hare_height_in_inches : ℕ) :
  (camel_height_ft = 28) ∧ (hare_height_in_inches * 24 = camel_height_ft * 12) → hare_height_in_inches = 14 :=
by
  sorry

end hare_height_l56_56559


namespace simplify_polynomial_l56_56880

theorem simplify_polynomial : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 3 * x^2 + 6 * x - 8) = x^3 + x^2 + 3 * x + 3 :=
by
  sorry

end simplify_polynomial_l56_56880


namespace angle_sum_property_l56_56557

theorem angle_sum_property
  (angle1 angle2 angle3 : ℝ) 
  (h1 : angle1 = 58) 
  (h2 : angle2 = 35) 
  (h3 : angle3 = 42) : 
  angle1 + angle2 + angle3 + (180 - (angle1 + angle2 + angle3)) = 180 := 
by 
  sorry

end angle_sum_property_l56_56557


namespace coin_flip_probability_difference_l56_56746

theorem coin_flip_probability_difference :
  let p := 1 / 2,
  p_3 := (Nat.choose 4 3) * (p ^ 3) * (p ^ 1),
  p_4 := p ^ 4
  in abs (p_3 - p_4) = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l56_56746


namespace book_cost_l56_56556

theorem book_cost (x : ℝ) 
  (h1 : Vasya_has = x - 150)
  (h2 : Tolya_has = x - 200)
  (h3 : (x - 150) + (x - 200) / 2 = x + 100) : x = 700 :=
sorry

end book_cost_l56_56556


namespace sheelas_total_net_monthly_income_l56_56582

noncomputable def totalNetMonthlyIncome
    (PrimaryJobIncome : ℝ)
    (FreelanceIncome : ℝ)
    (FreelanceIncomeTaxRate : ℝ)
    (AnnualInterestIncome : ℝ)
    (InterestIncomeTaxRate : ℝ) : ℝ :=
    let PrimaryJobMonthlyIncome := 5000 / 0.20
    let FreelanceIncomeTax := FreelanceIncome * FreelanceIncomeTaxRate
    let NetFreelanceIncome := FreelanceIncome - FreelanceIncomeTax
    let InterestIncomeTax := AnnualInterestIncome * InterestIncomeTaxRate
    let NetAnnualInterestIncome := AnnualInterestIncome - InterestIncomeTax
    let NetMonthlyInterestIncome := NetAnnualInterestIncome / 12
    PrimaryJobMonthlyIncome + NetFreelanceIncome + NetMonthlyInterestIncome

theorem sheelas_total_net_monthly_income :
    totalNetMonthlyIncome 25000 3000 0.10 2400 0.05 = 27890 := 
by
    sorry

end sheelas_total_net_monthly_income_l56_56582


namespace positive_difference_prob_3_and_4_heads_l56_56751

theorem positive_difference_prob_3_and_4_heads :
  let p_3 := (choose 4 3) * (1 / 2) ^ 3 * (1 / 2) in
  let p_4 := (1 / 2) ^ 4 in
  abs (p_3 - p_4) = 7 / 16 :=
by
  -- Definitions for binomial coefficient and probabilities
  let p_3 := (Nat.choose 4 3) * (1 / 2)^3 * (1 / 2)
  let p_4 := (1 / 2)^4
  -- The difference between probabilities
  let diff := p_3 - p_4
  -- The desired equality to prove
  show abs diff = 7 / 16
  sorry

end positive_difference_prob_3_and_4_heads_l56_56751


namespace inequality_a4_b4_c4_l56_56092

theorem inequality_a4_b4_c4 (a b c : Real) : a^4 + b^4 + c^4 ≥ abc * (a + b + c) := 
by
  sorry

end inequality_a4_b4_c4_l56_56092


namespace sticker_sum_mod_problem_l56_56190

theorem sticker_sum_mod_problem :
  ∃ N < 100, (N % 6 = 5) ∧ (N % 8 = 6) ∧ (N = 47 ∨ N = 95) ∧ (47 + 95 = 142) :=
by
  sorry

end sticker_sum_mod_problem_l56_56190


namespace max_sum_of_digits_l56_56264

theorem max_sum_of_digits (X Y Z : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (hY : 1 ≤ Y ∧ Y ≤ 9) (hZ : 1 ≤ Z ∧ Z ≤ 9) (hXYZ : X > Y ∧ Y > Z) : 
  10 * X + 11 * Y + Z ≤ 185 :=
  sorry

end max_sum_of_digits_l56_56264


namespace solve_for_x_and_y_l56_56761

theorem solve_for_x_and_y (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : x = 10 ∧ y = 5 :=
by
  sorry

end solve_for_x_and_y_l56_56761


namespace units_digit_square_l56_56759

theorem units_digit_square (n : ℕ) (h1 : n ≥ 10 ∧ n < 100) (h2 : (n % 10 = 2) ∨ (n % 10 = 7)) :
  ∀ (d : ℕ), (d = 2 ∨ d = 6 ∨ d = 3) → (n^2 % 10 ≠ d) :=
by
  sorry

end units_digit_square_l56_56759


namespace chair_arrangements_l56_56694

theorem chair_arrangements :
  let total_chairs := 10
  let unique_positions := total_chairs + 1
  let stool_positions := choose total_chairs 3
  unique_positions * stool_positions = 1320 :=
by
  sorry

end chair_arrangements_l56_56694


namespace polygon_sides_l56_56988

theorem polygon_sides (n : ℕ) : 
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by
  sorry

end polygon_sides_l56_56988


namespace positive_difference_even_odd_l56_56151

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l56_56151


namespace tony_squat_capacity_l56_56458

theorem tony_squat_capacity :
  let curl_weight := 90 in
  let military_press_weight := 2 * curl_weight in
  let total_squat_weight := 5 * military_press_weight in
  total_squat_weight = 900 := by
  sorry

end tony_squat_capacity_l56_56458


namespace positive_difference_sums_l56_56171

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l56_56171


namespace simplify_fraction_l56_56872

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end simplify_fraction_l56_56872


namespace jelly_ratio_l56_56927

theorem jelly_ratio (G S R P : ℕ) 
  (h1 : G = 2 * S)
  (h2 : R = 2 * P) 
  (h3 : P = 6) 
  (h4 : S = 18) : 
  R / G = 1 / 3 := by
  sorry

end jelly_ratio_l56_56927


namespace fraction_of_money_left_l56_56642

theorem fraction_of_money_left 
  (m c : ℝ) 
  (h1 : (1/4 : ℝ) * m = (1/2) * c) : 
  (m - c) / m = (1/2 : ℝ) :=
by
  -- the proof will be written here
  sorry

end fraction_of_money_left_l56_56642


namespace activity_order_l56_56106

open Rat

-- Define the fractions of students liking each activity
def soccer_fraction : ℚ := 13 / 40
def swimming_fraction : ℚ := 9 / 24
def baseball_fraction : ℚ := 11 / 30
def hiking_fraction : ℚ := 3 / 10

-- Statement of the problem
theorem activity_order :
  soccer_fraction = 39 / 120 ∧
  swimming_fraction = 45 / 120 ∧
  baseball_fraction = 44 / 120 ∧
  hiking_fraction = 36 / 120 ∧
  45 / 120 > 44 / 120 ∧
  44 / 120 > 39 / 120 ∧
  39 / 120 > 36 / 120 →
  ("Swimming, Baseball, Soccer, Hiking" = "Swimming, Baseball, Soccer, Hiking") :=
by
  sorry

end activity_order_l56_56106


namespace inequality_proof_l56_56691

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  1 < (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) ∧
  (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) < 2 :=
sorry

end inequality_proof_l56_56691


namespace f_diff_l56_56979

def f (n : ℕ) : ℚ := (1 / 3 : ℚ) * n * (n + 1) * (n + 2)

theorem f_diff (r : ℕ) : f r - f (r - 1) = r * (r + 1) := 
by {
  -- proof goes here
  sorry
}

end f_diff_l56_56979


namespace sum_of_a3_a4_a5_l56_56558

def geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = 3 * q ^ n

theorem sum_of_a3_a4_a5 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : geometric_sequence_sum a q)
  (h_pos : ∀ n, a n > 0)
  (h_first_term : a 0 = 3)
  (h_sum_first_three : a 0 + a 1 + a 2 = 21) :
  a 2 + a 3 + a 4 = 84 :=
sorry

end sum_of_a3_a4_a5_l56_56558


namespace find_number_l56_56008

theorem find_number (x : ℝ) (h : (168 / 100) * x / 6 = 354.2) : x = 1265 := 
by
  sorry

end find_number_l56_56008


namespace positive_difference_even_odd_sums_l56_56161

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l56_56161


namespace GCD_40_48_l56_56115

theorem GCD_40_48 : Int.gcd 40 48 = 8 :=
by sorry

end GCD_40_48_l56_56115


namespace parabola_hyperbola_focus_vertex_l56_56269

theorem parabola_hyperbola_focus_vertex (p : ℝ) : 
  (∃ (focus_vertex : ℝ × ℝ), focus_vertex = (2, 0) 
    ∧ focus_vertex = (p / 2, 0)) → p = 4 :=
by
  sorry

end parabola_hyperbola_focus_vertex_l56_56269


namespace expected_value_xi_l56_56110

open Finset  
open Classical  

noncomputable def pmax : Finset (Finset ℕ) := 
  (powerset (range 5).erase 0).filter (λ s, card s = 3)

noncomputable def xi (s : Finset ℕ) : ℕ := 
  s.max' (by { apply pmax.card_pos.2, simp, simp })

noncomputable def px (k : ℕ) : ℚ := 
  (pmax.filter (λ s, xi s = k)).card / pmax.card

theorem expected_value_xi : 
  let E_xi := (3 * (px 3) + 4 * (px 4) + 5 * (px 5)) in 
  E_xi = 4.5 := by
  sorry

end expected_value_xi_l56_56110


namespace graph_always_passes_fixed_point_l56_56680

theorem graph_always_passes_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  ∃ A : ℝ × ℝ, A = (-2, -1) ∧ (∀ x : ℝ, y = a^(x+2)-2 → y = -1 ∧ x = -2) :=
by
  use (-2, -1)
  sorry

end graph_always_passes_fixed_point_l56_56680


namespace square_area_l56_56072

theorem square_area (x : ℝ) (A B C D E F : ℝ)
  (h1 : E = x / 3)
  (h2 : F = (2 * x) / 3)
  (h3 : abs (B - E) = 40)
  (h4 : abs (E - F) = 40)
  (h5 : abs (F - D) = 40) :
  x^2 = 2880 :=
by
  -- Main proof here
  sorry

end square_area_l56_56072


namespace problem1a_problem1b_l56_56211

noncomputable theory

def valid_purchase_price (a : ℤ) : Prop :=
  600 * a = 1300 * (a - 140)

def maximize_profit (x : ℤ) : Prop :=
  let y := (200 * x) / 2 + 120 * x / 2 + 20 * (5 * x + 20 - 2 * x)
  x + 5 * x + 20 ≤ 200 ∧ y = 9200

theorem problem1a (a : ℤ) : valid_purchase_price a ↔ a = 260 := sorry

theorem problem1b (x : ℤ) : maximize_profit x ↔ (x = 30 ∧ (5 * x + 20 = 170)) := sorry

end problem1a_problem1b_l56_56211


namespace find_b_in_geometric_sequence_l56_56073

theorem find_b_in_geometric_sequence (a_1 : ℤ) :
  ∀ (n : ℕ), ∃ (b : ℤ), (3^n - b = (a_1 * (3^n - 1)) / 2) :=
by
  sorry

example (a_1 : ℤ) :
  ∃ (b : ℤ), ∀ (n : ℕ), 3^n - b = (a_1 * (3^n - 1)) / 2 :=
by
  use 1
  sorry

end find_b_in_geometric_sequence_l56_56073


namespace identify_linear_equation_l56_56612

def is_linear_equation (eq : String) : Prop := sorry

theorem identify_linear_equation :
  is_linear_equation "2x = 0" ∧ ¬is_linear_equation "x^2 - 4x = 3" ∧ ¬is_linear_equation "x + 2y = 1" ∧ ¬is_linear_equation "x - 1 = 1 / x" :=
by 
  sorry

end identify_linear_equation_l56_56612


namespace jason_egg_consumption_l56_56804

-- Definition for the number of eggs Jason consumes per day
def eggs_per_day : ℕ := 3

-- Definition for the number of days in a week
def days_in_week : ℕ := 7

-- Definition for the number of weeks we are considering
def weeks : ℕ := 2

-- The statement we want to prove, which combines all the conditions and provides the final answer
theorem jason_egg_consumption : weeks * days_in_week * eggs_per_day = 42 := by
sorry

end jason_egg_consumption_l56_56804


namespace correct_operation_l56_56914

variables {a b : ℝ}

theorem correct_operation : (5 * a * b - 6 * a * b = -1 * a * b) := by
  sorry

end correct_operation_l56_56914


namespace number_of_terms_in_arithmetic_sequence_is_20_l56_56670

theorem number_of_terms_in_arithmetic_sequence_is_20
  (a : ℕ → ℤ)
  (common_difference : ℤ)
  (h1 : common_difference = 2)
  (even_num_terms : ℕ)
  (h2 : ∃ k, even_num_terms = 2 * k)
  (sum_odd_terms sum_even_terms : ℤ)
  (h3 : sum_odd_terms = 15)
  (h4 : sum_even_terms = 35)
  (h5 : ∀ n, a n = a 0 + n * common_difference) :
  even_num_terms = 20 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_is_20_l56_56670


namespace deck_width_l56_56018

theorem deck_width (w : ℝ) : 
  (10 + 2 * w) * (12 + 2 * w) = 360 → w = 4 := 
by 
  sorry

end deck_width_l56_56018


namespace intersection_eq_l56_56258

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof Problem
theorem intersection_eq : M ∩ N = {2, 3} := 
by
  sorry

end intersection_eq_l56_56258


namespace inequality_solution_l56_56094

theorem inequality_solution (x : ℝ) :
  (x ≠ -1 ∧ x ≠ -2 ∧ x ≠ 5) →
  ((x * x - 4 * x - 5) / (x * x + 3 * x + 2) < 0 ↔ (x ∈ Set.Ioo (-2:ℝ) (-1:ℝ) ∨ x ∈ Set.Ioo (-1:ℝ) (5:ℝ))) :=
by
  sorry

end inequality_solution_l56_56094


namespace single_discount_equivalence_l56_56782

variable (p : ℝ) (d1 d2 d3 : ℝ)

def apply_discount (price discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_multiple_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem single_discount_equivalence :
  p = 1200 →
  d1 = 0.15 →
  d2 = 0.10 →
  d3 = 0.05 →
  let final_price_multiple := apply_multiple_discounts p [d1, d2, d3]
  let single_discount := (p - final_price_multiple) / p
  single_discount = 0.27325 :=
by
  intros h1 h2 h3 h4
  let final_price_multiple := apply_multiple_discounts p [d1, d2, d3]
  let single_discount := (p - final_price_multiple) / p
  sorry

end single_discount_equivalence_l56_56782


namespace Megan_not_lead_actress_l56_56421

-- Define the conditions: total number of plays and lead actress percentage
def totalPlays : ℕ := 100
def leadActressPercentage : ℕ := 80

-- Define what we need to prove: the number of times Megan was not the lead actress
theorem Megan_not_lead_actress (totalPlays: ℕ) (leadActressPercentage: ℕ) : 
  (totalPlays * (100 - leadActressPercentage)) / 100 = 20 :=
by
  -- proof omitted
  sorry

end Megan_not_lead_actress_l56_56421


namespace plants_needed_correct_l56_56852

def total_plants_needed (ferns palms succulents total_desired : ℕ) : ℕ :=
 total_desired - (ferns + palms + succulents)

theorem plants_needed_correct : total_plants_needed 3 5 7 24 = 9 := by
  sorry

end plants_needed_correct_l56_56852


namespace no_solution_part_a_no_solution_part_b_l56_56616

theorem no_solution_part_a 
  (x y z : ℕ) :
  ¬(x^2 + y^2 + z^2 = 2 * x * y * z) := 
sorry

theorem no_solution_part_b 
  (x y z u : ℕ) :
  ¬(x^2 + y^2 + z^2 + u^2 = 2 * x * y * z * u) := 
sorry

end no_solution_part_a_no_solution_part_b_l56_56616


namespace baron_munchausen_max_people_l56_56199

theorem baron_munchausen_max_people :
  ∃ x : ℕ, (x = 37) ∧ 
  (1 / 2 * x).nat_ceil + (1 / 3 * x).nat_ceil + (1 / 5 * x).nat_ceil = x := sorry

end baron_munchausen_max_people_l56_56199


namespace sum_of_cubes_1998_l56_56037

theorem sum_of_cubes_1998 : 1998 = 334^3 + 332^3 + (-333)^3 + (-333)^3 := by
  sorry

end sum_of_cubes_1998_l56_56037


namespace question1_question2_l56_56679

def f (x : ℝ) : ℝ := |x + 7| + |x - 1|

theorem question1 (x : ℝ) : ∀ m : ℝ, (∀ x : ℝ, f x ≥ m) → m ≤ 8 :=
by sorry

theorem question2 (x : ℝ) : (∀ x : ℝ, |x - 3| - 2 * x ≤ 2 * 8 - 12) ↔ (x ≥ -1/3) :=
by sorry

end question1_question2_l56_56679


namespace B_joined_after_8_months_l56_56477

-- Define the initial investments and time
def A_investment : ℕ := 36000
def B_investment : ℕ := 54000
def profit_ratio_A_B := 2 / 1

-- Define a proposition which states that B joined the business after x = 8 months
theorem B_joined_after_8_months (x : ℕ) (h : (A_investment * 12) / (B_investment * (12 - x)) = profit_ratio_A_B) : x = 8 :=
by
  sorry

end B_joined_after_8_months_l56_56477


namespace ratio_of_spinsters_to_cats_l56_56449

theorem ratio_of_spinsters_to_cats :
  (∀ S C : ℕ, (S : ℚ) / (C : ℚ) = 2 / 9) ↔
  (∃ S C : ℕ, S = 18 ∧ C = S + 63 ∧ (S : ℚ) / (C : ℚ) = 2 / 9) :=
sorry

end ratio_of_spinsters_to_cats_l56_56449


namespace find_number_l56_56064

-- Definitions from the conditions
def condition1 (x : ℝ) := 16 * x = 3408
def condition2 (x : ℝ) := 1.6 * x = 340.8

-- The statement to prove
theorem find_number (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x = 213 :=
by
  sorry

end find_number_l56_56064


namespace find_ratio_l56_56267

variables {a b c d : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
variables (h5 : (5 * a + b) / (5 * c + d) = (6 * a + b) / (6 * c + d))
variables (h6 : (7 * a + b) / (7 * c + d) = 9)

theorem find_ratio (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
    (h5 : (5 * a + b) / (5 * c + d) = (6 * a + b) / (6 * c + d))
    (h6 : (7 * a + b) / (7 * c + d) = 9) :
    (9 * a + b) / (9 * c + d) = 9 := 
by {
    sorry
}

end find_ratio_l56_56267


namespace positive_difference_even_odd_sum_l56_56182

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l56_56182


namespace Megan_not_lead_plays_l56_56425

-- Define the problem's conditions as variables
def total_plays : ℕ := 100
def lead_play_ratio : ℤ := 80

-- Define the proposition we want to prove
theorem Megan_not_lead_plays : 
  (total_plays - (total_plays * lead_play_ratio / 100)) = 20 := 
by sorry

end Megan_not_lead_plays_l56_56425


namespace smallest_k_for_Δk_un_zero_l56_56227

def u (n : ℕ) : ℤ := n^3 - n

def Δ (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0     => u
  | (k+1) => λ n => Δ k u (n+1) - Δ k u n

theorem smallest_k_for_Δk_un_zero (u : ℕ → ℤ) (h : ∀ n, u n = n^3 - n) :
  ∀ n, Δ 4 u n = 0 ∧ (∀ k < 4, ∃ n, Δ k u n ≠ 0) :=
by
  sorry

end smallest_k_for_Δk_un_zero_l56_56227


namespace maria_trip_distance_l56_56950

theorem maria_trip_distance
  (D : ℝ)
  (h1 : D/2 = D/8 + 210) :
  D = 560 :=
sorry

end maria_trip_distance_l56_56950


namespace find_a_from_conditions_l56_56813

theorem find_a_from_conditions
  (a b : ℤ)
  (h₁ : 2584 * a + 1597 * b = 0)
  (h₂ : 1597 * a + 987 * b = -1) :
  a = 1597 :=
by sorry

end find_a_from_conditions_l56_56813


namespace factorial_inequality_l56_56716

theorem factorial_inequality (n : ℕ) (h : n > 1) : n! < ( (n + 1) / 2 )^n := by
  sorry

end factorial_inequality_l56_56716


namespace problem_solution_l56_56023

theorem problem_solution : (324^2 - 300^2) / 24 = 624 :=
by 
  -- The proof will be inserted here.
  sorry

end problem_solution_l56_56023


namespace sum_y_coordinates_of_intersection_with_y_axis_l56_56796

-- Define the center and radius of the circle
def center : ℝ × ℝ := (-4, 5)
def radius : ℝ := 9

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + center.1)^2 + (y - center.2)^2 = radius^2

theorem sum_y_coordinates_of_intersection_with_y_axis : 
  ∃ y1 y2 : ℝ, circle_eq 0 y1 ∧ circle_eq 0 y2 ∧ y1 + y2 = 10 :=
by
  sorry

end sum_y_coordinates_of_intersection_with_y_axis_l56_56796


namespace sheets_per_day_l56_56218

-- Definitions based on conditions
def total_sheets : ℕ := 60
def total_days_per_week : ℕ := 7
def days_off : ℕ := 2

-- Derived condition from the problem
def work_days_per_week : ℕ := total_days_per_week - days_off

-- The statement to prove
theorem sheets_per_day : total_sheets / work_days_per_week = 12 :=
by
  sorry

end sheets_per_day_l56_56218


namespace tables_chairs_legs_l56_56357

theorem tables_chairs_legs (t : ℕ) (c : ℕ) (total_legs : ℕ) 
  (h1 : c = 8 * t) 
  (h2 : total_legs = 4 * c + 6 * t) 
  (h3 : total_legs = 798) : 
  t = 21 :=
by
  sorry

end tables_chairs_legs_l56_56357


namespace conditional_probability_of_A_given_target_hit_l56_56071

theorem conditional_probability_of_A_given_target_hit :
  (3 / 5 : ℚ) * ( ( 4 / 5 + 1 / 5) ) = (15 / 23 : ℚ) :=
  sorry

end conditional_probability_of_A_given_target_hit_l56_56071


namespace digit_difference_is_7_l56_56066

def local_value (d : Nat) (place : Nat) : Nat :=
  d * (10^place)

def face_value (d : Nat) : Nat :=
  d

def difference (d : Nat) (place : Nat) : Nat :=
  local_value d place - face_value d

def numeral : Nat := 65793

theorem digit_difference_is_7 :
  ∃ d place, 0 ≤ d ∧ d < 10 ∧ difference d place = 693 ∧ d ∈ [6, 5, 7, 9, 3] ∧ numeral = 65793 ∧
  (local_value 6 4 = 60000 ∧ local_value 5 3 = 5000 ∧ local_value 7 2 = 700 ∧ local_value 9 1 = 90 ∧ local_value 3 0 = 3 ∧
   face_value 6 = 6 ∧ face_value 5 = 5 ∧ face_value 7 = 7 ∧ face_value 9 = 9 ∧ face_value 3 = 3) ∧ 
  d = 7 :=
sorry

end digit_difference_is_7_l56_56066


namespace john_replace_bedroom_doors_l56_56697

variable (B O : ℕ)
variable (cost_outside cost_bedroom total_cost : ℕ)

def john_has_to_replace_bedroom_doors : Prop :=
  let outside_doors_replaced := 2
  let cost_of_outside_door := 20
  let cost_of_bedroom_door := 10
  let total_replacement_cost := 70
  O = outside_doors_replaced ∧
  cost_outside = cost_of_outside_door ∧
  cost_bedroom = cost_of_bedroom_door ∧
  total_cost = total_replacement_cost ∧
  20 * O + 10 * B = total_cost →
  B = 3

theorem john_replace_bedroom_doors : john_has_to_replace_bedroom_doors B O cost_outside cost_bedroom total_cost :=
sorry

end john_replace_bedroom_doors_l56_56697


namespace find_digit_l56_56577

theorem find_digit:
  ∃ d: ℕ, d < 1000 ∧ 1995 * d = 610470 :=
  sorry

end find_digit_l56_56577


namespace total_students_in_lunchroom_l56_56200

theorem total_students_in_lunchroom (students_per_table : ℕ) (num_tables : ℕ) (total_students : ℕ) :
  students_per_table = 6 → 
  num_tables = 34 → 
  total_students = students_per_table * num_tables → 
  total_students = 204 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_students_in_lunchroom_l56_56200


namespace find_a_of_exp_function_l56_56839

theorem find_a_of_exp_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a ^ 2 = 9) : a = 3 :=
sorry

end find_a_of_exp_function_l56_56839


namespace parallel_lines_slope_l56_56671

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 → 2 * x + (a + 1) * y + 1 = 0) →
  a = -3 :=
by
  sorry

end parallel_lines_slope_l56_56671


namespace determine_N_l56_56503

/-- 
Each row and two columns in the grid forms distinct arithmetic sequences.
Given:
- First column values: 10 and 18 (arithmetic sequence).
- Second column top value: N, bottom value: -23 (arithmetic sequence).
Prove that N = -15.
 -/
theorem determine_N : ∃ N : ℤ, (∀ n : ℕ, 10 + n * 8 = 10 ∨ 10 + n * 8 = 18) ∧ (∀ m : ℕ, N + m * 8 = N ∨ N + m * 8 = -23) ∧ N = -15 :=
by {
  sorry
}

end determine_N_l56_56503


namespace find_b_l56_56003

theorem find_b (h1 : 2.236 = 1 + (b - 1) * 0.618) 
               (h2 : 2.236 = b - (b - 1) * 0.618) : 
               b = 3 ∨ b = 4.236 := 
by
  sorry

end find_b_l56_56003


namespace solution_set_f_x_sq_gt_2f_x_plus_1_l56_56250

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_x_sq_gt_2f_x_plus_1
  (h_domain : ∀ x, 0 < x → ∃ y, f y = f x)
  (h_func_equation : ∀ x y, 0 < x → 0 < y → f (x + y) = f x * f y)
  (h_greater_than_2 : ∀ x, 1 < x → f x > 2)
  (h_f2 : f 2 = 4) :
  ∀ x, x^2 > x + 2 → x > 2 :=
by
  intros x h
  sorry

end solution_set_f_x_sq_gt_2f_x_plus_1_l56_56250


namespace product_of_two_numbers_l56_56306

theorem product_of_two_numbers (x y : ℕ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 221) : x * y = 60 := sorry

end product_of_two_numbers_l56_56306


namespace checkered_square_division_l56_56849

theorem checkered_square_division (m n k d m1 n1 : ℕ) (h1 : m^2 = n * k)
  (h2 : d = Nat.gcd m n) (hm : m = m1 * d) (hn : n = n1 * d)
  (h3 : Nat.gcd m1 n1 = 1) : 
  ∃ (part_size : ℕ), 
    part_size = n ∧ (∃ (pieces : ℕ), pieces = k) ∧ m^2 = pieces * part_size := 
sorry

end checkered_square_division_l56_56849


namespace impossible_arrangement_of_300_numbers_in_circle_l56_56771

theorem impossible_arrangement_of_300_numbers_in_circle :
  ¬ ∃ (nums : Fin 300 → ℕ), (∀ i : Fin 300, nums i > 0) ∧
    ∃ unique_exception : Fin 300,
      ∀ i : Fin 300, i ≠ unique_exception → nums i = Int.natAbs (nums (Fin.mod (i.val - 1) 300) - nums (Fin.mod (i.val + 1) 300)) := 
sorry

end impossible_arrangement_of_300_numbers_in_circle_l56_56771


namespace brocard_inequality_part_a_brocard_inequality_part_b_l56_56297

variable (α β γ φ : ℝ)

theorem brocard_inequality_part_a (h_sum_angles : α + β + γ = π) (h_brocard : 0 < φ ∧ φ < π/2) :
  φ^3 ≤ (α - φ) * (β - φ) * (γ - φ) := 
sorry

theorem brocard_inequality_part_b (h_sum_angles : α + β + γ = π) (h_brocard : 0 < φ ∧ φ < π/2) :
  8 * φ^3 ≤ α * β * γ := 
sorry

end brocard_inequality_part_a_brocard_inequality_part_b_l56_56297


namespace assistant_stop_time_l56_56779

-- Define the start time for the craftsman
def craftsmanStartTime : Nat := 8 * 60 -- in minutes

-- Craftsman starts at 8:00 AM and stops at 12:00 PM
def craftsmanEndTime : Nat := 12 * 60 -- in minutes

-- Craftsman produces 6 bracelets every 20 minutes
def craftsmanProductionPerMinute : Nat := 6 / 20

-- Assistant starts working at 9:00 AM
def assistantStartTime : Nat := 9 * 60 -- in minutes

-- Assistant produces 8 bracelets every 30 minutes
def assistantProductionPerMinute : Nat := 8 / 30

-- Total production duration for craftsman in minutes
def craftsmanWorkDuration : Nat := craftsmanEndTime - craftsmanStartTime

-- Total bracelets produced by craftsman
def totalBraceletsCraftsman : Nat := craftsmanWorkDuration * craftsmanProductionPerMinute

-- Time it takes for the assistant to produce the same number of bracelets
def assistantWorkDuration : Nat := totalBraceletsCraftsman / assistantProductionPerMinute

-- Time the assistant will stop working
def assistantEndTime : Nat := assistantStartTime + assistantWorkDuration

-- Convert time in minutes to hours and minutes format (output as a string for clarity)
def formatTime (timeInMinutes: Nat) : String :=
  let hours := timeInMinutes / 60
  let minutes := timeInMinutes % 60
  s! "{hours}:{if minutes < 10 then "0" else ""}{minutes}"

-- Proof goal: assistant will stop working at "13:30" (or 1:30 PM)
theorem assistant_stop_time : 
  formatTime assistantEndTime = "13:30" := 
by
  sorry

end assistant_stop_time_l56_56779


namespace greatest_fraction_lt_17_l56_56909

theorem greatest_fraction_lt_17 :
  ∃ (x : ℚ), x = 15 / 4 ∧ x^2 < 17 ∧ ∀ y : ℚ, y < 4 → y^2 < 17 → y ≤ 15 / 4 := 
by
  use 15 / 4
  sorry

end greatest_fraction_lt_17_l56_56909


namespace vaclav_multiplication_correct_l56_56600

-- Definitions of the involved numbers and their multiplication consistency.
def a : ℕ := 452
def b : ℕ := 125
def result : ℕ := 56500

-- The main theorem statement proving the correctness of the multiplication.
theorem vaclav_multiplication_correct : a * b = result :=
by sorry

end vaclav_multiplication_correct_l56_56600


namespace rational_equation_solutions_l56_56365

open Real

theorem rational_equation_solutions :
  (∃ x : ℝ, (x ≠ 1 ∧ x ≠ -1) ∧ ((x^2 - 6*x + 9) / (x - 1) - (3 - x) / (x^2 - 1) = 0)) →
  ∃ S : Finset ℝ, S.card = 2 ∧ ∀ x ∈ S, (x ≠ 1 ∧ x ≠ -1) :=
by
  sorry

end rational_equation_solutions_l56_56365


namespace eval_fraction_l56_56224

theorem eval_fraction : (144 : ℕ) = 12 * 12 → (12 ^ 10 / (144 ^ 4) : ℝ) = 144 := by
  intro h
  have h1 : (144 : ℕ) = 12 ^ 2 := by
    exact h
  sorry

end eval_fraction_l56_56224


namespace mass_percentage_of_Ca_in_CaO_is_correct_l56_56961

noncomputable def molarMass_Ca : ℝ := 40.08
noncomputable def molarMass_O : ℝ := 16.00
noncomputable def molarMass_CaO : ℝ := molarMass_Ca + molarMass_O
noncomputable def massPercentageCaInCaO : ℝ := (molarMass_Ca / molarMass_CaO) * 100

theorem mass_percentage_of_Ca_in_CaO_is_correct :
  massPercentageCaInCaO = 71.47 :=
by
  -- This is where the proof would go
  sorry

end mass_percentage_of_Ca_in_CaO_is_correct_l56_56961


namespace simple_interest_correct_l56_56931

-- Define the given conditions
def Principal : ℝ := 9005
def Rate : ℝ := 0.09
def Time : ℝ := 5

-- Define the simple interest function
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- State the theorem to prove the total interest earned
theorem simple_interest_correct : simple_interest Principal Rate Time = 4052.25 := sorry

end simple_interest_correct_l56_56931


namespace walter_age_1999_l56_56639

variable (w g : ℕ) -- represents Walter's age (w) and his grandmother's age (g) in 1994
variable (birth_sum : ℕ) (w_age_1994 : ℕ) (g_age_1994 : ℕ)

axiom h1 : g = 2 * w
axiom h2 : (1994 - w) + (1994 - g) = 3838

theorem walter_age_1999 (w g : ℕ) (h1 : g = 2 * w) (h2 : (1994 - w) + (1994 - g) = 3838) : w + 5 = 55 :=
by
  sorry

end walter_age_1999_l56_56639


namespace number_of_girls_sampled_in_third_grade_l56_56481

-- Number of total students in the high school
def total_students : ℕ := 3000

-- Number of students in each grade
def first_grade_students : ℕ := 800
def second_grade_students : ℕ := 1000
def third_grade_students : ℕ := 1200

-- Number of boys and girls in each grade
def first_grade_boys : ℕ := 500
def first_grade_girls : ℕ := 300

def second_grade_boys : ℕ := 600
def second_grade_girls : ℕ := 400

def third_grade_boys : ℕ := 800
def third_grade_girls : ℕ := 400

-- Total number of students sampled
def total_sampled_students : ℕ := 150

-- Hypothesis: stratified sampling method according to grade proportions
theorem number_of_girls_sampled_in_third_grade :
  third_grade_girls * (total_sampled_students / total_students) = 20 :=
by
  -- We will add the proof here
  sorry

end number_of_girls_sampled_in_third_grade_l56_56481


namespace compressor_stations_l56_56452

/-- 
Problem: Given three compressor stations connected by straight roads and not on the same line,
with distances satisfying:
1. x + y = 4z
2. x + z + y = x + a
3. z + y + x = 85

Prove:
- The range of values for 'a' such that the described configuration of compressor stations is 
  possible is 60.71 < a < 68.
- The distances between the compressor stations for a = 5 are x = 70, y = 0, z = 15.
--/
theorem compressor_stations (x y z a : ℝ) 
  (h1 : x + y = 4 * z)
  (h2 : x + z + y = x + a)
  (h3 : z + y + x = 85) :
  (60.71 < a ∧ a < 68) ∧ (a = 5 → x = 70 ∧ y = 0 ∧ z = 15) :=
  sorry

end compressor_stations_l56_56452


namespace eval_expression_eq_one_l56_56654

theorem eval_expression_eq_one (x : ℝ) (hx1 : x^3 + 1 = (x+1)*(x^2 - x + 1)) (hx2 : x^3 - 1 = (x-1)*(x^2 + x + 1)) :
  ( ((x+1)^3 * (x^2 - x + 1)^3 / (x^3 + 1)^3)^2 * ((x-1)^3 * (x^2 + x + 1)^3 / (x^3 - 1)^3)^2 ) = 1 :=
by
  sorry

end eval_expression_eq_one_l56_56654


namespace emily_small_gardens_l56_56367

theorem emily_small_gardens (total_seeds planted_big_garden seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41) 
  (h2 : planted_big_garden = 29) 
  (h3 : seeds_per_small_garden = 4) : 
  (total_seeds - planted_big_garden) / seeds_per_small_garden = 3 := 
by
  sorry

end emily_small_gardens_l56_56367


namespace jinho_total_distance_l56_56090

theorem jinho_total_distance (bus_distance_km : ℝ) (bus_distance_m : ℝ) (walk_distance_m : ℝ) :
  bus_distance_km = 4 → bus_distance_m = 436 → walk_distance_m = 1999 → 
  (2 * (bus_distance_km + bus_distance_m / 1000 + walk_distance_m / 1000)) = 12.87 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jinho_total_distance_l56_56090


namespace three_obtuse_impossible_l56_56007

-- Define the type for obtuse angle
def is_obtuse (θ : ℝ) : Prop :=
  90 < θ ∧ θ < 180

-- Define the main theorem stating the problem
theorem three_obtuse_impossible 
  (A B C D O : Type) 
  (angle_AOB angle_COD angle_AOD angle_COB
   angle_OAB angle_OBA angle_OBC angle_OCB
   angle_OAD angle_ODA angle_ODC angle_OCC : ℝ)
  (h1 : angle_AOB = angle_COD)
  (h2 : angle_AOD = angle_COB)
  (h_sum : angle_AOB + angle_COD + angle_AOD + angle_COB = 360)
  : ¬ (is_obtuse angle_OAB ∧ is_obtuse angle_OBC ∧ is_obtuse angle_ODA) := 
sorry

end three_obtuse_impossible_l56_56007


namespace positive_difference_even_odd_sum_l56_56187

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l56_56187


namespace Megan_not_lead_plays_l56_56424

def total_plays : ℕ := 100
def lead_percentage : ℝ := 0.80
def lead_plays : ℕ := (total_plays : ℝ * lead_percentage).toNat
def not_lead_plays : ℕ := total_plays - lead_plays

theorem Megan_not_lead_plays : not_lead_plays = 20 := by
  sorry

end Megan_not_lead_plays_l56_56424


namespace polynomial_div_remainder_l56_56461

open Polynomial

noncomputable def p : Polynomial ℤ := 2 * X^4 + 10 * X^3 - 45 * X^2 - 52 * X + 63
noncomputable def d : Polynomial ℤ := X^2 + 6 * X - 7
noncomputable def r : Polynomial ℤ := 48 * X - 70

theorem polynomial_div_remainder : p % d = r :=
sorry

end polynomial_div_remainder_l56_56461


namespace coordinates_of_N_l56_56519

theorem coordinates_of_N
  (M : ℝ × ℝ)
  (a : ℝ × ℝ)
  (x y : ℝ)
  (hM : M = (5, -6))
  (ha : a = (1, -2))
  (hMN : (x - M.1, y - M.2) = (-3 * a.1, -3 * a.2)) :
  (x, y) = (2, 0) :=
by
  sorry

end coordinates_of_N_l56_56519


namespace fraction_of_journey_by_rail_l56_56016

theorem fraction_of_journey_by_rail :
  ∀ (x : ℝ), x * 130 + (17 / 20) * 130 + 6.5 = 130 → x = 1 / 10 :=
by
  -- proof
  sorry

end fraction_of_journey_by_rail_l56_56016


namespace two_digit_number_satisfying_conditions_l56_56948

theorem two_digit_number_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 8 ∧
  ∀ p ∈ s, ∃ (a b : ℕ), p = (a, b) ∧
    (10 * a + b < 100) ∧
    (a ≥ 2) ∧
    (10 * a + b + 10 * b + a = 110) :=
by
  sorry

end two_digit_number_satisfying_conditions_l56_56948


namespace christina_total_payment_l56_56795

def item1_ticket_price : ℝ := 200
def item1_discount1 : ℝ := 0.25
def item1_discount2 : ℝ := 0.15
def item1_tax_rate : ℝ := 0.07

def item2_ticket_price : ℝ := 150
def item2_discount : ℝ := 0.30
def item2_tax_rate : ℝ := 0.10

def item3_ticket_price : ℝ := 100
def item3_discount : ℝ := 0.20
def item3_tax_rate : ℝ := 0.05

def expected_total : ℝ := 335.93

theorem christina_total_payment :
  let item1_final_price :=
    (item1_ticket_price * (1 - item1_discount1) * (1 - item1_discount2)) * (1 + item1_tax_rate)
  let item2_final_price :=
    (item2_ticket_price * (1 - item2_discount)) * (1 + item2_tax_rate)
  let item3_final_price :=
    (item3_ticket_price * (1 - item3_discount)) * (1 + item3_tax_rate)
  item1_final_price + item2_final_price + item3_final_price = expected_total :=
by
  sorry

end christina_total_payment_l56_56795


namespace simplify_fraction_l56_56877

theorem simplify_fraction : ∃ (a b : ℕ), a = 90 ∧ b = 150 ∧ (90:ℚ) / (150:ℚ) = (3:ℚ) / (5:ℚ) :=
by {
  use 90,
  use 150,
  split,
  refl,
  split,
  refl,
  sorry,
}

end simplify_fraction_l56_56877


namespace frog_escape_probability_l56_56993

def P : ℕ → ℚ
noncomputable def P 0 := 0
noncomputable def P 12 := 1
noncomputable def P (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 12 then 1
  else if 0 < n ∧ n < 12 then 
    (n.to_nat / 12) * P (n - 1) + (1 - (n.to_nat / 12)) * P (n + 1)
  else 0

theorem frog_escape_probability : P 2 = 109 / 221 := sorry

end frog_escape_probability_l56_56993


namespace bc_sum_eq_neg_one_l56_56079

variables {b1 b2 b3 b4 c1 c2 c3 c4 : ℝ}

/-- Given the equation for all real numbers x:
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4),
    prove that b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1. -/
theorem bc_sum_eq_neg_one :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4))
  → b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by {
  sorry,
}

end bc_sum_eq_neg_one_l56_56079


namespace license_plates_count_l56_56348

-- Definitions from conditions
def num_digits : ℕ := 4
def num_digits_choices : ℕ := 10
def num_letters : ℕ := 3
def num_letters_choices : ℕ := 26

-- Define the blocks and their possible arrangements
def digits_permutations : ℕ := num_digits_choices^num_digits
def letters_permutations : ℕ := num_letters_choices^num_letters
def block_positions : ℕ := 5

-- We need to show that total possible license plates is 878,800,000.
def total_plates : ℕ := digits_permutations * letters_permutations * block_positions

-- The theorem statement
theorem license_plates_count :
  total_plates = 878800000 := by
  sorry

end license_plates_count_l56_56348


namespace find_quadruples_l56_56368

def is_solution (x y z n : ℕ) : Prop :=
  x^2 + y^2 + z^2 + 1 = 2^n

theorem find_quadruples :
  ∀ x y z n : ℕ, is_solution x y z n ↔ 
  (x, y, z, n) = (1, 1, 1, 2) ∨
  (x, y, z, n) = (0, 0, 1, 1) ∨
  (x, y, z, n) = (0, 1, 0, 1) ∨
  (x, y, z, n) = (1, 0, 0, 1) ∨
  (x, y, z, n) = (0, 0, 0, 0) :=
by
  sorry

end find_quadruples_l56_56368


namespace find_t_l56_56977

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

-- Define the perpendicular condition and solve for t
theorem find_t (t : ℝ) : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 → t = -2 :=
by
  sorry

end find_t_l56_56977


namespace cory_initial_money_l56_56947

variable (cost_per_pack : ℝ) (packs : ℕ) (additional_needed : ℝ) (total_cost : ℝ) (initial_money : ℝ)

-- Conditions
def cost_per_pack_def : Prop := cost_per_pack = 49
def packs_def : Prop := packs = 2
def additional_needed_def : Prop := additional_needed = 78
def total_cost_def : Prop := total_cost = packs * cost_per_pack
def initial_money_def : Prop := initial_money = total_cost - additional_needed

-- Theorem
theorem cory_initial_money : cost_per_pack = 49 ∧ packs = 2 ∧ additional_needed = 78 → initial_money = 20 := by
  intro h
  have h1 : cost_per_pack = 49 := h.1
  have h2 : packs = 2 := h.2.1
  have h3 : additional_needed = 78 := h.2.2
  -- sorry
  sorry

end cory_initial_money_l56_56947


namespace positive_difference_even_odd_sums_l56_56167

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l56_56167


namespace positive_difference_of_sums_l56_56138

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l56_56138


namespace A_eq_B_l56_56289

def A : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 + 2 * y^2
}

def B : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 - 6 * x * y + 11 * y^2
}

theorem A_eq_B : A = B :=
by {
  sorry
}

end A_eq_B_l56_56289


namespace complement_M_l56_56974

open Set

-- Definitions and conditions
def U : Set ℝ := univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- Theorem stating the complement of M with respect to the universal set U
theorem complement_M : compl M = {x | x < -2 ∨ x > 2} :=
by
  sorry

end complement_M_l56_56974


namespace right_triangle_hypotenuse_l56_56070

theorem right_triangle_hypotenuse :
  ∃ b a : ℕ, a^2 + 1994^2 = b^2 ∧ b = 994010 :=
by
  sorry

end right_triangle_hypotenuse_l56_56070


namespace exists_no_zero_digits_divisible_by_2_pow_100_l56_56584

theorem exists_no_zero_digits_divisible_by_2_pow_100 :
  ∃ (N : ℕ), (2^100 ∣ N) ∧ (∀ d ∈ (N.digits 10), d ≠ 0) := sorry

end exists_no_zero_digits_divisible_by_2_pow_100_l56_56584


namespace positive_difference_even_odd_l56_56148

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l56_56148


namespace factorize_a_squared_plus_2a_l56_56806

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2 * a = a * (a + 2) :=
  sorry

end factorize_a_squared_plus_2a_l56_56806


namespace quadratic_passes_through_point_l56_56488

theorem quadratic_passes_through_point (a b : ℝ) (h : a ≠ 0) (h₁ : ∃ y : ℝ, y = a * 1^2 + b * 1 - 1 ∧ y = 1) : a + b + 1 = 3 :=
by
  obtain ⟨y, hy1, hy2⟩ := h₁
  sorry

end quadratic_passes_through_point_l56_56488


namespace positive_difference_even_odd_sum_l56_56186

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l56_56186


namespace largest_even_not_sum_of_two_composite_odds_l56_56029

-- Definitions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ k, k > 1 ∧ k < n ∧ n % k = 0

-- Theorem statement
theorem largest_even_not_sum_of_two_composite_odds :
  ∀ n : ℕ, is_even n → n > 0 → (¬ (∃ a b : ℕ, is_odd a ∧ is_odd b ∧ is_composite a ∧ is_composite b ∧ n = a + b)) ↔ n = 38 := 
by
  sorry

end largest_even_not_sum_of_two_composite_odds_l56_56029


namespace chandler_weeks_to_save_l56_56363

theorem chandler_weeks_to_save :
  let birthday_money := 50 + 35 + 15 + 20
  let weekly_earnings := 18
  let bike_cost := 650
  ∃ x : ℕ, (birthday_money + x * weekly_earnings) ≥ bike_cost ∧ (birthday_money + (x - 1) * weekly_earnings) < bike_cost := 
by
  sorry

end chandler_weeks_to_save_l56_56363


namespace divides_if_not_divisible_by_4_l56_56864

theorem divides_if_not_divisible_by_4 (n : ℕ) :
  (¬ (4 ∣ n)) → (5 ∣ (1^n + 2^n + 3^n + 4^n)) :=
by sorry

end divides_if_not_divisible_by_4_l56_56864


namespace parallel_x_axis_implies_conditions_l56_56261

variable (a b : ℝ)

theorem parallel_x_axis_implies_conditions (h1 : (5, a) ≠ (b, -2)) (h2 : (5, -2) = (5, a)) : a = -2 ∧ b ≠ 5 :=
sorry

end parallel_x_axis_implies_conditions_l56_56261


namespace intersecting_lines_k_value_l56_56502

theorem intersecting_lines_k_value (k : ℝ) : 
  (∃ x y : ℝ, y = 7 * x + 5 ∧ y = -3 * x - 35 ∧ y = 4 * x + k) → k = -7 :=
by
  sorry

end intersecting_lines_k_value_l56_56502


namespace binomial_coeff_x2_l56_56650

noncomputable def binomial_coeff (n k : ℕ) : ℕ := nat.choose n k

theorem binomial_coeff_x2 (x : ℝ) : 
  (∑ k in Finset.range (7 + 1), binomial_coeff 7 k * (1:ℝ) ^ (7 - k) * x ^ k) = 
  1 * x ^ 2 * 21 + ∑ k in (Finset.range (7 + 1)).filter (λ k, k ≠ 2), binomial_coeff 7 k * (1:ℝ) ^ (7 - k) * x ^ k := 
by {
    sorry
}

end binomial_coeff_x2_l56_56650


namespace min_value_at_2_l56_56626

noncomputable def min_value (x : ℝ) := x + 4 / x + 5

theorem min_value_at_2 (x : ℝ) (h : x > 0) : min_value x ≥ 9 :=
sorry

end min_value_at_2_l56_56626


namespace positive_difference_sums_even_odd_l56_56144

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l56_56144


namespace horner_method_operations_l56_56497

-- Define the polynomial
def poly (x : ℤ) : ℤ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

-- Define Horner's method evaluation for the specific polynomial at x = 2
def horners_method_evaluated (x : ℤ) : ℤ :=
  (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1)

-- Count multiplication and addition operations
def count_mul_ops : ℕ := 5
def count_add_ops : ℕ := 5

-- Proof statement
theorem horner_method_operations :
  ∀ (x : ℤ), x = 2 → 
  (count_mul_ops = 5) ∧ (count_add_ops = 5) :=
by
  intros x h
  sorry

end horner_method_operations_l56_56497


namespace regular_decagon_interior_angle_degree_measure_l56_56603

theorem regular_decagon_interior_angle_degree_measure :
  ∀ (n : ℕ), n = 10 → (2 * 180 / n : ℝ) = 144 :=
by
  sorry

end regular_decagon_interior_angle_degree_measure_l56_56603


namespace positive_difference_sums_even_odd_l56_56140

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l56_56140


namespace positive_difference_even_odd_l56_56149

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l56_56149


namespace Gumble_words_total_l56_56730

noncomputable def num_letters := 25
noncomputable def exclude_B := 24

noncomputable def total_5_letters_or_less (n : ℕ) : ℕ :=
  if h : 1 ≤ n ∧ n ≤ 5 then num_letters^n - exclude_B^n else 0

noncomputable def total_Gumble_words : ℕ :=
  (total_5_letters_or_less 1) + (total_5_letters_or_less 2) + (total_5_letters_or_less 3) +
  (total_5_letters_or_less 4) + (total_5_letters_or_less 5)

theorem Gumble_words_total :
  total_Gumble_words = 1863701 := by
  sorry

end Gumble_words_total_l56_56730


namespace equation_holds_except_two_values_l56_56923

noncomputable def check_equation (a y : ℝ) (h : a ≠ 0) : Prop :=
  (a / (a + y) + y / (a - y)) / (y / (a + y) - a / (a - y)) = -1 ↔ y ≠ a ∧ y ≠ -a

theorem equation_holds_except_two_values (a y: ℝ) (h: a ≠ 0): check_equation a y h := sorry

end equation_holds_except_two_values_l56_56923


namespace sheets_per_day_l56_56217

theorem sheets_per_day (total_sheets : ℕ) (days_per_week : ℕ) (sheets_per_day : ℕ) :
  total_sheets = 60 → days_per_week = 5 → sheets_per_day = total_sheets / days_per_week → sheets_per_day = 12 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3.symm.trans (by norm_num)

end sheets_per_day_l56_56217


namespace expand_and_simplify_expression_l56_56509

theorem expand_and_simplify_expression : 
  ∀ (x : ℝ), (3 * x - 4) * (2 * x + 6) = 6 * x^2 + 10 * x - 24 := 
by 
  intro x
  sorry

end expand_and_simplify_expression_l56_56509


namespace monkey_distance_l56_56925

-- Define the initial speeds and percentage adjustments
def swing_speed : ℝ := 10
def run_speed : ℝ := 15
def wind_resistance_percentage : ℝ := 0.10
def branch_assistance_percentage : ℝ := 0.05

-- Conditions
def adjusted_swing_speed : ℝ := swing_speed * (1 - wind_resistance_percentage)
def adjusted_run_speed : ℝ := run_speed * (1 + branch_assistance_percentage)
def run_time : ℝ := 5
def swing_time : ℝ := 10

-- Define the distance formulas based on the conditions
def run_distance : ℝ := adjusted_run_speed * run_time
def swing_distance : ℝ := adjusted_swing_speed * swing_time

-- Total distance calculation
def total_distance : ℝ := run_distance + swing_distance

-- Statement for the proof
theorem monkey_distance : total_distance = 168.75 := by
  sorry

end monkey_distance_l56_56925


namespace tangent_line_slope_angle_l56_56317

theorem tangent_line_slope_angle (θ : ℝ) : 
  (∃ k : ℝ, (∀ x y, k * x - y = 0) ∧ ∀ x y, x^2 + y^2 - 4 * x + 3 = 0) →
  θ = π / 6 ∨ θ = 5 * π / 6 := by
  sorry

end tangent_line_slope_angle_l56_56317


namespace quadrilateral_perimeter_proof_l56_56741

noncomputable def perimeter_quadrilateral (AB BC CD AD : ℝ) : ℝ :=
  AB + BC + CD + AD

theorem quadrilateral_perimeter_proof
  (AB BC CD AD : ℝ)
  (h1 : AB = 15)
  (h2 : BC = 10)
  (h3 : CD = 6)
  (h4 : AB = AD)
  (h5 : AD = Real.sqrt 181)
  : perimeter_quadrilateral AB BC CD AD = 31 + Real.sqrt 181 := by
  unfold perimeter_quadrilateral
  rw [h1, h2, h3, h5]
  sorry

end quadrilateral_perimeter_proof_l56_56741


namespace simplify_fraction_l56_56869

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end simplify_fraction_l56_56869


namespace triangle_inequality_l56_56417

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_l56_56417


namespace positive_difference_even_odd_sums_l56_56168

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l56_56168


namespace positive_difference_even_odd_sums_l56_56179

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l56_56179


namespace solve_for_h_l56_56510

-- Define the given polynomials
def p1 (x : ℝ) : ℝ := 2*x^5 + 4*x^3 - 3*x^2 + x + 7
def p2 (x : ℝ) : ℝ := -x^3 + 2*x^2 - 5*x + 4

-- Define h(x) as the unknown polynomial to solve for
def h (x : ℝ) : ℝ := -2*x^5 - x^3 + 5*x^2 - 6*x - 3

-- The theorem to prove
theorem solve_for_h : 
  (∀ (x : ℝ), p1 x + h x = p2 x) → (∀ (x : ℝ), h x = -2*x^5 - x^3 + 5*x^2 - 6*x - 3) :=
by
  intro h_cond
  sorry

end solve_for_h_l56_56510


namespace prime_divisors_of_factorial_50_l56_56541

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

theorem prime_divisors_of_factorial_50 :
  (nat.filter nat.prime (list.fin_range 51)).length = 15 :=
by
  sorry

end prime_divisors_of_factorial_50_l56_56541


namespace melina_age_l56_56451

theorem melina_age (A M : ℕ) (alma_score : ℕ := 40) 
    (h1 : A + M = 2 * alma_score) 
    (h2 : M = 3 * A) : 
    M = 60 :=
by 
  sorry

end melina_age_l56_56451


namespace number_of_positive_prime_divisors_of_factorial_l56_56547

theorem number_of_positive_prime_divisors_of_factorial :
  {p : ℕ | p.prime ∧ p ≤ 50}.card = 15 := 
sorry

end number_of_positive_prime_divisors_of_factorial_l56_56547


namespace simplify_fraction_90_150_l56_56867

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end simplify_fraction_90_150_l56_56867


namespace baron_munchausen_max_crowd_size_l56_56196

theorem baron_munchausen_max_crowd_size :
  ∃ n : ℕ, (∀ k, (k : ℕ) = n → 
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= n + 1) ∧ 
  (∀ x : ℕ, x > 37 → ¬(∀ k, (k : ℕ) = x →
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= x + 1)) :=
begin
  have h : 37 = 18 + 12 + 7,
  sorry,
end

end baron_munchausen_max_crowd_size_l56_56196


namespace positive_difference_sums_l56_56173

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l56_56173


namespace minimum_area_rectangle_l56_56821

noncomputable def minimum_rectangle_area (a : ℝ) : ℝ :=
  if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
  else if a < 1 / 2 then 1 - 2 * a
  else 0

theorem minimum_area_rectangle (a : ℝ) :
  minimum_rectangle_area a =
    if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
    else if a < 1 / 2 then 1 - 2 * a
    else 0 :=
by
  sorry

end minimum_area_rectangle_l56_56821


namespace widgets_production_l56_56215

variables (A B C : ℝ)
variables (P : ℝ)

-- Conditions provided
def condition1 : Prop := 7 * A + 11 * B = 305
def condition2 : Prop := 8 * A + 22 * C = P

-- The question we need to answer
def question : Prop :=
  ∃ Q : ℝ, Q = 8 * (A + B + C)

theorem widgets_production (h1 : condition1 A B) (h2 : condition2 A C P) :
  question A B C :=
sorry

end widgets_production_l56_56215


namespace senate_subcommittee_l56_56774

/-- 
Proof of the number of ways to form a Senate subcommittee consisting of 7 Republicans
and 2 Democrats from the available 12 Republicans and 6 Democrats.
-/
theorem senate_subcommittee (R D : ℕ) (choose_R choose_D : ℕ) (hR : R = 12) (hD : D = 6) 
  (h_choose_R : choose_R = 7) (h_choose_D : choose_D = 2) : 
  (Nat.choose R choose_R) * (Nat.choose D choose_D) = 11880 := by
  sorry

end senate_subcommittee_l56_56774


namespace rice_mixing_ratio_l56_56619

-- Definitions based on conditions
def rice_1_price : ℝ := 6
def rice_2_price : ℝ := 8.75
def mixture_price : ℝ := 7.50

-- Proof of the required ratio
theorem rice_mixing_ratio (x y : ℝ) (h : (rice_1_price * x + rice_2_price * y) / (x + y) = mixture_price) :
  y / x = 6 / 5 :=
by 
  sorry

end rice_mixing_ratio_l56_56619


namespace subtract_one_from_solution_l56_56334

theorem subtract_one_from_solution (x : ℝ) (h : 15 * x = 45) : (x - 1) = 2 := 
by {
  sorry
}

end subtract_one_from_solution_l56_56334


namespace prob_exactly_two_approve_l56_56784

def voter_approval_pmf (p : ℝ) : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.binomial 4 p

theorem prob_exactly_two_approve (p : ℝ) (h_p : p = 0.6) :
  voter_approval_pmf p 2 = 0.3456 := 
by
  have hp : p = 0.6 := h_p
  sorry

end prob_exactly_two_approve_l56_56784


namespace positive_difference_l56_56153

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l56_56153


namespace line_through_circle_center_l56_56371

theorem line_through_circle_center
  (C : ℝ × ℝ)
  (hC : C = (-1, 0))
  (hCircle : ∀ (x y : ℝ), x^2 + 2 * x + y^2 = 0 → (x, y) = (-1, 0))
  (hPerpendicular : ∀ (m₁ m₂ : ℝ), (m₁ * m₂ = -1) → m₁ = -1 → m₂ = 1)
  (line_eq : ∀ (x y : ℝ), y = x + 1)
  : ∀ (x y : ℝ), x - y + 1 = 0 :=
sorry

end line_through_circle_center_l56_56371


namespace positive_difference_prob_3_and_4_heads_l56_56754

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_prob_3_and_4_heads (p : ℚ) (hp : p = 1/2) (n : ℕ) (hn : n = 4) :
  let p1 := binomial_prob n 3 p in
  let p2 := binomial_prob n 4 p in
  p1 - p2 = 3/16 :=
by
  sorry

end positive_difference_prob_3_and_4_heads_l56_56754


namespace points_on_opposite_sides_l56_56827

theorem points_on_opposite_sides (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by sorry

end points_on_opposite_sides_l56_56827


namespace matilda_father_chocolates_left_l56_56709

-- definitions for each condition
def initial_chocolates : ℕ := 20
def persons : ℕ := 5
def chocolates_per_person := initial_chocolates / persons
def half_chocolates_per_person := chocolates_per_person / 2
def total_given_to_father := half_chocolates_per_person * persons
def chocolates_given_to_mother := 3
def chocolates_eaten_by_father := 2

-- statement to prove
theorem matilda_father_chocolates_left :
  total_given_to_father - chocolates_given_to_mother - chocolates_eaten_by_father = 5 :=
by
  sorry

end matilda_father_chocolates_left_l56_56709


namespace exists_x0_condition_l56_56095

theorem exists_x0_condition
  (f : ℝ → ℝ)
  (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_diff : ∀ x ∈ Set.Ioo 0 1, DifferentiableAt ℝ f x)
  (h_f0 : f 0 = 1)
  (h_f1 : f 1 = 0) :
  ∃ x₀ ∈ Set.Ioo 0 1, |fderiv ℝ f x₀| ≥ 2018 * (f x₀) ^ 2018 := by
  sorry

end exists_x0_condition_l56_56095


namespace part_a_part_a_rev_l56_56416

variable (x y : ℝ)

theorem part_a (hx : x > 0) (hy : y > 0) : x + y > |x - y| :=
sorry

theorem part_a_rev (h : x + y > |x - y|) : x > 0 ∧ y > 0 :=
sorry

end part_a_part_a_rev_l56_56416


namespace profit_function_expression_l56_56623

def dailySalesVolume (x : ℝ) : ℝ := 300 + 3 * (99 - x)

def profitPerItem (x : ℝ) : ℝ := x - 50

def dailyProfit (x : ℝ) : ℝ := (x - 50) * (300 + 3 * (99 - x))

theorem profit_function_expression (x : ℝ) :
  dailyProfit x = (x - 50) * dailySalesVolume x :=
by sorry

end profit_function_expression_l56_56623


namespace non_union_employees_women_percent_l56_56618

-- Define the conditions
variables (total_employees men_percent women_percent unionized_percent unionized_men_percent : ℕ)
variables (total_men total_women total_unionized total_non_unionized unionized_men non_unionized_men non_unionized_women : ℕ)

axiom condition1 : men_percent = 52
axiom condition2 : unionized_percent = 60
axiom condition3 : unionized_men_percent = 70

axiom calc1 : total_employees = 100
axiom calc2 : total_men = total_employees * men_percent / 100
axiom calc3 : total_women = total_employees - total_men
axiom calc4 : total_unionized = total_employees * unionized_percent / 100
axiom calc5 : unionized_men = total_unionized * unionized_men_percent / 100
axiom calc6 : non_unionized_men = total_men - unionized_men
axiom calc7 : total_non_unionized = total_employees - total_unionized
axiom calc8 : non_unionized_women = total_non_unionized - non_unionized_men

-- Define the proof statement
theorem non_union_employees_women_percent : 
  (non_unionized_women / total_non_unionized) * 100 = 75 :=
by 
  sorry

end non_union_employees_women_percent_l56_56618


namespace number_of_groups_is_correct_l56_56333

-- Define the number of students
def number_of_students : ℕ := 16

-- Define the group size
def group_size : ℕ := 4

-- Define the expected number of groups
def expected_number_of_groups : ℕ := 4

-- Prove the expected number of groups when grouping students into groups of four
theorem number_of_groups_is_correct :
  number_of_students / group_size = expected_number_of_groups := by
  sorry

end number_of_groups_is_correct_l56_56333


namespace positive_difference_even_odd_sums_l56_56165

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l56_56165


namespace third_chest_coin_difference_l56_56320

variable (g1 g2 g3 s1 s2 s3 : ℕ)

-- Conditions
axiom h1 : g1 + g2 + g3 = 40
axiom h2 : s1 + s2 + s3 = 40
axiom h3 : g1 = s1 + 7
axiom h4 : g2 = s2 + 15

-- Goal
theorem third_chest_coin_difference : s3 = g3 + 22 :=
sorry

end third_chest_coin_difference_l56_56320


namespace seeds_total_l56_56508

variable (seedsInBigGarden : Nat)
variable (numSmallGardens : Nat)
variable (seedsPerSmallGarden : Nat)

theorem seeds_total (h1 : seedsInBigGarden = 36) (h2 : numSmallGardens = 3) (h3 : seedsPerSmallGarden = 2) : 
  seedsInBigGarden + numSmallGardens * seedsPerSmallGarden = 42 := by
  sorry

end seeds_total_l56_56508


namespace number_of_prime_divisors_of_50_factorial_l56_56537

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to (n : ℕ) : List ℕ :=
  List.filter is_prime (List.range (n + 1))

theorem number_of_prime_divisors_of_50_factorial :
  (primes_up_to 50).length = 15 :=
by
  sorry

end number_of_prime_divisors_of_50_factorial_l56_56537


namespace nth_term_sequence_sum_first_n_terms_l56_56473

def a_n (n : ℕ) : ℕ :=
  (2 * n - 1) * (2 * n + 2)

def S_n (n : ℕ) : ℚ :=
  4 * (n * (n + 1) * (2 * n + 1)) / 6 + n * (n + 1) - 2 * n

theorem nth_term_sequence (n : ℕ) : a_n n = 4 * n^2 + 2 * n - 2 :=
  sorry

theorem sum_first_n_terms (n : ℕ) : S_n n = (4 * n^3 + 9 * n^2 - n) / 3 :=
  sorry

end nth_term_sequence_sum_first_n_terms_l56_56473


namespace replace_floor_cost_l56_56455

-- Define the conditions
def floor_removal_cost : ℝ := 50
def new_floor_cost_per_sqft : ℝ := 1.25
def room_length : ℝ := 8
def room_width : ℝ := 7

-- Define the area of the room
def room_area : ℝ := room_length * room_width

-- Define the cost of the new floor
def new_floor_cost : ℝ := room_area * new_floor_cost_per_sqft

-- Define the total cost to replace the floor
def total_cost : ℝ := floor_removal_cost + new_floor_cost

-- State the proof problem
theorem replace_floor_cost : total_cost = 120 := by
  sorry

end replace_floor_cost_l56_56455


namespace line_through_point_inequality_l56_56271

theorem line_through_point_inequality
  (a b θ : ℝ)
  (h : (b * Real.cos θ + a * Real.sin θ = a * b)) :
  1 / a^2 + 1 / b^2 ≥ 1 := 
  sorry

end line_through_point_inequality_l56_56271


namespace picture_distance_l56_56323

theorem picture_distance (w t s p d : ℕ) (h1 : w = 25) (h2 : t = 2) (h3 : s = 1) (h4 : 2 * p + s = t + s + t) 
  (h5 : w = 2 * d + p) : d = 10 :=
by
  sorry

end picture_distance_l56_56323


namespace find_line_equation_proj_origin_l56_56104

theorem find_line_equation_proj_origin (P : ℝ × ℝ) (hP : P = (-2, 1)) :
    ∃ (a b c : ℝ), a * 2 + b * (-1) + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 5 := 
by
  sorry

end find_line_equation_proj_origin_l56_56104


namespace value_of_a1_plus_a10_l56_56522

noncomputable def geometric_sequence {α : Type*} [Field α] (a : ℕ → α) :=
  ∃ q : α, ∀ n : ℕ, a (n + 1) = a n * q

theorem value_of_a1_plus_a10 (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : a 4 + a 7 = 2) 
  (h3 : a 5 * a 6 = -8) 
  : a 1 + a 10 = -7 := 
by
  sorry

end value_of_a1_plus_a10_l56_56522


namespace find_m_l56_56669

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9

def symmetric_line (x y m : ℝ) : Prop := x + m * y + 4 = 0

theorem find_m (m : ℝ) (h1 : circle_equation (-1) 3) (h2 : symmetric_line (-1) 3 m) : m = -1 := by
  sorry

end find_m_l56_56669


namespace replace_floor_cost_l56_56456

def cost_of_removal := 50
def cost_per_sqft := 1.25
def room_length := 8
def room_width := 7

def total_cost_to_replace_floor : ℝ :=
  cost_of_removal + (cost_per_sqft * (room_length * room_width))

theorem replace_floor_cost :
  total_cost_to_replace_floor = 120 :=
by
  sorry

end replace_floor_cost_l56_56456


namespace pizza_slices_with_both_l56_56207

theorem pizza_slices_with_both (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24) (h_pepperoni : pepperoni_slices = 15) (h_mushrooms : mushroom_slices = 14) :
  ∃ n, n = 5 ∧ total_slices = pepperoni_slices + mushroom_slices - n := 
by
  use 5
  sorry

end pizza_slices_with_both_l56_56207


namespace solve_sin_cos_eqn_l56_56369

theorem solve_sin_cos_eqn (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x + Real.cos x = 1) :
  x = 0 ∨ x = Real.pi / 2 :=
sorry

end solve_sin_cos_eqn_l56_56369


namespace minimum_familiar_pairs_l56_56203

open Finset

-- Define the set of students and the relationship of familiarity
variable (students : Finset ℕ)
variable (n : ℕ := 175)
variable (familiar : ℕ → ℕ → Prop)

-- Assumption: students set has 175 members
axiom student_count : students.card = n

-- Assumption: familiarity is symmetric
axiom familiar_symm (a b : ℕ) : familiar a b → familiar b a

-- Assumption: familiarity within any group of six
axiom familiar_in_groups_of_six (s : Finset ℕ) (h₁ : s.card = 6) :
  ∃ t₁ t₂ : Finset ℕ, t₁.card = 3 ∧ t₂.card = 3 ∧ (∀ x ∈ t₁, ∀ y ∈ t₁, x ≠ y → familiar x y) ∧
  (∀ x ∈ t₂, ∀ y ∈ t₂, x ≠ y → familiar x y) ∧ t₁ ∪ t₂ = s ∧ t₁ ∩ t₂ = ∅

-- Theorem: minimum number of familiar pairs
theorem minimum_familiar_pairs :
  ∃ k : ℕ, (∑ a in students, (students.filter (familiar a)).card) / 2 ≥ 15050 :=
sorry

end minimum_familiar_pairs_l56_56203


namespace positive_difference_of_probabilities_l56_56753

theorem positive_difference_of_probabilities :
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1 in
  let p2 := (1 / 2) ^ 4 in
  abs (p1 - p2) = 7 / 16 :=
by
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1
  let p2 := (1 / 2) ^ 4
  have p1_calc: p1 = 1 / 2
  { rw [Nat.choose, _root_.choose, binomial_eq, algebraic_ident]
    sorry  -- The detailed proof of p1 = 1 / 2 will go here
  }
  have p2_calc: p2 = 1 / 16
  { rw [(1/2)^4]
    sorry  -- The detailed proof of p2 = 1 / 16 will go here
  }
  have diff_calc: abs (p1 - p2) = 7 / 16
  { rw [p1_calc, p2_calc]
    sorry  -- The detailed proof of abs (p1 - p2) = 7 / 16 will go here
  }
  exact diff_calc

end positive_difference_of_probabilities_l56_56753


namespace cos_832_eq_cos_l56_56815

theorem cos_832_eq_cos (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * Real.pi / 180) = Real.cos (832 * Real.pi / 180)) : n = 112 := 
  sorry

end cos_832_eq_cos_l56_56815


namespace sum_of_money_l56_56631

theorem sum_of_money (A B C : ℝ) (hB : B = 0.65 * A) (hC : C = 0.40 * A) (hC_val : C = 56) :
  A + B + C = 287 :=
by {
  sorry
}

end sum_of_money_l56_56631


namespace points_lie_on_hyperbola_l56_56964

noncomputable
def point_on_hyperbola (t : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ 
    (2 * t * x - 3 * y - 4 * t = 0 ∧ x - 3 * t * y + 4 = 0) }

theorem points_lie_on_hyperbola : 
  ∀ t : ℝ, ∀ x y : ℝ, (2 * t * x - 3 * y - 4 * t = 0 ∧ x - 3 * t * y + 4 = 0) → (x^2 / 16) - (y^2 / 1) = 1 :=
by 
  intro t x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end points_lie_on_hyperbola_l56_56964


namespace rectangle_perimeter_is_28_l56_56954

-- Define the variables and conditions
variables (h w : ℝ)

-- Problem conditions
def rectangle_area (h w : ℝ) : Prop := h * w = 40
def width_greater_than_twice_height (h w : ℝ) : Prop := w > 2 * h
def parallelogram_area (h w : ℝ) : Prop := h * (w - h) = 24

-- The theorem stating the perimeter of the rectangle given the conditions
theorem rectangle_perimeter_is_28 (h w : ℝ) 
  (H1 : rectangle_area h w) 
  (H2 : width_greater_than_twice_height h w) 
  (H3 : parallelogram_area h w) :
  2 * h + 2 * w = 28 :=
sorry

end rectangle_perimeter_is_28_l56_56954


namespace angle_sum_l56_56208

-- Define the angles in the isosceles triangles
def angle_BAC := 40
def angle_EDF := 50

-- Using the property of isosceles triangles to calculate other angles
def angle_ABC := (180 - angle_BAC) / 2
def angle_DEF := (180 - angle_EDF) / 2

-- Since AD is parallel to CE, angles DAC and ACB are equal as are ADE and DEF
def angle_DAC := angle_ABC
def angle_ADE := angle_DEF

-- The theorem to be proven
theorem angle_sum :
  angle_DAC + angle_ADE = 135 :=
by
  sorry

end angle_sum_l56_56208


namespace bakery_storage_l56_56920

theorem bakery_storage (S F B : ℕ) (h1 : S * 8 = 3 * F) (h2 : F * 1 = 10 * B) (h3 : F * 1 = 8 * (B + 60)) : S = 900 :=
by
  -- We would normally put the proof steps here, but since it's specified to include only the statement
  sorry

end bakery_storage_l56_56920


namespace find_x_l56_56674

-- We are given points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 2)

-- Vector a is (2x + 3, x^2 - 4)
def vec_a (x : ℝ) : ℝ × ℝ := (2 * x + 3, x^2 - 4)

-- Vector AB is calculated as
def vec_AB : ℝ × ℝ := (3 - 1, 2 - 2)

-- Define the condition that vec_a and vec_AB form 0° angle
def forms_zero_angle (u v : ℝ × ℝ) : Prop := (u.1 * v.2 - u.2 * v.1) = 0 ∧ (u.1 = v.1 ∧ v.2 = 0)

-- The proof statement
theorem find_x (x : ℝ) (h₁ : forms_zero_angle (vec_a x) vec_AB) : x = 2 :=
by
  sorry

end find_x_l56_56674


namespace minimum_value_l56_56534

variable (a b : ℝ)
variable (ab_nonzero : a ≠ 0 ∧ b ≠ 0)
variable (circle1 : ∀ x y, x^2 + y^2 + 2 * a * x + a^2 - 9 = 0)
variable (circle2 : ∀ x y, x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0)
variable (centers_distance : a^2 + 4 * b^2 = 16)

theorem minimum_value :
  (4 / a^2 + 1 / b^2) = 1 := sorry

end minimum_value_l56_56534


namespace minimum_familiar_pairs_l56_56204

theorem minimum_familiar_pairs (n : ℕ) (students : Finset (Fin n)) 
  (familiar : Finset (Fin n × Fin n))
  (h_n : n = 175)
  (h_condition : ∀ (s : Finset (Fin n)), s.card = 6 → 
    ∃ (s1 s2 : Finset (Fin n)), s1 ∪ s2 = s ∧ s1.card = 3 ∧ s2.card = 3 ∧ 
    ∀ x ∈ s1, ∀ y ∈ s1, (x ≠ y → (x, y) ∈ familiar) ∧
    ∀ x ∈ s2, ∀ y ∈ s2, (x ≠ y → (x, y) ∈ familiar)) :
  ∃ m : ℕ, m = 15050 ∧ ∀ p : ℕ, (∃ g : Finset (Fin n × Fin n), g.card = p) → p ≥ m := 
sorry

end minimum_familiar_pairs_l56_56204


namespace non_equivalent_paintings_wheel_l56_56214

theorem non_equivalent_paintings_wheel :
  let num_sections := 7
  let num_colors := 2
  let total_paintings := num_colors ^ num_sections
  let single_color_cases := 2
  let non_single_color_paintings := total_paintings - single_color_cases
  let equivalent_rotation_count := num_sections
  (non_single_color_paintings / equivalent_rotation_count) + single_color_cases = 20 :=
by
  let num_sections := 7
  let num_colors := 2
  let total_paintings := num_colors ^ num_sections
  let single_color_cases := 2
  let non_single_color_paintings := total_paintings - single_color_cases
  let equivalent_rotation_count := num_sections
  have h1 := (non_single_color_paintings / equivalent_rotation_count) + single_color_cases
  sorry

end non_equivalent_paintings_wheel_l56_56214


namespace remainder_when_divided_l56_56968

/-- Given integers T, E, N, S, E', N', S'. When T is divided by E, 
the quotient is N and the remainder is S. When N is divided by E', 
the quotient is N' and the remainder is S'. Prove that the remainder 
when T is divided by E + E' is ES' + S. -/
theorem remainder_when_divided (T E N S E' N' S' : ℤ) (h1 : T = N * E + S) (h2 : N = N' * E' + S') :
  (T % (E + E')) = (E * S' + S) :=
by
  sorry

end remainder_when_divided_l56_56968


namespace find_a_from_conditions_l56_56812

theorem find_a_from_conditions
  (a b : ℤ)
  (h₁ : 2584 * a + 1597 * b = 0)
  (h₂ : 1597 * a + 987 * b = -1) :
  a = 1597 :=
by sorry

end find_a_from_conditions_l56_56812


namespace oblong_perimeter_182_l56_56356

variables (l w : ℕ) (x : ℤ)

def is_oblong (l w : ℕ) : Prop :=
l * w = 4624 ∧ l = 4 * x ∧ w = 3 * x

theorem oblong_perimeter_182 (l w x : ℕ) (hlw : is_oblong l w x) : 
  2 * l + 2 * w = 182 :=
by
  sorry

end oblong_perimeter_182_l56_56356


namespace fraction_to_decimal_l56_56026

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 :=
by
  sorry

end fraction_to_decimal_l56_56026


namespace find_line_l_l56_56311

theorem find_line_l :
  ∃ l : ℝ × ℝ → Prop,
    (∀ (B : ℝ × ℝ), (2 * B.1 + B.2 - 8 = 0) → 
      (∀ A : ℝ × ℝ, (A.1 = -B.1 ∧ A.2 = 2 * B.1 - 6 ) → 
        (A.1 - 3 * A.2 + 10 = 0) → 
          B.1 = 4 ∧ B.2 = 0 ∧ ∀ p : ℝ × ℝ, B.1 * p.1 + 4 * p.2 - 4 = 0)) := 
  sorry

end find_line_l_l56_56311


namespace point_in_fourth_quadrant_coords_l56_56997

theorem point_in_fourth_quadrant_coords 
  (P : ℝ × ℝ)
  (h1 : P.2 < 0)
  (h2 : abs P.2 = 2)
  (h3 : P.1 > 0)
  (h4 : abs P.1 = 5) :
  P = (5, -2) :=
sorry

end point_in_fourth_quadrant_coords_l56_56997


namespace molecular_weight_BaCl2_l56_56330

theorem molecular_weight_BaCl2 (mw8 : ℝ) (n : ℝ) (h : mw8 = 1656) : (mw8 / n = 207) ↔ n = 8 := 
by
  sorry

end molecular_weight_BaCl2_l56_56330


namespace express_x_in_terms_of_y_l56_56255

variable {x y : ℝ}

theorem express_x_in_terms_of_y (h : 3 * x - 4 * y = 6) : x = (6 + 4 * y) / 3 := 
sorry

end express_x_in_terms_of_y_l56_56255


namespace final_weight_is_correct_l56_56352

-- Define the various weights after each week
def initial_weight : ℝ := 180
def first_week_removed : ℝ := 0.28 * initial_weight
def first_week_remaining : ℝ := initial_weight - first_week_removed
def second_week_removed : ℝ := 0.18 * first_week_remaining
def second_week_remaining : ℝ := first_week_remaining - second_week_removed
def third_week_removed : ℝ := 0.20 * second_week_remaining
def final_weight : ℝ := second_week_remaining - third_week_removed

-- State the theorem to prove the final weight equals 85.0176 kg
theorem final_weight_is_correct : final_weight = 85.0176 := 
by 
  sorry

end final_weight_is_correct_l56_56352


namespace work_rate_ab_l56_56340

variables (A B C : ℝ)

-- Defining the work rates as per the conditions
def work_rate_bc := 1 / 6 -- (b and c together in 6 days)
def work_rate_ca := 1 / 3 -- (c and a together in 3 days)
def work_rate_c := 1 / 8 -- (c alone in 8 days)

-- The main theorem that proves a and b together can complete the work in 4 days,
-- based on the above conditions.
theorem work_rate_ab : 
  (B + C = work_rate_bc) ∧ (C + A = work_rate_ca) ∧ (C = work_rate_c) 
  → (A + B = 1 / 4) :=
by sorry

end work_rate_ab_l56_56340


namespace not_partition_1985_1987_partition_1987_1989_l56_56562

-- Define the number of squares in an L-shape
def squares_in_lshape : ℕ := 3

-- Question 1: Can 1985 x 1987 be partitioned into L-shapes?
def partition_1985_1987 (m n : ℕ) (L_shape_size : ℕ) : Prop :=
  ∃ k : ℕ, m * n = k * L_shape_size ∧ (m % L_shape_size = 0 ∨ n % L_shape_size = 0)

theorem not_partition_1985_1987 :
  ¬ partition_1985_1987 1985 1987 squares_in_lshape :=
by {
  -- Proof omitted
  sorry
}

-- Question 2: Can 1987 x 1989 be partitioned into L-shapes?
theorem partition_1987_1989 :
  partition_1985_1987 1987 1989 squares_in_lshape :=
by {
  -- Proof omitted
  sorry
}

end not_partition_1985_1987_partition_1987_1989_l56_56562


namespace principal_amount_l56_56919

variable (P : ℝ)

/-- Prove the principal amount P given that the simple interest at 4% for 5 years is Rs. 2400 less than the principal --/
theorem principal_amount : 
  (4/100) * P * 5 = P - 2400 → 
  P = 3000 := 
by 
  sorry

end principal_amount_l56_56919


namespace max_area_of_sector_l56_56676

theorem max_area_of_sector (r l : ℝ) (h₁ : 2 * r + l = 12) : 
  (1 / 2) * l * r ≤ 9 :=
by sorry

end max_area_of_sector_l56_56676


namespace simplify_fraction_l56_56876

theorem simplify_fraction (num denom : ℕ) (h_num : num = 90) (h_denom : denom = 150) : 
  num / denom = 3 / 5 := by
  rw [h_num, h_denom]
  norm_num
  sorry

end simplify_fraction_l56_56876


namespace energy_savings_l56_56074

theorem energy_savings (x y : ℝ) 
  (h1 : x = y + 27) 
  (h2 : x + 2.1 * y = 405) :
  x = 149 ∧ y = 122 :=
by
  sorry

end energy_savings_l56_56074


namespace initial_girls_is_11_l56_56930

-- Definitions of initial parameters and transformations
def initially_girls_percent : ℝ := 0.35
def final_girls_percent : ℝ := 0.25
def three : ℝ := 3

-- 35% of the initial total is girls
def initially_girls (p : ℝ) : ℝ := initially_girls_percent * p
-- After three girls leave and three boys join, the count of girls
def final_girls (p : ℝ) : ℝ := initially_girls p - three

-- Using the condition that after the change, 25% are girls
def proof_problem : Prop := ∀ (p : ℝ), 
  (final_girls p) / p = final_girls_percent →
  (0.1 * p) = 3 → 
  initially_girls p = 11

-- The statement of the theorem to be proved in Lean 4
theorem initial_girls_is_11 : proof_problem := sorry

end initial_girls_is_11_l56_56930


namespace decagon_interior_angle_measure_l56_56602

-- Define the type for a regular polygon
structure RegularPolygon (n : Nat) :=
  (interior_angle_sum : Nat := (n - 2) * 180)
  (side_count : Nat := n)
  (regularity : Prop := True)  -- All angles are equal

-- Define the degree measure of an interior angle of a regular polygon
def interiorAngle (p : RegularPolygon 10) : Nat :=
  (p.interior_angle_sum) / p.side_count

-- The theorem to be proved
theorem decagon_interior_angle_measure : 
  ∀ (p : RegularPolygon 10), interiorAngle p = 144 := by
  -- The proof will be here, but for now, we use sorry
  sorry

end decagon_interior_angle_measure_l56_56602


namespace sarahs_loan_amount_l56_56436

theorem sarahs_loan_amount 
  (down_payment : ℕ := 10000)
  (monthly_payment : ℕ := 600)
  (repayment_years : ℕ := 5)
  (interest_rate : ℚ := 0) : down_payment + (monthly_payment * (12 * repayment_years)) = 46000 :=
by
  sorry

end sarahs_loan_amount_l56_56436


namespace value_of_a_l56_56844

-- Define the three lines as predicates
def line1 (x y : ℝ) : Prop := x + y = 1
def line2 (x y : ℝ) : Prop := x - y = 1
def line3 (a x y : ℝ) : Prop := a * x + y = 1

-- Define the condition that the lines do not form a triangle
def lines_do_not_form_triangle (a x y : ℝ) : Prop :=
  (∀ x y, line1 x y → ¬line3 a x y) ∨
  (∀ x y, line2 x y → ¬line3 a x y) ∨
  (a = 1)

theorem value_of_a (a : ℝ) :
  (¬ ∃ x y, line1 x y ∧ line2 x y ∧ line3 a x y) →
  lines_do_not_form_triangle a 1 0 →
  a = -1 :=
by
  intro h1 h2
  sorry

end value_of_a_l56_56844


namespace volume_of_intersection_of_two_perpendicular_cylinders_l56_56044

theorem volume_of_intersection_of_two_perpendicular_cylinders (R : ℝ) : 
  ∃ V : ℝ, V = (16 / 3) * R^3 := 
sorry

end volume_of_intersection_of_two_perpendicular_cylinders_l56_56044


namespace triangle_inequality_third_side_l56_56826

theorem triangle_inequality_third_side (a : ℝ) (h1 : 3 + a > 7) (h2 : 7 + a > 3) (h3 : 3 + 7 > a) : 
  4 < a ∧ a < 10 :=
by sorry

end triangle_inequality_third_side_l56_56826


namespace hyperbola_distance_to_foci_l56_56842

theorem hyperbola_distance_to_foci
  (E : ∀ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1)
  (F1 F2 : ℝ)
  (P : ℝ)
  (dist_PF1 : P = 5)
  (a : ℝ)
  (ha : a = 3): 
  |P - F2| = 11 :=
by
  sorry

end hyperbola_distance_to_foci_l56_56842


namespace find_a_l56_56390

-- Defining the problem conditions
def rational_eq (x a : ℝ) :=
  x / (x - 3) - 2 * a / (x - 3) = 2

def extraneous_root (x : ℝ) : Prop :=
  x = 3

-- Theorem: Given the conditions, prove that a = 3 / 2
theorem find_a (a : ℝ) : (∃ x, extraneous_root x ∧ rational_eq x a) → a = 3 / 2 :=
  by
    sorry

end find_a_l56_56390


namespace positive_difference_eq_250_l56_56118

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l56_56118


namespace sin_75_is_option_D_l56_56374

noncomputable def sin_75 : ℝ := Real.sin (75 * Real.pi / 180)

noncomputable def option_D : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4

theorem sin_75_is_option_D : sin_75 = option_D :=
by
  sorry

end sin_75_is_option_D_l56_56374


namespace percentage_increase_area_rectangle_l56_56006

theorem percentage_increase_area_rectangle (L W : ℝ) :
  let new_length := 1.20 * L
  let new_width := 1.20 * W
  let original_area := L * W
  let new_area := new_length * new_width
  let percentage_increase := ((new_area - original_area) / original_area) * 100
  percentage_increase = 44 := by
  sorry

end percentage_increase_area_rectangle_l56_56006


namespace slope_of_line_l56_56606

theorem slope_of_line : 
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 1) ∧ (y1 = 3) ∧ (x2 = 7) ∧ (y2 = -9)
  → (y2 - y1) / (x2 - x1) = -2 := by
  sorry

end slope_of_line_l56_56606


namespace min_total_cost_of_container_l56_56350

-- Definitions from conditions
def container_volume := 4 -- m^3
def container_height := 1 -- m
def cost_per_square_meter_base : ℝ := 20
def cost_per_square_meter_sides : ℝ := 10

-- Proving the minimum total cost
theorem min_total_cost_of_container :
  ∃ (a b : ℝ), a * b = container_volume ∧
                (20 * (a + b) + 20 * (a * b)) = 160 :=
by
  sorry

end min_total_cost_of_container_l56_56350


namespace simplify_fraction_l56_56875

theorem simplify_fraction (num denom : ℕ) (h_num : num = 90) (h_denom : denom = 150) : 
  num / denom = 3 / 5 := by
  rw [h_num, h_denom]
  norm_num
  sorry

end simplify_fraction_l56_56875


namespace remaining_milk_and_coffee_l56_56926

/-- 
Given:
1. A cup initially contains 1 glass of coffee.
2. A quarter glass of milk is added to the cup.
3. The mixture is thoroughly stirred.
4. One glass of the mixture is poured back.

Prove:
The remaining content in the cup is 1/5 glass of milk and 4/5 glass of coffee. 
--/
theorem remaining_milk_and_coffee :
  let coffee_initial := 1  -- initial volume of coffee
  let milk_added := 1 / 4  -- volume of milk added
  let total_volume := coffee_initial + milk_added  -- total volume after mixing = 5/4 glasses
  let milk_fraction := milk_added / total_volume  -- fraction of milk in the mixture = 1/5
  let coffee_fraction := coffee_initial / total_volume  -- fraction of coffee in the mixture = 4/5
  let volume_poured := 1 / 4  -- volume of mixture poured out
  let milk_poured := (milk_fraction * volume_poured : ℝ)  -- volume of milk poured out = 1/20 glass
  let coffee_poured := (coffee_fraction * volume_poured : ℝ)  -- volume of coffee poured out = 1/5 glass
  let remaining_milk := milk_added - milk_poured  -- remaining volume of milk = 1/5 glass
  let remaining_coffee := coffee_initial - coffee_poured  -- remaining volume of coffee = 4/5 glass
  remaining_milk = 1 / 5 ∧ remaining_coffee = 4 / 5 :=
by
  sorry

end remaining_milk_and_coffee_l56_56926


namespace inequality_solution_set_l56_56242

noncomputable def solution_set : Set ℝ := { x : ℝ | x > 5 ∨ x < -2 }

theorem inequality_solution_set (x : ℝ) :
  x^2 - 3 * x - 10 > 0 ↔ x > 5 ∨ x < -2 :=
by
  sorry

end inequality_solution_set_l56_56242


namespace extra_flour_l56_56086

-- Define the conditions
def recipe_flour : ℝ := 7.0
def mary_flour : ℝ := 9.0

-- Prove the number of extra cups of flour Mary puts in
theorem extra_flour : mary_flour - recipe_flour = 2 :=
by
  sorry

end extra_flour_l56_56086


namespace maximize_profit_l56_56634

-- Definitions from the conditions
def cost_price : ℝ := 16
def initial_selling_price : ℝ := 20
def initial_sales_volume : ℝ := 80
def price_decrease_per_step : ℝ := 0.5
def sales_increase_per_step : ℝ := 20

def functional_relationship (x : ℝ) : ℝ := -40 * x + 880

-- The main theorem we need to prove
theorem maximize_profit :
  (∀ x, 16 ≤ x → x ≤ 20 → functional_relationship x = -40 * x + 880) ∧
  (∃ x, 16 ≤ x ∧ x ≤ 20 ∧ (∀ y, 16 ≤ y → y ≤ 20 → 
    ((-40 * x + 880) * (x - cost_price) ≥ (-40 * y + 880) * (y - cost_price)) ∧
    (-40 * x + 880) * (x - cost_price) = 360 ∧ x = 19)) :=
by
  sorry

end maximize_profit_l56_56634


namespace positive_difference_of_sums_l56_56136

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l56_56136


namespace P_plus_Q_is_26_l56_56082

theorem P_plus_Q_is_26 (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (P / (x - 3) + Q * (x + 2) = (-2 * x^2 + 8 * x + 34) / (x - 3))) : 
  P + Q = 26 :=
sorry

end P_plus_Q_is_26_l56_56082


namespace find_a_l56_56810

theorem find_a (a b : ℤ) (h : ∀ x, x^2 - x - 1 = 0 → ax^18 + bx^17 + 1 = 0) : a = 1597 :=
sorry

end find_a_l56_56810


namespace function_domain_l56_56726

noncomputable def domain_function (x : ℝ) : Prop :=
  x > 0 ∧ (Real.log x / Real.log 2)^2 - 1 > 0

theorem function_domain :
  { x : ℝ | domain_function x } = { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end function_domain_l56_56726


namespace z_is_233_percent_greater_than_w_l56_56991

theorem z_is_233_percent_greater_than_w
  (w e x y z : ℝ)
  (h1 : w = 0.5 * e)
  (h2 : e = 0.4 * x)
  (h3 : x = 0.3 * y)
  (h4 : z = 0.2 * y) :
  z = 2.3333 * w :=
by
  sorry

end z_is_233_percent_greater_than_w_l56_56991


namespace conic_section_hyperbola_l56_56233

theorem conic_section_hyperbola (x y : ℝ) :
  (x - 3) ^ 2 = 9 * (y + 2) ^ 2 - 81 → conic_section := by
  sorry

end conic_section_hyperbola_l56_56233


namespace seating_arrangements_correct_l56_56572

-- Conditions
def num_children : ℕ := 3
def num_front_seats : ℕ := 2
def num_back_seats : ℕ := 3
def driver_choices : ℕ := 2

-- Function to calculate the number of arrangements
noncomputable def seating_arrangements (children : ℕ) (front_seats : ℕ) (back_seats : ℕ) (driver_choices : ℕ) : ℕ :=
  driver_choices * (children + 1) * (back_seats.factorial)

-- Problem Statement
theorem seating_arrangements_correct : 
  seating_arrangements num_children num_front_seats num_back_seats driver_choices = 48 :=
by
  -- Translate conditions to computation
  have h1: num_children = 3 := rfl
  have h2: num_front_seats = 2 := rfl
  have h3: num_back_seats = 3 := rfl
  have h4: driver_choices = 2 := rfl
  sorry

end seating_arrangements_correct_l56_56572


namespace positive_difference_sums_even_odd_l56_56143

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l56_56143


namespace amount_with_r_l56_56620

theorem amount_with_r (p q r : ℝ) (h₁ : p + q + r = 7000) (h₂ : r = (2 / 3) * (p + q)) : r = 2800 :=
  sorry

end amount_with_r_l56_56620


namespace positive_difference_of_sums_l56_56134

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l56_56134


namespace solving_inequality_l56_56890

theorem solving_inequality (x : ℝ) : 
  (x > 2 ∨ x < -2 ∨ (-1 < x ∧ x < 1)) ↔ ((x^2 - 4) / (x^2 - 1) > 0) :=
by 
  sorry

end solving_inequality_l56_56890


namespace opposite_of_six_is_negative_six_l56_56313

theorem opposite_of_six_is_negative_six : -6 = -6 :=
by
  sorry

end opposite_of_six_is_negative_six_l56_56313


namespace bags_on_monday_l56_56737

/-- Define the problem conditions -/
def t : Nat := 8  -- total number of bags
def f : Nat := 4  -- number of bags found the next day

-- Define the statement to be proven
theorem bags_on_monday : t - f = 4 := by
  -- Sorry to skip the proof
  sorry

end bags_on_monday_l56_56737


namespace correct_operation_l56_56915

variables {a b : ℝ}

theorem correct_operation : (5 * a * b - 6 * a * b = -1 * a * b) := by
  sorry

end correct_operation_l56_56915


namespace vertical_asymptotes_l56_56504

noncomputable def f (x : ℝ) := (x^3 + 3*x^2 + 2*x + 12) / (x^2 - 5*x + 6)

theorem vertical_asymptotes (x : ℝ) : 
  (x^2 - 5*x + 6 = 0) ∧ (x^3 + 3*x^2 + 2*x + 12 ≠ 0) ↔ (x = 2 ∨ x = 3) :=
by
  sorry

end vertical_asymptotes_l56_56504


namespace trigonometric_sign_l56_56468

open Real

theorem trigonometric_sign :
  (0 < 1 ∧ 1 < π / 2) ∧ 
  (∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → sin x ≤ sin y)) ∧ 
  (∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → cos x ≥ cos y)) →
  (cos (cos 1) - cos 1) * (sin (sin 1) - sin 1) < 0 :=
by
  sorry

end trigonometric_sign_l56_56468


namespace positive_difference_sums_l56_56172

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l56_56172


namespace total_cost_with_discounts_l56_56696

theorem total_cost_with_discounts :
  let red_roses := 2 * 12
  let white_roses := 1 * 12
  let yellow_roses := 2 * 12
  let cost_red := red_roses * 6
  let cost_white := white_roses * 7
  let cost_yellow := yellow_roses * 5
  let total_cost_before_discount := cost_red + cost_white + cost_yellow
  let first_discount := 0.15 * total_cost_before_discount
  let cost_after_first_discount := total_cost_before_discount - first_discount
  let additional_discount := 0.10 * cost_after_first_discount
  let total_cost := cost_after_first_discount - additional_discount
  total_cost = 266.22 := by
  sorry

end total_cost_with_discounts_l56_56696


namespace alice_max_score_after_100_ops_l56_56491

def alice_operations : ℕ → ℕ → ℕ
| 0 x := x
| (n + 1) x := let y1 := alice_operations n (x + 1) in
               let y2 := alice_operations n (x * x) in
               max y1 y2

def min_dist_to_perf_square (n : ℕ) : ℕ :=
let upper_bound := nat.sqrt n + 1 in
min (n - (nat.sqrt n)^2) ((upper_bound^2) - n)

theorem alice_max_score_after_100_ops :
  ∃ n, alice_operations 100 0 = n ∧ min_dist_to_perf_square n = 94 := sorry

end alice_max_score_after_100_ops_l56_56491


namespace sarah_loan_amount_l56_56437

-- Define the conditions and question as a Lean theorem
theorem sarah_loan_amount (down_payment : ℕ) (monthly_payment : ℕ) (years : ℕ) (months_in_year : ℕ) :
  (down_payment = 10000) →
  (monthly_payment = 600) →
  (years = 5) →
  (months_in_year = 12) →
  down_payment + (monthly_payment * (years * months_in_year)) = 46000 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end sarah_loan_amount_l56_56437


namespace largest_number_is_34_l56_56108

theorem largest_number_is_34 (a b c : ℕ) (h1 : a + b + c = 82) (h2 : c - b = 8) (h3 : b - a = 4) : c = 34 := 
by 
  sorry

end largest_number_is_34_l56_56108


namespace find_missing_number_l56_56499

theorem find_missing_number
  (a b c d e : ℝ) (mean : ℝ) (f : ℝ)
  (h1 : a = 13) 
  (h2 : b = 8)
  (h3 : c = 13)
  (h4 : d = 7)
  (h5 : e = 23)
  (hmean : mean = 14.2) :
  (a + b + c + d + e + f) / 6 = mean → f = 21.2 :=
by
  sorry

end find_missing_number_l56_56499


namespace interest_rate_l56_56590

-- Define the given conditions
def principal : ℝ := 4000
def total_interest : ℝ := 630.50
def future_value : ℝ := principal + total_interest
def time : ℝ := 1.5  -- 1 1/2 years
def times_compounded : ℝ := 2  -- Compounded half yearly

-- Statement to prove the annual interest rate
theorem interest_rate (P A t n : ℝ) (hP : P = principal) (hA : A = future_value) 
    (ht : t = time) (hn : n = times_compounded) :
    ∃ r : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ r = 0.1 := 
by 
  sorry

end interest_rate_l56_56590


namespace number_of_ordered_pairs_l56_56651

theorem number_of_ordered_pairs :
  ∃ n : ℕ, n = 89 ∧ (∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x < y ∧ 2 * x * y = 8 ^ 30 * (x + y)) := sorry

end number_of_ordered_pairs_l56_56651


namespace jessica_carrots_l56_56076

theorem jessica_carrots
  (joan_carrots : ℕ)
  (total_carrots : ℕ)
  (jessica_carrots : ℕ) :
  joan_carrots = 29 →
  total_carrots = 40 →
  jessica_carrots = total_carrots - joan_carrots →
  jessica_carrots = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jessica_carrots_l56_56076


namespace positive_difference_even_odd_l56_56147

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l56_56147


namespace range_of_a_l56_56254

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l56_56254


namespace positive_difference_of_sums_l56_56139

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l56_56139


namespace sin_double_angle_value_l56_56831

theorem sin_double_angle_value (α : ℝ) (h₁ : Real.sin (π / 4 - α) = 3 / 5) (h₂ : 0 < α ∧ α < π / 4) : 
  Real.sin (2 * α) = 7 / 25 := 
sorry

end sin_double_angle_value_l56_56831


namespace negation_of_statement_l56_56594

theorem negation_of_statement (x : ℝ) :
  (¬ (x^2 = 1 → x = 1 ∨ x = -1)) ↔ (x^2 = 1 ∧ (x ≠ 1 ∧ x ≠ -1)) :=
sorry

end negation_of_statement_l56_56594


namespace max_product_of_sum_2016_l56_56328

theorem max_product_of_sum_2016 (x y : ℤ) (h : x + y = 2016) : x * y ≤ 1016064 :=
by
  -- Proof goes here, but is not needed as per instructions
  sorry

end max_product_of_sum_2016_l56_56328


namespace new_energy_vehicle_price_l56_56088

theorem new_energy_vehicle_price (x : ℝ) :
  (5000 / (x + 1)) = (5000 * (1 - 0.2)) / x :=
sorry

end new_energy_vehicle_price_l56_56088


namespace number_of_selection_plans_l56_56934

-- Definitions based on conditions
def male_students : Nat := 5
def female_students : Nat := 4
def total_volunteers : Nat := 3

def choose (n k : Nat) : Nat :=
  Nat.choose n k

def arrangement_count : Nat :=
  Nat.factorial total_volunteers

-- Theorem that states the total number of selection plans
theorem number_of_selection_plans :
  (choose male_students 2 * choose female_students 1 + choose male_students 1 * choose female_students 2) * arrangement_count = 420 :=
by
  sorry

end number_of_selection_plans_l56_56934


namespace average_first_18_even_numbers_l56_56460

theorem average_first_18_even_numbers : 
  let first_even := 2
  let difference := 2
  let n := 18
  let last_even := first_even + (n - 1) * difference
  let sum := (n / 2) * (first_even + last_even)
  let average := sum / n
  average = 19 :=
by
  -- Definitions
  let first_even := 2
  let difference := 2
  let n := 18
  let last_even := first_even + (n - 1) * difference
  let sum := (n / 2) * (first_even + last_even)
  let average := sum / n
  -- The claim
  show average = 19
  sorry

end average_first_18_even_numbers_l56_56460


namespace no_determinable_cost_of_2_pans_l56_56853

def pots_and_pans_problem : Prop :=
  ∀ (P Q : ℕ), 3 * P + 4 * Q = 100 → ¬∃ Q_cost : ℕ, Q_cost = 2 * Q

theorem no_determinable_cost_of_2_pans : pots_and_pans_problem :=
by
  sorry

end no_determinable_cost_of_2_pans_l56_56853


namespace subset_inequality_l56_56411

open Finset

variable {A : Type*} [Fintype A]

def is_pretty (P : Finset A) : Prop := sorry  -- Define the pretty condition here

def is_small (S P : Finset A) : Prop := S ⊆ P ∧ is_pretty P

def is_big (B P : Finset A) : Prop := P ⊆ B ∧ is_pretty P

def num_pretty : ℕ := (Fintype.card {P // is_pretty P})

def num_small : ℕ := (Fintype.card {S // ∃ P, is_pretty P ∧ S ⊆ P})

def num_big : ℕ := (Fintype.card {B // ∃ P, is_pretty P ∧ P ⊆ B})

theorem subset_inequality :
  2 ^ Fintype.card A * num_pretty ≤ num_small * num_big :=
begin
  sorry  -- Proof goes here
end

end subset_inequality_l56_56411


namespace positive_difference_of_probabilities_l56_56752

theorem positive_difference_of_probabilities :
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1 in
  let p2 := (1 / 2) ^ 4 in
  abs (p1 - p2) = 7 / 16 :=
by
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1
  let p2 := (1 / 2) ^ 4
  have p1_calc: p1 = 1 / 2
  { rw [Nat.choose, _root_.choose, binomial_eq, algebraic_ident]
    sorry  -- The detailed proof of p1 = 1 / 2 will go here
  }
  have p2_calc: p2 = 1 / 16
  { rw [(1/2)^4]
    sorry  -- The detailed proof of p2 = 1 / 16 will go here
  }
  have diff_calc: abs (p1 - p2) = 7 / 16
  { rw [p1_calc, p2_calc]
    sorry  -- The detailed proof of abs (p1 - p2) = 7 / 16 will go here
  }
  exact diff_calc

end positive_difference_of_probabilities_l56_56752


namespace molecular_weight_calculation_l56_56740

theorem molecular_weight_calculation :
  let atomic_weight_K := 39.10
  let atomic_weight_Br := 79.90
  let atomic_weight_O := 16.00
  let num_K := 1
  let num_Br := 1
  let num_O := 3
  let molecular_weight := (num_K * atomic_weight_K) + (num_Br * atomic_weight_Br) + (num_O * atomic_weight_O)
  molecular_weight = 167.00 :=
by
  sorry

end molecular_weight_calculation_l56_56740


namespace not_all_perfect_squares_l56_56292

theorem not_all_perfect_squares (d : ℕ) (hd : 0 < d) :
  ¬ (∃ (x y z : ℕ), 2 * d - 1 = x^2 ∧ 5 * d - 1 = y^2 ∧ 13 * d - 1 = z^2) :=
by
  sorry

end not_all_perfect_squares_l56_56292


namespace probability_of_premium_best_selling_option_compare_probabilities_l56_56213

theorem probability_of_premium (premium_boxes total_boxes : ℕ) (h_premium : premium_boxes = 40) (h_total : total_boxes = 100) :
  (premium_boxes / total_boxes : ℝ) = 2 / 5 := 
by {
  sorry
}

theorem best_selling_option (premium_boxes special_boxes superior_boxes first_grade_boxes : ℕ) 
  (p1_price : ℝ) (p2_price : ℝ) (p3_price : ℝ) (p4_price : ℝ)
  (h_premium : premium_boxes = 40) (h_special : special_boxes = 30) 
  (h_superior : superior_boxes = 10) (h_first_grade : first_grade_boxes = 20)
  (h_p1_price : p1_price = 36) (h_p2_price : p2_price = 30) 
  (h_p3_price : p3_price = 24) (h_p4_price : p4_price = 18) :
  let avg_price : ℝ := (p1_price * premium_boxes + p2_price * special_boxes + p3_price * superior_boxes + p4_price * first_grade_boxes) / 100 in
  avg_price = 29.4 :=
by {
  sorry
}

theorem compare_probabilities (p1 p2 : ℝ) 
  (h_p1 : p1 = 1465 / 1617) (h_p2 : p2 = 53 / 57) :
  p1 < p2 :=
by {
  sorry
}

end probability_of_premium_best_selling_option_compare_probabilities_l56_56213


namespace total_number_of_toys_is_105_l56_56707

-- Definitions
variables {a k : ℕ}

-- Conditions
def condition_1 (a k : ℕ) : Prop := k ≥ 2
def katya_toys (a : ℕ) : ℕ := a
def lena_toys (a k : ℕ) : ℕ := k * a
def masha_toys (a k : ℕ) : ℕ := k^2 * a

def after_katya_gave_toys (a : ℕ) : ℕ := a - 2
def after_lena_received_toys (a k : ℕ) : ℕ := k * a + 5
def after_masha_gave_toys (a k : ℕ) : ℕ := k^2 * a - 3

def arithmetic_progression (x1 x2 x3 : ℕ) : Prop :=
  2 * x2 = x1 + x3

-- Problem statement to prove
theorem total_number_of_toys_is_105 (a k : ℕ) (h1 : condition_1 a k)
  (h2 : arithmetic_progression (after_katya_gave_toys a) (after_lena_received_toys a k) (after_masha_gave_toys a k)) :
  katya_toys a + lena_toys a k + masha_toys a k = 105 :=
sorry

end total_number_of_toys_is_105_l56_56707


namespace left_seats_equals_15_l56_56273

variable (L : ℕ)

noncomputable def num_seats_left (L : ℕ) : Prop :=
  ∃ L, 3 * L + 3 * (L - 3) + 8 = 89

theorem left_seats_equals_15 : num_seats_left L → L = 15 :=
by
  intro h
  sorry

end left_seats_equals_15_l56_56273


namespace sum_of_a_and_b_is_two_l56_56729

variable (a b : ℝ)
variable (h_a_nonzero : a ≠ 0)
variable (h_fn_passes_through_point : (a * 1^2 + b * 1 - 1) = 1)

theorem sum_of_a_and_b_is_two : a + b = 2 := 
by
  sorry

end sum_of_a_and_b_is_two_l56_56729


namespace proposition_not_true_3_l56_56928

theorem proposition_not_true_3 (P : ℕ → Prop) (h1 : ∀ n, P n → P (n + 1)) (h2 : ¬ P 4) : ¬ P 3 :=
by
  sorry

end proposition_not_true_3_l56_56928


namespace count_of_valid_triplets_l56_56686

open Finset

noncomputable def orig_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def mean_of_remaining_set (s : Finset ℕ) : Prop :=
  (∑ x in s, (x : ℝ)) / (s.card : ℝ) = 5

def valid_triplets : Finset (Finset ℕ) :=
  orig_set.powerset.filter (λ s, s.card = 3 ∧ ∑ x in s, x = 15)

theorem count_of_valid_triplets : valid_triplets.card = 5 := sorry

end count_of_valid_triplets_l56_56686


namespace part_a_part_b_l56_56474

theorem part_a (N : ℕ) : ∃ (a : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i → i ≤ N → a i > 0) ∧ (∀ i : ℕ, 2 ≤ i → i ≤ N → a i > a (i - 1)) ∧ 
(∀ i j : ℕ, 1 ≤ i → i < j → j ≤ N → (1 : ℚ) / a i - (1 : ℚ) / a j = (1 : ℚ) / a 1 - (1 : ℚ) / a 2) := sorry

theorem part_b : ¬ ∃ (a : ℕ → ℕ), (∀ i : ℕ, a i > 0) ∧ (∀ i : ℕ, a i < a (i + 1)) ∧ 
(∀ i j : ℕ, i < j → (1 : ℚ) / a i - (1 : ℚ) / a j = (1 : ℚ) / a 0 - (1 : ℚ) / a 1) := sorry

end part_a_part_b_l56_56474


namespace largest_triangle_angle_l56_56109

-- Define the angles
def angle_sum := (105 : ℝ) -- Degrees
def delta_angle := (36 : ℝ) -- Degrees
def total_sum := (180 : ℝ) -- Degrees

-- Theorem statement
theorem largest_triangle_angle (a b c : ℝ) (h1 : a + b = angle_sum)
  (h2 : b = a + delta_angle) (h3 : a + b + c = total_sum) : c = 75 :=
sorry

end largest_triangle_angle_l56_56109


namespace ellipse_properties_l56_56788

theorem ellipse_properties 
  (foci1 foci2 : ℝ × ℝ) 
  (point_on_ellipse : ℝ × ℝ) 
  (h k a b : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0) 
  (ellipse_condition : foci1 = (-4, 1) ∧ foci2 = (-4, 5) ∧ point_on_ellipse = (1, 3))
  (ellipse_eqn : (x y : ℝ) → ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1) :
  a + k = 8 :=
by
  sorry

end ellipse_properties_l56_56788


namespace expansion_coeff_sum_l56_56098

theorem expansion_coeff_sum :
  (∃ a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ, 
    (2*x - 1)^10 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 + a6*x^6 + a7*x^7 + a8*x^8 + a9*x^9 + a10*x^10)
  → (1 - 20 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 1 → a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 20) :=
by
  sorry

end expansion_coeff_sum_l56_56098


namespace alex_ahead_of_max_after_even_l56_56490

theorem alex_ahead_of_max_after_even (x : ℕ) (h1 : x - 200 + 170 + 440 = 1110) : x = 300 :=
sorry

end alex_ahead_of_max_after_even_l56_56490


namespace positive_difference_sums_l56_56175

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l56_56175


namespace stewart_farm_horseFood_l56_56223

variable (sheep horses horseFoodPerHorse : ℕ)
variable (ratio_sh_to_hs : ℕ × ℕ)
variable (totalHorseFood : ℕ)

noncomputable def horse_food_per_day (sheep : ℕ) (ratio_sh_to_hs : ℕ × ℕ) (totalHorseFood : ℕ) : ℕ :=
  let horses := (sheep * ratio_sh_to_hs.2) / ratio_sh_to_hs.1
  totalHorseFood / horses

theorem stewart_farm_horseFood (h_ratio : ratio_sh_to_hs = (4, 7))
                                (h_sheep : sheep = 32)
                                (h_total : totalHorseFood = 12880) :
    horse_food_per_day sheep ratio_sh_to_hs totalHorseFood = 230 := by
  sorry

end stewart_farm_horseFood_l56_56223


namespace positive_difference_even_odd_sums_l56_56166

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l56_56166


namespace value_of_a1_plus_a10_l56_56523

noncomputable def geometric_sequence {α : Type*} [Field α] (a : ℕ → α) :=
  ∃ q : α, ∀ n : ℕ, a (n + 1) = a n * q

theorem value_of_a1_plus_a10 (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : a 4 + a 7 = 2) 
  (h3 : a 5 * a 6 = -8) 
  : a 1 + a 10 = -7 := 
by
  sorry

end value_of_a1_plus_a10_l56_56523


namespace combined_distance_l56_56406

theorem combined_distance (t1 t2 : ℕ) (s1 s2 : ℝ)
  (h1 : t1 = 30) (h2 : s1 = 9.5) (h3 : t2 = 45) (h4 : s2 = 8.3)
  : (s1 * t1 + s2 * t2) = 658.5 := 
by
  sorry

end combined_distance_l56_56406


namespace beef_not_used_l56_56850

-- Define the context and necessary variables
variable (totalBeef : ℕ) (usedVegetables : ℕ)
variable (beefUsed : ℕ)

-- The conditions given in the problem
def initial_beef : Prop := totalBeef = 4
def used_vegetables : Prop := usedVegetables = 6
def relation_vegetables_beef : Prop := usedVegetables = 2 * beefUsed

-- The statement we need to prove
theorem beef_not_used
  (h1 : initial_beef totalBeef)
  (h2 : used_vegetables usedVegetables)
  (h3 : relation_vegetables_beef usedVegetables beefUsed) :
  (totalBeef - beefUsed) = 1 := by
  sorry

end beef_not_used_l56_56850


namespace probability_different_tens_digit_l56_56441

theorem probability_different_tens_digit :
  let n := 6
  let range := Set.Icc 10 59
  let tens_digit (x : ℕ) := x / 10
  let valid_set (s : Set ℕ) := ∀ x ∈ s, 10 ≤ x ∧ x ≤ 59
  let different_tens_digits (s : Set ℕ) := (∀ (x y : ℕ), x ∈ s → y ∈ s → x ≠ y → tens_digit x ≠ tens_digit y)
  let total_ways := Nat.choose 50 6
  let favorable_ways := 5 * 10 * 9 * 10^4
  let probability := favorable_ways * 1 / total_ways
  valid_set ({ x | x ∈ range } : Set ℕ) →
  different_tens_digits ({ x | x ∈ range } : Set ℕ) →
  probability = (1500000 : ℚ) / 5296900 :=
by
  sorry

end probability_different_tens_digit_l56_56441


namespace positive_difference_even_odd_sums_l56_56177

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l56_56177


namespace polygon_stats_l56_56945

-- Definitions based on the problem's conditions
def total_number_of_polygons : ℕ := 207
def median_position : ℕ := 104
def m : ℕ := 14
def sum_of_squares_of_sides : ℕ := 2860
def mean_value : ℚ := sum_of_squares_of_sides / total_number_of_polygons
def mode_median : ℚ := 11.5

-- The proof statement
theorem polygon_stats (d μ M : ℚ)
  (h₁ : μ = mean_value)
  (h₂ : d = mode_median)
  (h₃ : M = m) :
  d < μ ∧ μ < M :=
by
  rw [h₁, h₂, h₃]
  -- The exact proof steps are omitted
  sorry

end polygon_stats_l56_56945


namespace simplify_fraction_90_150_l56_56866

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end simplify_fraction_90_150_l56_56866


namespace BoxMullerTransform_normal_independent_l56_56857

open MeasureTheory ProbabilityTheory

/-- Lean statement only defining conditions -/

noncomputable def BoxMullerTransform (U V : ℝ) : ℝ × ℝ := 
  (sqrt (- 2 * log V) * cos (2 * π * U), sqrt (- 2 * log V) * sin (2 * π * U))

/-- Main theorem to prove X and Y are independent and normally distributed -/

theorem BoxMullerTransform_normal_independent 
  (U V : MeasureTheory.MeasureSpace ℝ) 
  (hU : ProbabilityTheory.Independent U) 
  (hV : ProbabilityTheory.Independent V) 
  (hU_dist : MeasureTheory.ProbabilityMeasure U)
  (hV_dist : MeasureTheory.ProbabilityMeasure V)
  (hU_uniform : ∀ a b, ∫ x in set.Ioo a b, U.density x = (b - a))
  (hV_uniform : ∀ a b, ∫ x in set.Ioo a b, V.density x = (b - a)) :
  let (X, Y) := BoxMullerTransform U V in
  (ProbabilityTheory.Independent X Y 
  ∧ ProbabilityTheory.HasPDF X (λ x, exp (-(x ^ 2) / 2) / sqrt (2 * π))
  ∧ ProbabilityTheory.HasPDF Y (λ y, exp (-(y ^ 2) / 2) / sqrt (2 * π))) := 
sorry

end BoxMullerTransform_normal_independent_l56_56857


namespace bobby_last_10_throws_successful_l56_56792

theorem bobby_last_10_throws_successful :
    let initial_successful := 18 -- Bobby makes 18 successful throws out of his initial 30 throws.
    let total_throws := 30 + 10 -- Bobby makes a total of 40 throws.
    let final_successful := 0.64 * total_throws -- Bobby needs to make 64% of 40 throws to achieve a 64% success rate.
    let required_successful := 26 -- Adjusted to the nearest whole number.
    -- Bobby makes 8 successful throws in his last 10 attempts.
    required_successful - initial_successful = 8 := by
  sorry

end bobby_last_10_throws_successful_l56_56792


namespace slope_of_line_l56_56605

theorem slope_of_line (x₁ y₁ x₂ y₂ : ℝ) (h₁ : 2/x₁ + 3/y₁ = 0) (h₂ : 2/x₂ + 3/y₂ = 0) (h_diff : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -3/2 :=
sorry

end slope_of_line_l56_56605


namespace find_m_from_expansion_l56_56061

theorem find_m_from_expansion (m n : ℤ) (h : (x : ℝ) → (x + 3) * (x + n) = x^2 + m * x - 21) : m = -4 :=
by
  sorry

end find_m_from_expansion_l56_56061


namespace complementary_angles_of_same_angle_are_equal_l56_56337

def complementary_angles (α β : ℝ) := α + β = 90 

theorem complementary_angles_of_same_angle_are_equal 
        (θ : ℝ) (α β : ℝ) 
        (h1 : complementary_angles θ α) 
        (h2 : complementary_angles θ β) : 
        α = β := 
by 
  sorry

end complementary_angles_of_same_angle_are_equal_l56_56337


namespace max_regions_by_five_lines_l56_56091

theorem max_regions_by_five_lines : 
  ∀ (R : ℕ → ℕ), R 1 = 2 → R 2 = 4 → (∀ n, R (n + 1) = R n + (n + 1)) → R 5 = 16 :=
by
  intros R hR1 hR2 hRec
  sorry

end max_regions_by_five_lines_l56_56091


namespace union_A_B_complement_intersect_B_intersection_sub_C_l56_56700

-- Define set A
def A : Set ℝ := {x | -5 < x ∧ x < 1}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 8}

-- Define set C with variable parameter a
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Problem (1): Prove A ∪ B = { x | -5 < x < 8 }
theorem union_A_B : A ∪ B = {x | -5 < x ∧ x < 8} := 
by sorry

-- Problem (1): Prove (complement R A) ∩ B = { x | 1 ≤ x < 8 }
theorem complement_intersect_B : (Aᶜ) ∩ B = {x | 1 ≤ x ∧ x < 8} :=
by sorry

-- Problem (2): If A ∩ B ⊆ C, prove a ≥ 1
theorem intersection_sub_C (a : ℝ) (h : A ∩ B ⊆ C a) : 1 ≤ a :=
by sorry

end union_A_B_complement_intersect_B_intersection_sub_C_l56_56700


namespace arcsin_sqrt_three_over_two_l56_56942

theorem arcsin_sqrt_three_over_two : 
  ∃ θ, θ = Real.arcsin (Real.sqrt 3 / 2) ∧ θ = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_three_over_two_l56_56942


namespace expression_for_f_l56_56049

theorem expression_for_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x^2 - x - 2) : ∀ x : ℤ, f x = x^2 - 3 * x := 
by
  sorry

end expression_for_f_l56_56049


namespace proof_m_cd_value_l56_56054

theorem proof_m_cd_value (a b c d m : ℝ) 
  (H1 : a + b = 0) (H2 : c * d = 1) (H3 : |m| = 3) : 
  m + c * d - (a + b) / (m ^ 2) = 4 ∨ m + c * d - (a + b) / (m ^ 2) = -2 :=
by
  sorry

end proof_m_cd_value_l56_56054


namespace find_linear_equation_l56_56611

def is_linear_eq (eq : String) : Prop :=
  eq = "2x = 0"

theorem find_linear_equation :
  is_linear_eq "2x = 0" :=
by
  sorry

end find_linear_equation_l56_56611


namespace merchants_and_cost_l56_56863

theorem merchants_and_cost (n C : ℕ) (h1 : 8 * n = C + 3) (h2 : 7 * n = C - 4) : n = 7 ∧ C = 53 := 
by 
  sorry

end merchants_and_cost_l56_56863


namespace find_number_l56_56001

theorem find_number (x : ℤ) (h : 3 * x + 3 * 12 + 3 * 13 + 11 = 134) : x = 16 :=
by
  sorry

end find_number_l56_56001


namespace marching_band_l56_56445

theorem marching_band (total_members brass woodwind percussion : ℕ)
  (h1 : brass + woodwind + percussion = 110)
  (h2 : woodwind = 2 * brass)
  (h3 : percussion = 4 * woodwind) :
  brass = 10 := by
  sorry

end marching_band_l56_56445


namespace find_a_maximize_profit_l56_56210

-- Definition of parameters
def a := 260
def purchase_price_table := a
def purchase_price_chair := a - 140

-- Condition 1: The number of dining chairs purchased for 600 yuan is the same as the number of dining tables purchased for 1300 yuan.
def condition1 := (600 / (purchase_price_chair : ℚ)) = (1300 / (purchase_price_table : ℚ))

-- Given conditions for profit maximization
def qty_tables := 30
def qty_chairs := 5 * qty_tables + 20
def total_qty := qty_tables + qty_chairs

-- Condition: Total quantity of items does not exceed 200 units.
def condition2 := total_qty ≤ 200

-- Profit calculation
def profit := 280 * qty_tables + 800

-- Theorem statements
theorem find_a : condition1 → a = 260 := sorry

theorem maximize_profit : condition2 ∧ (8 * qty_tables + 800 > 0) → 
  (qty_tables = 30) ∧ (qty_chairs = 170) ∧ (profit = 9200) := sorry

end find_a_maximize_profit_l56_56210


namespace find_y_l56_56268

theorem find_y (steps distance : ℕ) (total_steps : ℕ) (marking_step : ℕ)
  (h1 : total_steps = 8)
  (h2 : distance = 48)
  (h3 : marking_step = 6) :
  steps = distance / total_steps * marking_step → steps = 36 :=
by
  intros
  sorry

end find_y_l56_56268


namespace find_k_l56_56101

-- Define the lines l1 and l2
def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x - y - 2 = 0

-- Define the fact that the quadrilateral formed by l1, l2, and the positive halves of the axes
-- has a circumscribed circle.
def has_circumscribed_circle (k : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), line1 x1 y1 ∧ line2 k x2 y2 ∧
  x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 > 0 ∧
  (x1 - x2 = 0 ∨ y1 - y2 = 0) ∧
  (x1 = 0 ∨ y1 = 0 ∨ x2 = 0 ∨ y2 = 0)

-- The statement we need to prove
theorem find_k : ∀ k : ℝ, has_circumscribed_circle k → k = 3 := by
  sorry

end find_k_l56_56101


namespace num_students_second_grade_l56_56785

structure School :=
(total_students : ℕ)
(prob_male_first_grade : ℝ)

def stratified_sampling (school : School) : ℕ := sorry

theorem num_students_second_grade (school : School) (total_selected : ℕ) : 
    school.total_students = 4000 →
    school.prob_male_first_grade = 0.2 →
    total_selected = 100 →
    stratified_sampling school = 30 :=
by
  intros
  sorry

end num_students_second_grade_l56_56785


namespace visibility_count_in_square_l56_56783

open Nat
open scoped BigOperators

theorem visibility_count_in_square :
  let points := Finset.Icc (0, 0) (25, 25).filter (λ p, p ≠ (0, 0) ∧ gcd p.fst p.snd = 1)
  Finset.card points = 399 := 
by {
  let points := Finset.Icc (0, 0) (25, 25),
  let filtered_points := points.filter (λ p, p ≠ (0, 0) ∧ gcd p.fst p.snd = 1),
  have visible_points_count : Finset.card filtered_points = 1 + 2 * ∑ x in Finset.range(25).filter (λ x, x > 0), Nat.totient (x+1),
  sorry
}

end visibility_count_in_square_l56_56783


namespace compare_costs_l56_56963

def cost_X (copies: ℕ) : ℝ :=
  if copies >= 40 then
    (copies * 1.25) * 0.95
  else
    copies * 1.25

def cost_Y (copies: ℕ) : ℝ :=
  if copies >= 100 then
    copies * 2.00
  else if copies >= 60 then
    copies * 2.25
  else
    copies * 2.75

def cost_Z (copies: ℕ) : ℝ :=
  if copies >= 50 then
    (copies * 3.00) * 0.90
  else
    copies * 3.00

def cost_W (copies: ℕ) : ℝ :=
  let bulk_groups := copies / 25
  let remainder := copies % 25
  (bulk_groups * 40) + (remainder * 2.00)

theorem compare_costs : 
  cost_X 60 < cost_Y 60 ∧ 
  cost_X 60 < cost_Z 60 ∧ 
  cost_X 60 < cost_W 60 ∧
  cost_Y 60 - cost_X 60 = 63.75 ∧
  cost_Z 60 - cost_X 60 = 90.75 ∧
  cost_W 60 - cost_X 60 = 28.75 :=
  sorry

end compare_costs_l56_56963


namespace projection_is_q_l56_56041

noncomputable def projection_matrix := 
  let u := ![1, -1, 2]
  (1 / 6: ℚ) • ![
    ![1, -1, 2],
    ![-1, 1, -2],
    ![2, -2, 4]
  ]

theorem projection_is_q (v : Fin 3 → ℚ) :
  let Q := ![
    ![(1 / 6): ℚ, (-1 / 6): ℚ, (1 / 3): ℚ],
    ![(-1 / 6): ℚ, (1 / 6): ℚ, (-1 / 3): ℚ],
    ![(1 / 3): ℚ, (-1 / 3): ℚ, (2 / 3): ℚ]]
  in 
  Q.mul_vec v = projection_matrix.mul_vec v :=
sorry

end projection_is_q_l56_56041


namespace value_of_expression_l56_56501

def delta (a b : ℕ) : ℕ := a * a - b

theorem value_of_expression :
  delta (5 ^ (delta 6 17)) (2 ^ (delta 7 11)) = 5 ^ 38 - 2 ^ 38 :=
by
  sorry

end value_of_expression_l56_56501


namespace general_term_formula_sum_first_n_terms_l56_56281

theorem general_term_formula :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a n > 0) →
  a 1 = 1 / 2 →
  (∀ n, (a (n + 1))^2 = a n^2 + 2 * ↑n) →
  (∀ n, a n = n - 1 / 2) := 
  sorry

theorem sum_first_n_terms :
  ∀ (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n, a n > 0) →
  a 1 = 1 / 2 →
  (∀ n, (a (n + 1))^2 = a n^2 + 2 * ↑n) →
  (∀ n, a n = n - 1 / 2) →
  (∀ n, b n = 1 / (a n * a (n + 1))) →
  (∀ n, S n = 2 * (1 - 1 / (2 * n + 1))) →
  (S n = 4 * n / (2 * n + 1)) :=
  sorry

end general_term_formula_sum_first_n_terms_l56_56281


namespace feathers_per_crown_l56_56378

theorem feathers_per_crown (total_feathers total_crowns feathers_per_crown : ℕ) 
  (h₁ : total_feathers = 6538) 
  (h₂ : total_crowns = 934) 
  (h₃ : feathers_per_crown = total_feathers / total_crowns) : 
  feathers_per_crown = 7 := 
by 
  sorry

end feathers_per_crown_l56_56378


namespace problem_solution_l56_56262

variable (a b c d m : ℝ)

-- Conditions
def opposite_numbers (a b : ℝ) : Prop := a + b = 0
def reciprocals (c d : ℝ) : Prop := c * d = 1
def absolute_value_eq (m : ℝ) : Prop := |m| = 3

theorem problem_solution
  (h1 : opposite_numbers a b)
  (h2 : reciprocals c d)
  (h3 : absolute_value_eq m) :
  (a + b) / 2023 - 4 * (c * d) + m^2 = 5 :=
by
  sorry

end problem_solution_l56_56262


namespace range_of_ab_c2_l56_56383

theorem range_of_ab_c2
  (a b c : ℝ)
  (h₁: -3 < b)
  (h₂: b < a)
  (h₃: a < -1)
  (h₄: -2 < c)
  (h₅: c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
by 
  sorry

end range_of_ab_c2_l56_56383


namespace b_3_value_S_m_formula_l56_56084

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

end b_3_value_S_m_formula_l56_56084


namespace stamps_needed_l56_56565

def paper_weight : ℚ := 1 / 5
def num_papers : ℕ := 8
def envelope_weight : ℚ := 2 / 5
def stamp_per_ounce : ℕ := 1

theorem stamps_needed : num_papers * paper_weight + envelope_weight = 2 →
  (num_papers * paper_weight + envelope_weight) * stamp_per_ounce = 2 :=
by
  intro h
  rw h
  simp
  sorry

end stamps_needed_l56_56565


namespace problem_statement_l56_56772

variables {x y P Q : ℝ}

theorem problem_statement (h1 : x^2 + y^2 = (x + y)^2 + P) (h2 : x^2 + y^2 = (x - y)^2 + Q) : P = -2 * x * y ∧ Q = 2 * x * y :=
by
  sorry

end problem_statement_l56_56772


namespace intersection_of_P_and_Q_l56_56392
-- Import the entire math library

-- Define the conditions for sets P and Q
def P := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | (x - 1)^2 ≤ 4}

-- Define the theorem to prove that P ∩ Q = {x | 1 ≤ x ∧ x ≤ 3}
theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 ≤ x ∧ x ≤ 3} :=
by
  -- Placeholder for the proof
  sorry

end intersection_of_P_and_Q_l56_56392


namespace range_of_x_in_function_l56_56999

theorem range_of_x_in_function : ∀ (x : ℝ), (2 - x ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ -2) :=
by
  intro x
  sorry

end range_of_x_in_function_l56_56999


namespace prime_factors_sum_correct_prime_factors_product_correct_l56_56232

-- The number we are considering
def n : ℕ := 172480

-- Prime factors of the number n
def prime_factors : List ℕ := [2, 3, 5, 719]

-- Sum of the prime factors
def sum_prime_factors : ℕ := 2 + 3 + 5 + 719

-- Product of the prime factors
def prod_prime_factors : ℕ := 2 * 3 * 5 * 719

theorem prime_factors_sum_correct :
  sum_prime_factors = 729 :=
by {
  -- Proof goes here
  sorry
}

theorem prime_factors_product_correct :
  prod_prime_factors = 21570 :=
by {
  -- Proof goes here
  sorry
}

end prime_factors_sum_correct_prime_factors_product_correct_l56_56232


namespace tom_candies_left_is_ten_l56_56901

-- Define initial conditions
def initial_candies: ℕ := 2
def friend_gave_candies: ℕ := 7
def bought_candies: ℕ := 10

-- Define total candies before sharing
def total_candies := initial_candies + friend_gave_candies + bought_candies

-- Define the number of candies Tom gives to his sister
def candies_given := total_candies / 2

-- Define the number of candies Tom has left
def candies_left := total_candies - candies_given

-- Prove the final number of candies left
theorem tom_candies_left_is_ten : candies_left = 10 :=
by
  -- The proof is left as an exercise
  sorry

end tom_candies_left_is_ten_l56_56901


namespace factorization_of_x12_sub_729_l56_56646

theorem factorization_of_x12_sub_729 (x : ℝ) :
  x^12 - 729 = (x^3 + 3) * (x - real.cbrt 3) * (x^2 + x * real.cbrt 3 + (real.cbrt 3)^2) * (x^12 + 9 * x^6 + 81) := 
sorry

end factorization_of_x12_sub_729_l56_56646


namespace complex_division_l56_56373

open Complex

theorem complex_division :
  (1 + 2 * I) / (3 - 4 * I) = -1 / 5 + 2 / 5 * I :=
by
  sorry

end complex_division_l56_56373


namespace JulieCompletesInOneHour_l56_56472

-- Define conditions
def JuliePeelsIn : ℕ := 10
def TedPeelsIn : ℕ := 8
def TimeTogether : ℕ := 4

-- Define their respective rates
def JulieRate : ℚ := 1 / JuliePeelsIn
def TedRate : ℚ := 1 / TedPeelsIn

-- Define the task completion in 4 hours together
def TaskCompletedTogether : ℚ := (JulieRate * TimeTogether) + (TedRate * TimeTogether)

-- Define remaining task after working together
def RemainingTask : ℚ := 1 - TaskCompletedTogether

-- Define time for Julie to complete the remaining task
def TimeForJulieToComplete : ℚ := RemainingTask / JulieRate

-- The theorem statement
theorem JulieCompletesInOneHour :
  TimeForJulieToComplete = 1 := by
  sorry

end JulieCompletesInOneHour_l56_56472


namespace coin_tosses_l56_56467

theorem coin_tosses (n : ℤ) (h : (1/2 : ℝ)^n = 0.125) : n = 3 :=
by
  sorry

end coin_tosses_l56_56467


namespace unique_number_not_in_range_l56_56727

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ :=
  (p * x + q) / (r * x + s)

theorem unique_number_not_in_range (p q r s : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : r ≠ 0) (h₃ : s ≠ 0) 
  (h₄ : g p q r s 23 = 23) (h₅ : g p q r s 101 = 101) (h₆ : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  p / r = 62 :=
sorry

end unique_number_not_in_range_l56_56727


namespace sasha_quarters_max_l56_56717

/-- Sasha has \$4.80 in U.S. coins. She has four times as many dimes as she has nickels 
and the same number of quarters as nickels. Prove that the greatest number 
of quarters she could have is 6. -/
theorem sasha_quarters_max (q n d : ℝ) (h1 : 0.25 * q + 0.05 * n + 0.1 * d = 4.80)
  (h2 : n = q) (h3 : d = 4 * n) : q = 6 := 
sorry

end sasha_quarters_max_l56_56717


namespace intersection_complement_l56_56830

universe u
variable {α : Type u}

-- Define the sets I, M, N, and their complement with respect to I
def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}
def complement_I (s : Set ℕ) : Set ℕ := { x ∈ I | x ∉ s }

-- Statement of the theorem
theorem intersection_complement :
  M ∩ (complement_I N) = {1} :=
by
  sorry

end intersection_complement_l56_56830


namespace playerA_mean_playerA_median_playerA_mode_playerB_mode_playerB_variance_playerB_variance_less_than_2_6_l56_56321

noncomputable def playerA_scores := [5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 9.0, 9.0, 10.0]
noncomputable def playerB_scores := [5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 9.0, 10.0]

def mean (l : List ℝ) : ℝ := l.sum / l.length

theorem playerA_mean : mean playerA_scores = 7 := by sorry
theorem playerA_median : Statistics.median playerA_scores = 6 := by sorry
theorem playerA_mode : Statistics.mode playerA_scores = 6 := by sorry

theorem playerB_mode : Statistics.mode playerB_scores = 7 := by sorry
theorem playerB_variance : Statistics.variance playerB_scores = 2 := by sorry
theorem playerB_variance_less_than_2_6 : Statistics.variance playerB_scores < 2.6 := by sorry

end playerA_mean_playerA_median_playerA_mode_playerB_mode_playerB_variance_playerB_variance_less_than_2_6_l56_56321


namespace capacity_of_second_bucket_l56_56476

theorem capacity_of_second_bucket (c1 : ∃ (tank_capacity : ℕ), tank_capacity = 12 * 49) (c2 : ∃ (bucket_count : ℕ), bucket_count = 84) :
  ∃ (bucket_capacity : ℕ), bucket_capacity = 7 :=
by
  -- Extract the total capacity of the tank from condition 1
  obtain ⟨tank_capacity, htank⟩ := c1
  -- Extract the number of buckets from condition 2
  obtain ⟨bucket_count, hcount⟩ := c2
  -- Use the given relations to calculate the capacity of each bucket
  use tank_capacity / bucket_count
  -- Provide the necessary calculations
  sorry

end capacity_of_second_bucket_l56_56476


namespace evaluate_expression_l56_56953

theorem evaluate_expression : (2^3002 * 3^3004) / (6^3003) = (3 / 2) := by
  sorry

end evaluate_expression_l56_56953


namespace exist_functions_fg_neq_f1f1_g1g1_l56_56342

-- Part (a)
theorem exist_functions_fg :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, (f ∘ g) x = (g ∘ f) x) ∧ 
    (∀ x, (f ∘ f) x = (g ∘ g) x) ∧ 
    (∀ x, f x ≠ g x) := 
sorry

-- Part (b)
theorem neq_f1f1_g1g1 
  (f1 g1 : ℝ → ℝ)
  (H_comm : ∀ x, (f1 ∘ g1) x = (g1 ∘ f1) x)
  (H_neq: ∀ x, f1 x ≠ g1 x) :
  ∀ x, (f1 ∘ f1) x ≠ (g1 ∘ g1) x :=
sorry

end exist_functions_fg_neq_f1f1_g1g1_l56_56342


namespace digit_d_multiple_of_9_l56_56381

theorem digit_d_multiple_of_9 (d : ℕ) (hd : d = 1) : ∃ k : ℕ, (56780 + d) = 9 * k := by
  have : 56780 + d = 56780 + 1 := by rw [hd]
  rw [this]
  use 6313
  sorry

end digit_d_multiple_of_9_l56_56381


namespace matilda_father_chocolates_left_l56_56708

-- definitions for each condition
def initial_chocolates : ℕ := 20
def persons : ℕ := 5
def chocolates_per_person := initial_chocolates / persons
def half_chocolates_per_person := chocolates_per_person / 2
def total_given_to_father := half_chocolates_per_person * persons
def chocolates_given_to_mother := 3
def chocolates_eaten_by_father := 2

-- statement to prove
theorem matilda_father_chocolates_left :
  total_given_to_father - chocolates_given_to_mother - chocolates_eaten_by_father = 5 :=
by
  sorry

end matilda_father_chocolates_left_l56_56708


namespace intersection_of_M_and_N_l56_56257

-- Define sets M and N as given in the conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- The theorem statement to prove the intersection of M and N is {2, 3}
theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by sorry  -- The proof is skipped with 'sorry'

end intersection_of_M_and_N_l56_56257


namespace journey_time_l56_56775

noncomputable def velocity_of_stream : ℝ := 4
noncomputable def speed_of_boat_in_still_water : ℝ := 14
noncomputable def distance_A_to_B : ℝ := 180
noncomputable def distance_B_to_C : ℝ := distance_A_to_B / 2
noncomputable def downstream_speed : ℝ := speed_of_boat_in_still_water + velocity_of_stream
noncomputable def upstream_speed : ℝ := speed_of_boat_in_still_water - velocity_of_stream

theorem journey_time : (distance_A_to_B / downstream_speed) + (distance_B_to_C / upstream_speed) = 19 := by
  sorry

end journey_time_l56_56775


namespace workers_production_l56_56325

theorem workers_production
    (x y : ℝ)
    (h1 : x + y = 72)
    (h2 : 1.15 * x + 1.25 * y = 86) :
    1.15 * x = 46 ∧ 1.25 * y = 40 :=
by {
  sorry
}

end workers_production_l56_56325


namespace intervals_of_monotonic_increase_triangle_properties_l56_56249

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x - π / 6) - 1

theorem intervals_of_monotonic_increase (k : ℤ) :
  ∃ (a b : ℝ), a = k * π - π / 6 ∧ b = k * π + π / 3 ∧
    ∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 < f x2 := sorry

theorem triangle_properties (b : ℝ) (B a c : ℝ) :
  b = Real.sqrt 7 ∧ B = π / 3 ∧ (∃ C, (Real.sin A = 3 * Real.sin C ∧ a = 3 * c) ∧
    a = 3 ∧ c = 1 ∧ (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 4) := sorry

end intervals_of_monotonic_increase_triangle_properties_l56_56249


namespace sum_of_five_integers_l56_56464

-- Definitions of the five integers based on the conditions given in the problem
def a := 12345
def b := 23451
def c := 34512
def d := 45123
def e := 51234

-- Statement of the proof problem
theorem sum_of_five_integers :
  a + b + c + d + e = 166665 :=
by
  -- The proof is omitted
  sorry

end sum_of_five_integers_l56_56464


namespace solve_for_x_l56_56609

theorem solve_for_x : (∃ x : ℝ, (x / 18) * (x / 72) = 1) → ∃ x : ℝ, x = 36 :=
by
  sorry

end solve_for_x_l56_56609


namespace positive_difference_of_sums_l56_56125

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l56_56125


namespace contrapositive_of_x_squared_eq_one_l56_56725

theorem contrapositive_of_x_squared_eq_one (x : ℝ) 
  (h : x^2 = 1 → x = 1 ∨ x = -1) : (x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1 :=
by
  sorry

end contrapositive_of_x_squared_eq_one_l56_56725


namespace prime_divisors_50fact_count_l56_56548

theorem prime_divisors_50fact_count : 
  (finset.filter prime (finset.range 51)).card = 15 := 
by 
  simp only [finset.filter, finset.range, prime],
  sorry

end prime_divisors_50fact_count_l56_56548


namespace buckets_required_l56_56900

theorem buckets_required (C : ℝ) (hC : C > 0) :
  let original_bucket_count := 25
  let reduction_factor := 2 / 5
  let new_bucket_count := original_bucket_count / reduction_factor
  new_bucket_count.ceil = 63 :=
by
  -- Definitions and conditions
  let original_bucket_count := 25
  let reduction_factor := 2 / 5
  let total_capacity := original_bucket_count * C
  let new_bucket_capacity := reduction_factor * C
  let new_bucket_count := total_capacity / new_bucket_capacity
  
  -- Main goal
  have : new_bucket_count = (25 * C) / ((2 / 5) * C) := by sorry
  have : new_bucket_count = 25 / (2 / 5) := by sorry
  have : new_bucket_count = 25 * (5 \ 2) := by sorry
  have : new_bucket_count = 62.5 := by sorry
  exact ceil_eq 63 _.mpr sorry

end buckets_required_l56_56900


namespace range_of_a_l56_56533

open Real

noncomputable def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + a > 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  let Δ := 1 - 4 * a
  Δ ≥ 0

theorem range_of_a (a : ℝ) :
  ((proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a))
  ↔ (a ≤ 0 ∨ (1/4 : ℝ) < a ∧ a < 4) :=
by
  sorry

end range_of_a_l56_56533


namespace gcd_72_168_gcd_98_280_f_at_3_l56_56906

/-- 
Prove that the GCD of 72 and 168 using the method of mutual subtraction is 24.
-/
theorem gcd_72_168 : Nat.gcd 72 168 = 24 :=
sorry

/-- 
Prove that the GCD of 98 and 280 using the Euclidean algorithm is 14.
-/
theorem gcd_98_280 : Nat.gcd 98 280 = 14 :=
sorry

/-- 
Prove that the value of f(3) where f(x) = x^5 + x^3 + x^2 + x + 1 is 283 using Horner's method.
-/
def f (x : ℕ) : ℕ := x^5 + x^3 + x^2 + x + 1

theorem f_at_3 : f 3 = 283 :=
sorry

end gcd_72_168_gcd_98_280_f_at_3_l56_56906


namespace num_prime_divisors_factorial_50_l56_56550

theorem num_prime_divisors_factorial_50 : 
  ∃ (n : ℕ), n = 15 ∧ 
  ∀ p, nat.prime p ∧ p ≤ 50 → (nat.factorial 50 % p = 0) :=
begin
  sorry
end

end num_prime_divisors_factorial_50_l56_56550


namespace positive_difference_even_odd_l56_56146

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l56_56146


namespace total_flowers_sold_l56_56221

theorem total_flowers_sold :
  let flowers_mon := 4
  let flowers_tue := 8
  let flowers_wed := flowers_mon + 3
  let flowers_thu := 6
  let flowers_fri := flowers_mon * 2
  let flowers_sat := 5 * 9
  flowers_mon + flowers_tue + flowers_wed + flowers_thu + flowers_fri + flowers_sat = 78 :=
by
  let flowers_mon := 4
  let flowers_tue := 8
  let flowers_wed := flowers_mon + 3
  let flowers_thu := 6
  let flowers_fri := flowers_mon * 2
  let flowers_sat := 5 * 9
  sorry

end total_flowers_sold_l56_56221


namespace cost_of_each_item_l56_56714

theorem cost_of_each_item (initial_order items : ℕ) (price per_item_reduction additional_orders : ℕ) (reduced_order total_order reduced_price profit_per_item : ℕ) 
  (h1 : initial_order = 60)
  (h2 : price = 100)
  (h3 : per_item_reduction = 1)
  (h4 : additional_orders = 3)
  (h5 : reduced_price = price - price * 4 / 100)
  (h6 : total_order = initial_order + additional_orders * (price * 4 / 100))
  (h7 : reduced_order = total_order)
  (h8 : profit_per_item = price - per_item_reduction )
  (h9 : profit_per_item = 24)
  (h10 : items * profit_per_item = reduced_order * (profit_per_item - per_item_reduction)) :
  (price - profit_per_item = 76) :=
by sorry

end cost_of_each_item_l56_56714


namespace sum_of_three_base4_numbers_l56_56817

theorem sum_of_three_base4_numbers :
  let a := "203_4".to_list.base_to_nat 4
  let b := "112_4".to_list.base_to_nat 4
  let c := "330_4".to_list.base_to_nat 4
  (a + b + c).nat_to_base 4 = "13110_4".to_list.base_to_nat 4 :=
by sorry

end sum_of_three_base4_numbers_l56_56817


namespace spent_on_new_tires_is_correct_l56_56408

-- Conditions
def amount_spent_on_speakers : ℝ := 136.01
def amount_spent_on_cd_player : ℝ := 139.38
def total_amount_spent : ℝ := 387.85

-- Goal
def amount_spent_on_tires : ℝ := total_amount_spent - (amount_spent_on_speakers + amount_spent_on_cd_player)

theorem spent_on_new_tires_is_correct : 
  amount_spent_on_tires = 112.46 :=
by
  sorry

end spent_on_new_tires_is_correct_l56_56408


namespace number_of_prime_divisors_of_50_factorial_l56_56549

theorem number_of_prime_divisors_of_50_factorial : 
  let primes_less_than_or_equal_to_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] in 
  primes_less_than_or_equal_to_50.length = 15 := 
by
  -- defining the list of prime numbers less than or equal to 50
  let primes_less_than_or_equal_to_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] 
  -- proving the length of this list is 15
  show primes_less_than_or_equal_to_50.length = 15
  sorry

end number_of_prime_divisors_of_50_factorial_l56_56549


namespace central_park_trash_cans_more_than_half_l56_56943

theorem central_park_trash_cans_more_than_half
  (C : ℕ)  -- Original number of trash cans in Central Park
  (V : ℕ := 24)  -- Original number of trash cans in Veteran's Park
  (V_now : ℕ := 34)  -- Number of trash cans in Veteran's Park after the move
  (H_move : (V_now - V) = C / 2)  -- Condition of trash cans moved
  (H_C : C = (1 / 2) * V + x)  -- Central Park had more than half trash cans as Veteran's Park, where x is an excess amount
  : C - (1 / 2) * V = 8 := 
sorry

end central_park_trash_cans_more_than_half_l56_56943


namespace find_f3_l56_56063

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 6

theorem find_f3 (a b c : ℝ) (h : f a b c (-3) = -12) : f a b c 3 = 24 :=
by
  sorry

end find_f3_l56_56063


namespace range_of_m_l56_56391

theorem range_of_m (m : ℝ) :
  (¬(∀ x y : ℝ, x^2 / (25 - m) + y^2 / (m - 7) = 1 → 25 - m > 0 ∧ m - 7 > 0 ∧ 25 - m > m - 7) ∨ 
   ¬(∀ x y : ℝ, y^2 / 5 - x^2 / m = 1 → 1 < (5 + m) / 5 ∧ (5 + m) / 5 < 4)) 
  → 7 < m ∧ m < 15 :=
by
  sorry

end range_of_m_l56_56391


namespace regular_decagon_interior_angle_degree_measure_l56_56604

theorem regular_decagon_interior_angle_degree_measure :
  ∀ (n : ℕ), n = 10 → (2 * 180 / n : ℝ) = 144 :=
by
  sorry

end regular_decagon_interior_angle_degree_measure_l56_56604


namespace sequence_inequality_l56_56298

theorem sequence_inequality 
  (a : ℕ → ℝ)
  (h_non_decreasing : ∀ i j : ℕ, i ≤ j → a i ≤ a j)
  (h_range : ∀ i, 1 ≤ i ∧ i ≤ 10 → a i = a (i - 1)) :
  (1 / 6) * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) ≤ (1 / 10) * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10) :=
by
  sorry

end sequence_inequality_l56_56298


namespace find_number_l56_56983

variable {x : ℝ}

theorem find_number (h : (30 / 100) * x = (40 / 100) * 40) : x = 160 / 3 :=
by
  sorry

end find_number_l56_56983


namespace jason_two_weeks_eggs_l56_56802

-- Definitions of given conditions
def eggs_per_omelet := 3
def days_per_week := 7
def weeks := 2

-- Statement to prove
theorem jason_two_weeks_eggs : (eggs_per_omelet * (days_per_week * weeks)) = 42 := by
  sorry

end jason_two_weeks_eggs_l56_56802


namespace math_problem_l56_56856

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48

theorem math_problem (a b c d : ℝ)
  (h1 : a + b + c + d = 6)
  (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
  proof_problem a b c d :=
by
  sorry

end math_problem_l56_56856


namespace calc_expression_solve_equation_l56_56625

-- Problem 1: Calculation

theorem calc_expression : 
  |Real.sqrt 3 - 2| + Real.sqrt 12 - 6 * Real.sin (Real.pi / 6) + (-1/2 : Real)⁻¹ = Real.sqrt 3 - 3 := 
by {
  sorry
}

-- Problem 2: Solve the Equation

theorem solve_equation (x : Real) : 
  x * (x + 6) = -5 ↔ (x = -5 ∨ x = -1) := 
by {
  sorry
}

end calc_expression_solve_equation_l56_56625


namespace xyz_value_l56_56521

theorem xyz_value
  (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19)
  (h3 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = 11) :
  x * y * z = 26 / 3 :=
sorry

end xyz_value_l56_56521


namespace part_a_part_b_l56_56704

def balanced (V : Finset (ℝ × ℝ)) : Prop :=
  ∀ (A B : ℝ × ℝ), A ∈ V → B ∈ V → A ≠ B → ∃ C : ℝ × ℝ, C ∈ V ∧ (dist C A = dist C B)

def center_free (V : Finset (ℝ × ℝ)) : Prop :=
  ¬ ∃ (A B C P : ℝ × ℝ), A ∈ V → B ∈ V → C ∈ V → P ∈ V →
                         A ≠ B ∧ B ≠ C ∧ A ≠ C →
                         (dist P A = dist P B ∧ dist P B = dist P C)

theorem part_a (n : ℕ) (hn : 3 ≤ n) :
  ∃ V : Finset (ℝ × ℝ), V.card = n ∧ balanced V :=
by sorry

theorem part_b : ∀ n : ℕ, 3 ≤ n →
  (∃ V : Finset (ℝ × ℝ), V.card = n ∧ balanced V ∧ center_free V ↔ n % 2 = 1) :=
by sorry

end part_a_part_b_l56_56704


namespace min_value_a_4b_l56_56515

theorem min_value_a_4b (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 / (a - 1) + 1 / (b - 1) = 1) : a + 4 * b = 14 := 
sorry

end min_value_a_4b_l56_56515


namespace expectation_fish_l56_56396

noncomputable def fish_distribution : ℕ → ℚ → ℚ → ℚ → ℚ :=
  fun N a b c => (a / b) * (1 - (c / (a + b + c) ^ N))

def x_distribution : ℚ := 0.18
def y_distribution : ℚ := 0.02
def other_distribution : ℚ := 0.80
def total_fish : ℕ := 10

theorem expectation_fish :
  fish_distribution total_fish x_distribution y_distribution other_distribution = 1.6461 :=
  by
    sorry

end expectation_fish_l56_56396


namespace positive_difference_even_odd_sum_l56_56184

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l56_56184


namespace integral_sin3_cos_l56_56655

open Real

theorem integral_sin3_cos :
  ∫ z in (π / 4)..(π / 2), sin z ^ 3 * cos z = 3 / 16 := by
  sorry

end integral_sin3_cos_l56_56655


namespace emily_sixth_quiz_score_l56_56799

theorem emily_sixth_quiz_score (a1 a2 a3 a4 a5 : ℕ) (target_mean : ℕ) (sixth_score : ℕ) :
  a1 = 94 ∧ a2 = 97 ∧ a3 = 88 ∧ a4 = 90 ∧ a5 = 102 ∧ target_mean = 95 →
  sixth_score = (target_mean * 6 - (a1 + a2 + a3 + a4 + a5)) →
  sixth_score = 99 :=
by
  sorry

end emily_sixth_quiz_score_l56_56799


namespace max_levels_prob_pass_three_levels_l56_56485

-- Definition of the conditions for part I
def pass_level (n : ℕ) : Prop := 6 * n > 2 ^ n

-- Part (I): Maximum number of levels a participant can win
theorem max_levels : ∃ n, n = 4 ∧ ∀ m, pass_level m → m ≤ 4 := by
  sorry

-- Probability calculations for part II
def die_prob : ℚ := 1 / 6
def prob_pass_level1 : ℚ := 4 / 6
def prob_pass_level2 : ℚ :=
  (30 : ℚ) / (36 : ℚ)
def prob_pass_level3 : ℚ :=
  (120 : ℚ) / (216 : ℚ)

-- Part (II): Probability of passing the first three levels
theorem prob_pass_three_levels :
  prob_pass_level1 * prob_pass_level2 * prob_pass_level3 = 100 / 243 := by
  sorry

end max_levels_prob_pass_three_levels_l56_56485


namespace segment_parametrization_pqrs_l56_56444

theorem segment_parametrization_pqrs :
  ∃ (p q r s : ℤ), 
    q = 1 ∧ 
    s = -3 ∧ 
    p + q = 6 ∧ 
    r + s = 4 ∧ 
    p^2 + q^2 + r^2 + s^2 = 84 :=
by
  use 5, 1, 7, -3
  sorry

end segment_parametrization_pqrs_l56_56444


namespace ellipse_standard_equation_parabola_standard_equation_l56_56043

-- Ellipse with major axis length 10 and eccentricity 4/5
theorem ellipse_standard_equation (a c b : ℝ) (h₀ : a = 5) (h₁ : c = 4) (h₂ : b = 3) :
  (x^2 / a^2) + (y^2 / b^2) = 1 := by sorry

-- Parabola with vertex at the origin and directrix y = 2
theorem parabola_standard_equation (p : ℝ) (h₀ : p = 4) :
  x^2 = -8 * y := by sorry

end ellipse_standard_equation_parabola_standard_equation_l56_56043


namespace positive_difference_even_odd_l56_56150

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l56_56150


namespace shark_ratio_l56_56647

theorem shark_ratio (N D : ℕ) (h1 : N = 22) (h2 : D + N = 110) (h3 : ∃ x : ℕ, D = x * N) : 
  (D / N) = 4 :=
by
  -- conditions use only definitions given in the problem.
  sorry

end shark_ratio_l56_56647


namespace positive_difference_even_odd_sum_l56_56183

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l56_56183


namespace nonnegative_integers_with_abs_value_less_than_4_l56_56239

theorem nonnegative_integers_with_abs_value_less_than_4 :
  {n : ℕ | abs (n : ℤ) < 4} = {0, 1, 2, 3} :=
by {
  sorry
}

end nonnegative_integers_with_abs_value_less_than_4_l56_56239


namespace num_prime_divisors_of_50_factorial_l56_56546

/-- The number of prime positive divisors of 50! is 15. -/
theorem num_prime_divisors_of_50_factorial : 
  (finset.filter nat.prime (finset.range 51)).card = 15 := 
by
  sorry

end num_prime_divisors_of_50_factorial_l56_56546


namespace digit_d_multiple_of_9_l56_56382

theorem digit_d_multiple_of_9 (d : ℕ) (hd : d = 1) : ∃ k : ℕ, (56780 + d) = 9 * k := by
  have : 56780 + d = 56780 + 1 := by rw [hd]
  rw [this]
  use 6313
  sorry

end digit_d_multiple_of_9_l56_56382


namespace positive_diff_even_odd_sums_l56_56131

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l56_56131


namespace ratio_of_roots_l56_56662

theorem ratio_of_roots 
  (a b c : ℝ) 
  (h : a * b * c ≠ 0)
  (x1 x2 : ℝ) 
  (root1 : x1 = 2022 * x2) 
  (root2 : a * x1 ^ 2 + b * x1 + c = 0) 
  (root3 : a * x2 ^ 2 + b * x2 + c = 0) : 
  2023 * a * c / b ^ 2 = 2022 / 2023 :=
by
  sorry

end ratio_of_roots_l56_56662


namespace replace_movies_cost_l56_56283

theorem replace_movies_cost
  (num_movies : ℕ)
  (trade_in_value_per_vhs : ℕ)
  (cost_per_dvd : ℕ)
  (h1 : num_movies = 100)
  (h2 : trade_in_value_per_vhs = 2)
  (h3 : cost_per_dvd = 10):
  (cost_per_dvd - trade_in_value_per_vhs) * num_movies = 800 :=
by sorry

end replace_movies_cost_l56_56283


namespace intersection_complement_eq_singleton_l56_56260

open Set

def U : Set ℤ := {-1, 0, 1, 2, 3, 4}
def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}
def CU_A : Set ℤ := U \ A

theorem intersection_complement_eq_singleton : B ∩ CU_A = {0} := 
by
  sorry

end intersection_complement_eq_singleton_l56_56260


namespace tan_alpha_half_l56_56971

theorem tan_alpha_half (α: ℝ) (h: Real.tan α = 1/2) :
  (1 + 2 * Real.sin (Real.pi - α) * Real.cos (-2 * Real.pi - α)) / (Real.sin (-α)^2 - Real.sin (5 * Real.pi / 2 - α)^2) = -3 := 
by
  sorry

end tan_alpha_half_l56_56971


namespace milan_rate_per_minute_l56_56818

-- Definitions based on the conditions
def monthly_fee : ℝ := 2.0
def total_bill : ℝ := 23.36
def total_minutes : ℕ := 178
def expected_rate_per_minute : ℝ := 0.12

-- Theorem statement based on the question
theorem milan_rate_per_minute :
  (total_bill - monthly_fee) / total_minutes = expected_rate_per_minute := 
by 
  sorry

end milan_rate_per_minute_l56_56818


namespace sequence_increasing_l56_56814

theorem sequence_increasing (a : ℕ → ℝ) (a0 : ℝ) (h0 : a 0 = 1 / 5)
  (H : ∀ n : ℕ, a (n + 1) = 2^n - 3 * a n) :
  ∀ n : ℕ, a (n + 1) > a n :=
sorry

end sequence_increasing_l56_56814


namespace find_x_l56_56513

theorem find_x (x : ℝ) : (x = 2 ∨ x = -2) ↔ (|x|^2 - 5 * |x| + 6 = 0 ∧ x^2 - 4 = 0) :=
by
  sorry

end find_x_l56_56513


namespace minimum_value_k_eq_2_l56_56517

noncomputable def quadratic_function_min (a m k : ℝ) (h : 0 < a) : ℝ :=
  a * (-(k / 2)) * (-(k / 2) - k)

theorem minimum_value_k_eq_2 (a m : ℝ) (h : 0 < a) :
  quadratic_function_min a m 2 h = -a := 
by
  unfold quadratic_function_min
  sorry

end minimum_value_k_eq_2_l56_56517


namespace positive_difference_even_odd_sums_l56_56159

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l56_56159


namespace positive_difference_even_odd_sums_l56_56169

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l56_56169


namespace distance_between_towns_l56_56905

theorem distance_between_towns 
  (rate1 rate2 : ℕ) (time : ℕ) (distance : ℕ)
  (h_rate1 : rate1 = 48)
  (h_rate2 : rate2 = 42)
  (h_time : time = 5)
  (h_distance : distance = rate1 * time + rate2 * time) : 
  distance = 450 :=
by
  sorry

end distance_between_towns_l56_56905


namespace range_of_a_h_diff_l56_56530

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x

theorem range_of_a (a : ℝ) (h : a < 0) : (∀ x, 0 < x ∧ x < Real.log 3 → 
  (a * x - 1) / x < 0 ∧ Real.exp x + a ≠ 0 ∧ (a ≤ -3)) :=
sorry

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + Real.log x

theorem h_diff (a : ℝ) (x1 x2 : ℝ) (hx1 : 0 < x1 ∧ x1 < 1/2) : 
    x1 * x2 = 1/2 ∧ h a x1 - h a x2 > 3/4 - Real.log 2 :=
sorry

end range_of_a_h_diff_l56_56530


namespace simplify_fraction_l56_56299

theorem simplify_fraction :
  (175 / 1225) * 25 = 25 / 7 :=
by
  -- Code to indicate proof steps would go here.
  sorry

end simplify_fraction_l56_56299


namespace division_of_field_l56_56773

theorem division_of_field :
  (∀ (hectares : ℕ) (parts : ℕ), hectares = 5 ∧ parts = 8 →
  (1 / parts = 1 / 8) ∧ (hectares / parts = 5 / 8)) :=
by
  sorry


end division_of_field_l56_56773


namespace avg_age_boys_class_l56_56067

-- Definitions based on conditions
def avg_age_students : ℝ := 15.8
def avg_age_girls : ℝ := 15.4
def ratio_boys_girls : ℝ := 1.0000000000000044

-- Using the given conditions to define the average age of boys
theorem avg_age_boys_class (B G : ℕ) (A_b : ℝ) 
  (h1 : avg_age_students = (B * A_b + G * avg_age_girls) / (B + G)) 
  (h2 : B = ratio_boys_girls * G) : 
  A_b = 16.2 :=
  sorry

end avg_age_boys_class_l56_56067


namespace fraction_left_handed_l56_56941

def total_participants (k : ℕ) := 15 * k

def red (k : ℕ) := 5 * k
def blue (k : ℕ) := 5 * k
def green (k : ℕ) := 3 * k
def yellow (k : ℕ) := 2 * k

def left_handed_red (k : ℕ) := (1 / 3) * red k
def left_handed_blue (k : ℕ) := (2 / 3) * blue k
def left_handed_green (k : ℕ) := (1 / 2) * green k
def left_handed_yellow (k : ℕ) := (1 / 4) * yellow k

def total_left_handed (k : ℕ) := left_handed_red k + left_handed_blue k + left_handed_green k + left_handed_yellow k

theorem fraction_left_handed (k : ℕ) : 
  (total_left_handed k) / (total_participants k) = 7 / 15 := 
sorry

end fraction_left_handed_l56_56941


namespace sum_of_T_is_101110000_l56_56855

-- The set T consists of all positive integers with five digits in base 2
def T : Finset ℕ := Finset.Ico 16 32

-- Prove the sum of all elements in T is equal to 376 in decimal which is 101110000 in binary
theorem sum_of_T_is_101110000 :
  (∑ x in T, x) = 376 := by
  sorry

end sum_of_T_is_101110000_l56_56855


namespace total_students_correct_l56_56895

-- Define the given conditions
variables (A B C : ℕ)

-- Number of students in class B
def B_def : ℕ := 25

-- Number of students in class A (B is 8 fewer than A)
def A_def : ℕ := B_def + 8

-- Number of students in class C (C is 5 times B)
def C_def : ℕ := 5 * B_def

-- The total number of students
def total_students : ℕ := A_def + B_def + C_def

-- The proof statement
theorem total_students_correct : total_students = 183 := by
  sorry

end total_students_correct_l56_56895


namespace circle_sector_radius_l56_56351

theorem circle_sector_radius (r : ℝ) :
  (2 * r + (r * (Real.pi / 3)) = 144) → r = 432 / (6 + Real.pi) := by
  sorry

end circle_sector_radius_l56_56351


namespace probability_A_not_losing_l56_56276

theorem probability_A_not_losing 
  (P_A_tie P_A_win : ℚ)
  (h_A_tie : P_A_tie = 1/2) 
  (h_A_win : P_A_win = 1/3) :
  P_A_tie + P_A_win = 5 / 6 :=
by
  rw [h_A_tie, h_A_win]
  norm_num
  sorry

end probability_A_not_losing_l56_56276


namespace probability_of_event_a_l56_56825

-- Given conditions and question
variables (a b : Prop)
variables (p : Prop → ℝ)

-- Given conditions
axiom p_a : p a = 4 / 5
axiom p_b : p b = 2 / 5
axiom p_a_and_b_given : p (a ∧ b) = 0.32
axiom independent_a_b : p (a ∧ b) = p a * p b

-- The proof statement we need to prove: p a = 0.8
theorem probability_of_event_a :
  p a = 0.8 :=
sorry

end probability_of_event_a_l56_56825


namespace min_value_of_f_l56_56659

open Real

noncomputable def f (x : ℝ) := x + 1 / (x - 2)

theorem min_value_of_f : ∃ x : ℝ, x > 2 ∧ ∀ y : ℝ, y > 2 → f y ≥ f 3 := by
  sorry

end min_value_of_f_l56_56659


namespace number_of_students_l56_56506

theorem number_of_students (y c r n : ℕ) (h1 : y = 730) (h2 : c = 17) (h3 : r = 16) :
  y - r = n * c ↔ n = 42 :=
by
  have h4 : 730 - 16 = 714 := by norm_num
  have h5 : 714 / 17 = 42 := by norm_num
  sorry

end number_of_students_l56_56506


namespace integer_solutions_range_l56_56829

theorem integer_solutions_range (a : ℝ) :
  (∀ x : ℤ, x^2 - x + a - a^2 < 0 → x + 2 * a > 1) ↔ 1 < a ∧ a ≤ 2 := sorry

end integer_solutions_range_l56_56829


namespace train_crossing_time_l56_56324

-- Conditions
def length_train1 : ℕ := 200 -- Train 1 length in meters
def length_train2 : ℕ := 160 -- Train 2 length in meters
def speed_train1 : ℕ := 68 -- Train 1 speed in kmph
def speed_train2 : ℕ := 40 -- Train 2 speed in kmph

-- Conversion factors and formulas
def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600
def total_distance (l1 l2 : ℕ) := l1 + l2
def relative_speed (s1 s2 : ℕ) := kmph_to_mps (s1 + s2)
def crossing_time (dist speed : ℕ) := dist / speed

-- Proof statement
theorem train_crossing_time : 
  crossing_time (total_distance length_train1 length_train2) (relative_speed speed_train1 speed_train2) = 12 := by sorry

end train_crossing_time_l56_56324


namespace opposite_of_neg_twelve_l56_56595

def opposite (n : Int) : Int := -n

theorem opposite_of_neg_twelve : opposite (-12) = 12 := by
  sorry

end opposite_of_neg_twelve_l56_56595


namespace perfect_square_condition_l56_56720

-- Definitions from conditions
def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

-- Theorem statement
theorem perfect_square_condition (n : ℤ) (h1 : 0 < n) (h2 : is_integer (2 + 2 * Real.sqrt (1 + 12 * (n: ℝ)^2))) : 
  is_perfect_square n :=
by
  sorry

end perfect_square_condition_l56_56720


namespace y_gets_per_rupee_l56_56019

theorem y_gets_per_rupee (a p : ℝ) (ha : a * p = 63) (htotal : p + a * p + 0.3 * p = 245) : a = 0.63 :=
by
  sorry

end y_gets_per_rupee_l56_56019


namespace overlapping_region_area_l56_56112

noncomputable def radius : ℝ := 15
noncomputable def central_angle_radians : ℝ := Real.pi / 2
noncomputable def area_of_sector : ℝ := (1 / 4) * Real.pi * (radius^2)
noncomputable def side_length_equilateral_triangle : ℝ := radius
noncomputable def area_of_equilateral_triangle : ℝ := (Real.sqrt 3 / 4) * (side_length_equilateral_triangle^2)
noncomputable def overlapping_area : ℝ := 2 * area_of_sector - area_of_equilateral_triangle

theorem overlapping_region_area :
  overlapping_area = 112.5 * Real.pi - 56.25 * Real.sqrt 3 :=
by
  sorry
 
end overlapping_region_area_l56_56112


namespace solve_for_x_l56_56439

theorem solve_for_x (x : ℕ) : 8 * 4^x = 2048 → x = 4 := by
  sorry

end solve_for_x_l56_56439


namespace roger_cookie_price_l56_56789

open Classical

theorem roger_cookie_price
  (art_base1 art_base2 art_height : ℕ) 
  (art_cookies_per_batch art_cookie_price roger_cookies_per_batch : ℕ)
  (art_area : ℕ := (art_base1 + art_base2) * art_height / 2)
  (total_dough : ℕ := art_cookies_per_batch * art_area)
  (roger_area : ℚ := total_dough / roger_cookies_per_batch)
  (art_total_earnings : ℚ := art_cookies_per_batch * art_cookie_price) :
  ∀ (roger_cookie_price : ℚ), roger_cookies_per_batch * roger_cookie_price = art_total_earnings →
  roger_cookie_price = 100 / 3 :=
sorry

end roger_cookie_price_l56_56789


namespace factorize_expression_l56_56038

theorem factorize_expression (x : ℝ) : 4 * x^2 - 4 = 4 * (x + 1) * (x - 1) := 
  sorry

end factorize_expression_l56_56038


namespace heartsuit_example_l56_56980

def heartsuit (x y: ℤ) : ℤ := 4 * x + 6 * y

theorem heartsuit_example : heartsuit 3 8 = 60 :=
by
  sorry

end heartsuit_example_l56_56980


namespace arithmetic_sequence_eighth_term_l56_56898

theorem arithmetic_sequence_eighth_term (a d : ℚ) 
  (h1 : 6 * a + 15 * d = 21) 
  (h2 : a + 6 * d = 8) : 
  a + 7 * d = 9 + 2/7 :=
by
  sorry

end arithmetic_sequence_eighth_term_l56_56898


namespace cube_faces_sum_39_l56_56234

theorem cube_faces_sum_39 (a b c d e f g h : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0)
    (vertex_sum : (a*e*b*h + a*e*c*h + a*f*b*h + a*f*c*h + d*e*b*h + d*e*c*h + d*f*b*h + d*f*c*h) = 2002) :
    (a + b + c + d + e + f + g + h) = 39 := 
sorry

end cube_faces_sum_39_l56_56234


namespace not_constant_expression_l56_56291

noncomputable def is_centroid (A B C G : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

noncomputable def squared_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem not_constant_expression (A B C P G : ℝ × ℝ)
  (hG : is_centroid A B C G)
  (hP_on_AB : ∃ x, P = (x, A.2) ∧ A.2 = B.2) :
  ∃ dPA dPB dPC dPG : ℝ,
    dPA = squared_distance P A ∧
    dPB = squared_distance P B ∧
    dPC = squared_distance P C ∧
    dPG = squared_distance P G ∧
    (dPA + dPB + dPC - dPG) ≠ dPA + dPB + dPC - dPG := by
  sorry

end not_constant_expression_l56_56291


namespace coin_flip_probability_difference_l56_56743

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l56_56743


namespace acres_used_for_corn_l56_56483

-- Define the conditions
def total_acres : ℝ := 5746
def ratio_beans : ℝ := 7.5
def ratio_wheat : ℝ := 3.2
def ratio_corn : ℝ := 5.6
def total_parts : ℝ := ratio_beans + ratio_wheat + ratio_corn

-- Define the statement to prove
theorem acres_used_for_corn : (total_acres / total_parts) * ratio_corn = 1975.46 :=
by
  -- Placeholder for the proof; to be completed separately
  sorry

end acres_used_for_corn_l56_56483


namespace minimum_value_fraction_l56_56833

theorem minimum_value_fraction (a : ℝ) (h : a > 1) : (a^2 - a + 1) / (a - 1) ≥ 3 :=
by
  sorry

end minimum_value_fraction_l56_56833


namespace comparison_abc_l56_56056

variable (f : Real → Real)
variable (a b c : Real)
variable (x : Real)
variable (h_even : ∀ x, f (-x + 1) = f (x + 1))
variable (h_periodic : ∀ x, f (x + 2) = f x)
variable (h_mono : ∀ x y, 0 < x ∧ y < 1 ∧ x < y → f x < f y)
variable (h_f0 : f 0 = 0)
variable (a_def : a = f (Real.log 2))
variable (b_def : b = f (Real.log 3))
variable (c_def : c = f 0.5)

theorem comparison_abc : b > a ∧ a > c :=
sorry

end comparison_abc_l56_56056


namespace pow_mod_26_l56_56910

theorem pow_mod_26 (a b n : ℕ) (hn : n = 2023) (h₁ : a = 17) (h₂ : b = 26) :
  a ^ n % b = 7 := by
  sorry

end pow_mod_26_l56_56910


namespace number_of_boxes_in_each_case_l56_56706

theorem number_of_boxes_in_each_case (a b : ℕ) :
    a + b = 2 → 9 = a * 8 + b :=
by
    intro h
    sorry

end number_of_boxes_in_each_case_l56_56706


namespace area_ratio_DFE_ABEF_l56_56672

noncomputable def parallelogram := 
(0, 0) ∧ (2, 3) ∧ (5, 3) ∧ (3, 0)

theorem area_ratio_DFE_ABEF : 
  let A := (0 : ℝ, 0 : ℝ),
      B := (2 : ℝ, 3 : ℝ),
      C := (5 : ℝ, 3 : ℝ),
      D := (3 : ℝ, 0 : ℝ),
      E := ((0 + 5) / 2, (0 + 3) / 2),
      F := (1, 0) in
  (2 * |3 * (0 - 2.5) + 1 * (2.5 - 0) + 1.5 * (0 - 0)| / 2) /
  ((|2 * (0 - 2.5) + 1.5 * (2.5 - 3)| / 2 + |1 * (0 - 2.5)| / 2) +  |2 * (3 - 0) + 1.5 * (0 - -0)| / 2) = 1.5 := 
by sorry

end area_ratio_DFE_ABEF_l56_56672


namespace maximum_sum_of_triplets_l56_56715

-- Define a list representing a 9-digit number consisting of digits 1 to 9 in some order
def valid_digits (digits : List ℕ) : Prop :=
  digits.length = 9 ∧ ∀ n, n ∈ digits → n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]
  
def sum_of_triplets (digits : List ℕ) : ℕ :=
  100 * digits[0]! + 10 * digits[1]! + digits[2]! +
  100 * digits[1]! + 10 * digits[2]! + digits[3]! +
  100 * digits[2]! + 10 * digits[3]! + digits[4]! +
  100 * digits[3]! + 10 * digits[4]! + digits[5]! +
  100 * digits[4]! + 10 * digits[5]! + digits[6]! +
  100 * digits[5]! + 10 * digits[6]! + digits[7]! +
  100 * digits[6]! + 10 * digits[7]! + digits[8]!

theorem maximum_sum_of_triplets :
  ∃ digits : List ℕ, valid_digits digits ∧ sum_of_triplets digits = 4648 :=
  sorry

end maximum_sum_of_triplets_l56_56715


namespace range_of_a_l56_56393

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, a ≤ x ∧ (x : ℝ) < 2 → x = -1 ∨ x = 0 ∨ x = 1) ↔ (-2 < a ∧ a ≤ -1) :=
by
  sorry

end range_of_a_l56_56393


namespace initial_puppies_l56_56487

-- Define the conditions
variable (a : ℕ) (t : ℕ) (p_added : ℕ) (p_total_adopted : ℕ)

-- State the theorem with the conditions and the target proof
theorem initial_puppies
  (h₁ : a = 3) 
  (h₂ : t = 2)
  (h₃ : p_added = 3)
  (h₄ : p_total_adopted = a * t) :
  (p_total_adopted - p_added) = 3 :=
sorry

end initial_puppies_l56_56487


namespace range_of_a_l56_56253

variable (f : ℝ → ℝ) (a : ℝ)

-- Definitions based on provided conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonically_increasing (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x ≤ f y

-- Main statement
theorem range_of_a
    (hf_even : is_even f)
    (hf_mono : is_monotonically_increasing f)
    (h_ineq : ∀ x : ℝ, f (Real.log (a) / Real.log 2) ≤ f (x^2 - 2 * x + 2)) :
  (1/2 : ℝ) ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l56_56253


namespace cattle_transport_problem_l56_56781

noncomputable def truck_capacity 
    (total_cattle : ℕ)
    (distance_one_way : ℕ)
    (speed : ℕ)
    (total_time : ℕ) : ℕ :=
  total_cattle / (total_time / ((distance_one_way * 2) / speed))

theorem cattle_transport_problem :
  truck_capacity 400 60 60 40 = 20 := by
  -- The theorem statement follows the structure from the conditions and question
  sorry

end cattle_transport_problem_l56_56781


namespace simplify_expression_l56_56882

theorem simplify_expression : 18 * (8 / 15) * (1 / 12) = 4 / 5 :=
by
  sorry

end simplify_expression_l56_56882


namespace average_percentage_l56_56982

theorem average_percentage (s1 s2 : ℕ) (a1 a2 : ℕ) (n : ℕ)
  (h1 : s1 = 15) (h2 : a1 = 70) (h3 : s2 = 10) (h4 : a2 = 90) (h5 : n = 25)
  : ((s1 * a1 + s2 * a2) / n : ℕ) = 78 :=
by
  -- We include sorry to skip the proof part.
  sorry

end average_percentage_l56_56982


namespace teddy_has_8_cats_l56_56587

theorem teddy_has_8_cats (dogs_teddy : ℕ) (cats_teddy : ℕ) (dogs_total : ℕ) (pets_total : ℕ)
  (h1 : dogs_teddy = 7)
  (h2 : dogs_total = dogs_teddy + (dogs_teddy + 9) + (dogs_teddy - 5))
  (h3 : pets_total = dogs_total + cats_teddy + (cats_teddy + 13))
  (h4 : pets_total = 54) :
  cats_teddy = 8 := by
  sorry

end teddy_has_8_cats_l56_56587


namespace Telegraph_Road_length_is_162_l56_56442

-- Definitions based on the conditions
def meters_to_kilometers (meters : ℕ) : ℕ := meters / 1000
def Pardee_Road_length_meters : ℕ := 12000
def Telegraph_Road_extra_length_kilometers : ℕ := 150

-- The length of Pardee Road in kilometers
def Pardee_Road_length_kilometers : ℕ := meters_to_kilometers Pardee_Road_length_meters

-- Lean statement to prove the length of Telegraph Road in kilometers
theorem Telegraph_Road_length_is_162 :
  Pardee_Road_length_kilometers + Telegraph_Road_extra_length_kilometers = 162 :=
sorry

end Telegraph_Road_length_is_162_l56_56442


namespace jessica_needs_stamps_l56_56566

-- Define the weights and conditions
def weight_of_paper := 1 / 5
def total_papers := 8
def weight_of_envelope := 2 / 5
def stamps_per_ounce := 1

-- Calculate the total weight and determine the number of stamps needed
theorem jessica_needs_stamps : 
  total_papers * weight_of_paper + weight_of_envelope = 2 :=
by
  sorry

end jessica_needs_stamps_l56_56566


namespace sleep_hours_for_desired_average_l56_56077

theorem sleep_hours_for_desired_average 
  (s_1 s_2 : ℝ) (h_1 h_2 : ℝ) (k : ℝ) 
  (h_inverse_relation : ∀ s h, s * h = k)
  (h_s1 : s_1 = 75)
  (h_h1 : h_1 = 6)
  (h_average : (s_1 + s_2) / 2 = 85) : 
  h_2 = 450 / 95 := 
by 
  sorry

end sleep_hours_for_desired_average_l56_56077


namespace cos_of_angle_l56_56048

theorem cos_of_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.cos (3 * Real.pi / 2 + 2 * θ) = 3 / 5 := 
by
  sorry

end cos_of_angle_l56_56048


namespace irwins_family_hike_total_distance_l56_56399

theorem irwins_family_hike_total_distance
    (d1 d2 d3 : ℝ)
    (h1 : d1 = 0.2)
    (h2 : d2 = 0.4)
    (h3 : d3 = 0.1)
    :
    d1 + d2 + d3 = 0.7 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end irwins_family_hike_total_distance_l56_56399


namespace find_functions_l56_56959

open Function

theorem find_functions (f g : ℚ → ℚ) :
  (∀ x y : ℚ, f (g x - g y) = f (g x) - y) →
  (∀ x y : ℚ, g (f x - f y) = g (f x) - y) →
  ∃ c : ℚ, c ≠ 0 ∧ (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) :=
by
  sorry

end find_functions_l56_56959


namespace distance_between_Petrovo_and_Nikolaevo_l56_56574

theorem distance_between_Petrovo_and_Nikolaevo :
  ∃ S : ℝ, (10 + (S - 10) / 4) + (20 + (S - 20) / 3) = S ∧ S = 50 := by
    sorry

end distance_between_Petrovo_and_Nikolaevo_l56_56574


namespace train_speed_l56_56020

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 300) (h_time : time = 15) : 
  (length / time) * 3.6 = 72 :=
by
  sorry

end train_speed_l56_56020


namespace line_translation_upwards_units_l56_56065

theorem line_translation_upwards_units:
  ∀ (x : ℝ), (y = x / 3) → (y = (x + 5) / 3) → (y' = y + 5 / 3) :=
by
  sorry

end line_translation_upwards_units_l56_56065


namespace distribution_schemes_l56_56505

-- Define the variables and conditions
def num_students : Nat := 5
def group_A : Set Nat := {1, 2, 3, 4, 5}
def at_least_two (s : Set Nat) := s.card >= 2
def at_least_one (s : Set Nat) := s.card >= 1
def group_B : Set Nat := {1, 2, 3, 4, 5}
def group_C : Set Nat := {1, 2, 3, 4, 5}

-- Final statement to prove (only the statement, without proof)
theorem distribution_schemes :
  let distrib : Finset (Finset Nat) := {s | s.card = 2 ∧ at_least_two s} ∪ {s | s.card = 3 ∧ at_least_two s}
  let distrib_B_C : Finset (Finset Nat × Finset Nat) := 
    { (s1, s2) | (s1 ∪ s2 = {3, 4, 5}) ∧ at_least_one s1 ∧ at_least_one s2 }
  distrib.card * distrib_B_C.card = 80 :=
sorry

end distribution_schemes_l56_56505


namespace negation_of_existential_l56_56448

theorem negation_of_existential :
  (¬ ∃ (x : ℝ), x^2 + x + 1 < 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_l56_56448


namespace inequality_proof_l56_56433

-- Define the inequality problem in Lean 4
theorem inequality_proof (x y : ℝ) (h1 : x ≠ -1) (h2 : y ≠ -1) (h3 : x * y = 1) : 
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 := 
by
  sorry

end inequality_proof_l56_56433


namespace compute_sum_of_products_of_coefficients_l56_56081

theorem compute_sum_of_products_of_coefficients (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : ∀ x : ℝ, (x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) =
    (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4)) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by
  -- Proof would go here
  sorry

end compute_sum_of_products_of_coefficients_l56_56081


namespace biased_die_probability_l56_56010

theorem biased_die_probability (P2 : ℝ) (h1 : P2 ≠ 1 / 6) (h2 : 3 * P2 * (1 - P2) ^ 2 = 1 / 4) : 
  P2 = 0.211 :=
sorry

end biased_die_probability_l56_56010


namespace max_constant_k_l56_56270

theorem max_constant_k (x y : ℤ) : 4 * x^2 + y^2 + 1 ≥ 3 * x * (y + 1) :=
sorry

end max_constant_k_l56_56270


namespace positive_difference_eq_250_l56_56116

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l56_56116


namespace add_A_to_10_eq_15_l56_56992

theorem add_A_to_10_eq_15 (A : ℕ) (h : A + 10 = 15) : A = 5 :=
sorry

end add_A_to_10_eq_15_l56_56992


namespace minimum_common_ratio_l56_56677

theorem minimum_common_ratio (a : ℕ) (n : ℕ) (q : ℝ) (h_pos : ∀ i, i < n → 0 < a * q^i) (h_geom : ∀ i j, i < j → a * q^i < a * q^j) (h_q : 1 < q ∧ q < 2) : q = 6 / 5 :=
by
  sorry

end minimum_common_ratio_l56_56677


namespace weight_of_daughter_l56_56103

def mother_daughter_grandchild_weight (M D C : ℝ) :=
  M + D + C = 130 ∧
  D + C = 60 ∧
  C = 1/5 * M

theorem weight_of_daughter (M D C : ℝ) 
  (h : mother_daughter_grandchild_weight M D C) : D = 46 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end weight_of_daughter_l56_56103


namespace find_c_l56_56703

structure ProblemData where
  (r : ℝ → ℝ)
  (s : ℝ → ℝ)
  (h : r (s 3) = 20)

def r (x : ℝ) : ℝ := 5 * x - 10
def s (x : ℝ) (c : ℝ) : ℝ := 4 * x - c

theorem find_c (c : ℝ) (h : (r (s 3 c)) = 20) : c = 6 :=
sorry

end find_c_l56_56703


namespace probability_of_number_between_21_and_30_l56_56859

-- Define the success condition of forming a two-digit number between 21 and 30.
def successful_number (d1 d2 : Nat) : Prop :=
  let n1 := 10 * d1 + d2
  let n2 := 10 * d2 + d1
  (21 ≤ n1 ∧ n1 ≤ 30) ∨ (21 ≤ n2 ∧ n2 ≤ 30)

-- Calculate the probability of a successful outcome.
def probability_success (favorable total : Nat) : Nat :=
  favorable / total

-- The main theorem claiming the probability that Melinda forms a number between 21 and 30.
theorem probability_of_number_between_21_and_30 :
  let successful_counts := 10
  let total_possible := 36
  probability_success successful_counts total_possible = 5 / 18 :=
by
  sorry

end probability_of_number_between_21_and_30_l56_56859


namespace minimum_familiar_pairs_l56_56202

open Finset

-- Define the set of students and the relationship of familiarity
variable (students : Finset ℕ)
variable (n : ℕ := 175)
variable (familiar : ℕ → ℕ → Prop)

-- Assumption: students set has 175 members
axiom student_count : students.card = n

-- Assumption: familiarity is symmetric
axiom familiar_symm (a b : ℕ) : familiar a b → familiar b a

-- Assumption: familiarity within any group of six
axiom familiar_in_groups_of_six (s : Finset ℕ) (h₁ : s.card = 6) :
  ∃ t₁ t₂ : Finset ℕ, t₁.card = 3 ∧ t₂.card = 3 ∧ (∀ x ∈ t₁, ∀ y ∈ t₁, x ≠ y → familiar x y) ∧
  (∀ x ∈ t₂, ∀ y ∈ t₂, x ≠ y → familiar x y) ∧ t₁ ∪ t₂ = s ∧ t₁ ∩ t₂ = ∅

-- Theorem: minimum number of familiar pairs
theorem minimum_familiar_pairs :
  ∃ k : ℕ, (∑ a in students, (students.filter (familiar a)).card) / 2 ≥ 15050 :=
sorry

end minimum_familiar_pairs_l56_56202


namespace positive_difference_even_odd_sums_l56_56163

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l56_56163


namespace number_of_students_at_end_of_year_l56_56996

def students_at_start_of_year : ℕ := 35
def students_left_during_year : ℕ := 10
def students_joined_during_year : ℕ := 10

theorem number_of_students_at_end_of_year : students_at_start_of_year - students_left_during_year + students_joined_during_year = 35 :=
by
  sorry -- Proof goes here

end number_of_students_at_end_of_year_l56_56996


namespace Neil_candy_collected_l56_56536

variable (M H N : ℕ)

-- Conditions
def Maggie_collected := M = 50
def Harper_collected := H = M + (30 * M) / 100
def Neil_collected := N = H + (40 * H) / 100

-- Theorem statement 
theorem Neil_candy_collected
  (hM : Maggie_collected M)
  (hH : Harper_collected M H)
  (hN : Neil_collected H N) :
  N = 91 := by
  sorry

end Neil_candy_collected_l56_56536


namespace angle_c_in_triangle_l56_56398

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A/B = 1/3) (h3 : A/C = 1/5) : C = 100 :=
by
  sorry

end angle_c_in_triangle_l56_56398


namespace solution_set_inequalities_l56_56105

theorem solution_set_inequalities (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 2 * x) / 3 > x - 1) → (x ≤ 1) :=
by
  intros h
  sorry

end solution_set_inequalities_l56_56105


namespace identify_linear_equation_l56_56613

def is_linear_equation (eq : String) : Prop := sorry

theorem identify_linear_equation :
  is_linear_equation "2x = 0" ∧ ¬is_linear_equation "x^2 - 4x = 3" ∧ ¬is_linear_equation "x + 2y = 1" ∧ ¬is_linear_equation "x - 1 = 1 / x" :=
by 
  sorry

end identify_linear_equation_l56_56613


namespace positive_difference_even_odd_sums_l56_56160

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l56_56160


namespace inequality_proof_l56_56525

variable (a b c : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : 0 < c)

theorem inequality_proof :
  (2 * a + b + c)^2 / (2 * a^2 + (b + c)^2) +
  (a + 2 * b + c)^2 / (2 * b^2 + (c + a)^2) +
  (a + b + 2 * c)^2 / (2 * c^2 + (a + b)^2) ≤ 8 := sorry

end inequality_proof_l56_56525


namespace tangents_product_is_constant_MN_passes_fixed_point_l56_56251

-- Define the parabola C and the tangency conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

variables {x1 y1 x2 y2 : ℝ}

-- Point G is on the axis of the parabola C (we choose the y-axis for part 2)
def point_G_on_axis (G : ℝ × ℝ) : Prop := G.2 = -1

-- Two tangent points from G to the parabola at A (x1, y1) and B (x2, y2)
def tangent_points (G : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂

-- Question 1 proof statement
theorem tangents_product_is_constant (G : ℝ × ℝ) (hG : point_G_on_axis G)
  (hT : tangent_points G x1 y1 x2 y2) : x1 * x2 + y1 * y2 = -3 := sorry

variables {M N : ℝ × ℝ}

-- Question 2 proof statement
theorem MN_passes_fixed_point {G : ℝ × ℝ} (hG : G.1 = 0) (xM yM xN yN : ℝ)
 (hMA : parabola M.1 M.2) (hMB : parabola N.1 N.2)
 (h_perpendicular : (M.1 - G.1) * (N.1 - G.1) + (M.2 - G.2) * (N.2 - G.2) = 0)
 : ∃ P, P = (2, 5) := sorry

end tangents_product_is_constant_MN_passes_fixed_point_l56_56251


namespace max_min_f_m1_possible_ns_l56_56535

noncomputable def f (a b : ℝ) (x : ℝ) (m : ℝ) : ℝ :=
  let a := (Real.sqrt 2 * Real.sin (Real.pi / 4 + m * x), -Real.sqrt 3)
  let b := (Real.sqrt 2 * Real.sin (Real.pi / 4 + m * x), Real.cos (2 * m * x))
  a.1 * b.1 + a.2 * b.2

theorem max_min_f_m1 (x : ℝ) (h₁ : x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) :
  2 ≤ f (Real.sqrt 2) 1 x 1 ∧ f (Real.sqrt 2) 1 x 1 ≤ 3 :=
by
  sorry

theorem possible_ns (n : ℤ) (h₂ : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 2017) ∧ f (Real.sqrt 2) ((n * Real.pi) / 2) x ((n * Real.pi) / 2) = 0) :
  n = 1 ∨ n = -1 :=
by
  sorry

end max_min_f_m1_possible_ns_l56_56535


namespace intermediate_circle_radius_l56_56998

theorem intermediate_circle_radius (r1 r3: ℝ) (h1: r1 = 5) (h2: r3 = 13) 
  (h3: π * r1 ^ 2 = π * r3 ^ 2 - π * r2 ^ 2) : r2 = 12 := sorry


end intermediate_circle_radius_l56_56998


namespace number_of_terms_ap_l56_56107

variables (a d n : ℤ) 

def sum_of_first_thirteen_terms := (13 / 2) * (2 * a + 12 * d)
def sum_of_last_thirteen_terms := (13 / 2) * (2 * a + (2 * n - 14) * d)

def sum_excluding_first_three := ((n - 3) / 2) * (2 * a + (n - 4) * d)
def sum_excluding_last_three := ((n - 3) / 2) * (2 * a + (n - 1) * d)

theorem number_of_terms_ap (h1 : sum_of_first_thirteen_terms a d = (1 / 2) * sum_of_last_thirteen_terms a d)
  (h2 : sum_excluding_first_three a d / sum_excluding_last_three a d = 5 / 4) : n = 22 :=
sorry

end number_of_terms_ap_l56_56107


namespace problem_solution_l56_56302

theorem problem_solution (x : ℝ) (h : x - 29 = 63) : (x - 47 = 45) :=
by
  sorry

end problem_solution_l56_56302


namespace solve_for_xy_l56_56978

theorem solve_for_xy (x y : ℕ) : 
  (4^x / 2^(x + y) = 16) ∧ (9^(x + y) / 3^(5 * y) = 81) → x * y = 32 :=
by
  sorry

end solve_for_xy_l56_56978


namespace scott_sold_40_cups_of_smoothies_l56_56580

theorem scott_sold_40_cups_of_smoothies
  (cost_smoothie : ℕ)
  (cost_cake : ℕ)
  (num_cakes : ℕ)
  (total_revenue : ℕ)
  (h1 : cost_smoothie = 3)
  (h2 : cost_cake = 2)
  (h3 : num_cakes = 18)
  (h4 : total_revenue = 156) :
  ∃ x : ℕ, (cost_smoothie * x + cost_cake * num_cakes = total_revenue ∧ x = 40) := 
sorry

end scott_sold_40_cups_of_smoothies_l56_56580


namespace probability_at_least_three_prime_dice_l56_56364

-- Definitions from the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def p := 5 / 12
def q := 7 / 12
def binomial (n k : ℕ) := Nat.choose n k

-- The probability of at least three primes
theorem probability_at_least_three_prime_dice :
  (binomial 5 3 * p ^ 3 * q ^ 2) +
  (binomial 5 4 * p ^ 4 * q ^ 1) +
  (binomial 5 5 * p ^ 5 * q ^ 0) = 40625 / 622080 :=
by
  sorry

end probability_at_least_three_prime_dice_l56_56364


namespace probability_of_sum_being_6_l56_56756

noncomputable def prob_sum_6 : ℚ :=
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_of_sum_being_6 :
  prob_sum_6 = 5 / 36 :=
by
  sorry

end probability_of_sum_being_6_l56_56756


namespace probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l56_56245

noncomputable def total_outcomes : ℕ := Nat.choose 6 2

noncomputable def prob_both_boys : ℚ := (Nat.choose 4 2 : ℚ) / total_outcomes

noncomputable def prob_exactly_one_girl : ℚ := ((Nat.choose 4 1) * (Nat.choose 2 1) : ℚ) / total_outcomes

noncomputable def prob_at_least_one_girl : ℚ := 1 - prob_both_boys

theorem probability_both_boys : prob_both_boys = 2 / 5 := by sorry
theorem probability_exactly_one_girl : prob_exactly_one_girl = 8 / 15 := by sorry
theorem probability_at_least_one_girl : prob_at_least_one_girl = 3 / 5 := by sorry

end probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l56_56245


namespace age_difference_is_100_l56_56731

-- Definition of the ages
variables {X Y Z : ℕ}

-- Conditions from the problem statement
axiom age_condition1 : X + Y > Y + Z
axiom age_condition2 : Z = X - 100

-- Proof to show the difference is 100 years
theorem age_difference_is_100 : (X + Y) - (Y + Z) = 100 :=
by sorry

end age_difference_is_100_l56_56731


namespace cost_of_ingredients_l56_56480

theorem cost_of_ingredients :
  let popcorn_earnings := 50
  let cotton_candy_earnings := 3 * popcorn_earnings
  let total_earnings_per_day := popcorn_earnings + cotton_candy_earnings
  let total_earnings := total_earnings_per_day * 5
  let rent := 30
  let earnings_after_rent := total_earnings - rent
  earnings_after_rent - 895 = 75 :=
by
  let popcorn_earnings := 50
  let cotton_candy_earnings := 3 * popcorn_earnings
  let total_earnings_per_day := popcorn_earnings + cotton_candy_earnings
  let total_earnings := total_earnings_per_day * 5
  let rent := 30
  let earnings_after_rent := total_earnings - rent
  show earnings_after_rent - 895 = 75
  sorry

end cost_of_ingredients_l56_56480


namespace Tim_sweets_are_multiple_of_4_l56_56454

-- Define the conditions
def sweets_are_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Given definitions
def Peter_sweets : ℕ := 44
def largest_possible_number_per_tray : ℕ := 4

-- Define the proposition to be proven
theorem Tim_sweets_are_multiple_of_4 (O : ℕ) (h1 : sweets_are_divisible_by_4 Peter_sweets) (h2 : sweets_are_divisible_by_4 largest_possible_number_per_tray) :
  sweets_are_divisible_by_4 O :=
sorry

end Tim_sweets_are_multiple_of_4_l56_56454


namespace correct_completion_of_sentence_l56_56305

def committee_discussing_problem : Prop := True -- Placeholder for the condition
def problem_expected_to_be_solved_next_week : Prop := True -- Placeholder for the condition

theorem correct_completion_of_sentence 
  (h1 : committee_discussing_problem) 
  (h2 : problem_expected_to_be_solved_next_week) 
  : "hopefully" = "hopefully" :=
by 
  sorry

end correct_completion_of_sentence_l56_56305


namespace amount_of_sugar_l56_56017

-- Let ratio_sugar_flour be the ratio of sugar to flour.
def ratio_sugar_flour : ℕ := 10

-- Let flour be the amount of flour used in ounces.
def flour : ℕ := 5

-- Let sugar be the amount of sugar used in ounces.
def sugar (ratio_sugar_flour : ℕ) (flour : ℕ) : ℕ := ratio_sugar_flour * flour

-- The proof goal: given the conditions, prove that the amount of sugar used is 50 ounces.
theorem amount_of_sugar (h_ratio : ratio_sugar_flour = 10) (h_flour : flour = 5) : sugar ratio_sugar_flour flour = 50 :=
by
  -- Proof omitted.
  sorry
 
end amount_of_sugar_l56_56017


namespace number_of_valid_pairs_l56_56661

-- Definitions based on conditions
def isValidPair (x y : ℕ) : Prop := 
  (1 ≤ x ∧ x ≤ 1000) ∧ (1 ≤ y ∧ y ≤ 1000) ∧ (x^2 + y^2) % 5 = 0

def countValidPairs : ℕ := 
  (Finset.range 1000).filter (λ x, (x + 1) % 5 = 0 ∨ (x + 1) % 5 = 1 ∨ (x + 1) % 5 = 4).card *
  (Finset.range 1000).filter (λ y, (y + 1) % 5 = 0 ∨ (y + 1) % 5 = 1 ∨ (y + 1) % 5 = 4).card +
  2 * (
    (Finset.range 1000).filter (λ x, (x + 1) % 5 = 1).card *
    (Finset.range 1000).filter (λ y, (y + 1) % 5 = 4).card *
    2
  )

theorem number_of_valid_pairs : countValidPairs = 200000 := by
  sorry

end number_of_valid_pairs_l56_56661


namespace binomial_ξ_properties_l56_56518

-- Define the binomial random variable ξ with parameters n = 10 and p = 0.6
def ξ : RandomVariable := Binomial 10 0.6

-- State the theorem to prove the expected value and variance of ξ
theorem binomial_ξ_properties : (E ξ = 6) ∧ (D ξ = 2.4) := by
  sorry

end binomial_ξ_properties_l56_56518


namespace total_hotdogs_sold_l56_56484

theorem total_hotdogs_sold : 
  let small := 58.3
  let medium := 21.7
  let large := 35.9
  let extra_large := 15.4
  small + medium + large + extra_large = 131.3 :=
by 
  sorry

end total_hotdogs_sold_l56_56484


namespace intersection_product_l56_56278

noncomputable def line_l (t : ℝ) := (1 + (Real.sqrt 3 / 2) * t, 1 + (1/2) * t)

def curve_C (x y : ℝ) : Prop := y^2 = 8 * x

theorem intersection_product :
  ∀ (t1 t2 : ℝ), 
  (1 + (1/2) * t1)^2 = 8 * (1 + (Real.sqrt 3 / 2) * t1) →
  (1 + (1/2) * t2)^2 = 8 * (1 + (Real.sqrt 3 / 2) * t2) →
  (1 + (1/2) * t1) * (1 + (1/2) * t2) = 28 := 
  sorry

end intersection_product_l56_56278


namespace AB_ratio_CD_l56_56496

variable (AB CD : ℝ)
variable (h : ℝ)
variable (O : Point)
variable (ABCD_isosceles : IsIsoscelesTrapezoid AB CD)
variable (areas_condition : List ℝ) 
-- where the list areas_condition represents: [S_OCD, S_OBC, S_OAB, S_ODA]

theorem AB_ratio_CD : 
  ABCD_isosceles ∧ areas_condition = [2, 3, 4, 5] → AB = 2 * CD :=
by
  sorry

end AB_ratio_CD_l56_56496


namespace number_of_fish_initially_tagged_l56_56274

theorem number_of_fish_initially_tagged {N T : ℕ}
  (hN : N = 1250)
  (h_ratio : 2 / 50 = T / N) :
  T = 50 :=
by
  sorry

end number_of_fish_initially_tagged_l56_56274


namespace perfect_square_transformation_l56_56192

theorem perfect_square_transformation (a : ℤ) :
  (∃ x y : ℤ, x^2 + a = y^2) ↔ 
  ∃ α β : ℤ, α * β = a ∧ (α % 2 = β % 2) ∧ 
  ∃ x y : ℤ, x = (β - α) / 2 ∧ y = (β + α) / 2 :=
by
  sorry

end perfect_square_transformation_l56_56192


namespace compute_ab_l56_56307

theorem compute_ab (a b : ℝ)
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 867.75 := 
by
  sorry

end compute_ab_l56_56307


namespace largest_triangle_angle_l56_56443

theorem largest_triangle_angle (h_ratio : ∃ (a b c : ℕ), a / b = 3 / 4 ∧ b / c = 4 / 9) 
  (h_external_angle : ∃ (θ1 θ2 θ3 θ4 : ℝ), θ1 = 3 * x ∧ θ2 = 4 * x ∧ θ3 = 9 * x ∧ θ4 = 3 * x ∧ θ1 + θ2 + θ3 = 180) :
  ∃ (θ3 : ℝ), θ3 = 101.25 := by
  sorry

end largest_triangle_angle_l56_56443


namespace Damien_jogs_miles_over_three_weeks_l56_56797

theorem Damien_jogs_miles_over_three_weeks :
  (5 * 5) * 3 = 75 :=
by sorry

end Damien_jogs_miles_over_three_weeks_l56_56797


namespace remainder_modulo_seven_l56_56514

theorem remainder_modulo_seven (n : ℕ)
  (h₁ : n^2 % 7 = 1)
  (h₂ : n^3 % 7 = 6) :
  n % 7 = 6 :=
sorry

end remainder_modulo_seven_l56_56514


namespace coin_probability_difference_l56_56748

theorem coin_probability_difference :
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  p3 - p4 = (7/16 : ℚ) :=
by
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  have h1 : p3 = (1/2 : ℚ), by norm_num [finset.range_succ]
  have h2 : p4 = (1/16 : ℚ), by norm_num
  rw [h1, h2]
  norm_num

end coin_probability_difference_l56_56748


namespace num_prime_divisors_50_fact_l56_56539

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to (n : ℕ) : list ℕ :=
(list.range (n + 1)).filter is_prime

theorem num_prime_divisors_50_fact : 
  (primes_up_to 50).length = 15 := 
by 
  sorry

end num_prime_divisors_50_fact_l56_56539


namespace Megan_not_lead_plays_l56_56423

def total_plays : ℕ := 100
def lead_percentage : ℝ := 0.80
def lead_plays : ℕ := (total_plays : ℝ * lead_percentage).toNat
def not_lead_plays : ℕ := total_plays - lead_plays

theorem Megan_not_lead_plays : not_lead_plays = 20 := by
  sorry

end Megan_not_lead_plays_l56_56423


namespace find_coordinates_of_B_l56_56822

-- Define points A and B, and vector a
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 5 }
def a : Point := { x := 2, y := 3 }

-- Define the proof problem
theorem find_coordinates_of_B (B : Point) 
  (h1 : B.x + 1 = 3 * a.x)
  (h2 : B.y - 5 = 3 * a.y) : 
  B = { x := 5, y := 14 } := 
sorry

end find_coordinates_of_B_l56_56822


namespace range_of_a_l56_56053

theorem range_of_a (a : ℝ) 
  (p : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a ≥ Real.exp x) 
  (q : ∃ x : ℝ, x^2 - 4 * x + a ≤ 0) : 
  e ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l56_56053


namespace muffins_apples_l56_56776

def apples_left_for_muffins (total_apples : ℕ) (pie_apples : ℕ) (refrigerator_apples : ℕ) : ℕ :=
  total_apples - (pie_apples + refrigerator_apples)

theorem muffins_apples (total_apples pie_apples refrigerator_apples : ℕ) (h_total : total_apples = 62) (h_pie : pie_apples = total_apples / 2) (h_refrigerator : refrigerator_apples = 25) : apples_left_for_muffins total_apples pie_apples refrigerator_apples = 6 := 
by 
  sorry

end muffins_apples_l56_56776


namespace a_gt_b_l56_56664

theorem a_gt_b (n : ℕ) (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hn_ge_two : n ≥ 2)
  (ha_eq : a^n = a + 1) (hb_eq : b^(2*n) = b + 3 * a) : a > b :=
by
  sorry

end a_gt_b_l56_56664


namespace sum_W_Y_eq_seven_l56_56653

theorem sum_W_Y_eq_seven :
  ∃ (W X Y Z : ℕ), W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
  W ∈ {1, 2, 3, 4} ∧ X ∈ {1, 2, 3, 4} ∧ Y ∈ {1, 2, 3, 4} ∧ Z ∈ {1, 2, 3, 4} ∧
  (W / X : ℚ) - (Y / Z : ℚ) = 1 ∧ W + Y = 7 :=
by
  sorry

end sum_W_Y_eq_seven_l56_56653


namespace positive_difference_l56_56154

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l56_56154


namespace cake_eaten_fraction_l56_56615

noncomputable def cake_eaten_after_four_trips : ℚ :=
  let consumption_ratio := (1/3 : ℚ)
  let first_trip := consumption_ratio
  let second_trip := consumption_ratio * consumption_ratio
  let third_trip := second_trip * consumption_ratio
  let fourth_trip := third_trip * consumption_ratio
  first_trip + second_trip + third_trip + fourth_trip

theorem cake_eaten_fraction : cake_eaten_after_four_trips = (40 / 81 : ℚ) :=
by
  sorry

end cake_eaten_fraction_l56_56615


namespace minimum_value_is_one_l56_56951

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  (1 / (3 * a + 2)) + (1 / (3 * b + 2)) + (1 / (3 * c + 2))

theorem minimum_value_is_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  minimum_value a b c = 1 := by
  sorry

end minimum_value_is_one_l56_56951


namespace find_y_l56_56526

theorem find_y 
  (α : Real)
  (P : Real × Real)
  (P_coord : P = (-Real.sqrt 3, y))
  (sin_alpha : Real.sin α = Real.sqrt 13 / 13) :
  P.2 = 1 / 2 :=
by
  sorry

end find_y_l56_56526


namespace range_of_a_l56_56841

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| > a) → a < 3 :=
by
  sorry

end range_of_a_l56_56841


namespace profit_percentage_is_25_l56_56494

variable (CP MP : ℝ) (d : ℝ)

/-- Given an article with a cost price of Rs. 85.5, a marked price of Rs. 112.5, 
    and a 5% discount on the marked price, the profit percentage on the cost 
    price is 25%. -/
theorem profit_percentage_is_25
  (hCP : CP = 85.5)
  (hMP : MP = 112.5)
  (hd : d = 0.05) :
  ((MP - (MP * d) - CP) / CP * 100) = 25 := 
sorry

end profit_percentage_is_25_l56_56494


namespace total_arrangements_l56_56778

-- Defining the selection and arrangement problem conditions
def select_and_arrange (n m : ℕ) : ℕ :=
  Nat.choose n m * Nat.factorial m

-- Specifying the specific problem's constraints and results
theorem total_arrangements : select_and_arrange 8 2 * select_and_arrange 6 2 = 60 := by
  -- Proof omitted
  sorry

end total_arrangements_l56_56778


namespace find_initial_marbles_l56_56614

def initial_marbles (W Y H : ℕ) : Prop :=
  (W + 2 = 20) ∧ (Y - 5 = 20) ∧ (H + 3 = 20)

theorem find_initial_marbles (W Y H : ℕ) (h : initial_marbles W Y H) : W = 18 :=
  by
    sorry

end find_initial_marbles_l56_56614


namespace math_proof_problem_l56_56042

noncomputable def problem_statement : Prop :=
  ∃ (x : ℝ), (x > 12) ∧ ((x - 5) / 12 = 5 / (x - 12)) ∧ (x = 17)

theorem math_proof_problem : problem_statement :=
by
  sorry

end math_proof_problem_l56_56042


namespace total_players_l56_56764

theorem total_players 
  (cricket_players : ℕ) (hockey_players : ℕ)
  (football_players : ℕ) (softball_players : ℕ)
  (h_cricket : cricket_players = 12)
  (h_hockey : hockey_players = 17)
  (h_football : football_players = 11)
  (h_softball : softball_players = 10)
  : cricket_players + hockey_players + football_players + softball_players = 50 :=
by sorry

end total_players_l56_56764


namespace simplify_fraction_l56_56879

theorem simplify_fraction : ∃ (a b : ℕ), a = 90 ∧ b = 150 ∧ (90:ℚ) / (150:ℚ) = (3:ℚ) / (5:ℚ) :=
by {
  use 90,
  use 150,
  split,
  refl,
  split,
  refl,
  sorry,
}

end simplify_fraction_l56_56879


namespace coin_flip_probability_difference_l56_56747

theorem coin_flip_probability_difference :
  let p := 1 / 2,
  p_3 := (Nat.choose 4 3) * (p ^ 3) * (p ^ 1),
  p_4 := p ^ 4
  in abs (p_3 - p_4) = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l56_56747


namespace different_tens_digit_probability_l56_56440

noncomputable def probability_diff_tens_digit : ℚ :=
  1000000 / 15890700

theorem different_tens_digit_probability :
  let selected : Finset ℕ := Finset.range (59 + 1) \ Finset.range 10 in
  let total_combinations := (selected.card).choose 6 in
  let favorable_combinations := 10^6 in
  (favorable_combinations : ℚ) / (total_combinations : ℚ) = probability_diff_tens_digit :=
by
  sorry

end different_tens_digit_probability_l56_56440


namespace sticks_problem_solution_l56_56428

theorem sticks_problem_solution :
  ∃ n : ℕ, n > 0 ∧ 1012 = 2 * n * (n + 1) ∧ 1012 > 1000 ∧ 
           1012 % 3 = 1 ∧ 1012 % 5 = 2 :=
by
  sorry

end sticks_problem_solution_l56_56428


namespace simplify_fraction_l56_56874

theorem simplify_fraction (num denom : ℕ) (h_num : num = 90) (h_denom : denom = 150) : 
  num / denom = 3 / 5 := by
  rw [h_num, h_denom]
  norm_num
  sorry

end simplify_fraction_l56_56874


namespace sufficient_not_necessary_condition_l56_56201

theorem sufficient_not_necessary_condition (x : ℝ) : (1 < x ∧ x < 2) → (x < 2) ∧ ((x < 2) → ¬(1 < x ∧ x < 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l56_56201


namespace find_30_cent_items_l56_56400

-- Define the parameters and their constraints
variables (a d b c : ℕ)

-- Define the conditions
def total_items : Prop := a + d + b + c = 50
def total_cost : Prop := 30 * a + 150 * d + 200 * b + 300 * c = 6000

-- The theorem to prove the number of 30-cent items purchased
theorem find_30_cent_items (h1 : total_items a d b c) (h2 : total_cost a d b c) : 
  ∃ a, a + d + b + c = 50 ∧ 30 * a + 150 * d + 200 * b + 300 * c = 6000 := 
sorry

end find_30_cent_items_l56_56400


namespace columbian_coffee_price_is_correct_l56_56402

-- Definitions based on the conditions
def total_mix_weight : ℝ := 100
def brazilian_coffee_price_per_pound : ℝ := 3.75
def final_mix_price_per_pound : ℝ := 6.35
def columbian_coffee_weight : ℝ := 52

-- Let C be the price per pound of the Columbian coffee
noncomputable def columbian_coffee_price_per_pound : ℝ := sorry

-- Define the Lean 4 proof problem
theorem columbian_coffee_price_is_correct :
  columbian_coffee_price_per_pound = 8.75 :=
by
  -- Total weight and calculation based on conditions
  let brazilian_coffee_weight := total_mix_weight - columbian_coffee_weight
  let total_value_of_columbian := columbian_coffee_weight * columbian_coffee_price_per_pound
  let total_value_of_brazilian := brazilian_coffee_weight * brazilian_coffee_price_per_pound
  let total_value_of_mix := total_mix_weight * final_mix_price_per_pound
  
  -- Main equation based on the mix
  have main_eq : total_value_of_columbian + total_value_of_brazilian = total_value_of_mix :=
    by sorry

  -- Solve for C (columbian coffee price per pound)
  sorry

end columbian_coffee_price_is_correct_l56_56402


namespace repeating_decimal_fraction_eq_l56_56035

-- Define repeating decimal and its equivalent fraction
def repeating_decimal_value : ℚ := 7 + 123 / 999

theorem repeating_decimal_fraction_eq :
  repeating_decimal_value = 2372 / 333 :=
by
  sorry

end repeating_decimal_fraction_eq_l56_56035


namespace find_value_l56_56981

theorem find_value (x y z : ℝ) (h₁ : y = 3 * x) (h₂ : z = 3 * y + x) : x + y + z = 14 * x :=
by
  sorry

end find_value_l56_56981


namespace positive_difference_probability_l56_56744

theorem positive_difference_probability (fair_coin : Prop) (four_flips : ℕ) (fair : fair_coin) (flips : four_flips = 4) :
  let p1 := (Nat.choose 4 3)*((1/2)^3)*((1/2)^1) in
  let p2 := (1/2)^4 in
  abs (p1 - p2) = 7/16 :=
by
  -- Definitions
  let p1 := (Nat.choose 4 3) * ((1/2)^3) * ((1/2)^1)
  let p2 := (1/2)^4
  have h : abs (p1 - p2) = 7 / 16
  sorry

end positive_difference_probability_l56_56744


namespace probability_blue_or_purple_l56_56629

def total_jelly_beans : ℕ := 35
def blue_jelly_beans : ℕ := 7
def purple_jelly_beans : ℕ := 10

theorem probability_blue_or_purple : (blue_jelly_beans + purple_jelly_beans: ℚ) / total_jelly_beans = 17 / 35 := 
by sorry

end probability_blue_or_purple_l56_56629


namespace probability_P_plus_S_is_two_less_than_multiple_of_7_l56_56113

def is_distinct (a b : ℕ) : Prop :=
  a ≠ b

def in_range (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100

def mod_condition (a b : ℕ) : Prop :=
  (a * b + a + b) % 7 = 5

noncomputable def probability_p_s (p q : ℕ) : ℚ :=
  p / q

theorem probability_P_plus_S_is_two_less_than_multiple_of_7 :
  probability_p_s (1295) (4950) = 259 / 990 := 
sorry

end probability_P_plus_S_is_two_less_than_multiple_of_7_l56_56113


namespace largest_angle_of_convex_hexagon_l56_56446

theorem largest_angle_of_convex_hexagon (a d : ℕ) (h_seq : ∀ i, a + i * d < 180 ∧ a + i * d > 0)
  (h_sum : 6 * a + 15 * d = 720)
  (h_seq_arithmetic : ∀ (i j : ℕ), (a + i * d) < (a + j * d) ↔ i < j) :
  ∃ m : ℕ, (m = a + 5 * d ∧ m = 175) :=
by
  sorry

end largest_angle_of_convex_hexagon_l56_56446


namespace combination_sum_l56_56361

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Given conditions
axiom combinatorial_identity (n r : ℕ) : combination n r + combination n (r + 1) = combination (n + 1) (r + 1)

-- The theorem we aim to prove
theorem combination_sum : combination 8 2 + combination 8 3 + combination 9 2 = 120 := 
by
  sorry

end combination_sum_l56_56361


namespace snowball_game_l56_56459

theorem snowball_game (x y z : ℕ) (h : 5 * x + 4 * y + 3 * z = 12) : 
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end snowball_game_l56_56459


namespace mean_home_runs_l56_56728

-- Declaring the given conditions as variables
variables (n1 n2 n3 n4 n5 : ℕ)
variables (h1 h2 h3 h4 h5 : ℕ)

-- Assigning the given values from the problem conditions
def players_1 := 7
def runs_1 := 5

def players_2 := 5
def runs_2 := 6

def players_3 := 4
def runs_3 := 8

def players_4 := 2
def runs_4 := 9

def players_5 := 1
def runs_5 := 11

-- Using these variables to state our theorem
theorem mean_home_runs : 
  ( (players_1 * runs_1) + (players_2 * runs_2) + (players_3 * runs_3) + (players_4 * runs_4) + (players_5 * runs_5) )
  / 
  (players_1 + players_2 + players_3 + players_4 + players_5) = 126 / 19 :=
sorry

end mean_home_runs_l56_56728


namespace impossible_arrangement_of_300_numbers_in_circle_l56_56770

theorem impossible_arrangement_of_300_numbers_in_circle :
  ¬ ∃ (nums : Fin 300 → ℕ), (∀ i : Fin 300, nums i > 0) ∧
    ∃ unique_exception : Fin 300,
      ∀ i : Fin 300, i ≠ unique_exception → nums i = Int.natAbs (nums (Fin.mod (i.val - 1) 300) - nums (Fin.mod (i.val + 1) 300)) := 
sorry

end impossible_arrangement_of_300_numbers_in_circle_l56_56770


namespace carpet_area_in_yards_l56_56078

def main_length_feet : ℕ := 15
def main_width_feet : ℕ := 12
def extension_length_feet : ℕ := 6
def extension_width_feet : ℕ := 5
def feet_per_yard : ℕ := 3

def main_length_yards : ℕ := main_length_feet / feet_per_yard
def main_width_yards : ℕ := main_width_feet / feet_per_yard
def extension_length_yards : ℕ := extension_length_feet / feet_per_yard
def extension_width_yards : ℕ := extension_width_feet / feet_per_yard

def main_area_yards : ℕ := main_length_yards * main_width_yards
def extension_area_yards : ℕ := extension_length_yards * extension_width_yards

theorem carpet_area_in_yards : (main_area_yards : ℚ) + (extension_area_yards : ℚ) = 23.33 := 
by
  apply sorry

end carpet_area_in_yards_l56_56078


namespace perfect_square_trinomial_m6_l56_56990

theorem perfect_square_trinomial_m6 (m : ℚ) (h₁ : 0 < m) (h₂ : ∃ a : ℚ, x^2 - 2 * m * x + 36 = (x - a)^2) : m = 6 :=
sorry

end perfect_square_trinomial_m6_l56_56990


namespace positive_difference_of_sums_l56_56135

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l56_56135


namespace multiple_of_9_digit_l56_56379

theorem multiple_of_9_digit :
  ∃ d : ℕ, d < 10 ∧ (5 + 6 + 7 + 8 + d) % 9 = 0 ∧ d = 1 :=
by
  sorry

end multiple_of_9_digit_l56_56379


namespace solution_k_values_l56_56658

theorem solution_k_values (k : ℕ) : 
  (∃ m n : ℕ, 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) 
  → k = 1 ∨ 4 ≤ k := 
by
  sorry

end solution_k_values_l56_56658


namespace sum_six_smallest_multiples_of_eleven_l56_56757

theorem sum_six_smallest_multiples_of_eleven : 
  (11 + 22 + 33 + 44 + 55 + 66) = 231 :=
by
  sorry

end sum_six_smallest_multiples_of_eleven_l56_56757


namespace Nora_to_Lulu_savings_ratio_l56_56586

-- Definitions
def L : ℕ := 6
def T (N : ℕ) : Prop := N = 3 * (N / 3)
def total_savings (N : ℕ) : Prop := 6 + N + (N / 3) = 46

-- Theorem statement
theorem Nora_to_Lulu_savings_ratio (N : ℕ) (hN_T : T N) (h_total_savings : total_savings N) :
  N / L = 5 :=
by
  -- Proof will be provided here
  sorry

end Nora_to_Lulu_savings_ratio_l56_56586


namespace largest_number_Ahn_can_get_l56_56935

theorem largest_number_Ahn_can_get :
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (100 ≤ m ∧ m ≤ 999) → 3 * (500 - m) ≤ 1200) := sorry

end largest_number_Ahn_can_get_l56_56935


namespace multiple_of_9_digit_l56_56380

theorem multiple_of_9_digit :
  ∃ d : ℕ, d < 10 ∧ (5 + 6 + 7 + 8 + d) % 9 = 0 ∧ d = 1 :=
by
  sorry

end multiple_of_9_digit_l56_56380


namespace determine_distance_l56_56994

noncomputable def distance_formula (d a b c : ℝ) : Prop :=
  (d / a = (d - 30) / b) ∧
  (d / b = (d - 15) / c) ∧
  (d / a = (d - 40) / c)

theorem determine_distance (d a b c : ℝ) (h : distance_formula d a b c) : d = 90 :=
by {
  sorry
}

end determine_distance_l56_56994


namespace perimeter_of_region_l56_56929

theorem perimeter_of_region : 
  let side := 1
  let diameter := side
  let radius := diameter / 2
  let full_circumference := 2 * Real.pi * radius
  let arc_length := (3 / 4) * full_circumference
  let total_arcs := 4
  let perimeter := total_arcs * arc_length
  perimeter = 3 * Real.pi :=
by 
  sorry

end perimeter_of_region_l56_56929


namespace factorize_a_squared_plus_2a_l56_56808

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2*a = a * (a + 2) :=
sorry

end factorize_a_squared_plus_2a_l56_56808


namespace Megan_not_lead_plays_l56_56426

-- Define the problem's conditions as variables
def total_plays : ℕ := 100
def lead_play_ratio : ℤ := 80

-- Define the proposition we want to prove
theorem Megan_not_lead_plays : 
  (total_plays - (total_plays * lead_play_ratio / 100)) = 20 := 
by sorry

end Megan_not_lead_plays_l56_56426


namespace will_initially_bought_seven_boxes_l56_56193

theorem will_initially_bought_seven_boxes :
  let given_away_pieces := 3 * 4
  let total_initial_pieces := given_away_pieces + 16
  let initial_boxes := total_initial_pieces / 4
  initial_boxes = 7 := 
by
  sorry

end will_initially_bought_seven_boxes_l56_56193


namespace independence_of_xi_and_zeta_l56_56570

noncomputable theory
open MeasureTheory ProbabilityTheory

variables {Ω : Type*} {ξ ζ : Ω → ℝ}
variables (μ : Measure Ω) [IsProbabilityMeasure μ]

/-- Statement of the problem translated to Lean 4 as a Theorem -/
theorem independence_of_xi_and_zeta
  (bounded ξ : ∃ C_ξ, ∀ ω, |ξ ω| ≤ C_ξ)
  (bounded ζ : ∃ C_ζ, ∀ ω, |ζ ω| ≤ C_ζ)
  (cond : ∀ (m n : ℕ), Expectation[ξ ^ m * ζ ^ n] = Expectation[ξ ^ m] * Expectation[ζ ^ n]) :
  IndepFun ξ ζ μ :=
sorry

end independence_of_xi_and_zeta_l56_56570


namespace team_A_match_win_probability_l56_56848

def probability_A_game_win : ℚ := 2/3

theorem team_A_match_win_probability :
  let P_A := probability_A_game_win,
      P_match := P_A * P_A + 2 * P_A * (1 - P_A) * P_A
  in P_match = 20 / 27 := by
  sorry

end team_A_match_win_probability_l56_56848


namespace gcd_117_182_l56_56512

theorem gcd_117_182 : Int.gcd 117 182 = 13 := 
by 
  sorry

end gcd_117_182_l56_56512


namespace number_of_people_l56_56347

-- Conditions
def cost_oysters : ℤ := 3 * 15
def cost_shrimp : ℤ := 2 * 14
def cost_clams : ℤ := 2 * 135 / 10  -- Using integers for better precision
def total_cost : ℤ := cost_oysters + cost_shrimp + cost_clams
def amount_owed_each_person : ℤ := 25

-- Goal
theorem number_of_people (number_of_people : ℤ) : total_cost = number_of_people * amount_owed_each_person → number_of_people = 4 := by
  -- Proof to be completed here.
  sorry

end number_of_people_l56_56347


namespace other_x_intercept_l56_56965

theorem other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, y = a * x ^ 2 + b * x + c → (x, y) = (4, -3)) (h_x_intercept : ∀ y, y = a * 1 ^ 2 + b * 1 + c → (1, y) = (1, 0)) : 
  ∃ x, x = 7 := by
sorry

end other_x_intercept_l56_56965


namespace problem_1_problem_2_l56_56571

open Set

variables (a x : ℝ)

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a ^ 2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem problem_1 (a : ℝ) (ha : a = 1) : 
  {x : ℝ | x^2 - 4 * a * x + 3 * a ^ 2 < 0} ∩ {x : ℝ | (x - 3) / (x - 2) ≤ 0} = Ioo 2 3 :=
sorry

theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 4 * a * x + 3 * a ^ 2 < 0) → ¬((x - 3) / (x - 2) ≤ 0)) →
  (∃ x : ℝ, ¬((x - 3) / (x - 2) ≤ 0) → ¬(x^2 - 4 * a * x + 3 * a ^ 2 < 0)) →
  1 < a ∧ a ≤ 2 :=
sorry

end problem_1_problem_2_l56_56571


namespace find_f_prime_zero_l56_56528

noncomputable def f (a : ℝ) (fd0 : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 + x - 1) * Real.exp x + fd0

theorem find_f_prime_zero (a fd0 : ℝ) : (deriv (f a fd0) 0 = 0) :=
by
  -- the proof would go here
  sorry

end find_f_prime_zero_l56_56528


namespace Jane_age_proof_l56_56447

theorem Jane_age_proof (D J : ℕ) (h1 : D + 6 = (J + 6) / 2) (h2 : D + 14 = 25) : J = 28 :=
by
  sorry

end Jane_age_proof_l56_56447


namespace third_team_pies_l56_56401

theorem third_team_pies (total first_team second_team : ℕ) (h_total : total = 750) (h_first : first_team = 235) (h_second : second_team = 275) :
  total - (first_team + second_team) = 240 := by
  sorry

end third_team_pies_l56_56401


namespace tony_squat_weight_l56_56457

-- Definitions from conditions
def curl_weight := 90
def military_press_weight := 2 * curl_weight
def squat_weight := 5 * military_press_weight

-- Theorem statement
theorem tony_squat_weight : squat_weight = 900 := by
  sorry

end tony_squat_weight_l56_56457


namespace sum_n_div_n4_add_16_eq_9_div_320_l56_56032

theorem sum_n_div_n4_add_16_eq_9_div_320 :
  ∑' n : ℕ, n / (n^4 + 16) = 9 / 320 :=
sorry

end sum_n_div_n4_add_16_eq_9_div_320_l56_56032


namespace composite_numbers_quotient_l56_56022

theorem composite_numbers_quotient :
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) /
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39) =
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) / 
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39) :=
by sorry

end composite_numbers_quotient_l56_56022


namespace new_sales_volume_monthly_profit_maximize_profit_l56_56212

-- Define assumptions and variables
variables (x : ℝ) (p : ℝ) (v : ℝ) (profit : ℝ)

-- Part 1: New sales volume after price increase
theorem new_sales_volume (h : 0 < x ∧ x < 20) : v = 600 - 10 * x :=
sorry

-- Part 2: Price and quantity for a monthly profit of 10,000 yuan
theorem monthly_profit (h : profit = (40 + x - 30) * (600 - 10 * x)) (h2: profit = 10000) : p = 50 ∧ v = 500 :=
sorry

-- Part 3: Price for maximizing monthly sales profit
theorem maximize_profit (h : profit = (40 + x - 30) * (600 - 10 * x)) : (∃ x_max: ℝ, x_max < 20 ∧ ∀ x, x < 20 → profit ≤ -10 * (x - 25)^2 + 12250 ∧ p = 59 ∧ profit = 11890) :=
sorry

end new_sales_volume_monthly_profit_maximize_profit_l56_56212


namespace find_k_l56_56665

noncomputable def sequence_sum (n : ℕ) (k : ℝ) : ℝ :=
  3 * 2^n + k

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a n * a (n + 2) = (a (n + 1))^2

theorem find_k
  (a : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, a n = sequence_sum (n + 1) k - sequence_sum n k)
  (h2 : geometric_sequence a) :
  k = -3 :=
  by sorry

end find_k_l56_56665


namespace solve_system_eq_solve_system_ineq_l56_56885

-- For the system of equations:
theorem solve_system_eq (x y : ℝ) (h1 : x + 2 * y = 7) (h2 : 3 * x + y = 6) : x = 1 ∧ y = 3 :=
sorry

-- For the system of inequalities:
theorem solve_system_ineq (x : ℝ) (h1 : 2 * (x - 1) + 1 > -3) (h2 : x - 1 ≤ (1 + x) / 3) : -1 < x ∧ x ≤ 2 :=
sorry

end solve_system_eq_solve_system_ineq_l56_56885


namespace correct_bio_experiment_technique_l56_56493

-- Let's define our conditions as hypotheses.
def yeast_count_method := "sampling_inspection"
def small_animal_group_method := "sampler_sampling"
def mitosis_rinsing_purpose := "wash_away_dissociation_solution"
def fat_identification_solution := "alcohol"

-- The question translated into a statement is to show that the method for counting yeast is the sampling inspection method.
theorem correct_bio_experiment_technique :
  yeast_count_method = "sampling_inspection" ∧
  small_animal_group_method ≠ "mark-recapture" ∧
  mitosis_rinsing_purpose ≠ "wash_away_dye" ∧
  fat_identification_solution ≠ "50%_hydrochloric_acid" :=
sorry

end correct_bio_experiment_technique_l56_56493


namespace curve_intersection_one_point_l56_56004

theorem curve_intersection_one_point (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = a^2 ↔ y = x^2 + a) → (x, y) = (0, a)) ↔ (a ≥ -1/2) := 
sorry

end curve_intersection_one_point_l56_56004


namespace decagon_interior_angle_measure_l56_56601

-- Define the type for a regular polygon
structure RegularPolygon (n : Nat) :=
  (interior_angle_sum : Nat := (n - 2) * 180)
  (side_count : Nat := n)
  (regularity : Prop := True)  -- All angles are equal

-- Define the degree measure of an interior angle of a regular polygon
def interiorAngle (p : RegularPolygon 10) : Nat :=
  (p.interior_angle_sum) / p.side_count

-- The theorem to be proved
theorem decagon_interior_angle_measure : 
  ∀ (p : RegularPolygon 10), interiorAngle p = 144 := by
  -- The proof will be here, but for now, we use sorry
  sorry

end decagon_interior_angle_measure_l56_56601


namespace cookies_and_sugar_needed_l56_56698

-- Definitions derived from the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def initial_sugar : ℝ := 1.5
def flour_needed : ℕ := 5

-- The proof statement
theorem cookies_and_sugar_needed :
  (initial_cookies / initial_flour) * flour_needed = 40 ∧ (initial_sugar / initial_flour) * flour_needed = 2.5 :=
by
  sorry

end cookies_and_sugar_needed_l56_56698


namespace price_second_day_is_81_percent_l56_56354

-- Define the original price P (for the sake of clarity in the proof statement)
variable (P : ℝ)

-- Define the reductions
def first_reduction (P : ℝ) : ℝ := P - 0.1 * P
def second_reduction (P : ℝ) : ℝ := first_reduction P - 0.1 * first_reduction P

-- Question translated to Lean statement
theorem price_second_day_is_81_percent (P : ℝ) : 
  (second_reduction P / P) * 100 = 81 := by
  sorry

end price_second_day_is_81_percent_l56_56354


namespace biased_die_odd_sum_probability_l56_56465

noncomputable def biased_die_probability_even_odd (sum_is_odd: ℝ) : Prop :=
  ∃ (p : ℝ), (1 - 3*p = 0) ∧ (2*p = 1) ∧
  let p_odd := p in
  let p_even := 2 * p in
  (2 * p * p_odd) + (p_odd * p_even) = sum_is_odd

theorem biased_die_odd_sum_probability :
  ∃ (p : ℝ), biased_die_probability_even_odd (4 / 9) :=
by
  sorry

end biased_die_odd_sum_probability_l56_56465


namespace probability_of_PAIR_letters_in_PROBABILITY_l56_56228

theorem probability_of_PAIR_letters_in_PROBABILITY : 
  let total_letters := 11
  let favorable_letters := 4
  favorable_letters / total_letters = 4 / 11 :=
by
  let total_letters := 11
  let favorable_letters := 4
  show favorable_letters / total_letters = 4 / 11
  sorry

end probability_of_PAIR_letters_in_PROBABILITY_l56_56228


namespace scientific_notation_suzhou_blood_donors_l56_56097

theorem scientific_notation_suzhou_blood_donors : ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 124000 = a * 10^n ∧ a = 1.24 ∧ n = 5 :=
by
  sorry

end scientific_notation_suzhou_blood_donors_l56_56097


namespace induction_step_l56_56575

theorem induction_step 
  (k : ℕ) 
  (hk : ∃ m: ℕ, 5^k - 2^k = 3 * m) : 
  ∃ n: ℕ, 5^(k+1) - 2^(k+1) = 5 * (5^k - 2^k) + 3 * 2^k :=
by
  sorry

end induction_step_l56_56575


namespace pq_necessary_not_sufficient_l56_56624

theorem pq_necessary_not_sufficient (p q : Prop) : (p ∨ q) → (p ∧ q) ↔ false :=
by sorry

end pq_necessary_not_sufficient_l56_56624


namespace opposite_of_6_is_neg_6_l56_56315

theorem opposite_of_6_is_neg_6 : -6 = -6 := by
  sorry

end opposite_of_6_is_neg_6_l56_56315


namespace Matilda_fathers_chocolate_bars_l56_56711

theorem Matilda_fathers_chocolate_bars
  (total_chocolates : ℕ) (sisters : ℕ) (chocolates_each : ℕ) (given_to_father_each : ℕ) 
  (chocolates_given : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) : 
  total_chocolates = 20 → 
  sisters = 4 → 
  chocolates_each = total_chocolates / (sisters + 1) → 
  given_to_father_each = chocolates_each / 2 → 
  chocolates_given = (sisters + 1) * given_to_father_each → 
  given_to_mother = 3 → 
  eaten_by_father = 2 → 
  chocolates_given - given_to_mother - eaten_by_father = 5 :=
by
  intros h_total h_sisters h_chocolates_each h_given_to_father_each h_chocolates_given h_given_to_mother h_eaten_by_father
  sorry

end Matilda_fathers_chocolate_bars_l56_56711


namespace probability_of_at_least_one_accurate_forecast_l56_56718

theorem probability_of_at_least_one_accurate_forecast (PA PB : ℝ) (hA : PA = 0.8) (hB : PB = 0.75) :
  1 - ((1 - PA) * (1 - PB)) = 0.95 :=
by
  rw [hA, hB]
  sorry

end probability_of_at_least_one_accurate_forecast_l56_56718


namespace factorization_correct_l56_56924

theorem factorization_correct (a : ℝ) : a^2 - 2 * a - 15 = (a + 3) * (a - 5) := 
by 
  sorry

end factorization_correct_l56_56924


namespace max_value_f_l56_56387

theorem max_value_f (x y z : ℝ) (hxyz : x * y * z = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (1 - y * z + z) * (1 - x * z + x) * (1 - x * y + y) ≤ 1 :=
sorry

end max_value_f_l56_56387


namespace quiz_minimum_correct_l56_56069

theorem quiz_minimum_correct (x : ℕ) (hx : 7 * x + 14 ≥ 120) : x ≥ 16 := 
by sorry

end quiz_minimum_correct_l56_56069


namespace positive_diff_even_odd_sums_l56_56128

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l56_56128


namespace find_constants_l56_56960

theorem find_constants
  (k m n : ℝ)
  (h : -x^3 + (k + 7) * x^2 + m * x - 8 = -(x - 2) * (x - 4) * (x - n)) :
  k = 7 ∧ m = 2 ∧ n = 1 :=
sorry

end find_constants_l56_56960


namespace original_number_div_eq_l56_56266

theorem original_number_div_eq (h : 204 / 12.75 = 16) : 2.04 / 1.6 = 1.275 :=
by sorry

end original_number_div_eq_l56_56266


namespace tan_75_degrees_eq_l56_56944

noncomputable def tan_75_degrees : ℝ := Real.tan (75 * Real.pi / 180)

theorem tan_75_degrees_eq : tan_75_degrees = 2 + Real.sqrt 3 := by
  sorry

end tan_75_degrees_eq_l56_56944


namespace tetrahedron_non_coplanar_points_count_l56_56734

theorem tetrahedron_non_coplanar_points_count :
  let points := (vertices_of_tetrahedron ∪ midpoints_of_edges_tetrahedron : set ℝ),
      num_points := 10,
      num_selected := 4,
      num_coplanar_on_face := 4,
      num_coplanar_on_edges := 6,
      num_coplanar_parallelogram := 3,
      total_coplanar := num_coplanar_on_face + num_coplanar_on_edges + num_coplanar_parallelogram
  in
  points.card = num_points →
  combinatorics.choose num_points num_selected - total_coplanar = 197 :=
sorry

end tetrahedron_non_coplanar_points_count_l56_56734


namespace area_of_EFGH_l56_56966

-- Definitions based on given conditions
def shorter_side : ℝ := 4
def longer_side : ℝ := 8
def smaller_rectangle_area : ℝ := shorter_side * longer_side
def larger_rectangle_width : ℝ := longer_side
def larger_rectangle_height : ℝ := 2 * longer_side

-- Theorem stating the area of the larger rectangle
theorem area_of_EFGH : larger_rectangle_width * larger_rectangle_height = 128 := by
  -- Proof goes here
  sorry

end area_of_EFGH_l56_56966


namespace find_y_in_terms_of_abc_l56_56265

theorem find_y_in_terms_of_abc 
  (x y z a b c : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0)
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (h1 : xy / (x - y) = a)
  (h2 : xz / (x - z) = b)
  (h3 : yz / (y - z) = c) :
  y = bcx / ((b + c) * x - bc) := 
sorry

end find_y_in_terms_of_abc_l56_56265


namespace general_equation_l56_56385

theorem general_equation (n : ℤ) : 
    ∀ (a b : ℤ), 
    (a = 2 ∧ b = 6) ∨ (a = 5 ∧ b = 3) ∨ (a = 7 ∧ b = 1) ∨ (a = 10 ∧ b = -2) → 
    (a / (a - 4) + b / (b - 4) = 2) →
    (n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2) :=
by
  intros a b h_cond h_eq
  sorry

end general_equation_l56_56385


namespace smallest_n_for_Sn_gt_10_l56_56568

noncomputable def harmonicSeriesSum : ℕ → ℝ
| 0       => 0
| (n + 1) => harmonicSeriesSum n + 1 / (n + 1)

theorem smallest_n_for_Sn_gt_10 : ∃ n : ℕ, (harmonicSeriesSum n > 10) ∧ ∀ k < 12367, harmonicSeriesSum k ≤ 10 :=
by
  sorry

end smallest_n_for_Sn_gt_10_l56_56568


namespace total_distance_traveled_l56_56589

-- Define the parameters and conditions
def hoursPerDay : ℕ := 2
def daysPerWeek : ℕ := 5
def daysPeriod1 : ℕ := 3
def daysPeriod2 : ℕ := 2
def speedPeriod1 : ℕ := 12 -- speed in km/h from Monday to Wednesday
def speedPeriod2 : ℕ := 9 -- speed in km/h from Thursday to Friday

-- This is the theorem we want to prove
theorem total_distance_traveled : (daysPeriod1 * hoursPerDay * speedPeriod1) + (daysPeriod2 * hoursPerDay * speedPeriod2) = 108 :=
by
  sorry

end total_distance_traveled_l56_56589


namespace positive_difference_eq_250_l56_56117

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l56_56117


namespace triangle_inequality_check_triangle_sets_l56_56916

theorem triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem check_triangle_sets :
  ¬triangle_inequality 1 2 3 ∧
  triangle_inequality 2 2 2 ∧
  ¬triangle_inequality 2 2 4 ∧
  ¬triangle_inequality 1 3 5 :=
by
  sorry

end triangle_inequality_check_triangle_sets_l56_56916


namespace find_A_l56_56648

def heartsuit (A B : ℤ) : ℤ := 4 * A + A * B + 3 * B + 6

theorem find_A (A : ℤ) : heartsuit A 3 = 75 ↔ A = 60 / 7 := sorry

end find_A_l56_56648


namespace plum_cost_l56_56431

theorem plum_cost
  (total_fruits : ℕ)
  (total_cost : ℕ)
  (peach_cost : ℕ)
  (plums_bought : ℕ)
  (peaches_bought : ℕ)
  (P : ℕ) :
  total_fruits = 32 →
  total_cost = 52 →
  peach_cost = 1 →
  plums_bought = 20 →
  peaches_bought = total_fruits - plums_bought →
  total_cost = 20 * P + peaches_bought * peach_cost →
  P = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end plum_cost_l56_56431


namespace smallest_base_l56_56000

theorem smallest_base (b : ℕ) (h1 : b^2 ≤ 125) (h2 : 125 < b^3) : b = 6 := by
  sorry

end smallest_base_l56_56000


namespace find_C_line_MN_l56_56272

def point := (ℝ × ℝ)

-- Given points A and B
def A : point := (5, -2)
def B : point := (7, 3)

-- Conditions: M is the midpoint of AC and is on the y-axis
def M_on_y_axis (M : point) (A C : point) : Prop :=
  M.1 = 0 ∧ M.2 = (A.2 + C.2) / 2

-- Conditions: N is the midpoint of BC and is on the x-axis
def N_on_x_axis (N : point) (B C : point) : Prop :=
  N.1 = (B.1 + C.1) / 2 ∧ N.2 = 0

-- Coordinates of point C
theorem find_C (C : point)
  (M : point) (N : point)
  (hM : M_on_y_axis M A C)
  (hN : N_on_x_axis N B C) : C = (-5, -8) := sorry

-- Equation of line MN
theorem line_MN (M N : point)
  (MN_eq : M_on_y_axis M A (-5, -8) ∧ N_on_x_axis N B (-5, -8)) :
   ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ ((y = M.2) ∧ (x = M.1)) ∨ ((y = N.2) ∧ (x = N.1))) ∧ m = (3/2) ∧ b = 0 := sorry

end find_C_line_MN_l56_56272


namespace factor_polynomial_l56_56238

noncomputable def gcd_coeffs : ℕ := Nat.gcd 72 180

theorem factor_polynomial (x : ℝ) (GCD_72_180 : gcd_coeffs = 36)
    (GCD_x5_x9 : ∃ (y: ℝ), x^5 = y ∧ x^9 = y * x^4) :
    72 * x^5 - 180 * x^9 = -36 * x^5 * (5 * x^4 - 2) :=
by
  sorry

end factor_polynomial_l56_56238


namespace smallest_norm_of_v_l56_56294

variables (v : ℝ × ℝ)

def vector_condition (v : ℝ × ℝ) : Prop :=
  ‖(v.1 - 2, v.2 + 4)‖ = 10

theorem smallest_norm_of_v
  (hv : vector_condition v) :
  ‖v‖ ≥ 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_of_v_l56_56294


namespace total_fuel_needed_l56_56736

/-- Given that Car B can travel 30 miles per gallon and needs to cover a distance of 750 miles,
    and Car C has a fuel consumption rate of 20 miles per gallon and will travel 900 miles,
    prove that the total combined fuel required for Cars B and C is 70 gallons. -/
theorem total_fuel_needed (miles_per_gallon_B : ℕ) (miles_per_gallon_C : ℕ)
  (distance_B : ℕ) (distance_C : ℕ)
  (hB : miles_per_gallon_B = 30) (hC : miles_per_gallon_C = 20)
  (dB : distance_B = 750) (dC : distance_C = 900) :
  (distance_B / miles_per_gallon_B) + (distance_C / miles_per_gallon_C) = 70 := by {
    sorry 
}

end total_fuel_needed_l56_56736


namespace koala_fiber_absorption_l56_56410

theorem koala_fiber_absorption (x : ℝ) (h1 : 0 < x) (h2 : x * 0.30 = 15) : x = 50 :=
sorry

end koala_fiber_absorption_l56_56410


namespace number_of_pairs_divisible_by_5_l56_56660

theorem number_of_pairs_divisible_by_5 :
  let n := 1000
  let count := 200000
  (∀ x y : ℕ, 1 ≤ x ∧ x ≤ n ∧ 1 ≤ y ∧ y ≤ n → (x^2 + y^2) % 5 = 0) →
  (∃ count : ℕ, count == 200000) :=
begin
  sorry
end

end number_of_pairs_divisible_by_5_l56_56660


namespace company_production_n_l56_56046

theorem company_production_n (n : ℕ) (P : ℕ) 
  (h1 : P = n * 50) 
  (h2 : (P + 90) / (n + 1) = 58) : n = 4 := by 
  sorry

end company_production_n_l56_56046


namespace opposite_of_6_is_neg_6_l56_56314

theorem opposite_of_6_is_neg_6 : -6 = -6 := by
  sorry

end opposite_of_6_is_neg_6_l56_56314


namespace largest_even_not_sum_of_two_odd_composites_l56_56028

def is_odd_composite (n : ℕ) : Prop :=
  n % 2 = 1 ∧ ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

theorem largest_even_not_sum_of_two_odd_composites : ∀ n : ℕ, 38 < n → 
  ∃ a b : ℕ, is_odd_composite a ∧ is_odd_composite b ∧ n = a + b :=
begin
  sorry
end

end largest_even_not_sum_of_two_odd_composites_l56_56028


namespace four_person_apartments_l56_56937

theorem four_person_apartments : 
  ∃ x : ℕ, 
    (4 * (10 + 20 * 2 + 4 * x)) * 3 / 4 = 210 → x = 5 :=
by
  sorry

end four_person_apartments_l56_56937


namespace unique_shirt_and_tie_outfits_l56_56303

theorem unique_shirt_and_tie_outfits :
  let shirts := 10
  let ties := 8
  let choose n k := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose shirts 5 * choose ties 4 = 17640 :=
by
  sorry

end unique_shirt_and_tie_outfits_l56_56303


namespace largest_crowd_size_l56_56195

theorem largest_crowd_size :
  ∃ (n : ℕ), 
    (⌊n / 2⌋ + ⌊n / 3⌋ + ⌊n / 5⌋ = n) ∧
    ∀ m : ℕ, (⌊m / 2⌋ + ⌊m / 3⌋ + ⌊m / 5⌋ = m) → m ≤ 37 :=
sorry

end largest_crowd_size_l56_56195


namespace subproblem1_l56_56793

theorem subproblem1 (a : ℝ) : a^3 * a + (2 * a^2)^2 = 5 * a^4 := 
by sorry

end subproblem1_l56_56793


namespace real_roots_exist_for_nonzero_K_l56_56666

theorem real_roots_exist_for_nonzero_K (K : ℝ) (hK : K ≠ 0) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by
  sorry

end real_roots_exist_for_nonzero_K_l56_56666


namespace eq_of_div_eq_div_l56_56338

theorem eq_of_div_eq_div {a b c : ℝ} (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
by
  sorry

end eq_of_div_eq_div_l56_56338


namespace nth_equation_l56_56429

theorem nth_equation (n : ℕ) (h : n > 0) : (1 / n) * ((n^2 + 2 * n) / (n + 1)) - (1 / (n + 1)) = 1 :=
by
  sorry

end nth_equation_l56_56429


namespace sector_area_is_8pi_over_3_l56_56724

noncomputable def sector_area {r θ1 θ2 : ℝ} 
  (hθ1 : θ1 = π / 3)
  (hθ2 : θ2 = 2 * π / 3)
  (hr : r = 4) : ℝ := 
    1 / 2 * (θ2 - θ1) * r ^ 2

theorem sector_area_is_8pi_over_3 (θ1 θ2 : ℝ) 
  (hθ1 : θ1 = π / 3)
  (hθ2 : θ2 = 2 * π / 3)
  (r : ℝ) (hr : r = 4) : 
  sector_area hθ1 hθ2 hr = 8 * π / 3 :=
by
  sorry

end sector_area_is_8pi_over_3_l56_56724


namespace estimate_value_l56_56235

theorem estimate_value : 1 < (3 - Real.sqrt 3) ∧ (3 - Real.sqrt 3) < 2 :=
by
  have h₁ : Real.sqrt 18 = 3 * Real.sqrt 2 :=
    by sorry
  have h₂ : Real.sqrt 6 = Real.sqrt 3 * Real.sqrt 2 :=
    by sorry
  have h₃ : (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 = (3 * Real.sqrt 2 - Real.sqrt 3 * Real.sqrt 2) / Real.sqrt 2 :=
    by sorry
  have h₄ : (3 * Real.sqrt 2 - Real.sqrt 3 * Real.sqrt 2) / Real.sqrt 2 = 3 - Real.sqrt 3 :=
    by sorry
  have h₅ : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 :=
    by sorry
  sorry

end estimate_value_l56_56235


namespace measured_percentage_weight_loss_l56_56194

variable (W : ℝ) -- W is the starting weight.
variable (weight_loss_percent : ℝ := 0.12) -- 12% weight loss.
variable (clothes_weight_percent : ℝ := 0.03) -- 3% clothes weight addition.
variable (beverage_weight_percent : ℝ := 0.005) -- 0.5% beverage weight addition.

theorem measured_percentage_weight_loss : 
  (W - ((0.88 * W) + (clothes_weight_percent * W) + (beverage_weight_percent * W))) / W * 100 = 8.5 :=
by
  sorry

end measured_percentage_weight_loss_l56_56194


namespace largest_sphere_radius_l56_56489

noncomputable def torus_inner_radius := 3
noncomputable def torus_outer_radius := 5
noncomputable def torus_center_circle := (4, 0, 1)
noncomputable def torus_radius := 1
noncomputable def torus_table_plane := 0

theorem largest_sphere_radius :
  ∀ (r : ℝ), 
  ∀ (O P : ℝ × ℝ × ℝ), 
  (P = (4, 0, 1)) → 
  (O = (0, 0, r)) → 
  4^2 + (r - 1)^2 = (r + 1)^2 → 
  r = 4 := 
by
  intros
  sorry

end largest_sphere_radius_l56_56489


namespace fraction_of_income_from_tips_l56_56637

variable (S T I : ℝ)
variable (h : T = (5 / 4) * S)

theorem fraction_of_income_from_tips (h : T = (5 / 4) * S) (I : ℝ) (w : I = S + T) : (T / I) = 5 / 9 :=
by
  -- The proof goes here
  sorry

end fraction_of_income_from_tips_l56_56637


namespace exchange_ways_count_l56_56597

theorem exchange_ways_count : ∃ n : ℕ, n = 46 ∧ ∀ x y z : ℕ, x + 2 * y + 5 * z = 20 → n = 46 :=
by
  sorry

end exchange_ways_count_l56_56597


namespace team_A_wins_at_least_5_matches_l56_56995

open Classical

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))

theorem team_A_wins_at_least_5_matches :
  let p := 1 / 2 in
  let n := 7 in
  (binomial_probability n 5 p + binomial_probability n 6 p + binomial_probability n 7 p) = 29 / 128 :=
by
  sorry

end team_A_wins_at_least_5_matches_l56_56995


namespace positive_difference_of_sums_l56_56126

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l56_56126


namespace find_normal_price_l56_56331

open Real

theorem find_normal_price (P : ℝ) (h1 : 0.612 * P = 108) : P = 176.47 := by
  sorry

end find_normal_price_l56_56331


namespace rationalize_denominator_l56_56434

theorem rationalize_denominator :
  (Real.sqrt (5 / 12)) = ((Real.sqrt 15) / 6) :=
sorry

end rationalize_denominator_l56_56434


namespace derivative_at_pi_div_2_l56_56837

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x

theorem derivative_at_pi_div_2 : (deriv f (Real.pi / 2)) = 4 := 
by
  sorry

end derivative_at_pi_div_2_l56_56837


namespace probability_neither_A_nor_B_l56_56341

noncomputable def pA : ℝ := 0.25
noncomputable def pB : ℝ := 0.35
noncomputable def pA_and_B : ℝ := 0.15

theorem probability_neither_A_nor_B :
  1 - (pA + pB - pA_and_B) = 0.55 :=
by
  simp [pA, pB, pA_and_B]
  norm_num
  sorry

end probability_neither_A_nor_B_l56_56341


namespace area_of_triangle_PQR_l56_56738

structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := -4, y := 2 }
def Q : Point := { x := 8, y := 2 }
def R : Point := { x := 6, y := -4 }

noncomputable def triangle_area (A B C : Point) : ℝ :=
  (1 / 2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem area_of_triangle_PQR : triangle_area P Q R = 36 := by
  sorry

end area_of_triangle_PQR_l56_56738


namespace increasing_function_a_l56_56389

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≥ 0 then
    x^2
  else
    x^3 - (a-1)*x + a^2 - 3*a - 4

theorem increasing_function_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f x a ≤ f y a) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end increasing_function_a_l56_56389


namespace sams_trip_length_l56_56579

theorem sams_trip_length (total_trip : ℚ) 
  (h1 : total_trip / 4 + 24 + total_trip / 6 = total_trip) : 
  total_trip = 288 / 7 :=
by
  -- proof placeholder
  sorry

end sams_trip_length_l56_56579


namespace smallest_ducks_l56_56427

theorem smallest_ducks :
  ∃ D : ℕ, 
  ∃ C : ℕ, 
  ∃ H : ℕ, 
  (13 * D = 17 * C) ∧
  (11 * H = (6 / 5) * 13 * D) ∧
  (17 * C = (3 / 8) * 11 * H) ∧ 
  (13 * D = 520) :=
by 
  sorry

end smallest_ducks_l56_56427


namespace cube_edge_adjacency_l56_56640

def is_beautiful (f: Finset ℕ) := 
  ∃ a b c d, f = {a, b, c, d} ∧ a = b + c + d

def cube_is_beautiful (faces: Finset (Finset ℕ)) :=
  ∃ t1 t2 t3, t1 ∈ faces ∧ t2 ∈ faces ∧ t3 ∈ faces ∧
  is_beautiful t1 ∧ is_beautiful t2 ∧ is_beautiful t3

def valid_adjacency (v: ℕ) (n1 n2 n3: ℕ) := 
  v = 6 ∧ ((n1 = 2 ∧ n2 = 3 ∧ n3 = 5) ∨
           (n1 = 2 ∧ n2 = 3 ∧ n3 = 7) ∨
           (n1 = 3 ∧ n2 = 5 ∧ n3 = 7))

theorem cube_edge_adjacency : 
  ∀ faces: Finset (Finset ℕ), 
  ∃ v n1 n2 n3, 
  (v = 6 ∧ (valid_adjacency v n1 n2 n3)) ∧
  cube_is_beautiful faces := 
by
  -- Entails the proof, which is not required here
  sorry

end cube_edge_adjacency_l56_56640


namespace angle_2016_216_in_same_quadrant_l56_56936

noncomputable def angle_in_same_quadrant (a b : ℝ) : Prop :=
  let normalized (x : ℝ) := x % 360
  normalized a = normalized b

theorem angle_2016_216_in_same_quadrant : angle_in_same_quadrant 2016 216 := by
  sorry

end angle_2016_216_in_same_quadrant_l56_56936


namespace positive_difference_even_odd_sums_l56_56162

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l56_56162


namespace percentage_seats_not_taken_l56_56938

theorem percentage_seats_not_taken
  (rows : ℕ) (seats_per_row : ℕ) 
  (ticket_price : ℕ)
  (earnings : ℕ)
  (H_rows : rows = 150)
  (H_seats_per_row : seats_per_row = 10) 
  (H_ticket_price : ticket_price = 10)
  (H_earnings : earnings = 12000) :
  (1500 - (12000 / 10)) / 1500 * 100 = 20 := 
by
  sorry

end percentage_seats_not_taken_l56_56938


namespace eeshas_usual_time_l56_56573

/-- Eesha's usual time to reach her office from home is 60 minutes,
given that she started 30 minutes late and reached her office
50 minutes late while driving 25% slower than her usual speed. -/
theorem eeshas_usual_time (T T' : ℝ) (h1 : T' = T / 0.75) (h2 : T' = T + 20) : T = 60 := by
  sorry

end eeshas_usual_time_l56_56573


namespace option_c_correct_l56_56335

-- Statement of the problem: Prove that (x-3)^2 = x^2 - 6x + 9

theorem option_c_correct (x : ℝ) : (x - 3) ^ 2 = x ^ 2 - 6 * x + 9 :=
by
  sorry

end option_c_correct_l56_56335


namespace sum_of_coefficients_l56_56308

theorem sum_of_coefficients (A B C : ℤ) 
  (h_factorization : ∀ x, x^3 + A * x^2 + B * x + C = (x + 2) * (x - 2) * (x - 1)) :
  A + B + C = -1 :=
by sorry

end sum_of_coefficients_l56_56308


namespace opposite_of_six_is_negative_six_l56_56312

theorem opposite_of_six_is_negative_six : -6 = -6 :=
by
  sorry

end opposite_of_six_is_negative_six_l56_56312


namespace impossibility_of_arrangement_l56_56766

-- Definitions based on identified conditions in the problem
def isValidArrangement (arr : List ℕ) : Prop :=
  arr.length = 300 ∧
  (∀ i, i < 300 - 1 → arr.get i = |arr.get (i - 1) - arr.get (i + 1)|) ∧
  (arr.all (λ x => x > 0))

theorem impossibility_of_arrangement :
  ¬ (∃ arr : List ℕ, isValidArrangement arr) :=
sorry

end impossibility_of_arrangement_l56_56766


namespace matrix_determinant_l56_56552

variable {a b c d : ℝ}
variable (h : a * d - b * c = 4)

theorem matrix_determinant :
  (a * (7 * c + 3 * d) - c * (7 * a + 3 * b)) = 12 := by
  sorry

end matrix_determinant_l56_56552


namespace problem_l56_56984

theorem problem (A : ℕ) (B : ℕ) (hA : A = 2011 ^ 2011) (hB : B = (nat.factorial 2011) ^ 2) : A < B :=
by
  rw [hA, hB]
  sorry

end problem_l56_56984


namespace oakwood_team_count_l56_56593

theorem oakwood_team_count :
  let girls := 5
  let boys := 7
  let choose_3_girls := Nat.choose girls 3
  let choose_2_boys := Nat.choose boys 2
  choose_3_girls * choose_2_boys = 210 := by
sorry

end oakwood_team_count_l56_56593


namespace parking_spots_full_iff_num_sequences_l56_56358

noncomputable def num_parking_sequences (n : ℕ) : ℕ :=
  (n + 1) ^ (n - 1)

-- Statement of the theorem
theorem parking_spots_full_iff_num_sequences (n : ℕ) :
  ∀ (a : ℕ → ℕ), (∀ (i : ℕ), i < n → a i ≤ n) → 
  (∀ (j : ℕ), j ≤ n → (∃ i, i < n ∧ a i = j)) ↔ 
  num_parking_sequences n = (n + 1) ^ (n - 1) :=
sorry

end parking_spots_full_iff_num_sequences_l56_56358


namespace inequality_holds_l56_56529

noncomputable def f (x : ℝ) := x^2 + 2 * Real.cos x

theorem inequality_holds (x1 x2 : ℝ) : 
  f x1 > f x2 → x1 > |x2| := 
sorry

end inequality_holds_l56_56529


namespace polygon_area_of_plane_intersection_l56_56014

noncomputable def P : EuclideanGeometry.Point ℝ 3 := ⟨6, 0, 0⟩
noncomputable def Q : EuclideanGeometry.Point ℝ 3 := ⟨30, 0, 17⟩
noncomputable def R : EuclideanGeometry.Point ℝ 3 := ⟨30, 3, 30⟩

-- Defining the cube vertices
noncomputable def A : EuclideanGeometry.Point ℝ 3 := ⟨0, 0, 0⟩
noncomputable def B : EuclideanGeometry.Point ℝ 3 := ⟨30, 0, 0⟩
noncomputable def C : EuclideanGeometry.Point ℝ 3 := ⟨30, 0, 30⟩
noncomputable def D : EuclideanGeometry.Point ℝ 3 := ⟨30, 30, 30⟩

theorem polygon_area_of_plane_intersection :
  let plane := EuclideanGeometry.Plane.mk_through_points P Q R in
  EuclideanGeometry.area_of_polygon_formed_by_plane_cube_intersection plane A B C D = 42 := -- Assume 42 is the placeholder area based on real calculations
sorry

end polygon_area_of_plane_intersection_l56_56014


namespace percentage_failed_in_english_l56_56560

theorem percentage_failed_in_english
  (H_perc : ℝ) (B_perc : ℝ) (Passed_in_English_alone : ℝ) (Total_candidates : ℝ)
  (H_perc_eq : H_perc = 36)
  (B_perc_eq : B_perc = 15)
  (Passed_in_English_alone_eq : Passed_in_English_alone = 630)
  (Total_candidates_eq : Total_candidates = 3000) :
  ∃ E_perc : ℝ, E_perc = 85 := by
  sorry

end percentage_failed_in_english_l56_56560


namespace jason_two_weeks_eggs_l56_56801

-- Definitions of given conditions
def eggs_per_omelet := 3
def days_per_week := 7
def weeks := 2

-- Statement to prove
theorem jason_two_weeks_eggs : (eggs_per_omelet * (days_per_week * weeks)) = 42 := by
  sorry

end jason_two_weeks_eggs_l56_56801


namespace smallest_C_l56_56967

-- Defining the problem and the conditions
theorem smallest_C (k : ℕ) (C : ℕ) :
  (∀ n : ℕ, n ≥ k → (C * Nat.choose (2 * n) (n + k)) % (n + k + 1) = 0) ↔
  C = 2 * k + 1 :=
by sorry

end smallest_C_l56_56967


namespace find_alpha_l56_56832

-- Given conditions
variables (α β : ℝ)
axiom h1 : α + β = 11
axiom h2 : α * β = 24
axiom h3 : α > β

-- Theorems to prove
theorem find_alpha : α = 8 :=
  sorry

end find_alpha_l56_56832


namespace prove_tan_2x_prove_sin_x_plus_pi_over_4_l56_56675

open Real

noncomputable def tan_2x (x : ℝ) (hx : x ∈ set.Ioo (π / 2) π) (h_tan : abs (tan x) = 2) : Prop :=
  tan (2 * x) = 4 / 3

noncomputable def sin_x_plus_pi_over_4 (x : ℝ) (hx : x ∈ set.Ioo (π / 2) π) (h_tan : abs (tan x) = 2) : Prop :=
  sin (x + π/4) = sqrt 10 / 10

theorem prove_tan_2x (x : ℝ) (hx : x ∈ set.Ioo (π / 2) π) (h_tan : abs (tan x) = 2) :
  tan_2x x hx h_tan :=
sorry

theorem prove_sin_x_plus_pi_over_4 (x : ℝ) (hx : x ∈ set.Ioo (π / 2) π) (h_tan : abs (tan x) = 2) :
  sin_x_plus_pi_over_4 x hx h_tan :=
sorry

end prove_tan_2x_prove_sin_x_plus_pi_over_4_l56_56675


namespace positive_difference_even_odd_sums_l56_56164

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l56_56164


namespace total_simple_interest_is_correct_l56_56633

noncomputable def principal : ℝ := 15041.875
noncomputable def rate : ℝ := 8
noncomputable def time : ℝ := 5
noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_simple_interest_is_correct :
  simple_interest principal rate time = 6016.75 := 
sorry

end total_simple_interest_is_correct_l56_56633


namespace at_least_one_truth_not_knight_l56_56904

-- Define roles
inductive Role
| knight
| liar
| normal

open Role

-- Statements
def A_statement (B_role : Role) : Bool :=
  B_role = knight

def B_statement (A_role : Role) : Bool :=
  A_role ≠ knight

-- The main theorem
theorem at_least_one_truth_not_knight (A B : Role) :
  (A ∈ [liar, normal] ∧ A_statement B = false ∧ (B = normal ∨ B = liar ∧ B_statement A = true)) ∨
  (B ∈ [normal] ∧ B_statement A = true) :=
by
  sorry

end at_least_one_truth_not_knight_l56_56904


namespace keiko_jogging_speed_l56_56407

variable (s : ℝ) -- Keiko's jogging speed
variable (b : ℝ) -- radius of the inner semicircle
variable (L_inner : ℝ := 200 + 2 * Real.pi * b) -- total length of the inner track
variable (L_outer : ℝ := 200 + 2 * Real.pi * (b + 8)) -- total length of the outer track
variable (t_inner : ℝ := L_inner / s) -- time to jog the inside edge
variable (t_outer : ℝ := L_outer / s) -- time to jog the outside edge
variable (time_difference : ℝ := 48) -- time difference between jogging inside and outside edges

theorem keiko_jogging_speed : L_inner = 200 + 2 * Real.pi * b →
                           L_outer = 200 + 2 * Real.pi * (b + 8) →
                           t_outer = t_inner + 48 →
                           s = Real.pi / 3 :=
by
  intro h1 h2 h3
  sorry

end keiko_jogging_speed_l56_56407


namespace original_wire_length_l56_56209

theorem original_wire_length 
(L : ℝ) 
(h1 : L / 2 - 3 / 2 > 0) 
(h2 : L / 2 - 3 > 0) 
(h3 : L / 4 - 11.5 > 0)
(h4 : L / 4 - 6.5 = 7) : 
L = 54 := 
sorry

end original_wire_length_l56_56209


namespace min_value_of_z_l56_56739

theorem min_value_of_z : ∀ (x : ℝ), ∃ z : ℝ, z = 5 * x^2 - 20 * x + 45 ∧ z ≥ 25 :=
by sorry

end min_value_of_z_l56_56739


namespace passengers_on_ship_l56_56762

theorem passengers_on_ship (P : ℕ)
  (h1 : P / 12 + P / 4 + P / 9 + P / 6 + 42 = P) :
  P = 108 := 
by sorry

end passengers_on_ship_l56_56762


namespace team_a_wins_at_least_2_l56_56561

def team_a_wins_at_least (total_games lost_games : ℕ) (points : ℕ) (won_points draw_points lost_points : ℕ) : Prop :=
  ∃ (won_games : ℕ), 
    total_games = won_games + (total_games - lost_games - won_games) + lost_games ∧
    won_games * won_points + (total_games - lost_games - won_games) * draw_points > points ∧
    won_games ≥ 2

theorem team_a_wins_at_least_2 :
  team_a_wins_at_least 5 1 7 3 1 0 :=
by
  -- Proof goes here
  sorry

end team_a_wins_at_least_2_l56_56561


namespace positive_difference_eq_250_l56_56120

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l56_56120


namespace find_y_for_line_slope_45_degrees_l56_56100

theorem find_y_for_line_slope_45_degrees :
  ∃ y, (∃ x₁ y₁ x₂ y₂, x₁ = 4 ∧ y₁ = y ∧ x₂ = 2 ∧ y₂ = -3 ∧ (y₂ - y₁) / (x₂ - x₁) = 1) → y = -1 :=
by
  sorry

end find_y_for_line_slope_45_degrees_l56_56100


namespace Matilda_fathers_chocolate_bars_l56_56710

theorem Matilda_fathers_chocolate_bars
  (total_chocolates : ℕ) (sisters : ℕ) (chocolates_each : ℕ) (given_to_father_each : ℕ) 
  (chocolates_given : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) : 
  total_chocolates = 20 → 
  sisters = 4 → 
  chocolates_each = total_chocolates / (sisters + 1) → 
  given_to_father_each = chocolates_each / 2 → 
  chocolates_given = (sisters + 1) * given_to_father_each → 
  given_to_mother = 3 → 
  eaten_by_father = 2 → 
  chocolates_given - given_to_mother - eaten_by_father = 5 :=
by
  intros h_total h_sisters h_chocolates_each h_given_to_father_each h_chocolates_given h_given_to_mother h_eaten_by_father
  sorry

end Matilda_fathers_chocolate_bars_l56_56710


namespace proof_problem_l56_56060

variable (a b c x y z : ℝ)

theorem proof_problem
  (h1 : x + y - z = a - b)
  (h2 : x - y + z = b - c)
  (h3 : - x + y + z = c - a) : 
  x + y + z = 0 := by
  sorry

end proof_problem_l56_56060


namespace sum_of_reciprocal_of_roots_l56_56252

theorem sum_of_reciprocal_of_roots :
  ∀ x1 x2 : ℝ, (x1 * x2 = 2) → (x1 + x2 = 3) → (1 / x1 + 1 / x2 = 3 / 2) :=
by
  intros x1 x2 h_prod h_sum
  sorry

end sum_of_reciprocal_of_roots_l56_56252


namespace negation_of_existence_l56_56059

theorem negation_of_existence (p : Prop) : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 ≤ x + 2)) ↔ (∀ x : ℝ, x > 0 → x^2 > x + 2) :=
by
  sorry

end negation_of_existence_l56_56059


namespace expression_value_at_neg3_l56_56002

theorem expression_value_at_neg3 (p q : ℤ) (h : 27 * p + 3 * q = 14) :
  (p * (-3)^3 + q * (-3) - 1) = -15 :=
sorry

end expression_value_at_neg3_l56_56002


namespace girls_more_than_boys_l56_56319

theorem girls_more_than_boys (total_students boys : ℕ) (h1 : total_students = 650) (h2 : boys = 272) :
  (total_students - boys) - boys = 106 :=
by
  sorry

end girls_more_than_boys_l56_56319


namespace pencil_price_units_l56_56690

def pencil_price_in_units (pencil_price : ℕ) : ℚ := pencil_price / 10000

theorem pencil_price_units 
  (price_of_pencil : ℕ) 
  (h1 : price_of_pencil = 5000 - 20) : 
  pencil_price_in_units price_of_pencil = 0.5 := 
by
  sorry

end pencil_price_units_l56_56690


namespace max_small_packages_l56_56893

theorem max_small_packages (L S : ℝ) (W : ℝ) (h1 : W = 12 * L) (h2 : W = 20 * S) :
  (∃ n_smalls, n_smalls = 5 ∧ W - 9 * L = n_smalls * S) :=
by
  sorry

end max_small_packages_l56_56893


namespace minimize_fuel_consumption_l56_56902

-- Define conditions as constants
def cargo_total : ℕ := 157
def cap_large : ℕ := 5
def cap_small : ℕ := 2
def fuel_large : ℕ := 20
def fuel_small : ℕ := 10

-- Define truck counts
def n_large : ℕ := 31
def n_small : ℕ := 1

-- Theorem: the number of large and small trucks that minimize fuel consumption
theorem minimize_fuel_consumption : 
  n_large * cap_large + n_small * cap_small = cargo_total ∧
  (∀ m_large m_small, m_large * cap_large + m_small * cap_small = cargo_total → 
    m_large * fuel_large + m_small * fuel_small ≥ n_large * fuel_large + n_small * fuel_small) :=
by
  -- Statement to be proven
  sorry

end minimize_fuel_consumption_l56_56902


namespace servant_position_for_28_purses_servant_position_for_27_purses_l56_56578

-- Definitions based on problem conditions
def total_wealthy_men: ℕ := 7

def valid_purse_placement (n: ℕ): Prop := 
  (n ≤ total_wealthy_men * (total_wealthy_men + 1) / 2)

def get_servant_position (n: ℕ): ℕ := 
  if n = 28 then total_wealthy_men else if n = 27 then 6 else 0

-- Proof statements to equate conditions with the answers
theorem servant_position_for_28_purses : 
  get_servant_position 28 = 7 :=
sorry

theorem servant_position_for_27_purses : 
  get_servant_position 27 = 6 ∨ get_servant_position 27 = 7 :=
sorry

end servant_position_for_28_purses_servant_position_for_27_purses_l56_56578


namespace M_sufficient_not_necessary_for_N_l56_56858

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 2}

theorem M_sufficient_not_necessary_for_N (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → ¬ (a ∈ M)) :=
sorry

end M_sufficient_not_necessary_for_N_l56_56858


namespace stacy_days_to_complete_paper_l56_56301

def total_pages : ℕ := 66
def pages_per_day : ℕ := 11

theorem stacy_days_to_complete_paper :
  total_pages / pages_per_day = 6 := by
  sorry

end stacy_days_to_complete_paper_l56_56301


namespace feed_mixture_hay_calculation_l56_56722

theorem feed_mixture_hay_calculation
  (hay_Stepan_percent oats_Pavel_percent corn_mixture_percent : ℝ)
  (hay_Stepan_mass_Stepan hay_Pavel_mass_Pavel total_mixture_mass : ℝ):
  hay_Stepan_percent = 0.4 ∧
  oats_Pavel_percent = 0.26 ∧
  (∃ (x : ℝ), 
  x > 0 ∧ 
  hay_Pavel_percent =  0.74 - x ∧ 
  0.15 * x + 0.25 * x = 0.3 * total_mixture_mass ∧
  hay_Stepan_mass_Stepan = 0.40 * 150 ∧
  hay_Pavel_mass_Pavel = (0.74 - x) * 250 ∧ 
  total_mixture_mass = 150 + 250) → 
  hay_Stepan_mass_Stepan + hay_Pavel_mass_Pavel = 170 := 
by
  intro h
  obtain ⟨h1, h2, ⟨x, hx1, hx2, hx3, hx4, hx5, hx6⟩⟩ := h
  /- proof -/
  sorry

end feed_mixture_hay_calculation_l56_56722


namespace positive_difference_probability_l56_56745

theorem positive_difference_probability (fair_coin : Prop) (four_flips : ℕ) (fair : fair_coin) (flips : four_flips = 4) :
  let p1 := (Nat.choose 4 3)*((1/2)^3)*((1/2)^1) in
  let p2 := (1/2)^4 in
  abs (p1 - p2) = 7/16 :=
by
  -- Definitions
  let p1 := (Nat.choose 4 3) * ((1/2)^3) * ((1/2)^1)
  let p2 := (1/2)^4
  have h : abs (p1 - p2) = 7 / 16
  sorry

end positive_difference_probability_l56_56745


namespace jennie_speed_difference_l56_56563

theorem jennie_speed_difference :
  (∀ (d t1 t2 : ℝ), (d = 200) → (t1 = 5) → (t2 = 4) → (40 = d / t1) → (50 = d / t2) → (50 - 40 = 10)) :=
by
  intros d t1 t2 h_d h_t1 h_t2 h_speed_heavy h_speed_no_traffic
  sorry

end jennie_speed_difference_l56_56563


namespace complex_product_l56_56225

theorem complex_product : (3 + 4 * I) * (-2 - 3 * I) = -18 - 17 * I :=
by
  sorry

end complex_product_l56_56225


namespace intersection_M_N_l56_56085

noncomputable def M : Set ℝ := { x | x^2 = x }
noncomputable def N : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_M_N :
  M ∩ N = {1} := by
  sorry

end intersection_M_N_l56_56085


namespace exists_equilateral_triangle_l56_56500

variables {d1 d2 d3 : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))}

theorem exists_equilateral_triangle (hne1 : d1 ≠ d2) (hne2 : d2 ≠ d3) (hne3 : d1 ≠ d3) : 
  ∃ (A1 A2 A3 : EuclideanSpace ℝ (Fin 2)), 
  (A1 ∈ d1 ∧ A2 ∈ d2 ∧ A3 ∈ d3) ∧ 
  dist A1 A2 = dist A2 A3 ∧ dist A2 A3 = dist A3 A1 := 
sorry

end exists_equilateral_triangle_l56_56500


namespace volume_ratio_of_cubes_l56_56332

def cube_volume (a : ℝ) : ℝ := a ^ 3

theorem volume_ratio_of_cubes :
  cube_volume 3 / cube_volume 18 = 1 / 216 :=
by
  sorry

end volume_ratio_of_cubes_l56_56332


namespace pyramid_volume_l56_56656

theorem pyramid_volume (a : ℝ) (h : a = 2)
  (b : ℝ) (hb : b = 18) :
  ∃ V, V = 2 * Real.sqrt 2 :=
by
  sorry

end pyramid_volume_l56_56656


namespace winning_ticket_probability_l56_56068

open BigOperators

-- Calculate n choose k
def choose (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Given conditions
def probability_PowerBall := (1 : ℚ) / 30
def probability_LuckyBalls := (1 : ℚ) / choose 49 6

-- Theorem to prove the result
theorem winning_ticket_probability :
  probability_PowerBall * probability_LuckyBalls = (1 : ℚ) / 419514480 := by
  sorry

end winning_ticket_probability_l56_56068


namespace part1_part2_part3_l56_56567

def set_A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def set_B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def set_C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

theorem part1 (a : ℝ) : (set_A a ∩ set_B) = (set_A a ∪ set_B) → a = 5 :=
by
  sorry

theorem part2 (a : ℝ) : (∅ ⊂ (set_A a ∩ set_B)) ∧ (set_A a ∩ set_C = ∅) → a = -2 :=
by
  sorry

theorem part3 (a : ℝ) : (set_A a ∩ set_B) = (set_A a ∩ set_C) ∧ (set_A a ∩ set_B ≠ ∅) → a = -3 :=
by
  sorry

end part1_part2_part3_l56_56567


namespace fibonacci_eighth_term_l56_56733

theorem fibonacci_eighth_term
  (F : ℕ → ℕ)
  (h1 : F 9 = 34)
  (h2 : F 10 = 55)
  (fib : ∀ n, F (n + 2) = F (n + 1) + F n) :
  F 8 = 21 :=
by
  sorry

end fibonacci_eighth_term_l56_56733


namespace sum_of_roots_eq_neg3_l56_56989

theorem sum_of_roots_eq_neg3
  (a b c : ℝ)
  (h_eq : 2 * x^2 + 6 * x - 1 = 0)
  (h_a : a = 2)
  (h_b : b = 6) :
  (x1 x2 : ℝ) → x1 + x2 = -b / a :=
by
  sorry

end sum_of_roots_eq_neg3_l56_56989


namespace remainder_of_prime_when_divided_by_240_l56_56096

theorem remainder_of_prime_when_divided_by_240 (n : ℕ) (hn : n > 0) (hp : Nat.Prime (2^n + 1)) : (2^n + 1) % 240 = 17 := 
sorry

end remainder_of_prime_when_divided_by_240_l56_56096


namespace pqr_value_l56_56887

theorem pqr_value (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h1 : p + q + r = 30) 
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + (420 : ℚ) / (p * q * r) = 1) : 
  p * q * r = 1800 := 
sorry

end pqr_value_l56_56887


namespace expansion_identity_l56_56359

theorem expansion_identity : 121 + 2 * 11 * 9 + 81 = 400 := by
  sorry

end expansion_identity_l56_56359


namespace factor_x12_minus_729_l56_56645

theorem factor_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^6 + 27) * (x^3 + 3 * Real.sqrt 3) * (x^3 - 3 * Real.sqrt 3) := 
by
  sorry

end factor_x12_minus_729_l56_56645


namespace vector_solution_l56_56976

theorem vector_solution
  (x y : ℝ)
  (h1 : (2*x - y = 0))
  (h2 : (x^2 + y^2 = 20)) :
  (x = 2 ∧ y = 4) ∨ (x = -2 ∧ y = -4) := 
by
  sorry

end vector_solution_l56_56976


namespace min_a_for_50_pow_2023_div_17_l56_56553

theorem min_a_for_50_pow_2023_div_17 (a : ℕ) (h : 17 ∣ (50 ^ 2023 + a)) : a = 18 :=
sorry

end min_a_for_50_pow_2023_div_17_l56_56553


namespace simplify_fraction_90_150_l56_56865

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end simplify_fraction_90_150_l56_56865


namespace miguel_paint_area_l56_56087

def wall_height := 10
def wall_length := 15
def window_side := 3

theorem miguel_paint_area :
  (wall_height * wall_length) - (window_side * window_side) = 141 := 
by
  sorry

end miguel_paint_area_l56_56087


namespace jason_egg_consumption_l56_56803

-- Definition for the number of eggs Jason consumes per day
def eggs_per_day : ℕ := 3

-- Definition for the number of days in a week
def days_in_week : ℕ := 7

-- Definition for the number of weeks we are considering
def weeks : ℕ := 2

-- The statement we want to prove, which combines all the conditions and provides the final answer
theorem jason_egg_consumption : weeks * days_in_week * eggs_per_day = 42 := by
sorry

end jason_egg_consumption_l56_56803


namespace sin_theta_val_sin_2theta_pi_div_6_val_l56_56524

open Real

theorem sin_theta_val (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2) 
  (hcos : cos (θ + π / 6) = 1 / 3) : 
  sin θ = (2 * sqrt 6 - 1) / 6 := 
by sorry

theorem sin_2theta_pi_div_6_val (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (hcos : cos (θ + π / 6) = 1 / 3) : 
  sin (2 * θ + π / 6) = (4 * sqrt 6 + 7) / 18 := 
by sorry

end sin_theta_val_sin_2theta_pi_div_6_val_l56_56524


namespace estimated_red_balls_l56_56277

-- Definitions based on conditions
def total_balls : ℕ := 15
def black_ball_frequency : ℝ := 0.6
def red_ball_frequency : ℝ := 1 - black_ball_frequency

-- Theorem stating the proof problem
theorem estimated_red_balls :
  (total_balls : ℝ) * red_ball_frequency = 6 := by
  sorry

end estimated_red_balls_l56_56277


namespace avg_age_of_community_l56_56693

def ratio_of_populations (w m : ℕ) : Prop := w * 2 = m * 3
def avg_age (total_age population : ℚ) : ℚ := total_age / population

theorem avg_age_of_community 
    (k : ℕ)
    (total_women : ℕ := 3 * k) 
    (total_men : ℕ := 2 * k)
    (total_children : ℚ := (2 * k : ℚ) / 3)
    (avg_women_age : ℚ := 40)
    (avg_men_age : ℚ := 36)
    (avg_children_age : ℚ := 10)
    (total_women_age : ℚ := 40 * (3 * k))
    (total_men_age : ℚ := 36 * (2 * k))
    (total_children_age : ℚ := 10 * (total_children)) : 
    avg_age (total_women_age + total_men_age + total_children_age) (total_women + total_men + total_children) = 35 := 
    sorry

end avg_age_of_community_l56_56693


namespace minimum_familiar_pairs_l56_56205

theorem minimum_familiar_pairs (n : ℕ) (students : Finset (Fin n)) 
  (familiar : Finset (Fin n × Fin n))
  (h_n : n = 175)
  (h_condition : ∀ (s : Finset (Fin n)), s.card = 6 → 
    ∃ (s1 s2 : Finset (Fin n)), s1 ∪ s2 = s ∧ s1.card = 3 ∧ s2.card = 3 ∧ 
    ∀ x ∈ s1, ∀ y ∈ s1, (x ≠ y → (x, y) ∈ familiar) ∧
    ∀ x ∈ s2, ∀ y ∈ s2, (x ≠ y → (x, y) ∈ familiar)) :
  ∃ m : ℕ, m = 15050 ∧ ∀ p : ℕ, (∃ g : Finset (Fin n × Fin n), g.card = p) → p ≥ m := 
sorry

end minimum_familiar_pairs_l56_56205


namespace kendra_sunday_shirts_l56_56409

def total_shirts := 22
def shirts_weekdays := 5 * 1
def shirts_after_school := 3
def shirts_saturday := 1

theorem kendra_sunday_shirts : 
  (total_shirts - 2 * (shirts_weekdays + shirts_after_school + shirts_saturday)) = 4 :=
by
  sorry

end kendra_sunday_shirts_l56_56409


namespace A_eq_B_l56_56288

def A : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 + 2 * y^2
}

def B : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 - 6 * x * y + 11 * y^2
}

theorem A_eq_B : A = B :=
by {
  sorry
}

end A_eq_B_l56_56288


namespace positive_diff_even_odd_sums_l56_56129

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l56_56129


namespace complete_square_l56_56908

-- Definitions based on conditions
def row_sum_piece2 := 2 + 1 + 3 + 1
def total_sum_square := 4 * row_sum_piece2
def sum_piece1 := 7
def sum_piece2 := 8
def sum_piece3 := 8
def total_given_pieces := sum_piece1 + sum_piece2 + sum_piece3
def sum_missing_piece := total_sum_square - total_given_pieces

-- Statement to prove that the missing piece has the correct sum
theorem complete_square : (sum_missing_piece = 5) :=
by 
  -- It is a placeholder for the proof steps, the actual proof steps are not needed
  sorry

end complete_square_l56_56908


namespace cyclic_points_exist_l56_56414

noncomputable def f (x : ℝ) : ℝ := 
if x < (1 / 3) then 
  2 * x + (1 / 3) 
else 
  (3 / 2) * (1 - x)

theorem cyclic_points_exist :
  ∃ (x0 x1 x2 x3 x4 : ℝ), 
  0 ≤ x0 ∧ x0 ≤ 1 ∧
  0 ≤ x1 ∧ x1 ≤ 1 ∧
  0 ≤ x2 ∧ x2 ≤ 1 ∧
  0 ≤ x3 ∧ x3 ≤ 1 ∧
  0 ≤ x4 ∧ x4 ≤ 1 ∧
  x0 ≠ x1 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ x4 ≠ x0 ∧
  f x0 = x1 ∧ f x1 = x2 ∧ f x2 = x3 ∧ f x3 = x4 ∧ f x4 = x0 :=
sorry

end cyclic_points_exist_l56_56414


namespace pyramid_volume_of_unit_cube_l56_56636

noncomputable def volume_of_pyramid : ℝ :=
  let s := (Real.sqrt 2) / 2
  let base_area := (Real.sqrt 3) / 8
  let height := 1
  (1 / 3) * base_area * height

theorem pyramid_volume_of_unit_cube :
  volume_of_pyramid = (Real.sqrt 3) / 24 := by
  sorry

end pyramid_volume_of_unit_cube_l56_56636


namespace minimum_value_of_m_l56_56045

-- Define a function to determine if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Define a function to determine if a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

-- Our main theorem statement
theorem minimum_value_of_m :
  ∃ m : ℕ, (600 < m ∧ m ≤ 800) ∧
           is_perfect_square (3 * m) ∧
           is_perfect_cube (5 * m) :=
sorry

end minimum_value_of_m_l56_56045


namespace parabolas_vertex_condition_l56_56903

theorem parabolas_vertex_condition (p q x₁ x₂ y₁ y₂ : ℝ) (h1: y₂ = p * (x₂ - x₁)^2 + y₁) (h2: y₁ = q * (x₁ - x₂)^2 + y₂) (h3: x₁ ≠ x₂) : p + q = 0 :=
sorry

end parabolas_vertex_condition_l56_56903


namespace area_of_triangle_ABC_l56_56293

open Real

noncomputable def triangle_area (b c : ℝ) : ℝ :=
  (sqrt 2 / 4) * (sqrt (4 + b^2)) * (sqrt (4 + c^2))

theorem area_of_triangle_ABC (b c : ℝ) :
  let O : ℝ × ℝ × ℝ := (0, 0, 0)
  let A : ℝ × ℝ × ℝ := (2, 0, 0)
  let B : ℝ × ℝ × ℝ := (0, b, 0)
  let C : ℝ × ℝ × ℝ := (0, 0, c)
  let angle_BAC : ℝ := 45
  (cos (angle_BAC * π / 180) = sqrt 2 / 2) →
  (sin (angle_BAC * π / 180) = sqrt 2 / 2) →
  let AB := sqrt (2^2 + b^2)
  let AC := sqrt (2^2 + c^2)
  let area := (1/2) * AB * AC * (sin (45 * π / 180))
  area = triangle_area b c :=
sorry

end area_of_triangle_ABC_l56_56293


namespace positive_difference_prob_3_and_4_heads_l56_56755

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_prob_3_and_4_heads (p : ℚ) (hp : p = 1/2) (n : ℕ) (hn : n = 4) :
  let p1 := binomial_prob n 3 p in
  let p2 := binomial_prob n 4 p in
  p1 - p2 = 3/16 :=
by
  sorry

end positive_difference_prob_3_and_4_heads_l56_56755


namespace find_numbers_l56_56243

theorem find_numbers (x y : ℕ) :
  x + y = 1244 →
  10 * x + 3 = (y - 2) / 10 →
  x = 12 ∧ y = 1232 :=
by
  intro h_sum h_trans
  -- We'll use sorry here to state that the proof is omitted.
  sorry

end find_numbers_l56_56243


namespace minimum_value_l56_56055

variable {a b : ℝ}

noncomputable def given_conditions (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + 2 * b = 2

theorem minimum_value :
  given_conditions a b →
  ∃ x, x = (1 + 4 * a + 3 * b) / (a * b) ∧ x ≥ 25 / 2 :=
by
  sorry

end minimum_value_l56_56055


namespace positive_difference_eq_250_l56_56119

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l56_56119


namespace jim_taxi_distance_l56_56282

theorem jim_taxi_distance (initial_fee charge_per_segment total_charge : ℝ) (segment_len_miles : ℝ)
(init_fee_eq : initial_fee = 2.5)
(charge_per_seg_eq : charge_per_segment = 0.35)
(total_charge_eq : total_charge = 5.65)
(segment_length_eq : segment_len_miles = 2/5):
  let charge_for_distance := total_charge - initial_fee
  let num_segments := charge_for_distance / charge_per_segment
  let total_miles := num_segments * segment_len_miles
  total_miles = 3.6 :=
by
  intros
  sorry

end jim_taxi_distance_l56_56282


namespace solve_quadratic_equation_l56_56554

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ (x = 4 ∨ x = -2) :=
by sorry

end solve_quadratic_equation_l56_56554


namespace evaluate_f_x_l56_56394

def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 2 * x^2 + 4 * x

theorem evaluate_f_x : f 3 - f (-3) = 672 :=
by
  -- Proof omitted
  sorry

end evaluate_f_x_l56_56394


namespace weekly_charge_for_motel_l56_56794

theorem weekly_charge_for_motel (W : ℝ) (h1 : ∀ t : ℝ, t = 3 * 4 → t = 12)
(h2 : ∀ cost_weekly : ℝ, cost_weekly = 12 * W)
(h3 : ∀ cost_monthly : ℝ, cost_monthly = 3 * 1000)
(h4 : cost_monthly + 360 = 12 * W) : 
W = 280 := 
sorry

end weekly_charge_for_motel_l56_56794


namespace interest_rate_decrease_l56_56787

theorem interest_rate_decrease (initial_rate final_rate : ℝ) (x : ℝ) 
  (h_initial_rate : initial_rate = 2.25 * 0.01)
  (h_final_rate : final_rate = 1.98 * 0.01) :
  final_rate = initial_rate * (1 - x)^2 := 
  sorry

end interest_rate_decrease_l56_56787


namespace reporters_cover_local_politics_l56_56012

structure Reporters :=
(total : ℕ)
(politics : ℕ)
(local_politics : ℕ)

def percentages (reporters : Reporters) : Prop :=
  reporters.politics = (40 * reporters.total) / 100 ∧
  reporters.local_politics = (75 * reporters.politics) / 100

theorem reporters_cover_local_politics (reporters : Reporters) (h : percentages reporters) :
  (reporters.local_politics * 100) / reporters.total = 30 :=
by
  -- Proof steps would be added here
  sorry

end reporters_cover_local_politics_l56_56012


namespace walk_to_bus_stop_usual_time_l56_56765

variable (S : ℝ) -- assuming S is the usual speed, a positive real number
variable (T : ℝ) -- assuming T is the usual time, which we need to determine
variable (new_speed : ℝ := (4 / 5) * S) -- the new speed is 4/5 of usual speed
noncomputable def time_to_bus_at_usual_speed : ℝ := T -- time to bus stop at usual speed

theorem walk_to_bus_stop_usual_time :
  (time_to_bus_at_usual_speed S = 30) ↔ (S * (T + 6) = (4 / 5) * S * T) :=
by
  sorry

end walk_to_bus_stop_usual_time_l56_56765


namespace positive_difference_sums_even_odd_l56_56142

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l56_56142


namespace factorize_a_squared_plus_2a_l56_56809

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2*a = a * (a + 2) :=
sorry

end factorize_a_squared_plus_2a_l56_56809


namespace correct_reference_l56_56687

variable (house : String) 
variable (beautiful_garden_in_front : Bool)
variable (I_like_this_house : Bool)
variable (enough_money_to_buy : Bool)

-- Statement: Given the conditions, prove that the correct word to fill in the blank is "it".
theorem correct_reference : I_like_this_house ∧ beautiful_garden_in_front ∧ ¬ enough_money_to_buy → "it" = "correct choice" :=
by
  sorry

end correct_reference_l56_56687


namespace modulus_of_z_l56_56834

open Complex

theorem modulus_of_z (z : ℂ) (hz : (1 + I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l56_56834


namespace skating_minutes_needed_l56_56246

-- Define the conditions
def minutes_per_day (day: ℕ) : ℕ :=
  if day ≤ 4 then 80 else if day ≤ 6 then 100 else 0

-- Define total skating time up to 6 days
def total_time_six_days := (4 * 80) + (2 * 100)

-- Prove that Gage needs to skate 180 minutes on the seventh day
theorem skating_minutes_needed : 
  (total_time_six_days + x = 7 * 100) → x = 180 :=
by sorry

end skating_minutes_needed_l56_56246


namespace second_largest_consecutive_odd_195_l56_56897

theorem second_largest_consecutive_odd_195 :
  ∃ x : Int, (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 195 ∧ (x + 2) = 41 := by
  sorry

end second_largest_consecutive_odd_195_l56_56897


namespace voldemort_calorie_intake_limit_l56_56599

theorem voldemort_calorie_intake_limit :
  let breakfast := 560
  let lunch := 780
  let cake := 110
  let chips := 310
  let coke := 215
  let dinner := cake + chips + coke
  let remaining := 525
  breakfast + lunch + dinner + remaining = 2500 :=
by
  -- to clarify, the statement alone is provided, so we add 'sorry' to omit the actual proof steps
  sorry

end voldemort_calorie_intake_limit_l56_56599


namespace area_of_DEF_isosceles_right_triangle_l56_56475

noncomputable def area_triangle_DEF (A B C D E F : ℝ) : ℝ :=
    if hABC : ∀ (A B C : ℝ), isosceles_right_triangle A B C with (AB = 2),
       hDBC : ∀ (D B C : ℝ), isosceles_right_triangle D B C with (hypotenuse = BC),
       hAEC : ∀ (A E C : ℝ), isosceles_right_triangle A E C with (hypotenuse = AC),
       hABF : ∀ (A B F : ℝ), isosceles_right_triangle A B F with (hypotenuse = AB)
    then
       let AB : ℝ := 2
       let AC : ℝ := AB / real.sqrt 2
       let BC : ℝ := AB / real.sqrt 2
       let BD : ℝ := BC / real.sqrt 2
       let DC : ℝ := BC / real.sqrt 2
       let AE : ℝ := AC / real.sqrt 2
       let EC : ℝ := AC / real.sqrt 2
       let AF : ℝ := AB / real.sqrt 2
       let BF : ℝ := AB / real.sqrt 2
       (1/2) * 2 * 2
    else 0

theorem area_of_DEF_isosceles_right_triangle :
    ∀ (A B C D E F : ℝ), 
        isosceles_right_triangle A B C → 
        isosceles_right_triangle D B C → 
        isosceles_right_triangle A E C → 
        isosceles_right_triangle A B F → 
        area_triangle_DEF A B C D E F = 2 :=
by intros; exact sorry

end area_of_DEF_isosceles_right_triangle_l56_56475


namespace automobile_travel_distance_l56_56222

theorem automobile_travel_distance (a r : ℝ) :
  (2 * a / 5) / (2 * r) * 5 * 60 / 3 = 20 * a / r :=
by 
  -- skipping proof details
  sorry

end automobile_travel_distance_l56_56222


namespace h_eq_x_solution_l56_56415

noncomputable def h (x : ℝ) : ℝ := (3 * ((x + 3) / 5) + 10)

theorem h_eq_x_solution (x : ℝ) (h_cond : ∀ y, h (5 * y - 3) = 3 * y + 10) : h x = x → x = 29.5 :=
by
  sorry

end h_eq_x_solution_l56_56415


namespace negation_universal_to_particular_l56_56058

theorem negation_universal_to_particular :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  sorry

end negation_universal_to_particular_l56_56058


namespace sum_of_consecutive_pairs_eq_pow_two_l56_56732

theorem sum_of_consecutive_pairs_eq_pow_two (n m : ℕ) :
  ∃ n m : ℕ, (n * (n + 1) + m * (m + 1) = 2 ^ 2021) :=
sorry

end sum_of_consecutive_pairs_eq_pow_two_l56_56732


namespace major_axis_length_l56_56638

def length_of_major_axis 
  (tangent_x : ℝ) (f1 : ℝ × ℝ) (f2 : ℝ × ℝ) : ℝ :=
  sorry

theorem major_axis_length 
  (hx_tangent : (4, 0) = (4, 0)) 
  (foci : (4, 2 + 2 * Real.sqrt 2) = (4, 2 + 2 * Real.sqrt 2) ∧ 
         (4, 2 - 2 * Real.sqrt 2) = (4, 2 - 2 * Real.sqrt 2)) :
  length_of_major_axis 4 
  (4, 2 + 2 * Real.sqrt 2) (4, 2 - 2 * Real.sqrt 2) = 4 :=
sorry

end major_axis_length_l56_56638


namespace smallest_solution_fraction_eq_l56_56816

theorem smallest_solution_fraction_eq (x : ℝ) (h : x ≠ 3) :
    3 * x / (x - 3) + (3 * x^2 - 27) / x = 16 ↔ x = (2 - Real.sqrt 31) / 3 := 
sorry

end smallest_solution_fraction_eq_l56_56816


namespace point_on_x_axis_l56_56836

theorem point_on_x_axis (a : ℝ) (h : (1, a + 1).snd = 0) : a = -1 :=
by
  sorry

end point_on_x_axis_l56_56836


namespace square_of_cube_of_smallest_prime_l56_56189

def smallest_prime : Nat := 2

theorem square_of_cube_of_smallest_prime :
  ((smallest_prime ^ 3) ^ 2) = 64 := by
  sorry

end square_of_cube_of_smallest_prime_l56_56189


namespace range_of_k_l56_56823

theorem range_of_k 
  (x1 x2 y1 y2 k : ℝ)
  (h1 : y1 = 2 * x1 - k * x1 + 1)
  (h2 : y2 = 2 * x2 - k * x2 + 1)
  (h3 : x1 ≠ x2)
  (h4 : (x1 - x2) * (y1 - y2) < 0) : k > 2 := 
sorry

end range_of_k_l56_56823


namespace count_valid_n_l56_56244

theorem count_valid_n : ∃ (count : ℕ), count = 6 ∧ ∀ n : ℕ,
  0 < n ∧ n < 42 → (∃ m : ℕ, m > 0 ∧ n = 42 * m / (m + 1)) :=
by
  sorry

end count_valid_n_l56_56244


namespace cylinder_volume_options_l56_56309

theorem cylinder_volume_options (length width : ℝ) (h₀ : length = 4) (h₁ : width = 2) :
  ∃ V, (V = (4 / π) ∨ V = (8 / π)) :=
by
  sorry

end cylinder_volume_options_l56_56309


namespace velocity_of_current_l56_56760

theorem velocity_of_current
  (v c : ℝ) 
  (h1 : 32 = (v + c) * 6) 
  (h2 : 14 = (v - c) * 6) :
  c = 1.5 :=
by
  sorry

end velocity_of_current_l56_56760


namespace perimeter_of_triangle_equation_of_line_through_vertex_line_AD_tangent_to_C_l56_56052

-- Definition of an Ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Definition of a Circle
def circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Conditions about foci, points, lines, and tangent property
def foci_condition (x y : ℝ) : Prop := ellipse x y
def right_vertex (x y : ℝ) : Prop := x = 2 ∧ y = 0
def tangent_condition (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y → l x y

-- Problem 1: Prove the perimeter of Δ AF₁F₂ is 4 + 2√2
theorem perimeter_of_triangle (A F1 F2 : ℝ × ℝ) (hA : foci_condition A.1 A.2)
  (hF1 : ellipse F1.1 F1.2) (hF2 : ellipse F2.1 F2.2) : 
  dist A F1 + dist A F2 + dist F1 F2 = 4 + 2 * sqrt 2 := sorry

-- Problem 2: Find the equation of line l passing through (2,0)
theorem equation_of_line_through_vertex : 
  ∃ k : ℝ, ∀ x y : ℝ, (x = 2 → y = k * (x - 2)) → tangent_condition (λ x y, y = k * (x - 2)) := sorry

-- Problem 3: Prove line AD is tangent to circle C
theorem line_AD_tangent_to_C (A D : ℝ × ℝ) (hD : D.2 = 2) (hO : 0) (hperpendicular : A.1 * D.1 + A.2 * D.2 = 0) :
  ∃ l : ℝ → ℝ → Prop, tangent_condition l ∧ (l A.1 A.2 = A.2 → A ≠ O) := sorry

end perimeter_of_triangle_equation_of_line_through_vertex_line_AD_tangent_to_C_l56_56052


namespace usual_time_to_school_l56_56471

-- Define the conditions
variables (R T : ℝ) (h1 : 0 < T) (h2 : 0 < R)
noncomputable def boy_reaches_school_early : Prop :=
  (7/6 * R) * (T - 5) = R * T

-- The theorem stating the usual time to reach the school
theorem usual_time_to_school (h : boy_reaches_school_early R T) : T = 35 :=
by
  sorry

end usual_time_to_school_l56_56471


namespace belt_and_road_scientific_notation_l56_56588

theorem belt_and_road_scientific_notation : 
  4600000000 = 4.6 * 10^9 := 
by
  sorry

end belt_and_road_scientific_notation_l56_56588


namespace positive_difference_l56_56157

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l56_56157


namespace store_owner_uniforms_l56_56353

theorem store_owner_uniforms (U E : ℕ) (h1 : U + 1 = 2 * E) (h2 : U % 2 = 1) : U = 3 := 
sorry

end store_owner_uniforms_l56_56353


namespace tens_digit_2023_pow_2024_minus_2025_l56_56030

theorem tens_digit_2023_pow_2024_minus_2025 :
  (2023 ^ 2024 - 2025) % 100 / 10 % 10 = 5 :=
sorry

end tens_digit_2023_pow_2024_minus_2025_l56_56030


namespace coin_probability_difference_l56_56749

theorem coin_probability_difference :
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  p3 - p4 = (7/16 : ℚ) :=
by
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  have h1 : p3 = (1/2 : ℚ), by norm_num [finset.range_succ]
  have h2 : p4 = (1/16 : ℚ), by norm_num
  rw [h1, h2]
  norm_num

end coin_probability_difference_l56_56749


namespace kathryn_gave_56_pencils_l56_56939

-- Define the initial and total number of pencils
def initial_pencils : ℕ := 9
def total_pencils : ℕ := 65

-- Define the number of pencils Kathryn gave to Anthony
def pencils_given : ℕ := total_pencils - initial_pencils

-- Prove that Kathryn gave Anthony 56 pencils
theorem kathryn_gave_56_pencils : pencils_given = 56 :=
by
  -- Proof is omitted as per the requirement
  sorry

end kathryn_gave_56_pencils_l56_56939


namespace age_difference_correct_l56_56405

-- Define the ages of John and his parents based on given conditions
def John_age (father_age : ℕ) : ℕ :=
  father_age / 2

def mother_age (father_age : ℕ) : ℕ :=
  father_age - 4

def age_difference (john_age : ℕ) (mother_age : ℕ) : ℕ :=
  abs (john_age - mother_age)

-- Main theorem stating the age difference between John and his mother
theorem age_difference_correct (father_age : ℕ) (h : father_age = 40) :
  age_difference (John_age father_age) (mother_age father_age) = 16 :=
by
  sorry

end age_difference_correct_l56_56405


namespace identity_proof_l56_56114

theorem identity_proof (A B C A1 B1 C1 : ℝ) :
  (A^2 + B^2 + C^2) * (A1^2 + B1^2 + C1^2) - (A * A1 + B * B1 + C * C1)^2 =
    (A * B1 + A1 * B)^2 + (A * C1 + A1 * C)^2 + (B * C1 + B1 * C)^2 :=
by
  sorry

end identity_proof_l56_56114


namespace booster_club_tickets_l56_56304

theorem booster_club_tickets (x : ℕ) : 
  (11 * 9 + x * 7 = 225) → 
  (x + 11 = 29) := 
by
  sorry

end booster_club_tickets_l56_56304


namespace minimize_m_at_l56_56050

noncomputable def m (x y : ℝ) : ℝ := 4 * x ^ 2 - 12 * x * y + 10 * y ^ 2 + 4 * y + 9

theorem minimize_m_at (x y : ℝ) : m x y = 5 ↔ (x = -3 ∧ y = -2) := 
sorry

end minimize_m_at_l56_56050


namespace revolutions_same_distance_l56_56216

theorem revolutions_same_distance (r R : ℝ) (revs_30 : ℝ) (dist_30 dist_10 : ℝ)
  (h_radius: r = 10) (H_radius: R = 30) (h_revs_30: revs_30 = 15) 
  (H_dist_30: dist_30 = 2 * Real.pi * R * revs_30) 
  (H_dist_10: dist_10 = 2 * Real.pi * r * 45) :
  dist_30 = dist_10 :=
by {
  sorry
}

end revolutions_same_distance_l56_56216


namespace suitable_for_experimental_method_is_meters_run_l56_56336

-- Define the options as a type
inductive ExperimentalOption
| recommending_class_monitor_candidates
| surveying_classmates_birthdays
| meters_run_in_10_seconds
| avian_influenza_occurrences_world

-- Define a function that checks if an option is suitable for the experimental method
def is_suitable_for_experimental_method (option: ExperimentalOption) : Prop :=
  option = ExperimentalOption.meters_run_in_10_seconds

-- The theorem stating which option is suitable for the experimental method
theorem suitable_for_experimental_method_is_meters_run :
  is_suitable_for_experimental_method ExperimentalOption.meters_run_in_10_seconds :=
by
  sorry

end suitable_for_experimental_method_is_meters_run_l56_56336


namespace simplify_fraction_l56_56878

theorem simplify_fraction : ∃ (a b : ℕ), a = 90 ∧ b = 150 ∧ (90:ℚ) / (150:ℚ) = (3:ℚ) / (5:ℚ) :=
by {
  use 90,
  use 150,
  split,
  refl,
  split,
  refl,
  sorry,
}

end simplify_fraction_l56_56878


namespace factorize_a_squared_plus_2a_l56_56807

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2 * a = a * (a + 2) :=
  sorry

end factorize_a_squared_plus_2a_l56_56807


namespace value_of_m_l56_56969

theorem value_of_m (m : ℝ) :
  (∀ A B : ℝ × ℝ, A = (m + 1, -2) → B = (3, m - 1) → (A.snd = B.snd) → m = -1) :=
by
  intros A B hA hB h_parallel
  -- Apply the given conditions and assumptions to prove the value of m.
  sorry

end value_of_m_l56_56969


namespace olive_needs_two_colours_l56_56430

theorem olive_needs_two_colours (α : Type) [Finite α] (G : SimpleGraph α) (colour : α → Fin 2) :
  (∀ v : α, ∃! w : α, G.Adj v w ∧ colour v = colour w) → ∃ color_map : α → Fin 2, ∀ v, ∃! w, G.Adj v w ∧ color_map v = color_map w :=
sorry

end olive_needs_two_colours_l56_56430


namespace part_I_part_II_l56_56384

-- Define the sets A and B for the given conditions
def setA : Set ℝ := {x | -3 ≤ x - 2 ∧ x - 2 ≤ 1}
def setB (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Part (Ⅰ) When a = 1, find A ∩ B
theorem part_I (a : ℝ) (ha : a = 1) :
  (setA ∩ setB a) = {x | 0 ≤ x ∧ x ≤ 3} :=
by
  sorry

-- Part (Ⅱ) If A ∪ B = A, find the range of real number a
theorem part_II : 
  (∀ a : ℝ, setA ∪ setB a = setA → 0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end part_I_part_II_l56_56384


namespace quadratic_complete_square_l56_56805

theorem quadratic_complete_square :
  ∀ x : ℝ, x^2 - 4 * x + 5 = (x - 2)^2 + 1 :=
by
  intro x
  sorry

end quadratic_complete_square_l56_56805


namespace tangent_line_eq_l56_56591

theorem tangent_line_eq (x y: ℝ):
  (x^2 + y^2 = 4) → ((2, 3) = (x, y)) →
  (x = 2 ∨ 5 * x - 12 * y + 26 = 0) :=
by
  sorry

end tangent_line_eq_l56_56591


namespace probability_of_defective_product_l56_56492

theorem probability_of_defective_product :
  let total_products := 10
  let defective_products := 2
  (defective_products: ℚ) / total_products = 1 / 5 :=
by
  let total_products := 10
  let defective_products := 2
  have h : (defective_products: ℚ) / total_products = 1 / 5
  {
    exact sorry
  }
  exact h

end probability_of_defective_product_l56_56492


namespace triangle_base_l56_56840

theorem triangle_base (h : ℝ) (A : ℝ) (b : ℝ) (h_eq : h = 10) (A_eq : A = 46) (area_eq : A = (b * h) / 2) : b = 9.2 :=
by
  -- sorry to be replaced with the actual proof
  sorry

end triangle_base_l56_56840


namespace Daria_money_l56_56027

theorem Daria_money (num_tickets : ℕ) (price_per_ticket : ℕ) (amount_needed : ℕ) (h1 : num_tickets = 4) (h2 : price_per_ticket = 90) (h3 : amount_needed = 171) : 
  (num_tickets * price_per_ticket) - amount_needed = 189 := 
by 
  sorry

end Daria_money_l56_56027


namespace positive_difference_even_odd_sum_l56_56185

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l56_56185


namespace arithmetic_sequence_sum_l56_56701

variable {α : Type*} [LinearOrderedField α]

noncomputable def S (n a_1 d : α) : α :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem arithmetic_sequence_sum (a_1 d : α) :
  S 5 a_1 d = 5 → S 9 a_1 d = 27 → S 7 a_1 d = 14 :=
by
  sorry

end arithmetic_sequence_sum_l56_56701


namespace number_of_prime_divisors_of_50_l56_56543

theorem number_of_prime_divisors_of_50! : 
  (∃ primes : Finset ℕ, (∀ p ∈ primes, nat.prime p) ∧ primes.card = 15 ∧ ∀ p ∈ primes, p ≤ 50) := by
sorry

end number_of_prime_divisors_of_50_l56_56543


namespace absolute_value_example_l56_56689

theorem absolute_value_example (x : ℝ) (h : x = 4) : |x - 5| = 1 :=
by
  sorry

end absolute_value_example_l56_56689


namespace impossible_300_numbers_l56_56769

theorem impossible_300_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) (hp : ∀ i, 0 < a i)
(hdiff : ∃ k, ∀ i ≠ k, a i = a ((i + 1) % n) - a ((i - 1 + n) % n)) 
: false :=
by {
  sorry
}

end impossible_300_numbers_l56_56769


namespace triangle_right_angled_and_common_difference_equals_inscribed_circle_radius_l56_56576

noncomputable def a : ℝ := sorry
noncomputable def d : ℝ := a / 4
noncomputable def half_perimeter : ℝ := (a - d + a + (a + d)) / 2
noncomputable def r : ℝ := ((a - d) + a + (a + d)) / 2

theorem triangle_right_angled_and_common_difference_equals_inscribed_circle_radius :
  (half_perimeter > a + d) →
  ((a - d) + a + (a + d) = 2 * half_perimeter) →
  (a - d)^2 + a^2 = (a + d)^2 →
  d = r :=
by
  intros h1 h2 h3
  sorry

end triangle_right_angled_and_common_difference_equals_inscribed_circle_radius_l56_56576


namespace trajectory_of_C_l56_56397

-- Definitions of points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 3)

-- Definition of point C as a linear combination of points A and B
def C (α β : ℝ) : ℝ × ℝ := (α * A.1 + β * B.1, α * A.2 + β * B.2)

-- The main theorem statement to prove the equation of the trajectory of point C
theorem trajectory_of_C (x y α β : ℝ)
  (h_cond : α + β = 1)
  (h_C : (x, y) = C α β) : 
  x + 2*y = 5 := 
sorry -- Proof to be skipped

end trajectory_of_C_l56_56397


namespace quadratic_real_roots_probability_l56_56884

theorem quadratic_real_roots_probability :
  let outcomes := (fin_fun (λ n : Fin 36, (1, 1) * (n + 1))),
  let B_C := finset.product (finset.of_finite outcomes) (finset.of_finite outcomes),
  let real_roots_count := B_C.filter (λ bc, let ⟨b, c⟩ := bc in b^2 - 4 * c ≥ 0),
  (real_roots_count.card : ℝ) / (B_C.card : ℝ) = 19 / 36 :=
by sorry

end quadratic_real_roots_probability_l56_56884


namespace positive_difference_prob_3_and_4_heads_l56_56750

theorem positive_difference_prob_3_and_4_heads :
  let p_3 := (choose 4 3) * (1 / 2) ^ 3 * (1 / 2) in
  let p_4 := (1 / 2) ^ 4 in
  abs (p_3 - p_4) = 7 / 16 :=
by
  -- Definitions for binomial coefficient and probabilities
  let p_3 := (Nat.choose 4 3) * (1 / 2)^3 * (1 / 2)
  let p_4 := (1 / 2)^4
  -- The difference between probabilities
  let diff := p_3 - p_4
  -- The desired equality to prove
  show abs diff = 7 / 16
  sorry

end positive_difference_prob_3_and_4_heads_l56_56750


namespace soccer_claim_fraction_l56_56790

theorem soccer_claim_fraction 
  (total_students enjoy_soccer do_not_enjoy_soccer claim_do_not_enjoy honesty fraction_3_over_11 : ℕ)
  (h1 : enjoy_soccer = total_students / 2)
  (h2 : do_not_enjoy_soccer = total_students / 2)
  (h3 : claim_do_not_enjoy = enjoy_soccer * 3 / 10)
  (h4 : honesty = do_not_enjoy_soccer * 8 / 10)
  (h5 : fraction_3_over_11 = enjoy_soccer * 3 / (10 * (enjoy_soccer * 3 / 10 + do_not_enjoy_soccer * 2 / 10)))
  : fraction_3_over_11 = 3 / 11 :=
sorry

end soccer_claim_fraction_l56_56790


namespace find_optimal_addition_l56_56111

theorem find_optimal_addition (m : ℝ) : 
  (1000 + (m - 1000) * 0.618 = 1618 ∨ 1000 + (m - 1000) * 0.618 = 2618) →
  (m = 2000 ∨ m = 2618) :=
sorry

end find_optimal_addition_l56_56111


namespace negation_proposition_l56_56388

variable (a : ℝ)

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) :=
by
  sorry

end negation_proposition_l56_56388


namespace bus_travel_time_l56_56346

theorem bus_travel_time :
  let departure_time := Time.mk ⟨9, 30⟩
  let arrival_time := Time.mk ⟨12, 30⟩
  (arrival_time - departure_time).hours = 3 :=
by
  let departure_time := Time.mk ⟨9, 30⟩
  let arrival_time := Time.mk ⟨12, 30⟩
  have h : (arrival_time - departure_time).hours = 3 := sorry
  exact h

end bus_travel_time_l56_56346


namespace forming_n_and_m_l56_56327

def is_created_by_inserting_digit (n: ℕ) (base: ℕ): Prop :=
  ∃ d1 d2 d3 d: ℕ, n = d1 * 1000 + d * 100 + d2 * 10 + d3 ∧ base = d1 * 100 + d2 * 10 + d3

theorem forming_n_and_m (a b: ℕ) (base: ℕ) (sum: ℕ) 
  (h1: is_created_by_inserting_digit a base)
  (h2: is_created_by_inserting_digit b base) 
  (h3: a + b = sum):
  (a = 2195 ∧ b = 2165) 
  ∨ (a = 2185 ∧ b = 2175) 
  ∨ (a = 2215 ∧ b = 2145) 
  ∨ (a = 2165 ∧ b = 2195) 
  ∨ (a = 2175 ∧ b = 2185) 
  ∨ (a = 2145 ∧ b = 2215) := 
sorry

end forming_n_and_m_l56_56327


namespace average_after_discard_l56_56621

theorem average_after_discard (avg : ℝ) (n : ℕ) (a b : ℝ) (new_avg : ℝ) :
  avg = 62 →
  n = 50 →
  a = 45 →
  b = 55 →
  new_avg = 62.5 →
  (avg * n - (a + b)) / (n - 2) = new_avg := 
by
  intros h_avg h_n h_a h_b h_new_avg
  rw [h_avg, h_n, h_a, h_b, h_new_avg]
  sorry

end average_after_discard_l56_56621


namespace value_of_f_at_112_5_l56_56083

noncomputable def f : ℝ → ℝ := sorry

lemma f_even_func (x : ℝ) : f x = f (-x) := sorry
lemma f_func_eq (x : ℝ) : f x + f (x + 1) = 4 := sorry
lemma f_interval : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x = 4 * x + 12 := sorry

theorem value_of_f_at_112_5 : f 112.5 = 2 := sorry

end value_of_f_at_112_5_l56_56083


namespace gcd_sequence_condition_l56_56520

theorem gcd_sequence_condition (p q : ℕ) (hp : 0 < p) (hq : 0 < q)
  (a : ℕ → ℕ)
  (ha1 : a 1 = 1) (ha2 : a 2 = 1) 
  (ha_rec : ∀ n, a (n + 2) = p * a (n + 1) + q * a n) 
  (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (gcd (a m) (a n) = a (gcd m n)) ↔ (p = 1) := 
sorry

end gcd_sequence_condition_l56_56520


namespace positive_difference_even_odd_sums_l56_56180

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l56_56180


namespace percentage_goods_lost_eq_l56_56015

-- Define the initial conditions
def initial_value : ℝ := 100
def profit_margin : ℝ := 0.10 * initial_value
def selling_price : ℝ := initial_value + profit_margin
def loss_percentage : ℝ := 0.12

-- Define the correct answer as a constant
def correct_percentage_loss : ℝ := 13.2

-- Define the target theorem
theorem percentage_goods_lost_eq : (0.12 * selling_price / initial_value * 100) = correct_percentage_loss := 
by
  -- sorry is used to skip the proof part as per instructions
  sorry

end percentage_goods_lost_eq_l56_56015


namespace largest_crowd_size_l56_56197

theorem largest_crowd_size (x : ℕ) : 
  (ceil (x * (1 / 2)) + ceil (x * (1 / 3)) + ceil (x * (1 / 5)) = x) →
  x ≤ 37 :=
sorry

end largest_crowd_size_l56_56197


namespace blue_paper_side_length_l56_56581

theorem blue_paper_side_length (side_red : ℝ) (side_blue : ℝ) (same_area : side_red^2 = side_blue * x) (side_red_val : side_red = 5) (side_blue_val : side_blue = 4) : x = 6.25 :=
by
  sorry

end blue_paper_side_length_l56_56581


namespace skyscraper_anniversary_l56_56987

theorem skyscraper_anniversary (current_year_event future_happens_year target_anniversary_year : ℕ) :
  current_year_event + future_happens_year = target_anniversary_year - 5 →
  target_anniversary_year > current_year_event →
  future_happens_year = 95 := 
by
  sorry

-- Definitions for conditions:
def current_year_event := 100
def future_happens_year := 95
def target_anniversary_year := 200

end skyscraper_anniversary_l56_56987


namespace steve_speed_back_l56_56922

theorem steve_speed_back :
  ∀ (v : ℝ), v > 0 → (20 / v + 20 / (2 * v) = 6) → 2 * v = 10 := 
by
  intros v v_pos h
  sorry

end steve_speed_back_l56_56922


namespace problem_l56_56377

open Matrix

-- Define the system of equations as a matrix multiplication equated to zero
def system_matrix (k : ℚ) : Matrix (Fin 4) (Fin 4) ℚ :=
  ![![1, 2*k, 4, -1],
    ![4, k, 2, 1],
    ![3, 5, -3, 2],
    ![2, 3, 1, -4]]

-- The theorem stating the required conclusion
theorem problem (a x y z w : ℚ) (h1 : a ≠ 0) (h2 : x = 3 * a) (h3 : y = a)
  (h4 : z = 5 * a) (h5 : w = 2 * a) (k := 60 / 7) (h : (system_matrix k).mul_vec ![x, y, z, w] = 0) :
  (x * y) / (z * w) = 3 / 10 :=
by
  sorry

end problem_l56_56377


namespace solve_equation_l56_56883

theorem solve_equation (x : ℝ) :
  (2 * x - 1)^2 - 25 = 0 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end solve_equation_l56_56883


namespace lines_parallel_l56_56972

theorem lines_parallel (a : ℝ) 
  (h₁ : (∀ x y : ℝ, ax + (a + 2) * y + 2 = 0)) 
  (h₂ : (∀ x y : ℝ, x + a * y + 1 = 0)) 
  : a = -1 :=
sorry

end lines_parallel_l56_56972


namespace find_D_E_l56_56889

/--
Consider the circle given by \( x^2 + y^2 + D \cdot x + E \cdot y + F = 0 \) that is symmetrical with
respect to the line \( l_1: x - y + 4 = 0 \) and the line \( l_2: x + 3y = 0 \). Prove that the values 
of \( D \) and \( E \) are \( 12 \) and \( -4 \), respectively.
-/
theorem find_D_E (D E F : ℝ) (h1 : -D/2 + E/2 + 4 = 0) (h2 : -D/2 - 3*E/2 = 0) : D = 12 ∧ E = -4 :=
by
  sorry

end find_D_E_l56_56889


namespace jim_saving_amount_l56_56862

theorem jim_saving_amount
    (sara_initial_savings : ℕ)
    (sara_weekly_savings : ℕ)
    (jim_weekly_savings : ℕ)
    (weeks_elapsed : ℕ)
    (sara_total_savings : ℕ := sara_initial_savings + weeks_elapsed * sara_weekly_savings)
    (jim_total_savings : ℕ := weeks_elapsed * jim_weekly_savings)
    (savings_equal: sara_total_savings = jim_total_savings)
    (sara_initial_savings_value : sara_initial_savings = 4100)
    (sara_weekly_savings_value : sara_weekly_savings = 10)
    (weeks_elapsed_value : weeks_elapsed = 820) :
    jim_weekly_savings = 15 := 
by
  sorry

end jim_saving_amount_l56_56862


namespace trailing_zeros_sum_15_factorial_l56_56191

theorem trailing_zeros_sum_15_factorial : 
  let k := 5
  let h := 3
  k + h = 8 := by
  sorry

end trailing_zeros_sum_15_factorial_l56_56191


namespace enthalpy_change_correct_l56_56643

def CC_bond_energy : ℝ := 347
def CO_bond_energy : ℝ := 358
def OH_bond_energy_CH2OH : ℝ := 463
def CO_double_bond_energy_COOH : ℝ := 745
def OH_bond_energy_COOH : ℝ := 467
def OO_double_bond_energy : ℝ := 498
def OH_bond_energy_H2O : ℝ := 467

def total_bond_energy_reactants : ℝ :=
  CC_bond_energy + CO_bond_energy + OH_bond_energy_CH2OH + 1.5 * OO_double_bond_energy

def total_bond_energy_products : ℝ :=
  CO_double_bond_energy_COOH + OH_bond_energy_COOH + OH_bond_energy_H2O

def deltaH : ℝ := total_bond_energy_reactants - total_bond_energy_products

theorem enthalpy_change_correct :
  deltaH = 236 := by
  sorry

end enthalpy_change_correct_l56_56643


namespace percent_of_Q_l56_56062

theorem percent_of_Q (P Q : ℝ) (h : (50 / 100) * P = (20 / 100) * Q) : P = 0.4 * Q :=
sorry

end percent_of_Q_l56_56062
