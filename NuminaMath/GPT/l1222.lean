import Mathlib

namespace max_value_sine_cosine_l1222_122207

/-- If the maximum value of the function f(x) = 4 * sin x + a * cos x is 5, then a = ±3. -/
theorem max_value_sine_cosine (a : ℝ) :
  (∀ x : ℝ, 4 * Real.sin x + a * Real.cos x ≤ 5) →
  (∃ x : ℝ, 4 * Real.sin x + a * Real.cos x = 5) →
  a = 3 ∨ a = -3 :=
by
  sorry

end max_value_sine_cosine_l1222_122207


namespace least_positive_value_l1222_122272

theorem least_positive_value (x y z : ℤ) : ∃ x y z : ℤ, 0 < 72 * x + 54 * y + 36 * z ∧ ∀ (a b c : ℤ), 0 < 72 * a + 54 * b + 36 * c → 72 * x + 54 * y + 36 * z ≤ 72 * a + 54 * b + 36 * c :=
sorry

end least_positive_value_l1222_122272


namespace fraction_simplification_l1222_122248

theorem fraction_simplification
  (a b c x : ℝ)
  (hb : b ≠ 0)
  (hxc : c ≠ 0)
  (h : x = a / b)
  (ha : a ≠ c * b) :
  (a + c * b) / (a - c * b) = (x + c) / (x - c) :=
by
  sorry

end fraction_simplification_l1222_122248


namespace det_scaled_matrices_l1222_122201

variable (a b c d : ℝ)

-- Given condition: determinant of the original matrix
def det_A : ℝ := Matrix.det ![![a, b], ![c, d]]

-- Problem statement: determinants of the scaled matrices
theorem det_scaled_matrices
    (h: det_A a b c d = 3) :
  Matrix.det ![![3 * a, 3 * b], ![3 * c, 3 * d]] = 27 ∧
  Matrix.det ![![4 * a, 2 * b], ![4 * c, 2 * d]] = 24 :=
by
  sorry

end det_scaled_matrices_l1222_122201


namespace train_speed_l1222_122205

theorem train_speed (length_train length_bridge time_crossing speed : ℝ)
  (h1 : length_train = 100)
  (h2 : length_bridge = 300)
  (h3 : time_crossing = 24)
  (h4 : speed = (length_train + length_bridge) / time_crossing) :
  speed = 16.67 := 
sorry

end train_speed_l1222_122205


namespace distance_between_stations_l1222_122254

-- distance calculation definitions
def distance (rate time : ℝ) := rate * time

-- problem conditions as definitions
def rate_slow := 20 -- km/hr
def rate_fast := 25 -- km/hr
def extra_distance := 50 -- km

-- final statement
theorem distance_between_stations :
  ∃ (D : ℝ) (T : ℝ),
    (distance rate_slow T = D) ∧
    (distance rate_fast T = D + extra_distance) ∧
    (D + (D + extra_distance) = 450) :=
by
  sorry

end distance_between_stations_l1222_122254


namespace sums_of_adjacent_cells_l1222_122267

theorem sums_of_adjacent_cells (N : ℕ) (h : N ≥ 2) :
  ∃ (f : ℕ → ℕ → ℝ), (∀ i j, 1 ≤ i ∧ i < N → 1 ≤ j ∧ j < N → 
    (∃ (s : ℕ), 1 ≤ s ∧ s ≤ 2*(N-1)*N ∧ s = f i j + f (i + 1) j) ∧
    (∃ (s : ℕ), 1 ≤ s ∧ s ≤ 2*(N-1)*N ∧ s = f i j + f i (j + 1))) := sorry

end sums_of_adjacent_cells_l1222_122267


namespace solution_set_of_inequality_l1222_122279

theorem solution_set_of_inequality :
  {x : ℝ | 2 ≥ 1 / (x - 1)} = {x : ℝ | x < 1} ∪ {x : ℝ | x ≥ 3 / 2} :=
by
  sorry

end solution_set_of_inequality_l1222_122279


namespace polynomial_has_roots_l1222_122289

-- Define the polynomial
def polynomial (x : ℂ) : ℂ := 7 * x^4 - 48 * x^3 + 93 * x^2 - 48 * x + 7

-- Theorem to prove the existence of roots for the polynomial equation
theorem polynomial_has_roots : ∃ x : ℂ, polynomial x = 0 := by
  sorry

end polynomial_has_roots_l1222_122289


namespace area_of_border_l1222_122216

theorem area_of_border (height_painting width_painting border_width : ℕ)
    (area_painting framed_height framed_width : ℕ)
    (H1 : height_painting = 12)
    (H2 : width_painting = 15)
    (H3 : border_width = 3)
    (H4 : area_painting = height_painting * width_painting)
    (H5 : framed_height = height_painting + 2 * border_width)
    (H6 : framed_width = width_painting + 2 * border_width)
    (area_framed : ℕ)
    (H7 : area_framed = framed_height * framed_width) :
    area_framed - area_painting = 198 := 
sorry

end area_of_border_l1222_122216


namespace count_p_shape_points_l1222_122285

-- Define the problem conditions
def side_length : ℕ := 10
def point_interval : ℕ := 1
def num_sides : ℕ := 3
def correction_corners : ℕ := 2

-- Define the total expected points
def total_expected_points : ℕ := 31

-- Proof statement
theorem count_p_shape_points :
  ((side_length / point_interval + 1) * num_sides - correction_corners) = total_expected_points := by
  sorry

end count_p_shape_points_l1222_122285


namespace geometric_series_first_term_l1222_122242

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by 
  sorry

end geometric_series_first_term_l1222_122242


namespace effective_rate_proof_l1222_122219

noncomputable def nominal_rate : ℝ := 0.08
noncomputable def compounding_periods : ℕ := 2
noncomputable def effective_annual_rate (i : ℝ) (n : ℕ) : ℝ := (1 + i / n) ^ n - 1

theorem effective_rate_proof :
  effective_annual_rate nominal_rate compounding_periods = 0.0816 :=
by
  sorry

end effective_rate_proof_l1222_122219


namespace negation_of_proposition_l1222_122245

variable (f : ℕ+ → ℕ)

theorem negation_of_proposition :
  (¬ ∀ n : ℕ+, f n ≤ n) ↔ (∃ n : ℕ+, f n > n) :=
by sorry

end negation_of_proposition_l1222_122245


namespace compute_difference_of_squares_l1222_122215

theorem compute_difference_of_squares :
  262^2 - 258^2 = 2080 := 
by
  sorry

end compute_difference_of_squares_l1222_122215


namespace book_price_l1222_122271

theorem book_price (B P : ℝ) 
  (h1 : (1 / 3) * B = 36) 
  (h2 : (2 / 3) * B * P = 252) : 
  P = 3.5 :=
by {
  sorry
}

end book_price_l1222_122271


namespace socorro_training_days_l1222_122260

def total_hours := 5
def minutes_per_hour := 60
def total_training_minutes := total_hours * minutes_per_hour

def minutes_multiplication_per_day := 10
def minutes_division_per_day := 20
def daily_training_minutes := minutes_multiplication_per_day + minutes_division_per_day

theorem socorro_training_days:
  total_training_minutes / daily_training_minutes = 10 :=
by
  -- proof omitted
  sorry

end socorro_training_days_l1222_122260


namespace printer_paper_last_days_l1222_122236

def packs : Nat := 2
def sheets_per_pack : Nat := 240
def prints_per_day : Nat := 80
def total_sheets : Nat := packs * sheets_per_pack
def number_of_days : Nat := total_sheets / prints_per_day

theorem printer_paper_last_days :
  number_of_days = 6 :=
by
  sorry

end printer_paper_last_days_l1222_122236


namespace summation_values_l1222_122211

theorem summation_values (x y : ℝ) (h1 : x = y * (3 - y) ^ 2) (h2 : y = x * (3 - x) ^ 2) : 
  x + y = 0 ∨ x + y = 3 ∨ x + y = 4 ∨ x + y = 5 ∨ x + y = 8 :=
sorry

end summation_values_l1222_122211


namespace int_values_satisfying_l1222_122256

theorem int_values_satisfying (x : ℤ) : (∃ k : ℤ, (5 * x + 2) = 17 * k) ↔ (∃ m : ℤ, x = 17 * m + 3) :=
by
  sorry

end int_values_satisfying_l1222_122256


namespace range_of_m_l1222_122283

open Classical

variable {m : ℝ}

def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * x + m ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, (3 - m) > 1 → ((3 - m) ^ x > 0)

theorem range_of_m (hm : (p m ∨ q m) ∧ ¬(p m ∧ q m)) : 1 < m ∧ m < 2 :=
  sorry

end range_of_m_l1222_122283


namespace passing_marks_l1222_122222

theorem passing_marks :
  ∃ P T : ℝ, (0.2 * T = P - 40) ∧ (0.3 * T = P + 20) ∧ P = 160 :=
by
  sorry

end passing_marks_l1222_122222


namespace neither_5_nor_6_nice_1200_l1222_122239

def is_k_nice (N k : ℕ) : Prop := N % k = 1

def count_k_nice_up_to (k n : ℕ) : ℕ :=
(n + (k - 1)) / k

def count_neither_5_nor_6_nice_up_to (n : ℕ) : ℕ :=
  let count_5_nice := count_k_nice_up_to 5 n
  let count_6_nice := count_k_nice_up_to 6 n
  let count_5_and_6_nice := count_k_nice_up_to 30 n
  n - (count_5_nice + count_6_nice - count_5_and_6_nice)

theorem neither_5_nor_6_nice_1200 : count_neither_5_nor_6_nice_up_to 1200 = 800 := 
by
  sorry

end neither_5_nor_6_nice_1200_l1222_122239


namespace avg_salary_of_employees_is_1500_l1222_122238

-- Definitions for conditions
def num_employees : ℕ := 20
def num_people_incl_manager : ℕ := 21
def manager_salary : ℝ := 4650
def salary_increase : ℝ := 150

-- Definition for average salary of employees excluding the manager
def avg_salary_employees (A : ℝ) : Prop :=
    21 * (A + salary_increase) = 20 * A + manager_salary

-- The target proof statement
theorem avg_salary_of_employees_is_1500 :
  ∃ A : ℝ, avg_salary_employees A ∧ A = 1500 := by
  -- Proof goes here
  sorry

end avg_salary_of_employees_is_1500_l1222_122238


namespace second_machine_completion_time_l1222_122208

variable (time_first_machine : ℝ) (rate_first_machine : ℝ) (rate_combined : ℝ)
variable (rate_second_machine: ℝ) (y : ℝ)

def processing_rate_first_machine := rate_first_machine = 100
def processing_rate_combined := rate_combined = 1000 / 3
def processing_rate_second_machine := rate_second_machine = rate_combined - rate_first_machine
def completion_time_second_machine := y = 1000 / rate_second_machine

theorem second_machine_completion_time
  (h1: processing_rate_first_machine rate_first_machine)
  (h2: processing_rate_combined rate_combined)
  (h3: processing_rate_second_machine rate_combined rate_first_machine rate_second_machine)
  (h4: completion_time_second_machine rate_second_machine y) :
  y = 30 / 7 :=
sorry

end second_machine_completion_time_l1222_122208


namespace points_on_line_l1222_122243

theorem points_on_line (b m n : ℝ) (hA : m = -(-5) + b) (hB : n = -(4) + b) :
  m > n :=
by
  sorry

end points_on_line_l1222_122243


namespace combined_mpg_l1222_122200

theorem combined_mpg :
  let mR := 150 -- miles Ray drives
  let mT := 300 -- miles Tom drives
  let mpgR := 50 -- miles per gallon for Ray's car
  let mpgT := 20 -- miles per gallon for Tom's car
  -- Total gasoline used by Ray and Tom
  let gR := mR / mpgR
  let gT := mT / mpgT
  -- Total distance driven
  let total_distance := mR + mT
  -- Total gasoline used
  let total_gasoline := gR + gT
  -- Combined miles per gallon
  let combined_mpg := total_distance / total_gasoline
  combined_mpg = 25 := by
    sorry

end combined_mpg_l1222_122200


namespace triangle_ABC_area_l1222_122224

open Real

-- Define points A, B, and C
structure Point :=
  (x: ℝ)
  (y: ℝ)

def A : Point := ⟨-1, 2⟩
def B : Point := ⟨8, 2⟩
def C : Point := ⟨6, -1⟩

-- Function to calculate the area of a triangle given vertices A, B, and C
noncomputable def triangle_area (A B C : Point) : ℝ := 
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

-- The statement to be proved
theorem triangle_ABC_area : triangle_area A B C = 13.5 :=
by
  sorry

end triangle_ABC_area_l1222_122224


namespace sum_of_possible_radii_l1222_122241

-- Define the geometric and algebraic conditions of the problem
noncomputable def circleTangentSum (r : ℝ) : Prop :=
  let center_C := (r, r)
  let center_other := (3, 3)
  let radius_other := 2
  (∃ r : ℝ, (r > 0) ∧ ((center_C.1 - center_other.1)^2 + (center_C.2 - center_other.2)^2 = (r + radius_other)^2))

-- Define the theorem statement
theorem sum_of_possible_radii : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ circleTangentSum r1 ∧ circleTangentSum r2 ∧ r1 + r2 = 16 :=
sorry

end sum_of_possible_radii_l1222_122241


namespace R_depends_on_a_d_m_l1222_122240

theorem R_depends_on_a_d_m (a d m : ℝ) :
    let s1 := (m / 2) * (2 * a + (m - 1) * d)
    let s2 := m * (2 * a + (2 * m - 1) * d)
    let s3 := 2 * m * (2 * a + (4 * m - 1) * d)
    let R := s3 - 2 * s2 + s1
    R = m * (a + 12 * m * d - (d / 2)) := by
  sorry

end R_depends_on_a_d_m_l1222_122240


namespace bug_crawl_distance_l1222_122218

-- Define the conditions
def initial_position : ℤ := -2
def first_move : ℤ := -6
def second_move : ℤ := 5

-- Define the absolute difference function (distance on a number line)
def abs_diff (a b : ℤ) : ℤ :=
  abs (b - a)

-- Define the total distance crawled function
def total_distance (p1 p2 p3 : ℤ) : ℤ :=
  abs_diff p1 p2 + abs_diff p2 p3

-- Prove that total distance starting at -2, moving to -6, and then to 5 is 15 units
theorem bug_crawl_distance : total_distance initial_position first_move second_move = 15 := by
  sorry

end bug_crawl_distance_l1222_122218


namespace solve_for_x_l1222_122250

theorem solve_for_x (x: ℚ) (h: (3/5 - 1/4) = 4/x) : x = 80/7 :=
by
  sorry

end solve_for_x_l1222_122250


namespace sphere_volume_l1222_122217

theorem sphere_volume (S : ℝ) (hS : S = 4 * π) : ∃ V : ℝ, V = (4 / 3) * π := 
by
  sorry

end sphere_volume_l1222_122217


namespace blue_notebook_cost_l1222_122232

theorem blue_notebook_cost
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (red_notebook_cost : ℕ)
  (green_notebooks : ℕ)
  (green_notebook_cost : ℕ)
  (blue_notebook_cost : ℕ)
  (h₀ : total_spent = 37)
  (h₁ : total_notebooks = 12)
  (h₂ : red_notebooks = 3)
  (h₃ : red_notebook_cost = 4)
  (h₄ : green_notebooks = 2)
  (h₅ : green_notebook_cost = 2)
  (h₆ : total_spent = red_notebooks * red_notebook_cost + green_notebooks * green_notebook_cost + blue_notebook_cost * (total_notebooks - red_notebooks - green_notebooks)) :
  blue_notebook_cost = 3 := by
  sorry

end blue_notebook_cost_l1222_122232


namespace find_x_value_l1222_122213

noncomputable def check_x (x : ℝ) : Prop :=
  (0 < x) ∧ (Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (10 * x) = 10)

theorem find_x_value (x : ℝ) (h : check_x x) : x = 1 / 6 :=
by 
  sorry

end find_x_value_l1222_122213


namespace wrapping_paper_area_l1222_122259

theorem wrapping_paper_area (l w h : ℝ) (hlw : l > w) (hwh : w > h) (hl : l = 2 * w) : 
    (∃ a : ℝ, a = 5 * w^2 + h^2) :=
by 
  sorry

end wrapping_paper_area_l1222_122259


namespace mul_inv_800_mod_7801_l1222_122276

theorem mul_inv_800_mod_7801 :
  ∃ x : ℕ, 0 ≤ x ∧ x < 7801 ∧ (800 * x) % 7801 = 1 := by
  use 3125
  dsimp
  norm_num1
  sorry

end mul_inv_800_mod_7801_l1222_122276


namespace value_of_x_l1222_122275

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 :=
by
  sorry

end value_of_x_l1222_122275


namespace rectangle_area_l1222_122274

theorem rectangle_area (d : ℝ) (w : ℝ) (h : (3 * w)^2 + w^2 = d^2) : 
  3 * w^2 = d^2 / 10 :=
by
  sorry

end rectangle_area_l1222_122274


namespace total_revenue_correct_l1222_122226

-- Definitions and conditions
def number_of_fair_tickets : ℕ := 60
def price_per_fair_ticket : ℕ := 15
def price_per_baseball_ticket : ℕ := 10
def number_of_baseball_tickets : ℕ := number_of_fair_tickets / 3

-- Calculate revenues
def revenue_from_fair_tickets : ℕ := number_of_fair_tickets * price_per_fair_ticket
def revenue_from_baseball_tickets : ℕ := number_of_baseball_tickets * price_per_baseball_ticket
def total_revenue : ℕ := revenue_from_fair_tickets + revenue_from_baseball_tickets

-- Proof statement
theorem total_revenue_correct : total_revenue = 1100 := by
  sorry

end total_revenue_correct_l1222_122226


namespace pythagorean_theorem_l1222_122249

theorem pythagorean_theorem (a b c : ℝ) : (a^2 + b^2 = c^2) ↔ (a^2 + b^2 = c^2) :=
by sorry

end pythagorean_theorem_l1222_122249


namespace g_pi_over_4_eq_neg_sqrt2_over_4_l1222_122237

noncomputable def g (x : Real) : Real := 
  Real.sqrt (5 * (Real.sin x)^4 + 4 * (Real.cos x)^2) - 
  Real.sqrt (6 * (Real.cos x)^4 + 4 * (Real.sin x)^2)

theorem g_pi_over_4_eq_neg_sqrt2_over_4 :
  g (Real.pi / 4) = - (Real.sqrt 2) / 4 := 
sorry

end g_pi_over_4_eq_neg_sqrt2_over_4_l1222_122237


namespace smallest_positive_period_symmetry_axis_not_even_function_decreasing_interval_l1222_122277

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (4 * Real.pi / 3))

theorem smallest_positive_period (T : ℝ) : T = Real.pi ↔ (∀ x : ℝ, f (x + T) = f x) := by
  sorry

theorem symmetry_axis (x : ℝ) : x = (7 * Real.pi / 12) ↔ (∀ y : ℝ, f (2 * x - y) = f y) := by
  sorry

theorem not_even_function : ¬ (∀ x : ℝ, f (x + (Real.pi / 3)) = f (-x - (Real.pi / 3))) := by
  sorry

theorem decreasing_interval (k : ℤ) (x : ℝ) : (k * Real.pi - (5 * Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 12)) ↔ (∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2) := by
  sorry

end smallest_positive_period_symmetry_axis_not_even_function_decreasing_interval_l1222_122277


namespace factorial_subtraction_l1222_122292

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end factorial_subtraction_l1222_122292


namespace albums_created_l1222_122268

def phone_pics : ℕ := 2
def camera_pics : ℕ := 4
def pics_per_album : ℕ := 2
def total_pics : ℕ := phone_pics + camera_pics

theorem albums_created : total_pics / pics_per_album = 3 := by
  sorry

end albums_created_l1222_122268


namespace solve_z_for_complex_eq_l1222_122298

theorem solve_z_for_complex_eq (i : ℂ) (h : i^2 = -1) : ∀ (z : ℂ), 3 - 2 * i * z = -4 + 5 * i * z → z = -i :=
by
  intro z
  intro eqn
  -- The proof would go here
  sorry

end solve_z_for_complex_eq_l1222_122298


namespace set_intersection_l1222_122282

open Set

variable (x : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := { x | |x - 1| > 2 }
def B : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }

theorem set_intersection (x : ℝ) : x ∈ (U \ A) ∩ B ↔ 2 < x ∧ x ≤ 3 := sorry

end set_intersection_l1222_122282


namespace solution_set_f_inequality_l1222_122235

noncomputable def f (x : ℝ) : ℝ := 
if x > 0 then 1 - 2^(-x)
else if x < 0 then 2^x - 1
else 0

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem solution_set_f_inequality : 
  is_odd_function f →
  {x | f x < -1/2} = {x | x < -1} := 
by
  sorry

end solution_set_f_inequality_l1222_122235


namespace inequality_example_l1222_122291

theorem inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 = 3) : 
    1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + a * c) ≥ 3 / 2 :=
by
  sorry

end inequality_example_l1222_122291


namespace range_of_m_l1222_122227

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → x^(m-1) > y^(m-1)) → m < 1 :=
by
  sorry

end range_of_m_l1222_122227


namespace factorial_power_of_two_iff_power_of_two_l1222_122288

-- Assuming n is a positive integer
variable {n : ℕ} (h : n > 0)

theorem factorial_power_of_two_iff_power_of_two :
  (∃ k : ℕ, n = 2^k ) ↔ ∃ m : ℕ, 2^(n-1) ∣ n! :=
by {
  sorry
}

end factorial_power_of_two_iff_power_of_two_l1222_122288


namespace original_apples_l1222_122299

-- Define the conditions using the given data
def sells_fraction : ℝ := 0.40 -- Fraction of apples sold
def remaining_apples : ℝ := 420 -- Apples remaining after selling

-- Theorem statement for proving the original number of apples given the conditions
theorem original_apples (x : ℝ) (sells_fraction : ℝ := 0.40) (remaining_apples : ℝ := 420) : 
  420 / (1 - sells_fraction) = x :=
sorry

end original_apples_l1222_122299


namespace joan_first_payment_l1222_122269

theorem joan_first_payment (P : ℝ) 
  (total_amount : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (h_total : total_amount = 109300)
  (h_r : r = 3)
  (h_n : n = 7)
  (h_sum : total_amount = P * (1 - r^n) / (1 - r)) : 
  P = 100 :=
by
  -- proof goes here
  sorry

end joan_first_payment_l1222_122269


namespace probability_student_less_than_25_l1222_122294

def total_students : ℕ := 100

-- Percentage conditions translated to proportions
def proportion_male : ℚ := 0.48
def proportion_female : ℚ := 0.52

def proportion_male_25_or_older : ℚ := 0.50
def proportion_female_25_or_older : ℚ := 0.20

-- Definition of probability that a randomly selected student is less than 25 years old.
def probability_less_than_25 : ℚ :=
  (proportion_male * (1 - proportion_male_25_or_older)) +
  (proportion_female * (1 - proportion_female_25_or_older))

theorem probability_student_less_than_25 :
  probability_less_than_25 = 0.656 := by
  sorry

end probability_student_less_than_25_l1222_122294


namespace distinct_non_zero_reals_square_rational_l1222_122293

theorem distinct_non_zero_reals_square_rational
  {a : Fin 10 → ℝ}
  (distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (non_zero : ∀ i, a i ≠ 0)
  (rational_condition : ∀ i j, ∃ (q : ℚ), a i + a j = q ∨ a i * a j = q) :
  ∀ i, ∃ (q : ℚ), (a i)^2 = q :=
by
  sorry

end distinct_non_zero_reals_square_rational_l1222_122293


namespace proof_problem_l1222_122270

variable {a b c : ℝ}
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variable (h4 : (a+1) * (b+1) * (c+1) = 8)

theorem proof_problem :
  a + b + c ≥ 3 ∧ abc ≤ 1 :=
by
  sorry

end proof_problem_l1222_122270


namespace digit_sum_of_nines_l1222_122261

theorem digit_sum_of_nines (k : ℕ) (n : ℕ) (h : n = 9 * (10^k - 1) / 9):
  (8 + 9 * (k - 1) + 1 = 500) → k = 55 := 
by 
  sorry

end digit_sum_of_nines_l1222_122261


namespace simplify_sqrt_expression_correct_l1222_122280

noncomputable def simplify_sqrt_expression (m : ℝ) (h_triangle : (2 < m + 5) ∧ (m < 2 + 5) ∧ (5 < 2 + m)) : ℝ :=
  (Real.sqrt (9 - 6 * m + m^2)) - (Real.sqrt (m^2 - 14 * m + 49))

theorem simplify_sqrt_expression_correct (m : ℝ) (h_triangle : (2 < m + 5) ∧ (m < 2 + 5) ∧ (5 < 2 + m)) :
  simplify_sqrt_expression m h_triangle = 2 * m - 10 :=
sorry

end simplify_sqrt_expression_correct_l1222_122280


namespace problem_equivalent_l1222_122265

theorem problem_equivalent :
  ∀ m n : ℤ, |m - n| = n - m ∧ |m| = 4 ∧ |n| = 3 → m + n = -1 ∨ m + n = -7 :=
by
  intros m n h
  have h1 : |m - n| = n - m := h.1
  have h2 : |m| = 4 := h.2.1
  have h3 : |n| = 3 := h.2.2
  sorry

end problem_equivalent_l1222_122265


namespace percentage_of_number_l1222_122204

theorem percentage_of_number (N P : ℝ) (h1 : 0.60 * N = 240) (h2 : (P / 100) * N = 160) : P = 40 :=
by
  sorry

end percentage_of_number_l1222_122204


namespace initial_ratio_proof_l1222_122221

variable (p q : ℕ) -- Define p and q as non-negative integers

-- Condition: The initial total volume of the mixture is 30 liters
def initial_volume (p q : ℕ) : Prop := p + q = 30

-- Condition: Adding 12 liters of q changes the ratio to 3:4
def new_ratio (p q : ℕ) : Prop := p * 4 = (q + 12) * 3

-- The final goal: prove the initial ratio is 3:2
def initial_ratio (p q : ℕ) : Prop := p * 2 = q * 3

-- The main proof problem statement
theorem initial_ratio_proof (p q : ℕ) 
  (h1 : initial_volume p q) 
  (h2 : new_ratio p q) : initial_ratio p q :=
  sorry

end initial_ratio_proof_l1222_122221


namespace new_person_weight_l1222_122263

-- Define the conditions of the problem
variables (avg_weight : ℝ) (weight_replaced_person : ℝ) (num_persons : ℕ)
variable (weight_increase : ℝ)

-- Given conditions
def condition (avg_weight weight_replaced_person : ℝ) (num_persons : ℕ) (weight_increase : ℝ) : Prop :=
  num_persons = 10 ∧ weight_replaced_person = 60 ∧ weight_increase = 5

-- The proof problem
theorem new_person_weight (avg_weight weight_replaced_person : ℝ) (num_persons : ℕ)
  (weight_increase : ℝ) (h : condition avg_weight weight_replaced_person num_persons weight_increase) :
  weight_replaced_person + num_persons * weight_increase = 110 :=
sorry

end new_person_weight_l1222_122263


namespace find_p_q_l1222_122287

theorem find_p_q (p q : ℚ) : 
    (∀ x, x^5 - x^4 + x^3 - p*x^2 + q*x + 9 = 0 → (x = -3 ∨ x = 2)) →
    (p, q) = (-19.5, -55.5) :=
by {
  sorry
}

end find_p_q_l1222_122287


namespace cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8_l1222_122290

theorem cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8
  (α : ℝ) (h : Real.cos α = 2 * Real.cos (α + Real.pi / 4)) :
  Real.tan (α + Real.pi / 8) = 3 * (Real.sqrt 2 + 1) := 
sorry

end cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8_l1222_122290


namespace total_sum_money_l1222_122284

theorem total_sum_money (a b c : ℝ) (h1 : b = 0.65 * a) (h2 : c = 0.40 * a) (h3 : c = 64) :
  a + b + c = 328 :=
by
  sorry

end total_sum_money_l1222_122284


namespace minimum_squares_and_perimeter_l1222_122230

theorem minimum_squares_and_perimeter 
  (length width : ℕ) 
  (h_length : length = 90) 
  (h_width : width = 42) 
  (h_gcd : Nat.gcd length width = 6) 
  : 
  ((length / Nat.gcd length width) * (width / Nat.gcd length width) = 105) ∧ 
  (105 * (4 * Nat.gcd length width) = 2520) := 
by 
  sorry

end minimum_squares_and_perimeter_l1222_122230


namespace parabola_trajectory_l1222_122246

theorem parabola_trajectory :
  ∀ P : ℝ × ℝ, (dist P (0, -1) + 1 = dist P (0, 3)) ↔ (P.1 ^ 2 = -8 * P.2) := by
  sorry

end parabola_trajectory_l1222_122246


namespace kids_in_group_l1222_122202

theorem kids_in_group :
  ∃ (K : ℕ), (∃ (A : ℕ), A + K = 9 ∧ 2 * A = 14) ∧ K = 2 :=
by
  sorry

end kids_in_group_l1222_122202


namespace store_total_profit_l1222_122231

theorem store_total_profit
  (purchase_price : ℕ)
  (selling_price_total : ℕ)
  (max_selling_price : ℕ)
  (profit : ℕ)
  (N : ℕ)
  (selling_price_per_card : ℕ)
  (h1 : purchase_price = 21)
  (h2 : selling_price_total = 1457)
  (h3 : max_selling_price = 2 * purchase_price)
  (h4 : selling_price_per_card * N = selling_price_total)
  (h5 : selling_price_per_card ≤ max_selling_price)
  (h_profit : profit = (selling_price_per_card - purchase_price) * N)
  : profit = 470 :=
sorry

end store_total_profit_l1222_122231


namespace exponential_growth_equation_l1222_122296

-- Define the initial and final greening areas and the years in consideration.
def initial_area : ℝ := 1000
def final_area : ℝ := 1440
def years : ℝ := 2

-- Define the average annual growth rate.
variable (x : ℝ)

-- State the theorem about the exponential growth equation.
theorem exponential_growth_equation :
  initial_area * (1 + x) ^ years = final_area :=
sorry

end exponential_growth_equation_l1222_122296


namespace find_angle_A_and_triangle_perimeter_l1222_122203

-- Declare the main theorem using the provided conditions and the desired results
theorem find_angle_A_and_triangle_perimeter
  (a b c : ℝ) (A B : ℝ)
  (h1 : 0 < A ∧ A < Real.pi)
  (h2 : (Real.sqrt 3) * b * c * (Real.cos A) = a * (Real.sin B))
  (h3 : a = Real.sqrt 2)
  (h4 : (c / a) = (Real.sin A / Real.sin B)) :
  (A = Real.pi / 3) ∧ (a + b + c = 3 * Real.sqrt 2) :=
  sorry -- Proof is left as an exercise

end find_angle_A_and_triangle_perimeter_l1222_122203


namespace largest_number_among_l1222_122253

theorem largest_number_among (π: ℝ) (sqrt_2: ℝ) (neg_2: ℝ) (three: ℝ)
  (h1: 3.14 ≤ π)
  (h2: 1 < sqrt_2 ∧ sqrt_2 < 2)
  (h3: neg_2 < 1)
  (h4: 3 < π) :
  (neg_2 < sqrt_2) ∧ (sqrt_2 < 3) ∧ (3 < π) :=
by {
  sorry
}

end largest_number_among_l1222_122253


namespace number_of_hens_is_50_l1222_122258

def number_goats : ℕ := 45
def number_camels : ℕ := 8
def number_keepers : ℕ := 15
def extra_feet : ℕ := 224

def total_heads (number_hens number_goats number_camels number_keepers : ℕ) : ℕ :=
  number_hens + number_goats + number_camels + number_keepers

def total_feet (number_hens number_goats number_camels number_keepers : ℕ) : ℕ :=
  2 * number_hens + 4 * number_goats + 4 * number_camels + 2 * number_keepers

theorem number_of_hens_is_50 (H : ℕ) :
  total_feet H number_goats number_camels number_keepers = (total_heads H number_goats number_camels number_keepers) + extra_feet → H = 50 :=
sorry

end number_of_hens_is_50_l1222_122258


namespace cost_per_game_l1222_122297

theorem cost_per_game 
  (x : ℝ)
  (shoe_rent : ℝ := 0.50)
  (total_money : ℝ := 12.80)
  (games : ℕ := 7)
  (h1 : total_money - shoe_rent = 12.30)
  (h2 : 7 * x = 12.30) :
  x = 1.76 := 
sorry

end cost_per_game_l1222_122297


namespace find_g5_l1222_122233

variable (g : ℝ → ℝ)

-- Formal definition of the condition for the function g in the problem statement.
def functional_eq_condition :=
  ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2

-- The main statement to prove g(5) = 1 under the given condition.
theorem find_g5 (h : functional_eq_condition g) :
  g 5 = 1 :=
sorry

end find_g5_l1222_122233


namespace smallest_N_divisibility_l1222_122255

theorem smallest_N_divisibility :
  ∃ N : ℕ, 
    (N + 2) % 2 = 0 ∧
    (N + 3) % 3 = 0 ∧
    (N + 4) % 4 = 0 ∧
    (N + 5) % 5 = 0 ∧
    (N + 6) % 6 = 0 ∧
    (N + 7) % 7 = 0 ∧
    (N + 8) % 8 = 0 ∧
    (N + 9) % 9 = 0 ∧
    (N + 10) % 10 = 0 ∧
    N = 2520 := 
sorry

end smallest_N_divisibility_l1222_122255


namespace bus_speed_l1222_122273

theorem bus_speed (S : ℝ) (h1 : 36 = S * (2 / 3)) : S = 54 :=
by
sorry

end bus_speed_l1222_122273


namespace solve_inequalities_l1222_122262

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l1222_122262


namespace find_q_l1222_122281

def Q (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_q (p q r : ℝ) (h1 : -p = 2 * (-r)) (h2 : -p = 1 + p + q + r) (hy_intercept : r = 5) : q = -24 :=
by
  sorry

end find_q_l1222_122281


namespace Q_evaluation_at_2_l1222_122209

noncomputable def Q : Polynomial ℚ := 
  (Polynomial.X^2 + Polynomial.C 4)^2

theorem Q_evaluation_at_2 : 
  Q.eval 2 = 64 :=
by 
  -- We'll skip the proof as per the instructions.
  sorry

end Q_evaluation_at_2_l1222_122209


namespace c_in_terms_of_t_l1222_122251

theorem c_in_terms_of_t (t a b c : ℝ) (h_t_ne_zero : t ≠ 0)
    (h1 : t^3 + a * t = 0)
    (h2 : b * t^2 + c = 0)
    (h3 : 3 * t^2 + a = 2 * b * t) :
    c = -t^3 :=
by
sorry

end c_in_terms_of_t_l1222_122251


namespace divisible_by_72_l1222_122220

theorem divisible_by_72 (a b : ℕ) (h1 : 0 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10) :
  (b = 2 ∧ a = 3) → (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) % 72 = 0 :=
by
  sorry

end divisible_by_72_l1222_122220


namespace roots_of_unity_sum_l1222_122206

theorem roots_of_unity_sum (x y z : ℂ) (n m p : ℕ)
  (hx : x^n = 1) (hy : y^m = 1) (hz : z^p = 1) :
  (∃ k : ℕ, (x + y + z)^k = 1) ↔ (x + y = 0 ∨ y + z = 0 ∨ z + x = 0) :=
sorry

end roots_of_unity_sum_l1222_122206


namespace roots_poly_eval_l1222_122286

theorem roots_poly_eval : ∀ (c d : ℝ), (c + d = 6 ∧ c * d = 8) → c^4 + c^3 * d + d^3 * c + d^4 = 432 :=
by
  intros c d h
  sorry

end roots_poly_eval_l1222_122286


namespace system_of_equations_implies_quadratic_l1222_122214

theorem system_of_equations_implies_quadratic (x y : ℝ) :
  (3 * x^2 + 9 * x + 4 * y + 2 = 0) ∧ (3 * x + y + 4 = 0) → (y^2 + 11 * y - 14 = 0) := by
  sorry

end system_of_equations_implies_quadratic_l1222_122214


namespace cos_triple_angle_l1222_122212

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = -1 / 3) : Real.cos (3 * θ) = 23 / 27 :=
by
  sorry

end cos_triple_angle_l1222_122212


namespace smallest_and_largest_group_sizes_l1222_122247

theorem smallest_and_largest_group_sizes (S T : Finset ℕ) (hS : S.card + T.card = 20)
  (h_union: (S ∪ T) = (Finset.range 21) \ {0}) (h_inter: S ∩ T = ∅)
  (sum_S : S.sum id = 210 - T.sum id) (prod_T : T.prod id = 210 - S.sum id) :
  T.card = 3 ∨ T.card = 5 := 
sorry

end smallest_and_largest_group_sizes_l1222_122247


namespace magnolia_trees_below_threshold_l1222_122264

-- Define the initial number of trees and the function describing the decrease
def initial_tree_count (N₀ : ℕ) (t : ℕ) : ℝ := N₀ * (0.8 ^ t)

-- Define the year when the number of trees is less than 25% of initial trees
theorem magnolia_trees_below_threshold (N₀ : ℕ) : (t : ℕ) -> initial_tree_count N₀ t < 0.25 * N₀ -> t > 14 := 
-- Provide the required statement but omit the actual proof with "sorry"
by sorry

end magnolia_trees_below_threshold_l1222_122264


namespace no_infinite_positive_integer_sequence_l1222_122244

theorem no_infinite_positive_integer_sequence (a : ℕ → ℕ) :
  ¬(∀ n, a (n - 1) ^ 2 ≥ 2 * a n * a (n + 2)) :=
sorry

end no_infinite_positive_integer_sequence_l1222_122244


namespace tan_3theta_l1222_122210

-- Let θ be an angle such that tan θ = 3.
variable (θ : ℝ)
noncomputable def tan_theta : ℝ := 3

-- Claim: tan(3 * θ) = 9/13
theorem tan_3theta :
  Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_3theta_l1222_122210


namespace final_segment_position_correct_l1222_122295

def initial_segment : ℝ × ℝ := (1, 6)
def rotate_180_about (p : ℝ) (x : ℝ) : ℝ := p - (x - p)
def first_rotation_segment : ℝ × ℝ := (rotate_180_about 2 6, rotate_180_about 2 1)
def second_rotation_segment : ℝ × ℝ := (rotate_180_about 1 3, rotate_180_about 1 (-2))

theorem final_segment_position_correct :
  second_rotation_segment = (-1, 4) :=
by
  -- This is a placeholder for the actual proof.
  sorry

end final_segment_position_correct_l1222_122295


namespace problem_statement_l1222_122257

theorem problem_statement
  (g : ℝ → ℝ)
  (p q r s : ℝ)
  (h_roots : ∃ n1 n2 n3 n4 : ℕ, 
                ∀ x, g x = (x + 2 * n1) * (x + 2 * n2) * (x + 2 * n3) * (x + 2 * n4))
  (h_pqrs : p + q + r + s = 2552)
  (h_g : ∀ x, g x = x^4 + p * x^3 + q * x^2 + r * x + s) :
  s = 3072 :=
by
  sorry

end problem_statement_l1222_122257


namespace sectors_not_equal_l1222_122223

theorem sectors_not_equal (a1 a2 a3 a4 a5 a6 : ℕ) :
  ¬(∃ k : ℕ, (∀ n : ℕ, n = k) ↔
    ∃ m, (a1 + m) = k ∧ (a2 + m) = k ∧ (a3 + m) = k ∧ 
         (a4 + m) = k ∧ (a5 + m) = k ∧ (a6 + m) = k) :=
sorry

end sectors_not_equal_l1222_122223


namespace line_exists_symmetric_diagonals_l1222_122234

-- Define the initial conditions
def Circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 6 * x = 0
def Line_l1 (x y : ℝ) : Prop := y = 2 * x + 1

-- Define the symmetric circle C about the line l1
def Symmetric_Circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the origion and intersection points
def Point_O : (ℝ × ℝ) := (0, 0)
def Point_Intersection (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop := ∃ x_A y_A x_B y_B : ℝ,
  l x_A = y_A ∧ l x_B = y_B ∧ Symmetric_Circle x_A y_A ∧ Symmetric_Circle x_B y_B

-- Define diagonal equality condition
def Diagonals_Equal (O A S B : ℝ × ℝ) : Prop := 
  let (xO, yO) := O
  let (xA, yA) := A
  let (xS, yS) := S
  let (xB, yB) := B
  (xA - xO)^2 + (yA - yO)^2 = (xB - xS)^2 + (yB - yS)^2

-- Prove existence of line where diagonals are equal and find the equation
theorem line_exists_symmetric_diagonals :
  ∃ l : ℝ → ℝ, (l (-1) = 0) ∧
    (∃ (A B S : ℝ × ℝ), Point_Intersection l A B ∧ Diagonals_Equal Point_O A S B) ∧
    (∀ x : ℝ, l x = x + 1) :=
by
  sorry

end line_exists_symmetric_diagonals_l1222_122234


namespace circle_equation_line_equation_l1222_122229

noncomputable def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 * x + 6 * y = 0

noncomputable def point_O : ℝ × ℝ := (0, 0)
noncomputable def point_A : ℝ × ℝ := (1, 1)
noncomputable def point_B : ℝ × ℝ := (4, 2)

theorem circle_equation :
  circle_C point_O.1 point_O.2 ∧
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 :=
by sorry

noncomputable def line_l_case1 (x : ℝ) : Prop :=
  x = 3 / 2

noncomputable def line_l_case2 (x y : ℝ) : Prop :=
  8 * x + 6 * y - 39 = 0

noncomputable def center_C : ℝ × ℝ := (4, -3)
noncomputable def radius_C : ℝ := 5

noncomputable def point_through_l : ℝ × ℝ := (3 / 2, 9 / 2)

theorem line_equation : 
(∀ (M N : ℝ × ℝ), circle_C M.1 M.2 ∧ circle_C N.1 N.2 → ∃ C_slave : Prop, 
(C_slave → 
((line_l_case1 (point_through_l.1)) ∨ 
(line_l_case2 point_through_l.1 point_through_l.2)))) :=
by sorry

end circle_equation_line_equation_l1222_122229


namespace seventh_observation_is_eight_l1222_122225

theorem seventh_observation_is_eight
  (s₆ : ℕ)
  (a₆ : ℕ)
  (s₇ : ℕ)
  (a₇ : ℕ)
  (h₁ : s₆ = 6 * a₆)
  (h₂ : a₆ = 15)
  (h₃ : s₇ = 7 * a₇)
  (h₄ : a₇ = 14) :
  s₇ - s₆ = 8 :=
by
  -- Place proof here
  sorry

end seventh_observation_is_eight_l1222_122225


namespace price_difference_is_300_cents_l1222_122228

noncomputable def list_price : ℝ := 59.99
noncomputable def tech_bargains_price : ℝ := list_price - 15
noncomputable def digital_deal_price : ℝ := 0.7 * list_price
noncomputable def price_difference : ℝ := tech_bargains_price - digital_deal_price
noncomputable def price_difference_in_cents : ℝ := price_difference * 100

theorem price_difference_is_300_cents :
  price_difference_in_cents = 300 := by
  sorry

end price_difference_is_300_cents_l1222_122228


namespace correct_calculated_value_l1222_122278

theorem correct_calculated_value (x : ℤ) 
  (h : x / 16 = 8 ∧ x % 16 = 4) : (x * 16 + 8 = 2120) := by
  sorry

end correct_calculated_value_l1222_122278


namespace change_given_l1222_122252

-- Define the given conditions
def oranges_cost := 40
def apples_cost := 50
def mangoes_cost := 60
def initial_amount := 300

-- Calculate total cost of fruits
def total_fruits_cost := oranges_cost + apples_cost + mangoes_cost

-- Define the given change
def given_change := initial_amount - total_fruits_cost

-- Prove that the given change is equal to 150
theorem change_given (h_oranges : oranges_cost = 40)
                     (h_apples : apples_cost = 50)
                     (h_mangoes : mangoes_cost = 60)
                     (h_initial : initial_amount = 300) :
  given_change = 150 :=
by
  -- Proof is omitted, indicated by sorry
  sorry

end change_given_l1222_122252


namespace intersection_A_B_l1222_122266

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x | x - 2 < 0}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_A_B_l1222_122266
