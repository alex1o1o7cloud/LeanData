import Mathlib

namespace NUMINAMATH_GPT_jason_egg_consumption_l1288_128844

-- Definition for the number of eggs Jason consumes per day
def eggs_per_day : ℕ := 3

-- Definition for the number of days in a week
def days_in_week : ℕ := 7

-- Definition for the number of weeks we are considering
def weeks : ℕ := 2

-- The statement we want to prove, which combines all the conditions and provides the final answer
theorem jason_egg_consumption : weeks * days_in_week * eggs_per_day = 42 := by
sorry

end NUMINAMATH_GPT_jason_egg_consumption_l1288_128844


namespace NUMINAMATH_GPT_arithmetic_sequence_99th_term_l1288_128800

-- Define the problem with conditions and question
theorem arithmetic_sequence_99th_term (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : S 9 = 27) (h2 : a 10 = 8) :
  a 99 = 97 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_99th_term_l1288_128800


namespace NUMINAMATH_GPT_no_solution_lines_parallel_l1288_128897

theorem no_solution_lines_parallel (m : ℝ) :
  (∀ t s : ℝ, (1 + 5 * t = 4 - 2 * s) ∧ (-3 + 2 * t = 1 + m * s) → false) ↔ m = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_lines_parallel_l1288_128897


namespace NUMINAMATH_GPT_sales_tax_rate_l1288_128833

-- Given conditions
def cost_of_video_game : ℕ := 50
def weekly_allowance : ℕ := 10
def weekly_savings : ℕ := weekly_allowance / 2
def weeks_to_save : ℕ := 11
def total_savings : ℕ := weeks_to_save * weekly_savings

-- Proof problem statement
theorem sales_tax_rate : 
  total_savings - cost_of_video_game = (cost_of_video_game * 10) / 100 := by
  sorry

end NUMINAMATH_GPT_sales_tax_rate_l1288_128833


namespace NUMINAMATH_GPT_value_of_b_minus_a_l1288_128843

theorem value_of_b_minus_a (a b : ℕ) (h1 : a * b = 2 * (a + b) + 1) (h2 : b = 7) : b - a = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_minus_a_l1288_128843


namespace NUMINAMATH_GPT_part1_part2_l1288_128863

theorem part1 (m : ℝ) (h_m_not_zero : m ≠ 0) : m ≤ 4 / 3 :=
by
  -- The proof would go here, but we are inserting sorry to skip it.
  sorry

theorem part2 (m : ℕ) (h_m_range : m = 1) :
  ∃ x1 x2 : ℝ, (m * x1^2 - 4 * x1 + 3 = 0) ∧ (m * x2^2 - 4 * x2 + 3 = 0) ∧ x1 = 1 ∧ x2 = 3 :=
by
  -- The proof would go here, but we are inserting sorry to skip it.
  sorry

end NUMINAMATH_GPT_part1_part2_l1288_128863


namespace NUMINAMATH_GPT_find_notebook_price_l1288_128884

noncomputable def notebook_and_pencil_prices : Prop :=
  ∃ (x y : ℝ),
    5 * x + 4 * y = 16.5 ∧
    2 * x + 2 * y = 7 ∧
    x = 2.5

theorem find_notebook_price : notebook_and_pencil_prices :=
  sorry

end NUMINAMATH_GPT_find_notebook_price_l1288_128884


namespace NUMINAMATH_GPT_max_ab_correct_l1288_128861

noncomputable def max_ab (k : ℝ) (a b: ℝ) : ℝ :=
if k = -3 then 9 else sorry

theorem max_ab_correct (k : ℝ) (a b: ℝ)
  (h1 : (-3 ≤ k ∧ k ≤ 1))
  (h2 : a + b = 2 * k)
  (h3 : a^2 + b^2 = k^2 - 2 * k + 3) :
  max_ab k a b = 9 :=
sorry

end NUMINAMATH_GPT_max_ab_correct_l1288_128861


namespace NUMINAMATH_GPT_vertical_strips_count_l1288_128814

/- Define the conditions -/

variables {a b x y : ℕ}

-- The outer rectangle has a perimeter of 50 cells
axiom outer_perimeter : 2 * a + 2 * b = 50

-- The inner hole has a perimeter of 32 cells
axiom inner_perimeter : 2 * x + 2 * y = 32

-- Cutting along all horizontal lines produces 20 strips
axiom horizontal_cuts : a + x = 20

-- We want to prove that cutting along all vertical grid lines produces 21 strips
theorem vertical_strips_count : b + y = 21 :=
by
  sorry

end NUMINAMATH_GPT_vertical_strips_count_l1288_128814


namespace NUMINAMATH_GPT_prime_angle_triangle_l1288_128857

theorem prime_angle_triangle (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (h_sum : a + b + c = 180) : a = 2 ∨ b = 2 ∨ c = 2 :=
sorry

end NUMINAMATH_GPT_prime_angle_triangle_l1288_128857


namespace NUMINAMATH_GPT_smallest_equal_cost_l1288_128825

def decimal_cost (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_cost (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

theorem smallest_equal_cost :
  ∃ n : ℕ, n < 200 ∧ decimal_cost n = binary_cost n ∧ (∀ m : ℕ, m < 200 ∧ decimal_cost m = binary_cost m → m ≥ n) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_smallest_equal_cost_l1288_128825


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_iff_l1288_128873

theorem quadratic_distinct_real_roots_iff (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (∀ (z : ℝ), z^2 - 2 * (m - 2) * z + m^2 = (z - x) * (z - y))) ↔ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_iff_l1288_128873


namespace NUMINAMATH_GPT_parabola_solution_l1288_128882

noncomputable def parabola_coefficients (a b c : ℝ) : Prop :=
  (6 : ℝ) = a * (5 : ℝ)^2 + b * (5 : ℝ) + c ∧
  0 = a * (3 : ℝ)^2 + b * (3 : ℝ) + c

theorem parabola_solution :
  ∃ (a b c : ℝ), parabola_coefficients a b c ∧ (a + b + c = 6) :=
by {
  -- definitions and constraints based on problem conditions
  sorry
}

end NUMINAMATH_GPT_parabola_solution_l1288_128882


namespace NUMINAMATH_GPT_smallest_a_inequality_l1288_128829

theorem smallest_a_inequality 
  (x : ℝ)
  (h1 : x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi)) : 
  (∃ a : ℝ, a = -2.52 ∧ (∀ x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi), 
    ( ((Real.sqrt (Real.cos x / Real.sin x)^2) - (Real.sqrt (Real.sin x / Real.cos x)^2))
    / ((Real.sqrt (Real.sin x)^2) - (Real.sqrt (Real.cos x)^2)) ) < a )) :=
  sorry

end NUMINAMATH_GPT_smallest_a_inequality_l1288_128829


namespace NUMINAMATH_GPT_sum_of_cubes_of_roots_l1288_128836

theorem sum_of_cubes_of_roots (x₁ x₂ : ℝ) (h₀ : 3 * x₁ ^ 2 - 5 * x₁ - 2 = 0)
  (h₁ : 3 * x₂ ^ 2 - 5 * x₂ - 2 = 0) :
  x₁^3 + x₂^3 = 215 / 27 :=
by sorry

end NUMINAMATH_GPT_sum_of_cubes_of_roots_l1288_128836


namespace NUMINAMATH_GPT_cannot_form_right_triangle_l1288_128886

theorem cannot_form_right_triangle :
  ¬ (6^2 + 7^2 = 8^2) :=
by
  sorry

end NUMINAMATH_GPT_cannot_form_right_triangle_l1288_128886


namespace NUMINAMATH_GPT_find_f_log_value_l1288_128830

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 1 then 2^x + 1 else sorry

theorem find_f_log_value (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_spec : ∀ x, 0 < x → x < 1 → f x = 2^x + 1) :
  f (Real.logb (1/2) (1/15)) = -31/15 :=
sorry

end NUMINAMATH_GPT_find_f_log_value_l1288_128830


namespace NUMINAMATH_GPT_tan_pi_div_a_of_point_on_cubed_function_l1288_128804

theorem tan_pi_div_a_of_point_on_cubed_function (a : ℝ) (h : (a, 27) ∈ {p : ℝ × ℝ | p.snd = p.fst ^ 3}) : 
  Real.tan (Real.pi / a) = Real.sqrt 3 := sorry

end NUMINAMATH_GPT_tan_pi_div_a_of_point_on_cubed_function_l1288_128804


namespace NUMINAMATH_GPT_part1_part2_l1288_128864

noncomputable def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

theorem part1 (a b : ℝ) (h₀ : a = 1) (h₁ : b = 2) :
  {x : ℝ | f x a b ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
by
  sorry

theorem part2 (a b : ℝ) (h_min_value : ∀ x : ℝ, f x a b ≥ 3) :
  a + b = 3 → (a > 0 ∧ b > 0) →
  (∃ a b : ℝ, a = b ∧ a + b = 3 ∧ (a = b → f x a b = 3)) →
  (∀ a b : ℝ, (a^2/b + b^2/a) ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1288_128864


namespace NUMINAMATH_GPT_complex_product_l1288_128858

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex numbers z1 and z2
def z1 : ℂ := 1 - i
def z2 : ℂ := 3 + i

-- Statement of the problem
theorem complex_product : z1 * z2 = 4 - 2 * i := by
  sorry

end NUMINAMATH_GPT_complex_product_l1288_128858


namespace NUMINAMATH_GPT_smaller_cylinder_diameter_l1288_128862

theorem smaller_cylinder_diameter
  (vol_large : ℝ)
  (height_large : ℝ)
  (diameter_large : ℝ)
  (height_small : ℝ)
  (ratio : ℝ)
  (π : ℝ)
  (volume_large_eq : vol_large = π * (diameter_large / 2)^2 * height_large)  -- Volume formula for the larger cylinder
  (ratio_eq : ratio = 74.07407407407408) -- Given ratio
  (height_large_eq : height_large = 10)  -- Given height of the larger cylinder
  (diameter_large_eq : diameter_large = 20)  -- Given diameter of the larger cylinder
  (height_small_eq : height_small = 6)  -- Given height of smaller cylinders):
  :
  ∃ (diameter_small : ℝ), diameter_small = 3 := 
by
  sorry

end NUMINAMATH_GPT_smaller_cylinder_diameter_l1288_128862


namespace NUMINAMATH_GPT_total_vehicles_is_120_l1288_128869

def num_trucks : ℕ := 20
def num_tanks : ℕ := 5 * num_trucks
def total_vehicles : ℕ := num_tanks + num_trucks

theorem total_vehicles_is_120 : total_vehicles = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_vehicles_is_120_l1288_128869


namespace NUMINAMATH_GPT_value_of_f_log_half_24_l1288_128854

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_log_half_24 :
  (∀ x : ℝ, f x * -1 = f (-x)) → -- Condition 1: f(x) is an odd function.
  (∀ x : ℝ, f (x + 1) = f (x - 1)) → -- Condition 2: f(x + 1) = f(x - 1).
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2^x - 2) → -- Condition 3: For 0 < x < 1, f(x) = 2^x - 2.
  f (Real.logb 0.5 24) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_value_of_f_log_half_24_l1288_128854


namespace NUMINAMATH_GPT_sum_of_digits_of_N_eq_14_l1288_128847

theorem sum_of_digits_of_N_eq_14 :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ (N % 10 + N / 10 % 10 = 14) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_eq_14_l1288_128847


namespace NUMINAMATH_GPT_find_first_episode_l1288_128892

variable (x : ℕ)
variable (w y z : ℕ)
variable (total_minutes: ℕ)
variable (h1 : w = 62)
variable (h2 : y = 65)
variable (h3 : z = 55)
variable (h4 : total_minutes = 240)

theorem find_first_episode :
  x + w + y + z = total_minutes → x = 58 := 
by
  intro h
  rw [h1, h2, h3, h4] at h
  linarith

end NUMINAMATH_GPT_find_first_episode_l1288_128892


namespace NUMINAMATH_GPT_time_to_cross_pole_l1288_128859

def train_length := 3000 -- in meters
def train_speed_kmh := 90 -- in kilometers per hour

noncomputable def train_speed_mps : ℝ := train_speed_kmh * (1000 / 3600) -- converting speed to meters per second

theorem time_to_cross_pole : (train_length : ℝ) / train_speed_mps = 120 := 
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_time_to_cross_pole_l1288_128859


namespace NUMINAMATH_GPT_bisecting_chord_line_eqn_l1288_128872

theorem bisecting_chord_line_eqn :
  ∀ (x1 y1 x2 y2 : ℝ),
  y1 ^ 2 = 16 * x1 →
  y2 ^ 2 = 16 * x2 →
  (x1 + x2) / 2 = 2 →
  (y1 + y2) / 2 = 1 →
  ∃ (a b c : ℝ), a = 8 ∧ b = -1 ∧ c = -15 ∧
  ∀ (x y : ℝ), y = 8 * x - 15 → a * x + b * y + c = 0 :=
by 
  sorry

end NUMINAMATH_GPT_bisecting_chord_line_eqn_l1288_128872


namespace NUMINAMATH_GPT_andrew_purchased_mangoes_l1288_128838

theorem andrew_purchased_mangoes
  (m : Nat)
  (h1 : 14 * 54 = 756)
  (h2 : 756 + 62 * m = 1376) :
  m = 10 :=
by
  sorry

end NUMINAMATH_GPT_andrew_purchased_mangoes_l1288_128838


namespace NUMINAMATH_GPT_find_n_sin_eq_l1288_128823

theorem find_n_sin_eq (n : ℤ) (h₁ : -180 ≤ n) (h₂ : n ≤ 180) (h₃ : Real.sin (n * Real.pi / 180) = Real.sin (680 * Real.pi / 180)) :
  n = 40 ∨ n = 140 :=
by
  sorry

end NUMINAMATH_GPT_find_n_sin_eq_l1288_128823


namespace NUMINAMATH_GPT_largest_inscribed_square_size_l1288_128888

noncomputable def side_length_of_largest_inscribed_square : ℝ :=
  6 - 2 * Real.sqrt 3

theorem largest_inscribed_square_size (side_length_of_square : ℝ)
  (equi_triangles_shared_side : ℝ)
  (vertexA_of_square : ℝ)
  (vertexB_of_square : ℝ)
  (vertexC_of_square : ℝ)
  (vertexD_of_square : ℝ)
  (vertexF_of_triangles : ℝ)
  (vertexG_of_triangles : ℝ) :
  side_length_of_square = 12 →
  equi_triangles_shared_side = vertexB_of_square - vertexA_of_square →
  vertexF_of_triangles = vertexD_of_square - vertexC_of_square →
  vertexG_of_triangles = vertexB_of_square - vertexA_of_square →
  side_length_of_largest_inscribed_square = 6 - 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_largest_inscribed_square_size_l1288_128888


namespace NUMINAMATH_GPT_nonagon_diagonals_l1288_128842

theorem nonagon_diagonals (n : ℕ) (h1 : n = 9) : (n * (n - 3)) / 2 = 27 := by
  sorry

end NUMINAMATH_GPT_nonagon_diagonals_l1288_128842


namespace NUMINAMATH_GPT_max_cursed_roads_l1288_128894

theorem max_cursed_roads (cities roads N kingdoms : ℕ) (h1 : cities = 1000) (h2 : roads = 2017)
  (h3 : cities = 1 → cities = 1000 → N ≤ 1024 → kingdoms = 7 → True) :
  max_N = 1024 :=
by
  sorry

end NUMINAMATH_GPT_max_cursed_roads_l1288_128894


namespace NUMINAMATH_GPT_compound_interest_rate_l1288_128845

theorem compound_interest_rate (P : ℝ) (r : ℝ) (t : ℕ) (A : ℝ) 
  (h1 : t = 15) (h2 : A = (9 / 5) * P) :
  (1 + r) ^ t = (9 / 5) → 
  r ≠ 0.05 ∧ r ≠ 0.06 ∧ r ≠ 0.07 ∧ r ≠ 0.08 :=
by
  -- Sorry could be placed here for now
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l1288_128845


namespace NUMINAMATH_GPT_trajectory_of_square_is_line_l1288_128832

open Complex

theorem trajectory_of_square_is_line (z : ℂ) (h : z.re = z.im) : ∃ c : ℝ, z^2 = Complex.I * (c : ℂ) :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_square_is_line_l1288_128832


namespace NUMINAMATH_GPT_trajectory_is_parabola_l1288_128890

theorem trajectory_is_parabola (C : ℝ × ℝ) (M : ℝ × ℝ) (l : ℝ → ℝ)
  (hM : M = (0, 3)) (hl : ∀ y, l y = -3)
  (h : dist C M = |C.2 + 3|) : C.1^2 = 12 * C.2 := by
  sorry

end NUMINAMATH_GPT_trajectory_is_parabola_l1288_128890


namespace NUMINAMATH_GPT_total_cost_of_trip_l1288_128885

def totalDistance (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def gallonsUsed (distance miles_per_gallon : ℕ) : ℕ :=
  distance / miles_per_gallon

def totalCost (gallons : ℕ) (cost_per_gallon : ℕ) : ℕ :=
  gallons * cost_per_gallon

theorem total_cost_of_trip :
  (totalDistance 10 6 5 9 = 30) →
  (gallonsUsed 30 15 = 2) →
  totalCost 2 35 = 700 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_trip_l1288_128885


namespace NUMINAMATH_GPT_eight_times_10x_plus_14pi_l1288_128846

theorem eight_times_10x_plus_14pi (x : ℝ) (Q : ℝ) (h : 4 * (5 * x + 7 * π) = Q) : 
  8 * (10 * x + 14 * π) = 4 * Q := 
by {
  sorry  -- proof is omitted
}

end NUMINAMATH_GPT_eight_times_10x_plus_14pi_l1288_128846


namespace NUMINAMATH_GPT_chord_length_in_circle_l1288_128851

theorem chord_length_in_circle 
  (radius : ℝ) 
  (chord_midpoint_perpendicular_radius : ℝ)
  (r_eq_10 : radius = 10)
  (cmp_eq_5 : chord_midpoint_perpendicular_radius = 5) : 
  ∃ (chord_length : ℝ), chord_length = 10 * Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_chord_length_in_circle_l1288_128851


namespace NUMINAMATH_GPT_osmanthus_trees_variance_l1288_128822

variable (n : Nat) (p : ℚ)

def variance_binomial_distribution (n : Nat) (p : ℚ) : ℚ :=
  n * p * (1 - p)

theorem osmanthus_trees_variance (n : Nat) (p : ℚ) (h₁ : n = 4) (h₂ : p = 4 / 5) :
  variance_binomial_distribution n p = 16 / 25 := by
  sorry

end NUMINAMATH_GPT_osmanthus_trees_variance_l1288_128822


namespace NUMINAMATH_GPT_parallel_line_slope_l1288_128853

theorem parallel_line_slope (x y : ℝ) :
  (∃ k b : ℝ, 3 * x + 6 * y = k * x + b) ∧ (∃ a b, y = a * x + b) ∧ 3 * x + 6 * y = -24 → 
  ∃ m : ℝ, m = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_line_slope_l1288_128853


namespace NUMINAMATH_GPT_who_scored_full_marks_l1288_128883

-- Define students and their statements
inductive Student
| A | B | C

open Student

def scored_full_marks (s : Student) : Prop :=
  match s with
  | A => true
  | B => true
  | C => true

def statement_A : Prop := scored_full_marks A
def statement_B : Prop := ¬ scored_full_marks C
def statement_C : Prop := statement_B

-- Given conditions
def exactly_one_lied (a b c : Prop) : Prop :=
  (a ∧ ¬ b ∧ ¬ c) ∨ (¬ a ∧ b ∧ ¬ c) ∨ (¬ a ∧ ¬ b ∧ c)

-- Main proof statement: Prove that B scored full marks
theorem who_scored_full_marks (h : exactly_one_lied statement_A statement_B statement_C) : scored_full_marks B :=
sorry

end NUMINAMATH_GPT_who_scored_full_marks_l1288_128883


namespace NUMINAMATH_GPT_infinitely_many_solutions_b_value_l1288_128849

theorem infinitely_many_solutions_b_value :
  ∀ (x : ℝ) (b : ℝ), (5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := 
by
  intro x b
  sorry

end NUMINAMATH_GPT_infinitely_many_solutions_b_value_l1288_128849


namespace NUMINAMATH_GPT_parabola_focus_l1288_128810

theorem parabola_focus (a : ℝ) (h : a ≠ 0) : ∃ q : ℝ, q = 1/(4*a) ∧ (0, q) = (0, 1/(4*a)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l1288_128810


namespace NUMINAMATH_GPT_quadratic_completing_square_t_value_l1288_128852

theorem quadratic_completing_square_t_value :
  ∃ q t : ℝ, 4 * x^2 - 24 * x - 96 = 0 → (x + q) ^ 2 = t ∧ t = 33 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_completing_square_t_value_l1288_128852


namespace NUMINAMATH_GPT_part1_part2_l1288_128815

noncomputable def triangleABC (a : ℝ) (cosB : ℝ) (b : ℝ) (SinA : ℝ) : Prop :=
  cosB = 3 / 5 ∧ b = 4 → SinA = 2 / 5

noncomputable def triangleABC2 (a : ℝ) (cosB : ℝ) (S : ℝ) (b c : ℝ) : Prop :=
  cosB = 3 / 5 ∧ S = 4 → b = Real.sqrt 17 ∧ c = 5

theorem part1 :
  triangleABC 2 (3 / 5) 4 (2 / 5) :=
by {
  sorry
}

theorem part2 :
  triangleABC2 2 (3 / 5) 4 (Real.sqrt 17) 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_l1288_128815


namespace NUMINAMATH_GPT_smallest_possible_area_of_ellipse_l1288_128848

theorem smallest_possible_area_of_ellipse
  (a b : ℝ)
  (h_ellipse : ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → 
    (((x - 1/2)^2 + y^2 = 1/4) ∨ ((x + 1/2)^2 + y^2 = 1/4))) :
  ∃ (k : ℝ), (a * b * π = 4 * π) :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_area_of_ellipse_l1288_128848


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1288_128889

theorem solution_set_of_inequality : {x : ℝ | -3 < x ∧ x < 1} = {x : ℝ | x^2 + 2 * x < 3} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1288_128889


namespace NUMINAMATH_GPT_kris_suspension_days_per_instance_is_three_l1288_128856

-- Define the basic parameters given in the conditions
def total_fingers_toes : ℕ := 20
def total_bullying_instances : ℕ := 20
def multiplier : ℕ := 3

-- Define total suspension days according to the conditions
def total_suspension_days : ℕ := multiplier * total_fingers_toes

-- Define the goal: to find the number of suspension days per instance
def suspension_days_per_instance : ℕ := total_suspension_days / total_bullying_instances

-- The theorem to prove that Kris was suspended for 3 days per instance
theorem kris_suspension_days_per_instance_is_three : suspension_days_per_instance = 3 := by
  -- Skip the actual proof, focus only on the statement
  sorry

end NUMINAMATH_GPT_kris_suspension_days_per_instance_is_three_l1288_128856


namespace NUMINAMATH_GPT_cost_of_450_candies_l1288_128839

theorem cost_of_450_candies (box_cost : ℝ) (box_candies : ℕ) (total_candies : ℕ) 
  (h1 : box_cost = 7.50) (h2 : box_candies = 30) (h3 : total_candies = 450) : 
  (total_candies / box_candies) * box_cost = 112.50 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_450_candies_l1288_128839


namespace NUMINAMATH_GPT_find_min_value_of_quadratic_l1288_128818

theorem find_min_value_of_quadratic : ∀ x : ℝ, ∃ c : ℝ, (∃ a b : ℝ, (y = 2*x^2 + 8*x + 7 ∧ (∀ x : ℝ, y ≥ c)) ∧ c = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_min_value_of_quadratic_l1288_128818


namespace NUMINAMATH_GPT_line_l_equation_symmetrical_line_equation_l1288_128868

theorem line_l_equation (x y : ℝ) (h₁ : 3 * x + 4 * y - 2 = 0) (h₂ : 2 * x + y + 2 = 0) :
  2 * x + y + 2 = 0 :=
sorry

theorem symmetrical_line_equation (x y : ℝ) :
  (2 * x + y + 2 = 0) → (2 * x + y - 2 = 0) :=
sorry

end NUMINAMATH_GPT_line_l_equation_symmetrical_line_equation_l1288_128868


namespace NUMINAMATH_GPT_arithmetic_sequences_diff_l1288_128865

theorem arithmetic_sequences_diff
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (d_a d_b : ℤ)
  (ha : ∀ n, a n = 3 + n * d_a)
  (hb : ∀ n, b n = -3 + n * d_b)
  (h : a 19 - b 19 = 16) :
  a 10 - b 10 = 11 := by
    sorry

end NUMINAMATH_GPT_arithmetic_sequences_diff_l1288_128865


namespace NUMINAMATH_GPT_sum_first_5_arithmetic_l1288_128834

theorem sum_first_5_arithmetic (u : ℕ → ℝ) (h : u 3 = 0) : 
  (u 1 + u 2 + u 3 + u 4 + u 5) = 0 :=
sorry

end NUMINAMATH_GPT_sum_first_5_arithmetic_l1288_128834


namespace NUMINAMATH_GPT_max_value_ad_bc_l1288_128811

theorem max_value_ad_bc (a b c d : ℤ) (h₁ : a ∈ ({-1, 1, 2} : Set ℤ))
                          (h₂ : b ∈ ({-1, 1, 2} : Set ℤ))
                          (h₃ : c ∈ ({-1, 1, 2} : Set ℤ))
                          (h₄ : d ∈ ({-1, 1, 2} : Set ℤ)) :
  ad - bc ≤ 6 :=
by sorry

end NUMINAMATH_GPT_max_value_ad_bc_l1288_128811


namespace NUMINAMATH_GPT_solve_q_l1288_128817

theorem solve_q (n m q : ℤ) 
  (h₁ : 5/6 = n/72) 
  (h₂ : 5/6 = (m + n)/90) 
  (h₃ : 5/6 = (q - m)/150) : 
  q = 140 := by
  sorry

end NUMINAMATH_GPT_solve_q_l1288_128817


namespace NUMINAMATH_GPT_product_remainder_l1288_128875

theorem product_remainder (a b c d : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 4) (hd : d % 7 = 5) :
  (a * b * c * d) % 7 = 1 :=
by
  sorry

end NUMINAMATH_GPT_product_remainder_l1288_128875


namespace NUMINAMATH_GPT_k_is_2_l1288_128802

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x - 1
def g (x : ℝ) : ℝ := 0
noncomputable def h (x : ℝ) : ℝ := (x + 1) * Real.log x

theorem k_is_2 :
  (∀ x ∈ Set.Icc 1 (2 * Real.exp 1), 0 ≤ f k x ∧ f k x ≤ h x) ↔ (k = 2) :=
  sorry

end NUMINAMATH_GPT_k_is_2_l1288_128802


namespace NUMINAMATH_GPT_int_product_negative_max_negatives_l1288_128877

theorem int_product_negative_max_negatives (n : ℤ) (hn : n ≤ 9) (hp : n % 2 = 1) :
  ∃ m : ℤ, n + m = m ∧ m ≥ 0 :=
by
  use 9
  sorry

end NUMINAMATH_GPT_int_product_negative_max_negatives_l1288_128877


namespace NUMINAMATH_GPT_min_cost_speed_l1288_128876

noncomputable def fuel_cost (v : ℝ) : ℝ := (1/200) * v^3

theorem min_cost_speed 
  (v : ℝ) 
  (u : ℝ) 
  (other_costs : ℝ) 
  (h1 : u = (1/200) * v^3) 
  (h2 : u = 40) 
  (h3 : v = 20) 
  (h4 : other_costs = 270) 
  (b : ℝ) 
  : ∃ v_min, v_min = 30 ∧ 
    ∀ (v : ℝ), (0 < v ∧ v ≤ b) → 
    ((fuel_cost v / v + other_costs / v) ≥ (fuel_cost v_min / v_min + other_costs / v_min)) := 
sorry

end NUMINAMATH_GPT_min_cost_speed_l1288_128876


namespace NUMINAMATH_GPT_donny_paid_l1288_128835

variable (total_capacity initial_fuel price_per_liter change : ℕ)

theorem donny_paid (h1 : total_capacity = 150) 
                   (h2 : initial_fuel = 38) 
                   (h3 : price_per_liter = 3) 
                   (h4 : change = 14) : 
                   (total_capacity - initial_fuel) * price_per_liter + change = 350 := 
by
  sorry

end NUMINAMATH_GPT_donny_paid_l1288_128835


namespace NUMINAMATH_GPT_additional_time_due_to_leak_l1288_128899

theorem additional_time_due_to_leak 
  (normal_time_per_barrel : ℕ)
  (leak_time_per_barrel : ℕ)
  (barrels : ℕ)
  (normal_duration : normal_time_per_barrel = 3)
  (leak_duration : leak_time_per_barrel = 5)
  (barrels_needed : barrels = 12) :
  (leak_time_per_barrel * barrels - normal_time_per_barrel * barrels) = 24 := 
by
  sorry

end NUMINAMATH_GPT_additional_time_due_to_leak_l1288_128899


namespace NUMINAMATH_GPT_xiaoqiang_xiaolin_stamps_l1288_128819

-- Definitions for initial conditions and constraints
noncomputable def x : ℤ := 227
noncomputable def y : ℤ := 221
noncomputable def k : ℤ := sorry

-- Proof problem as a theorem
theorem xiaoqiang_xiaolin_stamps:
  x + y > 400 ∧
  x - k = (13 / 19) * (y + k) ∧
  y - k = (11 / 17) * (x + k) ∧
  x = 227 ∧ 
  y = 221 :=
by
  sorry

end NUMINAMATH_GPT_xiaoqiang_xiaolin_stamps_l1288_128819


namespace NUMINAMATH_GPT_root_polynomial_h_l1288_128828

theorem root_polynomial_h (h : ℤ) : (2^3 + h * 2 + 10 = 0) → h = -9 :=
by
  sorry

end NUMINAMATH_GPT_root_polynomial_h_l1288_128828


namespace NUMINAMATH_GPT_fraction_multiplication_l1288_128840

-- Define the problem as a theorem in Lean
theorem fraction_multiplication
  (a b x : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) (ha : a ≠ 0): 
  (3 * a * b / x) * (2 * x^2 / (9 * a * b^2)) = (2 * x) / (3 * b) := 
by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_l1288_128840


namespace NUMINAMATH_GPT_find_x_l1288_128871

theorem find_x (x : ℚ) : |x + 3| = |x - 4| → x = 1/2 := 
by 
-- Add appropriate content here
sorry

end NUMINAMATH_GPT_find_x_l1288_128871


namespace NUMINAMATH_GPT_min_value_geometric_seq_l1288_128803

theorem min_value_geometric_seq (a : ℕ → ℝ) (m n : ℕ) (h_pos : ∀ k, a k > 0)
  (h1 : a 1 = 1)
  (h2 : a 7 = a 6 + 2 * a 5)
  (h3 : a m * a n = 16) :
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_geometric_seq_l1288_128803


namespace NUMINAMATH_GPT_widget_production_l1288_128801

theorem widget_production (p q r s t : ℕ) :
  (s * q * t) / (p * r) = (sqt / pr) := 
sorry

end NUMINAMATH_GPT_widget_production_l1288_128801


namespace NUMINAMATH_GPT_melinda_doughnuts_picked_l1288_128879

theorem melinda_doughnuts_picked :
  (∀ d h_coffee m_coffee : ℕ, d = 3 → h_coffee = 4 → m_coffee = 6 →
    ∀ cost_d cost_h cost_m : ℝ, cost_d = 0.45 → 
    cost_h = 4.91 → cost_m = 7.59 → 
    ∃ m_doughnuts : ℕ, cost_m - m_coffee * ((cost_h - d * cost_d) / h_coffee) = m_doughnuts * cost_d) → 
  ∃ n : ℕ, n = 5 := 
by sorry

end NUMINAMATH_GPT_melinda_doughnuts_picked_l1288_128879


namespace NUMINAMATH_GPT_not_enough_info_sweets_l1288_128878

theorem not_enough_info_sweets
    (S : ℕ)         -- Initial number of sweet cookies.
    (initial_salty : ℕ := 6)  -- Initial number of salty cookies given as 6.
    (eaten_sweets : ℕ := 20)   -- Number of sweet cookies Paco ate.
    (eaten_salty : ℕ := 34)    -- Number of salty cookies Paco ate.
    (diff_eaten : eaten_salty - eaten_sweets = 14) -- Paco ate 14 more salty cookies than sweet cookies.
    : (∃ S', S' = S) → False :=  -- Conclusion: Not enough information to determine initial number of sweet cookies S.
by
  sorry

end NUMINAMATH_GPT_not_enough_info_sweets_l1288_128878


namespace NUMINAMATH_GPT_linear_function_not_in_first_quadrant_l1288_128824

theorem linear_function_not_in_first_quadrant:
  ∀ x y : ℝ, y = -2 * x - 3 → ¬ (x > 0 ∧ y > 0) :=
by
 -- proof steps would go here
 sorry

end NUMINAMATH_GPT_linear_function_not_in_first_quadrant_l1288_128824


namespace NUMINAMATH_GPT_train_crosses_bridge_in_12_2_seconds_l1288_128806

def length_of_train : ℕ := 110
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 134

def speed_of_train_ms : ℚ := speed_of_train_kmh * (1000 : ℚ) / (3600 : ℚ)
def total_distance : ℕ := length_of_train + length_of_bridge

noncomputable def time_to_cross_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_12_2_seconds : time_to_cross_bridge = 12.2 := by
  sorry

end NUMINAMATH_GPT_train_crosses_bridge_in_12_2_seconds_l1288_128806


namespace NUMINAMATH_GPT_alex_silver_tokens_l1288_128826

-- Definitions and conditions
def initialRedTokens : ℕ := 100
def initialBlueTokens : ℕ := 50
def firstBoothRedChange (x : ℕ) : ℕ := 3 * x
def firstBoothSilverGain (x : ℕ) : ℕ := 2 * x
def firstBoothBlueGain (x : ℕ) : ℕ := x
def secondBoothBlueChange (y : ℕ) : ℕ := 2 * y
def secondBoothSilverGain (y : ℕ) : ℕ := y
def secondBoothRedGain (y : ℕ) : ℕ := y

-- Final conditions when no more exchanges are possible
def finalRedTokens (x y : ℕ) : ℕ := initialRedTokens - firstBoothRedChange x + secondBoothRedGain y
def finalBlueTokens (x y : ℕ) : ℕ := initialBlueTokens + firstBoothBlueGain x - secondBoothBlueChange y

-- Total silver tokens calculation
def totalSilverTokens (x y : ℕ) : ℕ := firstBoothSilverGain x + secondBoothSilverGain y

-- Proof that in the end, Alex has 147 silver tokens
theorem alex_silver_tokens : 
  ∃ (x y : ℕ), finalRedTokens x y = 2 ∧ finalBlueTokens x y = 1 ∧ totalSilverTokens x y = 147 :=
by
  -- the proof logic will be filled here
  sorry

end NUMINAMATH_GPT_alex_silver_tokens_l1288_128826


namespace NUMINAMATH_GPT_tim_total_points_l1288_128809

theorem tim_total_points :
  let single_points := 1000
  let tetris_points := 8 * single_points
  let singles := 6
  let tetrises := 4
  let total_points := singles * single_points + tetrises * tetris_points
  total_points = 38000 :=
by
  sorry

end NUMINAMATH_GPT_tim_total_points_l1288_128809


namespace NUMINAMATH_GPT_base5_to_octal_1234_eval_f_at_3_l1288_128821

-- Definition of base conversion from base 5 to decimal and to octal
def base5_to_decimal (n : Nat) : Nat :=
  match n with
  | 1234 => 1 * 5^3 + 2 * 5^2 + 3 * 5 + 4
  | _ => 0

def decimal_to_octal (n : Nat) : Nat :=
  match n with
  | 194 => 302
  | _ => 0

-- Definition of the polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x
def f (x : Nat) : Nat :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

-- Definition of Horner's method evaluation
def horner_eval (x : Nat) : Nat :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1) * x

-- Theorem statement for base-5 to octal conversion
theorem base5_to_octal_1234 : base5_to_decimal 1234 = 194 ∧ decimal_to_octal 194 = 302 :=
  by
    sorry

-- Theorem statement for polynomial evaluation using Horner's method
theorem eval_f_at_3 : horner_eval 3 = f 3 ∧ f 3 = 21324 :=
  by
    sorry

end NUMINAMATH_GPT_base5_to_octal_1234_eval_f_at_3_l1288_128821


namespace NUMINAMATH_GPT_odd_ints_divisibility_l1288_128808

theorem odd_ints_divisibility (a b : ℤ) (ha_odd : a % 2 = 1) (hb_odd : b % 2 = 1) (hdiv : 2 * a * b + 1 ∣ a^2 + b^2 + 1) : a = b :=
sorry

end NUMINAMATH_GPT_odd_ints_divisibility_l1288_128808


namespace NUMINAMATH_GPT_range_of_k_l1288_128870

variables (k : ℝ)

def vector_a (k : ℝ) : ℝ × ℝ := (-k, 4)
def vector_b (k : ℝ) : ℝ × ℝ := (k, k + 3)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem range_of_k (h : 0 < dot_product (vector_a k) (vector_b k)) : 
  -2 < k ∧ k < 0 ∨ 0 < k ∧ k < 6 :=
sorry

end NUMINAMATH_GPT_range_of_k_l1288_128870


namespace NUMINAMATH_GPT_mortgage_loan_amount_l1288_128812

theorem mortgage_loan_amount (C : ℝ) (hC : C = 8000000) : 0.75 * C = 6000000 :=
by
  sorry

end NUMINAMATH_GPT_mortgage_loan_amount_l1288_128812


namespace NUMINAMATH_GPT_second_third_parts_length_l1288_128807

variable (total_length : ℝ) (first_part : ℝ) (last_part : ℝ)
variable (second_third_part_length : ℝ)

def is_equal_length (x y : ℝ) := x = y

theorem second_third_parts_length :
  total_length = 74.5 ∧ first_part = 15.5 ∧ last_part = 16 → 
  is_equal_length (second_third_part_length) 21.5 :=
by
  intros h
  let remaining_distance := total_length - first_part - last_part
  let second_third_part_length := remaining_distance / 2
  sorry

end NUMINAMATH_GPT_second_third_parts_length_l1288_128807


namespace NUMINAMATH_GPT_contrapositive_example_l1288_128881

theorem contrapositive_example (x : ℝ) (h : -2 < x ∧ x < 2) : x^2 < 4 :=
sorry

end NUMINAMATH_GPT_contrapositive_example_l1288_128881


namespace NUMINAMATH_GPT_helen_needed_gas_l1288_128887

-- Definitions based on conditions
def cuts_per_month_routine_1 : ℕ := 2 -- Cuts per month for March, April, September, October
def cuts_per_month_routine_2 : ℕ := 4 -- Cuts per month for May, June, July, August
def months_routine_1 : ℕ := 4 -- Number of months with routine 1
def months_routine_2 : ℕ := 4 -- Number of months with routine 2
def gas_per_fill : ℕ := 2 -- Gallons of gas used every 4th cut
def cuts_per_fill : ℕ := 4 -- Number of cuts per fill

-- Total number of cuts in routine 1 months
def total_cuts_routine_1 : ℕ := cuts_per_month_routine_1 * months_routine_1

-- Total number of cuts in routine 2 months
def total_cuts_routine_2 : ℕ := cuts_per_month_routine_2 * months_routine_2

-- Total cuts from March to October
def total_cuts : ℕ := total_cuts_routine_1 + total_cuts_routine_2

-- Total fills needed from March to October
def total_fills : ℕ := total_cuts / cuts_per_fill

-- Total gallons of gas needed
def total_gal_of_gas : ℕ := total_fills * gas_per_fill

-- The statement to prove
theorem helen_needed_gas : total_gal_of_gas = 12 :=
by
  -- This would be replaced by our solution steps.
  sorry

end NUMINAMATH_GPT_helen_needed_gas_l1288_128887


namespace NUMINAMATH_GPT_geometric_sum_formula_l1288_128896

noncomputable def geometric_sequence_sum (n : ℕ) : ℕ :=
  sorry

theorem geometric_sum_formula (a : ℕ → ℕ)
  (h_geom : ∀ n, a (n + 1) = 2 * a n)
  (h_a1_a2 : a 0 + a 1 = 3)
  (h_a1_a2_a3 : a 0 * a 1 * a 2 = 8) :
  geometric_sequence_sum n = 2^n - 1 :=
sorry

end NUMINAMATH_GPT_geometric_sum_formula_l1288_128896


namespace NUMINAMATH_GPT_xy_value_l1288_128837

namespace ProofProblem

variables {x y : ℤ}

theorem xy_value (h1 : x * (x + y) = x^2 + 12) (h2 : x - y = 3) : x * y = 12 :=
by
  -- The proof is not required here
  sorry

end ProofProblem

end NUMINAMATH_GPT_xy_value_l1288_128837


namespace NUMINAMATH_GPT_power_equality_l1288_128841

-- Definitions based on conditions
def nine := 3^2

-- Theorem stating the given mathematical problem
theorem power_equality : nine^4 = 3^8 := by
  sorry

end NUMINAMATH_GPT_power_equality_l1288_128841


namespace NUMINAMATH_GPT_solution_set_inequality_l1288_128850

theorem solution_set_inequality {x : ℝ} : 
  ((x - 1)^2 < 1) ↔ (0 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1288_128850


namespace NUMINAMATH_GPT_father_son_skating_ratio_l1288_128893

theorem father_son_skating_ratio (v_f v_s : ℝ) (h1 : v_f > v_s) (h2 : (v_f + v_s) / (v_f - v_s) = 5) :
  v_f / v_s = 1.5 :=
sorry

end NUMINAMATH_GPT_father_son_skating_ratio_l1288_128893


namespace NUMINAMATH_GPT_homogeneous_diff_eq_solution_l1288_128831

open Real

theorem homogeneous_diff_eq_solution (C : ℝ) : 
  ∀ (x y : ℝ), (y^4 - 2 * x^3 * y) * (dx) + (x^4 - 2 * x * y^3) * (dy) = 0 ↔ x^3 + y^3 = C * x * y :=
by
  sorry

end NUMINAMATH_GPT_homogeneous_diff_eq_solution_l1288_128831


namespace NUMINAMATH_GPT_functional_relationship_inversely_proportional_l1288_128895

-- Definitions based on conditions
def table_data : List (ℝ × ℝ) := [(100, 1.00), (200, 0.50), (400, 0.25), (500, 0.20)]

-- The main conjecture to be proved
theorem functional_relationship_inversely_proportional (y x : ℝ) (h : (x, y) ∈ table_data) : y = 100 / x :=
sorry

end NUMINAMATH_GPT_functional_relationship_inversely_proportional_l1288_128895


namespace NUMINAMATH_GPT_fraction_eggs_used_for_cupcakes_l1288_128874

theorem fraction_eggs_used_for_cupcakes:
  ∀ (total_eggs crepes_fraction remaining_eggs after_cupcakes_eggs used_for_cupcakes_fraction: ℚ),
  total_eggs = 36 →
  crepes_fraction = 1 / 4 →
  after_cupcakes_eggs = 9 →
  used_for_cupcakes_fraction = 2 / 3 →
  (total_eggs * (1 - crepes_fraction) - after_cupcakes_eggs) / (total_eggs * (1 - crepes_fraction)) = used_for_cupcakes_fraction :=
by
  intros
  sorry

end NUMINAMATH_GPT_fraction_eggs_used_for_cupcakes_l1288_128874


namespace NUMINAMATH_GPT_apples_per_pie_l1288_128867

-- Definitions of given conditions
def total_apples : ℕ := 75
def handed_out_apples : ℕ := 19
def remaining_apples : ℕ := total_apples - handed_out_apples
def pies_made : ℕ := 7

-- Statement of the problem to be proved
theorem apples_per_pie : remaining_apples / pies_made = 8 := by
  sorry

end NUMINAMATH_GPT_apples_per_pie_l1288_128867


namespace NUMINAMATH_GPT_min_eq_neg_one_implies_x_eq_two_l1288_128813

theorem min_eq_neg_one_implies_x_eq_two (x : ℝ) (h : min (2*x - 5) (x + 1) = -1) : x = 2 :=
sorry

end NUMINAMATH_GPT_min_eq_neg_one_implies_x_eq_two_l1288_128813


namespace NUMINAMATH_GPT_inequality_abc_equality_condition_abc_l1288_128820

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  (a / (2 * a + 1)) + (b / (3 * b + 1)) + (c / (6 * c + 1)) ≤ 1 / 2 :=
sorry

theorem equality_condition_abc (a b c : ℝ) :
  (a / (2 * a + 1)) + (b / (3 * b + 1)) + (c / (6 * c + 1)) = 1 / 2 ↔ 
  a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6 :=
sorry

end NUMINAMATH_GPT_inequality_abc_equality_condition_abc_l1288_128820


namespace NUMINAMATH_GPT_highest_financial_backing_l1288_128880

-- Let x be the lowest level of financial backing
-- Define the five levels of backing as x, 6x, 36x, 216x, 1296x
-- Given that the total raised is $200,000

theorem highest_financial_backing (x : ℝ) 
  (h₁: 50 * x + 20 * 6 * x + 12 * 36 * x + 7 * 216 * x + 4 * 1296 * x = 200000) : 
  1296 * x = 35534 :=
sorry

end NUMINAMATH_GPT_highest_financial_backing_l1288_128880


namespace NUMINAMATH_GPT_negation_proposition_l1288_128805

open Set

theorem negation_proposition :
  ¬ (∀ x : ℝ, x^2 + 2 * x + 5 > 0) → (∃ x : ℝ, x^2 + 2 * x + 5 ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_proposition_l1288_128805


namespace NUMINAMATH_GPT_determine_pq_value_l1288_128827

noncomputable def p : ℝ → ℝ := λ x => 16 * x
noncomputable def q : ℝ → ℝ := λ x => (x + 4) * (x - 1)

theorem determine_pq_value : (p (-1) / q (-1)) = 8 / 3 := by
  sorry

end NUMINAMATH_GPT_determine_pq_value_l1288_128827


namespace NUMINAMATH_GPT_min_sum_of_ab_l1288_128891

theorem min_sum_of_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + 3 * b = a * b) :
  a + b ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_min_sum_of_ab_l1288_128891


namespace NUMINAMATH_GPT_find_base_a_l1288_128866

theorem find_base_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : (if a < 1 then a + a^2 else a^2 + a) = 12) : a = 3 := 
sorry

end NUMINAMATH_GPT_find_base_a_l1288_128866


namespace NUMINAMATH_GPT_find_x_value_l1288_128860

def acid_solution (m : ℕ) (x : ℕ) (h : m > 25) : Prop :=
  let initial_acid := m^2 / 100
  let total_volume := m + x
  let new_acid_concentration := (m - 5) / 100 * (m + x)
  initial_acid = new_acid_concentration

theorem find_x_value (m : ℕ) (h : m > 25) (x : ℕ) :
  (acid_solution m x h) → x = 5 * m / (m - 5) :=
sorry

end NUMINAMATH_GPT_find_x_value_l1288_128860


namespace NUMINAMATH_GPT_find_a_l1288_128898

def has_root_greater_than_zero (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ ((3 * x - 1) / (x - 3) = a / (3 - x) - 1)

theorem find_a (a : ℝ) : has_root_greater_than_zero a → a = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1288_128898


namespace NUMINAMATH_GPT_math_problem_l1288_128855

theorem math_problem :
  let a := 481 * 7
  let b := 426 * 5
  ((a + b) ^ 3 - 4 * a * b) = 166021128033 := 
by
  let a := 481 * 7
  let b := 426 * 5
  sorry

end NUMINAMATH_GPT_math_problem_l1288_128855


namespace NUMINAMATH_GPT_probability_both_boys_or_both_girls_l1288_128816

theorem probability_both_boys_or_both_girls 
  (total_students : ℕ) (boys : ℕ) (girls : ℕ) :
  total_students = 5 → boys = 2 → girls = 3 →
    (∃ (p : ℚ), p = 2/5) :=
by
  intros ht hb hg
  sorry

end NUMINAMATH_GPT_probability_both_boys_or_both_girls_l1288_128816
