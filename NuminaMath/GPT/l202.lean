import Mathlib

namespace number_of_six_digit_palindromes_l202_202019

def is_six_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9 ∧ 
    n = 100001 * a + 10010 * b + 1100 * c + 100 * d 

theorem number_of_six_digit_palindromes : 
  {n : ℕ | is_six_digit_palindrome n}.to_finset.card = 9000 :=
sorry

end number_of_six_digit_palindromes_l202_202019


namespace smaller_number_is_270_l202_202267

theorem smaller_number_is_270 (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : S = 270 :=
sorry

end smaller_number_is_270_l202_202267


namespace part1_part2_part3_l202_202895

def A (x : ℝ) : Prop := x^2 - x - 2 > 0
def B (x : ℝ) : Prop := 3 - |x| ≥ 0
def C (x : ℝ) (p : ℝ) : Prop := 4 * x + p < 0

theorem part1 : 
  {x : ℝ | A x} ∩ {x | B x} = {x | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} :=
sorry

theorem part2 : 
  {x : ℝ | A x} ∪ {x | B x} = set.univ :=
sorry

theorem part3 (p : ℝ) : 
  (∀ x, C x p → A x) → p ≥ 4 :=
sorry

end part1_part2_part3_l202_202895


namespace total_modules_in_stock_l202_202838

-- Given conditions
def module_cost_high : ℝ := 10
def module_cost_low : ℝ := 3.5
def total_stock_value : ℝ := 45
def low_module_count : ℕ := 10

-- To be proved: total number of modules in stock
theorem total_modules_in_stock (x : ℕ) (y : ℕ) (h1 : y = low_module_count) 
  (h2 : module_cost_high * x + module_cost_low * y = total_stock_value) : 
  x + y = 11 := 
sorry

end total_modules_in_stock_l202_202838


namespace measure_of_angle_l202_202825

-- Define the condition that α lies on the line y = x
def terminal_side_on_line (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 4

-- State the theorem
theorem measure_of_angle (α : ℝ) (h : terminal_side_on_line α) : 
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 4 :=
begin
  sorry -- The proof would go here
end

end measure_of_angle_l202_202825


namespace find_n_l202_202054

/-- Given: 
1. The second term in the expansion of (x + a)^n is binom n 1 * x^(n-1) * a = 210.
2. The third term in the expansion of (x + a)^n is binom n 2 * x^(n-2) * a^2 = 840.
3. The fourth term in the expansion of (x + a)^n is binom n 3 * x^(n-3) * a^3 = 2520.
We are to prove that n = 10. -/
theorem find_n (x a : ℕ) (n : ℕ)
  (h1 : Nat.choose n 1 * x^(n-1) * a = 210)
  (h2 : Nat.choose n 2 * x^(n-2) * a^2 = 840)
  (h3 : Nat.choose n 3 * x^(n-3) * a^3 = 2520) : 
  n = 10 := by sorry

end find_n_l202_202054


namespace integral_abs_eq_split_l202_202349

theorem integral_abs_eq_split :
  ∫ x in -1..1, |x| = ∫ x in -1..0, -x + ∫ x in 0..1, x :=
by
  sorry

end integral_abs_eq_split_l202_202349


namespace cos_B_in_triangle_l202_202152

-- Defining the problem conditions and what needs to be proved
theorem cos_B_in_triangle 
  (a b c A B C : ℝ)
  (h1 : b * Real.sin B - a * Real.sin A = (1/2) * a * Real.sin C)
  (h2 : (1/2) * a * c * Real.sin B = a^2 * Real.sin B)
  (h3 : IsTriangle a b c A B C): 
  Real.cos B = 3 / 4 :=
by
  sorry

end cos_B_in_triangle_l202_202152


namespace arithmetic_sequence_problem_l202_202203

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) (h_incr : ∀ n, a (n + 1) > a n) (h_prod : a 4 * a 5 = 13) : a 3 * a 6 = -275 := 
sorry

end arithmetic_sequence_problem_l202_202203


namespace diagonals_in_nine_sided_polygon_l202_202742

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202742


namespace solution_set_x_f_x_lt_0_l202_202205

/-- Definition of an odd function: f(-x) = -f(x) for all x -/
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Definition of an increasing function on (a, b) -/
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x → x < b → a < y → y < b → x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := sorry -- We don't need the explicit definition for this proof statement

theorem solution_set_x_f_x_lt_0 : 
  (is_odd_function f) ∧ 
  (is_increasing_on f 0 ∞) ∧ 
  (f (-3) = 0) → 
  {x : ℝ | x * f x < 0} = {x : ℝ | -3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3} :=
by
  sorry

end solution_set_x_f_x_lt_0_l202_202205


namespace diagonals_in_nine_sided_polygon_l202_202788

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202788


namespace number_of_tetrominoes_is_7_cannot_tile_4xn_rectangle_l202_202379

noncomputable section

def tetromino := fin 7 -- There are exactly 7 different tetrominoes

def can_tile_4xn_rectangle_with_all_tetrominoes (n : ℕ) : Prop :=
  ∀ (t : fin 7 → ℕ), (∑ i : fin 7, t i) = 4 * n → ∃ m : ℕ, ¬ can_tile_in_checkerboard (4 * n) (t i)


theorem number_of_tetrominoes_is_7 : tetromino = fin 7 :=
by
  sorry

theorem cannot_tile_4xn_rectangle (n : ℕ) : can_tile_4xn_rectangle_with_all_tetrominoes n = false :=
by
  sorry

end number_of_tetrominoes_is_7_cannot_tile_4xn_rectangle_l202_202379


namespace graph_symmetric_origin_l202_202273

def function_symmetry_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

theorem graph_symmetric_origin : function_symmetry_origin (λ x, x^5 + x^3) :=
sorry

end graph_symmetric_origin_l202_202273


namespace part1_part2_part3_l202_202896

def A (x : ℝ) : Prop := x^2 - x - 2 > 0
def B (x : ℝ) : Prop := 3 - |x| ≥ 0
def C (x : ℝ) (p : ℝ) : Prop := 4 * x + p < 0

theorem part1 : 
  {x : ℝ | A x} ∩ {x | B x} = {x | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} :=
sorry

theorem part2 : 
  {x : ℝ | A x} ∪ {x | B x} = set.univ :=
sorry

theorem part3 (p : ℝ) : 
  (∀ x, C x p → A x) → p ≥ 4 :=
sorry

end part1_part2_part3_l202_202896


namespace proof_x1plusx2_greater_2a_l202_202507

def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a / x - 2

variable (a : ℝ)
variable (x₁ x₂ : ℝ)
variable (h₁ : 0 < x₁) (h₂ : x₁ < x₂)
variable (hx1 : f x₁ a = 0) (hx2 : f x₂ a = 0)

theorem proof_x1plusx2_greater_2a :
  x₁ + x₂ > 2 * a :=
sorry

end proof_x1plusx2_greater_2a_l202_202507


namespace symmetric_circle_eq_l202_202079

theorem symmetric_circle_eq (C_1_eq : ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 1)
    (line_eq : ∀ x y : ℝ, x - y - 2 = 0) :
    ∀ x y : ℝ, (x - 1)^2 + y^2 = 1 :=
sorry

end symmetric_circle_eq_l202_202079


namespace sin_alpha_l202_202814

theorem sin_alpha (α : ℝ) (h1 : α ∈ set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.cos (α + Real.pi / 3) = -4 / 5) : 
  Real.sin α = (3 + 4 * Real.sqrt 3) / 10 := 
sorry

end sin_alpha_l202_202814


namespace cassandra_watch_time_l202_202401

-- Noncomputable because we use real number division and conversion
noncomputable def actual_time_when_watch_reads_8PM (t₀ t₁ : ℕ) (w₁ : ℕ) (w : ℕ) : ℕ :=
  let actual_time_passed := (w * t₁) / w₁ in
  t₀ + actual_time_passed

-- Define the constants and time equivalents
def noon := 12 * 60 -- 12:00 PM in minutes
def onePM := 13 * 60 -- 1:00 PM in minutes
def watchAtOnePM := 12 * 60 + 58 + 12 / 60 -- 12:58:12 in minutes
def watchAtEightPM := 20 * 60 -- 8:00 PM in minutes
def correctTime := 20 * 60 + 14 + 51 / 60 -- 8:14:51 PM in minutes

-- Main statement
theorem cassandra_watch_time :
  actual_time_when_watch_reads_8PM noon onePM watchAtOnePM watchAtEightPM = correctTime := by
  sorry

end cassandra_watch_time_l202_202401


namespace diagonals_in_nine_sided_polygon_l202_202796

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202796


namespace sum_of_first_10_common_elements_eq_13981000_l202_202450

def arithmetic_prog (n : ℕ) : ℕ := 4 + 3 * n
def geometric_prog (k : ℕ) : ℕ := 20 * 2 ^ k

theorem sum_of_first_10_common_elements_eq_13981000 :
  let common_elements : List ℕ := 
    [40, 160, 640, 2560, 10240, 40960, 163840, 655360, 2621440, 10485760]
  let sum_common_elements : ℕ := common_elements.sum
  sum_common_elements = 13981000 := by
  sorry

end sum_of_first_10_common_elements_eq_13981000_l202_202450


namespace find_acid_percentage_l202_202803

theorem find_acid_percentage (P : ℕ) (x : ℕ) (h1 : 4 + x = 20) 
  (h2 : x = 20 - 4) 
  (h3 : (P : ℝ)/100 * 4 + 0.75 * 16 = 0.72 * 20) : P = 60 :=
by
  sorry

end find_acid_percentage_l202_202803


namespace balloon_highest_elevation_l202_202855

theorem balloon_highest_elevation
  (time_rise1 time_rise2 time_descent : ℕ)
  (rate_rise rate_descent : ℕ)
  (t1 : time_rise1 = 15)
  (t2 : time_rise2 = 15)
  (t3 : time_descent = 10)
  (rr : rate_rise = 50)
  (rd : rate_descent = 10)
  : (time_rise1 * rate_rise - time_descent * rate_descent + time_rise2 * rate_rise) = 1400 := 
by
  sorry

end balloon_highest_elevation_l202_202855


namespace find_eccentricity_l202_202515

-- Define the conceptual elements for the problem first
variables {a b : ℝ} (ha : a > b) (hb : b > 0)

def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

def foci_distance : ℝ := sqrt (a^2 + b^2)
def eccentricity : ℝ := sqrt (1 + (b^2 / a^2))

-- Define the geometric points and conditions
variables {F1 F2 P Q : ℝ × ℝ}
variables 
  (hF1 : F1 = (-c, 0)) (hF2 : F2 = (c, 0)) -- assuming standard form of foci positions
  (hPQ : dist P F2 = foci_distance)
  (angle_condition : ∃ A, P.2 = sqrt (3 * F1.2))

-- Theorem stating the eccentricity based on the problem conditions
theorem find_eccentricity : 
  hyperbola P.1 P.2 -> 
  1 + (b^2 / a^2) = ((sqrt 3 + 1) / 2) :=
sorry

end find_eccentricity_l202_202515


namespace diagonals_in_nonagon_l202_202722

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202722


namespace number_of_distinct_rectangles_l202_202922

-- Define the side length of the square and the number of squares
def side_length : ℕ := 2
def num_squares : ℕ := 210

-- Define the total area of the squares
def total_area : ℕ := num_squares * (side_length * side_length)

-- Define the even divisors of the total area
def even_divisors : list ℕ := [2, 4, 6, 8, 10, 12, 14, 20, 24, 28, 30, 40, 42, 56, 60, 70, 84, 120, 140, 168, 210, 280, 420, 840]

-- Define what it means for a pair of divisors to form a valid rectangle
def valid_rectangle_pairs (a b : ℕ) : Prop :=
  a * b = total_area ∧ a ≠ b ∧ a ∈ even_divisors ∧ b ∈ even_divisors

-- Count the number of valid (unique by rotation) rectangle pairs
noncomputable def count_valid_rectangle_pairs : ℕ :=
  (even_divisors.product even_divisors).count (λ p, (p.1 <= p.2) ∧ valid_rectangle_pairs p.1 p.2)

-- The main theorem statement
theorem number_of_distinct_rectangles : count_valid_rectangle_pairs = 8 :=
sorry

end number_of_distinct_rectangles_l202_202922


namespace problem1_problem2_l202_202090

variable (α : ℝ) (tan_alpha_eq_three : Real.tan α = 3)

theorem problem1 : (4 * Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 11 / 14 :=
by sorry

theorem problem2 : Real.sin α * Real.cos α = 3 / 10 :=
by sorry

end problem1_problem2_l202_202090


namespace diagonals_in_nine_sided_polygon_l202_202774

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202774


namespace find_a_l202_202431

theorem find_a (r s a : ℚ) (h₁ : 2 * r * s = 18) (h₂ : s^2 = 16) (h₃ : a = r^2) : 
  a = 81 / 16 := 
sorry

end find_a_l202_202431


namespace regular_nine_sided_polygon_diagonals_l202_202674

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202674


namespace turnover_june_l202_202833

variable (TurnoverApril TurnoverMay : ℝ)

theorem turnover_june (h1 : TurnoverApril = 10) (h2 : TurnoverMay = 12) :
  TurnoverMay * (1 + (TurnoverMay - TurnoverApril) / TurnoverApril) = 14.4 := by
  sorry

end turnover_june_l202_202833


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202606

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202606


namespace chicken_buying_group_l202_202347

theorem chicken_buying_group (x : ℕ) 
  (h₁ : 9 * x - 11 > 0) 
  (h₂ : 6 * x + 16 > 0) 
  (h₃ : ∀ x, 9 * x - 11 = 6 * x + 16) : 
  x = ?m_1 := 
by 
  sorry

end chicken_buying_group_l202_202347


namespace y_intercept_of_line_l202_202321

theorem y_intercept_of_line : ∃ y : ℝ, 3 * 0 - 5 * y = 10 ∧ y = -2 :=
by
  use -2
  split
  { norm_num }
  { refl }

end y_intercept_of_line_l202_202321


namespace total_fruits_eaten_eq_54_l202_202871

def initial_fruits := (apples : nat) × (bananas : nat) × (oranges : nat) × (strawberries : nat) × (kiwis : nat)

def fruits_last_night : initial_fruits := (3, 1, 4, 2, 3)

def fruits_today (fruits_last_night : initial_fruits) : initial_fruits :=
  let (ln_apples, ln_bananas, ln_oranges, ln_strawberries, ln_kiwis) := fruits_last_night
  let today_apples := ln_apples + 4
  let today_bananas := ln_bananas * 10
  let today_oranges := today_apples * 2
  let today_strawberries := (ln_strawberries : rat) * 1.5 -- Moving to rationals to handle multiplication by 1.5
  let today_kiwis := today_bananas - 3
  (today_apples, today_bananas, nat.ofRat today_oranges, nat.ofRat today_strawberries, today_kiwis)

theorem total_fruits_eaten_eq_54 : (let ln := fruits_last_night
                                    let today := fruits_today ln
                                    let total_apples := ln.1 + today.1
                                    let total_bananas := ln.2 + today.2
                                    let total_oranges := ln.3 + today.3
                                    let total_strawberries := ln.4 + today.4
                                    let total_kiwis := ln.5 + today.5
                                    total_apples + total_bananas + total_oranges + total_strawberries + total_kiwis = 54) := sorry

end total_fruits_eaten_eq_54_l202_202871


namespace miles_run_l202_202231

theorem miles_run (B_sun J_sat S_sat : ℕ) (B_sat J_sun : ℕ) :
  B_sun = 10 →
  B_sun = B_sat + 4 →
  J_sat = 0 →
  J_sun = 2 * B_sun →
  S_sat = B_sat + J_sat →
  B_sun + J_sun + S_sat = 36 :=
begin
  intros h1 h2 h3 h4 h5,
  rw h1 at h4,
  rw h2 at h4,
  rw h1 at h5,
  rw h2 at h5,
  rw h3 at h5,
  linarith,
end

end miles_run_l202_202231


namespace rhombus_perimeter_l202_202943

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 15) (h3 : ∀ (a b : ℝ), (a^2 + b^2) = (d1/2)^2 + (d2/2)^2) : 
  4 * real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 25 :=
by
  -- This is a placeholder for where you would start your proof.
  sorry

end rhombus_perimeter_l202_202943


namespace class_student_count_l202_202941

-- Statement: Prove that under the given conditions, the number of students in the class is 19.
theorem class_student_count (n : ℕ) (avg_students_age : ℕ) (teacher_age : ℕ) (avg_with_teacher : ℕ):
  avg_students_age = 20 → 
  teacher_age = 40 → 
  avg_with_teacher = 21 → 
  21 * (n + 1) = 20 * n + 40 → 
  n = 19 := 
by 
  intros h1 h2 h3 h4 
  sorry

end class_student_count_l202_202941


namespace find_m_given_root_of_quadratic_l202_202094

theorem find_m_given_root_of_quadratic (m : ℝ) : (∃ x : ℝ, x = 3 ∧ x^2 - m * x - 6 = 0) → m = 1 := 
by
  sorry

end find_m_given_root_of_quadratic_l202_202094


namespace word_count_with_a_l202_202172

-- Defining the constants for the problem
def alphabet_size : ℕ := 26
def no_a_size : ℕ := 25

-- Calculating words that contain 'A' for lengths 1 to 5
def words_with_a (len : ℕ) : ℕ :=
  alphabet_size ^ len - no_a_size ^ len

-- The main theorem statement
theorem word_count_with_a : words_with_a 1 + words_with_a 2 + words_with_a 3 + words_with_a 4 + words_with_a 5 = 2186085 :=
by
  -- Calculations are established in the problem statement
  sorry

end word_count_with_a_l202_202172


namespace horizontal_asymptote_l202_202418

noncomputable def f (x : ℝ) : ℝ := (7 * x^2 - 4) / (4 * x^2 + 2 * x + 1)

theorem horizontal_asymptote : 
  Tendsto f atTop (𝓝 (7 / 4)) :=
sorry

end horizontal_asymptote_l202_202418


namespace hyperbola_asymptote_correct_l202_202108

variables (x y a b : ℝ) (k : ℝ) (e : ℝ)

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ ((x^2 / a^2) - (y^2 / b^2) = 1)

noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

theorem hyperbola_asymptote_correct :
  ∀ a b e k : ℝ,
    hyperbola a b x y ∧
    e = sqrt 5 / 2 ∧
    a = 2 * k ∧
    eccentricity a (sqrt 5 * k) = e ∧
    b = k
    → 
    asymptote a b x y ∧
    (b / a) = 1 / 2 :=
by
  sorry

end hyperbola_asymptote_correct_l202_202108


namespace harper_spending_l202_202525

noncomputable def daily_consumption : ℝ := 1 / 2 
noncomputable def total_days : ℕ := 240
noncomputable def bottles_per_case : ℕ := 24
noncomputable def price_per_case : ℝ := 12

theorem harper_spending :
  let total_bottles := total_days * daily_consumption in
  let cases_needed := total_bottles / bottles_per_case in
  let total_cost := cases_needed * price_per_case in
  total_cost = 60 :=
by
  sorry

end harper_spending_l202_202525


namespace increasing_function_l202_202821

def f (a : ℝ) : ℝ → ℝ :=
  fun x => if x ≥ -1 then (4-a) * x + a else -x^2 + 1

theorem increasing_function (a : ℝ) : (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (2 ≤ a ∧ a < 4) :=
by
  sorry

end increasing_function_l202_202821


namespace balloon_highest_elevation_l202_202857

theorem balloon_highest_elevation 
  (lift_rate : ℕ)
  (descend_rate : ℕ)
  (pull_time1 : ℕ)
  (release_time : ℕ)
  (pull_time2 : ℕ) :
  lift_rate = 50 →
  descend_rate = 10 →
  pull_time1 = 15 →
  release_time = 10 →
  pull_time2 = 15 →
  (lift_rate * pull_time1 - descend_rate * release_time + lift_rate * pull_time2) = 1400 :=
by
  sorry

end balloon_highest_elevation_l202_202857


namespace solution_set_inequality_l202_202487

theorem solution_set_inequality (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  {x : ℝ | (x - a) * (x - (1 / a)) < 0} = {x : ℝ | a < x ∧ x < 1 / a} := sorry

end solution_set_inequality_l202_202487


namespace nine_sided_polygon_diagonals_l202_202628

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202628


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202610

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202610


namespace part1_part2_l202_202485

section

-- Conditions
variables {f g : ℕ → ℝ} {λ : ℝ} (h_f1 : f 1 = 1) (h_f2 : f 2 = 1)
           (h_g1 : g 1 = 2) (h_g2 : g 2 = 2)
           (h_f : ∀ n ≥ 2, f(n + 1) = λ * f n)
           (h_g : ∀ n ≥ 2, g(n + 1) ≥ λ * g n)
           (h_λ_pos : λ > 0)

-- Part 1: ∀ n ∈ ℕ⁺, g(n+1) / f(n+1) ≥ g(n) / f(n)
theorem part1 (n : ℕ) (hn : n > 0) : g(n+1) / f(n+1) ≥ g(n) / f(n) :=
sorry

-- Part 2: ∀ n ∈ ℕ⁺, λ > 1 → ∑ i in range(n), (g(i+1) - f(i+1)) / (g(i+2) - f(i+2)) < λ / (λ - 1)
theorem part2 (n : ℕ) (hn : n > 0) (h_λ_gt1 : λ > 1) : 
  (∑ i in finset.range n, (g(i + 1) - f(i + 1)) / (g(i + 2) - f(i + 2))) < (λ / (λ - 1)) :=
sorry

end

end part1_part2_l202_202485


namespace reflection_matrix_solution_l202_202110

variable (a b : ℚ)

def matrix_R : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b], ![-(3/4 : ℚ), (4/5 : ℚ)]]

theorem reflection_matrix_solution (h : matrix_R a b ^ 2 = 1) :
    (a, b) = (-4/5, -3/5) := sorry

end reflection_matrix_solution_l202_202110


namespace area_enclosed_l202_202259

-- Define the exponential function and the boundaries
def exp (x : ℝ) := Real.exp x

theorem area_enclosed : (∫ x in 0..1, exp 1 - exp x) = 1 := by
  sorry

end area_enclosed_l202_202259


namespace diagonals_in_nine_sided_polygon_l202_202773

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202773


namespace clock_strikes_l202_202355

theorem clock_strikes (t n : ℕ) (h_t : 13 * t = 26) (h_n : 2 * n - 1 * t = 22) : n = 6 :=
by
  sorry

end clock_strikes_l202_202355


namespace number_of_diagonals_l202_202559

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202559


namespace diamonds_in_F25_l202_202269

noncomputable def number_of_diamonds : ℕ → ℕ
| 1 := 1
| 2 := 7  -- Figure it out based on conditions
| 3 := 15 -- Given in conditions
| n + 1 := number_of_diamonds n + 4 * n + 2

theorem diamonds_in_F25 : number_of_diamonds 25 = 1249 := by
  sorry

end diamonds_in_F25_l202_202269


namespace nine_sided_polygon_diagonals_l202_202623

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202623


namespace smallest_degree_polynomial_l202_202256

-- Definition of roots
def root1 := 3 - Real.sqrt 7
def root1_conjugate := 3 + Real.sqrt 7
def root2 := 5 + Real.sqrt 12
def root2_conjugate := 5 - Real.sqrt 12
def root3 := 16 - 2 * Real.sqrt 10
def root3_conjugate := 16 + 2 * Real.sqrt 10
def root4 := - Real.sqrt 3
def root4_conjugate := Real.sqrt 3

-- Statement to prove
theorem smallest_degree_polynomial :
  ∃ (p : Polynomial ℚ), 
  p.eval root1 = 0 ∧ p.eval root1_conjugate = 0 ∧
  p.eval root2 = 0 ∧ p.eval root2_conjugate = 0 ∧
  p.eval root3 = 0 ∧ p.eval root3_conjugate = 0 ∧
  p.eval root4 = 0 ∧ p.eval root4_conjugate = 0 ∧
  p.degree = 8 := 
sorry

end smallest_degree_polynomial_l202_202256


namespace find_m_l202_202878

/-- S is the set of positive integer divisors of 18^7,
    three numbers are chosen independently and at random 
    with replacement from the set S and labeled a1, a2, and a3 
    in the order they are chosen. The probability that a1 divides a2 
    and a2 divides a3 is m/n, where m and n are relatively prime 
    positive integers.
-/
theorem find_m : 
  let S := {d | ∃ (k₁ k₂ : ℕ), d = 2^k₁ * 3^k₂ ∧ k₁ ≤ 7 ∧ k₂ ≤ 14} in
  ∀ (a₁ a₂ a₃ : ℕ), 
  a₁ ∈ S → a₂ ∈ S → a₃ ∈ S → 
  (∃ m n : ℕ, (nat.coprime m n) ∧ (m * 36 = n * 17) ∧ (m = 17)) :=
begin
  sorry
end

end find_m_l202_202878


namespace B_subsetneq_A_l202_202114

def A : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }
def B : Set ℝ := { x : ℝ | 1 - x^2 > 0 }

theorem B_subsetneq_A : B ⊂ A :=
by
  sorry

end B_subsetneq_A_l202_202114


namespace regular_nine_sided_polygon_diagonals_l202_202675

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202675


namespace limit_value_l202_202404

noncomputable def limit_expression (x : ℝ) : ℝ :=
  (sin (3 * real.pi * x) / sin (real.pi * x)) ^ (sin (x - 2) ^ 2)

theorem limit_value :
  filter.tendsto limit_expression (nhds 2) (nhds 1) :=
begin
  sorry,
end

end limit_value_l202_202404


namespace plane_area_l202_202282

theorem plane_area (h : ∀ (x y : ℝ), (|x| + |4 - |y|| - 4)^2 ≤ 4): 
  ∃ (area : ℝ), area = 120 :=
sorry

end plane_area_l202_202282


namespace intersection_values_approx_l202_202022

theorem intersection_values_approx (x : ℝ) (y : ℝ) (h1 : y = 8 / (x^2 + 9)) (h2 : x - y = 1) :
  x ≈ 3.58 ∨ x ≈ -1.45 :=
by
  sorry

end intersection_values_approx_l202_202022


namespace construct_triangle_l202_202012

variable (h_a h_b h_c : ℝ)

noncomputable def triangle_exists_and_similar :=
  ∃ (a b c : ℝ), (a = h_b) ∧ (b = h_a) ∧ (c = h_a * h_b / h_c) ∧
  (∃ (area : ℝ), area = 1/2 * a * (h_a * h_c / h_b) ∧ area = 1/2 * b * (h_b * h_c / h_a) ∧ area = 1/2 * c * h_c)

theorem construct_triangle (h_a h_b h_c : ℝ) :
  ∃ a b c, a = h_b ∧ b = h_a ∧ c = h_a * h_b / h_c ∧
  ∃ area, area = 1/2 * a * (h_a * h_c / h_b) ∧ area = 1/2 * b * (h_b * h_c / h_a) ∧ area = 1/2 * c * h_c := 
  sorry

end construct_triangle_l202_202012


namespace centroid_path_area_l202_202219

theorem centroid_path_area {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (hAB : dist A B = 18)
  (hC_on_circle : ∀ P, P ≠ A ∧ P ≠ B → dist P (midpoint A B) = 9) :
  (let R := 9 in let r := R / 3 in let area := π * (r ^ 2) in (area.round : ℤ)) = 28 :=
by
  sorry

end centroid_path_area_l202_202219


namespace plane_through_midpoints_divides_equal_volume_l202_202491

theorem plane_through_midpoints_divides_equal_volume
  (A B C D E F : Point)
  (hE : midpoint A B E)
  (hF : midpoint C D F)
  (α : Plane) 
  (hα : passes_through α E ∧ passes_through α F) :
  divides_equal_volume α (tetrahedron A B C D) :=
sorry

end plane_through_midpoints_divides_equal_volume_l202_202491


namespace diagonals_in_regular_nine_sided_polygon_l202_202535

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202535


namespace distinct_ordered_pairs_count_l202_202131

theorem distinct_ordered_pairs_count :
  ∃ S : Finset (ℕ × ℕ), 
    (∀ p ∈ S, 1 ≤ p.1 ∧ 1 ≤ p.2 ∧ (1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6)) ∧
    S.card = 9 := 
by
  sorry

end distinct_ordered_pairs_count_l202_202131


namespace max_value_sincos_sum_l202_202048

theorem max_value_sincos_sum (x y z : ℝ) :
  (∀ x y z, (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5) :=
by sorry

end max_value_sincos_sum_l202_202048


namespace max_value_of_expression_l202_202044

theorem max_value_of_expression :
  ∃ (x y z : ℝ), 
    let expr := (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) in 
    expr = 4.5 ∧
    ∀ (x y z : ℝ), 
      (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 :=
begin
  sorry,
end

end max_value_of_expression_l202_202044


namespace regular_nine_sided_polygon_diagonals_l202_202752

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202752


namespace lcm_of_numbers_l202_202959

-- Definitions for the conditions
def hcf (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

variables (a b : ℕ)

-- Main theorem to prove
theorem lcm_of_numbers (h_hcf: hcf a b = 84) (h_ratio: a / b = 1 / 4) (h_max : max a b = 84) : lcm a b = 21 :=
by
  sorry

end lcm_of_numbers_l202_202959


namespace nine_sided_polygon_diagonals_count_l202_202688

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202688


namespace train_length_proof_l202_202985

-- Define the given conditions
def speed_faster_train_km_per_hr : ℝ := 75
def speed_slower_train_km_per_hr : ℝ := 60
def time_seconds : ℝ := 45
def km_to_m_conversion_factor : ℝ := 5 / 18

-- Convert speeds from km/hr to m/s
def speed_faster_train_m_per_s : ℝ := speed_faster_train_km_per_hr * km_to_m_conversion_factor
def speed_slower_train_m_per_s : ℝ := speed_slower_train_km_per_hr * km_to_m_conversion_factor

-- Calculate relative speed in m/s
def relative_speed := speed_faster_train_m_per_s - speed_slower_train_m_per_s

-- Calculate the combined length of both trains (relative speed * time)
def combined_length_of_trains := relative_speed * time_seconds

-- Calculate the length of each train
def length_of_each_train := combined_length_of_trains / 2

theorem train_length_proof : length_of_each_train = 93.75 := by
  -- Prove the theorem using predefined conditions and calculations
  sorry

end train_length_proof_l202_202985


namespace greatest_possible_sum_xy_l202_202993

noncomputable def greatest_possible_xy (x y : ℝ) :=
  x^2 + y^2 = 100 ∧ xy = 40 → x + y = 6 * Real.sqrt 5

theorem greatest_possible_sum_xy {x y : ℝ} (h1 : x^2 + y^2 = 100) (h2 : xy = 40) :
  x + y ≤ 6 * Real.sqrt 5 :=
sorry

end greatest_possible_sum_xy_l202_202993


namespace mrs_williams_points_l202_202391

theorem mrs_williams_points :
    ∃(p : ℕ), p = 50 ∧ 
              (57 + 49 + 57 + p) / 4 = 53.3 :=
by
  sorry

end mrs_williams_points_l202_202391


namespace infinite_points_midpoints_l202_202989

theorem infinite_points_midpoints (points : Set Point) (H : ∀ p ∈ points, ∃ q r ∈ points, p = (q + r) / 2) : ∃ S : Set Point, Infinite S :=
by 
  sorry

end infinite_points_midpoints_l202_202989


namespace diagonals_in_nine_sided_polygon_l202_202741

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202741


namespace solution_proof_l202_202287

noncomputable def solution_set (a : ℝ) (ha : a < 0) : set ℝ :=
  {x : ℝ | a * x^2 - (a + 2) * x + 2 ≥ 0}

theorem solution_proof (a : ℝ) (ha : a < 0) :
  solution_set a ha = set.Icc (2 / a) 1 :=
sorry

end solution_proof_l202_202287


namespace max_wind_power_speed_l202_202955

def sail_force (A S ρ v0 v: ℝ) : ℝ :=
  (A * S * ρ * (v0 - v)^2) / 2

def wind_power (A S ρ v0 v: ℝ) : ℝ :=
  (sail_force A S ρ v0 v) * v

theorem max_wind_power_speed (A ρ: ℝ) (v0: ℝ) (S: ℝ) (h: v0 = 4.8 ∧ S = 4) :
  ∃ v, (wind_power A S ρ v0 v) = max ((wind_power A S ρ v0 v)) ∧ v = 1.6 :=
begin
  sorry
end

end max_wind_power_speed_l202_202955


namespace regular_nine_sided_polygon_diagonals_l202_202754

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202754


namespace arithmetic_sequence_general_formula_value_of_a_p_plus_q_arithmetic_sequence_inequality_l202_202481

-- Part (1)
theorem arithmetic_sequence_general_formula
  (a1 : ℤ)
  (d : ℤ)
  (S3 : ℤ)
  (h1 : a1 = -1)
  (h2 : S3 = 12)
  (h3 : d ≠ 0) :
  (∀ n, ∃ an, an = a1 + (n - 1) * d) 
  → (∀ n, an = 5 * n - 6) :=
  sorry

-- Part (2)
theorem value_of_a_p_plus_q
  (a1 : ℤ)
  (a3 : ℤ)
  (a13 : ℤ)
  (p q : ℤ)
  (h1 : a1 = 1)
  (h2 : a1 * a13 = a3 * a3)
  (h3 : ∃ s t : ℤ, s ≠ t ∧ (2 * p - 1) % q = 0 ∧ (2 * q - 1) % p = 0)
  (h4 : p ≠ q) :
  a(p + q) = 15 :=
  sorry

-- Part (3)
theorem arithmetic_sequence_inequality
  (n : ℕ)
  (an : ℕ → ℤ)
  (f : ℤ → ℤ)
  (h1 : ∀ x, f x = (2^x - 1) / (2^x + 1)) 
  (h2 : n = 2022) :
  (∑ i in (finset.range n), an i) * (∑ i in (finset.range n), f(an i)) ≥ 0 :=
  sorry

end arithmetic_sequence_general_formula_value_of_a_p_plus_q_arithmetic_sequence_inequality_l202_202481


namespace find_k_even_function_find_m_minimum_zero_l202_202097

def f (x k : ℝ) : ℝ := Real.log (Real.exp (x * Real.log 4) + 1) / Real.log 4 + k * x
def h (f x m : ℝ) : ℝ := Real.exp ((f + 0.5 * x) * Real.log 4) + m * 2^x - 1

-- Given that the function f(x) is even, find k
theorem find_k_even_function (k : ℝ) (h_even : ∀ x : ℝ, f x k = f (-x) k) : k = -0.5 := by
  sorry

-- If the minimum of h(x) over [0, log_2(3)] is 0, find m
theorem find_m_minimum_zero :
  ∃ m : ℝ, (∀ x ∈ Set.Icc 0 (Real.log 3 / Real.log 2), h (f x (-0.5)) x m ≥ 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.log 3 / Real.log 2), h (f x (-0.5)) x m = 0) ↔ m = -1 :=
  by sorry

end find_k_even_function_find_m_minimum_zero_l202_202097


namespace triangle_area_18_l202_202440

theorem triangle_area_18 (A B C : Type*) [metric_space B]
  (angle_BAC : ∠ B A C = real.pi / 4)
  (right_angle : right_angleindicator A B C)
  (AC_length : dist A C = 6) :
  area_of_triangle A B C = 18 :=
sorry

end triangle_area_18_l202_202440


namespace regular_nonagon_diagonals_correct_l202_202705

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202705


namespace min_cells_to_mark_l202_202326

theorem min_cells_to_mark (n : ℕ) (h1 : n = 5) (h2 : (2 * n - 1) = 9) :
  (∃ m : ℕ, ∀ grid : fin 9 × fin 9 → bool,
  (∀ i : fin 9,
    (∃ j₁, grid ⟨i, j₁⟩) ∧
    (∃ j₂, grid ⟨i, j₂⟩) ∧
    (∃ j₃, grid ⟨i, j₃⟩) ∧
    (∃ j₄, grid ⟨i, j₄⟩) ∧
    (∃ j₅, grid ⟨i, j₅⟩)) ∧
  (∀ j : fin 9,
    (∃ i₁, grid ⟨i₁, j⟩) ∧
    (∃ i₂, grid ⟨i₂, j⟩) ∧
    (∃ i₃, grid ⟨i₃, j⟩) ∧
    (∃ i₄, grid ⟨i₄, j⟩) ∧
    (∃ i₅, grid ⟨i₅, j⟩)) ∧
  (∑ i j, grid ⟨i, j⟩) = m ∧ m = 16 :=
sorry

end min_cells_to_mark_l202_202326


namespace parallel_lines_or_perpendicular_l202_202840

theorem parallel_lines_or_perpendicular (L1 L2 M1 M2 : ℝ → ℝ → ℝ → Prop) 
  (hL1_distinct: L1 ≠ L2) (hM1_distinct: M1 ≠ M2)
  (hL_red: ∀ x ∈ L1, ∀ y ∈ L2, ∀ z ∈ L2, z ∈ L1 → x = y → y = z → x = z)
  (hM_blue: ∀ x ∈ M1, ∀ y ∈ M2, ∀ z ∈ M2, z ∈ M1 → x = y → y = z → x = z)
  (h_perpendicular: ∀ x ∈ L1, ∀ y ∈ L2, ∀ u ∈ M1, ∀ v ∈ M2, x ≠ u ∧ y ≠ v ∧ (x * u + y * v = 0)) :
  (L1 ∥ L2) ∨ (M1 ∥ M2) :=
sorry

end parallel_lines_or_perpendicular_l202_202840


namespace perfect_square_sequence_l202_202015

theorem perfect_square_sequence (x : ℕ → ℤ) (h₀ : x 0 = 0) (h₁ : x 1 = 3) 
  (h₂ : ∀ n, x (n + 1) + x (n - 1) = 4 * x n) : 
  ∀ n, ∃ k : ℤ, x (n + 1) * x (n - 1) + 9 = k^2 :=
by 
  sorry

end perfect_square_sequence_l202_202015


namespace abraham_initial_budget_l202_202382

-- Definitions based on conditions
def shower_gel_price := 4
def shower_gel_quantity := 4
def toothpaste_price := 3
def laundry_detergent_price := 11
def remaining_budget := 30

-- Calculations based on the conditions
def spent_on_shower_gels := shower_gel_quantity * shower_gel_price
def spent_on_toothpaste := toothpaste_price
def spent_on_laundry_detergent := laundry_detergent_price
def total_spent := spent_on_shower_gels + spent_on_toothpaste + spent_on_laundry_detergent

-- The theorem to prove
theorem abraham_initial_budget :
  (total_spent + remaining_budget) = 60 :=
by
  sorry

end abraham_initial_budget_l202_202382


namespace regular_nine_sided_polygon_diagonals_l202_202679

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202679


namespace sin_double_angle_l202_202490

theorem sin_double_angle (α : ℝ) (h1 : π/2 < α ∧ α < π) (h2 : tan α = -5/12) : sin (2 * α) = -120 / 169 := 
sorry

end sin_double_angle_l202_202490


namespace diagonals_in_regular_nine_sided_polygon_l202_202568

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202568


namespace inradius_triangle_l202_202280

theorem inradius_triangle (p A : ℝ) (h1 : p = 39) (h2 : A = 29.25) :
  ∃ r : ℝ, A = (1 / 2) * r * p ∧ r = 1.5 := by
  sorry

end inradius_triangle_l202_202280


namespace locus_is_line_l202_202961

theorem locus_is_line : 
  let A := { x := 1, y := 2 }
      (l : x*2 + y - 4 = 0)
  in from Locus A and l 
      proves it is a line :=
sorry

end locus_is_line_l202_202961


namespace smallest_M_ineq_l202_202447

theorem smallest_M_ineq (a b c : ℝ) :
  ∃ M : ℝ, (∀ a b c : ℝ, 
    abs (a * b * (a^2 - b^2) + a * c * (a^2 - c^2) + b * c * (b^2 - c^2)) ≤ 
    M * (a^2 + b^2 + c^2)^2) ∧ 
    M = 9 * Real.sqrt 2 / 32 :=
by
  use (9 : ℝ) * Real.sqrt 2 / 32
  intro a b c
  sorry

end smallest_M_ineq_l202_202447


namespace product_173_240_l202_202969

theorem product_173_240 :
  ∃ n : ℕ, n = 3460 ∧ n * 12 = 173 * 240 ∧ 173 * 240 = 41520 :=
by
  sorry

end product_173_240_l202_202969


namespace arrange_dogs_Alik_in_front_Punta_l202_202224

theorem arrange_dogs_Alik_in_front_Punta :
  let dogs := ["Alik", "Brok", "Muk", "Raf", "Punta"] in
  let arrangements := {p : List String // p ~ dogs ∧ p.nodup } in
  let alik_in_front_punta := λ (p : List String), p.indexOf "Alik" < p.indexOf "Punta" in
  arrangements.count alik_in_front_punta = 60 :=
by
  sorry

end arrange_dogs_Alik_in_front_Punta_l202_202224


namespace banana_permutations_l202_202802

theorem banana_permutations : 
  let n := 6 in
  let n_b := 1 in
  let n_n := 2 in
  let n_a := 3 in
  (n! / (n_b! * n_n! * n_a!)) = 60 := by
  unfold n n_b n_n n_a
  sorry

end banana_permutations_l202_202802


namespace nine_sided_polygon_diagonals_l202_202619

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202619


namespace gcd_lcm_product_correct_l202_202052

noncomputable def gcd_lcm_product : ℕ :=
  let a := 90
  let b := 135
  gcd a b * lcm a b

theorem gcd_lcm_product_correct : gcd_lcm_product = 12150 :=
  by
  sorry

end gcd_lcm_product_correct_l202_202052


namespace journey_time_calculation_l202_202364

theorem journey_time_calculation (dist totalDistance : ℝ) (rate1 rate2 : ℝ)
  (firstHalfDistance secondHalfDistance : ℝ) (time1 time2 totalTime : ℝ) :
  totalDistance = 224 ∧ rate1 = 21 ∧ rate2 = 24 ∧
  firstHalfDistance = totalDistance / 2 ∧ secondHalfDistance = totalDistance / 2 ∧
  time1 = firstHalfDistance / rate1 ∧ time2 = secondHalfDistance / rate2 ∧
  totalTime = time1 + time2 →
  totalTime = 10 :=
sorry

end journey_time_calculation_l202_202364


namespace length_of_XY_l202_202281

theorem length_of_XY (ABC : Triangle) (P Q X Y : Point) (h : perimeter ABC = 1) 
  (hA : ABC.inscribed P Q)
  (hB : midpoint_line_intersect_circumcircle ABC P Q X Y) :
  length_segment X Y = 1 / 2 := 
sorry

end length_of_XY_l202_202281


namespace niki_wins_triangle_game_l202_202226

theorem niki_wins_triangle_game (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a > 0) 
    (h3 : b > 0) (h4 : c > 0) : 
    let S := 4 * (a * b + a * c + b * c) - 1
    in (1 - S = 2 / 3) :=
begin
  -- Lean proof would go here, but skipped with sorry.
  sorry,
end

end niki_wins_triangle_game_l202_202226


namespace find_g50_l202_202957

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g50 (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x * y) = y * g x)
  (h1 : g 1 = 10) : g 50 = 50 * 10 :=
by
  -- The proof sketch here; the detailed proof is omitted
  sorry

end find_g50_l202_202957


namespace nine_sided_polygon_diagonals_l202_202625

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202625


namespace box_growth_factor_l202_202333

/-
Problem: When a large box in the shape of a cuboid measuring 6 centimeters (cm) wide,
4 centimeters (cm) long, and 1 centimeters (cm) high became larger into a volume of
30 centimeters (cm) wide, 20 centimeters (cm) long, and 5 centimeters (cm) high,
find how many times it has grown.
-/

def original_box_volume (w l h : ℕ) : ℕ := w * l * h
def larger_box_volume (w l h : ℕ) : ℕ := w * l * h

theorem box_growth_factor :
  original_box_volume 6 4 1 * 125 = larger_box_volume 30 20 5 :=
by
  -- Proof goes here
  sorry

end box_growth_factor_l202_202333


namespace sin_double_angle_l202_202466

theorem sin_double_angle :
  ∀ (x : ℝ), cos (π/4 - x) = 3/5 → sin (2 * x) = -7/25 :=
by
  intro x
  intro h
  sorry

end sin_double_angle_l202_202466


namespace solve_for_m_l202_202811

theorem solve_for_m (θ : ℝ) (m : ℝ) (h1 : sin θ = (m-3)/(m+5)) (h2 : cos θ = (4-2m)/(m+5)) : m = 0 ∨ m = 8 :=
by
  sorry

end solve_for_m_l202_202811


namespace diagonals_in_nine_sided_polygon_l202_202794

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202794


namespace diagonals_in_nonagon_l202_202729

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202729


namespace planes_parallel_or_coincide_l202_202498

-- Define the setup
variables {α β : Type} [plane α] [plane β]

-- The condition that there are infinitely many lines in plane α that are parallel to plane β
def infinitely_many_parallel_lines (α β : plane) : Prop :=
  ∃ (L : set line), infinite L ∧ ∀ l ∈ L, l ∈ α ∧ parallel l β

-- The theorem
theorem planes_parallel_or_coincide (h : infinitely_many_parallel_lines α β) : 
  (α ∥ β ∨ α = β) :=
sorry

end planes_parallel_or_coincide_l202_202498


namespace constant_term_binomial_expansion_l202_202264

theorem constant_term_binomial_expansion : 
  let T_r (r : ℕ) := (Nat.choose 6 r) * 2^(6 - r) * x^(6 - 3 * r / 2)
  in (∃ r : ℕ, 6 - 3 * r / 2 = 0 ∧ T_r r = 60) 
  :=
  sorry

end constant_term_binomial_expansion_l202_202264


namespace intersection_A_B_union_A_B_subset_C_A_l202_202898

def set_A : Set ℝ := { x | x^2 - x - 2 > 0 }
def set_B : Set ℝ := { x | 3 - abs x ≥ 0 }
def set_C (p : ℝ) : Set ℝ := { x | 4 * x + p < 0 }

theorem intersection_A_B : set_A ∩ set_B = { x | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3) } :=
sorry

theorem union_A_B : set_A ∪ set_B = Set.univ :=
sorry

theorem subset_C_A (p : ℝ) : set_C p ⊆ set_A → p ≥ 4 :=
sorry

end intersection_A_B_union_A_B_subset_C_A_l202_202898


namespace nine_sided_polygon_diagonals_l202_202636

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202636


namespace irreducible_fraction_in_segments_l202_202917

open Set

-- Defining the problem statement in Lean
theorem irreducible_fraction_in_segments (x : ℕ → ℝ) (n : ℕ) (h₀ : x 0 = 0) (h₁ : x n = 1) 
    (h₂ : ∀ i : ℕ, i < n → x i < x (i + 1)) : 
  ∃ m : ℕ, (∀ i : ℕ, i < n → ∃ l : ℕ, is_coprime l m ∧ l < m ∧ ((l : ℝ) / m) ∈ Ioo (x i) (x (i + 1))) ∧ 
  ¬ nat.prime m :=
sorry

end irreducible_fraction_in_segments_l202_202917


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202599

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202599


namespace total_lobster_pounds_l202_202121

variable (lobster_other_harbor1 : ℕ)
variable (lobster_other_harbor2 : ℕ)
variable (lobster_hooper_bay : ℕ)

-- Conditions
axiom h_eq : lobster_hooper_bay = 2 * (lobster_other_harbor1 + lobster_other_harbor2)
axiom other_harbors_eq : lobster_other_harbor1 = 80 ∧ lobster_other_harbor2 = 80

-- Proof statement
theorem total_lobster_pounds : 
  lobster_other_harbor1 + lobster_other_harbor2 + lobster_hooper_bay = 480 :=
by
  sorry

end total_lobster_pounds_l202_202121


namespace square_perimeter_eq_area_perimeter_16_l202_202261

theorem square_perimeter_eq_area_perimeter_16 (s : ℕ) (h : s^2 = 4 * s) : 4 * s = 16 := by
  sorry

end square_perimeter_eq_area_perimeter_16_l202_202261


namespace nine_sided_polygon_diagonals_l202_202639

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202639


namespace root_sum_reciprocal_l202_202212

theorem root_sum_reciprocal (p q r s : ℂ)
  (h1 : (∀ x : ℂ, x^4 - 6*x^3 + 11*x^2 - 6*x + 3 = 0 → x = p ∨ x = q ∨ x = r ∨ x = s))
  (h2 : p*q*r*s = 3) 
  (h3 : p*q + p*r + p*s + q*r + q*s + r*s = 11) :
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s)) = 11/3 :=
by
  sorry

end root_sum_reciprocal_l202_202212


namespace regular_nonagon_diagonals_correct_l202_202715

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202715


namespace max_k_subset_l202_202886

theorem max_k_subset :
  ∃ (k : ℕ), k ≤ 16 ∧ (∀ A : Finset ℕ, A.card = k → A ⊆ {n | n ∈ Finset.range 17 \ {-1}} → 
  (∀ B ⊆ A, ∃ S : Finset (ℕ → ℕ), S.card = 2^k - 1 ∧ 
  (∀ s1 s2 : ℕ, s1 ∈ S → s2 ∈ S → s1 ≠ s2)) ∧ k = 5 
   ) := 
   sorry

end max_k_subset_l202_202886


namespace diagonals_in_nine_sided_polygon_l202_202737

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202737


namespace diagonals_in_nine_sided_polygon_l202_202745

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202745


namespace quadrant_of_z_l202_202471

noncomputable def complex_quadrant : Prop :=
  let z := (2 + 3 * complex.i) / (1 + 2 * complex.i)
  (complex.re z > 0) ∧ (complex.im z < 0)

theorem quadrant_of_z : complex_quadrant :=
  sorry

end quadrant_of_z_l202_202471


namespace num_groups_l202_202974

theorem num_groups (boys girls group_size : ℕ) (h_boys : boys = 9) (h_girls : girls = 12) (h_group_size : group_size = 3) : 
  (boys + girls) / group_size = 7 :=
by 
  rw [h_boys, h_girls, h_group_size]
  norm_num
  sorry

end num_groups_l202_202974


namespace johns_hats_cost_l202_202863

theorem johns_hats_cost 
  (weeks : ℕ)
  (days_in_week : ℕ)
  (cost_per_hat : ℕ) 
  (h : weeks = 2 ∧ days_in_week = 7 ∧ cost_per_hat = 50) 
  : (weeks * days_in_week * cost_per_hat) = 700 :=
by
  sorry

end johns_hats_cost_l202_202863


namespace problem_1_problem_2_l202_202304

noncomputable def probability_exactly_one_success (pA pB pC : ℝ) : ℝ :=
  pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC

noncomputable def min_institutes_B (pB : ℝ) : ℕ :=
  let n := (-2) / real.log (2 / 3) in
  nat_ceil n

theorem problem_1
  (pA : ℝ) (pB : ℝ) (pC : ℝ)
  (hA : pA = 1 / 2) (hB : pB = 1 / 3) (hC : pC = 1 / 4) :
  probability_exactly_one_success pA pB pC = 11 / 24 :=
by
  sorry

theorem problem_2
  (pB : ℝ)
  (hB : pB = 1 / 3) :
  min_institutes_B pB = 12 :=
by
  sorry

end problem_1_problem_2_l202_202304


namespace diagonals_in_nonagon_l202_202717

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202717


namespace diagonals_in_regular_nine_sided_polygon_l202_202580

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202580


namespace nine_sided_polygon_diagonals_l202_202649

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202649


namespace proof_problem_l202_202027

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 1 = 1) ∧ 
  (f (-1) = -1) ∧ 
  (∀ x, 0 < x ∧ x < 1 → f x ≤ f 0) ∧ 
  (∀ x y, f (x + y) ≥ f x + f y) ∧
  (∀ x y, f (x + y) ≤ f x + f y + 1)

noncomputable def correct_function : ℝ → ℝ := Int.floor

theorem proof_problem : satisfies_conditions correct_function := 
  sorry

end proof_problem_l202_202027


namespace fixed_monthly_fee_december_l202_202402

theorem fixed_monthly_fee_december (x y : ℝ) 
    (h1 : x + y = 15.00) 
    (h2 : x + 2 + 3 * y = 25.40) : 
    x = 10.80 :=
by
  sorry

end fixed_monthly_fee_december_l202_202402


namespace jet_flight_time_l202_202284

theorem jet_flight_time :
  let radius := 5000 -- radius of Earth in miles
  let speed := 600 -- speed of jet in miles/hour
  let tailwind := 50 -- tailwind speed increment in miles/hour
  let circumference := 2 * Real.pi * radius -- circumference of Earth
  let effective_speed := speed + tailwind -- effective speed of jet
  let time := circumference / effective_speed -- time to circumnavigate
  Int.floor (time) = 48 := 
by
  let radius := 5000
  let speed := 600
  let tailwind := 50
  let circumference := 2 * Real.pi * radius
  let effective_speed := speed + tailwind
  let time := circumference / effective_speed
  have h : floor time = 48 := sorry
  exact h

end jet_flight_time_l202_202284


namespace min_value_term_l202_202112

def seq (n : ℕ) : ℝ := 2 * n^2 - 10 * n + 3

theorem min_value_term :
  ∃ n : ℕ, (n = 2 ∨ n = 3) ∧ ∀ m : ℕ, m > 0 → seq n ≤ seq m :=
by
  sorry

end min_value_term_l202_202112


namespace distance_QR_l202_202243

theorem distance_QR (D E F : Point) (Q R : Point)
  (h_triangle : is_right_triangle D E F)
  (h_DE : distance D E = 5)
  (h_EF : distance E F = 12)
  (h_DF : distance D F = 13)
  (h_circleQ : is_tangent_circle Q E F E D)
  (h_circleR : is_tangent_circle R D F D E) :
  distance Q R = 25 / 12 :=
sorry

end distance_QR_l202_202243


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202605

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202605


namespace correct_calculation_only_A_l202_202334

-- Definitions of the expressions
def exprA (a : ℝ) : Prop := 3 * a + 2 * a = 5 * a
def exprB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def exprC (a : ℝ) : Prop := 3 * a * 2 * a = 6 * a
def exprD (a : ℝ) : Prop := 3 * a / (2 * a) = (3 / 2) * a

-- The theorem stating that only exprA is correct
theorem correct_calculation_only_A (a : ℝ) :
  exprA a ∧ ¬exprB a ∧ ¬exprC a ∧ ¬exprD a :=
by
  sorry

end correct_calculation_only_A_l202_202334


namespace complex_magnitude_l202_202095

theorem complex_magnitude (z : ℂ) (h : z * (2 - 4 * Complex.I) = 1 + 3 * Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 :=
by
  sorry

end complex_magnitude_l202_202095


namespace distinct_three_digit_numbers_count_l202_202132

theorem distinct_three_digit_numbers_count : 
  let digits := {1, 2, 3, 4, 5}
  in 5^3 = 125 := 
by 
  sorry

end distinct_three_digit_numbers_count_l202_202132


namespace greatest_possible_sum_xy_l202_202994

noncomputable def greatest_possible_xy (x y : ℝ) :=
  x^2 + y^2 = 100 ∧ xy = 40 → x + y = 6 * Real.sqrt 5

theorem greatest_possible_sum_xy {x y : ℝ} (h1 : x^2 + y^2 = 100) (h2 : xy = 40) :
  x + y ≤ 6 * Real.sqrt 5 :=
sorry

end greatest_possible_sum_xy_l202_202994


namespace cab_time_l202_202354

theorem cab_time (d t : ℝ) (v : ℝ := d / t)
    (v1 : ℝ := (5 / 6) * v)
    (t1 : ℝ := d / v1)
    (v2 : ℝ := (2 / 3) * v)
    (t2 : ℝ := d / v2)
    (T : ℝ := t1 + t2)
    (delay : ℝ := 5) :
    let total_time := 2 * t + delay
    t * d ≠ 0 → T = total_time → t = 50 / 7 := by
    sorry

end cab_time_l202_202354


namespace maximum_good_rows_l202_202939

def labeled_unit_squares : ℕ := 40
def table_size : ℕ := 9
def minimum_labeled_squares_in_good_row : ℕ := 5

theorem maximum_good_rows (table_size = 9) (labeled_unit_squares = 40) (minimum_labeled_squares_in_good_row = 5) : 
  ∃ (good_rows : ℕ), good_rows = 8 := sorry

end maximum_good_rows_l202_202939


namespace verify_total_amount_l202_202936

noncomputable def total_withdrawable_amount (a r : ℝ) : ℝ :=
  a / r * ((1 + r) ^ 5 - (1 + r))

theorem verify_total_amount (a r : ℝ) (h_r_nonzero : r ≠ 0) :
  total_withdrawable_amount a r = a / r * ((1 + r)^5 - (1 + r)) :=
by
  sorry

end verify_total_amount_l202_202936


namespace alice_sales_surplus_l202_202384

-- Define the constants
def adidas_cost : ℕ := 45
def nike_cost : ℕ := 60
def reebok_cost : ℕ := 35
def quota : ℕ := 1000

-- Define the quantities sold
def adidas_sold : ℕ := 6
def nike_sold : ℕ := 8
def reebok_sold : ℕ := 9

-- Calculate total sales
def total_sales : ℕ := adidas_sold * adidas_cost + nike_sold * nike_cost + reebok_sold * reebok_cost

-- Prove that Alice's total sales minus her quota is 65
theorem alice_sales_surplus : total_sales - quota = 65 := by
  -- Calculation is omitted here. Here is the mathematical fact to prove:
  sorry

end alice_sales_surplus_l202_202384


namespace min_cells_to_mark_l202_202327

theorem min_cells_to_mark (n : ℕ) (h1 : n = 5) (h2 : (2 * n - 1) = 9) :
  (∃ m : ℕ, ∀ grid : fin 9 × fin 9 → bool,
  (∀ i : fin 9,
    (∃ j₁, grid ⟨i, j₁⟩) ∧
    (∃ j₂, grid ⟨i, j₂⟩) ∧
    (∃ j₃, grid ⟨i, j₃⟩) ∧
    (∃ j₄, grid ⟨i, j₄⟩) ∧
    (∃ j₅, grid ⟨i, j₅⟩)) ∧
  (∀ j : fin 9,
    (∃ i₁, grid ⟨i₁, j⟩) ∧
    (∃ i₂, grid ⟨i₂, j⟩) ∧
    (∃ i₃, grid ⟨i₃, j⟩) ∧
    (∃ i₄, grid ⟨i₄, j⟩) ∧
    (∃ i₅, grid ⟨i₅, j⟩)) ∧
  (∑ i j, grid ⟨i, j⟩) = m ∧ m = 16 :=
sorry

end min_cells_to_mark_l202_202327


namespace diagonals_in_nine_sided_polygon_l202_202750

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202750


namespace omega_bound_l202_202502

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - Real.sin (ω * x)

theorem omega_bound (ω : ℝ) (h₁ : ω > 0)
  (h₂ : ∀ x : ℝ, -π / 2 < x ∧ x < π / 2 → (f ω x) ≤ (f ω (-π / 2))) :
  ω ≤ 1 / 2 :=
sorry

end omega_bound_l202_202502


namespace find_f_f_2sqrt2_l202_202105

def f : ℝ → ℝ :=
λ x, if x < 1 then 3 * x + 5 else log (1/2) x - 1

theorem find_f_f_2sqrt2 : f (f (2 * real.sqrt 2)) = -5 / 2 := by
  sorry

end find_f_f_2sqrt2_l202_202105


namespace diagonals_in_nine_sided_polygon_l202_202783

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202783


namespace total_teams_l202_202964

theorem total_teams (m n : ℕ) (hmn : m > n) : 
  (m - n) + 1 = m - n + 1 := 
by sorry

end total_teams_l202_202964


namespace car_count_l202_202968

theorem car_count (x y : ℕ) (h1 : x + y = 36) (h2 : 6 * x + 4 * y = 176) :
  x = 16 ∧ y = 20 :=
by
  sorry

end car_count_l202_202968


namespace determine_n_l202_202412

theorem determine_n 
    (n : ℕ) (h2 : n ≥ 2) 
    (a : ℕ) (ha_div_n : a ∣ n) 
    (ha_min : ∀ d : ℕ, d ∣ n → d > 1 → d ≥ a) 
    (b : ℕ) (hb_div_n : b ∣ n)
    (h_eq : n = a^2 + b^2) : 
    n = 8 ∨ n = 20 :=
sorry

end determine_n_l202_202412


namespace dry_fruits_weight_l202_202060

variable (fresh_grapes_total_weight fresh_apples_total_weight : ℝ)
variable (fresh_grapes_water_percentage fresh_apples_water_percentage : ℝ)

-- Conditions
def fresh_grapes_non_water_content : ℝ :=
  (1 - fresh_grapes_water_percentage) * fresh_grapes_total_weight

def fresh_apples_non_water_content : ℝ :=
  (1 - fresh_apples_water_percentage) * fresh_apples_total_weight

-- Constants
def fresh_grapes_weight := 400
def fresh_apples_weight := 300
def fresh_grapes_water_pct := 0.65
def fresh_apples_water_pct := 0.84

-- The proof statement
theorem dry_fruits_weight :
  fresh_grapes_non_water_content fresh_grapes_weight fresh_grapes_water_pct +
  fresh_apples_non_water_content fresh_apples_weight fresh_apples_water_pct = 188 :=
by
  -- Proof to be added
  sorry

end dry_fruits_weight_l202_202060


namespace range_of_a_l202_202513

noncomputable def f (a x : ℝ) : ℝ := real.exp x + a * real.log (1 / (a * x + a)) - a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x > 0) → (0 < a ∧ a < 1) :=
begin
  sorry
end

end range_of_a_l202_202513


namespace diagonals_in_nine_sided_polygon_l202_202781

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202781


namespace correct_number_of_propositions_l202_202091

-- Definitions for lines and planes
def is_line (l : Type) := 
  ∃ p1 p2 : Type, p1 ≠ p2 ∧ (p1, p2 ∈ l)

def is_plane (p : Type) := 
  ∃ p1 p2 p3 : Type, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ (p1, p2, p3 ∈ p)

-- Distinctness of lines and planes
variables {a b c : Type}
variables {α β : Type}

-- Conditions
axiom distinct_lines : a ≠ b ∧ b ≠ c ∧ a ≠ c
axiom distinct_planes : α ≠ β

-- Propositions
def prop1 := (a ∥ b ∧ b ∥ α) → (a ∥ α)
def prop2 := (a ⊂ α ∧ b ⊂ α ∧ a ∥ β ∧ b ∥ β) → (α ∥ β)
def prop3 := (a ⊥ α ∧ a ∥ β) → (α ⊥ β)
def prop4 := (a ⊥ α ∧ b ∥ α) → (a ⊥ b)

-- Correct number of propositions
def num_correct_propositions := 2

theorem correct_number_of_propositions : 
  let p1 := (a ∥ b ∧ b ∥ α) → (a ∥ α),
      p2 := (a ⊂ α ∧ b ⊂ α ∧ a ∥ β ∧ b ∥ β) → (α ∥ β),
      p3 := (a ⊥ α ∧ a ∥ β) → (α ⊥ β),
      p4 := (a ⊥ α ∧ b ∥ α) → (a ⊥ b)
  in 
  (p3 ∧ p4) = 2 := 
sorry

end correct_number_of_propositions_l202_202091


namespace approximate_root_in_interval_l202_202147

def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x - 2

theorem approximate_root_in_interval :
  f(1) = -2 ∧
  f(1.5) = 0.65 ∧
  f(1.25) = -0.984 ∧
  f(1.375) = -0.26 ∧
  f(1.4375) = 0.162 ∧
  f(1.40625) = -0.054 →
  1.375 < 1.4 ∧ 1.4 < 1.4375 ∧
  ∃ x, 1.375 < x ∧ x < 1.4375 ∧ f x = 0 :=
sorry

end approximate_root_in_interval_l202_202147


namespace sum_first_10_terms_of_arithmetic_sequence_l202_202480

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_10_terms_of_arithmetic_sequence (h1 : a 1 = -4) (h2 : a 4 + a 6 = 16) (d : ℝ) :
  is_arithmetic_sequence a d →
  (∑ i in Finset.range 10, a (i + 1)) = 95 :=
by
  sorry

end sum_first_10_terms_of_arithmetic_sequence_l202_202480


namespace Harper_spends_60_dollars_l202_202527

-- Define the conditions
variable (dailyConsumption : ℚ := 1/2)                -- Harper drinks 1/2 bottle of water per day.
variable (bottlesPerCase : ℤ := 24)                   -- Each case contains 24 bottles.
variable (caseCost : ℚ := 12)                        -- A case costs $12.00.
variable (totalDays : ℤ := 240)                      -- Harper wants to buy enough cases for 240 days.

-- Theorem statement
theorem Harper_spends_60_dollars :
  let durationCase : ℤ := bottlesPerCase / (dailyConsumption.natAbs : ℤ)
  let casesNeeded : ℤ := totalDays / durationCase
  let totalCost : ℚ := casesNeeded * caseCost
  totalCost = 60 := by
  sorry

end Harper_spends_60_dollars_l202_202527


namespace diagonals_in_regular_nine_sided_polygon_l202_202532

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202532


namespace find_set_M_l202_202216

variable (U : Set ℕ) (M : Set ℕ)

def isUniversalSet : Prop := U = {1, 2, 3, 4, 5, 6}
def isComplement : Prop := U \ M = {1, 2, 4}

theorem find_set_M (hU : isUniversalSet U) (hC : isComplement U M) : M = {3, 5, 6} :=
  sorry

end find_set_M_l202_202216


namespace one_value_for_ffx_eq_zero_l202_202143

def f (x : ℝ) : ℝ :=
  if x > -3 then x ^ 2 - 9 else 2 * x + 6

theorem one_value_for_ffx_eq_zero :
  ∃! x : ℝ, f (f x) = 0 :=
sorry

end one_value_for_ffx_eq_zero_l202_202143


namespace intersection_C1_C2_minimum_distance_to_C2_l202_202357

def C1_parametric (α : ℝ) : ℝ × ℝ :=
  (1 + cos α, sin α ^ 2 - 9 / 4)

def C2_cartesian (x y : ℝ) : Prop :=
  x + y + 1 = 0

def C3_cartesian (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + y ^ 2 = 1

theorem intersection_C1_C2 :
  (∃ α x y, C1_parametric α = (x, y) ∧ C2_cartesian x y) →
  (∃ x y, x = 1 / 2 ∧ y = -3 / 2) :=
sorry

theorem minimum_distance_to_C2 :
  (∃ A B : ℝ × ℝ, C2_cartesian A.1 A.2 ∧ C3_cartesian B.1 B.2 ∧ ∀ d : ℝ, d = dist A B) →
  (∃ d, d = sqrt 2 - 1) :=
sorry

end intersection_C1_C2_minimum_distance_to_C2_l202_202357


namespace nine_sided_polygon_diagonals_l202_202629

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202629


namespace matrix_determinant_zero_l202_202005

noncomputable def matrix_det_is_zero : Prop :=
  let M := ![
    [Real.sin 1, Real.sin 2, Real.sin 3],
    [Real.sin 4, Real.sin 5, Real.sin 6],
    [Real.sin 7, Real.sin 8, Real.sin 9]
  ]
  in Matrix.det M = 0

theorem matrix_determinant_zero : matrix_det_is_zero := by
  sorry

end matrix_determinant_zero_l202_202005


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202612

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202612


namespace wrong_number_is_26_l202_202262

theorem wrong_number_is_26 (initial_avg correct_avg : ℝ) (n wrong_num : ℝ) :
  n = 10 → initial_avg = 5 → correct_avg = 6 → wrong_num = 36 →
  (let init_sum := n * initial_avg in
   let correct_sum := n * correct_avg in
   let diff := correct_sum - init_sum in
   let x := wrong_num - diff in
   x = 26) :=
by
  intros n_eq i_avg_eq c_avg_eq w_num_eq
  rw [n_eq, i_avg_eq, c_avg_eq, w_num_eq]
  let init_sum := 10 * 5
  let correct_sum := 10 * 6
  let diff := correct_sum - init_sum
  let x := 36 - diff
  have h1 : init_sum = 50 := rfl
  have h2 : correct_sum = 60 := rfl
  have h3 : diff = 10 := rfl
  have h4 : x = 26 := rfl
  exact h4


end wrong_number_is_26_l202_202262


namespace arithmetic_sequence_fifth_term_l202_202496

noncomputable def fifth_term_of_arithmetic_sequence (x y : ℝ)
  (h1 : 2 * x + y = 2 * x + y)
  (h2 : 2 * x - y = 2 * x + y - 2 * y)
  (h3 : 2 * x * y = 2 * x - 2 * y - 2 * y)
  (h4 : 2 * x / y = 2 * x * y - 5 * y^2 - 2 * y)
  : ℝ :=
(2 * x / y) - 2 * y

theorem arithmetic_sequence_fifth_term (x y : ℝ)
  (h1 : 2 * x + y = 2 * x + y)
  (h2 : 2 * x - y = 2 * x + y - 2 * y)
  (h3 : 2 * x * y = 2 * x - 2 * y - 2 * y)
  (h4 : 2 * x / y = 2 * x * y - 5 * y^2 - 2 * y)
  : fifth_term_of_arithmetic_sequence x y h1 h2 h3 h4 = -77 / 10 :=
sorry

end arithmetic_sequence_fifth_term_l202_202496


namespace max_p_real_roots_l202_202011

noncomputable def quadratic_prod_max (a b p : ℝ) : Prop :=
  let Δ := b * b - 4 * a * p in
  a = 5 ∧ b = -6 ∧ Δ ≥ 0 ∧ (p / a) ≤ (1.8 / 5)

theorem max_p_real_roots : ∃ p : ℝ, quadratic_prod_max 5 (-6) p ∧ p = 1.8 :=
by {
  use 1.8,
  unfold quadratic_prod_max,
  split,
  { exact rfl },
  split,
  { exact rfl },
  split,
  { linarith },
  { linarith }
}

end max_p_real_roots_l202_202011


namespace num_groups_l202_202975

theorem num_groups (boys girls group_size : ℕ) (h_boys : boys = 9) (h_girls : girls = 12) (h_group_size : group_size = 3) : 
  (boys + girls) / group_size = 7 :=
by 
  rw [h_boys, h_girls, h_group_size]
  norm_num
  sorry

end num_groups_l202_202975


namespace regular_nonagon_diagonals_correct_l202_202707

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202707


namespace cone_lateral_surface_area_l202_202096

theorem cone_lateral_surface_area (l r h : ℝ) 
  (h1 : isosceles_right_triangle (r) (h) (l)) 
  (h2 : h = r) 
  (h3 : r = (Math.sqrt 2 * l) / 2)
  (h_volume : (1 / 3) * Real.pi * r^2 * h = (8 / 3) * Real.pi) :
  (Real.pi * r * l = 4 * Real.sqrt 2 * Real.pi) := 
begin
  -- The proof is skipped by using sorry
  sorry
end

end cone_lateral_surface_area_l202_202096


namespace vector_sum_theorem_l202_202071

noncomputable def vector_sum_zero (n : ℕ) (h : n ≥ 1) : Prop :=
  let vertices := finset.range (2 * n)
  let edges := { (i, j) | i < j ∧ j - i ≠ n }  -- i.e., considering diagonals and edges
  let arrow_placement : Π (e : (ℤ × ℤ)), (ℤ × ℤ)
  in 
    let vector_sum := ∑ e in edges, arrow_placement e
    ⟨vector_sum = (0, 0)⟩

theorem vector_sum_theorem (n : ℕ) (h : n ≥ 1) :
  vector_sum_zero n h :=
sorry

end vector_sum_theorem_l202_202071


namespace parallelogram_area_approximation_l202_202439

noncomputable def area_of_parallelogram (a b slant_height: ℝ) (angle: ℝ) : ℝ :=
  let h := b * Real.sin angle in
  a * h

theorem parallelogram_area_approximation :
  area_of_parallelogram 18 24 26 (120 * Real.pi / 180) ≈ 374.112 :=
by
  have h := 24 * Real.sin (120 * Real.pi / 180)
  -- Simplifying h to 24 * (√3 / 2)
  have h_simplified : h = 12 * Real.sqrt 3 := by sorry
  -- Calculate the area
  have area := 18 * h_simplified
  -- Approximate the area to 374.112
  show area ≈ 374.112 from by sorry

end parallelogram_area_approximation_l202_202439


namespace isosceles_right_triangle_perpendiculars_equal_distances_l202_202912

theorem isosceles_right_triangle_perpendiculars_equal_distances
  (A B C D E P Q : ℝ → ℝ)
  (hABC : isosceles_right_triangle A B C)
  (hD : D ∈ segment A C)
  (hE : E ∈ segment B C)
  (hCDCE : dist C D = dist C E)
  (hP : P = point_intersection C (perpendicular C (line_through A E)) A B)
  (hQ : Q = point_intersection D (perpendicular D (line_through A E)) A B) :
  dist B P = dist P Q :=
sorry

end isosceles_right_triangle_perpendiculars_equal_distances_l202_202912


namespace question_I_question_II_question_III_l202_202107

open Real

noncomputable def f (x : ℝ) (m : ℝ) := 2 * log x + 1 / x - m * x

theorem question_I (x : ℝ) (h1 : x = 1) (m : ℝ) (h2 : m = -1) :
  ∃ (a b : ℝ), a * x - b = 0 := by
  sorry

theorem question_II (m : ℝ) :
  (∀ x ∈ Ioo 0 ∞, deriv (f x m) x ≤ 0) → m ≥ 1 := by
  sorry

theorem question_III (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  (ln b - ln a) / (b - a) < 1 / sqrt (a * b) := by
  sorry

end question_I_question_II_question_III_l202_202107


namespace regular_nine_sided_polygon_diagonals_l202_202669

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202669


namespace total_area_swept_l202_202310

theorem total_area_swept (L : ℝ) : 
  let A := (2/3 * Real.pi + (Real.sqrt 3 / 4)) * L^2 in
  A ≈ 2.527 * L^2 := 
by
  -- Sorry, skipping the proof
  sorry

end total_area_swept_l202_202310


namespace sachin_is_younger_by_8_years_l202_202930

variable (S R : ℕ)

-- Conditions
axiom age_of_sachin : S = 28
axiom ratio_of_ages : S * 9 = R * 7

-- Goal
theorem sachin_is_younger_by_8_years (S R : ℕ) (h1 : S = 28) (h2 : S * 9 = R * 7) : R - S = 8 :=
by
  sorry

end sachin_is_younger_by_8_years_l202_202930


namespace find_a_l202_202486

def A := {x : ℝ | 1 < x ∧ x < 7}
def B (a : ℝ) := {x : ℝ | a + 1 < x ∧ x < 2a + 5}
def Intersect (a : ℝ) := {x : ℝ | 3 < x ∧ x < 7}

theorem find_a (a : ℝ) (h : A ∩ B a = Intersect a) : a = 2 :=
by
  sorry

end find_a_l202_202486


namespace reciprocal_real_roots_l202_202057

theorem reciprocal_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 * x2 = 1 ∧ x1 + x2 = 2 * (m + 2)) ∧ 
  (x1^2 - 2 * (m + 2) * x1 + (m^2 - 4) = 0) → m = Real.sqrt 5 := 
sorry

end reciprocal_real_roots_l202_202057


namespace sum_of_digits_M_l202_202171

def digit_sum (n : ℕ) : ℕ := sorry -- Define the sum of the digits function

def valid_digits (n : ℕ) : Prop :=
  ∀ d ∈ digits (10 : ℕ) (n : ℕ), d ∈ {0, 2, 4, 5, 7, 9}

theorem sum_of_digits_M (M : ℕ) 
  (even_M : M % 2 = 0) 
  (valid_M : valid_digits M) 
  (H1 : digit_sum (2 * M) = 39) 
  (H2 : digit_sum (M / 2) = 30) : 
  digit_sum M = 33 :=
sorry

end sum_of_digits_M_l202_202171


namespace nine_sided_polygon_diagonals_l202_202647

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202647


namespace nine_sided_polygon_diagonals_l202_202632

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202632


namespace max_ab_value_l202_202461

-- Definitions based on conditions
def condition (x a b : ℝ) : Prop := exp x ≥ a * x + b

-- Problem statement using the condition
theorem max_ab_value : (∀ x : ℝ, ∀ a b : ℝ, condition x a b) → ∃ ab_max : ℝ, ab_max = (exp 1)/2 :=
sorry

end max_ab_value_l202_202461


namespace regular_nonagon_diagonals_correct_l202_202716

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202716


namespace rate_of_interest_l202_202927

-- Define the conditions
def P : ℝ := 1200
def SI : ℝ := 432
def T (R : ℝ) : ℝ := R

-- Define the statement to be proven
theorem rate_of_interest (R : ℝ) (h : SI = (P * R * T R) / 100) : R = 6 :=
by sorry

end rate_of_interest_l202_202927


namespace angle_BAC_eq_90_l202_202151

theorem angle_BAC_eq_90 {A B C D K L : Point} (h1 : A ≠ B) (h2 : A ≠ C)
  (h3 : AD ⊥ BC) (h4 : foot A BC D) 
  (O1 O2 : Point) (h5 : incenter (triangle A B D) O1) (h6 : incenter (triangle A C D) O2)
  (h7 : intersect_line O1 O2 AB K) (h8 : intersect_line O1 O2 AC L)
  (h9 : AK = AL) : ∠BAC = 90° := 
sorry

end angle_BAC_eq_90_l202_202151


namespace max_compliant_cards_l202_202232

theorem max_compliant_cards : 
  let card_set : Finset ℕ := Finset.range 21
  let compliant (a b : ℕ) : Prop := a = 2 * b + 2
  ∃ (S : Finset ℕ), S.card = 12 ∧ ∀ {a}, a ∈ S → ∃ b, compliant a b :=
begin
  -- Problem Conditions
  let card_set : Finset ℕ := Finset.range 21,
  let compliant (a b : ℕ) : Prop := a = 2 * b + 2,

  -- Desired Result
  use {1, 2, 3, 5, 7, 9, 4, 6, 8, 12, 16, 20},
  split,
  { refl },
  {
    intros a ha,
    simp at ha,
    rcases ha with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl,
    { use 1 },
    { use 2 },
    { use 3 },
    { use 5 },
    { use 7 },
    { use 9 },
    { use 1 },
    { use 2 },
    { use 3 },
    { use 5 },
    { use 7 },
    { use 9 },
  },
end

end max_compliant_cards_l202_202232


namespace find_x_when_z_is_27_l202_202288

theorem find_x_when_z_is_27 (x : ℝ) (h1 : ∃ y : ℤ, y ≤ x ∧ z = 2^y - y ∧ z = 27) : x = 5.5 := by
  sorry

end find_x_when_z_is_27_l202_202288


namespace nine_sided_polygon_diagonals_l202_202626

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202626


namespace nine_sided_polygon_diagonals_count_l202_202696

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202696


namespace greatest_prime_factor_of_sum_l202_202998

def a := 5^8
def b := 8^7
def n := a + b

theorem greatest_prime_factor_of_sum : 
  ∃ p : ℕ, p.prime ∧ p ∣ n ∧ ∀ q : ℕ, q.prime ∧ q ∣ n → q ≤ p :=
sorry

end greatest_prime_factor_of_sum_l202_202998


namespace apples_per_basket_l202_202299

theorem apples_per_basket (total_apples : ℕ) (baskets : ℕ) (h1 : total_apples = 629) (h2 : baskets = 37) :
  total_apples / baskets = 17 :=
by
  sorry

end apples_per_basket_l202_202299


namespace f_expression_correct_f_range_l202_202070

noncomputable def f (x : ℝ) : ℝ :=
if h₁ : 0 < x ∧ x < π/2 then tan x / (tan x + 1)
else if h₂ : x = 0 then 0
else if h₃ : -π/2 < x ∧ x < 0 then tan x / (1 - tan x)
else 0

theorem f_expression_correct :
  ∀ x, -π/2 < x ∧ x < π/2 →
    (f x = if 0 < x ∧ x < π/2 then tan x / (tan x + 1)
           else if x = 0 then 0
           else tan x / (1 - tan x)) :=
begin
  intros x hx,
  unfold f,
  by_cases 0 < x ∧ x < π/2,
  { rw if_pos h, },
  by_cases x = 0,
  { rw [if_neg h, if_pos h_1] },
  { rw [if_neg h, if_neg h_1, if_pos hx], }
end

theorem f_range (m : ℝ) :
  (∃ x, -π/2 < x ∧ x < π/2 ∧ f x = m) ↔ m ∈ set.Ioo (-1 : ℝ) 1 :=
begin
  sorry
end

end f_expression_correct_f_range_l202_202070


namespace same_function_option_d_l202_202388

def f₁ (x : ℝ) := x
def g₁ (x : ℝ) := if x = 0 then 0 else x

def f₂ (x : ℝ) := 1
def g₂ (x : ℝ) := x^0

def f₃ (x : ℝ) := x
def g₃ (x : ℝ) := real.sqrt (x^2)

def f₄ (x : ℝ) := abs (x + 2)
def g₄ (x : ℝ) := if x ≥ -2 then x + 2 else -x - 2

theorem same_function_option_d : (∀ x, f₄ x = g₄ x) :=
by {
  -- The proof steps would go here, but they are omitted as per instructions.
  sorry
}

end same_function_option_d_l202_202388


namespace nine_sided_polygon_diagonals_l202_202616

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202616


namespace complex_plane_quadrant_l202_202473

theorem complex_plane_quadrant :
  let z := (2 + 3 * Complex.I) / (1 + 2 * Complex.I) in
  Re z > 0 ∧ Im z < 0 :=
by
  let z := (2 + 3 * Complex.I) / (1 + 2 * Complex.I)
  have : Re z = 8 / 5 := by sorry
  have : Im z = -1 / 5 := by sorry
  split
  · linarith
  
  · linarith

end complex_plane_quadrant_l202_202473


namespace mabel_steps_l202_202900

theorem mabel_steps :
  ∃ (M : ℕ), (fun M => let helen_steps := (3 / 4 : ℚ) * M in M + helen_steps = 7875) M ∧ M = 4500 :=
by
  let M := 4500
  have M_pos : 0 < M := by decide
  use M
  let helen_steps := (3 / 4 : ℚ) * M
  have total_steps : M + helen_steps = 7875 := by norm_num
  constructor
  · exact total_steps
  · rfl

end mabel_steps_l202_202900


namespace minimal_n_for_partition_l202_202883

theorem minimal_n_for_partition (n : ℕ) (h : n ≥ 4) :
  (∀ C D : set ℕ, (C ∪ D = { x : ℕ | 4 ≤ x ∧ x ≤ n } ∧ C ∩ D = ∅) →
  (∃ a b c ∈ (C ∪ D), a + b = c)) ↔ (n = 16) :=
begin
  sorry -- The proof goes here.
end

end minimal_n_for_partition_l202_202883


namespace question_one_question_two_l202_202103

variable (b x : ℝ)
def f (x : ℝ) : ℝ := x^2 - b * x + 3

theorem question_one (h : f b 0 = f b 4) : ∃ x1 x2 : ℝ, f b x1 = 0 ∧ f b x2 = 0 ∧ (x1 = 3 ∧ x2 = 1) ∨ (x1 = 1 ∧ x2 = 3) := by 
  sorry

theorem question_two (h1 : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f b x1 = 0 ∧ f b x2 = 0) : b > 4 := by
  sorry

end question_one_question_two_l202_202103


namespace number_of_diagonals_l202_202561

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202561


namespace find_ab_l202_202928

theorem find_ab (a b : ℝ) : 
  (∀ x : ℝ, (3 * x - a) * (2 * x + 5) - x = 6 * x^2 + 2 * (5 * x - b)) → a = 2 ∧ b = 5 :=
by
  intro h
  -- We assume the condition holds for all x
  sorry -- Proof not needed as per instructions

end find_ab_l202_202928


namespace verify_part_a_verify_part_b_l202_202843

noncomputable def part_a (R AO_A h_a r_a : ℝ) : Prop :=
  R / AO_A = (h_a - r_a) / h_a

noncomputable def part_b (AO_A BO_B CO_C R : ℝ) : Prop :=
  1 / AO_A + 1 / BO_B + 1 / CO_C = 2 / R

theorem verify_part_a (O : Point) (A B C : Point) (R AO_A h_a r_a : ℝ)
  (hA : circle_center O A B C)
  (hR : radius O A = R)
  (h_ha : altitude O A B C = h_a)
  (h_r_a : segment O A B C = r_a)
  (h_AO_A : point_def O A AO_A) :
  part_a R AO_A h_a r_a :=
by sorry

theorem verify_part_b (O : Point) (A B C : Point) (R AO_A BO_B CO_C : ℝ)
  (hA : circle_center O A B C)
  (hR : radius O A = R)
  (h_AO_A : point_def O A AO_A)
  (h_BO_B : point_def O B BO_B)
  (h_CO_C : point_def O C CO_C) :
  part_b AO_A BO_B CO_C R :=
by sorry

end verify_part_a_verify_part_b_l202_202843


namespace greg_ppo_reward_l202_202523

-- Define the parameters
def max_procgen_reward : ℝ := 240
def max_coinrun_reward : ℝ := max_procgen_reward / 2
def ppo_algorithm_percentage : ℝ := 0.9
def expected_reward : ℝ := max_coinrun_reward * ppo_algorithm_percentage

-- Theorem statement
theorem greg_ppo_reward : 
  (max_coinrun_reward = 120) → 
  (ppo_algorithm_percentage = 0.9) → 
  (max_procgen_reward = 240) → 
  expected_reward = 108 := 
by 
  intros h1 h2 h3 h4
  sorry

end greg_ppo_reward_l202_202523


namespace students_remaining_after_three_stops_l202_202376

theorem students_remaining_after_three_stops
  (initial_students : ℕ)
  (one_third_off_1 : ℕ → ℕ := λ n, n * 2 / 3)
  (one_third_off_2 : ℕ → ℕ := λ n, n * 2 / 3)
  (one_fourth_off_3 : ℕ → ℕ := λ n, n * 3 / 4)
  (initial_condition : initial_students = 60) :
  one_fourth_off_3 (one_third_off_2 (one_third_off_1 initial_students)) = 20 :=
by
  rw initial_condition
  have h1 : one_third_off_1 60 = 40 := by norm_num
  have h2 : one_third_off_2 40 = 40 * 2 / 3 := by norm_num
  have h3 : one_fourth_off_3 (40 * 2 / 3) = 20 := by norm_num
  exact h3

end students_remaining_after_three_stops_l202_202376


namespace diagonals_in_nine_sided_polygon_l202_202780

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202780


namespace regular_nine_sided_polygon_diagonals_l202_202767

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202767


namespace log_sin_alpha_interval_l202_202813

open Real

theorem log_sin_alpha_interval (α : ℝ) (hα : 0 < α ∧ α < π / 3) :
  3 ^ abs (log 3 (sin α)) = 1 / sin α :=
sorry

end log_sin_alpha_interval_l202_202813


namespace alice_sales_above_goal_l202_202387

theorem alice_sales_above_goal :
  let quota := 1000
  let nike_price := 60
  let adidas_price := 45
  let reebok_price := 35
  let nike_sold := 8
  let adidas_sold := 6
  let reebok_sold := 9
  let total_sales := nike_price * nike_sold + adidas_price * adidas_sold + reebok_price * reebok_sold
  in total_sales - quota = 65 :=
by
  let quota := 1000
  let nike_price := 60
  let adidas_price := 45
  let reebok_price := 35
  let nike_sold := 8
  let adidas_sold := 6
  let reebok_sold := 9
  let total_sales := nike_price * nike_sold + adidas_price * adidas_sold + reebok_price * reebok_sold
  show total_sales - quota = 65 from sorry

end alice_sales_above_goal_l202_202387


namespace exists_point_D_min_EF_l202_202075

variables {A B C D E F : Point} {l1 l2 : Line}
variables [triangle_ABC : Triangle A B C] [on_AB : D ∈ (AB : Segment A B)]
variables [parallel_l1_AC : ∀ {D}, parallel (Line_through D ← l1) (Line_through D E)]
variables [parallel_l2_BC : ∀ {D}, parallel (Line_through D ← l2) (Line_through D F)]

theorem exists_point_D_min_EF
  (triangle_ABC : Triangle A B C)
  (l1 l2 : Line)
  (D : Point)
  (on_AB : D ∈ (AB : Segment A B))
  (parallel_l1_AC : ∀ {D}, parallel (Line_through D ← l1) (Line_through D E))
  (parallel_l2_BC : ∀ {D}, parallel (Line_through D ← l2) (Line_through D F)) :
  ∃ D₀ : Point, D₀ ∈ (AB : Segment A B) ∧ (∀ D ∈ (AB : Segment A B), length (EF : Segment E F) ≥ length (Segment E₀ F₀)) :=
sorry

end exists_point_D_min_EF_l202_202075


namespace total_workers_count_l202_202848

/-
Jack and Jill work at a hospital with 7 other workers. For an internal review, 2 of the total workers will be randomly chosen to be interviewed.
Given that the probability that Jack and Jill will both be chosen is 0.027777777777777776, we need to prove that the total number of workers is 9.
-/
theorem total_workers_count (N : ℕ) (h1 : N = 9) (h2 : 0.027777777777777776 = 1 / (N.choose 2)) : 
  N = 9 :=
by sorry

end total_workers_count_l202_202848


namespace find_n_value_l202_202366

theorem find_n_value (n a b : ℕ) 
    (h1 : n = 12 * b + a)
    (h2 : n = 10 * a + b)
    (h3 : 0 ≤ a ∧ a ≤ 11)
    (h4 : 0 ≤ b ∧ b ≤ 9) : 
    n = 119 :=
by
  sorry

end find_n_value_l202_202366


namespace midpoint_trajectory_l202_202361

theorem midpoint_trajectory (x y : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (8, 0) ∧ (B.1, B.2) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 } ∧ 
   ∃ P : ℝ × ℝ, P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ P = (x, y)) → (x - 4)^2 + y^2 = 1 :=
by sorry

end midpoint_trajectory_l202_202361


namespace leila_savings_l202_202872

theorem leila_savings (S : ℝ) (h : (1 / 4) * S = 20) : S = 80 :=
by
  sorry

end leila_savings_l202_202872


namespace find_area_outside_of_triangle_l202_202873

open Real

def right_triangle {α : Type} [MetricSpace α] (A B C X Y : α) :=
  dist A C = dist B C ∧
  ∠ A B C = π/2

def tangent_to_sides (A B C X Y O : Point) (r : ℝ) :=
  dist O X = r ∧ dist O Y = r ∧
  dist B C = 2 * r

noncomputable def area_outside_triangle (A B C X Y O : Point) (r : ℝ) :=
  π * r^2 - (1/2 * dist A B * dist A C)

theorem find_area_outside_of_triangle (A B C X Y O : Point) (r : ℝ)
  (h1 : right_triangle A B C X Y)
  (h2 : tangent_to_sides A B C X Y O r)
  (h3 : dist A B = 6) :
  area_outside_triangle A B C X Y O r = 18 * π - 18 :=
by sorry

end find_area_outside_of_triangle_l202_202873


namespace diagonals_in_nine_sided_polygon_l202_202747

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202747


namespace fraction_r_over_b_l202_202929

-- Definition of the conditions
def initial_expression (k : ℝ) : ℝ := 8 * k^2 - 12 * k + 20

-- Proposition statement
theorem fraction_r_over_b : ∃ a b r : ℝ, 
  (∀ k : ℝ, initial_expression k = a * (k + b)^2 + r) ∧ 
  r / b = -47.33 :=
sorry

end fraction_r_over_b_l202_202929


namespace diagonals_in_nine_sided_polygon_l202_202746

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202746


namespace coal_production_read_and_rewrite_l202_202829

-- Define the number
def number_in_tons : ℕ := 15500000000

-- Define the number read as a string
def number_read_as : String := "one hundred and fifty-five billion"

-- Define the number converted to billions
def number_in_billions : ℕ := 155

-- The theorem stating the equivalent proof problem
theorem coal_production_read_and_rewrite :
  (read_number number_in_tons = number_read_as) ∧ 
  (rewrite_to_billion number_in_tons = number_in_billions) :=
by sorry

end coal_production_read_and_rewrite_l202_202829


namespace nine_sided_polygon_diagonals_l202_202658

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202658


namespace nine_sided_polygon_diagonals_l202_202640

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202640


namespace sin_alpha_value_l202_202817

theorem sin_alpha_value (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h_cos : cos (α + π / 3) = -4 / 5) :
  sin α = (3 + 4 * sqrt 3) / 10 := by
  sorry

end sin_alpha_value_l202_202817


namespace limit_of_f_at_2_l202_202406

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (sin (3 * π * x) / sin (π * x)) ^ (sin (x - 2)) ^ 2

theorem limit_of_f_at_2 : filter.tendsto (λ x, f x) (𝓝 2) (𝓝 1) :=
sorry

end limit_of_f_at_2_l202_202406


namespace trapezoidConstructionCondition_l202_202410

-- Define the parameters for the trapezoid
variables {c d : ℝ}

-- Define the necessary constraints for the construction
def canConstructTrapezoid (c d : ℝ) : Prop :=
  d / 2 ≤ c ∧ c < d

-- The main theorem statement
theorem trapezoidConstructionCondition (c d : ℝ) : 
  canConstructTrapezoid c d ↔ (d / 2 ≤ c ∧ c < d) :=
begin
  sorry,
end

end trapezoidConstructionCondition_l202_202410


namespace nine_sided_polygon_diagonals_l202_202633

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202633


namespace minimum_value_integral_l202_202874

noncomputable def d (a b t : ℝ) := 1 / (Real.sqrt (a^2 * Real.exp(2 * t) + (b^2 / Real.exp(2 * t))))

theorem minimum_value_integral (a b : ℝ) (hab : a * b = 1) :
  ∃! c : ℝ, ∫ t in 0..1, (a^2 * Real.exp(2 * t) + b^2 * Real.exp(-2 * t)) = c ∧ c = Real.exp(1) - Real.exp(-1) := sorry

end minimum_value_integral_l202_202874


namespace number_of_valid_Ns_l202_202133

-- Definition of the problem conditions
def is_valid_N (N : ℕ) :=
  ∃ (j n : ℕ), j ≥ 1 ∧ n ≥ 0 ∧ N = j * (2 * n + j)

def has_exactly_5_factors (N : ℕ) :=
  (N.factors.Nub.length = 5)

-- The main theorem
theorem number_of_valid_Ns : 
  ∃ count : ℕ, (count = 15) ∧ (∀ N < 1000, is_valid_N N → has_exactly_5_factors N → N < 1000) :=
  sorry

end number_of_valid_Ns_l202_202133


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202609

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202609


namespace complex_plane_quadrant_l202_202474

theorem complex_plane_quadrant :
  let z := (2 + 3 * Complex.I) / (1 + 2 * Complex.I) in
  Re z > 0 ∧ Im z < 0 :=
by
  let z := (2 + 3 * Complex.I) / (1 + 2 * Complex.I)
  have : Re z = 8 / 5 := by sorry
  have : Im z = -1 / 5 := by sorry
  split
  · linarith
  
  · linarith

end complex_plane_quadrant_l202_202474


namespace find_a_b_solution_set_l202_202468

-- Given function
def f (x : ℝ) (a b : ℝ) := x^2 - (a + b) * x + 3 * a

-- Part 1: Prove the values of a and b given the solution set of the inequality
theorem find_a_b (a b : ℝ) 
  (h1 : 1^2 - (a + b) * 1 + 3 * 1 = 0)
  (h2 : 3^2 - (a + b) * 3 + 3 * 1 = 0) :
  a = 1 ∧ b = 3 :=
sorry

-- Part 2: Find the solution set of the inequality f(x) > 0 given b = 3
theorem solution_set (a : ℝ)
  (h : b = 3) :
  (a > 3 → (∀ x, f x a 3 > 0 ↔ x < 3 ∨ x > a)) ∧
  (a < 3 → (∀ x, f x a 3 > 0 ↔ x < a ∨ x > 3)) ∧
  (a = 3 → (∀ x, f x a 3 > 0 ↔ x ≠ 3)) :=
sorry

end find_a_b_solution_set_l202_202468


namespace certain_percentage_of_1600_l202_202352

theorem certain_percentage_of_1600 (P : ℝ) 
  (h : 0.05 * (P / 100 * 1600) = 20) : 
  P = 25 :=
by 
  sorry

end certain_percentage_of_1600_l202_202352


namespace dem_a_pct_is_75_l202_202831

variable (dem_pct rep_pct rep_a_pct a_total_pct dem_a_pct : ℝ)

axiom h1 : dem_pct = 0.60
axiom h2 : rep_pct = 0.40
axiom h3 : rep_a_pct = 0.20
axiom h4 : a_total_pct = 0.53

theorem dem_a_pct_is_75 : dem_a_pct = 0.75 :=
by
  have h : dem_pct * dem_a_pct + rep_pct * rep_a_pct = a_total_pct := 
    calc 
      dem_pct * dem_a_pct + rep_pct * rep_a_pct
        = 0.60 * dem_a_pct + 0.40 * 0.20 : by rw [h1, h2, h3]
    ... = 0.60 * dem_a_pct + 0.08 : by norm_num
    ... = 0.53 : by rw [h4]
  have h5 : 0.60 * dem_a_pct = 0.45 := by linarith
  have h6 : dem_a_pct = 0.75 := by field_simp at h5
  rw h6 at h5
  exact h6

end dem_a_pct_is_75_l202_202831


namespace nine_sided_polygon_diagonals_l202_202665

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202665


namespace diagonals_in_nine_sided_polygon_l202_202790

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202790


namespace nine_sided_polygon_diagonals_l202_202644

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202644


namespace nine_sided_polygon_diagonals_l202_202627

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202627


namespace problem_statement_l202_202065

-- Defining the condition x^3 = 8
def condition1 (x : ℝ) : Prop := x^3 = 8

-- Defining the function f(x) = (x-1)(x+1)(x^2 + x + 1)
def f (x : ℝ) : ℝ := (x - 1) * (x + 1) * (x^2 + x + 1)

-- The theorem we want to prove: For any x satisfying the condition, the function value is 21
theorem problem_statement (x : ℝ) (h : condition1 x) : f x = 21 := 
by
  sorry

end problem_statement_l202_202065


namespace solve_for_x_l202_202249

theorem solve_for_x (x : ℝ) (hx_pos : x > 0) (h_eq : 3 * x^2 + 13 * x - 10 = 0) : x = 2 / 3 :=
sorry

end solve_for_x_l202_202249


namespace proof_equation_l202_202522

open Set

variable (U : Set ℕ) (M N : Set ℕ)

noncomputable def complement_U (U M : Set ℕ) : Set ℕ := U \ M

theorem proof_equation :
  U = {1, 2, 3, 4, 5} →
  M = {1, 4} →
  N = {1, 3, 5} →
  (complement_U U M ∪ complement_U U N) = {2, 3, 4, 5} :=
by
  intro hU hM hN
  rw [hU, hM, hN]
  have : complement_U U M = {2, 3, 5} := by sorry
  have : complement_U U N = {2, 4} := by sorry
  rw [this, this]
  sorry

end proof_equation_l202_202522


namespace sum_of_coefficients_l202_202465

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) :
  (1 - 2 * x)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + 
                  a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 = -1 :=
sorry

end sum_of_coefficients_l202_202465


namespace sum_of_two_rel_prime_numbers_l202_202437

theorem sum_of_two_rel_prime_numbers (k : ℕ) : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1 ∧ k = a + b) ↔ (k = 5 ∨ k ≥ 7) := sorry

end sum_of_two_rel_prime_numbers_l202_202437


namespace regular_nine_sided_polygon_diagonals_l202_202760

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202760


namespace regular_nonagon_diagonals_correct_l202_202706

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202706


namespace parabola_coefficients_l202_202257

theorem parabola_coefficients (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ (y = (x + 2)^2 + 5) ∧ y = 9 ↔ x = 0) →
  (a, b, c) = (1, 4, 9) :=
by
  intros h
  sorry

end parabola_coefficients_l202_202257


namespace board_configurations_count_l202_202394

-- Define the main problem as a theorem
theorem board_configurations_count : 
  let grid_size := 4
  let remaining_cells := 13
  let lines_count := 10
  let valid_x_placements := 290 
  in grid_size * remaining_cells * valid_x_placements = 37700 := 
by 
  sorry

end board_configurations_count_l202_202394


namespace find_particular_number_l202_202252

theorem find_particular_number (x : ℤ) (h : x - 7 = 2) : x = 9 :=
by {
  -- The proof will be written here.
  sorry
}

end find_particular_number_l202_202252


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202614

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202614


namespace holmium166_neutron_proton_difference_l202_202242

-- Definitions for the conditions
def mass_number_166Ho : ℕ := 166
def protons_166Ho : ℕ := 67
def neutrons_166Ho : ℕ := mass_number_166Ho - protons_166Ho

-- Problem statement in Lean: 
theorem holmium166_neutron_proton_difference :
  let difference := neutrons_166Ho - protons_166Ho
  in difference = 32 :=
by
  sorry

end holmium166_neutron_proton_difference_l202_202242


namespace diagonals_in_nine_sided_polygon_l202_202777

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202777


namespace y_intercept_of_line_l202_202320

theorem y_intercept_of_line : ∃ y : ℝ, 3 * 0 - 5 * y = 10 ∧ y = -2 :=
by
  use -2
  split
  { norm_num }
  { refl }

end y_intercept_of_line_l202_202320


namespace journey_time_calculation_l202_202365

theorem journey_time_calculation (dist totalDistance : ℝ) (rate1 rate2 : ℝ)
  (firstHalfDistance secondHalfDistance : ℝ) (time1 time2 totalTime : ℝ) :
  totalDistance = 224 ∧ rate1 = 21 ∧ rate2 = 24 ∧
  firstHalfDistance = totalDistance / 2 ∧ secondHalfDistance = totalDistance / 2 ∧
  time1 = firstHalfDistance / rate1 ∧ time2 = secondHalfDistance / rate2 ∧
  totalTime = time1 + time2 →
  totalTime = 10 :=
sorry

end journey_time_calculation_l202_202365


namespace eccentricity_is_one_fourth_l202_202488

noncomputable def ellipse_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a > b) (c : ℝ) (P : ℝ × ℝ) (hP : P.fst^2 / a^2 + P.snd^2 / b^2 = 1) 
  (hPF_perp_x : P.snd / (P.fst - (-c)) = 0) (hPF_AF : P.snd^2 + (P.fst + c)^2 = (3 / 4)^2 * a^2) : ℝ :=
let e := c / a in
if h : 4 * e^2 + 3 * e - 1 = 0 then
  e
else
  0 -- Invalid result just to make type signatures work. We expect h to be true.

-- Here we assert the actual result
theorem eccentricity_is_one_fourth (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a > b) (c : ℝ) (P : ℝ × ℝ) (hP : P.fst^2 / a^2 + P.snd^2 / b^2 = 1) 
  (hPF_perp_x : P.snd / (P.fst - (-c)) = 0) (hPF_AF : P.snd^2 + (P.fst + c)^2 = (3 / 4)^2 * a^2) :
  ellipse_eccentricity a b ha hb hab c P hP hPF_perp_x hPF_AF = 1 / 4 := sorry

end eccentricity_is_one_fourth_l202_202488


namespace nine_sided_polygon_diagonals_l202_202624

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202624


namespace regular_nonagon_diagonals_correct_l202_202709

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202709


namespace y_intercept_of_line_l202_202323

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 5 * y = 10) : 
  3 * 0 - 5 * y = 10 → y = -2 :=
by
  intros h0
  rw [mul_zero, sub_eq_add_neg, add_zero] at h0
  exact eq_of_mul_eq_mul_left (ne_of_gt (by norm_num)) (by simpa only [neg_mul, one_mul, neg_eq_iff_neg_eq] using h0)

end y_intercept_of_line_l202_202323


namespace find_x_l202_202434

theorem find_x (x : ℝ) (h : 3^(Real.log x / Real.log 8) = 81) : x = 4096 := by
  sorry

end find_x_l202_202434


namespace mariel_dogs_count_l202_202901

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

end mariel_dogs_count_l202_202901


namespace max_value_of_expression_l202_202046

theorem max_value_of_expression :
  ∃ (x y z : ℝ), 
    let expr := (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) in 
    expr = 4.5 ∧
    ∀ (x y z : ℝ), 
      (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 :=
begin
  sorry,
end

end max_value_of_expression_l202_202046


namespace exists_16_non_collinear_positions_l202_202178

def is_collinear (p1 p2 p3 : (ℕ × ℕ)) : Prop :=
  let ⟨x1, y1⟩ := p1
  let ⟨x2, y2⟩ := p2
  let ⟨x3, y3⟩ := p3
  x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) = 0

theorem exists_16_non_collinear_positions :
  ∃ (positions : fin 16 → (ℕ × ℕ)),
    (∀ i : fin 16, 1 ≤ (positions i).fst ∧ (positions i).fst ≤ 8 ∧
                   1 ≤ (positions i).snd ∧ (positions i).snd ≤ 8) ∧ 
    (∀ (i j k : fin 16), i ≠ j → j ≠ k → i ≠ k → ¬ is_collinear (positions i) (positions j) (positions k)) :=
begin
  sorry,
end

end exists_16_non_collinear_positions_l202_202178


namespace find_x_parallel_find_x_perpendicular_l202_202117

def a (x : ℝ) : ℝ × ℝ := (x, x + 2)
def b : ℝ × ℝ := (1, 2)

-- Given that a vector is proportional to another
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Given that the dot product is zero
def are_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_x_parallel (x : ℝ) (h : are_parallel (a x) b) : x = 2 :=
by sorry

theorem find_x_perpendicular (x : ℝ) (h : are_perpendicular (a x - b) b) : x = (1 / 3 : ℝ) :=
by sorry

end find_x_parallel_find_x_perpendicular_l202_202117


namespace rank_classmates_l202_202411

-- Definitions of the conditions
def emma_tallest (emma david fiona : ℕ) : Prop := emma > david ∧ emma > fiona
def fiona_not_shortest (david emma fiona : ℕ) : Prop := david > fiona ∧ emma > fiona
def david_not_tallest (david emma fiona : ℕ) : Prop := emma > david ∧ fiona > david

def exactly_one_true (david emma fiona : ℕ) : Prop :=
  (emma_tallest emma david fiona ∧ ¬fiona_not_shortest david emma fiona ∧ ¬david_not_tallest david emma fiona) ∨
  (¬emma_tallest emma david fiona ∧ fiona_not_shortest david emma fiona ∧ ¬david_not_tallest david emma fiona) ∨
  (¬emma_tallest emma david fiona ∧ ¬fiona_not_shortest david emma fiona ∧ david_not_tallest david emma fiona)

-- The final proof statement
theorem rank_classmates (david emma fiona : ℕ) (h : exactly_one_true david emma fiona) : david > fiona ∧ fiona > emma :=
  sorry

end rank_classmates_l202_202411


namespace regular_nonagon_diagonals_correct_l202_202710

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202710


namespace maximum_area_right_triangle_hypotenuse_8_l202_202962

theorem maximum_area_right_triangle_hypotenuse_8 :
  ∃ a b : ℝ, (a^2 + b^2 = 64) ∧ (a * b) / 2 = 16 :=
by
  sorry

end maximum_area_right_triangle_hypotenuse_8_l202_202962


namespace double_angle_quadrant_l202_202489

noncomputable theory

-- Define α based on given conditions.
def α : ℝ := arbitrary ℝ 

-- Given conditions
axiom cos_α : Real.cos α = -4/5
axiom sin_α : Real.sin α = 3/5

-- The type for the quadrant
inductive Quadrant
| First
| Second
| Third
| Fourth

-- Prove that 2α is in the fourth quadrant
theorem double_angle_quadrant : (2 * α : ℝ) ∈ Quadrant.Fourth :=
by
  -- Proof omitted
  sorry

end double_angle_quadrant_l202_202489


namespace angle_BAC_60_l202_202846

noncomputable def equilateral_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  dist A B = dist A C ∧ dist A B = dist B C

noncomputable def incenter (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (I : Type) : Prop :=
  sorry -- Definition of incenter, can be filled with actual construction

noncomputable def Fermat_point (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (F : Type) : Prop :=
  sorry -- Properties of Fermat point, can be filled with actual properties

def point_in_circumcenter_of_inscribed_circle {A B C D E F P R O} [MetricSpace P] :
    equilateral_triangle A B D ∧ equilateral_triangle A C E ∧
    incenter D E F A → angle B A C = 60 :=
by
  sorry

theorem angle_BAC_60 {A B C D E F : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] :
  let BD := equilateral_triangle A B D in
  let CE := equilateral_triangle A C E in
  let I := incenter D E F A in
  BD ∧ CE ∧ I → angle B A C = 60 :=
by
  sorry

end angle_BAC_60_l202_202846


namespace notebook_cost_l202_202905

theorem notebook_cost (total_spent ruler_cost pencil_count pencil_cost: ℕ)
  (h1 : total_spent = 74)
  (h2 : ruler_cost = 18)
  (h3 : pencil_count = 3)
  (h4 : pencil_cost = 7) :
  total_spent - (ruler_cost + pencil_count * pencil_cost) = 35 := 
by 
  sorry

end notebook_cost_l202_202905


namespace diagonals_in_nonagon_l202_202733

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202733


namespace regular_nine_sided_polygon_diagonals_l202_202766

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202766


namespace bella_steps_to_meet_ella_l202_202397

theorem bella_steps_to_meet_ella :
  ∀ (distance : ℕ) (b_speed : ℕ) (e_speed : ℕ) (step_length : ℕ), 
  distance = 15840 → e_speed = 4 * b_speed → step_length = 3 → 
  (∃ steps : ℕ, steps = 1056) :=
by {
  intros,
  use 1056,
  sorry
}

end bella_steps_to_meet_ella_l202_202397


namespace nine_sided_polygon_diagonals_l202_202643

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202643


namespace nine_sided_polygon_diagonals_count_l202_202692

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202692


namespace find_p_q_r_l202_202200

variables (a b c : ℝ^3) (p q r : ℝ)

-- Definitions of the conditions from part a).
def orthogonal (u v : ℝ^3) : Prop := u ⬝ v = 0
def unit_vector (u : ℝ^3) : Prop := ∥u∥ = 1

-- The problem conditions
axiom cond1 : orthogonal a b ∧ orthogonal b c ∧ orthogonal c a
axiom cond2 : unit_vector a ∧ unit_vector b ∧ unit_vector c
axiom cond3 : a = p * (a × b) + q * (b × c) + r * (c × a)
axiom cond4 : a ⬝ (b × c) = 2

-- Problem to solve
theorem find_p_q_r : p + q + r = 1/2 := sorry

end find_p_q_r_l202_202200


namespace max_sum_at_1008_l202_202077

noncomputable def sum_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem max_sum_at_1008 (a : ℕ → ℝ) : 
  sum_sequence a 2015 > 0 → 
  sum_sequence a 2016 < 0 → 
  ∃ n, n = 1008 ∧ ∀ m, sum_sequence a m ≤ sum_sequence a 1008 :=
by
  intros h1 h2
  sorry

end max_sum_at_1008_l202_202077


namespace interval_length_difference_l202_202960

theorem interval_length_difference (a b : ℝ) 
  (h1 : Icc a b ≠ ∅) 
  (h2 : ∀ x ∈ Icc a b, 1 ≤ 4 ^ |x| ∧ 4 ^ |x| ≤ 4) :
  ∃ (M m : ℝ), (∃ x1 x2 in Icc a b, M = abs ((x2 - x1) : ℝ)) 
  ∧ (∃ x1 x2 in Icc a b, m = abs ((x2 - x1) : ℝ)) ∧ (M - m = 1) :=
by
  sorry

end interval_length_difference_l202_202960


namespace regular_nine_sided_polygon_diagonals_l202_202764

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202764


namespace diagonals_in_regular_nine_sided_polygon_l202_202540

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202540


namespace isaac_wakes_up_at_isaac_wakes_up_at_am_l202_202427

-- Definitions
def sleep_time := 23 -- 11 PM in 24-hour format
def sleep_duration := 8 -- 8 hours

-- Assertion
theorem isaac_wakes_up_at :
  sleep_time - sleep_duration = 15 :=
  by
    sorry

# We need to define the final proof to convert 15:00 in 24-hour to 3:00 AM
theorem isaac_wakes_up_at_am :
  (sleep_time - sleep_duration) - 12 = 3 :=
  by
    sorry

end isaac_wakes_up_at_isaac_wakes_up_at_am_l202_202427


namespace a_sub_b_eq_2_l202_202807

theorem a_sub_b_eq_2 (a b : ℝ)
  (h : (a - 5) ^ 2 + |b ^ 3 - 27| = 0) : a - b = 2 :=
by
  sorry

end a_sub_b_eq_2_l202_202807


namespace diagonals_in_nine_sided_polygon_l202_202740

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202740


namespace number_of_diagonals_l202_202551

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202551


namespace balloon_highest_elevation_l202_202858

theorem balloon_highest_elevation 
  (lift_rate : ℕ)
  (descend_rate : ℕ)
  (pull_time1 : ℕ)
  (release_time : ℕ)
  (pull_time2 : ℕ) :
  lift_rate = 50 →
  descend_rate = 10 →
  pull_time1 = 15 →
  release_time = 10 →
  pull_time2 = 15 →
  (lift_rate * pull_time1 - descend_rate * release_time + lift_rate * pull_time2) = 1400 :=
by
  sorry

end balloon_highest_elevation_l202_202858


namespace range_of_a_l202_202409

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then (1/2) ^ x - 7 else real.sqrt x

theorem range_of_a (a : ℝ) : f a < 1 → a ∈ Set.Ioo (-3 : ℝ) 1 :=
sorry

end range_of_a_l202_202409


namespace regular_nine_sided_polygon_diagonals_l202_202763

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202763


namespace diagonals_in_nonagon_l202_202725

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202725


namespace nine_sided_polygon_diagonals_count_l202_202695

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202695


namespace number_of_diagonals_l202_202555

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202555


namespace largest_lambda_property_l202_202040

noncomputable def largest_lambda (a b c : ℝ) (f : ℝ → ℝ) : ℝ := 
  if ∃ (α β γ : ℝ), (∀ x, f x = (x - α) * (x - β) * (x - γ)) ∧ (0 ≤ α ∧ 0 ≤ β ∧ 0 ≤ γ) then
    -1 / 27
  else 0  -- This is a fallback value; in practice, the preconditions ensure the "if" condition holds.

theorem largest_lambda_property (a b c : ℝ) (f : ℝ → ℝ) :
  (∃ (α β γ : ℝ), (∀ x, f x = (x - α) * (x - β) * (x - γ)) ∧ (0 ≤ α ∧ 0 ≤ β ∧ 0 ≤ γ)) →
  (∀ x ≥ 0, f x ≥ largest_lambda a b c f * (x - a)^3) ∧
  (∀ x ≥ 0, f x = largest_lambda a b c f * (x - a)^3 ↔ (x = 0 ∧ α = β ∧ β = γ) ∨ (α = β = 0 ∧ γ = 2 * x)) :=
begin
  sorry
end

end largest_lambda_property_l202_202040


namespace subset_count_with_no_sum_11_l202_202804

theorem subset_count_with_no_sum_11 :
  let S := ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ)
  in Finset.filter (λ A : Finset ℕ, ∀ {a b}, a ∈ A → b ∈ A → a + b ≠ 11) (S.powerset).card = 243 := by 
  sorry

end subset_count_with_no_sum_11_l202_202804


namespace parallel_lines_angle_sum_l202_202156

noncomputable def angle_sum_triangle (angle_A angle_C : ℝ) : ℝ :=
  180 - angle_A - angle_C

theorem parallel_lines_angle_sum
  (A C EDF ABD BDF : ℝ)
  (hA : A = 50)
  (hC : C = 40)
  (hEDF : EDF = 30)
  (h_parallel : ABD = EDF)
  : BDF = 60 :=
by
  let B := angle_sum_triangle A C
  have hB : B = 90 := by
    simp [angle_sum_triangle, hA, hC]
  have hABD : ABD = 30 := by
    simp [h_parallel, hEDF]
  have hBDF : BDF = B - ABD := by
    simp [hB, hABD]
  exact hBDF

end parallel_lines_angle_sum_l202_202156


namespace angle_B_of_right_triangle_l202_202981

theorem angle_B_of_right_triangle (B C : ℝ) (hA : A = 90) (hC : C = 3 * B) (h_sum : A + B + C = 180) : B = 22.5 :=
sorry

end angle_B_of_right_triangle_l202_202981


namespace magic_square_A_plus_E_l202_202270

-- Given conditions
variables {A B C D E : ℤ}
def odd_integers := [1, 3, 5, 7, 9, 11, 13, 15, 17]
def magic_sum := 27

-- Structure of the magic square with additional constraints
def magic_square (A B C D E : ℤ) : Prop :=
  A ∈ odd_integers ∧ B ∈ odd_integers ∧ C ∈ odd_integers ∧ D ∈ odd_integers ∧ E ∈ odd_integers ∧
  A + 1 + B = magic_sum ∧
  5 + C + 13 = magic_sum ∧
  D + E + 3 = magic_sum ∧
  A + 5 + D = magic_sum ∧
  1 + C + E = magic_sum ∧
  B + 13 + 3 = magic_sum ∧
  A + C + 3 = magic_sum ∧
  B + C + D = magic_sum

-- Proof goal
theorem magic_square_A_plus_E : magic_square A B C D E → A + E = 32 :=
by
  intro h
  sorry

end magic_square_A_plus_E_l202_202270


namespace area_of_rectangle_l202_202343

theorem area_of_rectangle (a b c: ℕ) (h_a: a = 15) (h_c: c = 17)
  (h_diag: a^2 + b^2 = c^2) : a * b = 120 :=
by
  rw [h_a, h_c] at h_diag
  have h1: 15^2 + b^2 = 17^2 := h_diag
  have h2: 225 + b^2 = 289 := by norm_num [(15:ℕ)^2, (17:ℕ)^2]
  have h3: b^2 = 64 := by linarith
  have h_b: b = 8 := nat.eq_of_sq_eq_sq h3
  rw [h_a, h_b]
  norm_num

end area_of_rectangle_l202_202343


namespace regular_nine_sided_polygon_diagonals_l202_202762

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202762


namespace diagonals_in_regular_nine_sided_polygon_l202_202569

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202569


namespace scattered_blue_notes_l202_202854

def rows_of_red_notes : ℕ := 5
def notes_per_row : ℕ := 6
def blue_notes_under_each_red : ℕ := 2
def total_notes_in_bins : ℕ := 100

theorem scattered_blue_notes :
  let total_red_notes := rows_of_red_notes * notes_per_row in
  let total_blue_notes_under_red := total_red_notes * blue_notes_under_each_red in
  let total_non_scattered_notes := total_red_notes + total_blue_notes_under_red in
  total_notes_in_bins - total_non_scattered_notes = 10 :=
by
  sorry

end scattered_blue_notes_l202_202854


namespace regular_nine_sided_polygon_diagonals_l202_202680

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202680


namespace nine_sided_polygon_diagonals_l202_202652

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202652


namespace sqrt_0_0001_sqrt_10_pow_6_decimal_point_relationship_sqrt_0_001_y_in_terms_of_x_l202_202467

-- Part 1: sqrt 0.0001 and sqrt 10^6 calculations
theorem sqrt_0_0001 : Real.sqrt 0.0001 = 0.01 := 
  by sorry

theorem sqrt_10_pow_6 : Real.sqrt (10^6) = 1000 := 
  by sorry

-- Part 2: Relationship between movement of decimal point in a and sqrt(a)
theorem decimal_point_relationship (a : ℝ) (h : a > 0) (n : ℝ) : 
  (Real.sqrt a) = Real.sqrt (10 ^ (2 * n)) → (Real.sqrt a) = 10 ^ n :=
  by sorry

-- Part 3: Further calculations using the rule
theorem sqrt_0_001 (h_sqrt_0_1 : Real.sqrt 0.1 = 0.316) : Real.sqrt 0.001 ≈ 0.0316 :=
  by sorry

theorem y_in_terms_of_x (x y : ℝ) (h_sqrt_x : Real.sqrt x = 1.414) (h_sqrt_y : Real.sqrt y = 141.4) : 
  y = 10000 * x := 
  by sorry

end sqrt_0_0001_sqrt_10_pow_6_decimal_point_relationship_sqrt_0_001_y_in_terms_of_x_l202_202467


namespace diagonals_in_regular_nine_sided_polygon_l202_202565

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202565


namespace vertex_coloring_l202_202350

-- Proof Problem 1
theorem vertex_coloring (m : ℕ) : 
  ∃ n, n = (1 / 24 : ℚ) * (m ^ 6 + 3 * m ^ 4 + 12 * m ^ 3 + 8 * m ^ 2) :=
by
  let numColorings := (1 / 24 : ℚ) * (m ^ 6 + 3 * m ^ 4 + 12 * m ^ 3 + 8 * m ^ 2)
  use numColorings
  sorry

end vertex_coloring_l202_202350


namespace particular_solution_l202_202435

theorem particular_solution (y : ℝ → ℝ) (h : y'' - 3 * y' + 2 * y = 0) 
    (h₀ : y 0 = 1) (h₁ : deriv y 0 = -1) : y = λ x, 3 * exp x - 2 * exp (2 * x) := 
sorry

end particular_solution_l202_202435


namespace regular_nine_sided_polygon_diagonals_l202_202757

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202757


namespace cos_C_value_l202_202826

noncomputable theory
open Real

-- Define the given conditions
def cos_A : ℝ := 4 / 5
def cos_B : ℝ := 5 / 13

-- Define the angles such that they sum to pi
axiom angle_sum : ∀ (A B C : ℝ), A + B + C = π

-- Prove the value of cos C given the conditions
theorem cos_C_value 
  (A B C : ℝ)
  (hcosA : cos A = cos_A)
  (hcosB : cos B = cos_B)
  (hsum : angle_sum A B C) :
  cos C = 16 / 65 :=
by
  have sin_A := sqrt (1 - cos_A ^ 2)
  have sin_B := sqrt (1 - cos_B ^ 2)
  rw [← hsA] at sin_A
  rw [← hsB] at sin_B
  have cos_A_value : sin A = sin_A := sorry  -- derived from cos A value
  have cos_B_value : sin B = sin_B := sorry  -- derived from cos B value
  rw [cos_sub, hsA, hsB]
  field_simp
  linarith

end cos_C_value_l202_202826


namespace six_digit_palindrome_count_l202_202016

theorem six_digit_palindrome_count : 
  let palindromes_of_form := ∃ a b c : ℕ, a ≠ 0 ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9)
  in ∃ n : ℕ, palindromes_of_form ∧ n = 900 :=
begin
  sorry
end

end six_digit_palindrome_count_l202_202016


namespace nine_sided_polygon_diagonals_l202_202659

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202659


namespace intervals_of_monotonicity_range_of_a_given_zeros_l202_202512

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) := x^2 * ℯ^(-a * x) - 1

-- Conditions
variables (a x : ℝ)

-- Prove the intervals of monotonicity for y=f(x)
theorem intervals_of_monotonicity :
  (a = 0 → (∀ x ∈ Ioi (0 : ℝ), f 0 x ≤ f 0 x) ∧ (∀ x ∈ Iio 0, f 0 x ≥ f 0 x)) ∧
  (a > 0 → (∀ x ∈ (Ioi (frac (2) (a))), deriv (f a) x ≤ 0) ∧
    (∀ x ∈ Icc 0 (frac (2) (a)), deriv (f a) x ≥ 0)) ∧
  (a < 0 → (∀ x ∈ Ioi (0 : ℝ), deriv (f a) x ≥ 0) ∧
    (∀ x ∈ Iio (frac (2) (a)), deriv (f a) x ≥ 0) ∧
    (∀ x ∈ Icc (frac (2) (a)) (0 : ℝ), deriv (f a) x ≤ 0)) :=
by sorry

-- Prove the range of values for a given x in (0,16) and f(x) has two zeros
theorem range_of_a_given_zeros :
  x ∈ Ioo 0 16 → f a x = 0 → (frac (1) (2) * log 2 < a ∧ a < frac (2) (ℯ)) :=
by sorry

end intervals_of_monotonicity_range_of_a_given_zeros_l202_202512


namespace unit_digit_of_sum_of_factorials_upto_2012_l202_202140

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def sum_of_factorials_upto (n : ℕ) : ℕ :=
(1 + factorial 2 + factorial 3 + factorial 4) % 10

theorem unit_digit_of_sum_of_factorials_upto_2012 : sum_of_factorials_upto 2012 = 3 :=
sorry

end unit_digit_of_sum_of_factorials_upto_2012_l202_202140


namespace GA_tangent_to_K_at_A_l202_202068

-- Definition of the problem

variables {K : Type*} [metric_space K] [normed_group K] [normed_space ℝ K] 
variables (circle K : set K) (C D A B E F G : K) (CD_diam : line_through C D = K)
variables (AB_par_CD : parallel A B C D) (AE_par_CB : parallel A E C B)
variables (AB_inter_DE : intersect_lines A B D E F) (FG_par_CB : parallel F G C B)
variables (tangent : tangent_at_point G A K)

-- Main theorem, to prove GA is tangent to circle K at A
theorem GA_tangent_to_K_at_A :
  tangent_line G A K := 
sorry

end GA_tangent_to_K_at_A_l202_202068


namespace diagonals_in_nine_sided_polygon_l202_202738

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202738


namespace sin_alpha_plus_beta_eq_33_by_65_l202_202076

theorem sin_alpha_plus_beta_eq_33_by_65 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (hcosα : Real.cos α = 12 / 13) 
  (hcos_2α_β : Real.cos (2 * α + β) = 3 / 5) :
  Real.sin (α + β) = 33 / 65 := 
by 
  sorry

end sin_alpha_plus_beta_eq_33_by_65_l202_202076


namespace sum_of_last_three_digits_in_fib_factorial_series_l202_202330

-- Define the Fibonacci sequence and the specific factorials we need
def fib_seq : List ℕ := [1, 2, 3, 5, 8, 13, 21]

-- Function to compute the last three digits of a factorial
def last_three_digits (n : ℕ) : ℕ :=
  (Nat.factorial n) % 1000

-- Function to sum the last three digits of the specific factorials in the sequence
def sum_last_three_digits : ℕ :=
  fib_seq.map last_three_digits |>.sum % 1000

theorem sum_of_last_three_digits_in_fib_factorial_series :
  sum_last_three_digits = 249 :=
by
  sorry

end sum_of_last_three_digits_in_fib_factorial_series_l202_202330


namespace arccos_lt_arctan_in_interval_l202_202030

noncomputable def arccos_lt_arctan : Prop :=
  ∃ a : ℝ, 0 < a ∧ a < 1 ∧ ∀ x : ℝ, (a < x ∧ x ≤ 1) → arccos x < arctan x

-- Here we write our theorem statement
theorem arccos_lt_arctan_in_interval (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (x : ℝ) (hx : a < x ∧ x ≤ 1) : 
  arccos x < arctan x := 
sorry

end arccos_lt_arctan_in_interval_l202_202030


namespace sum_of_first_10_common_elements_eq_13981000_l202_202449

def arithmetic_prog (n : ℕ) : ℕ := 4 + 3 * n
def geometric_prog (k : ℕ) : ℕ := 20 * 2 ^ k

theorem sum_of_first_10_common_elements_eq_13981000 :
  let common_elements : List ℕ := 
    [40, 160, 640, 2560, 10240, 40960, 163840, 655360, 2621440, 10485760]
  let sum_common_elements : ℕ := common_elements.sum
  sum_common_elements = 13981000 := by
  sorry

end sum_of_first_10_common_elements_eq_13981000_l202_202449


namespace nine_sided_polygon_diagonals_l202_202582

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202582


namespace monotonically_increasing_interval_symmetry_center_minimum_a_l202_202508

noncomputable def f (x : ℝ) : ℝ :=
  sin x * cos (x - π / 6) + cos x ^ 2 - 1 / 2

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x, f (x) = 1 / 2 → (kπ - π / 3) ≤ x ∧ x ≤ (kπ + π / 6) := sorry

theorem symmetry_center (k : ℤ) :
  ∃ x, f (x) = 1 / 4 ∧ x = kπ / 2 - π / 12 := sorry

theorem minimum_a (A a b c : ℝ) :
  A = π / 3 → b + c = 3 → a^2 = b^2 + c^2 - 2 * b * c * cos A →
  a ≥ 3 / 2 := sorry

end monotonically_increasing_interval_symmetry_center_minimum_a_l202_202508


namespace nine_sided_polygon_diagonals_l202_202581

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202581


namespace sum_of_paintable_integers_l202_202524

noncomputable def is_paintable (h t u v : ℕ) : Prop :=
  h > 1 ∧ t > 1 ∧ u > 1 ∧ v > 1 ∧
  (∀ n : ℕ, ∃ k1 k2 k3 k4 : ℕ, 
      n = 1 + k1 * h ∨ n = 2 + k2 * t ∨ n = 3 + k3 * u ∨ n = 4 + k4 * v)

theorem sum_of_paintable_integers : ∑ (h t u v : ℕ) in { (4, 8, 4, 12) }, 1000 * h + 100 * t + 10 * u + v = 4812 :=
by
  sorry

end sum_of_paintable_integers_l202_202524


namespace number_of_diagonals_l202_202557

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202557


namespace find_fraction_l202_202145

variable (F N : ℚ)

-- Defining the conditions
def condition1 : Prop := (1 / 3) * F * N = 18
def condition2 : Prop := (3 / 10) * N = 64.8

-- Proof statement
theorem find_fraction (h1 : condition1 F N) (h2 : condition2 N) : F = 1 / 4 := by 
  sorry

end find_fraction_l202_202145


namespace pyramid_volume_l202_202495

theorem pyramid_volume (a : ℝ) (h : a > 0) : (1 / 6) * a^3 = 1 / 6 * a^3 :=
by
  sorry

end pyramid_volume_l202_202495


namespace ratio_of_areas_l202_202241

-- Definitions based on conditions
def square_A_perimeter : ℝ := 16
def square_B_perimeter : ℝ := 32

-- Definitions for side lengths of squares
def side_length_of_square (perimeter : ℝ) : ℝ := perimeter / 4

def side_length_A : ℝ := side_length_of_square square_A_perimeter
def side_length_B : ℝ := side_length_of_square square_B_perimeter
def side_length_C : ℝ := 2 * side_length_A -- given condition for side length of square C

-- Definitions for area of squares
def area_of_square (side_length : ℝ) : ℝ := side_length ^ 2

def area_A : ℝ := area_of_square side_length_A
def area_C : ℝ := area_of_square side_length_C

-- Proving the ratio of the areas
theorem ratio_of_areas : 
  (area_A / area_C) = (1 / 4) :=
sorry

end ratio_of_areas_l202_202241


namespace possible_value_of_2n_plus_m_l202_202177

variable (n m : ℤ)

theorem possible_value_of_2n_plus_m : (3 * n - m < 5) → (n + m > 26) → (3 * m - 2 * n < 46) → (2 * n + m = 36) :=
by
  sorry

end possible_value_of_2n_plus_m_l202_202177


namespace regular_nine_sided_polygon_diagonals_l202_202678

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202678


namespace nine_sided_polygon_diagonals_l202_202630

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202630


namespace joe_total_cars_l202_202190

def initial_cars := 50
def multiplier := 3

theorem joe_total_cars : initial_cars + (multiplier * initial_cars) = 200 := by
  sorry

end joe_total_cars_l202_202190


namespace correct_conclusions_count_l202_202946

theorem correct_conclusions_count :
  let concl1 := (2 ≠ -2)   -- Incorrect, base is 2 (True statement implies incorrect conclusion)
  let concl2 := (∀ a b : ℚ, a = -b → a + b = 0) -- Correct
  let concl3 := (Float.round_nearest 1.804 0.01 = 1.80) -- Correct
  let concl4 := (∀ a b : ℚ, (5 * a - 3 * b) - 3 * (a ^ 2 - 2 * b) = -3 * (a ^ 2) + 5 * a + 3 * b) -- Correct
  let concl5 := (∀ a : ℝ, |a + 2| + 6 ≤ 6) -- Incorrect
  (concl1 = false) + (concl2 = true) + (concl3 = true) + (concl4 = true) + (concl5 = false) = 3 :=
by
  sorry

end correct_conclusions_count_l202_202946


namespace possible_value_of_2n_plus_m_l202_202176

variable (n m : ℤ)

theorem possible_value_of_2n_plus_m : (3 * n - m < 5) → (n + m > 26) → (3 * m - 2 * n < 46) → (2 * n + m = 36) :=
by
  sorry

end possible_value_of_2n_plus_m_l202_202176


namespace diagonals_in_regular_nine_sided_polygon_l202_202539

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202539


namespace number_of_diagonals_l202_202558

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202558


namespace nine_sided_polygon_diagonals_l202_202588

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202588


namespace regular_nine_sided_polygon_diagonals_l202_202751

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202751


namespace negation_of_original_l202_202965

-- Defining N as the set of natural numbers
def N := Nat

-- Original statement
def original_statement := ∀ x : N, x ≥ 0

-- Negated statement
def negated_statement := ∃ x : N, x < 0

-- The theorem stating the equivalence of the negation process
theorem negation_of_original : ¬original_statement ↔ negated_statement := 
  by sorry

end negation_of_original_l202_202965


namespace range_of_a_l202_202517

-- Definitions of the conditions
def is_on_parabola (x y : ℝ) : Prop := y = x^2

def on_both_sides_y_axis (x1 x2 : ℝ) : Prop := x1 * x2 < 0

def intersection_at_y_axis (a k : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = k ∧ x1 * x2 = -a

def acute_angle_between_vectors (a : ℝ) (x1 x2 : ℝ) : Prop :=
  let y1 := x1^2 in
  let y2 := x2^2 in
  (x1 * x2 + y1 * y2 > 0)

-- Proof Problem
theorem range_of_a (a k x1 x2 : ℝ) 
    (hx1 : is_on_parabola x1 (x1^2))
    (hx2 : is_on_parabola x2 (x2^2))
    (h_sides : on_both_sides_y_axis x1 x2)
    (h_intersect : intersection_at_y_axis a k x1 x2)
    (h_acute : acute_angle_between_vectors a x1 x2) :
    1 < a := 
sorry

end range_of_a_l202_202517


namespace diagonals_in_nonagon_l202_202727

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202727


namespace regular_nine_sided_polygon_diagonals_l202_202673

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202673


namespace product_of_p_r_s_l202_202806

theorem product_of_p_r_s
  (p r s : ℕ)
  (h1 : 3^p + 3^4 = 90)
  (h2 : 2^r + 44 = 76)
  (h3 : 5^3 + 6^s = 1421) :
  p * r * s = 40 := 
sorry

end product_of_p_r_s_l202_202806


namespace regular_nine_sided_polygon_diagonals_l202_202761

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202761


namespace regular_nonagon_diagonals_correct_l202_202708

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202708


namespace smallest_odd_five_digit_tens_place_l202_202988

-- Definitions
def digits : list ℕ := [1, 2, 3, 5, 8]

def is_smallest_odd_five_digit (n : ℕ) : Prop :=
  ∀ m, 
    (m < n ∧ (∀ d ∈ digits, m ≠ read_digits d)) → m % 2 = 1 → false

def read_digits (d : list ℕ) : ℕ :=
d.foldr (λ a b, a + b * 10) 0

-- The main theorem
theorem smallest_odd_five_digit_tens_place :
  ∃ n, is_smallest_odd_five_digit n ∧ 
       (tens_place n = 8) :=
sorry

-- Helper function to extract the tens place digit
def tens_place (n : ℕ) : ℕ :=
(n / 10) % 10

end smallest_odd_five_digit_tens_place_l202_202988


namespace smallest_b_l202_202937

theorem smallest_b (a b : ℕ) (hp : a > 0) (hq : b > 0) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 8) : b = 4 :=
sorry

end smallest_b_l202_202937


namespace sequence_converges_to_one_l202_202529

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 0 = Real.sqrt 2 ∧ ∀ n, a (n + 1) = Real.sqrt (2 - a n)

theorem sequence_converges_to_one (a : ℕ → ℝ) (h : sequence a) : 
  ∃ L, L = 1 ∧ filter.tendsto a filter.at_top (filter.nhds L) :=
sorry

end sequence_converges_to_one_l202_202529


namespace eval_x_power_x_power_x_at_3_l202_202023

theorem eval_x_power_x_power_x_at_3 : (3^3)^(3^3) = 27^27 := by
    sorry

end eval_x_power_x_power_x_at_3_l202_202023


namespace cos_180_eq_neg1_l202_202006

theorem cos_180_eq_neg1 : cos (Real.pi) = -1 :=
by sorry

end cos_180_eq_neg1_l202_202006


namespace third_generation_tail_length_l202_202181

theorem third_generation_tail_length (tail_length : ℕ → ℕ) (h0 : tail_length 0 = 16)
    (h_next : ∀ n, tail_length (n + 1) = tail_length n + (25 * tail_length n) / 100) :
    tail_length 2 = 25 :=
by
  sorry

end third_generation_tail_length_l202_202181


namespace donuts_purchased_l202_202192

/-- John goes to a bakery every day for a four-day workweek and chooses between a 
    60-cent croissant or a 90-cent donut. At the end of the week, he spent a whole 
    number of dollars. Prove that he must have purchased 2 donuts. -/
theorem donuts_purchased (d c : ℕ) (h1 : d + c = 4) (h2 : 90 * d + 60 * c % 100 = 0) : d = 2 :=
sorry

end donuts_purchased_l202_202192


namespace complex_number_solution_l202_202064

def i : ℂ := Complex.I

theorem complex_number_solution (z : ℂ) (h : z * (1 - i) = 2 * i) : z = -1 + i :=
by
  sorry

end complex_number_solution_l202_202064


namespace diagonals_in_nine_sided_polygon_l202_202778

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202778


namespace max_trig_expression_l202_202042

open Real

theorem max_trig_expression (x y z : ℝ) :
  (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 := sorry

end max_trig_expression_l202_202042


namespace original_price_of_cupcakes_l202_202014

theorem original_price_of_cupcakes
  (revenue : ℕ := 32) 
  (cookies_sold : ℕ := 8) 
  (cupcakes_sold : ℕ := 16) 
  (cookie_price: ℕ := 2)
  (half_price_of_cookie: ℕ := 1) :
  (x : ℕ) → (16 * (x / 2)) + (8 * 1) = 32 → x = 3 := 
by
  sorry

end original_price_of_cupcakes_l202_202014


namespace regular_nonagon_diagonals_correct_l202_202701

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202701


namespace conjugate_in_third_quadrant_l202_202002

theorem conjugate_in_third_quadrant : 
 ∀ z : ℂ, z = -2 / I + 3 / I * I → complex.conj z.re < 0 ∧ complex.conj z.im < 0 :=
by
  sorry

end conjugate_in_third_quadrant_l202_202002


namespace correct_calculation_l202_202335

theorem correct_calculation (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end correct_calculation_l202_202335


namespace polynomial_factors_count_l202_202415

theorem polynomial_factors_count (x : ℤ) : 
  ∃ (n : ℕ), n = 5 ∧ 
  (∃ (p1 p2 p3 p4 p5 : polynomial ℤ),
      x^15 - x = p1 * p2 * p3 * p4 * p5) := 
by
  use 5
  split
  · rfl
  · use [polynomial.X, polynomial.X - 1, polynomial.X + 1, (polynomial.X^6 + polynomial.X^5 + polynomial.X^4 + polynomial.X^3 + polynomial.X^2 + polynomial.X + 1), (polynomial.X^6 - polynomial.X^5 + polynomial.X^4 - polynomial.X^3 + polynomial.X^2 - polynomial.X + 1)]
  sorry

end polynomial_factors_count_l202_202415


namespace problem_statement_l202_202360

noncomputable def find_k (l : ℝ → ℝ) (k : ℝ) : Prop :=
∃ (A B : ℝ × ℝ), 
  (l = λ x, k * x + B.2) ∧ 
  (A.1 ^ 2 = 2 * A.2) ∧ 
  (B.1 ^ 2 = 2 * B.2) ∧ 
  ((A.1 + B.1) / 2 = 1) ∧ 
  k = 1

noncomputable def find_line_eq (l : ℝ → ℝ) (k : ℝ) (m : ℝ) : Prop :=
∃ (A B C D : ℝ × ℝ), 
  k = 1 ∧ 
  (l = λ x, k * x + m) ∧ 
  (A.1 ^ 2 = 2 * A.2) ∧ 
  (B.1 ^ 2 = 2 * B.2) ∧ 
  (A.1 + B.1) / 2 = 1 ∧ 
  ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 12 * 2 - m ^ 2 / 2) ∧
  ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 = (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) ∧
  (C.1 ^ 2 + C.2 ^ 2 = 12) ∧
  (D.1 ^ 2 + D.2 ^ 2 = 12) ∧ 
  l = λ x, x + 2

theorem problem_statement : ∃ (l : ℝ → ℝ) (k m : ℝ), find_k l k ∧ find_line_eq l k m :=
  sorry

end problem_statement_l202_202360


namespace factorial_sum_mod_12_l202_202824

theorem factorial_sum_mod_12 :
  (1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9! + 10!) % 12 = 9 := 
by sorry

end factorial_sum_mod_12_l202_202824


namespace coeff_x3_in_expansion_l202_202441

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coeff_x3_in_expansion : 
  (∃ c : ℕ, c = 160 ∧ (x : ℕ → ℕ) ∃ x^3 == (binomial 6 3 * 2^3))  := by
sorry

end coeff_x3_in_expansion_l202_202441


namespace part_a_part_b_l202_202315

-- Define the subaveraging condition for a sequence
def subaveraging (s : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, s n = (s (n - 1) + s (n + 1)) / 4

-- Part (a): Prove that the sequence s_0 = 0, s_1 = 1, and s_{n+1} = 4s_n - s_{n-1} has all distinct elements
theorem part_a :
  ∀ (s : ℤ → ℝ),
  (s 0 = 0) ∧ (s 1 = 1) ∧ (∀ n : ℤ, s (n + 1) = 4 * s n - s (n - 1))
  → (∀ i j : ℤ, i ≠ j → s i ≠ s j) :=
begin
  sorry
end

-- Part (b): If a subaveraging sequence has s_m = s_n for some distinct integers m, n, then there are infinitely many pairs of distinct integers i, j with s_i = s_j
theorem part_b :
  ∀ (s : ℤ → ℝ),
  subaveraging s
  → ∀ m n : ℤ, m ≠ n → s m = s n → (∃ i j : ℤ, i ≠ j ∧ s i = s j) :=
begin
  sorry
end

end part_a_part_b_l202_202315


namespace diagonals_in_regular_nine_sided_polygon_l202_202575

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202575


namespace sum_of_squares_of_sides_l202_202877

def A := (x1 y1 : ℝ)
def B := (x2 y2 : ℝ)
def C := (x3 y3 : ℝ)
def G := (((x1 + x2 + x3) / 3), ((y1 + y2 + y3) / 3))

def dist_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2

theorem sum_of_squares_of_sides {x1 y1 x2 y2 x3 y3 : ℝ} 
  (h : dist_sq G A + dist_sq G B + dist_sq G C = 72) : 
  dist_sq A B + dist_sq A C + dist_sq B C = 216 :=
sorry

end sum_of_squares_of_sides_l202_202877


namespace diagonals_in_nonagon_l202_202723

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202723


namespace divisor_exists_l202_202973

theorem divisor_exists (n : ℕ) : (∃ k, 10 ≤ k ∧ k ≤ 50 ∧ n ∣ k) →
                                (∃ k, 10 ≤ k ∧ k ≤ 50 ∧ n ∣ k) ∧
                                (n = 3) :=
by
  sorry

end divisor_exists_l202_202973


namespace trapezoid_area_110_l202_202847

theorem trapezoid_area_110
  (AB CD : ℝ) (AC BD : Type*) (E : AC → BD → Prop)
  (area_abe area_ade : ℝ)
  (ratio_de_be : ℝ)
  (h_AB_parallel_CD : AB = CD)
  (h_diag_intersect_E : E AC BD)
  (h_area_abe : area_abe = 45)
  (h_area_ade : area_ade = 15)
  (h_ratio_de_be : ratio_de_be = 1 / 3) : 
  (∃ area_abcd : ℝ, area_abcd = 110) := 
begin
  sorry
end

end trapezoid_area_110_l202_202847


namespace nine_sided_polygon_diagonals_count_l202_202693

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202693


namespace roots_on_circle_l202_202891

def polynomial (a b c : ℝ) : ℂ[X] := X^6 + a*X^4 + b*X^2 + c

theorem roots_on_circle (a b c : ℝ) (P : ℂ[X]) (hP : P = polynomial a b c)
  (h_roots : ∃ (Γ : ℂ → Prop), (∀ z : ℂ, P.eval z = 0 → Γ z) ∧ Γ = (λ z, ∥z∥ = r) ∧ is_circle_unique Γ) :
  b^3 = a^3 * c := by
  sorry

end roots_on_circle_l202_202891


namespace coefficient_x2_term_in_expansion_l202_202037

theorem coefficient_x2_term_in_expansion :
  let p1 := (2 : ℤ) * X^3 + (4 : ℤ) * X^2 + (5 : ℤ) * X - (3 : ℤ)
  let p2 := (6 : ℤ) * X^2 + (-5 : ℤ) * X + (1 : ℤ)
  (p1 * p2).coeff 2 = -39 :=
by
  -- The proof is not required
  sorry

end coefficient_x2_term_in_expansion_l202_202037


namespace proposition_P_l202_202478

theorem proposition_P (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) := 
by 
  sorry

end proposition_P_l202_202478


namespace intersection_points_zero_l202_202812

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem intersection_points_zero
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h_gp : geometric_sequence a b c)
  (h_ac_pos : a * c > 0) :
  ∃ x : ℝ, quadratic_function a b c x = 0 → false :=
by
  -- Proof to be completed
  sorry

end intersection_points_zero_l202_202812


namespace nine_sided_polygon_diagonals_l202_202631

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202631


namespace negation_of_universal_proposition_l202_202081

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 2 → x^3 - 8 > 0)) ↔ (∃ x : ℝ, x > 2 ∧ x^3 - 8 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l202_202081


namespace perfect_squares_difference_count_l202_202925

def is_perfect_square (k : ℕ) : Prop :=
  ∃ m : ℕ, k = m * m

def is_odd (k : ℕ) : Prop :=
  k % 2 = 1

def count_perfect_squares_less_than_20000 : ℕ :=
  (Finset.range 200).filter (λ k, is_perfect_square k ∧ k < 20000 ∧ is_odd k).card

theorem perfect_squares_difference_count :
  count_perfect_squares_less_than_20000 = 71 :=
sorry

end perfect_squares_difference_count_l202_202925


namespace green_paint_mixture_l202_202458

theorem green_paint_mixture :
  ∀ (x : ℝ), 
    let light_green_paint := 5
    let darker_green_paint := x
    let final_paint := light_green_paint + darker_green_paint
    1 + 0.4 * darker_green_paint = 0.25 * final_paint -> x = 5 / 3 := 
by 
  intros x
  let light_green_paint := 5
  let darker_green_paint := x
  let final_paint := light_green_paint + darker_green_paint
  sorry

end green_paint_mixture_l202_202458


namespace solve_for_x_l202_202248

theorem solve_for_x (x : ℝ) (hx_pos : x > 0) (h_eq : 3 * x^2 + 13 * x - 10 = 0) : x = 2 / 3 :=
sorry

end solve_for_x_l202_202248


namespace diagonals_in_nine_sided_polygon_l202_202768

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202768


namespace regular_nine_sided_polygon_diagonals_l202_202753

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202753


namespace max_cross_section_area_l202_202373

def square_cross_section (length : ℝ) (vertex : (ℝ × ℝ × ℝ)) : Prop :=
  -- Definition of vertices A, B, C, D base on length and centroid
  let A : ℝ × ℝ × ℝ := (length / 2, 0, 0) in
  let B : ℝ × ℝ × ℝ := (0, length / 2, 0) in
  let C : ℝ × ℝ × ℝ := (- (length / 2), 0, 0) in
  let D : ℝ × ℝ × ℝ := (0, - (length / 2), 0) in
  vertex = A ∨ vertex = B ∨ vertex = C ∨ vertex = D

def prism_side_length := 12

def is_plane_intersection (vertex : (ℝ × ℝ × ℝ)) : Prop :=
  let (x, y, z) := vertex in
  3 * x - 6 * y + 2 * z = 24

theorem max_cross_section_area :
  ∀ (vertex : (ℝ × ℝ × ℝ)), 
    (square_cross_section prism_side_length vertex ∧ is_plane_intersection vertex) → 
    -- Assuming correct formula for area computation
    (vertex_reach_area vertex = 144) := 
sorry

end max_cross_section_area_l202_202373


namespace nine_sided_polygon_diagonals_l202_202654

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202654


namespace find_abc_l202_202414

theorem find_abc (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≤ b ∧ b ≤ c) (h5 : a + b + c + a * b + b * c + c * a = a * b * c + 1) :
  (a = 2 ∧ b = 5 ∧ c = 8) ∨ (a = 3 ∧ b = 4 ∧ c = 13) :=
sorry

end find_abc_l202_202414


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202611

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202611


namespace arc_length_of_curve_l202_202346

-- Define the parametric functions x(t) and y(t)
def x (t : ℝ) : ℝ := (t^2 - 2) * Real.sin t + 2 * t * Real.cos t
def y (t : ℝ) : ℝ := (2 - t^2) * Real.cos t + 2 * t * Real.sin t

-- Define the derivatives x'(t) and y'(t)
def x' (t : ℝ) : ℝ := t^2 * Real.cos t
def y' (t : ℝ) : ℝ := t^2 * Real.sin t

-- Define the integrand for the arc length formula
def integrand (t : ℝ) : ℝ := Real.sqrt ((x' t) ^ 2 + (y' t) ^ 2)

-- State the problem to prove
theorem arc_length_of_curve : (∫ t in (0:ℝ)..(2 * Real.pi : ℝ), t^2) = 8 * Real.pi^3 / 3 :=
by
  sorry

end arc_length_of_curve_l202_202346


namespace find_a_l202_202433

theorem find_a (a : ℚ) (h : ∃ r s : ℚ, (r*x + s)^2 = ax^2 + 18*x + 16) : a = 81 / 16 := 
by sorry 

end find_a_l202_202433


namespace arccos_less_arctan_l202_202033

theorem arccos_less_arctan {x : ℝ} (hx : -1 ≤ x ∧ x ≤ 1) :
  ∃ α ∈ Icc (-1 : ℝ) (1 : ℝ), α ≈ (1/2 : ℝ) ∧ (x > α → arccos x < arctan x) := by
sorry

end arccos_less_arctan_l202_202033


namespace regular_nonagon_diagonals_correct_l202_202711

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202711


namespace sum_of_triangle_areas_greater_than_pentagon_l202_202475

variable {V : Type} [real_vector_space V]

noncomputable def area (p : polygon V) : ℝ := sorry

noncomputable def triangle (a b c : V) : polygon V := sorry

noncomputable def diagonal_triangulation (pentagon : polygon V) : List (polygon V) := sorry  -- function to get list of triangles formed by diagonals

theorem sum_of_triangle_areas_greater_than_pentagon 
  (pent : polygon V) (h_pent : convex pent) (h_pent_size : size pent = 5) :
  let diagonals := diagonal_triangulation pent
  in (∑ triangle in diagonals, area triangle) > area pent := 
sorry

end sum_of_triangle_areas_greater_than_pentagon_l202_202475


namespace max_sum_x_y_l202_202997

theorem max_sum_x_y 
  (x y : ℝ)
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
sorry

end max_sum_x_y_l202_202997


namespace diagonals_in_nine_sided_polygon_l202_202797

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202797


namespace net_income_difference_l202_202258

-- Define Terry's and Jordan's daily income and working days
def terryDailyIncome : ℝ := 24
def terryWorkDays : ℝ := 7
def jordanDailyIncome : ℝ := 30
def jordanWorkDays : ℝ := 6

-- Define the tax rate
def taxRate : ℝ := 0.10

-- Calculate weekly gross incomes
def terryGrossWeeklyIncome : ℝ := terryDailyIncome * terryWorkDays
def jordanGrossWeeklyIncome : ℝ := jordanDailyIncome * jordanWorkDays

-- Calculate tax deductions
def terryTaxDeduction : ℝ := taxRate * terryGrossWeeklyIncome
def jordanTaxDeduction : ℝ := taxRate * jordanGrossWeeklyIncome

-- Calculate net weekly incomes
def terryNetWeeklyIncome : ℝ := terryGrossWeeklyIncome - terryTaxDeduction
def jordanNetWeeklyIncome : ℝ := jordanGrossWeeklyIncome - jordanTaxDeduction

-- Calculate the difference
def incomeDifference : ℝ := jordanNetWeeklyIncome - terryNetWeeklyIncome

-- The theorem to be proven
theorem net_income_difference :
  incomeDifference = 10.80 :=
by
  sorry

end net_income_difference_l202_202258


namespace diagonals_in_nine_sided_polygon_l202_202789

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202789


namespace stating_sum_first_10_common_elements_l202_202452

/-- 
  Theorem stating that the sum of the first 10 elements that appear 
  in both the given arithmetic progression and the geometric progression 
  equals 13981000.
-/
theorem sum_first_10_common_elements :
  let AP := λ n : ℕ => 4 + 3 * n in
  let GP := λ k : ℕ => 20 * 2^k in
  let common_elements := (range 10).map (λ i => GP (2 * i + 1)) in
  (∑ i in (finset.range 10), common_elements[i]) = 13981000 :=
by
  let AP := λ n : ℕ => 4 + 3 * n
  let GP := λ k : ℕ => 20 * 2^k
  let common_elements := (finset.range 10).map (λ i => GP (2 * i + 1))
  have S : (∑ i in (finset.range 10), common_elements[i]) = 40 * (4^10 - 1) / 3,
  {
    sorry,
  }
  have : 40 * 349525 = 13981000 := by norm_num,
  exact this ▸ S

end stating_sum_first_10_common_elements_l202_202452


namespace ratio_children_to_adults_l202_202383

variable (male_adults : ℕ) (female_adults : ℕ) (total_people : ℕ)
variable (total_adults : ℕ) (children : ℕ)

theorem ratio_children_to_adults :
  male_adults = 100 →
  female_adults = male_adults + 50 →
  total_people = 750 →
  total_adults = male_adults + female_adults →
  children = total_people - total_adults →
  children / total_adults = 2 :=
by
  intros h_male h_female h_total h_adults h_children
  sorry

end ratio_children_to_adults_l202_202383


namespace part_one_magnitude_part_two_perpendicular_l202_202169

variables (a b : ℝ × ℝ) (m n : ℝ × ℝ) (t : ℝ)

-- Conditions
def a := (1, 1)
def b := (2, -1)
def m := 2 • a - b
def n := t • a + b

-- Part (1): Prove the magnitude of 3a - b is sqrt(17)
theorem part_one_magnitude :
  ‖3 • a - b‖ = sqrt 17 :=
sorry

-- Part (2): Prove that if m ⟂ n, then t = 1
theorem part_two_perpendicular (h : m ⟂ n) :
  t = 1 :=
sorry

end part_one_magnitude_part_two_perpendicular_l202_202169


namespace regular_nonagon_diagonals_correct_l202_202702

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202702


namespace f_4_eq_16_l202_202501

def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≥ 0 then x^a else |x - 2|

theorem f_4_eq_16 (a : ℝ) (h : f (-2) a = f 2 a) : f 4 a = 16 :=
by
  /- Define the value of 'f' at specific points based on provided conditions -/
  have h1 : f (-2) a = |(-2 : ℝ) - 2| := by simp [f, abs]
  have h2 : |(-2 : ℝ) - 2| = 4 := by norm_num
  have h3 : f (2) a = (2 : ℝ)^a := by simp [f]
  have h4 : (2 : ℝ)^a = 4 := by rw [←h, h1, h2]
  have a_val : a = 2 := by linarith
  rw [a_val] at *
  simp [f, real.pow_two]
  norm_num
  sorry

end f_4_eq_16_l202_202501


namespace number_of_diagonals_l202_202550

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202550


namespace exist_monic_polynomials_l202_202887

theorem exist_monic_polynomials (n : ℕ) (P : RealPolynomial) 
    (hP : P.degree = n ∧ P.leading_coeff = 1) :
    ∃ Q R : RealPolynomial, 
    (Q.degree = n ∧ Q.leading_coeff = 1) ∧ 
    (R.degree = n ∧ R.leading_coeff = 1) ∧
    (∀ x : ℝ, Q.eval x = 0 → Real.root_multiplicity x Q = 1) ∧
    (∀ x : ℝ, R.eval x = 0 → Real.root_multiplicity x R = 1) ∧
    (∀ x : ℝ, (P.eval x) = (Q.eval x + R.eval x) / 2) :=
sorry

end exist_monic_polynomials_l202_202887


namespace probability_not_adjacent_irrational_terms_l202_202024

noncomputable def binomial_expansion := (x + 2 / (Real.sqrt x)) ^ 6

def general_term (n k : ℕ) := 
  (Nat.choose n k) * (x ^ (n - k)) * ((2 / (Real.sqrt x)) ^ k)

def is_rational (n k : ℕ) := 
  2 * k ∈ [0, 2, 4, 6]

def total_basic_events := 7.factorial

def arrangements_non_adjacent_irrational :=
  4.factorial * 5.choose 3

def probability_non_adjacent_irrational := 
  arrangements_non_adjacent_irrational / total_basic_events

theorem probability_not_adjacent_irrational_terms : 
  probability_non_adjacent_irrational = 2 / 7 :=
sorry

end probability_not_adjacent_irrational_terms_l202_202024


namespace sum_of_first_10_common_elements_eq_13981000_l202_202448

def arithmetic_prog (n : ℕ) : ℕ := 4 + 3 * n
def geometric_prog (k : ℕ) : ℕ := 20 * 2 ^ k

theorem sum_of_first_10_common_elements_eq_13981000 :
  let common_elements : List ℕ := 
    [40, 160, 640, 2560, 10240, 40960, 163840, 655360, 2621440, 10485760]
  let sum_common_elements : ℕ := common_elements.sum
  sum_common_elements = 13981000 := by
  sorry

end sum_of_first_10_common_elements_eq_13981000_l202_202448


namespace tan_half_angle_l202_202092

theorem tan_half_angle (α : Real) (h1 : Real.sin α + Real.cos α = -3 / Real.sqrt 5)
  (h2 : Real.abs (Real.sin α) > Real.abs (Real.cos α)) :
  Real.tan (α / 2) = -((Real.sqrt 5 + 1) / 2) :=
by
  sorry

end tan_half_angle_l202_202092


namespace nine_sided_polygon_diagonals_l202_202637

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202637


namespace y_intercept_of_line_l202_202318

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 5 * y = 10) (hx : x = 0) : y = -2 :=
by {
  rw [hx, zero_mul, sub_zero] at h,
  linarith,
}

end y_intercept_of_line_l202_202318


namespace sequence_value_238_l202_202377

theorem sequence_value_238 (a : ℕ → ℚ) :
  (a 1 = 1) ∧
  (∀ n, n ≥ 2 → (n % 2 = 0 → a n = a (n - 1) / 2 + 1) ∧ (n % 2 = 1 → a n = 1 / a (n - 1))) ∧
  (∃ n, a n = 30 / 19) → ∃ n, a n = 30 / 19 ∧ n = 238 :=
by
  sorry

end sequence_value_238_l202_202377


namespace max_wind_power_speed_l202_202953

def sail_force (A S ρ v0 v: ℝ) : ℝ :=
  (A * S * ρ * (v0 - v)^2) / 2

def wind_power (A S ρ v0 v: ℝ) : ℝ :=
  (sail_force A S ρ v0 v) * v

theorem max_wind_power_speed (A ρ: ℝ) (v0: ℝ) (S: ℝ) (h: v0 = 4.8 ∧ S = 4) :
  ∃ v, (wind_power A S ρ v0 v) = max ((wind_power A S ρ v0 v)) ∧ v = 1.6 :=
begin
  sorry
end

end max_wind_power_speed_l202_202953


namespace six_digit_palindrome_count_l202_202017

theorem six_digit_palindrome_count : 
  let palindromes_of_form := ∃ a b c : ℕ, a ≠ 0 ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9)
  in ∃ n : ℕ, palindromes_of_form ∧ n = 900 :=
begin
  sorry
end

end six_digit_palindrome_count_l202_202017


namespace parallel_segments_in_equilateral_triangle_l202_202372

theorem parallel_segments_in_equilateral_triangle 
  (A B C K N L M : Point)
  (h1 : is_equilateral_triangle A B C)
  (h2 : segment K N ⊆ segment A B)
  (h3 : L ∈ segment A C)
  (h4 : M ∈ segment B C)
  (h5 : CL = AK)
  (h6 : CM = BN)
  (h7 : ML = KN) : 
  parallel KL MN :=
sorry

end parallel_segments_in_equilateral_triangle_l202_202372


namespace contractor_hourly_rate_l202_202860

def permits_cost : ℤ := 250
def total_cost : ℤ := 2950
def contractor_days : ℤ := 3
def contractor_hours_per_day : ℤ := 5
def inspector_discount : ℤ := 80

theorem contractor_hourly_rate :
  ∃ (X : ℤ), 
  let contractor_total_hours := contractor_days * contractor_hours_per_day in
  let contractor_total_cost := contractor_total_hours * X in
  let inspector_hourly_rate := X * (100 - inspector_discount) / 100 in
  let inspector_total_cost := contractor_total_hours * inspector_hourly_rate in
  let combined_cost := contractor_total_cost + inspector_total_cost in
  total_cost - permits_cost = combined_cost ∧ X = 150 :=
sorry

end contractor_hourly_rate_l202_202860


namespace diagonals_in_regular_nine_sided_polygon_l202_202572

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202572


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202604

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202604


namespace jacob_additional_money_needed_l202_202852

/-- Jacob's total trip cost -/
def trip_cost : ℕ := 5000

/-- Jacob's hourly wage -/
def hourly_wage : ℕ := 20

/-- Jacob's working hours -/
def working_hours : ℕ := 10

/-- Income from job -/
def job_income : ℕ := hourly_wage * working_hours

/-- Price per cookie -/
def cookie_price : ℕ := 4

/-- Number of cookies sold -/
def cookies_sold : ℕ := 24

/-- Income from cookies -/
def cookie_income : ℕ := cookie_price * cookies_sold

/-- Lottery ticket cost -/
def lottery_ticket_cost : ℕ := 10

/-- Lottery win amount -/
def lottery_win : ℕ := 500

/-- Money received from each sister -/
def sister_gift : ℕ := 500

/-- Total income from job and cookies -/
def income_without_expenses : ℕ := job_income + cookie_income

/-- Income after lottery ticket purchase -/
def income_after_ticket : ℕ := income_without_expenses - lottery_ticket_cost

/-- Total income after lottery win -/
def income_with_lottery : ℕ := income_after_ticket + lottery_win

/-- Total gift from sisters -/
def total_sisters_gift : ℕ := 2 * sister_gift

/-- Total money Jacob has -/
def total_money : ℕ := income_with_lottery + total_sisters_gift

/-- Additional amount needed by Jacob -/
def additional_needed : ℕ := trip_cost - total_money

theorem jacob_additional_money_needed : additional_needed = 3214 := by
  sorry

end jacob_additional_money_needed_l202_202852


namespace exists_two_distinct_points_difference_is_integer_l202_202220

open Set
noncomputable theory

def M (n : ℕ) (A : Fin n → Set ℝ) : Set ℝ := ⋃ i, A i

def total_length_gt_one (n : ℕ) (A : Fin n → Set ℝ) (length : (Set ℝ) → ℝ) : Prop :=
  ∑ i : Fin n, length (A i) > 1

theorem exists_two_distinct_points_difference_is_integer {n : ℕ} (A : Fin n → Set ℝ) (length : (Set ℝ) → ℝ)
  (h_disjoint : ∀ i j, i ≠ j → Disjoint (A i) (A j))
  (h_total_length : total_length_gt_one n A length) :
  ∃ x y ∈ (M n A), x ≠ y ∧ (x - y).is_int := 
sorry

end exists_two_distinct_points_difference_is_integer_l202_202220


namespace diagonals_in_nonagon_l202_202718

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202718


namespace percentage_of_employees_speak_french_l202_202154

-- Definitions based on the conditions
def total_employees : ℕ := 100
def men_percentage : ℝ := 0.35
def men_speak_french_percentage : ℝ := 0.60
def women_do_not_speak_french_percentage : ℝ := 0.7077

-- Derived definitions
def men_count : ℕ := (men_percentage * total_employees).to_nat
def women_count : ℕ := (total_employees - men_count)
def men_speak_french_count : ℕ := (men_speak_french_percentage * men_count).to_nat
def women_speak_french_percentage : ℝ := 1.0 - women_do_not_speak_french_percentage
def women_speak_french_count : ℕ := (women_speak_french_percentage * women_count).to_nat
def total_speak_french_count : ℕ := men_speak_french_count + women_speak_french_count

-- The theorem to prove
theorem percentage_of_employees_speak_french : 
  (total_speak_french_count.to_real / total_employees.to_real * 100) = 40 := by
  sorry

end percentage_of_employees_speak_french_l202_202154


namespace sqrt_simplification_l202_202933

theorem sqrt_simplification :
  (real.sqrt 16 = 4) ∧ 
  (real.sqrt ((-5 : ℝ)^2) = 5) ∧ 
  (real.sqrt 5 * real.sqrt 10 = 5 * real.sqrt 2) :=
by
  sorry

end sqrt_simplification_l202_202933


namespace third_generation_tail_length_is_25_l202_202186

def first_generation_tail_length : ℝ := 16
def growth_rate : ℝ := 0.25

def second_generation_tail_length : ℝ := first_generation_tail_length * (1 + growth_rate)
def third_generation_tail_length : ℝ := second_generation_tail_length * (1 + growth_rate)

theorem third_generation_tail_length_is_25 :
  third_generation_tail_length = 25 := by
  sorry

end third_generation_tail_length_is_25_l202_202186


namespace nine_sided_polygon_diagonals_l202_202664

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202664


namespace slope_of_l2_l202_202217

-- Definitions based on conditions
def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1
def inclination_angle_to_slope (θ : ℝ) : ℝ := Real.tan θ

-- Given conditions
variable (l1_ang : ℝ) (l1_slope : ℝ) (l2_slope : ℝ)
hypothesis (h1 : l1_ang = Real.pi / 6)  -- 30 degrees in radians
hypothesis (h2 : l1_slope = inclination_angle_to_slope l1_ang)
hypothesis (h3 : is_perpendicular l1_slope l2_slope)

-- Prove the slope of l2 is -√3
theorem slope_of_l2 :
  l2_slope = -Real.sqrt 3 :=
sorry

end slope_of_l2_l202_202217


namespace sin_alpha_2beta_over_sin_alpha_eq_three_l202_202084

theorem sin_alpha_2beta_over_sin_alpha_eq_three
  (α β : ℝ)
  (h1 : ∀ (k : ℤ), α ≠ k * (π / 2) ∧ β ≠ k * (π / 2))
  (h2 : tan (α + β) = 2 * tan β) :
  (sin (α + 2 * β) / sin α) = 3 := 
sorry

end sin_alpha_2beta_over_sin_alpha_eq_three_l202_202084


namespace find_r_floor_r_add_r_eq_18point2_l202_202029

theorem find_r_floor_r_add_r_eq_18point2 (r : ℝ) (h : ⌊r⌋ + r = 18.2) : r = 9.2 := 
sorry

end find_r_floor_r_add_r_eq_18point2_l202_202029


namespace parabola_distance_focus_directrix_l202_202168

theorem parabola_distance_focus_directrix (p : ℝ) (P : ℝ × ℝ) (hP : P = (1, 2)) (h_vertex : (0, 0)) (h_focus : ∃ fx : ℝ, fx > 0 ∧ focus = (fx, 0)) :
  let C := { point : ℝ × ℝ | point.snd^2 = 2 * p * point.fst } in
  P ∈ C → distance (p / 2, 0) (focus_directrix : ℝ) = |p| :=
by
  sorry

end parabola_distance_focus_directrix_l202_202168


namespace nine_sided_polygon_diagonals_count_l202_202698

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202698


namespace min_sum_prod_l202_202210

theorem min_sum_prod (a : Fin 10 → ℕ) (distinct : ∀ i j, i ≠ j → a i ≠ a j) (sum_1995 : ∑ i, a i = 1995) :
  (∑ i : Fin 10, a i * a (i + 1)) = 5820 :=
by
  sorry

end min_sum_prod_l202_202210


namespace diagonals_in_nine_sided_polygon_l202_202798

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202798


namespace eraser_cost_l202_202311

theorem eraser_cost (initial_money : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) (erasers_count : ℕ) (remaining_money : ℕ) :
    initial_money = 100 →
    scissors_count = 8 →
    scissors_price = 5 →
    erasers_count = 10 →
    remaining_money = 20 →
    (initial_money - scissors_count * scissors_price - remaining_money) / erasers_count = 4 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end eraser_cost_l202_202311


namespace hotel_guest_count_l202_202306

theorem hotel_guest_count
  (O : ℕ) (H : ℕ) (B : ℕ)
  (hO : O = 70)
  (hH : H = 52)
  (hB : B = 28) :
  O + H - B = 94 :=
by
  rw [hO, hH, hB]
  exact rfl

end hotel_guest_count_l202_202306


namespace tail_length_third_generation_l202_202182

theorem tail_length_third_generation (initial_length : ℕ) (growth_rate : ℕ) :
  initial_length = 16 ∧ growth_rate = 25 → 
  let sec_len := initial_length * (100 + growth_rate) / 100 in
  let third_len := sec_len * (100 + growth_rate) / 100 in
  third_len = 25 := by
  intros h
  sorry

end tail_length_third_generation_l202_202182


namespace sum_of_first_10_common_elements_is_correct_l202_202456

-- Define arithmetic progression
def a (n : ℕ) : ℕ := 4 + 3 * n

-- Define geometric progression
def b (k : ℕ) : ℕ := 20 * (2 ^ k)

-- Define the sum of the first 10 common elements in both sequences
def sum_first_10_common_elements : ℕ := 13981000

-- Statement of the proof problem in Lean 4
theorem sum_of_first_10_common_elements_is_correct :
  ∑ i in (finset.range 10).image (λ k, b(2*k + 1)), id = sum_first_10_common_elements :=
by
  -- Proof omitted
  sorry

end sum_of_first_10_common_elements_is_correct_l202_202456


namespace decreasing_function_proof_l202_202956

variables 
  {f : ℝ → ℝ}
  {α β : ℝ}
  (h_f : ∀ ⦃x y⦄, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 → x ≤ y → f y ≤ f x)
  (h_α_acute : 0 < α ∧ α < π / 2)
  (h_β_acute : 0 < β ∧ β < π / 2)
  (h_α_neq_β : α ≠ β)
  (h_cosα_le_sinβ : cos α < sin β)

theorem decreasing_function_proof : f (cos α) > f (sin β) :=
by
  sorry

end decreasing_function_proof_l202_202956


namespace jacob_additional_money_needed_l202_202853

/-- Jacob's total trip cost -/
def trip_cost : ℕ := 5000

/-- Jacob's hourly wage -/
def hourly_wage : ℕ := 20

/-- Jacob's working hours -/
def working_hours : ℕ := 10

/-- Income from job -/
def job_income : ℕ := hourly_wage * working_hours

/-- Price per cookie -/
def cookie_price : ℕ := 4

/-- Number of cookies sold -/
def cookies_sold : ℕ := 24

/-- Income from cookies -/
def cookie_income : ℕ := cookie_price * cookies_sold

/-- Lottery ticket cost -/
def lottery_ticket_cost : ℕ := 10

/-- Lottery win amount -/
def lottery_win : ℕ := 500

/-- Money received from each sister -/
def sister_gift : ℕ := 500

/-- Total income from job and cookies -/
def income_without_expenses : ℕ := job_income + cookie_income

/-- Income after lottery ticket purchase -/
def income_after_ticket : ℕ := income_without_expenses - lottery_ticket_cost

/-- Total income after lottery win -/
def income_with_lottery : ℕ := income_after_ticket + lottery_win

/-- Total gift from sisters -/
def total_sisters_gift : ℕ := 2 * sister_gift

/-- Total money Jacob has -/
def total_money : ℕ := income_with_lottery + total_sisters_gift

/-- Additional amount needed by Jacob -/
def additional_needed : ℕ := trip_cost - total_money

theorem jacob_additional_money_needed : additional_needed = 3214 := by
  sorry

end jacob_additional_money_needed_l202_202853


namespace sum_a_i_eq_l202_202876

variable {n : ℕ}
variable (hn : n ≥ 2)

def min_a_i (m : ℕ) : ℕ := 
  let imin := (sqrt m).toNat in
  let a := (imin + m / imin : ℝ).toNNReal
  let b := (imin + 1 + m / (imin + 1) : ℝ).toNNReal
  min a b

def S_n_squared (n : ℕ) : ℕ :=
  ∑ i in finset.range (n^2), floor (min_a_i i).val

theorem sum_a_i_eq : S_n_squared n = (8 * n^3 - 3 * n^2 + 13 * n - 6) / 6 := 
  by
  sorry

end sum_a_i_eq_l202_202876


namespace nine_sided_polygon_diagonals_count_l202_202685

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202685


namespace gcd_lcm_product_correct_l202_202051

noncomputable def gcd_lcm_product : ℕ :=
  let a := 90
  let b := 135
  gcd a b * lcm a b

theorem gcd_lcm_product_correct : gcd_lcm_product = 12150 :=
  by
  sorry

end gcd_lcm_product_correct_l202_202051


namespace program_calculations_difficulty_l202_202970

-- Define the options
inductive StatementOption
| input_output
| assignment
| conditional
| loop

open StatementOption

-- Define the question and conditions as hypotheses
def question: StatementOption → Prop := 
λ s, s = loop

-- The mathematical proof problem statement in Lean 4
theorem program_calculations_difficulty : 
  question input_output = False ∧ 
  question assignment = False ∧ 
  question conditional = False ∧ 
  question loop = True := 
sorry

end program_calculations_difficulty_l202_202970


namespace solve_inequality_l202_202250

theorem solve_inequality (x : ℝ) : 
  (frac {2*x - 1} {x + 1} < 0) ↔ (-1 < x ∧ x < 1/2) := by
  sorry

end solve_inequality_l202_202250


namespace find_y_l202_202086

variable (a b c x : ℝ) (p q r y : ℝ)

-- Define the conditions
def condition1 := (log a / (2 * p) = log b / (3 * q)) ∧ (log b / (3 * q) = log c / (4 * r)) ∧ (log c / (4 * r) = log x)
def condition2 := x ≠ 1
def condition3 := b^3 = a^2 * c * x^y

theorem find_y (h1 : condition1 a b c x p q r) (h2 : condition2 x) (h3 : condition3 a b c x y) : y = 9 * q - 4 * p - 4 * r :=
by
  sorry

end find_y_l202_202086


namespace third_generation_tail_length_l202_202179

theorem third_generation_tail_length (tail_length : ℕ → ℕ) (h0 : tail_length 0 = 16)
    (h_next : ∀ n, tail_length (n + 1) = tail_length n + (25 * tail_length n) / 100) :
    tail_length 2 = 25 :=
by
  sorry

end third_generation_tail_length_l202_202179


namespace diagonals_in_nine_sided_polygon_l202_202791

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202791


namespace maximum_speed_l202_202952

variable (A S ρ v0 v : ℝ)

def force (A S ρ v0 v : ℝ) : ℝ := (A * S * ρ * (v0 - v) ^ 2) / 2

def power (A S ρ v0 v : ℝ) : ℝ := force A S ρ v0 v * v

theorem maximum_speed : 
  S = 4 ∧ v0 = 4.8 ∧ 
  ∃ A ρ v,
    (∀ v, power A S ρ v0 v ≤ power A S ρ v0 1.6) → 
    v == 1.6 :=
by
  sorry

end maximum_speed_l202_202952


namespace regular_nonagon_diagonals_correct_l202_202712

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202712


namespace cricket_team_members_l202_202977

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℚ) (wk_keeper_age : ℚ) 
  (avg_whole_team : ℚ) (avg_remaining_players : ℚ)
  (h1 : captain_age = 25)
  (h2 : wk_keeper_age = 28)
  (h3 : avg_whole_team = 22)
  (h4 : avg_remaining_players = 21)
  (h5 : 22 * n = 25 + 28 + 21 * (n - 2)) :
  n = 11 :=
by sorry

end cricket_team_members_l202_202977


namespace diagonals_in_nine_sided_polygon_l202_202782

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202782


namespace area_of_triangle_l202_202479

noncomputable def side_a := (1 : ℝ)
noncomputable def func_f (x B : ℝ) : ℝ := Real.sin (2 * x + B) + Real.sqrt 3 * Real.cos (2 * x + B)
noncomputable def odd_function_y (x B : ℝ) : ℝ := func_f (x - Real.pi / 3) B

theorem area_of_triangle
  (B : ℝ)
  (h₁ : ∀ x, odd_function_y x B = -odd_function_y (-x) B)
  (h₂ : B = Real.pi / 3) :
  let f_0 := func_f 0 B
  let b := f_0
  let C := Real.pi / 2
  let S := 1 / 2 * side_a * b * Real.sin C
  S = Real.sqrt 3 / 4 :=
by
  have H : Real.sin (2 * 0 + B) + Real.sqrt 3 * Real.cos (2 * 0 + B) = f_0
  sorry

end area_of_triangle_l202_202479


namespace frank_fraction_spent_l202_202059

noncomputable def fractionSpentOnMagazine (initial_money total_spent remaining_money : ℕ) : ℚ :=
(total_spent : ℚ) / (initial_money - remaining_money)

theorem frank_fraction_spent
  (initial_money : ℕ)
  (fraction_spent_on_groceries : ℚ)
  (money_left_after_magazine : ℕ)
  (h_initial : initial_money = 600)
  (h_fraction_groceries : fraction_spent_on_groceries = 1 / 5)
  (h_left_after_magazine : money_left_after_magazine = 360)
  (h_total_spent_groceries : ∃ (spent_on_groceries : ℕ), spent_on_groceries = (fraction_spent_on_groceries * initial_money).toNat) :
  fractionSpentOnMagazine initial_money (initial_money - money_left_after_magazine) money_left_after_magazine = 1 / 4 :=
sorry

end frank_fraction_spent_l202_202059


namespace true_propositions_l202_202880

variables {a b c : Line} {γ : Plane}

def proposition1 (a b c : Line) : Prop := (a ∥ b) ∧ (b ∥ c) → (a ∥ c)
def proposition2 (a b c : Line) : Prop := (a ⊥ b) ∧ (b ⊥ c) → (a ⊥ c)
def proposition3 (a b : Line) (γ : Plane) : Prop := (a ∥ γ) ∧ (b ∥ γ) → (a ∥ b)
def proposition4 (a b : Line) (γ : Plane) : Prop := (a ⊥ γ) ∧ (b ⊥ γ) → (a ∥ b)

theorem true_propositions (a b c : Line) (γ : Plane) : proposition1 a b c ∧ ¬ (proposition2 a b c) ∧ ¬ (proposition3 a b γ) ∧ (proposition4 a b γ) :=
by sorry

end true_propositions_l202_202880


namespace limit_value_l202_202405

noncomputable def limit_expression (x : ℝ) : ℝ :=
  (sin (3 * real.pi * x) / sin (real.pi * x)) ^ (sin (x - 2) ^ 2)

theorem limit_value :
  filter.tendsto limit_expression (nhds 2) (nhds 1) :=
begin
  sorry,
end

end limit_value_l202_202405


namespace swap_sum_2007_l202_202314

theorem swap_sum_2007 (n : ℕ) (hn : n = 2006) :
  (∃ (A : Fin n → Fin n) (B : (Fin n) × (Fin n) → ℕ), ∀ i : Fin n,
    B ((mk (i.val) i.is_lt), (mk ((i.val + 1) % n) (by {apply Nat.mod_lt, apply Nat.add_pos_left, exact i.is_lt }))) =
      swap_moves * (A (mk (i.val) i.is_lt)))
  → (∃ (i j : Fin n), i.val + j.val = 2006 ∧ swapped B i j) := sorry

end swap_sum_2007_l202_202314


namespace collinear_points_trajectory_of_Q_l202_202493

-- Assumptions
variable (A B C : Point) (O : Point)
variable (l m : ℝ)
variable (Q : Point)

-- Assumed conditions
def parabola (P : Point) : Prop := P.y^2 = 4 * P.x
def coordinates_C : C = Point.mk 4 0 := rfl
def distinct_points_on_parabola (hA : parabola A) (hB : parabola B) (hC : parabola C) : Prop :=
  A ≠ O ∧ B ≠ O ∧ A ≠ B
def perpendicular_vectors (OA OB : ℝ) : Prop :=
  OA / Math.sqrt (A.x^2 + A.y^2) = 1 ∧ OB / Math.sqrt (B.x^2 + B.y^2) = 1 ∧ (OA * OB) = 0

-- Proof goals
theorem collinear_points
  (hA : parabola A)
  (hB : parabola B)
  (hO : distinct_points_on_parabola)
  (hC : coordinates_C)
  (h_perp : perpendicular_vectors O A B) :
  collinear A B C := sorry

theorem trajectory_of_Q
  (hA : parabola A)
  (hB : parabola B)
  (hO : distinct_points_on_parabola)
  (h_perp : perpendicular_vectors O A B)
  (h_Q : AQ = l * QB ∧ inner_product O Q = 0) :
  equation_of_trajectory Q := sorry

end collinear_points_trajectory_of_Q_l202_202493


namespace expand_polynomial_l202_202025

theorem expand_polynomial : 
  (∀ (x : ℝ), (5 * x^3 + 7) * (3 * x + 4) = 15 * x^4 + 20 * x^3 + 21 * x + 28) :=
by
  intro x
  sorry

end expand_polynomial_l202_202025


namespace nine_sided_polygon_diagonals_count_l202_202683

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202683


namespace find_a_for_real_roots_l202_202035

theorem find_a_for_real_roots :
  {a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
    (sqrt (6 * x1 - x1^2 - 4) + a - 2) * (a - 2) * x1 - 3 * a + 4 = 0 ∧
    (sqrt (6 * x2 - x2^2 - 4) + a - 2) * (a - 2) * x2 - 3 * a + 4 = 0} =
  {a | a = 2 - sqrt 5 ∨ a = 0 ∨ a = 1 ∨ (2 - 2 / sqrt 5 < a ∧ a ≤ 2)} :=
sorry

end find_a_for_real_roots_l202_202035


namespace difference_in_cents_l202_202849

def coins_problem (p d : ℕ) : Prop :=
  p + d = 2020 ∧ p ≥ 1 ∧ d ≥ 1

theorem difference_in_cents :
  ∃ (p d : ℕ), coins_problem p d ∧ (10 * 2019 + 1 + 5050) - (10 * 1 + 2019 + 5050) = 13220 :=
by { existsi 1, existsi 2019, simp [coins_problem], norm_num, sorry }

end difference_in_cents_l202_202849


namespace nine_sided_polygon_diagonals_l202_202657

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202657


namespace counting_number_leaves_remainder_of_6_l202_202125

theorem counting_number_leaves_remainder_of_6:
  ∃! d : ℕ, d > 6 ∧ d ∣ (53 - 6) ∧ 53 % d = 6 :=
begin
  sorry
end

end counting_number_leaves_remainder_of_6_l202_202125


namespace nine_sided_polygon_diagonals_l202_202615

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202615


namespace total_students_l202_202295

theorem total_students (orchestra band choir_boys choir_girls : ℕ)
  (h_orchestra : orchestra = 20)
  (h_band : band = 2 * orchestra)
  (h_choir_boys : choir_boys = 12)
  (h_choir_girls : choir_girls = 16)
  (h_disjoint : ∀ x, x ∈ orchestra ∨ x ∈ band ∨ x ∈ (choir_boys + choir_girls) → 
                    (x ∈ orchestra → x ∉ band ∧ x ∉ (choir_boys + choir_girls)) ∧
                    (x ∈ band → x ∉ orchestra ∧ x ∉ (choir_boys + choir_girls)) ∧
                    (x ∈ (choir_boys + choir_girls) → x ∉ orchestra ∧ x ∉ band)) :
  orchestra + band + (choir_boys + choir_girls) = 88 := 
by
  rw [h_orchestra, h_band, h_choir_boys, h_choir_girls]
  show 20 + 2 * 20 + (12 + 16) = 88 from
  calc
    20 + 2 * 20 + (12 + 16) = 20 + 40 + 28 : by rfl
    ...                      = 88            : by rfl

end total_students_l202_202295


namespace shift_sin_graph_l202_202980

theorem shift_sin_graph :
  ∀ x : ℝ, sin (2 * (x + π / 12) - π / 6) = sin (2 * x) :=
by sorry

end shift_sin_graph_l202_202980


namespace residuals_correct_l202_202338

theorem residuals_correct (H1 : ¬ (Residuals = RandomErrors)) 
                          (H2 : ¬ (Residuals = Variance)) 
                          (H3 : ¬ ∀ r, r ∈ Residuals → r > 0) :
                          (Residuals can_be_used_to_assess_fit_of_model) := sorry

end residuals_correct_l202_202338


namespace value_of_a_minus_b_l202_202810

theorem value_of_a_minus_b (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 :=
by
  sorry

end value_of_a_minus_b_l202_202810


namespace range_of_f_sum_of_xn_l202_202509

noncomputable def f (x : ℝ) : ℝ :=
  (sin (x / 2 + π / 12))^2 + sqrt 3 * sin (x / 2 + π / 12) * cos (x / 2 + π / 12) - 1 / 2

theorem range_of_f :
  (set.range f) = set.Icc (-1) 1 := sorry

theorem sum_of_xn (n : ℕ) (x : ℕ → ℝ) (h : ∀ k, 1 ≤ k → x (2 * k - 1) + x (2 * k) = 2 * (k - 1) * π + π / 2) :
  (∀ k : ℕ, 1 ≤ k → x k ∈ set.range f) →
  (finset.range (2 * n)).sum (λ i, x (i + 1)) = (2 * n^2 - n) * π := sorry

end range_of_f_sum_of_xn_l202_202509


namespace hyperbola_equation_l202_202039

theorem hyperbola_equation (x y : ℝ) (h_asymptotes : ∀ x y, x^2 - 4 * y^2 = λ) (h_point : 2^2 - 4 * (real.sqrt 5)^2 = -16) :
  y^2 / 4 - x^2 / 16 = 1 :=
sorry

end hyperbola_equation_l202_202039


namespace Harper_spends_60_dollars_l202_202528

-- Define the conditions
variable (dailyConsumption : ℚ := 1/2)                -- Harper drinks 1/2 bottle of water per day.
variable (bottlesPerCase : ℤ := 24)                   -- Each case contains 24 bottles.
variable (caseCost : ℚ := 12)                        -- A case costs $12.00.
variable (totalDays : ℤ := 240)                      -- Harper wants to buy enough cases for 240 days.

-- Theorem statement
theorem Harper_spends_60_dollars :
  let durationCase : ℤ := bottlesPerCase / (dailyConsumption.natAbs : ℤ)
  let casesNeeded : ℤ := totalDays / durationCase
  let totalCost : ℚ := casesNeeded * caseCost
  totalCost = 60 := by
  sorry

end Harper_spends_60_dollars_l202_202528


namespace sequence_2002_eq_4008003_l202_202909

-- Define the sequence function
def sequence (n : ℕ) : ℕ := n^2 - 1

theorem sequence_2002_eq_4008003 : sequence 2002 = 4008003 :=
by
  sorry

end sequence_2002_eq_4008003_l202_202909


namespace algebraic_expression_value_l202_202138

theorem algebraic_expression_value (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : m ^ 2 = 25) :
  m^2 - 100*a - 99*b - b*c*d + |c*d - 2| = -74 :=
by
  sorry

end algebraic_expression_value_l202_202138


namespace diagonals_in_regular_nine_sided_polygon_l202_202576

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202576


namespace ab_cd_eq_one_l202_202144

theorem ab_cd_eq_one (a b c d : ℤ) (w x y z : ℕ) (hw : Nat.Prime w) (hx : Nat.Prime x) (hy : Nat.Prime y) (hz : Nat.Prime z)
  (h_ineq : w < x ∧ x < y ∧ y < z)
  (h_eq : w^a * x^b * y^c * z^d = 660) : (a + b) - (c + d) = 1 := 
  sorry

end ab_cd_eq_one_l202_202144


namespace karen_wrong_questions_l202_202870

theorem karen_wrong_questions (k l n : ℕ) (h1 : k + l = 6 + n) (h2 : k + n = l + 9) : k = 6 := 
by
  sorry

end karen_wrong_questions_l202_202870


namespace total_land_l202_202859

variable (land_house : ℕ) (land_expansion : ℕ) (land_cattle : ℕ) (land_crop : ℕ)

theorem total_land (h1 : land_house = 25) 
                   (h2 : land_expansion = 15) 
                   (h3 : land_cattle = 40) 
                   (h4 : land_crop = 70) : 
  land_house + land_expansion + land_cattle + land_crop = 150 := 
by 
  sorry

end total_land_l202_202859


namespace simplified_complex_expr_l202_202246

-- Definitions
def complex_expr : ℂ := 7 * (2 - 3 * complex.i) + 2 * complex.i * (7 - 3 * complex.i)

-- Theorem to prove
theorem simplified_complex_expr : complex_expr = 20 - 7 * complex.i :=
  by sorry

end simplified_complex_expr_l202_202246


namespace geometric_sequence_general_formula_sum_of_sequence_l202_202958

noncomputable def a (n : ℕ) : ℝ := (1 / 2) ^ n

def b (n : ℕ) : ℝ := 3 * n - 2

def c (n : ℕ) : ℝ := (3 * n - 2) * (1 / 2) ^ n

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, (c i)

theorem geometric_sequence_general_formula (a_1 a_2 a_3 a_6 : ℝ) (h1 : a_1 + 2 * a_2 = 1) 
  (h2 : a_3 * a_3 = 4 * a_2 * a_6) :
  (∀ n, a n = (1 / 2) ^ n) :=
sorry

theorem sum_of_sequence (n : ℕ) :
  S(n) = 4 - (3 * n + 4) / 2^n :=
sorry

end geometric_sequence_general_formula_sum_of_sequence_l202_202958


namespace work_days_x_l202_202345

-- Define the conditions
def work_rate_x (W : ℝ) : ℝ := W / 40
def work_rate_y (W : ℝ) : ℝ := W / 45
def total_work_done (W d : ℝ) : ℝ := d * work_rate_x W + 36 * work_rate_y W

-- Define the problem as a theorem
theorem work_days_x (W : ℝ) (d : ℝ) (h : total_work_done W d = W) : d = 8 :=
by
  -- The proof is omitted
  sorry

end work_days_x_l202_202345


namespace inscribed_circle_area_correct_l202_202416

noncomputable def inscribed_circle_area (R a : ℝ) : ℝ :=
  π * ((R * a) / (R + a))^2

theorem inscribed_circle_area_correct (R a : ℝ) :
  ∀ (R a : ℝ), R > 0 → a > 0 → inscribed_circle_area R a = π * ((R * a) / (R + a))^2 :=
by
  intros R a hR ha
  unfold inscribed_circle_area
  sorry

end inscribed_circle_area_correct_l202_202416


namespace f_even_f_increasing_on_solve_inequality_l202_202511

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * abs x

theorem f_even (x : ℝ) : f (-x) = f x :=
by
  unfold f
  sorry

theorem f_increasing_on (a x : ℝ) (h : 1 < x) : f (x) ≤ f (|a| + 3/2) :=
by
  unfold f
  sorry

theorem solve_inequality (a : ℝ) : a > 1/2 ∨ a < -1/2 :=
by
  -- Assume |a| + 3/2 > 1
  have h : |a| > 1/2 := sorry
  exact Or.inl (abs_pos_iff.mp h)
  sorry

end f_even_f_increasing_on_solve_inequality_l202_202511


namespace nine_sided_polygon_diagonals_l202_202634

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202634


namespace diagonals_in_nine_sided_polygon_l202_202800

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202800


namespace min_positive_period_f_symmetry_center_f_max_value_g_min_value_g_l202_202506

variable (x : ℝ)

def f (x : ℝ) : ℝ := 
  cos x * sin (x + (π / 3)) - sqrt 3 * (cos x)^2 + (sqrt 3 / 4)

def g (x : ℝ) : ℝ :=
  f (x + (π / 4))

theorem min_positive_period_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
sorry

theorem symmetry_center_f : ∃ k : ℤ, (∀ x, f x = 0 → x = (π / 6 + (k * π / 2))) :=
sorry

theorem max_value_g : ∀ x, x ∈ Icc (-(π / 6)) (π / 3) → g x ≤ (1 / 2) :=
sorry

theorem min_value_g : ∀ x, x ∈ Icc (-(π / 6)) (π / 3) → g x ≥ - (1 / 4) :=
sorry

end min_positive_period_f_symmetry_center_f_max_value_g_min_value_g_l202_202506


namespace nine_sided_polygon_diagonals_l202_202653

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202653


namespace diagonals_in_regular_nine_sided_polygon_l202_202541

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202541


namespace combinations_10_choose_3_l202_202061

theorem combinations_10_choose_3 : fintype.card ({s : finset ℕ // s.card = 3} : set (finset ℕ)) = 120 :=
by sorry

end combinations_10_choose_3_l202_202061


namespace incorrect_reasoning_C_l202_202395

-- Define base entities
variable (A B C : Point)
variable (a l : Line)
variable (α β : Plane)

-- Define the conditions given in the problem
axiom axiom1 : ∀ {A B : Point} {l : Line} {α : Plane}, (A ∈ l) → (A ∈ α) → (B ∈ l) → (B ∈ α) → l ⊆ α
axiom axiom2 : ∀ {A B : Point} {α β : Plane}, (A ∈ α) → (A ∈ β) → (B ∈ α) → (B ∈ β) → α ∩ β = line_through A B
axiom axiom3 : ∀ {A : Point} {l : Line} {α : Plane}, (l ⊈ α) → (A ∈ l) → (A ∉ α)
axiom axiom4 : ∀ {A B C : Point} {α β : Plane}, (C ∈ α) → (B ∈ β) → (C ∈ β) → ¬are_collinear A B C → α = β

-- The main theorem to prove option C is incorrect
theorem incorrect_reasoning_C :
  ∀ {A B C : Point} {a l : Line} {α β : Plane}, 
  (l ⊆ α → A ∉ α) cannot always be true := sorry

end incorrect_reasoning_C_l202_202395


namespace diagonals_in_regular_nine_sided_polygon_l202_202571

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202571


namespace verify_propositions_correct_l202_202010

open Real

noncomputable def verify_propositions : Prop :=
  let prop1 := (∀ x : ℝ, log x = 0 → x = 1) → (∀ x : ℝ, log x ≠ 0 → x ≠ 1)
  let prop2 := (∀ p q : Prop, ¬(p ∧ q) → ¬p ∧ ¬q)
  let prop3 := (∃ x : ℝ, sin x > 1) = false
  let prop4 := (∀ x : ℝ, (x > 2 → 1 / x < 1 / 2) ∧ (¬(x > 2) → ∃ y : ℝ, y ≠ 0 ∧ abs y = abs (1 / x)))
  let true_props := [prop1, prop3, prop4].count (λ p, p = true)
  true_props = 3

theorem verify_propositions_correct : verify_propositions := by
  sorry

end verify_propositions_correct_l202_202010


namespace sum_diff_p_convergent_l202_202135

open Real

noncomputable def is_convergent (s : ℕ → ℝ) : Prop := Summable s

theorem sum_diff_p_convergent 
  (u v : ℕ → ℝ)
  (hu : is_convergent (λ i, u i ^ 2))
  (hv : is_convergent (λ i, v i ^ 2))
  (p : ℕ) 
  (hp : p ≥ 2) 
  : is_convergent (λ i, (u i - v i) ^ p) :=
sorry

end sum_diff_p_convergent_l202_202135


namespace arithmetic_sequence_a12_l202_202170

theorem arithmetic_sequence_a12 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 7 + a 9 = 16) (h2 : a 4 = 1) 
  (h3 : ∀ n, a (n + 1) = a n + d) : a 12 = 15 := 
by {
  -- Proof steps would go here
  sorry
}

end arithmetic_sequence_a12_l202_202170


namespace twelve_pretty_sum_div_12_l202_202400

def is_12_pretty (n : ℕ) : Prop :=
  n > 0 ∧ n % 12 = 0 ∧ Nat.totient ((d : ℕ) → (d ∣ n)) = 12

noncomputable def T : ℕ :=
  ∑ n in (Finset.range 1000).filter is_12_pretty, n

theorem twelve_pretty_sum_div_12 : T / 12 = 15 :=
by
  sorry

end twelve_pretty_sum_div_12_l202_202400


namespace regular_nine_sided_polygon_diagonals_l202_202677

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202677


namespace height_increase_percentage_l202_202274

noncomputable def volume_cone (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h

theorem height_increase_percentage (r h x : ℝ) (h_pos : 0 < h)
  (V : ℝ := volume_cone r h)
  (V' : ℝ := 2.30 * V)
  (h' : ℝ := h * (1 + x / 100))
  (V'_eq : volume_cone r h' = V') :
  x = 130 :=
by
  have : (1 + x / 100) * V = 2.30 * V,
  calc (1 + x / 100) * V = V' : by rw [volume_cone, V'_eq]
                      ... = 2.30 * V : by rw [V]
  sorry

end height_increase_percentage_l202_202274


namespace fifth_smallest_odd_with_four_prime_factors_l202_202325

noncomputable def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_at_least_four_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d (ha : prime a) (hb : prime b) (hc : prime c) (hd : prime d),
    n = a * b * c * d * p

theorem fifth_smallest_odd_with_four_prime_factors :
  ∃ n : ℕ, is_odd n ∧ has_at_least_four_prime_factors n ∧
  (∀ m : ℕ, (m < n → is_odd m ∧ has_at_least_four_prime_factors m) → m = 135 ∨ m = 225 ∨ m = 315 ∨ m = 1155) ∧
  n = 1925 :=
sorry

end fifth_smallest_odd_with_four_prime_factors_l202_202325


namespace diagonals_in_nine_sided_polygon_l202_202776

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202776


namespace find_m_l202_202504

theorem find_m (m : ℝ) : 
  (m^2 - 1 = 0) ∧ (m^2 - 7m + 6 = 0) ∧ (m - 2 ≠ 0) → m = 1 :=
by 
  intro h,
  cases h with h1 h234,
  cases h234 with h2 h3,
  -- continue proof here
  sorry

end find_m_l202_202504


namespace tricycles_count_l202_202398

-- Define the conditions
variable (b t s : ℕ)

def total_children := b + t + s = 10
def total_wheels := 2 * b + 3 * t + 2 * s = 29

-- Provide the theorem to prove
theorem tricycles_count (h1 : total_children b t s) (h2 : total_wheels b t s) : t = 9 := 
by
  sorry

end tricycles_count_l202_202398


namespace solve_problem_l202_202174

-- Declare the variables n and m
variables (n m : ℤ)

-- State the theorem with given conditions and prove that 2n + m = 36
theorem solve_problem
  (h1 : 3 * n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3 * m - 2 * n < 46) :
  2 * n + m = 36 :=
sorry

end solve_problem_l202_202174


namespace limit_fraction_l202_202235

theorem limit_fraction :
  ∀ ε > 0, ∃ (N : ℕ), ∀ n ≥ N, |((4 * n - 1) / (2 * n + 1) : ℚ) - 2| < ε := 
  by sorry

end limit_fraction_l202_202235


namespace log_eq_sum_of_logs_iff_l202_202034

noncomputable def log_eq_sum_of_logs (x : ℝ) : Prop :=
  (log (x^2 - 10 * x + 21) / log (x^2) = log (x^2 / (x - 7)) / log (x^2) + log (x^2 / (x - 3)) / log (x^2)) ∨
  (log (x^2 / (x - 7)) / log (x^2) = log (x^2 - 10 * x + 21) / log (x^2) + log (x^2 / (x - 3)) / log (x^2)) ∨
  (log (x^2 / (x - 3)) / log (x^2) = log (x^2 - 10 * x + 21) / log (x^2) + log (x^2 / (x - 7)) / log (x^2))

theorem log_eq_sum_of_logs_iff (x : ℝ) : log_eq_sum_of_logs x ↔ x = 8 :=
by
  sorry

end log_eq_sum_of_logs_iff_l202_202034


namespace find_k_l202_202050

variable (x y z k : ℝ)

def fractions_are_equal : Prop := (9 / (x + y) = k / (x + z) ∧ k / (x + z) = 15 / (z - y))

theorem find_k (h : fractions_are_equal x y z k) : k = 24 := by
  sorry

end find_k_l202_202050


namespace nine_sided_polygon_diagonals_count_l202_202684

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202684


namespace harper_spending_l202_202526

noncomputable def daily_consumption : ℝ := 1 / 2 
noncomputable def total_days : ℕ := 240
noncomputable def bottles_per_case : ℕ := 24
noncomputable def price_per_case : ℝ := 12

theorem harper_spending :
  let total_bottles := total_days * daily_consumption in
  let cases_needed := total_bottles / bottles_per_case in
  let total_cost := cases_needed * price_per_case in
  total_cost = 60 :=
by
  sorry

end harper_spending_l202_202526


namespace y_intercept_of_line_l202_202319

theorem y_intercept_of_line : ∃ y : ℝ, 3 * 0 - 5 * y = 10 ∧ y = -2 :=
by
  use -2
  split
  { norm_num }
  { refl }

end y_intercept_of_line_l202_202319


namespace solution_l202_202438

noncomputable def problem (x : ℝ) : Prop :=
  (Real.sqrt (Real.sqrt (53 - 3 * x)) + Real.sqrt (Real.sqrt (39 + 3 * x))) = 5

theorem solution :
  ∀ x : ℝ, problem x → x = -23 / 3 :=
by
  intro x
  intro h
  sorry

end solution_l202_202438


namespace probability_divisible_by_4_l202_202918

theorem probability_divisible_by_4 :
  let S := finset.range 1006 \ {0} in -- The set {1, 2, ..., 1005}
  let event := {adbd_sum | ∃ (a b d ∈ S), (a^2 * b * d + a * b + d) % 4 = 0} in
  (event.card : ℚ) / (S.card ^ 3 : ℚ) = 7 / 60 := 
sorry

end probability_divisible_by_4_l202_202918


namespace dice_product_divisible_by_8_probability_l202_202987

open ProbabilityTheory

/-- Representation of a single roll of a 6-sided die -/
def roll := Finset.range 1 7

/-- Definition of the event that the product of 8 dice rolls is divisible by 8 -/
def event_product_divisible_by_8 : Event (roll ^ 8) :=
  { ω | 8 ∣ List.prod ω.toList }

/-- Calculate the probability of the above event -/
theorem dice_product_divisible_by_8_probability:
  (event_product_divisible_by_8).prob = 277 / 288 := sorry

end dice_product_divisible_by_8_probability_l202_202987


namespace max_v_l202_202947

/-- conditions --/
def F (A S ρ v₀ v : ℝ) : ℝ := (A * S * ρ * (v₀ - v) ^ 2) / 2

def N (A S ρ v₀ v : ℝ) : ℝ := F A S ρ v₀ v * v

/-- variables and constants --/
variables (A ρ : ℝ)

/-- given values --/
def S : ℝ := 4
def v₀ : ℝ := 4.8

/-- theorem statement --/
theorem max_v : ∃ v, v = 1.6 ∧ ∀ v', N A (S := 4) ρ (v₀ := 4.8) v' ≤ N A S ρ v :=
sorry

end max_v_l202_202947


namespace number_of_diagonals_l202_202553

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202553


namespace minimum_slit_length_l202_202307

theorem minimum_slit_length (circumference : ℝ) (speed_ratio : ℝ) (reliability : ℝ) :
  circumference = 1 → speed_ratio = 2 → (∀ (s : ℝ), (s < 2/3) → (¬ reliable)) → reliability =
    2 / 3 :=
by
  intros hcirc hspeed hrel
  have s := (2 : ℝ) / 3
  sorry

end minimum_slit_length_l202_202307


namespace integral_f_l202_202104

def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x - 1 else Real.cos x

theorem integral_f : ∫ x in -1..π, f x = 1 := by
  sorry

end integral_f_l202_202104


namespace g_10_is_100_l202_202215

noncomputable def g : ℕ → ℝ := sorry

axiom g_conditions :
  (g(1) = 1) ∧
  (∀ m n : ℕ, m ≥ n → g(m + n) + g(m - n) = (g(3 * m) + g(3 * n)) / 3)

theorem g_10_is_100 : g 10 = 100 := by
  sorry

end g_10_is_100_l202_202215


namespace max_value_sincos_sum_l202_202047

theorem max_value_sincos_sum (x y z : ℝ) :
  (∀ x y z, (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5) :=
by sorry

end max_value_sincos_sum_l202_202047


namespace batsman_average_after_17th_inning_l202_202834

-- Definitions and conditions
variable A : ℕ -- Average before 17th inning
variable total_runs_before : ℕ := 16 * A -- Total runs before 17th inning
variable total_runs_after : ℕ := total_runs_before + 56 -- Total runs after 17th inning
variable new_average : ℕ := (total_runs_after / 17) -- New average

axiom cond1 : (total_runs_after / 17) = A + 3
axiom cond2 : 3 <= (56 / 4) -- Minimum boundaries condition, at least 3 boundaries
axiom cond3 : 56 >= 125 * (45 / 100) -- Strike rate condition, at least 125 per 100 balls faced

-- Goal: Prove the batsman's new average is 8 after the 17th inning
theorem batsman_average_after_17th_inning : 8 = new_average := by
  sorry

end batsman_average_after_17th_inning_l202_202834


namespace probability_density_l202_202201
open Complex IsROrC MeasureTheory

variable {Ω : Type} [MeasurableSpace Ω] {ξ : Ω → ℤ} [IsProbabilityMeasure (MeasureTheory.MeasureSpace.volume : MeasureTheory.Measure Ω)]

-- Define the characteristic function of ξ
def char_func (t : ℝ) : ℂ :=  Integral.hasIntegral_ofReal (λ ω, cexp (I * t * (ξ ω))) MeasureTheory.MeasureSpace.volume

theorem probability_density (n : ℤ) :
  MeasureTheory.Probability.measure_set_eq ξ n =
  (1 / (2 * Real.pi) : ℂ) * Integral.hasIntegral (λ t : ℝ, cexp (-I * n * t) * char_func t) (Real.interval_integrable (-Real.pi) Real.pi) :=
by
  sorry

end probability_density_l202_202201


namespace nine_sided_polygon_diagonals_l202_202663

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202663


namespace total_calculators_assembled_l202_202157

theorem total_calculators_assembled (H₁ : ∀ t, calc_by_erika t = 3 * t → calc_by_nick t = 2 * t)
    (H₂ : ∀ t, calc_by_nick t = t → calc_by_sam t = 3 * t) :
    calc_by_erika (9 / 3) + calc_by_nick (9 / 3) + calc_by_sam (6 / 1) = 33 := by
  sorry

end total_calculators_assembled_l202_202157


namespace nine_sided_polygon_diagonals_l202_202597

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202597


namespace eventually_primes_only_l202_202910

theorem eventually_primes_only
    (initial_numbers : Finset ℕ)
    (h_initial : ∀ n ∈ initial_numbers, n > 1) :
    ∃ N, ∀ n ≥ N, is_prime n := 
by
  let M := initial_numbers.max' h_initial -- M is the largest number in the initial set.
  let S := {n : ℕ | ∃ k, n = (smallest_positive_integer_not_divisible_by_any_from S).k} -- Define S as the sequence of numbers written by Nordi.
  sorry

end eventually_primes_only_l202_202910


namespace diagonals_in_regular_nine_sided_polygon_l202_202544

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202544


namespace red_tulips_l202_202297

theorem red_tulips (white_tulips : ℕ) (bouquets : ℕ)
  (hw : white_tulips = 21)
  (hb : bouquets = 7)
  (div_prop : ∀ n, white_tulips % n = 0 ↔ bouquets % n = 0) : 
  ∃ red_tulips : ℕ, red_tulips = 7 :=
by
  sorry

end red_tulips_l202_202297


namespace total_lobster_pounds_l202_202119

theorem total_lobster_pounds
  (combined_other_harbors : ℕ)
  (hooper_bay : ℕ)
  (H1 : combined_other_harbors = 160)
  (H2 : hooper_bay = 2 * combined_other_harbors) :
  combined_other_harbors + hooper_bay = 480 :=
by
  -- proof goes here
  sorry

end total_lobster_pounds_l202_202119


namespace number_of_zeros_f_l202_202278

def f (x : ℝ) : ℝ := x * Real.exp x - x - 2

theorem number_of_zeros_f : ∃ n, (∃ a b : ℝ, f a = 0 ∧ f b = 0 ∧ a ≠ b) ∧ n = 2 := by
  sorry

end number_of_zeros_f_l202_202278


namespace sequence_general_term_l202_202289

theorem sequence_general_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) :
  (∀ n, a_n n > 0) →
  (∀ n, a_n n ^ 2 + 2 * a_n n = 4 * S_n n - 1) →
  (∀ n, b_n n = 1 / (a_n n * a_n (n + 1))) →
  (∀ n, a_n n = 2 * n - 1) ∧ (∀ n, T_n n = n / (2 * n + 1)) := 
by
  intros h1 h2 h3,
  sorry

end sequence_general_term_l202_202289


namespace platform_length_l202_202340

theorem platform_length (train_speed_kmph : ℕ) (train_time_man_seconds : ℕ) (train_time_platform_seconds : ℕ) (train_speed_mps : ℕ) : 
  train_speed_kmph = 54 →
  train_time_man_seconds = 20 →
  train_time_platform_seconds = 30 →
  train_speed_mps = (54 * 1000 / 3600) →
  (54 * 5 / 18) = 15 →
  ∃ (P : ℕ), (train_speed_mps * train_time_platform_seconds) = (train_speed_mps * train_time_man_seconds) + P ∧ P = 150 :=
by
  sorry

end platform_length_l202_202340


namespace new_price_of_computer_l202_202823

theorem new_price_of_computer (d : ℝ) (h : 2 * d = 520) : d * 1.3 = 338 := 
sorry

end new_price_of_computer_l202_202823


namespace total_mistakes_l202_202907

-- Definitions of the conditions
def mistakes_per_40_notes : ℕ := 3
def notes_per_minute : ℕ := 60
def duration_in_minutes : ℕ := 8

-- Claim to prove
theorem total_mistakes (m : ℕ) (n : ℕ) (d : ℕ) (hm : m = mistakes_per_40_notes) (hn : n = notes_per_minute) (hd : d = duration_in_minutes) :
  (d * n / 40) * m = 36 :=
by
  rw [← hm, ← hn, ← hd]
  sorry

end total_mistakes_l202_202907


namespace find_ellipse_eccentricity_l202_202518

open Real

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b) (h_pos : 0 < b) : ℝ :=
  let c := sqrt (a^2 - b^2) in
  c / a

theorem find_ellipse_eccentricity (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (f := sqrt (a^2 - b^2)) (semi_latus_rectum := a^2 / f)
  (circle_radius := f) (h_triangle : equilateral_triangle 0 (semi_latus_rectum, 0) (semi_latus_rectum, 0)) :
  ellipse_eccentricity a b h₁ h₂ = sqrt(6) / 3 :=
begin
  sorry
end

end find_ellipse_eccentricity_l202_202518


namespace solve_problem_l202_202175

-- Declare the variables n and m
variables (n m : ℤ)

-- State the theorem with given conditions and prove that 2n + m = 36
theorem solve_problem
  (h1 : 3 * n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3 * m - 2 * n < 46) :
  2 * n + m = 36 :=
sorry

end solve_problem_l202_202175


namespace y_intercept_of_line_l202_202316

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 5 * y = 10) (hx : x = 0) : y = -2 :=
by {
  rw [hx, zero_mul, sub_zero] at h,
  linarith,
}

end y_intercept_of_line_l202_202316


namespace third_generation_tail_length_l202_202180

theorem third_generation_tail_length (tail_length : ℕ → ℕ) (h0 : tail_length 0 = 16)
    (h_next : ∀ n, tail_length (n + 1) = tail_length n + (25 * tail_length n) / 100) :
    tail_length 2 = 25 :=
by
  sorry

end third_generation_tail_length_l202_202180


namespace banana_pieces_l202_202301

theorem banana_pieces (B G P : ℕ) 
  (h1 : P = 4 * G)
  (h2 : G = B + 5)
  (h3 : P = 192) : B = 43 := 
by
  sorry

end banana_pieces_l202_202301


namespace regular_nonagon_diagonals_correct_l202_202713

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202713


namespace six_digit_even_numbers_count_l202_202277

theorem six_digit_even_numbers_count :
  let digits := {1, 2, 3, 4, 5, 6}
  let is_even (n : ℕ) := n % 2 = 0
  let not_adjacent (x y : ℕ) (list : List ℕ) := 
    ∀ i < list.length - 1, list.get? i = some x → list.get? (i+1) ≠ some y

  (digits.permutations.map (λ l, l.take 6)).count (λ l, 
    is_even l.last ∧ 
    not_adjacent 1 5 l ∧ 
    not_adjacent 3 5 l) = 108 :=
sorry

end six_digit_even_numbers_count_l202_202277


namespace nine_sided_polygon_diagonals_count_l202_202686

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202686


namespace third_generation_tail_length_is_25_l202_202187

def first_generation_tail_length : ℝ := 16
def growth_rate : ℝ := 0.25

def second_generation_tail_length : ℝ := first_generation_tail_length * (1 + growth_rate)
def third_generation_tail_length : ℝ := second_generation_tail_length * (1 + growth_rate)

theorem third_generation_tail_length_is_25 :
  third_generation_tail_length = 25 := by
  sorry

end third_generation_tail_length_is_25_l202_202187


namespace garden_length_l202_202342

-- Define the perimeter and breadth
def perimeter : ℕ := 900
def breadth : ℕ := 190

-- Define a function to calculate the length using given conditions
def length (P : ℕ) (B : ℕ) : ℕ := (P / 2) - B

-- Theorem stating that for the given perimeter and breadth, the length is 260.
theorem garden_length : length perimeter breadth = 260 :=
by
  -- placeholder for proof
  sorry

end garden_length_l202_202342


namespace diagonals_in_regular_nine_sided_polygon_l202_202573

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202573


namespace sum_of_b_is_negative_twelve_l202_202423

-- Conditions: the quadratic equation and its property having exactly one solution
def quadratic_equation (b : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + b * x + 6 * x + 10 = 0

-- Statement to prove: sum of the values of b is -12, 
-- given the condition that the equation has exactly one solution
theorem sum_of_b_is_negative_twelve :
  ∀ b1 b2 : ℝ, (quadratic_equation b1 ∧ quadratic_equation b2) ∧
  (∀ x : ℝ, 3 * x^2 + (b1 + 6) * x + 10 = 0 ∧ 3 * x^2 + (b2 + 6) * x + 10 = 0) ∧
  (∀ b : ℝ, b = b1 ∨ b = b2) →
  b1 + b2 = -12 :=
by
  sorry

end sum_of_b_is_negative_twelve_l202_202423


namespace total_lobster_pounds_l202_202120

variable (lobster_other_harbor1 : ℕ)
variable (lobster_other_harbor2 : ℕ)
variable (lobster_hooper_bay : ℕ)

-- Conditions
axiom h_eq : lobster_hooper_bay = 2 * (lobster_other_harbor1 + lobster_other_harbor2)
axiom other_harbors_eq : lobster_other_harbor1 = 80 ∧ lobster_other_harbor2 = 80

-- Proof statement
theorem total_lobster_pounds : 
  lobster_other_harbor1 + lobster_other_harbor2 + lobster_hooper_bay = 480 :=
by
  sorry

end total_lobster_pounds_l202_202120


namespace part1_part2_l202_202348

-- Part 1
theorem part1 (x y : ℝ) (m n : ℤ) :
  (4 * x^2 * y^(m + 2) + x * y^2 + (n - 2) * x^2 * y^3 + x * y - 4).degree = 7 →
  m = 3 ∧ n = 2 := sorry

-- Part 2
theorem part2 (x y a b : ℝ) :
  (5 * a - 2 = 0) →
  (10 * a + b = 0) →
  5 * a + b = -2 := sorry

end part1_part2_l202_202348


namespace cost_formula_l202_202265

def cost (P : ℕ) : ℕ :=
  if P ≤ 5 then 5 * P + 10 else 5 * P + 5

theorem cost_formula (P : ℕ) : 
  cost P = (if P ≤ 5 then 5 * P + 10 else 5 * P + 5) :=
by 
  sorry

end cost_formula_l202_202265


namespace nine_sided_polygon_diagonals_l202_202642

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202642


namespace calculate_f_at_1_l202_202101

theorem calculate_f_at_1 : ∃ m : ℝ, (∀ x : ℝ, 
  (x ∈ Ioo (-2 : ℝ) (4 * (-2)) → 2 * x^2 - m * x + 3 < 2 * (-2)^2 - m * (-2) + 3) ∧
  (x ∈ Ioo (4 * (-2)) (4 : ℝ) → 2 * x^2 - m * x + 3 > 2 * (-2)^2 - m * (-2) + 3)) → 
  (let f : ℝ → ℝ := λ x, 2 * x^2 - m * x + 3 in f 1 = 13) :=
begin
  sorry
end

end calculate_f_at_1_l202_202101


namespace diagonals_in_nonagon_l202_202721

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202721


namespace rectangle_breadth_l202_202260

theorem rectangle_breadth (l b : ℕ) (hl : l = 15) (h : l * b = 15 * b) (h2 : l - b = 10) : b = 5 := 
sorry

end rectangle_breadth_l202_202260


namespace diagonals_in_regular_nine_sided_polygon_l202_202567

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202567


namespace volume_removed_tetrahedra_l202_202313

def side_length : ℝ := 2

def slice_equilateral_triangle : Prop := true -- Meaning each corner forms an equilateral triangle

theorem volume_removed_tetrahedra (h: slice_equilateral_triangle) : 
  ∃ V : ℝ, V = (48 * Real.sqrt 3 - 16) / 9 :=
by
  use (48 * Real.sqrt 3 - 16) / 9
  sorry

end volume_removed_tetrahedra_l202_202313


namespace counting_numbers_leave_remainder_6_divide_53_l202_202130

theorem counting_numbers_leave_remainder_6_divide_53 :
  ∃! n : ℕ, (∃ k : ℕ, 53 = n * k + 6) ∧ n > 6 :=
sorry

end counting_numbers_leave_remainder_6_divide_53_l202_202130


namespace smallest_positive_integer_l202_202446

noncomputable def rot_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

def rotation_period (A : Matrix (Fin 2) (Fin 2) ℝ) (n : ℕ) : Prop :=
  A ^ n = (1 : Matrix (Fin 2) (Fin 2) ℝ)

theorem smallest_positive_integer {
  n,
  1 ≤ n ∧ rotation_period (rot_matrix (145 * Real.pi / 180)) n
} : n = 72 :=
sorry

end smallest_positive_integer_l202_202446


namespace product_of_next_palindromic_year_after_2020_l202_202421

-- Define a function to check if a number reads the same forwards and backwards
def is_palindrome (n : ℕ) : Bool :=
  let str := n.toString
  str == str.reverse

-- Define the digits of a number
def digits (n : ℕ) : List ℕ :=
  n.toString.toList.map (λ c => c.toNat - '0'.toNat)

-- Define the function to calculate the product of the digits
def product_of_digits (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => acc * x) 1

-- Statement to prove
theorem product_of_next_palindromic_year_after_2020 : product_of_digits (digits 2022) = 0 := by
  -- Here's where the proof would go
  sorry

end product_of_next_palindromic_year_after_2020_l202_202421


namespace no_first_quadrant_l202_202497

theorem no_first_quadrant (a b : ℝ) (h_a : a < 0) (h_b : b < 0) (h_am : (a - b) < 0) :
  ¬∃ x : ℝ, (a - b) * x + b > 0 ∧ x > 0 :=
sorry

end no_first_quadrant_l202_202497


namespace math_problem_proof_l202_202514

-- Define the function y
def y (x a : ℝ) : ℝ := -2 * sin x ^ 2 - 2 * a * cos x - 2 * a + 1

-- Define the minimum value function f(a)
def f (a : ℝ) : ℝ :=
if a ≤ -2 then 1
else if -2 < a ∧ a < 2 then -a^2 / 2 - 2 * a - 1
else 1 - 4 * a

-- Assertion: Given the conditions, prove the expressions and maximum value of y
theorem math_problem_proof :
  (∀ x, f 2 = -4 + 1) ∧
  (∀ x, f (-1) = -1/2 * (-1)^2 - 2 * (-1) - 1) ∧
  (∀ x, y x (-1) ≤ 2 * cos x ^ 2 + 2 * cos x + 1) ∧
  (y 0 (-1) = 5) :=
by {
  -- Details of the proof would go here
  sorry
}

end math_problem_proof_l202_202514


namespace diagonals_in_nine_sided_polygon_l202_202743

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202743


namespace union_set_when_m_neg3_range_of_m_for_intersection_l202_202519

def setA (x : ℝ) : Prop := x^2 - x - 12 ≤ 0
def setB (x m : ℝ) : Prop := 2*m - 1 ≤ x ∧ x ≤ m + 1

theorem union_set_when_m_neg3 : 
  (∀ x, setA x ∨ setB x (-3) ↔ -7 ≤ x ∧ x ≤ 4) := 
by sorry

theorem range_of_m_for_intersection :
  (∀ m x, (setA x ∧ setB x m ↔ setB x m) → m ≥ -1) := 
by sorry

end union_set_when_m_neg3_range_of_m_for_intersection_l202_202519


namespace scientific_notation_correct_l202_202230

-- Defining the given number in terms of its scientific notation components.
def million : ℝ := 10^6
def num_million : ℝ := 15.276

-- Expressing the number 15.276 million using its definition.
def fifteen_point_two_seven_six_million : ℝ := num_million * million

-- Scientific notation representation to be proved.
def scientific_notation : ℝ := 1.5276 * 10^7

-- The theorem statement.
theorem scientific_notation_correct :
  fifteen_point_two_seven_six_million = scientific_notation :=
by
  sorry

end scientific_notation_correct_l202_202230


namespace num_values_between_l202_202919

theorem num_values_between (x y : ℕ) (h1 : x + y ≥ 200) (h2 : x + y ≤ 1000) 
  (h3 : (x * (x - 1) + y * (y - 1)) * 2 = (x + y) * (x + y - 1)) : 
  ∃ n : ℕ, n - 1 = 17 := by
  sorry

end num_values_between_l202_202919


namespace max_gift_sets_l202_202004

theorem max_gift_sets (chocolates candies chocolates_left candies_left : ℕ) (h1 : chocolates = 69) (h2 : candies = 86) (h3 : chocolates_left = 5) (h4 : candies_left = 6) : 
  let chocolates_div := chocolates - chocolates_left, candies_div := candies - candies_left in 
  ∃ (n : ℕ), n = Nat.gcd chocolates_div candies_div ∧ n = 16 :=
by
  have chocolates_div : ℕ := 64
  have candies_div : ℕ := 80
  use Nat.gcd chocolates_div candies_div
  sorry

end max_gift_sets_l202_202004


namespace sara_staircase_l202_202932

theorem sara_staircase (n : ℕ) (h : 2 * n * (n + 1) = 360) : n = 13 :=
sorry

end sara_staircase_l202_202932


namespace nine_sided_polygon_diagonals_l202_202635

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202635


namespace total_cost_of_hats_l202_202865

-- Definition of conditions
def weeks := 2
def days_per_week := 7
def cost_per_hat := 50

-- Definition of the number of hats
def num_hats := weeks * days_per_week

-- Statement of the problem
theorem total_cost_of_hats : num_hats * cost_per_hat = 700 := 
by sorry

end total_cost_of_hats_l202_202865


namespace regular_ngon_on_lattice_l202_202920

theorem regular_ngon_on_lattice (n : ℕ) (h : n ≠ 4) : 
  ¬∃ (A : fin n → ℤ × ℤ), (∀ i : fin n, ∃ k m : ℤ, 
    ((A i.1 - A i.2) = (k, m)) ∧ 
    ((sqrt (k^2 + m^2) = A (i + 1).1 - A (i + 1).2)) ∧ 
    (∃ j : fin n, A j = A i ∧ sqrt (k^2 + m^2) ∈ ℚ)) :=
  sorry

end regular_ngon_on_lattice_l202_202920


namespace solve_eq_l202_202199

theorem solve_eq {x : ℝ} (h: (⌊ tan x ⌋ = 2 * cos x ^ 2)) :
  ∃ (k : ℤ), x = k * π + π / 4 :=
by
  sorry

end solve_eq_l202_202199


namespace perceived_temperature_difference_l202_202000

theorem perceived_temperature_difference (N : ℤ) (M L : ℤ)
  (h1 : M = L + N)
  (h2 : M - 11 - (L + 5) = 6 ∨ M - 11 - (L + 5) = -6) :
  N = 22 ∨ N = 10 := by
  sorry

end perceived_temperature_difference_l202_202000


namespace num_divisors_47_gt_6_l202_202123

theorem num_divisors_47_gt_6 : (finset.filter (λ d, d > 6) (finset.divisors 47)).card = 1 :=
by 
  sorry

end num_divisors_47_gt_6_l202_202123


namespace diagonals_in_regular_nine_sided_polygon_l202_202533

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202533


namespace M_intersection_N_is_empty_l202_202244

-- Define sets of lines and parabolas
def is_line (f : ℝ → ℝ) : Prop := ∃ (m b : ℝ), ∀ x, f x = m * x + b

def is_parabola (f : ℝ → ℝ) : Prop := ∃ (a b c : ℝ) (h : a ≠ 0), ∀ x, f x = a * x ^ 2 + b * x + c

-- Define sets M and N
def M : set (ℝ → ℝ) := {f | is_line f}
def N : set (ℝ → ℝ) := {f | is_parabola f}

-- Prove the intersection of M and N is empty
theorem M_intersection_N_is_empty : M ∩ N = ∅ :=
by sorry

end M_intersection_N_is_empty_l202_202244


namespace diagonals_in_regular_nine_sided_polygon_l202_202570

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202570


namespace diagonals_in_regular_nine_sided_polygon_l202_202546

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202546


namespace line_equation_l202_202286

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_sq := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_sq) • u

theorem line_equation :
  ∀ (x y : ℝ), projection (4, 3) (x, y) = (-4, -3) → y = (-4 / 3) * x - 25 / 3 :=
by
  intros x y h
  sorry

end line_equation_l202_202286


namespace diagonals_in_nine_sided_polygon_l202_202784

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202784


namespace nine_sided_polygon_diagonals_l202_202589

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202589


namespace alice_sales_surplus_l202_202385

-- Define the constants
def adidas_cost : ℕ := 45
def nike_cost : ℕ := 60
def reebok_cost : ℕ := 35
def quota : ℕ := 1000

-- Define the quantities sold
def adidas_sold : ℕ := 6
def nike_sold : ℕ := 8
def reebok_sold : ℕ := 9

-- Calculate total sales
def total_sales : ℕ := adidas_sold * adidas_cost + nike_sold * nike_cost + reebok_sold * reebok_cost

-- Prove that Alice's total sales minus her quota is 65
theorem alice_sales_surplus : total_sales - quota = 65 := by
  -- Calculation is omitted here. Here is the mathematical fact to prove:
  sorry

end alice_sales_surplus_l202_202385


namespace not_in_range_of_g_l202_202890

def g : ℝ → ℤ
| x => if x > -3 then ⌈ 1 / (x + 3) ⌉ else ⌊ 1 / (x + 3) ⌋

theorem not_in_range_of_g : ¬ (∃ x : ℝ, g x = 0) :=
by
  sorry

end not_in_range_of_g_l202_202890


namespace complex_modulus_l202_202819

theorem complex_modulus {z : ℂ} (h: conj z = (1 + 2 * complex.i) / complex.i) : complex.abs z = real.sqrt 5 := by
  sorry

end complex_modulus_l202_202819


namespace paint_per_door_l202_202403

variable (cost_per_pint : ℕ) (cost_per_gallon : ℕ) (num_doors : ℕ) (pints_per_gallon : ℕ) (savings : ℕ)

theorem paint_per_door :
  cost_per_pint = 8 →
  cost_per_gallon = 55 →
  num_doors = 8 →
  pints_per_gallon = 8 →
  savings = 9 →
  (pints_per_gallon / num_doors = 1) :=
by
  intros h_cpint h_cgallon h_nd h_pgallon h_savings
  sorry

end paint_per_door_l202_202403


namespace c_symmetry_l202_202483

def c : ℕ → ℕ → ℕ
| n, 0       => 1
| n, n       => 1
| (n + 1), k => if k = 0 then 1 else 2^k * c n k + c n (k - 1)

theorem c_symmetry (n k : ℕ) (h1 : n ≥ k) : c n k = c n (n - k) := 
sorry

end c_symmetry_l202_202483


namespace total_students_l202_202296

theorem total_students (orchestra band choir_boys choir_girls : ℕ)
  (h_orchestra : orchestra = 20)
  (h_band : band = 2 * orchestra)
  (h_choir_boys : choir_boys = 12)
  (h_choir_girls : choir_girls = 16)
  (h_disjoint : ∀ x, x ∈ orchestra ∨ x ∈ band ∨ x ∈ (choir_boys + choir_girls) → 
                    (x ∈ orchestra → x ∉ band ∧ x ∉ (choir_boys + choir_girls)) ∧
                    (x ∈ band → x ∉ orchestra ∧ x ∉ (choir_boys + choir_girls)) ∧
                    (x ∈ (choir_boys + choir_girls) → x ∉ orchestra ∧ x ∉ band)) :
  orchestra + band + (choir_boys + choir_girls) = 88 := 
by
  rw [h_orchestra, h_band, h_choir_boys, h_choir_girls]
  show 20 + 2 * 20 + (12 + 16) = 88 from
  calc
    20 + 2 * 20 + (12 + 16) = 20 + 40 + 28 : by rfl
    ...                      = 88            : by rfl

end total_students_l202_202296


namespace combined_semicircles_perimeter_l202_202308

noncomputable def semicircle_perimeter (r : ℝ) : ℝ :=
  (π * r) + (2 * r)

noncomputable def combined_perimeter_approx (r1 r2 : ℝ) : ℝ :=
  (π * r1 + 2 * r1) + (π * r2 + 2 * r2) - (2 * r1 + 2 * r2)

theorem combined_semicircles_perimeter :
  let r1 := 3 * Real.sqrt 2
  let r2 := 4 * π
  let total_perimeter := combined_perimeter_approx r1 r2
  abs (total_perimeter - 52.8117) < 0.1 :=
by
  let r1 := 3 * Real.sqrt 2
  let r2 := 4 * π
  let total_perimeter := combined_perimeter_approx r1 r2
  sorry

end combined_semicircles_perimeter_l202_202308


namespace nine_sided_polygon_diagonals_l202_202638

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202638


namespace perpendicular_lines_l202_202109

def line_l1 (m x y : ℝ) : Prop := m * x - y + 1 = 0
def line_l2 (m x y : ℝ) : Prop := 2 * x - (m - 1) * y + 1 = 0

theorem perpendicular_lines (m : ℝ): (∃ x y : ℝ, line_l1 m x y) ∧ (∃ x y : ℝ, line_l2 m x y) ∧ (∀ x y : ℝ, line_l1 m x y → line_l2 m x y → m * (2 / (m - 1)) = -1) → m = 1 / 3 := by
  sorry

end perpendicular_lines_l202_202109


namespace abs_eq_of_unique_solution_l202_202908

theorem abs_eq_of_unique_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
    (unique_solution : ∃! x : ℝ, a * (x - a) ^ 2 + b * (x - b) ^ 2 = 0) :
    |a| = |b| :=
sorry

end abs_eq_of_unique_solution_l202_202908


namespace find_m_given_slope_condition_l202_202093

variable (m : ℝ)

theorem find_m_given_slope_condition
  (h : (m - 4) / (3 - 2) = 1) : m = 5 :=
sorry

end find_m_given_slope_condition_l202_202093


namespace multiply_difference_of_cubes_l202_202906

def multiply_and_simplify (x : ℝ) : ℝ :=
  (x^4 + 25 * x^2 + 625) * (x^2 - 25)

theorem multiply_difference_of_cubes (x : ℝ) :
  multiply_and_simplify x = x^6 - 15625 :=
by
  sorry

end multiply_difference_of_cubes_l202_202906


namespace nine_sided_polygon_diagonals_l202_202648

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202648


namespace find_a_l202_202430

theorem find_a (r s a : ℚ) (h₁ : 2 * r * s = 18) (h₂ : s^2 = 16) (h₃ : a = r^2) : 
  a = 81 / 16 := 
sorry

end find_a_l202_202430


namespace diagonals_in_regular_nine_sided_polygon_l202_202534

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202534


namespace johns_hats_cost_l202_202861

theorem johns_hats_cost 
  (weeks : ℕ)
  (days_in_week : ℕ)
  (cost_per_hat : ℕ) 
  (h : weeks = 2 ∧ days_in_week = 7 ∧ cost_per_hat = 50) 
  : (weeks * days_in_week * cost_per_hat) = 700 :=
by
  sorry

end johns_hats_cost_l202_202861


namespace nine_sided_polygon_diagonals_l202_202662

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202662


namespace southeast_corner_visits_l202_202367

theorem southeast_corner_visits
  (n : ℕ) (p : ℕ)
  (H_odd_n : n % 2 = 1)
  (H_grid : n = 2003)
  (H_room_visits : ∀ i j : ℕ, (i = n - 1 ∧ j = n - 1) ∨ (i ≠ 0 ∨ j ≠ 0) → visits(i, j) = if (i = n - 1 ∧ j = n - 1) then x else if (i = 0 ∧ j = 0) then 2 else 100) :
  x = 99 := 
sorry

end southeast_corner_visits_l202_202367


namespace tail_length_third_generation_l202_202183

theorem tail_length_third_generation (initial_length : ℕ) (growth_rate : ℕ) :
  initial_length = 16 ∧ growth_rate = 25 → 
  let sec_len := initial_length * (100 + growth_rate) / 100 in
  let third_len := sec_len * (100 + growth_rate) / 100 in
  third_len = 25 := by
  intros h
  sorry

end tail_length_third_generation_l202_202183


namespace shop_charges_per_object_l202_202914

-- Definitions based on the conditions and final answer.
def number_of_people : ℕ := 3
def total_cost : ℝ := 165
def total_number_of_objects : ℕ := 6 + 6 + 3 -- 6 single shoes + 6 single socks + 3 mobiles

noncomputable def charge_per_object : ℝ := total_cost / total_number_of_objects

theorem shop_charges_per_object : charge_per_object = 11 :=
by
  rw [charge_per_object, total_cost, total_number_of_objects]
  norm_num
  sorry

end shop_charges_per_object_l202_202914


namespace mariel_dogs_count_l202_202902

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

end mariel_dogs_count_l202_202902


namespace no_gar_is_tren_l202_202520

variables (Gar Plin Tren : Type) 
variables (IsGar : Gar → Prop) (IsPlin : Plin → Prop) (IsTren : Tren → Prop)

/-- All Gars are Plins --/
variable (H1 : ∀ x, IsGar x → IsPlin x)

/-- No Plins are Trens --/
variable (H2 : ∀ y, IsPlin y → ¬ IsTren y)

theorem no_gar_is_tren :
  ∀ x, IsGar x → ¬ IsTren x :=
by
  intros x HGar HTren
  specialize H1 x HGar
  exact H2 x H1 HTren

end no_gar_is_tren_l202_202520


namespace diagonals_in_nonagon_l202_202726

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202726


namespace area_of_triangle_l202_202940

theorem area_of_triangle : 
  let line_eq := λ x: ℝ, x + 2 in
  let intersection_x := (0, line_eq 0) in
  let intersection_y := (-(line_eq 0), 0) in
  (1 / 2) * (abs (0 - intersection_y.1)) * (abs (0 - intersection_x.2)) = 2 :=
by
  sorry

end area_of_triangle_l202_202940


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202602

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202602


namespace regular_nine_sided_polygon_diagonals_l202_202758

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202758


namespace volume_pyramid_D_BD1E_l202_202842

noncomputable def volume_pyramid (a b c : ℝ) : ℝ := (1/3) * a * b * c

theorem volume_pyramid_D_BD1E :
  let A B C D A1 B1 C1 D1 E : ℝ := 2 in
  E = (C + D) / 2 →
  a = 1 →
  b = 2 →
  c = 1 →
  d = 2 →
  volume_pyramid a b c = 2/3 :=
by
  intro hE ha hb hc hd
  rw [←hE, ←ha, ←hb, ←hc, ←hd]
  sorry

end volume_pyramid_D_BD1E_l202_202842


namespace inequality_always_negative_l202_202148

theorem inequality_always_negative (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3 / 4 < 0) ↔ (-3 < k ∧ k ≤ 0) :=
by
  -- Proof omitted
  sorry

end inequality_always_negative_l202_202148


namespace maximum_value_l202_202443

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (x + 64) + sqrt (20 - x) + sqrt (2 * x)

theorem maximum_value :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 20 → f x ≤ sqrt 285.72 :=
by
  intros x hx
  sorry

end maximum_value_l202_202443


namespace max_value_sincos_sum_l202_202049

theorem max_value_sincos_sum (x y z : ℝ) :
  (∀ x y z, (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5) :=
by sorry

end max_value_sincos_sum_l202_202049


namespace tom_divided_by_lara_l202_202305

theorem tom_divided_by_lara : 
  (2 * (finset.range 300).sum (λ k, (k + 1))) / ((finset.range 200).sum (λ k, (k + 1))) = 4.5 :=
by
  sorry

end tom_divided_by_lara_l202_202305


namespace nine_sided_polygon_diagonals_l202_202590

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202590


namespace inscribed_circle_radius_l202_202983

noncomputable def radius_inscribed_circle (O1 O2 D : ℝ × ℝ) (r1 r2 : ℝ) :=
  if (r1 = 2 ∧ r2 = 6) ∧ ((O1.fst - O2.fst)^2 + (O1.snd - O2.snd)^2 = 64) then
    2 * (Real.sqrt 3 - 1)
  else
    0

theorem inscribed_circle_radius (O1 O2 D : ℝ × ℝ) (r1 r2 : ℝ)
  (h1 : r1 = 2) (h2 : r2 = 6)
  (h3 : (O1.fst - O2.fst)^2 + (O1.snd - O2.snd)^2 = 64) :
  radius_inscribed_circle O1 O2 D r1 r2 = 2 * (Real.sqrt 3 - 1) :=
by
  sorry

end inscribed_circle_radius_l202_202983


namespace total_cost_of_hats_l202_202866

-- Definition of conditions
def weeks := 2
def days_per_week := 7
def cost_per_hat := 50

-- Definition of the number of hats
def num_hats := weeks * days_per_week

-- Statement of the problem
theorem total_cost_of_hats : num_hats * cost_per_hat = 700 := 
by sorry

end total_cost_of_hats_l202_202866


namespace diagonals_in_regular_nine_sided_polygon_l202_202566

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202566


namespace y_intercept_of_line_l202_202322

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 5 * y = 10) : 
  3 * 0 - 5 * y = 10 → y = -2 :=
by
  intros h0
  rw [mul_zero, sub_eq_add_neg, add_zero] at h0
  exact eq_of_mul_eq_mul_left (ne_of_gt (by norm_num)) (by simpa only [neg_mul, one_mul, neg_eq_iff_neg_eq] using h0)

end y_intercept_of_line_l202_202322


namespace john_expense_l202_202191

theorem john_expense (computer_cost : ℕ) (accessories_cost : ℕ) (playstation_value : ℕ) (discount_rate : ℚ) :
  computer_cost = 700 →
  accessories_cost = 200 →
  playstation_value = 400 →
  discount_rate = 0.20 →
  let selling_price := playstation_value - (playstation_value * discount_rate).to_nat in
  let total_cost := computer_cost + accessories_cost in
  let pocket_outcome := total_cost - selling_price in
  pocket_outcome = 580 :=
by
  intros
  sorry

end john_expense_l202_202191


namespace complex_translation_proof_l202_202986

noncomputable def translation1 : ℂ := -7 - -3 + -i - 2 * I
noncomputable def translation2 : ℂ := -10 - -7 - I

theorem complex_translation_proof :
  let w1 := translation1 in
  let w2 := translation2 in
  (-4 + 5 * I + w1 + w2) = -11 + 3 * I :=
by 
  -- proof skipped
  sorry

end complex_translation_proof_l202_202986


namespace negation_of_existence_l202_202082

theorem negation_of_existence:
  (¬ (∃ (x_0 : ℝ), 2 ^ x_0 ≠ 1)) → (∀ (x_0 : ℝ), 2 ^ x_0 = 1) :=
by
  sorry

end negation_of_existence_l202_202082


namespace average_of_all_results_l202_202263

noncomputable def average (lst : List ℝ) : ℝ := (list.sum lst) / (list.length lst)

theorem average_of_all_results (results1 results2 : List ℝ)
  (h1 : results1.length = 40) (h2 : average results1 = 30)
  (h3 : results2.length = 30) (h4 : average results2 = 40) :
  average (results1 ++ results2) = 34.2857 :=
by
  sorry

end average_of_all_results_l202_202263


namespace diagonals_in_nine_sided_polygon_l202_202785

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202785


namespace possible_values_of_expression_l202_202055

theorem possible_values_of_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  let sign (x : ℝ) := x / |x| in
  let expression := sign a + sign b + sign c + sign (a * b * c) in
  expression ∈ {-4, 0, 4} :=
sorry

end possible_values_of_expression_l202_202055


namespace maximum_speed_l202_202950

variable (A S ρ v0 v : ℝ)

def force (A S ρ v0 v : ℝ) : ℝ := (A * S * ρ * (v0 - v) ^ 2) / 2

def power (A S ρ v0 v : ℝ) : ℝ := force A S ρ v0 v * v

theorem maximum_speed : 
  S = 4 ∧ v0 = 4.8 ∧ 
  ∃ A ρ v,
    (∀ v, power A S ρ v0 v ≤ power A S ρ v0 1.6) → 
    v == 1.6 :=
by
  sorry

end maximum_speed_l202_202950


namespace inversely_proportional_example_l202_202254

theorem inversely_proportional_example (x y k : ℝ) (h₁ : x * y = k) (h₂ : x = 30) (h₃ : y = 8) :
  y = 24 → x = 10 :=
by
  sorry

end inversely_proportional_example_l202_202254


namespace value_of_a_minus_b_l202_202809

theorem value_of_a_minus_b (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 :=
by
  sorry

end value_of_a_minus_b_l202_202809


namespace minimum_YP_PQ_QZ_value_l202_202828

theorem minimum_YP_PQ_QZ_value (X Y Z P Q : Point)
  (h_triangle : triangle X Y Z)
  (h_angle : angle X Y Z = 30)
  (h_XY : dist X Y = 12)
  (h_XZ : dist X Z = 8)
  (h_P_on_XY : on_segment P X Y)
  (h_Q_on_XZ : on_segment Q X Z) :
  min_value (dist Y P + dist P Q + dist Q Z) = sqrt (208 + 96 * sqrt 3) := by
  sorry

end minimum_YP_PQ_QZ_value_l202_202828


namespace regular_nine_sided_polygon_diagonals_l202_202755

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202755


namespace cone_generatrix_length_l202_202942

theorem cone_generatrix_length (r : ℝ) (l : ℝ) (h_radius : r = 6) (h_lateral_semicircle : 2 * π * r = π * l) : 
  l = 12 :=
by
  have : 2 * 6 = l,
  { rw [←h_lateral_semicircle, h_radius], ring },
  rw ←this,
  norm_num
  sorry

end cone_generatrix_length_l202_202942


namespace scientific_notation_of_000000301_l202_202428

/--
Expressing a small number in scientific notation:
Prove that \(0.000000301\) can be written as \(3.01 \times 10^{-7}\).
-/
theorem scientific_notation_of_000000301 :
  0.000000301 = 3.01 * 10 ^ (-7) :=
sorry

end scientific_notation_of_000000301_l202_202428


namespace candy_count_l202_202459

def initial_candy : ℕ := 47
def eaten_candy : ℕ := 25
def sister_candy : ℕ := 40
def final_candy : ℕ := 62

theorem candy_count : initial_candy - eaten_candy + sister_candy = final_candy := 
by
  sorry

end candy_count_l202_202459


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202598

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202598


namespace sum_arithmetic_sequence_geom_series_l202_202087

section ArithmeticSequenceSum

variable (a₁ d q : ℝ) (n : ℕ)

#check q ≠ 0 ∧ q ≠ 1 -- Ensure q is not equal to 0 or 1
#check a₁ + d * (n - 1) -- Arithmetic sequence term

theorem sum_arithmetic_sequence_geom_series 
    (h₁ : q ≠ 0) 
    (h₂ : q ≠ 1) :
    let a := a₁,
        a_n := λ i : ℕ, a + d * (i - 1),
        S_n := ∑ i in finset.range n, a_n i * q ^ (i - 1)
    in
    S_n = a₁ * (1 - q)⁻¹ + (d * q - a_n (n-1) * q^n) * (1 - q)⁻²
:= sorry

end ArithmeticSequenceSum

end sum_arithmetic_sequence_geom_series_l202_202087


namespace y_intercept_of_line_l202_202324

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 5 * y = 10) : 
  3 * 0 - 5 * y = 10 → y = -2 :=
by
  intros h0
  rw [mul_zero, sub_eq_add_neg, add_zero] at h0
  exact eq_of_mul_eq_mul_left (ne_of_gt (by norm_num)) (by simpa only [neg_mul, one_mul, neg_eq_iff_neg_eq] using h0)

end y_intercept_of_line_l202_202324


namespace max_abs_z_2_2i_l202_202351

open Complex

theorem max_abs_z_2_2i (z : ℂ) (h : abs (z + 2 - 2 * I) = 1) : 
  ∃ w : ℂ, abs (w - 2 - 2 * I) = 5 :=
sorry

end max_abs_z_2_2i_l202_202351


namespace max_trig_expression_l202_202043

open Real

theorem max_trig_expression (x y z : ℝ) :
  (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 := sorry

end max_trig_expression_l202_202043


namespace nine_sided_polygon_diagonals_l202_202587

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202587


namespace max_min_product_is_minus_one_l202_202066

noncomputable def max_min_product_of_numbers (a : Fin 2018 → ℝ) (h1 : ∑ i, a i = 0)
  (h2 : ∑ i, (a i)^2 = 2018) : ℝ :=
  sorry

theorem max_min_product_is_minus_one (a : Fin 2018 → ℝ) (h1 : ∑ i, a i = 0)
  (h2 : ∑ i, (a i)^2 = 2018) : max_min_product_of_numbers a h1 h2 = -1 :=
  sorry

end max_min_product_is_minus_one_l202_202066


namespace counting_numbers_leave_remainder_6_divide_53_l202_202128

theorem counting_numbers_leave_remainder_6_divide_53 :
  ∃! n : ℕ, (∃ k : ℕ, 53 = n * k + 6) ∧ n > 6 :=
sorry

end counting_numbers_leave_remainder_6_divide_53_l202_202128


namespace average_student_headcount_l202_202417

def student_headcount_fall_0203 : ℕ := 11700
def student_headcount_fall_0304 : ℕ := 11500
def student_headcount_fall_0405 : ℕ := 11600

theorem average_student_headcount : 
  (student_headcount_fall_0203 + student_headcount_fall_0304 + student_headcount_fall_0405) / 3 = 11600 := by
  sorry

end average_student_headcount_l202_202417


namespace problem1_problem2_problem3_l202_202209

noncomputable def f (x : ℝ) : ℝ := 1 / 2 + Real.log (x / (1 - x)) / Real.log 2

theorem problem1 (x1 x2 : ℝ) (h : x1 + x2 = 1) : 
  f x1 + f x2 = 1 :=
sorry

noncomputable def Sn (n : ℕ) : ℝ := 
  ∑ k in Finset.range n, f ((k + 1 : ℝ) / (n + 1 : ℝ))

theorem problem2 (n : ℕ) (h : 0 < n) : Sn n = n / 2 := 
sorry

noncomputable def an (n : ℕ) : ℝ := (1 / (Sn n + 1))^2

noncomputable def Tn (n : ℕ) : ℝ := 
  ∑ i in Finset.range n, an (i + 1)

theorem problem3 (n : ℕ) (h : 0 < n) : 
  4 / 9 ≤ Tn n ∧ Tn n < 5 / 3 :=
sorry

end problem1_problem2_problem3_l202_202209


namespace percentage_of_money_donated_l202_202369

noncomputable def percentage_donated (raised: ℝ) (num_orgs: ℝ) (amount_per_org: ℝ) : ℝ :=
  (num_orgs * amount_per_org / raised) * 100

theorem percentage_of_money_donated (raised: ℝ) (num_orgs: ℝ) (amount_per_org: ℝ)
  (h1: raised = 2500) (h2: num_orgs = 8) (h3: amount_per_org = 250) : percentage_donated raised num_orgs amount_per_org = 80 :=
by
  rw [percentage_donated, h1, h2, h3]
  norm_num
  sorry

end percentage_of_money_donated_l202_202369


namespace counting_number_leaves_remainder_of_6_l202_202127

theorem counting_number_leaves_remainder_of_6:
  ∃! d : ℕ, d > 6 ∧ d ∣ (53 - 6) ∧ 53 % d = 6 :=
begin
  sorry
end

end counting_number_leaves_remainder_of_6_l202_202127


namespace functions_equivalence_l202_202389

-- Definition for Option A
def fA (x : ℤ) : ℤ := x^2
def gA (x : ℤ) : ℤ :=
  if x = 0 then 0 else
  if x = 1 ∨ x = -1 then 1 else 0

-- Definition for Option B
def fB (x : ℝ) : ℝ := x * abs x
def gB (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 else -x^2

-- Definition for Option C
def fC (x : ℝ) : ℝ := x
def gC (x : ℝ) : ℝ := abs x

-- Definition for Option D
def fD (x : ℝ) [H : 0 < x] : ℝ := 1 / x
def gD (x : ℝ) [H : 0 < x] : ℝ := (x + 1) / (x^2 + x)

-- The main theorem
theorem functions_equivalence :
  (∀ x : ℤ, x ∈ {-1, 0, 1} → fA x = gA x) ∧
  (∀ x : ℝ, fB x = gB x) ∧
  (∀ x : ℝ, fC x = gC x → x ≥ 0) ∧
  (∀ (x : ℝ) (H : 0 < x), fD x = gD x) :=
by {
  -- we only provide the statement here, no proof.
  sorry
}

end functions_equivalence_l202_202389


namespace part_i_part_ii_part_iii_l202_202196

/-- Given a right-angled triangle ABC with A = 90°, AH is the altitude from A to BC.
    P and Q are the feet of the perpendiculars from H to AB and AC respectively.
    M is a variable point on the line PQ. The line through M perpendicular to MH
    meets the lines AB at R and AC at S. -/
theorem part_i (ABC : Triangle)
  (A B C H P Q M R S: Point)
  (h_triangle_ABC : right_angle (∠ABC))
  (h_AH : altitude A H BC)
  (h_P : foot P H AB)
  (h_Q : foot Q H AC)
  (h_M : line_segment PQ contains M)
  (h_R : perpendicular_line_through M H intersects AB at R)
  (h_S : perpendicular_line_through M H intersects AC at S) :
  lies_on_circumcircle H A R S := sorry

/-- For any points M and M₁ on the line PQ with corresponding points R, R₁ on AB
    and points S, S₁ on AC, the ratio RR₁/SS₁ is constant. -/
theorem part_ii (ABC M M₁ R R₁ S S₁ P Q : Point)
  (h_line_PQ : line_segment PQ contains M)
  (h_line_PQ₁ : line_segment PQ contains M₁)
  (h_R : perpendicular_line_through M H intersects AB at R)
  (h_R₁ : perpendicular_line_through M₁ H intersects AB at R₁)
  (h_S : perpendicular_line_through M H intersects AC at S)
  (h_S₁ : perpendicular_line_through M₁ H intersects AC at S₁) :
  ratio_eq (line_segment R R₁) (line_segment S S₁) := sorry

/-- Point K is symmetric to H with respect to M. The line through K perpendicular
    to PQ meets RS at D. Prove ∠BHR = ∠DHR and ∠DHS = ∠CHS. -/
theorem part_iii (ABC H K M R S D P Q : Point)
  (h_symmetric_K : symmetric H M K)
  (h_perpendicular_K : perpendicular_line K PQ intersects RS at D)
  (h_BHR_intersect : line_contains B H R)
  (h_DHR_intersect : line_contains D H R)
  (h_DHS_intersect : line_contains D H S) :
  angle_eq (angle B H R) (angle D H R) ∧
  angle_eq (angle D H S) (angle C H S) := sorry

end part_i_part_ii_part_iii_l202_202196


namespace nine_sided_polygon_diagonals_l202_202583

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202583


namespace problem1_problem2_l202_202213

noncomputable def f (x : ℝ) : ℝ :=
if -1 ≤ x ∧ x ≤ 1 then x^3
else if 1 ≤ x ∧ x < 3 then -((x - 2)^3)
else if 3 ≤ x ∧ x ≤ 5 then (x - 4)^3
else 0 -- We won't define it outside [1,5] for simplicity in this statement

def odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

def periodic (f : ℝ → ℝ) := ∀ x k : ℝ, f (x + k) = f x

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
odd f ∧ periodic f ∧ (∀ x, (-1 ≤ x ∧ x ≤ 1) → f x = x^3)

-- Provide the equivalent of the proof goal here
theorem problem1 (x : ℝ) (h : satisfies_conditions f) (h1 : 1 ≤ x ∧ x ≤ 5) :
  (f x = 
  if (1 ≤ x ∧ x < 3) then -(x-2)^3
  else if (3 ≤ x ∧ x ≤ 5) then (x-4)^3
  else 0) :=
sorry

def A (a : ℝ) : set ℝ := {x : ℝ | f x > a}

theorem problem2 (a : ℝ) (h : satisfies_conditions f) :
  (∃ x : ℝ, (x ∈ A a)) ↔ (a < 1) :=
sorry

end problem1_problem2_l202_202213


namespace number_of_diagonals_l202_202554

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202554


namespace a_is_geometric_T_n_value_l202_202290

-- Definitions from conditions
def S (a : ℕ → ℕ) (n : ℕ) := ∑ i in range (n + 1), a i
def T (b : ℕ → ℕ) (n : ℕ) := ∑ i in range (n + 1), b i

-- Given conditions
variables {a b : ℕ → ℕ}
hypotheses
  (ha1 : a 1 = 1)
  (ha2 : ∀ n : ℕ, a (n + 2) = 2 * S a (n + 1) + 1)
  (hb : ∀ n : ℕ, b (n + 1) = b n + 2) -- since d = 2
  (hb_sum : b 1 + b 2 + b 3 = 15)

-- Proof Problem 1: Show a is geometric with ratio 3
theorem a_is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n :=
sorry

-- Additional condition for Proof Problem 2
hypotheses (geom_seq : ∀ n : ℕ, n > 0 → (a (n + 1) + b (n + 1))^2 = (a n + b n) * (a (n + 2) + b (n + 2)))

-- Proof Problem 2: Show T = n^2 + 2n
theorem T_n_value : ∀ n : ℕ, T b n = n^2 + 2n :=
sorry

end a_is_geometric_T_n_value_l202_202290


namespace julie_works_days_per_week_l202_202194

def hourly_rate : ℕ := 5
def daily_hours : ℕ := 8
def monthly_salary_missed_day : ℕ := 920

theorem julie_works_days_per_week : 
  let daily_earnings := hourly_rate * daily_hours in
  let potential_monthly_salary := monthly_salary_missed_day + daily_earnings in
  let workdays_in_month := potential_monthly_salary / daily_earnings in
  let weeks_in_month := 4 in
  workdays_in_month / weeks_in_month = 6 := 
by
  sorry

end julie_works_days_per_week_l202_202194


namespace value_of_k_l202_202822

theorem value_of_k (k : ℝ) (x : ℝ) (h : (k - 3) * x^2 + 6 * x + k^2 - k = 0) (r : x = -1) : 
  k = -3 := 
by
  sorry

end value_of_k_l202_202822


namespace tan_x0_eq_neg_sqrt3_l202_202503

noncomputable def f (x : ℝ) : ℝ := (1/2 : ℝ) * x - (1/4 : ℝ) * Real.sin x - (Real.sqrt 3 / 4 : ℝ) * Real.cos x

theorem tan_x0_eq_neg_sqrt3 (x₀ : ℝ) (h : HasDerivAt f (1 : ℝ) x₀) : Real.tan x₀ = -Real.sqrt 3 := by
  sorry

end tan_x0_eq_neg_sqrt3_l202_202503


namespace greatest_possible_sum_xy_l202_202992

noncomputable def greatest_possible_xy (x y : ℝ) :=
  x^2 + y^2 = 100 ∧ xy = 40 → x + y = 6 * Real.sqrt 5

theorem greatest_possible_sum_xy {x y : ℝ} (h1 : x^2 + y^2 = 100) (h2 : xy = 40) :
  x + y ≤ 6 * Real.sqrt 5 :=
sorry

end greatest_possible_sum_xy_l202_202992


namespace nine_sided_polygon_diagonals_l202_202641

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202641


namespace number_of_solutions_l202_202020

noncomputable def count_solutions_pos_int (n : ℕ) : ℕ :=
  let solutions := { (x, y, z) : ℕ × ℕ × ℕ | 2 * x + 3 * y + z = 1000 ∧ x + y + z = 340 ∧ x > 0 ∧ y > 0 ∧ z > 0 }
  solutions.toFinset.card

theorem number_of_solutions : count_solutions_pos_int 9 = 9 :=
by {
  sorry
}

end number_of_solutions_l202_202020


namespace nine_sided_polygon_diagonals_l202_202650

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202650


namespace minimum_people_with_all_luxuries_l202_202837

theorem minimum_people_with_all_luxuries {P : Type} (N : ℕ) 
  (R T C A : P → Prop) 
  (hR : ∀ p, R p → p ∈ finset.range N → finset.card (finset.filter R (finset.range N)) = 70) 
  (hT : ∀ p, T p → p ∈ finset.range N → finset.card (finset.filter T (finset.range N)) = 75)
  (hC : ∀ p, C p → p ∈ finset.range N → finset.card (finset.filter C (finset.range N)) = 65)
  (hA : ∀ p, A p → p ∈ finset.range N → finset.card (finset.filter A (finset.range N)) = 95)
  (hN : N = 100) : 
  ∃ p, R p ∧ T p ∧ C p ∧ A p → finset.card (finset.filter (λ p, R p ∧ T p ∧ C p ∧ A p) (finset.range N)) = 95 := 
by {
  admit
}

end minimum_people_with_all_luxuries_l202_202837


namespace hats_cost_l202_202869

variables {week_days : ℕ} {weeks : ℕ} {cost_per_hat : ℕ}

-- Conditions
def num_hats (week_days : ℕ) (weeks : ℕ) : ℕ := week_days * weeks
def total_cost (num_hats : ℕ) (cost_per_hat : ℕ) : ℕ := num_hats * cost_per_hat

-- Proof problem
theorem hats_cost (h1 : week_days = 7) (h2 : weeks = 2) (h3 : cost_per_hat = 50) : 
  total_cost (num_hats week_days weeks) cost_per_hat = 700 :=
by 
  sorry

end hats_cost_l202_202869


namespace piravena_trip_distance_l202_202233

theorem piravena_trip_distance :
  let XZ := 4000
  let XY := 4500
  let YZ := Real.sqrt (XY^2 - XZ^2)
  XY + YZ + XZ = 10562 :=
by
  let XZ := 4000
  let XY := 4500
  let YZ := Real.sqrt (XY^2 - XZ^2)
  calc
    XY + YZ + XZ = 4500 + Real.sqrt (4500^2 - 4000^2) + 4000 := by sorry
                ... = 4500 + 2062 + 4000                      := by sorry
                ... = 10562                                    := by sorry

end piravena_trip_distance_l202_202233


namespace diagonals_in_nonagon_l202_202732

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202732


namespace dodecagon_to_square_l202_202072

theorem dodecagon_to_square (s : ℝ) :
  ∃ a : ℝ, (3 * (2 + real.sqrt 3) * s^2 = a^2) :=
sorry

end dodecagon_to_square_l202_202072


namespace find_missing_number_l202_202426

theorem find_missing_number (x : ℕ) (h : 10111 - x * 2 * 5 = 10011) : x = 5 := 
sorry

end find_missing_number_l202_202426


namespace greatest_k_value_l202_202442

theorem greatest_k_value 
  (k : ℝ)
  (h_eq : ∀ x, x^2 + k * x + 8 = 0)
  (h_diff_roots : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 - r2).abs = sqrt 85 ∧ (r1^2 + k * r1 + 8 = 0) ∧ (r2^2 + k * r2 + 8 = 0)) :
  k = sqrt 117 :=
by
  sorry

end greatest_k_value_l202_202442


namespace area_of_kite_l202_202460

theorem area_of_kite (A B C D : ℝ × ℝ) (hA : A = (2, 3)) (hB : B = (6, 7)) (hC : C = (10, 3)) (hD : D = (6, 0)) : 
  let base := (C.1 - A.1)
  let height := (B.2 - D.2)
  let area := 2 * (1 / 2 * base * height)
  area = 56 := 
by
  sorry

end area_of_kite_l202_202460


namespace diagonals_in_regular_nine_sided_polygon_l202_202542

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202542


namespace third_generation_tail_length_is_25_l202_202185

def first_generation_tail_length : ℝ := 16
def growth_rate : ℝ := 0.25

def second_generation_tail_length : ℝ := first_generation_tail_length * (1 + growth_rate)
def third_generation_tail_length : ℝ := second_generation_tail_length * (1 + growth_rate)

theorem third_generation_tail_length_is_25 :
  third_generation_tail_length = 25 := by
  sorry

end third_generation_tail_length_is_25_l202_202185


namespace number_of_diagonals_l202_202549

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202549


namespace m_equals_p_of_odd_prime_and_integers_l202_202211

theorem m_equals_p_of_odd_prime_and_integers (p m : ℕ) (x y : ℕ) (hp : p > 1 ∧ ¬ (p % 2 = 0)) 
    (hx : x > 1) (hy : y > 1) 
    (h : (x ^ p + y ^ p) / 2 = ((x + y) / 2) ^ m): 
    m = p := 
by 
  sorry

end m_equals_p_of_odd_prime_and_integers_l202_202211


namespace range_of_k_l202_202149

noncomputable def quadratic_inequality (k x : ℝ) : ℝ :=
  k * x^2 + 2 * k * x - (k + 2)

theorem range_of_k :
  (∀ x : ℝ, quadratic_inequality k x < 0) ↔ -1 < k ∧ k < 0 :=
by
  sorry

end range_of_k_l202_202149


namespace probability_of_drawing_white_ball_is_zero_l202_202298

theorem probability_of_drawing_white_ball_is_zero
  (red_balls blue_balls : ℕ)
  (h1 : red_balls = 3)
  (h2 : blue_balls = 5)
  (white_balls : ℕ)
  (h3 : white_balls = 0) : 
  (0 / (red_balls + blue_balls + white_balls) = 0) :=
sorry

end probability_of_drawing_white_ball_is_zero_l202_202298


namespace find_r6_s6_roots_l202_202884

theorem find_r6_s6_roots :
  let r s : ℂ in
  (r^2 - 2*r*(sqrt 7 : ℂ) + 1 = 0) ∧ (s^2 - 2*s*(sqrt 7 : ℂ) + 1 = 0) →
  r^6 + s^6 = 389374 :=
by
  sorry

end find_r6_s6_roots_l202_202884


namespace number_of_diagonals_l202_202563

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202563


namespace channel_cross_section_area_l202_202266

def area_trapezium (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

theorem channel_cross_section_area :
  area_trapezium 12 6 70 = 630 :=
by
  sorry

end channel_cross_section_area_l202_202266


namespace regular_nonagon_diagonals_correct_l202_202714

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202714


namespace noelle_total_assignments_l202_202228

-- Given conditions
def points_needed : ℕ := 30
def assignments_per_point : ℕ → ℕ
| n :=
  if n <= 6 then 1
  else if n <= 12 then 2
  else if n <= 18 then 3
  else if n <= 24 then 4
  else 5

-- Calculate total assignments
def total_assignments : ℕ :=
  ((list.range (points_needed // 6)).map (λ i, assignments_per_point (6 * (i + 1)) * 6)).sum

-- Proof statement
theorem noelle_total_assignments : total_assignments = 90 :=
by sorry

end noelle_total_assignments_l202_202228


namespace parallel_lines_ratio_l202_202979

theorem parallel_lines_ratio (A B C O F E N : Point) 
  (hO: is_internal_point O (triangle A B C))
  (hFO: parallel_line_through_point O F A C)
  (hEO: parallel_line_through_point O E A B)
  (hNO: parallel_line_through_point O N B C):
  ∃ k l m : ℝ,
    k = |AF| / |AB| ∧
    l = |BE| / |BC| ∧
    m = |CN| / |CA| ∧
    k + l + m = 1 :=
by 
  sorry

end parallel_lines_ratio_l202_202979


namespace volleyball_team_matches_at_least_800_l202_202300

noncomputable def volleyball_team_matches : Prop :=
  ∃ (T : Finset (Fin 1000)), 
    (∀ (t ∈ T), 
      t.matches_count = 2000 ∧ 
      (∀ (A B : Fin 1000), A ∈ T → B ∈ T → 
        (A.won_against B) → 
        (∀ C, C ∉ B.played_against → A.won_against C)
      )
    )

theorem volleyball_team_matches_at_least_800 :
  volleyball_team_matches :=
begin
  sorry
end

end volleyball_team_matches_at_least_800_l202_202300


namespace monotonic_intervals_max_n_for_g_sin_sum_lt_ln3_l202_202102

noncomputable def f (x : ℝ) := x - Real.log x - 1

theorem monotonic_intervals (x : ℝ) : 
  (x ∈ Ioo 0 1 → deriv f x < 0) ∧ (x ∈ Ioi 1 → deriv f x > 0) :=
sorry

noncomputable def g (x : ℝ) := f x + x * Real.log x - 2 * x

theorem max_n_for_g (n : ℤ) : 
  (∀ x > 0, g x ≥ n) → n ≤ -3 :=
sorry

theorem sin_sum_lt_ln3 (n : ℕ) (hn : 0 < n) : 
  (Finset.sum (Finset.range (n+1) 3 * n + 1) (λ k, Real.sin (1 / (n + k + 1)))) < Real.log 3 :=
sorry

end monotonic_intervals_max_n_for_g_sin_sum_lt_ln3_l202_202102


namespace max_value_of_f_l202_202963

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 5) * Real.sin (x + Real.pi / 3) + Real.cos (x - Real.pi / 6)

theorem max_value_of_f : 
  ∃ x : ℝ, f x = 6 / 5 ∧ ∀ y : ℝ, f y ≤ 6 / 5 :=
sorry

end max_value_of_f_l202_202963


namespace regular_nine_sided_polygon_diagonals_l202_202666

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202666


namespace number_of_diagonals_l202_202560

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202560


namespace photos_uploaded_from_phone_l202_202229

theorem photos_uploaded_from_phone (P : ℕ) : 
  (∃ (P : ℕ), P + 35 = 40) →
  P = 5 :=
by
  intro h
  cases h with P hP
  have h : P = 40 - 35 := by sorry
  show P = 5 from sorry

end photos_uploaded_from_phone_l202_202229


namespace number_of_diagonals_l202_202556

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202556


namespace diagonals_in_nonagon_l202_202719

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202719


namespace find_real_triples_l202_202028

theorem find_real_triples :
  ∀ (a b c : ℝ), a^2 + a * b + c = 0 ∧ b^2 + b * c + a = 0 ∧ c^2 + c * a + b = 0
  ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = -1/2 ∧ b = -1/2 ∧ c = -1/2) :=
by
  sorry

end find_real_triples_l202_202028


namespace coin_order_is_correct_l202_202291

def Coin : Type := ℕ
def F : Coin := 0
def A : Coin := 1
def B : Coin := 2
def C : Coin := 3
def D : Coin := 4
def E : Coin := 5

def covers (x y : Coin) (order : List Coin) : Prop :=
  order.indexOf x < order.indexOf y

noncomputable def correct_order : List Coin := [F, A, D, E, C, B]

theorem coin_order_is_correct (order : List Coin) : 
  (order = correct_order) ↔ 
  covers F A order ∧ 
  covers F B order ∧ 
  covers F C order ∧ 
  covers F D order ∧ 
  covers F E order ∧ 
  covers A B order ∧ 
  covers A C order ∧ 
  covers D E order ∧ 
  covers E B order ∧ 
  covers F C order ∧ 
  covers A C order :=
sorry

end coin_order_is_correct_l202_202291


namespace jacoby_needs_l202_202850

-- Given conditions
def total_goal : ℤ := 5000
def job_earnings_per_hour : ℤ := 20
def total_job_hours : ℤ := 10
def cookie_price_each : ℤ := 4
def total_cookies_sold : ℤ := 24
def lottery_ticket_cost : ℤ := 10
def lottery_winning : ℤ := 500
def gift_from_sister_one : ℤ := 500
def gift_from_sister_two : ℤ := 500

-- Total money Jacoby has so far
def current_total_money : ℤ := 
  job_earnings_per_hour * total_job_hours +
  cookie_price_each * total_cookies_sold +
  lottery_winning +
  gift_from_sister_one + gift_from_sister_two -
  lottery_ticket_cost

-- The amount Jacoby needs to reach his goal
def amount_needed : ℤ := total_goal - current_total_money

-- The main statement to be proved
theorem jacoby_needs : amount_needed = 3214 := by
  -- The proof is skipped
  sorry

end jacoby_needs_l202_202850


namespace school_count_l202_202158

theorem school_count (n : ℕ) (h1 : 2 * n - 1 = 69) (h2 : n < 76) (h3 : n > 29) : (2 * n - 1) / 3 = 23 :=
by
  sorry

end school_count_l202_202158


namespace sin_2theta_l202_202083

variable (θ : ℝ)

theorem sin_2theta (h : 2^(-3/2 + 3 * Real.sin θ) + 2 = 2^(3/4 + Real.sin θ)) : 
  Real.sin (2 * θ) = (3 * Real.sqrt 7) / 8 :=
sorry

end sin_2theta_l202_202083


namespace find_a1_l202_202160

-- Definitions of the conditions
def Sn (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of the sequence
def a₁ : ℤ := sorry          -- First term of the sequence

axiom S_2016_eq_2016 : Sn 2016 = 2016
axiom diff_seq_eq_2000 : (Sn 2016 / 2016) - (Sn 16 / 16) = 2000

-- Proof statement
theorem find_a1 : a₁ = -2014 :=
by
  -- The proof would go here
  sorry

end find_a1_l202_202160


namespace balloon_highest_elevation_l202_202856

theorem balloon_highest_elevation
  (time_rise1 time_rise2 time_descent : ℕ)
  (rate_rise rate_descent : ℕ)
  (t1 : time_rise1 = 15)
  (t2 : time_rise2 = 15)
  (t3 : time_descent = 10)
  (rr : rate_rise = 50)
  (rd : rate_descent = 10)
  : (time_rise1 * rate_rise - time_descent * rate_descent + time_rise2 * rate_rise) = 1400 := 
by
  sorry

end balloon_highest_elevation_l202_202856


namespace delta_one_two_three_eq_one_l202_202516

def delta (x y : ℝ) : ℝ := (x + y) / (x * y)

theorem delta_one_two_three_eq_one : delta (delta 1 2) 3 = 1 :=
by
  sorry

end delta_one_two_three_eq_one_l202_202516


namespace diagonals_in_nine_sided_polygon_l202_202799

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202799


namespace sum_edges_divisible_l202_202845

-- Define the conditions as described in the problem
structure ConvexPolyhedron (D : Type) :=
  (vertices : set D)
  (edges : set (D × D))
  (even_edges_per_vertex : ∀ v ∈ vertices, (edges.filter (λ e, e.fst = v ∨ e.snd = v)).card % 2 = 0)

def chosen_face {D : Type} (P : ConvexPolyhedron D) : Type := {F : set D // ∃ f ∈ P.faces, f = F}

def edge_assignment {D : Type} (P : ConvexPolyhedron D) : (D × D) → ℕ :=
  λ e, if e ∈ P.edges then classical.some (exists_pos_nat e) else 0

def valid_assignment {D : Type} (P : ConvexPolyhedron D) (F : chosen_face P) : Prop :=
  ∀ (G ∈ P.faces), G ≠ F → (∑ e ∈ G.edges, edge_assignment P e) % 2024 = 0

-- The main theorem to prove
theorem sum_edges_divisible {D : Type} (P : ConvexPolyhedron D) (F : chosen_face P) :
  valid_assignment P F →
  (∑ e in (F : set D).edges, edge_assignment P e) % 2024 = 0 := 
sorry

end sum_edges_divisible_l202_202845


namespace find_f_inv_127_l202_202139

def f : ℕ → ℕ :=
  sorry  -- The actual function definition will be determined by the conditions.

theorem find_f_inv_127 :
  f(4) = 3 → (∀ x, f(2 * x) = 2 * f(x) + 1) → f 128 = 127 :=
by
  intros h1 h2
  -- The proof of this theorem will need to be filled in.
  sorry

end find_f_inv_127_l202_202139


namespace intersection_polar_coords_l202_202844

noncomputable def polar_coord_intersection (rho theta : ℝ) : Prop :=
  (rho * (Real.sqrt 3 * Real.cos theta - Real.sin theta) = 2) ∧ (rho = 4 * Real.sin theta)

theorem intersection_polar_coords :
  ∃ (rho theta : ℝ), polar_coord_intersection rho theta ∧ rho = 2 ∧ theta = (Real.pi / 6) := 
sorry

end intersection_polar_coords_l202_202844


namespace diagonals_in_nine_sided_polygon_l202_202775

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202775


namespace stating_sum_first_10_common_elements_l202_202451

/-- 
  Theorem stating that the sum of the first 10 elements that appear 
  in both the given arithmetic progression and the geometric progression 
  equals 13981000.
-/
theorem sum_first_10_common_elements :
  let AP := λ n : ℕ => 4 + 3 * n in
  let GP := λ k : ℕ => 20 * 2^k in
  let common_elements := (range 10).map (λ i => GP (2 * i + 1)) in
  (∑ i in (finset.range 10), common_elements[i]) = 13981000 :=
by
  let AP := λ n : ℕ => 4 + 3 * n
  let GP := λ k : ℕ => 20 * 2^k
  let common_elements := (finset.range 10).map (λ i => GP (2 * i + 1))
  have S : (∑ i in (finset.range 10), common_elements[i]) = 40 * (4^10 - 1) / 3,
  {
    sorry,
  }
  have : 40 * 349525 = 13981000 := by norm_num,
  exact this ▸ S

end stating_sum_first_10_common_elements_l202_202451


namespace space_shuttle_speed_l202_202378

theorem space_shuttle_speed :
  ∀ (speed_kph : ℕ) (minutes_per_hour seconds_per_minute : ℕ),
    speed_kph = 32400 →
    minutes_per_hour = 60 →
    seconds_per_minute = 60 →
    (speed_kph / (minutes_per_hour * seconds_per_minute)) = 9 :=
by
  intros speed_kph minutes_per_hour seconds_per_minute
  intro h_speed
  intro h_minutes
  intro h_seconds
  sorry

end space_shuttle_speed_l202_202378


namespace point_in_even_number_of_triangles_l202_202069

variable {m : ℕ}
variable (A : Fin (2*m) → ℝ × ℝ)
variable (P : ℝ × ℝ)

-- Assuming the conditions
def is_convex_polygon (A : Fin (2*m) → ℝ × ℝ) : Prop := sorry
def inside_polygon (P : ℝ × ℝ) (A : Fin (2*m) → ℝ × ℝ) : Prop := sorry
def not_on_diagonals (P : ℝ × ℝ) (A : Fin (2*m) → ℝ × ℝ) : Prop := sorry
def triangle_contains_point (P : ℝ × ℝ) (A i j k : ℝ × ℝ) : Prop := sorry

theorem point_in_even_number_of_triangles :
  is_convex_polygon A →
  inside_polygon P A →
  not_on_diagonals P A →
  (Finset.filter (λ ⟨i, j, k⟩, triangle_contains_point P (A i) (A j) (A k))
                 (Finset.unordered_triple_combinations (Finset.univ : Finset (Fin (2*m)))).card % 2 = 0 :=
by sorry

end point_in_even_number_of_triangles_l202_202069


namespace problem_statement_l202_202889

theorem problem_statement (a b n : ℤ) (h1 : n > 0) (h2 : a ≠ b) (h3 : n ∣ (a^n - b^n)) : n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end problem_statement_l202_202889


namespace scheduling_arrangements_l202_202978

theorem scheduling_arrangements (A B C : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] :
  (∀ (days : Finset ℕ), (A ∈ days) ∧ (B ∈ days) ∧ (C ∈ days) ∧ (days.card = 3) ∧
  (∀ d1 d2 d3 : ℕ, d1 ∈ days → d2 ∈ days → d3 ∈ days → d1 < d2 → d2 < d3 → 
  (A = d1) ∧ (B ≠ d1) ∧ (C ≠ d2) )) → 
  (∃ arrangements : ℕ, arrangements = 20) :=
begin
  sorry
end

end scheduling_arrangements_l202_202978


namespace sin_alpha_value_l202_202816

theorem sin_alpha_value (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h_cos : cos (α + π / 3) = -4 / 5) :
  sin α = (3 + 4 * sqrt 3) / 10 := by
  sorry

end sin_alpha_value_l202_202816


namespace value_of_a_2008_l202_202892

open Nat

def is_rel_prime_75 (n : ℕ) : Prop :=
  gcd n 75 = 1

def rel_prime_75_seq : ℕ → ℕ
| n := (75 * (n / 40) + (Finset.filter is_rel_prime_75 (Finset.range (75 + 1))).toList.get! (n % 40))

theorem value_of_a_2008 : rel_prime_75_seq 2008 = 3764 :=
sorry

end value_of_a_2008_l202_202892


namespace diagonals_in_nonagon_l202_202724

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202724


namespace diagonals_in_nine_sided_polygon_l202_202771

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202771


namespace sequence_a_n_a5_eq_21_l202_202945

theorem sequence_a_n_a5_eq_21 
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n) :
  a 5 = 21 :=
by
  sorry

end sequence_a_n_a5_eq_21_l202_202945


namespace diagonals_in_regular_nine_sided_polygon_l202_202577

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202577


namespace all_visitors_can_buy_ticket_l202_202153

-- Define the coin types
inductive Coin
  | Three
  | Five

-- Define a function to calculate the total money from a list of coins
def totalMoney (coins : List Coin) : Int :=
  coins.foldr (fun c acc => acc + (match c with | Coin.Three => 3 | Coin.Five => 5)) 0

-- Define the initial state: each person has 22 tugriks in some combination of 3 and 5 tugrik coins
def initial_money := 22
def ticket_cost := 4

-- Each visitor and the cashier has 22 tugriks initially
axiom visitor_money_all_22 (n : Nat) : n ≤ 200 → totalMoney (List.replicate 2 Coin.Five ++ List.replicate 4 Coin.Three) = initial_money

-- We want to prove that all visitors can buy a ticket
theorem all_visitors_can_buy_ticket :
  ∀ n, n ≤ 200 → ∃ coins: List Coin, totalMoney coins = initial_money ∧ totalMoney coins ≥ ticket_cost := by
    sorry -- Proof goes here

end all_visitors_can_buy_ticket_l202_202153


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202613

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202613


namespace sequence_57th_pair_l202_202074

theorem sequence_57th_pair : 
  let sequence : ℕ → (ℕ × ℕ) := 
    λ n, let k := (nat.find (λ k, (k * (k + 1)) / 2 ≥ n)) in
    let offset := n - (k - 1) * k / 2 in
    (offset, k - (offset - 1)) in
  sequence 57 = (2, 10) :=
by
  sorry

end sequence_57th_pair_l202_202074


namespace diagonals_in_nonagon_l202_202730

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202730


namespace diagonals_in_nine_sided_polygon_l202_202735

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202735


namespace log12_div_log15_eq_2m_n_div_1_m_n_l202_202137

variable (m n : Real)

theorem log12_div_log15_eq_2m_n_div_1_m_n 
  (h1 : Real.log 2 = m) 
  (h2 : Real.log 3 = n) : 
  Real.log 12 / Real.log 15 = (2 * m + n) / (1 - m + n) :=
by sorry

end log12_div_log15_eq_2m_n_div_1_m_n_l202_202137


namespace diagonals_in_regular_nine_sided_polygon_l202_202537

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202537


namespace maximum_a_l202_202938

open Nat

theorem maximum_a (n : ℕ) (h : n > 0) :
  (∑ k in Finset.range (2 * n + 2) \ Finset.range (n + 1), (1 : ℝ) / (n + 1 + k)) > 25 / 24 :=
sorry

end maximum_a_l202_202938


namespace max_v_l202_202949

/-- conditions --/
def F (A S ρ v₀ v : ℝ) : ℝ := (A * S * ρ * (v₀ - v) ^ 2) / 2

def N (A S ρ v₀ v : ℝ) : ℝ := F A S ρ v₀ v * v

/-- variables and constants --/
variables (A ρ : ℝ)

/-- given values --/
def S : ℝ := 4
def v₀ : ℝ := 4.8

/-- theorem statement --/
theorem max_v : ∃ v, v = 1.6 ∧ ∀ v', N A (S := 4) ρ (v₀ := 4.8) v' ≤ N A S ρ v :=
sorry

end max_v_l202_202949


namespace diagonals_in_regular_nine_sided_polygon_l202_202578

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202578


namespace problem_statement_l202_202337

def set_definite (A : Type) (P : A → Prop) : Prop :=
  ∃ (S : Set A), ∀ x, S x ↔ P x

def undergraduates_in_2013 (x : Type) : Prop :=
  -- Placeholder for actual predicate defining undergraduates in 2013
  sorry

def cities_with_high_wheat_production_in_China_2013 (x : Type) : Prop :=
  -- Placeholder for actual predicate
  sorry

def famous_mathematicians (x : Type) : Prop :=
  -- Placeholder for actual predicate
  sorry

def numbers_infinitely_close_to_pi (x : ℝ) : Prop :=
  -- Placeholder for actual predicate
  sorry

theorem problem_statement : 
  set_definite ? (undergraduates_in_2013) ∧
  ¬ set_definite ? (cities_with_high_wheat_production_in_China_2013) ∧
  ¬ set_definite ? (famous_mathematicians) ∧
  ¬ set_definite ? (numbers_infinitely_close_to_pi)
:= 
  sorry

end problem_statement_l202_202337


namespace probability_of_selecting_at_least_one_girl_l202_202464

theorem probability_of_selecting_at_least_one_girl :
  let total_people : ℕ := 6
  let total_boys : ℕ := 4
  let total_girls : ℕ := 2
  let selections : ℕ := 3
  (finset.card (finset.filter 
      (λ s : finset (fin total_people), 
        selections = finset.card s ∧ 
        (0 < finset.card (finset.filter (λ x, x.val < total_boys) s))) 
      (finset.powerset (finset.univ : finset (fin total_people))))).val = 
     finset.card (finset.filter 
      (λ s : finset (fin total_people), 
        selections = finset.card s ∧ 
        (0 < finset.card (finset.filter (λ x, x.val >= total_boys) s))) 
      (finset.powerset (finset.univ : finset (fin total_people)))).val then do
sorry

end probability_of_selecting_at_least_one_girl_l202_202464


namespace regular_nine_sided_polygon_diagonals_l202_202668

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202668


namespace counting_number_leaves_remainder_of_6_l202_202126

theorem counting_number_leaves_remainder_of_6:
  ∃! d : ℕ, d > 6 ∧ d ∣ (53 - 6) ∧ 53 % d = 6 :=
begin
  sorry
end

end counting_number_leaves_remainder_of_6_l202_202126


namespace hats_cost_l202_202868

variables {week_days : ℕ} {weeks : ℕ} {cost_per_hat : ℕ}

-- Conditions
def num_hats (week_days : ℕ) (weeks : ℕ) : ℕ := week_days * weeks
def total_cost (num_hats : ℕ) (cost_per_hat : ℕ) : ℕ := num_hats * cost_per_hat

-- Proof problem
theorem hats_cost (h1 : week_days = 7) (h2 : weeks = 2) (h3 : cost_per_hat = 50) : 
  total_cost (num_hats week_days weeks) cost_per_hat = 700 :=
by 
  sorry

end hats_cost_l202_202868


namespace intersection_P_Q_l202_202115

-- Define sets P and Q based on given conditions.
def P_set (x : ℝ) : Set ℝ := { y | y = x^2 + 3 * x + 1 }
def Q_set (x : ℝ) : Set ℝ := { y | y = - x^2 - 3 * x + 1 }

-- State the theorem that P ∩ Q is the interval [-5/4, 13/4]
theorem intersection_P_Q :
  ∀ x : ℝ, ∃ y : ℝ, y ∈ P_set x ∧ y ∈ Q_set x ↔ y ∈ set.Icc (-5/4) (13/4) :=
sorry

end intersection_P_Q_l202_202115


namespace initial_chicken_wings_l202_202358

theorem initial_chicken_wings 
  (num_friends : ℕ)
  (wings_per_friend : ℕ)
  (extra_wings : ℕ)
  (total_wings_received : ℕ := num_friends * wings_per_friend)
  (initial_wings : ℕ := total_wings_received - extra_wings) :
  num_friends = 4 → 
  wings_per_friend = 4 → 
  extra_wings = 7 → 
  initial_wings = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end initial_chicken_wings_l202_202358


namespace placing_balls_ABCDe_is_270_l202_202396

theorem placing_balls_ABCDe_is_270 :
  ∃ (boxes : Finset ℕ) (A B C D E : ℕ),
    boxes = {1, 2, 3, 4, 5, 6, 7} ∧
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ A ∧
    A ∈ boxes ∧ B ∈ boxes ∧ C ∈ boxes ∧ D ∈ boxes ∧ E ∈ boxes ∧
    ((A + 1 = B) ∨ (A - 1 = B)) ∧
    ((C + 1 = D) ∨ (C - 1 = D)) ∧
  nat.combinations 7 4 = 10 ∧ 
  (7 - 4 = 3) ∧
  ((3 * 3 * 3) * 10 = 270) :=
sorry

end placing_balls_ABCDe_is_270_l202_202396


namespace three_n_plus_two_not_perfect_square_l202_202236

theorem three_n_plus_two_not_perfect_square (n : ℕ) : ¬ ∃ (a : ℕ), 3 * n + 2 = a * a :=
by
  sorry

end three_n_plus_two_not_perfect_square_l202_202236


namespace gcd_digits_le_three_l202_202150

theorem gcd_digits_le_three (a b : ℕ) (h1 : a < 10^5) (h2 : b < 10^5) (h3 : 10^7 ≤ Mathlib.lcm a b) (h4 : Mathlib.lcm a b < 10^8) : 
  ∃ (g : ℕ), Mathlib.gcd a b = g ∧ g < 10^3 := sorry

end gcd_digits_le_three_l202_202150


namespace brad_start_time_after_maxwell_l202_202222

-- Assuming time is measured in hours, distance in kilometers, and speed in km/h
def meet_time (d : ℕ) (v_m : ℕ) (v_b : ℕ) (t_m : ℕ) : ℕ :=
  let d_m := t_m * v_m
  let t_b := t_m - 1
  let d_b := t_b * v_b
  d_m + d_b

theorem brad_start_time_after_maxwell (d : ℕ) (v_m : ℕ) (v_b : ℕ) (t_m : ℕ) :
  d = 54 → v_m = 4 → v_b = 6 → t_m = 6 → 
  meet_time d v_m v_b t_m = 54 :=
by
  intros hd hv_m hv_b ht_m
  have : meet_time d v_m v_b t_m = t_m * v_m + (t_m - 1) * v_b := rfl
  rw [hd, hv_m, hv_b, ht_m] at this
  sorry

end brad_start_time_after_maxwell_l202_202222


namespace regular_nine_sided_polygon_diagonals_l202_202682

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202682


namespace bus_travel_time_kimovsk_moscow_l202_202285

noncomputable def travel_time_kimovsk_moscow (d1 d2 d3: ℝ) (max_speed: ℝ) (t_kt: ℝ) (t_nm: ℝ) : Prop :=
  35 ≤ d1 ∧ d1 ≤ 35 ∧
  60 ≤ d2 ∧ d2 ≤ 60 ∧
  200 ≤ d3 ∧ d3 ≤ 200 ∧
  max_speed <= 60 ∧
  2 ≤ t_kt ∧ t_kt ≤ 2 ∧
  5 ≤ t_nm ∧ t_nm ≤ 5 ∧
  (5 + 7/12 : ℝ) ≤ t_kt + t_nm ∧ t_kt + t_nm ≤ 6

theorem bus_travel_time_kimovsk_moscow
  (d1 d2 d3 : ℝ) (max_speed : ℝ) (t_kt : ℝ) (t_nm : ℝ) :
  travel_time_kimovsk_moscow d1 d2 d3 max_speed t_kt t_nm := 
by
  sorry

end bus_travel_time_kimovsk_moscow_l202_202285


namespace nine_sided_polygon_diagonals_l202_202592

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202592


namespace right_triangle_median_to_hypotenuse_l202_202839

theorem right_triangle_median_to_hypotenuse 
    {DEF : Type} [MetricSpace DEF] 
    (D E F M : DEF) 
    (h_triangle : dist D E = 15 ∧ dist D F = 20 ∧ dist E F = 25) 
    (h_midpoint : dist D M = dist E M ∧ dist D E = 2 * dist D M ∧ dist E F * dist E F = dist E D * dist E D + dist D F * dist D F) :
    dist F M = 12.5 :=
by sorry

end right_triangle_median_to_hypotenuse_l202_202839


namespace distance_A_to_line_l202_202268

-- Define the point A
def point_A : ℝ × ℝ := (2, 1)

-- Define the line equation coefficients
def a : ℝ := 1
def b : ℝ := -1
def c : ℝ := 1

-- Distance from a point to a line in standard form
def distance_from_point_to_line (A : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  (abs (a * A.1 + b * A.2 + c)) / (sqrt (a^2 + b^2))

-- Prove that the distance from point A to the line x - y + 1 = 0 is sqrt(2)
theorem distance_A_to_line : distance_from_point_to_line point_A a b c = sqrt 2 := by
  sorry

end distance_A_to_line_l202_202268


namespace find_a_l202_202432

theorem find_a (a : ℚ) (h : ∃ r s : ℚ, (r*x + s)^2 = ax^2 + 18*x + 16) : a = 81 / 16 := 
by sorry 

end find_a_l202_202432


namespace choose_three_numbers_sum_at_least_five_l202_202198

theorem choose_three_numbers_sum_at_least_five 
  (x : Fin 9 → ℝ)
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_sum_squares : ∑ i, (x i)^2 ≥ 25) :
  ∃ i j k : Fin 9, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ x i + x j + x k ≥ 5 :=
by { sorry }

end choose_three_numbers_sum_at_least_five_l202_202198


namespace arccos_lt_arctan_in_interval_l202_202031

noncomputable def arccos_lt_arctan : Prop :=
  ∃ a : ℝ, 0 < a ∧ a < 1 ∧ ∀ x : ℝ, (a < x ∧ x ≤ 1) → arccos x < arctan x

-- Here we write our theorem statement
theorem arccos_lt_arctan_in_interval (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (x : ℝ) (hx : a < x ∧ x ≤ 1) : 
  arccos x < arctan x := 
sorry

end arccos_lt_arctan_in_interval_l202_202031


namespace part_I_part_II_l202_202116

-- Definitions for vectors a and b
def vector_a : ℝ × ℝ := (1, -2)

def mag_b : ℝ := 2 * Real.sqrt 5

def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

noncomputable def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def angle_between (u v : ℝ × ℝ) : ℝ :=
  Real.acos (dot_product u v / (magnitude u * magnitude v))

-- Proof statements

-- Part I
theorem part_I (b : ℝ × ℝ) (hb : magnitude b = mag_b) (hb_parallel : is_parallel vector_a b) :
  b = (-2, 4) ∨ b = (2, -4) := 
sorry

-- Part II
theorem part_II (b : ℝ × ℝ) (hb : magnitude b = mag_b) 
(h_dot : dot_product (2 * vector_a.1 - 3 * b.1, 2 * vector_a.2 - 3 * b.2)
                     (2 * vector_a.1 + b.1, 2 * vector_a.2 + b.2) = -20) :
  angle_between vector_a b = 2 * Real.pi / 3 := 
sorry

end part_I_part_II_l202_202116


namespace negation_of_existence_statement_l202_202275

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x^2 - 8 * x + 18 < 0) ↔ (∀ x : ℝ, x^2 - 8 * x + 18 ≥ 0) :=
by
  sorry

end negation_of_existence_statement_l202_202275


namespace nine_sided_polygon_diagonals_l202_202617

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202617


namespace area_trapezoid_constant_midpoint_locus_parallel_l202_202971

variables {A B C M N Q P D : Type}

-- Given conditions
variables [IsoscelesTriangle ABC AB AC]
variables (MN : Segment BC) [FixedLengthSegment MN]
variables (Q : Point) (P : Point) (Perpendicular QM BC) (Perpendicular PN BC)
variables [Intersects QM AB Q] [Intersects PN AC P]

-- Theorems to prove
theorem area_trapezoid_constant :
  area_trapezoid M N P Q = constant :=
sorry

theorem midpoint_locus_parallel :
  locus_of_midpoint D P Q = parallel_to BC :=
sorry

end area_trapezoid_constant_midpoint_locus_parallel_l202_202971


namespace find_acute_angle_as_pi_over_4_l202_202099
open Real

-- Definitions from the problem's conditions
variables (x : ℝ)
def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2
def trig_eq (x : ℝ) : Prop := (sin x) ^ 3 + (cos x) ^ 3 = sqrt 2 / 2

-- The math proof problem statement
theorem find_acute_angle_as_pi_over_4 (h_acute : is_acute x) (h_trig_eq : trig_eq x) : x = π / 4 := 
sorry

end find_acute_angle_as_pi_over_4_l202_202099


namespace number_of_real_solutions_l202_202444

-- Define the function f.
noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 100).sum (λ n, 2^(n+1) / (x - (n + 1)))

-- The theorem statement that the number of real solutions to f(x) = x is 101.
theorem number_of_real_solutions : (setOf (λ x, f x = x)).count = 101 := sorry

end number_of_real_solutions_l202_202444


namespace min_distance_PQ_l202_202279

noncomputable def curve_C1 (α : ℝ) : ℝ × ℝ :=
  (√2 * Real.cos α, 1 + √2 * Real.sin α)

noncomputable def curve_C2 := {p θ : ℝ | √2 * p * Real.sin (θ + Real.pi / 4) = 5}

theorem min_distance_PQ :
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ),
    (P ∈ {p | ∃ α, p = curve_C1 α}) ∧
    (Q ∈ {q | ∃ (p θ : ℝ), q = (p * Real.cos θ, p * Real.sin θ) ∧ curve_C2 (p, θ)}) ∧
    ∀ (P Q : ℝ × ℝ), (P ∈ {p | ∃ α, p = curve_C1 α}) →
                      (Q ∈ {q | ∃ (p θ : ℝ), q = (p * Real.cos θ, p * Real.sin θ) ∧ curve_C2 (p, θ)}) →
                      dist P Q ≥ √2 ∧
                      (∀ (P' Q' : ℝ × ℝ), (P'.fst - Q'.fst)^2 + (P'.snd - Q'.snd)^2 = 2) →
                      dist P Q = √2 :=
sorry

end min_distance_PQ_l202_202279


namespace diagonals_in_nine_sided_polygon_l202_202801

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202801


namespace max_sum_x_y_l202_202995

theorem max_sum_x_y 
  (x y : ℝ)
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
sorry

end max_sum_x_y_l202_202995


namespace diagonals_in_nine_sided_polygon_l202_202787

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202787


namespace colored_paper_area_l202_202913

theorem colored_paper_area :
  ∀ (side_length : ℝ), (side_length = 30) → (side_length * side_length = 900) :=
by
  assume side_length h,
  rw h,
  norm_num,
  done,
  sorry

end colored_paper_area_l202_202913


namespace orthogonality_implies_x_value_l202_202062

theorem orthogonality_implies_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, -1)
  a.1 * b.1 + a.2 * b.2 = 0 → x = 1 :=
sorry

end orthogonality_implies_x_value_l202_202062


namespace diagonals_in_nine_sided_polygon_l202_202772

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202772


namespace min_frac_a_n_l202_202111

noncomputable def a : ℕ → ℕ
| 1     := 33
| (n+1) := a n + 2 * n

theorem min_frac_a_n {n : ℕ} (h₁ : ∀ n, a (n + 1) = a n + 2 * n) (h₂ : a 1 = 33) :
  ∃ n, (n ≠ 0) ∧ (n + 33 / n - 1 = 21 / 2) :=
sorry

end min_frac_a_n_l202_202111


namespace diagonals_in_nine_sided_polygon_l202_202779

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202779


namespace product_of_roots_l202_202445

theorem product_of_roots:
  let a : ℕ := 24,
      b : ℕ := 36,
      c : ℤ := -648,
      prod_roots : ℤ := c / (a : ℤ)
  in prod_roots = -27 :=
by 
  let a : ℤ := 24
  let c : ℤ := -648
  have product_of_roots : ℤ := c / a
  have h : product_of_roots = -27 := by norm_num
  exact h

end product_of_roots_l202_202445


namespace regular_nonagon_diagonals_correct_l202_202703

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202703


namespace limit_of_f_at_2_l202_202407

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (sin (3 * π * x) / sin (π * x)) ^ (sin (x - 2)) ^ 2

theorem limit_of_f_at_2 : filter.tendsto (λ x, f x) (𝓝 2) (𝓝 1) :=
sorry

end limit_of_f_at_2_l202_202407


namespace sqrt_expression1_sqrt_expression2_sqrt_general_gt_sqrt_general_lt_sqrt_series_sum_l202_202056

-- Part (1): Specific cases
theorem sqrt_expression1 : sqrt ((10 - 6)^2) = 10 - 6 := by
  sorry

theorem sqrt_expression2 : sqrt ((7 - 9)^2) = 9 - 7 := by
  sorry

-- Part (2): General cases
theorem sqrt_general_gt (a b : ℝ) (h : a > b) : sqrt ((a - b)^2) = a - b := by
  sorry

theorem sqrt_general_lt (a b : ℝ) (h : a < b) : sqrt ((a - b)^2) = b - a := by
  sorry

-- Part (3): Summation of the series
theorem sqrt_series_sum : 
  (finset.range 2021).sum (λ i, sqrt ((1 / (i + 3).toℝ - 1 / (i + 2).toℝ)^2)) = (2021 / 4046) := by
  sorry

end sqrt_expression1_sqrt_expression2_sqrt_general_gt_sqrt_general_lt_sqrt_series_sum_l202_202056


namespace exists_four_non_intersecting_spheres_l202_202424

-- Define the point A and the concept of a ray starting from A
variables (A : point)

-- Define the concept of a sphere in space
structure sphere (center : point) (radius : ℝ) :=
(radius_pos : radius > 0)

-- Define non-intersecting spheres
def non_intersecting (s1 s2 : sphere) : Prop :=
  -- Assuming a basic non-intersection condition for the spheres
  (distance s1.center s2.center > s1.radius + s2.radius)

-- Define the main theorem
theorem exists_four_non_intersecting_spheres (A : point) :
  ∃ (s1 s2 s3 s4 : sphere), 
    (∀ (s i : sphere), s ≠ i → non_intersecting s i) ∧
    (∀ (s : sphere), s.center ≠ A) ∧
    (∀ (ray : ray), ∃ (s : sphere), intersects ray s) :=
sorry

end exists_four_non_intersecting_spheres_l202_202424


namespace diagonals_in_nine_sided_polygon_l202_202793

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202793


namespace nine_sided_polygon_diagonals_l202_202594

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202594


namespace quadrant_of_z_l202_202472

noncomputable def complex_quadrant : Prop :=
  let z := (2 + 3 * complex.i) / (1 + 2 * complex.i)
  (complex.re z > 0) ∧ (complex.im z < 0)

theorem quadrant_of_z : complex_quadrant :=
  sorry

end quadrant_of_z_l202_202472


namespace f_n_eq_2n_minus_1_iff_n_is_power_of_2_l202_202477

noncomputable def f (n : ℕ) : ℕ :=
  Inf { k : ℕ | ∑ i in finset.range k, (i + 1) % n = 0 }

theorem f_n_eq_2n_minus_1_iff_n_is_power_of_2 (n : ℕ) (hn : 0 < n) :
  (f n = 2 * n - 1) ↔ ∃ m : ℕ, n = 2 ^ m :=
sorry

end f_n_eq_2n_minus_1_iff_n_is_power_of_2_l202_202477


namespace nine_sided_polygon_diagonals_l202_202651

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202651


namespace find_p_q_r_sum_l202_202085

theorem find_p_q_r_sum (p q r : ℕ) (hpq_rel_prime : Nat.gcd p q = 1) (hq_nonzero : q ≠ 0) 
  (h1 : ∃ t, (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4) 
  (h2 : ∃ t, (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r) : 
  p + q + r = 7 :=
sorry

end find_p_q_r_sum_l202_202085


namespace intersection_complement_N_l202_202521

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {4, 5, 6}
def C_U_M : Set ℕ := U \ M

theorem intersection_complement_N : (C_U_M ∩ N) = {4, 6} :=
by
  sorry

end intersection_complement_N_l202_202521


namespace y_intercept_of_line_l202_202317

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 5 * y = 10) (hx : x = 0) : y = -2 :=
by {
  rw [hx, zero_mul, sub_zero] at h,
  linarith,
}

end y_intercept_of_line_l202_202317


namespace solution_set_equivalence_l202_202462

theorem solution_set_equivalence (a : ℝ) : 
    (-1 < a ∧ a < 1) ∧ (3 * a^2 - 2 * a - 5 < 0) → 
    (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) :=
by
    sorry

end solution_set_equivalence_l202_202462


namespace tail_length_third_generation_l202_202184

theorem tail_length_third_generation (initial_length : ℕ) (growth_rate : ℕ) :
  initial_length = 16 ∧ growth_rate = 25 → 
  let sec_len := initial_length * (100 + growth_rate) / 100 in
  let third_len := sec_len * (100 + growth_rate) / 100 in
  third_len = 25 := by
  intros h
  sorry

end tail_length_third_generation_l202_202184


namespace hats_cost_l202_202867

variables {week_days : ℕ} {weeks : ℕ} {cost_per_hat : ℕ}

-- Conditions
def num_hats (week_days : ℕ) (weeks : ℕ) : ℕ := week_days * weeks
def total_cost (num_hats : ℕ) (cost_per_hat : ℕ) : ℕ := num_hats * cost_per_hat

-- Proof problem
theorem hats_cost (h1 : week_days = 7) (h2 : weeks = 2) (h3 : cost_per_hat = 50) : 
  total_cost (num_hats week_days weeks) cost_per_hat = 700 :=
by 
  sorry

end hats_cost_l202_202867


namespace total_students_is_88_l202_202294

def orchestra_students : Nat := 20
def band_students : Nat := 2 * orchestra_students
def choir_boys : Nat := 12
def choir_girls : Nat := 16
def choir_students : Nat := choir_boys + choir_girls

def total_students : Nat := orchestra_students + band_students + choir_students

theorem total_students_is_88 : total_students = 88 := by
  sorry

end total_students_is_88_l202_202294


namespace equation1_solution_equation2_solution_equation3_solution_l202_202935

theorem equation1_solution (x : ℝ) : (x - 2)^2 = 25 → (x = 7 ∨ x = -3) := by
  sorry

theorem equation2_solution (x : ℝ) : x^2 + 4x + 3 = 0 → (x = -3 ∨ x = -1) := by
  sorry

theorem equation3_solution (x : ℝ) : 2 * x^2 + 4 * x - 1 = 0 → (x = (-2 + Real.sqrt 6) / 2 ∨ x = (-2 - Real.sqrt 6) / 2) := by
  sorry

end equation1_solution_equation2_solution_equation3_solution_l202_202935


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202607

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202607


namespace cos_F_right_triangle_l202_202166

noncomputable def right_triangle_cos : ℝ :=
  let DF := 9
  let EF := 15
  let DE := real.sqrt (DF^2 + EF^2)
  (DF / DE)

theorem cos_F_right_triangle :
  let DF := 9
  let EF := 15
  let DE := real.sqrt (DF^2 + EF^2)
  right_triangle_cos = 3 * real.sqrt 34 / 34 :=
by
  sorry

end cos_F_right_triangle_l202_202166


namespace journey_time_l202_202362

theorem journey_time :
  let total_distance := 224
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 21
  let speed_second_half := 24
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  total_time = 10 := by
  have first_half_distance_eq : first_half_distance = 112 := by norm_num
  have second_half_distance_eq : second_half_distance = 112 := by norm_num
  have time_first_half_eq : time_first_half = (112 / 21) := rfl
  have time_second_half_eq : time_second_half = (112 / 24) := rfl
  have total_time_eq : total_time = (112 / 21) + (112 / 24) := rfl
  have fraction_sum_eq : (112 / 21) + (112 / 24) = 10 := by norm_num
  show total_time = 10 from fraction_sum_eq

end journey_time_l202_202362


namespace nonneg_int_solution_coprime_l202_202921

theorem nonneg_int_solution_coprime (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : c ≥ (a - 1) * (b - 1)) :
  ∃ (x y : ℕ), c = a * x + b * y :=
sorry

end nonneg_int_solution_coprime_l202_202921


namespace count_of_Tims_coins_l202_202303

theorem count_of_Tims_coins 
  (initial_quarters : ℕ)
  (initial_nickels : ℕ)
  (initial_dimes : ℕ)
  (initial_pennies : ℕ)
  (dad_quarters : ℕ)
  (dad_nickels : ℕ)
  (dad_pennies : ℕ)
  (exchange_dimes : ℕ)
  (exchange_quarters : ℕ)
  (sister_nickels : ℕ)
  (store_quarters : ℕ)
  (store_nickels : ℕ)
  (exchange_quarters_nickels : ℕ)
  (exchange_quarters_new_nickels : ℕ) :
  initial_quarters = 7 →
  initial_nickels = 9 →
  initial_dimes = 12 →
  initial_pennies = 5 →
  dad_quarters = 2 →
  dad_nickels = 3 →
  dad_pennies = 5 →
  exchange_dimes = 10 →
  exchange_quarters = 4 →
  sister_nickels = 5 →
  store_quarters = 2 →
  store_nickels = 4 →
  exchange_quarters_nickels = 1 →
  exchange_quarters_new_nickels = 5 →
  let final_quarters := initial_quarters + dad_quarters + exchange_quarters - store_quarters - exchange_quarters_nickels,
      final_nickels := initial_nickels + dad_nickels - sister_nickels - store_nickels + exchange_quarters_new_nickels,
      final_dimes := initial_dimes - exchange_dimes,
      final_pennies := initial_pennies + dad_pennies in
  final_quarters = 10 ∧
  final_nickels = 8 ∧
  final_dimes = 2 ∧
  final_pennies = 10 :=
by 
  intros
  sorry

end count_of_Tims_coins_l202_202303


namespace nine_sided_polygon_diagonals_l202_202585

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202585


namespace diagonals_in_nine_sided_polygon_l202_202739

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202739


namespace angle_E_measure_l202_202218

theorem angle_E_measure (p q : Type) [parallel p q]
  (angle_A angle_B angle_C angle_E : ℝ)
  (h1 : angle_A = angle_B / 9)
  (h2 : angle_B = 3 * angle_C - 10)
  (h3 : angle_E = angle_A)
  (h4 : angle_C = angle_E)
  : angle_E = 20 := 
sorry

end angle_E_measure_l202_202218


namespace sequence_properties_sum_Tn_l202_202894

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℤ := 2^(n - 1)
noncomputable def c_n (n : ℕ) : ℤ := (2 * n - 1) / 2^(n - 1)
noncomputable def T_n (n : ℕ) : ℤ := 6 - (2 * n + 3) / 2^(n - 1)

theorem sequence_properties : (d = 2) → (S₁₀ = 100) → 
  (∀ n : ℕ, a_n n = 2 * n - 1) ∧ (∀ n : ℕ, b_n n = 2^(n - 1)) := by
  sorry

theorem sum_Tn : (d > 1) → 
  (∀ n : ℕ, T_n n = 6 - (2 * n + 3) / 2^(n - 1)) := by
  sorry

end sequence_properties_sum_Tn_l202_202894


namespace evaluate_expression_l202_202990

theorem evaluate_expression :
  54 + 98 / 14 + 23 * 17 - 200 - 312 / 6 = 200 :=
by
  sorry

end evaluate_expression_l202_202990


namespace students_same_row_l202_202375

theorem students_same_row (rows : ℕ) (seats_per_row : ℕ) (students : ℕ)
  (h_rows : rows = 7) (h_seats_per_row : seats_per_row = 10) (h_students : students = 50) :
  ∃ (a b : ℕ), a ≠ b ∧ ∀ (morning_seat_afternoon_seat : Fin (students → rows)), 
  morning_seat_afternoon_seat a = morning_seat_afternoon_seat b :=
by
  sorry

end students_same_row_l202_202375


namespace find_bicycle_speed_l202_202227

-- Let's define the conditions first
def distance := 10  -- Distance in km
def time_diff := 1 / 3  -- Time difference in hours
def speed_of_bicycle (x : ℝ) := x
def speed_of_car (x : ℝ) := 2 * x

-- Prove the equation using the given conditions
theorem find_bicycle_speed (x : ℝ) (h : x ≠ 0) :
  (distance / speed_of_bicycle x) = (distance / speed_of_car x) + time_diff :=
by {
  sorry
}

end find_bicycle_speed_l202_202227


namespace nine_sided_polygon_diagonals_l202_202646

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202646


namespace diagonals_in_regular_nine_sided_polygon_l202_202536

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202536


namespace number_of_diagonals_l202_202547

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202547


namespace product_of_cubes_l202_202408

theorem product_of_cubes :
  (∏ x in {7, 8, 9, 10, 11}, (x^3 - 1) / (x^3 + 1)) = 931 / 946 :=
begin
  sorry -- proof is omitted as per the instructions
end

end product_of_cubes_l202_202408


namespace fibonacci_sequence_sixth_term_l202_202915

theorem fibonacci_sequence_sixth_term :
    ∃ x : ℕ, x = 8 ∧ (λ seq : List ℕ,
        seq.headI = 1 ∧ 
        seq.tail.headI = 1 ∧ 
        seq.tail.tail.headI = 2 ∧ 
        seq.tail.tail.tail.headI = 3 ∧ 
        seq.tail.tail.tail.tail.headI = 5 ∧ 
        seq.tail.tail.tail.tail.tail.headI = x ∧ 
        seq.tail.tail.tail.tail.tail.tail.headI = 13 ∧ 
        ∀ n, n > 1 → seq.get? n = some ((seq.getI (n-1)) + (seq.getI (n-2)))) [1, 1, 2, 3, 5, x, 13] :=
sorry

end fibonacci_sequence_sixth_term_l202_202915


namespace isosceles_trapezoid_perimeter_l202_202162

/-- In an isosceles trapezoid ABCD with bases AB = 10 units and CD = 18 units, 
and height from AB to CD is 4 units, the perimeter of ABCD is 28 + 8 * sqrt(2) units. -/
theorem isosceles_trapezoid_perimeter :
  ∃ (A B C D : Type) (AB CD AD BC h : ℝ), 
      AB = 10 ∧ 
      CD = 18 ∧ 
      AD = BC ∧ 
      h = 4 →
      ∀ (P : ℝ), P = AB + BC + CD + DA → 
      P = 28 + 8 * Real.sqrt 2 :=
by
  sorry

end isosceles_trapezoid_perimeter_l202_202162


namespace tangent_line_at_point_one_min_value_of_f_on_interval_l202_202510

noncomputable def f (a x : ℝ) : ℝ := a * x - 1 - Real.log x

theorem tangent_line_at_point_one (a : ℝ) (h : a = 1) :
  ∀ x : ℝ, f a x = 0 → x = 1 → 0 = x - 1 := sorry

theorem min_value_of_f_on_interval (a : ℝ) :
  ∀ x ∈ set.Icc (1/2 : ℝ) (2 : ℝ),
    (a <= 1/2 ∧ f a x = 2 * a - 1 - Real.log 2) ∨
    (1/2 < a ∧ a < 2 ∧ f a x = Real.log a) ∨
    (a >= 2 ∧ f a x = a / 2 - 1 + Real.log 2) := sorry

end tangent_line_at_point_one_min_value_of_f_on_interval_l202_202510


namespace find_parallel_and_perpendicular_lines_through_A_l202_202080

def point_A : ℝ × ℝ := (2, 2)

def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 20 = 0

def parallel_line_l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 14 = 0

def perpendicular_line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y - 2 = 0

theorem find_parallel_and_perpendicular_lines_through_A :
  (∀ x y, line_l x y → parallel_line_l1 x y) ∧
  (∀ x y, line_l x y → perpendicular_line_l2 x y) :=
by
  sorry

end find_parallel_and_perpendicular_lines_through_A_l202_202080


namespace sum_reciprocals_of_roots_l202_202053

-- Problem statement: Prove that the sum of the reciprocals of the roots of the quadratic equation x^2 - 11x + 6 = 0 is 11/6.
theorem sum_reciprocals_of_roots : 
  ∀ (p q : ℝ), p + q = 11 → p * q = 6 → (1 / p + 1 / q = 11 / 6) :=
by
  intro p q hpq hprod
  sorry

end sum_reciprocals_of_roots_l202_202053


namespace nine_sided_polygon_diagonals_l202_202645

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l202_202645


namespace problem_statement_l202_202469

variables (Line Plane : Type)
variables (m n : Line) (α β γ : Plane)

-- Definition of perpendicular and parallel relations
variable perp : Line → Plane → Prop
variable par : Line → Line → Prop

-- Problem statement in Lean 4
theorem problem_statement (hmα : perp m α) (hnα : perp n α) : par m n :=
sorry

end problem_statement_l202_202469


namespace diagonals_in_nonagon_l202_202731

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202731


namespace count_intersections_l202_202393

-- Define the lattice and shapes on the plane
def lattice_points : ℤ × ℤ → bool := λ p, true  -- The boolean value "true" simplifies the model indicating it's a lattice point

def has_square (p : ℤ × ℤ) : Prop := lattice_points p
def has_circle (p : ℤ × ℤ) : Prop := lattice_points p

-- Define the line segment from (0,0) to (2017,917)
def on_line_segment (p : ℤ × ℤ) : Prop := 
  ∃ (k : ℤ), 0 ≤ k ∧ k ≤ 917 ∧ p = (2.2 * k, k)

-- Define an intersection calculation
def intersects_square (p : ℤ × ℤ) (q : ℤ × ℤ) : Prop := 
  p ∈ [(q.1 - 1/8, q.2 - 1/8), (q.1 + 1/8, q.2 + 1/8)]

def intersects_circle (p : ℤ × ℤ) (q : ℤ × ℤ) : Prop := 
  ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 ≤ 1/8 ^ 2)

-- Problem statement
theorem count_intersections : 
  let segment := [(0, 0), (2017, 917)] in
  let points := { p : ℤ × ℤ | on_line_segment p } in
  let intersections := 
    { p : ℤ × ℤ × ℤ × ℤ | let (pt, q) := p in ((intersects_square pt q) ∨ (intersects_circle pt q)) } in
  finset.card intersections = 1838 :=
sorry

end count_intersections_l202_202393


namespace find_x_y_l202_202208

noncomputable def complex_eq (x y : ℝ) : Prop :=
  let z := (x : ℂ) + (y : ℂ) * complex.I in
  complex.abs z ^ 2 + (z + complex.conj z) * complex.I = (3 - complex.I) / (2 + complex.I)

theorem find_x_y (x y : ℝ) (h : complex_eq x y) :
  (x = -1/2 ∧ (y = (real.sqrt 3)/2 ∨ y = -(real.sqrt 3)/2)) :=
begin
  sorry
end

end find_x_y_l202_202208


namespace number_of_diagonals_l202_202552

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202552


namespace derivative_of_cos_over_x_l202_202038

open Real

noncomputable def f (x : ℝ) : ℝ := (cos x) / x

theorem derivative_of_cos_over_x (x : ℝ) (h : x ≠ 0) : 
  deriv f x = - (x * sin x + cos x) / (x^2) :=
sorry

end derivative_of_cos_over_x_l202_202038


namespace range_of_eccentricity_l202_202359

variable {a b : ℝ}
variable (e : ℝ)

open Classical

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

noncomputable def eccentricity : ℝ := 
  Real.sqrt (1 + b^2 / a^2)

theorem range_of_eccentricity
  (h₁ : a > 0) 
  (h₂ : b > 0)
  (h₃ : ∃ l, l passes through the left focus of the hyperbola ∧ ∀ AB, l intersects the hyperbola at A and B ∧ |AB| = 4 * b)
  (h₄ : there are exactly two such lines)
  : e > Real.sqrt 5 ∨ 1 < e ∧ e < (Real.sqrt 5) / 2 := 
sorry

end range_of_eccentricity_l202_202359


namespace num_divisors_47_gt_6_l202_202124

theorem num_divisors_47_gt_6 : (finset.filter (λ d, d > 6) (finset.divisors 47)).card = 1 :=
by 
  sorry

end num_divisors_47_gt_6_l202_202124


namespace diagonals_in_nine_sided_polygon_l202_202770

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202770


namespace koala_fiber_eaten_l202_202195

-- Definitions based on conditions
def absorbs_percentage : ℝ := 0.40
def fiber_absorbed : ℝ := 12

-- The theorem statement to prove the total amount of fiber eaten
theorem koala_fiber_eaten : 
  (fiber_absorbed / absorbs_percentage) = 30 :=
by 
  sorry

end koala_fiber_eaten_l202_202195


namespace nine_sided_polygon_diagonals_l202_202621

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202621


namespace regular_nine_sided_polygon_diagonals_l202_202681

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202681


namespace line_angle_of_inclination_l202_202100

def angle_of_inclination (α : ℝ) : Prop :=
  tan (α * real.pi / 180) = real.sqrt 3 ∧ 0 ≤ α ∧ α < 180

theorem line_angle_of_inclination : angle_of_inclination 60 :=
by
  unfold angle_of_inclination
  split
  sorry
  split
  sorry
  sorry

end line_angle_of_inclination_l202_202100


namespace cube_root_simplification_l202_202247

theorem cube_root_simplification : 
  (∛54880000000 = 2800 * 2^(2/3) * ∛5) :=
begin
  sorry
end

end cube_root_simplification_l202_202247


namespace equation_of_l_l202_202818

-- Defining the equations of the circles
def circle_O (x y : ℝ) := x^2 + y^2 = 4
def circle_C (x y : ℝ) := x^2 + y^2 + 4 * x - 4 * y + 4 = 0

-- Assuming the line l makes circles O and C symmetric
def symmetric (l : ℝ → ℝ → Prop) := ∀ (x y : ℝ), l x y → 
  (∃ (x' y' : ℝ), circle_O x y ∧ circle_C x' y' ∧ (x + x') / 2 = x' ∧ (y + y') / 2 = y')

-- Stating the theorem to be proven
theorem equation_of_l :
  ∀ l : ℝ → ℝ → Prop, symmetric l → (∀ x y : ℝ, l x y ↔ x - y + 2 = 0) :=
by
  sorry

end equation_of_l_l202_202818


namespace ways_to_choose_marbles_l202_202134

theorem ways_to_choose_marbles :
  let my_bag := {n : ℕ | n ∈ (Finset.range 6).image (+1)}
  let mathew_bag := {n : ℕ | n ∈ (Finset.range 15).image (+1)}
  my_bag.card = 6 ∧ mathew_bag.card = 15 →
  let valid_triples := {t : (ℕ × ℕ × ℕ) | t.1 ∈ my_bag ∧ t.2 ∈ my_bag ∧ t.3 ∈ my_bag ∧ t.1 + t.2 + t.3 ∈ mathew_bag}
  let unique_ways (t : (ℕ × ℕ × ℕ)) := if t.1 + t.2 + t.3 ∈ mathew_bag then t.1 + t.2 + t.3 = t.1 + t.2 + t.3 else 0
  (∑ t in valid_triples, unique_ways t) = 90 :=
by sorry

end ways_to_choose_marbles_l202_202134


namespace no_geometric_sequence_sin_cos_tan_l202_202419

theorem no_geometric_sequence_sin_cos_tan (θ : ℝ) :
  (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ -- θ is between 0 and 2π
  (θ % (Real.pi / 2) ≠ 0) →   -- θ is not an integer multiple of π/2
  ¬((∃ a b c : ℝ, (a = Real.sin θ ∨ a = Real.cos θ ∨ a = Real.tan θ) ∧
                  (b = Real.sin θ ∨ b = Real.cos θ ∨ b = Real.tan θ) ∧
                  (c = Real.sin θ ∨ c = Real.cos θ ∨ c = Real.tan θ) ∧
                  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧  -- All three are distinct
                  a * c = b ^ 2 ∧             -- a, b, c form a geometric sequence
                  (Real.tan θ = Real.sin θ / Real.cos θ))) := -- definition of tan
by
  intro h
  cases h with h₁ h₂
  sorry -- actual proof is not needed here

end no_geometric_sequence_sin_cos_tan_l202_202419


namespace nine_sided_polygon_diagonals_l202_202593

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202593


namespace ellipse_properties_l202_202078

open Real

theorem ellipse_properties (a b c : ℝ) (ha : a > b) (hb : b > 0) (he : c / a = sqrt 6 / 3)
  (tangent_condition : ∀ x y, x^2 + y^2 = a^2 → 2 * x - sqrt 2 * y + 6 ≠ 0) :
  (∃ A B : ℝ × ℝ, ∀ k : ℝ, k ≠ 0 → 
    let k_line (k : ℝ) := (λ x : ℝ, (k * (x - 2))) in
    ∃ x1 y1 x2 y2 : ℝ,
    let inter_pts := (λ x y : ℝ, x^2 / 6 + y^2 / 2 = 1 ∧ y = k * (x - 2)) in
    inter_pts x1 y1 ∧ inter_pts x2 y2 ∧ 
    let ea (m : ℝ) (A : ℝ × ℝ) := sqrt((A.1 - m)^2 + A.2^2),
    ab (A B : ℝ × ℝ) := sqrt((B.1 - A.1)^2 + (B.2 - A.2)^2) in
    (∀ m, ∃ (E : ℝ × ℝ), E = (m, 0) → 
      ea m (x1, y1) + ea m (x2, y2) = sqrt 5 ∧ ea m (x1, y1) * (ab (x1, y1) (x2, y2)) = -5 / 9) ∧ 
      E = (7 / 3, 0)) ∧ 
    (a = sqrt 6 ∧ c = 2 ∧ b = sqrt 2 ∧ x^2 / 6 + y^2 / 2 = 1) 
by 
  let eq_ellipse := (λ x y : ℝ, x^2 / 6 + y^2 / 2 = 1) in
  sorry

end ellipse_properties_l202_202078


namespace arithmetic_sequence_nth_term_l202_202271

theorem arithmetic_sequence_nth_term (a : ℤ) (a_1 a_2 a_3 : ℤ) 
  (h1 : a_1 = a - 1) (h2 : a_2 = a + 1) (h3 : a_3 = 2 * a + 3) : 
  ∃ f : ℕ → ℤ, (∀ n, f n = 2 * n - 3) :=
begin
  use λ n : ℕ, 2 * (n : ℤ) - 3,
  intro n,
  sorry
end

end arithmetic_sequence_nth_term_l202_202271


namespace sin_minus_cos_value_complex_expression_value_l202_202470

theorem sin_minus_cos_value (x : ℝ) (hx : 0 < x ∧ x < π)
  (h : sin x + cos x = sqrt 5 / 5): sin x - cos x = 3 * sqrt 5 / 5 :=
  sorry

theorem complex_expression_value (x : ℝ) (hx : 0 < x ∧ x < π)
  (h : sin x + cos x = sqrt 5 / 5):
  (sin (2 * x) + 2 * (sin x)^2) / (1 - tan x) = 4 / 15 :=
  sorry

end sin_minus_cos_value_complex_expression_value_l202_202470


namespace dihedral_angle_of_regular_tetrahedron_l202_202073

noncomputable def dihedral_angle (A B C D E F G : ℝ) : ℝ :=
  π - arcctg (sqrt 2 / 2)

theorem dihedral_angle_of_regular_tetrahedron :
  ∀ {A B C D : ℝ}, regular_tetrahedron A B C D →
  ∀ {E F G : ℝ}, midpoint A B E → midpoint B C F → midpoint C D G →
  dihedral_angle A B C D E F G = π - arcctg (sqrt 2 / 2) :=
by
  intros A B C D h_reg_tetrahedron E F G h_midpoint_AB h_midpoint_BC h_midpoint_CD
  -- Proof will go here.
  sorry

end dihedral_angle_of_regular_tetrahedron_l202_202073


namespace weekly_allowance_l202_202189

theorem weekly_allowance (A : ℝ) (h1 : A / 2 + 6 = 11) : A = 10 := 
by 
  sorry

end weekly_allowance_l202_202189


namespace ellipse_equation_l202_202482

theorem ellipse_equation (a b c : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (C_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b)
  (focus_distance : |F1.1 - F2.1| = 2)
  (triangle_perimeter : ∀ {P:ℝ×ℝ} {F1 F2: ℝ×ℝ}, (real.dist F1 P) + (real.dist P F2) + (real.dist F1 F2) = 6):
  ∃ a b c : ℝ, P = (2,1) ∧ F1 ≠ F2 ∧  ∃ line : ℝ × ℝ → ℝ, 
    let A B : ℝ × ℝ := ⟦A,B | line (A) ≠ 0 ∧ line (B) ≠ 0 ∧ line(A) = line (B) ∧
        (∃ M : ℝ × ℝ, 
        (M ≠ (0,0)) ∧ M = ((A.1 + B.1)/2, (A.2 + B.2)/2) ) in
  (∃ m : ℝ, abs(12 - m^2/13) + (m- 4)^2/4 = 52/3) → sorry

end ellipse_equation_l202_202482


namespace construct_triangle_exists_l202_202013

open Classical

noncomputable def exists_triangle (l h m : ℝ) (AB : ℝ) (ϕ : ℝ) 
  (h_pos : 0 < h) (ϕ_pos : 0 < ϕ) (AB_pos : 0 < AB)
  (sinϕ : Real.sin ϕ = h / AB) : Prop :=
  ∃ (A B C : Type) [HasDist A ℝ] [HasDist B ℝ] [HasDist C ℝ], 
    let CL := l
    let CH := h
    let BM := m
    AB > 0 ∧
    0 < h ∧
    0 < ϕ ∧
    Real.sin ϕ = h / AB ∧
    (∃ A1 B1 C1 : ℝ, CL = l ∧ CH = h ∧ BM = m) ∧
    (∃ A2 B2 C2 : ℝ, CL = l ∧ CH = h ∧ BM = m)

theorem construct_triangle_exists (l h m : ℝ) (AB : ℝ) (ϕ : ℝ)
  (h_pos : 0 < h) (ϕ_pos : 0 < ϕ) (AB_pos : 0 < AB)
  (sinϕ : Real.sin ϕ = h / AB) : 
  exists_triangle l h m AB ϕ h_pos ϕ_pos AB_pos sinϕ :=
sorry

end construct_triangle_exists_l202_202013


namespace diagonals_in_nine_sided_polygon_l202_202769

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l202_202769


namespace total_lobster_pounds_l202_202118

theorem total_lobster_pounds
  (combined_other_harbors : ℕ)
  (hooper_bay : ℕ)
  (H1 : combined_other_harbors = 160)
  (H2 : hooper_bay = 2 * combined_other_harbors) :
  combined_other_harbors + hooper_bay = 480 :=
by
  -- proof goes here
  sorry

end total_lobster_pounds_l202_202118


namespace find_a₃_l202_202161

variable (a₁ a₂ a₃ a₄ a₅ : ℝ)
variable (S₅ : ℝ) (a_seq : ℕ → ℝ)

-- Define the conditions for arithmetic sequence and given sum
def is_arithmetic_sequence (a_seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_seq (n+1) - a_seq n = a_seq 1 - a_seq 0

axiom sum_first_five_terms (S₅ : ℝ) (hS : S₅ = 20) : 
  S₅ = (5 * (a₁ + a₅)) / 2

-- Main theorem we need to prove
theorem find_a₃ (hS₅ : S₅ = 20) (h_seq : is_arithmetic_sequence a_seq) :
  (∃ (a₃ : ℝ), a₃ = 4) :=
sorry

end find_a₃_l202_202161


namespace area_of_triangle_PF1F2_l202_202893

theorem area_of_triangle_PF1F2 :
  ∀ (a : ℝ) (P : ℝ × ℝ), 
  a > 2 → 
  (P.1 ^ 2) / (a ^ 2) + (P.2 ^ 2) / 4 = 1 → 
  ∃ (F₁ F₂ : ℝ × ℝ), 
    let F₁ := (-(√(a ^ 2 - 4)), 0)
    let F₂ := (√(a ^ 2 - 4), 0)
    ∠P F₁ F₂ = 60° ∧ 
    (F₁P.1 - P.1) ^ 2 + (F₁P.2 - P.2) ^ 2 = (2a) ^ 2 ∧ 
    ((F₁P.1 - F₂P.1) ^ 2 + (F₁P.2 - F₂P.2) ^ 2) = 2(√(a ^ 2 - 4)) ^ 2 ∧ 
  area_of_triangle P F₁ F₂ = (4 * √3) / 3 :=
sorry

end area_of_triangle_PF1F2_l202_202893


namespace alice_sales_above_goal_l202_202386

theorem alice_sales_above_goal :
  let quota := 1000
  let nike_price := 60
  let adidas_price := 45
  let reebok_price := 35
  let nike_sold := 8
  let adidas_sold := 6
  let reebok_sold := 9
  let total_sales := nike_price * nike_sold + adidas_price * adidas_sold + reebok_price * reebok_sold
  in total_sales - quota = 65 :=
by
  let quota := 1000
  let nike_price := 60
  let adidas_price := 45
  let reebok_price := 35
  let nike_sold := 8
  let adidas_sold := 6
  let reebok_sold := 9
  let total_sales := nike_price * nike_sold + adidas_price * adidas_sold + reebok_price * reebok_sold
  show total_sales - quota = 65 from sorry

end alice_sales_above_goal_l202_202386


namespace square_same_area_as_rectangle_l202_202371

theorem square_same_area_as_rectangle (l w : ℝ) (rect_area sq_side : ℝ) :
  l = 25 → w = 9 → rect_area = l * w → sq_side^2 = rect_area → sq_side = 15 :=
by
  intros h_l h_w h_rect_area h_sq_area
  rw [h_l, h_w] at h_rect_area
  sorry

end square_same_area_as_rectangle_l202_202371


namespace diagonals_in_regular_nine_sided_polygon_l202_202545

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202545


namespace regular_nonagon_diagonals_correct_l202_202704

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202704


namespace exists_subsequence_product_is_perfect_square_l202_202206

theorem exists_subsequence_product_is_perfect_square (c : Fin 16 → Nat) (h : ∀ i, c i < 10) :
  ∃ i j, i ≤ j ∧ (∃ k, (∏ x in Finset.range (j - i + 1), c (i + x)) = k * k) := sorry

end exists_subsequence_product_is_perfect_square_l202_202206


namespace find_f_of_2x_l202_202494

theorem find_f_of_2x (f : ℝ → ℝ)
  (h1 : ∀ x, x ≠ 0 → f(x) + 2 * f(1/x) = 6/x - 3)
  (x : ℝ) (hx : x ≠ 0) :
  f(2 * x) = -1/x + 8 * x - 1 :=
sorry

end find_f_of_2x_l202_202494


namespace minimal_black_cells_l202_202197

/--
For a given positive integer \( N \),
consider an \( N \times N \) array where two corner cells
on the same longest diagonal are initially black. Each move
consists of changing the color of every cell in a chosen row
or column. Prove that the minimal number of additional cells
required to be colored black, so that the entire array can be
turned completely black after a finite number of moves, is \( 2N - 4 \).
-/
theorem minimal_black_cells (N : ℕ) (hN : N > 0) :
  ∃ k : ℕ, k = 2 * N - 4 ∧ 
  (∀ (A : array ℕ ℕ (fin N → fin N → bool)) (B : array ℕ ℕ (fin N → fin N → bool)),
    (∀ i j, (i = j ∧ (i = 0 ∨ i = N - 1)) → A.read ⟨i, sorry⟩ ⟨j, sorry⟩ = tt) →
    ∃ moves : list (ℕ ⊕ ℕ), (∀ move ∈ moves, move = 0 ∧ move < N ∨ move = 1 ∧ move < N) ∧
    ∀ i j, B.read ⟨i, sorry⟩ ⟨j, sorry⟩ = tt) :=
sorry

end minimal_black_cells_l202_202197


namespace sum_of_perfect_squares_between_100_and_700_l202_202329

theorem sum_of_perfect_squares_between_100_and_700 :
  ∑ k in finset.range 27, k^2 - ∑ k in finset.range 10, k^2 =  5933 :=
by
  sorry

end sum_of_perfect_squares_between_100_and_700_l202_202329


namespace Galerkin_solution_l202_202312

noncomputable def φ (x : ℝ) : ℝ := 3 * x

theorem Galerkin_solution :
  ∀ x : ℝ, φ x = x + ∫ t in -1..1, x * t * φ t :=
by
  intro x
  sorry

end Galerkin_solution_l202_202312


namespace paul_cookie_price_l202_202058

noncomputable def price_per_cookie_same_earnings (
  art_cookies : ℕ,
  art_price_per_cookie : ℕ,
  art_cookie_area : ℕ,
  paul_cookies : ℕ,
  equal_dough : ℕ
) : ℕ :=
(art_cookies * art_price_per_cookie) / paul_cookies

-- Definitions according to conditions
def art_cookies : ℕ := 10
def art_price_per_cookie : ℕ := 50
def art_cookie_area : ℕ := 12
def paul_cookies : ℕ := 20
def equal_dough : ℕ := 120 -- Calculated as 10 * 12

theorem paul_cookie_price : price_per_cookie_same_earnings art_cookies art_price_per_cookie art_cookie_area paul_cookies equal_dough = 25 :=
by
  -- Proof steps omitted; proof to show price_per_cookie_same_earnings evaluates to 25
  sorry

end paul_cookie_price_l202_202058


namespace hyperbola_thm_l202_202499

noncomputable def hyperbola_equation (a b c : ℝ) := 
  (∀ (x y : ℝ), (y^2 / a^2 - x^2 / b^2 = 1)) 

theorem hyperbola_thm (a b c : ℝ):
  (∃ (F : ℝ×ℝ), F = (0, 2)) ∧
  (∀ (x y : ℝ), y = real.sqrt 3 * x ∨ y = -real.sqrt 3 * x) → 
  hyperbola_equation (real.sqrt (1 / 3)) 1 2 := by 
  sorry

end hyperbola_thm_l202_202499


namespace problem_statement_l202_202436

def are_collinear (A B C : Point) : Prop := sorry -- Definition for collinearity should be expanded.
def area (A B C : Point) : ℝ := sorry -- Definition for area must be provided.

theorem problem_statement :
  ∀ n : ℕ, (n > 3) →
  (∃ (A : Fin n → Point) (r : Fin n → ℝ),
    (∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → ¬ are_collinear (A i) (A j) (A k)) ∧
    (∀ i j k : Fin n, area (A i) (A j) (A k) = r i + r j + r k)) →
  n = 4 :=
by sorry

end problem_statement_l202_202436


namespace arccos_less_arctan_l202_202032

theorem arccos_less_arctan {x : ℝ} (hx : -1 ≤ x ∧ x ≤ 1) :
  ∃ α ∈ Icc (-1 : ℝ) (1 : ℝ), α ≈ (1/2 : ℝ) ∧ (x > α → arccos x < arctan x) := by
sorry

end arccos_less_arctan_l202_202032


namespace no_zeros_for_x_gt_2_at_least_n_zeros_for_x_in_1_to_2_l202_202500

noncomputable def f : ℝ → ℝ := λ x, x * (x - 1)
noncomputable def f_n : ℕ → ℝ → ℝ
| 1     := f
| (n+1) := λ x, f (f_n n x)

theorem no_zeros_for_x_gt_2 (n : ℕ) (x : ℝ) (h : x > 2) : f_n n x ≠ 0 :=
sorry

theorem at_least_n_zeros_for_x_in_1_to_2 (n : ℕ) (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) :
  ∃ (xs : Fin n → ℝ), ∀ i, (1 ≤ xs i ∧ xs i ≤ 2) ∧ (f_n n (xs i) = 0) :=
sorry

end no_zeros_for_x_gt_2_at_least_n_zeros_for_x_in_1_to_2_l202_202500


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202603

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202603


namespace packages_per_truck_l202_202976

theorem packages_per_truck (total_packages : ℕ) (number_of_trucks : ℕ) (h1 : total_packages = 490) (h2 : number_of_trucks = 7) :
  (total_packages / number_of_trucks) = 70 := by
  sorry

end packages_per_truck_l202_202976


namespace range_m_l202_202067

def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

noncomputable def problem :=
  ∀ (m : ℝ), (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

theorem range_m (m : ℝ) : problem := 
  sorry

end range_m_l202_202067


namespace min_jumps_l202_202272

/-- Kuzya the flea needs to jump from point A to point B.
    Prove that the minimum number of 17 mm jumps required to cover a distance of 1947 cm is 1146. -/
theorem min_jumps (d_cm : ℕ) (jump_mm : ℕ) (distance_needed_mm : ℕ)
  (h_dist_cm : d_cm = 1947)
  (h_jump : jump_mm = 17)
  (h_distance_needed : distance_needed_mm = d_cm * 10) :
  ∃ n : ℕ, n * jump_mm ≥ distance_needed_mm ∧ (∀ m : ℕ, m * jump_mm ≥ distance_needed_mm → n ≤ m) ∧ n = 1146 :=
begin
  sorry
end

end min_jumps_l202_202272


namespace nine_sided_polygon_diagonals_l202_202661

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202661


namespace ending_point_divisible_by_9_l202_202292

theorem ending_point_divisible_by_9 (n : ℕ) (ending_point : ℕ) 
  (h1 : n = 11110) 
  (h2 : ∃ k : ℕ, 10 + 9 * k = ending_point) : 
  ending_point = 99999 := 
  sorry

end ending_point_divisible_by_9_l202_202292


namespace median_of_scores_l202_202167

theorem median_of_scores : ∀ (scores : List ℚ),
  scores = [90, 78, 82, 85, 90] → median scores = 85 :=
begin
  intros scores h,
  rw h,
  sorry,
end

end median_of_scores_l202_202167


namespace diagonals_in_nine_sided_polygon_l202_202734

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202734


namespace Mrs_Fredricksons_chickens_l202_202225

theorem Mrs_Fredricksons_chickens (C : ℕ) (h1 : 1/4 * C + 1/4 * (3/4 * C) = 35) : C = 80 :=
by
  sorry

end Mrs_Fredricksons_chickens_l202_202225


namespace total_coins_count_l202_202001

variable (dimes nickels quarters total_coins : ℕ)

theorem total_coins_count :
  dimes = 2 → 
  nickels = 2 → 
  quarters = 7 → 
  total_coins = dimes + nickels + quarters → 
  total_coins = 11 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  rw h4
  unfold_coes
  norm_num
  sorry

end total_coins_count_l202_202001


namespace range_of_a_l202_202820

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 0 then (x - a) ^ 2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ f 0 a) ↔ 0 ≤ a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l202_202820


namespace length_PQ_range_of_a_l202_202484

noncomputable def circle_M : circle :=
  sorry -- Define the circle passing through points A, B, and C

def points_A : point := (1, 3)
def points_B : point := (4, 2)
def points_C : point := (1, -7)

def line_l : linear_equation := (2, -1, 6) -- 2x - y + 6 = 0

def radius_N : ℝ := 10

theorem length_PQ (P Q : point) (hP : on_circle P circle_M) (hQ : on_circle Q circle_M) (y : ℝ):
  P.1 = 0 ∧ Q.1 = 0 ∧ P.2 = y + 2 * sqrt 6 ∧ Q.2 = y - 2 * sqrt 6 → (dist P Q = 4 * sqrt 6) :=
sorry

theorem range_of_a (a b : ℝ) (center_N_on_line : (2 * a - b + 6 = 0)) (intersects : circles_intersect circle_M 
(circle_centered (a, b) radius_N)):
  (-3 - sqrt(41) ≤ a ∧ a ≤ -4) ∨ (-2 ≤ a ∧ a ≤ -3 + sqrt(41)) :=
sorry

end length_PQ_range_of_a_l202_202484


namespace nine_sided_polygon_diagonals_l202_202655

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202655


namespace regular_nine_sided_polygon_diagonals_l202_202756

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202756


namespace regular_nine_sided_polygon_diagonals_l202_202765

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202765


namespace number_of_digits_3_pow_24_7_pow_36_l202_202966

noncomputable def num_digits_base_10 (n : ℕ) : ℕ :=
  ⌊log 10 n⌋ + 1

theorem number_of_digits_3_pow_24_7_pow_36 :
  num_digits_base_10 (3 ^ 24 * 7 ^ 36) = 32 :=
by
  sorry

end number_of_digits_3_pow_24_7_pow_36_l202_202966


namespace nine_sided_polygon_diagonals_l202_202596

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202596


namespace num_divisors_47_gt_6_l202_202122

theorem num_divisors_47_gt_6 : (finset.filter (λ d, d > 6) (finset.divisors 47)).card = 1 :=
by 
  sorry

end num_divisors_47_gt_6_l202_202122


namespace no_sol_n4_minus_m4_eq_42_l202_202413

theorem no_sol_n4_minus_m4_eq_42 :
  ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ n^4 - m^4 = 42 :=
by
  sorry

end no_sol_n4_minus_m4_eq_42_l202_202413


namespace nine_sided_polygon_diagonals_l202_202622

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202622


namespace find_ratio_s_l202_202827

-- We will use noncomputable theory because rational division isn't directly computable within Lean by default.
noncomputable theory

-- Definitions of the points and the ratios as conditions.
variables {A B C D F Q : Type} 
variables (CD DB AF FB CQ QF : ℝ)
variables (hCDBD : CD / DB = 2 / 3)
variables (hAFFB : AF / FB = 1 / 3)
variables (hQ_intersect : true) -- representing the fact that Q is the intersection point of AD and CF

-- The statement we wish to prove
theorem find_ratio_s : CQ / QF = 3 / 5 :=
sorry

end find_ratio_s_l202_202827


namespace diagonals_in_nine_sided_polygon_l202_202749

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202749


namespace regular_nine_sided_polygon_diagonals_l202_202670

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202670


namespace nine_sided_polygon_diagonals_l202_202595

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202595


namespace triangle_area_ABC_l202_202916

theorem triangle_area_ABC
    (X Y Z A B C : Point)
    (hx : X.x = 6) (hy : X.y = 0)
    (ix : Y.x = 8) (iy : Y.y = 4)
    (jx : Z.x = 10) (jy : Z.y = 0)
    (h_ratio : 0.5 * |X.x * (Y.y - Z.y) + Y.x * (Z.y - X.y) + Z.x * (X.y - Y.y)| = 0.1111111111111111 * 0.5 * | A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y) |) :
  0.5 * |A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)| = 72 :=
sorry

end triangle_area_ABC_l202_202916


namespace polynomial_remainder_l202_202021

theorem polynomial_remainder 
  (y: ℤ) 
  (root_cond: y^3 + y^2 + y + 1 = 0) 
  (beta_is_root: ∃ β: ℚ, β^3 + β^2 + β + 1 = 0) 
  (beta_four: ∀ β: ℚ, β^3 + β^2 + β + 1 = 0 → β^4 = 1) : 
  ∃ q r, (y^20 + y^15 + y^10 + y^5 + 1) = q * (y^3 + y^2 + y + 1) + r ∧ (r = 1) :=
by
  sorry

end polynomial_remainder_l202_202021


namespace max_v_l202_202948

/-- conditions --/
def F (A S ρ v₀ v : ℝ) : ℝ := (A * S * ρ * (v₀ - v) ^ 2) / 2

def N (A S ρ v₀ v : ℝ) : ℝ := F A S ρ v₀ v * v

/-- variables and constants --/
variables (A ρ : ℝ)

/-- given values --/
def S : ℝ := 4
def v₀ : ℝ := 4.8

/-- theorem statement --/
theorem max_v : ∃ v, v = 1.6 ∧ ∀ v', N A (S := 4) ρ (v₀ := 4.8) v' ≤ N A S ρ v :=
sorry

end max_v_l202_202948


namespace number_of_lattice_points_in_T_l202_202899

open Int

def region_T (x y : ℕ) : Prop := x > 0 ∧ y > 0 ∧ x * y ≤ 48

theorem number_of_lattice_points_in_T :
  (finset.card (finset.filter (λ ⟨x, y⟩, region_T x y) ((finset.range 49).product (finset.range 49)))) = 202 :=
sorry

end number_of_lattice_points_in_T_l202_202899


namespace range_of_a_l202_202805

theorem range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - 2 * a * x + 2 < 0) → a ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
  sorry

end range_of_a_l202_202805


namespace nine_sided_polygon_diagonals_count_l202_202691

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202691


namespace remainderMmod500_l202_202879

-- Define the sequence T
def isInSequenceT (n : ℕ) : Prop :=
  (nat.popcount n = 9)

-- Define the nth number property to find the M in the sequence T
noncomputable def nthNumberInT (n : ℕ) : ℕ :=
  classical.some (exists_nat_of_coe_nat n ∧ isInSequenceT (classical.some (exists_nat_of_coe_nat n)))

noncomputable def M : ℕ := nthNumberInT 1500

-- Define the main theorem
theorem remainderMmod500 : M % 500 = remainder := sorry

end remainderMmod500_l202_202879


namespace sequence_is_arithmetic_max_value_a_n_b_n_l202_202202

open Real

theorem sequence_is_arithmetic (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (Sn : ℕ → ℝ) 
  (h_Sn : ∀ n, Sn n = (a n ^ 2 + a n) / 2) :
    ∀ n, a n = n := sorry 

theorem max_value_a_n_b_n (a b : ℕ → ℝ)
  (h_b : ∀ n, b n = - n + 5)
  (h_a : ∀ n, a n = n) :
    ∀ n, n ≥ 2 → n ≤ 3 → 
    ∃ k, a k * b k = 25 / 4 := by 
      sorry

end sequence_is_arithmetic_max_value_a_n_b_n_l202_202202


namespace find_k_l202_202088

variable {V : Type*} [AddCommGroup V] [Module ℝ V] (e₁ e₂ : V) (k : ℝ)
variable {A B C : V}

-- Conditions: e1 and e2 are non-collinear vectors
-- Vector expressions for AB and BC
def AB : V := 2 • e₁ + k • e₂
def BC : V := e₁ - 3 • e₂

-- Collinearity condition: Points A, B, and C are collinear
def collinear (A B C : V) : Prop := ∃ λ : ℝ, AB = λ • BC

-- Proof problem: Prove that k = -6
theorem find_k (h1 : ¬ (e₁ = (0 : V))) (h2 : ¬ (e₂ = (0 : V))) (h3 : ¬ (e₁ = e₂)) (collinear A B C) : k = -6 :=
by
  unfold AB BC at collinear
  sorry

end find_k_l202_202088


namespace quadrilateral_area_l202_202982

noncomputable def AreaOfQuadrilateral (AB AC AD : ℝ) : ℝ :=
  let BC := Real.sqrt (AC^2 - AB^2)
  let CD := Real.sqrt (AC^2 - AD^2)
  let AreaABC := (1 / 2) * AB * BC
  let AreaACD := (1 / 2) * AD * CD
  AreaABC + AreaACD

theorem quadrilateral_area :
  AreaOfQuadrilateral 5 13 12 = 60 :=
by
  sorry

end quadrilateral_area_l202_202982


namespace number_of_six_digit_palindromes_l202_202018

def is_six_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9 ∧ 
    n = 100001 * a + 10010 * b + 1100 * c + 100 * d 

theorem number_of_six_digit_palindromes : 
  {n : ℕ | is_six_digit_palindrome n}.to_finset.card = 9000 :=
sorry

end number_of_six_digit_palindromes_l202_202018


namespace roots_poly_eval_l202_202204

theorem roots_poly_eval : ∀ (c d : ℝ), (c + d = 6 ∧ c * d = 8) → c^4 + c^3 * d + d^3 * c + d^4 = 432 :=
by
  intros c d h
  sorry

end roots_poly_eval_l202_202204


namespace circle_sine_intersections_l202_202390

theorem circle_sine_intersections (a b r: ℝ) : 
  ∃ circle_curve_intersections : ℝ → ℝ, 
  (∃ x, (x - a)^2 + (sin x - b)^2 = r^2 ∧ ∃ y, (x ≠ y → (x - a)^2 + (sin x - b)^2 = r^2)) 
  ∧ cardinality ( {x | (x - a)^2 + (sin x - b)^2 = r^2}) > 16
  sorry

end circle_sine_intersections_l202_202390


namespace sum_of_squares_sum_of_cubes_l202_202245

theorem sum_of_squares (n : ℕ) (h : n ≥ 1) : 
  (∑ k in Finset.range (n + 1), k^2) = (n * (n + 1) * (2 * n + 1)) / 6 := 
sorry

theorem sum_of_cubes (n : ℕ) (h : n ≥ 1) : 
  (∑ k in Finset.range (n + 1), k^3) = (∑ k in Finset.range (n + 1), k)^2 := 
sorry

end sum_of_squares_sum_of_cubes_l202_202245


namespace trigonometric_identity_l202_202934

theorem trigonometric_identity :
  sqrt (1 + sin 6) + sqrt (1 - sin 6) = -2 * cos 3 :=
sorry

end trigonometric_identity_l202_202934


namespace roots_of_quadratic_sum_of_sixth_powers_l202_202885

theorem roots_of_quadratic_sum_of_sixth_powers {u v : ℝ} 
  (h₀ : u^2 - 2*u*Real.sqrt 3 + 1 = 0)
  (h₁ : v^2 - 2*v*Real.sqrt 3 + 1 = 0)
  : u^6 + v^6 = 970 := 
by 
  sorry

end roots_of_quadratic_sum_of_sixth_powers_l202_202885


namespace diagonals_in_regular_nine_sided_polygon_l202_202579

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202579


namespace part1_solution_part2_solution_l202_202240

section part1

variables 
  (x y : ℝ)
  (hx : 8000 / x - 8000 / y = 80)
  (hy : y = 5 / 4 * x)

theorem part1_solution : x = 20 ∧ y = 25 :=
sorry

end part1

section part2

variables 
  (m n : ℝ)
  (hmn : m > 0 ∧ n > 0 ∧ m ≠ n)

theorem part2_solution : (m + n) / 2 > 2 * m * n / (m + n) :=
begin
  have h : ((m - n)^2) / (2 * (m + n)) > 0,
  { exact div_pos (mul_pos (pow_two_pos_of_ne_zero _ (hmn.2.2.symm)) zero_lt_two) (add_pos (hmn.1) (hmn.2.1)) },
  convert h,
  ring,
end

end part2

end part1_solution_part2_solution_l202_202240


namespace new_pressure_l202_202392

-- Conditions
constant p : ℝ       -- Initial pressure in kPa
constant v : ℝ       -- Initial volume in liters
constant k : ℝ       -- Constant k = p * v

-- Conditions given in the problem
constant initial_p : p = 6
constant initial_v : v = 3
constant volume_new : ℝ     -- New volume in liters
constant volume_new_val : volume_new = 6

-- Theorem statement to prove
theorem new_pressure : ∃ p' : ℝ, initialize_p ∧ initial_v ∧ volume_new_val ∧ (p * v = k) ∧ (volume_new * p' = k) ∧ (p' = 3) :=
by
  sorry

end new_pressure_l202_202392


namespace jacoby_needs_l202_202851

-- Given conditions
def total_goal : ℤ := 5000
def job_earnings_per_hour : ℤ := 20
def total_job_hours : ℤ := 10
def cookie_price_each : ℤ := 4
def total_cookies_sold : ℤ := 24
def lottery_ticket_cost : ℤ := 10
def lottery_winning : ℤ := 500
def gift_from_sister_one : ℤ := 500
def gift_from_sister_two : ℤ := 500

-- Total money Jacoby has so far
def current_total_money : ℤ := 
  job_earnings_per_hour * total_job_hours +
  cookie_price_each * total_cookies_sold +
  lottery_winning +
  gift_from_sister_one + gift_from_sister_two -
  lottery_ticket_cost

-- The amount Jacoby needs to reach his goal
def amount_needed : ℤ := total_goal - current_total_money

-- The main statement to be proved
theorem jacoby_needs : amount_needed = 3214 := by
  -- The proof is skipped
  sorry

end jacoby_needs_l202_202851


namespace nine_sided_polygon_diagonals_count_l202_202689

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202689


namespace max_sum_x_y_l202_202996

theorem max_sum_x_y 
  (x y : ℝ)
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
sorry

end max_sum_x_y_l202_202996


namespace diagonals_in_regular_nine_sided_polygon_l202_202564

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202564


namespace regular_nine_sided_polygon_diagonals_l202_202759

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l202_202759


namespace sum_of_coefficients_l202_202368

noncomputable def g (x : ℂ) : ℂ := x^4 + (-6 : ℂ) * x^3 + (19 : ℂ) * x^2 + (-54 : ℂ) * x + (90 : ℂ)

theorem sum_of_coefficients (p q r s : ℂ) (h : g(x) = x^4 + p * x^3 + q * x^2 + r * x + s) :
  p + q + r + s = 49 := 
by
  sorry

end sum_of_coefficients_l202_202368


namespace points_per_member_correct_l202_202381

noncomputable def points_per_member (total_members: ℝ) (absent_members: ℝ) (total_points: ℝ) :=
  (total_points / (total_members - absent_members))

theorem points_per_member_correct:
  points_per_member 5.0 2.0 6.0 = 2.0 :=
by 
  sorry

end points_per_member_correct_l202_202381


namespace polar_equation_of_circle_intersection_point_distance_l202_202841

-- Parametric equations of circle C
def parametric_equation_x (α : ℝ) : ℝ := 2 * (1 + Real.cos α)
def parametric_equation_y (α : ℝ) : ℝ := 2 * Real.sin α

-- Condition for the polar line
def theta_0 : ℝ := Real.arctan (Real.sqrt 7 / 3)

-- Polar equation of circle C
theorem polar_equation_of_circle : ∀ (θ : ℝ), ∀ (ρ : ℝ),
  (x ρ θ = 2 * (1 + Real.cos (θ))) → 
  (y ρ θ = 2 * Real.sin θ) → 
  (ρ = 4 * Real.cos θ) :=
sorry

-- Intersection point distance
theorem intersection_point_distance : ∀ ρ θ,
  (θ = theta_0) →
  (ρ = 4 * Real.cos θ) →
  (ρ = 3) :=
sorry

end polar_equation_of_circle_intersection_point_distance_l202_202841


namespace diagonals_in_regular_nine_sided_polygon_l202_202531

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202531


namespace unique_intersection_l202_202422

open Real

def circle (x y b : ℝ) : Prop :=
  x^2 + y^2 = 4 * b^2

def parabola (x b : ℝ) : ℝ :=
  x^2 - 2 * b

theorem unique_intersection (b : ℝ) :
  (∃ x y : ℝ, circle x y b ∧ y = parabola x b) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, circle x₁ y₁ b ∧ y₁ = parabola x₁ b ∧ circle x₂ y₂ b ∧ y₂ = parabola x₂ b → (x₁ = x₂ ∧ y₁ = y₂))
  ↔ b = 1/4 := 
sorry

end unique_intersection_l202_202422


namespace two_m_lt_three_n_l202_202875

variables (G : SimpleGraph) (n m : Nat)
  (h_vertices : G.numVertices = n)
  (h_edges : G.numEdges = m)
  (h_no_shared_cycle_edges : ∀ (c1 c2 : SimpleGraph.Cycle G), c1 ≠ c2 → c1.edges ∩ c2.edges = ∅)

theorem two_m_lt_three_n
  (h_simple : G.simple)
  : 2 * m < 3 * n :=
sorry

end two_m_lt_three_n_l202_202875


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202601

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202601


namespace note_relationship_l202_202353

theorem note_relationship
  (x y z : ℕ) 
  (h1 : x + 5 * y + 10 * z = 480)
  (h2 : x + y + z = 90)
  (h3 : y = 2 * x)
  (h4 : z = 3 * x) : 
  x = 15 ∧ y = 30 ∧ z = 45 :=
by 
  sorry

end note_relationship_l202_202353


namespace maximize_profit_l202_202165

open Real

-- Definitions and conditions
def x (t : ℝ) : ℝ := 4 - 3 / (2 * t + 1)
def fixed_investment : ℝ := 6
def additional_investment (x : ℝ) : ℝ := 12 * x
def selling_price_per_unit (x : ℝ) : ℝ := 1.5 * (fixed_investment + additional_investment x) / x
def profit (t : ℝ) : ℝ := selling_price_per_unit (x t) * (x t) - (fixed_investment + additional_investment (x t) + t)
def m (t : ℝ) : ℝ := 2 * t + 1

theorem maximize_profit :
  (∀ t ≥ 0, 27 - 18 / (2 * t + 1) - t ≤ 21.5) ∧
  (27 - 18 / (2 * 2.5 + 1) - 2.5 = 21.5) :=
by
  sorry

end maximize_profit_l202_202165


namespace max_height_l202_202370

def height (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

theorem max_height : ∃ t : ℝ, height t = 41.25 :=
by 
  sorry

end max_height_l202_202370


namespace incorrect_derivative_operation_l202_202336

theorem incorrect_derivative_operation :
  ¬(∀ x: ℝ, deriv (λ x, (cos x) / x) x = (x * sin x - cos x) / (x^2)) :=
by sorry

end incorrect_derivative_operation_l202_202336


namespace diagonals_in_regular_nine_sided_polygon_l202_202530

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202530


namespace commentator_mistake_l202_202836

def round_robin_tournament : Prop :=
  ∀ (x y : ℝ),
    x + 2 * x + 13 * y = 105 ∧ x < y ∧ y < 2 * x → False

theorem commentator_mistake : round_robin_tournament :=
  by {
    sorry
  }

end commentator_mistake_l202_202836


namespace diagonals_in_nine_sided_polygon_l202_202786

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202786


namespace function_not_monotonically_increasing_on_interval_l202_202106

-- Define the function
def f (x : ℝ) : ℝ := -Real.cos (2 * x + (3 * Real.pi / 4))

-- Define the interval
def interval : Set ℝ := Set.Icc (Real.pi / 8) (5 * Real.pi / 8)

-- Theorem stating the incorrect claim about monotonicity
theorem function_not_monotonically_increasing_on_interval : 
  ¬ (∀ x y ∈ interval, x ≤ y → f x ≤ f y) := 
sorry

end function_not_monotonically_increasing_on_interval_l202_202106


namespace find_f_x_l202_202063

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem find_f_x (x : ℝ) : (f (x+1)) = x^2 - 3*x + 2 :=
by
  sorry

end find_f_x_l202_202063


namespace max_wind_power_speed_l202_202954

def sail_force (A S ρ v0 v: ℝ) : ℝ :=
  (A * S * ρ * (v0 - v)^2) / 2

def wind_power (A S ρ v0 v: ℝ) : ℝ :=
  (sail_force A S ρ v0 v) * v

theorem max_wind_power_speed (A ρ: ℝ) (v0: ℝ) (S: ℝ) (h: v0 = 4.8 ∧ S = 4) :
  ∃ v, (wind_power A S ρ v0 v) = max ((wind_power A S ρ v0 v)) ∧ v = 1.6 :=
begin
  sorry
end

end max_wind_power_speed_l202_202954


namespace probability_even_sum_l202_202026

open Set

def balls : Set ℕ := {i | 1 ≤ i ∧ i ≤ 15}

def event_even_sum (x y : ℕ) : Prop := (x + y) % 2 = 0

theorem probability_even_sum : 
  (∑ i in balls, ∑ j in (balls \ {i}), if event_even_sum i j then 1 else 0)
  / (∑ i in balls, ∑ j in (balls \ {i}), 1)
  = 7 / 15 :=
by
  sorry

end probability_even_sum_l202_202026


namespace regular_nine_sided_polygon_diagonals_l202_202667

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202667


namespace Oddland_Squareland_bijection_l202_202830

-- Definitions according to conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_square (n : ℕ) : Prop := ∃ k : ℕ, k ^ 2 = n

-- Representing valid frankings in Oddland
structure OddlandFranking (n : ℕ) :=
(stamps : ℕ → ℕ) -- stamps d where d is the value of the stamp which is an odd number
(valid : ∀ d₁ d₂ : ℕ, is_odd d₁ → is_odd d₂ → d₁ ≤ d₂ → stamps d₁ ≥ stamps d₂)
(sum_eq : ∑ d in (Finset.filter is_odd (Finset.range (n+1))), d * stamps d = n)

-- Representing valid frankings in Squareland
structure SquarelandFranking (n : ℕ) :=
(stamps : ℕ → ℕ) -- stamps e where e is the value of the stamp which is a square number
(sum_eq : ∑ e in (Finset.filter is_square (Finset.range (n+1))), e * stamps e = n)

-- The main theorem stating the bijective correspondence
theorem Oddland_Squareland_bijection (n : ℕ) :
  ∃ (f : OddlandFranking n → SquarelandFranking n) (g : SquarelandFranking n → OddlandFranking n),
    (∀ o, g (f o) = o) ∧ (∀ s, f (g s) = s) := sorry

end Oddland_Squareland_bijection_l202_202830


namespace diagonals_in_nine_sided_polygon_l202_202795

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202795


namespace distance_between_points_l202_202399

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distance_between_points : distance (3, 4) (0, 1) = 3 * real.sqrt 2 := 
by
  sorry

end distance_between_points_l202_202399


namespace total_circle_area_l202_202457

open Real

noncomputable def total_area (r1 : ℝ) (r : ℝ) : ℝ :=
let A1 := π * r1^2 in
A1 / (1 - r^2)

theorem total_circle_area :
  total_area (10/3) (4/9) = 180 * π / 13 :=
by
  sorry

end total_circle_area_l202_202457


namespace triangle_area_l202_202173

open Real

-- Define the conditions
variables (a : ℝ) (B : ℝ) (cosA : ℝ)
variable (S : ℝ)

-- Given conditions of the problem
def triangle_conditions : Prop :=
  a = 5 ∧ B = π / 3 ∧ cosA = 11 / 14

-- State the theorem to be proved
theorem triangle_area (h : triangle_conditions a B cosA) : S = 10 * sqrt 3 :=
sorry

end triangle_area_l202_202173


namespace regular_nine_sided_polygon_diagonals_l202_202671

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202671


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202600

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202600


namespace set1_harmonious_set4_harmonious_l202_202476

-- Definitions for harmonious set
def harmonious_set {α : Type*} (G : set α) (op : α → α → α) : Prop :=
  ∃ (e : α), (∀ a b ∈ G, op a b ∈ G) ∧ (∀ a ∈ G, op a e = a ∧ op e a = a)

-- Define the sets and operations
def set1 := {n : ℕ | true}
def add_op (a b : ℝ) : ℝ := a + b

def set4 := {x : ℝ | ∃ (a b : ℚ), x = a + b * (real.sqrt 2)}
def mul_op (a b : ℝ) : ℝ := a * b

-- Theorem statements
theorem set1_harmonious : harmonious_set set1 add_op := sorry

theorem set4_harmonious : harmonious_set set4 mul_op := sorry

end set1_harmonious_set4_harmonious_l202_202476


namespace largest_n_divisibility_l202_202999

theorem largest_n_divisibility (n : ℕ) (h : n + 12 ∣ n^3 + 144) : n ≤ 132 :=
  sorry

end largest_n_divisibility_l202_202999


namespace abs_inequality_solution_l202_202924

theorem abs_inequality_solution (x : ℝ) : 
  (|2 * x + 1| > 3) ↔ (x > 1 ∨ x < -2) :=
sorry

end abs_inequality_solution_l202_202924


namespace total_time_on_missions_l202_202193

def original_time_first_mission : ℤ := 5
def percent_longer : ℤ := 60
def time_second_mission : ℤ := 3

theorem total_time_on_missions : 
  let time_first_mission := original_time_first_mission + (original_time_first_mission * percent_longer / 100)
  (time_first_mission + time_second_mission) = 11 := 
by
  let time_first_mission := original_time_first_mission + (original_time_first_mission * percent_longer / 100)
  have h : time_first_mission = 8 := sorry
  show (time_first_mission + time_second_mission) = 11, from
    calc
      time_first_mission + time_second_mission = 8 + 3 : by rw h
      ... = 11 : by norm_num

end total_time_on_missions_l202_202193


namespace regular_nine_sided_polygon_diagonals_l202_202676

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202676


namespace mariel_dogs_count_l202_202904

theorem mariel_dogs_count
  (num_dogs_other: Nat)
  (num_legs_tangled: Nat)
  (num_legs_per_dog: Nat)
  (num_legs_per_human: Nat)
  (num_dog_walkers: Nat)
  (num_dogs_mariel: Nat):
  num_dogs_other = 3 →
  num_legs_tangled = 36 →
  num_legs_per_dog = 4 →
  num_legs_per_human = 2 →
  num_dog_walkers = 2 →
  4*num_dogs_mariel + 4*num_dogs_other + 2*num_dog_walkers = num_legs_tangled →
  num_dogs_mariel = 5 :=
by 
  intros h_other h_tangled h_legs_dog h_legs_human h_walkers h_eq
  sorry

end mariel_dogs_count_l202_202904


namespace quadrilateral_bisector_rhombus_AB_eq_CD_l202_202164

/-- In a quadrilateral ABCD, where sides AD and BC are parallel, if the bisectors of 
    angles DAC, DBC, ACB, and ADB form a rhombus, then AB equals CD. -/
theorem quadrilateral_bisector_rhombus_AB_eq_CD {A B C D : Type*} [quadrilateral ABCD] 
  (par_AD_BC : AD ∥ BC) 
  (rhombus_bisectors : is_rhombus (bisectors_quad ABCD)) :
  AB = CD := 
sorry

end quadrilateral_bisector_rhombus_AB_eq_CD_l202_202164


namespace nine_sided_polygon_diagonals_count_l202_202690

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202690


namespace sphere_surface_area_l202_202234

theorem sphere_surface_area 
  (A B C D : Point) 
  (h_pts: are_collinear_on_sphere A B C D) 
  (h_angle_ABC: ∠ABC = 2 * π / 3) 
  (h_AB_eq_AC: AB = AC) 
  (h_AD_perp_ABC: AD ⊥ Plane_ABC) 
  (h_AD_eq_6: AD = 6) 
  (h_AB_eq: AB = 2 * sqrt 3) : 
  surface_area (bounding_sphere A B C D) = 144 * π := 
sorry

end sphere_surface_area_l202_202234


namespace regular_nine_sided_polygon_diagonals_l202_202672

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l202_202672


namespace nine_sided_polygon_diagonals_count_l202_202687

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202687


namespace find_M_plus_N_l202_202136

theorem find_M_plus_N (M N : ℕ) 
  (h1 : 5 / 7 = M / 63) 
  (h2 : 5 / 7 = 70 / N) : 
  M + N = 143 :=
by
  sorry

end find_M_plus_N_l202_202136


namespace problem_solution_l202_202089

noncomputable def circle_constant : ℝ := Real.pi
noncomputable def natural_base : ℝ := Real.exp 1

theorem problem_solution (π : ℝ) (e : ℝ) (h₁ : π = Real.pi) (h₂ : e = Real.exp 1) :
  π * Real.log e / Real.log 3 > 3 * Real.log e / Real.log π := by
  sorry

end problem_solution_l202_202089


namespace negation_of_existential_prop_l202_202276

theorem negation_of_existential_prop :
  (¬ ∃ (x₀ : ℝ), x₀^2 + x₀ + 1 < 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_prop_l202_202276


namespace prove_identity_correct_l202_202141

noncomputable def identity_proof (x : ℚ) (h : 0 < x) : Prop :=
  ∀ (a b : ℚ), (a / (2^x - 1) + b / (2^x + 2) = (2 * 2^x - 1) / ((2^x - 1) * (2^x + 2))) → (a - b = -4/3)

theorem prove_identity_correct : identity_proof :=
sorry

end prove_identity_correct_l202_202141


namespace total_population_l202_202159

-- Definitions based on conditions
variables (b g t : ℕ)
hypothesis h1 : b = 4 * g
hypothesis h2 : g = 8 * t

-- Main statement
theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) :
  b + g + t = 41 * b / 32 :=
by sorry

end total_population_l202_202159


namespace range_of_a_l202_202146

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Icc (-2 : ℝ) 2, 2 * x^3 - 3 * x^2 + a = 0) → 
  a ∈ Ioo (-4 : ℝ) 0 ∪ Ioc (1 : ℝ) 28 :=
by sorry

end range_of_a_l202_202146


namespace nine_sided_polygon_diagonals_l202_202586

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202586


namespace alpha_dominates_beta_l202_202888

-- Define the types of functions T_alpha and T_beta
variable {α β : Type}
variable {Tₐ T_b : ℝ → ℝ → ℝ → ℝ}

def dominates (α β : Type) : Prop :=
  ∀ (x y z: ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → Tₐ x y z ≥ T_b x y z → α ≻ β

-- Main theorem statement
theorem alpha_dominates_beta (h : ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → Tₐ x y z ≥ T_b x y z) : 
  dominates α β :=
by
  intros x y z hx hy hz hT
  sorry

end alpha_dominates_beta_l202_202888


namespace area_complex_geometry_l202_202463

-- Define side length of the squares
def side_length : ℝ := 8

-- Define the angle of rotation for the second sheet from the bottom
def rotation_angle_2nd_sheet : ℝ := 45

-- Define the angle of rotation for the topmost sheet
def rotation_angle_top_sheet : ℝ := 90

-- Required to compute and prove the area of the complex geometry formed
theorem area_complex_geometry :
  ∃ (a b c : ℕ), c ≠ 0 ∧ (∀ (p : ℕ), prime p → ¬(p^2 ∣ c)) ∧ 
                 (area : ℝ) = a - b * real.sqrt c ∧
                 192 - 128 * real.sqrt 2 = a - b * real.sqrt c :=
sorry

end area_complex_geometry_l202_202463


namespace moles_of_HCl_formed_l202_202420

-- Define a constant representing the number of moles of a substance.
constant moles (substance: Type) : ℕ

-- Define the substances involved in the reaction.
inductive Substance
| Methane : Substance
| Chlorine : Substance
| HydrochloricAcid : Substance
| MethylChloride : Substance

open Substance

-- Define a hypothesis that states the initial moles of substances.
axiom initial_moles_methane : moles Methane = 3
axiom initial_moles_chlorine : moles Chlorine = 3

-- Intermediary formations (not explicitly quantified here for simplicity)
axiom intermediate_methyl_chloride : ∃ n, moles MethylChloride = n

-- Define the reaction conditions and outcome.
noncomputable def number_of_moles_of_HCl (methane chlorine: ℕ) : ℕ :=
  if (methane = 3 ∧ chlorine = 3) then 3 else 0

-- Main theorem stating the outcome.
theorem moles_of_HCl_formed :
  number_of_moles_of_HCl (moles Methane) (moles Chlorine) = 3 :=
sorry

end moles_of_HCl_formed_l202_202420


namespace increase_in_expenditure_l202_202835

theorem increase_in_expenditure :
  ∃ (A : ℝ), 
    let orig_students : ℝ := 100 in
    let new_students : ℝ := orig_students + 20 in
    let decrease_per_student : ℝ := 5 in
    let new_total_expenditure : ℝ := 5400 in
    let new_average_expenditure := A - decrease_per_student in
    let orig_total_expenditure := orig_students * A in
    (new_students * new_average_expenditure = new_total_expenditure) →
    (new_total_expenditure - orig_total_expenditure = 400) → 
    new_total_expenditure - orig_total_expenditure = 400 :=
begin
  intro h,
  use 50, -- the original average expenditure per student
  sorry
end

end increase_in_expenditure_l202_202835


namespace nine_sided_polygon_diagonals_count_l202_202694

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202694


namespace max_gcd_a_n_a_n_plus_1_l202_202881

def a_n (n : ℕ) : ℚ := (10^n - 1) / 9
def d_n (n : ℕ) : ℚ := Int.gcd (a_n n) (a_n (n + 1))

theorem max_gcd_a_n_a_n_plus_1 (n : ℕ) : d_n n = 1 := by
  sorry

end max_gcd_a_n_a_n_plus_1_l202_202881


namespace diagonals_in_nonagon_l202_202720

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202720


namespace nine_sided_polygon_diagonals_count_l202_202697

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202697


namespace num_digits_smallest_n_l202_202207

open Nat

noncomputable def smallest_n : ℕ :=
  let n := 2^6 * 3^6 * 5^6 in
  n

theorem num_digits_smallest_n :
  (Nat.digits 10 smallest_n).length = 9 :=
  by
    sorry

end num_digits_smallest_n_l202_202207


namespace hyperbola_eccentricity_sqrt2_l202_202009

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : b = a) : ℝ :=
  let c := real.sqrt (a^2 + b^2)
  in c / a

theorem hyperbola_eccentricity_sqrt2 (a : ℝ) (ha : 0 < a) : hyperbola_eccentricity a a ha ha rfl = real.sqrt 2 :=
by
  unfold hyperbola_eccentricity
  rw [← add_self_eq_add_self, ← real.sqrt_mul_self (add_nonneg (pow_two_nonneg a) (pow_two_nonneg a)), real.mul_self_sqrt (add_nonneg (mul_self_nonneg a) (mul_self_nonneg a))]
  ring

-- Note:
-- ha and hb stand for the conditions a > 0 and b > 0 respectively.
-- h represents the condition b = a
-- We ensure that the definitions and conditions are translated directly to match the provided problem.

end hyperbola_eccentricity_sqrt2_l202_202009


namespace max_k_sum_ai_l202_202255

theorem max_k_sum_ai (n : ℕ) (h : n ≥ 3) : ∃ (k : ℕ), (∀ j : ℕ, j ≤ k → ∃ (a : ℕ → ℝ), (∀ i : ℕ, i < n → a i ∈ set.Ico 0 1) ∧ (∃ (S : finset ℕ), S.card > 0 ∧ S.sum a = j)) ∧ k = n - 2 :=
by {
  sorry
}

end max_k_sum_ai_l202_202255


namespace required_moles_of_H2O_l202_202036

-- Definition of the balanced chemical reaction
def balanced_reaction_na_to_naoh_and_H2 : Prop :=
  ∀ (NaH H2O NaOH H2 : ℕ), NaH + H2O = NaOH + H2

-- The given moles of NaH
def moles_NaH : ℕ := 2

-- Assertion that we need to prove: amount of H2O required is 2 moles
theorem required_moles_of_H2O (balanced : balanced_reaction_na_to_naoh_and_H2) : 
  (2 * 1) = 2 :=
by
  sorry

end required_moles_of_H2O_l202_202036


namespace product_roots_example_l202_202007

def cubic_eq (a b c d : ℝ) (x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

noncomputable def product_of_roots (a b c d : ℝ) : ℝ := -d / a

theorem product_roots_example : product_of_roots 4 (-2) (-25) 36 = -9 := by
  sorry

end product_roots_example_l202_202007


namespace price_of_peaches_is_2_l202_202223

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

end price_of_peaches_is_2_l202_202223


namespace intersection_A_B_union_A_B_subset_C_A_l202_202897

def set_A : Set ℝ := { x | x^2 - x - 2 > 0 }
def set_B : Set ℝ := { x | 3 - abs x ≥ 0 }
def set_C (p : ℝ) : Set ℝ := { x | 4 * x + p < 0 }

theorem intersection_A_B : set_A ∩ set_B = { x | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3) } :=
sorry

theorem union_A_B : set_A ∪ set_B = Set.univ :=
sorry

theorem subset_C_A (p : ℝ) : set_C p ⊆ set_A → p ≥ 4 :=
sorry

end intersection_A_B_union_A_B_subset_C_A_l202_202897


namespace journey_time_l202_202363

theorem journey_time :
  let total_distance := 224
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 21
  let speed_second_half := 24
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  total_time = 10 := by
  have first_half_distance_eq : first_half_distance = 112 := by norm_num
  have second_half_distance_eq : second_half_distance = 112 := by norm_num
  have time_first_half_eq : time_first_half = (112 / 21) := rfl
  have time_second_half_eq : time_second_half = (112 / 24) := rfl
  have total_time_eq : total_time = (112 / 21) + (112 / 24) := rfl
  have fraction_sum_eq : (112 / 21) + (112 / 24) = 10 := by norm_num
  show total_time = 10 from fraction_sum_eq

end journey_time_l202_202363


namespace nine_sided_polygon_diagonals_l202_202660

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202660


namespace diagonals_in_nine_sided_polygon_l202_202748

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202748


namespace perimeter_triangleABC_l202_202163

-- Define the basic structure and conditions of the problem
variable (A B C : Type) [LinearOrderedField A]

-- Given conditions
variables 
  (triangleABC : IsoscelesTriangle A B C) 
  (sin_ratio : (sin A) / (sin B) = 1 / 2) 
  (BC_len : BC = 10)

-- Prove the statement
theorem perimeter_triangleABC : (perimeter triangleABC) = 50 := 
by 
  -- Include a placeholder for the proof
  sorry

end perimeter_triangleABC_l202_202163


namespace state_b_more_candidates_selected_l202_202155

theorem state_b_more_candidates_selected 
  (total_candidates : ℕ)
  (state_a_percentage : ℚ)
  (state_b_percentage : ℚ)
  (num_candidates : total_candidates = 7900)
  (state_a : state_a_percentage = 6 / 100)
  (state_b : state_b_percentage = 7 / 100)
  : (state_b_percentage * total_candidates - state_a_percentage * total_candidates) = 79 :=
by 
  have h : total_candidates = 7900, from num_candidates,
  have a : state_a_percentage * total_candidates = 6 / 100 * 7900, by rw [state_a, h],
  have b : state_b_percentage * total_candidates = 7 / 100 * 7900, by rw [state_b, h],
  calc
    (state_b_percentage * total_candidates - state_a_percentage * total_candidates)
      = (7 / 100 * 7900 - 6 / 100 * 7900) : by rw [a, b]
  ... = 7900 / 100 * (7 - 6) : by norm_num
  ... = 79 : by norm_num

end state_b_more_candidates_selected_l202_202155


namespace sum_of_roots_quadratic_l202_202972

theorem sum_of_roots_quadratic : 
  let x1, x2 : ℝ in
  (∃ x1 x2, 2 * x1^2 - 3 * x1 - 5 = 0 ∧ 2 * x2^2 - 3 * x2 - 5 = 0 ∧ x1 ≠ x2) →
  x1 + x2 = 3 / 2 := 
by
  sorry

end sum_of_roots_quadratic_l202_202972


namespace nine_sided_polygon_diagonals_l202_202591

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202591


namespace counting_numbers_leave_remainder_6_divide_53_l202_202129

theorem counting_numbers_leave_remainder_6_divide_53 :
  ∃! n : ℕ, (∃ k : ℕ, 53 = n * k + 6) ∧ n > 6 :=
sorry

end counting_numbers_leave_remainder_6_divide_53_l202_202129


namespace nine_sided_polygon_diagonals_count_l202_202699

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l202_202699


namespace find_number_l202_202331

-- Define the number 40 and the percentage 90.
def num : ℝ := 40
def percent : ℝ := 0.9

-- Define the condition that 4/5 of x is smaller than 90% of 40 by 16
def condition (x : ℝ) : Prop := (4/5 : ℝ) * x = percent * num - 16

-- Proof statement in Lean 4
theorem find_number : ∃ x : ℝ, condition x ∧ x = 25 :=
by 
  use 25
  unfold condition
  norm_num
  sorry

end find_number_l202_202331


namespace area_triangle_EFG_l202_202926

-- Definitions for the problem
variable {A B C D E F G : Type}
variable [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D]
variable [EuclideanGeometry E] [EuclideanGeometry F] [EuclideanGeometry G]

-- Conditions provided in a) 
def rectangle_ABCD_area (A B C D : Type) [r: EuclideanGeometry A] [r: EuclideanGeometry B] [r: EuclideanGeometry C] [r: EuclideanGeometry D] : Prop := 
  -- Assuming that the area of rectangle ABCD is 96 square meters
  euclidean_geometry.Area_rectangle A B C D = 96

def midpoints (A B C D E F G : Type) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry E] [EuclideanGeometry F] [EuclideanGeometry G] : Prop :=
  -- E, F, and G are midpoints of AB, BC, and CD
  euclidean_geometry.Midpoint E A B ∧ euclidean_geometry.Midpoint F B C ∧ euclidean_geometry.Midpoint G C D

-- The proof statement
theorem area_triangle_EFG (A B C D E F G : Type) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry E] [EuclideanGeometry F] [EuclideanGeometry G]:
  rectangle_ABCD_area A B C D →
  midpoints A B C D E F G →
  euclidean_geometry.Area_triangle E F G = 12 := 
by
  intro h_area_rect
  intro h_midpoints
  -- Proof would go here
  sorry

end area_triangle_EFG_l202_202926


namespace diagonals_in_nine_sided_polygon_l202_202744

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202744


namespace sum_of_first_10_common_elements_is_correct_l202_202455

-- Define arithmetic progression
def a (n : ℕ) : ℕ := 4 + 3 * n

-- Define geometric progression
def b (k : ℕ) : ℕ := 20 * (2 ^ k)

-- Define the sum of the first 10 common elements in both sequences
def sum_first_10_common_elements : ℕ := 13981000

-- Statement of the proof problem in Lean 4
theorem sum_of_first_10_common_elements_is_correct :
  ∑ i in (finset.range 10).image (λ k, b(2*k + 1)), id = sum_first_10_common_elements :=
by
  -- Proof omitted
  sorry

end sum_of_first_10_common_elements_is_correct_l202_202455


namespace largest_possible_sum_l202_202142

theorem largest_possible_sum (a b c : ℕ) (y : ℕ) (h₀ : a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h₁ : b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (h₂ : c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h₃ : y > 0) (h₄ : y ≤ 15) (h₅ : (a * 100 + b * 10 + c) * y = 1000) :
  a + b + c = 8 :=
sorry

end largest_possible_sum_l202_202142


namespace regular_nine_sided_polygon_has_27_diagonals_l202_202608

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l202_202608


namespace Cat_favorite_number_is_24_l202_202003

noncomputable def Cat_favorite_number : ℕ :=
  if h : ∃ A B : ℕ, 10 ≤ 10 * A + B ∧ 10 * A + B < 100 ∧
    A ≠ 0 ∧ B ≠ 0 ∧ A ≠ B ∧ (10 * A + B) % A = 0 ∧ (10 * A + B) % B = 0 ∧
    ∃ (nums : Finset ℕ), nums = {12, 24, 36, 48} ∧ 
    ∀ n ∈ nums, 
      ((∃ m₁ m₂ m₃, (m₁ ∈ nums ∧ m₂ ∈ nums ∧ m₃ ∈ nums ∧ m₁ ≠ m₂ ∧ m₁ ≠ m₃ ∧ m₂ ≠ m₃ ∧ n + m₁ - m₂ = m₃)) ∧ 
       ∃ m₁ m₂, (m₁ ∈ nums ∧ m₂ ∈ nums ∧ n^2 = m₁ * m₂)) 
  then 24 else 0

theorem Cat_favorite_number_is_24 : Cat_favorite_number = 24 :=
by sorry

end Cat_favorite_number_is_24_l202_202003


namespace alex_class_size_l202_202832

theorem alex_class_size 
  (n : ℕ) 
  (h_top : 30 ≤ n)
  (h_bottom : 30 ≤ n) 
  (h_better : n - 30 > 0)
  (h_worse : n - 30 > 0)
  : n = 59 := 
sorry

end alex_class_size_l202_202832


namespace volume_tetrahedron_PACD_l202_202492

variable {A B C D P : ℝ^3}
variable {unit_square : ∃(A B C D : ℝ^3), A = (0, 0, 0) ∧ B = (1, 0, 0) ∧ C = (1, 1, 0) ∧ D = (0, 1, 0) ∧ (B - A) = 0.5 * (C - D)}
variable {midpoint_P : P = ((B + A) / 2)}

theorem volume_tetrahedron_PACD (unit_square: ∃(A B C D : ℝ^3), A = (0, 0, 0) ∧ B = (1, 0, 0) ∧ C = (1, 1, 0) ∧ D = (0, 1, 0) ∧ (B - A) = 0.5 * (C - D)) (midpoint_P: P = ((B + A) / 2)) :
  volume_tetrahedron P A C D = (Real.sqrt 3) / 24 :=
sorry

end volume_tetrahedron_PACD_l202_202492


namespace number_of_men_is_nine_l202_202251

noncomputable def num_men_went_to_hotel : ℕ :=
  let total_expense := 29.25
  let expense_eight_men := 8 * 3
  let remaining_expense := total_expense - expense_eight_men
  let man_expense := remaining_expense - 2 
  nat.floor (total_expense / man_expense) 

theorem number_of_men_is_nine (n : ℕ) (total_expense := 29.25) (expense_eight_men := 8 * 3) (remaining_expense := total_expense - expense_eight_men) (man_expense := remaining_expense - 2) : (k : ℕ) -> nat.floor (total_expense / man_expense) = k -> 
  k= n  ->
  n = 9
:= sorry

end number_of_men_is_nine_l202_202251


namespace sin_alpha_l202_202815

theorem sin_alpha (α : ℝ) (h1 : α ∈ set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.cos (α + Real.pi / 3) = -4 / 5) : 
  Real.sin α = (3 + 4 * Real.sqrt 3) / 10 := 
sorry

end sin_alpha_l202_202815


namespace number_of_diagonals_l202_202548

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202548


namespace problem_1_problem_2_l202_202113

theorem problem_1 (a : ℝ) : (∀ x1 x2 : ℝ, (a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) → x1 = x2) → (a = 0 ∨ a = 1) :=
by sorry

theorem problem_2 (a : ℝ) : (∀ x1 x2 : ℝ, (a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) → x1 = x2 ∨ ¬ ∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a ≥ 1 ∨ a = 0) :=
by sorry

end problem_1_problem_2_l202_202113


namespace num_sequences_to_initial_position_8_l202_202374

def validSequenceCount : ℕ := 4900

noncomputable def numberOfSequencesToInitialPosition (n : ℕ) : ℕ :=
if h : n = 8 then validSequenceCount else 0

theorem num_sequences_to_initial_position_8 :
  numberOfSequencesToInitialPosition 8 = 4900 :=
by
  sorry

end num_sequences_to_initial_position_8_l202_202374


namespace diagonals_in_regular_nine_sided_polygon_l202_202538

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202538


namespace combined_rate_of_mpg_l202_202923

-- Defining the average rate of Ray's car
def ray_car_mpg : ℚ := 50

-- Defining the average rate of Tom's car
def tom_car_mpg : ℚ := 20

-- Defining the distance driven by Ray
def ray_distance : ℚ := 150

-- Defining the distance driven by Tom
def tom_distance : ℚ := 300

-- Calculating the gasoline used by Ray
def ray_gallons_used : ℚ := ray_distance / ray_car_mpg

-- Calculating the gasoline used by Tom
def tom_gallons_used : ℚ := tom_distance / tom_car_mpg

-- Calculating the total gasoline used
def total_gallons_used : ℚ := ray_gallons_used + tom_gallons_used

-- Calculating the total distance driven
def total_distance_driven : ℚ := ray_distance + tom_distance

-- Calculating the combined miles per gallon
def combined_mpg : ℚ := total_distance_driven / total_gallons_used

-- The proof statement
theorem combined_rate_of_mpg : combined_mpg = 25 :=  
by 
  /- Proof goes here -/
  sorry

end combined_rate_of_mpg_l202_202923


namespace max_trig_expression_l202_202041

open Real

theorem max_trig_expression (x y z : ℝ) :
  (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 := sorry

end max_trig_expression_l202_202041


namespace diagonals_in_regular_nine_sided_polygon_l202_202574

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l202_202574


namespace max_value_of_expression_l202_202045

theorem max_value_of_expression :
  ∃ (x y z : ℝ), 
    let expr := (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) in 
    expr = 4.5 ∧
    ∀ (x y z : ℝ), 
      (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 :=
begin
  sorry,
end

end max_value_of_expression_l202_202045


namespace quadratic_complete_square_l202_202429

-- Given a quadratic polynomial
def quadratic_polynomial (x : ℝ) : ℝ := 6 * x^2 - 12 * x + 4

-- Define the completed square form
def completed_square_form (x : ℝ) (a h k : ℝ) : ℝ := a * (x - h)^2 + k

-- The theorem to prove the correctness of the transformation and equality of the sum a + h + k
theorem quadratic_complete_square :
  ∃ a h k : ℝ, (∀ x : ℝ, quadratic_polynomial x = completed_square_form x a h k) ∧ a + h + k = 5 :=
by
  exists 6 1 (-2)
  split
  { intro x
    -- This asserts the equality of the transformed forms
    calc
      quadratic_polynomial x
          = 6 * x^2 - 12 * x + 4                : rfl
      ... = 6 * (x^2 - 2 * x) + 4               : by ring
      ... = 6 * ((x - 1)^2 - 1) + 4             : by ring_exp
      ... = 6 * (x - 1)^2 - 6 + 4               : by ring
      ... = 6 * (x - 1)^2 - 2                   : by ring }
  -- Proves the sum of a + h + k
  calc
    6 + 1 + (-2) = 5 : by ring

-- Using sorry as required to skip the proof for the equivalence of both forms.
sorry

end quadratic_complete_square_l202_202429


namespace top_card_probability_l202_202984

theorem top_card_probability :
  let total_cards := 104
  let favorable_outcomes := 4
  let probability := (favorable_outcomes : ℚ) / total_cards
  probability = 1 / 26 :=
by
  sorry

end top_card_probability_l202_202984


namespace half_time_score_30_l202_202425

-- Define sequence conditions
def arithmetic_sequence (a d : ℕ) : ℕ × ℕ × ℕ × ℕ := (a, a + d, a + 2 * d, a + 3 * d)
def geometric_sequence (b r : ℕ) : ℕ × ℕ × ℕ × ℕ := (b, b * r, b * r^2, b * r^3)

-- Define the sum of the first team
def first_team_sum (a d : ℕ) : ℕ := 4 * a + 6 * d

-- Define the sum of the second team
def second_team_sum (b r : ℕ) : ℕ := b * (1 + r + r^2 + r^3)

-- Define the winning condition
def winning_condition (a d b r : ℕ) : Prop := first_team_sum a d = second_team_sum b r + 2

-- Define the point sum constraint
def point_sum_constraint (a d b r : ℕ) : Prop := first_team_sum a d ≤ 100 ∧ second_team_sum b r ≤ 100

-- Define the constraints on r and d
def r_d_positive (r d : ℕ) : Prop := r > 1 ∧ d > 0

-- Define the half-time score for the first team
def first_half_first_team (a d : ℕ) : ℕ := a + (a + d)

-- Define the half-time score for the second team
def first_half_second_team (b r : ℕ) : ℕ := b + (b * r)

-- Define the total half-time score
def total_half_time_score (a d b r : ℕ) : ℕ := first_half_first_team a d + first_half_second_team b r

-- Main theorem: Total half-time score is 30 under given conditions
theorem half_time_score_30 (a d b r : ℕ) 
  (r_d_pos : r_d_positive r d) 
  (win_cond : winning_condition a d b r)
  (point_sum_cond : point_sum_constraint a d b r) : 
  total_half_time_score a d b r = 30 :=
sorry

end half_time_score_30_l202_202425


namespace max_red_squares_no_axis_parallel_rectangle_l202_202008

/--
In a 5x5 grid, the maximum number of squares that can be colored red
such that no four red squares form an axis-parallel rectangle is 12.
-/
theorem max_red_squares_no_axis_parallel_rectangle : 
  ∃ (R : Fin 5 → Fin 5 → Prop), (∀ i j, R i j → i < 5 ∧ j < 5) ∧ 
                               (∀ (i1 i2 i3 i4 j1 j2 j3 j4 : Fin 5), 
                                 R i1 j1 ∧ R i2 j2 ∧ R i3 j3 ∧ R i4 j4 →
                                 ¬(i1 = i2 ∧ i3 = i4 ∧ j1 = j3 ∧ j2 = j4)) ∧ 
                               (∃ n : ℕ, n = 12 ∧ 
                                 ∑ i, ∑ j, if R i j then 1 else 0 = n) :=
sorry

end max_red_squares_no_axis_parallel_rectangle_l202_202008


namespace johns_hats_cost_l202_202862

theorem johns_hats_cost 
  (weeks : ℕ)
  (days_in_week : ℕ)
  (cost_per_hat : ℕ) 
  (h : weeks = 2 ∧ days_in_week = 7 ∧ cost_per_hat = 50) 
  : (weeks * days_in_week * cost_per_hat) = 700 :=
by
  sorry

end johns_hats_cost_l202_202862


namespace Mason_tables_needed_l202_202221

theorem Mason_tables_needed
  (w_silverware_piece : ℕ := 4) 
  (n_silverware_piece_per_setting : ℕ := 3) 
  (w_plate : ℕ := 12) 
  (n_plates_per_setting : ℕ := 2) 
  (n_settings_per_table : ℕ := 8) 
  (n_backup_settings : ℕ := 20) 
  (total_weight : ℕ := 5040) : 
  ∃ (n_tables : ℕ), n_tables = 15 :=
by
  sorry

end Mason_tables_needed_l202_202221


namespace diagonals_in_regular_nine_sided_polygon_l202_202543

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l202_202543


namespace stating_sum_first_10_common_elements_l202_202453

/-- 
  Theorem stating that the sum of the first 10 elements that appear 
  in both the given arithmetic progression and the geometric progression 
  equals 13981000.
-/
theorem sum_first_10_common_elements :
  let AP := λ n : ℕ => 4 + 3 * n in
  let GP := λ k : ℕ => 20 * 2^k in
  let common_elements := (range 10).map (λ i => GP (2 * i + 1)) in
  (∑ i in (finset.range 10), common_elements[i]) = 13981000 :=
by
  let AP := λ n : ℕ => 4 + 3 * n
  let GP := λ k : ℕ => 20 * 2^k
  let common_elements := (finset.range 10).map (λ i => GP (2 * i + 1))
  have S : (∑ i in (finset.range 10), common_elements[i]) = 40 * (4^10 - 1) / 3,
  {
    sorry,
  }
  have : 40 * 349525 = 13981000 := by norm_num,
  exact this ▸ S

end stating_sum_first_10_common_elements_l202_202453


namespace nine_sided_polygon_diagonals_l202_202656

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l202_202656


namespace largest_number_is_27_l202_202344

-- Define the condition as a predicate
def three_consecutive_multiples_sum_to (k : ℕ) (sum : ℕ) : Prop :=
  ∃ n : ℕ, (3 * n) + (3 * n + 3) + (3 * n + 6) = sum

-- Define the proof statement
theorem largest_number_is_27 : three_consecutive_multiples_sum_to 3 72 → 3 * 7 + 6 = 27 :=
by
  intro h
  cases' h with n h_eq
  sorry

end largest_number_is_27_l202_202344


namespace stream_speed_l202_202328

theorem stream_speed (c v : ℝ) (h1 : c - v = 9) (h2 : c + v = 12) : v = 1.5 :=
by
  sorry

end stream_speed_l202_202328


namespace infinitely_many_solutions_l202_202238

theorem infinitely_many_solutions (t : ℕ) : 
  ∃ (l m n : ℕ), l = 2 * t ∧ m = 2 * t^2 - 2 ∧ n = 2 * t^2 - 1 ∧ l^2 + m^2 = n^2 + 3 := 
by {
  existsi [2 * t, 2 * t^2 - 2, 2 * t^2 - 1],
  simp,
  sorry  -- Proof to be filled in
}

end infinitely_many_solutions_l202_202238


namespace nine_sided_polygon_diagonals_l202_202620

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202620


namespace maximum_speed_l202_202951

variable (A S ρ v0 v : ℝ)

def force (A S ρ v0 v : ℝ) : ℝ := (A * S * ρ * (v0 - v) ^ 2) / 2

def power (A S ρ v0 v : ℝ) : ℝ := force A S ρ v0 v * v

theorem maximum_speed : 
  S = 4 ∧ v0 = 4.8 ∧ 
  ∃ A ρ v,
    (∀ v, power A S ρ v0 v ≤ power A S ρ v0 1.6) → 
    v == 1.6 :=
by
  sorry

end maximum_speed_l202_202951


namespace percentage_error_approx_l202_202332

noncomputable def central_angle_heptagon : ℝ := 360 / 7 / 2
noncomputable def side_length_heptagon : ℝ := 2 * Real.sin (central_angle_heptagon * Real.pi / 180)
noncomputable def half_chord_60_deg : ℝ := Real.sin (60 * Real.pi / 180)

noncomputable def percentage_error (a a' : ℝ) : ℝ :=
  100 * Real.abs ((a - a') / a)

theorem percentage_error_approx : 
  percentage_error side_length_heptagon half_chord_60_deg = 0.2 :=
by
  -- Proof steps should be added here
  sorry

end percentage_error_approx_l202_202332


namespace expression_value_l202_202253

theorem expression_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 :=
by
  sorry

end expression_value_l202_202253


namespace find_b_l202_202098

-- Definitions and conditions
def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x + b
def line (x : ℝ) : ℝ := 2 * x + 1
def tangent_at (f : ℝ → ℝ) (line : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = line x₀ ∧ deriv f x₀ = deriv line x₀

-- The actual statement to prove
theorem find_b (a b : ℝ) (h_tangent : tangent_at (λ x, f x a b) line 1) : b = 3 :=
by
  sorry

end find_b_l202_202098


namespace nine_sided_polygon_diagonals_l202_202618

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l202_202618


namespace length_of_faster_train_is_correct_l202_202309

def speed_faster_train := 54 -- kmph
def speed_slower_train := 36 -- kmph
def crossing_time := 27 -- seconds

def kmph_to_mps (s : ℕ) : ℕ :=
  s * 1000 / 3600

def relative_speed_faster_train := kmph_to_mps (speed_faster_train - speed_slower_train)

def length_faster_train := relative_speed_faster_train * crossing_time

theorem length_of_faster_train_is_correct : length_faster_train = 135 := 
  by
  sorry

end length_of_faster_train_is_correct_l202_202309


namespace trigonometric_properties_l202_202505

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem trigonometric_properties (ω φ : ℝ) (hω_pos : ω > 0) (hφ_bounds : 0 < φ ∧ φ < π) 
  (h1 : f ω φ (π / 8) = sqrt 2) (h2 : f ω φ (π / 2) = 0) (h_mono : ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < π → f ω φ x ≤ f ω φ y) :
  ∀ x : ℝ, -π ≤ x ∧ x ≤ -π / 2 → ∀ y : ℝ, -π ≤ y ∧ y ≤ -π / 2 ∧ x < y → f ω φ x ≤ f ω φ y :=
by
  sorry

end trigonometric_properties_l202_202505


namespace perimeter_feet_of_altitudes_l202_202239

theorem perimeter_feet_of_altitudes (ABC : Triangle) (D E F : Point) 
  (h1 : is_acute_triangle ABC) 
  (h2 : feet_of_altitudes ABC D E F) 
  (p p1 : ℝ) (S : ℝ)
  (r R : ℝ) 
  (hp : p = semi_perimeter ABC)
  (hS : S = area ABC)
  (hr : r = inradius ABC)
  (hR : R = circumradius ABC)
  (hp1 : p1 = semi_perimeter (triangle DEF)) :
  p1 ≤ (1/2) * p :=
by
  sorry

end perimeter_feet_of_altitudes_l202_202239


namespace square_side_length_l202_202944

theorem square_side_length (radius : ℝ) (s1 s2 : ℝ) (h1 : s1 = s2) (h2 : radius = 2 - Real.sqrt 2):
  s1 = 1 :=
  sorry

end square_side_length_l202_202944


namespace regular_nonagon_diagonals_correct_l202_202700

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l202_202700


namespace thomas_books_l202_202302

def number_of_books := Nat

variable (B : number_of_books)
variable (books_sell_price : ℝ := 1.5)
variable (record_cost : ℝ := 3)
variable (records_bought : number_of_books := 75)
variable (money_left_over : ℝ := 75)
variable (total_money_obtained : ℝ := 225 + 75)

theorem thomas_books (h : 1.5 * (B : ℝ) = total_money_obtained) : B = 200 :=
by sorry

end thomas_books_l202_202302


namespace Dan_gave_her_cards_l202_202931

variable (initial_cards total_cards bought_cards received_cards : ℕ)

theorem Dan_gave_her_cards :
  initial_cards = 27 →
  total_cards = 88 →
  bought_cards = 20 →
  total_cards - bought_cards - initial_cards = received_cards →
  received_cards = 41 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end Dan_gave_her_cards_l202_202931


namespace total_students_is_88_l202_202293

def orchestra_students : Nat := 20
def band_students : Nat := 2 * orchestra_students
def choir_boys : Nat := 12
def choir_girls : Nat := 16
def choir_students : Nat := choir_boys + choir_girls

def total_students : Nat := orchestra_students + band_students + choir_students

theorem total_students_is_88 : total_students = 88 := by
  sorry

end total_students_is_88_l202_202293


namespace arithmetic_mean_of_scores_l202_202188

open Real

theorem arithmetic_mean_of_scores :
    let scores := [89, 92, 88, 95, 91, 93] in
    let total_sum := (scores.sum : ℝ) in
    let num_assignments := (6 : ℝ) in
    let mean := total_sum / num_assignments in
    mean = 91.3 :=
by
  let scores := [89, 92, 88, 95, 91, 93]
  let total_sum := (scores.sum : ℝ)
  let num_assignments := (6 : ℝ)
  let mean := total_sum / num_assignments
  have h1 : total_sum = 548 := by norm_num
  have h2 : mean = total_sum / num_assignments := rfl
  have h3 : mean = 91.3333 := by norm_num at *
  show mean = 91.3 from sorry

end arithmetic_mean_of_scores_l202_202188


namespace sum_of_first_10_common_elements_is_correct_l202_202454

-- Define arithmetic progression
def a (n : ℕ) : ℕ := 4 + 3 * n

-- Define geometric progression
def b (k : ℕ) : ℕ := 20 * (2 ^ k)

-- Define the sum of the first 10 common elements in both sequences
def sum_first_10_common_elements : ℕ := 13981000

-- Statement of the proof problem in Lean 4
theorem sum_of_first_10_common_elements_is_correct :
  ∑ i in (finset.range 10).image (λ k, b(2*k + 1)), id = sum_first_10_common_elements :=
by
  -- Proof omitted
  sorry

end sum_of_first_10_common_elements_is_correct_l202_202454


namespace diagonals_in_nine_sided_polygon_l202_202736

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l202_202736


namespace diagonals_in_nine_sided_polygon_l202_202792

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l202_202792


namespace number_of_diagonals_l202_202562

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l202_202562


namespace mariel_dogs_count_l202_202903

theorem mariel_dogs_count
  (num_dogs_other: Nat)
  (num_legs_tangled: Nat)
  (num_legs_per_dog: Nat)
  (num_legs_per_human: Nat)
  (num_dog_walkers: Nat)
  (num_dogs_mariel: Nat):
  num_dogs_other = 3 →
  num_legs_tangled = 36 →
  num_legs_per_dog = 4 →
  num_legs_per_human = 2 →
  num_dog_walkers = 2 →
  4*num_dogs_mariel + 4*num_dogs_other + 2*num_dog_walkers = num_legs_tangled →
  num_dogs_mariel = 5 :=
by 
  intros h_other h_tangled h_legs_dog h_legs_human h_walkers h_eq
  sorry

end mariel_dogs_count_l202_202903


namespace polynomial_non_real_roots_l202_202237

theorem polynomial_non_real_roots
  (n m : ℕ)
  (a : Fin (n + 1) → ℝ)
  (h_n : a n ≠ 0)
  (missing_2m : ∀ i, 0 < i ∧ i ≤ 2 * m → a (n - i) = 0) :
  (∃ i, a i ≠ 0 ∧ 2 * m ≤ n - i ∧ ∃ j, i < j ∧ a j ≠ 0 ∧ j - i = 2 * m + 1) →
    (if (a (n - i) * a (n - (i + 2 * m + 1)) < 0) then
      (∃ k, 2 * m ≤ k ∧ k ≤ n ∧ a k = 0) →
        (∀ i j, (2 * m + 2) = j - i →
          (a i ≠ 0 ∧ a j ≠ 0) →
            ∃ l, (l ≥ 2 * m ∧ l < 2 * m + 2)) else
      (∃ k, 2 * m ≤ k ∧ k ≤ n ∧ a k = 0) →
        (∀ i j, (2 * m + 2) = j - i →
          (a i ≠ 0 ∧ a j ≠ 0) →
            ∃ l, (l ≥ 2 * m + 2))) :=
  sorry

end polynomial_non_real_roots_l202_202237


namespace diagonals_in_nonagon_l202_202728

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l202_202728


namespace area_ratio_l202_202380

noncomputable def AreaOfTrapezoid (AD BC : ℝ) (R : ℝ) : ℝ :=
  let s_π := Real.pi
  let height1 := 2 -- One of the heights considered
  let height2 := 14 -- Another height considered
  (AD + BC) / 2 * height1  -- First case area
  -- Here we assume the area uses sine which is arc-related, but provide fixed coefficients for area representation

noncomputable def AreaOfRectangle (R : ℝ) : ℝ :=
  let d := 2 * R
  -- Using the equation for area discussed
  d * d / 2

theorem area_ratio (AD BC : ℝ) (R : ℝ) (hAD : AD = 16) (hBC : BC = 12) (hR : R = 10) :
  let area_trap := AreaOfTrapezoid AD BC R
  let area_rect := AreaOfRectangle R
  area_trap / area_rect = 1 / 2 ∨ area_trap / area_rect = 49 / 50 :=
by
  sorry

end area_ratio_l202_202380


namespace a_sub_b_eq_2_l202_202808

theorem a_sub_b_eq_2 (a b : ℝ)
  (h : (a - 5) ^ 2 + |b ^ 3 - 27| = 0) : a - b = 2 :=
by
  sorry

end a_sub_b_eq_2_l202_202808


namespace fib_inequality_l202_202341

def Fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => Fib n + Fib (n + 1)

theorem fib_inequality {n : ℕ} (h : 2 ≤ n) : Fib (n + 5) > 10 * Fib n :=
  sorry

end fib_inequality_l202_202341


namespace nine_sided_polygon_diagonals_l202_202584

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l202_202584


namespace parallel_B1C1_AD_l202_202283

theorem parallel_B1C1_AD (A B C D C1 B1 : Point) (h_cyclic: CyclicQuadrilateral A B C D) 
(h_perp_C: Perpendicular C (angle_bisector ABD) C1) 
(h_perp_B: Perpendicular B (angle_bisector ACD) B1) 
(h_line_C1: LiesOnLine C1 A B) 
(h_line_B1: LiesOnLine B1 D C): 
Parallel B1 C1 A D :=
by {
    sorry
}

end parallel_B1C1_AD_l202_202283


namespace Carrie_charged_phone_14_times_l202_202911

-- Define the driving distances per day
def driving_distances : List ℕ := [135, 124, 159, 189, 210, 156, 240]

-- Define the initial phone charging capacity and daily increment
def initial_capacity : ℕ := 106
def daily_increment : ℕ := 15

-- Define the weather reduction factors and the days they affect
def weather_reduction_days : List ℕ := [4, 7]
def reduction_factor : ℕ := 5

-- Define the stop intervals and the days they affect
def stop_interval_days : List ℕ := [2, 6]
def stop_interval : ℕ := 55

-- Define the main theorem to prove that Carrie charged her phone 14 times
theorem Carrie_charged_phone_14_times :
  ∃ total_charges : ℕ,
    total_charges = 14 ∧ 
    ( ∀ day (1 ≤ day ∧ day ≤ 7), 
      ∃ distance capacity stops, 
        distance ∈ driving_distances ∧
        capacity = initial_capacity + (daily_increment * (day - 1)) - 
                  if day ∈ weather_reduction_days then (initial_capacity + (daily_increment * (day - 1))) * reduction_factor / 100 else 0 ∧
        stops = if day ∈ stop_interval_days then (distance / stop_interval) else (distance / capacity) ∧
        total_charges = ∑ n in driving_distances, (ceil (n / capacity))) :=
sorry

end Carrie_charged_phone_14_times_l202_202911


namespace area_of_region_l202_202991

-- Define the condition: the equation of the region
def region_equation (x y : ℝ) : Prop := x^2 + y^2 + 10 * x - 4 * y + 9 = 0

-- State the theorem: the area of the region defined by the equation is 20π
theorem area_of_region : ∀ x y : ℝ, region_equation x y → ∃ A : ℝ, A = 20 * Real.pi :=
by sorry

end area_of_region_l202_202991


namespace minimum_M_value_l202_202214

def max (p q : ℝ) : ℝ := if p >= q then p else q

def M (x y : ℝ) : ℝ := max (abs (x^2 + y + 1)) (abs (y^2 - x + 1))

theorem minimum_M_value : ∀ (x y : ℝ), M x y >= (3 / 4) :=
by
  intros
  sorry

end minimum_M_value_l202_202214


namespace total_cost_of_hats_l202_202864

-- Definition of conditions
def weeks := 2
def days_per_week := 7
def cost_per_hat := 50

-- Definition of the number of hats
def num_hats := weeks * days_per_week

-- Statement of the problem
theorem total_cost_of_hats : num_hats * cost_per_hat = 700 := 
by sorry

end total_cost_of_hats_l202_202864


namespace additional_people_needed_35_l202_202356

-- Define the parameters of the problem
def total_days : ℕ := 50
def initial_people : ℕ := 70
def days_passed : ℕ := 25
def work_done : ℕ := 40 -- representing 40% as a percentage

-- Calculations for the remaining work and required workforce
def remaining_days := total_days - days_passed
def remaining_work := 100 - work_done -- representing 60% as a percentage

-- Proportional calculations to determine the additional number of people needed
def additional_people_needed (initial_people : ℕ) (work_done : ℕ) (remaining_work : ℕ)
  (days_passed : ℕ) (remaining_days : ℕ) : ℕ :=
  let work_per_day_initial := work_done / days_passed in
  let work_per_day_needed := remaining_work / remaining_days in
  let total_people_needed := (work_per_day_needed * initial_people) / work_per_day_initial in
  total_people_needed - initial_people

-- The theorem statement to be proved
theorem additional_people_needed_35 : additional_people_needed initial_people work_done remaining_work days_passed remaining_days = 35 :=
  sorry

end additional_people_needed_35_l202_202356


namespace man_speed_in_still_water_l202_202339

theorem man_speed_in_still_water
  (vm vs : ℝ)
  (h1 : vm + vs = 6)  -- effective speed downstream
  (h2 : vm - vs = 4)  -- effective speed upstream
  : vm = 5 := 
by
  sorry

end man_speed_in_still_water_l202_202339


namespace arithmetic_sequence_product_l202_202882

theorem arithmetic_sequence_product (b : ℕ → ℤ) (h1 : ∀ n, b (n + 1) = b n + d) 
  (h2 : b 5 * b 6 = 35) : b 4 * b 7 = 27 :=
sorry

end arithmetic_sequence_product_l202_202882


namespace number_of_incorrect_propositions_l202_202967

-- Definition of each proposition based on given conditions
def prop1 (f : ℝ → ℝ) :=
  (∀ x : ℝ, x ≤ 0 → f(x) ≤ f(x + 1)) ∧ (∀ x : ℝ, x > 0 → f(x - 1) ≥ f(x)) →
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁) ≥ f(x₂)

def prop2 (x y : ℝ) :=
  let p := (x ≠ 2 ∨ y ≠ 3)
  let q := (x + y ≠ 5)
  p ∧ ¬q

def prop3 : Prop :=
  ∀ (x : ℝ), x ∈ Icc (-3 : ℝ) 3 → ¬(f(x) = f(-x) ∨ f(x) = -f(-x))

def prop4 (m : ℝ) :=
  ¬(∃ x₀ : ℝ, x₀^2 + m * x₀ + 2 * m - 3 < 0) → (2 < m ∧ m < 6)

-- Main theorem derivation
theorem number_of_incorrect_propositions :
  let props := [prop1, prop2, prop3, prop4]
  let incorrects := [true, false, true, true] -- Manually indicated incorrect propositions
  props.filter (λ p, p) = incorrects.filter (λ b, b) → props.length = 4 ∧ incorrects.count true = 3 ∧ approx (filter (λ p : Prop, not p) props) = 3
:= sorry

end number_of_incorrect_propositions_l202_202967
