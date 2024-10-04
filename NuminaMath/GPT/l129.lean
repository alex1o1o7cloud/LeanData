import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Matrix.Determinant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Composition
import Mathlib.Data.Binomial
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Primes
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Order
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.List.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Basic
import Mathlib.Probability.Notation
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch

namespace find_triangle_value_l129_129852

variables (triangle q r : ℝ)
variables (h1 : triangle + q = 75) (h2 : triangle + q + r = 138) (h3 : r = q / 3)

theorem find_triangle_value : triangle = -114 :=
by
  sorry

end find_triangle_value_l129_129852


namespace range_f_minus_x_l129_129171

def f (x : ℝ) : ℝ := abs x

theorem range_f_minus_x : set.range (λ x : ℝ, f x - x) = set.Icc 0 8 :=
begin
  sorry
end

end range_f_minus_x_l129_129171


namespace no_sequence_exists_l129_129479

theorem no_sequence_exists (C : ℕ) (hC : C > 1) : 
  ¬ (∃ (a : ℕ → ℕ), (∀ i j, i ≠ j → a i ≠ a j) ∧ 
      ∀ k ≥ 1, a (k + 1) ^ k ∣ C ^ k * (∏ i in Finset.range (k + 1), a (i + 1))) :=
by
  sorry

end no_sequence_exists_l129_129479


namespace distance_between_parallel_lines_intersection_of_perpendicular_lines_l129_129763

variables {x y : ℝ} (m n : ℝ)

-- Definitions for the first problem
def parallel_line1 := 3 * x + 4 * y - 12 = 0
def parallel_line2 := m * x + 8 * y + 6 = 0

-- Definitions for the second problem
def perpendicular_line1 := 2 * x + y + 2 = 0
def perpendicular_line2 := n * x + 4 * y - 2 = 0

-- The proof problem: proving findings given the conditions
theorem distance_between_parallel_lines (h1 : parallel_line1) (h2 : parallel_line2) (hm : m = 6) :
  let line2 := 3 * x + 4 * y + 3 = 0 in 
  ∀ d, d = abs (3 + 12) / real.sqrt (3^2 + 4^2) → d = 3 := 
by
  intro h1 h2 _
  let line2 := 3 * x + 4 * y + 3 = 0
  intro d hd
  rw [hd]
  sorry

theorem intersection_of_perpendicular_lines (h1 : perpendicular_line1) (h2 : perpendicular_line2) (hn : n = -2) :
  let line2 := x - 2 * y + 1 = 0 in 
  ∀ x y, (2 * x + y + 2 = 0) ∧ (x - 2 * y + 1 = 0) → x = -1 ∧ y = 0 :=
by
  intro h1 h2 _
  let line2 := x - 2 * y + 1 = 0
  intro x' y' hxy
  sorry

end distance_between_parallel_lines_intersection_of_perpendicular_lines_l129_129763


namespace problem_l129_129960

noncomputable def k : ℝ := 2.9

theorem problem (k : ℝ) (hₖ : k > 1) 
    (h_sum : ∑' n, (7 * n + 2) / k^n = 20 / 3) : 
    k = 2.9 := 
sorry

end problem_l129_129960


namespace max_n_arithmetic_sequence_l129_129459

open List

theorem max_n_arithmetic_sequence :
  ∃ (a d : ℝ) (n : ℕ), n ≥ 3 ∧ 
  (aₙ : ℕ → ℝ) (∀ i, aₙ i = a - i * d) ∧
  ∑ i in range n, abs (aₙ i) = 507 ∧
  ∑ i in range n, abs (aₙ i + 1) = 507 ∧
  ∑ i in range n, abs (aₙ i - 2) = 507 ∧
  n = 26 :=
sorry

end max_n_arithmetic_sequence_l129_129459


namespace proof_problem_l129_129203

def poem := "金人捧露盘"
def author := "Zeng Di"
def context := "Zeng Di's mission to the Jin Dynasty's capital in the spring of the Gengyin year (AD 1170)"

def reasons_for_sentimentality (poem : String) (author : String) (context : String) : Prop :=
  (poem = "金人捧露盘" ∧ author = "Zeng Di" ∧ context = "Zeng Di's mission to the Jin Dynasty's capital in the spring of the Gengyin year (AD 1170)")
  → "The reasons for the author's sentimentality include the humiliating mission, the decline of his homeland, and his own aging."

def artistic_techniques_used (poem : String) (author : String) (context : String) : Prop :=
  (poem = "金人捧露盘" ∧ author = "Zeng Di" ∧ context = "Zeng Di's mission to the Jin Dynasty's capital in the spring of the Gengyin year (AD 1170)")
  → "The artistic technique used in the second stanza is using scenery to express emotions, effectively conveying the author's deep and sorrowful inner feelings."

theorem proof_problem : reasons_for_sentimentality poem author context ∧ artistic_techniques_used poem author context :=
by 
  sorry

end proof_problem_l129_129203


namespace will_gave_3_boxes_l129_129753

theorem will_gave_3_boxes (boxes_bought : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ)
  (h1 : boxes_bought = 7) 
  (h2 : pieces_per_box = 4) 
  (h3 : pieces_left = 16) : 
  (boxes_bought * pieces_per_box - pieces_left) / pieces_per_box = 3 := 
by {
  rw [h1, h2, h3],
  norm_num,
}

end will_gave_3_boxes_l129_129753


namespace camera_lens_distance_l129_129771

theorem camera_lens_distance (f u : ℝ) (h_fu : f ≠ u) (h_f : f ≠ 0) (h_u : u ≠ 0) :
  (∃ v : ℝ, (1 / f) = (1 / u) + (1 / v) ∧ v = (f * u) / (u - f)) :=
by {
  sorry
}

end camera_lens_distance_l129_129771


namespace train_length_l129_129334

theorem train_length (speed_kmph : ℕ) (time_sec : ℕ) (speed_mps : ℕ) (length_train : ℕ) :
  speed_kmph = 360 →
  time_sec = 5 →
  speed_mps = 100 →
  length_train = speed_mps * time_sec →
  length_train = 500 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end train_length_l129_129334


namespace complex_product_polar_form_l129_129394

-- Define complex numbers in polar form
def complex_polar (r : ℝ) (θ : ℝ) : Complex := Complex.mkPolar r θ

-- Define the specific complex numbers given in the problem
def z1 : Complex := complex_polar 5 (30 * Real.pi / 180)
def z2 : Complex := complex_polar (-3) (45 * Real.pi / 180)

-- Define the expected result as a complex number in polar form
def z_product_expected : Complex := complex_polar 15 (255 * Real.pi / 180)

-- State the theorem to be proven
theorem complex_product_polar_form :
  (z1 * z2) = z_product_expected :=
sorry

end complex_product_polar_form_l129_129394


namespace shaded_region_area_l129_129145

theorem shaded_region_area (PQ RS : set (ℝ × ℝ)) (O : ℝ × ℝ)
  (h₁ : PQ ≠ RS) (h₂ : ∀ P Q R S : ℝ × ℝ, (P, Q) ∈ PQ → (R, S) ∈ RS → PQ ∩ RS = {O})
  (h₃ : ∀ P Q R S : ℝ × ℝ, (P, Q) ∈ PQ → (R, S) ∈ RS → angle O P Q = π / 4)
  (h₄ : radius_of_circle = 6) :
  area_of_shaded_region PQ RS = 72 + 9 * π := sorry

end shaded_region_area_l129_129145


namespace triangle_inequality_satisfied_for_n_six_l129_129029

theorem triangle_inequality_satisfied_for_n_six :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ a + c > b ∧ b + c > a) := sorry

end triangle_inequality_satisfied_for_n_six_l129_129029


namespace problem1_partial_derivatives_problem2_partial_derivatives_l129_129821

-- Proof Problem 1
theorem problem1_partial_derivatives (x y z : ℝ) (hz : x^2 + y^2 + z^2 - z = 0) :
  has_deriv_at (λ x, z) (-2 * x / (2 * z - 1)) x ∧ has_deriv_at (λ y, z) (-2 * y / (2 * z - 1)) y :=
sorry

-- Proof Problem 2
theorem problem2_partial_derivatives (a b c k x y z : ℝ)
  (hz : a * x + b * y - c * z = k * real.cos (a * x + b * y - c * z)) :
  has_deriv_at (λ x, z) (a / c) x ∧ has_deriv_at (λ y, z) (b / c) y :=
sorry

end problem1_partial_derivatives_problem2_partial_derivatives_l129_129821


namespace integer_roots_of_quadratic_l129_129507

theorem integer_roots_of_quadratic (a : ℚ) :
  (∃ x₁ x₂ : ℤ, 
    a * x₁ * x₁ + (a + 1) * x₁ + (a - 1) = 0 ∧ 
    a * x₂ * x₂ + (a + 1) * x₂ + (a - 1) = 0 ∧ 
    x₁ ≠ x₂) ↔ 
      a = 0 ∨ a = -1/7 ∨ a = 1 :=
by
  sorry

end integer_roots_of_quadratic_l129_129507


namespace hexagon_area_l129_129716

theorem hexagon_area :
  let points := [(0, 0), (2, 4), (5, 4), (7, 0), (5, -4), (2, -4), (0, 0)]
  ∃ (area : ℝ), area = 52 := by
  sorry

end hexagon_area_l129_129716


namespace opposite_of_neg_three_fourths_l129_129641

theorem opposite_of_neg_three_fourths : ∃ x : ℚ, -3 / 4 + x = 0 ∧ x = 3 / 4 :=
by
  use 3 / 4
  split
  . norm_num
  . refl

end opposite_of_neg_three_fourths_l129_129641


namespace mary_seashells_l129_129972

theorem mary_seashells (M : ℕ) (Keith_seashells : ℕ = 5) (Total_seashells : M + Keith_seashells = 7) : M = 2 :=
by {
  sorry
}

end mary_seashells_l129_129972


namespace volume_intersection_l129_129726

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

def region1 (x y z : ℝ) : Prop := abs x + abs y + abs z ≤ 1
def region2 (x y z : ℝ) : Prop := abs x + abs y + abs (z - 2) ≤ 1

theorem volume_intersection : 
  (volume {p : ℝ × ℝ × ℝ | region1 p.1 p.2 p.3 ∧ region2 p.1 p.2 p.3}) = (1 / 12 : ℝ) :=
by
  sorry

end volume_intersection_l129_129726


namespace sum_of_integral_c_l129_129826

theorem sum_of_integral_c :
  let discriminant (a b c : ℤ) := b * b - 4 * a * c
  ∃ (valid_c : List ℤ),
    (∀ c ∈ valid_c, c ≤ 30 ∧ ∃ k : ℤ, discriminant 1 (-9) (c) = k * k ∧ k > 0) ∧
    valid_c.sum = 32 := 
by
  sorry

end sum_of_integral_c_l129_129826


namespace volume_of_intersection_l129_129735

def condition1 (x y z : ℝ) : Prop := |x| + |y| + |z| ≤ 1
def condition2 (x y z : ℝ) : Prop := |x| + |y| + |z - 2| ≤ 1
def in_intersection (x y z : ℝ) : Prop := condition1 x y z ∧ condition2 x y z

theorem volume_of_intersection : 
  (∫ x y z in { p : ℝ × ℝ × ℝ | in_intersection p.1 p.2 p.3 }, 1) = 1/12 := 
by
  sorry

end volume_of_intersection_l129_129735


namespace obtuse_angles_in_second_quadrant_l129_129324

theorem obtuse_angles_in_second_quadrant
  (θ : ℝ) 
  (is_obtuse : θ > 90 ∧ θ < 180) :
  90 < θ ∧ θ < 180 :=
by sorry

end obtuse_angles_in_second_quadrant_l129_129324


namespace percentage_books_returned_l129_129366

theorem percentage_books_returned (initial_books : ℕ) (remaining_books : ℕ) (loaned_books : ℝ) : 
  initial_books = 75 →
  remaining_books = 54 →
  loaned_books = 60.00000000000001 →
  let books_missing := initial_books - remaining_books in
  let books_returned := loaned_books - books_missing in
  let percentage_returned := (books_returned / loaned_books) * 100 in
  percentage_returned = 65 :=
by
  intros h_initial h_remaining h_loaned
  let books_missing := initial_books - remaining_books
  let books_returned := loaned_books - books_missing
  let percentage_returned := (books_returned / loaned_books) * 100
  have h1 : books_missing = 21 := by norm_num [h_initial, h_remaining]
  have h2 : books_returned = 39.00000000000001 := by norm_num [h_loaned, h1]
  have h3 : percentage_returned = 65 := by norm_num [h2, h_loaned]
  exact h3

end percentage_books_returned_l129_129366


namespace gcd_1680_1683_l129_129046

theorem gcd_1680_1683 :
  ∀ (n : ℕ), n = 1683 →
  (∀ m, (m = 5 ∨ m = 67 ∨ m = 8) → n % m = 3) →
  (∃ d, d > 1 ∧ d ∣ 1683 ∧ d = Nat.gcd 1680 n ∧ Nat.gcd 1680 n = 3) :=
by
  sorry

end gcd_1680_1683_l129_129046


namespace incorrect_guess_l129_129696

-- Define the conditions
def bears : ℕ := 1000

inductive Color
| White
| Brown
| Black

constant bear_color : ℕ → Color -- The color of the bear at each position

axiom condition : ∀ n : ℕ, n < bears - 2 → 
  ∃ i j k, (i, j, k ∈ {Color.White, Color.Brown, Color.Black}) ∧ 
  (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ 
  (bear_color n = i ∧ bear_color (n+1) = j ∧ bear_color (n+2) = k) 

constants (g1 : bear_color 2 = Color.White)
          (g2 : bear_color 20 = Color.Brown)
          (g3 : bear_color 400 = Color.Black)
          (g4 : bear_color 600 = Color.Brown)
          (g5 : bear_color 800 = Color.White)

-- The proof problem
theorem incorrect_guess : bear_color 20 ≠ Color.Brown :=
by sorry

end incorrect_guess_l129_129696


namespace triangle_inequality_condition_l129_129041

theorem triangle_inequality_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (ineq : 6 * (a * b + b * c + c * a) > 5 * (a ^ 2 + b ^ 2 + c ^ 2)) : 
  (a < b + c ∧ b < a + c ∧ c < a + b) :=
sorry

end triangle_inequality_condition_l129_129041


namespace even_function_implies_a_zero_l129_129910

theorem even_function_implies_a_zero (a : ℝ) : 
  (∀ x : ℝ, (λ x, x^2 - |x + a|) (-x) = (λ x, x^2 - |x + a|) (x)) → a = 0 :=
by
  sorry

end even_function_implies_a_zero_l129_129910


namespace sum_geometric_sequence_l129_129881

theorem sum_geometric_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 2) (h2 : ∀ n, 2 * a n - 2 = S n) : 
  S n = 2^(n+1) - 2 :=
sorry

end sum_geometric_sequence_l129_129881


namespace intersection_M_N_l129_129104

def M : Set ℝ := { x | x^2 - x - 2 = 0 }
def N : Set ℝ := { -1, 0 }

theorem intersection_M_N : M ∩ N = {-1} :=
by
  sorry

end intersection_M_N_l129_129104


namespace q1_q2_l129_129090

def f (a x : ℝ) : ℝ := Real.log x + 2 * a / x

theorem q1 (a : ℝ) (h : ∀ x, x ∈ set.Ici 4 → 0 ≤ (f a)' x) : a ≤ 2 :=
by
  sorry

theorem q2 (a : ℝ) (h : ∀ x, x ∈ set.Icc 1 Real.exp 1 → ∃ f_min, f_min = 3 ∧ ∀ y, y ∈ set.Icc 1 Real.exp 1 → f a y ≥ f_min) : a = Real.exp 1 :=
by
  sorry

end q1_q2_l129_129090


namespace ed_money_left_l129_129426

theorem ed_money_left
  (cost_per_hour_night : ℝ := 1.5)
  (cost_per_hour_morning : ℝ := 2)
  (initial_money : ℝ := 80)
  (hours_night : ℝ := 6)
  (hours_morning : ℝ := 4) :
  initial_money - (cost_per_hour_night * hours_night + cost_per_hour_morning * hours_morning) = 63 := 
  by
  sorry

end ed_money_left_l129_129426


namespace yangyang_departure_time_l129_129754

noncomputable def departure_time : Nat := 373 -- 6:13 in minutes from midnight (6 * 60 + 13)

theorem yangyang_departure_time :
  let arrival_at_60_mpm := 413 -- 6:53 in minutes from midnight
  let arrival_at_75_mpm := 405 -- 6:45 in minutes from midnight
  let difference := arrival_at_60_mpm - arrival_at_75_mpm -- time difference
  let x := 40 -- time taken to walk to school at 60 meters per minute
  departure_time = arrival_at_60_mpm - x :=
by
  -- Definitions
  let arrival_at_60_mpm := 413
  let arrival_at_75_mpm := 405
  let difference := 8
  let x := 40
  have h : departure_time = (413 - 40) := rfl
  sorry

end yangyang_departure_time_l129_129754


namespace least_number_of_cans_l129_129331

theorem least_number_of_cans (maaza pepsi sprite : ℕ) (h_maaza : maaza = 80) (h_pepsi : pepsi = 144) (h_sprite : sprite = 368) :
  ∃ n, n = 37 := sorry

end least_number_of_cans_l129_129331


namespace part_length_of_scale_l129_129757

theorem part_length_of_scale (total_length_ft : ℕ) (extra_inches : ℕ) (parts : ℕ) :
  total_length_ft = 10 → extra_inches = 5 → parts = 5 →
  let total_length_in_inches := (total_length_ft * 12) + extra_inches in
  total_length_in_inches / parts = 25 :=
by
  intros h1 h2 h3
  let total_length_in_inches := (total_length_ft * 12) + extra_inches
  sorry

end part_length_of_scale_l129_129757


namespace find_m_n_l129_129840

noncomputable def f (z : ℂ) : ℂ := z^2 - 19 * z

def isRightTriangle (A B C : ℂ) : Prop :=
  (B - A) * conj (C - B) = 0

theorem find_m_n (m n : ℕ) (h₁ : m > 0) (h₂ : n > 0)
  (z := (m : ℂ) + real.sqrt n + 11 * complex.I)
  (h₃ : isRightTriangle z (f z) (f (f z))) :
  m + n = 230 := by
  sorry

end find_m_n_l129_129840


namespace Joel_non_hot_peppers_l129_129161

constant Sunday_peppers : ℕ := 7
constant Monday_peppers : ℕ := 12
constant Tuesday_peppers : ℕ := 14
constant Wednesday_peppers : ℕ := 12
constant Thursday_peppers : ℕ := 5
constant Friday_peppers : ℕ := 18
constant Saturday_peppers : ℕ := 12

constant hot_pepper_percentage : ℝ := 0.20

noncomputable def total_peppers : ℕ :=
  Sunday_peppers + Monday_peppers + Tuesday_peppers +
  Wednesday_peppers + Thursday_peppers + Friday_peppers + Saturday_peppers

noncomputable def non_hot_peppers : ℕ :=
  (1 - hot_pepper_percentage) * total_peppers

theorem Joel_non_hot_peppers :
  non_hot_peppers = 64 :=
by
  sorry

end Joel_non_hot_peppers_l129_129161


namespace exists_initial_segment_of_power_of_2_l129_129200

theorem exists_initial_segment_of_power_of_2 (m : ℕ) : ∃ n : ℕ, ∃ k : ℕ, k ≥ m ∧ 2^n = 10^k * m ∨ 2^n = 10^k * (m+1) := 
by
  sorry

end exists_initial_segment_of_power_of_2_l129_129200


namespace find_2017th_term_in_odd_digit_sequence_l129_129413

theorem find_2017th_term_in_odd_digit_sequence :
  ∃ n : ℕ, (sequence_position n = 2017) ∧ (n = 34441) :=
sorry

end find_2017th_term_in_odd_digit_sequence_l129_129413


namespace rectangle_area_inscribed_circle_l129_129768

theorem rectangle_area_inscribed_circle 
  (radius : ℝ) (width len : ℝ) 
  (h_radius : radius = 5) 
  (h_width : width = 2 * radius) 
  (h_len_ratio : len = 3 * width) 
  : width * len = 300 := 
by
  sorry

end rectangle_area_inscribed_circle_l129_129768


namespace reflection_points_concyclic_l129_129133

-- Definitions of geometric entities
variables (A B C D E P Q R S : Type) 
-- Conditions
variables [quadrilateral : convex_quad A B C D] 
variables [AC : line_segment A C]
variables [BD : line_segment B D]
variables [perpendicular_intersection : intersects_perpendicularly AC BD at E]
-- Points of reflection
variables (P : reflection_point E over AB)
variables (Q : reflection_point E over BC)
variables (R : reflection_point E over CD)
variables (S : reflection_point E over DA)

-- Proven statement
theorem reflection_points_concyclic : concyclic P Q R S :=
sorry

end reflection_points_concyclic_l129_129133


namespace average_allowance_rest_students_l129_129349

theorem average_allowance_rest_students 
  (n_students : ℕ) (n_allowance_per_day : ℕ) (n_total_allowance : ℤ) 
  (fraction_students : ℚ) (average_allowance1 : ℚ) 
  (h1 : n_students = 60)
  (h2 : n_allowance_per_day = 6)
  (h3 : n_total_allowance = 320)
  (h4 : fraction_students = 2/3) :
  let remaining_students := n_students * (1 - fraction_students)
      total_remaining_allowance := n_total_allowance - n_students * fraction_students * n_allowance_per_day in
  average_allowance remaining_students total_remaining_allowance = 4 :=
by
  sorry

end average_allowance_rest_students_l129_129349


namespace problem_1_problem_2_problem_3_l129_129966

theorem problem_1 (x y : ℝ) : x^2 + y^2 + x * y + x + y ≥ -1 / 3 := 
by sorry

theorem problem_2 (x y z : ℝ) : x^2 + y^2 + z^2 + x * y + y * z + z * x + x + y + z ≥ -3 / 8 := 
by sorry

theorem problem_3 (x y z r : ℝ) : x^2 + y^2 + z^2 + r^2 + x * y + x * z + x * r + y * z + y * r + z * r + x + y + z + r ≥ -2 / 5 := 
by sorry

end problem_1_problem_2_problem_3_l129_129966


namespace second_month_sale_l129_129359

theorem second_month_sale 
  (sale_1st: ℕ) (sale_3rd: ℕ) (sale_4th: ℕ) (sale_5th: ℕ) (sale_6th: ℕ) (avg_sale: ℕ)
  (h1: sale_1st = 5266) (h3: sale_3rd = 5864)
  (h4: sale_4th = 6122) (h5: sale_5th = 6588)
  (h6: sale_6th = 4916) (h_avg: avg_sale = 5750) :
  ∃ sale_2nd, (sale_1st + sale_2nd + sale_3rd + sale_4th + sale_5th + sale_6th) / 6 = avg_sale :=
by
  sorry

end second_month_sale_l129_129359


namespace draw_line_through_A_l129_129710

/-- Given two intersecting circles with centers O and O1 at point A, 
    we can draw a line through A such that the segment BC intercepted 
    on it by the circles O and O1 has a length equal to a given length a, 
    provided the construction of the right triangle is possible, that is 
    unless a/2 > dist O O1. In the case of a/2 = dist O O1, the solution is unique. -/
theorem draw_line_through_A (O O1 A : Point) (r1 r2: ℝ) (h_inter : intersect_circles O r1 O1 r2 A) 
  (a : ℝ) (h_pos : 0 < a) :
  ∃ (line : Line), intercepted_by_circles O O1 line = a :=
by
  sorry

end draw_line_through_A_l129_129710


namespace decreasing_function_range_l129_129489

theorem decreasing_function_range (a : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x ≤ 1, f x = (2 * a - 1) * x + 3 * a)
  (h₂ : ∀ x > 1, f x = log a x)
  (h₃ : ∀ x y, x ≤ y → f x ≥ f y) :
  ∃ a, a ∈ Ico (1 / 5) (1 / 2) :=
by
  have lower_bound := 2 * a - 1 < 0 -- i.e., 2 * a < 1, so a < 1 / 2
  have range_a := 0 < a < 1 
  have continuity := (2 * a - 1) * 1 + 3 * a ≥ log a 1 -- i.e., (2 * a - 1) + 3 * a ≥ 0 as log a 1 = 0 for any a
  sorry

end decreasing_function_range_l129_129489


namespace triangle_DEP_is_isosceles_l129_129586

-- Define the main theorem
theorem triangle_DEP_is_isosceles (A B C P D E : Point)
  (h_triangle_ABC : is_right_isosceles_triangle A B C C)
  (h_P_on_AB : on_segment P A B)
  (h_CP_intersects_circumcircle_ABC_at_D : intersects_line_circle_except_C (line_through C P) (circumcircle A B C) D)
  (h_tangent_to_circumcircle_at_D_intersects_AB_at_E : tangent_intersects_line (circumcircle A B C) D (line_through A B) E) :
  is_isosceles_triangle DEP E :=
sorry

end triangle_DEP_is_isosceles_l129_129586


namespace boxes_in_case_correct_l129_129486

-- Given conditions
def total_boxes : Nat := 2
def blocks_per_box : Nat := 6
def total_blocks : Nat := 12

-- Define the number of boxes in a case as a result of total_blocks divided by blocks_per_box
def boxes_in_case : Nat := total_blocks / blocks_per_box

-- Prove the number of boxes in a case is 2
theorem boxes_in_case_correct : boxes_in_case = 2 := by
  -- Place the actual proof here
  sorry

end boxes_in_case_correct_l129_129486


namespace fraction_of_cracked_pots_is_2_over_5_l129_129802

-- Definitions for the problem conditions
def total_pots : ℕ := 80
def price_per_pot : ℕ := 40
def total_revenue : ℕ := 1920

-- Statement to prove the fraction of cracked pots
theorem fraction_of_cracked_pots_is_2_over_5 
  (C : ℕ) 
  (h1 : (total_pots - C) * price_per_pot = total_revenue) : 
  C / total_pots = 2 / 5 :=
by
  sorry

end fraction_of_cracked_pots_is_2_over_5_l129_129802


namespace count_brown_and_orange_cubes_l129_129330

-- Definition of the problem
def cube_painted_and_cut : Prop :=
  ∃ (n : ℕ) (brown_faces orange_faces : set (fin 3)) (small_cubes : fin n → fin n → fin n → Prop),
    n = 4 ∧
    brown_faces = {0, 1} ∧
    orange_faces = {2, 3} ∧
    ∀ (x y z : fin 4),
      (small_cubes x y z ↔
      ((x = 0 ∨ x = 3) ∧ (y = 0 ∨ y = 3) ∧ (z = 0 ∨ z = 3)))

theorem count_brown_and_orange_cubes : ∃ (n : ℕ), n = 16 :=
by
  exists 16
  sorry

end count_brown_and_orange_cubes_l129_129330


namespace equilateral_triangle_area_decrease_l129_129381

theorem equilateral_triangle_area_decrease :
  let original_area : ℝ := 100 * Real.sqrt 3
  let side_length_s := 20
  let decreased_side_length := side_length_s - 6
  let new_area := (decreased_side_length * decreased_side_length * Real.sqrt 3) / 4
  let decrease_in_area := original_area - new_area
  decrease_in_area = 51 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_decrease_l129_129381


namespace trig_identity_simplification_l129_129016

theorem trig_identity_simplification : 
  sin (71 * pi / 180) * cos (26 * pi / 180) - sin (19 * pi / 180) * sin (26 * pi / 180) = sqrt 2 / 2 :=
by sorry

end trig_identity_simplification_l129_129016


namespace complement_of_M_in_U_l129_129883

theorem complement_of_M_in_U :
  let U := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
  let M := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
  ∀ x, x ∉ M → x ∈ U → x ∈ {x : ℝ | 1 < x ∧ x ≤ 3} :=
by
  let U : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
  let M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
  have h : ∀ x, x ∉ M → x ∈ U → x ∈ {x | 1 < x ∧ x ≤ 3}
  { sorry }
  exact h

end complement_of_M_in_U_l129_129883


namespace carolyns_total_time_l129_129402

def stiching_time (stitches : ℕ) (speed : ℕ) : ℕ :=
  stitches / speed

def total_time_no_breaks (n_f n_u n_g : ℕ) : ℕ :=
  n_f * stiching_time 60 4 + n_u * stiching_time 180 5 + n_g * stiching_time 800 3

def total_breaks (total_time : ℕ) : ℕ :=
  total_time / 30

def total_time_with_breaks (total_time : ℕ) : ℕ :=
  total_time + total_breaks total_time * 5

theorem carolyns_total_time :
  let total_time := total_time_no_breaks 50 3 1 in
  total_time_with_breaks total_time = 1310 :=
by
  sorry

end carolyns_total_time_l129_129402


namespace sequence_sum_remainder_l129_129320

theorem sequence_sum_remainder :
  let seq_sum (n : ℕ) : ℕ := (6 * n - 3)
  let total_terms := 52
  (finset.sum (finset.range total_terms) seq_sum) % 8 = 4 :=
by
  let seq_sum (n : ℕ) : ℕ := (6 * n - 3)
  let total_terms := 52
  sorry

end sequence_sum_remainder_l129_129320


namespace opposite_of_neg_three_quarters_l129_129639

theorem opposite_of_neg_three_quarters : ∃ (b : ℚ), (-3/4) + b = 0 ∧ b = 3/4 := by
  use 3/4
  split
  · exact add_right_neg (-3/4)
  · rfl
  sorry

end opposite_of_neg_three_quarters_l129_129639


namespace counterexample_exists_l129_129010

-- Definitions for the problem conditions
def is_power_of_prime (n : ℕ) : Prop :=
  ∃ (p k : ℕ), Prime p ∧ k > 0 ∧ n = p ^ k

def is_not_prime (n : ℕ) : Prop :=
  ¬ Prime n

-- Translate the problem into a proof statement
theorem counterexample_exists : ∃ (n : ℕ), 
  (n ∈ {16, 14, 25, 32, 49}) ∧ is_power_of_prime n ∧ Prime (n - 2) :=
by
  sorry

end counterexample_exists_l129_129010


namespace sequence_value_a6_l129_129563

noncomputable def a : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := a (n + 1) + a n

theorem sequence_value_a6 : a 5 = 8 :=
sorry

end sequence_value_a6_l129_129563


namespace pies_from_apples_l129_129342

theorem pies_from_apples 
  (initial_apples : ℕ) (handed_out_apples : ℕ) (apples_per_pie : ℕ) 
  (remaining_apples := initial_apples - handed_out_apples) 
  (pies := remaining_apples / apples_per_pie) 
  (h1 : initial_apples = 75) 
  (h2 : handed_out_apples = 19) 
  (h3 : apples_per_pie = 8) : 
  pies = 7 :=
by
  rw [h1, h2, h3]
  sorry

end pies_from_apples_l129_129342


namespace jellybean_total_l129_129400

theorem jellybean_total :
  let Caleb_jellybeans := 3 * 12 in
  let Sophie_jellybeans := Caleb_jellybeans / 2 in
  Caleb_jellybeans + Sophie_jellybeans = 54 := by
  sorry

end jellybean_total_l129_129400


namespace area_enclosed_by_region_l129_129307

theorem area_enclosed_by_region : ∀ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) → (π * (4 ^ 2) = 16 * π) :=
by
  intro x y h
  sorry

end area_enclosed_by_region_l129_129307


namespace car_y_travel_distance_l129_129001

theorem car_y_travel_distance :
  (∀ (time_x : ℝ) (distance_x : ℝ) (time_y : ℝ) (speed_multiplier : ℝ), 
    time_x = 3 → distance_x = 150 → speed_multiplier = 3 → time_y = 4 → 
    let speed_x := distance_x / time_x in
    let speed_y := speed_multiplier * speed_x in
    let distance_y := speed_y * time_y in
    distance_y = 600) :=
by
  intros time_x distance_x time_y speed_multiplier h_time_x h_distance_x 
        h_speed_multiplier h_time_y;
  rw [h_time_x, h_distance_x, h_speed_multiplier, h_time_y];
  have speed_x := distance_x / time_x;
  rw [h_time_x] at speed_x;
  have speed_y := speed_multiplier * speed_x;
  rw [h_speed_multiplier] at speed_y;
  have distance_y := speed_y * time_y;
  rw [h_time_y] at distance_y;
  simp [speed_x, speed_y, distance_y];
  norm_num

end car_y_travel_distance_l129_129001


namespace ed_money_left_after_hotel_stay_l129_129428

theorem ed_money_left_after_hotel_stay 
  (night_rate : ℝ) (morning_rate : ℝ) 
  (initial_money : ℝ) (hours_night : ℕ) (hours_morning : ℕ) 
  (remaining_money : ℝ) : 
  night_rate = 1.50 → morning_rate = 2.00 → initial_money = 80 → 
  hours_night = 6 → hours_morning = 4 → 
  remaining_money = 63 :=
by
  intros h1 h2 h3 h4 h5
  let cost_night := night_rate * hours_night
  let cost_morning := morning_rate * hours_morning
  let total_cost := cost_night + cost_morning
  let money_left := initial_money - total_cost
  sorry

end ed_money_left_after_hotel_stay_l129_129428


namespace ratio_of_areas_l129_129775

theorem ratio_of_areas (A_large A_cut A_trapezoid : ℝ)
  (h1: A_large = (12^2 * real.sqrt 3) / 4)
  (h2: A_cut = (3^2 * real.sqrt 3) / 4)
  (h3: A_trapezoid = A_large - A_cut) :
  (A_cut / A_trapezoid) = 1 / 15 :=
by
  sorry

end ratio_of_areas_l129_129775


namespace _l129_129089

variables {a b : ℝ} (h₁ : a > b) (h₂ : b > 0) (h₃ : a > 0) (e : ℝ)

-- Ellipse definition
def ellipse (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Eccentricity condition
def eccentricity : Prop :=
  e = 1 / 2

-- Point F as a focus of the ellipse
def focus (F : ℝ × ℝ) : Prop :=
  F = (1, 0)

-- Point A on the ellipse
def point_A : ℝ × ℝ :=
  (-2, 0)

-- Distance AF condition
def af_distance (F : ℝ × ℝ) : Prop :=
  |F.1 + 2| = 3

-- Define points
structure points :=
  (O : ℝ × ℝ)
  (P : ℝ × ℝ)
  (M : ℝ × ℝ)
  
variables (p : points)

-- Midpoint definition
def midpoint (P : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 - 2) / 2, P.2 / 2)

lemma prove_angles_equal (O D E F : ℝ × ℝ) : Prop :=
  ∠ O D F = ∠ O E F

-- Main theorem statement
def main_theorem :=
  prove_angles_equal (0, 0) (4, ?) (4, ?) (1, 0)

#check main_theorem

end _l129_129089


namespace find_m_l129_129880

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (4, 2)
def vector_c (m : ℝ) : ℝ × ℝ := (m * vector_a.1 + vector_b.1, m * vector_a.2 + vector_b.2)

def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

def norm (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

lemma angle_condition (m : ℝ) :
  dot_product (vector_c m) vector_a / norm vector_a = dot_product (vector_c m) vector_b / norm vector_b :=
  sorry

theorem find_m : ∃ (m : ℝ), 
  (∀ a b c : ℝ × ℝ, 
   a = vector_a → b = vector_b → c = vector_c m →  
   angle_condition m) ∧ m = 2 :=
begin
  use 2,
  split,
  { intros a b c ha hb hc,
    rw [ha, hb, hc],
    exact angle_condition 2,
  },
  { refl},
end

end find_m_l129_129880


namespace wrong_guess_is_20_l129_129664

-- Define the colors
inductive Color
| white
| brown
| black

-- Assume we have a sequence of 1000 bears
def bears : fin 1000 → Color := sorry

-- Hypotheses
axiom colors_per_three : ∀ (i : fin 998), 
  ({bears i, bears (i + 1), bears (i + 2)} = {Color.white, Color.brown, Color.black} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.black, Color.white, Color.brown} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.brown, Color.black, Color.white})

axiom exactly_one_wrong : 
  (bears 1 = Color.white ∧ bears 19 ≠ Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 ≠ Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 ≠ Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 ≠ Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 ≠ Color.white)

-- Define the theorem to prove
theorem wrong_guess_is_20 : 
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) →
  ¬(bears 19 = Color.brown) := 
sorry

end wrong_guess_is_20_l129_129664


namespace solution_set_fraction_inequality_l129_129269

theorem solution_set_fraction_inequality : 
  { x : ℝ | 0 < x ∧ x < 1/3 } = { x : ℝ | 1/x > 3 } :=
by
  sorry

end solution_set_fraction_inequality_l129_129269


namespace identifyIncorrectGuess_l129_129680

-- Define the colors of the bears
inductive BearColor
| white
| brown
| black

-- Conditions as defined in the problem statement
def isValidBearRow (bears : Fin 1000 → BearColor) : Prop :=
  ∀ (i : Fin 998), 
    (bears i = BearColor.white ∨ bears i = BearColor.brown ∨ bears i = BearColor.black) ∧
    (bears ⟨i + 1, by linarith⟩ = BearColor.white ∨ bears ⟨i + 1, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 1, by linarith⟩ = BearColor.black) ∧
    (bears ⟨i + 2, by linarith⟩ = BearColor.white ∨ bears ⟨i + 2, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 2, by linarith⟩ = BearColor.black)

-- Iskander's guesses
def iskanderGuesses (bears : Fin 1000 → BearColor) : Prop :=
  bears 1 = BearColor.white ∧
  bears 19 = BearColor.brown ∧
  bears 399 = BearColor.black ∧
  bears 599 = BearColor.brown ∧
  bears 799 = BearColor.white

-- Exactly one guess is incorrect
def oneIncorrectGuess (bears : Fin 1000 → BearColor) : Prop :=
  ∃ (idx : Fin 5), 
    ¬iskanderGuesses bears ∧
    ∀ (j : Fin 5), (j ≠ idx → (bearGuessesIdx j bears = true))

-- The proof problem
theorem identifyIncorrectGuess (bears : Fin 1000 → BearColor) :
  isValidBearRow bears → iskanderGuesses bears → oneIncorrectGuess bears := sorry

end identifyIncorrectGuess_l129_129680


namespace sin_theta_correct_l129_129589

noncomputable def sin_theta {V : Type*} [InnerProductSpace ℝ V] (a b c : V) (ha : ∥a∥ = 2) (hb : ∥b∥ = 7) (hc : ∥c∥ = 4)
  (h : a × (a × b) = c) : ℝ :=
begin
  have h1 : ∥a × b∥ = 2 * 7 * sin (Real.angle a b),
  {
    sorry
  },
  have h2 : 4 = 14 * sin (Real.angle a b),
  {
    sorry
  },
  exact (4 / 14),
end

theorem sin_theta_correct {V : Type*} [InnerProductSpace ℝ V] (a b c : V) (ha : ∥a∥ = 2) (hb : ∥b∥ = 7) (hc : ∥c∥ = 4)
  (h : a × (a × b) = c) : sin_theta a b c ha hb hc h = 2 / 7 :=
begin
  sorry
end

end sin_theta_correct_l129_129589


namespace geometric_product_l129_129934

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : α) (r : α) (n : ℕ) : α :=
  a * r ^ n

theorem geometric_product {a r : α}
  (h1 : a * (geometric_sequence a r 4) * (geometric_sequence a r 6) = -3 * real.sqrt 3) :
  (geometric_sequence a r 1) * (geometric_sequence a r 7) = 3 :=
by
  sorry

end geometric_product_l129_129934


namespace tangerine_boxes_l129_129701

theorem tangerine_boxes
  (num_boxes_apples : ℕ)
  (apples_per_box : ℕ)
  (num_boxes_tangerines : ℕ)
  (tangerines_per_box : ℕ)
  (total_fruits : ℕ)
  (h1 : num_boxes_apples = 19)
  (h2 : apples_per_box = 46)
  (h3 : tangerines_per_box = 170)
  (h4 : total_fruits = 1894)
  : num_boxes_tangerines = 6 := 
  sorry

end tangerine_boxes_l129_129701


namespace tessa_needs_more_apples_l129_129223

/-- Tessa starts with 4 apples.
    Anita gives her 5 more apples.
    She needs 10 apples to make a pie.
    Prove that she needs 1 more apple to make the pie.
-/
theorem tessa_needs_more_apples:
  ∀ initial_apples extra_apples total_needed extra_needed: ℕ,
    initial_apples = 4 → extra_apples = 5 → total_needed = 10 →
    extra_needed = total_needed - (initial_apples + extra_apples) →
    extra_needed = 1 :=
by
  intros initial_apples extra_apples total_needed extra_needed hi he ht heq
  rw [hi, he, ht] at heq
  simp at heq
  assumption

end tessa_needs_more_apples_l129_129223


namespace area_of_given_region_l129_129302

noncomputable def radius_squared : ℝ := 16 -- Completing the square gives us a radius squared value of 16.
def area_of_circle (r : ℝ) : ℝ := π * r ^ 2

theorem area_of_given_region : area_of_circle (real.sqrt radius_squared) = 16 * π := by
  sorry

end area_of_given_region_l129_129302


namespace abc_one_eq_sum_l129_129953

theorem abc_one_eq_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a * b * c = 1) :
  (a^2 * b^2) / ((a^2 + b * c) * (b^2 + a * c))
  + (a^2 * c^2) / ((a^2 + b * c) * (c^2 + a * b))
  + (b^2 * c^2) / ((b^2 + a * c) * (c^2 + a * b))
  = 1 / (a^2 + 1 / a) + 1 / (b^2 + 1 / b) + 1 / (c^2 + 1 / c) := by
  sorry

end abc_one_eq_sum_l129_129953


namespace pieces_per_box_correct_l129_129752

-- Define the number of boxes Will bought
def total_boxes_bought := 7

-- Define the number of boxes Will gave to his brother
def boxes_given := 3

-- Define the number of pieces left with Will
def pieces_left := 16

-- Define the function to find the pieces per box
def pieces_per_box (total_boxes : Nat) (given_away : Nat) (remaining_pieces : Nat) : Nat :=
  remaining_pieces / (total_boxes - given_away)

-- Prove that each box contains 4 pieces of chocolate candy
theorem pieces_per_box_correct : pieces_per_box total_boxes_bought boxes_given pieces_left = 4 :=
by
  sorry

end pieces_per_box_correct_l129_129752


namespace first_nonzero_digit_right_of_decimal_1_199_l129_129720

theorem first_nonzero_digit_right_of_decimal_1_199 :
  let x := (1 / 199 : ℚ)
  let first_nonzero_digit := 2
  (∃ (n : ℕ), x * 10^n - (x * 10^n).floor = first_nonzero_digit * 10^(-(n-1))) :=
begin
  sorry
end


end first_nonzero_digit_right_of_decimal_1_199_l129_129720


namespace solution_l129_129523

-- Definitions (conditions from a)
def vec_a : ℝ × ℝ := (1, 0)
def vec_b (λ : ℝ) : ℝ × ℝ := (λ, 1)
def vec_sum (λ : ℝ) : ℝ × ℝ := (vec_a.1 + vec_b(λ).1, vec_a.2 + vec_b(λ).2)  -- vector addition
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2  -- dot product definition

-- Statement of the problem
theorem solution (λ : ℝ) (h : dot_product (vec_sum λ) vec_a = 0) : λ = -1 := by
  sorry

end solution_l129_129523


namespace final_score_correct_l129_129767

-- defining constants and conditions
constant song_content_percentage : ℝ := 0.30
constant singing_skills_percentage : ℝ := 0.40
constant spirit_percentage : ℝ := 0.30

constant song_content_points : ℝ := 90
constant singing_skills_points : ℝ := 94
constant spirit_points : ℝ := 98

-- defining the final score calculation
noncomputable def final_score : ℝ :=
  song_content_percentage * song_content_points +
  singing_skills_percentage * singing_skills_points +
  spirit_percentage * spirit_points

-- theorem stating the proof problem
theorem final_score_correct : final_score = 94 := 
  sorry

end final_score_correct_l129_129767


namespace sally_seashell_solution_l129_129993

variable (T : ℕ) (seashells_monday : ℕ) (price : ℕ → ℝ) (total_revenue : ℝ)

def seashell_ratio : Prop :=
  price 1 * (seashells_monday + T) = total_revenue →
  (T = total_revenue / price 1 - seashells_monday) →
  (T : ℝ) / seashells_monday = 1 / 2

noncomputable def sally_seashell_problem : Prop :=
  seashell_ratio (T := 15) (seashells_monday := 30) (price := λ n, 1.20)
                 (total_revenue := 54)

theorem sally_seashell_solution : sally_seashell_problem := 
  by
    sorry

end sally_seashell_solution_l129_129993


namespace g_function_property_l129_129808

theorem g_function_property :
  ∀ (g : ℤ → ℤ),
  (∀ m n : ℤ, g(m + n) + g(mn - 1) = g(m) * g(n) - g(m) + g(n) + 1) →
  g(0) = 0 →
  (g(2) = 1 ∧ 
   ∃! (x : ℕ), x = (λ g_2_values, (∑ y in g_2_values, y) * g_2_values.card) (finset.image (λ n : ℤ, g(n)) {2})) := by
  intro g h₁ h₂
  sorry

end g_function_property_l129_129808


namespace part_I_interval_part_II_range_l129_129105

noncomputable def f (x : ℝ) : ℝ := 2 * (cos x)^2 + 2 * sqrt 3 * sin x * cos x - 1

theorem part_I_interval :
  ∀ k : ℤ, ∀ x : ℝ, (π/6 + k*π ≤ x ∧ x ≤ 2*π/3 + k*π) → 
  (f x has_deriv_at (2 * (cos (2*x + π/6)))) x → 
  2 * (cos (2*x + π/6)) < 0 := sorry

theorem part_II_range :
  ∀ a b c : ℝ, ∀ A : ℝ, (tan B = sqrt 3 * a * c / (a^2 + c^2 - b^2)) -> 
  (π/6 < A ∧ A < π/2) → 
  (f A ≥ -1 ∧ f A < 2) := sorry

end part_I_interval_part_II_range_l129_129105


namespace number_of_birds_l129_129387

/-- 
  Given the number of dogs, snakes, spiders, and total legs, prove the number of birds.
-/
theorem number_of_birds (num_dogs : ℕ) (num_snakes : ℕ) (num_spiders : ℕ) (total_legs : ℕ) : 
  num_dogs = 5 → num_snakes = 4 → num_spiders = 1 → total_legs = 34 → 
  (∃ num_birds : ℕ, num_birds * 2 + num_dogs * 4 + num_snakes * 0 + num_spiders * 8 = total_legs ∧ num_birds = 3) :=
by
  intros h1 h2 h3 h4
  use 3
  split
  {
    rw [h1, h2, h3, h4]
    norm_num
  }
  {
    refl
  }

end number_of_birds_l129_129387


namespace find_last_score_l129_129548

def last_score_is_81 (scores : List ℤ) : Prop :=
  scores = [73, 77, 81, 85, 92] ∧
  (∃ last_score, scores.sum = 408 ∧ scores.avg = 81 ∧ last_score = 81)

theorem find_last_score
  (h : last_score_is_81 [73, 77, 81, 85, 92]) : (∃ last_score, last_score = 81) :=
sorry

end find_last_score_l129_129548


namespace sum_of_three_ints_product_5_4_l129_129259

theorem sum_of_three_ints_product_5_4 :
  ∃ (a b c: ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 51 :=
by
  sorry

end sum_of_three_ints_product_5_4_l129_129259


namespace complex_cubic_root_l129_129221

def positive_integer (n : ℤ) := n > 0

theorem complex_cubic_root :
  ∃ a b : ℤ, positive_integer a ∧ positive_integer b ∧ (a + b * complex.I)^3 = (2 : ℂ) + 11 * complex.I ∧ 
  (a + b * complex.I = 2 + complex.I) := 
by 
  sorry

end complex_cubic_root_l129_129221


namespace trigonometric_simplification_l129_129398

theorem trigonometric_simplification :
  (1 - cos (10 * (pi/180))^2) / (cos (800 * (pi/180)) * sqrt (1 - cos (20 * (pi/180)))) = sqrt 2 / 2 := 
sorry

end trigonometric_simplification_l129_129398


namespace bullet_trains_crossing_time_l129_129709

theorem bullet_trains_crossing_time :
  ∀ (length : ℝ) (time1 time2 : ℝ),
    length = 120 →
    time1 = 10 →
    time2 = 20 →
    let v1 := length / time1 in
    let v2 := length / time2 in
    let relative_speed := v1 + v2 in
    let total_distance := 2 * length in
    let crossing_time := total_distance / relative_speed in
    crossing_time = 13.33 :=
begin
  intros length time1 time2 h_length h_time1 h_time2,
  let v1 := length / time1,
  let v2 := length / time2,
  let relative_speed := v1 + v2,
  let total_distance := 2 * length,
  let crossing_time := total_distance / relative_speed,
  sorry,
end

end bullet_trains_crossing_time_l129_129709


namespace exists_positive_integer_s_l129_129946

open Nat

noncomputable def sequence {α : Type*} := ℕ → α

def nondecreasing (a : sequence ℕ) : Prop :=
  ∀ n m, n ≤ m → a n ≤ a m

theorem exists_positive_integer_s (a : sequence ℕ) (k r : ℕ)
  (h1 : nondecreasing a)
  (h2 : a r > 0)
  (h3 : r / a r = k + 1) :
  ∃ s, s > 0 ∧ s / a s = k := 
sorry

end exists_positive_integer_s_l129_129946


namespace rational_root_count_l129_129005

theorem rational_root_count (b4 b3 b2 b1 : ℤ) :
  let coeff := [12, b4, b3, b2, b1, 18]
  let leading_coef_factors := [±1, ±2, ±3, ±4, ±6, ±12]
  let const_factors := [±1, ±2, ±3, ±6, ±9, ±18]
  let possible_rational_roots := set.image (λ (p : ℤ × ℤ), (p.1, p.2)) (list.prod const_factors leading_coef_factors) 
        in possible_rational_roots.card = 24 :=
by {
  sorry
}

end rational_root_count_l129_129005


namespace angle_2a_minus_b_l129_129908

variables {V : Type*} [InnerProductSpace ℝ V]

-- Definitions based on given conditions
def angle (u v : V) : ℝ :=
  real.arccos (real_inner u v / (∥u∥ * ∥v∥))

variables (a b : V)

-- Given condition: the angle between vectors (a and b) is 60 degrees
axiom angle_a_b : angle a b = real.pi / 3  -- 60 degrees in radians

-- Proof problem: Prove the angle between 2a and -b is 120 degrees
theorem angle_2a_minus_b : angle (2 • a) (-b) = 2 * real.pi / 3 := -- 120 degrees in radians
sorry

end angle_2a_minus_b_l129_129908


namespace area_of_circle_eq_sixteen_pi_l129_129304

theorem area_of_circle_eq_sixteen_pi :
  ∃ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) ↔ (π * 4^2 = 16 * π) :=
by
  sorry

end area_of_circle_eq_sixteen_pi_l129_129304


namespace max_S_at_8_l129_129844

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Define sequence a_n according to the given problem conditions
axiom a_1 : a 1 = -1 / 2
axiom a_rec (n : ℕ) : a (n + 1) * b n = b (n + 1) * a n + b n

-- Define sequence b_n according to the given problem conditions
axiom b_def (n : ℕ) : b n = (1 + (-1) ^ n * 5) / 2

-- Define the sum S_{2n}
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range (2 * n), a i

-- Statement to prove that S_{2n} is maximized at n = 8
theorem max_S_at_8 : ∃ (n : ℕ), n = 8 ∧ ∀ m, S n ≥ S m := by
  sorry

end max_S_at_8_l129_129844


namespace cube_volume_l129_129739

theorem cube_volume (SA : ℕ) (h : SA = 294) : 
  ∃ V : ℕ, V = 343 := 
by
  sorry

end cube_volume_l129_129739


namespace reduced_price_per_kg_l129_129368

/-- Given that:
1. There is a reduction of 25% in the price of oil.
2. The housewife can buy 5 kgs more for Rs. 700 after the reduction.

Prove that the reduced price per kg of oil is Rs. 35. -/
theorem reduced_price_per_kg (P : ℝ) (R : ℝ) (X : ℝ)
  (h1 : R = 0.75 * P)
  (h2 : 700 = X * P)
  (h3 : 700 = (X + 5) * R)
  : R = 35 := 
sorry

end reduced_price_per_kg_l129_129368


namespace sally_garden_area_l129_129991

theorem sally_garden_area :
  ∃ (a b : ℕ), 2 * (a + b) = 24 ∧ b + 1 = 3 * (a + 1) ∧ 
     (3 * (a - 1) * 3 * (b - 1) = 297) :=
by {
  sorry
}

end sally_garden_area_l129_129991


namespace value_of_y_l129_129903

theorem value_of_y (k : ℕ) (y : ℝ) (h_k : k = 9) (h_eq : (1/2) ^ 18 * (1/81) ^ k = y) : 
  y = 1 / (2 ^ 18 * 3 ^ 36) := 
by 
  rw [h_k] at h_eq 
  rw [← (rat.cast_pow : ∀ (m : ℚ) (n : ℕ), (m ^ n : ℝ) = ((m : ℝ) ^ n)) (1/81) 9] at h_eq
  rw [← (rat.cast_pow : ∀ (m : ℚ) (n : ℕ), (m ^ n : ℝ) = ((m : ℝ) ^ n)) (1/2) 18] at h_eq
  rw [← rat.cast_mul, ← pow_mul] at h_eq
  simp only [nat.cast_one, nat.cast_bit0, nat.cast_bit1, pow_one] at h_eq
  exact h_eq

end value_of_y_l129_129903


namespace distribution_scheme_count_l129_129017

noncomputable def NumberOfDistributionSchemes : Nat :=
  let plumbers := 5
  let residences := 4
  Nat.choose plumbers (residences - 1) * Nat.factorial residences

theorem distribution_scheme_count :
  NumberOfDistributionSchemes = 240 :=
by
  sorry

end distribution_scheme_count_l129_129017


namespace num_integer_values_l129_129256

theorem num_integer_values (x : ℤ) :
  (∃ y : ℤ, x = -3 * y + 2 ∨ x = 3 * y ∨ x = -3 * y + 1 ∨ x = 3 * y - 1) ↔ 
  (∃ k : ℤ, number_of_solutions_eq : 4) := 
sorry

end num_integer_values_l129_129256


namespace first_nonzero_digit_right_of_decimal_1_199_l129_129719

theorem first_nonzero_digit_right_of_decimal_1_199 :
  let x := (1 / 199 : ℚ)
  let first_nonzero_digit := 2
  (∃ (n : ℕ), x * 10^n - (x * 10^n).floor = first_nonzero_digit * 10^(-(n-1))) :=
begin
  sorry
end


end first_nonzero_digit_right_of_decimal_1_199_l129_129719


namespace wrong_guess_is_20_l129_129665

-- Define the colors
inductive Color
| white
| brown
| black

-- Assume we have a sequence of 1000 bears
def bears : fin 1000 → Color := sorry

-- Hypotheses
axiom colors_per_three : ∀ (i : fin 998), 
  ({bears i, bears (i + 1), bears (i + 2)} = {Color.white, Color.brown, Color.black} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.black, Color.white, Color.brown} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.brown, Color.black, Color.white})

axiom exactly_one_wrong : 
  (bears 1 = Color.white ∧ bears 19 ≠ Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 ≠ Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 ≠ Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 ≠ Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 ≠ Color.white)

-- Define the theorem to prove
theorem wrong_guess_is_20 : 
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) →
  ¬(bears 19 = Color.brown) := 
sorry

end wrong_guess_is_20_l129_129665


namespace relationship_y_n_general_term_y_odd_limit_of_y_odd_seq_point_approach_b_formula_sum_products_l129_129869

noncomputable def y1 : ℝ := 4

def y_n : ℕ+ -> ℝ
| 1 := y1
| (Nat.succ_p n) := let pn := y_n n in 2 * 2^(-n)

theorem relationship_y_n (n : ℕ) (n_pos : n > 0) : 
  y_n (n + 1) + y_n n = 4 * (1 / 2) ^ n :=
sorry

theorem general_term_y_odd (n : ℕ+) : 
  ∀ (k : ℕ), k = 2 * n + 1 -> y_n k = (8 / 3) + (4 / 3) * (1 / 4) ^ (n - 1) :=
sorry

theorem limit_of_y_odd_seq (n : ℕ) : 
  tendsto (λ n, y_n (2 * n + 1)) at_top (𝓝 (8 / 3)) :=
sorry

theorem point_approach (n : ℕ) : 
  (∃ x, P_n (2 * n + 1) = (16 / 9, 8 / 3)) :=
sorry

def a (n : ℕ+) : ℝ :=
y_n (2 * n + 1) - y_n (2 * n - 1)

def S (n : ℕ+) : ℝ :=
∑ k in Finset.range n, a k

def b (n : ℕ+) : ℝ :=
1 / (3 / 4 * S n + 1)

theorem b_formula (n : ℕ+) : b n = 4 ^ n := sorry

theorem sum_products (n : ℕ) : 
∑ i in Finset.range n, ∑ j in Finset.range (i + 1), b i * b j = 
(4 ^ (2 * n + 3) - 5 * 4 ^ (n + 2) + 16) / 45 :=
sorry

end relationship_y_n_general_term_y_odd_limit_of_y_odd_seq_point_approach_b_formula_sum_products_l129_129869


namespace sum_of_areas_l129_129620

def base_width : ℕ := 3
def lengths : List ℕ := [1, 8, 27, 64, 125, 216]
def area (w l : ℕ) : ℕ := w * l
def total_area : ℕ := (lengths.map (area base_width)).sum

theorem sum_of_areas : total_area = 1323 := 
by sorry

end sum_of_areas_l129_129620


namespace special_fractions_sum_is_14_l129_129798

theorem special_fractions_sum_is_14 :
  {n : ℕ | ∃ (a1 a2 b1 b2 : ℕ), a1 + b1 = 20 ∧ a2 + b2 = 20 ∧ n = (a1 * b2 + a2 * b1) / (b1 * b2))}.to_finset.card = 14 :=
sorry

end special_fractions_sum_is_14_l129_129798


namespace directrix_eq_l129_129452

noncomputable def parabola_eq : (ℝ → ℝ) := λ x, (x^2 - 8 * x + 12) / 16

theorem directrix_eq : ∀ (y : ℝ), y = parabola_eq (x : ℝ) → ∃ d, d = -1 / 2 := by
  sorry

end directrix_eq_l129_129452


namespace margot_jogging_timetables_l129_129597

theorem margot_jogging_timetables : 
  ∃ n, (n = (∑ x in Finset.range 7, Finset.card {y ∈ Finset.range 7 | abs (y - x) ≥ 2}) / 2) ∧ n = 14 := 
by {
  sorry
}

end margot_jogging_timetables_l129_129597


namespace sum_of_first_five_terms_l129_129792

theorem sum_of_first_five_terms
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h_sum_n : ∀ n, S n = n / 2 * (a 1 + a n))
  (h_roots : ∀ x, x^2 - x - 3 = 0 → x = a 2 ∨ x = a 4)
  (h_vieta : a 2 + a 4 = 1) :
  S 5 = 5 / 2 :=
  sorry

end sum_of_first_five_terms_l129_129792


namespace solution_set_of_absolute_inequality_l129_129648

theorem solution_set_of_absolute_inequality :
  {x : ℝ | |2 * x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_absolute_inequality_l129_129648


namespace tan_B_value_triangle_area_l129_129939

variables {A B C a b c : ℝ}

noncomputable def given_conditions (A B C a b c : ℝ) : Prop :=
  A + B + C = Real.pi ∧
  cos A = (4 * Real.sqrt 3 / 3) * (sin C) ^ 2 - cos (B - C) ∧
  (Real.pi / 2) = (A + 3 * C) / 2

theorem tan_B_value {A B C a b c : ℝ} (h : given_conditions A B C a b c) :
  tan B = -2 * Real.sqrt 2 := 
sorry

noncomputable def area_triangle (a b c : ℝ) : ℝ := 
  (1 / 2) * b * c * sin A

theorem triangle_area {A B C a b c : ℝ} 
  (h : given_conditions A B C a b c) 
  (hb : b = 2 * Real.sqrt 2) :
  area_triangle a b c = (2 * Real.sqrt 2) / 3 :=
sorry

end tan_B_value_triangle_area_l129_129939


namespace point_inside_circle_slope_of_chord_l129_129631

-- Condition: Equation of circle and point P(-1, 2)
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8
def point_P := (-1, 2)

-- Question 1: Prove point P is inside the circle
theorem point_inside_circle : ∀ (x y : ℝ), point_P = (x, y) → (x + 1)^2 + y^2 < 8 :=
by
  intro x y hPx
  rw [hPx]
  sorry

-- Condition: Chord length |AB| = 2√7 and the center is (-1, 0)
def chord_length_AB := 2 * Real.sqrt 7
def center_O := (-1, 0)

-- Question 2: Prove the slope k of line AB
theorem slope_of_chord : ∃ k : ℝ, k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by
  sorry

end point_inside_circle_slope_of_chord_l129_129631


namespace prop_A_prop_B_prop_C_prop_D_l129_129112

variable {a b c x : ℝ}

-- Proposition A
theorem prop_A (h : b > a ∧ a > 0) : (1 / a) > (1 / b) :=
sorry

-- Proposition B (negated, demonstrating the non-universality with an example)
theorem prop_B (h : a > b) : ¬(∀ c : ℝ, a * c > b * c) :=
sorry
  
-- Proposition C
theorem prop_C (h : ac^2 > bc^2 ∧ c ≠ 0) : a > b :=
sorry

-- Proposition D
theorem prop_D : (¬ ∃ x : ℝ, x ∈ set.Ioi (-3) ∧ x^2 ≤ 9) ↔ ∀ x : ℝ, x ∈ set.Ioi (-3) → x^2 > 9 :=
sorry

end prop_A_prop_B_prop_C_prop_D_l129_129112


namespace minimum_value_of_abcd_minus_product_l129_129805

theorem minimum_value_of_abcd_minus_product (A B C D : ℕ)
  (hA1 : 1 ≤ A) (hA2 : A ≤ 9)
  (hB : 0 ≤ B ∧ B ≤ 9)
  (hC1 : 1 ≤ C) (hC2 : C ≤ 9)
  (hD : 0 ≤ D ∧ D ≤ 9) :
  ∃ (x y : ℕ), (x = 10 * A + B) ∧ (y = 10 * C + D) ∧ (100 * x + y - x * y) = 109 :=
begin
  sorry
end

end minimum_value_of_abcd_minus_product_l129_129805


namespace decagonal_die_expected_value_is_correct_l129_129312

def decagonalDieExpectedValue : ℕ := 5 -- A decagonal die has faces 1 to 10

def expectedValueDecagonalDie : ℝ := 5.5 -- The expected value as calculated.

theorem decagonal_die_expected_value_is_correct (p : fin 10 → ℝ) (i : fin 10) :
  p i = 1 / 10 ∧ (∑ i in finset.univ, p i * (i + 1 : ℝ)) = expectedValueDecagonalDie := by
    sorry

end decagonal_die_expected_value_is_correct_l129_129312


namespace angle_2a_minus_b_l129_129907

variables {V : Type*} [InnerProductSpace ℝ V]

-- Definitions based on given conditions
def angle (u v : V) : ℝ :=
  real.arccos (real_inner u v / (∥u∥ * ∥v∥))

variables (a b : V)

-- Given condition: the angle between vectors (a and b) is 60 degrees
axiom angle_a_b : angle a b = real.pi / 3  -- 60 degrees in radians

-- Proof problem: Prove the angle between 2a and -b is 120 degrees
theorem angle_2a_minus_b : angle (2 • a) (-b) = 2 * real.pi / 3 := -- 120 degrees in radians
sorry

end angle_2a_minus_b_l129_129907


namespace Portia_school_students_l129_129199

theorem Portia_school_students:
  ∃ (P L : ℕ), P = 2 * L ∧ P + L = 3000 ∧ P = 2000 :=
by
  sorry

end Portia_school_students_l129_129199


namespace largest_divisor_of_four_consecutive_odds_l129_129581

theorem largest_divisor_of_four_consecutive_odds :
    ∀ (a b c d : ℕ), odd a → odd b → odd c → odd d → a + 2 = b → b + 2 = c → c + 2 = d →
    ∃ k : ℕ, k = 15 ∧ ∀ n : ℕ, (∀ q : ℕ, q ∣ (a * b * c * d) → q ∣ n) ↔ n = k :=
by sorry

end largest_divisor_of_four_consecutive_odds_l129_129581


namespace current_expression_l129_129409

open Complex

-- Define the voltage V, impedance Z, and current I as complex numbers, depending on k
def V (k : ℝ) : ℂ := ⟨4 - 2 * k, 0⟩
def Z : ℂ := ⟨2, 4⟩
def I (k : ℝ) : ℂ := V k / Z

theorem current_expression (k : ℝ) : 
  I k = ⟨(2 - k) / 5, -(4 - 2 * k) / 5⟩ :=
sorry

end current_expression_l129_129409


namespace greatest_number_of_balloons_l129_129982

-- Let p be the regular price of one balloon, and M be the total amount of money Orvin has
variable (p M : ℝ)

-- Initial condition: Orvin can buy 45 balloons at the regular price.
-- Thus, he has money M = 45 * p
def orvin_has_enough_money : Prop :=
  M = 45 * p

-- Special Sale condition: The first balloon costs p and the second balloon costs p/2,
-- so total cost for 2 balloons = 1.5 * p
def special_sale_condition : Prop :=
  ∀ pairs : ℝ, M / (1.5 * p) = pairs ∧ pairs * 2 = 60

-- Given the initial condition and the special sale condition, prove the greatest 
-- number of balloons Orvin could purchase is 60
theorem greatest_number_of_balloons (p : ℝ) (M : ℝ) (h1 : orvin_has_enough_money p M) (h2 : special_sale_condition p M) : 
∀ N : ℝ, N = 60 :=
sorry

end greatest_number_of_balloons_l129_129982


namespace wolf_does_not_catch_hare_l129_129375

-- Define the distance the hare needs to cover
def distanceHare := 250 -- meters

-- Define the initial separation between the wolf and the hare
def separation := 30 -- meters

-- Define the speed of the hare
def speedHare := 550 -- meters per minute

-- Define the speed of the wolf
def speedWolf := 600 -- meters per minute

-- Define the time it takes for the hare to reach the refuge
def tHare := (distanceHare : ℚ) / speedHare

-- Define the total distance the wolf needs to cover
def totalDistanceWolf := distanceHare + separation

-- Define the time it takes for the wolf to cover the total distance
def tWolf := (totalDistanceWolf : ℚ) / speedWolf

-- Final proposition to be proven
theorem wolf_does_not_catch_hare : tHare < tWolf :=
by
  sorry

end wolf_does_not_catch_hare_l129_129375


namespace man_mass_calculation_l129_129764

/-- A boat has a length of 4 m, a breadth of 2 m, and a weight of 300 kg.
    The density of the water is 1000 kg/m³.
    When the man gets on the boat, it sinks by 1 cm.
    Prove that the mass of the man is 80 kg. -/
theorem man_mass_calculation :
  let length_boat := 4     -- in meters
  let breadth_boat := 2    -- in meters
  let weight_boat := 300   -- in kg
  let density_water := 1000  -- in kg/m³
  let additional_depth := 0.01 -- in meters (1 cm)
  volume_displaced = length_boat * breadth_boat * additional_depth →
  mass_water_displaced = volume_displaced * density_water →
  mass_of_man = mass_water_displaced →
  mass_of_man = 80 :=
by 
  intros length_boat breadth_boat weight_boat density_water additional_depth volume_displaced
  intros mass_water_displaced mass_of_man
  sorry

end man_mass_calculation_l129_129764


namespace arrangement_count_is_5040_l129_129529

theorem arrangement_count_is_5040 :
  ∃ (arrangements : ℕ), arrangements = nat.factorial 7 ∧ arrangements = 5040 :=
by
  use nat.factorial 7
  split
  -- The first part equals the 7! calculation
  refl
  -- The second part confirms that 7! actually equals 5040
  exact nat.factorial_eq 7 

-- Proof left as an exercise
sorry

end arrangement_count_is_5040_l129_129529


namespace find_d_in_triangle_l129_129917

theorem find_d_in_triangle (XY YZ XZ : ℕ) (hXY : XY = 390) (hYZ : YZ = 480) (hXZ : XZ = 560)
  (d : ℕ) (Q : Type) (h_parallel : ∀ {a b c : Type}, a = b → b = c → a = c)
  (h_segments_eq : ∀ {a b c : Type}, a = b ∧ b = c → c = d) : d = 320 := 
by
  -- Proof will be placed here
  sorry

end find_d_in_triangle_l129_129917


namespace simplify_and_evaluate_l129_129213

theorem simplify_and_evaluate (x : ℤ) (h : x = 2) :
  (2 * x + 1) ^ 2 - (x + 3) * (x - 3) = 30 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l129_129213


namespace product_of_divisors_72_l129_129465

theorem product_of_divisors_72 : (∏ d in (finset.divisors 72), d) = 139314069504 :=
by 
  -- Proof goes here
  sorry

end product_of_divisors_72_l129_129465


namespace dogs_in_school_l129_129794

theorem dogs_in_school
  (sit: ℕ) (sit_and_stay: ℕ) (stay: ℕ) (stay_and_roll_over: ℕ)
  (roll_over: ℕ) (sit_and_roll_over: ℕ) (all_three: ℕ) (none: ℕ)
  (h1: sit = 50) (h2: sit_and_stay = 17) (h3: stay = 29)
  (h4: stay_and_roll_over = 12) (h5: roll_over = 34)
  (h6: sit_and_roll_over = 18) (h7: all_three = 9) (h8: none = 9) :
  sit + stay + roll_over + sit_and_stay + stay_and_roll_over + sit_and_roll_over - 2 * all_three + none = 84 :=
by sorry

end dogs_in_school_l129_129794


namespace divide_by_10_result_l129_129975

theorem divide_by_10_result (x : ℕ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end divide_by_10_result_l129_129975


namespace king_squares_even_adjacent_l129_129154

theorem king_squares_even_adjacent :
  ¬∃ path : List (Fin 8 × Fin 8), (∀ (i : Fin (path.length - 1)), 
  let visited := path.take (i+1) in visited.Nodup ∧ (Finset.card (Finset.filter (λ sq, ∃ (prev : Fin 8 × Fin 8), prev ∈ visited ∧ sq ∼ prev) 
  (Finset.image id Finset.univ)) % 2 = 0)
  } := sorry

end king_squares_even_adjacent_l129_129154


namespace volume_intersection_l129_129725

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

def region1 (x y z : ℝ) : Prop := abs x + abs y + abs z ≤ 1
def region2 (x y z : ℝ) : Prop := abs x + abs y + abs (z - 2) ≤ 1

theorem volume_intersection : 
  (volume {p : ℝ × ℝ × ℝ | region1 p.1 p.2 p.3 ∧ region2 p.1 p.2 p.3}) = (1 / 12 : ℝ) :=
by
  sorry

end volume_intersection_l129_129725


namespace inequality_proof_l129_129180

-- Let x be a sequence of real numbers: x1, x2, ..., x_n
variable {n : ℕ} (x : Fin n → ℝ)

theorem inequality_proof :
  (∑ i in Finset.range n, x i / (1 + ∑ j in Finset.range (i + 1), (x j)^2)) < Real.sqrt n := 
sorry

end inequality_proof_l129_129180


namespace expansion_coefficient_l129_129860

theorem expansion_coefficient (m : ℝ) (z : ℝ) :
  (∀ (C : ℕ → ℝ), C 5 4 * 2 - C 5 3 * m * (-1) = -20) → m = 3 :=
sorry

end expansion_coefficient_l129_129860


namespace square_garden_perimeter_l129_129626

theorem square_garden_perimeter (A : ℝ) (hA : A = 450) : 
    ∃ P : ℝ, P = 60 * Real.sqrt 2 :=
  by
    sorry

end square_garden_perimeter_l129_129626


namespace sum_of_coefficients_of_quadratic_with_distinct_negative_solutions_l129_129197

theorem sum_of_coefficients_of_quadratic_with_distinct_negative_solutions :
  (∑ b in {b : ℤ | ∃ r s : ℤ, r ≠ s ∧ r < 0 ∧ s < 0 ∧ r * s = 24 ∧ r + s = b}, b) = -60 :=
by
  sorry

end sum_of_coefficients_of_quadratic_with_distinct_negative_solutions_l129_129197


namespace trajectory_of_midpoint_l129_129497

-- Define the points and conditions
variables {A B : ℝ × ℝ} (M : ℝ × ℝ)
variables {x1 x2 : ℝ} (hA : A = (x1, x1)) (hB : B = (x2, -x2))
variables (h_pos1 : 0 < x1) (h_pos2 : 0 < x2)

-- Define the area condition
def area_OAB : ℝ := (1/2) * (x1 * x2)

-- Define the midpoint M
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- State that the area of triangle OAB is 2
axiom area_OAB_eq_2 : area_OAB x1 x2 = 2

-- State the proof problem
theorem trajectory_of_midpoint 
    (h_mid : M = midpoint A B)
    (h_area : area_OAB x1 x2 = 2) 
    : M.fst ^ 2 - M.snd ^ 2 = 2 ∧ 0 < M.fst :=
  sorry

end trajectory_of_midpoint_l129_129497


namespace marbles_start_count_l129_129401

theorem marbles_start_count (marbles_bought : ℝ) (total_marbles : ℝ) (initial_marbles : ℝ) : 
  marbles_bought = 489.0 ∧ total_marbles = 2778.0 → initial_marbles = 2289.0 :=
by
  intros h,
  cases h with hb ht,
  have h1 : initial_marbles = total_marbles - marbles_bought,
  sorry

end marbles_start_count_l129_129401


namespace probability_log3_integer_l129_129367

noncomputable def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n <= 999

noncomputable def is_power_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3^k

noncomputable def three_digit_powers_of_three : finset ℕ := finset.filter is_power_of_three (finset.range 1000).filter is_three_digit

theorem probability_log3_integer :
  let num_three_digit := 900 
  ∧ let num_valid := three_digit_powers_of_three.card in
  (num_valid.to_rat / num_three_digit.to_rat) = (1 / 450) := 
by
  sorry

end probability_log3_integer_l129_129367


namespace opposite_of_neg_three_quarters_l129_129638

theorem opposite_of_neg_three_quarters : ∃ (b : ℚ), (-3/4) + b = 0 ∧ b = 3/4 := by
  use 3/4
  split
  · exact add_right_neg (-3/4)
  · rfl
  sorry

end opposite_of_neg_three_quarters_l129_129638


namespace julia_mile_time_l129_129977

variable (x : ℝ)

theorem julia_mile_time
  (h1 : ∀ x, x > 0)
  (h2 : ∀ x, x <= 13)
  (h3 : 65 = 5 * 13)
  (h4 : 50 = 65 - 15)
  (h5 : 50 = 5 * x) :
  x = 10 := by
  sorry

end julia_mile_time_l129_129977


namespace interval_of_monotonic_increase_l129_129515

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (π * x + π / 3)
def omega : ℝ := π
def phi : ℝ := π / 3
def x1 : ℝ
def x2 : ℝ
def k : ℤ

axiom h1 : (ω > 0 ∧ 0 < φ ∧ φ < π / 2)
axiom h2 : f x1 = 2
axiom h3 : f x2 = 0
axiom h4 : |x1 - x2| = 1 / 2
axiom h5 : f (1 / 2) = 1

theorem interval_of_monotonic_increase :
  ∃ x1 x2 k : ℤ, 
  f x1 = 2 ∧ f x2 = 0 ∧ |x1 - x2| = 1 / 2 ∧ f (1 / 2) = 1 ∧
  (2 * k * π - π / 2 ≤ π * x1 + π / 3 ∧ π * x1 + π / 3 ≤ 2 * k * π + π / 2) ∧
  -5 / 6 + 2 * k ≤ x1 ∧ x1 ≤ 1 / 6 + 2 * k := 
sorry

end interval_of_monotonic_increase_l129_129515


namespace sum_of_number_and_its_square_l129_129126

theorem sum_of_number_and_its_square (x : ℕ) (h : x = 11) : x + x^2 = 132 :=
by {
  rw h,
  sorry
}

end sum_of_number_and_its_square_l129_129126


namespace correct_option_is_C_l129_129749

theorem correct_option_is_C : 
  (∀ (x : ℝ), sqrt 16 ≠ ±4) ∧
  (∀ (y : ℝ), sqrt ((-3)^2) ≠ -3) ∧
  (∀ (z : ℝ), ±sqrt 81 = ±9) ∧
  (∀ (w : ℝ), sqrt (-4) ≠ 2) → 
  (true) :=
sorry

end correct_option_is_C_l129_129749


namespace perpendicular_lines_implies_m_values_l129_129879

-- Define the equations of the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x - (m - 2) * y + 2 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y - 1 = 0

-- Define the condition of perpendicularity between lines l1 and l2
def perpendicular (m : ℝ) : Prop :=
  let a1 := (m + 2) / (m - 2)
  let a2 := -3 / m
  a1 * a2 = -1

-- The statement to be proved
theorem perpendicular_lines_implies_m_values (m : ℝ) :
  (∀ x y : ℝ, l1 m x y ∧ l2 m x y → perpendicular m) → (m = -1 ∨ m = 6) :=
sorry

end perpendicular_lines_implies_m_values_l129_129879


namespace delta_k_f_l129_129067

open Nat

-- Define the function
def f (n : ℕ) : ℕ := 3^n

-- Define the discrete difference operator
def Δ (g : ℕ → ℕ) (n : ℕ) : ℕ := g (n + 1) - g n

-- Define the k-th discrete difference
def Δk (g : ℕ → ℕ) (k : ℕ) (n : ℕ) : ℕ :=
  if k = 0 then g n else Δk (Δ g) (k - 1) n

-- State the theorem
theorem delta_k_f (k : ℕ) (n : ℕ) (h : k ≥ 1) : Δk f k n = 2^k * 3^n := by
  sorry

end delta_k_f_l129_129067


namespace right_angled_triangle_rotated_around_side_forms_cone_l129_129207

-- Define the geometrical conditions
variable {α : Type}
structure RightAngledTriangle :=
  (side1 side2 hypotenuse : α)
  (right_angle : side1 * side2 / 2 = hypotenuse * hypotenuse / 2)

-- Define the rotation operation
structure Rotation :=
  (triangle : RightAngledTriangle)
  (rotation_axis : α)

-- State the theorem
theorem right_angled_triangle_rotated_around_side_forms_cone 
  (T : RightAngledTriangle) 
  (axis : T.side1 ∨ T.side2) : 
  ∃ (B : Type), B = 'Cone := sorry

end right_angled_triangle_rotated_around_side_forms_cone_l129_129207


namespace real_root_is_five_l129_129196

noncomputable def find_real_root (a b r : ℝ) : Prop :=
  (a ≠ 0) → (∃ x : ℂ, x^3 + 3*x^2 + b*x - 125 = 0 ∧ (x = -3 - 4*complex.I ∨ x = -3 + 4*complex.I ∨ x = r))

theorem real_root_is_five
  (a b : ℝ) (ha : a ≠ 0)
  (hroot : find_real_root a b 5) :
  (∃ r : ℝ, r = 5) :=
by {
  sorry
}

end real_root_is_five_l129_129196


namespace volume_intersection_l129_129724

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

def region1 (x y z : ℝ) : Prop := abs x + abs y + abs z ≤ 1
def region2 (x y z : ℝ) : Prop := abs x + abs y + abs (z - 2) ≤ 1

theorem volume_intersection : 
  (volume {p : ℝ × ℝ × ℝ | region1 p.1 p.2 p.3 ∧ region2 p.1 p.2 p.3}) = (1 / 12 : ℝ) :=
by
  sorry

end volume_intersection_l129_129724


namespace prove_abs_diff_of_xy_l129_129964

noncomputable def x : ℝ := 3.2
noncomputable def y : ℝ := 4.7

def floorFracSumEq1 : Prop := (⌊ x ⌋₊ : ℝ) + frac y = 3.7
def fracFloorSumEq2 : Prop := frac x + (⌊ y ⌋₊ : ℝ) = 4.2

theorem prove_abs_diff_of_xy :
  floorFracSumEq1 ∧ fracFloorSumEq2 → abs (x - y) = 1.5 :=
sorry

end prove_abs_diff_of_xy_l129_129964


namespace each_boy_earns_14_dollars_l129_129294

theorem each_boy_earns_14_dollars :
  let Victor_shrimp := 26 in
  let Austin_shrimp := Victor_shrimp - 8 in
  let total_Victor_Austin_shrimp := Victor_shrimp + Austin_shrimp in
  let Brian_shrimp := total_Victor_Austin_shrimp / 2 in
  let total_shrimp := Victor_shrimp + Austin_shrimp + Brian_shrimp in
  let total_money := (total_shrimp / 11) * 7 in
  let money_per_boy := total_money / 3 in
  money_per_boy = 14 :=
by
  sorry

end each_boy_earns_14_dollars_l129_129294


namespace jen_scored_more_l129_129390

def bryan_score : ℕ := 20
def total_points : ℕ := 35
def sammy_mistakes : ℕ := 7
def sammy_score : ℕ := total_points - sammy_mistakes
def jen_score : ℕ := sammy_score + 2

theorem jen_scored_more :
  jen_score - bryan_score = 10 := by
  -- Proof to be filled in
  sorry

end jen_scored_more_l129_129390


namespace unique_polynomial_P_l129_129433

open Polynomial

/-- The only polynomial P with real coefficients such that
    xP(y/x) + yP(x/y) = x + y for all nonzero real numbers x and y 
    is P(x) = x. --/
theorem unique_polynomial_P (P : ℝ[X]) (hP : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x * P.eval (y / x) + y * P.eval (x / y) = x + y) :
P = Polynomial.C 1 * X :=
by sorry

end unique_polynomial_P_l129_129433


namespace central_angle_of_sector_l129_129264

theorem central_angle_of_sector (r l : ℝ) (h1 : r = 1) (h2 : l = 4 - 2*r) : 
    ∃ α : ℝ, α = 2 :=
by
  use l / r
  have hr : r = 1 := h1
  have hl : l = 4 - 2*r := h2
  sorry

end central_angle_of_sector_l129_129264


namespace laser_beam_total_distance_l129_129776

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def total_distance : ℝ :=
  let A : point := (1, 3)
  let B : point := (0, 3)
  let C : point := (0, -3)
  let D : point := (8, -3)
  let E : point := (8, 3)
  distance A B + distance B C + distance C D + distance D E

theorem laser_beam_total_distance : total_distance = 15 := 
by 
sory

end laser_beam_total_distance_l129_129776


namespace minimum_rubles_needed_l129_129760

theorem minimum_rubles_needed : ∃ r, r = 5 ∧ 
  ∀ n ∈ {x | 2 ≤ x ∧ x ≤ 30},
    ∃ S : finset ℕ,
      S.card = r ∧ (∀ m ∈ S, m ∈ {x | 2 ≤ x ∧ x ≤ 30}) ∧
      (∀ x ∈ {x | 2 ≤ x ∧ x ≤ 30}, S.some (λ m, m ∣ x ∨ x ∣ m)) :=
sorry

end minimum_rubles_needed_l129_129760


namespace find_boys_l129_129894

-- Variable declarations
variables (B G : ℕ)

-- Conditions
def total_students (B G : ℕ) : Prop := B + G = 466
def more_girls_than_boys (B G : ℕ) : Prop := G = B + 212

-- Proof statement: Prove B = 127 given both conditions
theorem find_boys (h1 : total_students B G) (h2 : more_girls_than_boys B G) : B = 127 :=
sorry

end find_boys_l129_129894


namespace employee_discount_percentage_l129_129370

-- Definitions of the conditions
def wholesale_cost : ℝ := 200
def retail_price (wholesale_cost : ℝ) : ℝ := wholesale_cost + 0.2 * wholesale_cost
def employee_paid : ℝ := 216
def discount_amount (retail_price employee_paid : ℝ) : ℝ := retail_price - employee_paid
def discount_percentage (discount_amount retail_price: ℝ) : ℝ := (discount_amount / retail_price) * 100

-- The proof problem statement
theorem employee_discount_percentage : 
  retail_price wholesale_cost = 240 →
  discount_amount (retail_price wholesale_cost) employee_paid = 24 →
  discount_percentage (discount_amount (retail_price wholesale_cost) employee_paid) (retail_price wholesale_cost) = 10 := 
by 
  intros h1 h2 
  rw [h1, h2] 
  sorry

end employee_discount_percentage_l129_129370


namespace find_speed_of_boat_l129_129649

theorem find_speed_of_boat (r d t : ℝ) (x : ℝ) (h_rate : r = 4) (h_dist : d = 33.733333333333334) (h_time : t = 44 / 60) 
  (h_eq : d = (x + r) * t) : x = 42.09090909090909 :=
  sorry

end find_speed_of_boat_l129_129649


namespace angle_EBC_20_l129_129170

noncomputable def angle_ABC := 40
noncomputable def angle_ABD := 30
noncomputable def angle_DBE := 10

def point_E_on_ray_BD : Prop := True -- Placeholder for geometric construction

theorem angle_EBC_20 : 
  angle_ABC = 40 → 
  angle_ABD = 30 → 
  angle_DBE = 10 → 
  point_E_on_ray_BD →
  (angle_ABC - (angle_ABD - angle_DBE)) = 20 :=
by
  intros hABC hABD hDBE hEonBD
  simp [hABC, hABD, hDBE]
  sorry

end angle_EBC_20_l129_129170


namespace polynomial_expansion_l129_129025

theorem polynomial_expansion :
  (7 * x^2 + 3 * x + 1) * (5 * x^3 + 2 * x + 6) = 
  35 * x^5 + 15 * x^4 + 19 * x^3 + 48 * x^2 + 20 * x + 6 := 
by
  sorry

end polynomial_expansion_l129_129025


namespace ed_money_left_l129_129427

theorem ed_money_left
  (cost_per_hour_night : ℝ := 1.5)
  (cost_per_hour_morning : ℝ := 2)
  (initial_money : ℝ := 80)
  (hours_night : ℝ := 6)
  (hours_morning : ℝ := 4) :
  initial_money - (cost_per_hour_night * hours_night + cost_per_hour_morning * hours_morning) = 63 := 
  by
  sorry

end ed_money_left_l129_129427


namespace count_similar_S_l129_129373

def string := list char

def similar (s1 s2: string) : Prop :=
  ∃ (a b c: string), (s1 = a ++ b ++ c) ∧ (s2 = a ++ b.reverse ++ c)

def S : string := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def count_similar (S : string) : nat :=
  sorry

theorem count_similar_S : count_similar S = 1126 :=
  sorry

end count_similar_S_l129_129373


namespace sufficient_but_not_necessary_condition_l129_129892

variable {x k : ℝ}

def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (2 - x) / (x + 1) < 0

theorem sufficient_but_not_necessary_condition (h_suff : ∀ x, p x k → q x) (h_not_necessary : ∃ x, q x ∧ ¬p x k) : k > 2 :=
sorry

end sufficient_but_not_necessary_condition_l129_129892


namespace range_of_a_empty_intersection_range_of_a_sufficient_condition_l129_129008

def set_A (a : ℝ) := {x : ℝ | |x - a| ≤ 4}
def set_B := {x : ℝ | (x - 2) * (x - 3) ≤ 0}

-- Problem 1
theorem range_of_a_empty_intersection (a : ℝ) : set_A a ∩ set_B = ∅ → a ∈ set.Iio (-2) ∪ set.Ioi 7 :=
by
  intro h
  sorry

-- Problem 2
theorem range_of_a_sufficient_condition (a : ℝ) : (∀ x, set_B x → set_A a x) ∧ (∃ x, set_A a x ∧ ¬ set_B x) → a ∈ set.Icc 1 6 :=
by
  intro h
  sorry

end range_of_a_empty_intersection_range_of_a_sufficient_condition_l129_129008


namespace isosceles_triangle_base_length_l129_129238

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l129_129238


namespace find_angle_B_l129_129545

theorem find_angle_B (a b c : ℝ) (h : a^2 + c^2 - b^2 = a * c) : 
  ∃ B : ℝ, 0 < B ∧ B < 180 ∧ B = 60 :=
by 
  sorry

end find_angle_B_l129_129545


namespace stamps_gcd_l129_129569

theorem stamps_gcd (a b : ℕ) (h1 : a = 924) (h2 : b = 1386) : Nat.gcd a b = 462 := by
  rw [h1, h2]
  rw [Nat.gcd_comm a b]
  sorry

end stamps_gcd_l129_129569


namespace general_formula_a_Tn_bound_l129_129072

section
  -- Define sequence {a_n}
  variable (a : ℕ+ → ℝ)
  variable (S : ℕ+ → ℝ)
  axiom a_n_condition (n : ℕ+) : S n = (3 / 2) * a n - 1 / 2
  axiom a1_condition : a 1 = 1
  axiom a_ge2_condition (n : ℕ+) (hn : n ≥ 2) : a n = 3 * a (n - 1)

  -- Part (1): Prove general formula for sequence {a_n}
  theorem general_formula_a (n : ℕ+) :
    (∀ n, a 1 = 1 ∧ (n ≥ 2 → a n = 3 * a (n - 1))) → a n = 3^(n - 1) :=
  sorry

  -- Define sequence {b_n}
  variable (b : ℕ+ → ℝ)
  axiom b_n_condition (n : ℕ+) : b n = (n : ℝ) / 3^n
  
  -- Define sum of first n terms of {b_n} as T_n
  def T (n : ℕ+) : ℝ := ∑ i in (Finset.range n), b (i + 1)

  -- Part (2): Prove T_n < 3/4
  theorem Tn_bound (n : ℕ+) : T n < 3 / 4 :=
  sorry
end

end general_formula_a_Tn_bound_l129_129072


namespace directrix_eq_l129_129453

noncomputable def parabola_eq : (ℝ → ℝ) := λ x, (x^2 - 8 * x + 12) / 16

theorem directrix_eq : ∀ (y : ℝ), y = parabola_eq (x : ℝ) → ∃ d, d = -1 / 2 := by
  sorry

end directrix_eq_l129_129453


namespace sum_reciprocal_S_10_l129_129086

noncomputable def a (n : ℕ) : ℕ := 2 * n

def S (n : ℕ) : ℕ := n * (n + 1)

theorem sum_reciprocal_S_10 : (∑ i in finset.range 10, 1 / S (i + 1)) = 10 / 11 := sorry

end sum_reciprocal_S_10_l129_129086


namespace no_two_consecutive_and_minimum_constraint_l129_129109

theorem no_two_consecutive_and_minimum_constraint :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  ∃ (n : ℕ), ∑ k in (finset.range 6), nat.choose (11 - k) k = 143 
:=
sorry

end no_two_consecutive_and_minimum_constraint_l129_129109


namespace ratio_rounding_l129_129605

theorem ratio_rounding :
  let r := (9 : ℝ) / 13 in
  Real.round (r * 10) / 10 = 0.7 :=
by
  let r := (9 : ℝ) / 13
  show Real.round (r * 10) / 10 = 0.7
  sorry

end ratio_rounding_l129_129605


namespace correct_answers_l129_129810

noncomputable def is_periodic_2 (f : ℝ → ℝ) : Prop :=
∀ x, f (x + 2) = f x

noncomputable def is_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
∀ x, f(2 * a - x) = f x

noncomputable def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

noncomputable def problem_def (f : ℝ → ℝ) : Prop :=
(even_function : ∀ x, f(-x) = f(x)) ∧
(f_x1_negf_x : ∀ x, f(x + 1) = -f(x)) ∧
(increasing_on_minus1_0 : is_increasing_on_interval f (-1) 0)

theorem correct_answers (f : ℝ → ℝ) (H : problem_def f) :
  is_periodic_2 f ∧
  is_symmetric_about_line f 1 ∧
  ¬is_increasing_on_interval f 0 1 ∧
  f 2 = f 0 :=
by sorry

end correct_answers_l129_129810


namespace average_value_of_series_l129_129392

theorem average_value_of_series (z : ℤ) :
  let series := [0^2, (2*z)^2, (4*z)^2, (8*z)^2]
  let sum_series := series.sum
  let n := series.length
  sum_series / n = 21 * z^2 :=
by
  let series := [0^2, (2*z)^2, (4*z)^2, (8*z)^2]
  let sum_series := series.sum
  let n := series.length
  sorry

end average_value_of_series_l129_129392


namespace tangent_line_at_1_monotonic_intervals_and_maximum_value_l129_129095

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 3 - 9 * x + 11

theorem tangent_line_at_1 : 
  let p := (1, f 1) in
  f 1 = 5 ∧ deriv f 1 = 0 ∧ ∀ x, (y = f 1 ↔ y = 5) :=
by 
  sorry

theorem monotonic_intervals_and_maximum_value : 
  (∀ x, -1 < x ∧ x < 1 → deriv f x < 0) ∧ 
  (∀ x, (x < -1 ∨ 1 < x) → deriv f x > 0) ∧ 
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f x ≤ 17 ∧ (f x = 17 ↔ x = -1)) :=
by 
  sorry

end tangent_line_at_1_monotonic_intervals_and_maximum_value_l129_129095


namespace distances_related_l129_129327

variable (distance_traveled remaining_distance total_distance : ℝ)

theorem distances_related (h : distance_traveled + remaining_distance = total_distance) :
  (distance_traveled, remaining_distance).related quantitative :=
by
  sorry

end distances_related_l129_129327


namespace tangency_condition_l129_129412

-- Definitions based on conditions
def ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 6
def hyperbola (x y n : ℝ) : Prop := 3 * x^2 - n * (y - 1)^2 = 3

-- The theorem statement based on the question and correct answer:
theorem tangency_condition (n : ℝ) (x y : ℝ) : 
  ellipse x y → hyperbola x y n → n = -6 :=
sorry

end tangency_condition_l129_129412


namespace sequence_property_l129_129562

noncomputable def a (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), ((-1)^i / (i + 1 : ℝ))

theorem sequence_property (k : ℕ) :
  a (k + 1) = a k + 1/(2*k + 1) - 1/(2*k + 2) :=
sorry

end sequence_property_l129_129562


namespace total_distance_l129_129018

theorem total_distance (D : ℝ) 
  (h1 : 1/4 * (3/8 * D) = 210) : D = 840 := 
by
  -- proof steps would go here
  sorry

end total_distance_l129_129018


namespace probability_subinterval_l129_129120

noncomputable def probability_2_pow_x_lt_2_in_0_to_4 : ℝ :=
  let indicator_function (x : ℝ) : ℝ :=
    if x ∈ set.Ioo 0 4 ∧ 2 ^ x < 2 then 1 else 0
  let length_interval_0_1 : ℝ := 1
  let length_interval_0_4 : ℝ := 4
  length_interval_0_1 / length_interval_0_4

theorem probability_subinterval :
  probability_2_pow_x_lt_2_in_0_to_4 = 1 / 4 :=
by
  sorry

end probability_subinterval_l129_129120


namespace thirteenth_number_is_9036_l129_129411

theorem thirteenth_number_is_9036 :
  ∃ (n : ℕ), n = 9036 ∧
  nth_permutation_of_digits_not_starting_with_zero {0, 3, 6, 9} 4 13 = some n := by
sorry

end thirteenth_number_is_9036_l129_129411


namespace student_ticket_price_l129_129286

theorem student_ticket_price
  (S : ℕ)
  (num_tickets : ℕ := 2000)
  (num_student_tickets : ℕ := 520)
  (price_non_student : ℕ := 11)
  (total_revenue : ℕ := 20960)
  (h : 520 * S + (2000 - 520) * 11 = 20960) :
  S = 9 :=
sorry

end student_ticket_price_l129_129286


namespace problem_statement_l129_129904

theorem problem_statement (x : ℝ) (h : 8 * x - 6 = 10) : 200 * (1 / x) = 100 := by
  sorry

end problem_statement_l129_129904


namespace exponents_to_99_l129_129609

theorem exponents_to_99 :
  (1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99) :=
sorry

end exponents_to_99_l129_129609


namespace books_bound_l129_129143

theorem books_bound (x : ℕ) (w c : ℕ) (h₀ : w = 92) (h₁ : c = 135) 
(h₂ : 92 - x = 2 * (135 - x)) :
x = 178 :=
by
  sorry

end books_bound_l129_129143


namespace probability_of_at_least_one_solves_l129_129858

theorem probability_of_at_least_one_solves (pA pB : ℝ) (hA : pA = 0.4) (hB : pB = 0.5) : 
  1 - ((1 - pA) * (1 - pB)) = 0.7 :=
by {
  rw [hA, hB],
  norm_num,
  sorry
}

end probability_of_at_least_one_solves_l129_129858


namespace max_distance_pairs_le_n_l129_129299
-- Import the, Mathlib library for mathematical constructs

-- Define the maximal distance and assumption on the number of points
variables {α : Type*} [MetricSpace α] {points : Finset α}

-- Assume the conditions given in part (a)
def condition_1 (n : ℕ) : Prop := n ≥ 3
def condition_2 (points : Finset α) (d : ℝ) : Prop := 
  ∀ (x y : α), x ∈ points → y ∈ points → dist x y ≤ d ∧ 
  ∃ (x y : α), x ∈ points → y ∈ points → dist x y = d

-- Define the main statement we aim to prove
theorem max_distance_pairs_le_n (n : ℕ) (d : ℝ) (points : Finset α) :
  condition_1 n →
  condition_2 points d →
  (points.card = n) →
  (Finset.card ((points.product points).filter (λ p, dist p.1 p.2 = d)) ≤ n) :=
sorry

end max_distance_pairs_le_n_l129_129299


namespace rhombus_area_correct_l129_129247

def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct : rhombus_area 30 18 = 270 :=
by
  -- We assert that the expected result follows from the provided formula and values.
  sorry

end rhombus_area_correct_l129_129247


namespace sum_of_squares_mod_17_l129_129316

theorem sum_of_squares_mod_17 :
  (∑ i in Finset.range 16, i^2) % 17 = 11 := 
sorry

end sum_of_squares_mod_17_l129_129316


namespace triangle_area_eq_sqrt3_given_conditions_l129_129550

open Real

variables {A B C : ℝ}

def acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2

def triangle_area (a b c : ℝ) : ℝ := (sqrt 3) / 4 * b ^ 2

theorem triangle_area_eq_sqrt3_given_conditions
  (h_acute : acute_triangle A B C)
  (h_b : 2 = 2)
  (h_B : B = π / 3)
  (h_trig_identity : sin (2 * A) + sin (A - C) - sin B = 0) :
  triangle_area 2 2 2 = sqrt 3 := sorry

end triangle_area_eq_sqrt3_given_conditions_l129_129550


namespace probability_of_yellow_l129_129624

def faces_total : ℕ := 8
def red_faces : ℕ := 4
def yellow_faces : ℕ := 3
def blue_face : ℕ := 1
def total_faces := red_faces + yellow_faces + blue_face

theorem probability_of_yellow : (yellow_faces : ℚ) / (faces_total : ℚ) = 3 / 8 := by
  have h1 : total_faces = faces_total := by sorry
  have h2 : yellow_faces = 3 := by sorry
  have h3 : faces_total = 8 := by sorry
  rw [h2, h3]
  norm_num
  sorry

end probability_of_yellow_l129_129624


namespace max_f_value_exists_triangle_area_l129_129092

noncomputable def f (x : ℝ) : ℝ := 
  real.cos x * (real.sin x - real.sqrt 3 * real.cos x)

-- Part I: Maximum value and x-values when maximum is attained
theorem max_f_value_exists : 
  ∃ (M : ℝ) (S : set ℝ), ∀ x ∈ S, f x = M ∧ (∀ y : ℝ, f y ≤ M) :=
sorry

-- Part II: Area of triangle ABC
theorem triangle_area {a b c : ℝ} (A : ℝ) (hA : f (A / 2) = -real.sqrt 3 / 2)
  (ha : a = 3) (hb : b + c = 2 * real.sqrt 3) :
  ∃ S : ℝ, S = (1 / 2) * b * c * real.sin A ∧ S = real.sqrt 3 / 4 :=
sorry

end max_f_value_exists_triangle_area_l129_129092


namespace square_window_side_length_is_20_l129_129163

noncomputable theory
open Classical

-- Definitions
def width_pane (w : ℝ) : ℝ := w
def height_pane (w : ℝ) : ℝ := 3 * w

def border_width : ℝ := 2
def panes_per_side : ℕ := 3

-- Width and height of the window including borders
def width_window (w : ℝ) : ℝ := panes_per_side * width_pane w + (panes_per_side + 1) * border_width
def height_window (w : ℝ) : ℝ := panes_per_side * height_pane w + (panes_per_side + 1) * border_width

-- The main theorem
theorem square_window_side_length_is_20 : ∃ w : ℝ, width_window w = 20 :=
by
  use 4
  have h1 : width_window 4 = 3 * 4 + 4 * 2 :=
    by simp [width_window, panes_per_side, border_width, width_pane]; norm_num
  exact h1

end square_window_side_length_is_20_l129_129163


namespace triangular_number_200_l129_129385

theorem triangular_number_200 : 
  let a_n := λ n : ℕ, n * (n + 1) / 2
  in a_n 200 = 20100 := 
by 
  let a_n := λ n : ℕ, n * (n + 1) / 2
  show a_n 200 = 20100 
  sorry

end triangular_number_200_l129_129385


namespace fifth_term_sum_of_powers_of_4_l129_129004

theorem fifth_term_sum_of_powers_of_4 :
  (4^0 + 4^1 + 4^2 + 4^3 + 4^4) = 341 := 
by
  sorry

end fifth_term_sum_of_powers_of_4_l129_129004


namespace hare_dormouse_drink_all_tea_l129_129386

theorem hare_dormouse_drink_all_tea (num_teacups : ℕ) 
  (drink_positions : ℕ → ℕ) 
  (initial_teacups : fin num_teacups → bool)
  (H : num_teacups = 30) 
  (H1 : ∀ n, drink_positions (n + 1) = (drink_positions n + 2) % num_teacups) 
  (initial_full_teacups : ∃ t1 t2 : fin num_teacups, t1 ≠ t2 ∧ initial_teacups t1 ∧ initial_teacups t2 ∧ (∀ t : fin num_teacups, t ≠ t1 ∧ t ≠ t2 → ¬initial_teacups t)) :
  ∃ sequence : ℕ → fin num_teacups × fin num_teacups,
    (∀ n, initial_teacups (sequence n).1 ∧ initial_teacups (sequence n).2) ∧ 
    (∀ t : fin num_teacups, ∃ n : ℕ, (sequence n).1 = t ∨ (sequence n).2 = t) :=
sorry

end hare_dormouse_drink_all_tea_l129_129386


namespace sum_of_three_integers_with_product_5_pow_4_l129_129261

noncomputable def a : ℕ := 1
noncomputable def b : ℕ := 5
noncomputable def c : ℕ := 125

theorem sum_of_three_integers_with_product_5_pow_4 (h : a * b * c = 5^4) : 
  a + b + c = 131 := by
  have ha : a = 1 := rfl
  have hb : b = 5 := rfl
  have hc : c = 125 := rfl
  rw [ha, hb, hc, mul_assoc] at h
  exact sorry

end sum_of_three_integers_with_product_5_pow_4_l129_129261


namespace rob_leaves_home_at_11_am_l129_129205

theorem rob_leaves_home_at_11_am :
  let travel_time_rob := 1
  let travel_time_mark := travel_time_rob * 3
  let mark_departure_time := 9
  let arrival_time := mark_departure_time + travel_time_mark
  in (arrival_time - travel_time_rob) = 11 :=
by
  sorry

end rob_leaves_home_at_11_am_l129_129205


namespace sin_alpha_sufficient_not_necessary_cos_2alpha_l129_129534

theorem sin_alpha_sufficient_not_necessary_cos_2alpha (α : ℝ) : 
  (sin α = sqrt 2 / 2) → (cos (2 * α) = 0) ∧ 
  (cos (2 * α) = 0 → sin α = sqrt 2 / 2 ∨ sin α = - sqrt 2 / 2) :=
by
  intro h
  split
  {
    rw [sin_sq], -- the formula sin^2(alpha)
    rwa [cos_double_angle, h, pow_two, mul_div_cancel', sub_self] -- calculation of sin alpha = sqrt 2 /2
  }
  sorry

end sin_alpha_sufficient_not_necessary_cos_2alpha_l129_129534


namespace closest_to_five_cm_is_thumb_l129_129742

def length_cm := ℝ

def bus_length : length_cm := 1000
def picnic_table_height : length_cm := 75
def elephant_height : length_cm := 300
def foot_length : length_cm := 25
def thumb_length_range : set length_cm := {x | 4 ≤ x ∧ x ≤ 5}

theorem closest_to_five_cm_is_thumb : 
  (∀ x ∈ {bus_length, picnic_table_height, elephant_height, foot_length},
    abs (x - 5) > abs (some thumb_length_range - 5)) :=
sorry

end closest_to_five_cm_is_thumb_l129_129742


namespace inequality_solution_set_l129_129272

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : 
  (1 / x > 3) ↔ (0 < x ∧ x < 1 / 3) := 
by 
  sorry

end inequality_solution_set_l129_129272


namespace incorrect_guess_20_l129_129671

-- Define the assumptions and conditions
def bears : Nat → String := sorry -- function that determines the color of the bear at position n
axiom bears_color_constraint : ∀ n:Nat, exists b:List String, b.length = 3 ∧ Set ("W" "B" "Bk") = List.toSet b ∧ 
  List.all (List.sublist b (n, n+1, n+2) bears = fun c=> c = "W" or c = "B" or c = "Bk") 

-- Iskander's guesses
def guess1 := (2, "W")
def guess2 := (20, "B")
def guess3 := (400, "Bk")
def guess4 := (600, "B")
def guess5 := (800, "W")

-- Function to check the bear at each position
def check_bear (n:Nat) : String := sorry

-- Iskander's guess correctness, exactly one is wrong
axiom one_wrong : count (check_bear 2 =="W") 
                         + count (check_bear 20 == "B") 
                         + count (check_bear 400 =="Bk") 
                         + count (check_bear 600 =="B") 
                         + count (check_bear 800 =="W") = 4

-- Prove that the guess for the 20th bear is incorrect
theorem incorrect_guess_20 : ∀ {n:Nat} (h : n=20), (check_bear n != "B") := sorry

end incorrect_guess_20_l129_129671


namespace area_enclosed_by_region_l129_129309

theorem area_enclosed_by_region : ∀ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) → (π * (4 ^ 2) = 16 * π) :=
by
  intro x y h
  sorry

end area_enclosed_by_region_l129_129309


namespace min_value_of_a_l129_129099

/-- Given the inequality |x - 1| + |x + a| ≤ 8, prove that the minimum value of a is -9 -/

theorem min_value_of_a (a : ℝ) (h : ∀ x : ℝ, |x - 1| + |x + a| ≤ 8) : a = -9 :=
sorry

end min_value_of_a_l129_129099


namespace min_value_sqrt_expression_l129_129076

theorem min_value_sqrt_expression (x y : ℝ) (h : 6 * x + 8 * y - 1 = 0) : 
  sqrt (x^2 + y^2 - 2 * y + 1) ≥ 7 / 10 :=
sorry

end min_value_sqrt_expression_l129_129076


namespace polar_line_eq_l129_129462

theorem polar_line_eq 
  (r θ : ℝ) 
  (h_point : r = 2 ∧ θ = Real.pi / 4) 
  (h_parallel : ∀ θ', sin θ' = sin θ) :
  r * sin θ = Real.sqrt 2 := 
sorry

end polar_line_eq_l129_129462


namespace closed_form_a_0_value_for_increasing_seq_l129_129372

noncomputable def a_seq (a_0 : ℝ) : ℕ → ℝ
| 0     := a_0
| (n+1) := 2^n - 3 * a_seq n

theorem closed_form (a_0 : ℝ) : 
  ∃ a_closed_form : ℕ → ℝ, 
  (∀ n, a_seq a_0 n = a_closed_form n)
  ∧ 
  (∀ n, a_closed_form n = (- 3)^n * a_0 + 1/5 * ((- 3)^n - 2^n)) :=
sorry 

theorem a_0_value_for_increasing_seq : 
  ∃ a_0 : ℝ, (∀ n, a_0 > -1 / 15 ∧ (a_seq a_0 (n + 1) > a_seq a_0 n)) :=
sorry

end closed_form_a_0_value_for_increasing_seq_l129_129372


namespace calculate_x_l129_129829

def percentage (p : ℚ) (n : ℚ) := (p / 100) * n

theorem calculate_x : 
  (percentage 47 1442 - percentage 36 1412) + 65 = 234.42 := 
by 
  sorry

end calculate_x_l129_129829


namespace only_one_true_l129_129870

-- Definitions based on conditions
def line := Type
def plane := Type
def parallel (m n : line) : Prop := sorry
def perpendicular (m n : line) : Prop := sorry
def subset (m : line) (alpha : plane) : Prop := sorry

-- Propositions derived from conditions
def prop1 (m n : line) (alpha : plane) : Prop := parallel m alpha ∧ parallel n alpha → ¬ parallel m n
def prop2 (m n : line) (alpha : plane) : Prop := perpendicular m alpha ∧ perpendicular n alpha → parallel m n
def prop3 (m n : line) (alpha beta : plane) : Prop := parallel alpha beta ∧ subset m alpha ∧ subset n beta → parallel m n
def prop4 (m n : line) (alpha beta : plane) : Prop := perpendicular alpha beta ∧ perpendicular m n ∧ perpendicular m alpha → perpendicular n beta

-- Theorem statement that only one proposition is true
theorem only_one_true (m n : line) (alpha beta : plane) :
  (prop1 m n alpha = false) ∧
  (prop2 m n alpha = true) ∧
  (prop3 m n alpha beta = false) ∧
  (prop4 m n alpha beta = false) :=
by sorry

end only_one_true_l129_129870


namespace similar_iff_condition_l129_129615

-- Define the similarity of triangles and the necessary conditions.
variables {α : Type*} [LinearOrderedField α]
variables (a b c a' b' c' : α)

-- Statement of the problem in Lean 4
theorem similar_iff_condition : 
  (∃ z w : α, a' = a * z + w ∧ b' = b * z + w ∧ c' = c * z + w) ↔ 
  (a' * (b - c) + b' * (c - a) + c' * (a - b) = 0) :=
sorry

end similar_iff_condition_l129_129615


namespace height_of_water_is_50cube2_l129_129275

variable (r h : ℕ) (Vfilling : ℚ)
variable (a b : ℕ)

def water_tank := cone r h
def water_filled_tank := cone (r * (h / r)^(1/3)) (h / 2)

noncomputable def height_of_water := 50 * (2)^(1/3)

-- The height of the water in the tank, expressed as a lambda where a = 50 and b = 2
theorem height_of_water_is_50cube2 :
    water_tank = cone 20 100 → 
    water_filled_tank = cone 20 (100 / 2) →
    a = 50 → b = 2 →
    height_of_water = a * (b)^(1/3)
:=
  by
    sorry

end height_of_water_is_50cube2_l129_129275


namespace find_wrong_guess_l129_129692

-- Define the three colors as an inductive type.
inductive Color
| white
| brown
| black

-- Define the bears as a list of colors.
def bears (n : ℕ) : Type := list Color

-- Define the conditions: 
-- There are 1000 bears and each tuple of 3 consecutive bears has all three colors.
def valid_bears (b : bears 1000) : Prop :=
  ∀ i : ℕ, i + 2 < 1000 → 
    ∃ c1 c2 c3 : Color, 
      c1 ∈ b.nth i ∧ c2 ∈ b.nth (i+1) ∧ c3 ∈ b.nth (i+2) ∧ 
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3

-- Define Iskander's guesses.
def guesses (b : bears 1000) : Prop :=
  b.nth 1 = some Color.white ∧
  b.nth 19 = some Color.brown ∧
  b.nth 399 = some Color.black ∧
  b.nth 599 = some Color.brown ∧
  b.nth 799 = some Color.white

-- Prove that exactly one of Iskander's guesses is wrong.
def wrong_guess (b : bears 1000) : Prop :=
  (b.nth 19 ≠ some Color.brown) ∧
  valid_bears b ∧
  guesses b →
  ∃ i, i ∈ {1, 19, 399, 599, 799} ∧ (b.nth i ≠ some Color.white ∧ b.nth i ≠ some Color.brown ∧ b.nth i ≠ some Color.black)

theorem find_wrong_guess : 
  ∀ b : bears 1000, 
  valid_bears b → guesses b → wrong_guess b :=
  by
  intros b vb gs
  sorry

end find_wrong_guess_l129_129692


namespace pyramid_volume_l129_129628

theorem pyramid_volume :
  ∃(V : ℝ),
    (∃ (TRIANGLE_BASE : Type) (hypotenuse : TRIANGLE_BASE) (acute_angle : TRIANGLE_BASE),
      hypotenuse = 6 ∧ acute_angle = 15.0 ∧
      ∃ (lateral_edge_inclination : TRIANGLE_BASE),
        lateral_edge_inclination = 45.0 ∧
        V = 4.5) :=
sorry

end pyramid_volume_l129_129628


namespace isosceles_triangle_base_length_l129_129236

-- Definitions based on the conditions
def congruent_side : Nat := 7
def perimeter : Nat := 23

-- Statement to prove
theorem isosceles_triangle_base_length :
  let b := perimeter - 2 * congruent_side in b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l129_129236


namespace number_of_solution_pairs_l129_129015

theorem number_of_solution_pairs (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 4 * x + 7 * y = 600) : 
  (∃! (x y : ℕ), 4 * x + 7 * y = 600 ∧ x > 0 ∧ y > 0) :=
begin
  sorry
end

end number_of_solution_pairs_l129_129015


namespace divisibility_by_seven_l129_129343

theorem divisibility_by_seven : (∃ k : ℤ, (-8)^2019 + (-8)^2018 = 7 * k) :=
sorry

end divisibility_by_seven_l129_129343


namespace correct_option_is_C_l129_129748

theorem correct_option_is_C : 
  (∀ (x : ℝ), sqrt 16 ≠ ±4) ∧
  (∀ (y : ℝ), sqrt ((-3)^2) ≠ -3) ∧
  (∀ (z : ℝ), ±sqrt 81 = ±9) ∧
  (∀ (w : ℝ), sqrt (-4) ≠ 2) → 
  (true) :=
sorry

end correct_option_is_C_l129_129748


namespace Vasya_missed_lessons_64_impossible_l129_129290

theorem Vasya_missed_lessons_64_impossible : 
  (∀ (M T W Th F : ℕ), 
   (M = 1) ∧ (T = 2) ∧ (W = 3) ∧ (Th = 4) ∧ (F = 5) →
   let total_lessons_per_week := M + T + W + Th + F in
   let full_weeks := 4 in
   let extra_days := 2 in
   (∀ (extra_1 extra_2 : ℕ), 
    (extra_1 ∈ ({0, 1, 2, 3, 4, 5} : set ℕ)) ∧ (extra_2 ∈ ({0, 1, 2, 3, 4, 5} : set ℕ)) →
    let total_lessons := (full_weeks * total_lessons_per_week) + extra_1 + extra_2 in
    total_lessons ≠ 64)) :=
by
  intros M T W Th F h1 h2
  let total_lessons_per_week := M + T + W + Th + F
  let full_weeks := 4
  let extra_days := 2
  intros extra_1 extra_2 h3 h4
  let total_lessons := (full_weeks * total_lessons_per_week) + extra_1 + extra_2
  sorry

end Vasya_missed_lessons_64_impossible_l129_129290


namespace right_triangle_area_midpoints_l129_129554

/-- In right triangle PQR, points M and N are midpoints of hypotenuse PQ and leg PR,
    respectively. If the area of triangle PQR is 32 square units, then the area of
    triangle QMN is 8 square units. -/
theorem right_triangle_area_midpoints (P Q R M N : Point)
  (h1 : is_right_triangle P Q R)
  (h2 : M = midpoint P Q)
  (h3 : N = midpoint P R)
  (h4 : area (triangle P Q R) = 32) :
  area (triangle Q M N) = 8 :=
sorry

end right_triangle_area_midpoints_l129_129554


namespace train_A_speed_60_l129_129248

noncomputable def speed_of_train_A (distance_AB : ℝ) (start_time_A : ℝ) (start_time_B : ℝ) (meet_time : ℝ) (speed_B : ℝ) :=
let time_A := meet_time - start_time_A in
let time_B := meet_time - start_time_B in
let distance_B := time_B * speed_B in
let distance_A := distance_AB - distance_B in
distance_A / time_A

theorem train_A_speed_60 :
  speed_of_train_A 330 8 9 11 75 = 60 :=
sorry

end train_A_speed_60_l129_129248


namespace find_wrong_guess_l129_129691

-- Define the three colors as an inductive type.
inductive Color
| white
| brown
| black

-- Define the bears as a list of colors.
def bears (n : ℕ) : Type := list Color

-- Define the conditions: 
-- There are 1000 bears and each tuple of 3 consecutive bears has all three colors.
def valid_bears (b : bears 1000) : Prop :=
  ∀ i : ℕ, i + 2 < 1000 → 
    ∃ c1 c2 c3 : Color, 
      c1 ∈ b.nth i ∧ c2 ∈ b.nth (i+1) ∧ c3 ∈ b.nth (i+2) ∧ 
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3

-- Define Iskander's guesses.
def guesses (b : bears 1000) : Prop :=
  b.nth 1 = some Color.white ∧
  b.nth 19 = some Color.brown ∧
  b.nth 399 = some Color.black ∧
  b.nth 599 = some Color.brown ∧
  b.nth 799 = some Color.white

-- Prove that exactly one of Iskander's guesses is wrong.
def wrong_guess (b : bears 1000) : Prop :=
  (b.nth 19 ≠ some Color.brown) ∧
  valid_bears b ∧
  guesses b →
  ∃ i, i ∈ {1, 19, 399, 599, 799} ∧ (b.nth i ≠ some Color.white ∧ b.nth i ≠ some Color.brown ∧ b.nth i ≠ some Color.black)

theorem find_wrong_guess : 
  ∀ b : bears 1000, 
  valid_bears b → guesses b → wrong_guess b :=
  by
  intros b vb gs
  sorry

end find_wrong_guess_l129_129691


namespace constant_term_binomial_l129_129080

noncomputable def integral_value : ℝ :=
  ∫ x in 0..(real.pi / 2), 6 * real.sin x

theorem constant_term_binomial (n : ℝ) (h : n = integral_value) :
  (∑ r in finset.range 7, nat.choose 6 r * (x^(6-r) * ((-2 / x^2)^r))).filter (λ term, term = 0) = 60 :=
by sorry

end constant_term_binomial_l129_129080


namespace hyperbola_foci_distance_l129_129045

def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

theorem hyperbola_foci_distance :
  (∀ (y x : ℝ), (y^2 / 75) - (x^2 / 11) = 1) →
  distance_between_foci (Real.sqrt 75) (Real.sqrt 11) = 2 * Real.sqrt 86 :=
by
  intro h
  sorry

end hyperbola_foci_distance_l129_129045


namespace area_enclosed_by_region_l129_129308

theorem area_enclosed_by_region : ∀ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) → (π * (4 ^ 2) = 16 * π) :=
by
  intro x y h
  sorry

end area_enclosed_by_region_l129_129308


namespace impossible_even_product_of_n_and_m_l129_129114

theorem impossible_even_product_of_n_and_m (n m : ℤ) (h : odd (n^3 + m^3)) : ¬even (n * m) :=
by sorry

end impossible_even_product_of_n_and_m_l129_129114


namespace prob_rain_next_day_given_today_rain_l129_129107

variable (P_rain : ℝ) (P_rain_2_days : ℝ)
variable (p_given_rain : ℝ)

-- Given conditions
def condition_P_rain : Prop := P_rain = 1/3
def condition_P_rain_2_days : Prop := P_rain_2_days = 1/5

-- The question to prove
theorem prob_rain_next_day_given_today_rain (h1 : condition_P_rain P_rain) (h2 : condition_P_rain_2_days P_rain_2_days) :
  p_given_rain = 3/5 :=
by
  sorry

end prob_rain_next_day_given_today_rain_l129_129107


namespace find_least_positive_integer_n_l129_129457

theorem find_least_positive_integer_n :
  (∃ n : ℕ, ∀ (i : ℕ), (30 ≤ i ∧ i ≤ 88) → (∑ j in finset.range (59), (1 / (real.sin ((30 + 2 * j) * real.pi / 180) * real.sin ((31 + 2 * j) * real.pi / 180)))) = 1 / (real.sin (n * real.pi / 180)) ∧ 1 ≤ n)
  ∧ (∀ m : ℕ, (∑ j in finset.range (59), (1 / (real.sin ((30 + 2 * j) * real.pi / 180) * real.sin ((31 + 2 * j) * real.pi / 180)))) ≠ 1 / (real.sin (m * real.pi / 180)) ∨ 1 > m) :=
by
  sorry

end find_least_positive_integer_n_l129_129457


namespace find_xyz_l129_129539

variable (x y z : ℝ)

theorem find_xyz (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * (y + z) = 168)
  (h2 : y * (z + x) = 180)
  (h3 : z * (x + y) = 192) : x * y * z = 842 :=
sorry

end find_xyz_l129_129539


namespace rectangle_area_l129_129214

namespace ProofExample

-- Define the side length of the smaller squares
constant side_length : ℕ := 1

-- Define the variables for each square
constant A : ℕ
constant B : ℕ
constant C : ℕ
constant D : ℕ
constant E : ℕ

-- Define the side lengths based on the problem conditions
axiom side_length_A : A = side_length
axiom side_length_B : B = A
axiom side_length_C : C = 2 * A - 1
axiom side_length_D : D = A + 2
axiom side_length_E : E = A + 3

-- Equation from the problem stating the properties of the rectangle
axiom rectangle_sides : A + (2 * A - 1) = (A + 2) + (A + 3)

-- Solve for the side length of the squares.
axiom side_length_solved : A = 6

-- Calculate the dimensions of the rectangle
def horizontal_dimension : ℕ := 6 + (2 * 6 - 1)
def vertical_dimension : ℕ := 6 + 6 + (6 + 2)

-- Prove that the area equals to 340 cm²
theorem rectangle_area :
  (horizontal_dimension * vertical_dimension) = 340 := by 
  sorry

end ProofExample

end rectangle_area_l129_129214


namespace largest_of_w_l129_129902

variable {x y z w : ℝ}

namespace MathProof

theorem largest_of_w
  (h1 : x + 3 = y - 1)
  (h2 : x + 3 = z + 5)
  (h3 : x + 3 = w - 2) :  
  w > y ∧ w > x ∧ w > z :=
by
  sorry

end MathProof

end largest_of_w_l129_129902


namespace trajectory_equation_l129_129085

theorem trajectory_equation (x y : ℝ) (M O A : ℝ × ℝ)
    (hO : O = (0, 0)) (hA : A = (3, 0))
    (h_ratio : dist M O / dist M A = 1 / 2) : 
    x^2 + y^2 + 2 * x - 3 = 0 :=
by
  -- Definition of points
  let M := (x, y)
  exact sorry

end trajectory_equation_l129_129085


namespace find_value_divided_by_4_l129_129347

theorem find_value_divided_by_4 (x : ℝ) (h : 812 / x = 25) : x / 4 = 8.12 :=
by
  sorry

end find_value_divided_by_4_l129_129347


namespace evaluate_f_neg3_5_l129_129839

-- Condition: [x] represents the largest integer not greater than x
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Definition of the function f
def f (x : ℝ) : ℝ := 3 * ((floor x + 3) : ℝ)^2 - 2

-- Proof that f(-3.5) = 1
theorem evaluate_f_neg3_5 : f (-3.5) = 1 := by
  sorry

end evaluate_f_neg3_5_l129_129839


namespace nitin_ranks_from_last_l129_129193

def total_students : ℕ := 75

def math_rank_start : ℕ := 24
def english_rank_start : ℕ := 18

def rank_from_last (total : ℕ) (rank_start : ℕ) : ℕ :=
  total - rank_start + 1

theorem nitin_ranks_from_last :
  rank_from_last total_students math_rank_start = 52 ∧
  rank_from_last total_students english_rank_start = 58 :=
by
  sorry

end nitin_ranks_from_last_l129_129193


namespace incorrect_proposition_l129_129789

theorem incorrect_proposition :
  (¬ (∀ (Q : Type) [quad : Quadrilateral Q] (d1 d2 : Diagonal Q),
      perpendicular d1 d2 ∧ length d1 = length d2 → square Q))
 ∧ (∀ (R : Type) [rhomb : Rhombus R] (d : Diagonal R), bisects d ∠ bisected_angles R)
 ∧ (∀ (Q : Type) [quad1 : Quadrilateral Q] [quad2 : Quadrilateral (midpoint_quad Q)], parallelogram quad2)
 ∧ (∀ (T : Type) [trap : IsoscelesTrapezoid T] (d1 d2 : Diagonal T), length d1 = length d2) :=
sorry

end incorrect_proposition_l129_129789


namespace isosceles_triangle_base_length_l129_129243

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l129_129243


namespace incorrect_guess_20_l129_129674

-- Define the assumptions and conditions
def bears : Nat → String := sorry -- function that determines the color of the bear at position n
axiom bears_color_constraint : ∀ n:Nat, exists b:List String, b.length = 3 ∧ Set ("W" "B" "Bk") = List.toSet b ∧ 
  List.all (List.sublist b (n, n+1, n+2) bears = fun c=> c = "W" or c = "B" or c = "Bk") 

-- Iskander's guesses
def guess1 := (2, "W")
def guess2 := (20, "B")
def guess3 := (400, "Bk")
def guess4 := (600, "B")
def guess5 := (800, "W")

-- Function to check the bear at each position
def check_bear (n:Nat) : String := sorry

-- Iskander's guess correctness, exactly one is wrong
axiom one_wrong : count (check_bear 2 =="W") 
                         + count (check_bear 20 == "B") 
                         + count (check_bear 400 =="Bk") 
                         + count (check_bear 600 =="B") 
                         + count (check_bear 800 =="W") = 4

-- Prove that the guess for the 20th bear is incorrect
theorem incorrect_guess_20 : ∀ {n:Nat} (h : n=20), (check_bear n != "B") := sorry

end incorrect_guess_20_l129_129674


namespace geometric_sequence_sum_l129_129147

-- Definitions for the geometric sequence
def a (n : ℕ) (q : ℝ) : ℝ := 1 * q^(n-1)  -- since a₁ = 1

-- Definition for the sum of the first n terms of a geometric sequence
def S (n : ℕ) (q : ℝ) : ℝ := if q = 1 then n else (1 - q^n) / (1 - q)

-- Given information
def q := 2
def a₁ := 1
def a₄ := 8  -- one of the given conditions

-- Statement to prove
theorem geometric_sequence_sum : a₁ = 1 ∧ a₄ = 8 → S 6 q = 63 := by
  sorry

end geometric_sequence_sum_l129_129147


namespace minimal_intersection_cardinality_l129_129831

theorem minimal_intersection_cardinality (a b m : ℕ) (A : set ℕ) 
(h : ∀ n, a * n ∈ A ∨ b * n ∈ A) :
  ∃ min_card, min_card = 
    if a = 1 ∧ b = 1 then m 
    else ∑ i in (finset.range (m.nat_abs)), (-1)^(i+1) * (m / (max a b)^i) :=
sorry

end minimal_intersection_cardinality_l129_129831


namespace cost_of_tax_free_items_is_20_l129_129417

-- Define the given conditions
def total_spent : ℝ := 25
def sales_tax_paise : ℝ := 30
def sales_tax_rate : ℝ := 6 / 100  -- Convert 6% to a decimal

-- Convert paise to rupees
def sales_tax : ℝ := sales_tax_paise / 100

-- Calculate the cost of taxable items
def cost_of_taxable_items : ℝ := sales_tax / sales_tax_rate

-- Calculate the cost of tax-free items
def cost_of_tax_free_items : ℝ := total_spent - cost_of_taxable_items

-- Prove the cost of tax-free items is 20 rupees
theorem cost_of_tax_free_items_is_20 : cost_of_tax_free_items = 20 := by
  sorry

end cost_of_tax_free_items_is_20_l129_129417


namespace probability_gcd_2_l129_129404

noncomputable def choose_two_and_check_gcd (s : Set ℕ) (n m : ℕ) : ℕ :=
  if n ≠ m ∧ n ∈ s ∧ m ∈ s ∧ gcd n m = 2 then 1 else 0

noncomputable def count_pairs_with_gcd_2 (s : Set ℕ) : ℕ :=
  s.to_finset.sum (λ n, s.to_finset.sum (λ m, choose_two_and_check_gcd s n m)) / 2

noncomputable def total_pairs (s : Set ℕ) : ℕ :=
  s.to_finset.card * (s.to_finset.card - 1) / 2

theorem probability_gcd_2 (s : Set ℕ) (h : s = {1, 2, 3, 4, 5, 6, 7, 8}) : 
  (count_pairs_with_gcd_2 s) / (total_pairs s) = 3 / 14 := 
by sorry

end probability_gcd_2_l129_129404


namespace k_value_and_p_existence_l129_129512

theorem k_value_and_p_existence (k : ℤ) (p : ℝ) :
  (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^(-k^2 + k + 2)) →
  (f 2 < f 3) →
  (k = 0 ∨ k = 1) ∧ 
  (∃ p > 0, (∀ x ∈ ["-1,2"], g x = 1 - p * f x + (2 * p - 1) * x) ∧
              (g (-1) = -4) ∧ (g 2 = 17 / 8)) :=
begin
  sorry
end

end k_value_and_p_existence_l129_129512


namespace players_joined_l129_129704

theorem players_joined (num_friends : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) (h_num_friends : num_friends = 2) (h_lives_per_player : lives_per_player = 6) (h_total_lives : total_lives = 24) : 
  (total_lives / lives_per_player) - num_friends = 2 :=
by
  rw [h_num_friends, h_lives_per_player, h_total_lives]
  norm_num
  sorry

end players_joined_l129_129704


namespace percent_of_democrats_voting_for_A_l129_129921

variables (V : ℕ) (D : ℚ)

-- Conditions
def is_democrats (V : ℕ) : ℚ := 0.70 * V
def is_republicans (V : ℕ) : ℚ := 0.30 * V
def votes_from_democrats (D : ℚ) (V : ℕ) : ℚ := D * (is_democrats V)
def votes_from_republicans (V : ℕ) : ℚ := 0.30 * (is_republicans V)
def total_votes_for_A (D : ℚ) (V : ℕ) : ℚ := votes_from_democrats D V + votes_from_republicans V

-- Given that candidate A is expected to receive 65% of the total votes
axiom total_votes_for_A_eq : total_votes_for_A D V = 0.65 * V

-- Prove what percent of the registered voters who are Democrats are expected to vote for candidate A
theorem percent_of_democrats_voting_for_A (V : ℕ) (D : ℚ) (h : total_votes_for_A_eq D V) : D = 0.8 :=
sorry

end percent_of_democrats_voting_for_A_l129_129921


namespace sum_of_squares_to_15_mod_17_eq_10_l129_129319

def sum_of_squares_modulo_17 : ℕ :=
  let sum := (Finset.sum (Finset.range 16) (λ n, n^2 % 17)) in
  sum % 17

theorem sum_of_squares_to_15_mod_17_eq_10 : sum_of_squares_modulo_17 = 10 :=
  sorry

end sum_of_squares_to_15_mod_17_eq_10_l129_129319


namespace triangle_area_l129_129361

/-- The area of the triangle enclosed by a line with slope -1/2 passing through (2, -3) and the coordinate axes is 4. -/
theorem triangle_area {l : ℝ → ℝ} (h1 : ∀ x, l x = -1/2 * x + b)
  (h2 : l 2 = -3) : 
  ∃ (A : ℝ) (B : ℝ), 
  ((l 0 = B) ∧ (l A = 0) ∧ (A ≠ 0) ∧ (B ≠ 0)) ∧
  (1/2 * |A| * |B| = 4) := 
sorry

end triangle_area_l129_129361


namespace solution_set_f_g_inequality_l129_129590

variable {R : Type*} [LinearOrderedField R]

/-- Define f and g as odd and even functions respectively. -/
def odd_function (f : R → R) :=
  ∀ x, f (-x) = -f x

def even_function (g : R → R) :=
  ∀ x, g (-x) = g x

theorem solution_set_f_g_inequality {f g : R → R}
  (f_odd : odd_function f)
  (g_even : even_function g)
  (ineq : ∀ x, x < 0 → f' x * g x + f x * g' x > 0)
  (g_zero_at_3 : g 3 = 0) :
  ∀ x, f x * g x < 0 ↔ x ∈ set.Ioo (-∞) (-3) ∪ set.Ioo 0 3 :=
sorry

end solution_set_f_g_inequality_l129_129590


namespace sum_eq_prod_nat_numbers_l129_129273

theorem sum_eq_prod_nat_numbers (A B C D E F : ℕ) :
  A + B + C + D + E + F = A * B * C * D * E * F →
  (A = 0 ∧ B = 0 ∧ C = 0 ∧ D = 0 ∧ E = 0 ∧ F = 0) ∨
  (A = 1 ∧ B = 1 ∧ C = 1 ∧ D = 1 ∧ E = 2 ∧ F = 6) :=
by
  sorry

end sum_eq_prod_nat_numbers_l129_129273


namespace not_partitionable_1_to_15_l129_129988

theorem not_partitionable_1_to_15 :
  ∀ (A B : Finset ℕ), (∀ x ∈ A, x ∈ Finset.range 16) →
    (∀ x ∈ B, x ∈ Finset.range 16) →
    A.card = 2 → B.card = 13 →
    A ∪ B = Finset.range 16 →
    ¬(A.sum id = B.prod id) :=
by
  -- To be proved
  sorry

end not_partitionable_1_to_15_l129_129988


namespace problem_solution_l129_129012

noncomputable def F (n : ℕ) : ℕ :=
if odd n then 3 * n + 5 else
  let m := n in
  let k := nat.find (λ k : ℕ, odd (n / 2^k)) in
  n / 2^k

def F_iter : ℕ → ℕ → ℕ
| n 0 := n
| n (m + 1) := F (F_iter n m)

theorem problem_solution (n : ℕ) (h : n = 9) :
  F_iter n 2017 = 8 :=
by {
  -- proof goes here
  sorry
}

end problem_solution_l129_129012


namespace triangle_inequality_for_n6_l129_129036

variables {a b c : ℝ} {n : ℕ}
open Real

-- Define the main statement as a theorem
theorem triangle_inequality_for_n6 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_for_n6_l129_129036


namespace solution_set_equivalence_l129_129809

noncomputable def f : ℝ → ℝ := sorry

axiom f_derivative : ∀ x : ℝ, deriv f x > 1 - f x
axiom f_at_0 : f 0 = 3

theorem solution_set_equivalence :
  {x : ℝ | (Real.exp x) * f x > (Real.exp x) + 2} = {x : ℝ | x > 0} :=
by sorry

end solution_set_equivalence_l129_129809


namespace toms_balloons_l129_129288

-- Define the original number of balloons that Tom had
def original_balloons : ℕ := 30

-- Define the number of balloons that Tom gave to Fred
def balloons_given_to_Fred : ℕ := 16

-- Define the number of balloons that Tom has now
def balloons_left : ℕ := original_balloons - balloons_given_to_Fred

-- The theorem to prove
theorem toms_balloons : balloons_left = 14 := 
by
  -- The proof steps would go here
  sorry

end toms_balloons_l129_129288


namespace sum_modified_riemann_zeta_l129_129477

noncomputable def modified_riemann_zeta (x : ℝ) : ℝ :=
  ∑' n, (1 / (2 * n : ℕ)^x)

theorem sum_modified_riemann_zeta :
  (∑' k in finset.Ico 2 (⊤:finset ℕ), {modified_riemann_zeta (2 * k - 1 : ℝ)}) = 1 / 8 :=
sorry

end sum_modified_riemann_zeta_l129_129477


namespace general_formula_sequence_smallest_positive_integer_l129_129843

variable seq : ℕ → ℝ

-- Conditions for the first part
variable (mono_incr_geometric : ∀ n, seq (n + 1) / seq n = 2)
variable (a2_a3_a4_eq_28 : seq 2 + seq 3 + seq 4 = 28)
variable (arithmetic_mean_condition : seq 3 + 2 = (seq 2 + seq 4) / 2)

theorem general_formula_sequence (n : ℕ)
  (mono_incr_geometric : ∀ n, seq (n + 1) / seq n = 2)
  (a2_a3_a4_eq_28 : seq 2 + seq 3 + seq 4 = 28)
  (arithmetic_mean_condition : seq 3 + 2 = (seq 2 + seq 4) / 2) :
  seq n = 2 ^ n := sorry

-- Conditions for the second part
def bn (n : ℕ) : ℝ := seq n * Real.log (1 / 2) (seq n)
def Sn (n : ℕ) : ℝ := ∑ i in Finset.range n, bn (i + 1)

theorem smallest_positive_integer (Sn : ℕ → ℝ) (seq : ℕ → ℝ)
  (bn_def : ∀ n, seq n * Real.log (1 / 2) (seq n) = -n * 2^n)
  (Sn_def : ∀ n, Sn n = ∑ i in Finset.range n, bn (i + 1))
  (seq_def : ∀ n, seq n = 2 ^ n)
  :
  ∃ n : ℕ, Sn (n + 1) + (n + 1) * 2 ^ (n + 2) > 50 ∧ ∀ k < n + 1, Sn (k + 1) + (k + 1) * 2 ^ (k + 2) ≤ 50 := sorry

end general_formula_sequence_smallest_positive_integer_l129_129843


namespace knicks_win_tournament_probability_l129_129136

noncomputable def knicks_win_probability : ℚ :=
  let knicks_win_proba := 2 / 5
  let heat_win_proba := 3 / 5
  let first_4_games_scenarios := 6 * (knicks_win_proba^2 * heat_win_proba^2)
  first_4_games_scenarios * knicks_win_proba

theorem knicks_win_tournament_probability :
  knicks_win_probability = 432 / 3125 :=
by
  sorry

end knicks_win_tournament_probability_l129_129136


namespace gcd_18222_24546_66364_eq_2_l129_129818

/-- Definition of three integers a, b, c --/
def a : ℕ := 18222 
def b : ℕ := 24546
def c : ℕ := 66364

/-- Proof of the gcd of the three integers being 2 --/
theorem gcd_18222_24546_66364_eq_2 : Nat.gcd (Nat.gcd a b) c = 2 := by
  sorry

end gcd_18222_24546_66364_eq_2_l129_129818


namespace exists_same_digit_sum_in_arith_prog_l129_129986

def digit_sum (n : ℕ) : ℕ := 
  n.digits.sum

theorem exists_same_digit_sum_in_arith_prog (A d : ℕ) (h : d > 0) :
  ∃ (k l : ℕ), k ≠ l ∧ digit_sum (A + k * d) = digit_sum (A + l * d) :=
by
  sorry

end exists_same_digit_sum_in_arith_prog_l129_129986


namespace n_does_not_divide_ak_a1_minus_1_l129_129945

theorem n_does_not_divide_ak_a1_minus_1
  (n : ℕ) 
  (h_pos : n > 0)
  (k : ℕ)
  (h_k : k ≥ 2)
  (a : Fin k → ℕ)
  (h_distinct : Function.Injective a)
  (h_range : ∀ i, a i ∈ Finset.range (n + 1))
  (h_div : ∀ i : Fin (k - 1), n ∣ (a i * (a (i + 1) - 1))) :
  ¬ (n ∣ (a (Fin.mk (k - 1) (by simp [h_k])) * (a 0 - 1))) :=
by
  sorry

end n_does_not_divide_ak_a1_minus_1_l129_129945


namespace L_plus_R_equals_38_l129_129007

-- Define a regular 18-gon
def regular_18gon := sorry -- geometry definition is abstractly handled

-- Define number of lines of symmetry, L
def L : ℕ := 18

-- Define the smallest positive angle for rotational symmetry, R
def R : ℕ := 360 / 18

-- Prove that L + R equals 38
theorem L_plus_R_equals_38 (regular_18gon : RegularPolygon 18) :
  L + R = 38 := by
  have h_L : L = 18 := by rfl
  have h_R : R = 20 := by rfl
  rw [h_L, h_R]
  norm_num
  sorry

end L_plus_R_equals_38_l129_129007


namespace volume_of_intersection_is_zero_l129_129734

-- Definition of the regions
def region1 (x y z : ℝ) : Prop := abs x + abs y + abs z ≤ 1
def region2 (x y z : ℝ) : Prop := abs x + abs y + abs (z - 2) ≤ 1

-- Volume of the intersection of region1 and region2
theorem volume_of_intersection_is_zero : 
  let volume_intersection : ℝ := 0 
  in volume_intersection = 0 := 
by
  sorry

end volume_of_intersection_is_zero_l129_129734


namespace dave_non_working_games_l129_129418

def total_games : ℕ := 10
def price_per_game : ℕ := 4
def total_earnings : ℕ := 32

theorem dave_non_working_games : (total_games - (total_earnings / price_per_game)) = 2 := by
  sorry

end dave_non_working_games_l129_129418


namespace parallel_line_through_P_perpendicular_line_through_P_l129_129084

-- Define the line equations
def line1 (x y : ℝ) : Prop := 2 * x + y - 5 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y = 0
def line_l (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Define the equations for parallel and perpendicular lines through point P
def parallel_line (x y : ℝ) : Prop := 3 * x - y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x + 3 * y - 5 = 0

-- Define the point P where the lines intersect
def point_P : (ℝ × ℝ) := (2, 1)

-- Assert the proof statements
theorem parallel_line_through_P : parallel_line point_P.1 point_P.2 :=
by 
  -- proof content skipped with sorry
  sorry
  
theorem perpendicular_line_through_P : perpendicular_line point_P.1 point_P.2 :=
by 
  -- proof content skipped with sorry
  sorry

end parallel_line_through_P_perpendicular_line_through_P_l129_129084


namespace cylinder_diameter_l129_129651

theorem cylinder_diameter (height_sphere height_cylinder : ℝ) (radius_sphere: ℝ) 
  (hs_height: height_sphere = height_cylinder) : 
  height_sphere = 6 → radius_sphere = 3 → 
  2 * (36 * radius_sphere * height_cylinder / (2 * real.pi * radius_sphere * height_cylinder)) = 6 :=
by
  intros h_cylinder h_sphere r_sphere
  sorry

end cylinder_diameter_l129_129651


namespace directrix_of_parabola_l129_129450

noncomputable def parabola_directrix (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / 16

theorem directrix_of_parabola :
  let d := parabola_directrix y in d = -(1 / 2) := sorry

end directrix_of_parabola_l129_129450


namespace icosahedron_inscribed_in_cube_l129_129980

theorem icosahedron_inscribed_in_cube (a m : ℝ) (points_on_faces : Fin 6 → Fin 2 → ℝ × ℝ × ℝ) :
  (∃ points : Fin 12 → ℝ × ℝ × ℝ, 
   (∀ i : Fin 12, ∃ j : Fin 6, (points i).fst = (points_on_faces j 0).fst ∨ (points i).fst = (points_on_faces j 1).fst) ∧
   ∃ segments : Fin 12 → Fin 12 → ℝ, 
   (∀ i j : Fin 12, (segments i j) = m ∨ (segments i j) = a)) →
  a^2 - a*m - m^2 = 0 := sorry

end icosahedron_inscribed_in_cube_l129_129980


namespace isosceles_base_length_l129_129228

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l129_129228


namespace isosceles_triangle_base_length_l129_129232

-- Definitions based on the conditions
def congruent_side : Nat := 7
def perimeter : Nat := 23

-- Statement to prove
theorem isosceles_triangle_base_length :
  let b := perimeter - 2 * congruent_side in b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l129_129232


namespace range_of_independent_variable_l129_129265

theorem range_of_independent_variable (x : ℝ) : (x - 4) ≠ 0 ↔ x ≠ 4 :=
by
  sorry

end range_of_independent_variable_l129_129265


namespace intersection_A_B_l129_129498

-- Define set A
def set_A : Set ℕ := {0, 1, 2, 3, 4, 5}

-- Define set B
def set_B : Set ℝ := {x | x^2 < 10}

-- Define the intersection of A and B
def A_inter_B : Set ℝ := {x | x ∈ set_A ∧ x ∈ set_B}

-- The statement to be proved
theorem intersection_A_B : A_inter_B = {0, 1, 2, 3} :=
by
  sorry

end intersection_A_B_l129_129498


namespace janous_inequality_l129_129168

theorem janous_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry

end janous_inequality_l129_129168


namespace proof_l129_129531

def question (x : ℝ) : ℝ :=
  let a := x / 10
  let b := x % 10
  10 * b + a

theorem proof (h : 61 = question 61) : question 61 = 16.1 :=
by 
  sorry

end proof_l129_129531


namespace math_problem_l129_129423

theorem math_problem 
  (a b c : ℝ) 
  (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) 
  (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : a^2 + b^2 = c^2 + ab) : 
  c^2 + ab < a*c + b*c := 
sorry

end math_problem_l129_129423


namespace length_equality_l129_129878

variable {a b : ℝ} (ha : a > 0) (hb : b > 0)

def hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

def focus_right : ℝ := real.sqrt (a^2 + b^2)

def intersect_points (p : ℝ × ℝ) : Prop := 
  ∃ (l : ℝ), (l passes through (focus_right, 0) ∧ intersects p)

theorem length_equality (A B C D : ℝ × ℝ) (hA : intersect_points A) (hB : intersect_points B) 
  (hC : intersect_points C) (hD : intersect_points D) :
  (hyperbola A.1 A.2) →
  (hyperbola B.1 B.2) →
  (hyperbola C.1 C.2) →
  (hyperbola D.1 D.2) →
  (A and B are on the hyperbola and line l) →
  (C and D are on the hyperbola and line l) →
  |A.1 - C.1| = |B.1 - D.1| :=
sorry

end length_equality_l129_129878


namespace sally_and_carl_owe_equal_amounts_l129_129204

theorem sally_and_carl_owe_equal_amounts (total_promised total_received amy_owed : ℝ)
  (derek_owed_is_half_amy : derek_owed = amy_owed / 2) 
  (sally_and_carl_owe_equal : sally_owed = carl_owed) 
  (total_promised = 400) 
  (total_received = 285) 
  (amy_owed = 30) :
  sally_owed = 35 ∧ carl_owed = 35 := 
by
  sorry

end sally_and_carl_owe_equal_amounts_l129_129204


namespace minimum_players_l129_129356

theorem minimum_players (a : Finset ℕ) (h : ∀ x ∈ a, x > 0) : 
  ∃ (k : ℕ), k = (a.max' (finset.nonempty_iff_ne_empty.mpr (finset.ne_empty_of_card_ne_zero (ne_of_gt (finset.card_pos.mpr (finset.card_pos_iff.mpr (finset.nonempty_of_ne_empty sorry))))))) / 6 ∧ 
    3 * k + 3 = 
    ∃ (n : ℕ) (players : Finset ℕ) (doubles : Finset (Finset ℕ)), 
      (∀ p ∈ players, ∃ b ∈ doubles, p ∈ b ∧ ∀ b1 b2 ∈ doubles, b1 ≠ b2 → b1 ∩ b2 = ∅ ∨ p ∉ b1) ∧ 
      (∀ d ∈ doubles, (∀ p ∈ d, ∃ m ∈ a, ∃ q ∈ players, p ≠ q ∧ (m ∈ a → ∃ p1 p2 ∈ players, p1 ≠ p2 ∧ (p ∈ p1 ∧ p1 ≠ p ∧ p ∉ p2) ∨ p ∉ p1))) ∧ 
      (∀ m ∈ a, ∃ p ∈ players, ∃ d ∈ doubles, p ∈ d ∧ d.card = m) ∧
      players.card = 3 * k + 3 := 
sorry

end minimum_players_l129_129356


namespace can_transform_to_zero_l129_129130

def initial_grid : list (list ℕ) := [[0, 3, 6], [2, 6, 9], [7, 4, 5]]

def is_zero_grid (grid : list (list ℕ)) : Prop :=
  grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

def valid_operation (grid : list (list ℕ)) (i j : nat) (delta : ℕ) : list (list ℕ) :=
  -- Assuming some function to compute a valid operation
  [[0, 0, 0], [0, 0, 0], [0, 0, 0]] -- placeholder for actual implementation

theorem can_transform_to_zero :
  ∃ seq_of_ops : list (nat × nat × ℕ),
    is_zero_grid (seq_of_ops.foldl (λ g op, valid_operation g op.1 op.2 op.3) initial_grid) :=
sorry

end can_transform_to_zero_l129_129130


namespace books_arrangement_count_l129_129777

theorem books_arrangement_count :
  let n_geom := 5
  let n_numth := 6
  let total := n_geom + n_numth
  nat.choose total n_geom = 462 :=
by
  let n_geom := 5
  let n_numth := 6
  let total := n_geom + n_numth
  have h1 : total = 11 := rfl
  rw [←h1]
  exact sorry

end books_arrangement_count_l129_129777


namespace sum_of_real_values_l129_129053

theorem sum_of_real_values (x : ℝ) (h : abs x < 1) :
  x = 2 - 2 * x + 2 * x^2 - 2 * x^3 + 2 * x^4 - ⋯ →
  x = 1 :=
sorry

end sum_of_real_values_l129_129053


namespace cube_rotation_axes_count_l129_129791

theorem cube_rotation_axes_count : 
  let axes_through_faces := 3,
      axes_through_edges := 6,
      axes_through_vertices := 4,
      total_axes := axes_through_faces + axes_through_edges + axes_through_vertices
  in total_axes = 13 := 
by
  sorry

end cube_rotation_axes_count_l129_129791


namespace numDistinctSubsets_l129_129800

def isSpecial (a b : ℕ) : Prop := a + b = 20

def specialFractions : List ℚ :=
  [1 / 19, 1 / 9, 3 / 17, 1 / 4, 1 / 3, 3 / 7, 7 / 13, 2 / 3, 1, 11 / 9,
   3 / 2, 13 / 7, 7 / 3, 7 / 2, 3, 4, 17 / 3, 9, 19]

def sumsOfSpecialFractions : List ℚ :=
  (specialFractions.product specialFractions).map (λ (f1, f2) => f1 + f2)

def distinctIntegerSums : Finset ℕ := 
  (sumsOfSpecialFractions.filterMap (λ q => if q.den = 1 then some q.num else none)).toFinset

theorem numDistinctSubsets : distinctIntegerSums.card = 16 := by
  sorry

end numDistinctSubsets_l129_129800


namespace kuzya_probability_distance_2h_l129_129830

noncomputable def probability_kuzya_at_distance_2h : ℚ :=
  let h := 1 -- treat each jump length as 1 for simplicity
  let events := finset.range 6 -- number of jumps from 2 to 5
  let prob_at_2h (n : ℕ) : ℚ := 
    if n < 2 then 0
    else if n = 2 then 1/2
    else if n = 3 then 3/8
    else if n = 4 then 3/8
    else if n = 5 then 15/32
    else 0
  (events.sum prob_at_2h) / events.sum (λ n, 1)

theorem kuzya_probability_distance_2h :
  probability_kuzya_at_distance_2h = 5 / 8 :=
sorry

end kuzya_probability_distance_2h_l129_129830


namespace incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129653

def is_color (col : String) (pos : Nat) : Prop := sorry
def valid_guesses : Prop :=
  (is_color "white" 2) ∧
  (is_color "brown" 20) ∧
  (is_color "black" 400) ∧
  (is_color "brown" 600) ∧
  (is_color "white" 800)

theorem incorrect_guess_at_20 :
  (∃ x, (x ∈ [2, 20, 400, 600, 800]) ∧ ¬ is_color_correct x) :=
begin
  sorry -- proof is not required
end

/-- Main theorem to identify the incorrect guess position. -/
theorem Iskander_incorrect_guess_20 :
  valid_guesses →
  (∃! x ∈ [2, 20, 400, 600, 800], ¬ is_color_correct x) →
  ¬ is_color "brown" 20 :=
begin
  admit -- proof is not required
end

end incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129653


namespace initial_milk_volume_is_10_l129_129189

theorem initial_milk_volume_is_10 :
  (∀ (x : ℝ), 0.05 * x = 0.02 * (x + 15) → x = 10) :=
by
  intro x
  intro h
  have h1 : 0.05 * x = 0.02 * x + 0.3 := by
    rw [mul_add]
    exact h
  have h2 : 0.05 * x - 0.02 * x = 0.3 := by
    rw sub_eq_sub_iff_sub_eq_sub
    exact h1
  have h3 : 0.03 * x = 0.3 := by
    exact h2
  have h4 : x = 0.3 / 0.03 := by
    exact eq_div_of_mul_eq h3
  have h5 : x = 10 := by
    norm_num at h4
    exact h4
  exact h5

end initial_milk_volume_is_10_l129_129189


namespace imaginary_part_of_z_pow_2017_l129_129506

def z : ℂ := (1 + complex.I) / (1 - complex.I) -- Define the given complex number z

theorem imaginary_part_of_z_pow_2017 :
  complex.im (z^2017) = complex.im complex.I := -- Prove the imaginary part of z^2017 is the same as i
sorry -- Proof is omitted

end imaginary_part_of_z_pow_2017_l129_129506


namespace perfect_square_sequence_l129_129646

theorem perfect_square_sequence :
  ∃ (a b : ℤ), a = 1 ∧ b = 2008 ∧
  (∀ n : ℕ, ∃ k : ℤ, 1 + 2006 * (seq n) * (seq (n + 1)) = k^2)
  where
    seq : ℕ → ℤ
    | 0       => 1
    | 1       => 2008
    | (n + 2) => 2008 * (seq (n + 1)) - (seq n) :=
begin
  sorry
end

end perfect_square_sequence_l129_129646


namespace num_permutations_with_3_inversions_l129_129948

open Finset

def num_inversions (n : ℕ) (σ : Perm (Fin n)) : ℕ :=
  univ.filter (λ ⟨i, j⟩, i ≤ j ∧ σ i > σ j).card

theorem num_permutations_with_3_inversions {n : ℕ} (h : n ≥ 3) :
  (univ.filter (λ σ : Perm (Fin n), num_inversions n σ = 3)).card = n * (n^2 - 7) / 6 :=
sorry

end num_permutations_with_3_inversions_l129_129948


namespace max_sum_of_square_roots_l129_129365

theorem max_sum_of_square_roots (k : ℝ) (h : 0 ≤ k) : 
  ∃ x : ℝ, x^2 = k / 2 ∧ (k - x^2) = k / 2 ∧ x + real.sqrt (k - x^2) = real.sqrt (2 * k) :=
by 
  sorry

end max_sum_of_square_roots_l129_129365


namespace dog_weights_l129_129796

structure DogWeightProgression where
  initial: ℕ   -- initial weight in pounds
  week_9: ℕ    -- weight at 9 weeks in pounds
  month_3: ℕ  -- weight at 3 months in pounds
  month_5: ℕ  -- weight at 5 months in pounds
  year_1: ℕ   -- weight at 1 year in pounds

theorem dog_weights :
  ∃ (golden_retriever labrador poodle : DogWeightProgression),
  golden_retriever.initial = 6 ∧
  golden_retriever.week_9 = 12 ∧
  golden_retriever.month_3 = 24 ∧
  golden_retriever.month_5 = 48 ∧
  golden_retriever.year_1 = 78 ∧
  labrador.initial = 8 ∧
  labrador.week_9 = 24 ∧
  labrador.month_3 = 36 ∧
  labrador.month_5 = 72 ∧
  labrador.year_1 = 102 ∧
  poodle.initial = 4 ∧
  poodle.week_9 = 16 ∧
  poodle.month_3 = 32 ∧
  poodle.month_5 = 32 ∧
  poodle.year_1 = 52 :=
by 
  have golden_retriever : DogWeightProgression := { initial := 6, week_9 := 12, month_3 := 24, month_5 := 48, year_1 := 78 }
  have labrador : DogWeightProgression := { initial := 8, week_9 := 24, month_3 := 36, month_5 := 72, year_1 := 102 }
  have poodle : DogWeightProgression := { initial := 4, week_9 := 16, month_3 := 32, month_5 := 32, year_1 := 52 }
  use golden_retriever, labrador, poodle
  repeat { split };
  { sorry }

end dog_weights_l129_129796


namespace intersection_is_x_gt_1_l129_129882

def A (x : ℝ) : Prop := x^2 > 1
def B (x : ℝ) : Prop := log 2 x > 0

theorem intersection_is_x_gt_1 :
  {x : ℝ | A x} ∩ {x | B x} = {x | x > 1} :=
by
  sorry

end intersection_is_x_gt_1_l129_129882


namespace fraction_problem_l129_129124

theorem fraction_problem (N D : ℚ) (h1 : 1.30 * N / (0.85 * D) = 25 / 21) : 
  N / D = 425 / 546 :=
sorry

end fraction_problem_l129_129124


namespace probability_one_absent_one_present_l129_129135

theorem probability_one_absent_one_present :
  (∀ (students : Type) (student : students),
    let absent_days := 1,
        total_days := 20,
        prob_absent := (absent_days : ℝ) / (total_days : ℝ),
        prob_present := 1 - prob_absent
    in
    (prob_absent * prob_present) + (prob_present * prob_absent) = 0.095) :=
by
  sorry

end probability_one_absent_one_present_l129_129135


namespace directrix_of_parabola_l129_129445

def parabola (x : ℝ) : ℝ := (x^2 - 8*x + 12) / 16

theorem directrix_of_parabola :
  ∀ x, parabola x = (x-4)^2 / 16 - 1/4 →
  let a := 1/16 in
  let h := 4 in
  let k := -1/4 in
  let directrix := k - 1/(4*a) in
  directrix = -17/4 :=
by
  intro x h1
  simp only [parabola] at h1
  dsimp [a, h, k] at h1
  have := calc
    k - 1/(4*a) = -1/4 - 4 : by field_simp [a]
    ... = -17/4 : by norm_num
  exact this

end directrix_of_parabola_l129_129445


namespace roots_eq_recurrence_relation_find_a_b_l129_129855

-- Conditions
theorem roots_eq (α β: ℝ) (h1: α * α - α - 1 = 0) (h2: β * β - β - 1 = 0) : 
  α + β = 1 ∧ α * β = -1 := sorry

noncomputable def a_n (n : ℕ) (α β : ℝ) : ℝ := (α^n - β^n) / (α - β)

-- Part 1: Prove recurrence relation
theorem recurrence_relation (α β: ℝ) (h1: α * α - α - 1 = 0) (h2: β * β - β - 1 = 0) :
  ∀ n : ℕ, a_n (n+2) α β = a_n (n+1) α β + a_n n α β := sorry

-- Part 2: Find a and b
theorem find_a_b (α β: ℝ) (h1: α * α - α - 1 = 0) (h2: β * β - β - 1 = 0) :
  ∃ a b : ℕ, a < b ∧ (∀ n : ℕ, b ∣ (a_n n α β - 2 * n * a^n)) ∧ a = 3 ∧ b = 5 := sorry

end roots_eq_recurrence_relation_find_a_b_l129_129855


namespace problem_triangle_circle_tangent_l129_129151

theorem problem_triangle_circle_tangent 
  (A B C O D : Point) 
  (h1 : B ∈ circle O r) 
  (h2 : C ∈ circle O r) 
  (h3 : tangent B A (circle O r)) 
  (h4 : tangent C A (circle O r)) 
  (h5 : angle A B C = 80)
  (h6 : intersects BO (circle O r) = {B, D}) :
  BD / BO = sin 10 := 
sorry

end problem_triangle_circle_tangent_l129_129151


namespace palindromes_with_seven_percent_l129_129780

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.to_string.data;
  digits = digits.reverse

def valid_palindromes : finset ℕ :=
  (finset.range 2000).filter (λ n, 1000 ≤ n ∧ is_palindrome n)

def contains_seven (n : ℕ) : Prop :=
  '7' ∈ n.to_string.data

theorem palindromes_with_seven_percent :
  ((valid_palindromes.filter contains_seven).card * 10 = valid_palindromes.card) → 
  10% := 
begin
  sorry
end

end palindromes_with_seven_percent_l129_129780


namespace min_value_expression_l129_129838

open Real

theorem min_value_expression 
  (a : ℝ) 
  (b : ℝ) 
  (hb : 0 < b) 
  (e : ℝ) 
  (he : e = 2.718281828459045) :
  ∃ x : ℝ, 
  (x = 2 * (1 - log 2)^2) ∧
  ∀ a b, 
    0 < b → 
    ((1 / 2) * exp a - log (2 * b))^2 + (a - b)^2 ≥ x :=
sorry

end min_value_expression_l129_129838


namespace monotonic_decreasing_interval_l129_129255

noncomputable def function_f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = -1 ∧ b = 11 ∧ 
  ∀ x, a < x ∧ x < b → derivative ℝ (λ x, x^3 - 15 * x^2 - 33 * x + 6) x < 0 :=
begin
  sorry
end

end monotonic_decreasing_interval_l129_129255


namespace coeffs_of_expansion_l129_129849

theorem coeffs_of_expansion : 
  (a a_1 a_2 a_3 a_4 : ℝ) (h : (∀ x : ℝ, (2 * x + 1)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 )) :
  a * (a_1 + a_3) = 40 :=
sorry

end coeffs_of_expansion_l129_129849


namespace ratio_Bill_Cary_l129_129403

noncomputable def Cary_height : ℝ := 72
noncomputable def Jan_height : ℝ := 42
noncomputable def Bill_height : ℝ := Jan_height - 6

theorem ratio_Bill_Cary : Bill_height / Cary_height = 1 / 2 :=
by
  sorry

end ratio_Bill_Cary_l129_129403


namespace prob_at_least_one_on_l129_129643

theorem prob_at_least_one_on (p: ℚ) (h : p = 1 / 2) : (1 - (p ^ 3)) = 7 / 8 := 
by
  rw [h]
  norm_num
  sorry

end prob_at_least_one_on_l129_129643


namespace find_t_l129_129865

theorem find_t (t : ℝ) : (∃ y : ℝ, y = -(t - 1) ∧ 2 * y - 4 = 3 * (y - 2)) ↔ t = -1 :=
by sorry

end find_t_l129_129865


namespace Δy_over_Δx_l129_129873

-- Conditions
def f (x : ℝ) : ℝ := 2 * x^2 - 4
def y1 : ℝ := f 1
def y2 (Δx : ℝ) : ℝ := f (1 + Δx)
def Δy (Δx : ℝ) : ℝ := y2 Δx - y1

-- Theorem statement
theorem Δy_over_Δx (Δx : ℝ) : Δy Δx / Δx = 4 + 2 * Δx := by
  sorry

end Δy_over_Δx_l129_129873


namespace non_acute_vectors_groups_l129_129551

variables {n r : ℕ}
variable {α : Fin (n + r) → ℝⁿ}
variable [InnerProductSpace ℝ ℝⁿ]
variable (h : ∀ i j : Fin (n + r), i < j → ⟪α i, α j⟫_ℝ ≤ 0)
include h

theorem non_acute_vectors_groups :
  r ≤ n ∧ (∃ (groups : Fin r → Finset (Fin (n + r))), 
  (∀ i j, i ≠ j → ∀ v ∈ groups i, ∀ w ∈ groups j, ⟪α v, α w⟫_ℝ = 0) ∧ 
  (∀ i, ∃ v w ∈ groups i, i ≠ j → ⟪α v, α w⟫_ℝ = -⟪α v, α v⟫_ℝ))
|-- sorry

end non_acute_vectors_groups_l129_129551


namespace magnitude_of_angle_C_max_area_l129_129916

variable (A B C a b c : ℝ)

-- The given condition
axiom condition : (2 * a + b) / c = (Real.cos (A + C)) / (Real.cos C)

-- Proof that C is equal to 2π/3
theorem magnitude_of_angle_C (h : condition) : C = 2 * Real.pi / 3 := sorry

-- Given area maximization problem
variable (S : ℝ)

-- The given condition for c=2
axiom condition_c : c = 2

-- The maximum area when a = b = (2 * sqrt(3)) / 3
theorem max_area (h : condition_c) : (a = (2 * Real.sqrt 3) / 3) ∧ (b = (2 * Real.sqrt 3) / 3) ∧ (S = Real.sqrt 3 / 3) := sorry

end magnitude_of_angle_C_max_area_l129_129916


namespace smallest_n_valid_l129_129558

noncomputable def smallest_valid_n : ℕ := 15

theorem smallest_n_valid :
  ∀ (n : ℕ),
    (∃ f : ℕ → ℕ, 
      (∀ i j, connected i j → gcd (f i + f j) n > 1) ∧
      (∀ i j, ¬connected i j → gcd (f i + f j) n = 1)) → 
  n ≥ 15 :=
by sorry

end smallest_n_valid_l129_129558


namespace aquarium_width_calculation_l129_129191

theorem aquarium_width_calculation :
  ∃ (W : ℝ), (4 * W * 3) / 2 / 2 * 3 = 54 ∧ W = 6 :=
by
  use 6
  split
  sorry

end aquarium_width_calculation_l129_129191


namespace find_wrong_guess_l129_129689

-- Define the three colors as an inductive type.
inductive Color
| white
| brown
| black

-- Define the bears as a list of colors.
def bears (n : ℕ) : Type := list Color

-- Define the conditions: 
-- There are 1000 bears and each tuple of 3 consecutive bears has all three colors.
def valid_bears (b : bears 1000) : Prop :=
  ∀ i : ℕ, i + 2 < 1000 → 
    ∃ c1 c2 c3 : Color, 
      c1 ∈ b.nth i ∧ c2 ∈ b.nth (i+1) ∧ c3 ∈ b.nth (i+2) ∧ 
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3

-- Define Iskander's guesses.
def guesses (b : bears 1000) : Prop :=
  b.nth 1 = some Color.white ∧
  b.nth 19 = some Color.brown ∧
  b.nth 399 = some Color.black ∧
  b.nth 599 = some Color.brown ∧
  b.nth 799 = some Color.white

-- Prove that exactly one of Iskander's guesses is wrong.
def wrong_guess (b : bears 1000) : Prop :=
  (b.nth 19 ≠ some Color.brown) ∧
  valid_bears b ∧
  guesses b →
  ∃ i, i ∈ {1, 19, 399, 599, 799} ∧ (b.nth i ≠ some Color.white ∧ b.nth i ≠ some Color.brown ∧ b.nth i ≠ some Color.black)

theorem find_wrong_guess : 
  ∀ b : bears 1000, 
  valid_bears b → guesses b → wrong_guess b :=
  by
  intros b vb gs
  sorry

end find_wrong_guess_l129_129689


namespace greatest_integer_not_exceed_x_squared_over_150_l129_129604

theorem greatest_integer_not_exceed_x_squared_over_150 (b h: ℝ) (x: ℝ) :
  (b + 150 = x) → 
  (b = 300) → 
  x = 300√(275) →
  let result := ⌊x^2 / 150⌋ in
  result = 550 :=
by
  intro base_longer condition base_equality segment_value
  sorry

end greatest_integer_not_exceed_x_squared_over_150_l129_129604


namespace find_numbers_l129_129475

theorem find_numbers (x y : ℝ) (r : ℝ) (d : ℝ) 
  (h_geom_x : x = 5 * r) 
  (h_geom_y : y = 5 * r^2)
  (h_arith_1 : y = x + d) 
  (h_arith_2 : 15 = y + d) : 
  x + y = 10 :=
by
  sorry

end find_numbers_l129_129475


namespace product_of_three_integers_sum_l129_129263
-- Import necessary libraries

-- Define the necessary conditions and the goal
theorem product_of_three_integers_sum (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
(h4 : a * b * c = 11^3) : a + b + c = 133 :=
sorry

end product_of_three_integers_sum_l129_129263


namespace problem1_problem2_problem3_problem4_l129_129797

-- 1. Prove $\sqrt{12}-|\sqrt{3}-3|+(\sqrt{3})^2 = 3\sqrt{3}$
theorem problem1 : sqrt 12 - abs (sqrt 3 - 3) + (sqrt 3) ^ 2 = 3 * sqrt 3 :=
  sorry

-- 2. Prove $2\sqrt{18} \times \frac{\sqrt{3}}{4} \div 5\sqrt{2} = \frac{3\sqrt{3}}{2}$
theorem problem2 : 2 * sqrt 18 * (sqrt 3 / 4) / (5 * sqrt 2) = 3 * sqrt 3 / 2 :=
  sorry

-- 3. Prove $\frac{\sqrt{6}-\sqrt{3}}{\sqrt{3}}+(2+\sqrt{2})(2-\sqrt{2}) = \sqrt{2}+1$
theorem problem3 : (sqrt 6 - sqrt 3) / sqrt 3 + (2 + sqrt 2) * (2 - sqrt 2) = sqrt 2 + 1 :=
  sorry

-- 4. Prove $\frac{2}{3}\sqrt{9x}+6\sqrt{\frac{x}{4}}-2\sqrt{\frac{1}{x}} = 3\sqrt{x}$ when $x=\sqrt{\frac{1}{x}}$
theorem problem4 (x : ℝ) (hx : x = sqrt (1 / x)) : (2 / 3) * sqrt (9 * x) + 6 * sqrt (x / 4) - 2 * sqrt (1 / x) = 3 * sqrt x :=
  sorry

end problem1_problem2_problem3_problem4_l129_129797


namespace total_fruit_pieces_correct_l129_129923

/-
  Define the quantities of each type of fruit.
-/
def red_apples : Nat := 9
def green_apples : Nat := 4
def purple_grapes : Nat := 3
def yellow_bananas : Nat := 6
def orange_oranges : Nat := 2

/-
  The total number of fruit pieces in the basket.
-/
def total_fruit_pieces : Nat := red_apples + green_apples + purple_grapes + yellow_bananas + orange_oranges

/-
  Prove that the total number of fruit pieces is 24.
-/
theorem total_fruit_pieces_correct : total_fruit_pieces = 24 := by
  sorry

end total_fruit_pieces_correct_l129_129923


namespace minimum_distance_AB_l129_129198

def point_A := (x1 : ℝ) => (x1, (15/8) * x1 - 8)
def point_B := (x2 : ℝ) => (x2, x2^2)

def distance (A B : ℝ × ℝ) : ℝ :=
  (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

-- The statement of the theorem
theorem minimum_distance_AB : 
  ∃ x1 x2 : ℝ, distance (point_A x1) (point_B x2) = 1823 / 544 :=
sorry

end minimum_distance_AB_l129_129198


namespace friend_redistribute_l129_129056

-- Definition and total earnings
def earnings : List Int := [30, 45, 15, 10, 60]
def total_earnings := earnings.sum

-- Number of friends
def number_of_friends : Int := 5

-- Calculate the equal share
def equal_share := total_earnings / number_of_friends

-- Calculate the amount to redistribute by the friend who earned 60
def amount_to_give := 60 - equal_share

theorem friend_redistribute :
  earnings.sum = 160 ∧ equal_share = 32 ∧ amount_to_give = 28 :=
by
  -- Proof goes here, skipped with 'sorry'
  sorry

end friend_redistribute_l129_129056


namespace ideal_contains_sum_polynomial_l129_129572

variable {R : Type*} [CommRing R] [Nontrivial R] 

def contains_polynomial_with_constant_term_one
  (I : Ideal R[X]) : Prop :=
∃ f : R[X], f ∈ I ∧ f.coeff 0 = 1

def no_common_divisor_greater_than_zero
  (I : Ideal R[X]) : Prop :=
∃ f g : R[X], f ∈ I ∧ g ∈ I ∧ ∀ (d : R[X]), d.degree > 0 → ¬ (d ∣ f ∧ d ∣ g)

theorem ideal_contains_sum_polynomial
  (I : Ideal R[X])
  (h1 : no_common_divisor_greater_than_zero I)
  (h2 : contains_polynomial_with_constant_term_one I) :
  ∃ r : ℕ, (Polynomial.sum (fun k => R[X].X ^ k) r) ∈ I :=
sorry

end ideal_contains_sum_polynomial_l129_129572


namespace differentiable_additive_zero_derivative_l129_129959

theorem differentiable_additive_zero_derivative {f : ℝ → ℝ}
  (h1 : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
  (h_diff : Differentiable ℝ f) : 
  deriv f 0 = 0 :=
sorry

end differentiable_additive_zero_derivative_l129_129959


namespace proof_problem_l129_129419

-- Define the operation [a, b, c] as the quotient of the sum of a and b by c, given c ≠ 0

def myOperation (a b c : ℚ) (h : c ≠ 0) : ℚ :=
  (a + b) / c

theorem proof_problem : 
  myOperation 
    (myOperation 120 60 180 (by norm_num))
    (myOperation 4 2 6 (by norm_num))
    (myOperation 20 10 30 (by norm_num)) 
    (by norm_num) 
  = 2 := sorry

end proof_problem_l129_129419


namespace probability_correct_min_attempts_correct_l129_129547

noncomputable def probability_event : ℚ :=
  36 / 45 * 14 / 44 * 15 / 43 + 36 / 45 * 9 / 44 * 15 / 43 + 9 / 45 * 8 / 44 * 15 / 43

noncomputable def min_attempts (target_probability : ℚ) : ℕ :=
  let fail_probability := 1 - probability_event
  let log_base := fail_probability.log
  let log_target := target_probability.log
  ((log_target / log_base).ceil : ℕ) + 1

theorem probability_correct :
  probability_event = 54 / 473 :=
by
  sorry

theorem min_attempts_correct (h : 0.9 ≤ 1) :
  min_attempts 0.1 = 19 :=
by
  sorry

end probability_correct_min_attempts_correct_l129_129547


namespace find_wrong_guess_l129_129685

-- Define the three colors as an inductive type.
inductive Color
| white
| brown
| black

-- Define the bears as a list of colors.
def bears (n : ℕ) : Type := list Color

-- Define the conditions: 
-- There are 1000 bears and each tuple of 3 consecutive bears has all three colors.
def valid_bears (b : bears 1000) : Prop :=
  ∀ i : ℕ, i + 2 < 1000 → 
    ∃ c1 c2 c3 : Color, 
      c1 ∈ b.nth i ∧ c2 ∈ b.nth (i+1) ∧ c3 ∈ b.nth (i+2) ∧ 
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3

-- Define Iskander's guesses.
def guesses (b : bears 1000) : Prop :=
  b.nth 1 = some Color.white ∧
  b.nth 19 = some Color.brown ∧
  b.nth 399 = some Color.black ∧
  b.nth 599 = some Color.brown ∧
  b.nth 799 = some Color.white

-- Prove that exactly one of Iskander's guesses is wrong.
def wrong_guess (b : bears 1000) : Prop :=
  (b.nth 19 ≠ some Color.brown) ∧
  valid_bears b ∧
  guesses b →
  ∃ i, i ∈ {1, 19, 399, 599, 799} ∧ (b.nth i ≠ some Color.white ∧ b.nth i ≠ some Color.brown ∧ b.nth i ≠ some Color.black)

theorem find_wrong_guess : 
  ∀ b : bears 1000, 
  valid_bears b → guesses b → wrong_guess b :=
  by
  intros b vb gs
  sorry

end find_wrong_guess_l129_129685


namespace integer_solutions_count_l129_129895

theorem integer_solutions_count : 
  ∀ x : ℤ, 
    (x^2 - 2*x - 3)^(x+1) = 1 
    → x = 4 ∨ x = -1 ∨ x = 0 ∨ x = 3 
      ∧ (¬ ∃ x1 x2 x3 x4, 
          {x1, x2, x3, x4} = {4, -1, 0, 3} ∧
          x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ x1 ≠ x4 ∧ x4 ≠ x2 ∧ x4 ≠ x3) :=
by
  intro x hx
  sorry

end integer_solutions_count_l129_129895


namespace square_completing_l129_129013

theorem square_completing (b c : ℤ) (h : (x^2 - 10 * x + 15 = 0) → ((x + b)^2 = c)) : 
  b + c = 5 :=
sorry

end square_completing_l129_129013


namespace final_cost_of_dress_l129_129220

theorem final_cost_of_dress (original_price : ℝ) (discount_percentage : ℝ) 
  (h1 : original_price = 50) (h2 : discount_percentage = 0.30) : 
  let discount := discount_percentage * original_price in
  let final_cost := original_price - discount in
  final_cost = 35 := 
by
  sorry

end final_cost_of_dress_l129_129220


namespace prob_A_is_0_8_l129_129542

theorem prob_A_is_0_8 (A B : Prop) [independent : independent A B]
  (prob_intersection : P(A ∧ B) = 0.08)
  (ineq : P(A) > P(B)) :
  P(A) = 0.8 :=
sorry

end prob_A_is_0_8_l129_129542


namespace even_fn_a_eq_zero_l129_129912

def f (x a : ℝ) : ℝ := x^2 - |x + a|

theorem even_fn_a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end even_fn_a_eq_zero_l129_129912


namespace cartesian_equation_C2_distance_AB_l129_129560

-- Definitions
def parametric_curve_C1 (theta : ℝ) : ℝ × ℝ :=
  (1 + sqrt 3 * cos theta, sqrt 3 * sin theta)

def point_P_on_C2 (theta : ℝ) : ℝ × ℝ :=
  let M := parametric_curve_C1 theta
  (2 * M.1, 2 * M.2)

-- Proving the Cartesian equation of C2
theorem cartesian_equation_C2 : ∀ (x y : ℝ),
  ((x = 2 * (1 + sqrt 3 * cos theta)) ∧ (y = 2 * (sqrt 3 * sin theta))) → 
  (x - 2)^2 + y^2 = 12 :=
sorry

-- Proving the distance between points A and B in the polar coordinate system
theorem distance_AB : |AB| = 2 :=
sorry

end cartesian_equation_C2_distance_AB_l129_129560


namespace sin_graph_shift_l129_129287

theorem sin_graph_shift :
  ∀ x : ℝ, y = sin (1/2 * x) ↔ y = sin (1/2 * (x - π / 3)) :=
by sorry

end sin_graph_shift_l129_129287


namespace exists_city_from_which_all_are_accessible_l129_129546

variables {V : Type*} [Fintype V] (G : V → V → Prop)

def accessible (G : V → V → Prop) (a b : V) : Prop :=
  ∃ (path : List V), path.head = some a ∧ path.ilast = some b ∧ (∀ x y ∈ path, G x y)

theorem exists_city_from_which_all_are_accessible :
  (∀ (P Q : V), ∃ R : V, accessible G R P ∧ accessible G R Q) →
  ∃ C : V, ∀ (X : V), accessible G C X :=
sorry

end exists_city_from_which_all_are_accessible_l129_129546


namespace quadratic_polynomial_real_coeff_l129_129467

open Complex

theorem quadratic_polynomial_real_coeff (a b : ℝ) :
  (∀ x : ℂ, x = 4 + 2 * I ∨ x = 4 - 2 * I → (3 * (x - (4 + 2 * I)) * (x - (4 - 2 * I)) = 0)) →
    (∀ c : ℝ, c = 3 → 
    (∃ p : ℝ[X], p = 3 * (X - C(4 + 2 * I)) * (X - C(4 - 2 * I)) → p = 3 * X^2 - 24 * X + 60)) :=
by {
  intros h1 h2,
  sorry
}

end quadratic_polynomial_real_coeff_l129_129467


namespace angle_scaled_vectors_l129_129905

variables (a b : ℝ → ℝ) -- assuming a and b are vectors (ℝ → ℝ kind e.g. transforming real numbers)
noncomputable def angle (v1 v2 : ℝ → ℝ) : ℝ := sorry -- placeholder for angle calculation between two vectors

theorem angle_scaled_vectors (h : angle a b = 60) : angle (λ x, 2 * a x) (λ x, - b x) = 120 :=
by sorry

end angle_scaled_vectors_l129_129905


namespace find_f_2_l129_129585

theorem find_f_2 (f : ℝ → ℝ) (h₁ : f 1 = 0)
  (h₂ : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)) :
  f 2 = 0 :=
sorry

end find_f_2_l129_129585


namespace sum_of_combinations_eq_two_to_the_n_l129_129596

theorem sum_of_combinations_eq_two_to_the_n (n : ℕ) : 
  (∑ k in Finset.range (n + 1), Nat.choose n k) = 2^n := by
  sorry

end sum_of_combinations_eq_two_to_the_n_l129_129596


namespace probability_of_rain_given_northeast_winds_l129_129788

theorem probability_of_rain_given_northeast_winds
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ) (h1 : P_A = 0.7)
  (h2 : P_B = 0.8) (h3 : P_A_and_B = 0.65) :
  P_A_and_B / P_A = 13 / 14 :=
by
  have h : P_A_and_B = 0.65 := h3
  have h_2 : P_A = 0.7 := h1
  unfold at h1 h2 h3
  sorry

end probability_of_rain_given_northeast_winds_l129_129788


namespace gold_problem_proof_l129_129935

noncomputable def solve_gold_problem : Prop :=
  ∃ (a : ℕ → ℝ), 
  (a 1) + (a 2) + (a 3) = 4 ∧ 
  (a 8) + (a 9) + (a 10) = 3 ∧
  (a 5) + (a 6) = 7 / 3

theorem gold_problem_proof : solve_gold_problem := 
  sorry

end gold_problem_proof_l129_129935


namespace magnitude_of_complex_number_l129_129047

noncomputable def complex_number : Complex := 2 / (1 + Complex.i)

theorem magnitude_of_complex_number : Complex.abs complex_number = Real.sqrt 2 :=
by
  -- Proof is not required as per instructions
  sorry

end magnitude_of_complex_number_l129_129047


namespace mark_initial_fries_l129_129992

variable (Sally_fries_before : ℕ)
variable (Sally_fries_after : ℕ)
variable (Mark_fries_given : ℕ)
variable (Mark_fries_initial : ℕ)

theorem mark_initial_fries (h1 : Sally_fries_before = 14) (h2 : Sally_fries_after = 26) (h3 : Mark_fries_given = Sally_fries_after - Sally_fries_before) (h4 : Mark_fries_given = 1/3 * Mark_fries_initial) : Mark_fries_initial = 36 :=
by sorry

end mark_initial_fries_l129_129992


namespace overall_percent_change_in_stock_l129_129389

noncomputable def stock_change (initial_value : ℝ) : ℝ :=
  let value_after_first_day := 0.85 * initial_value
  let value_after_second_day := 1.25 * value_after_first_day
  (value_after_second_day - initial_value) / initial_value * 100

theorem overall_percent_change_in_stock (x : ℝ) : stock_change x = 6.25 :=
by
  sorry

end overall_percent_change_in_stock_l129_129389


namespace minimum_marty_score_l129_129598

theorem minimum_marty_score (M : ℕ) : M >= 61 :=
  let S := 80 in  -- Total score
  have h₁ : S = 4 * 20 := by rfl,
  have h₂ : (80 - M) / 3 < 20 := by sorry,  -- Everyone else scores below 20.
  have h₃ : M > 20 := by sorry,
  have h₄ : (80 - M) < 60 := by sorry,
  by sorry

end minimum_marty_score_l129_129598


namespace max_value_of_2_power_x_1_minus_x_l129_129963

theorem max_value_of_2_power_x_1_minus_x (x : ℝ) : 
  ∃ (y : ℝ), (∀ (z : ℝ), 2^(z * (1 - z)) ≤ 2^y) ∧ y = (1 : ℝ) / 4 :=
begin
  let f := λ x : ℝ, 2^((1 / 4 : ℝ)),
  use f 1,
  split,
  {
    intros,
    let g := λ z : ℝ, -z^2 + z,
    have hg : ∀ z : ℝ, g z ≤ 1 / 4,
    { 
      sorry 
    },
    calc 2^((z * (1 - z)) : ℝ) ≤ 2^((1 / 4 : ℝ)) : by sorry
  },
  {
    sorry
  }
end

end max_value_of_2_power_x_1_minus_x_l129_129963


namespace tangents_from_A_to_circle_sum_l129_129804

noncomputable def circle_radius : ℝ := 7
noncomputable def circle_center_O : ℝ × ℝ := (0, 0)
noncomputable def point_A : ℝ × ℝ := (15, 0)
noncomputable def distance_OA : ℝ := 15
noncomputable def distance_BC : ℝ := 10

theorem tangents_from_A_to_circle_sum 
  (circle_radius = 7)
  (OA = 15)
  (BC = 10) : 
  let AB := 2 * real.sqrt(11)
  let AC := 2 * real.sqrt(11)
  in AB + AC + BC = 8 * real.sqrt(11) + 10 :=
by 
  sorry

end tangents_from_A_to_circle_sum_l129_129804


namespace directrix_eq_l129_129451

noncomputable def parabola_eq : (ℝ → ℝ) := λ x, (x^2 - 8 * x + 12) / 16

theorem directrix_eq : ∀ (y : ℝ), y = parabola_eq (x : ℝ) → ∃ d, d = -1 / 2 := by
  sorry

end directrix_eq_l129_129451


namespace triangle_inequality_satisfied_for_n_six_l129_129030

theorem triangle_inequality_satisfied_for_n_six :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ a + c > b ∧ b + c > a) := sorry

end triangle_inequality_satisfied_for_n_six_l129_129030


namespace roots_on_unit_circle_l129_129985

noncomputable def P (n : ℕ) (x : ℂ) : ℂ :=
  (2 * n : ℂ) * x^(2 * n) + (2 * n - 1) * x^(2 * n - 1) +
  ∑ i in Finset.range (n - 1), (n + 1 + i) * (x^(n + 1 + i) + x^(n - 1 - i)) + 
  n * x^n + (n + 1) * x^(n - 1) + ∑ i in Finset.range (n - 1), (n - 1 - i) * x^((n-1) - i) + 
  (2*n - 1) * x + (2*n : ℂ)

theorem roots_on_unit_circle (n : ℕ) (hn : 0 < n) (r : ℂ) (hr : P n r = 0) : 
  |r| = 1 :=
sorry

end roots_on_unit_circle_l129_129985


namespace find_x_l129_129561

noncomputable def midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def on_circle (mid : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  distance mid p = radius

theorem find_x (x : ℝ) (radius : ℝ) (hx : on_circle (midpoint (x, 0) (-8, 0)) radius (x, 0))
  (Hmax_radius : radius = 8) :
  x = -24 :=
sorry

end find_x_l129_129561


namespace find_value_of_2x_minus_y_l129_129488

theorem find_value_of_2x_minus_y (x y : ℝ) (h1 : 5^x = 3) (h2 : y = log 5 (9 / 25)) : 2 * x - y = 2 := by
  sorry

end find_value_of_2x_minus_y_l129_129488


namespace find_R_l129_129850

theorem find_R (R x y z : ℤ) (h1 : R > x) (h2 : x > y) (h3 : y > z) (h4 : 16 * (2^R + 2^x + 2^y + 2^z) = 330) : R = 4 :=
by 
  sorry

end find_R_l129_129850


namespace find_k_prove_monotonicity_find_t_l129_129970

open Real

-- Given conditions
noncomputable def f (x : ℝ) : ℝ := k * 2^(x+1) + (k-3) * 2^(-x)

-- Proof problems
theorem find_k (h_odd : ∀ x, f(-x) = -f(x)) : k = 1 := sorry

theorem prove_monotonicity (k_eq_one : k = 1) : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := sorry

theorem find_t (k_eq_one : k = 1) (f_mono : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂)
  (h_ineq : ∀ x ∈ (Icc 1 3 : Set ℝ), f (x^2 - x) + f (t * x + 4) > 0) : t > -3 := sorry

end find_k_prove_monotonicity_find_t_l129_129970


namespace distance_from_point_to_polar_line_l129_129148

-- Define the polar point and the polar line equation
def polar_point : ℝ × ℝ := (2, 5 * Real.pi / 6)
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 3) = 4

-- Function to find the distance from a point to a line in polar coordinates.
noncomputable def distance_from_point_to_line_polar (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
sorry -- Implementation skipped

-- The theorem to prove the distance
theorem distance_from_point_to_polar_line :
  distance_from_point_to_line_polar polar_point polar_line = 2 := 
sorry

end distance_from_point_to_polar_line_l129_129148


namespace valid_triangle_inequality_l129_129038

theorem valid_triangle_inequality (n : ℕ) (h : n = 6) :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c →
  n * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  intros a b c ha hb hc hineq
  have h₁ : n = 6 := h
  simplify_eq [h₁] at hineq
  have h₂ := nat.add_comm a b
  exact sorry

end valid_triangle_inequality_l129_129038


namespace calculate_expression_l129_129397

theorem calculate_expression :
  1.1 ^ 0 + real.cbrt 216 - 0.5 ^ (-2) + real.log 25 / real.log 10 + 2 * (real.log 2 / real.log 10) = 5 :=
by sorry

end calculate_expression_l129_129397


namespace find_x_intervals_l129_129435

noncomputable def a (x: ℝ) := x^3 - 100 * x
noncomputable def b (x: ℝ) := x^4 - 16
noncomputable def c (x: ℝ) := x + 20 - x^2

theorem find_x_intervals:
  (∀ x : ℝ, x > -10 ∧ x < 0 → median (a x) (b x) (c x) > 0) ∧
  (∀ x : ℝ, x > 2 ∧ x < 5 → median (a x) (b x) (c x) > 0) ∧
  (∀ x : ℝ, x > 10 → median (a x) (b x) (c x) > 0) :=
sorry

end find_x_intervals_l129_129435


namespace triangle_inequality_satisfied_for_n_six_l129_129031

theorem triangle_inequality_satisfied_for_n_six :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ a + c > b ∧ b + c > a) := sorry

end triangle_inequality_satisfied_for_n_six_l129_129031


namespace emma_prob_at_least_one_correct_l129_129928

-- Define the probability of getting a question wrong
def prob_wrong : ℚ := 4 / 5

-- Define the probability of getting all five questions wrong
def prob_all_wrong : ℚ := prob_wrong ^ 5

-- Define the probability of getting at least one question correct
def prob_at_least_one_correct : ℚ := 1 - prob_all_wrong

-- Define the main theorem to be proved
theorem emma_prob_at_least_one_correct : prob_at_least_one_correct = 2101 / 3125 := by
  sorry  -- This is where the proof would go

end emma_prob_at_least_one_correct_l129_129928


namespace find_orig_denominator_l129_129374

-- Definitions as per the conditions
def orig_numer : ℕ := 2
def mod_numer : ℕ := orig_numer + 3

-- The modified fraction yields 1/3
def new_fraction (d : ℕ) : Prop :=
  (mod_numer : ℚ) / (d + 4) = 1 / 3

-- Proof Problem Statement
theorem find_orig_denominator (d : ℕ) : new_fraction d → d = 11 :=
  sorry

end find_orig_denominator_l129_129374


namespace problem1_solution_problem2_expected_value_solution_problem2_variance_solution_l129_129920

noncomputable def problem1 (P : ProbabilityMassFunction (Fin 6)) : ℝ :=
let P_white := 1 / 3 in
let P_black := 1 - P_white in
P_white * P_black + P_black * P_white

theorem problem1_solution : 
  problem1 = 4 / 9 := 
sorry

noncomputable def problem2_expected_value (P : ProbabilityMassFunction (Fin 2)) : ℝ :=
let P_xi_0 := (4 / 6) * (3 / 5) in
let P_xi_1 := (4 / 6) * (2 / 5) + (2 / 6) * (4 / 5) in
let P_xi_2 := (2 / 6) * (1 / 5) in
0 * P_xi_0 + 1 * P_xi_1 + 2 * P_xi_2

theorem problem2_expected_value_solution : 
  problem2_expected_value = 2 / 3 := 
sorry

noncomputable def problem2_variance (P : ProbabilityMassFunction (Fin 2)) : ℝ :=
let expected_value := problem2_expected_value in
let P_xi_0 := (4 / 6) * (3 / 5) in
let P_xi_1 := (4 / 6) * (2 / 5) + (2 / 6) * (4 / 5) in
let P_xi_2 := (2 / 6) * (1 / 5) in
(0 - expected_value)^2 * P_xi_0 + (1 - expected_value)^2 * P_xi_1 + (2 - expected_value)^2 * P_xi_2

theorem problem2_variance_solution : 
  problem2_variance = 16 / 45 := 
sorry

end problem1_solution_problem2_expected_value_solution_problem2_variance_solution_l129_129920


namespace problem_statement_l129_129518

def line (k : ℝ) (x y : ℝ) : Prop := k * x - y - k + 1 = 0
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def fixed_point_p := (-1, 1)
def fixed_point_q := (1, 1)
def min_MN := 2 * Real.sqrt 3

theorem problem_statement (k : ℝ) :
  (¬ ∀ k, ∃ x y, line k x y ∧ (x, y) = fixed_point_p) ∧
  (∀ k, ∃ x y, line k x y ∧ circle x y) ∧
  (∃ M N : ℝ × ℝ, (M ≠ N) ∧ line k M.1 M.2 ∧ circle M.1 M.2 ∧ 
    line k N.1 N.2 ∧ circle N.1 N.2 ∧
    dist M N = min_MN) ∧
  (∀ k, ¬ ∃ p q : ℝ × ℝ, p ≠ q ∧ circle p.1 p.2 ∧ circle q.1 q.2 ∧ 
    (p.1 + q.1) / 2 = (k * (p.2 + q.2) + k - 1) / (k + 1))
:= sorry

end problem_statement_l129_129518


namespace ratio_triangle_areas_l129_129384

theorem ratio_triangle_areas (M : ℝ) :
  let area_ABJ := 2 * M
  let area_ADE := 8 * M
  area_ABJ / area_ADE = 1 / 4 :=
by
  -- Definitions and conditions
  let area_ABJ := 2 * M
  let area_ADE := 8 * M
  -- Ratio calculation
  have h : area_ABJ / area_ADE = (2 * M) / (8 * M) := by sorry
  -- Simplify the ratio
  have h2 : (2 * M) / (8 * M) = 2 / 8 := by sorry
  have h3 : 2 / 8 = 1 / 4 := by sorry
  show area_ABJ / area_ADE = 1 / 4 from h3

end ratio_triangle_areas_l129_129384


namespace distance_between_P_and_Q_l129_129888

def point : Type := ℝ × ℝ × ℝ

def distance (P Q : point) : ℝ :=
  let (x1, y1, z1) := P;
  let (x2, y2, z2) := Q;
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem distance_between_P_and_Q :
  distance (-1, 2, -3) (3, -2, -1) = 6 := by
  sorry

end distance_between_P_and_Q_l129_129888


namespace volume_intersection_zero_l129_129727

/-- The set of points satisfying |x| + |y| + |z| ≤ 1. -/
def region1 (x y z : ℝ) : Prop :=
  |x| + |y| + |z| ≤ 1

/-- The set of points satisfying |x| + |y| + |z-2| ≤ 1. -/
def region2 (x y z : ℝ) : Prop :=
  |x| + |y| + |z-2| ≤ 1

/-- The intersection of region1 and region2 forms a region with volume 0. -/
theorem volume_intersection_zero : 
  (∫ x y z, (region1 x y z ∧ region2 x y z)) = 0 := sorry

end volume_intersection_zero_l129_129727


namespace range_m_l129_129502

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 3 then log (x + 1) / log 2
else if -3 ≤ x ∧ x < 0 then log (1 / (1 - x)) / log 2
else 0

def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 2 * x + m

theorem range_m (m : ℝ) :
  (∀ x1 ∈ set.Icc (-3 : ℝ) 3, ∃ x2 ∈ set.Icc (-3 : ℝ) 3, g x2 m = f x1) →
  -13 ≤ m ∧ m ≤ -1 :=
by
  sorry

end range_m_l129_129502


namespace slopes_product_hyperbola_eq_one_l129_129969

theorem slopes_product_hyperbola_eq_one
  (x y : ℝ)
  (h : x^2 - y^2 = 6)
  (h1 : 0 < x)
  (h2 : 0 < y) :
  let k1 := y / (x + sqrt 6)
      k2 := y / (x - sqrt 6)
  in k1 * k2 = 1 := by
  sorry

end slopes_product_hyperbola_eq_one_l129_129969


namespace weight_of_a_and_b_l129_129627

-- Define variables for weights
variables {A B C D E F G H : ℝ}

-- Define conditions as Lean statements
def condition1 : Prop := (A + B + C + F) / 4 = 80
def condition2 : Prop := (A + B + C + F + D + E) / 6 = 82
def condition3 : Prop := G = D + 5
def condition4 : Prop := H = E - 4
def condition5 : Prop := (C + D + E + F + G + H) / 6 = 83

-- Define the theorem
theorem weight_of_a_and_b 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : condition3) 
  (h4 : condition4) 
  (h5 : condition5) :
  A + B = 167 :=
sorry

end weight_of_a_and_b_l129_129627


namespace slope_probability_proof_l129_129949

noncomputable def probability_slope_greater (Q : ℝ × ℝ) : ℝ :=
  if 0 < Q.1 ∧ Q.1 < 1 ∧ 0 < Q.2 ∧ Q.2 < 1 then if (Q.2 - 3/4) / (Q.1 - 1/4) > 3/4 then 1 else 0 else 0

theorem slope_probability_proof :
  let Q := (UniformSpace.uniform [0,1] [0,1]).sample in
  P (probability_slope_greater Q) = 29 / 32 :=
sorry

end slope_probability_proof_l129_129949


namespace smallest_n_for_coprime_elements_l129_129122

theorem smallest_n_for_coprime_elements (m : ℕ) (h_m_pos : 0 < m) :
  ∃ n, (n = 68) ∧ ∀ T : Finset ℕ, (∀ t ∈ T, t ∈ Finset.Icc m (m + 99)) ∧ T.card ≥ 68 →
    ∃ a b c, a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ Nat.coprime a b ∧ Nat.coprime b c ∧ Nat.coprime a c :=
begin
  sorry
end

end smallest_n_for_coprime_elements_l129_129122


namespace geometric_seq_sum_l129_129595

-- Definitions of the conditions
def a (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | _ => (-3)^(n - 1)

theorem geometric_seq_sum : 
  a 0 + |a 1| + a 2 + |a 3| + a 4 = 121 := by
  sorry

end geometric_seq_sum_l129_129595


namespace circle_through_A_Y_Z_passes_through_midpoint_BC_l129_129790

-- Definitions and constructs for the problem
variables (A B C H X Y Z M : Type) [PlaneGeometry A B C H X Y Z M]

-- Defining the conditions
variables (triangle_ABC_is_acute : acute_triangle A B C)
variables (H_is_orthocenter : orthocenter H A B C)
variables (omega_is_circle : circle ω B C H)
variables (Gamma_is_circle_diameter_AH : diameter_circle Gamma A H AH)
variables (X_in_omega_Gamma : Point X ∈ ω ∩ Gamma ∧ X ≠ H)
variables (gamma_is_reflection : γ = reflection_circ_gamma_over_AX Gamma X)
variables (intersection_Y : Point Y ∈ γ ∩ ω ∧ Y ≠ X)
variables (intersection_Z : Point Z ∈ omega ∧ Z ≠ H)
variables (M_is_midpoint_BC : midpoint M B C)

-- The theorem to prove
theorem circle_through_A_Y_Z_passes_through_midpoint_BC : 
  circle_through_points A Y Z M :=
begin
  sorry,
end

end circle_through_A_Y_Z_passes_through_midpoint_BC_l129_129790


namespace smallest_positive_period_of_f_g_max_on_interval_g_min_on_interval_l129_129091

noncomputable def f (x : ℝ) : ℝ := 
  2 * real.sqrt 3 * real.sin ((x / 2) + (real.pi / 4)) * real.cos ((x / 2) + (real.pi / 4)) - real.sin (x + real.pi)

noncomputable def g (x : ℝ) : ℝ := 
  f (x - real.pi / 6)

theorem smallest_positive_period_of_f : 
  is_periodic f (2 * real.pi) :=
sorry

theorem g_max_on_interval : 
  ∃ x ∈ set.Icc 0 real.pi, g x = 2 :=
sorry

theorem g_min_on_interval : 
  ∃ x ∈ set.Icc 0 real.pi, g x = -1 :=
sorry

end smallest_positive_period_of_f_g_max_on_interval_g_min_on_interval_l129_129091


namespace log2_P_equals_614_519_l129_129587

open Real

noncomputable def P : ℕ :=
-- define P as the number of ordered partitions of 2013 into prime numbers, use a mathematical function or expression for exact definition

theorem log2_P_equals_614_519 :
  log 2 P = 614.519 := 
sorry

end log2_P_equals_614_519_l129_129587


namespace roots_r_s_l129_129961

theorem roots_r_s (r s : ℝ) (h1 : r + s = 2 * sqrt 3) (h2 : r * s = 2) :
  r^6 + s^6 = 416 := sorry

end roots_r_s_l129_129961


namespace problem1_problem2_l129_129846

-- Definition of complex numbers z1 and z2
def z1 (x : ℝ) : Complex := ⟨2 * x + 1, 2⟩
def z2 (x y : ℝ) : Complex := -⟨x, y⟩

-- First problem statement
theorem problem1 (x y : ℝ) (h1 : z1 x + z2 x y = 0) : x^2 - y^2 = -3 :=
  sorry

-- Definition for purely imaginary
def is_purely_imaginary (z : Complex) : Prop := z.re = 0

-- Second problem statement
theorem problem2 (x : ℝ) (h1 : is_purely_imaginary ((⟨1, 1⟩) * (z1 x))) : Complex.norm (z1 x) = 2 * Real.sqrt 2 :=
  sorry

end problem1_problem2_l129_129846


namespace decagonal_die_expected_value_is_correct_l129_129313

def decagonalDieExpectedValue : ℕ := 5 -- A decagonal die has faces 1 to 10

def expectedValueDecagonalDie : ℝ := 5.5 -- The expected value as calculated.

theorem decagonal_die_expected_value_is_correct (p : fin 10 → ℝ) (i : fin 10) :
  p i = 1 / 10 ∧ (∑ i in finset.univ, p i * (i + 1 : ℝ)) = expectedValueDecagonalDie := by
    sorry

end decagonal_die_expected_value_is_correct_l129_129313


namespace area_of_triangle_ABC_l129_129630

theorem area_of_triangle_ABC :
  let A := (0, 0)
  let B := (4, 0)
  let D := (7, 0)
  let E := (7, 5)
  let C := (98 / 37, 70 / 37)
  let AB := dist A B
  let BD := dist B D
  let ED := dist E D
  let circle_center := (2, 0)
  let circle_radius := 2
  let in_circle := (dist circle_center C = circle_radius)
  let on_AE := C.2 = (5 / 7) * C.1
  in AB = 4 ∧ BD = 3 ∧ ED = 5 ∧ 
     angle E D ((0, 0) - D) = π / 2 ∧ 
     in_circle ∧
     on_AE →
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = 140 / 37 :=
by sorry

end area_of_triangle_ABC_l129_129630


namespace magnitude_of_z_l129_129594

-- Mathematical condition: z + 3i = 3 - i
def complex_condition (z : ℂ) : Prop := z + 3 * complex.i = 3 - complex.i

-- The proof statement: Given the condition, the magnitude |z| is 5.
theorem magnitude_of_z (z : ℂ) (h : complex_condition z) : complex.abs z = 5 := by
  sorry

end magnitude_of_z_l129_129594


namespace find_circle_equation_l129_129438

def point_on_circle (x y xc yc r : ℝ) : Prop := (x - xc) ^ 2 + (y - yc) ^ 2 = r ^ 2

theorem find_circle_equation :
  ∃ (xc yc : ℝ), 
    (point_on_circle 3 2 xc yc (Float.sqrt 5)) ∧ 
    (yc = 2 * xc) ∧ 
    ((point_on_circle xc yc xc (yc - 2 * xc + 5) (Float.sqrt 5)) → 
      ( (xc = 2 ∧ yc = 4 ∧ (point_on_circle 0 0 2 4 (Float.sqrt 5)))
      ∨ 
        (xc = 4 / 5 ∧ yc = 8 / 5 ∧ (point_on_circle 0 0 (4 / 5) (8 / 5) (Float.sqrt 5)))
      )
    ) := 
sorry

end find_circle_equation_l129_129438


namespace trip_time_l129_129751

theorem trip_time (speed_AB : ℝ) (time_AB_min : ℝ) (speed_BA : ℝ) 
    (time_AB_hr : ℝ := time_AB_min / 60) 
    (distance_AB : ℝ := speed_AB * time_AB_hr) 
    (time_BA_hr : ℝ := distance_AB / speed_BA) : 
    speed_AB = 95 → time_AB_min = 186 → speed_BA = 155 → time_AB_hr + time_BA_hr = 5 :=
by
  intros h1 h2 h3
  unfold time_AB_hr distance_AB time_BA_hr
  rw [h1, h2, h3]
  norm_num
  rfl

end trip_time_l129_129751


namespace numDistinctSubsets_l129_129801

def isSpecial (a b : ℕ) : Prop := a + b = 20

def specialFractions : List ℚ :=
  [1 / 19, 1 / 9, 3 / 17, 1 / 4, 1 / 3, 3 / 7, 7 / 13, 2 / 3, 1, 11 / 9,
   3 / 2, 13 / 7, 7 / 3, 7 / 2, 3, 4, 17 / 3, 9, 19]

def sumsOfSpecialFractions : List ℚ :=
  (specialFractions.product specialFractions).map (λ (f1, f2) => f1 + f2)

def distinctIntegerSums : Finset ℕ := 
  (sumsOfSpecialFractions.filterMap (λ q => if q.den = 1 then some q.num else none)).toFinset

theorem numDistinctSubsets : distinctIntegerSums.card = 16 := by
  sorry

end numDistinctSubsets_l129_129801


namespace longest_line_segment_square_in_sector_l129_129371

theorem longest_line_segment_square_in_sector (d : ℝ) (n : ℕ) (h1 : d = 12) (h2 : n = 6) :
  let r := d / 2
      θ := (360 / n : ℝ) / 2
      l := 2 * r * Real.sin (θ * Real.pi / 180) in
  l^2 = 36 :=
by
  -- Defining radius
  let r := d / 2
  -- Central angle in degrees divided by two
  let θ := (360 / n : ℝ) / 2
  -- Longest line segment calculation using the chord length formula
  let l := 2 * r * Real.sin (θ * Real.pi / 180)
  -- The square of the longest line segment
  have h : l^2 = (6 : ℝ)^2, by sorry
  rw [h]
  norm_num
  exact rfl

end longest_line_segment_square_in_sector_l129_129371


namespace magnitude_vector_expression_l129_129898

variables (a b c : ℝ^3)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 2) (hc : ∥c∥ = 3)
variables (h_perp : a ⬝ b = 0)
variables (h_angle_a : a ⬝ c = 3 / 2)
variables (h_angle_b : b ⬝ c = 3)

theorem magnitude_vector_expression :
  ∥a + 2 • b - c∥ = √11 :=
sorry

end magnitude_vector_expression_l129_129898


namespace volume_of_intersection_is_zero_l129_129731

-- Definition of the regions
def region1 (x y z : ℝ) : Prop := abs x + abs y + abs z ≤ 1
def region2 (x y z : ℝ) : Prop := abs x + abs y + abs (z - 2) ≤ 1

-- Volume of the intersection of region1 and region2
theorem volume_of_intersection_is_zero : 
  let volume_intersection : ℝ := 0 
  in volume_intersection = 0 := 
by
  sorry

end volume_of_intersection_is_zero_l129_129731


namespace isosceles_triangle_perimeter_l129_129139

theorem isosceles_triangle_perimeter (a b : ℕ) (h_a : a = 8 ∨ a = 9) (h_b : b = 8 ∨ b = 9) 
(h_iso : a = a) (h_tri_ineq : a + a > b ∧ a + b > a ∧ b + a > a) :
  a + a + b = 25 ∨ a + a + b = 26 := 
by
  sorry

end isosceles_triangle_perimeter_l129_129139


namespace area_of_rectangle_EFGH_l129_129407

noncomputable def side_length_of_smaller_square : ℝ := real.sqrt 4
noncomputable def side_length_of_larger_square : ℝ := side_length_of_smaller_square + 2
noncomputable def area_of_smaller_square : ℝ := side_length_of_smaller_square * side_length_of_smaller_square
noncomputable def area_of_larger_square : ℝ := side_length_of_larger_square * side_length_of_larger_square
noncomputable def width_EFGH : ℝ := side_length_of_smaller_square + side_length_of_larger_square
noncomputable def height_EFGH : ℝ := side_length_of_larger_square
noncomputable def area_EFGH : ℝ := width_EFGH * height_EFGH

theorem area_of_rectangle_EFGH : area_EFGH = 24 := by
  -- You would normally enter the proof steps here, but we'll skip since only the statement is required
  sorry

end area_of_rectangle_EFGH_l129_129407


namespace f_n_iff_power_of_2_l129_129579

def isPowerOf2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

def sumTo (k : ℕ) : ℕ :=
  k * (k + 1) / 2

def leastK (n : ℕ) : ℕ :=
  Nat.find_greatest (λ k, n ∣ sumTo k) n

theorem f_n_iff_power_of_2 (n : ℕ) : leastK(n) = 2 * n - 1 ↔ isPowerOf2(n) := 
by 
  sorry

end f_n_iff_power_of_2_l129_129579


namespace area_r2_eq_l129_129848

-- Definitions of given conditions
def side_r1 := 4 -- inches
def area_r1 := 24 -- square inches
def diag_r2 := 17 -- inches

-- Definitions as per the similarity condition
def side_relation := 1.5 -- ratio between corresponding sides of R1 and R2

-- Main theorem statement
theorem area_r2_eq :
  let a := sqrt (289 / 3.25),
      b := 1.5 * a in
  a * b = 433.5 / 3.25 :=
by {
  let a := sqrt (289 / 3.25),
      b := 1.5 * a;
  sorry
}

end area_r2_eq_l129_129848


namespace identifyIncorrectGuess_l129_129679

-- Define the colors of the bears
inductive BearColor
| white
| brown
| black

-- Conditions as defined in the problem statement
def isValidBearRow (bears : Fin 1000 → BearColor) : Prop :=
  ∀ (i : Fin 998), 
    (bears i = BearColor.white ∨ bears i = BearColor.brown ∨ bears i = BearColor.black) ∧
    (bears ⟨i + 1, by linarith⟩ = BearColor.white ∨ bears ⟨i + 1, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 1, by linarith⟩ = BearColor.black) ∧
    (bears ⟨i + 2, by linarith⟩ = BearColor.white ∨ bears ⟨i + 2, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 2, by linarith⟩ = BearColor.black)

-- Iskander's guesses
def iskanderGuesses (bears : Fin 1000 → BearColor) : Prop :=
  bears 1 = BearColor.white ∧
  bears 19 = BearColor.brown ∧
  bears 399 = BearColor.black ∧
  bears 599 = BearColor.brown ∧
  bears 799 = BearColor.white

-- Exactly one guess is incorrect
def oneIncorrectGuess (bears : Fin 1000 → BearColor) : Prop :=
  ∃ (idx : Fin 5), 
    ¬iskanderGuesses bears ∧
    ∀ (j : Fin 5), (j ≠ idx → (bearGuessesIdx j bears = true))

-- The proof problem
theorem identifyIncorrectGuess (bears : Fin 1000 → BearColor) :
  isValidBearRow bears → iskanderGuesses bears → oneIncorrectGuess bears := sorry

end identifyIncorrectGuess_l129_129679


namespace product_of_divisors_of_72_l129_129463

theorem product_of_divisors_of_72 :
  let n := 72 in
  let prime_factors := [(2, 3), (3, 2)] in
  let divisor_count := (3 + 1) * (2 + 1) in
  divisor_count = 12 →
  (n^(divisor_count / 2) = 2^18 * 3^12) :=
by
  intros
  sorry

end product_of_divisors_of_72_l129_129463


namespace distance_from_p_to_l_max_distance_from_q_to_l_l129_129142

-- Define the constants and assumptions
def parametric_curve_c (θ : ℝ) : (ℝ × ℝ) :=
  (3 * real.sqrt 3 * real.cos θ, real.sqrt 3 * real.sin θ)

def polar_coordinates_p : (ℝ × ℝ) :=
  (2 * real.cos (-real.pi / 3), 2 * real.sin (-real.pi / 3))

def polar_line_equation (ρ θ : ℝ) : Prop :=
  ρ * real.cos ((real.pi / 3) + θ) = 6

def cartesian_line_equation (x y : ℝ) : Prop :=
  x - real.sqrt 3 * y - 12 = 0

-- Formally state the mathematical proof problems
theorem distance_from_p_to_l : 
  let P := polar_coordinates_p in
  let l := cartesian_line_equation in
  ∃ (d : ℝ), d = (|1 + 3 - 12|) / 2 ∧ d = 4 := 
  sorry

theorem max_distance_from_q_to_l :
  let Q := parametric_curve_c in
  let l := cartesian_line_equation in
  ∃ (d_max : ℝ), d_max = max (|6 * real.cos θ - 12| / 2) ∧ d_max = 9 := 
  sorry

end distance_from_p_to_l_max_distance_from_q_to_l_l129_129142


namespace wrong_guess_is_20_l129_129666

-- Define the colors
inductive Color
| white
| brown
| black

-- Assume we have a sequence of 1000 bears
def bears : fin 1000 → Color := sorry

-- Hypotheses
axiom colors_per_three : ∀ (i : fin 998), 
  ({bears i, bears (i + 1), bears (i + 2)} = {Color.white, Color.brown, Color.black} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.black, Color.white, Color.brown} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.brown, Color.black, Color.white})

axiom exactly_one_wrong : 
  (bears 1 = Color.white ∧ bears 19 ≠ Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 ≠ Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 ≠ Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 ≠ Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 ≠ Color.white)

-- Define the theorem to prove
theorem wrong_guess_is_20 : 
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) →
  ¬(bears 19 = Color.brown) := 
sorry

end wrong_guess_is_20_l129_129666


namespace area_of_shaded_region_l129_129292

theorem area_of_shaded_region (r_small : ℝ) (r_large : ℝ)
  (A B C D : ℝ → ℝ) (AB_diameter : r_small * 2)
  (CD_eq_sqroot_5 : CD = sqrt (3^2 - 2^2))
  (angle_CAD : real.angle (2 / 3)) :
  ∃ area : ℝ, area = 1.5276 * real.pi - 10.88 :=
by
  sorry

end area_of_shaded_region_l129_129292


namespace part_a_part_b_l129_129714

noncomputable def p_n (n k : ℕ) : ℕ := sorry  -- Function to count permutations with k fixed points.
noncomputable def R (s i : ℕ) : ℕ := sorry  -- Function to count the partitions.

theorem part_a (n : ℕ) : 
  ∑ k in Finset.range (n + 1), k * p_n n k = Nat.factorial n :=
sorry

theorem part_b (n s : ℕ) : 
  ∑ k in Finset.range (n + 1), k^s * p_n n k = Nat.factorial n * (∑ i in Finset.range (min s n + 1), R s i) :=
sorry

end part_a_part_b_l129_129714


namespace simplify_sqrt_expression_l129_129212

theorem simplify_sqrt_expression :
  ∀ (x : ℝ), sqrt (1 + 2 * sin (real.pi - 2) * cos (real.pi - 2)) = sin 2 - cos 2 :=
by
  intro x
  sorry

end simplify_sqrt_expression_l129_129212


namespace club_of_club_eq_4_10_values_l129_129582

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def club (x : ℕ) : ℕ :=
  sum_of_digits x

def club_of_club_is_4 (x : ℕ) : Prop :=
  club (club x) = 4

theorem club_of_club_eq_4_10_values :
  {x : ℕ | is_two_digit x ∧ club_of_club_is_4 x}.toFinset.card = 10 := 
  sorry

end club_of_club_eq_4_10_values_l129_129582


namespace unique_zero_point_range_l129_129511

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 - 2*x + b

theorem unique_zero_point_range (b : ℝ) :
  (∃! x ∈ set.Ioo (2:ℝ) (4:ℝ), f x b = 0) ↔ -8 < b ∧ b < 0 := 
by
  sorry

end unique_zero_point_range_l129_129511


namespace remainder_when_s_10_is_100_mod_l129_129591

def q (x : ℚ) : ℚ := x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

def t (x : ℚ) : ℚ := x^3 + x^2 + 1

def s (x : ℚ) : ℚ :=
  (q(x) % t(x)).mod_by // ensure this is the right polynomial remainder operator if Lean 4 library has other specific remainder operations

theorem remainder_when_s_10_is_100_mod :
  (abs (s 10)) % 100 = 23 :=
sorry

end remainder_when_s_10_is_100_mod_l129_129591


namespace evaluate_expression_l129_129022

theorem evaluate_expression :
  (Int.floor ((Int.ceil ((11/5:ℚ)^2)) * (19/3:ℚ))) = 31 :=
by
  sorry

end evaluate_expression_l129_129022


namespace tangent_line_intercept_l129_129711

theorem tangent_line_intercept:
  ∃ (m b : ℚ), 
    m > 0 ∧ 
    b = 135 / 28 ∧ 
    (∀ x y : ℚ, (y - 3)^2 + (x - 1)^2 ≥ 3^2 → (y - 8)^2 + (x - 10)^2 ≥ 6^2 → y = m * x + b) := 
sorry

end tangent_line_intercept_l129_129711


namespace angle_C_in_parallelogram_l129_129553

theorem angle_C_in_parallelogram (A B C D : Type) (ABCD_is_parallelogram : Parallelogram A B C D)
  (angle_ratio : ∃ k, angle A = 2*k ∧ angle B = 7*k) :
  angle C = 40 :=
by
  sorry

end angle_C_in_parallelogram_l129_129553


namespace find_m_l129_129890

theorem find_m (m : ℤ) (ha : vector ℤ 2 := ![m, 4]) (hb : vector ℤ 2 := ![3, -2]) :
  (m = -6) ↔ ha = λ i, 3 * hb i := 
by
  sorry

end find_m_l129_129890


namespace center_equals_focus_l129_129473

-- Given the center of the circle and the focus of the parabola
def center_of_circle (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

def focus_of_parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x ∧ y = 0 ∧ x = p / 2

theorem center_equals_focus (p : ℝ) (p_pos : 0 < p) :
  center_of_circle 3 0 ∧ focus_of_parabola p (p / 2) 0 → p = 6 :=
by
  intro h
  have h_center : center_of_circle 3 0 := and.left h
  have h_focus : focus_of_parabola p (p / 2) 0 := and.right h
  sorry

end center_equals_focus_l129_129473


namespace sum_of_angles_l129_129633

-- Definitions of the key components
variables {Point : Type} [MetricSpace Point]
variables {A B C D E : Point}
variables {angle : Point → Point → Point → ℝ}

-- Conditions of the problem
def is_right_triangle (A B C : Point) : Prop := 
  ∃ (r : ℝ), angle A C B = 90

def divided_into_three_equal_parts (B C D E : Point) : Prop := 
  dist B D = dist D E ∧ dist D E = dist E C

def length_relation (A B C : Point) : Prop := 
  dist C B = 3 * dist A C

-- The proof problem
theorem sum_of_angles (h1 : is_right_triangle A B C) (h2 : divided_into_three_equal_parts B C D E)
    (h3 : length_relation A B C) :
    angle A E C + angle A D C + angle A B C = 90 := 
begin
  sorry
end

end sum_of_angles_l129_129633


namespace perpendicular_vec_l129_129885

def vec_a : ℝ × ℝ := (3, 1)
def vec_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vec_c : ℝ × ℝ := (0, 2)

def vec_diff (x : ℝ) := (vec_b x).1 - vec_c.1, (vec_b x).2 - vec_c.2

theorem perpendicular_vec {x : ℝ} (h : vec_a.1 * (vec_diff x).1 + vec_a.2 * (vec_diff x).2 = 0) : 
  x = 4 / 3 :=
sorry

end perpendicular_vec_l129_129885


namespace route_Y_quicker_than_route_X_l129_129600

theorem route_Y_quicker_than_route_X : 
  let t_X := (8 / 25) * 60 -- time taken for Route X in minutes
  let t_Y := ((6 / 35) * 60) + ((1 / 15) * 60) + 2 -- time taken for Route Y in minutes
  t_X - t_Y ≈ 2.9 :=
by {
  sorry
}

end route_Y_quicker_than_route_X_l129_129600


namespace angle_scaled_vectors_l129_129906

variables (a b : ℝ → ℝ) -- assuming a and b are vectors (ℝ → ℝ kind e.g. transforming real numbers)
noncomputable def angle (v1 v2 : ℝ → ℝ) : ℝ := sorry -- placeholder for angle calculation between two vectors

theorem angle_scaled_vectors (h : angle a b = 60) : angle (λ x, 2 * a x) (λ x, - b x) = 120 :=
by sorry

end angle_scaled_vectors_l129_129906


namespace actual_yield_is_7_percent_l129_129758

-- Define the given conditions 
def initial_deposit (P : ℝ) : ℝ := P

def interest_first_3_months (P : ℝ) : ℝ := P * 1.03

def interest_second_3_months (P : ℝ) : ℝ := (P * 1.03) * 1.02

def interest_last_6_months (P : ℝ) : ℝ := ((P * 1.03) * 1.02) * 1.03

def final_amount (P : ℝ) : ℝ := ((P * 1.03) * 1.02) * 1.03

def final_amount_after_fee (P : ℝ) : ℝ := ((P * 1.03 * 1.02 * 1.03) - 0.01 * P)

def effective_return_rate (P : ℝ) : ℝ := (final_amount_after_fee P) / P - 1

-- The theorem to prove the actual annual yield that a depositor will receive
theorem actual_yield_is_7_percent (P : ℝ) (hP : P > 0) :
  effective_return_rate P ≈ 0.072118 :=
by
  sorry

end actual_yield_is_7_percent_l129_129758


namespace angle_AFE_is_175_l129_129146

-- Definitions of points and angles in the problem
variables (A B C D E F: Point)
variables (AB BC CD DE DF: ℝ)
variables (angle_CDE: ℝ)
variables (AFE: ℝ)

-- Define the conditions
def rectangle_ABCD : Prop :=
  rectangle A B C D ∧ AB = 2 * BC

def point_E_conditions : Prop :=
  yes_point_is_opposite_half_plane E A D C ∧ angle_CDE = 100

def point_F_conditions : Prop :=
  point_on AD F ∧ DE = DF 

-- The main theorem 
theorem angle_AFE_is_175 :
  rectangle_ABCD ∧ point_E_conditions ∧ point_F_conditions →
  AFE = 175 :=
sorry

end angle_AFE_is_175_l129_129146


namespace find_circle_equation_l129_129121

-- Definition of tangent line to a circle at the center
def is_tangent_to_line (h k r : ℝ) (A B C : ℝ) : Prop :=
  abs (A * h + B * k + C) / real.sqrt (A^2 + B^2) = r

-- Definitions given in the problem
def circle_conditions (h k : ℝ) : Prop :=
  (is_tangent_to_line h k (real.sqrt 2 * h) 1 (-1) 0) ∧
  (is_tangent_to_line h k (real.sqrt 2 * h) 1 (-1) (-4)) ∧
  (k = -h)

-- The theorem stating the problem
theorem find_circle_equation (h k : ℝ) (h_eq : circle_conditions h k) : 
  (h = 1 ∧ k = -1) → 
  (∀ x y : ℝ, (x - h)^2 + (y + k)^2 = 2 → (x - 1)^2 + (y + 1)^2 = 2) :=
by
  sorry

end find_circle_equation_l129_129121


namespace incorrect_guess_l129_129693

-- Define the conditions
def bears : ℕ := 1000

inductive Color
| White
| Brown
| Black

constant bear_color : ℕ → Color -- The color of the bear at each position

axiom condition : ∀ n : ℕ, n < bears - 2 → 
  ∃ i j k, (i, j, k ∈ {Color.White, Color.Brown, Color.Black}) ∧ 
  (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ 
  (bear_color n = i ∧ bear_color (n+1) = j ∧ bear_color (n+2) = k) 

constants (g1 : bear_color 2 = Color.White)
          (g2 : bear_color 20 = Color.Brown)
          (g3 : bear_color 400 = Color.Black)
          (g4 : bear_color 600 = Color.Brown)
          (g5 : bear_color 800 = Color.White)

-- The proof problem
theorem incorrect_guess : bear_color 20 ≠ Color.Brown :=
by sorry

end incorrect_guess_l129_129693


namespace isosceles_triangle_base_length_l129_129234

-- Definitions based on the conditions
def congruent_side : Nat := 7
def perimeter : Nat := 23

-- Statement to prove
theorem isosceles_triangle_base_length :
  let b := perimeter - 2 * congruent_side in b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l129_129234


namespace cylinder_height_l129_129355

theorem cylinder_height (base_area : ℝ) (h s : ℝ)
  (h_base : base_area > 0)
  (h_ratio : (1 / 3 * base_area * 4.5) / (base_area * h) = 1 / 6)
  (h_cone_height : s = 4.5) :
  h = 9 :=
by
  -- Proof omitted
  sorry

end cylinder_height_l129_129355


namespace square_of_1003_l129_129003

theorem square_of_1003 : 1003^2 = 1006009 :=
by {
    -- General method that can be used: expanding the binomial (1000 + 3)^2 
    -- However, we are directly stating the result here for Lean verification.
    calc
    1003^2 = (1000 + 3)^2          : by rw [add_sq, mul_two, mul_assoc, add_assoc]
    ...    = 1000^2 + 2 * 1000 * 3 + 3^2 : by rw [sq]
    ...    = 1000000 + 6000 + 9     : by ring
    ...    = 1006009               : by norm_num
}

end square_of_1003_l129_129003


namespace sphere_surface_area_l129_129369

theorem sphere_surface_area (a : ℝ) (d : ℝ) (S : ℝ) : 
  a = 3 → d = Real.sqrt 7 → S = 40 * Real.pi := by
  sorry

end sphere_surface_area_l129_129369


namespace coefficient_of_8th_term_l129_129225

-- Define the general term of the binomial expansion
def binomial_expansion_term (n r : ℕ) (a b : ℕ) : ℕ := 
  Nat.choose n r * a^(n - r) * b^r

-- Define the specific scenario given in the problem
def specific_binomial_expansion_term : ℕ := 
  binomial_expansion_term 8 7 2 1  -- a = 2, b = x (consider b as 1 for coefficient calculation)

-- Problem statement to prove the coefficient of the 8th term is 16
theorem coefficient_of_8th_term : specific_binomial_expansion_term = 16 := by
  sorry

end coefficient_of_8th_term_l129_129225


namespace first_nonzero_digit_of_one_over_199_l129_129718

theorem first_nonzero_digit_of_one_over_199 :
  (∃ n : ℕ, (n < 10) ∧ (rat.of_int 2 / rat.of_int 100 < 1 / rat.of_int 199) ∧ (1 / rat.of_int 199 < rat.of_int 3 / rat.of_int 100)) :=
sorry

end first_nonzero_digit_of_one_over_199_l129_129718


namespace complex_min_value_l129_129175

theorem complex_min_value (z : ℂ) (hz : |z - (3 - 2 * I)| = 4) :
  |z + 1 - I|^2 + |z - 7 + 3 * I|^2 = 94 :=
by
  sorry

end complex_min_value_l129_129175


namespace minimum_value_of_expression_l129_129841

theorem minimum_value_of_expression (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    ∃ (c : ℝ), (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x^3 + y^3 - 5 * x * y ≥ c) ∧ c = -125 / 27 :=
by
  sorry

end minimum_value_of_expression_l129_129841


namespace f_at_three_bounds_l129_129862

theorem f_at_three_bounds (a c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^2 - c)
  (h2 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h3 : -1 ≤ f 2 ∧ f 2 ≤ 5) : -1 ≤ f 3 ∧ f 3 ≤ 20 :=
sorry

end f_at_three_bounds_l129_129862


namespace smallest_non_palindrome_product_is_111_l129_129051

def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString
  s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop := 
  100 ≤ n ∧ n < 1000 ∧ is_palindrome n

def smallest_non_palindromic_product_palindrome (n: ℕ) : Prop :=
  is_three_digit_palindrome n ∧ ¬ is_palindrome (n * 131)

theorem smallest_non_palindrome_product_is_111 : ∀ n : ℕ, smallest_non_palindromic_product_palindrome n → n = 111 :=
by
  sorry

end smallest_non_palindrome_product_is_111_l129_129051


namespace sum_of_leading_digits_of_roots_l129_129169

def M := 5555...555 -- 303-digit number, each digit being 5
def g (r : ℕ) : ℕ := leading_digit (root r M)

theorem sum_of_leading_digits_of_roots :
  g 2 + g 3 + g 4 + g 5 + g 6 = 10 := by
  sorry

end sum_of_leading_digits_of_roots_l129_129169


namespace quadratic_has_two_distinct_roots_l129_129123

theorem quadratic_has_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 2*x₁ + k = 0) ∧ (x₂^2 - 2*x₂ + k = 0))
  ↔ k < 1 :=
by sorry

end quadratic_has_two_distinct_roots_l129_129123


namespace odd_function_expression_l129_129496

/-- Given an odd function f defined on ℝ, when x > 0, f(x) = x^2 + |x| - 1,
 prove that for x < 0, the expression for f(x) is -x^2 - |x| + 1. -/
theorem odd_function_expression (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 < x → f x = x^2 + |x| - 1) :
  ∀ x, x < 0 → f x = -x^2 - |x| + 1 := 
begin
  intros x hx,
  have hx_pos : 0 < -x, from lt_trans hx (neg_neg' zero_lt_one),
  rw ←neg_neg x at hx_pos,
  rw [←h_odd (-x), h_pos (-x) hx_pos],
  simp,  -- Optional: simplifies the absolute value
  sorry
end

end odd_function_expression_l129_129496


namespace zero_in_interval_l129_129277

noncomputable def f (x : ℝ) := Real.log x / Real.log (1/2) - x + 4

theorem zero_in_interval :
  ∃ x ∈ Ioo (2 : ℝ) 3, f x = 0 :=
begin
  sorry
end

end zero_in_interval_l129_129277


namespace a23_is_5_over_7_l129_129493

noncomputable def sequence (a : ℕ → ℚ) : (ℕ → ℚ) :=
  λ n, if 0 ≤ a n ∧ a n < (1 / 2) then 2 * a n else 2 * a n - 1

theorem a23_is_5_over_7 (a : ℕ → ℚ) (h₀ : a 1 = 6/7)
  (h₁ : ∀ n, a (n + 1) = if 0 ≤ a n ∧ a n < (1 / 2) then 2 * a n else 2 * a n - 1) :
a 23 = 5 / 7 :=
sorry

end a23_is_5_over_7_l129_129493


namespace sum_of_real_solutions_l129_129469

theorem sum_of_real_solutions :
  let S := {x : ℝ | |x - 1| = 3 * |x + 1|} in
  (∑ x in S, x) = -5 / 2 :=
by sorry

end sum_of_real_solutions_l129_129469


namespace perpendicular_altitudes_l129_129494

open EuclideanGeometry Point

-- Definitions for the conditions of the problem
variables {Point : Type} [EuclideanGeometry Point]
variables (A B C D E F : Point) -- Points in the Euclidean plane
variables (ABC_acute : acute_triangle A B C) -- A given acute triangle ABC
variables (D_on_BC : on_line D B C) -- D on line BC
variables (E_on_CA : on_line E C A) -- E on line CA
variables (F_on_AB : on_line F A B) -- F on line AB
variables (DA_bisector_DEF : angle_bisector D F E A) -- DA is angle bisector of angle DEF
variables (EB_bisector_DEF : angle_bisector E D F B) -- EB is angle bisector of angle DEF
variables (FC_bisector_DEF : angle_bisector F E D C) -- FC is angle bisector of angle DEF

-- Statement to prove
theorem perpendicular_altitudes (h : h := DA_bisector_DEF ∧ EB_bisector_DEF ∧ FC_bisector_DEF) : 
  perpendicular (DA A) B C ∧ perpendicular (EB B) C A ∧ perpendicular (FC C) A B :=
by
  sorry

end perpendicular_altitudes_l129_129494


namespace isosceles_triangle_base_length_l129_129235

-- Definitions based on the conditions
def congruent_side : Nat := 7
def perimeter : Nat := 23

-- Statement to prove
theorem isosceles_triangle_base_length :
  let b := perimeter - 2 * congruent_side in b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l129_129235


namespace wrong_guess_is_20_l129_129668

-- Define the colors
inductive Color
| white
| brown
| black

-- Assume we have a sequence of 1000 bears
def bears : fin 1000 → Color := sorry

-- Hypotheses
axiom colors_per_three : ∀ (i : fin 998), 
  ({bears i, bears (i + 1), bears (i + 2)} = {Color.white, Color.brown, Color.black} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.black, Color.white, Color.brown} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.brown, Color.black, Color.white})

axiom exactly_one_wrong : 
  (bears 1 = Color.white ∧ bears 19 ≠ Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 ≠ Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 ≠ Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 ≠ Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 ≠ Color.white)

-- Define the theorem to prove
theorem wrong_guess_is_20 : 
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) →
  ¬(bears 19 = Color.brown) := 
sorry

end wrong_guess_is_20_l129_129668


namespace problem_statement_l129_129075

variable {n : ℕ}
variable {x : Fin n → ℝ}

theorem problem_statement (h1 : ∀ i, 0 ≤ x i) (h2 : n ≥ 3) (h3 : (Finset.univ.sum (λ i, x i) = 1)) :
  (Finset.univ.sum (λ i, x i * x (i + 1 % n) ^ 2)) ≤ 4 / 27 :=
sorry

end problem_statement_l129_129075


namespace triangle_area_ratio_correct_l129_129153

noncomputable def triangle_area_ratio 
    (A B C D E : Type) 
    [HasArea A] [HasArea B] [HasArea C] [HasArea D] [HasArea E]
    (area_ABC : HasArea A)
    (area_ADE : HasArea B)
    (area_CDE : HasArea C)
    (area_DBC : HasArea D)
    (parallel_DE_BC : Prop)
    (area_ratio_ADE_CDE : ℝ) 
    : ℝ :=
  if h1 : parallel_DE_BC ∧ area_ratio_ADE_CDE = 1 / 3 then 
    1 / 12 
  else 
    0

theorem triangle_area_ratio_correct 
    (A B C D E : Type) 
    [HasArea A] [HasArea B] [HasArea C] [HasArea D] [HasArea E]
    (area_ABC : HasArea A)
    (area_ADE : HasArea B)
    (area_CDE : HasArea C)
    (area_DBC : HasArea D)
    (parallel_DE_BC : Prop)
    (area_ratio_ADE_CDE : ℝ) :
  parallel_DE_BC ∧ area_ratio_ADE_CDE = 1 / 3 → 
  triangle_area_ratio A B C D E area_ABC area_ADE area_CDE area_DBC parallel_DE_BC area_ratio_ADE_CDE = 1 / 12 := 
by 
  intros h
  simp [triangle_area_ratio, h]


end triangle_area_ratio_correct_l129_129153


namespace volume_of_solid_l129_129268

-- Define the parameters and conditions of the problem
def side_length : ℝ := 6 * Real.sqrt 2
def upper_edge_length : ℝ := 2 * side_length

-- State the theorem to be proven
theorem volume_of_solid : 
  let V := (Real.sqrt 2 * (2 * side_length)^3) / 12 in
  V / 2 = 288 :=
by
  -- proof goes here
  sorry

end volume_of_solid_l129_129268


namespace nat_solution_count_eq_27_l129_129897

/-- Proves the number of solutions to the equation (a+1)(b+1)(c+1)=2abc in natural numbers equals 27. -/
theorem nat_solution_count_eq_27 :
  {abc : ℕ × ℕ × ℕ // (abc.1 + 1) * (abc.2 + 1) * (abc.3 + 1) = 2 * abc.1 * abc.2 * abc.3} = 27 := 
sorry

end nat_solution_count_eq_27_l129_129897


namespace no_valid_tiling_4x5_l129_129786

-- Definitions of the tetromino types in Lean
inductive Tetromino
| I
| Square
| Z
| T
| L
deriving DecidableEq

def valid_tiling (rect : ℕ × ℕ) (tiles : List Tetromino) : Prop :=
  ∃ arrangement : rect.fst × rect.snd → Option Tetromino,
    ∀ pos, pos.fst < rect.fst ∧ pos.snd < rect.snd →
      arrangement pos ≠ none →
      arrangement pos ∈ tiles

theorem no_valid_tiling_4x5 : ¬ (valid_tiling (4, 5) [Tetromino.I, Tetromino.Square, Tetromino.Z, Tetromino.T, Tetromino.L]) :=
begin
  -- Proof goes here
  sorry
end

end no_valid_tiling_4x5_l129_129786


namespace distribution_ways_l129_129377

def number_of_ways_to_distribute_problems : ℕ :=
  let friends := 10
  let problems := 7
  let max_receivers := 3
  let ways_to_choose_friends := Nat.choose friends max_receivers
  let ways_to_distribute_problems := max_receivers ^ problems
  ways_to_choose_friends * ways_to_distribute_problems

theorem distribution_ways :
  number_of_ways_to_distribute_problems = 262440 :=
by
  -- Proof is omitted
  sorry

end distribution_ways_l129_129377


namespace problem1_problem2_l129_129346

noncomputable def calculate_a (b : ℝ) (angleB : ℝ) (angleC : ℝ) : ℝ :=
  let angleA := 180 - (angleB + angleC)
  let sinA := Real.sin (angleA * Real.pi / 180)
  (b * sinA) / (Real.sin(angleB * Real.pi / 180))

theorem problem1 (b : ℝ) (angleB : ℝ) (angleC : ℝ) : 
  b = 2 → angleB = 30 → angleC = 135 → 
  calculate_a b angleB angleC = Real.sqrt 6 - Real.sqrt 2 := 
sorry

noncomputable def calculate_angleC (a b c : ℝ) : ℝ :=
  let S_triangle_ABC := (a^2 + b^2 - c^2) / 4
  let sinC := (a^2 + b^2 - c^2) / (2 * a * b)
  let cosC := (a^2 + b^2 - c^2) / (2 * a * b)
  if sinC = cosC then
    Real.pi / 4
  else
    0 -- this condition catches non-matching (impossible in given context)

theorem problem2 (a b c : ℝ) (S_triangle_ABC : ℝ) : 
  S_triangle_ABC = 1/4 * (a^2 + b^2 - c^2) → 
  calculate_angleC a b c = Real.pi / 4 := 
sorry

end problem1_problem2_l129_129346


namespace sum_series_form_l129_129395

theorem sum_series_form (n : ℕ) : ∑ k in Finset.range n, (1 / ((k + 1 : ℕ) * (k + 2))) = n / (n + 1) := 
sorry

end sum_series_form_l129_129395


namespace element_of_set_l129_129815

theorem element_of_set : 1 ∈ ({0, 1} : set ℕ) :=
by {
  -- Proof goes here
  sorry
}

end element_of_set_l129_129815


namespace identifyIncorrectGuess_l129_129681

-- Define the colors of the bears
inductive BearColor
| white
| brown
| black

-- Conditions as defined in the problem statement
def isValidBearRow (bears : Fin 1000 → BearColor) : Prop :=
  ∀ (i : Fin 998), 
    (bears i = BearColor.white ∨ bears i = BearColor.brown ∨ bears i = BearColor.black) ∧
    (bears ⟨i + 1, by linarith⟩ = BearColor.white ∨ bears ⟨i + 1, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 1, by linarith⟩ = BearColor.black) ∧
    (bears ⟨i + 2, by linarith⟩ = BearColor.white ∨ bears ⟨i + 2, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 2, by linarith⟩ = BearColor.black)

-- Iskander's guesses
def iskanderGuesses (bears : Fin 1000 → BearColor) : Prop :=
  bears 1 = BearColor.white ∧
  bears 19 = BearColor.brown ∧
  bears 399 = BearColor.black ∧
  bears 599 = BearColor.brown ∧
  bears 799 = BearColor.white

-- Exactly one guess is incorrect
def oneIncorrectGuess (bears : Fin 1000 → BearColor) : Prop :=
  ∃ (idx : Fin 5), 
    ¬iskanderGuesses bears ∧
    ∀ (j : Fin 5), (j ≠ idx → (bearGuessesIdx j bears = true))

-- The proof problem
theorem identifyIncorrectGuess (bears : Fin 1000 → BearColor) :
  isValidBearRow bears → iskanderGuesses bears → oneIncorrectGuess bears := sorry

end identifyIncorrectGuess_l129_129681


namespace arithmetic_sequence_ratio_l129_129251

theorem arithmetic_sequence_ratio (a x b : ℝ) 
  (h1 : x - a = b - x)
  (h2 : b - x = 2x - b) :
  a / b = 1 / 3 :=
by
  sorry

end arithmetic_sequence_ratio_l129_129251


namespace probability_of_at_least_one_contract_l129_129644

theorem probability_of_at_least_one_contract
  (P_A : ℝ) (P_B_complement : ℝ) (P_A_inter_B : ℝ) :
  P_A = 4 / 5 →
  P_B_complement = 3 / 5 →
  P_A_inter_B ≈ 0.3 →
  (P_A + (1 - P_B_complement) - P_A_inter_B = 0.9) :=
by
  intros hPA hPBcomplement hPAinterB
  calc
  P_A + (1 - P_B_complement) - P_A_inter_B 
  = (4 / 5) + (1 - (3 / 5)) - 0.3 : by rw [hPA, hPBcomplement, hPAinterB]
  ... = 0.9 : by norm_num

end probability_of_at_least_one_contract_l129_129644


namespace verify_share_trading_l129_129990

noncomputable def stock_profit {shares : ℕ} {buy_price sell_price : ℝ} 
  (stamp_duty_rate transfer_fee_rate commission_rate min_commission : ℝ) : ℝ :=
let cost := shares * buy_price in
let total_sell_price := shares * sell_price in
let total_transaction := cost + total_sell_price in
let stamp_duty := stamp_duty_rate * total_transaction in
let transfer_fee := transfer_fee_rate * total_transaction in
let commission := max min_commission (commission_rate * total_transaction) in
total_sell_price - cost - (stamp_duty + transfer_fee + commission)

def example_share_trading_profit : Prop :=
  stock_profit 1000 5 5.5 0.001 0.001 0.003 5 = 447.5

theorem verify_share_trading : example_share_trading_profit :=
sorry

end verify_share_trading_l129_129990


namespace tangent_line_at_pi_unique_l129_129249

noncomputable def tangent_line_eq_at_pi (x y : ℝ) : Prop :=
  y = 2 * sin x → ∃ m b, (∃ x₀ y₀, x₀ = π ∧ y₀ = 0) ∧
  ∀ x, y = m * x + b → m = -2 ∧ b = 2 * π

theorem tangent_line_at_pi_unique :
  tangent_line_eq_at_pi π 0 :=
sorry

end tangent_line_at_pi_unique_l129_129249


namespace angle_AHB_is_90_l129_129577

open EuclideanGeometry

variables {A B C D E F G H : Point}

variables (square : Square A B C D)
variables (pointOnDC : PointOnSegment E D C)
variables (footF : FootOfAltitude F B A E)
variables (footG : FootOfAltitude G A B E)
variables (intersectionH : Intersection H (Line DF) (Line CG))

theorem angle_AHB_is_90 (H1 : square)
                        (H2 : pointOnDC E D C)
                        (H3 : footF F B A E)
                        (H4 : footG G A B E)
                        (H5 : intersectionH H (Line DF) (Line CG)) :
    right_angle (angle A H B) :=
by sorry

end angle_AHB_is_90_l129_129577


namespace odd_n_zero_l129_129009

variable {a : ℕ → ℝ} 

def c_n (n : ℕ) : ℝ := ∑ i in finset.range 8, a (i+1) ^ n 

theorem odd_n_zero (h_inf : ∀ n : ℕ, ∃ N : ℕ, (∀ k ≥ N, c_n (2*k) > 0) ∧ (c_n (2*k+1) = 0)) :
  ∀ n : ℕ, c_n n = 0 ↔ odd n :=
begin
  -- The proof would go here
  sorry
end

end odd_n_zero_l129_129009


namespace colors_per_box_l129_129803

-- Define the conditions
def friends : ℕ := 5
def total_pencils : ℕ := 42 
def total_people : ℕ := 1 + friends -- Chloe + her 5 friends

-- Define the main question as a theorem
theorem colors_per_box : (total_pencils / total_people) = 7 :=
by
  simp only [total_pencils, total_people, add_comm]
  simp
  exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num) sorry

end colors_per_box_l129_129803


namespace linear_combination_nonzero_l129_129062

open Complex

noncomputable def linear_combination {n : ℕ} (vectors : Fin n → ℂ) (coeffs : Fin n → ℝ) : ℂ :=
  ∑ i in Finset.range n, (coeffs i) • (vectors i)

theorem linear_combination_nonzero {n : ℕ} (hn : 0 < n)
  (vectors : Fin n → ℂ) (coeffs : Fin n → ℝ) 
  (h_vectors : ∀ i, abs (vectors i) = 1) 
  (h_coeffs : ∀ i j, i < j → coeffs i > coeffs j)
  (h_pos : ∀ i, 0 < coeffs i) :
  linear_combination vectors coeffs ≠ 0 := by
  sorry

end linear_combination_nonzero_l129_129062


namespace altitudes_extension_l129_129026

variables {A B C A' B' C' : Type}
variables {a b c a' b' c' : ℝ}
variables {α β γ : ℝ}
variables {t t' : ℝ}

-- Define the conditions
def triangle_sides (ABC A'B'C' : Type) (a b c a' b' c' : ℝ) : Prop :=
  ∃ (ha : ℝ), ha = a ∧ ∃ (hb : ℝ), hb = b ∧ ∃ (hc : ℝ), hc = c ∧
              ∃ (ha' : ℝ), ha' = a' ∧ ∃ (hb' : ℝ), hb' = b' ∧ ∃ (hc' : ℝ), hc' = c'

def triangle_angles (α β γ : ℝ) : Prop :=
  α + β + γ = π

def altitudes_extended_triangle (t t' : ℝ) (α β γ : ℝ) (sides : a b c a' b' c'): Prop :=
  t' = t * (3 + 8 * cos α * cos β * cos γ)

-- Statement to prove:
theorem altitudes_extension
  (h1 : triangle_sides ABC A'B'C' a b c a' b' c')
  (h2 : triangle_angles α β γ)
  (h3 : ∀ (t : ℝ), t = (1 / 2) * a * b * sin γ) :
  a' ^ 2 + b' ^ 2 + c' ^ 2 - (a ^ 2 + b ^ 2 + c ^ 2) = 32 * t * sin α * sin β * sin γ ∧
  t' = t * (3 + 8 * cos α * cos β * cos γ) :=
by
  sorry

end altitudes_extension_l129_129026


namespace kennedy_larger_factor_l129_129942

theorem kennedy_larger_factor :
  ∀ (x : ℝ), (10000 = x * 2350 + 600) → (x = 4) :=
by
  intro x
  intro h
  have h1 : 10000 - 600 = x * 2350 := by linarith
  have h2 : 9400 = x * 2350 := h1
  have h3 : x = 9400 / 2350 := by field_simp [h2]
  have h4 : x = 4 := by norm_num 
  exact h4

end kennedy_larger_factor_l129_129942


namespace min_omega_value_l129_129874

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) := 2 * Real.sin (ω * x + φ)

theorem min_omega_value
  (ω : ℝ) (φ : ℝ)
  (hω : ω > 0)
  (h1 : f (π / 3) ω φ = 0)
  (h2 : f (π / 2) ω φ = 2) :
  ω = 3 :=
sorry

end min_omega_value_l129_129874


namespace find_x_y_l129_129962

theorem find_x_y (A B C : ℝ) (x y : ℝ) (hA : A = 120) (hB : B = 100) (hC : C = 150)
  (hx : A = B + (x / 100) * B) (hy : A = C - (y / 100) * C) : x = 20 ∧ y = 20 :=
by
  sorry

end find_x_y_l129_129962


namespace isosceles_triangle_base_length_l129_129233

-- Definitions based on the conditions
def congruent_side : Nat := 7
def perimeter : Nat := 23

-- Statement to prove
theorem isosceles_triangle_base_length :
  let b := perimeter - 2 * congruent_side in b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l129_129233


namespace volume_of_intersection_l129_129736

def condition1 (x y z : ℝ) : Prop := |x| + |y| + |z| ≤ 1
def condition2 (x y z : ℝ) : Prop := |x| + |y| + |z - 2| ≤ 1
def in_intersection (x y z : ℝ) : Prop := condition1 x y z ∧ condition2 x y z

theorem volume_of_intersection : 
  (∫ x y z in { p : ℝ × ℝ × ℝ | in_intersection p.1 p.2 p.3 }, 1) = 1/12 := 
by
  sorry

end volume_of_intersection_l129_129736


namespace maria_total_flowers_l129_129059

-- Define the initial conditions
def dozens := 3
def flowers_per_dozen := 12
def free_flowers_per_dozen := 2

-- Define the total number of flowers
def total_flowers := dozens * flowers_per_dozen + dozens * free_flowers_per_dozen

-- Assert the proof statement
theorem maria_total_flowers : total_flowers = 42 := sorry

end maria_total_flowers_l129_129059


namespace compare_areas_l129_129519

-- Define the conditions
variable (a k : ℝ)
variable P : ℝ × ℝ := (-1, 1)
variable Q : ℝ × ℝ := (-1/2, 0)
def parabola := λ x : ℝ => a * x^2
def line := λ x : ℝ => k * (x + 1/2)

-- Define the intersection points M and N
variable (M N : ℝ × ℝ)
variable M_between_QN : (M.1 > -1/2) ∧ (M.1 < N.1)

-- Define areas S_1 and S_2
def point_A := (-(M.2), M.2)
def point_B := (M.2 * N.1 / N.2, M.2)
def S_1 := 1/2 * ((M.2) + M.1) * (1 - M.2)
def S_2 := 1/2 * (M.2 * (1 + M.1))

theorem compare_areas (h : parabola (-1) = 1)
  (h' : parabola (-1/2) = 0) 
  (hM : parabola M.1 = M.2) 
  (hN : parabola N.1 = N.2)
  (h_line_M : line M.1 = M.2)
  (h_line_N : line N.1 = N.2) 
  (hk_pos : k > 0): 
  S_1 > 3 * S_2 := 
by sorry

end compare_areas_l129_129519


namespace smallest_b_factors_l129_129049

theorem smallest_b_factors (b p q : ℤ) (hb : b = p + q) (hpq : p * q = 2052) : b = 132 :=
sorry

end smallest_b_factors_l129_129049


namespace least_positive_integer_n_l129_129456

noncomputable theory -- Declare that the section does not have to be computable
open Real -- Open the real number space for using trigonometric functions and constants
open Finset -- Open finite sets for utility functions

theorem least_positive_integer_n : 
  ∃ (n : ℕ), 0 < n ∧ 
  (∑ k in (range 59) \ 30, 1 / (sin (Real.ofInt k + 30) * sin (Real.ofInt k + 31))) = 1 / (sin n) 
  ∧ n = 20 :=
by
  sorry

end least_positive_integer_n_l129_129456


namespace final_cost_is_35_l129_129217

-- Definitions based on conditions
def original_price : ℕ := 50
def discount_rate : ℚ := 0.30
def discount_amount : ℚ := original_price * discount_rate
def final_cost : ℚ := original_price - discount_amount

-- The theorem we need to prove
theorem final_cost_is_35 : final_cost = 35 := by
  sorry

end final_cost_is_35_l129_129217


namespace root_in_interval_l129_129345

def f (x : ℝ) : ℝ := 3^x - x - 3

theorem root_in_interval : ∃ x ∈ set.Ioo 1 2, f x = 0 := by
  sorry

end root_in_interval_l129_129345


namespace area_of_given_region_l129_129303

noncomputable def radius_squared : ℝ := 16 -- Completing the square gives us a radius squared value of 16.
def area_of_circle (r : ℝ) : ℝ := π * r ^ 2

theorem area_of_given_region : area_of_circle (real.sqrt radius_squared) = 16 * π := by
  sorry

end area_of_given_region_l129_129303


namespace exist_disk_intersect_l129_129177

-- Closed disks definition
structure ClosedDisk (center : ℝ × ℝ) (radius : ℝ) :=
(center_in_plane : center ∈ set.univ)
(radius_nonneg : radius ≥ 0)

-- Given conditions as assumptions
variables {n : ℕ} -- The number of disks
variables (disks : Fin n → ClosedDisk (center radius : ℝ × ℝ) (radius : ℝ)) 
variables (h1 : ∀ (x : ℝ × ℝ), (Finset.univ.filter (λ i, (EuclideanMetric.dist (disks i).center x ≤ (disks i).radius)).card ≤ 2003))

-- Proof statement
theorem exist_disk_intersect :
  ∃ k : Fin n, (Finset.univ.filter (λ i, (EuclideanMetric.dist (disks k).center (disks i).center) ≤ (disks k).radius + (disks i).radius)).card ≤ 7 * 2003 - 1 :=
sorry

end exist_disk_intersect_l129_129177


namespace radius_of_inner_circle_l129_129715

def right_triangle_legs (AC BC : ℝ) : Prop :=
  AC = 3 ∧ BC = 4

theorem radius_of_inner_circle (AC BC : ℝ) (h : right_triangle_legs AC BC) :
  ∃ r : ℝ, r = 2 :=
by
  sorry

end radius_of_inner_circle_l129_129715


namespace largest_N_l129_129492

-- Definition of the problem conditions
def problem_conditions (n : ℕ) (N : ℕ) (a : Fin (N + 1) → ℝ) : Prop :=
  (n ≥ 2) ∧
  (a 0 + a 1 = -(1 : ℝ) / n) ∧  
  (∀ k : ℕ, 1 ≤ k → k ≤ N - 1 → (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1))

-- The theorem stating that the largest integer N is n
theorem largest_N (n : ℕ) (N : ℕ) (a : Fin (N + 1) → ℝ) :
  problem_conditions n N a → N = n :=
sorry

end largest_N_l129_129492


namespace probability_two_queens_or_at_least_one_king_l129_129540

/-- Prove that the probability of either drawing two queens or drawing at least one king 
    when 2 cards are selected randomly from a standard deck of 52 cards is 2/13. -/
theorem probability_two_queens_or_at_least_one_king :
  (∃ (kq pk pq : ℚ), kq = 4 ∧
                     pk = 4 ∧
                     pq = 52 ∧
                     (∃ (p : ℚ), p = (kq*(kq-1))/(pq*(pq-1)) + (pk/pq)*(pq-pk)/(pq-1) + (kq*(kq-1))/(pq*(pq-1)) ∧
                            p = 2/13)) :=
by {
  sorry
}

end probability_two_queens_or_at_least_one_king_l129_129540


namespace initial_paint_amount_l129_129570

theorem initial_paint_amount (P : ℝ) (h1 : P > 0) (h2 : (1 / 4) * P + (1 / 3) * (3 / 4) * P = 180) : P = 360 := by
  sorry

end initial_paint_amount_l129_129570


namespace lines_concurrent_l129_129073

-- Define the triangle and its properties
structure Triangle :=
(A1 A2 A3 : Point)
(a1 a2 a3 : Line)
(M1 M2 M3 T1 T2 T3 S1 S2 S3 : Point)

-- Define the conditions of the problem
def conditions (Δ : Triangle) : Prop :=
  is_midpoint Δ.a1 Δ.M2 Δ.A3 ∧ is_midpoint Δ.a2 Δ.M3 Δ.A1 ∧ is_midpoint Δ.a3 Δ.M1 Δ.A2 ∧
  incircle_tangent Δ.A1 Δ.a1 Δ.T2 ∧ incircle_tangent Δ.A2 Δ.a2 Δ.T3 ∧ incircle_tangent Δ.A3 Δ.a3 Δ.T1 ∧
  is_reflection Δ.T1 Δ.A1 Δ.S1 ∧ is_reflection Δ.T2 Δ.A2 Δ.S2 ∧ is_reflection Δ.T3 Δ.A3 Δ.S3

-- Statement to be proved: The lines M1S1, M2S2, and M3S3 concur
theorem lines_concurrent (Δ : Triangle) (h : conditions Δ) : 
  concurrent (line_through Δ.M1 Δ.S1) (line_through Δ.M2 Δ.S2) (line_through Δ.M3 Δ.S3) :=
sorry

end lines_concurrent_l129_129073


namespace find_PB_l129_129929

-- Definitions of the given values and conditions
variables {A B C D P : Type}
variables [has_dist A] [has_dist B] [has_dist C] [has_dist D] [has_dist P]
variables (AB CD AD : A) [perpendicular CD AB] [perpendicular AB (diag DC)]
variables (AB_value : ℝ) (CD_value : ℝ) (AP_value : ℝ)
variables (P : point B)

-- Conditions given in the problem
axiom CD_eq : CD = 80
axiom AB_eq : AB = 35
axiom AP_eq : AP = 10

-- The proof goal
theorem find_PB (PB : ℝ) : PB = 112.5 :=
sorry

end find_PB_l129_129929


namespace area_of_rectangle_l129_129806

theorem area_of_rectangle (side_small_squares : ℝ) (side_smaller_square : ℝ) (side_larger_square : ℝ) 
  (h_small_squares : side_small_squares ^ 2 = 4) 
  (h_smaller_square : side_smaller_square ^ 2 = 1) 
  (h_larger_square : side_larger_square = 2 * side_smaller_square) :
  let horizontal_length := 2 * side_small_squares
  let vertical_length := side_small_squares + side_smaller_square
  let area := horizontal_length * vertical_length
  area = 12 
:= by 
  sorry

end area_of_rectangle_l129_129806


namespace math_proof_statement_l129_129522

open Real

noncomputable def proof_problem (x : ℝ) : Prop :=
  let a := (cos x, sin x)
  let b := (sqrt 2, sqrt 2)
  (a.1 * b.1 + a.2 * b.2 = 8 / 5) ∧ (π / 4 < x ∧ x < π / 2) ∧ 
  (cos (x - π / 4) = 4 / 5) ∧ (tan (x - π / 4) = 3 / 4) ∧ 
  (sin (2 * x) * (1 - tan x) / (1 + tan x) = -21 / 100)

theorem math_proof_statement (x : ℝ) : proof_problem x := 
by
  unfold proof_problem
  sorry

end math_proof_statement_l129_129522


namespace final_cost_of_dress_l129_129219

theorem final_cost_of_dress (original_price : ℝ) (discount_percentage : ℝ) 
  (h1 : original_price = 50) (h2 : discount_percentage = 0.30) : 
  let discount := discount_percentage * original_price in
  let final_cost := original_price - discount in
  final_cost = 35 := 
by
  sorry

end final_cost_of_dress_l129_129219


namespace isosceles_triangle_base_length_l129_129241

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l129_129241


namespace proof_problem_l129_129406

open EuclideanGeometry

noncomputable def problem_conditions := ∃ (A B C D E F G : Point), 
  convex_quadrilateral A B C D ∧
  (∃ BA CD, ray_intersects BA A B CD C D E) ∧
  (∃ DA CB, ray_intersects DA D A CB C B F) ∧
  (∃ AC BD, diagonals_intersect AC A C BD B D G) ∧
  (triangle_area D B F = triangle_area D B E) ∧
  (triangle_area A B D = 4) ∧
  (triangle_area C B D = 6)

theorem proof_problem : problem_conditions → 
  (∀ (A B C D E F G : Point), 
    (∃ BA CD, ray_intersects BA A B CD C D E) → 
    (∃ DA CB, ray_intersects DA D A CB C B F) → 
    (∃ AC BD, diagonals_intersect AC A C BD B D G) → 
    (triangle_area D B F = triangle_area D B E) →
    (triangle_area A B D = 4) → 
    (triangle_area C B D = 6) → 
    (parallel EF BD ∧ midpoint G B D ∧ 
      triangle_area E F G = 5/2)
  ) :=
by
  sorry

end proof_problem_l129_129406


namespace find_first_offset_l129_129817

/-- Define the problem -/
theorem find_first_offset 
  (area : ℝ) (diagonal : ℝ) (offset2 : ℝ) (offset1 : ℝ) :
  area = 75 →
  diagonal = 15 →
  offset2 = 4 →
  offset1 + offset2 = 10 →
  offset1 = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end find_first_offset_l129_129817


namespace path_difference_l129_129925

-- Definitions for the length of segment and sides of the rectangle
def l : ℝ := 4
def a : ℝ -- Width of the rectangle
def b : ℝ -- Height of the rectangle

-- Conditions
axiom h_l_lessthan_a : l < a
axiom h_l_lessthan_b : l < b

-- Perimeter of the rectangle
def perimeter : ℝ := 2 * (a + b)

-- Total length traveled by C
def path_C : ℝ := 4 * Real.pi + 2 * (a - l) + 2 * (b - l)
-- The difference between the perimeter and the path length of point C
def delta_s : ℝ := perimeter - path_C

-- The theorem to prove
theorem path_difference (a b : ℝ) (h1 : l < a) (h2 : l < b) : delta_s = 16 - 4 * Real.pi := by
  sorry

end path_difference_l129_129925


namespace cos_B_in_triangle_ABC_l129_129127

def a : ℝ := 15
def b : ℝ := 10
def A : ℝ := 60 * Real.pi / 180 -- converting degrees to radians

theorem cos_B_in_triangle_ABC (a b : ℝ) (A : ℝ) (ha : a = 15) (hb : b = 10) (hA : A = 60 * Real.pi / 180) :
  Real.cos B = 0 :=
sorry

end cos_B_in_triangle_ABC_l129_129127


namespace total_animal_legs_is_12_l129_129602

-- Define the number of legs per dog and chicken
def legs_per_dog : Nat := 4
def legs_per_chicken : Nat := 2

-- Define the number of dogs and chickens Mrs. Hilt saw
def number_of_dogs : Nat := 2
def number_of_chickens : Nat := 2

-- Calculate the total number of legs seen
def total_legs_seen : Nat :=
  (number_of_dogs * legs_per_dog) + (number_of_chickens * legs_per_chicken)

-- The theorem to be proven
theorem total_animal_legs_is_12 : total_legs_seen = 12 :=
by
  sorry

end total_animal_legs_is_12_l129_129602


namespace correct_option_is_C_l129_129750

theorem correct_option_is_C : 
  (∀ (x : ℝ), sqrt 16 ≠ ±4) ∧
  (∀ (y : ℝ), sqrt ((-3)^2) ≠ -3) ∧
  (∀ (z : ℝ), ±sqrt 81 = ±9) ∧
  (∀ (w : ℝ), sqrt (-4) ≠ 2) → 
  (true) :=
sorry

end correct_option_is_C_l129_129750


namespace dot_product_zero_l129_129889

theorem dot_product_zero (m : ℝ) :
  let a : ℝ × ℝ := (1, -m)
  let b : ℝ × ℝ := (m, 1)
  a.1 * b.1 + a.2 * b.2 = 0 :=
by
  unfold a b
  sorry

end dot_product_zero_l129_129889


namespace fraction_order_l129_129743

theorem fraction_order (a b c d e f : ℚ) (h1 : a = 21 ∧ b = 13) (h2 : c = 23 ∧ d = 17) (h3 : e = 25 ∧ f = 19) :
  (e / f < c / d ∧ c / d < a / b) :=
by {
  cases h1 with h1a h1b,
  cases h2 with h2a h2b,
  cases h3 with h3a h3b,
  sorry
}

end fraction_order_l129_129743


namespace original_bill_l129_129222

theorem original_bill (m : ℝ) (h1 : 10 * (m / 10) = m)
                      (h2 : 9 * ((m - 10) / 10 + 3) = m - 10) :
  m = 180 :=
  sorry

end original_bill_l129_129222


namespace sum_of_integral_values_l129_129823

theorem sum_of_integral_values (h1 : ∀ (x y c : ℤ), y = x^2 - 9 * x - c → y = 0 → ∃ r : ℚ, ∃ s : ℚ, r + s = 9 ∧ r * s = c)
    (h2 : ∀ (c : ℤ), (∃ k : ℤ, 81 + 4 * c = k^2 ∧ k^2 ≡ 1 [MOD 4]) ↔ ∃ k : ℤ, 81 + 4 * c = k^2 ∧ k % 2 = 1 ) :
    ∑ c in { c : ℤ | -20 ≤ c ∧ c ≤ 30 ∧ ∃ k : ℤ, 81 + 4 * c = k^2 ∧ k % 2 = 1 }, c = 32 :=
by {
  -- Proof omitted
  sorry
}

end sum_of_integral_values_l129_129823


namespace square_perimeter_l129_129783

theorem square_perimeter (area : ℝ) (h₁ : area = 625) : ∃ P : ℝ, P = 100 :=
by
  let s := real.sqrt area
  let P := 4 * s
  have h₂ : s = 25 := by sorry
  have h₃ : P = 100 := by sorry
  use P
  exact h₃

end square_perimeter_l129_129783


namespace valid_triangle_inequality_l129_129039

theorem valid_triangle_inequality (n : ℕ) (h : n = 6) :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c →
  n * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  intros a b c ha hb hc hineq
  have h₁ : n = 6 := h
  simplify_eq [h₁] at hineq
  have h₂ := nat.add_comm a b
  exact sorry

end valid_triangle_inequality_l129_129039


namespace solve_missing_number_l129_129023

open Nat

theorem solve_missing_number : 
  ∃ (x : ℕ), 1234562 - 12 * 3 * x = 1234490 ∧ x = 2 :=
by
  use 2
  split
  . simp
  . rfl

-- sorry for now due to simplification and manual proof step required.

end solve_missing_number_l129_129023


namespace simplify_expression_inequality_solution_l129_129762

-- Simplification part
theorem simplify_expression (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 2):
  (2 - (x - 1) / (x + 2)) / ((x^2 + 10 * x + 25) / (x^2 - 4)) = 
  (x - 2) / (x + 5) :=
sorry

-- Inequality system part
theorem inequality_solution (x : ℝ):
  (2 * x + 7 > 3) ∧ ((x + 1) / 3 > (x - 1) / 2) → -2 < x ∧ x < 5 :=
sorry

end simplify_expression_inequality_solution_l129_129762


namespace polygon_sides_l129_129119

theorem polygon_sides (n : ℕ) 
  (h1 : n ≥ 3) -- a polygon must have at least 3 sides to be convex
  (h2 : ∃ seq, 
        (∀ i : ℕ, 0 ≤ i < n → seq i = 100 + i * ((140 - 100) / (n - 1))) ∧ 
        seq 0 = 100 ∧ 
        seq (n - 1) = 140 ∧ 
        (∀ i : ℕ, 0 ≤ i < n → seq i < 180)) -- interior angles form an arithmetic sequence with 100° to 140°
  : n = 6 := 
sorry

end polygon_sides_l129_129119


namespace pencils_selected_l129_129131

theorem pencils_selected (n : ℕ) (h₁ : nat.choose 4 n * nat.choose 2 0 = nat.choose 6 n * 0.2) : n = 3 := 
by {
  sorry -- Proof omitted as per instructions
}

end pencils_selected_l129_129131


namespace points_scored_fourth_game_l129_129376

-- Define the conditions
def avg_score_3_games := 18
def avg_score_4_games := 17
def games_played_3 := 3
def games_played_4 := 4

-- Calculate total points after 3 games
def total_points_3_games := avg_score_3_games * games_played_3

-- Calculate total points after 4 games
def total_points_4_games := avg_score_4_games * games_played_4

-- Define a theorem to prove the points scored in the fourth game
theorem points_scored_fourth_game :
  total_points_4_games - total_points_3_games = 14 :=
by
  sorry

end points_scored_fourth_game_l129_129376


namespace chord_AF_approx_pi_div_2_l129_129836

noncomputable def circle_af_length (O A B C D E F : Point)
  (h1 : is_on_circle A O)
  (h2 : arc_with_radius_intersects_circle O A = B)
  (h3 : arc_with_radius_intersects_circle O B = C)
  (h4 : arc_with_radius_intersects_circle O C = D)
  (h5 : arc_with_radius_AC_intersects O A D C = E)
  (h6 : arc_with_radius_DB_intersects O B D = F) : Prop :=
  chord_length A F ≈ π / 2

theorem chord_AF_approx_pi_div_2
  (O A B C D E F : Point)
  (h1 : is_on_circle A O)
  (h2 : arc_with_radius_intersects_circle O A = B)
  (h3 : arc_with_radius_intersects_circle O B = C)
  (h4 : arc_with_radius_intersects_circle O C = D)
  (h5 : arc_with_radius_AC_intersects O A D C = E)
  (h6 : arc_with_radius_DB_intersects O B D = F) :
  circle_af_length O A B C D E F h1 h2 h3 h4 h5 h6 := sorry

end chord_AF_approx_pi_div_2_l129_129836


namespace complex_div_l129_129405

theorem complex_div (i : ℂ) (hi : i^2 = -1) : (1 + i) / i = 1 - i := by
  sorry

end complex_div_l129_129405


namespace sum_ABC_eq_5_l129_129410

noncomputable def A_log_eq_C (A B C : ℕ) : Prop :=
  A * (Real.log 5 / Real.log 100) + B * (Real.log 2 / Real.log 100) = C

noncomputable def coprime_A_B_C (A B C : ℕ) : Prop :=
  Nat.coprime A B ∧ Nat.coprime B C ∧ Nat.coprime A C

theorem sum_ABC_eq_5 (A B C : ℕ) (h1 : A_log_eq_C A B C) (h2 : coprime_A_B_C A B C) : 
  A + B + C = 5 :=
sorry

end sum_ABC_eq_5_l129_129410


namespace symmetric_points_on_parabola_l129_129859

theorem symmetric_points_on_parabola
  (x1 x2 : ℝ)
  (m : ℝ)
  (h1 : 2 * x1 * x1 = 2 * x2 * x2)
  (h2 : 2 * x1 * x1 = 2 * x2 * x2 + m)
  (h3 : x1 * x2 = -1 / 2)
  (h4 : x1 + x2 = -1 / 2)
  : m = 3 / 2 :=
sorry

end symmetric_points_on_parabola_l129_129859


namespace problem_104_l129_129165

-- Define the distances a, b, and c
noncomputable def a : ℝ := 13
noncomputable def b : ℝ := 14
noncomputable def c : ℝ := 15

-- Define GO and GI ratio
noncomputable def GO_to_GI_ratio (GO GI : ℝ) := GO / GI = 1 / 4

-- The main theorem to prove 100m + n = 104
theorem problem_104 : 
  ∀ (m n : ℕ), (1 = m) → (4 = n) → 100 * m + n = 104 :=
by
  intros m n hm hn
  rw [hm, hn]
  norm_num
  sorry

end problem_104_l129_129165


namespace special_fractions_sum_is_14_l129_129799

theorem special_fractions_sum_is_14 :
  {n : ℕ | ∃ (a1 a2 b1 b2 : ℕ), a1 + b1 = 20 ∧ a2 + b2 = 20 ∧ n = (a1 * b2 + a2 * b1) / (b1 * b2))}.to_finset.card = 14 :=
sorry

end special_fractions_sum_is_14_l129_129799


namespace solution_set_fraction_inequality_l129_129270

theorem solution_set_fraction_inequality : 
  { x : ℝ | 0 < x ∧ x < 1/3 } = { x : ℝ | 1/x > 3 } :=
by
  sorry

end solution_set_fraction_inequality_l129_129270


namespace sum_of_roots_zero_l129_129965

noncomputable def Q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_roots_zero (a b c : ℝ) 
  (h : ∀ x : ℝ, Q a b c (x^3 - x) ≥ Q a b c (x^2 - 1)) : 
  (b = 0) → (sum_of_roots (Q a b c) = 0) :=
sorry

end sum_of_roots_zero_l129_129965


namespace pencil_cost_is_correct_l129_129705

-- Defining the cost of a pen as x and the cost of a pencil as y in cents
def cost_of_pen_and_pencil (x y : ℕ) : Prop :=
  3 * x + 5 * y = 345 ∧ 4 * x + 2 * y = 280

-- Stating the theorem that proves y = 39
theorem pencil_cost_is_correct (x y : ℕ) (h : cost_of_pen_and_pencil x y) : y = 39 :=
by
  sorry

end pencil_cost_is_correct_l129_129705


namespace number_of_possible_sets_C_l129_129081

theorem number_of_possible_sets_C (A B : Finset ℕ) (hA : A.card = 12) (hB : B.card = 12)
  (hAB : (A ∩ B).card = 4) :
  (∃ C : Finset ℕ, C.card = 3 ∧ C ⊆ A ∪ B ∧ C ∩ A ≠ ∅ ∧ C ∩ B ≠ ∅) →
  ∃ n : ℕ, n = 1028 :=
begin
  sorry
end

end number_of_possible_sets_C_l129_129081


namespace expected_value_decagonal_die_l129_129310

-- Given conditions
def decagonal_die_faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def probability (n : ℕ) : ℚ := 1 / 10

-- The mathematical proof problem (statement only, no proof required)
theorem expected_value_decagonal_die : 
  (List.sum decagonal_die_faces : ℚ) / List.length decagonal_die_faces = 5.5 := by
  sorry

end expected_value_decagonal_die_l129_129310


namespace sum_and_product_of_roots_l129_129650

theorem sum_and_product_of_roots :
  let p := (2 : ℝ) * x ^ 2 - (3 : ℝ) * x - (5 : ℝ) = 0
  let q := x ^ 2 - (6 : ℝ) * x + (2 : ℝ) = 0
  (∑ root in Roots p, root) + (∑ root in Roots q, root) = (7.5 : ℝ) ∧
  (∏ root in Roots p, root) * (∏ root in Roots q, root) = (-5 : ℝ) :=
by
  -- Proof will go here
  sorry

end sum_and_product_of_roots_l129_129650


namespace simplify_expression_l129_129055

theorem simplify_expression : ( (7: ℝ) ^ (1 / 4) / (7: ℝ) ^ (1 / 6) ) ^ (-3) = (7: ℝ) ^ (-1 / 4) :=
by
  sorry

end simplify_expression_l129_129055


namespace num_integers_with_g_geq_3_l129_129832

-- Define the function g(n) which calculates the number of 1s in the binary representation of n
def g (n : ℕ) : ℕ := (Nat.digitCount Nat.binary 1 n)

theorem num_integers_with_g_geq_3 : 
  { n : ℕ | 1 ≤ n ∧ n ≤ 2007 ∧ g n ≥ 3 }.toFinset.card = 1941 := by
  sorry

end num_integers_with_g_geq_3_l129_129832


namespace modular_inverse_of_35_mod_36_l129_129820

theorem modular_inverse_of_35_mod_36 : 
  ∃ a : ℤ, (35 * a) % 36 = 1 % 36 ∧ a = 35 := 
by 
  sorry

end modular_inverse_of_35_mod_36_l129_129820


namespace people_did_not_show_up_l129_129769

variable (total_invited : ℕ) (total_seated : ℕ)

theorem people_did_not_show_up :
  total_invited = 24 → total_seated = 2 * 7 → total_invited - total_seated = 10 :=
by
  intros h1 h2
  rw [h1, h2]
  exact (24 - 14)

end people_did_not_show_up_l129_129769


namespace triangle_inequality_for_n6_l129_129034

variables {a b c : ℝ} {n : ℕ}
open Real

-- Define the main statement as a theorem
theorem triangle_inequality_for_n6 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_for_n6_l129_129034


namespace identifyIncorrectGuess_l129_129684

-- Define the colors of the bears
inductive BearColor
| white
| brown
| black

-- Conditions as defined in the problem statement
def isValidBearRow (bears : Fin 1000 → BearColor) : Prop :=
  ∀ (i : Fin 998), 
    (bears i = BearColor.white ∨ bears i = BearColor.brown ∨ bears i = BearColor.black) ∧
    (bears ⟨i + 1, by linarith⟩ = BearColor.white ∨ bears ⟨i + 1, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 1, by linarith⟩ = BearColor.black) ∧
    (bears ⟨i + 2, by linarith⟩ = BearColor.white ∨ bears ⟨i + 2, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 2, by linarith⟩ = BearColor.black)

-- Iskander's guesses
def iskanderGuesses (bears : Fin 1000 → BearColor) : Prop :=
  bears 1 = BearColor.white ∧
  bears 19 = BearColor.brown ∧
  bears 399 = BearColor.black ∧
  bears 599 = BearColor.brown ∧
  bears 799 = BearColor.white

-- Exactly one guess is incorrect
def oneIncorrectGuess (bears : Fin 1000 → BearColor) : Prop :=
  ∃ (idx : Fin 5), 
    ¬iskanderGuesses bears ∧
    ∀ (j : Fin 5), (j ≠ idx → (bearGuessesIdx j bears = true))

-- The proof problem
theorem identifyIncorrectGuess (bears : Fin 1000 → BearColor) :
  isValidBearRow bears → iskanderGuesses bears → oneIncorrectGuess bears := sorry

end identifyIncorrectGuess_l129_129684


namespace Teresa_age_when_Michiko_born_l129_129625

theorem Teresa_age_when_Michiko_born 
  (Teresa_age : ℕ) (Morio_age : ℕ) (Michiko_born_age : ℕ) (Kenji_diff : ℕ)
  (Emiko_diff : ℕ) (Hideki_same_as_Kenji : Prop) (Ryuji_age_same_as_Morio : Prop)
  (h1 : Teresa_age = 59) 
  (h2 : Morio_age = 71) 
  (h3 : Morio_age = Michiko_born_age + 33)
  (h4 : Kenji_diff = 4)
  (h5 : Emiko_diff = 10)
  (h6 : Hideki_same_as_Kenji = True)
  (h7 : Ryuji_age_same_as_Morio = True) : 
  ∃ Michiko_age Hideki_age Michiko_Hideki_diff Teresa_birth_age,
    Michiko_age = 33 ∧ 
    Hideki_age = 29 ∧ 
    Michiko_Hideki_diff = 4 ∧ 
    Teresa_birth_age = 26 :=
sorry

end Teresa_age_when_Michiko_born_l129_129625


namespace solve_absolute_inequality_l129_129622

theorem solve_absolute_inequality (x : ℝ) : |x - 1| - |x - 2| > 1 / 2 ↔ x > 7 / 4 :=
by sorry

end solve_absolute_inequality_l129_129622


namespace _l129_129006

structure Parallelepiped (A1 B1 C1 D1 A B C D : Type) :=
  (a b c : ℝ)
  (M N P Q : Type)
  -- edge lengths
  (edge_AA1 : A1 = A ∧ ∥A - A1∥ = a)
  (edge_B1A1 : B1 = B ∧ ∥B1 - A1∥ = b)
  (edge_A1D1 : D1 = A1 ∧ ∥A1 - D1∥ = c)
  -- midpoints
  (mid_M : M = (A1 + B1) / 2)
  (mid_N : N = (A1 + D1) / 2)
  (mid_P : P = (B + C) / 2)
  (mid_Q : Q = (C + D) / 2)

def centroid_distance (p : Parallelepiped A1 B1 C1 D1 A B C D) : ℝ :=
  let M := p.M
  let N := p.N
  let P := p.P
  let Q := p.Q
  let centroid_A_M_N := (A + M + N) / 3
  let centroid_C1_P_Q := (C1 + P + Q) / 3
  ∥centroid_A_M_N - centroid_C1_P_Q∥

noncomputable def distance_between_centroids (p : Parallelepiped A1 B1 C1 D1 A B C D) : ℝ :=
  let dist := centroid_distance p
  dist ∥= ∥(1 / 3) * ∥sqrt (a^2 + 4 * b^2 + 4 * c^2)

lemma distance_between_centroids_theorem : 
  forall (p : Parallelepiped A1 B1 C1 D1 A B C D), distance_between_centroids p = sorry

end _l129_129006


namespace tan_alpha_perpendicular_alpha_parallel_l129_129524

-- Given conditions
variables (α : ℝ) (hα : α ∈ Ioo 0 π)

-- Definitions of vectors a and b
noncomputable def a := (sin (α + π / 6), 3 : ℝ × ℝ)
noncomputable def b := (1 : ℝ, 4 * cos α)

-- a ⊥ b implies tan α = -25√3 / 3
theorem tan_alpha_perpendicular (h_perp : (sin (α + π / 6)) + 12 * cos α = 0) : 
  tan α = -25 * real.sqrt 3 / 3 :=
sorry

-- a ∥ b implies α = π / 6
theorem alpha_parallel (h_parallel : 4 * cos α * sin (α + π / 6) = 3) : 
  α = π / 6 :=
sorry

end tan_alpha_perpendicular_alpha_parallel_l129_129524


namespace directrix_of_parabola_l129_129449

noncomputable def parabola_directrix (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / 16

theorem directrix_of_parabola :
  let d := parabola_directrix y in d = -(1 / 2) := sorry

end directrix_of_parabola_l129_129449


namespace exponents_to_99_l129_129608

theorem exponents_to_99 :
  (1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99) :=
sorry

end exponents_to_99_l129_129608


namespace order_of_a_b_c_l129_129065

noncomputable def a := Real.sqrt 3 - Real.sqrt 2
noncomputable def b := Real.sqrt 6 - Real.sqrt 5
noncomputable def c := Real.sqrt 7 - Real.sqrt 6

theorem order_of_a_b_c : a > b ∧ b > c :=
by
  sorry

end order_of_a_b_c_l129_129065


namespace discs_cover_points_with_sum_diameters_less_n_l129_129201


-- Define the plane as a Type
variables {Point : Type}

-- Define distance between points
variable [metric_space Point]

-- Define a set of n arbitrary points
variables (n : ℕ) (points : finset Point)

-- Distance function between points
def dist (a b : Point) := dist a b

-- Define disc and its radius
structure Disc (P : Point) :=
(radius : real)

-- Define a set of discs covering the points in the plane
def cover (discs : finset (Disc Point)) (points : finset Point) : Prop :=
  ∀ p ∈ points,
    ∃ d ∈ discs, dist (p : Point) (d : Disc Point).center < (d : Disc Point).radius

theorem discs_cover_points_with_sum_diameters_less_n
  (points : finset Point)
  (h₁ : points.card = n) :
  ∃ (discs : finset (Disc Point)),
    cover discs points ∧
    (discs.sum (λ d, d.radius * 2)) < n ∧
    ∀ (d₁ d₂ ∈ discs), d₁ ≠ d₂ → 1 < dist d₁.center d₂.center - (d₁.radius + d₂.radius) := 
sorry

end discs_cover_points_with_sum_diameters_less_n_l129_129201


namespace y_intercept_of_line_l129_129813

theorem y_intercept_of_line : ∃ y, 2 * 0 + 7 * y = 35 ∧ y = 5 := 
by
  use 5
  split
  · simp
  · simp

end y_intercept_of_line_l129_129813


namespace true_propositions_among_conditions_l129_129174

variable {α : Type*} [topological_space α]

def even_function (f : α → α) : Prop :=
  ∀ x, f x = f (-x)

def symmetric_about_line (f : α → α) (a : α) : Prop :=
  ∀ x, f (a - x) = f (a + x)

def periodic_function (f : α → α) (T : α) : Prop :=
  ∀ x, f x = f (x + T)

theorem true_propositions_among_conditions (f : ℝ → ℝ) :
  even_function f ∧ symmetric_about_line f 1 → periodic_function f 2 ∧
  symmetric_about_line f 1 ∧ periodic_function f 2 → even_function f ∧
  even_function f ∧ periodic_function f 2 → symmetric_about_line f 1 :=
by
  intro h
  sorry

#eval true_propositions_among_conditions

end true_propositions_among_conditions_l129_129174


namespace xyz_sum_56_l129_129115

theorem xyz_sum_56 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + z = 55) (h2 : y * z + x = 55) (h3 : z * x + y = 55)
  (even_cond : x % 2 = 0 ∨ y % 2 = 0 ∨ z % 2 = 0) :
  x + y + z = 56 :=
sorry

end xyz_sum_56_l129_129115


namespace line_intersection_and_conditions_l129_129819

theorem line_intersection_and_conditions :
  -- Define the lines
  let l1 := λ x y : ℝ, 3 * x + 4 * y - 5 = 0 in
  let l2 := λ x y : ℝ, 2 * x - 3 * y + 8 = 0 in
  let l_parallel := λ x y a : ℝ, 2 * x + y + a = 0 in
  let l_perpendicular := λ x y a : ℝ, x - 2 * y + a = 0 in

  -- Intersection point of l1 and l2
  let M := (-1, 2) in

  -- Parallel line through M
  (∃ a : ℝ, l_parallel (-1) 2 a = 0) ∧

  -- Perpendicular line through M
  (∃ a : ℝ, l_perpendicular (-1) 2 a = 0) ∧

  -- Prove the correct equation forms
  M = (-1, 2) ∧
  (∀ a, 2 * -1 + 2 + a = 0 → a = 0) ∧
  (∀ a, -1 - 2 * 2 + a = 0 → a = 5) :=
sorry

end line_intersection_and_conditions_l129_129819


namespace monotonically_decreasing_when_a_half_l129_129510

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - a * x)

theorem monotonically_decreasing_when_a_half :
  ∀ x : ℝ, 0 < x → (f x (1 / 2)) ≤ 0 :=
by
  sorry

end monotonically_decreasing_when_a_half_l129_129510


namespace james_total_socks_l129_129157

-- Definitions based on conditions
def red_pairs : ℕ := 20
def black_pairs : ℕ := red_pairs / 2
def white_pairs : ℕ := 2 * (red_pairs + black_pairs)
def green_pairs : ℕ := (red_pairs + black_pairs + white_pairs) + 5

-- Total number of pairs
def total_pairs := red_pairs + black_pairs + white_pairs + green_pairs

-- Total number of socks
def total_socks := total_pairs * 2

-- The main theorem to prove the total number of socks
theorem james_total_socks : total_socks = 370 :=
  by
  -- proof is skipped
  sorry

end james_total_socks_l129_129157


namespace find_a_l129_129857

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, (1 - real.log10 a * real.log10 a) * x * x + (3 + real.log10 a) * x + 2 = 0 ∧ 
  ∀ x' : ℝ, (1 - real.log10 a * real.log10 a) * x' * x' + (3 + real.log10 a) * x' + 2 = 0 → x = x') → 
  a = 1 / (10 ^ (1 / 3 : ℝ)) :=
by
  intro h
  sorry

end find_a_l129_129857


namespace find_principal_l129_129333

-- Define the conditions
def principal_sum (R : ℝ) (P : ℝ) : Prop :=
  let T : ℝ := 10 in
  let SI1 := (P * R * T) / 100 in
  let SI2 := (P * (R + 5) * T) / 100 in
  SI2 = SI1 + 300

-- The theorem statement
theorem find_principal (R : ℝ) (P : ℝ) (h : principal_sum R P) : P = 600 :=
by
  sorry

end find_principal_l129_129333


namespace shoveling_time_is_13_hours_l129_129607

-- Define the conditions as per a)
def initial_rate : ℕ := 25
def rate_decrease : ℕ := 2
def driveway_volume : ℕ := 5 * 12 * 3.5

-- Define the question and its correct answer:
def time_to_clear_driveway (initial_rate : ℕ) (rate_decrease : ℕ) (driveway_volume : ℕ) : ℕ :=
  -- Sorry is used to skip the actual proof details
  sorry

-- State the theorem:
theorem shoveling_time_is_13_hours :
  time_to_clear_driveway initial_rate rate_decrease driveway_volume = 13 :=
  sorry

end shoveling_time_is_13_hours_l129_129607


namespace scooter_safety_gear_price_increase_l129_129944

theorem scooter_safety_gear_price_increase :
  let last_year_scooter_price := 200
  let last_year_gear_price := 50
  let scooter_increase_rate := 0.08
  let gear_increase_rate := 0.15
  let total_last_year_price := last_year_scooter_price + last_year_gear_price
  let this_year_scooter_price := last_year_scooter_price * (1 + scooter_increase_rate)
  let this_year_gear_price := last_year_gear_price * (1 + gear_increase_rate)
  let total_this_year_price := this_year_scooter_price + this_year_gear_price
  let total_increase := total_this_year_price - total_last_year_price
  let percent_increase := (total_increase / total_last_year_price) * 100
  percent_increase = 9 :=
by
  -- sorry is added here to skip the proof steps
  sorry

end scooter_safety_gear_price_increase_l129_129944


namespace locus_of_P_range_of_slope_l129_129559

-- Definitions for points A, B, and C in the plane
def A : ℝ × ℝ := (0, 4 / 3)
def B : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (1, 0)

-- Distance function from a point P to a line defined by points
def dist_to_line (P : ℝ × ℝ) (l1 l2 : ℝ × ℝ) : ℝ :=
  abs ((l2.2 - l1.2) * P.1 - (l2.1 - l1.1) * P.2 + l2.1 * l1.2 - l2.2 * l1.1) / 
      (sqrt ((l2.2 - l1.2) ^ 2 + (l2.1 - l1.1) ^ 2))

-- Equation of the locus
theorem locus_of_P :
  ∀ (P : ℝ × ℝ), 
    let d1 := dist_to_line P A B,
        d2 := dist_to_line P A C,
        d3 := abs P.2 in
    (d1 * d2 = d3 ^ 2) ↔ (2 * P.1^2 + 2 * P.2^2 + 3 * P.2 - 2 = 0 ∨ 8 * P.1^2 - 17 * P.2^2 + 12 * P.2 - 8 = 0) :=
by sorry

-- Range of values for the slope k
theorem range_of_slope (D : ℝ × ℝ := (0, 1 / 2)) :
  ∀ (k : ℝ) (l : ℝ × ℝ → Prop := λ P, P.2 = k * P.1 + 1 / 2), 
    (∀ P, (l P → (2 * P.1^2 + 2 * (k * P.1 + 1 / 2)^2 + 3 * (k * P.1 + 1 / 2) - 2 = 0 ∨ 
           8 * P.1^2 - 17 * (k * P.1 + 1 / 2)^2 + 12 * (k * P.1 + 1 / 2) - 8 = 0))) ↔ 
    (k = 0 ∨ k = 1 / 2 ∨ k = -1 / 2 ∨ k = 2 * sqrt 34 / 17 ∨ k = -2 * sqrt 34 / 17 ∨ k = sqrt 2 / 2 ∨ k = -sqrt 2 / 2) :=
by sorry

end locus_of_P_range_of_slope_l129_129559


namespace checkerboard_fraction_checkerboard_sum_mn_l129_129348

open Nat

-- Condition definitions
def checkerboard_size : ℕ := 10
def horizontal_lines : ℕ := checkerboard_size + 1
def vertical_lines : ℕ := checkerboard_size + 1

-- Number of rectangles calculation
def num_rectangles : ℕ :=
  (choose horizontal_lines 2) * (choose vertical_lines 2)

-- Number of squares calculation
def num_squares : ℕ :=
  ∑ i in Icc 1 checkerboard_size, i * i

-- Simplified fraction of squares to rectangles
def fraction_squares_to_rectangles : ℚ :=
  (num_squares : ℚ) / num_rectangles

theorem checkerboard_fraction :
  fraction_squares_to_rectangles = 7 / 55 :=
by {
  -- The proof would go here, but it's skipped with sorry
  sorry
}

theorem checkerboard_sum_mn :
  7 + 55 = 62 :=
by {
  -- This is a simple arithmetic proof
  exact rfl
}

end checkerboard_fraction_checkerboard_sum_mn_l129_129348


namespace min_expression_value_l129_129584

def elements : Set ℚ := {-7, -5, -3, -2, 2, 4, 6, 13}

noncomputable def min_possible_value : ℚ :=
  let pairs := { s | s ⊆ elements ∧ s.card = 4 }
  let sums := pairs.image (λ s, s.sum)
  let min_val := sums.fold
    (λ acc x, min acc (2*(x - 4)^2 + 32))
    infinity
  min_val

theorem min_expression_value :
  min_possible_value = 34 :=
sorry -- proof needed

end min_expression_value_l129_129584


namespace company_fund_amount_l129_129634

theorem company_fund_amount
  (n : ℕ) 
  (h1 : 50 * n + 120 = 60 * n - 10) :
  60 * n - 10 = 770 :=
begin
  -- use the given equation to show n = 13
  have hn : n = 13,
  { linarith, },
  -- substitute n = 13 into the initial amount calculation
  rw hn,
  linarith,
end

end company_fund_amount_l129_129634


namespace acute_angle_implies_x_range_perpendicular_and_magnitude_l129_129891

-- Define vectors a and b
def a : Vector ℝ := (2, 4)
def b (x : ℝ) : Vector ℝ := (x, 1)

-- Question (1): If the angle between a and b is acute, find the range of x
theorem acute_angle_implies_x_range (x : ℝ) (h : a.1 * b(x).1 + a.2 * b(x).2 > 0) : 
  x > -2 ∧ x ≠ 1 / 2 :=
  sorry

-- Question (2): If 2a - b is perpendicular to a, find the magnitude of a + b
theorem perpendicular_and_magnitude (x : ℝ) (h : (4 - x) * 2 + 7 * 4 = 0) :
  |(a.1 + b(18).1, a.2 + b(18).2)| = 5 * real.sqrt 17 :=
  sorry

end acute_angle_implies_x_range_perpendicular_and_magnitude_l129_129891


namespace find_angle_phi_l129_129822

theorem find_angle_phi : 
  ∃ φ : ℝ, (φ > 0 ∧ φ < 360 ∧ cos φ = sin (45 : ℝ) + cos (60 : ℝ) - sin (30 : ℝ) - cos (15 : ℝ)) ↔ φ = 45 :=
by
  sorry

end find_angle_phi_l129_129822


namespace incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129660

def is_color (col : String) (pos : Nat) : Prop := sorry
def valid_guesses : Prop :=
  (is_color "white" 2) ∧
  (is_color "brown" 20) ∧
  (is_color "black" 400) ∧
  (is_color "brown" 600) ∧
  (is_color "white" 800)

theorem incorrect_guess_at_20 :
  (∃ x, (x ∈ [2, 20, 400, 600, 800]) ∧ ¬ is_color_correct x) :=
begin
  sorry -- proof is not required
end

/-- Main theorem to identify the incorrect guess position. -/
theorem Iskander_incorrect_guess_20 :
  valid_guesses →
  (∃! x ∈ [2, 20, 400, 600, 800], ¬ is_color_correct x) →
  ¬ is_color "brown" 20 :=
begin
  admit -- proof is not required
end

end incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129660


namespace volume_of_intersection_l129_129738

def condition1 (x y z : ℝ) : Prop := |x| + |y| + |z| ≤ 1
def condition2 (x y z : ℝ) : Prop := |x| + |y| + |z - 2| ≤ 1
def in_intersection (x y z : ℝ) : Prop := condition1 x y z ∧ condition2 x y z

theorem volume_of_intersection : 
  (∫ x y z in { p : ℝ × ℝ × ℝ | in_intersection p.1 p.2 p.3 }, 1) = 1/12 := 
by
  sorry

end volume_of_intersection_l129_129738


namespace incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129659

def is_color (col : String) (pos : Nat) : Prop := sorry
def valid_guesses : Prop :=
  (is_color "white" 2) ∧
  (is_color "brown" 20) ∧
  (is_color "black" 400) ∧
  (is_color "brown" 600) ∧
  (is_color "white" 800)

theorem incorrect_guess_at_20 :
  (∃ x, (x ∈ [2, 20, 400, 600, 800]) ∧ ¬ is_color_correct x) :=
begin
  sorry -- proof is not required
end

/-- Main theorem to identify the incorrect guess position. -/
theorem Iskander_incorrect_guess_20 :
  valid_guesses →
  (∃! x ∈ [2, 20, 400, 600, 800], ¬ is_color_correct x) →
  ¬ is_color "brown" 20 :=
begin
  admit -- proof is not required
end

end incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129659


namespace bisecting_vector_exists_l129_129181

variables (a b : ℝ → ℝ → ℝ → ℝ)
def a := (4, 5, 1)
def b := (1, -2, 2)

theorem bisecting_vector_exists : 
  ∃ v : ℝ × ℝ × ℝ, 
  let k := -1 in 
  let sqrt_42 := Real.sqrt 42 in 
  v = (6 / (sqrt_42 + 4), 1 / (sqrt_42 + 4), 5 / (sqrt_42 + 4)) ∧
  (Real.norm (v.1, v.2, v.3) = 1) :=
sorry

end bisecting_vector_exists_l129_129181


namespace problem_solution_l129_129834

def f(x : ℝ) : ℝ := (x - 1) / (x + 1)

def fn : ℕ → (ℝ → ℝ)
| 0       := id
| (n + 1) := f ∘ fn n

def M : set ℝ := { x | fn 2036 x = x }

theorem problem_solution : M = set.univ :=
  sorry

end problem_solution_l129_129834


namespace find_least_positive_integer_n_l129_129458

theorem find_least_positive_integer_n :
  (∃ n : ℕ, ∀ (i : ℕ), (30 ≤ i ∧ i ≤ 88) → (∑ j in finset.range (59), (1 / (real.sin ((30 + 2 * j) * real.pi / 180) * real.sin ((31 + 2 * j) * real.pi / 180)))) = 1 / (real.sin (n * real.pi / 180)) ∧ 1 ≤ n)
  ∧ (∀ m : ℕ, (∑ j in finset.range (59), (1 / (real.sin ((30 + 2 * j) * real.pi / 180) * real.sin ((31 + 2 * j) * real.pi / 180)))) ≠ 1 / (real.sin (m * real.pi / 180)) ∨ 1 > m) :=
by
  sorry

end find_least_positive_integer_n_l129_129458


namespace hyperbolic_identity_l129_129556

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem hyperbolic_identity (x : ℝ) : (ch x) ^ 2 - (sh x) ^ 2 = 1 := 
sorry

end hyperbolic_identity_l129_129556


namespace total_carrots_grown_l129_129209

theorem total_carrots_grown :
  let Sandy := 6.5
  let Sam := 3.25
  let Sophie := 2.75 * Sam
  let Sara := (Sandy + Sam + Sophie) - 7.5
  Sandy + Sam + Sophie + Sara = 29.875 :=
by
  sorry

end total_carrots_grown_l129_129209


namespace tetrahedron_midpoints_length_l129_129425

theorem tetrahedron_midpoints_length :
  let a := 5
  let b := Real.sqrt 41
  let c := Real.sqrt 34
  -- Question: What are the lengths of the segments connecting the midpoints of the opposite edges?
  -- Conditions: Each vertex of the tetrahedron has edges of lengths 5, sqrt(41), and sqrt(34) meeting.
  holds ∃ (lengths : List ℝ), lengths = [3, 4, 5]
sorry

end tetrahedron_midpoints_length_l129_129425


namespace find_x_l129_129476

theorem find_x (x : ℝ) (h : x > 0) (area : 1 / 2 * (2 * x) * x = 72) : x = 6 * Real.sqrt 2 :=
by
  sorry

end find_x_l129_129476


namespace average_words_per_page_l129_129019

theorem average_words_per_page
  (sheets_to_pages : ℕ := 16)
  (total_sheets : ℕ := 12)
  (total_word_count : ℕ := 240000) :
  (total_word_count / (total_sheets * sheets_to_pages)) = 1250 :=
by
  sorry

end average_words_per_page_l129_129019


namespace simple_interest_principal_l129_129337

-- Definitions for the conditions
def CI (P r t : ℝ) : ℝ := P * (1 + r / 100) ^ t - P 
def SI (P r t : ℝ) : ℝ := P * r * t / 100 

-- The Lean statement for the proof problem
theorem simple_interest_principal :
  ∃ (P : ℝ), 
    P = 3225 ∧ 
    let CI_rs8000 := (CI 8000 15 2) in
        let SI_for_P := (SI P 8 5) in 
           SI_for_P = CI_rs8000 / 2 :=
begin
  sorry
end

end simple_interest_principal_l129_129337


namespace vector_magnitude_subtraction_l129_129884

noncomputable def vector_a : ℝ × ℝ := (-2, 2)
noncomputable def norm_vector_b : ℝ := 1
noncomputable def theta : ℝ := Real.pi / 4

theorem vector_magnitude_subtraction (b : ℝ × ℝ) (hb : ‖b‖ = norm_vector_b)
    (hab : Real.inner (vector_a.1, vector_a.2) b = (‖vector_a‖ * ‖b‖ * Real.cos theta)) :
    ‖(vector_a.1 - 2 * b.1, vector_a.2 - 2 * b.2)‖ = 2 := by
  sorry

end vector_magnitude_subtraction_l129_129884


namespace matrix_power_problem_l129_129950

def B : Matrix (Fin 2) (Fin 2) ℤ := 
  ![![4, 1], ![0, 2]]

theorem matrix_power_problem : B^15 - 3 * B^14 = ![![4, 3], ![0, -2]] :=
  by sorry

end matrix_power_problem_l129_129950


namespace area_of_circle_eq_sixteen_pi_l129_129306

theorem area_of_circle_eq_sixteen_pi :
  ∃ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) ↔ (π * 4^2 = 16 * π) :=
by
  sorry

end area_of_circle_eq_sixteen_pi_l129_129306


namespace certain_number_is_correct_l129_129474

def m : ℕ := 72483

theorem certain_number_is_correct : 9999 * m = 724827405 := by
  sorry

end certain_number_is_correct_l129_129474


namespace sum_of_intervals_length_l129_129999

noncomputable def sum_to_n (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem sum_of_intervals_length :
  let S := { x : ℝ | ∑ k in (Finset.range 63).image (λ k, k + 1), k / (x - k) ≥ 1 } in
  ∑ x in S, (function.find (λ k, k < x) (Finset.range 63) - function.find (λ k, k ≥ x) (Finset.range 63)) == 2016 := 
sorry

end sum_of_intervals_length_l129_129999


namespace sum_independent_of_choice_of_P_l129_129178

variables {A B C P Q R : Type} [metric_space A] [metric_space B] [metric_space C]

def isosceles_right_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :
  Prop := ∃ T : triangle A B C, T.is_isosceles ∧ T.right_angle = 90

def circle_centered_at (P : Type) [metric_space P] (C : Type) [metric_space C] :
  Prop := ∃ circle : circle P C, circle.center = P ∧ circle.passes_through C

theorem sum_independent_of_choice_of_P :
  ∀ (A B C P Q R : Type) [metric_space A] [metric_space B] [metric_space C],
    isosceles_right_triangle A B C → 
    (P ∈ hypotenuse A B) →
    circle_centered_at P C →
    let (Q, R) := (intersection (circle P) (line CA), intersection (circle P) (line CB)) in
    (CQ + CR) = constant := 
begin
  sorry
end

end sum_independent_of_choice_of_P_l129_129178


namespace sum_of_solutions_equation_l129_129052

theorem sum_of_solutions_equation : 
  let sum_of_solutions := (λ (x : ℝ), ∃ y1 y2 : ℝ, (y1 = √6 ∨ y1 = -√6) ∧ (y2 = √6 ∨ y2 = -√6) ∧ y1 + y2 = 0)
  in sum_of_solutions := 0 := 
sorry

end sum_of_solutions_equation_l129_129052


namespace num_ways_product_1000000_l129_129930

theorem num_ways_product_1000000 : ∃ n : ℕ, n = 139 ∧ 
  ∀ (a b c : ℕ), a * b * c = 1000000 -> 
    ∃ (x y z : ℕ), {2 ^ x * 5 ^ y, 2 ^ (6 - x) * 5 ^ (6 - y), 2 ^ z * 5 ^ (6 - z)} = 
      {a, b, c} →
        (∀ (i j k: ℕ), i + j + k = 6 → 
        ∃ (l m n: ℕ), {i, j, k} = {l, m, n}):= 
          ∃ comb, comb.length = n ∧ comb ≤ distributive :
                ∃  by sorry

end num_ways_product_1000000_l129_129930


namespace natasha_dimes_l129_129976

theorem natasha_dimes (n : ℕ) (h1 : 10 < n) (h2 : n < 100) (h3 : n % 3 = 1) (h4 : n % 4 = 1) (h5 : n % 5 = 1) : n = 61 :=
sorry

end natasha_dimes_l129_129976


namespace uniform_price_A_uniform_price_B_uniform_price_C_l129_129285

variables {Rs : Type} [LinearOrderedField Rs]

-- Conditions for Servant A
def condition_A (U_A Rs : Type) [LinearOrderedField Rs] : Prop :=
  (9 * (500 : Rs) + 9 * U_A = 3000 : Rs)

-- Conditions for Servant B
def condition_B (U_B Rs : Type) [LinearOrderedField Rs] : Prop :=
  (6 * (800 : Rs) + 6 * U_B = 3600 : Rs)

-- Conditions for Servant C
def condition_C (U_C Rs : Type) [LinearOrderedField Rs] : Prop :=
  (4 * (1200 : Rs) + 4 * U_C = 2400 : Rs)

-- Theorem for Servant A
theorem uniform_price_A (U_A : Rs) [noncomputable theory] : condition_A Rs ∧ U_A = (500 : Rs) := 
begin
  sorry
end

-- Theorem for Servant B
theorem uniform_price_B (U_B : Rs) [noncomputable theory] : condition_B Rs ∧ U_B = (200 : Rs) := 
begin
  sorry
end

-- Theorem for Servant C
theorem uniform_price_C (U_C : Rs) [noncomputable theory] : condition_C Rs ∧ U_C = (300 : Rs) := 
begin
  sorry
end

end uniform_price_A_uniform_price_B_uniform_price_C_l129_129285


namespace paired_fraction_l129_129927

theorem paired_fraction (t s : ℕ)
  (h1 : t / 4 = (3 * s) / 7) : 
  (3 : ℚ) / 19 = ( (3 : ℚ) / 7) / ( (12 : ℚ) / 7 + s) / (19 : ℚ / 7):=

begin
    sorry
end

end paired_fraction_l129_129927


namespace correct_option_among_sqrt_statements_l129_129747

theorem correct_option_among_sqrt_statements :
  ¬ (sqrt 16 = -4 ∨ sqrt 16 = 4) ∧
  ¬ (sqrt ((-3)^2) = -3) ∧
  (sqrt 81 = 9 ∨ -sqrt 81 = -9) ∧
  ¬ (sqrt (- 4) = 2) ∧
  ( (sqrt 16 = 4 ∨ sqrt 16 = -4) ∧
    (sqrt ((-3)^2) = 3) ∧
    (sqrt 81 = 9 ∨ -sqrt 81 = -9) ∧
    ¬ sqrt (-4)) →  
  true :=
by
  sorry

end correct_option_among_sqrt_statements_l129_129747


namespace total_amount_l129_129378

theorem total_amount (A N J : ℕ) (h1 : A = N - 5) (h2 : J = 4 * N) (h3 : J = 48) : A + N + J = 67 :=
by
  -- Proof will be constructed here
  sorry

end total_amount_l129_129378


namespace sum_of_first_11_terms_l129_129074

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

axiom a_3_condition : a 3 = 4
axiom a_4_9_condition : a 4 + a 9 = 22

theorem sum_of_first_11_terms (a : ℕ → ℤ) (S : ℕ → ℤ) [is_arithmetic_sequence a] [sum_of_first_n_terms a S] :
  S 11 = 110 :=
by
  sorry

end sum_of_first_11_terms_l129_129074


namespace f_240_equals_388_l129_129592

def strictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ n m, (n < m) → (f n < f m)

def sequenceProperty (f : ℕ → ℕ) : Prop :=
  ∀ n, f(f(n)) = f(n) + n - 1

variable (f : ℕ → ℕ)

theorem f_240_equals_388 (h1: strictlyIncreasing f) (h2: sequenceProperty f) : 
  f 240 = 388 := by
    sorry

end f_240_equals_388_l129_129592


namespace routes_have_8_stations_each_l129_129354

-- Definitions:
def Station := Type
def Route (α : Type) := Set α

constant station : Station → Station → Prop
constant route (r : Route Station) : Set Station
constant city_routes : Set (Route Station)
constant n_routes : city_routes.card = 57

-- Conditions:
axiom condition1 : ∀ s₁ s₂ : Station, ∃ r : Route Station, station s₁ r ∧ station s₂ r
axiom condition2 : ∀ r₁ r₂ : Route Station, r₁ ≠ r₂ → ∃! s : Station, station s r₁ ∧ station s r₂
axiom condition3 : ∀ r : Route Station, (route r).card ≥ 3

-- Theorem:
theorem routes_have_8_stations_each : ∀ r : Route Station, (route r).card = 8 :=
by
  sorry

end routes_have_8_stations_each_l129_129354


namespace cactus_jumping_difference_l129_129793

theorem cactus_jumping_difference :
  let num_cacti := 31
  let total_distance := 3720
  let derek_hops_per_gap := 30
  let rory_jumps_per_gap := 10
  let num_gaps := num_cacti - 1
  let derek_total_hops := num_gaps * derek_hops_per_gap
  let rory_total_jumps := num_gaps * rory_jumps_per_gap
  let derek_hop_length := total_distance / derek_total_hops
  let rory_jump_length := total_distance / rory_total_jumps
  (rory_jump_length - derek_hop_length) = 8.27 :=
by
  have num_cacti := 31
  have total_distance := 3720
  have derek_hops_per_gap := 30
  have rory_jumps_per_gap := 10
  have num_gaps := num_cacti - 1
  have derek_total_hops := num_gaps * derek_hops_per_gap
  have rory_total_jumps := num_gaps * rory_jumps_per_gap
  have derek_hop_length := total_distance / derek_total_hops
  have rory_jump_length := total_distance / rory_total_jumps
  show rory_jump_length - derek_hop_length = 8.27
  -- Proof omitted
  sorry

end cactus_jumping_difference_l129_129793


namespace find_b_value_l129_129254

-- Define the conditions: line equation and given range for b
def line_eq (x : ℝ) (b : ℝ) : ℝ := b - x

-- Define the points P, Q, S
def P (b : ℝ) : ℝ × ℝ := ⟨0, b⟩
def Q (b : ℝ) : ℝ × ℝ := ⟨b, 0⟩
def S (b : ℝ) : ℝ × ℝ := ⟨6, b - 6⟩

-- Define the area ratio condition
def area_ratio_condition (b : ℝ) : Prop :=
  (0 < b ∧ b < 6) ∧ ((6 - b) / b) ^ 2 = 4 / 25

-- Define the main theorem to prove
theorem find_b_value (b : ℝ) : area_ratio_condition b → b = 4.3 := by
  sorry

end find_b_value_l129_129254


namespace part1_part2_l129_129391

section
variables (s t r x y : ℝ)
variables (h_s : s > 0) (h_t : t > 0) (h_r : r > 0) (h_x : x > 0) (h_y : y > 0)

-- Part 1
theorem part1 : ( ( 8 * s^6 * t^(-3) / (125 * r^9)) ^ (-2/3) = 25 * r^6 * t^2 / (4 * s^4) ) :=
by sorry

-- Part 2
theorem part2 : ( (3 * x^(1/4) + 2 * y^(-1/2)) * (3 * x^(1/4) - 2 * y^(-1/2)) = 9 * x^(1/2) - 4 * y^(-1) ) :=
by sorry

end

end part1_part2_l129_129391


namespace digits_arithmetic_l129_129325

theorem digits_arithmetic :
  (12 / 3 / 4) * (56 / 7 / 8) = 1 :=
by
  sorry

end digits_arithmetic_l129_129325


namespace area_square_15_cm_l129_129436

-- Define the side length of the square
def side_length : ℝ := 15

-- Define the area calculation for a square given the side length
def area_of_square (s : ℝ) : ℝ := s * s

-- The theorem statement translating the problem to Lean
theorem area_square_15_cm :
  area_of_square side_length = 225 :=
by
  -- We need to provide proof here, but 'sorry' is used to skip the proof as per instructions
  sorry

end area_square_15_cm_l129_129436


namespace smallest_s_for_F_l129_129011

def F (a b c d : ℕ) : ℕ := a * b^(c^d)

theorem smallest_s_for_F :
  ∃ s : ℕ, F s s 2 2 = 65536 ∧ ∀ t : ℕ, F t t 2 2 = 65536 → s ≤ t :=
sorry

end smallest_s_for_F_l129_129011


namespace part1_part2_l129_129071

noncomputable def seq_a (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3^(n-1)

noncomputable def S (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), seq_a i

noncomputable def seq_b (n : ℕ) : ℝ :=
  let a_n := seq_a n
  in 1 / ((1 + Real.log 3 a_n) * (3 + Real.log 3 a_n))

noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), seq_b i

theorem part1 (n : ℕ) (h : n ≠ 0) : seq_a n = 3^(n-1) := by
  sorry

theorem part2 (m : ℝ) (h : ∀ n : ℕ, T n < m) : m ≥ 3/4 := by
  sorry

end part1_part2_l129_129071


namespace mr_williams_land_percentage_l129_129250

-- Given conditions
def farm_tax_percent : ℝ := 60
def total_tax_collected : ℝ := 5000
def mr_williams_tax_paid : ℝ := 480

-- Theorem statement
theorem mr_williams_land_percentage :
  (mr_williams_tax_paid / total_tax_collected) * 100 = 9.6 := by
  sorry

end mr_williams_land_percentage_l129_129250


namespace second_arrangement_people_per_column_l129_129924
-- Import necessary library

-- Define the number of people per column and number of columns for the first arrangement
def firstArrangement (people_per_column1 columns1 : ℕ) : ℕ :=
  people_per_column1 * columns1

-- Define the number of people per column needed for a given total number of people and columns
def peoplePerColumn (total_people columns2 : ℕ) : ℕ :=
  total_people / columns2

-- Theorem statement: Given the total number of people based on the first arrangement,
-- we need to prove the number of people per column for the second arrangement equals 32.
theorem second_arrangement_people_per_column :
  let total_people := firstArrangement 30 16 in
  peoplePerColumn total_people 15 = 32 :=
by
  sorry

end second_arrangement_people_per_column_l129_129924


namespace product_of_divisors_72_l129_129466

theorem product_of_divisors_72 : (∏ d in (finset.divisors 72), d) = 139314069504 :=
by 
  -- Proof goes here
  sorry

end product_of_divisors_72_l129_129466


namespace sum_of_three_ints_product_5_4_l129_129260

theorem sum_of_three_ints_product_5_4 :
  ∃ (a b c: ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 51 :=
by
  sorry

end sum_of_three_ints_product_5_4_l129_129260


namespace line_equation_through_point_and_max_distance_l129_129543

theorem line_equation_through_point_and_max_distance :
  ∃ l : ℝ → ℝ, (l 2 = 3) ∧ (∀ x y : ℝ, (x, y) = (-3, 2) → abs ((-5) * x + y - 13) / sqrt ((-5)^2 + 1) = sorry) :=
sorry

end line_equation_through_point_and_max_distance_l129_129543


namespace proof_problem_1_proof_problem_2_l129_129024

noncomputable def problem_1 : ℝ :=
  (9 / 4) ^ (1 / 2) - (-9.6) ^ 0 - (27 / 8) ^ (-2 / 3) + (3 / 2) ^ (-2)

theorem proof_problem_1 : problem_1 = 1 / 2 :=
by {
  let_problem_1 := (9 / 4) ^ (1 / 2) - (-9.6) ^ 0 - (27 / 8) ^ (-2 / 3) + (3 / 2) ^ (-2),
  have h1 : (9 / 4) ^ (1 / 2) = 3 / 2 := sorry,
  have h2 : (-9.6) ^ 0 = 1 := sorry,
  have h3 : (27 / 8) ^ (-2 / 3) = 4 / 9 := sorry,
  have h4 : (3 / 2) ^ (-2) = 4 / 9 := sorry,
  rw [←h1,←h2, ←h3,←h4],
  simp,
  ring,
}

noncomputable def problem_2 : ℝ :=
  log10 14 - 2 * log10 (7 / 3) + log10 7 - log10 18

theorem proof_problem_2 : problem_2 = 0 :=
by {
  let_problem_2 := log10 14 - 2 * log10 (7 / 3) + log10 7 - log10 18,
  have h1 : log10 (14 * 7 / ((7 / 3) ^ 2)) = log10 1 := sorry,
  rw [←h1],
  simp,
}

end proof_problem_1_proof_problem_2_l129_129024


namespace inradius_comparison_l129_129575

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def midpoint_of_arc (Γ : Circle) (A B C : Point) : Point := sorry
noncomputable def circumradius (A B C : Point) : Real := sorry
noncomputable def inradius (A B C : Point) : Real := sorry
noncomputable def distance (P Q : Point) : Real := sorry

theorem inradius_comparison 
  (A B C : Point) 
  (h_acute : IsAcuteTri (Triangle.mk A B C)) 
  (Γ : Circle) 
  (h_circumcircle : IsCircumcircle Γ (Triangle.mk A B C))
  (A1 B1 C1 : Point)
  (h_A1 : A1 = midpoint_of_arc Γ A B C)
  (h_B1 : B1 = midpoint_of_arc Γ B C A)
  (h_C1 : C1 = midpoint_of_arc Γ C A B) :
  (inradius (Triangle.mk A1 B1 C1) ≥ inradius (Triangle.mk A B C)) := 
sorry

end inradius_comparison_l129_129575


namespace isosceles_base_length_l129_129231

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l129_129231


namespace twenty_percent_l129_129117

-- Given condition
def condition (X : ℝ) : Prop := 0.4 * X = 160

-- Theorem to show that 20% of X equals 80 given the condition
theorem twenty_percent (X : ℝ) (h : condition X) : 0.2 * X = 80 :=
by sorry

end twenty_percent_l129_129117


namespace apples_percentage_difference_l129_129617

def percentage_difference : ℝ → ℝ → ℝ := λ S I, ((S - I) / I) * 100

theorem apples_percentage_difference (H : ℝ) (H_pos : 0 < H) :
  let S := 1.80 * H
  let I := 1.40 * H
  percentage_difference S I = 29 :=
by
  intro H H_pos
  let S := 1.80 * H
  let I := 1.40 * H
  have diff := percentage_difference S I
  rw [percentage_difference]
  sorry

end apples_percentage_difference_l129_129617


namespace integral_cos_power_eight_l129_129338

theorem integral_cos_power_eight : 
  ∫ x in - (Real.pi / 2), 0, 2^8 * (Real.cos x)^8 = 35 * Real.pi :=
sorry

end integral_cos_power_eight_l129_129338


namespace waxberry_problem_l129_129761

noncomputable def batch_cannot_be_sold : ℚ := 1 - (8 / 9 * 9 / 10)

def probability_distribution (X : ℚ) : ℚ := 
  if X = -3200 then (1 / 5)^4 else
  if X = -2000 then 4 * (1 / 5)^3 * (4 / 5) else
  if X = -800 then 6 * (1 / 5)^2 * (4 / 5)^2 else
  if X = 400 then 4 * (1 / 5) * (4 / 5)^3 else
  if X = 1600 then (4 / 5)^4 else 0

noncomputable def expected_profit : ℚ :=
  -3200 * probability_distribution (-3200) +
  -2000 * probability_distribution (-2000) +
  -800 * probability_distribution (-800) +
  400 * probability_distribution (400) +
  1600 * probability_distribution (1600)

theorem waxberry_problem : 
  batch_cannot_be_sold = 1 / 5 ∧ 
  (probability_distribution (-3200) = 1 / 625 ∧ 
   probability_distribution (-2000) = 16 / 625 ∧ 
   probability_distribution (-800) = 96 / 625 ∧ 
   probability_distribution (400) = 256 / 625 ∧ 
   probability_distribution (1600) = 256 / 625) ∧ 
  expected_profit = 640 :=
by 
  sorry

end waxberry_problem_l129_129761


namespace min_cards_to_guarantee_four_same_suit_l129_129134

theorem min_cards_to_guarantee_four_same_suit (n : ℕ) (suits : Fin n) (cards_per_suit : ℕ) (total_cards : ℕ)
  (h1 : n = 4) (h2 : cards_per_suit = 13) : total_cards ≥ 13 :=
by
  sorry

end min_cards_to_guarantee_four_same_suit_l129_129134


namespace find_intervals_find_k_max_max_k_value_l129_129876

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 2
noncomputable def f_derivative (x a : ℝ) := Real.exp x - a
noncomputable def g (x : ℝ) := (x + 1) / (Real.exp x - 1) + x

theorem find_intervals :
  ∀ (a x : ℝ), 
  (a ≤ 0 → ∀ (x : ℝ), f_derivative x a ≥ 0 ∧ Monotone f) ∧ 
  (a > 0 → ∀ (x : ℝ), (x < Real.log a → f_derivative x a < 0) ∧ (x > Real.log a → f_derivative x a > 0) ∧
  StrictMonoOn f (Set.Ioc (Real.log a) ∞) ∧ StrictAntiOn f (Set.Ioo (−∞) (Real.log a))) :=
sorry

theorem find_k_max (k : ℤ) :
  ∀ (x : ℝ), 0 < x → a = 1 → (k - x) * (Real.exp x - 1) < x + 1 → k < (g x) :=
sorry

theorem max_k_value :
  ∃ (k : ℤ), k ≤ 2 ∧ ∀ (k' : ℤ), k' > k → ¬ (∀ (x : ℝ), 0 < x → (k' - x) * (Real.exp x - 1) < x + 1) :=
sorry

end find_intervals_find_k_max_max_k_value_l129_129876


namespace price_calculation_l129_129383

def original_price (x : ℝ) : ℝ :=
  1.122 * x

theorem price_calculation (x : ℝ) :
  original_price x = 195.50 → x = 174.50 :=
begin
  intro h,
  sorry
end

end price_calculation_l129_129383


namespace sequence_divisible_by_11_l129_129842

theorem sequence_divisible_by_11 {a : ℕ → ℕ} (h1 : a 1 = 1) (h2 : a 2 = 3)
    (h_rec : ∀ n : ℕ, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n) :
    (a 4 % 11 = 0) ∧ (a 8 % 11 = 0) ∧ (a 10 % 11 = 0) ∧ (∀ n, n ≥ 11 → a n % 11 = 0) :=
by
  sorry

end sequence_divisible_by_11_l129_129842


namespace train_from_city_A_starts_at_8am_l129_129787

def train_start_time_from_city_A
    (distance_AB : ℝ)
    (speed_A : ℝ)
    (start_time_B : ℝ)
    (speed_B : ℝ)
    (meet_time : ℝ) : ℝ :=
if h : speed_A > 0 ∧ speed_B > 0 ∧ distance_AB > 0 ∧ meet_time > start_time_B then
  let time_B_travelled := meet_time - start_time_B in
  let distance_B_covered := speed_B * time_B_travelled in
  let distance_A_covered := distance_AB - distance_B_covered in
  let time_A_travelled := distance_A_covered / speed_A in
  meet_time - time_A_travelled
else
  0

theorem train_from_city_A_starts_at_8am :
  train_start_time_from_city_A 465 60 9 75 12 = 8 :=
by
  sorry

end train_from_city_A_starts_at_8am_l129_129787


namespace sequence_difference_l129_129845

-- Definition of the sequence a_n
def a_n (n : ℕ) : ℂ :=
  (1 + complex.I) *
  (list.prod (list.map (λ k : ℕ, (1 + complex.I / complex.sqrt (k + 1))) (list.range n)))

-- Statement of the problem
theorem sequence_difference (n : ℕ) :
  |a_n n - a_n (n + 1)| = 1 :=
sorry

end sequence_difference_l129_129845


namespace number_of_monograms_l129_129601

theorem number_of_monograms : 
  let middle_initial := 'M'
  ∃ (first_last_initials : (list char)), 
    (∀ xi ∈ first_last_initials, xi < middle_initial ∨ xi > middle_initial) ∧
    (first_last_initials.length = 2) ∧
    (first_last_initials.nodup) ∧
    ((first_last_initials.head < middle_initial) ∧ (first_last_initials.tail.head > middle_initial))
  → (12 * 13 = 156) :=
by
  sorry

end number_of_monograms_l129_129601


namespace translation_right_π_over_8_l129_129321

-- Define the two functions
def f1 (x : ℝ) : ℝ := cos (2 * x + π / 4)
def f2 (x : ℝ) : ℝ := cos (2 * x)

-- Define the translation
def translation (x δ : ℝ) : ℝ := x + δ

-- The theorem to be proven
theorem translation_right_π_over_8 :
  ∀ x : ℝ, f1 (translation x (π / 8)) = f2 x := 
by 
  sorry

end translation_right_π_over_8_l129_129321


namespace regular_nonagon_diagonal_relation_l129_129926

theorem regular_nonagon_diagonal_relation
  (a c d : ℝ)
  (h_c : c^2 = a^2)
  (h_d : d^2 = 2 * a^2 * (1 - real.cos (20 * real.pi / 180))) :
  d^2 = 5 * a^2 :=
by
  sorry

end regular_nonagon_diagonal_relation_l129_129926


namespace triangle_stability_application_l129_129323

theorem triangle_stability_application (tree_is_falling : Prop) (wooden_sticks_support_tree : Prop):
  (application_of_triangle_property == stability) :=
by
  sorry

end triangle_stability_application_l129_129323


namespace sin_660_eq_neg_sqrt_3_div_2_l129_129274

theorem sin_660_eq_neg_sqrt_3_div_2 : sin (660 * real.pi / 180) = - (real.sqrt 3) / 2 := 
by 
  -- Proof is omitted as the solution steps are not necessary
  sorry

end sin_660_eq_neg_sqrt_3_div_2_l129_129274


namespace least_number_to_add_l129_129314

theorem least_number_to_add (n divisor : ℕ) (h₁ : n = 27306) (h₂ : divisor = 151) : 
  ∃ k : ℕ, k = 25 ∧ (n + k) % divisor = 0 := 
by
  sorry

end least_number_to_add_l129_129314


namespace No_of_boxes_in_case_l129_129483

-- Define the conditions
def George_has_total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def George_has_boxes : ℕ := George_has_total_blocks / blocks_per_box

-- The theorem to prove
theorem No_of_boxes_in_case : George_has_boxes = 2 :=
by
  sorry

end No_of_boxes_in_case_l129_129483


namespace volume_of_intersection_l129_129737

def condition1 (x y z : ℝ) : Prop := |x| + |y| + |z| ≤ 1
def condition2 (x y z : ℝ) : Prop := |x| + |y| + |z - 2| ≤ 1
def in_intersection (x y z : ℝ) : Prop := condition1 x y z ∧ condition2 x y z

theorem volume_of_intersection : 
  (∫ x y z in { p : ℝ × ℝ × ℝ | in_intersection p.1 p.2 p.3 }, 1) = 1/12 := 
by
  sorry

end volume_of_intersection_l129_129737


namespace pyramid_volume_correct_l129_129782

noncomputable def PyramidVolume (base_area : ℝ) (triangle_area_1 : ℝ) (triangle_area_2 : ℝ) : ℝ :=
  let side := Real.sqrt base_area
  let height_1 := (2 * triangle_area_1) / side
  let height_2 := (2 * triangle_area_2) / side
  let h_sq := height_1 ^ 2 - (Real.sqrt (height_1 ^ 2 + height_2 ^ 2 - 512)) ^ 2
  let height := Real.sqrt h_sq
  (1/3) * base_area * height

theorem pyramid_volume_correct :
  PyramidVolume 256 120 112 = 1163 := by
  sorry

end pyramid_volume_correct_l129_129782


namespace speed_of_body_l129_129765

variable (v_0 g : ℝ) (t : ℝ)

-- Assuming initial conditions
axiom initial_velocity : v_0 > 0
axiom no_air_resistance : true  -- implicitly means air resistance is negligible
axiom eq_displacement : v_0 * t = (1 / 2) * g * t^2

-- Prove the speed of the body equals sqrt(5) * v_0 when vertical displacement equals horizontal displacement
theorem speed_of_body : sqrt (v_0^2 + (g * t)^2) = sqrt(5) * v_0 := by
  sorry

end speed_of_body_l129_129765


namespace sum_first_100_terms_l129_129505

def a (n : ℕ) : ℤ := (-1) ^ (n + 1) * n

def S (n : ℕ) : ℤ := Finset.sum (Finset.range n) (λ i => a (i + 1))

theorem sum_first_100_terms : S 100 = -50 := 
by 
  sorry

end sum_first_100_terms_l129_129505


namespace intervals_monotonically_increasing_sin_2theta_solution_l129_129096

open Real

-- Define the function f(x) based on the given condition
def f (x : ℝ) : ℝ := (cos x)^2 - sqrt 3 * (sin x) * (cos x) + 1

-- Problem Part I: Finding monotonic intervals
theorem intervals_monotonically_increasing :
  ∀ k : ℤ, monotonically_increasing_interval (λ x, f x) (k * π + π/3) (k * π + 5*π/6) :=
by
  -- proof omitted
  sorry

-- Problem Part II: Given condition and finding value of sin 2θ
theorem sin_2theta_solution (θ : ℝ) (h1 : θ ∈ Ioo (π/3) (2*π/3)) (h2 : f θ = 5/6) :
  sin (2 * θ) = (2 * sqrt 3 - sqrt 5) / 6 :=
by
  -- proof omitted
  sorry

end intervals_monotonically_increasing_sin_2theta_solution_l129_129096


namespace option_C_is_correct_l129_129536

theorem option_C_is_correct (a b c : ℝ) (h : a > b) : c - a < c - b := 
by
  linarith

end option_C_is_correct_l129_129536


namespace comparison_of_abc_l129_129955

noncomputable def a : ℝ := (4 - Real.log 4) / Real.exp 2
noncomputable def b : ℝ := Real.log 2 / 2
noncomputable def c : ℝ := 1 / Real.exp 1

theorem comparison_of_abc : b < a ∧ a < c :=
by
  sorry

end comparison_of_abc_l129_129955


namespace last_three_digits_N_l129_129578

theorem last_three_digits_N :
  let N := {n : ℕ | ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 2016 ∧ 1 ≤ b ∧ b ≤ 2016 ∧ 1 ≤ c ∧ c ≤ 2016 ∧ (a^2 + b^2 + c^2) % 2017 = 0 }.card in
  N % 1000 = 0 :=
by
  sorry

end last_three_digits_N_l129_129578


namespace count_lines_passing_through_three_points_l129_129896

theorem count_lines_passing_through_three_points :
  let points := { (i, j, k) : ℕ × ℕ × ℕ // i ≤ 5 ∧ j ≤ 5 ∧ k ≤ 5 ∧ 1 ≤ i ∧ 1 ≤ j ∧ 1 ≤ k } in
  let valid_directions := { (a, b, c) : ℤ × ℤ × ℤ // a ∈ {-2, -1, 0, 1, 2} ∧ b ∈ {-2, -1, 0, 1, 2} ∧ c ∈ {-2, -1, 0, 1, 2} ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) } in
  (∑ i1 ∈ (Finset.range 5).filter (λ x, x > 0), 
   ∑ j1 ∈ (Finset.range 5).filter (λ y, y > 0), 
   ∑ k1 ∈ (Finset.range 5).filter (λ z, z > 0),
   ∑ a ∈ valid_directions,
   ∑ j2 ∈ (Finset.range 5).filter (λ y, y > 0),
   ∑ k2 ∈ (Finset.range 5).filter (λ z, z > 0),
   if 1 ≤ i1 + a.1 ∧ i1 + a.1 ≤ 5 ∧ 1 ≤ j1 + j2 ∧ j1 + j2 ≤ 5 ∧ 1 ≤ k1 + k2 ∧ k1 + k2 ≤ 5 ∧ 
      1 ≤ i1 + 2 * a.1 ∧ i1 + 2 * a.1 ≤ 5 ∧ 1 ≤ j1 + 2 * j2 ∧ j1 + 2 * j2 ≤ 5 ∧ 1 ≤ k1 + 2 * k2 ∧ k1 + 2 * k2 ≤ 5
   then 1 else 0) = 120 :=
sorry

end count_lines_passing_through_three_points_l129_129896


namespace find_q_l129_129499

noncomputable def q : ℝ :=
  let a_1 : ℝ := 1 -- Let a_1 be an arbitrary positive number because it will cancel out.
  let a_2 := a_1 * q
  let a_3 := a_1 * q^2
  let a_4 := a_1 * q^3
  let a_7 := a_1 * q^6
  let condition1 := (a_2 * a_3 = 2 * a_1)
  let condition2 := ((a_4 + 2 * a_7) / 2 = 5 / 4)
  q -- This has to be defined as the solution we want to verify

theorem find_q (a_1 q : ℝ) (h1 : a_1 * q * (a_1 * q^2) = 2 * a_1)
  (h2 : (a_1 * q^3 + 2 * a_1 * (q^6)) / 2 = 5 / 4) :
q = 1 / 2 :=
by
    sorry

end find_q_l129_129499


namespace length_of_BC_l129_129565

theorem length_of_BC (A B C : Type*) [euclidean_geometry (A B C)] 
  (angle_A : has_angle A B C) (AngleA_eq : angle_A = 60) 
  (AC AB BC : ℝ)
  (roots_eq : ∀ x, x^2 - 5*x + 6 = 0 → x = AC ∨ x = AB)
  (vieta1 : AC + AB = 5)
  (vieta2 : AC * AB = 6) :
  BC = real.sqrt 7 :=
by
  -- sorry serves as a placeholder for the proof
  sorry

end length_of_BC_l129_129565


namespace find_wrong_guess_l129_129690

-- Define the three colors as an inductive type.
inductive Color
| white
| brown
| black

-- Define the bears as a list of colors.
def bears (n : ℕ) : Type := list Color

-- Define the conditions: 
-- There are 1000 bears and each tuple of 3 consecutive bears has all three colors.
def valid_bears (b : bears 1000) : Prop :=
  ∀ i : ℕ, i + 2 < 1000 → 
    ∃ c1 c2 c3 : Color, 
      c1 ∈ b.nth i ∧ c2 ∈ b.nth (i+1) ∧ c3 ∈ b.nth (i+2) ∧ 
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3

-- Define Iskander's guesses.
def guesses (b : bears 1000) : Prop :=
  b.nth 1 = some Color.white ∧
  b.nth 19 = some Color.brown ∧
  b.nth 399 = some Color.black ∧
  b.nth 599 = some Color.brown ∧
  b.nth 799 = some Color.white

-- Prove that exactly one of Iskander's guesses is wrong.
def wrong_guess (b : bears 1000) : Prop :=
  (b.nth 19 ≠ some Color.brown) ∧
  valid_bears b ∧
  guesses b →
  ∃ i, i ∈ {1, 19, 399, 599, 799} ∧ (b.nth i ≠ some Color.white ∧ b.nth i ≠ some Color.brown ∧ b.nth i ≠ some Color.black)

theorem find_wrong_guess : 
  ∀ b : bears 1000, 
  valid_bears b → guesses b → wrong_guess b :=
  by
  intros b vb gs
  sorry

end find_wrong_guess_l129_129690


namespace comparison_of_abc_l129_129954

noncomputable def a : ℝ := (4 - Real.log 4) / Real.exp 2
noncomputable def b : ℝ := Real.log 2 / 2
noncomputable def c : ℝ := 1 / Real.exp 1

theorem comparison_of_abc : b < a ∧ a < c :=
by
  sorry

end comparison_of_abc_l129_129954


namespace sum_of_reciprocals_of_squares_l129_129194

open BigOperators

theorem sum_of_reciprocals_of_squares (n : ℕ) (h : n ≥ 2) :
   (∑ k in Finset.range n, 1 / (k + 1)^2) < (2 * n - 1) / n :=
sorry

end sum_of_reciprocals_of_squares_l129_129194


namespace incorrect_guess_20_l129_129672

-- Define the assumptions and conditions
def bears : Nat → String := sorry -- function that determines the color of the bear at position n
axiom bears_color_constraint : ∀ n:Nat, exists b:List String, b.length = 3 ∧ Set ("W" "B" "Bk") = List.toSet b ∧ 
  List.all (List.sublist b (n, n+1, n+2) bears = fun c=> c = "W" or c = "B" or c = "Bk") 

-- Iskander's guesses
def guess1 := (2, "W")
def guess2 := (20, "B")
def guess3 := (400, "Bk")
def guess4 := (600, "B")
def guess5 := (800, "W")

-- Function to check the bear at each position
def check_bear (n:Nat) : String := sorry

-- Iskander's guess correctness, exactly one is wrong
axiom one_wrong : count (check_bear 2 =="W") 
                         + count (check_bear 20 == "B") 
                         + count (check_bear 400 =="Bk") 
                         + count (check_bear 600 =="B") 
                         + count (check_bear 800 =="W") = 4

-- Prove that the guess for the 20th bear is incorrect
theorem incorrect_guess_20 : ∀ {n:Nat} (h : n=20), (check_bear n != "B") := sorry

end incorrect_guess_20_l129_129672


namespace price_difference_is_25_l129_129364

-- Define the conditions
variables (actual_gallons expected_gallons : ℕ) (actual_price : ℕ)

-- Define the amount of money the motorist had
def total_money := actual_gallons * actual_price

-- Define the expected price per gallon
noncomputable def expected_price : ℕ := (total_money / expected_gallons : ℕ)

-- Define the difference in price
def price_difference := actual_price - expected_price

-- The main theorem we want to prove
theorem price_difference_is_25 
  (h1 : actual_gallons = 10) 
  (h2 : expected_gallons = 12)
  (h3 : actual_price = 150) :
  price_difference actual_gallons expected_gallons actual_price = 25 :=
by
  sorry

end price_difference_is_25_l129_129364


namespace count_L_shapes_l129_129061

theorem count_L_shapes (m n : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) : 
  ∃ k, k = 4 * (m - 1) * (n - 1) :=
by
  sorry

end count_L_shapes_l129_129061


namespace find_x_ge_0_l129_129434

-- Defining the condition and the proof problem
theorem find_x_ge_0 :
  {x : ℝ | (x^2 + 2*x^4 - 3*x^5) / (x + 2*x^3 - 3*x^4) ≥ 0} = {x : ℝ | 0 ≤ x} :=
by
  sorry -- proof steps not included

end find_x_ge_0_l129_129434


namespace find_a_l129_129149

theorem find_a (P : ℝ) (hP : P ≠ 0) (S : ℕ → ℝ) (a_n : ℕ → ℝ)
  (hSn : ∀ n, S n = 3^n + a)
  (ha_n : ∀ n, a_n (n + 1) = P * a_n n)
  (hS1 : S 1 = a_n 1)
  (hS2 : S 2 = S 1 + a_n 2 - a_n 1)
  (hS3 : S 3 = S 2 + a_n 3 - a_n 2) :
  a = -1 := sorry

end find_a_l129_129149


namespace beka_distance_l129_129388

theorem beka_distance (jackson_distance : ℕ) (beka_more_than_jackson : ℕ) :
  jackson_distance = 563 → beka_more_than_jackson = 310 → 
  (jackson_distance + beka_more_than_jackson = 873) :=
by
  sorry

end beka_distance_l129_129388


namespace correct_option_l129_129887

variables {α β γ : Plane} {m n : Line}

-- Conditions
def α_perpendicular_β : Prop := α ⟂ β
def intersect_line_m : α ∩ β = m
def n_perpendicular_α : n ⟂ α
def n_within_γ : n ∈ γ

-- The problem statement to be proved
theorem correct_option (h1 : α_perpendicular_β) (h2 : intersect_line_m) (h3 : n_perpendicular_α) (h4 : n_within_γ) : 
  m ⟂ n ∧ α ⟂ γ :=
sorry

end correct_option_l129_129887


namespace equilateral_triangle_area_decrease_l129_129382

theorem equilateral_triangle_area_decrease :
  let original_area : ℝ := 100 * Real.sqrt 3
  let side_length_s := 20
  let decreased_side_length := side_length_s - 6
  let new_area := (decreased_side_length * decreased_side_length * Real.sqrt 3) / 4
  let decrease_in_area := original_area - new_area
  decrease_in_area = 51 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_decrease_l129_129382


namespace number_2016_in_group_63_l129_129526

theorem number_2016_in_group_63 : 
  ∃ (k : ℕ), 2016 ∈ (∑ i in Finset.range k, (Finset.range (i + 1)).map (Nat.add (∑ j in Finset.range i, j + 1))) :=
sorry

end number_2016_in_group_63_l129_129526


namespace right_triangle_l129_129744

theorem right_triangle (a b c : ℝ) : 
  ( (a = 1 ∧ b = 2 ∧ c = 2)   ∨ 
    (a = 1 ∧ b = 2 ∧ c = √3)  ∨ 
    (a = 4 ∧ b = 5 ∧ c = 6)   ∨ 
    (a = 1 ∧ b = 1 ∧ c = √3) )
  → 
  (a^2 + b^2 = c^2 ↔ (a = 1 ∧ b = 2 ∧ c = √3)) := 
by
  sorry

end right_triangle_l129_129744


namespace fourth_power_sqrt_eq_256_l129_129827

theorem fourth_power_sqrt_eq_256 (x : ℝ) (h : (x^(1/2))^4 = 256) : x = 16 := by sorry

end fourth_power_sqrt_eq_256_l129_129827


namespace percentage_le_29_l129_129070

def sample_size : ℕ := 100
def freq_17_19 : ℕ := 1
def freq_19_21 : ℕ := 1
def freq_21_23 : ℕ := 3
def freq_23_25 : ℕ := 3
def freq_25_27 : ℕ := 18
def freq_27_29 : ℕ := 16
def freq_29_31 : ℕ := 28
def freq_31_33 : ℕ := 30

theorem percentage_le_29 : (freq_17_19 + freq_19_21 + freq_21_23 + freq_23_25 + freq_25_27 + freq_27_29) * 100 / sample_size = 42 :=
by
  sorry

end percentage_le_29_l129_129070


namespace solve_system_equations_l129_129623

def system_solutions (x y z w : ℝ) : Prop :=
  x + y + Real.sqrt z = 4 ∧ Real.sqrt x * Real.sqrt y - Real.sqrt w = 2

theorem solve_system_equations :
  system_solutions 2 2 0 0 :=
by
  simp [system_solutions]
  split
  { norm_num },
  { norm_num }

end solve_system_equations_l129_129623


namespace increase_prob_n_7_l129_129284

def S : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

noncomputable def N := {x ∈ S | ∃ y z ∈ S, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x + y + z = 15}

theorem increase_prob_n_7 : 
  ∃ n : ℕ, n = 7 ∧ (∀ T ⊆ S, T.card = 3 → (∃ x y z ∈ T, x + y + z = 15) → 
  P[T.erase n] > P[T]) := sorry

end increase_prob_n_7_l129_129284


namespace optimal_strategy_probability_l129_129922

theorem optimal_strategy_probability :
  (probability_of_winning = 21.77%) :=
by
  /- Definitions -/
  let initial_probability_case_I : ℚ := 1 / 36
  let initial_probability_case_II : ℚ := 5 / 12
  let initial_probability_case_III : ℚ := 5 / 9
  
  /- Strategy for Case II -/
  let probability_case_II_after_rerolls : ℚ := (1 / 6 + (5 / 6) * (1 / 6))
  
  /- Strategy for Case III -/
  let probability_case_III : ℚ := initial_probability_case_III * (1 / 12)
  
  /- Probability of winning -/
  let probability_of_winning : ℚ :=
    initial_probability_case_I +
    probability_case_II_after_rerolls * initial_probability_case_II +
    probability_case_III * initial_probability_case_III
  
  /- Check if the result is approximately 21.77% -/
  sorry

end optimal_strategy_probability_l129_129922


namespace problem_l129_129593

def p (x y : Int) : Int :=
  if x ≥ 0 ∧ y ≥ 0 then x * y
  else if x < 0 ∧ y < 0 then x - 2 * y
  else if x ≥ 0 ∧ y < 0 then 2 * x + 3 * y
  else if x < 0 ∧ y ≥ 0 then x + 3 * y
  else 3 * x + y

theorem problem : p (p 2 (-3)) (p (-1) 4) = 28 := by
  sorry

end problem_l129_129593


namespace carol_twice_as_cathy_l129_129971

-- Define variables for the number of cars each person owns
variables (C L S Ca x : ℕ)

-- Define conditions based on the problem statement
def lindsey_cars := L = C + 4
def susan_cars := S = Ca - 2
def carol_cars := Ca = 2 * x
def total_cars := C + L + S + Ca = 32
def cathy_cars := C = 5

-- State the theorem to prove
theorem carol_twice_as_cathy : 
  lindsey_cars C L ∧ 
  susan_cars S Ca ∧ 
  carol_cars Ca x ∧ 
  total_cars C L S Ca ∧ 
  cathy_cars C
  → x = 5 :=
by
  sorry

end carol_twice_as_cathy_l129_129971


namespace percent_of_part_l129_129322

variable (Part : ℕ) (Whole : ℕ)

theorem percent_of_part (hPart : Part = 70) (hWhole : Whole = 280) :
  (Part / Whole) * 100 = 25 := by
  sorry

end percent_of_part_l129_129322


namespace valid_triangle_inequality_l129_129040

theorem valid_triangle_inequality (n : ℕ) (h : n = 6) :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c →
  n * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  intros a b c ha hb hc hineq
  have h₁ : n = 6 := h
  simplify_eq [h₁] at hineq
  have h₂ := nat.add_comm a b
  exact sorry

end valid_triangle_inequality_l129_129040


namespace each_boy_makes_14_l129_129297

/-- Proof that each boy makes 14 dollars given the initial conditions and sales scheme. -/
theorem each_boy_makes_14 (victor_shrimp : ℕ)
                          (austin_shrimp : ℕ)
                          (brian_shrimp : ℕ)
                          (total_shrimp : ℕ)
                          (sets_sold : ℕ)
                          (total_earnings : ℕ)
                          (individual_earnings : ℕ)
                          (h1 : victor_shrimp = 26)
                          (h2 : austin_shrimp = victor_shrimp - 8)
                          (h3 : brian_shrimp = (victor_shrimp + austin_shrimp) / 2)
                          (h4 : total_shrimp = victor_shrimp + austin_shrimp + brian_shrimp)
                          (h5 : sets_sold = total_shrimp / 11)
                          (h6 : total_earnings = sets_sold * 7)
                          (h7 : individual_earnings = total_earnings / 3):
  individual_earnings = 14 := 
by
  sorry

end each_boy_makes_14_l129_129297


namespace least_positive_integer_n_l129_129455

noncomputable theory -- Declare that the section does not have to be computable
open Real -- Open the real number space for using trigonometric functions and constants
open Finset -- Open finite sets for utility functions

theorem least_positive_integer_n : 
  ∃ (n : ℕ), 0 < n ∧ 
  (∑ k in (range 59) \ 30, 1 / (sin (Real.ofInt k + 30) * sin (Real.ofInt k + 31))) = 1 / (sin n) 
  ∧ n = 20 :=
by
  sorry

end least_positive_integer_n_l129_129455


namespace dihedral_angle_range_l129_129339

noncomputable def tetrahedral_pyramid_dihedral_angle (k : ℝ) : ℝ :=
  2 * Real.arctan(2 * k * Real.sqrt 3)

theorem dihedral_angle_range (k : ℝ) (h : 0 < k ∧ k < Real.sqrt 3 / 6) :
  ∃ α : ℝ, α = tetrahedral_pyramid_dihedral_angle k ∧ 0 < α ∧ α < 2 * Real.pi :=
by
  use tetrahedral_pyramid_dihedral_angle k
  split
  repeat { sorry }

end dihedral_angle_range_l129_129339


namespace first_bag_brown_mms_l129_129206

theorem first_bag_brown_mms :
  ∀ (x : ℕ),
  (12 + 8 + 8 + 3 + x) / 5 = 8 → x = 9 :=
by
  intros x h
  sorry

end first_bag_brown_mms_l129_129206


namespace count_valid_pairs_l129_129812

theorem count_valid_pairs : 
  (∃ n : ℕ, n = 894 ∧ 
  n = ∑ b in finset.Icc 3 300, (if (∃ a : ℝ, 0 < a ∧ (log b a)^2018 = log b (a^2018)) then 1 else 0)) :=
begin
  sorry
end

end count_valid_pairs_l129_129812


namespace find_boys_l129_129893

-- Variable declarations
variables (B G : ℕ)

-- Conditions
def total_students (B G : ℕ) : Prop := B + G = 466
def more_girls_than_boys (B G : ℕ) : Prop := G = B + 212

-- Proof statement: Prove B = 127 given both conditions
theorem find_boys (h1 : total_students B G) (h2 : more_girls_than_boys B G) : B = 127 :=
sorry

end find_boys_l129_129893


namespace plate_weight_is_12_l129_129188

-- Definitions of the conditions
def silverware_weight_per_piece := 4
def pieces_of_silverware_per_setting := 3
def plates_per_setting := 2
def tables := 15
def settings_per_table := 8
def backup_settings := 20
def total_weight_of_all_settings := 5040

-- Statement of the problem
theorem plate_weight_is_12 :
  let total_settings := (tables * settings_per_table) + backup_settings in
  let weight_of_silverware_per_setting := pieces_of_silverware_per_setting * silverware_weight_per_piece in
  let total_weight_of_silverware := total_settings * weight_of_silverware_per_setting in
  let total_weight_of_plates := total_weight_of_all_settings - total_weight_of_silverware in
  let total_number_of_plates := total_settings * plates_per_setting in
  let plate_weight := total_weight_of_plates / total_number_of_plates in
  plate_weight = 12 :=
begin
  sorry
end

end plate_weight_is_12_l129_129188


namespace ways_to_write_1800_as_sum_of_4s_and_5s_l129_129530

theorem ways_to_write_1800_as_sum_of_4s_and_5s : 
  ∃ S : Finset (ℕ × ℕ), S.card = 91 ∧ ∀ (nm : ℕ × ℕ), nm ∈ S ↔ 4 * nm.1 + 5 * nm.2 = 1800 ∧ nm.1 ≥ 0 ∧ nm.2 ≥ 0 :=
by
  sorry

end ways_to_write_1800_as_sum_of_4s_and_5s_l129_129530


namespace identifyIncorrectGuess_l129_129683

-- Define the colors of the bears
inductive BearColor
| white
| brown
| black

-- Conditions as defined in the problem statement
def isValidBearRow (bears : Fin 1000 → BearColor) : Prop :=
  ∀ (i : Fin 998), 
    (bears i = BearColor.white ∨ bears i = BearColor.brown ∨ bears i = BearColor.black) ∧
    (bears ⟨i + 1, by linarith⟩ = BearColor.white ∨ bears ⟨i + 1, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 1, by linarith⟩ = BearColor.black) ∧
    (bears ⟨i + 2, by linarith⟩ = BearColor.white ∨ bears ⟨i + 2, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 2, by linarith⟩ = BearColor.black)

-- Iskander's guesses
def iskanderGuesses (bears : Fin 1000 → BearColor) : Prop :=
  bears 1 = BearColor.white ∧
  bears 19 = BearColor.brown ∧
  bears 399 = BearColor.black ∧
  bears 599 = BearColor.brown ∧
  bears 799 = BearColor.white

-- Exactly one guess is incorrect
def oneIncorrectGuess (bears : Fin 1000 → BearColor) : Prop :=
  ∃ (idx : Fin 5), 
    ¬iskanderGuesses bears ∧
    ∀ (j : Fin 5), (j ≠ idx → (bearGuessesIdx j bears = true))

-- The proof problem
theorem identifyIncorrectGuess (bears : Fin 1000 → BearColor) :
  isValidBearRow bears → iskanderGuesses bears → oneIncorrectGuess bears := sorry

end identifyIncorrectGuess_l129_129683


namespace isosceles_triangle_base_length_l129_129244

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l129_129244


namespace calculate_total_parts_l129_129057

theorem calculate_total_parts (sample_size : ℕ) (draw_probability : ℚ) (N : ℕ) 
  (h_sample_size : sample_size = 30) 
  (h_draw_probability : draw_probability = 0.25) 
  (h_relation : sample_size = N * draw_probability) : 
  N = 120 :=
by
  rw [h_sample_size, h_draw_probability] at h_relation
  sorry

end calculate_total_parts_l129_129057


namespace ellipse_equation_existence_of_fixed_point_l129_129100

-- Conditions
def line_l (x : ℝ) : ℝ := x + 1
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 3 / 2
def minor_axis_eq_chord (a b x y : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∃ h_chord : ∀ x y, 1 = 1, true -- Needs actual definition based on conditions provided

def eccentricity_e : ℝ := sqrt 2 / 2

-- To prove
theorem ellipse_equation (a b : ℝ) (h_ab : a > b ∧ b > 0)
    (h_eccen : (a^2 - b^2) / a^2 = 1 / 2) : (b = 1) → (a^2 = 2) →
    (∀ x y, minor_axis_eq_chord a b x y h_ab) → (∃ x y, x^2 / 2 + y^2 = 1) :=
by
  intros
  -- Details defined
  sorry

-- Second part of the problem
theorem existence_of_fixed_point : ∃ T : ℝ × ℝ, T = (0, 1) :=
by
  -- Details defined
  sorry

end ellipse_equation_existence_of_fixed_point_l129_129100


namespace count_valid_four_digit_numbers_l129_129252

theorem count_valid_four_digit_numbers : 
  let valid_first_digits := (4*5 + 4*4)
  let valid_last_digits := (5*5 + 4*4)
  valid_first_digits * valid_last_digits = 1476 :=
by
  sorry

end count_valid_four_digit_numbers_l129_129252


namespace range_of_a_l129_129125

noncomputable def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x : ℝ, f(x) = a * x^2 + b * x + c

theorem range_of_a (f : ℝ → ℝ) (h_quad : quadratic_function f) (symm : ∀ x : ℝ, f(2 + x) = f(2 - x))
  (ineq : f(1) < f(0) ∧ f(0) ≤ f(a)) : a ≤ 0 ∨ a ≥ 4 :=
sorry

end range_of_a_l129_129125


namespace original_employee_count_l129_129774

theorem original_employee_count (employees_operations : ℝ) 
                                (employees_sales : ℝ) 
                                (employees_finance : ℝ) 
                                (employees_hr : ℝ) 
                                (employees_it : ℝ) 
                                (h1 : employees_operations / 0.82 = 192)
                                (h2 : employees_sales / 0.75 = 135)
                                (h3 : employees_finance / 0.85 = 123)
                                (h4 : employees_hr / 0.88 = 66)
                                (h5 : employees_it / 0.90 = 90) : 
                                employees_operations + employees_sales + employees_finance + employees_hr + employees_it = 734 :=
sorry

end original_employee_count_l129_129774


namespace directrix_of_parabola_l129_129447

noncomputable def parabola_directrix (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / 16

theorem directrix_of_parabola :
  let d := parabola_directrix y in d = -(1 / 2) := sorry

end directrix_of_parabola_l129_129447


namespace quadratic_positivity_range_l129_129068

variable (a : ℝ)

def quadratic_function (x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 3

theorem quadratic_positivity_range :
  (∀ x, 0 < x ∧ x < 3 → quadratic_function a x > 0)
  ↔ (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3) := sorry

end quadratic_positivity_range_l129_129068


namespace semicircle_perimeter_is_71_98_l129_129332

noncomputable def semicircle_perimeter_approx (r : ℝ) (π_approx : ℝ) : ℝ := 
  (π_approx * r) + (2 * r)

theorem semicircle_perimeter_is_71_98 :
  semicircle_perimeter_approx 14 3.14159 ≈ 71.98 := 
by 
  sorry

end semicircle_perimeter_is_71_98_l129_129332


namespace length_GH_l129_129632

def length_AB : ℕ := 11
def length_FE : ℕ := 13
def length_CD : ℕ := 5

theorem length_GH : length_AB + length_CD + length_FE = 29 :=
by
  refine rfl -- This will unroll the constants and perform arithmetic

end length_GH_l129_129632


namespace incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129654

def is_color (col : String) (pos : Nat) : Prop := sorry
def valid_guesses : Prop :=
  (is_color "white" 2) ∧
  (is_color "brown" 20) ∧
  (is_color "black" 400) ∧
  (is_color "brown" 600) ∧
  (is_color "white" 800)

theorem incorrect_guess_at_20 :
  (∃ x, (x ∈ [2, 20, 400, 600, 800]) ∧ ¬ is_color_correct x) :=
begin
  sorry -- proof is not required
end

/-- Main theorem to identify the incorrect guess position. -/
theorem Iskander_incorrect_guess_20 :
  valid_guesses →
  (∃! x ∈ [2, 20, 400, 600, 800], ¬ is_color_correct x) →
  ¬ is_color "brown" 20 :=
begin
  admit -- proof is not required
end

end incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129654


namespace total_scissors_l129_129938

/-- In the supply room, there are initially 39 scissors and 22 pencils in one drawer, 
    and 54 pencils and 27 scissors in another drawer. Dan placed 13 new scissors into the first drawer 
    and 7 new scissors into the second drawer. Meanwhile, an art project required 17 pencils to be 
    removed from the first drawer and 22 pencils to be removed from the second drawer. -/
theorem total_scissors : 39 + 13 + (27 + 7) = 86 := 
by 
  calc
    (39 + 13) + (27 + 7) = 52 + 34 : by sorry
                        ... = 86 : by sorry

end total_scissors_l129_129938


namespace greatest_integer_condition_l129_129173

noncomputable def x : ℝ := (finset.sum (finset.range 60) (λ n, real.cos ((↑n + 1 : ℕ) * real.pi / 180))) / 
                           (finset.sum (finset.range 60) (λ n, real.sin ((↑n + 1 : ℕ) * real.pi / 180)))

theorem greatest_integer_condition : floor (100 * x) = 160 :=
sorry

end greatest_integer_condition_l129_129173


namespace exists_n_inequality_l129_129166

-- Define the Euler's totient function φ
def euler_totient : ℕ → ℕ
| 0       := 0
| (n + 1) := (n + 1).natAbs.coprime_count.succ

theorem exists_n_inequality :
  ∃ n : ℕ, euler_totient (2 * n - 1) + euler_totient (2 * n + 1) < euler_totient (2 * n) / 1000 :=
sorry

end exists_n_inequality_l129_129166


namespace c_plus_d_equality_l129_129900

-- Define the digits
def is_digit (n : ℕ) := n < 10

-- Define the conditions
variables (c d : ℕ)
variables (mult_result : ℕ)
noncomputable theory

-- Define the main condition of the problem
def condition_1 : Prop := is_digit c ∧ is_digit d ∧ mult_result = 100 * (3 * 10 + c) * (d * 10 + 4)

-- Define the Lean statement that needs to be proved
theorem c_plus_d_equality (hc : is_digit c) (hd : is_digit d) (hm : mult_result = 3 * c * (d4)) : c + d = 5 :=
sorry

end c_plus_d_equality_l129_129900


namespace lindy_running_speed_l129_129156

-- Definitions of the given conditions
def initial_distance : ℝ := 150
def jack_speed : ℝ := 7
def christina_speed : ℝ := 8
def lindy_total_distance : ℝ := 100

-- The time taken for Jack and Christina to meet
def meeting_time := initial_distance / (jack_speed + christina_speed)

-- Problem statement: Lindy's speed in feet per second
theorem lindy_running_speed : (lindy_total_distance / meeting_time) = 10 := 
by 
  -- This is where the problem would be proven
  sorry

end lindy_running_speed_l129_129156


namespace chairs_bought_l129_129300

theorem chairs_bought (C : ℕ) (tables chairs total_time time_per_furniture : ℕ)
  (h1 : tables = 4)
  (h2 : time_per_furniture = 6)
  (h3 : total_time = 48)
  (h4 : total_time = time_per_furniture * (tables + chairs)) :
  C = 4 :=
by
  -- proof steps are omitted
  sorry

end chairs_bought_l129_129300


namespace compare_abc_l129_129957

theorem compare_abc :
  let a := (4 - Real.log 4) / Real.exp 2
  let b := Real.log 2 / 2
  let c := 1 / Real.exp 1 in
  b < a ∧ a < c := 
sorry

end compare_abc_l129_129957


namespace angle_equality_of_geometry_l129_129712

open_locale classical

-- Define the geometric entities
variables (A B M N C P Q : Type) [IncidenceGeometry A B M N] [Parallelogram M A N C] [SegmentDivision B N P] [SegmentDivision M C Q]

-- Note: The specific properties regarding intersecting circles, chord tangency, and segment division need to be wrapped into corresponding typeclasses.
--       This is a broad abstraction. Definitions for each concept (chords being tangents, the specific parallelogram structure, etc.)
--       would typically be formalized within a broader formal system of Euclidean geometry in Lean.

-- Stating the main theorem
theorem angle_equality_of_geometry : ∠(A, P, Q) = ∠(A, N, C) :=
sorry -- Proof body

end angle_equality_of_geometry_l129_129712


namespace triangle_perimeter_correct_l129_129566

open Real

noncomputable def triangle_perimeter (A B C : ℝ) : ℝ := A + B + C

theorem triangle_perimeter_correct (A B C : ℝ) (h₁ : ∠C = 90) (h₂ : sin A = 1/2) (h₃ : AB = 2) :
  triangle_perimeter A B C = 3 + sqrt 3 := 
begin
  sorry
end

end triangle_perimeter_correct_l129_129566


namespace find_K_l129_129936

theorem find_K (K m n : ℝ) (p : ℝ) (hp : p = 0.3333333333333333)
  (eq1 : m = K * n + 5)
  (eq2 : m + 2 = K * (n + p) + 5) : 
  K = 6 := 
by
  sorry

end find_K_l129_129936


namespace apple_lovers_l129_129351

theorem apple_lovers :
  ∃ (x y : ℕ), 22 * x = 1430 ∧ 13 * (x + y) = 1430 ∧ y = 45 :=
by
  sorry

end apple_lovers_l129_129351


namespace sin_cos_roots_eqn_l129_129851

theorem sin_cos_roots_eqn (θ : ℝ) (m : ℝ) :
  (∀ x : ℝ, (4 * x^2 + 2 * m * x + m = 0) ↔ (x = Float.sin θ ∨ x = Float.cos θ)) ∧
  ((2 * m)^2 - 16 * m ≥ 0) →
  m = 1 - Real.sqrt 5 :=
by
  sorry

end sin_cos_roots_eqn_l129_129851


namespace directrix_eq_l129_129454

noncomputable def parabola_eq : (ℝ → ℝ) := λ x, (x^2 - 8 * x + 12) / 16

theorem directrix_eq : ∀ (y : ℝ), y = parabola_eq (x : ℝ) → ∃ d, d = -1 / 2 := by
  sorry

end directrix_eq_l129_129454


namespace find_wrong_guess_l129_129688

-- Define the three colors as an inductive type.
inductive Color
| white
| brown
| black

-- Define the bears as a list of colors.
def bears (n : ℕ) : Type := list Color

-- Define the conditions: 
-- There are 1000 bears and each tuple of 3 consecutive bears has all three colors.
def valid_bears (b : bears 1000) : Prop :=
  ∀ i : ℕ, i + 2 < 1000 → 
    ∃ c1 c2 c3 : Color, 
      c1 ∈ b.nth i ∧ c2 ∈ b.nth (i+1) ∧ c3 ∈ b.nth (i+2) ∧ 
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3

-- Define Iskander's guesses.
def guesses (b : bears 1000) : Prop :=
  b.nth 1 = some Color.white ∧
  b.nth 19 = some Color.brown ∧
  b.nth 399 = some Color.black ∧
  b.nth 599 = some Color.brown ∧
  b.nth 799 = some Color.white

-- Prove that exactly one of Iskander's guesses is wrong.
def wrong_guess (b : bears 1000) : Prop :=
  (b.nth 19 ≠ some Color.brown) ∧
  valid_bears b ∧
  guesses b →
  ∃ i, i ∈ {1, 19, 399, 599, 799} ∧ (b.nth i ≠ some Color.white ∧ b.nth i ≠ some Color.brown ∧ b.nth i ≠ some Color.black)

theorem find_wrong_guess : 
  ∀ b : bears 1000, 
  valid_bears b → guesses b → wrong_guess b :=
  by
  intros b vb gs
  sorry

end find_wrong_guess_l129_129688


namespace total_shells_is_correct_l129_129186

def morning_shells : Nat := 292
def afternoon_shells : Nat := 324
def total_shells : Nat := morning_shells + afternoon_shells

theorem total_shells_is_correct : total_shells = 616 :=
by
  sorry

end total_shells_is_correct_l129_129186


namespace isosceles_triangle_base_length_l129_129240

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l129_129240


namespace wrong_guess_is_20_l129_129667

-- Define the colors
inductive Color
| white
| brown
| black

-- Assume we have a sequence of 1000 bears
def bears : fin 1000 → Color := sorry

-- Hypotheses
axiom colors_per_three : ∀ (i : fin 998), 
  ({bears i, bears (i + 1), bears (i + 2)} = {Color.white, Color.brown, Color.black} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.black, Color.white, Color.brown} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.brown, Color.black, Color.white})

axiom exactly_one_wrong : 
  (bears 1 = Color.white ∧ bears 19 ≠ Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 ≠ Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 ≠ Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 ≠ Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 ≠ Color.white)

-- Define the theorem to prove
theorem wrong_guess_is_20 : 
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) →
  ¬(bears 19 = Color.brown) := 
sorry

end wrong_guess_is_20_l129_129667


namespace non_hot_peppers_count_l129_129160

-- Define the number of peppers Joel picks each day
def peppers_sunday : ℕ := 7
def peppers_monday : ℕ := 12
def peppers_tuesday : ℕ := 14
def peppers_wednesday : ℕ := 12
def peppers_thursday : ℕ := 5
def peppers_friday : ℕ := 18
def peppers_saturday : ℕ := 12

-- Define the fraction of hot peppers
def fraction_hot_peppers : ℚ := 0.20

-- Define the total number of peppers
def total_peppers : ℕ := 
  peppers_sunday + peppers_monday + peppers_tuesday + 
  peppers_wednesday + peppers_thursday + peppers_friday + peppers_saturday

-- Prove that the number of non-hot peppers picked by Joel is 64
theorem non_hot_peppers_count : (total_peppers * (1 - fraction_hot_peppers)).toInt = 64 := by
  sorry

end non_hot_peppers_count_l129_129160


namespace directrix_of_parabola_l129_129442

-- Define the parabola function
def parabola (x : ℝ) : ℝ := (x^2 - 8*x + 12) / 16

theorem directrix_of_parabola : 
  ∃ y : ℝ, (∀ x : ℝ, parabola x = y) → y = -17 / 4 := 
by
  sorry

end directrix_of_parabola_l129_129442


namespace rotation_sum_l129_129708

-- Defining the vertices of triangles ABC and A'B'C'
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 0, y := 12 }
def C : Point := { x := 16, y := 0 }
def A' : Point := { x := 24, y := 18 }
def B' : Point := { x := 36, y := 18 }
def C' : Point := { x := 24, y := 2 }

-- Defining the problem
theorem rotation_sum (m : ℝ) (x y : ℝ) (h1 : 0 < m ∧ m < 180) 
                  (h2 : is_rotation m (x, y) (A, B, C) (A', B', C')) : 
  m + x + y = 108 :=
sorry

end rotation_sum_l129_129708


namespace last_digit_sum_l129_129721

theorem last_digit_sum :
  (2^2 % 10 + 20^20 % 10 + 200^200 % 10 + 2006^2006 % 10) % 10 = 0 := 
by
  sorry

end last_digit_sum_l129_129721


namespace exists_m_for_all_l_l129_129058

def transform (s : String) : String :=
s.fold "" (λ acc c =>
  match c with
  | '0' => acc ++ "1"
  | '1' => acc ++ "100"
  | _=> acc
)

def A : ℕ → String
| 0 => "1"
| n + 1 => transform (A n)

def a (n l : ℕ) : Char :=
(A n).get ⟨l, by sorry⟩

theorem exists_m_for_all_l (m : ℕ := 6) : 
  ∃ m, ∀ l > 0, ∃ k, k ≤ m ∧
    (∀ i < 2017, a l (i + 1) = a 0 (i + 1)) := 
by 
  use m 
  sorry

end exists_m_for_all_l_l129_129058


namespace find_price_of_thermometer_l129_129713

open Nat

def price_thermometer {T : ℝ} : Prop :=
  (1200 - 60 * 6 = 840) → 
  (60 * 7 = 420) →
  (420 * T = 840) →
  (T = 2)

theorem find_price_of_thermometer : ∃ T : ℝ, price_thermometer :=
begin
  use 2,
  unfold price_thermometer,
  intros h1 h2 h3,
  exact sorry,
end

end find_price_of_thermometer_l129_129713


namespace max_length_MN_l129_129612

noncomputable def C1 := { p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 + 8 * p.2 - 8 = 0 }
noncomputable def C2 := { p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.1 - 5 = 0 }

theorem max_length_MN :
  let center_C1 := (-1 : ℝ, -4 : ℝ),
      radius_C1 := 5,
      center_C2 := (2 : ℝ, 0 : ℝ),
      radius_C2 := 3,
      distance_centers := real.sqrt ( (-1 - 2)^2 + (-4 - 0)^2 ) in
  radius_C1 + distance_centers + radius_C2 = 13 := by
  let center_C1 := (-1 : ℝ, -4 : ℝ)
  let radius_C1 := 5
  let center_C2 := (2 : ℝ, 0 : ℝ)
  let radius_C2 := 3
  let distance_centers := real.sqrt ( (-1 - 2)^2 + (-4 - 0)^2 )
  show radius_C1 + distance_centers + radius_C2 = 13
  sorry

end max_length_MN_l129_129612


namespace incorrect_guess_l129_129698

-- Define the conditions
def bears : ℕ := 1000

inductive Color
| White
| Brown
| Black

constant bear_color : ℕ → Color -- The color of the bear at each position

axiom condition : ∀ n : ℕ, n < bears - 2 → 
  ∃ i j k, (i, j, k ∈ {Color.White, Color.Brown, Color.Black}) ∧ 
  (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ 
  (bear_color n = i ∧ bear_color (n+1) = j ∧ bear_color (n+2) = k) 

constants (g1 : bear_color 2 = Color.White)
          (g2 : bear_color 20 = Color.Brown)
          (g3 : bear_color 400 = Color.Black)
          (g4 : bear_color 600 = Color.Brown)
          (g5 : bear_color 800 = Color.White)

-- The proof problem
theorem incorrect_guess : bear_color 20 ≠ Color.Brown :=
by sorry

end incorrect_guess_l129_129698


namespace directrix_of_parabola_l129_129441

-- Define the parabola function
def parabola (x : ℝ) : ℝ := (x^2 - 8*x + 12) / 16

theorem directrix_of_parabola : 
  ∃ y : ℝ, (∀ x : ℝ, parabola x = y) → y = -17 / 4 := 
by
  sorry

end directrix_of_parabola_l129_129441


namespace line_above_curve_l129_129866

def curve (x : ℝ) : ℝ := (1 / 3) * x^3 - x^2 - 4 * x + 1
def line (x k : ℝ) : ℝ := -x - 2 * k + 1

theorem line_above_curve (k : ℝ) : (∀ x ∈ set.Icc (-3 : ℝ) 3, line x k > curve x) ↔ k < -5 / 6 :=
begin
  sorry

end line_above_curve_l129_129866


namespace sponsorship_prob_zero_sponsorship_prob_gt_150k_l129_129770

noncomputable def prob_sponsorship_amount_zero
  (supports : Fin 3 → ℕ → ℕ → Prop)
  (prob : supports.has_probability (supports _ _ (1/2))) : ℙ :=
begin
  let empty_support := λ (student_supports : Fin 3 → ℕ), ∑ i, student_supports i = 0,
  exact
    @independent_product_prob _ _ _ supports (by_probability) (empty_support) sorry
end

noncomputable def prob_sponsorship_amount_gt_150k
  (supports : Fin 3 → ℕ → ℕ → Prop)
  (prob : supports.has_probability (supports _ _ (1/2))) : ℙ :=
begin
  let excess_support := λ (student_supports : Fin 3 → ℕ), ∑ i, student_supports i > 150000,
  exact
    @independent_product_prob _ _ _ supports (by_probability) (excess_support) sorry
end

theorem sponsorship_prob_zero :
  prob_sponsorship_amount_zero = 1 / 64 :=
sorry

theorem sponsorship_prob_gt_150k :
  prob_sponsorship_amount_gt_150k = 11 / 32 :=
sorry

end sponsorship_prob_zero_sponsorship_prob_gt_150k_l129_129770


namespace prove_distance_increase_l129_129978

noncomputable def distance_increase_after_compression
  (e1 e2 : ℝ) -- unit vectors along the lines
  (φ : ℝ) -- the acute angle between the vectors e1 and e2
  (λ μ : ℝ) -- scaling factors
  (h : cos(φ) > 3 / (4 * λ)) -- condition for the angle and scaling factor
  : Prop :=
let original_vector := λ * e1 + μ * e2,
    transformed_vector := λ * e1 + (μ / 2) * e2,
    original_length := sqrt(λ^2 + μ^2 + 2 * λ * μ * cos(φ)),
    transformed_length := sqrt(λ^2 + (μ / 2)^2 + λ * (μ / 2) * cos(φ))
in transformed_length > original_length

theorem prove_distance_increase
  (e1 e2 : ℝ) -- unit vectors along the lines
  (φ : ℝ) -- the acute angle between the vectors e1 and e2
  (λ μ : ℝ) -- scaling factors
  (h : cos(φ) > 3 / (4 * λ)) -- condition for the angle and scaling factor
  : distance_increase_after_compression e1 e2 φ λ μ h :=
begin
  sorry
end

end prove_distance_increase_l129_129978


namespace even_function_implies_a_zero_l129_129909

theorem even_function_implies_a_zero (a : ℝ) : 
  (∀ x : ℝ, (λ x, x^2 - |x + a|) (-x) = (λ x, x^2 - |x + a|) (x)) → a = 0 :=
by
  sorry

end even_function_implies_a_zero_l129_129909


namespace sum_of_integral_c_l129_129825

theorem sum_of_integral_c :
  let discriminant (a b c : ℤ) := b * b - 4 * a * c
  ∃ (valid_c : List ℤ),
    (∀ c ∈ valid_c, c ≤ 30 ∧ ∃ k : ℤ, discriminant 1 (-9) (c) = k * k ∧ k > 0) ∧
    valid_c.sum = 32 := 
by
  sorry

end sum_of_integral_c_l129_129825


namespace simplify_trig_expression_l129_129619

theorem simplify_trig_expression (x : ℝ) (h1 : cos x ≠ 0) (h2 : 1 + sin x ≠ 0) : 
  (cos x / (1 + sin x)) + ((1 + sin x) / cos x) = 2 * (1 / cos x) :=
by
  sorry

end simplify_trig_expression_l129_129619


namespace triangle_inequality_for_n6_l129_129033

variables {a b c : ℝ} {n : ℕ}
open Real

-- Define the main statement as a theorem
theorem triangle_inequality_for_n6 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_for_n6_l129_129033


namespace part1_part2_l129_129886

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : a * sin A * sin B + b * cos A^2 = 4 / 3 * a)
variable (h2 : c^2 = a^2 + (1 / 4) * b^2)

theorem part1 : b = 4 / 3 * a := by sorry

theorem part2 : C = π / 3 := by sorry

end part1_part2_l129_129886


namespace range_of_m_l129_129899

theorem range_of_m (x : ℝ) (m : ℝ) 
  (h : sqrt 3 * sin x + cos x = 4 - m) : 
  2 ≤ m ∧ m ≤ 6 :=
sorry

end range_of_m_l129_129899


namespace volume_of_stacked_spheres_l129_129974

def volume_sphere (r : ℝ) : ℝ :=
  (4/3) * real.pi * r^3

def total_volume_clay (radii : list ℝ) : ℝ :=
  radii.map volume_sphere |> list.sum

theorem volume_of_stacked_spheres :
  total_volume_clay [1, 4, 6, 3] = (1232/3) * real.pi :=
by
  sorry

end volume_of_stacked_spheres_l129_129974


namespace monkey_slips_2_feet_each_hour_l129_129363

/-- 
  A monkey climbs a 17 ft tree, hopping 3 ft and slipping back a certain distance each hour.
  The monkey takes 15 hours to reach the top. Prove that the monkey slips back 2 feet each hour.
-/
def monkey_slips_back_distance (s : ℝ) : Prop :=
  ∃ s : ℝ, (14 * (3 - s) + 3 = 17) ∧ s = 2

theorem monkey_slips_2_feet_each_hour : monkey_slips_back_distance 2 := by
  -- Sorry, proof omitted
  sorry

end monkey_slips_2_feet_each_hour_l129_129363


namespace tangent_perpendicular_compare_PQR_l129_129861

noncomputable def f (x : ℝ) : ℝ := Real.ln x

noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem tangent_perpendicular :
  ∀ (k1 k2 : ℝ), 
    (∃ x2 : ℝ, x2 ≠ 0 ∧ k1 = 1 / x2 ∧ f (-x2) = k1 * (-x2)) → 
    (∃ x1 : ℝ, k2 = g x1 ∧ g x1 = k2 * x1) → k1 * k2 = -1 :=
by sorry

theorem compare_PQR (a b : ℝ) (h : a ≠ b) :
  let P := g ((a + b) / 2)
  let Q := (g a - g b) / (a - b)
  let R := (g a + g b) / 2
  in P < Q ∧ Q < R :=
by sorry

end tangent_perpendicular_compare_PQR_l129_129861


namespace harry_book_pages_correct_l129_129210

-- Define the total pages in Selena's book.
def selena_book_pages : ℕ := 400

-- Define Harry's book pages as 20 fewer than half of Selena's book pages.
def harry_book_pages : ℕ := (selena_book_pages / 2) - 20

-- The theorem to prove the number of pages in Harry's book.
theorem harry_book_pages_correct : harry_book_pages = 180 := by
  sorry

end harry_book_pages_correct_l129_129210


namespace area_of_circle_eq_sixteen_pi_l129_129305

theorem area_of_circle_eq_sixteen_pi :
  ∃ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) ↔ (π * 4^2 = 16 * π) :=
by
  sorry

end area_of_circle_eq_sixteen_pi_l129_129305


namespace exists_real_x_l129_129573

noncomputable def fractional_part (y : ℝ) : ℝ := y - y.floor

theorem exists_real_x (m n : ℕ) (h : m ≠ n) (hm : 0 < m) (hn : 0 < n) :
  ∃ x : ℝ, (1 / 3) ≤ fractional_part (x * n) ∧ fractional_part (x * n) ≤ (2 / 3) ∧
           (1 / 3) ≤ fractional_part (x * m) ∧ fractional_part (x * m) ≤ (2 / 3) :=
by
  sorry

end exists_real_x_l129_129573


namespace sum_of_powers_of_three_is_1729_l129_129703

theorem sum_of_powers_of_three_is_1729 :
  ∃ (s : ℕ) (m : Fin s → ℕ) (b : Fin s → ℤ),
    (∀ i j : Fin s, i ≠ j → m i ≠ m j) ∧ -- uniqueness condition on m
    (∀ i, b i = 1 ∨ b i = -1) ∧            -- b_k are either 1 or -1
    (∑ i : Fin s, b i * 3 ^ m i = 1729) →  -- the sum equals 1729
    (∑ i : Fin s, m i) = 18 :=             -- sum of the indices equals 18
by
  sorry  -- proof to be filled

end sum_of_powers_of_three_is_1729_l129_129703


namespace distinct_terms_of_expansion_l129_129528

theorem distinct_terms_of_expansion :
  ∀ (a b c d e f g h i : Type),
  (a * e = b * f) →
  (count_distinct_terms ((a + b + c + d) * (e + f + g + h + i))) = 19 :=
by
  sorry

end distinct_terms_of_expansion_l129_129528


namespace longer_trip_due_to_red_lights_l129_129779

theorem longer_trip_due_to_red_lights :
  ∀ (num_lights : ℕ) (green_time first_route_base_time red_time_per_light second_route_time : ℕ),
  num_lights = 3 →
  first_route_base_time = 10 →
  red_time_per_light = 3 →
  second_route_time = 14 →
  (first_route_base_time + num_lights * red_time_per_light) - second_route_time = 5 :=
by
  intros num_lights green_time first_route_base_time red_time_per_light second_route_time
  sorry

end longer_trip_due_to_red_lights_l129_129779


namespace same_solution_of_equations_l129_129835

theorem same_solution_of_equations : ∃ k : ℝ, k = 17 ∧
  ∀ x : ℝ, (2 * x + 4 = 4 * (x - 2)) ↔ (-x + k = 2 * x - 1) :=
by
  use 17
  split
  { exact rfl }
  { intro x
    split
    { intro h1
      have h1' := h1
      linarith }
    { intro h2
      have h2' := h2
      linarith } }

end same_solution_of_equations_l129_129835


namespace area_square_eq_sum_rectangles_l129_129138

open_locale real

-- Define the mathematical objects and conditions
variables (A B C D E P Q M N K L : Type)
variables [meas : measurable_space A] [measurable_space B] 
        [measurable_space C] [measurable_space D] 
        [measurable_space E] [measurable_space P] 
        [measurable_space Q] [measurable_space M]
        [measurable_space N] [measurable_space K]
        [measurable_space L]

parameters 
(h_triangle : ∀ (A B C : Type), triangle A B C → Type)
(h_acute    : ∀ (A B C : Type) (t : triangle A B C), acute t)
(h_alt_ad   : ∀ (A B C D: Type) (t : triangle A B C), altitude t A D)
(h_alt_ce   : ∀ (A B C E: Type) (t : triangle A B C), altitude t C E)
(h_sq_acpq  : ∀ (A C P Q : Type), square A C P Q)
(h_rect_cdmn : ∀ (C D M N : Type), rectangle C D M N)
(h_rect_aekl : ∀ (A E K L : Type), rectangle A E K L)
(h_al_eq_ab : ∀ (A L B : Type), congruent A L B)
(h_cn_eq_cb : ∀ (C N B : Type), congruent C N B)

-- Statement of the theorem to prove
theorem area_square_eq_sum_rectangles 
  (A B C D E P Q M N K L: Type)
  (t : triangle A B C)
  (alt_ad : altitude t A D)
  (alt_ce : altitude t C E)
  (sq_acpq : square A C P Q)
  (rect_cdmn : rectangle C D M N)
  (rect_aekl : rectangle A E K L)
  (congr_a_l_eq_a_b : congruent A L A B)
  (congr_c_n_eq_c_b : congruent C N C B)
  : area sq_acpq = area rect_aekl + area rect_cdmn := by
  sorry

end area_square_eq_sum_rectangles_l129_129138


namespace average_defect_l129_129276

-- Define the defect function δ(n)
def defect (n : ℕ) : ℕ :=
  let digits := Int.toNat (n % 10) :: ((toDigits (n / 10)).map Int.toNat)
  let part := digits.partitions 2
  part.map (λ ⟨a, b⟩ => abs (a.sum - b.sum)).min!

-- Theorem stating the average defect
theorem average_defect : 
  (Real.lim (λ n : ℕ, (Real.ofInt (Part.sum (Part.map (λ k : ℕ => defect k) (Part.Fin n))) / n))
  = 1 / 2 :=
sorry

end average_defect_l129_129276


namespace inequality_positives_l129_129854

theorem inequality_positives {n : ℕ} (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_sum : ∑ i, a i = 1) : 
  (∏ i, (1 + 1 / a i)) ≥ (n + 1) ^ n := 
sorry

end inequality_positives_l129_129854


namespace trig_problem_l129_129583

theorem trig_problem
  (a b : ℝ)
  (h1 : sin a / sin b = 4)
  (h2 : cos a / cos b = 1 / 3) :
  (sin (2 * a) / sin (2 * b)) + (cos (2 * a) / cos (2 * b)) = 29 / 381 :=
  sorry

end trig_problem_l129_129583


namespace part1_part2_l129_129943

-- Definition of proposition p
def p (λ : ℝ) : Prop :=
  (λ^2 - 2*λ - 3 = 0)

-- Definition of proposition q
def q (λ : ℝ) : Prop :=
  (λ < 0) ∧ (λ^2 + λ - 6 < 0)

-- Proof statements

-- (1) If proposition p is true, then λ = 3 or λ = -1.
theorem part1 (λ : ℝ) (hp : p λ) : λ = 3 ∨ λ = -1 := by
  sorry

-- (2) If the proposition (¬p ∧ q) is true, then -3 < λ < -1 or -1 < λ < 0.
theorem part2 (λ : ℝ) (hnpq : ¬p λ ∧ q λ) : (-3 < λ ∧ λ < -1) ∨ (-1 < λ ∧ λ < 0) := by
  sorry

end part1_part2_l129_129943


namespace positive_rationals_exponents_integral_l129_129167

theorem positive_rationals_exponents_integral
    (u v : ℚ) (hu : 0 < u) (hv : 0 < v) (huv : u ≠ v)
    (H : ∀ N : ℕ, ∃ n : ℕ, N ≤ n ∧ u^n - v^n ∈ ℤ) :
    u ∈ ℤ ∧ v ∈ ℤ :=
by
  sorry

end positive_rationals_exponents_integral_l129_129167


namespace isosceles_base_length_l129_129229

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l129_129229


namespace find_C_l129_129329

theorem find_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 340) : C = 40 :=
by sorry

end find_C_l129_129329


namespace quadratic_complex_roots_l129_129414

open Complex

theorem quadratic_complex_roots :
  ∃ (a b c : ℂ), a = 1 ∧ b = 6 ∧ c = 13 ∧ 
  (a ≠ 0) ∧ 
  (b^2 - 4 * a * c < 0) ∧ 
  (Quadratic Formula for roots) ∧  
  (x: ℂ) (h: x^2 + 6 * x + 13 = 0), (x = -3 + 2 * Complex.i) ∨ (x = -3 - 2 * Complex.i) :=
by
  sorry

end quadratic_complex_roots_l129_129414


namespace boys_from_other_communities_l129_129335

def total_boys : Nat := 850
def percentage_muslims : Float := 44.0 / 100.0
def percentage_hindus : Float := 14.0 / 100.0
def percentage_sikhs : Float := 10.0 / 100.0

theorem boys_from_other_communities 
: total_boys * (1.0 - (percentage_muslims + percentage_hindus + percentage_sikhs)) = 272 :=
by
  sorry

end boys_from_other_communities_l129_129335


namespace triangle_inequality_condition_l129_129043

theorem triangle_inequality_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (ineq : 6 * (a * b + b * c + c * a) > 5 * (a ^ 2 + b ^ 2 + c ^ 2)) : 
  (a < b + c ∧ b < a + c ∧ c < a + b) :=
sorry

end triangle_inequality_condition_l129_129043


namespace area_reduction_is_correct_l129_129379

-- Define the original area of the equilateral triangle
def original_area := 100 * Real.sqrt 3

-- Define the reduction in side length of the triangle
def side_reduction := 6

-- Calculate the side length of the original equilateral triangle
noncomputable def original_side_length : ℝ := Real.sqrt (4 * original_area / Real.sqrt 3)

-- Define the new side length after reduction
def new_side_length := original_side_length - side_reduction

-- Define the area of an equilateral triangle given its side length
noncomputable def area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

-- Calculate the new area after the side length reduction
noncomputable def new_area : ℝ := area new_side_length

-- The decrease in area of the equilateral triangle
noncomputable def area_decrease : ℝ := original_area - new_area

-- The proof statement showing the decrease in area is 51√3 cm²
theorem area_reduction_is_correct : area_decrease = 51 * Real.sqrt 3 := 
by sorry

end area_reduction_is_correct_l129_129379


namespace find_difference_l129_129958

variable (f : ℝ → ℝ)

-- Conditions
axiom linear_f : ∀ x y a b, f (a * x + b * y) = a * f x + b * f y
axiom f_difference : f 6 - f 2 = 12

theorem find_difference : f 12 - f 2 = 30 :=
by
  sorry

end find_difference_l129_129958


namespace problem_1_problem_2_l129_129521

-- Definitions for the sets A and B:

def set_A : Set ℝ := { x | x^2 - x - 12 ≤ 0 }
def set_B (m : ℝ) : Set ℝ := { x | 2 * m - 1 < x ∧ x < 1 + m }

-- Problem 1: When m = -2, find A ∪ B
theorem problem_1 : set_A ∪ set_B (-2) = { x | -5 < x ∧ x ≤ 4 } :=
sorry

-- Problem 2: If A ∩ B = B, find the range of the real number m
theorem problem_2 : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ m ≥ -1 :=
sorry

end problem_1_problem_2_l129_129521


namespace solve_eq1_solve_eq2_l129_129215

-- Define the problem for equation (1)
theorem solve_eq1 (x : Real) : (x - 1)^2 = 2 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by 
  sorry

-- Define the problem for equation (2)
theorem solve_eq2 (x : Real) : x^2 - 6 * x - 7 = 0 ↔ (x = -1 ∨ x = 7) :=
by 
  sorry

end solve_eq1_solve_eq2_l129_129215


namespace cyclic_quadrilateral_inscribed_circle_l129_129989

/-- Let ABCD be a cyclic quadrilateral (inscribed in a circle). 
Let there be another circle with its center on the side AB, tangent to AD, BC, and CD. 
Prove that AD + BC = AB. --/
theorem cyclic_quadrilateral_inscribed_circle {A B C D M O: Point} 
  (hABCD_inscribed : inscribed_in_circle A B C D)
  (hO_on_AB : O ∈ segment A B)
  (hO_tangent_AD : tangent_to_circle O A D)
  (hO_tangent_BC : tangent_to_circle O B C)
  (hO_tangent_CD : tangent_to_circle O C D) : 
  segment_length A D + segment_length B C = segment_length A B :=
sorry

end cyclic_quadrilateral_inscribed_circle_l129_129989


namespace correct_option_among_sqrt_statements_l129_129746

theorem correct_option_among_sqrt_statements :
  ¬ (sqrt 16 = -4 ∨ sqrt 16 = 4) ∧
  ¬ (sqrt ((-3)^2) = -3) ∧
  (sqrt 81 = 9 ∨ -sqrt 81 = -9) ∧
  ¬ (sqrt (- 4) = 2) ∧
  ( (sqrt 16 = 4 ∨ sqrt 16 = -4) ∧
    (sqrt ((-3)^2) = 3) ∧
    (sqrt 81 = 9 ∨ -sqrt 81 = -9) ∧
    ¬ sqrt (-4)) →  
  true :=
by
  sorry

end correct_option_among_sqrt_statements_l129_129746


namespace arithmetic_sequence_general_term_specific_values_general_term_formula_l129_129088

-- Arithmetic sequence general term
theorem arithmetic_sequence_general_term (d a₁ : ℕ) (h : ∀ n : ℕ, S (n^2) = (S n)^2) :
  (∀ n : ℕ, a n = 1) ∨ (∀ n : ℕ, a n = 2 * n - 1) :=
sorry

-- Specific values a₁ and a₂
theorem specific_values (h : ∀ n : ℕ, S (n + 1) = S n + 2 * S n + 1) :
  a 1 = 1 ∧ a 2 = 3 :=
sorry

-- General term formula for the sequence
theorem general_term_formula (h : ∀ n : ℕ, S (n + 1) = S n + 2 * S n + 1) :
  ∀ n : ℕ, a n = 3 ^ (n - 1) :=
sorry

-- Definitions for the sum notation, assuming S_n is known appropriately
noncomputable def S : ℕ → ℕ := sorry -- Sum of the first n terms
noncomputable def a : ℕ → ℕ := sorry -- Term in the sequence

end arithmetic_sequence_general_term_specific_values_general_term_formula_l129_129088


namespace not_n_attainable_infinitely_many_three_attainable_all_but_seven_l129_129491

def n_admissible (n : ℕ) (seq : ℕ → ℕ) : Prop :=
  (seq 1 = 1) ∧
  (∀ k, seq (2 * (k + 1)) = seq (2 * k + 1) + 2 ∨ seq (2 * (k + 1)) = seq (2 * k + 1) + n) ∧
  (∀ k, seq (2 * k + 1) = 2 * seq (2 * k) ∨ seq (2 * k + 1) = n * seq (2 * k))

def n_even_admissible (n : ℕ) (seq : ℕ → ℕ) : Prop :=
  (seq 1 = 1) ∧
  (∀ k, seq (2 * (k + 1)) = 2 * seq (2 * k + 1) ∨ seq (2 * (k + 1)) = n * seq (2 * k + 1)) ∧
  (∀ k, seq (2 * k + 1) = seq (2 * k) + 2 ∨ seq (2 * k + 1) = seq (2 * k) + n)

def n_attainable (n m : ℕ) (seq : ℕ → ℕ) : Prop :=
  m > 1 ∧ (n_admissible n seq ∨ n_even_admissible n seq)

theorem not_n_attainable_infinitely_many (n : ℕ) (h : n > 8) : 
  ∃ (infinitely_many : ℕ → Prop), (∀ m, infinitely_many m → ¬ n_attainable n m (λ x, x)) :=
sorry

theorem three_attainable_all_but_seven : 
  (∀ m : ℕ, m ≠ 7 → n_attainable 3 m (λ x, x)) :=
sorry

end not_n_attainable_infinitely_many_three_attainable_all_but_seven_l129_129491


namespace max_one_segment_equal_side_l129_129781

theorem max_one_segment_equal_side (A B C M : Point) (hA : M ≠ A)
  (hB : M ≠ B) (hC : M ≠ C) (hM : InsideTriangle M A B C) :
  ∀ {AM BM CM}, 
  max_eq_segment_count AM BM CM A B C M ≤ 1 := 
sorry

end max_one_segment_equal_side_l129_129781


namespace triangle_inequality_condition_l129_129042

theorem triangle_inequality_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (ineq : 6 * (a * b + b * c + c * a) > 5 * (a ^ 2 + b ^ 2 + c ^ 2)) : 
  (a < b + c ∧ b < a + c ∧ c < a + b) :=
sorry

end triangle_inequality_condition_l129_129042


namespace tangent_at_1_min_value_of_a_l129_129513

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / 6) * x^3 + (1 / 2) * x - x * (log x)

theorem tangent_at_1 :
  let f' (x : ℝ) := (1 / 2) * x^2 - log x - (1 / 2)
    in f 1 = 1 / 3 ∧ f' 1 = 0 → 
    ∃ m b, ∀ x, f x = m * x + b :=
by sorry

theorem min_value_of_a :
  (∀ x, (1 / exp 1) < x ∧ x < exp 1 → f x < (1 / 6) * exp 1^3 - (1 / 2) * exp 1) ∧ 
  ∀ y, (1 / exp 1 < y) ∧ (y < exp 1) → f y < (1 / 6) * exp 1^3 - (1 / 2) * exp 1 :=
by sorry

end tangent_at_1_min_value_of_a_l129_129513


namespace no_positive_integer_n_ge_2_1001_n_is_square_of_prime_l129_129833

noncomputable def is_square_of_prime (m : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ m = p * p

theorem no_positive_integer_n_ge_2_1001_n_is_square_of_prime :
  ∀ n : ℕ, n ≥ 2 → ¬ is_square_of_prime (n^3 + 1) :=
by
  intro n hn
  sorry

end no_positive_integer_n_ge_2_1001_n_is_square_of_prime_l129_129833


namespace log_calculation_l129_129393

theorem log_calculation : log 3 (81 * (27^(1/3)) * sqrt 81) = 7 := by
  have h1 : 81 = 3^4 := by sorry
  have h2 : 27 = 3^3 := by sorry
  have h3 : sqrt (81 : ℝ) = 3^2 := by sorry -- (Note: sqrt function requires real numbers)
  have h4 : (27 : ℝ)^(1/3) = 3 := by sorry -- (Note: exponentiation with fractional requires real numbers)
  rw [h1, h2, h3, h4]
  rw [log_mul (by norm_num : 81 * (27^(1/3)) > 0) (by norm_num : (sqrt 81) > 0)]
  rw [log_mul (by norm_num : 81 > 0) (by norm_num : (27^(1/3)) > 0)]
  rw [log_pow, log_pow, log_pow, log_pow]
  rw [log_base_same, log_base_same, log_base_same, log_base_same]
  norm_num

end log_calculation_l129_129393


namespace directrix_of_parabola_l129_129446

def parabola (x : ℝ) : ℝ := (x^2 - 8*x + 12) / 16

theorem directrix_of_parabola :
  ∀ x, parabola x = (x-4)^2 / 16 - 1/4 →
  let a := 1/16 in
  let h := 4 in
  let k := -1/4 in
  let directrix := k - 1/(4*a) in
  directrix = -17/4 :=
by
  intro x h1
  simp only [parabola] at h1
  dsimp [a, h, k] at h1
  have := calc
    k - 1/(4*a) = -1/4 - 4 : by field_simp [a]
    ... = -17/4 : by norm_num
  exact this

end directrix_of_parabola_l129_129446


namespace count_lights_remain_on_after_switching_l129_129278

theorem count_lights_remain_on_after_switching :
  let lights := finset.range 2013
  let switched (k : ℕ) := {n ∈ lights | n % k = 0} 
  let multiples_2 := switched 2
  let multiples_3 := switched 3
  let multiples_5 := switched 5 
  let on_after_switching :=
    finset.filter (λ n, ¬(n ∈ multiples_2) ∧ ¬(n ∈ multiples_3) ∧ ¬(n ∈ multiples_5)) lights
  on_after_switching.card = 1006 := 
sorry

end count_lights_remain_on_after_switching_l129_129278


namespace trapezoids_in_22_gon_l129_129069

theorem trapezoids_in_22_gon (M : Type) [polygon M] (hM : regular M 22) :
  number_of_trapezoids M = 990 :=
sorry

end trapezoids_in_22_gon_l129_129069


namespace volume_intersection_zero_l129_129729

/-- The set of points satisfying |x| + |y| + |z| ≤ 1. -/
def region1 (x y z : ℝ) : Prop :=
  |x| + |y| + |z| ≤ 1

/-- The set of points satisfying |x| + |y| + |z-2| ≤ 1. -/
def region2 (x y z : ℝ) : Prop :=
  |x| + |y| + |z-2| ≤ 1

/-- The intersection of region1 and region2 forms a region with volume 0. -/
theorem volume_intersection_zero : 
  (∫ x y z, (region1 x y z ∧ region2 x y z)) = 0 := sorry

end volume_intersection_zero_l129_129729


namespace unit_vector_bisector_l129_129054

theorem unit_vector_bisector 
  (A B C : EuclideanSpace ℝ (fin 3))
  (hA : A = ![1,1,1]) 
  (hB : B = ![3,0,1]) 
  (hC : C = ![0,3,1]) : 
  ∃ e : Fin₃ → ℝ, e = ![(1/(Real.sqrt 2)), (1/(Real.sqrt 2)), 0] := 
by 
  sorry

end unit_vector_bisector_l129_129054


namespace f_2_is_H_function_f_3_is_H_function_l129_129772

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ < f x₂

def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

noncomputable def f_2 (x : ℝ) : ℝ := 3 * x - 2 * (sin x - cos x)
noncomputable def f_3 (x : ℝ) : ℝ := exp x + 1

theorem f_2_is_H_function : is_H_function f_2 := sorry
theorem f_3_is_H_function : is_H_function f_3 := sorry

end f_2_is_H_function_f_3_is_H_function_l129_129772


namespace sarah_bottle_caps_total_l129_129618

def initial_caps : ℕ := 450
def first_day_caps : ℕ := 175
def second_day_caps : ℕ := 95
def third_day_caps : ℕ := 220
def total_caps : ℕ := 940

theorem sarah_bottle_caps_total : 
    initial_caps + first_day_caps + second_day_caps + third_day_caps = total_caps :=
by
  sorry

end sarah_bottle_caps_total_l129_129618


namespace non_officers_count_l129_129224

theorem non_officers_count 
  (avg_salary_employees : ℝ)
  (avg_salary_officers : ℝ)
  (avg_salary_non_officers : ℝ)
  (num_officers : ℝ)
  (total_employees_avg_salary : ℝ) :
  avg_salary_employees = 120 →
  avg_salary_officers = 420 →
  avg_salary_non_officers = 110 →
  num_officers = 15 →
  ∃ (num_non_officers : ℝ), num_non_officers = 450 :=
by
  intros h1 h2 h3 h4
  use 450
  sorry

end non_officers_count_l129_129224


namespace area_of_triangle_BDE_is_2_l129_129613

noncomputable def points_with_conditions (A B C D E : ℝ^3) : Prop :=
  dist A B = 2 ∧
  dist B C = 2 ∧
  dist C D = 2 ∧
  dist D E = 2 ∧
  dist E A = 2 ∧
  ∠ A B C = π / 2 ∧
  ∠ C D E = π / 2 ∧
  ∠ D E A = π / 2 ∧
  ∀ P : ℝ^3, P ∈ affine_span ℝ (set.of_points {A, B, C}) ↔ P ∈ ⋃ p : ℝ, {p • ⇑(λ (x : ℝ), x • (E - D)) + D}

theorem area_of_triangle_BDE_is_2 :
  ∀ A B C D E : ℝ^3,
  points_with_conditions A B C D E →
  euclidean_volume (convex_hull ℝ (set.of_points {B, D, E})) = 2 :=
by
  intros A B C D E h
  sorry

end area_of_triangle_BDE_is_2_l129_129613


namespace main_theorem_l129_129915

variables {A B C D X K L M : Type*}
variables [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D]
variables [decidable_eq X] [decidable_eq K] [decidable_eq L] [decidable_eq M]

-- Definitions based on given conditions
def is_right_angle (α β γ : Type*) : Prop := α ≠ β ∧ α = β + (π / 2)
def is_foot_of_perpendicular (c d a b : Type*) : Prop := ∃ E, d = (c, foot a b)
def is_point_on_segment (p x1 x2: Type*) : Prop := ∃ λ ∈ (0, 1), p = λ * x1 + (1-λ) * x2
def distance_eq (x y d : Type*) : Prop := ¬(x ≠ d ∧ y = d)
def intersection_of (l1 l2 m : Type*) : Prop := ∃ E, m = intersection l1 l2

-- Given conditions translated into Lean
noncomputable theory
def problem_conditions : Prop :=
  is_right_angle B C A ∧
  is_foot_of_perpendicular C D A B ∧
  is_point_on_segment X C D ∧
  is_point_on_segment K A X ∧
  distance_eq B K B C  ∧
  is_point_on_segment L B X ∧
  distance_eq A L A C ∧
  intersection_of (line A L) (line B K) M

-- The main theorem to be proved that MK = ML
theorem main_theorem (h : problem_conditions) : distance_eq M K M L :=
sorry

end main_theorem_l129_129915


namespace mod_remainder_l129_129722

theorem mod_remainder (a b c x: ℤ):
    a = 9 → b = 5 → c = 3 → x = 7 →
    (a^6 + b^7 + c^8) % x = 4 :=
by
  intros
  sorry

end mod_remainder_l129_129722


namespace drying_time_l129_129155

theorem drying_time
  (time_short : ℕ := 10) -- Time to dry a short-haired dog in minutes
  (time_full : ℕ := time_short * 2) -- Time to dry a full-haired dog in minutes, which is twice as long
  (num_short : ℕ := 6) -- Number of short-haired dogs
  (num_full : ℕ := 9) -- Number of full-haired dogs
  : (time_short * num_short + time_full * num_full) / 60 = 4 := 
by
  sorry

end drying_time_l129_129155


namespace intersection_point_of_lines_distance_between_points_l129_129814

def parametric_eq_x (t : ℝ) := 1 + t
def parametric_eq_y (t : ℝ) := -5 + (Real.sqrt 3) * t

def line_eq (x y : ℝ) := x - y - 2 * (Real.sqrt 3) = 0

def point_Q := (1, -5 : ℝ × ℝ)

noncomputable def point_P : ℝ × ℝ := 
    let t := 4 * (3 - (Real.sqrt 3)) in
    (parametric_eq_x t, parametric_eq_y t)

theorem intersection_point_of_lines :
    let P := point_P in
    let Q := point_Q in
    let xP := P.1 in
    let yP := P.2 in
    line_eq xP yP ∧
    P = (13 - 4 * (Real.sqrt 3), -17 + 12 * (Real.sqrt 3)) :=
by
    sorry

theorem distance_between_points :
    let P := point_P in
    let Q := point_Q in
    Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) = 12 * Real.sqrt 3 :=
by
    sorry

end intersection_point_of_lines_distance_between_points_l129_129814


namespace find_average_speed_l129_129766

theorem find_average_speed :
  ∃ v : ℝ, (880 / v) - (880 / (v + 10)) = 2 ∧ v = 61.5 :=
by
  sorry

end find_average_speed_l129_129766


namespace sam_gave_mary_29_new_cards_l129_129599

variable (initial_cards torn_cards final_cards : ℕ)
variable (new_cards_given : ℕ)

-- Given conditions
def mary_initial_cards := 33
def torn_cards := 6
def mary_final_cards := 56

-- Calculations based on the conditions
def mary_usable_cards := mary_initial_cards - torn_cards
def new_cards_given := mary_final_cards - mary_usable_cards

-- Statement to prove
theorem sam_gave_mary_29_new_cards :
  new_cards_given = 29 := by 
  sorry

end sam_gave_mary_29_new_cards_l129_129599


namespace quadrilateral_with_rhombus_midpoints_has_equal_diagonals_l129_129913

theorem quadrilateral_with_rhombus_midpoints_has_equal_diagonals
    (Q : Type) [Quad Q]
    (M : MidPoints Q)
    (is_rhombus : Rhombus M) : EqualDiagonals Q := 
sorry

end quadrilateral_with_rhombus_midpoints_has_equal_diagonals_l129_129913


namespace identifyIncorrectGuess_l129_129678

-- Define the colors of the bears
inductive BearColor
| white
| brown
| black

-- Conditions as defined in the problem statement
def isValidBearRow (bears : Fin 1000 → BearColor) : Prop :=
  ∀ (i : Fin 998), 
    (bears i = BearColor.white ∨ bears i = BearColor.brown ∨ bears i = BearColor.black) ∧
    (bears ⟨i + 1, by linarith⟩ = BearColor.white ∨ bears ⟨i + 1, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 1, by linarith⟩ = BearColor.black) ∧
    (bears ⟨i + 2, by linarith⟩ = BearColor.white ∨ bears ⟨i + 2, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 2, by linarith⟩ = BearColor.black)

-- Iskander's guesses
def iskanderGuesses (bears : Fin 1000 → BearColor) : Prop :=
  bears 1 = BearColor.white ∧
  bears 19 = BearColor.brown ∧
  bears 399 = BearColor.black ∧
  bears 599 = BearColor.brown ∧
  bears 799 = BearColor.white

-- Exactly one guess is incorrect
def oneIncorrectGuess (bears : Fin 1000 → BearColor) : Prop :=
  ∃ (idx : Fin 5), 
    ¬iskanderGuesses bears ∧
    ∀ (j : Fin 5), (j ≠ idx → (bearGuessesIdx j bears = true))

-- The proof problem
theorem identifyIncorrectGuess (bears : Fin 1000 → BearColor) :
  isValidBearRow bears → iskanderGuesses bears → oneIncorrectGuess bears := sorry

end identifyIncorrectGuess_l129_129678


namespace cos_x_plus_pi_over_3_l129_129525

/-- Assume vectors m and n and their conditions -/
variables (m n : ℝ × ℝ)
variables (x : ℝ) 

/-- Define the vectors explicitly -/
def m : ℝ × ℝ := (2 * real.sqrt 3 * real.sin (x / 4), 2)
def n : ℝ × ℝ := (real.cos (x / 4), real.cos (x / 4) ^ 2)

/-- Dot product condition -/
axiom dot_product_condition : m.1 * n.1 + m.2 * n.2 = 2

/-- Proof statement: Prove cos(x + π/3) = 1/2 -/
theorem cos_x_plus_pi_over_3 : real.cos (x + real.pi / 3) = 1 / 2 := 
sorry

end cos_x_plus_pi_over_3_l129_129525


namespace area_of_given_region_l129_129301

noncomputable def radius_squared : ℝ := 16 -- Completing the square gives us a radius squared value of 16.
def area_of_circle (r : ℝ) : ℝ := π * r ^ 2

theorem area_of_given_region : area_of_circle (real.sqrt radius_squared) = 16 * π := by
  sorry

end area_of_given_region_l129_129301


namespace Joel_non_hot_peppers_l129_129162

constant Sunday_peppers : ℕ := 7
constant Monday_peppers : ℕ := 12
constant Tuesday_peppers : ℕ := 14
constant Wednesday_peppers : ℕ := 12
constant Thursday_peppers : ℕ := 5
constant Friday_peppers : ℕ := 18
constant Saturday_peppers : ℕ := 12

constant hot_pepper_percentage : ℝ := 0.20

noncomputable def total_peppers : ℕ :=
  Sunday_peppers + Monday_peppers + Tuesday_peppers +
  Wednesday_peppers + Thursday_peppers + Friday_peppers + Saturday_peppers

noncomputable def non_hot_peppers : ℕ :=
  (1 - hot_pepper_percentage) * total_peppers

theorem Joel_non_hot_peppers :
  non_hot_peppers = 64 :=
by
  sorry

end Joel_non_hot_peppers_l129_129162


namespace range_of_m_l129_129101

noncomputable def line_param : (ℝ → ℝ) × (ℝ → ℝ) := (λ t, 2 + real.sqrt 2 * t, λ t, real.sqrt 2 * t)
noncomputable def curve_polar : ℝ := 2
def intersection_points_polar (m : ℝ) : set (ℝ × ℝ) := 
    {(2, 0), (2, 3 * real.pi / 2)}

theorem range_of_m (m : ℝ) (A B : ℝ × ℝ) (h_intersect : A ∈ intersection_points_polar m ∧ B ∈ intersection_points_polar m) :
  |real.dist A B| ≤ 2 * real.sqrt 3 → m ∈ set.Icc (-2 * real.sqrt 2) (-real.sqrt 2) ∪ set.Icc (real.sqrt 2) (2 * real.sqrt 2) :=
sorry

end range_of_m_l129_129101


namespace complex_expression_equals_l129_129538

noncomputable def complex_expression : ℂ :=
  (1 + complex.I) / (3 - complex.I) - (complex.I) / (3 + complex.I)

theorem complex_expression_equals :
  complex_expression = (1 + complex.I) / 10 :=
by
  -- The proof is omitted (sorry serves as a placeholder)
  sorry

end complex_expression_equals_l129_129538


namespace triangle_area_l129_129437

def point := ℝ × ℝ

def A : point := (-3, 3)
def B : point := (5, -1)
def C : point := (13, 6)

def vec (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def area_of_parallelogram (v w : point) : ℝ :=
  real.abs (v.1 * w.2 - v.2 * w.1)

def area_of_triangle (A B C : point) : ℝ :=
  area_of_parallelogram (vec C A) (vec C B) / 2

theorem triangle_area :
  area_of_triangle A B C = 44 :=
sorry

end triangle_area_l129_129437


namespace m_plus_n_is_172_l129_129293

-- defining the conditions for m
def m := 3

-- helper function to count divisors
def count_divisors (x : ℕ) : ℕ :=
  (List.range x).filter (λ d, x % (d + 1) = 0).length.succ

-- defining the conditions for n
noncomputable def n :=
  let primes := List.filter nat.prime (List.range 100) in
  let candidates := primes.map (λ p, p * p) in
  (candidates.filter (λ x, x < 200)).maximum' sorry

theorem m_plus_n_is_172 : m + n = 172 :=
by
  -- filling in that m is 3
  let m : ℕ := 3
  -- filling in that n is 169
  let n : ℕ := 13 * 13
  show m + n = 172
  calc
    m + n = 3 + 169 := by rfl
    ... = 172 := by rfl

end m_plus_n_is_172_l129_129293


namespace intervals_of_monotonicity_range_of_a_l129_129093

-- Definitions and conditions
def f (x : ℝ) (k : ℝ) : ℝ := (k * x^2) / Real.exp x
def g (x : ℝ) : ℝ := (2 * Real.log x - x) / x

-- Theorem statements
theorem intervals_of_monotonicity (k : ℝ) (h : k > 0) :
  (∀ x, (x < 0 ∨ x > 2) → deriv (f x k) < 0) ∧
  (∀ x, (0 < x ∧ x < 2) → deriv (f x k) > 0) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x, (x > 0) → (Real.log (f x 1) > a * x) ↔ a < g e) := sorry

end intervals_of_monotonicity_range_of_a_l129_129093


namespace eccentricity_of_hyperbola_l129_129877

-- Define the elements based on given conditions
variables (a b c : ℝ) (λ μ : ℝ)
-- Condition 1: Equation of Hyperbola
variables (h1 : a > 0) (h2 : b > 0)
-- Condition 2: Right focus
def focus : ℝ × ℝ := (c, 0)
-- Condition 3 & 4: Coordinates, and origin condition
def A : ℝ × ℝ := (c, b * c / a)
def B : ℝ × ℝ := (c, -b * c / a)
def P : ℝ × ℝ := (c, b^2 / a)
variable (h3 : (c, b^2 / a) = ((λ + μ) * c, (λ - μ) * (b * c / a)))
variable (h4 : λ * μ = 4 / 25)

-- The main theorem to prove
theorem eccentricity_of_hyperbola : c / a = 5 / 4 :=
by
  sorry

end eccentricity_of_hyperbola_l129_129877


namespace min_shift_value_l129_129875

theorem min_shift_value (k : ℝ) (h : k > 0) : 
  ∃ k, 
    (∀ x, (sin x * cos x - sqrt 3 * cos x^2 = sin (2 * x - π / 3) - sqrt 3 / 2) → 
    (f x = g (x - k))) → 
    k = π / 3 :=
by
  sorry

end min_shift_value_l129_129875


namespace normal_dist_probability_l129_129863

variable {σ : ℝ} (X : ℝ → ℝ)

theorem normal_dist_probability
  (h1 : ∀ x, X x ∼ Normal 2 (σ^2))
  (h2 : P(X ≤ 4) = 0.84) :
  P(X < 0) = 0.16 :=
by
  sorry  -- Proof outline: P(X < 0) = P(X > 4), and given P(X ≤ 4) = 0.84, thus P(X > 4) = 1 - 0.84 = 0.16.

end normal_dist_probability_l129_129863


namespace valid_triangle_inequality_l129_129037

theorem valid_triangle_inequality (n : ℕ) (h : n = 6) :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c →
  n * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  intros a b c ha hb hc hineq
  have h₁ : n = 6 := h
  simplify_eq [h₁] at hineq
  have h₂ := nat.add_comm a b
  exact sorry

end valid_triangle_inequality_l129_129037


namespace flower_cost_l129_129919

-- Given conditions
variables {x y : ℕ} -- costs of type A and type B flowers respectively

-- Costs equations
def cost_equation_1 : Prop := 3 * x + 4 * y = 360
def cost_equation_2 : Prop := 4 * x + 3 * y = 340

-- Given the necessary planted pots and rates
variables {m n : ℕ} (Hmn : m + n = 600) 
-- Percentage survivals
def survival_rate_A : ℚ := 0.70
def survival_rate_B : ℚ := 0.90

-- Replacement condition
def replacement_cond : Prop := (1 - survival_rate_A) * m + (1 - survival_rate_B) * n ≤ 100

-- Minimum cost condition
def min_cost (m_plant : ℕ) (n_plant : ℕ) : ℕ := 40 * m_plant + 60 * n_plant

theorem flower_cost 
  (H1 : cost_equation_1)
  (H2 : cost_equation_2)
  (H3 : x = 40)
  (H4 : y = 60) 
  (Hmn : m + n = 600)
  (Hsurv : replacement_cond) : 
  (m = 200 ∧ n = 400) ∧ 
  (min_cost 200 400 = 32000) := 
sorry

end flower_cost_l129_129919


namespace walk_usual_time_l129_129298

theorem walk_usual_time (T : ℝ) (S : ℝ) (h1 : (5 / 4 : ℝ) = (T + 10) / T) : T = 40 :=
sorry

end walk_usual_time_l129_129298


namespace correct_propositions_l129_129501

universe u

variables {α β γ : Type u} [plane α] [plane β] [plane γ]
variables {l m n : Type u} [line l] [line m] [line n]

-- Proposition (1): Incorrect
def prop1 (h1 : α ⊥ β) (h2 : l ⊥ β) : Prop :=
  ¬ (l ∥ α)

-- Proposition (2): Correct
def prop2 (h1 : l ⊥ α) (h2 : l ⊥ β): Prop :=
  α ∥ β

-- Proposition (3): Correct
def prop3 (h1 : α ⊥ γ) (h2 : β ∥ γ): Prop :=
  α ⊥ β

-- Proposition (4): Incorrect
def prop4 (h1 : m ⊂ α) (h2 : n ⊂ α) (h3 : m ∥ β) (h4 : n ∥ β) : Prop :=
  ¬ (α ∥ β)

-- The main theorem combining all the propositions
theorem correct_propositions (h1_1 : α ⊥ β) (h1_2 : l ⊥ β) 
                            (h2_1 : l ⊥ α) (h2_2 : l ⊥ β)
                            (h3_1 : α ⊥ γ) (h3_2 : β ∥ γ)
                            (h4_1 : m ⊂ α) (h4_2 : n ⊂ α)
                            (h4_3 : m ∥ β) (h4_4 : n ∥ β) : 
  (¬ prop1 h1_1 h1_2) ∧ (prop2 h2_1 h2_2) ∧ (prop3 h3_1 h3_2) ∧ (¬ prop4 h4_1 h4_2 h4_3 h4_4) := 
by sorry

end correct_propositions_l129_129501


namespace faculty_student_count_l129_129336

theorem faculty_student_count 
  (N : ℕ) (A : ℕ) (B : ℕ) (T : ℕ) (F : ℕ) 
  (hN : N = 226) (hA : A = 450) (hB : B = 134)
  (hT : T = N + A - B)
  (hF : 0.80 * F ≈ T) : 
  F ≈ 678 := 
sorry

end faculty_student_count_l129_129336


namespace a_investment_l129_129360

theorem a_investment (B C total_profit A_share: ℝ) (hB: B = 7200) (hC: C = 9600) (htotal_profit: total_profit = 9000) 
  (hA_share: A_share = 1125) : ∃ x : ℝ, (A_share / total_profit) = (x / (x + B + C)) ∧ x = 2400 := 
by
  use 2400
  sorry

end a_investment_l129_129360


namespace min_sum_squares_roots_l129_129472

theorem min_sum_squares_roots (m : ℝ) :
  (∃ (α β : ℝ), 2 * α^2 - 3 * α + m = 0 ∧ 2 * β^2 - 3 * β + m = 0 ∧ α ≠ β) → 
  (9 - 8 * m ≥ 0) →
  (α^2 + β^2 = (3/2)^2 - 2 * (m/2)) →
  (α^2 + β^2 = 9/8) ↔ m = 9/8 :=
by
  sorry

end min_sum_squares_roots_l129_129472


namespace circumcircle_radius_POQ_is_sqrt_10_div_2_l129_129555

noncomputable def circumcircle_radius (O P Q : Point) : ℝ :=
  let C := ((O.x + P.x) / 2, (O.y + P.y) / 2) in
  Real.sqrt ((C.1 - P.x)^2 + (C.2 - P.y)^2)

theorem circumcircle_radius_POQ_is_sqrt_10_div_2 : 
  circumcircle_radius (0, 0) (1, 3) (-1, 1) = Real.sqrt 10 / 2 :=
sorry

end circumcircle_radius_POQ_is_sqrt_10_div_2_l129_129555


namespace exist_ints_a_b_l129_129060

theorem exist_ints_a_b (n : ℕ) : (∃ a b : ℤ, (n : ℤ) + a^2 = b^2) ↔ ¬ n % 4 = 2 := 
by
  sorry

end exist_ints_a_b_l129_129060


namespace gabriel_month_days_l129_129482

theorem gabriel_month_days (forgot_days took_days : ℕ) (h_forgot : forgot_days = 3) (h_took : took_days = 28) : 
  forgot_days + took_days = 31 :=
by
  sorry

end gabriel_month_days_l129_129482


namespace sequence_product_sum_l129_129087

theorem sequence_product_sum (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 2 ^ n - 1) →
  (∀ n, a n = if n = 1 then 1 else S n - S (n - 1)) →
  ∀ n, (∑ i in finset.range n, a (i + 1) * a (i + 2)) = (2/3 : ℚ) * (4 ^ n - 1) :=
begin
  intros hS ha n,
  sorry
end

end sequence_product_sum_l129_129087


namespace three_mathematicians_speak_same_language_l129_129192

theorem three_mathematicians_speak_same_language
  (mathematicians : Fin 9 → Finset ℕ)
  (h1 : ∀ (a b c : Fin 9), ∃ (l : ℕ), l ∈ mathematicians a ∧ l ∈ mathematicians b ∨ l ∈ mathematicians b ∧ l ∈ mathematicians c ∨ l ∈ mathematicians a ∧ l ∈ mathematicians c)
  (h2 : ∀ (a : Fin 9), (mathematicians a).card ≤ 3) : 
  ∃ (l : ℕ), (mathematicians.filter (λ s, l ∈ s)).card ≥ 3 :=
begin
  sorry
end

end three_mathematicians_speak_same_language_l129_129192


namespace executive_board_ways_l129_129190

-- Conditions stated in Lean 4
def total_members : ℕ := 40
def board_size : ℕ := 6

-- Helper functions for combinatorial calculations
noncomputable def choose (n k : ℕ) : ℕ := nat.choose n k

noncomputable def permutations (n : ℕ) : ℕ := nat.fact n
noncomputable def arrangements(n k : ℕ) : ℕ := permutations k / permutations (k - n)

-- Define the problem statement in Lean 4 
theorem executive_board_ways : 
  (choose total_members board_size) * 30 = 115151400 := 
by
  -- Using sorry to skip the proof
  sorry

end executive_board_ways_l129_129190


namespace increasing_interval_log_function_l129_129635

open Real

noncomputable def is_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

theorem increasing_interval_log_function :
  let f := λ x : ℝ, 4 + 3x - x^2
  let g := λ x : ℝ, log (4 + 3x - x^2)
  let I := Set.Ioc (-1) (3/2)
  ∀ x, x ∈ Set.Ioo (-1:ℝ) 4 → 0 < 4 + 3x - x^2 →
  is_increasing f I →
  is_increasing g I :=
sorry

end increasing_interval_log_function_l129_129635


namespace ratio_largest_to_sum_l129_129415

def geometric_series_sum (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

theorem ratio_largest_to_sum :
  let largest := 2^12 in
  let sum_others := geometric_series_sum 1 2 12 in
  ∃ (r : ℝ), r = (largest : ℝ) / (sum_others : ℝ) ∧ r ≈ 1.0002 :=
by
  let largest := 2^12
  let sum_others := (2^12) - 1
  use (largest : ℝ) / (sum_others : ℝ)
  split
  · rfl
  · sorry

end ratio_largest_to_sum_l129_129415


namespace find_m_l129_129111

theorem find_m (m n c : ℝ) (h : log 2 m = c - log 2 n + 1) : m = 2^(c + 1) / n :=
sorry

end find_m_l129_129111


namespace volume_intersection_zero_l129_129728

/-- The set of points satisfying |x| + |y| + |z| ≤ 1. -/
def region1 (x y z : ℝ) : Prop :=
  |x| + |y| + |z| ≤ 1

/-- The set of points satisfying |x| + |y| + |z-2| ≤ 1. -/
def region2 (x y z : ℝ) : Prop :=
  |x| + |y| + |z-2| ≤ 1

/-- The intersection of region1 and region2 forms a region with volume 0. -/
theorem volume_intersection_zero : 
  (∫ x y z, (region1 x y z ∧ region2 x y z)) = 0 := sorry

end volume_intersection_zero_l129_129728


namespace difference_in_earnings_in_currency_B_l129_129211

-- Definitions based on conditions
def num_red_stamps : Nat := 30
def num_white_stamps : Nat := 80
def price_per_red_stamp_currency_A : Nat := 5
def price_per_white_stamp_currency_B : Nat := 50
def exchange_rate_A_to_B : Nat := 2

-- Theorem based on the question and correct answer
theorem difference_in_earnings_in_currency_B : 
  num_white_stamps * price_per_white_stamp_currency_B - 
  (num_red_stamps * price_per_red_stamp_currency_A * exchange_rate_A_to_B) = 3700 := 
  by
  sorry

end difference_in_earnings_in_currency_B_l129_129211


namespace find_y_when_x4_l129_129344

theorem find_y_when_x4 : 
  (∀ x y : ℚ, 5 * y + 3 = 344 / (x ^ 3)) ∧ (5 * (8:ℚ) + 3 = 344 / (2 ^ 3)) → 
  (∃ y : ℚ, 5 * y + 3 = 344 / (4 ^ 3) ∧ y = 19 / 40) := 
by
  sorry

end find_y_when_x4_l129_129344


namespace emily_small_gardens_l129_129430

theorem emily_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) (num_small_gardens : ℕ) :
  total_seeds = 41 →
  big_garden_seeds = 29 →
  seeds_per_small_garden = 4 →
  num_small_gardens = (total_seeds - big_garden_seeds) / seeds_per_small_garden →
  num_small_gardens = 3 :=
by
  intros h_total h_big h_seeds_per_small h_num_small
  rw [h_total, h_big, h_seeds_per_small] at h_num_small
  exact h_num_small

end emily_small_gardens_l129_129430


namespace tetrahedron_circumradius_m_plus_n_l129_129141

noncomputable def circumradius_of_tetrahedron_base (a b c s: ℝ) (h1: a = 108) (h2: b = 108) (h3: c = 108) (SI: ℝ) (h4: SI = 125) : ℝ :=
let base_length := (500 / Real.sqrt 6) in
108 -- Since the base's circumradius calculation results in 108

theorem tetrahedron_circumradius : circumradius_of_tetrahedron_base 108 108 108 125 = 108 :=
by
  unfold circumradius_of_tetrahedron_base
  sorry

theorem m_plus_n {m n: ℕ} (mnp: R = Real.sqrt (m / n)) (gcd_mn: Nat.gcd m n = 1) (r_val: R = 108) :
  11664 = m ∧ n = 1 → m + n = 11665 :=
by
  intros h
  cases h with h_m h_n
  rw [h_m, h_n]
  norm_num
  sorry

end tetrahedron_circumradius_m_plus_n_l129_129141


namespace incorrect_guess_20_l129_129675

-- Define the assumptions and conditions
def bears : Nat → String := sorry -- function that determines the color of the bear at position n
axiom bears_color_constraint : ∀ n:Nat, exists b:List String, b.length = 3 ∧ Set ("W" "B" "Bk") = List.toSet b ∧ 
  List.all (List.sublist b (n, n+1, n+2) bears = fun c=> c = "W" or c = "B" or c = "Bk") 

-- Iskander's guesses
def guess1 := (2, "W")
def guess2 := (20, "B")
def guess3 := (400, "Bk")
def guess4 := (600, "B")
def guess5 := (800, "W")

-- Function to check the bear at each position
def check_bear (n:Nat) : String := sorry

-- Iskander's guess correctness, exactly one is wrong
axiom one_wrong : count (check_bear 2 =="W") 
                         + count (check_bear 20 == "B") 
                         + count (check_bear 400 =="Bk") 
                         + count (check_bear 600 =="B") 
                         + count (check_bear 800 =="W") = 4

-- Prove that the guess for the 20th bear is incorrect
theorem incorrect_guess_20 : ∀ {n:Nat} (h : n=20), (check_bear n != "B") := sorry

end incorrect_guess_20_l129_129675


namespace range_of_t_l129_129097

noncomputable def f (a x : ℝ) : ℝ :=
  a / x - x + a * Real.log x

noncomputable def g (a x : ℝ) : ℝ :=
  f a x + 1/2 * x^2 - (a - 1) * x - a / x

theorem range_of_t (a x₁ x₂ t : ℝ) (h1 : f a x₁ = f a x₂) (h2 : x₁ + x₂ = a)
  (h3 : x₁ * x₂ = a) (h4 : a > 4) (h5 : g a x₁ + g a x₂ > t * (x₁ + x₂)) :
  t < Real.log 4 - 3 :=
  sorry

end range_of_t_l129_129097


namespace find_positive_integer_n_l129_129741

theorem find_positive_integer_n (n : ℕ) (h₁ : 200 % n = 5) (h₂ : 395 % n = 5) : n = 13 :=
sorry

end find_positive_integer_n_l129_129741


namespace bag_weight_l129_129837

theorem bag_weight (W : ℕ) 
  (h1 : 2 * W + 82 * (2 * W) = 664) : 
  W = 4 := by
  sorry

end bag_weight_l129_129837


namespace hank_newspaper_reading_time_l129_129527

theorem hank_newspaper_reading_time
  (n_days_weekday : ℕ := 5)
  (novel_reading_time_weekday : ℕ := 60)
  (n_days_weekend : ℕ := 2)
  (total_weekly_reading_time : ℕ := 810)
  (x : ℕ)
  (h1 : n_days_weekday * x + n_days_weekday * novel_reading_time_weekday +
        n_days_weekend * 2 * x + n_days_weekend * 2 * novel_reading_time_weekday = total_weekly_reading_time) :
  x = 30 := 
by {
  sorry -- Proof would go here
}

end hank_newspaper_reading_time_l129_129527


namespace toys_secured_in_25_minutes_l129_129973

def net_toy_gain_per_minute (toys_mom_puts : ℕ) (toys_mia_takes : ℕ) : ℕ :=
  toys_mom_puts - toys_mia_takes

def total_minutes (total_toys : ℕ) (toys_mom_puts : ℕ) (toys_mia_takes : ℕ) : ℕ :=
  (total_toys - 1) / net_toy_gain_per_minute toys_mom_puts toys_mia_takes + 1

theorem toys_secured_in_25_minutes :
  total_minutes 50 5 3 = 25 :=
by
  sorry

end toys_secured_in_25_minutes_l129_129973


namespace father_sleep_hours_l129_129208

theorem father_sleep_hours (samantha_sleep : ℕ) (baby_factor : ℚ) (father_factor : ℚ) : 
  samantha_sleep = 8 → 
  baby_factor = 2.5 → 
  father_factor = 0.5 → 
  let baby_sleep_per_night := samantha_sleep * baby_factor in 
  let baby_sleep_per_week := baby_sleep_per_night * 7 in 
  let father_sleep_per_week  := baby_sleep_per_week * father_factor in 
  father_sleep_per_week = 70 :=
begin
  intros h_samantha h_baby h_father,
  simp [h_samantha, h_baby, h_father],
  have baby_sleep := 8 * 2.5,
  have baby_week := baby_sleep * 7,
  have father_sleep := baby_week * 0.5,
  norm_num at *,
end

end father_sleep_hours_l129_129208


namespace quartic_polynomial_r_4_l129_129172

-- Define the conditions with Lean definitions
variables (r : ℝ → ℝ)
hypothesis h_monic : nat_degree (polynomial.map polynomial.C r) = 4
hypothesis h_r0 : r 0 = 1
hypothesis h_r1 : r 1 = 4
hypothesis h_r2 : r 2 = 9
hypothesis h_r3 : r 3 = 16

-- Statement to prove
theorem quartic_polynomial_r_4 :
  r 4 = 41 :=
sorry

end quartic_polynomial_r_4_l129_129172


namespace sum_of_roots_in_interval_minimum_m_translation_l129_129514

def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 - 2 * Real.sin x * Real.cos x + 1

theorem sum_of_roots_in_interval (h : (0 : ℝ) < x ∧ x < π) :
  let x1 := π / 4, let x2 := π / 2 in x1 + x2 = 3 * π / 4 :=
sorry

theorem minimum_m_translation (h : (0 : ℝ) < m) :
  let m_min := π / 8 in m = m_min :=
sorry

end sum_of_roots_in_interval_minimum_m_translation_l129_129514


namespace sum_of_distinct_divisors_l129_129616

theorem sum_of_distinct_divisors (N : ℕ) : 
  (∃ N2004 : ℕ, ∀ n : ℕ, n ≥ N2004 → ∃ (a : Fin 2004 → ℕ), 
    (∀ i : Fin 2003, a i < a (Fin.succ i)) ∧ 
    (∀ i : Fin 2003, a i ∣ a (Fin.succ i)) ∧ 
    (Finset.univ.sum (λ i, a i) = n)) :=
sorry

end sum_of_distinct_divisors_l129_129616


namespace solve_rational_inequality_l129_129216

def rational_inequality_solution : set ℝ := 
  { x : ℝ | (10 * x^2 + 20 * x - 60) / ((3 * x - 5) * (x + 6)) < 4 }

theorem solve_rational_inequality :
    { x : ℝ | (10 * x^2 + 20 * x - 60) / ((3 * x - 5) * (x + 6)) < 4 } = { x | -6 < x ∧ x < 5 / 3 } ∪ { x | 2 < x } :=
by
  sorry

end solve_rational_inequality_l129_129216


namespace num_x_intersections_l129_129853

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x < 2 then x^3 - x else f (x - 2)

theorem num_x_intersections :
  ∃ n : ℕ, n = 7 ∧
  (∀ x ∈ set.Icc (0 : ℝ) (6 : ℝ), f x = 0 → true) :=
begin
  -- proof to be filled in
  sorry
end

end num_x_intersections_l129_129853


namespace right_triangle_hypotenuse_length_l129_129940

/-- In the given right triangle ABC with right angle at vertex A, 
there exists points P and Q on legs AB and AC respectively, such that 
AP:PB = 2:3 and AQ:QC = 3:1 with lengths BQ = 20 and CP = 36. 
We need to prove that the length of the hypotenuse BC is 6√27. -/
theorem right_triangle_hypotenuse_length (AB AC : ℝ) (P Q : ℝ) 
  (h1 : P = 2 / 5 * AB ∧ (AB - P) = 3 / 5 * AB)
  (h2 : Q = 3 / 4 * AC ∧ (AC - Q) = 1 / 4 * AC)
  (h3 : BQ = 20) (h4 : CP = 36) 
  : let BC := Real.sqrt (AB^2 + AC^2) in 
  BC = 6 * Real.sqrt (27) :=
  by
  sorry

end right_triangle_hypotenuse_length_l129_129940


namespace incenter_on_dividing_segment_l129_129778

-- Define the triangle and necessary points
variables {A B C D : Type} [has_zero A] [has_one A] [add_comm_group A] [module ℝ A]
variables (triangle : set A) (incenter : A) (line_segment : set A)

-- Assumptions based on the problem conditions
axiom triangle_is_divided : 
  ∃ line_segment, (triangle ∩ line_segment).nonempty ∧ 
                  triangle \ line_segment = ∅ ∧ 
                  let S₁ := {p : A | p ∈ triangle ∧ p ∉ line_segment}, 
                      S₂ := {p : A | p ∈ triangle ∧ p ∈ line_segment} 
                  in ∃ (S₁_perimeter S₂_perimeter : ℝ), S₁_perimeter = S₂_perimeter ∧ 
                     ∃ (S₁_area S₂_area : ℝ), S₁_area = S₂_area

-- Prove that the incenter lies on the dividing line segment
theorem incenter_on_dividing_segment 
  (h : ∃ line_segment, triangle_is_divided triangle incenter line_segment) : 
  incenter ∈ line_segment := 
sorry

end incenter_on_dividing_segment_l129_129778


namespace wrong_guess_is_20_l129_129662

-- Define the colors
inductive Color
| white
| brown
| black

-- Assume we have a sequence of 1000 bears
def bears : fin 1000 → Color := sorry

-- Hypotheses
axiom colors_per_three : ∀ (i : fin 998), 
  ({bears i, bears (i + 1), bears (i + 2)} = {Color.white, Color.brown, Color.black} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.black, Color.white, Color.brown} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.brown, Color.black, Color.white})

axiom exactly_one_wrong : 
  (bears 1 = Color.white ∧ bears 19 ≠ Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 ≠ Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 ≠ Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 ≠ Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 ≠ Color.white)

-- Define the theorem to prove
theorem wrong_guess_is_20 : 
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) →
  ¬(bears 19 = Color.brown) := 
sorry

end wrong_guess_is_20_l129_129662


namespace find_vec_b_l129_129951

def vec_a : ℝ × ℝ × ℝ := (3, 2, 4)
def vec_b : ℝ × ℝ × ℝ := (5, -1, -2)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

theorem find_vec_b :
  vec_a = (3, 2, 4) →
  dot_product vec_a vec_b = 20 ∧
  cross_product vec_a vec_b = (-15, 5, 2) :=
by
  intro ha
  simp [vec_a, vec_b, dot_product, cross_product]
  sorry

end find_vec_b_l129_129951


namespace lottery_prize_l129_129185

noncomputable def ticket_price (n : ℕ) : ℕ :=
  match n with
  | 0     => 2
  | (n+1) => ticket_price n + n + 2

noncomputable def total_income (num_tickets : ℕ) : ℕ :=
  (List.range num_tickets).sum ticket_price

noncomputable def tax (income : ℕ) : ℕ := (income / 10)

noncomputable def prize_money (total_income num_tickets : ℕ) : ℕ :=
  let income_after_tax_and_fee := total_income - tax total_income - 5
  income_after_tax_and_fee - 10

theorem lottery_prize :
  prize_money (total_income 10) 10 = 192 :=
by
  sorry

end lottery_prize_l129_129185


namespace three_digit_number_divisible_by_eleven_l129_129740

theorem three_digit_number_divisible_by_eleven
  (x : ℕ) (n : ℕ)
  (units_digit_is_two : n % 10 = 2)
  (hundreds_digit_is_seven : n / 100 = 7)
  (tens_digit : n = 700 + x * 10 + 2)
  (divisibility_condition : (7 - x + 2) % 11 = 0) :
  n = 792 := by
  sorry

end three_digit_number_divisible_by_eleven_l129_129740


namespace find_RS_l129_129706

variables (D E F Q R S N : Type)
variables [has_dist D F] [has_dist E F] (angle_EDF : angle θ E D F)

def DF := 360
def EF := 240
def DQ := 180
def DN := 150
def Q_is_midpoint_SN : symmetric_median Q S N := sorry
def ER_is_bisector : angle_bisector E R := sorry
axiom angle_EDF_bisected : angle_bisector θ E D R := sorry
def Q_on_DF := point_on Q D F 
def R_on_DE := point_on R D E
def E := point E

theorem find_RS : RS = 112 :=
by
  sorry

end find_RS_l129_129706


namespace smallest_N_l129_129353

-- Definitions corresponding to the conditions
def circular_table (chairs : ℕ) : Prop := chairs = 72

def proper_seating (N chairs : ℕ) : Prop :=
  ∀ (new_person : ℕ), new_person < chairs →
    (∃ seated, seated < N ∧ (seated - new_person).gcd chairs = 1)

-- Problem statement
theorem smallest_N (chairs : ℕ) :
  circular_table chairs →
  ∃ N, proper_seating N chairs ∧ (∀ M < N, ¬ proper_seating M chairs) ∧ N = 18 :=
by
  intro h
  sorry

end smallest_N_l129_129353


namespace sum_of_integral_values_l129_129824

theorem sum_of_integral_values (h1 : ∀ (x y c : ℤ), y = x^2 - 9 * x - c → y = 0 → ∃ r : ℚ, ∃ s : ℚ, r + s = 9 ∧ r * s = c)
    (h2 : ∀ (c : ℤ), (∃ k : ℤ, 81 + 4 * c = k^2 ∧ k^2 ≡ 1 [MOD 4]) ↔ ∃ k : ℤ, 81 + 4 * c = k^2 ∧ k % 2 = 1 ) :
    ∑ c in { c : ℤ | -20 ≤ c ∧ c ≤ 30 ∧ ∃ k : ℤ, 81 + 4 * c = k^2 ∧ k % 2 = 1 }, c = 32 :=
by {
  -- Proof omitted
  sorry
}

end sum_of_integral_values_l129_129824


namespace incorrect_guess_l129_129700

-- Define the conditions
def bears : ℕ := 1000

inductive Color
| White
| Brown
| Black

constant bear_color : ℕ → Color -- The color of the bear at each position

axiom condition : ∀ n : ℕ, n < bears - 2 → 
  ∃ i j k, (i, j, k ∈ {Color.White, Color.Brown, Color.Black}) ∧ 
  (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ 
  (bear_color n = i ∧ bear_color (n+1) = j ∧ bear_color (n+2) = k) 

constants (g1 : bear_color 2 = Color.White)
          (g2 : bear_color 20 = Color.Brown)
          (g3 : bear_color 400 = Color.Black)
          (g4 : bear_color 600 = Color.Brown)
          (g5 : bear_color 800 = Color.White)

-- The proof problem
theorem incorrect_guess : bear_color 20 ≠ Color.Brown :=
by sorry

end incorrect_guess_l129_129700


namespace angle_BAC_is_120_degrees_l129_129614

variables {A B C E : Type} -- Vertices of the triangle and point E
variable (AB AC BC AE : ℝ) -- Lengths of sides and the angle bisector

-- Definitions of key conditions
def bisector_length (AE: ℝ) : Prop := AE = 5
def angle_bisector_conditions (AB AC BC AE : ℝ) : Prop :=
  ∃ (l : Type), (lines_through_point_with_angles E l AB AC) ∧ (inclined_at_angle_with l BC 30) ∧
  (cuts_segment BAC l (2 * sqrt(3)))

noncomputable 
def right_angle_triangle_bisectors (AB AC BC AD BE CF : ℝ) : Prop :=
  -- Here we should define the properties about the triangle formed by the bisectors being right-angled
  sorry -- Placeholder since explicit definition will be complex

theorem angle_BAC_is_120_degrees 
  (AB AC BC AE : ℝ) 
  (h_bisector_length : bisector_length AE) 
  (h_conditions : angle_bisector_conditions AB AC BC AE) 
  (h_right_angle_triangle : right_angle_triangle_bisectors AB AC BC 5 5 5) :
  ∠BAC = 120 := 
  sorry -- Proof goes here, but is not required

end angle_BAC_is_120_degrees_l129_129614


namespace police_coverage_l129_129557

-- Define the intersections and streets
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

open Intersection

-- Define the streets
def Streets : List (List Intersection) :=
  [ [A, B, C, D],    -- Horizontal street 1
    [E, F, G],       -- Horizontal street 2
    [H, I, J, K],    -- Horizontal street 3
    [A, E, H],       -- Vertical street 1
    [B, F, I],       -- Vertical street 2
    [D, G, J],       -- Vertical street 3
    [H, F, C],       -- Diagonal street 1
    [C, G, K]        -- Diagonal street 2
  ]

-- Define the set of intersections where police officers are 
def policeIntersections : List Intersection := [B, G, H]

-- State the theorem to be proved
theorem police_coverage : 
  ∀ (street : List Intersection), street ∈ Streets → 
  ∃ (i : Intersection), i ∈ policeIntersections ∧ i ∈ street := 
sorry

end police_coverage_l129_129557


namespace number_of_incorrect_statements_l129_129421

-- Conditions
def cond1 (p q : Prop) : Prop := (p ∨ q) → (p ∧ q)

def cond2 (x : ℝ) : Prop := x > 5 → x^2 - 4*x - 5 > 0

def cond3 : Prop := ∃ x0 : ℝ, x0^2 + x0 - 1 < 0

def cond3_neg : Prop := ∀ x : ℝ, x^2 + x - 1 ≥ 0

def cond4 (x : ℝ) : Prop := (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)

-- Proof problem
theorem number_of_incorrect_statements : 
  (¬ cond1 (p := true) (q := false)) ∧ (cond2 (x := 6)) ∧ (cond3 → cond3_neg) ∧ (¬ cond4 (x := 0)) → 
  2 = 2 :=
by
  sorry

end number_of_incorrect_statements_l129_129421


namespace minimum_ratio_hypercube_l129_129574

theorem minimum_ratio_hypercube (n : ℕ) (x : Fin 2^n → ℝ) :
  let S := 1 / 2 * ∑ i in Finset.univ, ∑ j in (Finset.filter (λ j, vertex_adjacent i j) Finset.univ), x i * x j
  in (∑ i in Finset.univ, x i ^ 2) ≠ 0 →
     S / (∑ i in Finset.univ, x i ^ 2) = n / 2 :=
by
  sorry

def vertex_adjacent (i j : Fin 2^n) : Prop :=
  Finset.card (Finset.filter (λ p, p) (Finset.image (λ b:Fin n, i.val.toFin b ≠ j.val.toFin b) Finset.univ)) = 1

end minimum_ratio_hypercube_l129_129574


namespace probability_three_students_different_courses_probability_two_courses_not_chosen_l129_129280

/--
Problem (Ⅰ):
Calculate the probability that all three students choose different elective courses.
Given:
- There are four elective courses.
- Each student must choose exactly one elective course.
We need to prove that the probability that all three students choose different elective courses is 3 / 8.
-/
theorem probability_three_students_different_courses : 
  let total_ways := 4^3 in
  let different_ways := Nat.factorial 4 / Nat.factorial (4 - 3) in
  (different_ways : ℚ) / total_ways = 3 / 8 :=
by
  let total_ways := 4^3
  let different_ways := Nat.factorial 4 / Nat.factorial (4 - 3)
  have h : (3 : ℚ) / 8 = 3 / 8 := rfl
  rw [h]
  sorry

/--
Problem (Ⅱ):
Calculate the probability that exactly two elective courses are not chosen by any of the three students.
Given:
- There are four elective courses.
- Each student must choose exactly one elective course.
We need to prove that the probability that exactly two elective courses are not chosen by any of the three students is 9 / 16.
-/
theorem probability_two_courses_not_chosen : 
  let total_ways := 4^3 in
  let num_ways_not_chosen := (Nat.choose 4 2) * (Nat.choose 3 2) * (Nat.factorial 2 / Nat.factorial (2 - 2)) in
  (num_ways_not_chosen : ℚ) / total_ways = 9 / 16 :=
by
  let total_ways := 4^3
  let num_ways_not_chosen := (Nat.choose 4 2) * (Nat.choose 3 2) * (Nat.factorial 2 / Nat.factorial (2 - 2))
  have h : (9 : ℚ) / 16 = 9 / 16 := rfl
  rw [h]
  sorry

end probability_three_students_different_courses_probability_two_courses_not_chosen_l129_129280


namespace incorrect_guess_l129_129695

-- Define the conditions
def bears : ℕ := 1000

inductive Color
| White
| Brown
| Black

constant bear_color : ℕ → Color -- The color of the bear at each position

axiom condition : ∀ n : ℕ, n < bears - 2 → 
  ∃ i j k, (i, j, k ∈ {Color.White, Color.Brown, Color.Black}) ∧ 
  (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ 
  (bear_color n = i ∧ bear_color (n+1) = j ∧ bear_color (n+2) = k) 

constants (g1 : bear_color 2 = Color.White)
          (g2 : bear_color 20 = Color.Brown)
          (g3 : bear_color 400 = Color.Black)
          (g4 : bear_color 600 = Color.Brown)
          (g5 : bear_color 800 = Color.White)

-- The proof problem
theorem incorrect_guess : bear_color 20 ≠ Color.Brown :=
by sorry

end incorrect_guess_l129_129695


namespace circumcircle_radius_l129_129152

noncomputable def radius_of_circumcircle (a : ℝ) (sin_A : ℝ) : ℝ :=
  a / (2 * sin_A)

theorem circumcircle_radius (a : ℝ) (sin_A : ℝ) (R : ℝ) 
  (h1 : a = 2) (h2 : sin_A = 1 / 3) : R = 3 :=
by
  rw [h1, h2]
  have h : (2 : ℝ) / (2 * (1 / 3)) = 3 := sorry
  rw h
  sorry

end circumcircle_radius_l129_129152


namespace zephyrian_word_count_l129_129981

theorem zephyrian_word_count :
  ∃ (w : Nat), w = 8 + 8^2 + 8^3 ∧ w = 584 :=
by
  use 584
  split
  · exact rfl
  sorry

end zephyrian_word_count_l129_129981


namespace negation_of_proposition_l129_129520

theorem negation_of_proposition :
  (∀ x : ℝ, x > 0 → x + (1 / x) ≥ 2) →
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + (1 / x₀) < 2) :=
sorry

end negation_of_proposition_l129_129520


namespace net_effect_on_sale_value_l129_129544

theorem net_effect_on_sale_value
  (P Q : ℝ)
  (h1 : 0.82 * P = P - 0.18 * P)
  (h2 : 1.72 * Q = Q + 0.72 * Q) :
  (((0.82 * P) * (1.72 * Q) - (P * Q)) / (P * Q)) * 100 = 41.04 :=
begin
  sorry
end

end net_effect_on_sale_value_l129_129544


namespace largest_band_members_l129_129784

theorem largest_band_members :
  ∃ (r x : ℕ), r * x + 3 = 107 ∧ (r - 3) * (x + 2) = 107 ∧ r * x < 147 :=
sorry

end largest_band_members_l129_129784


namespace value_of_abs_sum_l129_129266

noncomputable def cos_squared (θ : ℝ) : ℝ := (Real.cos θ) ^ 2

theorem value_of_abs_sum (θ x : ℝ) (h : Real.log x / Real.log 2 = 3 - 2 * cos_squared θ) :
  |x - 2| + |x - 8| = 6 := by
    sorry

end value_of_abs_sum_l129_129266


namespace bricks_needed_to_build_wall_l129_129108

def volume_of_brick (length_brick height_brick thickness_brick : ℤ) : ℤ :=
  length_brick * height_brick * thickness_brick

def volume_of_wall (length_wall height_wall thickness_wall : ℤ) : ℤ :=
  length_wall * height_wall * thickness_wall

def number_of_bricks_needed (length_wall height_wall thickness_wall length_brick height_brick thickness_brick : ℤ) : ℤ :=
  (volume_of_wall length_wall height_wall thickness_wall + volume_of_brick length_brick height_brick thickness_brick - 1) / 
  volume_of_brick length_brick height_brick thickness_brick

theorem bricks_needed_to_build_wall : number_of_bricks_needed 800 100 5 25 11 6 = 243 := 
  by 
    sorry

end bricks_needed_to_build_wall_l129_129108


namespace wrong_guess_is_20_l129_129661

-- Define the colors
inductive Color
| white
| brown
| black

-- Assume we have a sequence of 1000 bears
def bears : fin 1000 → Color := sorry

-- Hypotheses
axiom colors_per_three : ∀ (i : fin 998), 
  ({bears i, bears (i + 1), bears (i + 2)} = {Color.white, Color.brown, Color.black} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.black, Color.white, Color.brown} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.brown, Color.black, Color.white})

axiom exactly_one_wrong : 
  (bears 1 = Color.white ∧ bears 19 ≠ Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 ≠ Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 ≠ Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 ≠ Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 ≠ Color.white)

-- Define the theorem to prove
theorem wrong_guess_is_20 : 
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) →
  ¬(bears 19 = Color.brown) := 
sorry

end wrong_guess_is_20_l129_129661


namespace calculate_f_at_5_l129_129967

noncomputable def g (y : ℝ) : ℝ := (1 / 2) * y^2

noncomputable def f (x y : ℝ) : ℝ := 2 * x^2 + g y

theorem calculate_f_at_5 (y : ℝ) (h1 : f 2 y = 50) (h2 : y = 2*Real.sqrt 21) :
  f 5 y = 92 :=
by
  sorry

end calculate_f_at_5_l129_129967


namespace combined_weight_l129_129541

variable (J S : ℝ)

-- Given conditions
def jake_current_weight := (J = 152)
def lose_weight_equation := (J - 32 = 2 * S)

-- Question: combined weight of Jake and his sister
theorem combined_weight (h1 : jake_current_weight J) (h2 : lose_weight_equation J S) : J + S = 212 :=
by
  sorry

end combined_weight_l129_129541


namespace sin_series_identity_l129_129983

theorem sin_series_identity (φ : ℝ) (n : ℕ) : 
  (∑ k in finset.range n, real.sin ((2 * k + 1) * φ)) = real.sin (n * φ) ^ 2 / real.sin φ :=
sorry

end sin_series_identity_l129_129983


namespace compute_f_1986_l129_129113

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined_for_nonneg_integers : ∀ x : ℕ, ∃ y : ℤ, f x = y
axiom f_one : f 1 = 1
axiom f_functional_equation : ∀ (a b : ℕ), f (a + b) = f a + f b - 2 * f (a * b)

theorem compute_f_1986 : f 1986 = 0 :=
  sorry

end compute_f_1986_l129_129113


namespace find_wrong_guess_l129_129687

-- Define the three colors as an inductive type.
inductive Color
| white
| brown
| black

-- Define the bears as a list of colors.
def bears (n : ℕ) : Type := list Color

-- Define the conditions: 
-- There are 1000 bears and each tuple of 3 consecutive bears has all three colors.
def valid_bears (b : bears 1000) : Prop :=
  ∀ i : ℕ, i + 2 < 1000 → 
    ∃ c1 c2 c3 : Color, 
      c1 ∈ b.nth i ∧ c2 ∈ b.nth (i+1) ∧ c3 ∈ b.nth (i+2) ∧ 
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3

-- Define Iskander's guesses.
def guesses (b : bears 1000) : Prop :=
  b.nth 1 = some Color.white ∧
  b.nth 19 = some Color.brown ∧
  b.nth 399 = some Color.black ∧
  b.nth 599 = some Color.brown ∧
  b.nth 799 = some Color.white

-- Prove that exactly one of Iskander's guesses is wrong.
def wrong_guess (b : bears 1000) : Prop :=
  (b.nth 19 ≠ some Color.brown) ∧
  valid_bears b ∧
  guesses b →
  ∃ i, i ∈ {1, 19, 399, 599, 799} ∧ (b.nth i ≠ some Color.white ∧ b.nth i ≠ some Color.brown ∧ b.nth i ≠ some Color.black)

theorem find_wrong_guess : 
  ∀ b : bears 1000, 
  valid_bears b → guesses b → wrong_guess b :=
  by
  intros b vb gs
  sorry

end find_wrong_guess_l129_129687


namespace minimize_surface_area_l129_129408

noncomputable def cone_minimal_surface_area (V : ℝ) (R H : ℝ) (h_pos : 0 < H) (r_pos : 0 < R) : Prop :=
  (H = R) ∧ (V = (1/3) * real.pi * R^2 * H)

theorem minimize_surface_area (V R H : ℝ) (h_pos : 0 < H) (r_pos : 0 < R) (ht: cone_minimal_surface_area V R H h_pos r_pos) :
  (H / R = 1) := by
  sorry

end minimize_surface_area_l129_129408


namespace sum_of_squares_mod_17_l129_129317

theorem sum_of_squares_mod_17 :
  (∑ i in Finset.range 16, i^2) % 17 = 11 := 
sorry

end sum_of_squares_mod_17_l129_129317


namespace incorrect_guess_20_l129_129673

-- Define the assumptions and conditions
def bears : Nat → String := sorry -- function that determines the color of the bear at position n
axiom bears_color_constraint : ∀ n:Nat, exists b:List String, b.length = 3 ∧ Set ("W" "B" "Bk") = List.toSet b ∧ 
  List.all (List.sublist b (n, n+1, n+2) bears = fun c=> c = "W" or c = "B" or c = "Bk") 

-- Iskander's guesses
def guess1 := (2, "W")
def guess2 := (20, "B")
def guess3 := (400, "Bk")
def guess4 := (600, "B")
def guess5 := (800, "W")

-- Function to check the bear at each position
def check_bear (n:Nat) : String := sorry

-- Iskander's guess correctness, exactly one is wrong
axiom one_wrong : count (check_bear 2 =="W") 
                         + count (check_bear 20 == "B") 
                         + count (check_bear 400 =="Bk") 
                         + count (check_bear 600 =="B") 
                         + count (check_bear 800 =="W") = 4

-- Prove that the guess for the 20th bear is incorrect
theorem incorrect_guess_20 : ∀ {n:Nat} (h : n=20), (check_bear n != "B") := sorry

end incorrect_guess_20_l129_129673


namespace exists_integers_for_expression_l129_129179

theorem exists_integers_for_expression (n : ℤ) : 
  ∃ a b c d : ℤ, n = a^2 + b^2 - c^2 - d^2 := 
sorry

end exists_integers_for_expression_l129_129179


namespace inequality_ab_gt_ac_l129_129064

theorem inequality_ab_gt_ac {a b c : ℝ} (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
sorry

end inequality_ab_gt_ac_l129_129064


namespace value_of_T_l129_129422

noncomputable def geometric_series_sum (n : ℕ) (x : ℚ) : ℚ :=
  (1 - x^(n+1)) / (1 - x)

noncomputable def T : ℚ :=
  3003 + (1 / 3) * (3002 + (1 / 3) * (3001 + ... + (1 / 3) * (4 + (1 / 3) * 3) ... ))

theorem value_of_T :
  T = 4503.75 :=
by
  have sum_geometric_series : ∀ n : ℕ, ∑ k in finset.range n, (1 / (3^(k+1))) = geometric_series_sum (n-1) (1/3),
    from sorry,
  
  have T_eq_series : T = 3003 + (3002 / 3) + (3001 / (3^2)) + ... + (4 / (3^2999)) + (3 / (3^3000)),
    from sorry,
  
  have main_eq : T = 4504 - 1 / 4 - 1 / (4 * 3^3000),
    from sorry,

  show T = 4503.75,
    from sorry

end value_of_T_l129_129422


namespace log_expression_l129_129828

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression :
  log_base 4 16 - (log_base 2 3 * log_base 3 2) = 1 := by
  sorry

end log_expression_l129_129828


namespace ratio_of_evaluation_l129_129933

/-- 
In the final Chinese language exam, a teacher evaluated a composition based on three aspects: 
"relevance to the topic, language, and structure." The evaluation criteria were as follows: 
- Relevance to the topic, central focus, and rich content: 45%
- Smooth language with occasional errors: 25%
- Complete structure and clear organization: 30%
Prove that the ratio of the weights of "relevance to the topic, language, and structure" is 9:5:6.
-/
theorem ratio_of_evaluation (A B C : ℕ) (hA : A = 45) (hB : B = 25) (hC : C = 30) :
  (A / 5) : (B / 5) : (C / 5) = 9 : 5 : 6 :=
by
  sorry

end ratio_of_evaluation_l129_129933


namespace impossible_to_capture_all_in_one_move_l129_129979

-- Define the conditions of the problem
variables {black_checker white_checker : Type} 
variables (C : black_checker → black_checker → Type)
variables (board : ℕ → ℕ → Prop)
variables (adjacent_diagonal : ∀ a b c d : ℕ, board a b → board c d → (a ≠ c) → (b ≠ d) → 
                                                        (|a - c| = 1 ∧ |b - d| = 1))

-- Define the movement and capturing rules
variables (captures_in_one_move : white_checker → black_checker → board ℕ ℕ → Prop)
variables (initial_black_checkers : black_checker)
variables (num_additional_black_checkers : ℕ)

-- State the problem in Lean
theorem impossible_to_capture_all_in_one_move 
    (inf_chessboard : ∀ x y : ℕ, board x y) 
    (two_initial_black_checkers : ∃ x1 y1 x2 y2 : ℕ, adjacent_diagonal x1 y1 x2 y2) 
    (white_checker_capture_moves : captures_in_one_move white_checker black_checker board) :
    ¬ ∃ w : white_checker ∀ black_checker ∈ C, captures_in_one_move white_checker black_checker board :=
sorry

end impossible_to_capture_all_in_one_move_l129_129979


namespace count_squares_with_dot_l129_129461

theorem count_squares_with_dot (n : ℕ) (dot_center : (n = 5)) :
  n = 5 → ∃ k, k = 19 :=
by sorry

end count_squares_with_dot_l129_129461


namespace smallest_domain_x_of_fff_l129_129901

def f (x : ℝ) : ℝ := real.sqrt (3 * x - 4)

theorem smallest_domain_x_of_fff :
  (∀ x : ℝ, 3 * x - 4 ≥ 0 → 3 * real.sqrt(3 * x - 4) - 4 ≥ 0 → x ≥ 52 / 27) ∧
  (52 / 27 ≥ 4 / 3 ∧ ∀ x : ℝ, 3 * x - 4 ≥ 0 → x ≥ 52 / 27 ↔ 3 * x - 4 ≥ (4/3)^2) :=
sorry

end smallest_domain_x_of_fff_l129_129901


namespace cars_meet_in_two_hours_l129_129291

theorem cars_meet_in_two_hours (t : ℝ) (d : ℝ) (v1 v2 : ℝ) (h1 : d = 60) (h2 : v1 = 13) (h3 : v2 = 17) (h4 : v1 * t + v2 * t = d) : t = 2 := 
by
  sorry

end cars_meet_in_two_hours_l129_129291


namespace range_of_a_l129_129516

def f (x a : ℝ) := x^2 - 2 * a * x + 4

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x → f' x a ≥ 0) ↔ a ≤ 0 :=
by
  intros
  sorry

end range_of_a_l129_129516


namespace sum_of_three_integers_with_product_5_pow_4_l129_129262

noncomputable def a : ℕ := 1
noncomputable def b : ℕ := 5
noncomputable def c : ℕ := 125

theorem sum_of_three_integers_with_product_5_pow_4 (h : a * b * c = 5^4) : 
  a + b + c = 131 := by
  have ha : a = 1 := rfl
  have hb : b = 5 := rfl
  have hc : c = 125 := rfl
  rw [ha, hb, hc, mul_assoc] at h
  exact sorry

end sum_of_three_integers_with_product_5_pow_4_l129_129262


namespace problem_statement_l129_129652

theorem problem_statement :
  let prod_term : ℕ -> ℚ := λ (k : ℕ) => 1 - 1 / (2^k - 1)
  let telescoping_product : ℚ := (∏ (k : ℕ) in finset.range 28, prod_term (k+2))
  let m : ℕ := 2^28
  let n : ℕ := 2^29 - 1
  (telescoping_product = m / n) ∧ nat.gcd m n = 1 ∧ (2 * m - n = 1) :=
by {
  sorry
}

end problem_statement_l129_129652


namespace opposite_of_neg_three_fourths_l129_129640

theorem opposite_of_neg_three_fourths : ∃ x : ℚ, -3 / 4 + x = 0 ∧ x = 3 / 4 :=
by
  use 3 / 4
  split
  . norm_num
  . refl

end opposite_of_neg_three_fourths_l129_129640


namespace solve_inequality_f_ge_x_no_positive_a_b_satisfy_conditions_l129_129094

noncomputable def f (x : ℝ) : ℝ :=
  |2 * x - 1| - |2 * x - 2|

theorem solve_inequality_f_ge_x :
  {x : ℝ | f x >= x} = {x : ℝ | x <= -1 ∨ x = 1} :=
by sorry

theorem no_positive_a_b_satisfy_conditions :
  ∀ (a b : ℝ), a > 0 → b > 0 → (a + 2 * b = 1) → (2 / a + 1 / b = 4 - 1 / (a * b)) → false :=
by sorry

end solve_inequality_f_ge_x_no_positive_a_b_satisfy_conditions_l129_129094


namespace area_reduction_is_correct_l129_129380

-- Define the original area of the equilateral triangle
def original_area := 100 * Real.sqrt 3

-- Define the reduction in side length of the triangle
def side_reduction := 6

-- Calculate the side length of the original equilateral triangle
noncomputable def original_side_length : ℝ := Real.sqrt (4 * original_area / Real.sqrt 3)

-- Define the new side length after reduction
def new_side_length := original_side_length - side_reduction

-- Define the area of an equilateral triangle given its side length
noncomputable def area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

-- Calculate the new area after the side length reduction
noncomputable def new_area : ℝ := area new_side_length

-- The decrease in area of the equilateral triangle
noncomputable def area_decrease : ℝ := original_area - new_area

-- The proof statement showing the decrease in area is 51√3 cm²
theorem area_reduction_is_correct : area_decrease = 51 * Real.sqrt 3 := 
by sorry

end area_reduction_is_correct_l129_129380


namespace non_hot_peppers_count_l129_129159

-- Define the number of peppers Joel picks each day
def peppers_sunday : ℕ := 7
def peppers_monday : ℕ := 12
def peppers_tuesday : ℕ := 14
def peppers_wednesday : ℕ := 12
def peppers_thursday : ℕ := 5
def peppers_friday : ℕ := 18
def peppers_saturday : ℕ := 12

-- Define the fraction of hot peppers
def fraction_hot_peppers : ℚ := 0.20

-- Define the total number of peppers
def total_peppers : ℕ := 
  peppers_sunday + peppers_monday + peppers_tuesday + 
  peppers_wednesday + peppers_thursday + peppers_friday + peppers_saturday

-- Prove that the number of non-hot peppers picked by Joel is 64
theorem non_hot_peppers_count : (total_peppers * (1 - fraction_hot_peppers)).toInt = 64 := by
  sorry

end non_hot_peppers_count_l129_129159


namespace posters_buy_count_l129_129997

/-- Shelby had $60 initially, bought a $15 book, a $9 book, a $3.5 bookmark, 
a $4.8 set of pencils, a $6.2 notebook, applied relevant discounts and a $5 coupon, 
considered an 8% sales tax, and needs to determine how many $6 posters she can buy. -/
theorem posters_buy_count :
  let initial_amount := 60
  let book1 := 15
  let book2 := 9
  let coupon := 5
  let bookmark := 3.5
  let pencils := 4.8
  let notebook := 6.2
  let poster_price := 6
  let tax_rate := 0.08
  let books_total := book1 + book2
  let books_after_coupon := books_total - coupon
  let additional_items_total := bookmark + pencils + notebook
  let subtotal_before_discount := books_after_coupon + additional_items_total
  let discount := if subtotal_before_discount > 25 then 0.10 * subtotal_before_discount else 0
  let subtotal_after_discount := subtotal_before_discount - discount
  let total_after_tax := subtotal_after_discount * (1 + tax_rate)
  let money_left := initial_amount - total_after_tax
  let posters_count := Int.floor (money_left / poster_price)
  posters_count = 4 :=
by
  sorry

end posters_buy_count_l129_129997


namespace place_signs_correct_l129_129610

theorem place_signs_correct :
  1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99 :=
by
  sorry

end place_signs_correct_l129_129610


namespace volume_intersection_l129_129723

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

def region1 (x y z : ℝ) : Prop := abs x + abs y + abs z ≤ 1
def region2 (x y z : ℝ) : Prop := abs x + abs y + abs (z - 2) ≤ 1

theorem volume_intersection : 
  (volume {p : ℝ × ℝ × ℝ | region1 p.1 p.2 p.3 ∧ region2 p.1 p.2 p.3}) = (1 / 12 : ℝ) :=
by
  sorry

end volume_intersection_l129_129723


namespace isosceles_base_length_l129_129227

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l129_129227


namespace problem1_problem2_l129_129184

-- Definitions based on conditions
variables {A B C D E F : Type*}
variables {a b c : ℝ}
variables [triangle : scalene_triangle A B C]
variables [angle_bisectors D E F]
variables [DE_eq_DF : DE = DF]

-- Prove fraction equality
theorem problem1 (h₁ : triangle A B C) (h₂ : angle_bisectors D E F) (h₃ : DE = DF) :
  (a / (b + c) = b / (c + a) + c / (a + b)) :=
sorry

-- Prove angle inequality
theorem problem2 (h₁ : triangle A B C) (h₂ : angle_bisectors D E F) (h₃ : DE = DF) :
  ∠A > 90 :=
sorry

end problem1_problem2_l129_129184


namespace expected_value_decagonal_die_l129_129311

-- Given conditions
def decagonal_die_faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def probability (n : ℕ) : ℚ := 1 / 10

-- The mathematical proof problem (statement only, no proof required)
theorem expected_value_decagonal_die : 
  (List.sum decagonal_die_faces : ℚ) / List.length decagonal_die_faces = 5.5 := by
  sorry

end expected_value_decagonal_die_l129_129311


namespace least_possible_value_l129_129258

theorem least_possible_value (a b c d : ℕ)
  (h_abcd_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_a_pos : 1 ≤ a) (h_a_max : a ≤ 10)
  (h_b_pos : 1 ≤ b) (h_b_max : b ≤ 10)
  (h_c_pos : 1 ≤ c) (h_c_max : c ≤ 10)
  (h_d_pos : 1 ≤ d) (h_d_max : d ≤ 10) :
  (a = 1 ∧ b = 9 ∧ c = 2 ∧ d = 10) ∨
  (a = 2 ∧ b = 10 ∧ c = 1 ∧ d = 9) →
  (a:ℚ / b + c:ℚ / d = 14 / 45) :=
by sorry

end least_possible_value_l129_129258


namespace integral_ineq_l129_129998

theorem integral_ineq (n : ℕ) (h : 0 < n) : 
  ∫ x in 0..1, 1 / (1 + x^n) > 1 - 1 / n := 
sorry

end integral_ineq_l129_129998


namespace incorrect_guess_l129_129699

-- Define the conditions
def bears : ℕ := 1000

inductive Color
| White
| Brown
| Black

constant bear_color : ℕ → Color -- The color of the bear at each position

axiom condition : ∀ n : ℕ, n < bears - 2 → 
  ∃ i j k, (i, j, k ∈ {Color.White, Color.Brown, Color.Black}) ∧ 
  (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ 
  (bear_color n = i ∧ bear_color (n+1) = j ∧ bear_color (n+2) = k) 

constants (g1 : bear_color 2 = Color.White)
          (g2 : bear_color 20 = Color.Brown)
          (g3 : bear_color 400 = Color.Black)
          (g4 : bear_color 600 = Color.Brown)
          (g5 : bear_color 800 = Color.White)

-- The proof problem
theorem incorrect_guess : bear_color 20 ≠ Color.Brown :=
by sorry

end incorrect_guess_l129_129699


namespace log_product_eq_three_l129_129532

theorem log_product_eq_three (k x : ℝ) (hk : log k x * log 5 k = 3) : x = 125 := 
sorry

end log_product_eq_three_l129_129532


namespace exists_sum_of_three_l129_129636

theorem exists_sum_of_three {a b c d : ℕ} 
  (h1 : Nat.Coprime a b) 
  (h2 : Nat.Coprime a c) 
  (h3 : Nat.Coprime a d)
  (h4 : Nat.Coprime b c) 
  (h5 : Nat.Coprime b d) 
  (h6 : Nat.Coprime c d) 
  (h7 : a * b + c * d = a * c - 10 * b * d) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ 
           (y = a ∨ y = b ∨ y = c ∨ y = d) ∧ 
           (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ 
           x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ 
           (x = y + z ∨ y = x + z ∨ z = x + y) :=
by
  sorry

end exists_sum_of_three_l129_129636


namespace incorrect_guess_20_l129_129676

-- Define the assumptions and conditions
def bears : Nat → String := sorry -- function that determines the color of the bear at position n
axiom bears_color_constraint : ∀ n:Nat, exists b:List String, b.length = 3 ∧ Set ("W" "B" "Bk") = List.toSet b ∧ 
  List.all (List.sublist b (n, n+1, n+2) bears = fun c=> c = "W" or c = "B" or c = "Bk") 

-- Iskander's guesses
def guess1 := (2, "W")
def guess2 := (20, "B")
def guess3 := (400, "Bk")
def guess4 := (600, "B")
def guess5 := (800, "W")

-- Function to check the bear at each position
def check_bear (n:Nat) : String := sorry

-- Iskander's guess correctness, exactly one is wrong
axiom one_wrong : count (check_bear 2 =="W") 
                         + count (check_bear 20 == "B") 
                         + count (check_bear 400 =="Bk") 
                         + count (check_bear 600 =="B") 
                         + count (check_bear 800 =="W") = 4

-- Prove that the guess for the 20th bear is incorrect
theorem incorrect_guess_20 : ∀ {n:Nat} (h : n=20), (check_bear n != "B") := sorry

end incorrect_guess_20_l129_129676


namespace cost_of_superman_game_l129_129289

theorem cost_of_superman_game : 
  ∀ (cost_batman total_spent: ℝ), 
  cost_batman = 13.6 → 
  total_spent = 18.66 → 
  (total_spent - cost_batman = 5.06) :=
by
  intros cost_batman total_spent h_cost_batman h_total_spent
  have h := calc
    total_spent - cost_batman
      = 18.66 - 13.6 : by rw [h_total_spent, h_cost_batman]
      = 5.06 : by norm_num
  exact h

end cost_of_superman_game_l129_129289


namespace grid_cut_990_l129_129424

theorem grid_cut_990 (grid : Matrix (Fin 1000) (Fin 1000) (Fin 2)) :
  (∃ (rows_to_remove : Finset (Fin 1000)), rows_to_remove.card = 990 ∧ 
   ∀ col : Fin 1000, ∃ row ∈ (Finset.univ \ rows_to_remove), grid row col = 1) ∨
  (∃ (cols_to_remove : Finset (Fin 1000)), cols_to_remove.card = 990 ∧ 
   ∀ row : Fin 1000, ∃ col ∈ (Finset.univ \ cols_to_remove), grid row col = 0) :=
sorry

end grid_cut_990_l129_129424


namespace correct_propositions_l129_129509

-- Proposition ①: The graph of f(x) is symmetric about x=1 if for all x in ℝ, f(x - 1) = f(x + 1)
def prop1 (f : ℝ → ℝ) : Prop := 
  (∀ x : ℝ, f(x - 1) = f(x + 1)) → ∀ x : ℝ, f(x) = f(2 - x)

-- Proposition ②: If f(x) is odd, then the graph of f(x-1) is symmetric about (1, 0)
def prop2 (f : ℝ → ℝ) : Prop := 
  (∀ x : ℝ, f(-x) = -f(x)) → ∀ x : ℝ, f(x - 1) = f(2 - x)

-- Proposition ③: If f(x+1) + f(1-x) = 0, then the graph of f(x) is symmetric about (1, 0)
def prop3 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f(x + 1) + f(1 - x) = 0) → ∀ x : ℝ, f(x) = -f(2 - x)

-- Proposition ④: The graph of f(x-1) is symmetric to the graph of f(1-x) about the y-axis
def prop4 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x - 1) = f(-(1 - x))

-- The main theorem, stating the correct propositions are ② and ③, and disproving ① and ④
theorem correct_propositions (f : ℝ → ℝ) :
  ¬ prop1 f ∧ prop2 f ∧ prop3 f ∧ ¬ prop4 f :=
sorry

end correct_propositions_l129_129509


namespace each_boy_makes_14_l129_129296

/-- Proof that each boy makes 14 dollars given the initial conditions and sales scheme. -/
theorem each_boy_makes_14 (victor_shrimp : ℕ)
                          (austin_shrimp : ℕ)
                          (brian_shrimp : ℕ)
                          (total_shrimp : ℕ)
                          (sets_sold : ℕ)
                          (total_earnings : ℕ)
                          (individual_earnings : ℕ)
                          (h1 : victor_shrimp = 26)
                          (h2 : austin_shrimp = victor_shrimp - 8)
                          (h3 : brian_shrimp = (victor_shrimp + austin_shrimp) / 2)
                          (h4 : total_shrimp = victor_shrimp + austin_shrimp + brian_shrimp)
                          (h5 : sets_sold = total_shrimp / 11)
                          (h6 : total_earnings = sets_sold * 7)
                          (h7 : individual_earnings = total_earnings / 3):
  individual_earnings = 14 := 
by
  sorry

end each_boy_makes_14_l129_129296


namespace profit_margin_l129_129352

variables (C S n : ℝ)
def M (C S n : ℝ) := (1 / (2 * n)) * C + (1 / (3 * n)) * S

theorem profit_margin (h1 : S = 3 * C) : M C S n = S / (2 * n) :=
by
  sorry

end profit_margin_l129_129352


namespace incorrect_guess_20_l129_129670

-- Define the assumptions and conditions
def bears : Nat → String := sorry -- function that determines the color of the bear at position n
axiom bears_color_constraint : ∀ n:Nat, exists b:List String, b.length = 3 ∧ Set ("W" "B" "Bk") = List.toSet b ∧ 
  List.all (List.sublist b (n, n+1, n+2) bears = fun c=> c = "W" or c = "B" or c = "Bk") 

-- Iskander's guesses
def guess1 := (2, "W")
def guess2 := (20, "B")
def guess3 := (400, "Bk")
def guess4 := (600, "B")
def guess5 := (800, "W")

-- Function to check the bear at each position
def check_bear (n:Nat) : String := sorry

-- Iskander's guess correctness, exactly one is wrong
axiom one_wrong : count (check_bear 2 =="W") 
                         + count (check_bear 20 == "B") 
                         + count (check_bear 400 =="Bk") 
                         + count (check_bear 600 =="B") 
                         + count (check_bear 800 =="W") = 4

-- Prove that the guess for the 20th bear is incorrect
theorem incorrect_guess_20 : ∀ {n:Nat} (h : n=20), (check_bear n != "B") := sorry

end incorrect_guess_20_l129_129670


namespace incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129658

def is_color (col : String) (pos : Nat) : Prop := sorry
def valid_guesses : Prop :=
  (is_color "white" 2) ∧
  (is_color "brown" 20) ∧
  (is_color "black" 400) ∧
  (is_color "brown" 600) ∧
  (is_color "white" 800)

theorem incorrect_guess_at_20 :
  (∃ x, (x ∈ [2, 20, 400, 600, 800]) ∧ ¬ is_color_correct x) :=
begin
  sorry -- proof is not required
end

/-- Main theorem to identify the incorrect guess position. -/
theorem Iskander_incorrect_guess_20 :
  valid_guesses →
  (∃! x ∈ [2, 20, 400, 600, 800], ¬ is_color_correct x) →
  ¬ is_color "brown" 20 :=
begin
  admit -- proof is not required
end

end incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129658


namespace parallelogram_area_l129_129931

theorem parallelogram_area (A B C D : Point) (BD : ℝ) (angleC : ℝ) (is_tangent : Bool) 
  (hBD : BD = 2) (hC : angleC = 45) (hTang : is_tangent = true) : 
  parallelogram A B C D → 
  area A B C D = 4 := 
sorry

end parallelogram_area_l129_129931


namespace graph_movement_l129_129110

noncomputable def f (x : ℝ) : ℝ := -2 * (x - 1) ^ 2 + 3

noncomputable def g (x : ℝ) : ℝ := -2 * x ^ 2

theorem graph_movement :
  ∀ (x y : ℝ),
  y = f x →
  g x = y → 
  (∃ Δx Δy, Δx = -1 ∧ Δy = -3 ∧ g (x + Δx) = y + Δy) :=
by
  sorry

end graph_movement_l129_129110


namespace volume_of_intersection_is_zero_l129_129733

-- Definition of the regions
def region1 (x y z : ℝ) : Prop := abs x + abs y + abs z ≤ 1
def region2 (x y z : ℝ) : Prop := abs x + abs y + abs (z - 2) ≤ 1

-- Volume of the intersection of region1 and region2
theorem volume_of_intersection_is_zero : 
  let volume_intersection : ℝ := 0 
  in volume_intersection = 0 := 
by
  sorry

end volume_of_intersection_is_zero_l129_129733


namespace find_annual_interest_rate_l129_129795

theorem find_annual_interest_rate (P CI t n : ℝ) (hP : P = 700) (hCI : CI = 147.0000000000001) (ht : t = 2) (hn : n = 1) :
  let A := P + CI in
  let r := (real.sqrt (A / P) - 1) in
  r = 0.1 :=
by
  sorry

end find_annual_interest_rate_l129_129795


namespace volume_of_T_l129_129647

def T (x y z : ℝ) : Prop :=
  abs x + abs y ≤ 2 ∧ abs x + abs z ≤ 1 ∧ abs z + abs y ≤ 1

def volume_T : ℝ := ∑ i in finrange 8, 1 / 6

theorem volume_of_T : ∫ T.volume = 4 / 3 :=
sorry

end volume_of_T_l129_129647


namespace isosceles_triangle_base_length_l129_129245

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l129_129245


namespace regular_polygon_sides_l129_129021

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l129_129021


namespace angle_CDE_eq_angle_CED_l129_129941

noncomputable def triangle {α : Type*} [nontrivial α] := 
  {a b c : α // a ≠ b ∧ b ≠ c ∧ a ≠ c}

variables {α : Type*} [nontrivial α] {A B C T : α}
variables (AD BE : α)
variable  (is_triangle : triangle α)
variable (altitude : ∃ h : α, h = T)
variables (right_tr_CAD : ∃ a b c : α, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ AD = T)
variables (right_tr_CBE : ∃ a b c : α, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ BE = T)
variables (AD_eq_TB : AD = T)
variables (BE_eq_TA : BE = T)

theorem angle_CDE_eq_angle_CED :
  ∠ A D E = ∠ A E D :=
sorry

end angle_CDE_eq_angle_CED_l129_129941


namespace linear_in_one_variable_linear_in_two_variables_l129_129868

namespace MathProof

-- Definition of the equation
def equation (k x y : ℝ) : ℝ := (k^2 - 1) * x^2 + (k + 1) * x + (k - 7) * y - (k + 2)

-- Theorem for linear equation in one variable
theorem linear_in_one_variable (k : ℝ) (x y : ℝ) :
  k = -1 → equation k x y = 0 → ∃ y' : ℝ, equation k 0 y' = 0 :=
by
  sorry

-- Theorem for linear equation in two variables
theorem linear_in_two_variables (k : ℝ) (x y : ℝ) :
  k = 1 → equation k x y = 0 → ∃ x' y' : ℝ, equation k x' y' = 0 :=
by
  sorry

end MathProof

end linear_in_one_variable_linear_in_two_variables_l129_129868


namespace total_distance_walked_l129_129328

theorem total_distance_walked 
  (d1 : ℝ) (d2 : ℝ)
  (h1 : d1 = 0.75)
  (h2 : d2 = 0.25) :
  d1 + d2 = 1 :=
by
  sorry

end total_distance_walked_l129_129328


namespace difference_between_extremes_l129_129362

noncomputable def iterative_average_process (seq : List ℚ) : ℚ :=
  let mean := λ a b : ℚ => (a + b) / 2
  let res1 := mean seq[0] seq[1]
  let res2 := mean res1 seq[2]
  let res3 := mean res2 seq[3]
  let res4 := mean res3 seq[4]
  let res5 := mean res4 2
  mean res5 seq[5]

#eval iterative_average_process [6, 5, 4, 3, 2, 1] -- yields 131/32
#eval iterative_average_process [1, 2, 3, 4, 5, 6] -- yields 201/32

theorem difference_between_extremes : 
  let s1 := iterative_average_process [6, 5, 4, 3, 2, 1]
  let s2 := iterative_average_process [1, 2, 3, 4, 5, 6]
  s2 - s1 = 35 / 16 := by
  sorry

end difference_between_extremes_l129_129362


namespace isosceles_triangle_base_length_l129_129237

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l129_129237


namespace isosceles_triangle_base_length_l129_129242

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l129_129242


namespace midpoints_and_projections_same_circle_l129_129987
open EuclideanGeometry

-- Define points A, B, C, D and point P
variables {A B C D P : Point}

-- Midpoints of the sides of a quadrilateral
variables {K L M N : Point} (hK : midpoint A B K) (hL : midpoint C D L) 
(hM : midpoint B C M) (hN : midpoint D A N)

-- Projections of P onto the sides
variables {PA PB PC PD : Line} 
(hPA: projection PA P A)
(hPB: projection PB P B)
(hPC: projection PC P C)
(hPD: projection PD P D)

-- Proof of the main statement: midpoints and projections lie on the same circle
theorem midpoints_and_projections_same_circle :
  ∃ (O : Point) (r : ℝ), circle O r [K, L, M, N, PA, PB, PC, PD] := sorry

end midpoints_and_projections_same_circle_l129_129987


namespace max_f_when_m_2_monotonicity_f_interval_0_1_range_x1_plus_x2_l129_129871

section

variables {f : ℝ → ℝ}
variables {m : ℝ} (h_m_gt_0 : m > 0)
variables {x1 x2 : ℝ}

noncomputable def f := λ x : ℝ, (m + 1/m) * Real.log x + 1/x - x

-- Condition 1: Maximum value when m = 2
theorem max_f_when_m_2 : 
  (∃ x, f x = (5/2 : ℝ) * Real.log 2 - (3/2 : ℝ)) := 
sorry

-- Condition 2: Monotonicity of f in (0, 1)
theorem monotonicity_f_interval_0_1 (h_m: m > 0) : 
  if 0 < m ∧ m < 1 then 
    (∀ x, 0 < x ∧ x < m → f' x < 0) ∧ (∀ x, m < x ∧ x < 1 → f' x > 0)
  else if m = 1 then 
    (∀ x, 0 < x ∧ x < 1 → f' x < 0)
  else 
    (∀ x, 0 < x ∧ x < 1/m → f' x < 0) ∧ (∀ x, 1/m < x ∧ x < 1 → f' x > 0) := 
sorry

-- Condition 3: Range of x1 + x2 for m ∈ [3, +∞)
theorem range_x1_plus_x2 (h_m_in : 3 ≤ m) (h_x1_ne: x1 ≠ x2) :
  x1 + x2 > 6/5 := 
sorry

end

end max_f_when_m_2_monotonicity_f_interval_0_1_range_x1_plus_x2_l129_129871


namespace baseball_card_decrease_first_year_l129_129350

theorem baseball_card_decrease_first_year (x : ℝ) 
  (h1 : ∃ y ∈ {10}, y = 10) 
  (h2 : ∃ z ∈ {37}, z = 37) 
  (h3 : (100 : ℝ) * (1 - x / 100) * 0.9 = 63) : 
  x = 30 :=
sorry

end baseball_card_decrease_first_year_l129_129350


namespace find_100th_integer_l129_129759

def initial_board : Set ℕ := {1, 2, 4, 6}

def next_integer (board : Set ℕ) (n : ℕ) : Prop :=
  n > (board.to_finset.sup id) ∧ (∀ a b ∈ board, a ≠ b → n ≠ a + b)

theorem find_100th_integer :
  let board := {1, 2, 4, 6} in
  ∃ k : ℕ, k = 100 ∧
  let integers := (List.range k).map (λ i, some (Classical.some (next_integer i))) in
  integers.nth 99 = some 772 :=
by
  sorry

end find_100th_integer_l129_129759


namespace distance_to_origin_of_z_l129_129144

open Complex

-- Define the complex number z as 2 / (1 + I)
def z : ℂ := 2 / (1 + I)

-- Define the calculation of the distance from a complex number to the origin
def distance_to_origin (c : ℂ) : ℝ := Complex.abs c

-- Statement to prove
theorem distance_to_origin_of_z : distance_to_origin z = Real.sqrt 2 := by
  sorry

end distance_to_origin_of_z_l129_129144


namespace sin_B_value_dot_product_min_value_l129_129128

-- Given conditions
variables {ABC : Type} [triangle ABC]
variable (BC : real)
variable (cos_B : real)

-- Assume given conditions hold
axiom BC_value : BC = 4
axiom cos_B_value : cos_B = 1 / 4

-- Statement: Prove $\sin B = \frac{\sqrt{15}}{4}$ given conditions
theorem sin_B_value (BC_value : BC = 4) (cos_B_value : cos_B = 1 / 4) :
  sin_B = (sqrt 15) / 4 := 
sorry

-- Statement: Prove the minimum value of $\overrightarrow{AB} \cdot \overrightarrow{AC}$ is $-\frac{1}{4}$ given conditions
theorem dot_product_min_value (BC_value : BC = 4) (cos_B_value : cos_B = 1 / 4) :
  min_dot_product_AB_AC = -1 / 4 :=
sorry

end sin_B_value_dot_product_min_value_l129_129128


namespace directrix_of_parabola_l129_129440

-- Define the parabola function
def parabola (x : ℝ) : ℝ := (x^2 - 8*x + 12) / 16

theorem directrix_of_parabola : 
  ∃ y : ℝ, (∀ x : ℝ, parabola x = y) → y = -17 / 4 := 
by
  sorry

end directrix_of_parabola_l129_129440


namespace no_more_than_four_obtuse_vectors_l129_129202

theorem no_more_than_four_obtuse_vectors :
  ∀ (v : ℕ) (vectors : Fin v → EuclideanSpace ℝ (Fin 3)), 
    v > 4 →
    (∀ i j : Fin v, i ≠ j → dot_product (vectors i) (vectors j) < 0) → False :=
by sorry

end no_more_than_four_obtuse_vectors_l129_129202


namespace range_of_m_l129_129856

def isEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then x - x^2 else (2 - x) / Real.exp x

theorem range_of_m (m : ℝ) : 
  isEven f → 
  (∀ x, f x - m = 0 → x ∈ Set.Icc 0 ⊤) → 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = m ∧ f b = m ∧ f c = m) ↔ 
  -1 / Real.exp 3 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l129_129856


namespace ceil_evaluation_l129_129816

theorem ceil_evaluation :
  (let x := (-7 / 4) ^ 2 in ⌈x⌉ = 4) :=
by
  sorry

end ceil_evaluation_l129_129816


namespace problem_statement_l129_129118

theorem problem_statement :
  ∀ (x : ℝ),
    (5 * x - 10 = 15 * x + 5) →
    (5 * (x + 3) = 15 / 2) :=
by
  intros x h
  sorry

end problem_statement_l129_129118


namespace totalPeaches_l129_129282

-- Definition of conditions in the problem
def redPeaches : Nat := 4
def greenPeaches : Nat := 6
def numberOfBaskets : Nat := 1

-- Mathematical proof problem
theorem totalPeaches : numberOfBaskets * (redPeaches + greenPeaches) = 10 := by
  sorry

end totalPeaches_l129_129282


namespace probability_of_selecting_greater_than_1_point_5_l129_129315

theorem probability_of_selecting_greater_than_1_point_5 :
  let interval := set.Icc 1 3
  let subinterval := set.Icc 1.5 3
  let total_length := 3 - 1
  let subinterval_length := 3 - 1.5
  let probability := subinterval_length / total_length
  probability = 0.75 :=
by
  let interval := set.Icc 1 3
  let subinterval := set.Icc 1.5 3
  let total_length := 3 - 1
  let subinterval_length := 3 - 1.5
  let probability := subinterval_length / total_length
  have h1 : total_length = 2 := by calc 3 - 1 = 2
  have h2 : subinterval_length = 1.5 := by calc 3 - 1.5 = 1.5
  have h3 : probability = subinterval_length / total_length := rfl
  rw [h1, h2] at h3
  exact h3

end probability_of_selecting_greater_than_1_point_5_l129_129315


namespace a_general_correct_T_seq_correct_A_seq_bounded_min_m_correct_l129_129078

noncomputable def a_seq : ℕ → ℕ 
| 0     := 0
| (n+1) := a_seq n + 1

def S_seq (n : ℕ) : ℕ := ∑ i in finset.range n, a_seq (i + 1)

noncomputable def a_general (n : ℕ) : ℕ := 2^(n-1)

noncomputable def b_seq (n : ℕ) : ℚ := n / (4 * a_general n)

noncomputable def T_seq (n : ℕ) : ℚ := ∑ i in finset.range n, b_seq (i+1)

noncomputable def c_seq (k : ℕ) : ℚ := 
  (k + 2) / (S_seq k * (T_seq k + (k + 1)))

noncomputable def A_seq (n : ℕ) : ℚ := ∑ i in finset.range n, c_seq (i + 1)

theorem a_general_correct (n : ℕ) : a_seq (n + 1) = a_general (n + 1) := 
sorry

theorem T_seq_correct (n : ℕ) : T_seq n = 1 - (n + 2) / (2^(n + 1)) := 
sorry

theorem A_seq_bounded (n : ℕ) : A_seq n < 2 := 
sorry

theorem min_m_correct (m : ℕ) (h : ∀ n, A_seq n < m) : m ≥ 2 := 
sorry

end a_general_correct_T_seq_correct_A_seq_bounded_min_m_correct_l129_129078


namespace part1_part2_l129_129102

theorem part1 (a : ℝ) : (a - 3 ≠ 0) ∧ (16 - 4 * (a-3) * (-1) = 0) → 
  a = -1 ∧ ∀ x : ℝ, (4 * x^2 + 4 * x + 1 = 0 ↔ x = -1/2) :=
sorry

theorem part2 (a : ℝ) : (a - 3 ≠ 0) ∧ (16 - 4 * (a-3) * (-1) > 0) → 
  a > -1 ∧ a ≠ 3 :=
sorry

end part1_part2_l129_129102


namespace count_ordered_pairs_satisfying_equation_l129_129460

-- Ensure necessary imports are present for the problem context. 

theorem count_ordered_pairs_satisfying_equation :
  { (x, y) : ℕ × ℕ | 0 < x ∧ 0 < y ∧ 
    x * real.sqrt y + y * real.sqrt x + real.sqrt (2009 * x * y) - real.sqrt (2009 * x) - real.sqrt (2009 * y) - 2009 = 0 }.to_finset.card = 6 :=
by
  sorry

end count_ordered_pairs_satisfying_equation_l129_129460


namespace sum_odd_multiples_of_5_in_fibonacci_sequence_l129_129564

-- Define the set of positive integers from 1 to 150
def is_in_range (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 150

-- Define the Fibonacci sequence
def fibonacci_sequence (n : ℕ) : ℕ :=
  nat.rec_on n 0 (λ n', nat.cases_on n' 1 (λ n'' f1, nat.rec f1 (λ f2, f1 + f2) n''))

-- Define odd multiples of 5
def is_odd_multiple_of_5 (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 5 = 0

-- Prove the sum of all odd multiples of 5 that are Fibonacci numbers within the range 1 to 150
theorem sum_odd_multiples_of_5_in_fibonacci_sequence : 
  ∑ n in finset.filter (λ x, is_in_range x ∧ is_odd_multiple_of_5 x ∧ fibonacci_sequence x ≤ 150) (finset.range 151), id n = 60 := 
  by sorry

end sum_odd_multiples_of_5_in_fibonacci_sequence_l129_129564


namespace find_a_plus_b_l129_129063

theorem find_a_plus_b (a b : ℝ) :
  let A := {x : ℝ | x ^ 3 + 3 * x ^ 2 + 2 * x > 0},
      B := {x : ℝ | x ^ 2 + a * x + b ≤ 0}
  in 
    (A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2}) 
    ∧ (A ∪ B = {x : ℝ | x > -2}) 
    → (a + b = -3) :=
by
  intro A B h
  sorry

end find_a_plus_b_l129_129063


namespace sum_of_reciprocals_of_roots_l129_129470

theorem sum_of_reciprocals_of_roots (s₁ s₂ : ℝ) (h₀ : s₁ + s₂ = 15) (h₁ : s₁ * s₂ = 36) :
  (1 / s₁) + (1 / s₂) = 5 / 12 :=
by
  sorry

end sum_of_reciprocals_of_roots_l129_129470


namespace wrong_guess_is_20_l129_129663

-- Define the colors
inductive Color
| white
| brown
| black

-- Assume we have a sequence of 1000 bears
def bears : fin 1000 → Color := sorry

-- Hypotheses
axiom colors_per_three : ∀ (i : fin 998), 
  ({bears i, bears (i + 1), bears (i + 2)} = {Color.white, Color.brown, Color.black} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.black, Color.white, Color.brown} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.brown, Color.black, Color.white})

axiom exactly_one_wrong : 
  (bears 1 = Color.white ∧ bears 19 ≠ Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 ≠ Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 ≠ Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 ≠ Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 ≠ Color.white)

-- Define the theorem to prove
theorem wrong_guess_is_20 : 
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) →
  ¬(bears 19 = Color.brown) := 
sorry

end wrong_guess_is_20_l129_129663


namespace tiling_ways_2xn_eq_fib_l129_129552

def fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fib n + fib (n+1)

theorem tiling_ways_2xn_eq_fib (n : ℕ) : 
    ∃ (F : ℕ → ℕ), F 0 = 1 ∧ F 1 = 1 ∧ (∀ n, F (n + 2) = F n + F (n + 1)) ∧ F n = fib n := 
sorry

end tiling_ways_2xn_eq_fib_l129_129552


namespace white_is_lightest_l129_129279

namespace PuppyWeights

inductive Puppy
| White | Black | Yellowy | Spotted

open Puppy

variables (w : Ord White)
variables (b : Ord Black)
variables (y : Ord Yellowy)
variables (s : Ord Spotted)

def black_heavier_than_white_and_lighter_than_yellowy : Prop :=
  b > w ∧ b < y

def spotted_heavier_than_yellowy : Prop :=
  s > y

theorem white_is_lightest : 
  black_heavier_than_white_and_lighter_than_yellowy b w y ∧ spotted_heavier_than_yellowy s y → w == @Ord.min Puppy _ _ _ _ := 
by
  sorry

end PuppyWeights

end white_is_lightest_l129_129279


namespace cos_diff_l129_129500

theorem cos_diff (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.tan α = 2) : 
  Real.cos (α - π / 4) = 3 * Real.sqrt 10 / 10 :=
sorry

end cos_diff_l129_129500


namespace polynomial_solutions_l129_129811

theorem polynomial_solutions (P : Polynomial ℝ) :
  (∀ x : ℝ, P.eval (x^2) = x * P.eval x) ↔ (∃ λ : ℝ, P = Polynomial.C λ * Polynomial.X) :=
by
  sorry

end polynomial_solutions_l129_129811


namespace incorrect_guess_20_l129_129669

-- Define the assumptions and conditions
def bears : Nat → String := sorry -- function that determines the color of the bear at position n
axiom bears_color_constraint : ∀ n:Nat, exists b:List String, b.length = 3 ∧ Set ("W" "B" "Bk") = List.toSet b ∧ 
  List.all (List.sublist b (n, n+1, n+2) bears = fun c=> c = "W" or c = "B" or c = "Bk") 

-- Iskander's guesses
def guess1 := (2, "W")
def guess2 := (20, "B")
def guess3 := (400, "Bk")
def guess4 := (600, "B")
def guess5 := (800, "W")

-- Function to check the bear at each position
def check_bear (n:Nat) : String := sorry

-- Iskander's guess correctness, exactly one is wrong
axiom one_wrong : count (check_bear 2 =="W") 
                         + count (check_bear 20 == "B") 
                         + count (check_bear 400 =="Bk") 
                         + count (check_bear 600 =="B") 
                         + count (check_bear 800 =="W") = 4

-- Prove that the guess for the 20th bear is incorrect
theorem incorrect_guess_20 : ∀ {n:Nat} (h : n=20), (check_bear n != "B") := sorry

end incorrect_guess_20_l129_129669


namespace num_integers_a_satisfying_conditions_l129_129914

theorem num_integers_a_satisfying_conditions :
  (∀ a : ℤ, (∀ x : ℝ, (3 * x - a > 2 * (1 - x) ∧ (x - 1) / 2 ≥ (x + 2) / 3 - 1) → x ≥ 1) → 
    (∃ y : ℤ, ((y : ℝ) / (y + 1) + (a : ℝ) / (y - 1) = 1) → 
      (a < 3) → (y ≠ -1) → (y ≠ 1)) → 
  (a = 2 ∨ a = -1)) → 
  2 :=
by
  sorry

end num_integers_a_satisfying_conditions_l129_129914


namespace centers_form_equilateral_l129_129176

variables {A B C X Y Z : Type} [linear_ordered_field Type*] [ordered_ring Type*]

-- Assume A, B, and C are vertices of a triangle
def triangle_ABC (A B C : Type*) := ∃ A B C, true

-- Assume X, Y, and Z are such that triangles XBC, YCA, and ZAB are equilateral and exterior to ABC
def equilateral_XBC (X B C : Type*) := ∃ X B C, true ∧ (∃ θ, θ = 60) -- Assume to describe the angle property for simplicity
def equilateral_YCA (Y C A : Type*) := ∃ Y C A, true ∧ (∃ θ, θ = 60)
def equilateral_ZAB (Z A B : Type*) := ∃ Z A B, true ∧ (∃ θ, θ = 60)

-- The problem: Show that the points X, Y, and Z form an equilateral triangle
theorem centers_form_equilateral (A B C X Y Z : Type*) 
  (hABC : triangle_ABC A B C) 
  (hXBC : equilateral_XBC X B C) 
  (hYCA : equilateral_YCA Y C A) 
  (hZAB : equilateral_ZAB Z A B) : 
  ∃ t : Type*, t = (X = Y ∧ Y = Z ∧ Z = X) := 
sorry -- proof placeholder

end centers_form_equilateral_l129_129176


namespace No_of_boxes_in_case_l129_129484

-- Define the conditions
def George_has_total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def George_has_boxes : ℕ := George_has_total_blocks / blocks_per_box

-- The theorem to prove
theorem No_of_boxes_in_case : George_has_boxes = 2 :=
by
  sorry

end No_of_boxes_in_case_l129_129484


namespace paula_bought_fewer_cookies_l129_129606
-- Import the necessary libraries

-- Definitions
def paul_cookies : ℕ := 45
def total_cookies : ℕ := 87

-- Theorem statement
theorem paula_bought_fewer_cookies : ∃ (paula_cookies : ℕ), paul_cookies + paula_cookies = total_cookies ∧ paul_cookies - paula_cookies = 3 := by
  sorry

end paula_bought_fewer_cookies_l129_129606


namespace general_term_arithmetic_sum_first_n_terms_geometric_l129_129082

-- Definitions and assumptions based on given conditions
def a (n : ℕ) : ℤ := 2 * n + 1

-- Given conditions
def initial_a1 : ℤ := 3
def common_difference : ℤ := 2

-- Validate the general formula for the arithmetic sequence
theorem general_term_arithmetic : ∀ n : ℕ, a n = 2 * n + 1 := 
by sorry

-- Definitions and assumptions for geometric sequence
def b (n : ℕ) : ℤ := 3^n

-- Sum of the first n terms of the geometric sequence
def Sn (n : ℕ) : ℤ := 3 / 2 * (3^n - 1)

-- Validate the sum formula for the geometric sequence
theorem sum_first_n_terms_geometric (n : ℕ) : Sn n = 3 / 2 * (3^n - 1) := 
by sorry

end general_term_arithmetic_sum_first_n_terms_geometric_l129_129082


namespace logs_form_arith_prog_l129_129340

-- Conditions
variables {a b c r k : ℝ} (hk : k > 0 ∧ k ≠ 1) (h : b = a * r ∧ c = a * r^2)

-- Question and proof goal
theorem logs_form_arith_prog (hk : k > 0 ∧ k ≠ 1) (h : b = a * r ∧ c = a * r^2) :
  2 * log k b = log k a + log k c :=
  sorry

end logs_form_arith_prog_l129_129340


namespace initial_cars_l129_129283

theorem initial_cars (X : ℕ) : (X - 13 + (13 + 5) = 85) → (X = 80) :=
by
  sorry

end initial_cars_l129_129283


namespace isosceles_triangle_base_length_l129_129246

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l129_129246


namespace equal_areas_ABO_ODCE_l129_129549

variables {A B C D E O : Type} [OrderedRing A] [AffineSpace A B]

-- Assuming points D on BC and E on AC
variables (BC_line : Line B C) (AC_line : Line A C)
variable (D : Point BC_line)
variable (E : Point AC_line)

-- Midpoints definition and properties
variable (M : Midpoint A C)
variable (N : Midpoint B C)

-- AD and BE intersect at O
noncomputable instance : Inhabited O := ⟨O⟩  -- to use O in intersection
axiom AD_intersects_BE_at_O : intersection (lineThrough A D) (lineThrough B E) = O

-- Midline between midpoints M and N is parallel to AB and bisects DE
axiom MN_parallel_to_AB : parallel (lineThrough M N) (lineThrough A B)
axiom MN_bisects_DE : bisects (lineThrough M N) D E

-- Prove areas of triangle ABO and quadrilateral ODCE are equal
theorem equal_areas_ABO_ODCE : 
    area (triangle A B O) = area (quadrilateral O D C E) := 
begin
  sorry -- Proof is omitted
end

end equal_areas_ABO_ODCE_l129_129549


namespace average_rate_of_change_l129_129517

noncomputable def f (x : ℝ) : ℝ := x^2 + 1

theorem average_rate_of_change (Δx : ℝ) : 
  (f (1 + Δx) - f 1) / Δx = 2 + Δx := 
by
  sorry

end average_rate_of_change_l129_129517


namespace solve_for_a_l129_129471

def E (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := a * b^2 + c

theorem solve_for_a :
  let a := (-5 : ℚ) / 14 in
  2 * a + E a 3 2 = 4 + E a 5 3 :=
by
  let a := (-5 : ℚ) / 14
  sorry

end solve_for_a_l129_129471


namespace cos_double_angle_l129_129533

theorem cos_double_angle (α : ℝ) (h : sin α + 3 * sin (π / 2 + α) = 0) : cos (2 * α) = -4 / 5 :=
sorry

end cos_double_angle_l129_129533


namespace increasing_function_inequality_l129_129503

open Function

-- Define that f is an increasing function on real numbers
variable (f : ℝ → ℝ)

-- Define the conditions
variables (a b : ℝ)
hypothesis (H1 : StrictMono f)
hypothesis (H2 : a + b > 0)

-- Define the statement to be proved
theorem increasing_function_inequality : f(a) + f(b) > f(-a) + f(-b) :=
  by
  sorry

end increasing_function_inequality_l129_129503


namespace joe_eats_at_least_two_kinds_l129_129158

noncomputable def probability_at_least_two_kinds_of_fruit : ℚ :=
  1 - (3 * (1 / 3)^4)

theorem joe_eats_at_least_two_kinds :
  probability_at_least_two_kinds_of_fruit = 26 / 27 := 
by
  sorry

end joe_eats_at_least_two_kinds_l129_129158


namespace smallest_value_of_M_l129_129952

theorem smallest_value_of_M :
  ∀ (a b c d e f g M : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 → g > 0 →
  a + b + c + d + e + f + g = 2024 →
  M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g))))) →
  M = 338 :=
by
  intro a b c d e f g M ha hb hc hd he hf hg hsum hmax
  sorry

end smallest_value_of_M_l129_129952


namespace find_angle_B_l129_129567

noncomputable def angle_B (A B C : ℝ) (a b c : ℝ) :=
  ∃ A B C a b c,
    A = 2 * Real.pi / 3 ∧
    a = 3 ∧
    b = Real.sqrt 6 ∧
    (∀ A B C a b c, A + B + C = Real.pi) ∧
    (∀ A B C a b c, c = 0 ↔ A = 0 ∧ B = 0 ∧ C = 0) ∧
    (∀ A B C a b c, a = b ∧ b = c ∧ c ≠ 0 → A = B ∧ B = C) ∧
    B = Real.pi / 4

theorem find_angle_B : angle_B (2 * Real.pi / 3) (Real.pi / 4) C (3) (Real.sqrt 6) (c) := 
  sorry

end find_angle_B_l129_129567


namespace exists_polynomial_degree_n_l129_129580

theorem exists_polynomial_degree_n (n : ℕ) (hn : 0 < n) : 
  ∃ (ω ψ : Polynomial ℤ), ω.degree = n ∧ (ω^2 = (X^2 - 1) * ψ^2 + 1) := 
sorry

end exists_polynomial_degree_n_l129_129580


namespace construct_triangle_l129_129416

-- Definitions according to given conditions
variables (α : Real) (a1 : Real) (q : Real)

-- The main Lean theorem statement for the problem
theorem construct_triangle (α : Real) (a1 : Real) (q : Real) : 
  ∃ (a b c : Real), 
    -- The conditions specifying triangle construction is possible with the given constraints 
    ∃ (m_b m_c : Real), 
    b > 0 ∧ c > 0 ∧ b ≠ c ∧
    (b > c → m_b > m_c ∧ m_b * c = m_c * b ∧ m_b / m_c = q) ∧
    -- Angle α must be part of the triangle construction
    ∃ (triangle_exists α a b c),
    -- Segment relation a1 + a2 = a
    ∃ a2 : Real, a1 + a2 = a :=
sorry

end construct_triangle_l129_129416


namespace amount_left_after_pool_l129_129253

def amount_left (total_earned : ℝ) (cost_per_person : ℝ) (num_people : ℕ) : ℝ :=
  total_earned - (cost_per_person * num_people)

theorem amount_left_after_pool :
  amount_left 30 2.5 10 = 5 :=
by
  sorry

end amount_left_after_pool_l129_129253


namespace area_of_equilateral_triangle_l129_129707

theorem area_of_equilateral_triangle
  (DEF : Type) [equilateral_triangle DEF]
  (P : point)
  (center_of_inscribed_circle : P)
  (area_of_circle : Real)
  (h1 : center_of_inscribed_circle = P)
  (h2 : area_of_circle = 9 * Real.pi) :
  ∃ area_of_triangle : Real, area_of_triangle = 27 * Real.sqrt 3 := 
sorry

end area_of_equilateral_triangle_l129_129707


namespace minimum_S_n_value_l129_129490

noncomputable def S (n : ℕ) : ℕ :=
  ∑ k in finset.range 10, (k + 1) * (|n - (k + 1)|)

theorem minimum_S_n_value : ∃ (n : ℕ), n = 7 ∧ S n = 112 := 
by {
  sorry
}

end minimum_S_n_value_l129_129490


namespace incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129656

def is_color (col : String) (pos : Nat) : Prop := sorry
def valid_guesses : Prop :=
  (is_color "white" 2) ∧
  (is_color "brown" 20) ∧
  (is_color "black" 400) ∧
  (is_color "brown" 600) ∧
  (is_color "white" 800)

theorem incorrect_guess_at_20 :
  (∃ x, (x ∈ [2, 20, 400, 600, 800]) ∧ ¬ is_color_correct x) :=
begin
  sorry -- proof is not required
end

/-- Main theorem to identify the incorrect guess position. -/
theorem Iskander_incorrect_guess_20 :
  valid_guesses →
  (∃! x ∈ [2, 20, 400, 600, 800], ¬ is_color_correct x) →
  ¬ is_color "brown" 20 :=
begin
  admit -- proof is not required
end

end incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129656


namespace locus_of_points_for_given_angle_l129_129103

-- Define the basic geometric concepts
structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def angle (A B C : Point) : ℝ := sorry -- Angle function is complex to noncomputably define here

-- Define a condition for the specific angle of 70 degrees
def angle_70_degrees (A B M : Point) : Prop :=
  angle M A B = 70

-- The theorem statement
theorem locus_of_points_for_given_angle (A B : Point) :
  {M : Point | angle_70_degrees A B M} = 
  {M : Point | ∃ C D, angle A B C = 70 ∧ angle A B D = 70 ∧ (M ∈ line_through A C ∨ M ∈ line_through A D)} :=
sorry

end locus_of_points_for_given_angle_l129_129103


namespace inequality_solution_l129_129028

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1/4) ∧ (x - 2 > 0) → x > 2 :=
by {
  sorry
}

end inequality_solution_l129_129028


namespace math_group_probability_distribution_table_and_expectation_l129_129132

noncomputable section

def boys : Finset ℕ := {1, 2, 3, 4, 5, 6}
def girls : Finset ℕ := {7, 8, 9, 10}
def students : Finset ℕ := boys ∪ girls

def select_math_group (students : Finset ℕ) : Finset (Finset ℕ) :=
  students.powerset.filter (λ s, s.card = 5)

def includes_boy_A_not_girl_a (grp : Finset ℕ) : Prop :=
  1 ∈ grp ∧ 7 ∉ grp

def probability_math_group : ℚ :=
  ((select_math_group students).filter includes_boy_A_not_girl_a).card / (select_math_group students).card

theorem math_group_probability : probability_math_group = 5 / 18 := by
  sorry

def X_distribution : Finset (Fin ℕ → ℚ) :=
  Finset.univ.image (λ x, (select_math_group students).filter (λ grp, grp.filter (∈ girls).card = x))

def E_X : ℚ :=
  ∑ x in Finset.range 5, x * (select_math_group students).filter (λ grp, grp.filter (∈ girls).card = x).card / (select_math_group students).card

theorem distribution_table_and_expectation :
  (X_distribution = 
    {0, 1, 2, 3, 4} →
    (λ x, ({0, 1 / 42, 5 / 21, 10 / 21, 5 / 21, 1 / 42}).nth-le x (x.elim0_succ_succ)).sum = 1) ∧
  E_X = 2 := by
  sorry

end math_group_probability_distribution_table_and_expectation_l129_129132


namespace flower_cost_l129_129918

-- Given conditions
variables {x y : ℕ} -- costs of type A and type B flowers respectively

-- Costs equations
def cost_equation_1 : Prop := 3 * x + 4 * y = 360
def cost_equation_2 : Prop := 4 * x + 3 * y = 340

-- Given the necessary planted pots and rates
variables {m n : ℕ} (Hmn : m + n = 600) 
-- Percentage survivals
def survival_rate_A : ℚ := 0.70
def survival_rate_B : ℚ := 0.90

-- Replacement condition
def replacement_cond : Prop := (1 - survival_rate_A) * m + (1 - survival_rate_B) * n ≤ 100

-- Minimum cost condition
def min_cost (m_plant : ℕ) (n_plant : ℕ) : ℕ := 40 * m_plant + 60 * n_plant

theorem flower_cost 
  (H1 : cost_equation_1)
  (H2 : cost_equation_2)
  (H3 : x = 40)
  (H4 : y = 60) 
  (Hmn : m + n = 600)
  (Hsurv : replacement_cond) : 
  (m = 200 ∧ n = 400) ∧ 
  (min_cost 200 400 = 32000) := 
sorry

end flower_cost_l129_129918


namespace equal_roots_of_quadratic_l129_129014

noncomputable def number_of_real_p_with_equal_roots : ℕ :=
  2

theorem equal_roots_of_quadratic :
  ∃ p₁ p₂ : ℝ, 
  (p₁ ≠ p₂) ∧ 
  (∀ p ∈ ({p₁, p₂} : set ℝ), ∀ x : ℝ, x^2 + p * x + p = 0 ∧ discriminant 1 p p = 0) ∧
  (∀ p : ℝ, (∀ x : ℝ, x^2 + p * x + p = 0 ∧ discriminant 1 p p = 0) → p ∈ ({p₁, p₂} : set ℝ)) ∧
  ∃ n : ℕ, n = 2 ∧ n = number_of_real_p_with_equal_roots :=
sorry

end equal_roots_of_quadratic_l129_129014


namespace cube_edge_length_l129_129996

-- Define the given conditions
def point (a x y z: ℝ) (M: ℝ × ℝ × ℝ) (A B C D: ℝ × ℝ × ℝ) : Prop :=
  let ax := ((M.1 - A.1)^2 + (M.2 - A.2)^2 + (M.3 - A.3)^2 = 50) in
  let by := ((M.1 - B.1)^2 + (M.2 - B.2)^2 + (M.3 - B.3)^2 = 70) in
  let cz := ((M.1 - C.1)^2 + (M.2 - C.2)^2 + (M.3 - C.3)^2 = 90) in
  let dx := ((M.1 - D.1)^2 + (M.2 - D.2)^2 + (M.3 - D.3)^2 = 110) in
  let same_edge := (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (A ≠ D) in
  let vertices := ( (A.1 = M.1 ∨ A.2 = M.2 ∨ A.3 = M.3) ∧ 
                    (B.1 = a - M.1 ∨ B.2 = a - M.2 ∨ B.3 = M.3) ∧ 
                    (C.1 = a - M.1 ∨ C.2 = M.2 ∨ C.3 = a - M.3) ∧ 
                    (D.1 = M.1 ∨ D.2 = a - M.2 ∨ D.3 = a - M.3) ) in
  ax ∧ by ∧ cz ∧ dx ∧ same_edge ∧ vertices

-- Define the main theorem
theorem cube_edge_length (a x y z: ℝ) (M A B C D: ℝ × ℝ × ℝ) :
  point a x y z M A B C D → a = 10 :=
sorry

end cube_edge_length_l129_129996


namespace solution_proof_l129_129077

noncomputable def proof_problem : Prop :=
  ∀ (x y : ℝ), (1 / x + 1 / y = 4) ∧ (xy + x + y = 5) → (x^2 * y + x * y^2 = 4)

theorem solution_proof : proof_problem :=
by {
  intros x y h,
  obtain ⟨h1, h2⟩ := h,
  sorry
}

end solution_proof_l129_129077


namespace sticker_price_is_250_l129_129106

noncomputable def sticker_price_solution (x : ℝ) : Prop :=
  let final_price_A := 0.80 * x - 100
  let final_price_B := 0.70 * x - 50
  final_price_A = final_price_B - 25

theorem sticker_price_is_250 : ∃ x : ℝ, sticker_price_solution x ∧ x = 250 :=
by {
  use 250,
  simp [sticker_price_solution],
  norm_num,
  exact calc
    0.80 * 250 - 100 = 200 - 100 : by ring
                   ... = 100 : by ring
    0.70 * 250 - 50 - 25 = 175 - 50 - 25 : by ring
                       ... = 100 : by ring,
  sorry
}

end sticker_price_is_250_l129_129106


namespace identifyIncorrectGuess_l129_129677

-- Define the colors of the bears
inductive BearColor
| white
| brown
| black

-- Conditions as defined in the problem statement
def isValidBearRow (bears : Fin 1000 → BearColor) : Prop :=
  ∀ (i : Fin 998), 
    (bears i = BearColor.white ∨ bears i = BearColor.brown ∨ bears i = BearColor.black) ∧
    (bears ⟨i + 1, by linarith⟩ = BearColor.white ∨ bears ⟨i + 1, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 1, by linarith⟩ = BearColor.black) ∧
    (bears ⟨i + 2, by linarith⟩ = BearColor.white ∨ bears ⟨i + 2, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 2, by linarith⟩ = BearColor.black)

-- Iskander's guesses
def iskanderGuesses (bears : Fin 1000 → BearColor) : Prop :=
  bears 1 = BearColor.white ∧
  bears 19 = BearColor.brown ∧
  bears 399 = BearColor.black ∧
  bears 599 = BearColor.brown ∧
  bears 799 = BearColor.white

-- Exactly one guess is incorrect
def oneIncorrectGuess (bears : Fin 1000 → BearColor) : Prop :=
  ∃ (idx : Fin 5), 
    ¬iskanderGuesses bears ∧
    ∀ (j : Fin 5), (j ≠ idx → (bearGuessesIdx j bears = true))

-- The proof problem
theorem identifyIncorrectGuess (bears : Fin 1000 → BearColor) :
  isValidBearRow bears → iskanderGuesses bears → oneIncorrectGuess bears := sorry

end identifyIncorrectGuess_l129_129677


namespace find_natural_numbers_l129_129048

def divisors (n m : ℕ) : Prop := m ∣ n
def is_prime (n : ℕ) : Prop := nat.prime n

theorem find_natural_numbers (a b : ℕ) :
  (a = 3 ∧ b = 1) ∨ (a = 7 ∧ b = 2) ∨ (a = 11 ∧ b = 3) ↔
  ¬divisors (a - b) 3 ∧ is_prime (a + 2 * b) ∧ a = 4 * b - 1 ∧ divisors (a + 7) b := by
  sorry

end find_natural_numbers_l129_129048


namespace boxes_in_case_correct_l129_129485

-- Given conditions
def total_boxes : Nat := 2
def blocks_per_box : Nat := 6
def total_blocks : Nat := 12

-- Define the number of boxes in a case as a result of total_blocks divided by blocks_per_box
def boxes_in_case : Nat := total_blocks / blocks_per_box

-- Prove the number of boxes in a case is 2
theorem boxes_in_case_correct : boxes_in_case = 2 := by
  -- Place the actual proof here
  sorry

end boxes_in_case_correct_l129_129485


namespace range_of_m_l129_129847

def P (m : ℝ) : Prop :=
  9 - m > 2 * m ∧ 2 * m > 0

def Q (m : ℝ) : Prop :=
  m > 0 ∧ (Real.sqrt (6) / 2 < Real.sqrt (5 + m) / Real.sqrt (5)) ∧ (Real.sqrt (5 + m) / Real.sqrt (5) < Real.sqrt (2))

theorem range_of_m (m : ℝ) : ¬(P m ∧ Q m) ∧ (P m ∨ Q m) → (0 < m ∧ m ≤ 5 / 2) ∨ (3 ≤ m ∧ m < 5) :=
sorry

end range_of_m_l129_129847


namespace find_x_l129_129116

theorem find_x (x : ℝ) (h : 2 ∈ ({x + 4, x^2 + x} : set ℝ)) : x = 1 :=
sorry

end find_x_l129_129116


namespace hyperbola_asymptote_b_value_l129_129098

theorem hyperbola_asymptote_b_value (b : ℝ) (hb : 0 < b) : 
  (∀ x y, x^2 - y^2 / b^2 = 1 → y = 3 * x ∨ y = -3 * x) → b = 3 := 
by
  sorry

end hyperbola_asymptote_b_value_l129_129098


namespace battery_lasts_2_more_hours_l129_129603

def battery_consumption_not_in_use : ℝ := 1 / 20

def battery_consumption_in_use : ℝ := 1 / 4

def time_not_in_use : ℝ := 8

def time_in_use : ℝ := 2

def total_battery_used : ℝ := (time_not_in_use * battery_consumption_not_in_use) + (time_in_use * battery_consumption_in_use)

def remaining_battery : ℝ := 1 - total_battery_used

def remaining_time (remaining_battery : ℝ) (battery_consumption_not_in_use : ℝ) : ℝ :=
  remaining_battery / battery_consumption_not_in_use

theorem battery_lasts_2_more_hours :
  remaining_time remaining_battery battery_consumption_not_in_use = 2 :=
by
  sorry

end battery_lasts_2_more_hours_l129_129603


namespace directrix_of_parabola_l129_129443

def parabola (x : ℝ) : ℝ := (x^2 - 8*x + 12) / 16

theorem directrix_of_parabola :
  ∀ x, parabola x = (x-4)^2 / 16 - 1/4 →
  let a := 1/16 in
  let h := 4 in
  let k := -1/4 in
  let directrix := k - 1/(4*a) in
  directrix = -17/4 :=
by
  intro x h1
  simp only [parabola] at h1
  dsimp [a, h, k] at h1
  have := calc
    k - 1/(4*a) = -1/4 - 4 : by field_simp [a]
    ... = -17/4 : by norm_num
  exact this

end directrix_of_parabola_l129_129443


namespace total_lunch_eaten_together_l129_129994

variable (y : ℝ)

def Sam_portion : ℝ := y
def Lee_portion : ℝ := 1.5 * y

def Sam_consumed_initial : ℝ := (2/3) * Sam_portion y
def Lee_consumed_initial : ℝ := (2/3) * Lee_portion y

def Lee_remaining : ℝ := Lee_portion y - Lee_consumed_initial y
def Portion_given_to_Sam : ℝ := 0.5 * Lee_remaining y

def Sam_total_consumed : ℝ := Sam_consumed_initial y + Portion_given_to_Sam y
def Lee_total_consumed : ℝ := Lee_consumed_initial y - Portion_given_to_Sam y

theorem total_lunch_eaten_together :
  Sam_total_consumed y = Lee_total_consumed y → (Sam_portion y + Lee_portion y) = 2.5 * y :=
by
  sorry

end total_lunch_eaten_together_l129_129994


namespace derivative_at_zero_l129_129872

def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem derivative_at_zero : deriv f 0 = 2 := by
  sorry

end derivative_at_zero_l129_129872


namespace number_of_proper_subsets_of_three_element_set_l129_129257

theorem number_of_proper_subsets_of_three_element_set (A : Set ℝ) (h : A = {1/3, -2, 0}) : 
  ∃ n : ℕ, n = 7 ∧ (A.card = 3 → (2^A.card - 1) = n) := 
by
  sorry

end number_of_proper_subsets_of_three_element_set_l129_129257


namespace angles_intersection_line_parabola_angles_intersection_ellipse_parabola_angles_intersection_sine_cosine_l129_129341

-- Prove that the angles of intersection between a line and a parabola are as specified
theorem angles_intersection_line_parabola :
  let L := λ x y, x + y - 4 = 0
  let P := λ x y, 2 * y = 8 - x^2
  (∃ θ1 θ2 : ℝ, θ1 = 45 ∧ θ2 = 18.5 ∧
  ∀ (x y : ℝ), L x y ∧ P x y → (θ1 = 45 ∨ θ1 = 18.5) ∧ (θ2 = 45 ∨ θ2 = 18.5)) :=
sorry

-- Prove that the angles of intersection between an ellipse and a parabola are as specified
theorem angles_intersection_ellipse_parabola :
  let E := λ x y, x^2 + 4 * y^2 = 4
  let P := λ x y, 4 * y = 4 - 5 * x^2
  (∃ θ1 θ2 : ℝ, θ1 = 92 ∧ θ2 = 0 ∧
  ∀ (x y : ℝ), E x y ∧ P x y → (θ1 = 92 ∨ θ1 = 0) ∧ (θ2 = 92 ∨ θ2 = 0)) :=
sorry

-- Prove that the angles of intersection between the sine curve and the cosine curve are as specified
theorem angles_intersection_sine_cosine :
  let S := λ x y, y = Real.sin x
  let C := λ x y, y = Real.cos x
  (∃ θ1 θ2 : ℝ, θ1 = 70.5 ∧ θ2 = 109.5 ∧
  ∀ (x y : ℝ), S x y ∧ C x y → (θ1 = 70.5 ∨ θ1 = 109.5) ∧ (θ2 = 70.5 ∨ θ2 = 109.5)) :=
sorry

end angles_intersection_line_parabola_angles_intersection_ellipse_parabola_angles_intersection_sine_cosine_l129_129341


namespace incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129657

def is_color (col : String) (pos : Nat) : Prop := sorry
def valid_guesses : Prop :=
  (is_color "white" 2) ∧
  (is_color "brown" 20) ∧
  (is_color "black" 400) ∧
  (is_color "brown" 600) ∧
  (is_color "white" 800)

theorem incorrect_guess_at_20 :
  (∃ x, (x ∈ [2, 20, 400, 600, 800]) ∧ ¬ is_color_correct x) :=
begin
  sorry -- proof is not required
end

/-- Main theorem to identify the incorrect guess position. -/
theorem Iskander_incorrect_guess_20 :
  valid_guesses →
  (∃! x ∈ [2, 20, 400, 600, 800], ¬ is_color_correct x) →
  ¬ is_color "brown" 20 :=
begin
  admit -- proof is not required
end

end incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129657


namespace find_EF_l129_129150

variable {α : Type*} [LinearOrderedField α]

variables (A D F E : Point α)
variables (a b : α)

-- Conditions
def is_trapezoid (ABCD : Trapezoid α) : Prop :=
  -- definitions for bases AD and BC, intersection of AC and BD, etc.
  sorry

def circles_tangent_at (E : Point α) (circle : Circle α) : Prop :=
  -- definition of tangent complexity
  sorry

def points_collinear (A D F : Point α) : Prop :=
  -- definition of collinearity
  sorry

theorem find_EF (ABCD : Trapezoid α) (circle : Circle α)
  (h1 : is_trapezoid ABCD)
  (h2 : circles_tangent_at E circle)
  (h3 : points_collinear A D F)
  (hAF : dist A F = a)
  (hAD : dist A D = b) :
  dist E F = Real.sqrt (a * (b - a)) :=
sorry

end find_EF_l129_129150


namespace union_M_N_l129_129182

open Set Real

def M : Set ℝ := {x | x - x^2 ≠ 0}
def N : Set ℝ := {x | ln (1 - x) < 0}

theorem union_M_N : M ∪ N = Iio 1 := by
  sorry

end union_M_N_l129_129182


namespace solution_l129_129478

noncomputable def transformation (v : List ℤ) : List ℤ :=
  List.zipWith (+) v (v.tail ++ [v.head])

theorem solution (n k : ℕ) (hn : 2 ≤ n) (hk : 2 ≤ k) :
  (∀ (a : List ℤ), a.length = n → 
    ∃ m, (m > 0) ∧ List.all (List.iterate transformation m a) (λ x, x % k = 0)) ↔ (n = 2) :=
by sorry

end solution_l129_129478


namespace baseball_card_ratio_l129_129195

-- Define the conditions
variable (T : ℤ) -- Number of baseball cards on Tuesday

-- Given conditions
-- On Monday, Buddy has 30 baseball cards
def monday_cards : ℤ := 30

-- On Wednesday, Buddy has T + 12 baseball cards
def wednesday_cards : ℤ := T + 12

-- On Thursday, Buddy buys a third of what he had on Tuesday
def thursday_additional_cards : ℤ := T / 3

-- Total number of cards on Thursday is 32
def thursday_cards (T : ℤ) : ℤ := T + 12 + T / 3

-- We are given that Buddy has 32 baseball cards on Thursday
axiom thursday_total : thursday_cards T = 32

-- The theorem we want to prove: the ratio of Tuesday's to Monday's cards is 1:2
theorem baseball_card_ratio
  (T : ℤ)
  (htotal : thursday_cards T = 32)
  (hmon : monday_cards = 30) :
  T = 15 ∧ (T : ℚ) / monday_cards = 1 / 2 := by
  -- Proof goes here
  sorry

end baseball_card_ratio_l129_129195


namespace place_signs_correct_l129_129611

theorem place_signs_correct :
  1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99 :=
by
  sorry

end place_signs_correct_l129_129611


namespace correct_methods_count_l129_129326

theorem correct_methods_count :
  let method1 := (x : ℝ) -> (x - 2) ^ 2
  let method2 := (x : ℝ) -> (x - 1) ^ 2 - 1
  let method3 := (x : ℝ) -> x ^ 2 - 4
  let method4 := (x : ℝ) -> -(x ^ 2) + 4 in
  method1 2 = 0 ∧ method2 2 = 0 ∧ method3 2 = 0 ∧ method4 2 = 0 :=
by
  intros
  split; simp [method1, method2, method3, method4]; sorry

end correct_methods_count_l129_129326


namespace seniors_to_be_selected_l129_129773

/-- Define the conditions as variables -/
variables (TotalStudents Freshmen Sophomores Seniors SampleSize : ℕ)
variables (SamplingRatio : ℚ)

/-- Set the specific values based on the problem conditions -/
def high_school_population := (TotalStudents = 900) ∧ (Freshmen = 300) ∧ (Sophomores = 200) ∧ (Seniors = 400)
def sample_size := (SampleSize = 45)
def sampling_ratio := (SamplingRatio = SampleSize / TotalStudents)

/-- Prove that the number of seniors to be selected is 20 given stratified sampling -/
theorem seniors_to_be_selected :
  high_school_population TotalStudents Freshmen Sophomores Seniors ∧ sample_size SampleSize →
  Seniors * SamplingRatio = 20 := by
  sorry

end seniors_to_be_selected_l129_129773


namespace triangle_ABC_area_l129_129642

-- Definition of points A, B, and C according to the conditions
def A : ℝ × ℝ := (2, 5)
def B : ℝ × ℝ := (-2, 5)
def C : ℝ × ℝ := (5, -2)

-- Goal: Prove the area of triangle ABC is 14
theorem triangle_ABC_area : 
  let AB := real.sqrt ((-2 - 2)^2 + (5 - 5)^2),
    height := abs (5 - (-2)) in
  (1/2) * AB * height = 14 := 
by {
  sorry,
}

end triangle_ABC_area_l129_129642


namespace incorrect_guess_l129_129694

-- Define the conditions
def bears : ℕ := 1000

inductive Color
| White
| Brown
| Black

constant bear_color : ℕ → Color -- The color of the bear at each position

axiom condition : ∀ n : ℕ, n < bears - 2 → 
  ∃ i j k, (i, j, k ∈ {Color.White, Color.Brown, Color.Black}) ∧ 
  (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ 
  (bear_color n = i ∧ bear_color (n+1) = j ∧ bear_color (n+2) = k) 

constants (g1 : bear_color 2 = Color.White)
          (g2 : bear_color 20 = Color.Brown)
          (g3 : bear_color 400 = Color.Black)
          (g4 : bear_color 600 = Color.Brown)
          (g5 : bear_color 800 = Color.White)

-- The proof problem
theorem incorrect_guess : bear_color 20 ≠ Color.Brown :=
by sorry

end incorrect_guess_l129_129694


namespace chip_paper_packs_needed_l129_129002

-- Define the daily page usage for each subject
def pages_per_day_math : ℕ := 1
def pages_per_day_science : ℕ := 2
def pages_per_day_history : ℕ := 3
def pages_per_day_language_arts : ℕ := 1
def pages_per_day_art : ℕ := 1

-- Define the number of school days in regular and short weeks
def regular_school_days_per_week : ℕ := 5
def short_school_days_per_week : ℕ := 3

-- Define the semester specifics
def total_weeks : ℕ := 15
def holiday_weeks : ℕ := 2
def short_weeks : ℕ := 3

-- Define the paper pack size
def pages_per_pack : ℕ := 100

-- Lean statement to prove the number of packs needed
theorem chip_paper_packs_needed : 
  (∑ i in [pages_per_day_math, pages_per_day_science, pages_per_day_history, pages_per_day_language_arts, pages_per_day_art], i * regular_school_days_per_week * (total_weeks - holiday_weeks - short_weeks) + 
  ∑ i in [pages_per_day_math, pages_per_day_science, pages_per_day_history, pages_per_day_language_arts, pages_per_day_art], i * short_school_days_per_week * short_weeks) / pages_per_pack = 6 :=
sorry

end chip_paper_packs_needed_l129_129002


namespace triangle_inequality_satisfied_for_n_six_l129_129032

theorem triangle_inequality_satisfied_for_n_six :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ a + c > b ∧ b + c > a) := sorry

end triangle_inequality_satisfied_for_n_six_l129_129032


namespace product_of_divisors_of_72_l129_129464

theorem product_of_divisors_of_72 :
  let n := 72 in
  let prime_factors := [(2, 3), (3, 2)] in
  let divisor_count := (3 + 1) * (2 + 1) in
  divisor_count = 12 →
  (n^(divisor_count / 2) = 2^18 * 3^12) :=
by
  intros
  sorry

end product_of_divisors_of_72_l129_129464


namespace proof_problem_l129_129495

noncomputable theory
open Classical Real

def ellipse_equation (a b : ℝ) (h : a > b) : Prop :=
  e = (sqrt 2 / 2) -> (a = sqrt 2 ∧ b = 1) ∧ (∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1 ↔ ((x^2) / 2 + y^2 = 1))

def fixed_point_exists (a b : ℝ) (S : ℝ × ℝ) (k : ℝ) : Prop :=
  a > b → S = (0, 1 / 3) →
  ∃ M : ℝ × ℝ, M = (0, 1) ∧ 
  ∀ (x1 x2 y1 y2 : ℝ), 
    -- Points A, B on ellipse
    ((x1^2) / a^2 + y1^2 / b^2 = 1) ∧ (x1 = x2) → 
    ((x2^2) / a^2 + y2^2 / b^2 = 1) ∧ (x2 = x1) ∧ 
    -- Points are on the line 
    (y1 = k * x1 + 1 / 3) ∧ (y2 = k * x2 + 1 / 3) →
    -- Fixed point condition
    let AB : ℝ := (x1 - M.1) * (x2 - M.1) + (y1 - M.2) * (y2 - M.2) in
    AB = 0

-- The theorem stating the problem

theorem proof_problem :
  ∀ (a b e : ℝ) (S : ℝ × ℝ) (k : ℝ),
    ellipse_equation a b (a > b) ∧ 
    fixed_point_exists a b S k :=
sorry

end proof_problem_l129_129495


namespace points_lie_on_parabola_l129_129481

noncomputable def x (s : ℝ) : ℝ := 3^s - 4
noncomputable def y (s : ℝ) : ℝ := 9^s - 7 * 3^s + 2

theorem points_lie_on_parabola : ∃ a b c : ℝ, ∀ s : ℝ, y s = (x s) ^ 2 + (a * (x s)) + b = y(s) := 
begin
  use [1, 1, -10],  -- Representing the equation y = x^2 + 1*x - 10
  intro s,
  unfold x y,
  simp,
  -- Proof will go here
  sorry
end

end points_lie_on_parabola_l129_129481


namespace sawyer_coaching_fee_l129_129995

def coaching_days_per_month : List Nat := [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 3]

def daily_charge : Nat := 39

def total_days : Nat := coaching_days_per_month.sum

def coaching_fee : Nat := total_days * daily_charge

theorem sawyer_coaching_fee :
  coaching_fee = 11934 := by
  sorry

end sawyer_coaching_fee_l129_129995


namespace fifth_row_sequence_l129_129431

theorem fifth_row_sequence (grid : ℕ → ℕ → ℕ) :
  -- Conditions for the grid:
  (∀ i ∈ finset.range 5, finset.card (finset.image (grid i) (finset.range 5)) = 5) ∧ -- Each row contains exactly one of {2, 0, 1, 5, 9}
  (∀ j ∈ finset.range 5, finset.card (finset.image (λ i, grid i j) (finset.range 5)) = 5) ∧ -- Each column contains exactly one of {2, 0, 1, 5, 9}
  (∀ i j ∈ finset.range 4, (grid i j ≠ grid (i + 1) (j + 1)) ∧ (grid (i + 1) j ≠ grid i (j + 1))) -- No two same numbers are diagonally adjacent

  → -- Target statement:
  (grid 4 0 = 1 ∧ grid 4 1 = 5 ∧ grid 4 2 = 9 ∧ grid 4 3 = 9 ∧ grid 4 4 = 2) := sorry

end fifth_row_sequence_l129_129431


namespace max_a_sqrt3b_l129_129079

theorem max_a_sqrt3b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : sqrt 3 * b = sqrt ((1 - a) * (1 + a))) :
  a + sqrt 3 * b ≤ sqrt 2 :=
by
  sorry

end max_a_sqrt3b_l129_129079


namespace ed_money_left_after_hotel_stay_l129_129429

theorem ed_money_left_after_hotel_stay 
  (night_rate : ℝ) (morning_rate : ℝ) 
  (initial_money : ℝ) (hours_night : ℕ) (hours_morning : ℕ) 
  (remaining_money : ℝ) : 
  night_rate = 1.50 → morning_rate = 2.00 → initial_money = 80 → 
  hours_night = 6 → hours_morning = 4 → 
  remaining_money = 63 :=
by
  intros h1 h2 h3 h4 h5
  let cost_night := night_rate * hours_night
  let cost_morning := morning_rate * hours_morning
  let total_cost := cost_night + cost_morning
  let money_left := initial_money - total_cost
  sorry

end ed_money_left_after_hotel_stay_l129_129429


namespace expression_equals_19_l129_129396

noncomputable def calc_expression : ℝ :=
  (2 * real.sqrt 2) ^ (2 / 3) * (0.1) ^ (-1) - real.log 2 / real.log 10 - real.log 5 / real.log 10

theorem expression_equals_19 : calc_expression = 19 := by
  sorry

end expression_equals_19_l129_129396


namespace neither_sufficient_nor_necessary_l129_129226

theorem neither_sufficient_nor_necessary (a b : ℝ) (h1 : a ≠ 5) (h2 : b ≠ -5) : ¬((a + b ≠ 0) ↔ (a ≠ 5 ∧ b ≠ -5)) :=
by sorry

end neither_sufficient_nor_necessary_l129_129226


namespace correct_option_among_sqrt_statements_l129_129745

theorem correct_option_among_sqrt_statements :
  ¬ (sqrt 16 = -4 ∨ sqrt 16 = 4) ∧
  ¬ (sqrt ((-3)^2) = -3) ∧
  (sqrt 81 = 9 ∨ -sqrt 81 = -9) ∧
  ¬ (sqrt (- 4) = 2) ∧
  ( (sqrt 16 = 4 ∨ sqrt 16 = -4) ∧
    (sqrt ((-3)^2) = 3) ∧
    (sqrt 81 = 9 ∨ -sqrt 81 = -9) ∧
    ¬ sqrt (-4)) →  
  true :=
by
  sorry

end correct_option_among_sqrt_statements_l129_129745


namespace geric_bills_l129_129487

variable (G K J : ℕ)

theorem geric_bills (h1 : G = 2 * K) 
                    (h2 : K = J - 2) 
                    (h3 : J = 7 + 3) : 
    G = 16 := by
  sorry

end geric_bills_l129_129487


namespace isosceles_triangle_base_length_l129_129239

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l129_129239


namespace find_alcohol_quantity_l129_129756

theorem find_alcohol_quantity 
  (A W : ℝ) 
  (h1 : A / W = 2 / 5)
  (h2 : A / (W + 10) = 2 / 7) : 
  A = 10 :=
sorry

end find_alcohol_quantity_l129_129756


namespace calc_expression_eq_neg_three_over_two_l129_129399

theorem calc_expression_eq_neg_three_over_two : 
  -2⁻¹ + (Real.sqrt 16 - Real.pi)⁰ - |Real.sqrt 3 - 2| - 2 * Real.cos (Real.pi / 6) = -3 / 2 :=
sorry

end calc_expression_eq_neg_three_over_two_l129_129399


namespace find_n_such_that_abs_squared_minus_6n_minus_27_is_prime_l129_129432

/-- 
  Find all integers \( n \) such that \( \left| n^2 - 6n - 27 \right| \) is prime.
  We are going to prove that the solutions are:
  n = -4, -2, 8, 10.
-/
theorem find_n_such_that_abs_squared_minus_6n_minus_27_is_prime :
  {n : ℤ | nat_abs (n^2 - 6 * n - 27).nat_abs }.prime = {-4, -2, 8, 10} :=
sorry

end find_n_such_that_abs_squared_minus_6n_minus_27_is_prime_l129_129432


namespace max_volume_day1_l129_129358

-- Define volumes of the containers
def volumes : List ℕ := [9, 13, 17, 19, 20, 38]

-- Define conditions: sold containers volumes
def condition_on_first_day (s: List ℕ) := s.length = 3
def condition_on_second_day (s: List ℕ) := s.length = 2

-- Define condition: total and relative volumes sold
def volume_sold_first_day (s: List ℕ) : ℕ := s.foldr (λ x acc => x + acc) 0
def volume_sold_second_day (s: List ℕ) : ℕ := s.foldr (λ x acc => x + acc) 0

def volume_sold_total (s1 s2: List ℕ) := volume_sold_first_day s1 + volume_sold_second_day s2 = 116
def volume_ratio (s1 s2: List ℕ) := volume_sold_first_day s1 = 2 * volume_sold_second_day s2 

-- The goal is to prove the maximum possible volume_sold_first_day
theorem max_volume_day1 (s1 s2: List ℕ) 
  (h1: condition_on_first_day s1)
  (h2: condition_on_second_day s2)
  (h3: volume_sold_total s1 s2)
  (h4: volume_ratio s1 s2) : 
  ∃(max_volume: ℕ), max_volume = 66 :=
sorry

end max_volume_day1_l129_129358


namespace each_boy_earns_14_dollars_l129_129295

theorem each_boy_earns_14_dollars :
  let Victor_shrimp := 26 in
  let Austin_shrimp := Victor_shrimp - 8 in
  let total_Victor_Austin_shrimp := Victor_shrimp + Austin_shrimp in
  let Brian_shrimp := total_Victor_Austin_shrimp / 2 in
  let total_shrimp := Victor_shrimp + Austin_shrimp + Brian_shrimp in
  let total_money := (total_shrimp / 11) * 7 in
  let money_per_boy := total_money / 3 in
  money_per_boy = 14 :=
by
  sorry

end each_boy_earns_14_dollars_l129_129295


namespace distance_origin_to_A_l129_129932

-- Define the conditions of the problem
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line_through_focus (x y : ℝ) : Prop := y = sqrt 3 * (x - 1)
def point_A (x y : ℝ) : Prop := x = 3 ∧ y = 2 * sqrt 3

-- Statement of the problem
theorem distance_origin_to_A :
  ∀ x y : ℝ, parabola x y → line_through_focus x y → point_A x y → real.sqrt (x^2 + y^2) = sqrt 21 :=
by
  intros x y h_parabola h_line h_A
  sorry

end distance_origin_to_A_l129_129932


namespace incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129655

def is_color (col : String) (pos : Nat) : Prop := sorry
def valid_guesses : Prop :=
  (is_color "white" 2) ∧
  (is_color "brown" 20) ∧
  (is_color "black" 400) ∧
  (is_color "brown" 600) ∧
  (is_color "white" 800)

theorem incorrect_guess_at_20 :
  (∃ x, (x ∈ [2, 20, 400, 600, 800]) ∧ ¬ is_color_correct x) :=
begin
  sorry -- proof is not required
end

/-- Main theorem to identify the incorrect guess position. -/
theorem Iskander_incorrect_guess_20 :
  valid_guesses →
  (∃! x ∈ [2, 20, 400, 600, 800], ¬ is_color_correct x) →
  ¬ is_color "brown" 20 :=
begin
  admit -- proof is not required
end

end incorrect_guess_at_20_Iskander_incorrect_guess_20_l129_129655


namespace spinner_final_direction_l129_129568

theorem spinner_final_direction :
  ∀ (initial_direction : string) (rotation1 : ℚ) (rotation2 : ℚ) (rotation3 : ℚ),
  initial_direction = "south" →
  rotation1 = 7 / 2 →
  rotation2 = -21 / 4 →
  rotation3 = 3 / 4 →
  let total_clockwise := rotation1 + rotation3 in
  let net_movement := total_clockwise + rotation2 in
  net_movement = -1 →
  (initial_direction = "south" → "south")
:=
sorry

end spinner_final_direction_l129_129568


namespace even_fn_a_eq_zero_l129_129911

def f (x a : ℝ) : ℝ := x^2 - |x + a|

theorem even_fn_a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end even_fn_a_eq_zero_l129_129911


namespace volume_intersection_zero_l129_129730

/-- The set of points satisfying |x| + |y| + |z| ≤ 1. -/
def region1 (x y z : ℝ) : Prop :=
  |x| + |y| + |z| ≤ 1

/-- The set of points satisfying |x| + |y| + |z-2| ≤ 1. -/
def region2 (x y z : ℝ) : Prop :=
  |x| + |y| + |z-2| ≤ 1

/-- The intersection of region1 and region2 forms a region with volume 0. -/
theorem volume_intersection_zero : 
  (∫ x y z, (region1 x y z ∧ region2 x y z)) = 0 := sorry

end volume_intersection_zero_l129_129730


namespace sum_of_squares_to_15_mod_17_eq_10_l129_129318

def sum_of_squares_modulo_17 : ℕ :=
  let sum := (Finset.sum (Finset.range 16) (λ n, n^2 % 17)) in
  sum % 17

theorem sum_of_squares_to_15_mod_17_eq_10 : sum_of_squares_modulo_17 = 10 :=
  sorry

end sum_of_squares_to_15_mod_17_eq_10_l129_129318


namespace isosceles_base_length_l129_129230

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l129_129230


namespace john_paintball_times_l129_129164

theorem john_paintball_times (x : ℕ) (cost_per_box : ℕ) (boxes_per_play : ℕ) (monthly_spending : ℕ) :
  (cost_per_box = 25) → (boxes_per_play = 3) → (monthly_spending = 225) → (boxes_per_play * cost_per_box * x = monthly_spending) → x = 3 :=
by
  intros h1 h2 h3 h4
  -- proof would go here
  sorry

end john_paintball_times_l129_129164


namespace incorrect_guess_l129_129697

-- Define the conditions
def bears : ℕ := 1000

inductive Color
| White
| Brown
| Black

constant bear_color : ℕ → Color -- The color of the bear at each position

axiom condition : ∀ n : ℕ, n < bears - 2 → 
  ∃ i j k, (i, j, k ∈ {Color.White, Color.Brown, Color.Black}) ∧ 
  (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ 
  (bear_color n = i ∧ bear_color (n+1) = j ∧ bear_color (n+2) = k) 

constants (g1 : bear_color 2 = Color.White)
          (g2 : bear_color 20 = Color.Brown)
          (g3 : bear_color 400 = Color.Black)
          (g4 : bear_color 600 = Color.Brown)
          (g5 : bear_color 800 = Color.White)

-- The proof problem
theorem incorrect_guess : bear_color 20 ≠ Color.Brown :=
by sorry

end incorrect_guess_l129_129697


namespace ratio_of_volumes_cone_cylinder_l129_129468

theorem ratio_of_volumes_cone_cylinder (r h_cylinder : ℝ) (h_cone : ℝ) (h_radius : r = 4) (h_height_cylinder : h_cylinder = 12) (h_height_cone : h_cone = h_cylinder / 2) :
  ((1/3) * (π * r^2 * h_cone)) / (π * r^2 * h_cylinder) = 1 / 6 :=
by
  -- Definitions and assumptions are directly included from the conditions.
  sorry

end ratio_of_volumes_cone_cylinder_l129_129468


namespace Karen_tote_weight_l129_129571

variable (B T F : ℝ)
variable (Papers Laptop : ℝ)

theorem Karen_tote_weight (h1: T = 2 * B)
                         (h2: F = 2 * T)
                         (h3: Papers = (1 / 6) * F)
                         (h4: Laptop = T + 2)
                         (h5: F = B + Laptop + Papers):
  T = 12 := 
sorry

end Karen_tote_weight_l129_129571


namespace xy_in_N_l129_129183

def M := {x : ℤ | ∃ m : ℤ, x = 3 * m + 1}
def N := {y : ℤ | ∃ n : ℤ, y = 3 * n + 2}

theorem xy_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : (x * y) ∈ N :=
by
  sorry

end xy_in_N_l129_129183


namespace triangle_inequality_condition_l129_129044

theorem triangle_inequality_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (ineq : 6 * (a * b + b * c + c * a) > 5 * (a ^ 2 + b ^ 2 + c ^ 2)) : 
  (a < b + c ∧ b < a + c ∧ c < a + b) :=
sorry

end triangle_inequality_condition_l129_129044


namespace Jia_age_is_24_l129_129281

variable (Jia Yi Bing Ding : ℕ)

theorem Jia_age_is_24
  (h1 : (Jia + Yi + Bing) / 3 = (Jia + Yi + Bing + Ding) / 4 + 1)
  (h2 : (Jia + Yi) / 2 = (Jia + Yi + Bing) / 3 + 1)
  (h3 : Jia = Yi + 4)
  (h4 : Ding = 17) :
  Jia = 24 :=
by
  sorry

end Jia_age_is_24_l129_129281


namespace area_square_117_l129_129140

variables {A B C D P Q R : Type}
variables [affine_space A]
variables [affine_space B]
variables [affine_space C]
variables [affine_space D]
variables [affine_space P]
variables [affine_space Q]
variables [affine_space R]

variables (ABCD : is_square A B C D)
variables (on_AD_P: lies_on AD P)
variables (on_AB_Q: lies_on AB Q)
variables (right_angle_intersection: is_right_angle (BP.stop) (CQ.stop))
variables (BR_length: distance B R = 6)
variables (PR_length: distance P R = 7)
variables (s : ℝ)

-- Prove the area of the square
theorem area_square_117 : area ABCD = 117 :=
by sorry

end area_square_117_l129_129140


namespace profit_relation_plan1_profit_relation_plan2_profit_comparison_6000_l129_129357

-- Define the constants and variables
def factory_price := 50
def cost_per_product := 25
def wastewater_per_product := 0.5
def plan1_material_cost_per_cubic_meter := 2
def plan1_wear_and_tear := 30000
def plan2_treatment_cost_per_cubic_meter := 14

-- Define the relationships
def profit_plan1 (x : ℕ) : ℤ :=
  factory_price * x - cost_per_product * x - (wastewater_per_product * x).to_int * plan1_material_cost_per_cubic_meter - plan1_wear_and_tear

def profit_plan2 (x : ℕ) : ℤ :=
  factory_price * x - cost_per_product * x - (wastewater_per_product * x).to_int * plan2_treatment_cost_per_cubic_meter

-- Proofs for part (1)
theorem profit_relation_plan1 (x : ℕ) : profit_plan1 x = 24 * x - 30000 := 
by 
  sorry

theorem profit_relation_plan2 (x : ℕ) : profit_plan2 x = 18 * x := 
by 
  sorry

-- Proof for part (2) when x = 6000
theorem profit_comparison_6000 :
  profit_plan1 6000 > profit_plan2 6000 := 
by 
  sorry

end profit_relation_plan1_profit_relation_plan2_profit_comparison_6000_l129_129357


namespace directrix_of_parabola_l129_129448

noncomputable def parabola_directrix (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / 16

theorem directrix_of_parabola :
  let d := parabola_directrix y in d = -(1 / 2) := sorry

end directrix_of_parabola_l129_129448


namespace angle_BAC_eq_pi_div_4_l129_129137

theorem angle_BAC_eq_pi_div_4 (α β : ℝ) (A B C D : Point) 
    (h1 : AcuteAngle (A, B, C)) (h2 : Perpendicular AD BC) (h3 : 2*|BD| = 3*|DC| ∧  AD = 6*|BD|)
    (h4 : ∠BAD = α) (h5 : ∠CAD = β) :
    α + β = π / 4 :=
by
  sorry

end angle_BAC_eq_pi_div_4_l129_129137


namespace final_cost_is_35_l129_129218

-- Definitions based on conditions
def original_price : ℕ := 50
def discount_rate : ℚ := 0.30
def discount_amount : ℚ := original_price * discount_rate
def final_cost : ℚ := original_price - discount_amount

-- The theorem we need to prove
theorem final_cost_is_35 : final_cost = 35 := by
  sorry

end final_cost_is_35_l129_129218


namespace find_a_b_sum_l129_129864

theorem find_a_b_sum (a b : ℝ) :
  (SetOf (λ x => (x - a) / (x - b) > 0)) = Set.union (Set.interval_oc ⟨-, 1⟩) (Set.interval_co ⟨4, +∞⟩) →
  a + b = 5 :=
sorry

end find_a_b_sum_l129_129864


namespace hyperbola_equation_point_P_fixed_line_l129_129083

-- Let the hyperbola have its center at the origin.
-- Define the constants given in the problem.
def c : ℝ := 2 * Real.sqrt 5
def e : ℝ := Real.sqrt 5

-- Define the equation of the hyperbola given the conditions.
def hyperbola_eqn (x y : ℝ) : Prop := (x ^ 2 / 4) - (y ^ 2 / 16) = 1

-- Prove the equation of the hyperbola C given the conditions.
theorem hyperbola_equation :
  ∃ (a b : ℝ), (c ^ 2 = a ^ 2 + b ^ 2) ∧ (e = c / a) ∧ hyperbola_eqn x y :=
sorry

-- Define the condition for point P lying on the line x = -1.
def point_P_on_fixed_line (P : ℝ × ℝ) : Prop := P.1 = -1

-- Prove that point P lies on the fixed line given the conditions.
theorem point_P_fixed_line 
  (A1 A2 : ℝ × ℝ)
  (M N P : ℝ × ℝ)
  (line_through_MN : ∀ (x y : ℝ), (line_through (⟨-4, 0⟩) = x = my - 4))
  (M_in_second_quadrant : M.1 < 0 ∧ M.2 > 0)
  (intersection_conditions : MA1 = line_through (A1) ∧ NA2 = line_through (A2)):
  point_P_on_fixed_line P :=
sorry

end hyperbola_equation_point_P_fixed_line_l129_129083


namespace area_inequality_l129_129576

theorem area_inequality (
  ABCD K L M N : Set ℝ
  (s1 s2 s3 s4 s : ℝ)
  (convex_quadrilateral : ∀ A B C D : ℝ, convex_hull ℝ ({A, B, C, D} : Set ℝ).nonempty)
  (K_on_AB : K ∈ segment ℝ A B)
  (L_on_BC : L ∈ segment ℝ B C)
  (M_on_CD : M ∈ segment ℝ C D)
  (N_on_DA : N ∈ segment ℝ D A)
  (area_AKN : s1 = Area (convex_hull ℝ ({A, K, N} : Set ℝ)))
  (area_BKL : s2 = Area (convex_hull ℝ ({B, K, L} : Set ℝ)))
  (area_CLM : s3 = Area (convex_hull ℝ ({C, L, M} : Set ℝ)))
  (area_DMN : s4 = Area (convex_hull ℝ ({D, M, N} : Set ℝ)))
  (area_ABCD : s = Area (convex_hull ℝ ({A, B, C, D} : Set ℝ)))
) :
  ∑ x in [s1, s2, s3, s4], real.cbrt x ≤ 2 * real.cbrt s :=
by
  sorry

end area_inequality_l129_129576


namespace white_pairs_coincide_l129_129020

def red_triangles : ℕ := 4
def blue_triangles : ℕ := 6
def white_triangles : ℕ := 10
def red_pairs : ℕ := 3
def blue_pairs : ℕ := 4
def red_white_pairs : ℕ := 3

theorem white_pairs_coincide : 
  (2 * (white_triangles - red_white_pairs - blue_triangles + blue_pairs) = 2 * 5) :=
by 
  have pairs_of_red_white_remaining : ℕ := (white_triangles - red_white_pairs)
  have pairs_of_blue_white_remaining : ℕ := pairs_of_red_white_remaining - (blue_triangles - blue_pairs)
  have coinciding_white_pairs: ℕ := pairs_of_blue_white_remaining
  exact congr_arg2 Nat.mul 2 coinciding_white_pairs sorry

end white_pairs_coincide_l129_129020


namespace max_area_triangle_ABC_l129_129968

noncomputable def maximum_area_triangle (x1 y1 x2 y2 : ℝ) : ℝ :=
  if h₁ : x1^2 + y1^2 = 25
  ∧ x2^2 + y2^2 = 25
  ∧ x1 * x2 + (y1 - 3) * (y2 - 3) = 0 then
    (25 + 3 * Real.sqrt 41) / 4
  else 0

theorem max_area_triangle_ABC :
  ∀ (x1 y1 x2 y2 : ℝ),
    x1^2 + y1^2 = 25 →
    x2^2 + y2^2 = 25 →
    x1 * x2 + (y1 - 3) * (y2 - 3) = 0 →
    ∃ (S : ℝ), S = maximum_area_triangle x1 y1 x2 y2 ∧ S = (25 + 3 * Real.sqrt 41) / 4 :=
by
  intros x1 y1 x2 y2 h₁ h₂ h₃
  use (25 + 3 * Real.sqrt 41) / 4
  split
  . simp [maximum_area_triangle, h₁, h₂, h₃]
  . rfl

end max_area_triangle_ABC_l129_129968


namespace continuous_finite_zeros_smp_l129_129947

open Set

variable {a b : ℝ} {f : ℝ → ℝ}

def smp (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, ∃ c : Fin (n+1) → ℝ,
    a = c 0 ∧ b = c n ∧
    StrictMono fun i => c i ∧
    ∀ i : Fin n, 
    (∀ x, c i < x ∧ x < c (i + 1) → f (c i) < f x ∧ f x < f (c (i + 1))) ∨
    (∀ x, c i > x ∧ x > c (i + 1) → f (c i) > f x ∧ f x > f (c (i + 1)))

theorem continuous_finite_zeros_smp
  (f_continuous : ContinuousOn f (Icc a b))
  (finite_zeros : ∀ v, (∃ S : Finset ℝ, ∀ x, f x = v ↔ x ∈ S)) :
  smp a b f :=
sorry

end continuous_finite_zeros_smp_l129_129947


namespace volume_of_intersection_is_zero_l129_129732

-- Definition of the regions
def region1 (x y z : ℝ) : Prop := abs x + abs y + abs z ≤ 1
def region2 (x y z : ℝ) : Prop := abs x + abs y + abs (z - 2) ≤ 1

-- Volume of the intersection of region1 and region2
theorem volume_of_intersection_is_zero : 
  let volume_intersection : ℝ := 0 
  in volume_intersection = 0 := 
by
  sorry

end volume_of_intersection_is_zero_l129_129732


namespace largest_8_11_double_l129_129000

def is_8_11_double (M : ℕ) : Prop :=
  let digits_8 := (Nat.digits 8 M)
  let M_11 := Nat.ofDigits 11 digits_8
  M_11 = 2 * M

theorem largest_8_11_double : ∃ (M : ℕ), is_8_11_double M ∧ ∀ (N : ℕ), is_8_11_double N → N ≤ M :=
sorry

end largest_8_11_double_l129_129000


namespace find_wrong_guess_l129_129686

-- Define the three colors as an inductive type.
inductive Color
| white
| brown
| black

-- Define the bears as a list of colors.
def bears (n : ℕ) : Type := list Color

-- Define the conditions: 
-- There are 1000 bears and each tuple of 3 consecutive bears has all three colors.
def valid_bears (b : bears 1000) : Prop :=
  ∀ i : ℕ, i + 2 < 1000 → 
    ∃ c1 c2 c3 : Color, 
      c1 ∈ b.nth i ∧ c2 ∈ b.nth (i+1) ∧ c3 ∈ b.nth (i+2) ∧ 
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3

-- Define Iskander's guesses.
def guesses (b : bears 1000) : Prop :=
  b.nth 1 = some Color.white ∧
  b.nth 19 = some Color.brown ∧
  b.nth 399 = some Color.black ∧
  b.nth 599 = some Color.brown ∧
  b.nth 799 = some Color.white

-- Prove that exactly one of Iskander's guesses is wrong.
def wrong_guess (b : bears 1000) : Prop :=
  (b.nth 19 ≠ some Color.brown) ∧
  valid_bears b ∧
  guesses b →
  ∃ i, i ∈ {1, 19, 399, 599, 799} ∧ (b.nth i ≠ some Color.white ∧ b.nth i ≠ some Color.brown ∧ b.nth i ≠ some Color.black)

theorem find_wrong_guess : 
  ∀ b : bears 1000, 
  valid_bears b → guesses b → wrong_guess b :=
  by
  intros b vb gs
  sorry

end find_wrong_guess_l129_129686


namespace directrix_of_parabola_l129_129439

-- Define the parabola function
def parabola (x : ℝ) : ℝ := (x^2 - 8*x + 12) / 16

theorem directrix_of_parabola : 
  ∃ y : ℝ, (∀ x : ℝ, parabola x = y) → y = -17 / 4 := 
by
  sorry

end directrix_of_parabola_l129_129439


namespace f_l129_129066

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x * (deriv (λ y, f y) 1)

theorem f'_at_0 : deriv f 0 = -4 := by
  sorry

end f_l129_129066


namespace first_nonzero_digit_of_one_over_199_l129_129717

theorem first_nonzero_digit_of_one_over_199 :
  (∃ n : ℕ, (n < 10) ∧ (rat.of_int 2 / rat.of_int 100 < 1 / rat.of_int 199) ∧ (1 / rat.of_int 199 < rat.of_int 3 / rat.of_int 100)) :=
sorry

end first_nonzero_digit_of_one_over_199_l129_129717


namespace compare_abc_l129_129956

theorem compare_abc :
  let a := (4 - Real.log 4) / Real.exp 2
  let b := Real.log 2 / 2
  let c := 1 / Real.exp 1 in
  b < a ∧ a < c := 
sorry

end compare_abc_l129_129956


namespace remainder_sum_equiv_l129_129480

-- Define the function r(n) as the sum of the remainders of n divided by each number from 1 to n
def r (n : ℕ) : ℕ := ∑ i in finset.range n, (n % (i + 1))

-- State the theorem to be proven
theorem remainder_sum_equiv (k : ℕ) (h : 1 ≤ k) : r (2^k - 1) = r (2^k) :=
sorry

end remainder_sum_equiv_l129_129480


namespace option_A_option_B_option_C_option_D_l129_129937

-- Conditions
def student := {A, B, C}
def community := {A, B, C, D, E}

-- Option A
theorem option_A (h: ∃ s ∈ student, ∃ c, c = community.A): 
  fintype.card (student → community) - fintype.card ({s : student → community // ∀ s, s ≠ community.A}) = 61 := 
begin
  sorry,
end

-- Option B
theorem option_B (h : ∀ s ∈ student, s = community.A → 
  fintype.card {s : student → community // ∀ s, s = community.A} = 25 :=
begin
  sorry,
end

-- Option C
theorem option_C (h : ∀ s₁ s₂ s₃ ∈ student, s₁ ≠ s₂ ∧ s₂ ≠ s₃ ∧ s₁ ≠ s₃ →   
  fintype.card {s : student → community // function.injective s } = 60 :=
begin
  sorry,
end

-- Option D
theorem option_D (h : ∀ s₁ s₂ ∈ student, s₁ = s₂ → 
  fintype.card {s : student → community // s₁ = s₂} = 20 :=
begin
  sorry,
end

end option_A_option_B_option_C_option_D_l129_129937


namespace identifyIncorrectGuess_l129_129682

-- Define the colors of the bears
inductive BearColor
| white
| brown
| black

-- Conditions as defined in the problem statement
def isValidBearRow (bears : Fin 1000 → BearColor) : Prop :=
  ∀ (i : Fin 998), 
    (bears i = BearColor.white ∨ bears i = BearColor.brown ∨ bears i = BearColor.black) ∧
    (bears ⟨i + 1, by linarith⟩ = BearColor.white ∨ bears ⟨i + 1, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 1, by linarith⟩ = BearColor.black) ∧
    (bears ⟨i + 2, by linarith⟩ = BearColor.white ∨ bears ⟨i + 2, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 2, by linarith⟩ = BearColor.black)

-- Iskander's guesses
def iskanderGuesses (bears : Fin 1000 → BearColor) : Prop :=
  bears 1 = BearColor.white ∧
  bears 19 = BearColor.brown ∧
  bears 399 = BearColor.black ∧
  bears 599 = BearColor.brown ∧
  bears 799 = BearColor.white

-- Exactly one guess is incorrect
def oneIncorrectGuess (bears : Fin 1000 → BearColor) : Prop :=
  ∃ (idx : Fin 5), 
    ¬iskanderGuesses bears ∧
    ∀ (j : Fin 5), (j ≠ idx → (bearGuessesIdx j bears = true))

-- The proof problem
theorem identifyIncorrectGuess (bears : Fin 1000 → BearColor) :
  isValidBearRow bears → iskanderGuesses bears → oneIncorrectGuess bears := sorry

end identifyIncorrectGuess_l129_129682


namespace logarithmic_function_fixed_point_l129_129535

def passes_through_fixed_point (a : ℝ) (x : ℝ) (y : ℝ) := a > 0 ∧ a ≠ 1 → ∃ x y, x = 2 ∧ y = -1 ∧ y = log a (x - 1) - 1

theorem logarithmic_function_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
  ∃ x y, x = 2 ∧ y = -1 ∧ y = log a (x - 1) - 1 :=
by { use [2, -1], split_and_repeat, sorry }

end logarithmic_function_fixed_point_l129_129535


namespace sin_cos_identity_l129_129984

theorem sin_cos_identity (α β γ : ℝ) (h : α + β + γ = 180) :
    Real.sin α + Real.sin β + Real.sin γ = 
    4 * Real.cos (α / 2) * Real.cos (β / 2) * Real.cos (γ / 2) := 
  sorry

end sin_cos_identity_l129_129984


namespace find_integer_pairs_l129_129027

theorem find_integer_pairs :
  {ab : ℤ × ℤ | ∃ (p : Polynomial ℤ), let (a, b) := ab in
    ((Polynomial.C a * X + Polynomial.C b) * p).coeffs.all (λ c, c = 1 ∨ c = -1)} =
  {ab : ℤ × ℤ | ab = (1, 1) ∨ ab = (1, -1) ∨ ab = (-1, 1) ∨ ab = (-1, -1) ∨
                  ab = (0, 1) ∨ ab = (0, -1) ∨
                  ab = (2, 1) ∨ ab = (2, -1) ∨ ab = (-2, 1) ∨ ab = (-2, -1)} :=
begin
  sorry
end

end find_integer_pairs_l129_129027


namespace number_of_groups_is_correct_l129_129645

-- Defining the conditions
def new_players : Nat := 48
def returning_players : Nat := 6
def players_per_group : Nat := 6
def total_players : Nat := new_players + returning_players

-- Theorem to prove the number of groups
theorem number_of_groups_is_correct : total_players / players_per_group = 9 := by
  sorry

end number_of_groups_is_correct_l129_129645


namespace inverse_function_value_l129_129504

theorem inverse_function_value (f : ℝ → ℝ) (h₁ : Function.HasInverse f)
  (h₂ : f 2 = -1) : Function.invFun f (-1) = 2 :=
sorry

end inverse_function_value_l129_129504


namespace bananas_left_l129_129420

-- Definitions based on conditions
def original_bananas : ℕ := 46
def bananas_removed : ℕ := 5

-- Statement of the problem using the definitions
theorem bananas_left : original_bananas - bananas_removed = 41 :=
by sorry

end bananas_left_l129_129420


namespace rate_of_interest_same_sum_l129_129267

theorem rate_of_interest_same_sum (P R' T : ℝ) (SI : ℝ) (hP : P = 2100) (hSI : SI = 840) :
  (SI = (P * R' * T) / 100) -> T = 5 -> R' = 8 :=
by
  intro h hT
  simp at h
  sorry

end rate_of_interest_same_sum_l129_129267


namespace max_enclosed_area_l129_129187

-- Given conditions: total fencing is 160 feet, the garden is a square, and side lengths are natural numbers
def perimeter := 160
def side_length {s : ℕ} (h : 4 * s = perimeter) : s := s

theorem max_enclosed_area (s : ℕ) (h : 4 * s = 160) : s ^ 2 = 1600 :=
by
  have h1 : s = 40 := by linarith
  rw [h1, pow_two]
  norm_num

#check max_enclosed_area

end max_enclosed_area_l129_129187


namespace triangle_inequality_for_n6_l129_129035

variables {a b c : ℝ} {n : ℕ}
open Real

-- Define the main statement as a theorem
theorem triangle_inequality_for_n6 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_for_n6_l129_129035


namespace evaluate_g_neg5_l129_129537

def g (x : ℝ) : ℝ := 4 * x - 2

theorem evaluate_g_neg5 : g (-5) = -22 := 
  by sorry

end evaluate_g_neg5_l129_129537


namespace find_integer_k_l129_129867

theorem find_integer_k (k x : ℤ) (h : (k^2 - 1) * x^2 - 6 * (3 * k - 1) * x + 72 = 0) (hx : x > 0) :
  k = 1 ∨ k = 2 ∨ k = 3 :=
sorry

end find_integer_k_l129_129867


namespace depth_multiple_of_rons_height_l129_129629

theorem depth_multiple_of_rons_height (h d : ℕ) (Ron_height : h = 13) (water_depth : d = 208) : d = 16 * h := by
  sorry

end depth_multiple_of_rons_height_l129_129629


namespace directrix_of_parabola_l129_129444

def parabola (x : ℝ) : ℝ := (x^2 - 8*x + 12) / 16

theorem directrix_of_parabola :
  ∀ x, parabola x = (x-4)^2 / 16 - 1/4 →
  let a := 1/16 in
  let h := 4 in
  let k := -1/4 in
  let directrix := k - 1/(4*a) in
  directrix = -17/4 :=
by
  intro x h1
  simp only [parabola] at h1
  dsimp [a, h, k] at h1
  have := calc
    k - 1/(4*a) = -1/4 - 4 : by field_simp [a]
    ... = -17/4 : by norm_num
  exact this

end directrix_of_parabola_l129_129444


namespace problem_3_l129_129508

variable (f : ℝ → ℝ) (t : ℝ)

-- Conditions as per the problem
def condition_2 (f : ℝ → ℝ) (t : ℝ) : Prop :=
∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < t) → (0 < x2 ∧ x2 < t) → x1 < x2 → f x1 > f x2

def max_value_t (f : ℝ → ℝ) : Prop :=
∀ t, condition_2 f t → t ≥ π

-- Let f(x) = sin(x)/x satisfying condition 2 and prove t_max ≥ π
theorem problem_3 : 
  max_value_t (λ x : ℝ, sin x / x) :=
sorry

end problem_3_l129_129508


namespace smallest_number_of_divisors_l129_129050

-- The condition 24 | n + 1
def condition_1 (n : ℕ) : Prop := 24 ∣ n + 1

-- The condition that the sum of squares of all divisors of n is divisible by 48
def condition_2 (n : ℕ) : Prop :=
  let divs := (List.divisors n).toFinset
  48 ∣ divs.sum (λ d, d * d)

-- The number of divisors of n
def number_of_divisors (n : ℕ) : ℕ :=
  (List.divisors n).toFinset.card

-- Prove that if both conditions hold, the smallest number of divisors n has is 48
theorem smallest_number_of_divisors (n : ℕ) :
  condition_1 n →
  condition_2 n →
  number_of_divisors n = 48 :=
sorry

end smallest_number_of_divisors_l129_129050


namespace interest_rate_first_bank_l129_129755

theorem interest_rate_first_bank :
  ∃ (r : ℝ), 5000 = 1700 + 3300 ∧ 
             0.065 = 6.5 / 100 ∧ 
             282.50 = 1700 * r + 3300 * 0.065 ∧ 
             r = 0.04 :=
by
  use 0.04
  split
  { exact rfl }
  split
  { norm_num }
  split
  { norm_num }
  { exact rfl }

end interest_rate_first_bank_l129_129755


namespace Dan_work_hours_l129_129807

theorem Dan_work_hours (x : ℝ) :
  (1 / 15) * x + 3 / 5 = 1 → x = 6 :=
by
  intro h
  sorry

end Dan_work_hours_l129_129807


namespace count_tree_automorphisms_l129_129588

def V : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8}

def isTree (G : SimpleGraph V) : Prop :=
  G.connected ∧ G.edgeFinset.card = V.card - 1

def isAutomorphismOfTree (σ : V → V) (G : SimpleGraph V) : Prop :=
  ∀ (i j : V), G.Adj i j ↔ G.Adj (σ i) (σ j)

theorem count_tree_automorphisms :
  let σ_count := {σ : V → V // ∃ G : SimpleGraph V, isTree G ∧ isAutomorphismOfTree σ G}.card
  σ_count = 30212 :=
sorry

end count_tree_automorphisms_l129_129588


namespace total_books_l129_129702

theorem total_books (x : ℕ) (h1 : 3 * x + 2 * x + (3 / 2) * x > 3000) : 
  ∃ (T : ℕ), T = 3 * x + 2 * x + (3 / 2) * x ∧ T > 3000 ∧ T = 3003 := 
by 
  -- Our theorem states there exists an integer T such that the total number of books is 3003.
  sorry

end total_books_l129_129702


namespace pentagon_x_eq_l129_129785

variable {Point : Type}
variable [Plane Point]
variable {A B C D E X : Point}

-- Assuming the pentagon is regular and X lies on the given arc of the circumcircle
variable (h1 : RegularPentagon A B C D E)
variable (h2 : OnArcCircumcircle A E X)

-- The theorem to be proved
theorem pentagon_x_eq (h1 : RegularPentagon A B C D E) (h2 : OnArcCircumcircle A E X) :
  dist A X + dist C X + dist E X = dist B X + dist D X :=
sorry

end pentagon_x_eq_l129_129785


namespace inequality_solution_set_l129_129271

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : 
  (1 / x > 3) ↔ (0 < x ∧ x < 1 / 3) := 
by 
  sorry

end inequality_solution_set_l129_129271


namespace problem_statement_l129_129637

variable (n : ℕ)
variable (a : ℕ → ℕ → ℕ) (b : ℕ → ℕ)

-- Conditions of the problem
-- a_ij is the number in position (i,j)
-- b_j is the number of possible values for a_jj
-- We assume the array has numbers in increasing order in each row and column

noncomputable def sum_b_eq : Prop :=
  ∑ i in finset.range n, b i = n * (n * n - 3 * n + 5) / 3

-- The final proposition to prove
theorem problem_statement : sum_b_eq n a b :=
sorry

end problem_statement_l129_129637


namespace consecutive_days_without_meeting_l129_129129

/-- In March 1987, there are 31 days, starting on a Sunday.
There are 11 club meetings to be held, and no meetings are on Saturdays or Sundays.
This theorem proves that there will be at least three consecutive days without a meeting. -/
theorem consecutive_days_without_meeting (meetings : Finset ℕ) :
  (∀ x ∈ meetings, 1 ≤ x ∧ x ≤ 31 ∧ ¬ ∃ k, x = 7 * k + 1 ∨ x = 7 * k + 2) →
  meetings.card = 11 →
  ∃ i, 1 ≤ i ∧ i + 2 ≤ 31 ∧ ¬ (i ∈ meetings ∨ (i + 1) ∈ meetings ∨ (i + 2) ∈ meetings) :=
by
  sorry

end consecutive_days_without_meeting_l129_129129


namespace solve_log_inequality_l129_129621

theorem solve_log_inequality (x : ℝ) :
  (x^2 - 2 * x - 3 > 0) → (log (x^2 - 2 * x - 3) / log 10 ≥ 0 ↔ x ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioi 3) :=
by
  sorry

end solve_log_inequality_l129_129621
