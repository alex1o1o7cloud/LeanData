import Mathlib

namespace NUMINAMATH_GPT_max_three_digit_sum_l819_81962

theorem max_three_digit_sum : ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (0 ≤ A ∧ A < 10) ∧ (0 ≤ B ∧ B < 10) ∧ (0 ≤ C ∧ C < 10) ∧ (111 * A + 10 * C + 2 * B = 976) := sorry

end NUMINAMATH_GPT_max_three_digit_sum_l819_81962


namespace NUMINAMATH_GPT_n_divides_2n_plus_1_implies_multiple_of_3_l819_81923

theorem n_divides_2n_plus_1_implies_multiple_of_3 {n : ℕ} (h₁ : n ≥ 2) (h₂ : n ∣ (2^n + 1)) : 3 ∣ n :=
sorry

end NUMINAMATH_GPT_n_divides_2n_plus_1_implies_multiple_of_3_l819_81923


namespace NUMINAMATH_GPT_find_k_l819_81961

-- Definitions for the conditions and the main theorem.
variables {x y k : ℝ}

-- The first equation of the system
def eq1 (x y k : ℝ) : Prop := 2 * x + 5 * y = k

-- The second equation of the system
def eq2 (x y : ℝ) : Prop := x - 4 * y = 15

-- Condition that x and y are opposites
def are_opposites (x y : ℝ) : Prop := x + y = 0

-- The theorem to prove
theorem find_k (hk : ∃ (x y : ℝ), eq1 x y k ∧ eq2 x y ∧ are_opposites x y) : k = -9 :=
sorry

end NUMINAMATH_GPT_find_k_l819_81961


namespace NUMINAMATH_GPT_polynomial_ascending_l819_81960

theorem polynomial_ascending (x : ℝ) :
  (x^2 - 2 - 5*x^4 + 3*x^3) = (-2 + x^2 + 3*x^3 - 5*x^4) :=
by sorry

end NUMINAMATH_GPT_polynomial_ascending_l819_81960


namespace NUMINAMATH_GPT_range_of_a_for_intersections_l819_81957

theorem range_of_a_for_intersections (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    (x₁^3 - 3 * x₁ = a) ∧ (x₂^3 - 3 * x₂ = a) ∧ (x₃^3 - 3 * x₃ = a)) ↔ 
  (-2 < a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_intersections_l819_81957


namespace NUMINAMATH_GPT_total_cement_used_l819_81955

def cement_used_lexi : ℝ := 10
def cement_used_tess : ℝ := 5.1

theorem total_cement_used : cement_used_lexi + cement_used_tess = 15.1 :=
by sorry

end NUMINAMATH_GPT_total_cement_used_l819_81955


namespace NUMINAMATH_GPT_point_on_coordinate_axes_l819_81933

theorem point_on_coordinate_axes {x y : ℝ} 
  (h : x * y = 0) : (x = 0 ∨ y = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_point_on_coordinate_axes_l819_81933


namespace NUMINAMATH_GPT_polynomial_expansion_l819_81922

theorem polynomial_expansion (x : ℝ) : 
  (1 - x^3) * (1 + x^4 - x^5) = 1 - x^3 + x^4 - x^5 - x^7 + x^8 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l819_81922


namespace NUMINAMATH_GPT_decimal_to_base8_conversion_l819_81901

theorem decimal_to_base8_conversion : (512 : ℕ) = 8^3 :=
by
  sorry

end NUMINAMATH_GPT_decimal_to_base8_conversion_l819_81901


namespace NUMINAMATH_GPT_exists_intersecting_line_l819_81988

/-- Represents a segment as a pair of endpoints in a 2D plane. -/
structure Segment where
  x : ℝ
  y1 : ℝ
  y2 : ℝ

open Segment

/-- Given several parallel segments with the property that for any three of these segments, 
there exists a line that intersects all three of them, prove that 
there is a line that intersects all the segments. -/
theorem exists_intersecting_line (segments : List Segment)
  (h : ∀ s1 s2 s3 : Segment, s1 ∈ segments → s2 ∈ segments → s3 ∈ segments → 
       ∃ a b : ℝ, (s1.y1 <= a * s1.x + b) ∧ (a * s1.x + b <= s1.y2) ∧ 
                   (s2.y1 <= a * s2.x + b) ∧ (a * s2.x + b <= s2.y2) ∧ 
                   (s3.y1 <= a * s3.x + b) ∧ (a * s3.x + b <= s3.y2)) :
  ∃ a b : ℝ, ∀ s : Segment, s ∈ segments → (s.y1 <= a * s.x + b) ∧ (a * s.x + b <= s.y2) := 
sorry

end NUMINAMATH_GPT_exists_intersecting_line_l819_81988


namespace NUMINAMATH_GPT_maximize_sqrt_expression_l819_81964

theorem maximize_sqrt_expression :
  let a := Real.sqrt 8
  let b := Real.sqrt 2
  (a + b) > max (max (a - b) (a * b)) (a / b) := by
  sorry

end NUMINAMATH_GPT_maximize_sqrt_expression_l819_81964


namespace NUMINAMATH_GPT_jack_total_cost_l819_81997

def plan_base_cost : ℕ := 25

def cost_per_text : ℕ := 8

def free_hours : ℕ := 25

def cost_per_extra_minute : ℕ := 10

def texts_sent : ℕ := 150

def hours_talked : ℕ := 26

def total_cost (base_cost : ℕ) (texts_sent : ℕ) (cost_per_text : ℕ) (hours_talked : ℕ) 
               (free_hours : ℕ) (cost_per_extra_minute : ℕ) : ℕ :=
  base_cost + (texts_sent * cost_per_text) / 100 + 
  ((hours_talked - free_hours) * 60 * cost_per_extra_minute) / 100

theorem jack_total_cost : 
  total_cost plan_base_cost texts_sent cost_per_text hours_talked free_hours cost_per_extra_minute = 43 :=
by
  sorry

end NUMINAMATH_GPT_jack_total_cost_l819_81997


namespace NUMINAMATH_GPT_triangle_construction_feasible_l819_81944

theorem triangle_construction_feasible (a b s : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (a - b) / 2 < s) (h4 : s < (a + b) / 2) :
  ∃ c, (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

end NUMINAMATH_GPT_triangle_construction_feasible_l819_81944


namespace NUMINAMATH_GPT_train_distance_l819_81942

def fuel_efficiency := 5 / 2 
def coal_remaining := 160
def expected_distance := 400

theorem train_distance : fuel_efficiency * coal_remaining = expected_distance := 
by
  sorry

end NUMINAMATH_GPT_train_distance_l819_81942


namespace NUMINAMATH_GPT_days_of_earning_l819_81991

theorem days_of_earning (T D d : ℕ) (hT : T = 165) (hD : D = 33) (h : d = T / D) :
  d = 5 :=
by sorry

end NUMINAMATH_GPT_days_of_earning_l819_81991


namespace NUMINAMATH_GPT_problem_1_problem_2_l819_81928

noncomputable def f (a b x : ℝ) := |x + a| + |2 * x - b|

theorem problem_1 (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
(h_min : ∀ x, f a b x ≥ 1 ∧ (∃ x₀, f a b x₀ = 1)) :
2 * a + b = 2 :=
sorry

theorem problem_2 (a b t : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
(h_tab : ∀ t > 0, a + 2 * b ≥ t * a * b)
(h_eq : 2 * a + b = 2) :
t ≤ 9 / 2 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l819_81928


namespace NUMINAMATH_GPT_solve_for_k_l819_81914

theorem solve_for_k (t k : ℝ) (h1 : t = 5 / 9 * (k - 32)) (h2 : t = 105) : k = 221 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l819_81914


namespace NUMINAMATH_GPT_certain_number_divisibility_l819_81974

theorem certain_number_divisibility (n : ℕ) (p : ℕ) (h : p = 1) (h2 : 4864 * 9 * n % 12 = 0) : n = 43776 :=
by {
  sorry
}

end NUMINAMATH_GPT_certain_number_divisibility_l819_81974


namespace NUMINAMATH_GPT_speed_of_train_in_km_per_hr_l819_81956

-- Definitions for the condition
def length_of_train : ℝ := 180 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor
def meters_per_second_to_kilometers_per_hour (speed : ℝ) := speed * 3.6

-- Proof statement
theorem speed_of_train_in_km_per_hr : 
  meters_per_second_to_kilometers_per_hour (length_of_train / time_to_cross_pole) = 72 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_train_in_km_per_hr_l819_81956


namespace NUMINAMATH_GPT_probability_of_three_blue_marbles_l819_81929

theorem probability_of_three_blue_marbles
  (red_marbles : ℕ) (blue_marbles : ℕ) (yellow_marbles : ℕ) (total_marbles : ℕ)
  (draws : ℕ) 
  (prob : ℚ) :
  red_marbles = 3 →
  blue_marbles = 4 →
  yellow_marbles = 13 →
  total_marbles = 20 →
  draws = 3 →
  prob = ((4 / 20) * (3 / 19) * (1 / 9)) →
  prob = 1 / 285 :=
by
  intros; 
  sorry

end NUMINAMATH_GPT_probability_of_three_blue_marbles_l819_81929


namespace NUMINAMATH_GPT_total_surface_area_of_prism_l819_81990

-- Define the conditions of the problem
def sphere_radius (R : ℝ) := R > 0
def prism_circumscribed_around_sphere (R : ℝ) := True  -- Placeholder as the concept assertion, actual geometry handling not needed here
def prism_height (R : ℝ) := 2 * R

-- Define the main theorem to be proved
theorem total_surface_area_of_prism (R : ℝ) (hR : sphere_radius R) (hCircumscribed : prism_circumscribed_around_sphere R) (hHeight : prism_height R = 2 * R) : 
  ∃ (S : ℝ), S = 12 * R^2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_total_surface_area_of_prism_l819_81990


namespace NUMINAMATH_GPT_solve_quadratic_eq1_solve_quadratic_eq2_l819_81971

-- Define the first equation
theorem solve_quadratic_eq1 (x : ℝ) : x^2 - 6 * x - 6 = 0 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15 := by
  sorry

-- Define the second equation
theorem solve_quadratic_eq2 (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 ∨ x = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq1_solve_quadratic_eq2_l819_81971


namespace NUMINAMATH_GPT_sum_of_integers_from_1_to_10_l819_81980

theorem sum_of_integers_from_1_to_10 :
  (Finset.range 11).sum id = 55 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_from_1_to_10_l819_81980


namespace NUMINAMATH_GPT_parallel_planes_perpendicular_planes_l819_81966

variables {A1 B1 C1 D1 A2 B2 C2 D2 : ℝ}

-- Parallelism Condition
theorem parallel_planes (h₁ : A1 ≠ 0) (h₂ : B1 ≠ 0) (h₃ : C1 ≠ 0) (h₄ : A2 ≠ 0) (h₅ : B2 ≠ 0) (h₆ : C2 ≠ 0) :
  (A1 / A2 = B1 / B2 ∧ B1 / B2 = C1 / C2) ↔ (∃ k : ℝ, (A1 = k * A2) ∧ (B1 = k * B2) ∧ (C1 = k * C2)) :=
sorry

-- Perpendicularity Condition
theorem perpendicular_planes :
  A1 * A2 + B1 * B2 + C1 * C2 = 0 :=
sorry

end NUMINAMATH_GPT_parallel_planes_perpendicular_planes_l819_81966


namespace NUMINAMATH_GPT_least_positive_integer_is_4619_l819_81954

noncomputable def least_positive_integer (N : ℕ) : Prop :=
  N % 4 = 3 ∧
  N % 5 = 4 ∧
  N % 6 = 5 ∧
  N % 7 = 6 ∧
  N % 11 = 10 ∧
  ∀ M : ℕ, (M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M % 11 = 10) → N ≤ M

theorem least_positive_integer_is_4619 : least_positive_integer 4619 :=
  sorry

end NUMINAMATH_GPT_least_positive_integer_is_4619_l819_81954


namespace NUMINAMATH_GPT_commute_days_l819_81927

theorem commute_days (a b d e x : ℕ) 
  (h1 : b + e = 12)
  (h2 : a + d = 20)
  (h3 : a + b = 15)
  (h4 : x = a + b + d + e) :
  x = 32 :=
by {
  sorry
}

end NUMINAMATH_GPT_commute_days_l819_81927


namespace NUMINAMATH_GPT_max_books_borrowed_l819_81912

theorem max_books_borrowed (total_students : ℕ) (students_no_books : ℕ) (students_1_book : ℕ)
  (students_2_books : ℕ) (avg_books_per_student : ℕ) (remaining_students_borrowed_at_least_3 :
  ∀ (s : ℕ), s ≥ 3) :
  total_students = 25 →
  students_no_books = 3 →
  students_1_book = 11 →
  students_2_books = 6 →
  avg_books_per_student = 2 →
  ∃ (max_books : ℕ), max_books = 15 :=
  by
  sorry

end NUMINAMATH_GPT_max_books_borrowed_l819_81912


namespace NUMINAMATH_GPT_quadratic_roots_real_and_values_l819_81900

theorem quadratic_roots_real_and_values (m : ℝ) (x : ℝ) :
  (x ^ 2 - x + 2 * m - 2 = 0) → (m ≤ 9 / 8) ∧ (m = 1 → (x = 0 ∨ x = 1)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_real_and_values_l819_81900


namespace NUMINAMATH_GPT_tim_younger_than_jenny_l819_81996

def tim_age : ℕ := 5
def rommel_age : ℕ := 3 * tim_age
def jenny_age : ℕ := rommel_age + 2
def combined_ages_rommel_jenny : ℕ := rommel_age + jenny_age
def uncle_age : ℕ := 2 * combined_ages_rommel_jenny
noncomputable def aunt_age : ℝ := (uncle_age + jenny_age : ℕ) / 2

theorem tim_younger_than_jenny : jenny_age - tim_age = 12 :=
by {
  -- Placeholder proof
  sorry
}

end NUMINAMATH_GPT_tim_younger_than_jenny_l819_81996


namespace NUMINAMATH_GPT_mrs_hilt_read_chapters_l819_81951

-- Define the problem conditions
def books : ℕ := 4
def chapters_per_book : ℕ := 17

-- State the proof problem
theorem mrs_hilt_read_chapters : (books * chapters_per_book) = 68 := 
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_read_chapters_l819_81951


namespace NUMINAMATH_GPT_solve_equation_l819_81940

theorem solve_equation :
  (3 * x - 6 = abs (-21 + 8 - 3)) → x = 22 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l819_81940


namespace NUMINAMATH_GPT_find_xy_l819_81979

theorem find_xy (x y : ℝ) :
  (x - 8) ^ 2 + (y - 9) ^ 2 + (x - y) ^ 2 = 1 / 3 ↔ 
  (x = 25 / 3 ∧ y = 26 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l819_81979


namespace NUMINAMATH_GPT_ab_divides_a_squared_plus_b_squared_l819_81950

theorem ab_divides_a_squared_plus_b_squared (a b : ℕ) (hab : a ≠ 1 ∨ b ≠ 1) (hpos : 0 < a ∧ 0 < b) (hdiv : (ab - 1) ∣ (a^2 + b^2)) :
  a^2 + b^2 = 5 * a * b - 5 := 
by
  sorry

end NUMINAMATH_GPT_ab_divides_a_squared_plus_b_squared_l819_81950


namespace NUMINAMATH_GPT_darnel_jogging_l819_81982

variable (j s : ℝ)

theorem darnel_jogging :
  s = 0.875 ∧ s = j + 0.125 → j = 0.750 :=
by
  intros h
  have h1 : s = 0.875 := h.1
  have h2 : s = j + 0.125 := h.2
  sorry

end NUMINAMATH_GPT_darnel_jogging_l819_81982


namespace NUMINAMATH_GPT_tangent_line_ln_x_xsq_l819_81976

theorem tangent_line_ln_x_xsq (x y : ℝ) (h_curve : y = Real.log x + x^2) (h_point : (x, y) = (1, 1)) :
  3 * x - y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_ln_x_xsq_l819_81976


namespace NUMINAMATH_GPT_num_diagonals_increase_by_n_l819_81907

-- Definitions of the conditions
def num_diagonals (n : ℕ) : ℕ := sorry  -- Consider f(n) to be a function that calculates diagonals for n-sided polygon

-- Lean 4 proof problem statement
theorem num_diagonals_increase_by_n (n : ℕ) :
  num_diagonals (n + 1) = num_diagonals n + n :=
sorry

end NUMINAMATH_GPT_num_diagonals_increase_by_n_l819_81907


namespace NUMINAMATH_GPT_exists_integers_u_v_l819_81908

theorem exists_integers_u_v (A : ℕ) (a b s : ℤ)
  (hA: A = 1 ∨ A = 2 ∨ A = 3)
  (hab_rel_prime: Int.gcd a b = 1)
  (h_eq: a^2 + A * b^2 = s^3) :
  ∃ u v : ℤ, s = u^2 + A * v^2 ∧ a = u^3 - 3 * A * u * v^2 ∧ b = 3 * u^2 * v - A * v^3 := 
sorry

end NUMINAMATH_GPT_exists_integers_u_v_l819_81908


namespace NUMINAMATH_GPT_sqrt_mul_l819_81998

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_mul_l819_81998


namespace NUMINAMATH_GPT_available_seats_l819_81946

/-- Two-fifths of the seats in an auditorium that holds 500 people are currently taken. --/
def seats_taken : ℕ := (2 * 500) / 5

/-- One-tenth of the seats in an auditorium that holds 500 people are broken. --/
def seats_broken : ℕ := 500 / 10

/-- Total seats in the auditorium --/
def total_seats := 500

/-- There are 500 total seats in an auditorium. Two-fifths of the seats are taken and 
one-tenth are broken. Prove that the number of seats still available is 250. --/
theorem available_seats : (total_seats - seats_taken - seats_broken) = 250 :=
by 
  sorry

end NUMINAMATH_GPT_available_seats_l819_81946


namespace NUMINAMATH_GPT_only_one_solution_l819_81947

theorem only_one_solution (n : ℕ) (h : 0 < n ∧ ∃ a : ℕ, a * a = 5^n + 4) : n = 1 :=
sorry

end NUMINAMATH_GPT_only_one_solution_l819_81947


namespace NUMINAMATH_GPT_lukas_games_played_l819_81911

-- Define the given conditions
def average_points_per_game : ℕ := 12
def total_points_scored : ℕ := 60

-- Define Lukas' number of games
def number_of_games (total_points : ℕ) (average_points : ℕ) : ℕ :=
  total_points / average_points

-- Theorem and statement to prove
theorem lukas_games_played :
  number_of_games total_points_scored average_points_per_game = 5 :=
by
  sorry

end NUMINAMATH_GPT_lukas_games_played_l819_81911


namespace NUMINAMATH_GPT_Megan_deleted_pictures_l819_81909

/--
Megan took 15 pictures at the zoo and 18 at the museum. She still has 2 pictures from her vacation.
Prove that Megan deleted 31 pictures.
-/
theorem Megan_deleted_pictures :
  let zoo_pictures := 15
  let museum_pictures := 18
  let remaining_pictures := 2
  let total_pictures := zoo_pictures + museum_pictures
  let deleted_pictures := total_pictures - remaining_pictures
  deleted_pictures = 31 :=
by
  sorry

end NUMINAMATH_GPT_Megan_deleted_pictures_l819_81909


namespace NUMINAMATH_GPT_sin_of_angle_l819_81992

theorem sin_of_angle (α : ℝ) (x y : ℝ) (h1 : x = -3) (h2 : y = -4) (r : ℝ) (hr : r = Real.sqrt (x^2 + y^2)) : 
  Real.sin α = -4 / r := 
by
  -- Definitions
  let y := -4
  let x := -3
  let r := Real.sqrt (x^2 + y^2)
  -- Proof
  sorry

end NUMINAMATH_GPT_sin_of_angle_l819_81992


namespace NUMINAMATH_GPT_election_votes_and_deposit_l819_81963

theorem election_votes_and_deposit (V : ℕ) (A B C D E : ℕ) (hA : A = 40 * V / 100) 
  (hB : B = 28 * V / 100) (hC : C = 20 * V / 100) (hDE : D + E = 12 * V / 100)
  (win_margin : A - B = 500) :
  V = 4167 ∧ (15 * V / 100 ≤ A) ∧ (15 * V / 100 ≤ B) ∧ (15 * V / 100 ≤ C) ∧ 
  ¬ (15 * V / 100 ≤ D) ∧ ¬ (15 * V / 100 ≤ E) :=
by 
  sorry

end NUMINAMATH_GPT_election_votes_and_deposit_l819_81963


namespace NUMINAMATH_GPT_fish_count_l819_81977

theorem fish_count (T : ℕ) :
  (T > 10 ∧ T ≤ 18) ∧ ((T > 18 ∧ T > 15 ∧ ¬(T > 10)) ∨ (¬(T > 18) ∧ T > 15 ∧ T > 10) ∨ (T > 18 ∧ ¬(T > 15) ∧ T > 10)) →
  T = 16 ∨ T = 17 ∨ T = 18 :=
sorry

end NUMINAMATH_GPT_fish_count_l819_81977


namespace NUMINAMATH_GPT_find_fake_coin_l819_81913

theorem find_fake_coin (k : ℕ) :
  ∃ (weighings : ℕ), (weighings ≤ 3 * k + 1) :=
sorry

end NUMINAMATH_GPT_find_fake_coin_l819_81913


namespace NUMINAMATH_GPT_whisker_ratio_l819_81915

theorem whisker_ratio 
  (p : ℕ) (c : ℕ) (h1 : p = 14) (h2 : c = 22) (s := c + 6) :
  s / p = 2 := 
by
  sorry

end NUMINAMATH_GPT_whisker_ratio_l819_81915


namespace NUMINAMATH_GPT_value_of_a_l819_81993

theorem value_of_a (a : ℝ) : (a^2 - 4) / (a - 2) = 0 → a ≠ 2 → a = -2 :=
by 
  intro h1 h2
  sorry

end NUMINAMATH_GPT_value_of_a_l819_81993


namespace NUMINAMATH_GPT_find_x_given_k_l819_81973

-- Define the equation under consideration
def equation (x : ℝ) : Prop := (x - 3) / (x - 4) = (x - 5) / (x - 8)

theorem find_x_given_k {k : ℝ} (h : k = 7) : ∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → equation x → x = 2 :=
by
  intro x hx h_eq
  sorry

end NUMINAMATH_GPT_find_x_given_k_l819_81973


namespace NUMINAMATH_GPT_pencils_purchased_l819_81938

theorem pencils_purchased 
  (total_cost : ℝ)
  (num_pens : ℕ)
  (price_per_pen : ℝ)
  (price_per_pencil : ℝ)
  (total_cost_condition : total_cost = 510)
  (num_pens_condition : num_pens = 30)
  (price_per_pen_condition : price_per_pen = 12)
  (price_per_pencil_condition : price_per_pencil = 2) :
  num_pens * price_per_pen + sorry = total_cost →
  150 / price_per_pencil = 75 :=
by
  sorry

end NUMINAMATH_GPT_pencils_purchased_l819_81938


namespace NUMINAMATH_GPT_cone_height_l819_81931

theorem cone_height (r l h : ℝ) (h_r : r = 1) (h_l : l = 4) : h = Real.sqrt 15 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_cone_height_l819_81931


namespace NUMINAMATH_GPT_calculate_rolls_of_toilet_paper_l819_81903

-- Definitions based on the problem conditions
def seconds_per_egg := 15
def minutes_per_roll := 30
def total_cleaning_minutes := 225
def number_of_eggs := 60
def time_per_minute := 60

-- Calculation of the time spent on eggs in minutes
def egg_cleaning_minutes := (number_of_eggs * seconds_per_egg) / time_per_minute

-- Total cleaning time minus time spent on eggs
def remaining_cleaning_minutes := total_cleaning_minutes - egg_cleaning_minutes

-- Verify the number of rolls of toilet paper cleaned up
def rolls_of_toilet_paper := remaining_cleaning_minutes / minutes_per_roll

-- Theorem statement to be proved
theorem calculate_rolls_of_toilet_paper : rolls_of_toilet_paper = 7 := by
  sorry

end NUMINAMATH_GPT_calculate_rolls_of_toilet_paper_l819_81903


namespace NUMINAMATH_GPT_min_value_expression_l819_81935

open Classical

theorem min_value_expression (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h : 1/x + 1/y = 1) :
  ∃ (m : ℝ), m = 25 ∧ ∀ x y : ℝ, 0 < x → 0 < y → 1/x + 1/y = 1 → (4*x/(x - 1) + 9*y/(y - 1)) ≥ m :=
by 
  sorry

end NUMINAMATH_GPT_min_value_expression_l819_81935


namespace NUMINAMATH_GPT_find_number_l819_81905

theorem find_number (x : Real) (h1 : (2 / 5) * 300 = 120) (h2 : 120 - (3 / 5) * x = 45) : x = 125 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l819_81905


namespace NUMINAMATH_GPT_x_intercept_of_line_l819_81936

theorem x_intercept_of_line (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, -4)) (h2 : (x2, y2) = (6, 8)) : 
  ∃ x0 : ℝ, (x0 = (10 / 3) ∧ ∃ m : ℝ, m = (y2 - y1) / (x2 - x1) ∧ ∀ y : ℝ, y = m * x0 + b) := 
sorry

end NUMINAMATH_GPT_x_intercept_of_line_l819_81936


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l819_81975

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : a = 4) (h₂ : b = 9) (h₃ : ∀ x y z : ℕ, 
  (x = a ∧ y = a ∧ z = b) ∨ (x = b ∧ y = b ∧ z = a) → 
  (x + y > z ∧ x + z > y ∧ y + z > x)) : 
  (a = 4 ∧ b = 9) → a + a + b = 22 :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l819_81975


namespace NUMINAMATH_GPT_find_constants_l819_81910

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 2 →
    (3 * x + 7) / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) →
  A = 19 / 4 ∧ B = -19 / 4 ∧ C = -13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l819_81910


namespace NUMINAMATH_GPT_first_and_second_bags_l819_81969

def bags_apples (A B C : ℕ) : Prop :=
  (A + B + C = 24) ∧ (B + C = 18) ∧ (A + C = 19)

theorem first_and_second_bags (A B C : ℕ) (h : bags_apples A B C) :
  A + B = 11 :=
sorry

end NUMINAMATH_GPT_first_and_second_bags_l819_81969


namespace NUMINAMATH_GPT_initial_investment_l819_81941

theorem initial_investment (A P : ℝ) (r : ℝ) (n t : ℕ) 
  (hA : A = 16537.5)
  (hr : r = 0.10)
  (hn : n = 2)
  (ht : t = 1)
  (hA_calc : A = P * (1 + r / n) ^ (n * t)) :
  P = 15000 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_investment_l819_81941


namespace NUMINAMATH_GPT_hexagon_angle_R_l819_81978

theorem hexagon_angle_R (F I G U R E : ℝ) 
  (h1 : F = I ∧ I = R ∧ R = E)
  (h2 : G + U = 180) 
  (sum_angles_hexagon : F + I + G + U + R + E = 720) : 
  R = 135 :=
by sorry

end NUMINAMATH_GPT_hexagon_angle_R_l819_81978


namespace NUMINAMATH_GPT_find_m_l819_81939

noncomputable def A (m : ℝ) : Set ℝ := {1, 3, 2 * m + 3}
noncomputable def B (m : ℝ) : Set ℝ := {3, m^2}

theorem find_m (m : ℝ) : B m ⊆ A m ↔ m = 1 ∨ m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l819_81939


namespace NUMINAMATH_GPT_base_conversion_subtraction_l819_81920

def base6_to_nat (d0 d1 d2 d3 d4 : ℕ) : ℕ :=
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0 * 6^0

def base7_to_nat (d0 d1 d2 d3 : ℕ) : ℕ :=
  d3 * 7^3 + d2 * 7^2 + d1 * 7^1 + d0 * 7^0

theorem base_conversion_subtraction :
  base6_to_nat 1 2 3 5 4 - base7_to_nat 1 2 3 4 = 4851 := by
  sorry

end NUMINAMATH_GPT_base_conversion_subtraction_l819_81920


namespace NUMINAMATH_GPT_distance_between_trees_l819_81999

theorem distance_between_trees (n : ℕ) (len : ℝ) (d : ℝ) 
  (h1 : n = 26) 
  (h2 : len = 400) 
  (h3 : len / (n - 1) = d) : 
  d = 16 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l819_81999


namespace NUMINAMATH_GPT_initial_passengers_l819_81984

theorem initial_passengers (P : ℕ) (H1 : P - 263 + 419 = 725) : P = 569 :=
by
  sorry

end NUMINAMATH_GPT_initial_passengers_l819_81984


namespace NUMINAMATH_GPT_escalator_rate_l819_81906

theorem escalator_rate
  (length_escalator : ℕ) 
  (person_speed : ℕ) 
  (time_taken : ℕ) 
  (total_length : length_escalator = 112) 
  (person_speed_rate : person_speed = 4)
  (time_taken_rate : time_taken = 8) :
  ∃ v : ℕ, (person_speed + v) * time_taken = length_escalator ∧ v = 10 :=
by
  sorry

end NUMINAMATH_GPT_escalator_rate_l819_81906


namespace NUMINAMATH_GPT_sum_of_ages_of_sarahs_friends_l819_81983

noncomputable def sum_of_ages (a b c : ℕ) : ℕ := a + b + c

theorem sum_of_ages_of_sarahs_friends (a b c : ℕ) (h_distinct : ∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_single_digits : ∀ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10)
  (h_product_36 : ∃ (x y : ℕ), x * y = 36 ∧ x ≠ y)
  (h_factor_36 : ∀ (x y z : ℕ), x ∣ 36 ∧ y ∣ 36 ∧ z ∣ 36) :
  ∃ (a b c : ℕ), sum_of_ages a b c = 16 := 
sorry

end NUMINAMATH_GPT_sum_of_ages_of_sarahs_friends_l819_81983


namespace NUMINAMATH_GPT_product_of_primes_is_even_l819_81965

-- Define the conditions for P and Q to cover P, Q, P-Q, and P+Q being prime and positive
def is_prime (n : ℕ) : Prop := ¬ (n = 0 ∨ n = 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem product_of_primes_is_even {P Q : ℕ} (hP : is_prime P) (hQ : is_prime Q) 
  (hPQ_diff : is_prime (P - Q)) (hPQ_sum : is_prime (P + Q)) 
  (hPosP : P > 0) (hPosQ : Q > 0) 
  (hPosPQ_diff : P - Q > 0) (hPosPQ_sum : P + Q > 0) : 
  ∃ k : ℕ, P * Q * (P - Q) * (P + Q) = 2 * k := 
sorry

end NUMINAMATH_GPT_product_of_primes_is_even_l819_81965


namespace NUMINAMATH_GPT_solve_quadratic_equation_l819_81932

theorem solve_quadratic_equation (x : ℝ) :
  2 * x * (x + 1) = x + 1 ↔ (x = -1 ∨ x = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l819_81932


namespace NUMINAMATH_GPT_Michelle_bought_14_chocolate_bars_l819_81917

-- Definitions for conditions
def sugar_per_chocolate_bar : ℕ := 10
def sugar_in_lollipop : ℕ := 37
def total_sugar_in_candy : ℕ := 177

-- Theorem to prove
theorem Michelle_bought_14_chocolate_bars :
  (total_sugar_in_candy - sugar_in_lollipop) / sugar_per_chocolate_bar = 14 :=
by
  -- Proof steps will go here, but are omitted as per the requirements.
  sorry

end NUMINAMATH_GPT_Michelle_bought_14_chocolate_bars_l819_81917


namespace NUMINAMATH_GPT_no_perfect_squares_l819_81934

theorem no_perfect_squares (x y : ℕ) : ¬ (∃ a b : ℕ, x^2 + y = a^2 ∧ x + y^2 = b^2) :=
sorry

end NUMINAMATH_GPT_no_perfect_squares_l819_81934


namespace NUMINAMATH_GPT_day_of_week_after_2_power_50_days_l819_81958

-- Conditions:
def today_is_monday : ℕ := 1  -- Monday corresponds to 1

def days_later (n : ℕ) : ℕ := (today_is_monday + n) % 7

theorem day_of_week_after_2_power_50_days :
  days_later (2^50) = 6 :=  -- Saturday corresponds to 6 (0 is Sunday)
by {
  -- Proof steps are skipped
  sorry
}

end NUMINAMATH_GPT_day_of_week_after_2_power_50_days_l819_81958


namespace NUMINAMATH_GPT_total_days_2003_to_2006_l819_81930

theorem total_days_2003_to_2006 : 
  let days_2003 := 365
  let days_2004 := 366
  let days_2005 := 365
  let days_2006 := 365
  days_2003 + days_2004 + days_2005 + days_2006 = 1461 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_days_2003_to_2006_l819_81930


namespace NUMINAMATH_GPT_factoring_correct_l819_81924

-- Definitions corresponding to the problem conditions
def optionA (a : ℝ) : Prop := a^2 - 5*a - 6 = (a - 6) * (a + 1)
def optionB (a x b c : ℝ) : Prop := a*x + b*x + c = (a + b)*x + c
def optionC (a b : ℝ) : Prop := (a + b)^2 = a^2 + 2*a*b + b^2
def optionD (a b : ℝ) : Prop := (a + b)*(a - b) = a^2 - b^2

-- The main theorem that proves option A is the correct answer
theorem factoring_correct : optionA a := by
  sorry

end NUMINAMATH_GPT_factoring_correct_l819_81924


namespace NUMINAMATH_GPT_total_items_correct_l819_81986

-- Defining the number of each type of items ordered by Betty
def slippers := 6
def lipstick := 4
def hair_color := 8

-- The total number of items ordered by Betty
def total_items := slippers + lipstick + hair_color

-- The statement asserting that the total number of items is 18
theorem total_items_correct : total_items = 18 := 
by 
  -- sorry allows us to skip the proof
  sorry

end NUMINAMATH_GPT_total_items_correct_l819_81986


namespace NUMINAMATH_GPT_find_max_side_length_l819_81904

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end NUMINAMATH_GPT_find_max_side_length_l819_81904


namespace NUMINAMATH_GPT_circle_tangent_x_axis_l819_81994

theorem circle_tangent_x_axis (x y : ℝ) (h_center : (x, y) = (-3, 4)) (h_tangent : y = 4) :
  ∃ r : ℝ, r = 4 ∧ (∀ x y, (x + 3)^2 + (y - 4)^2 = 16) :=
sorry

end NUMINAMATH_GPT_circle_tangent_x_axis_l819_81994


namespace NUMINAMATH_GPT_train_passing_time_l819_81972

/-- The problem defines a train of length 110 meters traveling at 40 km/hr, 
    passing a man who is running at 5 km/hr in the opposite direction.
    We want to prove that the time it takes for the train to pass the man is 8.8 seconds. -/
theorem train_passing_time :
  ∀ (train_length : ℕ) (train_speed man_speed : ℕ), 
  train_length = 110 → train_speed = 40 → man_speed = 5 →
  (∃ time : ℚ, time = 8.8) :=
by
  intros train_length train_speed man_speed h_train_length h_train_speed h_man_speed
  sorry

end NUMINAMATH_GPT_train_passing_time_l819_81972


namespace NUMINAMATH_GPT_number_wall_problem_l819_81967

theorem number_wall_problem (m : ℤ) : 
  ((m + 5) + 16 + 18 = 56) → (m = 17) :=
by
  sorry

end NUMINAMATH_GPT_number_wall_problem_l819_81967


namespace NUMINAMATH_GPT_probability_two_students_next_to_each_other_l819_81985

theorem probability_two_students_next_to_each_other : (2 * Nat.factorial 9) / Nat.factorial 10 = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_students_next_to_each_other_l819_81985


namespace NUMINAMATH_GPT_emily_total_spent_l819_81989

-- Define the given conditions.
def cost_per_flower : ℕ := 3
def num_roses : ℕ := 2
def num_daisies : ℕ := 2

-- Calculate the total number of flowers and the total cost.
def total_flowers : ℕ := num_roses + num_daisies
def total_cost : ℕ := total_flowers * cost_per_flower

-- Statement: Prove that Emily spent 12 dollars.
theorem emily_total_spent : total_cost = 12 := by
  sorry

end NUMINAMATH_GPT_emily_total_spent_l819_81989


namespace NUMINAMATH_GPT_winning_percentage_l819_81970

noncomputable def total_votes (votes_winner votes_margin : ℕ) : ℕ :=
  votes_winner + (votes_winner - votes_margin)

noncomputable def percentage_votes (votes_winner total_votes : ℕ) : ℝ :=
  (votes_winner : ℝ) / (total_votes : ℝ) * 100

theorem winning_percentage
  (votes_winner : ℕ)
  (votes_margin : ℕ)
  (h_winner : votes_winner = 775)
  (h_margin : votes_margin = 300) :
  percentage_votes votes_winner (total_votes votes_winner votes_margin) = 62 :=
sorry

end NUMINAMATH_GPT_winning_percentage_l819_81970


namespace NUMINAMATH_GPT_part_a_l819_81949

theorem part_a (a b : ℕ) (h : (3 * a + b) % 10 = (3 * b + a) % 10) : 
  (a % 10 = b % 10) := 
sorry

end NUMINAMATH_GPT_part_a_l819_81949


namespace NUMINAMATH_GPT_expression_evaluation_l819_81926

theorem expression_evaluation : 5^3 - 3 * 5^2 + 3 * 5 - 1 = 64 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l819_81926


namespace NUMINAMATH_GPT_bill_earnings_per_ounce_l819_81959

-- Given conditions
def ounces_sold : Nat := 8
def fine : Nat := 50
def money_left : Nat := 22
def total_money_earned : Nat := money_left + fine -- $72

-- The amount earned for every ounce of fool's gold
def price_per_ounce : Nat := total_money_earned / ounces_sold -- 72 / 8

-- The proof statement
theorem bill_earnings_per_ounce (h: price_per_ounce = 9) : True :=
by
  trivial

end NUMINAMATH_GPT_bill_earnings_per_ounce_l819_81959


namespace NUMINAMATH_GPT_class_8_3_final_score_is_correct_l819_81987

def class_8_3_singing_quality : ℝ := 92
def class_8_3_spirit : ℝ := 80
def class_8_3_coordination : ℝ := 70

def final_score (singing_quality spirit coordination : ℝ) : ℝ :=
  0.4 * singing_quality + 0.3 * spirit + 0.3 * coordination

theorem class_8_3_final_score_is_correct :
  final_score class_8_3_singing_quality class_8_3_spirit class_8_3_coordination = 81.8 :=
by
  sorry

end NUMINAMATH_GPT_class_8_3_final_score_is_correct_l819_81987


namespace NUMINAMATH_GPT_calculation_result_l819_81919

theorem calculation_result : (1000 * 7 / 10 * 17 * 5^2 = 297500) :=
by sorry

end NUMINAMATH_GPT_calculation_result_l819_81919


namespace NUMINAMATH_GPT_total_pastries_sum_l819_81918

   theorem total_pastries_sum :
     let lola_mini_cupcakes := 13
     let lola_pop_tarts := 10
     let lola_blueberry_pies := 8
     let lola_chocolate_eclairs := 6

     let lulu_mini_cupcakes := 16
     let lulu_pop_tarts := 12
     let lulu_blueberry_pies := 14
     let lulu_chocolate_eclairs := 9

     let lila_mini_cupcakes := 22
     let lila_pop_tarts := 15
     let lila_blueberry_pies := 10
     let lila_chocolate_eclairs := 12

     lola_mini_cupcakes + lulu_mini_cupcakes + lila_mini_cupcakes +
     lola_pop_tarts + lulu_pop_tarts + lila_pop_tarts +
     lola_blueberry_pies + lulu_blueberry_pies + lila_blueberry_pies +
     lola_chocolate_eclairs + lulu_chocolate_eclairs + lila_chocolate_eclairs = 147 :=
   by
     sorry
   
end NUMINAMATH_GPT_total_pastries_sum_l819_81918


namespace NUMINAMATH_GPT_find_abcd_l819_81945

theorem find_abcd 
    (a b c d : ℕ) 
    (h : 5^a + 6^b + 7^c + 11^d = 1999) : 
    (a, b, c, d) = (4, 2, 1, 3) :=
by
    sorry

end NUMINAMATH_GPT_find_abcd_l819_81945


namespace NUMINAMATH_GPT_moles_CO2_required_l819_81925

theorem moles_CO2_required
  (moles_MgO : ℕ) 
  (moles_MgCO3 : ℕ) 
  (balanced_equation : ∀ (MgO CO2 MgCO3 : ℕ), MgO + CO2 = MgCO3) 
  (reaction_produces : moles_MgO = 3 ∧ moles_MgCO3 = 3) :
  3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_moles_CO2_required_l819_81925


namespace NUMINAMATH_GPT_find_abc_l819_81921

theorem find_abc (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≤ b ∧ b ≤ c) (h5 : a + b + c + a * b + b * c + c * a = a * b * c + 1) :
  (a = 2 ∧ b = 5 ∧ c = 8) ∨ (a = 3 ∧ b = 4 ∧ c = 13) :=
sorry

end NUMINAMATH_GPT_find_abc_l819_81921


namespace NUMINAMATH_GPT_combined_sale_price_correct_l819_81981

-- Define constants for purchase costs of items A, B, and C.
def purchase_cost_A : ℝ := 650
def purchase_cost_B : ℝ := 350
def purchase_cost_C : ℝ := 400

-- Define profit percentages for items A, B, and C.
def profit_percentage_A : ℝ := 0.40
def profit_percentage_B : ℝ := 0.25
def profit_percentage_C : ℝ := 0.30

-- Define the desired sale prices for items A, B, and C based on profit margins.
def sale_price_A : ℝ := purchase_cost_A * (1 + profit_percentage_A)
def sale_price_B : ℝ := purchase_cost_B * (1 + profit_percentage_B)
def sale_price_C : ℝ := purchase_cost_C * (1 + profit_percentage_C)

-- Calculate the combined sale price for all three items.
def combined_sale_price : ℝ := sale_price_A + sale_price_B + sale_price_C

-- The theorem stating that the combined sale price for all three items is $1867.50.
theorem combined_sale_price_correct :
  combined_sale_price = 1867.50 := 
sorry

end NUMINAMATH_GPT_combined_sale_price_correct_l819_81981


namespace NUMINAMATH_GPT_melissa_total_repair_time_l819_81953

def time_flat_shoes := 3 + 8 + 9
def time_sandals :=  4 + 5
def time_high_heels := 6 + 12 + 10

def first_session_flat_shoes := 6 * time_flat_shoes
def first_session_sandals := 4 * time_sandals
def first_session_high_heels := 3 * time_high_heels

def second_session_flat_shoes := 4 * time_flat_shoes
def second_session_sandals := 7 * time_sandals
def second_session_high_heels := 5 * time_high_heels

def total_first_session := first_session_flat_shoes + first_session_sandals + first_session_high_heels
def total_second_session := second_session_flat_shoes + second_session_sandals + second_session_high_heels

def break_time := 15

def total_repair_time := total_first_session + total_second_session
def total_time_including_break := total_repair_time + break_time

theorem melissa_total_repair_time : total_time_including_break = 538 := by
  sorry

end NUMINAMATH_GPT_melissa_total_repair_time_l819_81953


namespace NUMINAMATH_GPT_quotient_korean_english_l819_81916

theorem quotient_korean_english (K M E : ℝ) (h1 : K / M = 1.2) (h2 : M / E = 5 / 6) : K / E = 1 :=
sorry

end NUMINAMATH_GPT_quotient_korean_english_l819_81916


namespace NUMINAMATH_GPT_mutually_exclusive_shots_proof_l819_81952

/-- Definition of a mutually exclusive event to the event "at most one shot is successful". -/
def mutual_exclusive_at_most_one_shot_successful (both_shots_successful at_most_one_shot_successful : Prop) : Prop :=
  (at_most_one_shot_successful ↔ ¬both_shots_successful)

variable (both_shots_successful : Prop)
variable (at_most_one_shot_successful : Prop)

/-- Given two basketball shots, prove that "both shots are successful" is a mutually exclusive event to "at most one shot is successful". -/
theorem mutually_exclusive_shots_proof : mutual_exclusive_at_most_one_shot_successful both_shots_successful at_most_one_shot_successful :=
  sorry

end NUMINAMATH_GPT_mutually_exclusive_shots_proof_l819_81952


namespace NUMINAMATH_GPT_min_sum_of_m_n_l819_81948

theorem min_sum_of_m_n (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 3) (h3 : 8 ∣ (180 * m * n - 360 * m)) : m + n = 5 :=
sorry

end NUMINAMATH_GPT_min_sum_of_m_n_l819_81948


namespace NUMINAMATH_GPT_sum_of_interior_angles_l819_81995

theorem sum_of_interior_angles (h_triangle : ∀ (a b c : ℝ), a + b + c = 180)
    (h_quadrilateral : ∀ (a b c d : ℝ), a + b + c + d = 360) :
  (∀ (n : ℕ), n ≥ 3 → ∀ (angles : Fin n → ℝ), (Finset.univ.sum angles) = (n-2) * 180) :=
by
  intro n h_n angles
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_l819_81995


namespace NUMINAMATH_GPT_inequality_solution_set_l819_81937

theorem inequality_solution_set (a b : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, ax^2 + bx - 1 < 0 ↔ -1/2 < x ∧ x < 1) :
  ∀ x : ℝ, (2 * x + 2) / (-x + 1) < 0 ↔ (x < -1 ∨ x > 1) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l819_81937


namespace NUMINAMATH_GPT_system_of_equations_solution_cases_l819_81943

theorem system_of_equations_solution_cases
  (x y a b : ℝ) :
  (a = b → x + y = 2 * a) ∧
  (a = -b → ¬ (∃ (x y : ℝ), (x / (x - a)) + (y / (y - b)) = 2 ∧ a * x + b * y = 2 * a * b)) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_cases_l819_81943


namespace NUMINAMATH_GPT_speed_of_stream_l819_81902

variables (V_d V_u V_m V_s : ℝ)
variables (h1 : V_d = V_m + V_s) (h2 : V_u = V_m - V_s) (h3 : V_d = 18) (h4 : V_u = 6) (h5 : V_m = 12)

theorem speed_of_stream : V_s = 6 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l819_81902


namespace NUMINAMATH_GPT_paving_cost_is_16500_l819_81968

-- Define the given conditions
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_meter : ℝ := 800

-- Define the area calculation
def area (L W : ℝ) : ℝ := L * W

-- Define the cost calculation
def cost (A rate : ℝ) : ℝ := A * rate

-- The theorem to prove that the cost of paving the floor is 16500
theorem paving_cost_is_16500 : cost (area length width) rate_per_sq_meter = 16500 :=
by
  -- Proof is omitted here
  sorry

end NUMINAMATH_GPT_paving_cost_is_16500_l819_81968
