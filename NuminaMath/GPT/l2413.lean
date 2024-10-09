import Mathlib

namespace ab_root_of_Q_l2413_241357

theorem ab_root_of_Q (a b : ℝ) (h : a ≠ b) (ha : a^4 + a^3 - 1 = 0) (hb : b^4 + b^3 - 1 = 0) :
  (ab : ℝ)^6 + (ab : ℝ)^4 + (ab : ℝ)^3 - (ab : ℝ)^2 - 1 = 0 := 
sorry

end ab_root_of_Q_l2413_241357


namespace eq_frac_l2413_241391

noncomputable def g : ℝ → ℝ := sorry

theorem eq_frac (h1 : ∀ c d : ℝ, c^3 * g d = d^3 * g c)
                (h2 : g 3 ≠ 0) : (g 7 - g 4) / g 3 = 279 / 27 :=
by
  sorry

end eq_frac_l2413_241391


namespace probability_two_green_apples_l2413_241317

theorem probability_two_green_apples :
  let total_apples := 9
  let total_red := 5
  let total_green := 4
  let ways_to_choose_two := Nat.choose total_apples 2
  let ways_to_choose_two_green := Nat.choose total_green 2
  ways_to_choose_two ≠ 0 →
  (ways_to_choose_two_green / ways_to_choose_two : ℚ) = 1 / 6 :=
by
  intros
  -- skipping the proof
  sorry

end probability_two_green_apples_l2413_241317


namespace trajectory_of_Q_l2413_241364

variable (x y m n : ℝ)

def line_l (x y : ℝ) : Prop := 2 * x + 4 * y + 3 = 0

def point_P_on_line_l (x y m n : ℝ) : Prop := line_l m n

def origin (O : (ℝ × ℝ)) := O = (0, 0)

def Q_condition (O Q P : (ℝ × ℝ)) : Prop := 2 • O + 2 • Q = Q + P

theorem trajectory_of_Q (x y m n : ℝ) (O : (ℝ × ℝ)) (P Q : (ℝ × ℝ)) :
  point_P_on_line_l x y m n → origin O → Q_condition O Q P → 
  2 * x + 4 * y + 1 = 0 := 
sorry

end trajectory_of_Q_l2413_241364


namespace exist_same_number_of_acquaintances_l2413_241333

-- Define a group of 2014 people
variable (People : Type) [Fintype People] [DecidableEq People]
variable (knows : People → People → Prop)
variable [DecidableRel knows]

-- Conditions
def mutual_acquaintance : Prop := 
  ∀ (a b : People), knows a b ↔ knows b a

def num_people : Prop := 
  Fintype.card People = 2014

-- Theorem to prove
theorem exist_same_number_of_acquaintances 
  (h1 : mutual_acquaintance People knows) 
  (h2 : num_people People) : 
  ∃ (p1 p2 : People), p1 ≠ p2 ∧
    (Fintype.card { x // knows p1 x } = Fintype.card { x // knows p2 x }) :=
sorry

end exist_same_number_of_acquaintances_l2413_241333


namespace child_l2413_241363

noncomputable def C (G : ℝ) := 60 - 46
noncomputable def G := 130 - 60
noncomputable def ratio := (C G) / G

theorem child's_weight_to_grandmother's_weight_is_1_5 :
  ratio = 1 / 5 :=
by
  sorry

end child_l2413_241363


namespace retirement_hiring_year_l2413_241358

theorem retirement_hiring_year (A W Y : ℕ)
  (hired_on_32nd_birthday : A = 32)
  (eligible_to_retire_in_2007 : 32 + (2007 - Y) = 70) : 
  Y = 1969 := by
  sorry

end retirement_hiring_year_l2413_241358


namespace quadratic_equation_roots_l2413_241376

theorem quadratic_equation_roots (a b c : ℝ) (h_a_nonzero : a ≠ 0) 
  (h_roots : ∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -1) : 
  a + b + c = 0 ∧ b = 0 :=
by
  -- Using Vieta's formulas and the properties given, we should show:
  -- h_roots means the sum of roots = -(b/a) = 0 → b = 0
  -- and the product of roots = (c/a) = -1/a → c = -a
  -- Substituting these into ax^2 + bx + c = 0 should give us:
  -- a + b + c = 0 → we need to show both parts to complete the proof.
  sorry

end quadratic_equation_roots_l2413_241376


namespace quadratic_root_iff_l2413_241346

theorem quadratic_root_iff (a b c : ℝ) :
  (∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0) ↔ (a + b + c = 0) :=
by
  sorry

end quadratic_root_iff_l2413_241346


namespace complex_number_solution_l2413_241350

-- Define that z is a complex number and the condition given in the problem.
theorem complex_number_solution (z : ℂ) (hz : (i / (z + i)) = 2 - i) : z = -1/5 - 3/5 * i :=
sorry

end complex_number_solution_l2413_241350


namespace tan_105_degree_is_neg_sqrt3_minus_2_l2413_241372

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l2413_241372


namespace train_crossing_time_l2413_241344

/-- Given the conditions that a moving train requires 10 seconds to pass a pole,
    its speed is 36 km/h, and the length of a stationary train is 300 meters,
    prove that the moving train takes 40 seconds to cross the stationary train. -/
theorem train_crossing_time (t_pole : ℕ)
  (v_kmh : ℕ)
  (length_stationary : ℕ) :
  t_pole = 10 →
  v_kmh = 36 →
  length_stationary = 300 →
  ∃ t_cross : ℕ, t_cross = 40 :=
by
  intros h1 h2 h3
  sorry

end train_crossing_time_l2413_241344


namespace proof_problem_l2413_241309

-- Definitions of sequence terms and their properties
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ a 4 = 16 ∧ ∀ n, a n = 2^n

-- Definition for the sum of the first n terms of the sequence
noncomputable def sum_of_sequence (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = 2^(n + 1) - 2

-- Definition for the transformed sequence b_n = log_2 a_n
def transformed_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ n, b n = Nat.log2 (a n)

-- Definition for the sum T_n related to b_n
noncomputable def sum_of_transformed_sequence (T : ℕ → ℚ) (b : ℕ → ℕ) : Prop :=
  ∀ n, T n = 1 - 1 / (n + 1)

theorem proof_problem :
  (∃ a : ℕ → ℕ, geometric_sequence a) ∧
  (∃ S : ℕ → ℕ, sum_of_sequence S) ∧
  (∃ (a b : ℕ → ℕ), geometric_sequence a ∧ transformed_sequence a b ∧
   (∃ T : ℕ → ℚ, sum_of_transformed_sequence T b)) :=
by {
  -- Definitions and proofs will go here
  sorry
}

end proof_problem_l2413_241309


namespace sum_of_first_column_l2413_241335

theorem sum_of_first_column (a b : ℕ) 
  (h1 : 16 * (a + b) = 96) 
  (h2 : 16 * (a - b) = 64) :
  a + b = 20 :=
by sorry

end sum_of_first_column_l2413_241335


namespace log_relation_l2413_241355

theorem log_relation (a b : ℝ) 
  (h₁ : a = Real.log 1024 / Real.log 16) 
  (h₂ : b = Real.log 32 / Real.log 2) : 
  a = 1 / 2 * b := 
by 
  sorry

end log_relation_l2413_241355


namespace discount_price_l2413_241353

theorem discount_price (original_price : ℝ) (discount_percent : ℝ) (final_price : ℝ) :
  original_price = 800 ∧ discount_percent = 15 → final_price = 680 :=
by
  intros h
  cases' h with hp hd
  sorry

end discount_price_l2413_241353


namespace find_m_for_one_real_solution_l2413_241354

theorem find_m_for_one_real_solution (m : ℝ) (h : 4 * m * 4 = m^2) : m = 8 := sorry

end find_m_for_one_real_solution_l2413_241354


namespace problem_equivalent_l2413_241359

variable (p : ℤ) 

theorem problem_equivalent (h : p = (-2023) * 100) : (-2023) * 99 = p + 2023 :=
by sorry

end problem_equivalent_l2413_241359


namespace boat_distance_along_stream_in_one_hour_l2413_241347

theorem boat_distance_along_stream_in_one_hour :
  ∀ (v_b v_s d_up t : ℝ),
  v_b = 7 →
  d_up = 3 →
  t = 1 →
  (t * (v_b - v_s) = d_up) →
  t * (v_b + v_s) = 11 :=
by
  intros v_b v_s d_up t Hv_b Hd_up Ht Hup
  sorry

end boat_distance_along_stream_in_one_hour_l2413_241347


namespace rectangle_area_is_140_l2413_241378

noncomputable def area_of_square (a : ℝ) : ℝ := a * a
noncomputable def length_of_rectangle (r : ℝ) : ℝ := (2 / 5) * r
noncomputable def area_of_rectangle (l : ℝ) (b : ℝ) : ℝ := l * b

theorem rectangle_area_is_140 :
  ∃ (a r l b : ℝ), area_of_square a = 1225 ∧ r = a ∧ l = length_of_rectangle r ∧ b = 10 ∧ area_of_rectangle l b = 140 :=
by
  use 35, 35, 14, 10
  simp [area_of_square, length_of_rectangle, area_of_rectangle]
  sorry

end rectangle_area_is_140_l2413_241378


namespace find_a2_b2_l2413_241306

noncomputable def imaginary_unit : ℂ := Complex.I

theorem find_a2_b2 (a b : ℝ) (h1 : (a - 2 * imaginary_unit) * imaginary_unit = b - imaginary_unit) : a^2 + b^2 = 5 :=
  sorry

end find_a2_b2_l2413_241306


namespace triangle_abs_diff_l2413_241315

theorem triangle_abs_diff (a b c : ℝ) 
  (h1 : a + b > c)
  (h2 : a + c > b) :
  |a + b - c| - |a - b - c| = 2 * a - 2 * c := 
by sorry

end triangle_abs_diff_l2413_241315


namespace Elon_has_10_more_Teslas_than_Sam_l2413_241328

noncomputable def TeslasCalculation : Nat :=
let Chris : Nat := 6
let Sam : Nat := Chris / 2
let Elon : Nat := 13
Elon - Sam

theorem Elon_has_10_more_Teslas_than_Sam :
  TeslasCalculation = 10 :=
by
  sorry

end Elon_has_10_more_Teslas_than_Sam_l2413_241328


namespace part_I_solution_part_II_solution_l2413_241385

def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

theorem part_I_solution (x : ℝ) :
  (|x + 1| - |x - 1| >= x) ↔ (x <= -2 ∨ (0 <= x ∧ x <= 2)) :=
by
  sorry

theorem part_II_solution (m : ℝ) :
  (∀ (x a : ℝ), (0 < m ∧ m < 1 ∧ (a <= -3 ∨ 3 <= a)) → (f x a m >= 2)) ↔ (m = 1/3) :=
by
  sorry

end part_I_solution_part_II_solution_l2413_241385


namespace identity_function_l2413_241369

theorem identity_function {f : ℕ → ℕ} (h : ∀ a b : ℕ, 0 < a → 0 < b → a - f b ∣ a * f a - b * f b) :
  ∀ a : ℕ, 0 < a → f a = a :=
by
  sorry

end identity_function_l2413_241369


namespace track_length_is_320_l2413_241327

noncomputable def length_of_track (x : ℝ) : Prop :=
  (∃ v_b v_s : ℝ, (v_b > 0 ∧ v_s > 0 ∧ v_b + v_s = x / 2 ∧ -- speeds of Brenda and Sally must sum up to half the track length against each other
                    80 / v_b = (x / 2 - 80) / v_s ∧ -- First meeting condition
                    120 / v_s + 80 / v_b = (x / 2 + 40) / v_s + (x - 80) / v_b -- Second meeting condition
                   )) ∧ x = 320

theorem track_length_is_320 : ∃ x : ℝ, length_of_track x :=
by
  use 320
  unfold length_of_track
  simp
  sorry

end track_length_is_320_l2413_241327


namespace solution_couples_l2413_241356

noncomputable def find_couples (n m k : ℕ) : Prop :=
  ∃ t : ℕ, (n = 2^k - 1 - t ∧ m = (Nat.factorial (2^k)) / 2^(2^k - 1 - t))

theorem solution_couples (k : ℕ) :
  ∃ n m : ℕ, (Nat.factorial (2^k)) = 2^n * m ∧ find_couples n m k :=
sorry

end solution_couples_l2413_241356


namespace total_songs_megan_bought_l2413_241310

-- Definitions for the problem conditions
def country_albums : ℕ := 2
def pop_albums : ℕ := 8
def songs_per_album : ℕ := 7
def total_albums : ℕ := country_albums + pop_albums

-- Theorem stating the conclusion we need to prove
theorem total_songs_megan_bought : total_albums * songs_per_album = 70 :=
by
  sorry

end total_songs_megan_bought_l2413_241310


namespace trader_sold_90_pens_l2413_241319

theorem trader_sold_90_pens (C N : ℝ) (gain_percent : ℝ) (H1 : gain_percent = 33.33333333333333) (H2 : 30 * C = (gain_percent / 100) * N * C) :
  N = 90 :=
by
  sorry

end trader_sold_90_pens_l2413_241319


namespace perfect_squares_with_specific_ones_digit_count_l2413_241331

theorem perfect_squares_with_specific_ones_digit_count : 
  ∃ n : ℕ, (∀ k : ℕ, k < 2500 → (k % 10 = 4 ∨ k % 10 = 5 ∨ k % 10 = 6) ↔ ∃ m : ℕ, m < n ∧ (m % 10 = 2 ∨ m % 10 = 8 ∨ m % 10 = 5 ∨ m % 10 = 4 ∨ m % 10 = 6) ∧ k = m * m) 
  ∧ n = 25 := 
by 
  sorry

end perfect_squares_with_specific_ones_digit_count_l2413_241331


namespace jim_ran_16_miles_in_2_hours_l2413_241390

-- Given conditions
variables (j f : ℝ) -- miles Jim ran in 2 hours, miles Frank ran in 2 hours
variables (h1 : f = 20) -- Frank ran 20 miles in 2 hours
variables (h2 : f / 2 = (j / 2) + 2) -- Frank ran 2 miles more than Jim in an hour

-- Statement to prove
theorem jim_ran_16_miles_in_2_hours (j f : ℝ) (h1 : f = 20) (h2 : f / 2 = (j / 2) + 2) : j = 16 :=
by
  sorry

end jim_ran_16_miles_in_2_hours_l2413_241390


namespace column_heights_achievable_l2413_241392

open Int

noncomputable def number_of_column_heights (n : ℕ) (h₁ h₂ h₃ : ℕ) : ℕ :=
  let min_height := n * h₁
  let max_height := n * h₃
  max_height - min_height + 1

theorem column_heights_achievable :
  number_of_column_heights 80 3 8 15 = 961 := by
  -- Proof goes here.
  sorry

end column_heights_achievable_l2413_241392


namespace find_fraction_l2413_241366

theorem find_fraction (a b : ℝ) (h : a = 2 * b) : (a / (a - b)) = 2 :=
by
  sorry

end find_fraction_l2413_241366


namespace symmetry_center_example_l2413_241336

-- Define the function tan(2x - π/4)
noncomputable def func (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

-- Define what it means to be a symmetry center for the function
def is_symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * (p.1) - x) = 2 * p.2 - f x

-- Statement of the proof problem
theorem symmetry_center_example : is_symmetry_center func (-Real.pi / 8, 0) :=
sorry

end symmetry_center_example_l2413_241336


namespace total_birds_l2413_241329

-- Definitions from conditions
def num_geese : ℕ := 58
def num_ducks : ℕ := 37

-- Proof problem statement
theorem total_birds : num_geese + num_ducks = 95 := by
  sorry

end total_birds_l2413_241329


namespace problem_solution_l2413_241388

noncomputable def quadratic_symmetric_b (a : ℝ) : ℝ :=
  2 * (1 - a)

theorem problem_solution (a : ℝ) (h1 : quadratic_symmetric_b a = 6) :
  b = 6 :=
by
  sorry

end problem_solution_l2413_241388


namespace swimmers_meet_times_l2413_241320

noncomputable def swimmers_passes (pool_length : ℕ) (time_minutes : ℕ) (speed_swimmer1 : ℕ) (speed_swimmer2 : ℕ) : ℕ :=
  let total_time_seconds := time_minutes * 60
  let speed_sum := speed_swimmer1 + speed_swimmer2
  let distance_in_time := total_time_seconds * speed_sum
  distance_in_time / pool_length

theorem swimmers_meet_times :
  swimmers_passes 120 15 4 3 = 53 :=
by
  -- Proof is omitted
  sorry

end swimmers_meet_times_l2413_241320


namespace dad_steps_eq_90_l2413_241373

-- Define the conditions given in the problem
variables (masha_steps yasha_steps dad_steps : ℕ)

-- Conditions:
-- 1. Dad takes 3 steps while Masha takes 5 steps
-- 2. Masha takes 3 steps while Yasha takes 5 steps
-- 3. Together, Masha and Yasha made 400 steps
def conditions := dad_steps * 5 = 3 * masha_steps ∧ masha_steps * yasha_steps = 3 * yasha_steps ∧ 3 * yasha_steps = 400

-- Theorem stating the proof problem
theorem dad_steps_eq_90 : conditions masha_steps yasha_steps dad_steps → dad_steps = 90 :=
by
  sorry

end dad_steps_eq_90_l2413_241373


namespace intersection_point_exists_correct_line_l2413_241300

noncomputable def line1 (x y : ℝ) : Prop := 2 * x - 3 * y + 2 = 0
noncomputable def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 2 = 0
noncomputable def parallel_line (x y : ℝ) : Prop := 4 * x - 2 * y + 7 = 0
noncomputable def target_line (x y : ℝ) : Prop := 2 * x - y - 18 = 0

theorem intersection_point_exists (x y : ℝ) : line1 x y ∧ line2 x y → (x = 14 ∧ y = 10) := 
by sorry

theorem correct_line (x y : ℝ) : 
  (∃ (x y : ℝ), line1 x y ∧ line2 x y) ∧ parallel_line x y 
  → target_line x y :=
by sorry

end intersection_point_exists_correct_line_l2413_241300


namespace unique_n_l2413_241349

theorem unique_n : ∃ n : ℕ, 0 < n ∧ n^3 % 1000 = n ∧ ∀ m : ℕ, 0 < m ∧ m^3 % 1000 = m → m = n :=
by
  sorry

end unique_n_l2413_241349


namespace g_at_100_l2413_241362

-- Defining that g is a function from positive real numbers to real numbers
def g : ℝ → ℝ := sorry

-- The given conditions
axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x * g y - y * g x = g (x / y)

axiom g_one : g 1 = 1

-- The theorem to prove
theorem g_at_100 : g 100 = 50 :=
by
  sorry

end g_at_100_l2413_241362


namespace intersect_is_one_l2413_241395

def SetA : Set ℝ := {x | 0 < x ∧ x < 2}

def SetB : Set ℝ := {0, 1, 2, 3}

theorem intersect_is_one : SetA ∩ SetB = {1} :=
by
  sorry

end intersect_is_one_l2413_241395


namespace John_max_tests_under_B_l2413_241323

theorem John_max_tests_under_B (total_tests first_tests tests_with_B goal_percentage B_tests_first_half : ℕ) :
  total_tests = 60 →
  first_tests = 40 → 
  tests_with_B = 32 → 
  goal_percentage = 75 →
  B_tests_first_half = 32 →
  let needed_B_tests := (goal_percentage * total_tests) / 100
  let remaining_tests := total_tests - first_tests
  let remaining_needed_B_tests := needed_B_tests - B_tests_first_half
  remaining_tests - remaining_needed_B_tests ≤ 7 := sorry

end John_max_tests_under_B_l2413_241323


namespace pieces_per_plant_yield_l2413_241324

theorem pieces_per_plant_yield 
  (rows : ℕ) (plants_per_row : ℕ) (total_harvest : ℕ) 
  (h1 : rows = 30) (h2 : plants_per_row = 10) (h3 : total_harvest = 6000) : 
  (total_harvest / (rows * plants_per_row) = 20) :=
by
  -- Insert math proof here.
  sorry

end pieces_per_plant_yield_l2413_241324


namespace find_angle_C_l2413_241377

open Real -- Opening Real to directly use real number functions and constants

noncomputable def triangle_angles_condition (A B C: ℝ) : Prop :=
  2 * sin A + 5 * cos B = 5 ∧ 5 * sin B + 2 * cos A = 2

-- Theorem statement
theorem find_angle_C (A B C: ℝ) (h: triangle_angles_condition A B C):
  C = arcsin (1 / 5) ∨ C = 180 - arcsin (1 / 5) :=
sorry

end find_angle_C_l2413_241377


namespace find_num_non_officers_l2413_241334

-- Define the average salaries and number of officers
def avg_salary_employees : Int := 120
def avg_salary_officers : Int := 470
def avg_salary_non_officers : Int := 110
def num_officers : Int := 15

-- States the problem of finding the number of non-officers
theorem find_num_non_officers : ∃ N : Int,
(15 * 470 + N * 110 = (15 + N) * 120) ∧ N = 525 := 
by {
  sorry
}

end find_num_non_officers_l2413_241334


namespace one_fifth_greater_than_decimal_by_term_l2413_241394

noncomputable def one_fifth := (1 : ℝ) / 5
noncomputable def decimal_value := 20000001 / 10^8
noncomputable def term := 1 / (5 * 10^8)

theorem one_fifth_greater_than_decimal_by_term :
  one_fifth > decimal_value ∧ one_fifth - decimal_value = term :=
  sorry

end one_fifth_greater_than_decimal_by_term_l2413_241394


namespace concyclic_H_E_N_N1_N2_l2413_241307

open EuclideanGeometry

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def nine_point_center (A B C : Point) : Point := sorry
noncomputable def altitude (A B C : Point) : Point := sorry
noncomputable def salmon_circle_center (A O O₁ O₂ : Point) : Point := sorry
noncomputable def foot_of_perpendicular (O' B C : Point) : Point := sorry
noncomputable def is_concyclic (points : List Point) : Prop := sorry

theorem concyclic_H_E_N_N1_N2 (A B C D : Point):
  let H := altitude A B C
  let O := circumcenter A B C
  let O₁ := circumcenter A B D
  let O₂ := circumcenter A C D
  let N := nine_point_center A B C
  let N₁ := nine_point_center A B D
  let N₂ := nine_point_center A C D
  let O' := salmon_circle_center A O O₁ O₂
  let E := foot_of_perpendicular O' B C
  is_concyclic [H, E, N, N₁, N₂] :=
sorry

end concyclic_H_E_N_N1_N2_l2413_241307


namespace find_coefficients_l2413_241381

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def h (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_coefficients :
  ∃ a b c : ℝ, (∀ s : ℝ, f s = 0 → h a b c (s^3) = 0) ∧
    (a, b, c) = (-6, -9, 20) :=
sorry

end find_coefficients_l2413_241381


namespace sequence_a_n_l2413_241361

theorem sequence_a_n (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → (n^2 + n) * (a (n + 1) - a n) = 2) :
  a 20 = 29 / 10 :=
by
  sorry

end sequence_a_n_l2413_241361


namespace probability_of_two_red_balls_l2413_241304

-- Definitions of quantities
def total_balls := 11
def red_balls := 3
def blue_balls := 4 
def green_balls := 4 
def balls_picked := 2

-- Theorem statement
theorem probability_of_two_red_balls :
  ((red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1) / balls_picked)) = 3 / 55 :=
by
  sorry

end probability_of_two_red_balls_l2413_241304


namespace candy_total_cost_l2413_241383

theorem candy_total_cost
    (grape_candies cherry_candies apple_candies : ℕ)
    (cost_per_candy : ℝ)
    (h1 : grape_candies = 3 * cherry_candies)
    (h2 : apple_candies = 2 * grape_candies)
    (h3 : cost_per_candy = 2.50)
    (h4 : grape_candies = 24) :
    (grape_candies + cherry_candies + apple_candies) * cost_per_candy = 200 := 
by
  sorry

end candy_total_cost_l2413_241383


namespace Charley_total_beads_pulled_l2413_241308

-- Definitions and conditions
def initial_white_beads := 105
def initial_black_beads := 210
def initial_blue_beads := 60

def first_round_black_pulled := (2 / 7) * initial_black_beads
def first_round_white_pulled := (3 / 7) * initial_white_beads
def first_round_blue_pulled := (1 / 4) * initial_blue_beads

def first_round_total_pulled := first_round_black_pulled + first_round_white_pulled + first_round_blue_pulled

def remaining_black_beads := initial_black_beads - first_round_black_pulled
def remaining_white_beads := initial_white_beads - first_round_white_pulled
def remaining_blue_beads := initial_blue_beads - first_round_blue_pulled

def added_white_beads := 45
def added_black_beads := 80

def total_black_beads := remaining_black_beads + added_black_beads
def total_white_beads := remaining_white_beads + added_white_beads

def second_round_black_pulled := (3 / 8) * total_black_beads
def second_round_white_pulled := (1 / 3) * added_white_beads

def second_round_total_pulled := second_round_black_pulled + second_round_white_pulled

def total_beads_pulled := first_round_total_pulled + second_round_total_pulled 

-- Theorem statement
theorem Charley_total_beads_pulled : total_beads_pulled = 221 := 
by
  -- we can ignore the proof step and leave it to be filled
  sorry

end Charley_total_beads_pulled_l2413_241308


namespace fisherman_gets_8_red_snappers_l2413_241305

noncomputable def num_red_snappers (R : ℕ) : Prop :=
  let cost_red_snapper := 3
  let cost_tuna := 2
  let num_tunas := 14
  let total_earnings := 52
  (R * cost_red_snapper) + (num_tunas * cost_tuna) = total_earnings

theorem fisherman_gets_8_red_snappers : num_red_snappers 8 :=
by
  sorry

end fisherman_gets_8_red_snappers_l2413_241305


namespace number_of_classes_l2413_241338

theorem number_of_classes (n : ℕ) (a₁ : ℕ) (d : ℤ) (S : ℕ) (h₁ : d = -2) (h₂ : a₁ = 25) (h₃ : S = 105) : n = 5 :=
by
  /- We state the theorem and the necessary conditions without proving it -/
  sorry

end number_of_classes_l2413_241338


namespace max_value_x_minus_2y_exists_max_value_x_minus_2y_l2413_241330

theorem max_value_x_minus_2y 
  (x y : ℝ) 
  (h : x^2 - 4 * x + y^2 = 0) : 
  x - 2 * y ≤ 2 + 2 * Real.sqrt 5 :=
sorry

theorem exists_max_value_x_minus_2y 
  (x y : ℝ) 
  (h : x^2 - 4 * x + y^2 = 0) : 
  ∃ (x y : ℝ), x - 2 * y = 2 + 2 * Real.sqrt 5 :=
sorry

end max_value_x_minus_2y_exists_max_value_x_minus_2y_l2413_241330


namespace trig_identity_proof_l2413_241302

theorem trig_identity_proof 
  (α : ℝ) 
  (h1 : Real.sin (4 * α) = 2 * Real.sin (2 * α) * Real.cos (2 * α))
  (h2 : Real.cos (4 * α) = Real.cos (2 * α) ^ 2 - Real.sin (2 * α) ^ 2) : 
  (1 - 2 * Real.sin (2 * α) ^ 2) / (1 - Real.sin (4 * α)) = 
  (1 + Real.tan (2 * α)) / (1 - Real.tan (2 * α)) := 
by 
  sorry

end trig_identity_proof_l2413_241302


namespace rental_cost_equal_mileage_l2413_241398

theorem rental_cost_equal_mileage :
  ∃ m : ℝ, 
    (21.95 + 0.19 * m = 18.95 + 0.21 * m) ∧ 
    m = 150 :=
by
  sorry

end rental_cost_equal_mileage_l2413_241398


namespace graph_properties_l2413_241321

noncomputable def f (x : ℝ) : ℝ := (x^2 - 5*x + 6) / (x - 1)

theorem graph_properties :
  (∀ x, x ≠ 1 → f x = (x-2)*(x-3)/(x-1)) ∧
  (∃ x, f x = 0 ∧ (x = 2 ∨ x = 3)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 1) < δ → abs (f x) > ε) ∧
  ((∀ ε > 0, ∃ M > 0, ∀ x > M, f x > ε) ∧ (∀ ε > 0, ∃ M < 0, ∀ x < M, f x < -ε)) := sorry

end graph_properties_l2413_241321


namespace first_fun_friday_is_march_30_l2413_241343

def month_days := 31
def start_day := 4 -- 1 for Sunday, 2 for Monday, ..., 7 for Saturday; 4 means Thursday
def first_friday := 2
def fun_friday (n : ℕ) : ℕ := first_friday + (n - 1) * 7

theorem first_fun_friday_is_march_30 (h1 : start_day = 4)
                                    (h2 : month_days = 31) :
                                    fun_friday 5 = 30 :=
by 
  -- Proof is omitted
  sorry

end first_fun_friday_is_march_30_l2413_241343


namespace basketball_court_width_l2413_241389

variable (width length : ℕ)

-- Given conditions
axiom h1 : length = width + 14
axiom h2 : 2 * length + 2 * width = 96

-- Prove the width is 17 meters
theorem basketball_court_width : width = 17 :=
by {
  sorry
}

end basketball_court_width_l2413_241389


namespace inequality_region_area_l2413_241318

noncomputable def area_of_inequality_region : ℝ :=
  let region := {p : ℝ × ℝ | |p.fst - p.snd| + |2 * p.fst + 2 * p.snd| ≤ 8}
  let vertices := [(2, 2), (-2, 2), (-2, -2), (2, -2)]
  let d1 := 8
  let d2 := 8
  (1 / 2) * d1 * d2

theorem inequality_region_area :
  area_of_inequality_region = 32 :=
by
  sorry  -- Proof to be provided

end inequality_region_area_l2413_241318


namespace geometric_sequence_common_ratio_l2413_241311

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h₁ : a 1 = 1) (h₂ : a 1 * a 2 * a 3 = -8) :
  q = -2 :=
sorry

end geometric_sequence_common_ratio_l2413_241311


namespace tangent_line_at_point_l2413_241314

theorem tangent_line_at_point (x y : ℝ) (h : y = Real.exp x) (t : x = 2) :
  y = Real.exp 2 * x - 2 * Real.exp 2 :=
by sorry

end tangent_line_at_point_l2413_241314


namespace factorization_correct_l2413_241375

theorem factorization_correct (x : ℝ) :
  (x - 3) * (x - 1) * (x - 2) * (x + 4) + 24 = (x - 2) * (x + 3) * (x^2 + x - 8) := 
sorry

end factorization_correct_l2413_241375


namespace determine_e_l2413_241332

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3*x^3 + d*x^2 + e*x + f

theorem determine_e (d f : ℝ) (h1 : f = 18) (h2 : -f/3 = -6) (h3 : -d/3 = -6) (h4 : 3 + d + e + f = -6) : e = -45 :=
sorry

end determine_e_l2413_241332


namespace teams_in_double_round_robin_l2413_241339
-- Import the standard math library

-- Lean statement for the proof problem
theorem teams_in_double_round_robin (m n : ℤ) 
  (h : 9 * n^2 + 6 * n + 32 = m * (m - 1) / 2) : 
  m = 8 ∨ m = 32 :=
sorry

end teams_in_double_round_robin_l2413_241339


namespace value_of_product_l2413_241374

theorem value_of_product (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) : (x + 2) * (y + 2) = 16 := by
  sorry

end value_of_product_l2413_241374


namespace greatest_product_two_ints_sum_300_l2413_241322

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l2413_241322


namespace quadratic_no_real_roots_l2413_241387

theorem quadratic_no_real_roots (a b : ℝ) (h : ∃ x : ℝ, x^2 + b * x + a = 0) : false :=
sorry

end quadratic_no_real_roots_l2413_241387


namespace sum_first_sequence_terms_l2413_241341

theorem sum_first_sequence_terms 
  (S : ℕ → ℕ) 
  (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → S n - S (n - 1) = 2 * n - 1)
  (h2 : S 2 = 3) 
  : a 1 + a 3 = 5 :=
sorry

end sum_first_sequence_terms_l2413_241341


namespace relationship_between_x1_x2_x3_l2413_241365

variable {x1 x2 x3 : ℝ}

theorem relationship_between_x1_x2_x3
  (A_on_curve : (6 : ℝ) = 6 / x1)
  (B_on_curve : (12 : ℝ) = 6 / x2)
  (C_on_curve : (-6 : ℝ) = 6 / x3) :
  x3 < x2 ∧ x2 < x1 := 
sorry

end relationship_between_x1_x2_x3_l2413_241365


namespace problem1_l2413_241370

theorem problem1 (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) :
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = -5 / 6 := 
  sorry

end problem1_l2413_241370


namespace sum_of_fourth_powers_l2413_241326

theorem sum_of_fourth_powers
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 15.5 := sorry

end sum_of_fourth_powers_l2413_241326


namespace range_of_a_l2413_241313

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x - 3 * a else a^x - 2

theorem range_of_a (a : ℝ) :
  (0 < a ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → f a x > f a y) ↔ (0 < a ∧ a ≤ 1 / 3) :=
by sorry

end range_of_a_l2413_241313


namespace frac_pow_zero_l2413_241367

def frac := 123456789 / (-987654321 : ℤ)

theorem frac_pow_zero : frac ^ 0 = 1 :=
by sorry

end frac_pow_zero_l2413_241367


namespace largest_n_proof_l2413_241325

def largest_n_less_than_50000_divisible_by_7 (n : ℕ) : Prop :=
  n < 50000 ∧ (10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36) % 7 = 0

theorem largest_n_proof : ∃ n, largest_n_less_than_50000_divisible_by_7 n ∧ ∀ m, largest_n_less_than_50000_divisible_by_7 m → m ≤ n := 
sorry

end largest_n_proof_l2413_241325


namespace roses_cut_from_garden_l2413_241397

-- Define the variables and conditions
variables {x : ℕ} -- x is the number of freshly cut roses

def initial_roses : ℕ := 17
def roses_thrown_away : ℕ := 8
def roses_final_vase : ℕ := 42
def roses_given_away : ℕ := 6

-- The condition that describes the total roses now
def condition (x : ℕ) : Prop :=
  initial_roses - roses_thrown_away + (1/3 : ℚ) * x = roses_final_vase

-- The verification step that checks the total roses concerning given away roses
def verification (x : ℕ) : Prop :=
  (1/3 : ℚ) * x + roses_given_away = roses_final_vase + roses_given_away

-- The main theorem to prove the number of roses cut
theorem roses_cut_from_garden (x : ℕ) (h1 : condition x) (h2 : verification x) : x = 99 :=
  sorry

end roses_cut_from_garden_l2413_241397


namespace simplify_expression_l2413_241379

theorem simplify_expression : (1 / (1 / ((1 / 3) ^ 1) + 1 / ((1 / 3) ^ 2) + 1 / ((1 / 3) ^ 3))) = (1 / 39) :=
by
  sorry

end simplify_expression_l2413_241379


namespace number_of_boys_l2413_241348

theorem number_of_boys (girls boys : ℕ) (total_books books_girls books_boys books_per_student : ℕ)
  (h1 : girls = 15)
  (h2 : total_books = 375)
  (h3 : books_girls = 225)
  (h4 : total_books = books_girls + books_boys)
  (h5 : books_girls = girls * books_per_student)
  (h6 : books_boys = boys * books_per_student)
  (h7 : books_per_student = 15) :
  boys = 10 :=
by
  sorry

end number_of_boys_l2413_241348


namespace range_of_a_l2413_241399

variable {α : Type} [LinearOrderedField α]

def A (a : α) : Set α := {x | |x - a| ≤ 1}

def B : Set α := {x | x^2 - 5*x + 4 ≥ 0}

theorem range_of_a (a : α) (h : A a ∩ B = ∅) : 2 < a ∧ a < 3 := sorry

end range_of_a_l2413_241399


namespace range_of_a_l2413_241312

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → |2 - x| + |x + 1| ≤ a) ↔ 9 ≤ a := 
by sorry

end range_of_a_l2413_241312


namespace opposite_sides_line_range_a_l2413_241386

theorem opposite_sides_line_range_a (a : ℝ) :
  (3 * 2 - 2 * 1 + a) * (3 * -1 - 2 * 3 + a) < 0 → -4 < a ∧ a < 9 := by
  sorry

end opposite_sides_line_range_a_l2413_241386


namespace final_stack_height_l2413_241342

theorem final_stack_height (x : ℕ) 
  (first_stack_height : ℕ := 7) 
  (second_stack_height : ℕ := first_stack_height + 5) 
  (final_stack_height : ℕ := second_stack_height + x) 
  (blocks_fell_first : ℕ := first_stack_height) 
  (blocks_fell_second : ℕ := second_stack_height - 2) 
  (blocks_fell_final : ℕ := final_stack_height - 3) 
  (total_blocks_fell : 33 = blocks_fell_first + blocks_fell_second + blocks_fell_final) 
  : x = 7 :=
  sorry

end final_stack_height_l2413_241342


namespace lift_time_15_minutes_l2413_241380

theorem lift_time_15_minutes (t : ℕ) (h₁ : 5 = 5) (h₂ : 6 * (t + 5) = 120) : t = 15 :=
by {
  sorry
}

end lift_time_15_minutes_l2413_241380


namespace largest_prime_factor_of_85_l2413_241384

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p

theorem largest_prime_factor_of_85 :
  let a := 65
  let b := 85
  let c := 91
  let d := 143
  let e := 169
  largest_prime_factor b 17 :=
by
  sorry

end largest_prime_factor_of_85_l2413_241384


namespace number_of_sides_l2413_241303

theorem number_of_sides (P s : ℝ) (hP : P = 108) (hs : s = 12) : P / s = 9 :=
by sorry

end number_of_sides_l2413_241303


namespace pq_sum_l2413_241340

open Real

theorem pq_sum (p q : ℝ) (hp : p^3 - 18 * p^2 + 81 * p - 162 = 0) (hq : 4 * q^3 - 24 * q^2 + 45 * q - 27 = 0) :
    p + q = 8 ∨ p + q = 8 + 6 * sqrt 3 ∨ p + q = 8 - 6 * sqrt 3 :=
sorry

end pq_sum_l2413_241340


namespace kiril_age_problem_l2413_241351

theorem kiril_age_problem (x : ℕ) (h1 : x % 5 = 0) (h2 : (x - 1) % 7 = 0) : 26 - x = 11 :=
by
  sorry

end kiril_age_problem_l2413_241351


namespace passing_grade_fraction_l2413_241301

theorem passing_grade_fraction (A B C D F : ℚ) (hA : A = 1/4) (hB : B = 1/2) (hC : C = 1/8) (hD : D = 1/12) (hF : F = 1/24) : 
  A + B + C = 7/8 :=
by
  sorry

end passing_grade_fraction_l2413_241301


namespace tangent_line_to_parabola_l2413_241352

theorem tangent_line_to_parabola :
  (∀ (x y : ℝ), y = x^2 → x = -1 → y = 1 → 2 * x + y + 1 = 0) :=
by
  intro x y parabola eq_x eq_y
  sorry

end tangent_line_to_parabola_l2413_241352


namespace sector_central_angle_l2413_241371

-- The conditions
def r : ℝ := 2
def S : ℝ := 4

-- The question
theorem sector_central_angle : ∃ α : ℝ, |α| = 2 ∧ S = 0.5 * α * r * r :=
by
  sorry

end sector_central_angle_l2413_241371


namespace initial_weight_l2413_241360

theorem initial_weight (W : ℝ) (h₁ : W > 0): 
  W * 0.85 * 0.75 * 0.90 = 450 := 
by 
  sorry

end initial_weight_l2413_241360


namespace nancy_shoes_l2413_241368

theorem nancy_shoes (boots slippers heels : ℕ) 
  (h₀ : boots = 6)
  (h₁ : slippers = boots + 9)
  (h₂ : heels = 3 * (boots + slippers)) :
  2 * (boots + slippers + heels) = 168 := by
  sorry

end nancy_shoes_l2413_241368


namespace probability_both_selected_is_correct_l2413_241393

def prob_selection_x : ℚ := 1 / 7
def prob_selection_y : ℚ := 2 / 9
def prob_both_selected : ℚ := prob_selection_x * prob_selection_y

theorem probability_both_selected_is_correct : prob_both_selected = 2 / 63 := 
by 
  sorry

end probability_both_selected_is_correct_l2413_241393


namespace triangle_area_ab_l2413_241345

theorem triangle_area_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hline : ∀ (x y : ℝ), a * x + b * y = 6) (harea : (1/2) * (6 / a) * (6 / b) = 6) : 
  a * b = 3 := 
by sorry

end triangle_area_ab_l2413_241345


namespace circle_parabola_intersection_l2413_241337

theorem circle_parabola_intersection (b : ℝ) : 
  b = 25 / 12 → 
  ∃ (r : ℝ) (cx : ℝ), 
  (∃ p1 p2 : ℝ × ℝ, 
    (p1.2 = 3/4 * p1.1 + b ∧ p2.2 = 3/4 * p2.1 + b) ∧ 
    (p1.2 = 3/4 * p1.1^2 ∧ p2.2 = 3/4 * p2.1^2) ∧ 
    (p1 ≠ (0, 0) ∧ p2 ≠ (0, 0))) ∧ 
  (cx^2 + b^2 = r^2) := 
by 
  sorry

end circle_parabola_intersection_l2413_241337


namespace car_distribution_l2413_241396

theorem car_distribution :
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let total_distributed_first_three := first_supplier + second_supplier + third_supplier
  let remaining_cars := total_cars - total_distributed_first_three
  let fourth_supplier := remaining_cars / 2
  let fifth_supplier := remaining_cars / 2
  fourth_supplier = 325000 :=
by
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let total_distributed_first_three := first_supplier + second_supplier + third_supplier
  let remaining_cars := total_cars - total_distributed_first_three
  let fourth_supplier := remaining_cars / 2
  let fifth_supplier := remaining_cars / 2
  sorry

end car_distribution_l2413_241396


namespace pencils_in_boxes_l2413_241382

theorem pencils_in_boxes (total_pencils : ℕ) (pencils_per_box : ℕ) (boxes_required : ℕ) 
    (h1 : total_pencils = 648) (h2 : pencils_per_box = 4) : boxes_required = 162 :=
sorry

end pencils_in_boxes_l2413_241382


namespace cost_price_of_A_l2413_241316

-- Assume the cost price of the bicycle for A which we need to prove
def CP_A : ℝ := 144

-- Given conditions
def profit_A_to_B (CP_A : ℝ) := 1.25 * CP_A
def profit_B_to_C (CP_B : ℝ) := 1.25 * CP_B
def SP_C := 225

-- Proof statement
theorem cost_price_of_A : 
  profit_B_to_C (profit_A_to_B CP_A) = SP_C :=
by
  sorry

end cost_price_of_A_l2413_241316
