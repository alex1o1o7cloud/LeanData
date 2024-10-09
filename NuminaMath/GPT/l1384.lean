import Mathlib

namespace maximum_n_l1384_138466

def number_of_trapezoids (n : ℕ) : ℕ := n * (n - 3) * (n - 2) * (n - 1) / 24

theorem maximum_n (n : ℕ) (h : number_of_trapezoids n ≤ 2012) : n ≤ 26 :=
by
  sorry

end maximum_n_l1384_138466


namespace ratio_of_larger_to_smaller_l1384_138475

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) :
  x / y = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l1384_138475


namespace problem_1_problem_2_l1384_138487

-- Proposition p
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2 * a * x + 2 - a)

-- Proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 + 2 * x + a ≥ 0

-- Problem 1: Prove that if p is true then a ≤ -2 or a ≥ 1
theorem problem_1 (a : ℝ) (hp : p a) : a ≤ -2 ∨ a ≥ 1 := sorry

-- Problem 2: Prove that if p ∨ q is true then a ≤ -2 or a ≥ 0
theorem problem_2 (a : ℝ) (hpq : p a ∨ q a) : a ≤ -2 ∨ a ≥ 0 := sorry

end problem_1_problem_2_l1384_138487


namespace Chris_age_proof_l1384_138436

theorem Chris_age_proof (m c : ℕ) (h1 : c = 3 * m - 22) (h2 : c + m = 70) : c = 47 := by
  sorry

end Chris_age_proof_l1384_138436


namespace pipe_B_fill_time_l1384_138453

-- Definitions based on the given conditions
def fill_time_by_ABC := 10  -- in hours
def B_is_twice_as_fast_as_C : Prop := ∀ C B, B = 2 * C
def A_is_twice_as_fast_as_B : Prop := ∀ A B, A = 2 * B

-- The main theorem to prove
theorem pipe_B_fill_time (A B C : ℝ) (h1: fill_time_by_ABC = 10) 
    (h2 : B_is_twice_as_fast_as_C) (h3 : A_is_twice_as_fast_as_B) : B = 1 / 35 :=
by
  sorry

end pipe_B_fill_time_l1384_138453


namespace correct_number_of_statements_l1384_138485

noncomputable def number_of_correct_statements := 1

def statement_1 : Prop := false -- Equal angles are not preserved
def statement_2 : Prop := false -- Equal lengths are not preserved
def statement_3 : Prop := false -- The longest segment feature is not preserved
def statement_4 : Prop := true  -- The midpoint feature is preserved

theorem correct_number_of_statements :
  (statement_1 ∧ statement_2 ∧ statement_3 ∧ statement_4) = true →
  number_of_correct_statements = 1 :=
by
  sorry

end correct_number_of_statements_l1384_138485


namespace simplify_complex_number_l1384_138481

theorem simplify_complex_number (i : ℂ) (h : i^2 = -1) : i * (1 - i)^2 = 2 := by
  sorry

end simplify_complex_number_l1384_138481


namespace angle_quadrant_l1384_138401

theorem angle_quadrant (theta : ℤ) (h_theta : theta = -3290) : 
  ∃ q : ℕ, q = 4 := 
by 
  sorry

end angle_quadrant_l1384_138401


namespace sin_squared_not_periodic_l1384_138404

noncomputable def sin_squared (x : ℝ) : ℝ := Real.sin (x^2)

theorem sin_squared_not_periodic : 
  ¬ (∃ T > 0, ∀ x ∈ Set.univ, sin_squared (x + T) = sin_squared x) := 
sorry

end sin_squared_not_periodic_l1384_138404


namespace age_ratio_in_8_years_l1384_138416

-- Define the conditions
variables (s l : ℕ) -- Sam's and Leo's current ages

def condition1 := s - 4 = 2 * (l - 4)
def condition2 := s - 10 = 3 * (l - 10)

-- Define the final problem
theorem age_ratio_in_8_years (h1 : condition1 s l) (h2 : condition2 s l) : 
  ∃ x : ℕ, x = 8 ∧ (s + x) / (l + x) = 3 / 2 :=
sorry

end age_ratio_in_8_years_l1384_138416


namespace theater_workshop_l1384_138449

-- Definitions of the conditions
def total_participants : ℕ := 120
def cannot_craft_poetry : ℕ := 52
def cannot_perform_painting : ℕ := 75
def not_skilled_in_photography : ℕ := 38
def participants_with_exactly_two_skills : ℕ := 195 - total_participants

-- The theorem stating the problem
theorem theater_workshop :
  participants_with_exactly_two_skills = 75 := by
  sorry

end theater_workshop_l1384_138449


namespace find_xy_l1384_138450

theorem find_xy (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : (x - 10)^2 + (y - 10)^2 = 18) : 
  x * y = 91 := 
by {
  sorry
}

end find_xy_l1384_138450


namespace sequence_bound_l1384_138463

-- Definitions and assumptions based on the conditions
def valid_sequence (a : ℕ → ℕ) (N : ℕ) (m : ℕ) :=
  (1 ≤ a 1) ∧ (a m ≤ N) ∧ (∀ i j, 1 ≤ i → i < j → j ≤ m → a i < a j) ∧ 
  (∀ i j, 1 ≤ i → i < j → j ≤ m → Nat.lcm (a i) (a j) ≤ N)

-- The main theorem to prove
theorem sequence_bound (a : ℕ → ℕ) (N : ℕ) (m : ℕ) 
  (h : valid_sequence a N m) : m ≤ 2 * Nat.floor (Real.sqrt N) :=
sorry

end sequence_bound_l1384_138463


namespace number_of_binders_l1384_138431

-- Definitions of given conditions
def book_cost : Nat := 16
def binder_cost : Nat := 2
def notebooks_cost : Nat := 6
def total_cost : Nat := 28

-- Variable for the number of binders
variable (b : Nat)

-- Proposition that the number of binders Léa bought is 3
theorem number_of_binders (h : book_cost + binder_cost * b + notebooks_cost = total_cost) : b = 3 :=
by
  sorry

end number_of_binders_l1384_138431


namespace add_fractions_l1384_138493

theorem add_fractions : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by sorry

end add_fractions_l1384_138493


namespace no_positive_integer_solutions_l1384_138454

theorem no_positive_integer_solutions (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^2017 - 1 ≠ (x - 1) * (y^2015 - 1) :=
by sorry

end no_positive_integer_solutions_l1384_138454


namespace edward_work_hours_edward_work_hours_overtime_l1384_138444

variable (H : ℕ) -- H represents the number of hours worked
variable (O : ℕ) -- O represents the number of overtime hours

theorem edward_work_hours (H_le_40 : H ≤ 40) (earning_eq_210 : 7 * H = 210) : H = 30 :=
by
  -- Proof to be filled in here
  sorry

theorem edward_work_hours_overtime (H_gt_40 : H > 40) (earning_eq_210 : 7 * 40 + 14 * (H - 40) = 210) : False :=
by
  -- Proof to be filled in here
  sorry

end edward_work_hours_edward_work_hours_overtime_l1384_138444


namespace bijective_bounded_dist_l1384_138400

open Int

theorem bijective_bounded_dist {k : ℕ} (f : ℤ → ℤ) 
    (hf_bijective : Function.Bijective f)
    (hf_property : ∀ i j : ℤ, |i - j| ≤ k → |f i - (f j)| ≤ k) :
    ∀ i j : ℤ, |f i - (f j)| = |i - j| := 
sorry

end bijective_bounded_dist_l1384_138400


namespace chicago_bulls_heat_games_total_l1384_138470

-- Statement of the problem in Lean 4
theorem chicago_bulls_heat_games_total :
  ∀ (bulls_games : ℕ) (heat_games : ℕ),
    bulls_games = 70 →
    heat_games = bulls_games + 5 →
    bulls_games + heat_games = 145 :=
by
  intros bulls_games heat_games h_bulls h_heat
  rw [h_bulls, h_heat]
  exact sorry

end chicago_bulls_heat_games_total_l1384_138470


namespace find_xyz_l1384_138459

theorem find_xyz
  (a b c x y z : ℂ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : a = (2 * b + 3 * c) / (x - 3))
  (h2 : b = (3 * a + 2 * c) / (y - 3))
  (h3 : c = (2 * a + 2 * b) / (z - 3))
  (h4 : x * y + x * z + y * z = -1)
  (h5 : x + y + z = 1) :
  x * y * z = 1 :=
sorry

end find_xyz_l1384_138459


namespace probability_of_drawing_black_ball_l1384_138422

/-- The bag contains 2 black balls and 3 white balls. 
    The balls are identical except for their colors. 
    A ball is randomly drawn from the bag. -/
theorem probability_of_drawing_black_ball (b w : ℕ) (hb : b = 2) (hw : w = 3) :
    (b + w > 0) → (b / (b + w) : ℚ) = 2 / 5 :=
by
  intros h
  rw [hb, hw]
  norm_num

end probability_of_drawing_black_ball_l1384_138422


namespace student_question_choices_l1384_138458

-- Definitions based on conditions
def partA_questions := 10
def partB_questions := 10
def choose_from_partA := 8
def choose_from_partB := 5

-- The proof problem statement
theorem student_question_choices :
  (Nat.choose partA_questions choose_from_partA) * (Nat.choose partB_questions choose_from_partB) = 11340 :=
by
  sorry

end student_question_choices_l1384_138458


namespace y_intercept_of_parallel_line_l1384_138495

theorem y_intercept_of_parallel_line (m x1 y1 : ℝ) (h_slope : m = -3) (h_point : (x1, y1) = (3, -1))
  (b : ℝ) (h_line_parallel : ∀ x, b = y1 + m * (x - x1)) :
  b = 8 :=
by
  sorry

end y_intercept_of_parallel_line_l1384_138495


namespace geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_l1384_138448

-- Define geometric body type
inductive GeometricBody
  | rectangularPrism
  | cylinder

-- Define the condition where both front and left views are rectangles
def hasRectangularViews (body : GeometricBody) : Prop :=
  body = GeometricBody.rectangularPrism ∨ body = GeometricBody.cylinder

-- The theorem statement
theorem geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder (body : GeometricBody) :
  hasRectangularViews body :=
sorry

end geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_l1384_138448


namespace f_log2_9_l1384_138410

def f (x : ℝ) : ℝ := sorry

theorem f_log2_9 : 
  (∀ x, f (x + 1) = 1 / f x) → 
  (∀ x, 0 < x ∧ x ≤ 1 → f x = 2^x) → 
  f (Real.log 9 / Real.log 2) = 8 / 9 :=
by
  intros h1 h2
  sorry

end f_log2_9_l1384_138410


namespace area_of_shaded_region_l1384_138461

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

end area_of_shaded_region_l1384_138461


namespace problem_l1384_138474

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

theorem problem 
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z) 
  (h4 : x * y = 30) 
  (h5 : x * z = 60) 
  (h6 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := 
  sorry

end problem_l1384_138474


namespace positive_difference_solutions_l1384_138405

theorem positive_difference_solutions : 
  ∀ (r : ℝ), r ≠ -3 → 
  (∃ r1 r2 : ℝ, (r^2 - 6*r - 20) / (r + 3) = 3*r + 10 → r1 ≠ r2 ∧ 
  |r1 - r2| = 20) :=
by
  sorry

end positive_difference_solutions_l1384_138405


namespace infinite_solutions_a_l1384_138486

theorem infinite_solutions_a (a : ℝ) :
  (∀ x : ℝ, 3 * (2 * x - a) = 2 * (3 * x + 12)) ↔ a = -8 :=
by
  sorry

end infinite_solutions_a_l1384_138486


namespace simple_interest_calculation_l1384_138403

-- Define the principal (P), rate (R), and time (T)
def principal : ℝ := 10000
def rate : ℝ := 0.08
def time : ℝ := 1

-- Define the simple interest formula
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- The theorem to be proved
theorem simple_interest_calculation : simple_interest principal rate time = 800 :=
by
  -- Proof steps would go here, but this is left as an exercise
  sorry

end simple_interest_calculation_l1384_138403


namespace intersection_point_l1384_138424

theorem intersection_point (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) :
  ∃ x y, (y = a * x^2 + b * x + c) ∧ (y = a * x^2 - b * x + c + d) ∧ x = d / (2 * b) ∧ y = a * (d / (2 * b))^2 + (d / 2) + c :=
by
  sorry

end intersection_point_l1384_138424


namespace sodas_to_take_back_l1384_138497

def num_sodas_brought : ℕ := 50
def num_sodas_drank : ℕ := 38

theorem sodas_to_take_back : (num_sodas_brought - num_sodas_drank) = 12 := by
  sorry

end sodas_to_take_back_l1384_138497


namespace combined_distance_l1384_138443

theorem combined_distance (second_lady_distance : ℕ) (first_lady_distance : ℕ) 
  (h1 : second_lady_distance = 4) 
  (h2 : first_lady_distance = 2 * second_lady_distance) : 
  first_lady_distance + second_lady_distance = 12 :=
by 
  sorry

end combined_distance_l1384_138443


namespace find_A_solution_l1384_138413

theorem find_A_solution (A : ℝ) (h : 32 * A^3 = 42592) : A = 11 :=
sorry

end find_A_solution_l1384_138413


namespace cats_given_by_Mr_Sheridan_l1384_138484

-- Definitions of the initial state and final state
def initial_cats : Nat := 17
def total_cats : Nat := 31

-- Proof statement that Mr. Sheridan gave her 14 cats
theorem cats_given_by_Mr_Sheridan : total_cats - initial_cats = 14 := by
  sorry

end cats_given_by_Mr_Sheridan_l1384_138484


namespace intersection_points_and_verification_l1384_138445

theorem intersection_points_and_verification :
  (∃ x y : ℝ, y = -3 * x ∧ y + 3 = 9 * x ∧ x = 1 / 4 ∧ y = -3 / 4) ∧
  ¬ (y = 2 * (1 / 4) - 1 ∧ (2 * (1 / 4) - 1 = -3 / 4)) :=
by
  sorry

end intersection_points_and_verification_l1384_138445


namespace garden_sparrows_l1384_138407

theorem garden_sparrows (ratio_b_s : ℕ) (bluebirds sparrows : ℕ)
  (h1 : ratio_b_s = 4 / 5) (h2 : bluebirds = 28) :
  sparrows = 35 :=
  sorry

end garden_sparrows_l1384_138407


namespace infinite_k_lcm_gt_ck_l1384_138447

theorem infinite_k_lcm_gt_ck 
  (a : ℕ → ℕ) 
  (distinct_pos : ∀ n m : ℕ, n ≠ m → a n ≠ a m) 
  (pos : ∀ n, 0 < a n) 
  (c : ℝ) 
  (c_pos : 0 < c) 
  (c_lt : c < 1.5) : 
  ∃ᶠ k in at_top, (Nat.lcm (a k) (a (k + 1)) : ℝ) > c * k :=
sorry

end infinite_k_lcm_gt_ck_l1384_138447


namespace smallest_n_interesting_meeting_l1384_138402

theorem smallest_n_interesting_meeting (m : ℕ) (hm : 2 ≤ m) :
  ∀ (n : ℕ), (n ≤ 3 * m - 1) ∧ (∀ (rep : Finset (Fin (3 * m))), rep.card = n →
  ∃ subrep : Finset (Fin (3 * m)), subrep.card = 3 ∧ ∀ (x y : Fin (3 * m)), x ∈ subrep → y ∈ subrep → x ≠ y → ∃ z : Fin (3 * m), z ∈ subrep ∧ z = x + y) → n = 2 * m + 1 := by
  sorry

end smallest_n_interesting_meeting_l1384_138402


namespace quadrilateral_front_view_iff_cylinder_or_prism_l1384_138498

inductive Solid
| cone : Solid
| cylinder : Solid
| triangular_pyramid : Solid
| quadrangular_prism : Solid

def has_quadrilateral_front_view (s : Solid) : Prop :=
  s = Solid.cylinder ∨ s = Solid.quadrangular_prism

theorem quadrilateral_front_view_iff_cylinder_or_prism (s : Solid) :
  has_quadrilateral_front_view s ↔ s = Solid.cylinder ∨ s = Solid.quadrangular_prism :=
by
  sorry

end quadrilateral_front_view_iff_cylinder_or_prism_l1384_138498


namespace interest_rate_per_annum_l1384_138423

noncomputable def principal : ℝ := 933.3333333333334
noncomputable def amount : ℝ := 1120
noncomputable def time : ℝ := 4

theorem interest_rate_per_annum (P A T : ℝ) (hP : P = principal) (hA : A = amount) (hT : T = time) :
  ∃ R : ℝ, R = 1.25 :=
sorry

end interest_rate_per_annum_l1384_138423


namespace floor_diff_l1384_138409

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l1384_138409


namespace inradius_of_triangle_l1384_138479

variable (A : ℝ) (p : ℝ) (r : ℝ) (s : ℝ)

theorem inradius_of_triangle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_l1384_138479


namespace initial_peanuts_count_l1384_138499

def peanuts_initial (P : ℕ) : Prop :=
  P - (1 / 4 : ℝ) * P - 29 = 82

theorem initial_peanuts_count (P : ℕ) (h : peanuts_initial P) : P = 148 :=
by
  -- The complete proof can be constructed here.
  sorry

end initial_peanuts_count_l1384_138499


namespace relationship_l1384_138480

-- Definitions for the points on the inverse proportion function
def on_inverse_proportion (x : ℝ) (y : ℝ) : Prop :=
  y = -6 / x

-- Given conditions
def A (y1 : ℝ) : Prop :=
  on_inverse_proportion (-3) y1

def B (y2 : ℝ) : Prop :=
  on_inverse_proportion (-1) y2

def C (y3 : ℝ) : Prop :=
  on_inverse_proportion (2) y3

-- The theorem that expresses the relationship
theorem relationship (y1 y2 y3 : ℝ) (hA : A y1) (hB : B y2) (hC : C y3) : y3 < y1 ∧ y1 < y2 :=
by
  -- skeleton of proof
  sorry

end relationship_l1384_138480


namespace exponentiation_equality_l1384_138489

theorem exponentiation_equality :
  3^12 * 8^12 * 3^3 * 8^8 = 24 ^ 15 * 32768 := by
  sorry

end exponentiation_equality_l1384_138489


namespace find_x2_plus_y2_l1384_138455

theorem find_x2_plus_y2 (x y : ℝ) (h : (x ^ 2 + y ^ 2 + 1) * (x ^ 2 + y ^ 2 - 3) = 5) : x ^ 2 + y ^ 2 = 4 := 
by 
  sorry

end find_x2_plus_y2_l1384_138455


namespace correct_average_l1384_138471

theorem correct_average (avg_incorrect : ℕ) (old_num new_num : ℕ) (n : ℕ)
  (h_avg : avg_incorrect = 15)
  (h_old_num : old_num = 26)
  (h_new_num : new_num = 36)
  (h_n : n = 10) :
  (avg_incorrect * n + (new_num - old_num)) / n = 16 := by
  sorry

end correct_average_l1384_138471


namespace error_percentage_calc_l1384_138464

theorem error_percentage_calc (y : ℝ) (hy : y > 0) : 
  let correct_result := 8 * y
  let erroneous_result := y / 8
  let error := abs (correct_result - erroneous_result)
  let error_percentage := (error / correct_result) * 100
  error_percentage = 98 := by
  sorry

end error_percentage_calc_l1384_138464


namespace lowest_discount_l1384_138446

theorem lowest_discount (c m : ℝ) (p : ℝ) (h_c : c = 100) (h_m : m = 150) (h_p : p = 0.05) :
  ∃ (x : ℝ), m * (x / 100) = c * (1 + p) ∧ x = 70 :=
by
  use 70
  sorry

end lowest_discount_l1384_138446


namespace point_on_parallel_line_with_P_l1384_138421

-- Definitions
def is_on_parallel_line_x_axis (P Q : ℝ × ℝ) : Prop :=
  P.snd = Q.snd

theorem point_on_parallel_line_with_P :
  let P := (3, -2)
  let D := (-3, -2)
  is_on_parallel_line_x_axis P D :=
by
  sorry

end point_on_parallel_line_with_P_l1384_138421


namespace range_of_a_l1384_138469

variable (f : ℝ → ℝ)
variable (a : ℝ)

theorem range_of_a (h1 : ∀ a : ℝ, (f (1 - 2 * a) / 2 ≥ f a))
                  (h2 : ∀ (x1 x2 : ℝ), x1 < x2 ∧ x1 + x2 ≠ 0 → f x1 > f x2) : a > (1 / 2) :=
by
  sorry

end range_of_a_l1384_138469


namespace percentage_change_l1384_138473

def original_income (P T : ℝ) : ℝ :=
  P * T

def new_income (P T : ℝ) : ℝ :=
  (P * 1.3333) * (T * 0.6667)

theorem percentage_change (P T : ℝ) (hP : P ≠ 0) (hT : T ≠ 0) :
  ((new_income P T - original_income P T) / original_income P T) * 100 = -11.11 :=
by
  sorry

end percentage_change_l1384_138473


namespace general_term_formula_sum_inequality_l1384_138428

noncomputable def a (n : ℕ) : ℝ := if n > 0 then (-1)^(n-1) * 3 / 2^n else 0

noncomputable def S (n : ℕ) : ℝ := if n > 0 then 1 - (-1/2)^n else 0

theorem general_term_formula (n : ℕ) (hn : n > 0) :
  a n = (-1)^(n-1) * (3/2^n) :=
by sorry

theorem sum_inequality (n : ℕ) (hn : n > 0) :
  S n + 1 / S n ≤ 13 / 6 :=
by sorry

end general_term_formula_sum_inequality_l1384_138428


namespace arithmetic_sequence_sum_ratio_l1384_138433

theorem arithmetic_sequence_sum_ratio
  (a_n : ℕ → ℝ)
  (d a1 : ℝ)
  (S_n : ℕ → ℝ)
  (h_arithmetic : ∀ n, a_n n = a1 + (n-1) * d)
  (h_sum : ∀ n, S_n n = n / 2 * (2 * a1 + (n-1) * d))
  (h_ratio : S_n 4 / S_n 6 = -2 / 3) :
  S_n 5 / S_n 8 = 1 / 40.8 :=
sorry

end arithmetic_sequence_sum_ratio_l1384_138433


namespace evaluate_expression_l1384_138408

theorem evaluate_expression (c d : ℕ) (hc : c = 4) (hd : d = 2) :
  (c^c - c * (c - d)^c)^c = 136048896 := by
  sorry

end evaluate_expression_l1384_138408


namespace abs_nonneg_rational_l1384_138406

theorem abs_nonneg_rational (a : ℚ) : |a| ≥ 0 :=
sorry

end abs_nonneg_rational_l1384_138406


namespace sum_of_solutions_of_quadratic_l1384_138477

theorem sum_of_solutions_of_quadratic :
  ∀ a b c x₁ x₂ : ℝ, a ≠ 0 →
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ (x = x₁ ∨ x = x₂)) →
  (∃ s : ℝ, s = x₁ + x₂ ∧ -b / a = s) :=
by
  sorry

end sum_of_solutions_of_quadratic_l1384_138477


namespace hyperbola_foci_x_axis_range_l1384_138420

theorem hyperbola_foci_x_axis_range (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (m + 2)) - (y^2 / (m - 1)) = 1) →
  (1 < m) ↔ 
  (∀ x y : ℝ, (m + 2 > 0) ∧ (m - 1 > 0)) :=
sorry

end hyperbola_foci_x_axis_range_l1384_138420


namespace correct_factoring_example_l1384_138488

-- Define each option as hypotheses
def optionA (a b : ℝ) : Prop := (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2
def optionB (a b : ℝ) : Prop := 2 * a ^ 2 - a * b - a = a * (2 * a - b - 1)
def optionC (a b : ℝ) : Prop := 8 * a ^ 5 * b ^ 2 = 4 * a ^ 3 * b * 2 * a ^ 2 * b
def optionD (a : ℝ) : Prop := a ^ 2 - 4 * a + 3 = (a - 1) * (a - 3)

-- The goal is to prove that optionD is the correct example of factoring
theorem correct_factoring_example (a b : ℝ) : optionD a ↔ (∀ a b, ¬ optionA a b) ∧ (∀ a b, ¬ optionB a b) ∧ (∀ a b, ¬ optionC a b) :=
by
  sorry

end correct_factoring_example_l1384_138488


namespace find_sets_l1384_138476

theorem find_sets (A B : Set ℕ) :
  A ∩ B = {1, 2, 3} ∧ A ∪ B = {1, 2, 3, 4, 5} →
    (A = {1, 2, 3} ∧ B = {1, 2, 3, 4, 5}) ∨
    (A = {1, 2, 3, 4, 5} ∧ B = {1, 2, 3}) ∨
    (A = {1, 2, 3, 4} ∧ B = {1, 2, 3, 5}) ∨
    (A = {1, 2, 3, 5} ∧ B = {1, 2, 3, 4}) :=
by
  sorry

end find_sets_l1384_138476


namespace squirrel_calories_l1384_138435

def rabbits_caught_per_hour := 2
def rabbits_calories := 800
def squirrels_caught_per_hour := 6
def extra_calories_squirrels := 200

theorem squirrel_calories : 
  ∀ (S : ℕ), 
  (6 * S = (2 * 800) + 200) → S = 300 := by
  intros S h
  sorry

end squirrel_calories_l1384_138435


namespace incorrect_inequality_l1384_138451

theorem incorrect_inequality (a b : ℝ) (h : a > b ∧ b > 0) :
  ¬ (1 / a > 1 / b) :=
by
  sorry

end incorrect_inequality_l1384_138451


namespace expr_div_24_l1384_138439

theorem expr_div_24 (a : ℤ) : 24 ∣ ((a^2 + 3*a + 1)^2 - 1) := 
by 
  sorry

end expr_div_24_l1384_138439


namespace min_value_of_sum_squares_l1384_138442

noncomputable def min_value_sum_squares 
  (y1 y2 y3 : ℝ) (h1 : 2 * y1 + 3 * y2 + 4 * y3 = 120) 
  (h2 : 0 < y1) (h3 : 0 < y2) (h4 : 0 < y3) : ℝ :=
  y1^2 + y2^2 + y3^2

theorem min_value_of_sum_squares 
  (y1 y2 y3 : ℝ) (h1 : 2 * y1 + 3 * y2 + 4 * y3 = 120) 
  (h2 : 0 < y1) (h3 : 0 < y2) (h4 : 0 < y3) : 
  min_value_sum_squares y1 y2 y3 h1 h2 h3 h4 = 14400 / 29 := 
sorry

end min_value_of_sum_squares_l1384_138442


namespace num_sets_l1384_138468

theorem num_sets {A : Set ℕ} :
  {1} ⊆ A ∧ A ⊆ {1, 2, 3, 4, 5} → ∃ n, n = 16 := 
by
  sorry

end num_sets_l1384_138468


namespace other_train_length_l1384_138417

noncomputable def relative_speed (speed1 speed2 : ℝ) : ℝ :=
  speed1 + speed2

noncomputable def speed_in_km_per_sec (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr / 3600

noncomputable def total_distance_crossed (relative_speed : ℝ) (time_sec : ℕ) : ℝ :=
  relative_speed * (time_sec : ℝ)

noncomputable def length_of_other_train (total_distance length_of_first_train : ℝ) : ℝ :=
  total_distance - length_of_first_train

theorem other_train_length :
  let speed1 := 210
  let speed2 := 90
  let length_of_first_train := 0.9
  let time_taken := 24
  let relative_speed_km_per_hr := relative_speed speed1 speed2
  let relative_speed_km_per_sec := speed_in_km_per_sec relative_speed_km_per_hr
  let total_distance := total_distance_crossed relative_speed_km_per_sec time_taken
  length_of_other_train total_distance length_of_first_train = 1.1 := 
by
  sorry

end other_train_length_l1384_138417


namespace set_equality_proof_l1384_138412

theorem set_equality_proof :
  (∃ (u : ℤ), ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l) ↔
  (∃ (u : ℤ), ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r) :=
sorry

end set_equality_proof_l1384_138412


namespace square_area_is_correct_l1384_138441

noncomputable def find_area_of_square (x : ℚ) : ℚ :=
  let side := 6 * x - 27
  side * side

theorem square_area_is_correct (x : ℚ) (h1 : 6 * x - 27 = 30 - 2 * x) :
  find_area_of_square x = 248.0625 :=
by
  sorry

end square_area_is_correct_l1384_138441


namespace stratified_sampling_b_members_l1384_138430

variable (groupA : ℕ) (groupB : ℕ) (groupC : ℕ) (sampleSize : ℕ)

-- Conditions from the problem
def condition1 : groupA = 45 := by sorry
def condition2 : groupB = 45 := by sorry
def condition3 : groupC = 60 := by sorry
def condition4 : sampleSize = 10 := by sorry

-- The proof problem statement
theorem stratified_sampling_b_members : 
  (sampleSize * groupB) / (groupA + groupB + groupC) = 3 :=
by sorry

end stratified_sampling_b_members_l1384_138430


namespace initial_hotdogs_l1384_138452

-- Definitions
variable (x : ℕ)

-- Conditions
def condition : Prop := x - 2 = 97 

-- Statement to prove
theorem initial_hotdogs (h : condition x) : x = 99 :=
  by
    sorry

end initial_hotdogs_l1384_138452


namespace max_q_value_l1384_138437

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l1384_138437


namespace red_minus_white_more_l1384_138414

variable (flowers_total yellow_white red_yellow red_white : ℕ)
variable (h1 : flowers_total = 44)
variable (h2 : yellow_white = 13)
variable (h3 : red_yellow = 17)
variable (h4 : red_white = 14)

theorem red_minus_white_more : 
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end red_minus_white_more_l1384_138414


namespace initial_kids_l1384_138426

theorem initial_kids {N : ℕ} (h1 : 1 / 2 * N = N / 2) (h2 : 1 / 2 * (N / 2) = N / 4) (h3 : N / 4 = 5) : N = 20 :=
by
  sorry

end initial_kids_l1384_138426


namespace consecutive_integer_sum_l1384_138440

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l1384_138440


namespace total_ingredient_cups_l1384_138415

def butter_flour_sugar_ratio_butter := 2
def butter_flour_sugar_ratio_flour := 5
def butter_flour_sugar_ratio_sugar := 3
def flour_used := 15

theorem total_ingredient_cups :
  butter_flour_sugar_ratio_butter + 
  butter_flour_sugar_ratio_flour + 
  butter_flour_sugar_ratio_sugar = 10 →
  flour_used / butter_flour_sugar_ratio_flour = 3 →
  6 + 15 + 9 = 30 := by
  intros
  sorry

end total_ingredient_cups_l1384_138415


namespace function_increasing_no_negative_roots_l1384_138472

noncomputable def f (a x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem function_increasing (a : ℝ) (h : a > 1) : 
  ∀ (x1 x2 : ℝ), (-1 < x1) → (x1 < x2) → (f a x1 < f a x2) := 
by
  -- placeholder proof
  sorry

theorem no_negative_roots (a : ℝ) (h : a > 1) : 
  ∀ (x : ℝ), (x < 0) → (f a x ≠ 0) := 
by
  -- placeholder proof
  sorry

end function_increasing_no_negative_roots_l1384_138472


namespace frequency_of_heads_l1384_138432

theorem frequency_of_heads (n h : ℕ) (h_n : n = 100) (h_h : h = 49) : (h : ℚ) / n = 0.49 :=
by
  rw [h_n, h_h]
  norm_num

end frequency_of_heads_l1384_138432


namespace pre_image_of_f_5_1_l1384_138456

def f (x y : ℝ) : ℝ × ℝ := (x + y, 2 * x - y)

theorem pre_image_of_f_5_1 : ∃ (x y : ℝ), f x y = (5, 1) ∧ (x, y) = (2, 3) :=
by
  sorry

end pre_image_of_f_5_1_l1384_138456


namespace abs_neg_two_l1384_138467

theorem abs_neg_two : abs (-2) = 2 := 
by 
  sorry

end abs_neg_two_l1384_138467


namespace smallest_difference_l1384_138429

theorem smallest_difference {a b : ℕ} (h1: a * b = 2010) (h2: a > b) : a - b = 37 :=
sorry

end smallest_difference_l1384_138429


namespace tan_alpha_ratio_expression_l1384_138496

variable (α : Real)
variable (h1 : Real.sin α = 3/5)
variable (h2 : π/2 < α ∧ α < π)

theorem tan_alpha {α : Real}
  (h1 : Real.sin α = 3/5)
  (h2 : π/2 < α ∧ α < π)
  : Real.tan α = -3/4 := sorry

theorem ratio_expression {α : Real}
  (h1 : Real.sin α = 3/5)
  (h2 : π/2 < α ∧ α < π)
  : (2 * Real.sin α + 3 * Real.cos α) / (Real.cos α - Real.sin α) = 6/7 := sorry

end tan_alpha_ratio_expression_l1384_138496


namespace units_digit_square_l1384_138482

theorem units_digit_square (n : ℕ) (h1 : n ≥ 10 ∧ n < 100) (h2 : (n % 10 = 2) ∨ (n % 10 = 7)) :
  ∀ (d : ℕ), (d = 2 ∨ d = 6 ∨ d = 3) → (n^2 % 10 ≠ d) :=
by
  sorry

end units_digit_square_l1384_138482


namespace problem_one_problem_two_l1384_138457

-- Define p and q
def p (a x : ℝ) : Prop := (x - 3 * a) * (x - a) < 0
def q (x : ℝ) : Prop := |x - 3| < 1

-- Problem (1)
theorem problem_one (a : ℝ) (h_a : a = 1) (h_pq : p a x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Problem (2)
theorem problem_two (a : ℝ) (h_a_pos : a > 0) (suff : ¬ p a x → ¬ q x) (not_necess : ¬ (¬ q x → ¬ p a x)) : 
  (4 / 3 ≤ a ∧ a ≤ 2) := by
  sorry

end problem_one_problem_two_l1384_138457


namespace max_dot_product_OB_OA_l1384_138478

theorem max_dot_product_OB_OA (P A O B : ℝ × ℝ)
  (h₁ : ∃ x y : ℝ, (x, y) = P ∧ x^2 / 16 - y^2 / 9 = 1)
  (t : ℝ)
  (h₂ : A = (t - 1) • P)
  (h₃ : P • O = 64)
  (h₄ : B = (0, 1)) :
  ∃ t : ℝ, abs (B • A) ≤ (24/5) := 
sorry

end max_dot_product_OB_OA_l1384_138478


namespace rectangle_perimeter_inscribed_l1384_138491

noncomputable def circle_area : ℝ := 32 * Real.pi
noncomputable def rectangle_area : ℝ := 34
noncomputable def rectangle_perimeter : ℝ := 28

theorem rectangle_perimeter_inscribed (area_circle : ℝ := 32 * Real.pi)
  (area_rectangle : ℝ := 34) : ∃ (P : ℝ), P = 28 :=
by
  use rectangle_perimeter
  sorry

end rectangle_perimeter_inscribed_l1384_138491


namespace man_cannot_row_against_stream_l1384_138419

theorem man_cannot_row_against_stream (rate_in_still_water speed_with_stream : ℝ)
  (h_rate : rate_in_still_water = 1)
  (h_speed_with : speed_with_stream = 6) :
  ¬ ∃ (speed_against_stream : ℝ), speed_against_stream = rate_in_still_water - (speed_with_stream - rate_in_still_water) :=
by
  sorry

end man_cannot_row_against_stream_l1384_138419


namespace prime_eq_solution_l1384_138483

theorem prime_eq_solution (a b : ℕ) (h1 : Nat.Prime a) (h2 : b > 0)
  (h3 : 9 * (2 * a + b) ^ 2 = 509 * (4 * a + 511 * b)) : 
  (a = 251 ∧ b = 7) :=
sorry

end prime_eq_solution_l1384_138483


namespace tank_empty_time_l1384_138425

theorem tank_empty_time (R L : ℝ) (h1 : R = 1 / 7) (h2 : R - L = 1 / 8) : 
  (1 / L) = 56 :=
by
  sorry

end tank_empty_time_l1384_138425


namespace exists_integers_cd_iff_divides_l1384_138438

theorem exists_integers_cd_iff_divides (a b : ℤ) :
  (∃ c d : ℤ, a + b + c + d = 0 ∧ a * c + b * d = 0) ↔ (a - b) ∣ (2 * a * b) := 
by
  sorry

end exists_integers_cd_iff_divides_l1384_138438


namespace sugar_needed_l1384_138494

variable (a b c d : ℝ)
variable (H1 : a = 2)
variable (H2 : b = 1)
variable (H3 : d = 5)

theorem sugar_needed (c : ℝ) : c = 2.5 :=
by
  have H : 2 / 1 = 5 / c := by {
    sorry
  }
  sorry

end sugar_needed_l1384_138494


namespace find_f_prime_at_2_l1384_138460

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l1384_138460


namespace circle_center_radius_1_circle_center_coordinates_radius_1_l1384_138427

theorem circle_center_radius_1 (x y : ℝ) : 
  x^2 + y^2 + 2*x - 4*y - 3 = 0 ↔ (x + 1)^2 + (y - 2)^2 = 8 :=
sorry

theorem circle_center_coordinates_radius_1 : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y - 3 = 0 ∧ (x, y) = (-1, 2)) ∧ 
  (∃ r : ℝ, r = 2*Real.sqrt 2) :=
sorry

end circle_center_radius_1_circle_center_coordinates_radius_1_l1384_138427


namespace factory_material_equation_correct_l1384_138418

variable (a b x : ℝ)
variable (h_a : a = 180)
variable (h_b : b = 120)
variable (h_condition : (a - 2 * x) - (b + x) = 30)

theorem factory_material_equation_correct : (180 - 2 * x) - (120 + x) = 30 := by
  rw [←h_a, ←h_b]
  exact h_condition

end factory_material_equation_correct_l1384_138418


namespace simplify_expression_l1384_138434

theorem simplify_expression (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 :=
by
  sorry

end simplify_expression_l1384_138434


namespace combined_weight_of_three_new_people_l1384_138462

theorem combined_weight_of_three_new_people 
  (W : ℝ) 
  (h_avg_increase : (W + 80) / 20 = W / 20 + 4) 
  (h_replaced_weights : 60 + 75 + 85 = 220) : 
  220 + 80 = 300 :=
by
  sorry

end combined_weight_of_three_new_people_l1384_138462


namespace no_integer_solutions_l1384_138465

theorem no_integer_solutions
  (x y : ℤ) :
  3 * x^2 = 16 * y^2 + 8 * y + 5 → false :=
by
  sorry

end no_integer_solutions_l1384_138465


namespace current_number_of_people_l1384_138490

theorem current_number_of_people (a b : ℕ) : 0 ≤ a → 0 ≤ b → 48 - a + b ≥ 0 := by
  sorry

end current_number_of_people_l1384_138490


namespace problem_statement_l1384_138492

theorem problem_statement (x m : ℝ) :
  (¬ (x > m) → ¬ (x^2 + x - 2 > 0)) ∧ (¬ (x > m) ↔ ¬ (x^2 + x - 2 > 0)) → m ≥ 1 :=
sorry

end problem_statement_l1384_138492


namespace angle_same_terminal_side_l1384_138411

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, 290 = k * 360 - 70 :=
by
  sorry

end angle_same_terminal_side_l1384_138411
