import Mathlib

namespace NUMINAMATH_GPT_fn_prime_factor_bound_l298_29877

theorem fn_prime_factor_bound (n : ℕ) (h : n ≥ 3) : 
  ∃ p : ℕ, Prime p ∧ (p ∣ (2^(2^n) + 1)) ∧ p > 2^(n+2) * (n+1) :=
sorry

end NUMINAMATH_GPT_fn_prime_factor_bound_l298_29877


namespace NUMINAMATH_GPT_solution_l298_29825

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), (x > 0 ∧ y > 0) ∧ (6 * x^2 + 18 * x * y = 2 * x^3 + 3 * x^2 * y^2) ∧ x = (3 + Real.sqrt 153) / 4

theorem solution : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_solution_l298_29825


namespace NUMINAMATH_GPT_upstream_swim_distance_l298_29861

-- Definition of the speeds and distances
def downstream_speed (v : ℝ) := 5 + v
def upstream_speed (v : ℝ) := 5 - v
def distance := 54
def time := 6
def woman_speed_in_still_water := 5

-- Given condition: downstream_speed * time = distance
def downstream_condition (v : ℝ) := downstream_speed v * time = distance

-- Given condition: upstream distance is 'd' km
def upstream_distance (v : ℝ) := upstream_speed v * time

-- Prove that given the above conditions and solving the necessary equations, 
-- the distance swam upstream is 6 km.
theorem upstream_swim_distance {d : ℝ} (v : ℝ) (h1 : downstream_condition v) : upstream_distance v = 6 :=
by
  sorry

end NUMINAMATH_GPT_upstream_swim_distance_l298_29861


namespace NUMINAMATH_GPT_original_cost_l298_29858

theorem original_cost (P : ℝ) (h : 0.76 * P = 608) : P = 800 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_l298_29858


namespace NUMINAMATH_GPT_mistaken_divisor_l298_29888

theorem mistaken_divisor (x : ℕ) (h1 : ∀ (d : ℕ), d ∣ 840 → d = 21 ∨ d = x) 
(h2 : 840 = 70 * x) : x = 12 := 
by sorry

end NUMINAMATH_GPT_mistaken_divisor_l298_29888


namespace NUMINAMATH_GPT_functional_equation_solution_l298_29882

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) = x * f x - y * f y) →
  ∃ m b : ℝ, ∀ t : ℝ, f t = m * t + b :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l298_29882


namespace NUMINAMATH_GPT_find_c_eq_3_l298_29883

theorem find_c_eq_3 (m b c : ℝ) :
  (∀ x y, y = m * x + c → ((x = b + 4 ∧ y = 5) ∨ (x = -2 ∧ y = 2))) →
  c = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_c_eq_3_l298_29883


namespace NUMINAMATH_GPT_joe_total_time_to_school_l298_29800

theorem joe_total_time_to_school:
  ∀ (d r_w: ℝ), (1 / 3) * d = r_w * 9 →
                  4 * r_w * (2 * (r_w * 9) / (3 * (4 * r_w))) = (2 / 3) * d →
                  (1 / 3) * d / r_w + (2 / 3) * d / (4 * r_w) = 13.5 :=
by
  intros d r_w h1 h2
  sorry

end NUMINAMATH_GPT_joe_total_time_to_school_l298_29800


namespace NUMINAMATH_GPT_reaction_completion_l298_29819

-- Definitions from conditions
def NaOH_moles : ℕ := 2
def H2O_moles : ℕ := 2

-- Given the balanced equation
-- 2 NaOH + H2SO4 → Na2SO4 + 2 H2O

theorem reaction_completion (H2SO4_moles : ℕ) :
  (2 * (NaOH_moles / 2)) = H2O_moles → H2SO4_moles = 1 :=
by 
  -- Skip proof
  sorry

end NUMINAMATH_GPT_reaction_completion_l298_29819


namespace NUMINAMATH_GPT_inverse_proportion_quadrant_l298_29824

theorem inverse_proportion_quadrant (k : ℝ) (h : k < 0) : 
  ∀ x : ℝ, (0 < x → y = k / x → y < 0) ∧ (x < 0 → y = k / x → 0 < y) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_quadrant_l298_29824


namespace NUMINAMATH_GPT_base6_divisible_19_l298_29874

theorem base6_divisible_19 (y : ℤ) : (19 ∣ (615 + 6 * y)) ↔ y = 2 := sorry

end NUMINAMATH_GPT_base6_divisible_19_l298_29874


namespace NUMINAMATH_GPT_symmetric_circle_eq_l298_29876

theorem symmetric_circle_eq (x y : ℝ) :
  (x^2 + y^2 - 4 * x = 0) ↔ (x^2 + y^2 - 4 * y = 0) :=
sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l298_29876


namespace NUMINAMATH_GPT_article_word_limit_l298_29829

theorem article_word_limit 
  (total_pages : ℕ) (large_font_pages : ℕ) (words_per_large_page : ℕ) 
  (words_per_small_page : ℕ) (remaining_pages : ℕ) (total_words : ℕ)
  (h1 : total_pages = 21) 
  (h2 : large_font_pages = 4) 
  (h3 : words_per_large_page = 1800) 
  (h4 : words_per_small_page = 2400) 
  (h5 : remaining_pages = total_pages - large_font_pages) 
  (h6 : total_words = large_font_pages * words_per_large_page + remaining_pages * words_per_small_page) :
  total_words = 48000 := 
by
  sorry

end NUMINAMATH_GPT_article_word_limit_l298_29829


namespace NUMINAMATH_GPT_smallest_m_l298_29897

theorem smallest_m (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n - m / n = 2011 / 3) : m = 1120 :=
sorry

end NUMINAMATH_GPT_smallest_m_l298_29897


namespace NUMINAMATH_GPT_trapezoid_CD_length_l298_29828

/-- In trapezoid ABCD with AD parallel to BC and diagonals intersecting:
  - BD = 2
  - ∠DBC = 36°
  - ∠BDA = 72°
  - The ratio BC : AD = 5 : 3

We are to show that the length of CD is 4/3. --/
theorem trapezoid_CD_length
  {A B C D : Type}
  (BD : ℝ) (DBC : ℝ) (BDA : ℝ) (BC_over_AD : ℝ)
  (AD_parallel_BC : Prop) (diagonals_intersect : Prop)
  (hBD : BD = 2) 
  (hDBC : DBC = 36) 
  (hBDA : BDA = 72)
  (hBC_over_AD : BC_over_AD = 5 / 3) 
  :  CD = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_CD_length_l298_29828


namespace NUMINAMATH_GPT_height_to_width_ratio_l298_29863

theorem height_to_width_ratio (w h l : ℝ) (V : ℝ) (x : ℝ) :
  (h = x * w) →
  (l = 7 * h) →
  (V = l * w * h) →
  (V = 129024) →
  (w = 8) →
  (x = 6) :=
by
  intros h_eq_xw l_eq_7h V_eq_lwh V_val w_val
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_height_to_width_ratio_l298_29863


namespace NUMINAMATH_GPT_orchard_apples_relation_l298_29839

/-- 
A certain orchard has 10 apple trees, and on average each tree can produce 200 apples. 
Based on experience, for each additional tree planted, the average number of apples produced per tree decreases by 5. 
We are to show that if the orchard has planted x additional apple trees and the total number of apples is y, then the relationship between y and x is:
y = (10 + x) * (200 - 5x)
-/
theorem orchard_apples_relation (x : ℕ) (y : ℕ) 
    (initial_trees : ℕ := 10)
    (initial_apples : ℕ := 200)
    (decrease_per_tree : ℕ := 5)
    (total_trees := initial_trees + x)
    (average_apples := initial_apples - decrease_per_tree * x)
    (total_apples := total_trees * average_apples) :
    y = total_trees * average_apples := 
  by 
    sorry

end NUMINAMATH_GPT_orchard_apples_relation_l298_29839


namespace NUMINAMATH_GPT_braiding_time_l298_29816

variables (n_dancers : ℕ) (b_braids_per_dancer : ℕ) (t_seconds_per_braid : ℕ)

theorem braiding_time : n_dancers = 8 → b_braids_per_dancer = 5 → t_seconds_per_braid = 30 → 
  (n_dancers * b_braids_per_dancer * t_seconds_per_braid) / 60 = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_braiding_time_l298_29816


namespace NUMINAMATH_GPT_local_minimum_condition_l298_29843

-- Define the function f(x)
def f (x b : ℝ) : ℝ := x ^ 3 - 3 * b * x + 3 * b

-- Define the first derivative of f(x)
def f_prime (x b : ℝ) : ℝ := 3 * x ^ 2 - 3 * b

-- Define the second derivative of f(x)
def f_double_prime (x b : ℝ) : ℝ := 6 * x

-- Theorem stating that f(x) has a local minimum if and only if b > 0
theorem local_minimum_condition (b : ℝ) (x : ℝ) (h : f_prime x b = 0) : f_double_prime x b > 0 ↔ b > 0 :=
by sorry

end NUMINAMATH_GPT_local_minimum_condition_l298_29843


namespace NUMINAMATH_GPT_fixed_point_exists_l298_29815

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y : ℝ, (x = 2 ∧ y = -2 ∧ (ax - 5 = y)) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_exists_l298_29815


namespace NUMINAMATH_GPT_initial_number_of_cards_l298_29813

theorem initial_number_of_cards (x : ℕ) (h : x + 76 = 79) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_cards_l298_29813


namespace NUMINAMATH_GPT_shoe_length_increase_l298_29835

theorem shoe_length_increase
  (L : ℝ)
  (x : ℝ)
  (h1 : L + 9*x = L * 1.2)
  (h2 : L + 7*x = 10.4) :
  x = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_shoe_length_increase_l298_29835


namespace NUMINAMATH_GPT_solutions_to_equation_l298_29845

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 10*x - 8)) + (1 / (x^2 + 3*x - 8)) + (1 / (x^2 - 12*x - 8)) = 0

theorem solutions_to_equation :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = -19 ∨ x = (5 + Real.sqrt 57) / 2 ∨ x = (5 - Real.sqrt 57) / 2) :=
sorry

end NUMINAMATH_GPT_solutions_to_equation_l298_29845


namespace NUMINAMATH_GPT_points_earned_l298_29885

-- Define the number of pounds required to earn one point
def pounds_per_point : ℕ := 4

-- Define the number of pounds Paige recycled
def paige_recycled : ℕ := 14

-- Define the number of pounds Paige's friends recycled
def friends_recycled : ℕ := 2

-- Define the total number of pounds recycled
def total_recycled : ℕ := paige_recycled + friends_recycled

-- Define the total number of points earned
def total_points : ℕ := total_recycled / pounds_per_point

-- Theorem to prove the total points earned
theorem points_earned : total_points = 4 := by
  sorry

end NUMINAMATH_GPT_points_earned_l298_29885


namespace NUMINAMATH_GPT_race_time_l298_29864

theorem race_time 
  (v t : ℝ)
  (h1 : 1000 = v * t)
  (h2 : 960 = v * (t + 10)) :
  t = 250 :=
by
  sorry

end NUMINAMATH_GPT_race_time_l298_29864


namespace NUMINAMATH_GPT_complement_A_is_closed_interval_l298_29810

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define the set A with the given condition
def A : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the complement of A with respect to U
def complement_A : Set ℝ := Set.compl A

theorem complement_A_is_closed_interval :
  complement_A = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry  -- Proof to be inserted

end NUMINAMATH_GPT_complement_A_is_closed_interval_l298_29810


namespace NUMINAMATH_GPT_maryann_work_time_l298_29814

variables (C A R : ℕ)

theorem maryann_work_time
  (h1 : A = 2 * C)
  (h2 : R = 6 * C)
  (h3 : C + A + R = 1440) :
  C = 160 ∧ A = 320 ∧ R = 960 :=
by
  sorry

end NUMINAMATH_GPT_maryann_work_time_l298_29814


namespace NUMINAMATH_GPT_cannot_all_white_without_diagonals_cannot_all_white_with_diagonals_l298_29827

/-- A 4x4 chessboard is entirely white except for one square which is black.
The allowed operations are flipping the colors of all squares in a column or in a row.
Prove that it is impossible to have all the squares the same color regardless of the position of the black square. -/
theorem cannot_all_white_without_diagonals :
  ∀ (i j : Fin 4), False :=
by sorry

/-- If diagonal flips are also allowed, prove that 
it is impossible to have all squares the same color if the black square is at certain positions. -/
theorem cannot_all_white_with_diagonals :
  ∀ (i j : Fin 4), (i, j) ≠ (0, 1) ∧ (i, j) ≠ (0, 2) ∧
                   (i, j) ≠ (1, 0) ∧ (i, j) ≠ (1, 3) ∧
                   (i, j) ≠ (2, 0) ∧ (i, j) ≠ (2, 3) ∧
                   (i, j) ≠ (3, 1) ∧ (i, j) ≠ (3, 2) → False :=
by sorry

end NUMINAMATH_GPT_cannot_all_white_without_diagonals_cannot_all_white_with_diagonals_l298_29827


namespace NUMINAMATH_GPT_trapezium_area_l298_29856

theorem trapezium_area (a b h : ℝ) (h_a : a = 4) (h_b : b = 5) (h_h : h = 6) :
  (1 / 2 * (a + b) * h) = 27 :=
by
  rw [h_a, h_b, h_h]
  norm_num

end NUMINAMATH_GPT_trapezium_area_l298_29856


namespace NUMINAMATH_GPT_book_pages_total_l298_29855

-- Definitions based on conditions
def pages_first_three_days: ℕ := 3 * 28
def pages_next_three_days: ℕ := 3 * 35
def pages_following_three_days: ℕ := 3 * 42
def pages_last_day: ℕ := 15

-- Total pages read calculated from above conditions
def total_pages_read: ℕ :=
  pages_first_three_days + pages_next_three_days + pages_following_three_days + pages_last_day

-- Proof problem statement: prove that the total pages read equal 330
theorem book_pages_total:
  total_pages_read = 330 :=
by
  sorry

end NUMINAMATH_GPT_book_pages_total_l298_29855


namespace NUMINAMATH_GPT_no_solution_frac_eq_l298_29830

theorem no_solution_frac_eq (k : ℝ) : (∀ x : ℝ, ¬(1 / (x + 1) = 3 * k / x)) ↔ (k = 0 ∨ k = 1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_frac_eq_l298_29830


namespace NUMINAMATH_GPT_cost_of_one_pencil_l298_29887

theorem cost_of_one_pencil (students : ℕ) (more_than_half : ℕ) (pencil_cost : ℕ) (pencils_each : ℕ)
  (total_cost : ℕ) (students_condition : students = 36) 
  (more_than_half_condition : more_than_half > 18) 
  (pencil_count_condition : pencils_each > 1) 
  (cost_condition : pencil_cost > pencils_each) 
  (total_cost_condition : students * pencil_cost * pencils_each = 1881) : 
  pencil_cost = 17 :=
sorry

end NUMINAMATH_GPT_cost_of_one_pencil_l298_29887


namespace NUMINAMATH_GPT_number_of_integer_solutions_is_zero_l298_29891

-- Define the problem conditions
def eq1 (x y z : ℤ) : Prop := x^2 - 3 * x * y + 2 * y^2 - z^2 = 27
def eq2 (x y z : ℤ) : Prop := -x^2 + 6 * y * z + 2 * z^2 = 52
def eq3 (x y z : ℤ) : Prop := x^2 + x * y + 8 * z^2 = 110

-- State the theorem to be proved
theorem number_of_integer_solutions_is_zero :
  ∀ (x y z : ℤ), eq1 x y z → eq2 x y z → eq3 x y z → false :=
by
  sorry

end NUMINAMATH_GPT_number_of_integer_solutions_is_zero_l298_29891


namespace NUMINAMATH_GPT_cube_volume_l298_29833

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V, V = 125 := 
by
  sorry

end NUMINAMATH_GPT_cube_volume_l298_29833


namespace NUMINAMATH_GPT_find_n_l298_29805

theorem find_n (x n : ℝ) (h1 : ((x / n) * 5) + 10 - 12 = 48) (h2 : x = 40) : n = 4 :=
sorry

end NUMINAMATH_GPT_find_n_l298_29805


namespace NUMINAMATH_GPT_exists_perfect_square_subtraction_l298_29826

theorem exists_perfect_square_subtraction {k : ℕ} (hk : k > 0) : 
  ∃ (n : ℕ), n > 0 ∧ ∃ m : ℕ, n * 2^k - 7 = m^2 := 
  sorry

end NUMINAMATH_GPT_exists_perfect_square_subtraction_l298_29826


namespace NUMINAMATH_GPT_my_op_eq_l298_29852

-- Define the custom operation
def my_op (m n : ℝ) : ℝ := m * n * (m - n)

-- State the theorem
theorem my_op_eq :
  ∀ (a b : ℝ), my_op (a + b) a = a^2 * b + a * b^2 :=
by intros a b; sorry

end NUMINAMATH_GPT_my_op_eq_l298_29852


namespace NUMINAMATH_GPT_prove_f_f_x_eq_4_prove_f_f_x_eq_5_l298_29841

variable (f : ℝ → ℝ)

-- Conditions
axiom f_of_4 : f (-2) = 4 ∧ f 2 = 4 ∧ f 6 = 4
axiom f_of_5 : f (-4) = 5 ∧ f 4 = 5

-- Intermediate Values
axiom f_inv_of_4 : f 0 = -2 ∧ f (-1) = 2 ∧ f 3 = 6
axiom f_inv_of_5 : f 2 = 4

theorem prove_f_f_x_eq_4 :
  {x : ℝ | f (f x) = 4} = {0, -1, 3} :=
by
  sorry

theorem prove_f_f_x_eq_5 :
  {x : ℝ | f (f x) = 5} = {2} :=
by
  sorry

end NUMINAMATH_GPT_prove_f_f_x_eq_4_prove_f_f_x_eq_5_l298_29841


namespace NUMINAMATH_GPT_average_of_consecutive_integers_l298_29871

theorem average_of_consecutive_integers (n m : ℕ) 
  (h1 : m = (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7) : 
  (n + 6) = (m + (m+1) + (m+2) + (m+3) + (m+4) + (m+5) + (m+6)) / 7 :=
by
  sorry

end NUMINAMATH_GPT_average_of_consecutive_integers_l298_29871


namespace NUMINAMATH_GPT_smallest_positive_integer_l298_29801

def is_prime_gt_60 (n : ℕ) : Prop :=
  n > 60 ∧ Prime n

def smallest_integer_condition (k : ℕ) : Prop :=
  ¬ Prime k ∧ ¬ (∃ m : ℕ, m * m = k) ∧ 
  ∀ p : ℕ, Prime p → p ∣ k → p > 60

theorem smallest_positive_integer : ∃ k : ℕ, k = 4087 ∧ smallest_integer_condition k := by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l298_29801


namespace NUMINAMATH_GPT_solve_for_diamond_l298_29804

-- Define what it means for a digit to represent a base-9 number and base-10 number
noncomputable def fromBase (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * b + d) 0

-- The theorem we want to prove
theorem solve_for_diamond (diamond : ℕ) (h_digit : diamond < 10) :
  fromBase 9 [diamond, 3] = fromBase 10 [diamond, 2] → diamond = 1 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_diamond_l298_29804


namespace NUMINAMATH_GPT_Punta_position_l298_29896

theorem Punta_position (N x y p : ℕ) (h1 : N = 36) (h2 : x = y / 4) (h3 : x + y = 35) : p = 8 := by
  sorry

end NUMINAMATH_GPT_Punta_position_l298_29896


namespace NUMINAMATH_GPT_shells_in_afternoon_l298_29875

-- Conditions: Lino picked up 292 shells in the morning and 616 shells in total.
def shells_in_morning : ℕ := 292
def total_shells : ℕ := 616

-- Theorem: The number of shells Lino picked up in the afternoon is 324.
theorem shells_in_afternoon : (total_shells - shells_in_morning) = 324 := 
by sorry

end NUMINAMATH_GPT_shells_in_afternoon_l298_29875


namespace NUMINAMATH_GPT_find_tangent_circle_l298_29867

-- Define circles and line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def line_l (x y : ℝ) : Prop := x + 2*y = 0

-- Define the problem statement as a theorem
theorem find_tangent_circle :
  ∃ (x0 y0 : ℝ), (x - x0)^2 + (y - y0)^2 = 5/4 ∧ (x0, y0) = (1/2, 1) ∧
                   ∀ (x y : ℝ), (circle1 x y → circle2 x y → line_l (x0 + x) (y0 + y) ) :=
sorry

end NUMINAMATH_GPT_find_tangent_circle_l298_29867


namespace NUMINAMATH_GPT_finite_perfect_squares_l298_29809

noncomputable def finite_squares (a b : ℕ) : Prop :=
  ∃ (f : Finset ℕ), ∀ n, n ∈ f ↔ 
    ∃ (x y : ℕ), a * n ^ 2 + b = x ^ 2 ∧ a * (n + 1) ^ 2 + b = y ^ 2

theorem finite_perfect_squares (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  finite_squares a b :=
sorry

end NUMINAMATH_GPT_finite_perfect_squares_l298_29809


namespace NUMINAMATH_GPT_machines_make_2550_copies_l298_29818

def total_copies (rate1 rate2 : ℕ) (time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

theorem machines_make_2550_copies :
  total_copies 30 55 30 = 2550 :=
by
  unfold total_copies
  decide

end NUMINAMATH_GPT_machines_make_2550_copies_l298_29818


namespace NUMINAMATH_GPT_tank_capacity_is_32_l298_29873

noncomputable def capacity_of_tank (C : ℝ) : Prop :=
  (3/4) * C + 4 = (7/8) * C

theorem tank_capacity_is_32 : ∃ C : ℝ, capacity_of_tank C ∧ C = 32 :=
sorry

end NUMINAMATH_GPT_tank_capacity_is_32_l298_29873


namespace NUMINAMATH_GPT_trapezium_area_example_l298_29811

noncomputable def trapezium_area (a b h : ℝ) : ℝ := 1/2 * (a + b) * h

theorem trapezium_area_example :
  trapezium_area 20 18 16 = 304 :=
by
  -- The proof steps would go here, but we're skipping them.
  sorry

end NUMINAMATH_GPT_trapezium_area_example_l298_29811


namespace NUMINAMATH_GPT_sin_600_eq_neg_sqrt_3_div_2_l298_29842

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * (Real.pi / 180)) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_600_eq_neg_sqrt_3_div_2_l298_29842


namespace NUMINAMATH_GPT_largest_common_term_up_to_150_l298_29894

theorem largest_common_term_up_to_150 :
  ∃ a : ℕ, a ≤ 150 ∧ (∃ n : ℕ, a = 2 + 8 * n) ∧ (∃ m : ℕ, a = 3 + 9 * m) ∧ (∀ b : ℕ, b ≤ 150 → (∃ n' : ℕ, b = 2 + 8 * n') → (∃ m' : ℕ, b = 3 + 9 * m') → b ≤ a) := 
sorry

end NUMINAMATH_GPT_largest_common_term_up_to_150_l298_29894


namespace NUMINAMATH_GPT_floor_div_eq_floor_floor_div_l298_29870

theorem floor_div_eq_floor_floor_div (α : ℝ) (d : ℕ) (hα : 0 < α) :
  ⌊α / d⌋ = ⌊⌊α⌋ / d⌋ :=
by sorry

end NUMINAMATH_GPT_floor_div_eq_floor_floor_div_l298_29870


namespace NUMINAMATH_GPT_fiftieth_term_arithmetic_seq_l298_29878

theorem fiftieth_term_arithmetic_seq : 
  (∀ (n : ℕ), (2 + (n - 1) * 5) = 247) := by
  sorry

end NUMINAMATH_GPT_fiftieth_term_arithmetic_seq_l298_29878


namespace NUMINAMATH_GPT_consecutive_negative_integers_product_sum_l298_29879

theorem consecutive_negative_integers_product_sum (n : ℤ) 
  (h_neg1 : n < 0) 
  (h_neg2 : n + 1 < 0) 
  (h_product : n * (n + 1) = 2720) :
  n + (n + 1) = -105 :=
sorry

end NUMINAMATH_GPT_consecutive_negative_integers_product_sum_l298_29879


namespace NUMINAMATH_GPT_theater_ticket_difference_l298_29860

theorem theater_ticket_difference
  (O B V : ℕ) 
  (h₁ : O + B + V = 550) 
  (h₂ : 15 * O + 10 * B + 20 * V = 8000) : 
  B - (O + V) = 370 := 
sorry

end NUMINAMATH_GPT_theater_ticket_difference_l298_29860


namespace NUMINAMATH_GPT_basketball_game_score_difference_l298_29859

theorem basketball_game_score_difference :
  let blueFreeThrows := 18
  let blueTwoPointers := 25
  let blueThreePointers := 6
  let redFreeThrows := 15
  let redTwoPointers := 22
  let redThreePointers := 5
  let blueScore := blueFreeThrows * 1 + blueTwoPointers * 2 + blueThreePointers * 3
  let redScore := redFreeThrows * 1 + redTwoPointers * 2 + redThreePointers * 3
  blueScore - redScore = 12 := by
  sorry

end NUMINAMATH_GPT_basketball_game_score_difference_l298_29859


namespace NUMINAMATH_GPT_complement_union_l298_29866

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_union :
  U \ (M ∪ N) = {4} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l298_29866


namespace NUMINAMATH_GPT_M_eq_N_l298_29820

-- Define the sets M and N
def M : Set ℤ := {u | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l}
def N : Set ℤ := {u | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r}

-- Prove that M equals N
theorem M_eq_N : M = N := 
by {
  sorry
}

end NUMINAMATH_GPT_M_eq_N_l298_29820


namespace NUMINAMATH_GPT_min_value_of_f_l298_29872

noncomputable def f (x : ℝ) := max (3 - x) (x^2 - 4*x + 3)

theorem min_value_of_f : ∃ x : ℝ, f x = -1 :=
by {
  use 2,
  sorry
}

end NUMINAMATH_GPT_min_value_of_f_l298_29872


namespace NUMINAMATH_GPT_distinct_numbers_on_board_l298_29851

def count_distinct_numbers (Mila_divisors : ℕ) (Zhenya_divisors : ℕ) (common : ℕ) : ℕ :=
  Mila_divisors + Zhenya_divisors - (common - 1)

theorem distinct_numbers_on_board :
  count_distinct_numbers 10 9 2 = 13 := by
  sorry

end NUMINAMATH_GPT_distinct_numbers_on_board_l298_29851


namespace NUMINAMATH_GPT_eddie_weekly_earnings_l298_29821

theorem eddie_weekly_earnings :
  let mon_hours := 2.5
  let tue_hours := 7 / 6
  let wed_hours := 7 / 4
  let sat_hours := 3 / 4
  let weekday_rate := 4
  let saturday_rate := 6
  let mon_earnings := mon_hours * weekday_rate
  let tue_earnings := tue_hours * weekday_rate
  let wed_earnings := wed_hours * weekday_rate
  let sat_earnings := sat_hours * saturday_rate
  let total_earnings := mon_earnings + tue_earnings + wed_earnings + sat_earnings
  total_earnings = 26.17 := by
  simp only
  norm_num
  sorry

end NUMINAMATH_GPT_eddie_weekly_earnings_l298_29821


namespace NUMINAMATH_GPT_triangle_type_and_area_l298_29844

theorem triangle_type_and_area (x : ℝ) (hpos : 0 < x) (h : 3 * x + 4 * x + 5 * x = 36) :
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x
  a^2 + b^2 = c^2 ∧ (1 / 2) * a * b = 54 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_type_and_area_l298_29844


namespace NUMINAMATH_GPT_range_of_b_l298_29838

-- Define the conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16
def line_eq (x y b : ℝ) : Prop := y = x + b
def distance_point_line_eq (x y b d : ℝ) : Prop := 
  d = abs (b) / (Real.sqrt 2)
def at_least_three_points_on_circle_at_distance_one (b : ℝ) : Prop := 
  ∃ p1 p2 p3 : ℝ × ℝ, circle_eq p1.1 p1.2 ∧ circle_eq p2.1 p2.2 ∧ circle_eq p3.1 p3.2 ∧ 
  distance_point_line_eq p1.1 p1.2 b 1 ∧ distance_point_line_eq p2.1 p2.2 b 1 ∧ distance_point_line_eq p3.1 p3.2 b 1

-- The theorem statement to prove
theorem range_of_b (b : ℝ) (h : at_least_three_points_on_circle_at_distance_one b) : 
  -3 * Real.sqrt 2 ≤ b ∧ b ≤ 3 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_range_of_b_l298_29838


namespace NUMINAMATH_GPT_min_value_max_value_l298_29834

theorem min_value (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 11) (h2 : b^2 + b * c + c^2 = 11) : 
  (∃ v, v = c^2 + c * a + a^2 ∧ v = 0) := sorry

theorem max_value (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 11) (h2 : b^2 + b * c + c^2 = 11) : 
  (∃ v, v = c^2 + c * a + a^2 ∧ v = 44) := sorry

end NUMINAMATH_GPT_min_value_max_value_l298_29834


namespace NUMINAMATH_GPT_isosceles_triangle_base_l298_29849

theorem isosceles_triangle_base (a b c : ℕ) (h_isosceles : a = b ∨ a = c ∨ b = c)
  (h_perimeter : a + b + c = 29) (h_side : a = 7 ∨ b = 7 ∨ c = 7) : 
  a = 7 ∨ b = 7 ∨ c = 7 ∧ (a = 7 ∨ a = 11) ∧ (b = 7 ∨ b = 11) ∧ (c = 7 ∨ c = 11) ∧ (a ≠ b ∨ c ≠ b) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_l298_29849


namespace NUMINAMATH_GPT_yard_flower_beds_fraction_l298_29898

theorem yard_flower_beds_fraction :
  let yard_length := 30
  let yard_width := 10
  let pool_length := 10
  let pool_width := 4
  let trap_parallel_diff := 22 - 16
  let triangle_leg := trap_parallel_diff / 2
  let triangle_area := (1 / 2) * (triangle_leg ^ 2)
  let total_triangle_area := 2 * triangle_area
  let total_yard_area := yard_length * yard_width
  let pool_area := pool_length * pool_width
  let usable_yard_area := total_yard_area - pool_area
  (total_triangle_area / usable_yard_area) = 9 / 260 :=
by 
  sorry

end NUMINAMATH_GPT_yard_flower_beds_fraction_l298_29898


namespace NUMINAMATH_GPT_angle_Z_of_triangle_l298_29880

theorem angle_Z_of_triangle (X Y Z : ℝ) (h1 : X + Y = 90) (h2 : X + Y + Z = 180) : 
  Z = 90 := 
sorry

end NUMINAMATH_GPT_angle_Z_of_triangle_l298_29880


namespace NUMINAMATH_GPT_team_a_score_l298_29857

theorem team_a_score : ∀ (A : ℕ), A + 9 + 4 = 15 → A = 2 :=
by
  intros A h
  sorry

end NUMINAMATH_GPT_team_a_score_l298_29857


namespace NUMINAMATH_GPT_total_cats_and_kittens_received_l298_29822

theorem total_cats_and_kittens_received 
  (adult_cats : ℕ) 
  (perc_female : ℕ) 
  (frac_litters : ℚ) 
  (kittens_per_litter : ℕ)
  (rescued_cats : ℕ) 
  (total_received : ℕ)
  (h1 : adult_cats = 120)
  (h2 : perc_female = 60)
  (h3 : frac_litters = 2/3)
  (h4 : kittens_per_litter = 3)
  (h5 : rescued_cats = 30)
  (h6 : total_received = 294) :
  adult_cats + rescued_cats + (frac_litters * (perc_female * adult_cats / 100) * kittens_per_litter) = total_received := 
sorry

end NUMINAMATH_GPT_total_cats_and_kittens_received_l298_29822


namespace NUMINAMATH_GPT_range_of_a_l298_29865

def f (x : ℝ) : ℝ := -x^5 - 3 * x^3 - 5 * x + 3

theorem range_of_a (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 :=
by
  -- Here, we would have to show the proof, but we're skipping it
  sorry

end NUMINAMATH_GPT_range_of_a_l298_29865


namespace NUMINAMATH_GPT_abs_diff_of_numbers_l298_29840

theorem abs_diff_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_of_numbers_l298_29840


namespace NUMINAMATH_GPT_value_of_x_l298_29812

def is_whole_number (n : ℝ) : Prop := ∃ (k : ℤ), n = k

theorem value_of_x (n : ℝ) (x : ℝ) :
  n = 1728 →
  is_whole_number (Real.log n / Real.log x + Real.log n / Real.log 12) →
  x = 12 :=
by
  intro h₁ h₂
  sorry

end NUMINAMATH_GPT_value_of_x_l298_29812


namespace NUMINAMATH_GPT_evaluate_expression_l298_29853

theorem evaluate_expression (x y : ℕ) (hx : 2^x ∣ 360 ∧ ¬ 2^(x+1) ∣ 360) (hy : 3^y ∣ 360 ∧ ¬ 3^(y+1) ∣ 360) :
  (3 / 7)^(y - x) = 7 / 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l298_29853


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l298_29884

theorem arithmetic_sequence_ratio (a d : ℕ) (h : b = a + 3 * d) : a = 1 -> d = 1 -> (a / b = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l298_29884


namespace NUMINAMATH_GPT_negation_of_exactly_one_is_even_l298_29807

def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_is_even (a b c : ℕ) : Prop :=
  ((is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
   (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
   (¬ is_even a ∧ ¬ is_even b ∧ is_even c))

def at_least_two_even (a b c : ℕ) : Prop :=
  ((is_even a ∧ is_even b) ∨ (is_even b ∧ is_even c) ∨ (is_even a ∧ is_even c))

def all_are_odd (a b c : ℕ) : Prop := ¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c 

theorem negation_of_exactly_one_is_even (a b c : ℕ) :
  ¬ exactly_one_is_even a b c ↔ at_least_two_even a b c ∨ all_are_odd a b c := by
  sorry

end NUMINAMATH_GPT_negation_of_exactly_one_is_even_l298_29807


namespace NUMINAMATH_GPT_teal_bakery_revenue_l298_29890

theorem teal_bakery_revenue :
    let pumpkin_pies := 4
    let pumpkin_pie_slices := 8
    let pumpkin_slice_price := 5
    let custard_pies := 5
    let custard_pie_slices := 6
    let custard_slice_price := 6
    let total_pumpkin_slices := pumpkin_pies * pumpkin_pie_slices
    let total_custard_slices := custard_pies * custard_pie_slices
    let pumpkin_revenue := total_pumpkin_slices * pumpkin_slice_price
    let custard_revenue := total_custard_slices * custard_slice_price
    let total_revenue := pumpkin_revenue + custard_revenue
    total_revenue = 340 :=
by
  sorry

end NUMINAMATH_GPT_teal_bakery_revenue_l298_29890


namespace NUMINAMATH_GPT_six_star_three_l298_29848

def binary_op (x y : ℕ) : ℕ := 4 * x + 5 * y - x * y

theorem six_star_three : binary_op 6 3 = 21 := by
  sorry

end NUMINAMATH_GPT_six_star_three_l298_29848


namespace NUMINAMATH_GPT_usable_area_is_correct_l298_29886

variable (x : ℝ)

def total_field_area : ℝ := (x + 9) * (x + 7)
def flooded_area : ℝ := (2 * x - 2) * (x - 1)
def usable_area : ℝ := total_field_area x - flooded_area x

theorem usable_area_is_correct : usable_area x = -x^2 + 20 * x + 61 :=
by
  sorry

end NUMINAMATH_GPT_usable_area_is_correct_l298_29886


namespace NUMINAMATH_GPT_matrix_exponentiation_l298_29846

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -2],
    ![2, -1]]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![-4, 6],
    ![-6, 5]]

theorem matrix_exponentiation :
  A^4 = B :=
by
  sorry

end NUMINAMATH_GPT_matrix_exponentiation_l298_29846


namespace NUMINAMATH_GPT_verify_compound_interest_rate_l298_29823

noncomputable def compound_interest_rate
  (P A : ℝ) (t n : ℕ) : ℝ :=
  let r := (A / P) ^ (1 / (n * t)) - 1
  n * r

theorem verify_compound_interest_rate :
  let P := 5000
  let A := 6800
  let t := 4
  let n := 1
  compound_interest_rate P A t n = 8.02 / 100 :=
by
  sorry

end NUMINAMATH_GPT_verify_compound_interest_rate_l298_29823


namespace NUMINAMATH_GPT_arsh_eq_arch_pos_eq_arch_neg_eq_arth_eq_l298_29850

noncomputable def arsh (x : ℝ) := Real.log (x + Real.sqrt (x^2 + 1))
noncomputable def arch_pos (x : ℝ) := Real.log (x + Real.sqrt (x^2 - 1))
noncomputable def arch_neg (x : ℝ) := Real.log (x - Real.sqrt (x^2 - 1))
noncomputable def arth (x : ℝ) := (1 / 2) * Real.log ((1 + x) / (1 - x))

theorem arsh_eq (x : ℝ) : arsh x = Real.log (x + Real.sqrt (x^2 + 1)) := by
  sorry

theorem arch_pos_eq (x : ℝ) : arch_pos x = Real.log (x + Real.sqrt (x^2 - 1)) := by
  sorry

theorem arch_neg_eq (x : ℝ) : arch_neg x = Real.log (x - Real.sqrt (x^2 - 1)) := by
  sorry

theorem arth_eq (x : ℝ) : arth x = (1 / 2) * Real.log ((1 + x) / (1 - x)) := by
  sorry

end NUMINAMATH_GPT_arsh_eq_arch_pos_eq_arch_neg_eq_arth_eq_l298_29850


namespace NUMINAMATH_GPT_problem_lean_l298_29889

theorem problem_lean (a : ℝ) (h : a - 1/a = 5) : a^2 + 1/a^2 = 27 := by
  sorry

end NUMINAMATH_GPT_problem_lean_l298_29889


namespace NUMINAMATH_GPT_july_percentage_is_correct_l298_29836

def total_scientists : ℕ := 120
def july_scientists : ℕ := 16
def july_percentage : ℚ := (july_scientists : ℚ) / (total_scientists : ℚ) * 100

theorem july_percentage_is_correct : july_percentage = 13.33 := 
by 
  -- Provides the proof directly as a statement
  sorry

end NUMINAMATH_GPT_july_percentage_is_correct_l298_29836


namespace NUMINAMATH_GPT_intersect_lines_l298_29869

theorem intersect_lines (k : ℝ) :
  (∃ y : ℝ, 3 * 5 - y = k ∧ -5 - y = -10) → k = 10 :=
by
  sorry

end NUMINAMATH_GPT_intersect_lines_l298_29869


namespace NUMINAMATH_GPT_num_dress_designs_l298_29803

-- Define the number of fabric colors and patterns
def fabric_colors : ℕ := 4
def patterns : ℕ := 5

-- Define the number of possible dress designs
def total_dress_designs : ℕ := fabric_colors * patterns

-- State the theorem that needs to be proved
theorem num_dress_designs : total_dress_designs = 20 := by
  sorry

end NUMINAMATH_GPT_num_dress_designs_l298_29803


namespace NUMINAMATH_GPT_time_for_A_and_C_l298_29837

variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := A + B = 1 / 8
def condition2 : Prop := B + C = 1 / 12
def condition3 : Prop := A + B + C = 1 / 6

theorem time_for_A_and_C (h1 : condition1 A B)
                        (h2 : condition2 B C)
                        (h3 : condition3 A B C) :
  1 / (A + C) = 8 :=
sorry

end NUMINAMATH_GPT_time_for_A_and_C_l298_29837


namespace NUMINAMATH_GPT_proposition_does_not_hold_6_l298_29847

-- Define P as a proposition over positive integers
variable (P : ℕ → Prop)

-- Assumptions
variables (h1 : ∀ k : ℕ, P k → P (k + 1))  
variable (h2 : ¬ P 7)

-- Statement of the Problem
theorem proposition_does_not_hold_6 : ¬ P 6 :=
sorry

end NUMINAMATH_GPT_proposition_does_not_hold_6_l298_29847


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l298_29806

variable (a b c : ℝ)
variable (h1 : a = 4 ∨ a = 8)
variable (h2 : b = 4 ∨ b = 8)
variable (h3 : a = b ∨ c = 8)

theorem isosceles_triangle_perimeter (h : a + b + c = 20) : a = b ∨ b = 8 ∧ (a = 8 ∧ c = 4 ∨ b = c) := 
  by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l298_29806


namespace NUMINAMATH_GPT_fixed_point_of_function_l298_29831

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : (2, 3) ∈ { (x, y) | y = 2 + a^(x-2) } :=
sorry

end NUMINAMATH_GPT_fixed_point_of_function_l298_29831


namespace NUMINAMATH_GPT_cos_pi_minus_alpha_cos_double_alpha_l298_29881

open Real

theorem cos_pi_minus_alpha (α : ℝ) (h1 : sin α = sqrt 2 / 3) (h2 : 0 < α ∧ α < π / 2) :
  cos (π - α) = - sqrt 7 / 3 :=
by
  sorry

theorem cos_double_alpha (α : ℝ) (h1 : sin α = sqrt 2 / 3) (h2 : 0 < α ∧ α < π / 2) :
  cos (2 * α) = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_alpha_cos_double_alpha_l298_29881


namespace NUMINAMATH_GPT_strawberries_harvest_l298_29892

theorem strawberries_harvest (length : ℕ) (width : ℕ) 
  (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) 
  (area := length * width) (total_plants := plants_per_sqft * area) 
  (total_strawberries := strawberries_per_plant * total_plants) :
  length = 10 → width = 9 →
  plants_per_sqft = 5 → strawberries_per_plant = 8 →
  total_strawberries = 3600 := by
  sorry

end NUMINAMATH_GPT_strawberries_harvest_l298_29892


namespace NUMINAMATH_GPT_slant_heights_of_cones_l298_29868

-- Define the initial conditions
variables (r r1 x y : Real)

-- Define the surface area condition
def surface_area_condition : Prop :=
  r * Real.sqrt (r ^ 2 + x ^ 2) + r ^ 2 = r1 * Real.sqrt (r1 ^ 2 + y ^ 2) + r1 ^ 2

-- Define the volume condition
def volume_condition : Prop :=
  r ^ 2 * Real.sqrt (x ^ 2 - r ^ 2) = r1 ^ 2 * Real.sqrt (y ^ 2 - r1 ^ 2)

-- Statement of the proof problem: Prove that the slant heights x and y are given by
theorem slant_heights_of_cones
  (h1 : surface_area_condition r r1 x y)
  (h2 : volume_condition r r1 x y) :
  x = (r ^ 2 + 2 * r1 ^ 2) / r ∧ y = (r1 ^ 2 + 2 * r ^ 2) / r1 := 
  sorry

end NUMINAMATH_GPT_slant_heights_of_cones_l298_29868


namespace NUMINAMATH_GPT_red_balls_count_l298_29802

theorem red_balls_count (w r : ℕ) (h1 : w = 16) (h2 : 4 * r = 3 * w) : r = 12 :=
by
  sorry

end NUMINAMATH_GPT_red_balls_count_l298_29802


namespace NUMINAMATH_GPT_area_of_rectangle_PQRS_l298_29893

-- Definitions for the lengths of the sides of triangle ABC.
def AB : ℝ := 15
def AC : ℝ := 20
def BC : ℝ := 25

-- Definition for the length of PQ in rectangle PQRS.
def PQ : ℝ := 12

-- Definition for the condition that PQ is parallel to BC and RS is parallel to AB.
def PQ_parallel_BC : Prop := True
def RS_parallel_AB : Prop := True

-- The theorem to be proved: the area of rectangle PQRS is 115.2.
theorem area_of_rectangle_PQRS : 
  (∃ h: ℝ, h = (AC * PQ / BC) ∧ PQ * h = 115.2) :=
by {
  sorry
}

end NUMINAMATH_GPT_area_of_rectangle_PQRS_l298_29893


namespace NUMINAMATH_GPT_function_domain_length_correct_l298_29808

noncomputable def function_domain_length : ℕ :=
  let p : ℕ := 240 
  let q : ℕ := 1
  p + q

theorem function_domain_length_correct : function_domain_length = 241 := by
  sorry

end NUMINAMATH_GPT_function_domain_length_correct_l298_29808


namespace NUMINAMATH_GPT_prob_of_three_successes_correct_l298_29862

noncomputable def prob_of_three_successes (p : ℝ) : ℝ :=
  (Nat.choose 10 3) * (p^3) * (1-p)^7

theorem prob_of_three_successes_correct (p : ℝ) :
  prob_of_three_successes p = (Nat.choose 10 3 : ℝ) * (p^3) * (1-p)^7 :=
by
  sorry

end NUMINAMATH_GPT_prob_of_three_successes_correct_l298_29862


namespace NUMINAMATH_GPT_evaluate_complex_fraction_l298_29895

theorem evaluate_complex_fraction : 
  (1 / (2 + (1 / (3 + 1 / 4)))) = (13 / 30) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_complex_fraction_l298_29895


namespace NUMINAMATH_GPT_log_equation_solution_l298_29854

theorem log_equation_solution {x : ℝ} (hx : x > 0) (hx1 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by 
  sorry

end NUMINAMATH_GPT_log_equation_solution_l298_29854


namespace NUMINAMATH_GPT_find_first_term_and_ratio_l298_29832

variable (b1 q : ℝ)

-- Conditions
def infinite_geometric_series (q : ℝ) : Prop := |q| < 1

def sum_odd_even_difference (b1 q : ℝ) : Prop := 
  b1 / (1 - q^2) = 2 + (b1 * q) / (1 - q^2)

def sum_square_odd_even_difference (b1 q : ℝ) : Prop :=
  b1^2 / (1 - q^4) - (b1^2 * q^2) / (1 - q^4) = 36 / 5

-- Proof problem
theorem find_first_term_and_ratio (b1 q : ℝ) 
  (h1 : infinite_geometric_series q) 
  (h2 : sum_odd_even_difference b1 q)
  (h3 : sum_square_odd_even_difference b1 q) : 
  b1 = 3 ∧ q = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_find_first_term_and_ratio_l298_29832


namespace NUMINAMATH_GPT_julia_monday_kids_l298_29817

theorem julia_monday_kids (x : ℕ) (h1 : x + 14 = 16) : x = 2 := 
by
  sorry

end NUMINAMATH_GPT_julia_monday_kids_l298_29817


namespace NUMINAMATH_GPT_reciprocal_of_fraction_l298_29899

noncomputable def fraction := (Real.sqrt 5 + 1) / 2

theorem reciprocal_of_fraction :
  (fraction⁻¹) = (Real.sqrt 5 - 1) / 2 :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_reciprocal_of_fraction_l298_29899
