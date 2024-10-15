import Mathlib

namespace NUMINAMATH_GPT_conic_sections_with_foci_at_F2_zero_l1552_155268

theorem conic_sections_with_foci_at_F2_zero (a b m n: ℝ) (h1 : a > b) (h2: b > 0) (h3: m > 0) (h4: n > 0) (h5: a^2 - b^2 = 4) (h6: m^2 + n^2 = 4):
  (∀ x y: ℝ, x^2 / (a^2) + y^2 / (b^2) = 1) ∧ (∀ x y: ℝ, x^2 / (11/60) + y^2 / (11/16) = 1) ∧ 
  ∀ x y: ℝ, x^2 / (m^2) - y^2 / (n^2) = 1 ∧ ∀ x y: ℝ, 5*x^2 / 4 - 5*y^2 / 16 = 1 := 
sorry

end NUMINAMATH_GPT_conic_sections_with_foci_at_F2_zero_l1552_155268


namespace NUMINAMATH_GPT_cos_sq_minus_sin_sq_l1552_155203

variable (α β : ℝ)

theorem cos_sq_minus_sin_sq (h : Real.cos (α + β) * Real.cos (α - β) = 1 / 3) :
  Real.cos α ^ 2 - Real.sin β ^ 2 = 1 / 3 :=
sorry

end NUMINAMATH_GPT_cos_sq_minus_sin_sq_l1552_155203


namespace NUMINAMATH_GPT_double_root_divisors_l1552_155264

theorem double_root_divisors (b3 b2 b1 s : ℤ) (h : 0 = (s^2) • (x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 50)) : 
  s = -5 ∨ s = -1 ∨ s = 1 ∨ s = 5 :=
by
  sorry

end NUMINAMATH_GPT_double_root_divisors_l1552_155264


namespace NUMINAMATH_GPT_richard_older_than_david_l1552_155215

variable {R D S : ℕ}

theorem richard_older_than_david (h1 : R > D) (h2 : D = S + 8) (h3 : R + 8 = 2 * (S + 8)) (h4 : D = 14) : R - D = 6 := by
  sorry

end NUMINAMATH_GPT_richard_older_than_david_l1552_155215


namespace NUMINAMATH_GPT_hyperbola_k_range_l1552_155210

theorem hyperbola_k_range {k : ℝ} 
  (h : ∀ x y : ℝ, x^2 + (k-1)*y^2 = k+1 → (k > -1 ∧ k < 1)) : 
  -1 < k ∧ k < 1 :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_k_range_l1552_155210


namespace NUMINAMATH_GPT_interest_rate_second_part_l1552_155211

theorem interest_rate_second_part (P1 P2: ℝ) (total_sum : ℝ) (rate1 : ℝ) (time1 : ℝ) (time2 : ℝ) (interest_second_part: ℝ ) : 
  total_sum = 2717 → P2 = 1672 → time1 = 8 → rate1 = 3 → time2 = 3 →
  P1 + P2 = total_sum →
  P1 * rate1 * time1 / 100 = P2 * interest_second_part * time2 / 100 →
  interest_second_part = 5 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_second_part_l1552_155211


namespace NUMINAMATH_GPT_parrots_per_cage_l1552_155201

-- Definitions of the given conditions
def num_cages : ℕ := 6
def num_parakeets_per_cage : ℕ := 7
def total_birds : ℕ := 54

-- Proposition stating the question and the correct answer
theorem parrots_per_cage : (total_birds - num_cages * num_parakeets_per_cage) / num_cages = 2 := 
by
  sorry

end NUMINAMATH_GPT_parrots_per_cage_l1552_155201


namespace NUMINAMATH_GPT_fixed_point_of_parabola_l1552_155236

theorem fixed_point_of_parabola :
  ∀ (m : ℝ), ∃ (a b : ℝ), (∀ (x : ℝ), (a = -3 ∧ b = 81) → (y = 9*x^2 + m*x + 3*m) → (y = 81)) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_parabola_l1552_155236


namespace NUMINAMATH_GPT_option_b_has_two_distinct_real_roots_l1552_155206

def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  let Δ := b^2 - 4 * a * c
  Δ > 0

theorem option_b_has_two_distinct_real_roots :
  has_two_distinct_real_roots 1 (-2) (-3) :=
by
  sorry

end NUMINAMATH_GPT_option_b_has_two_distinct_real_roots_l1552_155206


namespace NUMINAMATH_GPT_transform_fraction_l1552_155289

theorem transform_fraction (x : ℝ) (h : x ≠ 1) : - (1 / (1 - x)) = 1 / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_transform_fraction_l1552_155289


namespace NUMINAMATH_GPT_jackson_points_l1552_155245

theorem jackson_points (team_total_points : ℕ)
                       (num_other_players : ℕ)
                       (average_points_other_players : ℕ)
                       (points_other_players: ℕ)
                       (points_jackson: ℕ)
                       (h_team_total_points : team_total_points = 65)
                       (h_num_other_players : num_other_players = 5)
                       (h_average_points_other_players : average_points_other_players = 6)
                       (h_points_other_players : points_other_players = num_other_players * average_points_other_players)
                       (h_points_total: points_jackson + points_other_players = team_total_points) :
  points_jackson = 35 :=
by
  -- proof will be done here
  sorry

end NUMINAMATH_GPT_jackson_points_l1552_155245


namespace NUMINAMATH_GPT_median_eq_range_le_l1552_155214

def sample_data (x : ℕ → ℝ) :=
  x 1 ≤ x 2 ∧ x 2 ≤ x 3 ∧ x 3 ≤ x 4 ∧ x 4 ≤ x 5 ∧ x 5 ≤ x 6

theorem median_eq_range_le
  (x : ℕ → ℝ) 
  (h_sample_data : sample_data x) :
  ((x 3 + x 4) / 2 = (x 3 + x 4) / 2) ∧ (x 5 - x 2 ≤ x 6 - x 1) :=
by
  sorry

end NUMINAMATH_GPT_median_eq_range_le_l1552_155214


namespace NUMINAMATH_GPT_final_l1552_155209

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ∈ [-3, -2] then 4 * x
  else sorry

lemma f_periodic (h : ∀ x : ℝ, f (x + 3) = - (1 / f x)) :
 ∀ x : ℝ, f (x + 6) = f x :=
sorry

lemma f_even (h : ∀ x : ℝ, f x = f (-x)) : ℕ := sorry

theorem final (h1 : ∀ x : ℝ, f (x + 3) = - (1 / f x))
  (h2 : ∀ x : ℝ, f x = f (-x))
  (h3 : ∀ x : ℝ, x ∈ [-3, -2] → f x = 4 * x) :
  f 107.5 = 1 / 10 :=
sorry

end NUMINAMATH_GPT_final_l1552_155209


namespace NUMINAMATH_GPT_solve_equation_l1552_155226

theorem solve_equation :
  ∃ y : ℚ, 2 * (y - 3) - 6 * (2 * y - 1) = -3 * (2 - 5 * y) ↔ y = 6 / 25 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1552_155226


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1552_155270

theorem relationship_among_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = (4 : ℝ) ^ (1 / 2))
  (hb : b = (2 : ℝ) ^ (1 / 3))
  (hc : c = (5 : ℝ) ^ (1 / 2))
: b < a ∧ a < c := 
sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1552_155270


namespace NUMINAMATH_GPT_dan_has_3_potatoes_left_l1552_155295

-- Defining the number of potatoes Dan originally had
def original_potatoes : ℕ := 7

-- Defining the number of potatoes the rabbits ate
def potatoes_eaten : ℕ := 4

-- The theorem we want to prove: Dan has 3 potatoes left.
theorem dan_has_3_potatoes_left : original_potatoes - potatoes_eaten = 3 := by
  sorry

end NUMINAMATH_GPT_dan_has_3_potatoes_left_l1552_155295


namespace NUMINAMATH_GPT_triangle_area_solution_l1552_155272

noncomputable def triangle_area_problem 
  (a b c : ℝ) (A B C : ℝ) (h1 : A = 3 * C)
  (h2 : c = 6)
  (h3 : (2 * a - c) * Real.cos B - b * Real.cos C = 0)
  : ℝ := (1 / 2) * a * c * Real.sin B

theorem triangle_area_solution 
  (a b c : ℝ) (A B C : ℝ) (h1 : A = 3 * C)
  (h2 : c = 6)
  (h3 : (2 * a - c) * Real.cos B - b * Real.cos C = 0)
  (ha : a = 12)
  (hb : b = 6 * Real.sin (π / 3))
  (hA : A = π / 2)
  (hB : B = π / 3)
  (hC : C = π / 6) 
  : triangle_area_problem a b c A B C h1 h2 h3 = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_triangle_area_solution_l1552_155272


namespace NUMINAMATH_GPT_slant_asymptote_and_sum_of_slope_and_intercept_l1552_155248

noncomputable def f (x : ℚ) : ℚ := (3 * x^2 + 5 * x + 1) / (x + 2)

theorem slant_asymptote_and_sum_of_slope_and_intercept :
  (∀ x : ℚ, ∃ (m b : ℚ), (∃ r : ℚ, (r = f x ∧ (r + (m * x + b)) = f x)) ∧ m = 3 ∧ b = -1) →
  3 - 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_slant_asymptote_and_sum_of_slope_and_intercept_l1552_155248


namespace NUMINAMATH_GPT_fraction_remain_unchanged_l1552_155256

theorem fraction_remain_unchanged (m n a b : ℚ) (h : n ≠ 0 ∧ b ≠ 0) : 
  (a / b = (a + m) / (b + n)) ↔ (a / b = m / n) :=
sorry

end NUMINAMATH_GPT_fraction_remain_unchanged_l1552_155256


namespace NUMINAMATH_GPT_ferry_speed_difference_l1552_155266

open Nat

-- Define the time and speed of ferry P
def timeP := 3 -- hours
def speedP := 8 -- kilometers per hour

-- Define the distance of ferry P
def distanceP := speedP * timeP -- kilometers

-- Define the distance of ferry Q
def distanceQ := 3 * distanceP -- kilometers

-- Define the time of ferry Q
def timeQ := timeP + 5 -- hours

-- Define the speed of ferry Q
def speedQ := distanceQ / timeQ -- kilometers per hour

-- Define the speed difference
def speedDifference := speedQ - speedP -- kilometers per hour

-- The target theorem to prove
theorem ferry_speed_difference : speedDifference = 1 := by
  sorry

end NUMINAMATH_GPT_ferry_speed_difference_l1552_155266


namespace NUMINAMATH_GPT_range_of_a_l1552_155285

def quadratic_function (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, 1 ≤ x ∧ x ≤ y → quadratic_function a x ≤ quadratic_function a y) : a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1552_155285


namespace NUMINAMATH_GPT_vector_identity_l1552_155276

def vec_a : ℝ × ℝ := (2, 2)
def vec_b : ℝ × ℝ := (-1, 3)

theorem vector_identity : 2 • vec_a - vec_b = (5, 1) := by
  sorry

end NUMINAMATH_GPT_vector_identity_l1552_155276


namespace NUMINAMATH_GPT_math_problem_l1552_155204

theorem math_problem (a b : ℝ) 
  (h1 : a^2 - 3*a*b + 2*b^2 + a - b = 0)
  (h2 : a^2 - 2*a*b + b^2 - 5*a + 7*b = 0) :
  a*b - 12*a + 15*b = 0 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1552_155204


namespace NUMINAMATH_GPT_right_triangle_area_l1552_155238

theorem right_triangle_area (a b : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (right_triangle : ∃ c : ℝ, c^2 = a^2 + b^2) : 
  ∃ A : ℝ, A = 1/2 * a * b ∧ A = 30 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1552_155238


namespace NUMINAMATH_GPT_base7_perfect_square_values_l1552_155232

theorem base7_perfect_square_values (a b c : ℕ) (h1 : a ≠ 0) (h2 : b < 7) :
  ∃ (n : ℕ), (343 * a + 49 * c + 28 + b = n * n) → (b = 0 ∨ b = 1 ∨ b = 4) :=
by
  sorry

end NUMINAMATH_GPT_base7_perfect_square_values_l1552_155232


namespace NUMINAMATH_GPT_compare_sqrt_terms_l1552_155200

/-- Compare the sizes of 5 * sqrt 2 and 3 * sqrt 3 -/
theorem compare_sqrt_terms : 5 * Real.sqrt 2 > 3 * Real.sqrt 3 := 
by sorry

end NUMINAMATH_GPT_compare_sqrt_terms_l1552_155200


namespace NUMINAMATH_GPT_find_k_for_parallel_vectors_l1552_155257

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem find_k_for_parallel_vectors 
  (h_a : a = (2, -1)) 
  (h_b : b = (1, 1)) 
  (h_c : c = (-5, 1)) 
  (h_parallel : vector_parallel (a.1 + k * b.1, a.2 + k * b.2) c) : 
  k = 1 / 2 :=
by
  unfold vector_parallel at h_parallel
  simp at h_parallel
  sorry

end NUMINAMATH_GPT_find_k_for_parallel_vectors_l1552_155257


namespace NUMINAMATH_GPT_solve_for_r_l1552_155244

theorem solve_for_r (r : ℝ) : 
  (r^2 - 3) / 3 = (5 - r) / 2 ↔ 
  r = (-3 + Real.sqrt 177) / 4 ∨ r = (-3 - Real.sqrt 177) / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_r_l1552_155244


namespace NUMINAMATH_GPT_paperback_copies_sold_l1552_155278

theorem paperback_copies_sold 
(H : ℕ)
(hardback_sold : H = 36000)
(P : ℕ)
(paperback_relation : P = 9 * H)
(total_copies : H + P = 440000) :
P = 324000 :=
sorry

end NUMINAMATH_GPT_paperback_copies_sold_l1552_155278


namespace NUMINAMATH_GPT_steel_parts_count_l1552_155274

-- Definitions for conditions
variables (a b : ℕ)

-- Conditions provided in the problem
axiom machines_count : a + b = 21
axiom chrome_parts : 2 * a + 4 * b = 66

-- Statement to prove
theorem steel_parts_count : 3 * a + 2 * b = 51 :=
by
  sorry

end NUMINAMATH_GPT_steel_parts_count_l1552_155274


namespace NUMINAMATH_GPT_circle_radius_through_focus_and_tangent_l1552_155296

-- Define the given conditions of the problem
def ellipse_eq (x y : ℝ) : Prop := x^2 + 4 * y^2 = 16

-- State the problem as a theorem
theorem circle_radius_through_focus_and_tangent
  (x y : ℝ) (h : ellipse_eq x y) (r : ℝ) :
  r = 4 - 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_circle_radius_through_focus_and_tangent_l1552_155296


namespace NUMINAMATH_GPT_tree_heights_l1552_155269

theorem tree_heights :
  let Tree_A := 150
  let Tree_B := (2/3 : ℝ) * Tree_A
  let Tree_C := (1/2 : ℝ) * Tree_B
  let Tree_D := Tree_C + 25
  let Tree_E := 0.40 * Tree_A
  let Tree_F := (Tree_B + Tree_D) / 2
  let Tree_G := (3/8 : ℝ) * Tree_A
  let Tree_H := 1.25 * Tree_F
  let Tree_I := 0.60 * (Tree_E + Tree_G)
  let total_height := Tree_A + Tree_B + Tree_C + Tree_D + Tree_E + Tree_F + Tree_G + Tree_H + Tree_I
  Tree_A = 150 ∧
  Tree_B = 100 ∧
  Tree_C = 50 ∧
  Tree_D = 75 ∧
  Tree_E = 60 ∧
  Tree_F = 87.5 ∧
  Tree_G = 56.25 ∧
  Tree_H = 109.375 ∧
  Tree_I = 69.75 ∧
  total_height = 758.125 :=
by
  sorry

end NUMINAMATH_GPT_tree_heights_l1552_155269


namespace NUMINAMATH_GPT_number_of_valid_rods_l1552_155225

theorem number_of_valid_rods : ∃ n, n = 22 ∧
  (∀ (d : ℕ), 1 < d ∧ d < 25 ∧ d ≠ 4 ∧ d ≠ 9 ∧ d ≠ 12 → d ∈ {d | d > 0}) :=
by
  use 22
  sorry

end NUMINAMATH_GPT_number_of_valid_rods_l1552_155225


namespace NUMINAMATH_GPT_number_of_owls_joined_l1552_155207

-- Define the initial condition
def initial_owls : ℕ := 3

-- Define the current condition
def current_owls : ℕ := 5

-- Define the problem statement as a theorem
theorem number_of_owls_joined : (current_owls - initial_owls) = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_owls_joined_l1552_155207


namespace NUMINAMATH_GPT_smallest_d_l1552_155222

noncomputable def smallestPositiveD : ℝ := 1

theorem smallest_d (d : ℝ) : 
  (0 < d) →
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → 
    (Real.sqrt (x * y) + d * (x^2 - y^2)^2 ≥ x + y)) →
  d ≥ smallestPositiveD :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_smallest_d_l1552_155222


namespace NUMINAMATH_GPT_ratio_of_birds_to_trees_and_stones_l1552_155283

theorem ratio_of_birds_to_trees_and_stones (stones birds : ℕ) (h_stones : stones = 40)
  (h_birds : birds = 400) (trees : ℕ) (h_trees : trees = 3 * stones + stones) :
  (birds : ℚ) / (trees + stones) = 2 :=
by
  -- The actual proof steps would go here.
  sorry

end NUMINAMATH_GPT_ratio_of_birds_to_trees_and_stones_l1552_155283


namespace NUMINAMATH_GPT_length_of_hypotenuse_l1552_155282

/-- Define the problem's parameters -/
def perimeter : ℝ := 34
def area : ℝ := 24
def length_hypotenuse (a b c : ℝ) : Prop := a + b + c = perimeter 
  ∧ (1/2) * a * b = area
  ∧ a^2 + b^2 = c^2

/- Lean statement for the proof problem -/
theorem length_of_hypotenuse (a b c : ℝ) 
  (h1: a + b + c = 34)
  (h2: (1/2) * a * b = 24)
  (h3: a^2 + b^2 = c^2)
  : c = 62 / 4 := sorry

end NUMINAMATH_GPT_length_of_hypotenuse_l1552_155282


namespace NUMINAMATH_GPT_six_digit_number_divisible_by_37_l1552_155208

theorem six_digit_number_divisible_by_37 (a b : ℕ) (h1 : 100 ≤ a ∧ a < 1000) (h2 : 100 ≤ b ∧ b < 1000) (h3 : 37 ∣ (a + b)) : 37 ∣ (1000 * a + b) :=
sorry

end NUMINAMATH_GPT_six_digit_number_divisible_by_37_l1552_155208


namespace NUMINAMATH_GPT_find_real_solutions_l1552_155260

noncomputable def cubic_eq_solutions (x : ℝ) : Prop := 
  x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3

theorem find_real_solutions : {x : ℝ | cubic_eq_solutions x} = {6} :=
by
  sorry

end NUMINAMATH_GPT_find_real_solutions_l1552_155260


namespace NUMINAMATH_GPT_nested_sqrt_simplification_l1552_155246

theorem nested_sqrt_simplification (y : ℝ) (hy : y ≥ 0) : 
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt y)) = y^(9/4) := 
sorry

end NUMINAMATH_GPT_nested_sqrt_simplification_l1552_155246


namespace NUMINAMATH_GPT_students_answered_both_correctly_l1552_155252

theorem students_answered_both_correctly 
  (total_students : ℕ) (took_test : ℕ) 
  (q1_correct : ℕ) (q2_correct : ℕ)
  (did_not_take_test : ℕ)
  (h1 : total_students = 25)
  (h2 : q1_correct = 22)
  (h3 : q2_correct = 20)
  (h4 : did_not_take_test = 3)
  (h5 : took_test = total_students - did_not_take_test) :
  (q1_correct + q2_correct) - took_test = 20 := 
by 
  -- Proof skipped.
  sorry

end NUMINAMATH_GPT_students_answered_both_correctly_l1552_155252


namespace NUMINAMATH_GPT_solve_system_of_equations_l1552_155247

theorem solve_system_of_equations (x y : ℝ) :
  (1 / 2 * x - 3 / 2 * y = -1) ∧ (2 * x + y = 3) → 
  (x = 1) ∧ (y = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1552_155247


namespace NUMINAMATH_GPT_club_co_presidents_l1552_155239

def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem club_co_presidents : choose 18 3 = 816 := by
  sorry

end NUMINAMATH_GPT_club_co_presidents_l1552_155239


namespace NUMINAMATH_GPT_mildred_initial_oranges_l1552_155219

theorem mildred_initial_oranges (final_oranges : ℕ) (added_oranges : ℕ) 
  (final_oranges_eq : final_oranges = 79) (added_oranges_eq : added_oranges = 2) : 
  final_oranges - added_oranges = 77 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_mildred_initial_oranges_l1552_155219


namespace NUMINAMATH_GPT_tree_height_at_end_of_4_years_l1552_155277

theorem tree_height_at_end_of_4_years 
  (initial_growth : ℕ → ℕ)
  (height_7_years : initial_growth 7 = 64)
  (growth_pattern : ∀ n, initial_growth (n + 1) = 2 * initial_growth n) :
  initial_growth 4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_tree_height_at_end_of_4_years_l1552_155277


namespace NUMINAMATH_GPT_garden_fencing_l1552_155249

theorem garden_fencing (length width : ℕ) (h1 : length = 80) (h2 : width = length / 2) : 2 * (length + width) = 240 :=
by
  sorry

end NUMINAMATH_GPT_garden_fencing_l1552_155249


namespace NUMINAMATH_GPT_find_a_l1552_155273

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.sin x

theorem find_a (a : ℝ) : (∀ f', f' = (fun x => a * Real.exp x - Real.cos x) → f' 0 = 0) → a = 1 :=
by
  intros h
  specialize h (fun x => a * Real.exp x - Real.cos x) rfl
  sorry  -- proof is omitted

end NUMINAMATH_GPT_find_a_l1552_155273


namespace NUMINAMATH_GPT_greatest_five_digit_number_sum_of_digits_l1552_155241

def is_five_digit_number (n : ℕ) : Prop :=
  10000 <= n ∧ n < 100000

def digits_product (n : ℕ) : ℕ :=
  (n % 10) * ((n / 10) % 10) * ((n / 100) % 10) * ((n / 1000) % 10) * (n / 10000)

def digits_sum (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10) + (n / 10000)

theorem greatest_five_digit_number_sum_of_digits (M : ℕ) 
  (h1 : is_five_digit_number M) 
  (h2 : digits_product M = 210) :
  digits_sum M = 20 := 
sorry

end NUMINAMATH_GPT_greatest_five_digit_number_sum_of_digits_l1552_155241


namespace NUMINAMATH_GPT_solve_inequality_l1552_155297

theorem solve_inequality (a x : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 1 > 0) : 
  (-2 < a ∧ a < 1 → a < x ∧ x < 2 - a) ∧ 
  (a = 1 → False) ∧ 
  (1 < a ∧ a < 2 → 2 - a < x ∧ x < a) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1552_155297


namespace NUMINAMATH_GPT_partial_fraction_sum_zero_l1552_155281

theorem partial_fraction_sum_zero (A B C D E F : ℚ) :
  (∀ x : ℚ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 → x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end NUMINAMATH_GPT_partial_fraction_sum_zero_l1552_155281


namespace NUMINAMATH_GPT_cd_cost_l1552_155220

theorem cd_cost (mp3_cost savings father_amt lacks cd_cost : ℝ) :
  mp3_cost = 120 ∧ savings = 55 ∧ father_amt = 20 ∧ lacks = 64 →
  120 + cd_cost - (savings + father_amt) = lacks → 
  cd_cost = 19 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cd_cost_l1552_155220


namespace NUMINAMATH_GPT_angle_of_inclination_l1552_155298

theorem angle_of_inclination 
  (α : ℝ) 
  (h_tan : Real.tan α = -Real.sqrt 3)
  (h_range : 0 ≤ α ∧ α < 180) : α = 120 :=
by
  sorry

end NUMINAMATH_GPT_angle_of_inclination_l1552_155298


namespace NUMINAMATH_GPT_unique_digit_solution_l1552_155243

-- Define the constraints as Lean predicates.
def sum_top_less_7 (A B C D E : ℕ) := A + B = (C + D + E) / 7
def sum_left_less_5 (A B C D E : ℕ) := A + C = (B + D + E) / 5

-- The main theorem statement asserting there is a unique solution.
theorem unique_digit_solution :
  ∃! (A B C D E : ℕ), 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 
  sum_top_less_7 A B C D E ∧ sum_left_less_5 A B C D E ∧
  (A, B, C, D, E) = (1, 2, 3, 4, 6) := sorry

end NUMINAMATH_GPT_unique_digit_solution_l1552_155243


namespace NUMINAMATH_GPT_find_m_l1552_155259

theorem find_m 
  (x1 x2 : ℝ) 
  (m : ℝ)
  (h1 : x1 + x2 = m)
  (h2 : x1 * x2 = 2 * m - 1)
  (h3 : x1^2 + x2^2 = 7) : 
  m = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1552_155259


namespace NUMINAMATH_GPT_inequality_solution_l1552_155288

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) : (x^2 > x^(1 / 2)) ↔ (x > 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1552_155288


namespace NUMINAMATH_GPT_pool_full_capacity_is_2000_l1552_155253

-- Definitions based on the conditions given
def water_loss_per_jump : ℕ := 400 -- in ml
def jumps_before_cleaning : ℕ := 1000
def cleaning_threshold : ℚ := 0.80 -- 80%
def total_water_loss : ℕ := water_loss_per_jump * jumps_before_cleaning -- in ml
def water_loss_liters : ℚ := total_water_loss / 1000 -- converting ml to liters
def cleaning_loss_fraction : ℚ := 1 - cleaning_threshold -- 20% loss

-- The actual proof statement
theorem pool_full_capacity_is_2000 :
  (water_loss_liters : ℚ) / cleaning_loss_fraction = 2000 :=
by
  sorry

end NUMINAMATH_GPT_pool_full_capacity_is_2000_l1552_155253


namespace NUMINAMATH_GPT_pen_cost_l1552_155240

variable (p i : ℝ)

theorem pen_cost (h1 : p + i = 1.10) (h2 : p = 1 + i) : p = 1.05 :=
by 
  -- proof steps here
  sorry

end NUMINAMATH_GPT_pen_cost_l1552_155240


namespace NUMINAMATH_GPT_num_ways_two_different_colors_l1552_155280

theorem num_ways_two_different_colors 
  (red white blue : ℕ) 
  (total_balls : ℕ) 
  (choose : ℕ → ℕ → ℕ) 
  (h_red : red = 2) 
  (h_white : white = 3) 
  (h_blue : blue = 1) 
  (h_total : total_balls = red + white + blue) 
  (h_choose_total : choose total_balls 3 = 20)
  (h_choose_three_diff_colors : 2 * 3 * 1 = 6)
  (h_one_color : 1 = 1) :
  choose total_balls 3 - 6 - 1 = 13 := 
by
  sorry

end NUMINAMATH_GPT_num_ways_two_different_colors_l1552_155280


namespace NUMINAMATH_GPT_coffee_prices_purchase_ways_l1552_155233

-- Define the cost equations for coffee A and B
def cost_equation1 (x y : ℕ) : Prop := 10 * x + 15 * y = 230
def cost_equation2 (x y : ℕ) : Prop := 25 * x + 25 * y = 450

-- Define what we need to prove for task 1
theorem coffee_prices (x y : ℕ) (h1 : cost_equation1 x y) (h2 : cost_equation2 x y) : x = 8 ∧ y = 10 := 
sorry

-- Define the condition for valid purchases of coffee A and B
def valid_purchase (m n : ℕ) : Prop := 8 * m + 10 * n = 200

-- Prove that there are 4 ways to purchase coffee A and B with 200 yuan
theorem purchase_ways : ∃ several : ℕ, several = 4 ∧ (∃ m n : ℕ, valid_purchase m n) := 
sorry

end NUMINAMATH_GPT_coffee_prices_purchase_ways_l1552_155233


namespace NUMINAMATH_GPT_y_coordinate_of_P_l1552_155231

theorem y_coordinate_of_P (x y : ℝ) (h1 : |y| = 1/2 * |x|) (h2 : |x| = 12) :
  y = 6 ∨ y = -6 :=
sorry

end NUMINAMATH_GPT_y_coordinate_of_P_l1552_155231


namespace NUMINAMATH_GPT_abc_equality_l1552_155294

theorem abc_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
                      (h : a^3 + b^3 + c^3 - 3 * a * b * c = 0) : a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_abc_equality_l1552_155294


namespace NUMINAMATH_GPT_find_number_of_girls_l1552_155221

variable (G : ℕ)

-- Given conditions
def avg_weight_girls (total_weight_girls : ℕ) : Prop := total_weight_girls = 45 * G
def avg_weight_boys (total_weight_boys : ℕ) : Prop := total_weight_boys = 275
def avg_weight_students (total_weight_students : ℕ) : Prop := total_weight_students = 500

-- Proposition to prove
theorem find_number_of_girls 
  (total_weight_girls : ℕ) 
  (total_weight_boys : ℕ) 
  (total_weight_students : ℕ) 
  (h1 : avg_weight_girls G total_weight_girls)
  (h2 : avg_weight_boys total_weight_boys)
  (h3 : avg_weight_students total_weight_students) : 
  G = 5 :=
by sorry

end NUMINAMATH_GPT_find_number_of_girls_l1552_155221


namespace NUMINAMATH_GPT_last_number_is_five_l1552_155292

theorem last_number_is_five (seq : ℕ → ℕ) (h₀ : seq 0 = 5)
  (h₁ : ∀ n < 32, seq n + seq (n+1) + seq (n+2) + seq (n+3) + seq (n+4) + seq (n+5) = 29) :
  seq 36 = 5 :=
sorry

end NUMINAMATH_GPT_last_number_is_five_l1552_155292


namespace NUMINAMATH_GPT_find_efg_correct_l1552_155223

noncomputable def find_efg (M : ℕ) : ℕ :=
  let efgh := M % 10000
  let e := efgh / 1000
  let efg := efgh / 10
  if (M^2 % 10000 = efgh) ∧ (e ≠ 0) ∧ ((M % 32 = 0 ∧ (M - 1) % 125 = 0) ∨ (M % 125 = 0 ∧ (M - 1) % 32 = 0))
  then efg
  else 0
  
theorem find_efg_correct {M : ℕ} (h_conditions: (M^2 % 10000 = M % 10000) ∧ (M % 32 = 0 ∧ (M - 1) % 125 = 0 ∨ M % 125 = 0 ∧ (M-1) % 32 = 0) ∧ ((M % 10000 / 1000) ≠ 0)) :
  find_efg M = 362 :=
by
  sorry

end NUMINAMATH_GPT_find_efg_correct_l1552_155223


namespace NUMINAMATH_GPT_mean_properties_l1552_155299

theorem mean_properties (a b c : ℝ) 
    (h1 : a + b + c = 36) 
    (h2 : a * b * c = 125) 
    (h3 : a * b + b * c + c * a = 93.75) : 
    a^2 + b^2 + c^2 = 1108.5 := 
by 
  sorry

end NUMINAMATH_GPT_mean_properties_l1552_155299


namespace NUMINAMATH_GPT_quadratic_root_a_value_l1552_155293

theorem quadratic_root_a_value (a k : ℝ) (h1 : k = 65) (h2 : a * (5:ℝ)^2 + 3 * (5:ℝ) - k = 0) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_a_value_l1552_155293


namespace NUMINAMATH_GPT_coordinates_of_B_l1552_155235

-- Define the point A
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := { x := 2, y := 1 }

-- Define the rotation transformation for pi/2 clockwise
def rotate_clockwise_90 (p : Point) : Point :=
  { x := p.y, y := -p.x }

-- Define the point B after rotation
def B := rotate_clockwise_90 A

-- The theorem stating the coordinates of point B (the correct answer)
theorem coordinates_of_B : B = { x := 1, y := -2 } :=
  sorry

end NUMINAMATH_GPT_coordinates_of_B_l1552_155235


namespace NUMINAMATH_GPT_fraction_spent_first_week_l1552_155254

theorem fraction_spent_first_week
  (S : ℝ) (F : ℝ)
  (h1 : S > 0)
  (h2 : F * S + 3 * (0.20 * S) + 0.15 * S = S) : 
  F = 0.25 := 
sorry

end NUMINAMATH_GPT_fraction_spent_first_week_l1552_155254


namespace NUMINAMATH_GPT_grandma_gave_each_l1552_155265

-- Define the conditions
def gasoline: ℝ := 8
def lunch: ℝ := 15.65
def gifts: ℝ := 5 * 2  -- $5 each for two persons
def total_spent: ℝ := gasoline + lunch + gifts
def initial_amount: ℝ := 50
def amount_left: ℝ := 36.35

-- Define the proof problem
theorem grandma_gave_each :
  (amount_left - (initial_amount - total_spent)) / 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_grandma_gave_each_l1552_155265


namespace NUMINAMATH_GPT_limit_of_f_at_infinity_l1552_155287

open Filter
open Topology

variable (f : ℝ → ℝ)
variable (h_continuous : Continuous f)
variable (h_seq_limit : ∀ α > 0, Tendsto (fun n : ℕ => f (n * α)) atTop (nhds 0))

theorem limit_of_f_at_infinity : Tendsto f atTop (nhds 0) := by
  sorry

end NUMINAMATH_GPT_limit_of_f_at_infinity_l1552_155287


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l1552_155263

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l1552_155263


namespace NUMINAMATH_GPT_family_members_count_l1552_155290

-- Defining the conditions given in the problem
variables (cyrus_bites_arms_legs : ℕ) (cyrus_bites_body : ℕ) (total_bites_family : ℕ)
variables (family_bites_per_person : ℕ) (cyrus_total_bites : ℕ)

-- Given conditions
def condition1 : cyrus_bites_arms_legs = 14 := sorry
def condition2 : cyrus_bites_body = 10 := sorry
def condition3 : cyrus_total_bites = cyrus_bites_arms_legs + cyrus_bites_body := sorry
def condition4 : total_bites_family = cyrus_total_bites / 2 := sorry
def condition5 : ∀ n : ℕ, total_bites_family = n * family_bites_per_person := sorry

-- The theorem to prove: The number of people in the rest of Cyrus' family is 12
theorem family_members_count (n : ℕ) (h1 : cyrus_bites_arms_legs = 14)
    (h2 : cyrus_bites_body = 10) (h3 : cyrus_total_bites = cyrus_bites_arms_legs + cyrus_bites_body)
    (h4 : total_bites_family = cyrus_total_bites / 2)
    (h5 : ∀ n, total_bites_family = n * family_bites_per_person) : n = 12 :=
sorry

end NUMINAMATH_GPT_family_members_count_l1552_155290


namespace NUMINAMATH_GPT_qingyang_2015_mock_exam_l1552_155242

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def problem :=
  U = {1, 2, 3, 4, 5} ∧ A = {2, 3, 4} ∧ B = {2, 5} →
  B ∪ (U \ A) = {1, 2, 5}

theorem qingyang_2015_mock_exam (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) : problem U A B :=
by
  intros
  sorry

end NUMINAMATH_GPT_qingyang_2015_mock_exam_l1552_155242


namespace NUMINAMATH_GPT_net_price_change_is_twelve_percent_l1552_155227

variable (P : ℝ)

def net_price_change (P : ℝ) : ℝ := 
  let decreased_price := 0.8 * P
  let increased_price := 1.4 * decreased_price
  increased_price - P

theorem net_price_change_is_twelve_percent (P : ℝ) : net_price_change P = 0.12 * P := by
  sorry

end NUMINAMATH_GPT_net_price_change_is_twelve_percent_l1552_155227


namespace NUMINAMATH_GPT_domain_change_l1552_155291

theorem domain_change (f : ℝ → ℝ) :
  (∀ x : ℝ, -2 ≤ x + 1 ∧ x + 1 ≤ 3) →
  (∀ x : ℝ, -2 ≤ 1 - 2 * x ∧ 1 - 2 * x ≤ 3) →
  ∀ x : ℝ, -3 / 2 ≤ x ∧ x ≤ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_change_l1552_155291


namespace NUMINAMATH_GPT_triangle_acute_l1552_155250

theorem triangle_acute (A B C : ℝ) (h1 : A = 2 * (180 / 9)) (h2 : B = 3 * (180 / 9)) (h3 : C = 4 * (180 / 9)) :
  A < 90 ∧ B < 90 ∧ C < 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_acute_l1552_155250


namespace NUMINAMATH_GPT_rhombus_side_length_15_l1552_155224

variable {p : ℝ} (h_p : p = 60)
variable {n : ℕ} (h_n : n = 4)

noncomputable def side_length_of_rhombus (p : ℝ) (n : ℕ) : ℝ :=
p / n

theorem rhombus_side_length_15 (h_p : p = 60) (h_n : n = 4) :
  side_length_of_rhombus p n = 15 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_side_length_15_l1552_155224


namespace NUMINAMATH_GPT_ivanov_family_net_worth_l1552_155258

-- Define the financial values
def value_of_apartment := 3000000
def market_value_of_car := 900000
def bank_savings := 300000
def value_of_securities := 200000
def liquid_cash := 100000
def remaining_mortgage := 1500000
def car_loan := 500000
def debt_to_relatives := 200000

-- Calculate total assets and total liabilities
def total_assets := value_of_apartment + market_value_of_car + bank_savings + value_of_securities + liquid_cash
def total_liabilities := remaining_mortgage + car_loan + debt_to_relatives

-- Define the hypothesis and the final result of the net worth calculation
theorem ivanov_family_net_worth : total_assets - total_liabilities = 2300000 := by
  sorry

end NUMINAMATH_GPT_ivanov_family_net_worth_l1552_155258


namespace NUMINAMATH_GPT_Duke_broke_record_by_5_l1552_155212

theorem Duke_broke_record_by_5 :
  let free_throws := 5
  let regular_baskets := 4
  let normal_three_pointers := 2
  let extra_three_pointers := 1
  let points_per_free_throw := 1
  let points_per_regular_basket := 2
  let points_per_three_pointer := 3
  let points_to_tie_record := 17

  let total_points_scored := (free_throws * points_per_free_throw) +
                             (regular_baskets * points_per_regular_basket) +
                             ((normal_three_pointers + extra_three_pointers) * points_per_three_pointer)
  total_points_scored = 22 →
  total_points_scored - points_to_tie_record = 5 :=

by
  intros
  sorry

end NUMINAMATH_GPT_Duke_broke_record_by_5_l1552_155212


namespace NUMINAMATH_GPT_common_points_count_l1552_155284

noncomputable def eq1 (x y : ℝ) : Prop := (x - 2 * y + 3) * (4 * x + y - 5) = 0
noncomputable def eq2 (x y : ℝ) : Prop := (x + 2 * y - 5) * (3 * x - 4 * y + 6) = 0

theorem common_points_count : 
  (∃ x1 y1 : ℝ, eq1 x1 y1 ∧ eq2 x1 y1) ∧
  (∃ x2 y2 : ℝ, eq1 x2 y2 ∧ eq2 x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2)) ∧
  (∃ x3 y3 : ℝ, eq1 x3 y3 ∧ eq2 x3 y3 ∧ (x3 ≠ x1 ∧ x3 ≠ x2 ∧ y3 ≠ y1 ∧ y3 ≠ y2)) ∧ 
  (∃ x4 y4 : ℝ, eq1 x4 y4 ∧ eq2 x4 y4 ∧ (x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3 ∧ y4 ≠ y1 ∧ y4 ≠ y2 ∧ y4 ≠ y3)) ∧ 
  ∀ x y : ℝ, (eq1 x y ∧ eq2 x y) → (((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4))) :=
by
  sorry

end NUMINAMATH_GPT_common_points_count_l1552_155284


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1552_155271

theorem isosceles_triangle_perimeter :
  ∀ x y : ℝ, x^2 - 7*x + 10 = 0 → y^2 - 7*y + 10 = 0 → x ≠ y → x + x + y = 12 :=
by
  intros x y hx hy hxy
  -- Place for proof
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1552_155271


namespace NUMINAMATH_GPT_mary_carrots_correct_l1552_155275

def sandy_carrots := 8
def total_carrots := 14

def mary_carrots := total_carrots - sandy_carrots

theorem mary_carrots_correct : mary_carrots = 6 := by
  unfold mary_carrots
  unfold total_carrots
  unfold sandy_carrots
  sorry

end NUMINAMATH_GPT_mary_carrots_correct_l1552_155275


namespace NUMINAMATH_GPT_largest_x_satisfying_abs_eq_largest_x_is_correct_l1552_155230

theorem largest_x_satisfying_abs_eq (x : ℝ) (h : |x - 5| = 12) : x ≤ 17 :=
by
  sorry

noncomputable def largest_x : ℝ := 17

theorem largest_x_is_correct (x : ℝ) (h : |x - 5| = 12) : x ≤ largest_x :=
largest_x_satisfying_abs_eq x h

end NUMINAMATH_GPT_largest_x_satisfying_abs_eq_largest_x_is_correct_l1552_155230


namespace NUMINAMATH_GPT_algebraic_expression_value_l1552_155279

theorem algebraic_expression_value (x : ℝ) (h : x = 2 * Real.sqrt 3 - 1) : x^2 + 2 * x - 3 = 8 :=
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1552_155279


namespace NUMINAMATH_GPT_minimum_value_expression_l1552_155202

theorem minimum_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 12 * b^3 + 27 * c^3 + (3 / (27 * a * b * c)) ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l1552_155202


namespace NUMINAMATH_GPT_third_candidate_votes_l1552_155261

theorem third_candidate_votes (V A B W: ℕ) (hA : A = 2500) (hB : B = 15000) 
  (hW : W = (2 * V) / 3) (hV : V = W + A + B) : (V - (A + B)) = 35000 := by
  sorry

end NUMINAMATH_GPT_third_candidate_votes_l1552_155261


namespace NUMINAMATH_GPT_sequence_property_l1552_155217

theorem sequence_property (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n+1) + 2021 * a (n+2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := 
sorry

end NUMINAMATH_GPT_sequence_property_l1552_155217


namespace NUMINAMATH_GPT_cos_theta_minus_pi_six_l1552_155251

theorem cos_theta_minus_pi_six (θ : ℝ) (h : Real.sin (θ + π / 3) = 2 / 3) : 
  Real.cos (θ - π / 6) = 2 / 3 :=
sorry

end NUMINAMATH_GPT_cos_theta_minus_pi_six_l1552_155251


namespace NUMINAMATH_GPT_value_of_6_inch_cube_is_1688_l1552_155237

noncomputable def cube_value (side_length : ℝ) : ℝ :=
  let volume := side_length ^ 3
  (volume / 64) * 500

-- Main statement
theorem value_of_6_inch_cube_is_1688 :
  cube_value 6 = 1688 := by
  sorry

end NUMINAMATH_GPT_value_of_6_inch_cube_is_1688_l1552_155237


namespace NUMINAMATH_GPT_coeff_comparison_l1552_155262

def a_k (k : ℕ) : ℕ := (2 ^ k) * Nat.choose 100 k

theorem coeff_comparison :
  (Finset.filter (fun r => a_k r < a_k (r + 1)) (Finset.range 100)).card = 67 :=
by
  sorry

end NUMINAMATH_GPT_coeff_comparison_l1552_155262


namespace NUMINAMATH_GPT_find_percentage_l1552_155267

theorem find_percentage (P : ℝ) (N : ℝ) (h1 : N = 140) (h2 : (P / 100) * N = (4 / 5) * N - 21) : P = 65 := by
  sorry

end NUMINAMATH_GPT_find_percentage_l1552_155267


namespace NUMINAMATH_GPT_circle_diameter_equality_l1552_155213

theorem circle_diameter_equality (r d : ℝ) (h₁ : d = 2 * r) (h₂ : π * d = π * r^2) : d = 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_equality_l1552_155213


namespace NUMINAMATH_GPT_drowning_ratio_l1552_155255

variable (total_sheep total_cows total_dogs drowned_sheep drowned_cows total_animals : ℕ)

-- Conditions provided
variable (initial_conditions : total_sheep = 20 ∧ total_cows = 10 ∧ total_dogs = 14)
variable (sheep_drowned_condition : drowned_sheep = 3)
variable (dogs_shore_condition : total_dogs = 14)
variable (total_made_it_shore : total_animals = 35)

theorem drowning_ratio (h1 : total_sheep = 20) (h2 : total_cows = 10) (h3 : total_dogs = 14) 
    (h4 : drowned_sheep = 3) (h5 : total_animals = 35) 
    : (drowned_cows = 2 * drowned_sheep) :=
by
  sorry

end NUMINAMATH_GPT_drowning_ratio_l1552_155255


namespace NUMINAMATH_GPT_functional_equation_continuous_function_l1552_155234

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_continuous_function (f : ℝ → ℝ) (x₀ : ℝ) (h1 : Continuous f) (h2 : f x₀ ≠ 0) 
  (h3 : ∀ x y : ℝ, f (x + y) = f x * f y) : 
  ∃ a > 0, ∀ x : ℝ, f x = a ^ x := 
by
  sorry

end NUMINAMATH_GPT_functional_equation_continuous_function_l1552_155234


namespace NUMINAMATH_GPT_min_value_f_solve_inequality_f_l1552_155218

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

-- Proof Problem 1
theorem min_value_f : ∃ x : ℝ, f x = 3 :=
by { sorry }

-- Proof Problem 2
theorem solve_inequality_f : {x : ℝ | abs (f x - 6) ≤ 1} = 
    ({x : ℝ | -10/3 ≤ x ∧ x ≤ -8/3} ∪ 
    {x : ℝ | 0 ≤ x ∧ x ≤ 1} ∪ 
    {x : ℝ | 1 < x ∧ x ≤ 4/3}) :=
by { sorry }

end NUMINAMATH_GPT_min_value_f_solve_inequality_f_l1552_155218


namespace NUMINAMATH_GPT_min_ratio_of_cylinder_cone_l1552_155286

open Real

noncomputable def V1 (r : ℝ) : ℝ := 2 * π * r^3
noncomputable def V2 (R m r : ℝ) : ℝ := (1 / 3) * π * R^2 * m
noncomputable def geometric_constraint (R m r : ℝ) : Prop :=
  R / m = r / (sqrt ((m - r)^2 - r^2))

theorem min_ratio_of_cylinder_cone (r : ℝ) (hr : r > 0) : 
  ∃ R m, geometric_constraint R m r ∧ (V2 R m r) / (V1 r) = 4 / 3 := 
sorry

end NUMINAMATH_GPT_min_ratio_of_cylinder_cone_l1552_155286


namespace NUMINAMATH_GPT_no_n_in_range_l1552_155205

def g (n : ℕ) : ℕ := 7 + 4 * n + 6 * n ^ 2 + 3 * n ^ 3 + 4 * n ^ 4 + 3 * n ^ 5

theorem no_n_in_range
  : ¬ ∃ n : ℕ, 2 ≤ n ∧ n ≤ 100 ∧ g n % 11 = 0 := sorry

end NUMINAMATH_GPT_no_n_in_range_l1552_155205


namespace NUMINAMATH_GPT_log2_bounds_sum_l1552_155229

theorem log2_bounds_sum (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : (a : ℝ) < Real.log 50 / Real.log 2) (h4 : Real.log 50 / Real.log 2 < (b : ℝ)) :
  a + b = 11 :=
sorry

end NUMINAMATH_GPT_log2_bounds_sum_l1552_155229


namespace NUMINAMATH_GPT_max_temp_range_l1552_155228

-- Definitions based on given conditions
def average_temp : ℤ := 40
def lowest_temp : ℤ := 30

-- Total number of days
def days : ℕ := 5

-- Given that the average temperature and lowest temperature are provided, prove the maximum range.
theorem max_temp_range 
  (avg_temp_eq : (average_temp * days) = 200)
  (temp_min : lowest_temp = 30) : 
  ∃ max_temp : ℤ, max_temp - lowest_temp = 50 :=
by
  -- Assume maximum temperature
  let max_temp := 80
  have total_sum := (average_temp * days)
  have min_occurrences := 3 * lowest_temp
  have highest_temp := total_sum - min_occurrences - lowest_temp
  have range := highest_temp - lowest_temp
  use max_temp
  sorry

end NUMINAMATH_GPT_max_temp_range_l1552_155228


namespace NUMINAMATH_GPT_simplify_expression_l1552_155216

variable (b : ℝ) (hb : 0 < b)

theorem simplify_expression : 
  ( ( b ^ (16 / 8) ^ (1 / 4) ) ^ 3 * ( b ^ (16 / 4) ^ (1 / 8) ) ^ 3 ) = b ^ 3 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1552_155216
