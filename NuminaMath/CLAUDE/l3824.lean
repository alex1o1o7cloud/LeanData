import Mathlib

namespace exponent_equation_solution_l3824_382421

theorem exponent_equation_solution :
  ∃ x : ℤ, (5 : ℝ)^7 * (5 : ℝ)^x = 125 ∧ x = -4 :=
by sorry

end exponent_equation_solution_l3824_382421


namespace sqrt_2x_minus_4_meaningful_l3824_382462

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 := by sorry

end sqrt_2x_minus_4_meaningful_l3824_382462


namespace prob_at_least_one_girl_pair_approx_l3824_382451

/-- The number of boys in the group -/
def num_boys : ℕ := 8

/-- The number of girls in the group -/
def num_girls : ℕ := 8

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The probability of at least one pair consisting of two girls -/
noncomputable def prob_at_least_one_girl_pair : ℝ :=
  1 - (num_boys.factorial * num_girls.factorial * (2^num_pairs) * num_pairs.factorial) / total_people.factorial

/-- Theorem stating that the probability of at least one pair consisting of two girls is approximately 0.98 -/
theorem prob_at_least_one_girl_pair_approx :
  abs (prob_at_least_one_girl_pair - 0.98) < 0.01 := by
  sorry

end prob_at_least_one_girl_pair_approx_l3824_382451


namespace suitcase_lock_settings_l3824_382418

/-- The number of digits on each dial -/
def num_digits : ℕ := 10

/-- The number of dials on the lock -/
def num_dials : ℕ := 4

/-- The number of choices for each dial after the first -/
def choices_after_first : ℕ := num_digits - 1

/-- The total number of possible settings for the lock -/
def total_settings : ℕ := num_digits * choices_after_first^(num_dials - 1)

/-- Theorem stating that the total number of settings is 7290 -/
theorem suitcase_lock_settings : total_settings = 7290 := by
  sorry

end suitcase_lock_settings_l3824_382418


namespace range_of_a_l3824_382496

/-- Given a line l: x + y + a = 0 and a point A(0, 2), if there exists a point M on line l 
    such that |MA|^2 + |MO|^2 = 10 (where O is the origin), then -2√2 - 1 ≤ a ≤ 2√2 - 1 -/
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + (-x-a)^2 + x^2 + (-x-a-2)^2 = 10) → 
  -2 * Real.sqrt 2 - 1 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 1 :=
by sorry

end range_of_a_l3824_382496


namespace algebraic_expression_simplification_expression_value_at_negative_quarter_l3824_382459

theorem algebraic_expression_simplification (a : ℝ) :
  (a - 2)^2 + (a + 1) * (a - 1) - 2 * a * (a - 3) = 2 * a + 3 :=
by sorry

theorem expression_value_at_negative_quarter :
  let a : ℝ := -1/4
  (a - 2)^2 + (a + 1) * (a - 1) - 2 * a * (a - 3) = 5/2 :=
by sorry

end algebraic_expression_simplification_expression_value_at_negative_quarter_l3824_382459


namespace intersection_implies_k_geq_two_l3824_382452

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x : ℝ | x - k ≤ 0}

-- State the theorem
theorem intersection_implies_k_geq_two (k : ℝ) : M ∩ N k = M → k ≥ 2 := by
  sorry

end intersection_implies_k_geq_two_l3824_382452


namespace alpha_30_sufficient_not_necessary_for_sin_half_l3824_382468

theorem alpha_30_sufficient_not_necessary_for_sin_half :
  (∀ α : Real, α = 30 * π / 180 → Real.sin α = 1 / 2) ∧
  (∃ α : Real, Real.sin α = 1 / 2 ∧ α ≠ 30 * π / 180) := by
  sorry

end alpha_30_sufficient_not_necessary_for_sin_half_l3824_382468


namespace frog_paths_count_l3824_382467

/-- Represents a triangular grid -/
structure TriangularGrid :=
  (top_row_squares : ℕ)
  (total_squares : ℕ)

/-- Represents the possible moves of the frog -/
inductive Move
  | down
  | down_left

/-- Calculates the number of distinct paths in a triangular grid -/
def count_distinct_paths (grid : TriangularGrid) : ℕ :=
  sorry

/-- Theorem stating the number of distinct paths in the specific grid -/
theorem frog_paths_count (grid : TriangularGrid) 
  (h1 : grid.top_row_squares = 5)
  (h2 : grid.total_squares = 29) :
  count_distinct_paths grid = 256 :=
sorry

end frog_paths_count_l3824_382467


namespace total_bird_wings_l3824_382472

/-- The number of birds in the sky -/
def num_birds : ℕ := 10

/-- The number of wings each bird has -/
def wings_per_bird : ℕ := 2

/-- Theorem: The total number of bird wings in the sky is 20 -/
theorem total_bird_wings : num_birds * wings_per_bird = 20 := by
  sorry

end total_bird_wings_l3824_382472


namespace first_group_size_l3824_382414

/-- Represents the number of questions Cameron answers per tourist -/
def questions_per_tourist : ℕ := 2

/-- Represents the total number of tour groups -/
def total_groups : ℕ := 4

/-- Represents the number of people in the second group -/
def second_group : ℕ := 11

/-- Represents the number of people in the third group -/
def third_group : ℕ := 8

/-- Represents the number of people in the fourth group -/
def fourth_group : ℕ := 7

/-- Represents the total number of questions Cameron answered -/
def total_questions : ℕ := 68

/-- Proves that the number of people in the first tour group is 8 -/
theorem first_group_size : ℕ := by
  sorry

end first_group_size_l3824_382414


namespace problem_solution_l3824_382466

noncomputable section

variable (g : ℝ → ℝ)

-- g is invertible
variable (h : Function.Bijective g)

-- Define the values of g given in the table
axiom g_1 : g 1 = 4
axiom g_2 : g 2 = 6
axiom g_3 : g 3 = 9
axiom g_4 : g 4 = 10
axiom g_5 : g 5 = 12

-- The theorem to prove
theorem problem_solution :
  g (g 2) + g (Function.invFun g 12) + Function.invFun g (Function.invFun g 10) = 25 := by
  sorry

end

end problem_solution_l3824_382466


namespace bus_ride_difference_l3824_382430

theorem bus_ride_difference (vince_ride zachary_ride : ℝ) 
  (h1 : vince_ride = 0.625) 
  (h2 : zachary_ride = 0.5) : 
  vince_ride - zachary_ride = 0.125 := by
  sorry

end bus_ride_difference_l3824_382430


namespace near_square_quotient_l3824_382492

/-- A natural number is a near-square if it is the product of two consecutive natural numbers. -/
def is_near_square (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

/-- Theorem stating that any near-square can be represented as the quotient of two near-squares. -/
theorem near_square_quotient (n : ℕ) : 
  is_near_square (n * (n + 1)) → 
  ∃ a b c : ℕ, 
    is_near_square a ∧ 
    is_near_square b ∧ 
    is_near_square c ∧ 
    n * (n + 1) = a / c ∧
    b = c * (n + 2) :=
sorry

end near_square_quotient_l3824_382492


namespace triangle_abc_properties_l3824_382498

/-- Triangle ABC with given properties -/
structure TriangleABC where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  side_angle_correspondence : True  -- Sides a, b, c are opposite to angles A, B, C respectively
  cosine_relation : b * (1 - 2 * Real.cos A) = 2 * a * Real.cos B
  a_value : a = 1
  tan_A_value : Real.tan A = 2 * Real.sqrt 2

/-- Main theorem about the properties of TriangleABC -/
theorem triangle_abc_properties (t : TriangleABC) :
  t.b = 2 * t.c ∧ 
  (1/2 * t.b * t.c * Real.sin t.A : ℝ) = 2 * Real.sqrt 2 / 11 :=
sorry

end triangle_abc_properties_l3824_382498


namespace intersection_point_k_value_l3824_382486

theorem intersection_point_k_value :
  ∀ (x y k : ℝ),
  (x = -7.5) →
  (-3 * x + y = k) →
  (0.3 * x + y = 12) →
  (k = 36.75) := by
sorry

end intersection_point_k_value_l3824_382486


namespace distance_between_trees_l3824_382443

/-- Given a yard of length 441 meters with 22 equally spaced trees (including one at each end),
    the distance between two consecutive trees is 21 meters. -/
theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (distance : ℕ) :
  yard_length = 441 →
  num_trees = 22 →
  distance * (num_trees - 1) = yard_length →
  distance = 21 :=
by sorry

end distance_between_trees_l3824_382443


namespace sum_of_digits_of_power_l3824_382461

def base : ℕ := 3 + 4
def exponent : ℕ := 21

def last_two_digits (n : ℕ) : ℕ := n % 100

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_power :
  tens_digit (last_two_digits (base ^ exponent)) + ones_digit (last_two_digits (base ^ exponent)) = 7 := by
  sorry

end sum_of_digits_of_power_l3824_382461


namespace log_sqrt10_1000sqrt10_l3824_382495

theorem log_sqrt10_1000sqrt10 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end log_sqrt10_1000sqrt10_l3824_382495


namespace money_distribution_l3824_382457

theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 → 
  a + c = 200 → 
  b + c = 350 → 
  c = 50 := by
sorry

end money_distribution_l3824_382457


namespace bracelet_ratio_l3824_382438

theorem bracelet_ratio : 
  ∀ (x : ℕ), 
  (5 + x : ℚ) - (1/3) * (5 + x) = 6 → 
  (x : ℚ) / 16 = 1/4 := by
  sorry

end bracelet_ratio_l3824_382438


namespace continuous_fraction_identity_l3824_382411

-- Define x as the solution to the equation x = 1 + √3 / x
noncomputable def x : ℝ := Real.sqrt 3 + 1

-- State the theorem
theorem continuous_fraction_identity :
  1 / ((x + 2) * (x - 3)) = (Real.sqrt 3 + 6) / (-33) := by
  sorry

end continuous_fraction_identity_l3824_382411


namespace rectangle_rhombus_perimeter_ratio_l3824_382487

/-- The ratio of the perimeter of a 3 by 2 rectangle to the perimeter of a rhombus
    formed by rearranging four congruent right-angled triangles that the rectangle
    is split into is 1:1. -/
theorem rectangle_rhombus_perimeter_ratio :
  let rectangle_length : ℝ := 3
  let rectangle_width : ℝ := 2
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  let triangle_leg1 := rectangle_length / 2
  let triangle_leg2 := rectangle_width
  let triangle_hypotenuse := Real.sqrt (triangle_leg1^2 + triangle_leg2^2)
  let rhombus_side := triangle_hypotenuse
  let rhombus_perimeter := 4 * rhombus_side
  rectangle_perimeter / rhombus_perimeter = 1 := by
sorry

end rectangle_rhombus_perimeter_ratio_l3824_382487


namespace dormitory_to_city_distance_l3824_382497

theorem dormitory_to_city_distance : ∃ (D : ℝ), 
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 14 = D ∧ D = 105 := by
  sorry

end dormitory_to_city_distance_l3824_382497


namespace imaginary_part_of_z_l3824_382475

theorem imaginary_part_of_z (z : ℂ) : z = Complex.I * (-1 + 2 * Complex.I) → z.im = -1 := by
  sorry

end imaginary_part_of_z_l3824_382475


namespace equation_solution_l3824_382406

theorem equation_solution : 
  let f (x : ℝ) := 1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
                   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5))
  ∀ x : ℝ, f x = 1 / 10 ↔ x = 10 ∨ x = -3.5 :=
by sorry

end equation_solution_l3824_382406


namespace marble_count_l3824_382456

theorem marble_count (g y p : ℕ) : 
  y + p = 7 →  -- all but 7 are green
  g + p = 10 → -- all but 10 are yellow
  g + y = 5 →  -- all but 5 are purple
  g + y + p = 11 := by
sorry

end marble_count_l3824_382456


namespace min_digits_removal_l3824_382444

def original_number : ℕ := 20162016

def is_valid_removal (n : ℕ) : Prop :=
  ∃ (removed : ℕ),
    removed > 0 ∧
    removed < original_number ∧
    (original_number - removed) % 2016 = 0 ∧
    (String.length (toString removed) + String.length (toString (original_number - removed)) = 8)

theorem min_digits_removal :
  (∀ n : ℕ, n < 3 → ¬(is_valid_removal n)) ∧
  (∃ n : ℕ, n = 3 ∧ is_valid_removal n) :=
sorry

end min_digits_removal_l3824_382444


namespace solution_range_l3824_382409

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, 2 * (x + a) = x + 3 ∧ 2 * x - 10 > 8 * a) → 
  a < -1/3 := by
sorry

end solution_range_l3824_382409


namespace is_334th_term_l3824_382410

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem is_334th_term :
  arithmetic_sequence 7 6 334 = 2005 :=
by sorry

end is_334th_term_l3824_382410


namespace starting_lineup_count_l3824_382493

def total_members : ℕ := 12
def offensive_linemen : ℕ := 4
def quick_reflex_players : ℕ := 2

def starting_lineup_combinations : ℕ := offensive_linemen * quick_reflex_players * 1 * (total_members - 3)

theorem starting_lineup_count : starting_lineup_combinations = 72 := by
  sorry

end starting_lineup_count_l3824_382493


namespace b_40_mod_49_l3824_382408

def b (n : ℕ) : ℤ := 5^n - 7^n

theorem b_40_mod_49 : b 40 ≡ 2 [ZMOD 49] := by sorry

end b_40_mod_49_l3824_382408


namespace new_job_bonus_calculation_l3824_382415

/-- Represents Maisy's job options and earnings -/
structure JobOption where
  hours_per_week : ℕ
  hourly_wage : ℕ
  bonus : ℕ

/-- Calculates the weekly earnings for a job option -/
def weekly_earnings (job : JobOption) : ℕ :=
  job.hours_per_week * job.hourly_wage + job.bonus

theorem new_job_bonus_calculation (current_job new_job : JobOption) 
  (h1 : current_job.hours_per_week = 8)
  (h2 : current_job.hourly_wage = 10)
  (h3 : current_job.bonus = 0)
  (h4 : new_job.hours_per_week = 4)
  (h5 : new_job.hourly_wage = 15)
  (h6 : weekly_earnings new_job = weekly_earnings current_job + 15) :
  new_job.bonus = 15 := by
  sorry

#check new_job_bonus_calculation

end new_job_bonus_calculation_l3824_382415


namespace max_value_when_min_ratio_l3824_382453

theorem max_value_when_min_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 - 3*x*y + 4*y^2 - z = 0) :
  ∃ (max_value : ℝ), max_value = 2 ∧
  ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 →
  x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 →
  (z' / (x' * y') ≥ z / (x * y)) →
  x' + 2*y' - z' ≤ max_value :=
sorry

end max_value_when_min_ratio_l3824_382453


namespace square_perimeter_after_scaling_l3824_382482

theorem square_perimeter_after_scaling (a : ℝ) (h : a > 0) : 
  let s := Real.sqrt a
  let new_s := 3 * s
  a = 4 → 4 * new_s = 24 := by
sorry

end square_perimeter_after_scaling_l3824_382482


namespace line_plane_perpendicularity_l3824_382423

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : parallel a α) 
  (h4 : perpendicular b β) 
  (h5 : plane_parallel α β) : 
  line_perpendicular a b :=
sorry

end line_plane_perpendicularity_l3824_382423


namespace roots_sequence_sum_l3824_382425

theorem roots_sequence_sum (p q a b : ℝ) : 
  p > 0 → 
  q > 0 → 
  a ≠ b →
  a^2 - p*a + q = 0 →
  b^2 - p*b + q = 0 →
  (∃ d : ℝ, (a = -4 + d ∧ b = -4 + 2*d) ∨ (b = -4 + d ∧ a = -4 + 2*d)) →
  (∃ r : ℝ, (a = -4*r ∧ b = -4*r^2) ∨ (b = -4*r ∧ a = -4*r^2)) →
  p + q = 26 := by
sorry

end roots_sequence_sum_l3824_382425


namespace book_pages_calculation_l3824_382439

/-- Given a book where:
  1. The initial ratio of pages read to pages not read is 3:4
  2. After reading 33 more pages, the ratio becomes 5:3
  This theorem states that the total number of pages in the book
  is equal to 33 divided by the difference between 5/8 and 3/7. -/
theorem book_pages_calculation (initial_read : ℚ) (initial_unread : ℚ) 
  (final_read : ℚ) (final_unread : ℚ) :
  initial_read / initial_unread = 3 / 4 →
  (initial_read + 33) / initial_unread = 5 / 3 →
  (initial_read + initial_unread) = 33 / (5/8 - 3/7) := by
  sorry

end book_pages_calculation_l3824_382439


namespace least_multiple_of_17_greater_than_450_l3824_382447

theorem least_multiple_of_17_greater_than_450 :
  ∀ n : ℕ, n > 0 ∧ 17 ∣ n ∧ n > 450 → n ≥ 459 :=
by sorry

end least_multiple_of_17_greater_than_450_l3824_382447


namespace min_students_for_question_distribution_l3824_382433

theorem min_students_for_question_distribution (total_questions : Nat) 
  (folder_size : Nat) (num_folders : Nat) (max_unsolved : Nat) :
  total_questions = 2010 →
  folder_size = 670 →
  num_folders = 3 →
  max_unsolved = 2 →
  ∃ (min_students : Nat), 
    (∀ (n : Nat), n < min_students → 
      ¬(∀ (folder : Finset Nat), folder.card = folder_size → 
        ∃ (solved_by : Finset Nat), solved_by.card ≥ num_folders ∧ 
          ∀ (q : Nat), q ∈ folder → (n - solved_by.card) ≤ max_unsolved)) ∧
    (∀ (folder : Finset Nat), folder.card = folder_size → 
      ∃ (solved_by : Finset Nat), solved_by.card ≥ num_folders ∧ 
        ∀ (q : Nat), q ∈ folder → (min_students - solved_by.card) ≤ max_unsolved) ∧
    min_students = 6 := by
  sorry

end min_students_for_question_distribution_l3824_382433


namespace additional_wax_needed_l3824_382478

theorem additional_wax_needed (total_wax : ℕ) (available_wax : ℕ) (h1 : total_wax = 353) (h2 : available_wax = 331) :
  total_wax - available_wax = 22 := by
  sorry

end additional_wax_needed_l3824_382478


namespace x_eq_2_sufficient_not_necessary_l3824_382405

theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, x = 2 → (x + 1) * (x - 2) = 0) ∧
  (∃ x : ℝ, (x + 1) * (x - 2) = 0 ∧ x ≠ 2) :=
by sorry

end x_eq_2_sufficient_not_necessary_l3824_382405


namespace citrus_grove_orchards_l3824_382402

theorem citrus_grove_orchards (total : ℕ) (lemons : ℕ) (oranges : ℕ) (limes : ℕ) (grapefruits : ℕ) :
  total = 16 →
  lemons = 8 →
  oranges = lemons / 2 →
  limes + grapefruits = total - lemons - oranges →
  limes = grapefruits →
  grapefruits = 2 := by
sorry

end citrus_grove_orchards_l3824_382402


namespace solve_system_l3824_382432

theorem solve_system (x y : ℝ) (eq1 : 3 * x - 2 * y = 18) (eq2 : x + 2 * y = 10) : y = 1.5 := by
  sorry

end solve_system_l3824_382432


namespace soccer_stars_league_teams_l3824_382437

theorem soccer_stars_league_teams (n : ℕ) : n > 1 → (n * (n - 1)) / 2 = 28 → n = 8 := by
  sorry

end soccer_stars_league_teams_l3824_382437


namespace existence_of_least_t_for_geometric_progression_l3824_382450

open Real

theorem existence_of_least_t_for_geometric_progression :
  ∃ t : ℝ, t > 0 ∧
  ∃ α : ℝ, 0 < α ∧ α < π / 3 ∧
  ∃ r : ℝ, r > 0 ∧
  (arcsin (sin α) = α) ∧
  (arcsin (sin (3 * α)) = r * α) ∧
  (arcsin (sin (8 * α)) = r^2 * α) ∧
  (arcsin (sin (t * α)) = r^3 * α) ∧
  ∀ s : ℝ, s > 0 →
    (∃ β : ℝ, 0 < β ∧ β < π / 3 ∧
    ∃ q : ℝ, q > 0 ∧
    (arcsin (sin β) = β) ∧
    (arcsin (sin (3 * β)) = q * β) ∧
    (arcsin (sin (8 * β)) = q^2 * β) ∧
    (arcsin (sin (s * β)) = q^3 * β)) →
    t ≤ s :=
by sorry

end existence_of_least_t_for_geometric_progression_l3824_382450


namespace parabola_axis_l3824_382477

/-- The equation of the axis of the parabola y = x^2 -/
theorem parabola_axis (x y : ℝ) : 
  (y = x^2) → (∃ (axis : ℝ → ℝ), axis y = -1/4) :=
by sorry

end parabola_axis_l3824_382477


namespace triangle_properties_l3824_382420

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - t.c^2 = Real.sqrt 3 * t.a * t.b

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_condition t) 
  (h_angle : 0 < t.A ∧ t.A ≤ 2 * Real.pi / 3) : 
  t.C = Real.pi / 6 ∧ 
  ∀ m : ℝ, m = 2 * (Real.cos (t.A / 2))^2 - Real.sin t.B - 1 → 
  -1 ≤ m ∧ m < 1/2 := by
  sorry


end triangle_properties_l3824_382420


namespace solve_channels_problem_l3824_382483

def channels_problem (initial_channels : ℕ) 
                     (removed_channels : ℕ) 
                     (replaced_channels : ℕ) 
                     (sports_package_channels : ℕ) 
                     (supreme_sports_package_channels : ℕ) 
                     (final_channels : ℕ) : Prop :=
  let after_company_changes := initial_channels - removed_channels + replaced_channels
  let sports_packages_total := sports_package_channels + supreme_sports_package_channels
  let before_sports_packages := final_channels - sports_packages_total
  after_company_changes - before_sports_packages = 10

theorem solve_channels_problem : 
  channels_problem 150 20 12 8 7 147 := by
  sorry

end solve_channels_problem_l3824_382483


namespace sheets_used_for_printing_james_sheets_used_l3824_382442

/-- Calculate the number of sheets of paper used for printing books -/
theorem sheets_used_for_printing (num_books : ℕ) (pages_per_book : ℕ) 
  (pages_per_side : ℕ) (is_double_sided : Bool) : ℕ :=
  let total_pages := num_books * pages_per_book
  let pages_per_sheet := pages_per_side * (if is_double_sided then 2 else 1)
  total_pages / pages_per_sheet

/-- Prove that James uses 150 sheets of paper for printing his books -/
theorem james_sheets_used :
  sheets_used_for_printing 2 600 4 true = 150 := by
  sorry

end sheets_used_for_printing_james_sheets_used_l3824_382442


namespace second_train_length_second_train_length_solution_l3824_382473

/-- Calculates the length of the second train given the speeds of both trains,
    the time they take to clear each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (clear_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * (1000 / 3600)
  let total_distance := relative_speed_ms * clear_time
  total_distance - length1

/-- The length of the second train is approximately 1984 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, abs (second_train_length 75 65 7.353697418492236 121 - 1984) < ε :=
by
  sorry

end second_train_length_second_train_length_solution_l3824_382473


namespace cube_of_negative_half_x_squared_y_l3824_382413

theorem cube_of_negative_half_x_squared_y (x y : ℝ) : 
  (-1/2 * x^2 * y)^3 = -1/8 * x^6 * y^3 := by
  sorry

end cube_of_negative_half_x_squared_y_l3824_382413


namespace product_closest_to_640_l3824_382463

def product : ℝ := 0.0000421 * 15864300

def options : List ℝ := [620, 640, 660, 680, 700]

theorem product_closest_to_640 : 
  (options.argmin (fun x => |x - product|)) = some 640 := by sorry

end product_closest_to_640_l3824_382463


namespace smallest_valid_number_l3824_382458

def is_valid (n : ℕ) : Prop :=
  n % 10 = 9 ∧
  n % 9 = 8 ∧
  n % 8 = 7 ∧
  n % 7 = 6 ∧
  n % 6 = 5 ∧
  n % 5 = 4 ∧
  n % 4 = 3 ∧
  n % 3 = 2 ∧
  n % 2 = 1

theorem smallest_valid_number :
  is_valid 2519 ∧ ∀ m : ℕ, m < 2519 → ¬ is_valid m :=
sorry

end smallest_valid_number_l3824_382458


namespace sector_area_l3824_382436

/-- Given a sector with radius 4 cm and arc length 12 cm, its area is 24 cm². -/
theorem sector_area (radius : ℝ) (arc_length : ℝ) (area : ℝ) : 
  radius = 4 → arc_length = 12 → area = (1/2) * arc_length * radius → area = 24 := by
  sorry

end sector_area_l3824_382436


namespace min_score_on_last_two_l3824_382416

/-- The number of tests Shauna takes -/
def num_tests : ℕ := 5

/-- The maximum score possible on each test -/
def max_score : ℕ := 120

/-- The desired average score across all tests -/
def target_average : ℕ := 95

/-- Shauna's scores on the first three tests -/
def first_three_scores : Fin 3 → ℕ
  | 0 => 86
  | 1 => 112
  | 2 => 91

/-- The sum of Shauna's scores on the first three tests -/
def sum_first_three : ℕ := (first_three_scores 0) + (first_three_scores 1) + (first_three_scores 2)

/-- The theorem stating the minimum score needed on one of the last two tests -/
theorem min_score_on_last_two (score : ℕ) :
  (sum_first_three + score + max_score = target_average * num_tests) ∧
  (∀ s, s < score → sum_first_three + s + max_score < target_average * num_tests) →
  score = 66 := by
  sorry

end min_score_on_last_two_l3824_382416


namespace three_numbers_problem_l3824_382499

theorem three_numbers_problem :
  let x : ℚ := 1/9
  let y : ℚ := 1/6
  let z : ℚ := 1/3
  (x + y + z = 11/18) ∧
  (1/x + 1/y + 1/z = 18) ∧
  (2 * (1/y) = 1/x + 1/z) :=
by sorry

end three_numbers_problem_l3824_382499


namespace contrapositive_equivalence_l3824_382417

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) := by sorry

end contrapositive_equivalence_l3824_382417


namespace circle_radius_existence_l3824_382488

/-- Representation of a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Representation of a point -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Check if two circles intersect at two points -/
def circlesIntersect (c1 c2 : Circle) : Prop := sorry

/-- Check if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop := sorry

/-- Check if a circle is the circumcircle of a triangle -/
def isCircumcircle (c : Circle) (p1 p2 p3 : Point) : Prop := sorry

theorem circle_radius_existence :
  ∃! r : ℝ, r > 0 ∧
  ∃ (C1 C2 : Circle) (O X Y Z : Point),
    C1.radius = r ∧
    C1.center = O ∧
    isOnCircle O C2 ∧
    circlesIntersect C1 C2 ∧
    isOnCircle X C1 ∧ isOnCircle X C2 ∧
    isOnCircle Y C1 ∧ isOnCircle Y C2 ∧
    isOnCircle Z C2 ∧
    isOutside Z C1 ∧
    distance X Z = 15 ∧
    distance O Z = 13 ∧
    distance Y Z = 9 ∧
    isCircumcircle C2 X O Z ∧
    isCircumcircle C2 O Y Z :=
sorry

end circle_radius_existence_l3824_382488


namespace impossibleArrangement_l3824_382419

/-- Represents a 3x3 grid filled with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- The given fixed grid arrangement -/
def fixedGrid : Grid :=
  fun i j => Fin.mk ((i.val * 3 + j.val) % 9 + 1) (by sorry)

/-- Two positions are adjacent if they share a side -/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Two numbers are neighbors in a grid if they are in adjacent positions -/
def neighbors (g : Grid) (x y : Fin 9) : Prop :=
  ∃ (p q : Fin 3 × Fin 3), g p.1 p.2 = x ∧ g q.1 q.2 = y ∧ adjacent p q

theorem impossibleArrangement :
  ¬∃ (g₂ g₃ : Grid),
    (∀ x y : Fin 9, (neighbors fixedGrid x y ∨ neighbors g₂ x y ∨ neighbors g₃ x y) →
      ¬(neighbors fixedGrid x y ∧ neighbors g₂ x y) ∧
      ¬(neighbors fixedGrid x y ∧ neighbors g₃ x y) ∧
      ¬(neighbors g₂ x y ∧ neighbors g₃ x y)) :=
by sorry

end impossibleArrangement_l3824_382419


namespace elevator_is_translation_l3824_382465

/-- A structure representing a movement in space -/
structure Movement where
  is_straight_line : Bool

/-- Definition of translation in mathematics -/
def is_translation (m : Movement) : Prop :=
  m.is_straight_line = true

/-- Representation of an elevator's movement -/
def elevator_movement : Movement where
  is_straight_line := true

/-- Theorem stating that an elevator's movement is a translation -/
theorem elevator_is_translation : is_translation elevator_movement := by
  sorry

end elevator_is_translation_l3824_382465


namespace club_officers_count_l3824_382481

/-- Represents the number of ways to choose officers from a club with boys and girls --/
def chooseOfficers (boys girls : ℕ) : ℕ :=
  boys * girls * (boys - 1) + girls * boys * (girls - 1)

/-- Theorem stating the number of ways to choose officers in the given scenario --/
theorem club_officers_count :
  chooseOfficers 18 12 = 6048 := by
  sorry

end club_officers_count_l3824_382481


namespace complex_fraction_problem_l3824_382454

theorem complex_fraction_problem (x y : ℂ) 
  (h : (x - y) / (x + y) - (x + y) / (x - y) = 2) :
  ∃ (result : ℂ), (x^6 + y^6) / (x^6 - y^6) - (x^6 - y^6) / (x^6 + y^6) = result :=
by sorry

end complex_fraction_problem_l3824_382454


namespace inequality_proof_l3824_382469

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : |x - y| < 2) (hyz : |y - z| < 2) (hzx : |z - x| < 2) :
  Real.sqrt (x * y + 1) + Real.sqrt (y * z + 1) + Real.sqrt (z * x + 1) > x + y + z :=
by sorry

end inequality_proof_l3824_382469


namespace total_goals_theorem_l3824_382428

/-- Represents the number of goals scored by Louie in his last match -/
def louie_last_match_goals : ℕ := 4

/-- Represents the number of goals scored by Louie in previous matches -/
def louie_previous_goals : ℕ := 40

/-- Represents the number of seasons Donnie has played -/
def donnie_seasons : ℕ := 3

/-- Represents the number of games in each season -/
def games_per_season : ℕ := 50

/-- Represents the initial number of goals scored by Annie in her first game -/
def annie_initial_goals : ℕ := 2

/-- Represents the increase in Annie's goals per game -/
def annie_goal_increase : ℕ := 2

/-- Represents the number of seasons Annie has played -/
def annie_seasons : ℕ := 2

/-- Theorem stating that the total number of goals scored by all siblings is 11,344 -/
theorem total_goals_theorem :
  let louie_total := louie_last_match_goals + louie_previous_goals
  let donnie_total := 2 * louie_last_match_goals * donnie_seasons * games_per_season
  let annie_games := annie_seasons * games_per_season
  let annie_total := annie_games * (annie_initial_goals + annie_initial_goals + (annie_games - 1) * annie_goal_increase) / 2
  louie_total + donnie_total + annie_total = 11344 := by
  sorry

end total_goals_theorem_l3824_382428


namespace sum_of_max_min_g_l3824_382484

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 10|

-- Define the domain of x
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 10

-- Theorem statement
theorem sum_of_max_min_g : 
  ∃ (max min : ℝ), (∀ x, domain x → g x ≤ max) ∧ 
                    (∀ x, domain x → min ≤ g x) ∧ 
                    (max + min = 13) := by
  sorry

end sum_of_max_min_g_l3824_382484


namespace counterfeiters_payment_range_l3824_382401

/-- Represents a counterfeiter who can pay amounts between 1 and 25 rubles --/
structure Counterfeiter where
  pay : ℕ → ℕ
  pay_range : ∀ n, 1 ≤ pay n ∧ pay n ≤ 25

/-- The theorem states that three counterfeiters can collectively pay any amount from 100 to 200 rubles --/
theorem counterfeiters_payment_range (c1 c2 c3 : Counterfeiter) :
  ∀ n, 100 ≤ n ∧ n ≤ 200 → ∃ (x y z : ℕ), x + y + z = n ∧ 
    (∃ (a b c : ℕ), c1.pay a + c2.pay b + c3.pay c = x) ∧
    (∃ (d e f : ℕ), c1.pay d + c2.pay e + c3.pay f = y) ∧
    (∃ (g h i : ℕ), c1.pay g + c2.pay h + c3.pay i = z) :=
  sorry

end counterfeiters_payment_range_l3824_382401


namespace ellipse_foci_l3824_382474

/-- The equation of an ellipse in standard form -/
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 25 = 1

/-- The coordinates of a focus of the ellipse -/
def is_focus (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 3) ∨ (x = 0 ∧ y = -3)

/-- Theorem: The foci of the given ellipse are at (0, ±3) -/
theorem ellipse_foci :
  ∀ x y : ℝ, is_ellipse x y → (∃ fx fy : ℝ, is_focus fx fy ∧ 
    (x - fx)^2 + (y - fy)^2 = (x + fx)^2 + (y + fy)^2) :=
by sorry

end ellipse_foci_l3824_382474


namespace intersection_point_is_unique_l3824_382449

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-18/17, 46/17)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: 2y = -7x - 2 -/
def line2 (x y : ℚ) : Prop := 2 * y = -7 * x - 2

theorem intersection_point_is_unique :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = intersection_point :=
sorry

end intersection_point_is_unique_l3824_382449


namespace platform_length_l3824_382441

/-- Given a train of length 300 meters that takes 30 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is 200 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_cross_time = 30)
  (h3 : pole_cross_time = 18) :
  let platform_length := (train_length * platform_cross_time / pole_cross_time) - train_length
  platform_length = 200 := by
sorry

end platform_length_l3824_382441


namespace row_sum_1008_equals_2015_squared_l3824_382445

/-- Represents the sum of numbers in a row of the given pattern. -/
def row_sum (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The theorem stating that the sum of numbers in the 1008th row equals 2015². -/
theorem row_sum_1008_equals_2015_squared : row_sum 1008 = 2015 ^ 2 := by
  sorry

end row_sum_1008_equals_2015_squared_l3824_382445


namespace gcd_of_repeated_digit_ints_l3824_382490

/-- Represents a four-digit positive integer -/
def FourDigitInt := {n : ℕ // 1000 ≤ n ∧ n < 10000}

/-- Constructs an eight-digit integer by repeating a four-digit integer -/
def repeatFourDigits (n : FourDigitInt) : ℕ :=
  10000 * n.val + n.val

/-- The set of all eight-digit integers formed by repeating a four-digit integer -/
def RepeatedDigitInts : Set ℕ :=
  {m | ∃ n : FourDigitInt, m = repeatFourDigits n}

/-- Theorem stating that 10001 is the greatest common divisor of all eight-digit integers
    formed by repeating a four-digit integer -/
theorem gcd_of_repeated_digit_ints :
  ∃ d : ℕ, d > 0 ∧ (∀ m ∈ RepeatedDigitInts, d ∣ m) ∧
  (∀ d' : ℕ, d' > 0 → (∀ m ∈ RepeatedDigitInts, d' ∣ m) → d' ≤ d) ∧
  d = 10001 := by
sorry

end gcd_of_repeated_digit_ints_l3824_382490


namespace well_depth_rope_length_l3824_382400

/-- 
Given a well of unknown depth and a rope of unknown length, prove that if:
1) Folding the rope three times and lowering it into the well leaves 4 feet outside
2) Folding the rope four times and lowering it into the well leaves 1 foot outside
Then the depth of the well (x) and the length of the rope (h) satisfy the system of equations:
{h/3 = x + 4, h/4 = x + 1}
-/
theorem well_depth_rope_length (x h : ℝ) 
  (h_positive : h > 0) 
  (fold_three : h / 3 = x + 4) 
  (fold_four : h / 4 = x + 1) : 
  h / 3 = x + 4 ∧ h / 4 = x + 1 := by
sorry


end well_depth_rope_length_l3824_382400


namespace meters_in_one_kilometer_l3824_382455

/-- Conversion factor from kilometers to hectometers -/
def km_to_hm : ℝ := 5

/-- Conversion factor from hectometers to dekameters -/
def hm_to_dam : ℝ := 10

/-- Conversion factor from dekameters to meters -/
def dam_to_m : ℝ := 15

/-- The number of meters in one kilometer -/
def meters_in_km : ℝ := km_to_hm * hm_to_dam * dam_to_m

theorem meters_in_one_kilometer :
  meters_in_km = 750 := by sorry

end meters_in_one_kilometer_l3824_382455


namespace uncle_bradley_money_l3824_382471

theorem uncle_bradley_money (M : ℚ) (F H : ℕ) : 
  F + H = 13 →
  50 * F = (3 / 10) * M →
  100 * H = (7 / 10) * M →
  M = 1300 := by
sorry

end uncle_bradley_money_l3824_382471


namespace max_safe_caffeine_value_l3824_382494

/-- The maximum safe amount of caffeine one can consume per day -/
def max_safe_caffeine : ℕ := sorry

/-- The amount of caffeine in one energy drink (in mg) -/
def caffeine_per_drink : ℕ := 120

/-- The number of energy drinks Brandy consumes -/
def drinks_consumed : ℕ := 4

/-- The additional amount of caffeine Brandy can safely consume (in mg) -/
def additional_safe_caffeine : ℕ := 20

/-- Theorem stating the maximum safe amount of caffeine one can consume per day -/
theorem max_safe_caffeine_value : 
  max_safe_caffeine = caffeine_per_drink * drinks_consumed + additional_safe_caffeine := by
  sorry

end max_safe_caffeine_value_l3824_382494


namespace fourth_root_256_times_cube_root_8_times_sqrt_4_l3824_382470

theorem fourth_root_256_times_cube_root_8_times_sqrt_4 : 
  (256 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (4 : ℝ) ^ (1/2) = 8 := by
  sorry

end fourth_root_256_times_cube_root_8_times_sqrt_4_l3824_382470


namespace total_pies_count_l3824_382403

/-- The number of miniature pumpkin pies made by Pinky -/
def pinky_pies : ℕ := 147

/-- The number of miniature pumpkin pies made by Helen -/
def helen_pies : ℕ := 56

/-- The total number of miniature pumpkin pies -/
def total_pies : ℕ := pinky_pies + helen_pies

theorem total_pies_count : total_pies = 203 := by
  sorry

end total_pies_count_l3824_382403


namespace seating_arrangements_count_l3824_382426

/-- Represents a seating arrangement -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Represents a married couple -/
structure Couple := (husband : Fin 6) (wife : Fin 6)

/-- Represents a profession -/
def Profession := Fin 3

/-- Check if two positions are adjacent or opposite on a 12-seat round table -/
def isAdjacentOrOpposite (a b : Fin 12) : Prop := 
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a + 6 = b) ∨ (b + 6 = a)

/-- Check if a seating arrangement is valid -/
def isValidArrangement (s : SeatingArrangement) (couples : Fin 6 → Couple) (professions : Fin 12 → Profession) : Prop :=
  ∀ i j : Fin 12, 
    -- Men and women alternate
    (i.val % 2 = 0 ↔ j.val % 2 = 1) →
    -- No one sits next to or across from their spouse
    (∃ c : Fin 6, (couples c).husband = s i ∧ (couples c).wife = s j) →
    ¬ isAdjacentOrOpposite i j ∧
    -- No one sits next to someone of the same profession
    (isAdjacentOrOpposite i j → professions (s i) ≠ professions (s j))

/-- The main theorem stating the number of valid seating arrangements -/
theorem seating_arrangements_count :
  ∃ (arrangements : Finset SeatingArrangement) (couples : Fin 6 → Couple) (professions : Fin 12 → Profession),
    arrangements.card = 2880 ∧
    ∀ s ∈ arrangements, isValidArrangement s couples professions :=
sorry

end seating_arrangements_count_l3824_382426


namespace quadratic_function_evaluation_l3824_382435

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem quadratic_function_evaluation :
  3 * g 2 + 2 * g (-2) = 85 := by
  sorry

end quadratic_function_evaluation_l3824_382435


namespace first_last_gender_l3824_382464

/-- Represents the gender of a person in line -/
inductive Gender
  | Man
  | Woman

/-- Represents the state of bottle passing -/
structure BottlePassing where
  total_people : Nat
  woman_to_woman : Nat
  woman_to_man : Nat
  man_to_man : Nat

/-- Theorem stating the first and last person's gender based on bottle passing information -/
theorem first_last_gender (bp : BottlePassing) 
  (h1 : bp.total_people = 16)
  (h2 : bp.woman_to_woman = 4)
  (h3 : bp.woman_to_man = 3)
  (h4 : bp.man_to_man = 6) :
  (Gender.Woman, Gender.Man) = 
    (match bp.total_people with
      | 0 => (Gender.Woman, Gender.Man)  -- Arbitrary choice for empty line
      | n + 1 => 
        let first := if bp.woman_to_woman + bp.woman_to_man > bp.man_to_man + (n - (bp.woman_to_woman + bp.woman_to_man + bp.man_to_man)) 
                     then Gender.Woman else Gender.Man
        let last := if bp.man_to_man + (n - (bp.woman_to_woman + bp.woman_to_man + bp.man_to_man)) > bp.woman_to_woman + bp.woman_to_man 
                    then Gender.Man else Gender.Woman
        (first, last)
    ) :=
by
  sorry


end first_last_gender_l3824_382464


namespace baby_tarantula_legs_l3824_382404

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

/-- The number of tarantulas in one egg sac -/
def tarantulas_per_sac : ℕ := 1000

/-- The number of egg sacs being considered -/
def egg_sacs : ℕ := 5 - 1

/-- The total number of baby tarantula legs in one less than 5 egg sacs -/
def total_baby_legs : ℕ := egg_sacs * tarantulas_per_sac * tarantula_legs

theorem baby_tarantula_legs :
  total_baby_legs = 32000 := by sorry

end baby_tarantula_legs_l3824_382404


namespace calculate_F_of_5_f_6_l3824_382427

-- Define the functions f and F
def f (a : ℝ) : ℝ := a + 3
def F (a b : ℝ) : ℝ := b^3 - 2*a

-- State the theorem
theorem calculate_F_of_5_f_6 : F 5 (f 6) = 719 := by
  sorry

end calculate_F_of_5_f_6_l3824_382427


namespace fence_perimeter_is_106_l3824_382476

/-- Given a square field enclosed by posts, calculates the outer perimeter of the fence. -/
def fence_perimeter (num_posts : ℕ) (post_width : ℝ) (gap : ℝ) : ℝ :=
  let posts_per_side : ℕ := (num_posts - 4) / 4 + 2
  let gaps_per_side : ℕ := posts_per_side - 1
  let side_length : ℝ := gaps_per_side * gap + posts_per_side * post_width
  4 * side_length

/-- Theorem stating that the fence with given specifications has a perimeter of 106 feet. -/
theorem fence_perimeter_is_106 :
  fence_perimeter 16 0.5 6 = 106 := by
  sorry

#eval fence_perimeter 16 0.5 6

end fence_perimeter_is_106_l3824_382476


namespace unique_b_l3824_382479

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the properties of b
def b_properties (b : ℝ) : Prop :=
  b > 1 ∧ 
  (∀ x, x ∈ Set.Icc 1 b ↔ f x ∈ Set.Icc 1 b) ∧
  (∀ x, x ∈ Set.Icc 1 b → f x ≤ b)

-- Theorem statement
theorem unique_b : ∃! b, b_properties b ∧ b = 2 := by sorry

end unique_b_l3824_382479


namespace inequality_proof_l3824_382489

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hx1 : x < 1) (hy : 0 < y) (hy1 : y < 1) :
  (x^2 / (x + y)) + (y^2 / (1 - x)) + ((1 - x - y)^2 / (1 - y)) ≥ 1/2 := by
  sorry

end inequality_proof_l3824_382489


namespace ripe_oranges_count_l3824_382431

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges : ℕ := 25

/-- The difference between the number of sacks of ripe and unripe oranges -/
def difference : ℕ := 19

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges : ℕ := unripe_oranges + difference

theorem ripe_oranges_count : ripe_oranges = 44 := by
  sorry

end ripe_oranges_count_l3824_382431


namespace lanas_boxes_l3824_382440

/-- Given that each box contains 7 pieces of clothing and the total number of pieces is 21,
    prove that the number of boxes is 3. -/
theorem lanas_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (h1 : pieces_per_box = 7) (h2 : total_pieces = 21) :
  total_pieces / pieces_per_box = 3 := by
  sorry

end lanas_boxes_l3824_382440


namespace sum_of_coefficients_l3824_382429

-- Define the function f
def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

-- State the theorem
theorem sum_of_coefficients : f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by
  sorry

end sum_of_coefficients_l3824_382429


namespace opposites_sum_to_zero_l3824_382448

theorem opposites_sum_to_zero (a b : ℚ) (h : a = -b) : a + b = 0 := by
  sorry

end opposites_sum_to_zero_l3824_382448


namespace right_triangle_consecutive_prime_angles_l3824_382424

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def consecutive_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p < q ∧ ∀ m, p < m → m < q → ¬is_prime m

theorem right_triangle_consecutive_prime_angles (p q : ℕ) :
  p < q →
  consecutive_primes p q →
  p + q = 90 →
  (∀ p' q' : ℕ, p' < q' → consecutive_primes p' q' → p' + q' = 90 → p ≤ p') →
  p = 43 := by sorry

end right_triangle_consecutive_prime_angles_l3824_382424


namespace city_rentals_cost_per_mile_l3824_382434

/-- Proves that the cost per mile for City Rentals is $0.16 given the rental rates and equal cost for 48.0 miles. -/
theorem city_rentals_cost_per_mile :
  let sunshine_daily_rate : ℝ := 17.99
  let sunshine_per_mile : ℝ := 0.18
  let city_daily_rate : ℝ := 18.95
  let miles : ℝ := 48.0
  ∀ city_per_mile : ℝ,
    sunshine_daily_rate + sunshine_per_mile * miles = city_daily_rate + city_per_mile * miles →
    city_per_mile = 0.16 := by
  sorry

end city_rentals_cost_per_mile_l3824_382434


namespace increasing_f_implies_a_in_range_l3824_382491

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then x^3 - 2*a*x + 1 else (a-1)^x - 7

theorem increasing_f_implies_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 < a ∧ a ≤ 3 := by
  sorry

end increasing_f_implies_a_in_range_l3824_382491


namespace digital_root_of_8_pow_1989_l3824_382480

def digital_root (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

theorem digital_root_of_8_pow_1989 :
  digital_root (8^1989) = 8 :=
sorry

end digital_root_of_8_pow_1989_l3824_382480


namespace sun_city_has_12000_people_l3824_382460

/-- The population of Willowdale City -/
def willowdale_population : ℕ := 2000

/-- The population of Roseville City -/
def roseville_population : ℕ := 3 * willowdale_population - 500

/-- The population of Sun City -/
def sun_city_population : ℕ := 2 * roseville_population + 1000

/-- Theorem stating that Sun City has 12000 people -/
theorem sun_city_has_12000_people : sun_city_population = 12000 := by
  sorry

end sun_city_has_12000_people_l3824_382460


namespace hyperbola_construction_equivalence_l3824_382412

/-- The equation of a hyperbola in standard form -/
def is_hyperbola_point (a b x y : ℝ) : Prop :=
  (x / a)^2 - (y / b)^2 = 1

/-- The construction equation for a point on the hyperbola -/
def satisfies_construction (a b x y : ℝ) : Prop :=
  x = (a / b) * Real.sqrt (b^2 + y^2)

/-- Theorem: Any point satisfying the hyperbola equation also satisfies the construction equation -/
theorem hyperbola_construction_equivalence (a b x y : ℝ) (ha : a > 0) (hb : b > 0) :
  is_hyperbola_point a b x y → satisfies_construction a b x y :=
by sorry

end hyperbola_construction_equivalence_l3824_382412


namespace expansion_activity_optimal_time_l3824_382407

theorem expansion_activity_optimal_time :
  ∀ (x y : ℕ),
    x + y = 15 →
    x = 2 * y - 3 →
    ∀ (m : ℕ),
      m ≤ 10 →
      10 - m > (m / 2) →
      6 * m + 8 * (10 - m) ≥ 68 :=
by
  sorry

end expansion_activity_optimal_time_l3824_382407


namespace complex_equation_solution_l3824_382485

def i : ℂ := Complex.I

theorem complex_equation_solution (x : ℝ) (h : (1 - 2*i) * (x + i) = 4 - 3*i) : x = 2 := by
  sorry

end complex_equation_solution_l3824_382485


namespace no_perfect_cubes_l3824_382422

theorem no_perfect_cubes (a b : ℤ) : ¬(∃ x y : ℤ, a^5*b + 3 = x^3 ∧ a*b^5 + 3 = y^3) := by
  sorry

end no_perfect_cubes_l3824_382422


namespace intersection_of_A_and_B_l3824_382446

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x < 0}

-- Define set B
def B : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1/3 ≤ x ∧ x < 4} :=
by sorry

end intersection_of_A_and_B_l3824_382446
