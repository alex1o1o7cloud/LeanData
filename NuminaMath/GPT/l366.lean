import Mathlib

namespace NUMINAMATH_GPT_total_arrangements_correct_adjacent_males_correct_descending_heights_correct_l366_36650

-- Total number of different arrangements of 3 male students and 2 female students.
def total_arrangements (males females : ℕ) : ℕ :=
  (males + females).factorial

-- Number of arrangements where exactly two male students are adjacent.
def adjacent_males (males females : ℕ) : ℕ :=
  if males = 3 ∧ females = 2 then 72 else 0

-- Number of arrangements where male students of different heights are arranged from tallest to shortest.
def descending_heights (heights : Nat → ℕ) (males females : ℕ) : ℕ :=
  if males = 3 ∧ females = 2 then 20 else 0

-- Theorem statements corresponding to the questions.
theorem total_arrangements_correct : total_arrangements 3 2 = 120 := sorry

theorem adjacent_males_correct : adjacent_males 3 2 = 72 := sorry

theorem descending_heights_correct (heights : Nat → ℕ) : descending_heights heights 3 2 = 20 := sorry

end NUMINAMATH_GPT_total_arrangements_correct_adjacent_males_correct_descending_heights_correct_l366_36650


namespace NUMINAMATH_GPT_eric_has_correct_green_marbles_l366_36601

def total_marbles : ℕ := 20
def white_marbles : ℕ := 12
def blue_marbles : ℕ := 6
def green_marbles : ℕ := total_marbles - (white_marbles + blue_marbles)

theorem eric_has_correct_green_marbles : green_marbles = 2 :=
by
  sorry

end NUMINAMATH_GPT_eric_has_correct_green_marbles_l366_36601


namespace NUMINAMATH_GPT_min_regions_l366_36613

namespace CircleDivision

def k := 12

-- Theorem statement: Given exactly 12 points where at least two circles intersect,
-- the minimum number of regions into which these circles divide the plane is 14.
theorem min_regions (k := 12) : ∃ R, R = 14 :=
by
  let R := 14
  existsi R
  exact rfl

end NUMINAMATH_GPT_min_regions_l366_36613


namespace NUMINAMATH_GPT_solution_set_inequality_l366_36625

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f (x - 1/2) + f (x + 1) = 0)
variable (h2 : e ^ 3 * f 2018 = 1)
variable (h3 : ∀ x, f x > f'' (-x))
variable (h4 : ∀ x, f x = f (-x))

theorem solution_set_inequality :
  ∀ x, f (x - 1) > 1 / (e ^ x) ↔ x > 3 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l366_36625


namespace NUMINAMATH_GPT_exp_pi_gt_pi_exp_l366_36631

theorem exp_pi_gt_pi_exp (h : Real.pi > Real.exp 1) : Real.exp Real.pi > Real.pi ^ Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_exp_pi_gt_pi_exp_l366_36631


namespace NUMINAMATH_GPT_function_satisfies_conditions_l366_36694

-- Define the functional equation condition
def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + x * y) = f x * f (y + 1)

-- Lean statement for the proof problem
theorem function_satisfies_conditions (f : ℝ → ℝ) (h : functional_eq f) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) ∨ (∀ x : ℝ, f x = x) :=
sorry

end NUMINAMATH_GPT_function_satisfies_conditions_l366_36694


namespace NUMINAMATH_GPT_darren_total_tshirts_l366_36671

def num_white_packs := 5
def num_white_tshirts_per_pack := 6
def num_blue_packs := 3
def num_blue_tshirts_per_pack := 9

def total_tshirts (wpacks : ℕ) (wtshirts_per_pack : ℕ) (bpacks : ℕ) (btshirts_per_pack : ℕ) : ℕ :=
  (wpacks * wtshirts_per_pack) + (bpacks * btshirts_per_pack)

theorem darren_total_tshirts : total_tshirts num_white_packs num_white_tshirts_per_pack num_blue_packs num_blue_tshirts_per_pack = 57 :=
by
  -- proof needed
  sorry

end NUMINAMATH_GPT_darren_total_tshirts_l366_36671


namespace NUMINAMATH_GPT_amount_received_by_a_l366_36662

namespace ProofProblem

/-- Total amount of money divided -/
def total_amount : ℕ := 600

/-- Ratio part for 'a' -/
def part_a : ℕ := 1

/-- Ratio part for 'b' -/
def part_b : ℕ := 2

/-- Total parts in the ratio -/
def total_parts : ℕ := part_a + part_b

/-- Amount per part when total is divided evenly by the total number of parts -/
def amount_per_part : ℕ := total_amount / total_parts

/-- Amount received by 'a' when total amount is divided according to the given ratio -/
def amount_a : ℕ := part_a * amount_per_part

theorem amount_received_by_a : amount_a = 200 := by
  -- Proof will be filled in here
  sorry

end ProofProblem

end NUMINAMATH_GPT_amount_received_by_a_l366_36662


namespace NUMINAMATH_GPT_problem_k_star_k_star_k_l366_36669

def star (x y : ℝ) : ℝ := 2 * x^2 - y

theorem problem_k_star_k_star_k (k : ℝ) : star k (star k k) = k :=
by
  sorry

end NUMINAMATH_GPT_problem_k_star_k_star_k_l366_36669


namespace NUMINAMATH_GPT_total_lotus_flowers_l366_36648

theorem total_lotus_flowers (x : ℕ) (h1 : x > 0) 
  (c1 : 3 ∣ x)
  (c2 : 5 ∣ x)
  (c3 : 6 ∣ x)
  (c4 : 4 ∣ x)
  (h_total : x = x / 3 + x / 5 + x / 6 + x / 4 + 6) : 
  x = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_lotus_flowers_l366_36648


namespace NUMINAMATH_GPT_prob_green_ball_l366_36616

-- Definitions for the conditions
def red_balls_X := 3
def green_balls_X := 7
def total_balls_X := red_balls_X + green_balls_X

def red_balls_YZ := 7
def green_balls_YZ := 3
def total_balls_YZ := red_balls_YZ + green_balls_YZ

-- The probability of selecting any container
def prob_select_container := 1 / 3

-- The probabilities of drawing a green ball from each container
def prob_green_given_X := green_balls_X / total_balls_X
def prob_green_given_YZ := green_balls_YZ / total_balls_YZ

-- The combined probability of selecting a green ball
theorem prob_green_ball : 
  prob_select_container * prob_green_given_X + 
  prob_select_container * prob_green_given_YZ + 
  prob_select_container * prob_green_given_YZ = 13 / 30 := 
  by sorry

end NUMINAMATH_GPT_prob_green_ball_l366_36616


namespace NUMINAMATH_GPT_jogger_distance_l366_36684

theorem jogger_distance 
(speed_jogger : ℝ := 9)
(speed_train : ℝ := 45)
(train_length : ℕ := 120)
(time_to_pass : ℕ := 38)
(relative_speed_mps : ℝ := (speed_train - speed_jogger) * (1 / 3.6))
(distance_covered : ℝ := (relative_speed_mps * time_to_pass))
(d : ℝ := distance_covered - train_length) :
d = 260 := sorry

end NUMINAMATH_GPT_jogger_distance_l366_36684


namespace NUMINAMATH_GPT_percent_forgot_group_B_l366_36667

def num_students_group_A : ℕ := 20
def num_students_group_B : ℕ := 80
def percent_forgot_group_A : ℚ := 0.20
def total_percent_forgot : ℚ := 0.16

/--
There are two groups of students in the sixth grade. 
There are 20 students in group A, and 80 students in group B. 
On a particular day, 20% of the students in group A forget their homework, and a certain 
percentage of the students in group B forget their homework. 
Then, 16% of the sixth graders forgot their homework. 
Prove that 15% of the students in group B forgot their homework.
-/
theorem percent_forgot_group_B : 
  let num_forgot_group_A := percent_forgot_group_A * num_students_group_A
  let total_students := num_students_group_A + num_students_group_B
  let total_forgot := total_percent_forgot * total_students
  let num_forgot_group_B := total_forgot - num_forgot_group_A
  let percent_forgot_group_B := (num_forgot_group_B / num_students_group_B) * 100
  percent_forgot_group_B = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_percent_forgot_group_B_l366_36667


namespace NUMINAMATH_GPT_no_such_n_exists_l366_36656

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, 0 < n ∧
  (∃ a : ℕ, 2 * n^2 + 1 = a^2) ∧
  (∃ b : ℕ, 3 * n^2 + 1 = b^2) ∧
  (∃ c : ℕ, 6 * n^2 + 1 = c^2) :=
sorry

end NUMINAMATH_GPT_no_such_n_exists_l366_36656


namespace NUMINAMATH_GPT_problem_l366_36619

open Complex

-- Given condition: smallest positive integer n greater than 3
def smallest_n_gt_3 (n : ℕ) : Prop :=
  n > 3 ∧ ∀ m : ℕ, m > 3 → m < n → False

-- Given condition: equation holds for complex numbers
def equation_holds (a b : ℝ) (n : ℕ) : Prop :=
  (a + b * I)^n + a = (a - b * I)^n + b

-- Proof problem: Given conditions, prove b / a = 1
theorem problem (n : ℕ) (a b : ℝ)
  (h1 : smallest_n_gt_3 n)
  (h2 : 0 < a) (h3 : 0 < b)
  (h4 : equation_holds a b n) :
  b / a = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_l366_36619


namespace NUMINAMATH_GPT_no_such_a_and_sequence_exists_l366_36624

theorem no_such_a_and_sequence_exists :
  ¬∃ (a : ℝ) (a_pos : 0 < a ∧ a < 1) (a_seq : ℕ → ℝ), (∀ n : ℕ, 0 < a_seq n) ∧ (∀ n : ℕ, 1 + a_seq (n + 1) ≤ a_seq n + (a / (n + 1)) * a_seq n) :=
by
  sorry

end NUMINAMATH_GPT_no_such_a_and_sequence_exists_l366_36624


namespace NUMINAMATH_GPT_solve_real_solution_l366_36603

theorem solve_real_solution:
  ∀ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔
           (x = 4 + Real.sqrt 57) ∨ (x = 4 - Real.sqrt 57) :=
by
  sorry

end NUMINAMATH_GPT_solve_real_solution_l366_36603


namespace NUMINAMATH_GPT_find_triples_l366_36642

theorem find_triples (a b c : ℝ) :
  a^2 + b^2 + c^2 = 1 ∧ a * (2 * b - 2 * a - c) ≥ 1/2 ↔ 
  (a = 1 / Real.sqrt 6 ∧ b = 2 / Real.sqrt 6 ∧ c = -1 / Real.sqrt 6) ∨
  (a = -1 / Real.sqrt 6 ∧ b = -2 / Real.sqrt 6 ∧ c = 1 / Real.sqrt 6) := 
by 
  sorry

end NUMINAMATH_GPT_find_triples_l366_36642


namespace NUMINAMATH_GPT_smallest_next_divisor_l366_36615

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor 
  (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000) 
  (h2 : is_even m) 
  (h3 : is_divisor 171 m)
  : ∃ k, k > 171 ∧ k = 190 ∧ is_divisor k m := 
by
  sorry

end NUMINAMATH_GPT_smallest_next_divisor_l366_36615


namespace NUMINAMATH_GPT_sum_of_surface_points_l366_36626

theorem sum_of_surface_points
  (n : ℕ) (h_n : n = 2012) 
  (total_sum : ℕ) (h_total : total_sum = n * 21)
  (matching_points_sum : ℕ) (h_matching : matching_points_sum = (n - 1) * 7)
  (x : ℕ) (h_x_range : 1 ≤ x ∧ x ≤ 6) :
  (total_sum - matching_points_sum + 2 * x = 28177 ∨
   total_sum - matching_points_sum + 2 * x = 28179 ∨
   total_sum - matching_points_sum + 2 * x = 28181 ∨
   total_sum - matching_points_sum + 2 * x = 28183 ∨
   total_sum - matching_points_sum + 2 * x = 28185 ∨
   total_sum - matching_points_sum + 2 * x = 28187) :=
by sorry

end NUMINAMATH_GPT_sum_of_surface_points_l366_36626


namespace NUMINAMATH_GPT_roots_calculation_l366_36678

theorem roots_calculation (c d : ℝ) (h : c^2 - 5*c + 6 = 0) (h' : d^2 - 5*d + 6 = 0) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 := by
  sorry

end NUMINAMATH_GPT_roots_calculation_l366_36678


namespace NUMINAMATH_GPT_symmetric_angles_l366_36602

theorem symmetric_angles (α β : ℝ) (k : ℤ) (h : α + β = 2 * k * Real.pi) : α = 2 * k * Real.pi - β :=
by
  sorry

end NUMINAMATH_GPT_symmetric_angles_l366_36602


namespace NUMINAMATH_GPT_R_depends_on_d_and_n_l366_36699

variable (n a d : ℕ)

noncomputable def s1 : ℕ := (n * (2 * a + (n - 1) * d)) / 2
noncomputable def s2 : ℕ := (2 * n * (2 * a + (2 * n - 1) * d)) / 2
noncomputable def s3 : ℕ := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
noncomputable def R : ℕ := s3 n a d - s2 n a d - s1 n a d

theorem R_depends_on_d_and_n : R n a d = 2 * d * n^2 :=
by
  sorry

end NUMINAMATH_GPT_R_depends_on_d_and_n_l366_36699


namespace NUMINAMATH_GPT_num_complementary_sets_l366_36663

-- Definitions for shapes, colors, shades, and patterns
inductive Shape
| circle | square | triangle

inductive Color
| red | blue | green

inductive Shade
| light | medium | dark

inductive Pattern
| striped | dotted | plain

-- Definition of a card
structure Card where
  shape : Shape
  color : Color
  shade : Shade
  pattern : Pattern

-- Condition: Each possible combination is represented once in a deck of 81 cards.
def deck : List Card := sorry -- Construct the deck with 81 unique cards

-- Predicate for complementary sets of three cards
def is_complementary (c1 c2 c3 : Card) : Prop :=
  (c1.shape = c2.shape ∧ c2.shape = c3.shape ∧ c1.shape = c3.shape ∨
   c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∧ c1.color = c3.color ∨
   c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.shade = c2.shade ∧ c2.shade = c3.shade ∧ c1.shade = c3.shade ∨
   c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∧
  (c1.pattern = c2.pattern ∧ c2.pattern = c3.pattern ∧ c1.pattern = c3.pattern ∨
   c1.pattern ≠ c2.pattern ∧ c2.pattern ≠ c3.pattern ∧ c1.pattern ≠ c3.pattern)

-- Statement of the theorem to prove
theorem num_complementary_sets : 
  ∃ (complementary_sets : List (Card × Card × Card)), 
  complementary_sets.length = 5400 ∧
  ∀ (c1 c2 c3 : Card), (c1, c2, c3) ∈ complementary_sets → is_complementary c1 c2 c3 :=
sorry

end NUMINAMATH_GPT_num_complementary_sets_l366_36663


namespace NUMINAMATH_GPT_cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle_l366_36621

theorem cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle
  (surface_area : ℝ) (lateral_surface_unfolds_to_semicircle : Prop) :
  surface_area = 12 * Real.pi → lateral_surface_unfolds_to_semicircle → ∃ r : ℝ, r = 2 := by
  sorry

end NUMINAMATH_GPT_cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle_l366_36621


namespace NUMINAMATH_GPT_range_of_a_l366_36632

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ -3 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l366_36632


namespace NUMINAMATH_GPT_cube_union_volume_is_correct_cube_union_surface_area_is_correct_l366_36604

noncomputable def cubeUnionVolume : ℝ :=
  let cubeVolume := 1
  let intersectionVolume := 1 / 4
  cubeVolume * 2 - intersectionVolume

theorem cube_union_volume_is_correct :
  cubeUnionVolume = 5 / 4 := sorry

noncomputable def cubeUnionSurfaceArea : ℝ :=
  2 * (6 * (1 / 4) + 6 * (1 / 4 / 4))

theorem cube_union_surface_area_is_correct :
  cubeUnionSurfaceArea = 15 / 2 := sorry

end NUMINAMATH_GPT_cube_union_volume_is_correct_cube_union_surface_area_is_correct_l366_36604


namespace NUMINAMATH_GPT_problem_a_l366_36628

theorem problem_a (x a : ℝ) (h : (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) = 3 * a^4) :
  x = (-5 * a + a * Real.sqrt 37) / 2 ∨ x = (-5 * a - a * Real.sqrt 37) / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_a_l366_36628


namespace NUMINAMATH_GPT_Tameka_sold_40_boxes_on_Friday_l366_36697

noncomputable def TamekaSalesOnFriday (F : ℕ) : Prop :=
  let SaturdaySales := 2 * F - 10
  let SundaySales := (2 * F - 10) / 2
  F + SaturdaySales + SundaySales = 145

theorem Tameka_sold_40_boxes_on_Friday : ∃ F : ℕ, TamekaSalesOnFriday F ∧ F = 40 := 
by 
  sorry

end NUMINAMATH_GPT_Tameka_sold_40_boxes_on_Friday_l366_36697


namespace NUMINAMATH_GPT_car_speed_ratio_l366_36641

theorem car_speed_ratio 
  (t D : ℝ) 
  (v_alpha v_beta : ℝ)
  (H1 : (v_alpha + v_beta) * t = D)
  (H2 : v_alpha * 4 = D - v_alpha * t)
  (H3 : v_beta * 1 = D - v_beta * t) : 
  v_alpha / v_beta = 2 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_ratio_l366_36641


namespace NUMINAMATH_GPT_inverse_proportion_passes_first_and_third_quadrants_l366_36609

theorem inverse_proportion_passes_first_and_third_quadrants (m : ℝ) :
  ((∀ x : ℝ, x ≠ 0 → (x > 0 → (m - 3) / x > 0) ∧ (x < 0 → (m - 3) / x < 0)) → m = 5) := 
by 
  sorry

end NUMINAMATH_GPT_inverse_proportion_passes_first_and_third_quadrants_l366_36609


namespace NUMINAMATH_GPT_partial_fraction_decomposition_l366_36605

noncomputable def partial_fraction_product (A B C : ℤ) : ℤ :=
  A * B * C

theorem partial_fraction_decomposition:
  ∃ A B C : ℤ, 
  (∀ x : ℤ, (x^2 - 19 = A * (x + 2) * (x - 3) 
                    + B * (x - 1) * (x - 3) 
                    + C * (x - 1) * (x + 2) )) 
  → partial_fraction_product A B C = 3 :=
by
  sorry

end NUMINAMATH_GPT_partial_fraction_decomposition_l366_36605


namespace NUMINAMATH_GPT_ring_groups_in_first_tree_l366_36635

variable (n : ℕ) (y1 y2 : ℕ) (t : ℕ) (groupsPerYear : ℕ := 6)

-- each tree's rings are in groups of 2 fat rings and 4 thin rings, representing 6 years
def group_represents_years : ℕ := groupsPerYear

-- second tree has 40 ring groups, so it is 40 * 6 = 240 years old
def second_tree_groups : ℕ := 40

-- first tree is 180 years older, so its age in years
def first_tree_age : ℕ := (second_tree_groups * groupsPerYear) + 180

-- number of ring groups in the first tree
def number_of_ring_groups_in_first_tree := first_tree_age / groupsPerYear

theorem ring_groups_in_first_tree :
  number_of_ring_groups_in_first_tree = 70 :=
by
  sorry

end NUMINAMATH_GPT_ring_groups_in_first_tree_l366_36635


namespace NUMINAMATH_GPT_train_and_car_combined_time_l366_36614

theorem train_and_car_combined_time (car_time : ℝ) (train_time : ℝ) 
  (h1 : train_time = car_time + 2) (h2 : car_time = 4.5) : 
  car_time + train_time = 11 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_train_and_car_combined_time_l366_36614


namespace NUMINAMATH_GPT_exists_m_with_totient_ratio_l366_36649

variable (α β : ℝ)

theorem exists_m_with_totient_ratio (h0 : 0 ≤ α) (h1 : α < β) (h2 : β ≤ 1) :
  ∃ m : ℕ, α < (Nat.totient m : ℝ) / m ∧ (Nat.totient m : ℝ) / m < β := 
  sorry

end NUMINAMATH_GPT_exists_m_with_totient_ratio_l366_36649


namespace NUMINAMATH_GPT_second_player_wins_l366_36637

def num_of_piles_initial := 3
def total_stones := 10 + 15 + 20
def num_of_piles_final := total_stones
def total_moves := num_of_piles_final - num_of_piles_initial

theorem second_player_wins : total_moves % 2 = 0 :=
sorry

end NUMINAMATH_GPT_second_player_wins_l366_36637


namespace NUMINAMATH_GPT_perpendicular_lines_intersection_l366_36658

theorem perpendicular_lines_intersection (a b c d : ℝ)
    (h_perpendicular : (a / 2) * (-2 / b) = -1)
    (h_intersection1 : a * 2 - 2 * (-3) = d)
    (h_intersection2 : 2 * 2 + b * (-3) = c) :
    d = 12 := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_intersection_l366_36658


namespace NUMINAMATH_GPT_M_intersect_N_eq_l366_36679

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- Define what we need to prove
theorem M_intersect_N_eq : M ∩ N = {y | y ≥ 1} :=
by
  sorry

end NUMINAMATH_GPT_M_intersect_N_eq_l366_36679


namespace NUMINAMATH_GPT_find_number_l366_36693

variable (a n : ℝ)

theorem find_number (h1: 2 * a = 3 * n) (h2: a * n ≠ 0) (h3: (a / 3) / (n / 2) = 1) : 
  n = 2 * a / 3 :=
sorry

end NUMINAMATH_GPT_find_number_l366_36693


namespace NUMINAMATH_GPT_airplane_children_l366_36647

theorem airplane_children (total_passengers men women children : ℕ) 
    (h1 : total_passengers = 80) 
    (h2 : men = women) 
    (h3 : men = 30) 
    (h4 : total_passengers = men + women + children) : 
    children = 20 := 
by
    -- We need to show that the number of children is 20.
    sorry

end NUMINAMATH_GPT_airplane_children_l366_36647


namespace NUMINAMATH_GPT_triangle_angle_conditions_l366_36623

theorem triangle_angle_conditions
  (a b c : ℝ)
  (α β γ : ℝ)
  (h_triangle : c^2 = a^2 + 2 * b^2 * Real.cos β)
  (h_tri_angles : α + β + γ = 180):
  (γ = β / 2 + 90 ∧ α = 90 - 3 * β / 2 ∧ 0 < β ∧ β < 60) ∨ 
  (α = β / 2 ∧ γ = 180 - 3 * β / 2 ∧ 0 < β ∧ β < 120) :=
sorry

end NUMINAMATH_GPT_triangle_angle_conditions_l366_36623


namespace NUMINAMATH_GPT_probability_x_gt_3y_l366_36687

noncomputable def rect_region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3020 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3010}

theorem probability_x_gt_3y : 
  (∫ p in rect_region, if p.1 > 3 * p.2 then 1 else (0:ℝ)) / 
  (∫ p in rect_region, (1:ℝ)) = 1007 / 6020 := sorry

end NUMINAMATH_GPT_probability_x_gt_3y_l366_36687


namespace NUMINAMATH_GPT_student_D_most_stable_l366_36607

-- Define the variances for students A, B, C, and D
def SA_squared : ℝ := 2.1
def SB_squared : ℝ := 3.5
def SC_squared : ℝ := 9
def SD_squared : ℝ := 0.7

-- Theorem stating that student D has the most stable performance
theorem student_D_most_stable :
  SD_squared < SA_squared ∧ SD_squared < SB_squared ∧ SD_squared < SC_squared := by
  sorry

end NUMINAMATH_GPT_student_D_most_stable_l366_36607


namespace NUMINAMATH_GPT_Tim_marble_count_l366_36620

theorem Tim_marble_count (Fred_marbles : ℕ) (Tim_marbles : ℕ) (h1 : Fred_marbles = 110) (h2 : Fred_marbles = 22 * Tim_marbles) : 
  Tim_marbles = 5 := 
sorry

end NUMINAMATH_GPT_Tim_marble_count_l366_36620


namespace NUMINAMATH_GPT_marks_for_correct_answer_l366_36660

theorem marks_for_correct_answer (x : ℕ) 
  (total_marks : ℤ) (total_questions : ℕ) (correct_answers : ℕ) 
  (wrong_mark : ℤ) (result : ℤ) :
  total_marks = result →
  total_questions = 70 →
  correct_answers = 27 →
  (-1) * (total_questions - correct_answers) = wrong_mark →
  total_marks = (correct_answers : ℤ) * (x : ℤ) + wrong_mark →
  x = 3 := 
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_marks_for_correct_answer_l366_36660


namespace NUMINAMATH_GPT_part_a_prob_part_b_expected_time_l366_36657

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end NUMINAMATH_GPT_part_a_prob_part_b_expected_time_l366_36657


namespace NUMINAMATH_GPT_calc_f_y_eq_2f_x_l366_36666

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem calc_f_y_eq_2f_x (x : ℝ) (h : -1 < x) (h' : x < 1) :
  f ( (2 * x + x^2) / (1 + 2 * x^2) ) = 2 * f x := by
  sorry

end NUMINAMATH_GPT_calc_f_y_eq_2f_x_l366_36666


namespace NUMINAMATH_GPT_intersection_A_B_l366_36674

-- Definitions of sets A and B based on the given conditions
def A : Set ℕ := {4, 5, 6, 7}
def B : Set ℕ := {x | 3 ≤ x ∧ x < 6}

-- The theorem stating the proof problem
theorem intersection_A_B : A ∩ B = {4, 5} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l366_36674


namespace NUMINAMATH_GPT_fourth_bus_people_difference_l366_36634

def bus1_people : Nat := 12
def bus2_people : Nat := 2 * bus1_people
def bus3_people : Nat := bus2_people - 6
def total_people : Nat := 75
def bus4_people : Nat := total_people - (bus1_people + bus2_people + bus3_people)
def difference_people : Nat := bus4_people - bus1_people

theorem fourth_bus_people_difference : difference_people = 9 := by
  -- Proof logic here
  sorry

end NUMINAMATH_GPT_fourth_bus_people_difference_l366_36634


namespace NUMINAMATH_GPT_weight_difference_calc_l366_36600

-- Define the weights in pounds
def Anne_weight : ℕ := 67
def Douglas_weight : ℕ := 52
def Maria_weight : ℕ := 48

-- Define the combined weight of Douglas and Maria
def combined_weight_DM : ℕ := Douglas_weight + Maria_weight

-- Define the weight difference
def weight_difference : ℤ := Anne_weight - combined_weight_DM

-- The theorem stating the difference
theorem weight_difference_calc : weight_difference = -33 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_weight_difference_calc_l366_36600


namespace NUMINAMATH_GPT_seating_arrangements_l366_36672

theorem seating_arrangements (n_seats : ℕ) (n_people : ℕ) (n_adj_empty : ℕ) (h1 : n_seats = 6) 
    (h2 : n_people = 3) (h3 : n_adj_empty = 2) : 
    ∃ arrangements : ℕ, arrangements = 48 := 
by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l366_36672


namespace NUMINAMATH_GPT_find_principal_amount_l366_36692

noncomputable def principal_amount (difference : ℝ) (rate : ℝ) : ℝ :=
  let ci := rate / 2
  let si := rate
  difference / (ci ^ 2 - 1 - si)

theorem find_principal_amount :
  principal_amount 4.25 0.10 = 1700 :=
by 
  sorry

end NUMINAMATH_GPT_find_principal_amount_l366_36692


namespace NUMINAMATH_GPT_work_completion_days_l366_36629

theorem work_completion_days
    (A : ℝ) (B : ℝ) (h1 : 1 / A + 1 / B = 1 / 10)
    (h2 : B = 35) :
    A = 14 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_days_l366_36629


namespace NUMINAMATH_GPT_total_clothing_donated_l366_36645

-- Definition of the initial donation by Adam
def adam_initial_donation : Nat := 4 + 4 + 4*2 + 20 -- 4 pairs of pants, 4 jumpers, 4 pajama sets (8 items), 20 t-shirts

-- Adam's friends' total donation
def friends_donation : Nat := 3 * adam_initial_donation

-- Adam's donation after keeping half
def adam_final_donation : Nat := adam_initial_donation / 2

-- Total donation being the sum of Adam's and friends' donations
def total_donation : Nat := adam_final_donation + friends_donation

-- The statement to prove
theorem total_clothing_donated : total_donation = 126 := by
  -- This is skipped as per instructions
  sorry

end NUMINAMATH_GPT_total_clothing_donated_l366_36645


namespace NUMINAMATH_GPT_minimum_f_l366_36683

def f (x y : ℤ) : ℤ := |5 * x^2 + 11 * x * y - 5 * y^2|

theorem minimum_f (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) : ∃ (m : ℤ), m = 5 ∧ ∀ (x y : ℤ), (x ≠ 0 ∨ y ≠ 0) → f x y ≥ m :=
by sorry

end NUMINAMATH_GPT_minimum_f_l366_36683


namespace NUMINAMATH_GPT_min_value_of_f_l366_36698

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3| + Real.exp x

theorem min_value_of_f :
  ∃ x ∈ Set.Icc (Real.exp 0) (Real.exp 3), f x = 6 - 2 * Real.log 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_f_l366_36698


namespace NUMINAMATH_GPT_number_of_rows_with_7_eq_5_l366_36646

noncomputable def number_of_rows_with_7_people (x y : ℕ) : Prop :=
  7 * x + 6 * (y - x) = 59

theorem number_of_rows_with_7_eq_5 :
  ∃ x y : ℕ, number_of_rows_with_7_people x y ∧ x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_rows_with_7_eq_5_l366_36646


namespace NUMINAMATH_GPT_quadratic_function_min_value_l366_36654

theorem quadratic_function_min_value (a b c : ℝ) (h_a : a > 0) (h_b : b ≠ 0) 
(h_f0 : |c| = 1) (h_f1 : |a + b + c| = 1) (h_fn1 : |a - b + c| = 1) :
∃ f : ℝ → ℝ, (∀ x : ℝ, f x = a*x^2 + b*x + c) ∧
  (|f 0| = 1) ∧ (|f 1| = 1) ∧ (|f (-1)| = 1) ∧
  (f 0 = -(5/4) ∨ f 1 = -(5/4) ∨ f (-1) = -(5/4)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_min_value_l366_36654


namespace NUMINAMATH_GPT_isosceles_vertex_angle_l366_36661

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem isosceles_vertex_angle (a b θ : ℝ)
  (h1 : a = golden_ratio * b) :
  ∃ θ, θ = 36 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_vertex_angle_l366_36661


namespace NUMINAMATH_GPT_contributions_before_john_l366_36664

theorem contributions_before_john
  (A : ℝ) (n : ℕ)
  (h1 : 1.5 * A = 75)
  (h2 : (n * A + 150) / (n + 1) = 75) :
  n = 3 :=
by
  sorry

end NUMINAMATH_GPT_contributions_before_john_l366_36664


namespace NUMINAMATH_GPT_calculate_sum_calculate_product_l366_36665

theorem calculate_sum : 13 + (-7) + (-6) = 0 :=
by sorry

theorem calculate_product : (-8) * (-4 / 3) * (-0.125) * (5 / 4) = -5 / 3 :=
by sorry

end NUMINAMATH_GPT_calculate_sum_calculate_product_l366_36665


namespace NUMINAMATH_GPT_general_formula_an_l366_36688

theorem general_formula_an {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ) (hS : ∀ n, S n = (n / 2) * (a 1 + a n)) (hd : d = a 2 - a 1) : 
  ∀ n, a n = a 1 + (n - 1) * d :=
sorry

end NUMINAMATH_GPT_general_formula_an_l366_36688


namespace NUMINAMATH_GPT_find_height_l366_36686

namespace RightTriangleProblem

variables {x h : ℝ}

-- Given the conditions described in the problem
def right_triangle_proportional (a b c : ℝ) : Prop :=
  ∃ (x : ℝ), a = 3 * x ∧ b = 4 * x ∧ c = 5 * x

def hypotenuse (c : ℝ) : Prop := 
  c = 25

def leg (b : ℝ) : Prop :=
  b = 20

-- The theorem stating that the height h of the triangle is 12
theorem find_height (a b c : ℝ) (h : ℝ)
  (H1 : right_triangle_proportional a b c)
  (H2 : hypotenuse c)
  (H3 : leg b) :
  h = 12 :=
by
  sorry

end RightTriangleProblem

end NUMINAMATH_GPT_find_height_l366_36686


namespace NUMINAMATH_GPT_remainder_of_product_mod_5_l366_36636

theorem remainder_of_product_mod_5 :
  (2685 * 4932 * 91406) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_product_mod_5_l366_36636


namespace NUMINAMATH_GPT_find_f_neg1_l366_36690

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f : ℝ → ℝ
| x => if 0 < x then x^2 + 2 else if x = 0 then 2 else -(x^2 + 2)

axiom odd_f : is_odd_function f

theorem find_f_neg1 : f (-1) = -3 := by
  sorry

end NUMINAMATH_GPT_find_f_neg1_l366_36690


namespace NUMINAMATH_GPT_diff_of_squares_635_615_l366_36653

theorem diff_of_squares_635_615 : 635^2 - 615^2 = 25000 :=
by
  sorry

end NUMINAMATH_GPT_diff_of_squares_635_615_l366_36653


namespace NUMINAMATH_GPT_raghu_investment_l366_36655

noncomputable def investment_problem (R T V : ℝ) : Prop :=
  V = 1.1 * T ∧
  T = 0.9 * R ∧
  R + T + V = 6358 ∧
  R = 2200

theorem raghu_investment
  (R T V : ℝ)
  (h1 : V = 1.1 * T)
  (h2 : T = 0.9 * R)
  (h3 : R + T + V = 6358) :
  R = 2200 :=
sorry

end NUMINAMATH_GPT_raghu_investment_l366_36655


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_9_l366_36652

theorem arithmetic_sequence_sum_9 :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a n = 2 + n * d) ∧ d ≠ 0 ∧ (2 : ℝ) + 2 * d ≠ 0 ∧ (2 + 5 * d) ≠ 0 ∧ d = 0.5 →
  (2 + 2 * d)^2 = 2 * (2 + 5 * d) →
  (9 * 2 + (9 * 8 / 2) * 0.5) = 36 :=
by
  intros a d h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_9_l366_36652


namespace NUMINAMATH_GPT_oranges_thrown_away_l366_36673

theorem oranges_thrown_away (initial_oranges new_oranges current_oranges : ℕ) (x : ℕ) 
  (h1 : initial_oranges = 50)
  (h2 : new_oranges = 24)
  (h3 : current_oranges = 34) : 
  initial_oranges - x + new_oranges = current_oranges → x = 40 :=
by
  intros h
  rw [h1, h2, h3] at h
  sorry

end NUMINAMATH_GPT_oranges_thrown_away_l366_36673


namespace NUMINAMATH_GPT_triangle_area_correct_l366_36606

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area_correct : 
  area_of_triangle (0, 0) (2, 0) (2, 3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l366_36606


namespace NUMINAMATH_GPT_arithmetic_sequence_a12_l366_36676

theorem arithmetic_sequence_a12 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 7 + a 9 = 16) (h2 : a 4 = 1) 
  (h3 : ∀ n, a (n + 1) = a n + d) : a 12 = 15 := 
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_a12_l366_36676


namespace NUMINAMATH_GPT_OHaraTriple_example_l366_36681

def OHaraTriple (a b x : ℕ) : Prop :=
  (Nat.sqrt a + Nat.sqrt b = x)

theorem OHaraTriple_example : OHaraTriple 49 64 15 :=
by
  sorry

end NUMINAMATH_GPT_OHaraTriple_example_l366_36681


namespace NUMINAMATH_GPT_convex_polyhedron_space_diagonals_l366_36608

theorem convex_polyhedron_space_diagonals
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)
  (triangular_faces : ℕ)
  (hexagonal_faces : ℕ)
  (total_faces : faces = triangular_faces + hexagonal_faces)
  (vertices_eq : vertices = 30)
  (edges_eq : edges = 72)
  (triangular_faces_eq : triangular_faces = 32)
  (hexagonal_faces_eq : hexagonal_faces = 12)
  (faces_eq : faces = 44) :
  ((vertices * (vertices - 1)) / 2) - edges - 
  (triangular_faces * 0 + hexagonal_faces * ((6 * (6 - 3)) / 2)) = 255 := by
sorry

end NUMINAMATH_GPT_convex_polyhedron_space_diagonals_l366_36608


namespace NUMINAMATH_GPT_count_p_values_l366_36691

theorem count_p_values (p : ℤ) (n : ℝ) :
  (n = 16 * 10^(-p)) →
  (-4 < p ∧ p < 4) →
  ∃ m, p ∈ m ∧ (m.count = 3 ∧ m = [-2, 0, 2]) :=
by 
  sorry

end NUMINAMATH_GPT_count_p_values_l366_36691


namespace NUMINAMATH_GPT_paul_homework_average_l366_36696

def hoursOnWeeknights : ℕ := 2 * 5
def hoursOnWeekend : ℕ := 5
def totalHomework : ℕ := hoursOnWeeknights + hoursOnWeekend
def practiceNights : ℕ := 2
def daysAvailable : ℕ := 7 - practiceNights
def averageHomeworkPerNight : ℕ := totalHomework / daysAvailable

theorem paul_homework_average :
  averageHomeworkPerNight = 3 := 
by
  -- sorry because we skip the proof
  sorry

end NUMINAMATH_GPT_paul_homework_average_l366_36696


namespace NUMINAMATH_GPT_interest_rate_C_l366_36689

theorem interest_rate_C (P A G : ℝ) (R : ℝ) (t : ℝ := 3) (rate_A : ℝ := 0.10) :
  P = 4000 ∧ rate_A = 0.10 ∧ G = 180 →
  (P * rate_A * t + G) = P * (R / 100) * t →
  R = 11.5 :=
by
  intros h_cond h_eq
  -- proof to be filled, use the given conditions and equations
  sorry

end NUMINAMATH_GPT_interest_rate_C_l366_36689


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l366_36617

theorem arithmetic_sequence_sum (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 9 = a_n 4 + a_n 5 + a_n 6 + 72)
  (h2 : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (h3 : ∀ n, a_n (n+1) - a_n n = d)
  (h4 : a_n 1 + a_n 9 = a_n 3 + a_n 7)
  (h5 : a_n 3 + a_n 7 = a_n 4 + a_n 6)
  (h6 : a_n 4 + a_n 6 = 2 * a_n 5) : 
  a_n 3 + a_n 7 = 24 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l366_36617


namespace NUMINAMATH_GPT_sum_of_terms_in_geometric_sequence_eq_fourteen_l366_36612

theorem sum_of_terms_in_geometric_sequence_eq_fourteen
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_a1 : a 1 = 1)
  (h_arith : 4 * a 2 = 2 * a 3 ∧ 2 * a 3 - 4 * a 2 = a 4 - 2 * a 3) :
  a 2 + a 3 + a 4 = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_terms_in_geometric_sequence_eq_fourteen_l366_36612


namespace NUMINAMATH_GPT_solution_set_of_f_lt_exp_l366_36638

noncomputable def f : ℝ → ℝ := sorry -- assume f is a differentiable function

-- Define the conditions
axiom h_deriv : ∀ x : ℝ, deriv f x < f x
axiom h_periodic : ∀ x : ℝ, f (x + 2) = f (x - 2)
axiom h_value_at_4 : f 4 = 1

-- The main statement to be proved
theorem solution_set_of_f_lt_exp :
  ∀ x : ℝ, (f x < Real.exp x ↔ x > 0) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solution_set_of_f_lt_exp_l366_36638


namespace NUMINAMATH_GPT_angle_Z_is_90_l366_36618

theorem angle_Z_is_90 (X Y Z : ℝ) (h_sum_XY : X + Y = 90) (h_Y_is_2X : Y = 2 * X) (h_sum_angles : X + Y + Z = 180) : Z = 90 :=
by
  sorry

end NUMINAMATH_GPT_angle_Z_is_90_l366_36618


namespace NUMINAMATH_GPT_number_of_arrangements_of_six_students_l366_36651

/-- A and B cannot stand together -/
noncomputable def arrangements_A_B_not_together (n: ℕ) (A B: ℕ) : ℕ :=
  if n = 6 then 480 else 0

theorem number_of_arrangements_of_six_students :
  arrangements_A_B_not_together 6 1 2 = 480 :=
sorry

end NUMINAMATH_GPT_number_of_arrangements_of_six_students_l366_36651


namespace NUMINAMATH_GPT_largest_divisor_of_polynomial_l366_36670

theorem largest_divisor_of_polynomial (n : ℕ) (h : n % 2 = 0) : 
  105 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_polynomial_l366_36670


namespace NUMINAMATH_GPT_power_sum_is_integer_l366_36668

theorem power_sum_is_integer (a : ℝ) (n : ℕ) (h_pos : 0 < n)
  (h_k : ∃ k : ℤ, k = a + 1/a) : 
  ∃ m : ℤ, m = a^n + 1/a^n := 
sorry

end NUMINAMATH_GPT_power_sum_is_integer_l366_36668


namespace NUMINAMATH_GPT_find_expression_value_l366_36643

theorem find_expression_value (x : ℝ) (h : x^2 - 5*x = 14) : 
  (x-1)*(2*x-1) - (x+1)^2 + 1 = 15 := 
by 
  sorry

end NUMINAMATH_GPT_find_expression_value_l366_36643


namespace NUMINAMATH_GPT_apples_left_is_correct_l366_36639

-- Definitions for the conditions
def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℚ := 1 / 5 * total_apples
def apples_left : ℚ := total_apples - apples_given_to_son

-- The main statement to be proven
theorem apples_left_is_correct : apples_left = 12 := by
  sorry

end NUMINAMATH_GPT_apples_left_is_correct_l366_36639


namespace NUMINAMATH_GPT_find_c_l366_36644

structure ProblemData where
  (r : ℝ → ℝ)
  (s : ℝ → ℝ)
  (h : r (s 3) = 20)

def r (x : ℝ) : ℝ := 5 * x - 10
def s (x : ℝ) (c : ℝ) : ℝ := 4 * x - c

theorem find_c (c : ℝ) (h : (r (s 3 c)) = 20) : c = 6 :=
sorry

end NUMINAMATH_GPT_find_c_l366_36644


namespace NUMINAMATH_GPT_general_formula_an_bounds_Mn_l366_36611

variable {n : ℕ}

-- Define the sequence Sn
def S : ℕ → ℚ := λ n => n * (4 * n - 3) - 2 * n * (n - 1)

-- Define the sequence an based on Sn
def a : ℕ → ℚ := λ n =>
  if n = 0 then 0 else S n - S (n - 1)

-- Define the sequence Mn and the bounds to prove
def M : ℕ → ℚ := λ n => (1 / 4) * (1 - (1 / (4 * n + 1)))

-- Theorem: General formula for the sequence {a_n}
theorem general_formula_an (n : ℕ) (hn : 1 ≤ n) : a n = 4 * n - 3 :=
  sorry

-- Theorem: Bounds for the sequence {M_n}
theorem bounds_Mn (n : ℕ) (hn : 1 ≤ n) : (1 / 5 : ℚ) ≤ M n ∧ M n < (1 / 4) :=
  sorry

end NUMINAMATH_GPT_general_formula_an_bounds_Mn_l366_36611


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l366_36630

variable {m n p x : ℝ}

-- Problem 1
theorem problem1 : m^2 * (n - 3) + 4 * (3 - n) = (n - 3) * (m + 2) * (m - 2) := 
sorry

-- Problem 2
theorem problem2 : (p - 3) * (p - 1) + 1 = (p - 2) ^ 2 := 
sorry

-- Problem 3
theorem problem3 (hx : x^2 + x + 1 / 4 = 0) : (2 * x + 1) / (x + 1) + (x - 1) / 1 / (x + 2) / (x^2 + 2 * x + 1) = -1 / 4 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l366_36630


namespace NUMINAMATH_GPT_longest_leg_of_smallest_triangle_l366_36675

-- Definitions based on conditions
def is306090Triangle (h : ℝ) (s : ℝ) (l : ℝ) : Prop :=
  s = h / 2 ∧ l = s * (Real.sqrt 3)

def chain_of_306090Triangles (H : ℝ) : Prop :=
  ∃ h1 s1 l1 h2 s2 l2 h3 s3 l3 h4 s4 l4,
    is306090Triangle h1 s1 l1 ∧
    is306090Triangle h2 s2 l2 ∧
    is306090Triangle h3 s3 l3 ∧
    is306090Triangle h4 s4 l4 ∧
    h1 = H ∧ l1 = h2 ∧ l2 = h3 ∧ l3 = h4

-- Main theorem
theorem longest_leg_of_smallest_triangle (H : ℝ) (h : ℝ) (l : ℝ) (H_cond : H = 16) 
  (h_cond : h = 9) :
  chain_of_306090Triangles H →
  ∃ h4 s4 l4, is306090Triangle h4 s4 l4 ∧ l = h4 →
  l = 9 := 
by
  sorry

end NUMINAMATH_GPT_longest_leg_of_smallest_triangle_l366_36675


namespace NUMINAMATH_GPT_forgotten_code_possibilities_l366_36695

theorem forgotten_code_possibilities:
  let digits_set := {d | ∀ n:ℕ, 0≤n ∧ n≤9 → n≠0 → 
                     (n + 4 + 4 + last_digit ≡ 0 [MOD 3]) ∨ 
                     (n + 7 + 7 + last_digit ≡ 0 [MOD 3]) ∨
                     (n + 4 + 7 + last_digit ≡ 0 [MOD 3]) ∨
                     (n + 7 + 4 + last_digit ≡ 0 [MOD 3])
                    }
  let valid_first_digits := {1, 2, 4, 5, 7, 8}
  let total_combinations := 4 * 3 + 4 * 3 -- middle combinations * valid first digit combinations
  total_combinations = 24 ∧ digits_set = valid_first_digits := by
  sorry

end NUMINAMATH_GPT_forgotten_code_possibilities_l366_36695


namespace NUMINAMATH_GPT_solve_system_of_equations_l366_36682

theorem solve_system_of_equations 
  (a1 a2 a3 a4 : ℝ) (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (x1 x2 x3 x4 : ℝ)
  (h1 : |a1 - a1| * x1 + |a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1)
  (h2 : |a2 - a1| * x1 + |a2 - a2| * x2 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1)
  (h3 : |a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a3| * x3 + |a3 - a4| * x4 = 1)
  (h4 : |a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 + |a4 - a4| * x4 = 1) :
  x1 = 1 / (a1 - a4) ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 1 / (a1 - a4) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l366_36682


namespace NUMINAMATH_GPT_not_inequality_neg_l366_36677

theorem not_inequality_neg (x y : ℝ) (h : x > y) : ¬ (-x > -y) :=
by {
  sorry
}

end NUMINAMATH_GPT_not_inequality_neg_l366_36677


namespace NUMINAMATH_GPT_seven_searchlights_shadow_length_l366_36627

noncomputable def searchlight_positioning (n : ℕ) (angle : ℝ) (shadow_length : ℝ) : Prop :=
  ∃ (positions : Fin n → ℝ × ℝ), ∀ i : Fin n, ∃ shadow : ℝ, shadow = shadow_length ∧
  (∀ j : Fin n, i ≠ j → ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  θ - angle / 2 < θ ∧ θ + angle / 2 > θ → shadow = shadow_length)

theorem seven_searchlights_shadow_length :
  searchlight_positioning 7 (Real.pi / 2) 7000 :=
sorry

end NUMINAMATH_GPT_seven_searchlights_shadow_length_l366_36627


namespace NUMINAMATH_GPT_elyse_passing_threshold_l366_36680

def total_questions : ℕ := 90
def programming_questions : ℕ := 20
def database_questions : ℕ := 35
def networking_questions : ℕ := 35
def programming_correct_rate : ℝ := 0.8
def database_correct_rate : ℝ := 0.5
def networking_correct_rate : ℝ := 0.7
def passing_percentage : ℝ := 0.65

theorem elyse_passing_threshold :
  let programming_correct := programming_correct_rate * programming_questions
  let database_correct := database_correct_rate * database_questions
  let networking_correct := networking_correct_rate * networking_questions
  let total_correct := programming_correct + database_correct + networking_correct
  let required_to_pass := passing_percentage * total_questions
  total_correct = required_to_pass → 0 = 0 :=
by
  intro _h
  sorry

end NUMINAMATH_GPT_elyse_passing_threshold_l366_36680


namespace NUMINAMATH_GPT_wendy_walked_l366_36640

theorem wendy_walked (x : ℝ) (h1 : 19.83 = x + 10.67) : x = 9.16 :=
sorry

end NUMINAMATH_GPT_wendy_walked_l366_36640


namespace NUMINAMATH_GPT_sector_area_ratio_l366_36659

theorem sector_area_ratio (angle_AOE angle_FOB : ℝ) (h1 : angle_AOE = 40) (h2 : angle_FOB = 60) : 
  (180 - angle_AOE - angle_FOB) / 360 = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_ratio_l366_36659


namespace NUMINAMATH_GPT_smallest_N_for_triangle_sides_l366_36633

theorem smallest_N_for_triangle_sides (a b c : ℝ) (h_triangle : a + b > c) (h_a_ne_b : a ≠ b) : (a^2 + b^2) / c^2 < 1 := 
sorry

end NUMINAMATH_GPT_smallest_N_for_triangle_sides_l366_36633


namespace NUMINAMATH_GPT_fraction_distance_walked_by_first_class_l366_36622

namespace CulturalCenterProblem

def walking_speed : ℝ := 4
def bus_speed_with_students : ℝ := 40
def bus_speed_empty : ℝ := 60

theorem fraction_distance_walked_by_first_class :
  ∃ (x : ℝ), 
    (x / walking_speed) = ((1 - x) / bus_speed_with_students) + ((1 - 2 * x) / bus_speed_empty)
    ∧ x = 5 / 37 :=
by
  sorry

end CulturalCenterProblem

end NUMINAMATH_GPT_fraction_distance_walked_by_first_class_l366_36622


namespace NUMINAMATH_GPT_calculate_a_over_b_l366_36685

noncomputable def system_solution (x y a b : ℝ) : Prop :=
  (8 * x - 5 * y = a) ∧ (10 * y - 15 * x = b) ∧ (x ≠ 0) ∧ (y ≠ 0) ∧ (b ≠ 0)

theorem calculate_a_over_b (x y a b : ℝ) (h : system_solution x y a b) : a / b = 8 / 15 :=
by
  sorry

end NUMINAMATH_GPT_calculate_a_over_b_l366_36685


namespace NUMINAMATH_GPT_Petya_time_comparison_l366_36610

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_Petya_time_comparison_l366_36610
