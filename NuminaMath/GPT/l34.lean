import Mathlib

namespace measure_of_angle_Q_l34_3461

variables (R S T U Q : ℝ)
variables (angle_R angle_S angle_T angle_U : ℝ)

-- Given conditions
def sum_of_angles_in_pentagon : ℝ := 540
def angle_measure_R : ℝ := 120
def angle_measure_S : ℝ := 94
def angle_measure_T : ℝ := 115
def angle_measure_U : ℝ := 101

theorem measure_of_angle_Q :
  angle_R = angle_measure_R →
  angle_S = angle_measure_S →
  angle_T = angle_measure_T →
  angle_U = angle_measure_U →
  (angle_R + angle_S + angle_T + angle_U + Q = sum_of_angles_in_pentagon) →
  Q = 110 :=
by { sorry }

end measure_of_angle_Q_l34_3461


namespace like_terms_constants_l34_3424

theorem like_terms_constants :
  ∀ (a b : ℚ), a = 1/2 → b = -1/3 → (a = 1/2 ∧ b = -1/3) → a + b = 1/2 + -1/3 :=
by
  intros a b ha hb h
  sorry

end like_terms_constants_l34_3424


namespace polynomials_equal_at_all_x_l34_3497

variable {R : Type} [CommRing R]

def f (a_5 a_4 a_3 a_2 a_1 a_0 : R) (x : R) := a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
def g (b_3 b_2 b_1 b_0 : R) (x : R) := b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0
def h (c_2 c_1 c_0 : R) (x : R) := c_2 * x^2 + c_1 * x + c_0

theorem polynomials_equal_at_all_x 
    (a_5 a_4 a_3 a_2 a_1 a_0 b_3 b_2 b_1 b_0 c_2 c_1 c_0 : ℤ)
    (bound_a : ∀ i ∈ [a_5, a_4, a_3, a_2, a_1, a_0], |i| ≤ 4)
    (bound_b : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1)
    (bound_c : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1)
    (H : f a_5 a_4 a_3 a_2 a_1 a_0 10 = g b_3 b_2 b_1 b_0 10 * h c_2 c_1 c_0 10) :
    ∀ x, f a_5 a_4 a_3 a_2 a_1 a_0 x = g b_3 b_2 b_1 b_0 x * h c_2 c_1 c_0 x := by
  sorry

end polynomials_equal_at_all_x_l34_3497


namespace bold_o_lit_cells_l34_3486

-- Define the conditions
def grid_size : ℕ := 5
def original_o_lit_cells : ℕ := 12 -- Number of cells lit in the original 'o'
def additional_lit_cells : ℕ := 12 -- Additional cells lit in the bold 'o'

-- Define the property to be proved
theorem bold_o_lit_cells : (original_o_lit_cells + additional_lit_cells) = 24 :=
by
  -- computation skipped
  sorry

end bold_o_lit_cells_l34_3486


namespace minute_hand_40_min_angle_l34_3408

noncomputable def minute_hand_rotation_angle (minutes : ℕ): ℝ :=
  if minutes = 60 then -2 * Real.pi 
  else (minutes / 60) * -2 * Real.pi

theorem minute_hand_40_min_angle :
  minute_hand_rotation_angle 40 = - (4 / 3) * Real.pi :=
by
  sorry

end minute_hand_40_min_angle_l34_3408


namespace maximize_area_partition_l34_3455

noncomputable def optimLengthPartition (material: ℝ) (partitions: ℕ) : ℝ :=
  (material / (4 + partitions))

theorem maximize_area_partition :
  optimLengthPartition 24 (2 * 1) = 3 / 100 :=
by
  sorry

end maximize_area_partition_l34_3455


namespace triangle_lattice_points_l34_3434

-- Given lengths of the legs of the right triangle
def DE : Nat := 15
def EF : Nat := 20

-- Calculate the hypotenuse using the Pythagorean theorem
def DF : Nat := Nat.sqrt (DE ^ 2 + EF ^ 2)

-- Calculate the area of the triangle
def Area : Nat := (DE * EF) / 2

-- Calculate the number of boundary points
def B : Nat :=
  let points_DE := DE + 1
  let points_EF := EF + 1
  let points_DF := DF + 1
  points_DE + points_EF + points_DF - 3

-- Calculate the number of interior points using Pick's Theorem
def I : Int := Area - (B / 2 - 1)

-- Calculate the total number of lattice points
def total_lattice_points : Int := I + Int.ofNat B

-- The theorem statement
theorem triangle_lattice_points : total_lattice_points = 181 := by
  -- The actual proof goes here
  sorry

end triangle_lattice_points_l34_3434


namespace sluice_fill_time_l34_3485

noncomputable def sluice_open_equal_time (x y : ℝ) (m : ℝ) : ℝ :=
  -- Define time (t) required for both sluice gates to be open equally to fill the lake
  m / 11

theorem sluice_fill_time :
  ∀ (x y : ℝ),
    (10 * x + 14 * y = 9900) →
    (18 * x + 12 * y = 9900) →
    sluice_open_equal_time x y 9900 = 900 := sorry

end sluice_fill_time_l34_3485


namespace average_score_all_test_takers_l34_3403

def avg (scores : List ℕ) : ℕ := scores.sum / scores.length

theorem average_score_all_test_takers (s_avg u_avg n : ℕ) 
  (H1 : s_avg = 42) (H2 : u_avg = 38) (H3 : n = 20) : avg ([s_avg * n, u_avg * n]) / (2 * n) = 40 := 
by sorry

end average_score_all_test_takers_l34_3403


namespace fraction_percent_of_y_l34_3489

theorem fraction_percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 10 + (3 * y) / 10) = 0.5 * y := by
  sorry

end fraction_percent_of_y_l34_3489


namespace find_x_for_parallel_vectors_l34_3450

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (4, x)
def b : ℝ × ℝ := (-4, 4)

-- Define parallelism condition for two 2D vectors
def are_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Define the main theorem statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : are_parallel (a x) b) : x = -4 :=
by sorry

end find_x_for_parallel_vectors_l34_3450


namespace point_in_fourth_quadrant_l34_3453

def x : ℝ := 8
def y : ℝ := -3

theorem point_in_fourth_quadrant (h1 : x > 0) (h2 : y < 0) : (x > 0 ∧ y < 0) :=
by {
  sorry
}

end point_in_fourth_quadrant_l34_3453


namespace fans_attended_show_l34_3495

-- Definitions from the conditions
def total_seats : ℕ := 60000
def sold_percentage : ℝ := 0.75
def fans_stayed_home : ℕ := 5000

-- The proof statement
theorem fans_attended_show :
  let sold_seats := sold_percentage * total_seats
  let fans_attended := sold_seats - fans_stayed_home
  fans_attended = 40000 :=
by
  -- Auto-generated proof placeholder.
  sorry

end fans_attended_show_l34_3495


namespace profit_percentage_is_23_16_l34_3433

   noncomputable def cost_price (mp : ℝ) : ℝ := 95 * mp
   noncomputable def selling_price (mp : ℝ) : ℝ := 120 * (mp - (0.025 * mp))
   noncomputable def profit_percent (cp sp : ℝ) : ℝ := ((sp - cp) / cp) * 100

   theorem profit_percentage_is_23_16 
     (mp : ℝ) (h_mp_gt_zero : mp > 0) : 
       profit_percent (cost_price mp) (selling_price mp) = 23.16 :=
   by 
     sorry
   
end profit_percentage_is_23_16_l34_3433


namespace quadratic_inequality_solution_l34_3483

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - (3 / 8) < 0) ↔ (-3 < k ∧ k < 0) :=
sorry

end quadratic_inequality_solution_l34_3483


namespace raft_drift_time_l34_3488

-- Define the conditions from the problem
variable (distance : ℝ := 1)
variable (steamboat_time : ℝ := 1) -- in hours
variable (motorboat_time : ℝ := 3 / 4) -- 45 minutes in hours
variable (motorboat_speed_ratio : ℝ := 2)

-- Variables for speeds
variable (vs vf : ℝ)

-- Conditions: the speeds and conditions of traveling from one village to another
variable (steamboat_eqn : vs + vf = distance / steamboat_time := by sorry)
variable (motorboat_eqn : (2 * vs) + vf = distance / motorboat_time := by sorry)

-- Time for the raft to travel the distance
theorem raft_drift_time : 90 = (distance / vf) * 60 := by
  -- Proof comes here
  sorry

end raft_drift_time_l34_3488


namespace problem1_proof_l34_3437

-- Define the mathematical conditions and problems
def problem1_expression (x y : ℝ) : ℝ := y * (4 * x - 3 * y) + (x - 2 * y) ^ 2

-- State the theorem with the simplified form as the conclusion
theorem problem1_proof (x y : ℝ) : problem1_expression x y = x^2 + y^2 :=
by
  sorry

end problem1_proof_l34_3437


namespace younger_person_age_l34_3402

theorem younger_person_age (e y : ℕ) 
  (h1: e = y + 20)
  (h2: e - 10 = 5 * (y - 10)) : 
  y = 15 := 
by
  sorry

end younger_person_age_l34_3402


namespace sum_of_squares_l34_3473

theorem sum_of_squares (a b c : ℝ) (h₁ : a + b + c = 31) (h₂ : ab + bc + ca = 10) :
  a^2 + b^2 + c^2 = 941 :=
by
  sorry

end sum_of_squares_l34_3473


namespace tan_alpha_two_implies_fraction_eq_three_fourths_l34_3444

variable {α : ℝ}

theorem tan_alpha_two_implies_fraction_eq_three_fourths (h1 : Real.tan α = 2) (h2 : Real.cos α ≠ 0) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

end tan_alpha_two_implies_fraction_eq_three_fourths_l34_3444


namespace fish_bird_apple_fraction_l34_3438

theorem fish_bird_apple_fraction (M : ℝ) (hM : 0 < M) :
  let R_fish := 120
  let R_bird := 60
  let R_total := 180
  let T := M / R_total
  let fish_fraction := (R_fish * T) / M
  let bird_fraction := (R_bird * T) / M
  fish_fraction = 2/3 ∧ bird_fraction = 1/3 := by
  sorry

end fish_bird_apple_fraction_l34_3438


namespace range_of_g_l34_3443

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : 
  Set.range g = Set.Icc ((π / 2) - (π / 3)) ((π / 2) + (π / 3)) := by
  sorry

end range_of_g_l34_3443


namespace new_bucket_capacity_l34_3422

theorem new_bucket_capacity (init_buckets : ℕ) (init_capacity : ℕ) (new_buckets : ℕ) (total_volume : ℕ) :
  init_buckets * init_capacity = total_volume →
  new_buckets * 9 = total_volume →
  9 = total_volume / new_buckets :=
by
  intros h₁ h₂
  sorry

end new_bucket_capacity_l34_3422


namespace problem_proof_l34_3480

theorem problem_proof (a b x y : ℝ) (h1 : a + b = 0) (h2 : x * y = 1) : 5 * |a + b| - 5 * (x * y) = -5 :=
by
  sorry

end problem_proof_l34_3480


namespace arithmetic_sequence_eleventh_term_l34_3415

theorem arithmetic_sequence_eleventh_term 
  (a d : ℚ)
  (h_sum_first_six : 6 * a + 15 * d = 30)
  (h_seventh_term : a + 6 * d = 10) : 
    a + 10 * d = 110 / 7 := 
by
  sorry

end arithmetic_sequence_eleventh_term_l34_3415


namespace frustumViews_l34_3462

-- Define the notion of a frustum
structure Frustum where
  -- You may add necessary geometric properties of a frustum if needed
  
-- Define a function to describe the view of the frustum
def frontView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type
def sideView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type
def topView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type

-- Define the properties of the views
def isCongruentIsoscelesTrapezoid (fig : Type) : Prop := sorry -- Define property for congruent isosceles trapezoid
def isTwoConcentricCircles (fig : Type) : Prop := sorry -- Define property for two concentric circles

-- State the theorem based on the given problem
theorem frustumViews (f : Frustum) :
  isCongruentIsoscelesTrapezoid (frontView f) ∧ 
  isCongruentIsoscelesTrapezoid (sideView f) ∧ 
  isTwoConcentricCircles (topView f) := 
sorry

end frustumViews_l34_3462


namespace arithmetic_sequence_a8_value_l34_3435

theorem arithmetic_sequence_a8_value
  (a : ℕ → ℤ) 
  (h1 : a 1 + 3 * a 8 + a 15 = 120)
  (h2 : a 1 + a 15 = 2 * a 8) :
  a 8 = 24 := 
sorry

end arithmetic_sequence_a8_value_l34_3435


namespace polynomial_solution_l34_3498

variable {R : Type*} [CommRing R]

theorem polynomial_solution (p : Polynomial R) :
  (∀ (a b c : R), 
    p.eval (a + b - 2 * c) + p.eval (b + c - 2 * a) + p.eval (c + a - 2 * b)
      = 3 * p.eval (a - b) + 3 * p.eval (b - c) + 3 * p.eval (c - a)
  ) →
  ∃ (a1 a2 : R), p = Polynomial.C a2 * Polynomial.X^2 + Polynomial.C a1 * Polynomial.X :=
by
  sorry

end polynomial_solution_l34_3498


namespace ratio_female_to_male_members_l34_3468

theorem ratio_female_to_male_members (f m : ℕ)
  (h1 : 35 * f = SumAgesFemales)
  (h2 : 20 * m = SumAgesMales)
  (h3 : (35 * f + 20 * m) / (f + m) = 25) :
  f / m = 1 / 2 := by
  sorry

end ratio_female_to_male_members_l34_3468


namespace bees_flew_in_l34_3459

theorem bees_flew_in (initial_bees additional_bees total_bees : ℕ) 
  (h1 : initial_bees = 16) (h2 : total_bees = 25) 
  (h3 : initial_bees + additional_bees = total_bees) : additional_bees = 9 :=
by sorry

end bees_flew_in_l34_3459


namespace total_canoes_by_end_of_april_l34_3470

def canoes_built_jan : Nat := 4

def canoes_built_next_month (prev_month : Nat) : Nat := 3 * prev_month

def canoes_built_feb : Nat := canoes_built_next_month canoes_built_jan
def canoes_built_mar : Nat := canoes_built_next_month canoes_built_feb
def canoes_built_apr : Nat := canoes_built_next_month canoes_built_mar

def total_canoes_built : Nat := canoes_built_jan + canoes_built_feb + canoes_built_mar + canoes_built_apr

theorem total_canoes_by_end_of_april : total_canoes_built = 160 :=
by
  sorry

end total_canoes_by_end_of_april_l34_3470


namespace lines_parallel_lines_perpendicular_l34_3411

-- Definition of lines
def l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) := x + (a - 1) * y + a ^ 2 - 1 = 0

-- Parallel condition proof problem
theorem lines_parallel (a : ℝ) : (a = -1) → ∀ x y : ℝ, l1 a x y ∧ l2 a x y →  
        (-(a / 2) = (1 / (1 - a))) ∧ (-3 ≠ -a - 1) :=
by
  intros
  sorry

-- Perpendicular condition proof problem
theorem lines_perpendicular (a : ℝ) : (a = 2 / 3) → ∀ x y : ℝ, l1 a x y ∧ l2 a x y → 
        (- (a / 2) * (1 / (1 - a)) = -1) :=
by
  intros
  sorry

end lines_parallel_lines_perpendicular_l34_3411


namespace remainder_when_divided_by_10_l34_3467

theorem remainder_when_divided_by_10 : 
  (2468 * 7391 * 90523) % 10 = 4 :=
by
  sorry

end remainder_when_divided_by_10_l34_3467


namespace no_adjacent_same_roll_probability_l34_3419

-- We define probabilistic event on rolling a six-sided die and sitting around a circular table
noncomputable def probability_no_adjacent_same_roll : ℚ :=
  1 * (5/6) * (5/6) * (5/6) * (5/6) * (4/6)

theorem no_adjacent_same_roll_probability :
  probability_no_adjacent_same_roll = 625/1944 :=
by
  sorry

end no_adjacent_same_roll_probability_l34_3419


namespace range_of_p_l34_3448

noncomputable def success_prob_4_engine (p : ℝ) : ℝ :=
  4 * p^3 * (1 - p) + p^4

noncomputable def success_prob_2_engine (p : ℝ) : ℝ :=
  p^2

theorem range_of_p (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  success_prob_4_engine p > success_prob_2_engine p ↔ (1/3 < p ∧ p < 1) :=
by
  sorry

end range_of_p_l34_3448


namespace minimum_total_number_of_balls_l34_3492

theorem minimum_total_number_of_balls (x y z t : ℕ) 
  (h1 : x ≥ 4)
  (h2 : x ≥ 3 ∧ y ≥ 1)
  (h3 : x ≥ 2 ∧ y ≥ 1 ∧ z ≥ 1)
  (h4 : x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ t ≥ 1) :
  x + y + z + t = 21 :=
  sorry

end minimum_total_number_of_balls_l34_3492


namespace least_cost_grass_seed_l34_3440

variable (cost_5_pound_bag : ℕ) [Fact (cost_5_pound_bag = 1380)]
variable (cost_10_pound_bag : ℕ) [Fact (cost_10_pound_bag = 2043)]
variable (cost_25_pound_bag : ℕ) [Fact (cost_25_pound_bag = 3225)]
variable (min_weight : ℕ) [Fact (min_weight = 65)]
variable (max_weight : ℕ) [Fact (max_weight = 80)]

theorem least_cost_grass_seed :
  ∃ (n5 n10 n25 : ℕ),
    n5 * 5 + n10 * 10 + n25 * 25 ≥ min_weight ∧
    n5 * 5 + n10 * 10 + n25 * 25 ≤ max_weight ∧
    n5 * cost_5_pound_bag + n10 * cost_10_pound_bag + n25 * cost_25_pound_bag = 9675 :=
  sorry

end least_cost_grass_seed_l34_3440


namespace area_square_l34_3413

-- Define the conditions
variables (l r s : ℝ)
variable (breadth : ℝ := 10)
variable (area_rect : ℝ := 180)

-- Given conditions
def length_is_two_fifths_radius : Prop := l = (2/5) * r
def radius_is_side_square : Prop := r = s
def area_of_rectangle : Prop := area_rect = l * breadth

-- The theorem statement
theorem area_square (h1 : length_is_two_fifths_radius l r)
                    (h2 : radius_is_side_square r s)
                    (h3 : area_of_rectangle l breadth area_rect) :
  s^2 = 2025 :=
by
  sorry

end area_square_l34_3413


namespace perfect_shells_l34_3463

theorem perfect_shells (P_spiral B_spiral P_total : ℕ) 
  (h1 : 52 = 2 * B_spiral)
  (h2 : B_spiral = P_spiral + 21)
  (h3 : P_total = P_spiral + 12) :
  P_total = 17 :=
by
  sorry

end perfect_shells_l34_3463


namespace space_per_bookshelf_l34_3482

-- Defining the conditions
def S_room : ℕ := 400
def S_reserved : ℕ := 160
def n_shelves : ℕ := 3

-- Theorem statement
theorem space_per_bookshelf (S_room S_reserved n_shelves : ℕ)
  (h1 : S_room = 400) (h2 : S_reserved = 160) (h3 : n_shelves = 3) :
  (S_room - S_reserved) / n_shelves = 80 :=
by
  -- Placeholder for the proof
  sorry

end space_per_bookshelf_l34_3482


namespace joe_needs_more_cars_l34_3466

-- Definitions based on conditions
def current_cars : ℕ := 50
def total_cars : ℕ := 62

-- Theorem based on the problem question and correct answer
theorem joe_needs_more_cars : (total_cars - current_cars) = 12 :=
by
  sorry

end joe_needs_more_cars_l34_3466


namespace solve_inequality_l34_3432

theorem solve_inequality (x : ℝ) : 2 * (5 * x + 3) ≤ x - 3 * (1 - 2 * x) → x ≤ -3 :=
by
  sorry

end solve_inequality_l34_3432


namespace boys_in_parkway_l34_3417

theorem boys_in_parkway (total_students : ℕ) (students_playing_soccer : ℕ) (percentage_boys_playing_soccer : ℝ)
                        (girls_not_playing_soccer : ℕ) :
                        total_students = 420 ∧ students_playing_soccer = 250 ∧ percentage_boys_playing_soccer = 0.86 
                        ∧ girls_not_playing_soccer = 73 → 
                        ∃ total_boys : ℕ, total_boys = 312 :=
by
  -- Proof omitted
  sorry

end boys_in_parkway_l34_3417


namespace problem_l34_3400

theorem problem (x : ℝ) (h : x^2 + 5 * x - 990 = 0) : x^3 + 6 * x^2 - 985 * x + 1012 = 2002 :=
sorry

end problem_l34_3400


namespace student_marks_l34_3479

def max_marks : ℕ := 600
def passing_percentage : ℕ := 30
def fail_by : ℕ := 100

theorem student_marks :
  ∃ x : ℕ, x + fail_by = (passing_percentage * max_marks) / 100 :=
sorry

end student_marks_l34_3479


namespace frank_hawaiian_slices_l34_3458

theorem frank_hawaiian_slices:
  ∀ (total_slices dean_slices sammy_slices leftover_slices frank_slices : ℕ),
  total_slices = 24 →
  dean_slices = 6 →
  sammy_slices = 4 →
  leftover_slices = 11 →
  (total_slices - leftover_slices) = (dean_slices + sammy_slices + frank_slices) →
  frank_slices = 3 :=
by
  intros total_slices dean_slices sammy_slices leftover_slices frank_slices
  intros h_total h_dean h_sammy h_leftovers h_total_eaten
  sorry

end frank_hawaiian_slices_l34_3458


namespace positive_integer_expression_l34_3457

-- Define the existence conditions for a given positive integer n
theorem positive_integer_expression (n : ℕ) (h : 0 < n) : ∃ a b c : ℤ, (n = a^2 + b^2 + c^2 + c) := 
sorry

end positive_integer_expression_l34_3457


namespace drinkable_amount_l34_3478

variable {LiquidBeforeTest : ℕ}
variable {Threshold : ℕ}

def can_drink_more (LiquidBeforeTest : ℕ) (Threshold : ℕ): ℕ :=
  Threshold - LiquidBeforeTest

theorem drinkable_amount :
  LiquidBeforeTest = 24 ∧ Threshold = 32 →
  can_drink_more LiquidBeforeTest Threshold = 8 := by
  sorry

end drinkable_amount_l34_3478


namespace num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10_l34_3449

def num_valid_pairs : Nat := 34

def num_valid_first_digits : Nat := 6

def num_valid_last_digits : Nat := 10

theorem num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10 :
  (num_valid_first_digits * num_valid_pairs * num_valid_last_digits) = 2040 :=
by
  sorry

end num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10_l34_3449


namespace original_number_l34_3484

theorem original_number (x : ℝ) (h : x + 0.5 * x = 90) : x = 60 :=
by
  sorry

end original_number_l34_3484


namespace jessica_older_than_claire_l34_3439

-- Define the current age of Claire
def claire_current_age := 20 - 2

-- Define the current age of Jessica
def jessica_current_age := 24

-- Prove that Jessica is 6 years older than Claire
theorem jessica_older_than_claire : jessica_current_age - claire_current_age = 6 :=
by
  -- Definitions of the ages
  let claire_current_age := 18
  let jessica_current_age := 24

  -- Prove the age difference
  sorry

end jessica_older_than_claire_l34_3439


namespace binary_to_decimal_l34_3464

theorem binary_to_decimal (x : ℕ) (h : x = 0b110010) : x = 50 := by
  sorry

end binary_to_decimal_l34_3464


namespace num_groups_of_consecutive_natural_numbers_l34_3474

theorem num_groups_of_consecutive_natural_numbers (n : ℕ) (h : 3 * n + 3 < 19) : n < 6 := 
  sorry

end num_groups_of_consecutive_natural_numbers_l34_3474


namespace eval_expression_l34_3416

-- Definitions for the problem conditions
def reciprocal (a : ℕ) : ℚ := 1 / a

-- The theorem statement
theorem eval_expression : (reciprocal 9 - reciprocal 6)⁻¹ = -18 := by
  sorry

end eval_expression_l34_3416


namespace booklet_cost_l34_3460

theorem booklet_cost (b : ℝ) : 
  (10 * b < 15) ∧ (12 * b > 17) → b = 1.42 := by
  sorry

end booklet_cost_l34_3460


namespace cos_thirteen_pi_over_four_l34_3429

theorem cos_thirteen_pi_over_four : Real.cos (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_thirteen_pi_over_four_l34_3429


namespace gcd_a2_13a_36_a_6_eq_6_l34_3436

namespace GCDProblem

variable (a : ℕ)
variable (h : ∃ k, a = 1632 * k)

theorem gcd_a2_13a_36_a_6_eq_6 (ha : ∃ k : ℕ, a = 1632 * k) : 
  Int.gcd (a^2 + 13 * a + 36 : Int) (a + 6 : Int) = 6 := by
  sorry

end GCDProblem

end gcd_a2_13a_36_a_6_eq_6_l34_3436


namespace remove_one_piece_l34_3431

theorem remove_one_piece (pieces : Finset (Fin 8 × Fin 8)) (h_card : pieces.card = 15)
  (h_row : ∀ r : Fin 8, ∃ c, (r, c) ∈ pieces)
  (h_col : ∀ c : Fin 8, ∃ r, (r, c) ∈ pieces) :
  ∃ pieces' : Finset (Fin 8 × Fin 8), pieces'.card = 14 ∧ 
  (∀ r : Fin 8, ∃ c, (r, c) ∈ pieces') ∧ 
  (∀ c : Fin 8, ∃ r, (r, c) ∈ pieces') :=
sorry

end remove_one_piece_l34_3431


namespace solve_equation_l34_3472

noncomputable def unique_solution (x : ℝ) : Prop :=
  2 * x * Real.log x + x - 1 = 0 → x = 1

-- Statement of our theorem
theorem solve_equation (x : ℝ) (h : 0 < x) : unique_solution x := sorry

end solve_equation_l34_3472


namespace inequality_proof_l34_3428

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  (1 + 1/x) * (1 + 1/y) ≥ 4 :=
sorry

end inequality_proof_l34_3428


namespace series_converges_to_half_l34_3421

noncomputable def series_value : ℝ :=
  ∑' (n : ℕ), (n^4 + 3*n^3 + 10*n + 10) / (3^n * (n^4 + 4))

theorem series_converges_to_half : series_value = 1 / 2 :=
  sorry

end series_converges_to_half_l34_3421


namespace find_number_l34_3465

theorem find_number (x : ℝ) (h : (x + 0.005) / 2 = 0.2025) : x = 0.400 :=
sorry

end find_number_l34_3465


namespace math_problem_l34_3496

-- Statement of the theorem
theorem math_problem :
  (0.66)^3 - ((0.1)^3 / ((0.66)^2 + 0.066 + (0.1)^2)) = 0.3612 :=
by
  sorry -- Proof is not required

end math_problem_l34_3496


namespace exponential_fixed_point_l34_3441

variable (a : ℝ)

noncomputable def f (x : ℝ) := a^(x - 1) + 3

theorem exponential_fixed_point (ha1 : a > 0) (ha2 : a ≠ 1) : f a 1 = 4 :=
by
  sorry

end exponential_fixed_point_l34_3441


namespace man_swim_upstream_distance_l34_3469

theorem man_swim_upstream_distance (dist_downstream : ℝ) (time_downstream : ℝ) (time_upstream : ℝ) (speed_still_water : ℝ) 
  (effective_speed_downstream : ℝ) (speed_current : ℝ) (effective_speed_upstream : ℝ) (dist_upstream : ℝ) :
  dist_downstream = 36 →
  time_downstream = 6 →
  time_upstream = 6 →
  speed_still_water = 4.5 →
  effective_speed_downstream = dist_downstream / time_downstream →
  effective_speed_downstream = speed_still_water + speed_current →
  effective_speed_upstream = speed_still_water - speed_current →
  dist_upstream = effective_speed_upstream * time_upstream →
  dist_upstream = 18 :=
by
  intros h_dist_downstream h_time_downstream h_time_upstream h_speed_still_water
         h_effective_speed_downstream h_eq_speed_current h_effective_speed_upstream h_dist_upstream
  sorry

end man_swim_upstream_distance_l34_3469


namespace quadratic_root_ratio_l34_3454

theorem quadratic_root_ratio (k : ℝ) (h : ∃ r : ℝ, r ≠ 0 ∧ 3 * r * r = k * r - 12 * r + k ∧ r * r = k + 9 * r - k) : k = 27 :=
sorry

end quadratic_root_ratio_l34_3454


namespace rectangle_diagonal_l34_3426

theorem rectangle_diagonal (P A: ℝ) (hP : P = 46) (hA : A = 120) : ∃ d : ℝ, d = 17 :=
by
  -- Sorry provides the placeholder for the actual proof.
  sorry

end rectangle_diagonal_l34_3426


namespace sqrt_three_is_irrational_and_infinite_non_repeating_decimal_l34_3410

theorem sqrt_three_is_irrational_and_infinite_non_repeating_decimal :
    ∀ r : ℝ, r = Real.sqrt 3 → ¬ ∃ (m n : ℤ), n ≠ 0 ∧ r = m / n := by
    sorry

end sqrt_three_is_irrational_and_infinite_non_repeating_decimal_l34_3410


namespace percentage_of_y_l34_3406

theorem percentage_of_y (x y P : ℝ) (h1 : 0.10 * x = (P/100) * y) (h2 : x / y = 2) : P = 20 :=
sorry

end percentage_of_y_l34_3406


namespace find_y_l34_3446

theorem find_y (y : ℝ) (h₁ : (y^2 - 7*y + 12) / (y - 3) + (3*y^2 + 5*y - 8) / (3*y - 1) = -8) : y = -6 :=
sorry

end find_y_l34_3446


namespace range_of_a_l34_3418

theorem range_of_a (a : ℝ) (h : a < 1) : ∀ x : ℝ, |x - 4| + |x - 5| > a :=
by
  sorry

end range_of_a_l34_3418


namespace balls_in_boxes_l34_3445

theorem balls_in_boxes : 
  let balls := 4
  let boxes := 3
  (boxes^balls = 81) :=
by sorry

end balls_in_boxes_l34_3445


namespace magazine_cost_l34_3494

variable (b m : ℝ)

theorem magazine_cost (h1 : 2 * b + 2 * m = 26) (h2 : b + 3 * m = 27) : m = 7 :=
by
  sorry

end magazine_cost_l34_3494


namespace find_q_l34_3471

open Polynomial

-- Define the conditions for the roots of the first polynomial
def roots_of_first_eq (a b m : ℝ) (h : a * b = 3) : Prop := 
  ∀ x, (x^2 - m*x + 3) = (x - a) * (x - b)

-- Define the problem statement
theorem find_q (a b m p q : ℝ) 
  (h1 : a * b = 3) 
  (h2 : ∀ x, (x^2 - m*x + 3) = (x - a) * (x - b)) 
  (h3 : ∀ x, (x^2 - p*x + q) = (x - (a + 2/b)) * (x - (b + 2/a))) :
  q = 25 / 3 :=
sorry

end find_q_l34_3471


namespace Andrena_more_than_Debelyn_l34_3481

-- Define initial dolls count for each person
def Debelyn_initial_dolls : ℕ := 20
def Christel_initial_dolls : ℕ := 24

-- Define dolls given by Debelyn and Christel
def Debelyn_gift_dolls : ℕ := 2
def Christel_gift_dolls : ℕ := 5

-- Define remaining dolls for Debelyn and Christel after giving dolls away
def Debelyn_final_dolls : ℕ := Debelyn_initial_dolls - Debelyn_gift_dolls
def Christel_final_dolls : ℕ := Christel_initial_dolls - Christel_gift_dolls

-- Define Andrena's dolls after transactions
def Andrena_dolls : ℕ := Christel_final_dolls + 2

-- Define the Lean statement for proving Andrena has 3 more dolls than Debelyn
theorem Andrena_more_than_Debelyn : Andrena_dolls = Debelyn_final_dolls + 3 := by
  -- Here you would prove the statement
  sorry

end Andrena_more_than_Debelyn_l34_3481


namespace expression_divisible_by_1961_l34_3404

theorem expression_divisible_by_1961 (n : ℕ) : 
  (5^(2*n) * 3^(4*n) - 2^(6*n)) % 1961 = 0 := by
  sorry

end expression_divisible_by_1961_l34_3404


namespace common_ratio_of_series_l34_3414

theorem common_ratio_of_series (a1 a2 : ℚ) (h1 : a1 = 5/6) (h2 : a2 = -4/9) :
  (a2 / a1) = -8/15 :=
by
  sorry

end common_ratio_of_series_l34_3414


namespace five_digit_number_divisible_by_B_is_multiple_of_1000_l34_3475

-- Definitions
def is_five_digit_number (A : ℕ) : Prop := 10000 ≤ A ∧ A < 100000
def B (A : ℕ) := (A / 1000 * 100) + (A % 100)
def is_four_digit_number (B : ℕ) : Prop := 1000 ≤ B ∧ B < 10000

-- Main theorem
theorem five_digit_number_divisible_by_B_is_multiple_of_1000
  (A : ℕ) (hA : is_five_digit_number A)
  (hAB : ∃ k : ℕ, B A = k) :
  A % 1000 = 0 := 
sorry

end five_digit_number_divisible_by_B_is_multiple_of_1000_l34_3475


namespace gcd_pow_sub_l34_3405

theorem gcd_pow_sub (a b : ℕ) (ha : a = 2000) (hb : b = 1990) :
  Nat.gcd (2^a - 1) (2^b - 1) = 1023 :=
sorry

end gcd_pow_sub_l34_3405


namespace necessary_and_sufficient_condition_l34_3487

theorem necessary_and_sufficient_condition (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
    (∃ x : ℝ, 0 < x ∧ a^x = 2) ↔ (1 < a) := 
sorry

end necessary_and_sufficient_condition_l34_3487


namespace problem_solution_l34_3499

theorem problem_solution
  (x y : ℝ)
  (h1 : (x - y)^2 = 25)
  (h2 : x * y = -10) :
  x^2 + y^2 = 5 := sorry

end problem_solution_l34_3499


namespace transformed_line_theorem_l34_3430

theorem transformed_line_theorem (k b : ℝ) (h₁ : k = 1) (h₂ : b = 1) (x : ℝ) :
  (k * x + b > 0) ↔ (x > -1) :=
by sorry

end transformed_line_theorem_l34_3430


namespace gcd_lcm_product_360_distinct_gcd_values_l34_3451

/-- 
  Given two integers a and b, such that the product of their gcd and lcm is 360,
  we need to prove that the number of distinct possible values for their gcd is 9.
--/
theorem gcd_lcm_product_360_distinct_gcd_values :
  ∀ (a b : ℕ), gcd a b * lcm a b = 360 → 
  (∃ gcd_values : Finset ℕ, gcd_values.card = 9 ∧ ∀ g, g ∈ gcd_values ↔ g = gcd a b) :=
by
  sorry

end gcd_lcm_product_360_distinct_gcd_values_l34_3451


namespace cube_sum_eq_one_l34_3427

theorem cube_sum_eq_one (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 2) (h3 : abc = 1) : a^3 + b^3 + c^3 = 1 :=
sorry

end cube_sum_eq_one_l34_3427


namespace solve_for_k_l34_3491

theorem solve_for_k (p q : ℝ) (k : ℝ) (hpq : 3 * p^2 + 6 * p + k = 0) (hq : 3 * q^2 + 6 * q + k = 0) 
    (h_diff : |p - q| = (1 / 2) * (p^2 + q^2)) : k = -16 + 12 * Real.sqrt 2 ∨ k = -16 - 12 * Real.sqrt 2 :=
by
  sorry

end solve_for_k_l34_3491


namespace poly_coeff_sum_l34_3423

variable {a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ}

theorem poly_coeff_sum :
  (∀ x : ℝ, (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 + 6 * a_6 = 12 :=
by
  sorry

end poly_coeff_sum_l34_3423


namespace red_marbles_difference_l34_3447

theorem red_marbles_difference 
  (x y : ℕ) 
  (h1 : 7 * x + 3 * x = 140) 
  (h2 : 3 * y + 2 * y = 140)
  (h3 : 10 * x = 5 * y) : 
  7 * x - 3 * y = 20 := 
by 
  sorry

end red_marbles_difference_l34_3447


namespace seating_arrangement_l34_3409

-- Define the problem in Lean
theorem seating_arrangement :
  let n := 9   -- Total number of people
  let r := 7   -- Number of seats at the circular table
  let combinations := Nat.choose n 2  -- Ways to select 2 people not seated
  let factorial (k : ℕ) := Nat.recOn k 1 (λ k' acc => (k' + 1) * acc)
  let arrangements := factorial (r - 1)  -- Ways to seat 7 people around a circular table
  combinations * arrangements = 25920 :=
by
  -- In Lean, sorry is used to indicate that we skip the proof for now.
  sorry

end seating_arrangement_l34_3409


namespace intersection_points_l34_3452

def equation1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9
def equation2 (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 25

theorem intersection_points :
  ∃ (x1 y1 x2 y2 : ℝ),
    equation1 x1 y1 ∧ equation2 x1 y1 ∧
    equation1 x2 y2 ∧ equation2 x2 y2 ∧
    (x1, y1) ≠ (x2, y2) ∧
    ∀ (x y : ℝ), equation1 x y ∧ equation2 x y → (x, y) = (x1, y1) ∨ (x, y) = (x2, y2) := sorry

end intersection_points_l34_3452


namespace money_left_after_bike_purchase_l34_3490

-- Definitions based on conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def quarter_value : ℝ := 0.25
def bike_cost : ℝ := 180

-- The theorem statement
theorem money_left_after_bike_purchase : (jars * quarters_per_jar * quarter_value) - bike_cost = 20 := by
  sorry

end money_left_after_bike_purchase_l34_3490


namespace distinct_real_roots_l34_3401

-- Define the polynomial equation as a Lean function
def polynomial (a x : ℝ) : ℝ :=
  (a + 1) * (x ^ 2 + 1) ^ 2 - (2 * a + 3) * (x ^ 2 + 1) * x + (a + 2) * x ^ 2

-- The theorem we need to prove
theorem distinct_real_roots (a : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ polynomial a x = 0 ∧ polynomial a y = 0) ↔ a ≠ -1 :=
by
  sorry

end distinct_real_roots_l34_3401


namespace standard_deviation_less_than_mean_l34_3493

theorem standard_deviation_less_than_mean 
  (μ : ℝ) (σ : ℝ) (x : ℝ) 
  (h1 : μ = 14.5) 
  (h2 : σ = 1.5) 
  (h3 : x = 11.5) : 
  (μ - x) / σ = 2 :=
by
  rw [h1, h2, h3]
  norm_num

end standard_deviation_less_than_mean_l34_3493


namespace monkeys_and_apples_l34_3407

theorem monkeys_and_apples
  {x a : ℕ}
  (h1 : a = 3 * x + 6)
  (h2 : 0 < a - 4 * (x - 1) ∧ a - 4 * (x - 1) < 4)
  : (x = 7 ∧ a = 27) ∨ (x = 8 ∧ a = 30) ∨ (x = 9 ∧ a = 33) :=
sorry

end monkeys_and_apples_l34_3407


namespace Trishul_invested_less_than_Raghu_l34_3442

-- Definitions based on conditions
def Raghu_investment : ℝ := 2500
def Total_investment : ℝ := 7225

def Vishal_invested_more_than_Trishul (T V : ℝ) : Prop :=
  V = 1.10 * T

noncomputable def percentage_decrease (original decrease : ℝ) : ℝ :=
  (decrease / original) * 100

theorem Trishul_invested_less_than_Raghu (T V : ℝ) 
  (h1 : Vishal_invested_more_than_Trishul T V)
  (h2 : T + V + Raghu_investment = Total_investment) :
  percentage_decrease Raghu_investment (Raghu_investment - T) = 10 := by
  sorry

end Trishul_invested_less_than_Raghu_l34_3442


namespace complex_projective_form_and_fixed_points_l34_3425

noncomputable def complex_projective_transformation (a b c d : ℂ) (z : ℂ) : ℂ :=
  (a * z + b) / (c * z + d)

theorem complex_projective_form_and_fixed_points (a b c d : ℂ) (h : d ≠ 0) :
  (∃ (f : ℂ → ℂ), ∀ z, f z = complex_projective_transformation a b c d z)
  ∧ ∃ (z₁ z₂ : ℂ), complex_projective_transformation a b c d z₁ = z₁ ∧ complex_projective_transformation a b c d z₂ = z₂ :=
by
  -- omitted proof, this is just the statement
  sorry

end complex_projective_form_and_fixed_points_l34_3425


namespace bicycle_meets_light_vehicle_l34_3420

noncomputable def meeting_time (v_1 v_2 v_3 v_4 : ℚ) : ℚ :=
  let x := 2 * (v_1 + v_4)
  let y := 6 * (v_2 - v_4)
  (x + y) / (v_3 + v_4) + 12

theorem bicycle_meets_light_vehicle (v_1 v_2 v_3 v_4 : ℚ) (h1 : 2 * (v_1 + v_4) = x)
  (h2 : x + y = 4 * (v_1 + v_2))
  (h3 : x + y = 5 * (v_2 + v_3))
  (h4 : 6 * (v_2 - v_4) = y) :
  meeting_time v_1 v_2 v_3 v_4 = 15 + 1/3 :=
by
  sorry

end bicycle_meets_light_vehicle_l34_3420


namespace problem1_problem2_problem3_l34_3412

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 1 ≤ x then x^2 - 2 * a * x + a
  else if 0 < x then 2 * x + a / x
  else 0 -- Undefined for x ≤ 0

theorem problem1 (a : ℝ) :
  (∀ x y : ℝ, (0 < x ∧ x < y) → f a x < f a y) ↔ (a ≤ -1 / 2) :=
sorry
  
theorem problem2 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ f a x1 = 1 ∧ f a x2 = 1 ∧ f a x3 = 1) ↔ (0 < a ∧ a < 1 / 8) :=
sorry

theorem problem3 (a : ℝ) :
  (∀ x : ℝ, f a x ≥ x - 2 * a) ↔ (0 ≤ a ∧ a ≤ 1 + Real.sqrt 3 / 2) :=
sorry

end problem1_problem2_problem3_l34_3412


namespace quadratic_distinct_real_roots_l34_3477

theorem quadratic_distinct_real_roots (a : ℝ) (h : a ≠ 1) : 
(a < 2) → 
(∃ x y : ℝ, x ≠ y ∧ (a-1)*x^2 - 2*x + 1 = 0 ∧ (a-1)*y^2 - 2*y + 1 = 0) :=
sorry

end quadratic_distinct_real_roots_l34_3477


namespace additional_slow_workers_needed_l34_3456

-- Definitions based on conditions
def production_per_worker_fast (m : ℕ) (n : ℕ) (a : ℕ) : ℚ := m / (n * a)
def production_per_worker_slow (m : ℕ) (n : ℕ) (b : ℕ) : ℚ := m / (n * b)

def required_daily_production (p : ℕ) (q : ℕ) : ℚ := p / q

def contribution_fast_workers (m : ℕ) (n : ℕ) (a : ℕ) (c : ℕ) : ℚ :=
  (m * c) / (n * a)

def remaining_production (p : ℕ) (q : ℕ) (m : ℕ) (n : ℕ) (a : ℕ) (c : ℕ) : ℚ :=
  (p / q) - ((m * c) / (n * a))

def required_slow_workers (p : ℕ) (q : ℕ) (m : ℕ) (n : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) : ℚ :=
  ((p * n * a - q * m * c) * b) / (q * m * a)

theorem additional_slow_workers_needed (m n a b p q c : ℕ) :
  required_slow_workers p q m n a b c = ((p * n * a - q * m * c) * b) / (q * m * a) := by
  sorry

end additional_slow_workers_needed_l34_3456


namespace paper_area_l34_3476

variable (L W : ℕ)

theorem paper_area (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) : L * W = 140 := by
  sorry

end paper_area_l34_3476
