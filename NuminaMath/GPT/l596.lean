import Mathlib

namespace NUMINAMATH_GPT_min_cubes_required_l596_59618

theorem min_cubes_required (length width height volume_cube : ℝ) 
  (h_length : length = 14.5) 
  (h_width : width = 17.8) 
  (h_height : height = 7.2) 
  (h_volume_cube : volume_cube = 3) : 
  ⌈(length * width * height) / volume_cube⌉ = 624 := sorry

end NUMINAMATH_GPT_min_cubes_required_l596_59618


namespace NUMINAMATH_GPT_volume_of_spheres_l596_59688

noncomputable def sphere_volume (a : ℝ) : ℝ :=
  (4 / 3) * Real.pi * ((3 * a - a * Real.sqrt 3) / 4)^3

theorem volume_of_spheres (a : ℝ) : 
  ∃ r : ℝ, r = (3 * a - a * Real.sqrt 3) / 4 ∧ 
  sphere_volume a = (4 / 3) * Real.pi * r^3 := 
sorry

end NUMINAMATH_GPT_volume_of_spheres_l596_59688


namespace NUMINAMATH_GPT_ratio_of_a_b_l596_59614

-- Define the problem
theorem ratio_of_a_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it.
  sorry

end NUMINAMATH_GPT_ratio_of_a_b_l596_59614


namespace NUMINAMATH_GPT_triangle_area_l596_59607

open Real

-- Define the angles A and C, side a, and state the goal as proving the area
theorem triangle_area (A C : ℝ) (a : ℝ) (hA : A = 30 * (π / 180)) (hC : C = 45 * (π / 180)) (ha : a = 2) : 
  (1 / 2) * ((sqrt 6 + sqrt 2) * (2 * sqrt 2) * sin (30 * (π / 180))) = sqrt 3 + 1 := 
by
  sorry

end NUMINAMATH_GPT_triangle_area_l596_59607


namespace NUMINAMATH_GPT_root_of_equation_l596_59605

theorem root_of_equation (x : ℝ) : 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ↔ x = 31 := 
by 
  sorry

end NUMINAMATH_GPT_root_of_equation_l596_59605


namespace NUMINAMATH_GPT_cos_trig_identity_l596_59694

theorem cos_trig_identity (α : Real) 
  (h : Real.cos (Real.pi / 6 - α) = 3 / 5) : 
  Real.cos (5 * Real.pi / 6 + α) = - (3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_cos_trig_identity_l596_59694


namespace NUMINAMATH_GPT_smallest_integer_inequality_l596_59601

theorem smallest_integer_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧ 
           (∀ m : ℤ, m < n → ¬∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_inequality_l596_59601


namespace NUMINAMATH_GPT_stack_of_logs_total_l596_59616

-- Define the given conditions as variables and constants in Lean
def bottom_row : Nat := 15
def top_row : Nat := 4
def rows : Nat := bottom_row - top_row + 1
def sum_arithmetic_series (a l n : Nat) : Nat := n * (a + l) / 2

-- Define the main theorem to prove
theorem stack_of_logs_total : sum_arithmetic_series top_row bottom_row rows = 114 :=
by
  -- Here you will normally provide the proof
  sorry

end NUMINAMATH_GPT_stack_of_logs_total_l596_59616


namespace NUMINAMATH_GPT_length_EF_l596_59674

theorem length_EF
  (AB CD GH EF : ℝ)
  (h1 : AB = 180)
  (h2 : CD = 120)
  (h3 : AB = 2 * GH)
  (h4 : CD = 2 * EF) :
  EF = 45 :=
by
  sorry

end NUMINAMATH_GPT_length_EF_l596_59674


namespace NUMINAMATH_GPT_min_xy_when_a_16_min_expr_when_a_0_l596_59602

-- Problem 1: Minimum value of xy when a = 16
theorem min_xy_when_a_16 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x * y = x + 4 * y + 16) : 16 ≤ x * y :=
    sorry

-- Problem 2: Minimum value of x + y + 2 / x + 1 / (2 * y) when a = 0
theorem min_expr_when_a_0 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x * y = x + 4 * y) : (11 : ℝ) / 2 ≤ x + y + 2 / x + 1 / (2 * y) :=
    sorry

end NUMINAMATH_GPT_min_xy_when_a_16_min_expr_when_a_0_l596_59602


namespace NUMINAMATH_GPT_vasya_gift_ways_l596_59691

theorem vasya_gift_ways :
  let cars := 7
  let constructor_sets := 5
  (cars * constructor_sets) + (Nat.choose cars 2) + (Nat.choose constructor_sets 2) = 66 :=
by
  let cars := 7
  let constructor_sets := 5
  sorry

end NUMINAMATH_GPT_vasya_gift_ways_l596_59691


namespace NUMINAMATH_GPT_rahul_work_days_l596_59627

variable (R : ℕ)

theorem rahul_work_days
  (rajesh_days : ℕ := 2)
  (total_money : ℕ := 355)
  (rahul_share : ℕ := 142)
  (rajesh_share : ℕ := total_money - rahul_share)
  (payment_ratio : ℕ := rahul_share / rajesh_share)
  (work_rate_ratio : ℕ := rajesh_days / R) :
  payment_ratio = work_rate_ratio → R = 3 :=
by
  sorry

end NUMINAMATH_GPT_rahul_work_days_l596_59627


namespace NUMINAMATH_GPT_beakers_with_copper_l596_59661

theorem beakers_with_copper :
  ∀ (total_beakers no_copper_beakers beakers_with_copper drops_per_beaker total_drops_used : ℕ),
    total_beakers = 22 →
    no_copper_beakers = 7 →
    drops_per_beaker = 3 →
    total_drops_used = 45 →
    total_drops_used = drops_per_beaker * beakers_with_copper →
    total_beakers = beakers_with_copper + no_copper_beakers →
    beakers_with_copper = 15 := 
-- inserting the placeholder proof 'sorry'
sorry

end NUMINAMATH_GPT_beakers_with_copper_l596_59661


namespace NUMINAMATH_GPT_conditional_probability_second_sci_given_first_sci_l596_59629

-- Definitions based on the conditions
def total_questions : ℕ := 6
def science_questions : ℕ := 4
def humanities_questions : ℕ := 2
def first_draw_is_science : Prop := true

-- The statement we want to prove
theorem conditional_probability_second_sci_given_first_sci : 
    first_draw_is_science → (science_questions - 1) / (total_questions - 1) = 3 / 5 := 
by
  intro h
  have num_sci_after_first : ℕ := science_questions - 1
  have total_after_first : ℕ := total_questions - 1
  have prob_second_sci := num_sci_after_first / total_after_first
  sorry

end NUMINAMATH_GPT_conditional_probability_second_sci_given_first_sci_l596_59629


namespace NUMINAMATH_GPT_AndyCoordinatesAfter1500Turns_l596_59696

/-- Definition for Andy's movement rules given his starting position. -/
def AndyPositionAfterTurns (turns : ℕ) : ℤ × ℤ :=
  let rec move (x y : ℤ) (length : ℤ) (dir : ℕ) (remainingTurns : ℕ) : ℤ × ℤ :=
    match remainingTurns with
    | 0 => (x, y)
    | n+1 => 
        let (dx, dy) := match dir % 4 with
                        | 0 => (0, 1)
                        | 1 => (1, 0)
                        | 2 => (0, -1)
                        | _ => (-1, 0)
        move (x + dx * length) (y + dy * length) (length + 1) (dir + 1) n
  move (-30) 25 2 0 turns

theorem AndyCoordinatesAfter1500Turns :
  AndyPositionAfterTurns 1500 = (-280141, 280060) :=
by
  sorry

end NUMINAMATH_GPT_AndyCoordinatesAfter1500Turns_l596_59696


namespace NUMINAMATH_GPT_BDD1H_is_Spatial_in_Cube_l596_59631

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A B C D A1 B1 C1 D1 : Point3D)
(midpoint_B1C1 : Point3D)
(middle_B1C1 : midpoint_B1C1 = ⟨(B1.x + C1.x) / 2, (B1.y + C1.y) / 2, (B1.z + C1.z) / 2⟩)

def is_not_planar (a b c d : Point3D) : Prop :=
¬ ∃ α β γ δ : ℝ, α * a.x + β * a.y + γ * a.z + δ = 0 ∧ 
                α * b.x + β * b.y + γ * b.z + δ = 0 ∧ 
                α * c.x + β * c.y + γ * c.z + δ = 0 ∧ 
                α * d.x + β * d.y + γ * d.z + δ = 0

def BDD1H_is_spatial (cube : Cube) : Prop :=
is_not_planar cube.B cube.D cube.D1 cube.midpoint_B1C1

theorem BDD1H_is_Spatial_in_Cube (cube : Cube) : BDD1H_is_spatial cube :=
sorry

end NUMINAMATH_GPT_BDD1H_is_Spatial_in_Cube_l596_59631


namespace NUMINAMATH_GPT_white_given_popped_l596_59663

-- Define the conditions
def white_kernels : ℚ := 1 / 2
def yellow_kernels : ℚ := 1 / 3
def blue_kernels : ℚ := 1 / 6

def white_kernels_pop : ℚ := 3 / 4
def yellow_kernels_pop : ℚ := 1 / 2
def blue_kernels_pop : ℚ := 1 / 3

def probability_white_popped : ℚ := white_kernels * white_kernels_pop
def probability_yellow_popped : ℚ := yellow_kernels * yellow_kernels_pop
def probability_blue_popped : ℚ := blue_kernels * blue_kernels_pop

def probability_popped : ℚ := probability_white_popped + probability_yellow_popped + probability_blue_popped

-- The theorem to be proved
theorem white_given_popped : (probability_white_popped / probability_popped) = (27 / 43) := 
by sorry

end NUMINAMATH_GPT_white_given_popped_l596_59663


namespace NUMINAMATH_GPT_outer_boundary_diameter_l596_59639

-- Define the given conditions
def fountain_diameter : ℝ := 12
def walking_path_width : ℝ := 6
def garden_ring_width : ℝ := 10

-- Define what we need to prove
theorem outer_boundary_diameter :
  2 * (fountain_diameter / 2 + garden_ring_width + walking_path_width) = 44 :=
by
  sorry

end NUMINAMATH_GPT_outer_boundary_diameter_l596_59639


namespace NUMINAMATH_GPT_quadratic_condition_l596_59684

theorem quadratic_condition (m : ℝ) (h1 : m^2 - 2 = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_condition_l596_59684


namespace NUMINAMATH_GPT_number_of_strikers_l596_59653

theorem number_of_strikers (goalies defenders total_players midfielders strikers : ℕ)
  (h1 : goalies = 3)
  (h2 : defenders = 10)
  (h3 : midfielders = 2 * defenders)
  (h4 : total_players = 40)
  (h5 : total_players = goalies + defenders + midfielders + strikers) :
  strikers = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_strikers_l596_59653


namespace NUMINAMATH_GPT_percentage_error_l596_59611

theorem percentage_error (x : ℝ) (hx : x ≠ 0) :
  let correct_result := 10 * x
  let incorrect_result := x / 10
  let error := correct_result - incorrect_result
  let percentage_error := (error / correct_result) * 100
  percentage_error = 99 :=
by
  sorry

end NUMINAMATH_GPT_percentage_error_l596_59611


namespace NUMINAMATH_GPT_largest_number_l596_59619

theorem largest_number 
  (A : ℝ) (B : ℝ) (C : ℝ) (D : ℝ) (E : ℝ)
  (hA : A = 0.986)
  (hB : B = 0.9851)
  (hC : C = 0.9869)
  (hD : D = 0.9807)
  (hE : E = 0.9819)
  : C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  sorry

end NUMINAMATH_GPT_largest_number_l596_59619


namespace NUMINAMATH_GPT_incorrect_assignment_statement_l596_59624

theorem incorrect_assignment_statement :
  ∀ (a x y : ℕ), ¬(x * y = a) := by
sorry

end NUMINAMATH_GPT_incorrect_assignment_statement_l596_59624


namespace NUMINAMATH_GPT_cafeteria_B_turnover_higher_in_May_l596_59677

noncomputable def initial_turnover (X a r : ℝ) : Prop :=
  ∃ (X a r : ℝ),
    (X + 8 * a = X * (1 + r) ^ 8) ∧
    ((X + 4 * a) < (X * (1 + r) ^ 4))

theorem cafeteria_B_turnover_higher_in_May (X a r : ℝ) :
    (X + 8 * a = X * (1 + r) ^ 8) → (X + 4 * a < X * (1 + r) ^ 4) :=
  sorry

end NUMINAMATH_GPT_cafeteria_B_turnover_higher_in_May_l596_59677


namespace NUMINAMATH_GPT_input_equals_output_l596_59680

theorem input_equals_output (x : ℝ) :
  (x ≤ 1 → 2 * x - 3 = x) ∨ (x > 1 → x^2 - 3 * x + 3 = x) ↔ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_input_equals_output_l596_59680


namespace NUMINAMATH_GPT_hotdogs_sold_l596_59678

-- Definitions of initial and remaining hotdogs
def initial : ℕ := 99
def remaining : ℕ := 97

-- The statement that needs to be proven
theorem hotdogs_sold : initial - remaining = 2 :=
by
  sorry

end NUMINAMATH_GPT_hotdogs_sold_l596_59678


namespace NUMINAMATH_GPT_cos_832_eq_cos_l596_59643

theorem cos_832_eq_cos (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * Real.pi / 180) = Real.cos (832 * Real.pi / 180)) : n = 112 := 
  sorry

end NUMINAMATH_GPT_cos_832_eq_cos_l596_59643


namespace NUMINAMATH_GPT_color_ball_ratios_l596_59620

theorem color_ball_ratios (white_balls red_balls blue_balls : ℕ)
  (h_white : white_balls = 12)
  (h_red_ratio : 4 * red_balls = 3 * white_balls)
  (h_blue_ratio : 4 * blue_balls = 2 * white_balls) :
  red_balls = 9 ∧ blue_balls = 6 :=
by
  sorry

end NUMINAMATH_GPT_color_ball_ratios_l596_59620


namespace NUMINAMATH_GPT_steven_set_aside_pears_l596_59676

theorem steven_set_aside_pears :
  ∀ (apples pears grapes neededSeeds seedPerApple seedPerPear seedPerGrape : ℕ),
    apples = 4 →
    grapes = 9 →
    neededSeeds = 60 →
    seedPerApple = 6 →
    seedPerPear = 2 →
    seedPerGrape = 3 →
    (neededSeeds - 3) = (apples * seedPerApple + grapes * seedPerGrape + pears * seedPerPear) →
    pears = 3 :=
by
  intros apples pears grapes neededSeeds seedPerApple seedPerPear seedPerGrape
  intros h_apple h_grape h_needed h_seedApple h_seedPear h_seedGrape
  intros h_totalSeeds
  sorry

end NUMINAMATH_GPT_steven_set_aside_pears_l596_59676


namespace NUMINAMATH_GPT_find_number_of_pencils_l596_59659

-- Define the conditions
def number_of_people : Nat := 6
def notebooks_per_person : Nat := 9
def number_of_notebooks : Nat := number_of_people * notebooks_per_person
def pencils_multiplier : Nat := 6
def number_of_pencils : Nat := pencils_multiplier * number_of_notebooks

-- Prove the main statement
theorem find_number_of_pencils : number_of_pencils = 324 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_pencils_l596_59659


namespace NUMINAMATH_GPT_inverse_proportion_function_m_neg_l596_59637

theorem inverse_proportion_function_m_neg
  (x : ℝ) (y : ℝ) (m : ℝ)
  (h1 : y = m / x)
  (h2 : (x < 0 → y > 0) ∧ (x > 0 → y < 0)) :
  m < 0 :=
sorry

end NUMINAMATH_GPT_inverse_proportion_function_m_neg_l596_59637


namespace NUMINAMATH_GPT_slope_of_line_through_intersecting_points_of_circles_l596_59695

theorem slope_of_line_through_intersecting_points_of_circles :
  let circle1 (x y : ℝ) := x^2 + y^2 - 6*x + 4*y - 5 = 0
  let circle2 (x y : ℝ) := x^2 + y^2 - 10*x + 16*y + 24 = 0
  ∀ (C D : ℝ × ℝ), circle1 C.1 C.2 → circle2 C.1 C.2 → circle1 D.1 D.2 → circle2 D.1 D.2 → 
  let dx := D.1 - C.1
  let dy := D.2 - C.2
  dx ≠ 0 → dy / dx = 1 / 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_slope_of_line_through_intersecting_points_of_circles_l596_59695


namespace NUMINAMATH_GPT_class_student_difference_l596_59682

theorem class_student_difference (A B : ℕ) (h : A - 4 = B + 4) : A - B = 8 := by
  sorry

end NUMINAMATH_GPT_class_student_difference_l596_59682


namespace NUMINAMATH_GPT_angle_difference_parallelogram_l596_59634

theorem angle_difference_parallelogram (A B : ℝ) (hA : A = 55) (h1 : A + B = 180) :
  B - A = 70 := 
by
  sorry

end NUMINAMATH_GPT_angle_difference_parallelogram_l596_59634


namespace NUMINAMATH_GPT_perfect_square_eq_m_val_l596_59640

theorem perfect_square_eq_m_val (m : ℝ) (h : ∃ a : ℝ, x^2 - m * x + 49 = (x - a)^2) : m = 14 ∨ m = -14 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_eq_m_val_l596_59640


namespace NUMINAMATH_GPT_no_such_natural_numbers_l596_59672

theorem no_such_natural_numbers :
  ¬(∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧
  (b ∣ a^2 - 1) ∧ (c ∣ a^2 - 1) ∧
  (a ∣ b^2 - 1) ∧ (c ∣ b^2 - 1) ∧
  (a ∣ c^2 - 1) ∧ (b ∣ c^2 - 1)) :=
by sorry

end NUMINAMATH_GPT_no_such_natural_numbers_l596_59672


namespace NUMINAMATH_GPT_cos_minus_sin_eq_neg_one_fifth_l596_59626

theorem cos_minus_sin_eq_neg_one_fifth
  (α : ℝ)
  (h1 : Real.sin (2 * α) = 24 / 25)
  (h2 : π < α ∧ α < 5 * π / 4) :
  Real.cos α - Real.sin α = -1 / 5 := sorry

end NUMINAMATH_GPT_cos_minus_sin_eq_neg_one_fifth_l596_59626


namespace NUMINAMATH_GPT_tree_planting_total_l596_59641

theorem tree_planting_total (t4 t5 t6 : ℕ) 
  (h1 : t4 = 30)
  (h2 : t5 = 2 * t4)
  (h3 : t6 = 3 * t5 - 30) : 
  t4 + t5 + t6 = 240 := 
by 
  sorry

end NUMINAMATH_GPT_tree_planting_total_l596_59641


namespace NUMINAMATH_GPT_max_free_squares_l596_59604

theorem max_free_squares (n : ℕ) :
  ∀ (initial_positions : ℕ), 
    (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → initial_positions = 2) →
    (∀ (i j : ℕ) (move1 move2 : ℕ × ℕ),
       1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n →
       move1 = (i + 1, j) ∨ move1 = (i - 1, j) ∨ move1 = (i, j + 1) ∨ move1 = (i, j - 1) →
       move2 = (i + 1, j) ∨ move2 = (i - 1, j) ∨ move2 = (i, j + 1) ∨ move2 = (i, j - 1) →
       move1 ≠ move2) →
    ∃ free_squares : ℕ, free_squares = n^2 :=
by
  sorry

end NUMINAMATH_GPT_max_free_squares_l596_59604


namespace NUMINAMATH_GPT_no_integer_coordinates_between_A_and_B_l596_59650

section
variable (A B : ℤ × ℤ)
variable (Aeq : A = (2, 3))
variable (Beq : B = (50, 305))

theorem no_integer_coordinates_between_A_and_B :
  (∀ P : ℤ × ℤ, P.1 > 2 ∧ P.1 < 50 ∧ P.2 = (151 * P.1 - 230) / 24 → False) :=
by
  sorry
end

end NUMINAMATH_GPT_no_integer_coordinates_between_A_and_B_l596_59650


namespace NUMINAMATH_GPT_find_digits_of_six_two_digit_sum_equals_528_l596_59648

theorem find_digits_of_six_two_digit_sum_equals_528
  (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_digits : a < 10 ∧ b < 10 ∧ c < 10)
  (h_sum_six_numbers : (10 * a + b) + (10 * a + c) + (10 * b + c) + (10 * b + a) + (10 * c + a) + (10 * c + b) = 528) :
  (a = 7 ∧ b = 8 ∧ c = 9) := 
sorry

end NUMINAMATH_GPT_find_digits_of_six_two_digit_sum_equals_528_l596_59648


namespace NUMINAMATH_GPT_distances_equal_l596_59622

noncomputable def distance_from_point_to_line (x y m : ℝ) : ℝ :=
  |m * x + y + 3| / Real.sqrt (m^2 + 1)

theorem distances_equal (m : ℝ) :
  distance_from_point_to_line 3 2 m = distance_from_point_to_line (-1) 4 m ↔
  (m = 1 / 2 ∨ m = -6) := 
sorry

end NUMINAMATH_GPT_distances_equal_l596_59622


namespace NUMINAMATH_GPT_reciprocal_of_repeating_decimal_l596_59647

theorem reciprocal_of_repeating_decimal :
  (1 / (0.33333333 : ℚ)) = 3 := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_repeating_decimal_l596_59647


namespace NUMINAMATH_GPT_min_total_cost_of_container_l596_59635

-- Definitions from conditions
def container_volume := 4 -- m^3
def container_height := 1 -- m
def cost_per_square_meter_base : ℝ := 20
def cost_per_square_meter_sides : ℝ := 10

-- Proving the minimum total cost
theorem min_total_cost_of_container :
  ∃ (a b : ℝ), a * b = container_volume ∧
                (20 * (a + b) + 20 * (a * b)) = 160 :=
by
  sorry

end NUMINAMATH_GPT_min_total_cost_of_container_l596_59635


namespace NUMINAMATH_GPT_window_treatments_cost_l596_59658

def cost_of_sheers (n : ℕ) (cost_per_pair : ℝ) : ℝ := n * cost_per_pair
def cost_of_drapes (n : ℕ) (cost_per_pair : ℝ) : ℝ := n * cost_per_pair
def total_cost (n : ℕ) (cost_sheers : ℝ) (cost_drapes : ℝ) : ℝ :=
  cost_of_sheers n cost_sheers + cost_of_drapes n cost_drapes

theorem window_treatments_cost :
  total_cost 3 40 60 = 300 :=
by
  sorry

end NUMINAMATH_GPT_window_treatments_cost_l596_59658


namespace NUMINAMATH_GPT_equal_probability_among_children_l596_59613

theorem equal_probability_among_children
    (n : ℕ := 100)
    (p : ℝ := 0.232818)
    (k : ℕ := 18)
    (h_pos : 0 < p)
    (h_lt : p < 1)
    (num_outcomes : ℕ := 2^k) :
  ∃ (dist : Fin n → Fin num_outcomes),
    ∀ i : Fin num_outcomes, ∃ j : Fin n, dist j = i ∧ p ^ k * (1 - p) ^ (num_outcomes - k) = 1 / n :=
by
  sorry

end NUMINAMATH_GPT_equal_probability_among_children_l596_59613


namespace NUMINAMATH_GPT_family_snails_l596_59669

def total_snails_family (n1 n2 n3 n4 : ℕ) (mother_find : ℕ) : ℕ :=
  n1 + n2 + n3 + mother_find

def first_ducklings_snails (num_ducklings : ℕ) (snails_per_duckling : ℕ) : ℕ :=
  num_ducklings * snails_per_duckling

def remaining_ducklings_snails (num_ducklings : ℕ) (mother_snails : ℕ) : ℕ :=
  num_ducklings * (mother_snails / 2)

def mother_find_snails (snails_group1 : ℕ) (snails_group2 : ℕ) : ℕ :=
  3 * (snails_group1 + snails_group2)

theorem family_snails : 
  ∀ (ducklings : ℕ) (group1_ducklings group2_ducklings : ℕ) 
    (snails1 snails2 : ℕ) 
    (total_ducklings : ℕ), 
    ducklings = 8 →
    group1_ducklings = 3 → 
    group2_ducklings = 3 → 
    snails1 = 5 →
    snails2 = 9 →
    total_ducklings = group1_ducklings + group2_ducklings + 2 →
    total_snails_family 
      (first_ducklings_snails group1_ducklings snails1)
      (first_ducklings_snails group2_ducklings snails2)
      (remaining_ducklings_snails 2 (mother_find_snails 
        (first_ducklings_snails group1_ducklings snails1)
        (first_ducklings_snails group2_ducklings snails2)))
      (mother_find_snails 
        (first_ducklings_snails group1_ducklings snails1)
        (first_ducklings_snails group2_ducklings snails2)) 
    = 294 :=
by intros; sorry

end NUMINAMATH_GPT_family_snails_l596_59669


namespace NUMINAMATH_GPT_maxSUVMileage_l596_59660

noncomputable def maxSUVDistance : ℝ := 217.12

theorem maxSUVMileage 
    (tripGal : ℝ) (mpgHighway : ℝ) (mpgCity : ℝ)
    (regularHighwayRatio : ℝ) (regularCityRatio : ℝ)
    (peakHighwayRatio : ℝ) (peakCityRatio : ℝ) :
    tripGal = 23 →
    mpgHighway = 12.2 →
    mpgCity = 7.6 →
    regularHighwayRatio = 0.4 →
    regularCityRatio = 0.6 →
    peakHighwayRatio = 0.25 →
    peakCityRatio = 0.75 →
    max ((tripGal * regularHighwayRatio * mpgHighway) + (tripGal * regularCityRatio * mpgCity))
        ((tripGal * peakHighwayRatio * mpgHighway) + (tripGal * peakCityRatio * mpgCity)) = maxSUVDistance :=
by
  intros
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_maxSUVMileage_l596_59660


namespace NUMINAMATH_GPT_distance_between_A_and_B_l596_59679

-- Definitions for the problem
def speed_fast_train := 65 -- speed of the first train in km/h
def speed_slow_train := 29 -- speed of the second train in km/h
def time_difference := 5   -- difference in hours

-- Given conditions and the final equation leading to the proof
theorem distance_between_A_and_B :
  ∃ (D : ℝ), D = 9425 / 36 :=
by
  existsi (9425 / 36 : ℝ)
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l596_59679


namespace NUMINAMATH_GPT_seventh_term_of_geometric_sequence_l596_59644

theorem seventh_term_of_geometric_sequence :
  ∀ (a r : ℝ), (a * r ^ 3 = 16) → (a * r ^ 8 = 2) → (a * r ^ 6 = 2) :=
by
  intros a r h1 h2
  sorry

end NUMINAMATH_GPT_seventh_term_of_geometric_sequence_l596_59644


namespace NUMINAMATH_GPT_circumscribed_circle_area_l596_59667

theorem circumscribed_circle_area (x y c : ℝ)
  (h1 : x + y + c = 24)
  (h2 : x * y = 48)
  (h3 : x^2 + y^2 = c^2) :
  ∃ R : ℝ, (x + y + 2 * R = 24) ∧ (π * R^2 = 25 * π) := 
sorry

end NUMINAMATH_GPT_circumscribed_circle_area_l596_59667


namespace NUMINAMATH_GPT_abs_sum_plus_two_eq_sum_abs_l596_59698

theorem abs_sum_plus_two_eq_sum_abs {a b c : ℤ} (h : |a + b + c| + 2 = |a| + |b| + |c|) :
  a^2 = 1 ∨ b^2 = 1 ∨ c^2 = 1 :=
sorry

end NUMINAMATH_GPT_abs_sum_plus_two_eq_sum_abs_l596_59698


namespace NUMINAMATH_GPT_subset_implies_range_a_intersection_implies_range_a_l596_59638

noncomputable def setA : Set ℝ := {x | -1 < x ∧ x < 2}
noncomputable def setB (a : ℝ) : Set ℝ := {x | 2 * a - 1 < x ∧ x < 2 * a + 3}

theorem subset_implies_range_a (a : ℝ) : (setA ⊆ setB a) → (-1/2 ≤ a ∧ a ≤ 0) :=
by
  sorry

theorem intersection_implies_range_a (a : ℝ) : (setA ∩ setB a = ∅) → (a ≤ -2 ∨ a ≥ 3/2) :=
by
  sorry

end NUMINAMATH_GPT_subset_implies_range_a_intersection_implies_range_a_l596_59638


namespace NUMINAMATH_GPT_sum_of_distinct_nums_l596_59687

theorem sum_of_distinct_nums (m n p q : ℕ) (hmn : m ≠ n) (hmp : m ≠ p) (hmq : m ≠ q) 
(hnp : n ≠ p) (hnq : n ≠ q) (hpq : p ≠ q) (pos_m : 0 < m) (pos_n : 0 < n) 
(pos_p : 0 < p) (pos_q : 0 < q) (h : (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4) : 
  m + n + p + q = 24 :=
sorry

end NUMINAMATH_GPT_sum_of_distinct_nums_l596_59687


namespace NUMINAMATH_GPT_greater_num_792_l596_59685

theorem greater_num_792 (x y : ℕ) (h1 : x + y = 1443) (h2 : x - y = 141) : x = 792 :=
by
  sorry

end NUMINAMATH_GPT_greater_num_792_l596_59685


namespace NUMINAMATH_GPT_equal_roots_for_specific_k_l596_59683

theorem equal_roots_for_specific_k (k : ℝ) :
  ((k - 1) * x^2 + 6 * x + 9 = 0) → (6^2 - 4*(k-1)*9 = 0) → (k = 2) :=
by sorry

end NUMINAMATH_GPT_equal_roots_for_specific_k_l596_59683


namespace NUMINAMATH_GPT_find_ax6_by6_l596_59645

variable {a b x y : ℝ}

theorem find_ax6_by6
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 12)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^6 + b * y^6 = 1531.25 :=
sorry

end NUMINAMATH_GPT_find_ax6_by6_l596_59645


namespace NUMINAMATH_GPT_kolya_advantageous_methods_l596_59673

-- Define the context and conditions
variables (n : ℕ) (h₀ : n ≥ 2)
variables (a b : ℕ) (h₁ : a + b = 2*n + 1) (h₂ : a ≥ 2) (h₃ : b ≥ 2)

-- Define outcomes of the methods
def method1_outcome (a b : ℕ) := max a b + min (a - 1) (b - 1)
def method2_outcome (a b : ℕ) := min a b + min (a - 1) (b - 1)
def method3_outcome (a b : ℕ) := max (method1_outcome a b - 1) (method2_outcome a b - 1)

-- Prove which methods are the most and least advantageous
theorem kolya_advantageous_methods :
  method1_outcome a b >= method2_outcome a b ∧ method1_outcome a b >= method3_outcome a b :=
sorry

end NUMINAMATH_GPT_kolya_advantageous_methods_l596_59673


namespace NUMINAMATH_GPT_simplify_140_210_l596_59670

noncomputable def simplify_fraction (num den : Nat) : Nat × Nat :=
  let d := Nat.gcd num den
  (num / d, den / d)

theorem simplify_140_210 :
  simplify_fraction 140 210 = (2, 3) :=
by
  have p140 : 140 = 2^2 * 5 * 7 := by rfl
  have p210 : 210 = 2 * 3 * 5 * 7 := by rfl
  sorry

end NUMINAMATH_GPT_simplify_140_210_l596_59670


namespace NUMINAMATH_GPT_relationship_of_abc_l596_59666

theorem relationship_of_abc (a b c : ℝ) 
  (h1 : b + c = 6 - 4 * a + 3 * a^2) 
  (h2 : c - b = 4 - 4 * a + a^2) : 
  a < b ∧ b ≤ c := 
sorry

end NUMINAMATH_GPT_relationship_of_abc_l596_59666


namespace NUMINAMATH_GPT_option_d_satisfies_equation_l596_59646

theorem option_d_satisfies_equation (x y z : ℤ) (h1 : x = z) (h2 : y = x + 1) : x * (x - y) + y * (y - z) + z * (z - x) = 2 :=
by
  sorry

end NUMINAMATH_GPT_option_d_satisfies_equation_l596_59646


namespace NUMINAMATH_GPT_division_of_fractions_l596_59603

theorem division_of_fractions :
  (5 : ℚ) / 6 / ((2 : ℚ) / 3) = (5 : ℚ) / 4 :=
by
  sorry

end NUMINAMATH_GPT_division_of_fractions_l596_59603


namespace NUMINAMATH_GPT_yan_ratio_l596_59651

variables (w x y : ℝ)

-- Given conditions
def yan_conditions : Prop :=
  w > 0 ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  (y / w = x / w + (x + y) / (7 * w))

-- The ratio of Yan's distance from his home to his distance from the stadium is 3/4
theorem yan_ratio (h : yan_conditions w x y) : 
  x / y = 3 / 4 :=
sorry

end NUMINAMATH_GPT_yan_ratio_l596_59651


namespace NUMINAMATH_GPT_jaden_time_difference_l596_59633

-- Define the conditions as hypotheses
def jaden_time_as_girl (distance : ℕ) (time : ℕ) : Prop :=
  distance = 20 ∧ time = 240

def jaden_time_as_woman (distance : ℕ) (time : ℕ) : Prop :=
  distance = 8 ∧ time = 240

-- Define the proof problem
theorem jaden_time_difference
  (d_girl t_girl d_woman t_woman : ℕ)
  (H_girl : jaden_time_as_girl d_girl t_girl)
  (H_woman : jaden_time_as_woman d_woman t_woman)
  : (t_woman / d_woman) - (t_girl / d_girl) = 18 :=
by
  sorry

end NUMINAMATH_GPT_jaden_time_difference_l596_59633


namespace NUMINAMATH_GPT_range_of_hx_l596_59697

open Real

theorem range_of_hx (h : ℝ → ℝ) (a b : ℝ) (H_def : ∀ x : ℝ, h x = 3 / (1 + 3 * x^4)) 
  (H_range : ∀ y : ℝ, (y > 0 ∧ y ≤ 3) ↔ ∃ x : ℝ, h x = y) : 
  a + b = 3 := 
sorry

end NUMINAMATH_GPT_range_of_hx_l596_59697


namespace NUMINAMATH_GPT_kanul_initial_amount_l596_59665

noncomputable def initial_amount : ℝ :=
  (5000 : ℝ) + 200 + 1200 + (11058.82 : ℝ) * 0.15 + 3000

theorem kanul_initial_amount (X : ℝ) 
  (raw_materials : ℝ := 5000) 
  (machinery : ℝ := 200) 
  (employee_wages : ℝ := 1200) 
  (maintenance_cost : ℝ := 0.15 * X)
  (remaining_balance : ℝ := 3000) 
  (expenses : ℝ := raw_materials + machinery + employee_wages + maintenance_cost) 
  (total_expenses : ℝ := expenses + remaining_balance) :
  X = total_expenses :=
by sorry

end NUMINAMATH_GPT_kanul_initial_amount_l596_59665


namespace NUMINAMATH_GPT_incorrect_judgment_l596_59662

theorem incorrect_judgment : (∀ x : ℝ, x^2 - 1 ≥ -1) ∧ (4 + 2 ≠ 7) :=
by 
  sorry

end NUMINAMATH_GPT_incorrect_judgment_l596_59662


namespace NUMINAMATH_GPT_solve_inequality_l596_59668

theorem solve_inequality (x : ℝ) : ((x + 3) ^ 2 < 1) ↔ (-4 < x ∧ x < -2) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l596_59668


namespace NUMINAMATH_GPT_floor_equiv_l596_59609

theorem floor_equiv {n : ℤ} (h : n > 2) : 
  Int.floor ((n * (n + 1) : ℚ) / (4 * n - 2 : ℚ)) = Int.floor ((n + 1 : ℚ) / 4) := 
sorry

end NUMINAMATH_GPT_floor_equiv_l596_59609


namespace NUMINAMATH_GPT_find_complex_z_l596_59693

theorem find_complex_z (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : z / (1 - 2 * i) = i) :
  z = 2 + i :=
sorry

end NUMINAMATH_GPT_find_complex_z_l596_59693


namespace NUMINAMATH_GPT_symmetry_in_mathematics_l596_59608

-- Define the options
def optionA := "summation of harmonic series from 1 to 100"
def optionB := "general quadratic equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0"
def optionC := "Law of Sines: a / sin A = b / sin B = c / sin C"
def optionD := "arithmetic operation: 123456789 * 9 + 10 = 1111111111"

-- Define the symmetry property
def exhibits_symmetry (option: String) : Prop :=
  option = optionC

-- The theorem to prove
theorem symmetry_in_mathematics : ∃ option, exhibits_symmetry option := by
  use optionC
  sorry

end NUMINAMATH_GPT_symmetry_in_mathematics_l596_59608


namespace NUMINAMATH_GPT_student_passing_percentage_l596_59623

def student_marks : ℕ := 80
def shortfall_marks : ℕ := 100
def total_marks : ℕ := 600

def passing_percentage (student_marks shortfall_marks total_marks : ℕ) : ℕ :=
  (student_marks + shortfall_marks) * 100 / total_marks

theorem student_passing_percentage :
  passing_percentage student_marks shortfall_marks total_marks = 30 :=
by
  sorry

end NUMINAMATH_GPT_student_passing_percentage_l596_59623


namespace NUMINAMATH_GPT_intersection_points_l596_59664

theorem intersection_points (x y : ℝ) (h1 : x^2 - 4 * y^2 = 4) (h2 : x = 3 * y) : 
  (x, y) = (3, 1) ∨ (x, y) = (-3, -1) :=
sorry

end NUMINAMATH_GPT_intersection_points_l596_59664


namespace NUMINAMATH_GPT_toothpicks_stage_20_l596_59600

-- Definition of the toothpick sequence
def toothpicks (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 3 + 3 * (n - 1)

-- Theorem statement
theorem toothpicks_stage_20 : toothpicks 20 = 60 := by
  sorry

end NUMINAMATH_GPT_toothpicks_stage_20_l596_59600


namespace NUMINAMATH_GPT_shirt_cost_l596_59652

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 66) : S = 12 :=
by
  sorry

end NUMINAMATH_GPT_shirt_cost_l596_59652


namespace NUMINAMATH_GPT_minimize_expression_at_c_l596_59615

theorem minimize_expression_at_c (c : ℝ) : (c = 7 / 4) → (∀ x : ℝ, 2 * c^2 - 7 * c + 4 ≤ 2 * x^2 - 7 * x + 4) :=
sorry

end NUMINAMATH_GPT_minimize_expression_at_c_l596_59615


namespace NUMINAMATH_GPT_abs_neg_2_plus_sqrt3_add_tan60_eq_2_l596_59617

theorem abs_neg_2_plus_sqrt3_add_tan60_eq_2 :
  abs (-2 + Real.sqrt 3) + Real.tan (Real.pi / 3) = 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_2_plus_sqrt3_add_tan60_eq_2_l596_59617


namespace NUMINAMATH_GPT_minimum_distance_from_midpoint_to_y_axis_l596_59632

theorem minimum_distance_from_midpoint_to_y_axis (M N : ℝ × ℝ) (P : ℝ × ℝ)
  (hM : M.snd ^ 2 = M.fst) (hN : N.snd ^ 2 = N.fst)
  (hlength : (M.fst - N.fst)^2 + (M.snd - N.snd)^2 = 16)
  (hP : P = ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)) :
  abs P.fst = 7 / 4 :=
sorry

end NUMINAMATH_GPT_minimum_distance_from_midpoint_to_y_axis_l596_59632


namespace NUMINAMATH_GPT_sales_tax_difference_l596_59628

noncomputable def price_before_tax : ℝ := 50
noncomputable def sales_tax_rate_7_5_percent : ℝ := 0.075
noncomputable def sales_tax_rate_8_percent : ℝ := 0.08

theorem sales_tax_difference :
  (price_before_tax * sales_tax_rate_8_percent) - (price_before_tax * sales_tax_rate_7_5_percent) = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l596_59628


namespace NUMINAMATH_GPT_customers_stayed_behind_l596_59649

theorem customers_stayed_behind : ∃ x : ℕ, (x + (x + 5) = 11) ∧ x = 3 := by
  sorry

end NUMINAMATH_GPT_customers_stayed_behind_l596_59649


namespace NUMINAMATH_GPT_remainder_when_multiplied_and_divided_l596_59606

theorem remainder_when_multiplied_and_divided (n k : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := 
by
  sorry

end NUMINAMATH_GPT_remainder_when_multiplied_and_divided_l596_59606


namespace NUMINAMATH_GPT_find_x_angle_l596_59656

theorem find_x_angle (x : ℝ) (h : x + x + 140 = 360) : x = 110 :=
by
  sorry

end NUMINAMATH_GPT_find_x_angle_l596_59656


namespace NUMINAMATH_GPT_usual_time_to_school_l596_59692

theorem usual_time_to_school (R T : ℝ) (h : (R * T = (6/5) * R * (T - 4))) : T = 24 :=
by 
  sorry

end NUMINAMATH_GPT_usual_time_to_school_l596_59692


namespace NUMINAMATH_GPT_correct_answer_l596_59630

-- Define the sentence structure and the requirement for a formal object
structure SentenceStructure where
  subject : String := "I"
  verb : String := "like"
  object_placeholder : String := "_"
  clause : String := "when the weather is clear and bright"

-- Correct choices provided
inductive Choice
  | this
  | that
  | it
  | one

-- Problem formulation: Based on SentenceStructure, prove that 'it' is the correct choice
theorem correct_answer {S : SentenceStructure} : Choice.it = Choice.it :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_correct_answer_l596_59630


namespace NUMINAMATH_GPT_trains_crossing_time_l596_59621

theorem trains_crossing_time
  (L speed1 speed2 : ℝ)
  (time_same_direction time_opposite_direction : ℝ) 
  (h1 : speed1 = 60)
  (h2 : speed2 = 40)
  (h3 : time_same_direction = 40)
  (h4 : 2 * L = (speed1 - speed2) * 5/18 * time_same_direction) :
  time_opposite_direction = 8 := 
sorry

end NUMINAMATH_GPT_trains_crossing_time_l596_59621


namespace NUMINAMATH_GPT_find_solutions_l596_59690

-- Defining the system of equations as conditions
def cond1 (a b : ℕ) := a * b + 2 * a - b = 58
def cond2 (b c : ℕ) := b * c + 4 * b + 2 * c = 300
def cond3 (c d : ℕ) := c * d - 6 * c + 4 * d = 101

-- Theorem to prove the solutions satisfy the system of equations
theorem find_solutions (a b c d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0):
  cond1 a b ∧ cond2 b c ∧ cond3 c d ↔ (a, b, c, d) ∈ [(3, 26, 7, 13), (15, 2, 73, 7)] :=
by sorry

end NUMINAMATH_GPT_find_solutions_l596_59690


namespace NUMINAMATH_GPT_double_recipe_total_l596_59671

theorem double_recipe_total 
  (butter_ratio : ℕ) (flour_ratio : ℕ) (sugar_ratio : ℕ) 
  (flour_cups : ℕ) 
  (h_ratio : butter_ratio = 2) 
  (h_flour : flour_ratio = 5) 
  (h_sugar : sugar_ratio = 3) 
  (h_flour_cups : flour_cups = 15) : 
  2 * ((butter_ratio * (flour_cups / flour_ratio)) + flour_cups + (sugar_ratio * (flour_cups / flour_ratio))) = 60 := 
by 
  sorry

end NUMINAMATH_GPT_double_recipe_total_l596_59671


namespace NUMINAMATH_GPT_total_amount_owed_l596_59654

theorem total_amount_owed :
  ∃ (P remaining_balance processing_fee new_total discount: ℝ),
    0.05 * P = 50 ∧
    remaining_balance = P - 50 ∧
    processing_fee = 0.03 * remaining_balance ∧
    new_total = remaining_balance + processing_fee ∧
    discount = 0.10 * new_total ∧
    new_total - discount = 880.65 :=
sorry

end NUMINAMATH_GPT_total_amount_owed_l596_59654


namespace NUMINAMATH_GPT_calc_expression_l596_59675

theorem calc_expression : 3 ^ 2022 * (1 / 3) ^ 2023 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l596_59675


namespace NUMINAMATH_GPT_train_crossing_time_l596_59636

def train_length : ℝ := 150
def train_speed : ℝ := 179.99999999999997

theorem train_crossing_time : train_length / train_speed = 0.8333333333333333 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l596_59636


namespace NUMINAMATH_GPT_third_snail_time_l596_59689

theorem third_snail_time
  (speed_first_snail : ℝ)
  (speed_second_snail : ℝ)
  (speed_third_snail : ℝ)
  (time_first_snail : ℝ)
  (distance : ℝ) :
  (speed_first_snail = 2) →
  (speed_second_snail = 2 * speed_first_snail) →
  (speed_third_snail = 5 * speed_second_snail) →
  (time_first_snail = 20) →
  (distance = speed_first_snail * time_first_snail) →
  (distance / speed_third_snail = 2) :=
by
  sorry

end NUMINAMATH_GPT_third_snail_time_l596_59689


namespace NUMINAMATH_GPT_max_ab_l596_59612

theorem max_ab (a b c : ℝ) (h1 : 3 * a + b = 1) (h2 : 0 ≤ a) (h3 : a < 1) (h4 : 0 ≤ b) 
(h5 : b < 1) (h6 : 0 ≤ c) (h7 : c < 1) (h8 : a + b + c = 1) : 
  ab ≤ 1 / 12 := by
  sorry

end NUMINAMATH_GPT_max_ab_l596_59612


namespace NUMINAMATH_GPT_int_solutions_l596_59610

theorem int_solutions (a b : ℤ) (h : a^2 + b = b^2022) : (a, b) = (0, 0) ∨ (a, b) = (0, 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_int_solutions_l596_59610


namespace NUMINAMATH_GPT_seven_does_not_always_divide_l596_59686

theorem seven_does_not_always_divide (n : ℤ) :
  ¬(7 ∣ (n ^ 2225 - n ^ 2005)) :=
by sorry

end NUMINAMATH_GPT_seven_does_not_always_divide_l596_59686


namespace NUMINAMATH_GPT_angle_is_40_l596_59681

theorem angle_is_40 (x : ℝ) 
  : (180 - x = 2 * (90 - x) + 40) → x = 40 :=
by
  sorry

end NUMINAMATH_GPT_angle_is_40_l596_59681


namespace NUMINAMATH_GPT_smallest_M_value_l596_59655

theorem smallest_M_value 
  (a b c d e : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) 
  (h_sum : a + b + c + d + e = 2010) : 
  (∃ M, M = max (a+b) (max (b+c) (max (c+d) (d+e))) ∧ M = 671) :=
by
  sorry

end NUMINAMATH_GPT_smallest_M_value_l596_59655


namespace NUMINAMATH_GPT_no_solutions_then_a_eq_zero_l596_59657

theorem no_solutions_then_a_eq_zero (a b : ℝ) :
  (∀ x y : ℝ, ¬ (y^2 = x^2 + a * x + b ∧ x^2 = y^2 + a * y + b)) → a = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_then_a_eq_zero_l596_59657


namespace NUMINAMATH_GPT_comb_10_3_eq_120_l596_59699

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_GPT_comb_10_3_eq_120_l596_59699


namespace NUMINAMATH_GPT_efficiency_and_days_l596_59642

noncomputable def sakshi_efficiency : ℝ := 1 / 25
noncomputable def tanya_efficiency : ℝ := 1.25 * sakshi_efficiency
noncomputable def ravi_efficiency : ℝ := 0.70 * sakshi_efficiency
noncomputable def combined_efficiency : ℝ := sakshi_efficiency + tanya_efficiency + ravi_efficiency
noncomputable def days_to_complete_work : ℝ := 1 / combined_efficiency

theorem efficiency_and_days:
  combined_efficiency = 29.5 / 250 ∧
  days_to_complete_work = 250 / 29.5 :=
by
  sorry

end NUMINAMATH_GPT_efficiency_and_days_l596_59642


namespace NUMINAMATH_GPT_a_mul_b_value_l596_59625

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end NUMINAMATH_GPT_a_mul_b_value_l596_59625
