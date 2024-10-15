import Mathlib

namespace NUMINAMATH_GPT_trajectory_equation_l1607_160766

-- Define the fixed points F1 and F2
structure Point where
  x : ℝ
  y : ℝ

def F1 : Point := ⟨-2, 0⟩
def F2 : Point := ⟨2, 0⟩

-- Define the moving point M and the condition it must satisfy
def satisfies_condition (M : Point) : Prop :=
  (Real.sqrt ((M.x + 2)^2 + M.y^2) - Real.sqrt ((M.x - 2)^2 + M.y^2)) = 4

-- The trajectory of the point M must satisfy y = 0 and x >= 2
def on_trajectory (M : Point) : Prop :=
  M.y = 0 ∧ M.x ≥ 2

-- The final theorem to be proved
theorem trajectory_equation (M : Point) (h : satisfies_condition M) : on_trajectory M := by
  sorry

end NUMINAMATH_GPT_trajectory_equation_l1607_160766


namespace NUMINAMATH_GPT_panthers_score_l1607_160757

-- Definitions as per the conditions
def total_points (C P : ℕ) : Prop := C + P = 48
def margin (C P : ℕ) : Prop := C = P + 20

-- Theorem statement proving Panthers score 14 points
theorem panthers_score (C P : ℕ) (h1 : total_points C P) (h2 : margin C P) : P = 14 :=
sorry

end NUMINAMATH_GPT_panthers_score_l1607_160757


namespace NUMINAMATH_GPT_find_m_l1607_160788

open Real

noncomputable def vec_a : ℝ × ℝ := (-1, 2)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ := (m, 3)

theorem find_m (m : ℝ) (h : -1 * m + 2 * 3 = 0) : m = 6 :=
sorry

end NUMINAMATH_GPT_find_m_l1607_160788


namespace NUMINAMATH_GPT_meryll_questions_l1607_160765

/--
Meryll wants to write a total of 35 multiple-choice questions and 15 problem-solving questions. 
She has written \(\frac{2}{5}\) of the multiple-choice questions and \(\frac{1}{3}\) of the problem-solving questions.
We need to prove that she needs to write 31 more questions in total.
-/
theorem meryll_questions : (35 - (2 / 5) * 35) + (15 - (1 / 3) * 15) = 31 := by
  sorry

end NUMINAMATH_GPT_meryll_questions_l1607_160765


namespace NUMINAMATH_GPT_probability_bob_wins_l1607_160749

theorem probability_bob_wins (P_lose : ℝ) (P_tie : ℝ) (h1 : P_lose = 5/8) (h2 : P_tie = 1/8) :
  (1 - P_lose - P_tie) = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_probability_bob_wins_l1607_160749


namespace NUMINAMATH_GPT_implication_a_lt_b_implies_a_lt_b_plus_1_l1607_160701

theorem implication_a_lt_b_implies_a_lt_b_plus_1 (a b : ℝ) (h : a < b) : a < b + 1 := by
  sorry

end NUMINAMATH_GPT_implication_a_lt_b_implies_a_lt_b_plus_1_l1607_160701


namespace NUMINAMATH_GPT_largest_number_2013_l1607_160721

theorem largest_number_2013 (x y : ℕ) (h1 : x + y = 2013)
    (h2 : y = 5 * (x / 100 + 1)) : max x y = 1913 := by
  sorry

end NUMINAMATH_GPT_largest_number_2013_l1607_160721


namespace NUMINAMATH_GPT_total_chapters_eq_l1607_160727

-- Definitions based on conditions
def days : ℕ := 664
def chapters_per_day : ℕ := 332

-- Theorem to prove the total number of chapters in the book is 220448
theorem total_chapters_eq : (chapters_per_day * days = 220448) :=
by
  sorry

end NUMINAMATH_GPT_total_chapters_eq_l1607_160727


namespace NUMINAMATH_GPT_total_money_taken_l1607_160738

def individual_bookings : ℝ := 12000
def group_bookings : ℝ := 16000
def returned_due_to_cancellations : ℝ := 1600

def total_taken (individual_bookings : ℝ) (group_bookings : ℝ) (returned_due_to_cancellations : ℝ) : ℝ :=
  (individual_bookings + group_bookings) - returned_due_to_cancellations

theorem total_money_taken :
  total_taken individual_bookings group_bookings returned_due_to_cancellations = 26400 := by
  sorry

end NUMINAMATH_GPT_total_money_taken_l1607_160738


namespace NUMINAMATH_GPT_cost_of_saddle_l1607_160779

theorem cost_of_saddle (S : ℝ) (H : 4 * S + S = 5000) : S = 1000 :=
by sorry

end NUMINAMATH_GPT_cost_of_saddle_l1607_160779


namespace NUMINAMATH_GPT_average_student_headcount_l1607_160751

theorem average_student_headcount :
  let count_0304 := 10500
  let count_0405 := 10700
  let count_0506 := 11300
  let total_count := count_0304 + count_0405 + count_0506
  let number_of_terms := 3
  let average := total_count / number_of_terms
  average = 10833 :=
by
  sorry

end NUMINAMATH_GPT_average_student_headcount_l1607_160751


namespace NUMINAMATH_GPT_solve_for_xy_l1607_160776

theorem solve_for_xy (x y : ℕ) : 
  (4^x / 2^(x + y) = 16) ∧ (9^(x + y) / 3^(5 * y) = 81) → x * y = 32 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_xy_l1607_160776


namespace NUMINAMATH_GPT_find_a_f_odd_f_increasing_l1607_160707

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a / x

theorem find_a : (f 1 a = 3) → (a = -1) :=
by
  sorry

noncomputable def f_1 (x : ℝ) : ℝ := 2 * x + 1 / x

theorem f_odd : ∀ x : ℝ, f_1 (-x) = -f_1 x :=
by
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, (x1 > 1) → (x2 > 1) → (x1 > x2) → (f_1 x1 > f_1 x2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_f_odd_f_increasing_l1607_160707


namespace NUMINAMATH_GPT_car_storm_distance_30_l1607_160717

noncomputable def car_position (t : ℝ) : ℝ × ℝ :=
  (0, 3/4 * t)

noncomputable def storm_center (t : ℝ) : ℝ × ℝ :=
  (150 - (3/4 / Real.sqrt 2) * t, -(3/4 / Real.sqrt 2) * t)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem car_storm_distance_30 :
  ∃ (t : ℝ), distance (car_position t) (storm_center t) = 30 :=
sorry

end NUMINAMATH_GPT_car_storm_distance_30_l1607_160717


namespace NUMINAMATH_GPT_max_segment_perimeter_l1607_160745

def isosceles_triangle (base height : ℝ) := true -- A realistic definition can define properties of an isosceles triangle

def equal_area_segments (triangle : isosceles_triangle 10 12) (n : ℕ) := true -- A realist definition can define cutting into equal area segments

noncomputable def perimeter_segment (base height : ℝ) (k : ℕ) (n : ℕ) : ℝ :=
  1 + Real.sqrt (height^2 + (base / n * k)^2) + Real.sqrt (height^2 + (base / n * (k + 1))^2)

theorem max_segment_perimeter (base height : ℝ) (n : ℕ) (h_base : base = 10) (h_height : height = 12) (h_segments : n = 10) :
  ∃ k, k ∈ Finset.range n ∧ perimeter_segment base height k n = 31.62 :=
by
  sorry

end NUMINAMATH_GPT_max_segment_perimeter_l1607_160745


namespace NUMINAMATH_GPT_ellipse_focus_distance_l1607_160790

theorem ellipse_focus_distance : ∀ (x y : ℝ), 9 * x^2 + y^2 = 900 → 2 * Real.sqrt (10^2 - 30^2) = 40 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_ellipse_focus_distance_l1607_160790


namespace NUMINAMATH_GPT_rectangle_perimeter_l1607_160795

-- Definitions based on conditions
def length (w : ℝ) : ℝ := 2 * w
def width (w : ℝ) : ℝ := w
def area (w : ℝ) : ℝ := length w * width w
def perimeter (w : ℝ) : ℝ := 2 * (length w + width w)

-- Problem statement: Prove that the perimeter is 120 cm given area is 800 cm² and length is twice the width
theorem rectangle_perimeter (w : ℝ) (h : area w = 800) : perimeter w = 120 := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1607_160795


namespace NUMINAMATH_GPT_beautiful_39th_moment_l1607_160783

def is_beautiful (h : ℕ) (mm : ℕ) : Prop :=
  (h + mm) % 12 = 0

def start_time := (7, 49)

noncomputable def find_39th_beautiful_moment : ℕ × ℕ :=
  (15, 45)

theorem beautiful_39th_moment :
  find_39th_beautiful_moment = (15, 45) :=
by
  sorry

end NUMINAMATH_GPT_beautiful_39th_moment_l1607_160783


namespace NUMINAMATH_GPT_find_number_l1607_160784

theorem find_number 
  (a b c d : ℤ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 6 * a + 9 * b + 3 * c + d = 88)
  (h6 : a - b + c - d = -6)
  (h7 : a - 9 * b + 3 * c - d = -46) : 
  1000 * a + 100 * b + 10 * c + d = 6507 := 
sorry

end NUMINAMATH_GPT_find_number_l1607_160784


namespace NUMINAMATH_GPT_fewest_fence_posts_l1607_160741

def fence_posts (length_wide short_side long_side : ℕ) (post_interval : ℕ) : ℕ :=
  let wide_side_posts := (long_side / post_interval) + 1
  let short_side_posts := (short_side / post_interval)
  wide_side_posts + 2 * short_side_posts

theorem fewest_fence_posts : fence_posts 40 10 100 10 = 19 :=
  by
    -- The proof will be completed here
    sorry

end NUMINAMATH_GPT_fewest_fence_posts_l1607_160741


namespace NUMINAMATH_GPT_Pat_height_l1607_160755

noncomputable def Pat_first_day_depth := 40 -- in cm
noncomputable def Mat_second_day_depth := 3 * Pat_first_day_depth -- Mat digs 3 times the depth on the second day
noncomputable def Pat_third_day_depth := Mat_second_day_depth - Pat_first_day_depth -- Pat digs the same amount on the third day
noncomputable def Total_depth_after_third_day := Mat_second_day_depth + Pat_third_day_depth -- Total depth after third day's digging
noncomputable def Depth_above_Pat_head := 50 -- The depth above Pat's head

theorem Pat_height : Total_depth_after_third_day - Depth_above_Pat_head = 150 := by
  sorry

end NUMINAMATH_GPT_Pat_height_l1607_160755


namespace NUMINAMATH_GPT_diminished_value_is_seven_l1607_160758

theorem diminished_value_is_seven (x y : ℕ) (hx : x = 280)
  (h_eq : x / 5 + 7 = x / 4 - y) : y = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_diminished_value_is_seven_l1607_160758


namespace NUMINAMATH_GPT_marbles_lost_correct_l1607_160769

-- Define the initial number of marbles
def initial_marbles : ℕ := 16

-- Define the current number of marbles
def current_marbles : ℕ := 9

-- Define the number of marbles lost
def marbles_lost (initial current : ℕ) : ℕ := initial - current

-- State the proof problem: Given the conditions, prove the number of marbles lost is 7
theorem marbles_lost_correct : marbles_lost initial_marbles current_marbles = 7 := by
  sorry

end NUMINAMATH_GPT_marbles_lost_correct_l1607_160769


namespace NUMINAMATH_GPT_option_d_can_form_triangle_l1607_160731

noncomputable def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem option_d_can_form_triangle : satisfies_triangle_inequality 2 3 4 :=
by {
  -- Using the triangle inequality theorem to check
  sorry
}

end NUMINAMATH_GPT_option_d_can_form_triangle_l1607_160731


namespace NUMINAMATH_GPT_find_oranges_l1607_160743

def A : ℕ := 3
def B : ℕ := 1

theorem find_oranges (O : ℕ) : A + B + O + (A + 4) + 10 * B + 2 * (A + 4) = 39 → O = 4 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_find_oranges_l1607_160743


namespace NUMINAMATH_GPT_negation_of_proposition_l1607_160753

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1)) ↔ (∃ x : ℝ, x ≥ 1 ∧ x^2 < 1) := 
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1607_160753


namespace NUMINAMATH_GPT_final_velocity_l1607_160740

variable (u a t : ℝ)

-- Defining the conditions
def initial_velocity := u = 0
def acceleration := a = 1.2
def time := t = 15

-- Statement of the theorem
theorem final_velocity : initial_velocity u ∧ acceleration a ∧ time t → (u + a * t = 18) := by
  sorry

end NUMINAMATH_GPT_final_velocity_l1607_160740


namespace NUMINAMATH_GPT_distinctPaintedCubeConfigCount_l1607_160789

-- Define a painted cube with given face colors
structure PaintedCube where
  blue_face : ℤ
  yellow_faces : Finset ℤ
  red_faces : Finset ℤ
  -- Ensure logical conditions about faces
  face_count : blue_face ∉ yellow_faces ∧ blue_face ∉ red_faces ∧
               yellow_faces ∩ red_faces = ∅ ∧ yellow_faces.card = 2 ∧
               red_faces.card = 3

-- There are no orientation-invariant rotations that change the configuration
def equivPaintedCube (c1 c2 : PaintedCube) : Prop :=
  ∃ (r: ℤ), 
    -- rotate c1 by r to get c2
    true -- placeholder for rotation logic

-- The set of all possible distinct painted cubes under rotation constraints is defined
def possibleConfigurations : Finset PaintedCube :=
  sorry  -- construct this set considering rotations

-- The main proposition
theorem distinctPaintedCubeConfigCount : (possibleConfigurations.card = 4) :=
  sorry

end NUMINAMATH_GPT_distinctPaintedCubeConfigCount_l1607_160789


namespace NUMINAMATH_GPT_shortest_distance_to_line_l1607_160708

open Classical

variables {P A B C : Type} [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (PA PB PC : ℝ)
variables (l : ℕ) -- l represents the line

-- Given conditions
def PA_dist : ℝ := 4
def PB_dist : ℝ := 5
def PC_dist : ℝ := 2

theorem shortest_distance_to_line (hPA : PA = PA_dist) (hPB : PB = PB_dist) (hPC : PC = PC_dist) :
  ∃ d, d ≤ 2 := 
sorry

end NUMINAMATH_GPT_shortest_distance_to_line_l1607_160708


namespace NUMINAMATH_GPT_ceilings_left_to_paint_l1607_160728

theorem ceilings_left_to_paint
    (floors : ℕ)
    (rooms_per_floor : ℕ)
    (ceilings_painted_this_week : ℕ)
    (hallways_per_floor : ℕ)
    (hallway_ceilings_per_hallway : ℕ)
    (ceilings_painted_ratio : ℚ)
    : floors = 4
    → rooms_per_floor = 7
    → ceilings_painted_this_week = 12
    → hallways_per_floor = 1
    → hallway_ceilings_per_hallway = 1
    → ceilings_painted_ratio = 1 / 4
    → (floors * rooms_per_floor + floors * hallways_per_floor * hallway_ceilings_per_hallway 
        - ceilings_painted_this_week 
        - (ceilings_painted_ratio * ceilings_painted_this_week + floors * hallway_ceilings_per_hallway) = 13) :=
by
  intros
  sorry

end NUMINAMATH_GPT_ceilings_left_to_paint_l1607_160728


namespace NUMINAMATH_GPT_triangle_perimeters_sum_l1607_160797

theorem triangle_perimeters_sum :
  ∃ (t : ℕ),
    (∀ (A B C D : Type) (x y : ℕ), 
      (AB = 7 ∧ BC = 17 ∧ AD = x ∧ CD = x ∧ BD = y ∧ x^2 - y^2 = 240) →
      t = 114) :=
sorry

end NUMINAMATH_GPT_triangle_perimeters_sum_l1607_160797


namespace NUMINAMATH_GPT_min_val_of_f_l1607_160791

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

-- Theorem stating the minimum value of f(x) for x > 0 is 5.5
theorem min_val_of_f : ∀ x : ℝ, x > 0 → f x ≥ 5.5 :=
by sorry

end NUMINAMATH_GPT_min_val_of_f_l1607_160791


namespace NUMINAMATH_GPT_geometric_sequence_smallest_n_l1607_160702

def geom_seq (n : ℕ) (r : ℝ) (b₁ : ℝ) : ℝ := 
  b₁ * r^(n-1)

theorem geometric_sequence_smallest_n 
  (b₁ b₂ b₃ : ℝ) (r : ℝ)
  (h₁ : b₁ = 2)
  (h₂ : b₂ = 6)
  (h₃ : b₃ = 18)
  (h_seq : ∀ n, bₙ = geom_seq n r b₁) :
  ∃ n, n = 5 ∧ geom_seq n r 2 = 324 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_smallest_n_l1607_160702


namespace NUMINAMATH_GPT_quadratic_has_single_solution_l1607_160720

theorem quadratic_has_single_solution (q : ℚ) (h : q ≠ 0) :
  (∀ x : ℚ, q * x^2 - 16 * x + 9 = 0 → q = 64 / 9) := by
  sorry

end NUMINAMATH_GPT_quadratic_has_single_solution_l1607_160720


namespace NUMINAMATH_GPT_sue_receives_correct_answer_l1607_160706

theorem sue_receives_correct_answer (x : ℕ) (y : ℕ) (z : ℕ) (h1 : y = 3 * (x + 2)) (h2 : z = 3 * (y - 2)) (hx : x = 6) : z = 66 :=
by
  sorry

end NUMINAMATH_GPT_sue_receives_correct_answer_l1607_160706


namespace NUMINAMATH_GPT_total_students_in_Lansing_l1607_160768

def n_schools : Nat := 25
def students_per_school : Nat := 247
def total_students : Nat := n_schools * students_per_school

theorem total_students_in_Lansing :
  total_students = 6175 :=
  by
    -- we can either compute manually or just put sorry for automated assistance
    sorry

end NUMINAMATH_GPT_total_students_in_Lansing_l1607_160768


namespace NUMINAMATH_GPT_function_monotonic_decreasing_interval_l1607_160730

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6)

theorem function_monotonic_decreasing_interval :
  ∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  ∀ y ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  (x ≤ y → f y ≤ f x) :=
by
  sorry

end NUMINAMATH_GPT_function_monotonic_decreasing_interval_l1607_160730


namespace NUMINAMATH_GPT_customer_purchases_90_percent_l1607_160734

variable (P Q : ℝ) 

theorem customer_purchases_90_percent (price_increase_expenditure_diff : 
  (1.25 * P * R / 100 * Q = 1.125 * P * Q)) : 
  R = 90 := 
by 
  sorry

end NUMINAMATH_GPT_customer_purchases_90_percent_l1607_160734


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l1607_160752

-- Given conditions and definitions
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def equation_has_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c ≥ 0

-- Problem translated into a Lean statement
theorem quadratic_real_roots_range (m : ℝ) :
  equation_has_real_roots 1 (-2) (-m) ↔ m ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l1607_160752


namespace NUMINAMATH_GPT_total_pages_in_book_l1607_160750

-- Define the given conditions
def chapters : Nat := 41
def days : Nat := 30
def pages_per_day : Nat := 15

-- Define the statement to be proven
theorem total_pages_in_book : (days * pages_per_day) = 450 := by
  sorry

end NUMINAMATH_GPT_total_pages_in_book_l1607_160750


namespace NUMINAMATH_GPT_gasoline_price_increase_l1607_160781

theorem gasoline_price_increase 
  (highest_price : ℝ) (lowest_price : ℝ) 
  (h_high : highest_price = 17) 
  (h_low : lowest_price = 10) : 
  (highest_price - lowest_price) / lowest_price * 100 = 70 := 
by
  /- proof can go here -/
  sorry

end NUMINAMATH_GPT_gasoline_price_increase_l1607_160781


namespace NUMINAMATH_GPT_great_circle_bisects_angle_l1607_160796

noncomputable def north_pole : Point := sorry
noncomputable def equator_point (C : Point) : Prop := sorry
noncomputable def great_circle_through (P Q : Point) : Circle := sorry
noncomputable def equidistant_from_N (A B N : Point) : Prop := sorry
noncomputable def spherical_triangle (A B C : Point) : Triangle := sorry
noncomputable def bisects_angle (C N A B : Point) : Prop := sorry

theorem great_circle_bisects_angle
  (N A B C: Point)
  (hN: N = north_pole)
  (hA: equidistant_from_N A B N)
  (hC: equator_point C)
  (hTriangle: spherical_triangle A B C)
  : bisects_angle C N A B :=
sorry

end NUMINAMATH_GPT_great_circle_bisects_angle_l1607_160796


namespace NUMINAMATH_GPT_simplify_polynomial_l1607_160782

variable (x : ℝ)

theorem simplify_polynomial :
  (2 * x^10 + 8 * x^9 + 3 * x^8) + (5 * x^12 - x^10 + 2 * x^9 - 5 * x^8 + 4 * x^5 + 6)
  = 5 * x^12 + x^10 + 10 * x^9 - 2 * x^8 + 4 * x^5 + 6 := by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1607_160782


namespace NUMINAMATH_GPT_disprove_prime_statement_l1607_160700

theorem disprove_prime_statement : ∃ n : ℕ, ((¬ Nat.Prime n) ∧ Nat.Prime (n + 2)) ∨ (Nat.Prime n ∧ ¬ Nat.Prime (n + 2)) :=
sorry

end NUMINAMATH_GPT_disprove_prime_statement_l1607_160700


namespace NUMINAMATH_GPT_outer_boundary_diameter_l1607_160762

def width_jogging_path : ℝ := 4
def width_garden_ring : ℝ := 10
def diameter_pond : ℝ := 12

theorem outer_boundary_diameter : 2 * (diameter_pond / 2 + width_garden_ring + width_jogging_path) = 40 := by
  sorry

end NUMINAMATH_GPT_outer_boundary_diameter_l1607_160762


namespace NUMINAMATH_GPT_b3_b7_equals_16_l1607_160712

variable {a b : ℕ → ℝ}
variable {d : ℝ}

-- Conditions: a is an arithmetic sequence with common difference d
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: b is a geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

-- Given condition on the arithmetic sequence a
def condition_on_a (a : ℕ → ℝ) (d : ℝ) : Prop :=
  2 * a 2 - (a 5) ^ 2 + 2 * a 8 = 0

-- Define the specific arithmetic sequence in terms of d and a5
noncomputable def a_seq (a5 d : ℝ) : ℕ → ℝ
| 0 => a5 - 5 * d
| 1 => a5 - 4 * d
| 2 => a5 - 3 * d
| 3 => a5 - 2 * d
| 4 => a5 - d
| 5 => a5
| 6 => a5 + d
| 7 => a5 + 2 * d
| 8 => a5 + 3 * d
| 9 => a5 + 4 * d
| n => 0 -- extending for unspecified

-- Condition: b_5 = a_5
def b_equals_a (a b : ℕ → ℝ) : Prop :=
  b 5 = a 5

-- Theorem: Given the conditions, prove b_3 * b_7 = 16
theorem b3_b7_equals_16 (a b : ℕ → ℝ) (d : ℝ)
  (ha_seq : is_arithmetic_sequence a d)
  (hb_seq : is_geometric_sequence b)
  (h_cond_a : condition_on_a a d)
  (h_b_equals_a : b_equals_a a b) : b 3 * b 7 = 16 :=
by
  sorry

end NUMINAMATH_GPT_b3_b7_equals_16_l1607_160712


namespace NUMINAMATH_GPT_turtles_remaining_proof_l1607_160724

noncomputable def turtles_original := 50
noncomputable def turtles_additional := 7 * turtles_original - 6
noncomputable def turtles_total_before_frightened := turtles_original + turtles_additional
noncomputable def turtles_frightened := (3 / 7) * turtles_total_before_frightened
noncomputable def turtles_remaining := turtles_total_before_frightened - turtles_frightened

theorem turtles_remaining_proof : turtles_remaining = 226 := by
  sorry

end NUMINAMATH_GPT_turtles_remaining_proof_l1607_160724


namespace NUMINAMATH_GPT_segment_length_of_points_A_l1607_160763

-- Define the basic setup
variable (d BA CA : ℝ)
variable {A B C : Point} -- Assume we have a type Point for the geometric points

-- Establish some conditions: A right triangle with given lengths
def is_right_triangle (A B C : Point) : Prop := sorry -- Placeholder for definition

def distance (P Q : Point) : ℝ := sorry -- Placeholder for the distance function

-- Conditions
variables (h_right_triangle : is_right_triangle A B C)
variables (h_hypotenuse : distance B C = d)
variables (h_smallest_leg : min (distance B A) (distance C A) = min BA CA)

-- The theorem statement
theorem segment_length_of_points_A (h_right_triangle : is_right_triangle A B C)
                                    (h_hypotenuse : distance B C = d)
                                    (h_smallest_leg : min (distance B A) (distance C A) = min BA CA) :
  ∃ A, (∀ t : ℝ, distance O A = d - min BA CA) :=
sorry -- Proof to be provided

end NUMINAMATH_GPT_segment_length_of_points_A_l1607_160763


namespace NUMINAMATH_GPT_monotonic_f_on_interval_l1607_160759

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 10) - 2

theorem monotonic_f_on_interval : 
  ∀ x y : ℝ, 
    x ∈ Set.Icc (Real.pi / 2) (7 * Real.pi / 5) → 
    y ∈ Set.Icc (Real.pi / 2) (7 * Real.pi / 5) → 
    x ≤ y → 
    f x ≤ f y :=
sorry

end NUMINAMATH_GPT_monotonic_f_on_interval_l1607_160759


namespace NUMINAMATH_GPT_largest_possible_cylindrical_tank_radius_in_crate_l1607_160764

theorem largest_possible_cylindrical_tank_radius_in_crate
  (crate_length : ℝ) (crate_width : ℝ) (crate_height : ℝ)
  (cylinder_height : ℝ) (cylinder_radius : ℝ)
  (h_cube : crate_length = 20 ∧ crate_width = 20 ∧ crate_height = 20)
  (h_cylinder_in_cube : cylinder_height = 20 ∧ 2 * cylinder_radius ≤ 20) :
  cylinder_radius = 10 :=
sorry

end NUMINAMATH_GPT_largest_possible_cylindrical_tank_radius_in_crate_l1607_160764


namespace NUMINAMATH_GPT_matrix_to_system_solution_l1607_160715

theorem matrix_to_system_solution :
  ∀ (x y : ℝ),
  (2 * x + y = 5) ∧ (x - 2 * y = 0) →
  3 * x - y = 5 :=
by
  sorry

end NUMINAMATH_GPT_matrix_to_system_solution_l1607_160715


namespace NUMINAMATH_GPT_total_percent_decrease_l1607_160760

theorem total_percent_decrease (initial_value : ℝ) (val1 val2 : ℝ) :
  initial_value > 0 →
  val1 = initial_value * (1 - 0.60) →
  val2 = val1 * (1 - 0.10) →
  (initial_value - val2) / initial_value * 100 = 64 :=
by
  intros h_initial h_val1 h_val2
  sorry

end NUMINAMATH_GPT_total_percent_decrease_l1607_160760


namespace NUMINAMATH_GPT_steve_travel_time_l1607_160798

theorem steve_travel_time :
  ∀ (d : ℕ) (v_back : ℕ) (v_to : ℕ),
  d = 20 →
  v_back = 10 →
  v_to = v_back / 2 →
  d / v_to + d / v_back = 6 := 
by
  intros d v_back v_to h1 h2 h3
  sorry

end NUMINAMATH_GPT_steve_travel_time_l1607_160798


namespace NUMINAMATH_GPT_lemon_heads_per_package_l1607_160794

theorem lemon_heads_per_package (total_lemon_heads boxes : ℕ)
  (H : total_lemon_heads = 54)
  (B : boxes = 9)
  (no_leftover : total_lemon_heads % boxes = 0) :
  total_lemon_heads / boxes = 6 :=
sorry

end NUMINAMATH_GPT_lemon_heads_per_package_l1607_160794


namespace NUMINAMATH_GPT_geo_seq_value_l1607_160736

variable (a : ℕ → ℝ)
variable (a_2 : a 2 = 2) 
variable (a_4 : a 4 = 8)
variable (geo_prop : a 2 * a 6 = (a 4) ^ 2)

theorem geo_seq_value : a 6 = 32 := 
by 
  sorry

end NUMINAMATH_GPT_geo_seq_value_l1607_160736


namespace NUMINAMATH_GPT_total_trolls_l1607_160754

theorem total_trolls (P B T : ℕ) (hP : P = 6) (hB : B = 4 * P - 6) (hT : T = B / 2) : P + B + T = 33 := by
  sorry

end NUMINAMATH_GPT_total_trolls_l1607_160754


namespace NUMINAMATH_GPT_rentExpenses_l1607_160735

noncomputable def monthlySalary : ℝ := 23000
noncomputable def milkExpenses : ℝ := 1500
noncomputable def groceriesExpenses : ℝ := 4500
noncomputable def educationExpenses : ℝ := 2500
noncomputable def petrolExpenses : ℝ := 2000
noncomputable def miscellaneousExpenses : ℝ := 5200
noncomputable def savings : ℝ := 2300

-- Calculating total non-rent expenses
noncomputable def totalNonRentExpenses : ℝ :=
  milkExpenses + groceriesExpenses + educationExpenses + petrolExpenses + miscellaneousExpenses

-- The rent expenses theorem
theorem rentExpenses : totalNonRentExpenses + savings + 5000 = monthlySalary :=
by sorry

end NUMINAMATH_GPT_rentExpenses_l1607_160735


namespace NUMINAMATH_GPT_pure_imaginary_value_l1607_160742

theorem pure_imaginary_value (a : ℝ) 
  (h1 : (a^2 - 3 * a + 2) = 0) 
  (h2 : (a - 2) ≠ 0) : a = 1 := sorry

end NUMINAMATH_GPT_pure_imaginary_value_l1607_160742


namespace NUMINAMATH_GPT_computer_price_decrease_l1607_160774

theorem computer_price_decrease 
  (initial_price : ℕ) 
  (decrease_factor : ℚ)
  (years : ℕ) 
  (final_price : ℕ) 
  (h1 : initial_price = 8100)
  (h2 : decrease_factor = 1/3)
  (h3 : years = 6)
  (h4 : final_price = 2400) : 
  initial_price * (1 - decrease_factor) ^ (years / 2) = final_price :=
by
  sorry

end NUMINAMATH_GPT_computer_price_decrease_l1607_160774


namespace NUMINAMATH_GPT_max_distance_from_B_to_P_l1607_160733

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -4, y := 1 }
def P : Point := { x := 3, y := -1 }

def line_l (m : ℝ) (pt : Point) : Prop :=
  (2 * m + 1) * pt.x - (m - 1) * pt.y - m - 5 = 0

theorem max_distance_from_B_to_P :
  ∃ B : Point, A = { x := -4, y := 1 } → 
               (∀ m : ℝ, line_l m B) →
               ∃ d, d = 5 + Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_max_distance_from_B_to_P_l1607_160733


namespace NUMINAMATH_GPT_prove_option_d_l1607_160747

-- Definitions of conditions
variables (a b : ℝ)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0)
variable (h_lt : a < b)

-- The theorem to be proved
theorem prove_option_d : a^3 < b^3 :=
sorry

end NUMINAMATH_GPT_prove_option_d_l1607_160747


namespace NUMINAMATH_GPT_father_age_38_l1607_160703

variable (F S : ℕ)
variable (h1 : S = 14)
variable (h2 : F - 10 = 7 * (S - 10))

theorem father_age_38 : F = 38 :=
by
  sorry

end NUMINAMATH_GPT_father_age_38_l1607_160703


namespace NUMINAMATH_GPT_triangle_side_ratio_l1607_160780

theorem triangle_side_ratio (a b c : ℝ) (h1 : a + b ≤ 2 * c) (h2 : b + c ≤ 3 * a) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  2 / 3 < c / a ∧ c / a < 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_ratio_l1607_160780


namespace NUMINAMATH_GPT_vertical_asymptote_values_l1607_160710

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := (x^2 - x + c) / (x^2 + x - 20)

theorem vertical_asymptote_values (c : ℝ) :
  (∃ x : ℝ, (x^2 + x - 20 = 0 ∧ x^2 - x + c = 0) ↔
   (c = -12 ∨ c = -30)) := sorry

end NUMINAMATH_GPT_vertical_asymptote_values_l1607_160710


namespace NUMINAMATH_GPT_bonus_implies_completion_l1607_160785

variable (John : Type)
variable (completes_all_tasks_perfectly : John → Prop)
variable (receives_bonus : John → Prop)

theorem bonus_implies_completion :
  (∀ e : John, completes_all_tasks_perfectly e → receives_bonus e) →
  (∀ e : John, receives_bonus e → completes_all_tasks_perfectly e) :=
by
  intros h e
  sorry

end NUMINAMATH_GPT_bonus_implies_completion_l1607_160785


namespace NUMINAMATH_GPT_gcd_pow_minus_one_l1607_160778

theorem gcd_pow_minus_one (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  Nat.gcd (2^m - 1) (2^n - 1) = 2^(Nat.gcd m n) - 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_pow_minus_one_l1607_160778


namespace NUMINAMATH_GPT_crow_eating_time_l1607_160748

theorem crow_eating_time (n : ℕ) (h : ∀ t : ℕ, t = (n / 5) → t = 4) : (4 + (4 / 5) = 4.8) :=
by
  sorry

end NUMINAMATH_GPT_crow_eating_time_l1607_160748


namespace NUMINAMATH_GPT_initial_bottles_count_l1607_160773

theorem initial_bottles_count : 
  ∀ (jason_buys harry_buys bottles_left initial_bottles : ℕ), 
  jason_buys = 5 → 
  harry_buys = 6 → 
  bottles_left = 24 → 
  initial_bottles = bottles_left + jason_buys + harry_buys → 
  initial_bottles = 35 :=
by
  intros jason_buys harry_buys bottles_left initial_bottles
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_initial_bottles_count_l1607_160773


namespace NUMINAMATH_GPT_sqrt_sum_eq_l1607_160799

theorem sqrt_sum_eq :
  (Real.sqrt (9 / 2) + Real.sqrt (2 / 9)) = (11 * Real.sqrt 2 / 6) :=
sorry

end NUMINAMATH_GPT_sqrt_sum_eq_l1607_160799


namespace NUMINAMATH_GPT_train_speed_with_coaches_l1607_160716

theorem train_speed_with_coaches (V₀ : ℝ) (V₉ V₁₆ : ℝ) (k : ℝ) :
  V₀ = 30 → V₁₆ = 14 → V₉ = 30 - k * (9: ℝ) ^ (1/2: ℝ) ∧ V₁₆ = 30 - k * (16: ℝ) ^ (1/2: ℝ) →
  V₉ = 18 :=
by sorry

end NUMINAMATH_GPT_train_speed_with_coaches_l1607_160716


namespace NUMINAMATH_GPT_parallelogram_area_example_l1607_160739

noncomputable def area_parallelogram (A B C D : (ℝ × ℝ)) : ℝ := 
  0.5 * |(A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * A.1)|

theorem parallelogram_area_example : 
  let A := (0, 0)
  let B := (20, 0)
  let C := (25, 7)
  let D := (5, 7)
  area_parallelogram A B C D = 140 := 
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_example_l1607_160739


namespace NUMINAMATH_GPT_calc1_calc2_calc3_calc4_l1607_160771

theorem calc1 : 23 + (-16) - (-7) = 14 :=
by
  sorry

theorem calc2 : (3/4 - 7/8 - 5/12) * (-24) = 13 :=
by
  sorry

theorem calc3 : ((7/4 - 7/8 - 7/12) / (-7/8)) + ((-7/8) / (7/4 - 7/8 - 7/12)) = -10/3 :=
by
  sorry

theorem calc4 : -1^4 - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_calc1_calc2_calc3_calc4_l1607_160771


namespace NUMINAMATH_GPT_Paul_dig_days_alone_l1607_160792

/-- Jake's daily work rate -/
def Jake_work_rate : ℚ := 1 / 16

/-- Hari's daily work rate -/
def Hari_work_rate : ℚ := 1 / 48

/-- Combined work rate of Jake, Paul, and Hari, when they work together they can dig the well in 8 days -/
def combined_work_rate (Paul_work_rate : ℚ) : Prop :=
  Jake_work_rate + Paul_work_rate + Hari_work_rate = 1 / 8

/-- Theorem stating that Paul can dig the well alone in 24 days -/
theorem Paul_dig_days_alone : ∃ (P : ℚ), combined_work_rate (1 / P) ∧ P = 24 :=
by
  use 24
  unfold combined_work_rate
  sorry

end NUMINAMATH_GPT_Paul_dig_days_alone_l1607_160792


namespace NUMINAMATH_GPT_solved_fraction_equation_l1607_160767

theorem solved_fraction_equation :
  ∀ (x : ℚ),
    x ≠ 2 →
    x ≠ 7 →
    x ≠ -5 →
    (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 4*x - 5) / (x^2 - 2*x - 35) →
    x = 55 / 13 := by
  sorry

end NUMINAMATH_GPT_solved_fraction_equation_l1607_160767


namespace NUMINAMATH_GPT_find_radius_yz_l1607_160714

-- Define the setup for the centers of the circles and their radii
def circle_with_center (c : Type*) (radius : ℝ) : Prop := sorry
def tangent_to (c₁ c₂ : Type*) : Prop := sorry

-- Given conditions
variable (O X Y Z : Type*)
variable (r : ℝ)
variable (Xe_radius : circle_with_center X 1)
variable (O_radius : circle_with_center O 2)
variable (XtangentO : tangent_to X O)
variable (YtangentO : tangent_to Y O)
variable (YtangentX : tangent_to Y X)
variable (YtangentZ : tangent_to Y Z)
variable (ZtangentO : tangent_to Z O)
variable (ZtangentX : tangent_to Z X)
variable (ZtangentY : tangent_to Z Y)

-- The theorem to prove
theorem find_radius_yz :
  r = 8 / 9 := sorry

end NUMINAMATH_GPT_find_radius_yz_l1607_160714


namespace NUMINAMATH_GPT_campers_rowing_morning_equals_41_l1607_160705

def campers_went_rowing_morning (hiking_morning : ℕ) (rowing_afternoon : ℕ) (total : ℕ) : ℕ :=
  total - (hiking_morning + rowing_afternoon)

theorem campers_rowing_morning_equals_41 :
  ∀ (hiking_morning rowing_afternoon total : ℕ), hiking_morning = 4 → rowing_afternoon = 26 → total = 71 → campers_went_rowing_morning hiking_morning rowing_afternoon total = 41 := by
  intros hiking_morning rowing_afternoon total hiking_morning_cond rowing_afternoon_cond total_cond
  rw [hiking_morning_cond, rowing_afternoon_cond, total_cond]
  exact rfl

end NUMINAMATH_GPT_campers_rowing_morning_equals_41_l1607_160705


namespace NUMINAMATH_GPT_swimmers_meet_l1607_160722

def time_to_meet (pool_length speed1 speed2 time: ℕ) : ℕ :=
  (time * (speed1 + speed2)) / pool_length

theorem swimmers_meet
  (pool_length : ℕ)
  (speed1 : ℕ)
  (speed2 : ℕ)
  (total_time : ℕ) :
  total_time = 12 * 60 →
  pool_length = 90 →
  speed1 = 3 →
  speed2 = 2 →
  time_to_meet pool_length speed1 speed2 total_time = 20 := by
  sorry

end NUMINAMATH_GPT_swimmers_meet_l1607_160722


namespace NUMINAMATH_GPT_approximation_of_11_28_relative_to_10000_l1607_160711

def place_value_to_approximate (x : Float) (reference : Float) : String :=
  if x < reference / 10 then "tens"
  else if x < reference / 100 then "hundreds"
  else if x < reference / 1000 then "thousands"
  else if x < reference / 10000 then "ten thousands"
  else "greater than ten thousands"

theorem approximation_of_11_28_relative_to_10000:
  place_value_to_approximate 11.28 10000 = "hundreds" :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_approximation_of_11_28_relative_to_10000_l1607_160711


namespace NUMINAMATH_GPT_problem_statement_l1607_160709

theorem problem_statement : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1607_160709


namespace NUMINAMATH_GPT_jellyfish_cost_l1607_160777

theorem jellyfish_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : J = 20 := by
  sorry

end NUMINAMATH_GPT_jellyfish_cost_l1607_160777


namespace NUMINAMATH_GPT_lemons_required_for_new_recipe_l1607_160737

noncomputable def lemons_needed_to_make_gallons (lemons_original : ℕ) (gallons_original : ℕ) (additional_lemons : ℕ) (additional_gallons : ℕ) (gallons_new : ℕ) : ℝ :=
  let lemons_per_gallon := (lemons_original : ℝ) / (gallons_original : ℝ)
  let additional_lemons_per_gallon := (additional_lemons : ℝ) / (additional_gallons : ℝ)
  let total_lemons_per_gallon := lemons_per_gallon + additional_lemons_per_gallon
  total_lemons_per_gallon * (gallons_new : ℝ)

theorem lemons_required_for_new_recipe : lemons_needed_to_make_gallons 36 48 2 6 18 = 19.5 :=
by
  sorry

end NUMINAMATH_GPT_lemons_required_for_new_recipe_l1607_160737


namespace NUMINAMATH_GPT_product_of_fractions_is_25_div_324_l1607_160713

noncomputable def product_of_fractions : ℚ := 
  (10 / 6) * (4 / 20) * (20 / 12) * (16 / 32) * 
  (40 / 24) * (8 / 40) * (60 / 36) * (32 / 64)

theorem product_of_fractions_is_25_div_324 : product_of_fractions = 25 / 324 := 
  sorry

end NUMINAMATH_GPT_product_of_fractions_is_25_div_324_l1607_160713


namespace NUMINAMATH_GPT_ratio_final_to_original_l1607_160772

-- Given conditions
variable (d : ℝ)
variable (h1 : 364 = d * 1.30)

-- Problem statement
theorem ratio_final_to_original : (364 / d) = 1.3 := 
by sorry

end NUMINAMATH_GPT_ratio_final_to_original_l1607_160772


namespace NUMINAMATH_GPT_range_of_a_for_three_distinct_zeros_l1607_160756

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a_for_three_distinct_zeros : 
  ∀ a : ℝ, (∀ x y : ℝ, x ≠ y → f x a = 0 → f y a = 0 → (f (1:ℝ) a < 0 ∧ f (-1:ℝ) a > 0)) ↔ (-2 < a ∧ a < 2) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_three_distinct_zeros_l1607_160756


namespace NUMINAMATH_GPT_type_2004_A_least_N_type_B_diff_2004_l1607_160723

def game_type_A (N : ℕ) : Prop :=
  ∀ n, (1 ≤ n ∧ n ≤ N) → (n % 2 = 0 → false) 

def game_type_B (N : ℕ) : Prop :=
  ∃ n, (1 ≤ n ∧ n ≤ N) ∧ (n % 2 = 0 → true)


theorem type_2004_A : game_type_A 2004 :=
sorry

theorem least_N_type_B_diff_2004 : ∀ N, N > 2004 → game_type_B N → N = 2048 :=
sorry

end NUMINAMATH_GPT_type_2004_A_least_N_type_B_diff_2004_l1607_160723


namespace NUMINAMATH_GPT_price_of_orange_is_60_l1607_160718

theorem price_of_orange_is_60
  (x a o : ℕ)
  (h1 : 40 * a + x * o = 540)
  (h2 : a + o = 10)
  (h3 : 40 * a + x * (o - 5) = 240) :
  x = 60 :=
by
  sorry

end NUMINAMATH_GPT_price_of_orange_is_60_l1607_160718


namespace NUMINAMATH_GPT_sum_odd_product_even_l1607_160744

theorem sum_odd_product_even (a b : ℤ) (h1 : ∃ k : ℤ, a = 2 * k) 
                             (h2 : ∃ m : ℤ, b = 2 * m + 1) 
                             (h3 : ∃ n : ℤ, a + b = 2 * n + 1) : 
  ∃ p : ℤ, a * b = 2 * p := 
  sorry

end NUMINAMATH_GPT_sum_odd_product_even_l1607_160744


namespace NUMINAMATH_GPT_jake_peaches_calculation_l1607_160770

variable (S_p : ℕ) (J_p : ℕ)

-- Given that Steven has 19 peaches
def steven_peaches : ℕ := 19

-- Jake has 12 fewer peaches than Steven
def jake_peaches : ℕ := S_p - 12

theorem jake_peaches_calculation (h1 : S_p = steven_peaches) (h2 : S_p = 19) :
  J_p = jake_peaches := 
by
  sorry

end NUMINAMATH_GPT_jake_peaches_calculation_l1607_160770


namespace NUMINAMATH_GPT_Luca_weight_loss_per_year_l1607_160719

def Barbi_weight_loss_per_month : Real := 1.5
def months_in_a_year : Nat := 12
def Luca_years : Nat := 11
def extra_weight_Luca_lost : Real := 81

theorem Luca_weight_loss_per_year :
  (Barbi_weight_loss_per_month * months_in_a_year + extra_weight_Luca_lost) / Luca_years = 9 := by
  sorry

end NUMINAMATH_GPT_Luca_weight_loss_per_year_l1607_160719


namespace NUMINAMATH_GPT_triangle_circle_area_relation_l1607_160775

theorem triangle_circle_area_relation (A B C : ℝ) (h : 15^2 + 20^2 = 25^2) (A_area_eq : A + B + 150 = C) :
  A + B + 150 = C :=
by
  -- The proof has been omitted.
  sorry

end NUMINAMATH_GPT_triangle_circle_area_relation_l1607_160775


namespace NUMINAMATH_GPT_green_apples_count_l1607_160761

-- Definitions for the conditions in the problem
def total_apples : ℕ := 19
def red_apples : ℕ := 3
def yellow_apples : ℕ := 14

-- Statement expressing that the number of green apples on the table is 2
theorem green_apples_count : (total_apples - red_apples - yellow_apples = 2) :=
by
  sorry

end NUMINAMATH_GPT_green_apples_count_l1607_160761


namespace NUMINAMATH_GPT_determine_m_ratio_l1607_160787

def ratio_of_C_to_A_investment (x : ℕ) (m : ℕ) (total_gain : ℕ) (a_share : ℕ) : Prop :=
  total_gain = 18000 ∧ a_share = 6000 ∧
  (12 * x / (12 * x + 4 * m * x) = 1 / 3)

theorem determine_m_ratio (x : ℕ) (m : ℕ) (h : ratio_of_C_to_A_investment x m 18000 6000) :
  m = 6 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_ratio_l1607_160787


namespace NUMINAMATH_GPT_percent_difference_l1607_160704

theorem percent_difference:
  let percent_value1 := (55 / 100) * 40
  let fraction_value2 := (4 / 5) * 25
  percent_value1 - fraction_value2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_percent_difference_l1607_160704


namespace NUMINAMATH_GPT_donuts_left_l1607_160725

def initial_donuts : ℕ := 50
def after_bill_eats (initial : ℕ) : ℕ := initial - 2
def after_secretary_takes (remaining_after_bill : ℕ) : ℕ := remaining_after_bill - 4
def coworkers_take (remaining_after_secretary : ℕ) : ℕ := remaining_after_secretary / 2
def final_donuts (initial : ℕ) : ℕ :=
  let remaining_after_bill := after_bill_eats initial
  let remaining_after_secretary := after_secretary_takes remaining_after_bill
  remaining_after_secretary - coworkers_take remaining_after_secretary

theorem donuts_left : final_donuts 50 = 22 := by
  sorry

end NUMINAMATH_GPT_donuts_left_l1607_160725


namespace NUMINAMATH_GPT_average_speed_of_car_l1607_160732

noncomputable def averageSpeed : ℚ := 
  let speed1 := 45     -- kph
  let distance1 := 15  -- km
  let speed2 := 55     -- kph
  let distance2 := 30  -- km
  let speed3 := 65     -- kph
  let time3 := 35 / 60 -- hours
  let speed4 := 52     -- kph
  let time4 := 20 / 60 -- hours
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4
  let totalDistance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let totalTime := time1 + time2 + time3 + time4
  totalDistance / totalTime

theorem average_speed_of_car :
  abs (averageSpeed - 55.85) < 0.01 := 
  sorry

end NUMINAMATH_GPT_average_speed_of_car_l1607_160732


namespace NUMINAMATH_GPT_find_value_of_t_l1607_160793

variable (a b v d t r : ℕ)

-- All variables are non-zero digits (1-9)
axiom non_zero_a : 0 < a ∧ a < 10
axiom non_zero_b : 0 < b ∧ b < 10
axiom non_zero_v : 0 < v ∧ v < 10
axiom non_zero_d : 0 < d ∧ d < 10
axiom non_zero_t : 0 < t ∧ t < 10
axiom non_zero_r : 0 < r ∧ r < 10

-- Given conditions
axiom condition1 : a + b = v
axiom condition2 : v + d = t
axiom condition3 : t + a = r
axiom condition4 : b + d + r = 18

theorem find_value_of_t : t = 9 :=
by sorry

end NUMINAMATH_GPT_find_value_of_t_l1607_160793


namespace NUMINAMATH_GPT_distinct_students_count_l1607_160786

-- Definition of the initial parameters
def num_gauss : Nat := 12
def num_euler : Nat := 10
def num_fibonnaci : Nat := 7
def overlap : Nat := 1

-- The main theorem to prove
theorem distinct_students_count : num_gauss + num_euler + num_fibonnaci - overlap = 28 := by
  sorry

end NUMINAMATH_GPT_distinct_students_count_l1607_160786


namespace NUMINAMATH_GPT_factor_expression_l1607_160729

theorem factor_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b)^2 * (b - c)^2 * (c - a)^2 * (a + b + c) :=
sorry

end NUMINAMATH_GPT_factor_expression_l1607_160729


namespace NUMINAMATH_GPT_positive_integer_pair_solution_l1607_160726

theorem positive_integer_pair_solution :
  ∃ a b : ℕ, (a > 0) ∧ (b > 0) ∧ 
    ¬ (7 ∣ (a * b * (a + b))) ∧ 
    (7^7 ∣ ((a + b)^7 - a^7 - b^7)) ∧ 
    (a, b) = (18, 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_positive_integer_pair_solution_l1607_160726


namespace NUMINAMATH_GPT_line_intersection_points_l1607_160746

def line_intersects_axes (x y : ℝ) : Prop :=
  (4 * y - 5 * x = 20)

theorem line_intersection_points :
  ∃ p1 p2, line_intersects_axes p1.1 p1.2 ∧ line_intersects_axes p2.1 p2.2 ∧
    (p1 = (-4, 0) ∧ p2 = (0, 5)) :=
by
  sorry

end NUMINAMATH_GPT_line_intersection_points_l1607_160746
