import Mathlib

namespace NUMINAMATH_GPT_find_x_of_floor_eq_72_l1419_141941

theorem find_x_of_floor_eq_72 (x : ℝ) (hx_pos : 0 < x) (hx_eq : x * ⌊x⌋ = 72) : x = 9 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_of_floor_eq_72_l1419_141941


namespace NUMINAMATH_GPT_basketball_campers_l1419_141938

theorem basketball_campers (total_campers soccer_campers football_campers : ℕ)
  (h_total : total_campers = 88)
  (h_soccer : soccer_campers = 32)
  (h_football : football_campers = 32) :
  total_campers - soccer_campers - football_campers = 24 :=
by
  sorry

end NUMINAMATH_GPT_basketball_campers_l1419_141938


namespace NUMINAMATH_GPT_scientific_notation_example_l1419_141958

theorem scientific_notation_example : 0.00001 = 1 * 10^(-5) :=
sorry

end NUMINAMATH_GPT_scientific_notation_example_l1419_141958


namespace NUMINAMATH_GPT_books_in_series_l1419_141943

theorem books_in_series (books_watched : ℕ) (movies_watched : ℕ) (read_more_movies_than_books : books_watched + 3 = movies_watched) (watched_movies : movies_watched = 19) : books_watched = 16 :=
by sorry

end NUMINAMATH_GPT_books_in_series_l1419_141943


namespace NUMINAMATH_GPT_sqrt_22_gt_4_l1419_141920

theorem sqrt_22_gt_4 : Real.sqrt 22 > 4 := 
sorry

end NUMINAMATH_GPT_sqrt_22_gt_4_l1419_141920


namespace NUMINAMATH_GPT_tan_fraction_identity_l1419_141910

theorem tan_fraction_identity (x : ℝ) 
  (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_tan_fraction_identity_l1419_141910


namespace NUMINAMATH_GPT_porche_project_time_l1419_141964

theorem porche_project_time :
  let total_time := 180
  let math_time := 45
  let english_time := 30
  let science_time := 50
  let history_time := 25
  let homework_time := math_time + english_time + science_time + history_time 
  total_time - homework_time = 30 :=
by
  sorry

end NUMINAMATH_GPT_porche_project_time_l1419_141964


namespace NUMINAMATH_GPT_collinear_points_l1419_141952

axiom collinear (A B C : ℝ × ℝ × ℝ) : Prop

theorem collinear_points (c d : ℝ) (h : collinear (2, c, d) (c, 3, d) (c, d, 4)) : c + d = 6 :=
sorry

end NUMINAMATH_GPT_collinear_points_l1419_141952


namespace NUMINAMATH_GPT_sphere_volume_from_area_l1419_141994

/-- Given the surface area of a sphere is 24π, prove that the volume of the sphere is 8√6π. -/ 
theorem sphere_volume_from_area :
  ∀ {R : ℝ},
    4 * Real.pi * R^2 = 24 * Real.pi →
    (4 / 3) * Real.pi * R^3 = 8 * Real.sqrt 6 * Real.pi :=
by
  intro R h
  sorry

end NUMINAMATH_GPT_sphere_volume_from_area_l1419_141994


namespace NUMINAMATH_GPT_people_in_group_l1419_141915

theorem people_in_group (n : ℕ) 
  (h1 : ∀ (new_weight old_weight : ℕ), old_weight = 70 → new_weight = 110 → (70 * n + (new_weight - old_weight) = 70 * n + 4 * n)) :
  n = 10 :=
sorry

end NUMINAMATH_GPT_people_in_group_l1419_141915


namespace NUMINAMATH_GPT_lollipop_problem_l1419_141955

def Henry_lollipops (A : Nat) : Nat := A + 30
def Diane_lollipops (A : Nat) : Nat := 2 * A
def Total_days (H A D : Nat) (daily_rate : Nat) : Nat := (H + A + D) / daily_rate

theorem lollipop_problem
  (A : Nat) (H : Nat) (D : Nat) (daily_rate : Nat)
  (h₁ : A = 60)
  (h₂ : H = Henry_lollipops A)
  (h₃ : D = Diane_lollipops A)
  (h₄ : daily_rate = 45)
  : Total_days H A D daily_rate = 6 := by
  sorry

end NUMINAMATH_GPT_lollipop_problem_l1419_141955


namespace NUMINAMATH_GPT_liquid_X_percentage_in_new_solution_l1419_141961

noncomputable def solutionY_initial_kg : ℝ := 10
noncomputable def percentage_liquid_X : ℝ := 0.30
noncomputable def evaporated_water_kg : ℝ := 2
noncomputable def added_solutionY_kg : ℝ := 2

-- Calculate the amount of liquid X in the original solution
noncomputable def initial_liquid_X_kg : ℝ :=
  percentage_liquid_X * solutionY_initial_kg

-- Calculate the remaining weight after evaporation
noncomputable def remaining_weight_kg : ℝ :=
  solutionY_initial_kg - evaporated_water_kg

-- Calculate the amount of liquid X after evaporation
noncomputable def remaining_liquid_X_kg : ℝ := initial_liquid_X_kg

-- Since only water evaporates, remaining water weight
noncomputable def remaining_water_kg : ℝ :=
  remaining_weight_kg - remaining_liquid_X_kg

-- Calculate the amount of liquid X in the added solution
noncomputable def added_liquid_X_kg : ℝ :=
  percentage_liquid_X * added_solutionY_kg

-- Total liquid X in the new solution
noncomputable def new_liquid_X_kg : ℝ :=
  remaining_liquid_X_kg + added_liquid_X_kg

-- Calculate the water in the added solution
noncomputable def percentage_water : ℝ := 0.70
noncomputable def added_water_kg : ℝ :=
  percentage_water * added_solutionY_kg

-- Total water in the new solution
noncomputable def new_water_kg : ℝ :=
  remaining_water_kg + added_water_kg

-- Total weight of the new solution
noncomputable def new_total_weight_kg : ℝ :=
  remaining_weight_kg + added_solutionY_kg

-- Percentage of liquid X in the new solution
noncomputable def percentage_new_liquid_X : ℝ :=
  (new_liquid_X_kg / new_total_weight_kg) * 100

-- The proof statement
theorem liquid_X_percentage_in_new_solution :
  percentage_new_liquid_X = 36 :=
by
  sorry

end NUMINAMATH_GPT_liquid_X_percentage_in_new_solution_l1419_141961


namespace NUMINAMATH_GPT_sum_of_three_squares_l1419_141978

theorem sum_of_three_squares (n : ℕ) (h_pos : 0 < n) (h_square : ∃ m : ℕ, 3 * n + 1 = m^2) : ∃ x y z : ℕ, n + 1 = x^2 + y^2 + z^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_squares_l1419_141978


namespace NUMINAMATH_GPT_min_value_of_linear_expression_l1419_141949

theorem min_value_of_linear_expression {x y : ℝ} (h1 : 2 * x - y ≥ 0) (h2 : x + y - 3 ≥ 0) (h3 : y - x ≥ 0) :
  ∃ z, z = 2 * x + y ∧ z = 4 := by
  sorry

end NUMINAMATH_GPT_min_value_of_linear_expression_l1419_141949


namespace NUMINAMATH_GPT_largest_n_with_integer_solutions_l1419_141923

theorem largest_n_with_integer_solutions : ∃ n, ∀ x y1 y2 y3 y4, 
 ( ((x + 1)^2 + y1^2) = ((x + 2)^2 + y2^2) ∧  ((x + 2)^2 + y2^2) = ((x + 3)^2 + y3^2) ∧ 
  ((x + 3)^2 + y3^2) = ((x + 4)^2 + y4^2)) → (n = 3) := sorry

end NUMINAMATH_GPT_largest_n_with_integer_solutions_l1419_141923


namespace NUMINAMATH_GPT_sum_modulo_seven_l1419_141973

theorem sum_modulo_seven :
  let s := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999
  s % 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_modulo_seven_l1419_141973


namespace NUMINAMATH_GPT_solve_player_coins_l1419_141957

def player_coins (n m k: ℕ) : Prop :=
  ∃ k, 
  (m = k * (n - 1) + 50) ∧ 
  (3 * m = 7 * n * k - 3 * k + 74) ∧ 
  (m = 69)

theorem solve_player_coins (n m k : ℕ) : player_coins n m k :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_player_coins_l1419_141957


namespace NUMINAMATH_GPT_no_prime_pairs_sum_53_l1419_141944

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_pairs_sum_53 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end NUMINAMATH_GPT_no_prime_pairs_sum_53_l1419_141944


namespace NUMINAMATH_GPT_mean_of_second_set_l1419_141974

def mean (l: List ℕ) : ℚ :=
  (l.sum: ℚ) / l.length

theorem mean_of_second_set (x: ℕ) 
  (h: mean [28, x, 42, 78, 104] = 90): 
  mean [128, 255, 511, 1023, x] = 423 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_second_set_l1419_141974


namespace NUMINAMATH_GPT_range_of_a_l1419_141993

noncomputable def f (a : ℝ) (x : ℝ) := Real.sqrt (Real.exp x + (Real.exp 1 - 1) * x - a)
def exists_b_condition (a : ℝ) : Prop := ∃ b : ℝ, b ∈ Set.Icc 0 1 ∧ f a b = b

theorem range_of_a (a : ℝ) : exists_b_condition a → a ∈ Set.Icc 1 (2 * Real.exp 1 - 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1419_141993


namespace NUMINAMATH_GPT_calc_f_2005_2007_zero_l1419_141977

variable {R : Type} [LinearOrderedField R]

def odd_function (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

def periodic_function (f : R → R) (p : R) : Prop :=
  ∀ x, f (x + p) = f x

theorem calc_f_2005_2007_zero
  {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (h_period : periodic_function f 4) :
  f 2005 + f 2006 + f 2007 = 0 :=
sorry

end NUMINAMATH_GPT_calc_f_2005_2007_zero_l1419_141977


namespace NUMINAMATH_GPT_hamsters_count_l1419_141997

-- Define the conditions as parameters
variables (ratio_rabbit_hamster : ℕ × ℕ)
variables (rabbits : ℕ)
variables (hamsters : ℕ)

-- Given conditions
def ratio_condition : ratio_rabbit_hamster = (4, 5) := sorry
def rabbits_condition : rabbits = 20 := sorry

-- The theorem to be proven
theorem hamsters_count : ratio_rabbit_hamster = (4, 5) -> rabbits = 20 -> hamsters = 25 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_hamsters_count_l1419_141997


namespace NUMINAMATH_GPT_window_width_is_28_l1419_141922

noncomputable def window_width (y : ℝ) : ℝ :=
  12 * y + 4

theorem window_width_is_28 : ∃ (y : ℝ), window_width y = 28 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_window_width_is_28_l1419_141922


namespace NUMINAMATH_GPT_juan_distance_l1419_141904

def running_time : ℝ := 80.0
def speed : ℝ := 10.0
def distance : ℝ := running_time * speed

theorem juan_distance :
  distance = 800.0 :=
by
  sorry

end NUMINAMATH_GPT_juan_distance_l1419_141904


namespace NUMINAMATH_GPT_truncated_pyramid_volume_l1419_141932

theorem truncated_pyramid_volume :
  let unit_cube_vol := 1
  let tetrahedron_base_area := 1 / 2
  let tetrahedron_height := 1 / 2
  let tetrahedron_vol := (1 / 3) * tetrahedron_base_area * tetrahedron_height
  let two_tetrahedra_vol := 2 * tetrahedron_vol
  let truncated_pyramid_vol := unit_cube_vol - two_tetrahedra_vol
  truncated_pyramid_vol = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_truncated_pyramid_volume_l1419_141932


namespace NUMINAMATH_GPT_decagon_triangle_probability_l1419_141929

theorem decagon_triangle_probability : 
  let total_vertices := 10
  let total_triangles := Nat.choose total_vertices 3
  let favorable_triangles := 10
  (total_triangles > 0) → 
  (favorable_triangles / total_triangles : ℚ) = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_decagon_triangle_probability_l1419_141929


namespace NUMINAMATH_GPT_intersection_P_Q_l1419_141968

open Set

noncomputable def P : Set ℝ := {x | abs (x - 1) < 4}
noncomputable def Q : Set ℝ := {x | ∃ y, y = Real.log (x + 2) }

theorem intersection_P_Q :
  (P ∩ Q) = {x : ℝ | -2 < x ∧ x < 5} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l1419_141968


namespace NUMINAMATH_GPT_option_B_is_incorrect_l1419_141908

-- Define the set A
def A := { x : ℤ | x ^ 2 - 4 = 0 }

-- Statement to prove that -2 is an element of A
theorem option_B_is_incorrect : -2 ∈ A :=
sorry

end NUMINAMATH_GPT_option_B_is_incorrect_l1419_141908


namespace NUMINAMATH_GPT_estimated_fish_in_pond_l1419_141916

theorem estimated_fish_in_pond :
  ∀ (number_marked_first_catch total_second_catch number_marked_second_catch : ℕ),
    number_marked_first_catch = 100 →
    total_second_catch = 108 →
    number_marked_second_catch = 9 →
    ∃ est_total_fish : ℕ, (number_marked_second_catch / total_second_catch : ℝ) = (number_marked_first_catch / est_total_fish : ℝ) ∧ est_total_fish = 1200 := 
by
  intros number_marked_first_catch total_second_catch number_marked_second_catch
  sorry

end NUMINAMATH_GPT_estimated_fish_in_pond_l1419_141916


namespace NUMINAMATH_GPT_emily_collected_total_eggs_l1419_141905

def eggs_in_setA : ℕ := (200 * 36) + (250 * 24)
def eggs_in_setB : ℕ := (375 * 42) - 80
def eggs_in_setC : ℕ := (560 / 2 * 50) + (560 / 2 * 32)

def total_eggs_collected : ℕ := eggs_in_setA + eggs_in_setB + eggs_in_setC

theorem emily_collected_total_eggs : total_eggs_collected = 51830 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_emily_collected_total_eggs_l1419_141905


namespace NUMINAMATH_GPT_find_m_l1419_141925

-- Defining vectors a and b
def a (m : ℝ) : ℝ × ℝ := (2, m)
def b : ℝ × ℝ := (1, -1)

-- Proving that if b is perpendicular to (a + 2b), then m = 6
theorem find_m (m : ℝ) :
  let a_vec := a m
  let b_vec := b
  let sum_vec := (a_vec.1 + 2 * b_vec.1, a_vec.2 + 2 * b_vec.2)
  (b_vec.1 * sum_vec.1 + b_vec.2 * sum_vec.2 = 0) → m = 6 :=
by
  intros a_vec b_vec sum_vec perp_cond
  sorry

end NUMINAMATH_GPT_find_m_l1419_141925


namespace NUMINAMATH_GPT_path_count_1800_l1419_141970

-- Define the coordinates of the points
def A := (0, 8)
def B := (4, 5)
def C := (7, 2)
def D := (9, 0)

-- Function to calculate the number of combinatorial paths
def comb_paths (steps_right steps_down : ℕ) : ℕ :=
  Nat.choose (steps_right + steps_down) steps_right

-- Define the number of steps for each segment
def steps_A_B := (4, 2)  -- 4 right, 2 down
def steps_B_C := (3, 3)  -- 3 right, 3 down
def steps_C_D := (2, 2)  -- 2 right, 2 down

-- Calculate the number of paths for each segment
def paths_A_B := comb_paths steps_A_B.1 steps_A_B.2
def paths_B_C := comb_paths steps_B_C.1 steps_B_C.2
def paths_C_D := comb_paths steps_C_D.1 steps_C_D.2

-- Calculate the total number of paths combining all segments
def total_paths : ℕ :=
  paths_A_B * paths_B_C * paths_C_D

theorem path_count_1800 :
  total_paths = 1800 := by
  sorry

end NUMINAMATH_GPT_path_count_1800_l1419_141970


namespace NUMINAMATH_GPT_distribution_count_l1419_141942

-- Making the function for counting the number of valid distributions
noncomputable def countValidDistributions : ℕ :=
  let cases1 := 4                            -- One box contains all five balls
  let cases2 := 4 * 3                        -- One box has 4 balls, another has 1
  let cases3 := 4 * 3                        -- One box has 3 balls, another has 2
  let cases4 := 6 * 2                        -- Two boxes have 2 balls, and one has 1
  let cases5 := 4 * 3                        -- One box has 3 balls, and two boxes have 1 each
  cases1 + cases2 + cases3 + cases4 + cases5 -- Sum of all cases

-- Theorem statement: the count of valid distributions equals 52
theorem distribution_count : countValidDistributions = 52 := 
  by
    sorry

end NUMINAMATH_GPT_distribution_count_l1419_141942


namespace NUMINAMATH_GPT_initial_percentage_of_jasmine_water_l1419_141945

-- Definitions
def v_initial : ℝ := 80
def v_jasmine_added : ℝ := 8
def v_water_added : ℝ := 12
def percentage_final : ℝ := 16
def v_final : ℝ := v_initial + v_jasmine_added + v_water_added

-- Lean 4 statement that frames the proof problem
theorem initial_percentage_of_jasmine_water (P : ℝ) :
  (P / 100) * v_initial + v_jasmine_added = (percentage_final / 100) * v_final → P = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_percentage_of_jasmine_water_l1419_141945


namespace NUMINAMATH_GPT_square_sum_l1419_141966

theorem square_sum (a b : ℝ) (h1 : a + b = 8) (h2 : a * b = -2) : a^2 + b^2 = 68 := 
by 
  sorry

end NUMINAMATH_GPT_square_sum_l1419_141966


namespace NUMINAMATH_GPT_roots_quadratic_sum_of_squares_l1419_141969

theorem roots_quadratic_sum_of_squares :
  ∀ x1 x2 : ℝ, (x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0) → x1^2 + x2^2 = 6 :=
by
  intros x1 x2 h
  -- proof goes here
  sorry

end NUMINAMATH_GPT_roots_quadratic_sum_of_squares_l1419_141969


namespace NUMINAMATH_GPT_probability_even_sum_is_half_l1419_141950

-- Definitions for probability calculations
def prob_even_A : ℚ := 2 / 5
def prob_odd_A : ℚ := 3 / 5
def prob_even_B : ℚ := 1 / 2
def prob_odd_B : ℚ := 1 / 2

-- Sum is even if both are even or both are odd
def prob_even_sum := prob_even_A * prob_even_B + prob_odd_A * prob_odd_B

-- Theorem stating the final probability
theorem probability_even_sum_is_half : prob_even_sum = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_even_sum_is_half_l1419_141950


namespace NUMINAMATH_GPT_octal_addition_l1419_141926

theorem octal_addition (x y : ℕ) (h1 : x = 1 * 8^3 + 4 * 8^2 + 6 * 8^1 + 3 * 8^0)
                     (h2 : y = 2 * 8^2 + 7 * 8^1 + 5 * 8^0) :
  x + y = 1 * 8^3 + 7 * 8^2 + 5 * 8^1 + 0 * 8^0 := sorry

end NUMINAMATH_GPT_octal_addition_l1419_141926


namespace NUMINAMATH_GPT_area_increase_l1419_141979

theorem area_increase (a : ℝ) : ((a + 2) ^ 2 - a ^ 2 = 4 * a + 4) := by
  sorry

end NUMINAMATH_GPT_area_increase_l1419_141979


namespace NUMINAMATH_GPT_solve_trig_problem_l1419_141940

noncomputable def trig_problem (α : ℝ) : Prop :=
  α ∈ (Set.Ioo 0 (Real.pi / 2)) ∪ Set.Ioo (Real.pi / 2) Real.pi ∧
  ∃ r : ℝ, r ≠ 0 ∧ Real.sin α * r = Real.sin (2 * α) ∧ Real.sin (2 * α) * r = Real.sin (4 * α)

theorem solve_trig_problem (α : ℝ) (h : trig_problem α) : α = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_trig_problem_l1419_141940


namespace NUMINAMATH_GPT_equation_holds_if_a_eq_neg_b_c_l1419_141907

-- Define the conditions and equation
variables {a b c : ℝ} (h1 : a ≠ 0) (h2 : a + b ≠ 0)

-- Statement to be proved
theorem equation_holds_if_a_eq_neg_b_c : 
  (a = -(b + c)) ↔ (a + b + c) / a = (b + c) / (a + b) := 
sorry

end NUMINAMATH_GPT_equation_holds_if_a_eq_neg_b_c_l1419_141907


namespace NUMINAMATH_GPT_sum_of_roots_eq_a_plus_b_l1419_141933

theorem sum_of_roots_eq_a_plus_b (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 - (a + b) * x + (ab + 1) = 0 → (x = a ∨ x = b)) :
  a + b = a + b :=
by sorry

end NUMINAMATH_GPT_sum_of_roots_eq_a_plus_b_l1419_141933


namespace NUMINAMATH_GPT_no_good_polygon_in_division_of_equilateral_l1419_141927

def is_equilateral_polygon (P : List Point) : Prop :=
  -- Definition of equilateral polygon
  sorry

def is_good_polygon (P : List Point) : Prop :=
  -- Definition of good polygon (having a pair of parallel sides)
  sorry

def is_divided_by_non_intersecting_diagonals (P : List Point) (polygons : List (List Point)) : Prop :=
  -- Definition for dividing by non-intersecting diagonals into several polygons
  sorry

def have_same_odd_sides (polygons : List (List Point)) : Prop :=
  -- Definition for all polygons having the same odd number of sides
  sorry

theorem no_good_polygon_in_division_of_equilateral (P : List Point) (polygons : List (List Point)) :
  is_equilateral_polygon P →
  is_divided_by_non_intersecting_diagonals P polygons →
  have_same_odd_sides polygons →
  ¬ ∃ gp ∈ polygons, is_good_polygon gp :=
by
  intro h_eq h_div h_odd
  intro h_good
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_no_good_polygon_in_division_of_equilateral_l1419_141927


namespace NUMINAMATH_GPT_triangle_angle_measure_l1419_141901

/-- Proving the measure of angle x in a defined triangle -/
theorem triangle_angle_measure (A B C x : ℝ) (hA : A = 85) (hB : B = 35) (hC : C = 30) : x = 150 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_measure_l1419_141901


namespace NUMINAMATH_GPT_min_value_A2_minus_B2_l1419_141937

noncomputable def A (p q r : ℝ) : ℝ := 
  Real.sqrt (p + 3) + Real.sqrt (q + 6) + Real.sqrt (r + 12)

noncomputable def B (p q r : ℝ) : ℝ :=
  Real.sqrt (p + 2) + Real.sqrt (q + 2) + Real.sqrt (r + 2)

theorem min_value_A2_minus_B2
  (h₁ : 0 ≤ p)
  (h₂ : 0 ≤ q)
  (h₃ : 0 ≤ r) :
  ∃ (p q r : ℝ), A p q r ^ 2 - B p q r ^ 2 = 35 + 10 * Real.sqrt 10 := 
sorry

end NUMINAMATH_GPT_min_value_A2_minus_B2_l1419_141937


namespace NUMINAMATH_GPT_ln_of_gt_of_pos_l1419_141996

variable {a b : ℝ}

theorem ln_of_gt_of_pos (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b :=
sorry

end NUMINAMATH_GPT_ln_of_gt_of_pos_l1419_141996


namespace NUMINAMATH_GPT_zoe_earnings_from_zachary_l1419_141987

noncomputable def babysitting_earnings 
  (total_earnings : ℕ) (pool_cleaning_earnings : ℕ) (earnings_julie_ratio : ℕ) 
  (earnings_chloe_ratio : ℕ) 
  (earnings_zachary : ℕ) : Prop := 
total_earnings = 8000 ∧ 
pool_cleaning_earnings = 2600 ∧ 
earnings_julie_ratio = 3 ∧ 
earnings_chloe_ratio = 5 ∧ 
9 * earnings_zachary = 5400

theorem zoe_earnings_from_zachary : babysitting_earnings 8000 2600 3 5 600 :=
by 
  unfold babysitting_earnings
  sorry

end NUMINAMATH_GPT_zoe_earnings_from_zachary_l1419_141987


namespace NUMINAMATH_GPT_correct_choice_l1419_141983

def PropA : Prop := ∀ x : ℝ, x^2 + 3 < 0
def PropB : Prop := ∀ x : ℕ, x^2 ≥ 1
def PropC : Prop := ∃ x : ℤ, x^5 < 1
def PropD : Prop := ∃ x : ℚ, x^2 = 3

theorem correct_choice : ¬PropA ∧ ¬PropB ∧ PropC ∧ ¬PropD := by
  sorry

end NUMINAMATH_GPT_correct_choice_l1419_141983


namespace NUMINAMATH_GPT_incorrect_ratio_implies_l1419_141917

variable {a b c d : ℝ} (h : a * d = b * c) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

theorem incorrect_ratio_implies :
  ¬ (c / b = a / d) :=
sorry

end NUMINAMATH_GPT_incorrect_ratio_implies_l1419_141917


namespace NUMINAMATH_GPT_smallest_positive_integer_form_3003_55555_l1419_141999

theorem smallest_positive_integer_form_3003_55555 :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 57 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_positive_integer_form_3003_55555_l1419_141999


namespace NUMINAMATH_GPT_maximize_cubic_quartic_l1419_141992

theorem maximize_cubic_quartic (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + 2 * y = 35) : 
  (x, y) = (21, 7) ↔ x^3 * y^4 = (21:ℝ)^3 * (7:ℝ)^4 := 
by
  sorry

end NUMINAMATH_GPT_maximize_cubic_quartic_l1419_141992


namespace NUMINAMATH_GPT_range_of_a_l1419_141965

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^3 - a * x^2 - 4 * a * x + 4 * a^2 - 1 = 0 ∧ ∀ y : ℝ, 
  (y ≠ x → y^3 - a * y^2 - 4 * a * y + 4 * a^2 - 1 ≠ 0)) ↔ a < 3 / 4 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1419_141965


namespace NUMINAMATH_GPT_park_area_l1419_141912

theorem park_area (l w : ℝ) (h1 : 2 * l + 2 * w = 80) (h2 : l = 3 * w) : l * w = 300 :=
sorry

end NUMINAMATH_GPT_park_area_l1419_141912


namespace NUMINAMATH_GPT_min_value_of_expression_l1419_141991

theorem min_value_of_expression (a b : ℝ) (h1 : 1 < a) (h2 : 0 < b) (h3 : a + 2 * b = 2) : 
  4 * (1 + Real.sqrt 2) ≤ (2 / (a - 1) + a / b) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1419_141991


namespace NUMINAMATH_GPT_tangent_to_parabola_l1419_141918

theorem tangent_to_parabola {k : ℝ} : 
  (∀ x y : ℝ, (4 * x + 3 * y + k = 0) ↔ (y ^ 2 = 16 * x)) → k = 9 :=
by
  sorry

end NUMINAMATH_GPT_tangent_to_parabola_l1419_141918


namespace NUMINAMATH_GPT_sum_powers_of_ab_l1419_141986

theorem sum_powers_of_ab (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1)
  (h3 : a^2 + b^2 = 7) (h4 : a^3 + b^3 = 18) (h5 : a^4 + b^4 = 47) :
  a^5 + b^5 = 123 :=
sorry

end NUMINAMATH_GPT_sum_powers_of_ab_l1419_141986


namespace NUMINAMATH_GPT_length_PZ_l1419_141995

-- Define the given conditions
variables (CD WX : ℝ) -- segments CD and WX
variable (CW : ℝ) -- length of segment CW
variable (DP : ℝ) -- length of segment DP
variable (PX : ℝ) -- length of segment PX

-- Define the similarity condition
-- segment CD is parallel to segment WX implies that the triangles CDP and WXP are similar

-- Define what we want to prove
theorem length_PZ (hCD_WX_parallel : CD = WX)
                  (hCW : CW = 56)
                  (hDP : DP = 18)
                  (hPX : PX = 36) :
  ∃ PZ : ℝ, PZ = 4 / 3 :=
by
  -- proof steps here (omitted)
  sorry

end NUMINAMATH_GPT_length_PZ_l1419_141995


namespace NUMINAMATH_GPT_domain_of_function_correct_l1419_141988

noncomputable def domain_of_function (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (2 - x > 0) ∧ (Real.logb 10 (2 - x) ≠ 0)

theorem domain_of_function_correct :
  {x : ℝ | domain_of_function x} = {x : ℝ | x ∈ Set.Icc (-1 : ℝ) 1 \ {1}} ∪ {x : ℝ | x ∈ Set.Ioc 1 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_correct_l1419_141988


namespace NUMINAMATH_GPT_markup_percentage_l1419_141902

theorem markup_percentage (PP SP SaleP : ℝ) (M : ℝ) (hPP : PP = 60) (h1 : SP = 60 + M * SP)
  (h2 : SaleP = SP * 0.8) (h3 : 4 = SaleP - PP) : M = 0.25 :=
by 
  sorry

end NUMINAMATH_GPT_markup_percentage_l1419_141902


namespace NUMINAMATH_GPT_problem_l1419_141963

theorem problem (a b c : ℝ) (h1 : a > b) (h2 : (a - b) * (b - c) * (c - a) > 0) : a > c :=
sorry

end NUMINAMATH_GPT_problem_l1419_141963


namespace NUMINAMATH_GPT_silenos_time_l1419_141959

theorem silenos_time :
  (∃ x : ℝ, ∃ b: ℝ, (x - 2 = x / 2) ∧ (b = x / 3)) → (∃ x : ℝ, x = 3) :=
by sorry

end NUMINAMATH_GPT_silenos_time_l1419_141959


namespace NUMINAMATH_GPT_min_value_of_c_l1419_141951

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m^2

noncomputable def isPerfectCube (x : ℕ) : Prop :=
  ∃ n : ℕ, x = n^3

theorem min_value_of_c (c : ℕ) :
  (∃ a b d e : ℕ, a = c-2 ∧ b = c-1 ∧ d = c+1 ∧ e = c+2 ∧ a < b ∧ b < c ∧ c < d ∧ d < e) ∧
  isPerfectSquare (3 * c) ∧
  isPerfectCube (5 * c) →
  c = 675 :=
sorry

end NUMINAMATH_GPT_min_value_of_c_l1419_141951


namespace NUMINAMATH_GPT_find_m_value_l1419_141934

theorem find_m_value :
  ∃ m : ℤ, 3 * 2^2000 - 5 * 2^1999 + 4 * 2^1998 - 2^1997 = m * 2^1997 ∧ m = 11 :=
by
  -- The proof would follow here.
  sorry

end NUMINAMATH_GPT_find_m_value_l1419_141934


namespace NUMINAMATH_GPT_area_first_side_l1419_141928

-- Define dimensions of the box
variables (L W H : ℝ)

-- Define conditions
def area_WH : Prop := W * H = 72
def area_LH : Prop := L * H = 60
def volume_box : Prop := L * W * H = 720

-- Prove the area of the first side
theorem area_first_side (h1 : area_WH W H) (h2 : area_LH L H) (h3 : volume_box L W H) : L * W = 120 :=
by sorry

end NUMINAMATH_GPT_area_first_side_l1419_141928


namespace NUMINAMATH_GPT_find_f_of_2_l1419_141924

noncomputable def f (x : ℝ) : ℝ := 
if x < 0 then x^3 + x^2 else 0

theorem find_f_of_2 :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, x < 0 → f x = x^3 + x^2) → f 2 = 4 :=
by
  intros h_odd h_def_neg
  sorry

end NUMINAMATH_GPT_find_f_of_2_l1419_141924


namespace NUMINAMATH_GPT_total_missing_keys_l1419_141989

theorem total_missing_keys :
  let total_vowels := 5
  let total_consonants := 21
  let missing_consonants := total_consonants / 7
  let missing_vowels := 2
  missing_consonants + missing_vowels = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_missing_keys_l1419_141989


namespace NUMINAMATH_GPT_difference_of_squares_l1419_141971

theorem difference_of_squares (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 10) : 
  x^2 - y^2 = 200 := 
sorry

end NUMINAMATH_GPT_difference_of_squares_l1419_141971


namespace NUMINAMATH_GPT_balloon_height_per_ounce_l1419_141985

theorem balloon_height_per_ounce
    (total_money : ℕ)
    (sheet_cost : ℕ)
    (rope_cost : ℕ)
    (propane_cost : ℕ)
    (helium_price : ℕ)
    (max_height : ℕ)
    :
    total_money = 200 →
    sheet_cost = 42 →
    rope_cost = 18 →
    propane_cost = 14 →
    helium_price = 150 →
    max_height = 9492 →
    max_height / ((total_money - (sheet_cost + rope_cost + propane_cost)) / helium_price) = 113 :=
by
  intros
  sorry

end NUMINAMATH_GPT_balloon_height_per_ounce_l1419_141985


namespace NUMINAMATH_GPT_campers_went_rowing_and_hiking_in_all_l1419_141972

def C_rm : Nat := 41
def C_hm : Nat := 4
def C_ra : Nat := 26

theorem campers_went_rowing_and_hiking_in_all : (C_rm + C_ra) + C_hm = 71 :=
by
  sorry

end NUMINAMATH_GPT_campers_went_rowing_and_hiking_in_all_l1419_141972


namespace NUMINAMATH_GPT_resultant_force_correct_l1419_141939

-- Define the conditions
def P1 : ℝ := 80
def P2 : ℝ := 130
def distance : ℝ := 12.035
def theta1 : ℝ := 125
def theta2 : ℝ := 135.1939

-- Calculate the correct answer
def result_magnitude : ℝ := 209.299
def result_direction : ℝ := 131.35

-- The goal statement to be proved
theorem resultant_force_correct :
  ∃ (R : ℝ) (theta_R : ℝ), 
    R = result_magnitude ∧ theta_R = result_direction := 
sorry

end NUMINAMATH_GPT_resultant_force_correct_l1419_141939


namespace NUMINAMATH_GPT_chloe_treasures_first_level_l1419_141953

def chloe_treasures_score (T : ℕ) (score_per_treasure : ℕ) (treasures_second_level : ℕ) (total_score : ℕ) :=
  T * score_per_treasure + treasures_second_level * score_per_treasure = total_score

theorem chloe_treasures_first_level :
  chloe_treasures_score T 9 3 81 → T = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_chloe_treasures_first_level_l1419_141953


namespace NUMINAMATH_GPT_pentagon_PTRSQ_area_proof_l1419_141921

-- Define the geometric setup and properties
def quadrilateral_PQRS_is_square (P Q R S T : Type) : Prop :=
  -- Here, we will skip the precise geometric construction and assume the properties directly.
  sorry

def segment_PT_perpendicular_to_TR (P T R : Type) : Prop :=
  sorry

def PT_eq_5 (PT : ℝ) : Prop :=
  PT = 5

def TR_eq_12 (TR : ℝ) : Prop :=
  TR = 12

def area_PTRSQ (area : ℝ) : Prop :=
  area = 139

theorem pentagon_PTRSQ_area_proof
  (P Q R S T : Type)
  (PQRS_is_square : quadrilateral_PQRS_is_square P Q R S T)
  (PT_perpendicular_TR : segment_PT_perpendicular_to_TR P T R)
  (PT_length : PT_eq_5 5)
  (TR_length : TR_eq_12 12)
  : area_PTRSQ 139 :=
  sorry

end NUMINAMATH_GPT_pentagon_PTRSQ_area_proof_l1419_141921


namespace NUMINAMATH_GPT_line_circle_intersect_a_le_0_l1419_141956

theorem line_circle_intersect_a_le_0 :
  (∃ (x y : ℝ), x + a * y + 2 = 0 ∧ x^2 + y^2 + 2 * x - 2 * y + 1 = 0) →
  a ≤ 0 :=
sorry

end NUMINAMATH_GPT_line_circle_intersect_a_le_0_l1419_141956


namespace NUMINAMATH_GPT_coordinates_of_vertex_B_equation_of_line_BC_l1419_141947

noncomputable def vertex_A : (ℝ × ℝ) := (5, 1)
def bisector_expr (x y : ℝ) : Prop := x + y - 5 = 0
def median_CM_expr (x y : ℝ) : Prop := 2 * x - y - 5 = 0

theorem coordinates_of_vertex_B (B : ℝ × ℝ) 
  (h1 : ∃ x y, bisector_expr x y ∧ median_CM_expr x y) :
  B = (2, 3) :=
sorry

theorem equation_of_line_BC (coeff_3x coeff_2y const : ℝ) 
  (h1 : ∃ x y, bisector_expr x y ∧ median_CM_expr x y) :
  coeff_3x = 3 ∧ coeff_2y = 2 ∧ const = -12 :=
sorry

end NUMINAMATH_GPT_coordinates_of_vertex_B_equation_of_line_BC_l1419_141947


namespace NUMINAMATH_GPT_fraction_of_cats_l1419_141954

theorem fraction_of_cats (C D : ℕ) 
  (h1 : C + D = 300)
  (h2 : 4 * D = 400) : 
  (C : ℚ) / (C + D) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_cats_l1419_141954


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1419_141990

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x = 1 → x^2 - 3 * x + 2 = 0) ∧ (∃ x : ℝ, x^2 - 3 * x + 2 = 0 ∧ x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1419_141990


namespace NUMINAMATH_GPT_number_of_tables_l1419_141914

-- Define the conditions
def seats_per_table : ℕ := 8
def total_seating_capacity : ℕ := 32

-- Define the main statement using the conditions
theorem number_of_tables : total_seating_capacity / seats_per_table = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_tables_l1419_141914


namespace NUMINAMATH_GPT_remaining_blocks_to_walk_l1419_141967

noncomputable def total_blocks : ℕ := 11 + 6 + 8
noncomputable def walked_blocks : ℕ := 5

theorem remaining_blocks_to_walk : total_blocks - walked_blocks = 20 := by
  sorry

end NUMINAMATH_GPT_remaining_blocks_to_walk_l1419_141967


namespace NUMINAMATH_GPT_intersection_complement_l1419_141948

-- Definitions
def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 2}

-- Statement to prove
theorem intersection_complement :
  (((I \ B) ∩ A : Set ℕ) = {3, 5}) :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1419_141948


namespace NUMINAMATH_GPT_rectangle_perimeter_l1419_141998

variables (L B P : ℝ)

theorem rectangle_perimeter (h1 : B = 0.60 * L) (h2 : L * B = 37500) : P = 800 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1419_141998


namespace NUMINAMATH_GPT_complement_U_A_l1419_141935

open Set

def U : Set ℝ := {x | -3 < x ∧ x < 3}
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

theorem complement_U_A : 
  (U \ A) = {x | -3 < x ∧ x ≤ -2} ∪ {x | 1 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_U_A_l1419_141935


namespace NUMINAMATH_GPT_fraction_of_quarters_from_1860_to_1869_l1419_141975

theorem fraction_of_quarters_from_1860_to_1869
  (total_quarters : ℕ) (quarters_from_1860s : ℕ)
  (h1 : total_quarters = 30) (h2 : quarters_from_1860s = 15) :
  (quarters_from_1860s : ℚ) / (total_quarters : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_fraction_of_quarters_from_1860_to_1869_l1419_141975


namespace NUMINAMATH_GPT_domain_v_l1419_141981

noncomputable def v (x : ℝ) : ℝ := 1 / (Real.sqrt x + x - 1)

theorem domain_v :
  {x : ℝ | x >= 0 ∧ Real.sqrt x + x - 1 ≠ 0} = {x : ℝ | x ∈ Set.Ico 0 (Real.sqrt 5 - 1) ∪ Set.Ioi (Real.sqrt 5 - 1)} :=
by
  sorry

end NUMINAMATH_GPT_domain_v_l1419_141981


namespace NUMINAMATH_GPT_problem_statement_l1419_141919

variable {x : Real}
variable {m : Int}
variable {n : Int}

theorem problem_statement (h1 : x^m = 5) (h2 : x^n = 10) : x^(2 * m - n) = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1419_141919


namespace NUMINAMATH_GPT_hours_learning_english_each_day_l1419_141984

theorem hours_learning_english_each_day (total_hours : ℕ) (days : ℕ) (learning_hours_per_day : ℕ) 
  (h1 : total_hours = 12) 
  (h2 : days = 2) 
  (h3 : total_hours = learning_hours_per_day * days) : 
  learning_hours_per_day = 6 := 
by
  sorry

end NUMINAMATH_GPT_hours_learning_english_each_day_l1419_141984


namespace NUMINAMATH_GPT_overall_percentage_increase_correct_l1419_141976

def initial_salary : ℕ := 60
def first_raise_salary : ℕ := 90
def second_raise_salary : ℕ := 120
def gym_deduction : ℕ := 10

def final_salary : ℕ := second_raise_salary - gym_deduction
def salary_difference : ℕ := final_salary - initial_salary
def percentage_increase : ℚ := (salary_difference : ℚ) / initial_salary * 100

theorem overall_percentage_increase_correct :
  percentage_increase = 83.33 := by
  sorry

end NUMINAMATH_GPT_overall_percentage_increase_correct_l1419_141976


namespace NUMINAMATH_GPT_race_time_l1419_141930

theorem race_time (v_A v_B : ℝ) (t_A t_B : ℝ) (h1 : v_A = 1000 / t_A) (h2 : v_B = 952 / (t_A + 6)) (h3 : v_A = v_B) : t_A = 125 :=
by
  sorry

end NUMINAMATH_GPT_race_time_l1419_141930


namespace NUMINAMATH_GPT_simplify_fraction_l1419_141960

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1419_141960


namespace NUMINAMATH_GPT_parabola_increasing_implies_a_lt_zero_l1419_141946

theorem parabola_increasing_implies_a_lt_zero (a : ℝ) :
  (∀ x : ℝ, x < 0 → a * (2 * x) > 0) → a < 0 :=
by
  sorry

end NUMINAMATH_GPT_parabola_increasing_implies_a_lt_zero_l1419_141946


namespace NUMINAMATH_GPT_geometric_sequence_ab_product_l1419_141911

theorem geometric_sequence_ab_product (a b : ℝ) (h₁ : 2 ≤ a) (h₂ : a ≤ 16) (h₃ : 2 ≤ b) (h₄ : b ≤ 16)
  (h₅ : ∃ r : ℝ, a = 2 * r ∧ b = 2 * r^2 ∧ 16 = 2 * r^3) : a * b = 32 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ab_product_l1419_141911


namespace NUMINAMATH_GPT_no_nonzero_solutions_l1419_141909

theorem no_nonzero_solutions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + x = y^2 - y) ∧ (y^2 + y = z^2 - z) ∧ (z^2 + z = x^2 - x) → false :=
by
  sorry

end NUMINAMATH_GPT_no_nonzero_solutions_l1419_141909


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_value_l1419_141900

theorem arithmetic_sequence_a3_value {a : ℕ → ℕ}
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) :
  a 3 = 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a3_value_l1419_141900


namespace NUMINAMATH_GPT_derivative_at_1_of_f_l1419_141906

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_at_1_of_f :
  (deriv f 1) = 2 * Real.log 2 - 3 :=
sorry

end NUMINAMATH_GPT_derivative_at_1_of_f_l1419_141906


namespace NUMINAMATH_GPT_min_nS_n_l1419_141913

open Function

noncomputable def a (n : ℕ) (a_1 : ℤ) (d : ℤ) : ℤ := a_1 + (n - 1) * d

noncomputable def S (n : ℕ) (a_1 : ℤ) (d : ℤ) : ℤ := n * a_1 + d * n * (n - 1) / 2

theorem min_nS_n (d : ℤ) (h_a7 : ∃ a_1 : ℤ, a 7 a_1 d = 5)
  (h_S5 : ∃ a_1 : ℤ, S 5 a_1 d = -55) :
  ∃ n : ℕ, n > 0 ∧ n * S n a_1 d = -343 :=
by
  sorry

end NUMINAMATH_GPT_min_nS_n_l1419_141913


namespace NUMINAMATH_GPT_S6_eq_24_l1419_141931

-- Definitions based on the conditions provided
def is_arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)

def S : ℕ → ℝ := sorry  -- Sum of the first n terms of some arithmetic sequence

-- Given conditions
axiom S2_eq_2 : S 2 = 2
axiom S4_eq_10 : S 4 = 10

-- The main theorem to prove
theorem S6_eq_24 : S 6 = 24 :=
by 
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_S6_eq_24_l1419_141931


namespace NUMINAMATH_GPT_simplify_expression_l1419_141980

theorem simplify_expression : 9 * (12 / 7) * ((-35) / 36) = -15 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1419_141980


namespace NUMINAMATH_GPT_fraction_of_quarters_in_1790s_l1419_141962

theorem fraction_of_quarters_in_1790s (total_coins : ℕ) (coins_in_1790s : ℕ) :
  total_coins = 30 ∧ coins_in_1790s = 7 → 
  (coins_in_1790s : ℚ) / total_coins = 7 / 30 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_quarters_in_1790s_l1419_141962


namespace NUMINAMATH_GPT_A_investment_l1419_141982

theorem A_investment (B_invest C_invest Total_profit A_share : ℝ) 
  (h1 : B_invest = 4200)
  (h2 : C_invest = 10500)
  (h3 : Total_profit = 12100)
  (h4 : A_share = 3630) 
  (h5 : ∀ {x : ℝ}, A_share / Total_profit = x / (x + B_invest + C_invest)) :
  ∃ A_invest : ℝ, A_invest = 6300 :=
by sorry

end NUMINAMATH_GPT_A_investment_l1419_141982


namespace NUMINAMATH_GPT_range_of_k_for_ellipse_l1419_141903

theorem range_of_k_for_ellipse (k : ℝ) :
  (4 - k > 0) ∧ (k - 1 > 0) ∧ (4 - k ≠ k - 1) ↔ (1 < k ∧ k < 4 ∧ k ≠ 5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_for_ellipse_l1419_141903


namespace NUMINAMATH_GPT_Razorback_total_revenue_l1419_141936

def t_shirt_price : ℕ := 51
def t_shirt_discount : ℕ := 8
def hat_price : ℕ := 28
def hat_discount : ℕ := 5
def t_shirts_sold : ℕ := 130
def hats_sold : ℕ := 85

def discounted_t_shirt_price : ℕ := t_shirt_price - t_shirt_discount
def discounted_hat_price : ℕ := hat_price - hat_discount

def revenue_from_t_shirts : ℕ := t_shirts_sold * discounted_t_shirt_price
def revenue_from_hats : ℕ := hats_sold * discounted_hat_price

def total_revenue : ℕ := revenue_from_t_shirts + revenue_from_hats

theorem Razorback_total_revenue : total_revenue = 7545 := by
  unfold total_revenue
  unfold revenue_from_t_shirts
  unfold revenue_from_hats
  unfold discounted_t_shirt_price
  unfold discounted_hat_price
  unfold t_shirts_sold
  unfold hats_sold
  unfold t_shirt_price
  unfold t_shirt_discount
  unfold hat_price
  unfold hat_discount
  sorry

end NUMINAMATH_GPT_Razorback_total_revenue_l1419_141936
