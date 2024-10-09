import Mathlib

namespace work_done_in_one_day_l2306_230626

theorem work_done_in_one_day (A_time B_time : ℕ) (hA : A_time = 4) (hB : B_time = A_time / 2) : 
  (1 / A_time + 1 / B_time) = (3 / 4) :=
by
  -- Here we are setting up the conditions as per our identified steps
  rw [hA, hB]
  -- The remaining steps to prove will be omitted as per instructions
  sorry

end work_done_in_one_day_l2306_230626


namespace verify_equation_l2306_230616

theorem verify_equation : (3^2 + 5^2)^2 = 16^2 + 30^2 := by
  sorry

end verify_equation_l2306_230616


namespace amy_created_albums_l2306_230624

theorem amy_created_albums (total_photos : ℕ) (photos_per_album : ℕ) 
  (h1 : total_photos = 180)
  (h2 : photos_per_album = 20) : 
  (total_photos / photos_per_album = 9) :=
by
  sorry

end amy_created_albums_l2306_230624


namespace sum_of_positive_integers_eq_32_l2306_230662

noncomputable def sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : ℕ :=
  x + y

theorem sum_of_positive_integers_eq_32 (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : sum_of_integers x y h1 h2 = 32 :=
  sorry

end sum_of_positive_integers_eq_32_l2306_230662


namespace selling_price_with_increase_l2306_230636

variable (a : ℝ)

theorem selling_price_with_increase (h : a > 0) : 1.1 * a = a + 0.1 * a := by
  -- Here you will add the proof, which we skip with sorry
  sorry

end selling_price_with_increase_l2306_230636


namespace quadratic_distinct_real_roots_range_l2306_230695

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ ∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) ↔ (k > -1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_distinct_real_roots_range_l2306_230695


namespace spencer_total_distance_l2306_230604

def d1 : ℝ := 1.2
def d2 : ℝ := 0.6
def d3 : ℝ := 0.9
def d4 : ℝ := 1.7
def d5 : ℝ := 2.1
def d6 : ℝ := 1.3
def d7 : ℝ := 0.8

theorem spencer_total_distance : d1 + d2 + d3 + d4 + d5 + d6 + d7 = 8.6 :=
by
  sorry

end spencer_total_distance_l2306_230604


namespace div_powers_same_base_l2306_230697

variable (x : ℝ)

theorem div_powers_same_base : x^8 / x^2 = x^6 :=
by
  sorry

end div_powers_same_base_l2306_230697


namespace prove_optionC_is_suitable_l2306_230666

def OptionA := "Understanding the height of students in Class 7(1)"
def OptionB := "Companies recruiting and interviewing job applicants"
def OptionC := "Investigating the impact resistance of a batch of cars"
def OptionD := "Selecting the fastest runner in our school to participate in the city-wide competition"

def is_suitable_for_sampling_survey (option : String) : Prop :=
  option = OptionC

theorem prove_optionC_is_suitable :
  is_suitable_for_sampling_survey OptionC :=
by
  sorry

end prove_optionC_is_suitable_l2306_230666


namespace parallel_lines_l2306_230652

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = a - 7) → a = 3 :=
by sorry

end parallel_lines_l2306_230652


namespace xy_product_l2306_230611

theorem xy_product (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end xy_product_l2306_230611


namespace sequence_problem_l2306_230696

theorem sequence_problem 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 5) 
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 3 + 4 * (n - 1)) : 
  a 50 = 4856 :=
sorry

end sequence_problem_l2306_230696


namespace paper_holes_symmetric_l2306_230619

-- Define the initial conditions
def folded_paper : Type := sorry -- Specific structure to represent the paper and its folds

def paper_fold_bottom_to_top (paper : folded_paper) : folded_paper := sorry
def paper_fold_right_half_to_left (paper : folded_paper) : folded_paper := sorry
def paper_fold_diagonal (paper : folded_paper) : folded_paper := sorry

-- Define a function that represents punching a hole near the folded edge
def punch_hole_near_folded_edge (paper : folded_paper) : folded_paper := sorry

-- Initial paper
def initial_paper : folded_paper := sorry

-- Folded and punched paper
def paper_after_folds_and_punch : folded_paper :=
  punch_hole_near_folded_edge (
    paper_fold_diagonal (
      paper_fold_right_half_to_left (
        paper_fold_bottom_to_top initial_paper)))

-- Unfolding the paper
def unfold_diagonal (paper : folded_paper) : folded_paper := sorry
def unfold_right_half (paper : folded_paper) : folded_paper := sorry
def unfold_bottom_to_top (paper : folded_paper) : folded_paper := sorry

def paper_after_unfolding : folded_paper :=
  unfold_bottom_to_top (
    unfold_right_half (
      unfold_diagonal paper_after_folds_and_punch))

-- Definition of hole pattern 'eight_symmetric_holes'
def eight_symmetric_holes (paper : folded_paper) : Prop := sorry

-- The proof problem
theorem paper_holes_symmetric :
  eight_symmetric_holes paper_after_unfolding := sorry

end paper_holes_symmetric_l2306_230619


namespace checkered_board_cut_l2306_230623

def can_cut_equal_squares (n : ℕ) : Prop :=
  n % 5 = 0 ∧ n > 5

theorem checkered_board_cut (n : ℕ) (h : n % 5 = 0 ∧ n > 5) :
  ∃ m, n^2 = 5 * m :=
by
  sorry

end checkered_board_cut_l2306_230623


namespace range_of_f_l2306_230668

noncomputable def f (x : ℝ) : ℝ := 4^x + 2^(x + 1) + 1

theorem range_of_f : Set.range f = {y : ℝ | y > 1} :=
by
  sorry

end range_of_f_l2306_230668


namespace sets_of_earrings_l2306_230615

namespace EarringsProblem

variables (magnets buttons gemstones earrings : ℕ)

theorem sets_of_earrings (h1 : gemstones = 24)
                         (h2 : gemstones = 3 * buttons)
                         (h3 : buttons = magnets / 2)
                         (h4 : earrings = magnets / 2)
                         (h5 : ∀ n : ℕ, n % 2 = 0 → ∃ k, n = 2 * k) :
  earrings = 8 :=
by
  sorry

end EarringsProblem

end sets_of_earrings_l2306_230615


namespace surface_area_hemisphere_l2306_230625

theorem surface_area_hemisphere
  (r : ℝ)
  (h₁ : 4 * Real.pi * r^2 = 4 * Real.pi * r^2)
  (h₂ : Real.pi * r^2 = 3) :
  3 * Real.pi * r^2 = 9 :=
by
  sorry

end surface_area_hemisphere_l2306_230625


namespace circle_area_difference_l2306_230656

noncomputable def difference_of_circle_areas (C1 C2 : ℝ) : ℝ :=
  let π := Real.pi
  let r1 := C1 / (2 * π)
  let r2 := C2 / (2 * π)
  let A1 := π * r1 ^ 2
  let A2 := π * r2 ^ 2
  A2 - A1

theorem circle_area_difference :
  difference_of_circle_areas 396 704 = 26948.4 :=
by
  sorry

end circle_area_difference_l2306_230656


namespace no5_battery_mass_l2306_230646

theorem no5_battery_mass :
  ∃ (x y : ℝ), 2 * x + 2 * y = 72 ∧ 3 * x + 2 * y = 96 ∧ x = 24 :=
by
  sorry

end no5_battery_mass_l2306_230646


namespace sarahs_brother_apples_l2306_230609

theorem sarahs_brother_apples (x : ℝ) (hx : 5 * x = 45.0) : x = 9.0 :=
by
  sorry

end sarahs_brother_apples_l2306_230609


namespace teams_dig_tunnel_in_10_days_l2306_230677

theorem teams_dig_tunnel_in_10_days (hA : ℝ) (hB : ℝ) (work_A : hA = 15) (work_B : hB = 30) : 
  (1 / (1 / hA + 1 / hB)) = 10 := 
by
  sorry

end teams_dig_tunnel_in_10_days_l2306_230677


namespace total_time_to_fill_tank_l2306_230657

noncomputable def pipe_filling_time : ℕ := 
  let tank_capacity := 2000
  let pipe_a_rate := 200
  let pipe_b_rate := 50
  let pipe_c_rate := 25
  let cycle_duration := 5
  let cycle_fill := (pipe_a_rate * 1 + pipe_b_rate * 2 - pipe_c_rate * 2)
  let num_cycles := tank_capacity / cycle_fill
  num_cycles * cycle_duration

theorem total_time_to_fill_tank : pipe_filling_time = 40 := 
by
  unfold pipe_filling_time
  sorry

end total_time_to_fill_tank_l2306_230657


namespace total_paint_area_l2306_230631

structure Room where
  length : ℕ
  width : ℕ
  height : ℕ

def livingRoom : Room := { length := 40, width := 40, height := 10 }
def bedroom : Room := { length := 12, width := 10, height := 10 }

def wallArea (room : Room) (n_walls : ℕ) : ℕ :=
  let longWallsArea := 2 * (room.length * room.height)
  let shortWallsArea := 2 * (room.width * room.height)
  if n_walls <= 2 then
    longWallsArea * n_walls / 2
  else if n_walls <= 4 then
    longWallsArea + (shortWallsArea * (n_walls - 2) / 2)
  else
    0

def totalWallArea (livingRoom : Room) (bedroom : Room) (n_livingRoomWalls n_bedroomWalls : ℕ) : ℕ :=
  wallArea livingRoom n_livingRoomWalls + wallArea bedroom n_bedroomWalls

theorem total_paint_area : totalWallArea livingRoom bedroom 3 4 = 1640 := by
  sorry

end total_paint_area_l2306_230631


namespace total_bottles_per_day_l2306_230673

def num_cases_per_day : ℕ := 7200
def bottles_per_case : ℕ := 10

theorem total_bottles_per_day : num_cases_per_day * bottles_per_case = 72000 := by
  sorry

end total_bottles_per_day_l2306_230673


namespace triangle_to_square_difference_l2306_230633

noncomputable def number_of_balls_in_triangle (T : ℕ) : ℕ :=
  T * (T + 1) / 2

noncomputable def number_of_balls_in_square (S : ℕ) : ℕ :=
  S * S

theorem triangle_to_square_difference (T S : ℕ) 
  (h1 : number_of_balls_in_triangle T = 1176) 
  (h2 : number_of_balls_in_square S = 1600) :
  T - S = 8 :=
by
  sorry

end triangle_to_square_difference_l2306_230633


namespace product_of_two_numbers_l2306_230637

-- State the conditions and the proof problem
theorem product_of_two_numbers (x y : ℤ) (h_sum : x + y = 30) (h_diff : x - y = 6) :
  x * y = 216 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end product_of_two_numbers_l2306_230637


namespace cos_A_eq_l2306_230674

variable (A : Real) (A_interior_angle_tri_ABC : A > π / 2 ∧ A < π) (tan_A_eq_neg_two : Real.tan A = -2)

theorem cos_A_eq : Real.cos A = - (Real.sqrt 5) / 5 := by
  sorry

end cos_A_eq_l2306_230674


namespace emily_age_l2306_230658

theorem emily_age (A B C D E : ℕ) (h1 : A = B - 4) (h2 : B = C + 5) (h3 : D = C + 2) (h4 : E = A + D - B) (h5 : B = 20) : E = 13 :=
by sorry

end emily_age_l2306_230658


namespace no_form3000001_is_perfect_square_l2306_230608

theorem no_form3000001_is_perfect_square (n : ℕ) : 
  ∀ k : ℤ, (3 * 10^n + 1 ≠ k^2) :=
by
  sorry

end no_form3000001_is_perfect_square_l2306_230608


namespace max_remainder_division_by_9_l2306_230654

theorem max_remainder_division_by_9 : ∀ (r : ℕ), r < 9 → r ≤ 8 :=
by sorry

end max_remainder_division_by_9_l2306_230654


namespace base15_mod_9_l2306_230627

noncomputable def base15_to_decimal : ℕ :=
  2 * 15^3 + 6 * 15^2 + 4 * 15^1 + 3 * 15^0

theorem base15_mod_9 (n : ℕ) (h : n = base15_to_decimal) : n % 9 = 0 :=
sorry

end base15_mod_9_l2306_230627


namespace floor_add_double_eq_15_4_l2306_230664

theorem floor_add_double_eq_15_4 (r : ℝ) (h : (⌊r⌋ : ℝ) + 2 * r = 15.4) : r = 5.2 := 
sorry

end floor_add_double_eq_15_4_l2306_230664


namespace find_matrix_calculate_M5_alpha_l2306_230690

-- Define the matrix M, eigenvalues, eigenvectors and vector α
def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![3, 2]]
def alpha : Fin 2 → ℝ := ![-1, 1]
def e1 : Fin 2 → ℝ := ![2, 3]
def e2 : Fin 2 → ℝ := ![1, -1]
def lambda1 : ℝ := 4
def lambda2 : ℝ := -1

-- Conditions: eigenvalues and their corresponding eigenvectors
axiom h1 : M.mulVec e1 = lambda1 • e1
axiom h2 : M.mulVec e2 = lambda2 • e2

-- Condition: given vector α
axiom h3 : alpha = - e2

-- Prove that M is the matrix given by the components
theorem find_matrix : M = ![![1, 2], ![3, 2]] :=
sorry

-- Prove that M^5 times α equals the given vector
theorem calculate_M5_alpha : (M^5).mulVec alpha = ![-1, 1] :=
sorry

end find_matrix_calculate_M5_alpha_l2306_230690


namespace P_eq_CU_M_union_CU_N_l2306_230610

open Set

-- Definitions of U, M, N
def U : Set (ℝ × ℝ) := { p | True }
def M : Set (ℝ × ℝ) := { p | p.2 ≠ p.1 }
def N : Set (ℝ × ℝ) := { p | p.2 ≠ -p.1 }
def CU_M : Set (ℝ × ℝ) := { p | p.2 = p.1 }
def CU_N : Set (ℝ × ℝ) := { p | p.2 = -p.1 }

-- Theorem statement
theorem P_eq_CU_M_union_CU_N :
  { p : ℝ × ℝ | p.2^2 ≠ p.1^2 } = CU_M ∪ CU_N :=
sorry

end P_eq_CU_M_union_CU_N_l2306_230610


namespace frustum_surface_area_l2306_230659

theorem frustum_surface_area (r r' l : ℝ) (h_r : r = 1) (h_r' : r' = 4) (h_l : l = 5) :
  π * r^2 + π * r'^2 + π * (r + r') * l = 42 * π :=
by
  rw [h_r, h_r', h_l]
  norm_num
  sorry

end frustum_surface_area_l2306_230659


namespace odometer_reading_before_trip_l2306_230639

-- Define the given conditions
def odometer_reading_lunch : ℝ := 372.0
def miles_traveled : ℝ := 159.7

-- Theorem to prove that the odometer reading before the trip was 212.3 miles
theorem odometer_reading_before_trip : odometer_reading_lunch - miles_traveled = 212.3 := by
  sorry

end odometer_reading_before_trip_l2306_230639


namespace range_of_a_l2306_230669

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem range_of_a :
  (∃ (a : ℝ), (a ≤ -2 ∨ a ≥ 0) ∧ (∃ (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4), f x ≤ a^2 + 2 * a)) :=
by sorry

end range_of_a_l2306_230669


namespace minimum_AB_l2306_230689

noncomputable def shortest_AB (a : ℝ) : ℝ :=
  let x := (Real.sqrt 3) / 4 * a
  x

theorem minimum_AB (a : ℝ) : ∃ x, (x = (Real.sqrt 3) / 4 * a) ∧ ∀ y, (y = (Real.sqrt 3) / 4 * a) → shortest_AB a = x :=
by
  sorry

end minimum_AB_l2306_230689


namespace peter_class_students_l2306_230693

def total_students (students_with_two_hands students_with_one_hand students_with_three_hands : ℕ) : ℕ :=
  students_with_two_hands + students_with_one_hand + students_with_three_hands + 1

theorem peter_class_students
  (students_with_two_hands students_with_one_hand students_with_three_hands : ℕ)
  (total_hands_without_peter : ℕ) :

  students_with_two_hands = 10 →
  students_with_one_hand = 3 →
  students_with_three_hands = 1 →
  total_hands_without_peter = 20 →
  total_students students_with_two_hands students_with_one_hand students_with_three_hands = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end peter_class_students_l2306_230693


namespace black_area_after_transformations_l2306_230620

theorem black_area_after_transformations :
  let initial_fraction : ℝ := 1
  let transformation_factor : ℝ := 3 / 4
  let number_of_transformations : ℕ := 5
  let final_fraction : ℝ := transformation_factor ^ number_of_transformations
  final_fraction = 243 / 1024 :=
by
  -- Proof omitted
  sorry

end black_area_after_transformations_l2306_230620


namespace problem_proof_l2306_230628

def problem_statement : Prop :=
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 * (1 / 4096) * 8192 = 64

theorem problem_proof : problem_statement := by
  sorry

end problem_proof_l2306_230628


namespace cricket_runs_l2306_230601

theorem cricket_runs (x a b c d : ℕ) 
    (h1 : a = 1 * x) 
    (h2 : b = 3 * x) 
    (h3 : c = 5 * x) 
    (h4 : d = 4 * x) 
    (total_runs : 1 * x + 3 * x + 5 * x + 4 * x = 234) :
  a = 18 ∧ b = 54 ∧ c = 90 ∧ d = 72 := by
  sorry

end cricket_runs_l2306_230601


namespace represent_2021_as_squares_l2306_230643

theorem represent_2021_as_squares :
  ∃ n : ℕ, n = 505 → 2021 = (n + 1)^2 - (n - 1)^2 + 1^2 :=
by
  sorry

end represent_2021_as_squares_l2306_230643


namespace min_range_of_three_test_takers_l2306_230650

-- Proposition: The minimum possible range in scores of the 3 test-takers
-- where the ranges of their scores in the 5 practice tests are 18, 26, and 32, is 76.
theorem min_range_of_three_test_takers (r1 r2 r3: ℕ) 
  (h1 : r1 = 18) (h2 : r2 = 26) (h3 : r3 = 32) : 
  (r1 + r2 + r3) = 76 := by
  sorry

end min_range_of_three_test_takers_l2306_230650


namespace g_x_squared_minus_3_l2306_230603

theorem g_x_squared_minus_3 (g : ℝ → ℝ)
  (h : ∀ x : ℝ, g (x^2 - 1) = x^4 - 4 * x^2 + 4) :
  ∀ x : ℝ, g (x^2 - 3) = x^4 - 6 * x^2 + 11 :=
by
  sorry

end g_x_squared_minus_3_l2306_230603


namespace negation_of_existence_implies_universal_l2306_230640

theorem negation_of_existence_implies_universal (x : ℝ) :
  (∀ x : ℝ, ¬(x^2 ≤ |x|)) ↔ (∀ x : ℝ, x^2 > |x|) :=
by 
  sorry

end negation_of_existence_implies_universal_l2306_230640


namespace find_quarters_l2306_230676

def num_pennies := 123
def num_nickels := 85
def num_dimes := 35
def cost_per_scoop_cents := 300  -- $3 = 300 cents
def num_family_members := 5
def leftover_cents := 48

def total_cost_cents := num_family_members * cost_per_scoop_cents
def total_initial_cents := total_cost_cents + leftover_cents

-- Values of coins in cents
def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25

def total_pennies_value := num_pennies * penny_value
def total_nickels_value := num_nickels * nickel_value
def total_dimes_value := num_dimes * dime_value
def total_initial_excluding_quarters := total_pennies_value + total_nickels_value + total_dimes_value

def total_quarters_value := total_initial_cents - total_initial_excluding_quarters
def num_quarters := total_quarters_value / quarter_value

theorem find_quarters : num_quarters = 26 := by
  sorry

end find_quarters_l2306_230676


namespace snickers_bars_needed_l2306_230663

-- Definitions of the conditions
def points_needed : ℕ := 2000
def chocolate_bunny_points : ℕ := 100
def number_of_chocolate_bunnies : ℕ := 8
def snickers_points : ℕ := 25

-- Derived conditions
def points_from_bunnies : ℕ := number_of_chocolate_bunnies * chocolate_bunny_points
def remaining_points : ℕ := points_needed - points_from_bunnies

-- Statement to prove
theorem snickers_bars_needed : ∀ (n : ℕ), n = remaining_points / snickers_points → n = 48 :=
by 
  sorry

end snickers_bars_needed_l2306_230663


namespace mixed_doubles_selection_l2306_230670

-- Given conditions
def num_male_players : ℕ := 5
def num_female_players : ℕ := 4

-- The statement to show the number of different ways to select two players is 20
theorem mixed_doubles_selection : (num_male_players * num_female_players) = 20 := by
  -- Proof to be filled in
  sorry

end mixed_doubles_selection_l2306_230670


namespace remainder_7_times_10_pow_20_plus_1_pow_20_mod_9_l2306_230681

theorem remainder_7_times_10_pow_20_plus_1_pow_20_mod_9 :
  (7 * 10 ^ 20 + 1 ^ 20) % 9 = 8 :=
by
  -- need to note down the known conditions to help guide proof writing.
  -- condition: 1 ^ 20 = 1
  -- condition: 10 % 9 = 1

  sorry

end remainder_7_times_10_pow_20_plus_1_pow_20_mod_9_l2306_230681


namespace weight_of_each_bag_l2306_230684

theorem weight_of_each_bag 
  (total_potatoes_weight : ℕ) (damaged_potatoes_weight : ℕ) 
  (bag_price : ℕ) (total_revenue : ℕ) (sellable_potatoes_weight : ℕ) (number_of_bags : ℕ) 
  (weight_of_each_bag : ℕ) :
  total_potatoes_weight = 6500 →
  damaged_potatoes_weight = 150 →
  sellable_potatoes_weight = total_potatoes_weight - damaged_potatoes_weight →
  bag_price = 72 →
  total_revenue = 9144 →
  number_of_bags = total_revenue / bag_price →
  weight_of_each_bag * number_of_bags = sellable_potatoes_weight →
  weight_of_each_bag = 50 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end weight_of_each_bag_l2306_230684


namespace expand_binomials_l2306_230642

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end expand_binomials_l2306_230642


namespace exists_xy_interval_l2306_230618

theorem exists_xy_interval (a b : ℝ) : 
  ∃ (x y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ |x * y - a * x - b * y| ≥ 1 / 3 :=
sorry

end exists_xy_interval_l2306_230618


namespace monotonicity_and_inequality_l2306_230648

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem monotonicity_and_inequality (a : ℝ) (p q : ℝ) (hp : 0 < p ∧ p < 1) (hq : 0 < q ∧ q < 1)
  (h_distinct: p ≠ q) (h_a : a ≥ 10) : 
  (f a (p + 1) - f a (q + 1)) / (p - q) > 1 := by
  sorry

end monotonicity_and_inequality_l2306_230648


namespace simplify_expression_l2306_230614

theorem simplify_expression (b : ℝ) (h1 : b ≠ 1) (h2 : b ≠ 1 / 2) :
  (1 / 2 - 1 / (1 + b / (1 - 2 * b))) = (3 * b - 1) / (2 * (1 - b)) :=
sorry

end simplify_expression_l2306_230614


namespace total_water_needed_l2306_230685

def adults : ℕ := 7
def children : ℕ := 3
def hours : ℕ := 24
def replenish_bottles : ℚ := 14
def water_per_hour_adult : ℚ := 1/2
def water_per_hour_child : ℚ := 1/3

theorem total_water_needed : 
  let total_water_per_hour := (adults * water_per_hour_adult) + (children * water_per_hour_child)
  let total_water := total_water_per_hour * hours 
  let initial_water_needed := total_water - replenish_bottles
  initial_water_needed = 94 := by 
  sorry

end total_water_needed_l2306_230685


namespace times_faster_l2306_230665

theorem times_faster (A B : ℝ) (h1 : A + B = 1 / 12) (h2 : A = 1 / 16) : 
  A / B = 3 :=
by
  sorry

end times_faster_l2306_230665


namespace integral_part_odd_l2306_230638

theorem integral_part_odd (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, (⌊(3 + Real.sqrt 5)^n⌋ = 2 * m + 1) := 
by
  -- Sorry used since the proof steps are not required in the task
  sorry

end integral_part_odd_l2306_230638


namespace miranda_pillows_l2306_230600

-- Define the conditions in the problem
def pounds_per_pillow := 2
def feathers_per_pound := 300
def total_feathers := 3600

-- Define the goal in terms of these conditions
def num_pillows : Nat :=
  (total_feathers / feathers_per_pound) / pounds_per_pillow

-- Prove that the number of pillows Miranda can stuff is 6
theorem miranda_pillows : num_pillows = 6 :=
by
  sorry

end miranda_pillows_l2306_230600


namespace probability_of_event_l2306_230630

theorem probability_of_event (favorable unfavorable : ℕ) (h : favorable = 3) (h2 : unfavorable = 5) :
  (favorable / (favorable + unfavorable) : ℚ) = 3 / 8 :=
by
  sorry

end probability_of_event_l2306_230630


namespace arithmetic_seq_sum_mul_3_l2306_230675

-- Definition of the arithmetic sequence
def arithmetic_sequence := [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121]

-- Prove that 3 times the sum of the arithmetic sequence is 3663
theorem arithmetic_seq_sum_mul_3 : 
  3 * (arithmetic_sequence.sum) = 3663 :=
by
  sorry

end arithmetic_seq_sum_mul_3_l2306_230675


namespace henry_time_proof_l2306_230605

-- Define the time Dawson took to run the first leg of the course
def dawson_time : ℝ := 38

-- Define the average time they took to run a leg of the course
def average_time : ℝ := 22.5

-- Define the time Henry took to run the second leg of the course
def henry_time : ℝ := 7

-- Prove that Henry took 7 seconds to run the second leg
theorem henry_time_proof : 
  (dawson_time + henry_time) / 2 = average_time :=
by
  -- This is where the proof would go
  sorry

end henry_time_proof_l2306_230605


namespace count_odd_expressions_l2306_230688

theorem count_odd_expressions : 
  let exp1 := 1^2
  let exp2 := 2^3
  let exp3 := 3^4
  let exp4 := 4^5
  let exp5 := 5^6
  (if exp1 % 2 = 1 then 1 else 0) + 
  (if exp2 % 2 = 1 then 1 else 0) + 
  (if exp3 % 2 = 1 then 1 else 0) + 
  (if exp4 % 2 = 1 then 1 else 0) + 
  (if exp5 % 2 = 1 then 1 else 0) = 3 :=
by 
  sorry

end count_odd_expressions_l2306_230688


namespace laura_garden_daisies_l2306_230660

/-
Laura's Garden Problem: Given the ratio of daisies to tulips is 3:4,
Laura currently has 32 tulips, and she plans to add 24 more tulips,
prove that Laura will have 42 daisies in total after the addition to
maintain the same ratio.
-/

theorem laura_garden_daisies (daisies tulips add_tulips : ℕ) (ratio_d : ℕ) (ratio_t : ℕ)
    (h1 : ratio_d = 3) (h2 : ratio_t = 4) (h3 : tulips = 32) (h4 : add_tulips = 24)
    (new_tulips : ℕ := tulips + add_tulips) :
  daisies = 42 :=
by
  sorry

end laura_garden_daisies_l2306_230660


namespace minimum_stamps_combination_l2306_230649

theorem minimum_stamps_combination (c f : ℕ) (h : 3 * c + 4 * f = 30) :
  c + f = 8 :=
sorry

end minimum_stamps_combination_l2306_230649


namespace cord_length_before_cut_l2306_230691

-- Definitions based on the conditions
def parts_after_cut := 20
def longest_piece := 8
def shortest_piece := 2
def initial_parts := 19

-- Lean statement to prove the length of the cord before it was cut
theorem cord_length_before_cut : 
  (initial_parts * ((longest_piece / 2) + shortest_piece) = 114) :=
by 
  sorry

end cord_length_before_cut_l2306_230691


namespace graphs_symmetric_about_a_axis_of_symmetry_l2306_230679

def graph_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (x - a)

theorem graphs_symmetric_about_a (f : ℝ → ℝ) (a : ℝ) :
  ∀ x, f (x - a) = f (a - (x - a)) :=
sorry

theorem axis_of_symmetry (f : ℝ → ℝ) :
  (∀ x : ℝ, f (1 + 2 * x) = f (1 - 2 * x)) →
  ∀ x, f x = f (2 - x) := 
sorry

end graphs_symmetric_about_a_axis_of_symmetry_l2306_230679


namespace monotonic_decreasing_interval_l2306_230672

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1 / x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x > 0 → (∀ a b : ℝ, a < b → (f b ≤ f a → b ≤ (1 : ℝ) / 2)) :=
by sorry

end monotonic_decreasing_interval_l2306_230672


namespace original_price_of_cycle_l2306_230617

theorem original_price_of_cycle (SP : ℝ) (gain_percent : ℝ) (original_price : ℝ) 
  (hSP : SP = 1260) (hgain : gain_percent = 0.40) (h_eq : SP = original_price * (1 + gain_percent)) :
  original_price = 900 :=
by
  sorry

end original_price_of_cycle_l2306_230617


namespace probability_grunters_win_all_5_games_l2306_230641

noncomputable def probability_grunters_win_game : ℚ := 4 / 5

theorem probability_grunters_win_all_5_games :
  (probability_grunters_win_game ^ 5) = 1024 / 3125 := 
  by 
  sorry

end probability_grunters_win_all_5_games_l2306_230641


namespace problem_statement_l2306_230606

noncomputable def g (x : ℝ) : ℝ := 3^(x + 1)

theorem problem_statement (x : ℝ) : g (x + 1) - 2 * g x = g x := by
  -- The proof here is omitted
  sorry

end problem_statement_l2306_230606


namespace max_x2_y2_on_circle_l2306_230692

noncomputable def max_value_on_circle : ℝ :=
  12 + 8 * Real.sqrt 2

theorem max_x2_y2_on_circle (x y : ℝ) (h : x^2 - 4 * x - 4 + y^2 = 0) : 
  x^2 + y^2 ≤ max_value_on_circle := 
by
  sorry

end max_x2_y2_on_circle_l2306_230692


namespace sum_of_midpoints_double_l2306_230698

theorem sum_of_midpoints_double (a b c : ℝ) (h : a + b + c = 15) : 
  (a + b) + (a + c) + (b + c) = 30 :=
by
  -- We skip the proof according to the instruction
  sorry

end sum_of_midpoints_double_l2306_230698


namespace partial_fraction_sum_zero_l2306_230682

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, 1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_sum_zero_l2306_230682


namespace backpack_prices_purchasing_plans_backpacks_given_away_l2306_230667

-- Part 1: Prices of Type A and Type B backpacks
theorem backpack_prices (x y : ℝ) (h1 : x = 2 * y - 30) (h2 : 2 * x + 3 * y = 255) : x = 60 ∧ y = 45 :=
sorry

-- Part 2: Possible purchasing plans
theorem purchasing_plans (m : ℕ) (h1 : 8900 ≥ 50 * m + 40 * (200 - m)) (h2 : m > 87) : 
  m = 88 ∨ m = 89 ∨ m = 90 :=
sorry

-- Part 3: Number of backpacks given away
theorem backpacks_given_away (m n : ℕ) (total_A : ℕ := 89) (total_B : ℕ := 111) 
(h1 : m + n = 4) 
(h2 : 1250 = (total_A - if total_A > 10 then total_A / 10 else 0) * 60 + (total_B - if total_B > 10 then total_B / 10 else 0) * 45 - (50 * total_A + 40 * total_B)) :
m = 1 ∧ n = 3 := 
sorry

end backpack_prices_purchasing_plans_backpacks_given_away_l2306_230667


namespace expected_value_of_win_l2306_230651

theorem expected_value_of_win :
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
   prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 :=
by
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  have E : (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
            prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 := sorry
  exact E

end expected_value_of_win_l2306_230651


namespace simplify_expression_l2306_230612

theorem simplify_expression (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5):
  ((x^2 - 4 * x + 3) / (x^2 - 6 * x + 9)) / ((x^2 - 6 * x + 8) / (x^2 - 8 * x + 15)) = 
  (x - 1) * (x - 5) / ((x - 3) * (x - 4) * (x - 2)) :=
sorry

end simplify_expression_l2306_230612


namespace local_minimum_f_is_1_maximum_local_minimum_g_is_1_l2306_230647

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

def local_minimum_value_f := 1

theorem local_minimum_f_is_1 : 
  ∃ x0 : ℝ, x0 > 0 ∧ (∀ x > 0, f x0 ≤ f x) ∧ f x0 = local_minimum_value_f :=
sorry

noncomputable def g (a x : ℝ) : ℝ := f x - a * (x - 1)

def maximum_value_local_minimum_g := 1

theorem maximum_local_minimum_g_is_1 :
  ∃ a x0 : ℝ, a = 0 ∧ x0 > 0 ∧ (∀ x > 0, g a x0 ≤ g a x) ∧ g a x0 = maximum_value_local_minimum_g :=
sorry

end local_minimum_f_is_1_maximum_local_minimum_g_is_1_l2306_230647


namespace pyramid_volume_l2306_230621

theorem pyramid_volume
  (FB AC FA FC AB BC : ℝ)
  (hFB : FB = 12)
  (hAC : AC = 4)
  (hFA : FA = 7)
  (hFC : FC = 7)
  (hAB : AB = 7)
  (hBC : BC = 7) :
  (1/3 * AC * (1/2 * FB * 3)) = 24 := by sorry

end pyramid_volume_l2306_230621


namespace gravitational_force_at_300000_l2306_230655

-- Definitions and premises
def gravitational_force (d : ℝ) : ℝ := sorry

axiom inverse_square_law (d : ℝ) (f : ℝ) (k : ℝ) : f * d^2 = k

axiom surface_force : gravitational_force 5000 = 800

-- Goal: Prove the gravitational force at 300,000 miles
theorem gravitational_force_at_300000 : gravitational_force 300000 = 1 / 45 := sorry

end gravitational_force_at_300000_l2306_230655


namespace part1_proof_part2_proof_l2306_230653

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x - 1| - |x - m|

theorem part1_proof : ∀ x, f x 2 ≥ 1 ↔ x ≥ 2 :=
by 
  sorry

theorem part2_proof : (∀ x : ℝ, f x m ≤ 5) → (-4 ≤ m ∧ m ≤ 6) :=
by
  sorry

end part1_proof_part2_proof_l2306_230653


namespace three_x4_plus_two_x5_l2306_230671

theorem three_x4_plus_two_x5 (x1 x2 x3 x4 x5 : ℤ)
  (h1 : 2 * x1 + x2 + x3 + x4 + x5 = 6)
  (h2 : x1 + 2 * x2 + x3 + x4 + x5 = 12)
  (h3 : x1 + x2 + 2 * x3 + x4 + x5 = 24)
  (h4 : x1 + x2 + x3 + 2 * x4 + x5 = 48)
  (h5 : x1 + x2 + x3 + x4 + 2 * x5 = 96) : 
  3 * x4 + 2 * x5 = 181 := 
sorry

end three_x4_plus_two_x5_l2306_230671


namespace negation_of_existential_proposition_l2306_230634

-- Define the propositions
def proposition (x : ℝ) := x^2 - 2 * x + 1 ≤ 0

-- Define the negation of the propositions
def negation_prop (x : ℝ) := x^2 - 2 * x + 1 > 0

-- Theorem to prove that the negation of the existential proposition is the universal proposition
theorem negation_of_existential_proposition
  (h : ¬ ∃ x : ℝ, proposition x) :
  ∀ x : ℝ, negation_prop x :=
by
  sorry

end negation_of_existential_proposition_l2306_230634


namespace mod_residue_l2306_230602

theorem mod_residue : (250 * 15 - 337 * 5 + 22) % 13 = 7 := by
  sorry

end mod_residue_l2306_230602


namespace pq_condition_l2306_230678

theorem pq_condition (p q : ℝ) (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 :=
by
  sorry

end pq_condition_l2306_230678


namespace sum_of_interior_angles_of_pentagon_l2306_230683

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  -- We skip the proof as per instruction
  sorry

end sum_of_interior_angles_of_pentagon_l2306_230683


namespace inequality_problem_l2306_230622

variable (a b c : ℝ)

theorem inequality_problem (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
sorry

end inequality_problem_l2306_230622


namespace both_selected_prob_l2306_230687

-- Given conditions
def prob_Ram := 6 / 7
def prob_Ravi := 1 / 5

-- The mathematically equivalent proof problem statement
theorem both_selected_prob : (prob_Ram * prob_Ravi) = 6 / 35 := by
  sorry

end both_selected_prob_l2306_230687


namespace ellipses_have_equal_focal_length_l2306_230629

-- Define ellipses and their focal lengths
def ellipse1_focal_length : ℝ := 8
def k_condition (k : ℝ) : Prop := 0 < k ∧ k < 9
def ellipse2_focal_length (k : ℝ) : ℝ := 8

-- The main statement
theorem ellipses_have_equal_focal_length (k : ℝ) (hk : k_condition k) :
  ellipse1_focal_length = ellipse2_focal_length k :=
sorry

end ellipses_have_equal_focal_length_l2306_230629


namespace exercise_l2306_230607

theorem exercise (a b : ℕ) (h1 : 656 = 3 * 7^2 + a * 7 + b) (h2 : 656 = 3 * 10^2 + a * 10 + b) : 
  (a * b) / 15 = 1 :=
by
  sorry

end exercise_l2306_230607


namespace candy_lasts_for_days_l2306_230694

-- Definitions based on conditions
def candy_from_neighbors : ℕ := 75
def candy_from_sister : ℕ := 130
def candy_traded : ℕ := 25
def candy_lost : ℕ := 15
def candy_eaten_per_day : ℕ := 7

-- Total candy calculation
def total_candy : ℕ := candy_from_neighbors + candy_from_sister - candy_traded - candy_lost
def days_candy_lasts : ℕ := total_candy / candy_eaten_per_day

-- Proof statement
theorem candy_lasts_for_days : days_candy_lasts = 23 := by
  -- sorry is used to skip the actual proof
  sorry

end candy_lasts_for_days_l2306_230694


namespace max_gcd_13n_plus_4_8n_plus_3_l2306_230686

theorem max_gcd_13n_plus_4_8n_plus_3 (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k = 9 ∧ gcd (13 * n + 4) (8 * n + 3) = k := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l2306_230686


namespace smaller_angle_measure_l2306_230645

theorem smaller_angle_measure (x : ℝ) (h₁ : 5 * x + 3 * x = 180) : 3 * x = 67.5 :=
by
  sorry

end smaller_angle_measure_l2306_230645


namespace divisor_of_number_l2306_230613

theorem divisor_of_number (n d q p : ℤ) 
  (h₁ : n = d * q + 3)
  (h₂ : n ^ 2 = d * p + 3) : 
  d = 6 := 
sorry

end divisor_of_number_l2306_230613


namespace minimum_value_expression_l2306_230699

theorem minimum_value_expression (α β : ℝ) : (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 144 :=
by
  sorry

end minimum_value_expression_l2306_230699


namespace ants_in_third_anthill_l2306_230661

-- Define the number of ants in the first anthill
def ants_first : ℕ := 100

-- Define the percentage reduction for each subsequent anthill
def percentage_reduction : ℕ := 20

-- Calculate the number of ants in the second anthill
def ants_second : ℕ := ants_first - (percentage_reduction * ants_first / 100)

-- Calculate the number of ants in the third anthill
def ants_third : ℕ := ants_second - (percentage_reduction * ants_second / 100)

-- Main theorem to prove that the number of ants in the third anthill is 64
theorem ants_in_third_anthill : ants_third = 64 := sorry

end ants_in_third_anthill_l2306_230661


namespace find_greatest_number_l2306_230680

def numbers := [0.07, -0.41, 0.8, 0.35, -0.9]

theorem find_greatest_number :
  ∃ x ∈ numbers, x > 0.7 ∧ ∀ y ∈ numbers, y > 0.7 → y = 0.8 :=
by
  sorry

end find_greatest_number_l2306_230680


namespace inequality_proof_l2306_230632

theorem inequality_proof (a b c : ℝ) (h : a ^ 2 + b ^ 2 + c ^ 2 = 3) :
  (a ^ 2) / (2 + b + c ^ 2) + (b ^ 2) / (2 + c + a ^ 2) + (c ^ 2) / (2 + a + b ^ 2) ≥ (a + b + c) ^ 2 / 12 :=
by sorry

end inequality_proof_l2306_230632


namespace alligator_population_at_end_of_year_l2306_230635

-- Define the conditions
def initial_population : ℕ := 4
def doubling_period_months : ℕ := 6
def total_months : ℕ := 12

-- Define the proof goal
theorem alligator_population_at_end_of_year (initial_population doubling_period_months total_months : ℕ)
  (h_init : initial_population = 4)
  (h_double : doubling_period_months = 6)
  (h_total : total_months = 12) :
  initial_population * (2 ^ (total_months / doubling_period_months)) = 16 := 
by
  sorry

end alligator_population_at_end_of_year_l2306_230635


namespace find_A_l2306_230644

theorem find_A (A : ℕ) (h : 59 = (A * 6) + 5) : A = 9 :=
by sorry

end find_A_l2306_230644
