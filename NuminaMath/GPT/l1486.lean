import Mathlib

namespace NUMINAMATH_GPT_probability_no_defective_pencils_l1486_148613

theorem probability_no_defective_pencils :
  let total_pencils := 6
  let defective_pencils := 2
  let pencils_chosen := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_ways := Nat.choose total_pencils pencils_chosen
  let non_defective_ways := Nat.choose non_defective_pencils pencils_chosen
  (non_defective_ways / total_ways : ℚ) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_no_defective_pencils_l1486_148613


namespace NUMINAMATH_GPT_mowing_field_time_l1486_148626

theorem mowing_field_time (h1 : (1 / 28 : ℝ) = (3 / 84 : ℝ))
                         (h2 : (1 / 84 : ℝ) = (1 / 84 : ℝ))
                         (h3 : (1 / 28 + 1 / 84 : ℝ) = (1 / 21 : ℝ)) :
                         21 = 1 / ((1 / 28) + (1 / 84)) := 
by {
  sorry
}

end NUMINAMATH_GPT_mowing_field_time_l1486_148626


namespace NUMINAMATH_GPT_real_root_of_system_l1486_148622

theorem real_root_of_system :
  (∃ x : ℝ, x^3 + 9 = 0 ∧ x + 3 = 0) ↔ x = -3 := 
by 
  sorry

end NUMINAMATH_GPT_real_root_of_system_l1486_148622


namespace NUMINAMATH_GPT_multiplication_solution_l1486_148643

theorem multiplication_solution 
  (x : ℤ) 
  (h : 72517 * x = 724807415) : 
  x = 9999 := 
sorry

end NUMINAMATH_GPT_multiplication_solution_l1486_148643


namespace NUMINAMATH_GPT_total_oranges_l1486_148654

def monday_oranges : ℕ := 100
def tuesday_oranges : ℕ := 3 * monday_oranges
def wednesday_oranges : ℕ := 70

theorem total_oranges : monday_oranges + tuesday_oranges + wednesday_oranges = 470 := by
  sorry

end NUMINAMATH_GPT_total_oranges_l1486_148654


namespace NUMINAMATH_GPT_exists_positive_m_f99_divisible_1997_l1486_148619

def f (x : ℕ) : ℕ := 3 * x + 2

noncomputable
def higher_order_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => sorry  -- placeholder since f^0 isn't defined in this context
  | 1 => f x
  | k + 1 => f (higher_order_f k x)

theorem exists_positive_m_f99_divisible_1997 :
  ∃ m : ℕ, m > 0 ∧ higher_order_f 99 m % 1997 = 0 :=
sorry

end NUMINAMATH_GPT_exists_positive_m_f99_divisible_1997_l1486_148619


namespace NUMINAMATH_GPT_chess_tournament_l1486_148611

-- Define the number of chess amateurs
def num_amateurs : ℕ := 5

-- Define the number of games each amateur plays
def games_per_amateur : ℕ := 4

-- Define the total number of chess games possible
def total_games : ℕ := num_amateurs * (num_amateurs - 1) / 2

-- The main statement to prove
theorem chess_tournament : total_games = 10 := 
by
  -- here should be the proof, but according to the task, we use sorry to skip
  sorry

end NUMINAMATH_GPT_chess_tournament_l1486_148611


namespace NUMINAMATH_GPT_number_of_interviewees_l1486_148627

theorem number_of_interviewees (n : ℕ) (h : (6 : ℚ) / (n * (n - 1)) = 1 / 70) : n = 21 :=
sorry

end NUMINAMATH_GPT_number_of_interviewees_l1486_148627


namespace NUMINAMATH_GPT_solve_system_l1486_148633

-- Define the conditions from the problem
def system_of_equations (x y : ℝ) : Prop :=
  (x = 4 * y) ∧ (x + 2 * y = -12)

-- Define the solution we want to prove
def solution (x y : ℝ) : Prop :=
  (x = -8) ∧ (y = -2)

-- State the theorem
theorem solve_system :
  ∃ x y : ℝ, system_of_equations x y ∧ solution x y :=
by 
  sorry

end NUMINAMATH_GPT_solve_system_l1486_148633


namespace NUMINAMATH_GPT_volunteers_per_class_l1486_148689

theorem volunteers_per_class (total_needed volunteers teachers_needed : ℕ) (classes : ℕ)
    (h_total : total_needed = 50) (h_teachers : teachers_needed = 13) (h_more_needed : volunteers = 7) (h_classes : classes = 6) :
  (total_needed - teachers_needed - volunteers) / classes = 5 :=
by
  -- calculation and simplification
  sorry

end NUMINAMATH_GPT_volunteers_per_class_l1486_148689


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1486_148679

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > 0) : 
  ((x > 2 ∧ x < 4) ↔ (2 < x ∧ x < 4)) :=
by {
    sorry
}

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1486_148679


namespace NUMINAMATH_GPT_race_time_diff_l1486_148602

-- Define the speeds and race distance
def Malcolm_speed : ℕ := 5  -- in minutes per mile
def Joshua_speed : ℕ := 7   -- in minutes per mile
def Alice_speed : ℕ := 6    -- in minutes per mile
def race_distance : ℕ := 12 -- in miles

-- Calculate times
def Malcolm_time : ℕ := Malcolm_speed * race_distance
def Joshua_time : ℕ := Joshua_speed * race_distance
def Alice_time : ℕ := Alice_speed * race_distance

-- Lean 4 statement to prove the time differences
theorem race_time_diff :
  Joshua_time - Malcolm_time = 24 ∧ Alice_time - Malcolm_time = 12 := by
  sorry

end NUMINAMATH_GPT_race_time_diff_l1486_148602


namespace NUMINAMATH_GPT_coordinates_of_point_P_l1486_148656

-- Define the function y = x^3
def cubic (x : ℝ) : ℝ := x^3

-- Define the derivative of the function
def derivative_cubic (x : ℝ) : ℝ := 3 * x^2

-- Define the condition for the slope of the tangent line to the function at point P
def slope_tangent_line := 3

-- Prove that the coordinates of point P are (1, 1) or (-1, -1) when the slope of the tangent line is 3
theorem coordinates_of_point_P (x : ℝ) (y : ℝ) 
    (h1 : y = cubic x) 
    (h2 : derivative_cubic x = slope_tangent_line) : 
    (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_point_P_l1486_148656


namespace NUMINAMATH_GPT_monomials_like_terms_l1486_148601

theorem monomials_like_terms (a b : ℕ) (h1 : 3 = a) (h2 : 4 = 2 * b) : a = 3 ∧ b = 2 :=
by
  sorry

end NUMINAMATH_GPT_monomials_like_terms_l1486_148601


namespace NUMINAMATH_GPT_gcd_282_470_l1486_148680

theorem gcd_282_470 : Nat.gcd 282 470 = 94 :=
by
  sorry

end NUMINAMATH_GPT_gcd_282_470_l1486_148680


namespace NUMINAMATH_GPT_sum_of_cubes_eq_zero_l1486_148649

theorem sum_of_cubes_eq_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -4) : a^3 + b^3 = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_eq_zero_l1486_148649


namespace NUMINAMATH_GPT_floor_length_l1486_148694

theorem floor_length (tile_length tile_width : ℕ) (floor_width max_tiles : ℕ)
  (h_tile : tile_length = 25 ∧ tile_width = 16)
  (h_floor_width : floor_width = 120)
  (h_max_tiles : max_tiles = 54) :
  ∃ floor_length : ℕ, 
    (∃ num_cols num_rows : ℕ, 
      num_cols * tile_width = floor_width ∧ 
      num_cols * num_rows = max_tiles ∧ 
      num_rows * tile_length = floor_length) ∧
    floor_length = 175 := 
by
  sorry

end NUMINAMATH_GPT_floor_length_l1486_148694


namespace NUMINAMATH_GPT_total_cupcakes_needed_l1486_148685

-- Definitions based on conditions
def cupcakes_per_event : ℝ := 96.0
def number_of_events : ℝ := 8.0

-- Theorem based on the question and the correct answer
theorem total_cupcakes_needed : (cupcakes_per_event * number_of_events) = 768.0 :=
by 
  sorry

end NUMINAMATH_GPT_total_cupcakes_needed_l1486_148685


namespace NUMINAMATH_GPT_min_rooms_needed_l1486_148691

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end NUMINAMATH_GPT_min_rooms_needed_l1486_148691


namespace NUMINAMATH_GPT_interval_of_monotonic_increase_l1486_148674

noncomputable def powerFunction (k n x : ℝ) : ℝ := k * x ^ n

variable {k n : ℝ}

theorem interval_of_monotonic_increase
    (h : ∃ k n : ℝ, powerFunction k n 4 = 2) :
    (∀ x y : ℝ, 0 < x ∧ x < y → powerFunction k n x < powerFunction k n y) ∨
    (∀ x y : ℝ, 0 ≤ x ∧ x < y → powerFunction k n x ≤ powerFunction k n y) := sorry

end NUMINAMATH_GPT_interval_of_monotonic_increase_l1486_148674


namespace NUMINAMATH_GPT_sum_of_coefficients_l1486_148620

-- Given polynomial definition
def P (x : ℝ) : ℝ := (1 + x - 3 * x^2) ^ 1965

-- Lean 4 statement for the proof problem
theorem sum_of_coefficients :
  P 1 = -1 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1486_148620


namespace NUMINAMATH_GPT_unique_positive_integer_solutions_l1486_148665

theorem unique_positive_integer_solutions : 
  ∀ (m n : ℕ), 0 < m ∧ 0 < n ∧ 7 ^ m - 3 * 2 ^ n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) :=
by sorry

end NUMINAMATH_GPT_unique_positive_integer_solutions_l1486_148665


namespace NUMINAMATH_GPT_montoya_budget_l1486_148644

def percentage_food (groceries: ℝ) (eating_out: ℝ) : ℝ :=
  groceries + eating_out

def percentage_transportation_rent_utilities (transportation: ℝ) (rent: ℝ) (utilities: ℝ) : ℝ :=
  transportation + rent + utilities

def total_percentage (food: ℝ) (transportation_rent_utilities: ℝ) : ℝ :=
  food + transportation_rent_utilities

theorem montoya_budget :
  ∀ (groceries : ℝ) (eating_out : ℝ) (transportation : ℝ) (rent : ℝ) (utilities : ℝ),
    groceries = 0.6 → eating_out = 0.2 → transportation = 0.1 → rent = 0.05 → utilities = 0.05 →
    total_percentage (percentage_food groceries eating_out) (percentage_transportation_rent_utilities transportation rent utilities) = 1 :=
by
sorry

end NUMINAMATH_GPT_montoya_budget_l1486_148644


namespace NUMINAMATH_GPT_rectangle_lengths_correct_l1486_148631

-- Definitions of the parameters and their relationships
noncomputable def AB := 1200
noncomputable def BC := 150
noncomputable def AB_ext := AB
noncomputable def BC_ext := BC + 350
noncomputable def CD := AB
noncomputable def DA := BC

-- Definitions of the calculated distances using the conditions
noncomputable def AP := Real.sqrt (AB^2 + BC_ext^2)
noncomputable def PD := Real.sqrt (BC_ext^2 + AB^2)

-- Using similarity of triangles for PQ and CQ
noncomputable def PQ := (350 / 500) * AP
noncomputable def CQ := (350 / 500) * AB

-- The theorem to prove the final results
theorem rectangle_lengths_correct :
    AP = 1300 ∧
    PD = 1250 ∧
    PQ = 910 ∧
    CQ = 840 :=
    by
    sorry

end NUMINAMATH_GPT_rectangle_lengths_correct_l1486_148631


namespace NUMINAMATH_GPT_quadratic_has_one_solution_l1486_148682

theorem quadratic_has_one_solution (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0) ∧ (∀ x₁ x₂ : ℝ, (3 * x₁^2 - 6 * x₁ + m = 0) → (3 * x₂^2 - 6 * x₂ + m = 0) → x₁ = x₂) → m = 3 :=
by
  -- intricate steps would go here
  sorry

end NUMINAMATH_GPT_quadratic_has_one_solution_l1486_148682


namespace NUMINAMATH_GPT_seulgi_stack_higher_l1486_148668

-- Define the conditions
def num_red_boxes : ℕ := 15
def num_yellow_boxes : ℕ := 20
def height_red_box : ℝ := 4.2
def height_yellow_box : ℝ := 3.3

-- Define the total height for each stack
def total_height_hyunjeong : ℝ := num_red_boxes * height_red_box
def total_height_seulgi : ℝ := num_yellow_boxes * height_yellow_box

-- Lean statement to prove the comparison of their heights
theorem seulgi_stack_higher : total_height_seulgi > total_height_hyunjeong :=
by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_seulgi_stack_higher_l1486_148668


namespace NUMINAMATH_GPT_principal_amount_l1486_148612

theorem principal_amount (P R : ℝ) (h1 : P + (P * R * 2) / 100 = 780) (h2 : P + (P * R * 7) / 100 = 1020) : P = 684 := 
sorry

end NUMINAMATH_GPT_principal_amount_l1486_148612


namespace NUMINAMATH_GPT_prove_sum_eq_9_l1486_148634

theorem prove_sum_eq_9 (a b : ℝ) (h : i * (a - i) = b - (2 * i) ^ 3) : a + b = 9 :=
by
  sorry

end NUMINAMATH_GPT_prove_sum_eq_9_l1486_148634


namespace NUMINAMATH_GPT_linear_function_intersects_x_axis_at_two_units_l1486_148603

theorem linear_function_intersects_x_axis_at_two_units (k : ℝ) :
  (∃ x : ℝ, y = k * x + 2 ∧ y = 0 ∧ |x| = 2) ↔ k = 1 ∨ k = -1 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_intersects_x_axis_at_two_units_l1486_148603


namespace NUMINAMATH_GPT_trapezoid_area_eq_c_l1486_148609

theorem trapezoid_area_eq_c (b c : ℝ) (hb : b = Real.sqrt c) (hc : 0 < c) :
    let shorter_base := b - 3
    let altitude := b
    let longer_base := b + 3
    let K := (1/2) * (shorter_base + longer_base) * altitude
    K = c :=
by
    sorry

end NUMINAMATH_GPT_trapezoid_area_eq_c_l1486_148609


namespace NUMINAMATH_GPT_biggest_number_l1486_148614

noncomputable def Yoongi_collected : ℕ := 4
noncomputable def Jungkook_collected : ℕ := 6 * 3
noncomputable def Yuna_collected : ℕ := 5

theorem biggest_number :
  Jungkook_collected = 18 ∧ Jungkook_collected > Yoongi_collected ∧ Jungkook_collected > Yuna_collected :=
by
  sorry

end NUMINAMATH_GPT_biggest_number_l1486_148614


namespace NUMINAMATH_GPT_ratio_problem_l1486_148692

open Classical 

variables {q r s t u : ℚ}

theorem ratio_problem (h1 : q / r = 8) (h2 : s / r = 5) (h3 : s / t = 1 / 4) (h4 : u / t = 3) :
  u / q = 15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_problem_l1486_148692


namespace NUMINAMATH_GPT_intersection_product_distance_eq_eight_l1486_148652

noncomputable def parametricCircle : ℝ → ℝ × ℝ :=
  λ θ => (4 * Real.cos θ, 4 * Real.sin θ)

noncomputable def parametricLine : ℝ → ℝ × ℝ :=
  λ t => (2 + (1 / 2) * t, 2 + (Real.sqrt 3 / 2) * t)

theorem intersection_product_distance_eq_eight :
  ∀ θ t,
    let (x1, y1) := parametricCircle θ
    let (x2, y2) := parametricLine t
    (x1^2 + y1^2 = 16) ∧ (x2 = x1 ∧ y2 = y1) →
    ∃ t1 t2,
      x1 = 2 + (1 / 2) * t1 ∧ y1 = 2 + (Real.sqrt 3 / 2) * t1 ∧
      x1 = 2 + (1 / 2) * t2 ∧ y1 = 2 + (Real.sqrt 3 / 2) * t2 ∧
      (t1 * t2 = -8) ∧ (|t1 * t2| = 8) := 
by
  intros θ t
  dsimp only
  intro h
  sorry

end NUMINAMATH_GPT_intersection_product_distance_eq_eight_l1486_148652


namespace NUMINAMATH_GPT_sum_of_smallest_x_and_y_for_540_l1486_148696

theorem sum_of_smallest_x_and_y_for_540 (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : ∃ k₁, 540 * x = k₁ * k₁)
  (h2 : ∃ k₂, 540 * y = k₂ * k₂ * k₂) :
  x + y = 65 := 
sorry

end NUMINAMATH_GPT_sum_of_smallest_x_and_y_for_540_l1486_148696


namespace NUMINAMATH_GPT_num_ways_to_remove_blocks_l1486_148677

-- Definitions based on the problem conditions
def stack_blocks := 85
def block_layers := [1, 4, 16, 64]

-- Theorem statement
theorem num_ways_to_remove_blocks : 
  (∃ f : (ℕ → ℕ), 
    (∀ n, f n = if n = 0 then 1 else if n ≤ 4 then n * f (n - 1) + 3 * (f (n - 1) - 1) else 4^3 * 16) ∧ 
    f 5 = 3384) := sorry

end NUMINAMATH_GPT_num_ways_to_remove_blocks_l1486_148677


namespace NUMINAMATH_GPT_star_shell_arrangements_l1486_148693

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Conditions
def outward_points : ℕ := 6
def inward_points : ℕ := 6
def total_points : ℕ := outward_points + inward_points
def unique_shells : ℕ := 12

-- The problem statement translated into Lean 4:
theorem star_shell_arrangements : (factorial unique_shells / 12 = 39916800) :=
by
  sorry

end NUMINAMATH_GPT_star_shell_arrangements_l1486_148693


namespace NUMINAMATH_GPT_top_card_probability_spades_or_clubs_l1486_148686

-- Definitions
def total_cards : ℕ := 52
def suits : ℕ := 4
def ranks : ℕ := 13
def spades_cards : ℕ := ranks
def clubs_cards : ℕ := ranks
def favorable_outcomes : ℕ := spades_cards + clubs_cards

-- Probability calculation statement
theorem top_card_probability_spades_or_clubs :
  (favorable_outcomes : ℚ) / (total_cards : ℚ) = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_top_card_probability_spades_or_clubs_l1486_148686


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1486_148658

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
    (h1 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
    (h2 : (S 2017) / 2017 - (S 17) / 17 = 100) :
    d = 1/10 := 
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1486_148658


namespace NUMINAMATH_GPT_find_original_workers_and_time_l1486_148667

-- Definitions based on the identified conditions
def original_workers (x : ℕ) (y : ℕ) : Prop :=
  (x - 2) * (y + 4) = x * y ∧
  (x + 3) * (y - 2) > x * y ∧
  (x + 4) * (y - 3) > x * y

-- Problem statement to prove
theorem find_original_workers_and_time (x y : ℕ) :
  original_workers x y → x = 6 ∧ y = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_original_workers_and_time_l1486_148667


namespace NUMINAMATH_GPT_cylinder_sphere_ratio_is_3_2_l1486_148683

noncomputable def cylinder_sphere_surface_ratio (r : ℝ) : ℝ :=
  let cylinder_surface_area := 2 * Real.pi * r^2 + 2 * r * Real.pi * (2 * r)
  let sphere_surface_area := 4 * Real.pi * r^2
  cylinder_surface_area / sphere_surface_area

theorem cylinder_sphere_ratio_is_3_2 (r : ℝ) (h : r > 0) :
  cylinder_sphere_surface_ratio r = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_sphere_ratio_is_3_2_l1486_148683


namespace NUMINAMATH_GPT_prob_rain_all_days_l1486_148657

/--
The probability of rain on Friday, Saturday, and Sunday is given by 
0.40, 0.60, and 0.35 respectively.
We want to prove that the combined probability of rain on all three days,
assuming independence, is 8.4%.
-/
theorem prob_rain_all_days :
  let p_friday := 0.40
  let p_saturday := 0.60
  let p_sunday := 0.35
  p_friday * p_saturday * p_sunday = 0.084 :=
by
  sorry

end NUMINAMATH_GPT_prob_rain_all_days_l1486_148657


namespace NUMINAMATH_GPT_hunter_rats_l1486_148641

-- Defining the conditions
variable (H : ℕ) (E : ℕ := H + 30) (K : ℕ := 3 * (H + E)) 
  
-- Defining the total number of rats condition
def total_rats : Prop := H + E + K = 200

-- Defining the goal: Prove Hunter has 10 rats
theorem hunter_rats (h : total_rats H) : H = 10 := by
  sorry

end NUMINAMATH_GPT_hunter_rats_l1486_148641


namespace NUMINAMATH_GPT_steven_name_day_44_l1486_148606

def W (n : ℕ) : ℕ :=
  2 * (n / 2) + 4 * ((n - 1) / 2)

theorem steven_name_day_44 : ∃ n : ℕ, W n = 44 :=
  by 
  existsi 16
  sorry

end NUMINAMATH_GPT_steven_name_day_44_l1486_148606


namespace NUMINAMATH_GPT_sum_first_ten_multiples_of_nine_l1486_148607

theorem sum_first_ten_multiples_of_nine :
  let a := 9
  let d := 9
  let n := 10
  let S_n := n * (2 * a + (n - 1) * d) / 2
  S_n = 495 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_ten_multiples_of_nine_l1486_148607


namespace NUMINAMATH_GPT_price_of_each_bracelet_l1486_148659

-- The conditions
def bike_cost : ℕ := 112
def days_in_two_weeks : ℕ := 14
def bracelets_per_day : ℕ := 8
def total_bracelets := days_in_two_weeks * bracelets_per_day

-- The question and the expected answer
def price_per_bracelet : ℕ := bike_cost / total_bracelets

theorem price_of_each_bracelet :
  price_per_bracelet = 1 := 
by
  sorry

end NUMINAMATH_GPT_price_of_each_bracelet_l1486_148659


namespace NUMINAMATH_GPT_percent_of_200_is_400_when_whole_is_50_l1486_148695

theorem percent_of_200_is_400_when_whole_is_50 (Part Whole : ℕ) (hPart : Part = 200) (hWhole : Whole = 50) :
  (Part / Whole) * 100 = 400 :=
by {
  -- Proof steps go here.
  sorry
}

end NUMINAMATH_GPT_percent_of_200_is_400_when_whole_is_50_l1486_148695


namespace NUMINAMATH_GPT_total_games_l1486_148661

-- Definitions and conditions
noncomputable def num_teams : ℕ := 12

noncomputable def regular_season_games_each : ℕ := 4

noncomputable def knockout_games_each : ℕ := 2

-- Calculate total number of games
theorem total_games : (num_teams * (num_teams - 1) / 2) * regular_season_games_each + 
                      (num_teams * knockout_games_each / 2) = 276 :=
by
  -- This is the statement to be proven
  sorry

end NUMINAMATH_GPT_total_games_l1486_148661


namespace NUMINAMATH_GPT_geom_series_eq_l1486_148638

noncomputable def C (n : ℕ) := 256 * (1 - 1 / (4^n)) / (3 / 4)
noncomputable def D (n : ℕ) := 1024 * (1 - 1 / ((-2)^n)) / (3 / 2)

theorem geom_series_eq (n : ℕ) (h : n ≥ 1) : C n = D n ↔ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_geom_series_eq_l1486_148638


namespace NUMINAMATH_GPT_nat_know_albums_l1486_148615

/-- Define the number of novels, comics, documentaries and crates properties --/
def novels := 145
def comics := 271
def documentaries := 419
def crates := 116
def items_per_crate := 9

/-- Define the total capacity of crates --/
def total_capacity := crates * items_per_crate

/-- Define the total number of other items --/
def other_items := novels + comics + documentaries

/-- Define the number of albums --/
def albums := total_capacity - other_items

/-- Theorem: Prove that the number of albums is equal to 209 --/
theorem nat_know_albums : albums = 209 := by
  sorry

end NUMINAMATH_GPT_nat_know_albums_l1486_148615


namespace NUMINAMATH_GPT_problem_value_eq_13_l1486_148648

theorem problem_value_eq_13 : 8 / 4 - 3^2 + 4 * 5 = 13 :=
by
  sorry

end NUMINAMATH_GPT_problem_value_eq_13_l1486_148648


namespace NUMINAMATH_GPT_solve_system1_l1486_148632

theorem solve_system1 (x y : ℝ) :
  x + y + 3 = 10 ∧ 4 * (x + y) - y = 25 →
  x = 4 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_system1_l1486_148632


namespace NUMINAMATH_GPT_dealer_decision_is_mode_l1486_148660

noncomputable def sales_A := 15
noncomputable def sales_B := 22
noncomputable def sales_C := 18
noncomputable def sales_D := 10

def is_mode (sales: List ℕ) (mode_value: ℕ) : Prop :=
  mode_value ∈ sales ∧ ∀ x ∈ sales, x ≤ mode_value

theorem dealer_decision_is_mode : 
  is_mode [sales_A, sales_B, sales_C, sales_D] sales_B :=
by
  sorry

end NUMINAMATH_GPT_dealer_decision_is_mode_l1486_148660


namespace NUMINAMATH_GPT_perimeter_smallest_square_l1486_148628

theorem perimeter_smallest_square 
  (d : ℝ) (side_largest : ℝ)
  (h1 : d = 3) 
  (h2 : side_largest = 22) : 
  4 * (side_largest - 2 * d - 2 * d) = 40 := by
  sorry

end NUMINAMATH_GPT_perimeter_smallest_square_l1486_148628


namespace NUMINAMATH_GPT_compute_a_plus_b_l1486_148651

theorem compute_a_plus_b (a b : ℝ) (h : ∃ (u v w : ℕ), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ u + v + w = 8 ∧ u * v * w = b ∧ u * v + v * w + w * u = a) : 
  a + b = 27 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_compute_a_plus_b_l1486_148651


namespace NUMINAMATH_GPT_lottery_most_frequent_number_l1486_148636

noncomputable def m (i : ℕ) : ℚ :=
  ((i - 1) * (90 - i) * (89 - i) * (88 - i)) / 6

theorem lottery_most_frequent_number :
  ∀ (i : ℕ), 2 ≤ i ∧ i ≤ 87 → m 23 ≥ m i :=
by 
  sorry -- Proof goes here. This placeholder allows the file to compile.

end NUMINAMATH_GPT_lottery_most_frequent_number_l1486_148636


namespace NUMINAMATH_GPT_boris_can_achieve_7_60_cents_l1486_148675

/-- Define the conditions as constants -/
def penny_value : ℕ := 1
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def quarter_value : ℕ := 25

def penny_to_dimes : ℕ := 69
def dime_to_pennies : ℕ := 5
def nickel_to_quarters : ℕ := 120

/-- Function to determine if a value can be produced by a sequence of machine operations -/
def achievable_value (start: ℕ) (target: ℕ) : Prop :=
  ∃ k : ℕ, target = start + k * penny_to_dimes

theorem boris_can_achieve_7_60_cents : achievable_value penny_value 760 :=
  sorry

end NUMINAMATH_GPT_boris_can_achieve_7_60_cents_l1486_148675


namespace NUMINAMATH_GPT_phone_calls_to_reach_Davina_l1486_148646

theorem phone_calls_to_reach_Davina : 
  (∀ (a b : ℕ), (0 ≤ a ∧ a < 10) ∧ (0 ≤ b ∧ b < 10)) → (least_num_calls : ℕ) = 100 :=
by
  sorry

end NUMINAMATH_GPT_phone_calls_to_reach_Davina_l1486_148646


namespace NUMINAMATH_GPT_largest_of_7_consecutive_numbers_with_average_20_l1486_148655

variable (n : ℤ) 

theorem largest_of_7_consecutive_numbers_with_average_20
  (h_avg : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6))/7 = 20) : 
  (n + 6) = 23 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_largest_of_7_consecutive_numbers_with_average_20_l1486_148655


namespace NUMINAMATH_GPT_smallest_negative_integer_solution_l1486_148671

theorem smallest_negative_integer_solution :
  ∃ x : ℤ, 45 * x + 8 ≡ 5 [ZMOD 24] ∧ x = -7 :=
sorry

end NUMINAMATH_GPT_smallest_negative_integer_solution_l1486_148671


namespace NUMINAMATH_GPT_sale_price_relationship_l1486_148635

/-- Elaine's Gift Shop increased the original prices of all items by 10% 
  and then offered a 30% discount on these new prices in a clearance sale 
  - proving the relationship between the final sale price and the original price of an item -/

theorem sale_price_relationship (p : ℝ) : 
  (0.7 * (1.1 * p) = 0.77 * p) :=
by 
  sorry

end NUMINAMATH_GPT_sale_price_relationship_l1486_148635


namespace NUMINAMATH_GPT_sum_of_midpoint_xcoords_l1486_148681

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end NUMINAMATH_GPT_sum_of_midpoint_xcoords_l1486_148681


namespace NUMINAMATH_GPT_line_equations_l1486_148673

theorem line_equations : 
  ∀ (x y : ℝ), (∃ a b c : ℝ, 2 * x + y - 12 = 0 ∨ 2 * x - 5 * y = 0 ∧ (x, y) = (5, 2) ∧ b = 2 * a) :=
by
  sorry

end NUMINAMATH_GPT_line_equations_l1486_148673


namespace NUMINAMATH_GPT_transistor_count_2010_l1486_148676

-- Define the known constants and conditions
def initial_transistors : ℕ := 2000000
def doubling_period : ℕ := 2
def years_elapsed : ℕ := 2010 - 1995
def number_of_doublings := years_elapsed / doubling_period -- we want floor division

-- The theorem statement we need to prove
theorem transistor_count_2010 : initial_transistors * 2^number_of_doublings = 256000000 := by
  sorry

end NUMINAMATH_GPT_transistor_count_2010_l1486_148676


namespace NUMINAMATH_GPT_min_xsq_ysq_zsq_l1486_148664

noncomputable def min_value_x_sq_y_sq_z_sq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : ℝ :=
  (x^2 + y^2 + z^2)

theorem min_xsq_ysq_zsq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  min_value_x_sq_y_sq_z_sq x y z h = 40 / 7 :=
  sorry

end NUMINAMATH_GPT_min_xsq_ysq_zsq_l1486_148664


namespace NUMINAMATH_GPT_largest_angle_heptagon_l1486_148697

theorem largest_angle_heptagon :
  ∃ (x : ℝ), 4 * x + 4 * x + 4 * x + 5 * x + 6 * x + 7 * x + 8 * x = 900 ∧ 8 * x = (7200 / 38) := 
by 
  sorry

end NUMINAMATH_GPT_largest_angle_heptagon_l1486_148697


namespace NUMINAMATH_GPT_angle_bisector_correct_length_l1486_148630

-- Define the isosceles triangle with the given conditions
structure IsoscelesTriangle :=
  (base : ℝ)
  (lateral : ℝ)
  (is_isosceles : lateral = 20 ∧ base = 5)

-- Define the problem of finding the angle bisector
noncomputable def angle_bisector_length (tri : IsoscelesTriangle) : ℝ :=
  6

-- The main theorem to state the problem
theorem angle_bisector_correct_length (tri : IsoscelesTriangle) : 
  angle_bisector_length tri = 6 :=
by
  -- We state the theorem, skipping the proof (sorry)
  sorry

end NUMINAMATH_GPT_angle_bisector_correct_length_l1486_148630


namespace NUMINAMATH_GPT_average_minutes_run_per_day_l1486_148698

theorem average_minutes_run_per_day (f : ℕ) (h_nonzero : f ≠ 0)
  (third_avg fourth_avg fifth_avg : ℕ)
  (third_avg_eq : third_avg = 14)
  (fourth_avg_eq : fourth_avg = 18)
  (fifth_avg_eq : fifth_avg = 8)
  (third_count fourth_count fifth_count : ℕ)
  (third_count_eq : third_count = 3 * fourth_count)
  (fourth_count_eq : fourth_count = f / 2)
  (fifth_count_eq : fifth_count = f) :
  (third_avg * third_count + fourth_avg * fourth_count + fifth_avg * fifth_count) / (third_count + fourth_count + fifth_count) = 38 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_minutes_run_per_day_l1486_148698


namespace NUMINAMATH_GPT_intersection_correct_l1486_148608

open Set

def M : Set ℤ := {0, 1, 2, -1}
def N : Set ℤ := {0, 1, 2, 3}

theorem intersection_correct : M ∩ N = {0, 1, 2} :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_intersection_correct_l1486_148608


namespace NUMINAMATH_GPT_number_of_Al_atoms_l1486_148670

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_Br : ℝ := 79.90
def number_of_Br_atoms : ℕ := 3
def molecular_weight : ℝ := 267

theorem number_of_Al_atoms (x : ℝ) : 
  molecular_weight = (atomic_weight_Al * x) + (atomic_weight_Br * number_of_Br_atoms) → 
  x = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_Al_atoms_l1486_148670


namespace NUMINAMATH_GPT_sum_of_medians_bounds_l1486_148610

theorem sum_of_medians_bounds (a b c m_a m_b m_c : ℝ) 
    (h1 : m_a < (b + c) / 2)
    (h2 : m_b < (a + c) / 2)
    (h3 : m_c < (a + b) / 2)
    (h4 : ∀a b c : ℝ, a + b > c) :
    (3 / 4) * (a + b + c) < m_a + m_b + m_c ∧ m_a + m_b + m_c < a + b + c := 
by
  sorry

end NUMINAMATH_GPT_sum_of_medians_bounds_l1486_148610


namespace NUMINAMATH_GPT_smallest_possible_e_l1486_148684

-- Define the polynomial with its roots and integer coefficients
def polynomial (x : ℝ) : ℝ := (x + 4) * (x - 6) * (x - 10) * (2 * x + 1)

-- Define e as the constant term
def e : ℝ := 200 -- based on the final expanded polynomial result

-- The theorem stating the smallest possible value of e
theorem smallest_possible_e : 
  ∃ (e : ℕ), e > 0 ∧ polynomial e = 200 := 
sorry

end NUMINAMATH_GPT_smallest_possible_e_l1486_148684


namespace NUMINAMATH_GPT_daysRequired_l1486_148672

-- Defining the structure of the problem
structure WallConstruction where
  m1 : ℕ    -- Number of men in the first scenario
  d1 : ℕ    -- Number of days in the first scenario
  m2 : ℕ    -- Number of men in the second scenario

-- Given values
def wallConstructionProblem : WallConstruction :=
  WallConstruction.mk 20 5 30

-- The total work constant
def totalWork (wc : WallConstruction) : ℕ :=
  wc.m1 * wc.d1

-- Proving the number of days required for m2 men
theorem daysRequired (wc : WallConstruction) (k : ℕ) : 
  k = totalWork wc → (wc.m2 * (k / wc.m2 : ℚ) = k) → (k / wc.m2 : ℚ) = 3.3 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_daysRequired_l1486_148672


namespace NUMINAMATH_GPT_negation_proposition_l1486_148687

theorem negation_proposition:
  (¬ (∀ x : ℝ, (1 ≤ x) → (x^2 - 2*x + 1 ≥ 0))) ↔ (∃ x : ℝ, (1 ≤ x) ∧ (x^2 - 2*x + 1 < 0)) := 
sorry

end NUMINAMATH_GPT_negation_proposition_l1486_148687


namespace NUMINAMATH_GPT_student_tickets_sold_l1486_148669

theorem student_tickets_sold
  (A S : ℕ)
  (h1 : A + S = 846)
  (h2 : 6 * A + 3 * S = 3846) :
  S = 410 :=
sorry

end NUMINAMATH_GPT_student_tickets_sold_l1486_148669


namespace NUMINAMATH_GPT_solution_set_inequality_l1486_148642

theorem solution_set_inequality (x : ℝ) : (1 < x ∧ x < 3) ↔ (x^2 - 4*x + 3 < 0) :=
by sorry

end NUMINAMATH_GPT_solution_set_inequality_l1486_148642


namespace NUMINAMATH_GPT_log_relationship_l1486_148618

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem log_relationship :
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_log_relationship_l1486_148618


namespace NUMINAMATH_GPT_ruel_usable_stamps_l1486_148650

def totalStamps (books10 books15 books25 books30 : ℕ) (stamps10 stamps15 stamps25 stamps30 : ℕ) : ℕ :=
  books10 * stamps10 + books15 * stamps15 + books25 * stamps25 + books30 * stamps30

def damagedStamps (damaged25 damaged30 : ℕ) : ℕ :=
  damaged25 + damaged30

def usableStamps (books10 books15 books25 books30 stamps10 stamps15 stamps25 stamps30 damaged25 damaged30 : ℕ) : ℕ :=
  totalStamps books10 books15 books25 books30 stamps10 stamps15 stamps25 stamps30 - damagedStamps damaged25 damaged30

theorem ruel_usable_stamps :
  usableStamps 4 6 3 2 10 15 25 30 5 3 = 257 := by
  sorry

end NUMINAMATH_GPT_ruel_usable_stamps_l1486_148650


namespace NUMINAMATH_GPT_seeds_total_l1486_148624

theorem seeds_total (wednesday_seeds thursday_seeds : ℕ) (h_wed : wednesday_seeds = 20) (h_thu : thursday_seeds = 2) : (wednesday_seeds + thursday_seeds) = 22 := by
  sorry

end NUMINAMATH_GPT_seeds_total_l1486_148624


namespace NUMINAMATH_GPT_three_digit_number_formed_by_1198th_1200th_digits_l1486_148653

def albertSequenceDigit (n : ℕ) : ℕ :=
  -- Define the nth digit in Albert's sequence
  sorry

theorem three_digit_number_formed_by_1198th_1200th_digits :
  let d1198 := albertSequenceDigit 1198
  let d1199 := albertSequenceDigit 1199
  let d1200 := albertSequenceDigit 1200
  (d1198 * 100 + d1199 * 10 + d1200) = 220 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_formed_by_1198th_1200th_digits_l1486_148653


namespace NUMINAMATH_GPT_description_of_T_l1486_148604

-- Define the set T
def T : Set (ℝ × ℝ) := 
  {p | (p.1 = 1 ∧ p.2 ≤ 9) ∨ (p.2 = 9 ∧ p.1 ≤ 1) ∨ (p.2 = p.1 + 8 ∧ p.1 ≥ 1)}

-- State the formal proof problem: T is three rays with a common point
theorem description_of_T :
  (∃ p : ℝ × ℝ, p = (1, 9) ∧ 
    ∀ q ∈ T, 
      (q.1 = 1 ∧ q.2 ≤ 9) ∨ 
      (q.2 = 9 ∧ q.1 ≤ 1) ∨ 
      (q.2 = q.1 + 8 ∧ q.1 ≥ 1)) :=
by
  sorry

end NUMINAMATH_GPT_description_of_T_l1486_148604


namespace NUMINAMATH_GPT_sum_of_cube_faces_l1486_148637

theorem sum_of_cube_faces :
  ∃ (a b c d e f : ℕ), 
    (a = 12) ∧ 
    (b = a + 3) ∧ 
    (c = b + 3) ∧ 
    (d = c + 3) ∧ 
    (e = d + 3) ∧ 
    (f = e + 3) ∧ 
    (a + f = 39) ∧ 
    (b + e = 39) ∧ 
    (c + d = 39) ∧ 
    (a + b + c + d + e + f = 117) :=
by
  let a := 12
  let b := a + 3
  let c := b + 3
  let d := c + 3
  let e := d + 3
  let f := e + 3
  have h1 : a + f = 39 := sorry
  have h2 : b + e = 39 := sorry
  have h3 : c + d = 39 := sorry
  have sum : a + b + c + d + e + f = 117 := sorry
  exact ⟨a, b, c, d, e, f, rfl, rfl, rfl, rfl, rfl, rfl, h1, h2, h3, sum⟩

end NUMINAMATH_GPT_sum_of_cube_faces_l1486_148637


namespace NUMINAMATH_GPT_correct_calculation_l1486_148666

/-- Conditions for the given calculations -/
def cond_a : Prop := (-2) ^ 3 = 8
def cond_b : Prop := (-3) ^ 2 = -9
def cond_c : Prop := -(3 ^ 2) = -9
def cond_d : Prop := (-2) ^ 2 = 4

/-- Prove that the correct calculation among the given is -3^2 = -9 -/
theorem correct_calculation : cond_c :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l1486_148666


namespace NUMINAMATH_GPT_hydrogen_atoms_in_compound_l1486_148663

theorem hydrogen_atoms_in_compound :
  ∀ (n : ℕ), 98 = 14 + n + 80 → n = 4 :=
by intro n h_eq
   sorry

end NUMINAMATH_GPT_hydrogen_atoms_in_compound_l1486_148663


namespace NUMINAMATH_GPT_largest_of_five_consecutive_odd_integers_with_product_93555_l1486_148640

theorem largest_of_five_consecutive_odd_integers_with_product_93555 : 
  ∃ n, (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) = 93555) ∧ (n + 8 = 19) :=
sorry

end NUMINAMATH_GPT_largest_of_five_consecutive_odd_integers_with_product_93555_l1486_148640


namespace NUMINAMATH_GPT_quadratic_roots_l1486_148629

theorem quadratic_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : a - b + c = 0) (h3 : (b^2 - 4 * a * c) = 0) : 2 * a - b = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_roots_l1486_148629


namespace NUMINAMATH_GPT_min_N_such_that_next_person_sits_next_to_someone_l1486_148678

def circular_table_has_80_chairs : Prop := ∃ chairs : ℕ, chairs = 80
def N_people_seated (N : ℕ) : Prop := N > 0
def next_person_sits_next_to_someone (N : ℕ) : Prop :=
  ∀ additional_person_seated : ℕ, additional_person_seated ≤ N → additional_person_seated > 0 
  → ∃ adjacent_person : ℕ, adjacent_person ≤ N ∧ adjacent_person > 0
def smallest_value_for_N (N : ℕ) : Prop :=
  (∀ k : ℕ, k < N → ¬next_person_sits_next_to_someone k)

theorem min_N_such_that_next_person_sits_next_to_someone :
  circular_table_has_80_chairs →
  smallest_value_for_N 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_min_N_such_that_next_person_sits_next_to_someone_l1486_148678


namespace NUMINAMATH_GPT_algebraic_identity_l1486_148688

theorem algebraic_identity (a b : ℝ) : a^2 - 2 * a * b + b^2 = (a - b)^2 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_identity_l1486_148688


namespace NUMINAMATH_GPT_total_population_correct_l1486_148662

-- Given conditions
def number_of_cities : ℕ := 25
def average_population : ℕ := 3800

-- Statement to prove
theorem total_population_correct : number_of_cities * average_population = 95000 :=
by
  sorry

end NUMINAMATH_GPT_total_population_correct_l1486_148662


namespace NUMINAMATH_GPT_find_second_sum_l1486_148699

theorem find_second_sum (x : ℝ) (h : 24 * x / 100 = (2730 - x) * 15 / 100) : 2730 - x = 1680 := by
  sorry

end NUMINAMATH_GPT_find_second_sum_l1486_148699


namespace NUMINAMATH_GPT_volume_conversion_l1486_148645

-- Define the given conditions
def V_feet : ℕ := 216
def C_factor : ℕ := 27

-- State the theorem to prove
theorem volume_conversion : V_feet / C_factor = 8 :=
  sorry

end NUMINAMATH_GPT_volume_conversion_l1486_148645


namespace NUMINAMATH_GPT_maximize_area_of_quadrilateral_l1486_148600

theorem maximize_area_of_quadrilateral (k : ℝ) (h0 : 0 < k) (h1 : k < 1) 
    (hE : ∀ E : ℝ, E = 2 * k) (hF : ∀ F : ℝ, F = 2 * k) :
    k = 1/2 ∧ (2 * (1 - k) ^ 2) = 1/2 := 
by 
  sorry

end NUMINAMATH_GPT_maximize_area_of_quadrilateral_l1486_148600


namespace NUMINAMATH_GPT_magnitude_of_z_l1486_148621

open Complex

theorem magnitude_of_z (z : ℂ) (h : z^2 + Complex.normSq z = 4 - 7 * Complex.I) : 
  Complex.normSq z = 65 / 8 := 
by
  sorry

end NUMINAMATH_GPT_magnitude_of_z_l1486_148621


namespace NUMINAMATH_GPT_total_cars_l1486_148616

theorem total_cars (yesterday today : ℕ) (h_yesterday : yesterday = 60) (h_today : today = 2 * yesterday) : yesterday + today = 180 := 
sorry

end NUMINAMATH_GPT_total_cars_l1486_148616


namespace NUMINAMATH_GPT_smallest_n_contains_constant_term_l1486_148605

theorem smallest_n_contains_constant_term :
  ∃ n : ℕ, (∀ x : ℝ, x ≠ 0 → (2 * x^3 + 1 / x^(1/2))^n = c ↔ n = 7) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_contains_constant_term_l1486_148605


namespace NUMINAMATH_GPT_coeff_x3_in_product_l1486_148623

open Polynomial

noncomputable def p : Polynomial ℤ := 3 * X^3 + 2 * X^2 + 5 * X + 3
noncomputable def q : Polynomial ℤ := 4 * X^3 + 5 * X^2 + 6 * X + 8

theorem coeff_x3_in_product :
  (p * q).coeff 3 = 61 :=
by sorry

end NUMINAMATH_GPT_coeff_x3_in_product_l1486_148623


namespace NUMINAMATH_GPT_rebus_decrypt_correct_l1486_148625

-- Definitions
def is_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 9
def is_odd (d : ℕ) : Prop := is_digit d ∧ d % 2 = 1
def is_even (d : ℕ) : Prop := is_digit d ∧ d % 2 = 0

-- Variables representing ċharacters H, Ч (C), A, D, Y, E, F, B, K
variables (H C A D Y E F B K : ℕ)

-- Conditions
axiom H_odd : is_odd H
axiom C_even : is_even C
axiom A_even : is_even A
axiom D_odd : is_odd D
axiom Y_even : is_even Y
axiom E_even : is_even E
axiom F_odd : is_odd F
axiom B_digit : is_digit B
axiom K_odd : is_odd K

-- Correct answers
def H_val : ℕ := 5
def C_val : ℕ := 3
def A_val : ℕ := 2
def D_val : ℕ := 9
def Y_val : ℕ := 8
def E_val : ℕ := 8
def F_val : ℕ := 5
def B_any : ℕ := B
def K_val : ℕ := 5

-- Proof statement
theorem rebus_decrypt_correct : 
  H = H_val ∧
  C = C_val ∧
  A = A_val ∧
  D = D_val ∧
  Y = Y_val ∧
  E = E_val ∧
  F = F_val ∧
  K = K_val :=
sorry

end NUMINAMATH_GPT_rebus_decrypt_correct_l1486_148625


namespace NUMINAMATH_GPT_arithmetic_sequence_max_n_pos_sum_l1486_148647

noncomputable def max_n (a : ℕ → ℤ) (d : ℤ) : ℕ :=
  8

theorem arithmetic_sequence_max_n_pos_sum
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith_seq : ∀ n, a (n+1) = a 1 + n * d)
  (h_a1 : a 1 > 0)
  (h_a4_a5_sum_pos : a 4 + a 5 > 0)
  (h_a4_a5_prod_neg : a 4 * a 5 < 0) :
  max_n a d = 8 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_max_n_pos_sum_l1486_148647


namespace NUMINAMATH_GPT_find_third_number_x_l1486_148639

variable {a b : ℝ}

theorem find_third_number_x (h : a < b) :
  (∃ x : ℝ, x = a * b / (2 * b - a) ∧ x < a) ∨ 
  (∃ x : ℝ, x = 2 * a * b / (a + b) ∧ a < x ∧ x < b) ∨ 
  (∃ x : ℝ, x = a * b / (2 * a - b) ∧ a < b ∧ b < x) :=
sorry

end NUMINAMATH_GPT_find_third_number_x_l1486_148639


namespace NUMINAMATH_GPT_largest_K_is_1_l1486_148690

noncomputable def largest_K_vip (K : ℝ) : Prop :=
  ∀ (k : ℝ) (a b c : ℝ), 
  0 ≤ k ∧ k ≤ K → 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → 
  a^2 + b^2 + c^2 + k * a * b * c = k + 3 → 
  a + b + c ≤ 3

theorem largest_K_is_1 : largest_K_vip 1 :=
sorry

end NUMINAMATH_GPT_largest_K_is_1_l1486_148690


namespace NUMINAMATH_GPT_expand_expression_l1486_148617

theorem expand_expression (x y z : ℝ) :
  (2 * x + 15) * (3 * y + 20 * z + 25) = 
  6 * x * y + 40 * x * z + 50 * x + 45 * y + 300 * z + 375 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1486_148617
