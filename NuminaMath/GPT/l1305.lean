import Mathlib

namespace mayo_bottle_count_l1305_130560

-- Define the given ratio and the number of ketchup bottles
def ratio_ketchup : ℕ := 3
def ratio_mustard : ℕ := 3
def ratio_mayo : ℕ := 2
def num_ketchup_bottles : ℕ := 6

-- Define the proof problem: The number of mayo bottles
theorem mayo_bottle_count :
  (num_ketchup_bottles / ratio_ketchup) * ratio_mayo = 4 :=
by sorry

end mayo_bottle_count_l1305_130560


namespace min_value_of_ab_l1305_130568

theorem min_value_of_ab {a b : ℝ} (ha : a > 0) (hb : b > 0) 
  (h_eq : ∀ (x y : ℝ), (x / a + y / b = 1) → (x^2 + y^2 = 1)) : a * b = 2 :=
by sorry

end min_value_of_ab_l1305_130568


namespace average_eq_y_value_l1305_130531

theorem average_eq_y_value :
  (y : ℤ) → (h : (15 + 25 + y) / 3 = 20) → y = 20 :=
by
  intro y h
  sorry

end average_eq_y_value_l1305_130531


namespace total_exercise_time_l1305_130503

theorem total_exercise_time :
  let javier_minutes_per_day := 50
  let javier_days := 7
  let sanda_minutes_per_day := 90
  let sanda_days := 3
  (javier_minutes_per_day * javier_days + sanda_minutes_per_day * sanda_days) = 620 :=
by
  sorry

end total_exercise_time_l1305_130503


namespace find_numbers_l1305_130547

-- Definitions for the conditions
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0
def difference_is_three (x y : ℕ) : Prop := x - y = 3

-- Statement of the proof problem
theorem find_numbers (x y : ℕ) (h1 : is_three_digit x) (h2 : is_even_two_digit y) (h3 : difference_is_three x y) :
  x = 101 ∧ y = 98 :=
sorry

end find_numbers_l1305_130547


namespace quilt_block_shading_fraction_l1305_130541

theorem quilt_block_shading_fraction :
  (fraction_shaded : ℚ) → 
  (quilt_block_size : ℕ) → 
  (fully_shaded_squares : ℕ) → 
  (half_shaded_squares : ℕ) → 
  quilt_block_size = 16 →
  fully_shaded_squares = 6 →
  half_shaded_squares = 4 →
  fraction_shaded = 1/2 :=
by 
  sorry

end quilt_block_shading_fraction_l1305_130541


namespace base5_number_l1305_130573

/-- A base-5 number only contains the digits 0, 1, 2, 3, and 4.
    Given the number 21340, we need to prove that it could possibly be a base-5 number. -/
theorem base5_number (n : ℕ) (h : n = 21340) : 
  ∀ d ∈ [2, 1, 3, 4, 0], d < 5 :=
by sorry

end base5_number_l1305_130573


namespace find_x_value_l1305_130528

noncomputable def x_value (x y z : ℝ) : Prop :=
  (26 = (z + x) / 2) ∧
  (z = 52 - x) ∧
  (52 - x = (26 + y) / 2) ∧
  (y = 78 - 2 * x) ∧
  (78 - 2 * x = (8 + (52 - x)) / 2) ∧
  (x = 32)

theorem find_x_value : ∃ x y z : ℝ, x_value x y z :=
by
  use 32  -- x
  use 14  -- y derived from 78 - 2x where x = 32 leads to y = 14
  use 20  -- z derived from 52 - x where x = 32 leads to z = 20
  unfold x_value
  simp
  sorry

end find_x_value_l1305_130528


namespace trigonometric_expression_equals_one_l1305_130595

theorem trigonometric_expression_equals_one :
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2

  (1 - 1 / cos30) * (1 + 1 / sin60) *
  (1 - 1 / sin30) * (1 + 1 / cos60) = 1 :=
by
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2
  sorry

end trigonometric_expression_equals_one_l1305_130595


namespace ways_to_divide_week_l1305_130584

def week_seconds : ℕ := 604800

theorem ways_to_divide_week (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : week_seconds = n * m) :
  (∃ (pairs : ℕ), pairs = 336) :=
sorry

end ways_to_divide_week_l1305_130584


namespace number_of_parrots_l1305_130582

noncomputable def daily_consumption_parakeet : ℕ := 2
noncomputable def daily_consumption_parrot : ℕ := 14
noncomputable def daily_consumption_finch : ℕ := 1  -- Each finch eats half of what a parakeet eats

noncomputable def num_parakeets : ℕ := 3
noncomputable def num_finches : ℕ := 4
noncomputable def required_birdseed : ℕ := 266
noncomputable def days_in_week : ℕ := 7

theorem number_of_parrots (num_parrots : ℕ) : 
  daily_consumption_parakeet * num_parakeets * days_in_week +
  daily_consumption_finch * num_finches * days_in_week + 
  daily_consumption_parrot * num_parrots * days_in_week = required_birdseed → num_parrots = 2 :=
by 
  -- The proof is omitted as per the instructions
  sorry

end number_of_parrots_l1305_130582


namespace cone_inscribed_spheres_distance_l1305_130539

noncomputable def distance_between_sphere_centers (R α : ℝ) : ℝ :=
  R * (Real.sqrt 2) * Real.sin (α / 2) / (2 * Real.cos (α / 8) * Real.cos ((45 : ℝ) - α / 8))

theorem cone_inscribed_spheres_distance (R α : ℝ) (h1 : R > 0) (h2 : α > 0) :
  distance_between_sphere_centers R α = R * (Real.sqrt 2) * Real.sin (α / 2) / (2 * Real.cos (α / 8) * Real.cos ((45 : ℝ) - α / 8)) :=
by 
  sorry

end cone_inscribed_spheres_distance_l1305_130539


namespace vector_evaluation_l1305_130554

-- Define the vectors
def v1 : ℝ × ℝ := (3, -2)
def v2 : ℝ × ℝ := (2, -6)
def v3 : ℝ × ℝ := (0, 3)
def scalar : ℝ := 5
def expected_result : ℝ × ℝ := (-7, 31)

-- Statement to be proved
theorem vector_evaluation : v1 - scalar • v2 + v3 = expected_result :=
by
  sorry

end vector_evaluation_l1305_130554


namespace count_distinct_four_digit_numbers_divisible_by_3_ending_in_45_l1305_130538

/--
Prove that the total number of distinct four-digit numbers that end with 45 and 
are divisible by 3 is 27.
-/
theorem count_distinct_four_digit_numbers_divisible_by_3_ending_in_45 :
  ∃ n : ℕ, n = 27 ∧ 
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → 
  (∃ k : ℕ, a + b + 9 = 3 * k) → 
  (10 * (10 * a + b) + 45) = 1000 * a + 100 * b + 45 → 
  1000 * a + 100 * b + 45 = n := sorry

end count_distinct_four_digit_numbers_divisible_by_3_ending_in_45_l1305_130538


namespace calculate_expression_value_l1305_130544

theorem calculate_expression_value :
  5 * 7 + 6 * 9 + 13 * 2 + 4 * 6 = 139 :=
by
  -- proof can be added here
  sorry

end calculate_expression_value_l1305_130544


namespace find_integer_roots_l1305_130511

open Int Polynomial

def P (x : ℤ) : ℤ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_integer_roots : {x : ℤ | P x = 0} = {-3, 1, 5} := by
  sorry

end find_integer_roots_l1305_130511


namespace combined_distance_all_birds_two_seasons_l1305_130508

-- Definition of the given conditions
def number_of_birds : Nat := 20
def distance_jim_to_disney : Nat := 50
def distance_disney_to_london : Nat := 60

-- The conclusion we need to prove
theorem combined_distance_all_birds_two_seasons :
  (distance_jim_to_disney + distance_disney_to_london) * number_of_birds = 2200 :=
by
  sorry

end combined_distance_all_birds_two_seasons_l1305_130508


namespace Amy_bought_tomato_soup_l1305_130514

-- Conditions
variables (chicken_soup_cans total_soups : ℕ)
variable (Amy_bought_soups : total_soups = 9)
variable (Amy_bought_chicken_soup : chicken_soup_cans = 6)

-- Question: How many cans of tomato soup did she buy?
def cans_of_tomato_soup (chicken_soup_cans total_soups : ℕ) : ℕ :=
  total_soups - chicken_soup_cans

-- Theorem: Prove that the number of cans of tomato soup Amy bought is 3
theorem Amy_bought_tomato_soup : 
  cans_of_tomato_soup chicken_soup_cans total_soups = 3 :=
by
  rw [Amy_bought_soups, Amy_bought_chicken_soup]
  -- The steps for the proof would follow here
  sorry

end Amy_bought_tomato_soup_l1305_130514


namespace heartsuit_xx_false_l1305_130599

def heartsuit (x y : ℝ) : ℝ := |x - y|

theorem heartsuit_xx_false (x : ℝ) : heartsuit x x ≠ x :=
by sorry

end heartsuit_xx_false_l1305_130599


namespace calculate_DA_l1305_130590

open Real

-- Definitions based on conditions
def AU := 90
def AN := 180
def UB := 270
def AB := AU + UB
def ratio := 3 / 4

-- Statement of the problem in Lean 
theorem calculate_DA :
  ∃ (p q : ℕ), (q ≠ 0) ∧ (∀ p' q' : ℕ, ¬ (q = p'^2 * q')) ∧ DA = p * sqrt q ∧ p + q = result :=
  sorry

end calculate_DA_l1305_130590


namespace final_result_l1305_130536

-- Define the number of letters in each name
def letters_in_elida : ℕ := 5
def letters_in_adrianna : ℕ := 2 * letters_in_elida - 2

-- Define the alphabetical positions and their sums for each name
def sum_positions_elida : ℕ := 5 + 12 + 9 + 4 + 1
def sum_positions_adrianna : ℕ := 1 + 4 + 18 + 9 + 1 + 14 + 14 + 1
def sum_positions_belinda : ℕ := 2 + 5 + 12 + 9 + 14 + 4 + 1

-- Define the total sum of alphabetical positions
def total_sum_positions : ℕ := sum_positions_elida + sum_positions_adrianna + sum_positions_belinda

-- Define the average of the total sum
def average_sum_positions : ℕ := total_sum_positions / 3

-- Prove the final result
theorem final_result : (average_sum_positions * 3 - sum_positions_elida) = 109 :=
by
  -- Proof skipped
  sorry

end final_result_l1305_130536


namespace triangle_area_proof_l1305_130550

-- Conditions
variables (P r : ℝ) (semi_perimeter : ℝ)
-- The perimeter of the triangle is 40 cm
def perimeter_condition : Prop := P = 40
-- The inradius of the triangle is 2.5 cm
def inradius_condition : Prop := r = 2.5
-- The semi-perimeter is half of the perimeter
def semi_perimeter_def : Prop := semi_perimeter = P / 2

-- The area of the triangle
def area_of_triangle : ℝ := r * semi_perimeter

-- Proof Problem
theorem triangle_area_proof (hP : perimeter_condition P) (hr : inradius_condition r) (hsemi : semi_perimeter_def P semi_perimeter) :
  area_of_triangle r semi_perimeter = 50 :=
  sorry

end triangle_area_proof_l1305_130550


namespace prove_healthy_diet_multiple_l1305_130505

variable (rum_on_pancakes rum_earlier rum_after_pancakes : ℝ)
variable (healthy_multiple : ℝ)

-- Definitions from conditions
def Sally_gave_rum_on_pancakes : Prop := rum_on_pancakes = 10
def Don_had_rum_earlier : Prop := rum_earlier = 12
def Don_can_have_rum_after_pancakes : Prop := rum_after_pancakes = 8

-- Concluding multiple for healthy diet
def healthy_diet_multiple : Prop := healthy_multiple = (rum_on_pancakes + rum_after_pancakes - rum_earlier) / rum_on_pancakes

theorem prove_healthy_diet_multiple :
  Sally_gave_rum_on_pancakes rum_on_pancakes →
  Don_had_rum_earlier rum_earlier →
  Don_can_have_rum_after_pancakes rum_after_pancakes →
  healthy_diet_multiple rum_on_pancakes rum_earlier rum_after_pancakes healthy_multiple →
  healthy_multiple = 0.8 := 
by
  intros h1 h2 h3 h4
  sorry

end prove_healthy_diet_multiple_l1305_130505


namespace parallel_lines_not_coincident_l1305_130565

theorem parallel_lines_not_coincident (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 6 = 0) ∧
  (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ (b : ℝ), ∀ x y : ℝ, ax + 2 * y + b = 0 ∧ x + (a - 1) * y + b = 0 →
  a = -1 :=
by
  sorry

end parallel_lines_not_coincident_l1305_130565


namespace find_a_minus_b_l1305_130553

theorem find_a_minus_b (a b : ℤ) 
  (h1 : 3015 * a + 3019 * b = 3023) 
  (h2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := 
sorry

end find_a_minus_b_l1305_130553


namespace proj_b_l1305_130563

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  let factor := (ux * vx + uy * vy) / (vx * vx + vy * vy)
  (factor * vx, factor * vy)

theorem proj_b (a b v : ℝ × ℝ) (h_ortho : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj v a = (1, 2)) : proj v b = (3, -4) :=
by
  sorry

end proj_b_l1305_130563


namespace annie_total_miles_l1305_130552

theorem annie_total_miles (initial_gallons : ℕ) (miles_per_gallon : ℕ)
  (initial_trip_miles : ℕ) (purchased_gallons : ℕ) (final_gallons : ℕ)
  (total_miles : ℕ) :
  initial_gallons = 12 →
  miles_per_gallon = 28 →
  initial_trip_miles = 280 →
  purchased_gallons = 6 →
  final_gallons = 5 →
  total_miles = 364 := by
  sorry

end annie_total_miles_l1305_130552


namespace square_87_l1305_130506

theorem square_87 : 87^2 = 7569 :=
by
  sorry

end square_87_l1305_130506


namespace nina_total_miles_l1305_130532

noncomputable def totalDistance (warmUp firstHillUp firstHillDown firstRecovery 
                                 tempoRun secondHillUp secondHillDown secondRecovery 
                                 fartlek sprintsYards jogsBetweenSprints coolDown : ℝ) 
                                 (mileInYards : ℝ) : ℝ :=
  warmUp + 
  (firstHillUp + firstHillDown + firstRecovery) + 
  tempoRun + 
  (secondHillUp + secondHillDown + secondRecovery) + 
  fartlek + 
  (sprintsYards / mileInYards) + 
  jogsBetweenSprints + 
  coolDown

theorem nina_total_miles : 
  totalDistance 0.25 0.15 0.25 0.15 1.5 0.2 0.35 0.1 1.8 (8 * 50) (8 * 0.2) 0.3 1760 = 5.877 :=
by
  sorry

end nina_total_miles_l1305_130532


namespace tan_theta_correct_l1305_130594

noncomputable def tan_theta : Real :=
  let θ : Real := sorry
  if h_θ : θ ∈ Set.Ioc 0 (Real.pi / 4) then
    if h : Real.sin θ + Real.cos θ = 17 / 13 then
      Real.tan θ
    else
      0
  else
    0

theorem tan_theta_correct {θ : Real} (h_θ : θ ∈ Set.Ioc 0 (Real.pi / 4))
  (h : Real.sin θ + Real.cos θ = 17 / 13) :
  Real.tan θ = 5 / 12 := sorry

end tan_theta_correct_l1305_130594


namespace tan_alpha_value_cos2_minus_sin2_l1305_130516

variable (α : Real) 

axiom is_internal_angle (angle : Real) : angle ∈ Set.Ico 0 Real.pi 

axiom sin_cos_sum (α : Real) : α ∈ Set.Ico 0 Real.pi → Real.sin α + Real.cos α = 1 / 5

theorem tan_alpha_value (h : α ∈ Set.Ico 0 Real.pi) : Real.tan α = -4 / 3 := by 
  sorry

theorem cos2_minus_sin2 (h : Real.tan α = -4 / 3) : 1 / (Real.cos α^2 - Real.sin α^2) = -25 / 7 := by 
  sorry

end tan_alpha_value_cos2_minus_sin2_l1305_130516


namespace parallel_vectors_x_eq_one_l1305_130549

/-- Given vectors a = (2x + 1, 3) and b = (2 - x, 1), prove that if they 
are parallel, then x = 1. -/
theorem parallel_vectors_x_eq_one (x : ℝ) :
  (∃ k : ℝ, (2 * x + 1) = k * (2 - x) ∧ 3 = k * 1) → x = 1 :=
by 
  sorry

end parallel_vectors_x_eq_one_l1305_130549


namespace product_profit_equation_l1305_130515

theorem product_profit_equation (purchase_price selling_price : ℝ) 
                                (initial_units units_decrease_per_dollar_increase : ℝ)
                                (profit : ℝ)
                                (hx : purchase_price = 35)
                                (hy : selling_price = 40)
                                (hz : initial_units = 200)
                                (hs : units_decrease_per_dollar_increase = 5)
                                (hp : profit = 1870) :
  ∃ x : ℝ, (x + (selling_price - purchase_price)) * (initial_units - units_decrease_per_dollar_increase * x) = profit :=
by { sorry }

end product_profit_equation_l1305_130515


namespace smallest_angle_of_quadrilateral_l1305_130555

theorem smallest_angle_of_quadrilateral 
  (x : ℝ) 
  (h1 : x + 2 * x + 3 * x + 4 * x = 360) : 
  x = 36 :=
by
  sorry

end smallest_angle_of_quadrilateral_l1305_130555


namespace solitaire_game_end_with_one_piece_l1305_130548

theorem solitaire_game_end_with_one_piece (n : ℕ) : 
  ∃ (remaining_pieces : ℕ), 
  remaining_pieces = 1 ↔ n % 3 ≠ 0 :=
sorry

end solitaire_game_end_with_one_piece_l1305_130548


namespace variable_cost_per_book_l1305_130545

theorem variable_cost_per_book
  (F : ℝ) (S : ℝ) (N : ℕ) (V : ℝ)
  (fixed_cost : F = 56430) 
  (selling_price_per_book : S = 21.75) 
  (num_books : N = 4180) 
  (production_eq_sales : S * N = F + V * N) :
  V = 8.25 :=
by sorry

end variable_cost_per_book_l1305_130545


namespace math_more_than_reading_homework_l1305_130572

-- Definitions based on given conditions
def M : Nat := 9  -- Math homework pages
def R : Nat := 2  -- Reading homework pages

theorem math_more_than_reading_homework :
  M - R = 7 :=
by
  -- Proof would go here, showing that 9 - 2 indeed equals 7
  sorry

end math_more_than_reading_homework_l1305_130572


namespace value_of_expression_l1305_130504

theorem value_of_expression : 2 - (-5) = 7 :=
by
  sorry

end value_of_expression_l1305_130504


namespace part_a_l1305_130526

theorem part_a (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : a + b = 1) : a * b = 0 := 
by 
  sorry

end part_a_l1305_130526


namespace calculate_drift_l1305_130564

def width_of_river : ℕ := 400
def speed_of_boat : ℕ := 10
def time_to_cross : ℕ := 50
def actual_distance_traveled := speed_of_boat * time_to_cross

theorem calculate_drift : actual_distance_traveled - width_of_river = 100 :=
by
  -- width_of_river = 400
  -- speed_of_boat = 10
  -- time_to_cross = 50
  -- actual_distance_traveled = 10 * 50 = 500
  -- expected drift = 500 - 400 = 100
  sorry

end calculate_drift_l1305_130564


namespace range_of_u_l1305_130533

variable (a b u : ℝ)

theorem range_of_u (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∀ x : ℝ, x > 0 → a^2 + b^2 ≥ x ↔ x ≤ 16) :=
sorry

end range_of_u_l1305_130533


namespace total_distance_traveled_l1305_130593

theorem total_distance_traveled (x : ℕ) (d_1 d_2 d_3 d_4 d_5 d_6 : ℕ) 
  (h1 : d_1 = 60 / x) 
  (h2 : d_2 = 60 / (x + 3)) 
  (h3 : d_3 = 60 / (x + 6)) 
  (h4 : d_4 = 60 / (x + 9)) 
  (h5 : d_5 = 60 / (x + 12)) 
  (h6 : d_6 = 60 / (x + 15)) 
  (hx1 : x ∣ 60) 
  (hx2 : (x + 3) ∣ 60) 
  (hx3 : (x + 6) ∣ 60) 
  (hx4 : (x + 9) ∣ 60) 
  (hx5 : (x + 12) ∣ 60) 
  (hx6 : (x + 15) ∣ 60) :
  d_1 + d_2 + d_3 + d_4 + d_5 + d_6 = 39 := 
sorry

end total_distance_traveled_l1305_130593


namespace max_volume_of_box_l1305_130535

theorem max_volume_of_box (sheetside : ℝ) (cutside : ℝ) (volume : ℝ) 
  (h1 : sheetside = 6) 
  (h2 : ∀ (x : ℝ), 0 < x ∧ x < (sheetside / 2) → volume = x * (sheetside - 2 * x)^2) : 
  cutside = 1 :=
by
  sorry

end max_volume_of_box_l1305_130535


namespace solution_set_intersection_l1305_130527

theorem solution_set_intersection (a b : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, x^2 + x - 6 < 0 ↔ -3 < x ∧ x < 2) →
  (∀ x : ℝ, x^2 + a * x + b < 0 ↔ (-1 < x ∧ x < 2)) →
  a + b = -3 :=
by 
  sorry

end solution_set_intersection_l1305_130527


namespace final_answer_after_subtracting_l1305_130581

theorem final_answer_after_subtracting (n : ℕ) (h : n = 990) : (n / 9) - 100 = 10 :=
by
  sorry

end final_answer_after_subtracting_l1305_130581


namespace Maria_trip_time_l1305_130567

/-- 
Given:
- Maria drove 80 miles on a freeway.
- Maria drove 20 miles on a rural road.
- Her speed on the rural road was half of her speed on the freeway.
- Maria spent 40 minutes driving on the rural road.

Prove that Maria's entire trip took 120 minutes.
-/ 
theorem Maria_trip_time
  (distance_freeway : ℕ)
  (distance_rural : ℕ)
  (rural_speed_ratio : ℕ → ℕ)
  (time_rural_minutes : ℕ) 
  (time_freeway : ℕ)
  (total_time : ℕ) 
  (speed_rural : ℕ)
  (speed_freeway : ℕ) 
  :
  distance_freeway = 80 ∧
  distance_rural = 20 ∧ 
  rural_speed_ratio (speed_freeway) = speed_rural ∧ 
  time_rural_minutes = 40 ∧
  time_rural_minutes = 20 / speed_rural ∧
  speed_freeway = 2 * speed_rural ∧
  time_freeway = distance_freeway / speed_freeway ∧
  total_time = time_rural_minutes + time_freeway → 
  total_time = 120 :=
by
  intros
  sorry

end Maria_trip_time_l1305_130567


namespace table_relationship_l1305_130537

theorem table_relationship (x y : ℕ) (h : (x, y) ∈ [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125)]) : y = x^3 :=
sorry

end table_relationship_l1305_130537


namespace temperature_decrease_l1305_130521

-- Define the conditions
def temperature_rise (temp_increase: ℤ) : ℤ := temp_increase

-- Define the claim to be proved
theorem temperature_decrease (temp_decrease: ℤ) : temperature_rise 3 = 3 → temperature_rise (-6) = -6 :=
by
  sorry

end temperature_decrease_l1305_130521


namespace find_third_test_score_l1305_130569

-- Definitions of the given conditions
def test_score_1 := 80
def test_score_2 := 70
variable (x : ℕ) -- the unknown third score
def test_score_4 := 100
def average_score (s1 s2 s3 s4 : ℕ) : ℕ := (s1 + s2 + s3 + s4) / 4

-- Theorem stating that given the conditions, the third test score must be 90
theorem find_third_test_score (h : average_score test_score_1 test_score_2 x test_score_4 = 85) : x = 90 :=
by
  sorry

end find_third_test_score_l1305_130569


namespace lewis_total_earnings_l1305_130577

def Weekly_earnings : ℕ := 92
def Number_of_weeks : ℕ := 5

theorem lewis_total_earnings : Weekly_earnings * Number_of_weeks = 460 := by
  sorry

end lewis_total_earnings_l1305_130577


namespace valid_parameterizations_l1305_130542

theorem valid_parameterizations :
  (∀ t : ℝ, ∃ x y : ℝ, (x = 0 + 4 * t) ∧ (y = -4 + 8 * t) ∧ (y = 2 * x - 4)) ∧
  (∀ t : ℝ, ∃ x y : ℝ, (x = 3 + 1 * t) ∧ (y = 2 + 2 * t) ∧ (y = 2 * x - 4)) ∧
  (∀ t : ℝ, ∃ x y : ℝ, (x = -1 + 2 * t) ∧ (y = -6 + 4 * t) ∧ (y = 2 * x - 4)) :=
by
  -- Proof goes here
  sorry

end valid_parameterizations_l1305_130542


namespace speed_of_man_upstream_l1305_130562

-- Conditions stated as definitions 
def V_m : ℝ := 33 -- Speed of the man in still water
def V_downstream : ℝ := 40 -- Speed of the man rowing downstream

-- Required proof problem
theorem speed_of_man_upstream : V_m - (V_downstream - V_m) = 26 := 
by
  -- the following sorry is a placeholder for the actual proof
  sorry

end speed_of_man_upstream_l1305_130562


namespace intersection_A_B_l1305_130502

-- Define sets A and B according to the conditions provided
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Define the theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l1305_130502


namespace evaluate_expression_l1305_130557

theorem evaluate_expression (b : ℕ) (h : b = 5) : b^3 * b^4 * 2 = 156250 :=
by
  sorry

end evaluate_expression_l1305_130557


namespace complex_number_property_l1305_130500

theorem complex_number_property (i : ℂ) (h : i^2 = -1) : (1 + i)^(20) - (1 - i)^(20) = 0 :=
by {
  sorry
}

end complex_number_property_l1305_130500


namespace given_cond_then_geq_eight_l1305_130597

theorem given_cond_then_geq_eight (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 1) : 
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 := 
  sorry

end given_cond_then_geq_eight_l1305_130597


namespace total_customers_l1305_130592

def initial_customers : ℝ := 29.0    -- 29.0 initial customers
def lunch_rush_customers : ℝ := 20.0 -- Adds 20.0 customers during lunch rush
def additional_customers : ℝ := 34.0 -- Adds 34.0 more customers

theorem total_customers : (initial_customers + lunch_rush_customers + additional_customers) = 83.0 :=
by
  sorry

end total_customers_l1305_130592


namespace ellipse_triangle_perimeter_l1305_130578

-- Definitions based on conditions
def is_ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Triangle perimeter calculation
def triangle_perimeter (a c : ℝ) : ℝ := 2 * a + 2 * c

-- Main theorem statement
theorem ellipse_triangle_perimeter :
  let a := 2
  let b2 := 2
  let c := Real.sqrt (a ^ 2 - b2)
  ∀ (P : ℝ × ℝ), (is_ellipse P.1 P.2) → triangle_perimeter a c = 4 + 2 * Real.sqrt 2 :=
by
  intros P hP
  -- Here, we would normally provide the proof.
  sorry

end ellipse_triangle_perimeter_l1305_130578


namespace find_m_l1305_130551

theorem find_m (a : ℕ → ℝ) (m : ℝ)
  (h1 : (∀ (x : ℝ), x^2 + m * x - 8 = 0 → x = a 2 ∨ x = a 8))
  (h2 : a 4 + a 6 = a 5 ^ 2 + 1) :
  m = -2 :=
sorry

end find_m_l1305_130551


namespace parabola_focus_l1305_130525

theorem parabola_focus (f : ℝ) :
  (∀ x : ℝ, 2*x^2 = x^2 + (2*x^2 - f)^2 - (2*x^2 - -f)^2) →
  f = -1/8 :=
by sorry

end parabola_focus_l1305_130525


namespace diameter_of_circle_with_inscribed_right_triangle_l1305_130518

theorem diameter_of_circle_with_inscribed_right_triangle (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (right_triangle : a^2 + b^2 = c^2) : c = 10 :=
by
  subst h1
  subst h2
  simp at right_triangle
  sorry

end diameter_of_circle_with_inscribed_right_triangle_l1305_130518


namespace pies_with_no_ingredients_l1305_130583

theorem pies_with_no_ingredients (total_pies : ℕ)
  (pies_with_chocolate : ℕ)
  (pies_with_blueberries : ℕ)
  (pies_with_vanilla : ℕ)
  (pies_with_almonds : ℕ)
  (H_total : total_pies = 60)
  (H_chocolate : pies_with_chocolate = 1 / 3 * total_pies)
  (H_blueberries : pies_with_blueberries = 3 / 4 * total_pies)
  (H_vanilla : pies_with_vanilla = 2 / 5 * total_pies)
  (H_almonds : pies_with_almonds = 1 / 10 * total_pies) :
  ∃ (pies_without_ingredients : ℕ), pies_without_ingredients = 15 :=
by
  sorry

end pies_with_no_ingredients_l1305_130583


namespace evaluate_expression_l1305_130501

theorem evaluate_expression :
  -2 ^ 2005 + (-2) ^ 2006 + 2 ^ 2007 - 2 ^ 2008 = 2 ^ 2005 :=
by
  -- The following proof is left as an exercise.
  sorry

end evaluate_expression_l1305_130501


namespace dot_product_result_l1305_130522

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 1)

theorem dot_product_result : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_result_l1305_130522


namespace bertha_daughters_and_granddaughters_have_no_daughters_l1305_130570

def total_daughters_and_granddaughters (daughters granddaughters : Nat) : Nat :=
daughters + granddaughters

def no_daughters (bertha_daughters bertha_granddaughters : Nat) : Nat :=
bertha_daughters + bertha_granddaughters

theorem bertha_daughters_and_granddaughters_have_no_daughters :
  (bertha_daughters : Nat) →
  (daughters_with_6_daughters : Nat) →
  (granddaughters : Nat) →
  (total_daughters_and_granddaughters bertha_daughters granddaughters = 30) →
  bertha_daughters = 6 →
  granddaughters = 6 * daughters_with_6_daughters →
  no_daughters (bertha_daughters - daughters_with_6_daughters) granddaughters = 26 :=
by
  intros bertha_daughters daughters_with_6_daughters granddaughters h_total h_bertha h_granddaughters
  sorry

end bertha_daughters_and_granddaughters_have_no_daughters_l1305_130570


namespace reena_loan_l1305_130524

/-- 
  Problem setup:
  Reena took a loan of $1200 at simple interest for a period equal to the rate of interest years. 
  She paid $192 as interest at the end of the loan period.
  We aim to prove that the rate of interest is 4%. 
-/
theorem reena_loan (P : ℝ) (SI : ℝ) (R : ℝ) (N : ℝ) 
  (hP : P = 1200) 
  (hSI : SI = 192) 
  (hN : N = R) 
  (hSI_formula : SI = P * R * N / 100) : 
  R = 4 := 
by 
  sorry

end reena_loan_l1305_130524


namespace power_function_value_at_9_l1305_130519

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_value_at_9 (h : f 2 = Real.sqrt 2) : f 9 = 3 :=
by sorry

end power_function_value_at_9_l1305_130519


namespace intersection_complement_eq_l1305_130589

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≥ 2}

-- Define the intersection of A and complement of B
def intersection : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- The theorem to be proved
theorem intersection_complement_eq : (A ∩ complement_B) = intersection :=
sorry

end intersection_complement_eq_l1305_130589


namespace percentage_saved_l1305_130540

noncomputable def calculateSavedPercentage : ℚ :=
  let first_tier_free_tickets := 1
  let second_tier_free_tickets_per_ticket := 2
  let number_of_tickets_purchased := 10
  let total_free_tickets :=
    first_tier_free_tickets +
    (number_of_tickets_purchased - 5) * second_tier_free_tickets_per_ticket
  let total_tickets_received := number_of_tickets_purchased + total_free_tickets
  let free_tickets := total_tickets_received - number_of_tickets_purchased
  (free_tickets / total_tickets_received) * 100

theorem percentage_saved : calculateSavedPercentage = 52.38 :=
by
  sorry

end percentage_saved_l1305_130540


namespace find_sum_of_xyz_l1305_130598

theorem find_sum_of_xyz (x y z : ℕ) (h1 : 0 < x ∧ 0 < y ∧ 0 < z)
  (h2 : (x + y + z)^3 - x^3 - y^3 - z^3 = 300) : x + y + z = 7 :=
by
  sorry

end find_sum_of_xyz_l1305_130598


namespace jakes_digging_time_l1305_130556

theorem jakes_digging_time
  (J : ℕ)
  (Paul_work_rate : ℚ := 1/24)
  (Hari_work_rate : ℚ := 1/48)
  (Combined_work_rate : ℚ := 1/8)
  (Combined_work_eq : 1 / J + Paul_work_rate + Hari_work_rate = Combined_work_rate) :
  J = 16 := sorry

end jakes_digging_time_l1305_130556


namespace seventeenth_replacement_month_l1305_130588

def months_after_january (n : Nat) : Nat :=
  n % 12

theorem seventeenth_replacement_month :
  months_after_january (7 * 16) = 4 :=
by
  sorry

end seventeenth_replacement_month_l1305_130588


namespace fencing_required_l1305_130523

theorem fencing_required (L W : ℕ) (A : ℕ) 
  (hL : L = 20) 
  (hA : A = 680) 
  (hArea : A = L * W) : 
  2 * W + L = 88 := 
by 
  sorry

end fencing_required_l1305_130523


namespace words_to_score_A_l1305_130546

-- Define the total number of words
def total_words : ℕ := 600

-- Define the target percentage
def target_percentage : ℚ := 90 / 100

-- Define the minimum number of words to learn
def min_words_to_learn : ℕ := 540

-- Define the condition for scoring at least 90%
def meets_requirement (learned_words : ℕ) : Prop :=
  learned_words / total_words ≥ target_percentage

-- The goal is to prove that learning 540 words meets the requirement
theorem words_to_score_A : meets_requirement min_words_to_learn :=
by
  sorry

end words_to_score_A_l1305_130546


namespace least_xy_l1305_130512

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 9) : xy = 108 := by
  sorry

end least_xy_l1305_130512


namespace minimize_distance_l1305_130561

theorem minimize_distance
  (a b c d : ℝ)
  (h1 : (a + 3 * Real.log a) / b = 1)
  (h2 : (d - 3) / (2 * c) = 1) :
  (a - c)^2 + (b - d)^2 = (9 / 5) * (Real.log (Real.exp 1 / 3))^2 :=
by sorry

end minimize_distance_l1305_130561


namespace max_value_f_l1305_130543

open Real

/-- Determine the maximum value of the function f(x) = 1 / (1 - x * (1 - x)). -/
theorem max_value_f (x : ℝ) : 
  ∃ y, y = (1 / (1 - x * (1 - x))) ∧ y ≤ 4/3 ∧ ∀ z, z = (1 / (1 - x * (1 - x))) → z ≤ 4/3 :=
by
  sorry

end max_value_f_l1305_130543


namespace original_number_l1305_130510

theorem original_number (x : ℝ) (h : 1.35 * x = 680) : x = 503.70 :=
sorry

end original_number_l1305_130510


namespace value_of_X_l1305_130513

noncomputable def M : ℕ := 3009 / 3
noncomputable def N : ℕ := (2 * M) / 3
noncomputable def X : ℕ := M - N

theorem value_of_X : X = 335 := by
  sorry

end value_of_X_l1305_130513


namespace simplify_sqrt_450_l1305_130587

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l1305_130587


namespace solution_set_of_floor_equation_l1305_130586

theorem solution_set_of_floor_equation (x : ℝ) : 
  (⌊⌊2 * x⌋ - 1/2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
by sorry

end solution_set_of_floor_equation_l1305_130586


namespace radar_placement_and_coverage_area_l1305_130591

theorem radar_placement_and_coverage_area (r : ℝ) (w : ℝ) (n : ℕ) (h_radars : n = 5) (h_radius : r = 13) (h_width : w = 10) :
  let max_dist := 12 / Real.sin (Real.pi / 5)
  let area_ring := (240 * Real.pi) / Real.tan (Real.pi / 5)
  max_dist = 12 / Real.sin (Real.pi / 5) ∧ area_ring = (240 * Real.pi) / Real.tan (Real.pi / 5) :=
by
  sorry

end radar_placement_and_coverage_area_l1305_130591


namespace MariaTotalPaid_l1305_130534

-- Define a structure to hold the conditions
structure DiscountProblem where
  discount_rate : ℝ
  discount_amount : ℝ

-- Define the given discount problem specific to Maria
def MariaDiscountProblem : DiscountProblem :=
  { discount_rate := 0.25, discount_amount := 40 }

-- Define our goal: proving the total amount paid by Maria
theorem MariaTotalPaid (p : DiscountProblem) (h₀ : p = MariaDiscountProblem) :
  let original_price := p.discount_amount / p.discount_rate
  let total_paid := original_price - p.discount_amount
  total_paid = 120 :=
by
  sorry

end MariaTotalPaid_l1305_130534


namespace largest_n_unique_k_l1305_130507

-- Defining the main theorem statement
theorem largest_n_unique_k :
  ∃ (n : ℕ), (n = 63) ∧ (∃! (k : ℤ), (9 / 17 : ℚ) < (n : ℚ) / ((n + k) : ℚ) ∧ (n : ℚ) / ((n + k) : ℚ) < (8 / 15 : ℚ)) :=
sorry

end largest_n_unique_k_l1305_130507


namespace line_equation_min_intercepts_l1305_130574

theorem line_equation_min_intercepts (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : 1 / a + 4 / b = 1) : 2 * 1 + 4 - 6 = 0 ↔ (a = 3 ∧ b = 6) :=
by
  sorry

end line_equation_min_intercepts_l1305_130574


namespace polynomial_factorization_l1305_130559

theorem polynomial_factorization :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 = 5 * x^4 + 180 * x^3 + 1431 * x^2 + 4900 * x + 5159 :=
by sorry

end polynomial_factorization_l1305_130559


namespace sale_price_is_207_l1305_130576

-- Definitions for the conditions given
def price_at_store_P : ℝ := 200
def regular_price_at_store_Q (price_P : ℝ) : ℝ := price_P * 1.15
def sale_price_at_store_Q (regular_price_Q : ℝ) : ℝ := regular_price_Q * 0.90

-- Goal: Prove the sale price of the bicycle at Store Q is 207
theorem sale_price_is_207 : sale_price_at_store_Q (regular_price_at_store_Q price_at_store_P) = 207 :=
by
  sorry

end sale_price_is_207_l1305_130576


namespace rectangle_area_solution_l1305_130596

theorem rectangle_area_solution (x : ℝ) (h1 : (x + 3) * (2*x - 1) = 12*x + 5) : 
  x = (7 + Real.sqrt 113) / 4 :=
by 
  sorry

end rectangle_area_solution_l1305_130596


namespace min_value_a_plus_3b_l1305_130566

theorem min_value_a_plus_3b (a b : ℝ) (h_positive : 0 < a ∧ 0 < b)
  (h_condition : (1 / (a + 3) + 1 / (b + 3) = 1 / 4)) :
  a + 3 * b ≥ 4 + 8 * Real.sqrt 3 := 
sorry

end min_value_a_plus_3b_l1305_130566


namespace greatest_three_digit_number_l1305_130571

theorem greatest_three_digit_number
  (n : ℕ) (h_3digit : 100 ≤ n ∧ n < 1000) (h_mod7 : n % 7 = 2) (h_mod4 : n % 4 = 1) :
  n = 989 :=
sorry

end greatest_three_digit_number_l1305_130571


namespace part_a_part_b_l1305_130517

def can_cut_into_equal_dominoes (n : ℕ) : Prop :=
  ∃ horiz_vert_dominoes : ℕ × ℕ,
    n % 2 = 1 ∧
    (n * n - 1) / 2 = horiz_vert_dominoes.1 + horiz_vert_dominoes.2 ∧
    horiz_vert_dominoes.1 = horiz_vert_dominoes.2

theorem part_a : can_cut_into_equal_dominoes 101 :=
by {
  sorry
}

theorem part_b : ¬can_cut_into_equal_dominoes 99 :=
by {
  sorry
}

end part_a_part_b_l1305_130517


namespace geometric_sequence_sum_ratio_l1305_130585

theorem geometric_sequence_sum_ratio 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n+1) = a 0 * q ^ n)
  (h2 : ∀ n, S n = (a 0 * (q ^ n - 1)) / (q - 1))
  (h3 : 6 * a 3 = a 0 * q ^ 5 - a 0 * q ^ 4) :
  S 4 / S 2 = 10 := 
sorry

end geometric_sequence_sum_ratio_l1305_130585


namespace required_earnings_correct_l1305_130558

-- Definitions of the given conditions
def retail_price : ℝ := 600
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def amount_saved : ℝ := 120
def amount_given_by_mother : ℝ := 250
def additional_costs : ℝ := 50

-- Required amount Maria must earn
def required_earnings : ℝ := 247

-- Lean 4 theorem statement
theorem required_earnings_correct :
  let discount_amount := discount_rate * retail_price
  let discounted_price := retail_price - discount_amount
  let sales_tax_amount := sales_tax_rate * discounted_price
  let total_bike_cost := discounted_price + sales_tax_amount
  let total_cost := total_bike_cost + additional_costs
  let total_have := amount_saved + amount_given_by_mother
  required_earnings = total_cost - total_have :=
by
  sorry

end required_earnings_correct_l1305_130558


namespace ian_says_1306_l1305_130509

noncomputable def number_i_say := 4 * (4 * (4 * (4 * (4 * (4 * (4 * (4 * 1 - 2) - 2) - 2) - 2) - 2) - 2) - 2) - 2

theorem ian_says_1306 (n : ℕ) : 1 ≤ n ∧ n ≤ 2000 → n = 1306 :=
by sorry

end ian_says_1306_l1305_130509


namespace find_number_l1305_130529

-- Define the number x and the condition as a theorem to be proven.
theorem find_number (x : ℝ) (h : (1/3) * x - 5 = 10) : x = 45 :=
sorry

end find_number_l1305_130529


namespace height_of_tree_in_kilmer_park_l1305_130575

-- Define the initial conditions
def initial_height_ft := 52
def growth_per_year_ft := 5
def years := 8
def ft_to_inch := 12

-- Define the expected result in inches
def expected_height_inch := 1104

-- State the problem as a theorem
theorem height_of_tree_in_kilmer_park :
  (initial_height_ft + growth_per_year_ft * years) * ft_to_inch = expected_height_inch :=
by
  sorry

end height_of_tree_in_kilmer_park_l1305_130575


namespace security_deposit_percentage_l1305_130530

theorem security_deposit_percentage
    (daily_rate : ℝ) (pet_fee : ℝ) (service_fee_rate : ℝ) (days : ℝ) (security_deposit : ℝ)
    (total_cost : ℝ) (expected_percentage : ℝ) :
    daily_rate = 125.0 →
    pet_fee = 100.0 →
    service_fee_rate = 0.20 →
    days = 14 →
    security_deposit = 1110 →
    total_cost = daily_rate * days + pet_fee + (daily_rate * days + pet_fee) * service_fee_rate →
    expected_percentage = (security_deposit / total_cost) * 100 →
    expected_percentage = 50 :=
by
  intros
  sorry

end security_deposit_percentage_l1305_130530


namespace function_inverse_necessary_not_sufficient_l1305_130579

theorem function_inverse_necessary_not_sufficient (f : ℝ → ℝ) :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, g (f x) = x ∧ f (g x) = x) →
  ¬ (∀ (x y : ℝ), x < y → f x < f y) :=
by
  sorry

end function_inverse_necessary_not_sufficient_l1305_130579


namespace intersect_sets_l1305_130580

def set_M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_N : Set ℝ := {x | abs x < 2}

theorem intersect_sets :
  (set_M ∩ set_N) = {x | -1 ≤ x ∧ x < 2} :=
sorry

end intersect_sets_l1305_130580


namespace tom_fruit_bowl_l1305_130520

def initial_lemons (oranges lemons removed remaining : ℕ) : ℕ :=
  lemons

theorem tom_fruit_bowl (oranges removed remaining : ℕ) (L : ℕ) 
  (h_oranges : oranges = 3)
  (h_removed : removed = 3)
  (h_remaining : remaining = 6)
  (h_initial : oranges + L - removed = remaining) : 
  initial_lemons oranges L removed remaining = 6 :=
by
  -- Implement the proof here
  sorry

end tom_fruit_bowl_l1305_130520
