import Mathlib

namespace NUMINAMATH_GPT_length_of_wall_l1938_193824

-- Define the dimensions of a brick
def brick_length : ℝ := 40
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the dimensions of the wall
def wall_height : ℝ := 600
def wall_width : ℝ := 22.5

-- Define the required number of bricks
def required_bricks : ℝ := 4000

-- Calculate the volume of a single brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Calculate the volume of the wall
def volume_wall (length : ℝ) : ℝ := length * wall_height * wall_width

-- The theorem to prove
theorem length_of_wall : ∃ (L : ℝ), required_bricks * volume_brick = volume_wall L → L = 800 :=
sorry

end NUMINAMATH_GPT_length_of_wall_l1938_193824


namespace NUMINAMATH_GPT_radius_of_two_equal_circles_eq_16_l1938_193876

noncomputable def radius_of_congruent_circles : ℝ := 16

theorem radius_of_two_equal_circles_eq_16 :
  ∃ x : ℝ, 
    (∀ r1 r2 r3 : ℝ, r1 = 4 ∧ r2 = r3 ∧ r2 = x ∧ 
    ∃ line : ℝ → ℝ → Prop, 
    (line 0 r1) ∧ (line 0 r2)  ∧ 
    (line 0 r3) ∧ 
    (line r2 r3) ∧
    (line r1 r2)  ∧ (line r1 r3) ∧ (line (r1 + r2) r2) ) 
    → x = 16 := sorry

end NUMINAMATH_GPT_radius_of_two_equal_circles_eq_16_l1938_193876


namespace NUMINAMATH_GPT_tape_recorder_cost_l1938_193839

-- Define the conditions
def conditions (x p : ℚ) : Prop :=
  170 < p ∧ p < 195 ∧
  2 * p = x * (x - 2) ∧
  1 * x = x - 2 + 2

-- Define the statement to be proved
theorem tape_recorder_cost (x : ℚ) (p : ℚ) : conditions x p → p = 180 := by
  sorry

end NUMINAMATH_GPT_tape_recorder_cost_l1938_193839


namespace NUMINAMATH_GPT_sin_double_angle_l1938_193811

open Real

theorem sin_double_angle (θ : ℝ) (h : cos (π / 4 - θ) = 1 / 2) : sin (2 * θ) = -1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1938_193811


namespace NUMINAMATH_GPT_area_of_quadrilateral_l1938_193822

-- Definitions of the given conditions
def diagonal_length : ℝ := 40
def offset1 : ℝ := 11
def offset2 : ℝ := 9

-- The area of the quadrilateral
def quadrilateral_area : ℝ := 400

-- Proof statement
theorem area_of_quadrilateral :
  (1/2 * diagonal_length * offset1 + 1/2 * diagonal_length * offset2) = quadrilateral_area :=
by sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l1938_193822


namespace NUMINAMATH_GPT_simplify_divide_expression_l1938_193834

noncomputable def a : ℝ := Real.sqrt 2 + 1

theorem simplify_divide_expression : 
  (1 - (a / (a + 1))) / ((a^2 - 1) / (a^2 + 2 * a + 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_divide_expression_l1938_193834


namespace NUMINAMATH_GPT_parabola_incorrect_statement_B_l1938_193828

theorem parabola_incorrect_statement_B 
  (y₁ y₂ : ℝ → ℝ) 
  (h₁ : ∀ x, y₁ x = 2 * x^2) 
  (h₂ : ∀ x, y₂ x = -2 * x^2) : 
  ¬ (∀ x < 0, y₁ x < y₁ (x + 1)) ∧ (∀ x < 0, y₂ x < y₂ (x + 1)) := 
by 
  sorry

end NUMINAMATH_GPT_parabola_incorrect_statement_B_l1938_193828


namespace NUMINAMATH_GPT_bill_needs_paint_cans_l1938_193886

theorem bill_needs_paint_cans :
  let bedrooms := 3
  let other_rooms := 2 * bedrooms
  let gallons_per_room := 2
  let color_paint_cans := 6 -- (bedrooms * gallons_per_room) / 1-gallon per can
  let white_paint_cans := 4 -- (other_rooms * gallons_per_room) / 3-gallons per can
  (color_paint_cans + white_paint_cans) = 10 := sorry

end NUMINAMATH_GPT_bill_needs_paint_cans_l1938_193886


namespace NUMINAMATH_GPT_length_of_greater_segment_l1938_193899

theorem length_of_greater_segment (x : ℤ) (h1 : (x + 2)^2 - x^2 = 32) : x + 2 = 9 := by
  sorry

end NUMINAMATH_GPT_length_of_greater_segment_l1938_193899


namespace NUMINAMATH_GPT_total_cost_correct_l1938_193857

noncomputable def total_cost (sandwiches: ℕ) (price_per_sandwich: ℝ) (sodas: ℕ) (price_per_soda: ℝ) (discount: ℝ) (tax: ℝ) : ℝ :=
  let total_sandwich_cost := sandwiches * price_per_sandwich
  let total_soda_cost := sodas * price_per_soda
  let discounted_sandwich_cost := total_sandwich_cost * (1 - discount)
  let total_before_tax := discounted_sandwich_cost + total_soda_cost
  let total_with_tax := total_before_tax * (1 + tax)
  total_with_tax

theorem total_cost_correct : 
  total_cost 2 3.49 4 0.87 0.10 0.05 = 10.25 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1938_193857


namespace NUMINAMATH_GPT_croissant_process_time_in_hours_l1938_193852

-- Conditions as definitions
def num_folds : ℕ := 4
def fold_time : ℕ := 5
def rest_time : ℕ := 75
def mix_time : ℕ := 10
def bake_time : ℕ := 30

-- The main theorem statement
theorem croissant_process_time_in_hours :
  (num_folds * (fold_time + rest_time) + mix_time + bake_time) / 60 = 6 := 
sorry

end NUMINAMATH_GPT_croissant_process_time_in_hours_l1938_193852


namespace NUMINAMATH_GPT_joes_speed_l1938_193889

theorem joes_speed (pete_speed : ℝ) (joe_speed : ℝ) (time_minutes : ℝ) (distance : ℝ) (h1 : joe_speed = 2 * pete_speed) (h2 : time_minutes = 40) (h3 : distance = 16) : joe_speed = 16 :=
by
  sorry

end NUMINAMATH_GPT_joes_speed_l1938_193889


namespace NUMINAMATH_GPT_line_through_parabola_vertex_unique_value_l1938_193809

theorem line_through_parabola_vertex_unique_value :
  ∃! a : ℝ, ∃ y : ℝ, y = x + a ∧ y = x^2 - 2*a*x + a^2 :=
sorry

end NUMINAMATH_GPT_line_through_parabola_vertex_unique_value_l1938_193809


namespace NUMINAMATH_GPT_positive_difference_of_complementary_angles_in_ratio_five_to_four_l1938_193829

theorem positive_difference_of_complementary_angles_in_ratio_five_to_four
  (a b : ℝ)
  (h1 : a / b = 5 / 4)
  (h2 : a + b = 90) :
  |a - b| = 10 :=
sorry

end NUMINAMATH_GPT_positive_difference_of_complementary_angles_in_ratio_five_to_four_l1938_193829


namespace NUMINAMATH_GPT_inverse_value_ratio_l1938_193868

noncomputable def g (x : ℚ) : ℚ := (3 * x + 1) / (x - 4)

theorem inverse_value_ratio :
  (∃ (a b c d : ℚ), ∀ x, g ((a * x + b) / (c * x + d)) = x) → ∃ a c : ℚ, a / c = -4 :=
by
  sorry

end NUMINAMATH_GPT_inverse_value_ratio_l1938_193868


namespace NUMINAMATH_GPT_problem_inequality_solution_l1938_193855

noncomputable def find_b_and_c (x : ℝ) (b c : ℝ) : Prop :=
  ∀ x, (x > 2 ∨ x < 1) ↔ x^2 + b*x + c > 0

theorem problem_inequality_solution (x : ℝ) :
  find_b_and_c x (-3) 2 ∧ (2*x^2 - 3*x + 1 ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_inequality_solution_l1938_193855


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1938_193898

-- Define the geometric mean condition between 2 and 8
def is_geometric_mean (m : ℝ) := m = 4 ∨ m = -4

-- Prove that m = 4 is a necessary but not sufficient condition for is_geometric_mean
theorem necessary_but_not_sufficient (m : ℝ) :
  (is_geometric_mean m) ↔ (m = 4) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1938_193898


namespace NUMINAMATH_GPT_student_tickets_sold_l1938_193830

theorem student_tickets_sold (S NS : ℕ) (h1 : S + NS = 150) (h2 : 5 * S + 8 * NS = 930) : S = 90 :=
by
  sorry

end NUMINAMATH_GPT_student_tickets_sold_l1938_193830


namespace NUMINAMATH_GPT_number_of_people_l1938_193896

theorem number_of_people (total_bowls : ℕ) (bowls_per_person : ℚ) : total_bowls = 55 ∧ bowls_per_person = 1 + 1/2 + 1/3 → total_bowls / bowls_per_person = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_l1938_193896


namespace NUMINAMATH_GPT_max_expression_tends_to_infinity_l1938_193858

noncomputable def maximize_expression (x y z : ℝ) : ℝ :=
  1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) + 1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))

theorem max_expression_tends_to_infinity : 
  ∀ (x y z : ℝ), -1 < x ∧ x < 1 ∧ -1 < y ∧ y < 1 ∧ -1 < z ∧ z < 1 → 
    ∃ M : ℝ, maximize_expression x y z > M :=
by
  intro x y z h
  sorry

end NUMINAMATH_GPT_max_expression_tends_to_infinity_l1938_193858


namespace NUMINAMATH_GPT_taller_building_height_l1938_193891

theorem taller_building_height
  (H : ℕ) -- H is the height of the taller building
  (h_ratio : (H - 36) / H = 5 / 7) -- heights ratio condition
  (h_diff : H > 36) -- height difference must respect physics
  : H = 126 := sorry

end NUMINAMATH_GPT_taller_building_height_l1938_193891


namespace NUMINAMATH_GPT_cubes_divisible_by_nine_l1938_193848

theorem cubes_divisible_by_nine (n : ℕ) (hn : n > 0) : 
    (n^3 + (n + 1)^3 + (n + 2)^3) % 9 = 0 := by
  sorry

end NUMINAMATH_GPT_cubes_divisible_by_nine_l1938_193848


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l1938_193826

theorem common_difference_arithmetic_sequence
  (a : ℕ → ℝ)
  (h1 : ∃ a1 d, (∀ n, a n = a1 + (n - 1) * d))
  (h2 : a 7 - 2 * a 4 = -1)
  (h3 : a 3 = 0) :
  ∃ d, (∀ a1, (a1 + 2 * d = 0 ∧ -d = -1) → d = -1/2) :=
by
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l1938_193826


namespace NUMINAMATH_GPT_sequence_x_y_sum_l1938_193841

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end NUMINAMATH_GPT_sequence_x_y_sum_l1938_193841


namespace NUMINAMATH_GPT_total_marbles_l1938_193813

-- There are only red, blue, and yellow marbles
universe u
variable {α : Type u}

-- The ratio of red marbles to blue marbles to yellow marbles is \(2:3:4\)
variables {r b y T : ℕ}
variable (ratio_cond : 2 * y = 4 * r ∧ 3 * y = 4 * b)

-- There are 40 yellow marbles in the container
variable (yellow_cond : y = 40)

-- Prove the total number of marbles in the container is 90
theorem total_marbles (ratio_cond : 2 * y = 4 * r ∧ 3 * y = 4 * b) (yellow_cond : y = 40) :
  T = r + b + y → T = 90 :=
sorry

end NUMINAMATH_GPT_total_marbles_l1938_193813


namespace NUMINAMATH_GPT_distance_origin_to_line_l1938_193881

theorem distance_origin_to_line : 
  let A := 1
  let B := Real.sqrt 3
  let C := -2
  let x1 := 0
  let y1 := 0
  let distance := |A*x1 + B*y1 + C| / Real.sqrt (A^2 + B^2)
  distance = 1 :=
by 
  let A := 1
  let B := Real.sqrt 3
  let C := -2
  let x1 := 0
  let y1 := 0
  let distance := |A*x1 + B*y1 + C| / Real.sqrt (A^2 + B^2)
  sorry

end NUMINAMATH_GPT_distance_origin_to_line_l1938_193881


namespace NUMINAMATH_GPT_point_A_coords_l1938_193880

theorem point_A_coords (x y : ℝ) (h : ∀ t : ℝ, (t + 1) * x - (2 * t + 5) * y - 6 = 0) : x = -4 ∧ y = -2 := by
  sorry

end NUMINAMATH_GPT_point_A_coords_l1938_193880


namespace NUMINAMATH_GPT_parabola_properties_l1938_193831

theorem parabola_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  (∀ x, a * x^2 + b * x + c >= a * (x^2)) ∧
  (c < 0) ∧ 
  (-b / (2 * a) < 0) :=
by
  sorry

end NUMINAMATH_GPT_parabola_properties_l1938_193831


namespace NUMINAMATH_GPT_range_of_t_l1938_193856

-- Define set A and set B as conditions
def setA := { x : ℝ | -3 < x ∧ x < 7 }
def setB (t : ℝ) := { x : ℝ | t + 1 < x ∧ x < 2 * t - 1 }

-- Lean statement to prove the range of t
theorem range_of_t (t : ℝ) : setB t ⊆ setA → t ≤ 4 :=
by
  -- sorry acts as a placeholder for the proof
  sorry

end NUMINAMATH_GPT_range_of_t_l1938_193856


namespace NUMINAMATH_GPT_students_passed_correct_l1938_193823

-- Define the number of students in ninth grade.
def students_total : ℕ := 180

-- Define the number of students who bombed their finals.
def students_bombed : ℕ := students_total / 4

-- Define the number of students remaining after removing those who bombed.
def students_remaining_after_bombed : ℕ := students_total - students_bombed

-- Define the number of students who didn't show up to take the test.
def students_didnt_show : ℕ := students_remaining_after_bombed / 3

-- Define the number of students remaining after removing those who didn't show up.
def students_remaining_after_no_show : ℕ := students_remaining_after_bombed - students_didnt_show

-- Define the number of students who got less than a D.
def students_less_than_d : ℕ := 20

-- Define the number of students who passed.
def students_passed : ℕ := students_remaining_after_no_show - students_less_than_d

-- Statement to prove the number of students who passed is 70.
theorem students_passed_correct : students_passed = 70 := by
  -- Proof will be inserted here.
  sorry

end NUMINAMATH_GPT_students_passed_correct_l1938_193823


namespace NUMINAMATH_GPT_abcd_sum_is_12_l1938_193817

theorem abcd_sum_is_12 (a b c d : ℤ) 
  (h1 : a + c = 2) 
  (h2 : a * c + b + d = -1) 
  (h3 : a * d + b * c = 18) 
  (h4 : b * d = 24) : 
  a + b + c + d = 12 :=
sorry

end NUMINAMATH_GPT_abcd_sum_is_12_l1938_193817


namespace NUMINAMATH_GPT_paper_clips_in_two_cases_l1938_193862

theorem paper_clips_in_two_cases (c b : ℕ) : 
  2 * c * b * 200 = 2 * (c * b * 200) :=
by
  sorry

end NUMINAMATH_GPT_paper_clips_in_two_cases_l1938_193862


namespace NUMINAMATH_GPT_billy_avoids_swimming_n_eq_2022_billy_wins_for_odd_n_billy_wins_for_even_n_l1938_193842

theorem billy_avoids_swimming_n_eq_2022 :
  ∀ n : ℕ, n = 2022 → (∃ (strategy : ℕ → ℕ), ∀ k, strategy (2022 + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

theorem billy_wins_for_odd_n (n : ℕ) (h : n > 10 ∧ n % 2 = 1) :
  ∃ (strategy : ℕ → ℕ), (∀ k, strategy (n + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

theorem billy_wins_for_even_n (n : ℕ) (h : n > 10 ∧ n % 2 = 0) :
  ∃ (strategy : ℕ → ℕ), (∀ k, strategy (n + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

end NUMINAMATH_GPT_billy_avoids_swimming_n_eq_2022_billy_wins_for_odd_n_billy_wins_for_even_n_l1938_193842


namespace NUMINAMATH_GPT_range_of_m_l1938_193825

variable (a b : ℝ)

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 1, x^3 - m ≤ a * x + b ∧ a * x + b ≤ x^3 + m) ↔ m ∈ Set.Ici (Real.sqrt 3 / 9) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1938_193825


namespace NUMINAMATH_GPT_pencils_lost_l1938_193854

theorem pencils_lost (bought_pencils remaining_pencils lost_pencils : ℕ)
                     (h1 : bought_pencils = 16)
                     (h2 : remaining_pencils = 8)
                     (h3 : lost_pencils = bought_pencils - remaining_pencils) :
                     lost_pencils = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_pencils_lost_l1938_193854


namespace NUMINAMATH_GPT_shaded_area_l1938_193894

theorem shaded_area (whole_squares partial_squares : ℕ) (area_whole area_partial : ℝ)
  (h1 : whole_squares = 5)
  (h2 : partial_squares = 6)
  (h3 : area_whole = 1)
  (h4 : area_partial = 0.5) :
  (whole_squares * area_whole + partial_squares * area_partial) = 8 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l1938_193894


namespace NUMINAMATH_GPT_scarf_cost_is_10_l1938_193871

-- Define the conditions as given in the problem statement
def initial_amount : ℕ := 53
def cost_per_toy_car : ℕ := 11
def num_toy_cars : ℕ := 2
def cost_of_beanie : ℕ := 14
def remaining_after_beanie : ℕ := 7

-- Calculate the cost of the toy cars
def total_cost_toy_cars : ℕ := num_toy_cars * cost_per_toy_car

-- Calculate the amount left after buying the toy cars
def amount_after_toys : ℕ := initial_amount - total_cost_toy_cars

-- Calculate the amount left after buying the beanie
def amount_after_beanie : ℕ := amount_after_toys - cost_of_beanie

-- Define the cost of the scarf
def cost_of_scarf : ℕ := amount_after_beanie - remaining_after_beanie

-- The theorem stating that cost_of_scarf is 10 dollars
theorem scarf_cost_is_10 : cost_of_scarf = 10 := by
  sorry

end NUMINAMATH_GPT_scarf_cost_is_10_l1938_193871


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l1938_193892

theorem first_term_of_geometric_series (a r : ℝ) 
    (h1 : a / (1 - r) = 18) 
    (h2 : a^2 / (1 - r^2) = 72) : 
    a = 72 / 11 := 
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l1938_193892


namespace NUMINAMATH_GPT_fifteenth_number_with_digit_sum_15_is_294_l1938_193897

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def numbers_with_digit_sum (s : ℕ) : List ℕ :=
  List.filter (λ n => digit_sum n = s) (List.range (10 ^ 3)) -- Assume a maximum of 3-digit numbers

def fifteenth_number_with_digit_sum (s : ℕ) : ℕ :=
  (numbers_with_digit_sum s).get! 14 -- Get the 15th element (0-indexed)

theorem fifteenth_number_with_digit_sum_15_is_294 : fifteenth_number_with_digit_sum 15 = 294 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_fifteenth_number_with_digit_sum_15_is_294_l1938_193897


namespace NUMINAMATH_GPT_quotient_base4_l1938_193843

def base4_to_base10 (n : ℕ) : ℕ :=
  n % 10 + 4 * (n / 10 % 10) + 4^2 * (n / 100 % 10) + 4^3 * (n / 1000)

def base10_to_base4 (n : ℕ) : ℕ :=
  let rec convert (n acc : ℕ) : ℕ :=
    if n < 4 then n * acc
    else convert (n / 4) ((n % 4) * acc * 10 + acc)
  convert n 1

theorem quotient_base4 (a b : ℕ) (h1 : a = 2313) (h2 : b = 13) :
  base10_to_base4 ((base4_to_base10 a) / (base4_to_base10 b)) = 122 :=
by
  sorry

end NUMINAMATH_GPT_quotient_base4_l1938_193843


namespace NUMINAMATH_GPT_second_player_wins_l1938_193866

theorem second_player_wins : 
  ∀ (a b c : ℝ), (a ≠ 0) → 
  (∃ (first_choice: ℝ), ∃ (second_choice: ℝ), 
    ∃ (third_choice: ℝ), 
    ((first_choice ≠ 0) → (b^2 + 4 * first_choice^2 > 0)) ∧ 
    ((first_choice = 0) → (b ≠ 0)) ∧ 
    first_choice * (first_choice * b + a) = 0 ↔ ∃ x : ℝ, a * x^2 + (first_choice + second_choice) * x + third_choice = 0) :=
by sorry

end NUMINAMATH_GPT_second_player_wins_l1938_193866


namespace NUMINAMATH_GPT_primes_dividing_expression_l1938_193840

theorem primes_dividing_expression (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  6 * p * q ∣ p^3 + q^2 + 38 ↔ (p = 3 ∧ (q = 5 ∨ q = 13)) := 
sorry

end NUMINAMATH_GPT_primes_dividing_expression_l1938_193840


namespace NUMINAMATH_GPT_odd_function_at_zero_l1938_193806

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_at_zero (f : ℝ → ℝ) (h : is_odd_function f) : f 0 = 0 :=
by
  -- assume the definitions but leave the proof steps and focus on the final conclusion
  sorry

end NUMINAMATH_GPT_odd_function_at_zero_l1938_193806


namespace NUMINAMATH_GPT_rectangular_prism_dimensions_l1938_193837

theorem rectangular_prism_dimensions (a b c : ℤ) (h1: c = (a * b) / 2) (h2: 2 * (a * b + b * c + c * a) = a * b * c) :
  (a = 3 ∧ b = 10 ∧ c = 15) ∨ (a = 4 ∧ b = 6 ∧ c = 12) :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangular_prism_dimensions_l1938_193837


namespace NUMINAMATH_GPT_determinant_scaled_l1938_193807

variables (x y z w : ℝ)
variables (det : ℝ)

-- Given condition: determinant of the 2x2 matrix is 7.
axiom det_given : det = x * w - y * z
axiom det_value : det = 7

-- The target to be proven: the determinant of the scaled matrix is 63.
theorem determinant_scaled (x y z w : ℝ) (det : ℝ) (h_det : det = x * w - y * z) (det_value : det = 7) : 
  3 * 3 * (x * w - y * z) = 63 :=
by
  sorry

end NUMINAMATH_GPT_determinant_scaled_l1938_193807


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1938_193846

-- Define the theorem for the first equation
theorem solve_eq1 (x : ℝ) (h : 2 * x - 7 = 5 * x - 1) : x = -2 :=
sorry

-- Define the theorem for the second equation
theorem solve_eq2 (x : ℝ) (h : (x - 2) / 2 - (x - 1) / 6 = 1) : x = 11 / 2 :=
sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1938_193846


namespace NUMINAMATH_GPT_factorize_polynomial_l1938_193815

noncomputable def polynomial_factorization : Prop :=
  ∀ x : ℤ, (x^12 + x^9 + 1) = (x^4 + x^3 + x^2 + x + 1) * (x^8 - x^7 + x^6 - x^5 + x^3 - x^2 + x - 1)

theorem factorize_polynomial : polynomial_factorization :=
by
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l1938_193815


namespace NUMINAMATH_GPT_no_real_solutions_to_equation_l1938_193870

theorem no_real_solutions_to_equation :
  ¬ ∃ y : ℝ, (3 * y - 4)^2 + 4 = -(y + 3) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_to_equation_l1938_193870


namespace NUMINAMATH_GPT_no_solution_exists_l1938_193836

theorem no_solution_exists (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : ¬(2 / a + 2 / b = 1 / (a + b)) :=
sorry

end NUMINAMATH_GPT_no_solution_exists_l1938_193836


namespace NUMINAMATH_GPT_evaluate_expression_l1938_193849

theorem evaluate_expression (x : ℝ) : (x+2)^2 + 2*(x+2)*(4-x) + (4-x)^2 = 36 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1938_193849


namespace NUMINAMATH_GPT_escalator_time_l1938_193805

theorem escalator_time
    {d i s : ℝ}
    (h1 : d = 90 * i)
    (h2 : d = 30 * (i + s))
    (h3 : s = 2 * i):
    d / s = 45 := by
  sorry

end NUMINAMATH_GPT_escalator_time_l1938_193805


namespace NUMINAMATH_GPT_percentage_of_boys_l1938_193800

theorem percentage_of_boys (total_students boys_per_group girls_per_group : ℕ)
  (ratio_condition : boys_per_group + girls_per_group = 7)
  (total_condition : total_students = 42)
  (ratio_b_condition : boys_per_group = 3)
  (ratio_g_condition : girls_per_group = 4) :
  (boys_per_group : ℚ) / (boys_per_group + girls_per_group : ℚ) * 100 = 42.86 :=
by sorry

end NUMINAMATH_GPT_percentage_of_boys_l1938_193800


namespace NUMINAMATH_GPT_largest_non_summable_composite_l1938_193865

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end NUMINAMATH_GPT_largest_non_summable_composite_l1938_193865


namespace NUMINAMATH_GPT_cosine_120_eq_neg_one_half_l1938_193878

theorem cosine_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1/2 :=
by
-- Proof omitted
sorry

end NUMINAMATH_GPT_cosine_120_eq_neg_one_half_l1938_193878


namespace NUMINAMATH_GPT_derivative_at_one_l1938_193801

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one : (deriv f 1) = 2 * Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_derivative_at_one_l1938_193801


namespace NUMINAMATH_GPT_tan_ratio_l1938_193867

-- Definitions of the problem conditions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to the angles

-- The given equation condition
axiom h : a * Real.cos B - b * Real.cos A = (4 / 5) * c

-- The goal is to prove the value of tan(A) / tan(B)
theorem tan_ratio (A B C : ℝ) (a b c : ℝ) (h : a * Real.cos B - b * Real.cos A = (4 / 5) * c) :
  Real.tan A / Real.tan B = 9 :=
sorry

end NUMINAMATH_GPT_tan_ratio_l1938_193867


namespace NUMINAMATH_GPT_constant_term_expansion_l1938_193851

theorem constant_term_expansion (a : ℝ) (h : (2 + a * x) * (1 + 1/x) ^ 5 = (2 + 5 * a)) : 2 + 5 * a = 12 → a = 2 :=
by
  intro h_eq
  have h_sum : 2 + 5 * a = 12 := h_eq
  sorry

end NUMINAMATH_GPT_constant_term_expansion_l1938_193851


namespace NUMINAMATH_GPT_evaluation_at_2_l1938_193859

def f (x : ℚ) : ℚ := (2 * x^2 + 7 * x + 12) / (x^2 + 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem evaluation_at_2 :
  f (g 2) + g (f 2) = 196 / 65 := by
  sorry

end NUMINAMATH_GPT_evaluation_at_2_l1938_193859


namespace NUMINAMATH_GPT_ab_value_l1938_193863

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 107) : a * b = 25.3125 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l1938_193863


namespace NUMINAMATH_GPT_distinct_gcd_numbers_l1938_193812

theorem distinct_gcd_numbers (nums : Fin 100 → ℕ) (h_distinct : Function.Injective nums) :
  ¬ ∃ a b c : Fin 100, 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (nums a + Nat.gcd (nums b) (nums c) = nums b + Nat.gcd (nums a) (nums c)) ∧ 
    (nums b + Nat.gcd (nums a) (nums c) = nums c + Nat.gcd (nums a) (nums b)) := 
sorry

end NUMINAMATH_GPT_distinct_gcd_numbers_l1938_193812


namespace NUMINAMATH_GPT_sequence_formula_l1938_193835

theorem sequence_formula (S : ℕ → ℤ) (a : ℕ → ℤ) (h : ∀ n : ℕ, n > 0 → S n = 2 * a n - 2^n + 1) : 
  ∀ n : ℕ, n > 0 → a n = n * 2^(n - 1) :=
by
  intro n hn
  sorry

end NUMINAMATH_GPT_sequence_formula_l1938_193835


namespace NUMINAMATH_GPT_inequality_solution_function_min_value_l1938_193821

theorem inequality_solution (a : ℕ) (h₁ : abs ((3 / 2 : ℚ) - 2) < a) (h₂ : abs ((1 / 2 : ℚ) - 2) ≥ a) : a = 1 := 
by
  -- proof omitted
  sorry

theorem function_min_value (a : ℕ) (h₁ : abs ((3 / 2 : ℚ) - 2) < a) (h₂ : abs ((1 / 2 : ℚ) - 2) ≥ a)
  (h₃ : a = 1) : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (abs (x + a) + abs (x - 2)) = 3 :=
by
  -- proof omitted
  use 0
  -- proof omitted
  sorry

end NUMINAMATH_GPT_inequality_solution_function_min_value_l1938_193821


namespace NUMINAMATH_GPT_roots_of_polynomial_equation_l1938_193845

theorem roots_of_polynomial_equation (x : ℝ) :
  4 * x ^ 4 - 21 * x ^ 3 + 34 * x ^ 2 - 21 * x + 4 = 0 ↔ x = 4 ∨ x = 1 / 4 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_equation_l1938_193845


namespace NUMINAMATH_GPT_range_of_a_l1938_193847

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ x1 * x1 + (a * a - 1) * x1 + a - 2 = 0 ∧ x2 * x2 + (a * a - 1) * x2 + a - 2 = 0) ↔ -2 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1938_193847


namespace NUMINAMATH_GPT_circle_radius_five_c_value_l1938_193844

theorem circle_radius_five_c_value {c : ℝ} :
  (∀ x y : ℝ, x^2 + 8 * x + y^2 + 2 * y + c = 0) → 
  (∃ x y : ℝ, (x + 4)^2 + (y + 1)^2 = 25) → 
  c = 42 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_five_c_value_l1938_193844


namespace NUMINAMATH_GPT_find_angle_x_l1938_193850

theorem find_angle_x (A B C D : Type) 
  (angleACB angleBCD : ℝ) 
  (h1 : angleACB = 90)
  (h2 : angleBCD = 40) 
  (h3 : angleACB + angleBCD + x = 180) : 
  x = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_x_l1938_193850


namespace NUMINAMATH_GPT_solve_inequality_l1938_193869

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_nonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def f_at_one_third (f : ℝ → ℝ) : Prop :=
  f (1/3) = 0

theorem solve_inequality (f : ℝ → ℝ) (x : ℝ) :
  even_function f →
  increasing_on_nonnegatives f →
  f_at_one_third f →
  (0 < x ∧ x < 1/2) ∨ (x > 2) ↔ f (Real.logb (1/8) x) > 0 :=
by
  -- the proof will be filled in here
  sorry

end NUMINAMATH_GPT_solve_inequality_l1938_193869


namespace NUMINAMATH_GPT_real_solution_for_any_y_l1938_193818

theorem real_solution_for_any_y (x : ℝ) :
  (∀ y z : ℝ, x^2 + y^2 + z^2 + 2 * x * y * z = 1 → ∃ z : ℝ,  x^2 + y^2 + z^2 + 2 * x * y * z = 1) ↔ (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_GPT_real_solution_for_any_y_l1938_193818


namespace NUMINAMATH_GPT_pieces_count_l1938_193860

def pieces_after_n_tears (n : ℕ) : ℕ :=
  3 * n + 1

theorem pieces_count (n : ℕ) : pieces_after_n_tears n = 3 * n + 1 :=
by
  sorry

end NUMINAMATH_GPT_pieces_count_l1938_193860


namespace NUMINAMATH_GPT_arithmetic_sequence_15th_term_l1938_193882

/-- 
The arithmetic sequence with first term 1 and common difference 3.
The 15th term of this sequence is 43.
-/
theorem arithmetic_sequence_15th_term :
  ∀ (a1 d n : ℕ), a1 = 1 → d = 3 → n = 15 → (a1 + (n - 1) * d) = 43 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_15th_term_l1938_193882


namespace NUMINAMATH_GPT_correct_statement_is_C_l1938_193874

theorem correct_statement_is_C :
  (∃ x : ℚ, ∀ y : ℚ, x < y) = false ∧
  (∃ x : ℚ, x < 0 ∧ ∀ y : ℚ, y < 0 → x < y) = false ∧
  (∃ x : ℝ, ∀ y : ℝ, abs x ≤ abs y) ∧
  (∃ x : ℝ, 0 < x ∧ ∀ y : ℝ, 0 < y → x ≤ y) = false :=
sorry

end NUMINAMATH_GPT_correct_statement_is_C_l1938_193874


namespace NUMINAMATH_GPT_tan_ratio_of_angles_l1938_193802

theorem tan_ratio_of_angles (a b : ℝ) (h1 : Real.sin (a + b) = 3/4) (h2 : Real.sin (a - b) = 1/2) :
    (Real.tan a / Real.tan b) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_tan_ratio_of_angles_l1938_193802


namespace NUMINAMATH_GPT_pies_difference_l1938_193804

theorem pies_difference (time : ℕ) (alice_time : ℕ) (bob_time : ℕ) (charlie_time : ℕ)
    (h_time : time = 90) (h_alice : alice_time = 5) (h_bob : bob_time = 6) (h_charlie : charlie_time = 7) :
    (time / alice_time - time / bob_time) + (time / alice_time - time / charlie_time) = 9 := by
  sorry

end NUMINAMATH_GPT_pies_difference_l1938_193804


namespace NUMINAMATH_GPT_quadratic_eq_proof_l1938_193890

noncomputable def quadratic_eq := ∀ (a b : ℝ), 
  (a ≠ 0 → (∃ (x : ℝ), a * x^2 + b * x + 1/4 = 0) →
    (a = b^2 ∧ a = 1 ∧ b = 1) ∨ (a > 1 ∧ 0 < b ∧ b < 1 → ¬ ∃ (x : ℝ), a * x^2 + b * x + 1/4 = 0))

theorem quadratic_eq_proof : quadratic_eq := 
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_proof_l1938_193890


namespace NUMINAMATH_GPT_angle_measure_l1938_193873

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end NUMINAMATH_GPT_angle_measure_l1938_193873


namespace NUMINAMATH_GPT_quadratic_function_property_l1938_193883

theorem quadratic_function_property
    (a b c : ℝ)
    (f : ℝ → ℝ)
    (h_f_def : ∀ x, f x = a * x^2 + b * x + c)
    (h_vertex : f (-2) = a^2)
    (h_point : f (-1) = 6)
    (h_vertex_condition : -b / (2 * a) = -2)
    (h_a_neg : a < 0) :
    (a + c) / b = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_property_l1938_193883


namespace NUMINAMATH_GPT_distance_between_trees_l1938_193884

-- Define the conditions
def yard_length : ℝ := 325
def number_of_trees : ℝ := 26
def number_of_intervals : ℝ := number_of_trees - 1

-- Define what we need to prove
theorem distance_between_trees:
  (yard_length / number_of_intervals) = 13 := 
  sorry

end NUMINAMATH_GPT_distance_between_trees_l1938_193884


namespace NUMINAMATH_GPT_books_sold_l1938_193810

-- Define the conditions
def initial_books : ℕ := 134
def books_given_away : ℕ := 39
def remaining_books : ℕ := 68

-- Define the intermediate calculation of books left after giving away
def books_after_giving_away : ℕ := initial_books - books_given_away

-- Prove the number of books sold
theorem books_sold (initial_books books_given_away remaining_books : ℕ) (h1 : books_after_giving_away = 95) (h2 : remaining_books = 68) :
  (books_after_giving_away - remaining_books) = 27 :=
by
  sorry

end NUMINAMATH_GPT_books_sold_l1938_193810


namespace NUMINAMATH_GPT_avg_diff_l1938_193816

theorem avg_diff (a x c : ℝ) (h1 : (a + x) / 2 = 40) (h2 : (x + c) / 2 = 60) :
  c - a = 40 :=
by
  sorry

end NUMINAMATH_GPT_avg_diff_l1938_193816


namespace NUMINAMATH_GPT_a_share_is_approx_560_l1938_193895

noncomputable def investment_share (a_invest b_invest c_invest total_months b_share : ℕ) : ℝ :=
  let total_invest := a_invest + b_invest + c_invest
  let total_profit := (b_share * total_invest) / b_invest
  let a_share_ratio := a_invest / total_invest
  (a_share_ratio * total_profit)

theorem a_share_is_approx_560 
  (a_invest : ℕ := 7000) 
  (b_invest : ℕ := 11000) 
  (c_invest : ℕ := 18000) 
  (total_months : ℕ := 8) 
  (b_share : ℕ := 880) : 
  ∃ (a_share : ℝ), abs (a_share - 560) < 1 :=
by
  let a_share := investment_share a_invest b_invest c_invest total_months b_share
  existsi a_share
  sorry

end NUMINAMATH_GPT_a_share_is_approx_560_l1938_193895


namespace NUMINAMATH_GPT_stone_statue_cost_l1938_193872

theorem stone_statue_cost :
  ∃ S : Real, 
    let total_earnings := 10 * S + 20 * 5
    let earnings_after_taxes := 0.9 * total_earnings
    earnings_after_taxes = 270 ∧ S = 20 :=
sorry

end NUMINAMATH_GPT_stone_statue_cost_l1938_193872


namespace NUMINAMATH_GPT_batsman_average_after_12th_innings_l1938_193803

theorem batsman_average_after_12th_innings (A : ℤ) :
  (∀ A : ℤ, (11 * A + 60 = 12 * (A + 2))) → (A = 36) → (A + 2 = 38) := 
by
  intro h_avg_increase h_init_avg
  sorry

end NUMINAMATH_GPT_batsman_average_after_12th_innings_l1938_193803


namespace NUMINAMATH_GPT_rebecca_has_22_eggs_l1938_193833

-- Define the conditions
def number_of_groups : ℕ := 11
def eggs_per_group : ℕ := 2

-- Define the total number of eggs calculated from the conditions.
def total_eggs : ℕ := number_of_groups * eggs_per_group

-- State the theorem and provide the proof outline.
theorem rebecca_has_22_eggs : total_eggs = 22 := by {
  -- Proof will go here, but for now we put sorry to indicate it is not yet provided.
  sorry
}

end NUMINAMATH_GPT_rebecca_has_22_eggs_l1938_193833


namespace NUMINAMATH_GPT_find_A_l1938_193879

axiom power_eq_A (A : ℝ) (x y : ℝ) : 2^x = A ∧ 7^(2*y) = A
axiom reciprocal_sum_eq_2 (x y : ℝ) : (1/x) + (1/y) = 2

theorem find_A (A x y : ℝ) : 
  (2^x = A) ∧ (7^(2*y) = A) ∧ ((1/x) + (1/y) = 2) -> A = 7*Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_A_l1938_193879


namespace NUMINAMATH_GPT_caterpillar_count_l1938_193814

theorem caterpillar_count 
    (initial_count : ℕ)
    (hatched : ℕ)
    (left : ℕ)
    (h_initial : initial_count = 14)
    (h_hatched : hatched = 4)
    (h_left : left = 8) :
    initial_count + hatched - left = 10 :=
by
    sorry

end NUMINAMATH_GPT_caterpillar_count_l1938_193814


namespace NUMINAMATH_GPT_terminal_side_in_quadrant_l1938_193888

theorem terminal_side_in_quadrant (k : ℤ) (α : ℝ)
  (h: π + 2 * k * π < α ∧ α < (3 / 2) * π + 2 * k * π) :
  (π / 2) + k * π < α / 2 ∧ α / 2 < (3 / 4) * π + k * π :=
sorry

end NUMINAMATH_GPT_terminal_side_in_quadrant_l1938_193888


namespace NUMINAMATH_GPT_no_integer_solutions_other_than_zero_l1938_193893

theorem no_integer_solutions_other_than_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = x^2 * y^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_integer_solutions_other_than_zero_l1938_193893


namespace NUMINAMATH_GPT_class_b_students_l1938_193864

theorem class_b_students (total_students : ℕ) (sample_size : ℕ) (class_a_sample : ℕ) :
  total_students = 100 → sample_size = 10 → class_a_sample = 4 → 
  (total_students - total_students * class_a_sample / sample_size = 60) :=
by
  intros
  sorry

end NUMINAMATH_GPT_class_b_students_l1938_193864


namespace NUMINAMATH_GPT_arithmetic_expression_eval_l1938_193885

theorem arithmetic_expression_eval : 
  5 * 7.5 + 2 * 12 + 8.5 * 4 + 7 * 6 = 137.5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_eval_l1938_193885


namespace NUMINAMATH_GPT_laila_scores_possible_values_l1938_193819

theorem laila_scores_possible_values :
  ∃ (num_y_values : ℕ), num_y_values = 4 ∧ 
  (∀ (x y : ℤ), 0 ≤ x ∧ x ≤ 100 ∧
                 0 ≤ y ∧ y ≤ 100 ∧
                 4 * x + y = 410 ∧
                 y > x → 
                 (y = 86 ∨ y = 90 ∨ y = 94 ∨ y = 98)
  ) :=
  ⟨4, by sorry⟩

end NUMINAMATH_GPT_laila_scores_possible_values_l1938_193819


namespace NUMINAMATH_GPT_range_of_a_l1938_193875

def new_operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, new_operation x (x - a) > 1) ↔ (a < -3 ∨ 1 < a) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1938_193875


namespace NUMINAMATH_GPT_cube_surface_area_l1938_193827

open Real

theorem cube_surface_area (V : ℝ) (a : ℝ) (S : ℝ)
  (h1 : V = a ^ 3)
  (h2 : a = 4)
  (h3 : V = 64) :
  S = 6 * a ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l1938_193827


namespace NUMINAMATH_GPT_roots_squared_sum_l1938_193820

theorem roots_squared_sum (a b : ℝ) (h : a^2 - 8 * a + 8 = 0 ∧ b^2 - 8 * b + 8 = 0) : a^2 + b^2 = 48 := 
sorry

end NUMINAMATH_GPT_roots_squared_sum_l1938_193820


namespace NUMINAMATH_GPT_measure_angle_PQR_is_55_l1938_193887

noncomputable def measure_angle_PQR (POQ QOR : ℝ) : ℝ :=
  let POQ := 120
  let QOR := 130
  let POR := 360 - (POQ + QOR)
  let OPR := (180 - POR) / 2
  let OPQ := (180 - POQ) / 2
  let OQR := (180 - QOR) / 2
  OPQ + OQR

theorem measure_angle_PQR_is_55 : measure_angle_PQR 120 130 = 55 := by
  sorry

end NUMINAMATH_GPT_measure_angle_PQR_is_55_l1938_193887


namespace NUMINAMATH_GPT_students_in_class_l1938_193853

theorem students_in_class (y : ℕ) (H : 2 * y^2 + 6 * y + 9 = 490) : 
  y + (y + 3) = 31 := by
  sorry

end NUMINAMATH_GPT_students_in_class_l1938_193853


namespace NUMINAMATH_GPT_missile_time_equation_l1938_193832

variable (x : ℝ)

def machToMetersPerSecond := 340
def missileSpeedInMach := 26
def secondsPerMinute := 60
def distanceToTargetInKilometers := 12000
def kilometersToMeters := 1000

theorem missile_time_equation :
  (missileSpeedInMach * machToMetersPerSecond * secondsPerMinute * x) / kilometersToMeters = distanceToTargetInKilometers :=
sorry

end NUMINAMATH_GPT_missile_time_equation_l1938_193832


namespace NUMINAMATH_GPT_sphere_volume_l1938_193861

theorem sphere_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 36 * Real.pi) : (4/3) * Real.pi * r^3 = 36 * Real.pi := 
sorry

end NUMINAMATH_GPT_sphere_volume_l1938_193861


namespace NUMINAMATH_GPT_fifth_term_arithmetic_sequence_l1938_193838

variable (a d : ℤ)

def arithmetic_sequence (n : ℤ) : ℤ :=
  a + (n - 1) * d

theorem fifth_term_arithmetic_sequence :
  arithmetic_sequence a d 20 = 12 →
  arithmetic_sequence a d 21 = 15 →
  arithmetic_sequence a d 5 = -33 :=
by
  intro h20 h21
  sorry

end NUMINAMATH_GPT_fifth_term_arithmetic_sequence_l1938_193838


namespace NUMINAMATH_GPT_has_buried_correct_number_of_bones_l1938_193808

def bones_received_per_month : ℕ := 10
def number_of_months : ℕ := 5
def bones_available : ℕ := 8

def total_bones_received : ℕ := bones_received_per_month * number_of_months
def bones_buried : ℕ := total_bones_received - bones_available

theorem has_buried_correct_number_of_bones : bones_buried = 42 := by
  sorry

end NUMINAMATH_GPT_has_buried_correct_number_of_bones_l1938_193808


namespace NUMINAMATH_GPT_probability_of_white_ball_l1938_193877

theorem probability_of_white_ball (red_balls white_balls : ℕ) (draws : ℕ)
    (h_red : red_balls = 4) (h_white : white_balls = 2) (h_draws : draws = 2) :
    ((4 * 2 + 1) / 15 : ℚ) = 3 / 5 := by sorry

end NUMINAMATH_GPT_probability_of_white_ball_l1938_193877
