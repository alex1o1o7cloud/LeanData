import Mathlib

namespace NUMINAMATH_GPT_convert_base10_to_base9_l659_65946

theorem convert_base10_to_base9 : 
  (2 * 9^3 + 6 * 9^2 + 7 * 9^1 + 7 * 9^0) = 2014 :=
by
  sorry

end NUMINAMATH_GPT_convert_base10_to_base9_l659_65946


namespace NUMINAMATH_GPT_calculate_order_cost_l659_65947

-- Defining the variables and given conditions
variables (C E S D W : ℝ)

-- Given conditions as assumptions
axiom h1 : (2 / 5) * C = E * S
axiom h2 : (1 / 4) * (3 / 5) * C = D * W

-- Theorem statement for the amount paid for the orders
theorem calculate_order_cost (C E S D W : ℝ) (h1 : (2 / 5) * C = E * S) (h2 : (1 / 4) * (3 / 5) * C = D * W) : 
  (9 / 20) * C = C - ((2 / 5) * C + (3 / 20) * C) :=
sorry

end NUMINAMATH_GPT_calculate_order_cost_l659_65947


namespace NUMINAMATH_GPT_souvenir_cost_l659_65986

def total_souvenirs : ℕ := 1000
def total_cost : ℝ := 220
def unknown_souvenirs : ℕ := 400
def known_cost : ℝ := 0.20

theorem souvenir_cost :
  ∃ x : ℝ, x = 0.25 ∧ total_cost = unknown_souvenirs * x + (total_souvenirs - unknown_souvenirs) * known_cost :=
by
  sorry

end NUMINAMATH_GPT_souvenir_cost_l659_65986


namespace NUMINAMATH_GPT_oil_output_per_capita_l659_65991

theorem oil_output_per_capita 
  (total_oil_output_russia : ℝ := 13737.1 * 100 / 9)
  (population_russia : ℝ := 147)
  (population_non_west : ℝ := 6.9)
  (oil_output_non_west : ℝ := 1480.689)
  : 
  (55.084 : ℝ) = 55.084 ∧ 
    (214.59 : ℝ) = (1480.689 / 6.9) ∧ 
    (1038.33 : ℝ) = (total_oil_output_russia / population_russia) :=
by
  sorry

end NUMINAMATH_GPT_oil_output_per_capita_l659_65991


namespace NUMINAMATH_GPT_parallel_lines_from_perpendicularity_l659_65995

variables (a b : Type) (α β : Type)

-- Define the necessary conditions
def is_line (l : Type) : Prop := sorry
def is_plane (p : Type) : Prop := sorry
def perpendicular (l : Type) (p : Type) : Prop := sorry
def parallel (l1 l2 : Type) : Prop := sorry

axiom line_a : is_line a
axiom line_b : is_line b
axiom plane_alpha : is_plane α
axiom plane_beta : is_plane β
axiom a_perp_alpha : perpendicular a α
axiom b_perp_alpha : perpendicular b α

-- State the theorem
theorem parallel_lines_from_perpendicularity : parallel a b :=
  sorry

end NUMINAMATH_GPT_parallel_lines_from_perpendicularity_l659_65995


namespace NUMINAMATH_GPT_speed_of_second_train_l659_65994

/-- 
Given:
1. A train leaves Mumbai at 9 am at a speed of 40 kmph.
2. After one hour, another train leaves Mumbai in the same direction at an unknown speed.
3. The two trains meet at a distance of 80 km from Mumbai.

Prove that the speed of the second train is 80 kmph.
-/
theorem speed_of_second_train (v : ℝ) :
  (∃ (distance_first : ℝ) (distance_meet : ℝ) (initial_speed_first : ℝ) (hours_later : ℤ),
    distance_first = 40 ∧ distance_meet = 80 ∧ initial_speed_first = 40 ∧ hours_later = 1 ∧
    v = distance_meet / (distance_meet / initial_speed_first - hours_later)) → v = 80 := by
  sorry

end NUMINAMATH_GPT_speed_of_second_train_l659_65994


namespace NUMINAMATH_GPT_oranges_per_group_l659_65965

theorem oranges_per_group (total_oranges groups : ℕ) (h1 : total_oranges = 384) (h2 : groups = 16) :
  total_oranges / groups = 24 := by
  sorry

end NUMINAMATH_GPT_oranges_per_group_l659_65965


namespace NUMINAMATH_GPT_find_smallest_d_l659_65998

-- Given conditions: The known digits sum to 26
def sum_known_digits : ℕ := 5 + 2 + 4 + 7 + 8 

-- Define the smallest digit d such that 52,d47,8 is divisible by 9
def smallest_d (d : ℕ) (sum_digits_with_d : ℕ) : Prop :=
  sum_digits_with_d = sum_known_digits + d ∧ (sum_digits_with_d % 9 = 0)

theorem find_smallest_d : ∃ d : ℕ, smallest_d d 27 :=
sorry

end NUMINAMATH_GPT_find_smallest_d_l659_65998


namespace NUMINAMATH_GPT_divide_condition_l659_65934

theorem divide_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ n : ℕ, 0 < n ∧ a ∣ (b^n - n) :=
by
  sorry

end NUMINAMATH_GPT_divide_condition_l659_65934


namespace NUMINAMATH_GPT_two_integers_divide_2_pow_96_minus_1_l659_65985

theorem two_integers_divide_2_pow_96_minus_1 : 
  ∃ a b : ℕ, (60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧ a ≠ b ∧ a ∣ (2^96 - 1) ∧ b ∣ (2^96 - 1) ∧ a = 63 ∧ b = 65) := 
sorry

end NUMINAMATH_GPT_two_integers_divide_2_pow_96_minus_1_l659_65985


namespace NUMINAMATH_GPT_max_f_l659_65996

noncomputable def f (x : ℝ) : ℝ :=
  1 / (|x + 3| + |x + 1| + |x - 2| + |x - 5|)

theorem max_f : ∃ x : ℝ, f x = 1 / 11 :=
by
  sorry

end NUMINAMATH_GPT_max_f_l659_65996


namespace NUMINAMATH_GPT_sum_of_consecutive_powers_divisible_l659_65999

theorem sum_of_consecutive_powers_divisible (a : ℕ) (n : ℕ) (h : 0 ≤ n) : 
  a^n + a^(n + 1) ∣ a * (a + 1) :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_powers_divisible_l659_65999


namespace NUMINAMATH_GPT_simplify_expression_l659_65970

def real_numbers (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a^3 + b^3 = a^2 + b^2

theorem simplify_expression (a b : ℝ) (h : real_numbers a b) :
  (a^2 / b + b^2 / a - 1 / (a * a * b * b)) = (a^4 + 2 * a * b + b^4 - 1) / (a * b) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l659_65970


namespace NUMINAMATH_GPT_ball_bounces_height_l659_65932

theorem ball_bounces_height : ∃ k : ℕ, ∀ n ≥ k, 800 * (2 / 3: ℝ) ^ n < 10 :=
by
  sorry

end NUMINAMATH_GPT_ball_bounces_height_l659_65932


namespace NUMINAMATH_GPT_uncle_bradley_bills_l659_65933

theorem uncle_bradley_bills :
  ∃ (fifty_bills hundred_bills : ℕ),
    (fifty_bills = 300 / 50) ∧ (hundred_bills = 700 / 100) ∧ (300 + 700 = 1000) ∧ (50 * fifty_bills + 100 * hundred_bills = 1000) ∧ (fifty_bills + hundred_bills = 13) :=
by
  sorry

end NUMINAMATH_GPT_uncle_bradley_bills_l659_65933


namespace NUMINAMATH_GPT_kekai_garage_sale_l659_65981

theorem kekai_garage_sale :
  let shirts := 5
  let shirt_price := 1
  let pants := 5
  let pant_price := 3
  let total_money := (shirts * shirt_price) + (pants * pant_price)
  let money_kept := total_money / 2
  money_kept = 10 :=
by
  sorry

end NUMINAMATH_GPT_kekai_garage_sale_l659_65981


namespace NUMINAMATH_GPT_total_fruit_salads_correct_l659_65910

-- Definitions for the conditions
def alayas_fruit_salads : ℕ := 200
def angels_fruit_salads : ℕ := 2 * alayas_fruit_salads
def total_fruit_salads : ℕ := alayas_fruit_salads + angels_fruit_salads

-- Theorem statement
theorem total_fruit_salads_correct : total_fruit_salads = 600 := by
  -- Proof goes here, but is not required for this task
  sorry

end NUMINAMATH_GPT_total_fruit_salads_correct_l659_65910


namespace NUMINAMATH_GPT_non_congruent_triangles_count_l659_65959

-- Let there be 15 equally spaced points on a circle,
-- and considering triangles formed by connecting 3 of these points.
def num_non_congruent_triangles (n : Nat) : Nat :=
  (if n = 15 then 19 else 0)

theorem non_congruent_triangles_count :
  num_non_congruent_triangles 15 = 19 :=
by
  sorry

end NUMINAMATH_GPT_non_congruent_triangles_count_l659_65959


namespace NUMINAMATH_GPT_correct_comprehensive_survey_l659_65936

-- Definitions for the types of surveys.
inductive Survey
| A : Survey
| B : Survey
| C : Survey
| D : Survey

-- Function that identifies the survey suitable for a comprehensive survey.
def is_comprehensive_survey (s : Survey) : Prop :=
  match s with
  | Survey.A => False            -- A is for sampling, not comprehensive
  | Survey.B => False            -- B is for sampling, not comprehensive
  | Survey.C => False            -- C is for sampling, not comprehensive
  | Survey.D => True             -- D is suitable for comprehensive survey

-- The theorem to prove that D is the correct answer.
theorem correct_comprehensive_survey : is_comprehensive_survey Survey.D = True := by
  sorry

end NUMINAMATH_GPT_correct_comprehensive_survey_l659_65936


namespace NUMINAMATH_GPT_find_m_l659_65919

-- Definitions for the lines and the condition of parallelism
def line1 (m : ℝ) (x y : ℝ): Prop := x + m * y + 6 = 0
def line2 (m : ℝ) (x y : ℝ): Prop := 3 * x + (m - 2) * y + 2 * m = 0

-- Condition for lines being parallel
def parallel_lines (m : ℝ) : Prop := 1 * (m - 2) - 3 * m = 0

-- Main formal statement
theorem find_m (m : ℝ) (h1 : ∀ x y, line1 m x y)
                (h2 : ∀ x y, line2 m x y)
                (h_parallel : parallel_lines m) : m = -1 :=
sorry

end NUMINAMATH_GPT_find_m_l659_65919


namespace NUMINAMATH_GPT_min_value_of_a_l659_65964

theorem min_value_of_a (x y : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) 
  (h : ∀ x y, 0 < x → 0 < y → (x + y) * (1 / x + a / y) ≥ 9) :
  4 ≤ a :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l659_65964


namespace NUMINAMATH_GPT_final_temperature_is_correct_l659_65904

def initial_temperature : ℝ := 40
def after_jerry_temperature (T : ℝ) : ℝ := 2 * T
def after_dad_temperature (T : ℝ) : ℝ := T - 30
def after_mother_temperature (T : ℝ) : ℝ := T - 0.30 * T
def after_sister_temperature (T : ℝ) : ℝ := T + 24

theorem final_temperature_is_correct :
  after_sister_temperature (after_mother_temperature (after_dad_temperature (after_jerry_temperature initial_temperature))) = 59 :=
sorry

end NUMINAMATH_GPT_final_temperature_is_correct_l659_65904


namespace NUMINAMATH_GPT_jerome_gave_to_meg_l659_65949

theorem jerome_gave_to_meg (init_money half_money given_away meg bianca : ℝ) 
    (h1 : half_money = 43) 
    (h2 : init_money = 2 * half_money) 
    (h3 : 54 = init_money - given_away)
    (h4 : given_away = meg + bianca)
    (h5 : bianca = 3 * meg) : 
    meg = 8 :=
by
  sorry

end NUMINAMATH_GPT_jerome_gave_to_meg_l659_65949


namespace NUMINAMATH_GPT_driving_time_to_beach_l659_65982

theorem driving_time_to_beach (total_trip_time : ℝ) (k : ℝ) (x : ℝ)
  (h1 : total_trip_time = 14)
  (h2 : k = 2.5)
  (h3 : total_trip_time = (2 * x) + (k * (2 * x))) :
  x = 2 := by 
  sorry

end NUMINAMATH_GPT_driving_time_to_beach_l659_65982


namespace NUMINAMATH_GPT_isosceles_triangle_angle_l659_65971

theorem isosceles_triangle_angle
  (A B C : ℝ)
  (h1 : A = C)
  (h2 : B = 2 * A - 40)
  (h3 : A + B + C = 180) :
  B = 70 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_l659_65971


namespace NUMINAMATH_GPT_find_number_of_rabbits_l659_65925

variable (R P : ℕ)

theorem find_number_of_rabbits (h1 : R + P = 60) (h2 : 4 * R + 2 * P = 192) : R = 36 := 
by
  sorry

end NUMINAMATH_GPT_find_number_of_rabbits_l659_65925


namespace NUMINAMATH_GPT_order_of_abc_l659_65903

noncomputable def a := Real.log 1.2
noncomputable def b := (11 / 10) - (10 / 11)
noncomputable def c := 1 / (5 * Real.exp 0.1)

theorem order_of_abc : b > a ∧ a > c :=
by
  sorry

end NUMINAMATH_GPT_order_of_abc_l659_65903


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_prop_l659_65992

theorem sufficient_but_not_necessary_condition_for_prop :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → a ≥ 5 :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_prop_l659_65992


namespace NUMINAMATH_GPT_clock_confusion_times_l659_65990

-- Conditions translated into Lean definitions
def h_move : ℝ := 0.5  -- hour hand moves at 0.5 degrees per minute
def m_move : ℝ := 6.0  -- minute hand moves at 6 degrees per minute

-- Overlap condition formulated
def overlap_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 10 ∧ 11 * (n : ℝ) = k * 360

-- The final theorem statement in Lean 4
theorem clock_confusion_times : 
  ∃ (count : ℕ), count = 132 ∧ 
    (∀ n < 144, (overlap_condition n → false)) :=
by
  -- Proof to be inserted here
  sorry

end NUMINAMATH_GPT_clock_confusion_times_l659_65990


namespace NUMINAMATH_GPT_length_of_room_l659_65948

theorem length_of_room (width : ℝ) (cost_per_sq_meter : ℝ) (total_cost : ℝ) (L : ℝ) 
  (h_width : width = 2.75)
  (h_cost_per_sq_meter : cost_per_sq_meter = 600)
  (h_total_cost : total_cost = 10725)
  (h_area_cost_eq : total_cost = L * width * cost_per_sq_meter) : 
  L = 6.5 :=
by 
  simp [h_width, h_cost_per_sq_meter, h_total_cost, h_area_cost_eq] at *
  sorry

end NUMINAMATH_GPT_length_of_room_l659_65948


namespace NUMINAMATH_GPT_intervals_increasing_max_min_value_range_of_m_l659_65918

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

theorem intervals_increasing : ∀ (x : ℝ), ∃ k : ℤ, -π/6 + k * π ≤ x ∧ x ≤ π/3 + k * π := sorry

theorem max_min_value (x : ℝ) (hx : π/4 ≤ x ∧ x ≤ π/2) :
  (f (π/3) = 0) ∧ (f (π/2) = -1/2) :=
  sorry

theorem range_of_m (x : ℝ) (hx : π/4 ≤ x ∧ x ≤ π/2) :
  ∀ m : ℝ, (∀ y : ℝ, (π/4 ≤ y ∧ y ≤ π/2) → |f y - m| < 1) ↔ (-1 < m ∧ m < 1/2) :=
  sorry

end NUMINAMATH_GPT_intervals_increasing_max_min_value_range_of_m_l659_65918


namespace NUMINAMATH_GPT_necessary_condition_for_ellipse_l659_65961

theorem necessary_condition_for_ellipse (m : ℝ) : 
  (5 - m > 0) → (m + 3 > 0) → (5 - m ≠ m + 3) → (-3 < m ∧ m < 5 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_GPT_necessary_condition_for_ellipse_l659_65961


namespace NUMINAMATH_GPT_rectangle_invalid_perimeter_l659_65926

-- Define conditions
def positive_integer (n : ℕ) : Prop := n > 0

-- Define the rectangle with given area
def area_24 (length width : ℕ) : Prop := length * width = 24

-- Define the function to calculate perimeter for given length and width
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- The theorem to prove
theorem rectangle_invalid_perimeter (length width : ℕ) (h₁ : positive_integer length) (h₂ : positive_integer width) (h₃ : area_24 length width) : 
  (perimeter length width) ≠ 36 :=
sorry

end NUMINAMATH_GPT_rectangle_invalid_perimeter_l659_65926


namespace NUMINAMATH_GPT_sad_employees_left_geq_cheerful_l659_65960

-- Define the initial number of sad employees
def initial_sad_employees : Nat := 36

-- Define the final number of remaining employees after the game
def final_remaining_employees : Nat := 1

-- Define the total number of employees hit and out of the game
def employees_out : Nat := initial_sad_employees - final_remaining_employees

-- Define the number of cheerful employees who have left
def cheerful_employees_left := employees_out

-- Define the number of sad employees who have left
def sad_employees_left := employees_out

-- The theorem stating the problem proof
theorem sad_employees_left_geq_cheerful:
    sad_employees_left ≥ cheerful_employees_left :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_sad_employees_left_geq_cheerful_l659_65960


namespace NUMINAMATH_GPT_water_required_to_prepare_saline_solution_l659_65944

theorem water_required_to_prepare_saline_solution (water_ratio : ℝ) (required_volume : ℝ) : 
  water_ratio = 3 / 8 ∧ required_volume = 0.64 → required_volume * water_ratio = 0.24 :=
by
  sorry

end NUMINAMATH_GPT_water_required_to_prepare_saline_solution_l659_65944


namespace NUMINAMATH_GPT_gcf_60_75_l659_65976

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_GPT_gcf_60_75_l659_65976


namespace NUMINAMATH_GPT_fisherman_bass_count_l659_65906

theorem fisherman_bass_count (B T G : ℕ) (h1 : T = B / 4) (h2 : G = 2 * B) (h3 : B + T + G = 104) : B = 32 :=
by
  sorry

end NUMINAMATH_GPT_fisherman_bass_count_l659_65906


namespace NUMINAMATH_GPT_new_probability_of_blue_ball_l659_65943

theorem new_probability_of_blue_ball 
  (initial_total_balls : ℕ) (initial_blue_balls : ℕ) (removed_blue_balls : ℕ) :
  initial_total_balls = 18 →
  initial_blue_balls = 6 →
  removed_blue_balls = 3 →
  (initial_blue_balls - removed_blue_balls) / (initial_total_balls - removed_blue_balls) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_new_probability_of_blue_ball_l659_65943


namespace NUMINAMATH_GPT_sugar_packs_l659_65973

variable (totalSugar : ℕ) (packWeight : ℕ) (sugarLeft : ℕ)

noncomputable def numberOfPacks (totalSugar packWeight sugarLeft : ℕ) : ℕ :=
  (totalSugar - sugarLeft) / packWeight

theorem sugar_packs : numberOfPacks 3020 250 20 = 12 := by
  sorry

end NUMINAMATH_GPT_sugar_packs_l659_65973


namespace NUMINAMATH_GPT_total_marks_secured_l659_65920

-- Define the conditions
def correct_points_per_question := 4
def wrong_points_per_question := 1
def total_questions := 60
def correct_questions := 40

-- Calculate the remaining incorrect questions
def wrong_questions := total_questions - correct_questions

-- Calculate total marks secured by the student
def total_marks := (correct_questions * correct_points_per_question) - (wrong_questions * wrong_points_per_question)

-- The statement to be proven
theorem total_marks_secured : total_marks = 140 := by
  -- This will be proven in Lean's proof assistant
  sorry

end NUMINAMATH_GPT_total_marks_secured_l659_65920


namespace NUMINAMATH_GPT_inequality_proof_l659_65916

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b * c) + b / (a * c) + c / (a * b) ≥ 2 / a + 2 / b - 2 / c := 
  sorry

end NUMINAMATH_GPT_inequality_proof_l659_65916


namespace NUMINAMATH_GPT_hindi_books_count_l659_65930

theorem hindi_books_count (H : ℕ) (h1 : 22 = 22) (h2 : Nat.choose 23 H = 1771) : H = 3 :=
sorry

end NUMINAMATH_GPT_hindi_books_count_l659_65930


namespace NUMINAMATH_GPT_value_of_x_l659_65938

theorem value_of_x (x : ℝ) (h : 2 ≤ |x - 3| ∧ |x - 3| ≤ 6) : x ∈ Set.Icc (-3 : ℝ) 1 ∪ Set.Icc 5 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l659_65938


namespace NUMINAMATH_GPT_option_A_cannot_be_true_l659_65937

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (r : ℝ) -- common ratio for the geometric sequence
variable (n : ℕ) -- number of terms

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

def sum_of_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem option_A_cannot_be_true
  (h_geom : is_geometric_sequence a r)
  (h_sum : sum_of_geometric_sequence a S) :
  a 2016 * (S 2016 - S 2015) ≠ 0 :=
sorry

end NUMINAMATH_GPT_option_A_cannot_be_true_l659_65937


namespace NUMINAMATH_GPT_lights_ratio_l659_65953

theorem lights_ratio (M S L : ℕ) (h1 : M = 12) (h2 : S = M + 10) (h3 : 118 = (S * 1) + (M * 2) + (L * 3)) :
  L = 24 ∧ L / M = 2 :=
by
  sorry

end NUMINAMATH_GPT_lights_ratio_l659_65953


namespace NUMINAMATH_GPT_maximum_correct_answers_l659_65901

theorem maximum_correct_answers (a b c : ℕ) (h1 : a + b + c = 60)
  (h2 : 5 * a - 2 * c = 150) : a ≤ 38 :=
by
  sorry

end NUMINAMATH_GPT_maximum_correct_answers_l659_65901


namespace NUMINAMATH_GPT_largest_root_divisible_by_17_l659_65955

theorem largest_root_divisible_by_17 (a : ℝ) (h : Polynomial.eval a (Polynomial.C 1 + Polynomial.C (-3) * Polynomial.X^2 + Polynomial.X^3) = 0) (root_large : ∀ x ∈ {b | Polynomial.eval b (Polynomial.C 1 + Polynomial.C (-3) * Polynomial.X^2 + Polynomial.X^3) = 0}, x ≤ a) :
  a^1788 % 17 = 0 ∧ a^1988 % 17 = 0 :=
by
  sorry

end NUMINAMATH_GPT_largest_root_divisible_by_17_l659_65955


namespace NUMINAMATH_GPT_parabola_focus_l659_65913

theorem parabola_focus :
  ∃ f, (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (-f + 1/4))^2)) ∧ f = 1/8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l659_65913


namespace NUMINAMATH_GPT_inequality_proof_l659_65908

variables (x y : ℝ) (n : ℕ)

theorem inequality_proof (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  (x^n / (x + y^3) + y^n / (x^3 + y)) ≥ (2^(4-n) / 5) := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l659_65908


namespace NUMINAMATH_GPT_smallest_n_exists_unique_k_l659_65951

/- The smallest positive integer n for which there exists
   a unique integer k such that 9/16 < n / (n + k) < 7/12 is n = 1. -/

theorem smallest_n_exists_unique_k :
  ∃! (n : ℕ), n > 0 ∧ (∃! (k : ℤ), (9 : ℚ)/16 < (n : ℤ)/(n + k) ∧ (n : ℤ)/(n + k) < (7 : ℚ)/12) :=
sorry

end NUMINAMATH_GPT_smallest_n_exists_unique_k_l659_65951


namespace NUMINAMATH_GPT_solve_system_of_equations_l659_65935

theorem solve_system_of_equations :
  ∃ (x y z : ℤ), (x + y + z = 6) ∧ (x + y * z = 7) ∧ 
  ((x = 7 ∧ y = 0 ∧ z = -1) ∨ 
   (x = 7 ∧ y = -1 ∧ z = 0) ∨ 
   (x = 1 ∧ y = 3 ∧ z = 2) ∨ 
   (x = 1 ∧ y = 2 ∧ z = 3)) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l659_65935


namespace NUMINAMATH_GPT_no_valid_height_configuration_l659_65950

-- Define the heights and properties
variables {a : Fin 7 → ℝ}
variables {p : ℝ}

-- Define the condition as a theorem
theorem no_valid_height_configuration (h : ∀ n : Fin 7, p * a n + (1 - p) * a (n + 2) % 7 > 
                                         p * a (n + 3) % 7 + (1 - p) * a (n + 1) % 7) :
  ¬ (∃ (a : Fin 7 → ℝ), 
    (∀ n : Fin 7, p * a n + (1 - p) * a (n + 2) % 7 > 
                  p * a (n + 3) % 7 + (1 - p) * a (n + 1) % 7) ∧
    true) :=
sorry

end NUMINAMATH_GPT_no_valid_height_configuration_l659_65950


namespace NUMINAMATH_GPT_find_third_number_l659_65997

theorem find_third_number 
  (h1 : (14 + 32 + x) / 3 = (21 + 47 + 22) / 3 + 3) : x = 53 := by
  sorry

end NUMINAMATH_GPT_find_third_number_l659_65997


namespace NUMINAMATH_GPT_intersection_P_Q_range_a_l659_65980

def set_P : Set ℝ := { x | 2 * x^2 - 3 * x + 1 ≤ 0 }
def set_Q (a : ℝ) : Set ℝ := { x | (x - a) * (x - a - 1) ≤ 0 }

theorem intersection_P_Q (a : ℝ) (h_a : a = 1) :
  set_P ∩ set_Q 1 = {1} :=
sorry

theorem range_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set_P → x ∈ set_Q a) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_GPT_intersection_P_Q_range_a_l659_65980


namespace NUMINAMATH_GPT_quadratic_roots_condition_l659_65993

theorem quadratic_roots_condition (k : ℝ) : 
  ((∃ x : ℝ, (k - 1) * x^2 + 4 * x + 1 = 0) ∧ ∃ x1 x2 : ℝ, x1 ≠ x2) ↔ (k < 5 ∧ k ≠ 1) :=
by {
  sorry  
}

end NUMINAMATH_GPT_quadratic_roots_condition_l659_65993


namespace NUMINAMATH_GPT_find_principal_amount_l659_65909

-- Define the parameters
def R : ℝ := 11.67
def T : ℝ := 5
def A : ℝ := 950

-- State the theorem
theorem find_principal_amount : ∃ P : ℝ, A = P * (1 + (R/100) * T) :=
by { 
  use 600, 
  -- Skip the proof 
  sorry 
}

end NUMINAMATH_GPT_find_principal_amount_l659_65909


namespace NUMINAMATH_GPT_number_corresponding_to_8_minutes_l659_65984

theorem number_corresponding_to_8_minutes (x : ℕ) : 
  (12 / 6 = x / 480) → x = 960 :=
by
  sorry

end NUMINAMATH_GPT_number_corresponding_to_8_minutes_l659_65984


namespace NUMINAMATH_GPT_f_sin_periodic_f_monotonically_increasing_f_minus_2_not_even_f_symmetric_about_point_l659_65988

noncomputable def f (x : ℝ) : ℝ := (4 * Real.exp x) / (Real.exp x + 1)

theorem f_sin_periodic : ∀ x, f (Real.sin (x + 2 * Real.pi)) = f (Real.sin x) := sorry

theorem f_monotonically_increasing : ∀ x y, x < y → f x < f y := sorry

theorem f_minus_2_not_even : ¬(∀ x, f x - 2 = f (-x) - 2) := sorry

theorem f_symmetric_about_point : ∀ x, f x + f (-x) = 4 := sorry

end NUMINAMATH_GPT_f_sin_periodic_f_monotonically_increasing_f_minus_2_not_even_f_symmetric_about_point_l659_65988


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l659_65989

theorem quadratic_inequality_solution (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * x + m ≤ 0) ↔ m ≤ 1 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l659_65989


namespace NUMINAMATH_GPT_x_14_and_inverse_x_14_l659_65972

theorem x_14_and_inverse_x_14 (x : ℂ) (h : x^2 + x + 1 = 0) : x^14 + x⁻¹^14 = -1 :=
by
  sorry

end NUMINAMATH_GPT_x_14_and_inverse_x_14_l659_65972


namespace NUMINAMATH_GPT_parallel_lines_sufficient_but_not_necessary_l659_65905

theorem parallel_lines_sufficient_but_not_necessary (a : ℝ) :
  (a = 1 ↔ ((ax + y - 1 = 0) ∧ (x + ay + 1 = 0) → False)) := 
sorry

end NUMINAMATH_GPT_parallel_lines_sufficient_but_not_necessary_l659_65905


namespace NUMINAMATH_GPT_largest_four_digit_sum_20_l659_65929

-- Defining the four-digit number and conditions.
def is_four_digit_number (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  ∃ a b c d : ℕ, a + b + c + d = s ∧ n = 1000 * a + 100 * b + 10 * c + d

-- Proof problem statement.
theorem largest_four_digit_sum_20 : ∃ n, is_four_digit_number n ∧ digits_sum_to n 20 ∧ ∀ m, is_four_digit_number m ∧ digits_sum_to m 20 → m ≤ n :=
  sorry

end NUMINAMATH_GPT_largest_four_digit_sum_20_l659_65929


namespace NUMINAMATH_GPT_sum_of_f_values_l659_65977

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem sum_of_f_values 
  (f : ℝ → ℝ)
  (hf_odd : is_odd_function f)
  (hf_periodic : ∀ x, f (2 - x) = f x)
  (hf_neg_one : f (-1) = 1) :
  f 1 + f 2 + f 3 + f 4 + (502 * (f 1 + f 2 + f 3 + f 4)) = -1 := 
sorry

end NUMINAMATH_GPT_sum_of_f_values_l659_65977


namespace NUMINAMATH_GPT_sqrt_inequality_l659_65912

theorem sqrt_inequality (n : ℕ) : 
  (n ≥ 0) → (Real.sqrt (n + 2) - Real.sqrt (n + 1) ≤ Real.sqrt (n + 1) - Real.sqrt n) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_sqrt_inequality_l659_65912


namespace NUMINAMATH_GPT_roots_negative_reciprocal_l659_65967

theorem roots_negative_reciprocal (a b c : ℝ) (α β : ℝ) (h_eq : a * α ^ 2 + b * α + c = 0)
  (h_roots : α * β = -1) : c = -a :=
sorry

end NUMINAMATH_GPT_roots_negative_reciprocal_l659_65967


namespace NUMINAMATH_GPT_probability_of_positive_l659_65978

-- Definitions based on the conditions
def balls : List ℚ := [-2, 0, 1/4, 3]
def total_balls : ℕ := 4
def positive_filter (x : ℚ) : Bool := x > 0
def positive_balls : List ℚ := balls.filter positive_filter
def positive_count : ℕ := positive_balls.length
def probability : ℚ := positive_count / total_balls

-- Statement to prove
theorem probability_of_positive : probability = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_of_positive_l659_65978


namespace NUMINAMATH_GPT_Shane_current_age_44_l659_65942

-- Declaring the known conditions and definitions
variable (Garret_present_age : ℕ) (Shane_past_age : ℕ) (Shane_present_age : ℕ)
variable (h1 : Garret_present_age = 12)
variable (h2 : Shane_past_age = 2 * Garret_present_age)
variable (h3 : Shane_present_age = Shane_past_age + 20)

theorem Shane_current_age_44 : Shane_present_age = 44 :=
by
  -- Proof to be filled here
  sorry

end NUMINAMATH_GPT_Shane_current_age_44_l659_65942


namespace NUMINAMATH_GPT_solution_unique_l659_65958

def is_solution (x : ℝ) : Prop :=
  ⌊x * ⌊x⌋⌋ = 48

theorem solution_unique (x : ℝ) : is_solution x → x = -48 / 7 :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solution_unique_l659_65958


namespace NUMINAMATH_GPT_unique_integers_exist_l659_65969

theorem unique_integers_exist (p : ℕ) (hp : p > 1) : 
  ∃ (a b c : ℤ), b^2 - 4*a*c = 1 - 4*p ∧ 0 < a ∧ a ≤ c ∧ -a ≤ b ∧ b < a :=
sorry

end NUMINAMATH_GPT_unique_integers_exist_l659_65969


namespace NUMINAMATH_GPT_radius_of_circle_l659_65914

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem radius_of_circle : 
  ∃ r : ℝ, circle_area r = circle_circumference r → r = 2 := 
by 
  sorry

end NUMINAMATH_GPT_radius_of_circle_l659_65914


namespace NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l659_65968

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ) -- circumferences of the smaller and larger circle
  (h : (1 / 6) * C1 = (2 / 15) * C2) -- condition given: 60-degree arc on the smaller circle equals 48-degree arc on the larger circle
  : (C1 / C2)^2 = (16 / 25) := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l659_65968


namespace NUMINAMATH_GPT_distribution_ways_l659_65915

theorem distribution_ways :
  let friends := 12
  let problems := 6
  (friends ^ problems = 2985984) :=
by
  sorry

end NUMINAMATH_GPT_distribution_ways_l659_65915


namespace NUMINAMATH_GPT_tan_A_in_right_triangle_l659_65962

theorem tan_A_in_right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] (angle_A angle_B angle_C : ℝ) 
  (sin_B : ℚ) (tan_A : ℚ) :
  angle_C = 90 ∧ sin_B = 3 / 5 → tan_A = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_A_in_right_triangle_l659_65962


namespace NUMINAMATH_GPT_mean_goals_l659_65902

theorem mean_goals :
  let goals := 2 * 3 + 4 * 2 + 5 * 1 + 6 * 1
  let players := 3 + 2 + 1 + 1
  goals / players = 25 / 7 :=
by
  sorry

end NUMINAMATH_GPT_mean_goals_l659_65902


namespace NUMINAMATH_GPT_average_speed_of_train_l659_65975

-- Condition: Distance traveled is 42 meters
def distance : ℕ := 42

-- Condition: Time taken is 6 seconds
def time : ℕ := 6

-- Average speed computation
theorem average_speed_of_train : distance / time = 7 := by
  -- Left to the prover
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l659_65975


namespace NUMINAMATH_GPT_percentage_increase_first_year_l659_65940

theorem percentage_increase_first_year (P : ℝ) (x : ℝ) :
  (1 + x / 100) * 0.7 = 1.0499999999999998 → x = 50 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_first_year_l659_65940


namespace NUMINAMATH_GPT_log_eq_15_given_log_base3_x_eq_5_l659_65922

variable (x : ℝ)
variable (log_base3_x : ℝ)
variable (h : log_base3_x = 5)

theorem log_eq_15_given_log_base3_x_eq_5 (h : log_base3_x = 5) : log_base3_x * 3 = 15 :=
by
  sorry

end NUMINAMATH_GPT_log_eq_15_given_log_base3_x_eq_5_l659_65922


namespace NUMINAMATH_GPT_total_hangers_is_65_l659_65931

noncomputable def calculate_hangers_total : ℕ :=
  let pink := 7
  let green := 4
  let blue := green - 1
  let yellow := blue - 1
  let orange := 2 * (pink + green)
  let purple := (blue - yellow) + 3
  let red := (pink + green + blue) / 3
  let brown := 3 * red + 1
  let gray := (3 * purple) / 5
  pink + green + blue + yellow + orange + purple + red + brown + gray

theorem total_hangers_is_65 : calculate_hangers_total = 65 := 
by 
  sorry

end NUMINAMATH_GPT_total_hangers_is_65_l659_65931


namespace NUMINAMATH_GPT_blocks_combination_count_l659_65921

-- Definition statements reflecting all conditions in the problem
def select_4_blocks_combinations : ℕ :=
  let choose (n k : ℕ) := Nat.choose n k
  let factorial (n : ℕ) := Nat.factorial n
  choose 6 4 * choose 6 4 * factorial 4

-- Theorem stating the result we want to prove
theorem blocks_combination_count : select_4_blocks_combinations = 5400 :=
by
  -- We will provide the proof steps here
  sorry

end NUMINAMATH_GPT_blocks_combination_count_l659_65921


namespace NUMINAMATH_GPT_proof_problem_l659_65945

variable {ι : Type} [LinearOrderedField ι]

-- Let A be a family of sets indexed by natural numbers
variables {A : ℕ → Set ι}

-- Hypotheses
def condition1 (A : ℕ → Set ι) : Prop :=
  (⋃ i, A i) = Set.univ

def condition2 (A : ℕ → Set ι) (a : ι) : Prop :=
  ∀ i b c, b > c → b - c ≥ a ^ i → b ∈ A i → c ∈ A i

theorem proof_problem (A : ℕ → Set ι) (a : ι) :
  condition1 A → condition2 A a → 0 < a → a < 2 :=
sorry

end NUMINAMATH_GPT_proof_problem_l659_65945


namespace NUMINAMATH_GPT_least_positive_divisible_by_five_primes_l659_65917

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_divisible_by_five_primes_l659_65917


namespace NUMINAMATH_GPT_total_cost_of_items_is_correct_l659_65928

theorem total_cost_of_items_is_correct :
  ∀ (M R F : ℝ),
  (10 * M = 24 * R) →
  (F = 2 * R) →
  (F = 24) →
  (4 * M + 3 * R + 5 * F = 271.2) :=
by
  intros M R F h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_cost_of_items_is_correct_l659_65928


namespace NUMINAMATH_GPT_fraction_difference_l659_65952

def A : ℕ := 3 + 6 + 9
def B : ℕ := 2 + 5 + 8

theorem fraction_difference : (A / B) - (B / A) = 11 / 30 := by
  sorry

end NUMINAMATH_GPT_fraction_difference_l659_65952


namespace NUMINAMATH_GPT_correct_exponentiation_operation_l659_65963

theorem correct_exponentiation_operation (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end NUMINAMATH_GPT_correct_exponentiation_operation_l659_65963


namespace NUMINAMATH_GPT_triangle_third_side_l659_65974

theorem triangle_third_side {x : ℕ} (h1 : 3 < x) (h2 : x < 7) (h3 : x % 2 = 1) : x = 5 := by
  sorry

end NUMINAMATH_GPT_triangle_third_side_l659_65974


namespace NUMINAMATH_GPT_radha_profit_percentage_l659_65979

theorem radha_profit_percentage (SP CP : ℝ) (hSP : SP = 144) (hCP : CP = 90) :
  ((SP - CP) / CP) * 100 = 60 := by
  sorry

end NUMINAMATH_GPT_radha_profit_percentage_l659_65979


namespace NUMINAMATH_GPT_courtyard_length_is_60_l659_65956

noncomputable def stone_length : ℝ := 2.5
noncomputable def stone_breadth : ℝ := 2.0
noncomputable def num_stones : ℕ := 198
noncomputable def courtyard_breadth : ℝ := 16.5

theorem courtyard_length_is_60 :
  ∃ (courtyard_length : ℝ), courtyard_length = 60 ∧
  num_stones * (stone_length * stone_breadth) = courtyard_length * courtyard_breadth :=
sorry

end NUMINAMATH_GPT_courtyard_length_is_60_l659_65956


namespace NUMINAMATH_GPT_distinct_digits_sum_base7_l659_65939

theorem distinct_digits_sum_base7
    (A B C : ℕ)
    (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A)
    (h_nonzero : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
    (h_base7 : A < 7 ∧ B < 7 ∧ C < 7)
    (h_sum_eq : ((7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B)) = (7^3 * A + 7^2 * A + 7 * A)) :
    B + C = 6 :=
by {
    sorry
}

end NUMINAMATH_GPT_distinct_digits_sum_base7_l659_65939


namespace NUMINAMATH_GPT_opposite_of_negative_2023_l659_65927

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end NUMINAMATH_GPT_opposite_of_negative_2023_l659_65927


namespace NUMINAMATH_GPT_tom_drives_distance_before_karen_wins_l659_65924

def karen_late_minutes := 4
def karen_speed_mph := 60
def tom_speed_mph := 45

theorem tom_drives_distance_before_karen_wins : 
  ∃ d : ℝ, d = 21 := by
  sorry

end NUMINAMATH_GPT_tom_drives_distance_before_karen_wins_l659_65924


namespace NUMINAMATH_GPT_certain_number_is_36_75_l659_65911

theorem certain_number_is_36_75 (A B C X : ℝ) (h_ratio_A : A = 5 * (C / 8)) (h_ratio_B : B = 6 * (C / 8)) (h_C : C = 42) (h_relation : A + C = B + X) :
  X = 36.75 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_36_75_l659_65911


namespace NUMINAMATH_GPT_xy_plus_four_is_square_l659_65957

theorem xy_plus_four_is_square (x y : ℕ) (h : ((1 / (x : ℝ)) + (1 / (y : ℝ)) + 1 / (x * y : ℝ)) = (1 / (x + 4 : ℝ) + 1 / (y - 4 : ℝ) + 1 / ((x + 4) * (y - 4) : ℝ))) : 
  ∃ (k : ℕ), xy + 4 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_xy_plus_four_is_square_l659_65957


namespace NUMINAMATH_GPT_at_least_one_negative_l659_65900

-- Defining the circle partition and the properties given in the problem.
def circle_partition (a : Fin 7 → ℤ) : Prop :=
  ∃ (l1 l2 l3 : Finset (Fin 7)),
    l1.card = 4 ∧ l2.card = 4 ∧ l3.card = 4 ∧
    (∀ i ∈ l1, ∀ j ∉ l1, a i + a j = 0) ∧
    (∀ i ∈ l2, ∀ j ∉ l2, a i + a j = 0) ∧
    (∀ i ∈ l3, ∀ j ∉ l3, a i + a j = 0) ∧
    ∃ i, a i = 0

-- The main theorem to prove.
theorem at_least_one_negative : 
  ∀ (a : Fin 7 → ℤ), 
  circle_partition a → 
  ∃ i, a i < 0 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_negative_l659_65900


namespace NUMINAMATH_GPT_triangle_side_c_l659_65923

noncomputable def area_of_triangle (a b C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C

noncomputable def law_of_cosines (a b C : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)

theorem triangle_side_c (a b C : ℝ) (h1 : a = 3) (h2 : C = Real.pi * 2 / 3) (h3 : area_of_triangle a b C = 15 * Real.sqrt 3 / 4) : law_of_cosines a b C = 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_c_l659_65923


namespace NUMINAMATH_GPT_triangle_inequality_l659_65966

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l659_65966


namespace NUMINAMATH_GPT_area_triangle_tangent_circles_l659_65907

theorem area_triangle_tangent_circles :
  ∃ (A B C : Type) (radius1 radius2 : ℝ) 
    (tangent1 tangent2 : ℝ → ℝ → Prop)
    (congruent_sides : ℝ → Prop),
    radius1 = 1 ∧ radius2 = 2 ∧
    (∀ x y, tangent1 x y) ∧ (∀ x y, tangent2 x y) ∧
    congruent_sides 1 ∧ congruent_sides 2 ∧
    ∃ (area : ℝ), area = 16 * Real.sqrt 2 :=
by
  -- This is where the proof would be written
  sorry

end NUMINAMATH_GPT_area_triangle_tangent_circles_l659_65907


namespace NUMINAMATH_GPT_new_foreign_students_l659_65983

theorem new_foreign_students 
  (total_students : ℕ)
  (percent_foreign : ℕ)
  (foreign_students_next_sem : ℕ)
  (current_foreign_students : ℕ := total_students * percent_foreign / 100) : 
  total_students = 1800 → 
  percent_foreign = 30 → 
  foreign_students_next_sem = 740 → 
  foreign_students_next_sem - current_foreign_students = 200 :=
by
  intros
  sorry

end NUMINAMATH_GPT_new_foreign_students_l659_65983


namespace NUMINAMATH_GPT_A_B_work_together_finish_l659_65954
noncomputable def work_rate_B := 1 / 12
noncomputable def work_rate_A := 2 * work_rate_B
noncomputable def combined_work_rate := work_rate_A + work_rate_B

theorem A_B_work_together_finish (hB: work_rate_B = 1/12) (hA: work_rate_A = 2 * work_rate_B) :
  (1 / combined_work_rate) = 4 :=
by
  -- Placeholder for the proof, we don't need to provide the proof steps
  sorry

end NUMINAMATH_GPT_A_B_work_together_finish_l659_65954


namespace NUMINAMATH_GPT_millie_bracelets_left_l659_65987

def millie_bracelets_initial : ℕ := 9
def millie_bracelets_lost : ℕ := 2

theorem millie_bracelets_left : millie_bracelets_initial - millie_bracelets_lost = 7 := 
by
  sorry

end NUMINAMATH_GPT_millie_bracelets_left_l659_65987


namespace NUMINAMATH_GPT_campaign_meaning_l659_65941

-- Define a function that gives the meaning of "campaign" as a noun
def meaning_of_campaign_noun : String :=
  "campaign, activity"

-- The theorem asserts that the meaning of "campaign" as a noun is "campaign, activity"
theorem campaign_meaning : meaning_of_campaign_noun = "campaign, activity" :=
by
  -- We add sorry here because we are not required to provide the proof
  sorry

end NUMINAMATH_GPT_campaign_meaning_l659_65941
