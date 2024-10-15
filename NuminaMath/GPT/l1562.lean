import Mathlib

namespace NUMINAMATH_GPT_history_homework_time_l1562_156252

def total_time := 180
def math_homework := 45
def english_homework := 30
def science_homework := 50
def special_project := 30

theorem history_homework_time : total_time - (math_homework + english_homework + science_homework + special_project) = 25 := by
  sorry

end NUMINAMATH_GPT_history_homework_time_l1562_156252


namespace NUMINAMATH_GPT_base_of_second_term_l1562_156214

theorem base_of_second_term (h : ℕ) (a b c : ℕ) (H1 : h > 0) 
  (H2 : 225 ∣ h) (H3 : 216 ∣ h) 
  (H4 : h = (2^a) * (some_number^b) * (5^c)) 
  (H5 : a + b + c = 8) : some_number = 3 :=
by
  sorry

end NUMINAMATH_GPT_base_of_second_term_l1562_156214


namespace NUMINAMATH_GPT_map_scale_l1562_156261

theorem map_scale (map_distance : ℝ) (time : ℝ) (speed : ℝ) (actual_distance : ℝ) (scale : ℝ) 
  (h1 : map_distance = 5) 
  (h2 : time = 1.5) 
  (h3 : speed = 60) 
  (h4 : actual_distance = speed * time) 
  (h5 : scale = map_distance / actual_distance) : 
  scale = 1 / 18 :=
by 
  sorry

end NUMINAMATH_GPT_map_scale_l1562_156261


namespace NUMINAMATH_GPT_parabola_directrix_eq_neg2_l1562_156210

-- Definitions based on conditions
def ellipse_focus (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ x = 2 ∧ y = 0

def parabola_directrix (p x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ ∃ x, x = -p / 2

theorem parabola_directrix_eq_neg2 (p : ℝ) (hp : p > 0) :
  (∀ (x y : ℝ), ellipse_focus 9 5 x y → parabola_directrix p x y) →
  (∃ x y : ℝ, parabola_directrix p x y → x = -2) :=
sorry

end NUMINAMATH_GPT_parabola_directrix_eq_neg2_l1562_156210


namespace NUMINAMATH_GPT_manufacturing_percentage_l1562_156256

theorem manufacturing_percentage (deg_total : ℝ) (deg_manufacturing : ℝ) (h1 : deg_total = 360) (h2 : deg_manufacturing = 126) : 
  (deg_manufacturing / deg_total * 100) = 35 := by
  sorry

end NUMINAMATH_GPT_manufacturing_percentage_l1562_156256


namespace NUMINAMATH_GPT_measure_of_angle_A_l1562_156215

theorem measure_of_angle_A
    (A B : ℝ)
    (h1 : A + B = 90)
    (h2 : A = 3 * B) :
    A = 67.5 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_A_l1562_156215


namespace NUMINAMATH_GPT_incorrect_operation_l1562_156206

variable (a : ℕ)

-- Conditions
def condition1 := 4 * a ^ 2 - a ^ 2 = 3 * a ^ 2
def condition2 := a ^ 3 * a ^ 6 = a ^ 9
def condition3 := (a ^ 2) ^ 3 = a ^ 5
def condition4 := (2 * a ^ 2) ^ 2 = 4 * a ^ 4

-- Theorem to prove
theorem incorrect_operation : (a ^ 2) ^ 3 ≠ a ^ 5 := 
by
  sorry

end NUMINAMATH_GPT_incorrect_operation_l1562_156206


namespace NUMINAMATH_GPT_solution_inequality_l1562_156293

-- Conditions
variables {a b x : ℝ}
theorem solution_inequality (h1 : a < 0) (h2 : b = a) :
  {x : ℝ | (ax + b) ≤ 0} = {x : ℝ | x ≥ -1} →
  {x : ℝ | (ax + b) / (x - 2) > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_inequality_l1562_156293


namespace NUMINAMATH_GPT_average_test_score_of_remainder_l1562_156280

variable (score1 score2 score3 totalAverage : ℝ)
variable (percentage1 percentage2 percentage3 : ℝ)

def equation (score1 score2 score3 totalAverage : ℝ) (percentage1 percentage2 percentage3: ℝ) : Prop :=
  (percentage1 * score1) + (percentage2 * score2) + (percentage3 * score3) = totalAverage

theorem average_test_score_of_remainder
  (h1 : percentage1 = 0.15)
  (h2 : score1 = 100)
  (h3 : percentage2 = 0.5)
  (h4 : score2 = 78)
  (h5 : percentage3 = 0.35)
  (total : totalAverage = 76.05) :
  (score3 = 63) :=
sorry

end NUMINAMATH_GPT_average_test_score_of_remainder_l1562_156280


namespace NUMINAMATH_GPT_Zachary_did_47_pushups_l1562_156281

-- Define the conditions and the question
def Zachary_pushups (David_pushups difference : ℕ) : ℕ :=
  David_pushups - difference

theorem Zachary_did_47_pushups :
  Zachary_pushups 62 15 = 47 :=
by
  -- Provide the proof here (we'll use sorry for now)
  sorry

end NUMINAMATH_GPT_Zachary_did_47_pushups_l1562_156281


namespace NUMINAMATH_GPT_boys_in_fifth_grade_l1562_156218

theorem boys_in_fifth_grade (T S : ℕ) (percent_boys_soccer : ℝ) (girls_not_playing_soccer : ℕ) 
    (hT : T = 420) (hS : S = 250) (h_percent : percent_boys_soccer = 0.86) 
    (h_girls_not_playing_soccer : girls_not_playing_soccer = 65) : 
    ∃ B : ℕ, B = 320 :=
by
  -- We don't need to provide the proof details here
  sorry

end NUMINAMATH_GPT_boys_in_fifth_grade_l1562_156218


namespace NUMINAMATH_GPT_initial_men_count_l1562_156275

variable (M : ℕ)

theorem initial_men_count
  (work_completion_time : ℕ)
  (men_leaving : ℕ)
  (remaining_work_time : ℕ)
  (completion_days : ℕ) :
  work_completion_time = 40 →
  men_leaving = 20 →
  remaining_work_time = 40 →
  completion_days = 10 →
  M = 80 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_men_count_l1562_156275


namespace NUMINAMATH_GPT_x_plus_p_eq_2p_plus_2_l1562_156274

-- Define the conditions and the statement to be proved
theorem x_plus_p_eq_2p_plus_2 (x p : ℝ) (h1 : x > 2) (h2 : |x - 2| = p) : x + p = 2 * p + 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_x_plus_p_eq_2p_plus_2_l1562_156274


namespace NUMINAMATH_GPT_area_of_EFGH_l1562_156228

def shorter_side := 6
def ratio := 2
def longer_side := shorter_side * ratio
def width := 2 * longer_side
def length := shorter_side

theorem area_of_EFGH : length * width = 144 := by
  sorry

end NUMINAMATH_GPT_area_of_EFGH_l1562_156228


namespace NUMINAMATH_GPT_employees_working_abroad_l1562_156279

theorem employees_working_abroad
  (total_employees : ℕ)
  (fraction_abroad : ℝ)
  (h_total : total_employees = 450)
  (h_fraction : fraction_abroad = 0.06) :
  total_employees * fraction_abroad = 27 := 
by
  sorry

end NUMINAMATH_GPT_employees_working_abroad_l1562_156279


namespace NUMINAMATH_GPT_triangle_inequality_l1562_156242

theorem triangle_inequality (a b c : ℝ) (h : a^2 = b^2 + c^2) : 
  (b - c)^2 * (a^2 + 4 * b * c)^2 ≤ 2 * a^6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1562_156242


namespace NUMINAMATH_GPT_scientific_notation_86560_l1562_156216

theorem scientific_notation_86560 : ∃ a n, (86560 : ℝ) = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8.656 ∧ n = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_scientific_notation_86560_l1562_156216


namespace NUMINAMATH_GPT_point_distance_units_l1562_156277

theorem point_distance_units (d : ℝ) (h : |d| = 4) : d = 4 ∨ d = -4 := 
sorry

end NUMINAMATH_GPT_point_distance_units_l1562_156277


namespace NUMINAMATH_GPT_find_divisor_l1562_156283

-- Definitions
def dividend := 199
def quotient := 11
def remainder := 1

-- Statement of the theorem
theorem find_divisor : ∃ x : ℕ, dividend = (x * quotient) + remainder ∧ x = 18 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l1562_156283


namespace NUMINAMATH_GPT_none_of_these_l1562_156211

def y_values_match (f : ℕ → ℕ) : Prop :=
  f 0 = 200 ∧ f 1 = 140 ∧ f 2 = 80 ∧ f 3 = 20 ∧ f 4 = 0

theorem none_of_these :
  ¬ (∃ f : ℕ → ℕ, 
    (∀ x, f x = 200 - 15 * x ∨ 
    f x = 200 - 20 * x + 5 * x^2 ∨ 
    f x = 200 - 30 * x + 10 * x^2 ∨ 
    f x = 150 - 50 * x) ∧ 
    y_values_match f) :=
by sorry

end NUMINAMATH_GPT_none_of_these_l1562_156211


namespace NUMINAMATH_GPT_line_circle_no_intersection_l1562_156226

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_GPT_line_circle_no_intersection_l1562_156226


namespace NUMINAMATH_GPT_problem_equivalence_l1562_156292

theorem problem_equivalence (n : ℕ) (H₁ : 2 * 2006 = 1) (H₂ : ∀ n : ℕ, (2 * n + 2) * 2006 = 3 * (2 * n * 2006)) :
  2008 * 2006 = 3 ^ 1003 :=
by
  sorry

end NUMINAMATH_GPT_problem_equivalence_l1562_156292


namespace NUMINAMATH_GPT_average_age_l1562_156249

theorem average_age (Jared Molly Hakimi : ℕ) (h1 : Jared = Hakimi + 10) (h2 : Molly = 30) (h3 : Hakimi = 40) :
  (Jared + Molly + Hakimi) / 3 = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_age_l1562_156249


namespace NUMINAMATH_GPT_smallest_perimeter_of_triangle_with_area_sqrt3_l1562_156227

open Real

-- Define an equilateral triangle with given area
def equilateral_triangle (a : ℝ) : Prop :=
  ∃ s: ℝ, s > 0 ∧ a = (sqrt 3 / 4) * s^2

-- Problem statement: Prove the smallest perimeter of such a triangle is 6.
theorem smallest_perimeter_of_triangle_with_area_sqrt3 : 
  equilateral_triangle (sqrt 3) → ∃ s: ℝ, s > 0 ∧ 3 * s = 6 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_perimeter_of_triangle_with_area_sqrt3_l1562_156227


namespace NUMINAMATH_GPT_coefficient_of_y_squared_l1562_156267

/-- Given the equation ay^2 - 8y + 55 = 59 and y = 2, prove that the coefficient a is 5. -/
theorem coefficient_of_y_squared (a y : ℝ) (h_y : y = 2) (h_eq : a * y^2 - 8 * y + 55 = 59) : a = 5 := by
  sorry

end NUMINAMATH_GPT_coefficient_of_y_squared_l1562_156267


namespace NUMINAMATH_GPT_min_value_expr_l1562_156237

open Real

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1562_156237


namespace NUMINAMATH_GPT_parallelogram_not_symmetrical_l1562_156285

-- Define the shapes
inductive Shape
| circle
| rectangle
| isosceles_trapezoid
| parallelogram

-- Define what it means for a shape to be symmetrical
def is_symmetrical (s: Shape) : Prop :=
  match s with
  | Shape.circle => True
  | Shape.rectangle => True
  | Shape.isosceles_trapezoid => True
  | Shape.parallelogram => False -- The condition we're interested in proving

-- The main theorem stating the problem
theorem parallelogram_not_symmetrical : is_symmetrical Shape.parallelogram = False :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_not_symmetrical_l1562_156285


namespace NUMINAMATH_GPT_compare_M_N_l1562_156222

variable (a : ℝ)

def M : ℝ := 2 * a * (a - 2) + 7
def N : ℝ := (a - 2) * (a - 3)

theorem compare_M_N : M a > N a :=
by
  sorry

end NUMINAMATH_GPT_compare_M_N_l1562_156222


namespace NUMINAMATH_GPT_number_of_primes_l1562_156287

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem number_of_primes (p : ℕ)
  (H_prime : is_prime p)
  (H_square : is_perfect_square (1 + p + p^2 + p^3 + p^4)) :
  p = 3 :=
sorry

end NUMINAMATH_GPT_number_of_primes_l1562_156287


namespace NUMINAMATH_GPT_fraction_meaningful_condition_l1562_156273

theorem fraction_meaningful_condition (x : ℝ) : (4 / (x + 2) ≠ 0) ↔ (x ≠ -2) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_meaningful_condition_l1562_156273


namespace NUMINAMATH_GPT_train_speed_l1562_156233

theorem train_speed (length : ℕ) (cross_time : ℕ) (speed : ℝ)
    (h1 : length = 250)
    (h2 : cross_time = 3)
    (h3 : speed = (length / cross_time : ℝ) * 3.6) :
    speed = 300 := 
sorry

end NUMINAMATH_GPT_train_speed_l1562_156233


namespace NUMINAMATH_GPT_half_guests_want_two_burgers_l1562_156204

theorem half_guests_want_two_burgers 
  (total_guests : ℕ) (half_guests : ℕ)
  (time_per_side : ℕ) (time_per_burger : ℕ)
  (grill_capacity : ℕ) (total_time : ℕ)
  (guests_one_burger : ℕ) (total_burgers : ℕ) : 
  total_guests = 30 →
  time_per_side = 4 →
  time_per_burger = 8 →
  grill_capacity = 5 →
  total_time = 72 →
  guests_one_burger = 15 →
  total_burgers = 45 →
  half_guests * 2 = total_burgers - guests_one_burger →
  half_guests = 15 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_half_guests_want_two_burgers_l1562_156204


namespace NUMINAMATH_GPT_locus_of_C_l1562_156282

variable (a : ℝ) (h : a > 0)

theorem locus_of_C : 
  ∃ (x y : ℝ), 
  (1 - a) * x^2 - 2 * a * x + (1 + a) * y^2 = 0 :=
sorry

end NUMINAMATH_GPT_locus_of_C_l1562_156282


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1562_156219

-- Problem 1
theorem problem1 : 13 + (-7) - (-9) + 5 * (-2) = 5 :=
by 
  sorry

-- Problem 2
theorem problem2 : abs (-7 / 2) * (12 / 7) / (4 / 3) / (3 ^ 2) = 1 / 2 :=
by 
  sorry

-- Problem 3
theorem problem3 : -1^4 - (1 / 6) * (2 - (-3)^2) = 1 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1562_156219


namespace NUMINAMATH_GPT_rectangle_perimeter_l1562_156212

-- Definitions and assumptions
variables (outer_square_area inner_square_area : ℝ) (rectangles_identical : Prop)

-- Given conditions
def outer_square_area_condition : Prop := outer_square_area = 9
def inner_square_area_condition : Prop := inner_square_area = 1
def rectangles_identical_condition : Prop := rectangles_identical

-- The main theorem to prove
theorem rectangle_perimeter (h_outer : outer_square_area_condition outer_square_area)
                            (h_inner : inner_square_area_condition inner_square_area)
                            (h_rectangles : rectangles_identical_condition rectangles_identical) :
  ∃ perimeter : ℝ, perimeter = 6 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1562_156212


namespace NUMINAMATH_GPT_reciprocal_sum_fractions_l1562_156225

theorem reciprocal_sum_fractions:
  let a := (3: ℚ) / 4
  let b := (5: ℚ) / 6
  let c := (1: ℚ) / 2
  (a + b + c)⁻¹ = 12 / 25 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_sum_fractions_l1562_156225


namespace NUMINAMATH_GPT_unique_n_l1562_156202

theorem unique_n (n : ℕ) (h_pos : 0 < n) :
  (∀ x y : ℕ, (xy + 1) % n = 0 → (x + y) % n = 0) ↔ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_n_l1562_156202


namespace NUMINAMATH_GPT_n_eq_14_l1562_156265

variable {a : ℕ → ℕ}  -- the arithmetic sequence
variable {S : ℕ → ℕ}  -- the sum function of the first n terms
variable {d : ℕ}      -- the common difference of the arithmetic sequence

-- Given Conditions
axiom Sn_eq_4 : S 4 = 40
axiom Sn_eq_210 : ∃ (n : ℕ), S n = 210
axiom Sn_minus_4_eq_130 : ∃ (n : ℕ), S (n - 4) = 130

-- Main theorem to prove
theorem n_eq_14 : ∃ (n : ℕ),  S n = 210 ∧ S (n - 4) = 130 ∧ n = 14 :=
by
  sorry

end NUMINAMATH_GPT_n_eq_14_l1562_156265


namespace NUMINAMATH_GPT_find_b_n_find_T_n_l1562_156248

-- Conditions
def S (n : ℕ) : ℕ := 3 * n^2 + 8 * n
def a (n : ℕ) : ℕ := S n - S (n - 1) -- provided n > 1
def b : ℕ → ℕ := sorry -- This is what we need to prove
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 2)^n  -- Definition of c_n
def T (n : ℕ) : ℕ := sorry -- The sum of the first n terms of c_n

-- Proof requirements
def proof_b_n := ∀ n : ℕ, b n = 3 * n + 1
def proof_T_n := ∀ n : ℕ, T n = 3 * n * 2^(n+2)

theorem find_b_n : proof_b_n := 
by sorry

theorem find_T_n : proof_T_n := 
by sorry

end NUMINAMATH_GPT_find_b_n_find_T_n_l1562_156248


namespace NUMINAMATH_GPT_determine_q_l1562_156247

theorem determine_q (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ k : ℝ, k < 3) ∧ -- indicating degree considerations for asymptotes
  (q 2 = 18) →
  q = (fun x => (-18 / 5) * x ^ 2 + 162 / 5) :=
by
  sorry

end NUMINAMATH_GPT_determine_q_l1562_156247


namespace NUMINAMATH_GPT_dangerous_animals_remaining_in_swamp_l1562_156295

-- Define the initial counts of each dangerous animals
def crocodiles_initial := 42
def alligators_initial := 35
def vipers_initial := 10
def water_moccasins_initial := 28
def cottonmouth_snakes_initial := 15
def piranha_fish_initial := 120

-- Define the counts of migrating animals
def crocodiles_migrating := 9
def alligators_migrating := 7
def vipers_migrating := 3

-- Define the total initial dangerous animals
def total_initial : Nat :=
  crocodiles_initial + alligators_initial + vipers_initial + water_moccasins_initial + cottonmouth_snakes_initial + piranha_fish_initial

-- Define the total migrating dangerous animals
def total_migrating : Nat :=
  crocodiles_migrating + alligators_migrating + vipers_migrating

-- Define the total remaining dangerous animals
def total_remaining : Nat :=
  total_initial - total_migrating

theorem dangerous_animals_remaining_in_swamp :
  total_remaining = 231 :=
by
  -- simply using the calculation we know
  sorry

end NUMINAMATH_GPT_dangerous_animals_remaining_in_swamp_l1562_156295


namespace NUMINAMATH_GPT_find_amount_l1562_156294

theorem find_amount (amount : ℝ) (h : 0.25 * amount = 75) : amount = 300 :=
sorry

end NUMINAMATH_GPT_find_amount_l1562_156294


namespace NUMINAMATH_GPT_cuboid_surface_area_l1562_156236

noncomputable def total_surface_area (x y z : ℝ) : ℝ :=
  2 * (x * y + y * z + z * x)

theorem cuboid_surface_area (x y z : ℝ) (h1 : x + y + z = 40) (h2 : x^2 + y^2 + z^2 = 625) :
  total_surface_area x y z = 975 :=
sorry

end NUMINAMATH_GPT_cuboid_surface_area_l1562_156236


namespace NUMINAMATH_GPT_problem_l1562_156291

theorem problem (a b : ℝ) :
  (∀ x : ℝ, 3 * x - 1 ≤ a ∧ 2 * x ≥ 6 - b → -1 ≤ x ∧ x ≤ 2) →
  a + b = 13 := by
  sorry

end NUMINAMATH_GPT_problem_l1562_156291


namespace NUMINAMATH_GPT_equal_pair_c_l1562_156278

theorem equal_pair_c : (-4)^3 = -(4^3) := 
by {
  sorry
}

end NUMINAMATH_GPT_equal_pair_c_l1562_156278


namespace NUMINAMATH_GPT_students_per_group_l1562_156245

theorem students_per_group (total_students not_picked groups : ℕ) 
    (h1 : total_students = 64) 
    (h2 : not_picked = 36) 
    (h3 : groups = 4) : (total_students - not_picked) / groups = 7 :=
by
  sorry

end NUMINAMATH_GPT_students_per_group_l1562_156245


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1562_156253

-- Boat's speed in still water in km/hr
variable (B S : ℝ)

-- Conditions given for the boat's speed along and against the stream
axiom cond1 : B + S = 11
axiom cond2 : B - S = 5

-- Prove that the speed of the boat in still water is 8 km/hr
theorem boat_speed_in_still_water : B = 8 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1562_156253


namespace NUMINAMATH_GPT_num_positive_int_values_l1562_156217

theorem num_positive_int_values (N : ℕ) :
  (∃ m : ℕ, N = m ∧ m > 0 ∧ 48 % (m + 3) = 0) ↔ (N < 7) :=
sorry

end NUMINAMATH_GPT_num_positive_int_values_l1562_156217


namespace NUMINAMATH_GPT_det_is_18_l1562_156298

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 1],
    ![2, 5]]

theorem det_is_18 : det A = 18 := by
  sorry

end NUMINAMATH_GPT_det_is_18_l1562_156298


namespace NUMINAMATH_GPT_tangent_line_circle_l1562_156299

theorem tangent_line_circle (a : ℝ) : (∀ x y : ℝ, a * x + y + 1 = 0) → (∀ x y : ℝ, x^2 + y^2 - 4 * x = 0) → a = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_circle_l1562_156299


namespace NUMINAMATH_GPT_ordered_pairs_count_l1562_156286

theorem ordered_pairs_count : 
    ∃ (s : Finset (ℝ × ℝ)), 
        (∀ (x y : ℝ), (x + 3 * y = 3 ∧ abs (abs x - abs y) = 1 ↔ (x, y) ∈ s)) ∧ 
        s.card = 3 :=
    by
    sorry

end NUMINAMATH_GPT_ordered_pairs_count_l1562_156286


namespace NUMINAMATH_GPT_parallel_lines_m_l1562_156239

theorem parallel_lines_m (m : ℝ) :
  (∀ (x y : ℝ), 3 * m * x + (m + 2) * y + 1 = 0) ∧
  (∀ (x y : ℝ), (m - 2) * x + (m + 2) * y + 2 = 0) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (3 * m) / (m + 2) = (m - 2) / (m + 2)) →
  (m = -1 ∨ m = -2) :=
sorry

end NUMINAMATH_GPT_parallel_lines_m_l1562_156239


namespace NUMINAMATH_GPT_smallest_solution_l1562_156235

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 1)) + (1 / (x - 5)) = 4 / (x - 4)

theorem smallest_solution : 
  ∃ x : ℝ, equation x ∧ x ≠ 1 ∧ x ≠ 5 ∧ x ≠ 4 ∧ x = (5 - Real.sqrt 33) / 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_l1562_156235


namespace NUMINAMATH_GPT_distance_swim_downstream_correct_l1562_156288

def speed_man_still_water : ℝ := 7
def time_taken : ℝ := 5
def distance_upstream : ℝ := 25

lemma distance_swim_downstream (V_m : ℝ) (t : ℝ) (d_up : ℝ) : 
  t * ((V_m + (V_m - d_up / t)) / 2) = 45 :=
by
  have h_speed_upstream : (V_m - (d_up / t)) = d_up / t := by sorry
  have h_speed_stream : (d_up / t) = (V_m - (d_up / t)) := by sorry
  have h_distance_downstream : t * ((V_m + (V_m - (d_up / t)) / 2)) = t * (V_m + (V_m - (V_m - d_up / t))) := by sorry
  sorry

noncomputable def distance_swim_downstream_value : ℝ :=
  9 * 5

theorem distance_swim_downstream_correct :
  distance_swim_downstream_value = 45 :=
by
  sorry

end NUMINAMATH_GPT_distance_swim_downstream_correct_l1562_156288


namespace NUMINAMATH_GPT_solve_for_x_l1562_156200

theorem solve_for_x (x : ℕ) (h1 : x > 0) (h2 : x % 6 = 0) (h3 : x^2 > 144) (h4 : x < 30) : x = 18 ∨ x = 24 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1562_156200


namespace NUMINAMATH_GPT_find_n_l1562_156246

theorem find_n (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 10) (h3 : n % 11 = 99999 % 11) : n = 9 :=
sorry

end NUMINAMATH_GPT_find_n_l1562_156246


namespace NUMINAMATH_GPT_michael_savings_l1562_156260

theorem michael_savings :
  let price := 45
  let tax_rate := 0.08
  let promo_A_dis := 0.40
  let promo_B_dis := 15
  let before_tax_A := price + price * (1 - promo_A_dis)
  let before_tax_B := price + (price - promo_B_dis)
  let after_tax_A := before_tax_A * (1 + tax_rate)
  let after_tax_B := before_tax_B * (1 + tax_rate)
  after_tax_B - after_tax_A = 3.24 :=
by
  sorry

end NUMINAMATH_GPT_michael_savings_l1562_156260


namespace NUMINAMATH_GPT_least_faces_combined_l1562_156207

theorem least_faces_combined (a b : ℕ) (h1 : a ≥ 6) (h2 : b ≥ 6)
  (h3 : (∃ k : ℕ, k * a * b = 20) → (∃ m : ℕ, 2 * m = 10 * (k + 10))) 
  (h4 : (∃ n : ℕ, n = (a * b) / 10)) (h5 : ∃ l : ℕ, l = 5) : a + b = 20 :=
by
  sorry

end NUMINAMATH_GPT_least_faces_combined_l1562_156207


namespace NUMINAMATH_GPT_john_spends_6_dollars_l1562_156230

-- Let treats_per_day, cost_per_treat, and days_in_month be defined by the conditions of the problem.
def treats_per_day : ℕ := 2
def cost_per_treat : ℝ := 0.1
def days_in_month : ℕ := 30

-- The total expenditure should be defined as the number of treats multiplied by their cost.
def total_number_of_treats := treats_per_day * days_in_month
def total_expenditure := total_number_of_treats * cost_per_treat

-- The statement to be proven: John spends $6 on the treats.
theorem john_spends_6_dollars :
  total_expenditure = 6 :=
sorry

end NUMINAMATH_GPT_john_spends_6_dollars_l1562_156230


namespace NUMINAMATH_GPT_probability_of_symmetry_line_l1562_156250

-- Define the conditions of the problem.
def is_on_symmetry_line (P Q : (ℤ × ℤ)) :=
  (Q.fst = P.fst) ∨ (Q.snd = P.snd) ∨ (Q.fst - P.fst = Q.snd - P.snd) ∨ (Q.fst - P.fst = P.snd - Q.snd)

-- Define the main statement of the theorem to be proved.
theorem probability_of_symmetry_line :
  let grid_size := 11
  let total_points := grid_size * grid_size
  let center : (ℤ × ℤ) := (grid_size / 2, grid_size / 2)
  let other_points := total_points - 1
  let symmetric_points := 40
  /- Here we need to calculate the probability, which is the ratio of symmetric points to other points,
     and this should equal 1/3 -/
  (symmetric_points : ℚ) / other_points = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_probability_of_symmetry_line_l1562_156250


namespace NUMINAMATH_GPT_hypotenuse_length_l1562_156208

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1562_156208


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1562_156205

theorem other_root_of_quadratic (m : ℝ) (x2 : ℝ) : (x^2 + m * x + 6 = 0) → (x + 2) * (x + x2) = 0 → x2 = -3 :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1562_156205


namespace NUMINAMATH_GPT_subset_condition_l1562_156269

noncomputable def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : (B m ⊆ A) ↔ m ≤ 3 :=
sorry

end NUMINAMATH_GPT_subset_condition_l1562_156269


namespace NUMINAMATH_GPT_symmetric_angle_of_inclination_l1562_156240

theorem symmetric_angle_of_inclination (α₁ : ℝ) (h : 0 ≤ α₁ ∧ α₁ < π) : 
  (∃ β₁ : ℝ, (α₁ = 0 ∧ β₁ = 0) ∨ (0 < α₁ ∧ α₁ < π ∧ β₁ = π - α₁)) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_angle_of_inclination_l1562_156240


namespace NUMINAMATH_GPT_Rose_has_20_crystal_beads_l1562_156223

noncomputable def num_crystal_beads (metal_beads_Nancy : ℕ) (pearl_beads_more_than_metal : ℕ) (beads_per_bracelet : ℕ)
    (total_bracelets : ℕ) (stone_to_crystal_ratio : ℕ) : ℕ :=
  let pearl_beads_Nancy := metal_beads_Nancy + pearl_beads_more_than_metal
  let total_beads_Nancy := metal_beads_Nancy + pearl_beads_Nancy
  let beads_needed := beads_per_bracelet * total_bracelets
  let beads_Rose := beads_needed - total_beads_Nancy
  beads_Rose / stone_to_crystal_ratio.succ

theorem Rose_has_20_crystal_beads :
  num_crystal_beads 40 20 8 20 2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_Rose_has_20_crystal_beads_l1562_156223


namespace NUMINAMATH_GPT_find_m_for_given_slope_l1562_156231

theorem find_m_for_given_slope (m : ℝ) :
  (∃ (P Q : ℝ × ℝ),
    P = (-2, m) ∧ Q = (m, 4) ∧
    (Q.2 - P.2) / (Q.1 - P.1) = 1) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_for_given_slope_l1562_156231


namespace NUMINAMATH_GPT_different_genre_pairs_count_l1562_156266

theorem different_genre_pairs_count 
  (mystery_books : Finset ℕ)
  (fantasy_books : Finset ℕ)
  (biographies : Finset ℕ)
  (h1 : mystery_books.card = 4)
  (h2 : fantasy_books.card = 4)
  (h3 : biographies.card = 4) :
  (mystery_books.product (fantasy_books ∪ biographies)).card +
  (fantasy_books.product (mystery_books ∪ biographies)).card +
  (biographies.product (mystery_books ∪ fantasy_books)).card = 48 := 
sorry

end NUMINAMATH_GPT_different_genre_pairs_count_l1562_156266


namespace NUMINAMATH_GPT_investment_duration_l1562_156244

theorem investment_duration 
  (P SI R : ℕ) (T : ℕ) 
  (hP : P = 800) 
  (hSI : SI = 128) 
  (hR : R = 4) 
  (h : SI = P * R * T / 100) 
  : T = 4 :=
by 
  rw [hP, hSI, hR] at h
  sorry

end NUMINAMATH_GPT_investment_duration_l1562_156244


namespace NUMINAMATH_GPT_minimal_functions_l1562_156232

open Int

theorem minimal_functions (f : ℤ → ℤ) (c : ℤ) :
  (∀ x, f (x + 2017) = f x) ∧
  (∀ x y, (f (f x + f y + 1) - f (f x + f y)) % 2017 = c) →
  (c = 1 ∨ c = 2016 ∨ c = 1008 ∨ c = 1009) :=
by
  sorry

end NUMINAMATH_GPT_minimal_functions_l1562_156232


namespace NUMINAMATH_GPT_smallest_four_digit_integer_l1562_156255

theorem smallest_four_digit_integer (n : ℕ) :
  (75 * n ≡ 225 [MOD 450]) ∧ (1000 ≤ n ∧ n < 10000) → n = 1005 :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_integer_l1562_156255


namespace NUMINAMATH_GPT_range_of_function_l1562_156264

theorem range_of_function : ∀ (y : ℝ), (0 < y ∧ y ≤ 1 / 2) ↔ ∃ (x : ℝ), y = 1 / (x^2 + 2) := 
by
  sorry

end NUMINAMATH_GPT_range_of_function_l1562_156264


namespace NUMINAMATH_GPT_cost_of_coffee_B_per_kg_l1562_156271

-- Define the cost of coffee A per kilogram
def costA : ℝ := 10

-- Define the amount of coffee A used in the mixture
def amountA : ℝ := 240

-- Define the amount of coffee B used in the mixture
def amountB : ℝ := 240

-- Define the total amount of the mixture
def totalAmount : ℝ := 480

-- Define the selling price of the mixture per kilogram
def sellingPrice : ℝ := 11

-- Define the cost of coffee B per kilogram as a variable B
variable (B : ℝ)

-- Define the total cost of the mixture
def totalCost : ℝ := totalAmount * sellingPrice

-- Define the cost of coffee A used
def costOfA : ℝ := amountA * costA

-- Define the cost of coffee B used as total cost minus the cost of A
def costOfB : ℝ := totalCost - costOfA

-- Calculate the cost of coffee B per kilogram
theorem cost_of_coffee_B_per_kg : B = 12 :=
by
  have h1 : costOfA = 2400 := by sorry
  have h2 : totalCost = 5280 := by sorry
  have h3 : costOfB = 2880 := by sorry
  have h4 : B = costOfB / amountB := by sorry
  have h5 : B = 2880 / 240 := by sorry
  have h6 : B = 12 := by sorry
  exact h6

end NUMINAMATH_GPT_cost_of_coffee_B_per_kg_l1562_156271


namespace NUMINAMATH_GPT_exist_positive_integers_x_y_z_l1562_156258

theorem exist_positive_integers_x_y_z (n : ℕ) (hn : n > 0) : 
  ∃ (x y z : ℕ), 
    x = 2^(n^2) * 3^(n+1) ∧
    y = 2^(n^2 - n) * 3^n ∧
    z = 2^(n^2 - 2*n + 2) * 3^(n-1) ∧
    x^(n-1) + y^n = z^(n+1) :=
by {
  -- placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_exist_positive_integers_x_y_z_l1562_156258


namespace NUMINAMATH_GPT_evaluate_101_times_101_l1562_156268

theorem evaluate_101_times_101 : (101 * 101 = 10201) :=
by {
  sorry
}

end NUMINAMATH_GPT_evaluate_101_times_101_l1562_156268


namespace NUMINAMATH_GPT_exists_positive_integer_special_N_l1562_156201

theorem exists_positive_integer_special_N : 
  ∃ (N : ℕ), 
    (∃ (m : ℕ), N = 1990 * (m + 995)) ∧ 
    (∀ (n : ℕ), (∃ (m : ℕ), 2 * N = (n + 1) * (2 * m + n)) ↔ (3980 = 2 * 1990)) := by
  sorry

end NUMINAMATH_GPT_exists_positive_integer_special_N_l1562_156201


namespace NUMINAMATH_GPT_index_card_area_reduction_l1562_156262

theorem index_card_area_reduction :
  ∀ (length width : ℕ),
  (length = 5 ∧ width = 7) →
  ((length - 2) * width = 21) →
  (length * (width - 2) = 25) :=
by
  intros length width h1 h2
  rcases h1 with ⟨h_length, h_width⟩
  sorry

end NUMINAMATH_GPT_index_card_area_reduction_l1562_156262


namespace NUMINAMATH_GPT_complement_is_correct_l1562_156221

def U : Set ℝ := {x | x^2 ≤ 4}
def A : Set ℝ := {x | abs (x + 1) ≤ 1}
def complement_U_A : Set ℝ := U \ A

theorem complement_is_correct :
  complement_U_A = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_GPT_complement_is_correct_l1562_156221


namespace NUMINAMATH_GPT_moles_of_ammonium_nitrate_formed_l1562_156276

def ammonia := ℝ
def nitric_acid := ℝ
def ammonium_nitrate := ℝ

-- Define the stoichiometric coefficients from the balanced equation.
def stoichiometric_ratio_ammonia : ℝ := 1
def stoichiometric_ratio_nitric_acid : ℝ := 1
def stoichiometric_ratio_ammonium_nitrate : ℝ := 1

-- Define the initial moles of reactants.
def initial_moles_ammonia (moles : ℝ) : Prop := moles = 3
def initial_moles_nitric_acid (moles : ℝ) : Prop := moles = 3

-- The reaction goes to completion as all reactants are used:
theorem moles_of_ammonium_nitrate_formed :
  ∀ (moles_ammonia moles_nitric_acid : ℝ),
    initial_moles_ammonia moles_ammonia →
    initial_moles_nitric_acid moles_nitric_acid →
    (moles_ammonia / stoichiometric_ratio_ammonia) = 
    (moles_nitric_acid / stoichiometric_ratio_nitric_acid) →
    (moles_ammonia / stoichiometric_ratio_ammonia) * stoichiometric_ratio_ammonium_nitrate = 3 :=
by
  intros moles_ammonia moles_nitric_acid h_ammonia h_nitric_acid h_ratio
  rw [h_ammonia, h_nitric_acid] at *
  simp only [stoichiometric_ratio_ammonia, stoichiometric_ratio_nitric_acid, stoichiometric_ratio_ammonium_nitrate] at *
  sorry

end NUMINAMATH_GPT_moles_of_ammonium_nitrate_formed_l1562_156276


namespace NUMINAMATH_GPT_part1_part2_l1562_156224

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (2 - a) * x + a

-- Part 1
theorem part1 (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≥ 2 * Real.sqrt 3 / 3 := sorry

-- Part 2
theorem part2 (a x : ℝ) : 
  (f a x < a + 2) ↔ 
    (a = 0 ∧ x < 1) ∨ 
    (a > 0 ∧ -2 / a < x ∧ x < 1) ∨ 
    (-2 < a ∧ a < 0 ∧ (x < 1 ∨ x > -2 / a)) ∨ 
    (a = -2) ∨ 
    (a < -2 ∧ (x < -2 / a ∨ x > 1)) := sorry

end NUMINAMATH_GPT_part1_part2_l1562_156224


namespace NUMINAMATH_GPT_intersection_and_complement_l1562_156259

open Set

def A := {x : ℝ | -4 ≤ x ∧ x ≤ -2}
def B := {x : ℝ | x + 3 ≥ 0}

theorem intersection_and_complement : 
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧ (compl (A ∩ B) = {x | x < -3 ∨ x > -2}) :=
by
  sorry

end NUMINAMATH_GPT_intersection_and_complement_l1562_156259


namespace NUMINAMATH_GPT_equation_of_perpendicular_line_intersection_l1562_156251

theorem equation_of_perpendicular_line_intersection  :
  ∃ (x y : ℝ), 4 * x + 2 * y + 5 = 0 ∧ 3 * x - 2 * y + 9 = 0 ∧ 
               (∃ (m : ℝ), m = 2 ∧ 4 * x - 2 * y + 11 = 0) := 
sorry

end NUMINAMATH_GPT_equation_of_perpendicular_line_intersection_l1562_156251


namespace NUMINAMATH_GPT_find_divisor_l1562_156257

theorem find_divisor (Q R D V : ℤ) (hQ : Q = 65) (hR : R = 5) (hV : V = 1565) (hEquation : V = D * Q + R) : D = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1562_156257


namespace NUMINAMATH_GPT_no_rational_roots_of_odd_coeffs_l1562_156272

theorem no_rational_roots_of_odd_coeffs (a b c : ℤ) (h_a_odd : a % 2 = 1) (h_b_odd : b % 2 = 1) (h_c_odd : c % 2 = 1)
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ (a * (p / q : ℚ)^2 + b * (p / q : ℚ) + c = 0)) : false :=
sorry

end NUMINAMATH_GPT_no_rational_roots_of_odd_coeffs_l1562_156272


namespace NUMINAMATH_GPT_value_of_x_l1562_156203

theorem value_of_x : ∀ x : ℝ, (x^2 - 4) / (x - 2) = 0 → x ≠ 2 → x = -2 := by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_value_of_x_l1562_156203


namespace NUMINAMATH_GPT_max_area_of_triangle_l1562_156270

theorem max_area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : a = 2)
  (h2 : 4 * (Real.cos (A / 2))^2 -  Real.cos (2 * (B + C)) = 7 / 2)
  (h3 : A + B + C = Real.pi) :
  (Real.sqrt 3 / 2 * b * c) ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_area_of_triangle_l1562_156270


namespace NUMINAMATH_GPT_complete_the_square_l1562_156289

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end NUMINAMATH_GPT_complete_the_square_l1562_156289


namespace NUMINAMATH_GPT_find_s_l1562_156297

variable {t s : Real}

theorem find_s (h1 : t = 8 * s^2) (h2 : t = 4) : s = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_s_l1562_156297


namespace NUMINAMATH_GPT_kolya_time_segment_DE_l1562_156284

-- Definitions representing the conditions
def time_petya_route : ℝ := 12  -- Petya takes 12 minutes
def time_kolya_route : ℝ := 12  -- Kolya also takes 12 minutes
def kolya_speed_factor : ℝ := 1.2

-- Proof problem: Prove that Kolya spends 1 minute traveling the segment D-E
theorem kolya_time_segment_DE 
    (v : ℝ)  -- Assume v is Petya's speed
    (time_petya_A_B_C : ℝ := time_petya_route)  
    (time_kolya_A_D_E_F_C : ℝ := time_kolya_route)
    (kolya_fast_factor : ℝ := kolya_speed_factor)
    : (time_petya_A_B_C / kolya_fast_factor - time_petya_A_B_C) / (2 / kolya_fast_factor) = 1 := 
by 
    sorry

end NUMINAMATH_GPT_kolya_time_segment_DE_l1562_156284


namespace NUMINAMATH_GPT_rational_sum_zero_cube_nonzero_fifth_power_zero_l1562_156220

theorem rational_sum_zero_cube_nonzero_fifth_power_zero
  (a b c : ℚ) 
  (h_sum : a + b + c = 0)
  (h_cube_nonzero : a^3 + b^3 + c^3 ≠ 0) 
  : a^5 + b^5 + c^5 = 0 :=
sorry

end NUMINAMATH_GPT_rational_sum_zero_cube_nonzero_fifth_power_zero_l1562_156220


namespace NUMINAMATH_GPT_number_of_members_l1562_156234

theorem number_of_members (n : ℕ) (h : n^2 = 9801) : n = 99 :=
sorry

end NUMINAMATH_GPT_number_of_members_l1562_156234


namespace NUMINAMATH_GPT_inequality_part_1_inequality_part_2_l1562_156209

theorem inequality_part_1 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≥ 1 := by
sorry

theorem inequality_part_2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  (a^2 / (b + c)) + (b^2 / (a + c)) + (c^2 / (a + b)) ≥ 1 / 2 := by
sorry

end NUMINAMATH_GPT_inequality_part_1_inequality_part_2_l1562_156209


namespace NUMINAMATH_GPT_min_value_of_expression_l1562_156263

open Classical

theorem min_value_of_expression (x : ℝ) (hx : x > 0) : 
  ∃ y, x + 16 / (x + 1) = y ∧ ∀ z, (z > 0 → z + 16 / (z + 1) ≥ y) := 
by
  use 7
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1562_156263


namespace NUMINAMATH_GPT_pipe_B_fill_time_l1562_156290

theorem pipe_B_fill_time (T : ℕ) (h1 : 50 > 0) (h2 : 30 > 0)
  (h3 : (1/50 + 1/T = 1/30)) : T = 75 := 
sorry

end NUMINAMATH_GPT_pipe_B_fill_time_l1562_156290


namespace NUMINAMATH_GPT_remainder_of_division_l1562_156296

theorem remainder_of_division (dividend divisor quotient remainder : ℕ)
  (h1 : dividend = 55053)
  (h2 : divisor = 456)
  (h3 : quotient = 120)
  (h4 : remainder = dividend - divisor * quotient) : 
  remainder = 333 := by
  sorry

end NUMINAMATH_GPT_remainder_of_division_l1562_156296


namespace NUMINAMATH_GPT_quadratic_at_most_two_roots_l1562_156229

theorem quadratic_at_most_two_roots (a b c x1 x2 x3 : ℝ) (ha : a ≠ 0) 
(h1 : a * x1^2 + b * x1 + c = 0)
(h2 : a * x2^2 + b * x2 + c = 0)
(h3 : a * x3^2 + b * x3 + c = 0)
(h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : 
false :=
sorry

end NUMINAMATH_GPT_quadratic_at_most_two_roots_l1562_156229


namespace NUMINAMATH_GPT_total_ear_muffs_bought_l1562_156254

-- Define the number of ear muffs bought before December
def ear_muffs_before_dec : ℕ := 1346

-- Define the number of ear muffs bought during December
def ear_muffs_during_dec : ℕ := 6444

-- The total number of ear muffs bought by customers
theorem total_ear_muffs_bought : ear_muffs_before_dec + ear_muffs_during_dec = 7790 :=
by
  sorry

end NUMINAMATH_GPT_total_ear_muffs_bought_l1562_156254


namespace NUMINAMATH_GPT_positive_inequality_l1562_156243

open Real

/-- Given positive real numbers x, y, z such that xyz ≥ 1, prove that
    (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + x^2 + z^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0.
-/
theorem positive_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h : x * y * z ≥ 1) : 
  (x^5 - x^2) / (x^5 + y^2 + z^2) + 
  (y^5 - y^2) / (y^5 + x^2 + z^2) + 
  (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_positive_inequality_l1562_156243


namespace NUMINAMATH_GPT_cards_per_deck_l1562_156213

theorem cards_per_deck (decks : ℕ) (cards_per_layer : ℕ) (layers : ℕ) 
  (h_decks : decks = 16) 
  (h_cards_per_layer : cards_per_layer = 26) 
  (h_layers : layers = 32) 
  (total_cards_used : ℕ := cards_per_layer * layers) 
  (number_of_cards_per_deck : ℕ := total_cards_used / decks) :
  number_of_cards_per_deck = 52 :=
by 
  sorry

end NUMINAMATH_GPT_cards_per_deck_l1562_156213


namespace NUMINAMATH_GPT_std_deviation_calc_l1562_156238

theorem std_deviation_calc 
  (μ : ℝ) (σ : ℝ) (V : ℝ) (k : ℝ)
  (hμ : μ = 14.0)
  (hσ : σ = 1.5)
  (hV : V = 11)
  (hk : k = (μ - V) / σ) :
  k = 2 := by
  sorry

end NUMINAMATH_GPT_std_deviation_calc_l1562_156238


namespace NUMINAMATH_GPT_lara_yesterday_more_than_sarah_l1562_156241

variable (yesterdaySarah todaySarah todayLara : ℕ)
variable (cansDifference : ℕ)

axiom yesterdaySarah_eq : yesterdaySarah = 50
axiom todaySarah_eq : todaySarah = 40
axiom todayLara_eq : todayLara = 70
axiom cansDifference_eq : cansDifference = 20

theorem lara_yesterday_more_than_sarah :
  let totalCansYesterday := yesterdaySarah + todaySarah + cansDifference
  let laraYesterday := totalCansYesterday - yesterdaySarah
  laraYesterday - yesterdaySarah = 30 :=
by
  sorry

end NUMINAMATH_GPT_lara_yesterday_more_than_sarah_l1562_156241
