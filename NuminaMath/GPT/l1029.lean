import Mathlib

namespace NUMINAMATH_GPT_identically_zero_on_interval_l1029_102960

variable (f : ℝ → ℝ) (a b : ℝ)
variable (h_cont : ContinuousOn f (Set.Icc a b))
variable (h_int : ∀ n : ℕ, ∫ x in a..b, (x : ℝ)^n * f x = 0)

theorem identically_zero_on_interval : ∀ x ∈ Set.Icc a b, f x = 0 := 
by 
  sorry

end NUMINAMATH_GPT_identically_zero_on_interval_l1029_102960


namespace NUMINAMATH_GPT_Carla_total_marbles_l1029_102985

def initial_marbles : ℝ := 187.0
def bought_marbles : ℝ := 134.0

theorem Carla_total_marbles : initial_marbles + bought_marbles = 321.0 := 
by 
  sorry

end NUMINAMATH_GPT_Carla_total_marbles_l1029_102985


namespace NUMINAMATH_GPT_rational_add_positive_square_l1029_102963

theorem rational_add_positive_square (a : ℚ) : a^2 + 1 > 0 := by
  sorry

end NUMINAMATH_GPT_rational_add_positive_square_l1029_102963


namespace NUMINAMATH_GPT_ratio_of_border_to_tile_l1029_102986

variable {s d : ℝ}

theorem ratio_of_border_to_tile (h1 : 900 = 30 * 30)
  (h2 : 0.81 = (900 * s^2) / (30 * s + 60 * d)^2) :
  d / s = 1 / 18 := by {
  sorry }

end NUMINAMATH_GPT_ratio_of_border_to_tile_l1029_102986


namespace NUMINAMATH_GPT_solve_for_constants_l1029_102939

theorem solve_for_constants : 
  ∃ (t s : ℚ), (∀ x : ℚ, (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + 12) = 15 * x^4 + s * x^3 + 33 * x^2 + 12 * x + 108) ∧ 
  t = 37 / 5 ∧ 
  s = 11 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_constants_l1029_102939


namespace NUMINAMATH_GPT_ways_to_go_home_via_library_l1029_102900

def ways_from_school_to_library := 2
def ways_from_library_to_home := 3

theorem ways_to_go_home_via_library : 
  ways_from_school_to_library * ways_from_library_to_home = 6 :=
by 
  sorry

end NUMINAMATH_GPT_ways_to_go_home_via_library_l1029_102900


namespace NUMINAMATH_GPT_exists_excircle_radius_at_least_three_times_incircle_radius_l1029_102930

variable (a b c s T r ra rb rc : ℝ)
variable (ha : ra = T / (s - a))
variable (hb : rb = T / (s - b))
variable (hc : rc = T / (s - c))
variable (hincircle : r = T / s)

theorem exists_excircle_radius_at_least_three_times_incircle_radius
  (ha : ra = T / (s - a)) (hb : rb = T / (s - b)) (hc : rc = T / (s - c)) (hincircle : r = T / s) :
  ∃ rc, rc ≥ 3 * r :=
by {
  use rc,
  sorry
}

end NUMINAMATH_GPT_exists_excircle_radius_at_least_three_times_incircle_radius_l1029_102930


namespace NUMINAMATH_GPT_find_certain_number_l1029_102922

theorem find_certain_number (D S X : ℕ): 
  D = 20 → 
  S = 55 → 
  X + (D - S) = 3 * D - 90 →
  X = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1029_102922


namespace NUMINAMATH_GPT_tan_product_equals_three_l1029_102917

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_product_equals_three_l1029_102917


namespace NUMINAMATH_GPT_find_m_l1029_102926

def l1 (m x y: ℝ) : Prop := 2 * x + m * y - 2 = 0
def l2 (m x y: ℝ) : Prop := m * x + 2 * y - 1 = 0
def perpendicular (m : ℝ) : Prop :=
  let slope_l1 := -2 / m
  let slope_l2 := -m / 2
  slope_l1 * slope_l2 = -1

theorem find_m (m : ℝ) (h : perpendicular m) : m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_l1029_102926


namespace NUMINAMATH_GPT_det_scaled_matrix_l1029_102945

theorem det_scaled_matrix (a b c d : ℝ) (h : a * d - b * c = 5) : 
  (3 * a) * (3 * d) - (3 * b) * (3 * c) = 45 :=
by 
  sorry

end NUMINAMATH_GPT_det_scaled_matrix_l1029_102945


namespace NUMINAMATH_GPT_complete_the_square_l1029_102920

theorem complete_the_square (y : ℤ) : y^2 + 14 * y + 60 = (y + 7)^2 + 11 :=
by
  sorry

end NUMINAMATH_GPT_complete_the_square_l1029_102920


namespace NUMINAMATH_GPT_oil_to_water_ratio_in_bottle_D_l1029_102987

noncomputable def bottle_oil_water_ratio (CA : ℝ) (CB : ℝ) (CC : ℝ) (CD : ℝ) : ℝ :=
  let oil_A := (1 / 2) * CA
  let water_A := (1 / 2) * CA
  let oil_B := (1 / 4) * CB
  let water_B := (1 / 4) * CB
  let total_water_B := CB - oil_B - water_B
  let oil_C := (1 / 3) * CC
  let water_C := 0.4 * CC
  let total_water_C := CC - oil_C - water_C
  let total_capacity_D := CD
  let total_oil_D := oil_A + oil_B + oil_C
  let total_water_D := water_A + total_water_B + water_C + total_water_C
  total_oil_D / total_water_D

theorem oil_to_water_ratio_in_bottle_D (CA : ℝ) :
  let CB := 2 * CA
  let CC := 3 * CA
  let CD := CA + CC
  bottle_oil_water_ratio CA CB CC CD = (2 / 3.7) :=
by 
  sorry

end NUMINAMATH_GPT_oil_to_water_ratio_in_bottle_D_l1029_102987


namespace NUMINAMATH_GPT_total_number_of_items_l1029_102951

theorem total_number_of_items (total_items : ℕ) (selected_items : ℕ) (h1 : total_items = 50) (h2 : selected_items = 10) : total_items = 50 :=
by
  exact h1

end NUMINAMATH_GPT_total_number_of_items_l1029_102951


namespace NUMINAMATH_GPT_man_speed_with_current_l1029_102984

-- Define the conditions
def current_speed : ℕ := 3
def man_speed_against_current : ℕ := 14

-- Define the man's speed in still water (v) based on the given speed against the current
def man_speed_in_still_water : ℕ := man_speed_against_current + current_speed

-- Prove that the man's speed with the current is 20 kmph
theorem man_speed_with_current : man_speed_in_still_water + current_speed = 20 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end NUMINAMATH_GPT_man_speed_with_current_l1029_102984


namespace NUMINAMATH_GPT_arithmetic_mean_of_primes_l1029_102940

variable (list : List ℕ) 
variable (primes : List ℕ)
variable (h1 : list = [24, 25, 29, 31, 33])
variable (h2 : primes = [29, 31])

theorem arithmetic_mean_of_primes : (primes.sum / primes.length : ℝ) = 30 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_primes_l1029_102940


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1029_102973

theorem sufficient_but_not_necessary_condition (a : ℝ) (h₁ : a > 2) : a ≥ 1 ∧ ¬(∀ (a : ℝ), a ≥ 1 → a > 2) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1029_102973


namespace NUMINAMATH_GPT_minimize_quadratic_expression_l1029_102997

theorem minimize_quadratic_expression:
  ∀ x : ℝ, (∃ a b c : ℝ, a = 1 ∧ b = -8 ∧ c = 15 ∧ x^2 + b * x + c ≥ (4 - 4)^2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_minimize_quadratic_expression_l1029_102997


namespace NUMINAMATH_GPT_range_of_a_l1029_102913

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x - a > 0) → a < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1029_102913


namespace NUMINAMATH_GPT_problem_statement_l1029_102944

noncomputable def f (n : ℕ) : ℝ := Real.log (n^2) / Real.log 3003

theorem problem_statement : f 33 + f 13 + f 7 = 2 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1029_102944


namespace NUMINAMATH_GPT_number_of_students_in_club_l1029_102910

variable (y : ℕ) -- Number of girls

def total_stickers_given (y : ℕ) : ℕ := y * y + (y + 3) * (y + 3)

theorem number_of_students_in_club :
  (total_stickers_given y = 640) → (2 * y + 3 = 35) := 
by
  intro h1
  sorry

end NUMINAMATH_GPT_number_of_students_in_club_l1029_102910


namespace NUMINAMATH_GPT_right_triangle_medians_l1029_102999

theorem right_triangle_medians
    (a b c d m : ℝ)
    (h1 : ∀(a b c d : ℝ), 2 * (c/d) = 3)
    (h2 : m = 4 * 3 ∨ m = (3/4)) :
    ∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ (m₁ = 12 ∨ m₁ = 3/4) ∧ (m₂ = 12 ∨ m₂ = 3/4) :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_medians_l1029_102999


namespace NUMINAMATH_GPT_sector_area_proof_l1029_102909

-- Define the sector with its characteristics
structure sector :=
  (r : ℝ)            -- radius
  (theta : ℝ)        -- central angle

-- Given conditions
def sector_example : sector := {r := 1, theta := 2}

-- Definition of perimeter for a sector
def perimeter (sec : sector) : ℝ :=
  2 * sec.r + sec.theta * sec.r

-- Definition of area for a sector
def area (sec : sector) : ℝ :=
  0.5 * sec.r * (sec.theta * sec.r)

-- Theorem statement based on the problem statement
theorem sector_area_proof (sec : sector) (h1 : perimeter sec = 4) (h2 : sec.theta = 2) : area sec = 1 := 
  sorry

end NUMINAMATH_GPT_sector_area_proof_l1029_102909


namespace NUMINAMATH_GPT_initial_books_l1029_102929

-- Definitions for the conditions.

def boxes (b : ℕ) : ℕ := 3 * b -- Box count
def booksInRoom : ℕ := 21 -- Books in the room
def booksOnTable : ℕ := 4 -- Books on the coffee table
def cookbooks : ℕ := 18 -- Cookbooks in the kitchen
def booksGrabbed : ℕ := 12 -- Books grabbed from the donation center
def booksNow : ℕ := 23 -- Books Henry has now

-- Define total number of books donated
def totalBooksDonated (inBoxes : ℕ) (additionalBooks : ℕ) : ℕ :=
  inBoxes + additionalBooks - booksGrabbed

-- Define number of books Henry initially had
def initialBooks (netDonated : ℕ) (booksCurrently : ℕ) : ℕ :=
  netDonated + booksCurrently

-- Proof goal
theorem initial_books (b : ℕ) (inBox : ℕ) (additionalBooks : ℕ) : 
  let totalBooks := booksInRoom + booksOnTable + cookbooks
  let inBoxes := boxes b
  let totalDonated := totalBooksDonated inBoxes totalBooks
  initialBooks totalDonated booksNow = 99 :=
by 
  simp [initialBooks, totalBooksDonated, boxes, booksInRoom, booksOnTable, cookbooks, booksGrabbed, booksNow]
  sorry

end NUMINAMATH_GPT_initial_books_l1029_102929


namespace NUMINAMATH_GPT_C_plus_D_l1029_102957

theorem C_plus_D (D C : ℚ) (h1 : ∀ x : ℚ, (Dx - 17) / ((x - 2) * (x - 4)) = C / (x - 2) + 4 / (x - 4))
  (h2 : ∀ x : ℚ, (x - 2) * (x - 4) = x^2 - 6 * x + 8) :
  C + D = 8.5 := sorry

end NUMINAMATH_GPT_C_plus_D_l1029_102957


namespace NUMINAMATH_GPT_sphere_volume_l1029_102949

theorem sphere_volume {r : ℝ} (h: 4 * Real.pi * r^2 = 256 * Real.pi) : (4 / 3) * Real.pi * r^3 = (2048 / 3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_volume_l1029_102949


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l1029_102971

theorem perpendicular_line_through_point 
 {x y : ℝ}
 (p : (ℝ × ℝ)) 
 (point : p = (-2, 1)) 
 (perpendicular : ∀ x y, 2 * x - y + 4 = 0) : 
 (∀ x y, x + 2 * y = 0) ∧ (p.fst = -2 ∧ p.snd = 1) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l1029_102971


namespace NUMINAMATH_GPT_scientific_notation_6500_l1029_102935

theorem scientific_notation_6500 : (6500 : ℝ) = 6.5 * 10^3 := 
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_6500_l1029_102935


namespace NUMINAMATH_GPT_original_price_of_house_l1029_102952

theorem original_price_of_house (P : ℝ) 
  (h1 : P * 0.56 = 56000) : P = 100000 :=
sorry

end NUMINAMATH_GPT_original_price_of_house_l1029_102952


namespace NUMINAMATH_GPT_cyclist_C_speed_l1029_102964

variable (c d : ℕ)

def distance_to_meeting (c d : ℕ) : Prop :=
  d = c + 6 ∧
  90 + 30 = 120 ∧
  ((90 - 30) / c) = (120 / d) ∧
  (60 / c) = (120 / (c + 6))

theorem cyclist_C_speed : distance_to_meeting c d → c = 6 :=
by
  intro h
  -- To be filled in with the proof using the conditions
  sorry

end NUMINAMATH_GPT_cyclist_C_speed_l1029_102964


namespace NUMINAMATH_GPT_min_n_consecutive_integers_sum_of_digits_is_multiple_of_8_l1029_102904

theorem min_n_consecutive_integers_sum_of_digits_is_multiple_of_8 
: ∃ n : ℕ, (∀ (nums : Fin n.succ → ℕ), 
              (∀ i j, i < j → nums i < nums j → nums j = nums i + 1) →
              ∃ i, (nums i) % 8 = 0) ∧ n = 15 := 
sorry

end NUMINAMATH_GPT_min_n_consecutive_integers_sum_of_digits_is_multiple_of_8_l1029_102904


namespace NUMINAMATH_GPT_greatest_possible_value_of_x_l1029_102977

theorem greatest_possible_value_of_x : 
  (∀ x : ℚ, ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20) → 
  x ≤ 9/5 := sorry

end NUMINAMATH_GPT_greatest_possible_value_of_x_l1029_102977


namespace NUMINAMATH_GPT_current_average_is_35_l1029_102938

noncomputable def cricket_avg (A : ℝ) : Prop :=
  let innings := 10
  let next_runs := 79
  let increase := 4
  (innings * A + next_runs = (A + increase) * (innings + 1))

theorem current_average_is_35 : cricket_avg 35 :=
by
  unfold cricket_avg
  simp only
  sorry

end NUMINAMATH_GPT_current_average_is_35_l1029_102938


namespace NUMINAMATH_GPT_range_of_third_side_l1029_102925

theorem range_of_third_side (y : ℝ) : (2 < y) ↔ (y < 8) :=
by sorry

end NUMINAMATH_GPT_range_of_third_side_l1029_102925


namespace NUMINAMATH_GPT_no_snuggly_numbers_l1029_102902

def isSnuggly (n : Nat) : Prop :=
  ∃ (a b : Nat), 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    n = 10 * a + b ∧ 
    n = a + b^3 + 5

theorem no_snuggly_numbers : 
  ¬ ∃ n : Nat, 10 ≤ n ∧ n < 100 ∧ isSnuggly n :=
by
  sorry

end NUMINAMATH_GPT_no_snuggly_numbers_l1029_102902


namespace NUMINAMATH_GPT_krishan_money_l1029_102966

/-- Given that the ratio of money between Ram and Gopal is 7:17, the ratio of money between Gopal and Krishan is 7:17, and Ram has Rs. 588, prove that Krishan has Rs. 12,065. -/
theorem krishan_money (R G K : ℝ) (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : R = 588) : K = 12065 :=
by
  sorry

end NUMINAMATH_GPT_krishan_money_l1029_102966


namespace NUMINAMATH_GPT_inequality_solution_set_l1029_102901

theorem inequality_solution_set (x : ℝ) : (x^2 + x) / (2*x - 1) ≤ 1 ↔ x < 1 / 2 := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1029_102901


namespace NUMINAMATH_GPT_initial_percentage_filled_l1029_102970

theorem initial_percentage_filled (capacity : ℝ) (added : ℝ) (final_fraction : ℝ) (initial_water : ℝ) :
  capacity = 80 → added = 20 → final_fraction = 3/4 → 
  initial_water = (final_fraction * capacity - added) → 
  100 * (initial_water / capacity) = 50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_percentage_filled_l1029_102970


namespace NUMINAMATH_GPT_calculate_students_l1029_102921

noncomputable def handshakes (m n : ℕ) : ℕ :=
  1/2 * (4 * 3 + 5 * (2 * (m - 2) + 2 * (n - 2)) + 8 * (m - 2) * (n - 2))

theorem calculate_students (m n : ℕ) (h_m : 3 ≤ m) (h_n : 3 ≤ n) (h_handshakes : handshakes m n = 1020) : m * n = 140 :=
by
  sorry

end NUMINAMATH_GPT_calculate_students_l1029_102921


namespace NUMINAMATH_GPT_max_value_of_z_l1029_102915

theorem max_value_of_z (x y z : ℝ) (h_add : x + y + z = 5) (h_mult : x * y + y * z + z * x = 3) : z ≤ 13 / 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_z_l1029_102915


namespace NUMINAMATH_GPT_find_x_l1029_102995

theorem find_x : ∃ x : ℕ, x + 1 = 5 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1029_102995


namespace NUMINAMATH_GPT_multiple_of_students_in_restroom_l1029_102946

theorem multiple_of_students_in_restroom 
    (num_desks_per_row : ℕ)
    (num_rows : ℕ)
    (desk_fill_fraction : ℚ)
    (total_students : ℕ)
    (students_restroom : ℕ)
    (absent_students : ℕ)
    (m : ℕ) :
    num_desks_per_row = 6 →
    num_rows = 4 →
    desk_fill_fraction = 2 / 3 →
    total_students = 23 →
    students_restroom = 2 →
    (num_rows * num_desks_per_row : ℕ) * desk_fill_fraction = 16 →
    (16 - students_restroom) = 14 →
    total_students - 14 - 2 = absent_students →
    absent_students = 7 →
    2 * m - 1 = 7 →
    m = 4
:= by
    intros;
    sorry

end NUMINAMATH_GPT_multiple_of_students_in_restroom_l1029_102946


namespace NUMINAMATH_GPT_bn_is_arithmetic_an_general_formula_l1029_102965

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end NUMINAMATH_GPT_bn_is_arithmetic_an_general_formula_l1029_102965


namespace NUMINAMATH_GPT_find_a_minus_b_l1029_102983

theorem find_a_minus_b (a b : ℝ) :
  (∀ (x : ℝ), x^4 - 8 * x^3 + a * x^2 + b * x + 16 = 0 → x > 0) →
  a - b = 56 :=
by
  sorry

end NUMINAMATH_GPT_find_a_minus_b_l1029_102983


namespace NUMINAMATH_GPT_intersection_of_sets_l1029_102942

def A (x : ℝ) : Prop := x > -2
def B (x : ℝ) : Prop := 1 - x > 0

theorem intersection_of_sets :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | x > -2 ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1029_102942


namespace NUMINAMATH_GPT_problem1_problem2_problem2_equality_l1029_102958

variable {a b c d : ℝ}

-- Problem 1
theorem problem1 (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : (a - b) * (b - c) * (c - d) * (d - a) = -3) (h5 : a + b + c + d = 6) : d < 0.36 :=
sorry

-- Problem 2
theorem problem2 (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : (a - b) * (b - c) * (c - d) * (d - a) = -3) (h5 : a^2 + b^2 + c^2 + d^2 = 14) : (a + c) * (b + d) ≤ 8 :=
sorry

theorem problem2_equality (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) (h4 : d = 0) : (a + c) * (b + d) = 8 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem2_equality_l1029_102958


namespace NUMINAMATH_GPT_intersection_A_B_l1029_102953

-- Define sets A and B
def A : Set ℤ := {1, 3, 5}
def B : Set ℤ := {-1, 0, 1}

-- Prove that the intersection of A and B is {1}
theorem intersection_A_B : A ∩ B = {1} := by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1029_102953


namespace NUMINAMATH_GPT_problem_statement_l1029_102956

theorem problem_statement
  (a b A B : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ θ : ℝ, f θ ≥ 0)
  (def_f : ∀ θ : ℝ, f θ = 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ)) :
  a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1029_102956


namespace NUMINAMATH_GPT_intersection_S_T_l1029_102950

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end NUMINAMATH_GPT_intersection_S_T_l1029_102950


namespace NUMINAMATH_GPT_last_three_digits_of_8_pow_104_l1029_102982

def last_three_digits_of_pow (x n : ℕ) : ℕ :=
  (x ^ n) % 1000

theorem last_three_digits_of_8_pow_104 : last_three_digits_of_pow 8 104 = 984 := 
by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_8_pow_104_l1029_102982


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_plus_2_l1029_102907

variable (D E F : ℝ)

def q (x : ℝ) := D * x^4 + E * x^2 + F * x + 7

theorem remainder_when_divided_by_x_plus_2 :
  q D E F (-2) = 21 - 2 * F :=
by
  have hq2 : q D E F 2 = 21 := sorry
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_plus_2_l1029_102907


namespace NUMINAMATH_GPT_smallest_five_digit_in_pascals_triangle_l1029_102912

/-- In Pascal's triangle, the smallest five-digit number is 10000. -/
theorem smallest_five_digit_in_pascals_triangle : 
  ∃ (n k : ℕ), (10000 = Nat.choose n k) ∧ (∀ m l : ℕ, Nat.choose m l < 10000) → (n > m) := 
sorry

end NUMINAMATH_GPT_smallest_five_digit_in_pascals_triangle_l1029_102912


namespace NUMINAMATH_GPT_partnership_profit_l1029_102988

noncomputable def total_profit
  (P : ℝ)
  (mary_investment : ℝ := 700)
  (harry_investment : ℝ := 300)
  (effort_share := P / 3 / 2)
  (remaining_share := 2 / 3 * P)
  (total_investment := mary_investment + harry_investment)
  (mary_share_remaining := (mary_investment / total_investment) * remaining_share)
  (harry_share_remaining := (harry_investment / total_investment) * remaining_share) : Prop :=
  (effort_share + mary_share_remaining) - (effort_share + harry_share_remaining) = 800

theorem partnership_profit : ∃ P : ℝ, total_profit P ∧ P = 3000 :=
  sorry

end NUMINAMATH_GPT_partnership_profit_l1029_102988


namespace NUMINAMATH_GPT_relationship_C1_C2_A_l1029_102943

variables (A B C C1 C2 : ℝ)

-- Given conditions
def TriangleABC : Prop := B = 2 * A
def AngleSumProperty : Prop := A + B + C = 180
def AltitudeDivides := C1 = 90 - A ∧ C2 = 90 - 2 * A

-- Theorem to prove the relationship between C1, C2, and A
theorem relationship_C1_C2_A (h1: TriangleABC A B) (h2: AngleSumProperty A B C) (h3: AltitudeDivides C1 C2 A) : 
  C1 - C2 = A :=
by sorry

end NUMINAMATH_GPT_relationship_C1_C2_A_l1029_102943


namespace NUMINAMATH_GPT_complex_expression_evaluation_l1029_102948

noncomputable def imaginary_i := Complex.I 

theorem complex_expression_evaluation : 
  ((2 + imaginary_i) / (1 - imaginary_i)) - (1 - imaginary_i) = -1/2 + (5/2) * imaginary_i :=
by 
  sorry

end NUMINAMATH_GPT_complex_expression_evaluation_l1029_102948


namespace NUMINAMATH_GPT_new_box_volume_eq_5_76_m3_l1029_102989

-- Given conditions:
def original_width_cm := 80
def original_length_cm := 75
def original_height_cm := 120
def conversion_factor_cm3_to_m3 := 1000000

-- New dimensions after doubling
def new_width_cm := 2 * original_width_cm
def new_length_cm := 2 * original_length_cm
def new_height_cm := 2 * original_height_cm

-- Statement of the problem
theorem new_box_volume_eq_5_76_m3 :
  (new_width_cm * new_length_cm * new_height_cm : ℝ) / conversion_factor_cm3_to_m3 = 5.76 := 
  sorry

end NUMINAMATH_GPT_new_box_volume_eq_5_76_m3_l1029_102989


namespace NUMINAMATH_GPT_cos_2x_eq_cos_2y_l1029_102998

theorem cos_2x_eq_cos_2y (x y : ℝ) 
  (h1 : Real.sin x + Real.cos y = 1) 
  (h2 : Real.cos x + Real.sin y = -1) : 
  Real.cos (2 * x) = Real.cos (2 * y) := by
  sorry

end NUMINAMATH_GPT_cos_2x_eq_cos_2y_l1029_102998


namespace NUMINAMATH_GPT_committee_probability_l1029_102955

/--
Suppose there are 24 members in a club: 12 boys and 12 girls.
A 5-person committee is chosen at random.
Prove that the probability of having at least 2 boys and at least 2 girls in the committee is 121/177.
-/
theorem committee_probability :
  let boys := 12
  let girls := 12
  let total_members := 24
  let committee_size := 5
  let all_ways := Nat.choose total_members committee_size
  let invalid_ways := 2 * Nat.choose boys committee_size + 2 * (Nat.choose boys 1 * Nat.choose girls 4)
  let valid_ways := all_ways - invalid_ways
  let probability := valid_ways / all_ways
  probability = 121 / 177 :=
by
  sorry

end NUMINAMATH_GPT_committee_probability_l1029_102955


namespace NUMINAMATH_GPT_perimeter_of_octagon_l1029_102936

theorem perimeter_of_octagon :
  let base := 10
  let left_side := 9
  let right_side := 11
  let top_left_diagonal := 6
  let top_right_diagonal := 7
  let small_side1 := 2
  let small_side2 := 3
  let small_side3 := 4
  base + left_side + right_side + top_left_diagonal + top_right_diagonal + small_side1 + small_side2 + small_side3 = 52 :=
by
  -- This automatically assumes all the definitions and shows the equation
  sorry

end NUMINAMATH_GPT_perimeter_of_octagon_l1029_102936


namespace NUMINAMATH_GPT_find_p_q_l1029_102916

theorem find_p_q : 
  (∀ x : ℝ, (x - 2) * (x + 1) ∣ (x ^ 5 - x ^ 4 + x ^ 3 - p * x ^ 2 + q * x - 8)) → (p = -1 ∧ q = -10) :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_l1029_102916


namespace NUMINAMATH_GPT_dave_won_tickets_l1029_102924

theorem dave_won_tickets (initial_tickets spent_tickets final_tickets won_tickets : ℕ) 
  (h1 : initial_tickets = 25) 
  (h2 : spent_tickets = 22) 
  (h3 : final_tickets = 18) 
  (h4 : won_tickets = final_tickets - (initial_tickets - spent_tickets)) :
  won_tickets = 15 := 
by 
  sorry

end NUMINAMATH_GPT_dave_won_tickets_l1029_102924


namespace NUMINAMATH_GPT_parabola_slope_l1029_102932

theorem parabola_slope (p k : ℝ) (h1 : p > 0)
  (h_focus_distance : (p / 2) * (3^(1/2)) / (3 + 1^(1/2))^(1/2) = 3^(1/2))
  (h_AF_FB : exists A B : ℝ × ℝ, (A.1 = 2 - p / 2 ∧ 2 * (B.1 - 2) = 2)
    ∧ (A.2 = p - p / 2 ∧ A.2 = -2 * B.2)) :
  abs k = 2 * (2^(1/2)) :=
sorry

end NUMINAMATH_GPT_parabola_slope_l1029_102932


namespace NUMINAMATH_GPT_eq_abs_distinct_solution_count_l1029_102969

theorem eq_abs_distinct_solution_count :
  ∃! x : ℝ, |x - 10| = |x + 5| + 2 := 
sorry

end NUMINAMATH_GPT_eq_abs_distinct_solution_count_l1029_102969


namespace NUMINAMATH_GPT_kathryn_more_pints_than_annie_l1029_102961

-- Definitions for conditions
def annie_pints : ℕ := 8
def ben_pints (kathryn_pints : ℕ) : ℕ := kathryn_pints - 3
def total_pints (annie_pints kathryn_pints ben_pints : ℕ) : ℕ := annie_pints + kathryn_pints + ben_pints

-- The problem statement
theorem kathryn_more_pints_than_annie (k : ℕ) (h1 : total_pints annie_pints k (ben_pints k) = 25) : k - annie_pints = 2 :=
sorry

end NUMINAMATH_GPT_kathryn_more_pints_than_annie_l1029_102961


namespace NUMINAMATH_GPT_elder_age_is_30_l1029_102979

-- Define the ages of the younger and elder persons
variables (y e : ℕ)

-- We have the following conditions:
-- Condition 1: The elder's age is 16 years more than the younger's age
def age_difference := e = y + 16

-- Condition 2: Six years ago, the elder's age was three times the younger's age
def six_years_ago := e - 6 = 3 * (y - 6)

-- We need to prove that the present age of the elder person is 30
theorem elder_age_is_30 (y e : ℕ) (h1 : age_difference y e) (h2 : six_years_ago y e) : e = 30 :=
sorry

end NUMINAMATH_GPT_elder_age_is_30_l1029_102979


namespace NUMINAMATH_GPT_line_containing_chord_l1029_102975

variable {x y x₁ y₁ x₂ y₂ : ℝ}

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 4 = 1)

def midpoint_condition (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) : Prop := 
  (x₁ + x₂ = 2) ∧ (y₁ + y₂ = 2)

theorem line_containing_chord (h₁ : ellipse_eq x₁ y₁) 
                               (h₂ : ellipse_eq x₂ y₂) 
                               (hmp : midpoint_condition x₁ x₂ y₁ y₂)
    : 4 * 1 + 9 * 1 - 13 = 0 := 
sorry

end NUMINAMATH_GPT_line_containing_chord_l1029_102975


namespace NUMINAMATH_GPT_min_cans_for_gallon_l1029_102976

-- Define conditions
def can_capacity : ℕ := 12
def gallon_to_ounces : ℕ := 128

-- Define the minimum number of cans function.
def min_cans (capacity : ℕ) (required : ℕ) : ℕ :=
  (required + capacity - 1) / capacity -- This is the ceiling of required / capacity

-- Statement asserting the required minimum number of cans.
theorem min_cans_for_gallon (h : min_cans can_capacity gallon_to_ounces = 11) : 
  can_capacity > 0 ∧ gallon_to_ounces > 0 := by
  sorry

end NUMINAMATH_GPT_min_cans_for_gallon_l1029_102976


namespace NUMINAMATH_GPT_carol_lollipops_l1029_102941

theorem carol_lollipops (total_lollipops : ℝ) (first_day_lollipops : ℝ) (delta_lollipops : ℝ) :
  total_lollipops = 150 → delta_lollipops = 5 →
  (first_day_lollipops + (first_day_lollipops + 5) + (first_day_lollipops + 10) +
  (first_day_lollipops + 15) + (first_day_lollipops + 20) + (first_day_lollipops + 25) = total_lollipops) →
  (first_day_lollipops = 12.5) →
  (first_day_lollipops + 15 = 27.5) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_carol_lollipops_l1029_102941


namespace NUMINAMATH_GPT_perfect_square_of_factorials_l1029_102908

open Nat

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfect_square_of_factorials :
  let E1 := factorial 98 * factorial 99
  let E2 := factorial 98 * factorial 100
  let E3 := factorial 99 * factorial 100
  let E4 := factorial 99 * factorial 101
  let E5 := factorial 100 * factorial 101
  is_perfect_square E3 :=
by
  -- definition of E1, E2, E3, E4, E5 as expressions given conditions
  let E1 := factorial 98 * factorial 99
  let E2 := factorial 98 * factorial 100
  let E3 := factorial 99 * factorial 100
  let E4 := factorial 99 * factorial 101
  let E5 := factorial 100 * factorial 101
  
  -- specify that E3 is the perfect square
  show is_perfect_square E3

  sorry

end NUMINAMATH_GPT_perfect_square_of_factorials_l1029_102908


namespace NUMINAMATH_GPT_discount_percentage_l1029_102972

theorem discount_percentage (SP CP SP' discount_gain_percentage: ℝ) 
  (h1 : SP = 30) 
  (h2 : SP = CP + 0.25 * CP) 
  (h3 : SP' = CP + 0.125 * CP) 
  (h4 : discount_gain_percentage = ((SP - SP') / SP) * 100) :
  discount_gain_percentage = 10 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_discount_percentage_l1029_102972


namespace NUMINAMATH_GPT_complex_number_quadrant_l1029_102959

theorem complex_number_quadrant :
  let z := (2 - (1 * Complex.I)) / (1 + (1 * Complex.I))
  (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end NUMINAMATH_GPT_complex_number_quadrant_l1029_102959


namespace NUMINAMATH_GPT_multiples_of_6_or_8_but_not_both_l1029_102947

theorem multiples_of_6_or_8_but_not_both (n : ℕ) : 
  n = 25 ∧ (n = 18) ∧ (n = 6) → (25 - 6) + (18 - 6) = 31 :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_6_or_8_but_not_both_l1029_102947


namespace NUMINAMATH_GPT_square_side_length_l1029_102996

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) : ∃ s : ℝ, s = 2 :=
by 
  sorry

end NUMINAMATH_GPT_square_side_length_l1029_102996


namespace NUMINAMATH_GPT_quadratic_one_pos_one_neg_l1029_102928

theorem quadratic_one_pos_one_neg (a : ℝ) : 
  (a < -1) → (∃ x1 x2 : ℝ, x1 * x2 < 0 ∧ x1 + x2 > 0 ∧ (x1^2 + x1 + a = 0 ∧ x2^2 + x2 + a = 0)) :=
sorry

end NUMINAMATH_GPT_quadratic_one_pos_one_neg_l1029_102928


namespace NUMINAMATH_GPT_necessary_and_sufficient_conditions_l1029_102918

-- Definitions for sets A and B
def U : Set (ℝ × ℝ) := {p | true}

def A (m : ℝ) : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + m > 0}

def B (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n ≤ 0}

-- Given point P(2, 3)
def P : ℝ × ℝ := (2, 3)

-- Complement of B
def B_complement (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n > 0}

-- Intersection of A and complement of B
def A_inter_B_complement (m n : ℝ) : Set (ℝ × ℝ) := A m ∩ B_complement n

-- Theorem stating the necessary and sufficient conditions for P to belong to A ∩ (complement of B)
theorem necessary_and_sufficient_conditions (m n : ℝ) : 
  P ∈ A_inter_B_complement m n ↔ m > -1 ∧ n < 5 :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_conditions_l1029_102918


namespace NUMINAMATH_GPT_num_factors_of_2310_with_more_than_three_factors_l1029_102991

theorem num_factors_of_2310_with_more_than_three_factors : 
  (∃ n : ℕ, n > 0 ∧ ∀ d : ℕ, d ∣ 2310 → (∀ f : ℕ, f ∣ d → f = 1 ∨ f = d ∨ f ∣ d) → 26 = n) := sorry

end NUMINAMATH_GPT_num_factors_of_2310_with_more_than_three_factors_l1029_102991


namespace NUMINAMATH_GPT_Mike_onions_grew_l1029_102914

-- Define the data:
variables (nancy_onions dan_onions total_onions mike_onions : ℕ)

-- Conditions:
axiom Nancy_onions_grew : nancy_onions = 2
axiom Dan_onions_grew : dan_onions = 9
axiom Total_onions_grew : total_onions = 15

-- Theorem to prove:
theorem Mike_onions_grew (h : total_onions = nancy_onions + dan_onions + mike_onions) : mike_onions = 4 :=
by
  -- The proof is not provided, so we use sorry:
  sorry

end NUMINAMATH_GPT_Mike_onions_grew_l1029_102914


namespace NUMINAMATH_GPT_triangle_side_lengths_l1029_102937

-- Define the variables a, b, and c
variables {a b c : ℝ}

-- Assume that a, b, and c are the lengths of the sides of a triangle
-- and the given equation holds
theorem triangle_side_lengths (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) 
    (h_eq : a^2 + 4*a*c + 3*c^2 - 3*a*b - 7*b*c + 2*b^2 = 0) : 
    a + c - 2*b = 0 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l1029_102937


namespace NUMINAMATH_GPT_cone_height_l1029_102962

theorem cone_height (r : ℝ) (θ : ℝ) (h : ℝ)
  (hr : r = 1)
  (hθ : θ = (2 / 3) * Real.pi)
  (h_eq : h = 2 * Real.sqrt 2) :
  ∃ l : ℝ, l = 3 ∧ h = Real.sqrt (l^2 - r^2) :=
by
  sorry

end NUMINAMATH_GPT_cone_height_l1029_102962


namespace NUMINAMATH_GPT_complement_intersection_l1029_102934

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2 * x > 0}

def B : Set ℝ := {x | -3 < x ∧ x < 1}

def compA : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_intersection :
  (compA ∩ B) = {x | 0 ≤ x ∧ x < 1} := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_complement_intersection_l1029_102934


namespace NUMINAMATH_GPT_no_valid_digit_replacement_l1029_102990

theorem no_valid_digit_replacement :
  ¬ ∃ (A B C D E M X : ℕ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ M ∧ A ≠ X ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ M ∧ B ≠ X ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ M ∧ C ≠ X ∧
     D ≠ E ∧ D ≠ M ∧ D ≠ X ∧
     E ≠ M ∧ E ≠ X ∧
     M ≠ X ∧
     0 ≤ A ∧ A < 10 ∧
     0 ≤ B ∧ B < 10 ∧
     0 ≤ C ∧ C < 10 ∧
     0 ≤ D ∧ D < 10 ∧
     0 ≤ E ∧ E < 10 ∧
     0 ≤ M ∧ M < 10 ∧
     0 ≤ X ∧ X < 10 ∧
     A * B * C * D + 1 = C * E * M * X) :=
sorry

end NUMINAMATH_GPT_no_valid_digit_replacement_l1029_102990


namespace NUMINAMATH_GPT_average_death_rate_l1029_102981

variable (birth_rate : ℕ) (net_increase_day : ℕ)

noncomputable def death_rate_per_two_seconds (birth_rate net_increase_day : ℕ) : ℕ :=
  let seconds_per_day := 86400
  let net_increase_per_second := net_increase_day / seconds_per_day
  let birth_rate_per_second := birth_rate / 2
  let death_rate_per_second := birth_rate_per_second - net_increase_per_second
  2 * death_rate_per_second

theorem average_death_rate
  (birth_rate : ℕ := 4) 
  (net_increase_day : ℕ := 86400) :
  death_rate_per_two_seconds birth_rate net_increase_day = 2 :=
sorry

end NUMINAMATH_GPT_average_death_rate_l1029_102981


namespace NUMINAMATH_GPT_neither_necessary_nor_sufficient_l1029_102954

theorem neither_necessary_nor_sufficient (x : ℝ) :
  ¬ ((-1 < x ∧ x < 2) → (|x - 2| < 1)) ∧ ¬ ((|x - 2| < 1) → (-1 < x ∧ x < 2)) :=
by
  sorry

end NUMINAMATH_GPT_neither_necessary_nor_sufficient_l1029_102954


namespace NUMINAMATH_GPT_larger_pie_flour_amount_l1029_102923

variable (p1 : ℕ) (f1 : ℚ) (p2 : ℕ) (f2 : ℚ)

def prepared_pie_crusts (p1 p2 : ℕ) (f1 : ℚ) (f2 : ℚ) : Prop :=
  p1 * f1 = p2 * f2

theorem larger_pie_flour_amount (h : prepared_pie_crusts 40 25 (1/8) f2) : f2 = 1/5 :=
by
  sorry

end NUMINAMATH_GPT_larger_pie_flour_amount_l1029_102923


namespace NUMINAMATH_GPT_multiplication_by_9_l1029_102903

theorem multiplication_by_9 (n : ℕ) (h1 : n < 10) : 9 * n = 10 * (n - 1) + (10 - n) := 
sorry

end NUMINAMATH_GPT_multiplication_by_9_l1029_102903


namespace NUMINAMATH_GPT_find_f_pi_over_4_l1029_102919

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem find_f_pi_over_4
  (ω φ : ℝ)
  (hω_gt_0 : ω > 0)
  (hφ_lt_pi_over_2 : |φ| < Real.pi / 2)
  (h_mono_dec : ∀ x₁ x₂, (Real.pi / 6 < x₁ ∧ x₁ < Real.pi / 3 ∧ Real.pi / 3 < x₂ ∧ x₂ < 2 * Real.pi / 3) → f x₁ ω φ > f x₂ ω φ)
  (h_values_decreasing : f (Real.pi / 6) ω φ = 1 ∧ f (2 * Real.pi / 3) ω φ = -1) : 
  f (Real.pi / 4) 2 (Real.pi / 6) = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_f_pi_over_4_l1029_102919


namespace NUMINAMATH_GPT_find_first_number_l1029_102911

theorem find_first_number : ∃ x : ℕ, x + 7314 = 3362 + 13500 ∧ x = 9548 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_find_first_number_l1029_102911


namespace NUMINAMATH_GPT_shaded_area_percentage_l1029_102967

def area_square (side : ℕ) : ℕ := side * side

def shaded_percentage (total_area shaded_area : ℕ) : ℚ :=
  ((shaded_area : ℚ) / total_area) * 100 

theorem shaded_area_percentage (side : ℕ) (total_area : ℕ) (shaded_area : ℕ) 
  (h_side : side = 7) (h_total_area : total_area = area_square side) 
  (h_shaded_area : shaded_area = 4 + 16 + 13) : 
  shaded_percentage total_area shaded_area = 3300 / 49 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_shaded_area_percentage_l1029_102967


namespace NUMINAMATH_GPT_age_ratio_problem_l1029_102978

def age_condition (s a : ℕ) : Prop :=
  s - 2 = 2 * (a - 2) ∧ s - 4 = 3 * (a - 4)

def future_ratio (s a x : ℕ) : Prop :=
  (s + x) * 2 = (a + x) * 3

theorem age_ratio_problem :
  ∃ s a x : ℕ, age_condition s a ∧ future_ratio s a x ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_problem_l1029_102978


namespace NUMINAMATH_GPT_james_parking_tickets_l1029_102927

-- Define the conditions
def ticket_cost_1 := 150
def ticket_cost_2 := 150
def ticket_cost_3 := 1 / 3 * ticket_cost_1
def total_cost := ticket_cost_1 + ticket_cost_2 + ticket_cost_3
def roommate_pays := total_cost / 2
def james_remaining_money := 325
def james_original_money := james_remaining_money + roommate_pays

-- Define the theorem we want to prove
theorem james_parking_tickets (h1: ticket_cost_1 = 150)
                              (h2: ticket_cost_1 = ticket_cost_2)
                              (h3: ticket_cost_3 = 1 / 3 * ticket_cost_1)
                              (h4: total_cost = ticket_cost_1 + ticket_cost_2 + ticket_cost_3)
                              (h5: roommate_pays = total_cost / 2)
                              (h6: james_remaining_money = 325)
                              (h7: james_original_money = james_remaining_money + roommate_pays):
                              total_cost = 350 :=
by
  sorry

end NUMINAMATH_GPT_james_parking_tickets_l1029_102927


namespace NUMINAMATH_GPT_cos_150_deg_eq_neg_half_l1029_102933

noncomputable def cos_of_angle (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem cos_150_deg_eq_neg_half :
  cos_of_angle 150 = -1/2 :=
by
  /-
    The conditions used directly in the problem include:
    - θ = 150 (Given angle)
  -/
  sorry

end NUMINAMATH_GPT_cos_150_deg_eq_neg_half_l1029_102933


namespace NUMINAMATH_GPT_right_angled_triangle_solution_l1029_102980

theorem right_angled_triangle_solution:
  ∃ (a b c : ℕ),
    (a^2 + b^2 = c^2) ∧
    (a + b + c = (a * b) / 2) ∧
    ((a, b, c) = (6, 8, 10) ∨ (a, b, c) = (5, 12, 13)) :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_solution_l1029_102980


namespace NUMINAMATH_GPT_largest_fraction_l1029_102906

theorem largest_fraction :
  let A := 2 / 5
  let B := 3 / 7
  let C := 4 / 9
  let D := 5 / 11
  let E := 6 / 13
  E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  let A := 2 / 5
  let B := 3 / 7
  let C := 4 / 9
  let D := 5 / 11
  let E := 6 / 13
  sorry

end NUMINAMATH_GPT_largest_fraction_l1029_102906


namespace NUMINAMATH_GPT_smallest_discount_l1029_102905

theorem smallest_discount (n : ℕ) (h1 : (1 - 0.12) * (1 - 0.18) = 0.88 * 0.82)
  (h2 : (1 - 0.08) * (1 - 0.08) * (1 - 0.08) = 0.92 * 0.92 * 0.92)
  (h3 : (1 - 0.20) * (1 - 0.10) = 0.80 * 0.90) :
  (29 > 27.84 ∧ 29 > 22.1312 ∧ 29 > 28) :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_discount_l1029_102905


namespace NUMINAMATH_GPT_melanies_mother_gave_l1029_102931

-- Define initial dimes, dad's contribution, and total dimes now
def initial_dimes : ℕ := 7
def dad_dimes : ℕ := 8
def total_dimes : ℕ := 19

-- Define the number of dimes the mother gave
def mother_dimes := total_dimes - (initial_dimes + dad_dimes)

-- Proof statement
theorem melanies_mother_gave : mother_dimes = 4 := by
  sorry

end NUMINAMATH_GPT_melanies_mother_gave_l1029_102931


namespace NUMINAMATH_GPT_function_domain_real_l1029_102974

theorem function_domain_real (k : ℝ) : 0 ≤ k ∧ k < 4 ↔ (∀ x : ℝ, k * x^2 + k * x + 1 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_function_domain_real_l1029_102974


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l1029_102994

noncomputable def xlnx (x : ℝ) : ℝ := x * Real.log x

theorem monotonic_decreasing_interval : 
  ∀ x, (0 < x) ∧ (x < 5) → (Real.log x + 1 < 0) ↔ (0 < x) ∧ (x < 1 / Real.exp 1) := 
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l1029_102994


namespace NUMINAMATH_GPT_probability_at_least_one_multiple_of_4_l1029_102992

/-- Definition for the total number of integers in the range -/
def total_numbers : ℕ := 60

/-- Definition for the number of multiples of 4 within the range -/
def multiples_of_4 : ℕ := 15

/-- Probability that a single number chosen is not a multiple of 4 -/
def prob_not_multiple_of_4 : ℚ := (total_numbers - multiples_of_4) / total_numbers

/-- Probability that none of the three chosen numbers is a multiple of 4 -/
def prob_none_multiple_of_4 : ℚ := prob_not_multiple_of_4 ^ 3

/-- Given condition that Linda choose three times -/
axiom linda_chooses_thrice (x y z : ℕ) : 
1 ≤ x ∧ x ≤ 60 ∧ 
1 ≤ y ∧ y ≤ 60 ∧ 
1 ≤ z ∧ z ≤ 60

/-- Theorem stating the desired probability -/
theorem probability_at_least_one_multiple_of_4 : 
1 - prob_none_multiple_of_4 = 37 / 64 := by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_multiple_of_4_l1029_102992


namespace NUMINAMATH_GPT_customer_paid_l1029_102993

def cost_price : ℝ := 7999.999999999999
def percentage_markup : ℝ := 0.10
def selling_price (cp : ℝ) (markup : ℝ) := cp + cp * markup

theorem customer_paid :
  selling_price cost_price percentage_markup = 8800 :=
by
  sorry

end NUMINAMATH_GPT_customer_paid_l1029_102993


namespace NUMINAMATH_GPT_imaginary_condition_l1029_102968

noncomputable def is_imaginary (z : ℂ) : Prop := z.im ≠ 0

theorem imaginary_condition (z1 z2 : ℂ) :
  ( ∃ (z1 : ℂ), is_imaginary z1 ∨ is_imaginary z2 ∨ (is_imaginary (z1 - z2))) ↔
  ∃ (z1 z2 : ℂ), is_imaginary z1 ∨ is_imaginary z2 ∧ ¬ (is_imaginary (z1 - z2)) :=
sorry

end NUMINAMATH_GPT_imaginary_condition_l1029_102968
