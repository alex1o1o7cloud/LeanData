import Mathlib

namespace NUMINAMATH_CALUDE_congruence_problem_l3005_300543

theorem congruence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n < 23 ∧ -135 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3005_300543


namespace NUMINAMATH_CALUDE_smallest_divisible_by_first_five_primes_l3005_300598

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem smallest_divisible_by_first_five_primes :
  (∀ p ∈ first_five_primes, 2310 % p = 0) ∧
  (∀ n < 2310, ∃ p ∈ first_five_primes, n % p ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_first_five_primes_l3005_300598


namespace NUMINAMATH_CALUDE_expression_lower_bound_l3005_300510

theorem expression_lower_bound :
  ∃ (L : ℤ), L = 3 ∧
  (∃ (S : Finset ℤ), S.card = 20 ∧
    ∀ n ∈ S, L < 4 * n + 7 ∧ 4 * n + 7 < 80) ∧
  ∀ (n : ℤ), 4 * n + 7 ≥ L :=
by sorry

end NUMINAMATH_CALUDE_expression_lower_bound_l3005_300510


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3005_300585

theorem sum_of_fractions : 
  (2 / 10 : ℚ) + (4 / 10 : ℚ) + (6 / 10 : ℚ) + (8 / 10 : ℚ) + (10 / 10 : ℚ) + 
  (12 / 10 : ℚ) + (14 / 10 : ℚ) + (16 / 10 : ℚ) + (18 / 10 : ℚ) + (32 / 10 : ℚ) = 
  (122 / 10 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3005_300585


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l3005_300521

theorem polynomial_factor_implies_coefficients 
  (p q : ℝ) 
  (h : ∃ (a b c : ℝ), px^4 + qx^3 + 40*x^2 - 24*x + 9 = (4*x^2 - 3*x + 2) * (a*x^2 + b*x + c)) :
  p = 12.5 ∧ q = -30.375 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l3005_300521


namespace NUMINAMATH_CALUDE_newspaper_collection_target_l3005_300523

structure Section where
  name : String
  first_week_collection : ℝ

def second_week_increase : ℝ := 0.10
def third_week_increase : ℝ := 0.30

def sections : List Section := [
  ⟨"A", 260⟩,
  ⟨"B", 290⟩,
  ⟨"C", 250⟩,
  ⟨"D", 270⟩,
  ⟨"E", 300⟩,
  ⟨"F", 310⟩,
  ⟨"G", 280⟩,
  ⟨"H", 265⟩
]

def first_week_total : ℝ := (sections.map (·.first_week_collection)).sum

def second_week_total : ℝ :=
  (sections.map (fun s => s.first_week_collection * (1 + second_week_increase))).sum

def third_week_total : ℝ :=
  (sections.map (fun s => s.first_week_collection * (1 + second_week_increase) * (1 + third_week_increase))).sum

def target : ℝ := first_week_total + second_week_total + third_week_total

theorem newspaper_collection_target :
  target = 7854.25 := by sorry

end NUMINAMATH_CALUDE_newspaper_collection_target_l3005_300523


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3005_300582

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 5) % 19 = 0 ∧
  (n + 5) % 73 = 0 ∧
  (n + 5) % 101 = 0 ∧
  (n + 5) % 89 = 0

theorem smallest_number_divisible_by_all :
  ∃! n : ℕ, is_divisible_by_all n ∧ ∀ m : ℕ, m < n → ¬is_divisible_by_all m :=
by
  use 1113805958
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3005_300582


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l3005_300536

-- Define the sample space
def Ω : Type := List Bool

-- Define the event "at most one hit"
def at_most_one_hit (ω : Ω) : Prop :=
  ω.length = 2 ∧ (ω.count true ≤ 1)

-- Define the event "two hits"
def two_hits (ω : Ω) : Prop :=
  ω.length = 2 ∧ (ω.count true = 2)

-- Theorem statement
theorem mutually_exclusive_events :
  ∀ ω : Ω, ¬(at_most_one_hit ω ∧ two_hits ω) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l3005_300536


namespace NUMINAMATH_CALUDE_trig_identity_l3005_300547

theorem trig_identity (α : ℝ) (h : Real.cos (75 * π / 180 + α) = 1/3) :
  Real.sin (60 * π / 180 + 2*α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3005_300547


namespace NUMINAMATH_CALUDE_parabola_f_value_l3005_300578

/-- A parabola with equation x = dy^2 + ey + f, vertex at (3, -1), and passing through (5, 1) -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_condition : 3 = d * (-1)^2 + e * (-1) + f
  point_condition : 5 = d * 1^2 + e * 1 + f

/-- The value of f for the given parabola is 7/2 -/
theorem parabola_f_value (p : Parabola) : p.f = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_f_value_l3005_300578


namespace NUMINAMATH_CALUDE_unit_digit_of_expression_l3005_300537

-- Define the expression
def expression : ℕ := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) - 1

-- Theorem statement
theorem unit_digit_of_expression : expression % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_expression_l3005_300537


namespace NUMINAMATH_CALUDE_solution_value_l3005_300505

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the parameters a and b
variable (a b : ℝ)

-- State the theorem
theorem solution_value (h : {x | a*x^2 + b*x + 2 > 0} = A ∩ B) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3005_300505


namespace NUMINAMATH_CALUDE_destination_distance_l3005_300526

/-- The distance to the destination in nautical miles -/
def distance : ℝ := sorry

/-- Theon's ship speed in nautical miles per hour -/
def theon_speed : ℝ := 15

/-- Yara's ship speed in nautical miles per hour -/
def yara_speed : ℝ := 30

/-- The time difference between Yara and Theon's arrivals in hours -/
def time_difference : ℝ := 3

theorem destination_distance : 
  distance = 90 ∧ 
  yara_speed = 2 * theon_speed ∧
  distance / yara_speed + time_difference = distance / theon_speed :=
by sorry

end NUMINAMATH_CALUDE_destination_distance_l3005_300526


namespace NUMINAMATH_CALUDE_divided_triangle_area_l3005_300511

/-- Represents a triangle with parallel lines dividing its sides -/
structure DividedTriangle where
  /-- The area of the original triangle -/
  area : ℝ
  /-- The number of equal segments the sides are divided into -/
  num_segments : ℕ
  /-- The area of the largest part after division -/
  largest_part_area : ℝ

/-- Theorem stating the relationship between the area of the largest part
    and the total area of the triangle -/
theorem divided_triangle_area (t : DividedTriangle)
    (h1 : t.num_segments = 10)
    (h2 : t.largest_part_area = 38) :
    t.area = 200 := by
  sorry

end NUMINAMATH_CALUDE_divided_triangle_area_l3005_300511


namespace NUMINAMATH_CALUDE_binomial_product_factorial_l3005_300538

theorem binomial_product_factorial (n : ℕ) : 
  (Nat.choose (n + 2) n) * n.factorial = ((n + 2) * (n + 1) * n.factorial) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_factorial_l3005_300538


namespace NUMINAMATH_CALUDE_not_divisible_by_power_of_two_l3005_300586

theorem not_divisible_by_power_of_two (n : ℕ) (h : n > 1) :
  ¬(2^n ∣ 3^n + 1) :=
by sorry

end NUMINAMATH_CALUDE_not_divisible_by_power_of_two_l3005_300586


namespace NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l3005_300571

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCardsCount : ℕ := 26

/-- The probability of a red card being followed by another red card -/
def probRedFollowedByRed : ℚ := 25 / 51

theorem expected_adjacent_red_pairs (deck_size : ℕ) (red_count : ℕ) (prob_red_red : ℚ) :
  deck_size = standardDeckSize →
  red_count = redCardsCount →
  prob_red_red = probRedFollowedByRed →
  (red_count : ℚ) * prob_red_red = 650 / 51 := by
  sorry

#check expected_adjacent_red_pairs

end NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l3005_300571


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3005_300524

-- Define the proposition
def P (x m : ℝ) : Prop := x / 2 + 1 / (2 * x) - 3 / 2 > m

-- Define the condition
def condition (m : ℝ) : Prop := ∀ x > 0, P x m

-- Define necessary condition
def necessary (m : ℝ) : Prop := condition m → m ≤ -1/2

-- Define sufficient condition
def sufficient (m : ℝ) : Prop := m ≤ -1/2 → condition m

-- Theorem statement
theorem necessary_not_sufficient :
  (∀ m : ℝ, necessary m) ∧ (∃ m : ℝ, ¬ sufficient m) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3005_300524


namespace NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l3005_300515

theorem three_numbers_in_unit_interval (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x < 1) (hy : 0 ≤ y ∧ y < 1) (hz : 0 ≤ z ∧ z < 1) :
  ∃ a b, (a = x ∨ a = y ∨ a = z) ∧ (b = x ∨ b = y ∨ b = z) ∧ a ≠ b ∧ |b - a| < (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l3005_300515


namespace NUMINAMATH_CALUDE_car_speeds_l3005_300583

theorem car_speeds (distance : ℝ) (time_difference : ℝ) (arrival_difference : ℝ) 
  (speed_ratio_small : ℝ) (speed_ratio_large : ℝ) 
  (h1 : distance = 135)
  (h2 : time_difference = 4)
  (h3 : arrival_difference = 1/2)
  (h4 : speed_ratio_small = 5)
  (h5 : speed_ratio_large = 2) :
  ∃ (speed_small : ℝ) (speed_large : ℝ),
    speed_small = 45 ∧ 
    speed_large = 18 ∧
    speed_small / speed_large = speed_ratio_small / speed_ratio_large ∧
    distance / speed_small = distance / speed_large - time_difference - arrival_difference :=
by
  sorry

end NUMINAMATH_CALUDE_car_speeds_l3005_300583


namespace NUMINAMATH_CALUDE_student_weight_is_75_l3005_300518

/-- The student's weight in kilograms -/
def student_weight : ℝ := sorry

/-- The sister's weight in kilograms -/
def sister_weight : ℝ := sorry

/-- The total weight of the student and his sister is 110 kilograms -/
axiom total_weight : student_weight + sister_weight = 110

/-- If the student loses 5 kilograms, he will weigh twice as much as his sister -/
axiom weight_relation : student_weight - 5 = 2 * sister_weight

/-- The student's present weight is 75 kilograms -/
theorem student_weight_is_75 : student_weight = 75 := by sorry

end NUMINAMATH_CALUDE_student_weight_is_75_l3005_300518


namespace NUMINAMATH_CALUDE_equal_interest_l3005_300560

def interest_rate_1_1 : ℝ := 0.07
def interest_rate_1_2 : ℝ := 0.10
def interest_rate_2_1 : ℝ := 0.05
def interest_rate_2_2 : ℝ := 0.12

def principal_1 : ℝ := 600
def principal_2 : ℝ := 800

def time_1_1 : ℕ := 3
def time_2_1 : ℕ := 2
def time_2_2 : ℕ := 3

def total_interest_2 : ℝ := principal_2 * interest_rate_2_1 * time_2_1 + principal_2 * interest_rate_2_2 * time_2_2

theorem equal_interest (n : ℕ) : 
  principal_1 * interest_rate_1_1 * time_1_1 + principal_1 * interest_rate_1_2 * n = total_interest_2 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_interest_l3005_300560


namespace NUMINAMATH_CALUDE_carpenter_theorem_l3005_300509

def carpenter_problem (total_woodblocks : ℕ) (current_logs : ℕ) (woodblocks_per_log : ℕ) : ℕ :=
  let current_woodblocks := current_logs * woodblocks_per_log
  let remaining_woodblocks := total_woodblocks - current_woodblocks
  remaining_woodblocks / woodblocks_per_log

theorem carpenter_theorem :
  carpenter_problem 80 8 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_theorem_l3005_300509


namespace NUMINAMATH_CALUDE_complement_of_N_in_U_l3005_300596

-- Define the universal set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set N
def N : Set ℝ := {x | 0 ≤ x ∧ x < 2}

-- Theorem statement
theorem complement_of_N_in_U :
  (U \ N) = {x | (-3 ≤ x ∧ x < 0) ∨ (2 ≤ x ∧ x ≤ 3)} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_N_in_U_l3005_300596


namespace NUMINAMATH_CALUDE_ellipse_circle_intersection_l3005_300506

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  25 * x^2 + 36 * y^2 = 900

noncomputable def point_M : ℝ × ℝ := (4.8, Real.sqrt (900 / 36 - 25 * 4.8^2 / 36))

noncomputable def tangent_line (x y : ℝ) : Prop :=
  25 * 4.8 * x + 36 * point_M.2 * y = 900

noncomputable def point_N : ℝ × ℝ := (0, 900 / (36 * point_M.2))

noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - (263 / 75))^2 = (362 / 75)^2

theorem ellipse_circle_intersection :
  ∀ x y : ℝ,
  ellipse_equation x y ∧ circle_equation x y ∧ y = 0 →
  x = Real.sqrt 11 ∨ x = -Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_intersection_l3005_300506


namespace NUMINAMATH_CALUDE_school_cleanup_participants_l3005_300575

/-- The expected number of participants after n years, given an initial number and annual increase rate -/
def expected_participants (initial : ℕ) (rate : ℚ) (years : ℕ) : ℚ :=
  initial * (1 + rate) ^ years

theorem school_cleanup_participants : expected_participants 1000 (60/100) 3 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_school_cleanup_participants_l3005_300575


namespace NUMINAMATH_CALUDE_binomial_coefficient_7_4_l3005_300579

theorem binomial_coefficient_7_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_7_4_l3005_300579


namespace NUMINAMATH_CALUDE_bens_debtor_payment_l3005_300527

/-- Calculates the amount paid by Ben's debtor given his financial transactions -/
theorem bens_debtor_payment (initial_amount cheque_amount maintenance_cost final_amount : ℕ) : 
  initial_amount = 2000 ∧ 
  cheque_amount = 600 ∧ 
  maintenance_cost = 1200 ∧ 
  final_amount = 1000 → 
  final_amount = initial_amount - cheque_amount - maintenance_cost + 800 := by
  sorry

#check bens_debtor_payment

end NUMINAMATH_CALUDE_bens_debtor_payment_l3005_300527


namespace NUMINAMATH_CALUDE_hyperbola_unique_solution_l3005_300542

/-- The hyperbola equation -/
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (2 * m^2) - y^2 / (3 * m) = 1

/-- The focal length of the hyperbola -/
def focal_length : ℝ := 6

/-- Theorem stating that 3/2 is the only positive real solution for m -/
theorem hyperbola_unique_solution :
  ∃! m : ℝ, m > 0 ∧ 
  (∀ x y : ℝ, hyperbola_equation x y m) ∧
  (∃ c : ℝ, c^2 = 2 * m^2 + 3 * m ∧ c = focal_length / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_unique_solution_l3005_300542


namespace NUMINAMATH_CALUDE_car_original_price_l3005_300591

/-- Given a car sold at a 15% loss and then resold with a 20% gain for Rs. 54000,
    prove that the original cost price of the car was Rs. 52,941.18 (rounded to two decimal places). -/
theorem car_original_price (loss_percent : ℝ) (gain_percent : ℝ) (final_price : ℝ) :
  loss_percent = 15 →
  gain_percent = 20 →
  final_price = 54000 →
  ∃ (original_price : ℝ),
    (1 - loss_percent / 100) * original_price * (1 + gain_percent / 100) = final_price ∧
    (round (original_price * 100) / 100 : ℝ) = 52941.18 := by
  sorry

end NUMINAMATH_CALUDE_car_original_price_l3005_300591


namespace NUMINAMATH_CALUDE_function_root_implies_m_range_l3005_300562

theorem function_root_implies_m_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ 2 * m * x + 4 = 0) → 
  (m ≤ -2 ∨ m ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_function_root_implies_m_range_l3005_300562


namespace NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l3005_300541

theorem expected_adjacent_red_pairs (total_cards : ℕ) (red_cards : ℕ) 
  (h1 : total_cards = 40) (h2 : red_cards = 20) :
  let prob_red_after_red := (red_cards - 1) / (total_cards - 1)
  let expected_pairs := red_cards * prob_red_after_red
  expected_pairs = 380 / 39 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l3005_300541


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3005_300563

/-- Given an election with two candidates where:
  - The winner received 1054 votes
  - The winner won by 408 votes
Prove that the percentage of votes the winner received is
(1054 / (1054 + (1054 - 408))) * 100 -/
theorem election_winner_percentage (winner_votes : ℕ) (winning_margin : ℕ) :
  winner_votes = 1054 →
  winning_margin = 408 →
  (winner_votes : ℝ) / (winner_votes + (winner_votes - winning_margin)) * 100 =
    1054 / 1700 * 100 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3005_300563


namespace NUMINAMATH_CALUDE_new_pyramid_volume_l3005_300593

/-- Represents the volume change of a pyramid -/
def pyramid_volume_change (initial_volume : ℝ) (length_scale : ℝ) (width_scale : ℝ) (height_scale : ℝ) : ℝ :=
  initial_volume * length_scale * width_scale * height_scale

/-- Theorem: New volume of the pyramid after scaling -/
theorem new_pyramid_volume :
  let initial_volume : ℝ := 100
  let length_scale : ℝ := 3
  let width_scale : ℝ := 2
  let height_scale : ℝ := 1.2
  pyramid_volume_change initial_volume length_scale width_scale height_scale = 720 := by
  sorry


end NUMINAMATH_CALUDE_new_pyramid_volume_l3005_300593


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_x_in_open_interval_l3005_300539

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_when_x_in_open_interval :
  {a : ℝ | ∀ x ∈ Set.Ioo 0 1, f a x > x} = Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_x_in_open_interval_l3005_300539


namespace NUMINAMATH_CALUDE_min_value_sum_l3005_300513

theorem min_value_sum (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y)
  (h : a / x + b / y = 2) : 
  x + y ≥ (a + b) / 2 + Real.sqrt (a * b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l3005_300513


namespace NUMINAMATH_CALUDE_hexagon_angle_sum_l3005_300545

/-- A hexagon is a polygon with six vertices and six edges. -/
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- The sum of interior angles of a hexagon in degrees. -/
def sum_of_angles (h : Hexagon) : ℝ := 
  sorry

/-- Theorem: In a hexagon where the sum of all interior angles is 90n degrees, n must equal 4. -/
theorem hexagon_angle_sum (h : Hexagon) (n : ℝ) 
  (h_sum : sum_of_angles h = 90 * n) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_sum_l3005_300545


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l3005_300570

/-- The side length of the square base of a right pyramid, given the area of a lateral face and the slant height. -/
theorem pyramid_base_side_length (lateral_face_area : ℝ) (slant_height : ℝ) :
  lateral_face_area = 120 →
  slant_height = 40 →
  (lateral_face_area / slant_height) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l3005_300570


namespace NUMINAMATH_CALUDE_video_votes_l3005_300514

theorem video_votes (score : ℕ) (like_percent : ℚ) (dislike_percent : ℚ) (neutral_percent : ℚ) :
  score = 180 →
  like_percent = 60 / 100 →
  dislike_percent = 20 / 100 →
  neutral_percent = 20 / 100 →
  like_percent + dislike_percent + neutral_percent = 1 →
  ∃ (total_votes : ℕ), 
    (↑score : ℚ) = (like_percent - dislike_percent) * ↑total_votes ∧
    total_votes = 450 := by
  sorry

end NUMINAMATH_CALUDE_video_votes_l3005_300514


namespace NUMINAMATH_CALUDE_ten_people_no_adjacent_standing_probability_l3005_300544

/-- Represents the number of valid arrangements for n people where no two adjacent people stand --/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing in a circular arrangement of n people --/
def noAdjacentStandingProbability (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n : ℚ)

theorem ten_people_no_adjacent_standing_probability :
  noAdjacentStandingProbability 10 = 123 / 1024 := by
  sorry


end NUMINAMATH_CALUDE_ten_people_no_adjacent_standing_probability_l3005_300544


namespace NUMINAMATH_CALUDE_minimum_value_sqrt_sum_l3005_300587

theorem minimum_value_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ Real.sqrt 2) ∧
  (∀ ε > 0, ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ Real.sqrt x + Real.sqrt y > Real.sqrt 2 - ε) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_sqrt_sum_l3005_300587


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l3005_300561

theorem smallest_five_digit_multiple_of_18 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 18 ∣ n → 10008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l3005_300561


namespace NUMINAMATH_CALUDE_meadow_grazing_l3005_300574

/-- Represents the amount of grass one cow eats per day -/
def daily_cow_consumption : ℝ := sorry

/-- Represents the amount of grass that grows on the meadow per day -/
def daily_grass_growth : ℝ := sorry

/-- Represents the initial amount of grass in the meadow -/
def initial_grass : ℝ := sorry

/-- Condition: 9 cows will graze the meadow empty in 4 days -/
axiom condition1 : initial_grass + 4 * daily_grass_growth = 9 * 4 * daily_cow_consumption

/-- Condition: 8 cows will graze the meadow empty in 6 days -/
axiom condition2 : initial_grass + 6 * daily_grass_growth = 8 * 6 * daily_cow_consumption

/-- The number of cows that can graze continuously in the meadow -/
def continuous_grazing_cows : ℕ := 6

theorem meadow_grazing :
  daily_grass_growth = continuous_grazing_cows * daily_cow_consumption :=
sorry

end NUMINAMATH_CALUDE_meadow_grazing_l3005_300574


namespace NUMINAMATH_CALUDE_cube_expansion_sum_l3005_300568

/-- Given that for any real number x, x^3 = a₀ + a₁(x-2) + a₂(x-2)² + a₃(x-2)³, 
    prove that a₁ + a₂ + a₃ = 19 -/
theorem cube_expansion_sum (a₀ a₁ a₂ a₃ : ℝ) 
    (h : ∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) : 
  a₁ + a₂ + a₃ = 19 := by
  sorry

end NUMINAMATH_CALUDE_cube_expansion_sum_l3005_300568


namespace NUMINAMATH_CALUDE_actual_weekly_earnings_increase_l3005_300555

/-- Calculates the actual increase in weekly earnings given a raise, work hours, and housing benefit reduction. -/
theorem actual_weekly_earnings_increase
  (hourly_raise : ℝ)
  (weekly_hours : ℝ)
  (monthly_benefit_reduction : ℝ)
  (h1 : hourly_raise = 0.50)
  (h2 : weekly_hours = 40)
  (h3 : monthly_benefit_reduction = 60)
  : ∃ (actual_increase : ℝ), abs (actual_increase - 6.14) < 0.01 := by
  sorry

#check actual_weekly_earnings_increase

end NUMINAMATH_CALUDE_actual_weekly_earnings_increase_l3005_300555


namespace NUMINAMATH_CALUDE_trig_identity_l3005_300592

theorem trig_identity (α β : Real) 
  (h1 : Real.cos (α + β) = 1/3)
  (h2 : Real.sin α * Real.sin β = 1/4) :
  (Real.cos α * Real.cos β = 7/12) ∧
  (Real.cos (2*α - 2*β) = 7/18) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3005_300592


namespace NUMINAMATH_CALUDE_dog_meal_amount_proof_l3005_300540

/-- The amount of food a dog eats at each meal, in pounds -/
def dog_meal_amount : ℝ := 4

/-- The number of puppies -/
def num_puppies : ℕ := 4

/-- The number of dogs -/
def num_dogs : ℕ := 3

/-- The number of times a dog eats per day -/
def dog_meals_per_day : ℕ := 3

/-- The total amount of food eaten by dogs and puppies in a day, in pounds -/
def total_food_per_day : ℝ := 108

theorem dog_meal_amount_proof :
  dog_meal_amount * num_dogs * dog_meals_per_day + 
  (dog_meal_amount / 2) * num_puppies * (3 * dog_meals_per_day) = total_food_per_day :=
by sorry

end NUMINAMATH_CALUDE_dog_meal_amount_proof_l3005_300540


namespace NUMINAMATH_CALUDE_job_farm_reserved_land_l3005_300565

/-- Represents the land allocation of a farm in hectares -/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  cattle : ℕ
  crops : ℕ

/-- Calculates the land reserved for future expansion -/
def reserved_land (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.cattle + farm.crops)

/-- Theorem stating that the reserved land for Job's farm is 15 hectares -/
theorem job_farm_reserved_land :
  let job_farm : FarmLand := {
    total := 150,
    house_and_machinery := 25,
    cattle := 40,
    crops := 70
  }
  reserved_land job_farm = 15 := by
  sorry


end NUMINAMATH_CALUDE_job_farm_reserved_land_l3005_300565


namespace NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l3005_300566

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * x^2 + m * x) / Real.exp x

theorem extreme_value_and_monotonicity (m : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, 0 < |x| ∧ |x| < ε → f 0 0 ≤ f 0 x) ∧
  (∀ x, x ≥ 3 → ∀ y, y > x → f m y ≤ f m x) ↔ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l3005_300566


namespace NUMINAMATH_CALUDE_simplify_expression_l3005_300519

theorem simplify_expression (x : ℝ) : 3*x + 5 - 2*x - 6 + 4*x + 7 - 5*x - 9 = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3005_300519


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_is_13_l3005_300584

theorem sum_of_A_and_B_is_13 (A B : ℕ) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  70 + A - (10 * B + 5) = 34 → 
  A + B = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_is_13_l3005_300584


namespace NUMINAMATH_CALUDE_ball_difference_l3005_300525

def soccer_boxes : ℕ := 8
def basketball_boxes : ℕ := 5
def balls_per_box : ℕ := 12

theorem ball_difference : 
  soccer_boxes * balls_per_box - basketball_boxes * balls_per_box = 36 := by
  sorry

end NUMINAMATH_CALUDE_ball_difference_l3005_300525


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3005_300594

theorem arithmetic_calculation : (180 / 6) * 2 + 5 = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3005_300594


namespace NUMINAMATH_CALUDE_raffle_ticket_average_l3005_300597

/-- Represents a charitable association with male and female members selling raffle tickets -/
structure CharitableAssociation where
  male_members : ℕ
  female_members : ℕ
  male_avg_tickets : ℕ
  female_avg_tickets : ℕ

/-- The overall average number of raffle tickets sold per member -/
def overall_average (ca : CharitableAssociation) : ℚ :=
  (ca.male_members * ca.male_avg_tickets + ca.female_members * ca.female_avg_tickets : ℚ) /
  (ca.male_members + ca.female_members : ℚ)

/-- Theorem stating the overall average of raffle tickets sold per member -/
theorem raffle_ticket_average (ca : CharitableAssociation) 
  (h1 : ca.female_members = 2 * ca.male_members)
  (h2 : ca.female_avg_tickets = 70)
  (h3 : ca.male_avg_tickets = 58) :
  overall_average ca = 66 := by
  sorry

end NUMINAMATH_CALUDE_raffle_ticket_average_l3005_300597


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3005_300576

/-- An ellipse with equation x² + my² = 1, where m > 0 -/
structure Ellipse (m : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 + m*y^2 = 1)
  (m_pos : m > 0)

/-- The focus of the ellipse is on the y-axis -/
def focus_on_y_axis (m : ℝ) : Prop := 0 < m ∧ m < 1

/-- The length of the major axis is twice that of the minor axis -/
def major_twice_minor (m : ℝ) : Prop := Real.sqrt (1/m) = 2

/-- Theorem: For an ellipse with equation x² + my² = 1 (m > 0), 
    if its focus is on the y-axis and the length of its major axis 
    is twice that of its minor axis, then m = 1/4 -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
  (h1 : focus_on_y_axis m) (h2 : major_twice_minor m) : m = 1/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3005_300576


namespace NUMINAMATH_CALUDE_a_squared_gt_one_sufficient_not_necessary_l3005_300546

-- Define the equation
def is_ellipse (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / a^2 + y^2 = 1 ∧ (x ≠ 0 ∨ y ≠ 0)

-- Theorem statement
theorem a_squared_gt_one_sufficient_not_necessary :
  (∀ a : ℝ, a^2 > 1 → is_ellipse a) ∧
  (∃ a : ℝ, is_ellipse a ∧ a^2 ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_a_squared_gt_one_sufficient_not_necessary_l3005_300546


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3005_300501

theorem trig_identity_proof : 
  (2 * Real.sin (80 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3005_300501


namespace NUMINAMATH_CALUDE_triangle_containing_all_points_l3005_300572

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t1 t2 t3 : Point) : Prop := sorry

theorem triangle_containing_all_points 
  (n : ℕ) 
  (points : Fin n → Point) 
  (h : ∀ (i j k : Fin n), triangleArea (points i) (points j) (points k) ≤ 1) :
  ∃ (t1 t2 t3 : Point), 
    (triangleArea t1 t2 t3 ≤ 4) ∧ 
    (∀ (i : Fin n), isInside (points i) t1 t2 t3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_containing_all_points_l3005_300572


namespace NUMINAMATH_CALUDE_min_container_cost_l3005_300567

/-- Represents the dimensions and cost of a rectangular container -/
structure Container where
  length : ℝ
  width : ℝ
  height : ℝ
  baseUnitCost : ℝ
  sideUnitCost : ℝ

/-- Calculates the total cost of the container -/
def totalCost (c : Container) : ℝ :=
  c.baseUnitCost * c.length * c.width + 
  c.sideUnitCost * 2 * (c.length + c.width) * c.height

/-- Theorem stating the minimum cost of the container -/
theorem min_container_cost :
  ∃ (c : Container),
    c.height = 1 ∧
    c.length * c.width * c.height = 4 ∧
    c.baseUnitCost = 20 ∧
    c.sideUnitCost = 10 ∧
    (∀ (d : Container),
      d.height = 1 →
      d.length * d.width * d.height = 4 →
      d.baseUnitCost = 20 →
      d.sideUnitCost = 10 →
      totalCost c ≤ totalCost d) ∧
    totalCost c = 160 := by
  sorry

end NUMINAMATH_CALUDE_min_container_cost_l3005_300567


namespace NUMINAMATH_CALUDE_matrix_equation_result_l3005_300589

/-- Given two 2x2 matrices A and B, where A is fixed and B has variable entries,
    if AB = BA and 4y ≠ z, then (x - w) / (z - 4y) = 3/8 -/
theorem matrix_equation_result (x y z w : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  4 * y ≠ z →
  A * B = B * A →
  (x - w) / (z - 4 * y) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_result_l3005_300589


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l3005_300550

/-- Given a point P(-3, 2), its symmetric point P' with respect to the origin O has coordinates (3, -2). -/
theorem symmetric_point_wrt_origin :
  let P : ℝ × ℝ := (-3, 2)
  let P' : ℝ × ℝ := (3, -2)
  let symmetric_wrt_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  symmetric_wrt_origin P = P' := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l3005_300550


namespace NUMINAMATH_CALUDE_congruence_solution_l3005_300508

theorem congruence_solution (x : ℤ) 
  (h1 : (2 + x) % (2^4) = 3^2 % (2^4))
  (h2 : (3 + x) % (3^4) = 2^3 % (3^4))
  (h3 : (4 + x) % (2^3) = 3^3 % (2^3)) :
  x % 24 = 23 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l3005_300508


namespace NUMINAMATH_CALUDE_hyperbola_distance_theorem_l3005_300556

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define the distance function
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

-- The main theorem
theorem hyperbola_distance_theorem (x y : ℝ) (P : ℝ × ℝ) :
  is_on_hyperbola x y →
  P = (x, y) →
  distance P F₁ = 12 →
  distance P F₂ = 2 ∨ distance P F₂ = 22 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_distance_theorem_l3005_300556


namespace NUMINAMATH_CALUDE_S_eq_EvenPositive_l3005_300551

/-- The set of all positive integers that can be written in the form ([x, y] + [y, z]) / [x, z] -/
def S : Set ℕ+ :=
  {n | ∃ (x y z : ℕ+), n = (Nat.lcm x y + Nat.lcm y z) / Nat.lcm x z}

/-- The set of all even positive integers -/
def EvenPositive : Set ℕ+ :=
  {n | ∃ (k : ℕ+), n = 2 * k}

/-- Theorem stating that S is equal to the set of all even positive integers -/
theorem S_eq_EvenPositive : S = EvenPositive := by
  sorry

end NUMINAMATH_CALUDE_S_eq_EvenPositive_l3005_300551


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_seven_undefined_inverse_smallest_is_seven_l3005_300599

theorem smallest_undefined_inverse (b : ℕ) : b > 0 ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 77]) → 
  b ≥ 7 :=
by sorry

theorem seven_undefined_inverse : 
  ¬ (∃ x : ℕ, x * 7 ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * 7 ≡ 1 [MOD 77]) :=
by sorry

theorem smallest_is_seven : 
  ∃ (b : ℕ), b > 0 ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 77]) ∧
  ∀ (c : ℕ), c > 0 ∧ 
  ¬ (∃ x : ℕ, x * c ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * c ≡ 1 [MOD 77]) →
  c ≥ b :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_seven_undefined_inverse_smallest_is_seven_l3005_300599


namespace NUMINAMATH_CALUDE_area_of_trapezoid_DBCE_l3005_300588

/-- A structure representing a triangle in our problem -/
structure Triangle where
  area : ℝ

/-- A structure representing the trapezoid DBCE -/
structure Trapezoid where
  area : ℝ

/-- The isosceles triangle ABC -/
def ABC : Triangle := { area := 36 }

/-- One of the smallest triangles -/
def smallTriangle : Triangle := { area := 1 }

/-- The number of smallest triangles -/
def numSmallTriangles : ℕ := 5

/-- Triangle ADE, composed of 3 smallest triangles -/
def ADE : Triangle := { area := 3 }

/-- The trapezoid DBCE -/
def DBCE : Trapezoid := { area := ABC.area - ADE.area }

/-- The theorem to be proved -/
theorem area_of_trapezoid_DBCE : DBCE.area = 33 := by
  sorry

end NUMINAMATH_CALUDE_area_of_trapezoid_DBCE_l3005_300588


namespace NUMINAMATH_CALUDE_gcd_abcd_plus_dcba_l3005_300569

def abcd_plus_dcba (a : ℕ) : ℕ := 2222 * a + 12667

theorem gcd_abcd_plus_dcba : 
  Nat.gcd (abcd_plus_dcba 0) (Nat.gcd (abcd_plus_dcba 1) (Nat.gcd (abcd_plus_dcba 2) (abcd_plus_dcba 3))) = 2222 := by
  sorry

end NUMINAMATH_CALUDE_gcd_abcd_plus_dcba_l3005_300569


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3005_300595

theorem repeating_decimal_sum (c d : ℕ) (h : (4 : ℚ) / 13 = 0.1 * c + 0.01 * d + 0.001 * (c + d / 10)) : c + d = 10 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3005_300595


namespace NUMINAMATH_CALUDE_bekah_reading_days_l3005_300557

/-- Given the total pages to read, pages already read, and pages to read per day,
    calculate the number of days left to finish reading. -/
def days_left_to_read (total_pages pages_read pages_per_day : ℕ) : ℕ :=
  (total_pages - pages_read) / pages_per_day

/-- Theorem: Given 408 total pages, 113 pages read, and 59 pages per day,
    the number of days left to finish reading is 5. -/
theorem bekah_reading_days : days_left_to_read 408 113 59 = 5 := by
  sorry

#eval days_left_to_read 408 113 59

end NUMINAMATH_CALUDE_bekah_reading_days_l3005_300557


namespace NUMINAMATH_CALUDE_box_min_height_l3005_300534

/-- Represents a rectangular box with square bases -/
structure Box where
  base_side : ℝ
  height : ℝ

/-- Calculates the surface area of a box -/
def surface_area (b : Box) : ℝ :=
  2 * b.base_side^2 + 4 * b.base_side * b.height

/-- The minimum height of a box satisfying the given conditions -/
def min_height : ℝ := 6

theorem box_min_height :
  ∀ (b : Box),
    b.height = b.base_side + 4 →
    surface_area b ≥ 120 →
    b.height ≥ min_height :=
by
  sorry

end NUMINAMATH_CALUDE_box_min_height_l3005_300534


namespace NUMINAMATH_CALUDE_cars_initial_distance_l3005_300516

/-- The distance between two cars traveling towards each other -/
def initial_distance (speed_a : ℝ) (speed_b : ℝ) (time : ℝ) (final_distance : ℝ) : ℝ :=
  speed_a * time + speed_b * time + final_distance

/-- Theorem stating the initial distance between the cars is 200 km -/
theorem cars_initial_distance :
  let speed_a : ℝ := 50
  let speed_b : ℝ := 4/5 * speed_a
  let time : ℝ := 2
  let final_distance : ℝ := 20
  initial_distance speed_a speed_b time final_distance = 200 := by
  sorry

end NUMINAMATH_CALUDE_cars_initial_distance_l3005_300516


namespace NUMINAMATH_CALUDE_bulk_bag_contains_40_oz_l3005_300522

/-- Calculates the number of ounces in a bulk bag of mixed nuts -/
def bulkBagOunces (originalCost : ℚ) (couponValue : ℚ) (costPerServing : ℚ) : ℚ :=
  (originalCost - couponValue) / costPerServing

/-- Theorem stating that the bulk bag contains 40 ounces of mixed nuts -/
theorem bulk_bag_contains_40_oz :
  bulkBagOunces 25 5 (1/2) = 40 := by
  sorry

end NUMINAMATH_CALUDE_bulk_bag_contains_40_oz_l3005_300522


namespace NUMINAMATH_CALUDE_proposition_implication_l3005_300554

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 :=
sorry

end NUMINAMATH_CALUDE_proposition_implication_l3005_300554


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l3005_300504

/-- If 9x^2 - 24x + c is a perfect square of a binomial, then c = 16 -/
theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l3005_300504


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l3005_300553

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) → 
  k = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l3005_300553


namespace NUMINAMATH_CALUDE_annulus_area_l3005_300500

/-- An annulus is formed by two concentric circles with radii R and r, where R > r.
    x is the length of a tangent line from a point on the outer circle to the inner circle. -/
theorem annulus_area (R r x : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = x^2) : 
  π * R^2 - π * r^2 = π * x^2 := by sorry

end NUMINAMATH_CALUDE_annulus_area_l3005_300500


namespace NUMINAMATH_CALUDE_m_range_l3005_300590

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 ≠ 0 ∨ ∃ y : ℝ, y ≠ x ∧ y^2 + m*y + 1 = 0 → False) →
  (∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0) →
  1 < m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3005_300590


namespace NUMINAMATH_CALUDE_susan_age_indeterminate_l3005_300549

/-- Represents a person's age at different points in time -/
structure PersonAge where
  current : ℕ
  eightYearsAgo : ℕ
  inFifteenYears : ℕ

/-- The given conditions of the problem -/
axiom james : PersonAge
axiom janet : PersonAge
axiom susan : ℕ → Prop

axiom james_age_condition : james.inFifteenYears = 37
axiom james_janet_age_relation : james.eightYearsAgo = 2 * janet.eightYearsAgo
axiom susan_birth_condition : ∃ (age : ℕ), susan age

/-- The statement that Susan's age in 5 years cannot be determined -/
theorem susan_age_indeterminate : ¬∃ (age : ℕ), ∀ (current_age : ℕ), susan current_age → current_age + 5 = age := by
  sorry

end NUMINAMATH_CALUDE_susan_age_indeterminate_l3005_300549


namespace NUMINAMATH_CALUDE_hexagon_c_x_coordinate_l3005_300535

/-- A hexagon with vertices A, B, C, D, E, F in 2D space -/
structure Hexagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- Check if a hexagon has a horizontal line of symmetry -/
def hasHorizontalSymmetry (h : Hexagon) : Prop := sorry

/-- Given a hexagon with specified properties, prove that the x-coordinate of vertex C is 22 -/
theorem hexagon_c_x_coordinate (h : Hexagon) 
  (sym : hasHorizontalSymmetry h)
  (area : hexagonArea h = 30)
  (vA : h.A = (0, 0))
  (vB : h.B = (0, 2))
  (vD : h.D = (5, 2))
  (vE : h.E = (5, 0))
  (vF : h.F = (2, 0)) :
  h.C.1 = 22 := by sorry

end NUMINAMATH_CALUDE_hexagon_c_x_coordinate_l3005_300535


namespace NUMINAMATH_CALUDE_polynomial_equality_l3005_300564

theorem polynomial_equality (a b c d e : ℝ) : 
  (∀ x : ℝ, (3*x + 1)^4 = a*x^4 + b*x^3 + c*x^2 + d*x + e) → 
  a - b + c - d + e = 16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3005_300564


namespace NUMINAMATH_CALUDE_cube_inequality_iff_l3005_300512

theorem cube_inequality_iff (a b : ℝ) : a < b ↔ a^3 < b^3 := by sorry

end NUMINAMATH_CALUDE_cube_inequality_iff_l3005_300512


namespace NUMINAMATH_CALUDE_product_eleven_reciprocal_squares_sum_l3005_300520

theorem product_eleven_reciprocal_squares_sum (a b : ℕ+) :
  a * b = 11 → (1 : ℚ) / a^2 + (1 : ℚ) / b^2 = 122 / 121 := by
  sorry

end NUMINAMATH_CALUDE_product_eleven_reciprocal_squares_sum_l3005_300520


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l3005_300528

/-- Given two circles in a 2D plane:
    Circle 1 with radius 3 and center (0, 0)
    Circle 2 with radius 5 and center (12, 0)
    The x-coordinate of the point where a line tangent to both circles
    intersects the x-axis (to the right of the origin) is 9/2. -/
theorem tangent_line_intersection (x : ℝ) : 
  (∃ (y : ℝ), (x^2 + y^2 = 3^2 ∧ ((x - 12)^2 + y^2 = 5^2))) → x = 9/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l3005_300528


namespace NUMINAMATH_CALUDE_total_spent_football_games_l3005_300517

/-- Calculates the total amount spent on football games over three months -/
def total_spent (games_this_month : ℕ) (price_this_month : ℕ)
                (games_last_month : ℕ) (price_last_month : ℕ)
                (games_next_month : ℕ) (price_next_month : ℕ) : ℕ :=
  games_this_month * price_this_month +
  games_last_month * price_last_month +
  games_next_month * price_next_month

/-- Theorem stating the total amount spent on football games -/
theorem total_spent_football_games :
  total_spent 11 25 17 30 16 35 = 1345 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_football_games_l3005_300517


namespace NUMINAMATH_CALUDE_tin_weight_in_water_l3005_300580

theorem tin_weight_in_water (total_weight : ℝ) (weight_lost : ℝ) (tin_silver_ratio : ℝ) 
  (tin_loss : ℝ) (silver_weight : ℝ) (silver_loss : ℝ) :
  total_weight = 60 →
  weight_lost = 6 →
  tin_silver_ratio = 2/3 →
  tin_loss = 1.375 →
  silver_weight = 5 →
  silver_loss = 0.375 →
  ∃ (tin_weight : ℝ), tin_weight * (weight_lost / total_weight) = tin_loss ∧ 
    tin_weight = 13.75 := by
  sorry

end NUMINAMATH_CALUDE_tin_weight_in_water_l3005_300580


namespace NUMINAMATH_CALUDE_circle_radius_l3005_300529

/-- The radius of the circle with equation x^2 - 10x + y^2 - 4y + 24 = 0 is √5 -/
theorem circle_radius (x y : ℝ) : 
  (x^2 - 10*x + y^2 - 4*y + 24 = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3005_300529


namespace NUMINAMATH_CALUDE_exists_solution_l3005_300577

theorem exists_solution : ∃ (a b : ℤ), a ≠ b ∧ 
  (a : ℚ) / 2015 + (b : ℚ) / 2016 = (2015 + 2016 : ℚ) / (2015 * 2016) := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_l3005_300577


namespace NUMINAMATH_CALUDE_pentagon_square_side_ratio_l3005_300507

/-- Given a regular pentagon and a square with the same perimeter of 20 inches,
    the ratio of the side length of the pentagon to the side length of the square is 4/5. -/
theorem pentagon_square_side_ratio :
  ∀ (p s : ℝ), 
    p > 0 → s > 0 →
    5 * p = 20 →  -- Perimeter of pentagon
    4 * s = 20 →  -- Perimeter of square
    p / s = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_square_side_ratio_l3005_300507


namespace NUMINAMATH_CALUDE_jeans_price_increase_l3005_300530

/-- Given a manufacturing cost C, calculate the percentage increase from the retailer's price to the customer's price -/
theorem jeans_price_increase (C : ℝ) : 
  let retailer_price := C * 1.4
  let customer_price := C * 1.82
  (customer_price - retailer_price) / retailer_price * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_jeans_price_increase_l3005_300530


namespace NUMINAMATH_CALUDE_manuscript_revision_problem_l3005_300559

/-- The number of pages revised twice in a manuscript -/
def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (total_cost : ℕ) : ℕ :=
  (total_cost - (5 * total_pages + 3 * pages_revised_once)) / 6

/-- Theorem stating the number of pages revised twice -/
theorem manuscript_revision_problem (total_pages : ℕ) (pages_revised_once : ℕ) (total_cost : ℕ) 
  (h1 : total_pages = 200)
  (h2 : pages_revised_once = 80)
  (h3 : total_cost = 1360) :
  pages_revised_twice total_pages pages_revised_once total_cost = 20 := by
sorry

#eval pages_revised_twice 200 80 1360

end NUMINAMATH_CALUDE_manuscript_revision_problem_l3005_300559


namespace NUMINAMATH_CALUDE_fifth_seat_is_37_l3005_300552

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  selectedSeats : Finset ℕ

/-- The seat number of the fifth selected student in the systematic sampling. -/
def fifthSelectedSeat (sampling : SystematicSampling) : ℕ :=
  37

/-- Theorem stating that given the conditions, the fifth selected seat is 37. -/
theorem fifth_seat_is_37 (sampling : SystematicSampling) 
  (h1 : sampling.totalStudents = 55)
  (h2 : sampling.sampleSize = 5)
  (h3 : sampling.selectedSeats = {4, 15, 26, 48}) :
  fifthSelectedSeat sampling = 37 := by
  sorry

#check fifth_seat_is_37

end NUMINAMATH_CALUDE_fifth_seat_is_37_l3005_300552


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3005_300573

def f (x : ℝ) := x^3 - 12*x + 8

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 3, f x = min) ∧
    max = 24 ∧ min = -6 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3005_300573


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_4_area_is_8_l3005_300503

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (Real.sqrt 2 * t.b - t.c) / t.a = Real.cos t.C / Real.cos t.A

-- Theorem 1: Angle A is π/4
theorem angle_A_is_pi_over_4 (t : Triangle) (h : satisfiesCondition t) : t.A = π / 4 := by
  sorry

-- Theorem 2: Area of the triangle is 8 under specific conditions
theorem area_is_8 (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.a = 10) (h3 : t.b = 8 * Real.sqrt 2) 
  (h4 : t.C < t.A ∧ t.C < t.B) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 8 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_4_area_is_8_l3005_300503


namespace NUMINAMATH_CALUDE_hexagon_ratio_l3005_300533

/-- A hexagon with specific properties -/
structure Hexagon :=
  (total_area : ℝ)
  (bisector : ℝ → ℝ → Prop)
  (lower_part : ℝ → ℝ → Prop)
  (triangle_base : ℝ)

/-- The theorem statement -/
theorem hexagon_ratio (h : Hexagon) (x y : ℝ) : 
  h.total_area = 7 ∧ 
  h.bisector x y ∧ 
  h.lower_part 1 (5/2) ∧ 
  h.triangle_base = 4 →
  x / y = 1 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_ratio_l3005_300533


namespace NUMINAMATH_CALUDE_certain_number_problem_l3005_300558

theorem certain_number_problem : ∃ x : ℤ, (5 + (x + 3) = 19) ∧ (x = 11) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3005_300558


namespace NUMINAMATH_CALUDE_three_gorges_electricity_production_l3005_300531

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a| 
  h2 : |a| < 10

/-- The number to be represented (798.5 billion) -/
def number : ℝ := 798.5e9

/-- Theorem stating that 798.5 billion can be represented as 7.985 × 10^2 billion in scientific notation -/
theorem three_gorges_electricity_production :
  ∃ (sn : ScientificNotation), sn.a * (10 : ℝ)^sn.n = number ∧ sn.a = 7.985 ∧ sn.n = 2 :=
sorry

end NUMINAMATH_CALUDE_three_gorges_electricity_production_l3005_300531


namespace NUMINAMATH_CALUDE_geometric_series_remainder_remainder_of_series_l3005_300502

theorem geometric_series_remainder (n : ℕ) (a r : ℤ) (m : ℕ) (h : m > 0) :
  (a * (r^n - 1) / (r - 1)) % m = 
  ((a * (r^n % (m * (r - 1))) - a) / (r - 1)) % m :=
sorry

theorem remainder_of_series : 
  (((3^1002 - 1) / 2) % 500) = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_remainder_remainder_of_series_l3005_300502


namespace NUMINAMATH_CALUDE_expression_evaluation_l3005_300532

theorem expression_evaluation :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 42 * 21^1006 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3005_300532


namespace NUMINAMATH_CALUDE_count_numbers_with_ten_digit_square_and_cube_l3005_300581

-- Define a function to count the number of digits in a natural number
def countDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + countDigits (n / 10)

-- Define the condition for a number to satisfy the problem requirement
def satisfiesCondition (n : ℕ) : Prop :=
  countDigits (n^2) + countDigits (n^3) = 10

-- Theorem statement
theorem count_numbers_with_ten_digit_square_and_cube :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfiesCondition n) ∧ S.card = 53 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_ten_digit_square_and_cube_l3005_300581


namespace NUMINAMATH_CALUDE_job_completion_time_l3005_300548

/-- Given two people P and Q working on a job, this theorem proves the time
    it takes P to complete the job alone, given the time it takes Q alone
    and the time it takes them working together. -/
theorem job_completion_time
  (time_Q : ℝ)
  (time_PQ : ℝ)
  (h1 : time_Q = 6)
  (h2 : time_PQ = 2.4) :
  ∃ (time_P : ℝ), time_P = 4 ∧ 1 / time_P + 1 / time_Q = 1 / time_PQ :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l3005_300548
