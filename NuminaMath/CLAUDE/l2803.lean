import Mathlib

namespace NUMINAMATH_CALUDE_unique_good_days_count_l2803_280308

/-- Represents the change factor for an ingot on a given day type -/
structure IngotFactor where
  good : ℝ
  bad : ℝ

/-- Calculates the final value of an ingot after a week -/
def finalValue (factor : IngotFactor) (goodDays : ℕ) : ℝ :=
  factor.good ^ goodDays * factor.bad ^ (7 - goodDays)

/-- The problem statement -/
theorem unique_good_days_count :
  ∃! goodDays : ℕ,
    goodDays ≤ 7 ∧
    let goldFactor : IngotFactor := { good := 1.3, bad := 0.7 }
    let silverFactor : IngotFactor := { good := 1.2, bad := 0.8 }
    (finalValue goldFactor goodDays < 1 ∧ finalValue silverFactor goodDays > 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_good_days_count_l2803_280308


namespace NUMINAMATH_CALUDE_rose_orchid_difference_l2803_280340

/-- Given the initial and final counts of roses and orchids in a vase, 
    prove that there are 10 more roses than orchids in the final state. -/
theorem rose_orchid_difference :
  let initial_roses : ℕ := 5
  let initial_orchids : ℕ := 3
  let final_roses : ℕ := 12
  let final_orchids : ℕ := 2
  final_roses - final_orchids = 10 := by
  sorry

end NUMINAMATH_CALUDE_rose_orchid_difference_l2803_280340


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l2803_280382

/-- A circle passing through three points (0,0), (4,0), and (-1,1) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- The three points that the circle passes through -/
def point1 : ℝ × ℝ := (0, 0)
def point2 : ℝ × ℝ := (4, 0)
def point3 : ℝ × ℝ := (-1, 1)

theorem circle_passes_through_points :
  circle_equation point1.1 point1.2 ∧
  circle_equation point2.1 point2.2 ∧
  circle_equation point3.1 point3.2 :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l2803_280382


namespace NUMINAMATH_CALUDE_check_error_proof_l2803_280389

theorem check_error_proof (x y : ℕ) : 
  x ≥ 10 ∧ x < 100 ∧ y ≥ 10 ∧ y < 100 →  -- x and y are two-digit numbers
  y = 3 * x - 6 →                        -- y = 3x - 6
  100 * y + x - (100 * x + y) = 2112 →   -- difference is $21.12 (2112 cents)
  x = 14 ∧ y = 36 :=                     -- conclusion: x = 14 and y = 36
by sorry

end NUMINAMATH_CALUDE_check_error_proof_l2803_280389


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2803_280306

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_mean1 : (a 2 + a 6) / 2 = 5)
  (h_mean2 : (a 3 + a 7) / 2 = 7) :
  ∃ (b c : ℝ), ∀ n : ℕ, a n = b * n + c ∧ b = 2 ∧ c = -3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2803_280306


namespace NUMINAMATH_CALUDE_car_truck_difference_l2803_280376

theorem car_truck_difference (total_vehicles : ℕ) (trucks : ℕ) 
  (h1 : total_vehicles = 69) 
  (h2 : trucks = 21) : 
  total_vehicles - trucks - trucks = 27 := by
  sorry

end NUMINAMATH_CALUDE_car_truck_difference_l2803_280376


namespace NUMINAMATH_CALUDE_distance_to_y_axis_angle_bisector_and_line_l2803_280344

-- Define the point M
def M (m : ℝ) : ℝ × ℝ := (2 - m, 1 + 2*m)

-- Part 1
theorem distance_to_y_axis (m : ℝ) :
  abs (2 - m) = 2 → (M m = (2, 1) ∨ M m = (-2, 9)) :=
sorry

-- Part 2
theorem angle_bisector_and_line (m k b : ℝ) :
  (2 - m = 1 + 2*m) →  -- M lies on angle bisector
  ((2 - m) = k*(2 - m) + b) →  -- Line passes through M
  (0 = k*0 + b) →  -- Line passes through (0,5)
  (5 = k*0 + b) →
  (k = -2 ∧ b = 5) :=  -- Line equation is y = -2x + 5
sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_angle_bisector_and_line_l2803_280344


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l2803_280348

/-- Profit calculation for Mary and Harry's partnership --/
theorem partnership_profit_calculation
  (mary_investment harry_investment : ℚ)
  (effort_share investment_share : ℚ)
  (mary_extra : ℚ)
  (h1 : mary_investment = 700)
  (h2 : harry_investment = 300)
  (h3 : effort_share = 1/3)
  (h4 : investment_share = 2/3)
  (h5 : mary_extra = 800) :
  ∃ (P : ℚ),
    P = 3000 ∧
    (P/6 + (mary_investment / (mary_investment + harry_investment)) * (investment_share * P)) -
    (P/6 + (harry_investment / (mary_investment + harry_investment)) * (investment_share * P)) = mary_extra :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l2803_280348


namespace NUMINAMATH_CALUDE_homework_problem_count_l2803_280313

/-- Calculates the total number of homework problems given the number of pages and problems per page -/
def total_problems (math_pages reading_pages problems_per_page : ℕ) : ℕ :=
  (math_pages + reading_pages) * problems_per_page

/-- Proves that given 6 pages of math homework, 4 pages of reading homework, and 3 problems per page, the total number of problems is 30 -/
theorem homework_problem_count : total_problems 6 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_count_l2803_280313


namespace NUMINAMATH_CALUDE_josh_work_hours_l2803_280309

/-- Proves that Josh works 8 hours a day given the problem conditions -/
theorem josh_work_hours :
  ∀ (h : ℝ),
  (20 * h * 9 + (20 * h - 40) * 4.5 = 1980) →
  h = 8 :=
by sorry

end NUMINAMATH_CALUDE_josh_work_hours_l2803_280309


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l2803_280397

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) 
  (h1 : total = 250)
  (h2 : football = 160)
  (h3 : cricket = 90)
  (h4 : neither = 50) :
  football + cricket - (total - neither) = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l2803_280397


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_theorem_l2803_280380

/-- A structure representing a line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A structure representing a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem line_plane_perpendicular_parallel_theorem 
  (l m : Line3D) (α β : Plane3D) 
  (h_distinct_lines : l ≠ m)
  (h_distinct_planes : α ≠ β)
  (h_l_perp_α : perpendicular l α)
  (h_m_perp_β : perpendicular m β) :
  (parallel_planes α β → parallel_lines l m) ∧
  (perpendicular_planes α β → perpendicular_lines l m) :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_theorem_l2803_280380


namespace NUMINAMATH_CALUDE_third_shiny_penny_probability_l2803_280396

theorem third_shiny_penny_probability :
  let total_pennies : ℕ := 9
  let shiny_pennies : ℕ := 4
  let dull_pennies : ℕ := 5
  let probability_more_than_four_draws : ℚ :=
    (Nat.choose 4 2 * Nat.choose 5 1 + Nat.choose 4 1 * Nat.choose 5 2) / Nat.choose 9 4
  probability_more_than_four_draws = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_third_shiny_penny_probability_l2803_280396


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2803_280301

/-- A line that passes through (3, 4) and is tangent to the circle x^2 + y^2 = 25 has the equation 3x + 4y - 25 = 0 -/
theorem tangent_line_equation :
  ∃! (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y + c = 0 → a * 3 + b * 4 + c = 0) ∧ 
    (∀ x y : ℝ, x^2 + y^2 = 25 → (a * x + b * y + c)^2 = (a^2 + b^2) * 25) ∧
    a = 3 ∧ b = 4 ∧ c = -25 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2803_280301


namespace NUMINAMATH_CALUDE_intersection_single_point_l2803_280329

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
def B (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- State the theorem
theorem intersection_single_point (r : ℝ) (h_r : r > 0) 
  (h_intersection : ∃! p, p ∈ A ∩ B r) : 
  r = 3 ∨ r = 7 := by sorry

end NUMINAMATH_CALUDE_intersection_single_point_l2803_280329


namespace NUMINAMATH_CALUDE_sugar_left_l2803_280362

/-- Given that Pamela bought 9.8 ounces of sugar and spilled 5.2 ounces,
    prove that the amount of sugar left is 4.6 ounces. -/
theorem sugar_left (bought spilled : ℝ) (h1 : bought = 9.8) (h2 : spilled = 5.2) :
  bought - spilled = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_sugar_left_l2803_280362


namespace NUMINAMATH_CALUDE_misha_phone_number_l2803_280366

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.reverse = digits

def is_consecutive (a b c : ℕ) : Prop :=
  b = a + 1 ∧ c = b + 1

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0))

theorem misha_phone_number :
  ∃! n : ℕ,
    n ≥ 1000000 ∧ n < 10000000 ∧
    is_palindrome (n / 100) ∧
    is_consecutive (n % 10) ((n / 10) % 10) ((n / 100) % 10) ∧
    (n / 10000) % 9 = 0 ∧
    ∃ i : ℕ, i < 5 → (n / (10^i)) % 1000 = 111 ∧
    (is_prime ((n / 100) % 100) ∨ is_prime (n % 100)) ∧
    n = 7111765 :=
  sorry

end NUMINAMATH_CALUDE_misha_phone_number_l2803_280366


namespace NUMINAMATH_CALUDE_greatest_b_for_quadratic_range_l2803_280398

theorem greatest_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 15 ≠ -6) ↔ b ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_b_for_quadratic_range_l2803_280398


namespace NUMINAMATH_CALUDE_gym_towels_theorem_l2803_280314

def gym_problem (first_hour_guests : ℕ) : Prop :=
  let second_hour_guests := first_hour_guests + (first_hour_guests * 20 / 100)
  let third_hour_guests := second_hour_guests + (second_hour_guests * 25 / 100)
  let fourth_hour_guests := third_hour_guests + (third_hour_guests * 33 / 100)
  let fifth_hour_guests := fourth_hour_guests - (fourth_hour_guests * 15 / 100)
  let sixth_hour_guests := fifth_hour_guests
  let seventh_hour_guests := sixth_hour_guests - (sixth_hour_guests * 30 / 100)
  let eighth_hour_guests := seventh_hour_guests - (seventh_hour_guests * 50 / 100)
  let total_guests := first_hour_guests + second_hour_guests + third_hour_guests + 
                      fourth_hour_guests + fifth_hour_guests + sixth_hour_guests + 
                      seventh_hour_guests + eighth_hour_guests
  let total_towels := total_guests * 2
  total_towels = 868

theorem gym_towels_theorem : 
  gym_problem 40 := by
  sorry

#check gym_towels_theorem

end NUMINAMATH_CALUDE_gym_towels_theorem_l2803_280314


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2803_280371

theorem total_cost_calculation : 
  let sandwich_price : ℚ := 349/100
  let soda_price : ℚ := 87/100
  let sandwich_quantity : ℕ := 2
  let soda_quantity : ℕ := 4
  let total_cost : ℚ := sandwich_price * sandwich_quantity + soda_price * soda_quantity
  total_cost = 1046/100 := by
sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2803_280371


namespace NUMINAMATH_CALUDE_cranberries_left_l2803_280345

/-- The number of cranberries left in a bog after harvesting and elk consumption -/
theorem cranberries_left (total : ℕ) (harvest_percent : ℚ) (elk_eaten : ℕ) 
  (h1 : total = 60000)
  (h2 : harvest_percent = 40 / 100)
  (h3 : elk_eaten = 20000) :
  total - (total * harvest_percent).floor - elk_eaten = 16000 := by
  sorry

#check cranberries_left

end NUMINAMATH_CALUDE_cranberries_left_l2803_280345


namespace NUMINAMATH_CALUDE_num_paths_equals_1287_l2803_280353

/-- The number of blocks to the right -/
def blocks_right : ℕ := 8

/-- The number of blocks up -/
def blocks_up : ℕ := 5

/-- The total number of moves -/
def total_moves : ℕ := blocks_right + blocks_up

/-- The number of different shortest paths -/
def num_paths : ℕ := Nat.choose total_moves blocks_up

theorem num_paths_equals_1287 : num_paths = 1287 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_equals_1287_l2803_280353


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2803_280383

-- Define the edge lengths
def edge_length_small : ℚ := 4
def edge_length_large : ℚ := 24  -- 2 feet = 24 inches

-- Define the volume ratio
def volume_ratio : ℚ := (edge_length_small / edge_length_large) ^ 3

-- Theorem statement
theorem cube_volume_ratio :
  volume_ratio = 1 / 216 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2803_280383


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_l2803_280307

theorem sum_real_imag_parts (z : ℂ) : z = 1 + I → (z.re + z.im = 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_l2803_280307


namespace NUMINAMATH_CALUDE_original_price_correct_l2803_280349

/-- The original price of an article before discounts -/
def original_price : ℝ := 81.30

/-- The final sale price after all discounts -/
def final_price : ℝ := 36

/-- The list of discount rates -/
def discount_rates : List ℝ := [0.15, 0.25, 0.20, 0.18]

/-- Calculate the price after applying all discounts -/
def price_after_discounts (price : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) price

theorem original_price_correct : 
  ∃ ε > 0, abs (price_after_discounts original_price discount_rates - final_price) < ε :=
by
  sorry

#eval price_after_discounts original_price discount_rates

end NUMINAMATH_CALUDE_original_price_correct_l2803_280349


namespace NUMINAMATH_CALUDE_ellipse_equation_l2803_280354

theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := Real.sqrt 3 / 2
  let perimeter := 16
  let eccentricity := (Real.sqrt (a^2 - b^2)) / a
  eccentricity = e ∧ perimeter = 4 * a → a^2 = 16 ∧ b^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2803_280354


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l2803_280333

theorem quadratic_inequality_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, a * x^2 - 2*x + 2 > 0) →
  a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l2803_280333


namespace NUMINAMATH_CALUDE_no_three_digit_even_sum_27_l2803_280395

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_three_digit_even_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ Even n :=
sorry

end NUMINAMATH_CALUDE_no_three_digit_even_sum_27_l2803_280395


namespace NUMINAMATH_CALUDE_rabbit_travel_time_l2803_280384

def rabbit_speed : ℝ := 10  -- miles per hour
def distance : ℝ := 3  -- miles

theorem rabbit_travel_time : 
  (distance / rabbit_speed) * 60 = 18 := by sorry

end NUMINAMATH_CALUDE_rabbit_travel_time_l2803_280384


namespace NUMINAMATH_CALUDE_min_staff_members_theorem_l2803_280316

/-- Represents the seating arrangement in a school hall --/
structure SchoolHall where
  male_students : ℕ
  female_students : ℕ
  benches_3_seats : ℕ
  benches_4_seats : ℕ

/-- Calculates the minimum number of staff members required --/
def min_staff_members (hall : SchoolHall) : ℕ :=
  let total_students := hall.male_students + hall.female_students
  let total_seats := 3 * hall.benches_3_seats + 4 * hall.benches_4_seats
  max (total_students - total_seats) 0

/-- Theorem stating the minimum number of staff members required --/
theorem min_staff_members_theorem (hall : SchoolHall) : 
  hall.male_students = 29 ∧ 
  hall.female_students = 4 * hall.male_students ∧ 
  hall.benches_3_seats = 15 ∧ 
  hall.benches_4_seats = 14 →
  min_staff_members hall = 44 := by
  sorry

#eval min_staff_members {
  male_students := 29,
  female_students := 116,
  benches_3_seats := 15,
  benches_4_seats := 14
}

end NUMINAMATH_CALUDE_min_staff_members_theorem_l2803_280316


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2803_280312

theorem quadratic_coefficient (c : ℝ) : 
  ((-9 : ℝ)^2 + c * (-9) - 36 = 0) → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2803_280312


namespace NUMINAMATH_CALUDE_number_circle_exists_l2803_280321

/-- A type representing a three-digit number with no zero digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_nonzero : hundreds ≠ 0
  tens_nonzero : tens ≠ 0
  ones_nonzero : ones ≠ 0
  hundreds_lt_ten : hundreds < 10
  tens_lt_ten : tens < 10
  ones_lt_ten : ones < 10

/-- A type representing a circle of six three-digit numbers -/
structure NumberCircle where
  numbers : Fin 6 → ThreeDigitNumber
  all_different : ∀ i j, i ≠ j → numbers i ≠ numbers j
  circular_property : ∀ i, 
    (numbers i).tens = (numbers ((i + 1) % 6)).hundreds ∧
    (numbers i).ones = (numbers ((i + 1) % 6)).tens

/-- Function to check if a number is divisible by n -/
def isDivisibleBy (num : ThreeDigitNumber) (n : Nat) : Prop :=
  (100 * num.hundreds + 10 * num.tens + num.ones) % n = 0

/-- The main theorem -/
theorem number_circle_exists (n : Nat) : 
  (∃ circle : NumberCircle, ∀ i, isDivisibleBy (circle.numbers i) n) ↔ n = 3 ∨ n = 7 :=
sorry

end NUMINAMATH_CALUDE_number_circle_exists_l2803_280321


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l2803_280341

-- Part 1
theorem factorization_1 (x y : ℝ) : 
  (x - y)^2 - 4*(x - y) + 4 = (x - y - 2)^2 := by sorry

-- Part 2
theorem factorization_2 (a b : ℝ) : 
  (a^2 + b^2)^2 - 4*a^2*b^2 = (a - b)^2 * (a + b)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l2803_280341


namespace NUMINAMATH_CALUDE_QR_length_l2803_280339

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the point N on QR
def N_on_QR (Q R N : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ N = (1 - t) • Q + t • R

-- Define the ratio condition for N on QR
def N_divides_QR_in_ratio (Q R N : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, dist Q N = 2 * x ∧ dist N R = 3 * x

-- Main theorem
theorem QR_length 
  (P Q R N : ℝ × ℝ) 
  (triangle : Triangle P Q R)
  (pr_length : dist P R = 5)
  (pq_length : dist P Q = 7)
  (n_on_qr : N_on_QR Q R N)
  (n_divides_qr : N_divides_QR_in_ratio Q R N)
  (pn_length : dist P N = 4) :
  dist Q R = 5 * Real.sqrt 3.9 := by
  sorry


end NUMINAMATH_CALUDE_QR_length_l2803_280339


namespace NUMINAMATH_CALUDE_fraction_of_muscle_gain_as_fat_l2803_280361

/-- Calculates the fraction of muscle gain that is fat given initial weight, muscle gain percentage, and final weight. -/
theorem fraction_of_muscle_gain_as_fat 
  (initial_weight : ℝ) 
  (muscle_gain_percentage : ℝ) 
  (final_weight : ℝ) 
  (h1 : initial_weight = 120)
  (h2 : muscle_gain_percentage = 0.20)
  (h3 : final_weight = 150) :
  (final_weight - initial_weight - muscle_gain_percentage * initial_weight) / (muscle_gain_percentage * initial_weight) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_muscle_gain_as_fat_l2803_280361


namespace NUMINAMATH_CALUDE_problem_solution_l2803_280323

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) :
  (x - 3)^2 + 16 / (x - 3)^2 = 23 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2803_280323


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l2803_280303

/-- Represents a modified cube with smaller cubes removed from alternate corners -/
structure ModifiedCube where
  original_side_length : ℕ
  removed_cube_side_length : ℕ
  corners_removed : ℕ

/-- Calculates the number of edges in the modified cube -/
def edge_count (cube : ModifiedCube) : ℕ :=
  12 + 6 * cube.corners_removed

/-- Theorem stating that a cube of side length 5 with 1x1 cubes removed from 4 corners has 36 edges -/
theorem modified_cube_edge_count :
  ∀ (cube : ModifiedCube),
    cube.original_side_length = 5 →
    cube.removed_cube_side_length = 1 →
    cube.corners_removed = 4 →
    edge_count cube = 36 :=
by
  sorry

#check modified_cube_edge_count

end NUMINAMATH_CALUDE_modified_cube_edge_count_l2803_280303


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l2803_280388

theorem quadratic_root_m_value :
  ∀ m : ℝ, (1 : ℝ)^2 + m * 1 + 2 = 0 → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l2803_280388


namespace NUMINAMATH_CALUDE_floor_equation_solution_set_l2803_280315

theorem floor_equation_solution_set (x : ℝ) :
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 5/3 ≤ x ∧ x < 7/3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_set_l2803_280315


namespace NUMINAMATH_CALUDE_rectangle_cylinder_max_volume_l2803_280302

theorem rectangle_cylinder_max_volume (x y : Real) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 9) :
  let V := π * x * y^2
  (∀ x' y' : Real, x' > 0 → y' > 0 → x' + y' = 9 → π * x' * y'^2 ≤ π * x * y^2) →
  x = 6 ∧ V = 108 * π :=
by sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_max_volume_l2803_280302


namespace NUMINAMATH_CALUDE_problem_solution_l2803_280320

def S : Set ℝ := {x | (x + 2) / (x - 5) < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 15}

theorem problem_solution :
  (S = {x : ℝ | -2 < x ∧ x < 5}) ∧
  (∀ a : ℝ, S ⊆ P a ↔ -5 ≤ a ∧ a ≤ -3) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2803_280320


namespace NUMINAMATH_CALUDE_partition_of_naturals_l2803_280327

/-- The set of natural numbers starting from 1 -/
def ℕ' : Set ℕ := {n : ℕ | n ≥ 1}

/-- The set S(x, y) for real x and y -/
def S (x y : ℝ) : Set ℕ := {s : ℕ | ∃ n : ℕ, n ∈ ℕ' ∧ s = ⌊n * x + y⌋}

/-- The main theorem -/
theorem partition_of_naturals (r : ℚ) (hr : r > 1) :
  ∃ u v : ℝ, (S r 0 ∩ S u v = ∅) ∧ (S r 0 ∪ S u v = ℕ') := by
  sorry

end NUMINAMATH_CALUDE_partition_of_naturals_l2803_280327


namespace NUMINAMATH_CALUDE_chord_length_polar_l2803_280322

/-- The length of the chord intercepted by a line on a circle in polar coordinates -/
theorem chord_length_polar (r : ℝ) (h : r > 0) :
  let line := {θ : ℝ | r * (Real.sin θ + Real.cos θ) = 2 * Real.sqrt 2}
  let circle := {ρ : ℝ | ρ = 2 * Real.sqrt 2}
  let chord_length := 2 * Real.sqrt ((2 * Real.sqrt 2)^2 - (2 * Real.sqrt 2 / Real.sqrt 2)^2)
  chord_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_polar_l2803_280322


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l2803_280351

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) (1/2 : ℝ) ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) (1/2 : ℝ) → f x ≤ f c ∧ f c = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l2803_280351


namespace NUMINAMATH_CALUDE_disrespectful_polynomial_max_value_l2803_280363

/-- A quadratic polynomial with real coefficients and leading coefficient 1 -/
structure QuadraticPolynomial where
  b : ℝ
  c : ℝ

/-- Evaluation of a quadratic polynomial at a point x -/
def evaluate (q : QuadraticPolynomial) (x : ℝ) : ℝ :=
  x^2 + q.b * x + q.c

/-- A quadratic polynomial is disrespectful if q(q(x)) = 0 has exactly three distinct real roots -/
def isDisrespectful (q : QuadraticPolynomial) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    evaluate q (evaluate q x) = 0 ∧
    evaluate q (evaluate q y) = 0 ∧
    evaluate q (evaluate q z) = 0 ∧
    ∀ w : ℝ, evaluate q (evaluate q w) = 0 → w = x ∨ w = y ∨ w = z

theorem disrespectful_polynomial_max_value :
  ∀ q : QuadraticPolynomial, isDisrespectful q → evaluate q 2 ≤ 45/16 :=
sorry

end NUMINAMATH_CALUDE_disrespectful_polynomial_max_value_l2803_280363


namespace NUMINAMATH_CALUDE_smoothie_ingredients_total_l2803_280356

theorem smoothie_ingredients_total (strawberries yogurt orange_juice : ℚ) 
  (h1 : strawberries = 0.2)
  (h2 : yogurt = 0.1)
  (h3 : orange_juice = 0.2) :
  strawberries + yogurt + orange_juice = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_ingredients_total_l2803_280356


namespace NUMINAMATH_CALUDE_equation_solutions_count_l2803_280310

theorem equation_solutions_count : 
  let count := Finset.filter (fun k => 
    k % 2 = 1 ∧ 
    (Finset.filter (fun p : ℕ × ℕ => 
      let (m, n) := p
      2^(4*m^2) + 2^(m^2 - n^2 + 4) = 2^(k + 4) + 2^(3*m^2 + n^2 + k) ∧ 
      m > 0 ∧ n > 0
    ) (Finset.product (Finset.range 101) (Finset.range 101))).card = 2
  ) (Finset.range 101)
  count.card = 18 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l2803_280310


namespace NUMINAMATH_CALUDE_fifth_month_sale_l2803_280350

/-- Given sales data for 6 months, prove the sale amount for the fifth month --/
theorem fifth_month_sale 
  (sale1 sale2 sale3 sale4 sale6 : ℕ)
  (average : ℚ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 7230)
  (h6 : sale6 = 7391)
  (h_avg : average = 6900)
  (h_avg_def : average = (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6) :
  sale5 = 6562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l2803_280350


namespace NUMINAMATH_CALUDE_committee_size_l2803_280311

/-- Given a committee of n members where any 3 individuals can be sent for a social survey,
    if the probability of female student B being chosen given that male student A is chosen is 0.4,
    then n = 6. -/
theorem committee_size (n : ℕ) : 
  (n ≥ 3) →  -- Ensure committee size is at least 3
  (((n - 2 : ℚ) / ((n - 1) * (n - 2) / 2)) = 0.4) → 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_committee_size_l2803_280311


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l2803_280387

/-- Given a cubic polynomial x^3 + bx + c = 0 with rational coefficients b and c,
    if 5 - √2 is a root, then -10 is also a root. -/
theorem integer_root_of_cubic (b c : ℚ) : 
  (5 - Real.sqrt 2)^3 + b*(5 - Real.sqrt 2) + c = 0 →
  (-10)^3 + b*(-10) + c = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l2803_280387


namespace NUMINAMATH_CALUDE_chocolate_solution_l2803_280373

def chocolate_problem (n : ℕ) (c s : ℝ) : Prop :=
  -- Condition 1: The cost price of n chocolates equals the selling price of 150 chocolates
  n * c = 150 * s ∧
  -- Condition 2: The gain percent is 10
  (s - c) / c = 0.1

theorem chocolate_solution :
  ∃ (n : ℕ) (c s : ℝ), chocolate_problem n c s ∧ n = 165 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_solution_l2803_280373


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l2803_280330

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem least_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 118 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l2803_280330


namespace NUMINAMATH_CALUDE_square_of_1009_l2803_280325

theorem square_of_1009 : 1009 ^ 2 = 1018081 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1009_l2803_280325


namespace NUMINAMATH_CALUDE_greatest_n_value_l2803_280364

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 ∧ ∃ m : ℤ, m = 7 ∧ 101 * m^2 ≤ 6400 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l2803_280364


namespace NUMINAMATH_CALUDE_franks_change_is_four_l2803_280375

/-- Calculates the change Frank has after buying peanuts -/
def franks_change (one_dollar_bills five_dollar_bills ten_dollar_bills twenty_dollar_bills : ℕ)
  (peanut_cost_per_pound : ℕ) (daily_consumption : ℕ) (days : ℕ) : ℕ :=
  let total_money := one_dollar_bills + 5 * five_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills
  let total_peanuts := daily_consumption * days
  let peanut_cost := peanut_cost_per_pound * total_peanuts
  total_money - peanut_cost

theorem franks_change_is_four :
  franks_change 7 4 2 1 3 3 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_franks_change_is_four_l2803_280375


namespace NUMINAMATH_CALUDE_bracelets_lost_l2803_280372

theorem bracelets_lost (initial_bracelets : ℕ) (remaining_bracelets : ℕ) 
  (h1 : initial_bracelets = 9) 
  (h2 : remaining_bracelets = 7) : 
  initial_bracelets - remaining_bracelets = 2 := by
  sorry

end NUMINAMATH_CALUDE_bracelets_lost_l2803_280372


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2803_280318

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 7*a + 7 = 0) → (b^2 - 7*b + 7 = 0) → a^2 + b^2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2803_280318


namespace NUMINAMATH_CALUDE_modulus_of_z_equals_one_l2803_280338

open Complex

theorem modulus_of_z_equals_one (z : ℂ) (h : z * (1 + I) = 1 - I) : abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_equals_one_l2803_280338


namespace NUMINAMATH_CALUDE_line_x_intercept_l2803_280346

/-- A straight line passing through two points (2, -3) and (6, 5) has an x-intercept of 7/2 -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ),
  (f 2 = -3) →
  (f 6 = 5) →
  (∀ x y : ℝ, f y - f x = (y - x) * ((5 - (-3)) / (6 - 2))) →
  (∃ x : ℝ, f x = 0 ∧ x = 7/2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_l2803_280346


namespace NUMINAMATH_CALUDE_correct_recommendation_count_l2803_280360

/-- Represents the number of recommendation spots for each language -/
structure SpotDistribution :=
  (korean : Nat)
  (japanese : Nat)
  (russian : Nat)

/-- Represents the gender distribution of candidates -/
structure CandidateDistribution :=
  (female : Nat)
  (male : Nat)

/-- Calculates the number of different recommendation methods -/
def recommendationMethods (spots : SpotDistribution) (candidates : CandidateDistribution) : Nat :=
  sorry

/-- Theorem stating the number of different recommendation methods -/
theorem correct_recommendation_count :
  let spots : SpotDistribution := ⟨2, 2, 1⟩
  let candidates : CandidateDistribution := ⟨3, 2⟩
  recommendationMethods spots candidates = 24 := by
  sorry

end NUMINAMATH_CALUDE_correct_recommendation_count_l2803_280360


namespace NUMINAMATH_CALUDE_polynomial_roots_l2803_280319

theorem polynomial_roots : 
  ∀ z : ℂ, z^4 - 6*z^2 + z + 8 = 0 ↔ z = -2 ∨ z = 1 ∨ z = Complex.I * Real.sqrt 7 ∨ z = -Complex.I * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2803_280319


namespace NUMINAMATH_CALUDE_deck_card_count_l2803_280326

theorem deck_card_count (r : ℕ) (b : ℕ) : 
  b = 2 * r →                 -- Initial condition: black cards are twice red cards
  (b + 4) = 3 * r →           -- After adding 4 black cards, they're three times red cards
  r + b = 12                  -- The initial total number of cards is 12
  := by sorry

end NUMINAMATH_CALUDE_deck_card_count_l2803_280326


namespace NUMINAMATH_CALUDE_g_twelve_equals_thirtysix_l2803_280386

/-- The area function of a rectangle with side lengths x and x+1 -/
def f (x : ℝ) : ℝ := x * (x + 1)

/-- The function g satisfying f(g(x)) = 9x^2 + 3x -/
noncomputable def g (x : ℝ) : ℝ := 
  (- 1 + Real.sqrt (36 * x^2 + 12 * x + 1)) / 2

theorem g_twelve_equals_thirtysix : g 12 = 36 := by sorry

end NUMINAMATH_CALUDE_g_twelve_equals_thirtysix_l2803_280386


namespace NUMINAMATH_CALUDE_equation_solution_l2803_280342

theorem equation_solution : 
  let x : ℚ := 30
  40 * x + (12 + 8) * 3 / 5 = 1212 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2803_280342


namespace NUMINAMATH_CALUDE_shopping_remaining_money_l2803_280352

/-- Calculates the remaining money after shopping --/
def remaining_money (initial_amount novel_cost : ℕ) : ℕ :=
  initial_amount - (novel_cost + 2 * novel_cost)

/-- Theorem: Given $50 initial amount and $7 novel cost, the remaining money is $29 --/
theorem shopping_remaining_money : remaining_money 50 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_shopping_remaining_money_l2803_280352


namespace NUMINAMATH_CALUDE_janes_drinks_l2803_280328

theorem janes_drinks (b m d : ℕ) : 
  b + m + d = 5 →
  (90 * b + 40 * m + 30 * d) % 100 = 0 →
  d = 4 := by
sorry

end NUMINAMATH_CALUDE_janes_drinks_l2803_280328


namespace NUMINAMATH_CALUDE_f_geq_g_l2803_280369

noncomputable def f (a b x : ℝ) : ℝ := x^2 * Real.exp (x - 1) + a * x^3 + b * x^2

noncomputable def g (x : ℝ) : ℝ := (2/3) * x^3 - x^2

theorem f_geq_g (a b : ℝ) :
  (∀ x : ℝ, (deriv (f a b)) x = 0 ↔ x = -2 ∨ x = 1) →
  ∀ x : ℝ, f (-1/3) (-1) x ≥ g x :=
by sorry

end NUMINAMATH_CALUDE_f_geq_g_l2803_280369


namespace NUMINAMATH_CALUDE_prob_more_ones_than_sixes_l2803_280343

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The total number of possible outcomes when rolling numDice dice -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The number of outcomes where the same number of 1's and 6's are rolled -/
def sameOnesSixes : ℕ := 2424

/-- The probability of rolling more 1's than 6's when rolling numDice fair numSides-sided dice -/
def probMoreOnesThanSixes : ℚ := 2676 / 7776

theorem prob_more_ones_than_sixes :
  probMoreOnesThanSixes = 1 / 2 * (1 - sameOnesSixes / totalOutcomes) :=
sorry

end NUMINAMATH_CALUDE_prob_more_ones_than_sixes_l2803_280343


namespace NUMINAMATH_CALUDE_inequality_proof_l2803_280399

noncomputable def a : ℝ := 1 + Real.tan (-0.2)
noncomputable def b : ℝ := Real.log (0.8 * Real.exp 1)
noncomputable def c : ℝ := 1 / Real.exp 0.2

theorem inequality_proof : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2803_280399


namespace NUMINAMATH_CALUDE_investment_interest_rate_l2803_280367

theorem investment_interest_rate 
  (total_investment : ℝ) 
  (investment_at_r : ℝ) 
  (total_interest : ℝ) 
  (known_rate : ℝ) :
  total_investment = 10000 →
  investment_at_r = 7200 →
  known_rate = 0.09 →
  total_interest = 684 →
  ∃ r : ℝ, 
    r * investment_at_r + known_rate * (total_investment - investment_at_r) = total_interest ∧
    r = 0.06 :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l2803_280367


namespace NUMINAMATH_CALUDE_votes_against_percentage_l2803_280304

theorem votes_against_percentage (total_votes : ℕ) (votes_difference : ℕ) : 
  total_votes = 290 → 
  votes_difference = 58 → 
  (((total_votes - votes_difference) / 2 : ℚ) / total_votes) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_votes_against_percentage_l2803_280304


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l2803_280379

/-- The area of a square with adjacent vertices at (0,3) and (4,0) is 25. -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 3)
  let p2 : ℝ × ℝ := (4, 0)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l2803_280379


namespace NUMINAMATH_CALUDE_sets_problem_l2803_280300

-- Define the sets A, B, and C
def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 5 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem statement
theorem sets_problem (a : ℝ) :
  (A ∪ B = {x : ℝ | 4 ≤ x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | 8 ≤ x ∧ x < 10}) ∧
  (Set.Nonempty (A ∩ C a) ↔ a < 8) := by
  sorry

end NUMINAMATH_CALUDE_sets_problem_l2803_280300


namespace NUMINAMATH_CALUDE_sector_central_angle_l2803_280357

/-- Given a circular sector with perimeter 8 and area 3, 
    its central angle is either 6 or 2/3 radians. -/
theorem sector_central_angle (r l : ℝ) : 
  (2 * r + l = 8) →  -- perimeter condition
  (1 / 2 * l * r = 3) →  -- area condition
  (l / r = 6 ∨ l / r = 2 / 3) := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2803_280357


namespace NUMINAMATH_CALUDE_line_equation_l2803_280377

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define when a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem line_equation (l : Line2D) :
  (pointOnLine ⟨0, 0⟩ l) →
  (perpendicular l ⟨1, -1, -3⟩) →
  l = ⟨1, 1, 0⟩ := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2803_280377


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2803_280332

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 3 + 2^3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2803_280332


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2803_280370

/-- Simple interest calculation -/
theorem simple_interest_principal (interest rate time principal : ℝ) :
  interest = principal * (rate / 100) * time →
  rate = 6.666666666666667 →
  time = 4 →
  interest = 160 →
  principal = 600 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2803_280370


namespace NUMINAMATH_CALUDE_chocolate_probability_theorem_l2803_280393

/- Define the type for a box of chocolates -/
structure ChocolateBox where
  white : ℕ
  total : ℕ
  h_total : total > 0

/- Define the probability of drawing a white chocolate from a box -/
def prob (box : ChocolateBox) : ℚ :=
  box.white / box.total

/- Define the combined box of chocolates -/
def combinedBox (box1 box2 : ChocolateBox) : ChocolateBox where
  white := box1.white + box2.white
  total := box1.total + box2.total
  h_total := by
    simp [gt_iff_lt, add_pos_iff]
    exact Or.inl box1.h_total

/- Theorem statement -/
theorem chocolate_probability_theorem (box1 box2 : ChocolateBox) :
  ∃ (box1' box2' : ChocolateBox),
    (prob (combinedBox box1' box2') = 7 / 12) ∧
    (prob (combinedBox box1 box2) = 11 / 19) ∧
    (prob (combinedBox box1 box2) > min (prob box1) (prob box2)) :=
  sorry

end NUMINAMATH_CALUDE_chocolate_probability_theorem_l2803_280393


namespace NUMINAMATH_CALUDE_problem_solution_l2803_280336

theorem problem_solution (x y : ℝ) :
  (Real.sqrt (x - 3 * y) + |x^2 - 9|) / ((x + 3)^2) = 0 →
  Real.sqrt (x + 2) / Real.sqrt (y + 1) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2803_280336


namespace NUMINAMATH_CALUDE_expected_worth_is_two_l2803_280381

/-- A biased coin with given probabilities and payoffs -/
structure BiasedCoin where
  probHeads : ℝ
  probTails : ℝ
  payoffHeads : ℝ
  payoffTails : ℝ

/-- The expected value of a flip of the biased coin -/
def expectedValue (coin : BiasedCoin) : ℝ :=
  coin.probHeads * coin.payoffHeads + coin.probTails * coin.payoffTails

/-- Theorem: The expected worth of the given biased coin flip is $2 -/
theorem expected_worth_is_two :
  ∃ (coin : BiasedCoin),
    coin.probHeads = 4/5 ∧
    coin.probTails = 1/5 ∧
    coin.payoffHeads = 5 ∧
    coin.payoffTails = -10 ∧
    expectedValue coin = 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_is_two_l2803_280381


namespace NUMINAMATH_CALUDE_business_partnership_gains_l2803_280365

/-- Represents the investment and gain of a partner in the business. -/
structure Partner where
  investment : ℕ
  time : ℕ
  gain : ℕ

/-- Represents the business partnership with four partners. -/
def BusinessPartnership (nandan gopal vishal krishan : Partner) : Prop :=
  -- Investment ratios
  krishan.investment = 6 * nandan.investment ∧
  gopal.investment = 3 * nandan.investment ∧
  vishal.investment = 2 * nandan.investment ∧
  -- Time ratios
  krishan.time = 2 * nandan.time ∧
  gopal.time = 3 * nandan.time ∧
  vishal.time = nandan.time ∧
  -- Nandan's gain
  nandan.gain = 6000 ∧
  -- Gain proportionality
  krishan.gain * nandan.investment * nandan.time = nandan.gain * krishan.investment * krishan.time ∧
  gopal.gain * nandan.investment * nandan.time = nandan.gain * gopal.investment * gopal.time ∧
  vishal.gain * nandan.investment * nandan.time = nandan.gain * vishal.investment * vishal.time

/-- The theorem to be proved -/
theorem business_partnership_gains 
  (nandan gopal vishal krishan : Partner) 
  (h : BusinessPartnership nandan gopal vishal krishan) : 
  krishan.gain = 72000 ∧ 
  gopal.gain = 54000 ∧ 
  vishal.gain = 12000 ∧ 
  nandan.gain + gopal.gain + vishal.gain + krishan.gain = 144000 := by
  sorry

end NUMINAMATH_CALUDE_business_partnership_gains_l2803_280365


namespace NUMINAMATH_CALUDE_probability_A_selected_l2803_280337

/-- The number of individuals in the group -/
def n : ℕ := 3

/-- The number of representatives to be chosen -/
def k : ℕ := 2

/-- The probability of selecting A as one of the representatives -/
def prob_A_selected : ℚ := 2/3

/-- Theorem stating that the probability of selecting A as one of two representatives
    from a group of three individuals is 2/3 -/
theorem probability_A_selected :
  prob_A_selected = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_A_selected_l2803_280337


namespace NUMINAMATH_CALUDE_simplify_expression_l2803_280358

theorem simplify_expression (x : ℝ) : ((-3 * x)^2) * (2 * x) = 18 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2803_280358


namespace NUMINAMATH_CALUDE_ice_cream_scoop_permutations_l2803_280385

theorem ice_cream_scoop_permutations :
  (Finset.range 5).card.factorial = 120 := by sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_permutations_l2803_280385


namespace NUMINAMATH_CALUDE_lucky_larry_coincidence_l2803_280347

theorem lucky_larry_coincidence :
  let a : ℤ := 2
  let b : ℤ := 3
  let c : ℤ := 4
  let d : ℤ := 5
  ∃ f : ℤ, (a + b - c + d - f = a + (b - (c + (d - f)))) ∧ f = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_coincidence_l2803_280347


namespace NUMINAMATH_CALUDE_soda_difference_l2803_280324

/-- The number of liters of soda in each bottle -/
def liters_per_bottle : ℕ := 2

/-- The number of orange soda bottles Julio has -/
def julio_orange : ℕ := 4

/-- The number of grape soda bottles Julio has -/
def julio_grape : ℕ := 7

/-- The number of orange soda bottles Mateo has -/
def mateo_orange : ℕ := 1

/-- The number of grape soda bottles Mateo has -/
def mateo_grape : ℕ := 3

/-- The difference in total liters of soda between Julio and Mateo -/
theorem soda_difference : 
  (julio_orange + julio_grape) * liters_per_bottle - 
  (mateo_orange + mateo_grape) * liters_per_bottle = 14 := by
sorry

end NUMINAMATH_CALUDE_soda_difference_l2803_280324


namespace NUMINAMATH_CALUDE_sum_abc_equals_33_l2803_280317

theorem sum_abc_equals_33 
  (a b c N : ℕ+) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_eq1 : N = 5*a + 3*b + 5*c)
  (h_eq2 : N = 4*a + 5*b + 4*c)
  (h_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_33_l2803_280317


namespace NUMINAMATH_CALUDE_inequalities_given_sum_positive_l2803_280335

theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_sum_positive_l2803_280335


namespace NUMINAMATH_CALUDE_water_bucket_addition_l2803_280391

theorem water_bucket_addition (initial_water : Real) (added_water : Real) :
  initial_water = 3 ∧ added_water = 6.8 → initial_water + added_water = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_water_bucket_addition_l2803_280391


namespace NUMINAMATH_CALUDE_oil_leak_during_repair_l2803_280392

/-- Represents the oil leak scenario -/
structure OilLeak where
  initial_leak : ℝ
  initial_time : ℝ
  repair_time : ℝ
  rate_reduction : ℝ
  total_leak : ℝ

/-- Calculates the amount of oil leaked during repair -/
def leak_during_repair (scenario : OilLeak) : ℝ :=
  let initial_rate := scenario.initial_leak / scenario.initial_time
  let reduced_rate := initial_rate * scenario.rate_reduction
  scenario.total_leak - scenario.initial_leak

/-- Theorem stating the amount of oil leaked during repair -/
theorem oil_leak_during_repair :
  let scenario : OilLeak := {
    initial_leak := 2475,
    initial_time := 7,
    repair_time := 5,
    rate_reduction := 0.75,
    total_leak := 6206
  }
  leak_during_repair scenario = 3731 := by sorry

end NUMINAMATH_CALUDE_oil_leak_during_repair_l2803_280392


namespace NUMINAMATH_CALUDE_value_of_a_l2803_280390

theorem value_of_a (S T : Set ℕ) (a : ℕ) : 
  S = {1, 2} → T = {a} → S ∪ T = S → a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2803_280390


namespace NUMINAMATH_CALUDE_fence_length_l2803_280355

/-- Given a straight wire fence with 12 equally spaced posts, where the distance between
    the third and the sixth post is 3.3 m, the total length of the fence is 12.1 meters. -/
theorem fence_length (num_posts : ℕ) (distance_3_to_6 : ℝ) :
  num_posts = 12 →
  distance_3_to_6 = 3.3 →
  (num_posts - 1 : ℝ) * (distance_3_to_6 / 3) = 12.1 := by
  sorry

end NUMINAMATH_CALUDE_fence_length_l2803_280355


namespace NUMINAMATH_CALUDE_mary_fruit_purchase_cost_l2803_280368

/-- Represents the cost of each fruit type -/
structure FruitCosts where
  apple : ℕ
  orange : ℕ
  banana : ℕ
  peach : ℕ
  grape : ℕ

/-- Represents the quantity of each fruit type bought -/
structure FruitQuantities where
  apple : ℕ
  orange : ℕ
  banana : ℕ
  peach : ℕ
  grape : ℕ

/-- Calculates the total cost before discounts -/
def totalCostBeforeDiscounts (costs : FruitCosts) (quantities : FruitQuantities) : ℕ :=
  costs.apple * quantities.apple +
  costs.orange * quantities.orange +
  costs.banana * quantities.banana +
  costs.peach * quantities.peach +
  costs.grape * quantities.grape

/-- Calculates the discount for every 5 fruits bought -/
def fiveForOneDiscount (totalFruits : ℕ) : ℕ :=
  totalFruits / 5

/-- Calculates the discount for peaches and grapes bought together -/
def peachGrapeDiscount (peaches : ℕ) (grapes : ℕ) : ℕ :=
  (min (peaches / 3) (grapes / 2)) * 3

/-- Calculates the final cost after applying discounts -/
def finalCost (costs : FruitCosts) (quantities : FruitQuantities) : ℕ :=
  let totalFruits := quantities.apple + quantities.orange + quantities.banana + quantities.peach + quantities.grape
  let costBeforeDiscounts := totalCostBeforeDiscounts costs quantities
  let fiveForOneDiscountAmount := fiveForOneDiscount totalFruits
  let peachGrapeDiscountAmount := peachGrapeDiscount quantities.peach quantities.grape
  costBeforeDiscounts - fiveForOneDiscountAmount - peachGrapeDiscountAmount

/-- Theorem: Mary will pay $51 for her fruit purchase -/
theorem mary_fruit_purchase_cost :
  let costs : FruitCosts := { apple := 1, orange := 2, banana := 3, peach := 4, grape := 5 }
  let quantities : FruitQuantities := { apple := 5, orange := 3, banana := 2, peach := 6, grape := 4 }
  finalCost costs quantities = 51 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_purchase_cost_l2803_280368


namespace NUMINAMATH_CALUDE_school_demographics_l2803_280331

theorem school_demographics (total_students : ℕ) (boys_avg_age girls_avg_age school_avg_age : ℚ) : 
  total_students = 632 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  school_avg_age = 47/4 →
  ∃ (num_girls : ℕ), num_girls = 156 ∧ num_girls ≤ total_students := by
  sorry

end NUMINAMATH_CALUDE_school_demographics_l2803_280331


namespace NUMINAMATH_CALUDE_no_base_for_square_202_l2803_280394

-- Define the base-b representation of 202_b
def base_b_representation (b : ℕ) : ℕ := 2 * b^2 + 2

-- Define the property of being a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

-- Theorem statement
theorem no_base_for_square_202 :
  ∀ b : ℕ, b > 2 → ¬(is_perfect_square (base_b_representation b)) := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_square_202_l2803_280394


namespace NUMINAMATH_CALUDE_range_of_a_eq_l2803_280305

/-- Proposition p: The solution set of the inequality x^2 + (a-1)x + a^2 < 0 is empty. -/
def prop_p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + a^2 ≥ 0

/-- Quadratic function f(x) = x^2 - mx + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2

/-- Proposition q: f(3/2 + x) = f(3/2 - x), and max(f(x)) = 2 for x ∈ [0, a] -/
def prop_q (a : ℝ) : Prop :=
  ∃ m, (∀ x, f m ((3:ℝ)/2 + x) = f m ((3:ℝ)/2 - x)) ∧
       (∀ x, x ∈ Set.Icc 0 a → f m x ≤ 2) ∧
       (∃ x, x ∈ Set.Icc 0 a ∧ f m x = 2)

/-- The range of a given the conditions -/
def range_of_a : Set ℝ :=
  {a | (¬(prop_p a ∧ prop_q a)) ∧ (prop_p a ∨ prop_q a)}

theorem range_of_a_eq :
  range_of_a = Set.Iic (-1) ∪ Set.Ioo 0 (1/3) ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_eq_l2803_280305


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l2803_280378

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x + 3)

theorem monotonic_decreasing_interval_of_f :
  ∀ x₁ x₂, x₁ < x₂ → x₁ < 1 → x₂ < 1 → f x₁ > f x₂ :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l2803_280378


namespace NUMINAMATH_CALUDE_cody_discount_l2803_280359

/-- The discount Cody got after taxes --/
def discount_after_taxes (initial_price tax_rate discount final_price : ℝ) : ℝ :=
  initial_price * (1 + tax_rate) - final_price

/-- Theorem stating the discount Cody got after taxes --/
theorem cody_discount :
  ∃ (discount : ℝ),
    discount_after_taxes 40 0.05 discount (2 * 17) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_cody_discount_l2803_280359


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l2803_280374

def incorrect_mean : ℝ := 120
def num_values : ℕ := 40
def original_values : List ℝ := [-50, 350, 100, 25, -80]
def incorrect_values : List ℝ := [-30, 320, 120, 60, -100]

theorem correct_mean_calculation :
  let incorrect_sum := incorrect_mean * num_values
  let difference := (List.sum original_values) - (List.sum incorrect_values)
  let correct_sum := incorrect_sum + difference
  correct_sum / num_values = 119.375 := by
sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l2803_280374


namespace NUMINAMATH_CALUDE_quadratic_and_trig_problem_l2803_280334

theorem quadratic_and_trig_problem :
  -- Part 1: Quadratic equation
  (∃ x1 x2 : ℝ, x1 = 1 + Real.sqrt 2 ∧ x2 = 1 - Real.sqrt 2 ∧
    x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0) ∧
  -- Part 2: Trigonometric expression
  (4 * (Real.sin (60 * π / 180))^2 - Real.tan (45 * π / 180) +
   Real.sqrt 2 * Real.cos (45 * π / 180) - Real.sin (30 * π / 180) = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_trig_problem_l2803_280334
