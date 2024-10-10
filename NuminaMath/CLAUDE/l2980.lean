import Mathlib

namespace starting_lineup_combinations_l2980_298058

def team_size : ℕ := 12
def offensive_linemen : ℕ := 5

theorem starting_lineup_combinations : 
  (offensive_linemen) *
  (team_size - 1) *
  (team_size - 2) *
  ((team_size - 3) * (team_size - 4) / 2) = 19800 :=
by sorry

end starting_lineup_combinations_l2980_298058


namespace sequence_problem_l2980_298043

theorem sequence_problem (a : ℕ → ℕ) (n : ℕ) :
  a 1 = 1 ∧
  (∀ k, a (k + 1) = a k + 3) ∧
  a n = 2014 →
  n = 672 := by
  sorry

end sequence_problem_l2980_298043


namespace binary_operation_equality_l2980_298065

/-- Convert a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Convert a decimal number to its binary representation -/
def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Perform binary multiplication -/
def binary_mult (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a * binary_to_decimal b)

/-- Perform binary division -/
def binary_div (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a / binary_to_decimal b)

theorem binary_operation_equality : 
  let a := [true, true, false, false, true, false]  -- 110010₂
  let b := [true, true, false, false]               -- 1100₂
  let c := [true, false, false]                     -- 100₂
  let d := [true, false]                            -- 10₂
  let result := [true, false, false, true, false, false]  -- 100100₂
  binary_div (binary_div (binary_mult a b) c) d = result := by
  sorry

end binary_operation_equality_l2980_298065


namespace presidency_meeting_arrangements_l2980_298002

/-- The number of schools -/
def num_schools : ℕ := 3

/-- The number of members per school -/
def members_per_school : ℕ := 6

/-- The number of representatives from the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives from each non-host school -/
def non_host_representatives : ℕ := 1

/-- The total number of ways to arrange the presidency meeting -/
def total_arrangements : ℕ := num_schools * (Nat.choose members_per_school host_representatives) * (Nat.choose members_per_school non_host_representatives)^2

theorem presidency_meeting_arrangements :
  total_arrangements = 2160 :=
sorry

end presidency_meeting_arrangements_l2980_298002


namespace expand_expression_l2980_298013

theorem expand_expression (x y : ℝ) : (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := by
  sorry

end expand_expression_l2980_298013


namespace x_value_l2980_298092

theorem x_value : ∃ x : ℝ, (3 * x = (16 - x) + 4) ∧ (x = 5) := by sorry

end x_value_l2980_298092


namespace negation_of_product_zero_implies_factor_zero_l2980_298044

theorem negation_of_product_zero_implies_factor_zero (a b c : ℝ) :
  (¬(abc = 0 → a = 0 ∨ b = 0 ∨ c = 0)) ↔ (abc = 0 → a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) := by
  sorry

end negation_of_product_zero_implies_factor_zero_l2980_298044


namespace octagon_diagonal_intersections_l2980_298031

/-- The number of vertices in a regular octagon -/
def n : ℕ := 8

/-- The number of diagonals in a regular octagon -/
def num_diagonals : ℕ := n * (n - 3) / 2

/-- The number of distinct intersection points of diagonals in the interior of a regular octagon -/
def num_intersection_points : ℕ := Nat.choose n 4

theorem octagon_diagonal_intersections :
  num_intersection_points = 70 :=
sorry

end octagon_diagonal_intersections_l2980_298031


namespace train_length_l2980_298085

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man completely. -/
theorem train_length (train_speed man_speed : ℝ) (time_to_cross : ℝ) :
  train_speed = 54.99520038396929 →
  man_speed = 5 →
  time_to_cross = 6 →
  let relative_speed := (train_speed + man_speed) * (1000 / 3600)
  let train_length := relative_speed * time_to_cross
  train_length = 99.99180063994882 := by
sorry

end train_length_l2980_298085


namespace cube_dimension_ratio_l2980_298003

theorem cube_dimension_ratio (v1 v2 : ℝ) (h1 : v1 = 216) (h2 : v2 = 1728) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
sorry

end cube_dimension_ratio_l2980_298003


namespace isi_club_member_count_l2980_298032

/-- Represents a club with committees and members -/
structure Club where
  committee_count : ℕ
  member_count : ℕ
  committees_per_member : ℕ
  common_members : ℕ

/-- The ISI club satisfies the given conditions -/
def isi_club : Club :=
  { committee_count := 5,
    member_count := 10,
    committees_per_member := 2,
    common_members := 1 }

/-- Theorem: The ISI club has 10 members -/
theorem isi_club_member_count :
  isi_club.member_count = (isi_club.committee_count.choose 2) :=
by sorry

end isi_club_member_count_l2980_298032


namespace no_square_divisible_by_six_between_50_and_120_l2980_298057

theorem no_square_divisible_by_six_between_50_and_120 : ¬ ∃ x : ℕ,
  (∃ y : ℕ, x = y^2) ∧ 
  (∃ z : ℕ, x = 6 * z) ∧ 
  50 < x ∧ x < 120 := by
sorry

end no_square_divisible_by_six_between_50_and_120_l2980_298057


namespace skt_lineups_l2980_298028

/-- The total number of StarCraft progamers -/
def total_progamers : ℕ := 111

/-- The number of progamers SKT starts with -/
def initial_skt_progamers : ℕ := 11

/-- The number of progamers in a lineup -/
def lineup_size : ℕ := 5

/-- The number of different ordered lineups SKT could field -/
def num_lineups : ℕ := 4015440

theorem skt_lineups :
  (total_progamers : ℕ) = 111 →
  (initial_skt_progamers : ℕ) = 11 →
  (lineup_size : ℕ) = 5 →
  num_lineups = (Nat.choose initial_skt_progamers lineup_size +
                 Nat.choose initial_skt_progamers (lineup_size - 1) * (total_progamers - initial_skt_progamers)) *
                (Nat.factorial lineup_size) :=
by sorry

end skt_lineups_l2980_298028


namespace extreme_values_and_range_of_a_l2980_298094

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * x^2 - x

theorem extreme_values_and_range_of_a :
  (∀ x : ℝ, f (1/4) x ≤ f (1/4) 0) ∧
  (∀ x : ℝ, f (1/4) x ≥ f (1/4) 1) ∧
  (f (1/4) 0 = 0) ∧
  (f (1/4) 1 = Real.log 2 - 3/4) ∧
  (∀ a : ℝ, (∀ b : ℝ, 1 < b → b < 2 → 
    (∀ x : ℝ, -1 < x → x ≤ b → f a x ≤ f a b)) →
    a ≥ 1 - Real.log 2) :=
sorry

end extreme_values_and_range_of_a_l2980_298094


namespace sum_xy_equals_negative_two_l2980_298024

theorem sum_xy_equals_negative_two (x y : ℝ) :
  (x + y + 2)^2 + |2*x - 3*y - 1| = 0 → x + y = -2 := by
  sorry

end sum_xy_equals_negative_two_l2980_298024


namespace orthocenter_of_specific_triangle_l2980_298079

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (101/33, 95/33, 47/33). -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, -1)
  let B : ℝ × ℝ × ℝ := (6, -1, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 4)
  orthocenter A B C = (101/33, 95/33, 47/33) := by sorry

end orthocenter_of_specific_triangle_l2980_298079


namespace quadratic_roots_relation_l2980_298034

theorem quadratic_roots_relation (p q n : ℝ) (r₁ r₂ : ℝ) : 
  (∀ x, x^2 + q*x + p = 0 ↔ x = r₁ ∨ x = r₂) →
  (∀ x, x^2 + p*x + n = 0 ↔ x = 3*r₁ ∨ x = 3*r₂) →
  p ≠ 0 → q ≠ 0 → n ≠ 0 →
  n / q = -3 := by
sorry

end quadratic_roots_relation_l2980_298034


namespace sum_of_base8_digits_878_l2980_298010

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits in the base 8 representation of 878 is 17 -/
theorem sum_of_base8_digits_878 :
  sumDigits (toBase8 878) = 17 := by
  sorry

end sum_of_base8_digits_878_l2980_298010


namespace max_value_cosine_sine_fraction_l2980_298061

theorem max_value_cosine_sine_fraction :
  ∀ x : ℝ, (1 + Real.cos x) / (Real.sin x + Real.cos x + 2) ≤ 1 := by
  sorry

end max_value_cosine_sine_fraction_l2980_298061


namespace baby_shower_parking_l2980_298068

/-- Proves that given the conditions of the baby shower parking scenario, each guest car has 4 wheels -/
theorem baby_shower_parking (num_guests : ℕ) (num_guest_cars : ℕ) (num_parent_cars : ℕ) (total_wheels : ℕ) :
  num_guests = 40 →
  num_guest_cars = 10 →
  num_parent_cars = 2 →
  total_wheels = 48 →
  (total_wheels - num_parent_cars * 4) / num_guest_cars = 4 := by
  sorry

end baby_shower_parking_l2980_298068


namespace flu_transmission_rate_l2980_298088

theorem flu_transmission_rate (initial_infected : ℕ) (total_infected : ℕ) (transmission_rate : ℝ) : 
  initial_infected = 1 →
  total_infected = 100 →
  initial_infected + transmission_rate + transmission_rate * (initial_infected + transmission_rate) = total_infected →
  transmission_rate = 9 := by
  sorry

end flu_transmission_rate_l2980_298088


namespace wheel_rotation_l2980_298020

/-- Proves that a wheel with given radius and arc length rotates by the calculated number of radians -/
theorem wheel_rotation (radius : ℝ) (arc_length : ℝ) (rotation : ℝ) 
  (h1 : radius = 20)
  (h2 : arc_length = 40)
  (h3 : rotation = arc_length / radius)
  (h4 : rotation > 0) : -- represents counterclockwise rotation
  rotation = 2 := by
  sorry

end wheel_rotation_l2980_298020


namespace jakes_and_sister_weight_l2980_298051

/-- The combined weight of Jake and his sister given Jake's current weight and the condition about their weight ratio after Jake loses weight. -/
theorem jakes_and_sister_weight (jake_weight : ℕ) (weight_loss : ℕ) : 
  jake_weight = 93 →
  weight_loss = 15 →
  (jake_weight - weight_loss) = 2 * ((jake_weight - weight_loss) / 2) →
  jake_weight + ((jake_weight - weight_loss) / 2) = 132 := by
sorry

end jakes_and_sister_weight_l2980_298051


namespace center_of_given_hyperbola_l2980_298026

/-- The center of a hyperbola is the point (h, k) in the standard form 
    (x-h)^2/a^2 - (y-k)^2/b^2 = 1 or (y-k)^2/a^2 - (x-h)^2/b^2 = 1 -/
def center_of_hyperbola (a b c d e f : ℝ) : ℝ × ℝ := sorry

/-- The equation of a hyperbola in general form is ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
def is_hyperbola (a b c d e f : ℝ) : Prop := sorry

theorem center_of_given_hyperbola :
  let a : ℝ := 9
  let b : ℝ := 0
  let c : ℝ := -16
  let d : ℝ := -54
  let e : ℝ := 128
  let f : ℝ := -400
  is_hyperbola a b c d e f →
  center_of_hyperbola a b c d e f = (3, 4) := by sorry

end center_of_given_hyperbola_l2980_298026


namespace second_week_rainfall_l2980_298081

/-- Proves that given a total rainfall of 35 inches over two weeks, 
    where the second week's rainfall is 1.5 times the first week's, 
    the rainfall in the second week is 21 inches. -/
theorem second_week_rainfall (first_week : ℝ) : 
  first_week + (1.5 * first_week) = 35 → 1.5 * first_week = 21 := by
  sorry

end second_week_rainfall_l2980_298081


namespace part_to_whole_ratio_l2980_298049

theorem part_to_whole_ratio (N P : ℝ) 
  (h1 : (1/4) * (1/3) * P = 15) 
  (h2 : 0.40 * N = 180) : 
  P / N = 2 / 5 := by
sorry

end part_to_whole_ratio_l2980_298049


namespace power_multiplication_l2980_298000

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end power_multiplication_l2980_298000


namespace cos_squared_minus_sin_squared_15_deg_l2980_298036

theorem cos_squared_minus_sin_squared_15_deg :
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end cos_squared_minus_sin_squared_15_deg_l2980_298036


namespace average_marks_of_combined_classes_l2980_298059

theorem average_marks_of_combined_classes (n₁ n₂ : ℕ) (avg₁ avg₂ : ℝ) :
  n₁ = 30 →
  n₂ = 50 →
  avg₁ = 40 →
  avg₂ = 60 →
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℝ) = 52.5 := by
  sorry

end average_marks_of_combined_classes_l2980_298059


namespace mod_fifteen_equivalence_l2980_298054

theorem mod_fifteen_equivalence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 15879 [MOD 15] ∧ n = 9 := by
  sorry

end mod_fifteen_equivalence_l2980_298054


namespace inequality_solution_range_l2980_298004

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, (x - 2*a)*(a*x - 1) < 0 ↔ (x > 1/a ∨ x < 2*a)) →
  a ≤ -Real.sqrt 2 / 2 :=
by sorry

end inequality_solution_range_l2980_298004


namespace earl_money_proof_l2980_298075

def earl_initial_money (e f g : ℕ) : Prop :=
  f = 48 ∧ 
  g = 36 ∧ 
  e - 28 + 40 + (g + 32 - 40) = 130 ∧
  e = 90

theorem earl_money_proof :
  ∀ e f g : ℕ, earl_initial_money e f g :=
by
  sorry

end earl_money_proof_l2980_298075


namespace arithmetic_expression_equality_l2980_298008

theorem arithmetic_expression_equality : 12 - 10 + 9 * 8 * 2 + 7 - 6 * 5 + 4 * 3 - 1 = 133 := by
  sorry

end arithmetic_expression_equality_l2980_298008


namespace roots_product_minus_one_l2980_298067

theorem roots_product_minus_one (d e : ℝ) : 
  (3 * d^2 + 5 * d - 2 = 0) → 
  (3 * e^2 + 5 * e - 2 = 0) → 
  (d - 1) * (e - 1) = 2 := by
sorry

end roots_product_minus_one_l2980_298067


namespace unique_pair_exists_l2980_298098

theorem unique_pair_exists (n : ℕ) : ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 := by
  sorry

end unique_pair_exists_l2980_298098


namespace largest_sphere_in_cone_l2980_298066

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a circle in the xy plane -/
structure Circle where
  center : Point3D
  radius : ℝ

/-- Represents a cone with circular base and vertex -/
structure Cone where
  base : Circle
  vertex : Point3D

/-- The largest possible radius of a sphere contained in a cone -/
def largestSphereRadius (cone : Cone) : ℝ :=
  sorry

theorem largest_sphere_in_cone :
  let c : Circle := { center := ⟨0, 0, 0⟩, radius := 1 }
  let p : Point3D := ⟨3, 4, 8⟩
  let cone : Cone := { base := c, vertex := p }
  largestSphereRadius cone = 3 - Real.sqrt 5 :=
by sorry

end largest_sphere_in_cone_l2980_298066


namespace extremum_implies_a_equals_e_l2980_298064

open Real

theorem extremum_implies_a_equals_e (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = exp x - a * x) →
  (∃ ε > 0, ∀ h ∈ Set.Ioo (-ε) ε, f (1 + h) ≤ f 1) ∨
  (∃ ε > 0, ∀ h ∈ Set.Ioo (-ε) ε, f (1 + h) ≥ f 1) →
  a = exp 1 := by
sorry

end extremum_implies_a_equals_e_l2980_298064


namespace shelby_gold_stars_yesterday_l2980_298007

/-- Proves that Shelby earned 4 gold stars yesterday -/
theorem shelby_gold_stars_yesterday (yesterday : ℕ) (today : ℕ) (total : ℕ)
  (h1 : today = 3)
  (h2 : total = 7)
  (h3 : yesterday + today = total) :
  yesterday = 4 := by
  sorry

end shelby_gold_stars_yesterday_l2980_298007


namespace product_expansion_l2980_298045

theorem product_expansion (x : ℝ) : 2 * (x + 3) * (x + 4) = 2 * x^2 + 14 * x + 24 := by
  sorry

end product_expansion_l2980_298045


namespace lemonade_cost_calculation_l2980_298019

/-- The cost of lemonade purchased by Coach Mike -/
def lemonade_cost : ℕ := sorry

/-- The amount Coach Mike gave to the girls -/
def amount_given : ℕ := 75

/-- The change Coach Mike received -/
def change_received : ℕ := 17

/-- Theorem stating that the lemonade cost is equal to the amount given minus the change received -/
theorem lemonade_cost_calculation : 
  lemonade_cost = amount_given - change_received := by sorry

end lemonade_cost_calculation_l2980_298019


namespace mistaken_divisor_l2980_298099

theorem mistaken_divisor (dividend : ℕ) (correct_divisor mistaken_divisor : ℕ) :
  correct_divisor = 21 →
  dividend = 36 * correct_divisor →
  dividend = 63 * mistaken_divisor →
  mistaken_divisor = 12 := by
sorry

end mistaken_divisor_l2980_298099


namespace log_y_equality_l2980_298014

theorem log_y_equality (y : ℝ) (h : y = (Real.log 3 / Real.log 4) ^ (Real.log 9 / Real.log 3)) :
  Real.log y / Real.log 2 = 2 * Real.log (Real.log 3 / Real.log 2) / Real.log 2 - 2 := by
  sorry

end log_y_equality_l2980_298014


namespace quiz_show_winning_probability_l2980_298027

def num_questions : ℕ := 4
def choices_per_question : ℕ := 3
def min_correct_to_win : ℕ := 3

def probability_of_correct_answer : ℚ := 1 / choices_per_question

/-- The probability of winning the quiz show -/
def probability_of_winning : ℚ :=
  (num_questions.choose min_correct_to_win) * (probability_of_correct_answer ^ min_correct_to_win) * ((1 - probability_of_correct_answer) ^ (num_questions - min_correct_to_win)) +
  (num_questions.choose (min_correct_to_win + 1)) * (probability_of_correct_answer ^ (min_correct_to_win + 1)) * ((1 - probability_of_correct_answer) ^ (num_questions - (min_correct_to_win + 1)))

theorem quiz_show_winning_probability :
  probability_of_winning = 1 / 9 := by
  sorry

end quiz_show_winning_probability_l2980_298027


namespace page_number_added_twice_l2980_298017

theorem page_number_added_twice (n : ℕ) (k : ℕ) : 
  k ≤ n →
  (n * (n + 1)) / 2 + k = 3050 →
  k = 47 :=
by sorry

end page_number_added_twice_l2980_298017


namespace number_divisible_by_six_l2980_298090

theorem number_divisible_by_six : ∃ n : ℕ, n % 6 = 0 ∧ n / 6 = 209 → n = 1254 := by
  sorry

end number_divisible_by_six_l2980_298090


namespace isosceles_triangle_area_theorem_l2980_298001

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  /-- The altitude to the base -/
  altitude : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The triangle is isosceles -/
  is_isosceles : True

/-- Calculate the area of an isosceles triangle -/
def triangle_area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area_theorem :
  ∀ (t : IsoscelesTriangle), t.altitude = 10 ∧ t.perimeter = 40 → triangle_area t = 75 :=
by sorry

end isosceles_triangle_area_theorem_l2980_298001


namespace sum_of_squares_of_roots_l2980_298096

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) 
  (h₁ : 6 * x₁^2 - 13 * x₁ + 5 = 0)
  (h₂ : 6 * x₂^2 - 13 * x₂ + 5 = 0)
  (h₃ : x₁ ≠ x₂) : 
  x₁^2 + x₂^2 = 109 / 36 := by
  sorry

end sum_of_squares_of_roots_l2980_298096


namespace marble_distribution_l2980_298074

theorem marble_distribution (total_marbles : ℕ) (group_size : ℕ) : 
  total_marbles = 220 →
  (total_marbles / group_size : ℚ) - 1 = (total_marbles / (group_size + 2) : ℚ) →
  group_size = 20 := by
sorry

end marble_distribution_l2980_298074


namespace periodic_decimal_as_fraction_l2980_298052

-- Define the periodic decimal expansion
def periodic_decimal : ℝ :=
  0.5123412341234123412341234123412341234

-- Theorem statement
theorem periodic_decimal_as_fraction :
  periodic_decimal = 51229 / 99990 := by
  sorry

end periodic_decimal_as_fraction_l2980_298052


namespace venkis_trip_speed_l2980_298037

/-- Venki's trip between towns X, Y, and Z -/
def venkis_trip (speed_xz speed_zy : ℝ) (time_xz time_zy : ℝ) : Prop :=
  let distance_xz := speed_xz * time_xz
  let distance_zy := distance_xz / 2
  speed_zy = distance_zy / time_zy

/-- The theorem statement for Venki's trip -/
theorem venkis_trip_speed :
  ∃ (speed_zy : ℝ),
    venkis_trip 80 speed_zy 5 (4 + 4/9) ∧
    abs (speed_zy - 42.86) < 0.01 := by
  sorry


end venkis_trip_speed_l2980_298037


namespace complex_fraction_equality_l2980_298030

theorem complex_fraction_equality (u v : ℂ) 
  (h : (u^3 + v^3) / (u^3 - v^3) + (u^3 - v^3) / (u^3 + v^3) = 2) :
  (u^9 + v^9) / (u^9 - v^9) + (u^9 - v^9) / (u^9 + v^9) = 2 := by
  sorry

end complex_fraction_equality_l2980_298030


namespace orthocenter_of_triangle_l2980_298055

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (2.2375, 2.675, 4.515). -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (1.5, 2, 3.5)
  let B : ℝ × ℝ × ℝ := (4, 3.5, 1)
  let C : ℝ × ℝ × ℝ := (3, 5, 4.5)
  orthocenter A B C = (2.2375, 2.675, 4.515) := by sorry

end orthocenter_of_triangle_l2980_298055


namespace calculate_speed_l2980_298029

/-- Given two people moving in opposite directions, calculate the unknown speed -/
theorem calculate_speed (known_speed time_minutes distance : ℝ) 
  (h1 : known_speed = 50)
  (h2 : time_minutes = 45)
  (h3 : distance = 60) : 
  ∃ unknown_speed : ℝ, 
    unknown_speed = 30 ∧ 
    (unknown_speed + known_speed) * (time_minutes / 60) = distance :=
by sorry

end calculate_speed_l2980_298029


namespace total_bottles_is_255_l2980_298042

-- Define the number of bottles in each box
def boxA_water : ℕ := 24
def boxA_orange : ℕ := 21
def boxA_apple : ℕ := boxA_water + 6

def boxB_water : ℕ := boxA_water + boxA_water / 4
def boxB_orange : ℕ := boxA_orange - boxA_orange * 3 / 10
def boxB_apple : ℕ := boxA_apple

def boxC_water : ℕ := 2 * boxB_water
def boxC_apple : ℕ := (3 * boxB_apple) / 2
def boxC_orange : ℕ := 0

-- Define the total number of bottles
def total_bottles : ℕ := 
  boxA_water + boxA_orange + boxA_apple + 
  boxB_water + boxB_orange + boxB_apple + 
  boxC_water + boxC_orange + boxC_apple

-- Theorem to prove
theorem total_bottles_is_255 : total_bottles = 255 := by
  sorry

end total_bottles_is_255_l2980_298042


namespace coin_array_problem_l2980_298060

/-- The number of coins in a triangular array -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem stating the problem -/
theorem coin_array_problem :
  ∃ (n : ℕ), triangular_sum n = 2211 ∧ sum_of_digits n = 12 :=
sorry

end coin_array_problem_l2980_298060


namespace least_number_of_cans_l2980_298082

theorem least_number_of_cans (maaza pepsi sprite : ℕ) 
  (h_maaza : maaza = 80)
  (h_pepsi : pepsi = 144)
  (h_sprite : sprite = 368) :
  let can_size := Nat.gcd maaza (Nat.gcd pepsi sprite)
  let total_cans := maaza / can_size + pepsi / can_size + sprite / can_size
  total_cans = 37 := by
  sorry

end least_number_of_cans_l2980_298082


namespace line_parallel_to_skew_line_l2980_298015

/-- Represents a line in 3D space -/
structure Line3D where
  -- Definition of a line in 3D space
  -- (We'll leave this abstract for simplicity)

/-- Two lines are skew if they are not coplanar -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- Two lines are parallel if they have the same direction -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Definition of parallel lines
  sorry

/-- Two lines intersect if they have a common point -/
def intersect (l1 l2 : Line3D) : Prop :=
  -- Definition of intersecting lines
  sorry

theorem line_parallel_to_skew_line (l1 l2 l3 : Line3D) 
  (h1 : are_skew l1 l2) 
  (h2 : are_parallel l3 l1) : 
  intersect l3 l2 ∨ are_skew l3 l2 :=
by
  sorry

end line_parallel_to_skew_line_l2980_298015


namespace no_valid_cube_labeling_l2980_298040

/-- A labeling of a cube's edges with 0s and 1s -/
def CubeLabeling := Fin 12 → Fin 2

/-- The set of edges for each face of a cube -/
def cube_faces : Fin 6 → Finset (Fin 12) := sorry

/-- The sum of labels on a face's edges -/
def face_sum (l : CubeLabeling) (face : Fin 6) : Nat :=
  (cube_faces face).sum (λ e => l e)

/-- A labeling is valid if the sum of labels on each face's edges equals 3 -/
def is_valid_labeling (l : CubeLabeling) : Prop :=
  ∀ face : Fin 6, face_sum l face = 3

theorem no_valid_cube_labeling :
  ¬ ∃ l : CubeLabeling, is_valid_labeling l := sorry

end no_valid_cube_labeling_l2980_298040


namespace sum_equals_power_of_two_l2980_298021

theorem sum_equals_power_of_two : 29 + 12 + 23 = 2^6 := by
  sorry

end sum_equals_power_of_two_l2980_298021


namespace least_multiple_21_greater_380_l2980_298077

theorem least_multiple_21_greater_380 : ∃ (n : ℕ), n * 21 = 399 ∧ 
  399 > 380 ∧ 
  (∀ m : ℕ, m * 21 > 380 → m * 21 ≥ 399) := by
  sorry

end least_multiple_21_greater_380_l2980_298077


namespace tangent_circle_intersection_theorem_l2980_298084

/-- A circle with center on y = 4x, tangent to x + y - 2 = 0 at (1,1) --/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.2 = 4 * center.1
  tangent_at_point : (1 : ℝ) + 1 - 2 = 0
  tangent_condition : (center.1 - 1)^2 + (center.2 - 1)^2 = radius^2

/-- The equation of the circle --/
def circle_equation (c : TangentCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The intersecting line --/
def intersecting_line (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 3 = 0

/-- Points A and B are on both the circle and the line --/
def intersection_points (c : TangentCircle) (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  circle_equation c A.1 A.2 ∧ circle_equation c B.1 B.2 ∧
  intersecting_line k A.1 A.2 ∧ intersecting_line k B.1 B.2

/-- Point M on the circle with OM = OA + OB --/
def point_M (c : TangentCircle) (A B M : ℝ × ℝ) : Prop :=
  circle_equation c M.1 M.2 ∧ M.1 = A.1 + B.1 ∧ M.2 = A.2 + B.2

/-- The main theorem --/
theorem tangent_circle_intersection_theorem (c : TangentCircle) 
  (k : ℝ) (A B M : ℝ × ℝ) :
  intersection_points c k A B → point_M c A B M → k^2 = 17 := by
  sorry

end tangent_circle_intersection_theorem_l2980_298084


namespace one_thirds_in_nine_thirds_l2980_298035

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by
  sorry

end one_thirds_in_nine_thirds_l2980_298035


namespace parabola_intersection_l2980_298091

theorem parabola_intersection :
  let f (x : ℝ) := 2 * x^2 + 5 * x - 3
  let g (x : ℝ) := x^2 + 8
  let x₁ := (-5 - Real.sqrt 69) / 2
  let x₂ := (-5 + Real.sqrt 69) / 2
  let y₁ := f x₁
  let y₂ := f x₂
  (∀ x, f x = g x ↔ x = x₁ ∨ x = x₂) ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ := by
  sorry

end parabola_intersection_l2980_298091


namespace restaurant_problem_solution_l2980_298078

def restaurant_problem (total_employees : ℕ) 
                       (family_buffet : ℕ) 
                       (dining_room : ℕ) 
                       (snack_bar : ℕ) 
                       (exactly_two : ℕ) : Prop :=
  let all_three : ℕ := total_employees + exactly_two - (family_buffet + dining_room + snack_bar)
  ∀ (e : ℕ), 1 ≤ e ∧ e ≤ 3 →
    total_employees = 39 ∧
    family_buffet = 17 ∧
    dining_room = 18 ∧
    snack_bar = 12 ∧
    exactly_two = 4 →
    all_three = 8

theorem restaurant_problem_solution : 
  restaurant_problem 39 17 18 12 4 :=
sorry

end restaurant_problem_solution_l2980_298078


namespace integer_root_implies_a_value_l2980_298063

theorem integer_root_implies_a_value (a : ℕ) : 
  (∃ x : ℤ, a^2 * x^2 - (3 * a^2 - 8 * a) * x + 2 * a^2 - 13 * a + 15 = 0) →
  (a = 1 ∨ a = 3 ∨ a = 5) :=
by sorry

end integer_root_implies_a_value_l2980_298063


namespace nickel_ate_two_chocolates_l2980_298083

-- Define the number of chocolates Robert ate
def robert_chocolates : ℕ := 9

-- Define the difference between Robert's and Nickel's chocolates
def chocolate_difference : ℕ := 7

-- Define Nickel's chocolates
def nickel_chocolates : ℕ := robert_chocolates - chocolate_difference

-- Theorem to prove
theorem nickel_ate_two_chocolates : nickel_chocolates = 2 := by
  sorry

end nickel_ate_two_chocolates_l2980_298083


namespace ellipse_dot_product_range_slope_product_constant_parallelogram_condition_l2980_298046

-- Define the ellipse
def ellipse (m : ℝ) (x y : ℝ) : Prop := 9 * x^2 + y^2 = m^2

-- Define the line
def line (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

-- Define the dot product
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem ellipse_dot_product_range :
  ∀ (x y : ℝ), ellipse 3 x y →
  ∃ (f1x f1y f2x f2y : ℝ),
    f1x = 0 ∧ f1y = 2 * Real.sqrt 2 ∧
    f2x = 0 ∧ f2y = -2 * Real.sqrt 2 ∧
    -7 ≤ dot_product (x - f1x) (y - f1y) (x - f2x) (y - f2y) ∧
    dot_product (x - f1x) (y - f1y) (x - f2x) (y - f2y) ≤ 1 :=
sorry

theorem slope_product_constant (m k b : ℝ) :
  k ≠ 0 → b ≠ 0 →
  ∃ (x1 y1 x2 y2 : ℝ),
    ellipse m x1 y1 ∧ ellipse m x2 y2 ∧
    line k b x1 y1 ∧ line k b x2 y2 ∧
    let x0 := (x1 + x2) / 2
    let y0 := (y1 + y2) / 2
    (y0 / x0) * k = -9 :=
sorry

theorem parallelogram_condition (m k : ℝ) :
  ellipse m (m/3) m →
  line k ((3-k)*m/3) (m/3) m →
  (∃ (x y : ℝ),
    ellipse m x y ∧
    line k ((3-k)*m/3) x y ∧
    x ≠ m/3 ∧ y ≠ m ∧
    (∃ (xp yp : ℝ),
      ellipse m xp yp ∧
      yp / xp = -9 / k ∧
      2 * (-(m - k*m/3)*k / (k^2 + 9)) = xp)) ↔
  (k = 4 + Real.sqrt 7 ∨ k = 4 - Real.sqrt 7) :=
sorry

end ellipse_dot_product_range_slope_product_constant_parallelogram_condition_l2980_298046


namespace roots_of_polynomial_l2980_298070

def p (x : ℝ) := x^3 - 2*x^2 - 5*x + 6

theorem roots_of_polynomial :
  ∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3 := by
  sorry

end roots_of_polynomial_l2980_298070


namespace cos_90_degrees_l2980_298047

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end cos_90_degrees_l2980_298047


namespace extended_triangle_similarity_l2980_298048

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the similarity of triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the extension of a line segment
def extend (p1 p2 : ℝ × ℝ) (length : ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem extended_triangle_similarity (ABC : Triangle) (P : ℝ × ℝ) :
  distance ABC.A ABC.B = 8 →
  distance ABC.B ABC.C = 7 →
  distance ABC.C ABC.A = 6 →
  P = extend ABC.B ABC.C (distance ABC.B P - 7) →
  similar (Triangle.mk P ABC.A ABC.B) (Triangle.mk P ABC.C ABC.A) →
  distance P ABC.C = 9 := by
  sorry

end extended_triangle_similarity_l2980_298048


namespace correct_transformation_l2980_298076

theorem correct_transformation (y : ℝ) : y + 2 = -3 → y = -5 := by
  sorry

end correct_transformation_l2980_298076


namespace projection_theorem_l2980_298086

/-- Given vectors a and b in R², prove that the projection of a onto b is -3/5 -/
theorem projection_theorem (a b : ℝ × ℝ) : 
  b = (3, 4) → (a.1 * b.1 + a.2 * b.2 = -3) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -3/5 := by
  sorry

end projection_theorem_l2980_298086


namespace relay_team_arrangements_l2980_298069

def RelayTeam := Fin 4 → Fin 4

def fixed_positions (team : RelayTeam) : Prop :=
  team 1 = 1 ∧ team 3 = 3

def valid_team (team : RelayTeam) : Prop :=
  Function.Injective team

theorem relay_team_arrangements :
  ∃ (n : ℕ), n = 2 ∧ 
  (∃ (teams : Finset RelayTeam), 
    (∀ t ∈ teams, fixed_positions t ∧ valid_team t) ∧
    teams.card = n) :=
sorry

end relay_team_arrangements_l2980_298069


namespace lines_perpendicular_to_plane_are_parallel_l2980_298023

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
by sorry

end lines_perpendicular_to_plane_are_parallel_l2980_298023


namespace infinite_points_in_S_l2980_298033

-- Define the set of points satisfying the conditions
def S : Set (ℚ × ℚ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 ≤ 5}

-- Theorem statement
theorem infinite_points_in_S : Set.Infinite S := by
  sorry

end infinite_points_in_S_l2980_298033


namespace fifteenth_digit_is_zero_l2980_298016

/-- The decimal representation of 1/8 -/
def frac_1_8 : ℚ := 1/8

/-- The decimal representation of 1/11 -/
def frac_1_11 : ℚ := 1/11

/-- The sum of the decimal representations of 1/8 and 1/11 -/
def sum_fracs : ℚ := frac_1_8 + frac_1_11

/-- The nth digit after the decimal point of a rational number -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem fifteenth_digit_is_zero :
  nth_digit_after_decimal sum_fracs 15 = 0 := by sorry

end fifteenth_digit_is_zero_l2980_298016


namespace abc_max_value_l2980_298056

/-- Given positive reals a, b, c satisfying the constraint b(a^2 + 2) + c(a + 2) = 12,
    the maximum value of abc is 3. -/
theorem abc_max_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (h_constraint : b * (a^2 + 2) + c * (a + 2) = 12) :
  a * b * c ≤ 3 := by
sorry

end abc_max_value_l2980_298056


namespace line_tangent_to_parabola_l2980_298009

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4 * x + 6 * y + k = 0 → y^2 = 16 * x) →
  (∃! p : ℝ × ℝ, (4 * p.1 + 6 * p.2 + k = 0) ∧ (p.2^2 = 16 * p.1)) →
  k = 36 :=
by sorry

end line_tangent_to_parabola_l2980_298009


namespace count_arith_seq_39_eq_12_l2980_298022

/-- An arithmetic sequence of positive integers containing 3 and 39 -/
structure ArithSeq39 where
  d : ℕ+  -- Common difference
  a : ℕ+  -- First term
  h1 : ∃ k : ℕ, a + k * d = 3
  h2 : ∃ m : ℕ, a + m * d = 39

/-- The count of arithmetic sequences containing 3 and 39 -/
def count_arith_seq_39 : ℕ := sorry

/-- Theorem: There are exactly 12 infinite arithmetic sequences of positive integers
    that contain both 3 and 39 -/
theorem count_arith_seq_39_eq_12 : count_arith_seq_39 = 12 := by sorry

end count_arith_seq_39_eq_12_l2980_298022


namespace smallest_angle_SQR_l2980_298062

-- Define the angles
def angle_PQR : ℝ := 40
def angle_PQS : ℝ := 28

-- Define the theorem
theorem smallest_angle_SQR : 
  let angle_SQR := angle_PQR - angle_PQS
  angle_SQR = 12 := by
  sorry

end smallest_angle_SQR_l2980_298062


namespace p_necessary_not_sufficient_l2980_298093

def p (x y : ℝ) : Prop := x > 1 ∨ y > 2

def q (x y : ℝ) : Prop := x + y > 3

theorem p_necessary_not_sufficient :
  (∀ x y : ℝ, q x y → p x y) ∧ 
  (∃ x y : ℝ, p x y ∧ ¬(q x y)) := by
  sorry

end p_necessary_not_sufficient_l2980_298093


namespace water_balloon_fight_l2980_298095

/-- The number of packs of neighbor's water balloons used in the water balloon fight -/
def neighbors_packs : ℕ := 2

/-- The number of their own water balloon packs used -/
def own_packs : ℕ := 3

/-- The number of balloons in each pack -/
def balloons_per_pack : ℕ := 6

/-- The number of extra balloons Milly takes -/
def extra_balloons : ℕ := 7

/-- The number of balloons Floretta is left with -/
def floretta_balloons : ℕ := 8

theorem water_balloon_fight :
  neighbors_packs = 2 ∧
  own_packs * balloons_per_pack + neighbors_packs * balloons_per_pack =
    2 * (floretta_balloons + extra_balloons) :=
by sorry

end water_balloon_fight_l2980_298095


namespace number_of_ways_to_assign_positions_l2980_298018

/-- The number of pavilions --/
def num_pavilions : ℕ := 4

/-- The total number of volunteers --/
def total_volunteers : ℕ := 5

/-- The number of ways A and B can independently choose positions --/
def ways_for_A_and_B : ℕ := num_pavilions * (num_pavilions - 1)

/-- The number of ways to distribute the remaining volunteers --/
def ways_for_remaining_volunteers : ℕ := 8

theorem number_of_ways_to_assign_positions : 
  ways_for_A_and_B * ways_for_remaining_volunteers = 96 := by
  sorry


end number_of_ways_to_assign_positions_l2980_298018


namespace circle_equation_l2980_298073

/-- The standard equation of a circle with center on y = 2x - 4 passing through (0, 0) and (2, 2) -/
theorem circle_equation :
  ∀ (h k : ℝ),
  (k = 2 * h - 4) →                          -- Center is on the line y = 2x - 4
  ((h - 0)^2 + (k - 0)^2 = (h - 2)^2 + (k - 2)^2) →  -- Equidistant from (0, 0) and (2, 2)
  (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = (h - 0)^2 + (k - 0)^2) →  -- Definition of circle
  (∀ (x y : ℝ), (x - 2)^2 + y^2 = 4) :=
by sorry

end circle_equation_l2980_298073


namespace triangle_existence_theorem_l2980_298087

def triangle_exists (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_x_values : Set ℕ :=
  {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

theorem triangle_existence_theorem :
  ∀ x : ℕ, x > 0 → (triangle_exists 6 15 x ↔ x ∈ valid_x_values) :=
by sorry

end triangle_existence_theorem_l2980_298087


namespace available_sandwich_kinds_l2980_298041

/-- The number of sandwich kinds initially available on the menu. -/
def initial_sandwich_kinds : ℕ := 9

/-- The number of sandwich kinds that were sold out. -/
def sold_out_sandwich_kinds : ℕ := 5

/-- Theorem stating that the number of currently available sandwich kinds is 4. -/
theorem available_sandwich_kinds : 
  initial_sandwich_kinds - sold_out_sandwich_kinds = 4 := by
  sorry

end available_sandwich_kinds_l2980_298041


namespace parallel_line_k_value_l2980_298097

/-- Given a line passing through points (3, -8) and (k, 20) that is parallel to the line 3x + 4y = 12, 
    the value of k is -103/3. -/
theorem parallel_line_k_value :
  ∀ (k : ℚ),
  (∃ (m b : ℚ), (∀ x y : ℚ, y = m * x + b → (x = 3 ∧ y = -8) ∨ (x = k ∧ y = 20)) ∧
                (m = -3/4)) →
  k = -103/3 := by
sorry

end parallel_line_k_value_l2980_298097


namespace double_inequality_solution_l2980_298011

theorem double_inequality_solution (x : ℝ) : 
  (0 < (x^2 - 8*x + 13) / (x^2 - 4*x + 7) ∧ (x^2 - 8*x + 13) / (x^2 - 4*x + 7) < 2) ↔ 
  (4 - Real.sqrt 17 < x ∧ x < 4 - Real.sqrt 3) ∨ (4 + Real.sqrt 3 < x ∧ x < 4 + Real.sqrt 17) :=
sorry

end double_inequality_solution_l2980_298011


namespace base_eight_representation_l2980_298050

theorem base_eight_representation : ∃ (a b : Nat), 
  a ≠ b ∧ 
  a < 8 ∧ 
  b < 8 ∧
  777 = a * 8^3 + b * 8^2 + b * 8^1 + a * 8^0 :=
by sorry

end base_eight_representation_l2980_298050


namespace locus_of_Q_l2980_298038

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/24 + y^2/16 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x/12 + y/8 = 1

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the intersection point R of OP and ellipse C
def point_R (x y : ℝ) : Prop := ellipse_C x y ∧ ∃ t : ℝ, x = t * (x - 0) ∧ y = t * (y - 0)

-- Define point Q on OP satisfying |OQ| * |OP| = |OR|^2
def point_Q (x y : ℝ) (xp yp xr yr : ℝ) : Prop :=
  ∃ t : ℝ, x = t * xp ∧ y = t * yp ∧ 
  (x^2 + y^2) * (xp^2 + yp^2) = (xr^2 + yr^2)^2

-- Theorem statement
theorem locus_of_Q (x y : ℝ) : 
  (∃ xp yp xr yr : ℝ, 
    point_P xp yp ∧ 
    point_R xr yr ∧ 
    point_Q x y xp yp xr yr) → 
  (x - 1)^2 / (5/2) + (y - 1)^2 / (5/3) = 1 :=
sorry

end locus_of_Q_l2980_298038


namespace circle_intersection_theorem_l2980_298025

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - 3*x = 0 ∧ 5/3 < x ∧ x ≤ 3

-- Define the line L
def L (k x y : ℝ) : Prop := y = k*(x - 4)

-- Theorem statement
theorem circle_intersection_theorem :
  -- Part 1: Center of the circle
  (∃! center : ℝ × ℝ, center.1 = 3/2 ∧ center.2 = 0 ∧
    ∀ x y : ℝ, C x y → (x - center.1)^2 + (y - center.2)^2 = (3/2)^2) ∧
  -- Part 2: Intersection conditions
  (∀ k : ℝ, (∃! p : ℝ × ℝ, C p.1 p.2 ∧ L k p.1 p.2) ↔
    k ∈ Set.Icc (-2*Real.sqrt 5/7) (2*Real.sqrt 5/7) ∪ {-3/4, 3/4}) :=
by sorry

end circle_intersection_theorem_l2980_298025


namespace t_cube_surface_area_l2980_298080

/-- Represents a T-shaped structure made of unit cubes -/
structure TCube where
  base_length : ℕ
  top_height : ℕ
  top_position : ℕ

/-- Calculates the surface area of a T-shaped structure -/
def surface_area (t : TCube) : ℕ :=
  sorry

/-- Theorem: The surface area of the specific T-shaped structure is 38 square units -/
theorem t_cube_surface_area :
  let t : TCube := ⟨7, 5, 3⟩
  surface_area t = 38 := by sorry

end t_cube_surface_area_l2980_298080


namespace perfect_square_sum_l2980_298053

theorem perfect_square_sum (n : ℤ) (h1 : n > 1) (h2 : ∃ x : ℤ, 3*n + 1 = x^2) :
  ∃ a b c : ℤ, n + 1 = a^2 + b^2 + c^2 := by
sorry

end perfect_square_sum_l2980_298053


namespace color_tv_price_l2980_298089

/-- The original price of a color TV before price changes --/
def original_price : ℝ := 2250

/-- The price increase percentage --/
def price_increase : ℝ := 0.4

/-- The discount percentage --/
def discount : ℝ := 0.2

/-- The additional profit per TV --/
def additional_profit : ℝ := 270

theorem color_tv_price : 
  (original_price * (1 + price_increase) * (1 - discount)) - original_price = additional_profit :=
by sorry

end color_tv_price_l2980_298089


namespace bruce_bags_theorem_l2980_298072

/-- Calculates the number of bags Bruce can buy with the change after purchasing crayons, books, and calculators. -/
def bags_bruce_can_buy (crayon_packs : ℕ) (crayon_price : ℕ) (books : ℕ) (book_price : ℕ) 
                       (calculators : ℕ) (calculator_price : ℕ) (initial_amount : ℕ) (bag_price : ℕ) : ℕ :=
  let total_cost := crayon_packs * crayon_price + books * book_price + calculators * calculator_price
  let change := initial_amount - total_cost
  change / bag_price

/-- Theorem stating that Bruce can buy 11 bags with the change. -/
theorem bruce_bags_theorem : 
  bags_bruce_can_buy 5 5 10 5 3 5 200 10 = 11 := by
  sorry

end bruce_bags_theorem_l2980_298072


namespace binomial_coefficient_ratio_l2980_298012

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 1 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 3 / 5 →
  n + k = 8 := by sorry

end binomial_coefficient_ratio_l2980_298012


namespace correct_multiplication_l2980_298039

theorem correct_multiplication (x : ℕ) : 
  987 * x = 559981 → 987 * x = 559989 := by
  sorry

end correct_multiplication_l2980_298039


namespace correspondence_theorem_l2980_298071

/-- Represents a correspondence between two people on a specific topic. -/
structure Correspondence (Person : Type) (Topic : Type) :=
  (person1 : Person)
  (person2 : Person)
  (topic : Topic)

/-- The main theorem to be proved. -/
theorem correspondence_theorem 
  (Person : Type) 
  [Fintype Person] 
  (Topic : Type) 
  [Fintype Topic] 
  (h_person_count : Fintype.card Person = 17)
  (h_topic_count : Fintype.card Topic = 3)
  (correspondence : Correspondence Person Topic)
  (h_all_correspond : ∀ (p1 p2 : Person), p1 ≠ p2 → ∃! t : Topic, correspondence.topic = t ∧ correspondence.person1 = p1 ∧ correspondence.person2 = p2) :
  ∃ (t : Topic) (p1 p2 p3 : Person), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    correspondence.topic = t ∧ 
    ((correspondence.person1 = p1 ∧ correspondence.person2 = p2) ∨ 
     (correspondence.person1 = p2 ∧ correspondence.person2 = p1)) ∧
    ((correspondence.person1 = p2 ∧ correspondence.person2 = p3) ∨ 
     (correspondence.person1 = p3 ∧ correspondence.person2 = p2)) ∧
    ((correspondence.person1 = p1 ∧ correspondence.person2 = p3) ∨ 
     (correspondence.person1 = p3 ∧ correspondence.person2 = p1)) :=
by sorry


end correspondence_theorem_l2980_298071


namespace beef_cabbage_cost_comparison_l2980_298006

/-- Represents the cost calculation for beef and spicy cabbage orders --/
theorem beef_cabbage_cost_comparison (a : ℝ) (h : a > 50) :
  (4500 + 27 * a) ≤ (4400 + 30 * a) := by
  sorry

#check beef_cabbage_cost_comparison

end beef_cabbage_cost_comparison_l2980_298006


namespace min_value_sum_l2980_298005

theorem min_value_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 1) : 
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a + 1/b + 1/c = 1 → x + 4*y + 9*z ≤ a + 4*b + 9*c :=
by sorry

end min_value_sum_l2980_298005
