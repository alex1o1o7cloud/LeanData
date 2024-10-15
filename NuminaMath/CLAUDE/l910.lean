import Mathlib

namespace NUMINAMATH_CALUDE_percentage_of_x_l910_91030

theorem percentage_of_x (x y : ℝ) (h1 : x / y = 4) (h2 : y ≠ 0) : (2 * x - y) / x = 175 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_x_l910_91030


namespace NUMINAMATH_CALUDE_first_number_problem_l910_91079

theorem first_number_problem (x y : ℤ) (h1 : y = 43) (h2 : x + 2 * y = 124) : x = 38 := by
  sorry

end NUMINAMATH_CALUDE_first_number_problem_l910_91079


namespace NUMINAMATH_CALUDE_lenyas_number_l910_91011

theorem lenyas_number (x : ℝ) : ((((x + 5) / 3) * 4) - 6) / 7 = 2 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_lenyas_number_l910_91011


namespace NUMINAMATH_CALUDE_factorization_equality_l910_91023

theorem factorization_equality (x : ℝ) : 4 * x^3 - x = x * (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l910_91023


namespace NUMINAMATH_CALUDE_proposition_q_false_l910_91093

open Real

theorem proposition_q_false (p q : Prop) 
  (hp : ¬ (∃ x : ℝ, (1/10)^(x-3) ≤ cos 2))
  (hpq : ¬((¬p) ∧ q)) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_proposition_q_false_l910_91093


namespace NUMINAMATH_CALUDE_system_solution_unique_l910_91034

theorem system_solution_unique :
  ∃! (x y : ℝ), x - y = 1 ∧ 2 * x + 3 * y = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l910_91034


namespace NUMINAMATH_CALUDE_triangle_midpoint_sum_l910_91005

theorem triangle_midpoint_sum (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_midpoint_sum_l910_91005


namespace NUMINAMATH_CALUDE_decimal_equals_base5_l910_91008

-- Define a function to convert a list of digits in base 5 to decimal
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 5 * acc) 0

-- Define the decimal number
def decimalNumber : Nat := 111

-- Define the base-5 representation as a list of digits
def base5Representation : List Nat := [4, 2, 1]

-- Theorem stating that the decimal number is equal to its base-5 representation
theorem decimal_equals_base5 : decimalNumber = base5ToDecimal base5Representation := by
  sorry

end NUMINAMATH_CALUDE_decimal_equals_base5_l910_91008


namespace NUMINAMATH_CALUDE_new_plan_cost_l910_91089

def old_plan_cost : ℝ := 150
def increase_percentage : ℝ := 0.3

theorem new_plan_cost : 
  old_plan_cost * (1 + increase_percentage) = 195 := by sorry

end NUMINAMATH_CALUDE_new_plan_cost_l910_91089


namespace NUMINAMATH_CALUDE_triangle_inequality_ratio_123_l910_91026

theorem triangle_inequality_ratio_123 :
  ∀ (x : ℝ), x > 0 → ¬(x + 2*x > 3*x) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_ratio_123_l910_91026


namespace NUMINAMATH_CALUDE_cookies_per_package_l910_91071

theorem cookies_per_package
  (num_friends : ℕ)
  (num_packages : ℕ)
  (cookies_per_child : ℕ)
  (h1 : num_friends = 4)
  (h2 : num_packages = 3)
  (h3 : cookies_per_child = 15) :
  (num_friends + 1) * cookies_per_child / num_packages = 25 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_package_l910_91071


namespace NUMINAMATH_CALUDE_train_speed_l910_91014

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 800 →
  crossing_time = 47.99616030717543 →
  man_speed_kmh = 5 →
  ∃ (train_speed : ℝ), abs (train_speed - 64.9848) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l910_91014


namespace NUMINAMATH_CALUDE_geometric_sequence_tenth_term_l910_91070

theorem geometric_sequence_tenth_term : 
  ∀ (a : ℚ) (r : ℚ),
    a = 5 →
    a * r = 20 / 3 →
    a * r^9 = 1310720 / 19683 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_tenth_term_l910_91070


namespace NUMINAMATH_CALUDE_cost_price_percentage_l910_91042

-- Define the profit percent
def profit_percent : ℝ := 25

-- Define the relationship between selling price (SP) and cost price (CP)
def selling_price_relation (CP SP : ℝ) : Prop :=
  SP = CP * (1 + profit_percent / 100)

-- Theorem statement
theorem cost_price_percentage (CP SP : ℝ) :
  selling_price_relation CP SP →
  CP / SP * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l910_91042


namespace NUMINAMATH_CALUDE_picnic_attendance_difference_picnic_attendance_difference_proof_l910_91004

/-- Proves that there are 80 more adults than children at a picnic -/
theorem picnic_attendance_difference : ℕ → Prop :=
  fun total_persons : ℕ =>
    ∀ (men women children adults : ℕ),
      total_persons = 240 →
      men = 120 →
      men = women + 80 →
      adults = men + women →
      total_persons = men + women + children →
      adults - children = 80

-- The proof is omitted
theorem picnic_attendance_difference_proof : picnic_attendance_difference 240 := by
  sorry

end NUMINAMATH_CALUDE_picnic_attendance_difference_picnic_attendance_difference_proof_l910_91004


namespace NUMINAMATH_CALUDE_cuboid_third_edge_length_l910_91049

/-- Given a cuboid with two edges of 4 cm and 5 cm, and a surface area of 148 cm², 
    the length of the third edge is 6 cm. -/
theorem cuboid_third_edge_length : 
  ∀ (x : ℝ), 
    (2 * (4 * 5 + 4 * x + 5 * x) = 148) → 
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_third_edge_length_l910_91049


namespace NUMINAMATH_CALUDE_clerical_staff_fraction_l910_91035

theorem clerical_staff_fraction (total_employees : ℕ) (f : ℚ) : 
  total_employees = 3600 →
  (2/3 : ℚ) * (f * total_employees) = (1/4 : ℚ) * (total_employees - (1/3 : ℚ) * (f * total_employees)) →
  f = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_clerical_staff_fraction_l910_91035


namespace NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l910_91000

theorem cos_pi_plus_2alpha (α : Real) (h : Real.sin (π / 2 + α) = 1 / 3) : 
  Real.cos (π + 2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l910_91000


namespace NUMINAMATH_CALUDE_additional_beads_needed_bella_needs_twelve_more_beads_l910_91025

/-- Given the number of friends, beads per bracelet, and beads Bella has,
    calculate the number of additional beads needed. -/
theorem additional_beads_needed 
  (num_friends : ℕ) 
  (beads_per_bracelet : ℕ) 
  (beads_bella_has : ℕ) : ℕ :=
  (num_friends * beads_per_bracelet) - beads_bella_has

/-- Prove that Bella needs 12 more beads to make bracelets for her friends. -/
theorem bella_needs_twelve_more_beads : 
  additional_beads_needed 6 8 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_beads_needed_bella_needs_twelve_more_beads_l910_91025


namespace NUMINAMATH_CALUDE_combined_distance_is_91261_136_l910_91003

/-- The combined distance traveled by friends in feet -/
def combined_distance : ℝ :=
  let mile_to_feet : ℝ := 5280
  let yard_to_feet : ℝ := 3
  let km_to_meter : ℝ := 1000
  let meter_to_feet : ℝ := 3.28084
  let lionel_miles : ℝ := 4
  let esther_yards : ℝ := 975
  let niklaus_feet : ℝ := 1287
  let isabella_km : ℝ := 18
  let sebastian_meters : ℝ := 2400
  lionel_miles * mile_to_feet +
  esther_yards * yard_to_feet +
  niklaus_feet +
  isabella_km * km_to_meter * meter_to_feet +
  sebastian_meters * meter_to_feet

/-- Theorem stating that the combined distance traveled by friends is 91261.136 feet -/
theorem combined_distance_is_91261_136 : combined_distance = 91261.136 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_is_91261_136_l910_91003


namespace NUMINAMATH_CALUDE_selection_ways_eq_756_l910_91048

/-- The number of ways to select 5 people from a group of 12, 
    where at most 2 out of 3 specific people can be selected -/
def selection_ways : ℕ :=
  Nat.choose 9 5 + 
  (Nat.choose 3 1 * Nat.choose 9 4) + 
  (Nat.choose 3 2 * Nat.choose 9 3)

/-- Theorem stating that the number of selection ways is 756 -/
theorem selection_ways_eq_756 : selection_ways = 756 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_eq_756_l910_91048


namespace NUMINAMATH_CALUDE_power_eight_divided_by_four_l910_91036

theorem power_eight_divided_by_four (n : ℕ) : n = 8^2022 → n/4 = 4^3032 := by
  sorry

end NUMINAMATH_CALUDE_power_eight_divided_by_four_l910_91036


namespace NUMINAMATH_CALUDE_alfonso_savings_l910_91099

theorem alfonso_savings (daily_rate : ℕ) (days_per_week : ℕ) (total_weeks : ℕ) (helmet_cost : ℕ) :
  let total_earned := daily_rate * days_per_week * total_weeks
  helmet_cost - total_earned = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_alfonso_savings_l910_91099


namespace NUMINAMATH_CALUDE_rachel_homework_difference_l910_91045

theorem rachel_homework_difference (math_pages reading_pages : ℕ) 
  (h1 : math_pages = 7) 
  (h2 : reading_pages = 3) : 
  math_pages - reading_pages = 4 := by
sorry

end NUMINAMATH_CALUDE_rachel_homework_difference_l910_91045


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l910_91010

-- Define a geometric sequence
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometric a →
  a 1 + a 2 + a 3 = 7 →
  a 2 + a 3 + a 4 = 14 →
  a 4 + a 5 + a 6 = 56 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l910_91010


namespace NUMINAMATH_CALUDE_divisibility_probability_l910_91077

def is_divisible (r k : ℤ) : Prop := ∃ m : ℤ, r = k * m

def count_divisible_pairs : ℕ := 30

def total_pairs : ℕ := 88

theorem divisibility_probability :
  (count_divisible_pairs : ℚ) / (total_pairs : ℚ) = 15 / 44 := by sorry

end NUMINAMATH_CALUDE_divisibility_probability_l910_91077


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l910_91012

theorem cookie_boxes_problem (n : ℕ) : 
  (n - 7 ≥ 1) →  -- Mark sold at least one box
  (n - 2 ≥ 1) →  -- Ann sold at least one box
  (n - 3 ≥ 1) →  -- Carol sold at least one box
  ((n - 7) + (n - 2) + (n - 3) < n) →  -- Together they sold less than n boxes
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l910_91012


namespace NUMINAMATH_CALUDE_evaluate_expression_l910_91015

theorem evaluate_expression (a : ℚ) (h : a = 4/3) : (6*a^2 - 11*a + 2)*(3*a - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l910_91015


namespace NUMINAMATH_CALUDE_train_distance_l910_91057

/-- Given a train traveling at a certain speed for a certain time, 
    calculate the distance covered. -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
    (h1 : speed = 150) 
    (h2 : time = 8) 
    (h3 : distance = speed * time) : 
  distance = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l910_91057


namespace NUMINAMATH_CALUDE_parabola_directrix_l910_91068

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y = 3 * x^2 - 6 * x + 1

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop :=
  y = -25 / 12

/-- Theorem stating that the given directrix equation is correct for the parabola -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola_equation x y → ∃ (d : ℝ), directrix_equation d ∧ 
  (d = y - 1 / (4 * 3) - (y - 1 / (4 * 3) - (-2 - 1 / (4 * 3)))) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l910_91068


namespace NUMINAMATH_CALUDE_teresas_pencil_sharing_l910_91052

/-- Proves that each sibling receives 13 pencils given the conditions of Teresa's pencil sharing problem -/
theorem teresas_pencil_sharing :
  -- Define the given conditions
  let total_pencils : ℕ := 14 + 35
  let pencils_to_keep : ℕ := 10
  let num_siblings : ℕ := 3
  let pencils_to_share : ℕ := total_pencils - pencils_to_keep
  -- Define the theorem
  pencils_to_share / num_siblings = 13 := by
  sorry

end NUMINAMATH_CALUDE_teresas_pencil_sharing_l910_91052


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l910_91084

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem intersection_equality_implies_a_range (a : ℝ) :
  A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l910_91084


namespace NUMINAMATH_CALUDE_prob_even_sum_two_balls_l910_91069

def num_balls : ℕ := 20

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem prob_even_sum_two_balls :
  let total_outcomes := num_balls * (num_balls - 1)
  let favorable_outcomes := (num_balls / 2) * (num_balls / 2 - 1) * 2
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 19 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_two_balls_l910_91069


namespace NUMINAMATH_CALUDE_circle_op_difference_l910_91072

/-- The custom operation ⊙ for three natural numbers -/
def circle_op (a b c : ℕ) : ℕ :=
  (a * b) * 100 + (b * c)

/-- Theorem stating the result of the calculation -/
theorem circle_op_difference : circle_op 5 7 4 - circle_op 7 4 5 = 708 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_difference_l910_91072


namespace NUMINAMATH_CALUDE_age_difference_l910_91064

/-- Given the ages of four individuals x, y, z, and w, prove that z is 1.2 decades younger than x. -/
theorem age_difference (x y z w : ℕ) : 
  (x + y = y + z + 12) → 
  (x + y + w = y + z + w + 12) → 
  (x : ℚ) - z = 12 ∧ (x - z : ℚ) / 10 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l910_91064


namespace NUMINAMATH_CALUDE_smallest_value_z_plus_i_l910_91085

theorem smallest_value_z_plus_i (z : ℂ) (h : Complex.abs (z^2 + 4) = Complex.abs (z * (z + 2*I))) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ (w : ℂ), Complex.abs (w^2 + 4) = Complex.abs (w * (w + 2*I)) →
    Complex.abs (w + I) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_z_plus_i_l910_91085


namespace NUMINAMATH_CALUDE_johns_age_fraction_l910_91037

theorem johns_age_fraction (john_age mother_age father_age : ℕ) : 
  father_age = 40 →
  father_age = mother_age + 4 →
  john_age = mother_age - 16 →
  (john_age : ℚ) / father_age = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_johns_age_fraction_l910_91037


namespace NUMINAMATH_CALUDE_probability_nine_red_in_eleven_draws_l910_91024

/-- The probability of drawing exactly 9 red balls in 11 draws, with the 11th draw being red,
    from a bag containing 6 white balls and 3 red balls (with replacement) -/
theorem probability_nine_red_in_eleven_draws :
  let total_balls : ℕ := 9
  let red_balls : ℕ := 3
  let white_balls : ℕ := 6
  let total_draws : ℕ := 11
  let red_draws : ℕ := 9
  let p_red : ℚ := red_balls / total_balls
  let p_white : ℚ := white_balls / total_balls
  Nat.choose (total_draws - 1) (red_draws - 1) * p_red ^ red_draws * p_white ^ (total_draws - red_draws) =
    Nat.choose 10 8 * (1 / 3) ^ 9 * (2 / 3) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_nine_red_in_eleven_draws_l910_91024


namespace NUMINAMATH_CALUDE_range_of_a_l910_91098

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l910_91098


namespace NUMINAMATH_CALUDE_relay_for_life_total_miles_l910_91094

/-- Calculates the total miles walked in a relay event -/
def total_miles_walked (john_speed bob_speed alice_speed : ℝ) 
                       (john_time bob_time alice_time : ℝ) : ℝ :=
  john_speed * john_time + alice_speed * alice_time + bob_speed * bob_time

/-- The combined total miles walked by John, Alice, and Bob during the Relay for Life event -/
theorem relay_for_life_total_miles : 
  total_miles_walked 3.5 4 2.8 4 6 8 = 62.8 := by
  sorry

end NUMINAMATH_CALUDE_relay_for_life_total_miles_l910_91094


namespace NUMINAMATH_CALUDE_bicyclist_speed_increase_l910_91029

theorem bicyclist_speed_increase (x : ℝ) : 
  (1 + x) * 1.1 = 1.43 → x = 0.3 := by sorry

end NUMINAMATH_CALUDE_bicyclist_speed_increase_l910_91029


namespace NUMINAMATH_CALUDE_family_average_age_l910_91087

theorem family_average_age 
  (n : ℕ) 
  (youngest_age : ℕ) 
  (past_average : ℚ) : 
  n = 7 → 
  youngest_age = 5 → 
  past_average = 28 → 
  (((n - 1) * past_average + (n - 1) * youngest_age + youngest_age) / n : ℚ) = 209/7 := by
  sorry

end NUMINAMATH_CALUDE_family_average_age_l910_91087


namespace NUMINAMATH_CALUDE_locus_of_T_l910_91001

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point P
def P : ℝ × ℝ := (1, 0)

-- Define vertices A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a point on the ellipse that is not A or B
def M (x y : ℝ) : Prop := ellipse x y ∧ (x, y) ≠ A ∧ (x, y) ≠ B

-- Define point N as the intersection of MP and the ellipse
def N (x y : ℝ) : Prop := 
  M x y → ∃ t : ℝ, ellipse (x + t * (1 - x)) (y + t * (-y)) ∧ t ≠ 0

-- Define point T as the intersection of AM and BN
def T (x y : ℝ) : Prop :=
  ∃ (xm ym : ℝ), M xm ym ∧
  ∃ (xn yn : ℝ), N xn yn ∧
  (y / (x + 2) = ym / (xm + 2)) ∧
  (y / (x - 2) = yn / (xn - 2))

-- Theorem statement
theorem locus_of_T : ∀ x y : ℝ, T x y → y ≠ 0 → x = 4 := by sorry

end NUMINAMATH_CALUDE_locus_of_T_l910_91001


namespace NUMINAMATH_CALUDE_fraction_equality_l910_91066

theorem fraction_equality (a b : ℝ) : - (a / (b - a)) = a / (a - b) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l910_91066


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l910_91055

theorem arithmetic_calculation : 6 * 100000 + 8 * 1000 + 6 * 100 + 7 * 1 = 608607 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l910_91055


namespace NUMINAMATH_CALUDE_max_visible_sum_three_cubes_l910_91044

/-- Represents a cube with six faces numbered 1, 3, 5, 7, 9, 11 -/
def Cube := Fin 6 → Nat

/-- The set of numbers on each cube -/
def cubeNumbers : Finset Nat := {1, 3, 5, 7, 9, 11}

/-- A function to calculate the sum of visible faces when stacking cubes -/
def visibleSum (c1 c2 c3 : Cube) : Nat := sorry

/-- The theorem stating the maximum visible sum when stacking three cubes -/
theorem max_visible_sum_three_cubes :
  ∃ (c1 c2 c3 : Cube),
    (∀ (i : Fin 6), c1 i ∈ cubeNumbers ∧ c2 i ∈ cubeNumbers ∧ c3 i ∈ cubeNumbers) ∧
    (∀ (c1' c2' c3' : Cube),
      (∀ (i : Fin 6), c1' i ∈ cubeNumbers ∧ c2' i ∈ cubeNumbers ∧ c3' i ∈ cubeNumbers) →
      visibleSum c1' c2' c3' ≤ visibleSum c1 c2 c3) ∧
    visibleSum c1 c2 c3 = 101 :=
  sorry

end NUMINAMATH_CALUDE_max_visible_sum_three_cubes_l910_91044


namespace NUMINAMATH_CALUDE_number_of_divisors_of_60_l910_91096

theorem number_of_divisors_of_60 : Nat.card {d : Nat | d > 0 ∧ 60 % d = 0} = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_60_l910_91096


namespace NUMINAMATH_CALUDE_william_tickets_l910_91095

/-- William's ticket problem -/
theorem william_tickets : ∀ (initial additional : ℕ), 
  initial = 15 → additional = 3 → initial + additional = 18 := by
  sorry

end NUMINAMATH_CALUDE_william_tickets_l910_91095


namespace NUMINAMATH_CALUDE_latest_departure_time_correct_l910_91031

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiffMinutes (t1 t2 : Time) : Nat :=
  (t1.hours - t2.hours) * 60 + (t1.minutes - t2.minutes)

/-- The flight departure time -/
def flightTime : Time := { hours := 20, minutes := 0, valid := by simp }

/-- The recommended check-in time in minutes -/
def checkInTime : Nat := 120

/-- The time needed to drive to the airport in minutes -/
def driveTime : Nat := 45

/-- The time needed to park and reach the terminal in minutes -/
def parkAndWalkTime : Nat := 15

/-- The latest time they can leave their house -/
def latestDepartureTime : Time := { hours := 17, minutes := 0, valid := by simp }

theorem latest_departure_time_correct :
  timeDiffMinutes flightTime latestDepartureTime = checkInTime + driveTime + parkAndWalkTime :=
sorry

end NUMINAMATH_CALUDE_latest_departure_time_correct_l910_91031


namespace NUMINAMATH_CALUDE_problem_solution_l910_91073

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) := Real.log x + a
noncomputable def h (x : ℝ) := x * f x

theorem problem_solution :
  (∃ (x_min : ℝ), ∀ (x : ℝ), h x ≥ h x_min ∧ h x_min = -1 / Real.exp 1) ∧
  (∀ (a : ℝ), (∃! (p : ℝ), f p = g a p) →
    (∃ (p : ℝ), f p = g a p ∧
      (deriv f p : ℝ) = (deriv (g a) p : ℝ) ∧
      2 < a ∧ a < 5/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l910_91073


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l910_91006

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x => Real.sin (x + Real.pi / 18) + Real.cos (x - Real.pi / 9)
  ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l910_91006


namespace NUMINAMATH_CALUDE_fraction_simplification_l910_91051

theorem fraction_simplification : (4 * 5) / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l910_91051


namespace NUMINAMATH_CALUDE_tv_production_theorem_l910_91075

/-- Represents the daily TV production in a factory for a month -/
structure TVProduction where
  totalDays : Nat
  firstPeriodDays : Nat
  firstPeriodAvg : Nat
  monthlyAvg : Nat

/-- Calculates the average daily production for the last period of the month -/
def lastPeriodAvg (p : TVProduction) : Nat :=
  let lastPeriodDays := p.totalDays - p.firstPeriodDays
  let totalProduction := p.totalDays * p.monthlyAvg
  let firstPeriodProduction := p.firstPeriodDays * p.firstPeriodAvg
  (totalProduction - firstPeriodProduction) / lastPeriodDays

theorem tv_production_theorem (p : TVProduction) 
  (h1 : p.totalDays = 30)
  (h2 : p.firstPeriodDays = 25)
  (h3 : p.firstPeriodAvg = 65)
  (h4 : p.monthlyAvg = 60) :
  lastPeriodAvg p = 35 := by
  sorry

#eval lastPeriodAvg ⟨30, 25, 65, 60⟩

end NUMINAMATH_CALUDE_tv_production_theorem_l910_91075


namespace NUMINAMATH_CALUDE_trigonometric_ratio_proof_l910_91016

theorem trigonometric_ratio_proof (α : Real) 
  (h : ∃ (x y : Real), x = 3/5 ∧ y = 4/5 ∧ x^2 + y^2 = 1 ∧ x = Real.cos α ∧ y = Real.sin α) : 
  (Real.cos (2*α)) / (1 + Real.sin (2*α)) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_ratio_proof_l910_91016


namespace NUMINAMATH_CALUDE_pearl_cutting_theorem_l910_91039

/-- Represents a string of pearls -/
structure PearlString where
  color : Bool  -- true for black, false for white
  length : Nat
  length_pos : length > 0

/-- The state of the pearl-cutting process -/
structure PearlState where
  strings : List PearlString
  step : Nat

/-- The rules for cutting pearls -/
def cut_pearls (k : Nat) (state : PearlState) : PearlState :=
  sorry

/-- Predicate to check if a white pearl is isolated -/
def has_isolated_white_pearl (state : PearlState) : Prop :=
  sorry

/-- Predicate to check if there's a string of at least two black pearls -/
def has_two_or_more_black_pearls (state : PearlState) : Prop :=
  sorry

/-- The main theorem -/
theorem pearl_cutting_theorem (k b w : Nat) (h1 : k > 0) (h2 : b > w) (h3 : w > 1) :
  ∀ (final_state : PearlState),
    (∃ (initial_state : PearlState),
      initial_state.strings = [PearlString.mk true b sorry, PearlString.mk false w sorry] ∧
      final_state = cut_pearls k initial_state) →
    has_isolated_white_pearl final_state →
    has_two_or_more_black_pearls final_state :=
  sorry

end NUMINAMATH_CALUDE_pearl_cutting_theorem_l910_91039


namespace NUMINAMATH_CALUDE_unique_solution_ab_minus_a_minus_b_equals_one_l910_91062

theorem unique_solution_ab_minus_a_minus_b_equals_one :
  ∀ a b : ℤ, a > b ∧ b > 0 ∧ a * b - a - b = 1 → a = 3 ∧ b = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_ab_minus_a_minus_b_equals_one_l910_91062


namespace NUMINAMATH_CALUDE_injective_function_property_l910_91082

theorem injective_function_property {A : Type*} (f : A → A) (h : Function.Injective f) :
  ∀ (x₁ x₂ : A), x₁ ≠ x₂ → f x₁ ≠ f x₂ := by
  sorry

end NUMINAMATH_CALUDE_injective_function_property_l910_91082


namespace NUMINAMATH_CALUDE_f_symmetry_l910_91058

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + Real.pi^2 * x^2) - Real.pi * x) + Real.pi

theorem f_symmetry (m : ℝ) : f m = 3 → f (-m) = 2 * Real.pi - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l910_91058


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_odds_l910_91020

theorem largest_of_five_consecutive_odds (n : ℤ) : 
  (n % 2 = 1) → 
  (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) = 93555) → 
  (n + 8 = 19) := by
  sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_odds_l910_91020


namespace NUMINAMATH_CALUDE_inequality_solution_set_l910_91086

theorem inequality_solution_set :
  ∀ x : ℝ, (x / 4 - 1 ≤ 3 + x ∧ 3 + x < 1 - 3 * (2 + x)) ↔ x ∈ Set.Icc (-16/3) (-2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l910_91086


namespace NUMINAMATH_CALUDE_smallest_n_for_given_mean_l910_91091

theorem smallest_n_for_given_mean : ∃ (n : ℕ) (m : ℕ),
  n > 0 ∧
  m ∈ Finset.range (n + 1) ∧
  (Finset.sum (Finset.range (n + 1) \ {m}) id) / ((n : ℚ) - 1) = 439 / 13 ∧
  ∀ (k : ℕ) (j : ℕ), k > 0 ∧ k < n →
    j ∈ Finset.range (k + 1) →
    (Finset.sum (Finset.range (k + 1) \ {j}) id) / ((k : ℚ) - 1) ≠ 439 / 13 ∧
  n = 68 ∧
  m = 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_given_mean_l910_91091


namespace NUMINAMATH_CALUDE_max_square_plots_l910_91088

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℕ
  width : ℕ

/-- Represents the available fencing and field dimensions -/
structure FencingProblem where
  field : FieldDimensions
  available_fencing : ℕ

/-- Calculates the number of square plots given the side length -/
def num_plots (f : FieldDimensions) (side : ℕ) : ℕ :=
  (f.length / side) * (f.width / side)

/-- Calculates the required internal fencing given the side length -/
def required_fencing (f : FieldDimensions) (side : ℕ) : ℕ :=
  f.length * ((f.width / side) - 1) + f.width * ((f.length / side) - 1)

/-- Theorem: The maximum number of square plots is 18 -/
theorem max_square_plots (p : FencingProblem) 
  (h1 : p.field.length = 30)
  (h2 : p.field.width = 60)
  (h3 : p.available_fencing = 2500) :
  ∃ (side : ℕ), 
    side > 0 ∧ 
    side ∣ p.field.length ∧ 
    side ∣ p.field.width ∧
    required_fencing p.field side ≤ p.available_fencing ∧
    num_plots p.field side = 18 ∧
    ∀ (other_side : ℕ), other_side > side → 
      ¬(other_side ∣ p.field.length ∧ 
        other_side ∣ p.field.width ∧
        required_fencing p.field other_side ≤ p.available_fencing) :=
  sorry

end NUMINAMATH_CALUDE_max_square_plots_l910_91088


namespace NUMINAMATH_CALUDE_divisibility_property_l910_91017

theorem divisibility_property (m : ℕ) (hm : m > 0) :
  ∃ q : Polynomial ℤ, (x + 1)^(2*m) - x^(2*m) - 2*x - 1 = x * (x + 1) * (2*x + 1) * q :=
sorry

end NUMINAMATH_CALUDE_divisibility_property_l910_91017


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l910_91065

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem twelfth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 5/6) (h₃ : a₃ = 7/6) :
  arithmetic_sequence a₁ (a₂ - a₁) 12 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l910_91065


namespace NUMINAMATH_CALUDE_family_average_age_l910_91047

theorem family_average_age
  (num_members : ℕ)
  (youngest_age : ℕ)
  (birth_average_age : ℚ)
  (h1 : num_members = 5)
  (h2 : youngest_age = 10)
  (h3 : birth_average_age = 12.5) :
  (birth_average_age * (num_members - 1) + youngest_age * num_members) / num_members = 20 :=
by sorry

end NUMINAMATH_CALUDE_family_average_age_l910_91047


namespace NUMINAMATH_CALUDE_smallest_solution_and_ratio_l910_91041

theorem smallest_solution_and_ratio (x : ℝ) (a b c d : ℤ) : 
  (7 * x / 8 - 1 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (x ≥ (4 - 4 * Real.sqrt 15) / 7) →
  (x = (4 - 4 * Real.sqrt 15) / 7 → a * c * d / b = -105) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_and_ratio_l910_91041


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l910_91043

theorem square_circle_area_ratio (a r : ℝ) (h : a > 0) (k : r > 0) : 
  4 * a = 2 * 2 * Real.pi * r → a^2 / (Real.pi * r^2) = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l910_91043


namespace NUMINAMATH_CALUDE_min_colors_theorem_l910_91019

theorem min_colors_theorem : ∃ (f : Fin 2013 → Fin 3), 
  (∀ i j : Fin 2013, f i = f j → ¬(((i.val + 1) * (j.val + 1)) % 2014 = 0)) ∧
  (∀ n : ℕ, n < 3 → ¬∃ (g : Fin 2013 → Fin n), 
    ∀ i j : Fin 2013, g i = g j → ¬(((i.val + 1) * (j.val + 1)) % 2014 = 0)) :=
sorry

end NUMINAMATH_CALUDE_min_colors_theorem_l910_91019


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_less_than_three_l910_91021

theorem inequality_solution_implies_a_less_than_three (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| > a) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_less_than_three_l910_91021


namespace NUMINAMATH_CALUDE_solve_equation_l910_91078

theorem solve_equation (y : ℝ) (h : (2 * y) / 3 = 12) : y = 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l910_91078


namespace NUMINAMATH_CALUDE_max_display_sum_l910_91056

def hour_sum (h : Nat) : Nat :=
  if h < 10 then h
  else if h < 20 then (h / 10) + (h % 10)
  else 2 + (h % 10)

def minute_sum (m : Nat) : Nat :=
  (m / 10) + (m % 10)

def display_sum (h m : Nat) : Nat :=
  hour_sum h + minute_sum m

theorem max_display_sum :
  (∀ h m, h < 24 → m < 60 → display_sum h m ≤ 24) ∧
  (∃ h m, h < 24 ∧ m < 60 ∧ display_sum h m = 24) :=
sorry

end NUMINAMATH_CALUDE_max_display_sum_l910_91056


namespace NUMINAMATH_CALUDE_daily_servings_sold_l910_91097

theorem daily_servings_sold (cost profit_A profit_B revenue total_profit : ℚ)
  (h1 : cost = 14)
  (h2 : profit_A = 20)
  (h3 : profit_B = 18)
  (h4 : revenue = 1120)
  (h5 : total_profit = 280) :
  ∃ (x y : ℚ), x + y = 60 ∧ 
    profit_A * x + profit_B * y = revenue ∧
    (profit_A - cost) * x + (profit_B - cost) * y = total_profit :=
by sorry

end NUMINAMATH_CALUDE_daily_servings_sold_l910_91097


namespace NUMINAMATH_CALUDE_square_sum_equals_48_l910_91032

theorem square_sum_equals_48 (x y : ℝ) (h1 : x - 2*y = 4) (h2 : x*y = 8) : x^2 + 4*y^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_48_l910_91032


namespace NUMINAMATH_CALUDE_weighted_average_plants_per_hour_l910_91054

def total_rows : ℕ := 400
def carrot_rows : ℕ := 250
def potato_rows : ℕ := 150

def carrot_first_rows : ℕ := 100
def carrot_first_plants_per_row : ℕ := 275
def carrot_first_time : ℕ := 10

def carrot_remaining_rows : ℕ := 150
def carrot_remaining_plants_per_row : ℕ := 325
def carrot_remaining_time : ℕ := 20

def potato_first_rows : ℕ := 50
def potato_first_plants_per_row : ℕ := 300
def potato_first_time : ℕ := 12

def potato_remaining_rows : ℕ := 100
def potato_remaining_plants_per_row : ℕ := 400
def potato_remaining_time : ℕ := 18

theorem weighted_average_plants_per_hour :
  let total_plants := 
    (carrot_first_rows * carrot_first_plants_per_row + 
     carrot_remaining_rows * carrot_remaining_plants_per_row +
     potato_first_rows * potato_first_plants_per_row + 
     potato_remaining_rows * potato_remaining_plants_per_row)
  let total_time := 
    (carrot_first_time + carrot_remaining_time + 
     potato_first_time + potato_remaining_time)
  (total_plants : ℚ) / total_time = 2187.5 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_plants_per_hour_l910_91054


namespace NUMINAMATH_CALUDE_decimal_to_binary_27_l910_91018

theorem decimal_to_binary_27 : 
  (27 : ℕ).digits 2 = [1, 1, 0, 1, 1] := by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_27_l910_91018


namespace NUMINAMATH_CALUDE_ababab_no_large_prime_factors_l910_91060

theorem ababab_no_large_prime_factors (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  ∀ p : ℕ, p.Prime → p ∣ (101010 * a + 10101 * b) → p ≤ 99 := by
  sorry

end NUMINAMATH_CALUDE_ababab_no_large_prime_factors_l910_91060


namespace NUMINAMATH_CALUDE_contaminated_constant_l910_91028

theorem contaminated_constant (x : ℝ) (h : 2 * (x - 3) - 2 = x + 1) (h_sol : x = 9) : 2 * (9 - 3) - (9 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_contaminated_constant_l910_91028


namespace NUMINAMATH_CALUDE_probability_two_first_grade_pens_l910_91067

/-- The probability of selecting 2 first-grade pens from a box of 6 pens, where 3 are first-grade -/
theorem probability_two_first_grade_pens (total_pens : ℕ) (first_grade_pens : ℕ) 
  (h1 : total_pens = 6) (h2 : first_grade_pens = 3) : 
  (Nat.choose first_grade_pens 2 : ℚ) / (Nat.choose total_pens 2) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_first_grade_pens_l910_91067


namespace NUMINAMATH_CALUDE_portias_school_students_l910_91046

theorem portias_school_students (portia_students lara_students : ℕ) : 
  portia_students = 2 * lara_students →
  portia_students + lara_students = 3000 →
  portia_students = 2000 := by
  sorry

end NUMINAMATH_CALUDE_portias_school_students_l910_91046


namespace NUMINAMATH_CALUDE_smallest_c_inequality_l910_91033

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → Real.sqrt (x^2 + y^2) + c * |x - y| ≥ (x + y) / 2) ↔ c ≥ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_smallest_c_inequality_l910_91033


namespace NUMINAMATH_CALUDE_math_city_intersections_l910_91080

/-- Represents a street in Math City -/
structure Street where
  curved : Bool

/-- Represents Math City -/
structure MathCity where
  streets : Finset Street
  no_parallel : True  -- Assumption that no streets are parallel
  curved_count : Nat
  curved_additional_intersections : Nat

/-- Calculates the maximum number of intersections in Math City -/
def max_intersections (city : MathCity) : Nat :=
  let basic_intersections := city.streets.card.choose 2
  let additional_intersections := city.curved_count * city.curved_additional_intersections
  basic_intersections + additional_intersections

/-- Theorem stating the maximum number of intersections in the given scenario -/
theorem math_city_intersections :
  ∀ (city : MathCity),
    city.streets.card = 10 →
    city.curved_count = 2 →
    city.curved_additional_intersections = 3 →
    max_intersections city = 51 := by
  sorry

end NUMINAMATH_CALUDE_math_city_intersections_l910_91080


namespace NUMINAMATH_CALUDE_spam_email_ratio_l910_91022

theorem spam_email_ratio (total : ℕ) (important : ℕ) (promotional_fraction : ℚ) 
  (h1 : total = 400)
  (h2 : important = 180)
  (h3 : promotional_fraction = 2/5) :
  (total - important - (total - important) * promotional_fraction : ℚ) / total = 33/100 := by
  sorry

end NUMINAMATH_CALUDE_spam_email_ratio_l910_91022


namespace NUMINAMATH_CALUDE_simple_interest_problem_l910_91074

/-- 
Given a sum P put at simple interest for 7 years, if increasing the interest rate 
by 2% results in $140 more interest, then P = $1000.
-/
theorem simple_interest_problem (P : ℚ) (R : ℚ) : 
  (P * (R + 2) * 7 / 100 = P * R * 7 / 100 + 140) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l910_91074


namespace NUMINAMATH_CALUDE_equal_size_meetings_l910_91013

/-- Given n sets representing daily meetings, prove that all sets have the same size. -/
theorem equal_size_meetings (n : ℕ) (A : Fin n → Finset (Fin n)) 
  (h_n : n ≥ 3)
  (h_size : ∀ i, (A i).card ≥ 3)
  (h_cover : ∀ i j, i < j → ∃! k, i ∈ A k ∧ j ∈ A k) :
  ∃ k, ∀ i, (A i).card = k :=
sorry

end NUMINAMATH_CALUDE_equal_size_meetings_l910_91013


namespace NUMINAMATH_CALUDE_cubic_expansion_l910_91040

theorem cubic_expansion (x : ℝ) : 
  3*x^3 - 10*x^2 + 13 = 3*(x-2)^3 + 8*(x-2)^2 - 4*(x-2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_l910_91040


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_multiplier_l910_91002

theorem consecutive_odd_integers_multiplier :
  ∀ (n : ℤ),
  (n + 4 = 15) →
  (∃ k : ℚ, 3 * n = k * (n + 4) + 3) →
  (∃ k : ℚ, 3 * n = k * (n + 4) + 3 ∧ k = 2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_multiplier_l910_91002


namespace NUMINAMATH_CALUDE_determinant_example_l910_91009

/-- Definition of a second-order determinant -/
def second_order_determinant (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem: The determinant of the matrix [[2, 1], [-3, 4]] is 11 -/
theorem determinant_example : second_order_determinant 2 (-3) 1 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_determinant_example_l910_91009


namespace NUMINAMATH_CALUDE_onion_rings_cost_l910_91092

/-- Proves that the cost of onion rings is $2 given the costs of other items and payment details --/
theorem onion_rings_cost (hamburger_cost smoothie_cost total_paid change : ℕ) :
  hamburger_cost = 4 →
  smoothie_cost = 3 →
  total_paid = 20 →
  change = 11 →
  total_paid - change - hamburger_cost - smoothie_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_onion_rings_cost_l910_91092


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_l910_91053

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if a circle with diameter equal to the distance between the foci
    intersects one of the hyperbola's asymptotes at point (4, 3),
    then a = 4 and b = 3 -/
theorem hyperbola_circle_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c^2 = 16 + 9 ∧ 3 = (b / a) * 4) → a = 4 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_l910_91053


namespace NUMINAMATH_CALUDE_cone_angle_calculation_l910_91061

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a cone -/
structure Cone where
  vertex : ℝ × ℝ × ℝ

/-- Configuration of spheres and cone -/
structure SphereConeConfiguration where
  sphere1 : Sphere
  sphere2 : Sphere
  sphere3 : Sphere
  cone : Cone
  spheresTouch : Bool
  coneTouchesSpheres : Bool
  vertexBetweenContacts : Bool

/-- The angle at the vertex of the cone -/
def coneAngle (config : SphereConeConfiguration) : ℝ :=
  sorry

theorem cone_angle_calculation (config : SphereConeConfiguration) 
  (h1 : config.sphere1.radius = 2)
  (h2 : config.sphere2.radius = 2)
  (h3 : config.sphere3.radius = 1)
  (h4 : config.spheresTouch = true)
  (h5 : config.coneTouchesSpheres = true)
  (h6 : config.vertexBetweenContacts = true) :
  coneAngle config = 2 * Real.arctan (1 / 8) :=
sorry

end NUMINAMATH_CALUDE_cone_angle_calculation_l910_91061


namespace NUMINAMATH_CALUDE_line_angle_problem_l910_91090

theorem line_angle_problem (a : ℝ) : 
  let line1 := {(x, y) : ℝ × ℝ | a * x - y + 3 = 0}
  let line2 := {(x, y) : ℝ × ℝ | x - 2 * y + 4 = 0}
  let angle := Real.arccos (Real.sqrt 5 / 5)
  (∃ (θ : ℝ), θ = angle ∧ 
    θ = Real.arccos ((1 + a * (1/2)) / Real.sqrt ((1 + a^2) * (1 + (1/2)^2))))
  → a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_line_angle_problem_l910_91090


namespace NUMINAMATH_CALUDE_unique_solution_l910_91007

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := log (2 * x + 1) + log x = 1

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, x > 0 ∧ 2 * x + 1 > 0 ∧ equation x ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l910_91007


namespace NUMINAMATH_CALUDE_exponent_division_l910_91059

theorem exponent_division (a : ℝ) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l910_91059


namespace NUMINAMATH_CALUDE_base7_product_and_sum_l910_91063

/-- Converts a base 7 number to decimal --/
def toDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a decimal number to base 7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Computes the sum of digits in a list --/
def sumDigits (n : List Nat) : Nat :=
  n.foldl (· + ·) 0

/-- The main theorem to prove --/
theorem base7_product_and_sum :
  let a := [5, 3]  -- 35 in base 7
  let b := [4, 2]  -- 24 in base 7
  let product := toBase7 (toDecimal a * toDecimal b)
  product = [6, 3, 2, 1] ∧ 
  toBase7 (sumDigits product) = [5, 1] := by
  sorry


end NUMINAMATH_CALUDE_base7_product_and_sum_l910_91063


namespace NUMINAMATH_CALUDE_min_bodyguards_tournament_l910_91038

/-- A tournament where each bodyguard is defeated by at least three others -/
def BodyguardTournament (n : ℕ) := 
  ∃ (defeats : Fin n → Fin n → Prop),
    (∀ i j k : Fin n, i ≠ j → ∃ l : Fin n, defeats l i ∧ defeats l j) ∧
    (∀ i : Fin n, ∃ j k l : Fin n, j ≠ i ∧ k ≠ i ∧ l ≠ i ∧ defeats j i ∧ defeats k i ∧ defeats l i)

/-- The minimum number of bodyguards in a tournament satisfying the conditions is 7 -/
theorem min_bodyguards_tournament : 
  (∃ n : ℕ, BodyguardTournament n) ∧ 
  (∀ m : ℕ, m < 7 → ¬BodyguardTournament m) ∧
  BodyguardTournament 7 :=
sorry

end NUMINAMATH_CALUDE_min_bodyguards_tournament_l910_91038


namespace NUMINAMATH_CALUDE_library_books_total_l910_91081

theorem library_books_total (initial_books additional_books : ℕ) : 
  initial_books = 54 → additional_books = 23 → initial_books + additional_books = 77 := by
  sorry

end NUMINAMATH_CALUDE_library_books_total_l910_91081


namespace NUMINAMATH_CALUDE_small_circle_radius_l910_91083

/-- Given a large circle with radius 6 meters containing five congruent smaller circles
    arranged such that the diameter of the large circle equals the sum of the diameters
    of three smaller circles, the radius of each smaller circle is 2 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 6 → 2 * R = 3 * (2 * r) → r = 2 :=
by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l910_91083


namespace NUMINAMATH_CALUDE_two_left_movements_l910_91050

-- Define the direction type
inductive Direction
| Left
| Right

-- Define a function to convert direction to sign
def directionToSign (d : Direction) : Int :=
  match d with
  | Direction.Left => -1
  | Direction.Right => 1

-- Define a single movement
def singleMovement (distance : ℝ) (direction : Direction) : ℝ :=
  (directionToSign direction : ℝ) * distance

-- Define the problem statement
theorem two_left_movements (distance : ℝ) :
  distance = 3 →
  (singleMovement distance Direction.Left + singleMovement distance Direction.Left) = -6 :=
by sorry

end NUMINAMATH_CALUDE_two_left_movements_l910_91050


namespace NUMINAMATH_CALUDE_no_perfect_square_ends_2012_l910_91027

theorem no_perfect_square_ends_2012 : ∀ a : ℤ, ¬(∃ k : ℤ, a^2 = 10000 * k + 2012) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_ends_2012_l910_91027


namespace NUMINAMATH_CALUDE_shark_count_l910_91076

theorem shark_count (cape_may_sharks : ℕ) (other_beach_sharks : ℕ) : 
  cape_may_sharks = 32 → 
  cape_may_sharks = 2 * other_beach_sharks + 8 → 
  other_beach_sharks = 12 := by
sorry

end NUMINAMATH_CALUDE_shark_count_l910_91076
