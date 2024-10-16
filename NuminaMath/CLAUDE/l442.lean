import Mathlib

namespace NUMINAMATH_CALUDE_files_per_folder_l442_44249

theorem files_per_folder (initial_files : ℝ) (additional_files : ℝ) (num_folders : ℝ) :
  let total_files := initial_files + additional_files
  total_files / num_folders = (initial_files + additional_files) / num_folders :=
by sorry

end NUMINAMATH_CALUDE_files_per_folder_l442_44249


namespace NUMINAMATH_CALUDE_yellow_sweets_count_l442_44253

theorem yellow_sweets_count (green_sweets blue_sweets total_sweets : ℕ) 
  (h1 : green_sweets = 212)
  (h2 : blue_sweets = 310)
  (h3 : total_sweets = 1024) : 
  total_sweets - (green_sweets + blue_sweets) = 502 := by
  sorry

end NUMINAMATH_CALUDE_yellow_sweets_count_l442_44253


namespace NUMINAMATH_CALUDE_walking_speed_proof_l442_44292

/-- The walking speed of person A in km/h -/
def a_speed : ℝ := 10

/-- The cycling speed of person B in km/h -/
def b_speed : ℝ := 20

/-- The time difference between A's start and B's start in hours -/
def time_diff : ℝ := 4

/-- The distance at which B catches up with A in km -/
def catch_up_distance : ℝ := 80

theorem walking_speed_proof :
  (catch_up_distance / a_speed = time_diff + catch_up_distance / b_speed) →
  a_speed = 10 := by
  sorry

#check walking_speed_proof

end NUMINAMATH_CALUDE_walking_speed_proof_l442_44292


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l442_44270

theorem units_digit_of_sum (n : ℕ) : n = 33^43 + 43^32 → n % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l442_44270


namespace NUMINAMATH_CALUDE_belt_cost_calculation_l442_44220

def initial_budget : ℕ := 200
def shirt_cost : ℕ := 30
def pants_cost : ℕ := 46
def coat_cost : ℕ := 38
def socks_cost : ℕ := 11
def shoes_cost : ℕ := 41
def amount_left : ℕ := 16

theorem belt_cost_calculation : 
  initial_budget - (shirt_cost + pants_cost + coat_cost + socks_cost + shoes_cost + amount_left) = 18 := by
  sorry

end NUMINAMATH_CALUDE_belt_cost_calculation_l442_44220


namespace NUMINAMATH_CALUDE_plane_equation_proof_l442_44248

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane equation in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointLiesOnPlane (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- Check if two planes are perpendicular -/
def planesArePerpendicular (eq1 eq2 : PlaneEquation) : Prop :=
  eq1.A * eq2.A + eq1.B * eq2.B + eq1.C * eq2.C = 0

/-- The greatest common divisor of the absolute values of four integers is 1 -/
def gcdOfFourIntsIsOne (a b c d : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

theorem plane_equation_proof (p1 p2 : Point3D) (givenPlane : PlaneEquation) 
    (h1 : p1 = ⟨2, -3, 4⟩) 
    (h2 : p2 = ⟨-1, 3, -2⟩)
    (h3 : givenPlane = ⟨3, -2, 1, -7⟩) :
  ∃ (resultPlane : PlaneEquation), 
    resultPlane.A > 0 ∧ 
    gcdOfFourIntsIsOne resultPlane.A resultPlane.B resultPlane.C resultPlane.D ∧
    pointLiesOnPlane p1 resultPlane ∧
    pointLiesOnPlane p2 resultPlane ∧
    planesArePerpendicular resultPlane givenPlane ∧
    resultPlane = ⟨2, 5, -4, 27⟩ := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l442_44248


namespace NUMINAMATH_CALUDE_range_of_expression_l442_44282

theorem range_of_expression (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 8 ∧ ∃ (y : ℝ), y = (a - b) * c^2 ∧ x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l442_44282


namespace NUMINAMATH_CALUDE_system_solution_l442_44257

theorem system_solution (x y z w : ℤ) : 
  x + y + z + w = 20 ∧
  y + 2*z - 3*w = 28 ∧
  x - 2*y + z = 36 ∧
  -7*x - y + 5*z + 3*w = 84 →
  x = 4 ∧ y = -6 ∧ z = 20 ∧ w = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l442_44257


namespace NUMINAMATH_CALUDE_f_max_value_l442_44227

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n : ℚ) / ((n + 32 : ℕ) * S (n + 1))

theorem f_max_value :
  (∀ n : ℕ, f n ≤ 1 / 50) ∧ (∃ n : ℕ, f n = 1 / 50) := by sorry

end NUMINAMATH_CALUDE_f_max_value_l442_44227


namespace NUMINAMATH_CALUDE_derivative_at_negative_one_l442_44299

/-- Given a function f(x) = ax^4 + bx^2 + c, if f'(1) = 2, then f'(-1) = -2 -/
theorem derivative_at_negative_one
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^4 + b * x^2 + c)
  (h2 : deriv f 1 = 2) :
  deriv f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_one_l442_44299


namespace NUMINAMATH_CALUDE_exactly_two_good_probability_l442_44201

def total_screws : ℕ := 10
def defective_screws : ℕ := 3
def drawn_screws : ℕ := 4

def probability_exactly_two_good : ℚ :=
  (Nat.choose (total_screws - defective_screws) 2 * Nat.choose defective_screws 2) /
  Nat.choose total_screws drawn_screws

theorem exactly_two_good_probability :
  probability_exactly_two_good = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_good_probability_l442_44201


namespace NUMINAMATH_CALUDE_population_increase_theorem_l442_44262

/-- Given birth and death rates per 1000 people, calculate the percentage increase in population rate -/
def population_increase_percentage (birth_rate death_rate : ℚ) : ℚ :=
  (birth_rate - death_rate) * 100 / 1000

theorem population_increase_theorem (birth_rate death_rate : ℚ) 
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11) : 
  population_increase_percentage birth_rate death_rate = (21 : ℚ) / 10 :=
by sorry

end NUMINAMATH_CALUDE_population_increase_theorem_l442_44262


namespace NUMINAMATH_CALUDE_distance_X_to_Y_l442_44200

/-- The distance between points X and Y -/
def D : ℝ := sorry

/-- Yolanda's walking rate in miles per hour -/
def yolanda_rate : ℝ := 3

/-- Bob's walking rate in miles per hour -/
def bob_rate : ℝ := 4

/-- Time difference between Yolanda and Bob's start in hours -/
def time_difference : ℝ := 1

/-- Distance Bob walked when they met -/
def bob_distance : ℝ := 4

/-- Theorem stating the distance between X and Y -/
theorem distance_X_to_Y : D = 10 := by sorry

end NUMINAMATH_CALUDE_distance_X_to_Y_l442_44200


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l442_44285

theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_fourth : a 4 = 81) 
  (h_fifth : a 5 = 162) : 
  a 1 = 10.125 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l442_44285


namespace NUMINAMATH_CALUDE_window_probability_l442_44290

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

def probability_BIRD : ℚ := 1 / (choose 4 2)
def probability_WINDS : ℚ := 3 / (choose 5 3)
def probability_FLOW : ℚ := 1 / (choose 4 2)

theorem window_probability : 
  probability_BIRD * probability_WINDS * probability_FLOW = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_window_probability_l442_44290


namespace NUMINAMATH_CALUDE_point_location_l442_44293

theorem point_location (m n : ℝ) : 2^m + 2^n < 2 * Real.sqrt 2 → m + n < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_location_l442_44293


namespace NUMINAMATH_CALUDE_math_city_intersections_l442_44219

/-- Represents a city layout with streets and intersections -/
structure CityLayout where
  num_streets : ℕ
  num_nonintersecting_pairs : ℕ

/-- Calculates the number of intersections in a city layout -/
def num_intersections (layout : CityLayout) : ℕ :=
  (layout.num_streets.choose 2) - layout.num_nonintersecting_pairs

/-- Theorem: In a city with 10 streets and 3 non-intersecting pairs, there are 42 intersections -/
theorem math_city_intersections :
  let layout : CityLayout := ⟨10, 3⟩
  num_intersections layout = 42 := by
  sorry

#eval num_intersections ⟨10, 3⟩

end NUMINAMATH_CALUDE_math_city_intersections_l442_44219


namespace NUMINAMATH_CALUDE_possible_values_of_a_l442_44289

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l442_44289


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l442_44265

theorem triangle_division_theorem (A B C : ℝ) :
  A + B + C = 180 →
  B = 120 →
  (∃ D : ℝ, (A = D ∧ B / 2 = D) ∨ (C = D ∧ B / 2 = D) ∨ (A = D ∧ C = D)) →
  ((A = 40 ∧ C = 20) ∨ (A = 45 ∧ C = 15) ∨ (A = 20 ∧ C = 40) ∨ (A = 15 ∧ C = 45)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l442_44265


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l442_44279

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 - 2*x + 3 < 0} = {x : ℝ | x < -3 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l442_44279


namespace NUMINAMATH_CALUDE_probability_red_socks_l442_44210

/-- Given a drawer with red, blue, and green socks, this function calculates
    the probability of picking a matching pair of a specific color. -/
def probability_matching_pair (red blue green : ℕ) (target_color : ℕ) : ℚ :=
  let total_socks := red + blue + green
  let matching_pairs := (red.choose 2) + (blue.choose 2) + (green.choose 2)
  let target_pairs := target_color.choose 2
  target_pairs / matching_pairs

/-- Theorem stating that the probability of picking a matching pair of red socks
    from a drawer with 4 red, 2 blue, and 2 green socks is 3/4. -/
theorem probability_red_socks : probability_matching_pair 4 2 2 4 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_socks_l442_44210


namespace NUMINAMATH_CALUDE_clara_age_problem_l442_44218

theorem clara_age_problem : ∃! x : ℕ+, 
  (∃ n : ℕ+, (x - 2 : ℤ) = n^2) ∧ 
  (∃ m : ℕ+, (x + 3 : ℤ) = m^3) ∧ 
  x = 123 := by
  sorry

end NUMINAMATH_CALUDE_clara_age_problem_l442_44218


namespace NUMINAMATH_CALUDE_min_value_of_transformed_sine_l442_44247

theorem min_value_of_transformed_sine (φ : ℝ) (h : |φ| < π/2) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2*x - π/3)
  ∃ (x : ℝ), x ∈ Set.Icc 0 (π/2) ∧ f x = -Real.sqrt 3 / 2 ∧
    ∀ y ∈ Set.Icc 0 (π/2), f y ≥ -Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_min_value_of_transformed_sine_l442_44247


namespace NUMINAMATH_CALUDE_sally_quarters_count_l442_44215

/-- Given an initial number of quarters, quarters spent, and quarters found,
    calculate the final number of quarters Sally has. -/
def final_quarters (initial spent found : ℕ) : ℕ :=
  initial - spent + found

/-- Theorem stating that Sally's final number of quarters is 492 -/
theorem sally_quarters_count :
  final_quarters 760 418 150 = 492 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_count_l442_44215


namespace NUMINAMATH_CALUDE_unique_solution_l442_44217

/-- Represents the number of students in a class -/
structure ClassSize where
  small : Nat
  large_min : Nat
  large_max : Nat

/-- Represents the number of classes for each school -/
structure SchoolClasses where
  shouchun_small : Nat
  binhu_small : Nat
  binhu_large : Nat

/-- Check if the given class distribution satisfies all conditions -/
def satisfies_conditions (cs : ClassSize) (sc : SchoolClasses) : Prop :=
  sc.shouchun_small + sc.binhu_small + sc.binhu_large = 45 ∧
  sc.binhu_small = 2 * sc.binhu_large ∧
  cs.small * (sc.shouchun_small + sc.binhu_small) + cs.large_min * sc.binhu_large ≤ 1800 ∧
  1800 ≤ cs.small * (sc.shouchun_small + sc.binhu_small) + cs.large_max * sc.binhu_large

theorem unique_solution (cs : ClassSize) (h_cs : cs.small = 36 ∧ cs.large_min = 70 ∧ cs.large_max = 75) :
  ∃! sc : SchoolClasses, satisfies_conditions cs sc ∧ 
    sc.shouchun_small = 30 ∧ sc.binhu_small = 10 ∧ sc.binhu_large = 5 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_l442_44217


namespace NUMINAMATH_CALUDE_initial_amount_proof_l442_44228

theorem initial_amount_proof (remaining_amount : ℝ) (spent_percentage : ℝ) (initial_amount : ℝ) : 
  remaining_amount = 3500 ∧ 
  spent_percentage = 30 ∧ 
  remaining_amount = initial_amount * (1 - spent_percentage / 100) → 
  initial_amount = 5000 := by
sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l442_44228


namespace NUMINAMATH_CALUDE_middle_term_binomial_coefficient_l442_44271

theorem middle_term_binomial_coefficient 
  (n : ℕ) 
  (x : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : 2^(n-1) = 1024) : 
  Nat.choose n ((n-1)/2) = 462 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_binomial_coefficient_l442_44271


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l442_44241

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l442_44241


namespace NUMINAMATH_CALUDE_part1_part2_l442_44237

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, -2)
def C : ℝ × ℝ := (4, 1)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Part 1
theorem part1 : ∀ D : ℝ × ℝ, AB = (D.1 - C.1, D.2 - C.2) → D = (5, -4) := by sorry

-- Part 2
theorem part2 : ∀ k : ℝ, (∃ t : ℝ, t ≠ 0 ∧ (k * AB.1 - BC.1, k * AB.2 - BC.2) = (t * (AB.1 + 3 * BC.1), t * (AB.2 + 3 * BC.2))) → k = -1/3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l442_44237


namespace NUMINAMATH_CALUDE_complement_A_union_B_l442_44286

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | Real.log (x - 2) ≤ 0}

-- State the theorem
theorem complement_A_union_B : (Set.univ \ A) ∪ B = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l442_44286


namespace NUMINAMATH_CALUDE_remainder_76_pow_77_div_7_l442_44244

theorem remainder_76_pow_77_div_7 : (76^77) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_76_pow_77_div_7_l442_44244


namespace NUMINAMATH_CALUDE_max_value_of_f_l442_44278

-- Define the function f
def f (x : ℝ) : ℝ := -x^4 + 2*x^2 + 3

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l442_44278


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l442_44284

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 7351 : ℤ) ≡ 3071 [ZMOD 17] ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 7351 : ℤ) ≡ 3071 [ZMOD 17] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l442_44284


namespace NUMINAMATH_CALUDE_inequality_proof_l442_44234

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l442_44234


namespace NUMINAMATH_CALUDE_cross_product_result_l442_44243

def u : ℝ × ℝ × ℝ := (3, 4, 2)
def v : ℝ × ℝ × ℝ := (1, -2, 5)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.2.2 * b.1 - a.1 * b.2.2,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_result : cross_product u v = (24, -13, -10) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_result_l442_44243


namespace NUMINAMATH_CALUDE_seven_lines_angle_l442_44298

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a function to check if two lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define a function to measure the angle between two lines
def angle_between (l1 l2 : Line) : ℝ := sorry

-- The main theorem
theorem seven_lines_angle (lines : Fin 7 → Line) :
  (∀ i j, i ≠ j → ¬ parallel (lines i) (lines j)) →
  ∃ i j, i ≠ j ∧ angle_between (lines i) (lines j) < 26 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_seven_lines_angle_l442_44298


namespace NUMINAMATH_CALUDE_base_4_9_digit_difference_l442_44204

theorem base_4_9_digit_difference (n : ℕ) (h : n = 1024) : 
  (Nat.log 4 n + 1) - (Nat.log 9 n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_4_9_digit_difference_l442_44204


namespace NUMINAMATH_CALUDE_parabola_vertex_l442_44250

/-- The parabola is defined by the equation y = (x + 3)^2 - 1 -/
def parabola (x : ℝ) : ℝ := (x + 3)^2 - 1

/-- The vertex of the parabola y = (x + 3)^2 - 1 is at the point (-3, -1) -/
theorem parabola_vertex : 
  (∃ (a : ℝ), ∀ (x : ℝ), parabola x = a * (x + 3)^2 - 1) → 
  (∀ (x : ℝ), parabola x ≥ parabola (-3)) ∧ parabola (-3) = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l442_44250


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l442_44212

theorem ticket_price_possibilities : ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ x ∈ S, x > 0 ∧ 72 % x = 0 ∧ 90 % x = 0 ∧ 150 % x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l442_44212


namespace NUMINAMATH_CALUDE_prize_distribution_l442_44239

theorem prize_distribution (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  n^k = 8000 := by
  sorry

end NUMINAMATH_CALUDE_prize_distribution_l442_44239


namespace NUMINAMATH_CALUDE_unique_campers_rowing_l442_44206

theorem unique_campers_rowing (total_campers : ℕ) (morning : ℕ) (afternoon : ℕ) (evening : ℕ)
  (morning_and_afternoon : ℕ) (afternoon_and_evening : ℕ) (morning_and_evening : ℕ) (all_three : ℕ)
  (h1 : total_campers = 500)
  (h2 : morning = 235)
  (h3 : afternoon = 387)
  (h4 : evening = 142)
  (h5 : morning_and_afternoon = 58)
  (h6 : afternoon_and_evening = 23)
  (h7 : morning_and_evening = 15)
  (h8 : all_three = 8) :
  morning + afternoon + evening - (morning_and_afternoon + afternoon_and_evening + morning_and_evening) + all_three = 572 :=
by sorry

end NUMINAMATH_CALUDE_unique_campers_rowing_l442_44206


namespace NUMINAMATH_CALUDE_product_of_squares_l442_44242

theorem product_of_squares (x y z : ℚ) 
  (hx : x = 1/4) 
  (hy : y = 1/2) 
  (hz : z = -8) : 
  x^2 * y^2 * z^2 = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_squares_l442_44242


namespace NUMINAMATH_CALUDE_walking_distance_calculation_l442_44245

-- Define the speeds and additional distances
def speed1_original : ℝ := 5
def speed1_alternative : ℝ := 15
def speed2_original : ℝ := 10
def speed2_alternative : ℝ := 20
def additional_distance1 : ℝ := 45
def additional_distance2 : ℝ := 30

-- Define the theorem
theorem walking_distance_calculation :
  ∃ (t1 t2 : ℝ),
    t1 > 0 ∧ t2 > 0 ∧
    (speed1_alternative * t1 - speed1_original * t1 = additional_distance1) ∧
    (speed2_alternative * t2 - speed2_original * t2 = additional_distance2) ∧
    (speed1_original * t1 = 22.5) ∧
    (speed2_original * t2 = 30) :=
  sorry

end NUMINAMATH_CALUDE_walking_distance_calculation_l442_44245


namespace NUMINAMATH_CALUDE_coin_coverage_probability_l442_44216

theorem coin_coverage_probability (square_side : ℝ) (triangle_leg : ℝ) (diamond_side : ℝ) (coin_diameter : ℝ) :
  square_side = 10 →
  triangle_leg = 3 →
  diamond_side = 3 * Real.sqrt 2 →
  coin_diameter = 2 →
  let coin_radius : ℝ := coin_diameter / 2
  let landing_area : ℝ := (square_side - 2 * coin_radius) ^ 2
  let triangle_area : ℝ := 4 * (triangle_leg ^ 2 / 2 + π * coin_radius ^ 2 / 4 + triangle_leg * coin_radius)
  let diamond_area : ℝ := diamond_side ^ 2 + 4 * (π * coin_radius ^ 2 / 4 + diamond_side * coin_radius / Real.sqrt 2)
  let total_black_area : ℝ := triangle_area + diamond_area
  let probability : ℝ := total_black_area / landing_area
  probability = (1 / 225) * (900 + 300 * Real.sqrt 2 + π) := by
    sorry

end NUMINAMATH_CALUDE_coin_coverage_probability_l442_44216


namespace NUMINAMATH_CALUDE_triangular_square_l442_44246

/-- Triangular numbers -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Main theorem -/
theorem triangular_square (m n : ℕ) (h : 2 * triangular m = triangular n) :
  triangular (2 * m - n) = (m - n) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangular_square_l442_44246


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l442_44275

theorem ferris_wheel_capacity (total_people : ℕ) (total_seats : ℕ) 
  (h1 : total_people = 16) (h2 : total_seats = 4) : 
  total_people / total_seats = 4 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l442_44275


namespace NUMINAMATH_CALUDE_gina_sister_choice_ratio_l442_44274

/-- The ratio of Gina's choices to her sister's choices on Netflix --/
theorem gina_sister_choice_ratio :
  ∀ (sister_shows : ℕ) (show_length : ℕ) (gina_minutes : ℕ),
  sister_shows = 24 →
  show_length = 50 →
  gina_minutes = 900 →
  (gina_minutes : ℚ) / (sister_shows * show_length : ℚ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_gina_sister_choice_ratio_l442_44274


namespace NUMINAMATH_CALUDE_equation_solution_l442_44252

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_l442_44252


namespace NUMINAMATH_CALUDE_log_inequality_l442_44295

theorem log_inequality (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x) / (x + 1) + 1 / x > (Real.log x) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l442_44295


namespace NUMINAMATH_CALUDE_trajectory_equation_of_P_l442_44266

/-- The trajectory equation of point P on the xOy plane, given its distance from A(0,0,4) -/
theorem trajectory_equation_of_P (P : ℝ × ℝ) (d : ℝ → ℝ → ℝ → ℝ → ℝ) :
  (∀ z, d P.1 P.2 0 z = d P.1 P.2 0 0) →  -- P is on the xOy plane
  d P.1 P.2 0 4 = 5 →                     -- distance between P and A is 5
  P.1^2 + P.2^2 = 9 :=                    -- trajectory equation
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_of_P_l442_44266


namespace NUMINAMATH_CALUDE_sin_pi_third_value_l442_44233

theorem sin_pi_third_value (f : ℝ → ℝ) :
  (∀ α : ℝ, f (Real.sin α + Real.cos α) = (1/2) * Real.sin (2 * α)) →
  f (Real.sin (π/3)) = -1/8 := by
sorry

end NUMINAMATH_CALUDE_sin_pi_third_value_l442_44233


namespace NUMINAMATH_CALUDE_length_of_24_l442_44281

def length_of_integer (k : ℕ) : ℕ := sorry

theorem length_of_24 : 
  let k : ℕ := 24
  length_of_integer k = 4 := by sorry

end NUMINAMATH_CALUDE_length_of_24_l442_44281


namespace NUMINAMATH_CALUDE_coffee_shop_total_sales_l442_44213

/-- Calculates the total money made by a coffee shop given the number of coffee and tea orders and their respective prices. -/
def coffee_shop_sales (coffee_orders : ℕ) (coffee_price : ℕ) (tea_orders : ℕ) (tea_price : ℕ) : ℕ :=
  coffee_orders * coffee_price + tea_orders * tea_price

/-- Theorem stating that the coffee shop made $67 given the specified orders and prices. -/
theorem coffee_shop_total_sales :
  coffee_shop_sales 7 5 8 4 = 67 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_total_sales_l442_44213


namespace NUMINAMATH_CALUDE_expression_evaluation_l442_44268

theorem expression_evaluation :
  let a : ℚ := 4/3
  (7 * a^2 - 15 * a + 2) * (3 * a - 4) = 0 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l442_44268


namespace NUMINAMATH_CALUDE_cauchy_schwarz_two_terms_l442_44226

theorem cauchy_schwarz_two_terms
  (a b x y : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hx : 0 < x)
  (hy : 0 < y) :
  a * x + b * y ≤ Real.sqrt (a^2 + b^2) * Real.sqrt (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_two_terms_l442_44226


namespace NUMINAMATH_CALUDE_intersection_M_N_l442_44238

def M : Set ℕ := {0, 2, 3, 4}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l442_44238


namespace NUMINAMATH_CALUDE_children_left_on_bus_l442_44214

theorem children_left_on_bus (initial_children : ℕ) (difference : ℕ) : 
  initial_children = 41 →
  difference = 23 →
  initial_children - difference = 18 :=
by sorry

end NUMINAMATH_CALUDE_children_left_on_bus_l442_44214


namespace NUMINAMATH_CALUDE_perry_vs_phil_l442_44287

/-- The number of games won by each player -/
structure GolfWins where
  phil : ℕ
  charlie : ℕ
  dana : ℕ
  perry : ℕ

/-- The conditions of the golf game results -/
def golf_conditions (w : GolfWins) : Prop :=
  w.perry = w.dana + 5 ∧
  w.charlie = w.dana - 2 ∧
  w.phil = w.charlie + 3 ∧
  w.phil = 12

theorem perry_vs_phil (w : GolfWins) (h : golf_conditions w) : w.perry = w.phil + 4 :=
sorry

end NUMINAMATH_CALUDE_perry_vs_phil_l442_44287


namespace NUMINAMATH_CALUDE_inequality_proof_l442_44294

/-- Given f(x) = e^x - x^2, prove that for all x > 0, (e^x + (2-e)x - 1) / x ≥ ln x + 1 -/
theorem inequality_proof (x : ℝ) (hx : x > 0) : (Real.exp x + (2 - Real.exp 1) * x - 1) / x ≥ Real.log x + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l442_44294


namespace NUMINAMATH_CALUDE_A_equals_one_two_l442_44264

def A : Set ℤ := {x | 0 < x ∧ x ≤ 2}

theorem A_equals_one_two : A = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_equals_one_two_l442_44264


namespace NUMINAMATH_CALUDE_no_pythagorean_triple_with_3_l442_44263

theorem no_pythagorean_triple_with_3 :
  ¬∃ (a b c : ℤ), a^2 + b^2 = 3 * c^2 ∧ Int.gcd a (Int.gcd b c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_pythagorean_triple_with_3_l442_44263


namespace NUMINAMATH_CALUDE_quadratic_equation_game_l442_44209

/-- Represents a strategy for playing the quadratic equation game -/
def Strategy := ℕ → ℕ → ℝ → ℝ → ℝ → ℝ

/-- Represents the outcome of the game given two strategies -/
def GameOutcome (n : ℕ) (stratA stratB : Strategy) : ℕ := sorry

/-- The maximum number of equations without real roots that can be guaranteed -/
def MaxGuaranteedNoRealRoots (n : ℕ) : ℕ := (n + 1) / 2

theorem quadratic_equation_game (n : ℕ) (h : Odd n) :
  ∀ (stratA : Strategy),
    ∃ (stratB : Strategy),
      GameOutcome n stratA stratB ≤ MaxGuaranteedNoRealRoots n ∧
    ∀ (stratB : Strategy),
      ∃ (stratA : Strategy),
        GameOutcome n stratA stratB ≥ MaxGuaranteedNoRealRoots n :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_game_l442_44209


namespace NUMINAMATH_CALUDE_ball_costs_and_max_purchase_l442_44276

/-- Represents the cost of basketballs and soccer balls -/
structure BallCosts where
  basketball : ℕ
  soccer : ℕ

/-- Represents the purchase constraints -/
structure PurchaseConstraints where
  total_balls : ℕ
  max_cost : ℕ

/-- Theorem stating the correct costs and maximum number of basketballs -/
theorem ball_costs_and_max_purchase 
  (costs : BallCosts) 
  (constraints : PurchaseConstraints) : 
  (2 * costs.basketball + 3 * costs.soccer = 310) → 
  (5 * costs.basketball + 2 * costs.soccer = 500) → 
  (constraints.total_balls = 60) → 
  (constraints.max_cost = 4000) → 
  (costs.basketball = 80 ∧ costs.soccer = 50 ∧ 
   (∀ m : ℕ, m * costs.basketball + (constraints.total_balls - m) * costs.soccer ≤ constraints.max_cost → m ≤ 33)) := by
  sorry

end NUMINAMATH_CALUDE_ball_costs_and_max_purchase_l442_44276


namespace NUMINAMATH_CALUDE_paper_towel_cost_l442_44221

theorem paper_towel_cost (case_price : ℝ) (num_rolls : ℕ) (savings_percent : ℝ) :
  case_price = 9 →
  num_rolls = 12 →
  savings_percent = 25 →
  ∃ (individual_price : ℝ),
    case_price = (1 - savings_percent / 100) * (num_rolls * individual_price) ∧
    individual_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_paper_towel_cost_l442_44221


namespace NUMINAMATH_CALUDE_magnitude_BD_l442_44283

def A : ℂ := Complex.I
def B : ℂ := 1
def C : ℂ := 4 + 2 * Complex.I

def parallelogram_ABCD (A B C : ℂ) : Prop :=
  ∃ D : ℂ, (C - B) = (D - A) ∧ (D - C) = (B - A)

theorem magnitude_BD (D : ℂ) (h : parallelogram_ABCD A B C) : 
  Complex.abs (D - B) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_BD_l442_44283


namespace NUMINAMATH_CALUDE_captain_times_proof_l442_44236

-- Define the points and captain times for each boy
def points_A : ℕ := sorry
def points_E : ℕ := sorry
def points_B : ℕ := sorry
def captain_time_A : ℕ := sorry
def captain_time_E : ℕ := sorry
def captain_time_B : ℕ := sorry

-- Define the total travel time
def total_time : ℕ := sorry

-- State the theorem
theorem captain_times_proof :
  -- Conditions
  (points_A = points_B + 3) →
  (points_E + points_B = 15) →
  (total_time / 10 = points_A + points_E + points_B + 25) →
  (captain_time_B = 160) →
  -- Proportionality condition
  (∃ (k : ℚ), 
    captain_time_A = k * points_A ∧
    captain_time_E = k * points_E ∧
    captain_time_B = k * points_B) →
  -- Conclusion
  (captain_time_A = 200 ∧ captain_time_B = 140) :=
by sorry

end NUMINAMATH_CALUDE_captain_times_proof_l442_44236


namespace NUMINAMATH_CALUDE_max_d_value_l442_44211

def a (n : ℕ+) : ℕ := 100 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ k : ℕ+, d k = 401) ∧ (∀ n : ℕ+, d n ≤ 401) := by
  sorry

end NUMINAMATH_CALUDE_max_d_value_l442_44211


namespace NUMINAMATH_CALUDE_system_implication_l442_44229

theorem system_implication (f g : ℝ → ℝ) :
  (∀ x, f x > 0 ∧ g x > 0) → (∀ x, f x > 0 ∧ f x + g x > 0) := by
  sorry

end NUMINAMATH_CALUDE_system_implication_l442_44229


namespace NUMINAMATH_CALUDE_coefficient_of_linear_term_l442_44277

theorem coefficient_of_linear_term (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 0) → (a = 1 ∧ b = 3 ∧ c = -1) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_linear_term_l442_44277


namespace NUMINAMATH_CALUDE_cryptarithm_solution_exists_and_unique_l442_44251

/-- Represents a cryptarithm solution -/
structure CryptarithmSolution where
  A : Nat
  H : Nat
  J : Nat
  O : Nat
  K : Nat
  E : Nat

/-- Checks if all digits in the solution are unique -/
def uniqueDigits (sol : CryptarithmSolution) : Prop :=
  sol.A ≠ sol.H ∧ sol.A ≠ sol.J ∧ sol.A ≠ sol.O ∧ sol.A ≠ sol.K ∧ sol.A ≠ sol.E ∧
  sol.H ≠ sol.J ∧ sol.H ≠ sol.O ∧ sol.H ≠ sol.K ∧ sol.H ≠ sol.E ∧
  sol.J ≠ sol.O ∧ sol.J ≠ sol.K ∧ sol.J ≠ sol.E ∧
  sol.O ≠ sol.K ∧ sol.O ≠ sol.E ∧
  sol.K ≠ sol.E

/-- Checks if the solution satisfies the cryptarithm equation -/
def satisfiesCryptarithm (sol : CryptarithmSolution) : Prop :=
  (100001 * sol.A + 11010 * sol.H) / (10 * sol.H + sol.A) = 
  1000 * sol.J + 100 * sol.O + 10 * sol.K + sol.E

/-- The main theorem stating that there exists a unique solution to the cryptarithm -/
theorem cryptarithm_solution_exists_and_unique :
  ∃! sol : CryptarithmSolution,
    uniqueDigits sol ∧
    satisfiesCryptarithm sol ∧
    sol.A = 3 ∧ sol.H = 7 ∧ sol.J = 5 ∧ sol.O = 1 ∧ sol.K = 6 ∧ sol.E = 9 :=
sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_exists_and_unique_l442_44251


namespace NUMINAMATH_CALUDE_fraction_equality_l442_44267

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - c) * (b - d) / ((c - d) * (d - a)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l442_44267


namespace NUMINAMATH_CALUDE_unique_recurrence_solution_l442_44205

/-- A sequence of positive real numbers satisfying the given recurrence relation. -/
def RecurrenceSequence (X : ℕ → ℝ) : Prop :=
  (∀ n, X n > 0) ∧ 
  (∀ n, X (n + 2) = (1 / X (n + 1) + X n) / 2)

/-- The theorem stating that the only sequence satisfying the recurrence relation is the constant sequence of 1. -/
theorem unique_recurrence_solution (X : ℕ → ℝ) :
  RecurrenceSequence X → (∀ n, X n = 1) := by
  sorry

#check unique_recurrence_solution

end NUMINAMATH_CALUDE_unique_recurrence_solution_l442_44205


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_digit_sum_l442_44231

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if all digits of a number are non-zero -/
def all_digits_nonzero (n : ℕ) : Prop := sorry

/-- Number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all positive integers n, there exists an n-digit number z
    such that none of its digits are 0 and z is divisible by the sum of its digits -/
theorem exists_number_divisible_by_digit_sum :
  ∀ n : ℕ, n > 0 → ∃ z : ℕ,
    num_digits z = n ∧
    all_digits_nonzero z ∧
    z % sum_of_digits z = 0 :=
by sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_digit_sum_l442_44231


namespace NUMINAMATH_CALUDE_square_inequality_not_sufficient_nor_necessary_l442_44224

theorem square_inequality_not_sufficient_nor_necessary (x y : ℝ) :
  ¬(∀ x y : ℝ, x^2 > y^2 → x > y) ∧ ¬(∀ x y : ℝ, x > y → x^2 > y^2) := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_not_sufficient_nor_necessary_l442_44224


namespace NUMINAMATH_CALUDE_min_distance_between_ellipses_l442_44291

/-- The minimum distance between two ellipses -/
theorem min_distance_between_ellipses :
  let ellipse1 := {(x, y) : ℝ × ℝ | x^2 / 4 + y^2 = 1}
  let ellipse2 := {(x, y) : ℝ × ℝ | (x - 1)^2 / 9 + y^2 / 9 = 1}
  (∃ (A B : ℝ × ℝ), A ∈ ellipse1 ∧ B ∈ ellipse2 ∧
    ∀ (C D : ℝ × ℝ), C ∈ ellipse1 → D ∈ ellipse2 →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)) ∧
  (∀ (A B : ℝ × ℝ), A ∈ ellipse1 → B ∈ ellipse2 →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ 2) ∧
  (∃ (A B : ℝ × ℝ), A ∈ ellipse1 ∧ B ∈ ellipse2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_ellipses_l442_44291


namespace NUMINAMATH_CALUDE_sqrt_inequality_l442_44272

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d)
  (h5 : a + d = b + c) : 
  Real.sqrt a + Real.sqrt d < Real.sqrt b + Real.sqrt c := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l442_44272


namespace NUMINAMATH_CALUDE_oysters_with_pearls_percentage_l442_44260

/-- The percentage of oysters with pearls, given the number of oysters collected per dive,
    the number of dives, and the total number of pearls collected. -/
def percentage_oysters_with_pearls (oysters_per_dive : ℕ) (num_dives : ℕ) (total_pearls : ℕ) : ℚ :=
  (total_pearls : ℚ) / ((oysters_per_dive * num_dives) : ℚ) * 100

/-- Theorem stating that the percentage of oysters with pearls is 25%,
    given the specific conditions from the problem. -/
theorem oysters_with_pearls_percentage :
  percentage_oysters_with_pearls 16 14 56 = 25 := by
  sorry


end NUMINAMATH_CALUDE_oysters_with_pearls_percentage_l442_44260


namespace NUMINAMATH_CALUDE_bus_express_speed_l442_44269

/-- Proves that the speed of a bus in express mode is 48 km/h given specific conditions -/
theorem bus_express_speed (route_length : ℝ) (time_reduction : ℝ) (speed_increase : ℝ)
  (h1 : route_length = 16)
  (h2 : time_reduction = 1 / 15)
  (h3 : speed_increase = 8)
  : ∃ x : ℝ, x = 48 ∧ 
    route_length / (x - speed_increase) - route_length / x = time_reduction :=
by sorry

end NUMINAMATH_CALUDE_bus_express_speed_l442_44269


namespace NUMINAMATH_CALUDE_largest_common_divisor_420_385_l442_44225

def largest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem largest_common_divisor_420_385 :
  largest_common_divisor 420 385 = 35 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_420_385_l442_44225


namespace NUMINAMATH_CALUDE_railway_distances_l442_44223

theorem railway_distances (total_distance : ℝ) 
  (moscow_mozhaysk_ratio : ℝ) (mozhaysk_vyazma_ratio : ℝ) 
  (vyazma_smolensk_ratio : ℝ) :
  total_distance = 415 ∧ 
  moscow_mozhaysk_ratio = 7/9 ∧ 
  mozhaysk_vyazma_ratio = 27/35 →
  ∃ (moscow_mozhaysk vyazma_smolensk mozhaysk_vyazma : ℝ),
    moscow_mozhaysk = 105 ∧
    mozhaysk_vyazma = 135 ∧
    vyazma_smolensk = 175 ∧
    moscow_mozhaysk + mozhaysk_vyazma + vyazma_smolensk = total_distance ∧
    moscow_mozhaysk = moscow_mozhaysk_ratio * mozhaysk_vyazma ∧
    mozhaysk_vyazma = mozhaysk_vyazma_ratio * vyazma_smolensk :=
by sorry

end NUMINAMATH_CALUDE_railway_distances_l442_44223


namespace NUMINAMATH_CALUDE_sqrt_30_bounds_l442_44232

theorem sqrt_30_bounds : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_30_bounds_l442_44232


namespace NUMINAMATH_CALUDE_graph_is_two_lines_l442_44254

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 25 * y^2 - 20 * x + 100 = 0

/-- Definition of a line in slope-intercept form -/
def is_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

/-- The graph represents two lines -/
theorem graph_is_two_lines :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, equation x y ↔ (is_line m₁ b₁ x y ∨ is_line m₂ b₂ x y)) ∧
    m₁ ≠ m₂ :=
sorry

end NUMINAMATH_CALUDE_graph_is_two_lines_l442_44254


namespace NUMINAMATH_CALUDE_equation_solution_l442_44230

theorem equation_solution : ∃ x : ℝ, (x + 1) / 2 = x - 2 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l442_44230


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l442_44288

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (A : ℚ),
    team_size = 11 →
    captain_age = 26 →
    wicket_keeper_age_diff = 5 →
    (team_size : ℚ) * A - (captain_age + (captain_age + wicket_keeper_age_diff)) = 
      (team_size - 2 : ℚ) * (A - 1) →
    A = 24 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l442_44288


namespace NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l442_44208

theorem square_area_to_perimeter_ratio (s1 s2 : ℝ) (h : s1 ^ 2 / s2 ^ 2 = 16 / 25) :
  (4 * s1) / (4 * s2) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l442_44208


namespace NUMINAMATH_CALUDE_family_weight_gain_l442_44207

/-- The total weight gained by three family members at a reunion --/
theorem family_weight_gain (orlando_gain jose_gain fernando_gain : ℕ) : 
  orlando_gain = 5 →
  jose_gain = 2 * orlando_gain + 2 →
  fernando_gain = jose_gain / 2 - 3 →
  orlando_gain + jose_gain + fernando_gain = 20 := by
sorry

end NUMINAMATH_CALUDE_family_weight_gain_l442_44207


namespace NUMINAMATH_CALUDE_matrix_value_equation_l442_44240

theorem matrix_value_equation (x : ℝ) : 
  (3 * x) * (4 * x) - 2 * (2 * x) = 6 ↔ x = -1/3 ∨ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_value_equation_l442_44240


namespace NUMINAMATH_CALUDE_lcm_72_98_l442_44202

theorem lcm_72_98 : Nat.lcm 72 98 = 3528 := by
  sorry

end NUMINAMATH_CALUDE_lcm_72_98_l442_44202


namespace NUMINAMATH_CALUDE_chicken_katsu_cost_is_25_l442_44203

/-- The cost of the chicken katsu given the following conditions:
  - The family ordered a smoky salmon for $40, a black burger for $15, and a chicken katsu.
  - The bill includes a 10% service charge and 5% tip.
  - Mr. Arevalo paid with $100 and received $8 in change.
-/
def chicken_katsu_cost : ℝ :=
  let salmon_cost : ℝ := 40
  let burger_cost : ℝ := 15
  let service_charge_rate : ℝ := 0.10
  let tip_rate : ℝ := 0.05
  let total_paid : ℝ := 100
  let change_received : ℝ := 8
  let total_bill : ℝ := total_paid - change_received
  25

theorem chicken_katsu_cost_is_25 :
  chicken_katsu_cost = 25 := by sorry

end NUMINAMATH_CALUDE_chicken_katsu_cost_is_25_l442_44203


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l442_44273

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 16 = -y^2 + 6*x + 36

-- Define the center and radius of the circle
def is_center_and_radius (a b r : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), is_center_and_radius a b r ∧ a + b + r = 5 + Real.sqrt 65 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l442_44273


namespace NUMINAMATH_CALUDE_time_to_cut_kids_hair_l442_44222

/-- Proves that the time to cut a kid's hair is 25 minutes given the specified conditions --/
theorem time_to_cut_kids_hair (
  time_woman : ℕ)
  (time_man : ℕ)
  (num_women : ℕ)
  (num_men : ℕ)
  (num_children : ℕ)
  (total_time : ℕ)
  (h1 : time_woman = 50)
  (h2 : time_man = 15)
  (h3 : num_women = 3)
  (h4 : num_men = 2)
  (h5 : num_children = 3)
  (h6 : total_time = 255)
  (h7 : total_time = time_woman * num_women + time_man * num_men + num_children * (total_time - time_woman * num_women - time_man * num_men) / num_children) :
  (total_time - time_woman * num_women - time_man * num_men) / num_children = 25 := by
  sorry

end NUMINAMATH_CALUDE_time_to_cut_kids_hair_l442_44222


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l442_44256

/-- 
In a right triangle ABC with right angle at B, 
the sum of arctan(b/(a+c)) and arctan(c/(a+b)) equals π/4, 
where a, b, and c are the lengths of the sides opposite to angles A, B, and C respectively.
-/
theorem right_triangle_arctan_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (right_angle : a^2 = b^2 + c^2) : 
  Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l442_44256


namespace NUMINAMATH_CALUDE_functional_equation_proof_l442_44297

open Real

theorem functional_equation_proof (f g : ℝ → ℝ) : 
  (∀ x y : ℝ, sin x + cos y = f x + f y + g x - g y) ↔ 
  (∃ c : ℝ, ∀ x : ℝ, f x = (sin x + cos x) / 2 ∧ g x = (sin x - cos x) / 2 + c) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_proof_l442_44297


namespace NUMINAMATH_CALUDE_cube_root_simplification_l442_44235

theorem cube_root_simplification :
  (40^3 + 50^3 + 60^3 : ℝ)^(1/3) = 10 * 405^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l442_44235


namespace NUMINAMATH_CALUDE_power_two_100_mod_3_l442_44255

theorem power_two_100_mod_3 : 2^100 ≡ 1 [ZMOD 3] := by sorry

end NUMINAMATH_CALUDE_power_two_100_mod_3_l442_44255


namespace NUMINAMATH_CALUDE_problem_solution_l442_44280

theorem problem_solution (a b : ℝ) (h1 : 4 + a = 5 - b) (h2 : 5 + b = 8 + a) : 4 - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l442_44280


namespace NUMINAMATH_CALUDE_trevor_placed_105_pieces_l442_44258

/-- Represents the puzzle problem --/
def PuzzleProblem (total : ℕ) (border : ℕ) (missing : ℕ) (joeMultiplier : ℕ) :=
  {trevor : ℕ // 
    trevor + joeMultiplier * trevor + border + missing = total ∧
    trevor > 0 ∧
    joeMultiplier > 0}

theorem trevor_placed_105_pieces :
  ∃ (p : PuzzleProblem 500 75 5 3), p.val = 105 := by
  sorry

end NUMINAMATH_CALUDE_trevor_placed_105_pieces_l442_44258


namespace NUMINAMATH_CALUDE_twelve_gon_consecutive_sides_sum_l442_44259

theorem twelve_gon_consecutive_sides_sum (sides : Fin 12 → ℕ) 
  (h1 : ∀ i : Fin 12, sides i = i.val + 1) : 
  ∃ i : Fin 12, sides i + sides (i + 1) + sides (i + 2) > 20 :=
by sorry

end NUMINAMATH_CALUDE_twelve_gon_consecutive_sides_sum_l442_44259


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l442_44296

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 19 →
  difference = 3 →
  friend_cost = total / 2 + difference / 2 →
  friend_cost = 11 := by
sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l442_44296


namespace NUMINAMATH_CALUDE_prime_neighbor_divisible_by_six_l442_44261

theorem prime_neighbor_divisible_by_six (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  6 ∣ (p - 1) ∨ 6 ∣ (p + 1) := by
  sorry

#check prime_neighbor_divisible_by_six

end NUMINAMATH_CALUDE_prime_neighbor_divisible_by_six_l442_44261
