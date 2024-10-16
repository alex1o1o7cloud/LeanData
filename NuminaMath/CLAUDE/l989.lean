import Mathlib

namespace NUMINAMATH_CALUDE_linear_equation_solve_l989_98922

theorem linear_equation_solve (x y : ℝ) : x + 2 * y = 6 → y = (-x + 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solve_l989_98922


namespace NUMINAMATH_CALUDE_light_2004_is_red_l989_98919

def light_color (n : ℕ) : String :=
  match n % 6 with
  | 0 => "red"
  | 1 => "green"
  | 2 => "yellow"
  | 3 => "yellow"
  | 4 => "red"
  | 5 => "red"
  | _ => "error" -- This case should never occur

theorem light_2004_is_red : light_color 2004 = "red" := by
  sorry

end NUMINAMATH_CALUDE_light_2004_is_red_l989_98919


namespace NUMINAMATH_CALUDE_lucy_fish_count_l989_98976

-- Define the given quantities
def current_fish : ℕ := 212
def additional_fish : ℕ := 68

-- Define the total fish Lucy wants to have
def total_fish : ℕ := current_fish + additional_fish

-- Theorem statement
theorem lucy_fish_count : total_fish = 280 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l989_98976


namespace NUMINAMATH_CALUDE_range_of_a_l989_98939

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) 
  (h2 : ∃ x : ℝ, x^2 + 4*x + a = 0) : 
  Real.exp 1 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l989_98939


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l989_98996

/-- 
Given a geometric sequence {a_n} with positive terms, 
if a_1 = 3 and S_3 = 21, then a_3 + a_4 + a_5 = 84.
-/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = a n * q) →  -- definition of geometric sequence
  a 1 = 3 →  -- first term
  (a 1 + a 2 + a 3 = 21) →  -- S_3 = 21
  (a 3 + a 4 + a 5 = 84) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l989_98996


namespace NUMINAMATH_CALUDE_fencemaker_problem_l989_98985

/-- Given a rectangular yard with one side of 40 feet and an area of 320 square feet,
    the perimeter minus one side equals 56 feet. -/
theorem fencemaker_problem (length width : ℝ) : 
  width = 40 ∧ 
  length * width = 320 → 
  2 * length + width = 56 :=
by sorry

end NUMINAMATH_CALUDE_fencemaker_problem_l989_98985


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l989_98920

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 84 →
  E = 4 * F + 18 →
  D + E + F = 180 →
  F = 15.6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l989_98920


namespace NUMINAMATH_CALUDE_total_goats_l989_98954

theorem total_goats (washington_goats : ℕ) (paddington_extra : ℕ) : 
  washington_goats = 180 → 
  paddington_extra = 70 → 
  washington_goats + (washington_goats + paddington_extra) = 430 := by
sorry

end NUMINAMATH_CALUDE_total_goats_l989_98954


namespace NUMINAMATH_CALUDE_cubic_root_exists_l989_98959

/-- Given a cubic expression ax³ - 2x + c with specific conditions, prove that x = 2 is a root -/
theorem cubic_root_exists (a c : ℝ) 
  (h1 : a * 1^3 - 2 * 1 + c = -5)
  (h2 : a * 4^3 - 2 * 4 + c = 52) :
  ∃ x : ℝ, x = 2 ∧ a * x^3 - 2 * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_exists_l989_98959


namespace NUMINAMATH_CALUDE_male_workers_percentage_l989_98937

theorem male_workers_percentage (female_workers : ℝ) (male_workers : ℝ) :
  male_workers = 0.6 * female_workers →
  (female_workers - male_workers) / female_workers = 0.4 :=
by
  sorry

end NUMINAMATH_CALUDE_male_workers_percentage_l989_98937


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l989_98940

theorem matrix_equation_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![43/7, -54/7; -33/14, 24/7]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l989_98940


namespace NUMINAMATH_CALUDE_total_cost_calculation_l989_98987

/-- The total cost of buying jerseys and basketballs -/
def total_cost (m n : ℝ) : ℝ := 8 * m + 5 * n

/-- Theorem: The total cost of buying 8 jerseys at m yuan each and 5 basketballs at n yuan each is 8m + 5n yuan -/
theorem total_cost_calculation (m n : ℝ) : 
  total_cost m n = 8 * m + 5 * n := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l989_98987


namespace NUMINAMATH_CALUDE_unique_solution_xy_l989_98945

/-- The unique solution to the system of equations x^y + 3 = y^x and 2x^y = y^x + 11 -/
theorem unique_solution_xy : ∃! (x y : ℕ+), 
  (x : ℝ) ^ (y : ℝ) + 3 = (y : ℝ) ^ (x : ℝ) ∧ 
  2 * (x : ℝ) ^ (y : ℝ) = (y : ℝ) ^ (x : ℝ) + 11 ∧
  x = 14 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_xy_l989_98945


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l989_98986

theorem system_of_equations_solution :
  let x : ℚ := -53/3
  let y : ℚ := -38/9
  (7 * x - 30 * y = 3) ∧ (3 * y - x = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l989_98986


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l989_98991

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  ((x^2 + a) / x^2 + (x^2 - a) / x^2) / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l989_98991


namespace NUMINAMATH_CALUDE_sqrt_of_negative_nine_l989_98971

theorem sqrt_of_negative_nine :
  (3 * Complex.I)^2 = -9 ∧ (-3 * Complex.I)^2 = -9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_negative_nine_l989_98971


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l989_98989

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + m = 0 ∧ y^2 + 2*y + m = 0) ↔ m < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l989_98989


namespace NUMINAMATH_CALUDE_length_of_AB_prime_l989_98998

/-- Given points A, B, and C, where A' and B' are on the line y=x, and AC and BC intersect at C,
    prove that the length of A'B' is 10√2/11 -/
theorem length_of_AB_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 7) →
  A'.1 = A'.2 →
  B'.1 = B'.2 →
  (C.2 - A.2) / (C.1 - A.1) = (A'.2 - A.2) / (A'.1 - A.1) →
  (C.2 - B.2) / (C.1 - B.1) = (B'.2 - B.2) / (B'.1 - B.1) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 10 * Real.sqrt 2 / 11 := by
  sorry

#check length_of_AB_prime

end NUMINAMATH_CALUDE_length_of_AB_prime_l989_98998


namespace NUMINAMATH_CALUDE_max_value_xyz_l989_98958

theorem max_value_xyz (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  x * y^2 * z^3 ≤ 1 / 432 :=
sorry

end NUMINAMATH_CALUDE_max_value_xyz_l989_98958


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l989_98924

theorem max_value_sum_of_roots (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) 
  (sum_constraint : a + b + c = 8) :
  (∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 8 ∧
    Real.sqrt (3 * x^2 + 1) + Real.sqrt (3 * y^2 + 1) + Real.sqrt (3 * z^2 + 1) > 
    Real.sqrt (3 * a^2 + 1) + Real.sqrt (3 * b^2 + 1) + Real.sqrt (3 * c^2 + 1)) ∨
  (Real.sqrt (3 * a^2 + 1) + Real.sqrt (3 * b^2 + 1) + Real.sqrt (3 * c^2 + 1) = Real.sqrt 201) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l989_98924


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l989_98984

theorem rational_inequality_solution (x : ℝ) : 
  (x^2 - 9) / (x^2 - 4) > 0 ∧ x ≠ 3 ↔ x ∈ Set.Ioi (-3) ∪ Set.Ioo (-2) 2 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l989_98984


namespace NUMINAMATH_CALUDE_leo_marbles_l989_98910

theorem leo_marbles (total_marbles : ℕ) (marbles_per_pack : ℕ) 
  (manny_fraction : ℚ) (neil_fraction : ℚ) : 
  total_marbles = 400 →
  marbles_per_pack = 10 →
  manny_fraction = 1/4 →
  neil_fraction = 1/8 →
  (total_marbles / marbles_per_pack : ℚ) * (1 - manny_fraction - neil_fraction) = 25 := by
  sorry

end NUMINAMATH_CALUDE_leo_marbles_l989_98910


namespace NUMINAMATH_CALUDE_max_distance_C_D_l989_98964

def C : Set ℂ := {z : ℂ | z^4 - 16 = 0}
def D : Set ℂ := {z : ℂ | z^3 - 27 = 0}

theorem max_distance_C_D : 
  ∃ (c : C) (d : D), ∀ (c' : C) (d' : D), Complex.abs (c - d) ≥ Complex.abs (c' - d') ∧ 
  Complex.abs (c - d) = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_max_distance_C_D_l989_98964


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_300_l989_98913

/-- Converts a natural number to its binary representation as a list of digits (0 or 1) --/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinaryAux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinaryAux (m / 2) ((m % 2) :: acc)
    toBinaryAux n []

/-- Sums a list of natural numbers --/
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

/-- The sum of digits in the binary representation of 300 is 3 --/
theorem sum_of_binary_digits_300 :
  sumList (toBinary 300) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_300_l989_98913


namespace NUMINAMATH_CALUDE_election_votes_calculation_l989_98982

theorem election_votes_calculation (total_votes : ℕ) : 
  (total_votes : ℝ) * 0.55 = (total_votes : ℝ) * 0.35 + 400 →
  total_votes = 2000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l989_98982


namespace NUMINAMATH_CALUDE_a_arithmetic_l989_98915

noncomputable def a (n : ℕ) : ℝ := sorry

noncomputable def S (n : ℕ) : ℝ := sorry

def q : ℝ := sorry

axiom q_neq_zero_one : q * (q - 1) ≠ 0

axiom sum_relation (n : ℕ) : (1 - q) * S n + q * a n = 1

axiom S_arithmetic : S 3 - S 9 = S 9 - S 6

theorem a_arithmetic : a 2 - a 8 = a 8 - a 5 := by sorry

end NUMINAMATH_CALUDE_a_arithmetic_l989_98915


namespace NUMINAMATH_CALUDE_farmers_market_sales_l989_98963

/-- The farmers' market sales problem -/
theorem farmers_market_sales
  (total_earnings : ℕ)
  (broccoli_sales : ℕ)
  (carrot_sales : ℕ)
  (spinach_sales : ℕ)
  (cauliflower_sales : ℕ)
  (h1 : total_earnings = 380)
  (h2 : broccoli_sales = 57)
  (h3 : carrot_sales = 2 * broccoli_sales)
  (h4 : spinach_sales = carrot_sales / 2 + 16)
  (h5 : total_earnings = broccoli_sales + carrot_sales + spinach_sales + cauliflower_sales) :
  cauliflower_sales = 136 := by
  sorry


end NUMINAMATH_CALUDE_farmers_market_sales_l989_98963


namespace NUMINAMATH_CALUDE_defective_ratio_for_given_shipment_l989_98923

/-- Given a shipment of chips with an expected number of defectives,
    calculate the ratio of defective chips to total chips -/
def defective_ratio (total : ℕ) (defective : ℕ) : ℚ :=
  defective / total

theorem defective_ratio_for_given_shipment :
  let total_chips : ℕ := 60000
  let expected_defective : ℕ := 15
  defective_ratio total_chips expected_defective = 1 / 4000 := by
  sorry

#eval defective_ratio 60000 15

end NUMINAMATH_CALUDE_defective_ratio_for_given_shipment_l989_98923


namespace NUMINAMATH_CALUDE_intersection_point_unique_l989_98942

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 1) / 6 = (y - 3) / 1 ∧ (y - 3) / 1 = (z + 5) / 3

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  3 * x - 2 * y + 5 * z - 3 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (7, 4, -2)

/-- Theorem stating that the intersection_point is the unique point satisfying both equations -/
theorem intersection_point_unique :
  line_equation intersection_point.1 intersection_point.2.1 intersection_point.2.2 ∧
  plane_equation intersection_point.1 intersection_point.2.1 intersection_point.2.2 ∧
  ∀ x y z : ℝ, line_equation x y z ∧ plane_equation x y z → (x, y, z) = intersection_point :=
by sorry


end NUMINAMATH_CALUDE_intersection_point_unique_l989_98942


namespace NUMINAMATH_CALUDE_problem_statement_l989_98977

/-- The number we're looking for -/
def x : ℝ := 640

/-- 50% of x is 190 more than 20% of 650 -/
theorem problem_statement : 0.5 * x = 0.2 * 650 + 190 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l989_98977


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l989_98957

theorem complex_modulus_squared (w : ℂ) (h : w + Complex.abs w = 4 + 5*I) : 
  Complex.abs w ^ 2 = 1681 / 64 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l989_98957


namespace NUMINAMATH_CALUDE_rational_root_condition_l989_98969

theorem rational_root_condition (n : ℕ+) :
  (∃ (x : ℚ), x^(n : ℕ) + (2 + x)^(n : ℕ) + (2 - x)^(n : ℕ) = 0) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_root_condition_l989_98969


namespace NUMINAMATH_CALUDE_unique_solution_l989_98966

theorem unique_solution (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_squares_eq : x^2 + y^2 + z^2 = 3)
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l989_98966


namespace NUMINAMATH_CALUDE_highest_score_is_179_l989_98967

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  totalInnings : ℕ
  averageScore : ℝ
  highestScore : ℝ
  lowestScore : ℝ
  averageExcludingExtremes : ℝ

/-- Theorem: Given the batsman's statistics, prove that the highest score is 179 runs -/
theorem highest_score_is_179 (stats : BatsmanStats)
  (h1 : stats.totalInnings = 46)
  (h2 : stats.averageScore = 60)
  (h3 : stats.highestScore - stats.lowestScore = 150)
  (h4 : stats.averageExcludingExtremes = 58) :
  stats.highestScore = 179 := by
  sorry

#check highest_score_is_179

end NUMINAMATH_CALUDE_highest_score_is_179_l989_98967


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l989_98950

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 96) :
  x^2 + y^2 = 1408 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l989_98950


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l989_98935

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) : 
  total = 36 → difference = 6 → 
  ∃ (girls boys : ℕ), 
    girls + boys = total ∧ 
    girls = boys + difference ∧
    girls * 5 = boys * 7 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l989_98935


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l989_98938

theorem midpoint_distance_theorem (t : ℝ) : 
  let A : ℝ × ℝ := (2*t - 3, t)
  let B : ℝ × ℝ := (t - 1, 2*t + 4)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ((midpoint.1 - A.1)^2 + (midpoint.2 - A.2)^2) = (t^2 + t) / 2 →
  t = -10 := by
sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l989_98938


namespace NUMINAMATH_CALUDE_two_digit_product_less_than_five_digits_l989_98936

theorem two_digit_product_less_than_five_digits : ∀ a b : ℕ, 
  10 ≤ a ∧ a ≤ 99 → 10 ≤ b ∧ b ≤ 99 → a * b < 10000 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_product_less_than_five_digits_l989_98936


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l989_98903

theorem purely_imaginary_complex_number (x : ℝ) :
  let z : ℂ := (x^2 - 1) + (x + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l989_98903


namespace NUMINAMATH_CALUDE_bens_old_car_cost_l989_98972

/-- Proves that Ben's old car cost $1900 given the problem conditions -/
theorem bens_old_car_cost :
  ∀ (old_car_cost new_car_cost : ℕ),
  new_car_cost = 2 * old_car_cost →
  old_car_cost = 1800 →
  new_car_cost = 1800 + 2000 →
  old_car_cost = 1900 := by
  sorry

#check bens_old_car_cost

end NUMINAMATH_CALUDE_bens_old_car_cost_l989_98972


namespace NUMINAMATH_CALUDE_seashell_points_sum_l989_98930

/-- The total points earned for seashells collected by Joan, Jessica, and Jeremy -/
def total_points (joan_shells : ℕ) (joan_points : ℕ) (jessica_shells : ℕ) (jessica_points : ℕ) (jeremy_shells : ℕ) (jeremy_points : ℕ) : ℕ :=
  joan_shells * joan_points + jessica_shells * jessica_points + jeremy_shells * jeremy_points

/-- Theorem stating that the total points earned is 48 -/
theorem seashell_points_sum :
  total_points 6 2 8 3 12 1 = 48 := by
  sorry

end NUMINAMATH_CALUDE_seashell_points_sum_l989_98930


namespace NUMINAMATH_CALUDE_quarter_sum_of_eighths_l989_98961

theorem quarter_sum_of_eighths :
  ∃ (n : ℕ), n > 0 ∧ (1 : ℚ) / 4 = (1 : ℚ) / n + (1 : ℚ) / n :=
by sorry

end NUMINAMATH_CALUDE_quarter_sum_of_eighths_l989_98961


namespace NUMINAMATH_CALUDE_total_fruits_shared_l989_98978

def persimmons_to_yuna : ℕ := 2
def apples_to_minyoung : ℕ := 7

theorem total_fruits_shared : persimmons_to_yuna + apples_to_minyoung = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_shared_l989_98978


namespace NUMINAMATH_CALUDE_remove_six_for_average_l989_98981

def original_list : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def removed_number : ℕ := 6

def remaining_list : List ℕ := original_list.filter (· ≠ removed_number)

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem remove_six_for_average :
  average remaining_list = 71/10 :=
sorry

end NUMINAMATH_CALUDE_remove_six_for_average_l989_98981


namespace NUMINAMATH_CALUDE_equation_solutions_l989_98952

theorem equation_solutions : 
  ∀ n m : ℕ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ 
  ((n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) ∨ (n = 6 ∧ m = 54) ∨ (n = 6 ∧ m = 27)) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l989_98952


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l989_98983

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l989_98983


namespace NUMINAMATH_CALUDE_smallest_with_digit_sum_41_plus_2021_l989_98905

def digit_sum (n : ℕ) : ℕ := sorry

def is_smallest_with_digit_sum (N : ℕ) (sum : ℕ) : Prop :=
  digit_sum N = sum ∧ ∀ m < N, digit_sum m ≠ sum

theorem smallest_with_digit_sum_41_plus_2021 :
  ∃ N : ℕ, is_smallest_with_digit_sum N 41 ∧ digit_sum (N + 2021) = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_digit_sum_41_plus_2021_l989_98905


namespace NUMINAMATH_CALUDE_society_of_beggars_voting_l989_98948

/-- The Society of Beggars voting problem -/
theorem society_of_beggars_voting (initial_for : ℕ) (initial_against : ℕ) (no_chair : ℕ) : 
  initial_for = 115 → 
  initial_against = 92 → 
  no_chair = 12 → 
  initial_for + initial_against + no_chair = 207 := by
sorry

end NUMINAMATH_CALUDE_society_of_beggars_voting_l989_98948


namespace NUMINAMATH_CALUDE_shopping_mall_probabilities_l989_98932

/-- Probability of purchasing product A -/
def prob_A : ℝ := 0.5

/-- Probability of purchasing product B -/
def prob_B : ℝ := 0.6

/-- Number of customers -/
def n : ℕ := 3

/-- Probability of purchasing at least one product -/
def p : ℝ := 0.8

theorem shopping_mall_probabilities :
  let prob_either := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B
  let prob_at_least_one := 1 - (1 - prob_A) * (1 - prob_B)
  let ξ := fun k => (n.choose k : ℝ) * p^k * (1 - p)^(n - k)
  (prob_either = 0.5) ∧
  (prob_at_least_one = 0.8) ∧
  (ξ 0 = 0.008) ∧
  (ξ 1 = 0.096) ∧
  (ξ 2 = 0.384) ∧
  (ξ 3 = 0.512) := by
  sorry

end NUMINAMATH_CALUDE_shopping_mall_probabilities_l989_98932


namespace NUMINAMATH_CALUDE_livestock_puzzle_l989_98933

theorem livestock_puzzle :
  ∃! (x y z : ℕ), 
    x + y + z = 100 ∧ 
    10 * x + 3 * y + (1/2) * z = 100 ∧
    x = 5 ∧ y = 1 ∧ z = 94 := by
  sorry

end NUMINAMATH_CALUDE_livestock_puzzle_l989_98933


namespace NUMINAMATH_CALUDE_picture_book_shelves_l989_98925

theorem picture_book_shelves (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ)
  (h1 : books_per_shelf = 9)
  (h2 : mystery_shelves = 6)
  (h3 : total_books = 72) :
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 := by
  sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l989_98925


namespace NUMINAMATH_CALUDE_race_result_l989_98970

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ → ℝ

/-- The race scenario -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner
  runner_c : Runner

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  r.distance = 100 ∧
  r.runner_a.speed > r.runner_b.speed ∧
  r.runner_b.speed > r.runner_c.speed ∧
  (∀ t, r.runner_a.position t = r.runner_a.speed * t) ∧
  (∀ t, r.runner_b.position t = r.runner_b.speed * t) ∧
  (∀ t, r.runner_c.position t = r.runner_c.speed * t) ∧
  (∃ t_a, r.runner_a.position t_a = r.distance ∧ r.runner_b.position t_a = r.distance - 10) ∧
  (∃ t_b, r.runner_b.position t_b = r.distance ∧ r.runner_c.position t_b = r.distance - 10)

/-- The theorem to be proved -/
theorem race_result (r : Race) (h : race_conditions r) :
  ∃ t, r.runner_a.position t = r.distance ∧ r.runner_c.position t = r.distance - 19 := by
  sorry

end NUMINAMATH_CALUDE_race_result_l989_98970


namespace NUMINAMATH_CALUDE_equation_solution_l989_98955

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 2 ∧ (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 3 :=
by
  use -1
  sorry

end NUMINAMATH_CALUDE_equation_solution_l989_98955


namespace NUMINAMATH_CALUDE_nine_eat_both_veg_and_non_veg_l989_98927

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_non_veg : ℕ
  total_veg : ℕ

/-- Calculates the number of people who eat both veg and non-veg -/
def both_veg_and_non_veg (f : FamilyDiet) : ℕ :=
  f.total_veg - f.only_veg

/-- Theorem stating that 9 people eat both veg and non-veg in the given family -/
theorem nine_eat_both_veg_and_non_veg (f : FamilyDiet)
    (h1 : f.only_veg = 11)
    (h2 : f.only_non_veg = 6)
    (h3 : f.total_veg = 20) :
    both_veg_and_non_veg f = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_eat_both_veg_and_non_veg_l989_98927


namespace NUMINAMATH_CALUDE_binary_sum_equals_11101101_l989_98941

/-- The sum of specific binary numbers equals 11101101₂ -/
theorem binary_sum_equals_11101101 :
  (0b10101 : Nat) + (0b11 : Nat) + (0b1010 : Nat) + (0b11100 : Nat) + (0b1101 : Nat) = (0b11101101 : Nat) := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_11101101_l989_98941


namespace NUMINAMATH_CALUDE_equation_solution_l989_98960

theorem equation_solution :
  ∃ x : ℝ, (16 : ℝ)^(x - 2) / (2 : ℝ)^(x - 2) = (32 : ℝ)^(3 * x) ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l989_98960


namespace NUMINAMATH_CALUDE_benny_seashells_l989_98906

/-- Represents the number of seashells Benny has after giving some to Jason -/
def seashells_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proves that Benny has 14 seashells remaining -/
theorem benny_seashells : seashells_remaining 66 52 = 14 := by
  sorry

end NUMINAMATH_CALUDE_benny_seashells_l989_98906


namespace NUMINAMATH_CALUDE_max_integer_k_for_distinct_roots_l989_98956

theorem max_integer_k_for_distinct_roots (k : ℤ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - (4*k - 2)*x + 4*k^2 = 0 ∧ y^2 - (4*k - 2)*y + 4*k^2 = 0) →
  k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_k_for_distinct_roots_l989_98956


namespace NUMINAMATH_CALUDE_negative_463_terminal_side_l989_98914

-- Define the concept of terminal side equality for angles
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

-- State the theorem
theorem negative_463_terminal_side :
  ∀ K : ℤ, same_terminal_side (-463) (K * 360 + 257) :=
sorry

end NUMINAMATH_CALUDE_negative_463_terminal_side_l989_98914


namespace NUMINAMATH_CALUDE_consecutive_integers_base_sum_l989_98975

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- Checks if a number is a valid base -/
def isValidBase (b : Nat) : Prop := b ≥ 2

theorem consecutive_integers_base_sum (C D : Nat) : 
  C.succ = D →
  isValidBase C →
  isValidBase D →
  isValidBase (C + D) →
  toBase10 [1, 4, 5] C + toBase10 [5, 6] D = toBase10 [9, 2] (C + D) →
  C + D = 11 := by
  sorry

#check consecutive_integers_base_sum

end NUMINAMATH_CALUDE_consecutive_integers_base_sum_l989_98975


namespace NUMINAMATH_CALUDE_diagonals_in_nonagon_l989_98999

/-- The number of diagonals in a regular nine-sided polygon -/
theorem diagonals_in_nonagon : 
  let n : ℕ := 9  -- number of sides
  let total_connections := n.choose 2  -- total number of connections between vertices
  let num_sides := n  -- number of sides (which are not diagonals)
  total_connections - num_sides = 27 := by sorry

end NUMINAMATH_CALUDE_diagonals_in_nonagon_l989_98999


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_bound_l989_98928

theorem sum_of_fourth_powers_bound (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 ≤ 1) :
  (a+b)^4 + (a+c)^4 + (a+d)^4 + (b+c)^4 + (b+d)^4 + (c+d)^4 ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_bound_l989_98928


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l989_98980

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l989_98980


namespace NUMINAMATH_CALUDE_largest_four_digit_number_l989_98988

def digits : Finset Nat := {5, 1, 6, 2, 4}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  (Finset.card (Finset.filter (λ d => d ∈ digits) (Finset.image (λ i => (n / 10^i) % 10) {0,1,2,3})) = 4)

theorem largest_four_digit_number :
  ∀ n : Nat, is_valid_number n → n ≤ 6542 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_number_l989_98988


namespace NUMINAMATH_CALUDE_inverse_log_inequality_l989_98990

theorem inverse_log_inequality (n : ℝ) (h1 : n ≥ 2) :
  (1 / Real.log n) > (1 / (n - 1) - 1 / (n + 1)) :=
by
  -- Proof goes here
  sorry

-- Given condition
axiom log_inequality (x : ℝ) (h : x > 1) : Real.log x < x - 1

end NUMINAMATH_CALUDE_inverse_log_inequality_l989_98990


namespace NUMINAMATH_CALUDE_determinant_maximum_value_l989_98979

open Real Matrix

theorem determinant_maximum_value (θ φ : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 1, 1 + sin θ, 1 + cos φ; 1 + cos θ, 1 + sin φ, 1]
  ∃ (θ' φ' : ℝ), ∀ (θ φ : ℝ), det A ≤ det (!![1, 1, 1; 1, 1 + sin θ', 1 + cos φ'; 1 + cos θ', 1 + sin φ', 1]) ∧
  det (!![1, 1, 1; 1, 1 + sin θ', 1 + cos φ'; 1 + cos θ', 1 + sin φ', 1]) = 1 :=
by sorry

end NUMINAMATH_CALUDE_determinant_maximum_value_l989_98979


namespace NUMINAMATH_CALUDE_smallest_integer_ending_in_3_divisible_by_11_l989_98901

theorem smallest_integer_ending_in_3_divisible_by_11 : ∃ n : ℕ, 
  (n % 10 = 3) ∧ (n % 11 = 0) ∧ (∀ m : ℕ, m < n → m % 10 = 3 → m % 11 ≠ 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_ending_in_3_divisible_by_11_l989_98901


namespace NUMINAMATH_CALUDE_fraction_equality_l989_98934

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l989_98934


namespace NUMINAMATH_CALUDE_adam_bought_seven_boxes_l989_98902

/-- The number of boxes Adam gave away -/
def boxes_given_away : ℕ := 7

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 6

/-- The number of pieces Adam still has -/
def remaining_pieces : ℕ := 36

/-- The number of boxes Adam bought initially -/
def initial_boxes : ℕ := 7

theorem adam_bought_seven_boxes :
  initial_boxes * pieces_per_box = boxes_given_away * pieces_per_box + remaining_pieces :=
by sorry

end NUMINAMATH_CALUDE_adam_bought_seven_boxes_l989_98902


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l989_98911

theorem stratified_sampling_proportion (second_year_total : ℕ) (third_year_total : ℕ) (third_year_sample : ℕ) :
  second_year_total = 1600 →
  third_year_total = 1400 →
  third_year_sample = 70 →
  (third_year_sample : ℚ) / third_year_total = 80 / second_year_total :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l989_98911


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l989_98917

def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l989_98917


namespace NUMINAMATH_CALUDE_median_list_i_is_eight_l989_98965

def list_i : List ℕ := [9, 2, 4, 7, 10, 11]
def list_ii : List ℕ := [3, 3, 4, 6, 7, 10]

def median (l : List ℕ) : ℚ := sorry
def mode (l : List ℕ) : ℕ := sorry

theorem median_list_i_is_eight :
  median list_i = 8 :=
by
  have h1 : median list_ii + mode list_ii = 8 := by sorry
  have h2 : median list_i = median list_ii + mode list_ii := by sorry
  sorry

end NUMINAMATH_CALUDE_median_list_i_is_eight_l989_98965


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l989_98916

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 4
def q (x : ℝ) : Prop := x^2 - 5*x + 4 ≥ 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l989_98916


namespace NUMINAMATH_CALUDE_range_of_m_l989_98995

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (2 / (x + 1) > 1) → (m ≤ x ∧ x ≤ 2)) →
  (∀ x : ℝ, (2 / (x + 1) > 1) → x ≤ 1) →
  m ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l989_98995


namespace NUMINAMATH_CALUDE_area_of_region_l989_98997

/-- The region defined by the inequality |4x-14| + |3y-9| ≤ 6 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |4 * p.1 - 14| + |3 * p.2 - 9| ≤ 6}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The area of the region defined by |4x-14| + |3y-9| ≤ 6 is 6 -/
theorem area_of_region : area Region = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_region_l989_98997


namespace NUMINAMATH_CALUDE_max_area_rectangle_l989_98949

theorem max_area_rectangle (l w : ℕ) : 
  (2 * (l + w) = 120) →  -- perimeter condition
  (∀ a b : ℕ, 2 * (a + b) = 120 → l * w ≥ a * b) →  -- maximum area condition
  l * w = 900 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l989_98949


namespace NUMINAMATH_CALUDE_rectangle_to_square_possible_l989_98907

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents a rectangular piece cut from the original rectangle -/
structure Piece where
  length : ℕ
  width : ℕ

def Rectangle.area (r : Rectangle) : ℕ := r.length * r.width

def Square.area (s : Square) : ℕ := s.side * s.side

def can_form_square (r : Rectangle) (s : Square) (pieces : List Piece) : Prop :=
  r.area = s.area ∧
  (pieces.foldl (fun acc p => acc + p.length * p.width) 0 = r.area) ∧
  (∀ p ∈ pieces, p.length ≤ r.length ∧ p.width ≤ r.width)

theorem rectangle_to_square_possible (r : Rectangle) (h1 : r.length = 16) (h2 : r.width = 9) :
  ∃ (s : Square) (pieces : List Piece), can_form_square r s pieces ∧ pieces.length ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_possible_l989_98907


namespace NUMINAMATH_CALUDE_max_weight_proof_l989_98912

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 5

/-- The minimum weight of each crate in kg -/
def min_crate_weight : ℕ := 120

/-- The maximum weight of crates on a single trip in kg -/
def max_trip_weight : ℕ := max_crates * min_crate_weight

theorem max_weight_proof :
  max_trip_weight = 600 := by
  sorry

end NUMINAMATH_CALUDE_max_weight_proof_l989_98912


namespace NUMINAMATH_CALUDE_two_numbers_problem_l989_98931

theorem two_numbers_problem (a b : ℕ) :
  a + b = 667 →
  Nat.lcm a b / Nat.gcd a b = 120 →
  ((a = 115 ∧ b = 552) ∨ (a = 552 ∧ b = 115)) ∨
  ((a = 232 ∧ b = 435) ∨ (a = 435 ∧ b = 232)) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l989_98931


namespace NUMINAMATH_CALUDE_phone_answer_probability_l989_98962

theorem phone_answer_probability : 
  let p1 : ℚ := 1/10  -- Probability of answering on the first ring
  let p2 : ℚ := 3/10  -- Probability of answering on the second ring
  let p3 : ℚ := 2/5   -- Probability of answering on the third ring
  let p4 : ℚ := 1/10  -- Probability of answering on the fourth ring
  p1 + p2 + p3 + p4 = 9/10 := by
sorry

end NUMINAMATH_CALUDE_phone_answer_probability_l989_98962


namespace NUMINAMATH_CALUDE_ricky_rose_distribution_l989_98904

/-- Calculates the number of roses each person receives when Ricky distributes his roses. -/
def roses_per_person (initial_roses : ℕ) (stolen_roses : ℕ) (num_people : ℕ) : ℕ :=
  (initial_roses - stolen_roses) / num_people

/-- Theorem: Given the problem conditions, each person will receive 4 roses. -/
theorem ricky_rose_distribution : roses_per_person 40 4 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ricky_rose_distribution_l989_98904


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_l989_98929

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 + 16 * x + 1

-- State the theorem
theorem range_of_f_on_interval :
  let a := 1
  let b := 2
  (∀ x ≤ -2, ∀ y ∈ Set.Ioo x (-2), f x ≥ f y) →
  (∀ x ≥ -2, ∀ y ∈ Set.Ioo (-2) x, f x ≥ f y) →
  Set.range (fun x ↦ f x) ∩ Set.Icc a b = Set.Icc (f a) (f b) :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_l989_98929


namespace NUMINAMATH_CALUDE_water_after_40_days_l989_98973

/-- Calculates the remaining water in a trough after a given number of days -/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (days : ℝ) : ℝ :=
  initial_amount - evaporation_rate * days

/-- Theorem stating that given the initial conditions, the remaining water after 40 days is 270 gallons -/
theorem water_after_40_days :
  let initial_amount : ℝ := 300
  let evaporation_rate : ℝ := 0.75
  let days : ℝ := 40
  remaining_water initial_amount evaporation_rate days = 270 := by
sorry

#eval remaining_water 300 0.75 40

end NUMINAMATH_CALUDE_water_after_40_days_l989_98973


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_roots_difference_condition_specific_km_values_l989_98947

theorem quadratic_roots_properties (x₁ x₂ k m : ℝ) : 
  x₁ + x₂ + x₁ * x₂ = 2 * m + k ∧ 
  (x₁ - 1) * (x₂ - 1) = m + 1 - k →
  ∃ (p q : ℝ), x₁ * x₂ = q ∧ x₁ + x₂ = -p ∧ p = k ∧ q = m :=
by sorry

theorem roots_difference_condition (k m : ℝ) (x₁ x₂ : ℝ) :
  x₁ - x₂ = 1 ∧ x₁ + x₂ = k ∧ x₁ * x₂ = m →
  k^2 = 4 * m + 1 :=
by sorry

theorem specific_km_values (k m : ℝ) :
  k - m = 1 ∧ k^2 = 4 * m + 1 →
  (m = 0 ∧ k = 1) ∨ (m = 2 ∧ k = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_roots_difference_condition_specific_km_values_l989_98947


namespace NUMINAMATH_CALUDE_unique_integer_between_sqrt5_and_sqrt15_l989_98944

theorem unique_integer_between_sqrt5_and_sqrt15 :
  ∃! n : ℤ, (Real.sqrt 5 < n) ∧ (n < Real.sqrt 15) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_between_sqrt5_and_sqrt15_l989_98944


namespace NUMINAMATH_CALUDE_unique_score_above_90_l989_98994

/-- Represents the scoring system for the exam -/
structure ScoringSystem where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ

/-- Calculates the score given the number of correct and incorrect answers -/
def calculate_score (system : ScoringSystem) (correct : ℕ) (incorrect : ℕ) : ℤ :=
  system.correct_points * correct + system.incorrect_points * incorrect

/-- Checks if a score uniquely determines the number of correct and incorrect answers -/
def is_unique_score (system : ScoringSystem) (score : ℤ) : Prop :=
  ∃! (correct incorrect : ℕ),
    correct + incorrect ≤ system.total_questions ∧
    calculate_score system correct incorrect = score

/-- The main theorem to prove -/
theorem unique_score_above_90 (system : ScoringSystem) : 
  system.total_questions = 35 →
  system.correct_points = 5 →
  system.incorrect_points = -2 →
  (∀ s, s > 90 ∧ s < 116 → ¬is_unique_score system s) →
  is_unique_score system 116 := 
by sorry

end NUMINAMATH_CALUDE_unique_score_above_90_l989_98994


namespace NUMINAMATH_CALUDE_square_of_one_plus_i_l989_98918

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem square_of_one_plus_i : (1 + i)^2 = 2*i := by sorry

end NUMINAMATH_CALUDE_square_of_one_plus_i_l989_98918


namespace NUMINAMATH_CALUDE_sum_of_compositions_l989_98943

def r (x : ℝ) : ℝ := |x + 1| - 3

def s (x : ℝ) : ℝ := -|x + 2|

def evaluation_points : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3]

theorem sum_of_compositions :
  (evaluation_points.map (fun x => s (r x))).sum = -37 := by sorry

end NUMINAMATH_CALUDE_sum_of_compositions_l989_98943


namespace NUMINAMATH_CALUDE_absolute_value_condition_l989_98921

theorem absolute_value_condition (x : ℝ) :
  (∀ x, x < -2 → |x| > 2) ∧ 
  (∃ x, |x| > 2 ∧ x ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_condition_l989_98921


namespace NUMINAMATH_CALUDE_min_value_ab_l989_98968

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 4/b = Real.sqrt (a*b)) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 4/y = Real.sqrt (x*y) → a*b ≤ x*y :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l989_98968


namespace NUMINAMATH_CALUDE_fifth_number_in_first_set_l989_98993

theorem fifth_number_in_first_set (x : ℝ) (fifth_number : ℝ) : 
  ((28 + x + 70 + 88 + fifth_number) / 5 = 67) →
  ((50 + 62 + 97 + 124 + x) / 5 = 75.6) →
  fifth_number = 104 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_in_first_set_l989_98993


namespace NUMINAMATH_CALUDE_unique_student_count_l989_98974

theorem unique_student_count : ∃! n : ℕ, 
  100 < n ∧ n < 200 ∧ 
  ∃ k : ℕ, n = 4 * k + 1 ∧
  ∃ m : ℕ, n = 3 * m + 2 ∧
  ∃ l : ℕ, n = 7 * l + 3 ∧
  n = 101 :=
sorry

end NUMINAMATH_CALUDE_unique_student_count_l989_98974


namespace NUMINAMATH_CALUDE_smallest_nonnegative_value_l989_98909

theorem smallest_nonnegative_value (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ x ≤ y ∧
    Real.sqrt (2 * p : ℝ) - Real.sqrt x - Real.sqrt y ≥ 0 ∧
    ∀ (a b : ℕ), a > 0 → b > 0 → a ≤ b →
      Real.sqrt (2 * p : ℝ) - Real.sqrt a - Real.sqrt b ≥ 0 →
      Real.sqrt (2 * p : ℝ) - Real.sqrt a - Real.sqrt b ≥ 
      Real.sqrt (2 * p : ℝ) - Real.sqrt x - Real.sqrt y ∧
    x = (p - 1) / 2 ∧ y = (p + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonnegative_value_l989_98909


namespace NUMINAMATH_CALUDE_right_triangle_with_53_hypotenuse_l989_98951

theorem right_triangle_with_53_hypotenuse (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 53 →           -- Hypotenuse is 53
  b = a + 1 →        -- Legs are consecutive integers
  a + b = 75 :=      -- Sum of legs is 75
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_53_hypotenuse_l989_98951


namespace NUMINAMATH_CALUDE_sqrt_5_is_simplest_l989_98992

/-- A quadratic radical is considered simplest if it cannot be simplified further
    and does not have denominators under the square root. -/
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, x = Real.sqrt y → (∀ z : ℝ, y ≠ z^2) ∧ (∀ n : ℕ, n > 1 → y ≠ Real.sqrt n)

/-- The theorem states that √5 is the simplest quadratic radical among the given options. -/
theorem sqrt_5_is_simplest :
  is_simplest_quadratic_radical (Real.sqrt 5) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 8) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (a^2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (0.2 * b)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_5_is_simplest_l989_98992


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l989_98908

theorem pure_imaginary_modulus (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 9) (m^2 + 2*m - 3)
  (Complex.re z = 0 ∧ Complex.im z ≠ 0) → Complex.abs z = 12 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l989_98908


namespace NUMINAMATH_CALUDE_hiking_rate_up_l989_98900

/-- Hiking problem statement -/
theorem hiking_rate_up (total_time : ℝ) (time_up : ℝ) (rate_down : ℝ) :
  total_time = 3 →
  time_up = 1.2 →
  rate_down = 6 →
  ∃ (rate_up : ℝ), rate_up = 9 ∧ rate_up * time_up = rate_down * (total_time - time_up) :=
by
  sorry

end NUMINAMATH_CALUDE_hiking_rate_up_l989_98900


namespace NUMINAMATH_CALUDE_max_k_value_l989_98946

theorem max_k_value : ∃ (k : ℕ) (A B C : ℕ), 
  A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ 
  (100 * A + 10 * C + B) = k * (10 * A + B) ∧ 
  (∀ (k' : ℕ) (A' B' C' : ℕ), 
    A' ≠ 0 → A' < 10 → B' < 10 → C' < 10 → 
    (100 * A' + 10 * C' + B') = k' * (10 * A' + B') → 
    k' ≤ k) ∧
  k = 19 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l989_98946


namespace NUMINAMATH_CALUDE_range_of_a_l989_98926

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a < 0) →
  (∀ x, ¬(p x a) → ¬(q x)) →
  (∃ x, ¬(p x a) ∧ (q x)) →
  -2/3 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l989_98926


namespace NUMINAMATH_CALUDE_percentage_relation_l989_98953

theorem percentage_relation (A B C x y : ℝ) 
  (h1 : A > 0) (h2 : B > 0) (h3 : C > 0)
  (h4 : A > B) (h5 : B > C)
  (h6 : A = B * (1 + x / 100))
  (h7 : B = C * (1 + y / 100)) : 
  x = 100 * (A / (C * (1 + y / 100)) - 1) := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l989_98953
