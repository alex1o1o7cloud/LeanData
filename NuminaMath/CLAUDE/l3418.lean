import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_focal_length_l3418_341806

/-- The focal length of an ellipse with given properties -/
theorem ellipse_focal_length (k : ℝ) : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2 - y^2 / k = 1}
  let focus_on_x_axis := true  -- This is a simplification, as we can't directly represent this in Lean
  let eccentricity := (1 : ℝ) / 2
  let focal_length := 1
  (∀ (x y : ℝ), (x, y) ∈ ellipse → x^2 - y^2 / k = 1) ∧ 
  focus_on_x_axis ∧ 
  eccentricity = 1 / 2 →
  focal_length = 1 := by
sorry


end NUMINAMATH_CALUDE_ellipse_focal_length_l3418_341806


namespace NUMINAMATH_CALUDE_banana_bread_ratio_l3418_341872

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves made on Monday -/
def loaves_monday : ℕ := 3

/-- The total number of bananas used for both days -/
def total_bananas : ℕ := 36

/-- The number of loaves made on Tuesday -/
def loaves_tuesday : ℕ := (total_bananas - loaves_monday * bananas_per_loaf) / bananas_per_loaf

theorem banana_bread_ratio :
  loaves_tuesday / loaves_monday = 2 := by sorry

end NUMINAMATH_CALUDE_banana_bread_ratio_l3418_341872


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l3418_341834

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 3) ^ 2 - 3 * (a 3) + 2 = 0 →
  (a 7) ^ 2 - 3 * (a 7) + 2 = 0 →
  a 5 = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l3418_341834


namespace NUMINAMATH_CALUDE_unique_divisor_square_sum_l3418_341860

theorem unique_divisor_square_sum (p n : ℕ) (hp : p.Prime) (hp2 : p > 2) (hn : n > 0) :
  ∃! d : ℕ, d > 0 ∧ d ∣ (p * n^2) ∧ ∃ k : ℕ, n^2 + d = k^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisor_square_sum_l3418_341860


namespace NUMINAMATH_CALUDE_intersection_M_N_l3418_341894

def M : Set ℝ := {0, 1, 2}

def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3418_341894


namespace NUMINAMATH_CALUDE_complement_union_problem_l3418_341829

theorem complement_union_problem (U A B : Set Nat) : 
  U = {1, 2, 3, 4} →
  A = {1, 2} →
  B = {2, 3} →
  (Aᶜ ∪ B) = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l3418_341829


namespace NUMINAMATH_CALUDE_complex_cube_root_unity_l3418_341843

theorem complex_cube_root_unity (i : ℂ) (x : ℂ) : 
  i^2 = -1 → 
  x = (-1 + i * Real.sqrt 3) / 2 → 
  1 / (x^3 - x) = -1/2 + (i * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_unity_l3418_341843


namespace NUMINAMATH_CALUDE_slab_rate_calculation_l3418_341891

/-- Given a room with specific dimensions and total flooring cost, 
    prove that the rate per square meter for slabs is as calculated. -/
theorem slab_rate_calculation (length width total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : width = 3.75)
    (h3 : total_cost = 12375) : 
  total_cost / (length * width) = 600 := by
  sorry

end NUMINAMATH_CALUDE_slab_rate_calculation_l3418_341891


namespace NUMINAMATH_CALUDE_at_least_one_triangle_l3418_341821

/-- Given 2n points (n ≥ 2) and n^2 + 1 segments, at least one triangle is formed. -/
theorem at_least_one_triangle (n : ℕ) (h : n ≥ 2) :
  ∃ (points : Finset (ℝ × ℝ × ℝ)) (segments : Finset (Fin 2 → ℝ × ℝ × ℝ)),
    Finset.card points = 2 * n ∧
    Finset.card segments = n^2 + 1 ∧
    ∃ (a b c : ℝ × ℝ × ℝ),
      a ∈ points ∧ b ∈ points ∧ c ∈ points ∧
      (λ i => if i = 0 then a else b) ∈ segments ∧
      (λ i => if i = 0 then b else c) ∈ segments ∧
      (λ i => if i = 0 then c else a) ∈ segments :=
by
  sorry


end NUMINAMATH_CALUDE_at_least_one_triangle_l3418_341821


namespace NUMINAMATH_CALUDE_adam_has_ten_apples_l3418_341884

def apples_problem (jackie_apples adam_more_apples : ℕ) : Prop :=
  let adam_apples := jackie_apples + adam_more_apples
  adam_apples = 10

theorem adam_has_ten_apples :
  apples_problem 2 8 := by sorry

end NUMINAMATH_CALUDE_adam_has_ten_apples_l3418_341884


namespace NUMINAMATH_CALUDE_geometric_series_calculation_l3418_341870

theorem geometric_series_calculation : 
  2016 * (1 / (1 + 1/2 + 1/4 + 1/8 + 1/16 + 1/32)) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_calculation_l3418_341870


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3418_341867

theorem largest_multiple_of_15_under_500 : ∃ n : ℕ, n * 15 = 495 ∧ 
  495 < 500 ∧ 
  (∀ m : ℕ, m * 15 < 500 → m * 15 ≤ 495) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3418_341867


namespace NUMINAMATH_CALUDE_base_6_addition_subtraction_l3418_341801

/-- Converts a base 6 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Theorem: The sum of 555₆ and 65₆ minus 11₆ equals 1053₆ in base 6 --/
theorem base_6_addition_subtraction :
  to_base_6 (to_base_10 [5, 5, 5] + to_base_10 [5, 6] - to_base_10 [1, 1]) = [3, 5, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_6_addition_subtraction_l3418_341801


namespace NUMINAMATH_CALUDE_fourth_power_sum_l3418_341824

theorem fourth_power_sum (x y : ℝ) (h1 : x * y = 4) (h2 : x - y = 2) :
  x^4 + y^4 = 112 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l3418_341824


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3418_341840

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℝ) -- Sum function for the arithmetic sequence
  (h1 : S 2 = 4) -- Given S_2 = 4
  (h2 : S 4 = 20) -- Given S_4 = 20
  : ∃ (a₁ d : ℝ), 
    (∀ n : ℕ, S n = n * (2 * a₁ + (n - 1) * d) / 2) ∧ 
    d = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3418_341840


namespace NUMINAMATH_CALUDE_sugar_package_weight_l3418_341832

theorem sugar_package_weight (x : ℝ) 
  (h1 : x > 0)
  (h2 : (4 * x - 10) / (x + 10) = 7 / 8) :
  4 * x + x = 30 := by
  sorry

end NUMINAMATH_CALUDE_sugar_package_weight_l3418_341832


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3418_341844

theorem imaginary_part_of_complex_number :
  let z : ℂ := 1 - 2*I
  Complex.im z = -2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3418_341844


namespace NUMINAMATH_CALUDE_target_hit_probability_l3418_341815

theorem target_hit_probability (p1 p2 : ℝ) (h1 : p1 = 0.8) (h2 : p2 = 0.7) :
  1 - (1 - p1) * (1 - p2) = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l3418_341815


namespace NUMINAMATH_CALUDE_joe_cars_count_l3418_341807

theorem joe_cars_count (initial_cars new_cars : ℕ) 
  (h1 : initial_cars = 50) 
  (h2 : new_cars = 12) : 
  initial_cars + new_cars = 62 := by
  sorry

end NUMINAMATH_CALUDE_joe_cars_count_l3418_341807


namespace NUMINAMATH_CALUDE_dress_shop_inventory_l3418_341880

theorem dress_shop_inventory (total_space : ℕ) (blue_extra : ℕ) (red_dresses : ℕ) 
  (h1 : total_space = 200)
  (h2 : blue_extra = 34)
  (h3 : red_dresses + (red_dresses + blue_extra) = total_space) :
  red_dresses = 83 := by
sorry

end NUMINAMATH_CALUDE_dress_shop_inventory_l3418_341880


namespace NUMINAMATH_CALUDE_benny_picked_two_apples_l3418_341889

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The difference between the number of apples Dan and Benny picked -/
def difference : ℕ := 7

/-- The number of apples Benny picked -/
def benny_apples : ℕ := dan_apples - difference

theorem benny_picked_two_apples : benny_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_benny_picked_two_apples_l3418_341889


namespace NUMINAMATH_CALUDE_system_solution_l3418_341874

theorem system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (eq1 : 2*x₁ + x₂ + x₃ + x₄ + x₅ = 6)
  (eq2 : x₁ + 2*x₂ + x₃ + x₄ + x₅ = 12)
  (eq3 : x₁ + x₂ + 2*x₃ + x₄ + x₅ = 24)
  (eq4 : x₁ + x₂ + x₃ + 2*x₄ + x₅ = 48)
  (eq5 : x₁ + x₂ + x₃ + x₄ + 2*x₅ = 96) :
  3*x₄ + 2*x₅ = 181 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3418_341874


namespace NUMINAMATH_CALUDE_amount_of_c_l3418_341858

/-- Given four people a, b, c, and d with monetary amounts, prove that c has 500 units of currency. -/
theorem amount_of_c (a b c d : ℕ) : 
  a + b + c + d = 1800 →
  a + c = 500 →
  b + c = 900 →
  a + d = 700 →
  a + b + d = 1300 →
  c = 500 := by
  sorry

end NUMINAMATH_CALUDE_amount_of_c_l3418_341858


namespace NUMINAMATH_CALUDE_max_triangles_from_lines_l3418_341813

/-- Given 2017 lines separated into three sets such that lines in the same set are parallel to each other,
    prove that the largest possible number of triangles that can be formed with vertices on these lines
    is 673 * 672^2. -/
theorem max_triangles_from_lines (total_lines : ℕ) (set1 set2 set3 : ℕ) :
  total_lines = 2017 →
  set1 + set2 + set3 = total_lines →
  set1 ≥ set2 →
  set2 ≥ set3 →
  set1 * set2 * set3 ≤ 673 * 672 * 672 :=
by sorry

end NUMINAMATH_CALUDE_max_triangles_from_lines_l3418_341813


namespace NUMINAMATH_CALUDE_ladder_slip_distance_l3418_341814

/-- The distance the top of a ladder slips down when its bottom moves from 5 feet to 10.658966865741546 feet away from a wall. -/
theorem ladder_slip_distance (ladder_length : Real) (initial_distance : Real) (final_distance : Real) :
  ladder_length = 14 →
  initial_distance = 5 →
  final_distance = 10.658966865741546 →
  let initial_height := Real.sqrt (ladder_length^2 - initial_distance^2)
  let final_height := Real.sqrt (ladder_length^2 - final_distance^2)
  abs ((initial_height - final_height) - 4.00392512594753) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_ladder_slip_distance_l3418_341814


namespace NUMINAMATH_CALUDE_addition_of_like_terms_l3418_341853

theorem addition_of_like_terms (a : ℝ) : a + 2*a = 3*a := by
  sorry

end NUMINAMATH_CALUDE_addition_of_like_terms_l3418_341853


namespace NUMINAMATH_CALUDE_cubic_roots_sum_squares_l3418_341846

theorem cubic_roots_sum_squares (a b c : ℝ) : 
  (3 * a^3 - 4 * a^2 + 100 * a - 3 = 0) →
  (3 * b^3 - 4 * b^2 + 100 * b - 3 = 0) →
  (3 * c^3 - 4 * c^2 + 100 * c - 3 = 0) →
  (a + b + 2)^2 + (b + c + 2)^2 + (c + a + 2)^2 = 1079/9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_squares_l3418_341846


namespace NUMINAMATH_CALUDE_math_city_intersections_l3418_341848

/-- Represents a city with straight streets -/
structure City where
  num_streets : ℕ
  no_parallel : Bool
  no_triple_intersections : Bool

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  (c.num_streets.choose 2)

/-- Theorem: A city with 10 streets meeting the given conditions has 45 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 10 → c.no_parallel → c.no_triple_intersections →
  num_intersections c = 45 := by
  sorry

#check math_city_intersections

end NUMINAMATH_CALUDE_math_city_intersections_l3418_341848


namespace NUMINAMATH_CALUDE_product_plus_one_l3418_341878

theorem product_plus_one (m n : ℕ) (h : m * n = 121) : (m + 1) * (n + 1) = 144 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_l3418_341878


namespace NUMINAMATH_CALUDE_expression_simplification_l3418_341886

theorem expression_simplification (m : ℝ) 
  (h1 : (m + 2) * (m - 3) = 0) 
  (h2 : m ≠ 3) : 
  ((m^2 - 9) / (m^2 - 6*m + 9) - 3 / (m - 3)) / (m^2 / m^3) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3418_341886


namespace NUMINAMATH_CALUDE_birds_on_fence_l3418_341895

/-- The total number of birds on a fence given initial birds, additional birds, and additional storks -/
def total_birds (initial : ℕ) (additional : ℕ) (storks : ℕ) : ℕ :=
  initial + additional + storks

/-- Theorem stating that given 6 initial birds, 4 additional birds, and 8 additional storks, 
    the total number of birds on the fence is 18 -/
theorem birds_on_fence : total_birds 6 4 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3418_341895


namespace NUMINAMATH_CALUDE_sum_of_square_roots_lower_bound_l3418_341842

theorem sum_of_square_roots_lower_bound
  (a b c d e : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) +
  Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) +
  Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_lower_bound_l3418_341842


namespace NUMINAMATH_CALUDE_remainder_3005_div_98_l3418_341800

theorem remainder_3005_div_98 : 3005 % 98 = 65 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3005_div_98_l3418_341800


namespace NUMINAMATH_CALUDE_exponent_rules_l3418_341828

theorem exponent_rules (a b : ℝ) : 
  ((-b)^2 * (-b)^3 * (-b)^5 = b^10) ∧ ((2*a*b^2)^3 = 8*a^3*b^6) := by
  sorry

end NUMINAMATH_CALUDE_exponent_rules_l3418_341828


namespace NUMINAMATH_CALUDE_no_m_exists_for_equality_m_range_for_subset_l3418_341826

-- Define the set P
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}

-- Define the set S parameterized by m
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem 1: There does not exist an m such that P = S(m)
theorem no_m_exists_for_equality : ¬∃ m : ℝ, P = S m := by
  sorry

-- Theorem 2: The set of m such that P ⊆ S(m) is {m | m ≤ 3}
theorem m_range_for_subset : {m : ℝ | P ⊆ S m} = {m : ℝ | m ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_no_m_exists_for_equality_m_range_for_subset_l3418_341826


namespace NUMINAMATH_CALUDE_olaf_water_requirement_l3418_341839

/-- Calculates the total water needed for a sailing trip -/
def water_needed_for_trip (crew_size : ℕ) (water_per_man_per_day : ℚ) 
  (boat_speed : ℕ) (total_distance : ℕ) : ℚ :=
  let trip_duration := total_distance / boat_speed
  let daily_water_requirement := crew_size * water_per_man_per_day
  daily_water_requirement * trip_duration

/-- Theorem: The total water needed for Olaf's sailing trip is 250 gallons -/
theorem olaf_water_requirement : 
  water_needed_for_trip 25 (1/2) 200 4000 = 250 := by
  sorry

end NUMINAMATH_CALUDE_olaf_water_requirement_l3418_341839


namespace NUMINAMATH_CALUDE_no_common_terms_except_one_l3418_341805

def x : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n+2) => x (n+1) + 2 * x n

def y : ℕ → ℕ
  | 0 => 1
  | 1 => 7
  | (n+2) => 2 * y (n+1) + 3 * y n

theorem no_common_terms_except_one (n : ℕ) (m : ℕ) (h : n ≥ 1) :
  x n ≠ y m ∨ (x n = y m ∧ n = 0 ∧ m = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_terms_except_one_l3418_341805


namespace NUMINAMATH_CALUDE_third_sample_is_43_l3418_341899

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (start : ℕ) (n : ℕ) : ℕ :=
  start + (n - 1) * (total / sample_size)

/-- Theorem for the specific problem -/
theorem third_sample_is_43 
  (total : ℕ) (sample_size : ℕ) (start : ℕ) 
  (h1 : total = 900) 
  (h2 : sample_size = 50) 
  (h3 : start = 7) :
  systematic_sample total sample_size start 3 = 43 := by
  sorry

#eval systematic_sample 900 50 7 3

end NUMINAMATH_CALUDE_third_sample_is_43_l3418_341899


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l3418_341808

theorem cone_sphere_volume_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 * π * r^2 * h) = (1 / 3) * (4 / 3 * π * r^3) → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l3418_341808


namespace NUMINAMATH_CALUDE_samias_walking_distance_l3418_341817

/-- Represents the problem of calculating Samia's walking distance --/
theorem samias_walking_distance
  (total_time : ℝ)
  (bike_speed : ℝ)
  (walk_speed : ℝ)
  (wait_time : ℝ)
  (h_total_time : total_time = 1.25)  -- 1 hour and 15 minutes
  (h_bike_speed : bike_speed = 20)
  (h_walk_speed : walk_speed = 4)
  (h_wait_time : wait_time = 0.25)  -- 15 minutes
  : ∃ (total_distance : ℝ),
    let bike_distance := total_distance / 3
    let walk_distance := 2 * total_distance / 3
    bike_distance / bike_speed + wait_time + walk_distance / walk_speed = total_time ∧
    (walk_distance ≥ 3.55 ∧ walk_distance ≤ 3.65) :=
by
  sorry

#check samias_walking_distance

end NUMINAMATH_CALUDE_samias_walking_distance_l3418_341817


namespace NUMINAMATH_CALUDE_quadratic_equation_factorization_l3418_341885

theorem quadratic_equation_factorization (n : ℝ) :
  (∃ m : ℝ, ∀ x : ℝ, x^2 - 4*x + 1 = n ↔ (x - m)^2 = 5) →
  n = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_factorization_l3418_341885


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3418_341869

theorem fractional_equation_solution (x : ℝ) (h : x ≠ 1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3418_341869


namespace NUMINAMATH_CALUDE_expression_equals_half_y_l3418_341803

theorem expression_equals_half_y (y d : ℝ) (hy : y > 0) : 
  (4 * y) / 20 + (3 * y) / d = 0.5 * y → d = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_half_y_l3418_341803


namespace NUMINAMATH_CALUDE_cubic_root_of_unity_solutions_l3418_341865

theorem cubic_root_of_unity_solutions (p q r s : ℂ) (m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (p * m^2 + q * m + r = 0) ∧ (q * m^2 + r * m + s = 0) →
  (m = 1) ∨ (m = Complex.exp ((2 * Real.pi * Complex.I) / 3)) ∨ (m = Complex.exp ((-2 * Real.pi * Complex.I) / 3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_of_unity_solutions_l3418_341865


namespace NUMINAMATH_CALUDE_sequence_sum_l3418_341893

theorem sequence_sum (n : ℕ) (x : ℕ → ℚ) : 
  (∀ k ∈ Finset.range (n - 1), x (k + 1) = x k + 1 / 3) →
  x 1 = 2 →
  n > 0 →
  Finset.sum (Finset.range n) (λ i => x (i + 1)) = n * (n + 11) / 6 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_l3418_341893


namespace NUMINAMATH_CALUDE_angle_c_measure_l3418_341890

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the properties of the isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.A = t.B

-- Define the relationship between angles A and C
def AngleCRelation (t : Triangle) : Prop :=
  t.C = t.A + 30

-- Define the sum of angles in a triangle
def AngleSum (t : Triangle) : Prop :=
  t.A + t.B + t.C = 180

-- Theorem statement
theorem angle_c_measure (t : Triangle) 
  (h1 : IsIsosceles t) 
  (h2 : AngleCRelation t) 
  (h3 : AngleSum t) : 
  t.C = 80 := by
    sorry

end NUMINAMATH_CALUDE_angle_c_measure_l3418_341890


namespace NUMINAMATH_CALUDE_sphere_diameter_equal_volume_cone_l3418_341802

/-- The diameter of a sphere with the same volume as a cone -/
theorem sphere_diameter_equal_volume_cone (r h : ℝ) (hr : r = 2) (hh : h = 8) :
  let cone_volume := (1/3) * Real.pi * r^2 * h
  let sphere_radius := (cone_volume * 3 / (4 * Real.pi))^(1/3)
  2 * sphere_radius = 4 := by sorry

end NUMINAMATH_CALUDE_sphere_diameter_equal_volume_cone_l3418_341802


namespace NUMINAMATH_CALUDE_largest_digit_sum_l3418_341841

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, and c are digits
  (10 ≤ y ∧ y ≤ 20) →  -- 10 ≤ y ≤ 20
  ((a * 100 + b * 10 + c) : ℚ) / 1000 = 1 / y →  -- 0.abc = 1/y
  a + b + c ≤ 5 :=  -- The sum is at most 5
by sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l3418_341841


namespace NUMINAMATH_CALUDE_cricket_ratio_l3418_341866

/-- Represents the number of crickets Spike hunts in the morning -/
def morning_crickets : ℕ := 5

/-- Represents the total number of crickets Spike hunts per day -/
def total_crickets : ℕ := 20

/-- Represents the number of crickets Spike hunts in the afternoon and evening -/
def afternoon_evening_crickets : ℕ := total_crickets - morning_crickets

/-- The theorem stating the ratio of crickets hunted in the afternoon and evening to morning -/
theorem cricket_ratio : 
  afternoon_evening_crickets / morning_crickets = 3 :=
sorry

end NUMINAMATH_CALUDE_cricket_ratio_l3418_341866


namespace NUMINAMATH_CALUDE_extremum_at_one_implies_f_two_equals_two_l3418_341822

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- State the theorem
theorem extremum_at_one_implies_f_two_equals_two (a b : ℝ) :
  (∃ (y : ℝ), y = f a b 1 ∧ y = 10 ∧ 
    (∀ (x : ℝ), f a b x ≤ y ∨ f a b x ≥ y)) →
  f a b 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_extremum_at_one_implies_f_two_equals_two_l3418_341822


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3418_341856

/-- Given a rectangle EFGH where:
  * EF is twice as long as FG
  * FG = 10 units
  * Diagonal EH = 26 units
Prove that the perimeter of EFGH is 60 units -/
theorem rectangle_perimeter (EF FG EH : ℝ) : 
  EF = 2 * FG →
  FG = 10 →
  EH = 26 →
  EH^2 = EF^2 + FG^2 →
  2 * (EF + FG) = 60 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_l3418_341856


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3418_341859

theorem unique_solution_equation :
  ∃! y : ℝ, (3 * y^2 - 12 * y) / (y^2 - 4 * y) = y - 2 ∧
             y ≠ 2 ∧
             y^2 - 4 * y ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3418_341859


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3418_341855

theorem unique_triple_solution : 
  ∃! (a b c : ℕ+), a * b + b * c = 72 ∧ a * c + b * c = 35 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3418_341855


namespace NUMINAMATH_CALUDE_orange_purchase_total_l3418_341838

/-- The total quantity of oranges bought over three weeks -/
def totalOranges (initialPurchase additionalPurchase : ℕ) : ℕ :=
  let week1Total := initialPurchase + additionalPurchase
  let weeklyPurchaseAfter := 2 * week1Total
  week1Total + weeklyPurchaseAfter + weeklyPurchaseAfter

/-- Proof that the total quantity of oranges bought after three weeks is 75 kgs -/
theorem orange_purchase_total :
  totalOranges 10 5 = 75 := by
  sorry


end NUMINAMATH_CALUDE_orange_purchase_total_l3418_341838


namespace NUMINAMATH_CALUDE_factorization_problems_l3418_341804

theorem factorization_problems :
  (∀ a b : ℝ, a^2 * b - a * b^2 = a * b * (a - b)) ∧
  (∀ x : ℝ, 2 * x^2 - 8 = 2 * (x + 2) * (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l3418_341804


namespace NUMINAMATH_CALUDE_dolly_dresses_shipment_l3418_341812

theorem dolly_dresses_shipment (total : ℕ) : 
  (70 : ℕ) * total = 140 * 100 → total = 200 := by
  sorry

end NUMINAMATH_CALUDE_dolly_dresses_shipment_l3418_341812


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3418_341877

theorem cube_root_equation_solution : 
  ∃! x : ℝ, (3 - x / 3) ^ (1/3 : ℝ) = -2 :=
by
  -- The unique solution is x = 33
  use 33
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3418_341877


namespace NUMINAMATH_CALUDE_min_value_expression_l3418_341852

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3418_341852


namespace NUMINAMATH_CALUDE_factorization_equality_l3418_341861

theorem factorization_equality (x : ℝ) : 84 * x^5 - 210 * x^9 = -42 * x^5 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3418_341861


namespace NUMINAMATH_CALUDE_hockey_goals_difference_l3418_341871

theorem hockey_goals_difference (layla_goals kristin_goals : ℕ) : 
  layla_goals = 104 →
  kristin_goals < layla_goals →
  (layla_goals + kristin_goals) / 2 = 92 →
  layla_goals - kristin_goals = 24 := by
sorry

end NUMINAMATH_CALUDE_hockey_goals_difference_l3418_341871


namespace NUMINAMATH_CALUDE_subsets_and_sum_of_M_l3418_341863

def M : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem subsets_and_sum_of_M :
  (Finset.powerset M).card = 2^10 ∧
  (Finset.powerset M).sum (λ s => s.sum id) = 55 * 2^9 := by
  sorry

end NUMINAMATH_CALUDE_subsets_and_sum_of_M_l3418_341863


namespace NUMINAMATH_CALUDE_johns_final_push_time_l3418_341851

/-- The time of John's final push in a race, given the initial and final distances between John and Steve, and their respective speeds. -/
theorem johns_final_push_time 
  (initial_distance : ℝ) 
  (john_speed : ℝ) 
  (steve_speed : ℝ) 
  (final_distance : ℝ) 
  (h1 : initial_distance = 12)
  (h2 : john_speed = 4.2)
  (h3 : steve_speed = 3.7)
  (h4 : final_distance = 2) :
  ∃ t : ℝ, t = 28 ∧ john_speed * t = steve_speed * t + initial_distance + final_distance :=
by sorry

end NUMINAMATH_CALUDE_johns_final_push_time_l3418_341851


namespace NUMINAMATH_CALUDE_inequality_proof_l3418_341831

theorem inequality_proof (a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) :
  (a₁ + a₃) / (a₁ + a₂) + (a₂ + a₄) / (a₂ + a₃) + (a₃ + a₁) / (a₃ + a₄) + (a₄ + a₂) / (a₄ + a₁) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3418_341831


namespace NUMINAMATH_CALUDE_james_new_friends_l3418_341898

def number_of_new_friends (initial_friends lost_friends final_friends : ℕ) : ℕ :=
  final_friends - (initial_friends - lost_friends)

theorem james_new_friends :
  number_of_new_friends 20 2 19 = 1 := by
  sorry

end NUMINAMATH_CALUDE_james_new_friends_l3418_341898


namespace NUMINAMATH_CALUDE_simplify_exponents_l3418_341887

theorem simplify_exponents (t s : ℝ) : (t^2 * t^5) * s^3 = t^7 * s^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponents_l3418_341887


namespace NUMINAMATH_CALUDE_smallest_palindrome_l3418_341825

/-- A number is a palindrome in a given base if it reads the same forwards and backwards when represented in that base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in a given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- The number of digits in the representation of a natural number in a given base. -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

/-- 10101₂ in decimal -/
def target : ℕ := 21

theorem smallest_palindrome :
  ∀ n : ℕ,
  (numDigits n 2 = 5 ∧ isPalindrome n 2) →
  (∃ base : ℕ, base > 4 ∧ numDigits n base = 3 ∧ isPalindrome n base) →
  n ≥ target :=
sorry

end NUMINAMATH_CALUDE_smallest_palindrome_l3418_341825


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_not_base_area_times_height_l3418_341888

/-- The lateral area of a cylinder is not equal to the base area multiplied by the height. -/
theorem cylinder_lateral_area_not_base_area_times_height 
  (r h : ℝ) (r_pos : 0 < r) (h_pos : 0 < h) :
  2 * π * r * h ≠ (π * r^2) * h := by sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_not_base_area_times_height_l3418_341888


namespace NUMINAMATH_CALUDE_complex_multiplication_l3418_341827

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 - i) = 1 + 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3418_341827


namespace NUMINAMATH_CALUDE_frustum_volume_l3418_341854

/-- Given a square pyramid and a smaller pyramid cut from it parallel to the base,
    calculate the volume of the resulting frustum. -/
theorem frustum_volume
  (base_edge : ℝ)
  (altitude : ℝ)
  (small_base_edge : ℝ)
  (small_altitude : ℝ)
  (h_base : base_edge = 15)
  (h_altitude : altitude = 10)
  (h_small_base : small_base_edge = 7.5)
  (h_small_altitude : small_altitude = 5) :
  (1 / 3 * base_edge^2 * altitude) - (1 / 3 * small_base_edge^2 * small_altitude) = 656.25 :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_l3418_341854


namespace NUMINAMATH_CALUDE_prism_volume_l3418_341897

/-- A right rectangular prism with specific face areas and a dimension relation -/
structure RectangularPrism where
  x : ℝ
  y : ℝ
  z : ℝ
  side_area : x * y = 24
  front_area : y * z = 15
  bottom_area : x * z = 8
  dimension_relation : z = 2 * x

/-- The volume of a rectangular prism is equal to 96 cubic inches -/
theorem prism_volume (p : RectangularPrism) : p.x * p.y * p.z = 96 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3418_341897


namespace NUMINAMATH_CALUDE_weeks_to_buy_iphone_l3418_341850

def iphone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80

theorem weeks_to_buy_iphone :
  (iphone_cost - trade_in_value) / weekly_earnings = 7 := by
  sorry

end NUMINAMATH_CALUDE_weeks_to_buy_iphone_l3418_341850


namespace NUMINAMATH_CALUDE_polynomial_infinite_solutions_l3418_341835

theorem polynomial_infinite_solutions (P : ℤ → ℤ) (d : ℤ) :
  (∃ (a b : ℤ), ∀ x, P x = a * x + b) ∨ (∀ x, P x = P 0) ↔
  (∃ (S : Set (ℤ × ℤ)), (∀ (x y : ℤ), (x, y) ∈ S → x ≠ y) ∧ 
                         Set.Infinite S ∧
                         (∀ (x y : ℤ), (x, y) ∈ S → P x - P y = d)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_infinite_solutions_l3418_341835


namespace NUMINAMATH_CALUDE_line_intersecting_circle_slope_l3418_341892

/-- A line passing through (4,0) intersecting the circle (x-2)^2 + y^2 = 1 has slope -√3/3 or √3/3 -/
theorem line_intersecting_circle_slope :
  ∀ (k : ℝ), 
    (∃ (x y : ℝ), y = k * (x - 4) ∧ (x - 2)^2 + y^2 = 1) →
    (k = -Real.sqrt 3 / 3 ∨ k = Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_line_intersecting_circle_slope_l3418_341892


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3418_341876

theorem absolute_value_inequality (a b : ℝ) : 
  (1 / |a| < 1 / |b|) → |a| > |b| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3418_341876


namespace NUMINAMATH_CALUDE_x_value_l3418_341847

theorem x_value : ∃ x : ℝ, (0.5 * x = 0.05 * 500 - 20) ∧ (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3418_341847


namespace NUMINAMATH_CALUDE_parentheses_expression_l3418_341896

theorem parentheses_expression (a b : ℝ) : (3*b + a) * (3*b - a) = 9*b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_expression_l3418_341896


namespace NUMINAMATH_CALUDE_polyhedron_inequality_l3418_341811

/-- A convex polyhedron bounded by quadrilateral faces -/
class ConvexPolyhedron where
  /-- The surface area of the polyhedron -/
  area : ℝ
  /-- The sum of the squares of the polyhedron's edges -/
  edge_sum_squares : ℝ
  /-- The polyhedron is bounded by quadrilateral faces -/
  quad_faces : Prop

/-- 
For a convex polyhedron bounded by quadrilateral faces, 
the sum of the squares of its edges is greater than or equal to twice its surface area 
-/
theorem polyhedron_inequality (p : ConvexPolyhedron) : p.edge_sum_squares ≥ 2 * p.area := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_inequality_l3418_341811


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l3418_341864

/-- Given a cylinder with volume 15 cubic meters, if its radius is tripled
    and its height is doubled, then its new volume is 270 cubic meters. -/
theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 15 → π * (3*r)^2 * (2*h) = 270 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l3418_341864


namespace NUMINAMATH_CALUDE_increase_by_percentage_seventy_increased_by_150_percent_l3418_341845

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem seventy_increased_by_150_percent :
  70 * (1 + 150 / 100) = 175 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_seventy_increased_by_150_percent_l3418_341845


namespace NUMINAMATH_CALUDE_product_equals_square_l3418_341818

theorem product_equals_square : 500 * 49.95 * 4.995 * 5000 = (24975 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l3418_341818


namespace NUMINAMATH_CALUDE_andy_restrung_seven_racquets_l3418_341816

/-- Calculates the number of racquets Andy restrung during his shift -/
def racquets_restrung (hourly_rate : ℤ) (restring_rate : ℤ) (grommet_rate : ℤ) (stencil_rate : ℤ)
                      (hours_worked : ℤ) (grommets_changed : ℤ) (stencils_painted : ℤ)
                      (total_earnings : ℤ) : ℤ :=
  let hourly_earnings := hourly_rate * hours_worked
  let grommet_earnings := grommet_rate * grommets_changed
  let stencil_earnings := stencil_rate * stencils_painted
  let restring_earnings := total_earnings - hourly_earnings - grommet_earnings - stencil_earnings
  restring_earnings / restring_rate

theorem andy_restrung_seven_racquets :
  racquets_restrung 9 15 10 1 8 2 5 202 = 7 := by
  sorry


end NUMINAMATH_CALUDE_andy_restrung_seven_racquets_l3418_341816


namespace NUMINAMATH_CALUDE_inequality_proof_l3418_341873

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b * c) / (a^2 + b * c) + (c * a) / (b^2 + c * a) + (a * b) / (c^2 + a * b) ≤
  a / (b + c) + b / (c + a) + c / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3418_341873


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l3418_341810

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 1, b = 2, and C = 60°, then c = √3 and the area is √3/2 -/
theorem triangle_side_and_area 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 1) 
  (h2 : b = 2) 
  (h3 : C = Real.pi / 3) -- 60° in radians
  (h4 : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) -- Law of cosines
  (h5 : (a*b*(Real.sin C))/2 = area_triangle) : 
  c = Real.sqrt 3 ∧ area_triangle = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_and_area_l3418_341810


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3418_341849

theorem quadratic_real_roots (k m : ℝ) (hm : m ≠ 0) :
  (∃ x : ℝ, x^2 + k*x + m = 0) ↔ m ≤ k^2/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3418_341849


namespace NUMINAMATH_CALUDE_blueberry_jelly_amount_l3418_341820

theorem blueberry_jelly_amount (total_jelly strawberry_jelly : ℕ) 
  (h1 : total_jelly = 6310)
  (h2 : strawberry_jelly = 1792) :
  total_jelly - strawberry_jelly = 4518 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_jelly_amount_l3418_341820


namespace NUMINAMATH_CALUDE_sequence_properties_l3418_341836

-- Define a sequence as a function from natural numbers to real numbers
def Sequence := ℕ → ℝ

-- Statement 1: Sequences appear as isolated points when graphed
def isolated_points (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ ε > 0, ∀ m : ℕ, m ≠ n → |s m - s n| ≥ ε

-- Statement 2: All sequences have infinite terms
def infinite_terms (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n

-- Statement 3: The general term formula of a sequence is unique
def unique_formula (s : Sequence) : Prop :=
  ∀ f g : Sequence, (∀ n : ℕ, f n = s n) → (∀ n : ℕ, g n = s n) → f = g

-- Theorem stating that only the first statement is correct
theorem sequence_properties :
  (∀ s : Sequence, isolated_points s) ∧
  (∃ s : Sequence, ¬infinite_terms s) ∧
  (∃ s : Sequence, ¬unique_formula s) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l3418_341836


namespace NUMINAMATH_CALUDE_holiday_customers_l3418_341819

def normal_rate : ℕ := 175
def holiday_multiplier : ℕ := 2
def hours : ℕ := 8

theorem holiday_customers :
  normal_rate * holiday_multiplier * hours = 2800 :=
by sorry

end NUMINAMATH_CALUDE_holiday_customers_l3418_341819


namespace NUMINAMATH_CALUDE_attendance_difference_l3418_341823

/-- The number of games Tara played each year -/
def games_per_year : ℕ := 20

/-- The percentage of games Tara's dad attended in the first year -/
def first_year_attendance_percentage : ℚ := 90 / 100

/-- The number of games Tara's dad attended in the second year -/
def second_year_attendance : ℕ := 14

/-- Theorem stating the difference in games attended between the first and second year -/
theorem attendance_difference :
  ⌊(first_year_attendance_percentage * games_per_year : ℚ)⌋ - second_year_attendance = 4 := by
  sorry

end NUMINAMATH_CALUDE_attendance_difference_l3418_341823


namespace NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l3418_341868

theorem a_gt_abs_b_sufficient_not_necessary_for_a_sq_gt_b_sq :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l3418_341868


namespace NUMINAMATH_CALUDE_total_difference_is_122_l3418_341833

/-- The total difference in the number of apples and peaches for Mia, Steven, and Jake -/
def total_difference (steven_apples steven_peaches : ℕ) : ℕ :=
  let mia_apples := 2 * steven_apples
  let jake_apples := steven_apples + 4
  let jake_peaches := steven_peaches - 3
  let mia_peaches := jake_peaches + 3
  (mia_apples + mia_peaches) + (steven_apples + steven_peaches) + (jake_apples + jake_peaches)

/-- Theorem stating the total difference in fruits for Mia, Steven, and Jake -/
theorem total_difference_is_122 :
  total_difference 19 15 = 122 :=
by sorry

end NUMINAMATH_CALUDE_total_difference_is_122_l3418_341833


namespace NUMINAMATH_CALUDE_union_equality_implies_a_greater_than_one_l3418_341875

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem union_equality_implies_a_greater_than_one (a : ℝ) :
  A ∪ B a = B a → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_greater_than_one_l3418_341875


namespace NUMINAMATH_CALUDE_min_value_expression_l3418_341883

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3418_341883


namespace NUMINAMATH_CALUDE_cricketer_average_score_l3418_341830

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_part_matches : ℕ) 
  (overall_average : ℝ) 
  (first_part_average : ℝ) 
  (h1 : total_matches = 12) 
  (h2 : first_part_matches = 8) 
  (h3 : overall_average = 48) 
  (h4 : first_part_average = 40) :
  let last_part_matches := total_matches - first_part_matches
  let total_runs := total_matches * overall_average
  let first_part_runs := first_part_matches * first_part_average
  let last_part_runs := total_runs - first_part_runs
  last_part_runs / last_part_matches = 64 := by
sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l3418_341830


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3418_341881

/-- Given a quadratic function f(x) = x^2 + ax + b with roots -2 and 3, prove that a + b = -7 -/
theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 ↔ x = -2 ∨ x = 3) → a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3418_341881


namespace NUMINAMATH_CALUDE_child_ticket_cost_l3418_341879

/-- Proves that the cost of a child ticket is $5 given the theater conditions --/
theorem child_ticket_cost (total_seats : ℕ) (adult_price : ℕ) (child_tickets : ℕ) (total_revenue : ℕ) :
  total_seats = 80 →
  adult_price = 12 →
  child_tickets = 63 →
  total_revenue = 519 →
  ∃ (child_price : ℕ), 
    child_price = 5 ∧
    total_revenue = (total_seats - child_tickets) * adult_price + child_tickets * child_price :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l3418_341879


namespace NUMINAMATH_CALUDE_crushing_load_calculation_l3418_341809

theorem crushing_load_calculation (T H K : ℚ) (L : ℚ) 
  (h1 : T = 5)
  (h2 : H = 10)
  (h3 : K = 2)
  (h4 : L = (30 * T^3 * K) / H^3) :
  L = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_crushing_load_calculation_l3418_341809


namespace NUMINAMATH_CALUDE_function_formula_l3418_341837

theorem function_formula (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2) :
  ∀ x : ℝ, f x = (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_function_formula_l3418_341837


namespace NUMINAMATH_CALUDE_rectangle_area_l3418_341857

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3418_341857


namespace NUMINAMATH_CALUDE_prob_all_suits_in_five_draws_l3418_341862

-- Define a standard deck
def standard_deck : ℕ := 52

-- Define the number of suits
def num_suits : ℕ := 4

-- Define the number of cards drawn
def cards_drawn : ℕ := 5

-- Define the probability of drawing a card from a specific suit
def prob_suit : ℚ := 1 / 4

-- Theorem statement
theorem prob_all_suits_in_five_draws :
  let prob_sequence := (1 : ℚ) * (3 / 4) * (1 / 2) * (1 / 4)
  let num_sequences := 24
  (prob_sequence * num_sequences : ℚ) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_suits_in_five_draws_l3418_341862


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3418_341882

theorem sufficient_not_necessary (x y : ℝ) :
  ((x + 3)^2 + (y - 4)^2 = 0 → (x + 3) * (y - 4) = 0) ∧
  ¬((x + 3) * (y - 4) = 0 → (x + 3)^2 + (y - 4)^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3418_341882
