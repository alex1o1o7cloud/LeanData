import Mathlib

namespace NUMINAMATH_CALUDE_line_touches_x_axis_twice_l1358_135837

/-- Represents the equation d = x^2 - x^3 -/
def d (x : ℝ) : ℝ := x^2 - x^3

/-- The line touches the x-axis when d(x) = 0 -/
def touches_x_axis (x : ℝ) : Prop := d x = 0

theorem line_touches_x_axis_twice :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ touches_x_axis x₁ ∧ touches_x_axis x₂ ∧
  ∀ (x : ℝ), touches_x_axis x → (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_line_touches_x_axis_twice_l1358_135837


namespace NUMINAMATH_CALUDE_race_tie_l1358_135826

/-- A race between two runners A and B -/
structure Race where
  length : ℝ
  speed_ratio : ℝ
  head_start : ℝ

/-- The race conditions -/
def race_conditions : Race where
  length := 100
  speed_ratio := 4
  head_start := 75

/-- Theorem stating that the given head start results in a tie -/
theorem race_tie (race : Race) (h1 : race.length = 100) (h2 : race.speed_ratio = 4) 
    (h3 : race.head_start = 75) : 
  race.length / race.speed_ratio = (race.length - race.head_start) / 1 := by
  sorry

#check race_tie race_conditions rfl rfl rfl

end NUMINAMATH_CALUDE_race_tie_l1358_135826


namespace NUMINAMATH_CALUDE_cindy_crayons_count_l1358_135891

/-- The number of crayons Karen has -/
def karen_crayons : ℕ := 639

/-- The difference between Karen's and Cindy's crayons -/
def difference : ℕ := 135

/-- The number of crayons Cindy has -/
def cindy_crayons : ℕ := karen_crayons - difference

theorem cindy_crayons_count : cindy_crayons = 504 := by
  sorry

end NUMINAMATH_CALUDE_cindy_crayons_count_l1358_135891


namespace NUMINAMATH_CALUDE_runt_pig_revenue_l1358_135893

/-- Calculates the revenue from selling bacon from a pig -/
def bacon_revenue (average_yield : ℝ) (price_per_pound : ℝ) (size_ratio : ℝ) : ℝ :=
  average_yield * size_ratio * price_per_pound

/-- Proves that the farmer will make $60 from the runt pig's bacon -/
theorem runt_pig_revenue :
  let average_yield : ℝ := 20
  let price_per_pound : ℝ := 6
  let size_ratio : ℝ := 0.5
  bacon_revenue average_yield price_per_pound size_ratio = 60 := by
sorry

end NUMINAMATH_CALUDE_runt_pig_revenue_l1358_135893


namespace NUMINAMATH_CALUDE_complex_magnitude_two_thirds_plus_three_i_l1358_135817

theorem complex_magnitude_two_thirds_plus_three_i :
  Complex.abs (2/3 + 3*Complex.I) = Real.sqrt 85 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_two_thirds_plus_three_i_l1358_135817


namespace NUMINAMATH_CALUDE_find_second_number_l1358_135832

/-- Given two positive integers with known HCF and LCM, find the second number -/
theorem find_second_number (A B : ℕ+) (h1 : A = 24) 
  (h2 : Nat.gcd A B = 13) (h3 : Nat.lcm A B = 312) : B = 169 := by
  sorry

end NUMINAMATH_CALUDE_find_second_number_l1358_135832


namespace NUMINAMATH_CALUDE_midpoint_specific_segment_l1358_135897

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ := sorry

theorem midpoint_specific_segment :
  let p1 : ℝ × ℝ := (6, π/4)
  let p2 : ℝ × ℝ := (6, 3*π/4)
  let (r, θ) := polar_midpoint p1.1 p1.2 p2.1 p2.2
  r = 3 * Real.sqrt 2 ∧ θ = π/2 :=
sorry

end NUMINAMATH_CALUDE_midpoint_specific_segment_l1358_135897


namespace NUMINAMATH_CALUDE_inequality_holds_l1358_135835

theorem inequality_holds (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^3 * (y + 1) + y^3 * (x + 1) ≥ x^2 * (y + y^2) + y^2 * (x + x^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1358_135835


namespace NUMINAMATH_CALUDE_range_of_fraction_l1358_135834

def f (a b c x : ℝ) : ℝ := 3 * a * x^2 - 2 * b * x + c

theorem range_of_fraction (a b c : ℝ) :
  (a - b + c = 0) →
  (f a b c 0 > 0) →
  (f a b c 1 > 0) →
  ∃ (y : ℝ), (4/3 < y ∧ y < 7/2 ∧ y = (a + 3*b + 7*c) / (2*a + b)) ∧
  ∀ (z : ℝ), (z = (a + 3*b + 7*c) / (2*a + b)) → (4/3 < z ∧ z < 7/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1358_135834


namespace NUMINAMATH_CALUDE_find_a_l1358_135843

def U (a : ℝ) : Set ℝ := {2, 3, a^2 - 2*a - 3}
def A (a : ℝ) : Set ℝ := {2, |a - 7|}

theorem find_a : ∀ a : ℝ, (U a) \ (A a) = {5} → a = 4 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_find_a_l1358_135843


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1358_135869

/-- The equation of the tangent line to the curve y = x sin x at the point (π, 0) is y = -πx + π² -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x * Real.sin x) → -- Curve equation
  (∃ (m b : ℝ), (y = m * x + b) ∧ -- Tangent line equation
                (0 = m * π + b) ∧ -- Point (π, 0) satisfies the tangent line equation
                (m = Real.sin π + π * Real.cos π)) → -- Slope of the tangent line
  (y = -π * x + π^2) -- Resulting tangent line equation
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1358_135869


namespace NUMINAMATH_CALUDE_permutation_distinct_differences_l1358_135820

theorem permutation_distinct_differences (n : ℕ+) :
  (∃ (a : Fin n → Fin n), Function.Bijective a ∧
    (∀ (i j : Fin n), i ≠ j → |a i - i| ≠ |a j - j|)) ↔
  (∃ (k : ℕ), n = 4 * k ∨ n = 4 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_permutation_distinct_differences_l1358_135820


namespace NUMINAMATH_CALUDE_fall_semester_duration_l1358_135818

/-- The duration of the fall semester in weeks -/
def semester_length : ℕ := 15

/-- The number of hours Paris studies during weekdays -/
def weekday_hours : ℕ := 3

/-- The number of hours Paris studies on Saturday -/
def saturday_hours : ℕ := 4

/-- The number of hours Paris studies on Sunday -/
def sunday_hours : ℕ := 5

/-- The total number of hours Paris studies during the semester -/
def total_study_hours : ℕ := 360

theorem fall_semester_duration :
  semester_length * (5 * weekday_hours + saturday_hours + sunday_hours) = total_study_hours := by
  sorry

end NUMINAMATH_CALUDE_fall_semester_duration_l1358_135818


namespace NUMINAMATH_CALUDE_property_P_theorems_l1358_135898

/-- Property (P): A number n ≥ 2 has property (P) if in its prime factorization,
    at least one of the factors has an exponent of 3 -/
def has_property_P (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∃ p : ℕ, Prime p ∧ (∃ k : ℕ, n = p^(3*k+3) * (n / p^(3*k+3)))

/-- The smallest N such that any N consecutive natural numbers contain
    at least one number with property (P) -/
def smallest_N : ℕ := 16

/-- The smallest 15 consecutive numbers without property (P) such that
    their sum multiplied by 5 has property (P) -/
def smallest_15_consecutive : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

theorem property_P_theorems :
  (∀ k : ℕ, ∃ n ∈ List.range smallest_N, has_property_P (k + n)) ∧
  (∀ n ∈ smallest_15_consecutive, ¬ has_property_P n) ∧
  has_property_P (5 * smallest_15_consecutive.sum) := by
  sorry

end NUMINAMATH_CALUDE_property_P_theorems_l1358_135898


namespace NUMINAMATH_CALUDE_expression_evaluation_l1358_135864

theorem expression_evaluation : (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1358_135864


namespace NUMINAMATH_CALUDE_larger_square_side_length_l1358_135836

theorem larger_square_side_length 
  (small_square_side : ℝ) 
  (larger_square_perimeter : ℝ) 
  (h1 : small_square_side = 20) 
  (h2 : larger_square_perimeter = 4 * small_square_side + 20) : 
  larger_square_perimeter / 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_square_side_length_l1358_135836


namespace NUMINAMATH_CALUDE_exists_prime_with_integer_roots_l1358_135889

theorem exists_prime_with_integer_roots :
  ∃ p : ℕ, Prime p ∧ 1 < p ∧ p ≤ 11 ∧
  ∃ x y : ℤ, x^2 + p*x - 720*p = 0 ∧ y^2 + p*y - 720*p = 0 :=
by sorry

end NUMINAMATH_CALUDE_exists_prime_with_integer_roots_l1358_135889


namespace NUMINAMATH_CALUDE_sin_pi_12_plus_theta_l1358_135895

theorem sin_pi_12_plus_theta (θ : ℝ) (h : Real.cos ((5 * Real.pi) / 12 - θ) = 1 / 3) :
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_12_plus_theta_l1358_135895


namespace NUMINAMATH_CALUDE_additional_week_cost_is_eleven_l1358_135879

/-- The cost per day for additional weeks in a student youth hostel -/
def additional_week_cost (first_week_daily_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) : ℚ :=
  let first_week_cost := 7 * first_week_daily_rate
  let additional_days := total_days - 7
  let additional_cost := total_cost - first_week_cost
  additional_cost / additional_days

theorem additional_week_cost_is_eleven :
  additional_week_cost 18 23 302 = 11 := by
sorry

end NUMINAMATH_CALUDE_additional_week_cost_is_eleven_l1358_135879


namespace NUMINAMATH_CALUDE_cherry_trees_planted_l1358_135868

/-- The number of trees planted by each group in a tree-planting event --/
structure TreePlanting where
  apple : ℕ
  orange : ℕ
  cherry : ℕ

/-- The conditions of the tree-planting event --/
def tree_planting_conditions (t : TreePlanting) : Prop :=
  t.apple = 2 * t.orange ∧
  t.orange = t.apple - 15 ∧
  t.cherry = t.apple + t.orange - 10 ∧
  t.apple = 47 ∧
  t.orange = 27

/-- Theorem stating that under the given conditions, 64 cherry trees were planted --/
theorem cherry_trees_planted (t : TreePlanting) 
  (h : tree_planting_conditions t) : t.cherry = 64 := by
  sorry


end NUMINAMATH_CALUDE_cherry_trees_planted_l1358_135868


namespace NUMINAMATH_CALUDE_number_problem_l1358_135838

theorem number_problem (x : ℝ) : (0.6 * x = 0.3 * 10 + 27) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1358_135838


namespace NUMINAMATH_CALUDE_kanul_cash_proof_l1358_135821

/-- The total amount of cash Kanul had -/
def total_cash : ℝ := 1000

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 500

/-- The amount spent on machinery -/
def machinery : ℝ := 400

/-- The percentage of total cash spent as cash -/
def cash_percentage : ℝ := 0.1

theorem kanul_cash_proof :
  total_cash = raw_materials + machinery + cash_percentage * total_cash :=
by sorry

end NUMINAMATH_CALUDE_kanul_cash_proof_l1358_135821


namespace NUMINAMATH_CALUDE_tim_apartment_complexes_l1358_135896

/-- The number of keys Tim needs to make -/
def total_keys : ℕ := 72

/-- The number of keys needed per lock -/
def keys_per_lock : ℕ := 3

/-- The number of apartments in each complex -/
def apartments_per_complex : ℕ := 12

/-- The number of apartment complexes Tim owns -/
def num_complexes : ℕ := total_keys / keys_per_lock / apartments_per_complex

theorem tim_apartment_complexes : num_complexes = 2 := by
  sorry

end NUMINAMATH_CALUDE_tim_apartment_complexes_l1358_135896


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1358_135807

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + n.choose 2 + n.choose 3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1358_135807


namespace NUMINAMATH_CALUDE_E_equality_condition_l1358_135808

/-- Definition of the function E --/
def E (a b c : ℚ) : ℚ := a * b^2 + b * c + c

/-- Theorem stating the equality condition for E(a,3,2) and E(a,5,3) --/
theorem E_equality_condition :
  ∀ a : ℚ, E a 3 2 = E a 5 3 ↔ a = -5/8 := by sorry

end NUMINAMATH_CALUDE_E_equality_condition_l1358_135808


namespace NUMINAMATH_CALUDE_x_value_when_z_is_64_l1358_135871

/-- Given that x is inversely proportional to y², y is directly proportional to √z,
    and x = 4 when z = 16, prove that x = 1 when z = 64 -/
theorem x_value_when_z_is_64 
  (x y z : ℝ) 
  (h1 : ∃ (k : ℝ), x * y^2 = k) 
  (h2 : ∃ (m : ℝ), y = m * Real.sqrt z) 
  (h3 : x = 4 ∧ z = 16) : 
  z = 64 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_z_is_64_l1358_135871


namespace NUMINAMATH_CALUDE_unique_triples_l1358_135884

theorem unique_triples : 
  ∀ (a b c : ℕ+), 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
    (∃ (k₁ : ℕ+), (2 * a - 1 : ℤ) = k₁ * b) →
    (∃ (k₂ : ℕ+), (2 * b - 1 : ℤ) = k₂ * c) →
    (∃ (k₃ : ℕ+), (2 * c - 1 : ℤ) = k₃ * a) →
    ((a = 7 ∧ b = 13 ∧ c = 25) ∨
     (a = 13 ∧ b = 25 ∧ c = 7) ∨
     (a = 25 ∧ b = 7 ∧ c = 13)) :=
by sorry

end NUMINAMATH_CALUDE_unique_triples_l1358_135884


namespace NUMINAMATH_CALUDE_work_completion_time_l1358_135810

/-- Represents the rate of work for person B -/
def rate_B : ℝ := 1

/-- Represents the rate of work for person A -/
def rate_A : ℝ := 3 * rate_B

/-- Represents the time taken by B to complete the work alone -/
def time_B : ℝ := 90

/-- Represents the time taken by A to complete the work alone -/
def time_A : ℝ := time_B - 60

/-- Represents the amount of work to be done -/
def work : ℝ := rate_B * time_B

/-- The theorem to be proved -/
theorem work_completion_time : 
  (work / (rate_A + rate_B)) = 22.5 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1358_135810


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l1358_135840

-- Define the original function
def f (x : ℝ) : ℝ := |x + 1| - 2

-- Define the translation
def translate_x : ℝ := 3
def translate_y : ℝ := 4

-- Define the translated function
def g (x : ℝ) : ℝ := f (x - translate_x) + translate_y

-- State the theorem
theorem minimum_point_of_translated_graph :
  ∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = 2 ∧
  ∀ (x : ℝ), g x ≥ g x₀ :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l1358_135840


namespace NUMINAMATH_CALUDE_sequence_constant_l1358_135809

theorem sequence_constant (a : ℤ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_inequality : ∀ n, a n ≥ (a (n + 2) + a (n + 1) + a (n - 1) + a (n - 2)) / 4) :
  ∀ m n, a m = a n :=
sorry

end NUMINAMATH_CALUDE_sequence_constant_l1358_135809


namespace NUMINAMATH_CALUDE_power_function_through_point_l1358_135812

/-- A power function that passes through the point (2, √2) -/
def f (x : ℝ) : ℝ := x ^ (1/2)

/-- Theorem: The power function f(x) that passes through (2, √2) satisfies f(8) = 2√2 -/
theorem power_function_through_point (x : ℝ) :
  f 2 = Real.sqrt 2 → f 8 = 2 * Real.sqrt 2 := by
  sorry

#check power_function_through_point

end NUMINAMATH_CALUDE_power_function_through_point_l1358_135812


namespace NUMINAMATH_CALUDE_y_value_l1358_135883

theorem y_value (x y : ℝ) (h1 : 2 * (x - y) = 12) (h2 : x + y = 14) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1358_135883


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1358_135833

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geometric : IsGeometricSequence a)
  (h_fourth : a 4 = 16)
  (h_ninth : a 9 = 2) :
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1358_135833


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l1358_135823

theorem reciprocal_of_negative_two :
  ((-2 : ℝ)⁻¹ = -1/2) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l1358_135823


namespace NUMINAMATH_CALUDE_sandbag_weight_increase_l1358_135847

/-- Proves that the percentage increase in weight of a heavier filling material compared to sand is 40% given specific conditions. -/
theorem sandbag_weight_increase (capacity : ℝ) (fill_level : ℝ) (actual_weight : ℝ) : 
  capacity = 250 →
  fill_level = 0.8 →
  actual_weight = 280 →
  (actual_weight - fill_level * capacity) / (fill_level * capacity) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_sandbag_weight_increase_l1358_135847


namespace NUMINAMATH_CALUDE_tim_total_score_l1358_135816

/-- The score for a single line in Tetris -/
def single_line_score : ℕ := 1000

/-- The score for a Tetris (four lines cleared at once) -/
def tetris_score : ℕ := 8 * single_line_score

/-- Tim's number of single lines cleared -/
def tim_singles : ℕ := 6

/-- Tim's number of Tetrises -/
def tim_tetrises : ℕ := 4

/-- Theorem stating Tim's total score -/
theorem tim_total_score : tim_singles * single_line_score + tim_tetrises * tetris_score = 38000 := by
  sorry

end NUMINAMATH_CALUDE_tim_total_score_l1358_135816


namespace NUMINAMATH_CALUDE_area_ratio_inscribed_squares_l1358_135876

/-- A square inscribed in a circle -/
structure InscribedSquare :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (side : ℝ)

/-- A square with two vertices on a side of another square and two vertices on a circle -/
structure PartiallyInscribedSquare :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (side : ℝ)

/-- The theorem stating the relationship between the areas of the two squares -/
theorem area_ratio_inscribed_squares 
  (ABCD : InscribedSquare) 
  (EFGH : PartiallyInscribedSquare) 
  (h1 : ABCD.center = EFGH.center) 
  (h2 : ABCD.radius = EFGH.radius) 
  (h3 : ABCD.side ^ 2 = 1) : 
  EFGH.side ^ 2 = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_inscribed_squares_l1358_135876


namespace NUMINAMATH_CALUDE_xiao_wang_processes_60_parts_l1358_135892

/-- Represents the number of parts processed by a worker in a given time period -/
def ProcessedParts (rate : ℕ) (workTime : ℕ) : ℕ := rate * workTime

/-- Represents the total time taken by Xiao Wang to process a given number of parts -/
def XiaoWangTotalTime (parts : ℕ) : ℚ :=
  let workHours := parts / 15
  let breaks := workHours / 2
  (workHours + breaks : ℚ)

/-- Represents the total time taken by Xiao Li to process a given number of parts -/
def XiaoLiTotalTime (parts : ℕ) : ℚ := parts / 12

/-- Theorem stating that Xiao Wang processes 60 parts when both finish at the same time -/
theorem xiao_wang_processes_60_parts :
  ∃ (parts : ℕ), parts = 60 ∧ XiaoWangTotalTime parts = XiaoLiTotalTime parts :=
sorry

end NUMINAMATH_CALUDE_xiao_wang_processes_60_parts_l1358_135892


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1358_135860

theorem unique_solution_for_equation (n p : ℕ) (h_pos_n : n > 0) (h_pos_p : p > 0) (h_prime : Nat.Prime p) :
  n^8 - p^5 = n^2 + p^2 → (n = 2 ∧ p = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1358_135860


namespace NUMINAMATH_CALUDE_f_composition_value_l1358_135855

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2^(x + 2) else x^3

theorem f_composition_value : f (f (-1)) = 8 := by sorry

end NUMINAMATH_CALUDE_f_composition_value_l1358_135855


namespace NUMINAMATH_CALUDE_multiples_of_four_between_100_and_350_l1358_135872

theorem multiples_of_four_between_100_and_350 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ 100 < n ∧ n < 350) (Finset.range 350)).card = 62 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_between_100_and_350_l1358_135872


namespace NUMINAMATH_CALUDE_tank_dimension_l1358_135845

/-- Proves that the third dimension of a rectangular tank is 2 feet given specific conditions -/
theorem tank_dimension (x : ℝ) : 
  (4 : ℝ) > 0 ∧ 
  (5 : ℝ) > 0 ∧ 
  x > 0 ∧
  (20 : ℝ) > 0 ∧
  1520 = 20 * (2 * (4 * 5) + 2 * (4 * x) + 2 * (5 * x)) →
  x = 2 := by
  sorry

#check tank_dimension

end NUMINAMATH_CALUDE_tank_dimension_l1358_135845


namespace NUMINAMATH_CALUDE_original_denominator_problem_l1358_135887

theorem original_denominator_problem (d : ℚ) : 
  (4 : ℚ) / d ≠ 0 →
  (4 + 3 : ℚ) / (d + 3) = 2 / 3 →
  d = 7.5 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l1358_135887


namespace NUMINAMATH_CALUDE_train_length_proof_l1358_135851

/-- Given a train crossing two platforms with constant speed, prove its length is 70 meters. -/
theorem train_length_proof (
  platform1_length : ℝ)
  (platform2_length : ℝ)
  (time1 : ℝ)
  (time2 : ℝ)
  (h1 : platform1_length = 170)
  (h2 : platform2_length = 250)
  (h3 : time1 = 15)
  (h4 : time2 = 20)
  (h5 : (platform1_length + train_length) / time1 = (platform2_length + train_length) / time2)
  : train_length = 70 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l1358_135851


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1358_135877

/-- The lateral surface area of a cone with base radius 6 and volume 30π is 39π -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1/3) * π * r^2 * h = 30 * π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39 * π := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1358_135877


namespace NUMINAMATH_CALUDE_visiting_students_theorem_l1358_135824

/-- Represents a set of students visiting each other's homes -/
structure VisitingStudents where
  n : ℕ  -- number of students
  d : ℕ  -- number of days
  assignment : Fin n → Finset (Fin d)

/-- A valid assignment means no subset is contained within another subset -/
def ValidAssignment (vs : VisitingStudents) : Prop :=
  ∀ i j : Fin vs.n, i ≠ j → ¬(vs.assignment i ⊆ vs.assignment j)

theorem visiting_students_theorem :
  (¬∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 4 ∧ ValidAssignment vs) ∧
  (¬∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 5 ∧ ValidAssignment vs) ∧
  (∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 7 ∧ ValidAssignment vs) ∧
  (∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 10 ∧ ValidAssignment vs) :=
by sorry

end NUMINAMATH_CALUDE_visiting_students_theorem_l1358_135824


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1358_135839

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if both roots of a quadratic equation are greater than 1 -/
def bothRootsGreaterThanOne (eq : QuadraticEquation) : Prop :=
  ∀ x, eq.a * x^2 + eq.b * x + eq.c = 0 → x > 1

/-- The main theorem stating the condition on m -/
theorem quadratic_roots_condition (m : ℝ) :
  let eq : QuadraticEquation := ⟨8, 1 - m, m - 7⟩
  bothRootsGreaterThanOne eq → m ≥ 25 := by
  sorry

#check quadratic_roots_condition

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1358_135839


namespace NUMINAMATH_CALUDE_remainder_of_sum_l1358_135880

theorem remainder_of_sum (a b c : ℕ) : 
  0 < a ∧ a < 7 ∧ 
  0 < b ∧ b < 7 ∧ 
  0 < c ∧ c < 7 ∧ 
  (a * b * c) % 7 = 2 ∧ 
  (3 * c) % 7 = 4 ∧ 
  (4 * b) % 7 = (2 + b) % 7 → 
  (a + b + c) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l1358_135880


namespace NUMINAMATH_CALUDE_smallest_d_for_perfect_square_l1358_135846

theorem smallest_d_for_perfect_square : ∃ (n : ℕ), 
  14 * 3150 = n^2 ∧ 
  ∀ (d : ℕ), d > 0 ∧ d < 14 → ¬∃ (m : ℕ), d * 3150 = m^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_for_perfect_square_l1358_135846


namespace NUMINAMATH_CALUDE_smallest_four_digit_solution_l1358_135813

theorem smallest_four_digit_solution (x : ℕ) : x = 1094 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y, y ≥ 1000 ∧ y < 10000 →
    (9 * y ≡ 27 [ZMOD 18] ∧
     3 * y + 5 ≡ 11 [ZMOD 7] ∧
     -3 * y + 2 ≡ 2 * y [ZMOD 16]) →
    x ≤ y) ∧
  (9 * x ≡ 27 [ZMOD 18]) ∧
  (3 * x + 5 ≡ 11 [ZMOD 7]) ∧
  (-3 * x + 2 ≡ 2 * x [ZMOD 16]) := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_solution_l1358_135813


namespace NUMINAMATH_CALUDE_company_p_employee_count_l1358_135811

theorem company_p_employee_count (jan_employees : ℝ) : 
  jan_employees * 1.10 * 1.15 * 1.20 = 470 →
  ⌊jan_employees⌋ = 310 := by
  sorry

end NUMINAMATH_CALUDE_company_p_employee_count_l1358_135811


namespace NUMINAMATH_CALUDE_building_height_ratio_l1358_135827

/-- Proves the ratio of building heights given specific conditions -/
theorem building_height_ratio :
  let h₁ : ℝ := 600  -- Height of first building
  let h₂ : ℝ := 2 * h₁  -- Height of second building
  let h_total : ℝ := 7200  -- Total height of all three buildings
  let h₃ : ℝ := h_total - (h₁ + h₂)  -- Height of third building
  (h₃ / (h₁ + h₂) = 3) :=
by sorry

end NUMINAMATH_CALUDE_building_height_ratio_l1358_135827


namespace NUMINAMATH_CALUDE_prize_probability_l1358_135829

/-- The probability of at least one person winning a prize when 5 people each buy 1 ticket
    from a pool of 10 tickets, where 3 tickets have prizes. -/
theorem prize_probability (total_tickets : ℕ) (prize_tickets : ℕ) (buyers : ℕ) :
  total_tickets = 10 →
  prize_tickets = 3 →
  buyers = 5 →
  (1 : ℚ) - (Nat.choose (total_tickets - prize_tickets) buyers : ℚ) / (Nat.choose total_tickets buyers : ℚ) = 11/12 :=
by sorry

end NUMINAMATH_CALUDE_prize_probability_l1358_135829


namespace NUMINAMATH_CALUDE_extreme_point_inequality_l1358_135856

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1) + (1/2) * x^2 - x

theorem extreme_point_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  0 < a → a < 1 →
  x₁ < x₂ →
  x₁ = -Real.sqrt (1 - a) →
  x₂ = Real.sqrt (1 - a) →
  f a x₁ < f a x₂ →
  f a x₂ > f a x₁ →
  f a x₂ / x₁ < 1/2 := by
sorry

end NUMINAMATH_CALUDE_extreme_point_inequality_l1358_135856


namespace NUMINAMATH_CALUDE_square_area_proof_l1358_135804

theorem square_area_proof (x : ℝ) : 
  (5 * x - 18 = 27 - 4 * x) → 
  ((5 * x - 18)^2 : ℝ) = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l1358_135804


namespace NUMINAMATH_CALUDE_cosine_theorem_triangle_l1358_135841

theorem cosine_theorem_triangle (a b c : ℝ) (A : ℝ) :
  a = 3 → b = 4 → A = π / 3 → c^2 = a^2 + b^2 - 2 * a * b * Real.cos A → c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_cosine_theorem_triangle_l1358_135841


namespace NUMINAMATH_CALUDE_incorrect_multiplication_l1358_135828

theorem incorrect_multiplication (correct_multiplier : ℕ) (number_to_multiply : ℕ) (difference : ℕ) 
  (h1 : correct_multiplier = 43)
  (h2 : number_to_multiply = 134)
  (h3 : difference = 1206) :
  ∃ (incorrect_multiplier : ℕ), 
    number_to_multiply * correct_multiplier - number_to_multiply * incorrect_multiplier = difference ∧
    incorrect_multiplier = 34 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_multiplication_l1358_135828


namespace NUMINAMATH_CALUDE_martha_children_count_l1358_135886

theorem martha_children_count (total_cakes num_cakes_per_child : ℕ) 
  (h1 : total_cakes = 18)
  (h2 : num_cakes_per_child = 6)
  (h3 : total_cakes % num_cakes_per_child = 0) :
  total_cakes / num_cakes_per_child = 3 := by
  sorry

end NUMINAMATH_CALUDE_martha_children_count_l1358_135886


namespace NUMINAMATH_CALUDE_rachels_homework_l1358_135852

theorem rachels_homework (total_pages math_pages : ℕ) 
  (h1 : total_pages = 7) 
  (h2 : math_pages = 5) : 
  total_pages - math_pages = 2 := by
sorry

end NUMINAMATH_CALUDE_rachels_homework_l1358_135852


namespace NUMINAMATH_CALUDE_geometric_series_r_value_l1358_135805

/-- Given a geometric series with first term a and common ratio r,
    S is the sum of the entire series,
    S_odd is the sum of terms with odd powers of r -/
def geometric_series (a r : ℝ) (S S_odd : ℝ) : Prop :=
  ∃ (n : ℕ), S = a * (1 - r^n) / (1 - r) ∧
             S_odd = a * r * (1 - r^(2*n)) / (1 - r^2)

theorem geometric_series_r_value (a r : ℝ) :
  geometric_series a r 20 8 → r = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_r_value_l1358_135805


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1358_135825

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1358_135825


namespace NUMINAMATH_CALUDE_edge_count_of_specific_polyhedron_l1358_135881

/-- A simple polyhedron is a polyhedron where each edge connects exactly two vertices and is part of exactly two faces. -/
structure SimplePolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- Euler's formula for simple polyhedra: F + V = E + 2 -/
axiom eulers_formula (p : SimplePolyhedron) : p.faces + p.vertices = p.edges + 2

theorem edge_count_of_specific_polyhedron :
  ∃ (p : SimplePolyhedron), p.faces = 12 ∧ p.vertices = 20 ∧ p.edges = 30 := by
  sorry

end NUMINAMATH_CALUDE_edge_count_of_specific_polyhedron_l1358_135881


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1358_135854

theorem complex_modulus_problem (i : ℂ) (h : i * i = -1) :
  Complex.abs (1 / (1 - i) + i) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1358_135854


namespace NUMINAMATH_CALUDE_no_real_a_for_single_solution_l1358_135859

theorem no_real_a_for_single_solution :
  ¬ ∃ (a : ℝ), ∃! (x : ℝ), |x^2 + 4*a*x + 5*a| ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_a_for_single_solution_l1358_135859


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_quadratics_l1358_135822

theorem unique_prime_with_prime_quadratics :
  ∃! p : ℕ, Nat.Prime p ∧ Nat.Prime (4 * p^2 + 1) ∧ Nat.Prime (6 * p^2 + 1) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_quadratics_l1358_135822


namespace NUMINAMATH_CALUDE_product_of_ab_l1358_135875

theorem product_of_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 31) : a * b = -11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_ab_l1358_135875


namespace NUMINAMATH_CALUDE_negation_of_existence_l1358_135885

theorem negation_of_existence (x : ℝ) :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1358_135885


namespace NUMINAMATH_CALUDE_b_geq_0_sufficient_not_necessary_for_a_squared_plus_b_geq_0_l1358_135878

theorem b_geq_0_sufficient_not_necessary_for_a_squared_plus_b_geq_0 :
  ∃ (a b : ℝ), (b ≥ 0 → a^2 + b ≥ 0) ∧ ¬(a^2 + b ≥ 0 → b ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_b_geq_0_sufficient_not_necessary_for_a_squared_plus_b_geq_0_l1358_135878


namespace NUMINAMATH_CALUDE_open_box_volume_is_24000_l1358_135806

/-- Represents the dimensions of a rectangular sheet -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a parallelogram cut from corners -/
structure ParallelogramCut where
  base : ℝ
  height : ℝ

/-- Calculates the volume of the open box created from a sheet with given dimensions and corner cuts -/
def openBoxVolume (sheet : SheetDimensions) (cut : ParallelogramCut) : ℝ :=
  (sheet.length - 2 * cut.base) * (sheet.width - 2 * cut.base) * cut.height

/-- Theorem stating that the volume of the open box is 24000 m^3 -/
theorem open_box_volume_is_24000 (sheet : SheetDimensions) (cut : ParallelogramCut)
    (h1 : sheet.length = 100)
    (h2 : sheet.width = 50)
    (h3 : cut.base = 10)
    (h4 : cut.height = 10) :
    openBoxVolume sheet cut = 24000 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_is_24000_l1358_135806


namespace NUMINAMATH_CALUDE_remainder_problem_l1358_135848

theorem remainder_problem (d : ℕ) (h1 : d = 170) (h2 : d ∣ (690 - 10)) (h3 : ∃ r, d ∣ (875 - r)) :
  875 % d = 25 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1358_135848


namespace NUMINAMATH_CALUDE_danny_collection_difference_l1358_135857

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 11

/-- The number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 28

/-- The difference between wrappers and bottle caps found at the park -/
def difference : ℕ := wrappers_found - bottle_caps_found

theorem danny_collection_difference : difference = 17 := by
  sorry

end NUMINAMATH_CALUDE_danny_collection_difference_l1358_135857


namespace NUMINAMATH_CALUDE_units_digit_power_seven_l1358_135801

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_power_seven (exponent : ℕ) : 
  (∃ k : ℕ, units_digit ((7^1)^exponent) = 9 ∧ ∀ m : ℕ, m < exponent → units_digit ((7^1)^m) ≠ 9) → 
  exponent = 2 :=
sorry

end NUMINAMATH_CALUDE_units_digit_power_seven_l1358_135801


namespace NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l1358_135814

/-- The vertex of a parabola defined by y = a(x-h)^2 + k has coordinates (h, k) -/
theorem parabola_vertex (a h k : ℝ) :
  let f : ℝ → ℝ := λ x => a * (x - h)^2 + k
  (∀ x, f x ≥ f h) ∧ f h = k :=
by sorry

/-- The vertex of the parabola y = 3(x-5)^2 + 4 has coordinates (5, 4) -/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x => 3 * (x - 5)^2 + 4
  (∀ x, f x ≥ f 5) ∧ f 5 = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l1358_135814


namespace NUMINAMATH_CALUDE_no_four_polynomials_exist_l1358_135850

-- Define a type for polynomials with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define a predicate to check if a polynomial has a real root
def has_real_root (p : RealPolynomial) : Prop :=
  ∃ x : ℝ, p x = 0

-- Define a predicate to check if a polynomial has no real root
def has_no_real_root (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, p x ≠ 0

theorem no_four_polynomials_exist :
  ¬ ∃ (P₁ P₂ P₃ P₄ : RealPolynomial),
    (has_real_root (λ x => P₁ x + P₂ x + P₃ x)) ∧
    (has_real_root (λ x => P₁ x + P₂ x + P₄ x)) ∧
    (has_real_root (λ x => P₁ x + P₃ x + P₄ x)) ∧
    (has_real_root (λ x => P₂ x + P₃ x + P₄ x)) ∧
    (has_no_real_root (λ x => P₁ x + P₂ x)) ∧
    (has_no_real_root (λ x => P₁ x + P₃ x)) ∧
    (has_no_real_root (λ x => P₁ x + P₄ x)) ∧
    (has_no_real_root (λ x => P₂ x + P₃ x)) ∧
    (has_no_real_root (λ x => P₂ x + P₄ x)) ∧
    (has_no_real_root (λ x => P₃ x + P₄ x)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_four_polynomials_exist_l1358_135850


namespace NUMINAMATH_CALUDE_smallest_a_value_l1358_135882

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem stating the smallest possible value of a for a parabola with given conditions -/
theorem smallest_a_value (p : Parabola) 
  (vertex_x : p.a * (3/5)^2 + p.b * (3/5) + p.c = -25/12)  -- vertex condition
  (pos_a : p.a > 0)  -- a > 0
  (int_sum : ∃ n : ℤ, p.a + p.b + p.c = n)  -- a + b + c is an integer
  : p.a ≥ 25/48 := by
  sorry


end NUMINAMATH_CALUDE_smallest_a_value_l1358_135882


namespace NUMINAMATH_CALUDE_onions_sum_is_eighteen_l1358_135802

/-- The total number of onions grown by Sara, Sally, and Fred -/
def total_onions (sara_onions sally_onions fred_onions : ℕ) : ℕ :=
  sara_onions + sally_onions + fred_onions

/-- Theorem stating that the total number of onions grown is 18 -/
theorem onions_sum_is_eighteen :
  total_onions 4 5 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_onions_sum_is_eighteen_l1358_135802


namespace NUMINAMATH_CALUDE_min_m_value_l1358_135803

-- Define the points A and B
def A (m : ℝ) : ℝ × ℝ := (1, m)
def B (x : ℝ) : ℝ × ℝ := (-1, 1 - |x|)

-- Define symmetry with respect to the origin
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

-- State the theorem
theorem min_m_value (m x : ℝ) 
  (h : symmetric_wrt_origin (A m) (B x)) : 
  ∀ k, m ≤ k → -1 ≤ k :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l1358_135803


namespace NUMINAMATH_CALUDE_day5_sale_correct_l1358_135867

/-- Represents the sales data for a grocer over 6 days -/
structure GrocerSales where
  average_target : ℕ  -- Target average sale for 5 consecutive days
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day5 : ℕ  -- The day we want to calculate
  day6 : ℕ
  total_days : ℕ  -- Number of days for average calculation

/-- Calculates the required sale on the fifth day to meet the average target -/
def calculate_day5_sale (sales : GrocerSales) : ℕ :=
  sales.average_target * sales.total_days - (sales.day1 + sales.day2 + sales.day3 + sales.day5 + sales.day6)

/-- Theorem stating that the calculated sale for day 5 is correct -/
theorem day5_sale_correct (sales : GrocerSales) 
  (h1 : sales.average_target = 625)
  (h2 : sales.day1 = 435)
  (h3 : sales.day2 = 927)
  (h4 : sales.day3 = 855)
  (h5 : sales.day5 = 562)
  (h6 : sales.day6 = 741)
  (h7 : sales.total_days = 5) :
  calculate_day5_sale sales = 167 := by
  sorry

#eval calculate_day5_sale { 
  average_target := 625, 
  day1 := 435, 
  day2 := 927, 
  day3 := 855, 
  day5 := 562, 
  day6 := 741, 
  total_days := 5 
}

end NUMINAMATH_CALUDE_day5_sale_correct_l1358_135867


namespace NUMINAMATH_CALUDE_monomial_properties_l1358_135873

/-- Represents a monomial in two variables -/
structure Monomial (α : Type*) [Ring α] where
  coefficient : α
  exponent_a : ℕ
  exponent_b : ℕ

/-- Calculate the degree of a monomial -/
def Monomial.degree {α : Type*} [Ring α] (m : Monomial α) : ℕ :=
  m.exponent_a + m.exponent_b

/-- The monomial -2a²b -/
def example_monomial : Monomial ℤ :=
  { coefficient := -2
    exponent_a := 2
    exponent_b := 1 }

theorem monomial_properties :
  (example_monomial.coefficient = -2) ∧
  (example_monomial.degree = 3) := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l1358_135873


namespace NUMINAMATH_CALUDE_square_of_difference_three_minus_sqrt_two_l1358_135862

theorem square_of_difference_three_minus_sqrt_two : (3 - Real.sqrt 2)^2 = 11 - 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_three_minus_sqrt_two_l1358_135862


namespace NUMINAMATH_CALUDE_absolute_value_of_w_l1358_135874

theorem absolute_value_of_w (w : ℂ) : w^2 - 6*w + 40 = 0 → Complex.abs w = Real.sqrt 40 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_w_l1358_135874


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1358_135894

theorem complex_expression_simplification :
  (-3 : ℂ) + 7 * Complex.I - 3 * (2 - 5 * Complex.I) + 4 * Complex.I = -9 + 26 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1358_135894


namespace NUMINAMATH_CALUDE_degenerate_ellipse_iff_c_eq_neg_nine_l1358_135861

/-- Represents the equation of an ellipse with a parameter c -/
def ellipse_equation (x y c : ℝ) : Prop :=
  x^2 + 9*y^2 - 6*x + 18*y = c

/-- Determines if the ellipse is degenerate (i.e., a point) -/
def is_degenerate_ellipse (c : ℝ) : Prop :=
  ∃ x₀ y₀ : ℝ, ∀ x y : ℝ, ellipse_equation x y c → x = x₀ ∧ y = y₀

/-- Theorem stating that the ellipse is degenerate if and only if c = -9 -/
theorem degenerate_ellipse_iff_c_eq_neg_nine :
  ∀ c : ℝ, is_degenerate_ellipse c ↔ c = -9 :=
sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_iff_c_eq_neg_nine_l1358_135861


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l1358_135842

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), n^300 < 3^500 ∧ ∀ (m : ℕ), m^300 < 3^500 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l1358_135842


namespace NUMINAMATH_CALUDE_integer_solutions_count_l1358_135863

theorem integer_solutions_count : ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ |7*x - 4| ≤ 14) ∧ Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l1358_135863


namespace NUMINAMATH_CALUDE_empty_set_proof_l1358_135849

theorem empty_set_proof : {x : ℝ | x^2 - x + 1 = 0} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_set_proof_l1358_135849


namespace NUMINAMATH_CALUDE_picnic_meals_count_l1358_135800

/-- The number of sandwich options available. -/
def num_sandwiches : ℕ := 4

/-- The number of salad options available. -/
def num_salads : ℕ := 5

/-- The number of drink options available. -/
def num_drinks : ℕ := 3

/-- The number of salads Julia must choose. -/
def salads_to_choose : ℕ := 3

/-- Calculate the number of ways to choose k items from n options. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of different picnic meals Julia can create. -/
theorem picnic_meals_count : 
  num_sandwiches * choose num_salads salads_to_choose * num_drinks = 120 := by
  sorry

end NUMINAMATH_CALUDE_picnic_meals_count_l1358_135800


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1358_135899

theorem magnitude_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1358_135899


namespace NUMINAMATH_CALUDE_blonde_to_total_ratio_l1358_135888

/-- Given a class with a specific hair color ratio and number of students, 
    prove the ratio of blonde-haired children to total children -/
theorem blonde_to_total_ratio 
  (red_ratio : ℕ) (blonde_ratio : ℕ) (black_ratio : ℕ)
  (red_count : ℕ) (total_count : ℕ)
  (h1 : red_ratio = 3)
  (h2 : blonde_ratio = 6)
  (h3 : black_ratio = 7)
  (h4 : red_count = 9)
  (h5 : total_count = 48)
  : (blonde_ratio * red_count / red_ratio) / total_count = 3 / 8 := by
  sorry

#check blonde_to_total_ratio

end NUMINAMATH_CALUDE_blonde_to_total_ratio_l1358_135888


namespace NUMINAMATH_CALUDE_probability_identical_value_l1358_135853

/-- Represents the colors that can be used to paint a cube face -/
inductive Color
| Red
| Blue

/-- Represents a cube with painted faces -/
def Cube := Fin 6 → Color

/-- Checks if two cubes are identical after rotation -/
def identical_after_rotation (cube1 cube2 : Cube) : Prop := sorry

/-- The set of all possible cube paintings -/
def all_cubes : Set Cube := sorry

/-- The set of pairs of cubes that are identical after rotation -/
def identical_pairs : Set (Cube × Cube) := sorry

/-- The probability of two independently painted cubes being identical after rotation -/
def probability_identical : ℚ := sorry

theorem probability_identical_value :
  probability_identical = 459 / 4096 := by sorry

end NUMINAMATH_CALUDE_probability_identical_value_l1358_135853


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l1358_135865

theorem right_triangle_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (right_triangle : a^2 + b^2 = c^2) (leg : b = 5) (hyp : c = 13) :
  a / c = 12 / 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l1358_135865


namespace NUMINAMATH_CALUDE_line_translation_l1358_135866

/-- Given a line y = 2x translated by vector (m, n) to y = 2x + 5, 
    prove the relationship between m and n. -/
theorem line_translation (m n : ℝ) : 
  (∀ x y : ℝ, y = 2*x + 5 ↔ y - n = 2*(x - m)) → n = 2*m + 5 := by
sorry

end NUMINAMATH_CALUDE_line_translation_l1358_135866


namespace NUMINAMATH_CALUDE_students_in_two_courses_l1358_135890

theorem students_in_two_courses 
  (total : ℕ) 
  (math : ℕ) 
  (chinese : ℕ) 
  (international : ℕ) 
  (all_three : ℕ) 
  (none : ℕ) 
  (h1 : total = 400) 
  (h2 : math = 169) 
  (h3 : chinese = 158) 
  (h4 : international = 145) 
  (h5 : all_three = 30) 
  (h6 : none = 20) : 
  ∃ (two_courses : ℕ), 
    two_courses = 32 ∧ 
    total = math + chinese + international - two_courses - 2 * all_three + none :=
by sorry

end NUMINAMATH_CALUDE_students_in_two_courses_l1358_135890


namespace NUMINAMATH_CALUDE_starting_number_is_100_l1358_135819

/-- The starting number of a range ending at 400, where the average of the integers
    in this range is 100 greater than the average of the integers from 50 to 250. -/
def starting_number : ℤ :=
  let avg_50_to_250 := (50 + 250) / 2
  let avg_x_to_400 := avg_50_to_250 + 100
  2 * avg_x_to_400 - 400

theorem starting_number_is_100 : starting_number = 100 := by
  sorry

end NUMINAMATH_CALUDE_starting_number_is_100_l1358_135819


namespace NUMINAMATH_CALUDE_cylinder_base_area_ratio_l1358_135831

/-- Represents a cylinder with base area S and volume V -/
structure Cylinder where
  S : ℝ
  V : ℝ

/-- 
Given two cylinders with equal lateral areas and a volume ratio of 3/2,
prove that the ratio of their base areas is 9/4
-/
theorem cylinder_base_area_ratio 
  (A B : Cylinder) 
  (h1 : A.V / B.V = 3 / 2) 
  (h2 : ∃ (r₁ r₂ h₁ h₂ : ℝ), 
    A.S = π * r₁^2 ∧ 
    B.S = π * r₂^2 ∧ 
    A.V = π * r₁^2 * h₁ ∧ 
    B.V = π * r₂^2 * h₂ ∧ 
    2 * π * r₁ * h₁ = 2 * π * r₂ * h₂) : 
  A.S / B.S = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_cylinder_base_area_ratio_l1358_135831


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1358_135830

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (3 * x + y = 8) ∧ (2 * x - y = 7) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1358_135830


namespace NUMINAMATH_CALUDE_lanas_muffin_goal_l1358_135858

/-- Lana's muffin sale problem -/
theorem lanas_muffin_goal (morning_sales afternoon_sales more_needed : ℕ) 
  (h1 : morning_sales = 12)
  (h2 : afternoon_sales = 4)
  (h3 : more_needed = 4) :
  morning_sales + afternoon_sales + more_needed = 20 := by
  sorry

end NUMINAMATH_CALUDE_lanas_muffin_goal_l1358_135858


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1358_135844

/-- The surface area of a sphere, given specific conditions for a hemisphere --/
theorem sphere_surface_area (r : ℝ) (h1 : π * r^2 = 3) (h2 : 3 * π * r^2 = 9) :
  ∃ S : ℝ → ℝ, ∀ x : ℝ, S x = 4 * π * x^2 := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1358_135844


namespace NUMINAMATH_CALUDE_quadratic_range_at_minus_two_l1358_135870

/-- A quadratic function passing through the origin -/
structure QuadraticThroughOrigin where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0

/-- The quadratic function f(x) = ax² + bx -/
def f (q : QuadraticThroughOrigin) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x

/-- Theorem: For a quadratic function f(x) = ax² + bx (a ≠ 0) passing through the origin,
    if 1 ≤ f(-1) ≤ 2 and 2 ≤ f(1) ≤ 4, then 5 ≤ f(-2) ≤ 10 -/
theorem quadratic_range_at_minus_two (q : QuadraticThroughOrigin) 
    (h1 : 1 ≤ f q (-1)) (h2 : f q (-1) ≤ 2)
    (h3 : 2 ≤ f q 1) (h4 : f q 1 ≤ 4) :
    5 ≤ f q (-2) ∧ f q (-2) ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_at_minus_two_l1358_135870


namespace NUMINAMATH_CALUDE_approximation_of_2026_l1358_135815

def approximate_to_hundredth (x : ℚ) : ℚ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem approximation_of_2026 :
  approximate_to_hundredth 2.026 = 2.03 := by
  sorry

end NUMINAMATH_CALUDE_approximation_of_2026_l1358_135815
