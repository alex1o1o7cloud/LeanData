import Mathlib

namespace NUMINAMATH_CALUDE_parallel_vectors_m_zero_l4045_404591

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_zero :
  let a : ℝ × ℝ := (-2, 3)
  let b : ℝ × ℝ := (1, m - 3/2)
  parallel a b → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_zero_l4045_404591


namespace NUMINAMATH_CALUDE_average_of_combined_data_points_l4045_404534

theorem average_of_combined_data_points (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  let total_points := n1 + n2
  let combined_avg := (n1 * avg1 + n2 * avg2) / total_points
  combined_avg = (n1 * avg1 + n2 * avg2) / (n1 + n2) :=
by sorry

end NUMINAMATH_CALUDE_average_of_combined_data_points_l4045_404534


namespace NUMINAMATH_CALUDE_multiple_of_nine_is_multiple_of_three_l4045_404505

theorem multiple_of_nine_is_multiple_of_three (n : ℤ) : 
  (∃ k : ℤ, n = 9 * k) → (∃ m : ℤ, n = 3 * m) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_is_multiple_of_three_l4045_404505


namespace NUMINAMATH_CALUDE_no_lcm_arithmetic_progression_l4045_404511

theorem no_lcm_arithmetic_progression (n : ℕ) (h : n > 100) :
  ¬ ∃ (S : Finset ℕ) (d : ℕ) (first : ℕ),
    S.card = n ∧
    (∀ x ∈ S, ∀ y ∈ S, x ≠ y) ∧
    d > 0 ∧
    ∃ (f : Finset ℕ),
      f.card = n * (n - 1) / 2 ∧
      (∀ x ∈ S, ∀ y ∈ S, x < y → Nat.lcm x y ∈ f) ∧
      (∀ i < n * (n - 1) / 2, first + i * d ∈ f) :=
by sorry

end NUMINAMATH_CALUDE_no_lcm_arithmetic_progression_l4045_404511


namespace NUMINAMATH_CALUDE_license_plate_count_l4045_404586

/-- The number of possible letters for each position in the license plate -/
def num_letters : ℕ := 26

/-- The number of possible odd digits -/
def num_odd_digits : ℕ := 5

/-- The number of possible even digits -/
def num_even_digits : ℕ := 5

/-- The total number of license plates with the given conditions -/
def total_license_plates : ℕ := num_letters^3 * num_odd_digits * (num_odd_digits * num_even_digits + num_even_digits * num_odd_digits)

theorem license_plate_count : total_license_plates = 455625 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l4045_404586


namespace NUMINAMATH_CALUDE_cube_root_four_solution_l4045_404565

theorem cube_root_four_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_four_solution_l4045_404565


namespace NUMINAMATH_CALUDE_work_relation_l4045_404559

/-- Represents an isothermal process of a gas -/
structure IsothermalProcess where
  pressure : ℝ → ℝ
  volume : ℝ → ℝ
  work : ℝ

/-- The work done on a gas during an isothermal process -/
def work_done (p : IsothermalProcess) : ℝ := p.work

/-- Condition: The volume in process 1-2 is twice the volume in process 3-4 for any given pressure -/
def volume_relation (p₁₂ p₃₄ : IsothermalProcess) : Prop :=
  ∀ t, p₁₂.volume t = 2 * p₃₄.volume t

/-- Theorem: The work done on the gas in process 1-2 is twice the work done in process 3-4 -/
theorem work_relation (p₁₂ p₃₄ : IsothermalProcess) 
  (h : volume_relation p₁₂ p₃₄) : 
  work_done p₁₂ = 2 * work_done p₃₄ := by
  sorry

end NUMINAMATH_CALUDE_work_relation_l4045_404559


namespace NUMINAMATH_CALUDE_associate_professor_pencils_l4045_404514

theorem associate_professor_pencils :
  ∀ (A B P : ℕ),
    A + B = 8 →
    P * A + B = 10 →
    A + 2 * B = 14 →
    P = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_associate_professor_pencils_l4045_404514


namespace NUMINAMATH_CALUDE_f_of_one_equals_fourteen_l4045_404500

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + 2 * x - 8

-- State the theorem
theorem f_of_one_equals_fourteen 
  (a b : ℝ) -- Parameters of the function
  (h : f a b (-1) = 10) -- Given condition
  : f a b 1 = 14 := by
  sorry -- Proof is omitted

end NUMINAMATH_CALUDE_f_of_one_equals_fourteen_l4045_404500


namespace NUMINAMATH_CALUDE_weight_of_doubled_cube_l4045_404553

/-- Given a cube of metal weighing 7 pounds, prove that another cube of the same metal
    with sides twice as long will weigh 56 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (ρ : ℝ) (h : ρ * s^3 = 7) :
  ρ * (2*s)^3 = 56 := by
sorry

end NUMINAMATH_CALUDE_weight_of_doubled_cube_l4045_404553


namespace NUMINAMATH_CALUDE_three_hundred_thousand_cubed_times_fifty_l4045_404550

theorem three_hundred_thousand_cubed_times_fifty :
  (300000 ^ 3) * 50 = 1350000000000000000 := by sorry

end NUMINAMATH_CALUDE_three_hundred_thousand_cubed_times_fifty_l4045_404550


namespace NUMINAMATH_CALUDE_lindsey_video_game_cost_l4045_404555

/-- The cost of Lindsey's video game -/
def video_game_cost (september_savings : ℕ) (october_savings : ℕ) (november_savings : ℕ) 
  (mom_gift : ℕ) (savings_threshold : ℕ) (amount_left : ℕ) : ℕ :=
  let total_savings := september_savings + october_savings + november_savings
  let total_with_gift := if total_savings > savings_threshold then total_savings + mom_gift else total_savings
  total_with_gift - amount_left

/-- Theorem stating the cost of Lindsey's video game -/
theorem lindsey_video_game_cost :
  video_game_cost 50 37 11 25 75 36 = 87 := by
  sorry

end NUMINAMATH_CALUDE_lindsey_video_game_cost_l4045_404555


namespace NUMINAMATH_CALUDE_tv_production_last_five_days_l4045_404512

theorem tv_production_last_five_days 
  (total_days : Nat) 
  (first_period : Nat) 
  (avg_first_period : Nat) 
  (avg_total : Nat) 
  (h1 : total_days = 30)
  (h2 : first_period = 25)
  (h3 : avg_first_period = 50)
  (h4 : avg_total = 48) :
  (total_days * avg_total - first_period * avg_first_period) / (total_days - first_period) = 38 := by
  sorry

#check tv_production_last_five_days

end NUMINAMATH_CALUDE_tv_production_last_five_days_l4045_404512


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l4045_404564

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := 27

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def total_diagonal_pairs (n : RegularNonagon) : ℕ := 351

/-- The number of ways to choose 4 vertices from the nonagon, 
    which correspond to intersecting diagonals -/
def intersecting_diagonal_pairs (n : RegularNonagon) : ℕ := 126

/-- The probability that two randomly chosen diagonals in a regular nonagon intersect -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  intersecting_diagonal_pairs n / total_diagonal_pairs n

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry


end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l4045_404564


namespace NUMINAMATH_CALUDE_S_excludes_A_and_B_only_l4045_404548

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -2)

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ((p.1 - 1)^2 + (p.2 - 1)^2) * ((p.1 - 2)^2 + (p.2 + 2)^2) ≠ 0}

theorem S_excludes_A_and_B_only :
  ∀ p : ℝ × ℝ, p ∉ S ↔ p = A ∨ p = B := by sorry

end NUMINAMATH_CALUDE_S_excludes_A_and_B_only_l4045_404548


namespace NUMINAMATH_CALUDE_house_painting_time_l4045_404577

/-- Represents the time taken to paint a house given individual rates and a break -/
theorem house_painting_time
  (alice_rate : ℝ) (bob_rate : ℝ) (carlos_rate : ℝ) (break_time : ℝ) (total_time : ℝ)
  (h_alice : alice_rate = 1 / 4)
  (h_bob : bob_rate = 1 / 6)
  (h_carlos : carlos_rate = 1 / 12)
  (h_break : break_time = 2)
  (h_equation : (alice_rate + bob_rate + carlos_rate) * (total_time - break_time) = 1) :
  (1 / 4 + 1 / 6 + 1 / 12) * (total_time - 2) = 1 := by
sorry


end NUMINAMATH_CALUDE_house_painting_time_l4045_404577


namespace NUMINAMATH_CALUDE_laura_running_speed_approx_l4045_404507

/-- Laura's workout parameters --/
structure WorkoutParams where
  totalDuration : ℝ  -- Total workout duration in minutes
  bikingDistance : ℝ  -- Biking distance in miles
  transitionTime : ℝ  -- Transition time in minutes
  runningDistance : ℝ  -- Running distance in miles

/-- Calculate Laura's running speed given workout parameters --/
def calculateRunningSpeed (params : WorkoutParams) (x : ℝ) : ℝ :=
  x^2 - 1

/-- Theorem stating that Laura's running speed is approximately 83.33 mph --/
theorem laura_running_speed_approx (params : WorkoutParams) :
  ∃ x : ℝ,
    params.totalDuration = 150 ∧
    params.bikingDistance = 30 ∧
    params.transitionTime = 10 ∧
    params.runningDistance = 5 ∧
    (params.totalDuration - params.transitionTime) / 60 = params.bikingDistance / (3*x + 2) + params.runningDistance / (x^2 - 1) ∧
    abs (calculateRunningSpeed params x - 83.33) < 0.01 :=
  sorry


end NUMINAMATH_CALUDE_laura_running_speed_approx_l4045_404507


namespace NUMINAMATH_CALUDE_certain_number_value_l4045_404587

theorem certain_number_value : ∃ x : ℝ, 
  (3 - (1/5) * 390 = x - (1/7) * 210 + 114) ∧ 
  (3 - (1/5) * 390 - (x - (1/7) * 210) = 114) → 
  x = -159 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l4045_404587


namespace NUMINAMATH_CALUDE_two_ways_to_combine_fractions_l4045_404543

theorem two_ways_to_combine_fractions : ∃ (f g : ℚ → ℚ → ℚ → ℚ),
  f (1/8) (1/9) (1/28) = 1/2016 ∧
  g (1/8) (1/9) (1/28) = 1/2016 ∧
  f ≠ g :=
by sorry

end NUMINAMATH_CALUDE_two_ways_to_combine_fractions_l4045_404543


namespace NUMINAMATH_CALUDE_problem_solution_l4045_404569

theorem problem_solution (x y : ℚ) : 
  x / y = 12 / 5 → y = 25 → x = 60 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l4045_404569


namespace NUMINAMATH_CALUDE_debby_candy_count_debby_candy_count_proof_l4045_404562

theorem debby_candy_count : ℕ → Prop :=
  fun d : ℕ =>
    (∃ (sister_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ),
      sister_candy = 42 ∧
      eaten_candy = 35 ∧
      remaining_candy = 39 ∧
      d + sister_candy - eaten_candy = remaining_candy) →
    d = 32

-- Proof
theorem debby_candy_count_proof : debby_candy_count 32 := by
  sorry

end NUMINAMATH_CALUDE_debby_candy_count_debby_candy_count_proof_l4045_404562


namespace NUMINAMATH_CALUDE_total_arrangements_eq_192_l4045_404585

/-- Represents the number of classes to be scheduled -/
def num_classes : ℕ := 6

/-- Represents the number of time slots in a day -/
def num_slots : ℕ := 6

/-- Represents the number of morning slots (first 4 periods) -/
def morning_slots : ℕ := 4

/-- Represents the number of afternoon slots (last 2 periods) -/
def afternoon_slots : ℕ := 2

/-- The number of ways to arrange the Chinese class in the morning -/
def chinese_arrangements : ℕ := morning_slots

/-- The number of ways to arrange the Biology class in the afternoon -/
def biology_arrangements : ℕ := afternoon_slots

/-- The number of remaining classes after scheduling Chinese and Biology -/
def remaining_classes : ℕ := num_classes - 2

/-- The number of remaining slots after scheduling Chinese and Biology -/
def remaining_slots : ℕ := num_slots - 2

/-- Calculates the total number of possible arrangements -/
def total_arrangements : ℕ :=
  chinese_arrangements * biology_arrangements * (remaining_classes.factorial)

/-- Theorem stating that the total number of arrangements is 192 -/
theorem total_arrangements_eq_192 : total_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_eq_192_l4045_404585


namespace NUMINAMATH_CALUDE_midpoint_locus_l4045_404502

/-- The locus of midpoints of line segments from P(4, -2) to points on x^2 + y^2 = 4 -/
theorem midpoint_locus (x y u v : ℝ) : 
  (u^2 + v^2 = 4) →  -- Point (u, v) is on the circle
  (x = (u + 4) / 2 ∧ y = (v - 2) / 2) →  -- (x, y) is the midpoint
  (x - 2)^2 + (y + 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_locus_l4045_404502


namespace NUMINAMATH_CALUDE_possible_theta_value_l4045_404503

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + θ) + Real.sqrt 3 * Real.cos (2 * x + θ)

theorem possible_theta_value :
  ∃ θ : ℝ,
    (∀ x : ℝ, (2015 : ℝ) ^ (f θ (-x)) = 1 / ((2015 : ℝ) ^ (f θ x))) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π/4 → f θ y < f θ x) ∧
    θ = 2 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_possible_theta_value_l4045_404503


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l4045_404523

def f (x : ℝ) := 2 * x^2 - 3 * x + 1

theorem f_has_two_zeros : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b := by
  sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l4045_404523


namespace NUMINAMATH_CALUDE_min_fence_length_is_650_l4045_404594

/-- Represents a triangular grid with side length 50 meters -/
structure TriangularGrid where
  side_length : ℝ
  side_length_eq : side_length = 50

/-- Represents the number of paths between cabbage and goat areas -/
def num_paths : ℕ := 13

/-- The minimum total length of fences required to separate cabbage from goats -/
def min_fence_length (grid : TriangularGrid) : ℝ :=
  (num_paths : ℝ) * grid.side_length

/-- Theorem stating the minimum fence length required -/
theorem min_fence_length_is_650 (grid : TriangularGrid) :
  min_fence_length grid = 650 := by
  sorry

#check min_fence_length_is_650

end NUMINAMATH_CALUDE_min_fence_length_is_650_l4045_404594


namespace NUMINAMATH_CALUDE_count_valid_a_l4045_404558

-- Define the system of inequalities
def system_inequalities (a : ℤ) (x : ℤ) : Prop :=
  6 * x - 5 ≥ a ∧ (x : ℚ) / 4 - (x - 1 : ℚ) / 6 < 1 / 2

-- Define the equation
def equation (a : ℤ) (y : ℚ) : Prop :=
  4 * y - 3 * (a : ℚ) = 2 * (y - 3)

-- Main theorem
theorem count_valid_a : 
  (∃ (s : Finset ℤ), s.card = 5 ∧ 
    (∀ a : ℤ, a ∈ s ↔ 
      (∃! (sol : Finset ℤ), sol.card = 2 ∧ 
        (∀ x : ℤ, x ∈ sol ↔ system_inequalities a x)) ∧
      (∃ y : ℚ, y > 0 ∧ equation a y))) := by sorry

end NUMINAMATH_CALUDE_count_valid_a_l4045_404558


namespace NUMINAMATH_CALUDE_equal_solution_implies_k_value_l4045_404541

theorem equal_solution_implies_k_value :
  ∀ (k : ℚ), 
  (∃ (x : ℚ), 3 * x - 6 = 0 ∧ 2 * x - 5 * k = 11) →
  (∀ (x : ℚ), 3 * x - 6 = 0 ↔ 2 * x - 5 * k = 11) →
  k = -7/5 := by
sorry

end NUMINAMATH_CALUDE_equal_solution_implies_k_value_l4045_404541


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l4045_404504

def polynomial (x : ℝ) : ℝ := 5*(x^2 - 2*x^3) + 3*(2*x - 3*x^2 + x^4) - (6*x^3 - 2*x^2)

theorem coefficient_of_x_squared :
  ∃ (a b c d : ℝ), ∀ x, polynomial x = a*x^4 + b*x^3 + (-2)*x^2 + c*x + d :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l4045_404504


namespace NUMINAMATH_CALUDE_extreme_points_condition_l4045_404533

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a*x + Real.log x

theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   (∀ x > 0, f a x ≥ f a x₁ ∨ f a x ≥ f a x₂) ∧
   |f a x₁ - f a x₂| ≥ 3/4 - Real.log 2) →
  a ≥ 3 * Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_extreme_points_condition_l4045_404533


namespace NUMINAMATH_CALUDE_basketball_win_requirement_l4045_404589

theorem basketball_win_requirement (total_games : ℕ) (games_played : ℕ) (games_won : ℕ) (target_percentage : ℚ) :
  total_games = 100 →
  games_played = 60 →
  games_won = 30 →
  target_percentage = 65 / 100 →
  ∃ (remaining_wins : ℕ), 
    remaining_wins = 35 ∧
    (games_won + remaining_wins : ℚ) / total_games = target_percentage :=
by sorry

end NUMINAMATH_CALUDE_basketball_win_requirement_l4045_404589


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4045_404584

theorem min_value_quadratic (a : ℝ) : a^2 - 4*a + 9 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4045_404584


namespace NUMINAMATH_CALUDE_correct_min_jars_l4045_404575

/-- Calculates the minimum number of jars needed to fill a large pack -/
def min_jars_needed (jar_capacity : ℕ) (pack_capacity : ℕ) : ℕ :=
  let n := (pack_capacity + jar_capacity - 1) / jar_capacity
  n

theorem correct_min_jars :
  let jar_capacity : ℕ := 140
  let pack_capacity : ℕ := 2000
  min_jars_needed jar_capacity pack_capacity = 15 := by
  sorry

#eval min_jars_needed 140 2000

end NUMINAMATH_CALUDE_correct_min_jars_l4045_404575


namespace NUMINAMATH_CALUDE_certain_number_problem_l4045_404527

theorem certain_number_problem : ∃ x : ℝ, x * 12 = 0.60 * 900 ∧ x = 45 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4045_404527


namespace NUMINAMATH_CALUDE_circular_arrangement_pairs_l4045_404572

/-- Represents the number of adjacent pairs of children of the same gender -/
def adjacentPairs (total : Nat) (groups : Nat) : Nat :=
  total - groups

/-- The problem statement -/
theorem circular_arrangement_pairs (boys girls groups : Nat) 
  (h1 : boys = 15)
  (h2 : girls = 20)
  (h3 : adjacentPairs boys groups = (2 : Nat) / (3 : Nat) * adjacentPairs girls groups) :
  boys + girls - (adjacentPairs boys groups + adjacentPairs girls groups) = 10 := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_pairs_l4045_404572


namespace NUMINAMATH_CALUDE_x_plus_y_plus_2009_l4045_404595

theorem x_plus_y_plus_2009 (x y : ℝ) 
  (h1 : |x| + x + 5*y = 2) 
  (h2 : |y| - y + x = 7) : 
  x + y + 2009 = 2012 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_plus_2009_l4045_404595


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4045_404598

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → ∃ s₁ s₂ : ℝ, s₁ + s₂ = 17 ∧ s₁ * s₂ = 8 ∧ s₁^2 + s₂^2 = 273 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4045_404598


namespace NUMINAMATH_CALUDE_square_dissection_divisible_perimeter_l4045_404518

theorem square_dissection_divisible_perimeter (n : Nat) (h : n = 2015) :
  ∃ (a b : Nat), a ≤ n ∧ b ≤ n ∧ (2 * (a + b)) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_dissection_divisible_perimeter_l4045_404518


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l4045_404574

theorem product_of_sums_equals_difference_of_powers : 
  (5 + 3) * (5^2 + 3^2) * (5^4 + 3^4) * (5^8 + 3^8) * (5^16 + 3^16) * (5^32 + 3^32) * (5^64 + 3^64) = 5^128 - 3^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l4045_404574


namespace NUMINAMATH_CALUDE_difference_of_squares_l4045_404592

theorem difference_of_squares (t : ℝ) : t^2 - 121 = (t - 11) * (t + 11) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4045_404592


namespace NUMINAMATH_CALUDE_unique_u_exists_l4045_404530

-- Define the variables as natural numbers
variable (a b u k p t : ℕ)

-- Define the conditions
def condition1 : Prop := a + b = u
def condition2 : Prop := u + k = p
def condition3 : Prop := p + a = t
def condition4 : Prop := b + k + t = 20

-- Define the uniqueness condition
def unique_digits : Prop := 
  a ≠ 0 ∧ b ≠ 0 ∧ u ≠ 0 ∧ k ≠ 0 ∧ p ≠ 0 ∧ t ≠ 0 ∧
  a ≠ b ∧ a ≠ u ∧ a ≠ k ∧ a ≠ p ∧ a ≠ t ∧
  b ≠ u ∧ b ≠ k ∧ b ≠ p ∧ b ≠ t ∧
  u ≠ k ∧ u ≠ p ∧ u ≠ t ∧
  k ≠ p ∧ k ≠ t ∧
  p ≠ t

-- Theorem statement
theorem unique_u_exists :
  ∃! u : ℕ, ∃ a b k p t : ℕ,
    condition1 a b u ∧
    condition2 u k p ∧
    condition3 p a t ∧
    condition4 b k t ∧
    unique_digits a b u k p t :=
  sorry

end NUMINAMATH_CALUDE_unique_u_exists_l4045_404530


namespace NUMINAMATH_CALUDE_conference_games_count_l4045_404522

/-- The number of teams in Division A -/
def teams_a : Nat := 7

/-- The number of teams in Division B -/
def teams_b : Nat := 5

/-- The number of games each team plays against others in its division -/
def intra_division_games : Nat := 2

/-- The number of games each team plays against teams in the other division (excluding rivalry game) -/
def inter_division_games : Nat := 1

/-- The number of special pre-season rivalry games per team -/
def rivalry_games : Nat := 1

/-- The total number of conference games scheduled -/
def total_games : Nat := 
  -- Games within Division A
  (teams_a * (teams_a - 1) / 2) * intra_division_games +
  -- Games within Division B
  (teams_b * (teams_b - 1) / 2) * intra_division_games +
  -- Regular inter-division games
  teams_a * teams_b * inter_division_games +
  -- Special pre-season rivalry games
  teams_a * rivalry_games

theorem conference_games_count : total_games = 104 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_count_l4045_404522


namespace NUMINAMATH_CALUDE_concrete_mixture_percentage_l4045_404520

/-- Proves that mixing 7 tons of 80% cement mixture with 3 tons of 20% cement mixture
    results in a 62% cement mixture when making 10 tons of concrete. -/
theorem concrete_mixture_percentage : 
  let total_concrete : ℝ := 10
  let mixture_80_percent : ℝ := 7
  let mixture_20_percent : ℝ := total_concrete - mixture_80_percent
  let cement_in_80_percent : ℝ := mixture_80_percent * 0.8
  let cement_in_20_percent : ℝ := mixture_20_percent * 0.2
  let total_cement : ℝ := cement_in_80_percent + cement_in_20_percent
  total_cement / total_concrete = 0.62 := by
sorry

end NUMINAMATH_CALUDE_concrete_mixture_percentage_l4045_404520


namespace NUMINAMATH_CALUDE_coin_probability_l4045_404560

theorem coin_probability (p q : ℝ) (hq : q = 1 - p) 
  (h : (Nat.choose 10 5 : ℝ) * p^5 * q^5 = (Nat.choose 10 6 : ℝ) * p^6 * q^4) : 
  p = 6/11 := by
sorry

end NUMINAMATH_CALUDE_coin_probability_l4045_404560


namespace NUMINAMATH_CALUDE_process_termination_and_difference_l4045_404576

-- Define the lists and their properties
def List1 : Type := {l : List ℕ // ∀ x ∈ l, x % 5 = 1}
def List2 : Type := {l : List ℕ // ∀ x ∈ l, x % 5 = 4}

-- Define the operation
def operation (l1 : List1) (l2 : List2) : List1 × List2 :=
  sorry

-- Define the termination condition
def is_terminated (l1 : List1) (l2 : List2) : Prop :=
  l1.val.length = 1 ∧ l2.val.length = 1

-- Theorem statement
theorem process_termination_and_difference 
  (l1_init : List1) (l2_init : List2) : 
  ∃ (l1_final : List1) (l2_final : List2),
    (is_terminated l1_final l2_final) ∧ 
    (l1_final.val.head? ≠ l2_final.val.head?) :=
  sorry

end NUMINAMATH_CALUDE_process_termination_and_difference_l4045_404576


namespace NUMINAMATH_CALUDE_square_sum_fourth_powers_l4045_404521

theorem square_sum_fourth_powers (a b c : ℝ) 
  (h1 : a^2 - b^2 = 5)
  (h2 : a * b = 2)
  (h3 : a^2 + b^2 + c^2 = 8) :
  a^4 + b^4 + c^4 = 38 := by
sorry

end NUMINAMATH_CALUDE_square_sum_fourth_powers_l4045_404521


namespace NUMINAMATH_CALUDE_problem_statement_l4045_404525

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4045_404525


namespace NUMINAMATH_CALUDE_multiples_of_6_factors_of_72_l4045_404528

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def is_factor_of_72 (n : ℕ) : Prop := 72 % n = 0

def solution_set : Set ℕ := {6, 12, 18, 24, 36, 72}

theorem multiples_of_6_factors_of_72 :
  ∀ n : ℕ, (is_multiple_of_6 n ∧ is_factor_of_72 n) ↔ n ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_multiples_of_6_factors_of_72_l4045_404528


namespace NUMINAMATH_CALUDE_tan_addition_subtraction_formulas_l4045_404542

noncomputable section

open Real

def tan_add (a b : ℝ) : ℝ := (tan a + tan b) / (1 - tan a * tan b)
def tan_sub (a b : ℝ) : ℝ := (tan a - tan b) / (1 + tan a * tan b)

theorem tan_addition_subtraction_formulas (a b : ℝ) :
  (tan (a + b) = tan_add a b) ∧ (tan (a - b) = tan_sub a b) :=
sorry

end

end NUMINAMATH_CALUDE_tan_addition_subtraction_formulas_l4045_404542


namespace NUMINAMATH_CALUDE_telephone_probability_l4045_404568

theorem telephone_probability (p1 p2 : ℝ) (h1 : p1 = 0.2) (h2 : p2 = 0.3) :
  p1 + p2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_telephone_probability_l4045_404568


namespace NUMINAMATH_CALUDE_expression_evaluation_l4045_404590

theorem expression_evaluation :
  ∀ x : ℝ, x = -2 → x * (x^2 - 4) = 0 →
  (x - 3) / (3 * x^2 - 6 * x) * (x + 2 - 5 / (x - 2)) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4045_404590


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_306_l4045_404581

/-- The function that returns the last two digits of 8^n -/
def lastTwoDigits (n : ℕ) : ℕ := (8^n) % 100

/-- The length of the cycle of last two digits of powers of 8 -/
def cycleLength : ℕ := 6

/-- The function that returns the tens digit of a number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_8_pow_306 : tensDigit (lastTwoDigits 306) = 6 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_306_l4045_404581


namespace NUMINAMATH_CALUDE_exam_scores_l4045_404571

theorem exam_scores (average : ℝ) (difference : ℝ) 
  (h_average : average = 98) 
  (h_difference : difference = 2) : 
  ∃ (chinese math : ℝ), 
    chinese + math = 2 * average ∧ 
    math = chinese + difference ∧ 
    chinese = 97 ∧ 
    math = 99 := by
  sorry

end NUMINAMATH_CALUDE_exam_scores_l4045_404571


namespace NUMINAMATH_CALUDE_complex_power_difference_l4045_404501

theorem complex_power_difference (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^2187 - 1/(x^2187) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l4045_404501


namespace NUMINAMATH_CALUDE_max_perimeter_is_29_l4045_404547

/-- Represents a triangle with two fixed sides of length 7 and 8, and a variable third side y --/
structure Triangle where
  y : ℤ
  is_valid : 0 < y ∧ y < 7 + 8 ∧ 7 < y + 8 ∧ 8 < y + 7

/-- The perimeter of the triangle --/
def perimeter (t : Triangle) : ℤ := 7 + 8 + t.y

/-- Theorem stating that the maximum perimeter is 29 --/
theorem max_perimeter_is_29 :
  ∀ t : Triangle, perimeter t ≤ 29 ∧ ∃ t' : Triangle, perimeter t' = 29 := by
  sorry

#check max_perimeter_is_29

end NUMINAMATH_CALUDE_max_perimeter_is_29_l4045_404547


namespace NUMINAMATH_CALUDE_slope_angle_MN_l4045_404597

/-- Given points M(1, 2) and N(0, 1), the slope angle of line MN is π/4. -/
theorem slope_angle_MN : 
  let M : ℝ × ℝ := (1, 2)
  let N : ℝ × ℝ := (0, 1)
  let slope : ℝ := (M.2 - N.2) / (M.1 - N.1)
  let slope_angle : ℝ := Real.arctan slope
  slope_angle = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_MN_l4045_404597


namespace NUMINAMATH_CALUDE_total_cost_calculation_l4045_404544

def cost_per_pound : ℝ := 0.45

def sugar_weight : ℝ := 40
def flour_weight : ℝ := 16

def total_cost : ℝ := cost_per_pound * (sugar_weight + flour_weight)

theorem total_cost_calculation : total_cost = 25.20 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l4045_404544


namespace NUMINAMATH_CALUDE_simplify_expression_l4045_404517

theorem simplify_expression (x : ℝ) : (3*x + 15) + (100*x + 15) + (10*x - 5) = 113*x + 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4045_404517


namespace NUMINAMATH_CALUDE_republican_votes_for_candidate_a_l4045_404509

theorem republican_votes_for_candidate_a (total_voters : ℝ) 
  (h1 : total_voters > 0) 
  (democrat_percent : ℝ) 
  (h2 : democrat_percent = 0.60)
  (republican_percent : ℝ) 
  (h3 : republican_percent = 1 - democrat_percent)
  (democrat_votes_for_a_percent : ℝ) 
  (h4 : democrat_votes_for_a_percent = 0.65)
  (total_votes_for_a_percent : ℝ) 
  (h5 : total_votes_for_a_percent = 0.47) : 
  (total_votes_for_a_percent * total_voters - democrat_votes_for_a_percent * democrat_percent * total_voters) / 
  (republican_percent * total_voters) = 0.20 := by
sorry

end NUMINAMATH_CALUDE_republican_votes_for_candidate_a_l4045_404509


namespace NUMINAMATH_CALUDE_age_of_other_man_is_21_l4045_404546

/-- The age of the other replaced man in a group replacement scenario -/
def age_of_other_replaced_man (initial_count : ℕ) (age_increase : ℝ) (age_of_one_replaced : ℕ) (avg_age_new_men : ℝ) : ℝ :=
  let total_age_increase := initial_count * age_increase
  let total_age_new_men := 2 * avg_age_new_men
  total_age_new_men - total_age_increase - age_of_one_replaced

/-- Theorem: The age of the other replaced man is 21 years -/
theorem age_of_other_man_is_21 :
  age_of_other_replaced_man 10 2 23 32 = 21 := by
  sorry

#eval age_of_other_replaced_man 10 2 23 32

end NUMINAMATH_CALUDE_age_of_other_man_is_21_l4045_404546


namespace NUMINAMATH_CALUDE_unique_polynomial_reconstruction_l4045_404561

/-- A polynomial with non-negative integer coefficients -/
def NonNegIntPolynomial (P : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, ∀ k > n, P k = 0

/-- The polynomial is non-constant -/
def NonConstant (P : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, P k ≠ P 0

theorem unique_polynomial_reconstruction
  (P : ℕ → ℕ)
  (h_non_neg : NonNegIntPolynomial P)
  (h_non_const : NonConstant P) :
  ∀ Q : ℕ → ℕ,
    NonNegIntPolynomial Q →
    NonConstant Q →
    P 2 = Q 2 →
    P (P 2) = Q (Q 2) →
    ∀ x, P x = Q x :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_reconstruction_l4045_404561


namespace NUMINAMATH_CALUDE_yellow_raisins_amount_l4045_404539

theorem yellow_raisins_amount (yellow_raisins black_raisins total_raisins : ℝ) 
  (h1 : black_raisins = 0.4)
  (h2 : total_raisins = 0.7)
  (h3 : yellow_raisins + black_raisins = total_raisins) : 
  yellow_raisins = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_raisins_amount_l4045_404539


namespace NUMINAMATH_CALUDE_max_value_of_product_l4045_404573

theorem max_value_of_product (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a + b = 4) :
  (∀ x y : ℝ, x > 1 → y > 1 → x + y = 4 → (x - 1) * (y - 1) ≤ (a - 1) * (b - 1)) →
  (a - 1) * (b - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_product_l4045_404573


namespace NUMINAMATH_CALUDE_same_color_isosceles_count_independent_l4045_404536

/-- Represents a coloring of vertices in a regular polygon -/
structure Coloring (n : ℕ) where
  red_count : ℕ
  blue_count : ℕ
  total_count : red_count + blue_count = 6 * n + 1

/-- Counts the number of isosceles triangles with vertices of the same color -/
def count_same_color_isosceles_triangles (n : ℕ) (coloring : Coloring n) : ℕ :=
  sorry

/-- Theorem stating that the count of same-color isosceles triangles is independent of coloring -/
theorem same_color_isosceles_count_independent (n : ℕ) 
  (coloring1 coloring2 : Coloring n) : 
  count_same_color_isosceles_triangles n coloring1 = 
  count_same_color_isosceles_triangles n coloring2 :=
sorry

end NUMINAMATH_CALUDE_same_color_isosceles_count_independent_l4045_404536


namespace NUMINAMATH_CALUDE_robbery_participants_l4045_404535

theorem robbery_participants (A B V G : Prop) 
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G ∧ ¬V := by sorry

end NUMINAMATH_CALUDE_robbery_participants_l4045_404535


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4045_404554

-- Define the polynomial and the divisor
def f (x : ℝ) : ℝ := x^6 - 2*x^5 - 3*x^4 + 4*x^3 + 5*x^2 - x - 2
def g (x : ℝ) : ℝ := (x-3)*(x^2-1)

-- Define the remainder
def r (x : ℝ) : ℝ := 18*x^2 + x - 17

-- Theorem statement
theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = g x * q x + r x :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4045_404554


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4045_404578

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 - 2*x^2 + 3 = (x^2 - 4*x + 7) * q + (28*x - 46) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4045_404578


namespace NUMINAMATH_CALUDE_range_of_f_l4045_404582

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 4)

-- State the theorem
theorem range_of_f :
  let S := {y : ℝ | ∃ x : ℝ, x ≠ -8 ∧ f x = y}
  S = {y : ℝ | y < -36 ∨ y > -36} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l4045_404582


namespace NUMINAMATH_CALUDE_track_circumference_is_720_l4045_404588

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  circumference : ℝ
  speed_A : ℝ
  speed_B : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : CircularTrack) : Prop :=
  ∃ (t₁ t₂ : ℝ),
    t₁ > 0 ∧ t₂ > t₁ ∧
    track.speed_B * t₁ = 150 ∧
    track.speed_A * t₁ = track.circumference / 2 - 150 ∧
    track.speed_A * t₂ = track.circumference - 90 ∧
    track.speed_B * t₂ = track.circumference / 2 + 90

/-- The theorem stating that the track circumference is 720 yards -/
theorem track_circumference_is_720 (track : CircularTrack) :
  problem_conditions track → track.circumference = 720 := by
  sorry


end NUMINAMATH_CALUDE_track_circumference_is_720_l4045_404588


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l4045_404551

/-- 
Given a line equation (2k-1)x-(k+3)y-(k-11)=0 where k is any real number,
prove that this line always passes through the point (2, 3).
-/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l4045_404551


namespace NUMINAMATH_CALUDE_min_sum_squares_l4045_404570

/-- Given five real numbers satisfying certain conditions, 
    the sum of their squares has a minimum value. -/
theorem min_sum_squares (a₁ a₂ a₃ a₄ a₅ : ℝ) 
    (h1 : a₁*a₂ + a₂*a₃ + a₃*a₄ + a₄*a₅ + a₅*a₁ = 20)
    (h2 : a₁*a₃ + a₂*a₄ + a₃*a₅ + a₄*a₁ + a₅*a₂ = 22) :
    ∃ (m : ℝ), m = a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 ∧ 
    ∀ (b₁ b₂ b₃ b₄ b₅ : ℝ), 
    (b₁*b₂ + b₂*b₃ + b₃*b₄ + b₄*b₅ + b₅*b₁ = 20) →
    (b₁*b₃ + b₂*b₄ + b₃*b₅ + b₄*b₁ + b₅*b₂ = 22) →
    m ≤ b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 ∧
    m = 21 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l4045_404570


namespace NUMINAMATH_CALUDE_lines_coincide_by_rotation_l4045_404596

/-- Given two lines l₁ and l₂ in the plane, prove that they can coincide by rotation -/
theorem lines_coincide_by_rotation (α c : ℝ) :
  ∃ (x₀ y₀ θ : ℝ), 
    (y₀ = x₀ * Real.sin α) ∧  -- Point (x₀, y₀) is on l₁
    (∀ x y : ℝ, 
      y = x * Real.sin α →  -- Original line l₁
      ∃ x' y' : ℝ, 
        x' = (x - x₀) * Real.cos θ - (y - y₀) * Real.sin θ + x₀ ∧
        y' = (x - x₀) * Real.sin θ + (y - y₀) * Real.cos θ + y₀ ∧
        y' = 2 * x' + c)  -- Rotated line coincides with l₂
  := by sorry

end NUMINAMATH_CALUDE_lines_coincide_by_rotation_l4045_404596


namespace NUMINAMATH_CALUDE_train_crossing_time_l4045_404526

/-- Given two trains of equal length, prove the time taken by one train to cross a telegraph post. -/
theorem train_crossing_time (train_length : ℝ) (time_second_train : ℝ) (time_crossing_each_other : ℝ) :
  train_length = 120 →
  time_second_train = 15 →
  time_crossing_each_other = 12 →
  ∃ (time_first_train : ℝ),
    time_first_train = 10 ∧
    train_length / time_first_train + train_length / time_second_train =
      2 * train_length / time_crossing_each_other :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l4045_404526


namespace NUMINAMATH_CALUDE_slope_angle_range_l4045_404515

/-- Given two lines and their intersection in the first quadrant, 
    prove the range of the slope angle of one line -/
theorem slope_angle_range (k : ℝ) : 
  let l1 : ℝ → ℝ := λ x => k * x - Real.sqrt 3
  let l2 : ℝ → ℝ := λ x => (6 - 2 * x) / 3
  let x_intersect := (3 * Real.sqrt 3 + 6) / (2 + 3 * k)
  let y_intersect := (6 * k - 2 * Real.sqrt 3) / (2 + 3 * k)
  (x_intersect > 0 ∧ y_intersect > 0) →
  let θ := Real.arctan k
  θ > π / 6 ∧ θ < π / 2 := by
sorry

end NUMINAMATH_CALUDE_slope_angle_range_l4045_404515


namespace NUMINAMATH_CALUDE_triangles_in_decagon_count_l4045_404580

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- Proof that the number of triangles in a regular decagon is correct -/
theorem triangles_in_decagon_count : 
  (Finset.univ.filter (λ s : Finset (Fin 10) => s.card = 3)).card = trianglesInDecagon := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_decagon_count_l4045_404580


namespace NUMINAMATH_CALUDE_ludek_unique_stamps_l4045_404563

theorem ludek_unique_stamps 
  (karel_mirek : ℕ) 
  (karel_ludek : ℕ) 
  (mirek_ludek : ℕ) 
  (karel_mirek_shared : ℕ) 
  (karel_ludek_shared : ℕ) 
  (mirek_ludek_shared : ℕ) 
  (h1 : karel_mirek = 101) 
  (h2 : karel_ludek = 115) 
  (h3 : mirek_ludek = 110) 
  (h4 : karel_mirek_shared = 5) 
  (h5 : karel_ludek_shared = 12) 
  (h6 : mirek_ludek_shared = 7) : 
  ∃ (ludek_total : ℕ), 
    ludek_total - karel_ludek_shared - mirek_ludek_shared = 43 :=
by sorry

end NUMINAMATH_CALUDE_ludek_unique_stamps_l4045_404563


namespace NUMINAMATH_CALUDE_cube_difference_l4045_404516

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 59) : 
  a^3 - b^3 = 448 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l4045_404516


namespace NUMINAMATH_CALUDE_power_of_power_l4045_404556

theorem power_of_power (a : ℝ) : (a^4)^2 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l4045_404556


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l4045_404579

/-- The set of real numbers x that satisfy (x+2)/(x-4) ≥ 3 is exactly the interval (4, 7]. -/
theorem solution_set_equivalence (x : ℝ) : (x + 2) / (x - 4) ≥ 3 ↔ x ∈ Set.Ioo 4 7 ∪ {7} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l4045_404579


namespace NUMINAMATH_CALUDE_shaanxi_temp_difference_l4045_404567

/-- The temperature difference between two regions -/
def temperature_difference (temp1 : ℝ) (temp2 : ℝ) : ℝ :=
  temp1 - temp2

/-- The highest temperature in Shaanxi South -/
def shaanxi_south_temp : ℝ := 6

/-- The highest temperature in Shaanxi North -/
def shaanxi_north_temp : ℝ := -3

/-- Theorem: The temperature difference between Shaanxi South and Shaanxi North is 9°C -/
theorem shaanxi_temp_difference :
  temperature_difference shaanxi_south_temp shaanxi_north_temp = 9 := by
  sorry

end NUMINAMATH_CALUDE_shaanxi_temp_difference_l4045_404567


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l4045_404532

/-- Given two quadratic equations where the roots of one are three times the roots of the other, 
    prove that the ratio of certain coefficients is 27. -/
theorem quadratic_root_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -c ∧ s₁ * s₂ = a) ∧
               (3 * s₁ + 3 * s₂ = -a ∧ 9 * s₁ * s₂ = b)) →
  b / c = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l4045_404532


namespace NUMINAMATH_CALUDE_chlorous_acid_weight_l4045_404531

/-- The atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- The atomic weight of Chlorine in g/mol -/
def Cl_weight : ℝ := 35.45

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The number of moles of Chlorous acid -/
def moles : ℝ := 6

/-- The molecular weight of Chlorous acid (HClO2) in g/mol -/
def HClO2_weight : ℝ := H_weight + Cl_weight + 2 * O_weight

/-- Theorem: The molecular weight of 6 moles of Chlorous acid (HClO2) is 410.76 grams -/
theorem chlorous_acid_weight : moles * HClO2_weight = 410.76 := by
  sorry

end NUMINAMATH_CALUDE_chlorous_acid_weight_l4045_404531


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4045_404566

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + 2*x*y + 2*y^2 ≥ 0 ∧ (x^2 + 2*x*y + 2*y^2 = 0 ↔ x = 0 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4045_404566


namespace NUMINAMATH_CALUDE_sons_present_age_l4045_404524

theorem sons_present_age (son_age father_age : ℕ) : 
  father_age = son_age + 45 →
  father_age + 10 = 4 * (son_age + 10) →
  son_age + 15 = 2 * son_age →
  son_age = 15 := by
sorry

end NUMINAMATH_CALUDE_sons_present_age_l4045_404524


namespace NUMINAMATH_CALUDE_binary_111_equals_7_l4045_404513

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 111 -/
def binary_111 : List Bool := [true, true, true]

theorem binary_111_equals_7 : binary_to_decimal binary_111 = 7 := by
  sorry

end NUMINAMATH_CALUDE_binary_111_equals_7_l4045_404513


namespace NUMINAMATH_CALUDE_total_stripes_is_22_l4045_404519

/-- The number of stripes on each of Olga's tennis shoes -/
def olga_stripes : ℕ := 3

/-- The number of stripes on each of Rick's tennis shoes -/
def rick_stripes : ℕ := olga_stripes - 1

/-- The number of stripes on each of Hortense's tennis shoes -/
def hortense_stripes : ℕ := olga_stripes * 2

/-- The number of shoes each person has -/
def shoes_per_person : ℕ := 2

/-- The total number of stripes on all their shoes combined -/
def total_stripes : ℕ := shoes_per_person * (olga_stripes + rick_stripes + hortense_stripes)

theorem total_stripes_is_22 : total_stripes = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_stripes_is_22_l4045_404519


namespace NUMINAMATH_CALUDE_chord_length_l4045_404545

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C₃ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 14/5 = 0

-- Define the common chord of C₁ and C₂
def common_chord (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- Theorem statement
theorem chord_length :
  ∃ (chord_length : ℝ),
    chord_length = 4 ∧
    ∀ (x y : ℝ),
      common_chord x y →
      C₃ x y →
      (∃ (x' y' : ℝ),
        common_chord x' y' ∧
        C₃ x' y' ∧
        (x - x')^2 + (y - y')^2 = chord_length^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l4045_404545


namespace NUMINAMATH_CALUDE_max_value_of_product_sum_l4045_404508

theorem max_value_of_product_sum (a b c : ℝ) (h : a + 3 * b + c = 5) :
  (∀ x y z : ℝ, x + 3 * y + z = 5 → a * b + a * c + b * c ≥ x * y + x * z + y * z) ∧
  a * b + a * c + b * c = 25 / 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_product_sum_l4045_404508


namespace NUMINAMATH_CALUDE_volume_union_tetrahedrons_is_half_l4045_404593

/-- A regular tetrahedron formed from vertices of a unit cube -/
structure CubeTetrahedron where
  vertices : Finset (Fin 8)
  is_regular : Bool
  from_cube : Bool

/-- The volume of the union of two regular tetrahedrons formed from the vertices of a unit cube -/
def volume_union_tetrahedrons (t1 t2 : CubeTetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the union of two regular tetrahedrons
    formed from the vertices of a unit cube is 1/2 -/
theorem volume_union_tetrahedrons_is_half
  (t1 t2 : CubeTetrahedron)
  (h1 : t1.is_regular)
  (h2 : t2.is_regular)
  (h3 : t1.from_cube)
  (h4 : t2.from_cube)
  (h5 : t1.vertices ≠ t2.vertices)
  : volume_union_tetrahedrons t1 t2 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_volume_union_tetrahedrons_is_half_l4045_404593


namespace NUMINAMATH_CALUDE_square_of_one_minus_i_l4045_404557

theorem square_of_one_minus_i (i : ℂ) : i^2 = -1 → (1 - i)^2 = -2*i := by
  sorry

end NUMINAMATH_CALUDE_square_of_one_minus_i_l4045_404557


namespace NUMINAMATH_CALUDE_base_2002_not_prime_l4045_404537

/-- For a positive integer n ≥ 2, 2002_n is defined as 2n^3 + 2 in base 10 -/
def base_2002 (n : ℕ) : ℕ := 2 * n^3 + 2

/-- Theorem: For all positive integers n ≥ 2, 2002_n is not a prime number -/
theorem base_2002_not_prime {n : ℕ} (h : n ≥ 2) : ¬ Nat.Prime (base_2002 n) := by
  sorry

end NUMINAMATH_CALUDE_base_2002_not_prime_l4045_404537


namespace NUMINAMATH_CALUDE_smallest_b_in_geometric_sequence_l4045_404506

/-- 
Given a geometric sequence of positive terms a, b, c with product 216,
this theorem states that the smallest possible value of b is 6.
-/
theorem smallest_b_in_geometric_sequence (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- all terms are positive
  (∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r) →  -- geometric sequence
  a * b * c = 216 →  -- product is 216
  (∀ b' : ℝ, b' > 0 → 
    (∃ a' c' : ℝ, a' > 0 ∧ c' > 0 ∧ 
      (∃ r : ℝ, r > 0 ∧ b' = a' * r ∧ c' = b' * r) ∧ 
      a' * b' * c' = 216) → 
    b' ≥ 6) →  -- for any valid b', b' is at least 6
  b = 6  -- therefore, the smallest possible b is 6
:= by sorry

end NUMINAMATH_CALUDE_smallest_b_in_geometric_sequence_l4045_404506


namespace NUMINAMATH_CALUDE_boys_running_speed_l4045_404599

/-- Given a square field with side length 60 meters and a boy who runs around it in 96 seconds,
    prove that the boy's speed is 9 km/hr. -/
theorem boys_running_speed (side_length : ℝ) (time : ℝ) (speed : ℝ) : 
  side_length = 60 →
  time = 96 →
  speed = (4 * side_length) / time * 3.6 →
  speed = 9 := by sorry

end NUMINAMATH_CALUDE_boys_running_speed_l4045_404599


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1023_l4045_404583

theorem largest_gcd_of_sum_1023 :
  ∃ (a b : ℕ+), a + b = 1023 ∧
  ∀ (c d : ℕ+), c + d = 1023 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 341 :=
by sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1023_l4045_404583


namespace NUMINAMATH_CALUDE_min_sum_of_dimensions_l4045_404529

theorem min_sum_of_dimensions (l w h : ℕ+) : 
  l * w * h = 2541 → l + w + h ≥ 191 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_dimensions_l4045_404529


namespace NUMINAMATH_CALUDE_perpendicular_tangents_trajectory_l4045_404510

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a tangent line from P to the unit circle
def is_tangent (P : Point) (A : Point) : Prop :=
  unit_circle A.x A.y ∧ 
  (P.x - A.x) * A.x + (P.y - A.y) * A.y = 0

-- State the theorem
theorem perpendicular_tangents_trajectory :
  ∀ P : Point,
  (∃ A B : Point,
    is_tangent P A ∧
    is_tangent P B ∧
    (P.x - A.x) * (P.x - B.x) + (P.y - A.y) * (P.y - B.y) = 0) →
  P.x^2 + P.y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_trajectory_l4045_404510


namespace NUMINAMATH_CALUDE_lizzy_shipment_cost_l4045_404538

/-- Calculates the total shipment cost for Lizzy's fish shipment --/
def total_shipment_cost (total_weight type_a_capacity type_b_capacity : ℕ)
  (type_a_cost type_b_cost surcharge flat_fee : ℚ)
  (num_type_a : ℕ) : ℚ :=
  let type_a_total_weight := num_type_a * type_a_capacity
  let type_b_total_weight := total_weight - type_a_total_weight
  let num_type_b := (type_b_total_weight + type_b_capacity - 1) / type_b_capacity
  let type_a_total_cost := num_type_a * (type_a_cost + surcharge)
  let type_b_total_cost := num_type_b * (type_b_cost + surcharge)
  type_a_total_cost + type_b_total_cost + flat_fee

theorem lizzy_shipment_cost :
  total_shipment_cost 540 30 50 (3/2) (5/2) (1/2) 10 6 = 46 :=
by sorry

end NUMINAMATH_CALUDE_lizzy_shipment_cost_l4045_404538


namespace NUMINAMATH_CALUDE_league_games_count_l4045_404549

/-- Calculates the number of games in a league season -/
def number_of_games (n : ℕ) (k : ℕ) : ℕ :=
  (n * (n - 1) / 2) * k

/-- Theorem: In a league with 50 teams, where each team plays every other team 4 times,
    the total number of games played in the season is 4900. -/
theorem league_games_count : number_of_games 50 4 = 4900 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l4045_404549


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4045_404540

theorem inequality_equivalence (x : ℝ) : 
  (-1/3 : ℝ) ≤ (5-x)/2 ∧ (5-x)/2 < (1/3 : ℝ) ↔ (13/3 : ℝ) < x ∧ x ≤ (17/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4045_404540


namespace NUMINAMATH_CALUDE_fourth_composition_is_even_l4045_404552

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem fourth_composition_is_even
  (f : ℝ → ℝ) (h : OddFunction f) :
  ∀ x, f (f (f (f x))) = f (f (f (f (-x)))) :=
sorry

end NUMINAMATH_CALUDE_fourth_composition_is_even_l4045_404552
