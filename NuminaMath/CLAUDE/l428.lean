import Mathlib

namespace NUMINAMATH_CALUDE_determinant_equal_polynomial_l428_42838

variable (x : ℝ)

def matrix : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
  match i, j with
  | 0, 0 => 2*x + 3
  | 0, 1 => x
  | 0, 2 => x
  | 1, 0 => 2*x
  | 1, 1 => 2*x + 3
  | 1, 2 => x
  | 2, 0 => 2*x
  | 2, 1 => x
  | 2, 2 => 2*x + 3

theorem determinant_equal_polynomial (x : ℝ) :
  Matrix.det (matrix x) = 2*x^3 + 27*x^2 + 27*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equal_polynomial_l428_42838


namespace NUMINAMATH_CALUDE_starting_lineup_theorem_l428_42892

/-- The number of ways to choose a starting lineup from a basketball team. -/
def starting_lineup_choices (team_size : ℕ) (lineup_size : ℕ) (point_guard_count : ℕ) : ℕ :=
  team_size * Nat.choose (team_size - 1) (lineup_size - 1)

/-- Theorem: The number of ways to choose a starting lineup of 5 players
    from a team of 12, where one player must be the point guard and
    the other four positions are interchangeable, is equal to 3960. -/
theorem starting_lineup_theorem :
  starting_lineup_choices 12 5 1 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_theorem_l428_42892


namespace NUMINAMATH_CALUDE_karen_is_ten_l428_42809

def sisters : Finset ℕ := {2, 4, 6, 8, 10, 12, 14}

def park_pair (a b : ℕ) : Prop :=
  a ∈ sisters ∧ b ∈ sisters ∧ a ≠ b ∧ a + b = 20

def pool_pair (a b : ℕ) : Prop :=
  a ∈ sisters ∧ b ∈ sisters ∧ a ≠ b ∧ 3 < a ∧ a < 9 ∧ 3 < b ∧ b < 9

def karen_age (k : ℕ) : Prop :=
  k ∈ sisters ∧ k ≠ 4 ∧
  ∃ (p1 p2 s1 s2 : ℕ),
    park_pair p1 p2 ∧
    pool_pair s1 s2 ∧
    p1 ≠ s1 ∧ p1 ≠ s2 ∧ p2 ≠ s1 ∧ p2 ≠ s2 ∧
    k ≠ p1 ∧ k ≠ p2 ∧ k ≠ s1 ∧ k ≠ s2

theorem karen_is_ten : ∃! k, karen_age k ∧ k = 10 := by
  sorry

end NUMINAMATH_CALUDE_karen_is_ten_l428_42809


namespace NUMINAMATH_CALUDE_min_xy_value_l428_42896

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (3 / (2 + x)) + (3 / (2 + y)) = 1) : 
  x * y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (3 / (2 + x)) + (3 / (2 + y)) = 1 ∧ x * y = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l428_42896


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l428_42861

theorem constant_term_binomial_expansion :
  ∀ x : ℝ, ∃ t : ℕ → ℝ,
    (∀ r, t r = (-1)^r * (Nat.choose 6 r) * (2^((12 - 3*r) * x))) ∧
    (∃ k, t k = 15 ∧ ∀ r ≠ k, ∃ n : ℤ, t r = 2^(n*x)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l428_42861


namespace NUMINAMATH_CALUDE_teachers_survey_l428_42824

theorem teachers_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_high_bp : high_bp = 90)
  (h_heart_trouble : heart_trouble = 60)
  (h_both : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_teachers_survey_l428_42824


namespace NUMINAMATH_CALUDE_two_times_greater_l428_42817

theorem two_times_greater (a b : ℚ) (h : a > b) : 2 * a > 2 * b := by
  sorry

end NUMINAMATH_CALUDE_two_times_greater_l428_42817


namespace NUMINAMATH_CALUDE_line_equation_proof_l428_42880

/-- Given a line passing through the point (√3, -3) with an inclination angle of 30°,
    prove that its equation is y = (√3/3)x - 4 -/
theorem line_equation_proof (x y : ℝ) :
  let point : ℝ × ℝ := (Real.sqrt 3, -3)
  let angle : ℝ := 30 * π / 180  -- Convert 30° to radians
  let slope : ℝ := Real.tan angle
  slope * (x - point.1) = y - point.2 →
  y = (Real.sqrt 3 / 3) * x - 4 := by
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l428_42880


namespace NUMINAMATH_CALUDE_angle_equality_l428_42822

theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 2 * Real.sin (20 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 25 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l428_42822


namespace NUMINAMATH_CALUDE_bouquet_cost_60_l428_42846

/-- The cost of a bouquet of tulips at Tony's Tulip Tower -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_rate := 36 / 18
  let threshold := 40
  let extra_rate := base_rate * (3/2)
  if n ≤ threshold then
    n * base_rate
  else
    threshold * base_rate + (n - threshold) * extra_rate

/-- The theorem stating the cost of a bouquet of 60 tulips -/
theorem bouquet_cost_60 : bouquet_cost 60 = 140 := by
  sorry

#eval bouquet_cost 60

end NUMINAMATH_CALUDE_bouquet_cost_60_l428_42846


namespace NUMINAMATH_CALUDE_product_real_condition_l428_42893

theorem product_real_condition (a b c d : ℝ) :
  (∃ (x : ℝ), (a + b * Complex.I) * (c + d * Complex.I) = x) ↔ a * d + b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_real_condition_l428_42893


namespace NUMINAMATH_CALUDE_sum_of_tenth_powers_l428_42894

/-- Given a sequence of sums of powers of a and b, prove that a^10 + b^10 = 123 -/
theorem sum_of_tenth_powers (a b : ℝ) 
  (sum1 : a + b = 1)
  (sum2 : a^2 + b^2 = 3)
  (sum3 : a^3 + b^3 = 4)
  (sum4 : a^4 + b^4 = 7)
  (sum5 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tenth_powers_l428_42894


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l428_42875

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 2} = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l428_42875


namespace NUMINAMATH_CALUDE_not_right_triangle_l428_42819

theorem not_right_triangle : ∃ (a b c : ℝ), 
  (a = Real.sqrt 3 ∧ b = 4 ∧ c = 5) ∧ 
  (a^2 + b^2 ≠ c^2) ∧
  (∀ (x y z : ℝ), 
    ((x = 1 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 3) ∨
     (x = 7 ∧ y = 24 ∧ z = 25) ∨
     (x = 5 ∧ y = 12 ∧ z = 13)) →
    (x^2 + y^2 = z^2)) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l428_42819


namespace NUMINAMATH_CALUDE_square_difference_equals_six_l428_42849

theorem square_difference_equals_six (a b : ℝ) 
  (sum_eq : a + b = 2) 
  (diff_eq : a - b = 3) : 
  a^2 - b^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_difference_equals_six_l428_42849


namespace NUMINAMATH_CALUDE_average_of_multiples_l428_42841

theorem average_of_multiples (x : ℝ) : 
  let terms := [0, 2*x, 4*x, 8*x, 16*x]
  let multiplied_terms := List.map (· * 3) terms
  List.sum multiplied_terms / 5 = 18 * x := by
sorry

end NUMINAMATH_CALUDE_average_of_multiples_l428_42841


namespace NUMINAMATH_CALUDE_car_speed_second_half_l428_42818

/-- Calculates the speed of a car during the second half of a journey given the total distance,
    speed for the first half, and average speed for the entire journey. -/
theorem car_speed_second_half
  (total_distance : ℝ)
  (first_half_distance : ℝ)
  (first_half_speed : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 320)
  (h2 : first_half_distance = 160)
  (h3 : first_half_speed = 90)
  (h4 : average_speed = 84.70588235294117)
  (h5 : first_half_distance * 2 = total_distance) :
  let second_half_speed := (total_distance / average_speed - first_half_distance / first_half_speed)⁻¹ * first_half_distance
  second_half_speed = 80 := by
sorry


end NUMINAMATH_CALUDE_car_speed_second_half_l428_42818


namespace NUMINAMATH_CALUDE_percentage_problem_l428_42879

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  (0.4 * x = 160) → (p * x = 120) → p = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l428_42879


namespace NUMINAMATH_CALUDE_cave_depth_remaining_l428_42865

/-- Given a cave of depth 974 feet and a current position of 588 feet,
    the remaining distance to the end of the cave is 386 feet. -/
theorem cave_depth_remaining (cave_depth : ℕ) (current_position : ℕ) :
  cave_depth = 974 → current_position = 588 → cave_depth - current_position = 386 := by
  sorry

end NUMINAMATH_CALUDE_cave_depth_remaining_l428_42865


namespace NUMINAMATH_CALUDE_cubic_equation_root_l428_42826

theorem cubic_equation_root (a b : ℚ) : 
  (3 + Real.sqrt 5 : ℝ) ^ 3 + a * (3 + Real.sqrt 5 : ℝ) ^ 2 + b * (3 + Real.sqrt 5 : ℝ) - 20 = 0 → 
  b = -26 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l428_42826


namespace NUMINAMATH_CALUDE_solve_problem_l428_42823

def problem (x : ℝ) : Prop :=
  let k_speed := x
  let m_speed := x - 0.5
  let k_time := 40 / k_speed
  let m_time := 40 / m_speed
  (m_time - k_time = 1/3) ∧ (k_time = 5)

theorem solve_problem :
  ∃ x : ℝ, problem x :=
sorry

end NUMINAMATH_CALUDE_solve_problem_l428_42823


namespace NUMINAMATH_CALUDE_hexagon_problem_l428_42885

/-- Regular hexagon with side length 3 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ)
  (regular : side_length = 3)

/-- L is the intersection point of diagonals CE and DF -/
def L (h : RegularHexagon) : ℝ × ℝ := sorry

/-- K is defined such that LK = 3AB - AC -/
def K (h : RegularHexagon) : ℝ × ℝ := sorry

/-- Determine if a point is outside the hexagon -/
def is_outside (h : RegularHexagon) (p : ℝ × ℝ) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hexagon_problem (h : RegularHexagon) :
  is_outside h (K h) ∧ distance (K h) h.C = 3 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_hexagon_problem_l428_42885


namespace NUMINAMATH_CALUDE_complex_equation_solution_l428_42832

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def satisfies_equation (z : ℂ) : Prop := (2 * i) / z = 1 - i

-- Theorem statement
theorem complex_equation_solution :
  ∀ z : ℂ, satisfies_equation z → z = -1 + i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l428_42832


namespace NUMINAMATH_CALUDE_protest_jail_time_ratio_l428_42800

theorem protest_jail_time_ratio : 
  let days_of_protest : ℕ := 30
  let num_cities : ℕ := 21
  let arrests_per_day : ℕ := 10
  let days_before_trial : ℕ := 4
  let sentence_weeks : ℕ := 2
  let total_jail_weeks : ℕ := 9900
  let total_arrests : ℕ := days_of_protest * num_cities * arrests_per_day
  let weeks_before_trial : ℕ := total_arrests * days_before_trial / 7
  let weeks_after_trial : ℕ := total_jail_weeks - weeks_before_trial
  let total_possible_weeks : ℕ := total_arrests * sentence_weeks
  (weeks_after_trial : ℚ) / total_possible_weeks = 1 / 2 := by
  sorry

#check protest_jail_time_ratio

end NUMINAMATH_CALUDE_protest_jail_time_ratio_l428_42800


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l428_42887

theorem smallest_solution_of_equation (x : ℝ) :
  (3 * x^2 + 33 * x - 90 = x * (x + 15)) →
  x ≥ -15 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l428_42887


namespace NUMINAMATH_CALUDE_tan_alpha_value_l428_42889

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan (2*α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l428_42889


namespace NUMINAMATH_CALUDE_max_value_theorem_l428_42871

theorem max_value_theorem (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 4 + 9 * y * z ≤ Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l428_42871


namespace NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l428_42882

/-- The sum of the infinite series ∑(1 / (n(n+3))) for n from 1 to infinity is equal to 7/9. -/
theorem sum_reciprocal_n_n_plus_three : 
  ∑' n : ℕ+, (1 : ℝ) / (n * (n + 3)) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l428_42882


namespace NUMINAMATH_CALUDE_max_cross_section_area_correct_l428_42807

noncomputable def max_cross_section_area (k : ℝ) (α : ℝ) : ℝ :=
  if Real.tan α < 2 then
    (1/2) * k^2 * (1 + 3 * Real.cos α ^ 2)
  else
    2 * k^2 * Real.sin (2 * α)

theorem max_cross_section_area_correct (k : ℝ) (α : ℝ) (h1 : k > 0) (h2 : 0 < α ∧ α < π/2) :
  ∀ A : ℝ, A ≤ max_cross_section_area k α := by
  sorry

end NUMINAMATH_CALUDE_max_cross_section_area_correct_l428_42807


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l428_42840

theorem quadratic_solution_range (a b c : ℝ) :
  (a * 0^2 + b * 0 + c = -15) →
  (a * 0.5^2 + b * 0.5 + c = -8.75) →
  (a * 1^2 + b * 1 + c = -2) →
  (a * 1.5^2 + b * 1.5 + c = 5.25) →
  (a * 2^2 + b * 2 + c = 13) →
  ∃ x : ℝ, (1 < x ∧ x < 1.5) ∧ (a * x^2 + b * x + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l428_42840


namespace NUMINAMATH_CALUDE_total_pet_time_is_108_minutes_l428_42876

-- Define the time spent on each activity
def dog_walk_play_time : ℚ := 1/2
def dog_feed_time : ℚ := 1/5
def cat_play_time : ℚ := 1/4
def cat_feed_time : ℚ := 1/10

-- Define the number of times each activity is performed daily
def dog_walk_play_frequency : ℕ := 2
def dog_feed_frequency : ℕ := 1
def cat_play_frequency : ℕ := 2
def cat_feed_frequency : ℕ := 1

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem total_pet_time_is_108_minutes :
  (dog_walk_play_time * dog_walk_play_frequency +
   dog_feed_time * dog_feed_frequency +
   cat_play_time * cat_play_frequency +
   cat_feed_time * cat_feed_frequency) * minutes_per_hour = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_pet_time_is_108_minutes_l428_42876


namespace NUMINAMATH_CALUDE_shell_distribution_l428_42812

theorem shell_distribution (jillian savannah clayton friends_share : ℕ) : 
  jillian = 29 →
  savannah = 17 →
  clayton = 8 →
  friends_share = 27 →
  (jillian + savannah + clayton) / friends_share = 2 :=
by sorry

end NUMINAMATH_CALUDE_shell_distribution_l428_42812


namespace NUMINAMATH_CALUDE_sin_seventeen_pi_fourths_l428_42867

theorem sin_seventeen_pi_fourths : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seventeen_pi_fourths_l428_42867


namespace NUMINAMATH_CALUDE_parabola_ratio_l428_42815

/-- A parabola passing through points (-1, 1) and (3, 1) has a/b = -2 --/
theorem parabola_ratio (a b c : ℝ) : 
  (a * (-1)^2 + b * (-1) + c = 1) → 
  (a * 3^2 + b * 3 + c = 1) → 
  a / b = -2 := by
sorry

end NUMINAMATH_CALUDE_parabola_ratio_l428_42815


namespace NUMINAMATH_CALUDE_intersects_iff_m_ge_neg_one_l428_42827

/-- A quadratic function f(x) = x^2 + 2x - m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - m

/-- The graph of f intersects the x-axis -/
def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, f m x = 0

/-- Theorem: The graph of f(x) = x^2 + 2x - m intersects the x-axis
    if and only if m ≥ -1 -/
theorem intersects_iff_m_ge_neg_one (m : ℝ) :
  intersects_x_axis m ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersects_iff_m_ge_neg_one_l428_42827


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l428_42897

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l428_42897


namespace NUMINAMATH_CALUDE_fraction_addition_l428_42898

theorem fraction_addition : (3 : ℚ) / 8 + (9 : ℚ) / 12 = (9 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l428_42898


namespace NUMINAMATH_CALUDE_rectangle_area_l428_42834

theorem rectangle_area (a c : ℝ) (ha : a = 15) (hc : c = 17) : 
  ∃ b : ℝ, a * b = 120 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l428_42834


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l428_42859

/-- Represents a point in 1D space -/
structure Point1D where
  x : ℝ

/-- Distance between two points in 1D -/
def distance (p q : Point1D) : ℝ := |p.x - q.x|

/-- Sum of squared distances from a point to multiple points -/
def sumSquaredDistances (p : Point1D) (points : List Point1D) : ℝ :=
  points.foldl (fun sum q => sum + (distance p q)^2) 0

/-- The problem statement -/
theorem min_sum_squared_distances :
  ∃ (a b c d e : Point1D),
    distance a b = 2 ∧
    distance b c = 2 ∧
    distance c d = 3 ∧
    distance d e = 7 ∧
    (∀ p : Point1D, sumSquaredDistances p [a, b, c, d, e] ≥ 133.2) ∧
    (∃ q : Point1D, sumSquaredDistances q [a, b, c, d, e] = 133.2) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l428_42859


namespace NUMINAMATH_CALUDE_curves_intersection_l428_42825

/-- The first curve -/
def curve1 (x : ℝ) : ℝ := 2 * x^3 + x^2 - 5 * x + 2

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 4

/-- Intersection points of the two curves -/
def intersection_points : Set (ℝ × ℝ) := {(-1, -7), (3, 41)}

theorem curves_intersection :
  ∀ p : ℝ × ℝ, p ∈ intersection_points ↔ 
    (curve1 p.1 = curve2 p.1 ∧ p.2 = curve1 p.1) ∧
    ∀ x : ℝ, curve1 x = curve2 x → x = p.1 ∨ x = (if p.1 = -1 then 3 else -1) := by
  sorry

end NUMINAMATH_CALUDE_curves_intersection_l428_42825


namespace NUMINAMATH_CALUDE_reasonable_reasoning_types_l428_42843

/-- Represents different types of reasoning --/
inductive ReasoningType
  | Analogy
  | Inductive
  | Deductive

/-- Determines if a reasoning type is considered reasonable --/
def is_reasonable (r : ReasoningType) : Prop :=
  match r with
  | ReasoningType.Analogy => true
  | ReasoningType.Inductive => true
  | ReasoningType.Deductive => false

/-- Theorem stating which reasoning types are reasonable --/
theorem reasonable_reasoning_types :
  (is_reasonable ReasoningType.Analogy) ∧
  (is_reasonable ReasoningType.Inductive) ∧
  ¬(is_reasonable ReasoningType.Deductive) :=
by sorry


end NUMINAMATH_CALUDE_reasonable_reasoning_types_l428_42843


namespace NUMINAMATH_CALUDE_nut_distribution_l428_42850

def distribute_nuts (total : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ := sorry

theorem nut_distribution (total : ℕ) :
  let (tamas, erzsi, bela, juliska, remaining) := distribute_nuts total
  (tamas + bela) - (erzsi + juliska) = 100 →
  total = 1021 ∧ remaining = 321 := by sorry

end NUMINAMATH_CALUDE_nut_distribution_l428_42850


namespace NUMINAMATH_CALUDE_abs_opposite_equal_l428_42853

theorem abs_opposite_equal (x : ℝ) : |x| = |-x| := by sorry

end NUMINAMATH_CALUDE_abs_opposite_equal_l428_42853


namespace NUMINAMATH_CALUDE_four_digit_number_satisfies_condition_l428_42808

/-- Represents a four-digit number -/
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- Splits a four-digit number into two two-digit numbers -/
def SplitNumber (n : ℕ) : ℕ × ℕ :=
  (n / 100, n % 100)

/-- Checks if a number satisfies the given condition -/
def SatisfiesCondition (n : ℕ) : Prop :=
  let (a, b) := SplitNumber n
  (10 * a + b / 10) * (b % 10 + 10 * (b / 10)) + 10 * a = n

theorem four_digit_number_satisfies_condition :
  FourDigitNumber 1995 ∧
  (SplitNumber 1995).2 % 10 = 5 ∧
  SatisfiesCondition 1995 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_satisfies_condition_l428_42808


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l428_42881

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents the total distance run in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ

def yardsPerMile : ℕ := 1760

def marathonLength : Marathon := { miles := 25, yards := 500 }

def numberOfMarathons : ℕ := 12

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yardsPerMile ∧
    TotalDistance.yards (
      { miles := m
      , yards := y
      } : TotalDistance
    ) = 720 ∧
    numberOfMarathons * (marathonLength.miles * yardsPerMile + marathonLength.yards) =
      m * yardsPerMile + y :=
by sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l428_42881


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l428_42836

-- Define an arithmetic sequence of integers
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define an increasing sequence
def is_increasing_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

-- Theorem statement
theorem arithmetic_sequence_product (a : ℕ → ℤ) :
  is_arithmetic_sequence a →
  is_increasing_sequence a →
  a 4 * a 5 = 45 →
  a 3 * a 6 = 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l428_42836


namespace NUMINAMATH_CALUDE_age_problem_l428_42857

theorem age_problem (p q : ℕ) 
  (h1 : p - 6 = (q - 6) / 2)  -- 6 years ago, p was half of q in age
  (h2 : p * 4 = q * 3)        -- The ratio of their present ages is 3:4
  : p + q = 21 := by           -- The total of their present ages is 21
sorry

end NUMINAMATH_CALUDE_age_problem_l428_42857


namespace NUMINAMATH_CALUDE_equation_unique_solution_l428_42845

theorem equation_unique_solution :
  ∃! x : ℝ, Real.sqrt (3 + Real.sqrt (4 + Real.sqrt x)) = (3 + Real.sqrt x) ^ (1/3) ∧ x = 576 := by
  sorry

end NUMINAMATH_CALUDE_equation_unique_solution_l428_42845


namespace NUMINAMATH_CALUDE_factorization_equality_l428_42851

theorem factorization_equality (y : ℝ) : 5*y*(y+2) + 8*(y+2) + 15 = (5*y+8)*(y+2) + 15 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l428_42851


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l428_42870

theorem decimal_sum_to_fraction :
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008 : ℚ) = 22839 / 50000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l428_42870


namespace NUMINAMATH_CALUDE_steps_to_distance_l428_42805

/-- Given that 625 steps correspond to 500 meters, prove that 10,000 steps at the same rate will result in a distance of 8 km. -/
theorem steps_to_distance (steps_short : ℕ) (distance_short : ℝ) (steps_long : ℕ) :
  steps_short = 625 →
  distance_short = 500 →
  steps_long = 10000 →
  (distance_short / steps_short) * steps_long = 8000 :=
by sorry

end NUMINAMATH_CALUDE_steps_to_distance_l428_42805


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l428_42866

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounces : ℕ) : ℝ :=
  3 * initialHeight - 2^(2 - bounces) * initialHeight

/-- Theorem: A ball dropped from 128 meters, bouncing to half its previous height each time,
    travels 383 meters after 9 bounces -/
theorem ball_bounce_distance :
  totalDistance 128 9 = 383 := by
  sorry

#eval totalDistance 128 9

end NUMINAMATH_CALUDE_ball_bounce_distance_l428_42866


namespace NUMINAMATH_CALUDE_novel_series_arrangement_l428_42839

def number_of_series : ℕ := 3
def volumes_per_series : ℕ := 4

def arrangement_count : ℕ := 34650

theorem novel_series_arrangement :
  (Nat.factorial (number_series * volumes_per_series)) / 
  (Nat.factorial volumes_per_series)^number_of_series = arrangement_count := by
  sorry

end NUMINAMATH_CALUDE_novel_series_arrangement_l428_42839


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l428_42862

/-- The imaginary part of the complex number z = (1-i)/(2i) is equal to -1/2 -/
theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - i) / (2 * i)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l428_42862


namespace NUMINAMATH_CALUDE_kennel_cats_dogs_difference_l428_42890

theorem kennel_cats_dogs_difference (num_dogs : ℕ) (num_cats : ℕ) : 
  num_dogs = 32 →
  num_cats * 4 = num_dogs * 3 →
  num_dogs - num_cats = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_kennel_cats_dogs_difference_l428_42890


namespace NUMINAMATH_CALUDE_money_ratio_problem_l428_42828

theorem money_ratio_problem (ram_money gopal_money krishan_money : ℕ) :
  ram_money = 588 →
  krishan_money = 3468 →
  gopal_money * 17 = krishan_money * 7 →
  ∃ (a b : ℕ), a * gopal_money = b * ram_money ∧ a = 3 ∧ b = 7 :=
by sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l428_42828


namespace NUMINAMATH_CALUDE_group_element_identity_l428_42848

theorem group_element_identity (G : Type) [Group G] (a b : G) 
  (h1 : a * b^2 = b^3 * a) (h2 : b * a^2 = a^3 * b) : a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_group_element_identity_l428_42848


namespace NUMINAMATH_CALUDE_prime_divides_abc_l428_42895

theorem prime_divides_abc (p a b c : ℤ) (hp : Prime p)
  (h1 : (6 : ℤ) ∣ p + 1)
  (h2 : p ∣ a + b + c)
  (h3 : p ∣ a^4 + b^4 + c^4) :
  p ∣ a ∧ p ∣ b ∧ p ∣ c := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_abc_l428_42895


namespace NUMINAMATH_CALUDE_linear_function_through_zero_one_l428_42860

/-- A linear function is a function of the form f(x) = kx + b where k and b are real numbers. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ x, f x = k * x + b

/-- Theorem: There exists a linear function that passes through the point (0,1). -/
theorem linear_function_through_zero_one : ∃ f : ℝ → ℝ, LinearFunction f ∧ f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_zero_one_l428_42860


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l428_42891

def M : Set ℤ := {0, 1}
def N : Set ℤ := {x | x^2 + x = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l428_42891


namespace NUMINAMATH_CALUDE_quadratic_sequence_exists_l428_42801

theorem quadratic_sequence_exists (b c : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ 
  ∀ i : ℕ, i ≥ 1 → i ≤ n → |a i - a (i-1)| = i^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_exists_l428_42801


namespace NUMINAMATH_CALUDE_inverse_sum_equals_target_l428_42877

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then x + 3 else x^2 - 4*x + 5

noncomputable def g_inverse (y : ℝ) : ℝ :=
  if y ≤ 5 then y - 3 else 2 + Real.sqrt (y - 1)

theorem inverse_sum_equals_target : g_inverse 1 + g_inverse 6 + g_inverse 11 = 2 + Real.sqrt 5 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_target_l428_42877


namespace NUMINAMATH_CALUDE_fraction_value_l428_42844

theorem fraction_value : 
  (10 + (-9) + 8 + (-7) + 6 + (-5) + 4 + (-3) + 2 + (-1)) / 
  (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l428_42844


namespace NUMINAMATH_CALUDE_investment_percentage_l428_42816

/-- The investment problem with Vishal, Trishul, and Raghu -/
theorem investment_percentage (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →                  -- Vishal invested 10% more than Trishul
  raghu = 2200 →                            -- Raghu invested Rs. 2200
  vishal + trishul + raghu = 6358 →         -- Total sum of investments
  trishul < raghu →                         -- Trishul invested less than Raghu
  (raghu - trishul) / raghu * 100 = 10 :=   -- Percentage Trishul invested less than Raghu
by sorry

end NUMINAMATH_CALUDE_investment_percentage_l428_42816


namespace NUMINAMATH_CALUDE_point_on_line_l428_42854

/-- Given two points (m, n) and (m + p, n + 9) on the line x = y/3 - 2/5, prove that p = 3 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 3 - 2 / 5) →
  (m + p = (n + 9) / 3 - 2 / 5) →
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l428_42854


namespace NUMINAMATH_CALUDE_waiter_initial_customers_l428_42847

/-- The number of customers who left the waiter's section -/
def customers_left : ℕ := 12

/-- The number of people at each table after some customers left -/
def people_per_table : ℕ := 3

/-- The number of tables in the waiter's section -/
def number_of_tables : ℕ := 3

/-- The initial number of customers in the waiter's section -/
def initial_customers : ℕ := customers_left + people_per_table * number_of_tables

theorem waiter_initial_customers :
  initial_customers = 21 :=
by sorry

end NUMINAMATH_CALUDE_waiter_initial_customers_l428_42847


namespace NUMINAMATH_CALUDE_solve_equation_l428_42814

theorem solve_equation : ∃ y : ℝ, (3 * y - 15) / 7 = 18 ∧ y = 47 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l428_42814


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l428_42868

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : Prop := x^2 - 6*x + k = 0

-- Define an isosceles triangle
def isosceles_triangle (a b c : ℝ) : Prop := (a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ a + c > b

-- Main theorem
theorem isosceles_triangle_quadratic_roots (k : ℝ) : 
  (∃ x y : ℝ, 
    x ≠ y ∧ 
    quadratic_equation x k ∧ 
    quadratic_equation y k ∧
    isosceles_triangle 2 x y ∧
    triangle_inequality 2 x y) → k = 9 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l428_42868


namespace NUMINAMATH_CALUDE_base7_246_to_base10_l428_42899

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (d2 d1 d0 : ℕ) : ℕ :=
  d2 * 7^2 + d1 * 7^1 + d0 * 7^0

/-- The base 10 representation of 246 in base 7 is 132 -/
theorem base7_246_to_base10 : base7ToBase10 2 4 6 = 132 := by
  sorry

end NUMINAMATH_CALUDE_base7_246_to_base10_l428_42899


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_average_l428_42873

theorem arithmetic_sequence_middle_average (a : ℕ → ℕ) :
  (∀ i j, i < j → a i < a j) →  -- ascending order
  (∀ i, a (i + 1) - a i = a (i + 2) - a (i + 1)) →  -- arithmetic sequence
  (a 1 + a 2 + a 3) / 3 = 20 →  -- average of first three
  (a 5 + a 6 + a 7) / 3 = 24 →  -- average of last three
  (a 3 + a 4 + a 5) / 3 = 22 :=  -- average of middle three
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_average_l428_42873


namespace NUMINAMATH_CALUDE_quadratic_function_zeros_l428_42863

theorem quadratic_function_zeros (a : ℝ) :
  (∃ x y : ℝ, x > 2 ∧ y < -1 ∧
   -x^2 + a*x + 4 = 0 ∧
   -y^2 + a*y + 4 = 0) →
  0 < a ∧ a < 3 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_zeros_l428_42863


namespace NUMINAMATH_CALUDE_log_8_4_equals_twice_log_8_2_l428_42811

-- Define log_8 as a function
noncomputable def log_8 (x : ℝ) : ℝ := Real.log x / Real.log 8

-- State the theorem
theorem log_8_4_equals_twice_log_8_2 :
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 0.00005 ∧ 
  ∃ (δ : ℝ), δ ≥ 0 ∧ δ < 0.00005 ∧
  |log_8 2 - 0.2525| ≤ ε →
  |log_8 4 - 2 * log_8 2| ≤ δ :=
sorry

end NUMINAMATH_CALUDE_log_8_4_equals_twice_log_8_2_l428_42811


namespace NUMINAMATH_CALUDE_four_common_tangents_l428_42804

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y - 2 = 0

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 4 = 0

/-- The number of common tangent lines between C₁ and C₂ -/
def num_common_tangents : ℕ := 4

/-- Theorem stating that the number of common tangent lines between C₁ and C₂ is 4 -/
theorem four_common_tangents : num_common_tangents = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_common_tangents_l428_42804


namespace NUMINAMATH_CALUDE_E_72_with_4_equals_9_l428_42833

/-- The number of ways to express an integer as a product of integers greater than 1 -/
def E (n : ℕ) : ℕ := sorry

/-- The number of ways to express 72 as a product of integers greater than 1,
    including at least one factor of 4, where the order of factors matters -/
def E_72_with_4 : ℕ := sorry

/-- The prime factorization of 72 -/
def prime_factorization_72 : List ℕ := [2, 2, 2, 3, 3]

theorem E_72_with_4_equals_9 : E_72_with_4 = 9 := by sorry

end NUMINAMATH_CALUDE_E_72_with_4_equals_9_l428_42833


namespace NUMINAMATH_CALUDE_hawks_score_l428_42842

/-- The number of touchdowns scored by the Hawks -/
def num_touchdowns : ℕ := 3

/-- The number of points for each touchdown -/
def points_per_touchdown : ℕ := 7

/-- The total number of points scored by the Hawks -/
def total_points : ℕ := num_touchdowns * points_per_touchdown

/-- Theorem stating that the total points scored by the Hawks is 21 -/
theorem hawks_score :
  total_points = 21 := by sorry

end NUMINAMATH_CALUDE_hawks_score_l428_42842


namespace NUMINAMATH_CALUDE_solution_set_l428_42820

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f(x+1) is an odd function
axiom f_odd : ∀ x : ℝ, f (x + 1) = -f (-x - 1)

-- For any unequal real numbers x₁, x₂: x₁f(x₁) + x₂f(x₂) > x₁f(x₂) + x₂f(x₁)
axiom f_inequality : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

-- Theorem: The solution set of f(2-x) < 0 is (1,+∞)
theorem solution_set : {x : ℝ | f (2 - x) < 0} = Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l428_42820


namespace NUMINAMATH_CALUDE_shoe_box_problem_l428_42831

theorem shoe_box_problem (n : ℕ) (pairs : ℕ) (prob : ℚ) : 
  pairs = 7 →
  prob = 1 / 13 →
  prob = (pairs : ℚ) / (n.choose 2) →
  n = 14 :=
by sorry

end NUMINAMATH_CALUDE_shoe_box_problem_l428_42831


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l428_42810

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3, 
    ∀ y ∈ Set.Ioo (-1 : ℝ) 3, 
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l428_42810


namespace NUMINAMATH_CALUDE_mystery_discount_rate_l428_42869

/-- Represents the discount rate for books -/
structure DiscountRate :=
  (biography : ℝ)
  (mystery : ℝ)

/-- Represents the problem parameters -/
structure BookstoreParams :=
  (biography_price : ℝ)
  (mystery_price : ℝ)
  (biography_count : ℕ)
  (mystery_count : ℕ)
  (total_savings : ℝ)
  (total_discount_rate : ℝ)

/-- Theorem stating that given the problem conditions, the discount rate on mysteries is 37.5% -/
theorem mystery_discount_rate 
  (params : BookstoreParams)
  (h1 : params.biography_price = 20)
  (h2 : params.mystery_price = 12)
  (h3 : params.biography_count = 5)
  (h4 : params.mystery_count = 3)
  (h5 : params.total_savings = 19)
  (h6 : params.total_discount_rate = 43)
  : ∃ (d : DiscountRate), 
    d.biography + d.mystery = params.total_discount_rate ∧ 
    params.biography_count * params.biography_price * (d.biography / 100) + 
    params.mystery_count * params.mystery_price * (d.mystery / 100) = params.total_savings ∧
    d.mystery = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_mystery_discount_rate_l428_42869


namespace NUMINAMATH_CALUDE_prob_different_topics_correct_l428_42830

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5 / 6

/-- Theorem stating that the probability of two students selecting different topics
    from num_topics options is equal to prob_different_topics -/
theorem prob_different_topics_correct :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics := by
  sorry

end NUMINAMATH_CALUDE_prob_different_topics_correct_l428_42830


namespace NUMINAMATH_CALUDE_chessboard_coverage_l428_42829

/-- Represents a subregion of a chessboard -/
structure Subregion where
  rows : Finset Nat
  cols : Finset Nat

/-- The chessboard and its subregions -/
structure Chessboard where
  n : Nat
  subregions : Finset Subregion

/-- The semi-perimeter of a subregion -/
def semiPerimeter (s : Subregion) : Nat :=
  s.rows.card + s.cols.card

/-- Whether a subregion covers a cell -/
def covers (s : Subregion) (i j : Nat) : Prop :=
  i ∈ s.rows ∧ j ∈ s.cols

/-- The main diagonal of the chessboard -/
def mainDiagonal (n : Nat) : Set (Nat × Nat) :=
  {p | p.1 = p.2 ∧ p.1 < n}

/-- The theorem to be proved -/
theorem chessboard_coverage (cb : Chessboard) : 
  (∀ s ∈ cb.subregions, semiPerimeter s ≥ cb.n) →
  (∀ p ∈ mainDiagonal cb.n, ∃ s ∈ cb.subregions, covers s p.1 p.2) →
  (cb.subregions.sum (λ s => (s.rows.card * s.cols.card)) ≥ cb.n^2 / 2) :=
sorry

end NUMINAMATH_CALUDE_chessboard_coverage_l428_42829


namespace NUMINAMATH_CALUDE_tomato_plant_ratio_l428_42821

/-- Proves that the ratio of dead tomato plants to initial tomato plants is 1/2 --/
theorem tomato_plant_ratio (total_vegetables : ℕ) (vegetables_per_plant : ℕ) 
  (initial_tomato : ℕ) (initial_eggplant : ℕ) (initial_pepper : ℕ) 
  (dead_pepper : ℕ) : 
  total_vegetables = 56 →
  vegetables_per_plant = 7 →
  initial_tomato = 6 →
  initial_eggplant = 2 →
  initial_pepper = 4 →
  dead_pepper = 1 →
  (initial_tomato - (total_vegetables / vegetables_per_plant - initial_eggplant - (initial_pepper - dead_pepper))) / initial_tomato = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tomato_plant_ratio_l428_42821


namespace NUMINAMATH_CALUDE_lion_king_earnings_l428_42837

/-- Represents movie financial data in millions of dollars -/
structure MovieData where
  productionCost : ℝ
  boxOfficeEarnings : ℝ
  profit : ℝ

/-- The Lion King's box office earnings -/
def lionKingEarnings : ℝ := 200

theorem lion_king_earnings (starWars lionKing : MovieData) :
  starWars.productionCost = 25 →
  starWars.boxOfficeEarnings = 405 →
  lionKing.productionCost = 10 →
  lionKing.profit = (starWars.boxOfficeEarnings - starWars.productionCost) / 2 →
  lionKing.boxOfficeEarnings = lionKingEarnings := by
  sorry

end NUMINAMATH_CALUDE_lion_king_earnings_l428_42837


namespace NUMINAMATH_CALUDE_opposite_solutions_imply_a_equals_one_l428_42872

theorem opposite_solutions_imply_a_equals_one (x y a : ℝ) :
  (x + 3 * y = 4 - a) →
  (x - y = -3 * a) →
  (x = -y) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_imply_a_equals_one_l428_42872


namespace NUMINAMATH_CALUDE_girls_count_l428_42835

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- Theorem stating that given the conditions, the number of girls in the college is 160 -/
theorem girls_count (c : College) 
  (ratio : c.boys * 5 = c.girls * 8) 
  (total : c.boys + c.girls = 416) : 
  c.girls = 160 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_l428_42835


namespace NUMINAMATH_CALUDE_eighteen_digit_divisible_by_99_l428_42883

def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

def is_single_digit (d : ℕ) : Prop := d ≤ 9

def construct_number (x y : ℕ) : ℕ :=
  x * 10^17 + 3640548981270644 + y

theorem eighteen_digit_divisible_by_99 (x y : ℕ) :
  is_single_digit x ∧ is_single_digit y →
  (is_divisible_by_99 (construct_number x y) ↔ x = 9 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_eighteen_digit_divisible_by_99_l428_42883


namespace NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l428_42858

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 :=
sorry

end NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l428_42858


namespace NUMINAMATH_CALUDE_prism_to_spheres_waste_l428_42874

/-- The volume of waste when polishing a regular triangular prism into spheres -/
theorem prism_to_spheres_waste (base_side : ℝ) (height : ℝ) (sphere_radius : ℝ) :
  base_side = 6 →
  height = 8 * Real.sqrt 3 →
  sphere_radius = Real.sqrt 3 →
  ((Real.sqrt 3 / 4) * base_side^2 * height) - (4 * (4 / 3) * Real.pi * sphere_radius^3) =
    216 - 16 * Real.sqrt 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_prism_to_spheres_waste_l428_42874


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l428_42888

-- Define the basic types
variable (Space : Type)
variable (Plane : Type)
variable (Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Line → Prop)

-- State the theorems
theorem planes_parallel_to_same_plane_are_parallel
  (P Q R : Plane) (h1 : parallel P R) (h2 : parallel Q R) : parallel P Q :=
sorry

theorem planes_perpendicular_to_same_line_are_parallel
  (P Q : Plane) (L : Line) (h1 : perpendicular P L) (h2 : perpendicular Q L) : parallel P Q :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l428_42888


namespace NUMINAMATH_CALUDE_tigers_wins_l428_42852

def total_games : ℕ := 56
def losses : ℕ := 12

theorem tigers_wins : 
  let ties := losses / 2
  let wins := total_games - (losses + ties)
  wins = 38 := by sorry

end NUMINAMATH_CALUDE_tigers_wins_l428_42852


namespace NUMINAMATH_CALUDE_count_specially_monotonous_is_65_l428_42803

/-- A number is specially monotonous if all its digits are either all even or all odd,
    and the digits form either a strictly increasing or a strictly decreasing sequence
    when read from left to right. --/
def SpeciallyMonotonous (n : ℕ) : Prop := sorry

/-- The set of digits we consider (0 to 8) --/
def Digits : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8}

/-- Count of specially monotonous numbers with digits from 0 to 8 --/
def CountSpeciallyMonotonous : ℕ := sorry

/-- Theorem stating that the count of specially monotonous numbers is 65 --/
theorem count_specially_monotonous_is_65 : CountSpeciallyMonotonous = 65 := by sorry

end NUMINAMATH_CALUDE_count_specially_monotonous_is_65_l428_42803


namespace NUMINAMATH_CALUDE_max_value_abcd_l428_42802

def S : Finset ℕ := {1, 3, 5, 7}

theorem max_value_abcd (a b c d : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) 
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  (∀ (w x y z : ℕ), w ∈ S → x ∈ S → y ∈ S → z ∈ S → 
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z → 
    w * x + x * y + y * z + z * w ≤ a * b + b * c + c * d + d * a) →
  a * b + b * c + c * d + d * a = 64 :=
sorry

end NUMINAMATH_CALUDE_max_value_abcd_l428_42802


namespace NUMINAMATH_CALUDE_arrangements_a_middle_arrangements_a_b_not_adjacent_arrangements_a_b_not_ends_l428_42856

-- Define the number of people
def n : ℕ := 5

-- Define the function for number of permutations
def permutations (n : ℕ) (r : ℕ) : ℕ := n.factorial / (n - r).factorial

-- Theorem 1: Person A in the middle
theorem arrangements_a_middle : permutations (n - 1) (n - 1) = 24 := by sorry

-- Theorem 2: Person A and B not adjacent
theorem arrangements_a_b_not_adjacent : 
  (permutations 3 3) * (permutations 4 2) = 72 := by sorry

-- Theorem 3: Person A and B not at ends
theorem arrangements_a_b_not_ends : 
  (permutations 3 2) * (permutations 3 3) = 36 := by sorry

end NUMINAMATH_CALUDE_arrangements_a_middle_arrangements_a_b_not_adjacent_arrangements_a_b_not_ends_l428_42856


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l428_42884

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 0 ∧ x ≠ 2 ∧ (2 / x - 1 / (x - 2) = 0) ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l428_42884


namespace NUMINAMATH_CALUDE_prime_consecutive_property_l428_42864

theorem prime_consecutive_property (p : ℕ) (hp : Prime p) (hp2 : Prime (p + 2)) :
  p = 3 ∨ 6 ∣ (p + 1) :=
sorry

end NUMINAMATH_CALUDE_prime_consecutive_property_l428_42864


namespace NUMINAMATH_CALUDE_snow_probability_l428_42878

theorem snow_probability (p : ℝ) (h : p = 3/4) : 
  1 - (1 - p)^4 = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l428_42878


namespace NUMINAMATH_CALUDE_initial_crayons_l428_42886

theorem initial_crayons (taken_out : ℕ) (left : ℕ) : 
  taken_out = 3 → left = 4 → taken_out + left = 7 :=
by sorry

end NUMINAMATH_CALUDE_initial_crayons_l428_42886


namespace NUMINAMATH_CALUDE_min_value_quadratic_l428_42806

theorem min_value_quadratic (k : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9 ≥ 0) ∧
  (∃ x y : ℝ, 3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9 = 0) →
  k = 3/2 ∨ k = -3/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l428_42806


namespace NUMINAMATH_CALUDE_shampoo_comparison_l428_42855

/-- Represents a bottle of shampoo with weight in grams and price in yuan -/
structure ShampooBottle where
  weight : ℚ
  price : ℚ

/-- Calculates the cost per gram of a shampoo bottle -/
def costPerGram (bottle : ShampooBottle) : ℚ :=
  bottle.price / bottle.weight

theorem shampoo_comparison (large small : ShampooBottle)
  (h_large_weight : large.weight = 450)
  (h_large_price : large.price = 36)
  (h_small_weight : small.weight = 150)
  (h_small_price : small.price = 25/2) :
  (∃ (a b : ℕ), a = 72 ∧ b = 25 ∧ large.price / small.price = a / b) ∧
  costPerGram large < costPerGram small :=
sorry

end NUMINAMATH_CALUDE_shampoo_comparison_l428_42855


namespace NUMINAMATH_CALUDE_plants_original_cost_l428_42813

/-- Given a discount and the amount spent on plants, calculate the original cost. -/
def original_cost (discount : ℚ) (amount_spent : ℚ) : ℚ :=
  discount + amount_spent

/-- Theorem stating that given the specific discount and amount spent, the original cost is $467.00 -/
theorem plants_original_cost :
  let discount : ℚ := 399
  let amount_spent : ℚ := 68
  original_cost discount amount_spent = 467 := by
sorry

end NUMINAMATH_CALUDE_plants_original_cost_l428_42813
