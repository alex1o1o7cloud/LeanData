import Mathlib

namespace exists_linear_approximation_l1795_179573

/-- Cyclic distance in Fp -/
def cyclic_distance (p : ℕ) (x : Fin p) : ℕ :=
  min x.val (p - x.val)

/-- Almost additive function property -/
def almost_additive (p : ℕ) (f : Fin p → Fin p) : Prop :=
  ∀ x y : Fin p, cyclic_distance p (f (x + y) - f x - f y) < 100

/-- Main theorem -/
theorem exists_linear_approximation
  (p : ℕ) (hp : Nat.Prime p) (f : Fin p → Fin p) (hf : almost_additive p f) :
  ∃ m : Fin p, ∀ x : Fin p, cyclic_distance p (f x - m * x) < 1000 :=
sorry

end exists_linear_approximation_l1795_179573


namespace book_selection_probabilities_l1795_179547

def chinese_books : ℕ := 4
def math_books : ℕ := 3
def total_books : ℕ := chinese_books + math_books
def books_to_select : ℕ := 2

def total_combinations : ℕ := Nat.choose total_books books_to_select

theorem book_selection_probabilities :
  let prob_two_math : ℚ := (Nat.choose math_books books_to_select : ℚ) / total_combinations
  let prob_one_each : ℚ := (chinese_books * math_books : ℚ) / total_combinations
  prob_two_math = 1/7 ∧ prob_one_each = 4/7 := by sorry

end book_selection_probabilities_l1795_179547


namespace special_collection_books_l1795_179510

/-- The number of books in a special collection at the beginning of a month,
    given the number of books loaned, returned, and remaining at the end. -/
theorem special_collection_books
  (loaned : ℕ)
  (return_rate : ℚ)
  (end_count : ℕ)
  (h1 : loaned = 40)
  (h2 : return_rate = 7/10)
  (h3 : end_count = 63) :
  loaned * (1 - return_rate) + end_count = 47 :=
sorry

end special_collection_books_l1795_179510


namespace lower_right_is_one_l1795_179519

/-- Represents a 4x4 grid of integers -/
def Grid := Fin 4 → Fin 4 → Fin 4

/-- Checks if a given grid satisfies the Latin square property -/
def is_latin_square (g : Grid) : Prop :=
  (∀ i j k, i ≠ k → g i j ≠ g k j) ∧ 
  (∀ i j k, j ≠ k → g i j ≠ g i k)

/-- The initial configuration of the grid -/
def initial_config (g : Grid) : Prop :=
  g 0 0 = 0 ∧ g 0 3 = 3 ∧ g 1 1 = 1 ∧ g 2 2 = 2

theorem lower_right_is_one (g : Grid) 
  (h1 : is_latin_square g) 
  (h2 : initial_config g) : 
  g 3 3 = 0 := by sorry

end lower_right_is_one_l1795_179519


namespace fraction_2021_2019_position_l1795_179552

def sequence_position (m n : ℕ) : ℕ :=
  let k := m + n
  let previous_terms := (k - 1) * (k - 2) / 2
  let current_group_position := m
  previous_terms + current_group_position

theorem fraction_2021_2019_position :
  sequence_position 2021 2019 = 8159741 :=
by sorry

end fraction_2021_2019_position_l1795_179552


namespace factorization_of_quadratic_l1795_179536

theorem factorization_of_quadratic (x : ℝ) : 4 * x^2 - 2 * x = 2 * x * (2 * x - 1) := by
  sorry

end factorization_of_quadratic_l1795_179536


namespace triangle_radius_inequality_l1795_179514

/-- A structure representing a triangle with its circumradius and inradius -/
structure Triangle where
  R : ℝ  -- circumradius
  r : ℝ  -- inradius

/-- The theorem stating the relationship between circumradius and inradius of a triangle -/
theorem triangle_radius_inequality (t : Triangle) : 
  t.R ≥ 2 * t.r ∧ 
  (t.R = 2 * t.r ↔ ∃ (s : ℝ), s > 0 ∧ t.R = s * Real.sqrt 3 / 3 ∧ t.r = s / 3) ∧
  ∀ (R r : ℝ), R ≥ 2 * r → R > 0 → r > 0 → ∃ (t : Triangle), t.R = R ∧ t.r = r :=
sorry


end triangle_radius_inequality_l1795_179514


namespace complement_of_M_l1795_179541

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem complement_of_M : Set.compl M = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end complement_of_M_l1795_179541


namespace factor_sum_relation_l1795_179579

theorem factor_sum_relation (P Q R : ℝ) : 
  (∃ b c : ℝ, x^4 + P*x^2 + R*x + Q = (x^2 + 3*x + 7) * (x^2 + b*x + c)) →
  P + Q + R = 11*P - 1 := by
sorry

end factor_sum_relation_l1795_179579


namespace polynomial_product_expansion_l1795_179594

theorem polynomial_product_expansion :
  ∀ x : ℝ, (x^2 - 2*x + 2) * (x^2 + 2*x + 2) = x^4 + 4 := by
  sorry

end polynomial_product_expansion_l1795_179594


namespace train_speed_calculation_l1795_179593

/-- The speed of a train given another train passing in the opposite direction -/
theorem train_speed_calculation (passing_time : ℝ) (goods_train_length : ℝ) (goods_train_speed : ℝ) :
  passing_time = 9 →
  goods_train_length = 280 →
  goods_train_speed = 52 →
  ∃ (man_train_speed : ℝ), abs (man_train_speed - 60.16) < 0.01 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l1795_179593


namespace solution_set_inequality_l1795_179533

theorem solution_set_inequality (x : ℝ) :
  (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := by
  sorry

end solution_set_inequality_l1795_179533


namespace floor_equation_solution_set_l1795_179504

theorem floor_equation_solution_set (x : ℝ) :
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 5/3 ≤ x ∧ x < 7/3 :=
sorry

end floor_equation_solution_set_l1795_179504


namespace point_position_l1795_179527

/-- An isosceles triangle with a point on its base satisfying certain conditions -/
structure IsoscelesTriangleWithPoint where
  -- The length of the base of the isosceles triangle
  a : ℝ
  -- The height of the isosceles triangle
  h : ℝ
  -- The distance from one endpoint of the base to the point on the base
  x : ℝ
  -- Condition: a > 0 (positive base length)
  a_pos : a > 0
  -- Condition: h > 0 (positive height)
  h_pos : h > 0
  -- Condition: 0 < x < a (point is on the base)
  x_on_base : 0 < x ∧ x < a
  -- Condition: BM + MA = 2h
  sum_condition : x + (2 * h - x) = 2 * h

/-- Theorem: The position of the point on the base satisfies the quadratic equation -/
theorem point_position (t : IsoscelesTriangleWithPoint) : 
  t.x = t.h + (Real.sqrt (t.a^2 - 8 * t.h^2)) / 4 ∨ 
  t.x = t.h - (Real.sqrt (t.a^2 - 8 * t.h^2)) / 4 := by
  sorry

end point_position_l1795_179527


namespace expand_product_l1795_179564

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x - 5) = 6 * x^2 - 7 * x - 20 := by
  sorry

end expand_product_l1795_179564


namespace water_depth_for_specific_cylinder_l1795_179537

/-- Represents a cylindrical tower partially submerged in water -/
structure SubmergedCylinder where
  height : ℝ
  radius : ℝ
  aboveWaterRatio : ℝ

/-- Calculates the depth of water at the base of a partially submerged cylinder -/
def waterDepth (c : SubmergedCylinder) : ℝ :=
  c.height * (1 - c.aboveWaterRatio)

/-- Theorem stating the water depth for a specific cylinder -/
theorem water_depth_for_specific_cylinder :
  let c : SubmergedCylinder := {
    height := 1200,
    radius := 100,
    aboveWaterRatio := 1/3
  }
  waterDepth c = 400 := by sorry

end water_depth_for_specific_cylinder_l1795_179537


namespace M_intersect_N_equals_M_l1795_179575

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | |x - 1| < 1}
def N : Set ℝ := {x : ℝ | x * (x - 3) < 0}

-- State the theorem
theorem M_intersect_N_equals_M : M ∩ N = M := by sorry

end M_intersect_N_equals_M_l1795_179575


namespace ratio_problem_l1795_179554

theorem ratio_problem (a b c : ℕ+) (x m : ℚ) :
  (∃ (k : ℕ+), a = 4 * k ∧ b = 5 * k ∧ c = 6 * k) →
  x = a + (25 / 100) * a →
  m = b - (40 / 100) * b →
  Even c →
  (∀ (a' b' c' : ℕ+), (∃ (k' : ℕ+), a' = 4 * k' ∧ b' = 5 * k' ∧ c' = 6 * k') → 
    a + b + c ≤ a' + b' + c') →
  m / x = 3 / 5 := by
sorry

end ratio_problem_l1795_179554


namespace parabola_hyperbola_tangent_l1795_179570

theorem parabola_hyperbola_tangent (m : ℝ) : 
  (∀ x y : ℝ, y = x^2 + 4 ∧ y^2 - 4*m*x^2 = 4 → 
    (∃! u : ℝ, u^2 + (8 - 4*m)*u + 12 = 0)) →
  m = 2 + Real.sqrt 3 ∨ m = 2 - Real.sqrt 3 := by
sorry

end parabola_hyperbola_tangent_l1795_179570


namespace lg_properties_l1795_179507

-- Define the base 10 logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_properties :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 →
    (lg (x₁ * x₂) = lg x₁ + lg x₂) ∧
    (x₁ ≠ x₂ → (lg x₁ - lg x₂) / (x₁ - x₂) > 0) :=
by sorry

end lg_properties_l1795_179507


namespace T_properties_l1795_179534

-- Define the set T
def T : Set ℤ := {x | ∃ n : ℤ, x = n^2 + (n+2)^2 + (n+4)^2}

-- Statement to prove
theorem T_properties :
  (∀ x ∈ T, ¬(4 ∣ x)) ∧ (∃ x ∈ T, 13 ∣ x) := by sorry

end T_properties_l1795_179534


namespace football_players_count_l1795_179520

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 39)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 10) :
  ∃ football : ℕ, football = 26 ∧ (football - both) + (tennis - both) + both + neither = total :=
by sorry

end football_players_count_l1795_179520


namespace xy_squared_minus_x_squared_y_l1795_179542

theorem xy_squared_minus_x_squared_y (x y : ℝ) 
  (h1 : x - y = 1/2) 
  (h2 : x * y = 4/3) : 
  x * y^2 - x^2 * y = -2/3 := by
sorry

end xy_squared_minus_x_squared_y_l1795_179542


namespace triangle_not_right_angle_l1795_179540

theorem triangle_not_right_angle (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : A / 3 = B / 4) (h3 : A / 3 = C / 5) : 
  A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end triangle_not_right_angle_l1795_179540


namespace homework_duration_equation_l1795_179526

-- Define the variables
variable (a b x : ℝ)

-- Define the conditions
variable (h1 : a > 0)  -- Initial average daily homework duration is positive
variable (h2 : b > 0)  -- Current average weekly homework duration is positive
variable (h3 : 0 < x ∧ x < 1)  -- Rate of decrease is between 0 and 1

-- Theorem statement
theorem homework_duration_equation : a * (1 - x)^2 = b := by
  sorry

end homework_duration_equation_l1795_179526


namespace tourist_journey_times_l1795_179532

-- Define the speeds of the tourists
variable (v1 v2 : ℝ)

-- Define the time (in minutes) it takes the second tourist to travel the distance the first tourist covers in 120 minutes
variable (x : ℝ)

-- Define the total journey times for each tourist
def first_tourist_time : ℝ := 120 + x + 28
def second_tourist_time : ℝ := 60 + x

-- State the theorem
theorem tourist_journey_times 
  (h1 : x * v2 = 120 * v1) -- Distance equality at meeting point
  (h2 : v2 * (x + 60) = 120 * v1 + v1 * (x + 28)) -- Total distance equality
  : first_tourist_time = 220 ∧ second_tourist_time = 132 := by
  sorry


end tourist_journey_times_l1795_179532


namespace wrapping_paper_problem_l1795_179563

theorem wrapping_paper_problem (x : ℝ) 
  (h1 : x + (3/4 * x) + (x + 3/4 * x) = 7) : x = 2 := by
  sorry

end wrapping_paper_problem_l1795_179563


namespace bottles_sold_eq_60_l1795_179558

/-- Represents the sales data for Wal-Mart's thermometers and hot-water bottles --/
structure SalesData where
  thermometer_price : ℕ
  bottle_price : ℕ
  total_sales : ℕ
  thermometer_to_bottle_ratio : ℕ

/-- Calculates the number of hot-water bottles sold given the sales data --/
def bottles_sold (data : SalesData) : ℕ :=
  data.total_sales / (data.bottle_price + data.thermometer_price * data.thermometer_to_bottle_ratio)

/-- Theorem stating that given the specific sales data, 60 hot-water bottles were sold --/
theorem bottles_sold_eq_60 (data : SalesData) 
    (h1 : data.thermometer_price = 2)
    (h2 : data.bottle_price = 6)
    (h3 : data.total_sales = 1200)
    (h4 : data.thermometer_to_bottle_ratio = 7) : 
  bottles_sold data = 60 := by
  sorry

#eval bottles_sold { thermometer_price := 2, bottle_price := 6, total_sales := 1200, thermometer_to_bottle_ratio := 7 }

end bottles_sold_eq_60_l1795_179558


namespace custom_mult_solution_l1795_179584

/-- Custom multiplication operation -/
def custom_mult (a b : ℚ) : ℚ := 3 * a - 2 * b^2

/-- Theorem stating that if a * 4 = -7 using the custom multiplication, then a = 25/3 -/
theorem custom_mult_solution :
  ∀ a : ℚ, custom_mult a 4 = -7 → a = 25/3 := by
sorry

end custom_mult_solution_l1795_179584


namespace min_team_a_size_l1795_179582

theorem min_team_a_size : ∃ (a : ℕ), a > 0 ∧ 
  (∃ (b : ℕ), b > 0 ∧ b + 90 = 2 * (a - 90)) ∧
  (∃ (k : ℕ), a + k = 6 * (b - k)) ∧
  (∀ (a' : ℕ), a' > 0 → 
    (∃ (b' : ℕ), b' > 0 ∧ b' + 90 = 2 * (a' - 90)) →
    (∃ (k' : ℕ), a' + k' = 6 * (b' - k')) →
    a ≤ a') ∧
  a = 153 := by
sorry

end min_team_a_size_l1795_179582


namespace parallel_line_through_point_l1795_179589

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (p : Point) 
  (h_given : given_line.a = 6 ∧ given_line.b = -5 ∧ given_line.c = 3) 
  (h_point : p.x = 1 ∧ p.y = 1) :
  ∃ (result_line : Line), 
    result_line.a = 6 ∧ 
    result_line.b = -5 ∧ 
    result_line.c = -1 ∧ 
    parallel result_line given_line ∧ 
    pointOnLine p result_line := by
  sorry

end parallel_line_through_point_l1795_179589


namespace ship_grain_calculation_l1795_179511

/-- The amount of grain (in tons) that spilled into the water -/
def spilled_grain : ℕ := 49952

/-- The amount of grain (in tons) that remained onboard -/
def remaining_grain : ℕ := 918

/-- The original amount of grain (in tons) on the ship -/
def original_grain : ℕ := spilled_grain + remaining_grain

theorem ship_grain_calculation : original_grain = 50870 := by
  sorry

end ship_grain_calculation_l1795_179511


namespace adjacent_number_in_triangular_arrangement_l1795_179561

/-- Function to calculate the first number in the k-th row -/
def first_number_in_row (k : ℕ) : ℕ := (k - 1)^2 + 1

/-- Function to calculate the last number in the k-th row -/
def last_number_in_row (k : ℕ) : ℕ := k^2

/-- Function to determine if a number is in the k-th row -/
def is_in_row (n : ℕ) (k : ℕ) : Prop :=
  first_number_in_row k ≤ n ∧ n ≤ last_number_in_row k

/-- Function to calculate the number below a given number in the triangular arrangement -/
def number_below (n : ℕ) : ℕ :=
  let k := (n.sqrt + 1 : ℕ)
  let position := n - first_number_in_row k + 1
  first_number_in_row (k + 1) + position - 1

theorem adjacent_number_in_triangular_arrangement :
  is_in_row 267 17 → number_below 267 = 301 := by sorry

end adjacent_number_in_triangular_arrangement_l1795_179561


namespace solution_set_implies_a_value_l1795_179502

theorem solution_set_implies_a_value (a : ℝ) :
  ({x : ℝ | |x - a| < 1} = {x : ℝ | 2 < x ∧ x < 4}) → a = 3 := by
  sorry

end solution_set_implies_a_value_l1795_179502


namespace cone_volume_from_semicircle_l1795_179546

theorem cone_volume_from_semicircle (r : ℝ) (h : r = 6) :
  let l := r  -- slant height
  let base_radius := r / 2  -- derived from circumference equality
  let height := Real.sqrt (l^2 - base_radius^2)
  let volume := (1/3) * Real.pi * base_radius^2 * height
  volume = 9 * Real.sqrt 3 * Real.pi :=
by sorry

end cone_volume_from_semicircle_l1795_179546


namespace memory_sequence_increment_prime_l1795_179550

def memory_sequence : ℕ → ℕ
  | 0 => 6
  | n + 1 => memory_sequence n + Nat.gcd (memory_sequence n) (n + 1)

theorem memory_sequence_increment_prime (n : ℕ) :
  n > 0 → (memory_sequence n - memory_sequence (n - 1) = 1) ∨
          Nat.Prime (memory_sequence n - memory_sequence (n - 1)) :=
by sorry

end memory_sequence_increment_prime_l1795_179550


namespace fourth_child_age_l1795_179596

theorem fourth_child_age (ages : Fin 4 → ℕ) 
  (avg_age : (ages 0 + ages 1 + ages 2 + ages 3) / 4 = 9)
  (known_ages : ages 0 = 6 ∧ ages 1 = 8 ∧ ages 2 = 11) :
  ages 3 = 11 := by
  sorry

end fourth_child_age_l1795_179596


namespace expression_value_l1795_179528

theorem expression_value (a b c : ℝ) 
  (h1 : a - b = 2) 
  (h2 : a - c = Real.rpow 7 (1/3)) : 
  (c - b) * ((a - b)^2 + (a - b)*(a - c) + (a - c)^2) = 1 := by
  sorry

end expression_value_l1795_179528


namespace smallest_natural_with_congruences_l1795_179513

theorem smallest_natural_with_congruences (m : ℕ) : 
  (∀ k : ℕ, k < m → (k % 3 ≠ 1 ∨ k % 7 ≠ 5 ∨ k % 11 ≠ 4)) → 
  m % 3 = 1 → 
  m % 7 = 5 → 
  m % 11 = 4 → 
  m % 4 = 3 := by
sorry

end smallest_natural_with_congruences_l1795_179513


namespace divisor_of_q_l1795_179522

theorem divisor_of_q (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 30)
  (h2 : Nat.gcd q.val r.val = 42)
  (h3 : Nat.gcd r.val s.val = 66)
  (h4 : 80 < Nat.gcd s.val p.val ∧ Nat.gcd s.val p.val < 120) :
  5 ∣ q.val :=
by sorry

end divisor_of_q_l1795_179522


namespace inscribed_triangle_sum_l1795_179525

/-- An equilateral triangle inscribed in an ellipse -/
structure InscribedTriangle where
  /-- The x-coordinate of a vertex of the triangle -/
  x : ℝ
  /-- The y-coordinate of a vertex of the triangle -/
  y : ℝ
  /-- The condition that the vertex lies on the ellipse -/
  on_ellipse : x^2 + 9*y^2 = 9
  /-- The condition that one vertex is at (0, 1) -/
  vertex_at_origin : x = 0 ∧ y = 1
  /-- The condition that one altitude is aligned with the y-axis -/
  altitude_aligned : True  -- This condition is implicitly satisfied by the symmetry of the problem

/-- The theorem stating the result about the inscribed equilateral triangle -/
theorem inscribed_triangle_sum (t : InscribedTriangle) 
  (p q : ℕ) (h_coprime : Nat.Coprime p q) 
  (h_side_length : (12 * Real.sqrt 3 / 13)^2 = p / q) : 
  p + q = 601 := by sorry

end inscribed_triangle_sum_l1795_179525


namespace seashell_collection_problem_l1795_179544

/-- Calculates the total number of seashells after Leo gives away a quarter of his collection. -/
def final_seashell_count (henry_shells : ℕ) (paul_shells : ℕ) (initial_total : ℕ) : ℕ :=
  let leo_shells := initial_total - (henry_shells + paul_shells)
  let leo_remaining := leo_shells - (leo_shells / 4)
  henry_shells + paul_shells + leo_remaining

/-- Theorem stating that given the initial conditions, the final seashell count is 53. -/
theorem seashell_collection_problem :
  final_seashell_count 11 24 59 = 53 := by
  sorry

end seashell_collection_problem_l1795_179544


namespace intersection_A_complement_B_when_m_3_union_A_B_equals_A_iff_m_in_range_l1795_179518

-- Define sets A and B
def A : Set ℝ := {x | -3 < 2*x + 1 ∧ 2*x + 1 < 11}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2*m + 1}

-- Theorem 1
theorem intersection_A_complement_B_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | -2 < x ∧ x < 2} := by sorry

-- Theorem 2
theorem union_A_B_equals_A_iff_m_in_range (m : ℝ) :
  A ∪ B m = A ↔ m < -2 ∨ (-1 < m ∧ m < 2) := by sorry

end intersection_A_complement_B_when_m_3_union_A_B_equals_A_iff_m_in_range_l1795_179518


namespace sixteen_cows_days_to_finish_l1795_179568

/-- Represents the grass consumption scenario in a pasture -/
structure GrassConsumption where
  /-- Daily grass growth rate -/
  daily_growth : ℝ
  /-- Amount of grass each cow eats per day -/
  cow_consumption : ℝ
  /-- Original amount of grass in the pasture -/
  initial_grass : ℝ

/-- Theorem stating that 16 cows will take 18 days to finish the grass -/
theorem sixteen_cows_days_to_finish (gc : GrassConsumption) : 
  gc.initial_grass + 6 * gc.daily_growth = 24 * 6 * gc.cow_consumption →
  gc.initial_grass + 8 * gc.daily_growth = 21 * 8 * gc.cow_consumption →
  gc.initial_grass + 18 * gc.daily_growth = 16 * 18 * gc.cow_consumption := by
  sorry

#check sixteen_cows_days_to_finish

end sixteen_cows_days_to_finish_l1795_179568


namespace square_ratio_side_length_l1795_179500

theorem square_ratio_side_length (area_ratio : ℚ) :
  area_ratio = 270 / 125 →
  ∃ (a b c : ℕ), 
    (a = 3 ∧ b = 30 ∧ c = 25) ∧
    (Real.sqrt area_ratio = a * Real.sqrt b / c) ∧
    (a + b + c = 58) := by
  sorry

end square_ratio_side_length_l1795_179500


namespace arithmetic_trapezoid_third_largest_angle_l1795_179574

/-- Represents a trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  /-- The smallest angle of the trapezoid -/
  smallest_angle : ℝ
  /-- The common difference between consecutive angles -/
  angle_diff : ℝ

/-- The theorem statement -/
theorem arithmetic_trapezoid_third_largest_angle
  (trap : ArithmeticTrapezoid)
  (sum_smallest_largest : trap.smallest_angle + (trap.smallest_angle + 3 * trap.angle_diff) = 200)
  (second_smallest : trap.smallest_angle + trap.angle_diff = 70) :
  trap.smallest_angle + 2 * trap.angle_diff = 130 := by
  sorry

#check arithmetic_trapezoid_third_largest_angle

end arithmetic_trapezoid_third_largest_angle_l1795_179574


namespace mean_of_five_numbers_with_sum_two_thirds_l1795_179576

theorem mean_of_five_numbers_with_sum_two_thirds :
  ∀ (a b c d e : ℚ),
  a + b + c + d + e = 2/3 →
  (a + b + c + d + e) / 5 = 2/15 := by
  sorry

end mean_of_five_numbers_with_sum_two_thirds_l1795_179576


namespace unique_xxyy_square_l1795_179560

/-- Represents a four-digit number in the form xxyy --/
def xxyy_number (x y : Nat) : Nat :=
  1100 * x + 11 * y

/-- Predicate to check if a number is a perfect square --/
def is_perfect_square (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem unique_xxyy_square :
  ∀ x y : Nat, x < 10 → y < 10 →
    (is_perfect_square (xxyy_number x y) ↔ x = 7 ∧ y = 4) :=
by sorry

end unique_xxyy_square_l1795_179560


namespace function_composition_l1795_179531

theorem function_composition (f : ℝ → ℝ) :
  (∀ x, f (x - 1) = x^2 - 3*x) → (∀ x, f x = x^2 - x - 2) := by
  sorry

end function_composition_l1795_179531


namespace line_properties_l1795_179515

/-- Definition of line l₁ -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := x + 2 * m * y + 6 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

/-- Two lines are parallel if their slopes are equal -/
def parallel (m : ℝ) : Prop := (-1 / (2 * m)) = (-(m - 2) / (3 * m))

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m : ℝ) : Prop := (-1 / (2 * m)) * (-(m - 2) / (3 * m)) = -1

theorem line_properties (m : ℝ) :
  (parallel m ↔ m = 0 ∨ m = 7/2) ∧
  (perpendicular m ↔ m = -1/2 ∨ m = 2/3) :=
sorry

end line_properties_l1795_179515


namespace black_ball_probability_l1795_179505

theorem black_ball_probability 
  (p_red : ℝ) 
  (p_white : ℝ) 
  (h_red : p_red = 0.42) 
  (h_white : p_white = 0.28) :
  1 - p_red - p_white = 0.3 := by
  sorry

end black_ball_probability_l1795_179505


namespace new_regression_line_after_point_removal_l1795_179517

/-- Represents a sample point -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- Represents a regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the regression line from a list of sample points -/
def calculateRegressionLine (sample : List SamplePoint) : RegressionLine :=
  sorry

/-- Theorem stating the properties of the new regression line after removing two specific points -/
theorem new_regression_line_after_point_removal 
  (sample : List SamplePoint)
  (initial_line : RegressionLine)
  (mean_x : ℝ) :
  sample.length = 10 →
  initial_line = { slope := 2, intercept := -0.4 } →
  mean_x = 2 →
  let new_sample := sample.filter (λ p => ¬(p.x = -3 ∧ p.y = 1) ∧ ¬(p.x = 3 ∧ p.y = -1))
  let new_line := calculateRegressionLine new_sample
  new_line.slope = 3 →
  new_line = { slope := 3, intercept := -3 } :=
sorry

end new_regression_line_after_point_removal_l1795_179517


namespace book_distribution_l1795_179565

theorem book_distribution (people : ℕ) (books : ℕ) : 
  (5 * people = books + 2) →
  (4 * people + 3 = books) →
  (people = 5 ∧ books = 23) := by
sorry

end book_distribution_l1795_179565


namespace exists_coprime_linear_combination_divisible_l1795_179588

theorem exists_coprime_linear_combination_divisible (a b p : ℤ) :
  ∃ k l : ℤ, (Nat.gcd k.natAbs l.natAbs = 1) ∧ (∃ m : ℤ, a * k + b * l = p * m) := by
  sorry

end exists_coprime_linear_combination_divisible_l1795_179588


namespace arithmetic_mean_geq_geometric_mean_l1795_179583

theorem arithmetic_mean_geq_geometric_mean 
  (a b c : ℝ) 
  (ha : a ≥ 0) 
  (hb : b ≥ 0) 
  (hc : c ≥ 0) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) :=
sorry

end arithmetic_mean_geq_geometric_mean_l1795_179583


namespace third_term_of_geometric_sequence_l1795_179548

/-- A geometric sequence of positive integers -/
structure GeometricSequence where
  terms : ℕ → ℕ
  first_term : terms 1 = 6
  is_geometric : ∀ n : ℕ, n > 0 → ∃ r : ℚ, terms (n + 1) = (terms n : ℚ) * r

/-- The theorem stating the third term of the specific geometric sequence -/
theorem third_term_of_geometric_sequence
  (seq : GeometricSequence)
  (h_fourth : seq.terms 4 = 384) :
  seq.terms 3 = 96 := by
sorry

end third_term_of_geometric_sequence_l1795_179548


namespace restaurant_gratuity_calculation_l1795_179553

theorem restaurant_gratuity_calculation (dish_prices : List ℝ) (tip_percentage : ℝ) : 
  dish_prices = [10, 13, 17, 15, 20] → 
  tip_percentage = 0.18 → 
  (dish_prices.sum * tip_percentage) = 13.50 := by
sorry

end restaurant_gratuity_calculation_l1795_179553


namespace function_constant_l1795_179598

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - x) - Real.log (1 + x) + a

theorem function_constant (a : ℝ) :
  (∃ (M N : ℝ), (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f a x ≤ M ∧ N ≤ f a x) ∧ M + N = 1) →
  a = 1/2 := by
  sorry

end function_constant_l1795_179598


namespace square_plus_one_eq_empty_l1795_179567

theorem square_plus_one_eq_empty : {x : ℝ | x^2 + 1 = 0} = ∅ := by sorry

end square_plus_one_eq_empty_l1795_179567


namespace pistachio_shell_percentage_l1795_179587

theorem pistachio_shell_percentage (total : ℕ) (shell_percent : ℚ) (opened_shells : ℕ) : 
  total = 80 →
  shell_percent = 95 / 100 →
  opened_shells = 57 →
  (opened_shells : ℚ) / (shell_percent * total) * 100 = 75 := by
sorry

end pistachio_shell_percentage_l1795_179587


namespace friend_symmetry_iff_d_mod_7_eq_2_l1795_179572

def isFriend (d : ℕ) (M N : ℕ) : Prop :=
  ∀ k, k < d → (M + 10^k * ((N / 10^k) % 10 - (M / 10^k) % 10)) % 7 = 0

theorem friend_symmetry_iff_d_mod_7_eq_2 (d : ℕ) :
  (∀ M N : ℕ, M < 10^d → N < 10^d → (isFriend d M N ↔ isFriend d N M)) ↔ d % 7 = 2 :=
sorry

end friend_symmetry_iff_d_mod_7_eq_2_l1795_179572


namespace f_of_3_eq_one_over_17_l1795_179578

/-- Given f(x) = (x-2)/(4x+5), prove that f(3) = 1/17 -/
theorem f_of_3_eq_one_over_17 (f : ℝ → ℝ) (h : ∀ x, f x = (x - 2) / (4 * x + 5)) : 
  f 3 = 1 / 17 := by
sorry

end f_of_3_eq_one_over_17_l1795_179578


namespace magic_money_box_l1795_179590

def tripleEachDay (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (3 ^ days)

theorem magic_money_box (initial : ℕ) (days : ℕ) 
  (h1 : initial = 5) (h2 : days = 7) : 
  tripleEachDay initial days = 10935 := by
  sorry

end magic_money_box_l1795_179590


namespace expression_evaluation_l1795_179581

theorem expression_evaluation (a b : ℚ) (h1 : a = 1/2) (h2 : b = -2) :
  (2*a + b)^2 - (2*a - b)*(a + b) - 2*(a - 2*b)*(a + 2*b) = 37 := by
  sorry

end expression_evaluation_l1795_179581


namespace solve_linear_equation_l1795_179530

theorem solve_linear_equation (x : ℝ) (h : x - 3*x + 4*x = 120) : x = 60 := by
  sorry

end solve_linear_equation_l1795_179530


namespace square_numbers_between_20_and_120_divisible_by_3_l1795_179501

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem square_numbers_between_20_and_120_divisible_by_3 :
  {x : ℕ | is_square x ∧ x % 3 = 0 ∧ 20 < x ∧ x < 120} = {36, 81} := by
  sorry

end square_numbers_between_20_and_120_divisible_by_3_l1795_179501


namespace b_plus_3c_positive_l1795_179529

theorem b_plus_3c_positive (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + 3 * c > 0 := by
  sorry

end b_plus_3c_positive_l1795_179529


namespace paths_on_specific_grid_l1795_179591

/-- The number of paths on a rectangular grid from (0,0) to (m,n) moving only right or up -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The specific grid dimensions -/
def grid_width : ℕ := 7
def grid_height : ℕ := 3

theorem paths_on_specific_grid :
  grid_paths grid_width grid_height = 120 := by
  sorry

end paths_on_specific_grid_l1795_179591


namespace simplify_expression_l1795_179597

theorem simplify_expression (a b : ℝ) (h : a = -b) :
  (2 * a * b * (a^3 - b^3)) / (a^2 + a*b + b^2) - 
  ((a - b) * (a^4 - b^4)) / (a^2 - b^2) = -8 * a^3 := by
  sorry

#check simplify_expression

end simplify_expression_l1795_179597


namespace min_handshakes_coach_l1795_179543

/-- Represents the number of handshakes in a volleyball tournament --/
structure VolleyballTournament where
  n : ℕ  -- Total number of players
  m : ℕ  -- Number of players in the smaller team
  k₁ : ℕ -- Number of handshakes by the coach with fewer players
  h : ℕ  -- Total number of handshakes

/-- Conditions for the volleyball tournament --/
def tournament_conditions (t : VolleyballTournament) : Prop :=
  t.n = 3 * t.m ∧                                  -- Total players is 3 times the smaller team
  t.h = (t.n * (t.n - 1)) / 2 + 3 * t.k₁ ∧         -- Total handshakes equation
  t.h = 435                                        -- Given total handshakes

/-- Theorem stating the minimum number of handshakes for the coach with fewer players --/
theorem min_handshakes_coach (t : VolleyballTournament) :
  tournament_conditions t → t.k₁ ≥ 0 → t.k₁ = 0 :=
by
  sorry


end min_handshakes_coach_l1795_179543


namespace original_equals_scientific_l1795_179509

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficientInRange : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def originalNumber : ℕ := 44300000

/-- The scientific notation representation of the original number -/
def scientificForm : ScientificNotation := {
  coefficient := 4.43
  exponent := 7
  coefficientInRange := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (originalNumber : ℝ) = scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent := by
  sorry

end original_equals_scientific_l1795_179509


namespace min_odd_counties_for_valid_island_l1795_179562

/-- A rectangular county with a diagonal road -/
structure County where
  has_diagonal_road : Bool

/-- A rectangular island composed of counties -/
structure Island where
  counties : List County
  is_rectangular : Bool
  has_closed_path : Bool
  no_self_intersections : Bool

/-- Predicate to check if an island satisfies all conditions -/
def satisfies_conditions (island : Island) : Prop :=
  island.is_rectangular ∧
  island.has_closed_path ∧
  island.no_self_intersections ∧
  island.counties.length % 2 = 1 ∧
  ∀ c ∈ island.counties, c.has_diagonal_road

theorem min_odd_counties_for_valid_island :
  ∀ island : Island,
    satisfies_conditions island →
    island.counties.length ≥ 9 :=
by sorry

end min_odd_counties_for_valid_island_l1795_179562


namespace satisfying_function_is_identity_l1795_179508

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℕ → ℕ) : Prop :=
  (∀ k : ℕ, k > 0 → ∃ S : Set ℕ, S.Infinite ∧ ∀ p ∈ S, Prime p ∧ ∃ c : ℕ, c > 0 ∧ f c = p^k) ∧
  (∀ m n : ℕ, m > 0 ∧ n > 0 → (f m + f n) ∣ f (m + n))

/-- The main theorem stating that any function satisfying the conditions is the identity function -/
theorem satisfying_function_is_identity (f : ℕ → ℕ) (h : SatisfyingFunction f) : 
  ∀ n : ℕ, f n = n := by
  sorry

end satisfying_function_is_identity_l1795_179508


namespace intersection_forms_right_triangle_l1795_179571

/-- An ellipse with equation x²/m + y² = 1, where m > 1 -/
structure Ellipse where
  m : ℝ
  h_m : m > 1

/-- A hyperbola with equation x²/n - y² = 1, where n > 0 -/
structure Hyperbola where
  n : ℝ
  h_n : n > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents two curves (ellipse and hyperbola) with shared foci -/
structure SharedFociCurves where
  e : Ellipse
  h : Hyperbola
  f₁ : Point  -- First focus
  f₂ : Point  -- Second focus

/-- A point P that lies on both the ellipse and the hyperbola -/
structure IntersectionPoint (curves : SharedFociCurves) where
  p : Point
  on_ellipse : p.x ^ 2 / curves.e.m + p.y ^ 2 = 1
  on_hyperbola : p.x ^ 2 / curves.h.n - p.y ^ 2 = 1

/-- The main theorem: Triangle F₁PF₂ is always a right triangle -/
theorem intersection_forms_right_triangle (curves : SharedFociCurves) 
  (p : IntersectionPoint curves) : 
  (p.p.x - curves.f₁.x) ^ 2 + (p.p.y - curves.f₁.y) ^ 2 +
  (p.p.x - curves.f₂.x) ^ 2 + (p.p.y - curves.f₂.y) ^ 2 =
  (curves.f₁.x - curves.f₂.x) ^ 2 + (curves.f₁.y - curves.f₂.y) ^ 2 := by
  sorry

end intersection_forms_right_triangle_l1795_179571


namespace bobby_candy_theorem_l1795_179595

/-- The number of candy pieces Bobby ate -/
def pieces_eaten : ℕ := 23

/-- The number of candy pieces Bobby has left -/
def pieces_left : ℕ := 7

/-- The initial number of candy pieces Bobby had -/
def initial_pieces : ℕ := pieces_eaten + pieces_left

theorem bobby_candy_theorem : initial_pieces = 30 := by
  sorry

end bobby_candy_theorem_l1795_179595


namespace ribbon_length_difference_equals_side_length_specific_box_ribbon_difference_l1795_179512

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the ribbon length for the first method -/
def ribbonLength1 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 2 * box.width + 4 * box.height + bowLength

/-- Calculates the ribbon length for the second method -/
def ribbonLength2 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 4 * box.width + 2 * box.height + bowLength

/-- The main theorem to prove -/
theorem ribbon_length_difference_equals_side_length 
  (box : BoxDimensions) (bowLength : ℝ) : 
  ribbonLength2 box bowLength - ribbonLength1 box bowLength = box.length :=
by
  sorry

/-- The specific case with given dimensions -/
theorem specific_box_ribbon_difference :
  let box : BoxDimensions := ⟨22, 22, 11⟩
  let bowLength : ℝ := 24
  ribbonLength2 box bowLength - ribbonLength1 box bowLength = 22 :=
by
  sorry

end ribbon_length_difference_equals_side_length_specific_box_ribbon_difference_l1795_179512


namespace symmetric_center_of_f_l1795_179569

/-- The function f(x) = x³ - 6x² --/
def f (x : ℝ) : ℝ := x^3 - 6*x^2

/-- A function g is odd if g(-x) = -g(x) for all x --/
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

/-- The point (a, b) is a symmetric center of f if f(x+a) - b is an odd function --/
def is_symmetric_center (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  is_odd (fun x ↦ f (x + a) - b)

/-- The point (2, -16) is the symmetric center of f(x) = x³ - 6x² --/
theorem symmetric_center_of_f :
  is_symmetric_center f 2 (-16) :=
sorry

end symmetric_center_of_f_l1795_179569


namespace circle_intersection_theorem_l1795_179566

-- Define a circle in 2D plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if two circles intersect
def intersect (c1 c2 : Circle) : Prop := sorry

-- Define a function to get intersection points of two circles
def intersection_points (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

-- Define a function to check if points are concyclic or collinear
def concyclic_or_collinear (points : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem circle_intersection_theorem (S1 S2 S3 S4 : Circle) :
  intersect S1 S2 ∧ intersect S1 S4 ∧ intersect S3 S2 ∧ intersect S3 S4 →
  concyclic_or_collinear (intersection_points S1 S2 ∪ intersection_points S3 S4) →
  concyclic_or_collinear (intersection_points S1 S4 ∪ intersection_points S2 S3) :=
by sorry

end circle_intersection_theorem_l1795_179566


namespace line_through_point_parallel_to_line_l1795_179503

/-- A line passing through point (2,3) and parallel to 2x+4y-3=0 has equation x + 2y - 8 = 0 -/
theorem line_through_point_parallel_to_line :
  let line1 : ℝ → ℝ → Prop := λ x y => x + 2*y - 8 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 2*x + 4*y - 3 = 0
  (line1 2 3) ∧ 
  (∀ (x y : ℝ), line1 x y ↔ ∃ (k : ℝ), y = (-1/2)*x + k) ∧
  (∀ (x y : ℝ), line2 x y ↔ ∃ (k : ℝ), y = (-1/2)*x + k) :=
by sorry

end line_through_point_parallel_to_line_l1795_179503


namespace negative_squared_greater_than_product_l1795_179539

theorem negative_squared_greater_than_product {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a^2 > a*b := by
  sorry

end negative_squared_greater_than_product_l1795_179539


namespace modular_inverse_72_l1795_179535

theorem modular_inverse_72 (h : (17⁻¹ : ZMod 89) = 53) : (72⁻¹ : ZMod 89) = 36 := by
  sorry

end modular_inverse_72_l1795_179535


namespace quadratic_root_problem_l1795_179524

theorem quadratic_root_problem (m : ℝ) : 
  ((-5)^2 + m*(-5) - 10 = 0) → (2^2 + m*2 - 10 = 0) := by
  sorry

end quadratic_root_problem_l1795_179524


namespace hyperbola_equation_l1795_179586

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 3 = 1

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = (Real.sqrt 5 / 2) * x

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y : ℝ,
  (∃ f₁ f₂ : ℝ × ℝ, (∀ x y : ℝ, ellipse x y ↔ (x - f₁.1)^2 + y^2 = (x - f₂.1)^2 + y^2) ∧
                    (∀ x y : ℝ, hyperbola_C x y ↔ |(x - f₁.1)^2 + y^2| - |(x - f₂.1)^2 + y^2| = 2 * Real.sqrt 4)) →
  (∃ x y : ℝ, asymptote x y) →
  hyperbola_C x y :=
by sorry

end hyperbola_equation_l1795_179586


namespace triangle_properties_l1795_179592

def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi / 2 ∧
  Real.cos (2 * A) = -1 / 3 ∧
  c = Real.sqrt 3 ∧
  Real.sin A = Real.sqrt 6 * Real.sin C

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : 
  a = 3 * Real.sqrt 2 ∧ 
  b = 5 ∧ 
  (1 / 2 : ℝ) * b * c * Real.sin A = (5 * Real.sqrt 2) / 2 :=
by sorry

end triangle_properties_l1795_179592


namespace polynomial_root_product_l1795_179580

theorem polynomial_root_product (b c : ℤ) : 
  (∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b*r - c = 0) → b*c = 110 := by
  sorry

end polynomial_root_product_l1795_179580


namespace eight_sided_die_product_l1795_179577

theorem eight_sided_die_product (x : ℕ) (h : 1 ≤ x ∧ x ≤ 8) : 
  192 ∣ (Nat.factorial 8 / x) := by sorry

end eight_sided_die_product_l1795_179577


namespace solution_set_equivalence_l1795_179557

theorem solution_set_equivalence :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end solution_set_equivalence_l1795_179557


namespace min_distance_parallel_lines_l1795_179555

/-- The minimum distance between two parallel lines -/
theorem min_distance_parallel_lines :
  let line1 : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 6 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 6 * x + 8 * y + 3 = 0
  ∀ (P Q : ℝ × ℝ), line1 P.1 P.2 → line2 Q.1 Q.2 →
  (∀ (P' Q' : ℝ × ℝ), line1 P'.1 P'.2 → line2 Q'.1 Q'.2 →
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 3/2 :=
by
  sorry


end min_distance_parallel_lines_l1795_179555


namespace total_rice_weight_l1795_179516

-- Define the number of containers
def num_containers : ℕ := 4

-- Define the weight of rice in each container (in ounces)
def rice_per_container : ℚ := 25

-- Define the conversion rate from ounces to pounds
def ounces_per_pound : ℚ := 16

-- Theorem to prove
theorem total_rice_weight :
  (num_containers : ℚ) * rice_per_container / ounces_per_pound = 6.25 := by
  sorry

end total_rice_weight_l1795_179516


namespace symmetric_point_coordinates_l1795_179506

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

theorem symmetric_point_coordinates :
  let A : Point2D := { x := 2, y := -8 }
  let B : Point2D := symmetricYAxis A
  B.x = -2 ∧ B.y = -8 := by
  sorry

end symmetric_point_coordinates_l1795_179506


namespace toms_beach_trip_l1795_179559

/-- Tom's beach trip problem -/
theorem toms_beach_trip (daily_seashells : ℕ) (total_seashells : ℕ) (days : ℕ) :
  daily_seashells = 7 →
  total_seashells = 35 →
  total_seashells = daily_seashells * days →
  days = 5 := by
  sorry

end toms_beach_trip_l1795_179559


namespace log_equation_solution_l1795_179551

theorem log_equation_solution :
  ∃ x : ℝ, (Real.log (3 * x) - 4 * Real.log 9 = 3) ∧ (x = 2187000) := by
  sorry

end log_equation_solution_l1795_179551


namespace pie_eating_contest_l1795_179556

theorem pie_eating_contest (first_round first_second_round second_total : ℚ) 
  (h1 : first_round = 5/6)
  (h2 : first_second_round = 1/6)
  (h3 : second_total = 2/3) :
  first_round + first_second_round - second_total = 1/3 := by
  sorry

end pie_eating_contest_l1795_179556


namespace jack_emails_afternoon_l1795_179599

theorem jack_emails_afternoon (morning_emails : ℕ) (total_emails : ℕ) (afternoon_emails : ℕ) 
  (h1 : morning_emails = 3)
  (h2 : total_emails = 8)
  (h3 : afternoon_emails = total_emails - morning_emails) :
  afternoon_emails = 5 := by
  sorry

end jack_emails_afternoon_l1795_179599


namespace diamond_calculation_l1795_179545

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation :
  (diamond (diamond 2 (1/2)) (-4)) - (diamond 2 (diamond (1/2) (-4))) = -5/12 :=
by sorry

end diamond_calculation_l1795_179545


namespace books_read_l1795_179538

theorem books_read (total : ℕ) (unread : ℕ) (h1 : total = 13) (h2 : unread = 4) :
  total - unread = 9 := by
  sorry

end books_read_l1795_179538


namespace species_decline_year_l1795_179549

def species_decrease_rate : ℝ := 0.3
def threshold : ℝ := 0.05
def base_year : ℕ := 2010

def species_count (n : ℕ) : ℝ := (1 - species_decrease_rate) ^ n

theorem species_decline_year :
  ∃ k : ℕ, (species_count k < threshold) ∧ (∀ m : ℕ, m < k → species_count m ≥ threshold) ∧ (base_year + k = 2019) :=
sorry

end species_decline_year_l1795_179549


namespace problem_solution_l1795_179585

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 9) : x = 30 := by
  sorry

end problem_solution_l1795_179585


namespace complex_square_l1795_179521

theorem complex_square (z : ℂ) (h : z = 5 + 6 * Complex.I) : z^2 = -11 + 60 * Complex.I := by
  sorry

end complex_square_l1795_179521


namespace product_simplification_l1795_179523

theorem product_simplification (x : ℝ) (h : x ≠ 0) :
  (12 * x^3) * (8 * x^2) * (1 / (4*x)^3) = (3/2) * x^2 := by
  sorry

end product_simplification_l1795_179523
