import Mathlib

namespace arrangement_counts_l2605_260582

/-- Represents the number of teachers -/
def num_teachers : Nat := 2

/-- Represents the number of students -/
def num_students : Nat := 4

/-- Represents the total number of people -/
def total_people : Nat := num_teachers + num_students

/-- Calculates the number of arrangements with teachers at the ends -/
def arrangements_teachers_at_ends : Nat :=
  Nat.factorial num_students * Nat.factorial num_teachers

/-- Calculates the number of arrangements with teachers next to each other -/
def arrangements_teachers_together : Nat :=
  Nat.factorial (total_people - 1) * Nat.factorial num_teachers

/-- Calculates the number of arrangements with teachers not next to each other -/
def arrangements_teachers_apart : Nat :=
  Nat.factorial num_students * (num_students + 1) * (num_students + 1)

/-- Calculates the number of arrangements with two students between teachers -/
def arrangements_two_students_between : Nat :=
  (Nat.factorial num_students / (Nat.factorial 2 * Nat.factorial (num_students - 2))) *
  Nat.factorial num_teachers * Nat.factorial 3

theorem arrangement_counts :
  arrangements_teachers_at_ends = 48 ∧
  arrangements_teachers_together = 240 ∧
  arrangements_teachers_apart = 480 ∧
  arrangements_two_students_between = 144 := by
  sorry

end arrangement_counts_l2605_260582


namespace train_length_calculation_l2605_260579

-- Define the given constants
def bridge_crossing_time : Real := 30  -- seconds
def train_speed : Real := 45  -- km/hr
def bridge_length : Real := 230  -- meters

-- Define the theorem
theorem train_length_calculation :
  let speed_in_meters_per_second : Real := train_speed * 1000 / 3600
  let total_distance : Real := speed_in_meters_per_second * bridge_crossing_time
  let train_length : Real := total_distance - bridge_length
  train_length = 145 := by sorry

end train_length_calculation_l2605_260579


namespace geometric_sequence_product_l2605_260581

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  roots_equation : a 1 ^ 2 - 10 * a 1 + 16 = 0 ∧ a 99 ^ 2 - 10 * a 99 + 16 = 0

/-- The main theorem -/
theorem geometric_sequence_product (seq : GeometricSequence) :
  seq.a 20 * seq.a 50 * seq.a 80 = 64 ∨ seq.a 20 * seq.a 50 * seq.a 80 = -64 := by
  sorry


end geometric_sequence_product_l2605_260581


namespace hens_and_cows_l2605_260503

theorem hens_and_cows (total_animals : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) : 
  total_animals = 48 →
  total_feet = 140 →
  total_animals = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 26 := by
sorry

end hens_and_cows_l2605_260503


namespace tangent_line_equation_l2605_260527

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (m = f' 1) ∧
    (f 1 = m * 1 + b) ∧
    (m * 1 - f 1 + b = 0) ∧
    (m = 2 ∧ b = -1) := by
  sorry

end tangent_line_equation_l2605_260527


namespace set_equality_condition_l2605_260539

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 3}

-- State the theorem
theorem set_equality_condition (a : ℝ) : 
  A ∪ B a = A ↔ a ≤ -2 ∨ a ≥ 4 :=
sorry

end set_equality_condition_l2605_260539


namespace diamond_weight_calculation_l2605_260584

/-- The weight of a single diamond in grams -/
def diamond_weight : ℝ := sorry

/-- The weight of a single jade in grams -/
def jade_weight : ℝ := sorry

/-- The total weight of 5 diamonds in grams -/
def five_diamonds_weight : ℝ := 5 * diamond_weight

theorem diamond_weight_calculation :
  (4 * diamond_weight + 2 * jade_weight = 140) →
  (jade_weight = diamond_weight + 10) →
  five_diamonds_weight = 100 := by sorry

end diamond_weight_calculation_l2605_260584


namespace sqrt_294_simplification_l2605_260530

theorem sqrt_294_simplification : Real.sqrt 294 = 7 * Real.sqrt 6 := by
  sorry

end sqrt_294_simplification_l2605_260530


namespace area_of_quadrilateral_l2605_260550

/-- A quadrilateral with specific properties -/
structure Quadrilateral :=
  (EF HG EH FG : ℕ)
  (right_angle_F : EF ^ 2 + FG ^ 2 = 25)
  (right_angle_H : EH ^ 2 + HG ^ 2 = 25)
  (different_sides : ∃ (a b : ℕ), (a ≠ b) ∧ ((a = EF ∧ b = FG) ∨ (a = EH ∧ b = HG) ∨ (a = EF ∧ b = HG) ∨ (a = EH ∧ b = FG)))

/-- The area of the quadrilateral EFGH is 12 -/
theorem area_of_quadrilateral (q : Quadrilateral) : (q.EF * q.FG + q.EH * q.HG) / 2 = 12 :=
sorry

end area_of_quadrilateral_l2605_260550


namespace eight_chairs_subsets_l2605_260562

/-- The number of subsets of n chairs arranged in a circle that contain at least k adjacent chairs. -/
def subsets_with_adjacent_chairs (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The main theorem: For 8 chairs arranged in a circle, there are 33 subsets containing at least 4 adjacent chairs. -/
theorem eight_chairs_subsets : subsets_with_adjacent_chairs 8 4 = 33 := by sorry

end eight_chairs_subsets_l2605_260562


namespace perpendicular_lines_b_value_l2605_260583

theorem perpendicular_lines_b_value (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + 1 = 0 ∧ 3*x + b*y + 5 = 0 → 
   ((-a/2) * (-3/b) = -1)) →
  b = -3 :=
by sorry

end perpendicular_lines_b_value_l2605_260583


namespace inverse_square_problem_l2605_260555

-- Define the relationship between x and y
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y ^ 2)

theorem inverse_square_problem (k : ℝ) :
  inverse_square_relation k 1 3 →
  inverse_square_relation k (1/9) 9 :=
by
  sorry

end inverse_square_problem_l2605_260555


namespace work_efficiency_ratio_l2605_260585

theorem work_efficiency_ratio (a b : ℝ) : 
  a + b = 1 / 26 → b = 1 / 39 → a / b = 1 / 2 := by sorry

end work_efficiency_ratio_l2605_260585


namespace compute_expression_l2605_260536

theorem compute_expression : 3 * 3^3 - 9^50 / 9^48 = 0 := by
  sorry

end compute_expression_l2605_260536


namespace distribute_negation_l2605_260502

theorem distribute_negation (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b := by
  sorry

end distribute_negation_l2605_260502


namespace sixteen_power_divided_by_four_l2605_260551

theorem sixteen_power_divided_by_four (n : ℕ) : n = 16^2023 → n/4 = 4^4045 := by
  sorry

end sixteen_power_divided_by_four_l2605_260551


namespace hyperbola_proof_l2605_260545

/-- Given hyperbola -/
def given_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

/-- Given ellipse -/
def given_ellipse (x y : ℝ) : Prop := y^2 / 8 + x^2 / 2 = 1

/-- Desired hyperbola -/
def desired_hyperbola (x y : ℝ) : Prop := y^2 / 2 - x^2 / 4 = 1

/-- The theorem to be proved -/
theorem hyperbola_proof :
  ∀ x y : ℝ,
  (∃ k : ℝ, k ≠ 0 ∧ given_hyperbola (k*x) (k*y)) ∧  -- Same asymptotes condition
  (∃ fx fy : ℝ, given_ellipse fx fy ∧ desired_hyperbola fx fy) →  -- Shared focus condition
  desired_hyperbola x y :=
sorry

end hyperbola_proof_l2605_260545


namespace max_xy_value_l2605_260506

theorem max_xy_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 * x + 3 * y = 4) :
  ∃ (M : ℝ), M = 2/3 ∧ xy ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀ * y₀ = M ∧ 2 * x₀ + 3 * y₀ = 4 :=
sorry

end max_xy_value_l2605_260506


namespace geometric_sequence_ratio_l2605_260599

/-- Given a geometric sequence with positive terms and common ratio q where q^2 = 4,
    prove that (a_3 + a_4) / (a_4 + a_5) = 1/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Common ratio is q
  q^2 = 4 →  -- Given condition
  (a 3 + a 4) / (a 4 + a 5) = 1/2 :=
by sorry

end geometric_sequence_ratio_l2605_260599


namespace find_divisor_l2605_260542

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 161 →
  quotient = 10 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 16 := by
sorry

end find_divisor_l2605_260542


namespace smallest_multiplier_for_all_ones_l2605_260556

def is_all_ones (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 1

theorem smallest_multiplier_for_all_ones :
  ∃! N : ℕ, (N > 0) ∧ 
    is_all_ones (999999 * N) ∧
    (∀ m : ℕ, m > 0 → is_all_ones (999999 * m) → N ≤ m) ∧
    N = 111112 := by sorry

end smallest_multiplier_for_all_ones_l2605_260556


namespace sum_of_two_numbers_l2605_260540

theorem sum_of_two_numbers (larger smaller : ℕ) : 
  larger = 22 → larger - smaller = 10 → larger + smaller = 34 := by
  sorry

end sum_of_two_numbers_l2605_260540


namespace nested_sum_equals_2002_l2605_260521

def nested_sum (n : ℕ) : ℚ :=
  if n = 0 then 2
  else n + 1 + (1 / 2) * nested_sum (n - 1)

theorem nested_sum_equals_2002 : nested_sum 1001 = 2002 := by
  sorry

end nested_sum_equals_2002_l2605_260521


namespace book_length_is_300_l2605_260571

/-- The length of a book in pages -/
def book_length : ℕ := 300

/-- The fraction of the book Soja has finished reading -/
def finished_fraction : ℚ := 2/3

/-- The difference between pages read and pages left to read -/
def pages_difference : ℕ := 100

/-- Theorem stating that the book length is 300 pages -/
theorem book_length_is_300 : 
  book_length = 300 ∧ 
  finished_fraction * book_length - (1 - finished_fraction) * book_length = pages_difference := by
  sorry

end book_length_is_300_l2605_260571


namespace floor_ceiling_sum_l2605_260565

theorem floor_ceiling_sum : ⌊(3.999 : ℝ)⌋ + ⌈(4.001 : ℝ)⌉ = 8 := by
  sorry

end floor_ceiling_sum_l2605_260565


namespace inequality_solution_range_l2605_260591

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, -1/2 < x ∧ x ≤ 1 ∧ 2^x - a > Real.arccos x) ↔ 
  a < Real.sqrt 2 / 2 - 2 * Real.pi / 3 :=
sorry

end inequality_solution_range_l2605_260591


namespace least_positive_integer_with_congruences_l2605_260572

theorem least_positive_integer_with_congruences : ∃ (b : ℕ), 
  b > 0 ∧ 
  b % 3 = 2 ∧ 
  b % 5 = 4 ∧ 
  b % 6 = 5 ∧ 
  b % 7 = 6 ∧ 
  (∀ (x : ℕ), x > 0 ∧ x % 3 = 2 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ x % 7 = 6 → x ≥ b) ∧
  b = 209 :=
by sorry

end least_positive_integer_with_congruences_l2605_260572


namespace work_completion_time_l2605_260593

/-- Given that person B can complete 2/3 of a job in 12 days, 
    prove that B can complete the entire job in 18 days. -/
theorem work_completion_time (B_partial_time : ℕ) (B_partial_work : ℚ) 
  (h1 : B_partial_time = 12) 
  (h2 : B_partial_work = 2/3) : 
  ∃ (B_full_time : ℕ), B_full_time = 18 ∧ 
  B_partial_work / B_partial_time = 1 / B_full_time :=
sorry

end work_completion_time_l2605_260593


namespace x_power_n_plus_reciprocal_l2605_260511

theorem x_power_n_plus_reciprocal (θ : ℝ) (x : ℂ) (n : ℕ) (h1 : 0 < θ) (h2 : θ < π) (h3 : x + 1/x = 2 * Real.cos θ) :
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end x_power_n_plus_reciprocal_l2605_260511


namespace compound_interest_problem_l2605_260546

def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r) ^ t

theorem compound_interest_problem : 
  ∃ (P : ℝ) (r : ℝ), 
    compound_interest P r 2 = 8880 ∧ 
    compound_interest P r 3 = 9261 := by
  sorry

end compound_interest_problem_l2605_260546


namespace least_number_with_remainders_l2605_260561

theorem least_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 56 = 3 ∧
  n % 78 = 3 ∧
  n % 9 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m % 56 = 3 ∧ m % 78 = 3 ∧ m % 9 = 0 → n ≤ m :=
by
  -- The proof goes here
  sorry

end least_number_with_remainders_l2605_260561


namespace no_geometric_sequence_sin_angles_l2605_260548

theorem no_geometric_sequence_sin_angles :
  ¬∃ a : Real, 0 < a ∧ a < 2 * Real.pi ∧
  ∃ r : Real, (Real.sin (2 * a) = r * Real.sin a) ∧
             (Real.sin (3 * a) = r * Real.sin (2 * a)) := by
  sorry

end no_geometric_sequence_sin_angles_l2605_260548


namespace divide_decimals_l2605_260552

theorem divide_decimals : (0.08 : ℚ) / (0.002 : ℚ) = 40 := by sorry

end divide_decimals_l2605_260552


namespace special_polygon_area_l2605_260596

/-- A polygon with 24 congruent sides, where each side is perpendicular to its adjacent sides -/
structure SpecialPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  sides_eq : sides = 24
  perimeter_eq : perimeter = 48
  perimeter_formula : perimeter = sides * side_length

/-- The area of the special polygon is 64 -/
theorem special_polygon_area (p : SpecialPolygon) : 16 * p.side_length ^ 2 = 64 := by
  sorry

#check special_polygon_area

end special_polygon_area_l2605_260596


namespace sqrt_calculation_problems_l2605_260568

theorem sqrt_calculation_problems :
  (∃ (x : ℝ), x = Real.sqrt 18 - Real.sqrt 8 - Real.sqrt 2 ∧ x = 0) ∧
  (∃ (y : ℝ), y = 6 * Real.sqrt 2 * Real.sqrt 3 + 3 * Real.sqrt 30 / Real.sqrt 5 ∧ y = 9 * Real.sqrt 6) := by
  sorry

end sqrt_calculation_problems_l2605_260568


namespace meaningful_expression_l2605_260578

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 2 / Real.sqrt (x - 1)) ↔ x > 1 := by sorry

end meaningful_expression_l2605_260578


namespace calculation_proof_l2605_260560

theorem calculation_proof :
  let expr1 := -1^4 - (1/6) * (2 - (-3)^2) / (-7)
  let expr2 := (1 + 1/2 - 5/8 + 7/12) / (-1/24) - 8 * (-1/2)^3
  expr1 = -7/6 ∧ expr2 = -34 := by
  sorry

end calculation_proof_l2605_260560


namespace trajectory_equation_1_trajectory_equation_1_converse_l2605_260524

/-- Given points A(3,0) and B(-3,0), and a point P(x,y) such that the product of slopes of AP and BP is -2,
    prove that the trajectory of P satisfies the equation x²/9 + y²/18 = 1 for x ≠ ±3 -/
theorem trajectory_equation_1 (x y : ℝ) (h : x ≠ 3 ∧ x ≠ -3) :
  (y / (x - 3)) * (y / (x + 3)) = -2 → x^2 / 9 + y^2 / 18 = 1 := by
sorry

/-- The converse: if a point P(x,y) satisfies x²/9 + y²/18 = 1 for x ≠ ±3,
    then the product of slopes of AP and BP is -2 -/
theorem trajectory_equation_1_converse (x y : ℝ) (h : x ≠ 3 ∧ x ≠ -3) :
  x^2 / 9 + y^2 / 18 = 1 → (y / (x - 3)) * (y / (x + 3)) = -2 := by
sorry

end trajectory_equation_1_trajectory_equation_1_converse_l2605_260524


namespace bridge_length_bridge_length_proof_l2605_260557

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length

/-- Proof that the bridge length is approximately 131.98 meters -/
theorem bridge_length_proof :
  ∃ ε > 0, |bridge_length 110 36 24.198064154867613 - 131.98| < ε :=
by
  sorry

end bridge_length_bridge_length_proof_l2605_260557


namespace right_triangle_leg_lengths_l2605_260574

theorem right_triangle_leg_lengths 
  (c : ℝ) 
  (α β : ℝ) 
  (h_right : α + β = π / 2) 
  (h_tan : 6 * Real.tan β = Real.tan α + 1) :
  ∃ (a b : ℝ), 
    a^2 + b^2 = c^2 ∧ 
    a = (2 * c * Real.sqrt 5) / 5 ∧ 
    b = (c * Real.sqrt 5) / 5 := by
  sorry

end right_triangle_leg_lengths_l2605_260574


namespace max_value_3xy_plus_yz_l2605_260569

theorem max_value_3xy_plus_yz (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  3*x*y + y*z ≤ Real.sqrt 10 / 2 :=
sorry

end max_value_3xy_plus_yz_l2605_260569


namespace frog_arrangement_count_l2605_260549

/-- Represents the number of ways to arrange frogs with color restrictions -/
def frog_arrangements (n_green n_red n_blue : ℕ) : ℕ :=
  2 * (n_red.factorial * n_green.factorial)

/-- Theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 3 4 1 = 288 :=
by sorry

end frog_arrangement_count_l2605_260549


namespace corner_value_theorem_l2605_260573

/-- Represents a 3x3 grid with the given corner values -/
structure Grid :=
  (top_left : ℤ)
  (top_right : ℤ)
  (bottom_left : ℤ)
  (bottom_right : ℤ)
  (top_middle : ℤ)
  (left_middle : ℤ)
  (right_middle : ℤ)
  (bottom_middle : ℤ)
  (center : ℤ)

/-- Checks if all 2x2 subgrids have the same sum -/
def equal_subgrid_sums (g : Grid) : Prop :=
  g.top_left + g.top_middle + g.left_middle + g.center =
  g.top_middle + g.top_right + g.center + g.right_middle ∧
  g.left_middle + g.center + g.bottom_left + g.bottom_middle =
  g.center + g.right_middle + g.bottom_middle + g.bottom_right

/-- The main theorem -/
theorem corner_value_theorem (g : Grid) 
  (h1 : g.top_left = 2)
  (h2 : g.top_right = 4)
  (h3 : g.bottom_right = 3)
  (h4 : equal_subgrid_sums g) :
  g.bottom_left = 1 := by
  sorry

end corner_value_theorem_l2605_260573


namespace uncovered_area_square_circle_l2605_260516

/-- The area of a square that cannot be covered by a moving circle -/
theorem uncovered_area_square_circle (square_side : ℝ) (circle_diameter : ℝ) 
  (h_square : square_side = 4)
  (h_circle : circle_diameter = 1) :
  (square_side - circle_diameter) ^ 2 + π * (circle_diameter / 2) ^ 2 = 4 + π / 4 := by
  sorry

end uncovered_area_square_circle_l2605_260516


namespace jogger_train_distance_l2605_260567

/-- Calculates the distance between a jogger and a train engine given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_train_distance
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (passing_time : ℝ)
  (h1 : jogger_speed = 9 * 1000 / 3600)  -- 9 km/hr in m/s
  (h2 : train_speed = 45 * 1000 / 3600)  -- 45 km/hr in m/s
  (h3 : train_length = 120)              -- 120 meters
  (h4 : passing_time = 31)               -- 31 seconds
  : ∃ (distance : ℝ), distance = 190 ∧ distance = (train_speed - jogger_speed) * passing_time - train_length :=
by
  sorry


end jogger_train_distance_l2605_260567


namespace inequality_solution_set_l2605_260519

theorem inequality_solution_set (x : ℝ) : (3 * x + 1) / (1 - 2 * x) ≥ 0 ↔ -1/3 ≤ x ∧ x < 1/2 := by
  sorry

end inequality_solution_set_l2605_260519


namespace coin_probability_l2605_260597

/-- The probability of a specific sequence of coin flips -/
def sequence_probability (p : ℝ) : ℝ := p^2 * (1 - p)^3

/-- Theorem: If the probability of getting heads on the first 2 flips
    and tails on the last 3 flips is 1/32, then the probability of
    getting heads on a single flip is 1/2 -/
theorem coin_probability (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : sequence_probability p = 1/32) : 
  p = 1/2 := by
  sorry

#check coin_probability

end coin_probability_l2605_260597


namespace square_not_always_positive_l2605_260504

theorem square_not_always_positive : ¬(∀ a : ℝ, a^2 > 0) := by
  sorry

end square_not_always_positive_l2605_260504


namespace cosine_value_proof_l2605_260592

theorem cosine_value_proof (α : Real) 
    (h : Real.sin (α - π/3) = 1/3) : 
    Real.cos (π/6 + α) = 1/3 := by
  sorry

end cosine_value_proof_l2605_260592


namespace largest_root_range_l2605_260529

def polynomial (x b₃ b₂ b₁ b₀ : ℝ) : ℝ := x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀

def is_valid_coefficient (b : ℝ) : Prop := abs b < 3

theorem largest_root_range :
  ∃ s : ℝ, 3 < s ∧ s < 4 ∧
  (∀ b₃ b₂ b₁ b₀ : ℝ, is_valid_coefficient b₃ → is_valid_coefficient b₂ →
    is_valid_coefficient b₁ → is_valid_coefficient b₀ →
    (∀ x : ℝ, x > s → polynomial x b₃ b₂ b₁ b₀ ≠ 0)) ∧
  (∃ b₃ b₂ b₁ b₀ : ℝ, is_valid_coefficient b₃ ∧ is_valid_coefficient b₂ ∧
    is_valid_coefficient b₁ ∧ is_valid_coefficient b₀ ∧
    polynomial s b₃ b₂ b₁ b₀ = 0) :=
by sorry

end largest_root_range_l2605_260529


namespace monomial_sum_l2605_260580

theorem monomial_sum (m n : ℤ) (a b : ℝ) : 
  (∀ a b : ℝ, -2 * a^2 * b^(m+1) + n * a^2 * b^4 = 0) → m + n = 5 := by
sorry

end monomial_sum_l2605_260580


namespace complement_of_union_l2605_260559

theorem complement_of_union (U A B : Set ℕ) : 
  U = {x : ℕ | x > 0 ∧ x < 6} →
  A = {1, 3} →
  B = {3, 5} →
  (U \ (A ∪ B)) = {2, 4} := by
sorry

end complement_of_union_l2605_260559


namespace unique_solution_values_l2605_260517

/-- The function representing the quadratic expression inside the absolute value -/
def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 3*a

/-- The inequality condition -/
def inequality_condition (a x : ℝ) : Prop := |f a x| ≤ 2

/-- The property of having exactly one solution -/
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃! x, inequality_condition a x

/-- The main theorem stating that a = 1 and a = 2 are the only values satisfying the condition -/
theorem unique_solution_values :
  ∀ a : ℝ, has_exactly_one_solution a ↔ (a = 1 ∨ a = 2) :=
sorry

end unique_solution_values_l2605_260517


namespace store_discount_percentage_l2605_260590

theorem store_discount_percentage (C : ℝ) (C_pos : C > 0) : 
  let initial_price := 1.20 * C
  let new_year_price := 1.25 * initial_price
  let february_price := 1.20 * C
  let discount := new_year_price - february_price
  discount / new_year_price = 0.20 := by
sorry

end store_discount_percentage_l2605_260590


namespace successive_discounts_equivalence_l2605_260526

/-- Proves that three successive discounts are equivalent to a single discount --/
theorem successive_discounts_equivalence : 
  let original_price : ℝ := 800
  let discount1 : ℝ := 0.15
  let discount2 : ℝ := 0.10
  let discount3 : ℝ := 0.05
  let final_price : ℝ := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let single_discount : ℝ := 0.27325
  final_price = original_price * (1 - single_discount) := by
  sorry

#check successive_discounts_equivalence

end successive_discounts_equivalence_l2605_260526


namespace a_divisible_by_133_l2605_260531

/-- Sequence definition -/
def a (n : ℕ) : ℕ := 11^(n+2) + 12^(2*n+1)

/-- Main theorem: a_n is divisible by 133 for all n ≥ 0 -/
theorem a_divisible_by_133 (n : ℕ) : 133 ∣ a n := by sorry

end a_divisible_by_133_l2605_260531


namespace other_root_of_quadratic_l2605_260589

theorem other_root_of_quadratic (a : ℝ) : 
  (2^2 + 3*2 + a = 0) → (-5^2 + 3*(-5) + a = 0) := by
sorry

end other_root_of_quadratic_l2605_260589


namespace last_term_is_zero_l2605_260522

def first_term : ℤ := 0
def differences : List ℤ := [2, 4, -1, 0, -5, -3, 3]

theorem last_term_is_zero :
  first_term + differences.sum = 0 := by sorry

end last_term_is_zero_l2605_260522


namespace intersection_line_slope_l2605_260512

theorem intersection_line_slope (u : ℝ) :
  let line1 := {(x, y) : ℝ × ℝ | 2 * x + 3 * y = 8 * u + 4}
  let line2 := {(x, y) : ℝ × ℝ | 3 * x + 2 * y = 9 * u + 1}
  let intersection := {(x, y) : ℝ × ℝ | (x, y) ∈ line1 ∩ line2}
  ∃ (m b : ℝ), m = 6 / 47 ∧ ∀ (x y : ℝ), (x, y) ∈ intersection → y = m * x + b :=
by sorry

end intersection_line_slope_l2605_260512


namespace ellipse_foci_distance_l2605_260520

/-- 
Given an ellipse with equation x²/25 + y²/16 = 1, 
if the distance from a point P on the ellipse to one focus is 3, 
then the distance from P to the other focus is 7.
-/
theorem ellipse_foci_distance (x y : ℝ) (P : ℝ × ℝ) :
  x^2 / 25 + y^2 / 16 = 1 →  -- Ellipse equation
  P.1^2 / 25 + P.2^2 / 16 = 1 →  -- Point P is on the ellipse
  ∃ (F₁ F₂ : ℝ × ℝ), -- There exist two foci F₁ and F₂
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 3 ∨
     Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 3) →
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 7 ∨
     Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 7) :=
by sorry

end ellipse_foci_distance_l2605_260520


namespace first_pickup_fraction_proof_l2605_260505

/-- Represents the carrying capacity of the bus -/
def bus_capacity : ℕ := 80

/-- Represents the number of people waiting at the second pickup point -/
def second_pickup_waiting : ℕ := 50

/-- Represents the number of people who couldn't board at the second pickup point -/
def unable_to_board : ℕ := 18

/-- Represents the fraction of bus capacity that entered at the first pickup point -/
def first_pickup_fraction : ℚ := 3 / 5

theorem first_pickup_fraction_proof :
  first_pickup_fraction = (bus_capacity - (second_pickup_waiting - unable_to_board)) / bus_capacity :=
by sorry

end first_pickup_fraction_proof_l2605_260505


namespace complement_of_M_union_N_in_U_l2605_260566

open Set

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2}

-- Define set N
def N : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_of_M_union_N_in_U :
  (U \ (M ∪ N)) = {4} := by
  sorry

end complement_of_M_union_N_in_U_l2605_260566


namespace triangle_side_sum_l2605_260595

theorem triangle_side_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 60) (h3 : b = 30) (h4 : c = 90) 
  (side_opposite_30 : ℝ) (h5 : side_opposite_30 = 8 * Real.sqrt 3) :
  ∃ (other_sides_sum : ℝ), other_sides_sum = 12 + 8 * Real.sqrt 3 := by
  sorry

end triangle_side_sum_l2605_260595


namespace unique_solution_ceiling_equation_l2605_260508

theorem unique_solution_ceiling_equation :
  ∃! b : ℝ, b + ⌈b⌉ = 25.3 :=
by
  -- The proof would go here
  sorry

end unique_solution_ceiling_equation_l2605_260508


namespace expression_value_l2605_260525

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end expression_value_l2605_260525


namespace min_value_of_expression_l2605_260576

theorem min_value_of_expression :
  (∀ x y : ℝ, (2*x*y - 3)^2 + (x + y)^2 ≥ 1) ∧
  (∃ x y : ℝ, (2*x*y - 3)^2 + (x + y)^2 = 1) := by
  sorry

end min_value_of_expression_l2605_260576


namespace cubic_equation_solutions_no_solutions_for_2891_l2605_260577

def cubic_equation (x y n : ℤ) : Prop :=
  x^3 - 3*x*y^2 + y^3 = n

theorem cubic_equation_solutions (n : ℤ) (hn : n > 0) :
  (∃ x y : ℤ, cubic_equation x y n) →
  (∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℤ, 
    cubic_equation x₁ y₁ n ∧ 
    cubic_equation x₂ y₂ n ∧ 
    cubic_equation x₃ y₃ n ∧ 
    (x₁, y₁) ≠ (x₂, y₂) ∧ 
    (x₁, y₁) ≠ (x₃, y₃) ∧ 
    (x₂, y₂) ≠ (x₃, y₃)) :=
sorry

theorem no_solutions_for_2891 :
  ¬ ∃ x y : ℤ, cubic_equation x y 2891 :=
sorry

end cubic_equation_solutions_no_solutions_for_2891_l2605_260577


namespace probability_P_equals_1_plus_i_l2605_260500

/-- The set of vertices of a regular hexagon in the complex plane -/
def V : Set ℂ := {1, -1, Complex.I, -Complex.I, (1/2) + (Real.sqrt 3 / 2) * Complex.I, -(1/2) - (Real.sqrt 3 / 2) * Complex.I}

/-- The number of elements chosen from V -/
def n : ℕ := 10

/-- The product of n randomly chosen elements from V -/
noncomputable def P : ℂ := sorry

/-- The probability that P equals 1 + i -/
noncomputable def prob_P_equals_1_plus_i : ℝ := sorry

/-- Theorem stating the probability of P equaling 1 + i -/
theorem probability_P_equals_1_plus_i : prob_P_equals_1_plus_i = 120 / 24649 := by sorry

end probability_P_equals_1_plus_i_l2605_260500


namespace opposite_lateral_angle_is_90_l2605_260547

/-- A regular quadrangular pyramid -/
structure RegularQuadrangularPyramid where
  /-- The angle between a lateral face and the base plane -/
  lateral_base_angle : ℝ
  /-- The angle between a lateral face and the base plane is 45° -/
  angle_is_45 : lateral_base_angle = 45

/-- The angle between opposite lateral faces of the pyramid -/
def opposite_lateral_angle (p : RegularQuadrangularPyramid) : ℝ := sorry

/-- Theorem: In a regular quadrangular pyramid where the lateral face forms a 45° angle 
    with the base plane, the angle between opposite lateral faces is 90° -/
theorem opposite_lateral_angle_is_90 (p : RegularQuadrangularPyramid) :
  opposite_lateral_angle p = 90 := by sorry

end opposite_lateral_angle_is_90_l2605_260547


namespace max_difference_is_five_point_five_l2605_260533

/-- A structure representing a set of segments on the ray (0, +∞) -/
structure SegmentSet where
  /-- The left end of the leftmost segment -/
  a : ℝ
  /-- The right end of the rightmost segment -/
  b : ℝ
  /-- The number of segments (more than two) -/
  n : ℕ
  /-- n > 2 -/
  h_n : n > 2
  /-- a > 0 -/
  h_a : a > 0
  /-- b > a -/
  h_b : b > a
  /-- For any two different segments, there exist numbers that differ by a factor of 2 -/
  factor_of_two : ∀ i j, i ≠ j → i < n → j < n → ∃ x y, x ∈ Set.Icc (a + i) (a + i + 1) ∧ y ∈ Set.Icc (a + j) (a + j + 1) ∧ (x = 2 * y ∨ y = 2 * x)

/-- The theorem stating that the maximum value of b - a is 5.5 -/
theorem max_difference_is_five_point_five (s : SegmentSet) : 
  (∃ (s' : SegmentSet), s'.b - s'.a ≥ s.b - s.a) → s.b - s.a ≤ 5.5 := by
  sorry

end max_difference_is_five_point_five_l2605_260533


namespace star_operation_associative_l2605_260587

-- Define the curve y = x^3
def cubic_curve (x : ℝ) : ℝ := x^3

-- Define a point on the curve
structure CurvePoint where
  x : ℝ
  y : ℝ
  on_curve : y = cubic_curve x

-- Define the * operation
def star_operation (A B : CurvePoint) : CurvePoint :=
  sorry

-- Theorem statement
theorem star_operation_associative :
  ∀ (A B C : CurvePoint),
    star_operation (star_operation A B) C = star_operation A (star_operation B C) := by
  sorry

end star_operation_associative_l2605_260587


namespace octagon_cannot_tile_l2605_260535

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The interior angle of a regular polygon with n sides --/
def interiorAngle (n : ℕ) (p : RegularPolygon n) : ℚ :=
  180 - (360 / n)

/-- A regular polygon can tile the plane if its interior angle divides 360° evenly --/
def canTilePlane (n : ℕ) (p : RegularPolygon n) : Prop :=
  ∃ k : ℕ, k * interiorAngle n p = 360

/-- The set of regular polygons we're considering --/
def consideredPolygons : Set (Σ n, RegularPolygon n) :=
  {⟨3, ⟨by norm_num⟩⟩, ⟨4, ⟨by norm_num⟩⟩, ⟨6, ⟨by norm_num⟩⟩, ⟨8, ⟨by norm_num⟩⟩}

theorem octagon_cannot_tile :
  ∀ p ∈ consideredPolygons, ¬(canTilePlane p.1 p.2) ↔ p.1 = 8 := by
  sorry

#check octagon_cannot_tile

end octagon_cannot_tile_l2605_260535


namespace songs_per_album_l2605_260513

theorem songs_per_album (total_albums : ℕ) (total_songs : ℕ) 
  (h1 : total_albums = 3 + 5) 
  (h2 : total_songs = 24) 
  (h3 : ∀ (x : ℕ), x * total_albums = total_songs → x = 3) :
  ∃ (songs_per_album : ℕ), songs_per_album * total_albums = total_songs ∧ songs_per_album = 3 :=
by
  sorry

end songs_per_album_l2605_260513


namespace farmer_apples_l2605_260515

theorem farmer_apples (apples_given : ℕ) (apples_left : ℕ) : apples_given = 88 → apples_left = 39 → apples_given + apples_left = 127 := by
  sorry

end farmer_apples_l2605_260515


namespace train_length_calculation_train_length_is_120m_l2605_260514

/-- Given a jogger and a train moving in the same direction, calculate the length of the train. -/
theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) 
  (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let distance_covered := relative_speed * passing_time
  distance_covered - initial_distance

/-- The length of the train is 120 meters given the specified conditions. -/
theorem train_length_is_120m : 
  train_length_calculation 9 45 270 39 = 120 := by
  sorry

end train_length_calculation_train_length_is_120m_l2605_260514


namespace stratified_sampling_survey_l2605_260532

/-- Given a stratified sampling survey with a total population of 2400 (including 1000 female students),
    if 80 female students are included in a sample of size n, and the sampling fraction is consistent
    across all groups, then n = 192. -/
theorem stratified_sampling_survey (total_population : ℕ) (female_students : ℕ) (sample_size : ℕ) 
    (sampled_females : ℕ) (h1 : total_population = 2400) (h2 : female_students = 1000) 
    (h3 : sampled_females = 80) (h4 : sampled_females * total_population = sample_size * female_students) : 
    sample_size = 192 := by
  sorry

end stratified_sampling_survey_l2605_260532


namespace harolds_car_payment_l2605_260575

def monthly_income : ℚ := 2500
def rent : ℚ := 700
def groceries : ℚ := 50
def remaining_money : ℚ := 650

def car_payment (x : ℚ) : Prop :=
  let utilities := x / 2
  let total_expenses := rent + x + utilities + groceries
  let retirement_contribution := (monthly_income - total_expenses) / 2
  monthly_income - total_expenses - retirement_contribution = remaining_money

theorem harolds_car_payment :
  ∃ (x : ℚ), car_payment x ∧ x = 300 :=
sorry

end harolds_car_payment_l2605_260575


namespace sum_of_roots_squared_equation_l2605_260570

theorem sum_of_roots_squared_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a + b = 6) :=
by sorry

end sum_of_roots_squared_equation_l2605_260570


namespace arithmetic_sequence_30th_term_l2605_260588

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 30th term of the specified arithmetic sequence is 264. -/
theorem arithmetic_sequence_30th_term :
  ∀ a : ℕ → ℝ, is_arithmetic_sequence a →
  a 1 = 3 → a 2 = 12 → a 3 = 21 →
  a 30 = 264 := by
sorry

end arithmetic_sequence_30th_term_l2605_260588


namespace first_digit_base_16_l2605_260564

def base_4_representation : List ℕ := [2, 0, 3, 1, 3, 3, 2, 0, 1, 3, 2, 2, 2, 0, 3, 1, 2, 0, 3, 1]

def y : ℕ := (List.foldl (λ acc d => acc * 4 + d) 0 base_4_representation)

theorem first_digit_base_16 : ∃ (rest : ℕ), y = 5 * 16^rest + (y % 16^rest) ∧ y < 6 * 16^rest :=
sorry

end first_digit_base_16_l2605_260564


namespace expression_evaluation_l2605_260538

theorem expression_evaluation :
  (3^102 + 7^103)^2 - (3^102 - 7^103)^2 = 240 * 10^206 := by
  sorry

end expression_evaluation_l2605_260538


namespace quadratic_minimum_l2605_260518

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x ≥ -9 ∧ ∃ y : ℝ, y^2 + 6*y = -9 := by
  sorry

end quadratic_minimum_l2605_260518


namespace minimum_value_implies_a_l2605_260501

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, f a x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 0, f a x = 1) →
  a = 3 := by
  sorry

end minimum_value_implies_a_l2605_260501


namespace tan_of_angle_on_x_plus_y_equals_zero_l2605_260507

/-- An angle whose terminal side lies on the line x + y = 0 -/
structure AngleOnXPlusYEqualsZero where
  α : Real
  terminal_side : ∀ (x y : Real), x + y = 0 → (∃ (t : Real), x = t * Real.cos α ∧ y = t * Real.sin α)

/-- The tangent of an angle whose terminal side lies on the line x + y = 0 is -1 -/
theorem tan_of_angle_on_x_plus_y_equals_zero (θ : AngleOnXPlusYEqualsZero) : Real.tan θ.α = -1 := by
  sorry

end tan_of_angle_on_x_plus_y_equals_zero_l2605_260507


namespace daisies_per_bouquet_l2605_260537

/-- Represents a flower shop selling bouquets of roses and daisies. -/
structure FlowerShop where
  roses_per_bouquet : ℕ
  total_bouquets : ℕ
  rose_bouquets : ℕ
  daisy_bouquets : ℕ
  total_flowers : ℕ

/-- Theorem stating the number of daisies in each bouquet. -/
theorem daisies_per_bouquet (shop : FlowerShop)
  (h1 : shop.roses_per_bouquet = 12)
  (h2 : shop.total_bouquets = 20)
  (h3 : shop.rose_bouquets = 10)
  (h4 : shop.daisy_bouquets = 10)
  (h5 : shop.total_flowers = 190)
  (h6 : shop.total_bouquets = shop.rose_bouquets + shop.daisy_bouquets) :
  (shop.total_flowers - shop.roses_per_bouquet * shop.rose_bouquets) / shop.daisy_bouquets = 7 :=
by
  sorry

end daisies_per_bouquet_l2605_260537


namespace base_notes_on_hour_l2605_260544

/-- Represents the number of notes rung at each quarter-hour mark --/
def quarter_hour_notes : Fin 3 → ℕ
| 0 => 2  -- quarter past
| 1 => 4  -- half past
| 2 => 6  -- three-quarters past

/-- The total number of notes rung from 1:00 p.m. to 5:00 p.m. --/
def total_notes : ℕ := 103

/-- The number of hours from 1:00 p.m. to 5:00 p.m. --/
def hours : ℕ := 5

/-- Calculates the total notes rung at quarter-hour marks between two consecutive hours --/
def notes_between_hours : ℕ := (Finset.sum Finset.univ quarter_hour_notes)

/-- Theorem stating that the number of base notes rung on the hour is 8 --/
theorem base_notes_on_hour : 
  ∃ (B : ℕ), 
    hours * B + (Finset.sum (Finset.range (hours + 1)) id) + 
    (hours - 1) * notes_between_hours = total_notes ∧ B = 8 := by
  sorry

end base_notes_on_hour_l2605_260544


namespace inequality_equivalence_l2605_260598

/-- The set of points satisfying the inequality -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    (abs x ≤ 1 ∧ abs y ≤ 1 ∧ x * y ≤ 0) ∨
    (x^2 + y^2 ≤ 1 ∧ x * y > 0)}

/-- The main theorem -/
theorem inequality_equivalence (x y : ℝ) :
  Real.sqrt (1 - x^2) * Real.sqrt (1 - y^2) ≥ x * y ↔ (x, y) ∈ S :=
sorry

end inequality_equivalence_l2605_260598


namespace system_of_equations_solution_l2605_260554

theorem system_of_equations_solution (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) :
  (∃ (x y : ℝ), a₁ * x - b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂ ∧ x = 2 ∧ y = -1) →
  (∃ (x y : ℝ), a₁ * (x + 3) - b₁ * (y - 2) = c₁ ∧ a₂ * (x + 3) + b₂ * (y - 2) = c₂ ∧ x = -1 ∧ y = 1) :=
by sorry

end system_of_equations_solution_l2605_260554


namespace martin_initial_hens_correct_martin_initial_hens_unique_l2605_260586

/-- Represents the farm's egg production scenario -/
structure FarmScenario where
  initial_hens : ℕ
  initial_eggs : ℕ
  initial_days : ℕ
  added_hens : ℕ
  final_eggs : ℕ
  final_days : ℕ

/-- The specific scenario from the problem -/
def martin_farm : FarmScenario :=
  { initial_hens := 25,  -- This is what we want to prove
    initial_eggs := 80,
    initial_days := 10,
    added_hens := 15,
    final_eggs := 300,
    final_days := 15 }

/-- Theorem stating that Martin's initial number of hens is correct -/
theorem martin_initial_hens_correct :
  martin_farm.initial_hens * martin_farm.final_days * martin_farm.initial_eggs =
  martin_farm.initial_days * martin_farm.final_eggs * martin_farm.initial_hens +
  martin_farm.initial_days * martin_farm.final_eggs * martin_farm.added_hens :=
by sorry

/-- Theorem proving that 25 is the only solution -/
theorem martin_initial_hens_unique (h : ℕ) :
  h * martin_farm.final_days * martin_farm.initial_eggs =
  martin_farm.initial_days * martin_farm.final_eggs * h +
  martin_farm.initial_days * martin_farm.final_eggs * martin_farm.added_hens →
  h = martin_farm.initial_hens :=
by sorry

end martin_initial_hens_correct_martin_initial_hens_unique_l2605_260586


namespace subset_condition_l2605_260528

def A : Set ℝ := {x | 3*x + 6 > 0 ∧ 2*x - 10 < 0}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m < 3 := by sorry

end subset_condition_l2605_260528


namespace expression_simplification_l2605_260510

theorem expression_simplification :
  ((3 + 5 + 7 + 9) / 3) - ((4 * 6 + 13) / 5) = 3 / 5 := by
  sorry

end expression_simplification_l2605_260510


namespace integer_fraction_values_l2605_260523

theorem integer_fraction_values (k : ℤ) : 
  (∃ n : ℤ, (2 * k^2 + k - 8) / (k - 1) = n) ↔ k ∈ ({6, 2, 0, -4} : Set ℤ) := by
sorry

end integer_fraction_values_l2605_260523


namespace matthew_crackers_l2605_260534

theorem matthew_crackers (friends : ℕ) (cakes : ℕ) (eaten_crackers : ℕ) :
  friends = 4 →
  cakes = 98 →
  eaten_crackers = 8 →
  ∃ (initial_crackers : ℕ),
    initial_crackers = 128 ∧
    ∃ (given_per_friend : ℕ),
      given_per_friend * friends ≤ cakes ∧
      given_per_friend * friends ≤ initial_crackers ∧
      initial_crackers = given_per_friend * friends + eaten_crackers * friends :=
by
  sorry

end matthew_crackers_l2605_260534


namespace reciprocal_of_two_l2605_260594

theorem reciprocal_of_two :
  ∃ x : ℚ, x * 2 = 1 ∧ x = 1 / 2 := by sorry

end reciprocal_of_two_l2605_260594


namespace contrapositive_real_roots_l2605_260509

-- Define the original proposition
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 3*x + m = 0

-- Define the contrapositive
def contrapositive (P Q : Prop) : Prop := ¬Q → ¬P

-- Theorem statement
theorem contrapositive_real_roots :
  contrapositive (m < 0) (has_real_roots m) ↔ (¬(has_real_roots m) → m ≥ 0) :=
by sorry

end contrapositive_real_roots_l2605_260509


namespace function_value_at_cos_15_degrees_l2605_260553

theorem function_value_at_cos_15_degrees 
  (f : ℝ → ℝ) 
  (h : ∀ x, f (Real.sin x) = Real.cos (2 * x) - 1) :
  f (Real.cos (15 * π / 180)) = -Real.sqrt 3 / 2 - 1 := by
  sorry

end function_value_at_cos_15_degrees_l2605_260553


namespace cafeteria_green_apples_l2605_260543

theorem cafeteria_green_apples :
  let red_apples : ℕ := 43
  let students_wanting_fruit : ℕ := 2
  let extra_apples : ℕ := 73
  let green_apples : ℕ := red_apples + extra_apples + students_wanting_fruit - red_apples
  green_apples = 32 :=
by sorry

end cafeteria_green_apples_l2605_260543


namespace max_cube_volume_in_pyramid_l2605_260563

/-- The maximum volume of a cube inscribed in a pyramid --/
theorem max_cube_volume_in_pyramid (base_side : ℝ) (pyramid_height : ℝ) : 
  base_side = 2 →
  pyramid_height = 3 →
  ∃ (cube_volume : ℝ), 
    cube_volume = (81 * Real.sqrt 6) / 32 ∧ 
    ∀ (other_volume : ℝ), 
      (∃ (cube_side : ℝ), 
        cube_side > 0 ∧
        other_volume = cube_side ^ 3 ∧
        cube_side * Real.sqrt 2 ≤ 3 * Real.sqrt 3 / 2) →
      other_volume ≤ cube_volume :=
by sorry

end max_cube_volume_in_pyramid_l2605_260563


namespace cake_sugar_amount_l2605_260541

theorem cake_sugar_amount (total_sugar frosting_sugar : ℚ)
  (h1 : total_sugar = 0.8)
  (h2 : frosting_sugar = 0.6) :
  total_sugar - frosting_sugar = 0.2 := by
sorry

end cake_sugar_amount_l2605_260541


namespace eggs_per_year_is_3320_l2605_260558

/-- Represents the number of eggs used for each family member on a given day --/
structure EggUsage where
  children : Nat
  husband : Nat
  lisa : Nat

/-- Represents the egg usage for each day of the week and holidays --/
structure WeeklyEggUsage where
  monday : EggUsage
  tuesday : EggUsage
  wednesday : EggUsage
  thursday : EggUsage
  friday : EggUsage
  holiday : EggUsage

/-- Calculates the total number of eggs used in a year based on the weekly egg usage and number of holidays --/
def totalEggsPerYear (usage : WeeklyEggUsage) (numHolidays : Nat) : Nat :=
  let weekdayTotal := 
    (usage.monday.children * 3 + usage.monday.husband + usage.monday.lisa) * 52 +
    (usage.tuesday.children * 2 + usage.tuesday.husband + usage.tuesday.lisa + 2) * 52 +
    (usage.wednesday.children * 4 + usage.wednesday.husband + usage.wednesday.lisa) * 52 +
    (usage.thursday.children * 3 + usage.thursday.husband + usage.thursday.lisa) * 52 +
    (usage.friday.children * 4 + usage.friday.husband + usage.friday.lisa) * 52
  let holidayTotal := (usage.holiday.children * 4 + usage.holiday.husband + usage.holiday.lisa) * numHolidays
  weekdayTotal + holidayTotal

/-- The main theorem to prove --/
theorem eggs_per_year_is_3320 : 
  ∃ (usage : WeeklyEggUsage) (numHolidays : Nat),
    usage.monday = EggUsage.mk 2 3 2 ∧
    usage.tuesday = EggUsage.mk 2 3 2 ∧
    usage.wednesday = EggUsage.mk 3 4 3 ∧
    usage.thursday = EggUsage.mk 1 2 1 ∧
    usage.friday = EggUsage.mk 2 3 2 ∧
    usage.holiday = EggUsage.mk 2 2 2 ∧
    numHolidays = 8 ∧
    totalEggsPerYear usage numHolidays = 3320 := by
  sorry

end eggs_per_year_is_3320_l2605_260558
