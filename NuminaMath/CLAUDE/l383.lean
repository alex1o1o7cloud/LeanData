import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_boxes_n_equals_nine_smallest_n_is_nine_l383_38326

theorem smallest_n_boxes (n : ℕ) : 
  (∃ k : ℕ, 15 * n - 3 = 11 * k) → n ≥ 9 :=
by sorry

theorem n_equals_nine : 
  ∃ k : ℕ, 15 * 9 - 3 = 11 * k :=
by sorry

theorem smallest_n_is_nine : 
  (∀ m : ℕ, m < 9 → ¬∃ k : ℕ, 15 * m - 3 = 11 * k) ∧
  (∃ k : ℕ, 15 * 9 - 3 = 11 * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_boxes_n_equals_nine_smallest_n_is_nine_l383_38326


namespace NUMINAMATH_CALUDE_toucan_count_l383_38321

theorem toucan_count (initial_toucans : ℕ) (joining_toucans : ℕ) : 
  initial_toucans = 2 → joining_toucans = 1 → initial_toucans + joining_toucans = 3 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l383_38321


namespace NUMINAMATH_CALUDE_grid_column_contains_all_numbers_l383_38390

/-- Represents the state of the grid after a certain number of transformations -/
structure GridState (n : ℕ) :=
  (grid : Fin n → Fin n → Fin n)

/-- Represents the transformation rule for the grid -/
def transform_row (n k m : ℕ) (row : Fin n → Fin n) : Fin n → Fin n :=
  sorry

/-- Fills the grid according to the given rule -/
def fill_grid (n k m : ℕ) : ℕ → GridState n :=
  sorry

theorem grid_column_contains_all_numbers
  (n k m : ℕ) 
  (h_m_gt_k : m > k) 
  (h_coprime : Nat.Coprime m (n - k)) :
  ∀ (col : Fin n), 
    ∃ (rows : Finset (Fin n)), 
      rows.card = n ∧ 
      (∀ i : Fin n, ∃ row ∈ rows, (fill_grid n k m n).grid row col = i) :=
sorry

end NUMINAMATH_CALUDE_grid_column_contains_all_numbers_l383_38390


namespace NUMINAMATH_CALUDE_total_distance_is_twenty_l383_38379

/-- Represents the travel time per mile for each day -/
def travel_time (day : Nat) : Nat :=
  10 + 6 * (day - 1)

/-- Represents the distance traveled on each day -/
def distance (day : Nat) : Nat :=
  60 / travel_time day

/-- The total distance traveled over 5 days -/
def total_distance : Nat :=
  (List.range 5).map (fun i => distance (i + 1)) |>.sum

/-- Theorem stating that the total distance traveled is 20 miles -/
theorem total_distance_is_twenty : total_distance = 20 := by
  sorry

#eval total_distance

end NUMINAMATH_CALUDE_total_distance_is_twenty_l383_38379


namespace NUMINAMATH_CALUDE_dans_music_store_spending_l383_38363

/-- The amount Dan spent at the music store -/
def amount_spent (clarinet_cost song_book_cost amount_left : ℚ) : ℚ :=
  clarinet_cost + song_book_cost - amount_left

/-- Proof that Dan spent $129.22 at the music store -/
theorem dans_music_store_spending :
  amount_spent 130.30 11.24 12.32 = 129.22 := by
  sorry

end NUMINAMATH_CALUDE_dans_music_store_spending_l383_38363


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l383_38341

theorem decimal_to_fraction : (2.75 : ℚ) = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l383_38341


namespace NUMINAMATH_CALUDE_stall_problem_l383_38377

theorem stall_problem (area_diff : ℝ) (cost_A cost_B : ℝ) (total_area_A total_area_B : ℝ) (total_stalls : ℕ) :
  area_diff = 2 →
  cost_A = 20 →
  cost_B = 40 →
  total_area_A = 150 →
  total_area_B = 120 →
  total_stalls = 100 →
  ∃ (area_A area_B : ℝ) (num_A num_B : ℕ),
    area_A = area_B + area_diff ∧
    (total_area_A / area_A : ℝ) = (3/4) * (total_area_B / area_B) ∧
    num_A + num_B = total_stalls ∧
    num_B ≥ 3 * num_A ∧
    area_A = 5 ∧
    area_B = 3 ∧
    cost_A * area_A * num_A + cost_B * area_B * num_B = 11500 ∧
    ∀ (other_num_A other_num_B : ℕ),
      other_num_A + other_num_B = total_stalls →
      other_num_B ≥ 3 * other_num_A →
      cost_A * area_A * other_num_A + cost_B * area_B * other_num_B ≥ 11500 :=
by sorry

end NUMINAMATH_CALUDE_stall_problem_l383_38377


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l383_38348

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) :
  perimeter = 36 →
  area = (perimeter / 4) ^ 2 →
  area = 81 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l383_38348


namespace NUMINAMATH_CALUDE_max_spend_amount_l383_38361

/-- Represents the number of coins of each denomination a person has --/
structure CoinCount where
  coin100 : Nat
  coin50  : Nat
  coin10  : Nat

/-- Calculates the total value in won from a given CoinCount --/
def totalValue (coins : CoinCount) : Nat :=
  100 * coins.coin100 + 50 * coins.coin50 + 10 * coins.coin10

/-- Jimin's coin count --/
def jiminCoins : CoinCount := { coin100 := 5, coin50 := 1, coin10 := 0 }

/-- Seok-jin's coin count --/
def seokJinCoins : CoinCount := { coin100 := 2, coin50 := 0, coin10 := 7 }

/-- The theorem stating the maximum amount Jimin and Seok-jin can spend together --/
theorem max_spend_amount :
  totalValue jiminCoins + totalValue seokJinCoins = 820 := by sorry

end NUMINAMATH_CALUDE_max_spend_amount_l383_38361


namespace NUMINAMATH_CALUDE_square_area_increase_l383_38383

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_side := 1.1 * s
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l383_38383


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l383_38337

theorem negation_of_existential_proposition (l : ℝ) :
  (¬ ∃ x : ℝ, x + l ≥ 0) ↔ (∀ x : ℝ, x + l < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l383_38337


namespace NUMINAMATH_CALUDE_sum_of_fractions_l383_38330

theorem sum_of_fractions : 
  (5 : ℚ) / 13 + (9 : ℚ) / 11 = (172 : ℚ) / 143 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l383_38330


namespace NUMINAMATH_CALUDE_sum_of_angles_in_triangle_l383_38393

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define angles in a triangle
def angle (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

-- State the theorem
theorem sum_of_angles_in_triangle (t : Triangle) : 
  angle t 0 + angle t 1 + angle t 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_triangle_l383_38393


namespace NUMINAMATH_CALUDE_factor_count_l383_38318

def n : ℕ := 2^2 * 3^2 * 7^2

def is_factor (d : ℕ) : Prop := d ∣ n

def is_even (d : ℕ) : Prop := d % 2 = 0

def is_odd (d : ℕ) : Prop := d % 2 = 1

theorem factor_count :
  (∃ (even_factors : Finset ℕ) (odd_factors : Finset ℕ),
    (∀ d ∈ even_factors, is_factor d ∧ is_even d) ∧
    (∀ d ∈ odd_factors, is_factor d ∧ is_odd d) ∧
    (Finset.card even_factors = 18) ∧
    (Finset.card odd_factors = 9) ∧
    (∀ d : ℕ, is_factor d → (d ∈ even_factors ∨ d ∈ odd_factors))) :=
by sorry

end NUMINAMATH_CALUDE_factor_count_l383_38318


namespace NUMINAMATH_CALUDE_program_flowchart_unique_start_end_l383_38346

/-- Represents a chart with start and end points -/
structure Chart where
  start_points : ℕ
  end_points : ℕ

/-- Definition of a general flowchart -/
def is_flowchart (c : Chart) : Prop :=
  c.start_points = 1 ∧ c.end_points ≥ 1

/-- Definition of a program flowchart -/
def is_program_flowchart (c : Chart) : Prop :=
  c.start_points = 1 ∧ c.end_points = 1

/-- Definition of a structure chart (assumed equivalent to process chart) -/
def is_structure_chart (c : Chart) : Prop :=
  c.start_points = 1 ∧ c.end_points ≥ 1

/-- Theorem stating that a program flowchart has exactly one start point and one end point -/
theorem program_flowchart_unique_start_end :
  ∀ c : Chart, is_program_flowchart c → c.start_points = 1 ∧ c.end_points = 1 := by
  sorry


end NUMINAMATH_CALUDE_program_flowchart_unique_start_end_l383_38346


namespace NUMINAMATH_CALUDE_m_range_l383_38343

/-- Two points are on opposite sides of a line if the product of their signed distances from the line is negative -/
def opposite_sides (x₁ y₁ x₂ y₂ : ℝ) (a b c : ℝ) : Prop :=
  (a * x₁ + b * y₁ + c) * (a * x₂ + b * y₂ + c) < 0

/-- The theorem stating the range of m given the conditions -/
theorem m_range (m : ℝ) : 
  opposite_sides m 0 2 m 1 1 (-1) → -1 < m ∧ m < 1 := by
  sorry


end NUMINAMATH_CALUDE_m_range_l383_38343


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l383_38329

theorem vector_magnitude_proof (a b : ℝ × ℝ × ℝ) :
  a = (1, 1, 0) ∧ b = (-1, 0, 2) →
  ‖(2 • a) - b‖ = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l383_38329


namespace NUMINAMATH_CALUDE_half_day_division_count_l383_38342

/-- The number of seconds in a half-day -/
def half_day_seconds : ℕ := 43200

/-- The number of ways to divide a half-day into periods -/
def num_divisions : ℕ := 60

/-- Theorem: The number of ordered pairs of positive integers (n, m) 
    satisfying n * m = half_day_seconds is equal to num_divisions -/
theorem half_day_division_count :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = half_day_seconds ∧ 
                                   p.1 > 0 ∧ p.2 > 0) 
                 (Finset.product (Finset.range (half_day_seconds + 1)) 
                                 (Finset.range (half_day_seconds + 1)))).card = num_divisions :=
sorry

end NUMINAMATH_CALUDE_half_day_division_count_l383_38342


namespace NUMINAMATH_CALUDE_garden_area_l383_38303

/-- Represents a rectangular garden with specific properties. -/
structure Garden where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  lengthCondition : length = 3 * width + 10
  perimeterCondition : perimeter = 2 * (length + width)
  perimeterValue : perimeter = 400

/-- The area of a rectangular garden. -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the area of the garden with given conditions. -/
theorem garden_area (g : Garden) : g.area = 7243.75 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l383_38303


namespace NUMINAMATH_CALUDE_exactly_two_red_prob_l383_38328

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def num_draws : ℕ := 4
def num_red_draws : ℕ := 2

def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem exactly_two_red_prob : 
  (Nat.choose num_draws num_red_draws : ℚ) * prob_red ^ num_red_draws * prob_white ^ (num_draws - num_red_draws) = 3456 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_red_prob_l383_38328


namespace NUMINAMATH_CALUDE_product_expansion_l383_38388

theorem product_expansion (a b c : ℝ) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l383_38388


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l383_38333

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_second : a 2 = 4)
  (h_sixth : a 6 = 2) :
  a 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l383_38333


namespace NUMINAMATH_CALUDE_equation_represents_circle_l383_38382

-- Define the equation
def equation (x y : ℝ) : Prop := (x - 0)^2 + (y - 0)^2 = 25

-- Define what a circle is in terms of its equation
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (center_x center_y radius : ℝ), 
    ∀ (x y : ℝ), f x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

-- Theorem statement
theorem equation_represents_circle : is_circle equation := by
  sorry

end NUMINAMATH_CALUDE_equation_represents_circle_l383_38382


namespace NUMINAMATH_CALUDE_max_cubes_from_seven_points_l383_38391

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Determines if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Determines if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (plane : Plane3D), pointOnPlane p1 plane ∧ pointOnPlane p2 plane ∧ 
                       pointOnPlane p3 plane ∧ pointOnPlane p4 plane

/-- Represents a cube determined by 7 points -/
structure Cube where
  a1 : Point3D
  a2 : Point3D
  f1 : Point3D
  f2 : Point3D
  e : Point3D
  h : Point3D
  j : Point3D
  lowerFace : Plane3D
  upperFace : Plane3D
  frontFace : Plane3D
  backFace : Plane3D
  rightFace : Plane3D

/-- The main theorem to prove -/
theorem max_cubes_from_seven_points 
  (a1 a2 f1 f2 e h j : Point3D)
  (h1 : pointOnPlane a1 (Cube.lowerFace cube))
  (h2 : pointOnPlane a2 (Cube.lowerFace cube))
  (h3 : pointOnPlane f1 (Cube.upperFace cube))
  (h4 : pointOnPlane f2 (Cube.upperFace cube))
  (h5 : ¬ areCoplanar a1 a2 f1 f2)
  (h6 : pointOnPlane e (Cube.frontFace cube))
  (h7 : pointOnPlane h (Cube.backFace cube))
  (h8 : pointOnPlane j (Cube.rightFace cube))
  : ∃ (n : ℕ), n ≤ 2 ∧ ∀ (m : ℕ), (∃ (cubes : Fin m → Cube), 
    (∀ (i : Fin m), 
      Cube.a1 (cubes i) = a1 ∧
      Cube.a2 (cubes i) = a2 ∧
      Cube.f1 (cubes i) = f1 ∧
      Cube.f2 (cubes i) = f2 ∧
      Cube.e (cubes i) = e ∧
      Cube.h (cubes i) = h ∧
      Cube.j (cubes i) = j) ∧
    (∀ (i j : Fin m), i ≠ j → cubes i ≠ cubes j)) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_cubes_from_seven_points_l383_38391


namespace NUMINAMATH_CALUDE_liam_strawberry_candies_l383_38309

theorem liam_strawberry_candies :
  ∀ (s g : ℕ),
  s = 3 * g →                     -- Initial condition
  s - 15 = 4 * (g - 15) →         -- Condition after giving away candies
  s = 135 :=                      -- Conclusion to prove
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_liam_strawberry_candies_l383_38309


namespace NUMINAMATH_CALUDE_number_equation_solution_l383_38344

theorem number_equation_solution : 
  ∃ (number : ℝ), 35 - (23 - (number - 32)) = 12 * 2 / (1 / 2) ∧ number = 68 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l383_38344


namespace NUMINAMATH_CALUDE_no_solution_condition_l383_38350

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, 5 * |x - 4*a| + |x - a^2| + 4*x - 3*a ≠ 0) ↔ (a < -9 ∨ a > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l383_38350


namespace NUMINAMATH_CALUDE_average_height_combined_groups_l383_38312

theorem average_height_combined_groups
  (group1_count : ℕ)
  (group2_count : ℕ)
  (total_count : ℕ)
  (average_height : ℝ)
  (h1 : group1_count = 20)
  (h2 : group2_count = 11)
  (h3 : total_count = group1_count + group2_count)
  (h4 : average_height = 20) :
  (group1_count * average_height + group2_count * average_height) / total_count = average_height :=
by sorry

end NUMINAMATH_CALUDE_average_height_combined_groups_l383_38312


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l383_38334

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 1 * a 3 + 2 * a 2 * a 4 + a 2 * a 6 = 9 →
  a 2 + a 4 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l383_38334


namespace NUMINAMATH_CALUDE_max_triangle_area_l383_38366

theorem max_triangle_area (a b : ℝ) (ha : a = 1984) (hb : b = 2016) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π → (1/2) * a * b * Real.sin θ ≤ 1998912) ∧
  (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ (1/2) * a * b * Real.sin θ = 1998912) := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_l383_38366


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l383_38398

theorem quadratic_equation_solutions (a : ℝ) : a^2 + 10 = a + 10^2 ↔ a = 10 ∨ a = -9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l383_38398


namespace NUMINAMATH_CALUDE_value_of_expression_l383_38354

theorem value_of_expression (x : ℝ) (h : 10000 * x + 2 = 4) : 5000 * x + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l383_38354


namespace NUMINAMATH_CALUDE_total_cost_theorem_l383_38356

def original_price : ℝ := 10
def child_discount : ℝ := 0.3
def senior_discount : ℝ := 0.1
def handling_fee : ℝ := 5
def num_child_tickets : ℕ := 2
def num_senior_tickets : ℕ := 2

def senior_ticket_price : ℝ := 14

theorem total_cost_theorem :
  let child_ticket_price := (1 - child_discount) * original_price + handling_fee
  let total_cost := num_child_tickets * child_ticket_price + num_senior_tickets * senior_ticket_price
  total_cost = 52 := by sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l383_38356


namespace NUMINAMATH_CALUDE_integral_cube_root_x_squared_plus_sqrt_x_l383_38394

theorem integral_cube_root_x_squared_plus_sqrt_x (x : ℝ) :
  (deriv (fun x => (3/5) * x * (x^2)^(1/3) + (2/3) * x * x^(1/2))) x = x^(2/3) + x^(1/2) :=
by sorry

end NUMINAMATH_CALUDE_integral_cube_root_x_squared_plus_sqrt_x_l383_38394


namespace NUMINAMATH_CALUDE_parabola_shift_l383_38306

/-- The original parabola function -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 7

/-- The reference parabola function -/
def g (x : ℝ) : ℝ := x^2

/-- The shifted reference parabola function -/
def h (x : ℝ) : ℝ := g (x + 3) - 2

theorem parabola_shift :
  ∀ x : ℝ, f x = h x :=
sorry

end NUMINAMATH_CALUDE_parabola_shift_l383_38306


namespace NUMINAMATH_CALUDE_f_odd_f_inequality_iff_a_range_l383_38347

noncomputable section

def f (x : ℝ) : ℝ := (Real.exp x - 1) / (Real.exp x + 1)

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem f_inequality_iff_a_range :
  ∀ a : ℝ, (∀ x : ℝ, x > 1 ∧ x < 2 → f (a * x^2 + 2) + f (2 * x - 1) > 0) ↔ a > -5/4 := by sorry

end NUMINAMATH_CALUDE_f_odd_f_inequality_iff_a_range_l383_38347


namespace NUMINAMATH_CALUDE_village_population_l383_38367

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 90 / 100 →
  partial_population = 45000 →
  (percentage : ℚ) * total_population = partial_population →
  total_population = 50000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l383_38367


namespace NUMINAMATH_CALUDE_letter_value_proof_l383_38364

/-- Given random integer values for letters of the alphabet, prove that A = 16 -/
theorem letter_value_proof (M A T E : ℤ) : 
  M + A + T + 8 = 28 →
  T + E + A + M = 34 →
  M + E + E + T = 30 →
  A = 16 := by
  sorry

end NUMINAMATH_CALUDE_letter_value_proof_l383_38364


namespace NUMINAMATH_CALUDE_grace_september_earnings_l383_38389

/-- Calculates Grace's earnings for landscaping in September --/
theorem grace_september_earnings :
  let mowing_rate : ℕ := 6
  let weeding_rate : ℕ := 11
  let mulching_rate : ℕ := 9
  let mowing_hours : ℕ := 63
  let weeding_hours : ℕ := 9
  let mulching_hours : ℕ := 10
  let total_earnings : ℕ := 
    mowing_rate * mowing_hours + 
    weeding_rate * weeding_hours + 
    mulching_rate * mulching_hours
  total_earnings = 567 := by
sorry

end NUMINAMATH_CALUDE_grace_september_earnings_l383_38389


namespace NUMINAMATH_CALUDE_union_complement_problem_l383_38325

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, -1, 2}

theorem union_complement_problem : A ∪ (U \ B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_problem_l383_38325


namespace NUMINAMATH_CALUDE_largest_difference_l383_38301

def A : ℕ := 3 * 1003^1004
def B : ℕ := 1003^1004
def C : ℕ := 1002 * 1003^1003
def D : ℕ := 3 * 1003^1003
def E : ℕ := 1003^1003
def F : ℕ := 1003^1002

def P : ℕ := A - B
def Q : ℕ := B - C
def R : ℕ := C - D
def S : ℕ := D - E
def T : ℕ := E - F

theorem largest_difference :
  P > max Q (max R (max S T)) := by sorry

end NUMINAMATH_CALUDE_largest_difference_l383_38301


namespace NUMINAMATH_CALUDE_no_natural_squares_l383_38358

theorem no_natural_squares (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ y^2 + x = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_squares_l383_38358


namespace NUMINAMATH_CALUDE_planting_schemes_count_l383_38372

/-- The number of seed types available -/
def total_seeds : ℕ := 5

/-- The number of plots to be planted -/
def plots : ℕ := 4

/-- The number of choices for the first plot -/
def first_plot_choices : ℕ := 2

/-- Calculates the number of permutations of r items chosen from n items -/
def permutations (n r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

/-- The main theorem stating the total number of planting schemes -/
theorem planting_schemes_count : 
  first_plot_choices * permutations (total_seeds - 1) (plots - 1) = 48 := by
  sorry

end NUMINAMATH_CALUDE_planting_schemes_count_l383_38372


namespace NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l383_38352

/-- Represents a wheel with sections labeled as even or odd numbers -/
structure Wheel where
  total_sections : ℕ
  even_sections : ℕ
  odd_sections : ℕ
  sections_sum : even_sections + odd_sections = total_sections

/-- Calculates the probability of getting an even sum when spinning two wheels -/
def probability_even_sum (wheel1 wheel2 : Wheel) : ℚ :=
  let p_even1 := wheel1.even_sections / wheel1.total_sections
  let p_odd1 := wheel1.odd_sections / wheel1.total_sections
  let p_even2 := wheel2.even_sections / wheel2.total_sections
  let p_odd2 := wheel2.odd_sections / wheel2.total_sections
  (p_even1 * p_even2) + (p_odd1 * p_odd2)

theorem probability_even_sum_two_wheels :
  let wheel1 : Wheel := ⟨3, 2, 1, by simp⟩
  let wheel2 : Wheel := ⟨5, 3, 2, by simp⟩
  probability_even_sum wheel1 wheel2 = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l383_38352


namespace NUMINAMATH_CALUDE_g_512_minus_g_256_eq_zero_l383_38351

-- Define σ(n) as the sum of all positive divisors of n
def σ (n : ℕ+) : ℕ := sorry

-- Define g(n) = 2σ(n)/n
def g (n : ℕ+) : ℚ := 2 * (σ n) / n

-- Theorem statement
theorem g_512_minus_g_256_eq_zero : g 512 - g 256 = 0 := by sorry

end NUMINAMATH_CALUDE_g_512_minus_g_256_eq_zero_l383_38351


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l383_38396

theorem absolute_value_equation_solutions :
  ∀ x : ℝ, |x - 3| = 5 - 2*x ↔ x = 2 ∨ x = 8/3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l383_38396


namespace NUMINAMATH_CALUDE_xy_value_l383_38386

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l383_38386


namespace NUMINAMATH_CALUDE_trapezoid_height_l383_38380

-- Define the trapezoid properties
structure IsoscelesTrapezoid where
  diagonal : ℝ
  area : ℝ

-- Define the theorem
theorem trapezoid_height (t : IsoscelesTrapezoid) (h_diagonal : t.diagonal = 10) (h_area : t.area = 48) :
  ∃ (height : ℝ), (height = 6 ∨ height = 8) ∧ 
  (∃ (base_avg : ℝ), base_avg * height = t.area ∧ base_avg^2 + height^2 = t.diagonal^2) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_height_l383_38380


namespace NUMINAMATH_CALUDE_find_n_l383_38359

theorem find_n : ∃ n : ℕ, 2^n = 2 * 16^2 * 4^3 ∧ n = 15 := by sorry

end NUMINAMATH_CALUDE_find_n_l383_38359


namespace NUMINAMATH_CALUDE_ratio_y_over_x_is_six_l383_38370

theorem ratio_y_over_x_is_six (x y : ℝ) 
  (h1 : Real.sqrt (3 * x) * (1 + 1 / (x + y)) = 2)
  (h2 : Real.sqrt (7 * y) * (1 - 1 / (x + y)) = 4 * Real.sqrt 2) :
  y / x = 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_y_over_x_is_six_l383_38370


namespace NUMINAMATH_CALUDE_function_value_2024_l383_38369

def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem function_value_2024 (a b c : ℝ) 
  (h2021 : f a b c 2021 = 2021)
  (h2022 : f a b c 2022 = 2022)
  (h2023 : f a b c 2023 = 2023) :
  f a b c 2024 = 2030 := by
  sorry

end NUMINAMATH_CALUDE_function_value_2024_l383_38369


namespace NUMINAMATH_CALUDE_survey_result_l383_38399

theorem survey_result (total : ℕ) (tv_dislike_percent : ℚ) (both_dislike_percent : ℚ)
  (h_total : total = 1800)
  (h_tv_dislike : tv_dislike_percent = 40 / 100)
  (h_both_dislike : both_dislike_percent = 25 / 100) :
  ↑⌊tv_dislike_percent * both_dislike_percent * total⌋ = 180 :=
by sorry

end NUMINAMATH_CALUDE_survey_result_l383_38399


namespace NUMINAMATH_CALUDE_difference_of_squares_l383_38335

theorem difference_of_squares (m n : ℝ) : m^2 - 4*n^2 = (m + 2*n) * (m - 2*n) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l383_38335


namespace NUMINAMATH_CALUDE_lines_coplanar_conditions_l383_38381

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a placeholder and should be replaced with actual representation
  dummy : Unit

-- Define what it means for three lines to be coplanar
def coplanar (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Add the actual definition here

-- Define the condition of three lines intersecting pairwise but not sharing a common point
def intersect_pairwise_no_common_point (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Add the actual definition here

-- Define the condition of two lines being parallel and the third intersecting both
def two_parallel_one_intersecting (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Add the actual definition here

-- The main theorem
theorem lines_coplanar_conditions (l1 l2 l3 : Line3D) :
  (intersect_pairwise_no_common_point l1 l2 l3 ∨ two_parallel_one_intersecting l1 l2 l3) →
  coplanar l1 l2 l3 := by
  sorry


end NUMINAMATH_CALUDE_lines_coplanar_conditions_l383_38381


namespace NUMINAMATH_CALUDE_proposition_logic_l383_38311

theorem proposition_logic (p q : Prop) (hp : p ↔ (3 ≥ 3)) (hq : q ↔ (3 > 4)) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_proposition_logic_l383_38311


namespace NUMINAMATH_CALUDE_equal_abc_l383_38365

theorem equal_abc (a b c x : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (x * b + (1 - x) * c) / a = (x * c + (1 - x) * a) / b)
  (h5 : (x * b + (1 - x) * c) / a = (x * a + (1 - x) * b) / c) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_abc_l383_38365


namespace NUMINAMATH_CALUDE_nine_circles_problem_l383_38317

/-- Represents a 3x3 grid of numbers -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if all numbers from 1 to 9 are used exactly once in the grid -/
def is_valid_grid (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), g i j = n

/-- Represents a triangle in the grid by its three vertex coordinates -/
structure Triangle where
  v1 : Fin 3 × Fin 3
  v2 : Fin 3 × Fin 3
  v3 : Fin 3 × Fin 3

/-- List of all 7 triangles in the grid -/
def triangles : List Triangle := sorry

/-- Checks if the sum of numbers at the vertices of a triangle is 15 -/
def triangle_sum_is_15 (g : Grid) (t : Triangle) : Prop :=
  (g t.v1.1 t.v1.2).val + (g t.v2.1 t.v2.2).val + (g t.v3.1 t.v3.2).val = 15

/-- The main theorem: there exists a valid grid where all triangles sum to 15 -/
theorem nine_circles_problem :
  ∃ (g : Grid), is_valid_grid g ∧ ∀ t ∈ triangles, triangle_sum_is_15 g t :=
sorry

end NUMINAMATH_CALUDE_nine_circles_problem_l383_38317


namespace NUMINAMATH_CALUDE_towel_area_decrease_l383_38305

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let original_area := L * B
  let new_length := 0.8 * L
  let new_breadth := 0.8 * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l383_38305


namespace NUMINAMATH_CALUDE_negation_of_negative_square_positive_is_false_l383_38323

theorem negation_of_negative_square_positive_is_false : 
  ¬(∀ x : ℝ, x < 0 → x^2 > 0) = False := by sorry

end NUMINAMATH_CALUDE_negation_of_negative_square_positive_is_false_l383_38323


namespace NUMINAMATH_CALUDE_problem_solution_l383_38345

theorem problem_solution : (12 : ℝ)^2 * 6^3 / 432 = 72 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l383_38345


namespace NUMINAMATH_CALUDE_soap_packages_per_box_l383_38385

theorem soap_packages_per_box (soaps_per_package : ℕ) (num_boxes : ℕ) (total_soaps : ℕ) :
  soaps_per_package = 192 →
  num_boxes = 2 →
  total_soaps = 2304 →
  ∃ (packages_per_box : ℕ), 
    packages_per_box * soaps_per_package * num_boxes = total_soaps ∧
    packages_per_box = 6 :=
by sorry

end NUMINAMATH_CALUDE_soap_packages_per_box_l383_38385


namespace NUMINAMATH_CALUDE_ellipse_properties_l383_38300

noncomputable def ellipse_C (x y a b : ℝ) : Prop :=
  (y^2 / a^2) + (x^2 / b^2) = 1

theorem ellipse_properties (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0)
  (h3 : (a^2 - b^2) / a^2 = 6/9)
  (h4 : ellipse_C (2*Real.sqrt 2/3) (Real.sqrt 3/3) a b) :
  (∃ (x y : ℝ), ellipse_C x y 1 (Real.sqrt 3)) ∧
  (∃ (S : ℝ → ℝ → ℝ), 
    (∀ A B : ℝ × ℝ, 
      ellipse_C A.1 A.2 1 (Real.sqrt 3) → 
      ellipse_C B.1 B.2 1 (Real.sqrt 3) → 
      (∃ m : ℝ, A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2) → 
      S A.1 A.2 ≤ Real.sqrt 3 / 2) ∧
    (∃ A B : ℝ × ℝ, 
      ellipse_C A.1 A.2 1 (Real.sqrt 3) ∧ 
      ellipse_C B.1 B.2 1 (Real.sqrt 3) ∧ 
      (∃ m : ℝ, A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2) ∧ 
      S A.1 A.2 = Real.sqrt 3 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l383_38300


namespace NUMINAMATH_CALUDE_model_M_completion_time_l383_38313

/-- The time (in minutes) it takes for a model M computer to complete the task -/
def model_M_time : ℝ := 24

/-- The time (in minutes) it takes for a model N computer to complete the task -/
def model_N_time : ℝ := 12

/-- The number of each model of computer used -/
def num_computers : ℕ := 8

/-- The time (in minutes) it takes for the combined computers to complete the task -/
def combined_time : ℝ := 1

theorem model_M_completion_time :
  (num_computers : ℝ) / model_M_time + (num_computers : ℝ) / model_N_time = 1 / combined_time :=
sorry

end NUMINAMATH_CALUDE_model_M_completion_time_l383_38313


namespace NUMINAMATH_CALUDE_strawberry_candies_count_candy_problem_l383_38360

theorem strawberry_candies_count : ℕ → ℕ → Prop :=
  fun total grape_diff =>
    ∀ (strawberry grape : ℕ),
      strawberry + grape = total →
      grape = strawberry - grape_diff →
      strawberry = 121

theorem candy_problem : strawberry_candies_count 240 2 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_candies_count_candy_problem_l383_38360


namespace NUMINAMATH_CALUDE_tim_placed_three_pencils_l383_38307

/-- Given that there were initially 2 pencils in a drawer and after Tim placed some pencils
    there are now 5 pencils in total, prove that Tim placed 3 pencils in the drawer. -/
theorem tim_placed_three_pencils (initial_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : initial_pencils = 2) 
  (h2 : total_pencils = 5) :
  total_pencils - initial_pencils = 3 := by
  sorry

end NUMINAMATH_CALUDE_tim_placed_three_pencils_l383_38307


namespace NUMINAMATH_CALUDE_square_of_divisibility_l383_38376

theorem square_of_divisibility (m n : ℤ) 
  (h1 : m ≠ 0) 
  (h2 : n ≠ 0) 
  (h3 : m % 2 = n % 2) 
  (h4 : (n^2 - 1) % (m^2 - n^2 + 1) = 0) : 
  ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := by
sorry

end NUMINAMATH_CALUDE_square_of_divisibility_l383_38376


namespace NUMINAMATH_CALUDE_package_cost_l383_38374

/-- The cost to mail each package, given the total amount spent, cost per letter, number of letters, and relationship between letters and packages. -/
theorem package_cost (total_spent : ℚ) (letter_cost : ℚ) (num_letters : ℕ) 
  (h1 : total_spent = 4.49)
  (h2 : letter_cost = 0.37)
  (h3 : num_letters = 5)
  (h4 : num_letters = num_packages + 2) : 
  (total_spent - num_letters * letter_cost) / (num_letters - 2) = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_package_cost_l383_38374


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l383_38375

/-- A quadratic function with vertex (-3, 2) passing through (2, -43) has a = -9/5 --/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x + 3)^2 + 2) →  -- vertex form
  (a * 2^2 + b * 2 + c = -43) →                       -- passes through (2, -43)
  a = -9/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l383_38375


namespace NUMINAMATH_CALUDE_sam_sticker_spending_l383_38302

/-- Given Sam's initial penny count and his spending on toys and candy, 
    calculate the amount spent on stickers. -/
theorem sam_sticker_spending 
  (total : ℕ) 
  (toy_cost : ℕ) 
  (candy_cost : ℕ) 
  (h1 : total = 2476) 
  (h2 : toy_cost = 1145) 
  (h3 : candy_cost = 781) :
  total - (toy_cost + candy_cost) = 550 := by
  sorry

#check sam_sticker_spending

end NUMINAMATH_CALUDE_sam_sticker_spending_l383_38302


namespace NUMINAMATH_CALUDE_square_sum_equals_fifty_l383_38395

theorem square_sum_equals_fifty (x y : ℝ) 
  (h1 : x + y = -10) 
  (h2 : x = 25 / y) : 
  x^2 + y^2 = 50 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_fifty_l383_38395


namespace NUMINAMATH_CALUDE_identity_unique_l383_38315

-- Define a group structure
class MyGroup (G : Type) where
  mul : G → G → G
  one : G
  inv : G → G
  mul_assoc : ∀ a b c : G, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a : G, mul one a = a
  mul_one : ∀ a : G, mul a one = a
  mul_left_inv : ∀ a : G, mul (inv a) a = one

-- State the theorem
theorem identity_unique {G : Type} [MyGroup G] (e e' : G)
    (h1 : ∀ g : G, MyGroup.mul e g = g ∧ MyGroup.mul g e = g)
    (h2 : ∀ g : G, MyGroup.mul e' g = g ∧ MyGroup.mul g e' = g) :
    e = e' := by sorry

end NUMINAMATH_CALUDE_identity_unique_l383_38315


namespace NUMINAMATH_CALUDE_inequality_proof_l383_38362

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l383_38362


namespace NUMINAMATH_CALUDE_min_value_of_f_l383_38392

/-- Given a function f(x) = (a + x^2) / x, where a > 0 and x ∈ (0, b),
    prove that the minimum value of f(x) is 2√a when b > √a. -/
theorem min_value_of_f (a b : ℝ) (ha : a > 0) (hb : b > Real.sqrt a) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt a ∧
    ∀ x ∈ Set.Ioo 0 b, (a + x^2) / x ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l383_38392


namespace NUMINAMATH_CALUDE_nested_root_simplification_l383_38397

theorem nested_root_simplification (b : ℝ) (h : b > 0) :
  (((b^16)^(1/8))^(1/4))^3 * (((b^16)^(1/4))^(1/8))^3 = b^3 := by sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l383_38397


namespace NUMINAMATH_CALUDE_base_five_digits_of_1297_l383_38310

theorem base_five_digits_of_1297 : ∃ n : ℕ, n > 0 ∧ 5^(n-1) ≤ 1297 ∧ 1297 < 5^n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_five_digits_of_1297_l383_38310


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l383_38338

/-- The x-intercept of the line 2x + 3y + 6 = 0 is -3 -/
theorem x_intercept_of_line (x y : ℝ) :
  2 * x + 3 * y + 6 = 0 → y = 0 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l383_38338


namespace NUMINAMATH_CALUDE_fourth_side_distance_l383_38387

/-- Given a square and a point inside it, if the distances from the point to three sides are 4, 7, and 12,
    then the distance to the fourth side is either 9 or 15. -/
theorem fourth_side_distance (s : ℝ) (d1 d2 d3 d4 : ℝ) : 
  s > 0 ∧ d1 = 4 ∧ d2 = 7 ∧ d3 = 12 ∧ 
  d1 + d2 + d3 + d4 = s → 
  d4 = 9 ∨ d4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_distance_l383_38387


namespace NUMINAMATH_CALUDE_blouse_price_proof_l383_38332

/-- The original price of a blouse before discount -/
def original_price : ℝ := 180

/-- The discount percentage applied to the blouse -/
def discount_percentage : ℝ := 18

/-- The price paid after applying the discount -/
def discounted_price : ℝ := 147.60

/-- Theorem stating that the original price is correct given the discount and discounted price -/
theorem blouse_price_proof : 
  original_price * (1 - discount_percentage / 100) = discounted_price := by
  sorry

end NUMINAMATH_CALUDE_blouse_price_proof_l383_38332


namespace NUMINAMATH_CALUDE_marble_problem_l383_38355

theorem marble_problem (a : ℚ) : 
  let brian := 3 * a - 4
  let caden := 2 * brian + 2
  let daryl := 4 * caden
  a + brian + caden + daryl = 122 → a = 78 / 17 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l383_38355


namespace NUMINAMATH_CALUDE_max_leftover_grapes_l383_38378

theorem max_leftover_grapes (n : ℕ) : ∃ (k : ℕ), n = 7 * k + (n % 7) ∧ n % 7 ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_grapes_l383_38378


namespace NUMINAMATH_CALUDE_f_max_at_neg_two_l383_38384

def f (x : ℝ) : ℝ := x^3 - 12*x

theorem f_max_at_neg_two :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≤ f m :=
sorry

end NUMINAMATH_CALUDE_f_max_at_neg_two_l383_38384


namespace NUMINAMATH_CALUDE_space_between_apple_trees_is_12_l383_38308

/-- The space needed between apple trees in Quinton's backyard --/
def space_between_apple_trees : ℝ :=
  let total_space : ℝ := 71
  let apple_tree_width : ℝ := 10
  let peach_tree_width : ℝ := 12
  let space_between_peach_trees : ℝ := 15
  let num_apple_trees : ℕ := 2
  let num_peach_trees : ℕ := 2
  let peach_trees_space : ℝ := num_peach_trees * peach_tree_width + space_between_peach_trees
  let apple_trees_space : ℝ := total_space - peach_trees_space
  apple_trees_space - (num_apple_trees * apple_tree_width)

theorem space_between_apple_trees_is_12 :
  space_between_apple_trees = 12 := by
  sorry

end NUMINAMATH_CALUDE_space_between_apple_trees_is_12_l383_38308


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l383_38371

theorem complex_fraction_equals_i (i : ℂ) (hi : i^2 = -1) :
  (2 + i) / (1 - 2*i) = i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l383_38371


namespace NUMINAMATH_CALUDE_solve_table_height_l383_38304

def table_height_problem (initial_measurement : ℝ) (rearranged_measurement : ℝ) 
  (block_width : ℝ) (table_thickness : ℝ) : Prop :=
  ∃ (h l : ℝ),
    l + h - block_width + table_thickness = initial_measurement ∧
    block_width + h - l + table_thickness = rearranged_measurement ∧
    h = 33

theorem solve_table_height :
  table_height_problem 40 34 6 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_table_height_l383_38304


namespace NUMINAMATH_CALUDE_min_cans_proof_l383_38322

/-- The capacity of a special edition soda can in ounces -/
def can_capacity : ℕ := 15

/-- Half a gallon in ounces -/
def half_gallon : ℕ := 64

/-- The minimum number of cans needed to provide at least half a gallon of soda -/
def min_cans : ℕ := 5

theorem min_cans_proof :
  (∀ n : ℕ, n * can_capacity ≥ half_gallon → n ≥ min_cans) ∧
  (min_cans * can_capacity ≥ half_gallon) :=
sorry

end NUMINAMATH_CALUDE_min_cans_proof_l383_38322


namespace NUMINAMATH_CALUDE_remaining_payment_l383_38373

/-- Given a 10% deposit of $80, prove that the remaining amount to be paid is $720 -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (total_price : ℝ) : 
  deposit = 80 →
  deposit_percentage = 0.1 →
  deposit = deposit_percentage * total_price →
  total_price - deposit = 720 := by
sorry

end NUMINAMATH_CALUDE_remaining_payment_l383_38373


namespace NUMINAMATH_CALUDE_speed_ratio_is_two_to_one_l383_38353

-- Define the given constants
def distance : ℝ := 30
def original_speed : ℝ := 5

-- Define Sameer's speed as a variable
variable (sameer_speed : ℝ)

-- Define Abhay's new speed as a variable
variable (new_speed : ℝ)

-- Define the conditions
def condition1 (sameer_speed : ℝ) : Prop :=
  distance / original_speed = distance / sameer_speed + 2

def condition2 (sameer_speed new_speed : ℝ) : Prop :=
  distance / new_speed = distance / sameer_speed - 1

-- Theorem to prove
theorem speed_ratio_is_two_to_one 
  (h1 : condition1 sameer_speed)
  (h2 : condition2 sameer_speed new_speed) :
  new_speed / original_speed = 2 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_is_two_to_one_l383_38353


namespace NUMINAMATH_CALUDE_sock_profit_calculation_l383_38368

/-- Calculates the total profit from selling socks given specific conditions. -/
theorem sock_profit_calculation : 
  let total_pairs : ℕ := 9
  let cost_per_pair : ℚ := 2
  let purchase_discount : ℚ := 0.1
  let profit_percentage_4_pairs : ℚ := 0.25
  let profit_per_pair_5_pairs : ℚ := 0.2
  let sales_tax : ℚ := 0.05

  let discounted_cost := total_pairs * cost_per_pair * (1 - purchase_discount)
  let selling_price_4_pairs := 4 * cost_per_pair * (1 + profit_percentage_4_pairs)
  let selling_price_5_pairs := 5 * cost_per_pair + 5 * profit_per_pair_5_pairs
  let total_selling_price := (selling_price_4_pairs + selling_price_5_pairs) * (1 + sales_tax)
  let total_profit := total_selling_price - discounted_cost

  total_profit = 5.85 := by sorry

end NUMINAMATH_CALUDE_sock_profit_calculation_l383_38368


namespace NUMINAMATH_CALUDE_expected_total_rain_l383_38324

/-- Represents the possible rain outcomes for a day --/
inductive RainOutcome
  | NoRain
  | ThreeInches
  | EightInches

/-- Probability of each rain outcome --/
def rainProbability (outcome : RainOutcome) : ℝ :=
  match outcome with
  | RainOutcome.NoRain => 0.5
  | RainOutcome.ThreeInches => 0.3
  | RainOutcome.EightInches => 0.2

/-- Amount of rain for each outcome in inches --/
def rainAmount (outcome : RainOutcome) : ℝ :=
  match outcome with
  | RainOutcome.NoRain => 0
  | RainOutcome.ThreeInches => 3
  | RainOutcome.EightInches => 8

/-- Number of days in the forecast --/
def forecastDays : ℕ := 5

/-- Expected value of rain for a single day --/
def dailyExpectedRain : ℝ :=
  (rainProbability RainOutcome.NoRain * rainAmount RainOutcome.NoRain) +
  (rainProbability RainOutcome.ThreeInches * rainAmount RainOutcome.ThreeInches) +
  (rainProbability RainOutcome.EightInches * rainAmount RainOutcome.EightInches)

/-- Theorem: The expected value of the total amount of rain for the forecast period is 12.5 inches --/
theorem expected_total_rain :
  forecastDays * dailyExpectedRain = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_expected_total_rain_l383_38324


namespace NUMINAMATH_CALUDE_coeff_x4_is_zero_l383_38327

/-- The coefficient of x^4 in the expansion of (x+2)(x-1)^5 -/
def coeff_x4 (x : ℝ) : ℝ :=
  let expansion := (x + 2) * (x - 1)^5
  sorry

theorem coeff_x4_is_zero :
  coeff_x4 x = 0 := by sorry

end NUMINAMATH_CALUDE_coeff_x4_is_zero_l383_38327


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l383_38319

theorem smallest_angle_solution (x : ℝ) : 
  (0 < x) → 
  (∀ y : ℝ, 0 < y → 
    Real.tan (8 * π / 180 * y) = (Real.cos (π / 180 * y) - Real.sin (π / 180 * y)) / (Real.cos (π / 180 * y) + Real.sin (π / 180 * y)) → 
    x ≤ y) → 
  Real.tan (8 * π / 180 * x) = (Real.cos (π / 180 * x) - Real.sin (π / 180 * x)) / (Real.cos (π / 180 * x) + Real.sin (π / 180 * x)) → 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l383_38319


namespace NUMINAMATH_CALUDE_expression_equality_l383_38331

theorem expression_equality (x y z : ℝ) : (x + (y + z)) - ((x + z) + y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l383_38331


namespace NUMINAMATH_CALUDE_square_sum_given_condition_l383_38340

theorem square_sum_given_condition (x y : ℝ) :
  (x - 3)^2 + |2 * y + 1| = 0 → x^2 + y^2 = 9 + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_condition_l383_38340


namespace NUMINAMATH_CALUDE_joey_studies_five_nights_per_week_l383_38336

/-- Represents Joey's study schedule and calculates the number of weekday study nights per week -/
def joeys_study_schedule (weekday_hours_per_night : ℕ) (weekend_hours_per_day : ℕ) 
  (total_weeks : ℕ) (total_study_hours : ℕ) : ℕ :=
  let weekend_days := 2 * total_weeks
  let weekend_hours := weekend_hours_per_day * weekend_days
  let weekday_hours := total_study_hours - weekend_hours
  let weekday_nights := weekday_hours / weekday_hours_per_night
  weekday_nights / total_weeks

/-- Theorem stating that Joey studies 5 nights per week on weekdays -/
theorem joey_studies_five_nights_per_week :
  joeys_study_schedule 2 3 6 96 = 5 := by
  sorry

end NUMINAMATH_CALUDE_joey_studies_five_nights_per_week_l383_38336


namespace NUMINAMATH_CALUDE_fifteen_hundredth_day_is_wednesday_l383_38314

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
  | Monday : DayOfWeek
  | Tuesday : DayOfWeek
  | Wednesday : DayOfWeek
  | Thursday : DayOfWeek
  | Friday : DayOfWeek
  | Saturday : DayOfWeek
  | Sunday : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to get the day of the week after n days -/
def dayAfter (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (dayAfter start m)

theorem fifteen_hundredth_day_is_wednesday :
  dayAfter DayOfWeek.Monday 1499 = DayOfWeek.Wednesday :=
by
  sorry


end NUMINAMATH_CALUDE_fifteen_hundredth_day_is_wednesday_l383_38314


namespace NUMINAMATH_CALUDE_max_surface_area_of_stacked_solids_l383_38339

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surface_area (d : Dimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Calculates the surface area of two stacked rectangular solids -/
def stacked_surface_area (d : Dimensions) (overlap_dim1 overlap_dim2 : ℝ) : ℝ :=
  2 * (overlap_dim1 * overlap_dim2) + 
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- The dimensions of the given rectangular solids -/
def solid_dimensions : Dimensions :=
  { length := 5, width := 4, height := 3 }

theorem max_surface_area_of_stacked_solids :
  let d := solid_dimensions
  let sa1 := stacked_surface_area d d.length d.width
  let sa2 := stacked_surface_area d d.length d.height
  let sa3 := stacked_surface_area d d.width d.height
  max sa1 (max sa2 sa3) = 164 := by sorry

end NUMINAMATH_CALUDE_max_surface_area_of_stacked_solids_l383_38339


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l383_38320

open Set

-- Define sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 3}
def B : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -3 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l383_38320


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l383_38316

theorem partial_fraction_decomposition :
  let C : ℚ := 81 / 16
  let D : ℚ := -49 / 16
  ∀ x : ℚ, x ≠ 12 → x ≠ -4 →
    (7 * x - 3) / (x^2 - 8*x - 48) = C / (x - 12) + D / (x + 4) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l383_38316


namespace NUMINAMATH_CALUDE_max_a_value_l383_38349

-- Define the quadratic polynomial
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem max_a_value :
  (∃ (a_max : ℝ), ∀ (a b : ℝ),
    (∀ (x : ℝ), ∃ (y : ℝ), f a b y = f a b x + y) →
    a ≤ a_max ∧
    (∃ (b : ℝ), ∀ (x : ℝ), ∃ (y : ℝ), f a_max b y = f a_max b x + y)) ∧
  (∀ (a_greater : ℝ),
    (∃ (a b : ℝ), a > a_greater ∧
      (∀ (x : ℝ), ∃ (y : ℝ), f a b y = f a b x + y)) →
    a_greater < 1/2) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l383_38349


namespace NUMINAMATH_CALUDE_boys_running_speed_l383_38357

theorem boys_running_speed (side_length : ℝ) (time : ℝ) (speed : ℝ) : 
  side_length = 55 →
  time = 88 →
  speed = (4 * side_length / time) * 3.6 →
  speed = 9 := by sorry

end NUMINAMATH_CALUDE_boys_running_speed_l383_38357
