import Mathlib

namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l727_72721

/-- The amount spent on shorts -/
def shorts_cost : ℚ := 1428 / 100

/-- The total amount spent on clothing -/
def total_cost : ℚ := 1902 / 100

/-- The amount spent on the jacket -/
def jacket_cost : ℚ := total_cost - shorts_cost

theorem jacket_cost_calculation : jacket_cost = 474 / 100 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l727_72721


namespace NUMINAMATH_CALUDE_total_candies_l727_72741

/-- The total number of candies for six people given specific relationships between their candy counts. -/
theorem total_candies (adam james rubert lisa chris emily : ℕ) : 
  adam = 6 ∧ 
  james = 3 * adam ∧ 
  rubert = 4 * james ∧ 
  lisa = 2 * rubert ∧ 
  chris = lisa + 5 ∧ 
  emily = 3 * chris - 7 → 
  adam + james + rubert + lisa + chris + emily = 829 := by
  sorry

#eval 6 + 3 * 6 + 4 * (3 * 6) + 2 * (4 * (3 * 6)) + (2 * (4 * (3 * 6)) + 5) + (3 * (2 * (4 * (3 * 6)) + 5) - 7)

end NUMINAMATH_CALUDE_total_candies_l727_72741


namespace NUMINAMATH_CALUDE_normal_pumping_rate_l727_72778

/-- Proves that given a pond with a capacity of 200 gallons, filled in 50 minutes at 2/3 of the normal pumping rate, the normal pumping rate is 6 gallons per minute. -/
theorem normal_pumping_rate (pond_capacity : ℝ) (filling_time : ℝ) (restriction_factor : ℝ) :
  pond_capacity = 200 →
  filling_time = 50 →
  restriction_factor = 2/3 →
  (restriction_factor * (pond_capacity / filling_time)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_normal_pumping_rate_l727_72778


namespace NUMINAMATH_CALUDE_max_colored_cells_4x3000_exists_optimal_board_l727_72710

/-- A tetromino is a geometric shape composed of four square cells connected orthogonally. -/
def Tetromino : Type := Unit

/-- A board is represented as a 2D array of boolean values, where true represents a colored cell. -/
def Board : Type := Array (Array Bool)

/-- Check if a given board contains a tetromino. -/
def containsTetromino (board : Board) : Bool :=
  sorry

/-- Count the number of colored cells in a board. -/
def countColoredCells (board : Board) : Nat :=
  sorry

/-- Create a 4 × 3000 board. -/
def create4x3000Board : Board :=
  sorry

/-- The main theorem stating the maximum number of cells that can be colored. -/
theorem max_colored_cells_4x3000 :
  ∀ (board : Board),
    board = create4x3000Board →
    ¬containsTetromino board →
    countColoredCells board ≤ 7000 :=
  sorry

/-- The existence of a board with exactly 7000 colored cells and no tetromino. -/
theorem exists_optimal_board :
  ∃ (board : Board),
    board = create4x3000Board ∧
    ¬containsTetromino board ∧
    countColoredCells board = 7000 :=
  sorry

end NUMINAMATH_CALUDE_max_colored_cells_4x3000_exists_optimal_board_l727_72710


namespace NUMINAMATH_CALUDE_lake_crossing_time_difference_l727_72761

theorem lake_crossing_time_difference 
  (lake_width : ℝ) 
  (janet_speed : ℝ) 
  (sister_speed : ℝ) 
  (h1 : lake_width = 60) 
  (h2 : janet_speed = 30) 
  (h3 : sister_speed = 12) : 
  (lake_width / sister_speed) - (lake_width / janet_speed) = 3 := by
sorry

end NUMINAMATH_CALUDE_lake_crossing_time_difference_l727_72761


namespace NUMINAMATH_CALUDE_largest_t_value_for_temperature_l727_72706

theorem largest_t_value_for_temperature (t : ℝ) :
  let f : ℝ → ℝ := λ x => -x^2 + 10*x + 60
  let solutions := {x : ℝ | f x = 80}
  ∃ max_t ∈ solutions, ∀ t ∈ solutions, t ≤ max_t ∧ max_t = 5 + 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_t_value_for_temperature_l727_72706


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l727_72701

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (10*x + 2) * (10*x + 6) * (5*x + 5) = 960 * k) ∧
  (∀ (m : ℤ), m > 960 → ¬(∀ (y : ℤ), Odd y → ∃ (l : ℤ), (10*y + 2) * (10*y + 6) * (5*y + 5) = m * l)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l727_72701


namespace NUMINAMATH_CALUDE_age_sum_proof_l727_72754

theorem age_sum_proof (asaf_age : ℕ) (alexander_age : ℕ) 
  (asaf_pencils : ℕ) (alexander_pencils : ℕ) : 
  asaf_age = 50 →
  asaf_age - alexander_age = asaf_pencils / 2 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 →
  asaf_age + alexander_age = 60 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l727_72754


namespace NUMINAMATH_CALUDE_triangle_max_area_l727_72748

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C,
    if (a-b+c)/c = b/(a+b-c) and a = 2, then the maximum area of triangle ABC is √3. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  (a - b + c) / c = b / (a + b - c) →
  a = 2 →
  ∃ (S : ℝ), S ≤ Real.sqrt 3 ∧ 
    (∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) ∧
    (∃ (b' c' : ℝ), (1/2) * b' * c' * Real.sin A = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l727_72748


namespace NUMINAMATH_CALUDE_inequality_proof_l727_72729

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) + Real.sqrt (c^2 - c*a + a^2) + 9 * (a*b*c)^(1/3) ≤ 4*(a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l727_72729


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_difference_l727_72767

theorem circle_radius_from_area_circumference_difference 
  (x y : ℝ) (h : x - y = 72 * Real.pi) : ∃ r : ℝ, r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 12 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_difference_l727_72767


namespace NUMINAMATH_CALUDE_area_triangle_PQR_l727_72789

/-- Square pyramid with given dimensions and points --/
structure SquarePyramid where
  baseSide : ℝ
  altitude : ℝ
  P : ℝ  -- Distance from W to P along WO
  Q : ℝ  -- Distance from Y to Q along YO
  R : ℝ  -- Distance from X to R along XO

/-- Theorem stating the area of triangle PQR in the given square pyramid --/
theorem area_triangle_PQR (pyramid : SquarePyramid)
  (h1 : pyramid.baseSide = 4)
  (h2 : pyramid.altitude = 8)
  (h3 : pyramid.P = 1/4 * (pyramid.baseSide * Real.sqrt 2 / 2))
  (h4 : pyramid.Q = 1/2 * (pyramid.baseSide * Real.sqrt 2 / 2))
  (h5 : pyramid.R = 3/4 * (pyramid.baseSide * Real.sqrt 2 / 2)) :
  let WO := Real.sqrt ((pyramid.baseSide * Real.sqrt 2 / 2)^2 + pyramid.altitude^2)
  let PQ := pyramid.Q - pyramid.P
  let RQ := pyramid.R - pyramid.Q
  1/2 * PQ * RQ = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_PQR_l727_72789


namespace NUMINAMATH_CALUDE_leyden_quadruple_theorem_l727_72731

/-- Definition of a Leyden quadruple -/
structure LeydenQuadruple where
  p : ℕ
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ

/-- The main theorem about Leyden quadruples -/
theorem leyden_quadruple_theorem (q : LeydenQuadruple) :
  (q.a₁ + q.a₂ + q.a₃) / 3 = q.p + 2 ↔ q.p = 5 := by
  sorry

end NUMINAMATH_CALUDE_leyden_quadruple_theorem_l727_72731


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l727_72790

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    left vertex A, top vertex B, right focus F, and midpoint M of AB,
    prove that the eccentricity e is in the range (0, -1+√3] 
    if 2⋅MA⋅MF + |BF|² ≥ 0 -/
theorem ellipse_eccentricity_range (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * (a/2 * (c + a/2) + b/2 * (-b/2)) + (b^2 + c^2) ≥ 0) :
  let e := c / a
  ∃ (e : ℝ), 0 < e ∧ e ≤ -1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l727_72790


namespace NUMINAMATH_CALUDE_profit_sharing_l727_72764

/-- The profit sharing problem -/
theorem profit_sharing
  (mary_investment mike_investment : ℚ)
  (equal_share_ratio investment_share_ratio : ℚ)
  (mary_extra : ℚ)
  (h1 : mary_investment = 650)
  (h2 : mike_investment = 350)
  (h3 : equal_share_ratio = 1/3)
  (h4 : investment_share_ratio = 2/3)
  (h5 : mary_extra = 600)
  : ∃ P : ℚ,
    P / 6 + (mary_investment / (mary_investment + mike_investment)) * (2 * P / 3) -
    (P / 6 + (mike_investment / (mary_investment + mike_investment)) * (2 * P / 3)) = mary_extra ∧
    P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_l727_72764


namespace NUMINAMATH_CALUDE_train_length_l727_72713

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmh = 108 → time_sec = 50 → length = (speed_kmh * (5/18)) * time_sec → length = 1500 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l727_72713


namespace NUMINAMATH_CALUDE_jake_coffee_drop_probability_l727_72791

theorem jake_coffee_drop_probability 
  (trip_probability : ℝ) 
  (not_drop_probability : ℝ) 
  (h1 : trip_probability = 0.4)
  (h2 : not_drop_probability = 0.9) :
  1 - not_drop_probability = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_jake_coffee_drop_probability_l727_72791


namespace NUMINAMATH_CALUDE_card_relationship_l727_72758

theorem card_relationship (c : ℝ) (h1 : c > 0) : 
  let b := 1.2 * c
  let d := 1.4 * b
  d = 1.68 * c := by sorry

end NUMINAMATH_CALUDE_card_relationship_l727_72758


namespace NUMINAMATH_CALUDE_unknown_number_proof_l727_72705

theorem unknown_number_proof (n : ℝ) (h : (12 : ℝ) * n^4 / 432 = 36) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l727_72705


namespace NUMINAMATH_CALUDE_luke_spent_eleven_l727_72799

/-- The amount of money Luke spent, given his initial amount, 
    the amount he received, and his current amount. -/
def money_spent (initial amount_received current : ℕ) : ℕ :=
  initial + amount_received - current

/-- Theorem stating that Luke spent $11 -/
theorem luke_spent_eleven : 
  money_spent 48 21 58 = 11 := by sorry

end NUMINAMATH_CALUDE_luke_spent_eleven_l727_72799


namespace NUMINAMATH_CALUDE_oliver_money_result_l727_72784

/-- Calculates the remaining money after Oliver's transactions -/
def oliver_money (initial : ℝ) (feb_spend_percent : ℝ) (march_add : ℝ) (final_spend_percent : ℝ) : ℝ :=
  let after_feb := initial * (1 - feb_spend_percent)
  let after_march := after_feb + march_add
  after_march * (1 - final_spend_percent)

/-- Theorem stating that Oliver's remaining money is $54.04 -/
theorem oliver_money_result :
  oliver_money 33 0.15 32 0.10 = 54.04 := by
  sorry

end NUMINAMATH_CALUDE_oliver_money_result_l727_72784


namespace NUMINAMATH_CALUDE_integer_equation_existence_l727_72763

theorem integer_equation_existence :
  (¬ ∃ (m n : ℕ+), m * (m + 2) = n * (n + 1)) ∧
  (¬ ∃ (m n : ℕ+), m * (m + 3) = n * (n + 1)) ∧
  (∀ k : ℕ+, k ≥ 4 → ∃ (m n : ℕ+), m * (m + k) = n * (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_equation_existence_l727_72763


namespace NUMINAMATH_CALUDE_rectangle_D_leftmost_l727_72707

-- Define the structure for a rectangle
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the sum of side labels for a rectangle
def sum_labels (r : Rectangle) : Int :=
  r.w + r.x + r.y + r.z

-- Define the five rectangles
def rectangle_A : Rectangle := ⟨3, 2, 5, 8⟩
def rectangle_B : Rectangle := ⟨2, 1, 4, 7⟩
def rectangle_C : Rectangle := ⟨4, 9, 6, 3⟩
def rectangle_D : Rectangle := ⟨8, 6, 5, 9⟩
def rectangle_E : Rectangle := ⟨10, 3, 8, 1⟩

-- Theorem: Rectangle D has the highest sum of side labels
theorem rectangle_D_leftmost :
  sum_labels rectangle_D > sum_labels rectangle_A ∧
  sum_labels rectangle_D > sum_labels rectangle_B ∧
  sum_labels rectangle_D > sum_labels rectangle_C ∧
  sum_labels rectangle_D > sum_labels rectangle_E :=
by sorry

end NUMINAMATH_CALUDE_rectangle_D_leftmost_l727_72707


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l727_72733

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l727_72733


namespace NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l727_72732

theorem largest_angle_convex_pentagon (x : ℚ) :
  (x + 2) + (2*x + 3) + (3*x + 6) + (4*x + 5) + (5*x + 4) = 540 →
  max (x + 2) (max (2*x + 3) (max (3*x + 6) (max (4*x + 5) (5*x + 4)))) = 532 / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l727_72732


namespace NUMINAMATH_CALUDE_energy_change_in_triangle_l727_72742

/-- The energy stored between two point charges -/
def energy_between_charges (distance : ℝ) : ℝ := sorry

/-- The total energy stored in a system of three point charges -/
def total_energy (d1 d2 d3 : ℝ) : ℝ := 
  energy_between_charges d1 + energy_between_charges d2 + energy_between_charges d3

theorem energy_change_in_triangle (initial_energy : ℝ) :
  initial_energy = 18 →
  ∃ (energy_func : ℝ → ℝ),
    (energy_func 1 + energy_func 1 + energy_func (Real.sqrt 2) = initial_energy) ∧
    (energy_func 1 + energy_func (Real.sqrt 2 / 2) + energy_func (Real.sqrt 2 / 2) = 6 + 12 * Real.sqrt 2) := by
  sorry

#check energy_change_in_triangle

end NUMINAMATH_CALUDE_energy_change_in_triangle_l727_72742


namespace NUMINAMATH_CALUDE_complex_equation_solution_l727_72746

theorem complex_equation_solution (a : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (1 + a * i) * i = -3 + i) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l727_72746


namespace NUMINAMATH_CALUDE_max_flowers_grown_l727_72744

theorem max_flowers_grown (total_seeds : ℕ) (seeds_per_bed : ℕ) : 
  total_seeds = 55 → seeds_per_bed = 15 → ∃ (max_flowers : ℕ), max_flowers ≤ 55 ∧ 
  ∀ (actual_flowers : ℕ), actual_flowers ≤ max_flowers := by
  sorry

end NUMINAMATH_CALUDE_max_flowers_grown_l727_72744


namespace NUMINAMATH_CALUDE_new_average_after_dropping_lowest_l727_72700

def calculate_new_average (num_tests : ℕ) (original_average : ℚ) (lowest_score : ℚ) : ℚ :=
  ((num_tests : ℚ) * original_average - lowest_score) / ((num_tests : ℚ) - 1)

theorem new_average_after_dropping_lowest
  (num_tests : ℕ)
  (original_average : ℚ)
  (lowest_score : ℚ)
  (h1 : num_tests = 4)
  (h2 : original_average = 35)
  (h3 : lowest_score = 20) :
  calculate_new_average num_tests original_average lowest_score = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_new_average_after_dropping_lowest_l727_72700


namespace NUMINAMATH_CALUDE_work_completion_time_l727_72776

theorem work_completion_time 
  (total_work : ℝ) 
  (p_q_together_time : ℝ) 
  (p_alone_time : ℝ) 
  (h1 : p_q_together_time = 6)
  (h2 : p_alone_time = 15)
  : ∃ q_alone_time : ℝ, q_alone_time = 10 ∧ 
    (1 / p_q_together_time = 1 / p_alone_time + 1 / q_alone_time) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l727_72776


namespace NUMINAMATH_CALUDE_area_ratio_second_third_neighbor_octagons_l727_72760

/-- A regular octagon -/
structure RegularOctagon where
  -- Add necessary fields

/-- The octagon formed by connecting second neighboring vertices -/
def secondNeighborOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The octagon formed by connecting third neighboring vertices -/
def thirdNeighborOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- The theorem stating the ratio of areas -/
theorem area_ratio_second_third_neighbor_octagons (o : RegularOctagon) :
  area (secondNeighborOctagon o) / area (thirdNeighborOctagon o) = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_second_third_neighbor_octagons_l727_72760


namespace NUMINAMATH_CALUDE_product_ab_equals_negative_one_l727_72709

theorem product_ab_equals_negative_one (a b : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) → 
  a * b = -1 := by
sorry

end NUMINAMATH_CALUDE_product_ab_equals_negative_one_l727_72709


namespace NUMINAMATH_CALUDE_triangle_side_ratio_sum_bounds_l727_72712

theorem triangle_side_ratio_sum_bounds (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  1 < a / (b + c) + b / (c + a) + c / (a + b) ∧ a / (b + c) + b / (c + a) + c / (a + b) < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_sum_bounds_l727_72712


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l727_72711

theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b →  -- non-square condition
  a * b - 3 * (a + b) = 3 * a * b - 9 →  -- given equation
  2 * (a + b) = 14 :=  -- perimeter = 14
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l727_72711


namespace NUMINAMATH_CALUDE_star_three_five_l727_72723

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

-- Theorem statement
theorem star_three_five : star 3 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_star_three_five_l727_72723


namespace NUMINAMATH_CALUDE_mary_nickels_problem_l727_72702

/-- The number of nickels Mary's dad gave her -/
def nickels_from_dad (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

theorem mary_nickels_problem :
  let initial_nickels : ℕ := 7
  let final_nickels : ℕ := 12
  nickels_from_dad initial_nickels final_nickels = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_problem_l727_72702


namespace NUMINAMATH_CALUDE_point_movement_l727_72738

/-- Given a point P in a Cartesian coordinate system, moving it upwards and to the left results in the expected new coordinates. -/
theorem point_movement (x y dx dy : ℤ) :
  let P : ℤ × ℤ := (x, y)
  let P' : ℤ × ℤ := (x - dx, y + dy)
  (P = (-2, 5) ∧ dx = 1 ∧ dy = 3) → P' = (-3, 8) := by
  sorry

#check point_movement

end NUMINAMATH_CALUDE_point_movement_l727_72738


namespace NUMINAMATH_CALUDE_min_coach_handshakes_zero_l727_72718

/-- Represents the total number of handshakes in the gymnastics meet -/
def total_handshakes : ℕ := 325

/-- Calculates the number of handshakes between gymnasts given the total number of gymnasts -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the minimum number of coach handshakes is 0 -/
theorem min_coach_handshakes_zero :
  ∃ (n : ℕ), gymnast_handshakes n = total_handshakes ∧ n > 1 :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_zero_l727_72718


namespace NUMINAMATH_CALUDE_women_average_age_l727_72759

/-- The average age of two women given specific conditions about a group of men --/
theorem women_average_age (n : ℕ) (A : ℝ) (age1 age2 : ℕ) (increase : ℝ) : 
  n = 10 ∧ age1 = 18 ∧ age2 = 22 ∧ increase = 6 →
  (n : ℝ) * (A + increase) - (n : ℝ) * A = 
    (((n : ℝ) * (A + increase) - (n : ℝ) * A + age1 + age2) / 2) * 2 - (age1 + age2) →
  ((n : ℝ) * (A + increase) - (n : ℝ) * A + age1 + age2) / 2 = 50 :=
by sorry

end NUMINAMATH_CALUDE_women_average_age_l727_72759


namespace NUMINAMATH_CALUDE_point_symmetry_wrt_origin_l727_72773

/-- Given a point M with coordinates (-2,3), its coordinates with respect to the origin are (2,-3). -/
theorem point_symmetry_wrt_origin : 
  let M : ℝ × ℝ := (-2, 3)
  (- M.1, - M.2) = (2, -3) := by sorry

end NUMINAMATH_CALUDE_point_symmetry_wrt_origin_l727_72773


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l727_72736

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a ternary number represented as a list of trits to its decimal equivalent -/
def ternary_to_decimal (trits : List ℕ) : ℕ :=
  trits.foldr (fun t acc => 3 * acc + t) 0

/-- The binary representation of 1101₂ -/
def binary_num : List Bool := [true, true, false, true]

/-- The ternary representation of 211₃ -/
def ternary_num : List ℕ := [2, 1, 1]

theorem product_of_binary_and_ternary :
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 286 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l727_72736


namespace NUMINAMATH_CALUDE_min_value_of_m_l727_72722

theorem min_value_of_m (x y : ℝ) (h1 : y = x^2 - 2) (h2 : x > Real.sqrt 3) :
  let m := (3*x + y - 4)/(x - 1) + (x + 3*y - 4)/(y - 1)
  m ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), y₀ = x₀^2 - 2 ∧ x₀ > Real.sqrt 3 ∧
    (3*x₀ + y₀ - 4)/(x₀ - 1) + (x₀ + 3*y₀ - 4)/(y₀ - 1) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_m_l727_72722


namespace NUMINAMATH_CALUDE_polynomial_factorization_1_l727_72780

theorem polynomial_factorization_1 (a : ℝ) : 
  a^7 + a^5 + 1 = (a^2 + a + 1) * (a^5 - a^4 + a^3 - a + 1) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_1_l727_72780


namespace NUMINAMATH_CALUDE_orange_theft_ratio_l727_72740

/-- Proves the ratio of stolen oranges to remaining oranges is 1:2 --/
theorem orange_theft_ratio :
  ∀ (initial_oranges eaten_oranges returned_oranges final_oranges : ℕ),
    initial_oranges = 60 →
    eaten_oranges = 10 →
    returned_oranges = 5 →
    final_oranges = 30 →
    ∃ (stolen_oranges : ℕ),
      stolen_oranges = initial_oranges - eaten_oranges - (final_oranges - returned_oranges) ∧
      2 * stolen_oranges = initial_oranges - eaten_oranges :=
by
  sorry

#check orange_theft_ratio

end NUMINAMATH_CALUDE_orange_theft_ratio_l727_72740


namespace NUMINAMATH_CALUDE_sequence_integer_count_l727_72793

def sequence_term (n : ℕ) : ℚ :=
  9720 / 2^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n))) ∧
  (∀ (k : ℕ), k > 0 →
    ((∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
     (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n)))
    → k = 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l727_72793


namespace NUMINAMATH_CALUDE_max_at_two_l727_72717

/-- The function f(x) defined as x(x-c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) with respect to x --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem max_at_two (c : ℝ) :
  (∀ x : ℝ, f c x ≤ f c 2) ↔ c = 6 := by sorry

end NUMINAMATH_CALUDE_max_at_two_l727_72717


namespace NUMINAMATH_CALUDE_expression_equals_eighteen_l727_72703

theorem expression_equals_eighteen (x : ℝ) (h : x + 1 = 4) :
  (-3)^3 + (-3)^2 + (-3*x)^1 + 3*x^1 + 3^2 + 3^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_eighteen_l727_72703


namespace NUMINAMATH_CALUDE_speed_ratio_l727_72730

/-- Represents the position and speed of an object moving in a straight line. -/
structure Mover where
  speed : ℝ
  initialPosition : ℝ

/-- The problem setup -/
def problem (a b : Mover) : Prop :=
  -- A and B move uniformly along two straight paths intersecting at right angles at point O
  -- When A is at O, B is 400 yards short of O
  a.initialPosition = 0 ∧ b.initialPosition = -400 ∧
  -- In 3 minutes, they are equidistant from O
  (3 * a.speed)^2 = (-400 + 3 * b.speed)^2 ∧
  -- In 10 minutes (3 + 7 minutes), they are again equidistant from O
  (10 * a.speed)^2 = (-400 + 10 * b.speed)^2

/-- The theorem to be proved -/
theorem speed_ratio (a b : Mover) :
  problem a b → a.speed / b.speed = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_l727_72730


namespace NUMINAMATH_CALUDE_complex_number_problem_l727_72795

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_problem (z : ℂ) 
  (h1 : is_purely_imaginary z) 
  (h2 : is_purely_imaginary ((z + 2)^2 - 8*I)) : 
  z = -2*I := by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l727_72795


namespace NUMINAMATH_CALUDE_competition_outcomes_l727_72786

/-- The number of possible outcomes for champions in a competition -/
def num_outcomes (num_students : ℕ) (num_events : ℕ) : ℕ :=
  num_students ^ num_events

/-- Theorem: Given 3 students competing in 2 events, where each event has one champion,
    the total number of possible outcomes for the champions is 9. -/
theorem competition_outcomes :
  num_outcomes 3 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_competition_outcomes_l727_72786


namespace NUMINAMATH_CALUDE_distinct_projections_exist_l727_72704

/-- Represents a student's marks as a point in 12-dimensional space -/
def Student := Fin 12 → ℝ

/-- The set of 7 students -/
def Students := Fin 7 → Student

theorem distinct_projections_exist (students : Students) 
  (h : ∀ i j, i ≠ j → students i ≠ students j) :
  ∃ (subjects : Fin 6 → Fin 12), 
    ∀ i j, i ≠ j → 
      ∃ k, (students i (subjects k)) ≠ (students j (subjects k)) := by
  sorry

end NUMINAMATH_CALUDE_distinct_projections_exist_l727_72704


namespace NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l727_72769

/-- The decimal representation of a repeating decimal with a single digit. -/
def repeating_decimal (d : ℕ) : ℚ :=
  (d : ℚ) / 9

/-- Theorem stating that 0.666... (repeating) is equal to 2/3 -/
theorem repeating_six_equals_two_thirds : repeating_decimal 6 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l727_72769


namespace NUMINAMATH_CALUDE_amc10_paths_count_l727_72724

/-- Represents the grid structure for spelling "AMC10" --/
structure AMC10Grid where
  a_to_m : Nat  -- Number of 'M's adjacent to central 'A'
  m_to_c : Nat  -- Number of 'C's adjacent to each 'M' (excluding path back to 'A')
  c_to_10 : Nat -- Number of '10' blocks adjacent to each 'C'

/-- Calculates the number of paths to spell "AMC10" in the given grid --/
def count_paths (grid : AMC10Grid) : Nat :=
  grid.a_to_m * grid.m_to_c * grid.c_to_10

/-- The specific grid configuration for the problem --/
def problem_grid : AMC10Grid :=
  { a_to_m := 4, m_to_c := 3, c_to_10 := 1 }

/-- Theorem stating that the number of paths to spell "AMC10" in the problem grid is 12 --/
theorem amc10_paths_count :
  count_paths problem_grid = 12 := by
  sorry

end NUMINAMATH_CALUDE_amc10_paths_count_l727_72724


namespace NUMINAMATH_CALUDE_paper_towel_savings_l727_72774

/-- Calculates the percent of savings per roll when buying a package of paper towels
    compared to buying individual rolls. -/
def percent_savings (package_price : ℚ) (package_size : ℕ) (individual_price : ℚ) : ℚ :=
  let package_price_per_roll := package_price / package_size
  let savings_per_roll := individual_price - package_price_per_roll
  (savings_per_roll / individual_price) * 100

/-- Theorem stating that the percent of savings for a 12-roll package priced at $9
    compared to buying 12 rolls individually at $1 each is 25%. -/
theorem paper_towel_savings :
  percent_savings 9 12 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_paper_towel_savings_l727_72774


namespace NUMINAMATH_CALUDE_not_A_inter_B_eq_open_closed_interval_l727_72755

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x - 1| > 2}

def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem not_A_inter_B_eq_open_closed_interval : 
  (Aᶜ ∩ B) = {x | 2 < x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_CALUDE_not_A_inter_B_eq_open_closed_interval_l727_72755


namespace NUMINAMATH_CALUDE_polynomial_factor_l727_72771

theorem polynomial_factor (a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + a*x - 5 = (x - 2) * (x + k)) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l727_72771


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l727_72734

/-- Given a parallelogram with opposite vertices (2, -3) and (14, 9),
    the intersection point of its diagonals is (8, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (14, 9)
  let midpoint := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l727_72734


namespace NUMINAMATH_CALUDE_rectangle_side_length_l727_72750

theorem rectangle_side_length (area : ℚ) (side1 : ℚ) (side2 : ℚ) : 
  area = 1/8 → side1 = 1/2 → area = side1 * side2 → side2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l727_72750


namespace NUMINAMATH_CALUDE_line_through_points_l727_72765

-- Define the line
def line (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem line_through_points :
  ∀ (a b : ℝ),
  (line a b 3 = 10) →
  (line a b 7 = 22) →
  a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l727_72765


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l727_72782

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x + 3| < 8 ↔ -4.5 < x ∧ x < 3.5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l727_72782


namespace NUMINAMATH_CALUDE_find_N_l727_72772

/-- Given three numbers a, b, and c, and a value N, satisfying certain conditions,
    prove that N = 41 is the integer solution that best satisfies all conditions. -/
theorem find_N : ∃ (a b c N : ℚ),
  a + b + c = 90 ∧
  a - 7 = N ∧
  b + 7 = N ∧
  5 * c = N ∧
  N.floor = 41 :=
by sorry

end NUMINAMATH_CALUDE_find_N_l727_72772


namespace NUMINAMATH_CALUDE_seashells_count_l727_72749

theorem seashells_count (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) 
  (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_seashells_count_l727_72749


namespace NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l727_72725

/-- Represents the dimensions and areas of a yard with flower beds -/
structure YardWithFlowerBeds where
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  trapezoid_height : ℝ
  total_length : ℝ

/-- Calculates the fraction of the yard occupied by flower beds -/
def flower_bed_fraction (yard : YardWithFlowerBeds) : ℚ :=
  125 / 310

/-- Theorem stating that the fraction of the yard occupied by flower beds is 125/310 -/
theorem flower_bed_fraction_is_correct (yard : YardWithFlowerBeds) 
  (h1 : yard.trapezoid_short_side = 30)
  (h2 : yard.trapezoid_long_side = 40)
  (h3 : yard.trapezoid_height = 6)
  (h4 : yard.total_length = 60) : 
  flower_bed_fraction yard = 125 / 310 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l727_72725


namespace NUMINAMATH_CALUDE_hyperbola_properties_l727_72770

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0
  h_asymptote : b / a = Real.sqrt 3
  h_vertex : a = 1

/-- Represents a point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hyperbola equation -/
def hyperbola_eq (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The line equation -/
def line_eq (m : ℝ) (p : Point) : Prop :=
  p.y = p.x + m

/-- Theorem stating the properties of the hyperbola and its intersection with a line -/
theorem hyperbola_properties (h : Hyperbola) (m : ℝ) (A B M : Point) 
    (h_distinct : A ≠ B)
    (h_intersect_A : hyperbola_eq h A ∧ line_eq m A)
    (h_intersect_B : hyperbola_eq h B ∧ line_eq m B)
    (h_midpoint : M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2)
    (h_nonzero : M.x ≠ 0) :
  (h.a = 1 ∧ h.b = Real.sqrt 3) ∧ M.y / M.x = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l727_72770


namespace NUMINAMATH_CALUDE_second_number_is_984_l727_72781

theorem second_number_is_984 (a b : ℕ) : 
  a < 10 ∧ b < 10 ∧ 
  a + b = 10 ∧
  (1000 + 300 + 10 * b + 7) % 11 = 0 →
  1000 + 300 + 10 * b + 7 - (400 + 10 * a + 3) = 984 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_984_l727_72781


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l727_72783

def A : Set ℤ := {-1, 1}

def B : Set ℤ := {x | |x + 1/2| < 3/2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l727_72783


namespace NUMINAMATH_CALUDE_bee_swarm_size_l727_72794

theorem bee_swarm_size :
  ∃ n : ℕ,
    n > 0 ∧
    (n : ℝ) = (Real.sqrt ((n : ℝ) / 2)) + (8 / 9 * n) + 1 ∧
    n = 72 := by
  sorry

end NUMINAMATH_CALUDE_bee_swarm_size_l727_72794


namespace NUMINAMATH_CALUDE_second_turkey_weight_proof_l727_72762

/-- The weight of the second turkey in kilograms -/
def second_turkey_weight : ℝ := 9

/-- The total cost of all turkeys in dollars -/
def total_cost : ℝ := 66

/-- The cost of turkey per kilogram in dollars -/
def cost_per_kg : ℝ := 2

/-- The weight of the first turkey in kilograms -/
def first_turkey_weight : ℝ := 6

theorem second_turkey_weight_proof :
  second_turkey_weight = 9 :=
by
  have h1 : total_cost = (first_turkey_weight + second_turkey_weight + 2 * second_turkey_weight) * cost_per_kg :=
    sorry
  have h2 : total_cost = (6 + 3 * second_turkey_weight) * 2 :=
    sorry
  have h3 : 66 = (6 + 3 * second_turkey_weight) * 2 :=
    sorry
  have h4 : 33 = 6 + 3 * second_turkey_weight :=
    sorry
  have h5 : 27 = 3 * second_turkey_weight :=
    sorry
  sorry

end NUMINAMATH_CALUDE_second_turkey_weight_proof_l727_72762


namespace NUMINAMATH_CALUDE_train_length_l727_72708

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) : 
  speed = 36 * (1000 / 3600) → 
  time = 27.997760179185665 → 
  bridge_length = 180 → 
  speed * time - bridge_length = 99.97760179185665 := by
sorry

#eval (36 * (1000 / 3600) * 27.997760179185665 - 180)

end NUMINAMATH_CALUDE_train_length_l727_72708


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l727_72716

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 →
  (-2 * x^2 + 5 * x - 6) / (x^3 + x) = -6 / x + (4 * x + 5) / (x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l727_72716


namespace NUMINAMATH_CALUDE_new_average_after_exclusion_l727_72756

theorem new_average_after_exclusion (total_students : ℕ) (initial_average : ℚ) 
  (excluded_students : ℕ) (excluded_average : ℚ) (new_average : ℚ) : 
  total_students = 20 →
  initial_average = 90 →
  excluded_students = 2 →
  excluded_average = 45 →
  new_average = (total_students * initial_average - excluded_students * excluded_average) / 
    (total_students - excluded_students) →
  new_average = 95 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_exclusion_l727_72756


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l727_72737

theorem exam_maximum_marks (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) : 
  percentage = 92 / 100 → 
  scored_marks = 460 → 
  percentage * max_marks = scored_marks → 
  max_marks = 500 := by
sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l727_72737


namespace NUMINAMATH_CALUDE_counterexample_exists_l727_72726

theorem counterexample_exists : ∃ n : ℕ, ¬ Nat.Prime n ∧ ¬ Nat.Prime (n - 3) ∧ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l727_72726


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_coincide_l727_72785

/-- The squared semi-major axis of the ellipse -/
def a_squared_ellipse : ℝ := 25

/-- The squared semi-major axis of the hyperbola -/
def a_squared_hyperbola : ℝ := 196

/-- The squared semi-minor axis of the hyperbola -/
def b_squared_hyperbola : ℝ := 121

/-- The equation of the ellipse -/
def ellipse_equation (x y b : ℝ) : Prop :=
  x^2 / a_squared_ellipse + y^2 / b^2 = 1

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / a_squared_hyperbola - y^2 / b_squared_hyperbola = 1/49

/-- The theorem stating that if the foci of the ellipse and hyperbola coincide,
    then the squared semi-minor axis of the ellipse is 908/49 -/
theorem ellipse_hyperbola_foci_coincide :
  ∃ b : ℝ, (∀ x y : ℝ, ellipse_equation x y b ↔ hyperbola_equation x y) →
    b^2 = 908/49 := by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_coincide_l727_72785


namespace NUMINAMATH_CALUDE_canoe_rowing_probability_l727_72751

theorem canoe_rowing_probability (p : ℝ) (h_p : p = 3/5) :
  let q := 1 - p
  p * p + p * q + q * p = 21/25 := by sorry

end NUMINAMATH_CALUDE_canoe_rowing_probability_l727_72751


namespace NUMINAMATH_CALUDE_initial_deficit_calculation_l727_72775

/-- Represents the score difference at the start of the final quarter -/
def initial_deficit : ℤ := sorry

/-- Liz's free throw points -/
def free_throw_points : ℕ := 5

/-- Liz's three-pointer points -/
def three_pointer_points : ℕ := 9

/-- Liz's jump shot points -/
def jump_shot_points : ℕ := 8

/-- Other team's points in the final quarter -/
def other_team_points : ℕ := 10

/-- Final score difference (negative means Liz's team lost) -/
def final_score_difference : ℤ := -8

theorem initial_deficit_calculation :
  initial_deficit = 20 :=
by sorry

end NUMINAMATH_CALUDE_initial_deficit_calculation_l727_72775


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l727_72727

/-- Given a rectangle with one side length b, diagonal length d, and the difference between
    the diagonal and the other side (d-a), prove that the side lengths of the rectangle
    are a = d - √(d² - b²) and b. -/
theorem rectangle_side_lengths
  (b d : ℝ) (h : b > 0) (h' : d > b) :
  let a := d - Real.sqrt (d^2 - b^2)
  (a > 0 ∧ a < d) ∧ 
  (a^2 + b^2 = d^2) ∧
  (d - a = Real.sqrt (d^2 - b^2)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l727_72727


namespace NUMINAMATH_CALUDE_jessica_cut_nineteen_orchids_l727_72798

/-- The number of orchids Jessica cut from her garden -/
def orchids_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) : ℕ :=
  final_orchids - initial_orchids

/-- Theorem stating that Jessica cut 19 orchids -/
theorem jessica_cut_nineteen_orchids :
  orchids_cut 12 2 10 21 = 19 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_nineteen_orchids_l727_72798


namespace NUMINAMATH_CALUDE_largest_number_proof_l727_72792

theorem largest_number_proof (a b c : ℝ) 
  (sum_eq : a + b + c = 100)
  (larger_diff : c - b = 8)
  (smaller_diff : b - a = 5) :
  c = 121 / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_proof_l727_72792


namespace NUMINAMATH_CALUDE_larger_field_time_calculation_l727_72715

-- Define the smaller field's dimensions
def small_width : ℝ := 1  -- We can use any positive real number as the base
def small_length : ℝ := 1.5 * small_width

-- Define the larger field's dimensions
def large_width : ℝ := 4 * small_width
def large_length : ℝ := 3 * small_length

-- Define the perimeters
def small_perimeter : ℝ := 2 * (small_length + small_width)
def large_perimeter : ℝ := 2 * (large_length + large_width)

-- Define the time to complete one round of the smaller field
def small_field_time : ℝ := 20

-- Theorem to prove
theorem larger_field_time_calculation :
  (large_perimeter / small_perimeter) * small_field_time = 68 := by
  sorry

end NUMINAMATH_CALUDE_larger_field_time_calculation_l727_72715


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l727_72766

theorem polynomial_division_degree (f d q r : Polynomial ℝ) :
  (Polynomial.degree f = 15) →
  (Polynomial.degree q = 7) →
  (r = 5 * X^2 + 3 * X - 8) →
  (f = d * q + r) →
  Polynomial.degree d = 8 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l727_72766


namespace NUMINAMATH_CALUDE_n_minus_m_equals_six_l727_72787

-- Define the sets M and N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 6}

-- Define the set difference operation
def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem n_minus_m_equals_six : set_difference N M = {6} := by
  sorry

end NUMINAMATH_CALUDE_n_minus_m_equals_six_l727_72787


namespace NUMINAMATH_CALUDE_complex_plane_properties_l727_72728

/-- Given complex numbers and their corresponding points in the complex plane, prove various geometric properties. -/
theorem complex_plane_properties (z₁ z₂ z₃ : ℂ) 
  (h₁ : z₁ = -3 + 4*I) 
  (h₂ : z₂ = 1 + 7*I) 
  (h₃ : z₃ = 3 - 4*I) : 
  (z₂.re > 0 ∧ z₂.im > 0) ∧ 
  (z₁ = -z₃) ∧ 
  (z₁.re * z₂.re + z₁.im * z₂.im > 0) ∧
  (z₁.re * (z₂.re + z₃.re) + z₁.im * (z₂.im + z₃.im) = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_plane_properties_l727_72728


namespace NUMINAMATH_CALUDE_man_downstream_speed_l727_72779

/-- The speed of a man rowing in a stream -/
structure RowingSpeed :=
  (still : ℝ)        -- Speed in still water
  (upstream : ℝ)     -- Speed upstream
  (downstream : ℝ)   -- Speed downstream

/-- Calculate the downstream speed given still water and upstream speeds -/
def calculate_downstream_speed (s : RowingSpeed) : Prop :=
  s.downstream = s.still + (s.still - s.upstream)

/-- Theorem: The man's downstream speed is 55 kmph -/
theorem man_downstream_speed :
  ∃ (s : RowingSpeed), s.still = 50 ∧ s.upstream = 45 ∧ s.downstream = 55 ∧ calculate_downstream_speed s :=
sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l727_72779


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l727_72768

/-- Given that the solution set of ax^2 - 1999x + b > 0 is {x | -3 < x < -1},
    prove that the solution set of ax^2 + 1999x + b > 0 is {x | 1 < x < 3} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x : ℝ, (a * x^2 - 1999 * x + b > 0) ↔ (-3 < x ∧ x < -1)) :
  ∀ x : ℝ, (a * x^2 + 1999 * x + b > 0) ↔ (1 < x ∧ x < 3) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l727_72768


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_twelve_l727_72739

theorem sum_of_divisors_of_twelve : (Finset.filter (λ x => 12 % x = 0) (Finset.range 13)).sum id = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_twelve_l727_72739


namespace NUMINAMATH_CALUDE_enrollment_difference_l727_72714

def maple_ridge_enrollment : ℕ := 1500
def south_park_enrollment : ℕ := 2100
def lakeside_enrollment : ℕ := 2700
def riverdale_enrollment : ℕ := 1800
def brookwood_enrollment : ℕ := 900

def school_enrollments : List ℕ := [
  maple_ridge_enrollment,
  south_park_enrollment,
  lakeside_enrollment,
  riverdale_enrollment,
  brookwood_enrollment
]

theorem enrollment_difference : 
  (List.maximum school_enrollments).get! - (List.minimum school_enrollments).get! = 1800 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_difference_l727_72714


namespace NUMINAMATH_CALUDE_transistor_growth_1992_to_2004_l727_72745

/-- Moore's Law: Number of transistors doubles every 2 years -/
def moores_law (initial_transistors : ℕ) (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / 2)

/-- Theorem: A CPU with 2,000,000 transistors in 1992 would have 128,000,000 transistors in 2004 -/
theorem transistor_growth_1992_to_2004 :
  moores_law 2000000 (2004 - 1992) = 128000000 := by
  sorry

#eval moores_law 2000000 (2004 - 1992)

end NUMINAMATH_CALUDE_transistor_growth_1992_to_2004_l727_72745


namespace NUMINAMATH_CALUDE_skin_cost_problem_l727_72747

/-- Given two skins with a total value of 2250 rubles, sold with a total profit of 40%,
    where the profit from the first skin is 25% and the profit from the second skin is -50%,
    prove that the cost of the first skin is 2700 rubles and the cost of the second skin is 450 rubles. -/
theorem skin_cost_problem (x : ℝ) (h1 : x + (2250 - x) = 2250) 
  (h2 : 1.25 * x + 0.5 * (2250 - x) = 1.4 * 2250) : x = 2700 ∧ 2250 - x = 450 := by
  sorry

#check skin_cost_problem

end NUMINAMATH_CALUDE_skin_cost_problem_l727_72747


namespace NUMINAMATH_CALUDE_expression_value_l727_72753

theorem expression_value : (3 * 12 + 18) / (6 - 3) = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l727_72753


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l727_72743

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l727_72743


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l727_72720

/-- Represents the composition of a student body -/
structure StudentBody where
  total : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  sum_eq_total : freshmen + sophomores + juniors = total

/-- Represents a stratified sample from a student body -/
structure StratifiedSample where
  body : StudentBody
  sample_size : ℕ
  sampled_freshmen : ℕ
  sampled_sophomores : ℕ
  sampled_juniors : ℕ
  sum_eq_sample_size : sampled_freshmen + sampled_sophomores + sampled_juniors = sample_size

/-- Checks if a stratified sample is proportionally correct -/
def is_proportional_sample (sample : StratifiedSample) : Prop :=
  sample.sampled_freshmen * sample.body.total = sample.body.freshmen * sample.sample_size ∧
  sample.sampled_sophomores * sample.body.total = sample.body.sophomores * sample.sample_size ∧
  sample.sampled_juniors * sample.body.total = sample.body.juniors * sample.sample_size

theorem correct_stratified_sample :
  let school : StudentBody := {
    total := 1000,
    freshmen := 400,
    sophomores := 340,
    juniors := 260,
    sum_eq_total := by sorry
  }
  let sample : StratifiedSample := {
    body := school,
    sample_size := 50,
    sampled_freshmen := 20,
    sampled_sophomores := 17,
    sampled_juniors := 13,
    sum_eq_sample_size := by sorry
  }
  is_proportional_sample sample := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l727_72720


namespace NUMINAMATH_CALUDE_valid_triples_eq_solution_set_l727_72777

/-- Represents a triple of side lengths (a, b, c) of a triangle -/
structure TriangleSides where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The set of all valid triangle side triples satisfying the given conditions -/
def validTriples : Set TriangleSides := {t | 
  t.a ≤ t.b ∧ t.b ≤ t.c ∧  -- a ≤ b ≤ c
  t.b ^ 2 = t.a * t.c ∧    -- geometric progression
  (t.a = 100 ∨ t.c = 100)  -- at least one of a or c is 100
}

/-- The set of all solutions given in the problem -/
def solutionSet : Set TriangleSides := {
  ⟨49, 70, 100⟩, ⟨64, 80, 100⟩, ⟨81, 90, 100⟩, ⟨100, 100, 100⟩,
  ⟨100, 110, 121⟩, ⟨100, 120, 144⟩, ⟨100, 130, 169⟩, ⟨100, 140, 196⟩,
  ⟨100, 150, 225⟩, ⟨100, 160, 256⟩
}

/-- Theorem stating that the set of valid triples is exactly the solution set -/
theorem valid_triples_eq_solution_set : validTriples = solutionSet := by
  sorry

end NUMINAMATH_CALUDE_valid_triples_eq_solution_set_l727_72777


namespace NUMINAMATH_CALUDE_intersection_condition_l727_72752

/-- Curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- Curve C₂ in Cartesian coordinates -/
def C₂ (x y : ℝ) : Prop := y = x

/-- C₂ translated downward by m units -/
def C₂_translated (x y m : ℝ) : Prop := y = x - m

/-- Two points in common between C₁ and translated C₂ -/
def two_intersections (m : ℝ) : Prop :=
  ∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ 
    C₁ p₁.1 p₁.2 ∧ C₂_translated p₁.1 p₁.2 m ∧
    C₁ p₂.1 p₂.2 ∧ C₂_translated p₂.1 p₂.2 m

/-- Main theorem -/
theorem intersection_condition (m : ℝ) :
  (m > 0 ∧ two_intersections m) ↔ (4 ≤ m ∧ m < 2 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l727_72752


namespace NUMINAMATH_CALUDE_total_books_l727_72797

theorem total_books (joan_books tom_books : ℕ) 
  (h1 : joan_books = 10) 
  (h2 : tom_books = 38) : 
  joan_books + tom_books = 48 := by
sorry

end NUMINAMATH_CALUDE_total_books_l727_72797


namespace NUMINAMATH_CALUDE_exactly_one_success_probability_l727_72796

theorem exactly_one_success_probability (p : ℝ) (h1 : p = 1/3) : 
  3 * (1 - p) * p^2 = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_success_probability_l727_72796


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l727_72788

/-- Calculates the number of medium stores to be drawn in stratified sampling -/
def medium_stores_drawn (total_stores : ℕ) (medium_stores : ℕ) (sample_size : ℕ) : ℕ :=
  (medium_stores * sample_size) / total_stores

theorem stratified_sampling_medium_stores 
  (total_stores : ℕ) (medium_stores : ℕ) (sample_size : ℕ)
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  medium_stores_drawn total_stores medium_stores sample_size = 5 := by
sorry

#eval medium_stores_drawn 300 75 20

end NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l727_72788


namespace NUMINAMATH_CALUDE_circle_properties_l727_72735

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m+3)*x + 2*(1-4*m^2)*y + 16*m^4 + 9 = 0

-- Define the theorem
theorem circle_properties :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) →
  ((-1/7 < m ∧ m < 1) ∧
   (∃ r : ℝ, 0 < r ∧ r ≤ 4 * Real.sqrt 7 / 7 ∧
    ∀ x y : ℝ, circle_equation x y m → (x - (m+3))^2 + (y - (4*m^2-1))^2 = r^2) ∧
   (∀ y : ℝ, (∃ x : ℝ, circle_equation x y m) → y ≥ -1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l727_72735


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_product_difference_l727_72719

theorem consecutive_integers_square_sum_product_difference : 
  let a : ℕ := 9
  let b : ℕ := 10
  (a^2 + b^2) - (a * b) = 91 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_product_difference_l727_72719


namespace NUMINAMATH_CALUDE_expand_product_l727_72757

theorem expand_product (x : ℝ) : (x + 4) * (x - 5 + 2) = x^2 + x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l727_72757
