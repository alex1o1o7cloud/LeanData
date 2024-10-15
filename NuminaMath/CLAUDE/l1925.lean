import Mathlib

namespace NUMINAMATH_CALUDE_age_difference_l1925_192524

theorem age_difference (albert_age mary_age betty_age : ℕ) : 
  albert_age = 2 * mary_age →
  albert_age = 4 * betty_age →
  betty_age = 4 →
  albert_age - mary_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1925_192524


namespace NUMINAMATH_CALUDE_central_position_theorem_l1925_192525

/-- Represents a row of stones -/
def StoneRow := List Bool

/-- An action changes the color of neighboring stones of a black stone -/
def action (row : StoneRow) (pos : Nat) : StoneRow :=
  sorry

/-- Checks if all stones in the row are black -/
def allBlack (row : StoneRow) : Prop :=
  sorry

/-- Checks if a given initial position can lead to all black stones -/
def canMakeAllBlack (initialPos : Nat) (totalStones : Nat) : Prop :=
  sorry

theorem central_position_theorem :
  ∀ initialPos : Nat,
    initialPos ≤ 2009 →
    canMakeAllBlack initialPos 2009 ↔ initialPos = 1005 :=
  sorry

end NUMINAMATH_CALUDE_central_position_theorem_l1925_192525


namespace NUMINAMATH_CALUDE_max_discarded_grapes_proof_l1925_192575

/-- The number of children among whom the grapes are to be distributed. -/
def num_children : ℕ := 8

/-- The maximum number of grapes that could be discarded. -/
def max_discarded_grapes : ℕ := num_children - 1

/-- Theorem stating that the maximum number of discarded grapes is one less than the number of children. -/
theorem max_discarded_grapes_proof :
  max_discarded_grapes = num_children - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_discarded_grapes_proof_l1925_192575


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1925_192558

theorem sum_of_fractions : 
  (1 / 15 : ℚ) + (2 / 15 : ℚ) + (3 / 15 : ℚ) + (4 / 15 : ℚ) + 
  (5 / 15 : ℚ) + (6 / 15 : ℚ) + (7 / 15 : ℚ) + (8 / 15 : ℚ) + 
  (30 / 15 : ℚ) = 4 + (2 / 5 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1925_192558


namespace NUMINAMATH_CALUDE_right_triangle_validity_and_area_l1925_192527

theorem right_triangle_validity_and_area :
  ∀ (a b c : ℝ),
  a = 5 ∧ c = 13 ∧ a < b ∧ b < c →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 30 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_validity_and_area_l1925_192527


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l1925_192577

theorem five_digit_multiple_of_nine : ∃ (d : ℕ), d < 10 ∧ (56780 + d) % 9 = 0 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l1925_192577


namespace NUMINAMATH_CALUDE_coords_wrt_origin_invariant_point_P_coords_l1925_192553

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- Coordinates of a point with respect to the origin -/
def coordsWrtOrigin (p : Point) : ℝ × ℝ := (p.x, p.y)

theorem coords_wrt_origin_invariant (p : Point) :
  coordsWrtOrigin p = (p.x, p.y) := by sorry

theorem point_P_coords :
  let P : Point := ⟨-1, -3⟩
  coordsWrtOrigin P = (-1, -3) := by sorry

end NUMINAMATH_CALUDE_coords_wrt_origin_invariant_point_P_coords_l1925_192553


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l1925_192501

theorem same_terminal_side_angle : ∃ (θ : Real), 
  0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
  ∃ (k : ℤ), θ = 2 * k * Real.pi + (-4 * Real.pi / 3) ∧
  θ = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l1925_192501


namespace NUMINAMATH_CALUDE_ladies_walking_group_miles_l1925_192592

/-- Calculates the total miles walked by a group of ladies over a number of days -/
def totalMilesWalked (groupSize : ℕ) (daysWalked : ℕ) (groupMiles : ℕ) (jamieExtra : ℕ) (sueExtra : ℕ) : ℕ :=
  groupSize * groupMiles * daysWalked + (jamieExtra + sueExtra) * daysWalked

/-- Theorem stating the total miles walked by the group under given conditions -/
theorem ladies_walking_group_miles :
  ∀ d : ℕ, totalMilesWalked 5 d 3 2 1 = 18 * d :=
by
  sorry

#check ladies_walking_group_miles

end NUMINAMATH_CALUDE_ladies_walking_group_miles_l1925_192592


namespace NUMINAMATH_CALUDE_negative_number_identification_l1925_192517

theorem negative_number_identification :
  (|-2023| ≥ 0) ∧ 
  (Real.sqrt ((-2)^2) ≥ 0) ∧ 
  (0 ≥ 0) ∧ 
  (-3^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_number_identification_l1925_192517


namespace NUMINAMATH_CALUDE_equilateral_triangles_in_decagon_l1925_192523

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Count of distinct equilateral triangles with at least two vertices in a regular polygon -/
def count_equilateral_triangles (n : ℕ) (polygon : RegularPolygon n) : ℕ :=
  sorry

theorem equilateral_triangles_in_decagon :
  ∃ (decagon : RegularPolygon 10),
    count_equilateral_triangles 10 decagon = 82 :=
  sorry

end NUMINAMATH_CALUDE_equilateral_triangles_in_decagon_l1925_192523


namespace NUMINAMATH_CALUDE_A_intersect_B_l1925_192509

def A : Set ℝ := {-1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}

theorem A_intersect_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1925_192509


namespace NUMINAMATH_CALUDE_cos_equality_implies_70_l1925_192510

theorem cos_equality_implies_70 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) 
  (h3 : Real.cos (n * π / 180) = Real.cos (1010 * π / 180)) : n = 70 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_implies_70_l1925_192510


namespace NUMINAMATH_CALUDE_equation_identity_l1925_192500

theorem equation_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_equation_identity_l1925_192500


namespace NUMINAMATH_CALUDE_triangle_problem_l1925_192596

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_problem (t : Triangle) 
  (h1 : 2 * Real.sin t.A * Real.cos t.B + Real.sin t.B = 2 * Real.sin t.C)
  (h2 : t.a = 4 * Real.sqrt 3)
  (h3 : t.b + t.c = 8) : 
  t.A = Real.pi / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A : Real) = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1925_192596


namespace NUMINAMATH_CALUDE_numerator_of_x_l1925_192576

theorem numerator_of_x (x y a : ℝ) : 
  x + y = -10 → 
  x^2 + y^2 = 50 → 
  x = a / y → 
  a = 25 := by sorry

end NUMINAMATH_CALUDE_numerator_of_x_l1925_192576


namespace NUMINAMATH_CALUDE_truck_speed_truck_speed_proof_l1925_192561

/-- Proves that a truck traveling 600 meters in 60 seconds has a speed of 36 kilometers per hour -/
theorem truck_speed : ℝ → Prop :=
  fun (speed : ℝ) =>
    let distance : ℝ := 600  -- meters
    let time : ℝ := 60       -- seconds
    let meters_per_km : ℝ := 1000
    let seconds_per_hour : ℝ := 3600
    speed = (distance / time) * (seconds_per_hour / meters_per_km) → speed = 36

/-- The actual speed of the truck in km/h -/
def actual_speed : ℝ := 36

theorem truck_speed_proof : truck_speed actual_speed :=
  sorry

end NUMINAMATH_CALUDE_truck_speed_truck_speed_proof_l1925_192561


namespace NUMINAMATH_CALUDE_exists_set_with_150_primes_l1925_192546

/-- The number of primes less than 1000 -/
def primes_lt_1000 : ℕ := 168

/-- Function that counts the number of primes in a set of 2002 consecutive integers starting from n -/
def count_primes (n : ℕ) : ℕ := sorry

theorem exists_set_with_150_primes :
  ∃ n : ℕ, count_primes n = 150 :=
sorry

end NUMINAMATH_CALUDE_exists_set_with_150_primes_l1925_192546


namespace NUMINAMATH_CALUDE_sphere_sum_l1925_192544

theorem sphere_sum (x y z : ℝ) : 
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 → x + y + z = 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_sum_l1925_192544


namespace NUMINAMATH_CALUDE_special_house_additional_profit_l1925_192567

/-- The additional profit made by building and selling a special house compared to a regular house -/
theorem special_house_additional_profit
  (C : ℝ)  -- Regular house construction cost
  (regular_selling_price : ℝ)
  (special_selling_price : ℝ)
  (h1 : regular_selling_price = 350000)
  (h2 : special_selling_price = 1.8 * regular_selling_price)
  : (special_selling_price - (C + 200000)) - (regular_selling_price - C) = 80000 := by
  sorry

#check special_house_additional_profit

end NUMINAMATH_CALUDE_special_house_additional_profit_l1925_192567


namespace NUMINAMATH_CALUDE_quadratic_two_roots_range_quadratic_solutions_when_k_is_one_l1925_192583

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := k * x^2 - (2*k + 4) * x + k - 6

-- Theorem for the range of k
theorem quadratic_two_roots_range (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) ↔ 
  (k > -2/5 ∧ k ≠ 0) :=
sorry

-- Theorem for solutions when k = 1
theorem quadratic_solutions_when_k_is_one :
  ∃ x y : ℝ, x ≠ y ∧ 
  quadratic 1 x = 0 ∧ quadratic 1 y = 0 ∧
  x = 3 + Real.sqrt 14 ∧ y = 3 - Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_range_quadratic_solutions_when_k_is_one_l1925_192583


namespace NUMINAMATH_CALUDE_marble_groups_l1925_192584

theorem marble_groups (total_marbles : ℕ) (marbles_per_group : ℕ) (num_groups : ℕ) :
  total_marbles = 64 →
  marbles_per_group = 2 →
  num_groups * marbles_per_group = total_marbles →
  num_groups = 32 := by
  sorry

end NUMINAMATH_CALUDE_marble_groups_l1925_192584


namespace NUMINAMATH_CALUDE_teacher_discount_l1925_192541

theorem teacher_discount (students : ℕ) (pens_per_student : ℕ) (notebooks_per_student : ℕ) 
  (binders_per_student : ℕ) (highlighters_per_student : ℕ) 
  (pen_cost : ℚ) (notebook_cost : ℚ) (binder_cost : ℚ) (highlighter_cost : ℚ) 
  (amount_spent : ℚ) : 
  students = 30 →
  pens_per_student = 5 →
  notebooks_per_student = 3 →
  binders_per_student = 1 →
  highlighters_per_student = 2 →
  pen_cost = 1/2 →
  notebook_cost = 5/4 →
  binder_cost = 17/4 →
  highlighter_cost = 3/4 →
  amount_spent = 260 →
  (students * pens_per_student * pen_cost + 
   students * notebooks_per_student * notebook_cost + 
   students * binders_per_student * binder_cost + 
   students * highlighters_per_student * highlighter_cost) - amount_spent = 100 := by
  sorry

end NUMINAMATH_CALUDE_teacher_discount_l1925_192541


namespace NUMINAMATH_CALUDE_tommys_nickels_l1925_192585

/-- The number of coins Tommy has in his collection. -/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ
  dollar_coins : ℕ

/-- Tommy's coin collection satisfies the given conditions. -/
def valid_collection (c : CoinCollection) : Prop :=
  c.dimes = c.pennies + 10 ∧
  c.nickels = 2 * c.dimes ∧
  c.quarters = 4 ∧
  c.pennies = 10 * c.quarters ∧
  c.half_dollars = c.quarters + 5 ∧
  c.dollar_coins = 3 * c.half_dollars

/-- The number of nickels in Tommy's collection is 100. -/
theorem tommys_nickels (c : CoinCollection) (h : valid_collection c) : c.nickels = 100 := by
  sorry

end NUMINAMATH_CALUDE_tommys_nickels_l1925_192585


namespace NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l1925_192550

-- Define the function h
def h (x : ℝ) : ℝ := 5 * x - 7

-- Theorem statement
theorem h_zero_at_seven_fifths : h (7 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l1925_192550


namespace NUMINAMATH_CALUDE_amp_four_two_l1925_192563

-- Define the & operation
def amp (a b : ℝ) : ℝ := ((a + b) * (a - b))^2

-- Theorem statement
theorem amp_four_two : amp 4 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_amp_four_two_l1925_192563


namespace NUMINAMATH_CALUDE_largest_divisor_power_l1925_192536

-- Define pow function
def pow (n : ℕ) : ℕ :=
  sorry

-- Define X
def X : ℕ := 2310

-- Define the product of pow(n) from 2 to 5400
def product : ℕ :=
  sorry

-- Theorem statement
theorem largest_divisor_power : 
  (∃ m : ℕ, X^m ∣ product ∧ ∀ k > m, ¬(X^k ∣ product)) → 
  (∃ m : ℕ, X^m ∣ product ∧ ∀ k > m, ¬(X^k ∣ product) ∧ m = 534) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_power_l1925_192536


namespace NUMINAMATH_CALUDE_B_subset_complement_A_iff_m_in_range_A_intersect_B_nonempty_iff_m_in_range_A_union_B_eq_A_iff_m_in_range_l1925_192555

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 > 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m*x^2 - (m+2)*x + 2 < 0}

-- State the theorems
theorem B_subset_complement_A_iff_m_in_range :
  ∀ m : ℝ, B m ⊆ (Set.univ \ A) ↔ m ∈ Set.Icc 1 2 := by sorry

theorem A_intersect_B_nonempty_iff_m_in_range :
  ∀ m : ℝ, (A ∩ B m).Nonempty ↔ m ∈ Set.Iic 1 ∪ Set.Ioi 2 := by sorry

theorem A_union_B_eq_A_iff_m_in_range :
  ∀ m : ℝ, A ∪ B m = A ↔ m ∈ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_B_subset_complement_A_iff_m_in_range_A_intersect_B_nonempty_iff_m_in_range_A_union_B_eq_A_iff_m_in_range_l1925_192555


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1925_192568

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  ((1/3)^2 + (1/4)^2) / ((1/5)^2 + (1/6)^2) = 21*x / (65*y) →
  Real.sqrt x / Real.sqrt y = 25 * Real.sqrt 65 / (2 * Real.sqrt 1281) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1925_192568


namespace NUMINAMATH_CALUDE_circle_has_longest_perimeter_l1925_192594

/-- The perimeter of a square with side length 7 cm -/
def square_perimeter : ℝ := 4 * 7

/-- The perimeter of an equilateral triangle with side length 9 cm -/
def triangle_perimeter : ℝ := 3 * 9

/-- The perimeter of a circle with radius 5 cm, using π = 3 -/
def circle_perimeter : ℝ := 2 * 3 * 5

theorem circle_has_longest_perimeter :
  circle_perimeter > square_perimeter ∧ circle_perimeter > triangle_perimeter :=
sorry

end NUMINAMATH_CALUDE_circle_has_longest_perimeter_l1925_192594


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l1925_192505

/-- The equation of a line passing through (-2, 3) with a 45° angle of inclination -/
theorem line_equation_through_point_with_inclination :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ (x + 2) = (y - 3)) ∧ 
    m = Real.tan (45 * π / 180) ∧
    (x - y + 5 = 0) = (y = m * x + b) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l1925_192505


namespace NUMINAMATH_CALUDE_framed_photo_border_area_l1925_192573

/-- The area of the border of a framed rectangular photograph. -/
theorem framed_photo_border_area 
  (photo_height : ℝ) 
  (photo_width : ℝ) 
  (border_width : ℝ) 
  (h1 : photo_height = 12) 
  (h2 : photo_width = 15) 
  (h3 : border_width = 3) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 198 := by
  sorry

end NUMINAMATH_CALUDE_framed_photo_border_area_l1925_192573


namespace NUMINAMATH_CALUDE_odd_number_grouping_l1925_192565

theorem odd_number_grouping (n : ℕ) (odd_number : ℕ) : 
  (odd_number = 2007) →
  (∀ k : ℕ, k < n → (k^2 < 1004 ∧ 1004 ≤ (k+1)^2)) →
  (n = 32) :=
by sorry

end NUMINAMATH_CALUDE_odd_number_grouping_l1925_192565


namespace NUMINAMATH_CALUDE_average_weight_decrease_l1925_192506

/-- Given a group of people and a new person joining, calculate the decrease in average weight -/
theorem average_weight_decrease (initial_count : ℕ) (initial_average : ℝ) (new_person_weight : ℝ) : 
  initial_count = 20 →
  initial_average = 55 →
  new_person_weight = 50 →
  let total_weight := initial_count * initial_average
  let new_total_weight := total_weight + new_person_weight
  let new_count := initial_count + 1
  let new_average := new_total_weight / new_count
  abs (initial_average - new_average - 0.24) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l1925_192506


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1925_192589

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + 3*x + 8 = 7 → 3*x^2 + 9*x - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1925_192589


namespace NUMINAMATH_CALUDE_sphere_surface_area_tangent_to_cube_l1925_192578

/-- The surface area of a sphere tangent to all six faces of a cube with edge length 2 is 4π. -/
theorem sphere_surface_area_tangent_to_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) : 
  cube_edge_length = 2 → 
  sphere_radius = cube_edge_length / 2 → 
  4 * Real.pi * sphere_radius^2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_tangent_to_cube_l1925_192578


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1925_192520

theorem degree_to_radian_conversion (π : ℝ) :
  (1 : ℝ) * π / 180 = π / 180 →
  (-300 : ℝ) * π / 180 = -5 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1925_192520


namespace NUMINAMATH_CALUDE_percentage_of_games_sold_l1925_192529

theorem percentage_of_games_sold (initial_cost : ℝ) (sold_price : ℝ) : 
  initial_cost = 200 → 
  sold_price = 240 → 
  (sold_price / (initial_cost * 3)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_games_sold_l1925_192529


namespace NUMINAMATH_CALUDE_stream_speed_l1925_192530

/-- Proves that the speed of a stream is 5 km/h, given a man's swimming speed in still water
    and the relative time taken to swim upstream vs downstream. -/
theorem stream_speed (man_speed : ℝ) (upstream_time_ratio : ℝ) 
  (h1 : man_speed = 15)
  (h2 : upstream_time_ratio = 2) : 
  ∃ (stream_speed : ℝ), stream_speed = 5 ∧
  (man_speed + stream_speed) * 1 = (man_speed - stream_speed) * upstream_time_ratio :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l1925_192530


namespace NUMINAMATH_CALUDE_unique_solution_system_l1925_192534

theorem unique_solution_system (x y : ℝ) :
  (2 * x - 3 * abs y = 1 ∧ abs x + 2 * y = 4) ↔ (x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1925_192534


namespace NUMINAMATH_CALUDE_alberts_earnings_increase_l1925_192542

theorem alberts_earnings_increase (E : ℝ) (P : ℝ) : 
  (1.26 * E = 693) → 
  ((1 + P) * E = 660) →
  P = 0.2 := by
sorry

end NUMINAMATH_CALUDE_alberts_earnings_increase_l1925_192542


namespace NUMINAMATH_CALUDE_sin_two_theta_l1925_192582

theorem sin_two_theta (θ : Real) 
  (h : Real.exp (Real.log 2 * (-2 + 3 * Real.sin θ)) + 1 = Real.exp (Real.log 2 * (1/2 + Real.sin θ))) : 
  Real.sin (2 * θ) = 4 * Real.sqrt 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_sin_two_theta_l1925_192582


namespace NUMINAMATH_CALUDE_union_of_A_and_B_is_reals_l1925_192539

-- Define sets A and B
def A : Set ℝ := {x | 4 * x - 3 > 0}
def B : Set ℝ := {x | x - 6 < 0}

-- State the theorem
theorem union_of_A_and_B_is_reals : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_is_reals_l1925_192539


namespace NUMINAMATH_CALUDE_sector_radius_l1925_192521

/-- Given a circular sector with area 13.75 cm² and arc length 5.5 cm, the radius is 5 cm -/
theorem sector_radius (area : Real) (arc_length : Real) (radius : Real) :
  area = 13.75 ∧ arc_length = 5.5 ∧ area = (1/2) * radius * arc_length →
  radius = 5 := by
sorry

end NUMINAMATH_CALUDE_sector_radius_l1925_192521


namespace NUMINAMATH_CALUDE_cube_root_minus_square_root_plus_power_l1925_192503

theorem cube_root_minus_square_root_plus_power : 
  ((-2 : ℝ)^3)^(1/3) - Real.sqrt 4 + (Real.sqrt 3)^0 = -3 := by sorry

end NUMINAMATH_CALUDE_cube_root_minus_square_root_plus_power_l1925_192503


namespace NUMINAMATH_CALUDE_problem_polygon_area_l1925_192590

/-- A polygon in 2D space defined by a list of points --/
def Polygon : Type := List (ℝ × ℝ)

/-- The polygon described in the problem --/
def problemPolygon : Polygon :=
  [(0,0), (5,0), (5,5), (0,5), (0,3), (3,3), (3,0), (0,0)]

/-- Calculates the area of a polygon --/
def polygonArea (p : Polygon) : ℝ :=
  sorry  -- The actual calculation would go here

/-- Theorem: The area of the problem polygon is 19 square units --/
theorem problem_polygon_area :
  polygonArea problemPolygon = 19 := by
  sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l1925_192590


namespace NUMINAMATH_CALUDE_sinusoidal_amplitude_l1925_192538

/-- Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
    if the function oscillates between 5 and -3, then a = 4. -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) :
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_amplitude_l1925_192538


namespace NUMINAMATH_CALUDE_last_term_is_123_l1925_192564

/-- A sequence of natural numbers -/
def Sequence : Type := ℕ → ℕ

/-- The specific sequence from the problem -/
def s : Sequence :=
  fun n =>
    match n with
    | 1 => 2
    | 2 => 3
    | 3 => 6
    | 4 => 15
    | 5 => 33
    | 6 => 123
    | _ => 0  -- For completeness, though we only care about the first 6 terms

/-- The theorem stating that the last (6th) term of the sequence is 123 -/
theorem last_term_is_123 : s 6 = 123 := by
  sorry


end NUMINAMATH_CALUDE_last_term_is_123_l1925_192564


namespace NUMINAMATH_CALUDE_odd_function_a_value_l1925_192597

/-- The logarithm function with base 10 -/
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The main theorem -/
theorem odd_function_a_value :
  ∃ a : ℝ, IsOdd (fun x ↦ lg ((2 / (1 - x)) + a)) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_a_value_l1925_192597


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1925_192543

theorem min_reciprocal_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (1 / m + 1 / n) ≥ 4 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ m + n = 1 ∧ 1 / m + 1 / n = 4 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1925_192543


namespace NUMINAMATH_CALUDE_median_is_106_l1925_192515

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list where each integer n (1 ≤ n ≤ 150) appears n times -/
def special_list : List ℕ := sorry

/-- The length of the special list -/
def special_list_length : ℕ := sum_to_n 150

/-- The median index of the special list -/
def median_index : ℕ := special_list_length / 2 + 1

theorem median_is_106 : 
  ∃ (l : List ℕ), l = special_list ∧ l.length = special_list_length ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 150 → (l.count n = n)) ∧
  (l.nthLe (median_index - 1) sorry = 106) :=
sorry

end NUMINAMATH_CALUDE_median_is_106_l1925_192515


namespace NUMINAMATH_CALUDE_base_representation_theorem_l1925_192599

theorem base_representation_theorem :
  (∃ b : ℕ, 1 < b ∧ b < 1993 ∧ 1994 = 2 * (1 + b)) ∧
  (∀ b : ℕ, 1 < b → b < 1992 → ¬∃ n : ℕ, n ≥ 2 ∧ 1993 * (b - 1) = b^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_base_representation_theorem_l1925_192599


namespace NUMINAMATH_CALUDE_phi_minus_phi_squared_l1925_192547

theorem phi_minus_phi_squared (Φ φ : ℝ) : 
  Φ ≠ φ → Φ^2 = Φ + 1 → φ^2 = φ + 1 → (Φ - φ)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_phi_minus_phi_squared_l1925_192547


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l1925_192514

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l1925_192514


namespace NUMINAMATH_CALUDE_green_hats_count_l1925_192581

/-- Proves that the number of green hats is 28 given the problem conditions --/
theorem green_hats_count : ∀ (blue green red : ℕ),
  blue + green + red = 85 →
  6 * blue + 7 * green + 8 * red = 600 →
  blue = 3 * green ∧ green = 2 * red →
  green = 28 := by
  sorry

end NUMINAMATH_CALUDE_green_hats_count_l1925_192581


namespace NUMINAMATH_CALUDE_consecutive_hits_arrangements_eq_30_l1925_192522

/-- Represents the number of ways to arrange 4 hits in 8 shots with exactly two consecutive hits -/
def consecutive_hits_arrangements : ℕ :=
  let total_shots : ℕ := 8
  let hits : ℕ := 4
  let misses : ℕ := total_shots - hits
  let spaces : ℕ := misses + 1
  let ways_to_place_consecutive_pair : ℕ := spaces
  let remaining_spaces : ℕ := spaces - 1
  let remaining_hits : ℕ := hits - 2
  let ways_to_place_remaining_hits : ℕ := Nat.choose remaining_spaces remaining_hits
  ways_to_place_consecutive_pair * ways_to_place_remaining_hits

/-- Theorem stating that the number of arrangements is 30 -/
theorem consecutive_hits_arrangements_eq_30 : consecutive_hits_arrangements = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_hits_arrangements_eq_30_l1925_192522


namespace NUMINAMATH_CALUDE_print_shop_price_difference_l1925_192566

def print_shop_x_price : ℚ := 120 / 100
def print_shop_y_price : ℚ := 170 / 100
def number_of_copies : ℕ := 40

theorem print_shop_price_difference :
  number_of_copies * print_shop_y_price - number_of_copies * print_shop_x_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_price_difference_l1925_192566


namespace NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l1925_192598

/-- Represents the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Represents the sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ :=
  (List.range n).map nthPrime |>.sum

/-- Theorem: For any n, there exists a perfect square between the sum of the first n primes
    and the sum of the first n+1 primes -/
theorem perfect_square_between_prime_sums (n : ℕ) :
  ∃ m : ℕ, sumFirstNPrimes n ≤ m^2 ∧ m^2 ≤ sumFirstNPrimes (n+1) := by sorry

end NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l1925_192598


namespace NUMINAMATH_CALUDE_existence_of_always_different_teams_l1925_192508

/-- Represents a team assignment for a single game -/
def GameAssignment := Fin 22 → Bool

/-- Represents the team assignments for all three games -/
def ThreeGamesAssignment := Fin 3 → GameAssignment

theorem existence_of_always_different_teams (games : ThreeGamesAssignment) : 
  ∃ (p1 p2 : Fin 22), p1 ≠ p2 ∧ 
    (∀ (g : Fin 3), games g p1 ≠ games g p2) :=
sorry

end NUMINAMATH_CALUDE_existence_of_always_different_teams_l1925_192508


namespace NUMINAMATH_CALUDE_min_value_theorem_l1925_192540

theorem min_value_theorem (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h : x * y - 2 * x - y + 1 = 0) : 
  ∀ z w : ℝ, z > 1 → w > 1 → z * w - 2 * z - w + 1 = 0 → 
  (3 / 2) * x^2 + y^2 ≤ (3 / 2) * z^2 + w^2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1925_192540


namespace NUMINAMATH_CALUDE_min_value_constrained_l1925_192537

theorem min_value_constrained (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (∃ m : ℝ, ∀ x y : ℝ, a * x^2 + b * y^2 = 1 → c * x + d * y^2 ≥ m) ∧
  (∀ ε > 0, ∃ x y : ℝ, a * x^2 + b * y^2 = 1 ∧ c * x + d * y^2 < -c / Real.sqrt a + ε) :=
sorry

end NUMINAMATH_CALUDE_min_value_constrained_l1925_192537


namespace NUMINAMATH_CALUDE_mitzi_bowling_score_l1925_192556

/-- Proves that given three bowlers with an average score of 106, where one bowler scores 120 
    and another scores 85, the third bowler's score must be 113. -/
theorem mitzi_bowling_score (average_score gretchen_score beth_score : ℕ) 
    (h1 : average_score = 106)
    (h2 : gretchen_score = 120)
    (h3 : beth_score = 85) : 
  ∃ mitzi_score : ℕ, mitzi_score = 113 ∧ 
    (gretchen_score + beth_score + mitzi_score) / 3 = average_score :=
by
  sorry


end NUMINAMATH_CALUDE_mitzi_bowling_score_l1925_192556


namespace NUMINAMATH_CALUDE_textbook_cost_decrease_l1925_192595

theorem textbook_cost_decrease (original_cost new_cost : ℝ) 
  (h1 : original_cost = 75)
  (h2 : new_cost = 60) :
  (1 - new_cost / original_cost) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_textbook_cost_decrease_l1925_192595


namespace NUMINAMATH_CALUDE_previous_year_300th_day_l1925_192569

/-- Represents the days of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.monday => DayOfWeek.tuesday
  | DayOfWeek.tuesday => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday => DayOfWeek.friday
  | DayOfWeek.friday => DayOfWeek.saturday
  | DayOfWeek.saturday => DayOfWeek.sunday
  | DayOfWeek.sunday => DayOfWeek.monday

/-- Returns the day of the week after n days -/
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (addDays d n)

theorem previous_year_300th_day 
  (current_year_200th_day : DayOfWeek)
  (next_year_100th_day : DayOfWeek)
  (h1 : current_year_200th_day = DayOfWeek.sunday)
  (h2 : next_year_100th_day = DayOfWeek.sunday)
  : addDays DayOfWeek.monday 299 = current_year_200th_day :=
by sorry

#check previous_year_300th_day

end NUMINAMATH_CALUDE_previous_year_300th_day_l1925_192569


namespace NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l1925_192586

theorem pythagorean_triple_divisibility (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ (k l m : ℤ), 
    (a = 3*k ∨ b = 3*k ∨ c = 3*k) ∧
    (a = 4*l ∨ b = 4*l ∨ c = 4*l) ∧
    (a = 5*m ∨ b = 5*m ∨ c = 5*m) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l1925_192586


namespace NUMINAMATH_CALUDE_total_pages_read_l1925_192531

theorem total_pages_read (pages_yesterday pages_today : ℕ) 
  (h1 : pages_yesterday = 21) 
  (h2 : pages_today = 17) : 
  pages_yesterday + pages_today = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_read_l1925_192531


namespace NUMINAMATH_CALUDE_smallest_m_pair_l1925_192512

/-- Given the equation 19m + 90 + 8n = 1998, where m and n are positive integers,
    the pair (m, n) with the smallest possible value for m is (4, 229). -/
theorem smallest_m_pair : 
  ∃ (m n : ℕ), 
    (∀ (m' n' : ℕ), 19 * m' + 90 + 8 * n' = 1998 → m ≤ m') ∧ 
    19 * m + 90 + 8 * n = 1998 ∧ 
    m = 4 ∧ 
    n = 229 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_pair_l1925_192512


namespace NUMINAMATH_CALUDE_project_hours_calculation_l1925_192518

theorem project_hours_calculation (kate : ℕ) (pat : ℕ) (mark : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : 3 * pat = mark)
  (h3 : mark = kate + 85) :
  kate + pat + mark = 153 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_calculation_l1925_192518


namespace NUMINAMATH_CALUDE_student_club_distribution_l1925_192554

-- Define the number of students and clubs
def num_students : ℕ := 5
def num_clubs : ℕ := 3

-- Define a function to calculate the number of ways to distribute students into clubs
def distribute_students (students : ℕ) (clubs : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem student_club_distribution :
  distribute_students num_students num_clubs = 150 :=
sorry

end NUMINAMATH_CALUDE_student_club_distribution_l1925_192554


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1925_192549

theorem quadratic_roots_condition (r : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (r - 4) * x₁^2 - 2*(r - 3) * x₁ + r = 0 ∧
   (r - 4) * x₂^2 - 2*(r - 3) * x₂ + r = 0 ∧
   x₁ > -1 ∧ x₂ > -1) ↔ 
  (3.5 < r ∧ r < 4.5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1925_192549


namespace NUMINAMATH_CALUDE_log_27_3_l1925_192513

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  have h : 27 = 3^3 := by sorry
  sorry

end NUMINAMATH_CALUDE_log_27_3_l1925_192513


namespace NUMINAMATH_CALUDE_price_reduction_options_best_discount_percentage_impossibility_of_higher_profit_l1925_192570

-- Define constants
def cost_price : ℝ := 240
def original_price : ℝ := 400
def initial_sales : ℝ := 200
def sales_increase_rate : ℝ := 4
def target_profit : ℝ := 41600
def impossible_profit : ℝ := 50000

-- Define function for weekly profit based on price reduction
def weekly_profit (price_reduction : ℝ) : ℝ :=
  (original_price - price_reduction - cost_price) * (initial_sales + sales_increase_rate * price_reduction)

-- Theorem 1: Price reduction options
theorem price_reduction_options :
  ∃ (x y : ℝ), x ≠ y ∧ weekly_profit x = target_profit ∧ weekly_profit y = target_profit :=
sorry

-- Theorem 2: Best discount percentage
theorem best_discount_percentage :
  ∃ (best_reduction : ℝ), weekly_profit best_reduction = target_profit ∧
    ∀ (other_reduction : ℝ), weekly_profit other_reduction = target_profit →
      best_reduction ≥ other_reduction ∧
      (original_price - best_reduction) / original_price = 0.8 :=
sorry

-- Theorem 3: Impossibility of higher profit
theorem impossibility_of_higher_profit :
  ∀ (price_reduction : ℝ), weekly_profit price_reduction ≠ impossible_profit :=
sorry

end NUMINAMATH_CALUDE_price_reduction_options_best_discount_percentage_impossibility_of_higher_profit_l1925_192570


namespace NUMINAMATH_CALUDE_salesman_pear_sales_l1925_192571

/-- A salesman's pear sales problem -/
theorem salesman_pear_sales (morning_sales afternoon_sales total_sales : ℕ) : 
  morning_sales = 120 →
  afternoon_sales = 240 →
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 360 := by
  sorry

#check salesman_pear_sales

end NUMINAMATH_CALUDE_salesman_pear_sales_l1925_192571


namespace NUMINAMATH_CALUDE_fred_cantaloupes_l1925_192533

theorem fred_cantaloupes (keith_cantaloupes jason_cantaloupes total_cantaloupes : ℕ)
  (h1 : keith_cantaloupes = 29)
  (h2 : jason_cantaloupes = 20)
  (h3 : total_cantaloupes = 65)
  (h4 : ∃ fred_cantaloupes : ℕ, keith_cantaloupes + jason_cantaloupes + fred_cantaloupes = total_cantaloupes) :
  ∃ fred_cantaloupes : ℕ, fred_cantaloupes = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_fred_cantaloupes_l1925_192533


namespace NUMINAMATH_CALUDE_fathers_catch_l1925_192516

/-- The number of fishes Hazel caught -/
def hazel_catch : ℕ := 48

/-- The total number of fishes caught by Hazel and her father -/
def total_catch : ℕ := 94

/-- Hazel's father's catch is the difference between the total catch and Hazel's catch -/
theorem fathers_catch (hazel_catch : ℕ) (total_catch : ℕ) : 
  total_catch - hazel_catch = 46 :=
by sorry

end NUMINAMATH_CALUDE_fathers_catch_l1925_192516


namespace NUMINAMATH_CALUDE_gcd_problems_l1925_192519

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 440 556 = 4) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l1925_192519


namespace NUMINAMATH_CALUDE_blue_face_area_l1925_192511

-- Define a tetrahedron with right-angled edges
structure RightAngledTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  red_area : ℝ
  yellow_area : ℝ
  green_area : ℝ
  blue_area : ℝ
  right_angle_condition : a^2 + b^2 = c^2
  red_area_condition : red_area = (1/2) * a * b
  yellow_area_condition : yellow_area = (1/2) * b * c
  green_area_condition : green_area = (1/2) * c * a
  blue_area_condition : blue_area = (1/2) * (a^2 + b^2 + c^2)

-- Theorem statement
theorem blue_face_area (t : RightAngledTetrahedron) 
  (h1 : t.red_area = 60) 
  (h2 : t.yellow_area = 20) 
  (h3 : t.green_area = 15) : 
  t.blue_area = 65 := by
  sorry


end NUMINAMATH_CALUDE_blue_face_area_l1925_192511


namespace NUMINAMATH_CALUDE_perpendicular_unit_vector_l1925_192504

theorem perpendicular_unit_vector (a : ℝ × ℝ) (v : ℝ × ℝ) : 
  a = (1, 1) → 
  v = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2) → 
  (a.1 * v.1 + a.2 * v.2 = 0) ∧ 
  (v.1^2 + v.2^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vector_l1925_192504


namespace NUMINAMATH_CALUDE_xy_value_l1925_192574

theorem xy_value (x y : ℝ) (h : |3*x + y - 2| + (2*x + 3*y + 1)^2 = 0) : x * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1925_192574


namespace NUMINAMATH_CALUDE_survey_theorem_l1925_192593

def survey_problem (total : ℕ) (high_bp : ℕ) (heart : ℕ) (diabetes : ℕ) 
                   (bp_heart : ℕ) (bp_diabetes : ℕ) (heart_diabetes : ℕ) 
                   (all_three : ℕ) : Prop :=
  let teachers_with_condition := 
    (high_bp - bp_heart - bp_diabetes + all_three) +
    (heart - bp_heart - heart_diabetes + all_three) +
    (diabetes - bp_diabetes - heart_diabetes + all_three) +
    (bp_heart - all_three) + (bp_diabetes - all_three) + 
    (heart_diabetes - all_three) + all_three
  let teachers_without_condition := total - teachers_with_condition
  (teachers_without_condition : ℚ) / (total : ℚ) * 100 = 50/3

theorem survey_theorem : 
  survey_problem 150 90 50 30 25 10 15 5 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_theorem_l1925_192593


namespace NUMINAMATH_CALUDE_dress_price_ratio_l1925_192528

theorem dress_price_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_ratio : ℝ := 2/3
  let cost : ℝ := cost_ratio * selling_price
  cost / marked_price = 1/2 := by
sorry

end NUMINAMATH_CALUDE_dress_price_ratio_l1925_192528


namespace NUMINAMATH_CALUDE_combined_swim_time_l1925_192580

def freestyle_time : ℕ := 48
def backstroke_time : ℕ := freestyle_time + 4
def butterfly_time : ℕ := backstroke_time + 3
def breaststroke_time : ℕ := butterfly_time + 2

theorem combined_swim_time : 
  freestyle_time + backstroke_time + butterfly_time + breaststroke_time = 212 := by
  sorry

end NUMINAMATH_CALUDE_combined_swim_time_l1925_192580


namespace NUMINAMATH_CALUDE_opposite_absolute_values_sum_l1925_192502

theorem opposite_absolute_values_sum (a b : ℝ) : 
  |a - 2| = -|b + 3| → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_absolute_values_sum_l1925_192502


namespace NUMINAMATH_CALUDE_probability_red_ball_experiment_l1925_192532

/-- The probability of picking a red ball in an experiment -/
def probability_red_ball (total_experiments : ℕ) (red_picks : ℕ) : ℚ :=
  red_picks / total_experiments

/-- Theorem: Given 10 experiments where red balls were picked 4 times, 
    the probability of picking a red ball is 0.4 -/
theorem probability_red_ball_experiment : 
  probability_red_ball 10 4 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_experiment_l1925_192532


namespace NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l1925_192545

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between a line and three circles -/
def max_line_circle_intersections : ℕ := 6

/-- The total maximum number of intersection points -/
def total_max_intersections : ℕ := max_circle_intersections + max_line_circle_intersections

theorem max_intersections_three_circles_one_line :
  total_max_intersections = 12 := by sorry

end NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l1925_192545


namespace NUMINAMATH_CALUDE_sine_cosine_product_l1925_192579

theorem sine_cosine_product (α : Real) (h : (1 / Real.sin α) + (1 / Real.cos α) = Real.sqrt 3) :
  Real.sin α * Real.cos α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_product_l1925_192579


namespace NUMINAMATH_CALUDE_justina_tallest_l1925_192526

-- Define a type for people
inductive Person : Type
  | Gisa : Person
  | Henry : Person
  | Ivan : Person
  | Justina : Person
  | Katie : Person

-- Define a height function
variable (height : Person → ℝ)

-- Define the conditions
axiom gisa_taller_than_henry : height Person.Gisa > height Person.Henry
axiom gisa_shorter_than_justina : height Person.Gisa < height Person.Justina
axiom ivan_taller_than_katie : height Person.Ivan > height Person.Katie
axiom ivan_shorter_than_gisa : height Person.Ivan < height Person.Gisa

-- Theorem to prove
theorem justina_tallest : 
  ∀ p : Person, height Person.Justina ≥ height p :=
sorry

end NUMINAMATH_CALUDE_justina_tallest_l1925_192526


namespace NUMINAMATH_CALUDE_average_equation_solution_l1925_192535

theorem average_equation_solution (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 69 → a = 26 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1925_192535


namespace NUMINAMATH_CALUDE_prime_relations_l1925_192572

theorem prime_relations (p : ℕ) : 
  (Prime p ∧ Prime (8*p - 1)) → (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = 8*p + 1) ∧
  (Prime p ∧ Prime (8*p^2 + 1)) → Prime (8*p^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_prime_relations_l1925_192572


namespace NUMINAMATH_CALUDE_bert_sandwiches_remaining_l1925_192559

def sandwiches_remaining (initial : ℕ) (first_day : ℕ) (second_day : ℕ) : ℕ :=
  initial - (first_day + second_day)

theorem bert_sandwiches_remaining :
  let initial := 12
  let first_day := initial / 2
  let second_day := first_day - 2
  sandwiches_remaining initial first_day second_day = 2 := by
sorry

end NUMINAMATH_CALUDE_bert_sandwiches_remaining_l1925_192559


namespace NUMINAMATH_CALUDE_expression_evaluation_l1925_192557

theorem expression_evaluation :
  let x : ℚ := 1/2
  6 * x^2 - (2*x + 1) * (3*x - 2) + (x + 3) * (x - 3) = -25/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1925_192557


namespace NUMINAMATH_CALUDE_jony_start_time_l1925_192548

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculate the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

/-- Represents Jony's walk -/
structure Walk where
  startBlock : Nat
  turnaroundBlock : Nat
  endBlock : Nat
  blockLength : Nat
  speed : Nat
  endTime : Time

theorem jony_start_time (w : Walk) (h1 : w.startBlock = 10)
    (h2 : w.turnaroundBlock = 90) (h3 : w.endBlock = 70)
    (h4 : w.blockLength = 40) (h5 : w.speed = 100)
    (h6 : w.endTime = ⟨7, 40⟩) :
    timeDifference w.endTime ⟨7, 0⟩ =
      ((w.turnaroundBlock - w.startBlock + w.turnaroundBlock - w.endBlock) * w.blockLength) / w.speed :=
  sorry

end NUMINAMATH_CALUDE_jony_start_time_l1925_192548


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l1925_192587

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  h_a : a < 10
  h_b : b < 10
  h_c : c < 10
  h_a_pos : 0 < a

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : ℕ :=
  100 * n.a + 10 * n.b + n.c

def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ℕ :=
  100 * n.c + 10 * n.b + n.a

def ThreeDigitNumber.sumDigits (n : ThreeDigitNumber) : ℕ :=
  n.a + n.b + n.c

theorem three_digit_number_theorem (n : ThreeDigitNumber) :
  (n.toNat / n.reverse = 3 ∧ n.toNat % n.reverse = n.sumDigits) →
  (n.toNat = 441 ∨ n.toNat = 882) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l1925_192587


namespace NUMINAMATH_CALUDE_jerry_syrup_time_l1925_192562

/-- Represents the time it takes Jerry to make cherry syrup -/
def make_cherry_syrup (cherries_per_quart : ℕ) (picking_time : ℕ) (picking_amount : ℕ) (syrup_making_time : ℕ) (quarts : ℕ) : ℕ :=
  let picking_rate : ℚ := picking_amount / picking_time
  let total_cherries : ℕ := cherries_per_quart * quarts
  let total_picking_time : ℕ := (total_cherries / picking_rate).ceil.toNat
  total_picking_time + syrup_making_time

/-- Proves that it takes Jerry 33 hours to make 9 quarts of cherry syrup -/
theorem jerry_syrup_time :
  make_cherry_syrup 500 2 300 3 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_jerry_syrup_time_l1925_192562


namespace NUMINAMATH_CALUDE_buffet_dishes_l1925_192551

theorem buffet_dishes (mango_salsa_dishes : ℕ) (mango_jelly_dishes : ℕ) (oliver_edible_dishes : ℕ) 
  (fresh_mango_ratio : ℚ) (oliver_pick_out_dishes : ℕ) :
  mango_salsa_dishes = 3 →
  mango_jelly_dishes = 1 →
  fresh_mango_ratio = 1 / 6 →
  oliver_pick_out_dishes = 2 →
  oliver_edible_dishes = 28 →
  ∃ (total_dishes : ℕ), 
    total_dishes = 36 ∧ 
    (fresh_mango_ratio * total_dishes : ℚ).num = oliver_pick_out_dishes + 
      (total_dishes - oliver_edible_dishes - mango_salsa_dishes - mango_jelly_dishes) :=
by sorry

end NUMINAMATH_CALUDE_buffet_dishes_l1925_192551


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_implies_sin_double_angle_l1925_192591

theorem tan_sum_reciprocal_implies_sin_double_angle (θ : Real) 
  (h : Real.tan θ + (Real.tan θ)⁻¹ = 4) : 
  Real.sin (2 * θ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_implies_sin_double_angle_l1925_192591


namespace NUMINAMATH_CALUDE_circle_center_coordinate_difference_l1925_192560

/-- Given two points as endpoints of a circle's diameter, 
    calculate the absolute difference between the x and y coordinates of the circle's center -/
theorem circle_center_coordinate_difference (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁ = 8 ∧ y₁ = -7)
  (h2 : x₂ = -4 ∧ y₂ = 5) : 
  |((x₁ + x₂) / 2) - ((y₁ + y₂) / 2)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_difference_l1925_192560


namespace NUMINAMATH_CALUDE_jake_weight_loss_l1925_192552

theorem jake_weight_loss (jake_current : ℕ) (combined_weight : ℕ) (weight_loss : ℕ) : 
  jake_current = 198 →
  combined_weight = 293 →
  jake_current - weight_loss = 2 * (combined_weight - jake_current) →
  weight_loss = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l1925_192552


namespace NUMINAMATH_CALUDE_four_digit_square_palindromes_l1925_192588

/-- A function that checks if a number is a 4-digit integer -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A function that checks if a number is a palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  digits = digits.reverse

/-- The main theorem stating that there are exactly 3 numbers satisfying all conditions -/
theorem four_digit_square_palindromes :
  ∃! (s : Finset ℕ), s.card = 3 ∧ 
  (∀ n ∈ s, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n) ∧
  (∀ n, is_four_digit n → is_perfect_square n → is_palindrome n → n ∈ s) :=
sorry

end NUMINAMATH_CALUDE_four_digit_square_palindromes_l1925_192588


namespace NUMINAMATH_CALUDE_pen_cost_is_six_l1925_192507

/-- The cost of the pen Joshua wants to buy -/
def pen_cost : ℚ := 6

/-- The amount of money Joshua has in his pocket -/
def pocket_money : ℚ := 5

/-- The amount of money Joshua borrowed from his neighbor -/
def borrowed_money : ℚ := 68 / 100

/-- The additional amount Joshua needs to buy the pen -/
def additional_money_needed : ℚ := 32 / 100

/-- Theorem stating that the cost of the pen is $6.00 -/
theorem pen_cost_is_six :
  pen_cost = pocket_money + borrowed_money + additional_money_needed :=
by sorry

end NUMINAMATH_CALUDE_pen_cost_is_six_l1925_192507
