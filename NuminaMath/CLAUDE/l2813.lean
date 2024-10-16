import Mathlib

namespace NUMINAMATH_CALUDE_flower_count_l2813_281361

theorem flower_count (minyoung_flowers yoojung_flowers : ℕ) : 
  minyoung_flowers = 24 → 
  minyoung_flowers = 4 * yoojung_flowers → 
  minyoung_flowers + yoojung_flowers = 30 := by
sorry

end NUMINAMATH_CALUDE_flower_count_l2813_281361


namespace NUMINAMATH_CALUDE_horizontal_arrangement_possible_l2813_281348

/-- Represents a chessboard with 65 cells -/
structure ExtendedChessboard :=
  (cells : Fin 65 → Bool)

/-- Represents a domino (1x2 rectangle) -/
structure Domino :=
  (start : Fin 65)
  (horizontal : Bool)

/-- Represents the state of the chessboard with dominos -/
structure BoardState :=
  (board : ExtendedChessboard)
  (dominos : Fin 32 → Domino)

/-- Checks if two cells are adjacent on the extended chessboard -/
def are_adjacent (a b : Fin 65) : Bool := sorry

/-- Checks if a domino placement is valid -/
def valid_domino_placement (board : ExtendedChessboard) (domino : Domino) : Prop := sorry

/-- Checks if all dominos are placed horizontally -/
def all_horizontal (state : BoardState) : Prop := sorry

/-- Represents a move of a domino to adjacent empty cells -/
def valid_move (state₁ state₂ : BoardState) : Prop := sorry

/-- Main theorem: It's always possible to arrange all dominos horizontally -/
theorem horizontal_arrangement_possible (initial_state : BoardState) : 
  (∀ d, valid_domino_placement initial_state.board (initial_state.dominos d)) → 
  ∃ final_state, (valid_move initial_state final_state ∧ all_horizontal final_state) := sorry

end NUMINAMATH_CALUDE_horizontal_arrangement_possible_l2813_281348


namespace NUMINAMATH_CALUDE_coin_count_l2813_281318

theorem coin_count (total_sum : ℕ) (coin_type1 coin_type2 : ℕ) (count_type1 : ℕ) :
  total_sum = 7100 →
  coin_type1 = 20 →
  coin_type2 = 25 →
  count_type1 = 290 →
  count_type1 * coin_type1 + (total_sum - count_type1 * coin_type1) / coin_type2 = 342 :=
by sorry

end NUMINAMATH_CALUDE_coin_count_l2813_281318


namespace NUMINAMATH_CALUDE_sequence_value_l2813_281303

/-- Given a sequence {aₙ} where a₁ = 3 and 2aₙ₊₁ - 2aₙ = 1 for all n ≥ 1,
    prove that a₉₉ = 52. -/
theorem sequence_value (a : ℕ → ℝ) (h1 : a 1 = 3) 
    (h2 : ∀ n : ℕ, 2 * a (n + 1) - 2 * a n = 1) : 
  a 99 = 52 := by
  sorry

end NUMINAMATH_CALUDE_sequence_value_l2813_281303


namespace NUMINAMATH_CALUDE_percentage_difference_in_gain_l2813_281328

/-- Given an article with cost price, and two selling prices, calculate the percentage difference in gain -/
theorem percentage_difference_in_gain 
  (cost_price : ℝ) 
  (selling_price1 : ℝ) 
  (selling_price2 : ℝ) 
  (h1 : cost_price = 250) 
  (h2 : selling_price1 = 350) 
  (h3 : selling_price2 = 340) : 
  (selling_price1 - cost_price - (selling_price2 - cost_price)) / (selling_price2 - cost_price) * 100 = 100 / 9 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_in_gain_l2813_281328


namespace NUMINAMATH_CALUDE_sandwich_meat_cost_l2813_281315

/-- The cost of a pack of sandwich meat given the following conditions:
  * 1 loaf of bread, 2 packs of sandwich meat, and 2 packs of sliced cheese make 10 sandwiches
  * Bread costs $4.00
  * Cheese costs $4.00 per pack
  * There's a $1.00 off coupon for one pack of cheese
  * There's a $1.00 off coupon for one pack of meat
  * Each sandwich costs $2.00
-/
theorem sandwich_meat_cost :
  let bread_cost : ℚ := 4
  let cheese_cost : ℚ := 4
  let cheese_discount : ℚ := 1
  let meat_discount : ℚ := 1
  let sandwich_cost : ℚ := 2
  let sandwich_count : ℕ := 10
  let total_cost : ℚ := sandwich_cost * sandwich_count
  let cheese_total : ℚ := 2 * cheese_cost - cheese_discount
  ∃ meat_cost : ℚ,
    bread_cost + cheese_total + 2 * meat_cost - meat_discount = total_cost ∧
    meat_cost = 5 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_meat_cost_l2813_281315


namespace NUMINAMATH_CALUDE_system_solution_l2813_281312

theorem system_solution (x y z : ℤ) : 
  (x + y + z = 6 ∧ x + y * z = 7) ↔ 
  ((x, y, z) = (7, 0, -1) ∨ 
   (x, y, z) = (7, -1, 0) ∨ 
   (x, y, z) = (1, 3, 2) ∨ 
   (x, y, z) = (1, 2, 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2813_281312


namespace NUMINAMATH_CALUDE_no_fractional_solution_l2813_281349

theorem no_fractional_solution (x y : ℝ) : 
  (∃ m n : ℤ, 13 * x + 4 * y = m ∧ 10 * x + 3 * y = n) → 
  (∃ a b : ℤ, x = a ∧ y = b) :=
by sorry

end NUMINAMATH_CALUDE_no_fractional_solution_l2813_281349


namespace NUMINAMATH_CALUDE_lars_baking_capacity_l2813_281376

/-- Represents the baking capacity of Lars' bakeshop -/
structure Bakeshop where
  baguettes_per_two_hours : ℕ
  baking_hours_per_day : ℕ
  total_breads_per_day : ℕ

/-- Calculates the number of loaves of bread that can be baked per hour -/
def loaves_per_hour (shop : Bakeshop) : ℚ :=
  let baguettes_per_day := shop.baguettes_per_two_hours * (shop.baking_hours_per_day / 2)
  let loaves_per_day := shop.total_breads_per_day - baguettes_per_day
  loaves_per_day / shop.baking_hours_per_day

/-- Theorem stating that Lars can bake 10 loaves of bread per hour -/
theorem lars_baking_capacity :
  let lars_shop : Bakeshop := {
    baguettes_per_two_hours := 30,
    baking_hours_per_day := 6,
    total_breads_per_day := 150
  }
  loaves_per_hour lars_shop = 10 := by
  sorry

end NUMINAMATH_CALUDE_lars_baking_capacity_l2813_281376


namespace NUMINAMATH_CALUDE_collinear_iff_sqrt_two_l2813_281391

def a (k : ℝ) : ℝ × ℝ := (k, 2)
def b (k : ℝ) : ℝ × ℝ := (1, k)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = (t • w.1, t • w.2)

theorem collinear_iff_sqrt_two (k : ℝ) :
  collinear (a k) (b k) ↔ k = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_iff_sqrt_two_l2813_281391


namespace NUMINAMATH_CALUDE_number_value_proof_l2813_281396

theorem number_value_proof (x : ℝ) : (8^3 * x^3) / 679 = 549.7025036818851 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_value_proof_l2813_281396


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2813_281389

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 3 - x) ↔ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2813_281389


namespace NUMINAMATH_CALUDE_stock_investment_change_l2813_281333

theorem stock_investment_change (x : ℝ) (x_pos : x > 0) : 
  x * (1 + 0.75) * (1 - 0.30) = 1.225 * x := by
  sorry

#check stock_investment_change

end NUMINAMATH_CALUDE_stock_investment_change_l2813_281333


namespace NUMINAMATH_CALUDE_total_fish_is_157_l2813_281379

/-- The number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- The number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish per multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def num_white_ducks : ℕ := 3

/-- The number of black ducks -/
def num_black_ducks : ℕ := 7

/-- The number of multicolor ducks -/
def num_multicolor_ducks : ℕ := 6

/-- The total number of fish in the lake -/
def total_fish : ℕ := fish_per_white_duck * num_white_ducks +
                      fish_per_black_duck * num_black_ducks +
                      fish_per_multicolor_duck * num_multicolor_ducks

theorem total_fish_is_157 : total_fish = 157 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_is_157_l2813_281379


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2813_281309

theorem quadratic_equation_solution :
  ∃! y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) :=
by
  -- The unique solution is y = -4
  use -4
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2813_281309


namespace NUMINAMATH_CALUDE_midpoint_parallelogram_area_ratio_l2813_281392

/-- Given a parallelogram, the area of the parallelogram formed by joining its midpoints is 1/4 of the original area -/
theorem midpoint_parallelogram_area_ratio (P : ℝ) (h : P > 0) :
  ∃ (smaller_area : ℝ), smaller_area = P / 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_parallelogram_area_ratio_l2813_281392


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2813_281372

def units_digit (n : ℕ) : ℕ := n % 10

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

theorem quadratic_function_theorem (a b c : ℤ) (p : ℕ) :
  is_positive_even p →
  10 ≤ p →
  p ≤ 50 →
  units_digit p > 0 →
  units_digit (p^3) - units_digit (p^2) = 0 →
  (a * p^2 + b * p + c : ℤ) = 0 →
  (a * p^4 + b * p^2 + c : ℤ) = (a * p^6 + b * p^3 + c : ℤ) →
  units_digit (p + 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2813_281372


namespace NUMINAMATH_CALUDE_problem_solution_l2813_281399

theorem problem_solution (x y : ℝ) (h : 3 * x - y ≤ Real.log (x + 2 * y - 3) + Real.log (2 * x - 3 * y + 5)) : 
  x + y = 16 / 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2813_281399


namespace NUMINAMATH_CALUDE_percentage_problem_l2813_281386

theorem percentage_problem (N P : ℝ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2813_281386


namespace NUMINAMATH_CALUDE_bus_stop_time_l2813_281360

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : 
  speed_without_stops = 40 →
  speed_with_stops = 30 →
  (1 - speed_with_stops / speed_without_stops) * 60 = 15 :=
by
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l2813_281360


namespace NUMINAMATH_CALUDE_dog_speed_l2813_281368

/-- Proves that a dog catching a rabbit with given parameters runs at 24 miles per hour -/
theorem dog_speed (rabbit_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  rabbit_speed = 15 →
  head_start = 0.6 →
  catch_up_time = 4 / 60 →
  let dog_distance := rabbit_speed * catch_up_time + head_start
  dog_distance / catch_up_time = 24 := by sorry

end NUMINAMATH_CALUDE_dog_speed_l2813_281368


namespace NUMINAMATH_CALUDE_student_B_more_stable_l2813_281385

/-- Represents a student's performance metrics -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if the first student has more stable performance than the second -/
def more_stable (s1 s2 : StudentPerformance) : Prop :=
  s1.average_score = s2.average_score ∧ s1.variance < s2.variance

/-- The performance metrics for student A -/
def student_A : StudentPerformance :=
  { average_score := 82
    variance := 245 }

/-- The performance metrics for student B -/
def student_B : StudentPerformance :=
  { average_score := 82
    variance := 190 }

/-- Theorem stating that student B has more stable performance than student A -/
theorem student_B_more_stable : more_stable student_B student_A := by
  sorry

end NUMINAMATH_CALUDE_student_B_more_stable_l2813_281385


namespace NUMINAMATH_CALUDE_range_of_z_l2813_281381

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  let z := x^2 + 4*y^2
  4 ≤ z ∧ z ≤ 12 := by
sorry

end NUMINAMATH_CALUDE_range_of_z_l2813_281381


namespace NUMINAMATH_CALUDE_unique_solution_proof_l2813_281305

/-- The positive value of k for which the equation 4x^2 + kx + 4 = 0 has exactly one solution -/
def unique_solution_k : ℝ := 8

theorem unique_solution_proof :
  ∃! (k : ℝ), k > 0 ∧
  (∃! (x : ℝ), 4 * x^2 + k * x + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_proof_l2813_281305


namespace NUMINAMATH_CALUDE_unique_representation_theorem_l2813_281313

theorem unique_representation_theorem (n : ℕ) :
  ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
sorry

end NUMINAMATH_CALUDE_unique_representation_theorem_l2813_281313


namespace NUMINAMATH_CALUDE_pyramid_theorem_l2813_281327

/-- A regular triangular pyramid with an inscribed sphere -/
structure RegularPyramidWithSphere where
  /-- The side length of the base triangle -/
  base_side : ℝ
  /-- The radius of the inscribed sphere -/
  sphere_radius : ℝ
  /-- The sphere is inscribed at the midpoint of the pyramid's height -/
  sphere_at_midpoint : True
  /-- The sphere touches the lateral faces of the pyramid -/
  sphere_touches_faces : True
  /-- A hemisphere supported by the inscribed circle in the base touches the sphere externally -/
  hemisphere_touches_sphere : True

/-- Properties of the regular triangular pyramid with inscribed sphere -/
def pyramid_properties (p : RegularPyramidWithSphere) : Prop :=
  p.sphere_radius = 1 ∧
  p.base_side = 2 * Real.sqrt 3 * (Real.sqrt 5 + 1)

/-- The lateral surface area of the pyramid -/
noncomputable def lateral_surface_area (p : RegularPyramidWithSphere) : ℝ :=
  3 * Real.sqrt 15 * (Real.sqrt 5 + 1)

/-- The angle between lateral faces of the pyramid -/
noncomputable def lateral_face_angle (p : RegularPyramidWithSphere) : ℝ :=
  Real.arccos (1 / Real.sqrt 5)

/-- Theorem stating the properties of the pyramid -/
theorem pyramid_theorem (p : RegularPyramidWithSphere) 
  (h : pyramid_properties p) :
  lateral_surface_area p = 3 * Real.sqrt 15 * (Real.sqrt 5 + 1) ∧
  lateral_face_angle p = Real.arccos (1 / Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_theorem_l2813_281327


namespace NUMINAMATH_CALUDE_m_range_l2813_281366

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, 
  m * x₁^2 - x₁ + m - 4 = 0 ∧ 
  m * x₂^2 - x₂ + m - 4 = 0 ∧ 
  x₁ > 0 ∧ x₂ < 0

-- Define the theorem
theorem m_range : 
  ∀ m : ℝ, (p m ∨ q m) → ¬(p m) → m ≥ 1 + Real.sqrt 2 ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2813_281366


namespace NUMINAMATH_CALUDE_coin_arrangement_coin_arrangement_proof_l2813_281324

theorem coin_arrangement (total_coins : ℕ) (walls : ℕ) (coins_per_wall : ℕ → Prop) : Prop :=
  (total_coins = 12 ∧ walls = 4) →
  (∀ n, coins_per_wall n → n ≥ 2 ∧ n ≤ 6) →
  (∃! n, coins_per_wall n ∧ n * walls = total_coins)

-- The proof goes here
theorem coin_arrangement_proof : coin_arrangement 12 4 (λ n ↦ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_coin_arrangement_coin_arrangement_proof_l2813_281324


namespace NUMINAMATH_CALUDE_remainder_problem_l2813_281308

theorem remainder_problem (k : ℕ+) (h : 90 % k.val^2 = 18) : 130 % k.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2813_281308


namespace NUMINAMATH_CALUDE_problem_solution_l2813_281358

-- Define the propositions
def proposition_A (x : ℝ) : Prop := (x^2 - 4*x + 3 = 0) → (x = 3)
def proposition_B (x : ℝ) : Prop := (x > 1) → (|x| > 0)
def proposition_C (p q : Prop) : Prop := (¬p ∧ ¬q) → (¬p ∧ ¬q)
def proposition_D : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define the correctness of each statement
def statement_A_correct : Prop :=
  ∀ x : ℝ, (x ≠ 3 → x^2 - 4*x + 3 ≠ 0) ↔ proposition_A x

def statement_B_correct : Prop :=
  (∀ x : ℝ, x > 1 → |x| > 0) ∧ (∃ x : ℝ, |x| > 0 ∧ x ≤ 1)

def statement_C_incorrect : Prop :=
  ∃ p q : Prop, ¬p ∧ ¬q ∧ ¬(proposition_C p q)

def statement_D_correct : Prop :=
  (¬proposition_D) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)

-- Main theorem
theorem problem_solution :
  statement_A_correct ∧ statement_B_correct ∧ statement_C_incorrect ∧ statement_D_correct :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2813_281358


namespace NUMINAMATH_CALUDE_discount_profit_calculation_l2813_281301

/-- Calculates the profit percentage with a given discount, based on the no-discount profit percentage. -/
def profit_with_discount (no_discount_profit : ℝ) (discount : ℝ) : ℝ :=
  ((1 + no_discount_profit) * (1 - discount) - 1) * 100

/-- Theorem stating that with a 5% discount and a 150% no-discount profit, the profit is 137.5% -/
theorem discount_profit_calculation :
  profit_with_discount 1.5 0.05 = 137.5 := by
  sorry

#eval profit_with_discount 1.5 0.05

end NUMINAMATH_CALUDE_discount_profit_calculation_l2813_281301


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2813_281359

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through D(4,0)
def line_through_D (x y : ℝ) : Prop := ∃ t : ℝ, x = t*y + 4

-- Define points A and B as intersections of the line and parabola
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line_through_D A.1 A.2 ∧ line_through_D B.1 B.2 ∧
  A ≠ B

-- State the theorem
theorem parabola_line_intersection 
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  (A.1 * B.1 + A.2 * B.2 = 0) ∧  -- OA ⊥ OB
  (∀ S : ℝ, S = (1/2) * abs (A.1 * B.2 - A.2 * B.1) → S ≥ 16) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2813_281359


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l2813_281338

/-- Represents a cone with specific properties -/
structure Cone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone
  l : ℝ  -- length of the generatrix
  lateral_area_eq : π * r * l = 2 * π * r^2  -- lateral surface area is twice the base area
  volume_eq : (1/3) * π * r^2 * h = 9 * Real.sqrt 3 * π  -- volume is 9√3π

/-- Theorem stating that a cone with the given properties has a generatrix of length 6 -/
theorem cone_generatrix_length (c : Cone) : c.l = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l2813_281338


namespace NUMINAMATH_CALUDE_gcd_204_85_l2813_281350

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l2813_281350


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2813_281395

theorem polynomial_expansion (x : ℝ) : 
  (7 * x + 3) * (5 * x^2 + 2 * x + 4) = 35 * x^3 + 29 * x^2 + 34 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2813_281395


namespace NUMINAMATH_CALUDE_twice_x_minus_one_negative_l2813_281374

theorem twice_x_minus_one_negative (x : ℝ) : (2 * x - 1 < 0) ↔ (∃ y, y = 2 * x - 1 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_twice_x_minus_one_negative_l2813_281374


namespace NUMINAMATH_CALUDE_logan_driving_time_l2813_281306

/-- Proves that Logan drove for 5 hours given the conditions of the problem -/
theorem logan_driving_time (tamika_speed : ℝ) (tamika_time : ℝ) (logan_speed : ℝ) (distance_difference : ℝ)
  (h_tamika_speed : tamika_speed = 45)
  (h_tamika_time : tamika_time = 8)
  (h_logan_speed : logan_speed = 55)
  (h_distance_difference : distance_difference = 85) :
  (tamika_speed * tamika_time - distance_difference) / logan_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_logan_driving_time_l2813_281306


namespace NUMINAMATH_CALUDE_cricket_innings_problem_l2813_281383

theorem cricket_innings_problem (current_average : ℚ) (next_innings_runs : ℚ) (average_increase : ℚ) :
  current_average = 32 →
  next_innings_runs = 158 →
  average_increase = 6 →
  ∃ n : ℕ,
    n * current_average + next_innings_runs = (n + 1) * (current_average + average_increase) ∧
    n = 20 := by
  sorry

end NUMINAMATH_CALUDE_cricket_innings_problem_l2813_281383


namespace NUMINAMATH_CALUDE_king_arthur_advisors_l2813_281314

theorem king_arthur_advisors (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) : 
  let q := 1 - p
  let prob_correct_two_advisors := p^2 + 2*p*q*(1/2)
  prob_correct_two_advisors = p :=
by sorry

end NUMINAMATH_CALUDE_king_arthur_advisors_l2813_281314


namespace NUMINAMATH_CALUDE_percent_y_of_x_l2813_281377

theorem percent_y_of_x (x y : ℝ) (h : 0.5 * (x - y) = 0.3 * (x + y)) : y = 0.25 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_y_of_x_l2813_281377


namespace NUMINAMATH_CALUDE_quadratic_root_to_coefficient_l2813_281384

theorem quadratic_root_to_coefficient (m : ℚ) : 
  (∀ x : ℂ, 6 * x^2 + 5 * x + m = 0 ↔ x = (-5 + Complex.I * Real.sqrt 231) / 12 ∨ x = (-5 - Complex.I * Real.sqrt 231) / 12) →
  m = 32 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_to_coefficient_l2813_281384


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2813_281365

/-- A line in the form kx - y + 1 - 3k = 0 always passes through (3, 1) -/
theorem line_passes_through_point :
  ∀ (k : ℝ), (3 * k : ℝ) - 1 + 1 - 3 * k = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2813_281365


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_over_sqrt_x_l2813_281343

theorem sqrt_x_plus_one_over_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_over_sqrt_x_l2813_281343


namespace NUMINAMATH_CALUDE_factored_equation_difference_l2813_281340

theorem factored_equation_difference (p q : ℝ) : 
  (∃ (x : ℝ), x^2 - 6*x + q = 0 ∧ (x - p)^2 = 7) → p - q = 1 := by
  sorry

end NUMINAMATH_CALUDE_factored_equation_difference_l2813_281340


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l2813_281344

/-- The sum of 0.666... (repeating) and 0.222... (repeating) minus 0.444... (repeating) equals 4/9 -/
theorem repeating_decimal_sum_difference (x y z : ℚ) :
  x = 2/3 ∧ y = 2/9 ∧ z = 4/9 →
  x + y - z = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l2813_281344


namespace NUMINAMATH_CALUDE_basketball_team_enrollment_l2813_281341

theorem basketball_team_enrollment (total : ℕ) (math : ℕ) (both : ℕ) (physics : ℕ) : 
  total = 15 → math = 9 → both = 4 → physics = total - (math - both) → physics = 10 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_enrollment_l2813_281341


namespace NUMINAMATH_CALUDE_base_2_digit_difference_l2813_281331

theorem base_2_digit_difference : ∀ (n m : ℕ), n = 300 → m = 1500 → 
  (Nat.log 2 m + 1) - (Nat.log 2 n + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_2_digit_difference_l2813_281331


namespace NUMINAMATH_CALUDE_average_running_time_l2813_281352

theorem average_running_time (sixth_grade_time seventh_grade_time eighth_grade_time : ℝ)
  (sixth_to_eighth_ratio sixth_to_seventh_ratio : ℝ) :
  sixth_grade_time = 10 →
  seventh_grade_time = 18 →
  eighth_grade_time = 14 →
  sixth_to_eighth_ratio = 3 →
  sixth_to_seventh_ratio = 3/2 →
  let e := 1  -- Assuming 1 eighth grader for simplicity
  let sixth_count := e * sixth_to_eighth_ratio
  let seventh_count := sixth_count / sixth_to_seventh_ratio
  let eighth_count := e
  let total_time := sixth_grade_time * sixth_count + 
                    seventh_grade_time * seventh_count + 
                    eighth_grade_time * eighth_count
  let total_students := sixth_count + seventh_count + eighth_count
  total_time / total_students = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_average_running_time_l2813_281352


namespace NUMINAMATH_CALUDE_solution_range_l2813_281304

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem solution_range (a b c : ℝ) :
  f a b c 1.1 < 3 ∧ 
  f a b c 1.2 < 3 ∧ 
  f a b c 1.3 < 3 ∧ 
  f a b c 1.4 > 3 →
  ∃ x, 1.3 < x ∧ x < 1.4 ∧ f a b c x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2813_281304


namespace NUMINAMATH_CALUDE_resulting_polygon_sides_bound_resulting_polygon_sides_bound_even_l2813_281322

/-- Represents a convex n-gon with all diagonals drawn --/
structure ConvexNGonWithDiagonals (n : ℕ) where
  -- Add necessary fields here

/-- Represents a polygon resulting from the division of the n-gon by its diagonals --/
structure ResultingPolygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of sides of a resulting polygon --/
def num_sides (p : ResultingPolygon n) : ℕ := sorry

theorem resulting_polygon_sides_bound (n : ℕ) (ngon : ConvexNGonWithDiagonals n) 
  (p : ResultingPolygon n) : num_sides p ≤ n := by sorry

theorem resulting_polygon_sides_bound_even (n : ℕ) (ngon : ConvexNGonWithDiagonals n) 
  (p : ResultingPolygon n) (h : Even n) : num_sides p ≤ n - 1 := by sorry

end NUMINAMATH_CALUDE_resulting_polygon_sides_bound_resulting_polygon_sides_bound_even_l2813_281322


namespace NUMINAMATH_CALUDE_table_length_proof_l2813_281367

theorem table_length_proof (table_width : ℕ) (sheet_width sheet_height : ℕ) :
  table_width = 80 ∧ 
  sheet_width = 8 ∧ 
  sheet_height = 5 ∧ 
  (∃ n : ℕ, table_width = sheet_width + n ∧ n + 1 = sheet_width - sheet_height + 1) →
  ∃ table_length : ℕ, table_length = 77 ∧ table_length = table_width - (sheet_width - sheet_height) :=
by sorry

end NUMINAMATH_CALUDE_table_length_proof_l2813_281367


namespace NUMINAMATH_CALUDE_least_subtrahend_l2813_281363

theorem least_subtrahend (n : ℕ) (h : n = 427398) : 
  ∃! x : ℕ, x ≤ n ∧ 
  (∀ y : ℕ, y < x → ¬((n - y) % 17 = 0 ∧ (n - y) % 19 = 0 ∧ (n - y) % 31 = 0)) ∧
  (n - x) % 17 = 0 ∧ (n - x) % 19 = 0 ∧ (n - x) % 31 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_l2813_281363


namespace NUMINAMATH_CALUDE_light_travel_distance_l2813_281397

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we want to calculate the light travel distance for -/
def years : ℕ := 50

/-- Theorem stating that the distance light travels in 50 years
    is equal to 293.5 × 10^12 miles -/
theorem light_travel_distance : 
  (light_year_distance * years : ℝ) = 293.5 * (10 ^ 12) := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l2813_281397


namespace NUMINAMATH_CALUDE_XZ_length_l2813_281355

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := ℝ × ℝ

-- Define the radius of the circle
def radius : ℝ := 7

-- Define the points on the circle
def X : Point := sorry
def Y : Point := sorry
def Z : Point := sorry
def W : Point := sorry

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- State the conditions
axiom on_circle_X : distance X (0, 0) = radius
axiom on_circle_Y : distance Y (0, 0) = radius
axiom XY_distance : distance X Y = 8
axiom Z_midpoint_arc : sorry  -- Z is the midpoint of the minor arc XY
axiom W_midpoint_XZ : distance X W = distance W Z
axiom YW_distance : distance Y W = 6

-- State the theorem to be proved
theorem XZ_length : distance X Z = 8 := by sorry

end NUMINAMATH_CALUDE_XZ_length_l2813_281355


namespace NUMINAMATH_CALUDE_range_of_3a_minus_2b_l2813_281339

theorem range_of_3a_minus_2b (a b : ℝ) 
  (h1 : -3 ≤ a + b ∧ a + b ≤ 2) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 4) : 
  -4 ≤ 3*a - 2*b ∧ 3*a - 2*b ≤ 11 := by sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_2b_l2813_281339


namespace NUMINAMATH_CALUDE_equation_solution_l2813_281325

theorem equation_solution (x : ℝ) : 
  (x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2) ↔ (2*x + 1)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2813_281325


namespace NUMINAMATH_CALUDE_company_average_salary_l2813_281394

/-- Calculate the average salary for a company given the number of managers and associates, and their respective average salaries. -/
theorem company_average_salary
  (num_managers : ℕ)
  (num_associates : ℕ)
  (avg_salary_managers : ℝ)
  (avg_salary_associates : ℝ)
  (h_managers : num_managers = 15)
  (h_associates : num_associates = 75)
  (h_salary_managers : avg_salary_managers = 90000)
  (h_salary_associates : avg_salary_associates = 30000) :
  let total_salary := num_managers * avg_salary_managers + num_associates * avg_salary_associates
  let total_employees := num_managers + num_associates
  total_salary / total_employees = 40000 := by
  sorry

#check company_average_salary

end NUMINAMATH_CALUDE_company_average_salary_l2813_281394


namespace NUMINAMATH_CALUDE_polygon_25_diagonals_l2813_281336

/-- The number of diagonals in a convex polygon with n sides,
    where each vertex connects only to vertices at least k places apart. -/
def diagonals (n : ℕ) (k : ℕ) : ℕ :=
  (n * (n - (2*k + 1))) / 2

/-- Theorem: A convex 25-sided polygon where each vertex connects only to
    vertices at least 2 places apart in sequence has 250 diagonals. -/
theorem polygon_25_diagonals :
  diagonals 25 2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_polygon_25_diagonals_l2813_281336


namespace NUMINAMATH_CALUDE_tom_tim_ratio_l2813_281371

/-- The typing speeds of Tim and Tom, and their relationship -/
structure TypingSpeed where
  tim : ℝ
  tom : ℝ
  total_normal : tim + tom = 15
  total_increased : tim + 1.6 * tom = 18

/-- The ratio of Tom's normal typing speed to Tim's is 1:2 -/
theorem tom_tim_ratio (s : TypingSpeed) : s.tom / s.tim = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tom_tim_ratio_l2813_281371


namespace NUMINAMATH_CALUDE_org_satisfies_conditions_l2813_281388

/-- Represents an organization with committees and members -/
structure Organization where
  num_committees : ℕ
  num_members : ℕ
  member_committee_count : ℕ
  unique_pair_member : Prop

/-- The specific organization described in the problem -/
def problem_org : Organization :=
  { num_committees := 5,
    num_members := 10,
    member_committee_count := 2,
    unique_pair_member := true }

/-- Theorem stating that the organization satisfies the given conditions -/
theorem org_satisfies_conditions (org : Organization) :
  org.num_committees = 5 ∧
  org.member_committee_count = 2 ∧
  org.unique_pair_member →
  org.num_members = 10 := by
  sorry

#check org_satisfies_conditions problem_org

end NUMINAMATH_CALUDE_org_satisfies_conditions_l2813_281388


namespace NUMINAMATH_CALUDE_unique_number_with_remainders_l2813_281369

theorem unique_number_with_remainders : ∃! m : ℤ,
  (m % 13 = 12) ∧
  (m % 12 = 11) ∧
  (m % 11 = 10) ∧
  (m % 10 = 9) ∧
  (m % 9 = 8) ∧
  (m % 8 = 7) ∧
  (m % 7 = 6) ∧
  (m % 6 = 5) ∧
  (m % 5 = 4) ∧
  (m % 4 = 3) ∧
  (m % 3 = 2) ∧
  m = 360359 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_remainders_l2813_281369


namespace NUMINAMATH_CALUDE_douglas_weight_is_52_l2813_281326

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- The difference in weight between Anne and Douglas in pounds -/
def weight_difference : ℕ := 15

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := anne_weight - weight_difference

/-- Theorem stating Douglas's weight -/
theorem douglas_weight_is_52 : douglas_weight = 52 := by
  sorry

end NUMINAMATH_CALUDE_douglas_weight_is_52_l2813_281326


namespace NUMINAMATH_CALUDE_square_difference_l2813_281354

theorem square_difference : (39 : ℕ)^2 = (40 : ℕ)^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2813_281354


namespace NUMINAMATH_CALUDE_age_difference_l2813_281382

theorem age_difference (a b : ℕ) : 
  (a > 0) → 
  (b > 0) → 
  (a < 100) → 
  (b < 100) → 
  (a = 10 * (b % 10) + (a / 10)) → 
  (b = 10 * (a % 10) + (b / 10)) → 
  (a + 7 = 3 * (b + 7)) → 
  (a - b = 45) := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2813_281382


namespace NUMINAMATH_CALUDE_hiker_distance_l2813_281390

/-- Hiker's walking problem -/
theorem hiker_distance 
  (x y : ℝ) 
  (h1 : x * y = 18) 
  (D2 : ℝ := (y - 1) * (x + 1))
  (D3 : ℝ := 5 * 3)
  (D_total : ℝ := 18 + D2 + D3)
  (T_total : ℝ := y + (y - 1) + 3)
  (Z : ℝ)
  (h2 : Z = D_total / T_total) :
  D_total = x * y + y - x + 32 := by
sorry

end NUMINAMATH_CALUDE_hiker_distance_l2813_281390


namespace NUMINAMATH_CALUDE_angle_sum_pi_half_l2813_281307

theorem angle_sum_pi_half (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) : 
  α + β = π/2 := by sorry

end NUMINAMATH_CALUDE_angle_sum_pi_half_l2813_281307


namespace NUMINAMATH_CALUDE_two_cos_45_equals_sqrt_2_l2813_281302

theorem two_cos_45_equals_sqrt_2 : 2 * Real.cos (π / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_two_cos_45_equals_sqrt_2_l2813_281302


namespace NUMINAMATH_CALUDE_chef_total_plates_l2813_281330

theorem chef_total_plates (lobster_rolls spicy_hot_noodles seafood_noodles : ℕ) 
  (h1 : lobster_rolls = 25)
  (h2 : spicy_hot_noodles = 14)
  (h3 : seafood_noodles = 16) :
  lobster_rolls + spicy_hot_noodles + seafood_noodles = 55 := by
  sorry

end NUMINAMATH_CALUDE_chef_total_plates_l2813_281330


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_1728_l2813_281337

theorem closest_integer_to_cube_root_1728 : 
  ∀ n : ℤ, |n - (1728 : ℝ)^(1/3)| ≥ |12 - (1728 : ℝ)^(1/3)| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_1728_l2813_281337


namespace NUMINAMATH_CALUDE_xy_positive_l2813_281332

theorem xy_positive (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ 0) : x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_positive_l2813_281332


namespace NUMINAMATH_CALUDE_jello_bathtub_cost_l2813_281335

/-- Calculate the total cost of filling a bathtub with jello mix -/
theorem jello_bathtub_cost :
  let bathtub_capacity : ℝ := 6  -- cubic feet
  let cubic_foot_to_gallon : ℝ := 7.5
  let gallon_weight : ℝ := 8  -- pounds
  let jello_per_pound : ℝ := 1.5  -- tablespoons
  let red_jello_cost : ℝ := 0.5  -- dollars per tablespoon
  let blue_jello_cost : ℝ := 0.4  -- dollars per tablespoon
  let green_jello_cost : ℝ := 0.6  -- dollars per tablespoon
  let red_jello_ratio : ℝ := 0.6
  let blue_jello_ratio : ℝ := 0.3
  let green_jello_ratio : ℝ := 0.1

  let total_water_weight := bathtub_capacity * cubic_foot_to_gallon * gallon_weight
  let total_jello_needed := total_water_weight * jello_per_pound
  let red_jello_amount := total_jello_needed * red_jello_ratio
  let blue_jello_amount := total_jello_needed * blue_jello_ratio
  let green_jello_amount := total_jello_needed * green_jello_ratio

  let total_cost := red_jello_amount * red_jello_cost +
                    blue_jello_amount * blue_jello_cost +
                    green_jello_amount * green_jello_cost

  total_cost = 259.2 := by sorry

end NUMINAMATH_CALUDE_jello_bathtub_cost_l2813_281335


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_l2813_281319

theorem sqrt_2_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 2 = (p : ℚ) / q := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_l2813_281319


namespace NUMINAMATH_CALUDE_intersection_chord_length_l2813_281346

theorem intersection_chord_length (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3 → (x - 3)^2 + (y - 2)^2 = 4 → 
    ∃ M N : ℝ × ℝ, 
      (M.1 - 3)^2 + (M.2 - 2)^2 = 4 ∧ 
      (N.1 - 3)^2 + (N.2 - 2)^2 = 4 ∧ 
      M.2 = k * M.1 + 3 ∧ 
      N.2 = k * N.1 + 3 ∧ 
      (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) → 
  -3/4 ≤ k ∧ k ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l2813_281346


namespace NUMINAMATH_CALUDE_problem_statement_l2813_281373

theorem problem_statement (a b : ℝ) 
  (h1 : 0 < (1 : ℝ) / a) 
  (h2 : (1 : ℝ) / a < (1 : ℝ) / b) 
  (h3 : (1 : ℝ) / b < 1) 
  (h4 : Real.log a * Real.log b = 1) : 
  (2 : ℝ) ^ a > (2 : ℝ) ^ b ∧ 
  a * b > Real.exp 2 ∧ 
  Real.exp (a - b) > a / b := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2813_281373


namespace NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l2813_281321

-- Define the number of faces on a die
def die_faces : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 4

-- Define the possible common differences
def common_differences : List ℕ := [1, 2]

-- Define a function to calculate the total number of outcomes
def total_outcomes : ℕ := die_faces ^ num_dice

-- Define a function to calculate the favorable outcomes
def favorable_outcomes : ℕ := sorry

-- The main theorem
theorem dice_arithmetic_progression_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l2813_281321


namespace NUMINAMATH_CALUDE_quadratic_equation_theorem_l2813_281347

/-- Quadratic equation parameters -/
structure QuadraticParams where
  m : ℝ

/-- Roots of the quadratic equation -/
structure QuadraticRoots where
  x1 : ℝ
  x2 : ℝ

/-- Theorem about the quadratic equation x^2 - (2m+3)x + m^2 + 2 = 0 -/
theorem quadratic_equation_theorem (p : QuadraticParams) (r : QuadraticRoots) : 
  /- The equation has real roots if and only if m ≥ -1/12 -/
  (∃ (x : ℝ), x^2 - (2*p.m + 3)*x + p.m^2 + 2 = 0) ↔ p.m ≥ -1/12 ∧
  
  /- If x1 and x2 are the roots of the equation and satisfy the given condition, then m = 13 -/
  (r.x1^2 - (2*p.m + 3)*r.x1 + p.m^2 + 2 = 0 ∧
   r.x2^2 - (2*p.m + 3)*r.x2 + p.m^2 + 2 = 0 ∧
   r.x1^2 + r.x2^2 = 3*r.x1*r.x2 - 14) →
  p.m = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_theorem_l2813_281347


namespace NUMINAMATH_CALUDE_last_home_game_score_l2813_281342

theorem last_home_game_score (H : ℕ) : 
  (H = 2 * (H / 2)) →  -- Last home game score is twice the first away game
  (∃ second_away : ℕ, second_away = H / 2 + 18) →  -- Second away game score
  (∃ third_away : ℕ, third_away = (H / 2 + 18) + 2) →  -- Third away game score
  ((5 * H) / 2 + 38 + 55 = 4 * H) →  -- Cumulative points condition
  H = 62 := by
sorry

end NUMINAMATH_CALUDE_last_home_game_score_l2813_281342


namespace NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l2813_281329

/-- The function f(x) defined as x^2 - 2ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

/-- The theorem stating the equivalence between the condition and the range of a -/
theorem f_geq_a_iff_a_in_range (a : ℝ) :
  (∀ x ≥ -1, f a x ≥ a) ↔ a ∈ Set.Icc (-3) 1 :=
by sorry

#check f_geq_a_iff_a_in_range

end NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l2813_281329


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2813_281393

theorem rectangle_perimeter (x y z : ℝ) : 
  x + y + z = 75 →
  x > 0 → y > 0 → z > 0 →
  2 * (x + 75) = (2 * (y + 75) + 2 * (z + 75)) / 2 →
  2 * (x + 75) = 20 * 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2813_281393


namespace NUMINAMATH_CALUDE_six_times_two_minus_three_l2813_281387

theorem six_times_two_minus_three : 6 * 2 - 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_six_times_two_minus_three_l2813_281387


namespace NUMINAMATH_CALUDE_sarah_candy_consumption_l2813_281398

theorem sarah_candy_consumption 
  (candy_from_neighbors : ℕ)
  (candy_from_sister : ℕ)
  (days_lasted : ℕ)
  (h1 : candy_from_neighbors = 66)
  (h2 : candy_from_sister = 15)
  (h3 : days_lasted = 9)
  (h4 : days_lasted > 0) :
  (candy_from_neighbors + candy_from_sister) / days_lasted = 9 :=
by sorry

end NUMINAMATH_CALUDE_sarah_candy_consumption_l2813_281398


namespace NUMINAMATH_CALUDE_seating_arrangements_l2813_281317

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def num_players : ℕ := 10
def num_teams : ℕ := 4
def team_sizes : List ℕ := [3, 4, 2, 1]

theorem seating_arrangements :
  (factorial num_teams) * (team_sizes.map factorial).prod = 6912 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2813_281317


namespace NUMINAMATH_CALUDE_determine_phi_l2813_281356

-- Define the functions and constants
noncomputable def ω : ℝ := 2
noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (ω * x + φ)
noncomputable def g (x : ℝ) := Real.cos (ω * x)

-- State the theorem
theorem determine_phi :
  (ω > 0) →
  (∀ φ, |φ| < π / 2 →
    (∀ x, f x φ = f (x + π) φ) →
    (∀ x, f (x - 2*π/3) φ = g x)) →
  ∃ φ, φ = -π / 6 :=
by sorry

end NUMINAMATH_CALUDE_determine_phi_l2813_281356


namespace NUMINAMATH_CALUDE_taller_tree_height_l2813_281320

/-- Given two trees where one is 20 feet taller than the other and their heights
    are in the ratio 2:3, prove that the height of the taller tree is 60 feet. -/
theorem taller_tree_height (h : ℝ) (h_pos : h > 0) : 
  (h - 20) / h = 2 / 3 → h = 60 := by
  sorry

end NUMINAMATH_CALUDE_taller_tree_height_l2813_281320


namespace NUMINAMATH_CALUDE_brother_statement_contradiction_l2813_281362

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

-- Define the brother's behavior
structure Brother where
  lying_days : Set Day
  today : Day

-- Define the brother's statement
def brother_statement (b : Brother) : Prop :=
  b.today ∈ b.lying_days

-- Theorem: The brother's statement leads to a contradiction
theorem brother_statement_contradiction (b : Brother) :
  ¬(brother_statement b ↔ ¬(brother_statement b)) :=
sorry

end NUMINAMATH_CALUDE_brother_statement_contradiction_l2813_281362


namespace NUMINAMATH_CALUDE_johns_remaining_money_l2813_281380

/-- Calculates the remaining money after John's pizza and drink purchase. -/
def remaining_money (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_spent := 2 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Proves that John's remaining money is equal to 50 - 8q. -/
theorem johns_remaining_money (q : ℝ) : remaining_money q = 50 - 8 * q := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l2813_281380


namespace NUMINAMATH_CALUDE_food_additives_percentage_l2813_281375

/-- Represents the budget allocation for a research category -/
structure BudgetAllocation where
  percentage : ℝ
  degrees : ℝ

/-- Represents the total budget and its allocations -/
structure Budget where
  total_degrees : ℝ
  microphotonics : BudgetAllocation
  home_electronics : BudgetAllocation
  genetically_modified_microorganisms : BudgetAllocation
  industrial_lubricants : BudgetAllocation
  basic_astrophysics : BudgetAllocation
  food_additives : BudgetAllocation

/-- The Megatech Corporation's research and development budget -/
def megatech_budget : Budget := {
  total_degrees := 360
  microphotonics := { percentage := 14, degrees := 0 }
  home_electronics := { percentage := 24, degrees := 0 }
  genetically_modified_microorganisms := { percentage := 19, degrees := 0 }
  industrial_lubricants := { percentage := 8, degrees := 0 }
  basic_astrophysics := { percentage := 0, degrees := 72 }
  food_additives := { percentage := 0, degrees := 0 }
}

/-- Theorem: The percentage of the budget allocated to food additives is 15% -/
theorem food_additives_percentage : megatech_budget.food_additives.percentage = 15 := by
  sorry


end NUMINAMATH_CALUDE_food_additives_percentage_l2813_281375


namespace NUMINAMATH_CALUDE_distinct_circular_arrangements_l2813_281370

/-- The number of distinct circular arrangements of girls and boys -/
def circularArrangements (girls boys : ℕ) : ℕ :=
  (Nat.factorial 16 * Nat.factorial 25) / Nat.factorial 9

/-- Theorem stating the number of distinct circular arrangements -/
theorem distinct_circular_arrangements :
  circularArrangements 8 25 = (Nat.factorial 16 * Nat.factorial 25) / Nat.factorial 9 :=
by sorry

end NUMINAMATH_CALUDE_distinct_circular_arrangements_l2813_281370


namespace NUMINAMATH_CALUDE_total_book_pages_l2813_281353

def book_pages : ℕ → ℕ
| 1 => 25
| 2 => 2 * book_pages 1
| 3 => 2 * book_pages 2
| 4 => 10
| _ => 0

def pages_written : ℕ := book_pages 1 + book_pages 2 + book_pages 3 + book_pages 4

def remaining_pages : ℕ := 315

theorem total_book_pages : pages_written + remaining_pages = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_book_pages_l2813_281353


namespace NUMINAMATH_CALUDE_function_m_minus_n_l2813_281334

def M (m : ℕ) : Set ℕ := {1, 2, 3, m}
def N (n : ℕ) : Set ℕ := {4, 7, n^4, n^2 + 3*n}

def f (x : ℕ) : ℕ := 3*x + 1

theorem function_m_minus_n (m n : ℕ) : 
  (∀ x ∈ M m, f x ∈ N n) → m - n = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_m_minus_n_l2813_281334


namespace NUMINAMATH_CALUDE_shawn_pebble_groups_l2813_281316

theorem shawn_pebble_groups :
  let total_pebbles : ℕ := 40
  let red_pebbles : ℕ := 9
  let blue_pebbles : ℕ := 13
  let remaining_pebbles : ℕ := total_pebbles - red_pebbles - blue_pebbles
  let blue_yellow_diff : ℕ := 7
  let yellow_pebbles : ℕ := blue_pebbles - blue_yellow_diff
  let num_colors : ℕ := 3  -- purple, yellow, and green
  ∃ (num_groups : ℕ),
    num_groups > 0 ∧
    num_groups ∣ remaining_pebbles ∧
    num_groups % num_colors = 0 ∧
    remaining_pebbles / num_groups = yellow_pebbles ∧
    num_groups = 3
  := by sorry

end NUMINAMATH_CALUDE_shawn_pebble_groups_l2813_281316


namespace NUMINAMATH_CALUDE_power_ranger_stickers_l2813_281345

theorem power_ranger_stickers (box1 box2 total : ℕ) : 
  box1 = 23 →
  box2 = box1 + 12 →
  total = box1 + box2 →
  total = 58 := by sorry

end NUMINAMATH_CALUDE_power_ranger_stickers_l2813_281345


namespace NUMINAMATH_CALUDE_product_xy_is_zero_l2813_281351

theorem product_xy_is_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_is_zero_l2813_281351


namespace NUMINAMATH_CALUDE_sqrt_a_squared_plus_one_is_quadratic_radical_l2813_281300

/-- A function is a quadratic radical if it involves a square root and its radicand is non-negative for all real inputs. -/
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g x ≥ 0) ∧ (∀ x, f x = Real.sqrt (g x))

/-- The function f(a) = √(a² + 1) is a quadratic radical. -/
theorem sqrt_a_squared_plus_one_is_quadratic_radical :
  is_quadratic_radical (fun a : ℝ ↦ Real.sqrt (a^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_a_squared_plus_one_is_quadratic_radical_l2813_281300


namespace NUMINAMATH_CALUDE_stadium_entrance_exit_plans_stadium_plans_eq_35_l2813_281357

/-- The number of possible entrance and exit plans for a student at a school stadium. -/
theorem stadium_entrance_exit_plans : ℕ :=
  let south_gates : ℕ := 4
  let north_gates : ℕ := 3
  let west_gates : ℕ := 2
  let entrance_options : ℕ := south_gates + north_gates
  let exit_options : ℕ := west_gates + north_gates
  entrance_options * exit_options

/-- Proof that the number of possible entrance and exit plans is 35. -/
theorem stadium_plans_eq_35 : stadium_entrance_exit_plans = 35 := by
  sorry

end NUMINAMATH_CALUDE_stadium_entrance_exit_plans_stadium_plans_eq_35_l2813_281357


namespace NUMINAMATH_CALUDE_polygon_sides_doubled_l2813_281310

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: If doubling the sides of a polygon increases the diagonals by 45, the polygon has 6 sides -/
theorem polygon_sides_doubled (n : ℕ) (h : n > 3) :
  diagonals (2 * n) - diagonals n = 45 → n = 6 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_doubled_l2813_281310


namespace NUMINAMATH_CALUDE_f_equals_f_inv_at_zero_l2813_281311

/-- The function f(x) = 3x^2 - 6x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

/-- The inverse function of f -/
noncomputable def f_inv (x : ℝ) : ℝ := 1 + Real.sqrt ((1 + x) / 3)

/-- Theorem stating that f(0) = f⁻¹(0) -/
theorem f_equals_f_inv_at_zero : f 0 = f_inv 0 := by
  sorry

end NUMINAMATH_CALUDE_f_equals_f_inv_at_zero_l2813_281311


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2813_281323

variable (a b : ℝ)

theorem problem_1 : (a - b)^2 - (2*a + b)*(b - 2*a) = 5*a^2 - 2*a*b := by sorry

theorem problem_2 : (3 / (a + 1) - a + 1) / ((a^2 + 4*a + 4) / (a + 1)) = (2 - a) / (a + 2) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2813_281323


namespace NUMINAMATH_CALUDE_max_sum_hexagonal_prism_with_pyramid_l2813_281364

/-- Represents a three-dimensional geometric shape --/
structure Shape3D where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- A hexagonal prism --/
def hexagonal_prism : Shape3D :=
  { faces := 8, vertices := 12, edges := 18 }

/-- Adds a pyramid to a hexagonal face of the prism --/
def add_pyramid_to_hexagonal_face (s : Shape3D) : Shape3D :=
  { faces := s.faces + 5,
    vertices := s.vertices + 1,
    edges := s.edges + 6 }

/-- Adds a pyramid to a rectangular face of the prism --/
def add_pyramid_to_rectangular_face (s : Shape3D) : Shape3D :=
  { faces := s.faces + 3,
    vertices := s.vertices + 1,
    edges := s.edges + 4 }

/-- Calculates the sum of faces, vertices, and edges --/
def sum_features (s : Shape3D) : ℕ :=
  s.faces + s.vertices + s.edges

/-- Theorem: The maximum sum of exterior faces, vertices, and edges 
    when adding a pyramid to a hexagonal prism is 50 --/
theorem max_sum_hexagonal_prism_with_pyramid : 
  max 
    (sum_features (add_pyramid_to_hexagonal_face hexagonal_prism))
    (sum_features (add_pyramid_to_rectangular_face hexagonal_prism)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_hexagonal_prism_with_pyramid_l2813_281364


namespace NUMINAMATH_CALUDE_repeating_decimal_four_eight_equals_forty_four_ninths_l2813_281378

def repeating_decimal (whole_part : ℕ) (repeating_part : ℕ) : ℚ :=
  whole_part + repeating_part / 99

theorem repeating_decimal_four_eight_equals_forty_four_ninths :
  repeating_decimal 4 8 = 44 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_four_eight_equals_forty_four_ninths_l2813_281378
