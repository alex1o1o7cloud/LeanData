import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l2746_274624

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  geometric_sequence a → a 2 = 4 → a 4 = 2 → a 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l2746_274624


namespace NUMINAMATH_CALUDE_johns_new_total_capacity_l2746_274653

/-- Represents the lifting capacities of a weightlifter -/
structure LiftingCapacities where
  cleanAndJerk : ℝ
  snatch : ℝ

/-- Calculates the new lifting capacities after improvement -/
def improvedCapacities (initial : LiftingCapacities) : LiftingCapacities :=
  { cleanAndJerk := initial.cleanAndJerk * 2,
    snatch := initial.snatch * 1.8 }

/-- Calculates the total lifting capacity -/
def totalCapacity (capacities : LiftingCapacities) : ℝ :=
  capacities.cleanAndJerk + capacities.snatch

/-- John's initial lifting capacities -/
def johnsInitialCapacities : LiftingCapacities :=
  { cleanAndJerk := 80,
    snatch := 50 }

theorem johns_new_total_capacity :
  totalCapacity (improvedCapacities johnsInitialCapacities) = 250 := by
  sorry


end NUMINAMATH_CALUDE_johns_new_total_capacity_l2746_274653


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2746_274672

theorem cube_sum_and_reciprocal (x R S : ℝ) (hx : x ≠ 0) :
  (x + 1 / x = R) → (x^3 + 1 / x^3 = S) → (S = R^3 - 3 * R) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2746_274672


namespace NUMINAMATH_CALUDE_intersection_area_of_circles_l2746_274689

/-- The area of intersection of two circles with radius 4 and centers 4/α apart -/
theorem intersection_area_of_circles (α : ℝ) (h : α = 1/2) : 
  let r : ℝ := 4
  let d : ℝ := 4/α
  let β : ℝ := (2*r)^2 - 2*(π*r^2/2)
  β = 64 - 16*π := by sorry

end NUMINAMATH_CALUDE_intersection_area_of_circles_l2746_274689


namespace NUMINAMATH_CALUDE_three_positions_from_eight_l2746_274615

/-- The number of ways to choose 3 distinct positions from a group of n people. -/
def choose_three_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The theorem stating that choosing 3 distinct positions from 8 people results in 336 ways. -/
theorem three_positions_from_eight : choose_three_positions 8 = 336 := by
  sorry

end NUMINAMATH_CALUDE_three_positions_from_eight_l2746_274615


namespace NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l2746_274633

theorem smallest_integer_quadratic_inequality :
  ∃ (n : ℤ), (∀ (m : ℤ), m^2 - 9*m + 18 ≥ 0 → n ≤ m) ∧ (n^2 - 9*n + 18 ≥ 0) ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l2746_274633


namespace NUMINAMATH_CALUDE_compare_exponentials_l2746_274647

theorem compare_exponentials :
  (4 : ℝ) ^ (1/4) > (5 : ℝ) ^ (1/5) ∧
  (5 : ℝ) ^ (1/5) > (16 : ℝ) ^ (1/16) ∧
  (16 : ℝ) ^ (1/16) > (25 : ℝ) ^ (1/25) :=
by sorry

end NUMINAMATH_CALUDE_compare_exponentials_l2746_274647


namespace NUMINAMATH_CALUDE_tangent_slope_at_negative_one_l2746_274637

def curve (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2

def tangent_slope (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2

theorem tangent_slope_at_negative_one (a : ℝ) :
  tangent_slope a (-1) = Real.tan (π/4) → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_negative_one_l2746_274637


namespace NUMINAMATH_CALUDE_motorcycle_car_profit_difference_l2746_274646

/-- Represents the production and sales data for a vehicle type -/
structure VehicleProduction where
  materialCost : ℕ
  quantity : ℕ
  price : ℕ

/-- Calculates the profit for a given vehicle production -/
def profit (v : VehicleProduction) : ℤ :=
  (v.quantity * v.price : ℤ) - v.materialCost

/-- Proves that the difference in profit between motorcycle and car production is $50 -/
theorem motorcycle_car_profit_difference 
  (car : VehicleProduction)
  (motorcycle : VehicleProduction)
  (h_car : car = { materialCost := 100, quantity := 4, price := 50 })
  (h_motorcycle : motorcycle = { materialCost := 250, quantity := 8, price := 50 }) :
  profit motorcycle - profit car = 50 := by
  sorry

#eval profit { materialCost := 250, quantity := 8, price := 50 } - 
      profit { materialCost := 100, quantity := 4, price := 50 }

end NUMINAMATH_CALUDE_motorcycle_car_profit_difference_l2746_274646


namespace NUMINAMATH_CALUDE_dot_product_zero_on_diagonal_l2746_274659

/-- A square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_unit_square : A.1 + 1 = B.1 ∧ A.2 + 1 = B.2 ∧
                   C.1 = B.1 ∧ C.2 = B.2 + 1 ∧
                   D.1 = A.1 ∧ D.2 = C.2

/-- A point on the diagonal AC of a unit square -/
def PointOnDiagonal (square : UnitSquare) : Type :=
  {P : ℝ × ℝ // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    P.1 = square.A.1 + t * (square.C.1 - square.A.1) ∧
    P.2 = square.A.2 + t * (square.C.2 - square.A.2)}

/-- Vector from point A to point P -/
def vec_AP (square : UnitSquare) (P : PointOnDiagonal square) : ℝ × ℝ :=
  (P.val.1 - square.A.1, P.val.2 - square.A.2)

/-- Vector from point P to point B -/
def vec_PB (square : UnitSquare) (P : PointOnDiagonal square) : ℝ × ℝ :=
  (square.B.1 - P.val.1, square.B.2 - P.val.2)

/-- Vector from point P to point D -/
def vec_PD (square : UnitSquare) (P : PointOnDiagonal square) : ℝ × ℝ :=
  (square.D.1 - P.val.1, square.D.2 - P.val.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_zero_on_diagonal (square : UnitSquare) (P : PointOnDiagonal square) :
  dot_product (vec_AP square P) (vec_PB square P + vec_PD square P) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_zero_on_diagonal_l2746_274659


namespace NUMINAMATH_CALUDE_inequality_proof_l2746_274652

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * b) - Real.sqrt (c * d)| ≤ 2) : 
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2746_274652


namespace NUMINAMATH_CALUDE_smallest_four_digit_negative_congruent_to_one_mod_37_l2746_274699

theorem smallest_four_digit_negative_congruent_to_one_mod_37 :
  ∀ n : ℤ, n < 0 ∧ n ≥ -9999 ∧ n ≡ 1 [ZMOD 37] → n ≥ -1034 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_negative_congruent_to_one_mod_37_l2746_274699


namespace NUMINAMATH_CALUDE_proposition_analysis_l2746_274658

def p : Prop := 6 ∣ 12
def q : Prop := 6 ∣ 24

theorem proposition_analysis :
  (p ∨ q) ∧ (p ∧ q) ∧ (¬¬p) := by sorry

end NUMINAMATH_CALUDE_proposition_analysis_l2746_274658


namespace NUMINAMATH_CALUDE_cubic_function_through_point_l2746_274613

/-- Given a function f(x) = ax³ - 3x that passes through the point (-1, 4), prove that a = -1 --/
theorem cubic_function_through_point (a : ℝ) : 
  (fun x : ℝ => a * x^3 - 3*x) (-1) = 4 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_through_point_l2746_274613


namespace NUMINAMATH_CALUDE_total_food_service_employees_l2746_274630

/-- Represents the number of employees trained for each restaurant combination --/
structure RestaurantTraining where
  b : ℕ  -- Trained for family buffet only
  d : ℕ  -- Trained for dining room only
  s : ℕ  -- Trained for snack bar only
  bd : ℕ -- Trained for family buffet and dining room
  bs : ℕ -- Trained for family buffet and snack bar
  ds : ℕ -- Trained for dining room and snack bar
  bds : ℕ -- Trained for all three restaurants

/-- Calculates the total number of employees trained for each restaurant --/
def total_per_restaurant (rt : RestaurantTraining) : (ℕ × ℕ × ℕ) :=
  (rt.b + rt.bd + rt.bs + rt.bds,
   rt.d + rt.bd + rt.ds + rt.bds,
   rt.s + rt.bs + rt.ds + rt.bds)

/-- Calculates the total number of food service employees --/
def total_employees (rt : RestaurantTraining) : ℕ :=
  rt.b + rt.d + rt.s + rt.bd + rt.bs + rt.ds + rt.bds

/-- Theorem stating the total number of food service employees --/
theorem total_food_service_employees :
  ∀ (rt : RestaurantTraining),
    total_per_restaurant rt = (15, 18, 12) →
    rt.bd + rt.bs + rt.ds = 4 →
    rt.bds = 1 →
    total_employees rt = 39 := by
  sorry


end NUMINAMATH_CALUDE_total_food_service_employees_l2746_274630


namespace NUMINAMATH_CALUDE_rectangle_count_6x5_grid_l2746_274623

/-- Represents a grid of lines in a coordinate plane -/
structure Grid :=
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Represents a point in a 2D coordinate plane -/
structure Point :=
  (x : ℕ)
  (y : ℕ)

/-- Counts the number of ways to form a rectangle enclosing a given point -/
def count_rectangles (g : Grid) (p : Point) : ℕ :=
  sorry

/-- Theorem stating the number of ways to form a rectangle enclosing (3, 4) in a 6x5 grid -/
theorem rectangle_count_6x5_grid :
  let g : Grid := ⟨6, 5⟩
  let p : Point := ⟨3, 4⟩
  count_rectangles g p = 24 :=
sorry

end NUMINAMATH_CALUDE_rectangle_count_6x5_grid_l2746_274623


namespace NUMINAMATH_CALUDE_coefficient_of_y_squared_l2746_274663

theorem coefficient_of_y_squared (a : ℝ) : 
  (∀ y : ℝ, a * y^2 - 8 * y + 55 = 59) → 
  (∃ y : ℝ, y = 2) → 
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_y_squared_l2746_274663


namespace NUMINAMATH_CALUDE_interesting_coeffs_of_product_l2746_274642

/-- A real number is interesting if it can be expressed as a + b√2 where a and b are integers -/
def interesting (r : ℝ) : Prop :=
  ∃ (a b : ℤ), r = a + b * Real.sqrt 2

/-- A polynomial with interesting coefficients -/
def interesting_poly (p : Polynomial ℝ) : Prop :=
  ∀ i, interesting (p.coeff i)

/-- The main theorem -/
theorem interesting_coeffs_of_product
  (A B Q : Polynomial ℝ)
  (hA : interesting_poly A)
  (hB : interesting_poly B)
  (hB_const : B.coeff 0 = 1)
  (hABQ : A = B * Q) :
  interesting_poly Q :=
sorry

end NUMINAMATH_CALUDE_interesting_coeffs_of_product_l2746_274642


namespace NUMINAMATH_CALUDE_intersection_size_l2746_274675

/-- Given a finite universe U and two subsets A and B, 
    this theorem calculates the size of their intersection. -/
theorem intersection_size 
  (U A B : Finset ℕ) 
  (h1 : A ⊆ U) 
  (h2 : B ⊆ U) 
  (h3 : Finset.card U = 215)
  (h4 : Finset.card A = 170)
  (h5 : Finset.card B = 142)
  (h6 : Finset.card (U \ (A ∪ B)) = 38) :
  Finset.card (A ∩ B) = 135 := by
sorry

end NUMINAMATH_CALUDE_intersection_size_l2746_274675


namespace NUMINAMATH_CALUDE_cube_sum_geq_triple_product_l2746_274606

theorem cube_sum_geq_triple_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 ≥ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_triple_product_l2746_274606


namespace NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_2_mod_17_l2746_274638

theorem smallest_five_digit_negative_congruent_to_2_mod_17 : 
  ∃ (n : ℤ), 
    n = -10011 ∧ 
    n ≡ 2 [ZMOD 17] ∧ 
    n < 0 ∧ 
    -99999 ≤ n ∧ 
    ∀ (m : ℤ), (m ≡ 2 [ZMOD 17] ∧ m < 0 ∧ -99999 ≤ m) → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_2_mod_17_l2746_274638


namespace NUMINAMATH_CALUDE_town_distance_bounds_l2746_274643

/-- Given two towns A and B that are 8 km apart, and towns B and C that are 10 km apart,
    prove that the distance between towns A and C is at least 2 km and at most 18 km. -/
theorem town_distance_bounds (A B C : ℝ × ℝ) : 
  dist A B = 8 → dist B C = 10 → 2 ≤ dist A C ∧ dist A C ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_town_distance_bounds_l2746_274643


namespace NUMINAMATH_CALUDE_floor_abs_sum_equals_57_l2746_274627

theorem floor_abs_sum_equals_57 : ⌊|(-57.85 : ℝ) + 0.1|⌋ = 57 := by sorry

end NUMINAMATH_CALUDE_floor_abs_sum_equals_57_l2746_274627


namespace NUMINAMATH_CALUDE_condition_relationship_l2746_274649

theorem condition_relationship (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a - b < a^2 - b^2) ∧
  (∃ a b : ℝ, a - b < a^2 - b^2 ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2746_274649


namespace NUMINAMATH_CALUDE_china_space_station_orbit_height_scientific_notation_l2746_274688

theorem china_space_station_orbit_height_scientific_notation :
  let orbit_height : ℝ := 400000
  orbit_height = 4 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_china_space_station_orbit_height_scientific_notation_l2746_274688


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l2746_274662

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def fairCoinProbability (n k : ℕ) : ℚ :=
  (binomial n k : ℚ) * (1 / 2) ^ n

theorem fair_coin_probability_difference :
  let p3 := fairCoinProbability 5 3
  let p4 := fairCoinProbability 5 4
  abs (p3 - p4) = 5 / 32 := by sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l2746_274662


namespace NUMINAMATH_CALUDE_election_margin_of_victory_l2746_274639

theorem election_margin_of_victory 
  (total_votes : ℕ) 
  (winning_percentage : ℚ) 
  (winning_votes : ℕ) : 
  winning_percentage = 29/50 → 
  winning_votes = 1044 → 
  (winning_votes : ℚ) / winning_percentage = total_votes → 
  winning_votes - (total_votes - winning_votes) = 288 :=
by sorry

end NUMINAMATH_CALUDE_election_margin_of_victory_l2746_274639


namespace NUMINAMATH_CALUDE_jeremy_watermelon_weeks_l2746_274668

/-- The number of weeks watermelons will last for Jeremy -/
def watermelon_weeks (total : ℕ) (eaten_per_week : ℕ) (given_to_dad : ℕ) : ℕ :=
  total / (eaten_per_week + given_to_dad)

/-- Theorem: Given Jeremy's watermelon consumption pattern, the watermelons will last 6 weeks -/
theorem jeremy_watermelon_weeks :
  watermelon_weeks 30 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_watermelon_weeks_l2746_274668


namespace NUMINAMATH_CALUDE_correct_operation_l2746_274669

theorem correct_operation (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2746_274669


namespace NUMINAMATH_CALUDE_line_slope_is_four_l2746_274612

/-- Given a line passing through points (0, 100) and (50, 300), prove that its slope is 4. -/
theorem line_slope_is_four :
  let x₁ : ℝ := 0
  let y₁ : ℝ := 100
  let x₂ : ℝ := 50
  let y₂ : ℝ := 300
  let slope : ℝ := (y₂ - y₁) / (x₂ - x₁)
  slope = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_is_four_l2746_274612


namespace NUMINAMATH_CALUDE_unique_solution_system_l2746_274656

theorem unique_solution_system (x y z : ℝ) : 
  x * (1 + y * z) = 9 ∧ 
  y * (1 + x * z) = 12 ∧ 
  z * (1 + x * y) = 10 ↔ 
  x = 1 ∧ y = 4 ∧ z = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2746_274656


namespace NUMINAMATH_CALUDE_amount_added_l2746_274673

theorem amount_added (N A : ℝ) : 
  N = 1.375 → 
  0.6667 * N + A = 1.6667 → 
  A = 0.750025 := by
sorry

end NUMINAMATH_CALUDE_amount_added_l2746_274673


namespace NUMINAMATH_CALUDE_fraction_simplification_l2746_274682

theorem fraction_simplification (c : ℝ) : (5 - 4 * c) / 9 - 3 = (-22 - 4 * c) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2746_274682


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2746_274677

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100 →
  a 4 + a 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2746_274677


namespace NUMINAMATH_CALUDE_sin_shift_l2746_274655

theorem sin_shift (x : ℝ) : Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l2746_274655


namespace NUMINAMATH_CALUDE_expression_evaluation_l2746_274618

theorem expression_evaluation :
  let a : ℚ := -2
  let b : ℚ := 1/5
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2746_274618


namespace NUMINAMATH_CALUDE_game_playing_time_l2746_274684

theorem game_playing_time (num_children : ℕ) (game_duration : ℕ) (players_at_once : ℕ) :
  num_children = 8 →
  game_duration = 120 →  -- 2 hours in minutes
  players_at_once = 2 →
  (game_duration * players_at_once) % num_children = 0 →
  (game_duration * players_at_once) / num_children = 30 :=
by sorry

end NUMINAMATH_CALUDE_game_playing_time_l2746_274684


namespace NUMINAMATH_CALUDE_passing_marks_l2746_274634

/-- The number of marks for passing an exam, given conditions about failing and passing candidates. -/
theorem passing_marks (T : ℝ) (P : ℝ) : 
  (0.4 * T = P - 40) →  -- Condition 1
  (0.6 * T = P + 20) →  -- Condition 2
  P = 160 := by
sorry

end NUMINAMATH_CALUDE_passing_marks_l2746_274634


namespace NUMINAMATH_CALUDE_bhanu_petrol_expense_l2746_274654

/-- Calculates Bhanu's petrol expense given his house rent expense and spending percentages -/
theorem bhanu_petrol_expense (house_rent : ℝ) (petrol_percent : ℝ) (rent_percent : ℝ) : 
  house_rent = 140 → 
  petrol_percent = 0.3 → 
  rent_percent = 0.2 → 
  ∃ (total_income : ℝ), 
    total_income > 0 ∧ 
    rent_percent * (1 - petrol_percent) * total_income = house_rent ∧
    petrol_percent * total_income = 300 :=
by sorry

end NUMINAMATH_CALUDE_bhanu_petrol_expense_l2746_274654


namespace NUMINAMATH_CALUDE_triangle_height_l2746_274657

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 12 → area = 30 → area = (base * height) / 2 → height = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l2746_274657


namespace NUMINAMATH_CALUDE_problem_solution_l2746_274628

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem problem_solution :
  (∀ a : ℝ, A ∪ B a = B a → a = 1) ∧
  (∀ C : Set ℝ, (∀ a : ℝ, A ∩ B a = B a → a ∈ C) → C = {a : ℝ | a ≤ -1 ∨ a = 1}) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2746_274628


namespace NUMINAMATH_CALUDE_union_A_complement_B_A_subset_B_iff_a_in_range_l2746_274620

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2*x + a ∧ 2*x + a ≤ 3}
def B : Set ℝ := {x | 2*x^2 - 3*x - 2 < 0}

-- Part 1
theorem union_A_complement_B :
  A 1 ∪ (Set.univ \ B) = {x : ℝ | x ≤ 1 ∨ x ≥ 2} := by sorry

-- Part 2
theorem A_subset_B_iff_a_in_range (a : ℝ) :
  A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_A_subset_B_iff_a_in_range_l2746_274620


namespace NUMINAMATH_CALUDE_unique_integer_modulo_l2746_274640

theorem unique_integer_modulo : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_modulo_l2746_274640


namespace NUMINAMATH_CALUDE_f_zero_one_eq_neg_one_one_l2746_274667

/-- The type of points in the real plane -/
def RealPair := ℝ × ℝ

/-- The mapping f: A → B -/
def f (p : RealPair) : RealPair :=
  (p.1 - p.2, p.1 + p.2)

/-- Theorem stating that f(0, 1) = (-1, 1) -/
theorem f_zero_one_eq_neg_one_one :
  f (0, 1) = (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_f_zero_one_eq_neg_one_one_l2746_274667


namespace NUMINAMATH_CALUDE_existence_of_integers_satisfying_inequality_l2746_274687

theorem existence_of_integers_satisfying_inequality :
  ∃ (a b : ℤ), (2003 : ℝ) < (a : ℝ) + (b : ℝ) * Real.sqrt 2 ∧ 
  (a : ℝ) + (b : ℝ) * Real.sqrt 2 < 2003.01 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_satisfying_inequality_l2746_274687


namespace NUMINAMATH_CALUDE_largest_four_digit_palindrome_divisible_by_three_l2746_274693

/-- A four-digit palindrome is a number between 1000 and 9999 that reads the same forwards and backwards -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n.digits 10).reverse = n.digits 10

/-- A number is divisible by 3 if it leaves no remainder when divided by 3 -/
def divisible_by_three (n : ℕ) : Prop :=
  n % 3 = 0

theorem largest_four_digit_palindrome_divisible_by_three :
  ∀ n : ℕ, is_four_digit_palindrome n → divisible_by_three n → n ≤ 9999 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_palindrome_divisible_by_three_l2746_274693


namespace NUMINAMATH_CALUDE_equation_solution_l2746_274636

theorem equation_solution : ∃ x : ℝ, 10111 - 10 * 2 * (5 + x) = 0 ∧ x = 500.55 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2746_274636


namespace NUMINAMATH_CALUDE_sum_of_twos_and_threes_3024_l2746_274601

/-- The number of ways to write a positive integer as an unordered sum of 2s and 3s -/
def sumOfTwosAndThrees (n : ℕ) : ℕ := 
  (n / 3 : ℕ) - (n % 3) / 3 + 1

/-- Theorem stating that there are 337 ways to write 3024 as an unordered sum of 2s and 3s -/
theorem sum_of_twos_and_threes_3024 : sumOfTwosAndThrees 3024 = 337 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_twos_and_threes_3024_l2746_274601


namespace NUMINAMATH_CALUDE_cubic_arithmetic_progression_complex_root_l2746_274602

/-- A cubic polynomial with coefficients in ℝ -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The roots of a cubic polynomial form an arithmetic progression -/
def roots_form_arithmetic_progression (p : CubicPolynomial) : Prop :=
  ∃ (r d : ℂ), p.a * (r - d)^3 + p.b * (r - d)^2 + p.c * (r - d) + p.d = 0 ∧
                p.a * r^3 + p.b * r^2 + p.c * r + p.d = 0 ∧
                p.a * (r + d)^3 + p.b * (r + d)^2 + p.c * (r + d) + p.d = 0

/-- One of the roots of a cubic polynomial is complex -/
def has_complex_root (p : CubicPolynomial) : Prop :=
  ∃ (z : ℂ), z.im ≠ 0 ∧ p.a * z^3 + p.b * z^2 + p.c * z + p.d = 0

/-- The main theorem -/
theorem cubic_arithmetic_progression_complex_root :
  ∃! (a : ℝ), roots_form_arithmetic_progression { a := 1, b := -9, c := 30, d := a } ∧
               has_complex_root { a := 1, b := -9, c := 30, d := a } ∧
               a = -12 := by sorry

end NUMINAMATH_CALUDE_cubic_arithmetic_progression_complex_root_l2746_274602


namespace NUMINAMATH_CALUDE_min_area_bounded_by_curve_and_lines_l2746_274666

noncomputable section

-- Define the curve C
def f (x : ℝ) : ℝ := 1 / (1 + x^2)

-- Define the area function T(α)
def T (α : ℝ) : ℝ :=
  Real.arctan α + Real.arctan (1 / α) - α / (1 + α^2)

-- Theorem statement
theorem min_area_bounded_by_curve_and_lines (α : ℝ) (h : α > 0) :
  ∃ (min_area : ℝ), min_area = π / 2 - 1 / 2 ∧
  ∀ β > 0, T β ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_min_area_bounded_by_curve_and_lines_l2746_274666


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l2746_274670

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 3 →  -- ratio of angles is 5:3
  b = 67.5 :=  -- smaller angle is 67.5°
by
  sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l2746_274670


namespace NUMINAMATH_CALUDE_row_arrangement_counts_l2746_274626

/-- Represents a person in the row -/
inductive Person : Type
| A | B | C | D | E

/-- A row is a permutation of five people -/
def Row := Fin 5 → Person

/-- Checks if A and B are adjacent with B to the right of A in a given row -/
def adjacent_AB (row : Row) : Prop :=
  ∃ i : Fin 4, row i = Person.A ∧ row (i.succ) = Person.B

/-- Checks if A, B, and C are in order from left to right in a given row -/
def ABC_in_order (row : Row) : Prop :=
  ∃ i j k : Fin 5, i < j ∧ j < k ∧ 
    row i = Person.A ∧ row j = Person.B ∧ row k = Person.C

/-- The main theorem to be proved -/
theorem row_arrangement_counts :
  (∃! (s : Finset Row), s.card = 24 ∧ ∀ row ∈ s, adjacent_AB row) ∧
  (∃! (s : Finset Row), s.card = 20 ∧ ∀ row ∈ s, ABC_in_order row) :=
sorry

end NUMINAMATH_CALUDE_row_arrangement_counts_l2746_274626


namespace NUMINAMATH_CALUDE_senior_citizen_tickets_l2746_274690

theorem senior_citizen_tickets (total_tickets : ℕ) (adult_price senior_price : ℚ) (total_receipts : ℚ) :
  total_tickets = 529 →
  adult_price = 25 →
  senior_price = 15 →
  total_receipts = 9745 →
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 348 :=
by sorry

end NUMINAMATH_CALUDE_senior_citizen_tickets_l2746_274690


namespace NUMINAMATH_CALUDE_no_infinite_line_family_l2746_274691

theorem no_infinite_line_family :
  ¬ ∃ (k : ℕ → ℝ),
    (∀ n, k n ≠ 0) ∧
    (∀ n, k (n + 1) = (1 - 1 / k n) - (1 - k n)) ∧
    (∀ n, k n * k (n + 1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_line_family_l2746_274691


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2746_274621

def M : Set Int := {-1, 1}
def N : Set Int := {-2, 1, 0}

theorem intersection_of_M_and_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2746_274621


namespace NUMINAMATH_CALUDE_f_inequality_solution_set_m_range_l2746_274648

def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|

theorem f_inequality_solution_set :
  {x : ℝ | f x < 4} = {x : ℝ | -4/3 < x ∧ x < 4/3} := by sorry

theorem m_range (m : ℝ) :
  (∃ x₀ : ℝ, ∀ t : ℝ, f x₀ < |m + t| + |t - m|) ↔ 
  (m < -3/4 ∨ m > 3/4) := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_set_m_range_l2746_274648


namespace NUMINAMATH_CALUDE_royalties_for_420_tax_l2746_274651

/-- Calculates the tax on royalties based on the given rules -/
def calculateTax (royalties : ℕ) : ℚ :=
  if royalties ≤ 800 then 0
  else if royalties ≤ 4000 then (royalties - 800) * 14 / 100
  else royalties * 11 / 100

/-- Theorem stating that 3800 yuan in royalties results in 420 yuan tax -/
theorem royalties_for_420_tax : calculateTax 3800 = 420 := by sorry

end NUMINAMATH_CALUDE_royalties_for_420_tax_l2746_274651


namespace NUMINAMATH_CALUDE_inclination_angle_range_l2746_274635

-- Define the slope k and inclination angle α
variable (k α : ℝ)

-- Define the relationship between k and α
def slope_angle_relation (k α : ℝ) : Prop := k = Real.tan α

-- Define the range of k
def slope_range (k : ℝ) : Prop := -1 ≤ k ∧ k < Real.sqrt 3

-- Define the range of α
def angle_range (α : ℝ) : Prop := 
  (0 ≤ α ∧ α < Real.pi/3) ∨ (3*Real.pi/4 ≤ α ∧ α < Real.pi)

-- State the theorem
theorem inclination_angle_range :
  ∀ k α, slope_angle_relation k α → slope_range k → angle_range α :=
sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l2746_274635


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2746_274614

theorem inequality_equivalence (x : ℝ) :
  (2 * (5 ^ (2 * x)) * Real.sin (2 * x) - 3 ^ x ≥ 5 ^ (2 * x) - 2 * (3 ^ x) * Real.sin (2 * x)) ↔
  (∃ k : ℤ, π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2746_274614


namespace NUMINAMATH_CALUDE_inequality_proof_l2746_274629

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b) + (b / Real.sqrt a) ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2746_274629


namespace NUMINAMATH_CALUDE_count_9_in_1_to_1000_l2746_274660

/-- Count of digit 9 in a specific place value for numbers from 1 to 1000 -/
def count_digit_9_in_place (place : Nat) : Nat :=
  1000 / (10 ^ place)

/-- Total count of digit 9 in all integers from 1 to 1000 -/
def total_count_9 : Nat :=
  count_digit_9_in_place 0 + count_digit_9_in_place 1 + count_digit_9_in_place 2

theorem count_9_in_1_to_1000 :
  total_count_9 = 300 := by
  sorry

end NUMINAMATH_CALUDE_count_9_in_1_to_1000_l2746_274660


namespace NUMINAMATH_CALUDE_movie_theater_revenue_is_6810_l2746_274664

/-- Represents the revenue calculation for a movie theater --/
def movie_theater_revenue : ℕ := by
  -- Matinee ticket prices and sales
  let matinee_price : ℕ := 5
  let matinee_early_bird_discount : ℚ := 0.5
  let matinee_early_bird_tickets : ℕ := 20
  let matinee_regular_tickets : ℕ := 180

  -- Evening ticket prices and sales
  let evening_price : ℕ := 12
  let evening_group_discount : ℚ := 0.1
  let evening_student_senior_discount : ℚ := 0.25
  let evening_group_tickets : ℕ := 150
  let evening_student_senior_tickets : ℕ := 75
  let evening_regular_tickets : ℕ := 75

  -- 3D ticket prices and sales
  let threeD_price : ℕ := 20
  let threeD_online_surcharge : ℕ := 3
  let threeD_family_discount : ℚ := 0.15
  let threeD_online_tickets : ℕ := 60
  let threeD_family_tickets : ℕ := 25
  let threeD_regular_tickets : ℕ := 15

  -- Late-night ticket prices and sales
  let late_night_price : ℕ := 10
  let late_night_high_demand_increase : ℚ := 0.2
  let late_night_high_demand_tickets : ℕ := 30
  let late_night_regular_tickets : ℕ := 20

  -- Calculate total revenue
  let total_revenue : ℕ := 6810

  exact total_revenue

/-- Theorem stating that the movie theater's revenue on this day is $6810 --/
theorem movie_theater_revenue_is_6810 : movie_theater_revenue = 6810 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_revenue_is_6810_l2746_274664


namespace NUMINAMATH_CALUDE_rectangle_breadth_calculation_l2746_274697

/-- Given a rectangle with original length 18 cm and unknown breadth,
    if the length is increased to 25 cm and the new breadth is 7.2 cm
    while maintaining the same area, then the original breadth was 10 cm. -/
theorem rectangle_breadth_calculation (original_breadth : ℝ) : 
  18 * original_breadth = 25 * 7.2 → original_breadth = 10 := by
  sorry

#check rectangle_breadth_calculation

end NUMINAMATH_CALUDE_rectangle_breadth_calculation_l2746_274697


namespace NUMINAMATH_CALUDE_subtraction_problem_l2746_274610

theorem subtraction_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2746_274610


namespace NUMINAMATH_CALUDE_total_students_l2746_274616

theorem total_students (students_3rd : ℕ) (students_4th : ℕ) (boys_2nd : ℕ) (girls_2nd : ℕ) :
  students_3rd = 19 →
  students_4th = 2 * students_3rd →
  boys_2nd = 10 →
  girls_2nd = 19 →
  students_3rd + students_4th + (boys_2nd + girls_2nd) = 86 :=
by sorry

end NUMINAMATH_CALUDE_total_students_l2746_274616


namespace NUMINAMATH_CALUDE_newborn_count_l2746_274676

theorem newborn_count (total_children : ℕ) (toddlers : ℕ) : 
  total_children = 40 → 
  toddlers = 6 → 
  total_children = toddlers + 5 * toddlers + (total_children - toddlers - 5 * toddlers) → 
  (total_children - toddlers - 5 * toddlers) = 4 := by
sorry

end NUMINAMATH_CALUDE_newborn_count_l2746_274676


namespace NUMINAMATH_CALUDE_union_complement_problem_l2746_274650

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem union_complement_problem : A ∪ (U \ B) = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_complement_problem_l2746_274650


namespace NUMINAMATH_CALUDE_gcf_of_lcms_main_result_l2746_274609

theorem gcf_of_lcms (a b c d : ℕ) : 
  Nat.gcd (Nat.lcm a b) (Nat.lcm c d) = Nat.gcd (Nat.lcm 16 21) (Nat.lcm 14 18) := by
  sorry

theorem main_result : Nat.gcd (Nat.lcm 16 21) (Nat.lcm 14 18) = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_main_result_l2746_274609


namespace NUMINAMATH_CALUDE_count_integer_root_cases_correct_l2746_274685

/-- The number of real values 'a' for which x^2 + ax + 12a = 0 has only integer roots -/
def count_integer_root_cases : ℕ := 8

/-- A function that returns true if the quadratic equation x^2 + ax + 12a = 0 has only integer roots -/
def has_only_integer_roots (a : ℝ) : Prop :=
  ∃ p q : ℤ, ∀ x : ℝ, x^2 + a*x + 12*a = 0 ↔ x = p ∨ x = q

/-- The theorem stating that there are exactly 8 real numbers 'a' for which
    the quadratic equation x^2 + ax + 12a = 0 has only integer roots -/
theorem count_integer_root_cases_correct :
  (∃ S : Finset ℝ, Finset.card S = count_integer_root_cases ∧
    (∀ a : ℝ, a ∈ S ↔ has_only_integer_roots a)) := by
  sorry


end NUMINAMATH_CALUDE_count_integer_root_cases_correct_l2746_274685


namespace NUMINAMATH_CALUDE_new_cube_edge_length_new_cube_edge_is_six_l2746_274641

/-- Given three cubes with edges 3, 4, and 5 decimeters, when melted and formed into a single cube,
    the edge length of the new cube is 6 decimeters. -/
theorem new_cube_edge_length : ℝ → Prop :=
  fun edge_length =>
    let cube1_volume : ℝ := 3^3
    let cube2_volume : ℝ := 4^3
    let cube3_volume : ℝ := 5^3
    let total_volume : ℝ := cube1_volume + cube2_volume + cube3_volume
    edge_length^3 = total_volume ∧ edge_length = 6

/-- The edge length of the new cube is indeed 6 decimeters. -/
theorem new_cube_edge_is_six : new_cube_edge_length 6 := by
  sorry

end NUMINAMATH_CALUDE_new_cube_edge_length_new_cube_edge_is_six_l2746_274641


namespace NUMINAMATH_CALUDE_bc_over_a_is_zero_l2746_274695

theorem bc_over_a_is_zero (a b c : ℝ) 
  (h1 : a = 2*b + Real.sqrt 2)
  (h2 : a*b + (Real.sqrt 3 / 2)*c^2 + 1/4 = 0) : 
  b*c/a = 0 := by
  sorry

end NUMINAMATH_CALUDE_bc_over_a_is_zero_l2746_274695


namespace NUMINAMATH_CALUDE_john_boxes_l2746_274679

theorem john_boxes (stan jules joseph : ℕ) (john : ℚ) : 
  stan = 100 →
  joseph = stan / 5 →
  jules = joseph + 5 →
  john = jules * (6/5) →
  john = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_john_boxes_l2746_274679


namespace NUMINAMATH_CALUDE_robertson_seymour_theorem_l2746_274600

-- Define a graph type
structure Graph (V : Type) where
  edge : V → V → Prop

-- Define a complete graph
def CompleteGraph (n : ℕ) : Graph (Fin n) where
  edge i j := i ≠ j

-- Define the concept of a minor
def IsMinor {V W : Type} (G : Graph V) (H : Graph W) : Prop := sorry

-- Define tree decomposition
structure TreeDecomposition (V : Type) where
  T : Type
  bags : T → Set V
  -- Other properties of tree decomposition

-- Define k-almost embeddable
def KAlmostEmbeddable (k : ℕ) (G : Graph V) (S : Type) : Prop := sorry

-- Define the concept of a surface where K^n cannot be embedded
def SurfaceWithoutKn (n : ℕ) (S : Type) : Prop := sorry

-- The main theorem
theorem robertson_seymour_theorem {V : Type} (n : ℕ) (hn : n ≥ 5) :
  ∃ k : ℕ, ∀ (G : Graph V),
    ¬IsMinor G (CompleteGraph n) →
    ∃ (td : TreeDecomposition V) (S : Type),
      SurfaceWithoutKn n S ∧
      KAlmostEmbeddable k G S :=
sorry

end NUMINAMATH_CALUDE_robertson_seymour_theorem_l2746_274600


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l2746_274608

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  (a^2 + b^2 = c^2 ∨ a^2 + d^2 = b^2) → 
  c * d = 20 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l2746_274608


namespace NUMINAMATH_CALUDE_pool_capacity_l2746_274632

theorem pool_capacity : 
  ∀ (initial_fraction final_fraction added_volume total_capacity : ℚ),
  initial_fraction = 1 / 8 →
  final_fraction = 2 / 3 →
  added_volume = 210 →
  (final_fraction - initial_fraction) * total_capacity = added_volume →
  total_capacity = 5040 / 13 := by
sorry

end NUMINAMATH_CALUDE_pool_capacity_l2746_274632


namespace NUMINAMATH_CALUDE_statement_d_is_incorrect_l2746_274611

theorem statement_d_is_incorrect : ∃ (a b : ℝ), a^2 > b^2 ∧ a * b > 0 ∧ 1 / a ≥ 1 / b := by
  sorry

end NUMINAMATH_CALUDE_statement_d_is_incorrect_l2746_274611


namespace NUMINAMATH_CALUDE_walnut_logs_per_tree_l2746_274671

theorem walnut_logs_per_tree (pine_trees maple_trees walnut_trees : ℕ)
  (logs_per_pine logs_per_maple total_logs : ℕ) :
  pine_trees = 8 →
  maple_trees = 3 →
  walnut_trees = 4 →
  logs_per_pine = 80 →
  logs_per_maple = 60 →
  total_logs = 1220 →
  ∃ logs_per_walnut : ℕ,
    logs_per_walnut = 100 ∧
    total_logs = pine_trees * logs_per_pine + maple_trees * logs_per_maple + walnut_trees * logs_per_walnut :=
by sorry

end NUMINAMATH_CALUDE_walnut_logs_per_tree_l2746_274671


namespace NUMINAMATH_CALUDE_gcd_count_for_product_180_l2746_274665

theorem gcd_count_for_product_180 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 180) :
  ∃! (s : Finset ℕ+), (∀ x ∈ s, ∃ (a' b' : ℕ+), Nat.gcd a' b' * Nat.lcm a' b' = 180 ∧ Nat.gcd a' b' = x) ∧ s.card = 8 :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_180_l2746_274665


namespace NUMINAMATH_CALUDE_product_seven_l2746_274680

theorem product_seven : ∃ (x y : ℤ), x * y = 7 :=
sorry

end NUMINAMATH_CALUDE_product_seven_l2746_274680


namespace NUMINAMATH_CALUDE_coefficient_of_x_fifth_power_l2746_274603

theorem coefficient_of_x_fifth_power (x : ℝ) : 
  ∃ (a₆ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ), 
    (x - 2) * (x + 2)^5 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀ ∧ 
    a₅ = 8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fifth_power_l2746_274603


namespace NUMINAMATH_CALUDE_conditional_probability_proof_l2746_274692

-- Define the number of balls of each color and total number of balls
def red_balls : ℕ := 2
def yellow_balls : ℕ := 2
def blue_balls : ℕ := 2
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the number of draws
def num_draws : ℕ := 3

-- Define events A and B
def event_A : ℕ := 3 * red_balls * red_balls * total_balls
def event_B : ℕ := 3 * red_balls * red_balls * red_balls

-- Define the conditional probability P(B|A)
def prob_B_given_A : ℚ := event_B / event_A

-- Theorem to prove
theorem conditional_probability_proof : prob_B_given_A = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_proof_l2746_274692


namespace NUMINAMATH_CALUDE_divide_inequality_l2746_274681

theorem divide_inequality (x : ℝ) : -6 * x > 2 ↔ x < -1/3 := by sorry

end NUMINAMATH_CALUDE_divide_inequality_l2746_274681


namespace NUMINAMATH_CALUDE_symmetric_point_parabola_l2746_274698

/-- Given a parabola y = a(x+2)^2 and a point A(1,4), 
    prove that the point (-5,4) is symmetric to A 
    with respect to the parabola's axis of symmetry -/
theorem symmetric_point_parabola (a : ℝ) : 
  let parabola := fun (x : ℝ) => a * (x + 2)^2
  let A : ℝ × ℝ := (1, 4)
  let axis_of_symmetry : ℝ := -2
  let symmetric_point : ℝ × ℝ := (-5, 4)
  (symmetric_point.1 - axis_of_symmetry = -(A.1 - axis_of_symmetry)) ∧ 
  (symmetric_point.2 = A.2) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_parabola_l2746_274698


namespace NUMINAMATH_CALUDE_base_conversion_correct_l2746_274617

-- Define the base 10 number
def base_10_num : ℕ := 3527

-- Define the base 7 representation
def base_7_representation : List ℕ := [1, 3, 1, 6, 6]

-- Theorem statement
theorem base_conversion_correct :
  base_10_num = (List.foldr (λ (digit : ℕ) (acc : ℕ) => digit + 7 * acc) 0 base_7_representation) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_correct_l2746_274617


namespace NUMINAMATH_CALUDE_cube_roots_l2746_274661

theorem cube_roots : (39 : ℕ)^3 = 59319 ∧ (47 : ℕ)^3 = 103823 := by
  sorry

end NUMINAMATH_CALUDE_cube_roots_l2746_274661


namespace NUMINAMATH_CALUDE_dogwood_trees_after_five_years_l2746_274644

/-- Calculates the expected number of dogwood trees in the park after a given number of years -/
def expected_trees (initial_trees : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) 
                   (growth_rate_today : ℕ) (growth_rate_tomorrow : ℕ) (years : ℕ) : ℕ :=
  initial_trees + planted_today + planted_tomorrow + 
  (planted_today * growth_rate_today * years) + 
  (planted_tomorrow * growth_rate_tomorrow * years)

/-- Theorem stating the expected number of dogwood trees after 5 years -/
theorem dogwood_trees_after_five_years :
  expected_trees 39 41 20 2 4 5 = 130 := by
  sorry

#eval expected_trees 39 41 20 2 4 5

end NUMINAMATH_CALUDE_dogwood_trees_after_five_years_l2746_274644


namespace NUMINAMATH_CALUDE_order_independent_divisibility_criterion_only_for_3_and_9_l2746_274605

/-- A divisibility criterion for a positive integer that depends only on its digits. -/
def DigitDivisibilityCriterion (n : ℕ+) : Type :=
  (digits : List ℕ) → Bool

/-- The property that a divisibility criterion is independent of digit order. -/
def OrderIndependent (n : ℕ+) (criterion : DigitDivisibilityCriterion n) : Prop :=
  ∀ (digits₁ digits₂ : List ℕ), Multiset.ofList digits₁ = Multiset.ofList digits₂ →
    criterion digits₁ = criterion digits₂

/-- Theorem stating that order-independent digit divisibility criteria exist only for 3 and 9. -/
theorem order_independent_divisibility_criterion_only_for_3_and_9 (n : ℕ+) :
    (∃ (criterion : DigitDivisibilityCriterion n), OrderIndependent n criterion) →
    n = 3 ∨ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_order_independent_divisibility_criterion_only_for_3_and_9_l2746_274605


namespace NUMINAMATH_CALUDE_square_root_of_four_l2746_274619

theorem square_root_of_four : ∃ (x : ℝ), x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l2746_274619


namespace NUMINAMATH_CALUDE_quadratic_comparison_l2746_274604

/-- Given a quadratic function f(x) = a(x-1)^2 + 3 where a < 0,
    if f(-1) = y₁ and f(2) = y₂, then y₁ < y₂ -/
theorem quadratic_comparison (a y₁ y₂ : ℝ) (ha : a < 0) 
    (h1 : y₁ = a * (-1 - 1)^2 + 3)
    (h2 : y₂ = a * (2 - 1)^2 + 3) :
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_comparison_l2746_274604


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2746_274607

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_arithmetic : (a + b) / 2 = 5/2) (h_geometric : Real.sqrt (a * b) = Real.sqrt 6) :
  let e := Real.sqrt (1 + b^2 / a^2)
  e = Real.sqrt 13 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2746_274607


namespace NUMINAMATH_CALUDE_person_a_parts_l2746_274678

/-- Represents the number of parts made by each person -/
structure PartProduction where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the production scenario described in the problem -/
def production_scenario (p : PartProduction) : Prop :=
  p.c = 20 ∧
  4 * p.b = 3 * p.c ∧
  10 * p.a = 3 * (p.a + p.b + p.c)

theorem person_a_parts :
  ∀ p : PartProduction, production_scenario p → p.a = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_person_a_parts_l2746_274678


namespace NUMINAMATH_CALUDE_equal_sequence_l2746_274622

theorem equal_sequence (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h1 : ∀ i : Fin 2011, x i + x (i + 1) = 2 * x' i)
  (h2 : ∃ σ : Equiv (Fin 2011) (Fin 2011), ∀ i, x' i = x (σ i)) :
  ∀ i j : Fin 2011, x i = x j :=
by sorry

end NUMINAMATH_CALUDE_equal_sequence_l2746_274622


namespace NUMINAMATH_CALUDE_circle_placement_possible_l2746_274631

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square -/
structure Square where
  center : Point

/-- The main theorem -/
theorem circle_placement_possible
  (rect : Rectangle)
  (squares : Finset Square)
  (h_rect_dim : rect.width = 20 ∧ rect.height = 25)
  (h_squares_count : squares.card = 120) :
  ∃ (p : Point),
    (0.5 ≤ p.x ∧ p.x ≤ rect.width - 0.5) ∧
    (0.5 ≤ p.y ∧ p.y ≤ rect.height - 0.5) ∧
    ∀ (s : Square), s ∈ squares →
      (p.x - s.center.x)^2 + (p.y - s.center.y)^2 ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_circle_placement_possible_l2746_274631


namespace NUMINAMATH_CALUDE_solution_to_system_l2746_274625

/-- Prove that (4, 2, 3) is the solution to the given system of equations --/
theorem solution_to_system : ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^2 + y^2 + Real.sqrt 3 * x * y = 20 + 8 * Real.sqrt 3 ∧
  y^2 + z^2 = 13 ∧
  z^2 + x^2 + x * z = 37 ∧
  x = 4 ∧ y = 2 ∧ z = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l2746_274625


namespace NUMINAMATH_CALUDE_van_capacity_l2746_274674

theorem van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) : 
  students = 22 → adults = 2 → vans = 3 → (students + adults) / vans = 8 := by
  sorry

end NUMINAMATH_CALUDE_van_capacity_l2746_274674


namespace NUMINAMATH_CALUDE_freddy_age_l2746_274694

theorem freddy_age (matthew rebecca freddy : ℕ) : 
  matthew + rebecca + freddy = 35 →
  matthew = rebecca + 2 →
  freddy = matthew + 4 →
  ∃ (x : ℕ), matthew = 4 * x ∧ rebecca = 5 * x ∧ freddy = 7 * x →
  freddy = 15 := by
sorry

end NUMINAMATH_CALUDE_freddy_age_l2746_274694


namespace NUMINAMATH_CALUDE_line_chart_division_l2746_274645

/-- Represents a line chart -/
structure LineChart where
  /-- The line chart uses rise or fall of a line to represent increase or decrease in statistical quantities -/
  represents_statistical_quantities : Bool

/-- Represents a simple line chart -/
structure SimpleLineChart extends LineChart

/-- Represents a compound line chart -/
structure CompoundLineChart extends LineChart

/-- Theorem stating that line charts can be divided into simple and compound line charts -/
theorem line_chart_division (lc : LineChart) : 
  (∃ (slc : SimpleLineChart), slc.toLineChart = lc) ∨ 
  (∃ (clc : CompoundLineChart), clc.toLineChart = lc) :=
sorry

end NUMINAMATH_CALUDE_line_chart_division_l2746_274645


namespace NUMINAMATH_CALUDE_tangent_point_relation_l2746_274686

-- Define the curve and tangent line
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b
def tangent_line (x k : ℝ) : ℝ := k*x + 1

-- State the theorem
theorem tangent_point_relation (a b k : ℝ) : 
  (∃ x y, x = 1 ∧ y = 3 ∧ 
    curve x a b = y ∧ 
    tangent_line x k = y ∧
    (∀ x', curve x' a b = tangent_line x' k → x' = x)) →
  2*a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_point_relation_l2746_274686


namespace NUMINAMATH_CALUDE_symmetric_point_correct_specific_case_l2746_274683

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem symmetric_point_correct (p : ℝ × ℝ) : 
  symmetric_point p = (-p.1, -p.2) := by sorry

theorem specific_case : 
  symmetric_point (3, -1) = (-3, 1) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_specific_case_l2746_274683


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2746_274696

theorem functional_equation_solution (f g : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y * g x) = g x + x * f y) →
  (f = id ∧ g = id) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2746_274696
