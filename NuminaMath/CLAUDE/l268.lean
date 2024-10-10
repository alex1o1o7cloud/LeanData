import Mathlib

namespace jelly_mold_radius_l268_26875

theorem jelly_mold_radius :
  let original_radius : ℝ := 1.5
  let num_molds : ℕ := 64
  let hemisphere_volume (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3
  let original_volume := hemisphere_volume original_radius
  let small_mold_radius := (3 : ℝ) / 8
  original_volume = num_molds * hemisphere_volume small_mold_radius :=
by sorry

end jelly_mold_radius_l268_26875


namespace segment_length_proof_l268_26870

theorem segment_length_proof (A B O P M : Real) : 
  -- Conditions
  (0 ≤ A) ∧ (A < O) ∧ (O < M) ∧ (M < P) ∧ (P < B) ∧  -- Points lie on the line segment in order
  (O - A = 4/5 * (B - A)) ∧                           -- AO = 4/5 * AB
  (B - P = 2/3 * (B - A)) ∧                           -- BP = 2/3 * AB
  (M - A = 1/2 * (B - A)) ∧                           -- M is the midpoint of AB
  (M - O = 2) →                                       -- OM = 2
  (P - M = 10/9)                                      -- PM = 10/9

:= by sorry

end segment_length_proof_l268_26870


namespace part_one_part_two_part_three_l268_26866

-- Definition of opposite equations
def are_opposite_equations (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ ∃ x y : ℝ, a * x - b = 0 ∧ b * y - a = 0

-- Part (1)
theorem part_one : 
  are_opposite_equations 4 3 → are_opposite_equations 3 c → c = 4 :=
sorry

-- Part (2)
theorem part_two :
  are_opposite_equations 4 (-3 * m - 1) → are_opposite_equations 5 (n - 2) → m / n = -1/3 :=
sorry

-- Part (3)
theorem part_three :
  (∃ x : ℤ, 3 * x - c = 0) → (∃ y : ℤ, c * y - 3 = 0) → c = 3 ∨ c = -3 :=
sorry

end part_one_part_two_part_three_l268_26866


namespace column_for_2023_l268_26878

def column_sequence : Fin 8 → Char
  | 0 => 'B'
  | 1 => 'C'
  | 2 => 'D'
  | 3 => 'E'
  | 4 => 'D'
  | 5 => 'C'
  | 6 => 'B'
  | 7 => 'A'

def column_for_number (n : ℕ) : Char :=
  column_sequence ((n - 2) % 8)

theorem column_for_2023 : column_for_number 2023 = 'C' := by
  sorry

end column_for_2023_l268_26878


namespace pencil_cost_l268_26867

theorem pencil_cost (total_students : Nat) (total_cost : Nat) 
  (h1 : total_students = 30)
  (h2 : total_cost = 1771)
  (h3 : ∃ (s n c : Nat), 
    s > total_students / 2 ∧ 
    n > 1 ∧ 
    c > n ∧ 
    s * n * c = total_cost) :
  ∃ (s n : Nat), s * n * 11 = total_cost :=
by sorry

end pencil_cost_l268_26867


namespace meaningful_expression_l268_26808

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 5)) / x) ↔ (x ≥ -5 ∧ x ≠ 0) := by
  sorry

end meaningful_expression_l268_26808


namespace mixture_ratio_l268_26834

theorem mixture_ratio (p q : ℝ) : 
  p + q = 20 →
  p / (q + 1) = 4 / 3 →
  p / q = 3 / 2 := by
sorry

end mixture_ratio_l268_26834


namespace denver_birdhouse_profit_l268_26897

/-- Represents the profit calculation for Denver's birdhouse business -/
theorem denver_birdhouse_profit :
  ∀ (wood_pieces : ℕ) (wood_cost : ℚ) (sale_price : ℚ),
    wood_pieces = 7 →
    wood_cost = 3/2 →
    sale_price = 32 →
    (sale_price / 2) - (wood_pieces : ℚ) * wood_cost = 11/2 :=
by
  sorry

end denver_birdhouse_profit_l268_26897


namespace arithmetic_sequence_problems_l268_26837

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_problems
  (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0) (h_arith : arithmetic_sequence a d)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_m : ∃ m : ℕ, a m = 8)
  (S : ℕ → ℝ)
  (h_S : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2)
  (h_S3 : S 3 = 9)
  (h_S6 : S 6 = 36) :
  (∃ m : ℕ, a m = 8 ∧ m = 8) ∧
  (a 7 + a 8 + a 9 = 45) := by
  sorry

end arithmetic_sequence_problems_l268_26837


namespace andrew_ate_77_donuts_l268_26890

/-- The number of donuts Andrew ate on Monday -/
def monday_donuts : ℕ := 14

/-- The number of donuts Andrew ate on Tuesday -/
def tuesday_donuts : ℕ := monday_donuts / 2

/-- The number of donuts Andrew ate on Wednesday -/
def wednesday_donuts : ℕ := 4 * monday_donuts

/-- The total number of donuts Andrew ate in three days -/
def total_donuts : ℕ := monday_donuts + tuesday_donuts + wednesday_donuts

/-- Theorem stating that Andrew ate 77 donuts in total -/
theorem andrew_ate_77_donuts : total_donuts = 77 := by
  sorry

end andrew_ate_77_donuts_l268_26890


namespace symmetric_point_l268_26858

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point2D := ⟨0, 0⟩

/-- Function to check if a point is the midpoint of two other points -/
def isMidpoint (m : Point2D) (p1 : Point2D) (p2 : Point2D) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- Function to check if two points are symmetric with respect to the origin -/
def isSymmetricToOrigin (p1 : Point2D) (p2 : Point2D) : Prop :=
  isMidpoint origin p1 p2

/-- Theorem: The point (2,-3) is symmetric to (-2,3) with respect to the origin -/
theorem symmetric_point : 
  isSymmetricToOrigin ⟨-2, 3⟩ ⟨2, -3⟩ := by
  sorry


end symmetric_point_l268_26858


namespace P_in_M_l268_26836

def P : Set Nat := {0, 1}

def M : Set (Set Nat) := {x | x ⊆ P}

theorem P_in_M : P ∈ M := by sorry

end P_in_M_l268_26836


namespace inequality_solutions_l268_26830

theorem inequality_solutions :
  (∀ x : ℝ, 2 + 3*x - 2*x^2 > 0 ↔ -1/2 < x ∧ x < 2) ∧
  (∀ x : ℝ, x*(3-x) ≤ x*(x+2) - 1 ↔ x ≤ -1/2 ∨ x ≥ 1) := by
  sorry

end inequality_solutions_l268_26830


namespace bouquets_to_buy_is_correct_l268_26851

/-- Represents the number of roses in a bouquet Bill buys -/
def roses_per_bought_bouquet : ℕ := 7

/-- Represents the number of roses in a bouquet Bill sells -/
def roses_per_sold_bouquet : ℕ := 5

/-- Represents the price of a bouquet (both buying and selling) -/
def price_per_bouquet : ℕ := 20

/-- Represents the target profit -/
def target_profit : ℕ := 1000

/-- Calculates the number of bouquets Bill needs to buy to earn the target profit -/
def bouquets_to_buy : ℕ :=
  let bought_bouquets_per_operation := roses_per_sold_bouquet
  let sold_bouquets_per_operation := roses_per_bought_bouquet
  let profit_per_operation := sold_bouquets_per_operation * price_per_bouquet - bought_bouquets_per_operation * price_per_bouquet
  let operations_needed := target_profit / profit_per_operation
  operations_needed * bought_bouquets_per_operation

theorem bouquets_to_buy_is_correct :
  bouquets_to_buy = 125 := by sorry

end bouquets_to_buy_is_correct_l268_26851


namespace max_area_triangle_abc_l268_26885

theorem max_area_triangle_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  let angle_C : ℝ := π / 3
  let area := (1 / 2) * a * b * Real.sin angle_C
  3 * a * b = 25 - c^2 →
  ∀ (a' b' c' : ℝ), 
    a' > 0 → b' > 0 → c' > 0 →
    3 * a' * b' = 25 - c'^2 →
    area ≤ ((25 : ℝ) / 16) * Real.sqrt 3 :=
by sorry

end max_area_triangle_abc_l268_26885


namespace geometric_series_common_ratio_l268_26818

/-- The common ratio of the geometric series 4/5 - 5/12 + 25/72 - ... is -25/48 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/5
  let a₂ : ℚ := -5/12
  let a₃ : ℚ := 25/72
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → a₂ = r * a₁ ∧ a₃ = r * a₂) →
  r = -25/48 :=
by sorry

end geometric_series_common_ratio_l268_26818


namespace multiplicative_inverse_7_mod_31_l268_26898

theorem multiplicative_inverse_7_mod_31 : ∃ x : ℕ, x < 31 ∧ (7 * x) % 31 = 1 :=
by
  -- The proof goes here
  sorry

end multiplicative_inverse_7_mod_31_l268_26898


namespace paint_needed_for_smaller_statues_l268_26849

/-- The height of the original statue in feet -/
def original_height : ℝ := 12

/-- The height of each smaller statue in feet -/
def smaller_height : ℝ := 2

/-- The number of smaller statues -/
def num_statues : ℕ := 720

/-- The amount of paint in pints needed for the original statue -/
def paint_for_original : ℝ := 1

/-- The amount of paint needed for all smaller statues -/
def paint_for_all_statues : ℝ := 20

theorem paint_needed_for_smaller_statues :
  (num_statues : ℝ) * paint_for_original * (smaller_height / original_height) ^ 2 = paint_for_all_statues :=
sorry

end paint_needed_for_smaller_statues_l268_26849


namespace concert_ticket_cost_is_181_l268_26872

/-- Calculates the cost of a concert ticket given hourly wage, weekly hours, percentage of monthly salary for outing, drink ticket cost, and number of drink tickets. -/
def concert_ticket_cost (hourly_wage : ℚ) (weekly_hours : ℚ) (outing_percentage : ℚ) (drink_ticket_cost : ℚ) (num_drink_tickets : ℕ) : ℚ :=
  let monthly_salary := hourly_wage * weekly_hours * 4
  let outing_budget := monthly_salary * outing_percentage
  let drink_tickets_cost := drink_ticket_cost * num_drink_tickets
  outing_budget - drink_tickets_cost

/-- Theorem stating that the cost of the concert ticket is $181 given the specified conditions. -/
theorem concert_ticket_cost_is_181 :
  concert_ticket_cost 18 30 (1/10) 7 5 = 181 := by
  sorry

#eval concert_ticket_cost 18 30 (1/10) 7 5

end concert_ticket_cost_is_181_l268_26872


namespace triangle_area_l268_26893

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  (1/2) * a * b = 24 :=
by sorry

end triangle_area_l268_26893


namespace perpendicular_vectors_l268_26854

def a : ℝ × ℝ := (1, 3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

theorem perpendicular_vectors (m : ℝ) : 
  (a.1 * (a.1 + 2 * (b m).1) + a.2 * (a.2 + 2 * (b m).2) = 0) → m = -1 := by
  sorry

end perpendicular_vectors_l268_26854


namespace problem_solution_l268_26833

theorem problem_solution (p_xavier p_yvonne p_zelda p_wendell : ℚ)
  (h_xavier : p_xavier = 1/4)
  (h_yvonne : p_yvonne = 1/3)
  (h_zelda : p_zelda = 5/8)
  (h_wendell : p_wendell = 1/2) :
  p_xavier * p_yvonne * (1 - p_zelda) * (1 - p_wendell) = 1/64 := by
  sorry

end problem_solution_l268_26833


namespace hexagon_diagonals_intersect_at_nine_point_center_l268_26894

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Finset Point
  is_convex : Bool

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point := sorry

/-- The perpendicular bisector of a line segment -/
def perp_bisector (p1 p2 : Point) : Set Point := sorry

/-- The intersection points of a line with a triangle's sides -/
def intersections_with_triangle (line : Set Point) (t : Triangle) : Finset Point := sorry

/-- The hexagon formed by the intersections of perpendicular bisectors with triangle sides -/
def form_hexagon (t : Triangle) (h : Point) : Hexagon := sorry

/-- The main diagonals of a hexagon -/
def main_diagonals (h : Hexagon) : Finset (Set Point) := sorry

/-- The intersection point of the main diagonals of a hexagon -/
def diagonals_intersection (h : Hexagon) : Option Point := sorry

/-- The center of the nine-point circle of a triangle -/
def nine_point_center (t : Triangle) : Point := sorry

/-- The theorem to be proved -/
theorem hexagon_diagonals_intersect_at_nine_point_center 
  (t : Triangle) (is_acute : Bool) : 
  let h := orthocenter t
  let hexagon := form_hexagon t h
  diagonals_intersection hexagon = some (nine_point_center t) := by sorry

end hexagon_diagonals_intersect_at_nine_point_center_l268_26894


namespace bus_problem_l268_26879

/-- Calculates the number of students remaining on a bus after a given number of stops,
    where half of the students get off at each stop. -/
def studentsRemaining (initial : ℕ) (stops : ℕ) : ℕ :=
  initial / (2 ^ stops)

/-- Theorem: If a bus starts with 48 students and half of the remaining students get off
    at each of three consecutive stops, then 6 students will remain on the bus after the third stop. -/
theorem bus_problem : studentsRemaining 48 3 = 6 := by
  sorry

end bus_problem_l268_26879


namespace perimeter_after_adding_tiles_l268_26819

/-- Represents a tile arrangement -/
structure TileArrangement where
  num_tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to an arrangement -/
def add_tiles (initial : TileArrangement) (added_tiles : ℕ) : TileArrangement :=
  { num_tiles := initial.num_tiles + added_tiles,
    perimeter := initial.perimeter }  -- Placeholder, actual calculation depends on arrangement

/-- The theorem to be proved -/
theorem perimeter_after_adding_tiles 
  (initial : TileArrangement) 
  (h1 : initial.num_tiles = 10) 
  (h2 : initial.perimeter = 20) :
  ∃ (final : TileArrangement), 
    final = add_tiles initial 2 ∧ 
    final.perimeter = 19 :=
sorry

end perimeter_after_adding_tiles_l268_26819


namespace circle_radius_and_diameter_l268_26873

theorem circle_radius_and_diameter 
  (M N : ℝ) 
  (h_area : M = π * r^2) 
  (h_circumference : N = 2 * π * r) 
  (h_ratio : M / N = 15) : 
  r = 30 ∧ 2 * r = 60 := by
sorry

end circle_radius_and_diameter_l268_26873


namespace books_about_sports_l268_26828

theorem books_about_sports (total_books school_books : ℕ) : 
  total_books = 58 → school_books = 19 → total_books - school_books = 39 := by
  sorry

end books_about_sports_l268_26828


namespace complex_arithmetic_equation_l268_26832

theorem complex_arithmetic_equation : 
  10 - 1.05 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.93 := by
  sorry

end complex_arithmetic_equation_l268_26832


namespace rectangular_box_area_product_l268_26883

/-- Given a rectangular box with dimensions length, width, and height,
    prove that the product of the areas of its base, side, and front
    is equal to the square of its volume. -/
theorem rectangular_box_area_product (length width height : ℝ) :
  (length * width) * (width * height) * (height * length) = (length * width * height) ^ 2 := by
  sorry

end rectangular_box_area_product_l268_26883


namespace small_box_tape_length_l268_26823

theorem small_box_tape_length (large_seal : ℕ) (medium_seal : ℕ) (label_tape : ℕ)
  (large_count : ℕ) (medium_count : ℕ) (small_count : ℕ) (total_tape : ℕ)
  (h1 : large_seal = 4)
  (h2 : medium_seal = 2)
  (h3 : label_tape = 1)
  (h4 : large_count = 2)
  (h5 : medium_count = 8)
  (h6 : small_count = 5)
  (h7 : total_tape = 44)
  (h8 : total_tape = large_count * large_seal + medium_count * medium_seal + 
        small_count * label_tape + large_count * label_tape + 
        medium_count * label_tape + small_count * label_tape + 
        small_count * small_seal) :
  small_seal = 1 :=
by sorry

end small_box_tape_length_l268_26823


namespace decagon_diagonal_intersections_l268_26822

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def intersection_points (n : ℕ) : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points n = 210 :=
sorry

end decagon_diagonal_intersections_l268_26822


namespace sum_due_proof_l268_26831

/-- Banker's discount (BD) is the simple interest on the face value (FV) of a bill for the unexpired time -/
def bankers_discount (face_value : ℝ) : ℝ := 288

/-- True discount (TD) is the simple interest on the present value (PV) of the bill for the unexpired time -/
def true_discount (face_value : ℝ) : ℝ := 240

/-- The relationship between banker's discount, true discount, and face value -/
def discount_relationship (face_value : ℝ) : Prop :=
  bankers_discount face_value = true_discount face_value + (true_discount face_value)^2 / face_value

theorem sum_due_proof : 
  ∃ (face_value : ℝ), face_value = 1200 ∧ discount_relationship face_value := by
  sorry

end sum_due_proof_l268_26831


namespace max_divisor_with_remainder_l268_26820

theorem max_divisor_with_remainder (A B : ℕ) : 
  (24 = A * B + 4) → A ≤ 20 :=
by sorry

end max_divisor_with_remainder_l268_26820


namespace sum_of_sqrt_ratios_geq_two_l268_26869

theorem sum_of_sqrt_ratios_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + z)) + Real.sqrt (y / (z + x)) + Real.sqrt (z / (x + y)) ≥ 2 := by
  sorry

end sum_of_sqrt_ratios_geq_two_l268_26869


namespace w_squared_value_l268_26864

theorem w_squared_value (w : ℝ) (h : (w + 13)^2 = (3*w + 7)*(2*w + 4)) : w^2 = 141/5 := by
  sorry

end w_squared_value_l268_26864


namespace empty_cell_exists_l268_26804

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- A function that checks if two cells are adjacent --/
def adjacent (a b : Fin 5 × Fin 5) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Represents the movement of bugs --/
def moves (before after : Grid) : Prop :=
  ∀ (i j : Fin 5), 
    before i j → ∃ (i' j' : Fin 5), adjacent (i, j) (i', j') ∧ after i' j'

/-- The main theorem --/
theorem empty_cell_exists (before after : Grid) 
  (h1 : ∀ (i j : Fin 5), before i j)
  (h2 : moves before after) : 
  ∃ (i j : Fin 5), ¬after i j :=
sorry

end empty_cell_exists_l268_26804


namespace min_minutes_for_cheaper_plan_y_l268_26801

/-- The cost in cents for Plan X given y minutes of usage -/
def costX (y : ℕ) : ℚ := 15 * y

/-- The cost in cents for Plan Y given y minutes of usage -/
def costY (y : ℕ) : ℚ := 2500 + 8 * y

/-- Theorem stating that 358 is the minimum whole number of minutes for Plan Y to be cheaper -/
theorem min_minutes_for_cheaper_plan_y : 
  (∀ y : ℕ, y < 358 → costY y ≥ costX y) ∧ 
  costY 358 < costX 358 := by
  sorry

end min_minutes_for_cheaper_plan_y_l268_26801


namespace elderly_arrangement_theorem_l268_26868

/-- The number of ways to arrange n distinct objects in a row -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects, where order matters -/
def arrangements (n k : ℕ) : ℕ := 
  if k ≤ n then
    Nat.factorial n / Nat.factorial (n - k)
  else
    0

/-- The number of ways to arrange volunteers and elderly people with given constraints -/
def arrangement_count (volunteers elderly : ℕ) : ℕ :=
  permutations volunteers * arrangements (volunteers + 1) elderly

theorem elderly_arrangement_theorem :
  arrangement_count 4 2 = 480 := by
  sorry

end elderly_arrangement_theorem_l268_26868


namespace squirrel_climb_time_l268_26865

/-- Represents the climbing behavior of a squirrel -/
structure SquirrelClimb where
  climb_rate : ℕ  -- metres climbed in odd minutes
  slip_rate : ℕ   -- metres slipped in even minutes
  total_height : ℕ -- total height of the pole to climb

/-- Calculates the time taken for a squirrel to climb a pole -/
def climb_time (s : SquirrelClimb) : ℕ :=
  sorry

/-- Theorem: A squirrel with given climbing behavior takes 17 minutes to climb 26 metres -/
theorem squirrel_climb_time :
  let s : SquirrelClimb := { climb_rate := 5, slip_rate := 2, total_height := 26 }
  climb_time s = 17 :=
by sorry

end squirrel_climb_time_l268_26865


namespace jeans_sale_savings_l268_26814

/-- Calculates the total savings when purchasing jeans with given prices and discounts -/
def total_savings (fox_price pony_price : ℚ) (fox_discount pony_discount : ℚ) 
  (fox_quantity pony_quantity : ℕ) : ℚ :=
  let regular_total := fox_price * fox_quantity + pony_price * pony_quantity
  let discounted_total := (fox_price * (1 - fox_discount)) * fox_quantity + 
                          (pony_price * (1 - pony_discount)) * pony_quantity
  regular_total - discounted_total

/-- Theorem stating that the total savings is $18 under the given conditions -/
theorem jeans_sale_savings :
  let fox_price : ℚ := 15
  let pony_price : ℚ := 18
  let fox_quantity : ℕ := 3
  let pony_quantity : ℕ := 2
  let pony_discount : ℚ := 1/2
  let fox_discount : ℚ := 1/2 - pony_discount
  total_savings fox_price pony_price fox_discount pony_discount fox_quantity pony_quantity = 18 :=
by sorry

end jeans_sale_savings_l268_26814


namespace nth_prime_power_bound_l268_26845

/-- p_nth n returns the n-th prime number -/
def p_nth : ℕ → ℕ := sorry

/-- Theorem stating that for any positive integers n and k, 
    n is less than the k-th power of the 2k-th prime number -/
theorem nth_prime_power_bound (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  n < (p_nth (2 * k)) ^ k := by sorry

end nth_prime_power_bound_l268_26845


namespace cone_base_diameter_l268_26844

theorem cone_base_diameter (sphere_radius : ℝ) (cone_height : ℝ) (waste_percentage : ℝ) :
  sphere_radius = 9 →
  cone_height = 9 →
  waste_percentage = 75 →
  let cone_volume := (1 - waste_percentage / 100) * (4 / 3) * Real.pi * sphere_radius ^ 3
  let cone_base_radius := Real.sqrt (3 * cone_volume / (Real.pi * cone_height))
  2 * cone_base_radius = 9 :=
by sorry

end cone_base_diameter_l268_26844


namespace circumscribed_isosceles_trapezoid_radius_l268_26853

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedIsoscelesTrapezoid where
  /-- The angle at the base of the trapezoid -/
  baseAngle : ℝ
  /-- The length of the midline of the trapezoid -/
  midline : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ

/-- The theorem stating the relationship between the trapezoid's properties and the inscribed circle's radius -/
theorem circumscribed_isosceles_trapezoid_radius 
  (t : CircumscribedIsoscelesTrapezoid) 
  (h1 : t.baseAngle = 30 * π / 180)  -- 30 degrees in radians
  (h2 : t.midline = 10) : 
  t.radius = 2.5 := by
  sorry


end circumscribed_isosceles_trapezoid_radius_l268_26853


namespace reflection_of_point_l268_26847

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis reflection of a point -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The x-axis reflection of point (-2, 3) is (-2, -3) -/
theorem reflection_of_point :
  let P : Point := { x := -2, y := 3 }
  reflect_x P = { x := -2, y := -3 } := by
sorry

end reflection_of_point_l268_26847


namespace smallest_rectangle_area_is_768_l268_26806

/-- The side length of each square in centimeters -/
def square_side : ℝ := 8

/-- The number of squares in the height of the L-shape -/
def height_squares : ℕ := 3

/-- The number of squares in the width of the L-shape -/
def width_squares : ℕ := 4

/-- The height of the L-shape in centimeters -/
def l_shape_height : ℝ := square_side * height_squares

/-- The width of the L-shape in centimeters -/
def l_shape_width : ℝ := square_side * width_squares

/-- The smallest possible area of a rectangle that can completely contain the L-shape -/
def smallest_rectangle_area : ℝ := l_shape_height * l_shape_width

theorem smallest_rectangle_area_is_768 : smallest_rectangle_area = 768 := by
  sorry

end smallest_rectangle_area_is_768_l268_26806


namespace calvin_collection_total_l268_26817

def insect_collection (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := 2 * scorpions
  roaches + scorpions + crickets + caterpillars

theorem calvin_collection_total :
  insect_collection 12 3 = 27 :=
by sorry

end calvin_collection_total_l268_26817


namespace abs_neg_five_halves_l268_26802

theorem abs_neg_five_halves : |(-5 : ℚ) / 2| = 5 / 2 := by
  sorry

end abs_neg_five_halves_l268_26802


namespace absolute_value_inequality_l268_26805

theorem absolute_value_inequality (b : ℝ) (h₁ : b > 0) :
  (∃ x : ℝ, |2*x - 8| + |2*x - 6| < b) → b > 2 := by sorry

end absolute_value_inequality_l268_26805


namespace age_ratio_problem_l268_26829

theorem age_ratio_problem (a b : ℕ) (h1 : 5 * b = 3 * a) (h2 : a - 4 = b + 4) :
  3 * (b - 4) = a + 4 := by
  sorry

end age_ratio_problem_l268_26829


namespace chess_team_photo_arrangements_l268_26859

/-- The number of ways to arrange a chess team in a line -/
def chessTeamArrangements (numBoys numGirls : ℕ) : ℕ :=
  Nat.factorial numGirls * Nat.factorial numBoys

/-- Theorem: There are 12 ways to arrange 2 boys and 3 girls in a line
    with girls in the middle and boys on the ends -/
theorem chess_team_photo_arrangements :
  chessTeamArrangements 2 3 = 12 := by
  sorry

#eval chessTeamArrangements 2 3

end chess_team_photo_arrangements_l268_26859


namespace partial_fraction_decomposition_l268_26827

theorem partial_fraction_decomposition :
  ∃ (A B C : ℝ), ∀ (x : ℝ), x ≠ 0 → x^2 + 1 ≠ 0 →
    (-x^2 + 3*x - 4) / (x^3 + x) = A / x + (B*x + C) / (x^2 + 1) ∧
    A = -4 ∧ B = 3 ∧ C = 3 := by
  sorry

end partial_fraction_decomposition_l268_26827


namespace playhouse_siding_cost_l268_26886

/-- Calculates the cost of siding for a playhouse with given dimensions --/
theorem playhouse_siding_cost
  (wall_width : ℝ)
  (wall_height : ℝ)
  (roof_width : ℝ)
  (roof_height : ℝ)
  (siding_width : ℝ)
  (siding_height : ℝ)
  (siding_cost : ℝ)
  (h_wall_width : wall_width = 10)
  (h_wall_height : wall_height = 7)
  (h_roof_width : roof_width = 10)
  (h_roof_height : roof_height = 6)
  (h_siding_width : siding_width = 10)
  (h_siding_height : siding_height = 15)
  (h_siding_cost : siding_cost = 35) :
  ⌈(wall_width * wall_height + 2 * roof_width * roof_height) / (siding_width * siding_height)⌉ * siding_cost = 70 :=
by sorry

end playhouse_siding_cost_l268_26886


namespace greatest_two_digit_multiple_of_3_and_5_l268_26852

theorem greatest_two_digit_multiple_of_3_and_5 : 
  ∀ n : ℕ, n ≤ 99 → n ≥ 10 → n % 3 = 0 → n % 5 = 0 → n ≤ 90 :=
by
  sorry

end greatest_two_digit_multiple_of_3_and_5_l268_26852


namespace minimum_value_of_function_l268_26887

theorem minimum_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  (1/x + 4/(1 - 2*x)) ≥ 6 + 4*Real.sqrt 2 := by
  sorry

end minimum_value_of_function_l268_26887


namespace audrey_peaches_l268_26895

def paul_peaches : ℕ := 48
def peach_difference : ℤ := 22

theorem audrey_peaches :
  ∃ (audrey : ℕ), (audrey : ℤ) - paul_peaches = peach_difference ∧ audrey = 70 := by
  sorry

end audrey_peaches_l268_26895


namespace odometer_sum_of_squares_l268_26813

/-- Represents a car's odometer reading as a 3-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Converts an OdometerReading to a natural number -/
def OdometerReading.toNat (r : OdometerReading) : Nat :=
  100 * r.hundreds + 10 * r.tens + r.ones

/-- Reverses the digits of an OdometerReading -/
def OdometerReading.reverse (r : OdometerReading) : OdometerReading where
  hundreds := r.ones
  tens := r.tens
  ones := r.hundreds
  valid := by sorry

theorem odometer_sum_of_squares (initial : OdometerReading) 
  (h1 : initial.hundreds + initial.tens + initial.ones ≤ 9)
  (h2 : ∃ (hours : Nat), 
    (OdometerReading.toNat (OdometerReading.reverse initial) - OdometerReading.toNat initial) = 75 * hours) :
  initial.hundreds^2 + initial.tens^2 + initial.ones^2 = 75 := by
  sorry

end odometer_sum_of_squares_l268_26813


namespace smallest_covering_rectangles_l268_26824

/-- Represents a rectangle with width and height. -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a rectangular region to be covered. -/
structure Region where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle. -/
def rectangleArea (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the area of a region. -/
def regionArea (r : Region) : ℕ := r.width * r.height

/-- Theorem: The smallest number of 3-by-5 rectangles needed to cover a 15-by-20 region is 20. -/
theorem smallest_covering_rectangles :
  let coveringRectangle : Rectangle := { width := 3, height := 5 }
  let regionToCover : Region := { width := 15, height := 20 }
  (regionArea regionToCover) / (rectangleArea coveringRectangle) = 20 ∧
  (regionToCover.width % coveringRectangle.width = 0) ∧
  (regionToCover.height % coveringRectangle.height = 0) := by
  sorry

#check smallest_covering_rectangles

end smallest_covering_rectangles_l268_26824


namespace steel_copper_weight_difference_l268_26841

/-- Represents the weight of a metal bar in kilograms. -/
structure MetalBar where
  weight : ℝ

/-- The container with metal bars. -/
structure Container where
  steel : MetalBar
  tin : MetalBar
  copper : MetalBar
  count : ℕ
  totalWeight : ℝ

/-- Theorem stating the weight difference between steel and copper bars. -/
theorem steel_copper_weight_difference (c : Container) : 
  c.steel.weight - c.copper.weight = 20 :=
  by
  have h1 : c.steel.weight = 2 * c.tin.weight := sorry
  have h2 : c.copper.weight = 90 := sorry
  have h3 : c.count = 20 := sorry
  have h4 : c.totalWeight = 5100 := sorry
  have h5 : c.count * (c.steel.weight + c.tin.weight + c.copper.weight) = c.totalWeight := sorry
  sorry

#check steel_copper_weight_difference

end steel_copper_weight_difference_l268_26841


namespace F_is_even_T_is_even_l268_26863

variable (f : ℝ → ℝ)

def F (x : ℝ) : ℝ := f x * f (-x)

def T (x : ℝ) : ℝ := f x + f (-x)

theorem F_is_even : ∀ x : ℝ, F f x = F f (-x) := by sorry

theorem T_is_even : ∀ x : ℝ, T f x = T f (-x) := by sorry

end F_is_even_T_is_even_l268_26863


namespace bus_problem_l268_26862

theorem bus_problem (initial_students : ℕ) (remaining_fraction : ℚ) : 
  initial_students = 64 →
  remaining_fraction = 2/3 →
  (initial_students : ℚ) * remaining_fraction^3 = 512/27 := by
  sorry

end bus_problem_l268_26862


namespace geometric_sequence_property_l268_26840

/-- Theorem: In a geometric sequence where a₅ = 4 and a₇ = 6, a₉ = 9 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 5 = 4 →
  a 7 = 6 →
  a 9 = 9 := by
sorry

end geometric_sequence_property_l268_26840


namespace angle_complement_l268_26891

/-- Given an angle α of 63°21', its complement is 26°39' -/
theorem angle_complement (α : Real) : α = 63 + 21 / 60 → 90 - α = 26 + 39 / 60 := by
  sorry

end angle_complement_l268_26891


namespace green_face_prob_half_l268_26800

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  green_faces : ℕ
  purple_faces : ℕ

/-- The probability of rolling a green face on a colored cube -/
def green_face_probability (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

/-- Theorem: The probability of rolling a green face on a cube with 3 green faces and 3 purple faces is 1/2 -/
theorem green_face_prob_half :
  let cube : ColoredCube := { total_faces := 6, green_faces := 3, purple_faces := 3 }
  green_face_probability cube = 1 / 2 := by sorry

end green_face_prob_half_l268_26800


namespace equation_solution_l268_26871

theorem equation_solution :
  let f (x : ℂ) := -x^2 * (x + 2) - (2 * x + 4)
  ∀ x : ℂ, x ≠ -2 → (f x = 0 ↔ x = -2 ∨ x = 2*I ∨ x = -2*I) :=
sorry

end equation_solution_l268_26871


namespace A_intersect_B_l268_26889

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem A_intersect_B : A ∩ B = {1, 3} := by sorry

end A_intersect_B_l268_26889


namespace cubic_equation_geometric_progression_solution_l268_26825

/-- Given a cubic equation ax^3 + bx^2 + cx + d = 0 where the coefficients form
    a geometric progression with ratio q, prove that x = -q is a solution. -/
theorem cubic_equation_geometric_progression_solution
  (a b c d q : ℝ) (hq : q ≠ 0) (ha : a ≠ 0)
  (hb : b = a * q) (hc : c = a * q^2) (hd : d = a * q^3) :
  a * (-q)^3 + b * (-q)^2 + c * (-q) + d = 0 :=
by sorry

end cubic_equation_geometric_progression_solution_l268_26825


namespace average_velocity_proof_l268_26880

/-- The average velocity of a particle with motion equation s(t) = 4 - 2t² 
    over the time interval [1, 1+Δt] is equal to -4 - 2Δt. -/
theorem average_velocity_proof (Δt : ℝ) : 
  let s (t : ℝ) := 4 - 2 * t^2
  let v_avg := (s (1 + Δt) - s 1) / Δt
  v_avg = -4 - 2 * Δt :=
by sorry

end average_velocity_proof_l268_26880


namespace triangle_area_on_grid_l268_26803

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem triangle_area_on_grid :
  let A : Point := { x := 0, y := 0 }
  let B : Point := { x := 2, y := 0 }
  let C : Point := { x := 2, y := 2.5 }
  triangleArea A B C = 2.5 := by sorry

end triangle_area_on_grid_l268_26803


namespace sufficient_not_necessary_l268_26877

theorem sufficient_not_necessary : ∀ x : ℝ, 
  (∀ x, x > 5 → x > 3) ∧ 
  (∃ x, x > 3 ∧ x ≤ 5) := by
  sorry

end sufficient_not_necessary_l268_26877


namespace prob_greater_than_three_is_half_l268_26811

/-- The probability of rolling a number greater than 3 on a standard six-sided die is 1/2. -/
theorem prob_greater_than_three_is_half : 
  let outcomes := Finset.range 6
  let favorable := {4, 5, 6}
  Finset.card favorable / Finset.card outcomes = 1 / 2 := by sorry

end prob_greater_than_three_is_half_l268_26811


namespace tax_fraction_proof_l268_26888

theorem tax_fraction_proof (gross_income : ℝ) (car_payment : ℝ) (car_payment_percentage : ℝ) :
  gross_income = 3000 →
  car_payment = 400 →
  car_payment_percentage = 0.20 →
  car_payment = car_payment_percentage * (gross_income * (1 - (1/3))) →
  1/3 = (gross_income - (car_payment / car_payment_percentage)) / gross_income :=
by sorry

end tax_fraction_proof_l268_26888


namespace sugar_mixture_theorem_l268_26838

/-- Given a bowl of sugar with the following properties:
  * Initially contains 320 grams of pure white sugar
  * Mixture Y is formed by removing x grams of white sugar and adding x grams of brown sugar
  * In Mixture Y, the ratio of white sugar to brown sugar is w:b in lowest terms
  * Mixture Z is formed by removing x grams of Mixture Y and adding x grams of brown sugar
  * In Mixture Z, the ratio of white sugar to brown sugar is 49:15
  Prove that x + w + b = 48 -/
theorem sugar_mixture_theorem (x w b : ℕ) : 
  x > 0 ∧ x < 320 ∧ 
  (320 - x : ℚ) / x = w / b ∧ 
  (320 - x) * (320 - x) / (320 : ℚ) / ((2 * x - x^2 / 320 : ℚ)) = 49 / 15 →
  x + w + b = 48 :=
by sorry

end sugar_mixture_theorem_l268_26838


namespace trapezoid_minimum_distance_l268_26861

-- Define the trapezoid ABCD
def Trapezoid (A B C D : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 0 ∧
  B.1 = 0 ∧ B.2 = 12 ∧
  C.1 = 10 ∧ C.2 = 12 ∧
  D.1 = 10 ∧ D.2 = 6

-- Define the circle centered at C with radius 8
def Circle (C F : ℝ × ℝ) : Prop :=
  (F.1 - C.1)^2 + (F.2 - C.2)^2 = 64

-- Define point E on AB
def PointOnAB (A B E : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

-- Define the theorem
theorem trapezoid_minimum_distance (A B C D E F : ℝ × ℝ) :
  Trapezoid A B C D →
  Circle C F →
  PointOnAB A B E →
  (∀ E' F', PointOnAB A B E' → Circle C F' →
    Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) + Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) ≤
    Real.sqrt ((D.1 - E'.1)^2 + (D.2 - E'.2)^2) + Real.sqrt ((E'.1 - F'.1)^2 + (E'.2 - F'.2)^2)) →
  E.2 - A.2 = 4.5 :=
sorry

end trapezoid_minimum_distance_l268_26861


namespace unique_solution_system_l268_26860

theorem unique_solution_system :
  ∃! (x y z : ℝ),
    x^2 - 2*x - 4*z = 3 ∧
    y^2 - 2*y - 2*x = -14 ∧
    z^2 - 4*y - 4*z = -18 ∧
    x = 2 ∧ y = 3 ∧ z = 4 := by
  sorry

end unique_solution_system_l268_26860


namespace race_speed_ratio_l268_26850

theorem race_speed_ratio (total_distance : ℝ) (head_start : ℝ) (speed_A : ℝ) (speed_B : ℝ) 
  (h1 : total_distance = 128)
  (h2 : head_start = 64)
  (h3 : total_distance / speed_A = (total_distance - head_start) / speed_B) :
  speed_A / speed_B = 2 := by
  sorry

end race_speed_ratio_l268_26850


namespace stating_solve_age_problem_l268_26842

/-- Represents the age-related problem described in the question. -/
def AgeProblem (current_age : ℕ) (years_ago : ℕ) : Prop :=
  3 * (current_age + 3) - 3 * (current_age - years_ago) = current_age

/-- 
Theorem stating that given the person's current age of 18, 
the number of years ago referred to in their statement is 3.
-/
theorem solve_age_problem : 
  ∃ (years_ago : ℕ), AgeProblem 18 years_ago ∧ years_ago = 3 :=
sorry

end stating_solve_age_problem_l268_26842


namespace marjs_wallet_problem_l268_26835

/-- Prove that given the conditions in Marj's wallet problem, the value of each of the two bills is $20. -/
theorem marjs_wallet_problem (bill_value : ℚ) : 
  (2 * bill_value + 3 * 5 + 4.5 = 42 + 17.5) → bill_value = 20 := by
  sorry

end marjs_wallet_problem_l268_26835


namespace sqrt_65_greater_than_8_l268_26843

theorem sqrt_65_greater_than_8 : Real.sqrt 65 > 8 := by
  sorry

end sqrt_65_greater_than_8_l268_26843


namespace x_minus_y_times_x_plus_y_equals_95_l268_26857

theorem x_minus_y_times_x_plus_y_equals_95 (x y : ℤ) (h1 : x = 12) (h2 : y = 7) : 
  (x - y) * (x + y) = 95 := by
sorry

end x_minus_y_times_x_plus_y_equals_95_l268_26857


namespace proposition_truth_l268_26816

theorem proposition_truth : 
  -- Proposition A
  (∃ a b m : ℝ, a < b ∧ ¬(a * m^2 < b * m^2)) ∧
  -- Proposition B
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
  -- Proposition C
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) ∧
  -- Proposition D
  ((∀ x : ℝ, x > 1 → 1/x < 1) ∧ (∃ x : ℝ, 1/x < 1 ∧ ¬(x > 1))) :=
by sorry

end proposition_truth_l268_26816


namespace triangle_area_is_64_l268_26882

/-- The area of the triangle bounded by y = x, y = -x, and y = 8 -/
def triangleArea : ℝ := 64

/-- The first bounding line of the triangle -/
def line1 (x : ℝ) : ℝ := x

/-- The second bounding line of the triangle -/
def line2 (x : ℝ) : ℝ := -x

/-- The third bounding line of the triangle -/
def line3 : ℝ := 8

theorem triangle_area_is_64 :
  triangleArea = (1/2) * (line3 - line1 0) * (line3 - line2 0) :=
by sorry

end triangle_area_is_64_l268_26882


namespace green_ball_count_l268_26884

theorem green_ball_count (blue_count : ℕ) (ratio_blue : ℕ) (ratio_green : ℕ) 
  (h1 : blue_count = 16)
  (h2 : ratio_blue = 4)
  (h3 : ratio_green = 3) :
  (blue_count * ratio_green) / ratio_blue = 12 :=
by sorry

end green_ball_count_l268_26884


namespace cube_sum_digits_eq_square_self_l268_26874

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The set of solutions to the problem -/
def solutionSet : Set ℕ := {1, 27}

/-- The main theorem -/
theorem cube_sum_digits_eq_square_self :
  ∀ n : ℕ, n < 1000 → (sumOfDigits n)^3 = n^2 ↔ n ∈ solutionSet := by sorry

end cube_sum_digits_eq_square_self_l268_26874


namespace basketball_court_perimeter_l268_26848

/-- The perimeter of a rectangular basketball court is 96 meters -/
theorem basketball_court_perimeter :
  ∀ (length width : ℝ),
  length = width + 14 →
  (length = 31 ∧ width = 17) →
  2 * (length + width) = 96 := by
  sorry

end basketball_court_perimeter_l268_26848


namespace circle_and_distance_l268_26856

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = 2 * ((P.1 - 1)^2 + P.2^2)

-- Define the circle C
def C : Set (ℝ × ℝ) :=
  {P | (P.1 - 3)^2 + P.2^2 = 8}

-- Define the parabola
def parabola : Set (ℝ × ℝ) :=
  {P | P.2^2 = P.1}

theorem circle_and_distance :
  (∀ P, P_condition P → P ∈ C) ∧
  (∃ Q ∈ parabola, ∀ R ∈ parabola, 
    dist (3, 0) Q ≤ dist (3, 0) R ∧ 
    dist (3, 0) Q = Real.sqrt 11 / 2) :=
sorry

end circle_and_distance_l268_26856


namespace geometric_sequence_product_l268_26896

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → ∃ r : ℝ, a (n + 1) = a n * r

-- Define the problem statement
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, n > 0 → a n > 0) →
  a 1 * a 99 = 16 →
  a 1 + a 99 = 10 →
  a 40 * a 50 * a 60 = 64 := by
  sorry

end geometric_sequence_product_l268_26896


namespace yoki_cans_count_l268_26839

def total_cans : ℕ := 85
def ladonna_cans : ℕ := 25
def prikya_cans : ℕ := 2 * ladonna_cans
def avi_initial_cans : ℕ := 8
def avi_remaining_cans : ℕ := avi_initial_cans / 2

theorem yoki_cans_count : 
  total_cans - (ladonna_cans + prikya_cans + avi_remaining_cans) = 6 := by
  sorry

end yoki_cans_count_l268_26839


namespace scientific_notation_15510000_l268_26807

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_15510000 :
  toScientificNotation 15510000 = ScientificNotation.mk 1.551 7 (by sorry) :=
sorry

end scientific_notation_15510000_l268_26807


namespace parallel_vectors_x_value_l268_26876

-- Define the vectors
def a : ℝ × ℝ := (4, -3)
def b (x : ℝ) : ℝ × ℝ := (x, 6)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem parallel_vectors_x_value :
  parallel a (b x) → x = -8 :=
by
  sorry

end parallel_vectors_x_value_l268_26876


namespace equation_solution_l268_26809

theorem equation_solution : 
  ∃! x : ℝ, (x - 60) / 3 = (4 - 3*x) / 6 ∧ x = 24.8 := by sorry

end equation_solution_l268_26809


namespace probability_yellow_second_marble_l268_26892

-- Define the number of marbles in each bag
def bag_A_white : ℕ := 5
def bag_A_black : ℕ := 2
def bag_B_yellow : ℕ := 4
def bag_B_blue : ℕ := 5
def bag_C_yellow : ℕ := 3
def bag_C_blue : ℕ := 4
def bag_D_yellow : ℕ := 8
def bag_D_blue : ℕ := 2

-- Define the probabilities of drawing from each bag
def prob_white_A : ℚ := bag_A_white / (bag_A_white + bag_A_black)
def prob_black_A : ℚ := bag_A_black / (bag_A_white + bag_A_black)
def prob_yellow_B : ℚ := bag_B_yellow / (bag_B_yellow + bag_B_blue)
def prob_yellow_C : ℚ := bag_C_yellow / (bag_C_yellow + bag_C_blue)
def prob_yellow_D : ℚ := bag_D_yellow / (bag_D_yellow + bag_D_blue)

-- Assume equal probability of odd and even weight for black marbles
def prob_odd_weight : ℚ := 1/2
def prob_even_weight : ℚ := 1/2

-- Define the theorem
theorem probability_yellow_second_marble :
  prob_white_A * prob_yellow_B +
  prob_black_A * prob_odd_weight * prob_yellow_C +
  prob_black_A * prob_even_weight * prob_yellow_D = 211/245 := by
  sorry

end probability_yellow_second_marble_l268_26892


namespace inverse_composition_l268_26826

-- Define the function f
def f : ℕ → ℕ
| 3 => 10
| 4 => 17
| 5 => 26
| 6 => 37
| 7 => 50
| _ => 0  -- Default case for other inputs

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 10 => 3
| 17 => 4
| 26 => 5
| 37 => 6
| 50 => 7
| _ => 0  -- Default case for other inputs

-- Theorem statement
theorem inverse_composition :
  f_inv (f_inv 50 * f_inv 10 + f_inv 26) = 5 := by
  sorry

end inverse_composition_l268_26826


namespace sum_of_first_n_naturals_l268_26815

theorem sum_of_first_n_naturals (n : ℕ) : 
  (List.range (n + 1)).sum = n * (n + 1) / 2 := by
  sorry

end sum_of_first_n_naturals_l268_26815


namespace greatest_integer_satisfying_inequality_l268_26812

theorem greatest_integer_satisfying_inequality :
  ∀ y : ℤ, (3 * |y| + 6 < 24) → y ≤ 5 ∧ ∃ (z : ℤ), z > 5 ∧ ¬(3 * |z| + 6 < 24) := by
  sorry

end greatest_integer_satisfying_inequality_l268_26812


namespace greatest_common_divisor_under_30_l268_26855

theorem greatest_common_divisor_under_30 : ∃ (d : ℕ), d = 18 ∧ 
  d ∣ 450 ∧ d ∣ 90 ∧ d < 30 ∧ 
  ∀ (x : ℕ), x ∣ 450 ∧ x ∣ 90 ∧ x < 30 → x ≤ d :=
by sorry

end greatest_common_divisor_under_30_l268_26855


namespace multiply_decimals_l268_26810

theorem multiply_decimals : 3.6 * 0.05 = 0.18 := by
  sorry

end multiply_decimals_l268_26810


namespace sqrt_360_simplification_l268_26846

theorem sqrt_360_simplification : Real.sqrt 360 = 6 * Real.sqrt 10 := by
  sorry

end sqrt_360_simplification_l268_26846


namespace second_term_of_arithmetic_sequence_l268_26821

def arithmetic_sequence (a₁ a₂ a₃ : ℤ) : Prop :=
  a₂ - a₁ = a₃ - a₂

theorem second_term_of_arithmetic_sequence :
  ∀ y : ℤ, arithmetic_sequence (3^2) y (3^4) → y = 45 :=
by
  sorry

end second_term_of_arithmetic_sequence_l268_26821


namespace min_a_for_ln_inequality_l268_26899

/-- The minimum value of a for which ln x ≤ ax + 1 holds for all x > 0 is 1/e^2 -/
theorem min_a_for_ln_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → Real.log x ≤ a * x + 1) ∧ 
  (∀ (a : ℝ), (∀ (x : ℝ), x > 0 → Real.log x ≤ a * x + 1) → a ≥ 1 / Real.exp 2) ∧
  (∃ (a : ℝ), a = 1 / Real.exp 2 ∧ ∀ (x : ℝ), x > 0 → Real.log x ≤ a * x + 1) :=
by sorry

end min_a_for_ln_inequality_l268_26899


namespace total_blue_balloons_l268_26881

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 9

/-- The number of blue balloons Sally has -/
def sally_balloons : ℕ := 5

/-- The number of blue balloons Jessica has -/
def jessica_balloons : ℕ := 2

/-- The total number of blue balloons -/
def total_balloons : ℕ := joan_balloons + sally_balloons + jessica_balloons

theorem total_blue_balloons : total_balloons = 16 := by
  sorry

end total_blue_balloons_l268_26881
