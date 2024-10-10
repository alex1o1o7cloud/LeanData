import Mathlib

namespace shiela_paper_stars_l1275_127578

/-- The number of classmates Shiela has -/
def num_classmates : ℕ := 9

/-- The number of stars Shiela places in each bottle -/
def stars_per_bottle : ℕ := 5

/-- The total number of paper stars Shiela prepared -/
def total_stars : ℕ := num_classmates * stars_per_bottle

/-- Theorem stating that the total number of paper stars Shiela prepared is 45 -/
theorem shiela_paper_stars : total_stars = 45 := by
  sorry

end shiela_paper_stars_l1275_127578


namespace complex_number_in_first_quadrant_l1275_127531

theorem complex_number_in_first_quadrant 
  (m n : ℝ) 
  (h : (m : ℂ) / (1 + Complex.I) = 1 - n * Complex.I) : 
  m > 0 ∧ n > 0 := by
sorry

end complex_number_in_first_quadrant_l1275_127531


namespace rhombus_inscribed_circle_area_ratio_l1275_127520

theorem rhombus_inscribed_circle_area_ratio (d₁ d₂ : ℝ) (h : d₁ / d₂ = 3 / 4) :
  let r := d₁ * d₂ / (2 * Real.sqrt ((d₁/2)^2 + (d₂/2)^2))
  (d₁ * d₂ / 2) / (π * r^2) = 25 / (6 * π) := by
  sorry

end rhombus_inscribed_circle_area_ratio_l1275_127520


namespace min_tiles_for_floor_l1275_127588

/-- Represents the dimensions of a rectangular shape in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangular shape given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Calculates the number of tiles needed to cover a floor -/
def tilesNeeded (floorDim : Dimensions) (tileDim : Dimensions) : ℕ :=
  (area floorDim) / (area tileDim)

theorem min_tiles_for_floor : 
  let tileDim : Dimensions := ⟨3, 4⟩
  let floorDimFeet : Dimensions := ⟨2, 5⟩
  let floorDimInches : Dimensions := ⟨feetToInches floorDimFeet.length, feetToInches floorDimFeet.width⟩
  tilesNeeded floorDimInches tileDim = 120 := by
  sorry

end min_tiles_for_floor_l1275_127588


namespace two_true_propositions_l1275_127596

theorem two_true_propositions :
  let P1 := ∀ a b c : ℝ, a > b → a*c^2 > b*c^2
  let P2 := ∀ a b c : ℝ, a*c^2 > b*c^2 → a > b
  let P3 := ∀ a b c : ℝ, a ≤ b → a*c^2 ≤ b*c^2
  let P4 := ∀ a b c : ℝ, a*c^2 ≤ b*c^2 → a ≤ b
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) ∨
  (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) ∨
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4) ∨
  (P1 ∧ ¬P2 ∧ ¬P3 ∧ P4) :=
by
  sorry

end two_true_propositions_l1275_127596


namespace total_dimes_l1275_127587

-- Define the initial number of dimes Melanie had
def initial_dimes : Nat := 19

-- Define the number of dimes given by her dad
def dimes_from_dad : Nat := 39

-- Define the number of dimes given by her mother
def dimes_from_mom : Nat := 25

-- Theorem to prove the total number of dimes
theorem total_dimes : 
  initial_dimes + dimes_from_dad + dimes_from_mom = 83 := by
  sorry

end total_dimes_l1275_127587


namespace sum_of_sqrt_inequality_l1275_127582

theorem sum_of_sqrt_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c = 1) : 
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) > 2 + Real.sqrt 5 := by
  sorry

end sum_of_sqrt_inequality_l1275_127582


namespace insurance_payment_percentage_l1275_127536

theorem insurance_payment_percentage
  (total_cost : ℝ)
  (individual_payment_percentage : ℝ)
  (individual_payment : ℝ)
  (h1 : total_cost = 110000)
  (h2 : individual_payment_percentage = 20)
  (h3 : individual_payment = 22000)
  (h4 : individual_payment = (individual_payment_percentage / 100) * total_cost) :
  100 - individual_payment_percentage = 80 := by
  sorry

end insurance_payment_percentage_l1275_127536


namespace axis_symmetry_implies_equal_coefficients_l1275_127570

theorem axis_symmetry_implies_equal_coefficients 
  (a b : ℝ) (h : a * b ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (2 * x) + b * Real.cos (2 * x)
  (∀ x, f (π/8 + x) = f (π/8 - x)) → a = b := by
  sorry

end axis_symmetry_implies_equal_coefficients_l1275_127570


namespace leak_emptying_time_l1275_127509

theorem leak_emptying_time (fill_time_no_leak fill_time_with_leak : ℝ) 
  (h1 : fill_time_no_leak = 8)
  (h2 : fill_time_with_leak = 12) :
  let fill_rate := 1 / fill_time_no_leak
  let combined_rate := 1 / fill_time_with_leak
  let leak_rate := fill_rate - combined_rate
  24 = 1 / leak_rate := by
sorry

end leak_emptying_time_l1275_127509


namespace tank_b_one_third_full_time_l1275_127561

/-- Represents a rectangular tank with given dimensions -/
structure RectangularTank where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular tank -/
def RectangularTank.volume (tank : RectangularTank) : ℝ :=
  tank.length * tank.width * tank.height

/-- Represents the filling process of a tank -/
structure TankFilling where
  tank : RectangularTank
  fillRate : ℝ  -- in cm³/s

/-- Theorem: Tank B will be 1/3 full after 30 seconds -/
theorem tank_b_one_third_full_time (tank_b : TankFilling) 
    (h1 : tank_b.tank.length = 5)
    (h2 : tank_b.tank.width = 9)
    (h3 : tank_b.tank.height = 8)
    (h4 : tank_b.fillRate = 4) : 
    tank_b.fillRate * 30 = (1/3) * tank_b.tank.volume := by
  sorry

#check tank_b_one_third_full_time

end tank_b_one_third_full_time_l1275_127561


namespace complex_fraction_simplification_l1275_127526

theorem complex_fraction_simplification :
  (3 + 4 * Complex.I) / (5 - 2 * Complex.I) = 7/29 + 26/29 * Complex.I :=
by sorry

end complex_fraction_simplification_l1275_127526


namespace polynomial_factorization_l1275_127563

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 6*x^2 - 4 = (x - Real.sqrt 2)^3 * (x + Real.sqrt 2)^3 := by
  sorry

end polynomial_factorization_l1275_127563


namespace birds_on_fence_l1275_127573

/-- Given a number of initial birds, additional birds, and additional storks,
    calculate the total number of birds on the fence. -/
def total_birds (initial : ℕ) (additional : ℕ) (storks : ℕ) : ℕ :=
  initial + additional + storks

/-- Theorem stating that with 6 initial birds, 4 additional birds, and 8 storks,
    the total number of birds on the fence is 18. -/
theorem birds_on_fence :
  total_birds 6 4 8 = 18 := by
  sorry

end birds_on_fence_l1275_127573


namespace milk_sharing_l1275_127556

theorem milk_sharing (don_milk : ℚ) (rachel_portion : ℚ) (rachel_milk : ℚ) : 
  don_milk = 3 / 7 → 
  rachel_portion = 1 / 2 → 
  rachel_milk = rachel_portion * don_milk → 
  rachel_milk = 3 / 14 := by
sorry

end milk_sharing_l1275_127556


namespace electronics_store_purchase_l1275_127532

theorem electronics_store_purchase (total people_tv people_computer people_both : ℕ) 
  (h1 : total = 15)
  (h2 : people_tv = 9)
  (h3 : people_computer = 7)
  (h4 : people_both = 3)
  : total - (people_tv + people_computer - people_both) = 2 :=
by sorry

end electronics_store_purchase_l1275_127532


namespace cartesian_oval_properties_l1275_127535

-- Define the Cartesian oval
def cartesian_oval (x y : ℝ) : Prop := x^3 + y^3 - 3*x*y = 0

theorem cartesian_oval_properties :
  -- 1. The curve does not pass through the third quadrant
  (∀ x y : ℝ, cartesian_oval x y → ¬(x < 0 ∧ y < 0)) ∧
  -- 2. The curve is symmetric about the line y = x
  (∀ x y : ℝ, cartesian_oval x y ↔ cartesian_oval y x) ∧
  -- 3. The curve has no common point with the line x + y = -1
  (∀ x y : ℝ, cartesian_oval x y → x + y ≠ -1) :=
by sorry

end cartesian_oval_properties_l1275_127535


namespace linear_function_composition_l1275_127584

/-- A linear function from ℝ to ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x - 1) →
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
by sorry

end linear_function_composition_l1275_127584


namespace quadratic_function_range_l1275_127521

/-- A quadratic function f(x) = ax^2 + bx satisfying given conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x^2 + b * x

theorem quadratic_function_range (f : ℝ → ℝ) 
  (hf : QuadraticFunction f)
  (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2)
  (h2 : 2 ≤ f 1 ∧ f 1 ≤ 4) :
  5 ≤ f (-2) ∧ f (-2) ≤ 10 := by
  sorry

end quadratic_function_range_l1275_127521


namespace yz_circle_radius_l1275_127516

/-- A sphere intersecting two planes -/
structure IntersectingSphere where
  /-- Center of the circle in xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in xy-plane -/
  xy_radius : ℝ
  /-- Center of the circle in yz-plane -/
  yz_center : ℝ × ℝ × ℝ

/-- Theorem: The radius of the circle formed by the intersection of the sphere and the yz-plane -/
theorem yz_circle_radius (s : IntersectingSphere) 
  (h_xy : s.xy_center = (3, 5, -2) ∧ s.xy_radius = 3)
  (h_yz : s.yz_center = (-2, 5, 3)) :
  ∃ r : ℝ, r = Real.sqrt 46 ∧ 
  r = Real.sqrt ((Real.sqrt 50 : ℝ) ^ 2 - 2 ^ 2) := by
  sorry

end yz_circle_radius_l1275_127516


namespace sum_square_plus_sqrt_sum_squares_l1275_127522

theorem sum_square_plus_sqrt_sum_squares :
  (5 + 9)^2 + Real.sqrt (5^2 + 9^2) = 196 + Real.sqrt 106 := by
  sorry

end sum_square_plus_sqrt_sum_squares_l1275_127522


namespace inequality_proof_l1275_127551

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a + b < 2 * b := by
  sorry

end inequality_proof_l1275_127551


namespace rational_equation_solution_l1275_127504

theorem rational_equation_solution : 
  ∀ x : ℝ, x ≠ 2 → 
  ((3 * x - 9) / (x^2 - 6*x + 8) = (x + 1) / (x - 2)) ↔ 
  (x = 1 ∨ x = 5) :=
by sorry

end rational_equation_solution_l1275_127504


namespace rice_yield_increase_l1275_127552

theorem rice_yield_increase : 
  let yield_changes : List Int := [50, -35, 10, -16, 27, -5, -20, 35]
  yield_changes.sum = 46 := by sorry

end rice_yield_increase_l1275_127552


namespace train_length_approximation_l1275_127553

/-- The length of a train given its speed and time to cross a fixed point -/
def trainLength (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train crossing a telegraph post in 13 seconds at 58.15384615384615 m/s has a length of approximately 756 meters -/
theorem train_length_approximation :
  let speed : ℝ := 58.15384615384615
  let time : ℝ := 13
  let length := trainLength speed time
  ∃ ε > 0, |length - 756| < ε :=
by sorry

end train_length_approximation_l1275_127553


namespace line_AB_passes_through_fixed_point_l1275_127513

-- Define the hyperbola D
def hyperbolaD (x y : ℝ) : Prop := y^2/2 - x^2 = 1/3

-- Define the parabola C
def parabolaC (x y : ℝ) : Prop := x^2 = 4*y

-- Define the point P on parabola C
def P : ℝ × ℝ := (2, 1)

-- Define a point on parabola C
def pointOnParabolaC (x y : ℝ) : Prop := parabolaC x y

-- Define the perpendicular condition for PA and PB
def perpendicularCondition (x1 y1 x2 y2 : ℝ) : Prop :=
  ((y1 - 1) / (x1 - 2)) * ((y2 - 1) / (x2 - 2)) = -1

-- The main theorem
theorem line_AB_passes_through_fixed_point :
  ∀ (x1 y1 x2 y2 : ℝ),
  pointOnParabolaC x1 y1 →
  pointOnParabolaC x2 y2 →
  perpendicularCondition x1 y1 x2 y2 →
  ∃ (t : ℝ), t ∈ (Set.Icc 0 1) ∧ 
  (t * x1 + (1 - t) * x2 = -2) ∧
  (t * y1 + (1 - t) * y2 = 5) :=
sorry

end line_AB_passes_through_fixed_point_l1275_127513


namespace max_value_of_a_l1275_127533

theorem max_value_of_a : 
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → 
  ∃ a_max : ℝ, a_max = 1 ∧ ∀ a : ℝ, (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ a_max :=
by sorry

end max_value_of_a_l1275_127533


namespace infinite_solutions_imply_a_equals_two_l1275_127568

/-- 
Given a system of equations:
  ax + y - 1 = 0
  4x + ay - 2 = 0
If there are infinitely many solutions, then a = 2.
-/
theorem infinite_solutions_imply_a_equals_two (a : ℝ) :
  (∀ x y : ℝ, a * x + y - 1 = 0 ∧ 4 * x + a * y - 2 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    a * x₁ + y₁ - 1 = 0 ∧ 4 * x₁ + a * y₁ - 2 = 0 ∧
    a * x₂ + y₂ - 1 = 0 ∧ 4 * x₂ + a * y₂ - 2 = 0) →
  a = 2 :=
by sorry


end infinite_solutions_imply_a_equals_two_l1275_127568


namespace first_set_cost_l1275_127527

/-- The cost of a football in dollars -/
def football_cost : ℝ := 35

/-- The cost of a soccer ball in dollars -/
def soccer_cost : ℝ := 50

/-- The cost of 2 footballs and 3 soccer balls in dollars -/
def two_footballs_three_soccer_cost : ℝ := 220

theorem first_set_cost : 3 * football_cost + soccer_cost = 155 :=
  by sorry

end first_set_cost_l1275_127527


namespace profit_maximization_and_threshold_l1275_127569

def price (x : ℕ) : ℕ :=
  if 1 ≤ x ∧ x < 40 then x + 45
  else if 40 ≤ x ∧ x ≤ 70 then 85
  else 0

def dailySales (x : ℕ) : ℕ := 150 - 2 * x

def costPrice : ℕ := 30

def dailyProfit (x : ℕ) : ℤ :=
  if 1 ≤ x ∧ x < 40 then -2 * x^2 + 120 * x + 2250
  else if 40 ≤ x ∧ x ≤ 70 then -110 * x + 8250
  else 0

theorem profit_maximization_and_threshold (x : ℕ) :
  (∀ x, 1 ≤ x ∧ x ≤ 70 → dailyProfit x ≤ dailyProfit 30) ∧
  dailyProfit 30 = 4050 ∧
  (Finset.filter (fun x => dailyProfit x ≥ 3250) (Finset.range 70)).card = 36 := by
  sorry


end profit_maximization_and_threshold_l1275_127569


namespace polygon_distance_inequality_l1275_127525

-- Define a polygon type
structure Polygon :=
  (vertices : List (ℝ × ℝ))

-- Define the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := sorry

-- Define the sum of distances from a point to vertices
def sum_distances_to_vertices (o : ℝ × ℝ) (p : Polygon) : ℝ := sorry

-- Define the sum of distances from a point to sides
def sum_distances_to_sides (o : ℝ × ℝ) (p : Polygon) : ℝ := sorry

-- State the theorem
theorem polygon_distance_inequality (o : ℝ × ℝ) (m : Polygon) :
  let ρ := perimeter m
  let d := sum_distances_to_vertices o m
  let h := sum_distances_to_sides o m
  d^2 - h^2 ≥ ρ^2 / 4 := by
  sorry

end polygon_distance_inequality_l1275_127525


namespace greatest_piece_length_l1275_127581

theorem greatest_piece_length (a b c : ℕ) (ha : a = 45) (hb : b = 75) (hc : c = 90) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end greatest_piece_length_l1275_127581


namespace janelle_has_72_marbles_l1275_127506

/-- The number of marbles Janelle has after buying and gifting some marbles -/
def janelles_marbles : ℕ :=
  let initial_green := 26
  let blue_bags := 6
  let marbles_per_bag := 10
  let gifted_green := 6
  let gifted_blue := 8
  
  let total_blue := blue_bags * marbles_per_bag
  let total_before_gift := initial_green + total_blue
  let total_gifted := gifted_green + gifted_blue
  
  total_before_gift - total_gifted

/-- Theorem stating that Janelle has 72 marbles after the transactions -/
theorem janelle_has_72_marbles : janelles_marbles = 72 := by
  sorry

end janelle_has_72_marbles_l1275_127506


namespace rachel_total_problems_l1275_127593

/-- The number of math problems Rachel solved in total -/
def total_problems (problems_per_minute : ℕ) (minutes_solved : ℕ) (problems_next_day : ℕ) : ℕ :=
  problems_per_minute * minutes_solved + problems_next_day

/-- Proof that Rachel solved 76 math problems in total -/
theorem rachel_total_problems :
  total_problems 5 12 16 = 76 := by
  sorry

end rachel_total_problems_l1275_127593


namespace min_value_theorem_l1275_127550

theorem min_value_theorem (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_cond : x + y + z + w = 2)
  (prod_cond : x * y * z * w = 1/16) :
  (∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → 
    a + b + c + d = 2 → a * b * c * d = 1/16 → 
    (x + y + z) / (x * y * z * w) ≤ (a + b + c) / (a * b * c * d)) →
  (x + y + z) / (x * y * z * w) = 24 :=
sorry

end min_value_theorem_l1275_127550


namespace problem_statement_l1275_127585

theorem problem_statement (x y : ℝ) (h1 : x + y = 7) (h2 : x * y = 10) : 3 * x^2 + 3 * y^2 = 87 := by
  sorry

end problem_statement_l1275_127585


namespace existence_of_unsolvable_linear_system_l1275_127537

theorem existence_of_unsolvable_linear_system :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x y : ℝ, a₁ * x + b₁ * y ≠ c₁ ∨ a₂ * x + b₂ * y ≠ c₂) :=
by sorry

end existence_of_unsolvable_linear_system_l1275_127537


namespace prime_square_sum_equation_l1275_127539

theorem prime_square_sum_equation :
  ∀ (a b c k : ℕ),
    Prime a ∧ Prime b ∧ Prime c ∧ k > 0 ∧
    a^2 + b^2 + 16*c^2 = 9*k^2 + 1 →
    ((a = 3 ∧ b = 3 ∧ c = 2 ∧ k = 3) ∨
     (a = 3 ∧ b = 37 ∧ c = 3 ∧ k = 13) ∨
     (a = 37 ∧ b = 3 ∧ c = 3 ∧ k = 13) ∨
     (a = 3 ∧ b = 17 ∧ c = 3 ∧ k = 7) ∨
     (a = 17 ∧ b = 3 ∧ c = 3 ∧ k = 7)) :=
by
  sorry

end prime_square_sum_equation_l1275_127539


namespace max_queens_8x8_l1275_127577

/-- Represents a chessboard configuration -/
def ChessBoard := Fin 8 → Fin 8

/-- Checks if two positions are on the same diagonal -/
def onSameDiagonal (p1 p2 : Fin 8 × Fin 8) : Prop :=
  (p1.1 : ℤ) - (p2.1 : ℤ) = (p1.2 : ℤ) - (p2.2 : ℤ) ∨
  (p1.1 : ℤ) - (p2.1 : ℤ) = (p2.2 : ℤ) - (p1.2 : ℤ)

/-- Checks if a chessboard configuration is valid (no queens attack each other) -/
def isValidConfiguration (board : ChessBoard) : Prop :=
  ∀ i j : Fin 8, i ≠ j →
    board i ≠ board j ∧
    ¬onSameDiagonal (i, board i) (j, board j)

/-- The theorem stating that the maximum number of non-attacking queens on an 8x8 chessboard is 8 -/
theorem max_queens_8x8 :
  (∃ (board : ChessBoard), isValidConfiguration board) ∧
  (∀ (n : ℕ) (f : Fin n → Fin 8 × Fin 8),
    (∀ i j : Fin n, i ≠ j → f i ≠ f j ∧ ¬onSameDiagonal (f i) (f j)) →
    n ≤ 8) :=
sorry

end max_queens_8x8_l1275_127577


namespace consecutive_integers_cube_sum_l1275_127524

theorem consecutive_integers_cube_sum (n : ℤ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 2106 →
  (n - 1)^3 + n^3 + (n + 1)^3 + (n + 2)^3 = 45900 := by
sorry

end consecutive_integers_cube_sum_l1275_127524


namespace two_true_propositions_l1275_127594

-- Define the original proposition
def original_prop (a b c : ℝ) : Prop :=
  a > b → a * c^2 > b * c^2

-- Define the converse of the original proposition
def converse_prop (a b c : ℝ) : Prop :=
  a * c^2 > b * c^2 → a > b

-- Define the negation of the original proposition
def negation_prop (a b c : ℝ) : Prop :=
  ¬(a > b → a * c^2 > b * c^2)

-- Theorem statement
theorem two_true_propositions :
  ∃ (p q : Prop) (r : Prop),
    (p = ∀ a b c : ℝ, original_prop a b c) ∧
    (q = ∀ a b c : ℝ, converse_prop a b c) ∧
    (r = ∀ a b c : ℝ, negation_prop a b c) ∧
    ((¬p ∧ q ∧ r) ∨ (p ∧ ¬q ∧ r) ∨ (p ∧ q ∧ ¬r)) :=
sorry

end two_true_propositions_l1275_127594


namespace base_seven_subtraction_l1275_127583

/-- Represents a number in base 7 --/
def BaseSevenNumber := List Nat

/-- Converts a base 7 number to its decimal representation --/
def to_decimal (n : BaseSevenNumber) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (7 ^ i)) 0

/-- Subtracts two base 7 numbers --/
def base_seven_subtract (a b : BaseSevenNumber) : BaseSevenNumber :=
  sorry

theorem base_seven_subtraction :
  let a : BaseSevenNumber := [4, 1, 2, 3]  -- 3214 in base 7
  let b : BaseSevenNumber := [4, 3, 2, 1]  -- 1234 in base 7
  let result : BaseSevenNumber := [0, 5, 6, 2]  -- 2650 in base 7
  base_seven_subtract a b = result := by sorry

end base_seven_subtraction_l1275_127583


namespace constant_in_toll_formula_l1275_127562

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ :=
  1.50 + constant * (x - 2)

/-- The number of axles on an 18-wheel truck -/
def axles_18_wheel_truck : ℕ := 9

/-- The toll for an 18-wheel truck -/
def toll_18_wheel_truck : ℝ := 5

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll_formula constant axles_18_wheel_truck = toll_18_wheel_truck ∧ 
    constant = 0.50 := by
  sorry

end constant_in_toll_formula_l1275_127562


namespace inequality_proof_l1275_127547

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end inequality_proof_l1275_127547


namespace inequality_proof_l1275_127590

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : 0 < b) (hb1 : b < 1) :
  ab^2 > ab ∧ ab > a := by
  sorry

end inequality_proof_l1275_127590


namespace parallel_line_plane_conditions_l1275_127555

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the parallel relation
def parallel (x y : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the subset relation for a line in a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem parallel_line_plane_conditions
  (a b : Line) (α : Plane) (h : line_in_plane a α) :
  ¬(∀ (h1 : parallel a b), parallel_line_plane b α) ∧
  ¬(∀ (h2 : parallel_line_plane b α), parallel a b) :=
sorry

end parallel_line_plane_conditions_l1275_127555


namespace candidate_vote_percentage_l1275_127597

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (loss_margin : ℕ)
  (h_total : total_votes = 10000)
  (h_margin : loss_margin = 4000) :
  (total_votes - loss_margin) * 2 * 100 / total_votes = 30 := by
  sorry

end candidate_vote_percentage_l1275_127597


namespace weight_ratio_proof_l1275_127574

/-- Prove the ratio of weight added back to initial weight lost --/
theorem weight_ratio_proof (initial_weight final_weight : ℕ) 
  (first_loss third_loss final_gain : ℕ) (weight_added : ℕ) : 
  initial_weight = 99 →
  final_weight = 81 →
  first_loss = 12 →
  third_loss = 3 * first_loss →
  final_gain = 6 →
  initial_weight - first_loss + weight_added - third_loss + final_gain = final_weight →
  weight_added / first_loss = 2 := by
  sorry

end weight_ratio_proof_l1275_127574


namespace math_class_size_l1275_127575

/-- Proves that the number of students in the mathematics class is 170/3 given the conditions of the problem. -/
theorem math_class_size (total : ℕ) (both : ℕ) (math_twice_physics : Prop) :
  total = 75 →
  both = 10 →
  math_twice_physics →
  (∃ (math physics : ℕ),
    math = (170 : ℚ) / 3 ∧
    physics = (total - both) - (math - both) ∧
    math = 2 * physics) :=
by sorry

end math_class_size_l1275_127575


namespace arithmetic_sequence_sum_l1275_127586

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  ArithmeticSequence a → a 2 = 5 → a 6 = 33 → a 3 + a 5 = 38 := by
  sorry

end arithmetic_sequence_sum_l1275_127586


namespace b_equals_three_l1275_127567

/-- The function f(x) -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 1

/-- f(x) is monotonically increasing in the interval (1, 2) -/
def monotone_increasing_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y ∧ y < 2 → f x < f y

/-- f(x) is monotonically decreasing in the interval (2, 3) -/
def monotone_decreasing_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 2 < x ∧ x < y ∧ y < 3 → f x > f y

/-- Main theorem: b equals 3 -/
theorem b_equals_three :
  ∃ b : ℝ, 
    (monotone_increasing_in_interval (f b)) ∧ 
    (monotone_decreasing_in_interval (f b)) → 
    b = 3 := by sorry

end b_equals_three_l1275_127567


namespace rectangle_area_l1275_127545

theorem rectangle_area (k : ℕ+) : 
  let square_side : ℝ := (16 : ℝ).sqrt
  let rectangle_length : ℝ := k * square_side
  let rectangle_breadth : ℝ := 11
  rectangle_length * rectangle_breadth = 220 :=
by sorry

end rectangle_area_l1275_127545


namespace decimal_representation_of_fraction_l1275_127512

theorem decimal_representation_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.36 ↔ n = 9 ∧ d = 25 :=
sorry

end decimal_representation_of_fraction_l1275_127512


namespace real_part_of_z_l1275_127519

theorem real_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I * Real.sqrt 3) + Complex.I) :
  z.re = 1/2 := by sorry

end real_part_of_z_l1275_127519


namespace ways_to_write_1800_as_sum_of_twos_and_threes_l1275_127576

/-- The number of ways to write a positive integer as a sum of 2s and 3s -/
def num_ways_as_sum_of_twos_and_threes (n : ℕ) : ℕ :=
  (n / 6 + 1) - (n % 6 / 2)

/-- Theorem stating that there are 301 ways to write 1800 as a sum of 2s and 3s -/
theorem ways_to_write_1800_as_sum_of_twos_and_threes :
  num_ways_as_sum_of_twos_and_threes 1800 = 301 := by
  sorry

#eval num_ways_as_sum_of_twos_and_threes 1800

end ways_to_write_1800_as_sum_of_twos_and_threes_l1275_127576


namespace correct_number_proof_l1275_127517

theorem correct_number_proof (n : ℕ) (initial_avg correct_avg : ℚ) 
  (first_error second_error : ℚ) (correct_second : ℚ) : 
  n = 10 → 
  initial_avg = 40.2 → 
  correct_avg = 40.3 → 
  first_error = 19 → 
  second_error = 13 → 
  (n : ℚ) * initial_avg - first_error - second_error + correct_second = (n : ℚ) * correct_avg → 
  correct_second = 33 := by
  sorry

end correct_number_proof_l1275_127517


namespace borrowed_amount_is_2500_l1275_127507

/-- Proves that the borrowed amount is 2500 given the problem conditions --/
theorem borrowed_amount_is_2500 
  (borrowed_rate : ℚ) 
  (lent_rate : ℚ) 
  (time : ℚ) 
  (yearly_gain : ℚ) 
  (h1 : borrowed_rate = 4 / 100)
  (h2 : lent_rate = 6 / 100)
  (h3 : time = 2)
  (h4 : yearly_gain = 100) : 
  ∃ (P : ℚ), P = 2500 ∧ 
    (lent_rate * P * time) - (borrowed_rate * P * time) = yearly_gain * time :=
by sorry

end borrowed_amount_is_2500_l1275_127507


namespace square_sum_theorem_l1275_127502

theorem square_sum_theorem (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 8)
  (eq2 : y^2 + 5*z = -9)
  (eq3 : z^2 + 7*x = -16) :
  x^2 + y^2 + z^2 = 20.75 := by
sorry

end square_sum_theorem_l1275_127502


namespace sum_of_three_numbers_l1275_127546

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + a*c = 131) : 
  a + b + c = 22 := by
sorry

end sum_of_three_numbers_l1275_127546


namespace crayon_eraser_difference_l1275_127505

def prove_crayon_eraser_difference 
  (initial_erasers : ℕ) 
  (initial_crayons : ℕ) 
  (remaining_crayons : ℕ) : Prop :=
  initial_erasers = 457 ∧ 
  initial_crayons = 617 ∧ 
  remaining_crayons = 523 → 
  remaining_crayons - initial_erasers = 66

theorem crayon_eraser_difference : 
  prove_crayon_eraser_difference 457 617 523 :=
by sorry

end crayon_eraser_difference_l1275_127505


namespace distribute_five_balls_four_boxes_l1275_127595

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes :
  distribute_balls 5 4 = 56 := by
  sorry

end distribute_five_balls_four_boxes_l1275_127595


namespace no_solution_to_system_l1275_127529

theorem no_solution_to_system :
  ¬ ∃ (x y z : ℝ), 
    (x^2 - 3*x*y + 2*y^2 - z^2 = 31) ∧
    (-x^2 + 6*y*z + 2*z^2 = 44) ∧
    (x^2 + x*y + 8*z^2 = 100) :=
by sorry

end no_solution_to_system_l1275_127529


namespace product_even_even_is_even_product_odd_odd_is_odd_product_even_odd_is_even_product_odd_even_is_even_l1275_127500

-- Define even and odd integers
def IsEven (n : Int) : Prop := ∃ k : Int, n = 2 * k
def IsOdd (n : Int) : Prop := ∃ k : Int, n = 2 * k + 1

-- Theorem statements
theorem product_even_even_is_even (a b : Int) (ha : IsEven a) (hb : IsEven b) :
  IsEven (a * b) := by sorry

theorem product_odd_odd_is_odd (a b : Int) (ha : IsOdd a) (hb : IsOdd b) :
  IsOdd (a * b) := by sorry

theorem product_even_odd_is_even (a b : Int) (ha : IsEven a) (hb : IsOdd b) :
  IsEven (a * b) := by sorry

theorem product_odd_even_is_even (a b : Int) (ha : IsOdd a) (hb : IsEven b) :
  IsEven (a * b) := by sorry

end product_even_even_is_even_product_odd_odd_is_odd_product_even_odd_is_even_product_odd_even_is_even_l1275_127500


namespace greatest_two_digit_with_digit_product_12_l1275_127598

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 43 :=
sorry

end greatest_two_digit_with_digit_product_12_l1275_127598


namespace exception_pair_of_equations_other_pairs_valid_l1275_127589

theorem exception_pair_of_equations (x : ℝ) : 
  (∃ y, y = x ∧ y = x - 2 ∧ x^2 - 2*x = 0) ↔ False :=
by sorry

theorem other_pairs_valid (x : ℝ) :
  ((∃ y, y = x^2 ∧ y = 2*x ∧ x^2 - 2*x = 0) ∨
   (∃ y, y = x^2 - 2*x ∧ y = 0 ∧ x^2 - 2*x = 0) ∨
   (∃ y, y = x^2 - 2*x + 1 ∧ y = 1 ∧ x^2 - 2*x = 0) ∨
   (∃ y, y = x^2 - 1 ∧ y = 2*x - 1 ∧ x^2 - 2*x = 0)) ↔ True :=
by sorry

end exception_pair_of_equations_other_pairs_valid_l1275_127589


namespace stock_worth_l1275_127592

-- Define the total worth of the stock
variable (X : ℝ)

-- Define the profit percentage on 20% of stock
def profit_percent : ℝ := 0.10

-- Define the loss percentage on 80% of stock
def loss_percent : ℝ := 0.05

-- Define the overall loss
def overall_loss : ℝ := 200

-- Theorem statement
theorem stock_worth :
  (0.20 * X * (1 + profit_percent) + 0.80 * X * (1 - loss_percent) = X - overall_loss) →
  X = 10000 := by
sorry

end stock_worth_l1275_127592


namespace function_increasing_implies_a_leq_neg_two_l1275_127540

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

-- State the theorem
theorem function_increasing_implies_a_leq_neg_two :
  ∀ a : ℝ, (∀ x y : ℝ, -2 < x ∧ x < y ∧ y < 2 → f a x < f a y) → a ≤ -2 :=
by sorry

end function_increasing_implies_a_leq_neg_two_l1275_127540


namespace min_box_value_l1275_127554

theorem min_box_value (a b Box : ℤ) :
  (∀ x, (a * x + b) * (b * x + a) = 26 * x^2 + Box * x + 26) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  (∃ a' b' Box' : ℤ, 
    (∀ x, (a' * x + b') * (b' * x + a') = 26 * x^2 + Box' * x + 26) ∧
    a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' ∧
    Box' < Box) →
  Box ≥ 173 :=
by sorry

end min_box_value_l1275_127554


namespace min_value_2m_plus_n_solution_set_f_gt_5_l1275_127538

-- Define the function f
def f (x m n : ℝ) : ℝ := |x + m| + |2*x - n|

-- Theorem for part I
theorem min_value_2m_plus_n (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f x m n ≥ 1) → 2*m + n ≥ 2 :=
sorry

-- Theorem for part II
theorem solution_set_f_gt_5 :
  {x : ℝ | f x 2 3 > 5} = {x : ℝ | x < 0 ∨ x > 2} :=
sorry

end min_value_2m_plus_n_solution_set_f_gt_5_l1275_127538


namespace quadratic_shift_theorem_l1275_127518

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal and vertical shift to a quadratic function -/
def shift_quadratic (f : QuadraticFunction) (h_shift v_shift : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := -2 * f.a * h_shift
  , c := f.a * h_shift^2 + f.c - v_shift }

theorem quadratic_shift_theorem (f : QuadraticFunction) 
  (h : f.a = -2 ∧ f.b = 0 ∧ f.c = 1) : 
  shift_quadratic f 3 2 = { a := -2, b := 12, c := -1 } := by
  sorry

#check quadratic_shift_theorem

end quadratic_shift_theorem_l1275_127518


namespace trigonometric_properties_l1275_127565

open Real

-- Define the concept of terminal side of an angle
def sameSide (α β : ℝ) : Prop := sorry

-- Define the set of angles with terminal side on x-axis
def xAxisAngles : Set ℝ := { α | ∃ k : ℤ, α = k * π }

-- Define the quadrants
def inFirstOrSecondQuadrant (α : ℝ) : Prop := 0 < α ∧ α < π

theorem trigonometric_properties :
  (∀ α β : ℝ, sameSide α β → sin α = sin β ∧ cos α = cos β) ∧
  (xAxisAngles ≠ { α | ∃ k : ℤ, α = 2 * k * π }) ∧
  (∃ α : ℝ, sin α > 0 ∧ ¬inFirstOrSecondQuadrant α) ∧
  (∃ α β : ℝ, sin α = sin β ∧ ¬(∃ k : ℤ, α = 2 * k * π + β)) := by sorry

end trigonometric_properties_l1275_127565


namespace cost_of_horse_l1275_127579

/-- Given Albert's purchase and sale of horses and cows, prove the cost of a horse -/
theorem cost_of_horse (total_cost : ℝ) (num_horses : ℕ) (num_cows : ℕ) 
  (horse_profit_rate : ℝ) (cow_profit_rate : ℝ) (total_profit : ℝ) :
  total_cost = 13400 ∧ 
  num_horses = 4 ∧ 
  num_cows = 9 ∧
  horse_profit_rate = 0.1 ∧
  cow_profit_rate = 0.2 ∧
  total_profit = 1880 →
  ∃ (horse_cost cow_cost : ℝ),
    num_horses * horse_cost + num_cows * cow_cost = total_cost ∧
    num_horses * horse_cost * horse_profit_rate + num_cows * cow_cost * cow_profit_rate = total_profit ∧
    horse_cost = 2000 :=
by sorry

end cost_of_horse_l1275_127579


namespace johns_allowance_l1275_127560

theorem johns_allowance (A : ℚ) : 
  (A > 0) →
  ((4 / 15 : ℚ) * A = 92 / 100) →
  A = 345 / 100 := by
sorry

end johns_allowance_l1275_127560


namespace range_of_expression_l1275_127534

-- Define the conditions
def condition1 (x y : ℝ) : Prop := -1 < x + y ∧ x + y < 4
def condition2 (x y : ℝ) : Prop := 2 < x - y ∧ x - y < 3

-- Define the expression we're interested in
def expression (x y : ℝ) : ℝ := 3*x + 2*y

-- State the theorem
theorem range_of_expression (x y : ℝ) 
  (h1 : condition1 x y) (h2 : condition2 x y) :
  -3/2 < expression x y ∧ expression x y < 23/2 := by
  sorry

end range_of_expression_l1275_127534


namespace club_has_25_seniors_l1275_127503

/-- Represents a high school club with juniors and seniors -/
structure Club where
  juniors : ℕ
  seniors : ℕ
  project_juniors : ℕ
  project_seniors : ℕ

/-- The conditions of the problem -/
def club_conditions (c : Club) : Prop :=
  c.juniors + c.seniors = 50 ∧
  c.project_juniors = (40 * c.juniors) / 100 ∧
  c.project_seniors = (20 * c.seniors) / 100 ∧
  c.project_juniors = 2 * c.project_seniors

/-- The theorem stating that a club satisfying the conditions has 25 seniors -/
theorem club_has_25_seniors (c : Club) (h : club_conditions c) : c.seniors = 25 := by
  sorry


end club_has_25_seniors_l1275_127503


namespace rectangle_fit_count_l1275_127544

/-- A rectangle with integer coordinates -/
structure Rectangle where
  x : ℤ
  y : ℤ

/-- The region defined by the problem -/
def inRegion (r : Rectangle) : Prop :=
  r.y ≤ 2 * r.x ∧ r.y ≥ -2 ∧ r.x ≤ 10 ∧ r.x ≥ 0

/-- A valid 2x1 rectangle within the region -/
def validRectangle (r : Rectangle) : Prop :=
  inRegion r ∧ inRegion ⟨r.x + 2, r.y⟩

/-- The count of valid rectangles -/
def rectangleCount : ℕ := sorry

theorem rectangle_fit_count : rectangleCount = 34 := by sorry

end rectangle_fit_count_l1275_127544


namespace three_digit_number_divided_by_11_l1275_127557

theorem three_digit_number_divided_by_11 : 
  ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 → 
  (n / 11 = (n / 100)^2 + ((n / 10) % 10)^2 + (n % 10)^2) ↔ 
  (n = 550 ∨ n = 803) := by
sorry

end three_digit_number_divided_by_11_l1275_127557


namespace median_and_mode_are_23_l1275_127514

/-- Represents a shoe size distribution --/
structure ShoeSizeDistribution where
  sizes : List Nat
  frequencies : List Nat
  total_students : Nat

/-- Calculates the median of a shoe size distribution --/
def median (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- Calculates the mode of a shoe size distribution --/
def mode (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- The given shoe size distribution --/
def class_distribution : ShoeSizeDistribution :=
  { sizes := [20, 21, 22, 23, 24],
    frequencies := [2, 8, 9, 19, 2],
    total_students := 40 }

theorem median_and_mode_are_23 :
  median class_distribution = 23 ∧ mode class_distribution = 23 := by
  sorry

end median_and_mode_are_23_l1275_127514


namespace smallest_sum_of_three_primes_l1275_127543

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def digits_used_once (a b c : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ),
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧
    d7 ≠ d8 ∧ d7 ≠ d9 ∧
    d8 ≠ d9 ∧
    d1 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d2 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d3 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d4 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d5 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d6 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d7 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d8 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d9 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    a = d1 * 100 + d2 * 10 + d3 ∧
    b = d4 * 100 + d5 * 10 + d6 ∧
    c = d7 * 100 + d8 * 10 + d9

theorem smallest_sum_of_three_primes :
  ∀ a b c : ℕ,
    is_prime a ∧ is_prime b ∧ is_prime c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    digits_used_once a b c →
    a + b + c ≥ 999 ∧
    (∃ x y z : ℕ, is_prime x ∧ is_prime y ∧ is_prime z ∧
      x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
      digits_used_once x y z ∧
      x + y + z = 999) :=
by sorry

end smallest_sum_of_three_primes_l1275_127543


namespace tangent_line_exists_tangent_line_equation_l1275_127571

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 3

/-- Theorem: There exists a tangent line to y = f(x) passing through (2, -6) -/
theorem tangent_line_exists : 
  ∃ (x₀ : ℝ), 
    (f x₀ + f_deriv x₀ * (2 - x₀) = -6) ∧ 
    ((f_deriv x₀ = -3) ∨ (f_deriv x₀ = 24)) :=
sorry

/-- Theorem: The tangent line equation is y = -3x or y = 24x - 54 -/
theorem tangent_line_equation (x₀ : ℝ) 
  (h : (f x₀ + f_deriv x₀ * (2 - x₀) = -6) ∧ 
       ((f_deriv x₀ = -3) ∨ (f_deriv x₀ = 24))) : 
  (∀ x y, y = -3*x) ∨ (∀ x y, y = 24*x - 54) :=
sorry

end tangent_line_exists_tangent_line_equation_l1275_127571


namespace last_box_contents_l1275_127541

-- Define the total number of bars for each type of chocolate
def total_A : ℕ := 853845
def total_B : ℕ := 537896
def total_C : ℕ := 729763

-- Define the box capacity for each type of chocolate
def capacity_A : ℕ := 9
def capacity_B : ℕ := 11
def capacity_C : ℕ := 15

-- Theorem to prove the number of bars in the last partially filled box for each type
theorem last_box_contents :
  (total_A % capacity_A = 4) ∧
  (total_B % capacity_B = 3) ∧
  (total_C % capacity_C = 8) := by
  sorry

end last_box_contents_l1275_127541


namespace smith_laundry_loads_l1275_127566

/-- The number of bath towels Kylie uses in one month -/
def kylie_towels : ℕ := 3

/-- The number of bath towels Kylie's daughters use in one month -/
def daughters_towels : ℕ := 6

/-- The number of bath towels Kylie's husband uses in one month -/
def husband_towels : ℕ := 3

/-- The number of bath towels that fit in one load of laundry -/
def towels_per_load : ℕ := 4

/-- The total number of bath towels used by the Smith family in one month -/
def total_towels : ℕ := kylie_towels + daughters_towels + husband_towels

/-- The number of laundry loads required to clean all used towels -/
def required_loads : ℕ := (total_towels + towels_per_load - 1) / towels_per_load

theorem smith_laundry_loads : required_loads = 3 := by
  sorry

end smith_laundry_loads_l1275_127566


namespace small_circle_radius_l1275_127564

/-- Given a large circle with radius 10 meters and four congruent smaller circles
    touching at its center, prove that the radius of each smaller circle is 5 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : R = 10 → 2 * r = R → r = 5 := by sorry

end small_circle_radius_l1275_127564


namespace vector_AB_coordinates_and_magnitude_l1275_127548

def OA : ℝ × ℝ := (1, 2)
def OB : ℝ × ℝ := (3, 1)

def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_AB_coordinates_and_magnitude :
  AB = (2, -1) ∧ Real.sqrt ((AB.1)^2 + (AB.2)^2) = Real.sqrt 5 := by
  sorry

end vector_AB_coordinates_and_magnitude_l1275_127548


namespace probability_even_sum_four_primes_l1275_127558

-- Define the set of first twelve prime numbers
def first_twelve_primes : Finset Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

-- Define a function to check if a number is even
def is_even (n : Nat) : Bool := n % 2 = 0

-- Define a function to calculate the sum of a list of numbers
def sum_list (l : List Nat) : Nat := l.foldl (·+·) 0

-- Theorem statement
theorem probability_even_sum_four_primes :
  let all_selections := Finset.powerset first_twelve_primes
  let valid_selections := all_selections.filter (fun s => s.card = 4)
  let even_sum_selections := valid_selections.filter (fun s => is_even (sum_list s.toList))
  (even_sum_selections.card : Rat) / valid_selections.card = 2 / 3 := by
  sorry

end probability_even_sum_four_primes_l1275_127558


namespace unique_solution_l1275_127508

-- Define the system of equations
def system (x y z w : ℝ) : Prop :=
  (x + 1 = z + w + z*w*x) ∧
  (y - 1 = w + x + w*x*y) ∧
  (z + 2 = x + y + x*y*z) ∧
  (w - 2 = y + z + y*z*w)

-- Theorem statement
theorem unique_solution : ∃! (x y z w : ℝ), system x y z w :=
sorry

end unique_solution_l1275_127508


namespace perfect_square_binomial_l1275_127523

theorem perfect_square_binomial (a b : ℝ) : ∃ (x : ℝ), a^2 + 2*a*b + b^2 = x^2 := by
  sorry

end perfect_square_binomial_l1275_127523


namespace expression_simplification_l1275_127580

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 2 + 1) :
  (1 - 1 / (m + 1)) * ((m^2 - 1) / m) = Real.sqrt 2 := by
  sorry

end expression_simplification_l1275_127580


namespace negation_of_existence_negation_of_exponential_equation_l1275_127599

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_exponential_equation :
  (¬ ∃ x : ℝ, Real.exp x = x - 1) ↔ (∀ x : ℝ, Real.exp x ≠ x - 1) :=
by sorry

end negation_of_existence_negation_of_exponential_equation_l1275_127599


namespace triangle_area_l1275_127572

theorem triangle_area (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A = π/3 ∧
  c = 4 ∧
  b = 2 * Real.sqrt 3 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 := by
  sorry

end triangle_area_l1275_127572


namespace second_circle_radius_l1275_127542

/-- Two circles are externally tangent, with one having radius 2. Their common tangent intersects
    another common tangent at a point 4 units away from the point of tangency. -/
structure TangentCircles where
  r : ℝ
  R : ℝ
  tangent_length : ℝ
  h_r : r = 2
  h_tangent : tangent_length = 4

/-- The radius of the second circle is 8. -/
theorem second_circle_radius (tc : TangentCircles) : tc.R = 8 := by sorry

end second_circle_radius_l1275_127542


namespace line_equation_l1275_127549

/-- Given a line with slope -2 and y-intercept 4, its equation is 2x+y-4=0 -/
theorem line_equation (x y : ℝ) : 
  (∃ (m b : ℝ), m = -2 ∧ b = 4 ∧ y = m * x + b) → 2 * x + y - 4 = 0 := by
  sorry

end line_equation_l1275_127549


namespace savings_theorem_l1275_127559

def savings_problem (monday_savings : ℕ) (tuesday_savings : ℕ) (wednesday_savings : ℕ) : ℕ :=
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  total_savings / 2

theorem savings_theorem (monday_savings tuesday_savings wednesday_savings : ℕ) :
  monday_savings = 15 →
  tuesday_savings = 28 →
  wednesday_savings = 13 →
  savings_problem monday_savings tuesday_savings wednesday_savings = 28 := by
  sorry

end savings_theorem_l1275_127559


namespace projection_problem_l1275_127510

/-- Given a projection that takes (3, 6) to (9/5, 18/5), prove that it takes (1, -1) to (-1/5, -2/5) -/
theorem projection_problem (proj : ℝ × ℝ → ℝ × ℝ) 
  (h : proj (3, 6) = (9/5, 18/5)) : 
  proj (1, -1) = (-1/5, -2/5) := by
  sorry

end projection_problem_l1275_127510


namespace sum_properties_l1275_127501

theorem sum_properties (x y : ℤ) (hx : ∃ m : ℤ, x = 5 * m) (hy : ∃ n : ℤ, y = 10 * n) :
  (∃ k : ℤ, x + y = 5 * k) ∧ x + y ≥ 15 := by
  sorry

end sum_properties_l1275_127501


namespace circle_equation_k_value_l1275_127515

theorem circle_equation_k_value (x y k : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 14*y - k = 0 ↔ (x + 4)^2 + (y + 7)^2 = 25) → 
  k = -40 :=
by sorry

end circle_equation_k_value_l1275_127515


namespace perimeter_of_square_C_l1275_127528

/-- Given squares A, B, and C with specified properties, prove that the perimeter of square C is 64 units -/
theorem perimeter_of_square_C (a b c : ℝ) : 
  (4 * a = 16) →  -- Perimeter of square A is 16 units
  (4 * b = 48) →  -- Perimeter of square B is 48 units
  (c = a + b) →   -- Side length of C is sum of side lengths of A and B
  (4 * c = 64) :=  -- Perimeter of square C is 64 units
by
  sorry


end perimeter_of_square_C_l1275_127528


namespace exists_parallel_planes_nonparallel_lines_perpendicular_line_implies_perpendicular_planes_l1275_127591

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Statement 1
theorem exists_parallel_planes_nonparallel_lines :
  ∃ (α β : Plane) (l m : Line),
    subset l α ∧ subset m β ∧ parallel_planes α β ∧ ¬parallel_lines l m :=
sorry

-- Statement 2
theorem perpendicular_line_implies_perpendicular_planes
  (α β : Plane) (l : Line) :
  subset l α → perpendicular_line_plane l β → perpendicular_planes α β :=
sorry

end exists_parallel_planes_nonparallel_lines_perpendicular_line_implies_perpendicular_planes_l1275_127591


namespace sequences_theorem_l1275_127530

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

def geometric_sequence (b : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = q * b n

def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem sequences_theorem (a b : ℕ → ℚ) :
  arithmetic_sequence a →
  geometric_sequence b →
  b 1 = 2 →
  b 2 + b 3 = 12 →
  b 3 = a 4 - 2 * a 1 →
  sum_arithmetic a 11 = 11 * b 4 →
  (∀ n : ℕ, n > 0 → a n = 3 * n - 2) ∧
  (∀ n : ℕ, n > 0 → b n = 2^n) ∧
  (∀ n : ℕ, n > 0 → 
    (Finset.range n).sum (λ i => a (2 * (i + 1)) * b (2 * i + 1)) = 
      (3 * n - 2) / 3 * 4^(n + 1) + 8 / 3) := by
  sorry

end sequences_theorem_l1275_127530


namespace count_integers_satisfying_inequality_l1275_127511

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), (∀ n ∈ S, (3 * (n - 1) * (n + 5) : ℤ) < 0) ∧ S.card = 5 := by
  sorry

end count_integers_satisfying_inequality_l1275_127511
