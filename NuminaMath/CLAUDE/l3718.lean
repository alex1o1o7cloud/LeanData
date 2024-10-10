import Mathlib

namespace f_is_odd_g_sum_one_l3718_371825

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the given conditions
axiom func_property : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_one_zero : f 1 = 0

-- State the theorems to be proved
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem g_sum_one : g 1 + g (-1) = 1 := by sorry

end f_is_odd_g_sum_one_l3718_371825


namespace jerry_age_l3718_371886

theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 18 → 
  mickey_age = 2 * jerry_age - 2 → 
  jerry_age = 10 := by
sorry

end jerry_age_l3718_371886


namespace greatest_b_for_quadratic_range_l3718_371889

theorem greatest_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 15 ≠ -6) ↔ b ≤ 9 :=
sorry

end greatest_b_for_quadratic_range_l3718_371889


namespace quadratic_roots_range_l3718_371879

theorem quadratic_roots_range (m l : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 - 2*x + l = 0 ∧ m * y^2 - 2*y + l = 0) → 
  (0 < m ∧ m < 1) :=
by sorry

end quadratic_roots_range_l3718_371879


namespace count_ordered_pairs_l3718_371836

theorem count_ordered_pairs : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 * 4 = 6 * p.2) (Finset.product (Finset.range 25) (Finset.range 25))).card ∧ n = 8 := by
  sorry

end count_ordered_pairs_l3718_371836


namespace class_size_l3718_371805

/-- Represents the number of students who borrowed at least 3 books -/
def R : ℕ := sorry

/-- Represents the total number of students in the class -/
def S : ℕ := sorry

/-- The average number of books per student -/
def average_books : ℕ := 2

theorem class_size :
  (0 * 2 + 1 * 12 + 2 * 4 + 3 * R = average_books * S) ∧
  (S = 2 + 12 + 4 + R) →
  S = 34 := by sorry

end class_size_l3718_371805


namespace homework_problem_count_l3718_371842

/-- Calculates the total number of homework problems given the number of pages and problems per page -/
def total_problems (math_pages reading_pages problems_per_page : ℕ) : ℕ :=
  (math_pages + reading_pages) * problems_per_page

/-- Proves that given 6 pages of math homework, 4 pages of reading homework, and 3 problems per page, the total number of problems is 30 -/
theorem homework_problem_count : total_problems 6 4 3 = 30 := by
  sorry

end homework_problem_count_l3718_371842


namespace acute_triangle_perimeter_bound_l3718_371899

/-- Given an acute-angled triangle with circumradius R and perimeter P, prove that P ≥ 4R. -/
theorem acute_triangle_perimeter_bound (R : ℝ) (P : ℝ) (α β γ : ℝ) :
  R > 0 →  -- R is positive (implied by being a radius)
  P > 0 →  -- P is positive (implied by being a perimeter)
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  0 < γ ∧ γ < π/2 →  -- γ is acute
  α + β + γ = π →  -- sum of angles in a triangle
  P = 2 * R * (Real.sin α + Real.sin β + Real.sin γ) →  -- perimeter formula using sine rule
  P ≥ 4 * R :=
by sorry

end acute_triangle_perimeter_bound_l3718_371899


namespace marble_remainder_l3718_371828

theorem marble_remainder (r p : ℕ) 
  (h_ringo : r % 6 = 4) 
  (h_paul : p % 6 = 3) : 
  (r + p) % 6 = 1 := by
sorry

end marble_remainder_l3718_371828


namespace catering_weight_calculation_l3718_371811

/-- Calculates the total weight of silverware and plates for a catering event --/
theorem catering_weight_calculation 
  (silverware_weight : ℕ) 
  (silverware_per_setting : ℕ) 
  (plate_weight : ℕ) 
  (plates_per_setting : ℕ) 
  (tables : ℕ) 
  (settings_per_table : ℕ) 
  (backup_settings : ℕ) : 
  silverware_weight = 4 →
  silverware_per_setting = 3 →
  plate_weight = 12 →
  plates_per_setting = 2 →
  tables = 15 →
  settings_per_table = 8 →
  backup_settings = 20 →
  (silverware_weight * silverware_per_setting + 
   plate_weight * plates_per_setting) * 
  (tables * settings_per_table + backup_settings) = 5040 := by
  sorry

end catering_weight_calculation_l3718_371811


namespace simplify_expression_l3718_371803

theorem simplify_expression : (4^7 + 2^6) * (1^5 - (-1)^5)^10 * (2^3 + 4^2) = 404225648 := by
  sorry

end simplify_expression_l3718_371803


namespace range_of_a_l3718_371835

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x > 0, x + 4 / x ≥ a) 
  (h2 : ∃ x : ℝ, x^2 + 2*x + a = 0) : 
  a ≤ 1 := by
  sorry

end range_of_a_l3718_371835


namespace min_floor_sum_l3718_371873

theorem min_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + b^2 + c^2 = a*b*c) : 
  ⌊(a^2 + b^2) / c⌋ + ⌊(b^2 + c^2) / a⌋ + ⌊(c^2 + a^2) / b⌋ ≥ 4 := by
  sorry

end min_floor_sum_l3718_371873


namespace compound_molecular_weight_l3718_371846

/-- Calculates the molecular weight of a compound given its atomic composition and atomic weights -/
def molecular_weight (num_C num_H num_O num_N num_S : ℕ) 
                     (weight_C weight_H weight_O weight_N weight_S : ℝ) : ℝ :=
  (num_C : ℝ) * weight_C + 
  (num_H : ℝ) * weight_H + 
  (num_O : ℝ) * weight_O + 
  (num_N : ℝ) * weight_N + 
  (num_S : ℝ) * weight_S

/-- The molecular weight of the given compound is approximately 134.184 g/mol -/
theorem compound_molecular_weight : 
  ∀ (ε : ℝ), ε > 0 → 
  |molecular_weight 4 8 2 1 1 12.01 1.008 16.00 14.01 32.07 - 134.184| < ε :=
sorry

end compound_molecular_weight_l3718_371846


namespace auction_theorem_l3718_371810

def auction_problem (starting_price : ℕ) (harry_first_bid : ℕ) (harry_final_bid : ℕ) : Prop :=
  let harry_bid := starting_price + harry_first_bid
  let second_bid := 2 * harry_bid
  let third_bid := second_bid + 3 * harry_first_bid
  harry_final_bid - third_bid = 2400

theorem auction_theorem : 
  auction_problem 300 200 4000 := by
  sorry

end auction_theorem_l3718_371810


namespace election_votes_proof_l3718_371855

theorem election_votes_proof (V : ℕ) (W L : ℕ) : 
  (W > L) →  -- Winner has more votes than loser
  (W - L = (V : ℚ) * (1 / 5)) →  -- Winner's margin is 20% of total votes
  ((L + 1000) - (W - 1000) = (V : ℚ) * (1 / 5)) →  -- Loser would win by 20% if 1000 votes change
  V = 5000 := by
  sorry

end election_votes_proof_l3718_371855


namespace complex_expression_equality_l3718_371862

theorem complex_expression_equality : 
  (Real.sqrt 3 - 1)^2 + (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 2 + Real.sqrt 3) + 
  (Real.sqrt 2 + 1) / (Real.sqrt 2 - 1) - 3 * Real.sqrt (1/2) = 
  8 - 2 * Real.sqrt 3 + Real.sqrt 2 / 2 := by
sorry

end complex_expression_equality_l3718_371862


namespace quadratic_real_roots_l3718_371845

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) := by
  sorry

end quadratic_real_roots_l3718_371845


namespace shelves_needed_l3718_371891

theorem shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) 
  (h1 : total_books = 46)
  (h2 : books_taken = 10)
  (h3 : books_per_shelf = 4) :
  (total_books - books_taken) / books_per_shelf = 9 :=
by sorry

end shelves_needed_l3718_371891


namespace cyclic_fraction_product_l3718_371897

theorem cyclic_fraction_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 :=
by sorry

end cyclic_fraction_product_l3718_371897


namespace bob_pays_48_percent_l3718_371861

-- Define the suggested retail price
variable (P : ℝ)

-- Define the marked price as 80% of the suggested retail price
def markedPrice (P : ℝ) : ℝ := 0.8 * P

-- Define Bob's purchase price as 60% of the marked price
def bobPrice (P : ℝ) : ℝ := 0.6 * markedPrice P

-- Theorem statement
theorem bob_pays_48_percent (P : ℝ) (h : P > 0) : 
  bobPrice P / P = 0.48 := by
sorry

end bob_pays_48_percent_l3718_371861


namespace rectangle_area_l3718_371858

/-- The area of a rectangle with length 1.2 meters and width 0.5 meters is 0.6 square meters. -/
theorem rectangle_area : 
  let length : ℝ := 1.2
  let width : ℝ := 0.5
  length * width = 0.6 := by sorry

end rectangle_area_l3718_371858


namespace book_cost_solution_l3718_371893

def book_cost_problem (p : ℝ) : Prop :=
  7 * p < 15 ∧ 11 * p > 22

theorem book_cost_solution :
  ∃ p : ℝ, book_cost_problem p ∧ p = 2.10 := by
sorry

end book_cost_solution_l3718_371893


namespace solution_set_correct_l3718_371859

/-- The system of equations --/
def system (x y z : ℝ) : Prop :=
  6 * (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) - 49 * x * y * z = 0 ∧
  6 * y * (x^2 - z^2) + 5 * x * z = 0 ∧
  2 * z * (x^2 - y^2) - 9 * x * y = 0

/-- The solution set --/
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (2, 1, 3), (2, -1, -3), (-2, 1, -3), (-2, -1, 3)}

/-- Theorem stating that the solution set is correct --/
theorem solution_set_correct :
  ∀ x y z, (x, y, z) ∈ solution_set ↔ system x y z :=
by sorry

end solution_set_correct_l3718_371859


namespace quadratic_monotonicity_l3718_371894

/-- A function f: ℝ → ℝ is monotonic on an interval [a, b] if it is either
    monotonically increasing or monotonically decreasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The quadratic function f(x) = 2x^2 - kx + 1 is monotonic on [1, 3]
    if and only if k ≤ 4 or k ≥ 12. -/
theorem quadratic_monotonicity (k : ℝ) :
  IsMonotonic (fun x => 2 * x^2 - k * x + 1) 1 3 ↔ k ≤ 4 ∨ k ≥ 12 :=
sorry

end quadratic_monotonicity_l3718_371894


namespace decagon_triangle_probability_l3718_371868

def regular_decagon : ℕ := 10

def total_triangles : ℕ := regular_decagon.choose 3

def favorable_outcomes : ℕ := regular_decagon * (regular_decagon - 4)

def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 1/2 := by
  sorry

end decagon_triangle_probability_l3718_371868


namespace train_route_length_l3718_371827

/-- Given two trains traveling towards each other on a route, where Train X takes 4 hours
    to complete the trip, Train Y takes 3 hours to complete the trip, and Train X has
    traveled 60 km when they meet, prove that the total length of the route is 140 km. -/
theorem train_route_length (x_time y_time x_distance : ℝ) 
    (hx : x_time = 4)
    (hy : y_time = 3)
    (hd : x_distance = 60) : 
  x_distance * (1 / x_time + 1 / y_time) = 140 := by
  sorry

#check train_route_length

end train_route_length_l3718_371827


namespace area_between_curves_l3718_371834

theorem area_between_curves : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := x^3
  ∫ x in (0)..(1), (f x - g x) = 1/12 := by
  sorry

end area_between_curves_l3718_371834


namespace sunny_candles_proof_l3718_371864

/-- Calculates the total number of candles used by Sunny --/
def total_candles (initial_cakes : ℕ) (given_away : ℕ) (candles_per_cake : ℕ) : ℕ :=
  (initial_cakes - given_away) * candles_per_cake

/-- Proves that Sunny will use 36 candles in total --/
theorem sunny_candles_proof :
  total_candles 8 2 6 = 36 := by
  sorry

end sunny_candles_proof_l3718_371864


namespace triangle_inequality_from_condition_l3718_371838

theorem triangle_inequality_from_condition 
  (a b c : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (h : 5 * (a^2 + b^2 + c^2) < 6 * (a*b + b*c + c*a)) : 
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) := by
  sorry

end triangle_inequality_from_condition_l3718_371838


namespace ab_value_l3718_371881

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 5) (h2 : a + b = 3) : a * b = 2 := by
  sorry

end ab_value_l3718_371881


namespace odometer_square_sum_l3718_371830

theorem odometer_square_sum : ∃ (a b c : ℕ),
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) ∧
  (100 ≤ 100 * b + 10 * c + a) ∧ (100 * b + 10 * c + a < 1000) ∧
  ((100 * b + 10 * c + a) - (100 * a + 10 * b + c)) % 60 = 0 ∧
  a^2 + b^2 + c^2 = 77 := by
sorry

end odometer_square_sum_l3718_371830


namespace third_shiny_penny_probability_l3718_371887

theorem third_shiny_penny_probability :
  let total_pennies : ℕ := 9
  let shiny_pennies : ℕ := 4
  let dull_pennies : ℕ := 5
  let probability_more_than_four_draws : ℚ :=
    (Nat.choose 4 2 * Nat.choose 5 1 + Nat.choose 4 1 * Nat.choose 5 2) / Nat.choose 9 4
  probability_more_than_four_draws = 5 / 9 := by
  sorry

end third_shiny_penny_probability_l3718_371887


namespace no_valid_tiling_l3718_371847

/-- Represents a chessboard with one corner removed -/
def ChessboardWithCornerRemoved := Fin 8 × Fin 8

/-- Represents a trimino (3x1 rectangle) -/
def Trimino := Fin 3 × Fin 1

/-- A tiling of the chessboard with triminos -/
def Tiling := ChessboardWithCornerRemoved → Option Trimino

/-- Predicate to check if a tiling is valid -/
def is_valid_tiling (t : Tiling) : Prop :=
  -- Each square is either covered by a trimino or is the removed corner
  ∀ (x : ChessboardWithCornerRemoved), 
    (x ≠ (7, 7) → t x ≠ none) ∧ 
    (x = (7, 7) → t x = none) ∧
  -- Each trimino covers exactly three squares
  ∀ (p : Trimino), ∃! (x y z : ChessboardWithCornerRemoved), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    t x = some p ∧ t y = some p ∧ t z = some p

/-- Theorem stating that no valid tiling exists -/
theorem no_valid_tiling : ¬∃ (t : Tiling), is_valid_tiling t := by
  sorry

end no_valid_tiling_l3718_371847


namespace range_of_f_range_of_m_l3718_371850

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 4|

-- Theorem for the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-2) 2 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ ≤ m - m^2) → m ∈ Set.Icc (-1) 2 := by sorry

end range_of_f_range_of_m_l3718_371850


namespace parallel_vectors_y_value_l3718_371800

/-- Two vectors in R² are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (6, y)
  are_parallel a b → y = 4 :=
by
  sorry

end parallel_vectors_y_value_l3718_371800


namespace homothetic_cubes_sum_l3718_371806

-- Define a cube in ℝ³
def Cube : Type := ℝ × ℝ × ℝ → Prop

-- Define a homothetic cube
def HomotheticCube (Q : Cube) (a : ℝ) : Cube := sorry

-- Define a sequence of homothetic cubes
def HomotheticCubeSequence (Q : Cube) : Type := ℕ → Cube

-- Define the property of completely filling a cube
def CompletelyFills (Q : Cube) (seq : HomotheticCubeSequence Q) : Prop := sorry

-- Define the coefficients of homothety for a sequence
def CoefficientsOfHomothety (Q : Cube) (seq : HomotheticCubeSequence Q) : ℕ → ℝ := sorry

-- The main theorem
theorem homothetic_cubes_sum (Q : Cube) (seq : HomotheticCubeSequence Q) :
  (∀ n, CoefficientsOfHomothety Q seq n < 1) →
  CompletelyFills Q seq →
  ∑' n, CoefficientsOfHomothety Q seq n ≥ 4 := by sorry

end homothetic_cubes_sum_l3718_371806


namespace lunch_cost_is_24_l3718_371839

/-- The cost of the Taco Grande Plate -/
def taco_grande_cost : ℕ := sorry

/-- The cost of Mike's additional items -/
def mike_additional_cost : ℕ := 2 + 4 + 2

/-- John's bill is equal to the cost of the Taco Grande Plate -/
def johns_bill : ℕ := taco_grande_cost

/-- Mike's bill is equal to the cost of the Taco Grande Plate plus the additional items -/
def mikes_bill : ℕ := taco_grande_cost + mike_additional_cost

/-- Mike's bill is twice as large as John's bill -/
axiom mikes_bill_twice_johns : mikes_bill = 2 * johns_bill

/-- The combined total cost of Mike and John's lunch -/
def total_cost : ℕ := johns_bill + mikes_bill

theorem lunch_cost_is_24 : total_cost = 24 := by sorry

end lunch_cost_is_24_l3718_371839


namespace all_equations_have_integer_roots_l3718_371880

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic equation has integer roots -/
def hasIntegerRoots (eq : QuadraticEquation) : Prop :=
  ∃ x y : ℤ, eq.a * x^2 + eq.b * x + eq.c = 0 ∧ eq.a * y^2 + eq.b * y + eq.c = 0 ∧ x ≠ y

/-- Generates the next equation by increasing coefficients by 1 -/
def nextEquation (eq : QuadraticEquation) : QuadraticEquation :=
  { a := eq.a, b := eq.b + 1, c := eq.c + 1 }

/-- The initial quadratic equation x^2 + 3x + 2 = 0 -/
def initialEquation : QuadraticEquation := { a := 1, b := 3, c := 2 }

theorem all_equations_have_integer_roots :
  hasIntegerRoots initialEquation ∧
  hasIntegerRoots (nextEquation initialEquation) ∧
  hasIntegerRoots (nextEquation (nextEquation initialEquation)) ∧
  hasIntegerRoots (nextEquation (nextEquation (nextEquation initialEquation))) ∧
  hasIntegerRoots (nextEquation (nextEquation (nextEquation (nextEquation initialEquation)))) :=
by sorry


end all_equations_have_integer_roots_l3718_371880


namespace sam_travel_time_l3718_371898

-- Define the points and distances
def point_A : ℝ := 0
def point_B : ℝ := 1000
def point_C : ℝ := 600

-- Define Sam's speed
def sam_speed : ℝ := 50

-- State the theorem
theorem sam_travel_time :
  let total_distance := point_B - point_A
  let time := total_distance / sam_speed
  (point_C - point_A = 600) ∧ 
  (point_B - point_C = 400) ∧ 
  (sam_speed = 50) →
  time = 20 := by sorry

end sam_travel_time_l3718_371898


namespace johns_weekly_earnings_l3718_371817

/-- Calculates John's total earnings per week from crab fishing --/
theorem johns_weekly_earnings :
  let monday_baskets : ℕ := 3
  let thursday_baskets : ℕ := 4
  let small_crabs_per_basket : ℕ := 4
  let large_crabs_per_basket : ℕ := 5
  let small_crab_price : ℕ := 3
  let large_crab_price : ℕ := 5

  let monday_crabs : ℕ := monday_baskets * small_crabs_per_basket
  let thursday_crabs : ℕ := thursday_baskets * large_crabs_per_basket

  let monday_earnings : ℕ := monday_crabs * small_crab_price
  let thursday_earnings : ℕ := thursday_crabs * large_crab_price

  let total_earnings : ℕ := monday_earnings + thursday_earnings

  total_earnings = 136 := by
  sorry

end johns_weekly_earnings_l3718_371817


namespace rosette_area_l3718_371813

/-- The area of a rosette formed by four semicircles on the sides of a square -/
theorem rosette_area (a : ℝ) (h : a > 0) :
  let square_side := a
  let semicircle_radius := a / 2
  let rosette_area := (a^2 * (Real.pi - 2)) / 2
  rosette_area = (square_side^2 * (Real.pi - 2)) / 2 :=
by sorry

end rosette_area_l3718_371813


namespace franks_change_is_four_l3718_371851

/-- Calculates the change Frank has after buying peanuts -/
def franks_change (one_dollar_bills five_dollar_bills ten_dollar_bills twenty_dollar_bills : ℕ)
  (peanut_cost_per_pound : ℕ) (daily_consumption : ℕ) (days : ℕ) : ℕ :=
  let total_money := one_dollar_bills + 5 * five_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills
  let total_peanuts := daily_consumption * days
  let peanut_cost := peanut_cost_per_pound * total_peanuts
  total_money - peanut_cost

theorem franks_change_is_four :
  franks_change 7 4 2 1 3 3 7 = 4 := by
  sorry

end franks_change_is_four_l3718_371851


namespace x_value_l3718_371866

theorem x_value : ∃ x : ℝ, 
  ((x * (9^2)) / ((8^2) * (3^5)) = 0.16666666666666666) ∧ 
  (x = 5.333333333333333) := by
  sorry

end x_value_l3718_371866


namespace car_speed_problem_l3718_371823

/-- Given a car traveling for two hours, where the speed in the second hour is 70 km/h
    and the average speed over two hours is 84 km/h, prove that the speed in the first hour
    must be 98 km/h. -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) 
  (h1 : speed_second_hour = 70)
  (h2 : average_speed = 84) :
  ∃ (speed_first_hour : ℝ),
    speed_first_hour = 98 ∧ 
    average_speed = (speed_first_hour + speed_second_hour) / 2 :=
by sorry

end car_speed_problem_l3718_371823


namespace gym_towels_theorem_l3718_371852

def gym_problem (first_hour_guests : ℕ) : Prop :=
  let second_hour_guests := first_hour_guests + (first_hour_guests * 20 / 100)
  let third_hour_guests := second_hour_guests + (second_hour_guests * 25 / 100)
  let fourth_hour_guests := third_hour_guests + (third_hour_guests * 33 / 100)
  let fifth_hour_guests := fourth_hour_guests - (fourth_hour_guests * 15 / 100)
  let sixth_hour_guests := fifth_hour_guests
  let seventh_hour_guests := sixth_hour_guests - (sixth_hour_guests * 30 / 100)
  let eighth_hour_guests := seventh_hour_guests - (seventh_hour_guests * 50 / 100)
  let total_guests := first_hour_guests + second_hour_guests + third_hour_guests + 
                      fourth_hour_guests + fifth_hour_guests + sixth_hour_guests + 
                      seventh_hour_guests + eighth_hour_guests
  let total_towels := total_guests * 2
  total_towels = 868

theorem gym_towels_theorem : 
  gym_problem 40 := by
  sorry

#check gym_towels_theorem

end gym_towels_theorem_l3718_371852


namespace age_sum_in_two_years_l3718_371814

theorem age_sum_in_two_years :
  let fem_current_age : ℕ := 11
  let matt_current_age : ℕ := 4 * fem_current_age
  let jake_current_age : ℕ := matt_current_age + 5
  let fem_future_age : ℕ := fem_current_age + 2
  let matt_future_age : ℕ := matt_current_age + 2
  let jake_future_age : ℕ := jake_current_age + 2
  fem_future_age + matt_future_age + jake_future_age = 110
  := by sorry

end age_sum_in_two_years_l3718_371814


namespace gerbils_sold_l3718_371802

theorem gerbils_sold (initial_gerbils : ℕ) (difference : ℕ) (h1 : initial_gerbils = 68) (h2 : difference = 54) :
  initial_gerbils - difference = 14 := by
  sorry

end gerbils_sold_l3718_371802


namespace arcsin_of_negative_one_l3718_371877

theorem arcsin_of_negative_one :
  Real.arcsin (-1) = -π / 2 := by
  sorry

end arcsin_of_negative_one_l3718_371877


namespace investment_income_is_575_l3718_371865

/-- Calculates the simple interest for a given principal, rate, and time. -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Represents the total annual income from two investments with simple interest. -/
def totalAnnualIncome (investment1 : ℝ) (rate1 : ℝ) (investment2 : ℝ) (rate2 : ℝ) : ℝ :=
  simpleInterest investment1 rate1 1 + simpleInterest investment2 rate2 1

/-- Theorem stating that the total annual income from the given investments is $575. -/
theorem investment_income_is_575 :
  totalAnnualIncome 3000 0.085 5000 0.064 = 575 := by
  sorry

end investment_income_is_575_l3718_371865


namespace add_ten_to_number_l3718_371826

theorem add_ten_to_number (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 := by
  sorry

end add_ten_to_number_l3718_371826


namespace committee_size_l3718_371840

/-- Given a committee of n members where any 3 individuals can be sent for a social survey,
    if the probability of female student B being chosen given that male student A is chosen is 0.4,
    then n = 6. -/
theorem committee_size (n : ℕ) : 
  (n ≥ 3) →  -- Ensure committee size is at least 3
  (((n - 2 : ℚ) / ((n - 1) * (n - 2) / 2)) = 0.4) → 
  n = 6 := by
  sorry

end committee_size_l3718_371840


namespace f_range_l3718_371884

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * Real.log (1 + x) + x^2
  else -x * Real.log (1 - x) + x^2

theorem f_range (a : ℝ) : f (-a) + f a ≤ 2 * f 1 → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end f_range_l3718_371884


namespace sixth_term_value_l3718_371849

/-- A sequence of positive integers where each term after the first is 1/4 of the sum of the term that precedes it and the term that follows it. -/
def SpecialSequence (a : ℕ → ℕ+) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n + 1) : ℚ) = (1 / 4) * ((a n : ℚ) + (a (n + 2) : ℚ))

theorem sixth_term_value (a : ℕ → ℕ+) (h : SpecialSequence a) (h1 : a 1 = 3) (h5 : a 5 = 43) :
  a 6 = 129 := by
  sorry

end sixth_term_value_l3718_371849


namespace equation_solution_l3718_371829

theorem equation_solution : 
  let x : ℚ := 30
  40 * x + (12 + 8) * 3 / 5 = 1212 := by
  sorry

end equation_solution_l3718_371829


namespace sphere_cube_paint_equivalence_l3718_371820

theorem sphere_cube_paint_equivalence (M : ℝ) : 
  let cube_side : ℝ := 3
  let cube_surface_area : ℝ := 6 * cube_side^2
  let sphere_surface_area : ℝ := cube_surface_area
  let sphere_volume : ℝ := (M * Real.sqrt 3) / Real.sqrt Real.pi
  (∃ (r : ℝ), 
    sphere_surface_area = 4 * Real.pi * r^2 ∧ 
    sphere_volume = (4 / 3) * Real.pi * r^3) →
  M = 36 := by
sorry

end sphere_cube_paint_equivalence_l3718_371820


namespace sequence_constant_l3718_371824

/-- A sequence of positive real numbers satisfying a specific inequality is constant. -/
theorem sequence_constant (a : ℤ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_ineq : ∀ n, a n ≥ (a (n + 2) + a (n + 1) + a (n - 1) + a (n - 2)) / 4) :
  ∀ m n, a m = a n :=
by sorry

end sequence_constant_l3718_371824


namespace gold_coin_distribution_l3718_371819

theorem gold_coin_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 13 * k + 3) → 
  n < 150 → 
  (∀ m : ℕ, (∃ j : ℕ, m = 13 * j + 3) → m < 150 → m ≤ n) → 
  n = 146 := by
sorry

end gold_coin_distribution_l3718_371819


namespace power_equality_l3718_371857

theorem power_equality (K : ℕ) : 32^2 * 4^4 = 2^K → K = 18 := by
  have h1 : 32 = 2^5 := by sorry
  have h2 : 4 = 2^2 := by sorry
  sorry

end power_equality_l3718_371857


namespace time_to_empty_tank_l3718_371804

/-- Time to empty a tank given its volume and pipe rates -/
theorem time_to_empty_tank 
  (tank_volume : ℝ) 
  (inlet_rate : ℝ) 
  (outlet_rate1 : ℝ) 
  (outlet_rate2 : ℝ) 
  (h1 : tank_volume = 30) 
  (h2 : inlet_rate = 3) 
  (h3 : outlet_rate1 = 12) 
  (h4 : outlet_rate2 = 6) : 
  (tank_volume * 1728) / (outlet_rate1 + outlet_rate2 - inlet_rate) = 3456 := by
  sorry


end time_to_empty_tank_l3718_371804


namespace check_error_proof_l3718_371821

theorem check_error_proof (x y : ℕ) : 
  x ≥ 10 ∧ x < 100 ∧ y ≥ 10 ∧ y < 100 →  -- x and y are two-digit numbers
  y = 3 * x - 6 →                        -- y = 3x - 6
  100 * y + x - (100 * x + y) = 2112 →   -- difference is $21.12 (2112 cents)
  x = 14 ∧ y = 36 :=                     -- conclusion: x = 14 and y = 36
by sorry

end check_error_proof_l3718_371821


namespace rachel_chocolate_sales_l3718_371856

/-- The amount of money Rachel made by selling chocolate bars -/
def rachel_money_made (total_bars : ℕ) (price_per_bar : ℚ) (unsold_bars : ℕ) : ℚ :=
  (total_bars - unsold_bars : ℚ) * price_per_bar

/-- Theorem stating that Rachel made $58.50 from selling chocolate bars -/
theorem rachel_chocolate_sales : rachel_money_made 25 3.25 7 = 58.50 := by
  sorry

end rachel_chocolate_sales_l3718_371856


namespace inverse_sum_equals_golden_ratio_minus_one_l3718_371878

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2 - x else x^3 - 2*x^2 + x

-- Define the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℝ :=
  if y ≤ 0 then (1 + Real.sqrt 5) / 2
  else if y = 1 then 1
  else -2

-- Theorem statement
theorem inverse_sum_equals_golden_ratio_minus_one :
  f_inv (-1) + f_inv 1 + f_inv 4 = (Real.sqrt 5 - 1) / 2 := by sorry

end inverse_sum_equals_golden_ratio_minus_one_l3718_371878


namespace vertex_angle_is_160_degrees_l3718_371816

/-- An isosceles triangle with specific properties -/
structure SpecialIsoscelesTriangle where
  -- The length of each equal side
  a : ℝ
  -- The base of the triangle
  b : ℝ
  -- The height of the triangle
  h : ℝ
  -- The vertex angle in radians
  θ : ℝ
  -- The triangle is isosceles
  isIsosceles : b = 2 * a * Real.cos θ
  -- The square of the length of each equal side is three times the product of the base and the height
  sideSquareProperty : a^2 = 3 * b * h
  -- The triangle is obtuse
  isObtuse : θ > Real.pi / 2

/-- The theorem stating that the vertex angle of the special isosceles triangle is 160 degrees -/
theorem vertex_angle_is_160_degrees (t : SpecialIsoscelesTriangle) : 
  t.θ = 160 * Real.pi / 180 := by
  sorry

end vertex_angle_is_160_degrees_l3718_371816


namespace geometric_sequence_sum_ratio_l3718_371801

/-- Given a geometric sequence with common ratio not equal to -1,
    prove that if S_12 = 7S_4, then S_8 / S_4 = 3 -/
theorem geometric_sequence_sum_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (hq : q ≠ -1) 
  (h_geom : ∀ n, a (n + 1) = q * a n) 
  (S : ℕ → ℝ) 
  (h_sum : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) 
  (h_ratio : S 12 = 7 * S 4) : 
  S 8 / S 4 = 3 := by
sorry

end geometric_sequence_sum_ratio_l3718_371801


namespace least_positive_linear_combination_l3718_371808

theorem least_positive_linear_combination (x y z : ℤ) : 
  ∃ (a b c : ℤ), 24*a + 20*b + 12*c = 4 ∧ 
  (∀ (x y z : ℤ), 24*x + 20*y + 12*z = 0 ∨ |24*x + 20*y + 12*z| ≥ 4) :=
by sorry

end least_positive_linear_combination_l3718_371808


namespace base5_sum_equality_l3718_371837

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a base-5 representation to a natural number --/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base-5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem base5_sum_equality :
  addBase5 (toBase5 122) (toBase5 78) = toBase5 200 :=
sorry

end base5_sum_equality_l3718_371837


namespace complex_equation_solution_l3718_371818

theorem complex_equation_solution (m n : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : m / (1 + i) = 1 - n * i) : 
  m = 2 ∧ n = 1 := by
  sorry

end complex_equation_solution_l3718_371818


namespace prism_with_21_edges_has_9_faces_l3718_371809

/-- The number of faces in a prism with a given number of edges -/
def prism_faces (edges : ℕ) : ℕ :=
  2 + edges / 3

/-- Theorem: A prism with 21 edges has 9 faces -/
theorem prism_with_21_edges_has_9_faces :
  prism_faces 21 = 9 := by
  sorry

end prism_with_21_edges_has_9_faces_l3718_371809


namespace subtracted_amount_l3718_371890

theorem subtracted_amount (n : ℚ) (x : ℚ) : 
  n = 25 / 3 → 3 * n + 15 = 6 * n - x → x = 10 := by
  sorry

end subtracted_amount_l3718_371890


namespace parabola_vertex_after_transformation_l3718_371843

/-- The vertex of a parabola after transformation -/
theorem parabola_vertex_after_transformation :
  let f (x : ℝ) := (x - 2)^2 - 2*(x - 2) + 6
  ∃! (h : ℝ × ℝ), (h.1 = 3 ∧ h.2 = 5 ∧ ∀ x, f x ≥ f h.1) :=
by sorry

end parabola_vertex_after_transformation_l3718_371843


namespace quadratic_expression_value_l3718_371833

theorem quadratic_expression_value : 
  let x : ℤ := -2
  (x^2 + 6*x - 7) = -15 := by sorry

end quadratic_expression_value_l3718_371833


namespace fraction_calculation_l3718_371885

theorem fraction_calculation : (5 / 6 : ℚ) * (1 / ((7 / 8 : ℚ) - (3 / 4 : ℚ))) = 20 / 3 := by
  sorry

end fraction_calculation_l3718_371885


namespace total_drying_time_in_hours_l3718_371848

/-- Time to dry a short-haired dog in minutes -/
def short_hair_time : ℕ := 10

/-- Time to dry a full-haired dog in minutes -/
def full_hair_time : ℕ := 2 * short_hair_time

/-- Time to dry a medium-haired dog in minutes -/
def medium_hair_time : ℕ := 15

/-- Number of short-haired dogs -/
def short_hair_count : ℕ := 12

/-- Number of full-haired dogs -/
def full_hair_count : ℕ := 15

/-- Number of medium-haired dogs -/
def medium_hair_count : ℕ := 8

/-- Total time to dry all dogs in minutes -/
def total_time : ℕ := 
  short_hair_time * short_hair_count + 
  full_hair_time * full_hair_count + 
  medium_hair_time * medium_hair_count

theorem total_drying_time_in_hours : 
  total_time / 60 = 9 := by sorry

end total_drying_time_in_hours_l3718_371848


namespace all_points_collinear_l3718_371876

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  ∃ l : Line, p.onLine l ∧ q.onLine l ∧ r.onLine l

/-- Main theorem -/
theorem all_points_collinear (M : Set Point) (h_finite : Set.Finite M)
  (h_line : ∀ p q r : Point, p ∈ M → q ∈ M → r ∈ M → p ≠ q → 
    (∃ l : Line, p.onLine l ∧ q.onLine l) → (∃ s : Point, s ∈ M ∧ s ≠ p ∧ s ≠ q ∧ s.onLine l)) :
  ∀ p q r : Point, p ∈ M → q ∈ M → r ∈ M → collinear p q r :=
sorry

end all_points_collinear_l3718_371876


namespace students_playing_both_sports_l3718_371888

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) 
  (h1 : total = 250)
  (h2 : football = 160)
  (h3 : cricket = 90)
  (h4 : neither = 50) :
  football + cricket - (total - neither) = 50 := by
  sorry

end students_playing_both_sports_l3718_371888


namespace ellipse_symmetric_points_range_l3718_371882

/-- Given an ellipse and a line, this theorem states the range of the y-intercept of the line for which
    there exist two distinct points on the ellipse symmetric about the line. -/
theorem ellipse_symmetric_points_range (x y : ℝ) (m : ℝ) : 
  (x^2 / 4 + y^2 / 3 = 1) →  -- Ellipse equation
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / 4 + y₁^2 / 3 = 1) ∧  -- Point 1 on ellipse
    (x₂^2 / 4 + y₂^2 / 3 = 1) ∧  -- Point 2 on ellipse
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧        -- Points are distinct
    (y₁ + y₂) / 2 = 4 * ((x₁ + x₂) / 2) + m)  -- Points symmetric about y = 4x + m
  ↔ 
  (-2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13) :=
by sorry

end ellipse_symmetric_points_range_l3718_371882


namespace fair_tickets_sold_l3718_371883

theorem fair_tickets_sold (total : ℕ) (second_week : ℕ) (left_to_sell : ℕ) 
  (h1 : total = 90)
  (h2 : second_week = 17)
  (h3 : left_to_sell = 35) :
  total - second_week - left_to_sell = 38 := by
sorry

end fair_tickets_sold_l3718_371883


namespace equation_solutions_l3718_371863

def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6 ∧ (3*x + 6) / ((x - 1) * (x + 6)) = (3 - x) / (x - 2)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 3 ∨ x = -4) :=
by sorry

end equation_solutions_l3718_371863


namespace expression_value_l3718_371832

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 1)    -- absolute value of m is 1
  : m + (2024 * (a + b)) / 2023 - (c * d)^2 = 0 ∨ 
    m + (2024 * (a + b)) / 2023 - (c * d)^2 = -2 :=
by sorry

end expression_value_l3718_371832


namespace homework_problem_l3718_371869

theorem homework_problem (a b c d : ℤ) : 
  (a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0) →
  (-a - b = -a * b) →
  (c * d = -182 * (1 / (-c - d))) →
  ((a = -2 ∧ b = -2) ∧ ((c = -1 ∧ d = -13) ∨ (c = -13 ∧ d = -1))) :=
by sorry

end homework_problem_l3718_371869


namespace cube_of_negative_l3718_371860

theorem cube_of_negative (a : ℝ) : (-a)^3 = -a^3 := by
  sorry

end cube_of_negative_l3718_371860


namespace prove_x_value_l3718_371895

theorem prove_x_value (x a b c d : ℕ) 
  (h1 : x = a + 7)
  (h2 : a = b + 12)
  (h3 : b = c + 15)
  (h4 : c = d + 25)
  (h5 : d = 95) : x = 154 := by
  sorry

end prove_x_value_l3718_371895


namespace workshop_nobel_laureates_l3718_371815

theorem workshop_nobel_laureates
  (total_scientists : ℕ)
  (wolf_laureates : ℕ)
  (wolf_and_nobel : ℕ)
  (h_total : total_scientists = 50)
  (h_wolf : wolf_laureates = 31)
  (h_both : wolf_and_nobel = 16)
  (h_diff : ∃ (non_nobel : ℕ), 
    wolf_laureates + non_nobel + (non_nobel + 3) = total_scientists) :
  ∃ (nobel_laureates : ℕ), 
    nobel_laureates = 27 ∧ 
    nobel_laureates ≤ total_scientists ∧
    wolf_and_nobel ≤ nobel_laureates ∧
    wolf_and_nobel ≤ wolf_laureates :=
by
  sorry


end workshop_nobel_laureates_l3718_371815


namespace value_of_a_l3718_371822

theorem value_of_a (S T : Set ℕ) (a : ℕ) : 
  S = {1, 2} → T = {a} → S ∪ T = S → a = 1 ∨ a = 2 := by
  sorry

end value_of_a_l3718_371822


namespace min_wednesday_birthdays_is_eight_l3718_371875

/-- The minimum number of employees with birthdays on Wednesday -/
def min_wednesday_birthdays (total_employees : ℕ) (days_in_week : ℕ) : ℕ :=
  let other_days := days_in_week - 1
  let max_other_day_birthdays := (total_employees - 1) / days_in_week
  max_other_day_birthdays + 1

/-- Prove that given 50 employees, excluding those born in March, and with Wednesday having more 
    birthdays than any other day of the week (which all have an equal number of birthdays), 
    the minimum number of employees having birthdays on Wednesday is 8. -/
theorem min_wednesday_birthdays_is_eight :
  min_wednesday_birthdays 50 7 = 8 := by
  sorry

#eval min_wednesday_birthdays 50 7

end min_wednesday_birthdays_is_eight_l3718_371875


namespace gardener_flower_expenses_l3718_371812

/-- The total expenses for flowers ordered by a gardener -/
theorem gardener_flower_expenses :
  let tulips : ℕ := 250
  let carnations : ℕ := 375
  let roses : ℕ := 320
  let price_per_flower : ℕ := 2
  let total_flowers : ℕ := tulips + carnations + roses
  let total_expenses : ℕ := total_flowers * price_per_flower
  total_expenses = 1890 := by sorry

end gardener_flower_expenses_l3718_371812


namespace length_of_AE_l3718_371874

-- Define the square and points
def Square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (4, 0) ∧ C = (4, 4) ∧ D = (0, 4)

def PointOnSide (E : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ E = (x, 0)

def ReflectionOverDiagonal (E F : ℝ × ℝ) : Prop :=
  F.1 + F.2 = 4 ∧ F.1 = 4 - E.1

def DistanceCondition (D E F : ℝ × ℝ) : Prop :=
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 4 * (E.1 - D.1)^2

-- Main theorem
theorem length_of_AE (A B C D E F : ℝ × ℝ) :
  Square A B C D →
  PointOnSide E →
  ReflectionOverDiagonal E F →
  DistanceCondition D E F →
  E.1 = 8/3 :=
sorry

end length_of_AE_l3718_371874


namespace xoxox_probability_l3718_371871

def total_tiles : ℕ := 5
def x_tiles : ℕ := 3
def o_tiles : ℕ := 2

theorem xoxox_probability :
  (x_tiles : ℚ) / total_tiles *
  (o_tiles : ℚ) / (total_tiles - 1) *
  ((x_tiles - 1) : ℚ) / (total_tiles - 2) *
  ((o_tiles - 1) : ℚ) / (total_tiles - 3) *
  ((x_tiles - 2) : ℚ) / (total_tiles - 4) = 1 / 10 :=
sorry

end xoxox_probability_l3718_371871


namespace kates_remaining_money_is_7_80_l3718_371853

/-- Calculates the amount of money Kate has left after her savings and expenses --/
def kates_remaining_money (march_savings april_savings may_savings june_savings : ℚ)
  (keyboard_cost mouse_cost headset_cost video_game_cost : ℚ)
  (book_cost : ℚ)
  (euro_to_dollar pound_to_dollar : ℚ) : ℚ :=
  let total_savings := march_savings + april_savings + may_savings + june_savings + 2 * april_savings
  let euro_expenses := (keyboard_cost + mouse_cost + headset_cost + video_game_cost) * euro_to_dollar
  let pound_expenses := book_cost * pound_to_dollar
  total_savings - euro_expenses - pound_expenses

/-- Theorem stating that Kate has $7.80 left after her savings and expenses --/
theorem kates_remaining_money_is_7_80 :
  kates_remaining_money 27 13 28 35 42 4 16 25 12 1.2 1.4 = 7.8 := by
  sorry

end kates_remaining_money_is_7_80_l3718_371853


namespace fly_distance_bounded_l3718_371892

/-- Represents a right triangle room -/
structure RightTriangleRoom where
  hypotenuse : ℝ
  hypotenuse_positive : hypotenuse > 0

/-- Represents a fly's path in the room -/
structure FlyPath where
  room : RightTriangleRoom
  num_turns : ℕ
  start_acute_angle : Bool

/-- The maximum distance a fly can travel in the room -/
noncomputable def max_fly_distance (path : FlyPath) : ℝ :=
  sorry

/-- Theorem stating that a fly cannot travel more than 10 meters in the given conditions -/
theorem fly_distance_bounded (path : FlyPath) 
  (h1 : path.room.hypotenuse = 5)
  (h2 : path.num_turns = 10)
  (h3 : path.start_acute_angle = true) : 
  max_fly_distance path ≤ 10 :=
sorry

end fly_distance_bounded_l3718_371892


namespace unique_two_digit_number_l3718_371807

/-- A function that returns the tens digit of a two-digit number -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- A function that returns the units digit of a two-digit number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A predicate that checks if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- A predicate that checks if the product of digits of a number is 8 -/
def productOfDigitsIs8 (n : ℕ) : Prop := tensDigit n * unitsDigit n = 8

/-- A predicate that checks if adding 18 to a number reverses its digits -/
def adding18ReversesDigits (n : ℕ) : Prop := 
  tensDigit (n + 18) = unitsDigit n ∧ unitsDigit (n + 18) = tensDigit n

theorem unique_two_digit_number : 
  ∃! n : ℕ, isTwoDigit n ∧ productOfDigitsIs8 n ∧ adding18ReversesDigits n ∧ n = 24 := by
  sorry

end unique_two_digit_number_l3718_371807


namespace quadratic_coefficient_l3718_371841

theorem quadratic_coefficient (c : ℝ) : 
  ((-9 : ℝ)^2 + c * (-9) - 36 = 0) → c = 5 := by
  sorry

end quadratic_coefficient_l3718_371841


namespace policeman_can_reach_gangster_side_l3718_371872

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length s -/
structure Square (s : ℝ) where
  center : Point
  vertex : Point

/-- Represents the maximum speeds of the policeman and gangster -/
structure Speeds where
  policeman : ℝ
  gangster : ℝ

/-- Theorem stating that the policeman can always reach the same side as the gangster -/
theorem policeman_can_reach_gangster_side (s : ℝ) (square : Square s) (speeds : Speeds) :
  s > 0 ∧
  square.center = Point.mk (s/2) (s/2) ∧
  (square.vertex = Point.mk 0 0 ∨ square.vertex = Point.mk s 0 ∨
   square.vertex = Point.mk 0 s ∨ square.vertex = Point.mk s s) ∧
  speeds.gangster = 2.9 * speeds.policeman →
  ∃ (t : ℝ), t > 0 ∧ 
    ∃ (p : Point), (p.x = 0 ∨ p.x = s ∨ p.y = 0 ∨ p.y = s) ∧
      (p.x - square.center.x)^2 + (p.y - square.center.y)^2 ≤ (speeds.policeman * t)^2 ∧
      ((p.x - square.vertex.x)^2 + (p.y - square.vertex.y)^2 ≤ (speeds.gangster * t)^2 ∨
       (p.x - square.vertex.x)^2 + (p.y - square.vertex.y)^2 = (s * speeds.gangster * t)^2) :=
by sorry

end policeman_can_reach_gangster_side_l3718_371872


namespace power_negative_multiply_l3718_371896

theorem power_negative_multiply (m : ℝ) : (-m)^2 * m^5 = m^7 := by
  sorry

end power_negative_multiply_l3718_371896


namespace alices_june_burger_spending_l3718_371831

/-- Calculate Alice's spending on burgers in June --/
def alices_burger_spending (
  days_in_june : Nat)
  (burgers_per_day : Nat)
  (burger_price : ℚ)
  (discount_days : Nat)
  (discount_percentage : ℚ)
  (free_burger_days : Nat)
  (coupon_count : Nat)
  (coupon_discount : ℚ) : ℚ :=
  let total_burgers := days_in_june * burgers_per_day
  let regular_cost := total_burgers * burger_price
  let discount_burgers := discount_days * burgers_per_day
  let discount_amount := discount_burgers * burger_price * discount_percentage
  let free_burgers := free_burger_days
  let free_burger_value := free_burgers * burger_price
  let coupon_savings := coupon_count * burger_price * coupon_discount
  regular_cost - discount_amount - free_burger_value - coupon_savings

/-- Theorem stating Alice's spending on burgers in June --/
theorem alices_june_burger_spending :
  alices_burger_spending 30 4 13 8 (1/10) 4 6 (1/2) = 1146.6 := by
  sorry

end alices_june_burger_spending_l3718_371831


namespace ellipse_m_range_l3718_371854

theorem ellipse_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
   (2 + m > 0 ∧ -(m + 1) > 0) ∧ 
   (2 + m ≠ -(m + 1))) ↔ 
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1) :=
sorry

end ellipse_m_range_l3718_371854


namespace prob_red_ball_specific_bag_l3718_371867

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  green : ℕ

/-- The probability of drawing a red ball from a bag of colored balls -/
def prob_red_ball (bag : ColoredBalls) : ℚ :=
  bag.red / bag.total

/-- Theorem stating the probability of drawing a red ball from a specific bag -/
theorem prob_red_ball_specific_bag :
  let bag : ColoredBalls := { total := 9, red := 6, green := 3 }
  prob_red_ball bag = 2 / 3 := by
sorry

end prob_red_ball_specific_bag_l3718_371867


namespace equation_coefficients_l3718_371844

/-- Given a quadratic equation of the form ax^2 + bx + c = 0,
    this function returns a triple (a, b, c) of the coefficients -/
def quadratic_coefficients (f : ℝ → ℝ) : ℝ × ℝ × ℝ := sorry

theorem equation_coefficients :
  let f : ℝ → ℝ := λ x => -x^2 + 3*x - 1
  quadratic_coefficients f = (-1, 3, -1) := by sorry

end equation_coefficients_l3718_371844


namespace problem_solution_l3718_371870

theorem problem_solution (a b c n : ℝ) (h : n = (2 * a * b * c) / (c - a)) :
  c = (n * a) / (n - 2 * a * b) := by
  sorry

end problem_solution_l3718_371870
