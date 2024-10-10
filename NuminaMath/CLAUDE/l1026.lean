import Mathlib

namespace iris_rose_ratio_l1026_102670

theorem iris_rose_ratio (initial_roses : ℕ) (added_roses : ℕ) : 
  initial_roses = 42 →
  added_roses = 35 →
  (3 : ℚ) / 7 = (irises_needed : ℚ) / (initial_roses + added_roses) →
  irises_needed = 33 :=
by
  sorry

end iris_rose_ratio_l1026_102670


namespace ms_elizabeth_has_five_investments_l1026_102671

-- Define the variables
def mr_banks_investments : ℕ := 8
def mr_banks_revenue_per_investment : ℕ := 500
def ms_elizabeth_revenue_per_investment : ℕ := 900
def revenue_difference : ℕ := 500

-- Define Ms. Elizabeth's number of investments as a function
def ms_elizabeth_investments : ℕ :=
  let mr_banks_total_revenue := mr_banks_investments * mr_banks_revenue_per_investment
  let ms_elizabeth_total_revenue := mr_banks_total_revenue + revenue_difference
  ms_elizabeth_total_revenue / ms_elizabeth_revenue_per_investment

-- Theorem statement
theorem ms_elizabeth_has_five_investments :
  ms_elizabeth_investments = 5 := by
  sorry

end ms_elizabeth_has_five_investments_l1026_102671


namespace polynomial_factor_implies_c_value_l1026_102607

theorem polynomial_factor_implies_c_value : ∀ (c q : ℝ),
  (∃ (a : ℝ), (X^3 + q*X + 1) * (3*X + a) = 3*X^4 + c*X^2 + 8*X + 9) →
  c = 24 := by
sorry

end polynomial_factor_implies_c_value_l1026_102607


namespace impossible_grouping_l1026_102639

theorem impossible_grouping : ¬ ∃ (partition : List (List Nat)),
  (∀ group ∈ partition, (∀ n ∈ group, 1 ≤ n ∧ n ≤ 77)) ∧
  (∀ group ∈ partition, group.length ≥ 3) ∧
  (∀ group ∈ partition, ∃ n ∈ group, n = (group.sum - n)) ∧
  (partition.join.toFinset = Finset.range 77) :=
by sorry

end impossible_grouping_l1026_102639


namespace infinitely_many_perfect_squares_l1026_102689

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A number is a perfect square if it's the square of some natural number. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- The main theorem stating that an arithmetic progression of positive integers
    with at least one perfect square contains infinitely many perfect squares. -/
theorem infinitely_many_perfect_squares
  (a : ℕ → ℕ)
  (h_arith : ArithmeticProgression a)
  (h_positive : ∀ n, a n > 0)
  (h_one_square : ∃ n, IsPerfectSquare (a n)) :
  ∀ k : ℕ, ∃ n > k, IsPerfectSquare (a n) :=
sorry

end infinitely_many_perfect_squares_l1026_102689


namespace expression_simplification_and_evaluation_expression_evaluation_at_3_l1026_102674

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((2 * x - 1) / (x - 2) - 1) / ((x + 1) / (x^2 - 4)) = x + 2 :=
by sorry

theorem expression_evaluation_at_3 :
  let x : ℝ := 3
  ((2 * x - 1) / (x - 2) - 1) / ((x + 1) / (x^2 - 4)) = 5 :=
by sorry

end expression_simplification_and_evaluation_expression_evaluation_at_3_l1026_102674


namespace shape_to_square_possible_l1026_102625

/-- Represents a shape on a graph paper --/
structure Shape :=
  (area : ℝ)

/-- Represents a triangle --/
structure Triangle :=
  (area : ℝ)

/-- Represents a square --/
structure Square :=
  (side_length : ℝ)

/-- Function to divide a shape into triangles --/
def divide_into_triangles (s : Shape) : List Triangle := sorry

/-- Function to assemble triangles into a square --/
def assemble_square (triangles : List Triangle) : Option Square := sorry

/-- Theorem stating that the shape can be divided into 5 triangles and assembled into a square --/
theorem shape_to_square_possible (s : Shape) : 
  ∃ (triangles : List Triangle) (sq : Square), 
    divide_into_triangles s = triangles ∧ 
    triangles.length = 5 ∧ 
    assemble_square triangles = some sq :=
by sorry

end shape_to_square_possible_l1026_102625


namespace infinite_primes_dividing_2017_power_plus_2017_l1026_102637

theorem infinite_primes_dividing_2017_power_plus_2017 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ 2017^(2^n) + 2017} := by
  sorry

end infinite_primes_dividing_2017_power_plus_2017_l1026_102637


namespace function_minimum_implies_inequality_l1026_102666

/-- Given a function f(x) = ax^2 + bx - ln(x) where a > 0 and b ∈ ℝ,
    if f(x) ≥ f(1) for all x > 0, then ln(a) < -2b -/
theorem function_minimum_implies_inequality 
  (a b : ℝ) 
  (ha : a > 0)
  (hf : ∀ x > 0, a * x^2 + b * x - Real.log x ≥ a + b) :
  Real.log a < -2 * b := by
  sorry

end function_minimum_implies_inequality_l1026_102666


namespace container_count_l1026_102690

theorem container_count (container_capacity : ℝ) (total_capacity : ℝ) : 
  (8 : ℝ) = 0.2 * container_capacity →
  total_capacity = 1600 →
  (total_capacity / container_capacity : ℝ) = 40 := by
  sorry

end container_count_l1026_102690


namespace complex_arithmetic_expression_l1026_102686

theorem complex_arithmetic_expression : 
  ∃ ε > 0, ε < 0.0001 ∧ 
  |(3.5 / 0.7) * (5/3 : ℝ) + (7.2 / 0.36) - ((5/3 : ℝ) * 0.75 / 0.25) - 23.3335| < ε := by
  sorry

end complex_arithmetic_expression_l1026_102686


namespace infiniteLoopDecimal_eq_fraction_l1026_102658

/-- Represents the infinite loop decimal 0.0 ̇1 ̇7 -/
def infiniteLoopDecimal : ℚ := sorry

/-- The infinite loop decimal 0.0 ̇1 ̇7 is equal to 17/990 -/
theorem infiniteLoopDecimal_eq_fraction : infiniteLoopDecimal = 17 / 990 := by sorry

end infiniteLoopDecimal_eq_fraction_l1026_102658


namespace complement_M_in_U_l1026_102606

-- Define the universal set U
def U : Set ℝ := {x : ℝ | x > 0}

-- Define the set M
def M : Set ℝ := {x : ℝ | x > 1}

-- Define the complement of M in U
def complementMU : Set ℝ := {x : ℝ | x ∈ U ∧ x ∉ M}

-- Theorem statement
theorem complement_M_in_U :
  complementMU = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by sorry

end complement_M_in_U_l1026_102606


namespace hexagon_walk_l1026_102633

/-- A regular hexagon with side length 3 km -/
structure RegularHexagon where
  sideLength : ℝ
  is_regular : sideLength = 3

/-- A point on the perimeter of the hexagon, represented by the distance traveled from a corner -/
def PerimeterPoint (h : RegularHexagon) (distance : ℝ) : ℝ × ℝ :=
  sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem hexagon_walk (h : RegularHexagon) :
  let start := (0, 0)
  let end_point := PerimeterPoint h 8
  distance start end_point = 1 := by
  sorry

end hexagon_walk_l1026_102633


namespace sock_pair_combinations_l1026_102682

def choose (n k : Nat) : Nat :=
  if k > n then 0
  else (List.range k).foldl (fun m i => m * (n - i) / (i + 1)) 1

theorem sock_pair_combinations : 
  let total_socks : Nat := 18
  let white_socks : Nat := 8
  let brown_socks : Nat := 6
  let blue_socks : Nat := 4
  choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 = 49 := by
  sorry

end sock_pair_combinations_l1026_102682


namespace lake_half_covered_l1026_102677

/-- Represents the number of lotuses in the lake on a given day -/
def lotuses (day : ℕ) : ℝ := 2^day

/-- The day when the lake is completely covered -/
def full_coverage_day : ℕ := 30

theorem lake_half_covered :
  lotuses (full_coverage_day - 1) = (1/2) * lotuses full_coverage_day :=
by sorry

end lake_half_covered_l1026_102677


namespace volleyball_tournament_wins_l1026_102660

theorem volleyball_tournament_wins (n m : ℕ) : 
  ∀ (x : ℕ), 0 < x ∧ x < 73 →
  x * n + (73 - x) * m = 36 * 73 →
  n = m := by
sorry

end volleyball_tournament_wins_l1026_102660


namespace sakshi_tanya_efficiency_increase_l1026_102651

/-- The percentage increase in efficiency between two work rates -/
def efficiency_increase (rate1 rate2 : ℚ) : ℚ :=
  (rate2 - rate1) / rate1 * 100

/-- Theorem stating the efficiency increase from Sakshi to Tanya -/
theorem sakshi_tanya_efficiency_increase :
  let sakshi_rate : ℚ := 1/5
  let tanya_rate : ℚ := 1/4
  efficiency_increase sakshi_rate tanya_rate = 25 := by
  sorry

end sakshi_tanya_efficiency_increase_l1026_102651


namespace annies_ride_distance_l1026_102678

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startFee : ℝ
  perMileFee : ℝ
  tollFee : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (fare : TaxiFare) (miles : ℝ) : ℝ :=
  fare.startFee + fare.tollFee + fare.perMileFee * miles

theorem annies_ride_distance (mikeFare annieFare : TaxiFare) 
  (h1 : mikeFare.startFee = 2.5)
  (h2 : mikeFare.perMileFee = 0.25)
  (h3 : mikeFare.tollFee = 0)
  (h4 : annieFare.startFee = 2.5)
  (h5 : annieFare.perMileFee = 0.25)
  (h6 : annieFare.tollFee = 5)
  (h7 : totalFare mikeFare 36 = totalFare annieFare (annies_miles : ℝ)) :
  annies_miles = 16 := by
  sorry

end annies_ride_distance_l1026_102678


namespace angle_equality_l1026_102673

theorem angle_equality (α : Real) : 
  0 ≤ α ∧ α < 2 * Real.pi ∧ 
  (Real.sin α = Real.sin (215 * Real.pi / 180)) ∧ 
  (Real.cos α = Real.cos (215 * Real.pi / 180)) →
  α = 235 * Real.pi / 180 := by
sorry

end angle_equality_l1026_102673


namespace max_value_sin_tan_function_l1026_102636

theorem max_value_sin_tan_function :
  ∀ x : ℝ, 2 * Real.sin x ^ 2 - Real.tan x ^ 2 ≤ 3 - 2 * Real.sqrt 2 ∧
  ∃ x : ℝ, 2 * Real.sin x ^ 2 - Real.tan x ^ 2 = 3 - 2 * Real.sqrt 2 :=
by sorry

end max_value_sin_tan_function_l1026_102636


namespace modulus_of_complex_number_l1026_102679

theorem modulus_of_complex_number : Complex.abs (2 / (1 + Complex.I)^2) = 1 := by
  sorry

end modulus_of_complex_number_l1026_102679


namespace seashell_theorem_l1026_102623

def seashell_problem (mary_shells jessica_shells : ℕ) : Prop :=
  let kevin_shells := 3 * mary_shells
  let laura_shells := jessica_shells / 2
  mary_shells + jessica_shells + kevin_shells + laura_shells = 134

theorem seashell_theorem :
  seashell_problem 18 41 := by sorry

end seashell_theorem_l1026_102623


namespace perpendicular_vectors_x_value_l1026_102641

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (4, 3)
  let b : ℝ → ℝ × ℝ := λ x ↦ (6, x)
  ∀ x : ℝ, perpendicular a (b x) → x = -8 := by
sorry

end perpendicular_vectors_x_value_l1026_102641


namespace masha_sasha_numbers_l1026_102691

theorem masha_sasha_numbers 
  (a b : ℕ) 
  (h_distinct : a ≠ b) 
  (h_greater_11 : a > 11 ∧ b > 11) 
  (h_sum_known : ∃ s, s = a + b) 
  (h_one_even : Even a ∨ Even b) 
  (h_unique : ∀ x y : ℕ, x ≠ y → x > 11 → y > 11 → x + y = a + b → (Even x ∨ Even y) → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
sorry

end masha_sasha_numbers_l1026_102691


namespace alcohol_percentage_in_solution_a_l1026_102613

/-- Proves that the percentage of alcohol in Solution A is 27% given the specified conditions. -/
theorem alcohol_percentage_in_solution_a : ∀ x : ℝ,
  -- Solution A has 6 liters of water and x% of alcohol
  -- Solution B has 9 liters of a solution containing 57% alcohol
  -- After mixing, the new mixture has 45% alcohol concentration
  (6 * x + 9 * 0.57 = 15 * 0.45) →
  x = 0.27 := by
  sorry

end alcohol_percentage_in_solution_a_l1026_102613


namespace four_digit_multiples_of_seven_l1026_102676

theorem four_digit_multiples_of_seven : 
  (Finset.filter (fun n : ℕ => n % 7 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999) (Finset.range 10000)).card = 1286 := by
  sorry

end four_digit_multiples_of_seven_l1026_102676


namespace inconsistent_fraction_problem_l1026_102624

theorem inconsistent_fraction_problem :
  ¬ ∃ (f : ℚ), (f * 4 = 8) ∧ ((1/8) * 4 = 3) := by
  sorry

end inconsistent_fraction_problem_l1026_102624


namespace difference_of_numbers_l1026_102688

theorem difference_of_numbers (x y : ℝ) 
  (sum_condition : x + y = 36) 
  (product_condition : x * y = 320) : 
  |x - y| = 4 := by
sorry

end difference_of_numbers_l1026_102688


namespace floor_mod_equivalence_l1026_102694

theorem floor_mod_equivalence (k : ℤ) (a b : ℝ) (h : k ≥ 2) :
  (∃ m : ℤ, a - b = m * k) ↔
  (∀ n : ℕ, n > 0 → ⌊a * n⌋ % k = ⌊b * n⌋ % k) :=
by sorry

end floor_mod_equivalence_l1026_102694


namespace paint_area_calculation_l1026_102693

/-- Calculates the area to be painted on a wall with a door. -/
def areaToPaint (wallHeight wallLength doorHeight doorWidth : ℝ) : ℝ :=
  wallHeight * wallLength - doorHeight * doorWidth

/-- Proves that the area to be painted on a 10ft by 15ft wall with a 3ft by 5ft door is 135 sq ft. -/
theorem paint_area_calculation :
  areaToPaint 10 15 3 5 = 135 := by
  sorry

end paint_area_calculation_l1026_102693


namespace B_parity_2021_2022_2023_l1026_102609

def B : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => B (n + 2) + B (n + 1)

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem B_parity_2021_2022_2023 :
  is_odd (B 2021) ∧ is_odd (B 2022) ∧ ¬is_odd (B 2023) := by sorry

end B_parity_2021_2022_2023_l1026_102609


namespace arithmetic_sequence_sum_l1026_102604

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum : 
  2 * arithmetic_sum 51 99 2 = 3750 := by
  sorry

end arithmetic_sequence_sum_l1026_102604


namespace charity_show_girls_l1026_102683

theorem charity_show_girls (initial_total : ℕ) (initial_girls : ℕ) : 
  initial_girls = initial_total / 2 →
  (initial_girls - 3 : ℚ) / (initial_total + 1 : ℚ) = 2/5 →
  initial_girls = 17 := by
sorry

end charity_show_girls_l1026_102683


namespace student_council_choices_l1026_102605

/-- Represents the composition of the student council -/
structure StudentCouncil where
  freshmen : Nat
  sophomores : Nat
  juniors : Nat

/-- The given student council composition -/
def council : StudentCouncil := ⟨6, 5, 4⟩

/-- Number of ways to choose one person as president -/
def choosePresident (sc : StudentCouncil) : Nat :=
  sc.freshmen + sc.sophomores + sc.juniors

/-- Number of ways to choose one person from each grade -/
def chooseOneFromEach (sc : StudentCouncil) : Nat :=
  sc.freshmen * sc.sophomores * sc.juniors

/-- Number of ways to choose two people from different grades -/
def chooseTwoFromDifferent (sc : StudentCouncil) : Nat :=
  sc.freshmen * sc.sophomores +
  sc.freshmen * sc.juniors +
  sc.sophomores * sc.juniors

theorem student_council_choices :
  choosePresident council = 15 ∧
  chooseOneFromEach council = 120 ∧
  chooseTwoFromDifferent council = 74 := by
  sorry

end student_council_choices_l1026_102605


namespace initial_water_percentage_l1026_102603

theorem initial_water_percentage
  (capacity : ℝ)
  (added_water : ℝ)
  (final_fraction : ℝ)
  (h1 : capacity = 80)
  (h2 : added_water = 28)
  (h3 : final_fraction = 3/4)
  (h4 : final_fraction * capacity = (initial_percentage / 100) * capacity + added_water) :
  initial_percentage = 40 := by
  sorry

end initial_water_percentage_l1026_102603


namespace complementary_not_supplementary_l1026_102655

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- Two angles are supplementary if their sum is 180 degrees -/
def supplementary (a b : ℝ) : Prop := a + b = 180

/-- Theorem: It is impossible for two angles to be both complementary and supplementary -/
theorem complementary_not_supplementary : ¬ ∃ (a b : ℝ), complementary a b ∧ supplementary a b := by
  sorry

end complementary_not_supplementary_l1026_102655


namespace smallest_norm_l1026_102698

open Real

/-- Given a vector v in ℝ² such that ‖v + (4, -2)‖ = 10, 
    the smallest possible value of ‖v‖ is 10 - 2√5 -/
theorem smallest_norm (v : ℝ × ℝ) 
  (h : ‖v + (4, -2)‖ = 10) : 
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * sqrt 5 ∧ ∀ u : ℝ × ℝ, ‖u + (4, -2)‖ = 10 → ‖w‖ ≤ ‖u‖ :=
sorry

end smallest_norm_l1026_102698


namespace exponent_division_l1026_102614

theorem exponent_division (a : ℝ) : 2 * a^3 / a^2 = 2 * a := by sorry

end exponent_division_l1026_102614


namespace distance_point_to_parametric_line_l1026_102653

/-- The distance from a point to a line defined parametrically -/
theorem distance_point_to_parametric_line :
  let P : ℝ × ℝ := (2, 0)
  let line (t : ℝ) : ℝ × ℝ := (1 + 4*t, 2 + 3*t)
  let distance (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) : ℝ :=
    -- Define distance function here (implementation not provided)
    sorry
  distance P line = 11/5 :=
by sorry

end distance_point_to_parametric_line_l1026_102653


namespace cubic_double_root_value_l1026_102626

theorem cubic_double_root_value (a b : ℝ) (p q r : ℝ) : 
  (∀ x : ℝ, x^3 + p*x^2 + q*x + r = (x - a)^2 * (x - b)) →
  p = -6 →
  q = 9 →
  r = 0 ∨ r = -4 :=
sorry

end cubic_double_root_value_l1026_102626


namespace parabola_sum_l1026_102611

/-- A parabola with equation y = ax^2 + bx + c, vertex (3, -2), and passing through (0, 5) -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ
  vertex_x : ℚ := 3
  vertex_y : ℚ := -2
  point_x : ℚ := 0
  point_y : ℚ := 5
  eq_at_vertex : -2 = a * 3^2 + b * 3 + c
  eq_at_point : 5 = c
  vertex_formula : b = -2 * a * vertex_x

theorem parabola_sum (p : Parabola) : p.a + p.b + p.c = 10/9 := by
  sorry

end parabola_sum_l1026_102611


namespace race_speed_ratio_l1026_102692

/-- Given a race with the following conditions:
  * The total race distance is 600 meters
  * Contestant A has a 100 meter head start
  * Contestant A wins by 200 meters
  This theorem proves that the ratio of the speeds of contestant A to contestant B is 5:4. -/
theorem race_speed_ratio (vA vB : ℝ) (vA_pos : vA > 0) (vB_pos : vB > 0) : 
  (600 - 100) / vA = 400 / vB → vA / vB = 5 / 4 := by
  sorry

end race_speed_ratio_l1026_102692


namespace units_digit_of_sequence_sum_l1026_102620

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sequence_term (n : ℕ) : ℕ := factorial n + 10

def sequence_sum (n : ℕ) : ℕ := (List.range n).map sequence_term |>.sum

theorem units_digit_of_sequence_sum :
  sequence_sum 10 % 10 = 3 := by sorry

end units_digit_of_sequence_sum_l1026_102620


namespace arctan_tan_difference_l1026_102654

theorem arctan_tan_difference (θ : Real) : 
  θ ∈ Set.Icc 0 (π / 2) →
  Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (30 * π / 180)) = 15 * π / 180 := by
  sorry

end arctan_tan_difference_l1026_102654


namespace sum_of_squares_not_divisible_by_17_l1026_102643

theorem sum_of_squares_not_divisible_by_17 (x y z : ℤ) :
  Nat.Coprime x.natAbs y.natAbs ∧
  Nat.Coprime x.natAbs z.natAbs ∧
  Nat.Coprime y.natAbs z.natAbs →
  (x + y + z) % 17 = 0 →
  (x * y * z) % 17 = 0 →
  (x^2 + y^2 + z^2) % 17 ≠ 0 := by
  sorry

end sum_of_squares_not_divisible_by_17_l1026_102643


namespace rocks_needed_l1026_102638

/-- The number of rocks Mrs. Hilt needs for her garden border -/
def total_rocks_needed : ℕ := 125

/-- The number of rocks Mrs. Hilt currently has -/
def rocks_on_hand : ℕ := 64

/-- Theorem: Mrs. Hilt needs 61 more rocks to complete her garden border -/
theorem rocks_needed : total_rocks_needed - rocks_on_hand = 61 := by
  sorry

end rocks_needed_l1026_102638


namespace most_suitable_sampling_methods_l1026_102632

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SystematicSampling
  | StratifiedSampling
  | SimpleRandomSampling

/-- Represents a survey scenario --/
structure SurveyScenario where
  description : String
  sampleSize : Nat

/-- Determines the most suitable sampling method for a given scenario --/
def mostSuitableSamplingMethod (scenario : SurveyScenario) : SamplingMethod :=
  sorry

/-- The three survey scenarios described in the problem --/
def scenario1 : SurveyScenario :=
  { description := "First-year high school students' mathematics learning, 2 students from each class",
    sampleSize := 2 }

def scenario2 : SurveyScenario :=
  { description := "Math competition results, 12 students selected from different score ranges",
    sampleSize := 12 }

def scenario3 : SurveyScenario :=
  { description := "Sports meeting, arranging tracks for 6 students in 400m race",
    sampleSize := 6 }

/-- Theorem stating the most suitable sampling methods for the given scenarios --/
theorem most_suitable_sampling_methods :
  (mostSuitableSamplingMethod scenario1 = SamplingMethod.SystematicSampling) ∧
  (mostSuitableSamplingMethod scenario2 = SamplingMethod.StratifiedSampling) ∧
  (mostSuitableSamplingMethod scenario3 = SamplingMethod.SimpleRandomSampling) :=
by sorry

end most_suitable_sampling_methods_l1026_102632


namespace unique_prime_square_equation_l1026_102696

theorem unique_prime_square_equation : 
  ∃! p : ℕ, Prime p ∧ ∃ k : ℕ, 2 * p^4 - 7 * p^2 + 1 = k^2 := by sorry

end unique_prime_square_equation_l1026_102696


namespace triangle_area_calculation_l1026_102662

theorem triangle_area_calculation (a b : Real) (C : Real) :
  a = 45 ∧ b = 60 ∧ C = 37 →
  abs ((1/2) * a * b * Real.sin (C * Real.pi / 180) - 812.45) < 0.01 :=
by sorry

end triangle_area_calculation_l1026_102662


namespace east_west_southwest_angle_l1026_102649

/-- Represents the directions of rays in the decagon arrangement -/
inductive Direction
| North
| East
| WestSouthWest

/-- Represents a regular decagon with rays -/
structure DecagonArrangement where
  rays : Fin 10 → Direction
  north_ray : ∃ i, rays i = Direction.North

/-- Calculates the number of sectors between two directions -/
def sectors_between (d1 d2 : Direction) : ℕ := sorry

/-- Calculates the angle in degrees between two rays -/
def angle_between (d1 d2 : Direction) : ℝ :=
  (sectors_between d1 d2 : ℝ) * 36

theorem east_west_southwest_angle (arrangement : DecagonArrangement) :
  angle_between Direction.East Direction.WestSouthWest = 180 := by sorry

end east_west_southwest_angle_l1026_102649


namespace circle_diameter_l1026_102634

theorem circle_diameter (C : ℝ) (h : C = 36) : 
  (C / π) = (36 : ℝ) / π := by sorry

end circle_diameter_l1026_102634


namespace michael_has_100_cards_l1026_102697

/-- The number of Pokemon cards each person has -/
structure PokemonCards where
  lloyd : ℕ
  mark : ℕ
  michael : ℕ

/-- The conditions of the Pokemon card collection problem -/
def PokemonCardsProblem (cards : PokemonCards) : Prop :=
  (cards.mark = 3 * cards.lloyd) ∧
  (cards.michael = cards.mark + 10) ∧
  (cards.lloyd + cards.mark + cards.michael + 80 = 300)

/-- Theorem stating that under the given conditions, Michael has 100 cards -/
theorem michael_has_100_cards (cards : PokemonCards) :
  PokemonCardsProblem cards → cards.michael = 100 := by
  sorry


end michael_has_100_cards_l1026_102697


namespace specific_grid_has_nine_triangles_l1026_102610

/-- Represents the structure of the triangular grid with an additional triangle -/
structure TriangularGrid :=
  (bottom_row : Nat)
  (middle_row : Nat)
  (top_row : Nat)
  (additional : Nat)

/-- Counts the total number of triangles in the given grid structure -/
def count_triangles (grid : TriangularGrid) : Nat :=
  sorry

/-- Theorem stating that the specific grid structure has 9 triangles in total -/
theorem specific_grid_has_nine_triangles :
  let grid := TriangularGrid.mk 3 2 1 1
  count_triangles grid = 9 :=
by sorry

end specific_grid_has_nine_triangles_l1026_102610


namespace roof_length_width_difference_l1026_102630

/-- Given a rectangular roof with length 4 times its width and an area of 900 square feet,
    prove that the difference between the length and width is 24√5 feet. -/
theorem roof_length_width_difference (w : ℝ) (h1 : w > 0) (h2 : 5 * w * w = 900) :
  5 * w - w = 24 * Real.sqrt 5 := by
  sorry

end roof_length_width_difference_l1026_102630


namespace rosie_pies_theorem_l1026_102621

/-- Represents the number of pies that can be made from a given number of apples -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples / 12) * 3

theorem rosie_pies_theorem :
  pies_from_apples 36 = 9 :=
by sorry

end rosie_pies_theorem_l1026_102621


namespace addison_sunday_ticket_sales_l1026_102629

/-- Proves that Addison sold 78 raffle tickets on Sunday given the conditions of the problem -/
theorem addison_sunday_ticket_sales : 
  ∀ (friday saturday sunday : ℕ),
  friday = 181 →
  saturday = 2 * friday →
  saturday = sunday + 284 →
  sunday = 78 := by sorry

end addison_sunday_ticket_sales_l1026_102629


namespace inequalities_hold_l1026_102664

theorem inequalities_hold (m n l : ℝ) (h1 : m > n) (h2 : n > l) : 
  (m + 1/m > n + 1/n) ∧ (m + 1/n > n + 1/m) := by
  sorry

end inequalities_hold_l1026_102664


namespace floor_sum_equals_126_l1026_102672

theorem floor_sum_equals_126 
  (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (eq1 : x^2 + y^2 = 2010)
  (eq2 : z^2 + w^2 = 2010)
  (eq3 : x * z = 1008)
  (eq4 : y * w = 1008) :
  ⌊x + y + z + w⌋ = 126 := by
sorry

end floor_sum_equals_126_l1026_102672


namespace circle_max_min_linear_function_l1026_102600

theorem circle_max_min_linear_function :
  ∀ x y : ℝ, x^2 + y^2 = 16*x + 8*y + 20 →
  (∀ x' y' : ℝ, x'^2 + y'^2 = 16*x' + 8*y' + 20 → 4*x' + 3*y' ≤ 116) ∧
  (∀ x' y' : ℝ, x'^2 + y'^2 = 16*x' + 8*y' + 20 → 4*x' + 3*y' ≥ -64) ∧
  (∃ x₁ y₁ : ℝ, x₁^2 + y₁^2 = 16*x₁ + 8*y₁ + 20 ∧ 4*x₁ + 3*y₁ = 116) ∧
  (∃ x₂ y₂ : ℝ, x₂^2 + y₂^2 = 16*x₂ + 8*y₂ + 20 ∧ 4*x₂ + 3*y₂ = -64) :=
by sorry


end circle_max_min_linear_function_l1026_102600


namespace nathan_ate_twenty_gumballs_l1026_102628

/-- The number of gumballs in each package -/
def gumballs_per_package : ℕ := 5

/-- The number of packages Nathan finished -/
def packages_finished : ℕ := 4

/-- The total number of gumballs Nathan ate -/
def gumballs_eaten : ℕ := gumballs_per_package * packages_finished

theorem nathan_ate_twenty_gumballs : gumballs_eaten = 20 := by
  sorry

end nathan_ate_twenty_gumballs_l1026_102628


namespace lucky_lacy_correct_percentage_l1026_102669

/-- The percentage of problems Lucky Lacy got correct on an algebra test -/
theorem lucky_lacy_correct_percentage :
  ∀ x : ℕ,
  x > 0 →
  let total_problems := 7 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  let correct_fraction : ℚ := correct_problems / total_problems
  let correct_percentage := correct_fraction * 100
  ∃ ε > 0, abs (correct_percentage - 71.43) < ε :=
by
  sorry

end lucky_lacy_correct_percentage_l1026_102669


namespace square_on_parabola_diagonal_l1026_102617

/-- Given a square ABOC where O is the origin, A and B are on the parabola y = -x^2,
    and C is opposite to O, the length of diagonal AC is 2a, where a is the x-coordinate of point A. -/
theorem square_on_parabola_diagonal (a : ℝ) :
  let A : ℝ × ℝ := (a, -a^2)
  let B : ℝ × ℝ := (-a, -a^2)
  let O : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (a, a^2)
  -- ABOC is a square
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = (O.1 - C.1)^2 + (O.2 - C.2)^2 ∧
  (O.1 - C.1)^2 + (O.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →
  -- Length of AC is 2a
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (2*a)^2 :=
by sorry


end square_on_parabola_diagonal_l1026_102617


namespace expression_simplification_and_evaluation_l1026_102647

theorem expression_simplification_and_evaluation :
  ∀ a : ℚ, a + 1 ≠ 0 → a + 2 ≠ 0 →
  (a + 1 - (5 + 2*a) / (a + 1)) / ((a^2 + 4*a + 4) / (a + 1)) = (a - 2) / (a + 2) ∧
  (let simplified := (a - 2) / (a + 2);
   a = -3 → simplified = 5) :=
by sorry

end expression_simplification_and_evaluation_l1026_102647


namespace seventeen_doors_max_attempts_l1026_102616

/-- The maximum number of attempts needed to open n doors with n keys --/
def max_attempts (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 17 doors and 17 keys, the maximum number of attempts is 136 --/
theorem seventeen_doors_max_attempts :
  max_attempts 17 = 136 := by sorry

end seventeen_doors_max_attempts_l1026_102616


namespace sqrt_500_simplification_l1026_102663

theorem sqrt_500_simplification : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end sqrt_500_simplification_l1026_102663


namespace sufficient_not_necessary_condition_l1026_102680

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (∀ x, x < -1 → x^2 - 1 > 0) ∧ 
  (∃ x, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by sorry

end sufficient_not_necessary_condition_l1026_102680


namespace arithmetic_expression_equality_l1026_102665

theorem arithmetic_expression_equality : 5 * 7 + 10 * 4 - 36 / 3 + 6 * 3 = 81 := by
  sorry

end arithmetic_expression_equality_l1026_102665


namespace complex_number_in_first_quadrant_l1026_102667

def complex_number : ℂ := 2 + Complex.I

theorem complex_number_in_first_quadrant :
  Complex.re complex_number > 0 ∧ Complex.im complex_number > 0 :=
by sorry

end complex_number_in_first_quadrant_l1026_102667


namespace equation_equality_l1026_102695

theorem equation_equality (a b : ℝ) : -0.25 * a * b + (1/4) * a * b = 0 := by
  sorry

end equation_equality_l1026_102695


namespace certain_number_equation_l1026_102652

theorem certain_number_equation (x : ℝ) : (10 + 20 + 60) / 3 = ((10 + x + 25) / 3) + 5 ↔ x = 40 := by
  sorry

end certain_number_equation_l1026_102652


namespace product_sum_fractions_l1026_102631

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 - 1/5) = 23 := by
  sorry

end product_sum_fractions_l1026_102631


namespace area_of_special_quadrilateral_l1026_102668

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (a b c : Point)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (a b c d : Point)

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
noncomputable def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- Main theorem -/
theorem area_of_special_quadrilateral 
  (mainTriangle : Triangle) 
  (smallTriangles : Fin 4 → Triangle)
  (quadrilaterals : Fin 3 → Quadrilateral)
  (h1 : ∀ i, triangleArea (smallTriangles i) = 1)
  (h2 : quadrilateralArea (quadrilaterals 0) = quadrilateralArea (quadrilaterals 1))
  (h3 : quadrilateralArea (quadrilaterals 1) = quadrilateralArea (quadrilaterals 2)) :
  quadrilateralArea (quadrilaterals 0) = 1 + Real.sqrt 5 := by
  sorry

end area_of_special_quadrilateral_l1026_102668


namespace quadratic_root_problem_l1026_102644

theorem quadratic_root_problem (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + m*x - 6
  f (-6) = 0 → ∃ (x : ℝ), x ≠ -6 ∧ f x = 0 ∧ x = 1 :=
by sorry

end quadratic_root_problem_l1026_102644


namespace function_range_implies_m_range_l1026_102618

-- Define the function f(x) = x^2 - 2x + 5
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

-- Define the theorem
theorem function_range_implies_m_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 5) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 4) →
  m ∈ Set.Icc 1 2 :=
by sorry

end function_range_implies_m_range_l1026_102618


namespace xiaolis_estimate_l1026_102646

theorem xiaolis_estimate (p q a b : ℝ) 
  (h1 : p > q) (h2 : q > 0) (h3 : a > b) (h4 : b > 0) : 
  (p + a) - (q + b) > p - q := by
  sorry

end xiaolis_estimate_l1026_102646


namespace canadian_olympiad_2008_l1026_102687

theorem canadian_olympiad_2008 (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  (a - b*c)/(a + b*c) + (b - c*a)/(b + c*a) + (c - a*b)/(c + a*b) ≤ 3/2 := by
  sorry

end canadian_olympiad_2008_l1026_102687


namespace complex_number_opposite_parts_l1026_102681

theorem complex_number_opposite_parts (m : ℝ) : 
  let z : ℂ := (1 - m * I) / (1 - 2 * I)
  (∃ (a : ℝ), z = a - a * I) → m = -3 := by
sorry

end complex_number_opposite_parts_l1026_102681


namespace luke_paula_commute_l1026_102661

/-- The problem of Luke and Paula's commute times -/
theorem luke_paula_commute :
  -- Luke's bus time to work
  ∀ (luke_bus : ℕ),
  -- Paula's bus time as a fraction of Luke's
  ∀ (paula_fraction : ℚ),
  -- Total travel time for both
  ∀ (total_time : ℕ),
  -- Conditions
  luke_bus = 70 →
  paula_fraction = 3/5 →
  total_time = 504 →
  -- Conclusion
  ∃ (bike_multiple : ℚ),
    -- Luke's bike time = bus time * bike_multiple
    (luke_bus * bike_multiple).floor +
    -- Luke's bus time to work
    luke_bus +
    -- Paula's bus time to work
    (paula_fraction * luke_bus).floor +
    -- Paula's bus time back home
    (paula_fraction * luke_bus).floor = total_time ∧
    bike_multiple = 5 :=
by sorry

end luke_paula_commute_l1026_102661


namespace intersection_of_M_and_N_l1026_102601

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end intersection_of_M_and_N_l1026_102601


namespace complex_number_in_fourth_quadrant_l1026_102657

theorem complex_number_in_fourth_quadrant : ∃ (z : ℂ), z = Complex.mk (Real.sin 3) (Real.cos 3) ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end complex_number_in_fourth_quadrant_l1026_102657


namespace min_sum_three_digit_numbers_l1026_102650

def is_valid_triple (a b c : Nat) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ 
  b ≥ 100 ∧ b < 1000 ∧ 
  c ≥ 100 ∧ c < 1000 ∧ 
  a + b = c

def uses_distinct_digits (a b c : Nat) : Prop :=
  let digits := a.digits 10 ++ b.digits 10 ++ c.digits 10
  digits.length = 9 ∧ digits.toFinset.card = 9 ∧ 
  ∀ d ∈ digits, d ≥ 1 ∧ d ≤ 9

theorem min_sum_three_digit_numbers :
  ∃ a b c : Nat, is_valid_triple a b c ∧ 
  uses_distinct_digits a b c ∧
  (∀ x y z : Nat, is_valid_triple x y z → uses_distinct_digits x y z → 
    a + b + c ≤ x + y + z) ∧
  a + b + c = 459 := by
  sorry

end min_sum_three_digit_numbers_l1026_102650


namespace negation_of_existence_negation_of_quadratic_equation_l1026_102612

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + 3*x = 4) ↔ (∀ x : ℝ, x^2 + 3*x ≠ 4) := by sorry

end negation_of_existence_negation_of_quadratic_equation_l1026_102612


namespace conic_is_ellipse_l1026_102602

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-1)^2) + Real.sqrt ((x-5)^2 + (y+3)^2) = 10

/-- The first focus of the ellipse -/
def focus1 : ℝ × ℝ := (0, 1)

/-- The second focus of the ellipse -/
def focus2 : ℝ × ℝ := (5, -3)

/-- The constant sum of distances from any point on the ellipse to the foci -/
def constant_sum : ℝ := 10

/-- Theorem stating that the given equation describes an ellipse -/
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), conic_equation x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

end conic_is_ellipse_l1026_102602


namespace inequality_proof_l1026_102675

theorem inequality_proof (e : ℝ) (h : e > 0) : 
  (1 : ℝ) / e > Real.log ((1 + e^2) / e^2) ∧ 
  Real.log ((1 + e^2) / e^2) > 1 / (1 + e^2) :=
sorry

end inequality_proof_l1026_102675


namespace problem_solution_l1026_102684

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 2) (h3 : c^2 / a = 3) : 
  a = 12^(1/7) := by
sorry

end problem_solution_l1026_102684


namespace sequence_length_l1026_102659

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem sequence_length : 
  ∃ n : ℕ, n > 0 ∧ 
  arithmetic_sequence 2.5 5 n = 62.5 ∧ 
  ∀ k : ℕ, k > n → arithmetic_sequence 2.5 5 k > 62.5 ∧
  n = 13 :=
by sorry

end sequence_length_l1026_102659


namespace mixture_volume_l1026_102640

theorem mixture_volume (V : ℝ) (h1 : V > 0) : 
  (0.20 * V = 0.15 * (V + 5)) → V = 15 := by
  sorry

end mixture_volume_l1026_102640


namespace vector_operation_l1026_102699

/-- Given vectors a and b in ℝ², prove that (1/2)a - (3/2)b equals (-1,2) -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (1, -1)) :
  (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by sorry

end vector_operation_l1026_102699


namespace bus_seat_difference_l1026_102615

/-- Represents the seating configuration of a bus --/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  back_seat_capacity : ℕ
  regular_seat_capacity : ℕ
  total_capacity : ℕ

/-- Theorem about the difference in seats between left and right sides of the bus --/
theorem bus_seat_difference (bus : BusSeating) : 
  bus.left_seats = 15 →
  bus.regular_seat_capacity = 3 →
  bus.back_seat_capacity = 8 →
  bus.total_capacity = 89 →
  bus.left_seats > bus.right_seats →
  bus.left_seats - bus.right_seats = 3 := by
  sorry

#check bus_seat_difference

end bus_seat_difference_l1026_102615


namespace dot_product_problem_l1026_102627

theorem dot_product_problem (a b : ℝ × ℝ) : 
  a = (2, 1) → a - b = (-1, 2) → a • b = 5 := by
  sorry

end dot_product_problem_l1026_102627


namespace no_real_solutions_l1026_102608

theorem no_real_solutions : 
  ¬∃ (x : ℝ), (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) := by
  sorry

end no_real_solutions_l1026_102608


namespace geometric_sequence_ratio_sum_l1026_102656

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) :
  k ≠ 0 →
  p ≠ 1 →
  r ≠ 1 →
  p ≠ r →
  a₂ = k * p →
  a₃ = k * p^2 →
  b₂ = k * r →
  b₃ = k * r^2 →
  a₃ - b₃ = 5 * (a₂ - b₂) →
  p + r = 5 := by
sorry

end geometric_sequence_ratio_sum_l1026_102656


namespace table_relationship_l1026_102642

def f (x : ℝ) : ℝ := 21 - x^2

theorem table_relationship : 
  (f 0 = 21) ∧ 
  (f 1 = 20) ∧ 
  (f 2 = 16) ∧ 
  (f 3 = 9) ∧ 
  (f 4 = 0) := by
  sorry

end table_relationship_l1026_102642


namespace hotline_probabilities_l1026_102685

theorem hotline_probabilities (p1 p2 p3 p4 : ℝ)
  (h1 : p1 = 0.1)
  (h2 : p2 = 0.2)
  (h3 : p3 = 0.3)
  (h4 : p4 = 0.35) :
  (p1 + p2 + p3 + p4 = 0.95) ∧ (1 - (p1 + p2 + p3 + p4) = 0.05) := by
  sorry

end hotline_probabilities_l1026_102685


namespace lost_card_number_l1026_102645

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ∈ Finset.range (n + 1)) : 
  (n * (n + 1)) / 2 - 101 = 4 :=
sorry

end lost_card_number_l1026_102645


namespace pencil_cost_l1026_102635

theorem pencil_cost (initial_amount : ℕ) (amount_left : ℕ) (candy_cost : ℕ) : 
  initial_amount = 43 → amount_left = 18 → candy_cost = 5 → 
  initial_amount - amount_left - candy_cost = 20 := by
  sorry

end pencil_cost_l1026_102635


namespace last_digit_of_total_edge_count_l1026_102619

/-- Represents an 8x8 chessboard -/
def Chessboard := Fin 8 × Fin 8

/-- Represents a 1x2 domino piece -/
def Domino := Σ' (i : Fin 8) (j : Fin 7), Unit

/-- A tiling of the chessboard with dominos -/
def Tiling := Chessboard → Option Domino

/-- The number of valid tilings of the chessboard -/
def numTilings : ℕ := 12988816

/-- An edge of the chessboard -/
inductive Edge
| horizontal (i : Fin 9) (j : Fin 8) : Edge
| vertical (i : Fin 8) (j : Fin 9) : Edge

/-- The number of tilings that include a given edge -/
def edgeCount (e : Edge) : ℕ := sorry

/-- The sum of edgeCount for all edges -/
def totalEdgeCount : ℕ := sorry

/-- Theorem: The last digit of totalEdgeCount is 4 -/
theorem last_digit_of_total_edge_count :
  totalEdgeCount % 10 = 4 := by sorry

end last_digit_of_total_edge_count_l1026_102619


namespace area_of_bounded_region_l1026_102648

/-- The area of the region bounded by x = 2, y = 2, and the coordinate axes is 4 square units. -/
theorem area_of_bounded_region : 
  let region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}
  ∃ (A : Set (ℝ × ℝ)), A = region ∧ MeasureTheory.volume A = 4 := by
  sorry

end area_of_bounded_region_l1026_102648


namespace quadratic_equation_transformation_l1026_102622

theorem quadratic_equation_transformation (x : ℝ) :
  (4 * x^2 + 8 * x - 468 = 0) →
  ∃ p q : ℝ, ((x + p)^2 = q) ∧ (q = 116) := by
sorry

end quadratic_equation_transformation_l1026_102622
