import Mathlib

namespace line_intercepts_sum_l3307_330781

/-- Given a line described by the equation y - 3 = -3(x + 2), 
    the sum of its x-intercept and y-intercept is -4 -/
theorem line_intercepts_sum : 
  ∀ (x y : ℝ), y - 3 = -3*(x + 2) → 
  ∃ (x_int y_int : ℝ), 
    (0 - 3 = -3*(x_int + 2)) ∧ 
    (y_int - 3 = -3*(0 + 2)) ∧ 
    (x_int + y_int = -4) := by
  sorry

end line_intercepts_sum_l3307_330781


namespace car_sale_percentage_increase_l3307_330779

theorem car_sale_percentage_increase 
  (P : ℝ) 
  (discount : ℝ) 
  (profit : ℝ) 
  (buying_price : ℝ) 
  (selling_price : ℝ) :
  discount = 0.1 →
  profit = 0.62000000000000014 →
  buying_price = P * (1 - discount) →
  selling_price = P * (1 + profit) →
  (selling_price - buying_price) / buying_price = 0.8000000000000002 :=
by sorry

end car_sale_percentage_increase_l3307_330779


namespace parallel_line_through_midpoint_l3307_330735

/-- Given two points A and B in ℝ², and a line L, 
    prove that the line passing through the midpoint of AB 
    and parallel to L has the equation 3x + y + 3 = 0 -/
theorem parallel_line_through_midpoint 
  (A B : ℝ × ℝ) 
  (hA : A = (-5, 2)) 
  (hB : B = (1, 4)) 
  (L : ℝ → ℝ) 
  (hL : ∀ x y, L x = y ↔ 3 * x + y - 2 = 0) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∀ x y, 3 * x + y + 3 = 0 ↔ 
    ∃ k, y - M.2 = k * (x - M.1) ∧ 
         ∀ x' y', L x' = y' → y' - M.2 = k * (x' - M.1) :=
by sorry

end parallel_line_through_midpoint_l3307_330735


namespace complex_fraction_simplification_l3307_330730

theorem complex_fraction_simplification :
  let z : ℂ := (1 - Complex.I * Real.sqrt 3) / (Complex.I + Real.sqrt 3) ^ 2
  z = -1/4 - (Complex.I * Real.sqrt 3) / 4 := by
    sorry

end complex_fraction_simplification_l3307_330730


namespace three_statements_incorrect_l3307_330790

-- Define the four statements
def statement1 : Prop := ∀ (a : ℕ → ℝ) (S : ℕ → ℝ), 
  (∀ n, a (n+1) - a n = a (n+2) - a (n+1)) → 
  (a 6 + a 7 > 0 ↔ S 9 ≥ S 3)

def statement2 : Prop := 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x < 1)

def statement3 : Prop := 
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 → (x = 1 ∨ x = 3)) ↔
  (∀ x : ℝ, (x ≠ 1 ∨ x ≠ 3) → x^2 - 4*x + 3 ≠ 0)

def statement4 : Prop := 
  ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

-- Theorem stating that exactly 3 statements are incorrect
theorem three_statements_incorrect : 
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ statement4) :=
sorry

end three_statements_incorrect_l3307_330790


namespace fraction_equality_l3307_330725

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 2) 
  (h2 : s / u = 11 / 7) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 2 * p * s) = -233 / 12 := by
  sorry

end fraction_equality_l3307_330725


namespace complement_of_union_l3307_330755

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union (h : U = {1, 2, 3, 4} ∧ M = {1, 2} ∧ N = {2, 3}) :
  U \ (M ∪ N) = {4} := by
  sorry

end complement_of_union_l3307_330755


namespace max_A_value_l3307_330794

def digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_arrangement (a b c d e f g h i : Nat) : Prop :=
  {a, b, c, d, e, f, g, h, i} = digits

def A (a b c d e f g h i : Nat) : Nat :=
  (100*a + 10*b + c) + (100*b + 10*c + d) + (100*c + 10*d + e) +
  (100*d + 10*e + f) + (100*e + 10*f + g) + (100*f + 10*g + h) +
  (100*g + 10*h + i)

theorem max_A_value :
  ∃ (a b c d e f g h i : Nat),
    is_valid_arrangement a b c d e f g h i ∧
    A a b c d e f g h i = 4648 ∧
    ∀ (a' b' c' d' e' f' g' h' i' : Nat),
      is_valid_arrangement a' b' c' d' e' f' g' h' i' →
      A a' b' c' d' e' f' g' h' i' ≤ 4648 :=
by sorry

end max_A_value_l3307_330794


namespace sin_30_degrees_l3307_330757

theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_30_degrees_l3307_330757


namespace robyn_cookie_sales_l3307_330736

theorem robyn_cookie_sales (total_sales lucy_sales : ℕ) 
  (h1 : total_sales = 76) 
  (h2 : lucy_sales = 29) : 
  total_sales - lucy_sales = 47 := by
  sorry

end robyn_cookie_sales_l3307_330736


namespace special_set_properties_l3307_330749

/-- A set M with specific closure properties -/
structure SpecialSet (M : Set ℝ) : Prop where
  zero_in : (0 : ℝ) ∈ M
  one_in : (1 : ℝ) ∈ M
  closed_sub : ∀ x y, x ∈ M → y ∈ M → (x - y) ∈ M
  closed_inv : ∀ x, x ∈ M → x ≠ 0 → (1 / x) ∈ M

/-- Properties of the special set M -/
theorem special_set_properties (M : Set ℝ) (h : SpecialSet M) :
  (1 / 3 ∈ M) ∧
  (-1 ∈ M) ∧
  (∀ x y, x ∈ M → y ∈ M → (x + y) ∈ M) ∧
  (∀ x, x ∈ M → x^2 ∈ M) := by
  sorry

end special_set_properties_l3307_330749


namespace unique_row_with_53_in_pascal_triangle_l3307_330714

theorem unique_row_with_53_in_pascal_triangle :
  ∃! n : ℕ, ∃ k : ℕ, k ≤ n ∧ Nat.choose n k = 53 :=
by sorry

end unique_row_with_53_in_pascal_triangle_l3307_330714


namespace jones_family_puzzle_l3307_330758

/-- Represents a 4-digit number where one digit appears three times and another once -/
structure LicensePlate where
  digits : Fin 4 → Nat
  pattern : ∃ (a b : Nat), (∀ i, digits i = a ∨ digits i = b) ∧
                           (∃ j, digits j ≠ digits ((j + 1) % 4))

/-- Mr. Jones' family setup -/
structure JonesFamily where
  license : LicensePlate
  children_ages : Finset Nat
  jones_age : Nat
  h1 : children_ages.card = 8
  h2 : 12 ∈ children_ages
  h3 : ∀ age ∈ children_ages, license.digits 0 * 1000 + license.digits 1 * 100 + 
                               license.digits 2 * 10 + license.digits 3 % age = 0
  h4 : jones_age = license.digits 1 * 10 + license.digits 0

theorem jones_family_puzzle (family : JonesFamily) : 11 ∉ family.children_ages := by
  sorry

end jones_family_puzzle_l3307_330758


namespace unique_solution_quadratic_linear_l3307_330711

/-- The system of equations y = x^2 and y = 2x + k has exactly one solution if and only if k = -1 -/
theorem unique_solution_quadratic_linear (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 ∧ p.2 = 2 * p.1 + k) ↔ k = -1 := by
  sorry

end unique_solution_quadratic_linear_l3307_330711


namespace floor_nested_equation_l3307_330720

theorem floor_nested_equation (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := by
  sorry

end floor_nested_equation_l3307_330720


namespace odd_function_sum_zero_l3307_330775

-- Define the function f
def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + x + c

-- Define the property of being an odd function on an interval
def is_odd_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f (-x) = -f x) ∧ a + b = 0

-- Theorem statement
theorem odd_function_sum_zero (a b c : ℝ) :
  is_odd_on (f a c) a b → a + b + c = 0 := by
  sorry

end odd_function_sum_zero_l3307_330775


namespace arithmetic_sequence_problem_l3307_330707

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  n : ℕ  -- number of terms
  first_sum : ℕ  -- sum of first 4 terms
  last_sum : ℕ  -- sum of last 4 terms
  total_sum : ℕ  -- sum of all terms

/-- The theorem stating the properties of the specific arithmetic sequence -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.first_sum = 26)
  (h2 : seq.last_sum = 110)
  (h3 : seq.total_sum = 187) :
  seq.n = 11 := by
  sorry

end arithmetic_sequence_problem_l3307_330707


namespace rice_bag_problem_l3307_330771

theorem rice_bag_problem (initial_stock : ℕ) : 
  initial_stock - 23 + 132 = 164 → initial_stock = 55 := by
  sorry

end rice_bag_problem_l3307_330771


namespace downstream_distance_is_36_l3307_330700

/-- Represents the swimming scenario --/
structure SwimmingScenario where
  still_water_speed : ℝ
  upstream_distance : ℝ
  swim_time : ℝ

/-- Calculates the downstream distance given a swimming scenario --/
def downstream_distance (s : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating that the downstream distance is 36 km for the given scenario --/
theorem downstream_distance_is_36 (s : SwimmingScenario) 
  (h1 : s.still_water_speed = 9)
  (h2 : s.upstream_distance = 18)
  (h3 : s.swim_time = 3) : 
  downstream_distance s = 36 :=
sorry

end downstream_distance_is_36_l3307_330700


namespace john_vowel_learning_days_l3307_330796

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of days John takes to learn one alphabet -/
def days_per_alphabet : ℕ := 3

/-- The total number of days John needs to finish learning all vowels -/
def total_days : ℕ := num_vowels * days_per_alphabet

theorem john_vowel_learning_days : total_days = 15 := by
  sorry

end john_vowel_learning_days_l3307_330796


namespace parallel_vectors_x_value_l3307_330795

/-- Given two parallel vectors a = (-1, 4) and b = (x, 2), prove that x = -1/2 -/
theorem parallel_vectors_x_value (x : ℚ) :
  let a : ℚ × ℚ := (-1, 4)
  let b : ℚ × ℚ := (x, 2)
  (∃ (k : ℚ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2) →
  x = -1/2 := by
  sorry

end parallel_vectors_x_value_l3307_330795


namespace product_from_hcf_lcm_l3307_330728

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 55) (h2 : Nat.lcm a b = 1500) :
  a * b = 82500 := by
  sorry

end product_from_hcf_lcm_l3307_330728


namespace policeman_catches_thief_l3307_330710

/-- Represents a point on the square --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a pathway on the square --/
inductive Pathway
  | Edge : Pathway
  | Diagonal : Pathway

/-- Represents the square with its pathways --/
structure Square :=
  (side_length : ℝ)
  (pathways : List Pathway)

/-- Represents the positions and speeds of the policeman and thief --/
structure ChaseState :=
  (policeman_pos : Point)
  (thief_pos : Point)
  (policeman_speed : ℝ)
  (thief_speed : ℝ)

/-- Defines the chase dynamics --/
def chase (square : Square) (initial_state : ChaseState) : Prop :=
  sorry

theorem policeman_catches_thief 
  (square : Square) 
  (initial_state : ChaseState) 
  (h1 : square.side_length > 0)
  (h2 : square.pathways.length = 6)
  (h3 : initial_state.policeman_speed > 2.1 * initial_state.thief_speed)
  (h4 : initial_state.thief_speed > 0) :
  chase square initial_state :=
by sorry

end policeman_catches_thief_l3307_330710


namespace smallest_valid_seating_l3307_330770

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement is valid. -/
def is_valid_seating (table : CircularTable) : Prop :=
  table.seated_people > 0 ∧
  table.seated_people ≤ table.total_chairs ∧
  ∀ (new_seat : ℕ), new_seat < table.total_chairs →
    ∃ (occupied_seat : ℕ), occupied_seat < table.total_chairs ∧
      (new_seat = (occupied_seat + 1) % table.total_chairs ∨
       new_seat = (occupied_seat - 1 + table.total_chairs) % table.total_chairs)

/-- The main theorem stating the smallest valid number of seated people. -/
theorem smallest_valid_seating :
  ∀ (table : CircularTable),
    table.total_chairs = 80 →
    (is_valid_seating table ↔ table.seated_people ≥ 20) :=
by sorry

end smallest_valid_seating_l3307_330770


namespace corral_area_ratio_l3307_330718

/-- The side length of a small equilateral triangular corral -/
def small_side : ℝ := sorry

/-- The side length of the large equilateral triangular corral -/
def large_side : ℝ := 3 * small_side

/-- The area of a single small equilateral triangular corral -/
def small_area : ℝ := sorry

/-- The area of the large equilateral triangular corral -/
def large_area : ℝ := sorry

/-- The total area of all nine small equilateral triangular corrals -/
def total_small_area : ℝ := 9 * small_area

theorem corral_area_ratio : total_small_area = large_area := by
  sorry

end corral_area_ratio_l3307_330718


namespace quadratic_expression_value_l3307_330788

theorem quadratic_expression_value (x y z : ℝ) 
  (eq1 : 4*x + 2*y + z = 20)
  (eq2 : x + 4*y + 2*z = 26)
  (eq3 : 2*x + y + 4*z = 28) :
  20*x^2 + 24*x*y + 20*y^2 + 12*z^2 = 500 := by
sorry

end quadratic_expression_value_l3307_330788


namespace prob_green_is_five_sevenths_l3307_330706

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from the given containers -/
def prob_green (containers : List Container) : ℚ :=
  let total_prob := containers.map (λ c => (c.green : ℚ) / (c.red + c.green))
  (total_prob.sum) / containers.length

/-- Theorem: The probability of selecting a green ball is 5/7 -/
theorem prob_green_is_five_sevenths : 
  let containers := [
    Container.mk 8 4,  -- Container I
    Container.mk 3 4,  -- Container II
    Container.mk 3 4   -- Container III
  ]
  prob_green containers = 5/7 := by
  sorry

end prob_green_is_five_sevenths_l3307_330706


namespace problem_statement_l3307_330769

theorem problem_statement : (2025^2 - 2025) / 2025 = 2024 := by
  sorry

end problem_statement_l3307_330769


namespace sin_cos_sum_equals_half_l3307_330737

theorem sin_cos_sum_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (47 * π / 180) * Real.cos (103 * π / 180) = 1/2 := by
  sorry

end sin_cos_sum_equals_half_l3307_330737


namespace money_distribution_l3307_330719

theorem money_distribution (a b c d total : ℕ) : 
  a + b + c + d = total →
  2 * a = b →
  5 * a = 2 * c →
  a = d →
  a + b = 1800 →
  total = 4500 := by
sorry

end money_distribution_l3307_330719


namespace simplify_expression_l3307_330732

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end simplify_expression_l3307_330732


namespace cereal_eating_time_l3307_330733

/-- The time it takes for two people to eat a certain amount of cereal together -/
def eating_time (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- The proposition that Mr. Fat and Mr. Thin can eat 4 pounds of cereal in 37.5 minutes -/
theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 25 -- Mr. Thin's eating rate in pounds per minute
  let total_amount : ℚ := 4   -- Total amount of cereal in pounds
  eating_time fat_rate thin_rate total_amount = 75 / 2 := by
  sorry

#eval eating_time (1/15 : ℚ) (1/25 : ℚ) 4

end cereal_eating_time_l3307_330733


namespace car_speed_proof_l3307_330723

/-- Proves that a car's speed is 36 km/h given the conditions of the problem -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v) * 3600 = (1 / 40) * 3600 + 10 → v = 36 :=
by
  sorry

#check car_speed_proof

end car_speed_proof_l3307_330723


namespace abc_sum_sqrt_l3307_330751

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17) 
  (h2 : c + a = 20) 
  (h3 : a + b = 23) : 
  Real.sqrt (a * b * c * (a + b + c)) = 10 * Real.sqrt 273 := by
  sorry

end abc_sum_sqrt_l3307_330751


namespace regular_polygon_lattice_points_regular_polyhedra_lattice_points_l3307_330721

-- Define a 3D lattice point
def LatticePoint := ℤ × ℤ × ℤ

-- Define a regular polygon
structure RegularPolygon (n : ℕ) where
  vertices : List LatticePoint
  is_regular : Bool  -- This should be a proof in a real implementation
  vertex_count : vertices.length = n

-- Define a regular polyhedron
structure RegularPolyhedron where
  vertices : List LatticePoint
  is_regular : Bool  -- This should be a proof in a real implementation

-- Theorem for regular polygons
theorem regular_polygon_lattice_points :
  ∀ n : ℕ, (∃ p : RegularPolygon n, True) ↔ n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

-- Define the Platonic solids
inductive PlatonicSolid
  | Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

-- Function to check if a Platonic solid can have lattice point vertices
def has_lattice_vertices : PlatonicSolid → Prop
  | PlatonicSolid.Tetrahedron => True
  | PlatonicSolid.Cube => True
  | PlatonicSolid.Octahedron => True
  | PlatonicSolid.Dodecahedron => False
  | PlatonicSolid.Icosahedron => False

-- Theorem for regular polyhedra
theorem regular_polyhedra_lattice_points :
  ∀ s : PlatonicSolid, has_lattice_vertices s ↔
    (s = PlatonicSolid.Tetrahedron ∨ s = PlatonicSolid.Cube ∨ s = PlatonicSolid.Octahedron) :=
sorry

end regular_polygon_lattice_points_regular_polyhedra_lattice_points_l3307_330721


namespace plot_length_proof_l3307_330776

/-- Given a rectangular plot with width 50 meters, prove that if 56 poles
    are needed when placed 5 meters apart along the perimeter,
    then the length of the plot is 80 meters. -/
theorem plot_length_proof (width : ℝ) (num_poles : ℕ) (pole_distance : ℝ) (length : ℝ) :
  width = 50 →
  num_poles = 56 →
  pole_distance = 5 →
  2 * ((length / pole_distance) + 1) + 2 * ((width / pole_distance) + 1) = num_poles →
  length = 80 := by
  sorry


end plot_length_proof_l3307_330776


namespace smallest_among_given_numbers_l3307_330797

theorem smallest_among_given_numbers :
  let a := 1
  let b := Real.sqrt 2 / 2
  let c := Real.sqrt 3 / 3
  let d := Real.sqrt 5 / 5
  d < c ∧ d < b ∧ d < a := by
  sorry

end smallest_among_given_numbers_l3307_330797


namespace percentage_comparison_l3307_330798

theorem percentage_comparison (x : ℝ) : 0.9 * x > 0.8 * 30 + 12 → x > 40 := by
  sorry

end percentage_comparison_l3307_330798


namespace dining_bill_share_l3307_330765

theorem dining_bill_share (people : ℕ) (bill tip_percent tax_percent : ℚ) 
  (h_people : people = 15)
  (h_bill : bill = 350)
  (h_tip_percent : tip_percent = 18 / 100)
  (h_tax_percent : tax_percent = 5 / 100) :
  (bill + bill * tip_percent + bill * tax_percent) / people = 287 / 10 := by
  sorry

end dining_bill_share_l3307_330765


namespace conference_room_arrangements_l3307_330747

/-- The number of distinct arrangements of seats in a conference room -/
theorem conference_room_arrangements (n m : ℕ) (hn : n = 12) (hm : m = 4) :
  (Nat.choose n m) = 495 := by sorry

end conference_room_arrangements_l3307_330747


namespace closest_whole_number_to_expression_l3307_330717

theorem closest_whole_number_to_expression : 
  ∃ (n : ℕ), n = 1000 ∧ 
  ∀ (m : ℕ), |((10^2010 + 5 * 10^2012) : ℚ) / ((2 * 10^2011 + 3 * 10^2011) : ℚ) - n| ≤ 
             |((10^2010 + 5 * 10^2012) : ℚ) / ((2 * 10^2011 + 3 * 10^2011) : ℚ) - m| :=
by sorry

end closest_whole_number_to_expression_l3307_330717


namespace rhombus_diagonals_perpendicular_rectangle_not_l3307_330748

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.A
  let (x₂, y₂) := q.B
  let (x₃, y₃) := q.C
  let (x₄, y₄) := q.D
  -- All sides are equal
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
  (x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₃ - x₄)^2 + (y₃ - y₄)^2 ∧
  (x₃ - x₄)^2 + (y₃ - y₄)^2 = (x₄ - x₁)^2 + (y₄ - y₁)^2

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.A
  let (x₂, y₂) := q.B
  let (x₃, y₃) := q.C
  let (x₄, y₄) := q.D
  -- Opposite sides are equal and all angles are right angles
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₃ - x₄)^2 + (y₃ - y₄)^2 ∧
  (x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₄ - x₁)^2 + (y₄ - y₁)^2 ∧
  (x₂ - x₁) * (x₃ - x₂) + (y₂ - y₁) * (y₃ - y₂) = 0 ∧
  (x₃ - x₂) * (x₄ - x₃) + (y₃ - y₂) * (y₄ - y₃) = 0 ∧
  (x₄ - x₃) * (x₁ - x₄) + (y₄ - y₃) * (y₁ - y₄) = 0 ∧
  (x₁ - x₄) * (x₂ - x₁) + (y₁ - y₄) * (y₂ - y₁) = 0

-- Define perpendicular diagonals
def perpendicular_diagonals (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.A
  let (x₃, y₃) := q.C
  let (x₂, y₂) := q.B
  let (x₄, y₄) := q.D
  (x₃ - x₁) * (x₄ - x₂) + (y₃ - y₁) * (y₄ - y₂) = 0

-- Theorem statement
theorem rhombus_diagonals_perpendicular_rectangle_not :
  (∀ q : Quadrilateral, is_rhombus q → perpendicular_diagonals q) ∧
  ¬(∀ q : Quadrilateral, is_rectangle q → perpendicular_diagonals q) :=
sorry

end rhombus_diagonals_perpendicular_rectangle_not_l3307_330748


namespace yarn_cost_calculation_l3307_330762

/-- Proves that the cost of each ball of yarn is $6 given the conditions of Chantal's sweater business -/
theorem yarn_cost_calculation (num_sweaters : ℕ) (yarn_per_sweater : ℕ) (price_per_sweater : ℚ) (total_profit : ℚ) : 
  num_sweaters = 28 →
  yarn_per_sweater = 4 →
  price_per_sweater = 35 →
  total_profit = 308 →
  (num_sweaters * price_per_sweater - total_profit) / (num_sweaters * yarn_per_sweater) = 6 := by
sorry

end yarn_cost_calculation_l3307_330762


namespace point_symmetry_and_quadrant_l3307_330782

theorem point_symmetry_and_quadrant (a : ℤ) : 
  (-1 - 2*a > 0) ∧ (2*a - 1 > 0) → a = 0 := by
  sorry

end point_symmetry_and_quadrant_l3307_330782


namespace system_solvability_l3307_330727

/-- The system of equations and inequalities -/
def system (a b x y : ℝ) : Prop :=
  x * Real.cos a - y * Real.sin a - 3 ≤ 0 ∧
  x^2 + y^2 - 8*x + 2*y - b^2 - 6*b + 8 = 0

/-- The set of valid b values -/
def valid_b_set : Set ℝ :=
  {b | b ≤ -Real.sqrt 17 ∨ b ≥ Real.sqrt 17 - 6}

/-- Theorem stating the equivalence between the system having a solution
    for any a and b being in the valid set -/
theorem system_solvability (b : ℝ) :
  (∀ a, ∃ x y, system a b x y) ↔ b ∈ valid_b_set :=
sorry

end system_solvability_l3307_330727


namespace walk_distance_proof_l3307_330705

/-- Given a total distance walked and a distance walked before rest,
    calculate the distance walked after rest. -/
def distance_after_rest (total : Real) (before_rest : Real) : Real :=
  total - before_rest

/-- Theorem: Given a total distance of 1 mile and a distance of 0.75 mile
    walked before rest, the distance walked after rest is 0.25 mile. -/
theorem walk_distance_proof :
  let total_distance : Real := 1
  let before_rest : Real := 0.75
  distance_after_rest total_distance before_rest = 0.25 := by
  sorry

end walk_distance_proof_l3307_330705


namespace average_book_width_l3307_330709

def book_widths : List ℝ := [3, 7.5, 1.25, 0.75, 4, 12]

theorem average_book_width : 
  (book_widths.sum / book_widths.length : ℝ) = 4.75 := by
  sorry

end average_book_width_l3307_330709


namespace coin_flip_sequences_l3307_330780

theorem coin_flip_sequences (n k : ℕ) (hn : n = 10) (hk : k = 6) :
  (Nat.choose n k) = 210 := by
  sorry

end coin_flip_sequences_l3307_330780


namespace u_converges_to_L_least_k_is_zero_l3307_330712

def u : ℕ → ℚ
  | 0 => 1/3
  | n+1 => 3 * u n - 3 * (u n)^2

def L : ℚ := 1/3

theorem u_converges_to_L (n : ℕ) : |u n - L| ≤ 1 / 2^100 := by
  sorry

theorem least_k_is_zero : ∀ k : ℕ, (∀ n : ℕ, n < k → |u n - L| > 1 / 2^100) → k = 0 := by
  sorry

end u_converges_to_L_least_k_is_zero_l3307_330712


namespace simplify_sqrt_neg_seven_squared_l3307_330726

theorem simplify_sqrt_neg_seven_squared : Real.sqrt ((-7)^2) = 7 := by
  sorry

end simplify_sqrt_neg_seven_squared_l3307_330726


namespace newspaper_printing_time_l3307_330764

/-- Represents the time taken to print newspapers -/
def print_time (presses : ℕ) (newspapers : ℕ) : ℚ :=
  6 * (4 : ℚ) * newspapers / (8000 * presses)

theorem newspaper_printing_time :
  print_time 2 6000 = 9 :=
by sorry

end newspaper_printing_time_l3307_330764


namespace liam_money_left_l3307_330702

/-- Calculates the amount of money Liam has left after paying his bills -/
def money_left_after_bills (monthly_savings : ℕ) (savings_duration_months : ℕ) (bills : ℕ) : ℕ :=
  monthly_savings * savings_duration_months - bills

/-- Theorem stating that Liam will have $8,500 left after paying his bills -/
theorem liam_money_left :
  money_left_after_bills 500 24 3500 = 8500 := by
  sorry

#eval money_left_after_bills 500 24 3500

end liam_money_left_l3307_330702


namespace expression_value_l3307_330722

theorem expression_value (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : 
  -2*a - b^2 + 2*a*b = -41 := by
sorry

end expression_value_l3307_330722


namespace max_corner_sum_l3307_330734

/-- Represents a face of the cube -/
structure Face where
  value : Nat
  inv_value : Nat
  sum_eq_eight : value + inv_value = 8
  value_in_range : 1 ≤ value ∧ value ≤ 6

/-- Represents a cube with six faces -/
structure Cube where
  faces : Fin 6 → Face
  distinct : ∀ i j, i ≠ j → (faces i).value ≠ (faces j).value

/-- Represents a corner of the cube -/
structure Corner where
  cube : Cube
  face1 : Fin 6
  face2 : Fin 6
  face3 : Fin 6
  distinct : face1 ≠ face2 ∧ face2 ≠ face3 ∧ face1 ≠ face3
  adjacent : ¬ ((cube.faces face1).inv_value = (cube.faces face2).value ∨
                (cube.faces face1).inv_value = (cube.faces face3).value ∨
                (cube.faces face2).inv_value = (cube.faces face3).value)

/-- The sum of values at a corner -/
def cornerSum (c : Corner) : Nat :=
  (c.cube.faces c.face1).value + (c.cube.faces c.face2).value + (c.cube.faces c.face3).value

/-- The theorem to be proved -/
theorem max_corner_sum (c : Cube) : 
  ∀ corner : Corner, corner.cube = c → cornerSum corner ≤ 15 :=
sorry

end max_corner_sum_l3307_330734


namespace sin_45_degrees_l3307_330708

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end sin_45_degrees_l3307_330708


namespace ship_passengers_with_round_trip_tickets_l3307_330760

theorem ship_passengers_with_round_trip_tickets 
  (total_passengers : ℝ) 
  (h1 : total_passengers > 0) 
  (round_trip_with_car : ℝ) 
  (h2 : round_trip_with_car = 0.15 * total_passengers) 
  (h3 : round_trip_with_car > 0) 
  (round_trip_without_car_ratio : ℝ) 
  (h4 : round_trip_without_car_ratio = 0.6) :
  (round_trip_with_car / (1 - round_trip_without_car_ratio)) / total_passengers = 0.375 := by
sorry

end ship_passengers_with_round_trip_tickets_l3307_330760


namespace arithmetic_geometric_mean_inequality_l3307_330756

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (h1 : a > b) (h2 : b > 0) : 
  a + b > 2 * Real.sqrt (a * b) := by
  sorry

end arithmetic_geometric_mean_inequality_l3307_330756


namespace cos_sixty_degrees_equals_half_l3307_330789

theorem cos_sixty_degrees_equals_half : Real.cos (π / 3) = 1 / 2 := by
  sorry

end cos_sixty_degrees_equals_half_l3307_330789


namespace equation_solution_l3307_330713

theorem equation_solution : 
  ∃ (x₁ x₂ : ℚ), x₁ = 7/15 ∧ x₂ = 4/5 ∧ 
  (∀ x : ℚ, ⌊(5 + 6*x)/8⌋ = (15*x - 7)/5 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solution_l3307_330713


namespace triangle_formation_l3307_330745

/-- Triangle inequality theorem check for three sides -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Check if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 5 10 13 ∧
  ¬can_form_triangle 1 2 3 ∧
  ¬can_form_triangle 4 5 10 ∧
  ∀ a : ℝ, a > 0 → ¬can_form_triangle (2*a) (3*a) (6*a) :=
by sorry


end triangle_formation_l3307_330745


namespace rectangle_area_change_l3307_330731

/-- Given a rectangle with width s and height h, where increasing the width by 3 and
    decreasing the height by 3 doesn't change the area, prove that decreasing the width
    by 4 and increasing the height by 4 results in an area decrease of 28 square units. -/
theorem rectangle_area_change (s h : ℝ) (h_area : (s + 3) * (h - 3) = s * h) :
  s * h - (s - 4) * (h + 4) = 28 := by
  sorry

end rectangle_area_change_l3307_330731


namespace brick_height_calculation_l3307_330777

/-- Calculates the height of a brick given the wall dimensions, mortar percentage, brick dimensions, and number of bricks --/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ)
  (mortar_percentage : ℝ) (brick_length brick_width : ℝ) (num_bricks : ℕ) :
  wall_length = 10 →
  wall_width = 4 →
  wall_height = 5 →
  mortar_percentage = 0.1 →
  brick_length = 0.25 →
  brick_width = 0.15 →
  num_bricks = 6000 →
  ∃ (brick_height : ℝ),
    brick_height = 0.8 ∧
    (1 - mortar_percentage) * wall_length * wall_width * wall_height =
    (brick_length * brick_width * brick_height) * num_bricks :=
by sorry

end brick_height_calculation_l3307_330777


namespace exists_special_function_l3307_330741

theorem exists_special_function : 
  ∃ f : ℕ+ → ℕ+, f 1 = 2 ∧ ∀ n : ℕ+, f (f n) = f n + n :=
by sorry

end exists_special_function_l3307_330741


namespace right_triangle_hypotenuse_longest_obtuse_triangle_longest_side_l3307_330703

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_angles : 0 < α ∧ 0 < β ∧ 0 < γ
  h_sum_angles : α + β + γ = π

-- Theorem for right-angled triangle
theorem right_triangle_hypotenuse_longest (t : Triangle) (h_right : t.γ = π/2) :
  t.c ≥ t.a ∧ t.c ≥ t.b := by sorry

-- Theorem for obtuse-angled triangle
theorem obtuse_triangle_longest_side (t : Triangle) (h_obtuse : t.γ > π/2) :
  t.c > t.a ∧ t.c > t.b := by sorry

end right_triangle_hypotenuse_longest_obtuse_triangle_longest_side_l3307_330703


namespace max_stuck_guests_l3307_330792

/-- Represents a guest with their galosh size -/
structure Guest where
  size : Nat

/-- Represents the state of remaining guests and galoshes -/
structure State where
  guests : List Guest
  galoshes : List Nat

/-- Checks if a guest can wear a galosh -/
def canWear (g : Guest) (s : Nat) : Bool :=
  g.size ≤ s

/-- Defines a valid initial state with 10 guests and galoshes -/
def validInitialState (s : State) : Prop :=
  s.guests.length = 10 ∧ 
  s.galoshes.length = 10 ∧
  s.guests.map Guest.size = s.galoshes ∧
  s.galoshes.Nodup

/-- Defines a stuck state where no remaining guest can wear any remaining galosh -/
def isStuckState (s : State) : Prop :=
  ∀ g ∈ s.guests, ∀ sz ∈ s.galoshes, ¬ canWear g sz

/-- The main theorem stating the maximum number of guests that could be left -/
theorem max_stuck_guests (s : State) (h : validInitialState s) :
  ∀ s' : State, (∃ seq : List (Guest × Nat), s.guests.Sublist s'.guests ∧ 
                                             s.galoshes.Sublist s'.galoshes ∧
                                             isStuckState s') →
    s'.guests.length ≤ 5 :=
sorry

end max_stuck_guests_l3307_330792


namespace paintable_area_theorem_l3307_330739

/-- Calculates the total paintable area of rooms with given dimensions and unpaintable area -/
def totalPaintableArea (numRooms length width height unpaintableArea : ℕ) : ℕ :=
  let wallArea := 2 * (length * height + width * height)
  let paintableAreaPerRoom := wallArea - unpaintableArea
  numRooms * paintableAreaPerRoom

/-- Theorem stating that the total paintable area of 4 rooms with given dimensions is 1644 sq ft -/
theorem paintable_area_theorem :
  totalPaintableArea 4 15 12 9 75 = 1644 := by
  sorry

end paintable_area_theorem_l3307_330739


namespace complex_product_magnitude_l3307_330784

theorem complex_product_magnitude : Complex.abs (3 - 5*Complex.I) * Complex.abs (3 + 5*Complex.I) = 34 := by
  sorry

end complex_product_magnitude_l3307_330784


namespace quadratic_root_implies_m_value_l3307_330783

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 + (m+2)*x - 2 = 0 ∧ x = 1) → m = -1 := by
  sorry

end quadratic_root_implies_m_value_l3307_330783


namespace divisibility_problem_l3307_330768

theorem divisibility_problem (x q : ℤ) (hx : x > 0) (h_pos : q * x + 197 > 0) 
  (h_197 : 197 % x = 3) : (q * x + 197) % x = 3 := by
  sorry

end divisibility_problem_l3307_330768


namespace quadratic_discriminant_l3307_330716

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + (5 + 1/2)x - 1/2 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := -1/2

theorem quadratic_discriminant : discriminant a b c = 161/4 := by
  sorry

end quadratic_discriminant_l3307_330716


namespace rhombus_diagonal_l3307_330701

/-- Given a rhombus with area 150 cm² and one diagonal of length 10 cm, 
    prove that the length of the other diagonal is 30 cm. -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_area : area = 150)
  (h_d1 : d1 = 10)
  (h_rhombus_area : area = (d1 * d2) / 2) :
  d2 = 30 := by
  sorry

end rhombus_diagonal_l3307_330701


namespace contractor_absent_days_l3307_330791

/-- Proves that given the contract conditions, the number of absent days is 6 -/
theorem contractor_absent_days
  (total_days : ℕ)
  (pay_per_day : ℚ)
  (fine_per_day : ℚ)
  (total_amount : ℚ)
  (h1 : total_days = 30)
  (h2 : pay_per_day = 25)
  (h3 : fine_per_day = 7.5)
  (h4 : total_amount = 555) :
  ∃ (absent_days : ℕ),
    absent_days = 6 ∧
    (pay_per_day * (total_days - absent_days) - fine_per_day * absent_days = total_amount) :=
by sorry

end contractor_absent_days_l3307_330791


namespace rice_consumption_l3307_330761

theorem rice_consumption (initial_rice : ℕ) (daily_consumption : ℕ) (days : ℕ) 
  (h1 : initial_rice = 52)
  (h2 : daily_consumption = 9)
  (h3 : days = 3) :
  initial_rice - (daily_consumption * days) = 25 := by
  sorry

end rice_consumption_l3307_330761


namespace inverse_negation_false_l3307_330772

theorem inverse_negation_false : 
  ¬(∀ x : ℝ, (x^2 = 1 ∧ x ≠ 1) → x^2 ≠ 1) :=
sorry

end inverse_negation_false_l3307_330772


namespace johns_age_l3307_330787

/-- Given the ages of John, his dad, and his sister, prove that John is 25 years old. -/
theorem johns_age (john dad sister : ℕ) 
  (h1 : john + 30 = dad)
  (h2 : john + dad = 80)
  (h3 : sister = john - 5) :
  john = 25 := by
  sorry

end johns_age_l3307_330787


namespace investment_partnership_profit_share_l3307_330759

/-- Investment partnership problem -/
theorem investment_partnership_profit_share
  (invest_A invest_B invest_C invest_D : ℚ)
  (total_profit : ℚ)
  (h1 : invest_A = 3 * invest_B)
  (h2 : invest_B = 2/3 * invest_C)
  (h3 : invest_D = 1/2 * invest_A)
  (h4 : total_profit = 19900) :
  invest_B / (invest_A + invest_B + invest_C + invest_D) * total_profit = 2842.86 := by
sorry


end investment_partnership_profit_share_l3307_330759


namespace difference_of_fractions_difference_for_7000_l3307_330724

theorem difference_of_fractions (n : ℝ) : n * (1 / 10) - n * (1 / 1000) = n * (99 / 1000) :=
by sorry

theorem difference_for_7000 : 7000 * (1 / 10) - 7000 * (1 / 1000) = 693 :=
by sorry

end difference_of_fractions_difference_for_7000_l3307_330724


namespace triangle_t_range_l3307_330752

theorem triangle_t_range (A B C : ℝ) (a b c : ℝ) (t : ℝ) :
  0 < B → B < π / 2 →  -- B is acute
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  a * c = (1 / 4) * b ^ 2 →  -- given condition
  Real.sin A + Real.sin C = t * Real.sin B →  -- given condition
  t ∈ Set.Ioo (Real.sqrt 6 / 2) (Real.sqrt 2) :=
by
  sorry

end triangle_t_range_l3307_330752


namespace shot_cost_calculation_l3307_330754

def total_shot_cost (num_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) : ℕ :=
  num_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot

theorem shot_cost_calculation :
  total_shot_cost 3 4 2 5 = 120 := by
  sorry

end shot_cost_calculation_l3307_330754


namespace average_of_nine_numbers_l3307_330767

theorem average_of_nine_numbers (numbers : Fin 9 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4) / 5 = 99)
  (h2 : (numbers 4 + numbers 5 + numbers 6 + numbers 7 + numbers 8) / 5 = 100)
  (h3 : numbers 4 = 59) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + 
   numbers 5 + numbers 6 + numbers 7 + numbers 8) / 9 = 104 := by
sorry

end average_of_nine_numbers_l3307_330767


namespace vacation_cost_per_person_l3307_330799

theorem vacation_cost_per_person 
  (num_people : ℕ) 
  (airbnb_cost car_cost : ℚ) 
  (h1 : num_people = 8)
  (h2 : airbnb_cost = 3200)
  (h3 : car_cost = 800) :
  (airbnb_cost + car_cost) / num_people = 500 :=
by sorry

end vacation_cost_per_person_l3307_330799


namespace k_h_negative_three_equals_fifteen_l3307_330750

-- Define the function h
def h (x : ℝ) : ℝ := 5 * x^2 - 12

-- Define a variable k as a function from ℝ to ℝ
variable (k : ℝ → ℝ)

-- State the theorem
theorem k_h_negative_three_equals_fifteen
  (h_def : ∀ x, h x = 5 * x^2 - 12)
  (k_h_three : k (h 3) = 15) :
  k (h (-3)) = 15 := by
sorry

end k_h_negative_three_equals_fifteen_l3307_330750


namespace exponent_equality_comparison_l3307_330743

theorem exponent_equality_comparison : 
  (4^3 ≠ 3^4) ∧ 
  (-5^3 = (-5)^3) ∧ 
  ((-6)^2 ≠ -6^2) ∧ 
  (((-5/2)^2 : ℚ) ≠ ((-2/5)^2 : ℚ)) :=
by sorry

end exponent_equality_comparison_l3307_330743


namespace belt_and_road_population_scientific_notation_l3307_330785

theorem belt_and_road_population_scientific_notation :
  (4500000000 : ℝ) = 4.5 * (10 ^ 9) := by sorry

end belt_and_road_population_scientific_notation_l3307_330785


namespace constructible_prism_dimensions_l3307_330786

/-- Represents a brick with dimensions 1 × 2 × 4 -/
structure Brick :=
  (length : ℕ := 1)
  (width : ℕ := 2)
  (height : ℕ := 4)

/-- Represents a rectangular prism -/
structure RectangularPrism :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Predicate to check if a prism can be constructed from bricks -/
def can_construct (p : RectangularPrism) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    p.length = a ∧ p.width = 2 * b ∧ p.height = 4 * c

/-- Theorem stating that any constructible prism has dimensions a × 2b × 4c -/
theorem constructible_prism_dimensions (p : RectangularPrism) :
  can_construct p ↔ ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    p.length = a ∧ p.width = 2 * b ∧ p.height = 4 * c :=
sorry

end constructible_prism_dimensions_l3307_330786


namespace four_projectors_illuminate_plane_l3307_330773

/-- Represents a point on a plane with a projector --/
structure ProjectorPoint where
  x : ℝ
  y : ℝ
  direction : Nat -- 0: North, 1: East, 2: South, 3: West

/-- Represents the illuminated area by a projector --/
def illuminatedArea (p : ProjectorPoint) : Set (ℝ × ℝ) :=
  sorry

/-- The entire plane --/
def entirePlane : Set (ℝ × ℝ) :=
  sorry

/-- Theorem stating that four projector points can illuminate the entire plane --/
theorem four_projectors_illuminate_plane (p1 p2 p3 p4 : ProjectorPoint) :
  ∃ (d1 d2 d3 d4 : Nat), 
    (d1 < 4 ∧ d2 < 4 ∧ d3 < 4 ∧ d4 < 4) ∧
    (illuminatedArea {p1 with direction := d1} ∪ 
     illuminatedArea {p2 with direction := d2} ∪
     illuminatedArea {p3 with direction := d3} ∪
     illuminatedArea {p4 with direction := d4}) = entirePlane :=
  sorry

end four_projectors_illuminate_plane_l3307_330773


namespace exist_special_integers_l3307_330793

theorem exist_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
  sorry

end exist_special_integers_l3307_330793


namespace triangle_properties_l3307_330774

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B + t.b * Real.cos t.A = t.c / (2 * Real.cos t.C) ∧
  t.c = 6 ∧
  2 * Real.sqrt 3 = t.c * Real.sin t.C / 2

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.C = π / 3 ∧ t.a + t.b + t.c = 6 * Real.sqrt 3 + 6 :=
sorry

end triangle_properties_l3307_330774


namespace inequality_solution_set_l3307_330746

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 1) > 4/x + 19/10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end inequality_solution_set_l3307_330746


namespace university_admission_problem_l3307_330753

theorem university_admission_problem :
  let n_universities : ℕ := 8
  let n_selected_universities : ℕ := 2
  let n_students : ℕ := 3
  
  (Nat.choose n_universities n_selected_universities) * (2 ^ n_students) = 224 :=
by sorry

end university_admission_problem_l3307_330753


namespace consecutive_even_integers_sum_l3307_330742

theorem consecutive_even_integers_sum (a : ℤ) : 
  (a + (a + 4) = 144) → 
  (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 370) := by
  sorry

end consecutive_even_integers_sum_l3307_330742


namespace divisible_by_thirteen_l3307_330704

theorem divisible_by_thirteen (n : ℤ) : 
  13 ∣ (n^2 - 6*n - 4) ↔ n ≡ 3 [ZMOD 13] := by
sorry

end divisible_by_thirteen_l3307_330704


namespace partial_fraction_decomposition_l3307_330715

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 2 → (45*x - 31) / (x^2 - 3*x + 2) = M₁ / (x - 1) + M₂ / (x - 2)) →
  M₁ * M₂ = -826 := by
sorry

end partial_fraction_decomposition_l3307_330715


namespace coefficient_of_x_squared_l3307_330778

theorem coefficient_of_x_squared (x : ℝ) :
  ∃ (k n : ℝ), (3 * x + 2) * (2 * x - 7) = 6 * x^2 + k * x + n := by
  sorry

end coefficient_of_x_squared_l3307_330778


namespace downstream_distance_l3307_330738

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 30) 
  (h2 : stream_speed = 5) 
  (h3 : time = 2) : 
  boat_speed + stream_speed * time = 70 := by
  sorry

#check downstream_distance

end downstream_distance_l3307_330738


namespace quadratic_symmetry_l3307_330744

/-- A quadratic function with specific properties -/
def q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, q a b c x = q a b c (15 - x)) →
  q a b c 0 = -3 →
  q a b c 15 = -3 := by sorry

end quadratic_symmetry_l3307_330744


namespace regular_polygon_perimeter_l3307_330763

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 →
  side_length = 7 →
  exterior_angle = 90 →
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 28 := by
  sorry

#check regular_polygon_perimeter

end regular_polygon_perimeter_l3307_330763


namespace sum_of_v_at_specific_points_l3307_330729

-- Define the function v
def v (x : ℝ) : ℝ := x^3 - 3*x + 1

-- State the theorem
theorem sum_of_v_at_specific_points : 
  v 2 + v (-2) + v 1 + v (-1) = 4 := by
  sorry

end sum_of_v_at_specific_points_l3307_330729


namespace broomsticks_count_l3307_330740

/-- Represents the Halloween decorations problem --/
def halloween_decorations (skulls spiderwebs pumpkins cauldrons budget_left to_put_up total broomsticks : ℕ) : Prop :=
  skulls = 12 ∧
  spiderwebs = 12 ∧
  pumpkins = 2 * spiderwebs ∧
  cauldrons = 1 ∧
  budget_left = 20 ∧
  to_put_up = 10 ∧
  total = 83 ∧
  total = skulls + spiderwebs + pumpkins + cauldrons + budget_left + to_put_up + broomsticks

/-- Theorem stating that the number of broomsticks is 4 --/
theorem broomsticks_count :
  ∀ skulls spiderwebs pumpkins cauldrons budget_left to_put_up total broomsticks,
  halloween_decorations skulls spiderwebs pumpkins cauldrons budget_left to_put_up total broomsticks →
  broomsticks = 4 := by
  sorry

end broomsticks_count_l3307_330740


namespace largest_special_number_l3307_330766

/-- A function that returns true if all digits in a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- A function that returns true if a natural number is divisible by all of its digits -/
def divisible_by_all_digits (n : ℕ) : Prop := sorry

/-- A function that returns true if a natural number contains the digit 5 -/
def contains_digit_five (n : ℕ) : Prop := sorry

theorem largest_special_number : 
  ∀ n : ℕ, 
    has_distinct_digits n ∧ 
    divisible_by_all_digits n ∧ 
    contains_digit_five n →
    n ≤ 9315 :=
sorry

end largest_special_number_l3307_330766
