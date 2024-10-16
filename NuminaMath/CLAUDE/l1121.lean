import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_problem_l1121_112159

theorem rectangle_problem (x : ℝ) :
  (∃ a b : ℝ, 
    a > 0 ∧ b > 0 ∧
    a = 2 * b ∧
    2 * (a + b) = x ∧
    a * b = x) →
  x = 18 := by
sorry

end NUMINAMATH_CALUDE_rectangle_problem_l1121_112159


namespace NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l1121_112129

noncomputable def curve (x : ℝ) : ℝ := x^3

theorem tangent_intersection_y_coordinate 
  (a b : ℝ) 
  (hA : ∃ y, y = curve a) 
  (hB : ∃ y, y = curve b) 
  (h_perp : (3 * a^2) * (3 * b^2) = -1) :
  ∃ x y, y = -1/3 ∧ 
    y = 3 * a^2 * (x - a) + a^3 ∧ 
    y = 3 * b^2 * (x - b) + b^3 :=
sorry

end NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l1121_112129


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1121_112139

-- Define f as a differentiable function on (0, +∞)
variable (f : ℝ → ℝ)

-- Define the domain of f
variable (hf_diff : Differentiable ℝ f)
variable (hf_domain : ∀ x, x > 0 → f x ≠ 0)

-- Define the condition f(x) > x * f'(x)
variable (hf_cond : ∀ x, x > 0 → f x > x * (deriv f x))

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem inequality_solution_set :
  ∀ x > 0, x^2 * f (1/x) - f x < 0 ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1121_112139


namespace NUMINAMATH_CALUDE_min_value_theorem_l1121_112199

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.exp x = y * Real.log x + y * Real.log y) : 
  ∃ (m : ℝ), ∀ (x' y' : ℝ) (hx' : x' > 0) (hy' : y' > 0) 
  (h' : Real.exp x' = y' * Real.log x' + y' * Real.log y'), 
  (Real.exp x' / x' - Real.log y') ≥ m ∧ 
  (Real.exp x / x - Real.log y = m) ∧ 
  m = Real.exp 1 - 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1121_112199


namespace NUMINAMATH_CALUDE_multiples_of_10_and_12_within_100_l1121_112111

theorem multiples_of_10_and_12_within_100 : 
  ∃! n : ℕ, n ≤ 100 ∧ 10 ∣ n ∧ 12 ∣ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_multiples_of_10_and_12_within_100_l1121_112111


namespace NUMINAMATH_CALUDE_final_result_l1121_112178

def initial_value : ℕ := 10^8

def operation (n : ℕ) : ℕ := n * 3 / 2

def repeated_operation (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | k + 1 => repeated_operation (operation n) k

theorem final_result :
  repeated_operation initial_value 16 = 3^16 * 5^8 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l1121_112178


namespace NUMINAMATH_CALUDE_cost_reduction_over_two_years_l1121_112142

theorem cost_reduction_over_two_years (total_reduction : ℝ) (annual_reduction : ℝ) :
  total_reduction = 0.19 →
  (1 - annual_reduction) * (1 - annual_reduction) = 1 - total_reduction →
  annual_reduction = 0.1 := by
sorry

end NUMINAMATH_CALUDE_cost_reduction_over_two_years_l1121_112142


namespace NUMINAMATH_CALUDE_f_max_value_l1121_112128

/-- The quadratic function f(x) = -2x^2 - 5 -/
def f (x : ℝ) : ℝ := -2 * x^2 - 5

/-- The maximum value of f(x) is -5 -/
theorem f_max_value : ∃ (M : ℝ), M = -5 ∧ ∀ x, f x ≤ M := by sorry

end NUMINAMATH_CALUDE_f_max_value_l1121_112128


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l1121_112132

theorem nested_fraction_equality : 
  (((((3 + 2)⁻¹ + 2)⁻¹ + 1)⁻¹ + 2)⁻¹ + 1 : ℚ) = 59 / 43 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l1121_112132


namespace NUMINAMATH_CALUDE_james_alice_equation_equivalence_l1121_112170

theorem james_alice_equation_equivalence (d e : ℝ) : 
  (∀ x, |x - 8| = 3 ↔ x^2 + d*x + e = 0) ↔ (d = -16 ∧ e = 55) := by
  sorry

end NUMINAMATH_CALUDE_james_alice_equation_equivalence_l1121_112170


namespace NUMINAMATH_CALUDE_campsite_coordinates_l1121_112186

/-- Calculates the coordinates of a point that divides a line segment in a given ratio -/
def divideLineSegment (x1 y1 x2 y2 m n : ℚ) : ℚ × ℚ :=
  ((m * x2 + n * x1) / (m + n), (m * y2 + n * y1) / (m + n))

/-- The campsite coordinates problem -/
theorem campsite_coordinates :
  let annaStart : ℚ × ℚ := (3, -5)
  let bobStart : ℚ × ℚ := (7, 4)
  let campsite := divideLineSegment annaStart.1 annaStart.2 bobStart.1 bobStart.2 2 1
  campsite = (17/3, 1) := by
  sorry


end NUMINAMATH_CALUDE_campsite_coordinates_l1121_112186


namespace NUMINAMATH_CALUDE_sports_conference_games_l1121_112141

theorem sports_conference_games (n : ℕ) (d : ℕ) (intra : ℕ) (inter : ℕ) 
  (h1 : n = 16)
  (h2 : d = 2)
  (h3 : n = d * 8)
  (h4 : intra = 3)
  (h5 : inter = 2) :
  d * (Nat.choose 8 2 * intra) + (n / 2) * (n / 2) * inter = 296 := by
  sorry

end NUMINAMATH_CALUDE_sports_conference_games_l1121_112141


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1121_112193

theorem yellow_balls_count (m : ℕ) : 
  (5 : ℝ) / ((5 : ℝ) + m) = (1 : ℝ) / 5 → m = 20 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1121_112193


namespace NUMINAMATH_CALUDE_absolute_value_inequality_range_l1121_112143

theorem absolute_value_inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 3| ≥ a^2 + a) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_range_l1121_112143


namespace NUMINAMATH_CALUDE_shelter_animals_count_l1121_112196

theorem shelter_animals_count (cats : ℕ) (dogs : ℕ) 
  (h1 : cats = 645) (h2 : dogs = 567) : cats + dogs = 1212 := by
  sorry

end NUMINAMATH_CALUDE_shelter_animals_count_l1121_112196


namespace NUMINAMATH_CALUDE_first_child_born_1982_l1121_112114

/-- Represents the year the first child was born -/
def first_child_birth_year : ℕ := sorry

/-- The year the couple got married -/
def marriage_year : ℕ := 1980

/-- The year the second child was born -/
def second_child_birth_year : ℕ := 1984

/-- The year when the combined ages of children equal the years of marriage -/
def reference_year : ℕ := 1986

theorem first_child_born_1982 :
  (reference_year - first_child_birth_year) + (reference_year - second_child_birth_year) = reference_year - marriage_year →
  first_child_birth_year = 1982 :=
by sorry

end NUMINAMATH_CALUDE_first_child_born_1982_l1121_112114


namespace NUMINAMATH_CALUDE_largest_angle_obtuse_triangle_l1121_112181

/-- Given an obtuse, scalene triangle ABC with angle A measuring 30 degrees and angle B measuring 55 degrees,
    the measure of the largest interior angle is 95 degrees. -/
theorem largest_angle_obtuse_triangle (A B C : ℝ) (h_obtuse : A + B + C = 180) 
  (h_A : A = 30) (h_B : B = 55) (h_scalene : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  max A (max B C) = 95 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_obtuse_triangle_l1121_112181


namespace NUMINAMATH_CALUDE_remaining_books_value_l1121_112136

def total_books : ℕ := 55
def hardback_books : ℕ := 10
def paperback_books : ℕ := total_books - hardback_books
def hardback_price : ℕ := 20
def paperback_price : ℕ := 10
def books_sold : ℕ := 14

def remaining_books : ℕ := total_books - books_sold
def remaining_hardback : ℕ := min hardback_books remaining_books
def remaining_paperback : ℕ := remaining_books - remaining_hardback

def total_value : ℕ := remaining_hardback * hardback_price + remaining_paperback * paperback_price

theorem remaining_books_value :
  total_value = 510 :=
sorry

end NUMINAMATH_CALUDE_remaining_books_value_l1121_112136


namespace NUMINAMATH_CALUDE_taxi_driver_theorem_l1121_112125

def driving_distances : List Int := [5, -3, 6, -7, 6, -2, -5, -4, 6, -8]

def starting_price : ℕ := 8
def base_distance : ℕ := 3
def additional_rate : ℚ := 3/2

theorem taxi_driver_theorem :
  (List.sum driving_distances = -6) ∧
  (List.sum (List.take 7 driving_distances) = 0) ∧
  (starting_price + (8 - base_distance) * additional_rate = 31/2) ∧
  (∀ x : ℕ, x > base_distance → starting_price + (x - base_distance) * additional_rate = (3 * x + 7) / 2) :=
by sorry

end NUMINAMATH_CALUDE_taxi_driver_theorem_l1121_112125


namespace NUMINAMATH_CALUDE_inscribed_rhombus_rectangle_perimeter_l1121_112122

/-- A rhombus inscribed in a rectangle -/
structure InscribedRhombus where
  /-- Length of EA -/
  ea : ℝ
  /-- Length of FB -/
  fb : ℝ
  /-- Length of AD (rhombus side) -/
  ad : ℝ
  /-- Length of BC (rhombus side) -/
  bc : ℝ
  /-- EA is positive -/
  ea_pos : 0 < ea
  /-- FB is positive -/
  fb_pos : 0 < fb
  /-- AD is positive -/
  ad_pos : 0 < ad
  /-- BC is positive -/
  bc_pos : 0 < bc

/-- The perimeter of the rectangle containing the inscribed rhombus -/
def rectangle_perimeter (r : InscribedRhombus) : ℝ :=
  2 * (r.ea + r.ad + r.fb + r.bc)

/-- Theorem stating that for the given measurements, the rectangle perimeter is 238 -/
theorem inscribed_rhombus_rectangle_perimeter :
  ∃ r : InscribedRhombus, r.ea = 12 ∧ r.fb = 25 ∧ r.ad = 37 ∧ r.bc = 45 ∧ rectangle_perimeter r = 238 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rhombus_rectangle_perimeter_l1121_112122


namespace NUMINAMATH_CALUDE_greatest_six_digit_divisible_l1121_112127

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem greatest_six_digit_divisible (n : ℕ) : 
  is_six_digit n ∧ 
  21 ∣ n ∧ 
  35 ∣ n ∧ 
  66 ∣ n ∧ 
  110 ∣ n ∧ 
  143 ∣ n → 
  n ≤ 990990 :=
sorry

end NUMINAMATH_CALUDE_greatest_six_digit_divisible_l1121_112127


namespace NUMINAMATH_CALUDE_peggy_dolls_l1121_112102

/-- The number of dolls Peggy has at the end -/
def final_dolls (initial : ℕ) (grandmother : ℕ) : ℕ :=
  initial + grandmother + grandmother / 2

/-- Theorem stating that Peggy ends up with 51 dolls -/
theorem peggy_dolls : final_dolls 6 30 = 51 := by
  sorry

end NUMINAMATH_CALUDE_peggy_dolls_l1121_112102


namespace NUMINAMATH_CALUDE_fraction_sum_equals_ten_thirds_l1121_112195

theorem fraction_sum_equals_ten_thirds (a b : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a = 2) (h2 : b = 1) : 
  (a + b) / (a - b) + (a - b) / (a + b) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_ten_thirds_l1121_112195


namespace NUMINAMATH_CALUDE_other_number_calculation_l1121_112182

theorem other_number_calculation (a b : ℕ+) 
  (h1 : Nat.lcm a b = 7700)
  (h2 : Nat.gcd a b = 11)
  (h3 : a = 308) :
  b = 275 := by
  sorry

end NUMINAMATH_CALUDE_other_number_calculation_l1121_112182


namespace NUMINAMATH_CALUDE_shortest_reflected_light_path_l1121_112105

/-- The shortest path length for a reflected light ray -/
theorem shortest_reflected_light_path :
  let A : ℝ × ℝ := (-3, 9)
  let C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}
  let reflect_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (p : ℝ × ℝ),
    p.2 = 0 ∧
    p ∉ C ∧
    (∀ (q : ℝ × ℝ), q.2 = 0 ∧ q ∉ C →
      dist A p + dist p (reflect_point (2, 3)) ≤ dist A q + dist q (reflect_point (2, 3))) ∧
    dist A p + dist p (reflect_point (2, 3)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_shortest_reflected_light_path_l1121_112105


namespace NUMINAMATH_CALUDE_circle_tangency_problem_l1121_112147

theorem circle_tangency_problem : 
  let divisors := (Finset.range 150).filter (λ x => x > 0 ∧ 150 % x = 0)
  (divisors.card : ℕ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_problem_l1121_112147


namespace NUMINAMATH_CALUDE_cryptarithmetic_solution_l1121_112149

theorem cryptarithmetic_solution : 
  ∃! (K I S : Nat), 
    K < 10 ∧ I < 10 ∧ S < 10 ∧
    K ≠ I ∧ K ≠ S ∧ I ≠ S ∧
    100 * K + 10 * I + S + 100 * K + 10 * S + I = 100 * I + 10 * S + K ∧
    K = 4 ∧ I = 9 ∧ S = 5 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithmetic_solution_l1121_112149


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l1121_112108

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 10 * x - 5

-- State the theorem
theorem derivative_f_at_1 : 
  (deriv f) 1 = 11 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l1121_112108


namespace NUMINAMATH_CALUDE_restaurant_customers_l1121_112116

theorem restaurant_customers (total : ℕ) : 
  (3 : ℚ) / 5 * total + 10 = total → total = 25 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_customers_l1121_112116


namespace NUMINAMATH_CALUDE_negation_of_existence_less_than_zero_l1121_112115

theorem negation_of_existence_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_less_than_zero_l1121_112115


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_area_l1121_112156

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral :=
  (A B C D : ℝ × ℝ)
  (is_cyclic : sorry)

/-- The area of a cyclic quadrilateral. -/
def area (q : CyclicQuadrilateral) : ℝ := sorry

/-- The distance between two points in ℝ². -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem cyclic_quadrilateral_area :
  ∀ (q : CyclicQuadrilateral),
    distance q.A q.B = 1 →
    distance q.B q.C = 3 →
    distance q.C q.D = 2 →
    distance q.D q.A = 2 →
    area q = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_area_l1121_112156


namespace NUMINAMATH_CALUDE_chain_merge_time_theorem_l1121_112158

/-- Represents a chain with a certain number of links -/
structure Chain where
  links : ℕ

/-- Represents the time required for chain operations -/
structure ChainOperationTime where
  openLinkTime : ℕ
  closeLinkTime : ℕ

/-- Calculates the minimum time required to merge chains -/
def minTimeMergeChains (chains : List Chain) (opTime : ChainOperationTime) : ℕ :=
  sorry

/-- Theorem statement for the chain merging problem -/
theorem chain_merge_time_theorem (chains : List Chain) (opTime : ChainOperationTime) :
  chains.length = 6 ∧ 
  chains.all (λ c => c.links = 4) ∧
  opTime.openLinkTime = 1 ∧
  opTime.closeLinkTime = 3 →
  minTimeMergeChains chains opTime = 20 :=
sorry

end NUMINAMATH_CALUDE_chain_merge_time_theorem_l1121_112158


namespace NUMINAMATH_CALUDE_james_passenger_count_l1121_112187

/-- Calculates the total number of passengers James has seen --/
def total_passengers (total_vehicles : ℕ) (trucks : ℕ) (buses : ℕ) (cars : ℕ) 
  (truck_passengers : ℕ) (bus_passengers : ℕ) (taxi_passengers : ℕ) 
  (motorbike_passengers : ℕ) (car_passengers : ℕ) : ℕ :=
  let taxis := 2 * buses
  let motorbikes := total_vehicles - trucks - buses - taxis - cars
  trucks * truck_passengers + 
  buses * bus_passengers + 
  taxis * taxi_passengers + 
  motorbikes * motorbike_passengers + 
  cars * car_passengers

theorem james_passenger_count : 
  total_passengers 52 12 2 30 2 15 2 1 3 = 156 := by
  sorry

end NUMINAMATH_CALUDE_james_passenger_count_l1121_112187


namespace NUMINAMATH_CALUDE_infinitely_many_t_with_same_digit_sum_as_kt_l1121_112173

/-- Given a natural number, returns true if it doesn't contain any zeros in its decimal representation -/
def no_zeros (n : ℕ) : Prop := sorry

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Main theorem statement -/
theorem infinitely_many_t_with_same_digit_sum_as_kt :
  ∀ k : ℕ, ∃ f : ℕ → ℕ, 
    (∀ n : ℕ, n < n.succ) ∧ 
    (∀ n : ℕ, no_zeros (f n)) ∧
    (∀ n : ℕ, digit_sum (f n) = digit_sum (k * f n)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_t_with_same_digit_sum_as_kt_l1121_112173


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1121_112183

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y + 1)^2) - Real.sqrt ((x - 7)^2 + (y + 1)^2) = 4

-- Define the positive asymptote slope
def positive_asymptote_slope (h : ℝ → ℝ → Prop) : ℝ :=
  1

-- Theorem statement
theorem hyperbola_asymptote_slope :
  positive_asymptote_slope hyperbola_equation = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1121_112183


namespace NUMINAMATH_CALUDE_inequality_range_l1121_112174

theorem inequality_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) 
  ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1121_112174


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l1121_112137

theorem partial_fraction_sum (P Q R : ℚ) : 
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ -1 ∧ x ≠ 2 → 
    (x^2 + 5*x - 14) / ((x - 3)*(x + 1)*(x - 2)) = 
    P / (x - 3) + Q / (x + 1) + R / (x - 2)) →
  P + Q + R = 11.5 / 3 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l1121_112137


namespace NUMINAMATH_CALUDE_solve_for_m_l1121_112123

/-- Given that the solution set of mx + 2 > 0 is {x | x < 2}, prove that m = -1 -/
theorem solve_for_m (m : ℝ) 
  (h : ∀ x, mx + 2 > 0 ↔ x < 2) : 
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l1121_112123


namespace NUMINAMATH_CALUDE_apple_problem_l1121_112167

/-- Proves that given the conditions of the apple problem, each child originally had 15 apples -/
theorem apple_problem (num_children : Nat) (apples_eaten : Nat) (apples_sold : Nat) (apples_left : Nat) :
  num_children = 5 →
  apples_eaten = 8 →
  apples_sold = 7 →
  apples_left = 60 →
  ∃ x : Nat, num_children * x - apples_eaten - apples_sold = apples_left ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_problem_l1121_112167


namespace NUMINAMATH_CALUDE_comprehensive_investigation_is_census_l1121_112135

/-- A comprehensive investigation conducted on the subject of examination for a specific purpose -/
def comprehensive_investigation : Type := Unit

/-- Census as a type -/
def census : Type := Unit

/-- Theorem stating that a comprehensive investigation is equivalent to a census -/
theorem comprehensive_investigation_is_census : 
  comprehensive_investigation ≃ census := by sorry

end NUMINAMATH_CALUDE_comprehensive_investigation_is_census_l1121_112135


namespace NUMINAMATH_CALUDE_favorite_number_is_25_l1121_112189

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def digit_diff (n : ℕ) : ℕ := Int.natAbs ((n / 10) - (n % 10))

def has_unique_digit (n : ℕ) : Prop :=
  ∀ m : ℕ, is_two_digit m → is_perfect_square m → m ≠ n →
    (n / 10 ≠ m / 10 ∧ n / 10 ≠ m % 10) ∨ (n % 10 ≠ m / 10 ∧ n % 10 ≠ m % 10)

def non_unique_sum (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_two_digit m ∧ digit_sum m = digit_sum n

def non_unique_diff (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_two_digit m ∧ digit_diff m = digit_diff n

theorem favorite_number_is_25 :
  ∃! n : ℕ, is_two_digit n ∧ is_perfect_square n ∧ has_unique_digit n ∧
    non_unique_sum n ∧ non_unique_diff n ∧ n = 25 :=
by sorry

end NUMINAMATH_CALUDE_favorite_number_is_25_l1121_112189


namespace NUMINAMATH_CALUDE_ping_pong_balls_count_l1121_112197

/-- The number of ping-pong balls in the gym storage -/
def ping_pong_balls : ℕ :=
  let total_balls : ℕ := 240
  let baseball_boxes : ℕ := 35
  let baseballs_per_box : ℕ := 4
  let tennis_ball_boxes : ℕ := 6
  let tennis_balls_per_box : ℕ := 3
  let baseballs : ℕ := baseball_boxes * baseballs_per_box
  let tennis_balls : ℕ := tennis_ball_boxes * tennis_balls_per_box
  total_balls - (baseballs + tennis_balls)

theorem ping_pong_balls_count : ping_pong_balls = 82 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_balls_count_l1121_112197


namespace NUMINAMATH_CALUDE_fifth_odd_with_odd_factors_is_81_l1121_112150

/-- A function that returns true if a number is a perfect square, false otherwise -/
def is_perfect_square (n : ℕ) : Bool := sorry

/-- A function that returns true if a number has an odd number of factors, false otherwise -/
def has_odd_factors (n : ℕ) : Bool := is_perfect_square n

/-- A function that returns the nth odd integer with an odd number of factors -/
def nth_odd_with_odd_factors (n : ℕ) : ℕ := sorry

theorem fifth_odd_with_odd_factors_is_81 :
  nth_odd_with_odd_factors 5 = 81 := by sorry

end NUMINAMATH_CALUDE_fifth_odd_with_odd_factors_is_81_l1121_112150


namespace NUMINAMATH_CALUDE_no_positive_integral_solutions_l1121_112152

theorem no_positive_integral_solutions : 
  ¬ ∃ (x y : ℕ+), x.val^6 * y.val^6 - 13 * x.val^3 * y.val^3 + 36 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integral_solutions_l1121_112152


namespace NUMINAMATH_CALUDE_vector_expression_evaluation_l1121_112119

/-- Given the vector expression, prove that it equals the result vector. -/
theorem vector_expression_evaluation :
  (⟨3, -2⟩ : ℝ × ℝ) - 5 • ⟨2, -6⟩ + ⟨0, 3⟩ = ⟨-7, 31⟩ := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_evaluation_l1121_112119


namespace NUMINAMATH_CALUDE_change_percentage_closest_to_five_l1121_112120

def item_prices : List ℚ := [12.99, 9.99, 7.99, 6.50, 4.99, 3.75, 1.27]
def payment : ℚ := 50

def total_price : ℚ := item_prices.sum
def change : ℚ := payment - total_price
def change_percentage : ℚ := (change / payment) * 100

theorem change_percentage_closest_to_five :
  ∀ x ∈ [3, 5, 7, 10, 12], |change_percentage - 5| ≤ |change_percentage - x| :=
by sorry

end NUMINAMATH_CALUDE_change_percentage_closest_to_five_l1121_112120


namespace NUMINAMATH_CALUDE_ellipse_satisfies_conditions_l1121_112157

/-- An ellipse with foci on the y-axis, focal distance 4, and passing through (3,2) -/
def ellipse_equation (x y : ℝ) : Prop :=
  y^2 / 16 + x^2 / 12 = 1

/-- The focal distance of the ellipse -/
def focal_distance : ℝ := 4

/-- A point on the ellipse -/
def point_on_ellipse : ℝ × ℝ := (3, 2)

/-- Theorem stating that the ellipse equation satisfies the given conditions -/
theorem ellipse_satisfies_conditions :
  (∀ x y, ellipse_equation x y → (x = point_on_ellipse.1 ∧ y = point_on_ellipse.2)) ∧
  (∃ f₁ f₂ : ℝ, f₁ = -f₂ ∧ f₁^2 = (focal_distance/2)^2 ∧
    ∀ x y, ellipse_equation x y →
      (x^2 + (y - f₁)^2)^(1/2) + (x^2 + (y - f₂)^2)^(1/2) = 2 * (16^(1/2))) :=
sorry

end NUMINAMATH_CALUDE_ellipse_satisfies_conditions_l1121_112157


namespace NUMINAMATH_CALUDE_reflect_F_twice_l1121_112175

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Theorem stating that reflecting point F(1,3) over y-axis then x-axis results in F''(-1,-3) -/
theorem reflect_F_twice :
  let F : ℝ × ℝ := (1, 3)
  let F' := reflect_y F
  let F'' := reflect_x F'
  F'' = (-1, -3) := by sorry

end NUMINAMATH_CALUDE_reflect_F_twice_l1121_112175


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1121_112191

open Real

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 3) + 3
  f 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1121_112191


namespace NUMINAMATH_CALUDE_winning_strategy_l1121_112146

/-- Represents the state of the game -/
structure GameState :=
  (pieces : ℕ)

/-- Represents a valid move in the game -/
def ValidMove (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 6

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : ℕ) : GameState :=
  { pieces := state.pieces - move }

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Prop :=
  state.pieces = 0

/-- Represents a winning strategy for the first player -/
def isWinningStrategy (firstMove : ℕ) : Prop :=
  ∀ (opponentMoves : ℕ → ℕ), 
    (∀ n, ValidMove (opponentMoves n)) →
    ∃ (playerMoves : ℕ → ℕ),
      (∀ n, ValidMove (playerMoves n)) ∧
      isGameOver (applyMove (applyMove (GameState.mk 32) firstMove) (opponentMoves 0))

/-- The main theorem stating that removing 4 pieces is a winning strategy -/
theorem winning_strategy : isWinningStrategy 4 := by
  sorry

end NUMINAMATH_CALUDE_winning_strategy_l1121_112146


namespace NUMINAMATH_CALUDE_connecting_point_on_line_connecting_point_on_line_x_plus_1_connecting_point_area_and_distance_l1121_112162

-- Define the concept of a "connecting point"
def is_connecting_point (P Q : ℝ × ℝ) : Prop :=
  Q.1 = P.1 ∧ 
  ((P.1 ≥ 0 ∧ Q.2 = P.2) ∨ (P.1 < 0 ∧ Q.2 = -P.2))

-- Part 1
theorem connecting_point_on_line (k : ℝ) (A A' : ℝ × ℝ) :
  k ≠ 0 →
  A.2 = k * A.1 →
  is_connecting_point A A' →
  A' = (-2, -6) →
  k = -3 :=
sorry

-- Part 2
theorem connecting_point_on_line_x_plus_1 (m : ℝ) (B B' : ℝ × ℝ) :
  B.2 = B.1 + 1 →
  is_connecting_point B B' →
  B' = (m, 2) →
  (m ≥ 0 → B = (1, 2)) ∧
  (m < 0 → B = (-3, -2)) :=
sorry

-- Part 3
theorem connecting_point_area_and_distance (P C C' : ℝ × ℝ) :
  P = (1, 0) →
  C.2 = -2 * C.1 + 2 →
  is_connecting_point C C' →
  abs ((P.1 - C.1) * (C'.2 - C.2)) / 2 = 18 →
  Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = 3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_connecting_point_on_line_connecting_point_on_line_x_plus_1_connecting_point_area_and_distance_l1121_112162


namespace NUMINAMATH_CALUDE_no_real_solutions_l1121_112177

theorem no_real_solutions (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - 2| ≥ a) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1121_112177


namespace NUMINAMATH_CALUDE_correct_average_l1121_112160

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 46 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 65 →
  (n : ℚ) * incorrect_avg - incorrect_num + correct_num = n * 50 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l1121_112160


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l1121_112144

/-- The maximum distance from any point on the unit circle to the line x - y + 3 = 0 -/
theorem max_distance_circle_to_line : 
  ∃ (d : ℝ), d = (3 * Real.sqrt 2) / 2 + 1 ∧
  ∀ (x y : ℝ), x^2 + y^2 = 1 → 
  |x - y + 3| / Real.sqrt 2 ≤ d ∧
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 1 ∧ |x₀ - y₀ + 3| / Real.sqrt 2 = d :=
sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l1121_112144


namespace NUMINAMATH_CALUDE_perfect_cube_factors_of_4410_l1121_112138

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_cube (n : ℕ) : Prop := sorry

def count_perfect_cube_factors (n : ℕ) : ℕ := sorry

theorem perfect_cube_factors_of_4410 :
  let factorization := prime_factorization 4410
  (factorization = [(2, 1), (3, 2), (5, 1), (7, 2)]) →
  (count_perfect_cube_factors 4410 = 1) := by sorry

end NUMINAMATH_CALUDE_perfect_cube_factors_of_4410_l1121_112138


namespace NUMINAMATH_CALUDE_fgh_supermarket_count_l1121_112155

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 47

/-- The difference between the number of US and Canadian supermarkets -/
def difference : ℕ := 10

/-- The set of countries where FGH supermarkets are located -/
inductive Country
| US
| Canada

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := us_supermarkets + (us_supermarkets - difference)

theorem fgh_supermarket_count :
  total_supermarkets = 84 :=
sorry

end NUMINAMATH_CALUDE_fgh_supermarket_count_l1121_112155


namespace NUMINAMATH_CALUDE_fourth_sample_is_37_l1121_112166

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  first_sample : ℕ
  h_total : total_students > 0
  h_sample : sample_size > 0
  h_first : first_sample > 0
  h_first_le_total : first_sample ≤ total_students

/-- The sampling interval for a systematic sampling. -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- The nth sample in a systematic sampling. -/
def nth_sample (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.first_sample + (n - 1) * sampling_interval s

/-- Theorem: In a systematic sampling of 64 students with a sample size of 4,
    if the first three samples are 5, 21, and 53, then the fourth sample must be 37. -/
theorem fourth_sample_is_37 :
  ∀ (s : SystematicSampling),
    s.total_students = 64 →
    s.sample_size = 4 →
    s.first_sample = 5 →
    nth_sample s 2 = 21 →
    nth_sample s 3 = 53 →
    nth_sample s 4 = 37 := by
  sorry


end NUMINAMATH_CALUDE_fourth_sample_is_37_l1121_112166


namespace NUMINAMATH_CALUDE_seating_arrangements_l1121_112103

def total_people : ℕ := 10
def restricted_group : ℕ := 4

theorem seating_arrangements (total : ℕ) (restricted : ℕ) :
  total = total_people ∧ restricted = restricted_group →
  (total.factorial - (total - restricted + 1).factorial * restricted.factorial) = 3507840 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1121_112103


namespace NUMINAMATH_CALUDE_impossibleIdenticalLongNumbers_l1121_112171

/-- Represents a long number formed by concatenating integers -/
def LongNumber := List Nat

/-- Checks if a number is in the valid range [0, 999] -/
def isValidNumber (n : Nat) : Prop := n ≤ 999

/-- Splits a list of numbers into two groups -/
def split (numbers : List Nat) : Prop := ∃ (group1 group2 : List Nat), 
  (group1 ++ group2).Perm numbers ∧ group1 ≠ [] ∧ group2 ≠ []

theorem impossibleIdenticalLongNumbers : 
  ¬∃ (numbers : List Nat), 
    (∀ n ∈ numbers, isValidNumber n) ∧ 
    (∀ n, isValidNumber n → n ∈ numbers) ∧
    (∃ (group1 group2 : LongNumber), 
      split numbers ∧ 
      group1.toString = group2.toString) := by
  sorry

end NUMINAMATH_CALUDE_impossibleIdenticalLongNumbers_l1121_112171


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1121_112194

theorem quadratic_real_roots (k : ℕ) : 
  (∃ x : ℝ, k * x^2 - 3 * x + 2 = 0) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1121_112194


namespace NUMINAMATH_CALUDE_tshirt_profit_calculation_l1121_112168

-- Define the profit per jersey
def profit_per_jersey : ℕ := 5

-- Define the profit per t-shirt
def profit_per_tshirt : ℕ := 215

-- Define the number of t-shirts sold
def tshirts_sold : ℕ := 20

-- Define the number of jerseys sold
def jerseys_sold : ℕ := 64

-- Theorem to prove
theorem tshirt_profit_calculation :
  tshirts_sold * profit_per_tshirt = 4300 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_profit_calculation_l1121_112168


namespace NUMINAMATH_CALUDE_data_median_and_mode_l1121_112190

def data : List ℕ := [15, 17, 14, 10, 15, 17, 17, 16, 14, 12]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem data_median_and_mode :
  median data = 14.5 ∧ mode data = 17 := by sorry

end NUMINAMATH_CALUDE_data_median_and_mode_l1121_112190


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1121_112131

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 → 
  (∃ r : ℝ, 160 * r = b ∧ b * r = 108 / 64) → 
  b = 15 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1121_112131


namespace NUMINAMATH_CALUDE_classroom_boys_count_l1121_112179

theorem classroom_boys_count (initial_girls : ℕ) : 
  let initial_boys := initial_girls + 5
  let final_girls := initial_girls + 10
  let final_boys := initial_boys + 3
  final_girls = 22 →
  final_boys = 20 := by
sorry

end NUMINAMATH_CALUDE_classroom_boys_count_l1121_112179


namespace NUMINAMATH_CALUDE_problem_proof_l1121_112124

theorem problem_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x*y - 3 = 0) :
  (0 < x*y ∧ x*y ≤ 1) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b + a*b - 3 = 0 → x + 2*y ≤ a + 2*b) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b + a*b - 3 = 0 ∧ a + 2*b = 4*Real.sqrt 2 - 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_proof_l1121_112124


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1121_112140

theorem cubic_roots_sum (x y z : ℝ) : 
  (x^3 - 2*x^2 - 9*x - 1 = 0) →
  (y^3 - 2*y^2 - 9*y - 1 = 0) →
  (z^3 - 2*z^2 - 9*z - 1 = 0) →
  (y*z/x + x*z/y + x*y/z = 77) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1121_112140


namespace NUMINAMATH_CALUDE_library_books_end_of_month_l1121_112154

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) 
  (h1 : initial_books = 300)
  (h2 : loaned_books = 160)
  (h3 : return_rate = 65 / 100) :
  initial_books - loaned_books + (return_rate * loaned_books).floor = 244 :=
by sorry

end NUMINAMATH_CALUDE_library_books_end_of_month_l1121_112154


namespace NUMINAMATH_CALUDE_total_weight_on_scale_l1121_112148

theorem total_weight_on_scale (alexa_weight katerina_weight : ℕ) 
  (h1 : alexa_weight = 46)
  (h2 : katerina_weight = 49) :
  alexa_weight + katerina_weight = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_on_scale_l1121_112148


namespace NUMINAMATH_CALUDE_equation_D_is_quadratic_l1121_112198

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 5x = 0 -/
def equation_D (x : ℝ) : ℝ := x^2 - 5*x

/-- Theorem: equation_D is a quadratic equation -/
theorem equation_D_is_quadratic : is_quadratic_equation equation_D := by
  sorry

end NUMINAMATH_CALUDE_equation_D_is_quadratic_l1121_112198


namespace NUMINAMATH_CALUDE_max_binomial_coeff_expansion_l1121_112104

theorem max_binomial_coeff_expansion (m : ℕ) : 
  (∀ x : ℝ, x > 0 → (5 / Real.sqrt x - x)^m = 256) → 
  (∃ k : ℕ, k ≤ m ∧ Nat.choose m k = 6 ∧ ∀ j : ℕ, j ≤ m → Nat.choose m j ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_max_binomial_coeff_expansion_l1121_112104


namespace NUMINAMATH_CALUDE_phoenix_equation_equal_roots_l1121_112117

theorem phoenix_equation_equal_roots (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : a + b + c = 0) (h3 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ 
  ∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x) : a = c :=
sorry

end NUMINAMATH_CALUDE_phoenix_equation_equal_roots_l1121_112117


namespace NUMINAMATH_CALUDE_julians_debt_l1121_112145

/-- The amount Julian owes Jenny after borrowing additional money -/
def total_debt (initial_debt : ℕ) (borrowed_amount : ℕ) : ℕ :=
  initial_debt + borrowed_amount

/-- Theorem stating that Julian's total debt is 28 dollars -/
theorem julians_debt : total_debt 20 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_julians_debt_l1121_112145


namespace NUMINAMATH_CALUDE_find_a_l1121_112126

theorem find_a : ∃ a : ℕ, 
  (∀ k : ℤ, k ≠ 27 → ∃ m : ℤ, a - k^1964 = m * (27 - k)) → 
  a = 27^1964 := by
sorry

end NUMINAMATH_CALUDE_find_a_l1121_112126


namespace NUMINAMATH_CALUDE_binomial_coefficient_bounds_l1121_112165

theorem binomial_coefficient_bounds (n : ℕ) : 2^n ≤ Nat.choose (2*n) n ∧ Nat.choose (2*n) n ≤ 2^(2*n) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_bounds_l1121_112165


namespace NUMINAMATH_CALUDE_truck_distance_on_rough_terrain_truck_travel_distance_l1121_112134

/-- Calculates the distance a truck can travel on rough terrain given its performance on a smooth highway and the efficiency decrease on rough terrain. -/
theorem truck_distance_on_rough_terrain 
  (highway_distance : ℝ) 
  (highway_gas : ℝ) 
  (rough_terrain_efficiency_decrease : ℝ) 
  (rough_terrain_gas : ℝ) : ℝ :=
  let highway_efficiency := highway_distance / highway_gas
  let rough_terrain_efficiency := highway_efficiency * (1 - rough_terrain_efficiency_decrease)
  rough_terrain_efficiency * rough_terrain_gas

/-- Proves that a truck traveling 300 miles on 10 gallons of gas on a smooth highway can travel 405 miles on 15 gallons of gas on rough terrain with a 10% efficiency decrease. -/
theorem truck_travel_distance : 
  truck_distance_on_rough_terrain 300 10 0.1 15 = 405 := by
  sorry

end NUMINAMATH_CALUDE_truck_distance_on_rough_terrain_truck_travel_distance_l1121_112134


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l1121_112180

theorem gcd_of_squares_sum : Nat.gcd (122^2 + 234^2 + 344^2) (123^2 + 235^2 + 343^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l1121_112180


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1121_112161

/-- An increasing arithmetic sequence of integers -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_product (a : ℕ → ℤ) :
  ArithmeticSequence a → a 4 * a 5 = 24 → a 3 * a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1121_112161


namespace NUMINAMATH_CALUDE_post_height_l1121_112192

/-- The height of a cylindrical post given a squirrel's spiral path --/
theorem post_height (post_circumference : ℝ) (travel_distance : ℝ) (rise_per_circuit : ℝ) 
  (h1 : post_circumference = 3)
  (h2 : travel_distance = 27)
  (h3 : rise_per_circuit = 3) :
  travel_distance = travel_distance / post_circumference * rise_per_circuit :=
by
  sorry

#check post_height

end NUMINAMATH_CALUDE_post_height_l1121_112192


namespace NUMINAMATH_CALUDE_second_quadrant_m_range_l1121_112185

theorem second_quadrant_m_range (m : ℝ) : 
  (m^2 - 1 < 0 ∧ m > 0) → (0 < m ∧ m < 1) := by sorry

end NUMINAMATH_CALUDE_second_quadrant_m_range_l1121_112185


namespace NUMINAMATH_CALUDE_restaurant_group_size_l1121_112176

theorem restaurant_group_size :
  ∀ (adult_meal_cost : ℕ) (num_kids : ℕ) (total_cost : ℕ),
    adult_meal_cost = 3 →
    num_kids = 7 →
    total_cost = 15 →
    ∃ (num_adults : ℕ),
      num_adults * adult_meal_cost = total_cost ∧
      num_adults + num_kids = 12 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_group_size_l1121_112176


namespace NUMINAMATH_CALUDE_plane_through_point_parallel_to_plane_l1121_112109

/-- A plane in 3D space --/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_pos : a > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

/-- A point in 3D space --/
structure Point where
  x : ℤ
  y : ℤ
  z : ℤ

def Plane.contains (p : Plane) (pt : Point) : Prop :=
  p.a * pt.x + p.b * pt.y + p.c * pt.z + p.d = 0

def Plane.isParallelTo (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

theorem plane_through_point_parallel_to_plane 
  (given_plane : Plane) 
  (point : Point) :
  ∃ (result_plane : Plane), 
    result_plane.contains point ∧ 
    result_plane.isParallelTo given_plane ∧
    result_plane.a = 3 ∧ 
    result_plane.b = -2 ∧ 
    result_plane.c = 4 ∧ 
    result_plane.d = -19 := by
  sorry

end NUMINAMATH_CALUDE_plane_through_point_parallel_to_plane_l1121_112109


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l1121_112106

/-- Prove that the given vectors are not coplanar -/
theorem vectors_not_coplanar (a b c : ℝ × ℝ × ℝ) :
  a = (-7, 10, -5) →
  b = (0, -2, -1) →
  c = (-2, 4, -1) →
  ¬(∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l1121_112106


namespace NUMINAMATH_CALUDE_probability_inner_circle_l1121_112151

/-- The probability of a random point from a circle with radius 3 falling within a concentric circle with radius 1.5 -/
theorem probability_inner_circle (outer_radius inner_radius : ℝ) 
  (h_outer : outer_radius = 3)
  (h_inner : inner_radius = 1.5) :
  (π * inner_radius^2) / (π * outer_radius^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_inner_circle_l1121_112151


namespace NUMINAMATH_CALUDE_circles_intersect_l1121_112133

/-- The equation of circle C₁ -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 2 = 0

/-- The equation of circle C₂ -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The circles C₁ and C₂ intersect -/
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y :=
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l1121_112133


namespace NUMINAMATH_CALUDE_max_d_value_l1121_112101

/-- The sequence term for a given n -/
def a (n : ℕ) : ℕ := 100 + n^2

/-- The greatest common divisor of consecutive terms -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The theorem stating the maximum value of d_n -/
theorem max_d_value : ∃ (N : ℕ), ∀ (n : ℕ), n > 0 → d n ≤ 401 ∧ d N = 401 := by
  sorry

end NUMINAMATH_CALUDE_max_d_value_l1121_112101


namespace NUMINAMATH_CALUDE_analysis_time_l1121_112188

/-- The number of bones in the human body -/
def num_bones : ℕ := 206

/-- The time in minutes spent analyzing each bone -/
def minutes_per_bone : ℕ := 45

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The total time in hours required to analyze all bones in the human body -/
theorem analysis_time : (num_bones * minutes_per_bone : ℚ) / minutes_per_hour = 154.5 := by
  sorry

end NUMINAMATH_CALUDE_analysis_time_l1121_112188


namespace NUMINAMATH_CALUDE_orchid_rose_difference_l1121_112110

theorem orchid_rose_difference (initial_roses initial_orchids final_roses final_orchids : ℕ) :
  initial_roses = 7 →
  initial_orchids = 12 →
  final_roses = 11 →
  final_orchids = 20 →
  final_orchids - final_roses = 9 := by
sorry

end NUMINAMATH_CALUDE_orchid_rose_difference_l1121_112110


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1121_112118

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1121_112118


namespace NUMINAMATH_CALUDE_exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l1121_112107

/-- The measure of the exterior angle DEF in a configuration where a regular pentagon
    and a regular octagon share a side. -/
theorem exterior_angle_pentagon_octagon : ℝ :=
  let pentagon_interior_angle : ℝ := 180 * (5 - 2) / 5
  let octagon_interior_angle : ℝ := 180 * (8 - 2) / 8
  let sum_of_angles_at_E : ℝ := 360
  117

/-- Proof that the exterior angle DEF measures 117° when a regular pentagon ABCDE
    and a regular octagon AEFGHIJK share a side AE in a plane. -/
theorem exterior_angle_pentagon_octagon_is_117 :
  exterior_angle_pentagon_octagon = 117 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l1121_112107


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l1121_112121

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem odd_periodic_function_property
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : has_period f (π / 2))
  (h_value : f (π / 3) = 1) :
  f (-5 * π / 6) = -1 :=
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l1121_112121


namespace NUMINAMATH_CALUDE_hotel_flat_fee_l1121_112163

/-- Given a hotel's pricing structure and two customers' stays, calculate the flat fee for the first night. -/
theorem hotel_flat_fee (linda_total linda_nights bob_total bob_nights : ℕ) 
  (h1 : linda_total = 205)
  (h2 : linda_nights = 4)
  (h3 : bob_total = 350)
  (h4 : bob_nights = 7) :
  ∃ (flat_fee nightly_rate : ℕ),
    flat_fee + (linda_nights - 1) * nightly_rate = linda_total ∧
    flat_fee + (bob_nights - 1) * nightly_rate = bob_total ∧
    flat_fee = 60 := by
  sorry

#check hotel_flat_fee

end NUMINAMATH_CALUDE_hotel_flat_fee_l1121_112163


namespace NUMINAMATH_CALUDE_star_properties_l1121_112169

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 2) * (y + 1) - 3

-- Theorem statement
theorem star_properties :
  (¬ ∀ x y : ℝ, star x y = star y x) ∧ 
  (¬ ∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x) ∧ 
  (star 0 1 = 1) := by
  sorry


end NUMINAMATH_CALUDE_star_properties_l1121_112169


namespace NUMINAMATH_CALUDE_digit_puzzle_l1121_112164

def is_not_zero (d : Nat) : Prop := d ≠ 0
def is_even (d : Nat) : Prop := d % 2 = 0
def is_five (d : Nat) : Prop := d = 5
def is_not_six (d : Nat) : Prop := d ≠ 6
def is_less_than_seven (d : Nat) : Prop := d < 7

theorem digit_puzzle (d : Nat) 
  (h_range : d ≤ 9)
  (h_statements : ∃! (s : Fin 5), ¬(
    match s with
    | 0 => is_not_zero d
    | 1 => is_even d
    | 2 => is_five d
    | 3 => is_not_six d
    | 4 => is_less_than_seven d
  )) :
  ¬(is_even d) :=
sorry

end NUMINAMATH_CALUDE_digit_puzzle_l1121_112164


namespace NUMINAMATH_CALUDE_unique_solution_cos_arctan_sin_arccos_l1121_112130

theorem unique_solution_cos_arctan_sin_arccos (z : ℝ) :
  (∃! z : ℝ, 0 ≤ z ∧ z ≤ 1 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos z))) = z) ∧
  (Real.cos (Real.arctan (Real.sin (Real.arccos (Real.sqrt 2 / 2)))) = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cos_arctan_sin_arccos_l1121_112130


namespace NUMINAMATH_CALUDE_basketball_team_score_l1121_112172

theorem basketball_team_score :
  ∀ (tobee jay sean remy alex : ℕ),
  tobee = 4 →
  jay = 2 * tobee + 6 →
  sean = jay / 2 →
  remy = tobee + jay - 3 →
  alex = sean + remy + 4 →
  tobee + jay + sean + remy + alex = 66 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_team_score_l1121_112172


namespace NUMINAMATH_CALUDE_game_problem_l1121_112184

/-- Represents the game setup -/
structure GameSetup :=
  (total_boxes : ℕ)
  (valuable_boxes : ℕ)
  (prob_threshold : ℚ)

/-- Calculates the minimum number of boxes to eliminate -/
def min_boxes_to_eliminate (setup : GameSetup) : ℕ :=
  setup.total_boxes - 2 * setup.valuable_boxes

/-- Theorem statement for the game problem -/
theorem game_problem (setup : GameSetup) 
  (h1 : setup.total_boxes = 30)
  (h2 : setup.valuable_boxes = 5)
  (h3 : setup.prob_threshold = 1/2) :
  min_boxes_to_eliminate setup = 20 := by
  sorry

#eval min_boxes_to_eliminate { total_boxes := 30, valuable_boxes := 5, prob_threshold := 1/2 }

end NUMINAMATH_CALUDE_game_problem_l1121_112184


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1121_112100

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ d : ℚ) : ℕ → ℚ := λ n => a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmeticSum (a₁ d : ℚ) (n : ℕ) : ℚ := n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℚ) 
  (h_arith : ∃ (d : ℚ), ∀ (n : ℕ), a (n + 1) = a n + d) 
  (h_a₁ : a 1 = 1/2) 
  (h_S₂ : arithmeticSum (a 1) (a 2 - a 1) 2 = a 3) :
  ∀ (n : ℕ), arithmeticSum (a 1) (a 2 - a 1) n = 1/4 * n^2 + 1/4 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1121_112100


namespace NUMINAMATH_CALUDE_gcd_of_special_numbers_l1121_112113

def a : ℕ := 6666666
def b : ℕ := 999999999

theorem gcd_of_special_numbers : Nat.gcd a b = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_of_special_numbers_l1121_112113


namespace NUMINAMATH_CALUDE_football_field_fertilizer_l1121_112112

theorem football_field_fertilizer 
  (total_area : ℝ) 
  (partial_area : ℝ) 
  (partial_fertilizer : ℝ) 
  (h1 : total_area = 9600) 
  (h2 : partial_area = 5600) 
  (h3 : partial_fertilizer = 700) 
  (h4 : partial_area > 0) 
  (h5 : total_area > partial_area) :
  (total_area * partial_fertilizer) / partial_area = 1200 := by
sorry

end NUMINAMATH_CALUDE_football_field_fertilizer_l1121_112112


namespace NUMINAMATH_CALUDE_john_lift_weight_l1121_112153

/-- Calculates the final weight John can lift after training and using a magical bracer -/
def final_lift_weight (initial_weight : ℕ) (weight_increase : ℕ) (bracer_multiplier : ℕ) : ℕ :=
  let after_training := initial_weight + weight_increase
  let bracer_increase := after_training * bracer_multiplier
  after_training + bracer_increase

/-- Proves that John can lift 2800 pounds after training and using the magical bracer -/
theorem john_lift_weight :
  final_lift_weight 135 265 6 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_john_lift_weight_l1121_112153
