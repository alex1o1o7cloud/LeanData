import Mathlib

namespace ratio_equals_five_sixths_l245_24570

theorem ratio_equals_five_sixths
  (a b c x y z : ℝ)
  (sum_squares_abc : a^2 + b^2 + c^2 = 25)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 36)
  (dot_product : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
  sorry

#check ratio_equals_five_sixths

end ratio_equals_five_sixths_l245_24570


namespace girls_money_and_scarf_price_l245_24598

-- Define variables
variable (x y s m v : ℝ)

-- Define the conditions
def conditions (x y s m v : ℝ) : Prop :=
  y + 40 < s ∧ s < y + 50 ∧
  x + 30 < s ∧ s ≤ x + 40 - m ∧ m < 10 ∧
  0.8 * s ≤ x + 20 ∧ 0.8 * s ≤ y + 30 ∧
  0.8 * s - 4 = y + 20 ∧
  y < 0.6 * s - 3 ∧ 0.6 * s - 3 < y + 10 ∧
  x - 10 < 0.6 * s - 3 ∧ 0.6 * s - 3 < x ∧
  x + y - 1.2 * s = v

-- Theorem statement
theorem girls_money_and_scarf_price (x y s m v : ℝ) 
  (h : conditions x y s m v) : 
  61 ≤ x ∧ x ≤ 69 ∧ 52 ≤ y ∧ y ≤ 60 ∧ 91 ≤ s ∧ s ≤ 106 := by
  sorry

end girls_money_and_scarf_price_l245_24598


namespace sea_horse_penguin_ratio_l245_24514

/-- The number of sea horses at the zoo -/
def sea_horses : ℕ := 70

/-- The number of penguins at the zoo -/
def penguins : ℕ := sea_horses + 85

/-- The ratio of sea horses to penguins -/
def ratio : ℕ × ℕ := (14, 31)

/-- Theorem stating that the ratio of sea horses to penguins is 14:31 -/
theorem sea_horse_penguin_ratio :
  (sea_horses : ℚ) / (penguins : ℚ) = (ratio.1 : ℚ) / (ratio.2 : ℚ) := by
  sorry

end sea_horse_penguin_ratio_l245_24514


namespace decimal_multiplication_meaning_l245_24583

theorem decimal_multiplication_meaning (a b : ℝ) : 
  ¬ (∀ (a b : ℝ), ∃ (n : ℕ), a * b = n * (min a b)) :=
sorry

end decimal_multiplication_meaning_l245_24583


namespace floor_ceiling_sum_l245_24551

theorem floor_ceiling_sum : ⌊(-0.123 : ℝ)⌋ + ⌈(4.567 : ℝ)⌉ = 4 := by
  sorry

end floor_ceiling_sum_l245_24551


namespace square_sum_equals_sixteen_l245_24567

theorem square_sum_equals_sixteen (a b : ℝ) (h : a + b = 4) : a^2 + 2*a*b + b^2 = 16 := by
  sorry

end square_sum_equals_sixteen_l245_24567


namespace slope_of_line_l245_24555

/-- The slope of a line given by the equation √3x - y + 1 = 0 is √3. -/
theorem slope_of_line (x y : ℝ) : 
  (Real.sqrt 3) * x - y + 1 = 0 → 
  ∃ m : ℝ, m = Real.sqrt 3 ∧ y = m * x + 1 := by
sorry

end slope_of_line_l245_24555


namespace multiples_of_three_is_closed_set_l245_24504

-- Define a closed set
def is_closed_set (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

-- Define the set A
def A : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

-- Theorem statement
theorem multiples_of_three_is_closed_set : is_closed_set A := by
  sorry

end multiples_of_three_is_closed_set_l245_24504


namespace car_distance_calculation_l245_24528

/-- The distance a car needs to cover given initial time and new speed requirements -/
theorem car_distance_calculation (initial_time : ℝ) (new_speed : ℝ) : 
  initial_time = 6 →
  new_speed = 36 →
  (new_speed * (3/2 * initial_time)) = 324 := by
  sorry

end car_distance_calculation_l245_24528


namespace train_bridge_crossing_time_l245_24534

/-- Proves that a train 165 meters long, running at 54 kmph, takes 59 seconds to cross a bridge 720 meters in length. -/
theorem train_bridge_crossing_time :
  let train_length : ℝ := 165
  let bridge_length : ℝ := 720
  let train_speed_kmph : ℝ := 54
  let train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
  let total_distance : ℝ := train_length + bridge_length
  let crossing_time : ℝ := total_distance / train_speed_mps
  crossing_time = 59 := by sorry

end train_bridge_crossing_time_l245_24534


namespace intersection_P_complement_M_l245_24573

-- Define the universal set U as the set of integers
def U : Set Int := Set.univ

-- Define set M
def M : Set Int := {1, 2}

-- Define set P
def P : Set Int := {-2, -1, 0, 1, 2}

-- Theorem statement
theorem intersection_P_complement_M : 
  P ∩ (U \ M) = {-2, -1, 0} := by sorry

end intersection_P_complement_M_l245_24573


namespace geometric_sequence_formula_and_sum_l245_24589

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_formula_and_sum 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a →
  a 1 + a 2 = 9 →
  a 2 + a 3 = 18 →
  (∀ n, b n = a n + 2 * n) →
  (∀ n, a n = 3 * 2^(n - 1)) ∧ 
  (∀ n, S n = 3 * 2^n + n * (n + 1) - 3) :=
by sorry

end geometric_sequence_formula_and_sum_l245_24589


namespace least_addition_for_divisibility_l245_24560

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 1057 ∧ m = 23) : 
  ∃ x : ℕ, x = 1 ∧ 
  (∀ y : ℕ, (n + y) % m = 0 → y ≥ x) ∧
  (n + x) % m = 0 :=
sorry

end least_addition_for_divisibility_l245_24560


namespace book_arrangement_count_l245_24590

/-- Number of ways to arrange books with specific conditions -/
def arrange_books (math_books : ℕ) (history_books : ℕ) : ℕ :=
  let total_books := math_books + history_books
  let middle_slots := total_books - 2
  let unrestricted_arrangements := Nat.factorial middle_slots
  let adjacent_arrangements := Nat.factorial (middle_slots - 1) * 2
  (math_books * (math_books - 1)) * (unrestricted_arrangements - adjacent_arrangements)

/-- Theorem stating the number of ways to arrange books under given conditions -/
theorem book_arrangement_count :
  arrange_books 4 6 = 362880 :=
by sorry

end book_arrangement_count_l245_24590


namespace halfway_point_fractions_l245_24571

theorem halfway_point_fractions (a b : ℚ) (ha : a = 1/12) (hb : b = 13/12) :
  (a + b) / 2 = 7/12 := by
  sorry

end halfway_point_fractions_l245_24571


namespace stratified_sampling_theorem_l245_24554

/-- Represents the total number of students in a school -/
def total_students : ℕ := 3500 + 1500

/-- Represents the number of middle school students -/
def middle_school_students : ℕ := 1500

/-- Represents the number of students sampled from the middle school stratum -/
def middle_school_sample : ℕ := 30

/-- Calculates the total sample size in a stratified sampling -/
def total_sample_size : ℕ := (middle_school_sample * total_students) / middle_school_students

theorem stratified_sampling_theorem :
  total_sample_size = 100 := by sorry

end stratified_sampling_theorem_l245_24554


namespace flu_infection_rate_l245_24529

/-- The average number of people infected by one person in each round -/
def average_infections : ℝ := 4

/-- The number of people initially infected -/
def initial_infected : ℕ := 2

/-- The total number of people infected after two rounds -/
def total_infected : ℕ := 50

theorem flu_infection_rate :
  initial_infected +
  initial_infected * average_infections +
  (initial_infected + initial_infected * average_infections) * average_infections =
  total_infected :=
sorry

end flu_infection_rate_l245_24529


namespace alpha_beta_sum_l245_24526

theorem alpha_beta_sum (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α - 4 = 0)
  (h2 : β^3 - 3*β^2 + 5*β - 2 = 0) : 
  α + β = 2 := by
  sorry

end alpha_beta_sum_l245_24526


namespace rationalize_sqrt_five_twelfths_l245_24557

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end rationalize_sqrt_five_twelfths_l245_24557


namespace sin_five_alpha_identity_l245_24584

theorem sin_five_alpha_identity (α : ℝ) : 
  16 * (Real.sin α)^5 - 20 * (Real.sin α)^3 + 5 * Real.sin α = Real.sin (5 * α) := by
  sorry

end sin_five_alpha_identity_l245_24584


namespace simplify_expression_l245_24517

theorem simplify_expression (a : ℝ) : 3 * a^2 - a * (2 * a - 1) = a^2 + a := by
  sorry

end simplify_expression_l245_24517


namespace bicycle_separation_l245_24525

/-- Adam's speed in miles per hour -/
def adam_speed : ℝ := 12

/-- Simon's speed in miles per hour -/
def simon_speed : ℝ := 16

/-- Time in hours after which Adam and Simon are 100 miles apart -/
def separation_time : ℝ := 5

/-- Distance between Adam and Simon after separation_time hours -/
def separation_distance : ℝ := 100

theorem bicycle_separation :
  let adam_distance := adam_speed * separation_time
  let simon_distance := simon_speed * separation_time
  (adam_distance ^ 2 + simon_distance ^ 2 : ℝ) = separation_distance ^ 2 := by
  sorry

end bicycle_separation_l245_24525


namespace triangle_angle_sum_l245_24536

theorem triangle_angle_sum (X Y Z : ℝ) (h1 : X + Y = 80) (h2 : X + Y + Z = 180) : Z = 100 := by
  sorry

end triangle_angle_sum_l245_24536


namespace tank_length_calculation_l245_24580

/-- Calculates the length of a tank given its dimensions and plastering costs. -/
theorem tank_length_calculation (width depth cost_per_sqm total_cost : ℝ) 
  (h_width : width = 12)
  (h_depth : depth = 6)
  (h_cost_per_sqm : cost_per_sqm = 0.75)
  (h_total_cost : total_cost = 558) :
  ∃ length : ℝ, length = 25 ∧ 
  total_cost = (2 * (length * depth) + 2 * (width * depth) + (length * width)) * cost_per_sqm :=
by sorry

end tank_length_calculation_l245_24580


namespace compressor_stations_theorem_l245_24595

/-- Represents the configuration of three compressor stations -/
structure CompressorStations where
  x : ℝ  -- Distance between first and second stations
  y : ℝ  -- Distance between second and third stations
  z : ℝ  -- Distance between first and third stations
  a : ℝ  -- Additional parameter

/-- The conditions for the compressor stations configuration -/
def validConfiguration (s : CompressorStations) : Prop :=
  s.x > 0 ∧ s.y > 0 ∧ s.z > 0 ∧  -- Positive distances
  s.x + s.y = 2 * s.z ∧          -- Condition 1
  s.x + s.z = s.y + s.a ∧        -- Condition 2
  s.x + s.z = 75                 -- Condition 3

/-- The theorem stating the properties of the compressor stations configuration -/
theorem compressor_stations_theorem (s : CompressorStations) 
  (h : validConfiguration s) : 
  0 < s.a ∧ s.a < 100 ∧ 
  (s.a = 15 → s.x = 42 ∧ s.y = 24 ∧ s.z = 33) := by
  sorry

#check compressor_stations_theorem

end compressor_stations_theorem_l245_24595


namespace exponential_decreasing_range_l245_24574

/-- Given a function f(x) = a^x where a > 0 and a ≠ 1, 
    if f(m) < f(n) for all m > n, then 0 < a < 1 -/
theorem exponential_decreasing_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ m n : ℝ, m > n → a^m < a^n) → 0 < a ∧ a < 1 := by
  sorry

end exponential_decreasing_range_l245_24574


namespace equation_solution_l245_24545

theorem equation_solution : ∃! y : ℚ, 4 * (5 * y + 3) - 3 = -3 * (2 - 8 * y) ∧ y = 15 / 4 := by
  sorry

end equation_solution_l245_24545


namespace sum_of_squares_of_coefficients_l245_24516

-- Define the polynomial
def p (x : ℝ) : ℝ := 6 * (x^5 + 2*x^3 + x^2 + 3)

-- Define a function to get the coefficients of the expanded polynomial
def coefficients (p : ℝ → ℝ) : List ℝ := sorry

-- Define a function to calculate the sum of squares of a list of numbers
def sum_of_squares (l : List ℝ) : ℝ := sorry

-- Theorem statement
theorem sum_of_squares_of_coefficients :
  sum_of_squares (coefficients p) = 540 := by sorry

end sum_of_squares_of_coefficients_l245_24516


namespace loan_principal_calculation_l245_24521

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest : ℚ) * 100 / (rate * time)

/-- Theorem: Given a loan with 12% p.a. simple interest rate, if the interest
    amount after 10 years is Rs. 1500, then the principal amount borrowed was Rs. 1250. -/
theorem loan_principal_calculation (rate : ℚ) (time : ℕ) (interest : ℕ) :
  rate = 12 → time = 10 → interest = 1500 →
  calculate_principal rate time interest = 1250 := by
  sorry

#eval calculate_principal 12 10 1500

end loan_principal_calculation_l245_24521


namespace incorrect_observation_value_l245_24572

/-- Given a set of observations with known properties, calculate the incorrect value. -/
theorem incorrect_observation_value 
  (n : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (correct_value : ℝ) 
  (h1 : n = 50) 
  (h2 : original_mean = 36) 
  (h3 : new_mean = 36.5) 
  (h4 : correct_value = 34) : 
  ∃ (incorrect_value : ℝ), 
    incorrect_value = n * new_mean - (n - 1) * original_mean - correct_value + n * (new_mean - original_mean) :=
by
  sorry

#check incorrect_observation_value

end incorrect_observation_value_l245_24572


namespace theme_park_youngest_child_age_l245_24576

theorem theme_park_youngest_child_age (father_charge : ℝ) (age_cost : ℝ) (total_cost : ℝ) :
  father_charge = 6.5 →
  age_cost = 0.55 →
  total_cost = 15.95 →
  ∃ (twin_age : ℕ) (youngest_age : ℕ),
    youngest_age < twin_age ∧
    youngest_age + 4 * twin_age = 17 ∧
    (youngest_age = 1 ∨ youngest_age = 5) :=
by sorry

end theme_park_youngest_child_age_l245_24576


namespace distribute_five_balls_two_boxes_l245_24502

/-- The number of ways to distribute n indistinguishable balls into 2 indistinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ :=
  (n + 1) / 2 + 1

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_five_balls_two_boxes : distribute_balls 5 = 3 := by
  sorry

end distribute_five_balls_two_boxes_l245_24502


namespace hexagonal_prism_square_pyramid_edge_lengths_l245_24518

/-- Represents a regular hexagonal prism -/
structure HexagonalPrism where
  edge_length : ℝ
  total_edge_length : ℝ
  edge_count : ℕ := 18
  h_total_edge : total_edge_length = edge_length * edge_count

/-- Represents a square pyramid -/
structure SquarePyramid where
  edge_length : ℝ
  total_edge_length : ℝ
  edge_count : ℕ := 8
  h_total_edge : total_edge_length = edge_length * edge_count

/-- Theorem stating the relationship between the total edge lengths of a hexagonal prism and a square pyramid with the same edge length -/
theorem hexagonal_prism_square_pyramid_edge_lengths 
  (h : HexagonalPrism) (p : SquarePyramid) 
  (h_same_edge : h.edge_length = p.edge_length) 
  (h_total_81 : h.total_edge_length = 81) : 
  p.total_edge_length = 36 := by
  sorry

end hexagonal_prism_square_pyramid_edge_lengths_l245_24518


namespace adams_sandwiches_l245_24548

/-- The number of sandwiches Adam bought -/
def num_sandwiches : ℕ := 3

/-- The cost of each sandwich in dollars -/
def sandwich_cost : ℕ := 3

/-- The cost of the water bottle in dollars -/
def water_cost : ℕ := 2

/-- The total cost of Adam's shopping in dollars -/
def total_cost : ℕ := 11

theorem adams_sandwiches :
  num_sandwiches * sandwich_cost + water_cost = total_cost :=
by sorry

end adams_sandwiches_l245_24548


namespace right_triangle_cone_properties_l245_24593

/-- Given a right triangle with legs a and b, if rotating about leg a produces a cone
    with volume 500π cm³ and rotating about leg b produces a cone with volume 1800π cm³,
    then the hypotenuse length is √(a² + b²) and the surface area of the smaller cone
    is πb√(a² + b²). -/
theorem right_triangle_cone_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1/3 * π * b^2 * a = 500 * π) →
  (1/3 * π * a^2 * b = 1800 * π) →
  ∃ (hypotenuse surface_area : ℝ),
    hypotenuse = Real.sqrt (a^2 + b^2) ∧
    surface_area = π * min a b * Real.sqrt (a^2 + b^2) := by
  sorry

#check right_triangle_cone_properties

end right_triangle_cone_properties_l245_24593


namespace product_of_roots_l245_24510

theorem product_of_roots (x : ℝ) : 
  ((x + 3) * (x - 4) = 22) → 
  (∃ y : ℝ, ((y + 3) * (y - 4) = 22) ∧ (x * y = -34)) := by
sorry

end product_of_roots_l245_24510


namespace means_of_reciprocals_of_first_four_primes_l245_24544

def first_four_primes : List Nat := [2, 3, 5, 7]

def reciprocals (lst : List Nat) : List Rat :=
  lst.map (λ x => 1 / x)

def arithmetic_mean (lst : List Rat) : Rat :=
  lst.sum / lst.length

def harmonic_mean (lst : List Rat) : Rat :=
  lst.length / (lst.map (λ x => 1 / x)).sum

theorem means_of_reciprocals_of_first_four_primes :
  let recip := reciprocals first_four_primes
  arithmetic_mean recip = 247 / 840 ∧
  harmonic_mean recip = 4 / 17 := by
  sorry

#eval arithmetic_mean (reciprocals first_four_primes)
#eval harmonic_mean (reciprocals first_four_primes)

end means_of_reciprocals_of_first_four_primes_l245_24544


namespace circle_line_intersection_range_l245_24541

/-- Given a line and a circle with common points, prove the range of the circle's center x-coordinate. -/
theorem circle_line_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) → 
  -3 ≤ a ∧ a ≤ 1 :=
by sorry

end circle_line_intersection_range_l245_24541


namespace dog_shampoo_count_l245_24556

def clean_time : ℕ := 55
def hose_time : ℕ := 10
def shampoo_time : ℕ := 15

theorem dog_shampoo_count : 
  ∃ n : ℕ, n * shampoo_time + hose_time = clean_time ∧ n = 3 :=
by sorry

end dog_shampoo_count_l245_24556


namespace min_value_of_f_l245_24527

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 4*x + 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem min_value_of_f : ∃ (x : ℝ), x ∈ interval ∧ f x = 0 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x := by
  sorry

end min_value_of_f_l245_24527


namespace square_field_perimeter_l245_24582

theorem square_field_perimeter (a : ℝ) :
  (∃ s : ℝ, a = s^2) →  -- area is a square number
  (∃ P : ℝ, P = 36) →  -- perimeter is 36 feet
  (6 * a = 6 * (2 * 36 + 9)) →  -- given equation
  (2 * 36 = 72) :=  -- twice the perimeter is 72 feet
by
  sorry

end square_field_perimeter_l245_24582


namespace triangle_angle_ratio_l245_24530

theorem triangle_angle_ratio (A B C : ℝ) : 
  A = 60 → B = 80 → A + B + C = 180 → B / C = 2 := by
  sorry

end triangle_angle_ratio_l245_24530


namespace set_intersection_theorem_l245_24523

def A : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}
def B : Set ℝ := {x : ℝ | (x+4)*(x-2) > 0}

theorem set_intersection_theorem : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

end set_intersection_theorem_l245_24523


namespace perimeter_of_quadrilateral_l245_24597

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled (q : Quadrilateral) : Prop :=
  (q.E.1 = q.F.1 ∧ q.F.2 = q.G.2) ∧ (q.G.1 = q.H.1 ∧ q.F.2 = q.G.2)

def side_lengths (q : Quadrilateral) : ℝ × ℝ × ℝ :=
  (15, 14, 7)

-- Theorem statement
theorem perimeter_of_quadrilateral (q : Quadrilateral) 
  (h1 : is_right_angled q) 
  (h2 : side_lengths q = (15, 14, 7)) : 
  ∃ (p : ℝ), p = 36 + 2 * Real.sqrt 65 ∧ 
  p = q.E.1 - q.F.1 + q.F.2 - q.G.2 + q.G.1 - q.H.1 + Real.sqrt ((q.E.1 - q.H.1)^2 + (q.E.2 - q.H.2)^2) :=
sorry

end perimeter_of_quadrilateral_l245_24597


namespace min_value_of_expression_l245_24531

theorem min_value_of_expression (x y : ℝ) :
  Real.sqrt (2 * x^2 - 6 * x + 5) + Real.sqrt (y^2 - 4 * y + 5) + Real.sqrt (2 * x^2 - 2 * x * y + y^2) ≥ Real.sqrt 10 := by
  sorry

end min_value_of_expression_l245_24531


namespace arbelos_external_tangent_l245_24535

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an arbelos configuration -/
structure Arbelos where
  A : Point
  B : Point
  C : Point
  D : Point
  M : Point
  N : Point
  O₁ : Point
  O₂ : Point
  smallCircle1 : Circle
  smallCircle2 : Circle
  largeCircle : Circle

/-- Checks if a line is tangent to a circle -/
def isTangent (p1 p2 : Point) (c : Circle) : Prop :=
  sorry

/-- Angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Main theorem: MN is a common external tangent to the small circles of the arbelos -/
theorem arbelos_external_tangent (arb : Arbelos) (α : ℝ) 
    (h1 : angle arb.B arb.A arb.D = α)
    (h2 : arb.smallCircle1.center = arb.O₁)
    (h3 : arb.smallCircle2.center = arb.O₂) :
  isTangent arb.M arb.N arb.smallCircle1 ∧ isTangent arb.M arb.N arb.smallCircle2 :=
by sorry

end arbelos_external_tangent_l245_24535


namespace favorite_numbers_l245_24561

def is_favorite_number (n : ℕ) : Prop :=
  100 < n ∧ n < 150 ∧
  n % 13 = 0 ∧
  n % 3 ≠ 0 ∧
  (n / 100 + (n / 10) % 10 + n % 10) % 4 = 0

theorem favorite_numbers :
  ∀ n : ℕ, is_favorite_number n ↔ n = 130 ∨ n = 143 :=
by sorry

end favorite_numbers_l245_24561


namespace zero_in_interval_l245_24513

theorem zero_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ Real.log c - 6 + 2 * c = 0 := by
  sorry

end zero_in_interval_l245_24513


namespace product_without_x3_x2_terms_l245_24547

theorem product_without_x3_x2_terms (m n : ℝ) : 
  (∀ x : ℝ, (x^2 + m*x) * (x^2 - 2*x + n) = x^4 + m*n*x) → 
  m = 2 ∧ n = 4 := by
sorry

end product_without_x3_x2_terms_l245_24547


namespace expression_values_l245_24581

theorem expression_values (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  let expr := p / |p| + q / |q| + r / |r| + s / |s| + (p * q * r) / |p * q * r| + (p * r * s) / |p * r * s|
  expr = 6 ∨ expr = 2 ∨ expr = 0 ∨ expr = -2 ∨ expr = -6 := by
  sorry

end expression_values_l245_24581


namespace west_representation_l245_24501

-- Define a type for direction
inductive Direction
  | East
  | West

-- Define a function to represent distance with direction
def representDistance (dir : Direction) (distance : ℝ) : ℝ :=
  match dir with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_representation :
  representDistance Direction.East 80 = 80 →
  representDistance Direction.West 200 = -200 :=
by sorry

end west_representation_l245_24501


namespace quadrilateral_properties_l245_24550

/-- Definition of a quadrilateral with given vertices -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Function to calculate the intersection point of diagonals -/
def diagonalIntersection (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Function to calculate the area of a quadrilateral -/
def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating the properties of the given quadrilateral -/
theorem quadrilateral_properties :
  let q := Quadrilateral.mk (5, 6) (-1, 2) (-2, -1) (4, -5)
  diagonalIntersection q = (-1/6, 5/6) ∧
  quadrilateralArea q = 42 := by sorry

end quadrilateral_properties_l245_24550


namespace function_increasing_condition_l245_24552

theorem function_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 2 ∧ x₂ > 2 →
    let f := fun x => x^2 - 2*a*x + 3
    (f x₁ - f x₂) / (x₁ - x₂) > 0) →
  a ≤ 2 :=
by sorry

end function_increasing_condition_l245_24552


namespace min_value_sum_squared_ratios_l245_24599

theorem min_value_sum_squared_ratios (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^2 / y) + (y^2 / z) + (z^2 / x) ≥ 3 ∧
  ((x^2 / y) + (y^2 / z) + (z^2 / x) = 3 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end min_value_sum_squared_ratios_l245_24599


namespace carries_mom_payment_ratio_l245_24515

/-- The ratio of the amount Carrie's mom pays to the total cost of all clothes -/
theorem carries_mom_payment_ratio :
  let shirt_count : ℕ := 4
  let pants_count : ℕ := 2
  let jacket_count : ℕ := 2
  let shirt_price : ℕ := 8
  let pants_price : ℕ := 18
  let jacket_price : ℕ := 60
  let carries_payment : ℕ := 94
  let total_cost : ℕ := shirt_count * shirt_price + pants_count * pants_price + jacket_count * jacket_price
  let moms_payment : ℕ := total_cost - carries_payment
  moms_payment * 2 = total_cost :=
by sorry

end carries_mom_payment_ratio_l245_24515


namespace total_earnings_theorem_l245_24586

/-- Represents the ticket tiers --/
inductive TicketTier
  | Standard
  | Premium
  | VIP

/-- Represents the ticket types for the second show --/
inductive TicketType
  | Regular
  | Student
  | Senior

/-- Ticket prices for the first show --/
def firstShowPrice (tier : TicketTier) : ℕ :=
  match tier with
  | TicketTier.Standard => 25
  | TicketTier.Premium => 40
  | TicketTier.VIP => 60

/-- Ticket quantities for the first show --/
def firstShowQuantity (tier : TicketTier) : ℕ :=
  match tier with
  | TicketTier.Standard => 120
  | TicketTier.Premium => 60
  | TicketTier.VIP => 20

/-- Discount rates for the second show --/
def discountRate (type : TicketType) : ℚ :=
  match type with
  | TicketType.Regular => 0
  | TicketType.Student => 0.1
  | TicketType.Senior => 0.15

/-- Ticket quantities for the second show --/
def secondShowQuantity (tier : TicketTier) (type : TicketType) : ℕ :=
  match tier, type with
  | TicketTier.Standard, TicketType.Student => 240
  | TicketTier.Standard, TicketType.Senior => 120
  | TicketTier.Premium, TicketType.Student => 120
  | TicketTier.Premium, TicketType.Senior => 60
  | TicketTier.VIP, TicketType.Student => 40
  | TicketTier.VIP, TicketType.Senior => 20
  | _, TicketType.Regular => 0

/-- Calculate the earnings from the first show --/
def firstShowEarnings : ℕ :=
  (firstShowQuantity TicketTier.Standard * firstShowPrice TicketTier.Standard) +
  (firstShowQuantity TicketTier.Premium * firstShowPrice TicketTier.Premium) +
  (firstShowQuantity TicketTier.VIP * firstShowPrice TicketTier.VIP)

/-- Calculate the discounted price for the second show --/
def secondShowPrice (tier : TicketTier) (type : TicketType) : ℚ :=
  (firstShowPrice tier : ℚ) * (1 - discountRate type)

/-- Calculate the earnings from the second show --/
def secondShowEarnings : ℚ :=
  (secondShowQuantity TicketTier.Standard TicketType.Student * secondShowPrice TicketTier.Standard TicketType.Student) +
  (secondShowQuantity TicketTier.Standard TicketType.Senior * secondShowPrice TicketTier.Standard TicketType.Senior) +
  (secondShowQuantity TicketTier.Premium TicketType.Student * secondShowPrice TicketTier.Premium TicketType.Student) +
  (secondShowQuantity TicketTier.Premium TicketType.Senior * secondShowPrice TicketTier.Premium TicketType.Senior) +
  (secondShowQuantity TicketTier.VIP TicketType.Student * secondShowPrice TicketTier.VIP TicketType.Student) +
  (secondShowQuantity TicketTier.VIP TicketType.Senior * secondShowPrice TicketTier.VIP TicketType.Senior)

/-- The main theorem stating that the total earnings from both shows equal $24,090 --/
theorem total_earnings_theorem :
  (firstShowEarnings : ℚ) + secondShowEarnings = 24090 := by
  sorry

end total_earnings_theorem_l245_24586


namespace max_regular_hours_is_40_l245_24537

/-- Calculates the maximum number of regular hours worked given total pay, overtime hours, and pay rates. -/
def max_regular_hours (total_pay : ℚ) (overtime_hours : ℚ) (regular_rate : ℚ) : ℚ :=
  let overtime_rate := 2 * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let regular_pay := total_pay - overtime_pay
  regular_pay / regular_rate

/-- Proves that given the specified conditions, the maximum number of regular hours is 40. -/
theorem max_regular_hours_is_40 :
  max_regular_hours 168 8 3 = 40 := by
  sorry

#eval max_regular_hours 168 8 3

end max_regular_hours_is_40_l245_24537


namespace volume_S_polynomial_bc_over_ad_value_l245_24587

/-- A right rectangular prism with edge lengths 2, 4, and 5 -/
structure RectPrism where
  length : ℝ := 2
  width : ℝ := 4
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : RectPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume_S (B : RectPrism) (r : ℝ) : ℝ := sorry

/-- The coefficients of the volume polynomial -/
structure VolumeCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem volume_S_polynomial (B : RectPrism) :
  ∃ coeffs : VolumeCoeffs,
    ∀ r : ℝ, volume_S B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d :=
  sorry

theorem bc_over_ad_value (B : RectPrism) (coeffs : VolumeCoeffs)
  (h : ∀ r : ℝ, volume_S B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d) :
  coeffs.b * coeffs.c / (coeffs.a * coeffs.d) = 15.675 :=
  sorry

end volume_S_polynomial_bc_over_ad_value_l245_24587


namespace inequality_solution_range_l245_24588

theorem inequality_solution_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/y = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 2/y = 1 ∧ x + y/2 < m^2 + 3*m) ↔ 
  (m < -4 ∨ m > 1) :=
sorry

end inequality_solution_range_l245_24588


namespace remaining_note_denomination_l245_24591

theorem remaining_note_denomination 
  (total_amount : ℕ)
  (total_notes : ℕ)
  (fifty_notes : ℕ)
  (h1 : total_amount = 10350)
  (h2 : total_notes = 72)
  (h3 : fifty_notes = 57) :
  (total_amount - 50 * fifty_notes) / (total_notes - fifty_notes) = 500 := by
  sorry

end remaining_note_denomination_l245_24591


namespace solution_product_l245_24592

theorem solution_product (p q : ℝ) : 
  (p - 6) * (2 * p + 8) + p^2 - 15 * p + 56 = 0 →
  (q - 6) * (2 * q + 8) + q^2 - 15 * q + 56 = 0 →
  p ≠ q →
  (p + 3) * (q + 3) = 92 / 3 := by
sorry

end solution_product_l245_24592


namespace square_sum_equality_l245_24577

theorem square_sum_equality : (-2)^2 + 2^2 = 8 := by
  sorry

end square_sum_equality_l245_24577


namespace red_balls_count_l245_24566

/-- Given a bag with 15 balls of red, yellow, and blue colors, 
    if the probability of drawing two non-red balls at the same time is 2/7, 
    then the number of red balls in the bag is 5. -/
theorem red_balls_count (total : ℕ) (red : ℕ) 
  (h_total : total = 15)
  (h_prob : (total - red : ℚ) / total * ((total - 1 - red) : ℚ) / (total - 1) = 2 / 7) :
  red = 5 := by
  sorry

end red_balls_count_l245_24566


namespace girls_with_tablets_l245_24569

/-- Proves that the number of girls who brought tablets is 13 -/
theorem girls_with_tablets (total_boys : ℕ) (students_with_tablets : ℕ) (boys_with_tablets : ℕ)
  (h1 : total_boys = 20)
  (h2 : students_with_tablets = 24)
  (h3 : boys_with_tablets = 11) :
  students_with_tablets - boys_with_tablets = 13 := by
  sorry

end girls_with_tablets_l245_24569


namespace rational_power_difference_integer_implies_integer_l245_24549

/-- Given distinct positive rational numbers a and b, if a^n - b^n is a positive integer
    for infinitely many positive integers n, then a and b are positive integers. -/
theorem rational_power_difference_integer_implies_integer
  (a b : ℚ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_distinct : a ≠ b)
  (h_infinite : ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, ∃ k : ℤ, k > 0 ∧ a^n - b^n = k) :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ a = m ∧ b = n :=
sorry

end rational_power_difference_integer_implies_integer_l245_24549


namespace circle_tangent_to_line_intersection_chord_length_l245_24564

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 2

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := y = x

-- Define the intersecting line
def intersecting_line (x y : ℝ) (a : ℝ) : Prop := x - y + a = 0

-- Statement 1: Circle C is tangent to y = x
theorem circle_tangent_to_line : ∃ (x y : ℝ), circle_C x y ∧ tangent_line x y :=
sorry

-- Statement 2: Finding the value of a
theorem intersection_chord_length (a : ℝ) :
  (a ≠ 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    intersecting_line x₁ y₁ a ∧ intersecting_line x₂ y₂ a ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) →
  (a = Real.sqrt 2 - 2 ∨ a = -Real.sqrt 2 - 2) :=
sorry

end circle_tangent_to_line_intersection_chord_length_l245_24564


namespace inequality_proof_l245_24553

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 9*b*c) + b / Real.sqrt (b^2 + 9*c*a) + c / Real.sqrt (c^2 + 9*a*b) ≥ 3 / Real.sqrt 10 := by
  sorry

end inequality_proof_l245_24553


namespace fair_rides_l245_24559

theorem fair_rides (initial_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) 
  (h1 : initial_tickets = 79)
  (h2 : spent_tickets = 23)
  (h3 : tickets_per_ride = 7) :
  (initial_tickets - spent_tickets) / tickets_per_ride = 8 := by
  sorry

end fair_rides_l245_24559


namespace shortest_hexpath_distribution_l245_24542

/-- Represents a direction in the hexagonal grid -/
inductive Direction
| Horizontal
| Diagonal1
| Diagonal2

/-- Represents a path in the hexagonal grid -/
structure HexPath where
  length : ℕ
  horizontal : ℕ
  diagonal1 : ℕ
  diagonal2 : ℕ
  sum_constraint : length = horizontal + diagonal1 + diagonal2

/-- A shortest path in a hexagonal grid -/
def is_shortest_path (path : HexPath) : Prop :=
  path.horizontal = path.diagonal1 + path.diagonal2

theorem shortest_hexpath_distribution (path : HexPath) 
  (h_shortest : is_shortest_path path) (h_length : path.length = 100) :
  path.horizontal = 50 ∧ path.diagonal1 + path.diagonal2 = 50 := by
  sorry

#check shortest_hexpath_distribution

end shortest_hexpath_distribution_l245_24542


namespace no_double_apply_add_2015_l245_24511

theorem no_double_apply_add_2015 : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := by
  sorry

end no_double_apply_add_2015_l245_24511


namespace prob_at_least_one_female_l245_24519

/-- The probability of selecting at least one female student when randomly choosing 2 students
    from a group of 3 males and 1 female is equal to 1/2. -/
theorem prob_at_least_one_female (total_students : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (team_size : ℕ) (h1 : total_students = male_students + female_students) 
  (h2 : total_students = 4) (h3 : male_students = 3) (h4 : female_students = 1) (h5 : team_size = 2) :
  1 - (Nat.choose male_students team_size : ℚ) / (Nat.choose total_students team_size : ℚ) = 1/2 :=
sorry

end prob_at_least_one_female_l245_24519


namespace sum_of_fifth_powers_l245_24532

theorem sum_of_fifth_powers (a b x y : ℝ) 
  (eq1 : a*x + b*y = 3)
  (eq2 : a*x^2 + b*y^2 = 7)
  (eq3 : a*x^3 + b*y^3 = 6)
  (eq4 : a*x^4 + b*y^4 = 42) :
  a*x^5 + b*y^5 = 20 := by
  sorry

end sum_of_fifth_powers_l245_24532


namespace complement_M_intersect_N_l245_24533

-- Define the sets M and N
def M : Set ℝ := {x | 2/x < 1}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = Set.Icc 0 2 := by sorry

end complement_M_intersect_N_l245_24533


namespace president_vice_president_selection_l245_24543

theorem president_vice_president_selection (n : ℕ) (h : n = 8) : 
  (n * (n - 1) : ℕ) = 56 := by
  sorry

end president_vice_president_selection_l245_24543


namespace smallest_four_digit_multiple_of_18_proof_1008_smallest_l245_24524

theorem smallest_four_digit_multiple_of_18 : ℕ → Prop :=
  fun n => (n ≥ 1000) ∧ (n < 10000) ∧ (n % 18 = 0) ∧
    ∀ m : ℕ, (m ≥ 1000) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m

theorem proof_1008_smallest : smallest_four_digit_multiple_of_18 1008 := by
  sorry

end smallest_four_digit_multiple_of_18_proof_1008_smallest_l245_24524


namespace logarithm_properties_l245_24563

theorem logarithm_properties :
  (Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 = 1) ∧
  (Real.log 2 / Real.log 4 + 2^(Real.log 3 / Real.log 2 - 1) = 2) := by
  sorry

end logarithm_properties_l245_24563


namespace monotonic_increasing_cubic_l245_24500

/-- A cubic function with parameters m and n. -/
def f (m n : ℝ) (x : ℝ) : ℝ := 4 * x^3 + m * x^2 + (m - 3) * x + n

/-- The derivative of f with respect to x. -/
def f' (m : ℝ) (x : ℝ) : ℝ := 12 * x^2 + 2 * m * x + (m - 3)

theorem monotonic_increasing_cubic (m n : ℝ) :
  (∀ x : ℝ, Monotone (f m n)) → m = 6 := by
  sorry

end monotonic_increasing_cubic_l245_24500


namespace dvd_total_count_l245_24579

theorem dvd_total_count (store_dvds : ℕ) (online_dvds : ℕ) : 
  store_dvds = 8 → online_dvds = 2 → store_dvds + online_dvds = 10 := by
  sorry

end dvd_total_count_l245_24579


namespace isosceles_trapezoid_equal_area_l245_24575

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The length of the diagonal -/
  diagonalLength : ℝ
  /-- The angle between the diagonals -/
  diagonalAngle : ℝ
  /-- The area of the trapezoid -/
  area : ℝ

/-- 
Theorem: If two isosceles trapezoids have equal diagonal lengths and equal angles between their diagonals, 
then their areas are equal.
-/
theorem isosceles_trapezoid_equal_area 
  (t1 t2 : IsoscelesTrapezoid) 
  (h1 : t1.diagonalLength = t2.diagonalLength) 
  (h2 : t1.diagonalAngle = t2.diagonalAngle) : 
  t1.area = t2.area :=
sorry

end isosceles_trapezoid_equal_area_l245_24575


namespace batsman_average_l245_24558

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) : 
  total_innings = 17 → 
  last_innings_score = 85 → 
  average_increase = 3 → 
  (((total_innings - 1) * ((total_innings * (37 - average_increase)) / total_innings) + last_innings_score) / total_innings : ℚ) = 37 := by
sorry

end batsman_average_l245_24558


namespace industrial_machine_output_l245_24538

theorem industrial_machine_output (shirts_per_minute : ℕ) 
  (yesterday_minutes : ℕ) (today_shirts : ℕ) (total_shirts : ℕ) :
  yesterday_minutes = 12 →
  today_shirts = 14 →
  total_shirts = 156 →
  yesterday_minutes * shirts_per_minute + today_shirts = total_shirts →
  shirts_per_minute = 11 :=
by sorry

end industrial_machine_output_l245_24538


namespace ounces_per_cup_l245_24522

theorem ounces_per_cup (container_capacity : ℕ) (soap_per_cup : ℕ) (total_soap : ℕ) :
  container_capacity = 40 ∧ soap_per_cup = 3 ∧ total_soap = 15 →
  ∃ (ounces_per_cup : ℕ), ounces_per_cup = 8 ∧ container_capacity = ounces_per_cup * (total_soap / soap_per_cup) :=
by sorry

end ounces_per_cup_l245_24522


namespace cat_mouse_positions_after_360_moves_l245_24594

/-- Represents the number of squares in the cat's path -/
def cat_squares : ℕ := 5

/-- Represents the number of segments in the mouse's path -/
def mouse_segments : ℕ := 10

/-- Represents the number of segments the mouse moves per turn -/
def mouse_move_rate : ℕ := 2

/-- Represents the total number of moves -/
def total_moves : ℕ := 360

/-- Calculates the cat's position after a given number of moves -/
def cat_position (moves : ℕ) : ℕ :=
  moves % cat_squares + 1

/-- Calculates the mouse's effective moves after accounting for skipped segments -/
def mouse_effective_moves (moves : ℕ) : ℕ :=
  (moves / mouse_segments) * (mouse_segments - 1) + (moves % mouse_segments)

/-- Calculates the mouse's position after a given number of effective moves -/
def mouse_position (effective_moves : ℕ) : ℕ :=
  (effective_moves * mouse_move_rate) % mouse_segments + 1

theorem cat_mouse_positions_after_360_moves :
  cat_position total_moves = 1 ∧ 
  mouse_position (mouse_effective_moves total_moves) = 4 := by
  sorry

end cat_mouse_positions_after_360_moves_l245_24594


namespace remainder_sum_reverse_order_l245_24546

theorem remainder_sum_reverse_order (n : ℕ) : 
  n % 12 = 56 → n % 34 = 78 → (n % 34) % 12 + n % 12 = 20 := by
  sorry

end remainder_sum_reverse_order_l245_24546


namespace new_R_value_l245_24578

/-- A function that calculates R given g and S -/
def R (g : ℝ) (S : ℝ) : ℝ := g * S - 7

/-- The theorem stating the new value of R after S increases by 50% -/
theorem new_R_value (g : ℝ) (S : ℝ) (h1 : S = 5) (h2 : R g S = 8) :
  R g (S * 1.5) = 15.5 := by
  sorry


end new_R_value_l245_24578


namespace base_5_divisible_by_7_l245_24540

def base_5_to_10 (d : ℕ) : ℕ := 3 * 5^3 + d * 5^2 + d * 5 + 4

theorem base_5_divisible_by_7 :
  ∃! d : ℕ, d < 5 ∧ (base_5_to_10 d) % 7 = 0 :=
by sorry

end base_5_divisible_by_7_l245_24540


namespace even_increasing_inequality_l245_24585

/-- A function f is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is increasing on [0, +∞) if f(x) ≤ f(y) for all 0 ≤ x ≤ y -/
def IncreasingOnNonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≤ f y

theorem even_increasing_inequality (f : ℝ → ℝ) (a : ℝ) 
    (heven : EvenFunction f) (hincr : IncreasingOnNonnegatives f) :
    f (-1) < f (a^2 - 2*a + 3) := by
  sorry

end even_increasing_inequality_l245_24585


namespace inequality_proof_l245_24568

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0) 
  (h5 : x + y + z + t = 5) : 
  Real.sqrt (x^2 + y^2) + Real.sqrt (x^2 + 1) + Real.sqrt (z^2 + y^2) + 
  Real.sqrt (z^2 + t^2) + Real.sqrt (t^2 + 9) ≥ 10 := by
  sorry

end inequality_proof_l245_24568


namespace quadratic_roots_sum_and_product_l245_24507

/-- Given a quadratic equation (2√3 + √2)x² + 2(√3 + √2)x + (√2 - 2√3) = 0,
    prove that the sum of its roots is -(4 + √6)/5 and the product of its roots is (2√6 - 7)/5 -/
theorem quadratic_roots_sum_and_product :
  let a : ℝ := 2 * Real.sqrt 3 + Real.sqrt 2
  let b : ℝ := 2 * (Real.sqrt 3 + Real.sqrt 2)
  let c : ℝ := Real.sqrt 2 - 2 * Real.sqrt 3
  (-(b / a) = -(4 + Real.sqrt 6) / 5) ∧ 
  (c / a = (2 * Real.sqrt 6 - 7) / 5) := by
sorry

end quadratic_roots_sum_and_product_l245_24507


namespace line_moved_down_l245_24512

/-- Given a line with equation y = -3x + 5, prove that moving it down 3 units
    results in the line with equation y = -3x + 2 -/
theorem line_moved_down (x y : ℝ) :
  (y = -3 * x + 5) → (y - 3 = -3 * x + 2) := by sorry

end line_moved_down_l245_24512


namespace quadratic_inequality_solution_l245_24503

-- Define the quadratic function
def f (x : ℝ) := x^2 - 5*x + 6

-- Define the solution set
def solution_set := { x : ℝ | 2 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem quadratic_inequality_solution :
  { x : ℝ | f x ≤ 0 } = solution_set := by sorry

end quadratic_inequality_solution_l245_24503


namespace black_area_after_changes_l245_24506

/-- Represents the fraction of the square that is black -/
def black_fraction : ℕ → ℚ
  | 0 => 1/2  -- Initially half the square is black
  | (n+1) => (3/4) * black_fraction n  -- Each change keeps 3/4 of the previous black area

/-- The number of changes applied to the square -/
def num_changes : ℕ := 6

theorem black_area_after_changes :
  black_fraction num_changes = 729/8192 := by
  sorry

end black_area_after_changes_l245_24506


namespace max_vertex_product_sum_l245_24505

/-- The set of numbers that can be assigned to the faces of the cube -/
def CubeNumbers : Finset ℕ := {1, 2, 3, 4, 8, 9}

/-- A valid assignment of numbers to the faces of a cube -/
structure CubeAssignment where
  faces : Fin 6 → ℕ
  valid : ∀ i, faces i ∈ CubeNumbers
  distinct : ∀ i j, i ≠ j → faces i ≠ faces j

/-- The sum of products at the vertices of a cube given a face assignment -/
def vertexProductSum (assignment : CubeAssignment) : ℕ :=
  let a := assignment.faces 0
  let b := assignment.faces 1
  let c := assignment.faces 2
  let d := assignment.faces 3
  let e := assignment.faces 4
  let f := assignment.faces 5
  (a + b) * (c + d) * (e + f)

/-- The maximum sum of products at the vertices of a cube -/
theorem max_vertex_product_sum :
  ∃ (assignment : CubeAssignment), ∀ (other : CubeAssignment),
    vertexProductSum assignment ≥ vertexProductSum other ∧
    vertexProductSum assignment = 729 :=
  sorry

end max_vertex_product_sum_l245_24505


namespace quadratic_equation_equivalence_l245_24596

/-- Given a quadratic equation (x + 3)² = x(3x - 1), prove it's equivalent to 2x² - 7x - 9 = 0 in general form -/
theorem quadratic_equation_equivalence (x : ℝ) : (x + 3)^2 = x * (3*x - 1) ↔ 2*x^2 - 7*x - 9 = 0 := by
  sorry

end quadratic_equation_equivalence_l245_24596


namespace batch_size_is_84_l245_24565

/-- The number of assignments in Mr. Wang's batch -/
def total_assignments : ℕ := 84

/-- The original grading rate (assignments per hour) -/
def original_rate : ℕ := 6

/-- The new grading rate (assignments per hour) -/
def new_rate : ℕ := 8

/-- The number of hours spent grading at the original rate -/
def hours_at_original_rate : ℕ := 2

/-- The number of hours saved compared to the initial plan -/
def hours_saved : ℕ := 3

/-- Theorem stating that the total number of assignments is 84 -/
theorem batch_size_is_84 :
  total_assignments = 84 ∧
  original_rate = 6 ∧
  new_rate = 8 ∧
  hours_at_original_rate = 2 ∧
  hours_saved = 3 ∧
  (total_assignments - original_rate * hours_at_original_rate) / new_rate + hours_at_original_rate + hours_saved = total_assignments / original_rate :=
by sorry

end batch_size_is_84_l245_24565


namespace flour_measurement_l245_24520

theorem flour_measurement (required : ℚ) (container : ℚ) (excess : ℚ) : 
  required = 15/4 ∧ container = 4/3 ∧ excess = 2/3 → 
  ∃ (n : ℕ), n * container = required - excess ∧ n = 3 := by
sorry

end flour_measurement_l245_24520


namespace cameron_typing_speed_l245_24539

/-- The number of words Cameron could type per minute before breaking his arm -/
def words_per_minute : ℕ := 10

/-- The number of words Cameron could type per minute after breaking his arm -/
def words_after_injury : ℕ := 8

/-- The time period in minutes -/
def time_period : ℕ := 5

/-- The difference in words typed over the time period -/
def word_difference : ℕ := 10

theorem cameron_typing_speed :
  words_per_minute = 10 ∧
  words_after_injury = 8 ∧
  time_period = 5 ∧
  word_difference = 10 ∧
  time_period * words_per_minute - time_period * words_after_injury = word_difference :=
by sorry

end cameron_typing_speed_l245_24539


namespace remainder_sum_of_three_l245_24562

theorem remainder_sum_of_three (a b c : ℕ) : 
  a % 14 = 5 → b % 14 = 5 → c % 14 = 5 → (a + b + c) % 14 = 1 := by
  sorry

end remainder_sum_of_three_l245_24562


namespace cyclic_sum_inequality_l245_24509

open Real

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) +
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) +
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 :=
sorry

end cyclic_sum_inequality_l245_24509


namespace S_singleton_I_singleton_l245_24508

-- Define the set X
inductive X
| zero : X
| a : X
| b : X
| c : X

-- Define the addition operation on X
def add : X → X → X
| X.zero, y => y
| X.a, X.zero => X.a
| X.a, X.a => X.zero
| X.a, X.b => X.c
| X.a, X.c => X.b
| X.b, X.zero => X.b
| X.b, X.a => X.c
| X.b, X.b => X.zero
| X.b, X.c => X.a
| X.c, X.zero => X.c
| X.c, X.a => X.b
| X.c, X.b => X.a
| X.c, X.c => X.zero

-- Define the set of all functions from X to X
def M : Type := X → X

-- Define the set S
def S : Set M := {f : M | ∀ x y : X, f (add (add x y) x) = add (add (f x) (f y)) (f x)}

-- Define the set I
def I : Set M := {f : M | ∀ x : X, f (add x x) = add (f x) (f x)}

-- Theorem: S contains only one function (the zero function)
theorem S_singleton : ∃! f : M, f ∈ S := sorry

-- Theorem: I contains only one function (the zero function)
theorem I_singleton : ∃! f : M, f ∈ I := sorry

end S_singleton_I_singleton_l245_24508
