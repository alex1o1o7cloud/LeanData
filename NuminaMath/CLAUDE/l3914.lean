import Mathlib

namespace triangle_problem_l3914_391492

theorem triangle_problem (AB : ℝ) (sinA sinC : ℝ) :
  AB = 30 →
  sinA = 4/5 →
  sinC = 1/4 →
  ∃ (DC : ℝ), DC = 24 * Real.sqrt 15 :=
by
  sorry

end triangle_problem_l3914_391492


namespace mahogany_count_l3914_391476

/-- The number of initially planted Mahogany trees -/
def initial_mahogany : ℕ := sorry

/-- The number of initially planted Narra trees -/
def initial_narra : ℕ := 30

/-- The total number of trees that fell -/
def total_fallen : ℕ := 5

/-- The number of Mahogany trees that fell -/
def mahogany_fallen : ℕ := sorry

/-- The number of Narra trees that fell -/
def narra_fallen : ℕ := sorry

/-- The number of new Mahogany trees planted after the typhoon -/
def new_mahogany : ℕ := sorry

/-- The number of new Narra trees planted after the typhoon -/
def new_narra : ℕ := sorry

/-- The total number of trees after replanting -/
def total_trees : ℕ := 88

theorem mahogany_count : initial_mahogany = 50 :=
  by sorry

end mahogany_count_l3914_391476


namespace hotel_loss_calculation_l3914_391435

def hotel_loss (expenses : ℝ) (payment_ratio : ℝ) : ℝ :=
  expenses - (payment_ratio * expenses)

theorem hotel_loss_calculation (expenses : ℝ) (payment_ratio : ℝ) 
  (h1 : expenses = 100)
  (h2 : payment_ratio = 3/4) :
  hotel_loss expenses payment_ratio = 25 := by
  sorry

end hotel_loss_calculation_l3914_391435


namespace complement_A_intersection_nonempty_union_equals_B_l3914_391484

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem for the complement of A
theorem complement_A : (Set.univ \ A) = {x : ℝ | x ≤ -1 ∨ x > 2} := by sorry

-- Theorem for the range of a when A ∩ B ≠ ∅
theorem intersection_nonempty (a : ℝ) : (A ∩ B a).Nonempty → a > -1 := by sorry

-- Theorem for the range of a when A ∪ B = B
theorem union_equals_B (a : ℝ) : A ∪ B a = B a → a > 2 := by sorry

end complement_A_intersection_nonempty_union_equals_B_l3914_391484


namespace magnitude_of_difference_vector_l3914_391401

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 4)

theorem magnitude_of_difference_vector :
  let dot_product := a.1 * b.1 + a.2 * b.2
  dot_product = 10 →
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = 5 := by sorry

end magnitude_of_difference_vector_l3914_391401


namespace unique_positive_solution_l3914_391438

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) :=
by
  -- The proof goes here
  sorry

end unique_positive_solution_l3914_391438


namespace line_polar_equation_l3914_391425

-- Define the line in Cartesian coordinates
def line (x y : ℝ) : Prop := (Real.sqrt 3 / 3) * x - y = 0

-- Define the polar coordinates
def polar_coords (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ ρ ≥ 0

-- State the theorem
theorem line_polar_equation :
  ∀ ρ θ x y : ℝ,
  polar_coords ρ θ x y →
  line x y →
  (θ = π / 6 ∨ θ = 7 * π / 6) :=
sorry

end line_polar_equation_l3914_391425


namespace sum_of_three_numbers_l3914_391482

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 267)
  (sum_of_products : a*b + b*c + c*a = 131) :
  a + b + c = 23 := by
  sorry

end sum_of_three_numbers_l3914_391482


namespace odd_function_condition_l3914_391418

/-- Given a > 1, f(x) = (a^x / (a^x - 1)) + m is an odd function if and only if m = -1/2 -/
theorem odd_function_condition (a : ℝ) (h : a > 1) :
  ∃ m : ℝ, ∀ x : ℝ, x ≠ 0 →
    (fun x : ℝ => (a^x / (a^x - 1)) + m) x = -((fun x : ℝ => (a^x / (a^x - 1)) + m) (-x)) ↔
    m = -1/2 := by sorry

end odd_function_condition_l3914_391418


namespace birthday_presents_total_l3914_391494

def leonard_wallets : ℕ := 3
def leonard_wallet_price : ℕ := 35
def leonard_sneakers : ℕ := 2
def leonard_sneaker_price : ℕ := 120
def leonard_belt_price : ℕ := 45

def michael_backpack_price : ℕ := 90
def michael_jeans : ℕ := 3
def michael_jeans_price : ℕ := 55
def michael_tie_price : ℕ := 25

def emily_shirts : ℕ := 2
def emily_shirt_price : ℕ := 70
def emily_books : ℕ := 4
def emily_book_price : ℕ := 15

def total_spent : ℕ := 870

theorem birthday_presents_total :
  (leonard_wallets * leonard_wallet_price + 
   leonard_sneakers * leonard_sneaker_price + 
   leonard_belt_price) +
  (michael_backpack_price + 
   michael_jeans * michael_jeans_price + 
   michael_tie_price) +
  (emily_shirts * emily_shirt_price + 
   emily_books * emily_book_price) = total_spent := by
  sorry

end birthday_presents_total_l3914_391494


namespace power_of_three_mod_five_l3914_391493

theorem power_of_three_mod_five : 3^2023 % 5 = 2 := by sorry

end power_of_three_mod_five_l3914_391493


namespace container_capacity_sum_l3914_391416

/-- Represents the capacity and fill levels of a container -/
structure Container where
  capacity : ℝ
  initial_fill : ℝ
  final_fill : ℝ
  added_water : ℝ

/-- Calculates the total capacity of three containers -/
def total_capacity (a b c : Container) : ℝ :=
  a.capacity + b.capacity + c.capacity

/-- The problem statement -/
theorem container_capacity_sum : 
  ∃ (a b c : Container),
    a.initial_fill = 0.3 * a.capacity ∧
    a.final_fill = 0.75 * a.capacity ∧
    a.added_water = 36 ∧
    b.initial_fill = 0.4 * b.capacity ∧
    b.final_fill = 0.7 * b.capacity ∧
    b.added_water = 20 ∧
    c.initial_fill = 0.5 * c.capacity ∧
    c.final_fill = 2/3 * c.capacity ∧
    c.added_water = 12 ∧
    total_capacity a b c = 218.6666666666667 := by
  sorry

end container_capacity_sum_l3914_391416


namespace apple_division_problem_l3914_391460

/-- Calculates the minimal number of pieces needed to evenly divide apples among students -/
def minimalPieces (apples : ℕ) (students : ℕ) : ℕ :=
  let components := apples.gcd students
  let applesPerComponent := apples / components
  let studentsPerComponent := students / components
  components * (applesPerComponent + studentsPerComponent - 1)

/-- Proves that the minimal number of pieces to evenly divide 221 apples among 403 students is 611 -/
theorem apple_division_problem :
  minimalPieces 221 403 = 611 := by
  sorry

#eval minimalPieces 221 403

end apple_division_problem_l3914_391460


namespace eq2_eq3_same_graph_eq1_different_graph_l3914_391446

-- Define the three equations
def eq1 (x y : ℝ) : Prop := y = x + 3
def eq2 (x y : ℝ) : Prop := y = (x^2 - 1) / (x - 1)
def eq3 (x y : ℝ) : Prop := (x - 1) * y = x^2 - 1

-- Define the concept of having the same graph
def same_graph (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, x ≠ 1 → (f x y ↔ g x y)

-- Theorem stating that eq2 and eq3 have the same graph
theorem eq2_eq3_same_graph : same_graph eq2 eq3 := by sorry

-- Theorem stating that eq1 has a different graph from eq2 and eq3
theorem eq1_different_graph :
  ¬(same_graph eq1 eq2) ∧ ¬(same_graph eq1 eq3) := by sorry

end eq2_eq3_same_graph_eq1_different_graph_l3914_391446


namespace perimeter_of_modified_square_l3914_391422

/-- The perimeter of a figure formed by cutting an equilateral triangle from a square and
    translating it to the right side. -/
theorem perimeter_of_modified_square (square_perimeter : ℝ) (h : square_perimeter = 40) :
  let side_length := square_perimeter / 4
  let triangle_side_length := side_length
  let new_perimeter := 2 * side_length + 4 * triangle_side_length
  new_perimeter = 60 := by sorry

end perimeter_of_modified_square_l3914_391422


namespace intersection_sum_l3914_391487

/-- Two functions f and g that intersect at given points -/
def f (a b x : ℝ) : ℝ := -2 * abs (x - a) + b
def g (c d x : ℝ) : ℝ := 2 * abs (x - c) + d

/-- Theorem stating that for functions f and g intersecting at (1, 7) and (11, -1), a + c = 12 -/
theorem intersection_sum (a b c d : ℝ) 
  (h1 : f a b 1 = g c d 1 ∧ f a b 1 = 7)
  (h2 : f a b 11 = g c d 11 ∧ f a b 11 = -1) :
  a + c = 12 := by
  sorry


end intersection_sum_l3914_391487


namespace xyz_value_l3914_391471

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 8)
  (eq5 : x + y + z = 4) :
  x * y * z = 10 := by
sorry

end xyz_value_l3914_391471


namespace translation_result_l3914_391453

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, 5)
def B : ℝ × ℝ := (-3, 0)
def C : ℝ × ℝ := (3, 8)

-- Define the translation vector
def translation_vector : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define point D as the result of translating B
def D : ℝ × ℝ := (B.1 + translation_vector.1, B.2 + translation_vector.2)

-- Theorem statement
theorem translation_result : D = (2, 3) := by
  sorry

end translation_result_l3914_391453


namespace expression_simplification_l3914_391419

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 - x / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l3914_391419


namespace consecutive_primes_as_greatest_divisors_l3914_391496

theorem consecutive_primes_as_greatest_divisors (p q : ℕ) 
  (hp : Prime p) (hq : Prime q) (hpq : p < q) (hqp : q < 2 * p) :
  ∃ n : ℕ, 
    (∃ k : ℕ+, n = k * p ∧ ∀ m : ℕ, m > p → m.Prime → ¬(m ∣ n)) ∧
    (∃ l : ℕ+, n + 1 = l * q ∧ ∀ m : ℕ, m > q → m.Prime → ¬(m ∣ (n + 1))) ∨
    (∃ k : ℕ+, n = k * q ∧ ∀ m : ℕ, m > q → m.Prime → ¬(m ∣ n)) ∧
    (∃ l : ℕ+, n + 1 = l * p ∧ ∀ m : ℕ, m > p → m.Prime → ¬(m ∣ (n + 1))) :=
by
  sorry

end consecutive_primes_as_greatest_divisors_l3914_391496


namespace complex_reciprocal_sum_l3914_391444

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
  sorry

end complex_reciprocal_sum_l3914_391444


namespace point_coordinates_l3914_391495

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : CartesianPoint) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : CartesianPoint) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : CartesianPoint) : ℝ :=
  |p.x|

theorem point_coordinates
  (p : CartesianPoint)
  (h1 : in_third_quadrant p)
  (h2 : distance_to_x_axis p = 5)
  (h3 : distance_to_y_axis p = 6) :
  p.x = -6 ∧ p.y = -5 :=
by sorry

end point_coordinates_l3914_391495


namespace complement_A_intersect_B_l3914_391481

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x^2 - 3*x - 4 < 0}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | 1 ≤ x ∧ x < 4} := by sorry

end complement_A_intersect_B_l3914_391481


namespace min_value_expression_min_value_achievable_l3914_391430

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x^3 + 4*x^2 + 2*x + 1) * (y^3 + 4*y^2 + 2*y + 1) * (z^3 + 4*z^2 + 2*z + 1)) / (x*y*z) ≥ 1331 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  ((x^3 + 4*x^2 + 2*x + 1) * (y^3 + 4*y^2 + 2*y + 1) * (z^3 + 4*z^2 + 2*z + 1)) / (x*y*z) = 1331 :=
by sorry

end min_value_expression_min_value_achievable_l3914_391430


namespace ten_thousand_squared_l3914_391411

theorem ten_thousand_squared : (10000 : ℕ) * 10000 = 100000000 := by
  sorry

end ten_thousand_squared_l3914_391411


namespace quadratic_two_distinct_roots_l3914_391498

theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x - a = 0 ∧ y^2 - 2*y - a = 0) ↔ a > -1 :=
by sorry

end quadratic_two_distinct_roots_l3914_391498


namespace percentage_difference_l3914_391470

theorem percentage_difference (x y : ℝ) : 
  3 = 0.15 * x → 3 = 0.30 * y → x - y = 10 := by sorry

end percentage_difference_l3914_391470


namespace quadratic_set_theorem_l3914_391457

theorem quadratic_set_theorem (a : ℝ) : 
  ({x : ℝ | x^2 + a*x = 0} = {0, 1}) → a = -1 := by
sorry

end quadratic_set_theorem_l3914_391457


namespace largest_unattainable_integer_l3914_391486

/-- Given positive integers a, b, c with no pairwise common divisor greater than 1,
    2abc-ab-bc-ca is the largest integer that cannot be expressed as xbc+yca+zab
    for non-negative integers x, y, z -/
theorem largest_unattainable_integer (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : Nat.gcd a b = 1) (hbc : Nat.gcd b c = 1) (hac : Nat.gcd a c = 1) :
  (∀ x y z : ℕ, x * b * c + y * c * a + z * a * b ≠ 2 * a * b * c - a * b - b * c - c * a) ∧
  (∀ n : ℕ, n > 2 * a * b * c - a * b - b * c - c * a →
    ∃ x y z : ℕ, x * b * c + y * c * a + z * a * b = n) :=
by sorry

end largest_unattainable_integer_l3914_391486


namespace value_of_a_l3914_391475

def A (a : ℝ) : Set ℝ := {0, 2, a^2}
def B (a : ℝ) : Set ℝ := {1, a}

theorem value_of_a : ∀ a : ℝ, A a ∪ B a = {0, 1, 2, 4} → a = 2 := by
  sorry

end value_of_a_l3914_391475


namespace balloon_distribution_l3914_391458

theorem balloon_distribution (yellow_balloons : ℕ) (blue_balloons : ℕ) (black_extra : ℕ) (schools : ℕ) :
  yellow_balloons = 3414 →
  blue_balloons = 5238 →
  black_extra = 1762 →
  schools = 15 →
  ((yellow_balloons + blue_balloons + (yellow_balloons + black_extra)) / schools : ℕ) = 921 :=
by sorry

end balloon_distribution_l3914_391458


namespace infinite_solutions_exist_l3914_391474

theorem infinite_solutions_exist (a b c d : ℝ) : 
  ((2*a + 16*b) + (3*c - 8*d)) / 2 = 74 →
  4*a + 6*b = 9*c - 12*d →
  ∃ (f : ℝ → ℝ → ℝ), b = f a d ∧ f a d = -a/21 - 2*d/7 :=
by sorry

end infinite_solutions_exist_l3914_391474


namespace heap_sheet_count_l3914_391424

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The number of sheets in a bunch -/
def sheets_per_bunch : ℕ := 4

/-- The number of sheets in a bundle -/
def sheets_per_bundle : ℕ := 2

/-- The total number of sheets removed -/
def total_sheets_removed : ℕ := 114

/-- The number of sheets in a heap -/
def sheets_per_heap : ℕ := 20

theorem heap_sheet_count :
  sheets_per_heap = 
    (total_sheets_removed - 
      (colored_bundles * sheets_per_bundle + 
       white_bunches * sheets_per_bunch)) / scrap_heaps :=
by sorry

end heap_sheet_count_l3914_391424


namespace most_reasonable_estimate_l3914_391455

/-- Represents the total number of female students in the first year -/
def total_female : ℕ := 504

/-- Represents the total number of male students in the first year -/
def total_male : ℕ := 596

/-- Represents the total number of students in the first year -/
def total_students : ℕ := total_female + total_male

/-- Represents the average weight of sampled female students -/
def avg_weight_female : ℝ := 49

/-- Represents the average weight of sampled male students -/
def avg_weight_male : ℝ := 57

/-- Theorem stating that the most reasonable estimate for the average weight
    of all first-year students is (504/1100) * 49 + (596/1100) * 57 -/
theorem most_reasonable_estimate :
  (total_female : ℝ) / total_students * avg_weight_female +
  (total_male : ℝ) / total_students * avg_weight_male =
  (504 : ℝ) / 1100 * 49 + (596 : ℝ) / 1100 * 57 := by
  sorry

end most_reasonable_estimate_l3914_391455


namespace symmetry_axis_phi_l3914_391403

/-- The value of φ when f(x) and g(x) have the same axis of symmetry --/
theorem symmetry_axis_phi : ∀ (ω : ℝ), ω > 0 →
  (∀ (φ : ℝ), |φ| < π/2 →
    (∀ (x : ℝ), 3 * Real.sin (ω * x - π/3) = 3 * Real.sin (ω * x + φ + π/2)) →
    φ = π/6) :=
by sorry

end symmetry_axis_phi_l3914_391403


namespace favorite_fruit_apples_l3914_391405

theorem favorite_fruit_apples (total students_oranges students_pears students_strawberries : ℕ) 
  (h1 : total = 450)
  (h2 : students_oranges = 70)
  (h3 : students_pears = 120)
  (h4 : students_strawberries = 113) :
  total - (students_oranges + students_pears + students_strawberries) = 147 := by
  sorry

end favorite_fruit_apples_l3914_391405


namespace calculate_x_l3914_391426

theorem calculate_x : ∀ (w y z x : ℕ),
  w = 90 →
  z = w + 25 →
  y = z + 15 →
  x = y + 7 →
  x = 137 := by
  sorry

end calculate_x_l3914_391426


namespace function_derivative_equality_l3914_391449

theorem function_derivative_equality (f : ℝ → ℝ) (x : ℝ) : 
  (∀ x, f x = x^2 * (x - 1)) → 
  (deriv f) x = x → 
  x = 0 ∨ x = 1 := by
sorry

end function_derivative_equality_l3914_391449


namespace xyz_mod_9_l3914_391463

theorem xyz_mod_9 (x y z : ℕ) : 
  x < 9 → y < 9 → z < 9 →
  (x + 3*y + 2*z) % 9 = 0 →
  (3*x + 2*y + z) % 9 = 5 →
  (2*x + y + 3*z) % 9 = 5 →
  (x*y*z) % 9 = 0 := by
sorry

end xyz_mod_9_l3914_391463


namespace carlotta_time_theorem_l3914_391432

def singing_time : ℕ := 6

def practice_time (n : ℕ) : ℕ := 2 * n

def tantrum_time (n : ℕ) : ℕ := 3 * n + 1

def total_time (singing : ℕ) : ℕ :=
  singing +
  singing * practice_time singing +
  singing * tantrum_time singing

theorem carlotta_time_theorem :
  total_time singing_time = 192 := by sorry

end carlotta_time_theorem_l3914_391432


namespace fuel_mixture_problem_l3914_391488

/-- Proves that the amount of fuel A added to the tank is 106 gallons -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) 
  (h1 : tank_capacity = 214)
  (h2 : ethanol_a = 0.12)
  (h3 : ethanol_b = 0.16)
  (h4 : total_ethanol = 30) :
  ∃ (fuel_a : ℝ), fuel_a = 106 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end fuel_mixture_problem_l3914_391488


namespace average_temperature_l3914_391428

def temperature_day1 : ℤ := -14
def temperature_day2 : ℤ := -8
def temperature_day3 : ℤ := 1
def num_days : ℕ := 3

theorem average_temperature :
  (temperature_day1 + temperature_day2 + temperature_day3) / num_days = -7 :=
by sorry

end average_temperature_l3914_391428


namespace florist_roses_l3914_391433

/-- 
Given a florist who:
- Sells 15 roses
- Picks 21 more roses
- Ends up with 56 roses
Prove that she must have started with 50 roses
-/
theorem florist_roses (initial : ℕ) : 
  initial - 15 + 21 = 56 → initial = 50 := by
  sorry

end florist_roses_l3914_391433


namespace shaded_area_equals_sixteen_twentyseventh_l3914_391404

/-- Represents the fraction of shaded area in each iteration -/
def shaded_fraction : ℕ → ℚ
  | 0 => 4/9
  | n + 1 => shaded_fraction n + (4/9) * (1/4)^(n+1)

/-- The limit of the shaded fraction as the number of iterations approaches infinity -/
def shaded_limit : ℚ := 16/27

theorem shaded_area_equals_sixteen_twentyseventh :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |shaded_fraction n - shaded_limit| < ε :=
sorry

end shaded_area_equals_sixteen_twentyseventh_l3914_391404


namespace rectangle_area_y_value_l3914_391445

theorem rectangle_area_y_value 
  (y : ℝ) 
  (h1 : y > 0) 
  (h2 : (5 - (-3)) * (y - (-1)) = 48) : 
  y = 5 := by
sorry

end rectangle_area_y_value_l3914_391445


namespace largest_divisor_of_expression_l3914_391479

theorem largest_divisor_of_expression (y : ℤ) (h : Even y) :
  (∃ (k : ℤ), (8*y+4)*(8*y+8)*(4*y+6)*(4*y+2) = 96 * k) ∧
  (∀ (n : ℤ), n > 96 → ¬(∀ (y : ℤ), Even y → ∃ (k : ℤ), (8*y+4)*(8*y+8)*(4*y+6)*(4*y+2) = n * k)) :=
sorry

end largest_divisor_of_expression_l3914_391479


namespace range_of_f_l3914_391491

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := arctan x + arctan ((x - 2) / (x + 2))

-- Theorem statement
theorem range_of_f :
  ∃ (S : Set ℝ), S = Set.range f ∧ S = {-π/4, arctan 2} :=
sorry

end range_of_f_l3914_391491


namespace square_minus_equal_two_implies_sum_equal_one_l3914_391485

theorem square_minus_equal_two_implies_sum_equal_one (m : ℝ) 
  (h : m^2 - m = 2) : 
  (m - 1)^2 + (m + 2) * (m - 2) = 1 := by
  sorry

end square_minus_equal_two_implies_sum_equal_one_l3914_391485


namespace range_of_a_l3914_391412

/-- The function f defined on positive real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.log x)^2 - Real.log x

/-- The function h defined on positive real numbers. -/
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (f a x + 1 - a) * (Real.log x)⁻¹

/-- The theorem stating the range of a given the conditions. -/
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (Real.exp (-3)) (Real.exp (-1)) →
                x₂ ∈ Set.Icc (Real.exp (-3)) (Real.exp (-1)) →
                |h a x₁ - h a x₂| ≤ a + 1/3) →
  a ∈ Set.Icc (1/11) (3/5) :=
by sorry

end range_of_a_l3914_391412


namespace complement_of_union_l3914_391472

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union : 
  (A ∪ B)ᶜ = {2, 4} :=
sorry

end complement_of_union_l3914_391472


namespace intersection_of_P_and_Q_l3914_391436

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 4}

theorem intersection_of_P_and_Q : P ∩ Q = {(3, -1)} := by
  sorry

end intersection_of_P_and_Q_l3914_391436


namespace factor_of_valid_Z_l3914_391464

def is_valid_Z (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000 ∧
  ∃ (a b c d : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    n = 10000000 * a + 1000000 * b + 100000 * c + 10000 * d +
        1000 * a + 100 * b + 10 * c + d

theorem factor_of_valid_Z (Z : ℕ) (h : is_valid_Z Z) : 
  10001 ∣ Z :=
sorry

end factor_of_valid_Z_l3914_391464


namespace log_sum_equals_two_l3914_391465

theorem log_sum_equals_two :
  2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end log_sum_equals_two_l3914_391465


namespace rectangle_semicircle_ratio_l3914_391423

theorem rectangle_semicircle_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a * b = π * b^2 → a / b = π := by
  sorry

end rectangle_semicircle_ratio_l3914_391423


namespace coin_division_sum_25_l3914_391441

/-- Represents the sum of products for coin divisions -/
def sum_of_products (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- Theorem: The sum of products for 25 coins is 300 -/
theorem coin_division_sum_25 :
  sum_of_products 25 = 300 := by
  sorry

#eval sum_of_products 25  -- Should output 300

end coin_division_sum_25_l3914_391441


namespace abs_neg_three_eq_three_l3914_391440

theorem abs_neg_three_eq_three : |(-3 : ℝ)| = 3 := by
  sorry

end abs_neg_three_eq_three_l3914_391440


namespace sara_marbles_l3914_391417

def marbles_problem (initial_marbles : ℕ) (remaining_marbles : ℕ) : Prop :=
  initial_marbles - remaining_marbles = 7

theorem sara_marbles : marbles_problem 10 3 := by
  sorry

end sara_marbles_l3914_391417


namespace slightly_used_crayons_l3914_391450

theorem slightly_used_crayons (total : ℕ) (new : ℕ) (broken : ℕ) (slightly_used : ℕ) : 
  total = 120 →
  new = total / 3 →
  broken = total / 5 →
  slightly_used = total - new - broken →
  slightly_used = 56 := by
sorry

end slightly_used_crayons_l3914_391450


namespace arcs_not_exceeding_120_degrees_l3914_391442

/-- Given 21 points on a circle, the number of arcs with these points as endpoints
    that have a measure of no more than 120° is equal to 100. -/
theorem arcs_not_exceeding_120_degrees (n : ℕ) (h : n = 21) : 
  (n.choose 2) - (n - 1) * (n / 2) = 100 := by
  sorry

end arcs_not_exceeding_120_degrees_l3914_391442


namespace cindy_travel_time_l3914_391431

/-- Calculates the total time for Cindy to travel 1 mile -/
theorem cindy_travel_time (run_speed walk_speed run_distance walk_distance : ℝ) :
  run_speed = 3 →
  walk_speed = 1 →
  run_distance = 0.5 →
  walk_distance = 0.5 →
  run_distance + walk_distance = 1 →
  (run_distance / run_speed + walk_distance / walk_speed) * 60 = 40 :=
by sorry

end cindy_travel_time_l3914_391431


namespace songs_learned_correct_l3914_391499

/-- The number of songs Vincent knew before summer camp -/
def songs_before : ℕ := 56

/-- The number of songs Vincent knows after summer camp -/
def songs_after : ℕ := 74

/-- The number of songs Vincent learned at summer camp -/
def songs_learned : ℕ := songs_after - songs_before

theorem songs_learned_correct : songs_learned = 18 := by sorry

end songs_learned_correct_l3914_391499


namespace reflection_matrix_condition_l3914_391461

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, b; -3/2, 1/2]

theorem reflection_matrix_condition (a b : ℚ) :
  (reflection_matrix a b) ^ 2 = 1 ↔ a = -1/2 ∧ b = -1/2 := by
  sorry

end reflection_matrix_condition_l3914_391461


namespace hexadecagon_triangles_l3914_391407

/-- The number of vertices in a regular hexadecagon -/
def n : ℕ := 16

/-- Represents that no three vertices are collinear in a regular hexadecagon -/
axiom no_collinear_vertices : True

/-- The number of triangles formed by choosing 3 vertices from n vertices -/
def num_triangles : ℕ := Nat.choose n 3

theorem hexadecagon_triangles : num_triangles = 560 := by
  sorry

end hexadecagon_triangles_l3914_391407


namespace sufficient_not_necessary_l3914_391489

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → (x - 2)^2 < 1) ∧
  (∃ x, (x - 2)^2 < 1 ∧ ¬(1 < x ∧ x < 2)) :=
by sorry

end sufficient_not_necessary_l3914_391489


namespace log_stack_sum_l3914_391473

theorem log_stack_sum (a l n : ℕ) (h1 : a = 5) (h2 : l = 15) (h3 : n = 11) :
  n * (a + l) / 2 = 110 := by
  sorry

end log_stack_sum_l3914_391473


namespace friction_force_on_rotated_board_l3914_391468

/-- The friction force on a block on a rotated rectangular board -/
theorem friction_force_on_rotated_board 
  (m g : ℝ) 
  (α β : ℝ) 
  (h_α_acute : 0 < α ∧ α < π / 2) 
  (h_β_acute : 0 < β ∧ β < π / 2) :
  ∃ F : ℝ, F = m * g * Real.sqrt (1 - Real.cos α ^ 2 * Real.cos β ^ 2) :=
by sorry

end friction_force_on_rotated_board_l3914_391468


namespace bakery_pastries_and_bagels_l3914_391490

/-- Proves that the total number of pastries and bagels is 474 given the bakery conditions -/
theorem bakery_pastries_and_bagels :
  let total_items : ℕ := 720
  let bread_rolls : ℕ := 240
  let croissants : ℕ := 75
  let muffins : ℕ := 145
  let cinnamon_rolls : ℕ := 110
  let pastries : ℕ := croissants + muffins + cinnamon_rolls
  let bagels : ℕ := total_items - (bread_rolls + pastries)
  let pastries_per_bread_roll : ℚ := 2.5
  let bagels_per_5_bread_rolls : ℕ := 3

  (pastries : ℚ) / bread_rolls = pastries_per_bread_roll ∧
  (bagels : ℚ) / bread_rolls = (bagels_per_5_bread_rolls : ℚ) / 5 →
  pastries + bagels = 474 := by
sorry

end bakery_pastries_and_bagels_l3914_391490


namespace unique_solution_iff_p_eq_neg_four_thirds_l3914_391427

/-- The equation has exactly one solution if and only if p = -4/3 -/
theorem unique_solution_iff_p_eq_neg_four_thirds :
  (∃! x : ℝ, (2 * x + 3) / (p * x - 2) = x) ↔ p = -4/3 :=
sorry

end unique_solution_iff_p_eq_neg_four_thirds_l3914_391427


namespace cubic_inequality_l3914_391439

theorem cubic_inequality (x y : ℝ) (h : x > y) : ¬(x^3 < y^3 ∨ x^3 = y^3) := by
  sorry

end cubic_inequality_l3914_391439


namespace shifted_quadratic_sum_l3914_391437

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 5

/-- The shifted quadratic function -/
def g (x : ℝ) : ℝ := f (x - 3)

/-- The coefficients of the shifted function -/
def a : ℝ := 3
def b : ℝ := -20
def c : ℝ := 38

theorem shifted_quadratic_sum :
  g x = a * x^2 + b * x + c ∧ a + b + c = 21 := by sorry

end shifted_quadratic_sum_l3914_391437


namespace total_seashells_eq_sum_l3914_391483

/-- The number of seashells Dan found on the beach -/
def total_seashells : ℕ := 56

/-- The number of seashells Dan gave to Jessica -/
def seashells_given : ℕ := 34

/-- The number of seashells Dan has left -/
def seashells_left : ℕ := 22

/-- Theorem stating that the total number of seashells is equal to
    the sum of seashells given away and seashells left -/
theorem total_seashells_eq_sum :
  total_seashells = seashells_given + seashells_left := by
  sorry

end total_seashells_eq_sum_l3914_391483


namespace binomial_17_4_l3914_391462

theorem binomial_17_4 : Nat.choose 17 4 = 2380 := by
  sorry

end binomial_17_4_l3914_391462


namespace shortest_side_is_15_l3914_391406

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of the first segment of the hypotenuse -/
  segment1 : ℝ
  /-- The length of the second segment of the hypotenuse -/
  segment2 : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- Assumption that segment1 is positive -/
  segment1_pos : segment1 > 0
  /-- Assumption that segment2 is positive -/
  segment2_pos : segment2 > 0
  /-- Assumption that radius is positive -/
  radius_pos : radius > 0

/-- The length of the shortest side in a right triangle with an inscribed circle -/
def shortest_side (t : RightTriangleWithInscribedCircle) : ℝ :=
  sorry

/-- Theorem stating that the shortest side is 15 units under given conditions -/
theorem shortest_side_is_15 (t : RightTriangleWithInscribedCircle) 
  (h1 : t.segment1 = 7) 
  (h2 : t.segment2 = 9) 
  (h3 : t.radius = 5) : 
  shortest_side t = 15 := by
  sorry

end shortest_side_is_15_l3914_391406


namespace distribute_5_balls_4_boxes_l3914_391421

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 56 := by
  sorry

end distribute_5_balls_4_boxes_l3914_391421


namespace sum_of_reciprocals_of_roots_l3914_391413

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 8 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ : ℝ) = 17 / 8 :=
sorry

end sum_of_reciprocals_of_roots_l3914_391413


namespace hyperbola_vertex_distance_l3914_391414

/-- The distance between the vertices of the hyperbola y^2/45 - x^2/20 = 1 is 6√5 -/
theorem hyperbola_vertex_distance : 
  let a := Real.sqrt 45
  let vertex_distance := 2 * a
  vertex_distance = 6 * Real.sqrt 5 := by
sorry

end hyperbola_vertex_distance_l3914_391414


namespace rectangle_perimeter_in_square_l3914_391452

theorem rectangle_perimeter_in_square (d : ℝ) (h : d = 6) : 
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 2 = d ∧
  ∃ (rect_side : ℝ), rect_side = s / Real.sqrt 2 ∧
  4 * rect_side = 12 := by
  sorry

end rectangle_perimeter_in_square_l3914_391452


namespace bunny_burrow_exits_l3914_391415

/-- The number of times a bunny comes out of its burrow per minute -/
def bunny_rate : ℕ := 3

/-- The number of bunnies -/
def num_bunnies : ℕ := 20

/-- The time period in hours -/
def time_period : ℕ := 10

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem bunny_burrow_exits :
  bunny_rate * minutes_per_hour * time_period * num_bunnies = 36000 := by
  sorry

end bunny_burrow_exits_l3914_391415


namespace trapezoid_long_side_is_correct_l3914_391447

/-- A rectangle with given dimensions divided into three equal-area shapes -/
structure DividedRectangle where
  length : ℝ
  width : ℝ
  trapezoid_long_side : ℝ
  is_valid : 
    length = 3 ∧ 
    width = 1 ∧
    0 < trapezoid_long_side ∧ 
    trapezoid_long_side < length

/-- The area of each shape is one-third of the rectangle's area -/
def equal_area_condition (r : DividedRectangle) : Prop :=
  let rectangle_area := r.length * r.width
  let trapezoid_area := (r.trapezoid_long_side + r.length / 2) * r.width / 2
  trapezoid_area = rectangle_area / 3

/-- The main theorem: the longer side of the trapezoid is 1.25 -/
theorem trapezoid_long_side_is_correct (r : DividedRectangle) 
  (h : equal_area_condition r) : r.trapezoid_long_side = 1.25 := by
  sorry

#check trapezoid_long_side_is_correct

end trapezoid_long_side_is_correct_l3914_391447


namespace sichuan_peppercorn_transport_l3914_391466

/-- Represents the capacity of a truck type -/
structure TruckCapacity where
  a : ℕ
  b : ℕ
  h : a = b + 20

/-- Represents the number of trucks needed for each type -/
structure TruckCount where
  a : ℕ
  b : ℕ

theorem sichuan_peppercorn_transport 
  (cap : TruckCapacity) 
  (h1 : 1000 / cap.a = 800 / cap.b)
  (count : TruckCount)
  (h2 : count.a + count.b = 18)
  (h3 : cap.a * count.a + cap.b * (count.b - 1) + 65 = 1625) :
  cap.a = 100 ∧ cap.b = 80 ∧ count.a = 10 ∧ count.b = 8 := by
  sorry

#check sichuan_peppercorn_transport

end sichuan_peppercorn_transport_l3914_391466


namespace mans_rate_in_still_water_l3914_391456

theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 6)
  (h2 : speed_against_stream = 2) : 
  (speed_with_stream + speed_against_stream) / 2 = 4 := by
  sorry

#check mans_rate_in_still_water

end mans_rate_in_still_water_l3914_391456


namespace calculation_result_l3914_391467

theorem calculation_result : (481 + 426)^2 - 4 * 481 * 426 = 3505 := by
  sorry

end calculation_result_l3914_391467


namespace a_share_is_4800_l3914_391448

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_share (contribution_a : ℕ) (months_a : ℕ) (contribution_b : ℕ) (months_b : ℕ) (total_profit : ℕ) : ℕ :=
  let money_months_a := contribution_a * months_a
  let money_months_b := contribution_b * months_b
  let total_money_months := money_months_a + money_months_b
  (money_months_a * total_profit) / total_money_months

/-- Theorem stating that A's share of the profit is 4800 given the problem conditions --/
theorem a_share_is_4800 :
  calculate_share 5000 8 6000 5 8400 = 4800 := by
  sorry

end a_share_is_4800_l3914_391448


namespace train_length_l3914_391408

/-- The length of a train given specific crossing times -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ) :
  bridge_length = 200 ∧ bridge_time = 10 ∧ post_time = 5 →
  ∃ train_length : ℝ, train_length = 200 ∧
    train_length / post_time = (train_length + bridge_length) / bridge_time :=
by sorry

end train_length_l3914_391408


namespace meeting_point_theorem_l3914_391451

/-- The meeting point of two people, given their positions and the fraction of the distance between them -/
def meetingPoint (x₁ y₁ x₂ y₂ t : ℝ) : ℝ × ℝ :=
  ((1 - t) * x₁ + t * x₂, (1 - t) * y₁ + t * y₂)

/-- Theorem stating that the meeting point one-third of the way from (2, 3) to (8, -5) is (4, 1/3) -/
theorem meeting_point_theorem :
  let mark_pos : ℝ × ℝ := (2, 3)
  let sandy_pos : ℝ × ℝ := (8, -5)
  let t : ℝ := 1/3
  meetingPoint mark_pos.1 mark_pos.2 sandy_pos.1 sandy_pos.2 t = (4, 1/3) := by
  sorry

end meeting_point_theorem_l3914_391451


namespace money_constraints_l3914_391480

theorem money_constraints (a b : ℝ) 
  (eq_constraint : 5 * a - b = 60)
  (ineq_constraint : 6 * a + b < 90) :
  a < 13.64 ∧ b < 8.18 := by
sorry

end money_constraints_l3914_391480


namespace thirteenth_term_is_30_l3914_391420

/-- An arithmetic sequence with specified terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  a5_eq_6 : a 5 = 6
  a8_eq_15 : a 8 = 15

/-- The 13th term of the arithmetic sequence is 30 -/
theorem thirteenth_term_is_30 (seq : ArithmeticSequence) : seq.a 13 = 30 := by
  sorry

end thirteenth_term_is_30_l3914_391420


namespace missing_number_l3914_391410

theorem missing_number (n : ℕ) : 
  (∀ k : ℕ, k < n → k * (k + 1) / 2 ≤ 575) ∧ 
  (n * (n + 1) / 2 > 575) → 
  n * (n + 1) / 2 - 575 = 20 := by
sorry

end missing_number_l3914_391410


namespace max_k_value_l3914_391409

open Real

noncomputable def f (x : ℝ) := exp x - x - 2

theorem max_k_value :
  ∃ (k : ℤ), k = 2 ∧
  (∀ (x : ℝ), x > 0 → (x - ↑k) * (exp x - 1) + x + 1 > 0) ∧
  (∀ (m : ℤ), m > 2 → ∃ (y : ℝ), y > 0 ∧ (y - ↑m) * (exp y - 1) + y + 1 ≤ 0) :=
sorry

end max_k_value_l3914_391409


namespace inverse_proportion_through_point_l3914_391478

/-- An inverse proportion function passing through the point (2,5) has k = 10 -/
theorem inverse_proportion_through_point (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (k / x = 5 ↔ x = 2)) → k = 10 := by
  sorry

end inverse_proportion_through_point_l3914_391478


namespace inequality_proof_l3914_391434

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 + b^3)^(1/3) < (a^2 + b^2)^(1/2) := by
  sorry

end inequality_proof_l3914_391434


namespace gas_pressure_change_l3914_391469

/-- Represents the state of a gas with pressure and volume -/
structure GasState where
  pressure : ℝ
  volume : ℝ

/-- The constant of proportionality for the gas -/
def gasConstant (state : GasState) : ℝ := state.pressure * state.volume

theorem gas_pressure_change 
  (initial : GasState) 
  (final : GasState) 
  (h1 : initial.pressure = 8) 
  (h2 : initial.volume = 3.5)
  (h3 : final.volume = 10.5)
  (h4 : gasConstant initial = gasConstant final) : 
  final.pressure = 8/3 := by
  sorry

#check gas_pressure_change

end gas_pressure_change_l3914_391469


namespace composite_power_sum_l3914_391477

theorem composite_power_sum (n : ℕ) (h : n > 1) :
  ∃ (k : ℕ), k > 1 ∧ k ∣ ((2^(2^(n+1)) + 2^(2^n) + 1) / 3) := by
  sorry

end composite_power_sum_l3914_391477


namespace inequality_proof_l3914_391497

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1/2) :
  1/(1-a) + 1/(1-b) ≥ 4 ∧ (1/(1-a) + 1/(1-b) = 4 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

end inequality_proof_l3914_391497


namespace least_divisible_n_divisors_l3914_391429

theorem least_divisible_n_divisors (n : ℕ) : 
  (∀ k < n, ¬(3^3 * 5^5 * 7^7 ∣ (149^k - 2^k))) →
  (3^3 * 5^5 * 7^7 ∣ (149^n - 2^n)) →
  (∀ m : ℕ, m > n → ¬(3^3 * 5^5 * 7^7 ∣ (149^m - 2^m))) →
  Nat.card {d : ℕ | d ∣ n} = 270 :=
sorry

end least_divisible_n_divisors_l3914_391429


namespace least_addend_for_divisibility_least_addend_for_1156_and_97_l3914_391443

theorem least_addend_for_divisibility (n m : ℕ) (h : n > 0) : 
  ∃ (x : ℕ), (n + x) % m = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % m ≠ 0 :=
sorry

theorem least_addend_for_1156_and_97 : 
  ∃ (x : ℕ), (1156 + x) % 97 = 0 ∧ ∀ (y : ℕ), y < x → (1156 + y) % 97 ≠ 0 ∧ x = 8 :=
sorry

end least_addend_for_divisibility_least_addend_for_1156_and_97_l3914_391443


namespace f_has_zero_in_interval_l3914_391400

/-- The function f(x) = x^3 + x - 8 -/
def f (x : ℝ) : ℝ := x^3 + x - 8

/-- Theorem: f(x) has a zero in the interval (1, 2) -/
theorem f_has_zero_in_interval :
  ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0 :=
by
  have h1 : f 1 < 0 := by sorry
  have h2 : f 2 > 0 := by sorry
  sorry

#check f_has_zero_in_interval

end f_has_zero_in_interval_l3914_391400


namespace jakes_snake_length_l3914_391402

theorem jakes_snake_length (j p : ℕ) : 
  j = p + 12 →  -- Jake's snake is 12 inches longer than Penny's snake
  j + p = 70 →  -- The combined length of the two snakes is 70 inches
  j = 41        -- Jake's snake is 41 inches long
:= by sorry

end jakes_snake_length_l3914_391402


namespace intersection_point_property_l3914_391459

theorem intersection_point_property (n : ℕ) (x₀ y₀ : ℝ) (hn : n ≥ 2) 
  (h1 : y₀^2 = n * x₀ - 1) (h2 : y₀ = x₀) :
  ∀ m : ℕ, m > 0 → ∃ k : ℕ, k ≥ 2 ∧ (x₀^m)^2 = k * (x₀^m) - 1 := by
  sorry

end intersection_point_property_l3914_391459


namespace min_third_side_right_triangle_l3914_391454

theorem min_third_side_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a = 7 ∨ b = 7 ∨ c = 7) → 
  (a = 24 ∨ b = 24 ∨ c = 24) → 
  a^2 + b^2 = c^2 → 
  min a (min b c) ≥ Real.sqrt 527 :=
by sorry

end min_third_side_right_triangle_l3914_391454
