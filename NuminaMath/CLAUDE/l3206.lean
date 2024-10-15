import Mathlib

namespace NUMINAMATH_CALUDE_museum_ticket_cost_l3206_320698

theorem museum_ticket_cost (num_students num_teachers : ℕ) 
  (student_ticket_price teacher_ticket_price : ℚ) : 
  num_students = 12 →
  num_teachers = 4 →
  student_ticket_price = 1 →
  teacher_ticket_price = 3 →
  (num_students : ℚ) * student_ticket_price + (num_teachers : ℚ) * teacher_ticket_price = 24 :=
by sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l3206_320698


namespace NUMINAMATH_CALUDE_digit_equation_solution_l3206_320690

theorem digit_equation_solution :
  ∀ x y z : ℕ,
    x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 →
    (10 * x + 5) * (300 + 10 * y + z) = 7850 →
    x = 2 ∧ y = 1 ∧ z = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l3206_320690


namespace NUMINAMATH_CALUDE_circle_circumference_increase_l3206_320649

theorem circle_circumference_increase (r : ℝ) : 
  2 * Real.pi * (r + 2) - 2 * Real.pi * r = 12.56 := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_increase_l3206_320649


namespace NUMINAMATH_CALUDE_circle_equation_l3206_320670

theorem circle_equation (x y : ℝ) : 
  (∃ h k r : ℝ, (5*h - 3*k = 8) ∧ 
    ((x - h)^2 + (y - k)^2 = r^2) ∧ 
    (h = r ∨ k = r) ∧ 
    (h = r ∨ k = -r)) →
  ((x - 4)^2 + (y - 4)^2 = 16 ∨ (x - 1)^2 + (y + 1)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3206_320670


namespace NUMINAMATH_CALUDE_no_solution_exists_l3206_320674

theorem no_solution_exists : ¬∃ (a b c d : ℤ),
  (a * b * c * d - a = 1961) ∧
  (a * b * c * d - b = 961) ∧
  (a * b * c * d - c = 61) ∧
  (a * b * c * d - d = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3206_320674


namespace NUMINAMATH_CALUDE_weight_replacement_l3206_320627

theorem weight_replacement (n : ℕ) (new_weight avg_increase : ℝ) :
  n = 8 →
  new_weight = 93 →
  avg_increase = 3.5 →
  new_weight - n * avg_increase = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l3206_320627


namespace NUMINAMATH_CALUDE_negative_numbers_l3206_320605

theorem negative_numbers (x y z : ℝ) 
  (h1 : 2 * x - y < 0) 
  (h2 : 3 * y - 2 * z < 0) 
  (h3 : 4 * z - 3 * x < 0) : 
  x < 0 ∧ y < 0 ∧ z < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_numbers_l3206_320605


namespace NUMINAMATH_CALUDE_ending_number_proof_l3206_320609

def starting_number : ℕ := 100
def multiples_count : ℚ := 13.5

theorem ending_number_proof :
  ∃ (n : ℕ), n ≥ starting_number ∧ 
  (n - starting_number) / 8 + 1 = multiples_count ∧
  n = 204 :=
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l3206_320609


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l3206_320668

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃)) : 
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l3206_320668


namespace NUMINAMATH_CALUDE_aang_fish_count_l3206_320678

theorem aang_fish_count :
  ∀ (aang_fish : ℕ),
  let sokka_fish : ℕ := 5
  let toph_fish : ℕ := 12
  let total_people : ℕ := 3
  let average_fish : ℕ := 8
  (aang_fish + sokka_fish + toph_fish) / total_people = average_fish →
  aang_fish = 7 := by
sorry

end NUMINAMATH_CALUDE_aang_fish_count_l3206_320678


namespace NUMINAMATH_CALUDE_max_value_implies_a_value_l3206_320657

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x

-- Define the theorem
theorem max_value_implies_a_value (a : ℝ) (h1 : a ≠ 0) :
  (∃ (M : ℝ), M = 3 ∧ ∀ x ∈ Set.Icc 0 3, f a x ≤ M) →
  (a = 1 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_value_l3206_320657


namespace NUMINAMATH_CALUDE_square_sum_xy_l3206_320687

theorem square_sum_xy (x y : ℝ) 
  (h1 : 2 * x * (x + y) = 72) 
  (h2 : 3 * y * (x + y) = 108) : 
  (x + y)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l3206_320687


namespace NUMINAMATH_CALUDE_square_area_200m_l3206_320623

/-- The area of a square with side length 200 meters is 40000 square meters. -/
theorem square_area_200m : 
  let side_length : ℝ := 200
  let area : ℝ := side_length * side_length
  area = 40000 := by sorry

end NUMINAMATH_CALUDE_square_area_200m_l3206_320623


namespace NUMINAMATH_CALUDE_line_equation_l3206_320606

-- Define the line l
def Line := ℝ → ℝ → Prop

-- Define the point type
def Point := ℝ × ℝ

-- Define the distance function between a point and a line
def distance (p : Point) (l : Line) : ℝ := sorry

-- Define the condition that line l passes through point P(1,2)
def passes_through (l : Line) : Prop :=
  l 1 2

-- Define the condition that line l is equidistant from A(2,3) and B(0,-5)
def equidistant (l : Line) : Prop :=
  distance (2, 3) l = distance (0, -5) l

-- State the theorem
theorem line_equation (l : Line) 
  (h1 : passes_through l) 
  (h2 : equidistant l) : 
  (∀ x y, l x y ↔ 4*x - y - 2 = 0) ∨ 
  (∀ x y, l x y ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3206_320606


namespace NUMINAMATH_CALUDE_chess_club_girls_l3206_320631

theorem chess_club_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  present = 18 →
  boys + girls = total →
  boys + (1/3 : ℚ) * girls = present →
  girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_chess_club_girls_l3206_320631


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3206_320603

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times that of the other, prove that the number of
    sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →                             -- Assume positive side length
  50 * (3 * s) = n * s →              -- Same perimeter condition
  n = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l3206_320603


namespace NUMINAMATH_CALUDE_expectation_linear_transform_binomial_probability_normal_probability_l3206_320693

/-- The expectation of a random variable -/
noncomputable def expectation (X : Real → Real) : Real := sorry

/-- The variance of a random variable -/
noncomputable def variance (X : Real → Real) : Real := sorry

/-- The probability mass function for a binomial distribution -/
noncomputable def binomial_pmf (n : Nat) (p : Real) (k : Nat) : Real := sorry

/-- The cumulative distribution function for a normal distribution -/
noncomputable def normal_cdf (μ σ : Real) (x : Real) : Real := sorry

theorem expectation_linear_transform (X : Real → Real) :
  expectation (fun x => 2 * x + 3) = 2 * expectation X + 3 := by sorry

theorem binomial_probability (X : Real → Real) :
  binomial_pmf 6 (1/2) 3 = 5/16 := by sorry

theorem normal_probability (X : Real → Real) (σ : Real) :
  normal_cdf 2 σ 4 = 0.9 →
  normal_cdf 2 σ 2 - normal_cdf 2 σ 0 = 0.4 := by sorry

end NUMINAMATH_CALUDE_expectation_linear_transform_binomial_probability_normal_probability_l3206_320693


namespace NUMINAMATH_CALUDE_equal_savings_l3206_320655

/-- Represents a person's financial data -/
structure Person where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup -/
def financialProblem (p1 p2 : Person) : Prop :=
  -- Income ratio condition
  p1.income * 4 = p2.income * 5 ∧
  -- Expenditure ratio condition
  p1.expenditure * 2 = p2.expenditure * 3 ∧
  -- P1's income is 5000
  p1.income = 5000 ∧
  -- Savings is income minus expenditure
  p1.savings = p1.income - p1.expenditure ∧
  p2.savings = p2.income - p2.expenditure ∧
  -- Both persons save the same amount
  p1.savings = p2.savings

/-- The theorem to prove -/
theorem equal_savings (p1 p2 : Person) :
  financialProblem p1 p2 → p1.savings = 2000 ∧ p2.savings = 2000 := by
  sorry

end NUMINAMATH_CALUDE_equal_savings_l3206_320655


namespace NUMINAMATH_CALUDE_bookcase_max_weight_bookcase_weight_proof_l3206_320619

/-- The maximum weight a bookcase can hold given the weights of various items and the excess weight -/
theorem bookcase_max_weight 
  (hardcover_weight : ℝ) 
  (textbook_weight : ℝ) 
  (knickknack_weight : ℝ) 
  (excess_weight : ℝ) : ℝ :=
  let total_weight := hardcover_weight + textbook_weight + knickknack_weight
  total_weight - excess_weight

/-- Proves that the bookcase can hold 80 pounds given the specified conditions -/
theorem bookcase_weight_proof 
  (hardcover_weight : ℝ) 
  (textbook_weight : ℝ) 
  (knickknack_weight : ℝ) 
  (excess_weight : ℝ) :
  hardcover_weight = 70 * 0.5 →
  textbook_weight = 30 * 2 →
  knickknack_weight = 3 * 6 →
  excess_weight = 33 →
  bookcase_max_weight hardcover_weight textbook_weight knickknack_weight excess_weight = 80 := by
  sorry

end NUMINAMATH_CALUDE_bookcase_max_weight_bookcase_weight_proof_l3206_320619


namespace NUMINAMATH_CALUDE_snickers_bars_proof_l3206_320654

/-- The number of points needed to win the Nintendo Switch -/
def total_points_needed : ℕ := 2000

/-- The number of chocolate bunnies sold -/
def chocolate_bunnies_sold : ℕ := 8

/-- The number of points earned per chocolate bunny -/
def points_per_bunny : ℕ := 100

/-- The number of points earned per Snickers bar -/
def points_per_snickers : ℕ := 25

/-- Calculates the number of Snickers bars needed to win the Nintendo Switch -/
def snickers_bars_needed : ℕ :=
  (total_points_needed - chocolate_bunnies_sold * points_per_bunny) / points_per_snickers

theorem snickers_bars_proof :
  snickers_bars_needed = 48 := by
  sorry

end NUMINAMATH_CALUDE_snickers_bars_proof_l3206_320654


namespace NUMINAMATH_CALUDE_first_train_speed_is_40_l3206_320628

/-- The speed of the first train in km/h -/
def first_train_speed : ℝ := sorry

/-- The speed of the second train in km/h -/
def second_train_speed : ℝ := 50

/-- The time difference between the departure of the two trains in hours -/
def time_difference : ℝ := 1

/-- The distance at which the two trains meet in km -/
def meeting_distance : ℝ := 200

/-- Theorem stating that given the conditions, the speed of the first train is 40 km/h -/
theorem first_train_speed_is_40 : first_train_speed = 40 := by sorry

end NUMINAMATH_CALUDE_first_train_speed_is_40_l3206_320628


namespace NUMINAMATH_CALUDE_fraction_product_proof_l3206_320659

theorem fraction_product_proof : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_proof_l3206_320659


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l3206_320653

/-- The measure of one interior angle of a regular octagon is 135 degrees -/
theorem regular_octagon_interior_angle : ℝ :=
  135

#check regular_octagon_interior_angle

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l3206_320653


namespace NUMINAMATH_CALUDE_sqrt_two_thirds_same_type_as_sqrt6_l3206_320664

-- Define what it means for a real number to be of the same type as √6
def same_type_as_sqrt6 (x : ℝ) : Prop :=
  ∃ (a b : ℚ), x = a * Real.sqrt 2 * b * Real.sqrt 3

-- State the theorem
theorem sqrt_two_thirds_same_type_as_sqrt6 :
  same_type_as_sqrt6 (Real.sqrt (2/3)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_two_thirds_same_type_as_sqrt6_l3206_320664


namespace NUMINAMATH_CALUDE_line_through_points_l3206_320633

/-- A line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  eq_at_point1 : (a * point1.1 + b) = point1.2
  eq_at_point2 : (a * point2.1 + b) = point2.2

/-- Theorem stating that for a line y = ax + b passing through (2, 3) and (6, 19), a - b = 9 -/
theorem line_through_points (l : Line) 
    (h1 : l.point1 = (2, 3))
    (h2 : l.point2 = (6, 19)) : 
  l.a - l.b = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3206_320633


namespace NUMINAMATH_CALUDE_intersecting_sets_implies_a_equals_one_l3206_320632

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | a * x^2 - 1 = 0 ∧ a > 0}
def N : Set ℝ := {-1/2, 1/2, 1}

-- Define the "intersect" property
def intersect (A B : Set ℝ) : Prop :=
  (∃ x, x ∈ A ∧ x ∈ B) ∧ (¬(A ⊆ B) ∧ ¬(B ⊆ A))

-- State the theorem
theorem intersecting_sets_implies_a_equals_one :
  ∀ a : ℝ, intersect (M a) N → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_sets_implies_a_equals_one_l3206_320632


namespace NUMINAMATH_CALUDE_pencil_cost_l3206_320666

/-- The cost of an item when paying with a dollar and receiving change -/
def item_cost (payment : ℚ) (change : ℚ) : ℚ :=
  payment - change

/-- Theorem: Given a purchase where the buyer pays with a one-dollar bill
    and receives 65 cents in change, the cost of the item is 35 cents. -/
theorem pencil_cost :
  let payment : ℚ := 1
  let change : ℚ := 65/100
  item_cost payment change = 35/100 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l3206_320666


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_log_sum_l3206_320683

theorem arithmetic_geometric_sequence_log_sum (a b c x y z : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z →
  (∃ d : ℝ, b - a = d ∧ c - b = d) →
  (∃ q : ℝ, y / x = q ∧ z / y = q) →
  (b - c) * Real.log x + (c - a) * Real.log y + (a - b) * Real.log z = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_log_sum_l3206_320683


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3206_320689

theorem imaginary_part_of_z (z : ℂ) : (3 - 4*I)*z = Complex.abs (4 + 3*I) → Complex.im z = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3206_320689


namespace NUMINAMATH_CALUDE_harkamal_payment_l3206_320634

/-- The total amount Harkamal paid to the shopkeeper -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1145 to the shopkeeper -/
theorem harkamal_payment :
  total_amount 8 70 9 65 = 1145 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l3206_320634


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3206_320639

theorem sum_of_x_and_y (x y : ℚ) 
  (eq1 : 5 * x - 7 * y = 17) 
  (eq2 : 3 * x + 5 * y = 11) : 
  x + y = 83 / 23 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3206_320639


namespace NUMINAMATH_CALUDE_factorization_equality_l3206_320672

theorem factorization_equality (m n : ℝ) : m^2*n - 2*m*n + n = n*(m-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3206_320672


namespace NUMINAMATH_CALUDE_ice_cost_theorem_l3206_320607

/-- The cost of ice for enterprise A given the specified conditions -/
theorem ice_cost_theorem 
  (a : ℝ) -- Price of ice from B in rubles per ton
  (p : ℝ) -- Transportation cost in rubles per ton-kilometer
  (n : ℝ) -- Ice melting rate (n/1000 of mass per kilometer)
  (s : ℝ) -- Distance from B to C through A in kilometers
  (h1 : 0 < a) -- Price is positive
  (h2 : 0 < p) -- Transportation cost is positive
  (h3 : 0 < n) -- Melting rate is positive
  (h4 : 0 < s) -- Distance is positive
  (h5 : n * s < 2000) -- Ensure denominator is positive
  : ∃ (z : ℝ), z = (2.5 * a + p * s) * 1000 / (2000 - n * s) ∧ 
    z * (2 - n * s / 1000) = 2.5 * a + p * s := by
  sorry

end NUMINAMATH_CALUDE_ice_cost_theorem_l3206_320607


namespace NUMINAMATH_CALUDE_smallBase_altitude_ratio_l3206_320625

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Length of the smaller base -/
  smallBase : ℝ
  /-- Length of the larger base -/
  largeBase : ℝ
  /-- Length of the diagonal -/
  diagonal : ℝ
  /-- Length of the altitude -/
  altitude : ℝ
  /-- The larger base is twice the smaller base -/
  largeBase_eq : largeBase = 2 * smallBase
  /-- The diagonal is 1.5 times the larger base -/
  diagonal_eq : diagonal = 1.5 * largeBase
  /-- The altitude equals the smaller base -/
  altitude_eq : altitude = smallBase

/-- Theorem: The ratio of the smaller base to the altitude is 1:1 -/
theorem smallBase_altitude_ratio (t : IsoscelesTrapezoid) : t.smallBase / t.altitude = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallBase_altitude_ratio_l3206_320625


namespace NUMINAMATH_CALUDE_A_nonempty_A_subset_B_l3206_320680

/-- Definition of set A -/
def A (a : ℝ) : Set ℝ := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}

/-- Definition of set B -/
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 16}

/-- Theorem for the non-emptiness of A -/
theorem A_nonempty (a : ℝ) : (A a).Nonempty ↔ a ≥ 6 := by sorry

/-- Theorem for A being a subset of B -/
theorem A_subset_B (a : ℝ) : A a ⊆ B ↔ a < 6 ∨ a > 15/2 := by sorry

end NUMINAMATH_CALUDE_A_nonempty_A_subset_B_l3206_320680


namespace NUMINAMATH_CALUDE_johns_cows_value_increase_l3206_320600

/-- Calculates the increase in value of cows after weight gain -/
def cow_value_increase (initial_weights : Fin 3 → ℝ) (increase_factors : Fin 3 → ℝ) (price_per_pound : ℝ) : ℝ :=
  let new_weights := fun i => initial_weights i * increase_factors i
  let initial_values := fun i => initial_weights i * price_per_pound
  let new_values := fun i => new_weights i * price_per_pound
  (Finset.sum Finset.univ new_values) - (Finset.sum Finset.univ initial_values)

/-- The increase in value of John's cows after weight gain -/
theorem johns_cows_value_increase :
  let initial_weights : Fin 3 → ℝ := ![732, 845, 912]
  let increase_factors : Fin 3 → ℝ := ![1.35, 1.28, 1.4]
  let price_per_pound : ℝ := 2.75
  cow_value_increase initial_weights increase_factors price_per_pound = 2358.40 := by
  sorry

end NUMINAMATH_CALUDE_johns_cows_value_increase_l3206_320600


namespace NUMINAMATH_CALUDE_production_rate_equation_correct_l3206_320645

/-- Represents the production rate of the master and apprentice -/
structure ProductionRate where
  master : ℝ
  apprentice : ℝ
  total : ℝ
  master_total : ℝ
  apprentice_total : ℝ

/-- The production rate equation is correct given the conditions -/
theorem production_rate_equation_correct (p : ProductionRate)
  (h1 : p.master + p.apprentice = p.total)
  (h2 : p.total = 40)
  (h3 : p.master_total = 300)
  (h4 : p.apprentice_total = 100) :
  300 / p.master = 100 / (40 - p.master) :=
sorry

end NUMINAMATH_CALUDE_production_rate_equation_correct_l3206_320645


namespace NUMINAMATH_CALUDE_dans_limes_l3206_320602

theorem dans_limes (initial_limes : ℝ) (given_limes : ℝ) (remaining_limes : ℝ) : 
  initial_limes = 9 → given_limes = 4.5 → remaining_limes = initial_limes - given_limes → remaining_limes = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_dans_limes_l3206_320602


namespace NUMINAMATH_CALUDE_no_separable_representation_l3206_320614

theorem no_separable_representation :
  ¬ ∃ (f g : ℝ → ℝ), ∀ x y : ℝ, 1 + x^2016 * y^2016 = f x * g y := by
  sorry

end NUMINAMATH_CALUDE_no_separable_representation_l3206_320614


namespace NUMINAMATH_CALUDE_chess_tournament_director_games_l3206_320692

theorem chess_tournament_director_games (total_games : ℕ) (h : total_games = 325) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_games ∧ 
  ∀ (k : ℕ), n * (n - 1) / 2 + k = total_games → k ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_director_games_l3206_320692


namespace NUMINAMATH_CALUDE_shooting_team_composition_l3206_320604

theorem shooting_team_composition (x y : ℕ) : 
  x > 0 → y > 0 →
  (22 * x + 47 * y) / (x + y) = 41 →
  (y : ℚ) / (x + y) = 19 / 25 := by
sorry

end NUMINAMATH_CALUDE_shooting_team_composition_l3206_320604


namespace NUMINAMATH_CALUDE_dataset_transformation_l3206_320667

theorem dataset_transformation (initial_points : ℕ) : 
  initial_points = 200 →
  let increased_points := initial_points + initial_points / 5
  let final_points := increased_points - increased_points / 4
  final_points = 180 := by
sorry

end NUMINAMATH_CALUDE_dataset_transformation_l3206_320667


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_997_l3206_320647

theorem modular_inverse_13_mod_997 :
  ∃ x : ℕ, x < 997 ∧ (13 * x) % 997 = 1 :=
by
  use 767
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_997_l3206_320647


namespace NUMINAMATH_CALUDE_flag_actions_total_time_l3206_320669

/-- Calculates the total time spent on flag actions throughout the day -/
theorem flag_actions_total_time 
  (pole_height : ℝ) 
  (half_mast : ℝ) 
  (speed_raise : ℝ) 
  (speed_lower_half : ℝ) 
  (speed_raise_half : ℝ) 
  (speed_lower_full : ℝ) 
  (h1 : pole_height = 60) 
  (h2 : half_mast = 30) 
  (h3 : speed_raise = 2) 
  (h4 : speed_lower_half = 3) 
  (h5 : speed_raise_half = 1.5) 
  (h6 : speed_lower_full = 2.5) :
  pole_height / speed_raise + 
  half_mast / speed_lower_half + 
  half_mast / speed_raise_half + 
  pole_height / speed_lower_full = 84 :=
by sorry


end NUMINAMATH_CALUDE_flag_actions_total_time_l3206_320669


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l3206_320626

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 5 distinct objects into 3 distinct groups,
    where each group must contain at least one object, is 150 -/
theorem distribute_five_to_three : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l3206_320626


namespace NUMINAMATH_CALUDE_difference_of_squares_l3206_320621

theorem difference_of_squares : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3206_320621


namespace NUMINAMATH_CALUDE_price_decrease_approx_l3206_320697

/-- Original price in dollars for 6 cups -/
def original_price : ℚ := 8

/-- Number of cups in original offer -/
def original_cups : ℕ := 6

/-- Promotional price in dollars for 8 cups -/
def promo_price : ℚ := 6

/-- Number of cups in promotional offer -/
def promo_cups : ℕ := 8

/-- Calculate the percent decrease in price per cup -/
def percent_decrease : ℚ :=
  (original_price / original_cups - promo_price / promo_cups) / (original_price / original_cups) * 100

/-- Theorem stating that the percent decrease is approximately 43.6% -/
theorem price_decrease_approx :
  abs (percent_decrease - 43.6) < 0.1 := by sorry

end NUMINAMATH_CALUDE_price_decrease_approx_l3206_320697


namespace NUMINAMATH_CALUDE_product_probability_l3206_320699

/-- Claire's spinner has 7 equally probable outcomes -/
def claire_spinner : ℕ := 7

/-- Jamie's spinner has 12 equally probable outcomes -/
def jamie_spinner : ℕ := 12

/-- The threshold for the product of spins -/
def threshold : ℕ := 42

/-- The probability that the product of Claire's and Jamie's spins is less than the threshold -/
theorem product_probability : 
  (Finset.filter (λ (pair : ℕ × ℕ) => pair.1 * pair.2 < threshold) 
    (Finset.product (Finset.range claire_spinner) (Finset.range jamie_spinner))).card / 
  (claire_spinner * jamie_spinner : ℚ) = 31 / 42 := by sorry

end NUMINAMATH_CALUDE_product_probability_l3206_320699


namespace NUMINAMATH_CALUDE_exam_candidates_count_l3206_320635

theorem exam_candidates_count :
  ∀ (x : ℕ),
  (x : ℝ) * 0.07 = (x : ℝ) * 0.06 + 82 →
  x = 8200 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_count_l3206_320635


namespace NUMINAMATH_CALUDE_optimal_profit_l3206_320671

/-- Profit function for n plants per pot -/
def P (n : ℕ) : ℝ := n * (5 - 0.5 * (n - 3))

/-- The optimal number of plants per pot -/
def optimal_plants : ℕ := 5

theorem optimal_profit :
  (P optimal_plants = 20) ∧ 
  (∀ n : ℕ, 3 ≤ n ∧ n ≤ 6 → P n ≤ 20) ∧
  (∀ n : ℕ, 3 ≤ n ∧ n < optimal_plants → P n < 20) ∧
  (∀ n : ℕ, optimal_plants < n ∧ n ≤ 6 → P n < 20) := by
  sorry

#eval P optimal_plants  -- Should output 20

end NUMINAMATH_CALUDE_optimal_profit_l3206_320671


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3206_320677

def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

def inclination_angle (f : ℝ → ℝ → Prop) : ℝ := sorry

theorem line_inclination_angle :
  inclination_angle line_equation = π * (5/6) := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3206_320677


namespace NUMINAMATH_CALUDE_volume_rotated_square_l3206_320646

/-- The volume of a solid formed by rotating a square around its diagonal -/
theorem volume_rotated_square (area : ℝ) (volume : ℝ) : 
  area = 4 → volume = (4 * Real.sqrt 2 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_rotated_square_l3206_320646


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l3206_320618

theorem geometric_sequence_solution (x : ℝ) :
  (1 : ℝ) * x = x * 9 → x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l3206_320618


namespace NUMINAMATH_CALUDE_artemon_distance_l3206_320684

-- Define the rectangle
def rectangle_length : ℝ := 6
def rectangle_width : ℝ := 2.5

-- Define speeds
def malvina_speed : ℝ := 4
def buratino_speed : ℝ := 6
def artemon_speed : ℝ := 12

-- Theorem statement
theorem artemon_distance :
  let diagonal : ℝ := Real.sqrt (rectangle_length^2 + rectangle_width^2)
  let meeting_time : ℝ := diagonal / (malvina_speed + buratino_speed)
  let artemon_distance : ℝ := artemon_speed * meeting_time
  artemon_distance = 7.8 := by sorry

end NUMINAMATH_CALUDE_artemon_distance_l3206_320684


namespace NUMINAMATH_CALUDE_art_museum_cost_l3206_320675

def total_cost (initial_fee : ℕ) (initial_visits_per_year : ℕ) (new_fee : ℕ) (new_visits_per_year : ℕ) (total_years : ℕ) : ℕ :=
  (initial_fee * initial_visits_per_year) + (new_fee * new_visits_per_year * (total_years - 1))

theorem art_museum_cost : 
  total_cost 5 12 7 4 3 = 116 := by sorry

end NUMINAMATH_CALUDE_art_museum_cost_l3206_320675


namespace NUMINAMATH_CALUDE_square_sum_given_linear_equations_l3206_320620

theorem square_sum_given_linear_equations (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x + y = 22) : x^2 + y^2 = 404 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_linear_equations_l3206_320620


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3206_320685

/-- Parabola defined by x^2 = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Point (x₀, y₀) is inside the parabola if x₀² < 4y₀ -/
def inside_parabola (x₀ y₀ : ℝ) : Prop := x₀^2 < 4*y₀

/-- Line defined by x₀x = 2(y + y₀) -/
def line (x₀ y₀ x y : ℝ) : Prop := x₀*x = 2*(y + y₀)

/-- No common points between the line and the parabola -/
def no_common_points (x₀ y₀ : ℝ) : Prop :=
  ∀ x y : ℝ, parabola x y → line x₀ y₀ x y → False

theorem parabola_line_intersection (x₀ y₀ : ℝ) 
  (h : inside_parabola x₀ y₀) : no_common_points x₀ y₀ := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3206_320685


namespace NUMINAMATH_CALUDE_sum_of_weighted_variables_l3206_320612

theorem sum_of_weighted_variables (x y z : ℝ) 
  (eq1 : x + y + z = 20) 
  (eq2 : x + 2*y + 3*z = 16) : 
  x + 3*y + 5*z = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_weighted_variables_l3206_320612


namespace NUMINAMATH_CALUDE_initial_birds_count_l3206_320613

theorem initial_birds_count (B : ℕ) : 
  (B + 4 - 3 + 6 = 12) → B = 5 := by
sorry

end NUMINAMATH_CALUDE_initial_birds_count_l3206_320613


namespace NUMINAMATH_CALUDE_seating_arrangements_special_guest_seating_l3206_320608

theorem seating_arrangements (n : Nat) (k : Nat) (h : n > k) :
  (n : Nat) * (n - 1).factorial = n * (n - 1 : Nat).factorial :=
by sorry

theorem special_guest_seating :
  8 * 7 * 6 * 5 * 4 * 3 * 2 = 20160 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_special_guest_seating_l3206_320608


namespace NUMINAMATH_CALUDE_point_meeting_time_l3206_320636

theorem point_meeting_time (b_initial c_initial b_speed c_speed : ℚ) (h1 : b_initial = -8)
  (h2 : c_initial = 16) (h3 : b_speed = 6) (h4 : c_speed = 2) :
  ∃ t : ℚ, t = 2 ∧ c_initial - b_initial - (b_speed + c_speed) * t = 8 :=
by sorry

end NUMINAMATH_CALUDE_point_meeting_time_l3206_320636


namespace NUMINAMATH_CALUDE_no_perfect_squares_l3206_320651

theorem no_perfect_squares (a b : ℕ) : 
  ¬(∃ (m n : ℕ), a^2 + b = m^2 ∧ b^2 + a = n^2) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l3206_320651


namespace NUMINAMATH_CALUDE_race_distance_is_140_l3206_320686

/-- The distance of a race, given the times of two runners and the difference in their finishing positions. -/
def race_distance (time_A time_B : ℕ) (difference : ℕ) : ℕ :=
  let speed_A := 140 / time_A
  let speed_B := 140 / time_B
  140

/-- Theorem stating that the race distance is 140 meters under the given conditions. -/
theorem race_distance_is_140 :
  race_distance 36 45 28 = 140 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_is_140_l3206_320686


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3206_320622

theorem lcm_from_product_and_hcf (A B : ℕ+) :
  A * B = 84942 →
  Nat.gcd A B = 33 →
  Nat.lcm A B = 2574 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3206_320622


namespace NUMINAMATH_CALUDE_function_range_equivalence_l3206_320610

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * a + 1

-- State the theorem
theorem function_range_equivalence (a : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Icc (-1) 1 ∧ y ∈ Set.Icc (-1) 1 ∧ f a x > 0 ∧ f a y < 0) ↔
  a ∈ Set.Ioo (-1) (-1/3) :=
sorry

end NUMINAMATH_CALUDE_function_range_equivalence_l3206_320610


namespace NUMINAMATH_CALUDE_pablo_puzzle_days_l3206_320682

def puzzles_400 : ℕ := 15
def pieces_per_400 : ℕ := 400
def puzzles_700 : ℕ := 10
def pieces_per_700 : ℕ := 700
def pieces_per_hour : ℕ := 100
def hours_per_day : ℕ := 6

def total_pieces : ℕ := puzzles_400 * pieces_per_400 + puzzles_700 * pieces_per_700

def total_hours : ℕ := (total_pieces + pieces_per_hour - 1) / pieces_per_hour

def days_required : ℕ := (total_hours + hours_per_day - 1) / hours_per_day

theorem pablo_puzzle_days : days_required = 22 := by
  sorry

end NUMINAMATH_CALUDE_pablo_puzzle_days_l3206_320682


namespace NUMINAMATH_CALUDE_billy_tickets_left_l3206_320658

theorem billy_tickets_left (tickets_won : ℕ) (difference : ℕ) (tickets_left : ℕ) : 
  tickets_won = 48 → 
  difference = 16 → 
  tickets_won - tickets_left = difference → 
  tickets_left = 32 := by
sorry

end NUMINAMATH_CALUDE_billy_tickets_left_l3206_320658


namespace NUMINAMATH_CALUDE_hyperbola_cosine_theorem_l3206_320696

/-- A hyperbola with equation x^2 - y^2 = 2 -/
def Hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_cosine_theorem :
  Hyperbola P.1 P.2 →
  distance P F₁ = 2 * distance P F₂ →
  let cosine_angle := (distance P F₁)^2 + (distance P F₂)^2 - (distance F₁ F₂)^2
                    / (2 * distance P F₁ * distance P F₂)
  cosine_angle = 3/4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_cosine_theorem_l3206_320696


namespace NUMINAMATH_CALUDE_factorial_sum_ratio_l3206_320648

theorem factorial_sum_ratio : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10) / 
  (1 * 2 + 3 * 4 + 5 * 6 + 7 * 8 + 9 * 10) = 19120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_ratio_l3206_320648


namespace NUMINAMATH_CALUDE_right_triangle_area_perimeter_relation_l3206_320630

theorem right_triangle_area_perimeter_relation (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  a * b = 3 * (a + b + c) →
  ((a = 7 ∧ b = 24 ∧ c = 25) ∨
   (a = 8 ∧ b = 15 ∧ c = 17) ∨
   (a = 9 ∧ b = 12 ∧ c = 15) ∨
   (b = 7 ∧ a = 24 ∧ c = 25) ∨
   (b = 8 ∧ a = 15 ∧ c = 17) ∨
   (b = 9 ∧ a = 12 ∧ c = 15)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_perimeter_relation_l3206_320630


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l3206_320601

/-- Given the selling price and profit per meter of cloth, calculate the cost price per meter. -/
theorem cost_price_per_meter
  (selling_price : ℚ)
  (cloth_length : ℚ)
  (profit_per_meter : ℚ)
  (h1 : selling_price = 8925)
  (h2 : cloth_length = 85)
  (h3 : profit_per_meter = 25) :
  (selling_price - cloth_length * profit_per_meter) / cloth_length = 80 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l3206_320601


namespace NUMINAMATH_CALUDE_largest_product_of_three_l3206_320640

def S : Finset Int := {-5, -4, -1, 2, 6}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a * b * c ≤ 120 :=
sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l3206_320640


namespace NUMINAMATH_CALUDE_circles_tangent_implies_a_eq_plus_minus_one_l3206_320681

/-- Circle E with equation x^2 + y^2 = 4 -/
def circle_E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- Circle F with equation x^2 + (y-a)^2 = 1, parameterized by a -/
def circle_F (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - a)^2 = 1}

/-- Two circles are internally tangent if they have exactly one point in common -/
def internally_tangent (C1 C2 : Set (ℝ × ℝ)) : Prop :=
  ∃! p : ℝ × ℝ, p ∈ C1 ∧ p ∈ C2

/-- Main theorem: If circles E and F are internally tangent, then a = ±1 -/
theorem circles_tangent_implies_a_eq_plus_minus_one (a : ℝ) :
  internally_tangent (circle_E) (circle_F a) → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_implies_a_eq_plus_minus_one_l3206_320681


namespace NUMINAMATH_CALUDE_unique_obtainable_pair_l3206_320663

-- Define the calculator operations
def calc_op1 (p : ℕ × ℕ) : ℕ × ℕ := (p.1 + p.2, p.1)
def calc_op2 (p : ℕ × ℕ) : ℕ × ℕ := (2 * p.1 + p.2 + 1, p.1 + p.2 + 1)

-- Define a predicate for pairs obtainable by the calculator
inductive Obtainable : ℕ × ℕ → Prop where
  | initial : Obtainable (1, 1)
  | op1 {p : ℕ × ℕ} : Obtainable p → Obtainable (calc_op1 p)
  | op2 {p : ℕ × ℕ} : Obtainable p → Obtainable (calc_op2 p)

-- State the theorem
theorem unique_obtainable_pair :
  ∀ n : ℕ, ∃! k : ℕ, Obtainable (n, k) :=
sorry

end NUMINAMATH_CALUDE_unique_obtainable_pair_l3206_320663


namespace NUMINAMATH_CALUDE_w_squared_value_l3206_320665

theorem w_squared_value (w : ℚ) (h : (w + 16)^2 = (4*w + 9)*(3*w + 6)) : 
  w^2 = 5929 / 484 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l3206_320665


namespace NUMINAMATH_CALUDE_pens_per_student_after_split_l3206_320629

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of red pens each student initially received --/
def red_pens_per_student : ℕ := 62

/-- The number of black pens each student initially received --/
def black_pens_per_student : ℕ := 43

/-- The total number of pens taken after the first month --/
def pens_taken_first_month : ℕ := 37

/-- The total number of pens taken after the second month --/
def pens_taken_second_month : ℕ := 41

/-- Theorem stating that each student will receive 79 pens when the remaining pens are split equally --/
theorem pens_per_student_after_split : 
  let total_pens := num_students * (red_pens_per_student + black_pens_per_student)
  let remaining_pens := total_pens - pens_taken_first_month - pens_taken_second_month
  remaining_pens / num_students = 79 := by
  sorry


end NUMINAMATH_CALUDE_pens_per_student_after_split_l3206_320629


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3206_320652

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt 3 = Real.sqrt (3^x * 3^y) → 1/x + 1/y ≥ 1/a + 1/b) →
  1/a + 1/b = 4 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3206_320652


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3206_320662

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((7 / x^2) - 5 * x^3) = 3 / x^2 - 15 * x^3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3206_320662


namespace NUMINAMATH_CALUDE_octal_734_equals_decimal_476_l3206_320615

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

/-- The octal number 734 is equal to 476 in decimal --/
theorem octal_734_equals_decimal_476 : octal_to_decimal 734 = 476 := by
  sorry

end NUMINAMATH_CALUDE_octal_734_equals_decimal_476_l3206_320615


namespace NUMINAMATH_CALUDE_group_size_l3206_320624

theorem group_size (num_children : ℕ) (num_women : ℕ) (num_men : ℕ) : 
  num_children = 30 →
  num_women = 3 * num_children →
  num_men = 2 * num_women →
  num_children + num_women + num_men = 300 := by
  sorry

#check group_size

end NUMINAMATH_CALUDE_group_size_l3206_320624


namespace NUMINAMATH_CALUDE_specific_box_volume_l3206_320679

/-- The volume of an open box constructed from a rectangular sheet of metal -/
def box_volume (length width x : ℝ) : ℝ :=
  (length - 2*x) * (width - 2*x) * x

/-- Theorem: The volume of the specific box described in the problem -/
theorem specific_box_volume (x : ℝ) :
  box_volume 16 12 x = 4*x^3 - 56*x^2 + 192*x :=
by sorry

end NUMINAMATH_CALUDE_specific_box_volume_l3206_320679


namespace NUMINAMATH_CALUDE_committee_formation_l3206_320641

theorem committee_formation (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_l3206_320641


namespace NUMINAMATH_CALUDE_network_connections_l3206_320676

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l3206_320676


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l3206_320617

theorem unique_two_digit_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (∃ q r : ℕ, n = q * (10 * (n % 10) + n / 10) + r ∧ q = 4 ∧ r = 3) ∧
  (∃ q r : ℕ, n = q * (n / 10 + n % 10) + r ∧ q = 8 ∧ r = 7) ∧
  n = 71 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l3206_320617


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3206_320650

theorem infinite_series_sum : 
  let a : ℕ → ℚ := λ n => (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)
  ∑' n, a n = 1 / 800 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3206_320650


namespace NUMINAMATH_CALUDE_toms_weekly_fluid_intake_l3206_320637

/-- Calculates the total fluid intake in ounces for a week given daily soda and water consumption --/
def weekly_fluid_intake (soda_cans : ℕ) (oz_per_can : ℕ) (water_oz : ℕ) : ℕ :=
  7 * (soda_cans * oz_per_can + water_oz)

/-- Theorem stating Tom's weekly fluid intake --/
theorem toms_weekly_fluid_intake :
  weekly_fluid_intake 5 12 64 = 868 := by
  sorry

end NUMINAMATH_CALUDE_toms_weekly_fluid_intake_l3206_320637


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l3206_320660

theorem triangle_side_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) = b / (a + c) + c / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l3206_320660


namespace NUMINAMATH_CALUDE_identity_equals_one_l3206_320643

theorem identity_equals_one (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (x - b) * (x - c) / ((a - b) * (a - c)) +
  (x - c) * (x - a) / ((b - c) * (b - a)) +
  (x - a) * (x - b) / ((c - a) * (c - b)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_identity_equals_one_l3206_320643


namespace NUMINAMATH_CALUDE_expression_evaluation_l3206_320695

theorem expression_evaluation :
  let a : ℚ := -1/2
  let b : ℚ := 3
  3 * a^2 - b^2 - (a^2 - 6*a) - 2*(-b^2 + 3*a) = 19/2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3206_320695


namespace NUMINAMATH_CALUDE_alice_win_probability_l3206_320644

/-- Represents a player in the tournament -/
inductive Player
| Alice
| Bob
| Other

/-- Represents a move in rock-paper-scissors -/
inductive Move
| Rock
| Paper
| Scissors

/-- The number of players in the tournament -/
def numPlayers : Nat := 8

/-- The number of rounds in the tournament -/
def numRounds : Nat := 3

/-- Returns the move of a given player -/
def playerMove (p : Player) : Move :=
  match p with
  | Player.Alice => Move.Rock
  | Player.Bob => Move.Paper
  | Player.Other => Move.Scissors

/-- Determines the winner of a match between two players -/
def matchWinner (p1 p2 : Player) : Player :=
  match playerMove p1, playerMove p2 with
  | Move.Rock, Move.Scissors => p1
  | Move.Scissors, Move.Paper => p1
  | Move.Paper, Move.Rock => p1
  | Move.Scissors, Move.Rock => p2
  | Move.Paper, Move.Scissors => p2
  | Move.Rock, Move.Paper => p2
  | _, _ => p1  -- In case of a tie, p1 wins (representing a coin flip)

/-- The probability of Alice winning the tournament -/
def aliceWinProbability : Rat := 6/7

theorem alice_win_probability :
  aliceWinProbability = 6/7 := by sorry


end NUMINAMATH_CALUDE_alice_win_probability_l3206_320644


namespace NUMINAMATH_CALUDE_simplify_radicals_l3206_320616

theorem simplify_radicals : 
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 + Real.sqrt 160 = 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l3206_320616


namespace NUMINAMATH_CALUDE_new_apples_grown_l3206_320611

theorem new_apples_grown (initial_apples picked_apples current_apples : ℕ) 
  (h1 : initial_apples = 11)
  (h2 : picked_apples = 7)
  (h3 : current_apples = 6) :
  current_apples - (initial_apples - picked_apples) = 2 :=
by sorry

end NUMINAMATH_CALUDE_new_apples_grown_l3206_320611


namespace NUMINAMATH_CALUDE_max_stone_value_l3206_320691

/-- Represents the types of stones --/
inductive StoneType
| FivePound
| FourPound
| OnePound

/-- Returns the weight of a stone type in pounds --/
def weight (s : StoneType) : ℕ :=
  match s with
  | StoneType.FivePound => 5
  | StoneType.FourPound => 4
  | StoneType.OnePound => 1

/-- Returns the value of a stone type in dollars --/
def value (s : StoneType) : ℕ :=
  match s with
  | StoneType.FivePound => 14
  | StoneType.FourPound => 11
  | StoneType.OnePound => 2

/-- Represents a combination of stones --/
structure StoneCombination where
  fivePound : ℕ
  fourPound : ℕ
  onePound : ℕ

/-- Calculates the total weight of a stone combination --/
def totalWeight (c : StoneCombination) : ℕ :=
  c.fivePound * weight StoneType.FivePound +
  c.fourPound * weight StoneType.FourPound +
  c.onePound * weight StoneType.OnePound

/-- Calculates the total value of a stone combination --/
def totalValue (c : StoneCombination) : ℕ :=
  c.fivePound * value StoneType.FivePound +
  c.fourPound * value StoneType.FourPound +
  c.onePound * value StoneType.OnePound

/-- Defines a valid stone combination --/
def isValidCombination (c : StoneCombination) : Prop :=
  totalWeight c ≤ 18 ∧ c.fivePound ≤ 20 ∧ c.fourPound ≤ 20 ∧ c.onePound ≤ 20

theorem max_stone_value :
  ∃ (c : StoneCombination), isValidCombination c ∧
    totalValue c = 50 ∧
    ∀ (c' : StoneCombination), isValidCombination c' → totalValue c' ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_max_stone_value_l3206_320691


namespace NUMINAMATH_CALUDE_required_weekly_hours_l3206_320656

/-- Calculates the required weekly work hours to meet a financial goal given previous work data and future plans. -/
theorem required_weekly_hours 
  (summer_weeks : ℕ) 
  (summer_hours_per_week : ℕ) 
  (summer_total_earnings : ℚ) 
  (future_weeks : ℕ) 
  (future_earnings_goal : ℚ) : 
  summer_weeks > 0 ∧ 
  summer_hours_per_week > 0 ∧ 
  summer_total_earnings > 0 ∧ 
  future_weeks > 0 ∧ 
  future_earnings_goal > 0 →
  (future_earnings_goal / (summer_total_earnings / (summer_weeks * summer_hours_per_week))) / future_weeks = 45 / 16 := by
  sorry

#eval (4500 : ℚ) / ((3600 : ℚ) / (8 * 45)) / 40

end NUMINAMATH_CALUDE_required_weekly_hours_l3206_320656


namespace NUMINAMATH_CALUDE_factorization_equality_l3206_320638

theorem factorization_equality (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) := by
sorry

end NUMINAMATH_CALUDE_factorization_equality_l3206_320638


namespace NUMINAMATH_CALUDE_dishwasher_manager_wage_ratio_l3206_320642

/-- Proves that the ratio of a dishwasher's hourly wage to a manager's hourly wage is 0.5 -/
theorem dishwasher_manager_wage_ratio :
  ∀ (manager_wage chef_wage dishwasher_wage : ℝ),
    manager_wage = 8.5 →
    chef_wage = manager_wage - 3.4 →
    chef_wage = dishwasher_wage * 1.2 →
    dishwasher_wage / manager_wage = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_dishwasher_manager_wage_ratio_l3206_320642


namespace NUMINAMATH_CALUDE_total_money_l3206_320688

/-- The amount of money Beth has -/
def beth_money : ℕ := 70

/-- The amount of money Jan has -/
def jan_money : ℕ := 80

/-- The condition that if Beth had $35 more, she would have $105 -/
axiom beth_condition : beth_money + 35 = 105

/-- The condition that if Jan had $10 less, he would have the same money as Beth -/
axiom jan_condition : jan_money - 10 = beth_money

/-- The theorem stating that Beth and Jan have $150 altogether -/
theorem total_money : beth_money + jan_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3206_320688


namespace NUMINAMATH_CALUDE_root_equation_value_l3206_320661

theorem root_equation_value (m : ℝ) : 
  (2 * m^2 + 3 * m - 1 = 0) → (4 * m^2 + 6 * m - 2019 = -2017) := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3206_320661


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l3206_320673

theorem sum_of_digits_of_large_number : ∃ S : ℕ, 
  S = 10^2021 - 2021 ∧ 
  (∃ digits : List ℕ, 
    digits.sum = 18185 ∧ 
    digits.all (λ d => d < 10) ∧
    S = digits.foldr (λ d acc => d + 10 * acc) 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l3206_320673


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3206_320694

-- Define the Cartesian coordinate system
def Cartesian := ℝ × ℝ

-- Define a point in the Cartesian coordinate system
def point : Cartesian := (1, -2)

-- Define the fourth quadrant
def fourth_quadrant (p : Cartesian) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant :
  fourth_quadrant point := by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3206_320694
