import Mathlib

namespace NUMINAMATH_CALUDE_min_area_special_square_l1655_165558

/-- A square with one side on y = 2x - 17 and two vertices on y = x^2 -/
structure SpecialSquare where
  /-- Side length of the square -/
  a : ℝ
  /-- Parameter for the line y = 2x + b passing through two vertices on the parabola -/
  b : ℝ
  /-- The square has one side on y = 2x - 17 -/
  side_on_line : a = (17 + b) / Real.sqrt 5
  /-- Two vertices of the square are on y = x^2 -/
  vertices_on_parabola : a^2 = 20 * (1 + b)

/-- The minimum area of a SpecialSquare is 80 -/
theorem min_area_special_square :
  ∀ s : SpecialSquare, s.a^2 ≥ 80 := by
  sorry

#check min_area_special_square

end NUMINAMATH_CALUDE_min_area_special_square_l1655_165558


namespace NUMINAMATH_CALUDE_functional_equation_bijection_l1655_165505

theorem functional_equation_bijection :
  ∃ f : ℕ → ℕ, Function.Bijective f ∧
    ∀ m n : ℕ, f (3*m*n + m + n) = 4*f m*f n + f m + f n :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_bijection_l1655_165505


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l1655_165565

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem unique_solution_factorial_equation : 
  ∃! n : ℕ, n * factorial n - factorial n = 5040 - factorial n :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l1655_165565


namespace NUMINAMATH_CALUDE_original_stones_count_l1655_165583

/-- The number of stones sent away to the Geological Museum in London. -/
def stones_sent_away : ℕ := 63

/-- The number of stones kept in the collection. -/
def stones_kept : ℕ := 15

/-- The original number of stones in the collection. -/
def original_stones : ℕ := stones_sent_away + stones_kept

/-- Theorem stating that the original number of stones in the collection is 78. -/
theorem original_stones_count : original_stones = 78 := by
  sorry

end NUMINAMATH_CALUDE_original_stones_count_l1655_165583


namespace NUMINAMATH_CALUDE_sqrt_five_irrational_and_greater_than_two_l1655_165552

theorem sqrt_five_irrational_and_greater_than_two :
  ∃ x : ℝ, Irrational x ∧ x > 2 ∧ x = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_five_irrational_and_greater_than_two_l1655_165552


namespace NUMINAMATH_CALUDE_star_three_five_l1655_165595

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

-- Theorem statement
theorem star_three_five : star 3 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_star_three_five_l1655_165595


namespace NUMINAMATH_CALUDE_smallest_square_side_is_14_l1655_165588

/-- The smallest side length of a square composed of equal numbers of unit squares with sides 1, 2, and 3 -/
def smallest_square_side : ℕ := 14

/-- Proposition: The smallest possible side length of a square composed of an equal number of squares with sides 1, 2, and 3 is 14 units -/
theorem smallest_square_side_is_14 :
  ∀ n : ℕ, n > 0 →
  ∃ s : ℕ, s * s = n * (1 * 1 + 2 * 2 + 3 * 3) →
  s ≥ smallest_square_side :=
sorry

end NUMINAMATH_CALUDE_smallest_square_side_is_14_l1655_165588


namespace NUMINAMATH_CALUDE_crazy_silly_school_movies_l1655_165549

/-- The number of remaining movies to watch in the 'crazy silly school' series -/
def remaining_movies (total : ℕ) (watched : ℕ) : ℕ :=
  total - watched

theorem crazy_silly_school_movies : 
  remaining_movies 17 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_movies_l1655_165549


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1655_165597

/-- Given a triangle ABC with vertices A(-3, 5), B(3, -3), and midpoint M(6, 1) of side BC,
    prove that the perimeter of the triangle is 32. -/
theorem triangle_perimeter (A B C M : ℝ × ℝ) : 
  A = (-3, 5) → B = (3, -3) → M = (6, 1) → 
  M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB + BC + AC = 32 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1655_165597


namespace NUMINAMATH_CALUDE_complex_rational_equation_root_l1655_165535

theorem complex_rational_equation_root :
  ∃! x : ℚ, (3*x^2 + 5)/(x-2) - (3*x + 10)/4 + (5 - 9*x)/(x-2) + 2 = 0 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_rational_equation_root_l1655_165535


namespace NUMINAMATH_CALUDE_vitamin_c_content_l1655_165525

/-- The amount of vitamin C (in mg) in one 8-oz glass of apple juice -/
def apple_juice_vc : ℕ := 103

/-- The total amount of vitamin C (in mg) in one 8-oz glass each of apple juice and orange juice -/
def total_vc : ℕ := 185

/-- The amount of vitamin C (in mg) in one 8-oz glass of orange juice -/
def orange_juice_vc : ℕ := total_vc - apple_juice_vc

/-- Theorem: Two 8-oz glasses of apple juice and three 8-oz glasses of orange juice contain 452 mg of vitamin C -/
theorem vitamin_c_content : 2 * apple_juice_vc + 3 * orange_juice_vc = 452 := by
  sorry

end NUMINAMATH_CALUDE_vitamin_c_content_l1655_165525


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1655_165575

theorem opposite_of_negative_2023 : 
  -((-2023) : ℤ) = (2023 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1655_165575


namespace NUMINAMATH_CALUDE_unique_triple_l1655_165572

theorem unique_triple : ∃! (a b c : ℤ), 
  a > 0 ∧ 0 > b ∧ b > c ∧ 
  a + b + c = 0 ∧ 
  ∃ (k : ℤ), 2017 - a^3*b - b^3*c - c^3*a = k^2 ∧
  a = 36 ∧ b = -12 ∧ c = -24 :=
sorry

end NUMINAMATH_CALUDE_unique_triple_l1655_165572


namespace NUMINAMATH_CALUDE_percentage_loss_l1655_165553

def cost_price : ℝ := 1800
def selling_price : ℝ := 1350

theorem percentage_loss : (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_loss_l1655_165553


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l1655_165500

theorem cubic_equation_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b*x + 25 = 0 := by
  sorry

#check cubic_equation_real_root

end NUMINAMATH_CALUDE_cubic_equation_real_root_l1655_165500


namespace NUMINAMATH_CALUDE_compute_expression_l1655_165587

theorem compute_expression : 5 + 7 * (2 - 9)^2 = 348 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1655_165587


namespace NUMINAMATH_CALUDE_grandson_age_l1655_165564

/-- Given the ages of three family members satisfying certain conditions,
    prove that the youngest member (grandson) is 20 years old. -/
theorem grandson_age (grandson_age son_age markus_age : ℕ) : 
  son_age = 2 * grandson_age →
  markus_age = 2 * son_age →
  grandson_age + son_age + markus_age = 140 →
  grandson_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_grandson_age_l1655_165564


namespace NUMINAMATH_CALUDE_rational_function_identity_l1655_165519

theorem rational_function_identity (x : ℝ) (h1 : x ≠ 2) (h2 : x^2 + x + 1 ≠ 0) :
  (x + 3)^2 / ((x - 2) * (x^2 + x + 1)) = 
  25 / (7 * (x - 2)) + (-18 * x - 19) / (7 * (x^2 + x + 1)) := by
  sorry

#check rational_function_identity

end NUMINAMATH_CALUDE_rational_function_identity_l1655_165519


namespace NUMINAMATH_CALUDE_triangle_area_l1655_165518

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) :
  (1/2) * a * b = 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1655_165518


namespace NUMINAMATH_CALUDE_cubic_root_identity_l1655_165581

theorem cubic_root_identity (a b c : ℂ) (n m : ℕ) :
  (∃ x : ℂ, x^3 = 1 ∧ a * x^(3*n + 2) + b * x^(3*m + 1) + c = 0) →
  a^3 + b^3 + c^3 - 3*a*b*c = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_identity_l1655_165581


namespace NUMINAMATH_CALUDE_derivative_of_f_at_2_l1655_165513

-- Define the function f(x) = x
def f (x : ℝ) : ℝ := x

-- State the theorem
theorem derivative_of_f_at_2 : 
  HasDerivAt f 1 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_at_2_l1655_165513


namespace NUMINAMATH_CALUDE_change_received_correct_l1655_165539

/-- Calculates the change received when buying steak -/
def change_received (cost_per_pound : ℝ) (pounds_bought : ℝ) (amount_paid : ℝ) : ℝ :=
  amount_paid - (cost_per_pound * pounds_bought)

/-- Theorem: The change received when buying steak is correct -/
theorem change_received_correct (cost_per_pound : ℝ) (pounds_bought : ℝ) (amount_paid : ℝ) :
  change_received cost_per_pound pounds_bought amount_paid =
  amount_paid - (cost_per_pound * pounds_bought) := by
  sorry

#eval change_received 7 2 20

end NUMINAMATH_CALUDE_change_received_correct_l1655_165539


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1655_165573

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1655_165573


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1655_165550

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1655_165550


namespace NUMINAMATH_CALUDE_brittany_age_after_vacation_l1655_165534

/-- Represents a person with an age --/
structure Person where
  age : ℕ

/-- Represents a vacation --/
structure Vacation where
  duration : ℕ
  birthdaysCelebrated : ℕ
  hasLeapYear : Bool

/-- Calculates the age of a person after a vacation --/
def ageAfterVacation (person : Person) (vacation : Vacation) : ℕ :=
  person.age + vacation.birthdaysCelebrated

theorem brittany_age_after_vacation (rebecca : Person) (brittany : Person) (vacation : Vacation) :
  rebecca.age = 25 →
  brittany.age = rebecca.age + 3 →
  vacation.duration = 4 →
  vacation.birthdaysCelebrated = 3 →
  vacation.hasLeapYear = true →
  ageAfterVacation brittany vacation = 31 := by
  sorry

#eval ageAfterVacation (Person.mk 28) (Vacation.mk 4 3 true)

end NUMINAMATH_CALUDE_brittany_age_after_vacation_l1655_165534


namespace NUMINAMATH_CALUDE_amc10_paths_count_l1655_165596

/-- Represents the grid structure for spelling "AMC10" --/
structure AMC10Grid where
  a_to_m : Nat  -- Number of 'M's adjacent to central 'A'
  m_to_c : Nat  -- Number of 'C's adjacent to each 'M' (excluding path back to 'A')
  c_to_10 : Nat -- Number of '10' blocks adjacent to each 'C'

/-- Calculates the number of paths to spell "AMC10" in the given grid --/
def count_paths (grid : AMC10Grid) : Nat :=
  grid.a_to_m * grid.m_to_c * grid.c_to_10

/-- The specific grid configuration for the problem --/
def problem_grid : AMC10Grid :=
  { a_to_m := 4, m_to_c := 3, c_to_10 := 1 }

/-- Theorem stating that the number of paths to spell "AMC10" in the problem grid is 12 --/
theorem amc10_paths_count :
  count_paths problem_grid = 12 := by
  sorry

end NUMINAMATH_CALUDE_amc10_paths_count_l1655_165596


namespace NUMINAMATH_CALUDE_min_shipping_cost_l1655_165559

/-- Represents the shipping problem with given stock, demand, and costs. -/
structure ShippingProblem where
  shanghai_stock : ℕ
  nanjing_stock : ℕ
  suzhou_demand : ℕ
  changsha_demand : ℕ
  cost_shanghai_suzhou : ℕ
  cost_shanghai_changsha : ℕ
  cost_nanjing_suzhou : ℕ
  cost_nanjing_changsha : ℕ

/-- Calculates the total shipping cost given the number of units shipped from Shanghai to Suzhou. -/
def total_cost (problem : ShippingProblem) (x : ℕ) : ℕ :=
  problem.cost_shanghai_suzhou * x +
  problem.cost_shanghai_changsha * (problem.shanghai_stock - x) +
  problem.cost_nanjing_suzhou * (problem.suzhou_demand - x) +
  problem.cost_nanjing_changsha * (x - (problem.suzhou_demand - problem.nanjing_stock))

/-- Theorem stating that the minimum shipping cost is 8600 yuan for the given problem. -/
theorem min_shipping_cost (problem : ShippingProblem) 
  (h1 : problem.shanghai_stock = 12)
  (h2 : problem.nanjing_stock = 6)
  (h3 : problem.suzhou_demand = 10)
  (h4 : problem.changsha_demand = 8)
  (h5 : problem.cost_shanghai_suzhou = 400)
  (h6 : problem.cost_shanghai_changsha = 800)
  (h7 : problem.cost_nanjing_suzhou = 300)
  (h8 : problem.cost_nanjing_changsha = 500) :
  ∃ x : ℕ, x ≥ 4 ∧ x ≤ 10 ∧ total_cost problem x = 8600 ∧ 
  ∀ y : ℕ, y ≥ 4 → y ≤ 10 → total_cost problem y ≥ total_cost problem x :=
sorry

end NUMINAMATH_CALUDE_min_shipping_cost_l1655_165559


namespace NUMINAMATH_CALUDE_fair_lines_theorem_l1655_165532

/-- Represents the number of people in the bumper cars line -/
def bumper_cars_line (initial : ℕ) (left : ℕ) (joined : ℕ) : ℕ :=
  initial - left + joined

/-- Represents the total number of people in both lines -/
def total_people (bumper_cars : ℕ) (roller_coaster : ℕ) : ℕ :=
  bumper_cars + roller_coaster

theorem fair_lines_theorem (x y Z : ℕ) (h1 : Z = bumper_cars_line 25 x y) 
  (h2 : Z ≥ x) : total_people Z 15 = 40 - x + y := by
  sorry

#check fair_lines_theorem

end NUMINAMATH_CALUDE_fair_lines_theorem_l1655_165532


namespace NUMINAMATH_CALUDE_joes_dad_marshmallows_joes_dad_marshmallows_proof_l1655_165509

theorem joes_dad_marshmallows : ℕ → Prop :=
  fun d : ℕ =>
    let joe_marshmallows : ℕ := 4 * d
    let dad_roasted : ℕ := d / 3
    let joe_roasted : ℕ := joe_marshmallows / 2
    dad_roasted + joe_roasted = 49 → d = 21

-- The proof goes here
theorem joes_dad_marshmallows_proof : joes_dad_marshmallows 21 := by
  sorry

end NUMINAMATH_CALUDE_joes_dad_marshmallows_joes_dad_marshmallows_proof_l1655_165509


namespace NUMINAMATH_CALUDE_wendy_picture_upload_l1655_165586

/-- The number of pictures Wendy uploaded to Facebook -/
def total_pictures : ℕ := 79

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 44

/-- The number of additional albums -/
def additional_albums : ℕ := 5

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 7

/-- Theorem stating that the total number of pictures is correct -/
theorem wendy_picture_upload :
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album :=
by sorry

end NUMINAMATH_CALUDE_wendy_picture_upload_l1655_165586


namespace NUMINAMATH_CALUDE_enrollment_difference_l1655_165590

def maple_ridge_enrollment : ℕ := 1500
def south_park_enrollment : ℕ := 2100
def lakeside_enrollment : ℕ := 2700
def riverdale_enrollment : ℕ := 1800
def brookwood_enrollment : ℕ := 900

def school_enrollments : List ℕ := [
  maple_ridge_enrollment,
  south_park_enrollment,
  lakeside_enrollment,
  riverdale_enrollment,
  brookwood_enrollment
]

theorem enrollment_difference : 
  (List.maximum school_enrollments).get! - (List.minimum school_enrollments).get! = 1800 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_difference_l1655_165590


namespace NUMINAMATH_CALUDE_power_of_two_in_factorial_eight_l1655_165533

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem power_of_two_in_factorial_eight :
  ∀ i k m p : ℕ,
  i > 0 → k > 0 → m > 0 → p > 0 →
  factorial 8 = 2^i * 3^k * 5^m * 7^p →
  i + k + m + p = 11 →
  i = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_power_of_two_in_factorial_eight_l1655_165533


namespace NUMINAMATH_CALUDE_school_distance_is_two_point_five_l1655_165507

/-- The distance from Philip's house to the school in miles -/
def school_distance : ℝ := sorry

/-- The round trip distance to the market in miles -/
def market_round_trip : ℝ := 4

/-- The number of round trips to school per week -/
def school_trips_per_week : ℕ := 8

/-- The number of round trips to the market per week -/
def market_trips_per_week : ℕ := 1

/-- The total mileage for a typical week in miles -/
def total_weekly_mileage : ℝ := 44

theorem school_distance_is_two_point_five : 
  school_distance = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_school_distance_is_two_point_five_l1655_165507


namespace NUMINAMATH_CALUDE_number_equation_l1655_165560

theorem number_equation (x : ℝ) : x - 2 + 4 = 9 ↔ x = 7 := by sorry

end NUMINAMATH_CALUDE_number_equation_l1655_165560


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l1655_165556

theorem inequality_and_minimum_value (a b c : ℝ) 
  (ha : 1 < a ∧ a < Real.sqrt 7)
  (hb : 1 < b ∧ b < Real.sqrt 7)
  (hc : 1 < c ∧ c < Real.sqrt 7) :
  (1 / (a^2 - 1) + 1 / (7 - a^2) ≥ 2/3) ∧
  (1 / Real.sqrt ((a^2 - 1) * (7 - b^2)) + 
   1 / Real.sqrt ((b^2 - 1) * (7 - c^2)) + 
   1 / Real.sqrt ((c^2 - 1) * (7 - a^2)) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l1655_165556


namespace NUMINAMATH_CALUDE_max_concert_tickets_l1655_165576

theorem max_concert_tickets (ticket_cost : ℚ) (available_money : ℚ) : 
  ticket_cost = 15 → available_money = 120 → 
  (∃ (n : ℕ), n * ticket_cost ≤ available_money ∧ 
    ∀ (m : ℕ), m * ticket_cost ≤ available_money → m ≤ n) → 
  (∃ (max_tickets : ℕ), max_tickets = 8) :=
by sorry

end NUMINAMATH_CALUDE_max_concert_tickets_l1655_165576


namespace NUMINAMATH_CALUDE_ln_inequality_implies_inequality_l1655_165599

theorem ln_inequality_implies_inequality (a b : ℝ) : 
  Real.log a > Real.log b → a > b := by sorry

end NUMINAMATH_CALUDE_ln_inequality_implies_inequality_l1655_165599


namespace NUMINAMATH_CALUDE_xyz_value_l1655_165530

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11)
  (h3 : x + y + z = 3) : 
  x * y * z = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1655_165530


namespace NUMINAMATH_CALUDE_ground_beef_cost_l1655_165511

/-- The price of ground beef per kilogram in dollars -/
def price_per_kg : ℝ := 5.00

/-- The quantity of ground beef in kilograms -/
def quantity : ℝ := 12

/-- The total cost of ground beef -/
def total_cost : ℝ := price_per_kg * quantity

theorem ground_beef_cost : total_cost = 60.00 := by
  sorry

end NUMINAMATH_CALUDE_ground_beef_cost_l1655_165511


namespace NUMINAMATH_CALUDE_triangle_property_l1655_165536

theorem triangle_property (A B C : Real) (h_triangle : A + B + C = Real.pi) 
  (h_condition : Real.sin A * Real.cos A = Real.sin B * Real.cos B) :
  (A = B ∨ C = Real.pi / 2) ∨ (B = C ∨ A = Real.pi / 2) ∨ (C = A ∨ B = Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_property_l1655_165536


namespace NUMINAMATH_CALUDE_largest_n_for_product_1764_l1655_165563

/-- Represents an arithmetic sequence with integer terms -/
structure ArithmeticSequence where
  firstTerm : ℤ
  commonDifference : ℤ

/-- The n-th term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.firstTerm + (n - 1 : ℤ) * seq.commonDifference

theorem largest_n_for_product_1764 (c d : ArithmeticSequence)
    (h1 : c.firstTerm = 1)
    (h2 : d.firstTerm = 1)
    (h3 : nthTerm c 2 ≤ nthTerm d 2)
    (h4 : ∃ n : ℕ, nthTerm c n * nthTerm d n = 1764) :
    (∃ n : ℕ, nthTerm c n * nthTerm d n = 1764) ∧
    (∀ m : ℕ, nthTerm c m * nthTerm d m = 1764 → m ≤ 1764) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_product_1764_l1655_165563


namespace NUMINAMATH_CALUDE_line_through_first_third_quadrants_l1655_165561

/-- A line y = kx passes through the first and third quadrants if and only if k > 0 -/
theorem line_through_first_third_quadrants (k : ℝ) (h1 : k ≠ 0) :
  (∀ x y : ℝ, y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) ↔ k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_first_third_quadrants_l1655_165561


namespace NUMINAMATH_CALUDE_other_divisor_proof_l1655_165501

theorem other_divisor_proof (x : ℕ) (h : x > 0) : 
  (261 % 37 = 2 ∧ 261 % x = 2) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_other_divisor_proof_l1655_165501


namespace NUMINAMATH_CALUDE_set_S_properties_l1655_165598

def S (m n : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ n}

theorem set_S_properties (m n : ℝ) (h_nonempty : (S m n).Nonempty) 
  (h_square : ∀ x ∈ S m n, x^2 ∈ S m n) :
  (m = -1/2 → 1/4 ≤ n ∧ n ≤ 1) ∧
  (n = 1/2 → -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_set_S_properties_l1655_165598


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l1655_165541

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ : ℝ} : 
  (∃ (b₁ b₂ : ℝ), ∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) → m₁ = m₂

/-- Given two lines 3y - 3b = 9x and y - 2 = (b + 9)x that are parallel, prove b = -6 -/
theorem parallel_lines_b_value (b : ℝ) :
  (∃ (y₁ y₂ : ℝ → ℝ), (∀ x, 3 * y₁ x - 3 * b = 9 * x) ∧ 
                       (∀ x, y₂ x - 2 = (b + 9) * x) ∧
                       (∀ x y, y = y₁ x ↔ y = y₂ x)) →
  b = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l1655_165541


namespace NUMINAMATH_CALUDE_second_team_cups_l1655_165502

def total_required : ℕ := 280
def first_team : ℕ := 90
def third_team : ℕ := 70

theorem second_team_cups : total_required - first_team - third_team = 120 := by
  sorry

end NUMINAMATH_CALUDE_second_team_cups_l1655_165502


namespace NUMINAMATH_CALUDE_roberto_healthcare_contribution_l1655_165526

/-- Calculates the healthcare contribution in cents per hour given an hourly wage in dollars and a contribution rate. -/
def healthcare_contribution (hourly_wage : ℚ) (contribution_rate : ℚ) : ℚ :=
  hourly_wage * 100 * contribution_rate

/-- Proves that Roberto's healthcare contribution is 50 cents per hour. -/
theorem roberto_healthcare_contribution :
  healthcare_contribution 25 (2/100) = 50 := by
  sorry

#eval healthcare_contribution 25 (2/100)

end NUMINAMATH_CALUDE_roberto_healthcare_contribution_l1655_165526


namespace NUMINAMATH_CALUDE_regular_polygon_angle_relation_l1655_165512

theorem regular_polygon_angle_relation : 
  ∀ n : ℕ, 
  n ≥ 3 →
  (360 / n : ℚ) = (120 / 5 : ℚ) →
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_relation_l1655_165512


namespace NUMINAMATH_CALUDE_binary_remainder_by_four_l1655_165543

theorem binary_remainder_by_four (n : Nat) :
  n = 0b101110011100 → n % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_binary_remainder_by_four_l1655_165543


namespace NUMINAMATH_CALUDE_probability_theorem_l1655_165523

def total_marbles : ℕ := 30
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10
def green_marbles : ℕ := 5
def marbles_selected : ℕ := 4

def probability_two_red_one_blue_one_green : ℚ :=
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1) /
  Nat.choose total_marbles marbles_selected

theorem probability_theorem :
  probability_two_red_one_blue_one_green = 350 / 1827 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1655_165523


namespace NUMINAMATH_CALUDE_largest_prime_factor_l1655_165537

def numbers : List Nat := [55, 63, 95, 133, 143]

theorem largest_prime_factor :
  ∃ (n : Nat), n ∈ numbers ∧ 19 ∣ n ∧
  ∀ (m : Nat), m ∈ numbers → ∀ (p : Nat), Prime p → p ∣ m → p ≤ 19 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l1655_165537


namespace NUMINAMATH_CALUDE_faye_points_l1655_165510

/-- Given a team of 5 players who scored 68 points in total, where 4 players scored 8 points each,
    prove that the remaining player (Faye) scored 36 points. -/
theorem faye_points (total_points : ℕ) (team_size : ℕ) (others_points : ℕ) 
  (h1 : total_points = 68)
  (h2 : team_size = 5)
  (h3 : others_points = 8)
  : total_points - (team_size - 1) * others_points = 36 := by
  sorry

end NUMINAMATH_CALUDE_faye_points_l1655_165510


namespace NUMINAMATH_CALUDE_function_symmetry_l1655_165515

theorem function_symmetry (a : ℝ) : 
  let f : ℝ → ℝ := λ x => (x^2 + x + 1) / (x^2 + 1)
  f a = 2/3 → f (-a) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_l1655_165515


namespace NUMINAMATH_CALUDE_percentage_problem_l1655_165540

/-- Given that (P/100 * 1265) / 7 = 271.07142857142856, prove that P = 150 -/
theorem percentage_problem (P : ℝ) : (P / 100 * 1265) / 7 = 271.07142857142856 → P = 150 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1655_165540


namespace NUMINAMATH_CALUDE_power_difference_seven_l1655_165516

theorem power_difference_seven (n k : ℕ) : 2^n - 5^k = 7 ↔ n = 5 ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_seven_l1655_165516


namespace NUMINAMATH_CALUDE_sum_squares_distances_to_chord_ends_l1655_165545

/-- Given a circle with radius R and a point M on its diameter at distance a from the center,
    the sum of squares of distances from M to the ends of any chord parallel to the diameter
    is equal to 2(a² + R²). -/
theorem sum_squares_distances_to_chord_ends
  (R a : ℝ) -- R is the radius, a is the distance from M to the center
  (h₁ : 0 < R) -- R is positive (circle has positive radius)
  (h₂ : 0 ≤ a ∧ a ≤ 2*R) -- M is on the diameter, so 0 ≤ a ≤ 2R
  : ∀ A B : ℝ × ℝ, -- For any points A and B
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * R^2 → -- If AB is a chord (distance AB = diameter)
    (∃ k : ℝ, A.2 = k ∧ B.2 = k) → -- If AB is parallel to x-axis (assuming diameter along x-axis)
    (A.1 - a)^2 + A.2^2 + (B.1 - a)^2 + B.2^2 = 2 * (a^2 + R^2) :=
by sorry

end NUMINAMATH_CALUDE_sum_squares_distances_to_chord_ends_l1655_165545


namespace NUMINAMATH_CALUDE_jacket_price_after_discounts_l1655_165584

def initial_price : ℝ := 20
def first_discount : ℝ := 0.40
def second_discount : ℝ := 0.25

theorem jacket_price_after_discounts :
  let price_after_first := initial_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  final_price = 9 := by sorry

end NUMINAMATH_CALUDE_jacket_price_after_discounts_l1655_165584


namespace NUMINAMATH_CALUDE_sequence_21st_term_l1655_165524

theorem sequence_21st_term (a : ℕ → ℚ) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = (4 * a n + 3) / 4) →
  a 1 = 1 →
  a 21 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_21st_term_l1655_165524


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l1655_165593

theorem sqrt_abs_sum_zero_implies_power (x y : ℝ) :
  Real.sqrt (2 * x + 8) + |y - 3| = 0 → (x + y)^2021 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l1655_165593


namespace NUMINAMATH_CALUDE_triangle_integer_points_l1655_165594

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Checks if a point has integer coordinates -/
def hasIntegerCoordinates (p : Point) : Prop :=
  ∃ (ix iy : ℤ), p.x = ↑ix ∧ p.y = ↑iy

/-- Checks if a point is inside or on the boundary of the triangle formed by three points -/
def isInsideOrOnBoundary (p A B C : Point) : Prop :=
  sorry -- Definition of this predicate

/-- Counts the number of points with integer coordinates inside or on the boundary of the triangle -/
def countIntegerPoints (A B C : Point) : ℕ :=
  sorry -- Definition of this function

/-- The main theorem -/
theorem triangle_integer_points (a : ℝ) :
  a > 0 →
  let A : Point := ⟨2 + a, 0⟩
  let B : Point := ⟨2 - a, 0⟩
  let C : Point := ⟨2, 1⟩
  (countIntegerPoints A B C = 4) ↔ (1 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_integer_points_l1655_165594


namespace NUMINAMATH_CALUDE_section_through_center_l1655_165580

-- Define a cube
def Cube := Set (ℝ × ℝ × ℝ)

-- Define a plane section
def PlaneSection := Set (ℝ × ℝ × ℝ)

-- Define the center of a cube
def centerOfCube (c : Cube) : ℝ × ℝ × ℝ := sorry

-- Define the volume of a set in 3D space
def volume (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define what it means for a plane to pass through a point
def passesThrough (p : PlaneSection) (point : ℝ × ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem section_through_center (c : Cube) (s : PlaneSection) :
  (∃ (A B : Set (ℝ × ℝ × ℝ)), A ∪ B = c ∧ A ∩ B = s ∧ volume A = volume B) →
  passesThrough s (centerOfCube c) := by sorry

end NUMINAMATH_CALUDE_section_through_center_l1655_165580


namespace NUMINAMATH_CALUDE_orchard_pure_gala_count_l1655_165557

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  cross_pollinated : ℕ
  pure_gala : ℕ

/-- The number of pure Gala trees in an orchard satisfying specific conditions -/
def pure_gala_count (o : Orchard) : Prop :=
  o.pure_fuji + o.cross_pollinated = 204 ∧
  o.pure_fuji = (3 * o.total) / 4 ∧
  o.cross_pollinated = o.total / 10 ∧
  o.pure_gala = 36

/-- Theorem stating that an orchard satisfying the given conditions has 36 pure Gala trees -/
theorem orchard_pure_gala_count :
  ∃ (o : Orchard), pure_gala_count o :=
sorry

end NUMINAMATH_CALUDE_orchard_pure_gala_count_l1655_165557


namespace NUMINAMATH_CALUDE_selling_price_ratio_l1655_165585

theorem selling_price_ratio 
  (cost_price : ℝ) 
  (profit_percentage : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : profit_percentage = 60) 
  (h2 : loss_percentage = 20) : 
  (cost_price - loss_percentage / 100 * cost_price) / 
  (cost_price + profit_percentage / 100 * cost_price) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l1655_165585


namespace NUMINAMATH_CALUDE_house_height_l1655_165542

theorem house_height (house_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ)
  (h1 : house_shadow = 84)
  (h2 : pole_height = 14)
  (h3 : pole_shadow = 28) :
  (house_shadow / pole_shadow) * pole_height = 42 :=
by sorry

end NUMINAMATH_CALUDE_house_height_l1655_165542


namespace NUMINAMATH_CALUDE_product_fraction_inequality_l1655_165522

theorem product_fraction_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (sum_eq_two : a + b + c = 2) : 
  (a / (1 - a)) * (b / (1 - b)) * (c / (1 - c)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_product_fraction_inequality_l1655_165522


namespace NUMINAMATH_CALUDE_regular_tetrahedron_properties_l1655_165514

-- Define a regular tetrahedron
structure RegularTetrahedron where
  -- Add any necessary fields here
  
-- Define the properties of a regular tetrahedron
def has_equal_edges_and_vertex_angles (t : RegularTetrahedron) : Prop :=
  sorry

def has_congruent_faces_and_equal_dihedral_angles (t : RegularTetrahedron) : Prop :=
  sorry

def has_congruent_faces_and_equal_vertex_angles (t : RegularTetrahedron) : Prop :=
  sorry

-- Theorem stating that a regular tetrahedron satisfies all three properties
theorem regular_tetrahedron_properties (t : RegularTetrahedron) :
  has_equal_edges_and_vertex_angles t ∧
  has_congruent_faces_and_equal_dihedral_angles t ∧
  has_congruent_faces_and_equal_vertex_angles t :=
sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_properties_l1655_165514


namespace NUMINAMATH_CALUDE_village_leadership_choices_l1655_165547

/-- The number of members in the village -/
def villageSize : ℕ := 16

/-- The number of deputy mayors -/
def numDeputyMayors : ℕ := 3

/-- The number of council members per deputy mayor -/
def councilMembersPerDeputy : ℕ := 3

/-- The total number of council members -/
def totalCouncilMembers : ℕ := numDeputyMayors * councilMembersPerDeputy

/-- The number of ways to choose the village leadership -/
def leadershipChoices : ℕ := 
  villageSize * 
  (villageSize - 1) * 
  (villageSize - 2) * 
  (villageSize - 3) * 
  Nat.choose (villageSize - 4) councilMembersPerDeputy * 
  Nat.choose (villageSize - 4 - councilMembersPerDeputy) councilMembersPerDeputy * 
  Nat.choose (villageSize - 4 - 2 * councilMembersPerDeputy) councilMembersPerDeputy

theorem village_leadership_choices : 
  leadershipChoices = 154828800 := by sorry

end NUMINAMATH_CALUDE_village_leadership_choices_l1655_165547


namespace NUMINAMATH_CALUDE_sturgeon_books_problem_l1655_165555

theorem sturgeon_books_problem (total_volumes : ℕ) (paperback_price hardcover_price total_cost : ℚ) 
  (h : total_volumes = 12)
  (hp : paperback_price = 15)
  (hh : hardcover_price = 25)
  (ht : total_cost = 240) :
  ∃ (hardcovers : ℕ), 
    hardcovers * hardcover_price + (total_volumes - hardcovers) * paperback_price = total_cost ∧ 
    hardcovers = 6 := by
  sorry

end NUMINAMATH_CALUDE_sturgeon_books_problem_l1655_165555


namespace NUMINAMATH_CALUDE_single_color_bound_l1655_165506

/-- A polygon on a checkered plane --/
structure CheckeredPolygon where
  /-- The area of the polygon --/
  area : ℕ
  /-- The perimeter of the polygon --/
  perimeter : ℕ

/-- The number of squares of a single color in the polygon --/
def singleColorCount (p : CheckeredPolygon) : ℕ := sorry

/-- Theorem: The number of squares of a single color is bounded --/
theorem single_color_bound (p : CheckeredPolygon) :
  singleColorCount p ≥ p.area / 2 - p.perimeter / 8 ∧
  singleColorCount p ≤ p.area / 2 + p.perimeter / 8 := by
  sorry

end NUMINAMATH_CALUDE_single_color_bound_l1655_165506


namespace NUMINAMATH_CALUDE_max_additional_tiles_l1655_165577

/-- Represents a rectangular board --/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a tile on the board --/
structure Tile :=
  (width : ℕ)
  (height : ℕ)

/-- The number of cells a tile covers --/
def Tile.area (t : Tile) : ℕ := t.width * t.height

/-- The total number of cells on a board --/
def Board.total_cells (b : Board) : ℕ := b.rows * b.cols

/-- The number of cells covered by a list of tiles --/
def covered_cells (tiles : List Tile) : ℕ :=
  tiles.foldl (λ acc t => acc + t.area) 0

theorem max_additional_tiles (board : Board) (initial_tiles : List Tile) :
  board.rows = 10 ∧ 
  board.cols = 9 ∧ 
  initial_tiles.length = 7 ∧ 
  ∀ t ∈ initial_tiles, t.width = 2 ∧ t.height = 1 →
  ∃ (max_additional : ℕ), 
    max_additional = 38 ∧
    covered_cells initial_tiles + 2 * max_additional = board.total_cells :=
by sorry

end NUMINAMATH_CALUDE_max_additional_tiles_l1655_165577


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1655_165546

theorem roots_of_quadratic_equation :
  ∀ x : ℝ, x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1655_165546


namespace NUMINAMATH_CALUDE_polyhedron_inequality_l1655_165569

-- Define a convex polyhedron
structure ConvexPolyhedron where
  edges : List ℝ
  dihedralAngles : List ℝ

-- Define the theorem
theorem polyhedron_inequality (R : ℝ) (P : ConvexPolyhedron) :
  R > 0 →
  P.edges.length = P.dihedralAngles.length →
  (List.sum (List.zipWith (λ l φ => l * (Real.pi - φ)) P.edges P.dihedralAngles)) ≤ 8 * Real.pi * R :=
by sorry

end NUMINAMATH_CALUDE_polyhedron_inequality_l1655_165569


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1655_165592

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 300 ∧ 
  9 * a = b - 11 ∧ 
  9 * a = c + 15 ∧ 
  a ≤ b ∧ 
  a ≤ c ∧ 
  c ≤ b → 
  a * b * c = 319760 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1655_165592


namespace NUMINAMATH_CALUDE_hexagon_covers_ground_l1655_165508

def interior_angle (n : ℕ) : ℚ :=
  (n - 2) * 180 / n

def can_cover_ground (n : ℕ) : Prop :=
  ∃ k : ℕ, k * interior_angle n = 360

theorem hexagon_covers_ground :
  can_cover_ground 6 ∧
  ¬can_cover_ground 5 ∧
  ¬can_cover_ground 8 ∧
  ¬can_cover_ground 12 :=
sorry

end NUMINAMATH_CALUDE_hexagon_covers_ground_l1655_165508


namespace NUMINAMATH_CALUDE_square_difference_equals_twelve_l1655_165528

theorem square_difference_equals_twelve : (2 + 3)^2 - (2^2 + 3^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_twelve_l1655_165528


namespace NUMINAMATH_CALUDE_min_monochromatic_triangles_K15_l1655_165529

/-- A coloring of the edges of a complete graph using two colors. -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- The number of monochromatic triangles in a two-colored complete graph. -/
def monochromaticTriangles (n : ℕ) (c : TwoColoring n) : ℕ := sorry

/-- Theorem: The minimum number of monochromatic triangles in K₁₅ is 88. -/
theorem min_monochromatic_triangles_K15 :
  (∃ c : TwoColoring 15, monochromaticTriangles 15 c = 88) ∧
  (∀ c : TwoColoring 15, monochromaticTriangles 15 c ≥ 88) := by
  sorry

end NUMINAMATH_CALUDE_min_monochromatic_triangles_K15_l1655_165529


namespace NUMINAMATH_CALUDE_zeros_of_quadratic_function_l1655_165527

theorem zeros_of_quadratic_function (f : ℝ → ℝ) :
  (f = λ x => x^2 - x - 2) →
  (∀ x, f x = 0 ↔ x = -1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_quadratic_function_l1655_165527


namespace NUMINAMATH_CALUDE_wrapping_paper_area_formula_l1655_165574

/-- Represents a rectangular box with length, width, and height. -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- Calculates the area of wrapping paper needed to wrap a box. -/
def wrappingPaperArea (box : Box) : ℝ :=
  6 * box.length * box.height + 2 * box.width * box.height

/-- Theorem stating that the wrapping paper area for a box is 6lh + 2wh. -/
theorem wrapping_paper_area_formula (box : Box) :
  wrappingPaperArea box = 6 * box.length * box.height + 2 * box.width * box.height :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_formula_l1655_165574


namespace NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l1655_165551

/-- Calculates the number of pages to write per day given total pages and number of days -/
def pagesPerDay (totalPages : ℕ) (numDays : ℕ) : ℚ :=
  totalPages / numDays

theorem stacy_paper_pages_per_day :
  let totalPages : ℕ := 33
  let numDays : ℕ := 3
  pagesPerDay totalPages numDays = 11 := by
  sorry

end NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l1655_165551


namespace NUMINAMATH_CALUDE_prime_pairs_theorem_l1655_165538

theorem prime_pairs_theorem : 
  ∀ p q : ℕ, 
    Prime p → Prime q → Prime (p * q + p - 6) → 
    ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_theorem_l1655_165538


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l1655_165504

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define a point on a line segment
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

-- Define the angle between two vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ := sorry

theorem triangle_angle_theorem 
  (A B C : ℝ × ℝ) 
  (E : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_bac : angle (B - A) (C - A) = 30 * π / 180)
  (h_e_on_bc : PointOnSegment E B C)
  (h_be_ec : 3 * ‖E - B‖ = 2 * ‖C - E‖)
  (h_eab : angle (E - A) (B - A) = 45 * π / 180) :
  angle (A - C) (B - C) = 15 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l1655_165504


namespace NUMINAMATH_CALUDE_max_sides_diagonal_polygon_13gon_l1655_165517

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a convex n-gon -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A polygon formed by diagonals of a larger polygon -/
structure DiagonalPolygon (n : ℕ) where
  sides : ℕ
  sides_le : sides ≤ n

/-- Theorem: In a convex 13-gon with all diagonals drawn, 
    the maximum number of sides of any polygon formed by these diagonals is 13 -/
theorem max_sides_diagonal_polygon_13gon :
  ∀ (p : ConvexPolygon 13) (d : DiagonalPolygon 13),
    d.sides ≤ 13 ∧ ∃ (d' : DiagonalPolygon 13), d'.sides = 13 :=
sorry

end NUMINAMATH_CALUDE_max_sides_diagonal_polygon_13gon_l1655_165517


namespace NUMINAMATH_CALUDE_room_width_calculation_l1655_165554

/-- Given a rectangular room with specified length, paving cost per square meter,
    and total paving cost, calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ)
  (h1 : length = 5.5)
  (h2 : cost_per_sqm = 950)
  (h3 : total_cost = 20900) :
  total_cost / cost_per_sqm / length = 4 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l1655_165554


namespace NUMINAMATH_CALUDE_larger_field_time_calculation_l1655_165591

-- Define the smaller field's dimensions
def small_width : ℝ := 1  -- We can use any positive real number as the base
def small_length : ℝ := 1.5 * small_width

-- Define the larger field's dimensions
def large_width : ℝ := 4 * small_width
def large_length : ℝ := 3 * small_length

-- Define the perimeters
def small_perimeter : ℝ := 2 * (small_length + small_width)
def large_perimeter : ℝ := 2 * (large_length + large_width)

-- Define the time to complete one round of the smaller field
def small_field_time : ℝ := 20

-- Theorem to prove
theorem larger_field_time_calculation :
  (large_perimeter / small_perimeter) * small_field_time = 68 := by
  sorry

end NUMINAMATH_CALUDE_larger_field_time_calculation_l1655_165591


namespace NUMINAMATH_CALUDE_train_car_estimate_l1655_165562

/-- Represents the number of cars that pass in a given time interval -/
structure CarPassage where
  cars : ℕ
  seconds : ℕ

/-- Calculates the estimated number of cars in a train given initial observations and total passage time -/
def estimateTrainCars (initialObservation : CarPassage) (totalPassageTime : ℕ) : ℕ :=
  (initialObservation.cars * totalPassageTime) / initialObservation.seconds

theorem train_car_estimate :
  let initialObservation : CarPassage := { cars := 8, seconds := 12 }
  let totalPassageTime : ℕ := 210
  estimateTrainCars initialObservation totalPassageTime = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_car_estimate_l1655_165562


namespace NUMINAMATH_CALUDE_erica_safari_animals_l1655_165582

/-- The number of animals Erica saw on Saturday -/
def saturday_animals : ℕ := 3 + 2

/-- The number of animals Erica saw on Sunday -/
def sunday_animals : ℕ := 2 + 5

/-- The number of animals Erica saw on Monday -/
def monday_animals : ℕ := 5 + 3

/-- The total number of animals Erica saw during her safari -/
def total_animals : ℕ := saturday_animals + sunday_animals + monday_animals

theorem erica_safari_animals : total_animals = 20 := by
  sorry

end NUMINAMATH_CALUDE_erica_safari_animals_l1655_165582


namespace NUMINAMATH_CALUDE_weakly_increasing_g_implies_m_eq_4_l1655_165503

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + (4-m)*x + m

-- Define what it means for a function to be increasing on an interval
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- Define what it means for a function to be decreasing on an interval
def IsDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y ≤ b → f x ≥ f y

-- Define what it means for a function to be weakly increasing on an interval
def IsWeaklyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  IsIncreasing f a b ∧ IsDecreasing (fun x => f x / x) a b

-- State the theorem
theorem weakly_increasing_g_implies_m_eq_4 :
  ∀ m : ℝ, IsWeaklyIncreasing (g m) 0 2 → m = 4 :=
by sorry

end NUMINAMATH_CALUDE_weakly_increasing_g_implies_m_eq_4_l1655_165503


namespace NUMINAMATH_CALUDE_log_range_l1655_165548

def log_defined (a : ℝ) : Prop :=
  a - 2 > 0 ∧ a - 2 ≠ 1 ∧ 5 - a > 0

theorem log_range : 
  {a : ℝ | log_defined a} = {a : ℝ | (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5)} :=
by sorry

end NUMINAMATH_CALUDE_log_range_l1655_165548


namespace NUMINAMATH_CALUDE_jaydee_typing_speed_l1655_165520

def typing_speed (hours : ℕ) (words : ℕ) : ℕ :=
  words / (hours * 60)

theorem jaydee_typing_speed :
  typing_speed 2 4560 = 38 :=
by sorry

end NUMINAMATH_CALUDE_jaydee_typing_speed_l1655_165520


namespace NUMINAMATH_CALUDE_ball_arrangement_theorem_l1655_165570

/-- The number of ways to arrange 8 balls in a row, with 5 red and 3 white,
    such that exactly 3 consecutive balls are red -/
def arrangement_count : ℕ := 24

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of consecutive red balls required -/
def consecutive_red : ℕ := 3

theorem ball_arrangement_theorem :
  arrangement_count = 24 ∧
  total_balls = 8 ∧
  red_balls = 5 ∧
  white_balls = 3 ∧
  consecutive_red = 3 :=
by sorry

end NUMINAMATH_CALUDE_ball_arrangement_theorem_l1655_165570


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1655_165531

theorem absolute_value_inequality (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1655_165531


namespace NUMINAMATH_CALUDE_simplify_expression_l1655_165578

theorem simplify_expression (x y : ℝ) : (3*x)^4 + (4*x)*(x^3) + (5*y)^2 = 85*x^4 + 25*y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1655_165578


namespace NUMINAMATH_CALUDE_room_length_proof_l1655_165579

/-- Given a rectangular room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 6 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 →
  total_cost = 25650 →
  paving_rate = 900 →
  (total_cost / paving_rate) / width = 6 := by
sorry

end NUMINAMATH_CALUDE_room_length_proof_l1655_165579


namespace NUMINAMATH_CALUDE_sum_base8_equals_1063_l1655_165544

/-- Converts a base-8 number to its decimal equivalent -/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to its base-8 equivalent -/
def decimalToBase8 (n : ℕ) : ℕ := sorry

theorem sum_base8_equals_1063 :
  let a := base8ToDecimal 236
  let b := base8ToDecimal 521
  let c := base8ToDecimal 74
  decimalToBase8 (a + b + c) = 1063 := by sorry

end NUMINAMATH_CALUDE_sum_base8_equals_1063_l1655_165544


namespace NUMINAMATH_CALUDE_A_contains_B_l1655_165567

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5

def B (x m : ℝ) : Prop := (x - m + 1) * (x - 2 * m - 1) < 0

theorem A_contains_B (m : ℝ) : 
  (∀ x, B x m → A x) ↔ (m = -2 ∨ (-1 ≤ m ∧ m ≤ 2)) := by sorry

end NUMINAMATH_CALUDE_A_contains_B_l1655_165567


namespace NUMINAMATH_CALUDE_income_calculation_l1655_165571

def original_tax_rate : ℝ := 0.46
def new_tax_rate : ℝ := 0.32
def differential_savings : ℝ := 5040

theorem income_calculation (income : ℝ) :
  (original_tax_rate - new_tax_rate) * income = differential_savings →
  income = 36000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l1655_165571


namespace NUMINAMATH_CALUDE_first_day_exceeding_target_day_exceeding_target_is_tuesday_l1655_165589

/-- Geometric sequence sum function -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

/-- First day of deposit (Sunday) -/
def initialDay : ℕ := 0

/-- Initial deposit amount in cents -/
def initialDeposit : ℚ := 3

/-- Daily deposit multiplier -/
def dailyMultiplier : ℚ := 2

/-- Target amount in cents -/
def targetAmount : ℚ := 2000

/-- Function to calculate the day of the week -/
def dayOfWeek (n : ℕ) : ℕ :=
  (initialDay + n) % 7

/-- Theorem: The 10th deposit day is the first to exceed the target amount -/
theorem first_day_exceeding_target :
  (∀ k < 10, geometricSum initialDeposit dailyMultiplier k ≤ targetAmount) ∧
  geometricSum initialDeposit dailyMultiplier 10 > targetAmount :=
sorry

/-- Corollary: The day when the total first exceeds the target is Tuesday -/
theorem day_exceeding_target_is_tuesday :
  dayOfWeek 10 = 2 :=
sorry

end NUMINAMATH_CALUDE_first_day_exceeding_target_day_exceeding_target_is_tuesday_l1655_165589


namespace NUMINAMATH_CALUDE_fashion_pricing_increase_l1655_165521

theorem fashion_pricing_increase (C : ℝ) : 
  let retailer_price := 1.40 * C
  let customer_price := 1.6100000000000001 * C
  ((customer_price - retailer_price) / retailer_price) * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_fashion_pricing_increase_l1655_165521


namespace NUMINAMATH_CALUDE_at_least_one_hits_target_l1655_165566

theorem at_least_one_hits_target (p_both : ℝ) (h : p_both = 0.6) :
  1 - (1 - p_both) * (1 - p_both) = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_hits_target_l1655_165566


namespace NUMINAMATH_CALUDE_dice_sum_probability_l1655_165568

-- Define the number of dice
def num_dice : ℕ := 8

-- Define the target sum
def target_sum : ℕ := 11

-- Define the function to calculate the number of ways to achieve the target sum
def num_ways_to_achieve_sum (n d s : ℕ) : ℕ :=
  Nat.choose (s - n + d - 1) (d - 1)

-- Theorem statement
theorem dice_sum_probability :
  num_ways_to_achieve_sum num_dice num_dice target_sum = 120 := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l1655_165568
