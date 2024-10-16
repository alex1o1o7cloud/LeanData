import Mathlib

namespace NUMINAMATH_CALUDE_project_hours_difference_l2377_237788

theorem project_hours_difference (total_hours : ℕ) 
  (h1 : total_hours = 189) 
  (kate_hours : ℕ) (pat_hours : ℕ) (mark_hours : ℕ) 
  (h2 : pat_hours = 2 * kate_hours) 
  (h3 : pat_hours = mark_hours / 3) 
  (h4 : kate_hours + pat_hours + mark_hours = total_hours) : 
  mark_hours - kate_hours = 105 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l2377_237788


namespace NUMINAMATH_CALUDE_tree_height_problem_l2377_237792

/-- Given a square ABCD with trees of heights a, b, c at vertices A, B, C respectively,
    and a point O inside the square equidistant from all vertices,
    prove that the height of the tree at vertex D is √(a² + c² - b²). -/
theorem tree_height_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (d : ℝ), d > 0 ∧ d^2 = a^2 + c^2 - b^2 := by
  sorry


end NUMINAMATH_CALUDE_tree_height_problem_l2377_237792


namespace NUMINAMATH_CALUDE_second_polygon_sides_l2377_237773

/-- Given two regular polygons with the same perimeter, where the first has 24 sides
    and a side length three times as long as the second, prove that the second has 72 sides. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 →
  24 * (3 * s) = n * s → n = 72 := by sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l2377_237773


namespace NUMINAMATH_CALUDE_triangle_shape_from_complex_product_l2377_237738

open Complex

/-- Given a triangle ABC with sides a, b and angles A, B, C,
    if z₁ = a + bi and z₂ = cos A + i cos B, and their product is purely imaginary,
    then the triangle is either isosceles or right-angled. -/
theorem triangle_shape_from_complex_product (a b : ℝ) (A B C : ℝ) :
  let z₁ : ℂ := ⟨a, b⟩
  let z₂ : ℂ := ⟨Real.cos A, Real.cos B⟩
  (0 < A) ∧ (0 < B) ∧ (0 < C) ∧ (A + B + C = π) →  -- Triangle conditions
  (z₁ * z₂).re = 0 →  -- Product is purely imaginary
  (A = B) ∨ (A + B = π / 2) :=  -- Triangle is isosceles or right-angled
by sorry

end NUMINAMATH_CALUDE_triangle_shape_from_complex_product_l2377_237738


namespace NUMINAMATH_CALUDE_john_pennies_l2377_237741

/-- Given that Kate has 223 pennies, John has more pennies than Kate,
    and the difference between their pennies is 165,
    prove that John has 388 pennies. -/
theorem john_pennies (kate_pennies : ℕ) (john_more : ℕ) (difference : ℕ)
    (h1 : kate_pennies = 223)
    (h2 : john_more > kate_pennies)
    (h3 : john_more - kate_pennies = difference)
    (h4 : difference = 165) :
    john_more = 388 := by
  sorry

end NUMINAMATH_CALUDE_john_pennies_l2377_237741


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2377_237790

/-- Given an infinite geometric sequence {a_n} with first term 1 and common ratio a - 3/2,
    if the sum of all terms is a, then a = 2. -/
theorem geometric_sequence_sum (a : ℝ) : 
  let a_1 : ℝ := 1
  let q : ℝ := a - 3/2
  let sum : ℝ := a_1 / (1 - q)
  (sum = a) → (a = 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2377_237790


namespace NUMINAMATH_CALUDE_quadratic_function_absolute_value_l2377_237746

theorem quadratic_function_absolute_value (p q : ℝ) :
  ∃ (x : ℝ), x ∈ ({1, 2, 3} : Set ℝ) ∧ |x^2 + p*x + q| ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_absolute_value_l2377_237746


namespace NUMINAMATH_CALUDE_unique_solution_1999_l2377_237766

theorem unique_solution_1999 :
  ∃! (a b c d : ℕ), 5^a + 6^b + 7^c + 11^d = 1999 ∧ 
  a = 4 ∧ b = 2 ∧ c = 1 ∧ d = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_1999_l2377_237766


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l2377_237705

theorem quadratic_completing_square : ∀ x : ℝ, x^2 - 8*x + 6 = 0 ↔ (x - 4)^2 = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l2377_237705


namespace NUMINAMATH_CALUDE_canoe_current_speed_l2377_237748

/-- Represents the speed of a canoe in still water and the speed of the current. -/
structure CanoeSpeedData where
  canoe_speed : ℝ
  current_speed : ℝ

/-- Calculates the effective speed of a canoe given the canoe's speed in still water and the current speed. -/
def effective_speed (upstream : Bool) (data : CanoeSpeedData) : ℝ :=
  if upstream then data.canoe_speed - data.current_speed else data.canoe_speed + data.current_speed

/-- Theorem stating that given the conditions of the canoe problem, the speed of the current is 7 miles per hour. -/
theorem canoe_current_speed : 
  ∀ (data : CanoeSpeedData),
    (effective_speed true data) * 6 = 12 →
    (effective_speed false data) * 0.75 = 12 →
    data.current_speed = 7 := by
  sorry


end NUMINAMATH_CALUDE_canoe_current_speed_l2377_237748


namespace NUMINAMATH_CALUDE_complex_number_proof_l2377_237722

theorem complex_number_proof (z : ℂ) :
  (z.re = Complex.im (-Real.sqrt 2 + 7 * Complex.I)) ∧
  (z.im = Complex.re (Real.sqrt 7 * Complex.I + 5 * Complex.I^2)) →
  z = 7 - 5 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_number_proof_l2377_237722


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2377_237767

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2377_237767


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2377_237762

/-- The ratio of the volume of a cube with edge length 10 inches to the volume of a cube with edge length 3 feet -/
theorem cube_volume_ratio : 
  let inch_to_foot : ℚ := 1 / 12
  let cube1_edge : ℚ := 10
  let cube2_edge : ℚ := 3 / inch_to_foot
  let cube1_volume : ℚ := cube1_edge ^ 3
  let cube2_volume : ℚ := cube2_edge ^ 3
  cube1_volume / cube2_volume = 125 / 5832 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2377_237762


namespace NUMINAMATH_CALUDE_pencil_count_l2377_237780

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := 71 - 30

/-- The number of pencils Mike added to the drawer -/
def added_pencils : ℕ := 30

/-- The total number of pencils after Mike's addition -/
def total_pencils : ℕ := 71

/-- Theorem stating that the original number of pencils plus the added pencils equals the total pencils -/
theorem pencil_count : original_pencils + added_pencils = total_pencils := by
  sorry

#eval original_pencils -- This will output 41

end NUMINAMATH_CALUDE_pencil_count_l2377_237780


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_l2377_237756

theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  -- a, b, c form a geometric sequence
  b^2 = a * c →
  -- cos B = 1/3
  Real.cos B = 1/3 →
  -- a/c = 1/2
  a / c = 1/2 →
  -- k is the first term of the geometric sequence
  ∃ k : ℝ, k > 0 ∧ a = k ∧ b = 2*k ∧ c = 4*k ∧ a + c = 5*k := by
sorry

end NUMINAMATH_CALUDE_triangle_geometric_sequence_l2377_237756


namespace NUMINAMATH_CALUDE_bill_sunday_miles_l2377_237714

/-- Represents the number of miles run by Bill and Julia over two days --/
structure RunningMiles where
  billSaturday : ℕ
  billSunday : ℕ
  juliaSunday : ℕ

/-- The conditions of the running problem --/
def runningProblem (r : RunningMiles) : Prop :=
  r.billSunday = r.billSaturday + 4 ∧
  r.juliaSunday = 2 * r.billSunday ∧
  r.billSaturday + r.billSunday + r.juliaSunday = 36

theorem bill_sunday_miles (r : RunningMiles) :
  runningProblem r → r.billSunday = 10 := by
  sorry

end NUMINAMATH_CALUDE_bill_sunday_miles_l2377_237714


namespace NUMINAMATH_CALUDE_common_tangent_implies_a_equals_one_l2377_237777

/-- Given two curves y = (1/2e)x^2 and y = a ln x with a common tangent at their common point P(s,t), prove that a = 1 --/
theorem common_tangent_implies_a_equals_one (e : ℝ) (a s t : ℝ) : 
  (t = (1/(2*Real.exp 1))*s^2) → 
  (t = a * Real.log s) → 
  ((s / Real.exp 1) = (a / s)) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_common_tangent_implies_a_equals_one_l2377_237777


namespace NUMINAMATH_CALUDE_unique_fraction_decomposition_l2377_237782

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_not_two : p ≠ 2) :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ (2 : ℚ) / p = 1 / x + 1 / y ∧ 
  x = (p^2 + p) / 2 ∧ y = (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_decomposition_l2377_237782


namespace NUMINAMATH_CALUDE_line_through_point_l2377_237718

/-- If the line ax + 3y - 5 = 0 passes through the point (2, 1), then a = 1 -/
theorem line_through_point (a : ℝ) : 
  (a * 2 + 3 * 1 - 5 = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2377_237718


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2377_237789

theorem stewart_farm_sheep_count :
  ∀ (num_sheep num_horses : ℕ),
    num_sheep / num_horses = 2 / 7 →
    num_horses * 230 = 12880 →
    num_sheep = 16 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2377_237789


namespace NUMINAMATH_CALUDE_sum_and_double_l2377_237779

theorem sum_and_double : (2345 + 3452 + 4523 + 5234) * 2 = 31108 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_double_l2377_237779


namespace NUMINAMATH_CALUDE_sock_pair_probability_l2377_237711

def number_of_socks : ℕ := 10
def number_of_colors : ℕ := 5
def socks_per_color : ℕ := 2
def socks_drawn : ℕ := 6

theorem sock_pair_probability :
  let total_combinations := Nat.choose number_of_socks socks_drawn
  let pair_combinations := Nat.choose number_of_colors 2 * Nat.choose (number_of_colors - 2) 2 * 4
  (pair_combinations : ℚ) / total_combinations = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_probability_l2377_237711


namespace NUMINAMATH_CALUDE_mixture_weight_approx_140_l2377_237785

/-- Represents the weight ratio of almonds to walnuts in the mixture -/
def almond_to_walnut_ratio : ℚ := 5

/-- Represents the weight of almonds in the mixture in pounds -/
def almond_weight : ℚ := 116.67

/-- Calculates the total weight of the mixture -/
def total_mixture_weight : ℚ :=
  almond_weight + (almond_weight / almond_to_walnut_ratio)

/-- Theorem stating that the total weight of the mixture is approximately 140 pounds -/
theorem mixture_weight_approx_140 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |total_mixture_weight - 140| < ε :=
sorry

end NUMINAMATH_CALUDE_mixture_weight_approx_140_l2377_237785


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_six_l2377_237758

theorem arithmetic_square_root_of_six : Real.sqrt 6 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_six_l2377_237758


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l2377_237772

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, is_perfect_square_trinomial 1 (2*m) 9 → m = 3 ∨ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l2377_237772


namespace NUMINAMATH_CALUDE_line_properties_l2377_237759

/-- Triangle PQR with vertices P(1, 9), Q(3, 2), and R(9, 2) -/
structure Triangle where
  P : ℝ × ℝ := (1, 9)
  Q : ℝ × ℝ := (3, 2)
  R : ℝ × ℝ := (9, 2)

/-- A line defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Function to calculate the area of a triangle given its vertices -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Function to check if a line cuts the triangle's area in half -/
def cutsAreaInHalf (t : Triangle) (l : Line) : Prop := sorry

/-- Theorem stating the properties of the line that cuts the triangle's area in half -/
theorem line_properties (t : Triangle) (l : Line) :
  cutsAreaInHalf t l ∧ l.yIntercept = 1 →
  l.slope = 1/3 ∧ l.slope + l.yIntercept = 4/3 := by sorry

end NUMINAMATH_CALUDE_line_properties_l2377_237759


namespace NUMINAMATH_CALUDE_hawkeye_fewer_maine_coons_l2377_237700

/-- Proves that Hawkeye owns 1 fewer Maine Coon than Gordon --/
theorem hawkeye_fewer_maine_coons (jamie_persians jamie_maine_coons gordon_persians gordon_maine_coons hawkeye_maine_coons : ℕ) :
  jamie_persians = 4 →
  jamie_maine_coons = 2 →
  gordon_persians = jamie_persians / 2 →
  gordon_maine_coons = jamie_maine_coons + 1 →
  jamie_persians + jamie_maine_coons + gordon_persians + gordon_maine_coons + hawkeye_maine_coons = 13 →
  gordon_maine_coons - hawkeye_maine_coons = 1 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_fewer_maine_coons_l2377_237700


namespace NUMINAMATH_CALUDE_factorization_equality_l2377_237795

theorem factorization_equality (y a : ℝ) : 3*y*a^2 - 6*y*a + 3*y = 3*y*(a-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2377_237795


namespace NUMINAMATH_CALUDE_opposite_reciprocal_problem_l2377_237701

theorem opposite_reciprocal_problem (a b c d m : ℤ) : 
  (a = -b) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (m = -1) →  -- m is the largest negative integer
  c * d - a - b + m ^ 2022 = 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_problem_l2377_237701


namespace NUMINAMATH_CALUDE_no_equal_divisors_for_squares_l2377_237707

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def divisors_3k_plus_1 (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d > 0 ∧ n % d = 0 ∧ d % 3 = 1)

def divisors_3k_plus_2 (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d > 0 ∧ n % d = 0 ∧ d % 3 = 2)

theorem no_equal_divisors_for_squares :
  ∀ n : ℕ, is_square n → (divisors_3k_plus_1 n).card ≠ (divisors_3k_plus_2 n).card :=
by sorry

end NUMINAMATH_CALUDE_no_equal_divisors_for_squares_l2377_237707


namespace NUMINAMATH_CALUDE_bankers_discount_calculation_l2377_237745

/-- Banker's discount calculation -/
theorem bankers_discount_calculation 
  (bankers_gain : ℝ) 
  (time : ℝ) 
  (rate : ℝ) :
  bankers_gain = 180 →
  time = 3 →
  rate = 12 →
  (bankers_gain / (1 - (rate * time) / 100)) = 281.25 :=
by sorry

end NUMINAMATH_CALUDE_bankers_discount_calculation_l2377_237745


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_5_cubed_plus_10_to_4_l2377_237727

theorem greatest_prime_factor_of_5_cubed_plus_10_to_4 :
  ∃ p : ℕ, p.Prime ∧ p ∣ (5^3 + 10^4) ∧ ∀ q : ℕ, q.Prime → q ∣ (5^3 + 10^4) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_5_cubed_plus_10_to_4_l2377_237727


namespace NUMINAMATH_CALUDE_cows_bought_is_two_l2377_237760

/-- The number of cows bought given the total cost, number of goats, and average prices -/
def number_of_cows (total_cost : ℕ) (num_goats : ℕ) (avg_price_goat : ℕ) (avg_price_cow : ℕ) : ℕ :=
  ((total_cost - num_goats * avg_price_goat) / avg_price_cow)

/-- Theorem stating that the number of cows bought is 2 -/
theorem cows_bought_is_two :
  number_of_cows 1400 8 60 460 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cows_bought_is_two_l2377_237760


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2377_237754

theorem inequality_solution_set : 
  {x : ℝ | -x^2 - x + 6 > 0} = Set.Ioo (-3 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2377_237754


namespace NUMINAMATH_CALUDE_cookies_calculation_l2377_237719

/-- The number of people Brenda's mother made cookies for -/
def num_people : ℕ := 25

/-- The number of cookies each person had -/
def cookies_per_person : ℕ := 45

/-- The total number of cookies Brenda's mother prepared -/
def total_cookies : ℕ := num_people * cookies_per_person

theorem cookies_calculation :
  total_cookies = 1125 :=
by sorry

end NUMINAMATH_CALUDE_cookies_calculation_l2377_237719


namespace NUMINAMATH_CALUDE_time_to_paint_one_room_l2377_237717

theorem time_to_paint_one_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (time_for_remaining : ℕ) : 
  total_rooms = 10 → 
  painted_rooms = 8 → 
  time_for_remaining = 16 → 
  (time_for_remaining / (total_rooms - painted_rooms) : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_time_to_paint_one_room_l2377_237717


namespace NUMINAMATH_CALUDE_sum_digits_2_5_power_1997_l2377_237721

/-- The number of decimal digits in a positive integer -/
def num_digits (x : ℕ) : ℕ := sorry

/-- Theorem: The sum of the number of decimal digits in 2^1997 and 5^1997 is 1998 -/
theorem sum_digits_2_5_power_1997 : num_digits (2^1997) + num_digits (5^1997) = 1998 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_2_5_power_1997_l2377_237721


namespace NUMINAMATH_CALUDE_jacob_final_score_l2377_237742

/-- Represents the score for a quiz contest -/
structure QuizScore where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  correct_points : ℚ
  incorrect_points : ℚ

/-- Calculates the final score for a quiz contest -/
def final_score (qs : QuizScore) : ℚ :=
  qs.correct * qs.correct_points + qs.incorrect * qs.incorrect_points

/-- Jacob's quiz score -/
def jacob_score : QuizScore :=
  { correct := 20
    incorrect := 10
    unanswered := 5
    correct_points := 1
    incorrect_points := -1/2 }

/-- Theorem: Jacob's final score is 15 points -/
theorem jacob_final_score :
  final_score jacob_score = 15 := by sorry

end NUMINAMATH_CALUDE_jacob_final_score_l2377_237742


namespace NUMINAMATH_CALUDE_coffee_maker_capacity_l2377_237740

/-- A cylindrical coffee maker with capacity x cups contains 30 cups when 25% full. -/
theorem coffee_maker_capacity (x : ℝ) (h : 0.25 * x = 30) : x = 120 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_capacity_l2377_237740


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l2377_237739

/-- Given that x₁ and x₂ are extremal points of f(x) = (1/3)ax³ - (1/2)ax² - x,
    prove that the line passing through A(x₁, 1/x₁) and B(x₂, 1/x₂)
    intersects the ellipse x²/2 + y² = 1 --/
theorem line_intersects_ellipse (a : ℝ) (x₁ x₂ : ℝ) :
  (x₁ ≠ x₂) →
  (∀ x, (a*x^2 - a*x - 1 = 0) ↔ (x = x₁ ∨ x = x₂)) →
  ∃ x y : ℝ, (y - 1/x₁ = (1/x₂ - 1/x₁)/(x₂ - x₁) * (x - x₁)) ∧
             (x^2/2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l2377_237739


namespace NUMINAMATH_CALUDE_expand_expression_l2377_237730

theorem expand_expression (x y : ℝ) : (2*x + 3) * (5*y + 7) = 10*x*y + 14*x + 15*y + 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2377_237730


namespace NUMINAMATH_CALUDE_binomial_coefficient_minus_two_divisible_by_prime_l2377_237799

theorem binomial_coefficient_minus_two_divisible_by_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (2 * p).factorial / (p.factorial * p.factorial) - 2 = k * p :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_minus_two_divisible_by_prime_l2377_237799


namespace NUMINAMATH_CALUDE_sine_cosine_identity_l2377_237725

theorem sine_cosine_identity : 
  Real.sin (50 * π / 180) * Real.cos (170 * π / 180) - 
  Real.sin (40 * π / 180) * Real.sin (170 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_identity_l2377_237725


namespace NUMINAMATH_CALUDE_mary_additional_flour_l2377_237761

/-- The number of cups of flour required by the recipe -/
def total_flour : ℕ := 8

/-- The number of cups of flour Mary has already added -/
def added_flour : ℕ := 2

/-- The number of additional cups of flour Mary needs to add -/
def additional_flour : ℕ := total_flour - added_flour

theorem mary_additional_flour : additional_flour = 6 := by sorry

end NUMINAMATH_CALUDE_mary_additional_flour_l2377_237761


namespace NUMINAMATH_CALUDE_competition_scores_l2377_237715

def student_scores : List ℝ := [80, 84, 86, 90]

theorem competition_scores (fifth_score : ℝ) 
  (h1 : (fifth_score :: student_scores).length = 5)
  (h2 : (fifth_score :: student_scores).sum / 5 = 87) :
  fifth_score = 95 ∧ 
  let all_scores := fifth_score :: student_scores
  (all_scores.map (λ x => (x - 87)^2)).sum / 5 = 26.4 := by
  sorry

end NUMINAMATH_CALUDE_competition_scores_l2377_237715


namespace NUMINAMATH_CALUDE_inverse_mod_31_l2377_237778

theorem inverse_mod_31 (h : (11⁻¹ : ZMod 31) = 3) : (20⁻¹ : ZMod 31) = 28 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_31_l2377_237778


namespace NUMINAMATH_CALUDE_expected_balls_original_value_l2377_237755

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 7

/-- The number of interchanges performed -/
def num_interchanges : ℕ := 4

/-- The probability of a specific ball being chosen for an interchange -/
def prob_chosen : ℚ := 2 / 7

/-- The probability of a ball being in its original position after the interchanges -/
def prob_original_position : ℚ :=
  (1 - prob_chosen) ^ num_interchanges +
  (num_interchanges.choose 2) * prob_chosen ^ 2 * (1 - prob_chosen) ^ 2 +
  prob_chosen ^ num_interchanges

/-- The expected number of balls in their original positions -/
def expected_balls_original : ℚ := num_balls * prob_original_position

theorem expected_balls_original_value :
  expected_balls_original = 3.61 := by sorry

end NUMINAMATH_CALUDE_expected_balls_original_value_l2377_237755


namespace NUMINAMATH_CALUDE_no_x_term_implies_m_value_l2377_237737

theorem no_x_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 + 0*x + (-8*m)) → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_no_x_term_implies_m_value_l2377_237737


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_l2377_237747

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | -1 < x ∧ x < 3} := by sorry

-- Theorem for ∁ℝA
theorem complement_A : (Set.univ : Set ℝ) \ A = {x | x ≤ -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_l2377_237747


namespace NUMINAMATH_CALUDE_bag_properties_l2377_237768

/-- A bag containing colored balls -/
structure Bag where
  red : ℕ
  black : ℕ
  white : ℕ

/-- The scoring system for the balls -/
def score (color : String) : ℕ :=
  match color with
  | "white" => 2
  | "black" => 1
  | "red" => 0
  | _ => 0

/-- The theorem stating the properties of the bag and the probabilities -/
theorem bag_properties (b : Bag) : 
  b.red = 1 ∧ b.black = 1 ∧ b.white = 2 →
  (b.white : ℚ) / (b.red + b.black + b.white : ℚ) = 1/2 ∧
  (2 : ℚ) / ((b.red + b.black + b.white) * (b.red + b.black + b.white - 1) : ℚ) = 1/3 :=
by sorry

#check bag_properties

end NUMINAMATH_CALUDE_bag_properties_l2377_237768


namespace NUMINAMATH_CALUDE_triangle_semiperimeter_from_side_and_excircle_radii_l2377_237706

/-- Given a side 'a' of a triangle and the radii of the excircles opposite 
    the other two sides 'ρ_b' and 'ρ_c', the semiperimeter 's' of the 
    triangle is equal to a/2 + √((a/2)² + ρ_b * ρ_c). -/
theorem triangle_semiperimeter_from_side_and_excircle_radii 
  (a ρ_b ρ_c : ℝ) (ha : a > 0) (hb : ρ_b > 0) (hc : ρ_c > 0) :
  ∃ s : ℝ, s > 0 ∧ s = a / 2 + Real.sqrt ((a / 2)^2 + ρ_b * ρ_c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_semiperimeter_from_side_and_excircle_radii_l2377_237706


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2377_237720

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 35)
  (sum2 : b + c = 47)
  (sum3 : c + a = 58) : 
  a + b + c = 70 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2377_237720


namespace NUMINAMATH_CALUDE_bicyclist_average_speed_l2377_237712

/-- Proves that the average speed of a bicyclist is 18 km/h given the specified conditions -/
theorem bicyclist_average_speed :
  let total_distance : ℝ := 450
  let first_part_distance : ℝ := 300
  let second_part_distance : ℝ := total_distance - first_part_distance
  let first_part_speed : ℝ := 20
  let second_part_speed : ℝ := 15
  let first_part_time : ℝ := first_part_distance / first_part_speed
  let second_part_time : ℝ := second_part_distance / second_part_speed
  let total_time : ℝ := first_part_time + second_part_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_bicyclist_average_speed_l2377_237712


namespace NUMINAMATH_CALUDE_arrangements_count_l2377_237764

/-- The number of candidates -/
def total_candidates : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The number of students who can be assigned to swimming -/
def swimming_candidates : ℕ := total_candidates - 1

/-- The number of different arrangements -/
def arrangements : ℕ := swimming_candidates * (total_candidates - 1) * (total_candidates - 2)

theorem arrangements_count : arrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l2377_237764


namespace NUMINAMATH_CALUDE_solution_range_l2377_237798

-- Define the equation
def equation (x m : ℝ) : Prop :=
  (2 * x - 1) / (x + 1) = 3 - m / (x + 1)

-- Define the theorem
theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ equation x m ∧ x ≠ -1) → m < 4 ∧ m ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l2377_237798


namespace NUMINAMATH_CALUDE_sum_difference_remainder_l2377_237794

theorem sum_difference_remainder (a b c : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k)
  (hb : ∃ k : ℤ, b = 3 * k + 1)
  (hc : ∃ k : ℤ, c = 3 * k - 1) :
  ∃ k : ℤ, a + b - c = 3 * k - 1 := by
sorry

end NUMINAMATH_CALUDE_sum_difference_remainder_l2377_237794


namespace NUMINAMATH_CALUDE_bennys_working_hours_l2377_237771

/-- Calculates the total working hours given hours per day and number of days worked -/
def totalWorkingHours (hoursPerDay : ℕ) (daysWorked : ℕ) : ℕ :=
  hoursPerDay * daysWorked

/-- Proves that Benny's total working hours is 18 given the conditions -/
theorem bennys_working_hours :
  let hoursPerDay : ℕ := 3
  let daysWorked : ℕ := 6
  totalWorkingHours hoursPerDay daysWorked = 18 := by
  sorry

end NUMINAMATH_CALUDE_bennys_working_hours_l2377_237771


namespace NUMINAMATH_CALUDE_min_value_inequality_l2377_237786

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = Real.sqrt 3) : 1/a^2 + 2/b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2377_237786


namespace NUMINAMATH_CALUDE_ordered_triples_solution_l2377_237728

theorem ordered_triples_solution :
  ∀ a b c : ℝ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (⌊a⌋ * b * c = 3 ∧ a * ⌊b⌋ * c = 4 ∧ a * b * ⌊c⌋ = 5) →
  ((a = Real.sqrt 30 / 3 ∧ b = Real.sqrt 30 / 4 ∧ c = 2 * Real.sqrt 30 / 5) ∨
   (a = Real.sqrt 30 / 3 ∧ b = Real.sqrt 30 / 2 ∧ c = Real.sqrt 30 / 5)) :=
by sorry

end NUMINAMATH_CALUDE_ordered_triples_solution_l2377_237728


namespace NUMINAMATH_CALUDE_intersecting_subset_exists_l2377_237732

theorem intersecting_subset_exists (X : Finset ℕ) (A : Fin 100 → Finset ℕ) 
  (h_size : X.card ≥ 4) 
  (h_subsets : ∀ i, A i ⊆ X) 
  (h_large : ∀ i, (A i).card > 3/4 * X.card) :
  ∃ Y : Finset ℕ, Y ⊆ X ∧ Y.card ≤ 4 ∧ ∀ i, (Y ∩ A i).Nonempty := by
  sorry


end NUMINAMATH_CALUDE_intersecting_subset_exists_l2377_237732


namespace NUMINAMATH_CALUDE_seventy_seven_base4_non_consecutive_digits_l2377_237724

/-- Converts a decimal number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of non-consecutive digits in a list of digits -/
def countNonConsecutiveDigits (digits : List ℕ) : ℕ :=
  sorry

theorem seventy_seven_base4_non_consecutive_digits :
  let base4Repr := toBase4 77
  countNonConsecutiveDigits base4Repr = 3 :=
by sorry

end NUMINAMATH_CALUDE_seventy_seven_base4_non_consecutive_digits_l2377_237724


namespace NUMINAMATH_CALUDE_perpendicular_vector_l2377_237751

def vector_AB : Fin 2 → ℝ := ![1, 1]
def vector_AC : Fin 2 → ℝ := ![2, 3]
def vector_BC : Fin 2 → ℝ := ![1, 2]
def vector_D : Fin 2 → ℝ := ![-6, 3]

theorem perpendicular_vector : 
  (vector_AB = ![1, 1]) → 
  (vector_AC = ![2, 3]) → 
  (vector_BC = vector_AC - vector_AB) →
  (vector_D • vector_BC = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vector_l2377_237751


namespace NUMINAMATH_CALUDE_min_value_expression_l2377_237757

theorem min_value_expression (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a * b + b * c + c * a > 0)
  (h3 : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (2 / |a - b| + 2 / |b - c| + 2 / |c - a| + 5 / Real.sqrt (a * b + b * c + c * a)) ≥ 10 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2377_237757


namespace NUMINAMATH_CALUDE_min_value_fraction_l2377_237787

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  4/x + 1/y ≥ 6 + 4*Real.sqrt 2 ∧
  (4/x + 1/y = 6 + 4*Real.sqrt 2 ↔ x = 2 - Real.sqrt 2 ∧ y = (Real.sqrt 2 - 1)/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2377_237787


namespace NUMINAMATH_CALUDE_unique_arithmetic_progression_l2377_237723

theorem unique_arithmetic_progression : ∃! (a b : ℝ),
  (a - 15 = b - a) ∧ (ab - b = b - a) ∧ (a - b = 5) ∧ (a = 10) ∧ (b = 5) := by
  sorry

end NUMINAMATH_CALUDE_unique_arithmetic_progression_l2377_237723


namespace NUMINAMATH_CALUDE_letterArrangements_eq_25_l2377_237763

/-- The number of ways to arrange 15 letters with specific constraints -/
def letterArrangements : ℕ :=
  let totalLetters := 15
  let numA := 4
  let numB := 6
  let numC := 5
  let firstSection := 5
  let middleSection := 5
  let lastSection := 5
  -- Define the constraints
  let noCInFirst := true
  let noAInMiddle := true
  let noBInLast := true
  -- Calculate the number of arrangements
  25

/-- Theorem stating that the number of valid arrangements is 25 -/
theorem letterArrangements_eq_25 : letterArrangements = 25 := by
  sorry

end NUMINAMATH_CALUDE_letterArrangements_eq_25_l2377_237763


namespace NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l2377_237743

theorem min_value_trig_expression (α β : ℝ) :
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 ≥ 100 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ α β : ℝ, (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l2377_237743


namespace NUMINAMATH_CALUDE_smallest_base_for_150_base_6_works_for_150_smallest_base_is_6_l2377_237702

theorem smallest_base_for_150 :
  ∀ b : ℕ, b > 0 → (b^2 ≤ 150 ∧ 150 < b^3) → b ≥ 6 :=
by sorry

theorem base_6_works_for_150 :
  6^2 ≤ 150 ∧ 150 < 6^3 :=
by sorry

theorem smallest_base_is_6 :
  ∃! b : ℕ, b > 0 ∧ b^2 ≤ 150 ∧ 150 < b^3 ∧ ∀ c : ℕ, (c > 0 ∧ c^2 ≤ 150 ∧ 150 < c^3) → c ≥ b :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_150_base_6_works_for_150_smallest_base_is_6_l2377_237702


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_l2377_237709

structure GeometricSpace where
  Line : Type
  Plane : Type
  is_parallel : Line → Plane → Prop
  intersect : Plane → Plane → Line
  line_parallel : Line → Line → Prop

theorem line_parallel_to_intersection
  (S : GeometricSpace)
  (l : S.Line)
  (p1 p2 : S.Plane)
  (h1 : S.is_parallel l p1)
  (h2 : S.is_parallel l p2)
  (h3 : p1 ≠ p2) :
  S.line_parallel l (S.intersect p1 p2) :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_l2377_237709


namespace NUMINAMATH_CALUDE_square_ratio_problem_l2377_237796

theorem square_ratio_problem (A₁ A₂ : ℝ) (s₁ s₂ : ℝ) :
  A₁ / A₂ = 300 / 147 →
  A₁ = s₁^2 →
  A₂ = s₂^2 →
  4 * s₁ = 60 →
  s₁ / s₂ = 10 / 7 ∧ s₂ = 21 / 2 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_problem_l2377_237796


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l2377_237797

-- Define the number of each type of plant
def num_basil : ℕ := 3
def num_tomato : ℕ := 3
def num_pepper : ℕ := 2

-- Define the number of tomato groups
def num_tomato_groups : ℕ := 2

-- Define the function to calculate the number of arrangements
def num_arrangements : ℕ :=
  (Nat.factorial num_basil) *
  (Nat.choose num_tomato 2) *
  (Nat.factorial 2) *
  (Nat.choose (num_basil + 1) num_tomato_groups) *
  (Nat.factorial num_pepper)

-- Theorem statement
theorem plant_arrangement_count :
  num_arrangements = 432 :=
sorry

end NUMINAMATH_CALUDE_plant_arrangement_count_l2377_237797


namespace NUMINAMATH_CALUDE_min_bench_sections_for_equal_seating_l2377_237744

/-- Represents the capacity of a bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Calculates the minimum number of bench sections needed -/
def minBenchSections (capacity : BenchCapacity) : Nat :=
  Nat.lcm capacity.adults capacity.children / capacity.adults

/-- Theorem stating the minimum number of bench sections needed -/
theorem min_bench_sections_for_equal_seating (capacity : BenchCapacity) 
  (h1 : capacity.adults = 8) 
  (h2 : capacity.children = 12) : 
  minBenchSections capacity = 3 := by
  sorry

#eval minBenchSections ⟨8, 12⟩

end NUMINAMATH_CALUDE_min_bench_sections_for_equal_seating_l2377_237744


namespace NUMINAMATH_CALUDE_smallest_norm_l2377_237784

open Real
open InnerProductSpace

/-- Given a vector v such that ‖v + (4, 2)‖ = 10, the smallest possible value of ‖v‖ is 10 - 2√5 -/
theorem smallest_norm (v : ℝ × ℝ) (h : ‖v + (4, 2)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ u : ℝ × ℝ, ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ :=
by sorry

end NUMINAMATH_CALUDE_smallest_norm_l2377_237784


namespace NUMINAMATH_CALUDE_isosceles_triangle_properties_l2377_237733

-- Define the triangle and its properties
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (D : ℝ × ℝ), 
    -- AB = AC (isosceles)
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
    -- Line AB: 2x + y - 4 = 0
    2 * A.1 + A.2 - 4 = 0 ∧ 2 * B.1 + B.2 - 4 = 0 ∧
    -- Median AD: x - y + 1 = 0
    D.1 - D.2 + 1 = 0 ∧
    -- D is midpoint of BC
    D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧
    -- Point D: (4, 5)
    D = (4, 5)

-- Theorem statement
theorem isosceles_triangle_properties (A B C : ℝ × ℝ) 
  (h : Triangle A B C) : 
  -- Line BC: x + y - 9 = 0
  B.1 + B.2 - 9 = 0 ∧ C.1 + C.2 - 9 = 0 ∧
  -- Point B: (-5, 14)
  B = (-5, 14) ∧
  -- Point C: (13, -4)
  C = (13, -4) ∧
  -- Line AC: x + 2y - 5 = 0
  A.1 + 2 * A.2 - 5 = 0 ∧ C.1 + 2 * C.2 - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_properties_l2377_237733


namespace NUMINAMATH_CALUDE_domain_of_f_is_all_reals_l2377_237735

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (5 - 2 * x) ^ (1/3)

-- Theorem stating that the domain of f is all real numbers
theorem domain_of_f_is_all_reals :
  ∀ x : ℝ, ∃ y : ℝ, f x = y :=
by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_is_all_reals_l2377_237735


namespace NUMINAMATH_CALUDE_max_gold_tokens_l2377_237753

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  gold : ℕ

/-- Represents an exchange booth --/
structure Booth where
  red_in : ℕ
  blue_in : ℕ
  red_out : ℕ
  blue_out : ℕ
  gold_out : ℕ

/-- Checks if an exchange is possible at a given booth --/
def canExchange (tokens : TokenCount) (booth : Booth) : Bool :=
  tokens.red ≥ booth.red_in ∧ tokens.blue ≥ booth.blue_in

/-- Performs an exchange at a given booth --/
def exchange (tokens : TokenCount) (booth : Booth) : TokenCount :=
  { red := tokens.red - booth.red_in + booth.red_out,
    blue := tokens.blue - booth.blue_in + booth.blue_out,
    gold := tokens.gold + booth.gold_out }

/-- The main theorem to prove --/
theorem max_gold_tokens : ∃ (final : TokenCount),
  let initial := TokenCount.mk 100 100 0
  let booth1 := Booth.mk 3 0 0 2 1
  let booth2 := Booth.mk 0 4 2 0 1
  (¬ canExchange final booth1 ∧ ¬ canExchange final booth2) ∧
  final.gold = 133 ∧
  (∀ (other : TokenCount),
    (¬ canExchange other booth1 ∧ ¬ canExchange other booth2) →
    other.gold ≤ final.gold) :=
sorry

end NUMINAMATH_CALUDE_max_gold_tokens_l2377_237753


namespace NUMINAMATH_CALUDE_grade_distribution_l2377_237765

theorem grade_distribution (total_students : ℕ) 
  (fraction_A : ℚ) (fraction_C : ℚ) (number_D : ℕ) :
  total_students = 100 →
  fraction_A = 1/5 →
  fraction_C = 1/2 →
  number_D = 5 →
  (total_students : ℚ) - (fraction_A * total_students + fraction_C * total_students + number_D) = 1/4 * total_students :=
by sorry

end NUMINAMATH_CALUDE_grade_distribution_l2377_237765


namespace NUMINAMATH_CALUDE_smallest_solutions_l2377_237793

def is_solution (k : ℕ) : Prop :=
  Real.cos (k^2 + 8^2 : ℝ) ^ 2 = 1

theorem smallest_solutions :
  (∀ k : ℕ, k > 0 ∧ k < 48 → ¬ is_solution k) ∧
  is_solution 48 ∧
  (∀ k : ℕ, k > 48 ∧ k < 53 → ¬ is_solution k) ∧
  is_solution 53 :=
sorry

end NUMINAMATH_CALUDE_smallest_solutions_l2377_237793


namespace NUMINAMATH_CALUDE_cos_minus_sin_seventeen_fourths_pi_equals_sqrt_two_l2377_237704

theorem cos_minus_sin_seventeen_fourths_pi_equals_sqrt_two :
  Real.cos (-17/4 * Real.pi) - Real.sin (-17/4 * Real.pi) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_seventeen_fourths_pi_equals_sqrt_two_l2377_237704


namespace NUMINAMATH_CALUDE_arccos_sin_three_equals_three_minus_pi_half_l2377_237774

theorem arccos_sin_three_equals_three_minus_pi_half :
  Real.arccos (Real.sin 3) = 3 - π / 2 := by sorry

end NUMINAMATH_CALUDE_arccos_sin_three_equals_three_minus_pi_half_l2377_237774


namespace NUMINAMATH_CALUDE_m_range_l2377_237752

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- State the theorem
theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, f m x > -m + 2) → m > 3 := by
  sorry


end NUMINAMATH_CALUDE_m_range_l2377_237752


namespace NUMINAMATH_CALUDE_divisibility_criterion_l2377_237783

theorem divisibility_criterion (n : ℕ+) : 
  (n + 2 : ℕ) ∣ (n^3 + 3*n + 29 : ℕ) ↔ n = 1 ∨ n = 3 ∨ n = 13 :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l2377_237783


namespace NUMINAMATH_CALUDE_regular_octagon_area_l2377_237708

theorem regular_octagon_area (s : ℝ) (h : s = Real.sqrt 2) :
  let square_side : ℝ := 2 + s
  let octagon_area : ℝ := square_side ^ 2 - 4 * (1 / 2)
  octagon_area = 4 + 4 * s := by sorry

end NUMINAMATH_CALUDE_regular_octagon_area_l2377_237708


namespace NUMINAMATH_CALUDE_expression_value_l2377_237710

theorem expression_value : 2 * Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + 2 * Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2377_237710


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2377_237769

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2377_237769


namespace NUMINAMATH_CALUDE_binary_sum_equals_669_l2377_237703

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0 -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : ℕ :=
  b.foldl (fun acc digit => 2 * acc + if digit then 1 else 0) 0

/-- The binary number 111111111₂ -/
def b1 : BinaryNumber := [true, true, true, true, true, true, true, true, true]

/-- The binary number 1111111₂ -/
def b2 : BinaryNumber := [true, true, true, true, true, true, true]

/-- The binary number 11111₂ -/
def b3 : BinaryNumber := [true, true, true, true, true]

theorem binary_sum_equals_669 :
  binary_to_decimal b1 + binary_to_decimal b2 + binary_to_decimal b3 = 669 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_669_l2377_237703


namespace NUMINAMATH_CALUDE_sin_translation_to_cos_l2377_237750

theorem sin_translation_to_cos (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x + π / 6)
  let g (x : ℝ) := f (x + π / 6)
  g x = Real.cos (2 * x) := by
sorry

end NUMINAMATH_CALUDE_sin_translation_to_cos_l2377_237750


namespace NUMINAMATH_CALUDE_determinant_MN_equals_negative_5mn_l2377_237736

-- Define the determinant function for 2x2 matrices
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define polynomials M and N
def M (m n : ℝ) : ℝ := m^2 - 2*m*n
def N (m n : ℝ) : ℝ := 3*m^2 - m*n

-- State the theorem
theorem determinant_MN_equals_negative_5mn (m n : ℝ) :
  det2x2 (M m n) (N m n) 1 3 = -5*m*n := by sorry

end NUMINAMATH_CALUDE_determinant_MN_equals_negative_5mn_l2377_237736


namespace NUMINAMATH_CALUDE_lattice_point_probability_l2377_237716

theorem lattice_point_probability (d : ℝ) : 
  (d > 0) → 
  (π * d^2 = 3/4) → 
  (d = Real.sqrt (3 / (4 * π))) :=
sorry

end NUMINAMATH_CALUDE_lattice_point_probability_l2377_237716


namespace NUMINAMATH_CALUDE_at_least_one_correct_l2377_237781

theorem at_least_one_correct (p_a p_b : ℚ) 
  (h_a : p_a = 3/5) 
  (h_b : p_b = 2/5) : 
  1 - (1 - p_a) * (1 - p_b) = 19/25 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_correct_l2377_237781


namespace NUMINAMATH_CALUDE_second_floor_cost_l2377_237731

/-- Represents the cost of rooms on each floor of an apartment building --/
structure ApartmentCosts where
  first_floor : ℕ
  second_floor : ℕ
  third_floor : ℕ

/-- Calculates the total monthly income from all rooms --/
def total_income (costs : ApartmentCosts) : ℕ :=
  3 * (costs.first_floor + costs.second_floor + costs.third_floor)

/-- Theorem stating the cost of rooms on the second floor --/
theorem second_floor_cost (costs : ApartmentCosts) :
  costs.first_floor = 15 →
  costs.third_floor = 2 * costs.first_floor →
  total_income costs = 165 →
  costs.second_floor = 10 := by
  sorry

#check second_floor_cost

end NUMINAMATH_CALUDE_second_floor_cost_l2377_237731


namespace NUMINAMATH_CALUDE_rain_problem_l2377_237734

theorem rain_problem (first_hour : ℝ) (second_hour : ℝ) : 
  (second_hour = 2 * first_hour + 7) → 
  (first_hour + second_hour = 22) → 
  (first_hour = 5) := by
sorry

end NUMINAMATH_CALUDE_rain_problem_l2377_237734


namespace NUMINAMATH_CALUDE_roots_sum_value_l2377_237775

-- Define the quadratic equation
def quadratic (x : ℝ) : Prop := x^2 - x - 1 = 0

-- Define the roots a and b
variable (a b : ℝ)

-- State the theorem
theorem roots_sum_value (ha : quadratic a) (hb : quadratic b) (hab : a ≠ b) :
  3 * a^2 + 4 * b + 2 / a^2 = 11 := by sorry

end NUMINAMATH_CALUDE_roots_sum_value_l2377_237775


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2377_237791

/-- A quadratic equation in x with parameter m, where one root is zero -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 + x + m^2 + 3*m = 0

/-- The theorem stating that m = -3 for the given quadratic equation -/
theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, quadratic_equation m x) ∧ 
  (quadratic_equation m 0) ∧ 
  (m ≠ 0) → 
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2377_237791


namespace NUMINAMATH_CALUDE_students_math_or_history_not_both_l2377_237776

theorem students_math_or_history_not_both 
  (both : ℕ) 
  (math_total : ℕ) 
  (history_only : ℕ) 
  (h1 : both = 15) 
  (h2 : math_total = 30) 
  (h3 : history_only = 12) : 
  (math_total - both) + history_only = 27 := by
  sorry

end NUMINAMATH_CALUDE_students_math_or_history_not_both_l2377_237776


namespace NUMINAMATH_CALUDE_expression_evaluation_l2377_237749

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(y+1) * y^x) / (y^(x+1) * x^y) = x / y := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2377_237749


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2377_237729

theorem repeating_decimal_sum : 
  (∃ (x y z : ℚ), 
    (1000 * x - x = 123) ∧ 
    (10000 * y - y = 4567) ∧ 
    (100 * z - z = 89) ∧ 
    (x + y + z = 14786 / 9999)) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2377_237729


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l2377_237770

theorem sqrt_sum_equals_eight : 
  Real.sqrt (18 - 8 * Real.sqrt 2) + Real.sqrt (18 + 8 * Real.sqrt 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l2377_237770


namespace NUMINAMATH_CALUDE_no_complete_set_in_matrix_l2377_237726

/-- Definition of the matrix A --/
def A (n : ℕ) (i j : ℕ) : ℕ :=
  if (i + j - 1) % n = 0 then n else (i + j - 1) % n

/-- Theorem statement --/
theorem no_complete_set_in_matrix (n : ℕ) (h_even : Even n) (h_pos : 0 < n) :
  ¬ ∃ σ : Fin n → Fin n, Function.Bijective σ ∧ (∀ i : Fin n, A n i.val (σ i).val = i.val + 1) :=
sorry

end NUMINAMATH_CALUDE_no_complete_set_in_matrix_l2377_237726


namespace NUMINAMATH_CALUDE_expansion_terms_example_l2377_237713

/-- The number of terms in the expansion of a product of two polynomials with distinct variables -/
def expansion_terms (m n : ℕ) : ℕ := m * n

/-- Theorem: The number of terms in the expansion of (a+b+c)(x+d+e+f+g+h) is 18 -/
theorem expansion_terms_example : expansion_terms 3 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_example_l2377_237713
