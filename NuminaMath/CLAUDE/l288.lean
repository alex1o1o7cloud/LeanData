import Mathlib

namespace NUMINAMATH_CALUDE_fourth_intersection_point_l288_28806

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve xy = 2 -/
def on_curve (p : Point) : Prop :=
  p.x * p.y = 2

/-- The circle that intersects the curve at four points -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point lies on the circle -/
def on_circle (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The theorem stating the fourth intersection point -/
theorem fourth_intersection_point (c : Circle) 
    (h1 : on_curve ⟨4, 1/2⟩ ∧ on_circle c ⟨4, 1/2⟩)
    (h2 : on_curve ⟨-2, -1⟩ ∧ on_circle c ⟨-2, -1⟩)
    (h3 : on_curve ⟨1/4, 8⟩ ∧ on_circle c ⟨1/4, 8⟩)
    (h4 : ∃ p, on_curve p ∧ on_circle c p ∧ p ≠ ⟨4, 1/2⟩ ∧ p ≠ ⟨-2, -1⟩ ∧ p ≠ ⟨1/4, 8⟩) :
    ∃ p, p = ⟨-1/8, -16⟩ ∧ on_curve p ∧ on_circle c p :=
sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l288_28806


namespace NUMINAMATH_CALUDE_sum_xy_given_condition_l288_28860

theorem sum_xy_given_condition (x y : ℝ) : 
  |x + 3| + (y - 2)^2 = 0 → x + y = -1 := by sorry

end NUMINAMATH_CALUDE_sum_xy_given_condition_l288_28860


namespace NUMINAMATH_CALUDE_number_problem_l288_28888

theorem number_problem (x : ℝ) : 2 * x - 12 = 20 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l288_28888


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l288_28819

theorem cubic_equation_solutions :
  let z₁ : ℂ := -3
  let z₂ : ℂ := (3/2) + (3/2) * Complex.I * Real.sqrt 3
  let z₃ : ℂ := (3/2) - (3/2) * Complex.I * Real.sqrt 3
  (∀ z : ℂ, z^3 = -27 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l288_28819


namespace NUMINAMATH_CALUDE_triangle_properties_l288_28833

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let ac := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (ab + bc + ac) / 2
  ab = 6 ∧ bc = 5 ∧ Real.sqrt (s * (s - ab) * (s - bc) * (s - ac)) = 9

-- Define an acute triangle
def AcuteTriangle (A B C : ℝ × ℝ) : Prop :=
  Triangle A B C ∧
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) > 0 ∧
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) > 0 ∧
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) > 0

theorem triangle_properties (A B C : ℝ × ℝ) :
  Triangle A B C →
  (∃ ac : ℝ, (ac = Real.sqrt 13 ∨ ac = Real.sqrt 109) ∧
   ac = Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)) ∧
  (AcuteTriangle A B C →
   ∃ angle_A : ℝ,
   Real.cos (2 * angle_A + π / 6) = (-5 * Real.sqrt 3 - 12) / 26) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l288_28833


namespace NUMINAMATH_CALUDE_mixture_temperature_swap_l288_28899

theorem mixture_temperature_swap (a b c : ℝ) :
  let x := a + b - c
  ∃ (m_a m_b : ℝ), m_a > 0 ∧ m_b > 0 ∧
    (m_a * (a - c) + m_b * (b - c) = 0) ∧
    (m_b * (a - x) + m_a * (b - x) = 0) :=
by sorry

end NUMINAMATH_CALUDE_mixture_temperature_swap_l288_28899


namespace NUMINAMATH_CALUDE_base_b_is_7_l288_28880

/-- Given a base b, this function represents the number 15 in that base -/
def number_15 (b : ℕ) : ℕ := b + 5

/-- Given a base b, this function represents the number 433 in that base -/
def number_433 (b : ℕ) : ℕ := 4*b^2 + 3*b + 3

/-- The theorem states that if the square of the number represented by 15 in base b
    equals the number represented by 433 in base b, then b must be 7 in base 10 -/
theorem base_b_is_7 : ∃ (b : ℕ), (number_15 b)^2 = number_433 b ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_b_is_7_l288_28880


namespace NUMINAMATH_CALUDE_cafeteria_green_apples_l288_28854

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 42

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 9

/-- The number of extra fruit the cafeteria ended up with -/
def extra_fruit : ℕ := 40

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 7

theorem cafeteria_green_apples :
  red_apples + green_apples - students_wanting_fruit = extra_fruit :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_green_apples_l288_28854


namespace NUMINAMATH_CALUDE_power_division_rule_l288_28848

theorem power_division_rule (x : ℝ) : x^6 / x^3 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l288_28848


namespace NUMINAMATH_CALUDE_multiple_problem_l288_28829

theorem multiple_problem (x y : ℕ) (k m : ℕ) : 
  x = 11 → 
  x + y = 55 → 
  y = k * x + m → 
  k = 4 ∧ m = 0 := by
sorry

end NUMINAMATH_CALUDE_multiple_problem_l288_28829


namespace NUMINAMATH_CALUDE_average_speed_two_segment_trip_l288_28863

theorem average_speed_two_segment_trip (d1 d2 v1 v2 : ℝ) 
  (h1 : d1 = 45) (h2 : d2 = 15) (h3 : v1 = 15) (h4 : v2 = 45) :
  (d1 + d2) / ((d1 / v1) + (d2 / v2)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_segment_trip_l288_28863


namespace NUMINAMATH_CALUDE_people_per_car_l288_28847

/-- Given 63 people and 3 cars, prove that each car will contain 21 people when evenly distributed. -/
theorem people_per_car (total_people : Nat) (num_cars : Nat) (people_per_car : Nat) : 
  total_people = 63 → num_cars = 3 → people_per_car * num_cars = total_people → people_per_car = 21 := by
  sorry

end NUMINAMATH_CALUDE_people_per_car_l288_28847


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l288_28851

theorem cubic_factorization_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 1001 * x^3 - 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 3458 :=
by sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l288_28851


namespace NUMINAMATH_CALUDE_geometric_increasing_condition_l288_28831

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_increasing_condition (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (is_increasing_sequence a ↔ a 1 < a 2 ∧ a 2 < a 3) :=
sorry

end NUMINAMATH_CALUDE_geometric_increasing_condition_l288_28831


namespace NUMINAMATH_CALUDE_complement_M_U_characterization_l288_28815

-- Define the universal set U
def U : Set Int := {x | ∃ k, x = 2 * k}

-- Define the set M
def M : Set Int := {x | ∃ k, x = 4 * k}

-- Define the complement of M with respect to U
def complement_M_U : Set Int := {x ∈ U | x ∉ M}

-- Theorem statement
theorem complement_M_U_characterization :
  complement_M_U = {x | ∃ k, x = 4 * k - 2} := by sorry

end NUMINAMATH_CALUDE_complement_M_U_characterization_l288_28815


namespace NUMINAMATH_CALUDE_event_probability_theorem_l288_28810

/-- Given an event A with constant probability in three independent trials, 
    if the probability of A occurring at least once is 63/64, 
    then the probability of A occurring exactly once is 9/64. -/
theorem event_probability_theorem (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^3 = 63/64) →
  (3 * p * (1 - p)^2 = 9/64) :=
by sorry

end NUMINAMATH_CALUDE_event_probability_theorem_l288_28810


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l288_28802

theorem quadratic_negative_root (a : ℝ) (h : a < 0) :
  ∃ (condition : Prop), condition → ∃ x : ℝ, x < 0 ∧ a * x^2 + 2*x + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l288_28802


namespace NUMINAMATH_CALUDE_impossible_configuration_l288_28890

theorem impossible_configuration : ¬ ∃ (arrangement : List ℕ) (sum : ℕ),
  (arrangement.toFinset = {1, 4, 9, 16, 25, 36, 49}) ∧
  (∀ radial_line : List ℕ, radial_line.sum = sum) ∧
  (∀ triangle : List ℕ, triangle.sum = sum) :=
by sorry

end NUMINAMATH_CALUDE_impossible_configuration_l288_28890


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l288_28812

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 6 = 4 → n ≤ 94 :=
by sorry

theorem ninety_four_satisfies_conditions : 94 < 100 ∧ 94 % 6 = 4 :=
by sorry

theorem ninety_four_is_largest : ∃ (n : ℕ), n = 94 ∧ n < 100 ∧ n % 6 = 4 ∧ ∀ (m : ℕ), m < 100 ∧ m % 6 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l288_28812


namespace NUMINAMATH_CALUDE_joan_football_games_l288_28853

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan went to -/
def total_games : ℕ := games_this_year + games_last_year

theorem joan_football_games : total_games = 13 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l288_28853


namespace NUMINAMATH_CALUDE_purchase_price_calculation_l288_28814

theorem purchase_price_calculation (P : ℝ) : 0.05 * P + 12 = 30 → P = 360 := by
  sorry

end NUMINAMATH_CALUDE_purchase_price_calculation_l288_28814


namespace NUMINAMATH_CALUDE_drama_club_problem_l288_28877

theorem drama_club_problem (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) (drama_only : ℕ)
  (h_total : total = 75)
  (h_math : math = 42)
  (h_physics : physics = 35)
  (h_both : both = 25)
  (h_drama_only : drama_only = 10) :
  total - ((math + physics - both) + drama_only) = 13 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_problem_l288_28877


namespace NUMINAMATH_CALUDE_couples_matching_l288_28828

structure Couple where
  wife : String
  husband : String
  wife_bottles : ℕ
  husband_bottles : ℕ

def total_bottles : ℕ := 44

def couples : List Couple := [
  ⟨"Anna", "", 2, 0⟩,
  ⟨"Betty", "", 3, 0⟩,
  ⟨"Carol", "", 4, 0⟩,
  ⟨"Dorothy", "", 5, 0⟩
]

def husbands : List String := ["Brown", "Green", "White", "Smith"]

theorem couples_matching :
  ∃ (matched_couples : List Couple),
    matched_couples.length = 4 ∧
    (matched_couples.map (λ c => c.wife_bottles + c.husband_bottles)).sum = total_bottles ∧
    (∃ c ∈ matched_couples, c.wife = "Anna" ∧ c.husband = "Smith" ∧ c.husband_bottles = 4 * c.wife_bottles) ∧
    (∃ c ∈ matched_couples, c.wife = "Betty" ∧ c.husband = "White" ∧ c.husband_bottles = 3 * c.wife_bottles) ∧
    (∃ c ∈ matched_couples, c.wife = "Carol" ∧ c.husband = "Green" ∧ c.husband_bottles = 2 * c.wife_bottles) ∧
    (∃ c ∈ matched_couples, c.wife = "Dorothy" ∧ c.husband = "Brown" ∧ c.husband_bottles = c.wife_bottles) ∧
    (matched_couples.map (λ c => c.husband)).toFinset = husbands.toFinset :=
by sorry

end NUMINAMATH_CALUDE_couples_matching_l288_28828


namespace NUMINAMATH_CALUDE_problem_solution_l288_28864

theorem problem_solution (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^2 * b - 2 * a * b + a * b^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l288_28864


namespace NUMINAMATH_CALUDE_prime_sum_of_squares_l288_28868

/-- A number is prime if it's greater than 1 and its only divisors are 1 and itself -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The expression we're interested in -/
def expression (p q : ℕ) : ℕ :=
  2^2 + p^2 + q^2

theorem prime_sum_of_squares : 
  ∀ p q : ℕ, isPrime p → isPrime q → isPrime (expression p q) → 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
sorry

end NUMINAMATH_CALUDE_prime_sum_of_squares_l288_28868


namespace NUMINAMATH_CALUDE_special_quadrilateral_integer_perimeter_l288_28816

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  O : ℝ × ℝ
  -- Perpendicular conditions
  ab_perp_bc : (A.1 - B.1) * (B.1 - C.1) + (A.2 - B.2) * (B.2 - C.2) = 0
  bc_perp_cd : (B.1 - C.1) * (C.1 - D.1) + (B.2 - C.2) * (C.2 - D.2) = 0
  -- BC tangent to circle condition
  bc_tangent : (B.1 - O.1) * (C.1 - O.1) + (B.2 - O.2) * (C.2 - O.2) = 0
  -- AD is diameter
  ad_diameter : (A.1 - O.1)^2 + (A.2 - O.2)^2 = (D.1 - O.1)^2 + (D.2 - O.2)^2

/-- Perimeter of the quadrilateral is an integer when AB and CD are integers with AB = 2CD -/
theorem special_quadrilateral_integer_perimeter 
  (q : SpecialQuadrilateral) 
  (ab cd : ℕ) 
  (h_ab : ab = 2 * cd) 
  (h_ab_length : (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2 = ab^2) 
  (h_cd_length : (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 = cd^2) :
  ∃ (n : ℕ), 
    (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2 +
    (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2 +
    (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 +
    (q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_integer_perimeter_l288_28816


namespace NUMINAMATH_CALUDE_one_female_selection_count_l288_28875

/-- The number of male students in Group A -/
def group_a_male : ℕ := 5

/-- The number of female students in Group A -/
def group_a_female : ℕ := 3

/-- The number of male students in Group B -/
def group_b_male : ℕ := 6

/-- The number of female students in Group B -/
def group_b_female : ℕ := 2

/-- The number of students to be selected from each group -/
def students_per_group : ℕ := 2

/-- The total number of selections with exactly one female student -/
def total_selections : ℕ := 345

theorem one_female_selection_count :
  (Nat.choose group_a_male 1 * Nat.choose group_a_female 1 * Nat.choose group_b_male 2) +
  (Nat.choose group_a_male 2 * Nat.choose group_b_male 1 * Nat.choose group_b_female 1) = total_selections :=
by sorry

end NUMINAMATH_CALUDE_one_female_selection_count_l288_28875


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l288_28871

theorem simplify_complex_fraction :
  1 / (1 / (Real.sqrt 2 + 1) + 2 / (Real.sqrt 3 - 1) + 3 / (Real.sqrt 5 + 2)) =
  1 / (Real.sqrt 2 + 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 5) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l288_28871


namespace NUMINAMATH_CALUDE_midpoint_distance_midpoint_path_l288_28885

/-- Represents a ladder sliding down a wall --/
structure SlidingLadder where
  L : ℝ  -- Length of the ladder
  x : ℝ  -- Horizontal distance from wall to bottom of ladder
  y : ℝ  -- Vertical distance from floor to top of ladder
  h_positive : L > 0  -- Ladder has positive length
  h_pythagorean : x^2 + y^2 = L^2  -- Pythagorean theorem

/-- The midpoint of a sliding ladder is always L/2 distance from the corner --/
theorem midpoint_distance (ladder : SlidingLadder) :
  (ladder.x / 2)^2 + (ladder.y / 2)^2 = (ladder.L / 2)^2 := by
  sorry

/-- The path of the midpoint forms a quarter circle --/
theorem midpoint_path (ladder : SlidingLadder) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (0, 0) ∧ 
    radius = ladder.L / 2 ∧
    (ladder.x / 2)^2 + (ladder.y / 2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_midpoint_path_l288_28885


namespace NUMINAMATH_CALUDE_factorization_2mn_minus_6m_l288_28882

theorem factorization_2mn_minus_6m (m n : ℝ) : 2*m*n - 6*m = 2*m*(n - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2mn_minus_6m_l288_28882


namespace NUMINAMATH_CALUDE_product_of_solutions_l288_28821

theorem product_of_solutions (y : ℝ) : 
  (∃ y₁ y₂ : ℝ, (|3 * y₁| = 2 * (|3 * y₁| - 1) ∧ 
                 |3 * y₂| = 2 * (|3 * y₂| - 1) ∧ 
                 y₁ ≠ y₂ ∧
                 (∀ y₃ : ℝ, |3 * y₃| = 2 * (|3 * y₃| - 1) → y₃ = y₁ ∨ y₃ = y₂)) →
                 y₁ * y₂ = -4/9) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l288_28821


namespace NUMINAMATH_CALUDE_tan_graph_problem_l288_28892

theorem tan_graph_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.tan (b * x) = 3 → x = π / 4) →
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + 3 * π / 4))) →
  a * b = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_graph_problem_l288_28892


namespace NUMINAMATH_CALUDE_boat_travel_l288_28807

theorem boat_travel (boat_speed : ℝ) (time_against : ℝ) (time_with : ℝ)
  (h1 : boat_speed = 12)
  (h2 : time_against = 10)
  (h3 : time_with = 6) :
  ∃ (current_speed : ℝ) (distance : ℝ),
    current_speed = 3 ∧
    distance = 90 ∧
    (boat_speed - current_speed) * time_against = (boat_speed + current_speed) * time_with :=
by sorry

end NUMINAMATH_CALUDE_boat_travel_l288_28807


namespace NUMINAMATH_CALUDE_ten_factorial_divided_by_nine_factorial_l288_28850

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem ten_factorial_divided_by_nine_factorial : 
  factorial 10 / factorial 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_factorial_divided_by_nine_factorial_l288_28850


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l288_28845

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  ¬ (∀ a b c : ℝ, a > b → b > c → a * c > b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l288_28845


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l288_28808

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 15| + |x - 25| = |3*x - 75| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l288_28808


namespace NUMINAMATH_CALUDE_prime_product_divisors_l288_28891

theorem prime_product_divisors (p q : ℕ) (n : ℕ) : 
  Prime p → Prime q → (Finset.card (Nat.divisors (p^n * q^7)) = 56) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_divisors_l288_28891


namespace NUMINAMATH_CALUDE_cube_root_equation_sum_l288_28811

theorem cube_root_equation_sum (x y z : ℕ+) :
  (4 : ℝ) * ((7 : ℝ)^(1/3) - (6 : ℝ)^(1/3))^(1/2) = x.val^(1/3) + y.val^(1/3) - z.val^(1/3) →
  x.val + y.val + z.val = 51 := by
sorry

end NUMINAMATH_CALUDE_cube_root_equation_sum_l288_28811


namespace NUMINAMATH_CALUDE_distributive_property_only_true_l288_28843

open Real

theorem distributive_property_only_true : ∀ b x y : ℝ,
  (b * (x + y) = b * x + b * y) ∧
  (b^(x + y) ≠ b^x + b^y) ∧
  (log (x + y) ≠ log x + log y) ∧
  (log x / log y ≠ log x - log y) ∧
  (b * (x / y) ≠ b * x / (b * y)) :=
by sorry

end NUMINAMATH_CALUDE_distributive_property_only_true_l288_28843


namespace NUMINAMATH_CALUDE_ngon_existence_uniqueness_l288_28842

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields

/-- Represents a point in a plane -/
structure Point where
  -- Add necessary fields

/-- Represents an n-gon -/
structure Polygon (n : ℕ) where
  vertices : Fin n → Point

/-- Checks if a line is perpendicular to a side of a polygon at its midpoint -/
def is_perpendicular_at_midpoint (l : Line) (p : Polygon n) (i : Fin n) : Prop :=
  sorry

/-- Checks if a line is a bisector of an internal or external angle of a polygon -/
def is_angle_bisector (l : Line) (p : Polygon n) (i : Fin n) : Prop :=
  sorry

/-- Represents the solution status of the problem -/
inductive SolutionStatus
| Unique
| Indeterminate
| NoSolution

/-- The main theorem stating the existence and uniqueness of the n-gon -/
theorem ngon_existence_uniqueness 
  (n : ℕ) 
  (lines : Fin n → Line) 
  (condition : (l : Line) → (p : Polygon n) → (i : Fin n) → Prop) : 
  SolutionStatus :=
sorry

end NUMINAMATH_CALUDE_ngon_existence_uniqueness_l288_28842


namespace NUMINAMATH_CALUDE_quadratic_function_property_l288_28824

-- Define the quadratic function
variable (f : ℝ → ℝ)

-- Define the interval [a, b]
variable (a b : ℝ)

-- Define the axis of symmetry
def axis_of_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f x = f x

-- Define the range condition
def range_condition (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ y ∈ Set.Icc (f b) (f a), ∃ x ∈ Set.Icc a b, f x = y

-- Theorem statement
theorem quadratic_function_property
  (h_axis : axis_of_symmetry f)
  (h_range : range_condition f a b) :
  ∀ x, x ∉ Set.Ioo a b :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l288_28824


namespace NUMINAMATH_CALUDE_smallest_other_integer_l288_28820

theorem smallest_other_integer (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 →
  (m = 72 ∨ n = 72) →
  Nat.gcd m n = x + 7 →
  Nat.lcm m n = x^2 * (x + 7) →
  (m ≠ 72 → m ≥ 15309) ∧ (n ≠ 72 → n ≥ 15309) :=
sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l288_28820


namespace NUMINAMATH_CALUDE_original_speed_correct_l288_28887

/-- The original speed of the car traveling between two locations. -/
def original_speed : ℝ := 80

/-- The distance between location A and location B in kilometers. -/
def distance : ℝ := 160

/-- The increase in speed as a percentage. -/
def speed_increase : ℝ := 0.25

/-- The time saved due to the increased speed, in hours. -/
def time_saved : ℝ := 0.4

/-- Theorem stating that the original speed satisfies the given conditions. -/
theorem original_speed_correct :
  distance / original_speed - distance / (original_speed * (1 + speed_increase)) = time_saved := by
  sorry

end NUMINAMATH_CALUDE_original_speed_correct_l288_28887


namespace NUMINAMATH_CALUDE_book_pages_calculation_l288_28862

theorem book_pages_calculation (pages_per_day : ℕ) (days_to_finish : ℕ) : 
  pages_per_day = 8 → days_to_finish = 72 → pages_per_day * days_to_finish = 576 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l288_28862


namespace NUMINAMATH_CALUDE_duplicated_page_number_l288_28874

/-- The sum of natural numbers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem statement -/
theorem duplicated_page_number :
  ∀ k : ℕ,
  (k ≤ 70) →
  (sum_to_n 70 + k = 2550) →
  (k = 65) :=
by sorry

end NUMINAMATH_CALUDE_duplicated_page_number_l288_28874


namespace NUMINAMATH_CALUDE_smaller_box_size_l288_28813

/-- Represents the size and cost of a box of macaroni and cheese -/
structure MacaroniBox where
  size : Float
  cost : Float

/-- Calculates the price per ounce of a MacaroniBox -/
def pricePerOunce (box : MacaroniBox) : Float :=
  box.cost / box.size

theorem smaller_box_size 
  (larger_box : MacaroniBox)
  (smaller_box : MacaroniBox)
  (better_value_price : Float)
  (h1 : larger_box.size = 30)
  (h2 : larger_box.cost = 4.80)
  (h3 : smaller_box.cost = 3.40)
  (h4 : better_value_price = 0.16)
  (h5 : pricePerOunce larger_box ≤ pricePerOunce smaller_box)
  (h6 : pricePerOunce larger_box = better_value_price) :
  smaller_box.size = 21.25 := by
  sorry

#check smaller_box_size

end NUMINAMATH_CALUDE_smaller_box_size_l288_28813


namespace NUMINAMATH_CALUDE_parabola_properties_l288_28872

/-- Represents a parabola of the form y = a(x-3)^2 + 2 -/
structure Parabola where
  a : ℝ

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating properties of a specific parabola -/
theorem parabola_properties (p : Parabola) (A B : Point) :
  (p.a * (1 - 3)^2 + 2 = -2) →  -- parabola passes through (1, -2)
  (A.y = p.a * (A.x - 3)^2 + 2) →  -- point A is on the parabola
  (B.y = p.a * (B.x - 3)^2 + 2) →  -- point B is on the parabola
  (A.x < B.x) →  -- m < n
  (B.x < 3) →  -- n < 3
  (p.a = -1 ∧ A.y < B.y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l288_28872


namespace NUMINAMATH_CALUDE_inequality_implies_a_nonpositive_l288_28839

theorem inequality_implies_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, x ∈ [1, 2] → 4^x - 2^(x+1) - a ≥ 0) →
  a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_nonpositive_l288_28839


namespace NUMINAMATH_CALUDE_stage_8_area_l288_28835

/-- The area of the rectangle at a given stage in the square-adding process -/
def rectangleArea (stage : ℕ) : ℕ :=
  stage * (4 * 4)

/-- Theorem: The area of the rectangle at Stage 8 is 128 square inches -/
theorem stage_8_area : rectangleArea 8 = 128 := by
  sorry

end NUMINAMATH_CALUDE_stage_8_area_l288_28835


namespace NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_five_thirds_l288_28852

/-- A structure representing a nested square figure -/
structure NestedSquareFigure where
  /-- The number of nested squares in the figure -/
  num_squares : ℕ
  /-- Predicate ensuring inner squares have vertices at midpoints of outer squares -/
  midpoint_property : num_squares > 1 → True

/-- The ratio of shaded to unshaded area in a nested square figure -/
def shaded_to_unshaded_ratio (figure : NestedSquareFigure) : Rat :=
  5 / 3

/-- Theorem stating the ratio of shaded to unshaded area is 5:3 -/
theorem shaded_to_unshaded_ratio_is_five_thirds (figure : NestedSquareFigure) :
  shaded_to_unshaded_ratio figure = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_five_thirds_l288_28852


namespace NUMINAMATH_CALUDE_blue_to_red_ratio_l288_28857

theorem blue_to_red_ratio (n : ℕ) (h : n = 13) : 
  (6 * n^3 - 6 * n^2) / (6 * n^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_blue_to_red_ratio_l288_28857


namespace NUMINAMATH_CALUDE_class_size_difference_l288_28897

theorem class_size_difference (A B : ℕ) (h : A - 4 = B + 4) : A - B = 8 := by
  sorry

end NUMINAMATH_CALUDE_class_size_difference_l288_28897


namespace NUMINAMATH_CALUDE_lenny_remaining_amount_l288_28884

def calculate_remaining_amount (initial_amount : ℝ) 
  (console_price game_price headphones_price : ℝ)
  (book1_price book2_price book3_price : ℝ)
  (tech_discount tech_tax bookstore_fee : ℝ) : ℝ :=
  let tech_total := console_price + 2 * game_price + headphones_price
  let tech_discounted := tech_total * (1 - tech_discount)
  let tech_with_tax := tech_discounted * (1 + tech_tax)
  let book_total := book1_price + book2_price
  let bookstore_total := book_total * (1 + bookstore_fee)
  let total_spent := tech_with_tax + bookstore_total
  initial_amount - total_spent

theorem lenny_remaining_amount :
  calculate_remaining_amount 500 200 50 75 25 30 15 0.2 0.1 0.02 = 113.90 := by
  sorry

end NUMINAMATH_CALUDE_lenny_remaining_amount_l288_28884


namespace NUMINAMATH_CALUDE_cafeteria_duty_assignments_l288_28893

def class_size : ℕ := 28
def duty_size : ℕ := 4

theorem cafeteria_duty_assignments :
  (Nat.choose class_size duty_size = 20475) ∧
  (Nat.choose (class_size - 1) (duty_size - 1) = 2925) := by
  sorry

#check cafeteria_duty_assignments

end NUMINAMATH_CALUDE_cafeteria_duty_assignments_l288_28893


namespace NUMINAMATH_CALUDE_cylinder_radius_comparison_l288_28876

theorem cylinder_radius_comparison (h : ℝ) (r₁ : ℝ) (r₂ : ℝ) : 
  h > 0 → r₁ > 0 → r₂ > 0 → h = 4 → r₁ = 6 → 
  (π * r₂^2 * h = 3 * (π * r₁^2 * h)) → r₂ = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_comparison_l288_28876


namespace NUMINAMATH_CALUDE_honey_servings_per_ounce_l288_28898

/-- Represents the number of servings of honey per ounce -/
def servings_per_ounce : ℕ := sorry

/-- Represents the number of servings used per cup of tea -/
def servings_per_cup : ℕ := 1

/-- Represents the number of cups of tea consumed per night -/
def cups_per_night : ℕ := 2

/-- Represents the size of the honey container in ounces -/
def container_size : ℕ := 16

/-- Represents the number of nights the honey lasts -/
def nights_lasted : ℕ := 48

theorem honey_servings_per_ounce :
  servings_per_ounce = 6 :=
sorry

end NUMINAMATH_CALUDE_honey_servings_per_ounce_l288_28898


namespace NUMINAMATH_CALUDE_polynomial_expansion_l288_28856

theorem polynomial_expansion (z : ℝ) :
  (3 * z^2 - 4 * z + 1) * (2 * z^3 + 3 * z^2 - 5 * z + 2) =
  6 * z^5 + z^4 - 25 * z^3 + 29 * z^2 - 13 * z + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l288_28856


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l288_28805

theorem product_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 30)
  (first_eq : x = 3 * (y + z))
  (second_eq : y = 5 * z) :
  x * y * z = 175.78125 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l288_28805


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l288_28834

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l288_28834


namespace NUMINAMATH_CALUDE_statement_contrapositive_and_negation_l288_28895

theorem statement_contrapositive_and_negation (x y : ℝ) :
  (((x - 1) * (y + 2) = 0 → x = 1 ∨ y = -2) ↔
   (x ≠ 1 ∧ y ≠ -2 → (x - 1) * (y + 2) ≠ 0)) ∧
  (¬((x - 1) * (y + 2) = 0 → x = 1 ∨ y = -2) ↔
   ((x - 1) * (y + 2) = 0 → x ≠ 1 ∧ y ≠ -2)) :=
by sorry

end NUMINAMATH_CALUDE_statement_contrapositive_and_negation_l288_28895


namespace NUMINAMATH_CALUDE_valid_card_distribution_l288_28830

/-- Represents the distribution of cards among three people. -/
structure CardDistribution :=
  (xiaoming : ℕ)
  (xiaohua : ℕ)
  (xiaogang : ℕ)

/-- Checks if the given distribution satisfies all constraints. -/
def is_valid_distribution (d : CardDistribution) : Prop :=
  d.xiaoming + d.xiaohua + d.xiaogang = 363 ∧
  7 * d.xiaohua = 6 * d.xiaoming ∧
  8 * d.xiaoming = 5 * d.xiaogang

/-- The theorem stating that the given distribution is valid. -/
theorem valid_card_distribution :
  is_valid_distribution ⟨105, 90, 168⟩ := by
  sorry

#check valid_card_distribution

end NUMINAMATH_CALUDE_valid_card_distribution_l288_28830


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l288_28858

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x₀ : ℝ, x₀^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l288_28858


namespace NUMINAMATH_CALUDE_bench_placement_l288_28823

theorem bench_placement (path_length : ℕ) (interval : ℕ) (bench_count : ℕ) : 
  path_length = 120 ∧ 
  interval = 10 ∧ 
  bench_count = (path_length / interval) + 1 →
  bench_count = 13 := by
  sorry

end NUMINAMATH_CALUDE_bench_placement_l288_28823


namespace NUMINAMATH_CALUDE_tony_running_distance_tony_running_distance_proof_l288_28849

/-- Proves that Tony runs 10 miles without the backpack each morning given his exercise routine. -/
theorem tony_running_distance : ℝ → Prop :=
  fun x =>
    let walk_distance : ℝ := 3
    let walk_speed : ℝ := 3
    let run_speed : ℝ := 5
    let total_exercise_time : ℝ := 21
    let days_per_week : ℝ := 7
    
    let daily_walk_time : ℝ := walk_distance / walk_speed
    let daily_run_time : ℝ := x / run_speed
    let weekly_exercise_time : ℝ := days_per_week * (daily_walk_time + daily_run_time)
    
    weekly_exercise_time = total_exercise_time → x = 10

/-- The proof of the theorem. -/
theorem tony_running_distance_proof : tony_running_distance 10 := by
  sorry

end NUMINAMATH_CALUDE_tony_running_distance_tony_running_distance_proof_l288_28849


namespace NUMINAMATH_CALUDE_average_difference_l288_28844

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 150) : 
  a - c = -80 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l288_28844


namespace NUMINAMATH_CALUDE_smallest_isosceles_perimeter_square_l288_28886

/-- A number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ¬ Nat.Prime n

/-- A natural number is a perfect square if it's equal to some integer squared -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The perimeter of an isosceles triangle with two sides of length a and one side of length b -/
def IsoscelesPerimeter (a b : ℕ) : ℕ := 2 * a + b

theorem smallest_isosceles_perimeter_square : 
  ∀ a b : ℕ, 
    IsComposite a → 
    IsComposite b → 
    a ≠ b → 
    IsPerfectSquare ((2 * a + b) * (2 * a + b)) → 
    2 * a > b → 
    a + b > a →
    ∀ c d : ℕ, 
      IsComposite c → 
      IsComposite d → 
      c ≠ d → 
      IsPerfectSquare ((2 * c + d) * (2 * c + d)) → 
      2 * c > d → 
      c + d > c →
      (IsoscelesPerimeter a b) * (IsoscelesPerimeter a b) ≤ (IsoscelesPerimeter c d) * (IsoscelesPerimeter c d) → 
      (IsoscelesPerimeter a b) * (IsoscelesPerimeter a b) = 256 :=
by sorry

end NUMINAMATH_CALUDE_smallest_isosceles_perimeter_square_l288_28886


namespace NUMINAMATH_CALUDE_polynomial_property_l288_28865

-- Define the polynomial Q(x)
def Q (p q d : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

-- State the theorem
theorem polynomial_property (p q d : ℝ) :
  -- The mean of zeros equals the product of zeros taken two at a time
  (-p/3 = q) →
  -- The mean of zeros equals the sum of coefficients
  (-p/3 = 1 + p + q + d) →
  -- The y-intercept is 5
  (Q p q d 0 = 5) →
  -- Then q = 2
  q = 2 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l288_28865


namespace NUMINAMATH_CALUDE_count_monomials_l288_28826

-- Define what a monomial is
def is_monomial (term : String) : Bool :=
  match term with
  | "0" => true  -- 0 is considered a monomial
  | t => (t.count '+' = 0) ∧ (t.count '-' ≤ 1) ∧ (t.count '/' = 0)  -- Simplified check for monomials

-- Define the list of terms in the expression
def expression : List String := ["1/x", "x+y", "0", "-a", "-3x^2y", "(x+1)/3"]

-- State the theorem
theorem count_monomials : 
  (expression.filter is_monomial).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_monomials_l288_28826


namespace NUMINAMATH_CALUDE_sixth_quiz_score_for_target_mean_l288_28841

def quiz_scores : List ℕ := [92, 96, 87, 89, 100]
def target_mean : ℕ := 94
def num_quizzes : ℕ := 6

theorem sixth_quiz_score_for_target_mean :
  ∃ (x : ℕ), (quiz_scores.sum + x) / num_quizzes = target_mean ∧ x = 100 := by
sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_for_target_mean_l288_28841


namespace NUMINAMATH_CALUDE_solve_equation_l288_28832

theorem solve_equation (x : ℝ) (h : 7 * (x - 1) = 21) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l288_28832


namespace NUMINAMATH_CALUDE_smallest_sum_abc_l288_28859

theorem smallest_sum_abc (a b c : ℕ+) (h : (3 : ℕ) * a.val = (4 : ℕ) * b.val ∧ (4 : ℕ) * b.val = (7 : ℕ) * c.val) : 
  (a.val + b.val + c.val : ℕ) ≥ 61 ∧ ∃ (a' b' c' : ℕ+), (3 : ℕ) * a'.val = (4 : ℕ) * b'.val ∧ (4 : ℕ) * b'.val = (7 : ℕ) * c'.val ∧ a'.val + b'.val + c'.val = 61 :=
by
  sorry

#check smallest_sum_abc

end NUMINAMATH_CALUDE_smallest_sum_abc_l288_28859


namespace NUMINAMATH_CALUDE_profit_at_8750_max_profit_price_l288_28800

-- Define constants
def cost_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_monthly_sales : ℝ := 500
def price_increase_step : ℝ := 1
def sales_decrease_step : ℝ := 10

-- Define functions
def selling_price (x : ℝ) : ℝ := initial_selling_price + x
def monthly_sales (x : ℝ) : ℝ := initial_monthly_sales - sales_decrease_step * x
def monthly_profit (x : ℝ) : ℝ := (monthly_sales x) * (selling_price x - cost_price)

-- Theorem statements
theorem profit_at_8750 (x : ℝ) : 
  monthly_profit x = 8750 → (x = 25 ∨ x = 15) := by sorry

theorem max_profit_price : 
  ∃ x : ℝ, ∀ y : ℝ, monthly_profit x ≥ monthly_profit y ∧ selling_price x = 70 := by sorry

end NUMINAMATH_CALUDE_profit_at_8750_max_profit_price_l288_28800


namespace NUMINAMATH_CALUDE_jake_initial_bitcoins_l288_28801

def initial_bitcoins : ℕ → Prop
| b => let after_first_donation := b - 20
       let after_giving_half := (after_first_donation) / 2
       let after_tripling := 3 * after_giving_half
       let final_amount := after_tripling - 10
       final_amount = 80

theorem jake_initial_bitcoins : initial_bitcoins 80 := by
  sorry

end NUMINAMATH_CALUDE_jake_initial_bitcoins_l288_28801


namespace NUMINAMATH_CALUDE_trajectory_of_M_l288_28883

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point A
def point_A : ℝ × ℝ := (1, 0)

-- Define a point Q on the circle
def point_Q (x y : ℝ) : Prop := circle_C x y

-- Define point M as the intersection of the perpendicular bisector of AQ and CQ
def point_M (x y : ℝ) (qx qy : ℝ) : Prop :=
  point_Q qx qy ∧
  (x - 1)^2 + y^2 = (x - qx)^2 + (y - qy)^2 ∧
  (x + 1) * (qx - x) + y * (qy - y) = 0

-- Theorem statement
theorem trajectory_of_M :
  ∀ x y : ℝ, (∃ qx qy : ℝ, point_M x y qx qy) →
  x^2 / 4 + y^2 / 3 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_M_l288_28883


namespace NUMINAMATH_CALUDE_count_equal_S_is_11_l288_28846

/-- S(n) is the smallest positive integer divisible by each of the positive integers 1, 2, 3, ..., n -/
def S (n : ℕ) : ℕ := sorry

/-- The count of positive integers n with 1 ≤ n ≤ 100 that have S(n) = S(n+4) -/
def count_equal_S : ℕ := sorry

theorem count_equal_S_is_11 : count_equal_S = 11 := by sorry

end NUMINAMATH_CALUDE_count_equal_S_is_11_l288_28846


namespace NUMINAMATH_CALUDE_line_through_point_with_angle_l288_28837

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parametric line -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Checks if a point lies on a parametric line -/
def pointOnLine (p : Point) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = p.x ∧ l.y t = p.y

/-- Calculates the angle between a parametric line and the positive x-axis -/
noncomputable def lineAngle (l : ParametricLine) : ℝ :=
  Real.arctan ((l.y 1 - l.y 0) / (l.x 1 - l.x 0))

theorem line_through_point_with_angle (M : Point) (θ : ℝ) :
  let l : ParametricLine := {
    x := λ t => 1 + (1/2) * t,
    y := λ t => 5 + (Real.sqrt 3 / 2) * t
  }
  pointOnLine M l ∧ lineAngle l = θ ∧ M.x = 1 ∧ M.y = 5 ∧ θ = π/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_with_angle_l288_28837


namespace NUMINAMATH_CALUDE_line_slope_angle_l288_28889

theorem line_slope_angle (x y : ℝ) :
  x - Real.sqrt 3 * y + 1 = 0 →
  Real.arctan (Real.sqrt 3 / 3) = 30 * Real.pi / 180 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l288_28889


namespace NUMINAMATH_CALUDE_equation_solution_sum_l288_28869

theorem equation_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 15 = 25) → 
  (d^2 - 6*d + 15 = 25) → 
  c ≥ d → 
  3*c + 2*d = 15 + Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_sum_l288_28869


namespace NUMINAMATH_CALUDE_new_boarders_correct_l288_28817

-- Define the initial conditions
def initial_boarders : ℕ := 120
def initial_ratio_boarders : ℕ := 2
def initial_ratio_day : ℕ := 5
def new_ratio_boarders : ℕ := 1
def new_ratio_day : ℕ := 2

-- Define the function to calculate the number of new boarders
def new_boarders : ℕ := 30

-- Theorem statement
theorem new_boarders_correct :
  let initial_day_students := (initial_boarders * initial_ratio_day) / initial_ratio_boarders
  (new_ratio_boarders * (initial_boarders + new_boarders)) = (new_ratio_day * initial_day_students) :=
by sorry

end NUMINAMATH_CALUDE_new_boarders_correct_l288_28817


namespace NUMINAMATH_CALUDE_toys_sold_l288_28870

def initial_toys : ℕ := 7
def remaining_toys : ℕ := 4

theorem toys_sold : initial_toys - remaining_toys = 3 := by
  sorry

end NUMINAMATH_CALUDE_toys_sold_l288_28870


namespace NUMINAMATH_CALUDE_perpendicular_to_vertical_line_l288_28855

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A vertical line represented by its x-coordinate -/
structure VerticalLine where
  x : ℝ

/-- Two lines are perpendicular if one is vertical and the other is horizontal -/
def isPerpendicular (l : Line) (v : VerticalLine) : Prop :=
  l.slope = 0

theorem perpendicular_to_vertical_line (k : ℝ) :
  isPerpendicular (Line.mk k 1) (VerticalLine.mk 1) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_to_vertical_line_l288_28855


namespace NUMINAMATH_CALUDE_solve_for_q_l288_28894

theorem solve_for_q (m n q : ℚ) : 
  (7/8 : ℚ) = m/96 ∧ 
  (7/8 : ℚ) = (n + m)/112 ∧ 
  (7/8 : ℚ) = (q - m)/144 → 
  q = 210 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l288_28894


namespace NUMINAMATH_CALUDE_parabola_focus_theorem_l288_28867

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = 4 * x^2 + 8 * x - 5

/-- The focus of a parabola y = a(x - h)^2 + k is at (h, k + 1/(4a)) -/
def parabola_focus (a h k x y : ℝ) : Prop :=
  x = h ∧ y = k + 1 / (4 * a)

/-- Theorem: The focus of the parabola y = 4x^2 + 8x - 5 is at (-1, -8.9375) -/
theorem parabola_focus_theorem :
  ∃ (x y : ℝ), parabola_equation x y ∧ parabola_focus 4 (-1) (-9) x y ∧ x = -1 ∧ y = -8.9375 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_theorem_l288_28867


namespace NUMINAMATH_CALUDE_system_solution_l288_28836

theorem system_solution (x y : ℝ) : 
  (4 * x + y = 6 ∧ 3 * x - y = 1) ↔ (x = 1 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l288_28836


namespace NUMINAMATH_CALUDE_function_property_l288_28866

open Set Function Real

theorem function_property (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x, x ≠ a → f x ≠ f a) :
  (∀ x, x ≠ a → f x ≠ f a) ∧ 
  ¬(∀ x, f x ≠ f a → x ≠ a) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l288_28866


namespace NUMINAMATH_CALUDE_derivative_at_e_l288_28838

open Real

theorem derivative_at_e (f : ℝ → ℝ) (h : Differentiable ℝ f) :
  (∀ x, f x = 2 * x * (deriv f e) - log x) →
  deriv f e = 1 / e :=
by sorry

end NUMINAMATH_CALUDE_derivative_at_e_l288_28838


namespace NUMINAMATH_CALUDE_trade_value_trade_value_correct_l288_28873

theorem trade_value (matt_cards : ℕ) (matt_card_value : ℕ) 
  (traded_cards : ℕ) (received_cheap_cards : ℕ) (cheap_card_value : ℕ) 
  (profit : ℕ) : ℕ :=
  let total_traded_value := traded_cards * matt_card_value
  let received_cheap_value := received_cheap_cards * cheap_card_value
  let total_received_value := total_traded_value + profit
  total_received_value - received_cheap_value

#check trade_value 8 6 2 3 2 3 = 9

theorem trade_value_correct : trade_value 8 6 2 3 2 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_trade_value_trade_value_correct_l288_28873


namespace NUMINAMATH_CALUDE_binary_subtraction_l288_28822

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def a : List Bool := [true, true, true, true, true, true, true, true, true]
def b : List Bool := [true, true, true, true]

theorem binary_subtraction :
  binary_to_decimal a - binary_to_decimal b = 496 := by
  sorry

end NUMINAMATH_CALUDE_binary_subtraction_l288_28822


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l288_28879

theorem sum_of_reciprocals (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 15) (h4 : a * b = 225) : 
  1 / a + 1 / b = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l288_28879


namespace NUMINAMATH_CALUDE_pi_arrangement_face_dots_l288_28825

/-- Represents a cube with dots on its faces -/
structure Cube where
  face1 : Nat
  face2 : Nat
  face3 : Nat
  face4 : Nat
  face5 : Nat
  face6 : Nat
  three_dot_face : face1 = 3 ∨ face2 = 3 ∨ face3 = 3 ∨ face4 = 3 ∨ face5 = 3 ∨ face6 = 3
  two_dot_faces : (face1 = 2 ∧ face2 = 2) ∨ (face1 = 2 ∧ face3 = 2) ∨ (face1 = 2 ∧ face4 = 2) ∨
                  (face1 = 2 ∧ face5 = 2) ∨ (face1 = 2 ∧ face6 = 2) ∨ (face2 = 2 ∧ face3 = 2) ∨
                  (face2 = 2 ∧ face4 = 2) ∨ (face2 = 2 ∧ face5 = 2) ∨ (face2 = 2 ∧ face6 = 2) ∨
                  (face3 = 2 ∧ face4 = 2) ∨ (face3 = 2 ∧ face5 = 2) ∨ (face3 = 2 ∧ face6 = 2) ∨
                  (face4 = 2 ∧ face5 = 2) ∨ (face4 = 2 ∧ face6 = 2) ∨ (face5 = 2 ∧ face6 = 2)
  one_dot_faces : face1 + face2 + face3 + face4 + face5 + face6 = 9

/-- Represents the "П" shape arrangement of cubes -/
structure PiArrangement where
  cubes : Fin 7 → Cube
  contacting_faces_same : ∀ i j, i ≠ j → (cubes i).face1 = (cubes j).face2

/-- The theorem to be proved -/
theorem pi_arrangement_face_dots (arr : PiArrangement) :
  ∃ (a b c : Cube), (a.face1 = 2 ∧ b.face1 = 2 ∧ c.face1 = 3) :=
sorry

end NUMINAMATH_CALUDE_pi_arrangement_face_dots_l288_28825


namespace NUMINAMATH_CALUDE_sin_20_cos_10_minus_cos_200_sin_10_l288_28818

open Real

theorem sin_20_cos_10_minus_cos_200_sin_10 :
  sin (20 * π / 180) * cos (10 * π / 180) - cos (200 * π / 180) * sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_20_cos_10_minus_cos_200_sin_10_l288_28818


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l288_28803

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 1}

-- Define set N
def N : Set ℝ := {x : ℝ | x ≤ 0}

-- State the theorem
theorem intersection_complement_equality :
  M ∩ (U \ N) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l288_28803


namespace NUMINAMATH_CALUDE_water_containers_capacity_l288_28896

/-- The problem of calculating the combined capacity of three water containers -/
theorem water_containers_capacity :
  ∀ (A B C : ℝ),
    (0.35 * A + 48 = 0.75 * A) →
    (0.45 * B + 36 = 0.95 * B) →
    (0.20 * C - 24 = 0.10 * C) →
    A + B + C = 432 :=
by sorry

end NUMINAMATH_CALUDE_water_containers_capacity_l288_28896


namespace NUMINAMATH_CALUDE_simplify_fraction_l288_28878

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  x / (x - 1)^2 - 1 / (x - 1)^2 = 1 / (x - 1) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l288_28878


namespace NUMINAMATH_CALUDE_shortest_wire_for_given_poles_l288_28861

/-- Represents a cylindrical pole with a given diameter -/
structure Pole where
  diameter : ℝ

/-- Calculates the shortest wire length to wrap around three poles -/
def shortestWireLength (pole1 pole2 pole3 : Pole) : ℝ :=
  sorry

/-- The theorem stating the shortest wire length for the given poles -/
theorem shortest_wire_for_given_poles :
  let pole1 : Pole := ⟨6⟩
  let pole2 : Pole := ⟨18⟩
  let pole3 : Pole := ⟨12⟩
  shortestWireLength pole1 pole2 pole3 = 6 * Real.sqrt 3 + 6 * Real.sqrt 6 + 18 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shortest_wire_for_given_poles_l288_28861


namespace NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l288_28881

theorem triangle_circumcircle_diameter 
  (a : ℝ) (B : ℝ) (S : ℝ) :
  a = 2 →
  B = π / 3 →  -- 60° in radians
  S = Real.sqrt 3 →
  (2 * S) / (a * Real.sin B) = 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l288_28881


namespace NUMINAMATH_CALUDE_vector_sum_zero_l288_28809

variable {V : Type*} [AddCommGroup V]
variable (A C D E : V)

theorem vector_sum_zero :
  (E - C) + (C - A) - (E - D) - (D - A) = (0 : V) := by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l288_28809


namespace NUMINAMATH_CALUDE_pepik_problem_l288_28827

def letter_sum (M A T R D E I K U : Nat) : Nat :=
  4*M + 4*A + R + D + 2*T + E + I + K + U

theorem pepik_problem :
  (∀ M A T R D E I K U : Nat,
    M ≤ 9 ∧ A ≤ 9 ∧ T ≤ 9 ∧ R ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧ I ≤ 9 ∧ K ≤ 9 ∧ U ≤ 9 ∧
    M ≠ 0 ∧ A ≠ 0 ∧ T ≠ 0 ∧ R ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ I ≠ 0 ∧ K ≠ 0 ∧ U ≠ 0 ∧
    M ≠ A ∧ M ≠ T ∧ M ≠ R ∧ M ≠ D ∧ M ≠ E ∧ M ≠ I ∧ M ≠ K ∧ M ≠ U ∧
    A ≠ T ∧ A ≠ R ∧ A ≠ D ∧ A ≠ E ∧ A ≠ I ∧ A ≠ K ∧ A ≠ U ∧
    T ≠ R ∧ T ≠ D ∧ T ≠ E ∧ T ≠ I ∧ T ≠ K ∧ T ≠ U ∧
    R ≠ D ∧ R ≠ E ∧ R ≠ I ∧ R ≠ K ∧ R ≠ U ∧
    D ≠ E ∧ D ≠ I ∧ D ≠ K ∧ D ≠ U ∧
    E ≠ I ∧ E ≠ K ∧ E ≠ U ∧
    I ≠ K ∧ I ≠ U ∧
    K ≠ U →
    (∀ x : Nat, letter_sum M A T R D E I K U ≤ 103) ∧
    (letter_sum M A T R D E I K U ≠ 50) ∧
    (letter_sum M A T R D E I K U = 59 → (T = 5 ∨ T = 2))) :=
by sorry

end NUMINAMATH_CALUDE_pepik_problem_l288_28827


namespace NUMINAMATH_CALUDE_translated_quadratic_increases_l288_28840

/-- Original quadratic function -/
def f (x : ℝ) : ℝ := -x^2 + 1

/-- Translated quadratic function -/
def g (x : ℝ) : ℝ := f (x - 2)

/-- Theorem stating that the translated function increases for x < 2 -/
theorem translated_quadratic_increases (x1 x2 : ℝ) 
  (h1 : x1 < 2) (h2 : x2 < 2) (h3 : x1 < x2) : 
  g x1 < g x2 := by
  sorry

end NUMINAMATH_CALUDE_translated_quadratic_increases_l288_28840


namespace NUMINAMATH_CALUDE_namjoon_marbles_l288_28804

def marble_problem (sets : ℕ) (marbles_per_set : ℕ) (boxes : ℕ) (marbles_per_box : ℕ) : ℕ :=
  boxes * marbles_per_box - sets * marbles_per_set

theorem namjoon_marbles : marble_problem 3 7 6 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_namjoon_marbles_l288_28804
