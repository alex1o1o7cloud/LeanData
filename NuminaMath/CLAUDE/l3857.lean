import Mathlib

namespace NUMINAMATH_CALUDE_set_c_is_proportional_l3857_385705

/-- A set of four real numbers is proportional if the product of its extremes equals the product of its means -/
def isProportional (a b c d : ℝ) : Prop := a * d = b * c

/-- The set (2, 3, 4, 6) is proportional -/
theorem set_c_is_proportional : isProportional 2 3 4 6 := by
  sorry

end NUMINAMATH_CALUDE_set_c_is_proportional_l3857_385705


namespace NUMINAMATH_CALUDE_observation_mean_invariance_l3857_385745

theorem observation_mean_invariance (n : ℕ) (h : n > 0) :
  let original_mean : ℚ := 200
  let decrement : ℚ := 6
  let new_mean : ℚ := 194
  n * original_mean - n * decrement = n * new_mean :=
by
  sorry

end NUMINAMATH_CALUDE_observation_mean_invariance_l3857_385745


namespace NUMINAMATH_CALUDE_shobha_current_age_l3857_385719

/-- Given the ratio of Shekhar's age to Shobha's age and Shekhar's future age, 
    prove Shobha's current age -/
theorem shobha_current_age 
  (shekhar_age shobha_age : ℕ) 
  (age_ratio : shekhar_age / shobha_age = 4 / 3) 
  (shekhar_future_age : shekhar_age + 6 = 26) : 
  shobha_age = 15 := by
sorry

end NUMINAMATH_CALUDE_shobha_current_age_l3857_385719


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l3857_385733

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, (x : ℝ)^6 / (x : ℝ)^3 < 18 → x ≤ 2 :=
by
  sorry

#check greatest_integer_satisfying_inequality

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l3857_385733


namespace NUMINAMATH_CALUDE_parking_lot_vehicles_l3857_385765

/-- Given a parking lot with tricycles and bicycles, prove the number of each type --/
theorem parking_lot_vehicles (total_vehicles : ℕ) (total_wheels : ℕ) 
  (h1 : total_vehicles = 15)
  (h2 : total_wheels = 40) :
  ∃ (tricycles bicycles : ℕ),
    tricycles + bicycles = total_vehicles ∧
    3 * tricycles + 2 * bicycles = total_wheels ∧
    tricycles = 10 ∧
    bicycles = 5 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_vehicles_l3857_385765


namespace NUMINAMATH_CALUDE_kopeck_enough_for_kvass_l3857_385749

/-- Represents the price of bread before any increase -/
def x : ℝ := sorry

/-- Represents the price of kvass before any increase -/
def y : ℝ := sorry

/-- The value of one kopeck -/
def kopeck : ℝ := 1

/-- Initial condition: total spending equals one kopeck -/
axiom initial_condition : x + y = kopeck

/-- Condition after first price increase -/
axiom first_increase : 0.6 * x + 1.2 * y = kopeck

/-- Theorem stating that one kopeck is enough for kvass after two 20% price increases -/
theorem kopeck_enough_for_kvass : kopeck > 1.44 * y := by sorry

end NUMINAMATH_CALUDE_kopeck_enough_for_kvass_l3857_385749


namespace NUMINAMATH_CALUDE_ed_hotel_stay_l3857_385725

def hotel_stay_problem (initial_amount : ℝ) (night_rate : ℝ) (morning_rate : ℝ) 
  (night_hours : ℝ) (morning_hours : ℝ) : Prop :=
  let night_cost := night_rate * night_hours
  let morning_cost := morning_rate * morning_hours
  let total_cost := night_cost + morning_cost
  let remaining_amount := initial_amount - total_cost
  remaining_amount = 63

theorem ed_hotel_stay :
  hotel_stay_problem 80 1.5 2 6 4 := by sorry

end NUMINAMATH_CALUDE_ed_hotel_stay_l3857_385725


namespace NUMINAMATH_CALUDE_parentheses_removal_l3857_385709

theorem parentheses_removal (a b c : ℝ) : 3*a - (2*b - c) = 3*a - 2*b + c := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l3857_385709


namespace NUMINAMATH_CALUDE_school_boys_count_l3857_385747

theorem school_boys_count :
  ∀ (total boys girls : ℕ),
  total = 900 →
  boys + girls = total →
  girls * total = boys * boys →
  boys = 810 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l3857_385747


namespace NUMINAMATH_CALUDE_simplification_exponent_sum_l3857_385708

-- Define the expression
def original_expression (a b c : ℝ) : ℝ := (40 * a^5 * b^8 * c^14) ^ (1/3)

-- Define the simplified expression
def simplified_expression (a b c : ℝ) : ℝ := 2 * a * b^2 * c^4 * ((5 * a * b^2 * c^2) ^ (1/3))

-- State the theorem
theorem simplification_exponent_sum :
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 →
  original_expression a b c = simplified_expression a b c ∧
  (1 + 2 + 4 = 7) := by sorry

end NUMINAMATH_CALUDE_simplification_exponent_sum_l3857_385708


namespace NUMINAMATH_CALUDE_farmer_brown_chickens_l3857_385748

/-- Given the number of sheep, total legs, and legs per animal, calculate the number of chickens -/
def calculate_chickens (num_sheep : ℕ) (total_legs : ℕ) (chicken_legs : ℕ) (sheep_legs : ℕ) : ℕ :=
  (total_legs - num_sheep * sheep_legs) / chicken_legs

/-- Theorem stating that under the given conditions, the number of chickens is 7 -/
theorem farmer_brown_chickens :
  let num_sheep : ℕ := 5
  let total_legs : ℕ := 34
  let chicken_legs : ℕ := 2
  let sheep_legs : ℕ := 4
  calculate_chickens num_sheep total_legs chicken_legs sheep_legs = 7 := by
  sorry

end NUMINAMATH_CALUDE_farmer_brown_chickens_l3857_385748


namespace NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l3857_385706

/-- The function g(n) returns the number of distinct ordered pairs of positive integers (a, b) 
    such that a^2 + b^2 = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 65 is the smallest positive integer n for which g(n) = 4 -/
theorem smallest_n_with_four_pairs : (∀ m : ℕ, 0 < m → m < 65 → g m ≠ 4) ∧ g 65 = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l3857_385706


namespace NUMINAMATH_CALUDE_find_number_l3857_385723

theorem find_number : ∃ n : ℕ, 72519 * n = 724827405 ∧ n = 10005 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3857_385723


namespace NUMINAMATH_CALUDE_integer_divisibility_problem_l3857_385767

theorem integer_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end NUMINAMATH_CALUDE_integer_divisibility_problem_l3857_385767


namespace NUMINAMATH_CALUDE_intersection_theorem_l3857_385763

/-- The number of intersection points between two curves -/
def intersection_count (a : ℝ) : ℕ := sorry

/-- First curve equation: x^2 + y^2 = 4a^2 -/
def curve1 (a x y : ℝ) : Prop := x^2 + y^2 = 4 * a^2

/-- Second curve equation: y = x^2 - 4a + 1 -/
def curve2 (a x y : ℝ) : Prop := y = x^2 - 4 * a + 1

theorem intersection_theorem (a : ℝ) :
  intersection_count a = 3 ↔ a > 1/8 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3857_385763


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l3857_385774

theorem gold_coin_distribution (x y : ℕ) (k : ℕ) 
  (h1 : x + y = 16) 
  (h2 : x > y) 
  (h3 : x^2 - y^2 = k * (x - y)) : 
  k = 16 :=
sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l3857_385774


namespace NUMINAMATH_CALUDE_famous_artists_not_set_l3857_385751

/-- A structure representing a collection of objects -/
structure Collection where
  elements : Set α
  is_definite : Bool
  is_distinct : Bool
  is_unordered : Bool

/-- Definition of a set -/
def is_set (c : Collection) : Prop :=
  c.is_definite ∧ c.is_distinct ∧ c.is_unordered

/-- Famous artists collection -/
def famous_artists : Collection := sorry

/-- Theorem stating that famous artists cannot form a set -/
theorem famous_artists_not_set : ¬(is_set famous_artists) := by
  sorry

end NUMINAMATH_CALUDE_famous_artists_not_set_l3857_385751


namespace NUMINAMATH_CALUDE_yellow_roses_count_l3857_385790

/-- The number of yellow roses on the third rose bush -/
def yellow_roses : ℕ := 20

theorem yellow_roses_count :
  let red_roses : ℕ := 12
  let pink_roses : ℕ := 18
  let orange_roses : ℕ := 8
  let red_picked : ℕ := red_roses / 2
  let pink_picked : ℕ := pink_roses / 2
  let orange_picked : ℕ := orange_roses / 4
  let yellow_picked : ℕ := yellow_roses / 4
  let total_picked : ℕ := 22
  red_picked + pink_picked + orange_picked + yellow_picked = total_picked →
  yellow_roses = 20 := by
sorry

end NUMINAMATH_CALUDE_yellow_roses_count_l3857_385790


namespace NUMINAMATH_CALUDE_employee_count_l3857_385744

theorem employee_count : 
  ∀ (E : ℕ) (M : ℝ),
    M = 0.99 * (E : ℝ) →
    (M - 99.99999999999991) / (E : ℝ) = 0.98 →
    E = 10000 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l3857_385744


namespace NUMINAMATH_CALUDE_equation_C_most_suitable_l3857_385761

-- Define the equations
def equation_A : ℝ → Prop := λ x ↦ 2 * x^2 = 8
def equation_B : ℝ → Prop := λ x ↦ x * (x + 2) = x + 2
def equation_C : ℝ → Prop := λ x ↦ x^2 - 2*x = 3
def equation_D : ℝ → Prop := λ x ↦ 2 * x^2 + x - 1 = 0

-- Define a predicate for suitability for completing the square method
def suitable_for_completing_square (eq : ℝ → Prop) : Prop := sorry

-- Theorem stating that equation C is most suitable for completing the square
theorem equation_C_most_suitable :
  suitable_for_completing_square equation_C ∧
  (¬suitable_for_completing_square equation_A ∨
   ¬suitable_for_completing_square equation_B ∨
   ¬suitable_for_completing_square equation_D) :=
sorry

end NUMINAMATH_CALUDE_equation_C_most_suitable_l3857_385761


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3857_385713

theorem divisibility_equivalence (n : ℕ+) :
  (n.val^5 + 5^n.val) % 11 = 0 ↔ (n.val^5 * 5^n.val + 1) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3857_385713


namespace NUMINAMATH_CALUDE_sum_plus_ten_is_three_times_square_l3857_385742

theorem sum_plus_ten_is_three_times_square (n : ℤ) (h : n ≠ 0) : 
  ∃ (m : ℤ), (n - 1)^4 + n^4 + (n + 1)^4 + 10 = 3 * m^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_plus_ten_is_three_times_square_l3857_385742


namespace NUMINAMATH_CALUDE_inequality_implication_l3857_385727

theorem inequality_implication (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_implication_l3857_385727


namespace NUMINAMATH_CALUDE_centroid_altitude_distance_l3857_385736

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (7, 15, 20)

-- Define the centroid G
def centroid (t : Triangle) : ℝ × ℝ := sorry

-- Define the foot of the altitude P
def altitude_foot (t : Triangle) (G : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem centroid_altitude_distance (t : Triangle) :
  let G := centroid t
  let P := altitude_foot t G
  distance G P = 1.4 := by sorry

end NUMINAMATH_CALUDE_centroid_altitude_distance_l3857_385736


namespace NUMINAMATH_CALUDE_inequality_proof_l3857_385737

theorem inequality_proof (a b : ℝ) (n : ℕ) (h : 2 * a + 3 * b = 12) :
  (a / 3) ^ n + (b / 2) ^ n ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3857_385737


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3857_385701

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a n > 0 ∧ a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : is_positive_geometric_sequence a)
  (h_prod : a 2 * a 8 = 4) :
  a 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3857_385701


namespace NUMINAMATH_CALUDE_largest_prime_sum_2500_l3857_385739

def isPrime (n : ℕ) : Prop := sorry

def sumOfPrimesUpTo (n : ℕ) : ℕ := sorry

theorem largest_prime_sum_2500 :
  ∀ p : ℕ, isPrime p →
    (p ≤ 151 → sumOfPrimesUpTo p ≤ 2500) ∧
    (p > 151 → sumOfPrimesUpTo p > 2500) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_sum_2500_l3857_385739


namespace NUMINAMATH_CALUDE_little_red_height_calculation_l3857_385710

/-- Little Ming's height in meters -/
def little_ming_height : ℝ := 1.3

/-- The difference in height between Little Ming and Little Red in meters -/
def height_difference : ℝ := 0.2

/-- Little Red's height in meters -/
def little_red_height : ℝ := little_ming_height - height_difference

theorem little_red_height_calculation :
  little_red_height = 1.1 := by sorry

end NUMINAMATH_CALUDE_little_red_height_calculation_l3857_385710


namespace NUMINAMATH_CALUDE_ln_inequality_l3857_385784

-- Define the natural logarithm function
noncomputable def f (x : ℝ) := Real.log x

-- State the theorem
theorem ln_inequality (x : ℝ) (h : x > 0) : f x ≤ x - 1 := by
  -- Define the derivative of f
  have f_deriv : ∀ x > 0, deriv f x = 1 / x := by sorry
  
  -- f(1) = 0
  have f_at_one : f 1 = 0 := by sorry
  
  -- The tangent line at x = 1 is y = x - 1
  have tangent_line : ∀ x, x - 1 = (x - 1) * (deriv f 1) + f 1 := by sorry
  
  -- The tangent line is above the graph of f for x > 0
  have tangent_above : ∀ x > 0, f x ≤ x - 1 := by sorry
  
  -- Apply the tangent_above property to prove the inequality
  exact tangent_above x h

end NUMINAMATH_CALUDE_ln_inequality_l3857_385784


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l3857_385779

theorem students_liking_both_desserts
  (total_students : ℕ)
  (ice_cream_fans : ℕ)
  (cookie_fans : ℕ)
  (neither_fans : ℕ)
  (h1 : total_students = 50)
  (h2 : ice_cream_fans = 28)
  (h3 : cookie_fans = 20)
  (h4 : neither_fans = 14) :
  total_students - neither_fans - (ice_cream_fans + cookie_fans - total_students + neither_fans) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l3857_385779


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3857_385746

theorem complex_fraction_simplification :
  (5 - 3 * Complex.I) / (2 - 3 * Complex.I) = -19/5 - 9/5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3857_385746


namespace NUMINAMATH_CALUDE_rajan_income_l3857_385759

/-- Represents the financial situation of two individuals --/
structure FinancialSituation where
  income_ratio : Rat
  expenditure_ratio : Rat
  savings : ℕ

/-- Calculates the income of the first person given a financial situation --/
def calculate_income (situation : FinancialSituation) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, Rajan's income is $7000 --/
theorem rajan_income (situation : FinancialSituation) 
  (h1 : situation.income_ratio = 7 / 6)
  (h2 : situation.expenditure_ratio = 6 / 5)
  (h3 : situation.savings = 1000) :
  calculate_income situation = 7000 :=
sorry

end NUMINAMATH_CALUDE_rajan_income_l3857_385759


namespace NUMINAMATH_CALUDE_simplify_expression_l3857_385786

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 256) - Real.sqrt (13/2))^2 = (45 - 8 * Real.sqrt 26) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3857_385786


namespace NUMINAMATH_CALUDE_total_watermelons_is_48_l3857_385738

/-- The number of watermelons Jason grew -/
def jason_watermelons : ℕ := 37

/-- The number of watermelons Sandy grew -/
def sandy_watermelons : ℕ := 11

/-- The total number of watermelons grown by Jason and Sandy -/
def total_watermelons : ℕ := jason_watermelons + sandy_watermelons

/-- Theorem stating that the total number of watermelons is 48 -/
theorem total_watermelons_is_48 : total_watermelons = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_watermelons_is_48_l3857_385738


namespace NUMINAMATH_CALUDE_hyperbola_chord_length_l3857_385753

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a chord of the hyperbola -/
structure Chord where
  A : Point
  B : Point

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a point lies on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- The foci of the hyperbola -/
def foci (h : Hyperbola) : (Point × Point) := sorry

theorem hyperbola_chord_length 
  (h : Hyperbola) 
  (c : Chord) 
  (F1 F2 : Point) :
  (foci h = (F1, F2)) →
  (on_hyperbola h c.A) →
  (on_hyperbola h c.B) →
  (distance F1 c.A = 0 ∨ distance F1 c.B = 0) →
  (distance c.A F2 + distance c.B F2 = 2 * distance c.A c.B) →
  distance c.A c.B = 4 * h.a :=
sorry

end NUMINAMATH_CALUDE_hyperbola_chord_length_l3857_385753


namespace NUMINAMATH_CALUDE_margarets_mean_score_l3857_385731

def scores : List ℕ := [85, 87, 92, 93, 94, 98]

theorem margarets_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℕ), 
        cyprian_scores.length = 3 ∧ 
        margaret_scores.length = 3 ∧ 
        cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℕ), 
        cyprian_scores.length = 3 ∧ 
        cyprian_scores.sum / cyprian_scores.length = 90) :
  ∃ (margaret_scores : List ℕ), 
    margaret_scores.length = 3 ∧ 
    margaret_scores.sum / margaret_scores.length = 93 :=
sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l3857_385731


namespace NUMINAMATH_CALUDE_sum_of_roots_l3857_385700

/-- The function f(x) = x³ + 3x² + 6x + 14 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

/-- Theorem: If f(a) + f(b) = 20, then a + b = -2 -/
theorem sum_of_roots (a b : ℝ) (h : f a + f b = 20) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3857_385700


namespace NUMINAMATH_CALUDE_haley_laundry_loads_l3857_385732

/-- The number of loads required to wash a given number of clothing items with a fixed-capacity washing machine. -/
def loads_required (machine_capacity : ℕ) (total_items : ℕ) : ℕ :=
  (total_items + machine_capacity - 1) / machine_capacity

theorem haley_laundry_loads :
  let machine_capacity : ℕ := 7
  let shirts : ℕ := 2
  let sweaters : ℕ := 33
  let total_items : ℕ := shirts + sweaters
  loads_required machine_capacity total_items = 5 := by
sorry

end NUMINAMATH_CALUDE_haley_laundry_loads_l3857_385732


namespace NUMINAMATH_CALUDE_student_score_l3857_385768

theorem student_score (max_score : ℕ) (pass_threshold : ℚ) (fail_margin : ℕ) (student_score : ℕ) : 
  max_score = 500 →
  pass_threshold = 33 / 100 →
  fail_margin = 40 →
  student_score = ⌊max_score * pass_threshold⌋ - fail_margin →
  student_score = 125 := by
sorry

end NUMINAMATH_CALUDE_student_score_l3857_385768


namespace NUMINAMATH_CALUDE_number_equation_proof_l3857_385799

theorem number_equation_proof : ∃ x : ℝ, x - (1004 / 20.08) = 4970 ∧ x = 5020 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l3857_385799


namespace NUMINAMATH_CALUDE_cone_base_area_l3857_385770

/-- Given a cone whose unfolded lateral surface is a semicircle with area 2π,
    prove that the area of its base is π. -/
theorem cone_base_area (r : ℝ) (h : r > 0) : 
  (2 * π = π * r^2) → (π * r^2 / 2 = π) :=
by sorry

end NUMINAMATH_CALUDE_cone_base_area_l3857_385770


namespace NUMINAMATH_CALUDE_no_integer_roots_for_odd_coeff_quadratic_l3857_385797

/-- A quadratic function with odd coefficients has no integer roots -/
theorem no_integer_roots_for_odd_coeff_quadratic (a b c : ℤ) (ha : a ≠ 0) 
  (hodd : Odd a ∧ Odd b ∧ Odd c) :
  ¬∃ x : ℤ, a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_odd_coeff_quadratic_l3857_385797


namespace NUMINAMATH_CALUDE_salary_spending_l3857_385757

theorem salary_spending (S : ℝ) (h1 : S > 0) : 
  let first_week := S / 4
  let unspent := S * 0.15
  let total_spent := S - unspent
  let last_three_weeks := total_spent - first_week
  last_three_weeks / (3 * S) = 0.2 := by sorry

end NUMINAMATH_CALUDE_salary_spending_l3857_385757


namespace NUMINAMATH_CALUDE_gcf_36_54_l3857_385789

theorem gcf_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_36_54_l3857_385789


namespace NUMINAMATH_CALUDE_million_factorizations_l3857_385755

def million : ℕ := 1000000

/-- The number of ways to represent 1,000,000 as a product of three factors when order matters -/
def distinct_factorizations : ℕ := 784

/-- The number of ways to represent 1,000,000 as a product of three factors when order doesn't matter -/
def identical_factorizations : ℕ := 139

/-- Function to count the number of ways to represent a number as a product of three factors -/
def count_factorizations (n : ℕ) (order_matters : Bool) : ℕ := sorry

theorem million_factorizations :
  (count_factorizations million true = distinct_factorizations) ∧
  (count_factorizations million false = identical_factorizations) := by sorry

end NUMINAMATH_CALUDE_million_factorizations_l3857_385755


namespace NUMINAMATH_CALUDE_ball_selection_ways_l3857_385752

/-- Represents the number of ways to select balls from a bag -/
def select_balls (total white red black : ℕ) (select : ℕ) 
  (white_min white_max red_min black_max : ℕ) : ℕ := sorry

/-- Theorem stating the number of ways to select balls under given conditions -/
theorem ball_selection_ways : 
  select_balls 20 9 5 6 10 2 8 2 3 = 16 := by sorry

end NUMINAMATH_CALUDE_ball_selection_ways_l3857_385752


namespace NUMINAMATH_CALUDE_max_value_on_curve_l3857_385773

/-- Given a point (a,b) on the curve y = e^2 / x where a > 1 and b > 1,
    the maximum value of a^(ln b) is e. -/
theorem max_value_on_curve (a b : ℝ) : 
  a > 1 → b > 1 → b = Real.exp 2 / a → (Real.exp 1 : ℝ) ≥ a^(Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l3857_385773


namespace NUMINAMATH_CALUDE_teagan_savings_proof_l3857_385734

def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100
def penny_value : ℚ := 1 / 100

def rex_nickels : ℕ := 100
def toni_dimes : ℕ := 330
def total_savings : ℚ := 40

def teagan_pennies : ℕ := 200

theorem teagan_savings_proof :
  (rex_nickels : ℚ) * nickel_value + (toni_dimes : ℚ) * dime_value + (teagan_pennies : ℚ) * penny_value = total_savings :=
by sorry

end NUMINAMATH_CALUDE_teagan_savings_proof_l3857_385734


namespace NUMINAMATH_CALUDE_gold_coin_puzzle_l3857_385760

theorem gold_coin_puzzle (n : ℕ) (c : ℕ) : 
  (∃ k : ℕ, n = 11 * (c - 3) ∧ k = c - 3) ∧ 
  n = 7 * c + 5 →
  n = 75 :=
by sorry

end NUMINAMATH_CALUDE_gold_coin_puzzle_l3857_385760


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3857_385726

/-- Given an isosceles triangle with base angle α and difference b between
    the radii of its circumscribed and inscribed circles, 
    the length of its base side is (2b * sin(2α)) / (1 - tan²(α/2)) -/
theorem isosceles_triangle_base_length 
  (α : ℝ) 
  (b : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < b) : 
  ∃ (x : ℝ), x = (2 * b * Real.sin (2 * α)) / (1 - Real.tan (α / 2) ^ 2) ∧ 
  x > 0 ∧ 
  ∃ (R r : ℝ), R > 0 ∧ r > 0 ∧ R - r = b ∧
  R = x / (2 * Real.sin (2 * α)) ∧
  r = x / 2 * Real.tan (α / 2) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3857_385726


namespace NUMINAMATH_CALUDE_function_minimum_implies_inequality_l3857_385730

/-- Given a function f(x) = ax^2 + bx - ln(x) where a, b ∈ ℝ,
    if a > 0 and for any x > 0, f(x) ≥ f(1), then ln(a) < -2b -/
theorem function_minimum_implies_inequality (a b : ℝ) :
  a > 0 →
  (∀ x > 0, a * x^2 + b * x - Real.log x ≥ a + b) →
  Real.log a < -2 * b :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_implies_inequality_l3857_385730


namespace NUMINAMATH_CALUDE_new_york_to_new_england_ratio_l3857_385754

/-- The population of New England -/
def new_england_population : ℕ := 2100000

/-- The combined population of New York and New England -/
def combined_population : ℕ := 3500000

/-- The population of New York -/
def new_york_population : ℕ := combined_population - new_england_population

/-- The ratio of New York's population to New England's population -/
theorem new_york_to_new_england_ratio :
  (new_york_population : ℚ) / (new_england_population : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_new_york_to_new_england_ratio_l3857_385754


namespace NUMINAMATH_CALUDE_paper_tape_overlap_l3857_385792

/-- Given 12 sheets of paper tape, each 18 cm long, glued to form a round loop
    with a perimeter of 210 cm and overlapped by the same length,
    the length of each overlapped part is 5 mm. -/
theorem paper_tape_overlap (num_sheets : ℕ) (sheet_length : ℝ) (perimeter : ℝ) :
  num_sheets = 12 →
  sheet_length = 18 →
  perimeter = 210 →
  (num_sheets * sheet_length - perimeter) / num_sheets * 10 = 5 :=
by sorry

end NUMINAMATH_CALUDE_paper_tape_overlap_l3857_385792


namespace NUMINAMATH_CALUDE_coffee_price_increase_l3857_385793

/-- Given the conditions of the tea and coffee pricing problem, prove that the price of coffee increased by 100% from June to July. -/
theorem coffee_price_increase (june_price : ℝ) (july_mixture_price : ℝ) (july_tea_price : ℝ) :
  -- In June, the price of green tea and coffee were the same
  -- In July, the price of green tea dropped by 70%
  -- In July, a mixture of equal quantities of green tea and coffee costs $3.45 for 3 lbs
  -- In July, a pound of green tea costs $0.3
  june_price > 0 ∧
  july_mixture_price = 3.45 ∧
  july_tea_price = 0.3 ∧
  july_tea_price = june_price * 0.3 →
  -- The price of coffee increased by 100%
  (((july_mixture_price - 3 * july_tea_price / 2) * 2 / 3 - june_price) / june_price) * 100 = 100 :=
by sorry

end NUMINAMATH_CALUDE_coffee_price_increase_l3857_385793


namespace NUMINAMATH_CALUDE_rectangle_perimeter_and_area_l3857_385795

/-- Perimeter and area of a rectangle with specific dimensions -/
theorem rectangle_perimeter_and_area :
  let l : ℝ := Real.sqrt 6 + 2 * Real.sqrt 5
  let w : ℝ := 2 * Real.sqrt 6 - Real.sqrt 5
  let perimeter : ℝ := 2 * (l + w)
  let area : ℝ := l * w
  (perimeter = 6 * Real.sqrt 6 + 2 * Real.sqrt 5) ∧
  (area = 2 + 3 * Real.sqrt 30) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_and_area_l3857_385795


namespace NUMINAMATH_CALUDE_arrangements_three_male_two_female_l3857_385721

/-- The number of ways to arrange 3 male and 2 female students in a row,
    such that the female students do not stand at either end -/
def arrangements (n_male : ℕ) (n_female : ℕ) : ℕ :=
  if n_male = 3 ∧ n_female = 2 then
    (n_male + n_female - 2).choose n_female * n_male.factorial
  else
    0

theorem arrangements_three_male_two_female :
  arrangements 3 2 = 36 :=
sorry

end NUMINAMATH_CALUDE_arrangements_three_male_two_female_l3857_385721


namespace NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l3857_385787

theorem root_in_interval_implies_m_range (m : ℝ) :
  (∃ x, x^2 + 2*x - m = 0 ∧ 2 < x ∧ x < 3) →
  (8 < m ∧ m < 15) := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l3857_385787


namespace NUMINAMATH_CALUDE_rock_collecting_contest_l3857_385758

/-- The rock collecting contest between Sydney and Conner --/
theorem rock_collecting_contest 
  (sydney_initial : ℕ) 
  (conner_initial : ℕ) 
  (sydney_day1 : ℕ) 
  (conner_day1_multiplier : ℕ) 
  (conner_day2 : ℕ) 
  (sydney_day3_multiplier : ℕ) 
  (h1 : sydney_initial = 837) 
  (h2 : conner_initial = 723) 
  (h3 : sydney_day1 = 4) 
  (h4 : conner_day1_multiplier = 8) 
  (h5 : conner_day2 = 123) 
  (h6 : sydney_day3_multiplier = 2) : 
  ∃ (conner_day3 : ℕ), conner_day3 ≥ 27 ∧ 
    conner_initial + conner_day1_multiplier * sydney_day1 + conner_day2 + conner_day3 ≥ 
    sydney_initial + sydney_day1 + sydney_day3_multiplier * (conner_day1_multiplier * sydney_day1) :=
by
  sorry

end NUMINAMATH_CALUDE_rock_collecting_contest_l3857_385758


namespace NUMINAMATH_CALUDE_prob_not_losing_l3857_385743

/-- Given a chess game between players A and B, this theorem proves
    the probability of A not losing, given the probabilities of a draw
    and A winning. -/
theorem prob_not_losing (p_draw p_win : ℝ) : 
  p_draw = 1/2 → p_win = 1/3 → p_draw + p_win = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_losing_l3857_385743


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3857_385764

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) ≤ 1 / 2 ∧
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) = 1 / 2 ↔ 
   a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3857_385764


namespace NUMINAMATH_CALUDE_train_distance_theorem_l3857_385707

/-- The distance between two trains after 8 hours of travel, given their initial positions and speeds -/
def distance_between_trains (initial_distance : ℝ) (speed1 speed2 : ℝ) (time : ℝ) : Set ℝ :=
  let distance1 := speed1 * time
  let distance2 := speed2 * time
  let diff := distance2 - distance1
  {initial_distance + diff, initial_distance - diff}

/-- Theorem stating the distance between two trains after 8 hours -/
theorem train_distance_theorem :
  distance_between_trains 892 40 48 8 = {956, 828} :=
by sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l3857_385707


namespace NUMINAMATH_CALUDE_man_double_son_age_l3857_385720

/-- Represents the age difference between a man and his son -/
def age_difference : ℕ := 35

/-- Represents the son's present age -/
def son_present_age : ℕ := 33

/-- Calculates the number of years until the man's age is twice his son's age -/
def years_until_double_age : ℕ := 2

/-- Theorem stating that the number of years until the man's age is twice his son's age is 2 -/
theorem man_double_son_age :
  (son_present_age + years_until_double_age) * 2 = 
  (son_present_age + age_difference + years_until_double_age) :=
by sorry

end NUMINAMATH_CALUDE_man_double_son_age_l3857_385720


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l3857_385780

theorem opposite_of_negative_one_third : 
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l3857_385780


namespace NUMINAMATH_CALUDE_parallelogram_height_l3857_385782

/-- The height of a parallelogram with given area and base -/
theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 864) 
  (h_base : base = 36) 
  (h_formula : area = base * height) : 
  height = 24 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3857_385782


namespace NUMINAMATH_CALUDE_real_numbers_greater_than_eight_is_set_real_numbers_greater_than_eight_definite_membership_real_numbers_greater_than_eight_fixed_standards_l3857_385769

-- Define the property of being greater than 8
def GreaterThanEight (x : ℝ) : Prop := x > 8

-- Define the set of real numbers greater than 8
def RealNumbersGreaterThanEight : Set ℝ := {x : ℝ | GreaterThanEight x}

-- Theorem stating that RealNumbersGreaterThanEight is a well-defined set
theorem real_numbers_greater_than_eight_is_set :
  ∀ (x : ℝ), x ∈ RealNumbersGreaterThanEight ↔ GreaterThanEight x :=
by
  sorry

-- Theorem stating that RealNumbersGreaterThanEight has definite membership criteria
theorem real_numbers_greater_than_eight_definite_membership :
  ∀ (x : ℝ), Decidable (x ∈ RealNumbersGreaterThanEight) :=
by
  sorry

-- Theorem stating that RealNumbersGreaterThanEight has fixed standards for inclusion
theorem real_numbers_greater_than_eight_fixed_standards :
  ∀ (x y : ℝ), x > 8 ∧ y > 8 → (x ∈ RealNumbersGreaterThanEight ∧ y ∈ RealNumbersGreaterThanEight) :=
by
  sorry

end NUMINAMATH_CALUDE_real_numbers_greater_than_eight_is_set_real_numbers_greater_than_eight_definite_membership_real_numbers_greater_than_eight_fixed_standards_l3857_385769


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l3857_385778

def M : Set Nat := {1, 2, 3, 4, 5}
def N : Set Nat := {2, 5}

theorem complement_of_N_in_M :
  M \ N = {1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l3857_385778


namespace NUMINAMATH_CALUDE_willow_catkin_diameter_scientific_notation_l3857_385718

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem willow_catkin_diameter_scientific_notation :
  toScientificNotation 0.0000105 = ScientificNotation.mk 1.05 (-5) (by sorry) :=
sorry

end NUMINAMATH_CALUDE_willow_catkin_diameter_scientific_notation_l3857_385718


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3857_385796

theorem roots_of_polynomial (x : ℝ) :
  let p : ℝ → ℝ := λ x => (x^2 - 5*x + 6)*(x - 3)*(x + 2)
  {x : ℝ | p x = 0} = {2, 3, -2} := by
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3857_385796


namespace NUMINAMATH_CALUDE_work_completion_time_l3857_385791

/-- Given workers a and b, where b completes a work in 7 days, and both a and b
    together complete the work in 4.117647058823529 days, prove that a can
    complete the work alone in 10 days. -/
theorem work_completion_time
  (total_work : ℝ)
  (rate_b : ℝ)
  (rate_combined : ℝ)
  (h1 : rate_b = total_work / 7)
  (h2 : rate_combined = total_work / 4.117647058823529)
  (h3 : rate_combined = rate_b + total_work / 10) :
  ∃ (days_a : ℝ), days_a = 10 ∧ total_work / days_a = total_work / 10 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3857_385791


namespace NUMINAMATH_CALUDE_cara_seating_arrangement_l3857_385781

theorem cara_seating_arrangement (n : ℕ) (k : ℕ) : n = 7 ∧ k = 2 → Nat.choose n k = 21 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangement_l3857_385781


namespace NUMINAMATH_CALUDE_train_length_l3857_385798

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 50 → time_s = 9 → 
  (speed_kmh * 1000 / 3600) * time_s = 125 := by sorry

end NUMINAMATH_CALUDE_train_length_l3857_385798


namespace NUMINAMATH_CALUDE_function_no_zeros_implies_a_less_than_neg_one_l3857_385794

theorem function_no_zeros_implies_a_less_than_neg_one (a : ℝ) : 
  (∀ x : ℝ, 4^x - 2^(x+1) - a ≠ 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_function_no_zeros_implies_a_less_than_neg_one_l3857_385794


namespace NUMINAMATH_CALUDE_retail_price_calculation_l3857_385714

/-- The retail price of a machine given wholesale price, discount, and profit margin -/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  let selling_price := (1 - discount_rate) * retail_price
  let profit := profit_rate * wholesale_price
  wholesale_price = 81 ∧ discount_rate = 0.1 ∧ profit_rate = 0.2 →
  ∃ retail_price : ℝ, selling_price = wholesale_price + profit ∧ retail_price = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l3857_385714


namespace NUMINAMATH_CALUDE_puffy_muffy_weight_l3857_385777

/-- The weight of Scruffy in ounces -/
def scruffy_weight : ℕ := 12

/-- The weight difference between Scruffy and Muffy in ounces -/
def muffy_scruffy_diff : ℕ := 3

/-- The weight difference between Puffy and Muffy in ounces -/
def puffy_muffy_diff : ℕ := 5

/-- The combined weight of Puffy and Muffy in ounces -/
def combined_weight : ℕ := scruffy_weight - muffy_scruffy_diff + (scruffy_weight - muffy_scruffy_diff + puffy_muffy_diff)

theorem puffy_muffy_weight : combined_weight = 23 := by
  sorry

end NUMINAMATH_CALUDE_puffy_muffy_weight_l3857_385777


namespace NUMINAMATH_CALUDE_sandwich_cost_l3857_385728

/-- The cost of tomatoes for N sandwiches, each using T slices at 4 cents per slice --/
def tomatoCost (N T : ℕ) : ℚ := (N * T * 4 : ℕ) / 100

/-- The total cost of ingredients for N sandwiches, each using C slices of cheese and T slices of tomato --/
def totalCost (N C T : ℕ) : ℚ := (N * (3 * C + 4 * T) : ℕ) / 100

theorem sandwich_cost (N C T : ℕ) : 
  N > 1 → C > 0 → T > 0 → totalCost N C T = 305 / 100 → tomatoCost N T = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_l3857_385728


namespace NUMINAMATH_CALUDE_shaded_area_is_twelve_l3857_385772

-- Define the rectangle
structure Rectangle where
  base : ℝ
  height : ℝ

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def problemSetup (rect : Rectangle) (tri : IsoscelesTriangle) : Prop :=
  rect.height = tri.height ∧
  rect.base = 12 ∧
  rect.height = 8 ∧
  tri.base = 12

-- Define the intersection point
def intersectionPoint : Point :=
  { x := 18, y := 2 }

-- Theorem statement
theorem shaded_area_is_twelve (rect : Rectangle) (tri : IsoscelesTriangle) 
  (h : problemSetup rect tri) : 
  (1/2 : ℝ) * tri.base * intersectionPoint.y = 12 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_twelve_l3857_385772


namespace NUMINAMATH_CALUDE_heels_savings_per_month_l3857_385756

theorem heels_savings_per_month 
  (months_saved : ℕ) 
  (sister_contribution : ℕ) 
  (total_spent : ℕ) : 
  months_saved = 3 → 
  sister_contribution = 50 → 
  total_spent = 260 → 
  (total_spent - sister_contribution) / months_saved = 70 :=
by sorry

end NUMINAMATH_CALUDE_heels_savings_per_month_l3857_385756


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3857_385766

/-- Given sets M and N, prove their intersection -/
theorem intersection_of_M_and_N :
  let M := {x : ℝ | |x - 1| < 2}
  let N := {x : ℝ | x * (x - 3) < 0}
  M ∩ N = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3857_385766


namespace NUMINAMATH_CALUDE_blueberry_count_l3857_385703

theorem blueberry_count (total : ℕ) (raspberries : ℕ) (blackberries : ℕ) (blueberries : ℕ)
  (h1 : total = 42)
  (h2 : raspberries = total / 2)
  (h3 : blackberries = total / 3)
  (h4 : total = raspberries + blackberries + blueberries) :
  blueberries = 7 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_count_l3857_385703


namespace NUMINAMATH_CALUDE_vector_equation_vectors_parallel_l3857_385724

/-- Given vectors in R² -/
def a : Fin 2 → ℚ := ![3, 2]
def b : Fin 2 → ℚ := ![-1, 2]
def c : Fin 2 → ℚ := ![4, 1]

/-- Theorem for part (1) -/
theorem vector_equation :
  a = (5/9 : ℚ) • b + (8/9 : ℚ) • c := by sorry

/-- Helper function to check if two vectors are parallel -/
def are_parallel (v w : Fin 2 → ℚ) : Prop :=
  ∃ (k : ℚ), v = k • w ∨ w = k • v

/-- Theorem for part (2) -/
theorem vectors_parallel :
  are_parallel (a + (-16/13 : ℚ) • c) (2 • b - a) := by sorry

end NUMINAMATH_CALUDE_vector_equation_vectors_parallel_l3857_385724


namespace NUMINAMATH_CALUDE_lashawn_double_kymbrea_after_25_months_l3857_385783

/-- Represents the number of comic books in a collection after a given number of months. -/
def comic_books (initial : ℕ) (rate : ℕ) (months : ℕ) : ℕ :=
  initial + rate * months

theorem lashawn_double_kymbrea_after_25_months :
  let kymbrea_initial := 30
  let kymbrea_rate := 2
  let lashawn_initial := 10
  let lashawn_rate := 6
  let months := 25
  comic_books lashawn_initial lashawn_rate months = 
    2 * comic_books kymbrea_initial kymbrea_rate months := by
  sorry

end NUMINAMATH_CALUDE_lashawn_double_kymbrea_after_25_months_l3857_385783


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l3857_385775

theorem quadratic_inequality_empty_solution (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) → a ∈ Set.Icc (-4 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l3857_385775


namespace NUMINAMATH_CALUDE_clock_angle_at_7_proof_l3857_385741

/-- The smaller angle formed by the hands of a clock at 7 o'clock -/
def clock_angle_at_7 : ℝ := 150

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℝ := 360

/-- The number of hour points on a clock -/
def clock_hour_points : ℕ := 12

/-- The position of the hour hand at 7 o'clock -/
def hour_hand_position : ℕ := 7

/-- The position of the minute hand at 7 o'clock -/
def minute_hand_position : ℕ := 12

theorem clock_angle_at_7_proof :
  clock_angle_at_7 = (minute_hand_position - hour_hand_position) * (full_circle_degrees / clock_hour_points) :=
by sorry

end NUMINAMATH_CALUDE_clock_angle_at_7_proof_l3857_385741


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l3857_385788

theorem merry_go_round_revolutions 
  (r₁ : ℝ) (r₂ : ℝ) (rev₁ : ℝ) 
  (h₁ : r₁ = 36) 
  (h₂ : r₂ = 12) 
  (h₃ : rev₁ = 40) 
  (h₄ : r₁ > 0) 
  (h₅ : r₂ > 0) : 
  ∃ rev₂ : ℝ, rev₂ * r₂ = rev₁ * r₁ ∧ rev₂ = 120 :=
sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l3857_385788


namespace NUMINAMATH_CALUDE_product_of_complements_bound_l3857_385771

theorem product_of_complements_bound (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ ≥ 0) (h₃ : x₃ ≥ 0) 
  (h_sum : x₁ + x₂ + x₃ ≤ 1/2) : 
  (1 - x₁) * (1 - x₂) * (1 - x₃) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_complements_bound_l3857_385771


namespace NUMINAMATH_CALUDE_cubic_function_property_l3857_385712

/-- Given a function f(x) = x^3 + ax + 3 where f(-m) = 1, prove that f(m) = 5 -/
theorem cubic_function_property (a m : ℝ) : 
  (fun x : ℝ => x^3 + a*x + 3) (-m) = 1 → 
  (fun x : ℝ => x^3 + a*x + 3) m = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3857_385712


namespace NUMINAMATH_CALUDE_min_moves_is_22_l3857_385711

/-- A move consists of transferring one coin to an adjacent box. -/
def Move := ℕ

/-- The configuration of coins in the boxes. -/
def Configuration := Fin 7 → ℕ

/-- The initial configuration of coins in the boxes. -/
def initial_config : Configuration :=
  fun i => [5, 8, 11, 17, 20, 15, 10].get i

/-- A configuration is balanced if all boxes have the same number of coins. -/
def is_balanced (c : Configuration) : Prop :=
  ∀ i j : Fin 7, c i = c j

/-- The number of moves required to transform one configuration into another. -/
def moves_required (start finish : Configuration) : ℕ := sorry

/-- The minimum number of moves required to balance the configuration. -/
def min_moves_to_balance (c : Configuration) : ℕ := sorry

/-- The theorem stating that the minimum number of moves required to balance
    the initial configuration is 22. -/
theorem min_moves_is_22 :
  min_moves_to_balance initial_config = 22 := by sorry

end NUMINAMATH_CALUDE_min_moves_is_22_l3857_385711


namespace NUMINAMATH_CALUDE_feifei_leilei_age_sum_feifei_leilei_age_sum_proof_l3857_385762

theorem feifei_leilei_age_sum : ℕ → ℕ → Prop :=
  fun feifei_age leilei_age =>
    (feifei_age = leilei_age / 2 + 12) →
    (feifei_age + 1 = 2 * (leilei_age + 1) - 34) →
    (feifei_age + leilei_age = 57)

theorem feifei_leilei_age_sum_proof : ∃ (f l : ℕ), feifei_leilei_age_sum f l :=
  sorry

end NUMINAMATH_CALUDE_feifei_leilei_age_sum_feifei_leilei_age_sum_proof_l3857_385762


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3857_385735

/-- The number of ways to arrange books on a shelf -/
def arrange_books (math_books : ℕ) (english_books : ℕ) (science_books : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial english_books) * (Nat.factorial science_books)

/-- Theorem: The number of ways to arrange 4 math books, 6 English books, and 2 science books
    on a shelf, where all books of the same subject must stay together and the books within
    each subject are different, is equal to 207360. -/
theorem book_arrangement_theorem :
  arrange_books 4 6 2 = 207360 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3857_385735


namespace NUMINAMATH_CALUDE_fourth_root_closest_to_6700_l3857_385704

def n : ℕ := 2001200120012001

def options : List ℕ := [2001, 6700, 21000, 12000, 2100]

theorem fourth_root_closest_to_6700 :
  ∃ (x : ℝ), x^4 = n ∧ 
  ∀ y ∈ options, |x - 6700| ≤ |x - y| :=
sorry

end NUMINAMATH_CALUDE_fourth_root_closest_to_6700_l3857_385704


namespace NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l3857_385729

theorem fourth_root_256_times_cube_root_64_times_sqrt_16 : 
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 32 := by sorry

end NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l3857_385729


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3857_385702

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3857_385702


namespace NUMINAMATH_CALUDE_triangle_area_16_triangle_AED_area_l3857_385740

/-- The area of a triangle with base 8 and height 4 is 16 square units. -/
theorem triangle_area_16 (base height : ℝ) (h1 : base = 8) (h2 : height = 4) :
  (1 / 2) * base * height = 16 := by sorry

/-- Given a triangle AED where AE = 8, height = 4, and ED = DA = 5,
    the area of triangle AED is 16 square units. -/
theorem triangle_AED_area (AE ED DA height : ℝ)
  (h1 : AE = 8)
  (h2 : height = 4)
  (h3 : ED = 5)
  (h4 : DA = 5) :
  (1 / 2) * AE * height = 16 := by sorry

end NUMINAMATH_CALUDE_triangle_area_16_triangle_AED_area_l3857_385740


namespace NUMINAMATH_CALUDE_sum_of_common_terms_equal_1472_l3857_385722

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def common_terms (seq1 seq2 : List ℕ) : List ℕ :=
  seq1.filter (fun x => seq2.contains x)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem sum_of_common_terms_equal_1472 :
  let seq1 := arithmetic_sequence 2 4 48
  let seq2 := arithmetic_sequence 2 6 34
  let common := common_terms seq1 seq2
  sum_list common = 1472 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_terms_equal_1472_l3857_385722


namespace NUMINAMATH_CALUDE_unique_palindrome_square_l3857_385717

/-- A function that returns true if a number is a three-digit palindrome with an even middle digit -/
def is_valid_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  (n / 100 = n % 10) ∧  -- first and last digits are the same
  (n / 10 % 10) % 2 = 0  -- middle digit is even

/-- The main theorem stating that there is exactly one number satisfying the conditions -/
theorem unique_palindrome_square : ∃! n : ℕ, 
  is_valid_palindrome n ∧ ∃ m : ℕ, n = m^2 :=
sorry

end NUMINAMATH_CALUDE_unique_palindrome_square_l3857_385717


namespace NUMINAMATH_CALUDE_bal_puzzle_l3857_385785

/-- Represents the possible meanings of the word "bal" -/
inductive BalMeaning
  | Yes
  | No

/-- Represents the possible types of inhabitants -/
inductive InhabitantType
  | Human
  | Zombie

/-- Represents the response to a yes/no question -/
def Response := BalMeaning

/-- Models the behavior of an inhabitant based on their type -/
def inhabitantBehavior (t : InhabitantType) (actual : BalMeaning) (response : Response) : Prop :=
  match t with
  | InhabitantType.Human => response = actual
  | InhabitantType.Zombie => response ≠ actual

/-- The main theorem capturing the essence of the problem -/
theorem bal_puzzle (response : Response) :
  (∀ meaning : BalMeaning, ∃ t : InhabitantType, inhabitantBehavior t meaning response) ∧
  (∃! t : InhabitantType, ∀ meaning : BalMeaning, inhabitantBehavior t meaning response) :=
by sorry

end NUMINAMATH_CALUDE_bal_puzzle_l3857_385785


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3857_385750

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 3 →
    (56 * x - 14) / (x^2 - 4*x + 3) = N₁ / (x - 1) + N₂ / (x - 3)) →
  N₁ * N₂ = -1617 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3857_385750


namespace NUMINAMATH_CALUDE_set_operations_l3857_385715

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 5}
def B : Set ℝ := {x | -1 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Define the universal set U
def U : Set ℝ := Set.univ

theorem set_operations :
  (A ∪ B = {x | -2 < x ∧ x < 5}) ∧
  (A ∩ B = {x | 0 ≤ x ∧ x ≤ 3}) ∧
  (A ∪ Bᶜ = U) ∧
  (A ∩ Bᶜ = {x | -2 < x ∧ x < 0} ∪ {x | 3 < x ∧ x < 5}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l3857_385715


namespace NUMINAMATH_CALUDE_visitors_in_scientific_notation_l3857_385776

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem visitors_in_scientific_notation :
  toScientificNotation 203000 = ScientificNotation.mk 2.03 5 sorry := by
  sorry

end NUMINAMATH_CALUDE_visitors_in_scientific_notation_l3857_385776


namespace NUMINAMATH_CALUDE_x_range_l3857_385716

theorem x_range (x : ℝ) : 
  (|x - 1| + |x - 5| = 4) ↔ (1 ≤ x ∧ x ≤ 5) := by
sorry

end NUMINAMATH_CALUDE_x_range_l3857_385716
