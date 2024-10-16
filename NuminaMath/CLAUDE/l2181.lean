import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_is_one_third_of_square_l2181_218199

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in 2D space -/
structure Square where
  bottomLeft : Point
  topRight : Point

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ :=
  (s.topRight.x - s.bottomLeft.x) * (s.topRight.y - s.bottomLeft.y)

/-- Main theorem: The area of the triangle formed by the line and the bottom of the square
    is 1/3 of the total square area -/
theorem triangle_area_is_one_third_of_square (s : Square)
  (p1 p2 : Point)
  (h1 : s.bottomLeft = ⟨2, 1⟩)
  (h2 : s.topRight = ⟨5, 4⟩)
  (h3 : p1 = ⟨2, 3⟩)
  (h4 : p2 = ⟨5, 1⟩) :
  triangleArea p1 p2 s.bottomLeft / squareArea s = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_is_one_third_of_square_l2181_218199


namespace NUMINAMATH_CALUDE_negation_equivalence_l2181_218123

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Doctor : U → Prop)
variable (ExcellentCook : U → Prop)
variable (PoorCook : U → Prop)

-- Define the statements
def AllDoctorsExcellentCooks : Prop := ∀ x, Doctor x → ExcellentCook x
def AtLeastOneDoctorPoorCook : Prop := ∃ x, Doctor x ∧ PoorCook x

-- Theorem to prove
theorem negation_equivalence :
  AtLeastOneDoctorPoorCook U Doctor PoorCook ↔ 
  ¬(AllDoctorsExcellentCooks U Doctor ExcellentCook) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2181_218123


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2181_218146

theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (first_term : a 1 = 1) 
  (fifth_term : a 5 = 4) : 
  a 3 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2181_218146


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2181_218156

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k = 0 ∧ y^2 - 2*y + k = 0) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2181_218156


namespace NUMINAMATH_CALUDE_rotation_180_maps_points_l2181_218181

def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotation_180_maps_points :
  let C : ℝ × ℝ := (-1, 2)
  let D : ℝ × ℝ := (3, 2)
  let C' : ℝ × ℝ := (1, -2)
  let D' : ℝ × ℝ := (-3, -2)
  rotate180 C = C' ∧ rotate180 D = D' := by sorry

end NUMINAMATH_CALUDE_rotation_180_maps_points_l2181_218181


namespace NUMINAMATH_CALUDE_point_coordinates_proof_l2181_218102

/-- Given points A and B, and the relation between vectors AP and AB, 
    prove that P has specific coordinates. -/
theorem point_coordinates_proof (A B P : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (4, -3) → 
  P - A = 3 • (B - A) → 
  P = (8, -15) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_proof_l2181_218102


namespace NUMINAMATH_CALUDE_right_triangle_equality_l2181_218113

/-- In a right triangle ABC with point M on the hypotenuse, if BM + MA = BC + CA,
    MB = x, CB = 2h, and CA = d, then x = hd / (2h + d) -/
theorem right_triangle_equality (h d x : ℝ) :
  h > 0 → d > 0 →
  x > 0 →
  x + Real.sqrt ((x + 2*h)^2 + d^2) = 2*h + d →
  x = h * d / (2*h + d) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_equality_l2181_218113


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2181_218117

-- Define the sets A and B
def A : Set ℝ := {-1, 0, 2}
def B (a : ℝ) : Set ℝ := {2^a}

-- State the theorem
theorem subset_implies_a_equals_one (a : ℝ) :
  B a ⊆ A → a = 1 := by sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2181_218117


namespace NUMINAMATH_CALUDE_constant_c_value_l2181_218166

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_c_value_l2181_218166


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2181_218129

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 25 = (x + a)^2) → m = 10 ∨ m = -10 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2181_218129


namespace NUMINAMATH_CALUDE_smallest_block_size_l2181_218133

/-- 
Given a rectangular block with dimensions a × b × c formed by N congruent 1-cm cubes,
where (a-1)(b-1)(c-1) = 252, the smallest possible value of N is 224.
-/
theorem smallest_block_size (a b c N : ℕ) : 
  (a - 1) * (b - 1) * (c - 1) = 252 → 
  N = a * b * c → 
  (∀ a' b' c' N', (a' - 1) * (b' - 1) * (c' - 1) = 252 → N' = a' * b' * c' → N ≤ N') →
  N = 224 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_size_l2181_218133


namespace NUMINAMATH_CALUDE_vector_b_value_l2181_218132

/-- Given two vectors a and b in ℝ³, prove that b equals (-2, 4, -2) -/
theorem vector_b_value (a b : ℝ × ℝ × ℝ) : 
  a = (1, -2, 1) → a + b = (-1, 2, -1) → b = (-2, 4, -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_b_value_l2181_218132


namespace NUMINAMATH_CALUDE_no_distinct_cube_sum_equality_l2181_218167

theorem no_distinct_cube_sum_equality (a b c d : ℕ) :
  a^3 + b^3 = c^3 + d^3 → a + b = c + d → ¬(a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_cube_sum_equality_l2181_218167


namespace NUMINAMATH_CALUDE_reciprocal_of_two_thirds_l2181_218159

def reciprocal (a b : ℚ) : ℚ := b / a

theorem reciprocal_of_two_thirds :
  reciprocal (2 : ℚ) 3 = (3 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_two_thirds_l2181_218159


namespace NUMINAMATH_CALUDE_expression_factorization_l2181_218178

theorem expression_factorization (x : ℚ) :
  (x^2 - 3*x + 2) - (x^2 - x + 6) + (x - 1)*(x - 2) + x^2 + 2 = (2*x - 1)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2181_218178


namespace NUMINAMATH_CALUDE_same_remainder_difference_divisible_l2181_218152

theorem same_remainder_difference_divisible (a m b : ℤ) : 
  (∃ r : ℤ, a % b = r ∧ m % b = r) → b ∣ (a - m) := by sorry

end NUMINAMATH_CALUDE_same_remainder_difference_divisible_l2181_218152


namespace NUMINAMATH_CALUDE_power_of_two_equality_l2181_218116

theorem power_of_two_equality (K : ℕ) : 32^5 * 64^2 = 2^K → K = 37 := by
  have h1 : 32 = 2^5 := by sorry
  have h2 : 64 = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l2181_218116


namespace NUMINAMATH_CALUDE_investors_in_securities_and_equities_l2181_218186

theorem investors_in_securities_and_equities 
  (total_investors : ℕ) 
  (investors_in_equities : ℕ) 
  (investors_in_both : ℕ) 
  (h1 : total_investors = 100)
  (h2 : investors_in_equities = 80)
  (h3 : investors_in_both = 25)
  (h4 : investors_in_both ≤ investors_in_equities)
  (h5 : investors_in_both ≤ total_investors) :
  investors_in_both = 25 := by
sorry

end NUMINAMATH_CALUDE_investors_in_securities_and_equities_l2181_218186


namespace NUMINAMATH_CALUDE_sports_store_sales_is_20_l2181_218130

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 10

/-- The number of customers per car -/
def customers_per_car : ℕ := 5

/-- The number of purchases each customer makes -/
def purchases_per_customer : ℕ := 1

/-- The number of sales made by the music store -/
def music_store_sales : ℕ := 30

/-- The total number of customers in the parking lot -/
def total_customers : ℕ := num_cars * customers_per_car

/-- The number of sales made by the sports store -/
def sports_store_sales : ℕ := total_customers - music_store_sales

theorem sports_store_sales_is_20 : sports_store_sales = 20 := by
  sorry

end NUMINAMATH_CALUDE_sports_store_sales_is_20_l2181_218130


namespace NUMINAMATH_CALUDE_container_count_l2181_218163

theorem container_count (x y : ℕ) : 
  27 * x = 65 * y + 34 → 
  y ≤ 44 → 
  x + y = 66 :=
by sorry

end NUMINAMATH_CALUDE_container_count_l2181_218163


namespace NUMINAMATH_CALUDE_john_finishes_ahead_l2181_218138

/-- The final push scenario in a race between John and Steve -/
structure FinalPush where
  initial_distance : ℝ  -- Initial distance John is behind Steve (in meters)
  john_speed : ℝ        -- John's speed (in m/s)
  steve_speed : ℝ       -- Steve's speed (in m/s)
  duration : ℝ          -- Duration of the final push (in seconds)

/-- Calculate the distance John finishes ahead of Steve -/
def distance_ahead (fp : FinalPush) : ℝ :=
  fp.john_speed * fp.duration - (fp.steve_speed * fp.duration + fp.initial_distance)

/-- Theorem stating that John finishes 2 meters ahead of Steve -/
theorem john_finishes_ahead (fp : FinalPush) 
  (h1 : fp.initial_distance = 16)
  (h2 : fp.john_speed = 4.2)
  (h3 : fp.steve_speed = 3.7)
  (h4 : fp.duration = 36) :
  distance_ahead fp = 2 := by
  sorry

#eval distance_ahead { initial_distance := 16, john_speed := 4.2, steve_speed := 3.7, duration := 36 }

end NUMINAMATH_CALUDE_john_finishes_ahead_l2181_218138


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_min_a_value_l2181_218150

noncomputable section

def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

def g (x : ℝ) := x * Real.exp (-x)

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def monotone_decreasing_intervals (f : ℝ → ℝ) : Set (Set ℝ) :=
  {I | ∃ k : ℤ, I = Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3) ∧
    is_monotone_decreasing f (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3)}

theorem f_monotone_decreasing : 
  monotone_decreasing_intervals f = {I | ∃ k : ℤ, I = Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3)} :=
sorry

theorem min_a_value :
  (∃ a : ℝ, ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 0 (Real.pi / 2) → 
    g x₁ + a + 3 > f x₂) ∧
  (∀ a' : ℝ, a' < -3 / Real.exp 3 → 
    ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 ∧ x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
      g x₁ + a' + 3 ≤ f x₂) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_min_a_value_l2181_218150


namespace NUMINAMATH_CALUDE_maria_score_is_15_l2181_218104

/-- Represents a quiz result -/
structure QuizResult where
  total_questions : Nat
  correct_answers : Nat
  incorrect_answers : Nat
  unanswered_questions : Nat
  deriving Repr

/-- Calculates the score for a quiz result -/
def calculate_score (result : QuizResult) : Nat :=
  result.correct_answers

/-- Maria's quiz result -/
def maria_result : QuizResult :=
  { total_questions := 20
  , correct_answers := 15
  , incorrect_answers := 3
  , unanswered_questions := 2
  }

theorem maria_score_is_15 :
  calculate_score maria_result = 15 ∧
  maria_result.total_questions = maria_result.correct_answers + maria_result.incorrect_answers + maria_result.unanswered_questions :=
by sorry

end NUMINAMATH_CALUDE_maria_score_is_15_l2181_218104


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2181_218169

theorem polynomial_simplification (x : ℝ) : 
  (3*x - 2) * (5*x^12 + 3*x^11 + 2*x^10 - x^9) = 
  15*x^13 - x^12 - 7*x^10 + 2*x^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2181_218169


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2181_218144

theorem complex_fraction_simplification :
  (3 + Complex.I) / (1 - Complex.I) = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2181_218144


namespace NUMINAMATH_CALUDE_short_show_episodes_count_l2181_218115

/-- The number of episodes of the short show -/
def short_show_episodes : ℕ := 24

/-- The duration of one episode of the short show in hours -/
def short_show_duration : ℚ := 1/2

/-- The duration of one episode of the long show in hours -/
def long_show_duration : ℚ := 1

/-- The number of episodes of the long show -/
def long_show_episodes : ℕ := 12

/-- The total time Tim watched TV in hours -/
def total_watch_time : ℕ := 24

theorem short_show_episodes_count :
  short_show_episodes * short_show_duration + long_show_episodes * long_show_duration = total_watch_time := by
  sorry

end NUMINAMATH_CALUDE_short_show_episodes_count_l2181_218115


namespace NUMINAMATH_CALUDE_value_of_d_l2181_218188

theorem value_of_d (r s t u d : ℕ+) 
  (h1 : r^5 = s^4)
  (h2 : t^3 = u^2)
  (h3 : t - r = 19)
  (h4 : d = u - s) :
  d = 757 := by
  sorry

end NUMINAMATH_CALUDE_value_of_d_l2181_218188


namespace NUMINAMATH_CALUDE_one_minus_repeating_eight_eq_one_ninth_l2181_218175

/-- The repeating decimal 0.overline{8} -/
def repeating_eight : ℚ := 8/9

/-- Theorem stating that 1 minus the repeating decimal 0.overline{8} equals 1/9 -/
theorem one_minus_repeating_eight_eq_one_ninth : 1 - repeating_eight = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_eight_eq_one_ninth_l2181_218175


namespace NUMINAMATH_CALUDE_eliminate_x_from_system_l2181_218111

theorem eliminate_x_from_system : ∀ x y : ℝ,
  (2 * x - 3 * y = 11 ∧ 2 * x + 5 * y = -5) →
  -8 * y = 16 := by
  sorry

end NUMINAMATH_CALUDE_eliminate_x_from_system_l2181_218111


namespace NUMINAMATH_CALUDE_scientific_notation_of_899000_l2181_218177

/-- Theorem: 899,000 expressed in scientific notation is 8.99 × 10^5 -/
theorem scientific_notation_of_899000 :
  899000 = 8.99 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_899000_l2181_218177


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2181_218108

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_correct : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h1 : seq.S 6 = 8 * seq.S 3)
    (h2 : seq.a 3 - seq.a 5 = 8) :
  seq.a 20 = -74 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2181_218108


namespace NUMINAMATH_CALUDE_bowling_score_problem_l2181_218184

theorem bowling_score_problem (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 50 →
  (sarah_score + greg_score) / 2 = 110 →
  sarah_score = 135 := by
sorry

end NUMINAMATH_CALUDE_bowling_score_problem_l2181_218184


namespace NUMINAMATH_CALUDE_log_equation_solution_l2181_218121

theorem log_equation_solution :
  ∃ x : ℝ, (Real.log x - 4 * Real.log 5 = -3) ∧ (x = 0.625) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2181_218121


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l2181_218197

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base : ℝ) (height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 cm and height 10 cm is 120 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 12 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l2181_218197


namespace NUMINAMATH_CALUDE_solve_equation_l2181_218106

theorem solve_equation (x : ℚ) : (2 * x + 3) / 5 = 11 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2181_218106


namespace NUMINAMATH_CALUDE_tims_car_value_l2181_218143

/-- Represents the value of a car over time -/
def car_value (initial_value : ℕ) (depreciation_rate : ℕ) (years : ℕ) : ℕ :=
  initial_value - depreciation_rate * years

/-- Theorem stating the value of Tim's car after 6 years -/
theorem tims_car_value : car_value 20000 1000 6 = 14000 := by
  sorry

end NUMINAMATH_CALUDE_tims_car_value_l2181_218143


namespace NUMINAMATH_CALUDE_f_always_negative_f_less_than_2x_minus_3_l2181_218185

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x - 1

-- Part 1: Range of a for which f(x) < 0 for all x
theorem f_always_negative (a : ℝ) : 
  (∀ x : ℝ, f a x < 0) ↔ a ∈ Set.Ioc (-4) 0 :=
sorry

-- Part 2: Solution to f(x) < 2x - 3 when a > 0
theorem f_less_than_2x_minus_3 (a : ℝ) (h : a > 0) :
  {x : ℝ | f a x < 2 * x - 3} = 
    if a < 2 then Set.Ioo 1 (2/a)
    else if a = 2 then ∅ 
    else Set.Ioo (2/a) 1 :=
sorry

end NUMINAMATH_CALUDE_f_always_negative_f_less_than_2x_minus_3_l2181_218185


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2181_218105

/-- Given k = 2012² + 2^2014, prove that (k² + 2^k) mod 10 = 5 -/
theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : k = 2012^2 + 2^2014 → (k^2 + 2^k) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2181_218105


namespace NUMINAMATH_CALUDE_exposed_sides_count_l2181_218101

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : Nat
  sides_positive : sides > 0

/-- The sequence of regular polygons in the construction. -/
def polygon_sequence : List RegularPolygon :=
  [⟨3, by norm_num⟩, ⟨4, by norm_num⟩, ⟨5, by norm_num⟩, 
   ⟨6, by norm_num⟩, ⟨7, by norm_num⟩, ⟨8, by norm_num⟩, ⟨9, by norm_num⟩]

/-- Calculates the number of exposed sides for a given polygon in the sequence. -/
def exposed_sides (p : RegularPolygon) (index : Nat) : Nat :=
  if index = 0 ∨ index = 6 then p.sides - 1 else p.sides - 2

/-- The total number of exposed sides in the polygon sequence. -/
def total_exposed_sides : Nat :=
  (List.zipWith exposed_sides polygon_sequence (List.range 7)).sum

theorem exposed_sides_count :
  total_exposed_sides = 30 := by
  sorry

end NUMINAMATH_CALUDE_exposed_sides_count_l2181_218101


namespace NUMINAMATH_CALUDE_apple_tree_production_decrease_l2181_218114

theorem apple_tree_production_decrease (season1 season2 season3 total : ℕ) : 
  season1 = 200 →
  season3 = 2 * season2 →
  total = season1 + season2 + season3 →
  total = 680 →
  (season1 - season2 : ℚ) / season1 = 1/5 := by sorry

end NUMINAMATH_CALUDE_apple_tree_production_decrease_l2181_218114


namespace NUMINAMATH_CALUDE_seven_digit_subtraction_l2181_218103

def is_seven_digit (n : ℕ) : Prop := 1000000 ≤ n ∧ n ≤ 9999999

def digit_sum_except_second (n : ℕ) : ℕ :=
  let digits := (Nat.digits 10 n).reverse
  List.sum (digits.take 1 ++ digits.drop 2)

theorem seven_digit_subtraction (n : ℕ) :
  is_seven_digit n →
  ∃ k, n - k = 9875352 →
  n - digit_sum_except_second n = 9875357 :=
sorry

end NUMINAMATH_CALUDE_seven_digit_subtraction_l2181_218103


namespace NUMINAMATH_CALUDE_problem_statement_l2181_218168

theorem problem_statement (m n : ℤ) : 
  |m - 2023| + (n + 2024)^2 = 0 → (m + n)^2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2181_218168


namespace NUMINAMATH_CALUDE_train_speed_l2181_218151

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length time_to_pass : ℝ) :
  train_length = 300 →
  bridge_length = 115 →
  time_to_pass = 42.68571428571429 →
  ∃ (speed : ℝ), abs (speed - 35.01) < 0.01 ∧ 
    speed = (train_length + bridge_length) / time_to_pass * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2181_218151


namespace NUMINAMATH_CALUDE_area_of_XYZW_l2181_218192

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the larger rectangle XYZW -/
def XYZW : Rectangle := { width := 14, height := 28 }

/-- Represents one of the smaller identical rectangles -/
def smallRect : Rectangle := { width := 7, height := 14 }

theorem area_of_XYZW :
  XYZW.width = smallRect.height ∧
  XYZW.height = 3 * smallRect.width + smallRect.width ∧
  XYZW.area = 392 := by
  sorry

#check area_of_XYZW

end NUMINAMATH_CALUDE_area_of_XYZW_l2181_218192


namespace NUMINAMATH_CALUDE_hexagon_game_theorem_l2181_218155

/-- Represents a hexagonal grid cell -/
structure HexCell where
  x : ℤ
  y : ℤ

/-- Represents the state of a cell (empty or filled) -/
inductive CellState
  | Empty
  | Filled

/-- Represents the game state -/
structure GameState where
  grid : HexCell → CellState
  turn : ℕ

/-- Represents a player's move -/
inductive Move
  | PlaceCounters (c1 c2 : HexCell)
  | RemoveCounter (c : HexCell)

/-- Checks if two hexagonal cells are adjacent -/
def are_adjacent (c1 c2 : HexCell) : Prop :=
  sorry

/-- Checks if there are k consecutive filled cells in a line -/
def has_k_consecutive_filled (g : GameState) (k : ℕ) : Prop :=
  sorry

/-- Applies a move to the game state -/
def apply_move (g : GameState) (m : Move) : GameState :=
  sorry

/-- Checks if a move is valid according to the game rules -/
def is_valid_move (g : GameState) (m : Move) : Prop :=
  sorry

/-- Represents a winning strategy for player A -/
def winning_strategy (k : ℕ) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (g : GameState),
      is_valid_move g (strategy g) ∧
      (∃ (n : ℕ), has_k_consecutive_filled (apply_move g (strategy g)) k)

/-- The main theorem stating that 6 is the minimum value of k for which A cannot win -/
theorem hexagon_game_theorem :
  (∀ (k : ℕ), k < 6 → winning_strategy k) ∧
  ¬(winning_strategy 6) :=
sorry

end NUMINAMATH_CALUDE_hexagon_game_theorem_l2181_218155


namespace NUMINAMATH_CALUDE_book_pages_l2181_218112

/-- Calculates the total number of pages in a book given reading rate and time spent reading. -/
def total_pages (pages_per_hour : ℝ) (monday_hours : ℝ) (tuesday_hours : ℝ) (remaining_hours : ℝ) : ℝ :=
  pages_per_hour * (monday_hours + tuesday_hours + remaining_hours)

/-- Theorem stating that the book has 248 pages given Joanna's reading rate and time spent. -/
theorem book_pages : 
  let pages_per_hour : ℝ := 16
  let monday_hours : ℝ := 3
  let tuesday_hours : ℝ := 6.5
  let remaining_hours : ℝ := 6
  total_pages pages_per_hour monday_hours tuesday_hours remaining_hours = 248 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l2181_218112


namespace NUMINAMATH_CALUDE_five_digit_palindromes_count_l2181_218195

/-- A function that returns the number of positive five-digit palindromic integers -/
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10

/-- Theorem stating that the number of positive five-digit palindromic integers is 900 -/
theorem five_digit_palindromes_count :
  count_five_digit_palindromes = 900 := by
  sorry

#eval count_five_digit_palindromes

end NUMINAMATH_CALUDE_five_digit_palindromes_count_l2181_218195


namespace NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l2181_218172

/-- The interest rate at which A lent money to B, given the conditions of the problem -/
def interest_rate_A_to_B (principal : ℚ) (rate_B_to_C : ℚ) (time : ℚ) (B_gain : ℚ) : ℚ :=
  let interest_from_C := principal * rate_B_to_C * time
  let interest_to_A := interest_from_C - B_gain
  (interest_to_A / (principal * time)) * 100

/-- Theorem stating that the interest rate from A to B is 10% under the given conditions -/
theorem interest_rate_is_ten_percent :
  interest_rate_A_to_B 3500 0.13 3 315 = 10 := by sorry

end NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l2181_218172


namespace NUMINAMATH_CALUDE_proposition_a_necessary_not_sufficient_l2181_218124

theorem proposition_a_necessary_not_sufficient :
  (∀ a b : ℝ, a > b ∧ a⁻¹ > b⁻¹ → a > 0) ∧
  (∃ a b : ℝ, a > 0 ∧ a > b ∧ a⁻¹ ≤ b⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_proposition_a_necessary_not_sufficient_l2181_218124


namespace NUMINAMATH_CALUDE_symmetry_composition_l2181_218165

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry operations
def symmetryXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

def symmetryYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

-- Theorem statement
theorem symmetry_composition (a b : ℝ) :
  let M : Point2D := { x := a, y := b }
  let N : Point2D := symmetryXAxis M
  let P : Point2D := symmetryYAxis N
  let Q : Point2D := symmetryXAxis P
  let R : Point2D := symmetryYAxis Q
  R = M := by sorry

end NUMINAMATH_CALUDE_symmetry_composition_l2181_218165


namespace NUMINAMATH_CALUDE_steves_oranges_l2181_218170

/-- Given that Steve shares some oranges and has a certain number left, 
    this theorem proves how many oranges Steve had initially. -/
theorem steves_oranges (shared : ℕ) (left : ℕ) (initial : ℕ) : 
  shared = 4 → left = 42 → initial = shared + left → initial = 46 := by
  sorry

end NUMINAMATH_CALUDE_steves_oranges_l2181_218170


namespace NUMINAMATH_CALUDE_equation_solution_l2181_218107

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (8 * x)^12 = (16 * x)^6 ↔ x = 1/4 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2181_218107


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2181_218180

def M : ℕ := 45 * 45 * 98 * 340

def sum_of_even_divisors (n : ℕ) : ℕ := (List.filter (λ x => x % 2 = 0) (List.range (n + 1))).sum

def sum_of_odd_divisors (n : ℕ) : ℕ := (List.filter (λ x => x % 2 ≠ 0) (List.range (n + 1))).sum

theorem ratio_of_divisor_sums :
  (sum_of_even_divisors M) / (sum_of_odd_divisors M) = 14 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2181_218180


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l2181_218145

theorem right_triangle_hypotenuse_and_perimeter :
  let a : ℝ := 8.5
  let b : ℝ := 15
  let h : ℝ := Real.sqrt (a^2 + b^2)
  let perimeter : ℝ := a + b + h
  h = 17.25 ∧ perimeter = 40.75 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l2181_218145


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l2181_218158

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, x * (f (x + 1) - f x) = f x) ∧
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|)

/-- The theorem stating the form of functions satisfying the conditions -/
theorem characterize_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfyingFunction f →
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ |k| ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l2181_218158


namespace NUMINAMATH_CALUDE_base_three_20121_equals_178_l2181_218139

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_three_20121_equals_178 :
  base_three_to_decimal [2, 0, 1, 2, 1] = 178 := by
  sorry

end NUMINAMATH_CALUDE_base_three_20121_equals_178_l2181_218139


namespace NUMINAMATH_CALUDE_coltons_marbles_l2181_218109

theorem coltons_marbles (white_marbles : ℕ) : 
  (∃ (groups : ℕ), groups = 8 ∧ (16 + white_marbles) % groups = 0) →
  ∃ (k : ℕ), white_marbles = 8 * k :=
by sorry

end NUMINAMATH_CALUDE_coltons_marbles_l2181_218109


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2181_218140

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given two vectors a and b in R², where a = (-4, 3) and b = (6, m),
    if a is perpendicular to b, then m = 8 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (-4, 3)
  let b : ℝ × ℝ := (6, m)
  perpendicular a b → m = 8 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2181_218140


namespace NUMINAMATH_CALUDE_simplify_expression_l2181_218153

theorem simplify_expression (w : ℝ) : 3*w + 6*w - 9*w + 12*w - 15*w + 21 = -3*w + 21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2181_218153


namespace NUMINAMATH_CALUDE_fibSum_eq_three_l2181_218149

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of F_n / 2^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (2 : ℝ) ^ n

/-- Theorem stating that the sum of F_n / 2^n from n = 0 to infinity equals 3 -/
theorem fibSum_eq_three : fibSum = 3 := by sorry

end NUMINAMATH_CALUDE_fibSum_eq_three_l2181_218149


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2181_218135

theorem complex_modulus_problem (z : ℂ) :
  (1 + Complex.I * Real.sqrt 3)^2 * z = 1 - Complex.I^3 →
  Complex.abs z = Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2181_218135


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_m_range_l2181_218147

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | m / 2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

def B (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2 * m + 1}

-- State the theorem
theorem intersection_nonempty_implies_m_range (m : ℝ) :
  (A m ∩ B m).Nonempty → 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_m_range_l2181_218147


namespace NUMINAMATH_CALUDE_M_divisible_by_41_l2181_218141

def M : ℕ := sorry

theorem M_divisible_by_41 : 41 ∣ M := by sorry

end NUMINAMATH_CALUDE_M_divisible_by_41_l2181_218141


namespace NUMINAMATH_CALUDE_minotaur_returns_l2181_218127

/-- A room in the Minotaur's palace -/
structure Room where
  id : Nat

/-- A direction the Minotaur can turn -/
inductive Direction
  | Left
  | Right

/-- The state of the Minotaur's journey -/
structure State where
  room : Room
  enteredThrough : Nat
  nextTurn : Direction

/-- The palace with its room connections -/
structure Palace where
  rooms : Finset Room
  connections : Room → Finset (Nat × Room)
  room_count : rooms.card = 1000000
  three_corridors : ∀ r : Room, (connections r).card = 3

/-- The function that determines the next state based on the current state -/
def nextState (p : Palace) (s : State) : State :=
  sorry

/-- The theorem stating that the Minotaur will eventually return to the starting room -/
theorem minotaur_returns (p : Palace) (start : State) :
  ∃ n : Nat, (Nat.iterate (nextState p) n start).room = start.room :=
sorry

end NUMINAMATH_CALUDE_minotaur_returns_l2181_218127


namespace NUMINAMATH_CALUDE_indira_cricket_time_l2181_218128

/-- Sean's daily cricket playing time in minutes -/
def sean_daily_time : ℕ := 50

/-- Number of days Sean played cricket -/
def sean_days : ℕ := 14

/-- Total time Sean and Indira played cricket together in minutes -/
def total_time : ℕ := 1512

/-- Calculate Indira's cricket playing time -/
def indira_time : ℕ := total_time - (sean_daily_time * sean_days)

/-- Theorem stating Indira's cricket playing time -/
theorem indira_cricket_time : indira_time = 812 := by sorry

end NUMINAMATH_CALUDE_indira_cricket_time_l2181_218128


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l2181_218119

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l2181_218119


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2181_218120

theorem largest_integer_satisfying_inequality :
  ∀ y : ℤ, y ≤ 5 ↔ y / 3 + 5 / 3 < 11 / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2181_218120


namespace NUMINAMATH_CALUDE_flower_shop_ratio_l2181_218183

/-- Flower shop problem -/
theorem flower_shop_ratio : 
  ∀ (roses lilacs gardenias : ℕ),
  roses = 3 * lilacs →
  lilacs = 10 →
  roses + lilacs + gardenias = 45 →
  gardenias / lilacs = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_ratio_l2181_218183


namespace NUMINAMATH_CALUDE_plane_line_relations_l2181_218171

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (intersects : Plane → Plane → Line → Prop)
variable (within : Line → Plane → Prop)
variable (ne : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem plane_line_relations
  (α β : Plane) (l m : Line)
  (h1 : intersects α β l)
  (h2 : within m α)
  (h3 : ne m l) :
  (parallel m β → parallel_lines m l) ∧
  (parallel_lines m l → parallel m β) ∧
  (perpendicular m β → perpendicular_lines m l) ∧
  ¬(perpendicular_lines m l → perpendicular m β) :=
by sorry

end NUMINAMATH_CALUDE_plane_line_relations_l2181_218171


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l2181_218148

theorem geometric_sequence_solution (a : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ (2*a + 2) = a * r ∧ (3*a + 3) = (2*a + 2) * r) → a = -4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l2181_218148


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l2181_218190

/-- Calculates the upstream distance traveled by a boat in one hour -/
def upstreamDistance (stillWaterSpeed : ℝ) (downstreamDistance : ℝ) : ℝ :=
  let streamSpeed := downstreamDistance - stillWaterSpeed
  stillWaterSpeed - streamSpeed

theorem boat_upstream_distance :
  upstreamDistance 5 8 = 2 := by
  sorry

#eval upstreamDistance 5 8

end NUMINAMATH_CALUDE_boat_upstream_distance_l2181_218190


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_stratified_sampling_female_count_correct_l2181_218198

theorem stratified_sampling_female_count 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 75) 
  (h2 : male_employees = 30) 
  (h3 : sample_size = 20) : ℕ :=
let female_employees := total_employees - male_employees
let sample_ratio := sample_size / total_employees
let female_sample := (female_employees : ℚ) * sample_ratio
12

theorem stratified_sampling_female_count_correct 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 75) 
  (h2 : male_employees = 30) 
  (h3 : sample_size = 20) : 
  stratified_sampling_female_count total_employees male_employees sample_size h1 h2 h3 = 12 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_stratified_sampling_female_count_correct_l2181_218198


namespace NUMINAMATH_CALUDE_set_range_proof_l2181_218194

theorem set_range_proof (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ensuring the order of numbers
  a = 2 ∧  -- Smallest number is 2
  b = 5 ∧  -- Median is 5
  (a + b + c) / 3 = 5 →  -- Mean is 5
  c - a = 6 :=  -- Range is 6
by sorry

end NUMINAMATH_CALUDE_set_range_proof_l2181_218194


namespace NUMINAMATH_CALUDE_family_money_difference_l2181_218189

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- Calculate the total value of coins for a person -/
def total_value (quarters dimes nickels : ℕ) : ℚ :=
  quarters * quarter_value + dimes * dime_value + nickels * nickel_value

/-- Karen's total value -/
def karen_value : ℚ := total_value 32 0 0

/-- Christopher's total value -/
def christopher_value : ℚ := total_value 64 0 0

/-- Emily's total value -/
def emily_value : ℚ := total_value 20 15 0

/-- Michael's total value -/
def michael_value : ℚ := total_value 12 10 25

/-- Sophia's total value -/
def sophia_value : ℚ := total_value 0 50 40

/-- Alex's total value -/
def alex_value : ℚ := total_value 0 25 100

/-- Total value for Karen and Christopher's family -/
def family1_value : ℚ := karen_value + christopher_value + emily_value + michael_value

/-- Total value for Sophia and Alex's family -/
def family2_value : ℚ := sophia_value + alex_value

theorem family_money_difference :
  family1_value - family2_value = 85/4 := by sorry

end NUMINAMATH_CALUDE_family_money_difference_l2181_218189


namespace NUMINAMATH_CALUDE_triangle_properties_l2181_218160

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle ABC -/
theorem triangle_properties (t : Triangle) 
  (h1 : 3 * Real.cos t.A * Real.cos t.C * (Real.tan t.A * Real.tan t.C - 1) = 1)
  (h2 : t.a + t.c = 3 * Real.sqrt 3 / 2)
  (h3 : t.b = Real.sqrt 3) : 
  Real.sin (2 * t.B - 5 * Real.pi / 6) = (7 - 4 * Real.sqrt 6) / 18 ∧ 
  t.a * t.c * Real.sin t.B / 2 = 15 * Real.sqrt 2 / 32 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2181_218160


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2181_218179

theorem expand_and_simplify (x : ℝ) : (2*x - 3)^2 - (x + 3)*(x - 2) = 3*x^2 - 13*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2181_218179


namespace NUMINAMATH_CALUDE_multiplication_equality_l2181_218157

theorem multiplication_equality : 500 * 3986 * 0.3986 * 5 = 0.25 * 3986^2 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equality_l2181_218157


namespace NUMINAMATH_CALUDE_auspicious_count_l2181_218126

/-- Returns true if n is an auspicious number (multiple of 6 with digit sum 6) -/
def isAuspicious (n : Nat) : Bool :=
  n % 6 = 0 && (n / 100 + (n / 10) % 10 + n % 10 = 6)

/-- Count of auspicious numbers between 100 and 999 -/
def countAuspicious : Nat :=
  (List.range 900).map (· + 100)
    |>.filter isAuspicious
    |>.length

theorem auspicious_count : countAuspicious = 12 := by
  sorry

end NUMINAMATH_CALUDE_auspicious_count_l2181_218126


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2181_218131

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℤ
  common_difference : ℤ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first_term + (n - 1 : ℤ) * seq.common_difference

theorem arithmetic_sequence_eighth_term
  (seq : ArithmeticSequence)
  (h4 : seq.nthTerm 4 = 23)
  (h6 : seq.nthTerm 6 = 47) :
  seq.nthTerm 8 = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2181_218131


namespace NUMINAMATH_CALUDE_quadratic_root_product_l2181_218182

theorem quadratic_root_product (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ - 2 = 0) → 
  (x₂^2 - 4*x₂ - 2 = 0) → 
  x₁ * x₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l2181_218182


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2181_218134

theorem quadratic_expression_value (a b c : ℝ) 
  (h1 : a - b = 2 + Real.sqrt 3)
  (h2 : b - c = 2 - Real.sqrt 3) :
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2181_218134


namespace NUMINAMATH_CALUDE_gcd_multiple_equivalence_l2181_218110

theorem gcd_multiple_equivalence (d : ℕ) (h : d ≥ 1) :
  {m : ℕ | m ≥ 2 ∧ d ∣ m} =
  {m : ℕ | m ≥ 2 ∧ ∃ n : ℕ, n ≥ 1 ∧ Nat.gcd m n = d ∧ Nat.gcd m (4 * n + 1) = 1} :=
by sorry

end NUMINAMATH_CALUDE_gcd_multiple_equivalence_l2181_218110


namespace NUMINAMATH_CALUDE_negation_true_l2181_218136

theorem negation_true : ¬(∃ x : ℝ, 0 < x ∧ x < π ∧ x + 1 / Real.sin x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_negation_true_l2181_218136


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2181_218193

theorem quadratic_always_positive : ∀ x : ℝ, x^2 - x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2181_218193


namespace NUMINAMATH_CALUDE_greatest_four_digit_number_l2181_218161

def reverse_number (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n)
  List.foldl (fun acc d => 10 * acc + d) 0 digits

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem greatest_four_digit_number (m : ℕ) 
  (h1 : is_four_digit m)
  (h2 : m % 36 = 0)
  (h3 : m % 7 = 0)
  (h4 : (reverse_number m) % 36 = 0) :
  m ≤ 5796 ∧ ∃ (n : ℕ), n = 5796 ∧ 
    is_four_digit n ∧ 
    n % 36 = 0 ∧ 
    n % 7 = 0 ∧ 
    (reverse_number n) % 36 = 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_number_l2181_218161


namespace NUMINAMATH_CALUDE_volume_of_rotated_figure_l2181_218173

-- Define the figure
def figure (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define the volume of the solid formed by rotation
def volume_of_rotation (f : ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem volume_of_rotated_figure :
  volume_of_rotation figure = 4 * Real.pi^2 := by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_figure_l2181_218173


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l2181_218164

theorem interest_difference_theorem (P : ℝ) : 
  let r : ℝ := 5 / 100  -- 5% interest rate
  let t : ℝ := 2        -- 2 years
  let simple_interest := P * r * t
  let compound_interest := P * ((1 + r) ^ t - 1)
  compound_interest - simple_interest = 20 → P = 8000 := by
sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l2181_218164


namespace NUMINAMATH_CALUDE_triangle_count_l2181_218118

theorem triangle_count (num_circles : ℕ) (num_triangles : ℕ) : 
  num_circles = 5 → num_triangles = 2 * num_circles → num_triangles = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l2181_218118


namespace NUMINAMATH_CALUDE_parking_lot_car_decrease_l2181_218174

theorem parking_lot_car_decrease (initial_cars : ℕ) (cars_out : ℕ) (cars_in : ℕ) : 
  initial_cars = 25 → cars_out = 18 → cars_in = 12 → 
  initial_cars - ((initial_cars - cars_out) + cars_in) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_lot_car_decrease_l2181_218174


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l2181_218100

theorem not_p_necessary_not_sufficient_for_not_q :
  ∃ (x : ℝ), (¬(x^2 < 1) → ¬(x < 1)) ∧
  ∃ (y : ℝ), ¬(y < 1) ∧ ¬¬(y^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l2181_218100


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2181_218191

theorem rectangular_prism_diagonal (a b c : ℝ) (ha : a = 12) (hb : b = 15) (hc : c = 8) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 433 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2181_218191


namespace NUMINAMATH_CALUDE_riddle_ratio_l2181_218125

theorem riddle_ratio (josh_riddles ivory_riddles taso_riddles : ℕ) : 
  josh_riddles = 8 →
  ivory_riddles = josh_riddles + 4 →
  taso_riddles = 24 →
  (taso_riddles : ℚ) / ivory_riddles = 2 := by
  sorry

end NUMINAMATH_CALUDE_riddle_ratio_l2181_218125


namespace NUMINAMATH_CALUDE_seventh_equation_values_first_four_equations_check_l2181_218137

/-- Represents the last number on the left side of each equation -/
def last_left (n : ℕ) : ℕ := 1 + 3 * (n - 1)

/-- Represents the result on the right side of each equation -/
def right_result (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The theorem to be proved -/
theorem seventh_equation_values :
  last_left 7 = 19 ∧ right_result 7 = 169 := by
  sorry

/-- Verification of the first four equations -/
theorem first_four_equations_check :
  (last_left 1 = 1 ∧ right_result 1 = 1) ∧
  (last_left 2 = 4 ∧ right_result 2 = 9) ∧
  (last_left 3 = 7 ∧ right_result 3 = 25) ∧
  (last_left 4 = 10 ∧ right_result 4 = 49) := by
  sorry

end NUMINAMATH_CALUDE_seventh_equation_values_first_four_equations_check_l2181_218137


namespace NUMINAMATH_CALUDE_red_balls_count_l2181_218196

/-- The number of blue balls in the bin -/
def blue_balls : ℕ := 7

/-- The amount won when drawing a blue ball -/
def blue_win : ℤ := 3

/-- The amount lost when drawing a red ball -/
def red_loss : ℤ := 1

/-- The expected value of the game -/
def expected_value : ℚ := 1

/-- The number of red balls in the bin -/
def red_balls : ℕ := sorry

theorem red_balls_count : red_balls = 7 := by sorry

end NUMINAMATH_CALUDE_red_balls_count_l2181_218196


namespace NUMINAMATH_CALUDE_expression_evaluation_l2181_218176

theorem expression_evaluation :
  ∀ x : ℕ, 
    x - 3 < 0 →
    x - 1 ≠ 0 →
    x - 2 ≠ 0 →
    (3 / (x - 1) - x - 1) / ((x - 2) / (x^2 - 2*x + 1)) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2181_218176


namespace NUMINAMATH_CALUDE_first_number_is_thirty_l2181_218122

theorem first_number_is_thirty (x y : ℝ) 
  (sum_eq : x + y = 50) 
  (diff_eq : 2 * (x - y) = 20) : 
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_first_number_is_thirty_l2181_218122


namespace NUMINAMATH_CALUDE_inverse_of_100_mod_101_l2181_218142

theorem inverse_of_100_mod_101 : ∃ x : ℕ, x ≥ 0 ∧ x ≤ 100 ∧ (100 * x) % 101 = 1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_100_mod_101_l2181_218142


namespace NUMINAMATH_CALUDE_sphere_wall_thickness_l2181_218154

/-- Represents a hollow glass sphere floating in water -/
structure FloatingSphere where
  outer_diameter : ℝ
  specific_gravity : ℝ
  dry_surface_fraction : ℝ

/-- Calculates the wall thickness of a floating sphere -/
noncomputable def wall_thickness (sphere : FloatingSphere) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the wall thickness of the sphere with given properties -/
theorem sphere_wall_thickness :
  let sphere : FloatingSphere := {
    outer_diameter := 16,
    specific_gravity := 2.523,
    dry_surface_fraction := 3/8
  }
  wall_thickness sphere = 0.8 := by sorry

end NUMINAMATH_CALUDE_sphere_wall_thickness_l2181_218154


namespace NUMINAMATH_CALUDE_max_value_theorem_l2181_218187

theorem max_value_theorem (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 2*x + y + z = 4) : 
  x^2 + x*(y + z) + y*z ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2181_218187


namespace NUMINAMATH_CALUDE_quadratic_root_is_one_l2181_218162

/-- 
Given a quadratic function f(x) = x^2 + ax + b, where:
- The graph of f intersects the y-axis at (0, b)
- The graph of f intersects the x-axis at (b, 0)
- b ≠ 0

Prove that the other root of f(x) = 0 is equal to 1.
-/
theorem quadratic_root_is_one (a b : ℝ) (hb : b ≠ 0) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  (f 0 = b) → (f b = 0) → ∃ c, c ≠ b ∧ f c = 0 ∧ c = 1 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_root_is_one_l2181_218162
