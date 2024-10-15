import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l4114_411400

theorem inequality_proof (x : ℝ) (n : ℕ) (a : ℝ) 
  (h1 : x > 0) (h2 : n > 0) (h3 : x + a / x^n ≥ n + 1) : 
  a = n^n := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4114_411400


namespace NUMINAMATH_CALUDE_peanuts_equation_initial_peanuts_count_l4114_411423

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Mary adds to the box -/
def peanuts_added : ℕ := 8

/-- The total number of peanuts after Mary adds more -/
def total_peanuts : ℕ := 12

/-- Theorem stating that the initial number of peanuts plus the added peanuts equals the total peanuts -/
theorem peanuts_equation : initial_peanuts + peanuts_added = total_peanuts := by sorry

/-- Theorem proving that the initial number of peanuts is 4 -/
theorem initial_peanuts_count : initial_peanuts = 4 := by sorry

end NUMINAMATH_CALUDE_peanuts_equation_initial_peanuts_count_l4114_411423


namespace NUMINAMATH_CALUDE_unit_square_folding_l4114_411476

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 1 -/
structure UnitSquare where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a line segment between two other points -/
def isOnSegment (P Q R : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    P.x = Q.x + t * (R.x - Q.x) ∧
    P.y = Q.y + t * (R.y - Q.y)

/-- Checks if two line segments intersect -/
def segmentsIntersect (P Q R S : Point) : Prop :=
  ∃ I : Point, isOnSegment I P Q ∧ isOnSegment I R S

theorem unit_square_folding (ABCD : UnitSquare) 
  (E : Point) (F : Point) 
  (hE : isOnSegment E ABCD.A ABCD.B) 
  (hF : isOnSegment F ABCD.C ABCD.B) 
  (hF_mid : F.x = 1 ∧ F.y = 1/2) 
  (hFold : segmentsIntersect ABCD.A ABCD.D E F ∧ 
           segmentsIntersect ABCD.C ABCD.D E F) : 
  E.x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unit_square_folding_l4114_411476


namespace NUMINAMATH_CALUDE_trip_distance_is_3_6_miles_l4114_411413

/-- Calculates the trip distance given the taxi fare parameters -/
def calculate_trip_distance (initial_fee : ℚ) (additional_charge : ℚ) (charge_distance : ℚ) (total_charge : ℚ) : ℚ :=
  let distance_charge := total_charge - initial_fee
  let segments := distance_charge / additional_charge
  segments * charge_distance

/-- Proves that the trip distance is 3.6 miles given the specified taxi fare parameters -/
theorem trip_distance_is_3_6_miles :
  let initial_fee : ℚ := 9/4  -- $2.25
  let additional_charge : ℚ := 3/10  -- $0.3
  let charge_distance : ℚ := 2/5  -- 2/5 mile
  let total_charge : ℚ := 99/20  -- $4.95
  calculate_trip_distance initial_fee additional_charge charge_distance total_charge = 18/5  -- 3.6 miles
  := by sorry

end NUMINAMATH_CALUDE_trip_distance_is_3_6_miles_l4114_411413


namespace NUMINAMATH_CALUDE_combined_earnings_l4114_411483

def dwayne_earnings : ℕ := 1500
def brady_extra : ℕ := 450

theorem combined_earnings :
  dwayne_earnings + (dwayne_earnings + brady_extra) = 3450 :=
by sorry

end NUMINAMATH_CALUDE_combined_earnings_l4114_411483


namespace NUMINAMATH_CALUDE_combination_sum_equality_l4114_411459

theorem combination_sum_equality (n k m : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  (Nat.choose n m) + 
  (Finset.sum (Finset.range k) (fun i => Nat.choose k (i + 1) * Nat.choose n (m - (i + 1)))) + 
  (Nat.choose n (m - k)) = 
  Nat.choose (n + k) m := by sorry

end NUMINAMATH_CALUDE_combination_sum_equality_l4114_411459


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4114_411496

theorem complex_equation_solution (z : ℂ) (h : 10 * Complex.normSq z = 3 * Complex.normSq (z + 3) + Complex.normSq (z^2 - 1) + 40) :
  z + 9 / z = (9 + Real.sqrt 61) / 2 ∨ z + 9 / z = (9 - Real.sqrt 61) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4114_411496


namespace NUMINAMATH_CALUDE_david_money_left_is_275_l4114_411488

/-- Represents the amount of money David has left at the end of his trip -/
def david_money_left (initial_amount accommodations food_euros food_exchange_rate souvenirs_yen souvenirs_exchange_rate loan : ℚ) : ℚ :=
  let total_spent := accommodations + (food_euros * food_exchange_rate) + (souvenirs_yen * souvenirs_exchange_rate)
  initial_amount - total_spent - 500

/-- Theorem stating that David has $275 left at the end of his trip -/
theorem david_money_left_is_275 :
  david_money_left 1500 400 300 1.1 5000 0.009 200 = 275 := by
  sorry

end NUMINAMATH_CALUDE_david_money_left_is_275_l4114_411488


namespace NUMINAMATH_CALUDE_two_plus_three_equals_twentysix_l4114_411432

/-- Defines the sequence operation for two consecutive terms -/
def sequenceOperation (a b : ℕ) : ℕ := (a + b)^2 + 1

/-- Theorem stating that 2 + 3 in the given sequence equals 26 -/
theorem two_plus_three_equals_twentysix :
  sequenceOperation 2 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_two_plus_three_equals_twentysix_l4114_411432


namespace NUMINAMATH_CALUDE_surface_area_is_34_l4114_411470

/-- A three-dimensional figure composed of unit cubes -/
structure CubeFigure where
  num_cubes : ℕ
  cube_side_length : ℝ
  top_area : ℝ
  bottom_area : ℝ
  front_area : ℝ
  back_area : ℝ
  left_area : ℝ
  right_area : ℝ

/-- The surface area of a CubeFigure -/
def surface_area (figure : CubeFigure) : ℝ :=
  figure.top_area + figure.bottom_area + figure.front_area + 
  figure.back_area + figure.left_area + figure.right_area

/-- Theorem stating that the surface area of the given figure is 34 -/
theorem surface_area_is_34 (figure : CubeFigure) 
  (h1 : figure.num_cubes = 10)
  (h2 : figure.cube_side_length = 1)
  (h3 : figure.top_area = 6)
  (h4 : figure.bottom_area = 6)
  (h5 : figure.front_area = 5)
  (h6 : figure.back_area = 5)
  (h7 : figure.left_area = 6)
  (h8 : figure.right_area = 6) :
  surface_area figure = 34 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_is_34_l4114_411470


namespace NUMINAMATH_CALUDE_main_diagonal_squares_diagonal_5_composite_diagonal_21_composite_l4114_411419

def a (k : ℕ) : ℕ := (2*k + 1)^2

def b (k : ℕ) : ℕ := (4*k - 3) * (4*k + 1)

def c (k : ℕ) : ℕ := 4*((4*k + 3)*(4*k - 1)) + 1

theorem main_diagonal_squares (k : ℕ) :
  ∃ (n : ℕ), a k = 4*n + 1 :=
sorry

theorem diagonal_5_composite (k : ℕ) (h : k > 1) :
  ¬ Nat.Prime (b k) :=
sorry

theorem diagonal_21_composite (k : ℕ) :
  ¬ Nat.Prime (c k) :=
sorry

end NUMINAMATH_CALUDE_main_diagonal_squares_diagonal_5_composite_diagonal_21_composite_l4114_411419


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4114_411456

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  h1 : a 7 = 1  -- 7th term is 1
  h2 : S 4 = -32  -- Sum of first 4 terms is -32
  h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Constant difference property

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n - 13) ∧
  (∀ n : ℕ, seq.S n = (n - 6)^2 - 36) ∧
  (∀ n : ℕ, seq.S n ≥ -36) ∧
  (∃ n : ℕ, seq.S n = -36) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4114_411456


namespace NUMINAMATH_CALUDE_simplify_expression_l4114_411461

theorem simplify_expression (m : ℝ) : m^2 - m*(m-3) = 3*m := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4114_411461


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l4114_411490

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 30 * x + c = 0) → 
  (a + c = 35) →
  (a < c) →
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l4114_411490


namespace NUMINAMATH_CALUDE_fair_coin_expectation_l4114_411471

/-- A fair coin is a coin with probability 1/2 for both heads and tails -/
def fairCoin (p : ℝ) : Prop := p = 1/2

/-- The expected value of heads for a single toss of a fair coin -/
def expectedValueSingleToss (p : ℝ) (h : fairCoin p) : ℝ := p

/-- The number of tosses -/
def numTosses : ℕ := 5

/-- The mathematical expectation of heads for multiple tosses of a fair coin -/
def expectedValueMultipleTosses (p : ℝ) (h : fairCoin p) : ℝ :=
  (expectedValueSingleToss p h) * numTosses

theorem fair_coin_expectation (p : ℝ) (h : fairCoin p) :
  expectedValueMultipleTosses p h = 5/2 := by sorry

end NUMINAMATH_CALUDE_fair_coin_expectation_l4114_411471


namespace NUMINAMATH_CALUDE_trapezoid_ed_length_l4114_411434

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  /-- Length of base AB -/
  base : ℝ
  /-- Length of top base CD -/
  top_base : ℝ
  /-- Length of non-parallel sides AD and BC -/
  side : ℝ
  /-- E is the midpoint of diagonal AC -/
  e_midpoint : Bool
  /-- AED is a right triangle -/
  aed_right : Bool
  /-- D lies on extended line segment AE -/
  d_on_ae : Bool

/-- Theorem stating the length of ED in the given trapezoid -/
theorem trapezoid_ed_length (t : Trapezoid) 
  (h1 : t.base = 8) 
  (h2 : t.top_base = 6) 
  (h3 : t.side = 5) 
  (h4 : t.e_midpoint) 
  (h5 : t.aed_right) 
  (h6 : t.d_on_ae) : 
  ∃ (ed : ℝ), ed = Real.sqrt 6.5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ed_length_l4114_411434


namespace NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l4114_411451

theorem angle_with_supplement_four_times_complement (x : ℝ) :
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l4114_411451


namespace NUMINAMATH_CALUDE_triangle_shape_l4114_411424

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition
def satisfiesCondition (t : Triangle) : Prop :=
  (Real.cos t.A) / (Real.cos t.B) = t.b / t.a

-- Define isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

-- Define right triangle
def isRight (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

-- Theorem statement
theorem triangle_shape (t : Triangle) :
  satisfiesCondition t → isIsosceles t ∨ isRight t :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_shape_l4114_411424


namespace NUMINAMATH_CALUDE_range_of_a_l4114_411468

-- Define the set of valid values for a
def ValidA : Set ℝ :=
  {x | x > -1 ∧ x ≠ -5/6 ∧ x ≠ (1 + Real.sqrt 21) / 4 ∧ x ≠ (1 - Real.sqrt 21) / 4 ∧ x ≠ -7/8}

-- State the theorem
theorem range_of_a (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (eq1 : b^2 + c^2 = 2*a^2 + 16*a + 14) 
  (eq2 : b*c = a^2 - 4*a - 5) : 
  a ∈ ValidA :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4114_411468


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l4114_411478

theorem units_digit_of_7_power_2023 : (7^2023) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l4114_411478


namespace NUMINAMATH_CALUDE_f_has_extrema_l4114_411438

/-- The function f(x) = 2 - x^2 - x^3 -/
def f (x : ℝ) : ℝ := 2 - x^2 - x^3

/-- Theorem stating that f has both a maximum and a minimum value -/
theorem f_has_extrema : 
  (∃ a : ℝ, ∀ x : ℝ, f x ≤ f a) ∧ (∃ b : ℝ, ∀ x : ℝ, f x ≥ f b) :=
sorry

end NUMINAMATH_CALUDE_f_has_extrema_l4114_411438


namespace NUMINAMATH_CALUDE_black_raisins_amount_l4114_411498

-- Define the variables
def yellow_raisins : ℝ := 0.3
def total_raisins : ℝ := 0.7

-- Define the theorem
theorem black_raisins_amount :
  total_raisins - yellow_raisins = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_black_raisins_amount_l4114_411498


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l4114_411433

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 15 → difference = 1 → friend_cost = (total + difference) / 2 → friend_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l4114_411433


namespace NUMINAMATH_CALUDE_fraction_of_complex_l4114_411420

def complex_i : ℂ := Complex.I

theorem fraction_of_complex (z : ℂ) (h : z = 1 + complex_i) : 2 / z = 1 - complex_i := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_complex_l4114_411420


namespace NUMINAMATH_CALUDE_wendy_trip_miles_l4114_411403

theorem wendy_trip_miles (total_miles second_day_miles first_day_miles third_day_miles : ℕ) :
  total_miles = 493 →
  first_day_miles = 125 →
  third_day_miles = 145 →
  second_day_miles = total_miles - first_day_miles - third_day_miles →
  second_day_miles = 223 := by
sorry

end NUMINAMATH_CALUDE_wendy_trip_miles_l4114_411403


namespace NUMINAMATH_CALUDE_kitten_weight_l4114_411425

/-- The weight of a kitten and two dogs satisfying certain conditions -/
structure AnimalWeights where
  kitten : ℝ
  smallDog : ℝ
  largeDog : ℝ
  total_weight : kitten + smallDog + largeDog = 36
  larger_pair : kitten + largeDog = 2 * smallDog
  smaller_pair : kitten + smallDog = largeDog

/-- The kitten's weight is 6 pounds given the conditions -/
theorem kitten_weight (w : AnimalWeights) : w.kitten = 6 := by
  sorry

end NUMINAMATH_CALUDE_kitten_weight_l4114_411425


namespace NUMINAMATH_CALUDE_cos_37_cos_23_minus_sin_37_sin_23_l4114_411437

theorem cos_37_cos_23_minus_sin_37_sin_23 :
  Real.cos (37 * π / 180) * Real.cos (23 * π / 180) - 
  Real.sin (37 * π / 180) * Real.sin (23 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_37_cos_23_minus_sin_37_sin_23_l4114_411437


namespace NUMINAMATH_CALUDE_max_weighing_ways_exact_89_ways_weighing_theorem_l4114_411479

/-- Represents the set of weights with masses 1, 2, 4, ..., 512 grams -/
def WeightSet : Set ℕ := {n : ℕ | ∃ k : ℕ, k ≤ 9 ∧ n = 2^k}

/-- Number of ways to weigh a load P using weights up to 2^n -/
def K (n : ℕ) (P : ℤ) : ℕ := sorry

/-- Maximum number of ways to weigh any load using weights up to 2^n -/
def MaxK (n : ℕ) : ℕ := sorry

/-- Theorem stating that no load can be weighed in more than 89 ways -/
theorem max_weighing_ways :
  ∀ P : ℤ, K 9 P ≤ 89 :=
sorry

/-- Theorem stating that 171 grams can be weighed in exactly 89 ways -/
theorem exact_89_ways :
  K 9 171 = 89 :=
sorry

/-- Main theorem combining both parts of the problem -/
theorem weighing_theorem :
  (∀ P : ℤ, K 9 P ≤ 89) ∧ (K 9 171 = 89) :=
sorry

end NUMINAMATH_CALUDE_max_weighing_ways_exact_89_ways_weighing_theorem_l4114_411479


namespace NUMINAMATH_CALUDE_negation_of_existential_absolute_value_l4114_411439

theorem negation_of_existential_absolute_value (x : ℝ) :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_absolute_value_l4114_411439


namespace NUMINAMATH_CALUDE_expression_evaluation_l4114_411446

theorem expression_evaluation : 5 * 7 + 9 * 4 - (15 / 3)^2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4114_411446


namespace NUMINAMATH_CALUDE_louie_junior_took_seven_cookies_l4114_411480

/-- Represents the number of cookies in various states --/
structure CookieJar where
  initial : Nat
  eatenByLouSenior : Nat
  remaining : Nat

/-- Calculates the number of cookies Louie Junior took --/
def cookiesTakenByLouieJunior (jar : CookieJar) : Nat :=
  jar.initial - jar.eatenByLouSenior - jar.remaining

/-- Theorem stating that Louie Junior took 7 cookies --/
theorem louie_junior_took_seven_cookies (jar : CookieJar) 
  (h1 : jar.initial = 22)
  (h2 : jar.eatenByLouSenior = 4)
  (h3 : jar.remaining = 11) :
  cookiesTakenByLouieJunior jar = 7 := by
  sorry

#eval cookiesTakenByLouieJunior { initial := 22, eatenByLouSenior := 4, remaining := 11 }

end NUMINAMATH_CALUDE_louie_junior_took_seven_cookies_l4114_411480


namespace NUMINAMATH_CALUDE_frog_population_equality_l4114_411410

theorem frog_population_equality : ∃ n : ℕ, n > 0 ∧ n = 6 ∧ ∀ m : ℕ, m > 0 → (5^(m+1) = 243 * 3^m → m ≥ n) := by
  sorry

end NUMINAMATH_CALUDE_frog_population_equality_l4114_411410


namespace NUMINAMATH_CALUDE_grandma_backpacks_l4114_411462

def backpack_problem (original_price : ℝ) (discount_rate : ℝ) (monogram_cost : ℝ) (total_cost : ℝ) : Prop :=
  let discounted_price := original_price * (1 - discount_rate)
  let final_price := discounted_price + monogram_cost
  let num_grandchildren := total_cost / final_price
  num_grandchildren = 5

theorem grandma_backpacks :
  backpack_problem 20 0.2 12 140 := by
  sorry

end NUMINAMATH_CALUDE_grandma_backpacks_l4114_411462


namespace NUMINAMATH_CALUDE_ten_percent_relation_l4114_411418

/-- If 10% of s is equal to t, then s equals 10t -/
theorem ten_percent_relation (s t : ℝ) (h : (10 : ℝ) / 100 * s = t) : s = 10 * t := by
  sorry

end NUMINAMATH_CALUDE_ten_percent_relation_l4114_411418


namespace NUMINAMATH_CALUDE_ball_return_to_start_l4114_411499

def ball_throw (n : ℕ) : ℕ → ℕ := λ x => (x + 3) % n

theorem ball_return_to_start :
  ∀ (start : ℕ), start < 13 →
  ∃ (k : ℕ), k > 0 ∧ (Nat.iterate (ball_throw 13) k start) = start ∧
  k = 13 :=
sorry

end NUMINAMATH_CALUDE_ball_return_to_start_l4114_411499


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_seven_thirds_l4114_411411

theorem ceiling_floor_sum_seven_thirds : ⌈(-7 : ℚ) / 3⌉ + ⌊(7 : ℚ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_seven_thirds_l4114_411411


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l4114_411464

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  -- Length of edge AB
  ab_length : ℝ
  -- Length of edge CD
  cd_length : ℝ
  -- Distance between lines AB and CD
  line_distance : ℝ
  -- Angle between lines AB and CD
  line_angle : ℝ

/-- Calculates the volume of the tetrahedron -/
def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 1/2 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    ab_length := 1,
    cd_length := Real.sqrt 3,
    line_distance := 2,
    line_angle := π / 3
  }
  tetrahedron_volume t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l4114_411464


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l4114_411486

/-- Given a geometric sequence {aₙ} with a₁ = 3 and a₅ = 75, prove that a₃ = 15 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (h1 : a 1 = 3) (h5 : a 5 = 75) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) : 
  a 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l4114_411486


namespace NUMINAMATH_CALUDE_unbounded_solution_set_l4114_411408

/-- The set of points (x, y) satisfying the given system of inequalities is unbounded -/
theorem unbounded_solution_set :
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ 
      ((abs x + x)^2 + (abs y + y)^2 ≤ 4 ∧ 3*y + x ≤ 0)) ∧
    ¬(∃ (M : ℝ), ∀ (p : ℝ × ℝ), p ∈ S → ‖p‖ ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_unbounded_solution_set_l4114_411408


namespace NUMINAMATH_CALUDE_dans_initial_green_marbles_l4114_411477

/-- Represents the number of marbles Dan has -/
structure DanMarbles where
  initial_green : ℕ
  violet : ℕ
  taken_green : ℕ
  remaining_green : ℕ

/-- Theorem stating that Dan's initial number of green marbles is 32 -/
theorem dans_initial_green_marbles 
  (dan : DanMarbles)
  (h1 : dan.taken_green = 23)
  (h2 : dan.remaining_green = 9)
  (h3 : dan.initial_green = dan.taken_green + dan.remaining_green) :
  dan.initial_green = 32 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_green_marbles_l4114_411477


namespace NUMINAMATH_CALUDE_difference_of_squares_l4114_411481

theorem difference_of_squares (a b : ℝ) : (2*a - b) * (2*a + b) = 4*a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4114_411481


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l4114_411402

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = (3 * m - 2) * x^2 + 5) ↔ 
  (m ≤ 12 - 8 * Real.sqrt 2 ∨ m ≥ 12 + 8 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l4114_411402


namespace NUMINAMATH_CALUDE_average_of_last_three_l4114_411436

theorem average_of_last_three (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 62 →
  ((list.take 4).sum / 4 : ℝ) = 55 →
  ((list.drop 4).sum / 3 : ℝ) = 71 + 1/3 := by
sorry

end NUMINAMATH_CALUDE_average_of_last_three_l4114_411436


namespace NUMINAMATH_CALUDE_prob_same_color_is_45_128_l4114_411484

def blue_chips : ℕ := 7
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def total_chips : ℕ := blue_chips + red_chips + yellow_chips

def prob_same_color : ℚ :=
  (blue_chips^2 + red_chips^2 + yellow_chips^2) / total_chips^2

theorem prob_same_color_is_45_128 : prob_same_color = 45 / 128 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_45_128_l4114_411484


namespace NUMINAMATH_CALUDE_triangle_side_length_l4114_411442

/-- Given a triangle ABC with perimeter √2 + 1 and sin A + sin B = √2 sin C, 
    prove that the length of side AB is 1 -/
theorem triangle_side_length 
  (A B C : ℝ) 
  (perimeter : ℝ) 
  (h_perimeter : perimeter = Real.sqrt 2 + 1)
  (h_sin_sum : Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C)
  (h_triangle : A + B + C = π)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  : ∃ (a b c : ℝ), a + b + c = perimeter ∧ 
                    a = 1 ∧
                    a / Real.sin A = b / Real.sin B ∧
                    b / Real.sin B = c / Real.sin C :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4114_411442


namespace NUMINAMATH_CALUDE_fraction_simplification_l4114_411454

theorem fraction_simplification : (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4114_411454


namespace NUMINAMATH_CALUDE_exist_three_naturals_with_prime_sum_and_product_l4114_411472

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Theorem statement
theorem exist_three_naturals_with_prime_sum_and_product :
  ∃ a b c : ℕ, isPrime (a + b + c) ∧ isPrime (a * b * c) :=
sorry

end NUMINAMATH_CALUDE_exist_three_naturals_with_prime_sum_and_product_l4114_411472


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l4114_411475

/-- Given two lines that intersect at (2, 3), prove their y-intercepts sum to 10/3 -/
theorem intersection_y_intercept_sum (a b : ℚ) : 
  (2 = (1/3) * 3 + a) → 
  (3 = (1/3) * 2 + b) → 
  a + b = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l4114_411475


namespace NUMINAMATH_CALUDE_total_animals_on_yacht_l4114_411417

theorem total_animals_on_yacht (cows foxes zebras sheep : ℕ) : 
  cows = 20 → 
  foxes = 15 → 
  zebras = 3 * foxes → 
  sheep = 20 → 
  cows + foxes + zebras + sheep = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_on_yacht_l4114_411417


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_not_sufficient_nor_necessary_l4114_411428

/-- A sequence is geometric if the ratio between consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The statement "If {a_n} is geometric, then {a_n + a_{n+1}} is geometric" is neither sufficient nor necessary. -/
theorem geometric_sequence_sum_not_sufficient_nor_necessary :
  (∃ a : ℕ → ℝ, IsGeometric a ∧ ¬IsGeometric (fun n ↦ a n + a (n + 1))) ∧
  (∃ a : ℕ → ℝ, ¬IsGeometric a ∧ IsGeometric (fun n ↦ a n + a (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_not_sufficient_nor_necessary_l4114_411428


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_pi_6_l4114_411447

theorem sin_2alpha_plus_pi_6 (α : Real) (h : Real.cos (α - π / 6) = 1 / 3) :
  Real.sin (2 * α + π / 6) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_pi_6_l4114_411447


namespace NUMINAMATH_CALUDE_largest_four_digit_square_base_7_l4114_411457

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def N : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- Number of digits in the base 7 representation of a natural number -/
def num_digits_base_7 (n : ℕ) : ℕ :=
  (to_base_7 n).length

/-- Theorem stating that N is the largest integer whose square has exactly 4 digits in base 7 -/
theorem largest_four_digit_square_base_7 :
  (∀ m : ℕ, m > N → num_digits_base_7 (m^2) > 4) ∧
  num_digits_base_7 (N^2) = 4 ∧
  to_base_7 N = [6, 6] :=
sorry

#eval N
#eval to_base_7 N
#eval num_digits_base_7 (N^2)

end NUMINAMATH_CALUDE_largest_four_digit_square_base_7_l4114_411457


namespace NUMINAMATH_CALUDE_average_weight_proof_l4114_411469

theorem average_weight_proof (a b c : ℝ) :
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 33 →
  (a + b) / 2 = 41 := by
sorry

end NUMINAMATH_CALUDE_average_weight_proof_l4114_411469


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l4114_411497

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isLuckyInteger (n : ℕ) : Prop :=
  n > 0 ∧ n % sumOfDigits n = 0

def isMultipleOf7 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

theorem least_non_lucky_multiple_of_7 :
  (∀ n : ℕ, n > 0 ∧ n < 14 ∧ isMultipleOf7 n → isLuckyInteger n) ∧
  isMultipleOf7 14 ∧
  ¬isLuckyInteger 14 :=
sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l4114_411497


namespace NUMINAMATH_CALUDE_personal_income_tax_example_l4114_411494

/-- Calculate the personal income tax for a citizen given their salary and prize information --/
def personal_income_tax (salary_jan_jun : ℕ) (salary_jul_dec : ℕ) (prize_value : ℕ) : ℕ :=
  let salary_tax_rate : ℚ := 13 / 100
  let prize_tax_rate : ℚ := 35 / 100
  let non_taxable_prize : ℕ := 4000
  let total_salary : ℕ := salary_jan_jun * 6 + salary_jul_dec * 6
  let salary_tax : ℕ := (total_salary * salary_tax_rate).floor.toNat
  let taxable_prize : ℕ := max (prize_value - non_taxable_prize) 0
  let prize_tax : ℕ := (taxable_prize * prize_tax_rate).floor.toNat
  salary_tax + prize_tax

/-- Theorem stating that the personal income tax for the given scenario is 39540 rubles --/
theorem personal_income_tax_example : 
  personal_income_tax 23000 25000 10000 = 39540 := by
  sorry

end NUMINAMATH_CALUDE_personal_income_tax_example_l4114_411494


namespace NUMINAMATH_CALUDE_trigonometric_sum_divisibility_l4114_411429

theorem trigonometric_sum_divisibility (n : ℕ) :
  ∃ k : ℤ, (2 * Real.sin (π / 7 : ℝ))^(2*n) + 
           (2 * Real.sin (2*π / 7 : ℝ))^(2*n) + 
           (2 * Real.sin (3*π / 7 : ℝ))^(2*n) = 
           k * (7 : ℝ)^(Int.floor (n / 3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_divisibility_l4114_411429


namespace NUMINAMATH_CALUDE_inequality_solution_l4114_411427

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 1) ↔ (x > -8 ∧ x < -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4114_411427


namespace NUMINAMATH_CALUDE_triangle_with_altitudes_is_obtuse_l4114_411401

/-- A triangle with given altitudes is obtuse -/
theorem triangle_with_altitudes_is_obtuse (h_a h_b h_c : ℝ) 
  (h_alt_a : h_a = 1/14)
  (h_alt_b : h_b = 1/10)
  (h_alt_c : h_c = 1/5)
  (h_positive_a : h_a > 0)
  (h_positive_b : h_b > 0)
  (h_positive_c : h_c > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ a + c > b ∧
    a * h_a = b * h_b ∧ b * h_b = c * h_c ∧
    (b^2 + c^2 - a^2) / (2 * b * c) < 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_altitudes_is_obtuse_l4114_411401


namespace NUMINAMATH_CALUDE_inequality_chain_l4114_411495

theorem inequality_chain (x : ℝ) 
  (h1 : 0 < x) (h2 : x < 1) 
  (a b c : ℝ) 
  (ha : a = x^2) 
  (hb : b = 1/x) 
  (hc : c = Real.sqrt x) : 
  b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_inequality_chain_l4114_411495


namespace NUMINAMATH_CALUDE_student_count_l4114_411492

theorem student_count (bags : ℕ) (nuts_per_bag : ℕ) (nuts_per_student : ℕ) : 
  bags = 65 → nuts_per_bag = 15 → nuts_per_student = 75 → 
  (bags * nuts_per_bag) / nuts_per_student = 13 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l4114_411492


namespace NUMINAMATH_CALUDE_log_inequality_l4114_411444

theorem log_inequality (x : ℝ) : 
  0 < x → x < 4 → (Real.log x / Real.log 9 ≥ (Real.log (Real.sqrt (1 - x / 4)) / Real.log 3)^2 ↔ x = 2 ∨ (4/5 ≤ x ∧ x < 4)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l4114_411444


namespace NUMINAMATH_CALUDE_wheel_diameter_l4114_411445

/-- The diameter of a wheel given its revolutions and distance covered -/
theorem wheel_diameter (revolutions : ℝ) (distance : ℝ) (π : ℝ) :
  revolutions = 8.007279344858963 →
  distance = 1056 →
  π = 3.14159 →
  ∃ (diameter : ℝ), abs (diameter - 41.975) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_wheel_diameter_l4114_411445


namespace NUMINAMATH_CALUDE_tower_arrangements_l4114_411416

def num_red_cubes : ℕ := 2
def num_blue_cubes : ℕ := 3
def num_green_cubes : ℕ := 4
def tower_height : ℕ := 8

theorem tower_arrangements :
  (Nat.choose (num_red_cubes + num_blue_cubes + num_green_cubes) tower_height *
   Nat.factorial tower_height) /
  (Nat.factorial num_red_cubes * Nat.factorial num_blue_cubes * Nat.factorial num_green_cubes) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_tower_arrangements_l4114_411416


namespace NUMINAMATH_CALUDE_shaded_area_between_squares_l4114_411406

/-- The area of the shaded region between two squares -/
theorem shaded_area_between_squares (large_side small_side : ℝ) 
  (h_large : large_side = 9)
  (h_small : small_side = 4) :
  large_side ^ 2 - small_side ^ 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_squares_l4114_411406


namespace NUMINAMATH_CALUDE_fraction_invariance_l4114_411405

theorem fraction_invariance (x y : ℝ) (hx : x ≠ 0) : (y + x) / x = (3*y + 3*x) / (3*x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_invariance_l4114_411405


namespace NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l4114_411422

theorem min_sum_given_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 2) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1 / x + 2 / y = 2 → a + b ≤ x + y ∧ 
  (a + b = (3 + 2 * Real.sqrt 2) / 2 ↔ a + b = x + y) :=
sorry

end NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l4114_411422


namespace NUMINAMATH_CALUDE_unique_n_divisibility_l4114_411441

theorem unique_n_divisibility : ∃! (n : ℕ), n > 1 ∧
  ∀ (p : ℕ), Prime p → (p ∣ (n^6 - 1)) → (p ∣ ((n^3 - 1) * (n^2 - 1))) :=
by
  -- The unique n that satisfies the condition is 2
  use 2
  sorry

end NUMINAMATH_CALUDE_unique_n_divisibility_l4114_411441


namespace NUMINAMATH_CALUDE_least_number_of_cookies_l4114_411467

theorem least_number_of_cookies (n : ℕ) : n ≥ 208 →
  (n % 6 = 4 ∧ n % 5 = 3 ∧ n % 8 = 6 ∧ n % 9 = 7) →
  n = 208 :=
sorry

end NUMINAMATH_CALUDE_least_number_of_cookies_l4114_411467


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l4114_411452

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l4114_411452


namespace NUMINAMATH_CALUDE_distance_to_line_l4114_411426

/-- Given two perpendicular lines and a plane, calculate the distance from a point to one of the lines -/
theorem distance_to_line (m θ ψ : ℝ) (hm : m > 0) (hθ : 0 < θ ∧ θ < π / 2) (hψ : 0 < ψ ∧ ψ < π / 2) :
  ∃ (d : ℝ), d = Real.sqrt (m^2 + (m * Real.sin θ / Real.sin ψ)^2) ∧ d ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_l4114_411426


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l4114_411430

/-- Given a quadrilateral with one diagonal of 50 cm, one offset of 8 cm, and an area of 450 cm²,
    the length of the other offset is 10 cm. -/
theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 50 ∧ offset1 = 8 ∧ area = 450 →
  ∃ offset2 : ℝ, offset2 = 10 ∧ area = (diagonal * (offset1 + offset2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l4114_411430


namespace NUMINAMATH_CALUDE_additional_sticks_needed_l4114_411455

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the problem setup -/
structure ProblemSetup where
  large_rectangle : Rectangle
  total_sticks : ℕ
  num_small_rectangles : ℕ
  small_rectangle_types : List Rectangle

/-- The main theorem statement -/
theorem additional_sticks_needed 
  (setup : ProblemSetup)
  (h1 : setup.large_rectangle = ⟨8, 12⟩)
  (h2 : setup.total_sticks = 40)
  (h3 : setup.num_small_rectangles = 40)
  (h4 : setup.small_rectangle_types = [⟨1, 2⟩, ⟨1, 3⟩])
  : ∃ (additional_sticks : ℕ), additional_sticks = 116 ∧
    ∃ (small_rectangles : List Rectangle),
      small_rectangles.length = setup.num_small_rectangles ∧
      (∀ r ∈ small_rectangles, r ∈ setup.small_rectangle_types) ∧
      (small_rectangles.map (λ r => r.width * r.height)).sum = 
        setup.large_rectangle.width * setup.large_rectangle.height :=
by
  sorry


end NUMINAMATH_CALUDE_additional_sticks_needed_l4114_411455


namespace NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l4114_411453

/-- A function is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (fun x ↦ f x + g x)) ∧
  (∃ f g : ℝ → ℝ, ¬(IsEven f ∧ IsEven g) ∧ IsEven (fun x ↦ f x + g x)) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l4114_411453


namespace NUMINAMATH_CALUDE_hcf_of_numbers_l4114_411463

theorem hcf_of_numbers (x y : ℕ+) 
  (sum_eq : x + y = 45)
  (lcm_eq : Nat.lcm x y = 120)
  (sum_recip_eq : (1 : ℚ) / x + (1 : ℚ) / y = 11 / 120) :
  Nat.gcd x y = 1 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_numbers_l4114_411463


namespace NUMINAMATH_CALUDE_tom_gave_cars_to_five_nephews_l4114_411440

/-- The number of nephews Tom gave cars to -/
def number_of_nephews : ℕ := by sorry

theorem tom_gave_cars_to_five_nephews :
  let packages := 10
  let cars_per_package := 5
  let total_cars := packages * cars_per_package
  let cars_left := 30
  let cars_given_away := total_cars - cars_left
  let fraction_per_nephew := 1 / 5
  number_of_nephews = (cars_given_away : ℚ) / (fraction_per_nephew * cars_given_away) := by sorry

end NUMINAMATH_CALUDE_tom_gave_cars_to_five_nephews_l4114_411440


namespace NUMINAMATH_CALUDE_same_height_antonio_maria_l4114_411491

-- Define the type for height comparisons
inductive HeightComparison
  | Taller : HeightComparison
  | Shorter : HeightComparison
  | Same : HeightComparison

-- Define the siblings
inductive Sibling
  | Luiza : Sibling
  | Maria : Sibling
  | Antonio : Sibling
  | Julio : Sibling

-- Define the height comparison function
def compareHeight : Sibling → Sibling → HeightComparison := sorry

-- State the theorem
theorem same_height_antonio_maria :
  (compareHeight Sibling.Luiza Sibling.Antonio = HeightComparison.Taller) →
  (compareHeight Sibling.Antonio Sibling.Julio = HeightComparison.Taller) →
  (compareHeight Sibling.Maria Sibling.Luiza = HeightComparison.Shorter) →
  (compareHeight Sibling.Julio Sibling.Maria = HeightComparison.Shorter) →
  (compareHeight Sibling.Antonio Sibling.Maria = HeightComparison.Same) :=
by
  sorry

end NUMINAMATH_CALUDE_same_height_antonio_maria_l4114_411491


namespace NUMINAMATH_CALUDE_current_calculation_l4114_411493

-- Define the variables and their types
variable (Q I R t : ℝ)

-- Define the theorem
theorem current_calculation 
  (heat_equation : Q = I^2 * R * t)
  (resistance : R = 5)
  (heat_generated : Q = 30)
  (time : t = 1) :
  I = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_current_calculation_l4114_411493


namespace NUMINAMATH_CALUDE_expression_simplification_l4114_411404

theorem expression_simplification (x : ℝ) : 
  (x^3 - 2)^2 + (x^2 + 2*x)^2 = x^6 + x^4 + 4*x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4114_411404


namespace NUMINAMATH_CALUDE_c_invests_after_eight_months_l4114_411485

/-- Represents the investment scenario of three partners A, B, and C -/
structure Investment where
  /-- A's initial investment amount -/
  a_amount : ℝ
  /-- Number of months after which C invests -/
  c_invest_time : ℝ
  /-- Total annual gain -/
  total_gain : ℝ
  /-- A's share of the profit -/
  a_share : ℝ
  /-- B invests double A's amount after 6 months -/
  b_amount_eq : a_amount * 2 = a_amount
  /-- C invests triple A's amount -/
  c_amount_eq : a_amount * 3 = a_amount
  /-- Total annual gain is Rs. 18600 -/
  total_gain_eq : total_gain = 18600
  /-- A's share is Rs. 6200 -/
  a_share_eq : a_share = 6200
  /-- Profit share is proportional to investment and time -/
  profit_share_prop : a_share / total_gain = 
    (a_amount * 12) / (a_amount * 12 + a_amount * 2 * 6 + a_amount * 3 * (12 - c_invest_time))

/-- Theorem stating that C invests after 8 months -/
theorem c_invests_after_eight_months (i : Investment) : i.c_invest_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_c_invests_after_eight_months_l4114_411485


namespace NUMINAMATH_CALUDE_kats_required_score_l4114_411435

/-- Given Kat's first two test scores and desired average, calculate the required score on the third test --/
theorem kats_required_score (score1 score2 desired_avg : ℚ) (h1 : score1 = 95/100) (h2 : score2 = 80/100) (h3 : desired_avg = 90/100) :
  ∃ score3 : ℚ, (score1 + score2 + score3) / 3 ≥ desired_avg ∧ score3 = 95/100 :=
by sorry

end NUMINAMATH_CALUDE_kats_required_score_l4114_411435


namespace NUMINAMATH_CALUDE_line_intersecting_parabola_l4114_411443

/-- The equation of a line that intersects a parabola at two points 8 units apart vertically -/
theorem line_intersecting_parabola (m b : ℝ) (h1 : b ≠ 0) :
  (∃ k : ℝ, abs ((k^2 + 4*k + 4) - (m*k + b)) = 8) →
  (9 = 2*m + b) →
  (m = 2 ∧ b = 5) :=
by sorry

end NUMINAMATH_CALUDE_line_intersecting_parabola_l4114_411443


namespace NUMINAMATH_CALUDE_blue_candy_count_l4114_411409

theorem blue_candy_count (total : ℕ) (red : ℕ) (h1 : total = 3409) (h2 : red = 145) :
  total - red = 3264 := by
  sorry

end NUMINAMATH_CALUDE_blue_candy_count_l4114_411409


namespace NUMINAMATH_CALUDE_triangle_area_heron_l4114_411414

theorem triangle_area_heron (a b c : ℝ) (h_a : a = 6) (h_b : b = 8) (h_c : c = 10) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_heron_l4114_411414


namespace NUMINAMATH_CALUDE_right_triangle_identification_l4114_411448

theorem right_triangle_identification (a b c : ℝ) : 
  (a = 3 ∧ b = 4 ∧ c = 5) → 
  (a^2 + b^2 = c^2) ∧ 
  ¬(2^2 + 4^2 = 5^2) ∧ 
  ¬((Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2) ∧ 
  ¬(5^2 + 13^2 = 14^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l4114_411448


namespace NUMINAMATH_CALUDE_alice_weight_l4114_411489

theorem alice_weight (alice carol : ℝ) 
  (h1 : alice + carol = 200)
  (h2 : alice - carol = (1 / 3) * alice) : 
  alice = 120 := by
sorry

end NUMINAMATH_CALUDE_alice_weight_l4114_411489


namespace NUMINAMATH_CALUDE_ranges_of_a_and_b_l4114_411458

theorem ranges_of_a_and_b (a b : ℝ) (h : Real.sqrt (a^2 * b) = -a * Real.sqrt b) :
  b ≥ 0 ∧ 
  (b > 0 → a ≤ 0) ∧
  (b = 0 → ∀ x : ℝ, ∃ a : ℝ, Real.sqrt ((a : ℝ)^2 * 0) = -(a : ℝ) * Real.sqrt 0) :=
by sorry

end NUMINAMATH_CALUDE_ranges_of_a_and_b_l4114_411458


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l4114_411482

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan (x^3 - x^(3/2) * Real.sin (1 / (3*x)))
  else 0

theorem f_derivative_at_zero : 
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l4114_411482


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4114_411412

/-- An equilateral triangle with three inscribed circles -/
structure TriangleWithCircles where
  -- The side length of the equilateral triangle
  side : ℝ
  -- The radius of each inscribed circle
  radius : ℝ
  -- The offset from each vertex to the nearest point on any circle
  offset : ℝ
  -- Condition: The radius is 2
  h_radius : radius = 2
  -- Condition: The offset is 1
  h_offset : offset = 1
  -- Condition: The circles touch each other and the sides of the triangle
  h_touch : side = 2 * (radius + offset) + 2 * radius * Real.sqrt 3

/-- The perimeter of the triangle is 6√3 + 12 -/
theorem triangle_perimeter (t : TriangleWithCircles) : 
  3 * t.side = 6 * Real.sqrt 3 + 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4114_411412


namespace NUMINAMATH_CALUDE_girl_sums_equal_iff_n_odd_l4114_411487

/-- Represents the sum of a girl's card number and the numbers of adjacent boys' cards -/
def girlSum (n : ℕ) (i : ℕ) : ℕ :=
  (n + i) + (i % n + 1) + ((i + 1) % n + 1)

/-- Theorem stating that all girl sums are equal if and only if n is odd -/
theorem girl_sums_equal_iff_n_odd (n : ℕ) (h : n ≥ 3) :
  (∀ i j, i < n → j < n → girlSum n i = girlSum n j) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_girl_sums_equal_iff_n_odd_l4114_411487


namespace NUMINAMATH_CALUDE_same_color_marble_probability_l4114_411415

/-- The probability of drawing three marbles of the same color from a bag containing
    red, white, and blue marbles, without replacement. -/
theorem same_color_marble_probability
  (red : ℕ) (white : ℕ) (blue : ℕ)
  (h_red : red = 5)
  (h_white : white = 7)
  (h_blue : blue = 8) :
  let total := red + white + blue
  let p_red := (red * (red - 1) * (red - 2)) / (total * (total - 1) * (total - 2))
  let p_white := (white * (white - 1) * (white - 2)) / (total * (total - 1) * (total - 2))
  let p_blue := (blue * (blue - 1) * (blue - 2)) / (total * (total - 1) * (total - 2))
  p_red + p_white + p_blue = 101 / 1140 :=
by sorry


end NUMINAMATH_CALUDE_same_color_marble_probability_l4114_411415


namespace NUMINAMATH_CALUDE_unique_zero_point_between_consecutive_integers_l4114_411449

open Real

noncomputable def f (a x : ℝ) : ℝ := a * (x^2 + 2/x) - log x

theorem unique_zero_point_between_consecutive_integers (a : ℝ) (h : a > 0) :
  ∃ (x₀ m n : ℝ), 
    (∀ x ≠ x₀, f a x ≠ 0) ∧ 
    (f a x₀ = 0) ∧
    (m < x₀ ∧ x₀ < n) ∧
    (n = m + 1) ∧
    (m + n = 5) := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_point_between_consecutive_integers_l4114_411449


namespace NUMINAMATH_CALUDE_leisurely_morning_time_l4114_411466

/-- Represents the time taken for each part of Aiden's morning routine -/
structure MorningRoutine where
  prep : ℝ  -- Preparation time
  bus : ℝ   -- Bus ride time
  walk : ℝ  -- Walking time

/-- Calculates the total time for a given morning routine -/
def totalTime (r : MorningRoutine) : ℝ := r.prep + r.bus + r.walk

/-- Represents the conditions given in the problem -/
axiom typical_morning : ∃ r : MorningRoutine, totalTime r = 120

axiom rushed_morning : ∃ r : MorningRoutine, 
  0.5 * r.prep + 1.25 * r.bus + 0.5 * r.walk = 96

/-- Theorem stating the time taken on the leisurely morning -/
theorem leisurely_morning_time : 
  ∀ r : MorningRoutine, 
  totalTime r = 120 → 
  0.5 * r.prep + 1.25 * r.bus + 0.5 * r.walk = 96 → 
  1.25 * r.prep + 0.75 * r.bus + 1.25 * r.walk = 126 := by
  sorry

end NUMINAMATH_CALUDE_leisurely_morning_time_l4114_411466


namespace NUMINAMATH_CALUDE_carol_position_after_2304_moves_l4114_411450

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction in the hexagonal grid -/
inductive Direction
  | North
  | NorthEast
  | SouthEast
  | South
  | SouthWest
  | NorthWest

/-- Represents Carol's movement pattern -/
def carolPattern (cycle : ℕ) : List (Direction × ℕ) :=
  [(Direction.North, cycle + 1),
   (Direction.NorthEast, cycle + 1),
   (Direction.SouthEast, cycle + 2),
   (Direction.South, cycle + 2),
   (Direction.SouthWest, cycle + 3),
   (Direction.NorthWest, cycle + 3)]

/-- Calculates the total steps in a given number of cycles -/
def totalStepsInCycles (k : ℕ) : ℕ :=
  k * (k + 1) + 2 * ((k + 1) * (k + 2))

/-- Theorem: Carol's position after 2304 moves -/
theorem carol_position_after_2304_moves :
  ∃ (finalPos : Point),
    (finalPos.x = 5 * Real.sqrt 3 / 2) ∧
    (finalPos.y = 23.5) ∧
    (∃ (k : ℕ),
      totalStepsInCycles k ≤ 2304 ∧
      totalStepsInCycles (k + 1) > 2304 ∧
      finalPos = -- position after completing k cycles and remaining steps
        let remainingSteps := 2304 - totalStepsInCycles k
        let partialCycle := carolPattern (k + 1)
        -- logic to apply remaining steps using partialCycle
        sorry) := by
  sorry

end NUMINAMATH_CALUDE_carol_position_after_2304_moves_l4114_411450


namespace NUMINAMATH_CALUDE_total_pencils_l4114_411407

def pencils_problem (monday tuesday wednesday thursday friday : ℕ) : Prop :=
  monday = 35 ∧
  tuesday = 42 ∧
  wednesday = 3 * tuesday ∧
  thursday = wednesday / 2 ∧
  friday = 2 * monday

theorem total_pencils :
  ∀ monday tuesday wednesday thursday friday : ℕ,
    pencils_problem monday tuesday wednesday thursday friday →
    monday + tuesday + wednesday + thursday + friday = 336 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l4114_411407


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l4114_411431

theorem sum_of_squares_and_square_of_sum : (3 + 9)^2 + (3^2 + 9^2) = 234 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l4114_411431


namespace NUMINAMATH_CALUDE_candy_probability_l4114_411460

theorem candy_probability (p1 p2 : ℚ) : 
  (3/8 : ℚ) ≤ p1 ∧ p1 ≤ (2/5 : ℚ) ∧ 
  (3/8 : ℚ) ≤ p2 ∧ p2 ≤ (2/5 : ℚ) ∧ 
  p1 = (5/13 : ℚ) ∧ p2 = (7/18 : ℚ) →
  ((3/8 : ℚ) ≤ (5/13 : ℚ) ∧ (5/13 : ℚ) ≤ (2/5 : ℚ)) ∧
  ((3/8 : ℚ) ≤ (7/18 : ℚ) ∧ (7/18 : ℚ) ≤ (2/5 : ℚ)) ∧
  ¬((3/8 : ℚ) ≤ (17/40 : ℚ) ∧ (17/40 : ℚ) ≤ (2/5 : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_candy_probability_l4114_411460


namespace NUMINAMATH_CALUDE_bicycle_selling_price_l4114_411421

/-- Calculates the final selling price of a bicycle given the initial cost and profit percentages -/
theorem bicycle_selling_price (initial_cost : ℝ) (profit_a profit_b : ℝ) :
  initial_cost = 120 ∧ profit_a = 50 ∧ profit_b = 25 →
  initial_cost * (1 + profit_a / 100) * (1 + profit_b / 100) = 225 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_selling_price_l4114_411421


namespace NUMINAMATH_CALUDE_divisibleByTwo_infinite_lessThanBillion_finite_l4114_411473

-- Define the set of numbers divisible by 2
def divisibleByTwo : Set Int := {x | ∃ n : Int, x = 2 * n}

-- Define the set of positive integers less than 1 billion
def lessThanBillion : Set Nat := {x | x > 0 ∧ x < 1000000000}

-- Theorem 1: The set of numbers divisible by 2 is infinite
theorem divisibleByTwo_infinite : Set.Infinite divisibleByTwo := by
  sorry

-- Theorem 2: The set of positive integers less than 1 billion is finite
theorem lessThanBillion_finite : Set.Finite lessThanBillion := by
  sorry

end NUMINAMATH_CALUDE_divisibleByTwo_infinite_lessThanBillion_finite_l4114_411473


namespace NUMINAMATH_CALUDE_grasshopper_jump_l4114_411474

/-- The jumping contest between a grasshopper and a frog -/
theorem grasshopper_jump (frog_jump grasshopper_jump difference : ℕ) 
  (h1 : frog_jump = 39)
  (h2 : frog_jump = grasshopper_jump + difference)
  (h3 : difference = 22) :
  grasshopper_jump = 17 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jump_l4114_411474


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l4114_411465

theorem reciprocal_sum_theorem (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 5 * x * y) :
  1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l4114_411465
