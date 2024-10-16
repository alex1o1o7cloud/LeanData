import Mathlib

namespace NUMINAMATH_CALUDE_bookstore_max_revenue_l1531_153124

/-- The revenue function for the bookstore -/
def R (p : ℝ) : ℝ := p * (200 - 8 * p)

/-- The theorem stating the maximum revenue and optimal price -/
theorem bookstore_max_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 25 ∧
  R p = 1250 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 25 → R q ≤ R p ∧
  p = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_max_revenue_l1531_153124


namespace NUMINAMATH_CALUDE_bus_ride_cost_l1531_153112

-- Define the cost of bus and train rides
def bus_cost : ℝ := 1.75
def train_cost : ℝ := bus_cost + 6.35

-- State the theorem
theorem bus_ride_cost : 
  (train_cost = bus_cost + 6.35) → 
  (train_cost + bus_cost = 9.85) → 
  (bus_cost = 1.75) :=
by
  sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l1531_153112


namespace NUMINAMATH_CALUDE_train_platform_ratio_l1531_153172

/-- Given a train of length L traveling at constant velocity v,
    if it passes a pole in time t and a platform in time 4t,
    then the ratio of the platform length P to the train length L is 3:1 -/
theorem train_platform_ratio
  (L : ℝ) -- Length of the train
  (v : ℝ) -- Velocity of the train
  (t : ℝ) -- Time to pass the pole
  (P : ℝ) -- Length of the platform
  (h1 : v > 0) -- Velocity is positive
  (h2 : L > 0) -- Train length is positive
  (h3 : t > 0) -- Time is positive
  (h4 : v = L / t) -- Velocity equation for passing the pole
  (h5 : v = (L + P) / (4 * t)) -- Velocity equation for passing the platform
  : P / L = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_ratio_l1531_153172


namespace NUMINAMATH_CALUDE_parallelogram45_diag_product_l1531_153163

/-- A parallelogram with one angle of 45° -/
structure Parallelogram45 where
  a : ℝ
  b : ℝ
  d₁ : ℝ
  d₂ : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_diag₁ : d₁^2 = a^2 + b^2 + Real.sqrt 2 * a * b
  h_diag₂ : d₂^2 = a^2 + b^2 - Real.sqrt 2 * a * b

/-- The product of squared diagonals equals the sum of fourth powers of sides -/
theorem parallelogram45_diag_product (p : Parallelogram45) :
    p.d₁^2 * p.d₂^2 = p.a^4 + p.b^4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram45_diag_product_l1531_153163


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1531_153197

-- Define the quadratic function f(x) = ax^2 + bx + c
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) :
  (∀ x, ax^2 + bx + c > 0 ↔ -2 < x ∧ x < 4) →
  f a b c 5 < f a b c (-1) ∧ f a b c (-1) < f a b c 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1531_153197


namespace NUMINAMATH_CALUDE_odd_binomial_coefficients_in_pascal_triangle_l1531_153131

theorem odd_binomial_coefficients_in_pascal_triangle (n : ℕ) :
  (Finset.sum (Finset.range (2^n)) (λ u =>
    (Finset.sum (Finset.range (u + 1)) (λ v =>
      if Nat.choose u v % 2 = 1 then 1 else 0
    ))
  )) = 3^n :=
sorry

end NUMINAMATH_CALUDE_odd_binomial_coefficients_in_pascal_triangle_l1531_153131


namespace NUMINAMATH_CALUDE_product_mod_400_l1531_153100

theorem product_mod_400 : (1567 * 2150) % 400 = 50 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_400_l1531_153100


namespace NUMINAMATH_CALUDE_student_grade_problem_l1531_153162

theorem student_grade_problem (courses_last_year : ℕ) (courses_year_before : ℕ) 
  (avg_grade_year_before : ℚ) (avg_grade_two_years : ℚ) :
  courses_last_year = 6 →
  courses_year_before = 5 →
  avg_grade_year_before = 40 →
  avg_grade_two_years = 72 →
  (courses_year_before * avg_grade_year_before + 
   courses_last_year * (592 : ℚ) / 6) / (courses_year_before + courses_last_year) = 
  avg_grade_two_years :=
by sorry

end NUMINAMATH_CALUDE_student_grade_problem_l1531_153162


namespace NUMINAMATH_CALUDE_expression_value_l1531_153153

theorem expression_value (m : ℝ) (h : m^2 + m - 1 = 0) :
  2 / (m^2 + m) - (m + 2) / (m^2 + 2*m + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1531_153153


namespace NUMINAMATH_CALUDE_sachins_age_l1531_153189

theorem sachins_age (sachin_age rahul_age : ℕ) : 
  rahul_age = sachin_age + 14 →
  sachin_age * 9 = rahul_age * 7 →
  sachin_age = 49 := by
sorry

end NUMINAMATH_CALUDE_sachins_age_l1531_153189


namespace NUMINAMATH_CALUDE_age_difference_l1531_153152

/-- Given the ages of four individuals x, y, z, and w, prove that z is 1.2 decades younger than x. -/
theorem age_difference (x y z w : ℕ) : 
  (x + y = y + z + 12) → 
  (x + y + w = y + z + w + 12) → 
  (x : ℚ) - z = 12 ∧ (x - z : ℚ) / 10 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1531_153152


namespace NUMINAMATH_CALUDE_sand_remaining_proof_l1531_153187

/-- The amount of sand remaining on a truck after transit -/
def sand_remaining (initial : ℝ) (lost : ℝ) : ℝ :=
  initial - lost

/-- Theorem: The amount of sand remaining on the truck is 1.7 pounds -/
theorem sand_remaining_proof (initial : ℝ) (lost : ℝ) 
    (h1 : initial = 4.1)
    (h2 : lost = 2.4) : 
  sand_remaining initial lost = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_sand_remaining_proof_l1531_153187


namespace NUMINAMATH_CALUDE_horner_v3_equals_16_l1531_153148

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^6 - 5x^4 + 2x^3 - x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ :=
  3 * x^6 - 5 * x^4 + 2 * x^3 - x^2 + 2 * x + 1

/-- v3 in Horner's method for f(x) -/
def v3 (x : ℝ) : ℝ :=
  (((3 * x - 0) * x - 5) * x + 2)

theorem horner_v3_equals_16 :
  v3 2 = 16 := by sorry

end NUMINAMATH_CALUDE_horner_v3_equals_16_l1531_153148


namespace NUMINAMATH_CALUDE_A_value_l1531_153191

noncomputable def A (x y : ℝ) : ℝ :=
  (Real.sqrt (x^3 + 2*x^2*y) + Real.sqrt (x^4 + 2 - x^3) - (x^(3/2) + x^2)) /
  (Real.sqrt (2*(x + y - Real.sqrt (x^2 + 2*x*y))) * (x^(2/3) - x^(5/6) + x))

theorem A_value (x y : ℝ) (hx : x > 0) :
  (y > 0 → A x y = x^(1/3) + x^(1/2)) ∧
  (-x/2 ≤ y ∧ y < 0 → A x y = -(x^(1/3) + x^(1/2))) :=
by sorry

end NUMINAMATH_CALUDE_A_value_l1531_153191


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1531_153159

-- Define a geometric sequence
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometric a →
  a 1 + a 2 + a 3 = 7 →
  a 2 + a 3 + a 4 = 14 →
  a 4 + a 5 + a 6 = 56 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1531_153159


namespace NUMINAMATH_CALUDE_successive_integers_product_l1531_153102

theorem successive_integers_product (n : ℤ) : n * (n + 1) = 4160 → n = 64 := by
  sorry

end NUMINAMATH_CALUDE_successive_integers_product_l1531_153102


namespace NUMINAMATH_CALUDE_product_of_solutions_l1531_153165

theorem product_of_solutions (x₁ x₂ : ℝ) (h₁ : x₁ * Real.exp x₁ = Real.exp 2) (h₂ : x₂ * Real.log x₂ = Real.exp 2) :
  x₁ * x₂ = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l1531_153165


namespace NUMINAMATH_CALUDE_four_weighings_sufficient_l1531_153179

/-- Represents the weight of a coin in grams -/
inductive CoinWeight
  | One
  | Two
  | Three
  | Four

/-- Represents the result of a weighing -/
inductive WeighingResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing action -/
def Weighing := (List CoinWeight) → (List CoinWeight) → WeighingResult

/-- The set of four coins with weights 1, 2, 3, and 4 grams -/
def CoinSet : Set CoinWeight := {CoinWeight.One, CoinWeight.Two, CoinWeight.Three, CoinWeight.Four}

/-- A strategy is a sequence of weighings -/
def Strategy := List Weighing

/-- Checks if a strategy can identify all coins uniquely -/
def canIdentifyAllCoins (s : Strategy) (coins : Set CoinWeight) : Prop := sorry

/-- Main theorem: There exists a strategy with at most 4 weighings that can identify all coins -/
theorem four_weighings_sufficient :
  ∃ (s : Strategy), s.length ≤ 4 ∧ canIdentifyAllCoins s CoinSet := by sorry

end NUMINAMATH_CALUDE_four_weighings_sufficient_l1531_153179


namespace NUMINAMATH_CALUDE_tylers_remaining_money_l1531_153146

/-- Calculates the remaining money after Tyler's purchase of scissors and erasers. -/
theorem tylers_remaining_money 
  (initial_money : ℕ) 
  (scissor_cost : ℕ) 
  (eraser_cost : ℕ) 
  (scissor_count : ℕ) 
  (eraser_count : ℕ) 
  (h1 : initial_money = 100)
  (h2 : scissor_cost = 5)
  (h3 : eraser_cost = 4)
  (h4 : scissor_count = 8)
  (h5 : eraser_count = 10) :
  initial_money - (scissor_cost * scissor_count + eraser_cost * eraser_count) = 20 := by
  sorry

#check tylers_remaining_money

end NUMINAMATH_CALUDE_tylers_remaining_money_l1531_153146


namespace NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l1531_153188

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) :
  speed1 > 0 →
  speed2 > 0 →
  (speed1 + speed2) / 2 = (speed1 * 1 + speed2 * 1) / (1 + 1) := by
  sorry

/-- The average speed of a car traveling 90 km/h for the first hour and 40 km/h for the second hour is 65 km/h -/
theorem car_average_speed :
  let speed1 : ℝ := 90
  let speed2 : ℝ := 40
  (speed1 + speed2) / 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l1531_153188


namespace NUMINAMATH_CALUDE_scientific_notation_3462_23_l1531_153169

theorem scientific_notation_3462_23 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 3462.23 = a * (10 : ℝ) ^ n ∧ a = 3.46223 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_3462_23_l1531_153169


namespace NUMINAMATH_CALUDE_max_distance_A_B_l1531_153123

def set_A : Set ℂ := {z : ℂ | z^4 - 16 = 0}
def set_B : Set ℂ := {z : ℂ | z^3 - 12*z^2 + 36*z - 64 = 0}

theorem max_distance_A_B : 
  ∃ (a : ℂ) (b : ℂ), a ∈ set_A ∧ b ∈ set_B ∧ 
    Complex.abs (a - b) = 10 ∧
    ∀ (x : ℂ) (y : ℂ), x ∈ set_A → y ∈ set_B → Complex.abs (x - y) ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_distance_A_B_l1531_153123


namespace NUMINAMATH_CALUDE_sibling_ages_theorem_l1531_153122

theorem sibling_ages_theorem :
  ∃ (a b c : ℕ+), 
    a * b * c = 72 ∧ 
    a + b + c = 13 ∧ 
    a > b ∧ b > c :=
by sorry

end NUMINAMATH_CALUDE_sibling_ages_theorem_l1531_153122


namespace NUMINAMATH_CALUDE_franks_remaining_money_l1531_153130

/-- Calculates the remaining money after Frank buys the most expensive lamp -/
def remaining_money (cheapest_lamp_cost : ℝ) (expensive_lamp_multiplier : ℝ) 
  (discount_rate : ℝ) (sales_tax_rate : ℝ) (initial_money : ℝ) : ℝ :=
  let expensive_lamp_cost := cheapest_lamp_cost * expensive_lamp_multiplier
  let discounted_price := expensive_lamp_cost * (1 - discount_rate)
  let final_price := discounted_price * (1 + sales_tax_rate)
  initial_money - final_price

/-- Theorem stating that Frank's remaining money is $31.68 -/
theorem franks_remaining_money :
  remaining_money 20 3 0.1 0.08 90 = 31.68 := by
  sorry

end NUMINAMATH_CALUDE_franks_remaining_money_l1531_153130


namespace NUMINAMATH_CALUDE_prob_odd_score_is_35_72_l1531_153106

/-- Represents the dartboard with given dimensions and point values -/
structure Dartboard :=
  (outer_radius : ℝ)
  (inner_radius : ℝ)
  (inner_points : Fin 3 → ℕ)
  (outer_points : Fin 3 → ℕ)

/-- Calculates the probability of scoring an odd sum with two darts -/
def prob_odd_score (db : Dartboard) : ℚ :=
  sorry

/-- The specific dartboard described in the problem -/
def problem_dartboard : Dartboard :=
  { outer_radius := 8
  , inner_radius := 4
  , inner_points := ![3, 4, 4]
  , outer_points := ![4, 3, 3] }

theorem prob_odd_score_is_35_72 :
  prob_odd_score problem_dartboard = 35 / 72 :=
sorry

end NUMINAMATH_CALUDE_prob_odd_score_is_35_72_l1531_153106


namespace NUMINAMATH_CALUDE_bertha_family_childless_count_l1531_153149

/-- Represents a family tree with two generations -/
structure FamilyTree where
  daughters : ℕ
  granddaughters : ℕ

/-- Bertha's family tree -/
def berthas_family : FamilyTree := { daughters := 10, granddaughters := 32 }

/-- The number of Bertha's daughters who have children -/
def daughters_with_children : ℕ := 8

/-- The number of daughters each child-bearing daughter has -/
def granddaughters_per_daughter : ℕ := 4

theorem bertha_family_childless_count :
  berthas_family.daughters + berthas_family.granddaughters - daughters_with_children = 34 :=
by sorry

end NUMINAMATH_CALUDE_bertha_family_childless_count_l1531_153149


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l1531_153154

/-- The function f(x) = x^2(ax + b) where a and b are real numbers. -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 * (a * x + b)

/-- The derivative of f(x) -/
def f_prime (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem f_decreasing_interval (a b : ℝ) :
  (f_prime a b 2 = 0) →  -- f has an extremum at x = 2
  (f_prime a b 1 = -3) →  -- tangent line at (1, f(1)) is parallel to 3x + y = 0
  ∀ x, 0 < x → x < 2 → f_prime a b x < 0 :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l1531_153154


namespace NUMINAMATH_CALUDE_hyperbola_minor_axis_length_l1531_153101

/-- Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
    if the distance from the foci to the asymptote is 3,
    then the length of the minor axis is 6. -/
theorem hyperbola_minor_axis_length (b : ℝ) (h1 : b > 0) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) →
  (∃ d : ℝ, d = 3 ∧ d = b) →
  (∃ l : ℝ, l = 6 ∧ l = 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_minor_axis_length_l1531_153101


namespace NUMINAMATH_CALUDE_average_marks_abcd_l1531_153170

theorem average_marks_abcd (a b c d e : ℝ) : 
  ((a + b + c) / 3 = 48) →
  ((b + c + d + e) / 4 = 48) →
  (e = d + 3) →
  (a = 43) →
  ((a + b + c + d) / 4 = 47) :=
by sorry

end NUMINAMATH_CALUDE_average_marks_abcd_l1531_153170


namespace NUMINAMATH_CALUDE_vector_sum_equals_l1531_153114

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := 2 • a + b

theorem vector_sum_equals : c = (5, 0) := by sorry

end NUMINAMATH_CALUDE_vector_sum_equals_l1531_153114


namespace NUMINAMATH_CALUDE_impossible_table_fill_l1531_153174

/-- A type representing a 6x6 table of integers -/
def Table : Type := Fin 6 → Fin 6 → ℤ

/-- Predicate to check if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  ∀ i j i' j', (i ≠ i' ∨ j ≠ j') → t i j ≠ t i' j'

/-- Predicate to check if the sum of numbers in a 1x5 rectangle is valid -/
def valid_sum (t : Table) (i j : Fin 6) (horizontal : Bool) : Prop :=
  let sum := if horizontal then
               (Finset.range 5).sum (fun k => t i (j + k))
             else
               (Finset.range 5).sum (fun k => t (i + k) j)
  sum = 2022 ∨ sum = 2023

/-- Predicate to check if all 1x5 rectangles have valid sums -/
def all_valid_sums (t : Table) : Prop :=
  ∀ i j, (j.val + 5 ≤ 6 → valid_sum t i j true) ∧
         (i.val + 5 ≤ 6 → valid_sum t i j false)

/-- Theorem stating that it's impossible to fill the table satisfying all conditions -/
theorem impossible_table_fill : ¬ ∃ t : Table, all_distinct t ∧ all_valid_sums t := by
  sorry

end NUMINAMATH_CALUDE_impossible_table_fill_l1531_153174


namespace NUMINAMATH_CALUDE_shark_count_l1531_153128

theorem shark_count (cape_may_sharks : ℕ) (other_beach_sharks : ℕ) : 
  cape_may_sharks = 32 → 
  cape_may_sharks = 2 * other_beach_sharks + 8 → 
  other_beach_sharks = 12 := by
sorry

end NUMINAMATH_CALUDE_shark_count_l1531_153128


namespace NUMINAMATH_CALUDE_marbles_fraction_l1531_153138

theorem marbles_fraction (total_marbles : ℕ) (marbles_taken : ℕ) :
  total_marbles = 100 →
  marbles_taken = 11 →
  (marbles_taken : ℚ) / (total_marbles : ℚ) = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_marbles_fraction_l1531_153138


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_equals_six_sqrt_three_l1531_153166

theorem sqrt_product_quotient_equals_six_sqrt_three :
  (Real.sqrt 12 * Real.sqrt 27) / Real.sqrt 3 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_equals_six_sqrt_three_l1531_153166


namespace NUMINAMATH_CALUDE_confectioner_pastries_l1531_153109

theorem confectioner_pastries :
  ∀ (P : ℕ) (x : ℕ),
    (P = 28 * (10 + x)) →
    (P = 49 * (4 + x)) →
    P = 392 :=
by
  sorry

end NUMINAMATH_CALUDE_confectioner_pastries_l1531_153109


namespace NUMINAMATH_CALUDE_shaded_cubes_count_total_cubes_count_face_size_edge_size_l1531_153118

/-- Represents a large cube constructed from smaller cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  shaded_cubes : Nat

/-- Defines the properties of our specific large cube -/
def our_cube : LargeCube :=
  { size := 4
  , total_cubes := 64
  , shaded_cubes := 28 }

/-- Calculates the number of cubes on one face of the large cube -/
def face_cubes (c : LargeCube) : Nat :=
  c.size * c.size

/-- Calculates the number of cubes along one edge of the large cube -/
def edge_cubes (c : LargeCube) : Nat :=
  c.size

/-- Calculates the number of corner cubes in the large cube -/
def corner_cubes : Nat := 8

/-- Theorem stating that the number of shaded cubes in our specific cube is 28 -/
theorem shaded_cubes_count (c : LargeCube) (h1 : c = our_cube) :
  c.shaded_cubes = 28 := by
  sorry

/-- Theorem stating that the total number of smaller cubes is 64 -/
theorem total_cubes_count (c : LargeCube) (h1 : c = our_cube) :
  c.total_cubes = 64 := by
  sorry

/-- Theorem stating that the size of each face is 4x4 -/
theorem face_size (c : LargeCube) (h1 : c = our_cube) :
  face_cubes c = 16 := by
  sorry

/-- Theorem stating that each edge has 4 cubes -/
theorem edge_size (c : LargeCube) (h1 : c = our_cube) :
  edge_cubes c = 4 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_total_cubes_count_face_size_edge_size_l1531_153118


namespace NUMINAMATH_CALUDE_sin_product_theorem_l1531_153141

theorem sin_product_theorem (x : ℝ) (h : Real.sin (5 * Real.pi / 2 - x) = 3 / 5) :
  Real.sin (x / 2) * Real.sin (5 * x / 2) = 86 / 125 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_theorem_l1531_153141


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1531_153110

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | 0 ≤ x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1531_153110


namespace NUMINAMATH_CALUDE_museum_trip_total_l1531_153133

/-- The total number of people going to the museum on four buses -/
def total_people (first_bus : ℕ) : ℕ :=
  let second_bus := 2 * first_bus
  let third_bus := second_bus - 6
  let fourth_bus := first_bus + 9
  first_bus + second_bus + third_bus + fourth_bus

/-- Theorem: Given the conditions about the four buses, 
    the total number of people going to the museum is 75 -/
theorem museum_trip_total : total_people 12 = 75 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_total_l1531_153133


namespace NUMINAMATH_CALUDE_range_of_m_l1531_153177

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (Set.compl (A m) ∩ B = ∅) → m ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1531_153177


namespace NUMINAMATH_CALUDE_motion_of_q_l1531_153126

/-- Point on a circle -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Motion of a point on a circle -/
structure CircularMotion where
  center : Point2D
  radius : ℝ
  angular_velocity : ℝ
  clockwise : Bool

/-- Given a point P moving counterclockwise on the unit circle with angular velocity ω,
    prove that the point Q(-2xy, y^2 - x^2) moves clockwise on the unit circle
    with angular velocity 2ω -/
theorem motion_of_q (ω : ℝ) (h_ω : ω > 0) :
  let p_motion : CircularMotion :=
    { center := ⟨0, 0⟩
    , radius := 1
    , angular_velocity := ω
    , clockwise := false }
  let q (p : Point2D) : Point2D :=
    ⟨-2 * p.x * p.y, p.y^2 - p.x^2⟩
  ∃ (q_motion : CircularMotion),
    q_motion.center = ⟨0, 0⟩ ∧
    q_motion.radius = 1 ∧
    q_motion.angular_velocity = 2 * ω ∧
    q_motion.clockwise = true :=
by sorry

end NUMINAMATH_CALUDE_motion_of_q_l1531_153126


namespace NUMINAMATH_CALUDE_corn_price_is_ten_cents_l1531_153136

/-- Represents the farmer's corn production and sales --/
structure CornFarmer where
  seeds_per_ear : ℕ
  seeds_per_bag : ℕ
  cost_per_bag : ℚ
  profit : ℚ
  ears_sold : ℕ

/-- Calculates the price per ear of corn --/
def price_per_ear (farmer : CornFarmer) : ℚ :=
  let total_seeds := farmer.ears_sold * farmer.seeds_per_ear
  let bags_needed := (total_seeds + farmer.seeds_per_bag - 1) / farmer.seeds_per_bag
  let seed_cost := bags_needed * farmer.cost_per_bag
  let total_revenue := farmer.profit + seed_cost
  total_revenue / farmer.ears_sold

/-- Theorem stating the price per ear of corn is $0.10 --/
theorem corn_price_is_ten_cents (farmer : CornFarmer) 
    (h1 : farmer.seeds_per_ear = 4)
    (h2 : farmer.seeds_per_bag = 100)
    (h3 : farmer.cost_per_bag = 1/2)
    (h4 : farmer.profit = 40)
    (h5 : farmer.ears_sold = 500) : 
  price_per_ear farmer = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_corn_price_is_ten_cents_l1531_153136


namespace NUMINAMATH_CALUDE_range_of_k_prove_k_range_l1531_153139

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}
def B (k : ℝ) : Set ℝ := {x | x < k}

-- Theorem statement
theorem range_of_k (k : ℝ) :
  (A ∪ B k = B k) → k > 2 := by
  sorry

-- The range of k
def k_range : Set ℝ := {k | k > 2}

-- Theorem to prove the range of k
theorem prove_k_range :
  ∀ k, (A ∪ B k = B k) ↔ k ∈ k_range := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_prove_k_range_l1531_153139


namespace NUMINAMATH_CALUDE_cubic_equation_root_difference_l1531_153161

theorem cubic_equation_root_difference (a b c : ℚ) : 
  ∃ (p q r : ℚ), p^3 + a*p^2 + b*p + c = 0 ∧ 
                  q^3 + a*q^2 + b*q + c = 0 ∧ 
                  r^3 + a*r^2 + b*r + c = 0 ∧ 
                  (q - p = 2014 ∨ r - q = 2014 ∨ r - p = 2014) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_root_difference_l1531_153161


namespace NUMINAMATH_CALUDE_andy_max_cookies_l1531_153129

/-- The maximum number of cookies Andy can eat given the conditions -/
def max_cookies_andy (total : ℕ) (bella_ratio : ℕ) : ℕ :=
  total / (bella_ratio + 1)

/-- Proof that Andy's maximum cookie consumption is correct -/
theorem andy_max_cookies :
  let total := 36
  let bella_ratio := 2
  let andy_cookies := max_cookies_andy total bella_ratio
  andy_cookies = 12 ∧
  andy_cookies + bella_ratio * andy_cookies = total ∧
  ∀ x : ℕ, x > andy_cookies → x + bella_ratio * x > total :=
by sorry

#eval max_cookies_andy 36 2  -- Should output 12

end NUMINAMATH_CALUDE_andy_max_cookies_l1531_153129


namespace NUMINAMATH_CALUDE_height_difference_l1531_153196

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Converts a Height to total inches -/
def heightToInches (h : Height) : ℕ := h.feet * 12 + h.inches

/-- Mark's height -/
def markHeight : Height := ⟨5, 3⟩

/-- Mike's height -/
def mikeHeight : Height := ⟨6, 1⟩

theorem height_difference : heightToInches mikeHeight - heightToInches markHeight = 10 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l1531_153196


namespace NUMINAMATH_CALUDE_deck_size_l1531_153175

theorem deck_size (r b : ℕ) : 
  r > 0 ∧ b > 0 → -- Ensure positive number of cards
  r / (r + b : ℚ) = 1 / 4 → -- Initial probability
  r / (r + b + 6 : ℚ) = 1 / 6 → -- Probability after adding 6 black cards
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l1531_153175


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l1531_153105

theorem imaginary_part_of_one_plus_i_squared (z : ℂ) : z = 1 + Complex.I → Complex.im (z^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l1531_153105


namespace NUMINAMATH_CALUDE_ceiling_minus_x_eq_half_l1531_153167

theorem ceiling_minus_x_eq_half (x : ℝ) (h : x - ⌊x⌋ = 1/2) : ⌈x⌉ - x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_eq_half_l1531_153167


namespace NUMINAMATH_CALUDE_profit_difference_maddox_profit_exceeds_theo_by_15_l1531_153194

/-- Calculates the profit difference between two sellers of Polaroid cameras. -/
theorem profit_difference (num_cameras : ℕ) (cost_per_camera : ℕ) 
  (maddox_selling_price : ℕ) (theo_selling_price : ℕ) : ℕ :=
  let maddox_profit := num_cameras * maddox_selling_price - num_cameras * cost_per_camera
  let theo_profit := num_cameras * theo_selling_price - num_cameras * cost_per_camera
  maddox_profit - theo_profit

/-- Proves that Maddox made $15 more profit than Theo. -/
theorem maddox_profit_exceeds_theo_by_15 : 
  profit_difference 3 20 28 23 = 15 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_maddox_profit_exceeds_theo_by_15_l1531_153194


namespace NUMINAMATH_CALUDE_unique_solution_ab_minus_a_minus_b_equals_one_l1531_153150

theorem unique_solution_ab_minus_a_minus_b_equals_one :
  ∀ a b : ℤ, a > b ∧ b > 0 ∧ a * b - a - b = 1 → a = 3 ∧ b = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_ab_minus_a_minus_b_equals_one_l1531_153150


namespace NUMINAMATH_CALUDE_tv_production_theorem_l1531_153127

/-- Represents the daily TV production in a factory for a month -/
structure TVProduction where
  totalDays : Nat
  firstPeriodDays : Nat
  firstPeriodAvg : Nat
  monthlyAvg : Nat

/-- Calculates the average daily production for the last period of the month -/
def lastPeriodAvg (p : TVProduction) : Nat :=
  let lastPeriodDays := p.totalDays - p.firstPeriodDays
  let totalProduction := p.totalDays * p.monthlyAvg
  let firstPeriodProduction := p.firstPeriodDays * p.firstPeriodAvg
  (totalProduction - firstPeriodProduction) / lastPeriodDays

theorem tv_production_theorem (p : TVProduction) 
  (h1 : p.totalDays = 30)
  (h2 : p.firstPeriodDays = 25)
  (h3 : p.firstPeriodAvg = 65)
  (h4 : p.monthlyAvg = 60) :
  lastPeriodAvg p = 35 := by
  sorry

#eval lastPeriodAvg ⟨30, 25, 65, 60⟩

end NUMINAMATH_CALUDE_tv_production_theorem_l1531_153127


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1531_153125

theorem quadratic_inequality (m : ℝ) : (∃ x : ℝ, x^2 - x - m = 0) → m ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1531_153125


namespace NUMINAMATH_CALUDE_sum_of_extrema_l1531_153111

theorem sum_of_extrema (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a^2 + b^2 + c^2 = 11) :
  ∃ (m M : ℝ), (∀ x, (∃ y z, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 11) → m ≤ x ∧ x ≤ M) ∧
                m + M = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_extrema_l1531_153111


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1531_153173

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  is_arithmetic_sequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 4 * b 6 = 17 →
  b 3 * b 7 = -175 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1531_153173


namespace NUMINAMATH_CALUDE_xy_max_value_l1531_153171

theorem xy_max_value (x y : ℝ) (h : x^2 + 2*y^2 - 2*x*y = 4) :
  x*y ≤ 2*Real.sqrt 2 + 2 := by
sorry

end NUMINAMATH_CALUDE_xy_max_value_l1531_153171


namespace NUMINAMATH_CALUDE_red_balls_count_l1531_153142

/-- Given a bag with 15 balls of red, yellow, and blue colors, 
    if the probability of drawing two non-red balls at the same time is 2/7, 
    then the number of red balls in the bag is 5. -/
theorem red_balls_count (total : ℕ) (red : ℕ) 
  (h_total : total = 15)
  (h_prob : (total - red : ℚ) / total * ((total - 1 - red) : ℚ) / (total - 1) = 2 / 7) :
  red = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1531_153142


namespace NUMINAMATH_CALUDE_calculation_proof_l1531_153143

theorem calculation_proof : (-1)^3 - 8 / (-2) + 4 * |(-5)| = 23 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1531_153143


namespace NUMINAMATH_CALUDE_positive_sum_and_product_iff_both_positive_l1531_153156

theorem positive_sum_and_product_iff_both_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_and_product_iff_both_positive_l1531_153156


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1531_153193

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (6 * x₁^2 + 5 * x₁ - 11 = 0) → 
  (6 * x₂^2 + 5 * x₂ - 11 = 0) → 
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 157 / 36 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1531_153193


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l1531_153120

/-- The equation of the trajectory of the midpoint of a line segment between a point on a unit circle and a fixed point -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (px py : ℝ), px^2 + py^2 = 1 ∧ x = (px + 3) / 2 ∧ y = py / 2) ↔ 
  (2*x - 3)^2 + 4*y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l1531_153120


namespace NUMINAMATH_CALUDE_martha_juice_bottles_l1531_153115

/-- The number of juice bottles left after a week -/
def bottles_left (initial_refrigerator : ℕ) (initial_pantry : ℕ) (bought : ℕ) (consumed : ℕ) : ℕ :=
  initial_refrigerator + initial_pantry + bought - consumed

/-- Theorem: Given Martha's initial bottles, purchases, and consumption, prove that 10 bottles are left -/
theorem martha_juice_bottles : 
  bottles_left 4 4 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_martha_juice_bottles_l1531_153115


namespace NUMINAMATH_CALUDE_window_side_length_l1531_153121

/-- Represents the dimensions of a window pane -/
structure Pane where
  height : ℝ
  width : ℝ

/-- Represents the dimensions and properties of a window -/
structure Window where
  paneCount : ℕ
  rows : ℕ
  columns : ℕ
  pane : Pane
  borderWidth : ℝ

/-- The theorem stating that given the specified conditions, the window's side length is 27 inches -/
theorem window_side_length (w : Window) : 
  w.paneCount = 8 ∧ 
  w.rows = 2 ∧ 
  w.columns = 4 ∧ 
  w.pane.height = 3 * w.pane.width ∧
  w.borderWidth = 3 →
  (w.columns * w.pane.width + (w.columns + 1) * w.borderWidth : ℝ) = 27 :=
by sorry


end NUMINAMATH_CALUDE_window_side_length_l1531_153121


namespace NUMINAMATH_CALUDE_sum_equals_14x_l1531_153113

-- Define variables
variable (x y z : ℝ)

-- State the theorem
theorem sum_equals_14x (h1 : y = 3 * x) (h2 : z = 3 * y + x) : 
  x + y + z = 14 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_14x_l1531_153113


namespace NUMINAMATH_CALUDE_candy_block_pieces_l1531_153140

/-- The number of candy pieces per block in Jan's candy necklace problem -/
def candy_pieces_per_block (total_necklaces : ℕ) (pieces_per_necklace : ℕ) (total_blocks : ℕ) : ℕ :=
  (total_necklaces * pieces_per_necklace) / total_blocks

/-- Theorem stating that the number of candy pieces per block is 30 -/
theorem candy_block_pieces :
  candy_pieces_per_block 9 10 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_candy_block_pieces_l1531_153140


namespace NUMINAMATH_CALUDE_penguin_giraffe_ratio_l1531_153145

/-- Represents the zoo with its animal composition -/
structure Zoo where
  total_animals : ℕ
  giraffes : ℕ
  penguins : ℕ
  elephants : ℕ

/-- The conditions of the zoo -/
def zoo_conditions (z : Zoo) : Prop :=
  z.giraffes = 5 ∧
  z.penguins = (20 : ℕ) * z.total_animals / 100 ∧
  z.elephants = 2 ∧
  z.elephants = (4 : ℕ) * z.total_animals / 100

/-- The theorem stating the ratio of penguins to giraffes -/
theorem penguin_giraffe_ratio (z : Zoo) (h : zoo_conditions z) : 
  z.penguins / z.giraffes = 2 := by
  sorry

#check penguin_giraffe_ratio

end NUMINAMATH_CALUDE_penguin_giraffe_ratio_l1531_153145


namespace NUMINAMATH_CALUDE_perpendicular_bisector_y_intercept_range_l1531_153168

/-- Given two distinct points on a parabola y = 2x², prove that the y-intercept of their perpendicular bisector with slope 2 is greater than 9/32. -/
theorem perpendicular_bisector_y_intercept_range 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂))
  (h_parabola₁ : y₁ = 2 * x₁^2)
  (h_parabola₂ : y₂ = 2 * x₂^2)
  (b : ℝ) 
  (h_perpendicular_bisector : ∃ (m : ℝ), 
    y₁ = -1/(2*m) * x₁ + b + 1/(4*m) ∧ 
    y₂ = -1/(2*m) * x₂ + b + 1/(4*m) ∧ 
    m = 2) : 
  b > 9/32 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_y_intercept_range_l1531_153168


namespace NUMINAMATH_CALUDE_parallelogram_area_l1531_153119

/-- The area of a parallelogram with given dimensions -/
theorem parallelogram_area (base slant_height : ℝ) (angle : ℝ) : 
  base = 20 → 
  slant_height = 10 → 
  angle = 30 * π / 180 → 
  base * (slant_height * Real.sin angle) = 100 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1531_153119


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l1531_153117

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry -- Additional conditions to ensure the octagon is regular

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (1 : ℝ) / 4 * area o := by
  sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l1531_153117


namespace NUMINAMATH_CALUDE_circle_radii_order_l1531_153185

theorem circle_radii_order (rA rB rC : ℝ) : 
  rA = Real.sqrt 16 →
  π * rB^2 = 16 * π →
  2 * π * rC = 10 * π →
  rA ≤ rB ∧ rB ≤ rC :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_order_l1531_153185


namespace NUMINAMATH_CALUDE_zero_not_identity_for_star_l1531_153104

-- Define the set S
def S : Set ℝ := {x : ℝ | x ≠ -1/3}

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * a * b + 1

-- Theorem statement
theorem zero_not_identity_for_star :
  ¬(∀ a ∈ S, (star 0 a = a ∧ star a 0 = a)) :=
sorry

end NUMINAMATH_CALUDE_zero_not_identity_for_star_l1531_153104


namespace NUMINAMATH_CALUDE_lg_sum_equals_two_l1531_153158

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_two : lg 4 + lg 25 = 2 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_two_l1531_153158


namespace NUMINAMATH_CALUDE_rotate_W_180_is_M_l1531_153198

/-- Represents an uppercase English letter -/
inductive UppercaseLetter
| W
| M

/-- Represents a geometric figure -/
class GeometricFigure where
  /-- Indicates if the figure is axisymmetric -/
  is_axisymmetric : Bool

/-- Represents the result of rotating a letter -/
def rotate_180_degrees (letter : UppercaseLetter) (is_axisymmetric : Bool) : UppercaseLetter :=
  sorry

/-- Theorem: Rotating W 180° results in M -/
theorem rotate_W_180_is_M :
  ∀ (w : UppercaseLetter) (fig : GeometricFigure),
    w = UppercaseLetter.W →
    fig.is_axisymmetric = true →
    rotate_180_degrees w fig.is_axisymmetric = UppercaseLetter.M :=
  sorry

end NUMINAMATH_CALUDE_rotate_W_180_is_M_l1531_153198


namespace NUMINAMATH_CALUDE_total_first_grade_muffins_l1531_153155

def mrs_brier_muffins : ℕ := 218
def mrs_macadams_muffins : ℕ := 320
def mrs_flannery_muffins : ℕ := 417
def mrs_smith_muffins : ℕ := 292
def mr_jackson_muffins : ℕ := 389

theorem total_first_grade_muffins :
  mrs_brier_muffins + mrs_macadams_muffins + mrs_flannery_muffins +
  mrs_smith_muffins + mr_jackson_muffins = 1636 := by
  sorry

end NUMINAMATH_CALUDE_total_first_grade_muffins_l1531_153155


namespace NUMINAMATH_CALUDE_cos_product_equation_l1531_153176

theorem cos_product_equation (α : Real) (h : Real.tan α = 2) :
  Real.cos (Real.pi + α) * Real.cos (Real.pi / 2 + α) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_equation_l1531_153176


namespace NUMINAMATH_CALUDE_library_book_purchase_l1531_153192

theorem library_book_purchase (initial_books : ℕ) (current_books : ℕ) (last_year_purchase : ℕ) : 
  initial_books = 100 →
  current_books = 300 →
  current_books = initial_books + last_year_purchase + 3 * last_year_purchase →
  last_year_purchase = 50 := by
  sorry

end NUMINAMATH_CALUDE_library_book_purchase_l1531_153192


namespace NUMINAMATH_CALUDE_target_probability_l1531_153107

/-- The probability of hitting a target in one shot. -/
def p : ℝ := 0.6

/-- The number of shots taken. -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in three shots. -/
def prob_at_least_two : ℝ := 3 * p^2 * (1 - p) + p^3

theorem target_probability :
  prob_at_least_two = 0.648 := by
  sorry

end NUMINAMATH_CALUDE_target_probability_l1531_153107


namespace NUMINAMATH_CALUDE_solve_system_l1531_153184

theorem solve_system (s t : ℤ) (eq1 : 11 * s + 7 * t = 160) (eq2 : s = 2 * t + 4) : t = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1531_153184


namespace NUMINAMATH_CALUDE_sin_52pi_over_3_l1531_153178

theorem sin_52pi_over_3 : Real.sin (52 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_52pi_over_3_l1531_153178


namespace NUMINAMATH_CALUDE_inequality_chain_l1531_153144

theorem inequality_chain (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  9 / (a + b + c) ≤ 2 / (a + b) + 2 / (b + c) + 2 / (c + a) ∧
  2 / (a + b) + 2 / (b + c) + 2 / (c + a) ≤ 1 / a + 1 / b + 1 / c :=
by sorry

end NUMINAMATH_CALUDE_inequality_chain_l1531_153144


namespace NUMINAMATH_CALUDE_sqrt_6_simplest_l1531_153147

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y^2 = x → ¬∃ z : ℝ, z > 0 ∧ z < y ∧ z^2 = x

theorem sqrt_6_simplest :
  is_simplest_sqrt 6 ∧
  ¬is_simplest_sqrt (1/6) ∧
  ¬is_simplest_sqrt 0.6 ∧
  ¬is_simplest_sqrt 60 :=
sorry

end NUMINAMATH_CALUDE_sqrt_6_simplest_l1531_153147


namespace NUMINAMATH_CALUDE_equation_solutions_l1531_153181

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = (-3 + Real.sqrt 13) / 2 ∧ x2 = (-3 - Real.sqrt 13) / 2 ∧
    x1^2 + 3*x1 - 1 = 0 ∧ x2^2 + 3*x2 - 1 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = -2 ∧ x2 = -1 ∧
    (x1 + 2)^2 = x1 + 2 ∧ (x2 + 2)^2 = x2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1531_153181


namespace NUMINAMATH_CALUDE_power_multiplication_l1531_153108

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1531_153108


namespace NUMINAMATH_CALUDE_unique_integer_fraction_l1531_153132

theorem unique_integer_fraction : ∃! n : ℕ, 
  1 ≤ n ∧ n ≤ 2014 ∧ ∃ k : ℤ, 8 * n = k * (9999 - n) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_fraction_l1531_153132


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1531_153164

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (plane_perp : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (line_perp_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (line_perp : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (m n : Line)
  (h1 : line_perp_plane m α)
  (h2 : line_perp_plane n β)
  (h3 : plane_perp α β) :
  line_perp m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1531_153164


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l1531_153160

theorem smallest_common_multiple_of_6_and_15 :
  ∃ a : ℕ+, (∀ b : ℕ+, (6 ∣ b) ∧ (15 ∣ b) → a ≤ b) ∧ (6 ∣ a) ∧ (15 ∣ a) ∧ a = 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l1531_153160


namespace NUMINAMATH_CALUDE_renovation_material_sum_l1531_153182

/-- The amount of sand required for the renovation project in truck-loads -/
def sand : ℚ := 0.16666666666666666

/-- The amount of dirt required for the renovation project in truck-loads -/
def dirt : ℚ := 0.3333333333333333

/-- The amount of cement required for the renovation project in truck-loads -/
def cement : ℚ := 0.16666666666666666

/-- The total amount of material required for the renovation project in truck-loads -/
def total_material : ℚ := 0.6666666666666666

/-- Theorem stating that the sum of sand, dirt, and cement equals the total material required -/
theorem renovation_material_sum :
  sand + dirt + cement = total_material := by sorry

end NUMINAMATH_CALUDE_renovation_material_sum_l1531_153182


namespace NUMINAMATH_CALUDE_triangle_area_change_l1531_153157

theorem triangle_area_change (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let a' := 2 * a
  let b' := 1.5 * b
  let c' := c
  let s' := (a' + b' + c') / 2
  let area' := Real.sqrt (s' * (s' - a') * (s' - b') * (s' - c'))
  2 * area < area' ∧ area' < 3 * area :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_change_l1531_153157


namespace NUMINAMATH_CALUDE_vasya_fraction_simplification_l1531_153180

theorem vasya_fraction_simplification (n : ℕ) : 
  (n = (2 * (10^1990 - 1)) / 3) → 
  (10^1990 + n) / (10 * n + 4) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_vasya_fraction_simplification_l1531_153180


namespace NUMINAMATH_CALUDE_sophie_donut_purchase_l1531_153186

/-- Calculates the total cost and remaining donuts for Sophie's purchase --/
theorem sophie_donut_purchase (budget : ℕ) (box_cost : ℕ) (discount_rate : ℚ) 
  (boxes_bought : ℕ) (donuts_per_box : ℕ) (boxes_given : ℕ) (donuts_given : ℕ) :
  budget = 50 ∧ 
  box_cost = 12 ∧ 
  discount_rate = 1/10 ∧ 
  boxes_bought = 4 ∧ 
  donuts_per_box = 12 ∧ 
  boxes_given = 1 ∧ 
  donuts_given = 6 →
  ∃ (total_cost : ℚ) (donuts_left : ℕ),
    total_cost = 43.2 ∧ 
    donuts_left = 30 :=
by sorry

end NUMINAMATH_CALUDE_sophie_donut_purchase_l1531_153186


namespace NUMINAMATH_CALUDE_computer_table_price_l1531_153195

theorem computer_table_price (cost_price : ℝ) (markup_percentage : ℝ) 
  (h1 : cost_price = 4090.9090909090905)
  (h2 : markup_percentage = 32) :
  cost_price * (1 + markup_percentage / 100) = 5400 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_price_l1531_153195


namespace NUMINAMATH_CALUDE_bisecting_line_sum_l1531_153190

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(9, 0) -/
structure Triangle where
  P : ℝ × ℝ := (0, 10)
  Q : ℝ × ℝ := (3, 0)
  R : ℝ × ℝ := (9, 0)

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The line through Q that bisects the area of the triangle -/
def bisectingLine (t : Triangle) : Line :=
  sorry

theorem bisecting_line_sum (t : Triangle) :
  let l := bisectingLine t
  l.slope + l.yIntercept = -20/3 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_l1531_153190


namespace NUMINAMATH_CALUDE_problem_statement_l1531_153199

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem problem_statement (f : ℝ → ℝ) (m n : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f m - f n > f (-m) - f (-n)) :
  m - n < 0 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1531_153199


namespace NUMINAMATH_CALUDE_odd_function_root_property_l1531_153135

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of being a root
def IsRoot (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

-- State the theorem
theorem odd_function_root_property
  (f : ℝ → ℝ) (x₀ : ℝ)
  (h_odd : OddFunction f)
  (h_root : IsRoot (fun x => f x - Real.exp x) x₀) :
  IsRoot (fun x => f x * Real.exp x + 1) (-x₀) := by
sorry

end NUMINAMATH_CALUDE_odd_function_root_property_l1531_153135


namespace NUMINAMATH_CALUDE_integral_proof_l1531_153116

open Real

noncomputable def f (x : ℝ) : ℝ := 
  (1/2) * log (abs (x^2 + x + 1)) + 
  (1/sqrt 3) * arctan ((2*x + 1)/sqrt 3) + 
  (1/2) * log (abs (x^2 + 1))

theorem integral_proof (x : ℝ) : 
  deriv f x = (2*x^3 + 2*x^2 + 2*x + 1) / ((x^2 + x + 1) * (x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l1531_153116


namespace NUMINAMATH_CALUDE_base7_product_and_sum_l1531_153151

/-- Converts a base 7 number to decimal --/
def toDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a decimal number to base 7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Computes the sum of digits in a list --/
def sumDigits (n : List Nat) : Nat :=
  n.foldl (· + ·) 0

/-- The main theorem to prove --/
theorem base7_product_and_sum :
  let a := [5, 3]  -- 35 in base 7
  let b := [4, 2]  -- 24 in base 7
  let product := toBase7 (toDecimal a * toDecimal b)
  product = [6, 3, 2, 1] ∧ 
  toBase7 (sumDigits product) = [5, 1] := by
  sorry


end NUMINAMATH_CALUDE_base7_product_and_sum_l1531_153151


namespace NUMINAMATH_CALUDE_norm_scalar_multiple_l1531_153137

theorem norm_scalar_multiple (v : ℝ × ℝ) :
  ‖v‖ = 7 → ‖(5 : ℝ) • v‖ = 35 := by
  sorry

end NUMINAMATH_CALUDE_norm_scalar_multiple_l1531_153137


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l1531_153134

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ i : ℕ, a ≤ i → i ≤ b → n % i = 0

theorem smallest_divisible_by_1_to_12 :
  ∃ (n : ℕ), n > 0 ∧ is_divisible_by_range n 1 12 ∧
  ∀ (m : ℕ), m > 0 → is_divisible_by_range m 1 12 → n ≤ m :=
by
  use 27720
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l1531_153134


namespace NUMINAMATH_CALUDE_lemonade_glasses_l1531_153103

/-- The number of glasses of lemonade that can be made -/
def glasses_of_lemonade (total_lemons : ℕ) (lemons_per_glass : ℕ) : ℕ :=
  total_lemons / lemons_per_glass

/-- Theorem: Given 18 lemons and 2 lemons required per glass, 9 glasses of lemonade can be made -/
theorem lemonade_glasses : glasses_of_lemonade 18 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_glasses_l1531_153103


namespace NUMINAMATH_CALUDE_assignment_methods_count_l1531_153183

def number_of_departments : ℕ := 5
def number_of_graduates : ℕ := 4
def departments_to_fill : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

def assignment_methods : ℕ := 
  (choose number_of_departments departments_to_fill) * 
  (choose number_of_graduates 2) * 
  (permute departments_to_fill departments_to_fill)

theorem assignment_methods_count : assignment_methods = 360 := by
  sorry

end NUMINAMATH_CALUDE_assignment_methods_count_l1531_153183
