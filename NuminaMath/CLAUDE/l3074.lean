import Mathlib

namespace NUMINAMATH_CALUDE_taxi_fare_distance_l3074_307492

/-- Represents the fare structure and total charge for a taxi ride -/
structure TaxiFare where
  initialCharge : ℚ  -- Initial charge for the first 1/5 mile
  additionalCharge : ℚ  -- Charge for each additional 1/5 mile
  totalCharge : ℚ  -- Total charge for the ride

/-- Calculates the distance of a taxi ride given the fare structure and total charge -/
def calculateDistance (fare : TaxiFare) : ℚ :=
  let additionalDistance := (fare.totalCharge - fare.initialCharge) / fare.additionalCharge
  (additionalDistance + 1) / 5

/-- Theorem stating that for the given fare structure and total charge, the ride distance is 8 miles -/
theorem taxi_fare_distance (fare : TaxiFare) 
    (h1 : fare.initialCharge = 280/100)
    (h2 : fare.additionalCharge = 40/100)
    (h3 : fare.totalCharge = 1840/100) : 
  calculateDistance fare = 8 := by
  sorry

#eval calculateDistance { initialCharge := 280/100, additionalCharge := 40/100, totalCharge := 1840/100 }

end NUMINAMATH_CALUDE_taxi_fare_distance_l3074_307492


namespace NUMINAMATH_CALUDE_tax_savings_proof_l3074_307450

def initial_tax_rate : ℝ := 0.46
def new_tax_rate : ℝ := 0.32
def annual_income : ℝ := 36000

def differential_savings : ℝ :=
  annual_income * initial_tax_rate - annual_income * new_tax_rate

theorem tax_savings_proof :
  differential_savings = 5040 := by
  sorry

end NUMINAMATH_CALUDE_tax_savings_proof_l3074_307450


namespace NUMINAMATH_CALUDE_negation_of_all_students_punctual_l3074_307414

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a student and being punctual
variable (student : U → Prop)
variable (punctual : U → Prop)

-- State the theorem
theorem negation_of_all_students_punctual :
  (¬ ∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_students_punctual_l3074_307414


namespace NUMINAMATH_CALUDE_absolute_value_problem_l3074_307446

theorem absolute_value_problem : |-5| + (3 - Real.sqrt 2) ^ 0 - 2 * Real.tan (π / 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l3074_307446


namespace NUMINAMATH_CALUDE_female_guests_from_jays_family_l3074_307433

def total_guests : ℕ := 240
def female_percentage : ℚ := 60 / 100
def jays_family_percentage : ℚ := 50 / 100

theorem female_guests_from_jays_family :
  (total_guests : ℚ) * female_percentage * jays_family_percentage = 72 := by
  sorry

end NUMINAMATH_CALUDE_female_guests_from_jays_family_l3074_307433


namespace NUMINAMATH_CALUDE_race_to_top_floor_l3074_307418

/-- Represents the time taken by a person to reach the top floor of a building -/
def TimeTaken (stories : ℕ) (timePerStory : ℕ) (stopTime : ℕ) (stopsPerStory : ℕ) : ℕ :=
  stories * timePerStory + (stories - 2) * stopTime * stopsPerStory

/-- The maximum time taken between two people to reach the top floor -/
def MaxTimeTaken (time1 : ℕ) (time2 : ℕ) : ℕ :=
  max time1 time2

theorem race_to_top_floor :
  let stories := 20
  let lolaTimePerStory := 10
  let elevatorTimePerStory := 8
  let elevatorStopTime := 3
  let elevatorStopsPerStory := 1
  let lolaTime := TimeTaken stories lolaTimePerStory 0 0
  let taraTime := TimeTaken stories elevatorTimePerStory elevatorStopTime elevatorStopsPerStory
  MaxTimeTaken lolaTime taraTime = 214 :=
by sorry


end NUMINAMATH_CALUDE_race_to_top_floor_l3074_307418


namespace NUMINAMATH_CALUDE_inequality_relation_to_line_l3074_307422

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := x + (a - 1) * y + 3 = 0

-- Define the inequality
def inequality (x y a : ℝ) : Prop := x + (a - 1) * y + 3 > 0

-- Theorem statement
theorem inequality_relation_to_line :
  ∀ (a : ℝ), 
    (a > 1 → ∀ (x y : ℝ), inequality x y a → ¬(line_equation x y a)) ∧
    (a < 1 → ∀ (x y : ℝ), ¬(inequality x y a) → line_equation x y a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relation_to_line_l3074_307422


namespace NUMINAMATH_CALUDE_area_inequality_l3074_307470

/-- A point with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A convex quadrilateral with integer vertices -/
structure ConvexQuadrilateral where
  A : IntPoint
  B : IntPoint
  C : IntPoint
  D : IntPoint
  convex : Bool  -- Assume this is true for a convex quadrilateral

/-- The intersection point of the diagonals -/
def diagonalIntersection (q : ConvexQuadrilateral) : IntPoint :=
  sorry  -- Definition of diagonal intersection

/-- The area of a shape -/
class HasArea (α : Type) where
  area : α → ℝ

instance : HasArea ConvexQuadrilateral where
  area := sorry  -- Definition of quadrilateral area

instance : HasArea (IntPoint × IntPoint × IntPoint) where
  area := sorry  -- Definition of triangle area

theorem area_inequality (q : ConvexQuadrilateral) :
  let S := diagonalIntersection q
  let P := HasArea.area q
  let P₁ := HasArea.area (q.A, q.B, S)
  Real.sqrt P ≥ Real.sqrt P₁ + Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_inequality_l3074_307470


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3074_307485

theorem sum_of_solutions_quadratic (a b c d e : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let g : ℝ → ℝ := λ x => d * x + e
  (∀ x, f x = g x) →
  (-(b - d) / (2 * a)) = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3074_307485


namespace NUMINAMATH_CALUDE_quadratic_points_theorem_l3074_307440

/-- Quadratic function -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + 2*m*x - 3

theorem quadratic_points_theorem (m n p q : ℝ) 
  (h_m : m > 0)
  (h_A : f m (n-2) = p)
  (h_B : f m 4 = q)
  (h_C : f m n = p)
  (h_q : -3 < q)
  (h_p : q < p) :
  (m = n - 1) ∧ ((3 < n ∧ n < 4) ∨ n > 6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_theorem_l3074_307440


namespace NUMINAMATH_CALUDE_shelter_dogs_l3074_307429

theorem shelter_dogs (D C R P : ℕ) : 
  D * 7 = C * 15 →  -- Initial ratio of dogs to cats
  R * 5 = P * 9 →   -- Initial ratio of rabbits to parrots
  D * 11 = (C + 8) * 15 →  -- New ratio of dogs to cats after adding 8 cats
  (R + 6) * 5 = P * 7 →    -- New ratio of rabbits to parrots after adding 6 rabbits
  D = 30 :=
by sorry

end NUMINAMATH_CALUDE_shelter_dogs_l3074_307429


namespace NUMINAMATH_CALUDE_total_flight_distance_l3074_307496

theorem total_flight_distance (beka_distance jackson_distance maria_distance : ℕ) 
  (h1 : beka_distance = 873)
  (h2 : jackson_distance = 563)
  (h3 : maria_distance = 786) :
  beka_distance + jackson_distance + maria_distance = 2222 := by
  sorry

end NUMINAMATH_CALUDE_total_flight_distance_l3074_307496


namespace NUMINAMATH_CALUDE_least_product_of_three_primes_greater_than_10_l3074_307498

theorem least_product_of_three_primes_greater_than_10 :
  ∃ (p q r : ℕ),
    Prime p ∧ Prime q ∧ Prime r ∧
    p > 10 ∧ q > 10 ∧ r > 10 ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    p * q * r = 2431 ∧
    (∀ (a b c : ℕ),
      Prime a ∧ Prime b ∧ Prime c ∧
      a > 10 ∧ b > 10 ∧ c > 10 ∧
      a ≠ b ∧ a ≠ c ∧ b ≠ c →
      a * b * c ≥ 2431) :=
by sorry


end NUMINAMATH_CALUDE_least_product_of_three_primes_greater_than_10_l3074_307498


namespace NUMINAMATH_CALUDE_point_on_765_degree_angle_l3074_307401

/-- Given that a point (4, m) lies on the terminal side of an angle of 765°, prove that m = 4 -/
theorem point_on_765_degree_angle (m : ℝ) : 
  (∃ (θ : ℝ), θ = 765 * Real.pi / 180 ∧ Real.tan θ = m / 4) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_765_degree_angle_l3074_307401


namespace NUMINAMATH_CALUDE_chef_used_41_apples_l3074_307424

/-- The number of apples the chef used to make pies -/
def apples_used (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Theorem: Given the initial number of apples and the remaining number of apples,
    prove that the number of apples used is 41. -/
theorem chef_used_41_apples (initial : ℕ) (remaining : ℕ)
  (h1 : initial = 43)
  (h2 : remaining = 2) :
  apples_used initial remaining = 41 := by
  sorry

end NUMINAMATH_CALUDE_chef_used_41_apples_l3074_307424


namespace NUMINAMATH_CALUDE_even_painted_faces_count_l3074_307412

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of painted faces in a block -/
def countEvenPaintedFaces (b : Block) : ℕ :=
  sorry

/-- The main theorem stating that a 6x3x2 block has 16 cubes with even number of painted faces -/
theorem even_painted_faces_count : 
  let b : Block := { length := 6, width := 3, height := 2 }
  countEvenPaintedFaces b = 16 := by
  sorry

end NUMINAMATH_CALUDE_even_painted_faces_count_l3074_307412


namespace NUMINAMATH_CALUDE_mixed_fraction_product_l3074_307451

theorem mixed_fraction_product (X Y : ℤ) : 
  (5 + 1 / X : ℚ) * (Y + 1 / 2 : ℚ) = 43 →
  5 < (5 + 1 / X : ℚ) →
  (5 + 1 / X : ℚ) ≤ 5.5 →
  X = 17 ∧ Y = 8 := by
sorry

end NUMINAMATH_CALUDE_mixed_fraction_product_l3074_307451


namespace NUMINAMATH_CALUDE_sin_negative_2055_degrees_l3074_307441

theorem sin_negative_2055_degrees : 
  Real.sin ((-2055 : ℝ) * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_2055_degrees_l3074_307441


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3074_307491

/-- Given that k, -1, and b form an arithmetic sequence,
    prove that the line y = kx + b passes through (1, -2) for all k. -/
theorem line_passes_through_fixed_point (k b : ℝ) :
  ((-1) = (k + b) / 2) →
  ∀ (x y : ℝ), y = k * x + b → (x = 1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3074_307491


namespace NUMINAMATH_CALUDE_soldiers_on_first_side_l3074_307448

theorem soldiers_on_first_side (food_per_soldier_first : ℕ)
                               (food_difference : ℕ)
                               (soldier_difference : ℕ)
                               (total_food : ℕ) :
  food_per_soldier_first = 10 →
  food_difference = 2 →
  soldier_difference = 500 →
  total_food = 68000 →
  ∃ (x : ℕ), 
    x * food_per_soldier_first + 
    (x - soldier_difference) * (food_per_soldier_first - food_difference) = total_food ∧
    x = 4000 := by
  sorry

end NUMINAMATH_CALUDE_soldiers_on_first_side_l3074_307448


namespace NUMINAMATH_CALUDE_kenneth_fabric_price_l3074_307461

/-- The price Kenneth paid for an oz of fabric -/
def kenneth_price : ℝ := 40

/-- The amount of fabric Kenneth bought in oz -/
def kenneth_amount : ℝ := 700

/-- The ratio of fabric Nicholas bought compared to Kenneth -/
def nicholas_ratio : ℝ := 6

/-- The additional amount Nicholas paid compared to Kenneth -/
def price_difference : ℝ := 140000

theorem kenneth_fabric_price :
  kenneth_price * kenneth_amount * nicholas_ratio =
  kenneth_price * kenneth_amount + price_difference :=
by sorry

end NUMINAMATH_CALUDE_kenneth_fabric_price_l3074_307461


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l3074_307480

theorem painted_cube_theorem (n : ℕ) (h : n > 2) :
  (12 * (n - 2) = (n - 2)^3) ↔ (n : ℝ) = 2 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l3074_307480


namespace NUMINAMATH_CALUDE_polynomial_factoring_l3074_307431

theorem polynomial_factoring (a x y : ℝ) : a * x^2 - a * y^2 = a * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factoring_l3074_307431


namespace NUMINAMATH_CALUDE_batsman_average_increase_l3074_307475

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  total_runs : Nat
  average : Rat

/-- Calculates the increase in average for a batsman -/
def average_increase (b : Batsman) (new_runs : Nat) (new_average : Rat) : Rat :=
  new_average - b.average

/-- Theorem: The increase in the batsman's average is 3 -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
  b.innings = 16 →
  average_increase b 56 8 = 3 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_increase_l3074_307475


namespace NUMINAMATH_CALUDE_perpendicular_sum_maximized_l3074_307408

theorem perpendicular_sum_maximized (r : ℝ) (α : ℝ) :
  let s := r * (Real.sin α + Real.cos α)
  ∀ β, 0 ≤ β ∧ β ≤ 2 * Real.pi → s ≤ r * (Real.sin (Real.pi / 4) + Real.cos (Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_sum_maximized_l3074_307408


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3074_307420

/-- Given a quadratic function y = ax² + bx - 1 where a ≠ 0, 
    if the graph passes through the point (1, 1), then a + b + 1 = 3 -/
theorem quadratic_function_theorem (a b : ℝ) (ha : a ≠ 0) :
  (a * 1^2 + b * 1 - 1 = 1) → (a + b + 1 = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3074_307420


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3074_307468

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧  -- Four-digit positive integer
  (n / 1000 ≠ n / 100 % 10) ∧ 
  (n / 1000 ≠ n / 10 % 10) ∧ 
  (n / 1000 ≠ n % 10) ∧ 
  (n / 100 % 10 ≠ n / 10 % 10) ∧ 
  (n / 100 % 10 ≠ n % 10) ∧ 
  (n / 10 % 10 ≠ n % 10) ∧  -- All digits are different
  (n / 1000 = 5 ∨ n / 100 % 10 = 5 ∨ n / 10 % 10 = 5 ∨ n % 10 = 5) ∧  -- Includes the digit 5
  (n % (n / 1000) = 0) ∧ 
  (n % (n / 100 % 10) = 0) ∧ 
  (n % (n / 10 % 10) = 0) ∧ 
  (n % (n % 10) = 0)  -- Divisible by each of its digits

theorem smallest_valid_number : 
  is_valid_number 5124 ∧ 
  ∀ m : ℕ, is_valid_number m → m ≥ 5124 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3074_307468


namespace NUMINAMATH_CALUDE_prime_factorization_of_9600_l3074_307474

theorem prime_factorization_of_9600 : 9600 = 2^6 * 3 * 5^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_of_9600_l3074_307474


namespace NUMINAMATH_CALUDE_quadratic_polynomial_integer_root_exists_l3074_307452

/-- Represents a quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluates a quadratic polynomial at x = -1 -/
def evalAtNegativeOne (p : QuadraticPolynomial) : ℤ :=
  p.a + -p.b + p.c

/-- Represents a single step change in the polynomial -/
inductive PolynomialStep
  | ChangeX : (δ : ℤ) → PolynomialStep
  | ChangeConstant : (δ : ℤ) → PolynomialStep

/-- Applies a step to a polynomial -/
def applyStep (p : QuadraticPolynomial) (step : PolynomialStep) : QuadraticPolynomial :=
  match step with
  | PolynomialStep.ChangeX δ => ⟨p.a, p.b + δ, p.c⟩
  | PolynomialStep.ChangeConstant δ => ⟨p.a, p.b, p.c + δ⟩

theorem quadratic_polynomial_integer_root_exists 
  (initial : QuadraticPolynomial)
  (final : QuadraticPolynomial)
  (h_initial : initial = ⟨1, 10, 20⟩)
  (h_final : final = ⟨1, 20, 10⟩)
  (steps : List PolynomialStep)
  (h_steps : ∀ step ∈ steps, 
    (∃ δ, step = PolynomialStep.ChangeX δ ∧ (δ = 1 ∨ δ = -1)) ∨
    (∃ δ, step = PolynomialStep.ChangeConstant δ ∧ (δ = 1 ∨ δ = -1)))
  (h_transform : final = steps.foldl applyStep initial) :
  ∃ p : QuadraticPolynomial, p ∈ initial :: (List.scanl applyStep initial steps) ∧ 
    ∃ x : ℤ, p.a * x^2 + p.b * x + p.c = 0 :=
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_integer_root_exists_l3074_307452


namespace NUMINAMATH_CALUDE_total_beignets_l3074_307436

/-- The number of beignets eaten per day -/
def beignets_per_day : ℕ := 3

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weeks we're considering -/
def weeks : ℕ := 16

/-- Theorem: The total number of beignets eaten in 16 weeks -/
theorem total_beignets : beignets_per_day * days_in_week * weeks = 336 := by
  sorry

end NUMINAMATH_CALUDE_total_beignets_l3074_307436


namespace NUMINAMATH_CALUDE_brick_width_proof_l3074_307467

/-- The width of a brick that satisfies the given conditions --/
def brick_width : ℝ := 11.25

theorem brick_width_proof (wall_volume : ℝ) (brick_length : ℝ) (brick_height : ℝ) (num_bricks : ℕ) 
  (h1 : wall_volume = 800 * 600 * 22.5)
  (h2 : ∀ w, brick_length * w * brick_height * num_bricks = wall_volume)
  (h3 : brick_length = 50)
  (h4 : brick_height = 6)
  (h5 : num_bricks = 3200) :
  brick_width = 11.25 := by
sorry

end NUMINAMATH_CALUDE_brick_width_proof_l3074_307467


namespace NUMINAMATH_CALUDE_jinas_mascots_l3074_307417

/-- The number of mascots Jina has -/
def total_mascots (x y z : ℕ) : ℕ := x + y + z

/-- The theorem stating the total number of Jina's mascots -/
theorem jinas_mascots :
  ∃ (x y z : ℕ),
    y = 3 * x ∧
    z = 2 * y ∧
    (x + (5/2 : ℚ) * y) / y = 3/7 ∧
    total_mascots x y z = 60 := by
  sorry


end NUMINAMATH_CALUDE_jinas_mascots_l3074_307417


namespace NUMINAMATH_CALUDE_gcf_of_2835_and_8960_l3074_307481

theorem gcf_of_2835_and_8960 : Nat.gcd 2835 8960 = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_2835_and_8960_l3074_307481


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3074_307419

/-- Given two functions f and g that intersect at (2,5) and (8,3), prove that a + c = 10 -/
theorem intersection_implies_sum (a b c d : ℝ) : 
  (∀ x, -|x - a| + b = |x - c| + d → x = 2 ∨ x = 8) →
  -|2 - a| + b = 5 →
  -|8 - a| + b = 3 →
  |2 - c| + d = 5 →
  |8 - c| + d = 3 →
  a + c = 10 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3074_307419


namespace NUMINAMATH_CALUDE_flowers_per_pot_l3074_307484

/-- Given 141 pots and 10011 flowers in total, prove that each pot contains 71 flowers. -/
theorem flowers_per_pot (total_pots : ℕ) (total_flowers : ℕ) (h1 : total_pots = 141) (h2 : total_flowers = 10011) :
  total_flowers / total_pots = 71 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_pot_l3074_307484


namespace NUMINAMATH_CALUDE_x_is_25_percent_greater_than_88_l3074_307410

theorem x_is_25_percent_greater_than_88 (x : ℝ) : 
  x = 88 * (1 + 0.25) → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_x_is_25_percent_greater_than_88_l3074_307410


namespace NUMINAMATH_CALUDE_pinecrest_academy_ratio_l3074_307427

theorem pinecrest_academy_ratio (j s : ℕ) (h1 : 3 * s = 6 * j) : s / j = 1 / 2 := by
  sorry

#check pinecrest_academy_ratio

end NUMINAMATH_CALUDE_pinecrest_academy_ratio_l3074_307427


namespace NUMINAMATH_CALUDE_reciprocal_problem_l3074_307428

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 5) : 200 * (1 / x) = 320 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l3074_307428


namespace NUMINAMATH_CALUDE_no_prime_of_form_3811_11_l3074_307489

def a (n : ℕ) : ℕ := 3 * 10^(n+1) + 8 * 10^n + (10^n - 1) / 9

theorem no_prime_of_form_3811_11 (n : ℕ) (h : n ≥ 1) : ¬ Nat.Prime (a n) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_of_form_3811_11_l3074_307489


namespace NUMINAMATH_CALUDE_parabola_chord_length_squared_l3074_307476

/-- Given a parabola y = 3x^2 + 4x + 2, with points C and D on the parabola,
    and the origin as the midpoint of CD, and the slope of the tangent at C is 10,
    prove that the square of the length of CD is 8. -/
theorem parabola_chord_length_squared (C D : ℝ × ℝ) : 
  (∃ (x y : ℝ), C = (x, y) ∧ D = (-x, -y)) →  -- Origin is midpoint of CD
  (C.2 = 3 * C.1^2 + 4 * C.1 + 2) →  -- C is on the parabola
  (D.2 = 3 * D.1^2 + 4 * D.1 + 2) →  -- D is on the parabola
  (6 * C.1 + 4 = 10) →  -- Slope of tangent at C is 10
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_squared_l3074_307476


namespace NUMINAMATH_CALUDE_linear_function_difference_l3074_307426

-- Define the properties of the linear function g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, ∃ a b : ℝ, g x = a * x + b) ∧ 
  (∀ d : ℝ, g (d + 2) - g d = 4)

-- State the theorem
theorem linear_function_difference 
  (g : ℝ → ℝ) 
  (h : g_properties g) : 
  g 4 - g 8 = -8 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_difference_l3074_307426


namespace NUMINAMATH_CALUDE_grandfather_grandson_age_relation_l3074_307437

theorem grandfather_grandson_age_relation :
  ∀ (grandfather_age grandson_age : ℕ) (years : ℕ),
    50 < grandfather_age →
    grandfather_age < 90 →
    grandfather_age = 31 * grandson_age →
    (grandfather_age + years = 7 * (grandson_age + years)) →
    years = 8 :=
by sorry

end NUMINAMATH_CALUDE_grandfather_grandson_age_relation_l3074_307437


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3074_307407

/-- The sum of the infinite series ∑(1/(n(n+3))) for n from 1 to infinity is equal to 11/18. -/
theorem infinite_series_sum : ∑' (n : ℕ), 1 / (n * (n + 3 : ℝ)) = 11 / 18 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3074_307407


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l3074_307488

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l3074_307488


namespace NUMINAMATH_CALUDE_pages_difference_l3074_307439

theorem pages_difference (total_pages book_length first_day fourth_day : ℕ) : 
  book_length = 354 → 
  first_day = 63 → 
  fourth_day = 29 → 
  total_pages = 4 → 
  (book_length - (first_day + 2 * first_day + fourth_day)) - 2 * first_day = 10 := by
  sorry

end NUMINAMATH_CALUDE_pages_difference_l3074_307439


namespace NUMINAMATH_CALUDE_vector_collinearity_l3074_307483

/-- Given vectors a, b, and c in ℝ², prove that if b - a is collinear with c, then the n-coordinate of b equals -3. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (n : ℝ) :
  a = (1, 2) →
  b = (n, 3) →
  c = (4, -1) →
  ∃ (k : ℝ), (b.1 - a.1, b.2 - a.2) = (k * c.1, k * c.2) →
  n = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l3074_307483


namespace NUMINAMATH_CALUDE_bobs_walking_rate_l3074_307432

/-- Proves that Bob's walking rate is 3 miles per hour given the conditions of the problem -/
theorem bobs_walking_rate (total_distance : ℝ) (yolanda_rate : ℝ) (bob_distance : ℝ) :
  total_distance = 17 →
  yolanda_rate = 3 →
  bob_distance = 8 →
  ∃ (bob_rate : ℝ), bob_rate = 3 ∧ bob_rate * (total_distance / (yolanda_rate + bob_rate) - 1) = bob_distance :=
by sorry

end NUMINAMATH_CALUDE_bobs_walking_rate_l3074_307432


namespace NUMINAMATH_CALUDE_magician_decks_l3074_307400

-- Define the problem parameters
def price_per_deck : ℕ := 2
def total_earnings : ℕ := 4
def decks_left : ℕ := 3

-- Define the theorem
theorem magician_decks : 
  ∃ (initial_decks : ℕ), 
    initial_decks * price_per_deck - total_earnings = decks_left * price_per_deck ∧ 
    initial_decks = 5 := by
  sorry

end NUMINAMATH_CALUDE_magician_decks_l3074_307400


namespace NUMINAMATH_CALUDE_min_value_of_f_over_x_range_of_a_l3074_307458

-- Define the function f
def f (a x : ℝ) := x^2 - 2*a*x - 1 + a

-- Part 1
theorem min_value_of_f_over_x (x : ℝ) (hx : x > 0) :
  ∃ (min : ℝ), min = -2 ∧ ∀ y : ℝ, y > 0 → (f 2 y) / y ≥ min :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x ≤ a) ↔ a ∈ Set.Ici (3/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_over_x_range_of_a_l3074_307458


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3074_307415

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 0, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 0, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = min) ∧
    max = 3 ∧ min = -17 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3074_307415


namespace NUMINAMATH_CALUDE_optimal_chair_removal_l3074_307494

theorem optimal_chair_removal (chairs_per_row : ℕ) (total_chairs : ℕ) (expected_participants : ℕ) 
  (h1 : chairs_per_row = 15)
  (h2 : total_chairs = 225)
  (h3 : expected_participants = 140) :
  let chairs_to_remove := 75
  let remaining_chairs := total_chairs - chairs_to_remove
  (remaining_chairs % chairs_per_row = 0) ∧ 
  (remaining_chairs ≥ expected_participants) ∧
  (∀ n : ℕ, n < chairs_to_remove → 
    (total_chairs - n) % chairs_per_row ≠ 0 ∨ 
    (total_chairs - n < expected_participants)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_chair_removal_l3074_307494


namespace NUMINAMATH_CALUDE_secret_spread_l3074_307469

/-- Represents the number of people each person tells the secret to on a given day -/
def tell_count (day : Nat) : Nat :=
  match day with
  | 1 => 1  -- Monday: Jessica tells 1 friend
  | 2 => 2  -- Tuesday
  | 3 => 2  -- Wednesday
  | 4 => 1  -- Thursday
  | _ => 2  -- Friday to Monday

/-- Calculates the total number of people knowing the secret after a given number of days -/
def total_knowing (days : Nat) : Nat :=
  match days with
  | 0 => 1  -- Only Jessica knows on day 0
  | n + 1 => total_knowing n + (total_knowing n - total_knowing (n - 1)) * tell_count (n + 1)

/-- The theorem stating that after 8 days, 132 people will know the secret -/
theorem secret_spread : total_knowing 8 = 132 := by
  sorry


end NUMINAMATH_CALUDE_secret_spread_l3074_307469


namespace NUMINAMATH_CALUDE_prime_residue_theorem_l3074_307493

/-- Definition of suitable triple -/
def suitable (p : ℕ) (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  a % p ≠ b % p ∧ b % p ≠ c % p ∧ a % p ≠ c % p

/-- Definition of f_k function -/
def f_k (p k a b c : ℕ) : ℤ :=
  a * (b - c)^(p - k) + b * (c - a)^(p - k) + c * (a - b)^(p - k)

theorem prime_residue_theorem (p : ℕ) (hp : p.Prime) (hp11 : p ≥ 11) :
  (∃ a b c : ℕ, suitable p a b c ∧ (p : ℤ) ∣ f_k p 2 a b c) ∧
  (∀ a b c : ℕ, suitable p a b c → (p : ℤ) ∣ f_k p 2 a b c →
    (∃ k : ℕ, k ≥ 3 ∧ ¬((p : ℤ) ∣ f_k p k a b c))) ∧
  (∀ a b c : ℕ, suitable p a b c → (p : ℤ) ∣ f_k p 2 a b c →
    (∀ k : ℕ, k ≥ 3 → k < 4 → (p : ℤ) ∣ f_k p k a b c)) :=
sorry

end NUMINAMATH_CALUDE_prime_residue_theorem_l3074_307493


namespace NUMINAMATH_CALUDE_cone_vertex_angle_l3074_307447

theorem cone_vertex_angle (l r : ℝ) (h : l > 0) (h2 : r > 0) : 
  (2 * π * l / 3 = 2 * π * r) → 
  (2 * Real.arcsin (1 / 3) : ℝ) = 2 * Real.arcsin (r / l) := by
sorry

end NUMINAMATH_CALUDE_cone_vertex_angle_l3074_307447


namespace NUMINAMATH_CALUDE_sum_inequality_l3074_307445

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) :
  a / (a^3 + b*c) + b / (b^3 + a*c) + c / (c^3 + a*b) > 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3074_307445


namespace NUMINAMATH_CALUDE_psychological_survey_selection_l3074_307486

theorem psychological_survey_selection (boys girls selected : ℕ) : 
  boys = 4 → girls = 2 → selected = 4 →
  (Nat.choose (boys + girls) selected) - (Nat.choose boys selected) = 14 :=
by sorry

end NUMINAMATH_CALUDE_psychological_survey_selection_l3074_307486


namespace NUMINAMATH_CALUDE_stock_percentage_change_l3074_307487

theorem stock_percentage_change 
  (initial_value : ℝ) 
  (day1_decrease_rate : ℝ) 
  (day2_increase_rate : ℝ) 
  (h1 : day1_decrease_rate = 0.3) 
  (h2 : day2_increase_rate = 0.4) : 
  (initial_value - (initial_value * (1 - day1_decrease_rate) * (1 + day2_increase_rate))) / initial_value = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_change_l3074_307487


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l3074_307411

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l3074_307411


namespace NUMINAMATH_CALUDE_fathers_age_l3074_307449

/-- Given information about Sebastian, his sister, and their father's ages, prove the father's current age. -/
theorem fathers_age (sebastian_age : ℕ) (age_difference : ℕ) (years_ago : ℕ) (fraction : ℚ) : 
  sebastian_age = 40 →
  age_difference = 10 →
  years_ago = 5 →
  fraction = 3/4 →
  (sebastian_age - years_ago + (sebastian_age - age_difference - years_ago) : ℚ) = 
    fraction * (sebastian_age - years_ago + (sebastian_age - age_difference - years_ago) + years_ago) →
  sebastian_age - years_ago + (sebastian_age - age_difference - years_ago) + years_ago = 85 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l3074_307449


namespace NUMINAMATH_CALUDE_jesse_blocks_theorem_l3074_307478

/-- The number of building blocks Jesse started with --/
def total_blocks : ℕ := sorry

/-- The number of blocks used for the cityscape --/
def cityscape_blocks : ℕ := 80

/-- The number of blocks used for the farmhouse --/
def farmhouse_blocks : ℕ := 123

/-- The number of blocks used for the zoo --/
def zoo_blocks : ℕ := 95

/-- The number of blocks used for the first fenced-in area --/
def fence1_blocks : ℕ := 57

/-- The number of blocks used for the second fenced-in area --/
def fence2_blocks : ℕ := 43

/-- The number of blocks used for the third fenced-in area --/
def fence3_blocks : ℕ := 62

/-- The number of blocks borrowed by Jesse's friend --/
def borrowed_blocks : ℕ := 35

/-- The number of blocks Jesse had left over --/
def leftover_blocks : ℕ := 84

/-- Theorem stating that the total number of blocks Jesse started with is equal to the sum of all blocks used in constructions, blocks left over, and blocks borrowed by his friend --/
theorem jesse_blocks_theorem : 
  total_blocks = cityscape_blocks + farmhouse_blocks + zoo_blocks + 
                 fence1_blocks + fence2_blocks + fence3_blocks + 
                 borrowed_blocks + leftover_blocks := by sorry

end NUMINAMATH_CALUDE_jesse_blocks_theorem_l3074_307478


namespace NUMINAMATH_CALUDE_least_candies_to_remove_daniel_candy_problem_l3074_307455

theorem least_candies_to_remove (total_candies : Nat) (sisters : Nat) : Nat :=
  let remainder := total_candies % sisters
  if remainder = 0 then 0 else sisters - remainder

theorem daniel_candy_problem :
  least_candies_to_remove 25 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_candies_to_remove_daniel_candy_problem_l3074_307455


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l3074_307465

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 12 := by sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l3074_307465


namespace NUMINAMATH_CALUDE_kevins_siblings_l3074_307499

-- Define the traits
inductive EyeColor
| Green
| Grey

inductive HairColor
| Red
| Brown

-- Define a child with their traits
structure Child where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor

-- Define the function to check if two children share a trait
def shareTrait (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor

-- Define the children
def Oliver : Child := ⟨"Oliver", EyeColor.Green, HairColor.Red⟩
def Kevin : Child := ⟨"Kevin", EyeColor.Grey, HairColor.Brown⟩
def Lily : Child := ⟨"Lily", EyeColor.Grey, HairColor.Red⟩
def Emma : Child := ⟨"Emma", EyeColor.Green, HairColor.Brown⟩
def Noah : Child := ⟨"Noah", EyeColor.Green, HairColor.Red⟩
def Mia : Child := ⟨"Mia", EyeColor.Green, HairColor.Brown⟩

-- Define the theorem
theorem kevins_siblings :
  (shareTrait Kevin Emma ∧ shareTrait Kevin Mia ∧ shareTrait Emma Mia) ∧
  (¬ (shareTrait Kevin Oliver ∧ shareTrait Kevin Noah ∧ shareTrait Oliver Noah)) ∧
  (¬ (shareTrait Kevin Lily ∧ shareTrait Kevin Noah ∧ shareTrait Lily Noah)) ∧
  (¬ (shareTrait Kevin Oliver ∧ shareTrait Kevin Lily ∧ shareTrait Oliver Lily)) :=
sorry

end NUMINAMATH_CALUDE_kevins_siblings_l3074_307499


namespace NUMINAMATH_CALUDE_adam_basswood_blocks_l3074_307442

/-- The number of figurines that can be created from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be created from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be created from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of butternut wood blocks Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 245

/-- Theorem stating that Adam owns 15 blocks of basswood -/
theorem adam_basswood_blocks : 
  ∃ (basswood_blocks : ℕ), 
    basswood_blocks * basswood_figurines + 
    butternut_blocks * butternut_figurines + 
    aspen_blocks * aspen_figurines = total_figurines ∧ 
    basswood_blocks = 15 := by
  sorry

end NUMINAMATH_CALUDE_adam_basswood_blocks_l3074_307442


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3074_307490

/-- A quadratic function passing through two points with constrained x values -/
def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

theorem quadratic_function_properties :
  ∃ (a b : ℝ),
    (quadratic_function a b 0 = 6) ∧
    (quadratic_function a b 1 = 5) ∧
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 →
      (quadratic_function a b x = x^2 - 2*x + 6) ∧
      (quadratic_function a b x ≥ 5) ∧
      (quadratic_function a b x ≤ 14) ∧
      (quadratic_function a b 1 = 5) ∧
      (quadratic_function a b (-2) = 14)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3074_307490


namespace NUMINAMATH_CALUDE_max_value_theorem_l3074_307495

theorem max_value_theorem (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 8 + 6 * y * z ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3074_307495


namespace NUMINAMATH_CALUDE_unique_prime_triple_sum_cube_ratio_l3074_307425

theorem unique_prime_triple_sum_cube_ratio (p q r : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧   -- p, q, r are prime
  p < q ∧ q < r ∧                             -- p < q < r
  (p^3 + q^3 + r^3) / (p + q + r) = 249 ∧     -- given equation
  (∀ p' q' r' : ℕ, 
    Nat.Prime p' ∧ Nat.Prime q' ∧ Nat.Prime r' ∧
    p' < q' ∧ q' < r' ∧
    (p'^3 + q'^3 + r'^3) / (p' + q' + r') = 249 →
    p' = p ∧ q' = q ∧ r' = r) →               -- uniqueness condition
  r = 19 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_triple_sum_cube_ratio_l3074_307425


namespace NUMINAMATH_CALUDE_savings_calculation_l3074_307409

theorem savings_calculation (initial_savings : ℝ) : 
  let february_spend := 0.20 * initial_savings
  let march_spend := 0.40 * initial_savings
  let april_spend := 1500
  let remaining := 2900
  february_spend + march_spend + april_spend + remaining = initial_savings →
  initial_savings = 11000 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l3074_307409


namespace NUMINAMATH_CALUDE_union_equals_reals_l3074_307403

def M : Set ℝ := {x : ℝ | |x| > 2}
def N : Set ℝ := {x : ℝ | x < 3}

theorem union_equals_reals : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_equals_reals_l3074_307403


namespace NUMINAMATH_CALUDE_absolute_value_sum_difference_l3074_307462

theorem absolute_value_sum_difference (x y : ℚ) 
  (hx : |x| = 9) (hy : |y| = 5) : 
  ((x < 0 ∧ y > 0) → x + y = -4) ∧
  (|x + y| = x + y → (x - y = 4 ∨ x - y = 14)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_difference_l3074_307462


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l3074_307466

theorem isosceles_triangle_angles (a b c : ℝ) : 
  -- The triangle is isosceles
  (a = b ∨ b = c ∨ a = c) →
  -- One of the interior angles is 50°
  (a = 50 ∨ b = 50 ∨ c = 50) →
  -- The sum of interior angles in a triangle is 180°
  a + b + c = 180 →
  -- The other two angles are either (65°, 65°) or (80°, 50°)
  ((a = 65 ∧ b = 65 ∧ c = 50) ∨ 
   (a = 65 ∧ c = 65 ∧ b = 50) ∨ 
   (b = 65 ∧ c = 65 ∧ a = 50) ∨
   (a = 80 ∧ b = 50 ∧ c = 50) ∨ 
   (a = 50 ∧ b = 80 ∧ c = 50) ∨ 
   (a = 50 ∧ b = 50 ∧ c = 80)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l3074_307466


namespace NUMINAMATH_CALUDE_thirteen_fifth_power_mod_seven_l3074_307421

theorem thirteen_fifth_power_mod_seven : (13^5 : ℤ) ≡ 6 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_thirteen_fifth_power_mod_seven_l3074_307421


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l3074_307454

/-- Given a triangle ABC with B > A, prove that C₁ - C₂ = B - A,
    where C₁ and C₂ are parts of angle C divided by the altitude,
    and C₂ is adjacent to side a. -/
theorem triangle_angle_relation (A B C C₁ C₂ : Real) : 
  B > A → 
  C = C₁ + C₂ → 
  A + B + C = Real.pi → 
  C₁ - C₂ = B - A := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l3074_307454


namespace NUMINAMATH_CALUDE_range_of_a_l3074_307471

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4*x + a ≤ 0

theorem range_of_a (a : ℝ) (hp : prop_p a) (hq : prop_q a) :
  Real.exp 1 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3074_307471


namespace NUMINAMATH_CALUDE_f_above_x_axis_iff_valid_a_range_l3074_307463

/-- The function f(x) = (a^2 - 3a + 2)x^2 + (a - 1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 3*a + 2)*x^2 + (a - 1)*x + 2

/-- The graph of f(x) is above the x-axis -/
def above_x_axis (a : ℝ) : Prop := ∀ x, f a x > 0

/-- The range of values for a -/
def valid_a_range (a : ℝ) : Prop := a > 15/7 ∨ a ≤ 1

theorem f_above_x_axis_iff_valid_a_range :
  ∀ a : ℝ, above_x_axis a ↔ valid_a_range a := by sorry

end NUMINAMATH_CALUDE_f_above_x_axis_iff_valid_a_range_l3074_307463


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_calculation_l3074_307460

/-- The cost of the Ferris wheel ride -/
def ferris_wheel_cost : ℝ := 2.0

/-- The cost of the roller coaster ride -/
def roller_coaster_cost : ℝ := 7.0

/-- The discount for multiple rides -/
def multiple_ride_discount : ℝ := 1.0

/-- The value of the newspaper coupon -/
def coupon_value : ℝ := 1.0

/-- The total number of tickets needed for both rides -/
def total_tickets_needed : ℝ := 7.0

theorem ferris_wheel_cost_calculation :
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - coupon_value = total_tickets_needed :=
sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_calculation_l3074_307460


namespace NUMINAMATH_CALUDE_max_m_value_l3074_307456

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, (3 / a + 1 / b ≥ m / (a + 3 * b))) → m ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l3074_307456


namespace NUMINAMATH_CALUDE_vidya_age_difference_l3074_307473

/-- Proves that the difference between Vidya's mother's age and three times Vidya's age is 5 years -/
theorem vidya_age_difference (vidya_age : ℕ) (mother_age : ℕ) 
  (h1 : vidya_age = 13)
  (h2 : mother_age = 44) :
  mother_age - 3 * vidya_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_vidya_age_difference_l3074_307473


namespace NUMINAMATH_CALUDE_terry_future_age_relation_nora_current_age_l3074_307444

/-- Nora's current age -/
def nora_age : ℕ := sorry

/-- Terry's current age -/
def terry_age : ℕ := 30

/-- In 10 years, Terry will be 4 times Nora's current age -/
theorem terry_future_age_relation : terry_age + 10 = 4 * nora_age := sorry

theorem nora_current_age : nora_age = 10 := by sorry

end NUMINAMATH_CALUDE_terry_future_age_relation_nora_current_age_l3074_307444


namespace NUMINAMATH_CALUDE_original_number_proof_l3074_307430

theorem original_number_proof (x : ℚ) : (1 / x) - 2 = 5 / 2 → x = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3074_307430


namespace NUMINAMATH_CALUDE_x_value_proof_l3074_307413

theorem x_value_proof (x : ℚ) (h : 3/5 - 1/4 = 4/x) : x = 80/7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3074_307413


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_absolute_values_l3074_307482

theorem min_value_of_sum_of_absolute_values :
  ∃ (m : ℝ), (∀ x : ℝ, m ≤ |x + 2| + |x - 2| + |x - 1|) ∧ (∃ y : ℝ, m = |y + 2| + |y - 2| + |y - 1|) ∧ m = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_absolute_values_l3074_307482


namespace NUMINAMATH_CALUDE_like_terms_sum_exponents_l3074_307472

/-- Two monomials are like terms if they have the same variables raised to the same powers. -/
def are_like_terms (a b : ℕ) (m n : ℤ) : Prop :=
  m + 1 = 1 ∧ 3 = n

/-- If 5x^(m+1)y^3 and -3xy^n are like terms, then m + n = 3. -/
theorem like_terms_sum_exponents (m n : ℤ) :
  are_like_terms 5 3 m n → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_exponents_l3074_307472


namespace NUMINAMATH_CALUDE_ashok_marks_average_l3074_307479

/-- Given a student's average marks and the marks in the last subject, 
    calculate the average marks in the remaining subjects. -/
def average_remaining_subjects (total_subjects : ℕ) (overall_average : ℚ) (last_subject_marks : ℕ) : ℚ :=
  ((overall_average * total_subjects) - last_subject_marks) / (total_subjects - 1)

/-- Theorem stating that given the conditions in the problem, 
    the average of marks in the first 5 subjects is 74. -/
theorem ashok_marks_average : 
  let total_subjects : ℕ := 6
  let overall_average : ℚ := 75
  let last_subject_marks : ℕ := 80
  average_remaining_subjects total_subjects overall_average last_subject_marks = 74 := by
  sorry

#eval average_remaining_subjects 6 75 80

end NUMINAMATH_CALUDE_ashok_marks_average_l3074_307479


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3074_307477

theorem expression_simplification_and_evaluation :
  let x : ℝ := 1 - 3 * Real.tan (π / 4)
  (1 / (3 - x) - (x^2 + 6*x + 9) / (x^2 + 3*x) / ((x^2 - 9) / x)) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3074_307477


namespace NUMINAMATH_CALUDE_right_triangle_area_l3074_307435

/-- The area of a right triangle with base 8 and hypotenuse 10 is 24 square units. -/
theorem right_triangle_area : 
  ∀ (base height hypotenuse : ℝ),
  base = 8 →
  hypotenuse = 10 →
  base ^ 2 + height ^ 2 = hypotenuse ^ 2 →
  (1 / 2) * base * height = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3074_307435


namespace NUMINAMATH_CALUDE_bisecting_cross_section_dihedral_angle_l3074_307464

/-- Regular tetrahedron with specific dimensions -/
structure RegularTetrahedron where
  -- Base side length
  base_side : ℝ
  -- Side edge length
  side_edge : ℝ
  -- Assumption that base_side = 1 and side_edge = 2
  base_side_eq_one : base_side = 1
  side_edge_eq_two : side_edge = 2

/-- Cross-section that bisects the tetrahedron's volume -/
structure BisectingCrossSection (t : RegularTetrahedron) where
  -- The cross-section passes through edge AB of the base
  passes_through_base_edge : Prop

/-- Dihedral angle between the cross-section and the base -/
def dihedralAngle (t : RegularTetrahedron) (cs : BisectingCrossSection t) : ℝ :=
  sorry -- Definition of dihedral angle

/-- Main theorem -/
theorem bisecting_cross_section_dihedral_angle 
  (t : RegularTetrahedron) (cs : BisectingCrossSection t) : 
  Real.cos (dihedralAngle t cs) = 2 * Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_cross_section_dihedral_angle_l3074_307464


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3074_307459

theorem tangent_line_to_circle (m : ℝ) :
  (∀ x y : ℝ, 3 * x - 4 * y - 6 = 0 →
    (x^2 + y^2 - 2*y + m = 0 →
      ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 2*y₀ + m = 0 ∧
        3 * x₀ - 4 * y₀ - 6 = 0 ∧
        ∀ (x' y' : ℝ), x'^2 + y'^2 - 2*y' + m = 0 →
          (x' - x₀)^2 + (y' - y₀)^2 > 0)) →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3074_307459


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l3074_307423

/-- Given a rectangle with width w, length 10, and perimeter 30, 
    prove that the ratio of width to length is 1:2 -/
theorem rectangle_width_length_ratio 
  (w : ℝ) 
  (h1 : w > 0)
  (h2 : 2 * w + 2 * 10 = 30) : 
  w / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l3074_307423


namespace NUMINAMATH_CALUDE_abs_eq_piecewise_l3074_307443

theorem abs_eq_piecewise (x : ℝ) : |x| = if x ≥ 0 then x else -x := by sorry

end NUMINAMATH_CALUDE_abs_eq_piecewise_l3074_307443


namespace NUMINAMATH_CALUDE_simplify_expression_l3074_307453

theorem simplify_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) :
  (1 - 2 / (x - 1)) * ((x^2 - x) / (x^2 - 6*x + 9)) = x / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3074_307453


namespace NUMINAMATH_CALUDE_dwarf_truth_count_l3074_307457

theorem dwarf_truth_count :
  ∀ (n : ℕ) (vanilla chocolate fruit : ℕ),
    n = 10 →
    vanilla = n →
    chocolate = n / 2 →
    fruit = 1 →
    ∃ (truthful : ℕ),
      truthful = 4 ∧
      truthful + (n - truthful) = n ∧
      truthful + 2 * (n - truthful) = vanilla + chocolate + fruit :=
by sorry

end NUMINAMATH_CALUDE_dwarf_truth_count_l3074_307457


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3074_307416

theorem simplify_fraction_product : (125 : ℚ) / 5000 * 40 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3074_307416


namespace NUMINAMATH_CALUDE_fiona_id_is_17_l3074_307405

/-- A structure representing a math club member with an ID number -/
structure MathClubMember where
  name : String
  id : Nat

/-- A predicate to check if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

/-- A predicate to check if a number is a two-digit number -/
def isTwoDigit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

theorem fiona_id_is_17 
  (dan emily fiona : MathClubMember)
  (h1 : isPrime dan.id ∧ isPrime emily.id ∧ isPrime fiona.id)
  (h2 : isTwoDigit dan.id ∧ isTwoDigit emily.id ∧ isTwoDigit fiona.id)
  (h3 : ∃ p q : Nat, dan.id < p ∧ p < q ∧ 
    (emily.id = p ∨ emily.id = q) ∧ 
    (fiona.id = p ∨ fiona.id = q) ∧
    isPrime p ∧ isPrime q)
  (h4 : ∃ today : Nat, emily.id + fiona.id = today ∧ today ≤ 31)
  (h5 : ∃ emilys_birthday : Nat, dan.id + fiona.id = emilys_birthday - 1 ∧ emilys_birthday ≤ 31)
  (h6 : dan.id + emily.id = (emily.id + fiona.id) + 1)
  : fiona.id = 17 := by
  sorry

end NUMINAMATH_CALUDE_fiona_id_is_17_l3074_307405


namespace NUMINAMATH_CALUDE_greatest_value_2q_minus_r_l3074_307404

theorem greatest_value_2q_minus_r : 
  ∃ (q r : ℕ+), 
    1027 = 21 * q + r ∧ 
    ∀ (q' r' : ℕ+), 1027 = 21 * q' + r' → 2 * q - r ≥ 2 * q' - r' ∧
    2 * q - r = 77 := by
  sorry

end NUMINAMATH_CALUDE_greatest_value_2q_minus_r_l3074_307404


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_sum_less_than_16_sixteen_is_smallest_l3074_307402

theorem smallest_whole_number_above_sum : ℕ → Prop :=
  fun n => (2 + 1/2 + 3 + 1/3 + 4 + 1/4 + 5 + 1/5 : ℚ) < n ∧
           ∀ m : ℕ, (2 + 1/2 + 3 + 1/3 + 4 + 1/4 + 5 + 1/5 : ℚ) < m → n ≤ m

theorem sum_less_than_16 :
  (2 + 1/2 + 3 + 1/3 + 4 + 1/4 + 5 + 1/5 : ℚ) < 16 :=
sorry

theorem sixteen_is_smallest : smallest_whole_number_above_sum 16 :=
sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_sum_less_than_16_sixteen_is_smallest_l3074_307402


namespace NUMINAMATH_CALUDE_gcd_power_of_two_minus_one_l3074_307434

theorem gcd_power_of_two_minus_one :
  Nat.gcd (2^2022 - 1) (2^2036 - 1) = 2^14 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_of_two_minus_one_l3074_307434


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3074_307406

def M : Set ℝ := {x | (x - 1)^2 < 4}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3074_307406


namespace NUMINAMATH_CALUDE_cost_of_bananas_l3074_307438

/-- The cost of bananas given the following conditions:
  * The cost of one banana is 800 won
  * The cost of one kiwi is 400 won
  * The total number of bananas and kiwis is 18
  * The total amount spent is 10,000 won
-/
theorem cost_of_bananas :
  let banana_cost : ℕ := 800
  let kiwi_cost : ℕ := 400
  let total_fruits : ℕ := 18
  let total_spent : ℕ := 10000
  ∃ (num_bananas : ℕ),
    num_bananas * banana_cost + (total_fruits - num_bananas) * kiwi_cost = total_spent ∧
    num_bananas * banana_cost = 5600 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_bananas_l3074_307438


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3074_307497

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  1/a + 1/b ≥ 3 + 2*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3074_307497
