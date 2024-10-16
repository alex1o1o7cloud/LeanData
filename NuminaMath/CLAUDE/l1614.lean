import Mathlib

namespace NUMINAMATH_CALUDE_total_travel_time_is_156_hours_l1614_161481

/-- Represents the total travel time of a car journey with specific conditions. -/
def total_travel_time (time_ngapara_zipra : ℝ) : ℝ :=
  let time_ningi_zipra : ℝ := 0.8 * time_ngapara_zipra
  let time_zipra_varnasi : ℝ := 0.75 * time_ningi_zipra
  let delay_time : ℝ := 0.25 * time_ningi_zipra
  time_ngapara_zipra + time_ningi_zipra + delay_time + time_zipra_varnasi

/-- Theorem stating that the total travel time is 156 hours given the specified conditions. -/
theorem total_travel_time_is_156_hours :
  total_travel_time 60 = 156 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_is_156_hours_l1614_161481


namespace NUMINAMATH_CALUDE_karen_grooms_six_rottweilers_l1614_161427

/-- Represents the time taken to groom different dog breeds and the total grooming time -/
structure GroomingInfo where
  rottweilerTime : ℕ
  borderCollieTime : ℕ
  chihuahuaTime : ℕ
  totalTime : ℕ
  borderCollieCount : ℕ
  chihuahuaCount : ℕ

/-- Calculates the number of Rottweilers groomed given the grooming information -/
def calculateRottweilers (info : GroomingInfo) : ℕ :=
  (info.totalTime - info.borderCollieTime * info.borderCollieCount - info.chihuahuaTime * info.chihuahuaCount) / info.rottweilerTime

/-- Theorem stating that Karen grooms 6 Rottweilers given the problem conditions -/
theorem karen_grooms_six_rottweilers (info : GroomingInfo)
  (h1 : info.rottweilerTime = 20)
  (h2 : info.borderCollieTime = 10)
  (h3 : info.chihuahuaTime = 45)
  (h4 : info.totalTime = 255)
  (h5 : info.borderCollieCount = 9)
  (h6 : info.chihuahuaCount = 1) :
  calculateRottweilers info = 6 := by
  sorry


end NUMINAMATH_CALUDE_karen_grooms_six_rottweilers_l1614_161427


namespace NUMINAMATH_CALUDE_rental_shop_problem_l1614_161471

/-- Rental shop problem -/
theorem rental_shop_problem 
  (first_hour_rate : ℝ) 
  (additional_hour_rate : ℝ)
  (sales_tax_rate : ℝ)
  (total_paid : ℝ)
  (h : ℕ)
  (h_def : h = (total_paid / (1 + sales_tax_rate) - first_hour_rate) / additional_hour_rate)
  (first_hour_rate_def : first_hour_rate = 25)
  (additional_hour_rate_def : additional_hour_rate = 10)
  (sales_tax_rate_def : sales_tax_rate = 0.08)
  (total_paid_def : total_paid = 125) :
  h + 1 = 10 := by
sorry


end NUMINAMATH_CALUDE_rental_shop_problem_l1614_161471


namespace NUMINAMATH_CALUDE_marbles_remaining_example_l1614_161494

/-- The number of marbles remaining after distribution -/
def marblesRemaining (chris ryan alex : ℕ) : ℕ :=
  let total := chris + ryan + alex
  let chrisShare := total / 4
  let ryanShare := total / 4
  let alexShare := total / 3
  total - (chrisShare + ryanShare + alexShare)

/-- Theorem stating the number of marbles remaining in the specific scenario -/
theorem marbles_remaining_example : marblesRemaining 12 28 18 = 11 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remaining_example_l1614_161494


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_a_value_l1614_161419

/-- Trapezoid inscribed in a parabola -/
structure InscribedTrapezoid where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_a_gt_b : a > b
  h_sides_equal : 2*a + 2*b = 3/4 + Real.sqrt ((a - b)^2 + (a^2 - b^2)^2)
  h_ab : Real.sqrt ((a - b)^2 + (a^2 - b^2)^2) = 3/4

theorem inscribed_trapezoid_a_value (t : InscribedTrapezoid) : t.a = 27/40 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_a_value_l1614_161419


namespace NUMINAMATH_CALUDE_geoffrey_games_l1614_161400

/-- The number of games Geoffrey bought -/
def num_games : ℕ := sorry

/-- The amount of money Geoffrey had before his birthday -/
def initial_money : ℕ := sorry

/-- The cost of each game -/
def game_cost : ℕ := 35

/-- The amount of money Geoffrey received from his grandmother -/
def grandmother_gift : ℕ := 20

/-- The amount of money Geoffrey received from his aunt -/
def aunt_gift : ℕ := 25

/-- The amount of money Geoffrey received from his uncle -/
def uncle_gift : ℕ := 30

/-- The total amount of money Geoffrey has after receiving gifts -/
def total_money : ℕ := 125

/-- The amount of money Geoffrey has left after buying games -/
def money_left : ℕ := 20

theorem geoffrey_games :
  num_games = 3 ∧
  initial_money + grandmother_gift + aunt_gift + uncle_gift = total_money ∧
  total_money - money_left = num_games * game_cost :=
sorry

end NUMINAMATH_CALUDE_geoffrey_games_l1614_161400


namespace NUMINAMATH_CALUDE_a_divisible_by_133_l1614_161442

/-- Sequence definition -/
def a (n : ℕ) : ℕ := 11^(n+2) + 12^(2*n+1)

/-- Main theorem: a_n is divisible by 133 for all n ≥ 0 -/
theorem a_divisible_by_133 (n : ℕ) : 133 ∣ a n := by sorry

end NUMINAMATH_CALUDE_a_divisible_by_133_l1614_161442


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1614_161450

theorem sphere_surface_area (v : ℝ) (h : v = 72 * Real.pi) :
  ∃ (r : ℝ), v = (4 / 3) * Real.pi * r^3 ∧ 4 * Real.pi * r^2 = 4 * Real.pi * (2916 ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1614_161450


namespace NUMINAMATH_CALUDE_impossible_arrangement_l1614_161408

/-- Represents the type of student: Knight (always tells the truth) or Liar (always lies) -/
inductive StudentType
| Knight
| Liar

/-- Represents a desk with two students -/
structure Desk where
  student1 : StudentType
  student2 : StudentType

/-- The initial arrangement of students -/
def initial_arrangement (desks : List Desk) : Prop :=
  desks.length = 13 ∧ 
  ∀ d ∈ desks, (d.student1 = StudentType.Knight ∧ d.student2 = StudentType.Liar) ∨
                (d.student1 = StudentType.Liar ∧ d.student2 = StudentType.Knight)

/-- The final arrangement of students -/
def final_arrangement (desks : List Desk) : Prop :=
  desks.length = 13 ∧
  ∀ d ∈ desks, d.student1 = d.student2

/-- Theorem stating the impossibility of the final arrangement -/
theorem impossible_arrangement :
  ∀ (initial_desks final_desks : List Desk),
    initial_arrangement initial_desks →
    ¬(final_arrangement final_desks) :=
by sorry


end NUMINAMATH_CALUDE_impossible_arrangement_l1614_161408


namespace NUMINAMATH_CALUDE_shaded_area_is_700_l1614_161464

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the large square -/
def squareVertices : List Point := [
  ⟨0, 0⟩, ⟨40, 0⟩, ⟨40, 40⟩, ⟨0, 40⟩
]

/-- The vertices of the shaded polygon -/
def shadedVertices : List Point := [
  ⟨0, 0⟩, ⟨10, 0⟩, ⟨40, 30⟩, ⟨40, 40⟩, ⟨30, 40⟩, ⟨0, 10⟩
]

/-- The side length of the large square -/
def squareSideLength : ℝ := 40

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)|

/-- Calculate the area of the shaded region -/
def shadedArea : ℝ :=
  squareSideLength ^ 2 -
  (triangleArea ⟨10, 0⟩ ⟨40, 0⟩ ⟨40, 30⟩ +
   triangleArea ⟨0, 10⟩ ⟨30, 40⟩ ⟨0, 40⟩)

/-- Theorem: The area of the shaded region is 700 square units -/
theorem shaded_area_is_700 : shadedArea = 700 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_700_l1614_161464


namespace NUMINAMATH_CALUDE_max_value_of_a_l1614_161472

-- Define the operation
def matrix_op (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the theorem
theorem max_value_of_a :
  (∀ x : ℝ, matrix_op (x - 1) (a - 2) (a + 1) x ≥ 1) →
  (∀ b : ℝ, (∀ x : ℝ, matrix_op (x - 1) (b - 2) (b + 1) x ≥ 1) → b ≤ 3/2) ∧
  (∃ x : ℝ, matrix_op (x - 1) (3/2 - 2) (3/2 + 1) x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1614_161472


namespace NUMINAMATH_CALUDE_inequality_proof_l1614_161423

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ≤ a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1/3) ∧
  a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1/3) ≤ (a - b)^2 + (b - c)^2 + (c - a)^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1614_161423


namespace NUMINAMATH_CALUDE_equivalent_statements_l1614_161443

theorem equivalent_statements (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_statements_l1614_161443


namespace NUMINAMATH_CALUDE_alice_bob_meet_l1614_161401

/-- Represents the number of points on the circle -/
def numPoints : ℕ := 18

/-- Represents Alice's movement per turn (clockwise) -/
def aliceMove : ℕ := 7

/-- Represents Bob's movement per turn (counterclockwise) -/
def bobMove : ℕ := 13

/-- Calculates the effective clockwise movement of a player given their movement -/
def effectiveMove (move : ℕ) : ℕ :=
  move % numPoints

/-- Calculates the relative movement between Alice and Bob in one turn -/
def relativeMove : ℤ :=
  (effectiveMove aliceMove : ℤ) - (effectiveMove (numPoints - bobMove) : ℤ)

/-- The number of turns it takes for Alice and Bob to meet -/
def numTurns : ℕ := 9

theorem alice_bob_meet :
  (numTurns : ℤ) * relativeMove % (numPoints : ℤ) = 0 :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l1614_161401


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1614_161447

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = 4/5 + 3/5 * Complex.I →
  Complex.exp (φ * Complex.I) = -5/13 + 12/13 * Complex.I →
  Real.sin (θ + φ) = 84/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1614_161447


namespace NUMINAMATH_CALUDE_arctan_equation_equivalence_l1614_161452

theorem arctan_equation_equivalence (x : ℝ) :
  Real.arctan (1 / x) + Real.arctan (1 / x^5) = π / 6 →
  x^6 - Real.sqrt 3 * x^5 - Real.sqrt 3 * x - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_equivalence_l1614_161452


namespace NUMINAMATH_CALUDE_largest_fraction_l1614_161444

theorem largest_fraction : 
  let f1 := 5 / 11
  let f2 := 6 / 13
  let f3 := 18 / 37
  let f4 := 101 / 202
  let f5 := 200 / 399
  f5 > f1 ∧ f5 > f2 ∧ f5 > f3 ∧ f5 > f4 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l1614_161444


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1614_161458

def expression : ℕ :=
  (List.range 17).foldl (λ acc i => acc * (2^(2^i) + 1)) 3 + 1

theorem units_digit_of_expression :
  expression % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1614_161458


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_zero_l1614_161437

-- Define the set of positive real numbers
def RealPos := {x : ℝ | x > 0}

-- Define the inequality condition
def SatisfiesInequality (f : RealPos → ℝ) (α β : ℝ) : Prop :=
  ∀ x y : RealPos, f x * f y ≥ 
    (y^α / (x^α + x^β)) * (f x)^2 + (x^β / (y^α + y^β)) * (f y)^2

-- State the theorem
theorem function_satisfying_inequality_is_zero 
  (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  ∀ f : RealPos → ℝ, SatisfiesInequality f α β → 
    (∀ x : RealPos, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_zero_l1614_161437


namespace NUMINAMATH_CALUDE_age_difference_l1614_161460

/-- Given the ages of Mandy, her brother, and her sister, prove the age difference between Mandy and her sister. -/
theorem age_difference (mandy_age brother_age sister_age : ℕ) 
  (h1 : mandy_age = 3)
  (h2 : brother_age = 4 * mandy_age)
  (h3 : sister_age = brother_age - 5) :
  sister_age - mandy_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1614_161460


namespace NUMINAMATH_CALUDE_max_popsicles_for_8_dollars_l1614_161415

/-- Represents the different popsicle purchase options -/
inductive PopsicleOption
  | Single
  | Box3
  | Box5

/-- Returns the cost of a given popsicle option -/
def cost (option : PopsicleOption) : ℕ :=
  match option with
  | .Single => 1
  | .Box3 => 2
  | .Box5 => 3

/-- Returns the number of popsicles in a given option -/
def popsicles (option : PopsicleOption) : ℕ :=
  match option with
  | .Single => 1
  | .Box3 => 3
  | .Box5 => 5

/-- Represents a purchase of popsicles -/
structure Purchase where
  singles : ℕ
  box3s : ℕ
  box5s : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.singles * cost PopsicleOption.Single +
  p.box3s * cost PopsicleOption.Box3 +
  p.box5s * cost PopsicleOption.Box5

/-- Calculates the total number of popsicles in a purchase -/
def totalPopsicles (p : Purchase) : ℕ :=
  p.singles * popsicles PopsicleOption.Single +
  p.box3s * popsicles PopsicleOption.Box3 +
  p.box5s * popsicles PopsicleOption.Box5

/-- Theorem: The maximum number of popsicles that can be purchased with $8 is 13 -/
theorem max_popsicles_for_8_dollars :
  ∀ p : Purchase, totalCost p ≤ 8 → totalPopsicles p ≤ 13 ∧
  ∃ p' : Purchase, totalCost p' = 8 ∧ totalPopsicles p' = 13 :=
sorry

end NUMINAMATH_CALUDE_max_popsicles_for_8_dollars_l1614_161415


namespace NUMINAMATH_CALUDE_apple_problem_l1614_161406

theorem apple_problem (older younger : ℕ) 
  (h1 : older - 1 = younger + 1)
  (h2 : older + 1 = 2 * (younger - 1)) :
  older + younger = 12 := by
  sorry

end NUMINAMATH_CALUDE_apple_problem_l1614_161406


namespace NUMINAMATH_CALUDE_rent_increase_problem_l1614_161433

/-- Given a group of 4 friends paying rent, where:
  - The initial average rent is $800
  - After one person's rent increases by 20%, the new average is $870
  This theorem proves that the original rent of the person whose rent increased was $1400. -/
theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800)
  (h2 : new_average = 870) (h3 : num_friends = 4) (h4 : increase_percentage = 0.2) :
  ∃ (original_rent : ℝ), 
    original_rent * (1 + increase_percentage) = 
      num_friends * new_average - (num_friends - 1) * initial_average ∧
    original_rent = 1400 :=
by sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l1614_161433


namespace NUMINAMATH_CALUDE_books_found_equals_26_l1614_161463

/-- The number of books Joan initially gathered -/
def initial_books : ℕ := 33

/-- The total number of books Joan has now -/
def total_books : ℕ := 59

/-- The number of additional books Joan found -/
def additional_books : ℕ := total_books - initial_books

theorem books_found_equals_26 : additional_books = 26 := by
  sorry

end NUMINAMATH_CALUDE_books_found_equals_26_l1614_161463


namespace NUMINAMATH_CALUDE_square_side_length_sum_l1614_161456

theorem square_side_length_sum : ∃ (a b : ℕ), a^2 + b^2 = 100 ∧ a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_sum_l1614_161456


namespace NUMINAMATH_CALUDE_intersection_M_N_l1614_161446

-- Define set M
def M : Set ℝ := {x | Real.sqrt (x + 1) ≥ 0}

-- Define set N
def N : Set ℝ := {x | x^2 + x - 2 < 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1614_161446


namespace NUMINAMATH_CALUDE_intersection_A_B_equals_open_interval_2_3_l1614_161448

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 5}

-- Define the open interval (2, 3)
def open_interval_2_3 : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B_equals_open_interval_2_3 : A ∩ B = open_interval_2_3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_equals_open_interval_2_3_l1614_161448


namespace NUMINAMATH_CALUDE_factor_expression_l1614_161465

theorem factor_expression (a b : ℝ) : 2*a^2*b - 4*a*b^2 + 2*b^3 = 2*b*(a-b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1614_161465


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l1614_161454

theorem square_rectangle_area_relation : 
  ∀ x : ℝ,
  let square_side := x - 4
  let rect_length := x - 2
  let rect_width := x + 6
  let square_area := square_side * square_side
  let rect_area := rect_length * rect_width
  rect_area = 3 * square_area →
  (∃ x₁ x₂ : ℝ, 
    (square_side = x₁ - 4 ∧ rect_length = x₁ - 2 ∧ rect_width = x₁ + 6 ∧
     square_side = x₂ - 4 ∧ rect_length = x₂ - 2 ∧ rect_width = x₂ + 6) ∧
    x₁ + x₂ = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l1614_161454


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1614_161430

/-- Given a quadratic function f(x) = x^2 + bx + c where f(-1) = f(3),
    prove that f(1) < c < f(-1) -/
theorem quadratic_inequality (b c : ℝ) : 
  let f := fun (x : ℝ) => x^2 + b*x + c
  (f (-1) = f 3) → (f 1 < c ∧ c < f (-1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1614_161430


namespace NUMINAMATH_CALUDE_hyperbola_center_l1614_161449

/-- The center of the hyperbola given by the equation (4y-6)²/6² - (5x-3)²/7² = -1 -/
theorem hyperbola_center : ∃ (h k : ℝ), 
  (∀ (x y : ℝ), (4*y - 6)^2 / 6^2 - (5*x - 3)^2 / 7^2 = -1 ↔ 
    (x - h)^2 / (7/5)^2 - (y - k)^2 / (3/2)^2 = 1) ∧ 
  h = 3/5 ∧ k = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_center_l1614_161449


namespace NUMINAMATH_CALUDE_unique_number_theorem_l1614_161476

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Generates the three numbers obtained by replacing one digit with 1 -/
def generateReplacedNumbers (n : ThreeDigitNumber) : List Nat :=
  [100 + 10 * n.tens + n.ones,
   100 * n.hundreds + 10 + n.ones,
   100 * n.hundreds + 10 * n.tens + 1]

/-- The main theorem stating that if the sum of replaced numbers is 1243,
    then the original number must be 566 -/
theorem unique_number_theorem (n : ThreeDigitNumber) :
  (generateReplacedNumbers n).sum = 1243 → n.toNat = 566 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_theorem_l1614_161476


namespace NUMINAMATH_CALUDE_annie_candy_cost_l1614_161416

/-- Calculates the total cost of candies Annie bought for her class -/
theorem annie_candy_cost (class_size : ℕ) (candies_per_classmate : ℕ) (leftover_candies : ℕ) (candy_cost : ℚ) : 
  class_size = 35 → 
  candies_per_classmate = 2 → 
  leftover_candies = 12 → 
  candy_cost = 1/10 →
  (class_size * candies_per_classmate + leftover_candies) * candy_cost = 82/10 := by
  sorry

end NUMINAMATH_CALUDE_annie_candy_cost_l1614_161416


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1614_161461

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₂ * a₆ = 4, then a₄ = 2 or a₄ = -2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) (h_prod : a 2 * a 6 = 4) : 
  a 4 = 2 ∨ a 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1614_161461


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l1614_161409

theorem square_perimeters_sum (a b : ℝ) (h1 : a ^ 2 + b ^ 2 = 145) (h2 : a ^ 2 - b ^ 2 = 25) :
  4 * Real.sqrt a ^ 2 + 4 * Real.sqrt b ^ 2 = 4 * Real.sqrt 85 + 4 * Real.sqrt 60 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l1614_161409


namespace NUMINAMATH_CALUDE_expected_votes_for_a_l1614_161474

-- Define the total number of voters (for simplicity, we'll use 100 as in the solution)
def total_voters : ℝ := 100

-- Define the percentage of Democratic voters
def dem_percentage : ℝ := 0.7

-- Define the percentage of Republican voters
def rep_percentage : ℝ := 1 - dem_percentage

-- Define the percentage of Democratic voters voting for candidate A
def dem_vote_for_a : ℝ := 0.8

-- Define the percentage of Republican voters voting for candidate A
def rep_vote_for_a : ℝ := 0.3

-- Theorem to prove
theorem expected_votes_for_a :
  (dem_percentage * dem_vote_for_a + rep_percentage * rep_vote_for_a) * 100 = 65 := by
  sorry


end NUMINAMATH_CALUDE_expected_votes_for_a_l1614_161474


namespace NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_5_l1614_161484

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem product_of_digits_not_divisible_by_5 (numbers : List ℕ) :
  numbers = [3640, 3855, 3922, 4025, 4120] →
  (∃ n ∈ numbers, ¬ is_divisible_by_5 n) →
  (∃ n ∈ numbers, ¬ is_divisible_by_5 n ∧ hundreds_digit n * tens_digit n = 18) :=
by sorry

end NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_5_l1614_161484


namespace NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l1614_161492

-- Define the types for points and lines
def Point : Type := ℝ × ℝ
def Line : Type := Point → Prop

-- Define the triangle type
structure Triangle :=
  (A B C : Point)

-- Define the properties of altitude, median, and angle bisector
def is_altitude (l : Line) (t : Triangle) : Prop := sorry
def is_median (l : Line) (t : Triangle) : Prop := sorry
def is_angle_bisector (l : Line) (t : Triangle) : Prop := sorry

-- Define the intersection of two lines
def intersection (l1 l2 : Line) : Point := sorry

-- Main theorem
theorem triangle_reconstruction_uniqueness 
  (X Y Z : Point) 
  (h_distinct : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) :
  ∃! t : Triangle,
    ∃ (alt med bis : Line),
      is_altitude alt t ∧
      is_median med t ∧
      is_angle_bisector bis t ∧
      X = intersection alt med ∧
      Y = intersection alt bis ∧
      Z = intersection med bis :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l1614_161492


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l1614_161491

theorem probability_at_least_one_multiple_of_four :
  let total_numbers : ℕ := 100
  let multiples_of_four : ℕ := 25
  let non_multiples_of_four : ℕ := total_numbers - multiples_of_four
  let prob_non_multiple : ℚ := non_multiples_of_four / total_numbers
  let prob_both_non_multiples : ℚ := prob_non_multiple * prob_non_multiple
  let prob_at_least_one_multiple : ℚ := 1 - prob_both_non_multiples
  prob_at_least_one_multiple = 7 / 16 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l1614_161491


namespace NUMINAMATH_CALUDE_dragon_boat_festival_probability_l1614_161457

theorem dragon_boat_festival_probability (pA pB pC : ℝ) 
  (hA : pA = 2/3) (hB : pB = 1/4) (hC : pC = 3/5) : 
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_dragon_boat_festival_probability_l1614_161457


namespace NUMINAMATH_CALUDE_two_digit_number_divisibility_l1614_161438

theorem two_digit_number_divisibility (a b : ℕ) :
  a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 →
  (∀ x y : ℕ, x ≥ 1 ∧ x ≤ 9 ∧ y ≤ 9 → x * y ≤ 35) →
  b * a = 35 →
  (∀ d : ℕ, d ∣ (10 * a + b) → d ≤ 75) ∧ 75 ∣ (10 * a + b) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_divisibility_l1614_161438


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1614_161405

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- There are 5 balls -/
def num_balls : ℕ := 5

/-- There are 4 boxes -/
def num_boxes : ℕ := 4

/-- The theorem stating that there are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : 
  distribute_balls num_balls num_boxes = 56 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1614_161405


namespace NUMINAMATH_CALUDE_multiplier_problem_l1614_161489

theorem multiplier_problem (x : ℝ) (h1 : x = 11) (h2 : 3 * x = (26 - x) + 18) :
  ∃ m : ℝ, m * x = (26 - x) + 18 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l1614_161489


namespace NUMINAMATH_CALUDE_wall_bricks_proof_l1614_161410

/-- Represents the number of bricks in the wall -/
def wall_bricks : ℕ := 127

/-- Bea's time to build the wall alone in hours -/
def bea_time : ℚ := 8

/-- Ben's time to build the wall alone in hours -/
def ben_time : ℚ := 12

/-- Bea's break time in minutes per hour -/
def bea_break : ℚ := 10

/-- Ben's break time in minutes per hour -/
def ben_break : ℚ := 15

/-- Decrease in output when working together in bricks per hour -/
def output_decrease : ℕ := 12

/-- Time taken to complete the wall when working together in hours -/
def combined_time : ℚ := 6

/-- Bea's effective working time per hour in minutes -/
def bea_effective_time : ℚ := 60 - bea_break

/-- Ben's effective working time per hour in minutes -/
def ben_effective_time : ℚ := 60 - ben_break

theorem wall_bricks_proof :
  let bea_rate : ℚ := wall_bricks / (bea_time * bea_effective_time / 60)
  let ben_rate : ℚ := wall_bricks / (ben_time * ben_effective_time / 60)
  let combined_rate : ℚ := bea_rate + ben_rate - output_decrease
  combined_rate * combined_time = wall_bricks :=
by sorry

end NUMINAMATH_CALUDE_wall_bricks_proof_l1614_161410


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1614_161486

theorem smallest_lcm_with_gcd_5 :
  ∃ (a b : ℕ), 
    1000 ≤ a ∧ a < 10000 ∧
    1000 ≤ b ∧ b < 10000 ∧
    Nat.gcd a b = 5 ∧
    Nat.lcm a b = 203010 ∧
    (∀ (c d : ℕ), 1000 ≤ c ∧ c < 10000 ∧ 1000 ≤ d ∧ d < 10000 ∧ Nat.gcd c d = 5 → 
      Nat.lcm c d ≥ 203010) :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1614_161486


namespace NUMINAMATH_CALUDE_number_division_problem_l1614_161480

theorem number_division_problem : ∃ N : ℕ,
  (N / (555 + 445) = 2 * (555 - 445)) ∧
  (N % (555 + 445) = 25) ∧
  (N = 220025) := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l1614_161480


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1614_161490

theorem fraction_to_decimal : (5 : ℚ) / 16 = (3125 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1614_161490


namespace NUMINAMATH_CALUDE_base_representation_five_digits_l1614_161434

theorem base_representation_five_digits (b' : ℕ+) : 
  (∃ (a b c d e : ℕ), a ≠ 0 ∧ 216 = a*(b'^4) + b*(b'^3) + c*(b'^2) + d*(b'^1) + e ∧ 
   a < b' ∧ b < b' ∧ c < b' ∧ d < b' ∧ e < b') ↔ b' = 3 :=
sorry

end NUMINAMATH_CALUDE_base_representation_five_digits_l1614_161434


namespace NUMINAMATH_CALUDE_total_onions_is_fifteen_l1614_161435

/-- The number of onions grown by Nancy -/
def nancy_onions : ℕ := 2

/-- The number of onions grown by Dan -/
def dan_onions : ℕ := 9

/-- The number of onions grown by Mike -/
def mike_onions : ℕ := 4

/-- The number of days they worked on the farm -/
def days_worked : ℕ := 6

/-- The total number of onions grown by Nancy, Dan, and Mike -/
def total_onions : ℕ := nancy_onions + dan_onions + mike_onions

theorem total_onions_is_fifteen : total_onions = 15 := by sorry

end NUMINAMATH_CALUDE_total_onions_is_fifteen_l1614_161435


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1614_161451

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  let angle := 60 * π / 180
  let mag_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let mag_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  mag_a = 1 ∧ mag_b = 2 ∧
  a.1 * b.1 + a.2 * b.2 = mag_a * mag_b * Real.cos angle →
  Real.sqrt (((a.1 + b.1) ^ 2) + ((a.2 + b.2) ^ 2)) = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1614_161451


namespace NUMINAMATH_CALUDE_max_product_l1614_161462

def digits : Finset Nat := {3, 5, 6, 8, 9}

def isValidPair (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def threeDigitNum (a b c : Nat) : Nat := 100 * a + 10 * b + c
def twoDigitNum (d e : Nat) : Nat := 10 * d + e

def product (a b c d e : Nat) : Nat :=
  threeDigitNum a b c * twoDigitNum d e

theorem max_product :
  ∀ a b c d e,
    isValidPair a b c d e →
    product a b c d e ≤ product 8 5 9 6 3 :=
by sorry

end NUMINAMATH_CALUDE_max_product_l1614_161462


namespace NUMINAMATH_CALUDE_solve_equation_l1614_161495

theorem solve_equation (x : ℝ) : (x / 5) + 3 = 4 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1614_161495


namespace NUMINAMATH_CALUDE_lcm_150_294_l1614_161497

theorem lcm_150_294 : Nat.lcm 150 294 = 7350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_150_294_l1614_161497


namespace NUMINAMATH_CALUDE_cos_sum_sevenths_pi_l1614_161466

theorem cos_sum_sevenths_pi : 
  Real.cos (π / 7) + Real.cos (2 * π / 7) + Real.cos (3 * π / 7) + 
  Real.cos (4 * π / 7) + Real.cos (5 * π / 7) + Real.cos (6 * π / 7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_sevenths_pi_l1614_161466


namespace NUMINAMATH_CALUDE_number_of_small_boxes_l1614_161483

/-- Given a large box containing small boxes of chocolates, this theorem proves
    the number of small boxes given the total number of chocolates and
    the number of chocolates per small box. -/
theorem number_of_small_boxes
  (total_chocolates : ℕ)
  (chocolates_per_box : ℕ)
  (h1 : total_chocolates = 400)
  (h2 : chocolates_per_box = 25)
  (h3 : total_chocolates % chocolates_per_box = 0) :
  total_chocolates / chocolates_per_box = 16 := by
  sorry

#check number_of_small_boxes

end NUMINAMATH_CALUDE_number_of_small_boxes_l1614_161483


namespace NUMINAMATH_CALUDE_binary_1100_eq_12_l1614_161455

/-- Converts a binary number represented as a list of bits (0 or 1) to its decimal equivalent. -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.reverse.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of 1100 -/
def binary_1100 : List Nat := [1, 1, 0, 0]

/-- Theorem stating that the binary number 1100 is equal to the decimal number 12 -/
theorem binary_1100_eq_12 : binary_to_decimal binary_1100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100_eq_12_l1614_161455


namespace NUMINAMATH_CALUDE_syrup_volume_l1614_161496

/-- The final volume of syrup after reduction and sugar addition -/
theorem syrup_volume (y : ℝ) : 
  let initial_volume : ℝ := 6 * 4  -- 6 quarts to cups
  let reduced_volume : ℝ := initial_volume * (1 / 12)
  let volume_with_sugar : ℝ := reduced_volume + 1
  let final_volume : ℝ := volume_with_sugar * y
  final_volume = 3 * y :=
by sorry

end NUMINAMATH_CALUDE_syrup_volume_l1614_161496


namespace NUMINAMATH_CALUDE_age_difference_l1614_161499

theorem age_difference (C D m : ℕ) : 
  C = D + m →                    -- Chris is m years older than Daniel
  C - 1 = 3 * (D - 1) →          -- Last year Chris was 3 times as old as Daniel
  C * D = 72 →                   -- This year, the product of their ages is 72
  m = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1614_161499


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1614_161428

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality_solution 
  (b c : ℝ) 
  (h1 : quadratic_function b c (-1) = 0)
  (h2 : quadratic_function b c 2 = 0) :
  {x : ℝ | quadratic_function b c x < 4} = Set.Ioo (-2) 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1614_161428


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1614_161413

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1614_161413


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1614_161459

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ x, x > a → x > 2) ∧ (∃ x, x > 2 ∧ x ≤ a) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1614_161459


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l1614_161440

/-- The expression to be simplified -/
def original_expression (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 5 * (x^3 - 2*x^2 + 4*x - 1)

/-- The fully simplified form of the expression -/
def simplified_expression (x : ℝ) : ℝ := -5*x^3 + 13*x^2 - 29*x + 14

/-- The coefficients of the simplified expression -/
def coefficients : List ℝ := [-5, 13, -29, 14]

/-- Theorem stating that the sum of squares of coefficients equals 1231 -/
theorem sum_of_squares_of_coefficients :
  (coefficients.map (λ c => c^2)).sum = 1231 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l1614_161440


namespace NUMINAMATH_CALUDE_perfect_squares_dividing_specific_l1614_161429

/-- The number of perfect squares dividing 2^3 * 3^5 * 5^7 * 7^9 -/
def perfectSquaresDividing (a b c d : ℕ) : ℕ :=
  (a/2 + 1) * (b/2 + 1) * (c/2 + 1) * (d/2 + 1)

/-- Theorem stating that the number of perfect squares dividing 2^3 * 3^5 * 5^7 * 7^9 is 120 -/
theorem perfect_squares_dividing_specific : perfectSquaresDividing 3 5 7 9 = 120 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_dividing_specific_l1614_161429


namespace NUMINAMATH_CALUDE_coin_value_equality_l1614_161403

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- Given that the value of 25 quarters and 15 dimes equals the value of m quarters and 40 dimes, 
    prove that m equals 15 -/
theorem coin_value_equality (m : ℕ) : 
  25 * quarter_value + 15 * dime_value = m * quarter_value + 40 * dime_value → m = 15 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_equality_l1614_161403


namespace NUMINAMATH_CALUDE_songs_per_album_l1614_161425

theorem songs_per_album (total_albums : ℕ) (total_songs : ℕ) 
  (h1 : total_albums = 3 + 5) 
  (h2 : total_songs = 24) 
  (h3 : ∀ (x : ℕ), x * total_albums = total_songs → x = 3) :
  ∃ (songs_per_album : ℕ), songs_per_album * total_albums = total_songs ∧ songs_per_album = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_songs_per_album_l1614_161425


namespace NUMINAMATH_CALUDE_jim_card_distribution_l1614_161431

theorem jim_card_distribution (initial_cards : ℕ) (brother_sets sister_sets : ℕ) 
  (total_given : ℕ) (cards_per_set : ℕ) : 
  initial_cards = 365 →
  brother_sets = 8 →
  sister_sets = 5 →
  total_given = 195 →
  cards_per_set = 13 →
  (brother_sets + sister_sets + (total_given - (brother_sets + sister_sets) * cards_per_set) / cards_per_set : ℕ) = 
    brother_sets + sister_sets + 2 := by
  sorry

#check jim_card_distribution

end NUMINAMATH_CALUDE_jim_card_distribution_l1614_161431


namespace NUMINAMATH_CALUDE_simplify_expression_l1614_161417

theorem simplify_expression (a b c : ℝ) (h : a * b ≠ c^2) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - c^2) = a / b + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1614_161417


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_6sqrt5_l1614_161412

theorem sqrt_sum_equals_6sqrt5 : 
  Real.sqrt ((2 - 3 * Real.sqrt 5) ^ 2) + Real.sqrt ((2 + 3 * Real.sqrt 5) ^ 2) = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_6sqrt5_l1614_161412


namespace NUMINAMATH_CALUDE_jane_pens_after_month_l1614_161422

def alex_pens (week : ℕ) : ℕ := 4 * 2^week

def jane_pens : ℕ := alex_pens 3 - 16

theorem jane_pens_after_month : jane_pens = 16 := by
  sorry

end NUMINAMATH_CALUDE_jane_pens_after_month_l1614_161422


namespace NUMINAMATH_CALUDE_factorial_divisor_differences_l1614_161478

def divisors (n : ℕ) : List ℕ := sorry

def consecutive_differences (l : List ℕ) : List ℕ := sorry

def is_non_decreasing (l : List ℕ) : Prop := sorry

theorem factorial_divisor_differences (n : ℕ) :
  n ≥ 3 ∧ is_non_decreasing (consecutive_differences (divisors (n.factorial))) ↔ n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisor_differences_l1614_161478


namespace NUMINAMATH_CALUDE_min_value_expression_l1614_161470

theorem min_value_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 2 + 50 / n ≥ 10 ∧ ∃ m : ℕ, m > 0 ∧ (m : ℝ) / 2 + 50 / m = 10 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1614_161470


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1614_161498

theorem right_triangle_hypotenuse : ∀ x₁ x₂ : ℝ,
  x₁^2 - 36*x₁ + 70 = 0 →
  x₂^2 - 36*x₂ + 70 = 0 →
  x₁ ≠ x₂ →
  Real.sqrt (x₁^2 + x₂^2) = 34 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1614_161498


namespace NUMINAMATH_CALUDE_mathildas_debt_l1614_161488

/-- Mathilda's debt problem -/
theorem mathildas_debt (initial_payment : ℝ) (remaining_percentage : ℝ) (original_debt : ℝ) : 
  initial_payment = 125 ∧ 
  remaining_percentage = 75 ∧ 
  initial_payment = (100 - remaining_percentage) / 100 * original_debt →
  original_debt = 500 := by
  sorry

end NUMINAMATH_CALUDE_mathildas_debt_l1614_161488


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1614_161468

theorem complex_equation_solution (i z : ℂ) (hi : i * i = -1) (hz : (2 * i) / z = 1 - i) : z = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1614_161468


namespace NUMINAMATH_CALUDE_polynomial_identity_l1614_161482

theorem polynomial_identity (p : ℝ → ℝ) 
  (h1 : ∀ x, p (x^2 + 1) = (p x)^2 + 1) 
  (h2 : p 0 = 0) : 
  ∀ x, p x = x := by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1614_161482


namespace NUMINAMATH_CALUDE_certain_number_exists_l1614_161485

theorem certain_number_exists : ∃ N : ℝ, (5/6 : ℝ) * N = (5/16 : ℝ) * N + 150 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l1614_161485


namespace NUMINAMATH_CALUDE_cafeteria_green_apples_l1614_161436

/-- Prove that the number of green apples ordered by the cafeteria is 23 -/
theorem cafeteria_green_apples :
  let red_apples : ℕ := 33
  let students_wanting_fruit : ℕ := 21
  let extra_apples : ℕ := 35
  let green_apples : ℕ := 23
  (red_apples + green_apples - students_wanting_fruit = extra_apples) →
  green_apples = 23 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_green_apples_l1614_161436


namespace NUMINAMATH_CALUDE_other_candidate_votes_l1614_161420

-- Define the total number of votes
def total_votes : ℕ := 7500

-- Define the percentage of invalid votes
def invalid_vote_percentage : ℚ := 20 / 100

-- Define the percentage of votes for the winning candidate
def winning_candidate_percentage : ℚ := 55 / 100

-- Theorem to prove
theorem other_candidate_votes :
  (total_votes * (1 - invalid_vote_percentage) * (1 - winning_candidate_percentage)).floor = 2700 :=
by sorry

end NUMINAMATH_CALUDE_other_candidate_votes_l1614_161420


namespace NUMINAMATH_CALUDE_opposite_reciprocal_abs_l1614_161473

theorem opposite_reciprocal_abs (x : ℚ) (h : x = -1.5) : 
  (-x = 1.5) ∧ (1 / x = -2/3) ∧ (abs x = 1.5) := by
  sorry

#check opposite_reciprocal_abs

end NUMINAMATH_CALUDE_opposite_reciprocal_abs_l1614_161473


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1614_161421

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ A ∩ B ↔ -1 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1614_161421


namespace NUMINAMATH_CALUDE_cycle_gains_and_overall_gain_l1614_161439

def cycle1_purchase : ℚ := 900
def cycle1_sale : ℚ := 1440
def cycle2_purchase : ℚ := 1200
def cycle2_sale : ℚ := 1680
def cycle3_purchase : ℚ := 1500
def cycle3_sale : ℚ := 1950

def gain_percentage (purchase : ℚ) (sale : ℚ) : ℚ :=
  ((sale - purchase) / purchase) * 100

def total_purchase : ℚ := cycle1_purchase + cycle2_purchase + cycle3_purchase
def total_sale : ℚ := cycle1_sale + cycle2_sale + cycle3_sale

theorem cycle_gains_and_overall_gain :
  (gain_percentage cycle1_purchase cycle1_sale = 60) ∧
  (gain_percentage cycle2_purchase cycle2_sale = 40) ∧
  (gain_percentage cycle3_purchase cycle3_sale = 30) ∧
  (gain_percentage total_purchase total_sale = 40 + 5/6) :=
sorry

end NUMINAMATH_CALUDE_cycle_gains_and_overall_gain_l1614_161439


namespace NUMINAMATH_CALUDE_sin_390_degrees_l1614_161414

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  have h1 : ∀ x, Real.sin (x + 2 * π) = Real.sin x := by sorry
  have h2 : Real.sin (π / 6) = 1 / 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l1614_161414


namespace NUMINAMATH_CALUDE_solve_rational_equation_l1614_161418

theorem solve_rational_equation (x : ℚ) :
  (x^2 - 10*x + 9) / (x - 1) + (2*x^2 + 17*x - 15) / (2*x - 3) = -5 →
  x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_rational_equation_l1614_161418


namespace NUMINAMATH_CALUDE_locus_equation_l1614_161404

-- Define the focus point F
def F : ℝ × ℝ := (2, 0)

-- Define the directrix line l: x + 3 = 0
def l (x : ℝ) : Prop := x + 3 = 0

-- Define the distance condition for point M
def distance_condition (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  let dist_to_F := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)
  let dist_to_l := |x + 3|
  dist_to_F + 1 = dist_to_l

-- State the theorem
theorem locus_equation :
  ∀ M : ℝ × ℝ, distance_condition M ↔ M.2^2 = 8 * M.1 :=
sorry

end NUMINAMATH_CALUDE_locus_equation_l1614_161404


namespace NUMINAMATH_CALUDE_barkley_bones_l1614_161467

/-- The number of new dog bones Barkley gets at the beginning of each month -/
def monthly_new_bones : ℕ := sorry

/-- The number of months -/
def months : ℕ := 5

/-- The number of bones available after 5 months -/
def available_bones : ℕ := 8

/-- The number of bones buried after 5 months -/
def buried_bones : ℕ := 42

theorem barkley_bones : monthly_new_bones = 10 := by
  sorry

end NUMINAMATH_CALUDE_barkley_bones_l1614_161467


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1614_161445

theorem sqrt_difference_equality : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1614_161445


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1614_161479

theorem right_triangle_inequality (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) 
  (h_a_nonneg : a ≥ 0) (h_b_nonneg : b ≥ 0) (h_c_pos : c > 0) : 
  c ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1614_161479


namespace NUMINAMATH_CALUDE_intersection_line_slope_l1614_161424

theorem intersection_line_slope (u : ℝ) :
  let line1 := {(x, y) : ℝ × ℝ | 2 * x + 3 * y = 8 * u + 4}
  let line2 := {(x, y) : ℝ × ℝ | 3 * x + 2 * y = 9 * u + 1}
  let intersection := {(x, y) : ℝ × ℝ | (x, y) ∈ line1 ∩ line2}
  ∃ (m b : ℝ), m = 6 / 47 ∧ ∀ (x y : ℝ), (x, y) ∈ intersection → y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l1614_161424


namespace NUMINAMATH_CALUDE_sqrt_294_simplification_l1614_161441

theorem sqrt_294_simplification : Real.sqrt 294 = 7 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_294_simplification_l1614_161441


namespace NUMINAMATH_CALUDE_E_opposite_Z_l1614_161487

/-- Represents a face of the cube -/
inductive Face : Type
| A : Face
| B : Face
| C : Face
| D : Face
| E : Face
| Z : Face

/-- Represents the net of the cube before folding -/
structure CubeNet :=
(faces : List Face)
(can_fold_to_cube : Bool)

/-- Represents the folded cube -/
structure Cube :=
(net : CubeNet)
(opposite_faces : Face → Face)

/-- The theorem stating that E is opposite to Z in the folded cube -/
theorem E_opposite_Z (net : CubeNet) (cube : Cube) :
  net.can_fold_to_cube = true →
  cube.net = net →
  cube.opposite_faces Face.Z = Face.E :=
sorry

end NUMINAMATH_CALUDE_E_opposite_Z_l1614_161487


namespace NUMINAMATH_CALUDE_equation_solution_l1614_161477

theorem equation_solution : ∃ x : ℝ, (1 / 6 + 6 / x = 15 / x + 1 / 15) ∧ x = 90 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1614_161477


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l1614_161426

theorem fraction_subtraction_equality : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l1614_161426


namespace NUMINAMATH_CALUDE_distinct_numbers_probability_l1614_161453

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice being rolled -/
def numDice : ℕ := 5

/-- The probability of rolling five standard, six-sided dice and getting five distinct numbers -/
def probabilityDistinctNumbers : ℚ := 5 / 54

theorem distinct_numbers_probability :
  (numSides.factorial / (numSides - numDice).factorial) / numSides ^ numDice = probabilityDistinctNumbers :=
sorry

end NUMINAMATH_CALUDE_distinct_numbers_probability_l1614_161453


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l1614_161469

/-- Given two employees A and B with a total weekly pay of 550 and B's pay of 220,
    prove that A's pay is 150% of B's pay. -/
theorem employee_pay_percentage (total_pay : ℝ) (b_pay : ℝ) (a_pay : ℝ)
  (h1 : total_pay = 550)
  (h2 : b_pay = 220)
  (h3 : a_pay + b_pay = total_pay) :
  a_pay / b_pay * 100 = 150 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l1614_161469


namespace NUMINAMATH_CALUDE_salary_problem_l1614_161411

theorem salary_problem (salary_a salary_b : ℝ) : 
  salary_a + salary_b = 2000 →
  salary_a * 0.05 = salary_b * 0.15 →
  salary_a = 1500 := by
sorry

end NUMINAMATH_CALUDE_salary_problem_l1614_161411


namespace NUMINAMATH_CALUDE_circle_radius_when_area_circumference_ratio_is_ten_l1614_161432

/-- Given a circle with area M cm² and circumference N cm, if M/N = 10, then the radius is 20 cm -/
theorem circle_radius_when_area_circumference_ratio_is_ten
  (M N : ℝ) -- M is the area, N is the circumference
  (h1 : M > 0) -- area is positive
  (h2 : N > 0) -- circumference is positive
  (h3 : M = π * (N / (2 * π))^2) -- area formula
  (h4 : M / N = 10) -- given ratio
  : N / (2 * π) = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_when_area_circumference_ratio_is_ten_l1614_161432


namespace NUMINAMATH_CALUDE_quadratic_properties_l1614_161407

/-- Quadratic function f(x) = 2x^2 + 4x - 6 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 6

/-- Vertex form of f(x) -/
def vertex_form (x : ℝ) : ℝ := 2 * (x + 1)^2 - 8

/-- Axis of symmetry -/
def axis_of_symmetry : ℝ := -1

/-- Vertex coordinates -/
def vertex : ℝ × ℝ := (-1, -8)

theorem quadratic_properties :
  (∀ x, f x = vertex_form x) ∧
  (axis_of_symmetry = -1) ∧
  (vertex = (-1, -8)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1614_161407


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1614_161493

/-- Calculate the total cost of tomatoes and apples --/
theorem total_cost_calculation (tomato_price : ℝ) (tomato_weight : ℝ) (apple_price : ℝ) (apple_weight : ℝ) 
  (h1 : tomato_price = 5)
  (h2 : tomato_weight = 2)
  (h3 : apple_price = 6)
  (h4 : apple_weight = 5) :
  tomato_weight * tomato_price + apple_weight * apple_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1614_161493


namespace NUMINAMATH_CALUDE_bottle_cap_calculation_l1614_161402

theorem bottle_cap_calculation (caps_per_box : ℝ) (num_boxes : ℝ) 
  (h1 : caps_per_box = 35.0) 
  (h2 : num_boxes = 7.0) : 
  caps_per_box * num_boxes = 245.0 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_calculation_l1614_161402


namespace NUMINAMATH_CALUDE_a_range_l1614_161475

def f (a : ℝ) (x : ℝ) : ℝ := x * |x - a|

theorem a_range (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ici 3 → x₂ ∈ Set.Ici 3 → x₁ ≠ x₂ → 
    (f a x₁ - f a x₂) / (x₁ - x₂) > 0) → 
  a ∈ Set.Iic 3 := by
sorry

end NUMINAMATH_CALUDE_a_range_l1614_161475
