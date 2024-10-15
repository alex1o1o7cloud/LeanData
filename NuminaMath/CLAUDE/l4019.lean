import Mathlib

namespace NUMINAMATH_CALUDE_interior_angle_regular_pentagon_l4019_401902

/-- The measure of one interior angle of a regular pentagon is 108 degrees. -/
theorem interior_angle_regular_pentagon : ℝ :=
  let n : ℕ := 5  -- number of sides in a pentagon
  let S : ℝ := 180 * (n - 2)  -- sum of interior angles formula
  let angle_measure : ℝ := S / n  -- measure of one interior angle
  108

/-- Proof of the theorem -/
lemma proof_interior_angle_regular_pentagon : interior_angle_regular_pentagon = 108 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_regular_pentagon_l4019_401902


namespace NUMINAMATH_CALUDE_pen_selling_problem_l4019_401984

/-- Proves that given the conditions of the pen selling problem, the initial number of pens purchased is 30 -/
theorem pen_selling_problem (n : ℕ) (P : ℝ) (h1 : P > 0) :
  (∃ (S : ℝ), S > 0 ∧ 20 * S = P ∧ n * (2/3 * S) = P) →
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_pen_selling_problem_l4019_401984


namespace NUMINAMATH_CALUDE_cauchy_equation_on_X_l4019_401911

-- Define the set X
def X : Set ℝ := {x : ℝ | ∃ (a b : ℤ), x = a + b * Real.sqrt 2}

-- Define the Cauchy equation property
def is_cauchy (f : X → ℝ) : Prop :=
  ∀ (x y : X), f (⟨x + y, sorry⟩) = f x + f y

-- State the theorem
theorem cauchy_equation_on_X (f : X → ℝ) (hf : is_cauchy f) :
  ∀ (a b : ℤ), f ⟨a + b * Real.sqrt 2, sorry⟩ = a * f ⟨1, sorry⟩ + b * f ⟨Real.sqrt 2, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_cauchy_equation_on_X_l4019_401911


namespace NUMINAMATH_CALUDE_ed_weight_l4019_401926

/-- Given the weights of Al, Ben, Carl, and Ed, prove that Ed weighs 146 pounds -/
theorem ed_weight (al ben carl ed : ℕ) : 
  al = ben + 25 →
  ben = carl - 16 →
  ed = al - 38 →
  carl = 175 →
  ed = 146 := by
  sorry

end NUMINAMATH_CALUDE_ed_weight_l4019_401926


namespace NUMINAMATH_CALUDE_misread_addition_l4019_401967

/-- Given a two-digit number XY where Y = 9, if 57 + X6 = 123, then XY = 69 -/
theorem misread_addition (X Y : Nat) : Y = 9 → 57 + (10 * X + 6) = 123 → 10 * X + Y = 69 := by
  sorry

end NUMINAMATH_CALUDE_misread_addition_l4019_401967


namespace NUMINAMATH_CALUDE_correct_remaining_contents_l4019_401953

/-- Represents the contents of a cup with coffee and milk -/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Calculates the remaining contents in the cup after mixing and removing some mixture -/
def remainingContents (initialCoffee : ℚ) (addedMilk : ℚ) (removedMixture : ℚ) : CupContents :=
  let totalVolume := initialCoffee + addedMilk
  let coffeeRatio := initialCoffee / totalVolume
  let milkRatio := addedMilk / totalVolume
  let remainingVolume := totalVolume - removedMixture
  { coffee := coffeeRatio * remainingVolume,
    milk := milkRatio * remainingVolume }

/-- Theorem stating the correct remaining contents after mixing and removing -/
theorem correct_remaining_contents :
  let result := remainingContents 1 (1/4) (1/4)
  result.coffee = 4/5 ∧ result.milk = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_correct_remaining_contents_l4019_401953


namespace NUMINAMATH_CALUDE_mork_tax_rate_l4019_401912

theorem mork_tax_rate (mork_income : ℝ) (mork_rate : ℝ) : 
  mork_income > 0 →
  mork_rate * mork_income + 0.2 * (4 * mork_income) = 0.25 * (5 * mork_income) →
  mork_rate = 0.45 := by
sorry

end NUMINAMATH_CALUDE_mork_tax_rate_l4019_401912


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l4019_401989

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l4019_401989


namespace NUMINAMATH_CALUDE_equation_natural_solution_l4019_401905

/-- Given an equation C - x = 2b - 2ax where C is a constant,
    a is a real parameter, and b = 7, this theorem states the
    conditions for the equation to have a natural number solution. -/
theorem equation_natural_solution (C : ℝ) (a : ℝ) :
  (∃ x : ℕ, C - x = 2 * 7 - 2 * a * x) ↔ 
  (a > (1 : ℝ) / 2 ∧ ∃ n : ℕ+, 2 * a - 1 = n) :=
sorry

end NUMINAMATH_CALUDE_equation_natural_solution_l4019_401905


namespace NUMINAMATH_CALUDE_unique_prime_squared_plus_eleven_with_six_divisors_l4019_401904

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n has exactly 6 positive divisors -/
def has_six_divisors (n : ℕ) : Prop := num_divisors n = 6

theorem unique_prime_squared_plus_eleven_with_six_divisors :
  ∃! p : ℕ, Nat.Prime p ∧ has_six_divisors (p^2 + 11) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_squared_plus_eleven_with_six_divisors_l4019_401904


namespace NUMINAMATH_CALUDE_johns_age_problem_l4019_401916

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem johns_age_problem :
  ∃! x : ℕ, x > 0 ∧ is_perfect_square (x - 5) ∧ is_perfect_cube (x + 3) ∧ x = 69 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_problem_l4019_401916


namespace NUMINAMATH_CALUDE_angle_POQ_is_72_degrees_l4019_401924

-- Define the regular pentagon
structure RegularPentagon where
  side_length : ℝ
  internal_angle : ℝ
  internal_angle_eq : internal_angle = 108

-- Define the inscribed circle
structure InscribedCircle (p : RegularPentagon) where
  center : Point
  radius : ℝ
  tangent_point1 : Point
  tangent_point2 : Point
  corner : Point
  is_tangent : Bool
  intersects_other_sides : Bool

-- Define the angle POQ
def angle_POQ (p : RegularPentagon) (c : InscribedCircle p) : ℝ :=
  sorry

-- Define the bisector property
def is_bisector (p : RegularPentagon) (c : InscribedCircle p) : Prop :=
  sorry

-- Theorem statement
theorem angle_POQ_is_72_degrees 
  (p : RegularPentagon) 
  (c : InscribedCircle p) 
  (h1 : c.is_tangent = true) 
  (h2 : c.intersects_other_sides = true) 
  (h3 : is_bisector p c) : 
  angle_POQ p c = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_POQ_is_72_degrees_l4019_401924


namespace NUMINAMATH_CALUDE_hildasAge_l4019_401965

def guesses : List Nat := [25, 29, 31, 33, 37, 39, 42, 45, 48, 50]

def isComposite (n : Nat) : Prop := ∃ a b, a > 1 ∧ b > 1 ∧ a * b = n

def countHighGuesses (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (λ g => g > age)).length

def offByTwo (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (λ g => g = age - 2 ∨ g = age + 2)).length

theorem hildasAge :
  ∃ age : Nat,
    age ∈ guesses ∧
    isComposite age ∧
    countHighGuesses age guesses ≥ guesses.length / 4 ∧
    offByTwo age guesses = 2 ∧
    age = 45 := by sorry

end NUMINAMATH_CALUDE_hildasAge_l4019_401965


namespace NUMINAMATH_CALUDE_scooter_rental_proof_l4019_401960

/-- Represents the rental cost structure for an electric scooter service -/
structure RentalCost where
  fixed : ℝ
  per_minute : ℝ

/-- Calculates the total cost for a given duration -/
def total_cost (rc : RentalCost) (duration : ℝ) : ℝ :=
  rc.fixed + rc.per_minute * duration

theorem scooter_rental_proof (rc : RentalCost) 
  (h1 : total_cost rc 3 = 78)
  (h2 : total_cost rc 8 = 108) :
  total_cost rc 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_scooter_rental_proof_l4019_401960


namespace NUMINAMATH_CALUDE_marathon_heart_beats_l4019_401981

/-- Calculates the number of heart beats during a marathon --/
def marathonHeartBeats (totalDistance : ℕ) (heartRate : ℕ) (firstHalfDistance : ℕ) (firstHalfPace : ℕ) (secondHalfPace : ℕ) : ℕ :=
  let firstHalfTime := firstHalfDistance * firstHalfPace
  let secondHalfTime := (totalDistance - firstHalfDistance) * secondHalfPace
  let totalTime := firstHalfTime + secondHalfTime
  totalTime * heartRate

/-- Theorem: The athlete's heart beats 23100 times during the marathon --/
theorem marathon_heart_beats :
  marathonHeartBeats 30 140 15 6 5 = 23100 := by
  sorry

#eval marathonHeartBeats 30 140 15 6 5

end NUMINAMATH_CALUDE_marathon_heart_beats_l4019_401981


namespace NUMINAMATH_CALUDE_expression_value_l4019_401930

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 5) :
  3 * x - 4 * y + 2 * z = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4019_401930


namespace NUMINAMATH_CALUDE_non_congruent_triangle_count_l4019_401992

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The set of 10 points in the problem -/
def problem_points : Finset Point2D := sorry

/-- Predicate to check if three points form a triangle -/
def is_triangle (p q r : Point2D) : Prop := sorry

/-- Predicate to check if two triangles are congruent -/
def are_congruent (t1 t2 : Point2D × Point2D × Point2D) : Prop := sorry

/-- The set of all possible triangles formed by the problem points -/
def all_triangles : Finset (Point2D × Point2D × Point2D) := sorry

/-- The set of non-congruent triangles -/
def non_congruent_triangles : Finset (Point2D × Point2D × Point2D) := sorry

theorem non_congruent_triangle_count :
  Finset.card non_congruent_triangles = 12 := by sorry

end NUMINAMATH_CALUDE_non_congruent_triangle_count_l4019_401992


namespace NUMINAMATH_CALUDE_jills_shopping_breakdown_l4019_401985

/-- Represents the shopping breakdown and tax calculation for Jill's purchase --/
theorem jills_shopping_breakdown (T : ℝ) (x : ℝ) 
  (h1 : T > 0) -- Total amount spent is positive
  (h2 : x ≥ 0 ∧ x ≤ 1) -- Percentage spent on other items is between 0 and 100%
  (h3 : 0.5 + 0.2 + x = 1) -- Total percentage spent is 100%
  (h4 : 0.02 * T + 0.1 * x * T = 0.05 * T) -- Tax equation
  : x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_jills_shopping_breakdown_l4019_401985


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4019_401954

/-- Given a right triangle with legs a and b, if the volume of the cone formed by
    rotating the triangle about leg a is 1000π cm³ and the volume of the cone formed by
    rotating the triangle about leg b is 2430π cm³, then the length of the hypotenuse c
    is approximately 28.12 cm. -/
theorem right_triangle_hypotenuse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / 3 * π * b^2 * a = 1000 * π) →
  (1 / 3 * π * a^2 * b = 2430 * π) →
  abs (Real.sqrt (a^2 + b^2) - 28.12) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4019_401954


namespace NUMINAMATH_CALUDE_cable_cost_per_roommate_l4019_401968

/-- Represents the cost structure of the cable program --/
structure CableProgram where
  tier1_channels : Nat
  tier1_cost : ℚ
  tier2_channels : Nat
  tier2_cost : ℚ
  tier3_channels : Nat
  tier3_cost_ratio : ℚ
  tier4_channels : Nat
  tier4_cost_ratio : ℚ

/-- Calculates the total cost of the cable program --/
def total_cost (program : CableProgram) : ℚ :=
  program.tier1_cost +
  program.tier2_cost +
  (program.tier2_cost * program.tier3_cost_ratio * program.tier3_channels / program.tier2_channels) +
  (program.tier2_cost * program.tier3_cost_ratio * (1 + program.tier4_cost_ratio) * program.tier4_channels / program.tier2_channels)

/-- Theorem stating that each roommate pays $81.25 --/
theorem cable_cost_per_roommate (program : CableProgram)
  (h1 : program.tier1_channels = 100)
  (h2 : program.tier1_cost = 100)
  (h3 : program.tier2_channels = 100)
  (h4 : program.tier2_cost = 75)
  (h5 : program.tier3_channels = 150)
  (h6 : program.tier3_cost_ratio = 1/2)
  (h7 : program.tier4_channels = 200)
  (h8 : program.tier4_cost_ratio = 1/4)
  (h9 : program.tier2_channels = 100) :
  (total_cost program) / 4 = 325 / 4 := by
  sorry

#eval (325 : ℚ) / 4

end NUMINAMATH_CALUDE_cable_cost_per_roommate_l4019_401968


namespace NUMINAMATH_CALUDE_remainder_4n_squared_mod_13_l4019_401919

theorem remainder_4n_squared_mod_13 (n : ℤ) (h : n % 13 = 7) : (4 * n^2) % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4n_squared_mod_13_l4019_401919


namespace NUMINAMATH_CALUDE_evaluate_expression_l4019_401962

theorem evaluate_expression : (-(18 / 3 * 12 - 80 + 4 * 12))^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4019_401962


namespace NUMINAMATH_CALUDE_total_climb_length_l4019_401969

def keaton_ladder_length : ℕ := 30
def keaton_climbs : ℕ := 20
def reece_ladder_difference : ℕ := 4
def reece_climbs : ℕ := 15
def inches_per_foot : ℕ := 12

theorem total_climb_length : 
  (keaton_ladder_length * keaton_climbs + 
   (keaton_ladder_length - reece_ladder_difference) * reece_climbs) * 
   inches_per_foot = 11880 := by
  sorry

end NUMINAMATH_CALUDE_total_climb_length_l4019_401969


namespace NUMINAMATH_CALUDE_subsets_of_size_two_l4019_401928

/-- Given a finite set S, returns the number of subsets of S with exactly k elements -/
def numSubsetsOfSize (n k : ℕ) : ℕ := Nat.choose n k

theorem subsets_of_size_two (S : Type) [Fintype S] :
  (numSubsetsOfSize (Fintype.card S) 7 = 36) →
  (numSubsetsOfSize (Fintype.card S) 2 = 36) := by
  sorry

end NUMINAMATH_CALUDE_subsets_of_size_two_l4019_401928


namespace NUMINAMATH_CALUDE_cubic_roots_l4019_401958

theorem cubic_roots : 
  ∀ x : ℝ, x^3 + 3*x^2 - 6*x - 8 = 0 ↔ x = -1 ∨ x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_l4019_401958


namespace NUMINAMATH_CALUDE_student_weight_l4019_401996

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight + sister_weight = 132)
  (h2 : student_weight - 6 = 2 * sister_weight) : 
  student_weight = 90 := by
  sorry

end NUMINAMATH_CALUDE_student_weight_l4019_401996


namespace NUMINAMATH_CALUDE_most_likely_final_number_is_54_l4019_401927

/-- The initial number on the blackboard -/
def initial_number : ℕ := 15

/-- The lower bound of the random number added in each move -/
def lower_bound : ℕ := 1

/-- The upper bound of the random number added in each move -/
def upper_bound : ℕ := 5

/-- The threshold number for ending the game -/
def threshold : ℕ := 51

/-- The expected value of the random number added in each move -/
def expected_value : ℚ := (lower_bound + upper_bound) / 2

/-- The most likely final number on the blackboard -/
def most_likely_final_number : ℕ := 54

/-- Theorem stating that the most likely final number is 54 -/
theorem most_likely_final_number_is_54 :
  ∃ (n : ℕ), initial_number + n * expected_value > threshold ∧
             initial_number + (n - 1) * expected_value ≤ threshold ∧
             most_likely_final_number = initial_number + n * ⌊expected_value⌋ := by
  sorry

end NUMINAMATH_CALUDE_most_likely_final_number_is_54_l4019_401927


namespace NUMINAMATH_CALUDE_scale_division_l4019_401999

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 90

/-- Represents the length of each part in inches -/
def part_length : ℕ := 18

/-- Theorem stating that the scale divided into equal parts results in 5 parts -/
theorem scale_division :
  scale_length / part_length = 5 := by sorry

end NUMINAMATH_CALUDE_scale_division_l4019_401999


namespace NUMINAMATH_CALUDE_max_value_fraction_l4019_401949

theorem max_value_fraction (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) :
  (k * x + y)^2 / (x^2 + k * y^2) ≤ k + 1 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (k * x + y)^2 / (x^2 + k * y^2) = k + 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l4019_401949


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4019_401970

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4019_401970


namespace NUMINAMATH_CALUDE_prime_sum_divides_power_sum_l4019_401956

theorem prime_sum_divides_power_sum (p q : ℕ) : 
  Prime p → Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
sorry

end NUMINAMATH_CALUDE_prime_sum_divides_power_sum_l4019_401956


namespace NUMINAMATH_CALUDE_cylinder_height_relation_l4019_401986

-- Define the cylinders
def Cylinder (r h : ℝ) := r > 0 ∧ h > 0

-- Theorem statement
theorem cylinder_height_relation 
  (r₁ h₁ r₂ h₂ : ℝ) 
  (cyl₁ : Cylinder r₁ h₁) 
  (cyl₂ : Cylinder r₂ h₂) 
  (volume_eq : r₁^2 * h₁ = r₂^2 * h₂) 
  (radius_relation : r₂ = 1.2 * r₁) : 
  h₁ = 1.44 * h₂ := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_relation_l4019_401986


namespace NUMINAMATH_CALUDE_tshirt_packages_l4019_401906

theorem tshirt_packages (package_size : ℕ) (desired_shirts : ℕ) (min_packages : ℕ) : 
  package_size = 6 →
  desired_shirts = 71 →
  min_packages * package_size ≥ desired_shirts →
  ∀ n : ℕ, n * package_size ≥ desired_shirts → n ≥ min_packages →
  min_packages = 12 :=
by sorry

end NUMINAMATH_CALUDE_tshirt_packages_l4019_401906


namespace NUMINAMATH_CALUDE_vasyas_numbers_l4019_401923

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l4019_401923


namespace NUMINAMATH_CALUDE_equality_check_l4019_401977

theorem equality_check : 
  (2^3 ≠ 3^2) ∧ 
  (-(-2) = |-2|) ∧ 
  ((-2)^2 ≠ -2^2) ∧ 
  ((2/3)^2 ≠ 2^2/3) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l4019_401977


namespace NUMINAMATH_CALUDE_F_lower_bound_F_max_value_l4019_401917

/-- The condition that x and y satisfy -/
def satisfies_condition (x y : ℝ) : Prop := x^2 + x*y + y^2 = 1

/-- The function F(x, y) -/
def F (x y : ℝ) : ℝ := x^3*y + x*y^3

/-- Theorem stating that F(x, y) ≥ -2 for any x and y satisfying the condition -/
theorem F_lower_bound {x y : ℝ} (h : satisfies_condition x y) : F x y ≥ -2 := by
  sorry

/-- Theorem stating that the maximum value of F(x, y) is 1/4 -/
theorem F_max_value : ∃ (x y : ℝ), satisfies_condition x y ∧ F x y = 1/4 ∧ ∀ (a b : ℝ), satisfies_condition a b → F a b ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_F_lower_bound_F_max_value_l4019_401917


namespace NUMINAMATH_CALUDE_white_smallest_probability_l4019_401997

def total_balls : ℕ := 16
def red_balls : ℕ := 9
def black_balls : ℕ := 5
def white_balls : ℕ := 2

theorem white_smallest_probability :
  (white_balls : ℚ) / total_balls < (red_balls : ℚ) / total_balls ∧
  (white_balls : ℚ) / total_balls < (black_balls : ℚ) / total_balls :=
by sorry

end NUMINAMATH_CALUDE_white_smallest_probability_l4019_401997


namespace NUMINAMATH_CALUDE_remaining_amount_l4019_401979

def initial_amount : ℕ := 20
def peach_quantity : ℕ := 3
def peach_price : ℕ := 2

theorem remaining_amount : 
  initial_amount - (peach_quantity * peach_price) = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_l4019_401979


namespace NUMINAMATH_CALUDE_product_closure_l4019_401943

def M : Set ℤ := {x | ∃ a b : ℤ, x = a^2 + b^2}

theorem product_closure (x₁ x₂ : ℤ) (h₁ : x₁ ∈ M) (h₂ : x₂ ∈ M) : x₁ * x₂ ∈ M := by
  sorry

end NUMINAMATH_CALUDE_product_closure_l4019_401943


namespace NUMINAMATH_CALUDE_equation_equivalence_product_l4019_401976

theorem equation_equivalence_product (a b x y : ℤ) (m n p q : ℕ) :
  (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) ↔ 
  ((a^m*x - a^n)*(a^p*y - a^q) = a^5*b^5) →
  m*n*p*q = 2 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_product_l4019_401976


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_irrational_l4019_401982

theorem inscribed_circle_radius_irrational (b c : ℕ) : 
  b ≥ 1 → c ≥ 1 → 1 + b > c → 1 + c > b → b + c > 1 → 
  ¬ ∃ (r : ℚ), r = (Real.sqrt ((b : ℝ)^2 - 1/4)) / (1 + 2*(b : ℝ)) := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_irrational_l4019_401982


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l4019_401950

theorem arithmetic_mean_problem (y : ℝ) : 
  (7 + y + 22 + 8 + 18) / 5 = 15 → y = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l4019_401950


namespace NUMINAMATH_CALUDE_line_position_l4019_401961

structure Line3D where
  -- Assume we have a suitable representation for 3D lines
  -- This is just a placeholder

def skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

def intersects (l1 l2 : Line3D) : Prop :=
  -- Definition of intersecting lines
  sorry

theorem line_position (L1 L2 m1 m2 : Line3D) 
  (h1 : skew L1 L2)
  (h2 : intersects m1 L1)
  (h3 : intersects m1 L2)
  (h4 : intersects m2 L1)
  (h5 : intersects m2 L2) :
  intersects m1 m2 ∨ skew m1 m2 :=
by
  sorry

end NUMINAMATH_CALUDE_line_position_l4019_401961


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4019_401994

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (25 - Real.sqrt n) = 3 → n = 256 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4019_401994


namespace NUMINAMATH_CALUDE_christopher_karen_money_difference_l4019_401959

theorem christopher_karen_money_difference : 
  let karen_quarters : ℕ := 32
  let christopher_quarters : ℕ := 64
  let quarter_value : ℚ := 1/4
  (christopher_quarters - karen_quarters) * quarter_value = 8 := by sorry

end NUMINAMATH_CALUDE_christopher_karen_money_difference_l4019_401959


namespace NUMINAMATH_CALUDE_classmate_pairs_l4019_401947

theorem classmate_pairs (n : ℕ) (h : n = 6) : (n.choose 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_classmate_pairs_l4019_401947


namespace NUMINAMATH_CALUDE_min_y_coordinate_polar_graph_l4019_401944

/-- The minimum y-coordinate of a point on the graph of r = cos(2θ) is -√6/3 -/
theorem min_y_coordinate_polar_graph :
  let r : ℝ → ℝ := λ θ ↦ Real.cos (2 * θ)
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  ∃ y_min : ℝ, y_min = -Real.sqrt 6 / 3 ∧ ∀ θ : ℝ, y θ ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_y_coordinate_polar_graph_l4019_401944


namespace NUMINAMATH_CALUDE_three_solutions_inequality_l4019_401939

theorem three_solutions_inequality (a : ℝ) : 
  (∃! (s : Finset ℕ), s.card = 3 ∧ 
    (∀ x : ℕ, x ∈ s ↔ (x > 0 ∧ 3 * (x - 1) < 2 * (x + a) - 5))) ↔ 
  (5/2 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_three_solutions_inequality_l4019_401939


namespace NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l4019_401987

-- Define the conversion factor from yards to feet
def yards_to_feet : ℝ := 3

-- Define the volume in cubic yards
def volume_cubic_yards : ℝ := 6

-- Theorem to prove
theorem cubic_yards_to_cubic_feet :
  volume_cubic_yards * (yards_to_feet ^ 3) = 162 := by
  sorry

end NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l4019_401987


namespace NUMINAMATH_CALUDE_game_preparation_time_l4019_401909

/-- The time taken to prepare all games is 150 minutes, given that each game takes 10 minutes to prepare and Andrew prepared 15 games. -/
theorem game_preparation_time : 
  let time_per_game : ℕ := 10
  let total_games : ℕ := 15
  let total_time := time_per_game * total_games
  total_time = 150 := by sorry

end NUMINAMATH_CALUDE_game_preparation_time_l4019_401909


namespace NUMINAMATH_CALUDE_total_amount_paid_l4019_401998

theorem total_amount_paid (total_work : ℚ) (ac_portion : ℚ) (b_payment : ℚ) : 
  total_work = 1 ∧ 
  ac_portion = 19/23 ∧ 
  b_payment = 12 →
  (1 - ac_portion) * (total_work * b_payment) / (1 - ac_portion) = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l4019_401998


namespace NUMINAMATH_CALUDE_nut_boxes_problem_l4019_401963

theorem nut_boxes_problem (first second third : ℕ) : 
  (second = (11 * first) / 10) →
  (second = (13 * third) / 10) →
  (first = third + 80) →
  (first = 520 ∧ second = 572 ∧ third = 440) :=
by sorry

end NUMINAMATH_CALUDE_nut_boxes_problem_l4019_401963


namespace NUMINAMATH_CALUDE_simple_interest_principal_l4019_401921

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 260)
  (h2 : rate = 7.142857142857143)
  (h3 : time = 4) :
  ∃ (principal : ℝ), principal = 910 ∧ interest = principal * rate * time / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l4019_401921


namespace NUMINAMATH_CALUDE_sequence_term_equation_l4019_401966

def sequence_term (n : ℕ+) : ℕ := 9 * (n - 1) + n

theorem sequence_term_equation (n : ℕ+) : sequence_term n = 10 * n - 9 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_equation_l4019_401966


namespace NUMINAMATH_CALUDE_smallest_divisor_with_remainder_one_l4019_401975

theorem smallest_divisor_with_remainder_one (total_boxes : Nat) (h1 : total_boxes = 301) 
  (h2 : total_boxes % 7 = 0) : 
  (∃ x : Nat, x > 0 ∧ total_boxes % x = 1) ∧ 
  (∀ y : Nat, y > 0 ∧ y < 3 → total_boxes % y ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_with_remainder_one_l4019_401975


namespace NUMINAMATH_CALUDE_mikes_shirt_cost_l4019_401907

/-- The cost of Mike's shirt given the profit sharing between Mike and Johnson -/
theorem mikes_shirt_cost (total_profit : ℚ) (mikes_share johnson_share : ℚ) : 
  mikes_share / johnson_share = 2 / 5 →
  johnson_share = 2500 →
  mikes_share - 800 = 200 :=
by sorry

end NUMINAMATH_CALUDE_mikes_shirt_cost_l4019_401907


namespace NUMINAMATH_CALUDE_square_circle_ratio_l4019_401948

theorem square_circle_ratio (r c d : ℝ) (h : r > 0) (hc : c > 0) (hd : d > c) :
  let s := 2 * r
  s^2 = (c / d) * (s^2 - π * r^2) →
  s / r = Real.sqrt (c * π) / Real.sqrt (d - c) := by
sorry

end NUMINAMATH_CALUDE_square_circle_ratio_l4019_401948


namespace NUMINAMATH_CALUDE_sum_of_roots_l4019_401952

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x - 6

-- State the theorem
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = -5) : a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4019_401952


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l4019_401971

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) :
  r > 0 →
  r * r = 2 →
  π * r * r = π :=
by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l4019_401971


namespace NUMINAMATH_CALUDE_num_boolean_structures_l4019_401991

/-- The transformation group of 3 Boolean variables -/
def TransformationGroup : Type := Fin 6

/-- The state configurations for 3 Boolean variables -/
def StateConfigurations : Type := Fin 8

/-- The number of colors (Boolean states) -/
def NumColors : Nat := 2

/-- A permutation on the state configurations -/
def Permutation : Type := StateConfigurations → StateConfigurations

/-- The group of permutations induced by the transformation group -/
def PermutationGroup : Type := TransformationGroup → Permutation

/-- Count the number of cycles in a permutation -/
def cycleCount (p : Permutation) : Nat :=
  sorry

/-- Pólya's Enumeration Theorem for this specific case -/
def polyaEnumeration (G : PermutationGroup) : Nat :=
  sorry

/-- The main theorem: number of different structures for a Boolean function device with 3 variables -/
theorem num_boolean_structures (G : PermutationGroup) : 
  polyaEnumeration G = 80 :=
sorry

end NUMINAMATH_CALUDE_num_boolean_structures_l4019_401991


namespace NUMINAMATH_CALUDE_first_year_interest_rate_is_four_percent_l4019_401935

/-- Calculates the final amount after two years of compound interest -/
def finalAmount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial * (1 + rate1) * (1 + rate2)

/-- Theorem stating that given the conditions, the first year interest rate must be 4% -/
theorem first_year_interest_rate_is_four_percent 
  (initial : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (final : ℝ) 
  (h1 : initial = 7000)
  (h2 : rate2 = 0.05)
  (h3 : final = 7644)
  (h4 : finalAmount initial rate1 rate2 = final) : 
  rate1 = 0.04 := by
  sorry

#check first_year_interest_rate_is_four_percent

end NUMINAMATH_CALUDE_first_year_interest_rate_is_four_percent_l4019_401935


namespace NUMINAMATH_CALUDE_mady_balls_after_2023_steps_l4019_401941

/-- Converts a natural number to its septenary (base 7) representation -/
def to_septenary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Sums the digits in a list of natural numbers -/
def sum_digits (l : List ℕ) : ℕ :=
  l.sum

/-- Represents Mady's ball placement process -/
def mady_process (steps : ℕ) : ℕ :=
  sum_digits (to_septenary steps)

theorem mady_balls_after_2023_steps :
  mady_process 2023 = 13 := by sorry

end NUMINAMATH_CALUDE_mady_balls_after_2023_steps_l4019_401941


namespace NUMINAMATH_CALUDE_intersection_condition_l4019_401983

/-- The set of possible values for a real number a, given the conditions. -/
def PossibleValues : Set ℝ := {-1, 0, 1}

/-- The set A defined by the equation ax + 1 = 0. -/
def A (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

/-- The set B containing -1 and 1. -/
def B : Set ℝ := {-1, 1}

/-- Theorem stating that if A ∩ B = A, then a must be in the set of possible values. -/
theorem intersection_condition (a : ℝ) : A a ∩ B = A a → a ∈ PossibleValues := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l4019_401983


namespace NUMINAMATH_CALUDE_jessica_payment_l4019_401951

/-- Calculates the payment for a given hour based on the repeating pattern --/
def hourly_rate (hour : ℕ) : ℕ :=
  match hour % 6 with
  | 0 => 2
  | 1 => 4
  | 2 => 6
  | 3 => 8
  | 4 => 10
  | 5 => 12
  | _ => 0  -- This case should never occur due to the modulo operation

/-- Calculates the total payment for a given number of hours --/
def total_payment (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_rate |>.sum

theorem jessica_payment : total_payment 45 = 306 := by
  sorry


end NUMINAMATH_CALUDE_jessica_payment_l4019_401951


namespace NUMINAMATH_CALUDE_jesse_carpet_need_l4019_401937

/-- The amount of additional carpet Jesse needs to cover two rooms -/
def additional_carpet_needed (jesse_carpet area_room_a area_room_b : ℝ) : ℝ :=
  area_room_a + area_room_b - jesse_carpet

/-- Proof that Jesse needs 94 more square feet of carpet -/
theorem jesse_carpet_need : 
  let jesse_carpet : ℝ := 18
  let room_a_length : ℝ := 4
  let room_a_width : ℝ := 20
  let area_room_a : ℝ := room_a_length * room_a_width
  let area_room_b : ℝ := area_room_a / 2.5
  additional_carpet_needed jesse_carpet area_room_a area_room_b = 94
  := by sorry

end NUMINAMATH_CALUDE_jesse_carpet_need_l4019_401937


namespace NUMINAMATH_CALUDE_brick_height_calculation_l4019_401931

/-- The height of a brick given wall dimensions, brick dimensions, and number of bricks --/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ)
  (brick_length brick_width : ℝ) (num_bricks : ℝ) :
  wall_length = 9 →
  wall_width = 5 →
  wall_height = 18.5 →
  brick_length = 0.21 →
  brick_width = 0.1 →
  num_bricks = 4955.357142857142 →
  ∃ (brick_height : ℝ),
    brick_height = 0.008 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry

end NUMINAMATH_CALUDE_brick_height_calculation_l4019_401931


namespace NUMINAMATH_CALUDE_log_inequality_l4019_401973

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l4019_401973


namespace NUMINAMATH_CALUDE_no_integer_y_prime_abs_quadratic_l4019_401922

theorem no_integer_y_prime_abs_quadratic : ¬ ∃ y : ℤ, Nat.Prime (Int.natAbs (8*y^2 - 55*y + 21)) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_y_prime_abs_quadratic_l4019_401922


namespace NUMINAMATH_CALUDE_candies_remaining_l4019_401938

theorem candies_remaining (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  red = 40 →
  yellow = 3 * red - 20 →
  blue = yellow / 2 →
  red + blue = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_candies_remaining_l4019_401938


namespace NUMINAMATH_CALUDE_row_properties_l4019_401946

/-- Definition of a number being in a row -/
def in_row (m n : ℕ) : Prop :=
  n ∣ m ∧ m ≤ n^2 ∧ ∀ k < n, ¬in_row m k

/-- The main theorem encompassing all parts of the problem -/
theorem row_properties :
  (∀ m < 50, m % 10 = 0 → ∃ k < 10, in_row m k) ∧
  (∀ n ≥ 3, in_row (n^2 - n) n ∧ in_row (n^2 - 2*n) n) ∧
  (∀ n > 30, in_row (n^2 - 10*n) n) ∧
  ¬in_row (30^2 - 10*30) 30 := by
  sorry

#check row_properties

end NUMINAMATH_CALUDE_row_properties_l4019_401946


namespace NUMINAMATH_CALUDE_square_card_arrangement_l4019_401945

theorem square_card_arrangement (perimeter_cards : ℕ) (h : perimeter_cards = 240) : 
  ∃ (side_length : ℕ), 
    4 * side_length - 4 = perimeter_cards ∧ 
    side_length * side_length = 3721 := by
  sorry

end NUMINAMATH_CALUDE_square_card_arrangement_l4019_401945


namespace NUMINAMATH_CALUDE_factorization_proofs_l4019_401988

theorem factorization_proofs (x y a b : ℝ) : 
  (2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2) ∧ 
  (a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proofs_l4019_401988


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4019_401990

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4019_401990


namespace NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l4019_401934

/-- Represents the number of family members -/
def family_members : ℕ := 5

/-- Represents the number of front seats in the van -/
def front_seats : ℕ := 2

/-- Represents the number of back seats in the van -/
def back_seats : ℕ := 3

/-- Represents the number of adults who can drive -/
def potential_drivers : ℕ := 2

/-- Calculates the number of possible seating arrangements -/
def seating_arrangements : ℕ :=
  potential_drivers * (family_members - 1) * (back_seats.factorial)

theorem lopez_family_seating_arrangements :
  seating_arrangements = 48 :=
sorry

end NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l4019_401934


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4019_401929

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (1 + 2*I) / I
  Complex.im z = -1 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4019_401929


namespace NUMINAMATH_CALUDE_traveler_distance_l4019_401903

/-- Calculates the distance traveled given initial conditions and new travel parameters. -/
def distance_traveled (initial_distance : ℚ) (initial_days : ℕ) (initial_hours_per_day : ℕ)
                      (new_days : ℕ) (new_hours_per_day : ℕ) : ℚ :=
  let initial_total_hours : ℚ := initial_days * initial_hours_per_day
  let speed : ℚ := initial_distance / initial_total_hours
  let new_total_hours : ℚ := new_days * new_hours_per_day
  speed * new_total_hours

/-- The theorem states that given the initial conditions and new travel parameters,
    the traveler will cover 93 23/29 kilometers. -/
theorem traveler_distance : 
  distance_traveled 112 29 7 17 10 = 93 + 23 / 29 := by
  sorry

end NUMINAMATH_CALUDE_traveler_distance_l4019_401903


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l4019_401918

theorem quadratic_inequality_condition (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + m > 0) → m > 0 ∧ ∃ m₀ > 0, ¬(∀ x : ℝ, x^2 - 2*x + m₀ > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l4019_401918


namespace NUMINAMATH_CALUDE_determinant_2x2_l4019_401936

open Matrix

theorem determinant_2x2 (a b c d : ℝ) : 
  det ![![a, c], ![b, d]] = a * d - b * c := by
  sorry

end NUMINAMATH_CALUDE_determinant_2x2_l4019_401936


namespace NUMINAMATH_CALUDE_pharmacy_work_hours_l4019_401933

/-- Proves that given the conditions of the pharmacy problem, 
    the number of hours worked by Ann and Becky is 8 --/
theorem pharmacy_work_hours : 
  ∀ (h : ℕ), 
  (7 * h + 7 * h + 7 * 6 = 154) → 
  h = 8 := by
sorry

end NUMINAMATH_CALUDE_pharmacy_work_hours_l4019_401933


namespace NUMINAMATH_CALUDE_square_diff_squared_l4019_401974

theorem square_diff_squared : (7^2 - 5^2)^2 = 576 := by sorry

end NUMINAMATH_CALUDE_square_diff_squared_l4019_401974


namespace NUMINAMATH_CALUDE_min_box_height_l4019_401900

theorem min_box_height (x : ℝ) (h : x > 0) : 
  (10 * x^2 ≥ 150) → (∀ y : ℝ, y > 0 → 10 * y^2 ≥ 150 → y ≥ x) → 2 * x = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_min_box_height_l4019_401900


namespace NUMINAMATH_CALUDE_total_cups_on_table_l4019_401914

theorem total_cups_on_table (juice_cups milk_cups : ℕ) 
  (h1 : juice_cups = 3) 
  (h2 : milk_cups = 4) : 
  juice_cups + milk_cups = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_cups_on_table_l4019_401914


namespace NUMINAMATH_CALUDE_smaller_ladder_steps_l4019_401972

theorem smaller_ladder_steps 
  (full_ladder_steps : ℕ) 
  (full_ladder_climbs : ℕ) 
  (smaller_ladder_climbs : ℕ) 
  (total_steps : ℕ) 
  (h1 : full_ladder_steps = 11)
  (h2 : full_ladder_climbs = 10)
  (h3 : smaller_ladder_climbs = 7)
  (h4 : total_steps = 152)
  (h5 : full_ladder_steps * full_ladder_climbs + smaller_ladder_climbs * x = total_steps) :
  x = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_ladder_steps_l4019_401972


namespace NUMINAMATH_CALUDE_clothing_store_profit_model_l4019_401993

/-- Represents the clothing store's sales and profit model -/
structure ClothingStore where
  originalCost : ℝ
  originalPrice : ℝ
  originalSales : ℝ
  salesIncrease : ℝ
  priceReduction : ℝ

/-- Calculate daily sales after price reduction -/
def dailySales (store : ClothingStore) : ℝ :=
  store.originalSales + store.salesIncrease * store.priceReduction

/-- Calculate profit per piece after price reduction -/
def profitPerPiece (store : ClothingStore) : ℝ :=
  store.originalPrice - store.originalCost - store.priceReduction

/-- Calculate total daily profit -/
def dailyProfit (store : ClothingStore) : ℝ :=
  dailySales store * profitPerPiece store

/-- The main theorem about the clothing store's profit model -/
theorem clothing_store_profit_model (store : ClothingStore) 
  (h1 : store.originalCost = 80)
  (h2 : store.originalPrice = 120)
  (h3 : store.originalSales = 20)
  (h4 : store.salesIncrease = 2) :
  (∀ x, dailySales { store with priceReduction := x } = 20 + 2 * x) ∧
  (∀ x, profitPerPiece { store with priceReduction := x } = 40 - x) ∧
  (dailyProfit { store with priceReduction := 20 } = 1200) ∧
  (∀ x, dailyProfit { store with priceReduction := x } ≠ 2000) := by
  sorry


end NUMINAMATH_CALUDE_clothing_store_profit_model_l4019_401993


namespace NUMINAMATH_CALUDE_emily_spending_l4019_401980

theorem emily_spending (x : ℝ) : 
  x + 2*x + 3*x = 120 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_spending_l4019_401980


namespace NUMINAMATH_CALUDE_systematic_sampling_methods_systematic_sampling_characterization_l4019_401995

/-- Represents a sampling method -/
inductive SamplingMethod
| Method1
| Method2
| Method3
| Method4

/-- Predicate to determine if a sampling method is systematic -/
def is_systematic (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.Method1 => true
  | SamplingMethod.Method2 => true
  | SamplingMethod.Method3 => false
  | SamplingMethod.Method4 => true

/-- Theorem stating which sampling methods are systematic -/
theorem systematic_sampling_methods :
  (is_systematic SamplingMethod.Method1) ∧
  (is_systematic SamplingMethod.Method2) ∧
  (¬is_systematic SamplingMethod.Method3) ∧
  (is_systematic SamplingMethod.Method4) :=
by sorry

/-- Characterization of systematic sampling -/
theorem systematic_sampling_characterization (method : SamplingMethod) :
  is_systematic method ↔ 
    (∃ (rule : Prop), 
      (rule ↔ method = SamplingMethod.Method1 ∨ 
               method = SamplingMethod.Method2 ∨ 
               method = SamplingMethod.Method4) ∧
      (rule → ∃ (interval : Nat), interval > 0)) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_methods_systematic_sampling_characterization_l4019_401995


namespace NUMINAMATH_CALUDE_malcolm_instagram_followers_l4019_401957

/-- Represents the number of followers on various social media platforms --/
structure SocialMediaFollowers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Calculates the total number of followers across all platforms --/
def totalFollowers (smf : SocialMediaFollowers) : ℕ :=
  smf.instagram + smf.facebook + smf.twitter + smf.tiktok + smf.youtube

/-- Theorem stating that Malcolm has 240 followers on Instagram --/
theorem malcolm_instagram_followers :
  ∃ (smf : SocialMediaFollowers),
    smf.facebook = 500 ∧
    smf.twitter = (smf.instagram + smf.facebook) / 2 ∧
    smf.tiktok = 3 * smf.twitter ∧
    smf.youtube = smf.tiktok + 510 ∧
    totalFollowers smf = 3840 ∧
    smf.instagram = 240 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_instagram_followers_l4019_401957


namespace NUMINAMATH_CALUDE_plate_distance_l4019_401913

/-- Given a square table with a circular plate, prove that the distance from the bottom edge
    of the table to the plate is 53 cm, given the distances from other edges. -/
theorem plate_distance (left_distance right_distance top_distance : ℝ) :
  left_distance = 10 →
  right_distance = 63 →
  top_distance = 20 →
  ∃ (plate_diameter bottom_distance : ℝ),
    left_distance + plate_diameter + right_distance = top_distance + plate_diameter + bottom_distance ∧
    bottom_distance = 53 :=
by sorry

end NUMINAMATH_CALUDE_plate_distance_l4019_401913


namespace NUMINAMATH_CALUDE_remainder_theorem_l4019_401915

theorem remainder_theorem (N : ℤ) : N % 13 = 3 → N % 39 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4019_401915


namespace NUMINAMATH_CALUDE_one_third_of_ten_y_minus_three_l4019_401908

theorem one_third_of_ten_y_minus_three (y : ℝ) : (1/3) * (10*y - 3) = (10*y)/3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_ten_y_minus_three_l4019_401908


namespace NUMINAMATH_CALUDE_grocery_solution_l4019_401920

/-- Represents the grocery shopping problem --/
def grocery_problem (mustard_oil_price : ℝ) (mustard_oil_amount : ℝ) 
  (pasta_price : ℝ) (sauce_price : ℝ) (sauce_amount : ℝ) 
  (initial_money : ℝ) (remaining_money : ℝ) : Prop :=
  ∃ (pasta_amount : ℝ),
    mustard_oil_price * mustard_oil_amount + 
    pasta_price * pasta_amount + 
    sauce_price * sauce_amount = 
    initial_money - remaining_money ∧
    pasta_amount = 3

/-- Theorem stating the solution to the grocery problem --/
theorem grocery_solution : 
  grocery_problem 13 2 4 5 1 50 7 := by
  sorry

end NUMINAMATH_CALUDE_grocery_solution_l4019_401920


namespace NUMINAMATH_CALUDE_train_time_difference_l4019_401942

def distance : ℝ := 425.80645161290323
def speed_slow : ℝ := 44
def speed_fast : ℝ := 75

theorem train_time_difference :
  (distance / speed_slow) - (distance / speed_fast) = 4 := by
  sorry

end NUMINAMATH_CALUDE_train_time_difference_l4019_401942


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l4019_401925

def a (n : ℕ) : ℚ := n + 1 / (2^n)

theorem sequence_formula_correct : 
  (a 1 = 3/2) ∧ (a 2 = 9/4) ∧ (a 3 = 25/8) ∧ (a 4 = 65/16) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l4019_401925


namespace NUMINAMATH_CALUDE_sum_of_sequences_l4019_401901

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + (n * (n - 1) * d) / 2

def sequence1_sum : ℕ := arithmetic_sum 3 10 6
def sequence2_sum : ℕ := arithmetic_sum 5 10 6

theorem sum_of_sequences : sequence1_sum + sequence2_sum = 348 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l4019_401901


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l4019_401932

-- Define the function f(x) = x³ - ax
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - a

theorem tangent_line_y_intercept (a : ℝ) :
  (f' a 1 = 1) →  -- Tangent line at x=1 is parallel to x - y - 1 = 0
  (∃ b c : ℝ, ∀ x : ℝ, b * x + c = f a 1 + f' a 1 * (x - 1)) →  -- Equation of tangent line
  (∃ y : ℝ, y = f a 1 + f' a 1 * (0 - 1) ∧ y = -2)  -- y-intercept is -2
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l4019_401932


namespace NUMINAMATH_CALUDE_xy_minimum_l4019_401955

theorem xy_minimum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1/2) :
  x * y ≥ 16 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 1/2 ∧ x₀ * y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_xy_minimum_l4019_401955


namespace NUMINAMATH_CALUDE_house_distance_theorem_l4019_401940

/-- Represents the position of a house on a street -/
structure House where
  position : ℝ

/-- Represents a street with four houses -/
structure Street where
  andrey : House
  borya : House
  vova : House
  gleb : House

/-- The distance between two houses -/
def distance (h1 h2 : House) : ℝ := 
  |h1.position - h2.position|

theorem house_distance_theorem (s : Street) : 
  (distance s.andrey s.borya = 600 ∧ 
   distance s.vova s.gleb = 600 ∧ 
   distance s.andrey s.gleb = 3 * distance s.borya s.vova) → 
  (distance s.andrey s.gleb = 900 ∨ distance s.andrey s.gleb = 1800) :=
sorry

end NUMINAMATH_CALUDE_house_distance_theorem_l4019_401940


namespace NUMINAMATH_CALUDE_weight_of_new_person_l4019_401978

theorem weight_of_new_person
  (n : ℕ)
  (initial_weight : ℝ)
  (replaced_weight : ℝ)
  (average_increase : ℝ)
  (h1 : n = 10)
  (h2 : replaced_weight = 70)
  (h3 : average_increase = 4) :
  initial_weight / n + average_increase = (initial_weight - replaced_weight + replaced_weight + n * average_increase) / n :=
by sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l4019_401978


namespace NUMINAMATH_CALUDE_stratified_sample_male_athletes_l4019_401910

/-- Represents the number of male athletes drawn in a stratified sample -/
def male_athletes_drawn (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * male_athletes) / total_athletes

/-- Theorem stating that in a stratified sample of 21 athletes from a population of 84 athletes 
    (48 male and 36 female), the number of male athletes drawn is 12 -/
theorem stratified_sample_male_athletes :
  male_athletes_drawn 84 48 21 = 12 := by
  sorry

#eval male_athletes_drawn 84 48 21

end NUMINAMATH_CALUDE_stratified_sample_male_athletes_l4019_401910


namespace NUMINAMATH_CALUDE_max_value_of_expression_l4019_401964

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 9 → (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ (a - b)^2 + (b - c)^2 + (c - a)^2) →
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 27 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l4019_401964
