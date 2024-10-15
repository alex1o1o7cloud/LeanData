import Mathlib

namespace NUMINAMATH_CALUDE_derivative_exp_sin_l1766_176650

theorem derivative_exp_sin (x : ℝ) : 
  deriv (fun x => Real.exp x * Real.sin x) x = Real.exp x * (Real.sin x + Real.cos x) := by
sorry

end NUMINAMATH_CALUDE_derivative_exp_sin_l1766_176650


namespace NUMINAMATH_CALUDE_monomial_sum_l1766_176668

/-- If the sum of two monomials is still a monomial, then it equals -5xy^2 --/
theorem monomial_sum (a b : ℕ) : 
  (∃ (c : ℚ) (d e : ℕ), (-4 * 10^a * X^a * Y^2 + 35 * X * Y^(b-2) = c * X^d * Y^e)) →
  (-4 * 10^a * X^a * Y^2 + 35 * X * Y^(b-2) = -5 * X * Y^2) :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_l1766_176668


namespace NUMINAMATH_CALUDE_canoe_production_sum_l1766_176662

theorem canoe_production_sum : 
  let a : ℕ := 5  -- first term
  let r : ℕ := 3  -- common ratio
  let n : ℕ := 8  -- number of terms
  a * (r^n - 1) / (r - 1) = 16400 :=
by sorry

end NUMINAMATH_CALUDE_canoe_production_sum_l1766_176662


namespace NUMINAMATH_CALUDE_reflection_of_point_l1766_176626

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Theorem: The reflection of the point (5,2) across the x-axis is (5,-2) -/
theorem reflection_of_point : reflect_x (5, 2) = (5, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_point_l1766_176626


namespace NUMINAMATH_CALUDE_total_wrapping_cost_l1766_176671

/-- Represents a wrapping paper design with its cost and wrapping capacities -/
structure WrappingPaper where
  cost : ℝ
  shirtBoxCapacity : ℕ
  xlBoxCapacity : ℕ
  xxlBoxCapacity : ℕ

/-- Calculates the number of rolls needed for a given number of boxes -/
def rollsNeeded (boxes : ℕ) (capacity : ℕ) : ℕ :=
  (boxes + capacity - 1) / capacity

/-- Calculates the cost for wrapping a specific type of box -/
def costForBoxType (paper : WrappingPaper) (boxes : ℕ) (capacity : ℕ) : ℝ :=
  paper.cost * (rollsNeeded boxes capacity : ℝ)

/-- Theorem stating the total cost of wrapping all boxes -/
theorem total_wrapping_cost (design1 design2 design3 : WrappingPaper)
    (shirtBoxes xlBoxes xxlBoxes : ℕ) :
    design1.cost = 4 →
    design1.shirtBoxCapacity = 5 →
    design2.cost = 8 →
    design2.xlBoxCapacity = 4 →
    design3.cost = 12 →
    design3.xxlBoxCapacity = 4 →
    shirtBoxes = 20 →
    xlBoxes = 12 →
    xxlBoxes = 6 →
    costForBoxType design1 shirtBoxes design1.shirtBoxCapacity +
    costForBoxType design2 xlBoxes design2.xlBoxCapacity +
    costForBoxType design3 xxlBoxes design3.xxlBoxCapacity = 76 := by
  sorry

end NUMINAMATH_CALUDE_total_wrapping_cost_l1766_176671


namespace NUMINAMATH_CALUDE_exactly_six_solutions_l1766_176606

/-- The number of ordered pairs of positive integers satisfying 3/m + 6/n = 1 -/
def solution_count : ℕ := 6

/-- Predicate for ordered pairs (m,n) satisfying the equation -/
def satisfies_equation (m n : ℕ+) : Prop :=
  (3 : ℚ) / m.val + (6 : ℚ) / n.val = 1

/-- The theorem stating that there are exactly 6 solutions -/
theorem exactly_six_solutions :
  ∃! (s : Finset (ℕ+ × ℕ+)), 
    s.card = solution_count ∧ 
    (∀ p ∈ s, satisfies_equation p.1 p.2) ∧
    (∀ m n : ℕ+, satisfies_equation m n → (m, n) ∈ s) :=
  sorry

end NUMINAMATH_CALUDE_exactly_six_solutions_l1766_176606


namespace NUMINAMATH_CALUDE_class_size_is_24_l1766_176651

-- Define the number of candidates
def num_candidates : Nat := 4

-- Define the number of absent students
def absent_students : Nat := 5

-- Define the function to calculate votes needed to win
def votes_to_win (x : Nat) : Nat :=
  if x % 2 = 0 then x / 2 + 1 else (x + 1) / 2

-- Define the function to calculate votes received by each candidate
def votes_received (x : Nat) (missed_by : Nat) : Nat :=
  votes_to_win x - missed_by

-- Define the theorem
theorem class_size_is_24 :
  ∃ (x : Nat),
    -- x is the number of students who voted
    x + absent_students = 24 ∧
    -- Sum of votes received by all candidates equals x
    votes_received x 3 + votes_received x 9 + votes_received x 5 + votes_received x 4 = x :=
by sorry

end NUMINAMATH_CALUDE_class_size_is_24_l1766_176651


namespace NUMINAMATH_CALUDE_yoongis_pets_l1766_176607

theorem yoongis_pets (dogs : ℕ) (cats : ℕ) : dogs = 5 → cats = 2 → dogs + cats = 7 := by
  sorry

end NUMINAMATH_CALUDE_yoongis_pets_l1766_176607


namespace NUMINAMATH_CALUDE_implicit_function_derivatives_l1766_176630

/-- Given an implicit function defined by x^y - y^x = 0, this theorem proves
    the expressions for its first and second derivatives. -/
theorem implicit_function_derivatives
  (x y : ℝ) (h : x^y = y^x) (hx : x > 0) (hy : y > 0) :
  let y' := (y^2 * (Real.log x - 1)) / (x^2 * (Real.log y - 1))
  let y'' := (x * (3 - 2 * Real.log x) * (Real.log y - 1)^2 +
              (Real.log x - 1)^2 * (2 * Real.log y - 3) * y) *
             y^2 / (x^4 * (Real.log y - 1)^3)
  ∃ f : ℝ → ℝ, (∀ t, t^(f t) = (f t)^t) ∧
               (deriv f x = y') ∧
               (deriv (deriv f) x = y'') := by
  sorry

end NUMINAMATH_CALUDE_implicit_function_derivatives_l1766_176630


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l1766_176676

theorem reciprocal_sum_pairs : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 4) ∧
    s.card = 5 :=
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l1766_176676


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_135_l1766_176678

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sequence is positive -/
def positive_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

/-- The sequence is increasing -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The condition a_1 < a_3 < a_5 -/
def condition_135 (a : ℕ → ℝ) : Prop :=
  a 1 < a 3 ∧ a 3 < a 5

theorem geometric_sequence_increasing_iff_135 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : positive_sequence a) :
  increasing_sequence a ↔ condition_135 a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_135_l1766_176678


namespace NUMINAMATH_CALUDE_board_cut_ratio_l1766_176629

/-- Proves that the ratio of the shorter piece to the longer piece is 1:1 for a 20-foot board cut into two pieces -/
theorem board_cut_ratio (total_length : ℝ) (shorter_length : ℝ) (longer_length : ℝ) :
  total_length = 20 →
  shorter_length = 8 →
  shorter_length = longer_length + 4 →
  shorter_length / longer_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_ratio_l1766_176629


namespace NUMINAMATH_CALUDE_rectangle_tiling_l1766_176619

/-- A rectangle can be tiled with 4x4 squares -/
def is_tileable (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m = 4 * a ∧ n = 4 * b

/-- If a rectangle with dimensions m × n can be tiled with 4 × 4 squares, 
    then m and n are divisible by 4 -/
theorem rectangle_tiling (m n : ℕ) :
  is_tileable m n → (4 ∣ m) ∧ (4 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_tiling_l1766_176619


namespace NUMINAMATH_CALUDE_dress_price_difference_l1766_176641

theorem dress_price_difference (original_price : ℝ) : 
  (original_price * 0.85 = 71.4) →
  (original_price - (71.4 * 1.25)) = 5.25 := by
sorry

end NUMINAMATH_CALUDE_dress_price_difference_l1766_176641


namespace NUMINAMATH_CALUDE_function_determination_l1766_176664

/-- Given a function f(x) = x³ - ax + b where x ∈ ℝ, 
    and the tangent line to f(x) at (1, f(1)) is 2x - y + 3 = 0,
    prove that f(x) = x³ - x + 5 -/
theorem function_determination (a b : ℝ) :
  (∀ x : ℝ, ∃ f : ℝ → ℝ, f x = x^3 - a*x + b) →
  (∃ f : ℝ → ℝ, ∀ x : ℝ, f x = x^3 - a*x + b ∧ 
    (2 * 1 - f 1 + 3 = 0) ∧
    (∀ x : ℝ, (2 * x - f x + 3 = 0) → x = 1)) →
  (∀ x : ℝ, ∃ f : ℝ → ℝ, f x = x^3 - x + 5) :=
by sorry

end NUMINAMATH_CALUDE_function_determination_l1766_176664


namespace NUMINAMATH_CALUDE_total_vegetarian_eaters_l1766_176628

/-- Represents the dietary preferences in a family -/
structure DietaryPreferences where
  vegetarian : ℕ
  nonVegetarian : ℕ
  bothVegNonVeg : ℕ
  vegan : ℕ
  veganAndVegetarian : ℕ
  pescatarian : ℕ
  pescatarianAndBoth : ℕ

/-- Theorem stating the total number of people eating vegetarian meals -/
theorem total_vegetarian_eaters (prefs : DietaryPreferences)
  (h1 : prefs.vegetarian = 13)
  (h2 : prefs.nonVegetarian = 7)
  (h3 : prefs.bothVegNonVeg = 8)
  (h4 : prefs.vegan = 5)
  (h5 : prefs.veganAndVegetarian = 3)
  (h6 : prefs.pescatarian = 4)
  (h7 : prefs.pescatarianAndBoth = 2) :
  prefs.vegetarian + prefs.bothVegNonVeg + (prefs.vegan - prefs.veganAndVegetarian) = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetarian_eaters_l1766_176628


namespace NUMINAMATH_CALUDE_smallest_d_for_inverse_l1766_176646

/-- The function g(x) = (x - 3)^2 - 7 -/
def g (x : ℝ) : ℝ := (x - 3)^2 - 7

/-- The property of being strictly increasing on an interval -/
def StrictlyIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y → f x < f y

/-- The smallest value of d for which g has an inverse function on [d, ∞) -/
theorem smallest_d_for_inverse : 
  (∃ d : ℝ, StrictlyIncreasing g d ∧ 
    (∀ c : ℝ, c < d → ¬StrictlyIncreasing g c)) ∧ 
  (∀ d : ℝ, StrictlyIncreasing g d → d ≥ 3) ∧
  StrictlyIncreasing g 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_inverse_l1766_176646


namespace NUMINAMATH_CALUDE_total_paid_is_117_l1766_176669

/-- Calculates the total amount paid after applying a senior citizen discount on Tuesday --/
def total_paid_after_discount (jimmy_shorts : ℕ) (jimmy_short_price : ℚ) 
                               (irene_shirts : ℕ) (irene_shirt_price : ℚ) 
                               (discount_rate : ℚ) : ℚ :=
  let total_before_discount := jimmy_shorts * jimmy_short_price + irene_shirts * irene_shirt_price
  let discount_amount := total_before_discount * discount_rate
  total_before_discount - discount_amount

/-- Proves that the total amount paid after the senior citizen discount is $117 --/
theorem total_paid_is_117 : 
  total_paid_after_discount 3 15 5 17 (1/10) = 117 := by
  sorry

end NUMINAMATH_CALUDE_total_paid_is_117_l1766_176669


namespace NUMINAMATH_CALUDE_total_legs_puppies_and_chicks_l1766_176688

/-- The number of legs for puppies and chicks -/
def total_legs (num_puppies num_chicks : ℕ) (puppy_legs chick_legs : ℕ) : ℕ :=
  num_puppies * puppy_legs + num_chicks * chick_legs

/-- Theorem: Given 3 puppies and 7 chicks, where puppies have 4 legs each and chicks have 2 legs each, the total number of legs is 26. -/
theorem total_legs_puppies_and_chicks :
  total_legs 3 7 4 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_puppies_and_chicks_l1766_176688


namespace NUMINAMATH_CALUDE_complex_fourth_quadrant_m_range_l1766_176609

theorem complex_fourth_quadrant_m_range (m : ℝ) : 
  let z : ℂ := Complex.mk (m + 3) (m - 1)
  (0 < z.re ∧ z.im < 0) → -3 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_quadrant_m_range_l1766_176609


namespace NUMINAMATH_CALUDE_four_prime_pairs_sum_50_l1766_176632

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given number -/
def countPrimePairs (sum : ℕ) : ℕ := sorry

/-- Theorem stating that there are exactly 4 unordered pairs of prime numbers that sum to 50 -/
theorem four_prime_pairs_sum_50 : countPrimePairs 50 = 4 := by sorry

end NUMINAMATH_CALUDE_four_prime_pairs_sum_50_l1766_176632


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1766_176663

open Real

theorem triangle_ABC_properties (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 3 →
  C = π / 3 →
  2 * sin (2 * A) + sin (A - B) = sin C →
  (A = π / 2 ∨ A = π / 6) ∧
  2 * Real.sqrt 3 ≤ a + b + c ∧ a + b + c ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l1766_176663


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_example_l1766_176612

/-- A triangle with sides a, b, and c is an isosceles right triangle -/
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧  -- Two sides are equal
  (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ a^2 + c^2 = b^2)  -- Pythagorean theorem holds

/-- The set {5, 5, 5√2} represents the sides of an isosceles right triangle -/
theorem isosceles_right_triangle_example : is_isosceles_right_triangle 5 5 (5 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_example_l1766_176612


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1766_176687

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2, 3}

theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {-2, -1, 0, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1766_176687


namespace NUMINAMATH_CALUDE_subtract_problem_l1766_176694

theorem subtract_problem (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtract_problem_l1766_176694


namespace NUMINAMATH_CALUDE_other_x_intercept_is_seven_l1766_176602

/-- A quadratic function with vertex (4, -3) and one x-intercept at (1, 0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 4
  vertex_y : ℝ := -3
  intercept_x : ℝ := 1

/-- The x-coordinate of the other x-intercept of the quadratic function -/
def other_x_intercept (f : QuadraticFunction) : ℝ := 7

/-- Theorem stating that the x-coordinate of the other x-intercept is 7 -/
theorem other_x_intercept_is_seven (f : QuadraticFunction) :
  other_x_intercept f = 7 := by sorry

end NUMINAMATH_CALUDE_other_x_intercept_is_seven_l1766_176602


namespace NUMINAMATH_CALUDE_expected_jumps_is_eight_l1766_176644

/-- Represents the behavior of a trainer --/
structure Trainer where
  jumps : ℕ
  gives_treat : Bool

/-- The expected number of jumps before getting a treat --/
def expected_jumps (trainers : List Trainer) : ℝ :=
  sorry

/-- The list of trainers with their behaviors --/
def dog_trainers : List Trainer :=
  [{ jumps := 0, gives_treat := true },
   { jumps := 5, gives_treat := true },
   { jumps := 3, gives_treat := false }]

/-- The main theorem stating the expected number of jumps --/
theorem expected_jumps_is_eight :
  expected_jumps dog_trainers = 8 := by
  sorry

end NUMINAMATH_CALUDE_expected_jumps_is_eight_l1766_176644


namespace NUMINAMATH_CALUDE_fraction_equality_l1766_176692

theorem fraction_equality (a b : ℝ) (h : a ≠ -b) : (-a + b) / (-a - b) = (a - b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1766_176692


namespace NUMINAMATH_CALUDE_exercise_distance_l1766_176639

/-- 
Proves that given a person who walks x miles at 3 miles per hour, 
runs 10 miles at 5 miles per hour, repeats this exercise 7 times a week, 
and spends a total of 21 hours exercising per week, 
the value of x must be 3 miles.
-/
theorem exercise_distance (x : ℝ) 
  (h_walk_speed : ℝ := 3)
  (h_run_speed : ℝ := 5)
  (h_run_distance : ℝ := 10)
  (h_days_per_week : ℕ := 7)
  (h_total_hours : ℝ := 21)
  (h_exercise_time : ℝ := h_total_hours / h_days_per_week)
  (h_time_equation : x / h_walk_speed + h_run_distance / h_run_speed = h_exercise_time) :
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_exercise_distance_l1766_176639


namespace NUMINAMATH_CALUDE_reciprocal_of_neg_sqrt_two_l1766_176656

theorem reciprocal_of_neg_sqrt_two :
  (1 : ℝ) / (-Real.sqrt 2) = -(Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_neg_sqrt_two_l1766_176656


namespace NUMINAMATH_CALUDE_gcd_119_153_l1766_176655

theorem gcd_119_153 : Nat.gcd 119 153 = 17 := by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_gcd_119_153_l1766_176655


namespace NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l1766_176689

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (hx : x > 0) : 
  2 * Real.sqrt x + 1 / x ≥ 3 ∧ 
  (2 * Real.sqrt x + 1 / x = 3 ↔ x = 1) := by
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l1766_176689


namespace NUMINAMATH_CALUDE_maurice_cookout_packages_l1766_176685

/-- The number of packages of ground beef Maurice needs to purchase for his cookout --/
def packages_needed (guests : ℕ) (burger_weight : ℕ) (package_weight : ℕ) : ℕ :=
  let total_people := guests + 1  -- Adding Maurice himself
  let total_weight := total_people * burger_weight
  (total_weight + package_weight - 1) / package_weight  -- Ceiling division

/-- Theorem stating that Maurice needs to purchase 4 packages of ground beef --/
theorem maurice_cookout_packages : packages_needed 9 2 5 = 4 := by
  sorry

#eval packages_needed 9 2 5

end NUMINAMATH_CALUDE_maurice_cookout_packages_l1766_176685


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1766_176657

theorem sufficient_not_necessary_condition :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x / y + y / x ≥ 2) ∧
  ¬(∀ x y : ℝ, x / y + y / x ≥ 2 → x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1766_176657


namespace NUMINAMATH_CALUDE_count_marquis_duels_l1766_176610

theorem count_marquis_duels (counts dukes marquises : ℕ) 
  (h1 : counts > 0) (h2 : dukes > 0) (h3 : marquises > 0)
  (h4 : 3 * counts = 2 * dukes)
  (h5 : 6 * dukes = 3 * marquises)
  (h6 : 2 * marquises = 2 * counts * k)
  (h7 : k > 0) :
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_marquis_duels_l1766_176610


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l1766_176691

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24 ∧ (a : ℕ) + b = 98 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 24 → (c : ℕ) + d ≥ 98 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l1766_176691


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_100_l1766_176643

theorem x_plus_2y_equals_100 (x y : ℝ) (h1 : y = 25) (h2 : x = 50) : x + 2*y = 100 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_100_l1766_176643


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1766_176640

theorem quadratic_one_solution (q : ℝ) :
  (∃! x : ℝ, q * x^2 - 10 * x + 2 = 0) ↔ q = 12.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1766_176640


namespace NUMINAMATH_CALUDE_quadratic_roots_existence_l1766_176637

theorem quadratic_roots_existence : 
  (∃ x : ℝ, x^2 + x = 0) ∧ 
  (∃ x : ℝ, 5*x^2 - 4*x - 1 = 0) ∧ 
  (∃ x : ℝ, 3*x^2 - 4*x + 1 = 0) ∧ 
  (∀ x : ℝ, 4*x^2 - 5*x + 2 ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_existence_l1766_176637


namespace NUMINAMATH_CALUDE_simplified_fourth_root_l1766_176621

theorem simplified_fourth_root (a b : ℕ+) :
  (2^9 * 3^5 : ℝ)^(1/4) = (a : ℝ) * ((b : ℝ)^(1/4)) → a + b = 18 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fourth_root_l1766_176621


namespace NUMINAMATH_CALUDE_sum_congruence_mod_seven_l1766_176683

theorem sum_congruence_mod_seven :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_mod_seven_l1766_176683


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1766_176615

theorem fraction_sum_equality : (3 : ℚ) / 5 - 2 / 15 + 1 / 3 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1766_176615


namespace NUMINAMATH_CALUDE_student_count_l1766_176616

/-- The number of students in the group -/
def num_students : ℕ := 6

/-- The weight decrease when replacing the heavier student with the lighter one -/
def weight_difference : ℕ := 80 - 62

/-- The average weight decrease per student -/
def avg_weight_decrease : ℕ := 3

theorem student_count :
  num_students * avg_weight_decrease = weight_difference :=
sorry

end NUMINAMATH_CALUDE_student_count_l1766_176616


namespace NUMINAMATH_CALUDE_linda_tees_sold_l1766_176660

/-- Calculates the number of tees sold given the prices, number of jeans sold, and total money -/
def tees_sold (jeans_price tee_price : ℕ) (jeans_sold : ℕ) (total_money : ℕ) : ℕ :=
  (total_money - jeans_price * jeans_sold) / tee_price

theorem linda_tees_sold :
  tees_sold 11 8 4 100 = 7 := by
  sorry

end NUMINAMATH_CALUDE_linda_tees_sold_l1766_176660


namespace NUMINAMATH_CALUDE_megan_earnings_l1766_176647

/-- The amount of money Megan earned from selling necklaces -/
def money_earned (bead_necklaces gem_necklaces cost_per_necklace : ℕ) : ℕ :=
  (bead_necklaces + gem_necklaces) * cost_per_necklace

/-- Theorem stating that Megan earned 90 dollars from selling necklaces -/
theorem megan_earnings : money_earned 7 3 9 = 90 := by
  sorry

end NUMINAMATH_CALUDE_megan_earnings_l1766_176647


namespace NUMINAMATH_CALUDE_line_properties_l1766_176667

/-- The line equation: (a+1)x + y + 2-a = 0 -/
def line_equation (a x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

/-- Equal intercepts on both coordinate axes -/
def equal_intercepts (a : ℝ) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ line_equation a t 0 ∧ line_equation a 0 t

/-- Line does not pass through the second quadrant -/
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a x y → ¬(x < 0 ∧ y > 0)

/-- Main theorem -/
theorem line_properties :
  (∀ a : ℝ, equal_intercepts a ↔ (a = 0 ∨ a = 2)) ∧
  (∀ a : ℝ, not_in_second_quadrant a ↔ a ≤ -1) := by sorry

end NUMINAMATH_CALUDE_line_properties_l1766_176667


namespace NUMINAMATH_CALUDE_untouchedShapesAfterGame_l1766_176613

/-- Represents a shape made of matches -/
inductive Shape
| Triangle
| Square
| Pentagon

/-- Represents the game state -/
structure GameState where
  triangles : Nat
  squares : Nat
  pentagons : Nat
  untouchedShapes : Nat
  currentPlayer : Bool  -- true for Petya, false for Vasya

/-- Represents a player's move -/
structure Move where
  shapeType : Shape
  isNewShape : Bool

/-- Optimal strategy for a player -/
def optimalMove (state : GameState) : Move :=
  sorry

/-- Apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Play the game for a given number of turns -/
def playGame (initialState : GameState) (turns : Nat) : GameState :=
  sorry

/-- The main theorem to prove -/
theorem untouchedShapesAfterGame :
  let initialState : GameState := {
    triangles := 3,
    squares := 4,
    pentagons := 5,
    untouchedShapes := 12,
    currentPlayer := true
  }
  let finalState := playGame initialState 10
  finalState.untouchedShapes = 6 := by
  sorry

end NUMINAMATH_CALUDE_untouchedShapesAfterGame_l1766_176613


namespace NUMINAMATH_CALUDE_expression_always_positive_l1766_176638

theorem expression_always_positive (a b : ℝ) : a^2 + b^2 + 4*b - 2*a + 6 > 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_positive_l1766_176638


namespace NUMINAMATH_CALUDE_equation_2x_squared_eq_1_is_quadratic_l1766_176672

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 2x^2 = 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- Theorem: The equation 2x^2 = 1 is a quadratic equation -/
theorem equation_2x_squared_eq_1_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_2x_squared_eq_1_is_quadratic_l1766_176672


namespace NUMINAMATH_CALUDE_function_equality_implies_m_value_l1766_176661

theorem function_equality_implies_m_value :
  ∀ (m : ℚ),
  let f : ℚ → ℚ := λ x => x^2 - 3*x + m
  let g : ℚ → ℚ := λ x => x^2 - 3*x + 5*m
  3 * f 5 = 2 * g 5 →
  m = 10/7 :=
by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_value_l1766_176661


namespace NUMINAMATH_CALUDE_davids_average_marks_l1766_176623

def english_marks : ℝ := 90
def mathematics_marks : ℝ := 92
def physics_marks : ℝ := 85
def chemistry_marks : ℝ := 87
def biology_marks : ℝ := 85

def total_marks : ℝ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def number_of_subjects : ℝ := 5

theorem davids_average_marks :
  total_marks / number_of_subjects = 87.8 := by
  sorry

end NUMINAMATH_CALUDE_davids_average_marks_l1766_176623


namespace NUMINAMATH_CALUDE_school_prizes_l1766_176603

theorem school_prizes (total_money : ℝ) (pen_cost notebook_cost : ℝ) 
  (h1 : total_money = 60 * (pen_cost + 2 * notebook_cost))
  (h2 : total_money = 50 * (pen_cost + 3 * notebook_cost)) :
  (total_money / pen_cost : ℝ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_school_prizes_l1766_176603


namespace NUMINAMATH_CALUDE_greatest_integer_third_side_l1766_176686

theorem greatest_integer_third_side (a b c : ℕ) : 
  a = 7 ∧ b = 10 ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧  -- Triangle inequality
  c ≤ a + b - 1 →                           -- Strict inequality
  (∀ d : ℕ, d > c → ¬(a + b > d ∧ a + d > b ∧ b + d > a)) →
  c = 16 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_third_side_l1766_176686


namespace NUMINAMATH_CALUDE_line_y_intercept_l1766_176604

/-- A line with slope 3 and x-intercept (7,0) has y-intercept (0, -21) -/
theorem line_y_intercept (m : ℝ) (x₀ : ℝ) (y : ℝ → ℝ) :
  m = 3 →
  x₀ = 7 →
  y 0 = 0 →
  (∀ x, y x = m * (x - x₀)) →
  y 0 = -21 :=
by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l1766_176604


namespace NUMINAMATH_CALUDE_f_composition_equals_126_l1766_176666

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x - 4

-- State the theorem
theorem f_composition_equals_126 : f (f (f 2)) = 126 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_126_l1766_176666


namespace NUMINAMATH_CALUDE_square_root_of_negative_two_fourth_power_l1766_176690

theorem square_root_of_negative_two_fourth_power :
  Real.sqrt ((-2)^4) = 4 ∨ Real.sqrt ((-2)^4) = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_negative_two_fourth_power_l1766_176690


namespace NUMINAMATH_CALUDE_min_handshakes_in_gathering_l1766_176631

/-- Represents a gathering of people and their handshakes -/
structure Gathering where
  people : Nat
  min_handshakes_per_person : Nat
  total_handshakes : Nat

/-- The minimum number of handshakes in a gathering of 30 people 
    where each person shakes hands with at least 3 others -/
theorem min_handshakes_in_gathering (g : Gathering) 
  (h1 : g.people = 30)
  (h2 : g.min_handshakes_per_person ≥ 3) :
  g.total_handshakes ≥ 45 ∧ 
  ∃ (arrangement : Gathering), 
    arrangement.people = 30 ∧ 
    arrangement.min_handshakes_per_person = 3 ∧ 
    arrangement.total_handshakes = 45 := by
  sorry

end NUMINAMATH_CALUDE_min_handshakes_in_gathering_l1766_176631


namespace NUMINAMATH_CALUDE_coat_price_l1766_176698

/-- The final price of a coat after discounts and tax -/
def finalPrice (originalPrice discountOne discountTwo coupon salesTax : ℚ) : ℚ :=
  ((originalPrice * (1 - discountOne) * (1 - discountTwo) - coupon) * (1 + salesTax))

/-- Theorem stating the final price of the coat -/
theorem coat_price : 
  finalPrice 150 0.3 0.1 10 0.05 = 88.725 := by sorry

end NUMINAMATH_CALUDE_coat_price_l1766_176698


namespace NUMINAMATH_CALUDE_max_pieces_is_100_l1766_176697

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 20

/-- The size of the small cake pieces in inches -/
def small_piece_size : ℕ := 2

/-- The area of the large cake in square inches -/
def large_cake_area : ℕ := large_cake_size * large_cake_size

/-- The area of a small cake piece in square inches -/
def small_piece_area : ℕ := small_piece_size * small_piece_size

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := large_cake_area / small_piece_area

theorem max_pieces_is_100 : max_pieces = 100 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_is_100_l1766_176697


namespace NUMINAMATH_CALUDE_max_log_product_l1766_176695

theorem max_log_product (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b = 100) :
  Real.log a * Real.log b ≤ 1 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 1 ∧ b₀ > 1 ∧ a₀ * b₀ = 100 ∧ Real.log a₀ * Real.log b₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_log_product_l1766_176695


namespace NUMINAMATH_CALUDE_bob_second_week_hours_l1766_176601

/-- Calculates the total pay for a given number of hours worked --/
def calculatePay (hours : ℕ) : ℕ :=
  if hours ≤ 40 then
    hours * 5
  else
    40 * 5 + (hours - 40) * 6

theorem bob_second_week_hours :
  ∃ (second_week_hours : ℕ),
    calculatePay 44 + calculatePay second_week_hours = 472 ∧
    second_week_hours = 48 := by
  sorry

end NUMINAMATH_CALUDE_bob_second_week_hours_l1766_176601


namespace NUMINAMATH_CALUDE_tom_barbados_trip_cost_l1766_176625

/-- The total cost for Tom's trip to Barbados -/
def total_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
  (insurance_coverage : ℚ) (trip_cost : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let out_of_pocket_medical := medical_cost * (1 - insurance_coverage)
  out_of_pocket_medical + trip_cost

/-- Theorem stating the total cost for Tom's trip to Barbados -/
theorem tom_barbados_trip_cost :
  total_cost 10 45 250 0.8 1200 = 1340 := by
  sorry

end NUMINAMATH_CALUDE_tom_barbados_trip_cost_l1766_176625


namespace NUMINAMATH_CALUDE_correct_sums_l1766_176652

theorem correct_sums (total : ℕ) (wrong_ratio : ℕ) (h1 : total = 54) (h2 : wrong_ratio = 2) :
  ∃ (correct : ℕ), correct * (1 + wrong_ratio) = total ∧ correct = 18 :=
by sorry

end NUMINAMATH_CALUDE_correct_sums_l1766_176652


namespace NUMINAMATH_CALUDE_smallest_ten_digit_max_sum_l1766_176624

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_ten_digit (n : Nat) : Prop :=
  1000000000 ≤ n ∧ n < 10000000000

theorem smallest_ten_digit_max_sum : 
  ∀ n : Nat, is_ten_digit n → n < 1999999999 → sum_of_digits n < sum_of_digits 1999999999 :=
sorry

#eval sum_of_digits 1999999999

end NUMINAMATH_CALUDE_smallest_ten_digit_max_sum_l1766_176624


namespace NUMINAMATH_CALUDE_cos_330_degrees_l1766_176679

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l1766_176679


namespace NUMINAMATH_CALUDE_constant_in_toll_formula_l1766_176633

/-- The toll formula for a truck crossing a bridge -/
def toll (x : ℕ) (constant : ℝ) : ℝ :=
  1.50 + 0.50 * (x - constant)

/-- The number of axles on an 18-wheel truck with 2 wheels on its front axle and 2 wheels on each of its other axles -/
def axles_18_wheel_truck : ℕ := 9

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll axles_18_wheel_truck constant = 5 ∧ 
    constant = 2 := by sorry

end NUMINAMATH_CALUDE_constant_in_toll_formula_l1766_176633


namespace NUMINAMATH_CALUDE_unique_three_digit_number_twelve_times_sum_of_digits_l1766_176600

theorem unique_three_digit_number_twelve_times_sum_of_digits : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    n = 12 * (n / 100 + (n / 10 % 10) + (n % 10)) := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_twelve_times_sum_of_digits_l1766_176600


namespace NUMINAMATH_CALUDE_comparison_theorem_l1766_176620

theorem comparison_theorem :
  (∀ m n : ℝ, m > n → -2*m + 1 < -2*n + 1) ∧
  (∀ m n a : ℝ, 
    (m < n ∧ a = 0 → m*a = n*a) ∧
    (m < n ∧ a > 0 → m*a < n*a) ∧
    (m < n ∧ a < 0 → m*a > n*a)) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l1766_176620


namespace NUMINAMATH_CALUDE_expression_simplification_l1766_176654

theorem expression_simplification (x : ℝ) (h : x = 2) :
  (1 / (x - 3)) / (1 / (x^2 - 9)) - (x / (x + 1)) * ((x^2 + x) / x^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1766_176654


namespace NUMINAMATH_CALUDE_modular_congruence_unique_solution_l1766_176649

theorem modular_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 38635 % 23 = n := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_unique_solution_l1766_176649


namespace NUMINAMATH_CALUDE_boat_speed_theorem_l1766_176665

/-- Represents the speed of a boat in a stream -/
structure BoatInStream where
  boatSpeed : ℝ  -- Speed of the boat in still water
  streamSpeed : ℝ  -- Speed of the stream

/-- Calculates the effective speed of the boat -/
def BoatInStream.effectiveSpeed (b : BoatInStream) (upstream : Bool) : ℝ :=
  if upstream then b.boatSpeed - b.streamSpeed else b.boatSpeed + b.streamSpeed

/-- Theorem: If the time taken to row upstream is twice the time taken to row downstream
    for the same distance, and the stream speed is 24, then the boat speed in still water is 72 -/
theorem boat_speed_theorem (b : BoatInStream) (distance : ℝ) 
    (h1 : b.streamSpeed = 24)
    (h2 : distance / b.effectiveSpeed true = 2 * (distance / b.effectiveSpeed false)) :
    b.boatSpeed = 72 := by
  sorry

#check boat_speed_theorem

end NUMINAMATH_CALUDE_boat_speed_theorem_l1766_176665


namespace NUMINAMATH_CALUDE_equation_solution_l1766_176658

theorem equation_solution : ∃ x : ℚ, (1 / 7 + 7 / x = 16 / x + 1 / 16) ∧ x = 112 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1766_176658


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_3_range_of_a_l1766_176677

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x + 1|

-- Part 1
theorem solution_set_when_a_eq_3 :
  {x : ℝ | f 3 x ≥ 2*x + 3} = {x : ℝ | x ≤ -1/4} := by sorry

-- Part 2
theorem range_of_a :
  ∀ a : ℝ, (∃ x ∈ Set.Icc 1 2, f a x ≤ |x - 5|) → a ∈ Set.Icc (-4) 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_3_range_of_a_l1766_176677


namespace NUMINAMATH_CALUDE_angle_subtraction_quadrant_l1766_176648

/-- An angle is in the second quadrant if it's between 90° and 180° -/
def is_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180

/-- An angle is in the first quadrant if it's between 0° and 90° -/
def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

theorem angle_subtraction_quadrant (α : ℝ) (h : is_second_quadrant α) : 
  is_first_quadrant (180 - α) := by
  sorry

end NUMINAMATH_CALUDE_angle_subtraction_quadrant_l1766_176648


namespace NUMINAMATH_CALUDE_sin_sum_special_angles_l1766_176608

theorem sin_sum_special_angles : 
  Real.sin (Real.arcsin (4/5) + Real.arctan (Real.sqrt 3)) = (2 + 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_special_angles_l1766_176608


namespace NUMINAMATH_CALUDE_digit_sum_subtraction_l1766_176681

theorem digit_sum_subtraction (M N P Q : ℕ) : 
  (M ≤ 9 ∧ N ≤ 9 ∧ P ≤ 9 ∧ Q ≤ 9) →
  (10 * M + N) + (10 * P + M) = 10 * Q + N →
  (10 * M + N) - (10 * P + M) = N →
  Q = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_subtraction_l1766_176681


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l1766_176699

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l1766_176699


namespace NUMINAMATH_CALUDE_ratio_solution_l1766_176618

theorem ratio_solution (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (3 * a + 4 * b) / (b + 12 * a) = 2) : 
  a / b = (5 - Real.sqrt 19) / 6 ∨ a / b = (5 + Real.sqrt 19) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_solution_l1766_176618


namespace NUMINAMATH_CALUDE_benny_apples_l1766_176680

theorem benny_apples (total : ℕ) (dan_apples : ℕ) (benny_apples : ℕ) :
  total = 11 → dan_apples = 9 → total = dan_apples + benny_apples → benny_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_benny_apples_l1766_176680


namespace NUMINAMATH_CALUDE_crabapple_theorem_l1766_176645

/-- The number of possible sequences of crabapple recipients in a week -/
def crabapple_sequences (num_students : ℕ) (classes_per_week : ℕ) : ℕ :=
  num_students ^ classes_per_week

/-- Theorem stating the number of possible sequences for Mrs. Crabapple's class -/
theorem crabapple_theorem :
  crabapple_sequences 15 5 = 759375 := by
  sorry

end NUMINAMATH_CALUDE_crabapple_theorem_l1766_176645


namespace NUMINAMATH_CALUDE_cube_edge_length_from_paint_cost_l1766_176675

/-- Proves that a cube with a specific edge length costs $1.60 to paint given certain paint properties -/
theorem cube_edge_length_from_paint_cost 
  (paint_cost_per_quart : ℝ) 
  (paint_coverage_per_quart : ℝ) 
  (total_paint_cost : ℝ) : 
  paint_cost_per_quart = 3.20 →
  paint_coverage_per_quart = 1200 →
  total_paint_cost = 1.60 →
  ∃ (edge_length : ℝ), 
    edge_length = 10 ∧ 
    total_paint_cost = (6 * edge_length^2) / paint_coverage_per_quart * paint_cost_per_quart :=
by
  sorry


end NUMINAMATH_CALUDE_cube_edge_length_from_paint_cost_l1766_176675


namespace NUMINAMATH_CALUDE_point_on_xOz_plane_l1766_176684

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane in 3D Cartesian space -/
def xOzPlane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- The given point (1, 0, 4) -/
def givenPoint : Point3D :=
  ⟨1, 0, 4⟩

/-- Theorem: The given point (1, 0, 4) lies on the xOz plane -/
theorem point_on_xOz_plane : givenPoint ∈ xOzPlane := by
  sorry

end NUMINAMATH_CALUDE_point_on_xOz_plane_l1766_176684


namespace NUMINAMATH_CALUDE_n3_equals_9_l1766_176614

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem n3_equals_9 
  (N : ℕ) 
  (h1 : 10^1989 ≤ 16*N ∧ 16*N < 10^1990) 
  (h2 : is_multiple_of_9 (16*N)) 
  (N1 : ℕ) (h3 : N1 = sum_of_digits N)
  (N2 : ℕ) (h4 : N2 = sum_of_digits N1)
  (N3 : ℕ) (h5 : N3 = sum_of_digits N2) :
  N3 = 9 :=
sorry

end NUMINAMATH_CALUDE_n3_equals_9_l1766_176614


namespace NUMINAMATH_CALUDE_bart_earnings_l1766_176693

/-- The amount of money Bart receives for each question he answers in a survey. -/
def amount_per_question : ℚ := 1/5

/-- The number of questions in each survey. -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart completed on Monday. -/
def monday_surveys : ℕ := 3

/-- The number of surveys Bart completed on Tuesday. -/
def tuesday_surveys : ℕ := 4

/-- The total amount of money Bart earned for the surveys completed on Monday and Tuesday. -/
def total_earnings : ℚ := 14

theorem bart_earnings : 
  amount_per_question * (questions_per_survey * (monday_surveys + tuesday_surveys)) = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_bart_earnings_l1766_176693


namespace NUMINAMATH_CALUDE_sum_of_four_rationals_l1766_176682

theorem sum_of_four_rationals (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Set ℚ) = 
    {-24, -2, -3/2, -1/8, 1, 3} → 
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_rationals_l1766_176682


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l1766_176659

/-- Given a rectangular metallic sheet with width 36 m, from which squares of 8 m are cut from each corner
    to form a box with volume 5120 m³, prove that the length of the original sheet is 48 m. -/
theorem metallic_sheet_length (L : ℝ) : 
  let W : ℝ := 36
  let cut_length : ℝ := 8
  let box_volume : ℝ := 5120
  (L - 2 * cut_length) * (W - 2 * cut_length) * cut_length = box_volume →
  L = 48 := by
sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_l1766_176659


namespace NUMINAMATH_CALUDE_cubic_function_zeros_l1766_176622

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem cubic_function_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_zeros_l1766_176622


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1766_176673

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 4*x - 3 > 0} = Set.Ioo 1 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1766_176673


namespace NUMINAMATH_CALUDE_expression_value_l1766_176674

theorem expression_value (a b c d x : ℝ) : 
  (c / 3 = -(-2 * d)) →
  (2 * a = 1 / (-b)) →
  (|x| = 9) →
  (2 * a * b - 6 * d + c - x / 3 = -4 ∨ 2 * a * b - 6 * d + c - x / 3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l1766_176674


namespace NUMINAMATH_CALUDE_least_beads_beads_solution_l1766_176634

theorem least_beads (b : ℕ) : 
  (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) → b ≥ 179 :=
by
  sorry

theorem beads_solution : 
  ∃ (b : ℕ), (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) ∧ b = 179 :=
by
  sorry

end NUMINAMATH_CALUDE_least_beads_beads_solution_l1766_176634


namespace NUMINAMATH_CALUDE_unique_mapping_l1766_176635

-- Define the property for the mapping
def SatisfiesProperty (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f (f n) ≤ (n + f n) / 2

-- Define the identity function on ℕ
def IdentityFunc : ℕ → ℕ := λ n => n

-- Theorem statement
theorem unique_mapping :
  ∀ f : ℕ → ℕ, Function.Injective f → SatisfiesProperty f → f = IdentityFunc :=
sorry

end NUMINAMATH_CALUDE_unique_mapping_l1766_176635


namespace NUMINAMATH_CALUDE_maximize_x2y5_l1766_176670

theorem maximize_x2y5 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 50) :
  x^2 * y^5 ≤ (100/7)^2 * (250/7)^5 ∧ 
  (x^2 * y^5 = (100/7)^2 * (250/7)^5 ↔ x = 100/7 ∧ y = 250/7) := by
sorry

end NUMINAMATH_CALUDE_maximize_x2y5_l1766_176670


namespace NUMINAMATH_CALUDE_equation_solution_l1766_176627

theorem equation_solution (x : ℝ) : 
  x ≠ -3 → (-x^2 = (3*x + 1) / (x + 3) ↔ x = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1766_176627


namespace NUMINAMATH_CALUDE_test_scores_mode_l1766_176617

/-- Represents a stem-and-leaf plot entry --/
structure StemLeafEntry where
  stem : ℕ
  leaves : List ℕ

/-- Finds the mode of a list of numbers --/
def mode (l : List ℕ) : ℕ := sorry

/-- Converts a stem-and-leaf plot to a list of numbers --/
def stemLeafToList (plot : List StemLeafEntry) : List ℕ := sorry

theorem test_scores_mode (plot : List StemLeafEntry) 
  (h1 : plot = [
    ⟨5, [1, 1]⟩,
    ⟨6, [5]⟩,
    ⟨7, [2, 4]⟩,
    ⟨8, [0, 3, 6, 6]⟩,
    ⟨9, [1, 5, 5, 5, 8, 8, 8]⟩,
    ⟨10, [2, 2, 2, 2, 4]⟩,
    ⟨11, [0, 0, 0]⟩
  ]) : 
  mode (stemLeafToList plot) = 102 := by sorry

end NUMINAMATH_CALUDE_test_scores_mode_l1766_176617


namespace NUMINAMATH_CALUDE_paths_count_l1766_176605

/-- The number of distinct paths from (0, n) to (m, m) on a plane,
    where only moves of 1 unit up or 1 unit left are allowed. -/
def numPaths (n m : ℕ) : ℕ :=
  Nat.choose n m

/-- Theorem stating that the number of distinct paths from (0, n) to (m, m)
    is equal to (n choose m) -/
theorem paths_count (n m : ℕ) (h : m ≤ n) :
  numPaths n m = Nat.choose n m := by
  sorry

end NUMINAMATH_CALUDE_paths_count_l1766_176605


namespace NUMINAMATH_CALUDE_thirty_three_not_enrolled_l1766_176696

/-- Calculates the number of students not enrolled in either French or German --/
def students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) : ℕ :=
  total - (french + german - both)

/-- Theorem stating that 33 students are not enrolled in either French or German --/
theorem thirty_three_not_enrolled : 
  students_not_enrolled 87 41 22 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_thirty_three_not_enrolled_l1766_176696


namespace NUMINAMATH_CALUDE_max_angle_APB_l1766_176653

-- Define the circles C and M
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1
def circle_M (x y θ : ℝ) : Prop := (x - 3 - 3 * Real.cos θ)^2 + (y - 3 * Real.sin θ)^2 = 1

-- Define a point P on circle M
def point_on_M (P : ℝ × ℝ) (θ : ℝ) : Prop := circle_M P.1 P.2 θ

-- Define points A and B on circle C
def points_on_C (A B : ℝ × ℝ) : Prop := circle_C A.1 A.2 ∧ circle_C B.1 B.2

-- Define the line PAB touching circle C
def line_touches_C (P A B : ℝ × ℝ) : Prop := 
  ∃ θ : ℝ, point_on_M P θ ∧ points_on_C A B

-- Theorem stating the maximum value of angle APB
theorem max_angle_APB : 
  ∀ P A B : ℝ × ℝ, line_touches_C P A B → 
  ∃ angle : ℝ, angle ≤ π / 3 ∧ 
  (∀ P' A' B' : ℝ × ℝ, line_touches_C P' A' B' → 
   ∃ angle' : ℝ, angle' ≤ angle) :=
sorry

end NUMINAMATH_CALUDE_max_angle_APB_l1766_176653


namespace NUMINAMATH_CALUDE_sum_of_magnitudes_of_roots_l1766_176636

theorem sum_of_magnitudes_of_roots (z₁ z₂ z₃ z₄ : ℂ) : 
  (z₁^4 + 3*z₁^3 + 3*z₁^2 + 3*z₁ + 1 = 0) →
  (z₂^4 + 3*z₂^3 + 3*z₂^2 + 3*z₂ + 1 = 0) →
  (z₃^4 + 3*z₃^3 + 3*z₃^2 + 3*z₃ + 1 = 0) →
  (z₄^4 + 3*z₄^3 + 3*z₄^2 + 3*z₄ + 1 = 0) →
  Complex.abs z₁ + Complex.abs z₂ + Complex.abs z₃ + Complex.abs z₄ = (7 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_magnitudes_of_roots_l1766_176636


namespace NUMINAMATH_CALUDE_trig_identity_l1766_176642

theorem trig_identity (α : Real) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1766_176642


namespace NUMINAMATH_CALUDE_sum_of_roots_l1766_176611

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

-- State the theorem
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1766_176611
