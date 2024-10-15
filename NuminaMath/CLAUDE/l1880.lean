import Mathlib

namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l1880_188079

theorem fraction_zero_implies_x_negative_three (x : ℝ) :
  (x^2 - 9) / (x - 3) = 0 ∧ x - 3 ≠ 0 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l1880_188079


namespace NUMINAMATH_CALUDE_chef_initial_apples_chef_had_46_apples_l1880_188046

/-- The number of apples a chef had initially, given the number of apples left after making pies
    and the difference between the initial and final number of apples. -/
theorem chef_initial_apples (apples_left : ℕ) (difference : ℕ) : ℕ :=
  apples_left + difference

/-- Proof that the chef initially had 46 apples -/
theorem chef_had_46_apples : chef_initial_apples 14 32 = 46 := by
  sorry

end NUMINAMATH_CALUDE_chef_initial_apples_chef_had_46_apples_l1880_188046


namespace NUMINAMATH_CALUDE_daughter_age_is_40_l1880_188066

/-- Represents the family members' weights and ages -/
structure Family where
  mother_weight : ℝ
  daughter_weight : ℝ
  grandchild_weight : ℝ
  son_in_law_weight : ℝ
  mother_age : ℝ
  daughter_age : ℝ
  son_in_law_age : ℝ

/-- The family satisfies the given conditions -/
def satisfies_conditions (f : Family) : Prop :=
  f.mother_weight + f.daughter_weight + f.grandchild_weight + f.son_in_law_weight = 200 ∧
  f.daughter_weight + f.grandchild_weight = 60 ∧
  f.grandchild_weight = (1/5) * f.mother_weight ∧
  f.son_in_law_weight = 2 * f.daughter_weight ∧
  f.mother_age / f.daughter_age = 2 ∧
  f.daughter_age / f.son_in_law_age = 3/2 ∧
  f.mother_age = 80

/-- The theorem stating that if a family satisfies the given conditions, the daughter's age is 40 -/
theorem daughter_age_is_40 (f : Family) (h : satisfies_conditions f) : f.daughter_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_daughter_age_is_40_l1880_188066


namespace NUMINAMATH_CALUDE_inverse_square_relation_l1880_188069

/-- Given that x varies inversely as the square of y, prove that x = 1 when y = 2,
    given that x = 0.1111111111111111 when y = 6. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) (h1 : x = k / y^2) 
    (h2 : 0.1111111111111111 = k / 6^2) : 
  1 = k / 2^2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l1880_188069


namespace NUMINAMATH_CALUDE_fourth_roll_five_probability_l1880_188044

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 6
def biased_die_five_prob : ℚ := 1 / 2
def biased_die_other_prob : ℚ := 1 / 10

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the number of rolls
def num_rolls : ℕ := 4

-- Define the probability of choosing each die
def choose_prob : ℚ := 1 / 2

-- Theorem statement
theorem fourth_roll_five_probability :
  let p_fair := fair_die_prob ^ 3 * choose_prob
  let p_biased := biased_die_five_prob ^ 3 * choose_prob
  let p_fair_given_three_fives := p_fair / (p_fair + p_biased)
  let p_biased_given_three_fives := p_biased / (p_fair + p_biased)
  p_fair_given_three_fives * fair_die_prob + p_biased_given_three_fives * biased_die_five_prob = 41 / 84 :=
by sorry

end NUMINAMATH_CALUDE_fourth_roll_five_probability_l1880_188044


namespace NUMINAMATH_CALUDE_N_bounds_l1880_188052

/-- The number of divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The number of ordered pairs (x,y) satisfying the given conditions -/
def N (p : ℕ) : ℕ := (Finset.filter (fun pair : ℕ × ℕ =>
  1 ≤ pair.1 ∧ pair.1 ≤ p * (p - 1) ∧
  1 ≤ pair.2 ∧ pair.2 ≤ p * (p - 1) ∧
  (pair.1 ^ pair.2) % p = 1 ∧
  (pair.2 ^ pair.1) % p = 1
) (Finset.product (Finset.range (p * (p - 1) + 1)) (Finset.range (p * (p - 1) + 1)))).card

theorem N_bounds (p : ℕ) (h : Nat.Prime p) :
  (Nat.totient (p - 1) * d (p - 1))^2 ≤ N p ∧ N p ≤ ((p - 1) * d (p - 1))^2 := by
  sorry

end NUMINAMATH_CALUDE_N_bounds_l1880_188052


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1880_188072

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (|x - 3| < 1 → x^2 + x - 6 > 0)) ∧
  (∃ x : ℝ, x^2 + x - 6 > 0 ∧ ¬(|x - 3| < 1)) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1880_188072


namespace NUMINAMATH_CALUDE_digit_deletion_divisibility_l1880_188005

theorem digit_deletion_divisibility (d : ℕ) (h : d > 0) : 
  ∃ (n n1 k a b c : ℕ), 
    n = 10^k * (10*a + b) + c ∧
    n1 = 10^k * a + c ∧
    0 < b ∧ b < 10 ∧
    c < 10^k ∧
    d ∣ n ∧
    d ∣ n1 :=
sorry

end NUMINAMATH_CALUDE_digit_deletion_divisibility_l1880_188005


namespace NUMINAMATH_CALUDE_certain_number_problem_l1880_188007

theorem certain_number_problem (x : ℝ) : 
  (x + 40 + 60) / 3 = (10 + 70 + 16) / 3 + 8 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1880_188007


namespace NUMINAMATH_CALUDE_checkerboard_domino_cover_l1880_188051

/-- A checkerboard is a rectangular grid of squares. -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- A domino covers exactly two squares. -/
def domino_cover := 2

/-- The total number of squares on a checkerboard. -/
def total_squares (board : Checkerboard) : ℕ :=
  board.rows * board.cols

/-- A checkerboard can be covered by dominoes if its total number of squares is even. -/
theorem checkerboard_domino_cover (board : Checkerboard) :
  ∃ (n : ℕ), total_squares board = n * domino_cover ↔ Even (total_squares board) :=
sorry

end NUMINAMATH_CALUDE_checkerboard_domino_cover_l1880_188051


namespace NUMINAMATH_CALUDE_conic_family_inscribed_in_square_l1880_188037

-- Define the square
def square : Set (ℝ × ℝ) :=
  {p | (p.1 = 3 ∨ p.1 = -3) ∧ p.2 ∈ [-3, 3] ∨
       (p.2 = 3 ∨ p.2 = -3) ∧ p.1 ∈ [-3, 3]}

-- Define the differential equation
def diff_eq (x y : ℝ) (dy_dx : ℝ) : Prop :=
  (9 - x^2) * dy_dx^2 = (9 - y^2)

-- Define a family of conics
def conic_family (C : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * (Real.cos C) * x * y = 9 * (Real.sin C)^2

-- State the theorem
theorem conic_family_inscribed_in_square :
  ∀ C : ℝ, ∃ p : ℝ × ℝ,
    p ∈ square ∧
    (∃ x y dy_dx : ℝ, diff_eq x y dy_dx ∧
      conic_family C x y ∧
      (x = p.1 ∧ y = p.2)) :=
sorry

end NUMINAMATH_CALUDE_conic_family_inscribed_in_square_l1880_188037


namespace NUMINAMATH_CALUDE_cubic_discriminant_l1880_188043

theorem cubic_discriminant (p q : ℝ) (x₁ x₂ x₃ : ℝ) : 
  x₁^3 + p*x₁ + q = 0 → 
  x₂^3 + p*x₂ + q = 0 → 
  x₃^3 + p*x₃ + q = 0 → 
  (x₁ - x₂)^2 * (x₂ - x₃)^2 * (x₃ - x₁)^2 = -4*p^3 - 27*q^2 := by
sorry

end NUMINAMATH_CALUDE_cubic_discriminant_l1880_188043


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1880_188094

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = ax² for some a ∈ ℚ -/
theorem functional_equation_solution :
  ∀ f : ℚ → ℚ, SatisfiesFunctionalEquation f →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 :=
by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l1880_188094


namespace NUMINAMATH_CALUDE_max_planes_from_three_parallel_lines_l1880_188074

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- A plane in 3D space -/
structure Plane where
  -- We don't need to define the internal structure of a plane for this problem

/-- Determines if two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Determines if a line lies on a plane -/
def lineOnPlane (l : Line3D) (p : Plane) : Prop :=
  sorry

/-- Determines if a plane is defined by two lines -/
def planeDefinedByLines (p : Plane) (l1 l2 : Line3D) : Prop :=
  sorry

/-- The main theorem: maximum number of planes defined by three parallel lines -/
theorem max_planes_from_three_parallel_lines (l1 l2 l3 : Line3D) 
  (h_parallel_12 : parallel l1 l2) 
  (h_parallel_23 : parallel l2 l3) 
  (h_parallel_13 : parallel l1 l3) :
  ∃ (p1 p2 p3 : Plane), 
    (∀ (p : Plane), (planeDefinedByLines p l1 l2 ∨ planeDefinedByLines p l2 l3 ∨ planeDefinedByLines p l1 l3) → 
      (p = p1 ∨ p = p2 ∨ p = p3)) ∧
    (∃ (p : Plane), planeDefinedByLines p l1 l2 ∧ planeDefinedByLines p l2 l3 ∧ planeDefinedByLines p l1 l3) →
      (p1 = p2 ∧ p2 = p3) :=
by
  sorry

end NUMINAMATH_CALUDE_max_planes_from_three_parallel_lines_l1880_188074


namespace NUMINAMATH_CALUDE_sum_of_squares_of_cubic_roots_l1880_188003

theorem sum_of_squares_of_cubic_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_cubic_roots_l1880_188003


namespace NUMINAMATH_CALUDE_min_sum_squares_l1880_188020

theorem min_sum_squares (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  ∀ m : ℝ, m = a^2 + b^2 + c^2 → m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1880_188020


namespace NUMINAMATH_CALUDE_product_xyz_equals_42_l1880_188061

theorem product_xyz_equals_42 (x y z : ℝ) 
  (h1 : y = x + 1) 
  (h2 : x + y = 2 * z) 
  (h3 : x = 3) : 
  x * y * z = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_equals_42_l1880_188061


namespace NUMINAMATH_CALUDE_min_cows_for_safe_ducks_l1880_188067

/-- Represents the arrangement of animals in Farmer Bill's circle -/
structure AnimalArrangement where
  total : Nat
  ducks : Nat
  cows : Nat
  rabbits : Nat

/-- Checks if the arrangement satisfies the safety condition for ducks -/
def isSafeArrangement (arr : AnimalArrangement) : Prop :=
  arr.ducks ≤ (arr.rabbits - 1) + 2 * arr.cows

/-- The main theorem stating the minimum number of cows required -/
theorem min_cows_for_safe_ducks (arr : AnimalArrangement) 
  (h1 : arr.total = 1000)
  (h2 : arr.ducks = 600)
  (h3 : arr.total = arr.ducks + arr.cows + arr.rabbits)
  (h4 : isSafeArrangement arr) :
  arr.cows ≥ 201 ∧ ∃ (safeArr : AnimalArrangement), 
    safeArr.total = 1000 ∧ 
    safeArr.ducks = 600 ∧ 
    safeArr.cows = 201 ∧
    isSafeArrangement safeArr :=
sorry

end NUMINAMATH_CALUDE_min_cows_for_safe_ducks_l1880_188067


namespace NUMINAMATH_CALUDE_car_distance_proof_l1880_188030

/-- Proves that a car traveling at 162 km/h for 5 hours covers a distance of 810 km -/
theorem car_distance_proof (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 162 → time = 5 → distance = speed * time → distance = 810 := by
sorry

end NUMINAMATH_CALUDE_car_distance_proof_l1880_188030


namespace NUMINAMATH_CALUDE_factorial_plus_one_divisible_implies_prime_l1880_188080

theorem factorial_plus_one_divisible_implies_prime (n : ℕ) :
  (Nat.factorial n + 1) % (n + 1) = 0 → Nat.Prime (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorial_plus_one_divisible_implies_prime_l1880_188080


namespace NUMINAMATH_CALUDE_cos_315_degrees_l1880_188000

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l1880_188000


namespace NUMINAMATH_CALUDE_different_elements_same_image_single_element_unique_image_elements_without_preimage_l1880_188042

-- Define the mapping f from A to B
variable {A B : Type}
variable (f : A → B)

-- Statement 1: Different elements in A can have the same image in B
theorem different_elements_same_image :
  ∃ (x y : A), x ≠ y ∧ f x = f y :=
sorry

-- Statement 2: A single element in A cannot have different images in B
theorem single_element_unique_image :
  ∀ (x : A) (y z : B), f x = y ∧ f x = z → y = z :=
sorry

-- Statement 3: There can be elements in B that do not have a pre-image in A
theorem elements_without_preimage :
  ∃ (y : B), ∀ (x : A), f x ≠ y :=
sorry

end NUMINAMATH_CALUDE_different_elements_same_image_single_element_unique_image_elements_without_preimage_l1880_188042


namespace NUMINAMATH_CALUDE_midpoint_distance_is_1300_l1880_188081

/-- The distance from the school to the midpoint of the total path -/
def midpoint_distance (school_to_kindergarten_km : ℕ) (school_to_kindergarten_m : ℕ) (kindergarten_to_house_m : ℕ) : ℕ :=
  ((school_to_kindergarten_km * 1000 + school_to_kindergarten_m + kindergarten_to_house_m) / 2)

/-- Theorem stating that the midpoint distance is 1300 meters -/
theorem midpoint_distance_is_1300 :
  midpoint_distance 1 700 900 = 1300 := by
  sorry

#eval midpoint_distance 1 700 900

end NUMINAMATH_CALUDE_midpoint_distance_is_1300_l1880_188081


namespace NUMINAMATH_CALUDE_soccer_survey_l1880_188092

/-- Represents the fraction of students who enjoy playing soccer -/
def enjoy_soccer : ℚ := 1/2

/-- Represents the fraction of students who honestly say they enjoy soccer among those who enjoy it -/
def honest_enjoy : ℚ := 7/10

/-- Represents the fraction of students who honestly say they do not enjoy soccer among those who do not enjoy it -/
def honest_not_enjoy : ℚ := 8/10

/-- The fraction of students who claim they do not enjoy playing soccer but actually enjoy it -/
def fraction_false_claim : ℚ := 3/11

theorem soccer_survey :
  (enjoy_soccer * (1 - honest_enjoy)) / 
  ((enjoy_soccer * (1 - honest_enjoy)) + ((1 - enjoy_soccer) * honest_not_enjoy)) = fraction_false_claim := by
  sorry

end NUMINAMATH_CALUDE_soccer_survey_l1880_188092


namespace NUMINAMATH_CALUDE_inequality_proof_l1880_188091

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0) 
  (h5 : x + y + z + t = 2) : 
  Real.sqrt (x^2 + z^2) + Real.sqrt (x^2 + 1) + Real.sqrt (z^2 + y^2) + 
  Real.sqrt (y^2 + t^2) + Real.sqrt (t^2 + 4) ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1880_188091


namespace NUMINAMATH_CALUDE_min_disks_for_jamal_files_l1880_188008

/-- Represents the minimum number of disks needed to store files --/
def min_disks (total_files : ℕ) (disk_capacity : ℚ) 
  (file_size_a : ℚ) (count_a : ℕ) 
  (file_size_b : ℚ) (count_b : ℕ) 
  (file_size_c : ℚ) : ℕ :=
  sorry

theorem min_disks_for_jamal_files : 
  min_disks 35 2 0.95 5 0.85 15 0.5 = 14 := by sorry

end NUMINAMATH_CALUDE_min_disks_for_jamal_files_l1880_188008


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l1880_188019

/-- Given two parabolas y = ax^2 + 4 and y = 6 - bx^2 that intersect the coordinate axes
    in exactly four points forming a kite with area 18, prove that a + b = 4/45 -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    -- First parabola intersects x-axis
    (a * x₁^2 + 4 = 0 ∧ a * x₂^2 + 4 = 0 ∧ x₁ ≠ x₂) ∧ 
    -- Second parabola intersects x-axis
    (6 - b * x₁^2 = 0 ∧ 6 - b * x₂^2 = 0 ∧ x₁ ≠ x₂) ∧
    -- First parabola intersects y-axis
    (a * 0^2 + 4 = y₁) ∧
    -- Second parabola intersects y-axis
    (6 - b * 0^2 = y₂) ∧
    -- Area of the kite formed by these points is 18
    (1/2 * (x₂ - x₁) * (y₂ - y₁) = 18)) →
  a + b = 4/45 := by
sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l1880_188019


namespace NUMINAMATH_CALUDE_three_right_angles_implies_rectangle_l1880_188001

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- An angle is right if it measures 90 degrees or π/2 radians. -/
def is_right_angle (q : Quadrilateral) (i : Fin 4) : Prop := sorry

/-- A rectangle is a quadrilateral with four right angles. -/
def is_rectangle (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, is_right_angle q i

/-- Theorem: If a quadrilateral has three right angles, it is a rectangle. -/
theorem three_right_angles_implies_rectangle (q : Quadrilateral) 
  (h : ∃ i j k : Fin 4, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    is_right_angle q i ∧ is_right_angle q j ∧ is_right_angle q k) : 
  is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_three_right_angles_implies_rectangle_l1880_188001


namespace NUMINAMATH_CALUDE_prob_at_least_one_heart_or_king_l1880_188038

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either a heart or a king -/
def target_cards : ℕ := 16

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The probability of drawing at least one heart or king in three draws with replacement -/
theorem prob_at_least_one_heart_or_king :
  1 - (((deck_size - target_cards : ℚ) / deck_size) ^ num_draws) = 1468 / 2197 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_heart_or_king_l1880_188038


namespace NUMINAMATH_CALUDE_five_divides_square_iff_five_divides_l1880_188010

theorem five_divides_square_iff_five_divides (a : ℤ) : 
  5 ∣ a^2 ↔ 5 ∣ a := by sorry

end NUMINAMATH_CALUDE_five_divides_square_iff_five_divides_l1880_188010


namespace NUMINAMATH_CALUDE_intersection_A_B_l1880_188086

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x^2 + 2*x < 0}

theorem intersection_A_B : A ∩ B = Set.Ioo (-1) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1880_188086


namespace NUMINAMATH_CALUDE_complex_power_four_l1880_188045

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_four (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l1880_188045


namespace NUMINAMATH_CALUDE_square_root_approximation_l1880_188064

theorem square_root_approximation : ∃ ε > 0, ε < 0.0001 ∧ 
  |Real.sqrt ((16^10 + 32^10) / (16^6 + 32^11)) - 0.1768| < ε :=
by sorry

end NUMINAMATH_CALUDE_square_root_approximation_l1880_188064


namespace NUMINAMATH_CALUDE_no_solution_fibonacci_equation_l1880_188089

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem no_solution_fibonacci_equation :
  ∀ n : ℕ, n * (fib n) * (fib (n + 1)) ≠ (fib (n + 2) - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_fibonacci_equation_l1880_188089


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1880_188062

theorem quadratic_equation_coefficients 
  (a b c : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : (7 : ℝ) * ((7 : ℝ) * a + b) + c = 0) 
  (h3 : (-1 : ℝ) * ((-1 : ℝ) * a + b) + c = 0) :
  b = -6 ∧ c = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1880_188062


namespace NUMINAMATH_CALUDE_subset_intersection_cardinality_l1880_188004

theorem subset_intersection_cardinality (n m : ℕ) (Z : Finset ℕ) 
  (A : Fin m → Finset ℕ) : 
  (Z.card = n) →
  (∀ i : Fin m, A i ⊂ Z) →
  (∀ i j : Fin m, i ≠ j → (A i ∩ A j).card = 1) →
  m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_cardinality_l1880_188004


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_l1880_188060

theorem sum_of_squares_zero (a b c : ℝ) : 
  (a - 2)^2 + (b + 3)^2 + (c - 7)^2 = 0 → a + b + c = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_l1880_188060


namespace NUMINAMATH_CALUDE_solve_equation_l1880_188017

theorem solve_equation (n m x : ℚ) : 
  (5 / 7 : ℚ) = n / 91 ∧ 
  (5 / 7 : ℚ) = (m + n) / 105 ∧ 
  (5 / 7 : ℚ) = (x - m) / 140 → 
  x = 110 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l1880_188017


namespace NUMINAMATH_CALUDE_decoration_time_proof_l1880_188024

/-- The time needed for Mia and Billy to decorate Easter eggs -/
def decoration_time (mia_rate : ℕ) (billy_rate : ℕ) (total_eggs : ℕ) : ℚ :=
  total_eggs / (mia_rate + billy_rate)

/-- Theorem stating that Mia and Billy will take 5 hours to decorate 170 eggs -/
theorem decoration_time_proof :
  decoration_time 24 10 170 = 5 := by
  sorry

end NUMINAMATH_CALUDE_decoration_time_proof_l1880_188024


namespace NUMINAMATH_CALUDE_special_isosceles_triangle_sides_l1880_188048

/-- An isosceles triangle with specific incenter properties -/
structure SpecialIsoscelesTriangle where
  -- The length of the two equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- The distance from the vertex to the incenter along the altitude
  vertexToIncenter : ℝ
  -- The distance from the incenter to the base along the altitude
  incenterToBase : ℝ
  -- Ensure the triangle is isosceles
  isIsosceles : side > 0
  -- Ensure the incenter divides the altitude as specified
  incenterDivision : vertexToIncenter = 5 ∧ incenterToBase = 3

/-- The theorem stating the side lengths of the special isosceles triangle -/
theorem special_isosceles_triangle_sides 
  (t : SpecialIsoscelesTriangle) : t.side = 10 ∧ t.base = 12 := by
  sorry

#check special_isosceles_triangle_sides

end NUMINAMATH_CALUDE_special_isosceles_triangle_sides_l1880_188048


namespace NUMINAMATH_CALUDE_rupert_ronald_jumps_l1880_188059

theorem rupert_ronald_jumps 
  (ronald_jumps : ℕ) 
  (total_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : total_jumps = 243)
  (h3 : ronald_jumps < total_jumps - ronald_jumps) :
  total_jumps - ronald_jumps - ronald_jumps = 86 := by
  sorry

end NUMINAMATH_CALUDE_rupert_ronald_jumps_l1880_188059


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1880_188033

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1880_188033


namespace NUMINAMATH_CALUDE_det_A_l1880_188065

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 8, 5, -4; 3, 3, 7]

def A' : Matrix (Fin 3) (Fin 3) ℤ := 
  Matrix.of (λ i j => 
    if i = 0 then A i j
    else A i j - A 0 j)

theorem det_A'_eq_55 : Matrix.det A' = 55 := by sorry

end NUMINAMATH_CALUDE_det_A_l1880_188065


namespace NUMINAMATH_CALUDE_factor_expression_l1880_188096

theorem factor_expression (x : ℝ) : 63 * x^2 + 28 * x = 7 * x * (9 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1880_188096


namespace NUMINAMATH_CALUDE_chandler_savings_weeks_l1880_188097

def bike_cost : ℕ := 650
def birthday_money : ℕ := 50 + 35 + 15 + 20
def weekly_earnings : ℕ := 18

def weeks_to_save (cost birthday_money weekly_earnings : ℕ) : ℕ :=
  ((cost - birthday_money + weekly_earnings - 1) / weekly_earnings)

theorem chandler_savings_weeks :
  weeks_to_save bike_cost birthday_money weekly_earnings = 30 := by
  sorry

end NUMINAMATH_CALUDE_chandler_savings_weeks_l1880_188097


namespace NUMINAMATH_CALUDE_distance_between_trees_l1880_188016

/-- Given a yard of length 275 meters with 26 trees planted at equal distances,
    including one at each end, the distance between consecutive trees is 11 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 275 →
  num_trees = 26 →
  yard_length / (num_trees - 1) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l1880_188016


namespace NUMINAMATH_CALUDE_sum_of_legs_is_48_l1880_188099

/-- A right triangle with consecutive even whole number legs and hypotenuse 34 -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ
  hypotenuse : ℕ
  is_right : leg1^2 + leg2^2 = hypotenuse^2
  consecutive_even : leg2 = leg1 + 2
  hypotenuse_34 : hypotenuse = 34

/-- The sum of the legs of the special right triangle is 48 -/
theorem sum_of_legs_is_48 (t : RightTriangle) : t.leg1 + t.leg2 = 48 := by
  sorry

#check sum_of_legs_is_48

end NUMINAMATH_CALUDE_sum_of_legs_is_48_l1880_188099


namespace NUMINAMATH_CALUDE_unique_triple_l1880_188006

theorem unique_triple : ∃! (x y z : ℕ+), 
  (z > 1) ∧ 
  ((y + 1 : ℕ) % x = 0) ∧ 
  ((z - 1 : ℕ) % y = 0) ∧ 
  ((x^2 + 1 : ℕ) % z = 0) ∧
  x = 1 ∧ y = 1 ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_triple_l1880_188006


namespace NUMINAMATH_CALUDE_logarithm_sum_inequality_l1880_188088

theorem logarithm_sum_inequality : 
  Real.log 6 / Real.log 5 + Real.log 7 / Real.log 6 + Real.log 8 / Real.log 7 + Real.log 5 / Real.log 8 > 4 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_inequality_l1880_188088


namespace NUMINAMATH_CALUDE_base_conversion_3012_to_octal_l1880_188027

theorem base_conversion_3012_to_octal :
  (3012 : ℕ) = 5 * (8 : ℕ)^3 + 7 * (8 : ℕ)^2 + 0 * (8 : ℕ)^1 + 4 * (8 : ℕ)^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_3012_to_octal_l1880_188027


namespace NUMINAMATH_CALUDE_regression_unit_increase_survey_regression_unit_increase_l1880_188095

/-- Linear regression equation parameters -/
structure RegressionParams where
  slope : ℝ
  intercept : ℝ

/-- Calculates the predicted y value for a given x -/
def predict (params : RegressionParams) (x : ℝ) : ℝ :=
  params.slope * x + params.intercept

/-- Theorem: The difference in predicted y when x increases by 1 is equal to the slope -/
theorem regression_unit_increase (params : RegressionParams) :
  ∀ x : ℝ, predict params (x + 1) - predict params x = params.slope := by
  sorry

/-- The specific regression equation from the problem -/
def survey_regression : RegressionParams :=
  { slope := 0.254, intercept := 0.321 }

/-- Theorem: For the given survey regression, the difference in predicted y
    when x increases by 1 is equal to 0.254 -/
theorem survey_regression_unit_increase :
  ∀ x : ℝ, predict survey_regression (x + 1) - predict survey_regression x = 0.254 := by
  sorry

end NUMINAMATH_CALUDE_regression_unit_increase_survey_regression_unit_increase_l1880_188095


namespace NUMINAMATH_CALUDE_negate_negate_eq_self_l1880_188090

theorem negate_negate_eq_self (n : ℤ) : -(-n) = n := by sorry

end NUMINAMATH_CALUDE_negate_negate_eq_self_l1880_188090


namespace NUMINAMATH_CALUDE_six_pieces_per_small_load_l1880_188073

/-- Given a total number of clothing pieces, the number of pieces in the first load,
    and the number of small loads, calculate the number of pieces in each small load. -/
def clothingPerSmallLoad (total : ℕ) (firstLoad : ℕ) (smallLoads : ℕ) : ℕ :=
  (total - firstLoad) / smallLoads

/-- Theorem stating that with 47 total pieces, 17 in the first load, and 5 small loads,
    each small load contains 6 pieces of clothing. -/
theorem six_pieces_per_small_load :
  clothingPerSmallLoad 47 17 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_pieces_per_small_load_l1880_188073


namespace NUMINAMATH_CALUDE_number_exists_l1880_188034

theorem number_exists : ∃ x : ℝ, x * 1.6 - (2 * 1.4) / 1.3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l1880_188034


namespace NUMINAMATH_CALUDE_product_of_coefficients_l1880_188036

theorem product_of_coefficients (b c : ℤ) : 
  (∀ r : ℝ, r^2 - 2*r - 1 = 0 → r^5 - b*r - c = 0) → 
  b * c = 348 := by
sorry

end NUMINAMATH_CALUDE_product_of_coefficients_l1880_188036


namespace NUMINAMATH_CALUDE_number_ordering_l1880_188012

theorem number_ordering (a b : ℝ) (ha : a > 0) (hb : 0 < b ∧ b < 1) :
  a^b > b^a ∧ b^a > Real.log b := by sorry

end NUMINAMATH_CALUDE_number_ordering_l1880_188012


namespace NUMINAMATH_CALUDE_rubble_short_by_8_75_l1880_188063

def initial_amount : ℚ := 45
def notebook_cost : ℚ := 4
def pen_cost : ℚ := 1.5
def eraser_cost : ℚ := 2.25
def pencil_case_cost : ℚ := 7.5
def notebook_count : ℕ := 5
def pen_count : ℕ := 8
def eraser_count : ℕ := 3
def pencil_case_count : ℕ := 2

def total_cost : ℚ :=
  notebook_cost * notebook_count +
  pen_cost * pen_count +
  eraser_cost * eraser_count +
  pencil_case_cost * pencil_case_count

theorem rubble_short_by_8_75 :
  initial_amount - total_cost = -8.75 := by
  sorry

end NUMINAMATH_CALUDE_rubble_short_by_8_75_l1880_188063


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l1880_188082

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → 
  a = 25 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l1880_188082


namespace NUMINAMATH_CALUDE_carters_baseball_cards_l1880_188078

/-- Given that Marcus has 350 baseball cards and 95 more cards than Carter,
    prove that Carter has 255 baseball cards. -/
theorem carters_baseball_cards : 
  ∀ (marcus_cards carter_cards : ℕ), 
    marcus_cards = 350 → 
    marcus_cards = carter_cards + 95 →
    carter_cards = 255 := by
  sorry

end NUMINAMATH_CALUDE_carters_baseball_cards_l1880_188078


namespace NUMINAMATH_CALUDE_quadratic_a_value_l1880_188047

/-- A quadratic function with vertex (h, k) passing through point (x₀, y₀) -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ
  vertex_form : ∀ x, a * (x - h)^2 + k = a * x^2 + (a * h * (-2)) * x + (a * h^2 + k)
  passes_through : a * (x₀ - h)^2 + k = y₀

/-- The theorem stating that for a quadratic function with vertex (3, 5) passing through (0, -20), a = -25/9 -/
theorem quadratic_a_value (f : QuadraticFunction) 
    (h_vertex : f.h = 3 ∧ f.k = 5) 
    (h_point : f.x₀ = 0 ∧ f.y₀ = -20) : 
    f.a = -25/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_a_value_l1880_188047


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l1880_188009

theorem quadratic_root_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x + 2 = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₁ + x₂ = -4 → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l1880_188009


namespace NUMINAMATH_CALUDE_homework_problem_count_l1880_188085

/-- The number of sub tasks per homework problem -/
def sub_tasks_per_problem : ℕ := 5

/-- The total number of sub tasks to solve -/
def total_sub_tasks : ℕ := 200

/-- The total number of homework problems -/
def total_problems : ℕ := total_sub_tasks / sub_tasks_per_problem

theorem homework_problem_count :
  total_problems = 40 :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_count_l1880_188085


namespace NUMINAMATH_CALUDE_sum_minimized_at_24_l1880_188041

/-- The sum of the first n terms of an arithmetic sequence with general term a_n = 2n - 49 -/
def S (n : ℕ) : ℝ := n^2 - 48*n

/-- The value of n that minimizes S_n -/
def n_min : ℕ := 24

theorem sum_minimized_at_24 :
  ∀ n : ℕ, n ≠ 0 → S n ≥ S n_min := by sorry

end NUMINAMATH_CALUDE_sum_minimized_at_24_l1880_188041


namespace NUMINAMATH_CALUDE_simple_interest_time_calculation_l1880_188021

theorem simple_interest_time_calculation (P : ℝ) (h1 : P > 0) : ∃ T : ℝ,
  (P * 5 * T) / 100 = P / 5 ∧ T = 4 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_calculation_l1880_188021


namespace NUMINAMATH_CALUDE_min_distance_squared_l1880_188022

theorem min_distance_squared (a b c d : ℝ) 
  (h1 : Real.log a - Real.log 3 = Real.log c) 
  (h2 : b * d = -3) : 
  ∃ (min_val : ℝ), min_val = 18/5 ∧ 
    ∀ (x y : ℝ), (x - b)^2 + (y - c)^2 ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l1880_188022


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_six_sqrt_two_l1880_188049

theorem sqrt_difference_equals_negative_six_sqrt_two :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) - Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = -6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_six_sqrt_two_l1880_188049


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1880_188014

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((2 / 5 : ℚ) + (4 / 7 : ℚ)) = (17 / 35 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1880_188014


namespace NUMINAMATH_CALUDE_root_between_roots_l1880_188053

theorem root_between_roots (a b c r s : ℝ) 
  (hr : a * r^2 + b * r + c = 0)
  (hs : -a * s^2 + b * s + c = 0) :
  ∃ t : ℝ, (t > r ∧ t < s ∨ t > s ∧ t < r) ∧ a/2 * t^2 + b * t + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_between_roots_l1880_188053


namespace NUMINAMATH_CALUDE_first_robber_guarantee_l1880_188084

/-- Represents the coin division game between two robbers --/
structure CoinGame where
  totalCoins : Nat
  maxBags : Nat

/-- Calculates the guaranteed minimum coins for the first robber --/
def guaranteedCoins (game : CoinGame) : Nat :=
  game.totalCoins - (game.maxBags - 1) * (game.totalCoins / (2 * game.maxBags - 1))

/-- Theorem: In a game with 300 coins and 11 max bags, the first robber can guarantee at least 146 coins --/
theorem first_robber_guarantee (game : CoinGame) 
  (h1 : game.totalCoins = 300) 
  (h2 : game.maxBags = 11) : 
  guaranteedCoins game ≥ 146 := by
  sorry

#eval guaranteedCoins { totalCoins := 300, maxBags := 11 }

end NUMINAMATH_CALUDE_first_robber_guarantee_l1880_188084


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_equals_four_l1880_188013

def A (a : ℝ) : Set ℝ := {x | a * x^2 + a * x + 1 = 0}

theorem unique_solution_implies_a_equals_four (a : ℝ) :
  (∃! x, x ∈ A a) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_equals_four_l1880_188013


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_l1880_188098

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x : ℝ | -a < x ∧ x ≤ a + 3}

-- Theorem statement
theorem set_operations_and_intersection (a : ℝ) : 
  (A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 8}) ∧ 
  ((Aᶜ : Set ℝ) ∩ B = {x : ℝ | 5 ≤ x ∧ x < 8}) ∧ 
  (C a ∩ A = C a ↔ a ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_l1880_188098


namespace NUMINAMATH_CALUDE_exponent_division_l1880_188050

theorem exponent_division (a : ℝ) (m n : ℕ) (h : a ≠ 0) : a ^ m / a ^ n = a ^ (m - n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1880_188050


namespace NUMINAMATH_CALUDE_survey_respondents_l1880_188025

theorem survey_respondents (prefer_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  prefer_x = 150 → ratio_x = 5 → ratio_y = 1 →
  ∃ (total : ℕ), total = prefer_x + (prefer_x / ratio_x * ratio_y) ∧ total = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l1880_188025


namespace NUMINAMATH_CALUDE_average_income_l1880_188015

/-- Given the average monthly incomes of pairs of individuals and the income of one individual,
    prove the average monthly income of a specific pair. -/
theorem average_income (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  P = 4000 →
  (P + R) / 2 = 5200 := by
  sorry

end NUMINAMATH_CALUDE_average_income_l1880_188015


namespace NUMINAMATH_CALUDE_bike_cost_calculation_l1880_188087

/-- The cost of Carrie's bike --/
def bike_cost (hourly_wage : ℕ) (weekly_hours : ℕ) (weeks_per_month : ℕ) (remaining_money : ℕ) : ℕ :=
  hourly_wage * weekly_hours * weeks_per_month - remaining_money

/-- Theorem stating the cost of the bike --/
theorem bike_cost_calculation :
  bike_cost 8 35 4 720 = 400 := by
  sorry

end NUMINAMATH_CALUDE_bike_cost_calculation_l1880_188087


namespace NUMINAMATH_CALUDE_distance_between_vertices_l1880_188077

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 1) = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -2)

-- Theorem statement
theorem distance_between_vertices :
  let (x1, y1) := vertex1
  let (x2, y2) := vertex2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l1880_188077


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l1880_188093

/-- The length of the generatrix of a cone with base radius √2 and lateral surface forming a semicircle when unfolded is 2√2. -/
theorem cone_generatrix_length :
  ∀ (base_radius : ℝ) (generatrix_length : ℝ),
  base_radius = Real.sqrt 2 →
  2 * Real.pi * base_radius = Real.pi * generatrix_length →
  generatrix_length = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l1880_188093


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1880_188057

theorem functional_equation_solution (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (2*m + 2*n) = f m * f n) : 
  ∀ x : ℕ, f x = 1 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1880_188057


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l1880_188058

/-- The cost of a movie ticket in dollars -/
def ticket_cost : ℝ := 5

/-- The cost of popcorn in dollars -/
def popcorn_cost : ℝ := 0.8 * ticket_cost

/-- The cost of soda in dollars -/
def soda_cost : ℝ := 0.5 * popcorn_cost

/-- Theorem stating that the given conditions result in a ticket cost of $5 -/
theorem movie_ticket_cost : 
  4 * ticket_cost + 2 * popcorn_cost + 4 * soda_cost = 36 := by
  sorry


end NUMINAMATH_CALUDE_movie_ticket_cost_l1880_188058


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1880_188056

/-- Given that x² and y vary inversely, prove that when y = 20 for x = 3,
    then x = 3√10/50 when y = 5000 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) : 
  (∀ x y, x^2 * y = k) →  -- x² and y vary inversely
  (3^2 * 20 = k) →        -- y = 20 when x = 3
  (x^2 * 5000 = k) →      -- y = 5000 for the x we're looking for
  x = 3 * Real.sqrt 10 / 50 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1880_188056


namespace NUMINAMATH_CALUDE_compacted_space_calculation_l1880_188039

/-- The number of cans Nick has -/
def num_cans : ℕ := 100

/-- The space each can takes up before compaction (in square inches) -/
def initial_space : ℝ := 30

/-- The percentage of space each can takes up after compaction -/
def compaction_ratio : ℝ := 0.35

/-- The total space occupied by all cans after compaction (in square inches) -/
def total_compacted_space : ℝ := num_cans * (initial_space * compaction_ratio)

theorem compacted_space_calculation :
  total_compacted_space = 1050 := by sorry

end NUMINAMATH_CALUDE_compacted_space_calculation_l1880_188039


namespace NUMINAMATH_CALUDE_exists_common_divisor_l1880_188068

/-- A function from positive integers to integers greater than 1 -/
def PositiveFunction := ℕ+ → ℕ+

/-- The property that f(m+n) divides f(m) + f(n) for all positive integers m and n -/
def HasDivisibilityProperty (f : PositiveFunction) : Prop :=
  ∀ m n : ℕ+, (f (m + n)) ∣ (f m + f n)

/-- The main theorem: if f has the divisibility property, then there exists c > 1 that divides all values of f -/
theorem exists_common_divisor (f : PositiveFunction) (h : HasDivisibilityProperty f) :
  ∃ c : ℕ+, c > 1 ∧ ∀ n : ℕ+, c ∣ f n :=
sorry

end NUMINAMATH_CALUDE_exists_common_divisor_l1880_188068


namespace NUMINAMATH_CALUDE_sales_solution_l1880_188055

def sales_problem (month1 month2 month4 month5 month6 average : ℕ) : Prop :=
  let total := average * 6
  let known_sum := month1 + month2 + month4 + month5 + month6
  let month3 := total - known_sum
  month3 = 7855

theorem sales_solution :
  sales_problem 7435 7920 8230 7560 6000 7500 := by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l1880_188055


namespace NUMINAMATH_CALUDE_line_divides_l_shape_in_half_l1880_188011

/-- L-shaped region in the xy-plane -/
structure LShapedRegion where
  vertices : List (ℝ × ℝ) := [(0,0), (0,4), (4,4), (4,2), (6,2), (6,0)]

/-- Line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the area of a polygon given its vertices -/
def calculateArea (vertices : List (ℝ × ℝ)) : ℝ := sorry

/-- Calculate the slope of a line -/
def calculateSlope (l : Line) : ℝ := sorry

/-- Check if a line divides a region in half -/
def divides_in_half (l : Line) (r : LShapedRegion) : Prop := sorry

/-- Theorem: The line through (0,0) and (2,4) divides the L-shaped region in half -/
theorem line_divides_l_shape_in_half :
  let l : Line := { point1 := (0, 0), point2 := (2, 4) }
  let r : LShapedRegion := {}
  divides_in_half l r ∧ calculateSlope l = 2 := by sorry

end NUMINAMATH_CALUDE_line_divides_l_shape_in_half_l1880_188011


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1880_188071

/-- The set of all possible integer roots for the polynomial x^4 + 4x^3 + a_2 x^2 + a_1 x - 60 = 0 -/
def possible_roots : Set ℤ := {1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 10, -10, 12, -12, 15, -15, 20, -20, 30, -30, 60, -60}

/-- The polynomial x^4 + 4x^3 + a_2 x^2 + a_1 x - 60 -/
def polynomial (a₂ a₁ x : ℤ) : ℤ := x^4 + 4*x^3 + a₂*x^2 + a₁*x - 60

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  ∀ x : ℤ, polynomial a₂ a₁ x = 0 → x ∈ possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1880_188071


namespace NUMINAMATH_CALUDE_walnut_trees_count_l1880_188018

/-- The number of walnut trees in the park after planting and removing trees -/
def final_tree_count (initial_trees : ℕ) (planted_group1 : ℕ) (planted_group2 : ℕ) (planted_group3 : ℕ) (removed_trees : ℕ) : ℕ :=
  initial_trees + planted_group1 + planted_group2 + planted_group3 - removed_trees

/-- Theorem stating that the final number of walnut trees in the park is 55 -/
theorem walnut_trees_count : final_tree_count 22 12 15 10 4 = 55 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_count_l1880_188018


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1880_188031

/-- An isosceles triangle with side lengths 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c => 
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Two sides are 9, one side is 4
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (b = c) →  -- Isosceles condition
    (a + b + c = 22)  -- Perimeter is 22

#check isosceles_triangle_perimeter

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : 
  ∃ (a b c : ℝ), isosceles_triangle_perimeter a b c :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1880_188031


namespace NUMINAMATH_CALUDE_tangent_line_and_bounds_l1880_188023

/-- The function f(x) = (ax+b)e^(-2x) -/
noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) * Real.exp (-2 * x)

/-- The function g(x) = f(x) + x * ln(x) -/
noncomputable def g (a b x : ℝ) : ℝ := f a b x + x * Real.log x

theorem tangent_line_and_bounds
  (a b : ℝ)
  (h1 : f a b 0 = 1)  -- f(0) = 1 from the tangent line equation
  (h2 : (deriv (f a b)) 0 = -1)  -- f'(0) = -1 from the tangent line equation
  : a = 1 ∧ b = 1 ∧ ∀ x, 0 < x → x < 1 → 2 * Real.exp (-2) - Real.exp (-1) < g a b x ∧ g a b x < 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_bounds_l1880_188023


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l1880_188076

def U : Set ℕ := {x | x < 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_complement_equals_set : B ∪ (U \ A) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l1880_188076


namespace NUMINAMATH_CALUDE_lending_duration_l1880_188040

/-- Proves that the number of years the first part is lent is 8, given the problem conditions -/
theorem lending_duration (total_sum : ℚ) (second_part : ℚ) 
  (first_rate : ℚ) (second_rate : ℚ) (second_duration : ℚ) :
  total_sum = 2678 →
  second_part = 1648 →
  first_rate = 3/100 →
  second_rate = 5/100 →
  second_duration = 3 →
  ∃ (first_duration : ℚ),
    (total_sum - second_part) * first_rate * first_duration = 
    second_part * second_rate * second_duration ∧
    first_duration = 8 := by
  sorry

end NUMINAMATH_CALUDE_lending_duration_l1880_188040


namespace NUMINAMATH_CALUDE_coffee_blend_weight_l1880_188002

/-- Represents the total weight of a coffee blend -/
def total_blend_weight (price_a price_b price_mix : ℚ) (weight_a : ℚ) : ℚ :=
  weight_a + (price_a * weight_a - price_mix * weight_a) / (price_mix - price_b)

/-- Theorem stating the total weight of the coffee blend -/
theorem coffee_blend_weight :
  total_blend_weight 9 8 (84/10) 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_coffee_blend_weight_l1880_188002


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_z_l1880_188054

theorem inequality_holds_for_all_z (x y : ℝ) (hx : x > 0) :
  ∀ z : ℝ, y - z < Real.sqrt (z^2 + x^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_z_l1880_188054


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1880_188075

open Set

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x * (x - 2) < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1880_188075


namespace NUMINAMATH_CALUDE_ratio_sum_equality_l1880_188029

theorem ratio_sum_equality (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h_abc : a^2 + b^2 + c^2 = 49)
  (h_xyz : x^2 + y^2 + z^2 = 64)
  (h_dot : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_equality_l1880_188029


namespace NUMINAMATH_CALUDE_sum_of_distances_constant_l1880_188035

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_pos : side > 0

/-- A point inside an equilateral triangle -/
structure InternalPoint (t : EquilateralTriangle) where
  x : ℝ
  y : ℝ
  inside : x > 0 ∧ y > 0 ∧ x + y < t.side

/-- The sum of perpendicular distances from an internal point to the three sides of an equilateral triangle -/
def sumOfDistances (t : EquilateralTriangle) (p : InternalPoint t) : ℝ :=
  p.x + p.y + (t.side - p.x - p.y)

/-- Theorem: The sum of perpendicular distances from any internal point to the three sides of an equilateral triangle is constant and equal to (√3/2) * side length -/
theorem sum_of_distances_constant (t : EquilateralTriangle) (p : InternalPoint t) :
  sumOfDistances t p = (Real.sqrt 3 / 2) * t.side := by
  sorry


end NUMINAMATH_CALUDE_sum_of_distances_constant_l1880_188035


namespace NUMINAMATH_CALUDE_min_questions_to_determine_product_l1880_188026

theorem min_questions_to_determine_product (n : ℕ) (h : n > 3) :
  let min_questions_any_three := Int.ceil (n / 3 : ℚ)
  let min_questions_consecutive_three := if n % 3 = 0 then n / 3 else n
  true := by
  sorry

#check min_questions_to_determine_product

end NUMINAMATH_CALUDE_min_questions_to_determine_product_l1880_188026


namespace NUMINAMATH_CALUDE_min_value_exponential_function_l1880_188028

theorem min_value_exponential_function :
  (∀ x : ℝ, Real.exp x + 4 * Real.exp (-x) ≥ 4) ∧
  (∃ x : ℝ, Real.exp x + 4 * Real.exp (-x) = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_exponential_function_l1880_188028


namespace NUMINAMATH_CALUDE_age_sum_is_75_l1880_188083

/-- Given the ages of Alice, Bob, and Carol satisfying certain conditions, prove that the sum of their current ages is 75 years. -/
theorem age_sum_is_75 (alice bob carol : ℕ) : 
  (alice - 10 = (bob - 10) / 2) →  -- 10 years ago, Alice was half of Bob's age
  (4 * alice = 3 * bob) →          -- The ratio of their present ages is 3:4
  (carol = alice + bob + 5) →      -- Carol is 5 years older than the sum of Alice and Bob's current ages
  alice + bob + carol = 75 :=
by sorry

end NUMINAMATH_CALUDE_age_sum_is_75_l1880_188083


namespace NUMINAMATH_CALUDE_speed_ratio_is_three_fourths_l1880_188070

/-- Represents the motion of objects A and B -/
structure Motion where
  vA : ℝ  -- Speed of A
  vB : ℝ  -- Speed of B

/-- The conditions of the problem -/
def satisfiesConditions (m : Motion) : Prop :=
  let distanceB := 800  -- Initial distance of B from O
  let t1 := 3           -- Time of first equidistance (in minutes)
  let t2 := 15          -- Time of second equidistance (in minutes)
  (t1 * m.vA = |distanceB - t1 * m.vB|) ∧   -- Equidistance at t1
  (t2 * m.vA = |distanceB - t2 * m.vB|)     -- Equidistance at t2

/-- The theorem to be proved -/
theorem speed_ratio_is_three_fourths :
  ∃ m : Motion, satisfiesConditions m ∧ m.vA / m.vB = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_is_three_fourths_l1880_188070


namespace NUMINAMATH_CALUDE_max_value_trigonometric_expression_l1880_188032

theorem max_value_trigonometric_expression (θ : Real) 
  (h : 0 < θ ∧ θ < π / 2) : 
  ∃ (max : Real), max = 4 * Real.sqrt 2 ∧ 
    ∀ φ, 0 < φ ∧ φ < π / 2 → 
      3 * Real.sin φ + 2 * Real.cos φ + 1 / Real.cos φ ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_trigonometric_expression_l1880_188032
