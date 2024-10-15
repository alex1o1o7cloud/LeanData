import Mathlib

namespace NUMINAMATH_CALUDE_urn_contents_l3261_326187

/-- Represents the contents of an urn with yellow, white, and red balls. -/
structure Urn :=
  (yellow : ℕ)
  (white : ℕ)
  (red : ℕ)

/-- Calculates the probability of drawing balls of given colors from the urn. -/
def probability (u : Urn) (colors : List ℕ) : ℚ :=
  (colors.sum : ℚ) / ((u.yellow + u.white + u.red) : ℚ)

/-- The main theorem about the urn contents. -/
theorem urn_contents : 
  ∀ (u : Urn), 
    u.yellow = 18 →
    probability u [u.white, u.red] = probability u [u.white, u.yellow] - 1/15 →
    probability u [u.red, u.yellow] = probability u [u.white, u.yellow] * 11/10 →
    u.white = 27 ∧ u.red = 16 := by
  sorry

end NUMINAMATH_CALUDE_urn_contents_l3261_326187


namespace NUMINAMATH_CALUDE_fraction_product_l3261_326199

theorem fraction_product : (2/3 : ℚ) * (5/11 : ℚ) * (3/8 : ℚ) = (5/44 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3261_326199


namespace NUMINAMATH_CALUDE_duck_race_charity_l3261_326134

/-- The amount of money raised in a charity duck race -/
def charity_amount (regular_price : ℝ) (large_price : ℝ) (regular_sold : ℕ) (large_sold : ℕ) : ℝ :=
  regular_price * (regular_sold : ℝ) + large_price * (large_sold : ℝ)

/-- Theorem stating the amount raised in the specific duck race -/
theorem duck_race_charity : 
  charity_amount 3 5 221 185 = 1588 := by
  sorry

end NUMINAMATH_CALUDE_duck_race_charity_l3261_326134


namespace NUMINAMATH_CALUDE_train_length_problem_l3261_326121

/-- Proves that under given conditions, the length of each train is 60 meters -/
theorem train_length_problem (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 48) (h2 : v_slow = 36) (h3 : t = 36) :
  let v_rel := v_fast - v_slow
  let d := v_rel * t * (5 / 18)
  d / 2 = 60 := by sorry

end NUMINAMATH_CALUDE_train_length_problem_l3261_326121


namespace NUMINAMATH_CALUDE_interior_lattice_points_collinear_l3261_326151

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle in the plane -/
structure Triangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Check if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if points are collinear -/
def areCollinear (points : List LatticePoint) : Prop := sorry

/-- The main theorem -/
theorem interior_lattice_points_collinear (T : Triangle) :
  (∀ p : LatticePoint, isOnBoundary p T → (p = T.A ∨ p = T.B ∨ p = T.C)) →
  (∃! (points : List LatticePoint), points.length = 4 ∧ 
    (∀ p ∈ points, isInside p T) ∧
    (∀ p : LatticePoint, isInside p T → p ∈ points)) →
  ∃ (points : List LatticePoint), points.length = 4 ∧
    (∀ p ∈ points, isInside p T) ∧ areCollinear points :=
by sorry


end NUMINAMATH_CALUDE_interior_lattice_points_collinear_l3261_326151


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_diff_l3261_326124

theorem crazy_silly_school_series_diff (total_books total_movies : ℕ) 
  (h1 : total_books = 20) 
  (h2 : total_movies = 12) : 
  total_books - total_movies = 8 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_diff_l3261_326124


namespace NUMINAMATH_CALUDE_cos_180_eq_neg_one_l3261_326129

-- Define the rotation function
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Define the cosine of 180 degrees
def cos_180 : ℝ := (rotate_180 (1, 0)).1

-- Theorem statement
theorem cos_180_eq_neg_one : cos_180 = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_eq_neg_one_l3261_326129


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3261_326141

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 + 2*a*x + a + 5| ≤ 3) ↔ (a = 4 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3261_326141


namespace NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l3261_326123

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l3261_326123


namespace NUMINAMATH_CALUDE_union_subset_implies_m_leq_three_l3261_326159

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 6}
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m}

theorem union_subset_implies_m_leq_three (m : ℝ) :
  B ∪ C m = B → m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_subset_implies_m_leq_three_l3261_326159


namespace NUMINAMATH_CALUDE_line_through_d_divides_equally_l3261_326174

-- Define the shape
structure Shape :=
  (area : ℝ)
  (is_unit_squares : Bool)

-- Define points
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line
structure Line :=
  (point1 : Point)
  (point2 : Point)

-- Define the problem setup
def problem_setup (s : Shape) (p a b c d e : Point) : Prop :=
  s.is_unit_squares ∧
  s.area = 9 ∧
  b.x = (a.x + c.x) / 2 ∧
  b.y = (a.y + c.y) / 2 ∧
  d.x = (c.x + e.x) / 2 ∧
  d.y = (c.y + e.y) / 2

-- Define the division of area by a line
def divides_area_equally (l : Line) (s : Shape) : Prop :=
  ∃ (area1 area2 : ℝ), 
    area1 = area2 ∧
    area1 + area2 = s.area

-- Theorem statement
theorem line_through_d_divides_equally 
  (s : Shape) (p a b c d e : Point) (l : Line) :
  problem_setup s p a b c d e →
  l.point1 = p →
  l.point2 = d →
  divides_area_equally l s :=
sorry

end NUMINAMATH_CALUDE_line_through_d_divides_equally_l3261_326174


namespace NUMINAMATH_CALUDE_train_length_calculation_l3261_326143

/-- The length of two trains passing each other on parallel tracks -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) 
  (passing_time : ℝ) (h1 : faster_speed = 46) (h2 : slower_speed = 36) 
  (h3 : passing_time = 54) : 
  let relative_speed := (faster_speed - slower_speed) * (1000 / 3600)
  let train_length := (relative_speed * passing_time) / 2
  train_length = 75 := by
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3261_326143


namespace NUMINAMATH_CALUDE_train_length_l3261_326107

/-- The length of a train given its speed and time to cross a pole. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 → time = 9 → speed * time * (5 / 18) = 90 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3261_326107


namespace NUMINAMATH_CALUDE_jensens_inequality_l3261_326189

/-- Jensen's inequality for convex functions -/
theorem jensens_inequality (f : ℝ → ℝ) (hf : ConvexOn ℝ Set.univ f) 
  (x₁ x₂ q₁ q₂ : ℝ) (hq₁ : q₁ > 0) (hq₂ : q₂ > 0) (hsum : q₁ + q₂ = 1) :
  f (q₁ * x₁ + q₂ * x₂) ≤ q₁ * f x₁ + q₂ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_jensens_inequality_l3261_326189


namespace NUMINAMATH_CALUDE_craigs_commission_problem_l3261_326177

/-- Craig's appliance sales commission problem -/
theorem craigs_commission_problem 
  (fixed_amount : ℝ) 
  (num_appliances : ℕ) 
  (total_selling_price : ℝ) 
  (total_commission : ℝ) 
  (h1 : num_appliances = 6)
  (h2 : total_selling_price = 3620)
  (h3 : total_commission = 662)
  (h4 : total_commission = num_appliances * fixed_amount + 0.1 * total_selling_price) :
  fixed_amount = 50 := by
sorry

end NUMINAMATH_CALUDE_craigs_commission_problem_l3261_326177


namespace NUMINAMATH_CALUDE_solution_correctness_l3261_326149

noncomputable def solution_set : Set ℂ :=
  {0, 15, (1 + Complex.I * Real.sqrt 7) / 2, (1 - Complex.I * Real.sqrt 7) / 2}

def original_equation (x : ℂ) : Prop :=
  (15 * x - x^2) / (x + 1) * (x + (15 - x) / (x + 1)) = 30

theorem solution_correctness :
  ∀ x : ℂ, x ∈ solution_set ↔ original_equation x :=
sorry

end NUMINAMATH_CALUDE_solution_correctness_l3261_326149


namespace NUMINAMATH_CALUDE_land_plot_side_length_l3261_326185

/-- For a square-shaped land plot with an area of 100 square units, 
    the length of one side is 10 units. -/
theorem land_plot_side_length (area : ℝ) (side : ℝ) : 
  area = 100 → side * side = area → side = 10 := by
  sorry

end NUMINAMATH_CALUDE_land_plot_side_length_l3261_326185


namespace NUMINAMATH_CALUDE_probability_not_all_same_dice_five_dice_not_all_same_l3261_326105

theorem probability_not_all_same_dice (n : ℕ) (s : ℕ) : 
  n > 0 → s > 1 → (1 - s / s^n : ℚ) = (s^n - s) / s^n := by sorry

-- The probability that five fair 6-sided dice don't all show the same number
theorem five_dice_not_all_same : 
  (1 - (6 : ℚ) / 6^5) = 1295 / 1296 := by
  have h : (1 - 6 / 6^5 : ℚ) = (6^5 - 6) / 6^5 := 
    probability_not_all_same_dice 5 6 (by norm_num) (by norm_num)
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_probability_not_all_same_dice_five_dice_not_all_same_l3261_326105


namespace NUMINAMATH_CALUDE_sector_area_l3261_326180

/-- Given a sector with central angle 60° and arc length π, its area is 3π/2 -/
theorem sector_area (angle : Real) (arc_length : Real) (area : Real) :
  angle = 60 * (π / 180) →
  arc_length = π →
  area = (angle / (2 * π)) * arc_length * arc_length / angle →
  area = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3261_326180


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3261_326128

theorem min_value_of_expression (a : ℝ) (ha : a > 0) :
  (a + 1)^2 / a ≥ 4 ∧ ((a + 1)^2 / a = 4 ↔ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3261_326128


namespace NUMINAMATH_CALUDE_smallest_sum_four_consecutive_primes_l3261_326168

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if four consecutive natural numbers are all prime -/
def fourConsecutivePrimes (n : ℕ) : Prop :=
  isPrime n ∧ isPrime (n + 1) ∧ isPrime (n + 2) ∧ isPrime (n + 3)

/-- The sum of four consecutive natural numbers starting from n -/
def sumFourConsecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (n + 3)

/-- The theorem stating that the smallest sum of four consecutive positive prime numbers
    that is divisible by 3 is 36 -/
theorem smallest_sum_four_consecutive_primes :
  ∃ n : ℕ, fourConsecutivePrimes n ∧ sumFourConsecutive n % 3 = 0 ∧
  sumFourConsecutive n = 36 ∧
  ∀ m : ℕ, m < n → ¬(fourConsecutivePrimes m ∧ sumFourConsecutive m % 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_four_consecutive_primes_l3261_326168


namespace NUMINAMATH_CALUDE_license_plate_count_l3261_326119

/-- The number of different license plate combinations with three unique letters 
    followed by a dash and three digits, where exactly one digit is repeated exactly once. -/
def license_plate_combinations : ℕ :=
  let letter_combinations := 26 * 25 * 24
  let digit_combinations := 10 * 3 * 9
  letter_combinations * digit_combinations

/-- Theorem stating that the number of license plate combinations is 4,212,000 -/
theorem license_plate_count :
  license_plate_combinations = 4212000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3261_326119


namespace NUMINAMATH_CALUDE_max_subsets_l3261_326109

/-- A set with 10 elements -/
def T : Finset (Fin 10) := Finset.univ

/-- The type of 5-element subsets of T -/
def Subset5 : Type := {S : Finset (Fin 10) // S.card = 5}

/-- The property that any two elements appear together in at most two subsets -/
def AtMostTwice (subsets : List Subset5) : Prop :=
  ∀ x y : Fin 10, x ≠ y → (subsets.filter (λ S => x ∈ S.1 ∧ y ∈ S.1)).length ≤ 2

/-- The main theorem -/
theorem max_subsets :
  (∃ subsets : List Subset5, AtMostTwice subsets ∧ subsets.length = 8) ∧
  (∀ subsets : List Subset5, AtMostTwice subsets → subsets.length ≤ 8) := by
  sorry

end NUMINAMATH_CALUDE_max_subsets_l3261_326109


namespace NUMINAMATH_CALUDE_functional_equation_problem_l3261_326173

theorem functional_equation_problem (f : ℝ → ℝ) 
  (h1 : ∀ a b : ℝ, f (a + b) = f a * f b) 
  (h2 : f 1 = 2) : 
  f 1^2 + f 2 / f 1 + f 2^2 + f 4 / f 3 + f 3^2 + f 6 / f 5 + f 4^2 + f 8 / f 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l3261_326173


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_six_l3261_326118

theorem sqrt_expression_equals_six :
  Real.sqrt ((16^10 / 16^9)^2 * 6^2) / 2^4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_six_l3261_326118


namespace NUMINAMATH_CALUDE_domino_arrangements_equal_combinations_l3261_326163

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a domino with length 2 and width 1 -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- The number of distinct domino arrangements on a grid -/
def num_arrangements (g : Grid) (d : Domino) (num_dominoes : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items -/
def choose (n : ℕ) (k : ℕ) : ℕ := sorry

theorem domino_arrangements_equal_combinations (g : Grid) (d : Domino) :
  g.width = 6 →
  g.height = 5 →
  d.length = 2 →
  d.width = 1 →
  num_arrangements g d 5 = choose 9 5 := by sorry

end NUMINAMATH_CALUDE_domino_arrangements_equal_combinations_l3261_326163


namespace NUMINAMATH_CALUDE_organization_size_l3261_326176

/-- The total number of employees in an organization -/
def total_employees : ℕ := sorry

/-- The number of employees earning below 10k $ -/
def below_10k : ℕ := 250

/-- The number of employees earning between 10k $ and 50k $ -/
def between_10k_50k : ℕ := 500

/-- The percentage of employees earning less than 50k $ -/
def percent_below_50k : ℚ := 75 / 100

theorem organization_size :
  (below_10k + between_10k_50k : ℚ) = percent_below_50k * total_employees ∧
  total_employees = 1000 := by sorry

end NUMINAMATH_CALUDE_organization_size_l3261_326176


namespace NUMINAMATH_CALUDE_volume_ratio_is_correct_l3261_326100

/-- The ratio of the volume of a cube with edge length 9 inches to the volume of a cube with edge length 2 feet -/
def volume_ratio : ℚ :=
  let inch_per_foot : ℚ := 12
  let edge1 : ℚ := 9  -- 9 inches
  let edge2 : ℚ := 2 * inch_per_foot  -- 2 feet in inches
  (edge1 / edge2) ^ 3

theorem volume_ratio_is_correct : volume_ratio = 27 / 512 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_is_correct_l3261_326100


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3261_326130

def A : Set ℝ := {2, 4, 6, 8}
def B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 6}

theorem intersection_of_A_and_B : A ∩ B = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3261_326130


namespace NUMINAMATH_CALUDE_school_boys_count_l3261_326127

theorem school_boys_count :
  ∀ (x : ℕ),
  (x + x = 100) →
  (x = 50) :=
by
  sorry

#check school_boys_count

end NUMINAMATH_CALUDE_school_boys_count_l3261_326127


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_palindromes_l3261_326104

-- Define a three-digit palindrome
def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 100 * a + 10 * b + a

-- Define the set of all three-digit palindromes
def three_digit_palindromes : Set ℕ :=
  {n : ℕ | is_three_digit_palindrome n}

-- Statement to prove
theorem greatest_common_factor_of_three_digit_palindromes :
  ∃ g : ℕ, g > 0 ∧ 
    (∀ n ∈ three_digit_palindromes, g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n ∈ three_digit_palindromes, d ∣ n) → d ≤ g) ∧
    g = 101 :=
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_palindromes_l3261_326104


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3261_326112

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → 
    ∃ (k₁ k₂ : ℝ), k₁ * k₂ ≠ 0 ∧ 
      (∀ (P : ℝ × ℝ), P.1^2 / a^2 + P.2^2 / b^2 = 1 → 
        (|k₁| + |k₂| ≥ |((P.2) / (P.1 - a))| + |((P.2) / (P.1 + a))|))) →
  (∃ (k₁ k₂ : ℝ), k₁ * k₂ ≠ 0 ∧ |k₁| + |k₂| = 1) →
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3261_326112


namespace NUMINAMATH_CALUDE_line_up_five_people_l3261_326114

theorem line_up_five_people (people : Finset Char) : 
  people.card = 5 → Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_line_up_five_people_l3261_326114


namespace NUMINAMATH_CALUDE_johann_oranges_l3261_326102

def oranges_problem (initial_oranges eaten_oranges stolen_fraction returned_oranges : ℕ) : Prop :=
  let remaining_after_eating := initial_oranges - eaten_oranges
  let stolen := (remaining_after_eating / 2 : ℕ)
  let final_oranges := remaining_after_eating - stolen + returned_oranges
  final_oranges = 30

theorem johann_oranges :
  oranges_problem 60 10 2 5 := by sorry

end NUMINAMATH_CALUDE_johann_oranges_l3261_326102


namespace NUMINAMATH_CALUDE_daily_houses_count_l3261_326156

/-- Represents Kyle's newspaper delivery route --/
structure NewspaperRoute where
  /-- Number of houses receiving daily paper Monday through Saturday --/
  daily_houses : ℕ
  /-- Total number of papers delivered in a week --/
  total_weekly_papers : ℕ
  /-- Number of regular customers not receiving Sunday paper --/
  sunday_skip : ℕ
  /-- Number of houses receiving paper only on Sunday --/
  sunday_only : ℕ
  /-- Ensures the total weekly papers match the given conditions --/
  papers_match : total_weekly_papers = 
    (6 * daily_houses) + (daily_houses - sunday_skip + sunday_only)

/-- Theorem stating the number of houses receiving daily paper --/
theorem daily_houses_count (route : NewspaperRoute) 
  (h1 : route.total_weekly_papers = 720)
  (h2 : route.sunday_skip = 10)
  (h3 : route.sunday_only = 30) : 
  route.daily_houses = 100 := by
  sorry

#check daily_houses_count

end NUMINAMATH_CALUDE_daily_houses_count_l3261_326156


namespace NUMINAMATH_CALUDE_circle_distance_characterization_l3261_326111

/-- Given two concentric circles C and S centered at P with radii r and s respectively,
    where s < r, and B is a point within S, this theorem characterizes the set of
    points A such that the distance from A to B is less than the distance from A
    to any point on circle C. -/
theorem circle_distance_characterization
  (P B : EuclideanSpace ℝ (Fin 2))  -- Points in 2D real Euclidean space
  (r s : ℝ)  -- Radii of circles C and S
  (h_s_lt_r : s < r)  -- Condition that s < r
  (h_B_in_S : ‖B - P‖ ≤ s)  -- B is within circle S
  (A : EuclideanSpace ℝ (Fin 2))  -- Arbitrary point A
  : (∀ (C : EuclideanSpace ℝ (Fin 2)), ‖C - P‖ = r → ‖A - B‖ < ‖A - C‖) ↔
    ‖A - B‖ < r - s :=
by sorry

end NUMINAMATH_CALUDE_circle_distance_characterization_l3261_326111


namespace NUMINAMATH_CALUDE_sector_angle_l3261_326197

theorem sector_angle (R : ℝ) (α : ℝ) : 
  R > 0 ∧ 2 * R + α * R = 6 ∧ (1/2) * R^2 * α = 2 → α = 1 ∨ α = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3261_326197


namespace NUMINAMATH_CALUDE_fewer_twos_equals_hundred_l3261_326101

theorem fewer_twos_equals_hundred : (222 / 2) - (22 / 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_fewer_twos_equals_hundred_l3261_326101


namespace NUMINAMATH_CALUDE_max_value_expression_l3261_326145

theorem max_value_expression (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  (∀ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) ≤ 3/2 * (c^2 + d^2)) ∧
  (∃ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) = 3/2 * (c^2 + d^2)) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3261_326145


namespace NUMINAMATH_CALUDE_P_when_a_is_3_range_of_a_for_Q_subset_P_l3261_326155

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x : ℝ | (x - a) * (x + 1) ≤ 0}
def Q : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

-- Theorem 1: When a = 3, P = {x | -1 ≤ x ≤ 3}
theorem P_when_a_is_3 : P 3 = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem 2: The range of positive a such that Q ⊆ P is [2, +∞)
theorem range_of_a_for_Q_subset_P : 
  {a : ℝ | a > 0 ∧ Q ⊆ P a} = {a : ℝ | a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_P_when_a_is_3_range_of_a_for_Q_subset_P_l3261_326155


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3261_326166

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 2) ↔ (∃ x : ℝ, x < 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3261_326166


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l3261_326116

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l3261_326116


namespace NUMINAMATH_CALUDE_definite_integral_x_cubed_l3261_326162

theorem definite_integral_x_cubed : ∫ (x : ℝ) in (-1)..(1), x^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_x_cubed_l3261_326162


namespace NUMINAMATH_CALUDE_additional_money_needed_l3261_326170

def lee_money : ℚ := 10
def friend_money : ℚ := 8
def chicken_wings_cost : ℚ := 6
def chicken_salad_cost : ℚ := 4
def cheeseburger_cost : ℚ := 3.5
def fries_cost : ℚ := 2
def soda_cost : ℚ := 1
def coupon_discount : ℚ := 0.15
def tax_rate : ℚ := 0.08

def total_order_cost : ℚ := chicken_wings_cost + chicken_salad_cost + 2 * cheeseburger_cost + fries_cost + 2 * soda_cost

def discounted_cost : ℚ := total_order_cost * (1 - coupon_discount)

def final_cost : ℚ := discounted_cost * (1 + tax_rate)

def total_money : ℚ := lee_money + friend_money

theorem additional_money_needed :
  final_cost - total_money = 1.28 := by sorry

end NUMINAMATH_CALUDE_additional_money_needed_l3261_326170


namespace NUMINAMATH_CALUDE_fraction_inequality_triangle_sine_inequality_l3261_326125

-- Part 1
theorem fraction_inequality (m n p : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hmn : m > n) :
  n / m < (n + p) / (m + p) := by sorry

-- Part 2
theorem triangle_sine_inequality (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (Real.sin C) / (Real.sin A + Real.sin B) + 
  (Real.sin A) / (Real.sin B + Real.sin C) + 
  (Real.sin B) / (Real.sin C + Real.sin A) < 2 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_triangle_sine_inequality_l3261_326125


namespace NUMINAMATH_CALUDE_no_solution_squared_equals_negative_one_l3261_326135

theorem no_solution_squared_equals_negative_one :
  ¬ ∃ x : ℝ, (3*x - 2)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_squared_equals_negative_one_l3261_326135


namespace NUMINAMATH_CALUDE_example_quadratic_equation_l3261_326171

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

/-- The equation x² + 2x - 1 = 0 is a quadratic equation -/
theorem example_quadratic_equation :
  is_quadratic_equation (λ x => x^2 + 2*x - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_example_quadratic_equation_l3261_326171


namespace NUMINAMATH_CALUDE_annie_bus_ride_l3261_326140

/-- The number of blocks Annie walked from her house to the bus stop -/
def blocks_to_bus_stop : ℕ := 5

/-- The total number of blocks Annie traveled -/
def total_blocks : ℕ := 24

/-- The number of blocks Annie rode the bus to the coffee shop -/
def blocks_by_bus : ℕ := (total_blocks - 2 * blocks_to_bus_stop) / 2

theorem annie_bus_ride : blocks_by_bus = 7 := by
  sorry

end NUMINAMATH_CALUDE_annie_bus_ride_l3261_326140


namespace NUMINAMATH_CALUDE_regular_hexagon_interior_angle_measure_l3261_326186

/-- The measure of an interior angle of a regular hexagon -/
def regular_hexagon_interior_angle : ℝ := 120

/-- A regular hexagon has 6 sides -/
def regular_hexagon_sides : ℕ := 6

/-- Theorem: The measure of each interior angle of a regular hexagon is 120 degrees -/
theorem regular_hexagon_interior_angle_measure :
  regular_hexagon_interior_angle = (((regular_hexagon_sides - 2) * 180) : ℝ) / regular_hexagon_sides :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_interior_angle_measure_l3261_326186


namespace NUMINAMATH_CALUDE_factorization_2y_squared_minus_8_l3261_326115

theorem factorization_2y_squared_minus_8 (y : ℝ) : 2 * y^2 - 8 = 2 * (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2y_squared_minus_8_l3261_326115


namespace NUMINAMATH_CALUDE_sqrt_less_implies_less_l3261_326157

theorem sqrt_less_implies_less (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a < Real.sqrt b → a < b :=
by sorry

end NUMINAMATH_CALUDE_sqrt_less_implies_less_l3261_326157


namespace NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l3261_326175

theorem least_positive_integer_for_multiple_of_five : ∃ (n : ℕ), n > 0 ∧ (365 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (365 + m) % 5 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l3261_326175


namespace NUMINAMATH_CALUDE_only_145_satisfies_condition_l3261_326110

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Check if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- Get the hundreds digit of a number -/
def hundredsDigit (n : ℕ) : ℕ :=
  n / 100

/-- Get the tens digit of a number -/
def tensDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Get the ones digit of a number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

/-- Check if a number is equal to the sum of the factorials of its digits -/
def isEqualToSumOfDigitFactorials (n : ℕ) : Prop :=
  n = factorial (hundredsDigit n) + factorial (tensDigit n) + factorial (onesDigit n)

theorem only_145_satisfies_condition :
  ∀ n : ℕ, isThreeDigit n ∧ isEqualToSumOfDigitFactorials n ↔ n = 145 := by
  sorry

#check only_145_satisfies_condition

end NUMINAMATH_CALUDE_only_145_satisfies_condition_l3261_326110


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3261_326103

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ y, y = f 1 → y = (1 : ℝ) + 2) : 
  f 1 + deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3261_326103


namespace NUMINAMATH_CALUDE_annie_purchase_problem_l3261_326142

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents20 : ℕ
  dollars4 : ℕ
  dollars5 : ℕ

/-- The problem statement -/
theorem annie_purchase_problem (counts : ItemCounts) : 
  counts.cents20 + counts.dollars4 + counts.dollars5 = 50 →
  20 * counts.cents20 + 400 * counts.dollars4 + 500 * counts.dollars5 = 5000 →
  counts.cents20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_annie_purchase_problem_l3261_326142


namespace NUMINAMATH_CALUDE_parallelogram_means_input_output_l3261_326195

/-- Represents the different symbols used in a program flowchart --/
inductive FlowchartSymbol
  | Parallelogram
  | Rectangle
  | Diamond
  | Oval

/-- Represents the different operations in a program flowchart --/
inductive FlowchartOperation
  | InputOutput
  | Process
  | Decision
  | Start_End

/-- Associates a FlowchartSymbol with its corresponding FlowchartOperation --/
def symbolMeaning : FlowchartSymbol → FlowchartOperation
  | FlowchartSymbol.Parallelogram => FlowchartOperation.InputOutput
  | FlowchartSymbol.Rectangle => FlowchartOperation.Process
  | FlowchartSymbol.Diamond => FlowchartOperation.Decision
  | FlowchartSymbol.Oval => FlowchartOperation.Start_End

theorem parallelogram_means_input_output :
  symbolMeaning FlowchartSymbol.Parallelogram = FlowchartOperation.InputOutput :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_means_input_output_l3261_326195


namespace NUMINAMATH_CALUDE_original_number_before_increase_l3261_326133

theorem original_number_before_increase (x : ℝ) : x * 1.5 = 525 → x = 350 := by
  sorry

end NUMINAMATH_CALUDE_original_number_before_increase_l3261_326133


namespace NUMINAMATH_CALUDE_debate_team_arrangements_l3261_326183

-- Define the number of students
def total_students : ℕ := 6

-- Define the number of team members
def team_size : ℕ := 4

-- Define the number of positions where student A can be placed
def positions_for_A : ℕ := 3

-- Define the number of remaining students after A is placed
def remaining_students : ℕ := total_students - 1

-- Define the number of remaining positions after A is placed
def remaining_positions : ℕ := team_size - 1

-- Theorem statement
theorem debate_team_arrangements :
  (positions_for_A * (remaining_students.factorial / (remaining_students - remaining_positions).factorial)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_arrangements_l3261_326183


namespace NUMINAMATH_CALUDE_F_less_than_G_l3261_326194

theorem F_less_than_G : ∀ x : ℝ, (2 * x^2 - 3 * x - 2) < (3 * x^2 - 7 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_F_less_than_G_l3261_326194


namespace NUMINAMATH_CALUDE_sqrt_720_equals_12_sqrt_5_l3261_326158

theorem sqrt_720_equals_12_sqrt_5 : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_equals_12_sqrt_5_l3261_326158


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3261_326191

universe u

def M : Set ℤ := {-1, 0, 1, 2, 3, 4}
def A : Set ℤ := {2, 3}
def AUnionB : Set ℤ := {1, 2, 3, 4}

theorem intersection_complement_theorem :
  ∃ B : Set ℤ, A ∪ B = AUnionB ∧ B ∩ (M \ A) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3261_326191


namespace NUMINAMATH_CALUDE_apothem_lateral_face_angle_l3261_326169

/-- Given a regular triangular pyramid where the lateral edge forms an angle of 60° with the base plane,
    the sine of the angle between the apothem and the plane of the adjacent lateral face
    is equal to (3√3) / 13. -/
theorem apothem_lateral_face_angle (a : ℝ) (h : a > 0) :
  let β : ℝ := 60 * π / 180  -- Convert 60° to radians
  let lateral_edge_angle : ℝ := β
  let apothem : ℝ := a * Real.sqrt 13 / (2 * Real.sqrt 3)
  let perpendicular_distance : ℝ := a * Real.sqrt 3 / 8
  let sin_φ : ℝ := perpendicular_distance / apothem
  sin_φ = 3 * Real.sqrt 3 / 13 := by
  sorry


end NUMINAMATH_CALUDE_apothem_lateral_face_angle_l3261_326169


namespace NUMINAMATH_CALUDE_remainder_of_nested_division_l3261_326131

theorem remainder_of_nested_division (P D K Q R R'q R'r : ℕ) :
  D > 0 →
  K > 0 →
  K < D →
  P = Q * D + R →
  R = R'q * K + R'r →
  R'r < K →
  P % (D * K) = R'r :=
sorry

end NUMINAMATH_CALUDE_remainder_of_nested_division_l3261_326131


namespace NUMINAMATH_CALUDE_tshirt_sale_revenue_per_minute_l3261_326164

/-- Calculates the money made per minute in a t-shirt sale. -/
theorem tshirt_sale_revenue_per_minute 
  (total_shirts : ℕ) 
  (sale_duration : ℕ) 
  (black_shirt_price : ℕ) 
  (white_shirt_price : ℕ) : 
  total_shirts = 200 →
  sale_duration = 25 →
  black_shirt_price = 30 →
  white_shirt_price = 25 →
  (total_shirts / 2 * black_shirt_price + total_shirts / 2 * white_shirt_price) / sale_duration = 220 :=
by
  sorry

#check tshirt_sale_revenue_per_minute

end NUMINAMATH_CALUDE_tshirt_sale_revenue_per_minute_l3261_326164


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l3261_326148

theorem largest_integer_less_than_100_remainder_5_mod_8 :
  ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l3261_326148


namespace NUMINAMATH_CALUDE_quadrilateral_interior_point_angles_l3261_326108

theorem quadrilateral_interior_point_angles 
  (a b c d x y z w : ℝ) 
  (h1 : a = x + y / 2)
  (h2 : b = y + z / 2)
  (h3 : c = z + w / 2)
  (h4 : d = w + x / 2)
  (h5 : x + y + z + w = 360) :
  x = (16 / 15) * (a - b / 2 + c / 4 - d / 8) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_interior_point_angles_l3261_326108


namespace NUMINAMATH_CALUDE_four_last_in_hundreds_of_fib_l3261_326120

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Function to get the hundreds digit of a natural number -/
def hundredsDigit (n : ℕ) : ℕ :=
  (n / 100) % 10

/-- Predicate to check if a digit has appeared in the hundreds position of any Fibonacci number up to the nth term -/
def digitAppearedInHundreds (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ hundredsDigit (fib k) = d

/-- The main theorem: 4 is the last digit to appear in the hundreds position of a Fibonacci number -/
theorem four_last_in_hundreds_of_fib :
  ∃ N, digitAppearedInHundreds 4 N ∧
    ∀ d, d ≠ 4 → ∃ n, n < N ∧ digitAppearedInHundreds d n :=
  sorry

end NUMINAMATH_CALUDE_four_last_in_hundreds_of_fib_l3261_326120


namespace NUMINAMATH_CALUDE_dish_initial_temp_l3261_326198

/-- The initial temperature of a dish given its heating rate and time to reach a final temperature --/
def initial_temperature (final_temp : ℝ) (heating_rate : ℝ) (heating_time : ℝ) : ℝ :=
  final_temp - heating_rate * heating_time

/-- Theorem stating that the initial temperature of the dish is 20 degrees --/
theorem dish_initial_temp : initial_temperature 100 5 16 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dish_initial_temp_l3261_326198


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisibility_l3261_326154

theorem prime_square_minus_one_divisibility (p : ℕ) :
  Prime p → p ≥ 7 →
  (∃ q : ℕ, Prime q ∧ q ≥ 7 ∧ 40 ∣ (q^2 - 1)) ∧
  (∃ r : ℕ, Prime r ∧ r ≥ 7 ∧ ¬(40 ∣ (r^2 - 1))) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisibility_l3261_326154


namespace NUMINAMATH_CALUDE_integer_pair_solution_l3261_326188

theorem integer_pair_solution (a b : ℤ) : 
  (a + b) / (a - b) = 3 ∧ (a + b) * (a - b) = 300 →
  (a = 20 ∧ b = 10) ∨ (a = -20 ∧ b = -10) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_solution_l3261_326188


namespace NUMINAMATH_CALUDE_initial_number_count_l3261_326179

theorem initial_number_count (n : ℕ) (S : ℝ) : 
  n > 0 ∧ 
  S / n = 12 ∧ 
  (S - 20) / (n - 1) = 10 →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_initial_number_count_l3261_326179


namespace NUMINAMATH_CALUDE_f_property_l3261_326132

/-- A function f(x) of the form ax^7 + bx - 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^7 + b * x - 2

/-- The theorem stating that if f(2009) = 10, then f(-2009) = -14 -/
theorem f_property (a b : ℝ) (h : f a b 2009 = 10) : f a b (-2009) = -14 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l3261_326132


namespace NUMINAMATH_CALUDE_exists_m_for_inequality_l3261_326196

def sequence_a : ℕ → ℚ
  | 7 => 16/3
  | n+1 => (3 * sequence_a n + 4) / (7 - sequence_a n)
  | _ => 0  -- Define for n < 7 to make the function total

theorem exists_m_for_inequality :
  ∃ m : ℕ, ∀ n ≥ m, sequence_a n > (sequence_a (n-1) + sequence_a (n+1)) / 2 :=
sorry

end NUMINAMATH_CALUDE_exists_m_for_inequality_l3261_326196


namespace NUMINAMATH_CALUDE_unique_nested_sqrt_integer_l3261_326113

theorem unique_nested_sqrt_integer : ∃! (n : ℕ+), ∃ (x : ℤ), x^2 = n + Real.sqrt (n + Real.sqrt (n + Real.sqrt n)) := by
  sorry

end NUMINAMATH_CALUDE_unique_nested_sqrt_integer_l3261_326113


namespace NUMINAMATH_CALUDE_closed_set_properties_l3261_326137

-- Define a closed set
def is_closed_set (M : Set Int) : Prop :=
  ∀ a b : Int, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Define the set M = {-4, -2, 0, 2, 4}
def M : Set Int := {-4, -2, 0, 2, 4}

-- Define the set of positive integers
def positive_integers : Set Int := {n : Int | n > 0}

-- Define a general closed set
def closed_set (A : Set Int) : Prop := is_closed_set A

theorem closed_set_properties :
  (¬ is_closed_set M) ∧
  (¬ is_closed_set positive_integers) ∧
  (∃ A₁ A₂ : Set Int, closed_set A₁ ∧ closed_set A₂ ∧ ¬ is_closed_set (A₁ ∪ A₂)) :=
sorry

end NUMINAMATH_CALUDE_closed_set_properties_l3261_326137


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_divisors_of_2520_l3261_326192

theorem sum_of_distinct_prime_divisors_of_2520 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.divisors 2520)) id) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_divisors_of_2520_l3261_326192


namespace NUMINAMATH_CALUDE_function_values_l3261_326150

def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem function_values (a b c : ℝ) :
  (∀ x, f x a b c ≤ f (-1) a b c) ∧
  (f (-1) a b c = 7) ∧
  (∀ x, f x a b c ≥ f 3 a b c) →
  a = -3 ∧ b = -9 ∧ c = 2 ∧ f 3 a b c = -25 :=
by sorry

end NUMINAMATH_CALUDE_function_values_l3261_326150


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3261_326184

/-- Represents a hyperbola with center (h, k), focus (h + c, k), and vertex (h - a, k) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  a : ℝ
  c : ℝ
  vertex_x : ℝ
  focus_x : ℝ
  h_pos : 0 < a
  h_c_gt_a : c > a
  h_vertex : vertex_x = h - a
  h_focus : focus_x = h + c

/-- The theorem stating the sum of h, k, a, and b for the given hyperbola -/
theorem hyperbola_sum (H : Hyperbola) (h_center : H.h = 1 ∧ H.k = -1)
    (h_vertex : H.vertex_x = -2) (h_focus : H.focus_x = 1 + Real.sqrt 41) :
    H.h + H.k + H.a + Real.sqrt (H.c^2 - H.a^2) = 3 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3261_326184


namespace NUMINAMATH_CALUDE_brady_current_yards_l3261_326181

/-- The passing yards record in a season -/
def record : ℕ := 5999

/-- The number of games left in the season -/
def games_left : ℕ := 6

/-- The average passing yards needed per game to beat the record -/
def average_needed : ℕ := 300

/-- Tom Brady's current passing yards -/
def current_yards : ℕ := 4200

theorem brady_current_yards : 
  current_yards = record + 1 - (games_left * average_needed) :=
sorry

end NUMINAMATH_CALUDE_brady_current_yards_l3261_326181


namespace NUMINAMATH_CALUDE_enough_beverages_l3261_326153

/-- Robin's hydration plan and beverage inventory --/
structure HydrationPlan where
  water_per_day : ℕ
  juice_per_day : ℕ
  soda_per_day : ℕ
  plan_duration : ℕ
  water_inventory : ℕ
  juice_inventory : ℕ
  soda_inventory : ℕ

/-- Theorem: Robin has enough beverages for her hydration plan --/
theorem enough_beverages (plan : HydrationPlan)
  (h1 : plan.water_per_day = 9)
  (h2 : plan.juice_per_day = 5)
  (h3 : plan.soda_per_day = 3)
  (h4 : plan.plan_duration = 60)
  (h5 : plan.water_inventory = 617)
  (h6 : plan.juice_inventory = 350)
  (h7 : plan.soda_inventory = 215) :
  plan.water_inventory ≥ plan.water_per_day * plan.plan_duration ∧
  plan.juice_inventory ≥ plan.juice_per_day * plan.plan_duration ∧
  plan.soda_inventory ≥ plan.soda_per_day * plan.plan_duration :=
by
  sorry

#check enough_beverages

end NUMINAMATH_CALUDE_enough_beverages_l3261_326153


namespace NUMINAMATH_CALUDE_total_deduction_in_cents_l3261_326193

/-- Elena's hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Local tax rate as a decimal -/
def tax_rate : ℝ := 0.02

/-- Health benefit rate as a decimal -/
def health_rate : ℝ := 0.015

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℝ := 100

/-- Theorem stating the total deduction in cents -/
theorem total_deduction_in_cents : 
  hourly_wage * dollars_to_cents * (tax_rate + health_rate) = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_total_deduction_in_cents_l3261_326193


namespace NUMINAMATH_CALUDE_total_shaded_area_l3261_326146

/-- The total shaded area of two squares with inscribed circles -/
theorem total_shaded_area (small_side large_side small_radius large_radius : ℝ)
  (h1 : small_side = 6)
  (h2 : large_side = 12)
  (h3 : small_radius = 3)
  (h4 : large_radius = 6) :
  (small_side ^ 2 - π * small_radius ^ 2) + (large_side ^ 2 - π * large_radius ^ 2) = 180 - 45 * π := by
  sorry

end NUMINAMATH_CALUDE_total_shaded_area_l3261_326146


namespace NUMINAMATH_CALUDE_exists_greatest_n_leq_2008_l3261_326144

/-- Checks if a number is a perfect square -/
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

/-- The sum of squares formula for natural numbers -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The formula for the sum of squares from n+1 to 3n -/
def sumOfSquaresNTo3N (n : ℕ) : ℕ := (26 * n^3 + 12 * n^2 + n) / 3

/-- The main theorem statement -/
theorem exists_greatest_n_leq_2008 :
  ∃ n : ℕ, n ≤ 2008 ∧ 
    isPerfectSquare (sumOfSquares n * sumOfSquaresNTo3N n) ∧
    ∀ m : ℕ, m > n → m ≤ 2008 → 
      ¬ isPerfectSquare (sumOfSquares m * sumOfSquaresNTo3N m) := by
  sorry

end NUMINAMATH_CALUDE_exists_greatest_n_leq_2008_l3261_326144


namespace NUMINAMATH_CALUDE_flu_infection_rate_l3261_326190

theorem flu_infection_rate : 
  ∀ (x : ℝ), 
  (1 : ℝ) + x + x * ((1 : ℝ) + x) = 144 → 
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_flu_infection_rate_l3261_326190


namespace NUMINAMATH_CALUDE_anna_final_mark_l3261_326139

/-- Calculates the final mark given term mark, exam mark, and their respective weights -/
def calculate_final_mark (term_mark : ℝ) (exam_mark : ℝ) (term_weight : ℝ) (exam_weight : ℝ) : ℝ :=
  term_mark * term_weight + exam_mark * exam_weight

/-- Anna's final mark calculation -/
theorem anna_final_mark :
  calculate_final_mark 80 90 0.7 0.3 = 83 := by
  sorry

#eval calculate_final_mark 80 90 0.7 0.3

end NUMINAMATH_CALUDE_anna_final_mark_l3261_326139


namespace NUMINAMATH_CALUDE_inequality_proof_l3261_326106

theorem inequality_proof (a : ℝ) (ha : a > 0) : 
  Real.sqrt (a^2 + 1/a^2) + 2 ≥ a + 1/a + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3261_326106


namespace NUMINAMATH_CALUDE_bake_sale_cookies_l3261_326126

/-- The number of chocolate chip cookies Jenny brought to the bake sale -/
def jenny_chocolate_chip : ℕ := 50

/-- The total number of peanut butter cookies at the bake sale -/
def total_peanut_butter : ℕ := 70

/-- The number of lemon cookies Marcus brought to the bake sale -/
def marcus_lemon : ℕ := 20

/-- The probability of picking a peanut butter cookie -/
def prob_peanut_butter : ℚ := 1/2

theorem bake_sale_cookies :
  jenny_chocolate_chip = 50 ∧
  total_peanut_butter = 70 ∧
  marcus_lemon = 20 ∧
  prob_peanut_butter = 1/2 →
  jenny_chocolate_chip + marcus_lemon = total_peanut_butter :=
by sorry

end NUMINAMATH_CALUDE_bake_sale_cookies_l3261_326126


namespace NUMINAMATH_CALUDE_product_of_ratios_l3261_326167

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2007 ∧ y₁^3 - 3*x₁^2*y₁ = 2006)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2007 ∧ y₂^3 - 3*x₂^2*y₂ = 2006)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2007 ∧ y₃^3 - 3*x₃^2*y₃ = 2006)
  (h₄ : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0) :
  (1 - x₁ / y₁) * (1 - x₂ / y₂) * (1 - x₃ / y₃) = 1 / 1003 := by
sorry

end NUMINAMATH_CALUDE_product_of_ratios_l3261_326167


namespace NUMINAMATH_CALUDE_ratio_fraction_value_l3261_326138

theorem ratio_fraction_value (a b : ℝ) (h : a / b = 4) :
  (a - 3 * b) / (2 * a - b) = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_fraction_value_l3261_326138


namespace NUMINAMATH_CALUDE_angle_E_measure_l3261_326160

-- Define the hexagon and its angles
def Hexagon (A B C D E F : ℝ) : Prop :=
  -- Convexity is implied by the sum of angles being 720°
  A + B + C + D + E + F = 720 ∧
  -- Angles A, C, and D are congruent
  A = C ∧ A = D ∧
  -- Angle B is 20 degrees more than angle A
  B = A + 20 ∧
  -- Angles E and F are congruent
  E = F ∧
  -- Angle A is 30 degrees less than angle E
  A + 30 = E

-- Theorem statement
theorem angle_E_measure (A B C D E F : ℝ) :
  Hexagon A B C D E F → E = 158 := by
  sorry

end NUMINAMATH_CALUDE_angle_E_measure_l3261_326160


namespace NUMINAMATH_CALUDE_min_value_expression_l3261_326122

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  ∃ (m : ℝ), m = 2/675 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 1/a + 1/b + 1/c = 9 → 2 * a^3 * b^2 * c ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3261_326122


namespace NUMINAMATH_CALUDE_benny_bought_two_cards_l3261_326178

/-- The number of Pokemon cards Benny bought -/
def cards_bought (initial_cards final_cards : ℕ) : ℕ :=
  initial_cards - final_cards

/-- Proof that Benny bought 2 Pokemon cards -/
theorem benny_bought_two_cards :
  let initial_cards := 3
  let final_cards := 1
  cards_bought initial_cards final_cards = 2 := by
sorry

end NUMINAMATH_CALUDE_benny_bought_two_cards_l3261_326178


namespace NUMINAMATH_CALUDE_max_wooden_pencils_l3261_326182

theorem max_wooden_pencils :
  ∀ (m w : ℕ),
  m + w = 72 →
  ∃ (p : ℕ), Nat.Prime p ∧ m = w + p →
  w ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_max_wooden_pencils_l3261_326182


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l3261_326117

-- Define the function f(x) = x^3 - 3x - 3
def f (x : ℝ) : ℝ := x^3 - 3*x - 3

-- Theorem: f(x) has a root in the interval (2, 3)
theorem f_has_root_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_f_has_root_in_interval_l3261_326117


namespace NUMINAMATH_CALUDE_rearrangement_time_theorem_l3261_326165

/-- The number of letters in the name -/
def name_length : ℕ := 9

/-- The number of times the repeated letter appears -/
def repeated_letter_count : ℕ := 2

/-- The number of rearrangements that can be written per minute -/
def rearrangements_per_minute : ℕ := 15

/-- Calculate the number of unique rearrangements -/
def unique_rearrangements : ℕ := name_length.factorial / repeated_letter_count.factorial

/-- Calculate the total time in hours to write all rearrangements -/
def total_time_hours : ℚ :=
  (unique_rearrangements / rearrangements_per_minute : ℚ) / 60

/-- Theorem stating the time required to write all rearrangements -/
theorem rearrangement_time_theorem :
  total_time_hours = 201.6 := by sorry

end NUMINAMATH_CALUDE_rearrangement_time_theorem_l3261_326165


namespace NUMINAMATH_CALUDE_green_bay_high_relay_race_length_l3261_326136

/-- Calculates the total length of a relay race given the number of team members and the distance each member runs. -/
def relay_race_length (team_members : ℕ) (distance_per_member : ℕ) : ℕ :=
  team_members * distance_per_member

/-- Theorem stating that a relay race with 5 team members, each running 30 meters, has a total length of 150 meters. -/
theorem green_bay_high_relay_race_length :
  relay_race_length 5 30 = 150 := by
  sorry

#eval relay_race_length 5 30

end NUMINAMATH_CALUDE_green_bay_high_relay_race_length_l3261_326136


namespace NUMINAMATH_CALUDE_shiela_neighbors_l3261_326172

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) 
  (h1 : total_drawings = 54)
  (h2 : drawings_per_neighbor = 9)
  : total_drawings / drawings_per_neighbor = 6 := by
  sorry

end NUMINAMATH_CALUDE_shiela_neighbors_l3261_326172


namespace NUMINAMATH_CALUDE_rectangle_diagonal_perimeter_ratio_l3261_326161

theorem rectangle_diagonal_perimeter_ratio :
  ∀ (long_side : ℝ),
  long_side > 0 →
  let short_side := (1/3) * long_side
  let diagonal := Real.sqrt (short_side^2 + long_side^2)
  let perimeter := 2 * (short_side + long_side)
  let saved_distance := (1/3) * long_side
  diagonal + saved_distance = long_side →
  diagonal / perimeter = Real.sqrt 10 / 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_perimeter_ratio_l3261_326161


namespace NUMINAMATH_CALUDE_xyz_equality_l3261_326147

theorem xyz_equality (x y z : ℕ+) (a b c d : ℝ) 
  (h1 : x ≤ y) (h2 : y ≤ z)
  (h3 : (x : ℝ) ^ a = (y : ℝ) ^ b)
  (h4 : (y : ℝ) ^ b = (z : ℝ) ^ c)
  (h5 : (z : ℝ) ^ c = 70 ^ d)
  (h6 : 1 / a + 1 / b + 1 / c = 1 / d) :
  x + y = z := by sorry

end NUMINAMATH_CALUDE_xyz_equality_l3261_326147


namespace NUMINAMATH_CALUDE_solution_correctness_l3261_326152

theorem solution_correctness : 
  (∃ x y : ℚ, 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℚ, 2 * x + y = 2 ∧ 8 * x + 3 * y = 9 ∧ x = 3/2 ∧ y = -1) := by
  sorry

#check solution_correctness

end NUMINAMATH_CALUDE_solution_correctness_l3261_326152
