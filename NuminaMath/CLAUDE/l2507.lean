import Mathlib

namespace NUMINAMATH_CALUDE_three_lines_six_points_intersection_l2507_250710

-- Define the type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the type for lines in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines intersect
def linesIntersect (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

-- Theorem statement
theorem three_lines_six_points_intersection :
  ∃ (l1 l2 l3 : Line2D),
    let A : Point2D := ⟨1, 1⟩
    let B : Point2D := ⟨2, 2⟩
    let C : Point2D := ⟨3, 3⟩
    let D : Point2D := ⟨1, 3⟩
    let E : Point2D := ⟨2, 4⟩
    let F : Point2D := ⟨3, 5⟩
    -- Each point lies on at least one line
    (pointOnLine A l1 ∨ pointOnLine A l2 ∨ pointOnLine A l3) ∧
    (pointOnLine B l1 ∨ pointOnLine B l2 ∨ pointOnLine B l3) ∧
    (pointOnLine C l1 ∨ pointOnLine C l2 ∨ pointOnLine C l3) ∧
    (pointOnLine D l1 ∨ pointOnLine D l2 ∨ pointOnLine D l3) ∧
    (pointOnLine E l1 ∨ pointOnLine E l2 ∨ pointOnLine E l3) ∧
    (pointOnLine F l1 ∨ pointOnLine F l2 ∨ pointOnLine F l3) ∧
    -- Each line contains at least two points
    ((pointOnLine A l1 ∧ pointOnLine B l1) ∨
     (pointOnLine A l1 ∧ pointOnLine C l1) ∨
     (pointOnLine A l1 ∧ pointOnLine D l1) ∨
     (pointOnLine A l1 ∧ pointOnLine E l1) ∨
     (pointOnLine A l1 ∧ pointOnLine F l1) ∨
     (pointOnLine B l1 ∧ pointOnLine C l1) ∨
     (pointOnLine B l1 ∧ pointOnLine D l1) ∨
     (pointOnLine B l1 ∧ pointOnLine E l1) ∨
     (pointOnLine B l1 ∧ pointOnLine F l1) ∨
     (pointOnLine C l1 ∧ pointOnLine D l1) ∨
     (pointOnLine C l1 ∧ pointOnLine E l1) ∨
     (pointOnLine C l1 ∧ pointOnLine F l1) ∨
     (pointOnLine D l1 ∧ pointOnLine E l1) ∨
     (pointOnLine D l1 ∧ pointOnLine F l1) ∨
     (pointOnLine E l1 ∧ pointOnLine F l1)) ∧
    ((pointOnLine A l2 ∧ pointOnLine B l2) ∨
     (pointOnLine A l2 ∧ pointOnLine C l2) ∨
     (pointOnLine A l2 ∧ pointOnLine D l2) ∨
     (pointOnLine A l2 ∧ pointOnLine E l2) ∨
     (pointOnLine A l2 ∧ pointOnLine F l2) ∨
     (pointOnLine B l2 ∧ pointOnLine C l2) ∨
     (pointOnLine B l2 ∧ pointOnLine D l2) ∨
     (pointOnLine B l2 ∧ pointOnLine E l2) ∨
     (pointOnLine B l2 ∧ pointOnLine F l2) ∨
     (pointOnLine C l2 ∧ pointOnLine D l2) ∨
     (pointOnLine C l2 ∧ pointOnLine E l2) ∨
     (pointOnLine C l2 ∧ pointOnLine F l2) ∨
     (pointOnLine D l2 ∧ pointOnLine E l2) ∨
     (pointOnLine D l2 ∧ pointOnLine F l2) ∨
     (pointOnLine E l2 ∧ pointOnLine F l2)) ∧
    ((pointOnLine A l3 ∧ pointOnLine B l3) ∨
     (pointOnLine A l3 ∧ pointOnLine C l3) ∨
     (pointOnLine A l3 ∧ pointOnLine D l3) ∨
     (pointOnLine A l3 ∧ pointOnLine E l3) ∨
     (pointOnLine A l3 ∧ pointOnLine F l3) ∨
     (pointOnLine B l3 ∧ pointOnLine C l3) ∨
     (pointOnLine B l3 ∧ pointOnLine D l3) ∨
     (pointOnLine B l3 ∧ pointOnLine E l3) ∨
     (pointOnLine B l3 ∧ pointOnLine F l3) ∨
     (pointOnLine C l3 ∧ pointOnLine D l3) ∨
     (pointOnLine C l3 ∧ pointOnLine E l3) ∨
     (pointOnLine C l3 ∧ pointOnLine F l3) ∨
     (pointOnLine D l3 ∧ pointOnLine E l3) ∨
     (pointOnLine D l3 ∧ pointOnLine F l3) ∨
     (pointOnLine E l3 ∧ pointOnLine F l3)) ∧
    -- All three lines intersect at one point
    linesIntersect l1 l2 ∧ linesIntersect l2 l3 ∧ linesIntersect l1 l3 := by
  sorry


end NUMINAMATH_CALUDE_three_lines_six_points_intersection_l2507_250710


namespace NUMINAMATH_CALUDE_no_squares_in_range_l2507_250723

theorem no_squares_in_range : ¬ ∃ (x y a b : ℕ),
  988 ≤ x ∧ x < y ∧ y ≤ 1991 ∧
  x * y + x = a^2 ∧
  x * y + y = b^2 :=
sorry

end NUMINAMATH_CALUDE_no_squares_in_range_l2507_250723


namespace NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l2507_250725

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_minus_cylinder_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ)
  (h1 : cube_side = 6)
  (h2 : cylinder_radius = 3)
  (h3 : cylinder_height = cube_side) :
  cube_side^3 - π * cylinder_radius^2 * cylinder_height = 216 - 54*π :=
by sorry

end NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l2507_250725


namespace NUMINAMATH_CALUDE_archer_expected_hits_l2507_250748

/-- The expected value of a binomial distribution with n trials and probability p -/
def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p

/-- The number of shots taken by the archer -/
def num_shots : ℕ := 10

/-- The probability of hitting the bullseye -/
def hit_probability : ℝ := 0.9

/-- Theorem: The expected number of bullseye hits for the archer -/
theorem archer_expected_hits : 
  binomial_expectation num_shots hit_probability = 9 := by
  sorry

end NUMINAMATH_CALUDE_archer_expected_hits_l2507_250748


namespace NUMINAMATH_CALUDE_same_heads_probability_l2507_250769

/-- Represents the outcome of a coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Represents the result of tossing two coins -/
def TwoCoins := (CoinToss × CoinToss)

/-- The sample space of all possible outcomes when two people each toss two coins -/
def SampleSpace := (TwoCoins × TwoCoins)

/-- Counts the number of heads in a two-coin toss -/
def countHeads : TwoCoins → Nat
| (CoinToss.Heads, CoinToss.Heads) => 2
| (CoinToss.Heads, CoinToss.Tails) => 1
| (CoinToss.Tails, CoinToss.Heads) => 1
| (CoinToss.Tails, CoinToss.Tails) => 0

/-- Checks if two two-coin tosses have the same number of heads -/
def sameHeads (t1 t2 : TwoCoins) : Bool :=
  countHeads t1 = countHeads t2

/-- The number of elements in the sample space -/
def totalOutcomes : Nat := 16

/-- The number of favorable outcomes (same number of heads) -/
def favorableOutcomes : Nat := 6

/-- The probability of getting the same number of heads -/
def probability : Rat := favorableOutcomes / totalOutcomes

theorem same_heads_probability : probability = 3 / 8 := by
  sorry


end NUMINAMATH_CALUDE_same_heads_probability_l2507_250769


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2507_250783

theorem min_reciprocal_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 3) :
  (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x + y + z = 3 → 1/x + 1/y + 1/z ≥ 1/a + 1/b + 1/c) →
  1/a + 1/b + 1/c = 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2507_250783


namespace NUMINAMATH_CALUDE_no_negative_roots_l2507_250770

theorem no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l2507_250770


namespace NUMINAMATH_CALUDE_keatons_annual_profit_l2507_250719

/-- Represents a fruit type with its harvest frequency (in months), selling price, and harvest cost. -/
structure Fruit where
  harvestFrequency : ℕ
  sellingPrice : ℕ
  harvestCost : ℕ

/-- Calculates the annual profit for a given fruit. -/
def annualProfit (fruit : Fruit) : ℕ :=
  ((12 / fruit.harvestFrequency) * (fruit.sellingPrice - fruit.harvestCost))

/-- Keaton's farm data -/
def orange : Fruit := ⟨2, 50, 20⟩
def apple : Fruit := ⟨3, 30, 15⟩
def peach : Fruit := ⟨4, 45, 25⟩
def blackberry : Fruit := ⟨6, 70, 30⟩

/-- Theorem stating Keaton's total annual profit -/
theorem keatons_annual_profit :
  annualProfit orange + annualProfit apple + annualProfit peach + annualProfit blackberry = 380 := by
  sorry


end NUMINAMATH_CALUDE_keatons_annual_profit_l2507_250719


namespace NUMINAMATH_CALUDE_geometric_series_problem_l2507_250768

theorem geometric_series_problem (a r : ℝ) (h1 : r ≠ 1) (h2 : r > 0) : 
  (a / (1 - r) = 15) → (a / (1 - r^4) = 9) → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l2507_250768


namespace NUMINAMATH_CALUDE_floor_range_l2507_250734

theorem floor_range (x : ℝ) : 
  Int.floor x = -3 → -3 ≤ x ∧ x < -2 := by sorry

end NUMINAMATH_CALUDE_floor_range_l2507_250734


namespace NUMINAMATH_CALUDE_smallest_n_value_l2507_250721

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2010 →
  is_even c →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  n ≥ 501 ∧ ∃ (a' b' c' m' : ℕ), 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' + b' + c' = 2010 ∧
    is_even c' ∧
    a'.factorial * b'.factorial * c'.factorial = m' * (10 ^ 501) ∧
    ¬(10 ∣ m') :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2507_250721


namespace NUMINAMATH_CALUDE_glass_volume_proof_l2507_250726

theorem glass_volume_proof (V : ℝ) 
  (h1 : 0.4 * V = V - (0.6 * V))  -- pessimist's glass is 60% empty
  (h2 : 0.6 * V - 0.4 * V = 46)   -- difference in water volume
  : V = 230 := by
sorry

end NUMINAMATH_CALUDE_glass_volume_proof_l2507_250726


namespace NUMINAMATH_CALUDE_factorization_of_4x_squared_plus_x_l2507_250779

theorem factorization_of_4x_squared_plus_x (x : ℝ) : 4 * x^2 + x = x * (4 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_squared_plus_x_l2507_250779


namespace NUMINAMATH_CALUDE_tensor_self_zero_tensor_dot_product_identity_l2507_250747

/-- Definition of the ⊗ operation for 2D vectors -/
def tensor (a b : ℝ × ℝ) : ℝ := a.1 * a.2 - b.1 * b.2

/-- The dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem tensor_self_zero (a : ℝ × ℝ) : tensor a a = 0 := by sorry

theorem tensor_dot_product_identity (a b : ℝ × ℝ) :
  (tensor a b)^2 + (dot_product a b)^2 = (a.1^2 + b.2^2) * (a.2^2 + b.1^2) := by sorry

end NUMINAMATH_CALUDE_tensor_self_zero_tensor_dot_product_identity_l2507_250747


namespace NUMINAMATH_CALUDE_sqrt_x_minus_four_defined_l2507_250741

theorem sqrt_x_minus_four_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 4) ↔ x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_four_defined_l2507_250741


namespace NUMINAMATH_CALUDE_connected_paper_area_l2507_250739

/-- The area of connected square papers -/
theorem connected_paper_area 
  (num_papers : ℕ) 
  (side_length : ℝ) 
  (overlap : ℝ) 
  (h_num : num_papers = 6)
  (h_side : side_length = 30)
  (h_overlap : overlap = 7) : 
  (side_length + (num_papers - 1) * (side_length - overlap)) * side_length = 4350 :=
sorry

end NUMINAMATH_CALUDE_connected_paper_area_l2507_250739


namespace NUMINAMATH_CALUDE_b_age_l2507_250796

def problem (a b c d : ℕ) : Prop :=
  (a = b + 2) ∧ 
  (b = 2 * c) ∧ 
  (d = b - 3) ∧ 
  (a + b + c + d = 60)

theorem b_age (a b c d : ℕ) (h : problem a b c d) : b = 17 := by
  sorry

end NUMINAMATH_CALUDE_b_age_l2507_250796


namespace NUMINAMATH_CALUDE_triangle_piece_count_l2507_250759

/-- Calculate the sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculate the sum of the first n natural numbers -/
def triangle_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The number of rows in the rod triangle -/
def rod_rows : ℕ := 10

/-- The number of rows in the connector triangle -/
def connector_rows : ℕ := rod_rows + 1

/-- The first term of the rod arithmetic sequence -/
def first_rod_count : ℕ := 3

/-- The common difference of the rod arithmetic sequence -/
def rod_increment : ℕ := 3

theorem triangle_piece_count : 
  arithmetic_sum first_rod_count rod_increment rod_rows + 
  triangle_number connector_rows = 231 := by
  sorry

end NUMINAMATH_CALUDE_triangle_piece_count_l2507_250759


namespace NUMINAMATH_CALUDE_kind_wizard_succeeds_for_odd_n_l2507_250740

/-- Represents a friendship between two dwarves -/
structure Friendship :=
  (dwarf1 : ℕ)
  (dwarf2 : ℕ)

/-- Creates a list of friendships based on the wizard's pairing strategy -/
def createFriendships (n : ℕ) : List Friendship := sorry

/-- Breaks n friendships from the list -/
def breakFriendships (friendships : List Friendship) (n : ℕ) : List Friendship := sorry

/-- Checks if the remaining friendships can form a valid circular arrangement -/
def canFormCircularArrangement (friendships : List Friendship) : Prop := sorry

theorem kind_wizard_succeeds_for_odd_n (n : ℕ) (h : Odd n) :
  ∀ (broken : List Friendship),
    broken.length = n →
    canFormCircularArrangement (breakFriendships (createFriendships n) n) :=
sorry

end NUMINAMATH_CALUDE_kind_wizard_succeeds_for_odd_n_l2507_250740


namespace NUMINAMATH_CALUDE_hallies_reading_l2507_250712

/-- Proves that given the conditions of Hallie's reading pattern, she read 63 pages on the first day -/
theorem hallies_reading (total_pages : ℕ) (day1 : ℕ) : 
  total_pages = 354 → 
  day1 + 2 * day1 + (2 * day1 + 10) + 29 = total_pages → 
  day1 = 63 := by
  sorry

#check hallies_reading

end NUMINAMATH_CALUDE_hallies_reading_l2507_250712


namespace NUMINAMATH_CALUDE_unique_cube_pair_l2507_250749

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def has_unique_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

theorem unique_cube_pair :
  ∃! (a b : ℕ),
    1000 ≤ a ∧ a < 10000 ∧
    100 ≤ b ∧ b < 1000 ∧
    is_perfect_cube a ∧
    is_perfect_cube b ∧
    a / 100 = b / 10 ∧
    has_unique_digits a ∧
    has_unique_digits b ∧
    a = 1728 ∧
    b = 125 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_pair_l2507_250749


namespace NUMINAMATH_CALUDE_f_at_5_l2507_250703

/-- A function satisfying the given functional equation -/
def f : ℝ → ℝ :=
  sorry

/-- The functional equation that f satisfies for all x -/
axiom f_eq (x : ℝ) : 3 * f x + f (2 - x) = 4 * x^2 + 1

/-- The theorem to be proved -/
theorem f_at_5 : f 5 = 133 / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_at_5_l2507_250703


namespace NUMINAMATH_CALUDE_ShortestDistance_l2507_250777

/-- Line1 represents the first line (1, 2, 3) + u(1, 1, 2) -/
def Line1 (u : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => u + 1
  | 1 => u + 2
  | 2 => 2*u + 3

/-- Line2 represents the second line (2, 4, 0) + v(2, -1, 1) -/
def Line2 (v : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 2*v + 2
  | 1 => -v + 4
  | 2 => v

/-- DistanceSquared calculates the squared distance between two points on the lines -/
def DistanceSquared (u v : ℝ) : ℝ :=
  (Line1 u 0 - Line2 v 0)^2 + (Line1 u 1 - Line2 v 1)^2 + (Line1 u 2 - Line2 v 2)^2

/-- ShortestDistance states that the minimum value of the square root of DistanceSquared is √5 -/
theorem ShortestDistance : 
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧ 
  ∀ (u v : ℝ), Real.sqrt (DistanceSquared u v) ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_ShortestDistance_l2507_250777


namespace NUMINAMATH_CALUDE_vector_equation_holds_l2507_250713

def vector2D := ℝ × ℝ

def dot_product (v w : vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def scale_vector (s : ℝ) (v : vector2D) : vector2D :=
  (s * v.1, s * v.2)

theorem vector_equation_holds (a c : vector2D) (b : vector2D) : 
  a = (1, 1) → c = (2, 2) → 
  scale_vector (dot_product a b) c = scale_vector (dot_product b c) a := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_holds_l2507_250713


namespace NUMINAMATH_CALUDE_equation_solutions_l2507_250767

theorem equation_solutions : 
  ∃! (s : Set ℝ), s = {x : ℝ | (50 - 3*x)^(1/4) + (30 + 3*x)^(1/4) = 4} ∧ s = {16, -14} :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2507_250767


namespace NUMINAMATH_CALUDE_pentagon_y_coordinate_l2507_250709

/-- A pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The area of a pentagon -/
def area (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a vertical line of symmetry -/
def has_vertical_symmetry (p : Pentagon) : Prop := sorry

/-- The y-coordinate of a point -/
def y_coord (point : ℝ × ℝ) : ℝ := point.2

theorem pentagon_y_coordinate :
  ∀ (p : Pentagon),
    p.A = (0, 0) →
    p.B = (0, 6) →
    p.D = (6, 6) →
    p.E = (6, 0) →
    has_vertical_symmetry p →
    area p = 72 →
    y_coord p.C = 18 := by sorry

end NUMINAMATH_CALUDE_pentagon_y_coordinate_l2507_250709


namespace NUMINAMATH_CALUDE_rotate_D_90_clockwise_l2507_250795

-- Define the rotation matrix for 90° clockwise rotation
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

-- Define the original point D
def D : ℝ × ℝ := (-2, 3)

-- Theorem to prove
theorem rotate_D_90_clockwise :
  rotate90Clockwise D = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_D_90_clockwise_l2507_250795


namespace NUMINAMATH_CALUDE_proportion_solution_l2507_250798

theorem proportion_solution (x y : ℝ) : 
  (0.75 : ℝ) / x = 5 / y ∧ x = 1.65 → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2507_250798


namespace NUMINAMATH_CALUDE_range_of_a_l2507_250701

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  2 * x + y ≥ 4 ∧ x - y ≥ 1 ∧ x - 2 * y ≤ 2

-- Define the function z
def z (a x y : ℝ) : ℝ := a * x + y

-- Define the minimum point
def min_point : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ,
  (∀ x y : ℝ, feasible_region x y → z a x y ≥ z a (min_point.1) (min_point.2)) →
  (∃ x y : ℝ, feasible_region x y ∧ z a x y = z a (min_point.1) (min_point.2) → (x, y) = min_point) →
  -1/2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2507_250701


namespace NUMINAMATH_CALUDE_converse_square_sum_nonzero_l2507_250756

theorem converse_square_sum_nonzero (x y : ℝ) : 
  (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_converse_square_sum_nonzero_l2507_250756


namespace NUMINAMATH_CALUDE_percentage_increase_l2507_250746

theorem percentage_increase (x : ℝ) : 
  x > 98 ∧ x = 117.6 → (x - 98) / 98 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2507_250746


namespace NUMINAMATH_CALUDE_horse_price_theorem_l2507_250780

/-- The sum of a geometric series with 32 terms, where the first term is 1
    and each subsequent term is twice the previous term, is 4294967295. -/
theorem horse_price_theorem :
  let n : ℕ := 32
  let a : ℕ := 1
  let r : ℕ := 2
  (a * (r^n - 1)) / (r - 1) = 4294967295 :=
by sorry

end NUMINAMATH_CALUDE_horse_price_theorem_l2507_250780


namespace NUMINAMATH_CALUDE_bo_words_per_day_l2507_250772

def words_per_day (total_flashcards : ℕ) (known_percentage : ℚ) (days_to_learn : ℕ) : ℚ :=
  (total_flashcards : ℚ) * (1 - known_percentage) / days_to_learn

theorem bo_words_per_day :
  words_per_day 800 (1/5) 40 = 16 := by sorry

end NUMINAMATH_CALUDE_bo_words_per_day_l2507_250772


namespace NUMINAMATH_CALUDE_tower_surface_area_l2507_250745

/-- Calculates the visible surface area of a cube in the tower -/
def visibleSurfaceArea (sideLength : ℕ) (isTop : Bool) : ℕ :=
  if isTop then 5 * sideLength^2 else 4 * sideLength^2

/-- Represents the tower of cubes -/
def cubesTower : List ℕ := [9, 1, 7, 3, 5, 4, 6, 8]

/-- Calculates the total visible surface area of the tower -/
def totalVisibleSurfaceArea (tower : List ℕ) : ℕ :=
  let n := tower.length
  tower.enum.foldl (fun acc (i, sideLength) =>
    acc + visibleSurfaceArea sideLength (i == n - 1)) 0

theorem tower_surface_area :
  totalVisibleSurfaceArea cubesTower = 1408 := by
  sorry

#eval totalVisibleSurfaceArea cubesTower

end NUMINAMATH_CALUDE_tower_surface_area_l2507_250745


namespace NUMINAMATH_CALUDE_imaginary_part_proof_l2507_250761

def i : ℂ := Complex.I

def z : ℂ := 1 - i

theorem imaginary_part_proof : Complex.im ((2 / z) + i ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_proof_l2507_250761


namespace NUMINAMATH_CALUDE_total_distinct_plants_l2507_250793

-- Define the flower beds as finite sets
variable (A B C D : Finset ℕ)

-- Define the cardinalities of the sets
variable (hA : A.card = 550)
variable (hB : B.card = 500)
variable (hC : C.card = 400)
variable (hD : D.card = 350)

-- Define the intersections
variable (hAB : (A ∩ B).card = 60)
variable (hAC : (A ∩ C).card = 110)
variable (hAD : (A ∩ D).card = 70)
variable (hABC : (A ∩ B ∩ C).card = 30)

-- Define the empty intersections
variable (hBC : (B ∩ C).card = 0)
variable (hBD : (B ∩ D).card = 0)

-- State the theorem
theorem total_distinct_plants :
  (A ∪ B ∪ C ∪ D).card = 1590 :=
sorry

end NUMINAMATH_CALUDE_total_distinct_plants_l2507_250793


namespace NUMINAMATH_CALUDE_det_trig_matrix_zero_l2507_250733

theorem det_trig_matrix_zero (a c : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![1, Real.sin (a + c), Real.sin a; 
                                        Real.sin (a + c), 1, Real.sin c; 
                                        Real.sin a, Real.sin c, 1]
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_trig_matrix_zero_l2507_250733


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l2507_250708

/-- The number of wrapping paper varieties -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 5

/-- The number of gift card types -/
def gift_card_types : ℕ := 4

/-- The number of decorative bow types -/
def bow_types : ℕ := 2

/-- The total number of distinct gift-wrapping combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types * bow_types

theorem gift_wrapping_combinations :
  total_combinations = 400 :=
by sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l2507_250708


namespace NUMINAMATH_CALUDE_circles_have_three_common_tangents_l2507_250743

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 9

-- Define the centers and radii
def center1 : ℝ × ℝ := (-1, -2)
def center2 : ℝ × ℝ := (2, 2)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem statement
theorem circles_have_three_common_tangents :
  ∃! (n : ℕ), n = 3 ∧ 
  (∃ (tangents : Finset (ℝ → ℝ)), tangents.card = n ∧ 
    (∀ f ∈ tangents, ∀ x y : ℝ, 
      (circle1 x y → (y = f x ∨ y = -f x)) ∧ 
      (circle2 x y → (y = f x ∨ y = -f x)))) := by sorry

end NUMINAMATH_CALUDE_circles_have_three_common_tangents_l2507_250743


namespace NUMINAMATH_CALUDE_square_rectangle_perimeter_sum_l2507_250730

theorem square_rectangle_perimeter_sum :
  ∀ (s l w : ℝ),
  s > 0 ∧ l > 0 ∧ w > 0 →
  s^2 + l * w = 130 →
  s^2 - l * w = 50 →
  l = 2 * w →
  4 * s + 2 * (l + w) = 12 * Real.sqrt 10 + 12 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_square_rectangle_perimeter_sum_l2507_250730


namespace NUMINAMATH_CALUDE_new_average_weight_l2507_250771

theorem new_average_weight (initial_students : Nat) (initial_avg_weight : ℝ) (new_student_weight : ℝ) :
  initial_students = 19 →
  initial_avg_weight = 15 →
  new_student_weight = 7 →
  let total_weight := initial_students * initial_avg_weight
  let new_total_weight := total_weight + new_student_weight
  let new_avg_weight := new_total_weight / (initial_students + 1)
  new_avg_weight = 14.6 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l2507_250771


namespace NUMINAMATH_CALUDE_function_property_l2507_250735

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (x - 1) * (deriv f x) ≤ 0)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- Define the theorem
theorem function_property (x₁ x₂ : ℝ) (h3 : |x₁ - 1| < |x₂ - 1|) :
  f (2 - x₁) ≥ f (2 - x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2507_250735


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_greater_than_two_l2507_250714

theorem quadratic_inequality_implies_a_greater_than_two (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 - a*x + 1 < 0) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_greater_than_two_l2507_250714


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2507_250797

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Main theorem: If S_6 / S_3 = 4 for an arithmetic sequence, then S_9 / S_6 = 9/4 -/
theorem arithmetic_sequence_sum_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 6 / seq.S 3 = 4) : 
  seq.S 9 / seq.S 6 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2507_250797


namespace NUMINAMATH_CALUDE_project_hours_difference_l2507_250737

theorem project_hours_difference (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 117 →
  pat_hours = 2 * kate_hours →
  pat_hours * 3 = mark_hours →
  total_hours = kate_hours + pat_hours + mark_hours →
  mark_hours - kate_hours = 65 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l2507_250737


namespace NUMINAMATH_CALUDE_boys_who_left_l2507_250784

theorem boys_who_left (initial_boys : ℕ) (initial_girls : ℕ) (additional_girls : ℕ) (final_total : ℕ) : 
  initial_boys = 5 →
  initial_girls = 4 →
  additional_girls = 2 →
  final_total = 8 →
  initial_boys - (final_total - (initial_girls + additional_girls)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_boys_who_left_l2507_250784


namespace NUMINAMATH_CALUDE_towel_shrinkage_l2507_250787

theorem towel_shrinkage (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let new_length := 0.8 * L
  let new_area := 0.64 * (L * B)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.8 * B ∧ 
    new_length * new_breadth = new_area := by
  sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l2507_250787


namespace NUMINAMATH_CALUDE_initial_average_score_l2507_250754

theorem initial_average_score 
  (total_students : Nat) 
  (remaining_students : Nat)
  (dropped_score : Real)
  (new_average : Real) :
  total_students = 16 →
  remaining_students = 15 →
  dropped_score = 24 →
  new_average = 64 →
  (total_students : Real) * (remaining_students * new_average + dropped_score) / total_students = 61.5 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_score_l2507_250754


namespace NUMINAMATH_CALUDE_smallest_among_four_rationals_l2507_250750

theorem smallest_among_four_rationals :
  let S : Set ℚ := {-1, 0, 1, 2}
  ∀ x ∈ S, -1 ≤ x
  ∧ ∃ y ∈ S, y = -1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_among_four_rationals_l2507_250750


namespace NUMINAMATH_CALUDE_total_slices_is_16_l2507_250751

/-- The number of pizzas Mrs. Hilt bought -/
def num_pizzas : ℕ := 2

/-- The number of slices per pizza -/
def slices_per_pizza : ℕ := 8

/-- The total number of pizza slices Mrs. Hilt had -/
def total_slices : ℕ := num_pizzas * slices_per_pizza

/-- Theorem stating that the total number of pizza slices is 16 -/
theorem total_slices_is_16 : total_slices = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_is_16_l2507_250751


namespace NUMINAMATH_CALUDE_cylinder_from_equation_l2507_250717

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Definition of a cylinder in cylindrical coordinates -/
def isCylinder (S : Set CylindricalPoint) (d : ℝ) : Prop :=
  d > 0 ∧ S = {p : CylindricalPoint | p.r = d}

/-- The main theorem: the set of points satisfying r = d forms a cylinder -/
theorem cylinder_from_equation (d : ℝ) :
  let S := {p : CylindricalPoint | p.r = d}
  d > 0 → isCylinder S d := by
  sorry


end NUMINAMATH_CALUDE_cylinder_from_equation_l2507_250717


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2507_250722

theorem reciprocal_inequality (a b : ℝ) :
  (a > b ∧ a * b > 0 → 1 / a < 1 / b) ∧
  (a > b ∧ a * b < 0 → 1 / a > 1 / b) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2507_250722


namespace NUMINAMATH_CALUDE_class_size_l2507_250778

/-- The number of chocolate bars Gerald brings --/
def gerald_bars : ℕ := 7

/-- The number of squares in each chocolate bar --/
def squares_per_bar : ℕ := 8

/-- The number of additional bars the teacher brings for each of Gerald's bars --/
def teacher_multiplier : ℕ := 2

/-- The number of squares each student gets --/
def squares_per_student : ℕ := 7

/-- The total number of chocolate bars --/
def total_bars : ℕ := gerald_bars + gerald_bars * teacher_multiplier

/-- The total number of chocolate squares --/
def total_squares : ℕ := total_bars * squares_per_bar

/-- The number of students in the class --/
def num_students : ℕ := total_squares / squares_per_student

theorem class_size : num_students = 24 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2507_250778


namespace NUMINAMATH_CALUDE_polynomial_roots_l2507_250729

theorem polynomial_roots : ∃ (x : ℝ), x^5 - 3*x^4 + 3*x^2 - x - 6 = 0 ↔ x = -1 ∨ x = 1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2507_250729


namespace NUMINAMATH_CALUDE_jar_red_marble_difference_l2507_250736

-- Define the ratios for each jar
def jar_a_ratio : Rat := 5 / 3
def jar_b_ratio : Rat := 3 / 2

-- Define the total number of white marbles
def total_white_marbles : ℕ := 70

-- Theorem statement
theorem jar_red_marble_difference :
  ∃ (total_marbles : ℕ) (jar_a_red jar_a_white jar_b_red jar_b_white : ℕ),
    -- Both jars have equal number of marbles
    jar_a_red + jar_a_white = total_marbles ∧
    jar_b_red + jar_b_white = total_marbles ∧
    -- Ratio conditions
    jar_a_red / jar_a_white = jar_a_ratio ∧
    jar_b_red / jar_b_white = jar_b_ratio ∧
    -- Total white marbles condition
    jar_a_white + jar_b_white = total_white_marbles ∧
    -- Difference in red marbles
    jar_a_red - jar_b_red = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jar_red_marble_difference_l2507_250736


namespace NUMINAMATH_CALUDE_tan_y_plus_pi_third_l2507_250752

theorem tan_y_plus_pi_third (y : ℝ) (h : Real.tan y = -1) : 
  Real.tan (y + π/3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_y_plus_pi_third_l2507_250752


namespace NUMINAMATH_CALUDE_meeting_time_is_lcm_l2507_250766

/-- The lap times of the four friends in minutes -/
def lap_times : List Nat := [5, 8, 9, 12]

/-- The time in minutes after 10:00 AM when all friends meet -/
def meeting_time : Nat := 360

/-- Theorem stating that the meeting time is the LCM of the lap times -/
theorem meeting_time_is_lcm : 
  meeting_time = Nat.lcm (Nat.lcm (Nat.lcm (lap_times.get! 0) (lap_times.get! 1)) (lap_times.get! 2)) (lap_times.get! 3) :=
by sorry

end NUMINAMATH_CALUDE_meeting_time_is_lcm_l2507_250766


namespace NUMINAMATH_CALUDE_frustum_volume_ratio_l2507_250715

/-- Given a frustum with base area ratio 1:9, prove the volume ratio of parts divided by midsection is 7:19 -/
theorem frustum_volume_ratio (A₁ A₂ V₁ V₂ : ℝ) (h_area_ratio : A₁ / A₂ = 1 / 9) :
  V₁ / V₂ = 7 / 19 := by
  sorry

end NUMINAMATH_CALUDE_frustum_volume_ratio_l2507_250715


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2507_250753

theorem quadratic_inequality_solution_set :
  {x : ℝ | 4*x^2 - 12*x + 5 < 0} = Set.Ioo (1/2 : ℝ) (5/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2507_250753


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_minus1_l2507_250792

theorem rationalize_denominator_sqrt3_minus1 : 
  (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_minus1_l2507_250792


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2507_250720

/-- Proves that a train with given length, crossing a bridge of given length in a given time, has a specific speed in km/h -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 200 →
  crossing_time = 31.99744020478362 →
  (((train_length + bridge_length) / crossing_time) * 3.6) = 36 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2507_250720


namespace NUMINAMATH_CALUDE_sixth_grade_total_l2507_250790

theorem sixth_grade_total (girls boys : ℕ) : 
  girls = boys + 2 →
  girls / 11 + 22 = (girls - girls / 11) / 2 + 22 →
  girls + boys = 86 :=
by sorry

end NUMINAMATH_CALUDE_sixth_grade_total_l2507_250790


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2507_250774

theorem cube_root_simplification (N : ℝ) (h : N > 1) :
  (N^2 * (N^3 * N^(2/3))^(1/3))^(1/3) = N^(29/27) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2507_250774


namespace NUMINAMATH_CALUDE_quadratic_no_rational_solution_l2507_250727

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluation of a quadratic polynomial at a point x -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Statement: For any quadratic polynomial with real coefficients, 
    there exists a natural number n such that p(x) = 1/n has no rational solutions -/
theorem quadratic_no_rational_solution (p : QuadraticPolynomial) :
  ∃ n : ℕ, ∀ x : ℚ, evaluate p x ≠ 1 / n := by sorry

end NUMINAMATH_CALUDE_quadratic_no_rational_solution_l2507_250727


namespace NUMINAMATH_CALUDE_prob_three_sixes_is_one_over_216_l2507_250755

/-- The number of faces on a standard die -/
def standard_die_faces : ℕ := 6

/-- The probability of rolling a specific number on a standard die -/
def prob_single_roll (n : ℕ) : ℚ := 1 / standard_die_faces

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The target sum we're aiming for -/
def target_sum : ℕ := 18

/-- The probability of rolling the target sum with the given number of dice -/
def prob_target_sum : ℚ := (prob_single_roll target_sum) ^ num_dice

theorem prob_three_sixes_is_one_over_216 : prob_target_sum = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_sixes_is_one_over_216_l2507_250755


namespace NUMINAMATH_CALUDE_unique_solution_system_l2507_250782

theorem unique_solution_system :
  ∃! (x y z : ℝ), x + 3 * y = 10 ∧ y = 3 ∧ 2 * x - y + z = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2507_250782


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2507_250731

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h1 : area = 36) (h2 : leg = 12) :
  ∃ (perimeter : ℝ), perimeter = 18 + 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2507_250731


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_sum_18_l2507_250773

theorem largest_of_three_consecutive_integers_sum_18 (a b c : ℤ) : 
  (b = a + 1) →  -- b is the next consecutive integer after a
  (c = b + 1) →  -- c is the next consecutive integer after b
  (a + b + c = 18) →  -- sum of the three integers is 18
  (c = 7) -- c (the largest) is 7
:= by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_sum_18_l2507_250773


namespace NUMINAMATH_CALUDE_jerry_walking_problem_l2507_250789

/-- Jerry's walking problem -/
theorem jerry_walking_problem (monday_miles tuesday_miles total_miles : ℕ) :
  monday_miles = 9 →
  total_miles = 18 →
  total_miles = monday_miles + tuesday_miles →
  tuesday_miles = 9 := by
sorry

end NUMINAMATH_CALUDE_jerry_walking_problem_l2507_250789


namespace NUMINAMATH_CALUDE_root_difference_theorem_l2507_250718

theorem root_difference_theorem (k : ℝ) : 
  (∃ α β : ℝ, (α^2 + k*α + 8 = 0 ∧ β^2 + k*β + 8 = 0) ∧
              ((α+3)^2 - k*(α+3) + 12 = 0 ∧ (β+3)^2 - k*(β+3) + 12 = 0)) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_theorem_l2507_250718


namespace NUMINAMATH_CALUDE_rodney_commission_l2507_250776

/-- Rodney's commission for selling home security systems --/
def commission_per_sale : ℕ := 25

/-- Number of streets in the neighborhood --/
def num_streets : ℕ := 4

/-- Number of houses on each street --/
def houses_per_street : ℕ := 8

/-- Sales on the second street --/
def sales_second_street : ℕ := 4

/-- Sales on the first street (half of second street) --/
def sales_first_street : ℕ := sales_second_street / 2

/-- Sales on the third street (no sales) --/
def sales_third_street : ℕ := 0

/-- Sales on the fourth street --/
def sales_fourth_street : ℕ := 1

/-- Total sales across all streets --/
def total_sales : ℕ := sales_first_street + sales_second_street + sales_third_street + sales_fourth_street

/-- Rodney's total commission --/
def total_commission : ℕ := total_sales * commission_per_sale

theorem rodney_commission : total_commission = 175 := by
  sorry

end NUMINAMATH_CALUDE_rodney_commission_l2507_250776


namespace NUMINAMATH_CALUDE_expression_evaluation_l2507_250799

theorem expression_evaluation : (-5)^5 / 5^3 + 3^4 - 6^1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2507_250799


namespace NUMINAMATH_CALUDE_gcd_13n_plus_4_7n_plus_2_max_2_l2507_250744

theorem gcd_13n_plus_4_7n_plus_2_max_2 :
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) ≤ 2) ∧
  (∃ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_gcd_13n_plus_4_7n_plus_2_max_2_l2507_250744


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2507_250762

/-- Definition of a geometric progression for three real numbers -/
def is_geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The condition b^2 = ac -/
def condition (a b c : ℝ) : Prop := b^2 = a * c

/-- Theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ a b c : ℝ, is_geometric_progression a b c → condition a b c) ∧
  ¬(∀ a b c : ℝ, condition a b c → is_geometric_progression a b c) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2507_250762


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2507_250785

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (7 * x - 50 * y = 3) ∧ 
    (3 * y - x = 5) ∧ 
    (x = -259/29) ∧ 
    (y = -38/29) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2507_250785


namespace NUMINAMATH_CALUDE_pencil_count_l2507_250775

theorem pencil_count (initial : ℕ) (added : ℕ) (total : ℕ) : 
  initial = 27 → added = 45 → total = initial + added → total = 72 := by sorry

end NUMINAMATH_CALUDE_pencil_count_l2507_250775


namespace NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_l2507_250794

theorem angle_with_complement_40percent_of_supplement (x : ℝ) : 
  (90 - x = 0.4 * (180 - x)) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_l2507_250794


namespace NUMINAMATH_CALUDE_perpendicular_vector_implies_y_equals_five_l2507_250732

/-- Given points A and B, and vector a, proves that if AB is perpendicular to a, then y = 5 -/
theorem perpendicular_vector_implies_y_equals_five (A B : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (10, 1) →
  B.1 = 2 →
  a = (1, 2) →
  (B.1 - A.1) * a.1 + (B.2 - A.2) * a.2 = 0 →
  B.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_implies_y_equals_five_l2507_250732


namespace NUMINAMATH_CALUDE_laptop_sticker_price_l2507_250705

/-- The sticker price of a laptop --/
def stickerPrice : ℝ := sorry

/-- The price at store A after discount and rebate --/
def priceA : ℝ := 0.82 * stickerPrice - 100

/-- The price at store B after discount --/
def priceB : ℝ := 0.75 * stickerPrice

/-- Theorem stating that the sticker price is $1300 given the conditions --/
theorem laptop_sticker_price : 
  priceB - priceA = 10 → stickerPrice = 1300 := by sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_l2507_250705


namespace NUMINAMATH_CALUDE_not_right_triangle_6_7_8_l2507_250707

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- Theorem stating that 6, 7, and 8 cannot form a right triangle --/
theorem not_right_triangle_6_7_8 : ¬ is_right_triangle 6 7 8 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_6_7_8_l2507_250707


namespace NUMINAMATH_CALUDE_curve_intersection_theorem_l2507_250786

/-- Curve C in polar coordinates -/
def curve_C (ρ θ a : ℝ) : Prop := ρ * Real.sin (2 * θ) = 2 * a * Real.cos θ

/-- Point P -/
def point_P : ℝ × ℝ := (-2, -4)

/-- Line l passing through point P -/
def line_l (t : ℝ) : ℝ × ℝ := (t - 2, t - 4)

/-- Cartesian equation of curve C -/
def curve_C_cartesian (x y a : ℝ) : Prop := y^2 = 2 * a * x

/-- Condition for points A and B on the intersection of line l and curve C -/
def intersection_condition (t₁ t₂ a : ℝ) : Prop :=
  curve_C_cartesian (t₁ - 2) (t₁ - 4) a ∧ curve_C_cartesian (t₂ - 2) (t₂ - 4) a

/-- The main theorem -/
theorem curve_intersection_theorem (a : ℝ) :
  (a > 0) →
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ intersection_condition t₁ t₂ a ∧
    (t₁ - 2)^2 + (t₁ - 4)^2 * (t₂ - 2)^2 + (t₂ - 4)^2 = ((t₁ - t₂)^2 + (t₁ - t₂)^2)) →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_intersection_theorem_l2507_250786


namespace NUMINAMATH_CALUDE_matrix_paths_count_l2507_250738

/-- Represents a position on the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents a letter in the word "MATRIX" -/
inductive Letter
| M | A | T | R | I | X

/-- Represents the 5x5 grid of letters -/
def grid : Position → Letter := sorry

/-- Checks if two positions are adjacent (horizontally, vertically, or diagonally) -/
def adjacent (p1 p2 : Position) : Prop := sorry

/-- Represents a valid path spelling "MATRIX" -/
def valid_path (path : List Position) : Prop := sorry

/-- Counts the number of valid paths starting from a given position -/
def count_paths_from (start : Position) : ℕ := sorry

/-- Counts the total number of valid paths in the grid -/
def total_paths : ℕ := sorry

/-- Theorem stating that the total number of paths spelling "MATRIX" is 48 -/
theorem matrix_paths_count :
  total_paths = 48 := by sorry

end NUMINAMATH_CALUDE_matrix_paths_count_l2507_250738


namespace NUMINAMATH_CALUDE_basket_balls_count_l2507_250757

theorem basket_balls_count (total : ℕ) (red : ℕ) (yellow : ℕ) (prob : ℚ) : 
  red = 8 →
  prob = 2/5 →
  total = red + yellow →
  prob = red / total →
  yellow = 12 := by
sorry

end NUMINAMATH_CALUDE_basket_balls_count_l2507_250757


namespace NUMINAMATH_CALUDE_larger_number_problem_l2507_250764

theorem larger_number_problem (smaller larger : ℚ) : 
  smaller = 48 → 
  larger - smaller = (1 : ℚ) / 3 * larger →
  larger = 72 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2507_250764


namespace NUMINAMATH_CALUDE_problem_solution_l2507_250700

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≥ 0 ↔ x ≤ -3 ∨ (23 ≤ x ∧ x < 27))
  (h2 : a < b) :
  a + 2*b + 3*c = 71 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2507_250700


namespace NUMINAMATH_CALUDE_rhombus_area_l2507_250763

/-- The area of a rhombus with vertices at (0, 3.5), (12, 0), (0, -3.5), and (-12, 0) is 84 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (12, 0), (0, -3.5), (-12, 0)]
  let diagonal1 : ℝ := |3.5 - (-3.5)|
  let diagonal2 : ℝ := |12 - (-12)|
  let area : ℝ := (diagonal1 * diagonal2) / 2
  area = 84 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l2507_250763


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_2_union_equals_B_iff_l2507_250711

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 6}

theorem intersection_complement_when_m_2 :
  A ∩ (Bᶜ 2) = {x | -1 ≤ x ∧ x < 2} := by sorry

theorem union_equals_B_iff (m : ℝ) :
  A ∪ B m = B m ↔ -3 ≤ m ∧ m ≤ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_2_union_equals_B_iff_l2507_250711


namespace NUMINAMATH_CALUDE_dihedral_angle_ge_line_angle_l2507_250706

/-- A dihedral angle with its plane angle -/
structure DihedralAngle where
  φ : Real
  φ_nonneg : 0 ≤ φ
  φ_le_pi : φ ≤ π

/-- A line contained in one plane of a dihedral angle -/
structure ContainedLine (d : DihedralAngle) where
  θ : Real
  θ_nonneg : 0 ≤ θ
  θ_le_pi_div_2 : θ ≤ π / 2

/-- The plane angle of a dihedral angle is always greater than or equal to 
    the angle between any line in one of its planes and the other plane -/
theorem dihedral_angle_ge_line_angle (d : DihedralAngle) (l : ContainedLine d) : 
  d.φ ≥ l.θ := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_ge_line_angle_l2507_250706


namespace NUMINAMATH_CALUDE_quiz_score_percentage_l2507_250788

/-- Given a quiz with 25 items where a student makes 5 mistakes, 
    the percentage score obtained is 80%. -/
theorem quiz_score_percentage (total_items : ℕ) (mistakes : ℕ) 
  (h1 : total_items = 25) (h2 : mistakes = 5) : 
  (((total_items - mistakes) / total_items) * 100 : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_percentage_l2507_250788


namespace NUMINAMATH_CALUDE_f_of_tan_squared_l2507_250728

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem f_of_tan_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/4) :
  (∀ x : ℝ, x ≠ 0 → x ≠ 1 → f (x / (x - 1)) = 1 / x) →
  f (Real.tan t ^ 2) = Real.tan t ^ 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_tan_squared_l2507_250728


namespace NUMINAMATH_CALUDE_phoebe_age_proof_l2507_250702

/-- Phoebe's current age -/
def phoebe_age : ℕ := 10

/-- Raven's current age -/
def raven_age : ℕ := 55

theorem phoebe_age_proof :
  (raven_age + 5 = 4 * (phoebe_age + 5)) → phoebe_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_phoebe_age_proof_l2507_250702


namespace NUMINAMATH_CALUDE_increasing_derivative_relation_l2507_250742

open Set
open Function
open Real

-- Define the interval (a, b)
variable (a b : ℝ) (hab : a < b)

-- Define a real-valued function on the interval (a, b)
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on (a, b)
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Define the derivative of f
variable (f' : ℝ → ℝ)
variable (hf' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)

-- State the theorem
theorem increasing_derivative_relation :
  (∀ x ∈ Ioo a b, f' x > 0 → IsIncreasing f a b) ∧
  ∃ f : ℝ → ℝ, IsIncreasing f a b ∧ ¬(∀ x ∈ Ioo a b, f' x > 0) :=
sorry

end NUMINAMATH_CALUDE_increasing_derivative_relation_l2507_250742


namespace NUMINAMATH_CALUDE_product_sum_base_k_l2507_250716

theorem product_sum_base_k (k : ℕ) (hk : k > 0) :
  (k + 3) * (k + 4) * (k + 7) = 4 * k^3 + 7 * k^2 + 3 * k + 5 →
  (3 * k + 14).digits k = [5, 0] :=
by sorry

end NUMINAMATH_CALUDE_product_sum_base_k_l2507_250716


namespace NUMINAMATH_CALUDE_sector_angle_l2507_250781

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, 
    prove that its central angle is 2 radians. -/
theorem sector_angle (r : ℝ) (θ : ℝ) 
  (h_area : (1/2) * θ * r^2 = 1)
  (h_perim : 2*r + θ*r = 4) : 
  θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l2507_250781


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l2507_250765

/-- Given a line y = kx + b tangent to the curve y = x³ + ax + 1 at (2,3), prove b = -15 -/
theorem tangent_line_b_value (k a b : ℝ) : 
  (3 = 2 * k + b) →  -- Line equation at (2,3)
  (3 = 2^3 + 2*a + 1) →  -- Curve equation at (2,3)
  (k = 3 * 2^2 + a) →  -- Slope equality condition for tangency
  (b = -15) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l2507_250765


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l2507_250724

theorem prime_square_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l2507_250724


namespace NUMINAMATH_CALUDE_buffy_whiskers_l2507_250791

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  juniper : ℕ
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ

/-- The conditions for the cat whiskers problem -/
def whiskerConditions (c : CatWhiskers) : Prop :=
  c.juniper = 12 ∧
  c.puffy = 3 * c.juniper ∧
  c.scruffy = 2 * c.puffy ∧
  c.buffy = (c.juniper + c.puffy + c.scruffy) / 3

/-- Theorem stating that under the given conditions, Buffy has 40 whiskers -/
theorem buffy_whiskers (c : CatWhiskers) :
  whiskerConditions c → c.buffy = 40 := by
  sorry

end NUMINAMATH_CALUDE_buffy_whiskers_l2507_250791


namespace NUMINAMATH_CALUDE_percentage_difference_l2507_250758

theorem percentage_difference : (60 / 100 * 50) - (50 / 100 * 30) = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2507_250758


namespace NUMINAMATH_CALUDE_quiche_volume_l2507_250760

/-- Calculate the total volume of a vegetable quiche --/
theorem quiche_volume 
  (spinach_initial : ℝ) 
  (mushrooms_initial : ℝ) 
  (onions_initial : ℝ)
  (spinach_reduction : ℝ) 
  (mushrooms_reduction : ℝ) 
  (onions_reduction : ℝ)
  (cream_cheese : ℝ)
  (eggs : ℝ)
  (h1 : spinach_initial = 40)
  (h2 : mushrooms_initial = 25)
  (h3 : onions_initial = 15)
  (h4 : spinach_reduction = 0.20)
  (h5 : mushrooms_reduction = 0.65)
  (h6 : onions_reduction = 0.50)
  (h7 : cream_cheese = 6)
  (h8 : eggs = 4) :
  spinach_initial * spinach_reduction + 
  mushrooms_initial * mushrooms_reduction + 
  onions_initial * onions_reduction + 
  cream_cheese + eggs = 41.75 := by
sorry

end NUMINAMATH_CALUDE_quiche_volume_l2507_250760


namespace NUMINAMATH_CALUDE_logarithm_simplification_l2507_250704

open Real

theorem logarithm_simplification (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a ≠ 1) :
  let log_a := fun x => log x / log a
  let log_ab := fun x => log x / log (a * b)
  (log_a b + log_a (b^(1/(2*log b / log (a^2)))))/(log_a b - log_ab b) *
  (log_ab b * log_a b)/(b^(2*log b * log_a b) - 1) = 1 / (log_a b - 1) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l2507_250704
