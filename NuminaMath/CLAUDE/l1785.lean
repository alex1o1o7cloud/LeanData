import Mathlib

namespace sum_of_products_l1785_178541

theorem sum_of_products (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = 9)
  (h2 : y^2 + y*z + z^2 = 16)
  (h3 : z^2 + z*x + x^2 = 25) :
  x*y + y*z + z*x = 8 * Real.sqrt 3 := by
  sorry

end sum_of_products_l1785_178541


namespace modified_number_wall_m_value_l1785_178539

/-- Represents a modified Number Wall with given values -/
structure ModifiedNumberWall where
  m : ℕ
  row1 : Vector ℕ 4
  row2 : Vector ℕ 3
  row3 : Vector ℕ 2
  row4 : ℕ

/-- The modified Number Wall satisfies the sum property -/
def is_valid_wall (wall : ModifiedNumberWall) : Prop :=
  wall.row1.get 0 = wall.m ∧
  wall.row1.get 1 = 5 ∧
  wall.row1.get 2 = 10 ∧
  wall.row1.get 3 = 6 ∧
  wall.row2.get 1 = 18 ∧
  wall.row4 = 56 ∧
  wall.row2.get 0 = wall.row1.get 0 + wall.row1.get 1 ∧
  wall.row2.get 1 = wall.row1.get 1 + wall.row1.get 2 ∧
  wall.row2.get 2 = wall.row1.get 2 + wall.row1.get 3 ∧
  wall.row3.get 0 = wall.row2.get 0 + wall.row2.get 1 ∧
  wall.row3.get 1 = wall.row2.get 1 + wall.row2.get 2 ∧
  wall.row4 = wall.row3.get 0 + wall.row3.get 1

/-- The value of 'm' in a valid modified Number Wall is 17 -/
theorem modified_number_wall_m_value (wall : ModifiedNumberWall) :
  is_valid_wall wall → wall.m = 17 := by sorry

end modified_number_wall_m_value_l1785_178539


namespace skew_lines_equivalent_l1785_178595

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line

-- Define a type for planes in 3D space
structure Plane3D where
  -- Add necessary fields to represent a plane

-- Define what it means for two lines to be parallel
def parallel (a b : Line3D) : Prop :=
  sorry

-- Define what it means for a line to be a subset of a plane
def line_subset_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Define what it means for two lines to intersect
def intersect (a b : Line3D) : Prop :=
  sorry

-- Define skew lines according to the first definition
def skew_def1 (a b : Line3D) : Prop :=
  ¬(intersect a b) ∧ ¬(parallel a b)

-- Define skew lines according to the second definition
def skew_def2 (a b : Line3D) : Prop :=
  ¬∃ (p : Plane3D), line_subset_plane a p ∧ line_subset_plane b p

-- Theorem stating the equivalence of the two definitions
theorem skew_lines_equivalent (a b : Line3D) :
  skew_def1 a b ↔ skew_def2 a b :=
sorry

end skew_lines_equivalent_l1785_178595


namespace sum_divisible_by_101_iff_digits_congruent_l1785_178509

/-- Represents a four-digit positive integer with different non-zero digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  d_pos : d > 0
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The value of the reverse of a four-digit number -/
def reverse_value (n : FourDigitNumber) : Nat :=
  1000 * n.d + 100 * n.c + 10 * n.b + n.a

/-- The theorem stating the condition for the sum of a number and its reverse to be divisible by 101 -/
theorem sum_divisible_by_101_iff_digits_congruent (n : FourDigitNumber) :
  (value n + reverse_value n) % 101 = 0 ↔ (n.a + n.d) % 101 = (n.b + n.c) % 101 := by
  sorry

end sum_divisible_by_101_iff_digits_congruent_l1785_178509


namespace exactly_one_statement_implies_negation_l1785_178592

def statement1 (p q : Prop) : Prop := p ∨ q
def statement2 (p q : Prop) : Prop := p ∧ ¬q
def statement3 (p q : Prop) : Prop := ¬p ∧ q
def statement4 (p q : Prop) : Prop := ¬p ∧ ¬q

def negation_of_or (p q : Prop) : Prop := ¬(p ∨ q)

theorem exactly_one_statement_implies_negation (p q : Prop) :
  (∃! i : Fin 4, match i with
    | 0 => statement1 p q → negation_of_or p q
    | 1 => statement2 p q → negation_of_or p q
    | 2 => statement3 p q → negation_of_or p q
    | 3 => statement4 p q → negation_of_or p q) :=
by sorry

end exactly_one_statement_implies_negation_l1785_178592


namespace intersection_condition_l1785_178587

def A : Set (ℕ × ℝ) := {p | 3 * p.1 + p.2 - 2 = 0}

def B (k : ℤ) : Set (ℕ × ℝ) := {p | k * (p.1^2 - p.1 + 1) - p.2 = 0}

theorem intersection_condition (k : ℤ) : 
  k ≠ 0 → (∃ p : ℕ × ℝ, p ∈ A ∩ B k) → k = -1 ∨ k = 2 := by
  sorry

end intersection_condition_l1785_178587


namespace bromine_only_liquid_l1785_178583

-- Define the set of elements
inductive Element : Type
| Bromine : Element
| Krypton : Element
| Phosphorus : Element
| Xenon : Element

-- Define the state of matter
inductive State : Type
| Solid : State
| Liquid : State
| Gas : State

-- Define the function to determine the state of an element at given temperature and pressure
def stateAtConditions (e : Element) (temp : ℝ) (pressure : ℝ) : State := sorry

-- Define the temperature and pressure conditions
def roomTemp : ℝ := 25
def atmPressure : ℝ := 1.0

-- Theorem statement
theorem bromine_only_liquid :
  ∀ e : Element, 
    stateAtConditions e roomTemp atmPressure = State.Liquid ↔ e = Element.Bromine :=
sorry

end bromine_only_liquid_l1785_178583


namespace male_average_score_l1785_178521

theorem male_average_score 
  (female_count : ℕ) 
  (male_count : ℕ) 
  (total_count : ℕ) 
  (female_avg : ℚ) 
  (total_avg : ℚ) 
  (h1 : female_count = 20)
  (h2 : male_count = 30)
  (h3 : total_count = female_count + male_count)
  (h4 : female_avg = 75)
  (h5 : total_avg = 72) :
  (total_count * total_avg - female_count * female_avg) / male_count = 70 := by
sorry

end male_average_score_l1785_178521


namespace helen_oranges_l1785_178534

/-- Given that Helen starts with 9 oranges and receives 29 more from Ann, 
    prove that she ends up with 38 oranges in total. -/
theorem helen_oranges : 
  let initial_oranges : ℕ := 9
  let oranges_from_ann : ℕ := 29
  initial_oranges + oranges_from_ann = 38 := by sorry

end helen_oranges_l1785_178534


namespace binomial_60_3_l1785_178578

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end binomial_60_3_l1785_178578


namespace min_area_of_B_l1785_178503

-- Define set A
def A : Set (ℝ × ℝ) := {p | |p.1 - 2| + |p.2 - 3| ≤ 1}

-- Define set B
def B (D E F : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F ≤ 0}

-- State the theorem
theorem min_area_of_B (D E F : ℝ) (h1 : D^2 + E^2 - 4*F > 0) (h2 : A ⊆ B D E F) :
  ∃ (S : ℝ), S = 2 * Real.pi ∧ ∀ (S' : ℝ), (∃ (D' E' F' : ℝ), D'^2 + E'^2 - 4*F' > 0 ∧ A ⊆ B D' E' F' ∧ S' = Real.pi * ((D'^2 + E'^2) / 4 - F')) → S ≤ S' :=
sorry

end min_area_of_B_l1785_178503


namespace unique_integer_divisible_by_24_with_specific_cube_root_l1785_178514

theorem unique_integer_divisible_by_24_with_specific_cube_root : 
  ∃! n : ℕ+, (∃ k : ℕ, n = 24 * k) ∧ 9 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 9.1 :=
by sorry

end unique_integer_divisible_by_24_with_specific_cube_root_l1785_178514


namespace smallest_factorization_coefficient_l1785_178551

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ (r s : ℤ), x^2 + b*x + 1764 = (x + r) * (x + s)) ∧ 
  (∀ (b' : ℕ), b' < b → ¬∃ (r s : ℤ), x^2 + b'*x + 1764 = (x + r) * (x + s)) → 
  b = 84 := by
  sorry

end smallest_factorization_coefficient_l1785_178551


namespace exists_n_with_specific_digit_sums_l1785_178570

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 100
    and the sum of the digits of n^3 is 1,000,000 -/
theorem exists_n_with_specific_digit_sums :
  ∃ n : ℕ, sum_of_digits n = 100 ∧ sum_of_digits (n^3) = 1000000 := by
  sorry

end exists_n_with_specific_digit_sums_l1785_178570


namespace sets_equality_implies_x_minus_y_l1785_178549

-- Define the sets A and B
def A (x y : ℝ) : Set ℝ := {1, x, y}
def B (x y : ℝ) : Set ℝ := {1, x^2, 2*y}

-- State the theorem
theorem sets_equality_implies_x_minus_y (x y : ℝ) : 
  A x y = B x y → x - y = 1/4 := by
  sorry

end sets_equality_implies_x_minus_y_l1785_178549


namespace possible_values_of_a_l1785_178565

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : B a ⊆ A → a ∈ ({1, -1, 0} : Set ℝ) := by
  sorry

end possible_values_of_a_l1785_178565


namespace square_window_side_length_l1785_178528

/-- Given three rectangles with perimeters 8, 10, and 12 that form a square window,
    prove that the side length of the square window is 4. -/
theorem square_window_side_length 
  (a b c : ℝ) 
  (h1 : 2*b + 2*c = 8)   -- perimeter of bottom-left rectangle
  (h2 : 2*(a - b) + 2*a = 10) -- perimeter of top rectangle
  (h3 : 2*b + 2*(a - c) = 12) -- perimeter of right rectangle
  : a = 4 := by
  sorry


end square_window_side_length_l1785_178528


namespace image_of_one_three_preimage_of_one_three_l1785_178532

-- Define the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Define the set A (which is the same as B)
def A : Set (ℝ × ℝ) := Set.univ

-- Theorem for the image of (1,3)
theorem image_of_one_three : f (1, 3) = (4, -2) := by sorry

-- Theorem for the preimage of (1,3)
theorem preimage_of_one_three : f (2, -1) = (1, 3) := by sorry

end image_of_one_three_preimage_of_one_three_l1785_178532


namespace semicircle_perimeter_l1785_178516

/-- The perimeter of a semi-circle with radius 7 cm is 7π + 14 cm. -/
theorem semicircle_perimeter : 
  ∀ (r : ℝ), r = 7 → (π * r + 2 * r) = 7 * π + 14 := by
  sorry

end semicircle_perimeter_l1785_178516


namespace least_common_multiple_5_to_10_l1785_178563

theorem least_common_multiple_5_to_10 : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, 5 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ 
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 5 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧
  n = 2520 :=
by sorry

end least_common_multiple_5_to_10_l1785_178563


namespace triangle_problem_l1785_178568

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  c = 1 →
  b * Real.sin A = a * Real.sin C →
  0 < A →
  A < Real.pi →
  -- Conclusions
  b = 1 ∧
  (∀ x y z : Real, x > 0 → y > 0 → z > 0 → x * Real.sin y ≤ 1/2 * z * Real.sin x) →
  (∃ x y : Real, x > 0 → y > 0 → 1/2 * c * b * Real.sin x = 1/2) :=
by sorry

end triangle_problem_l1785_178568


namespace circle_passes_800_squares_l1785_178589

/-- A circle on a unit square grid -/
structure GridCircle where
  radius : ℕ
  -- The circle does not touch any grid lines or pass through any lattice points
  no_grid_touch : True

/-- The number of squares a circle passes through on a unit square grid -/
def squares_passed (c : GridCircle) : ℕ :=
  4 * (2 * c.radius)

/-- Theorem: A circle with radius 100 passes through 800 squares -/
theorem circle_passes_800_squares (c : GridCircle) (h : c.radius = 100) :
  squares_passed c = 800 :=
by sorry

end circle_passes_800_squares_l1785_178589


namespace min_cost_tree_purchase_l1785_178560

/-- Represents the cost and quantity of trees --/
structure TreePurchase where
  cypress_price : ℕ
  pine_price : ℕ
  cypress_count : ℕ
  pine_count : ℕ

/-- The conditions of the tree purchasing problem --/
def tree_problem (p : TreePurchase) : Prop :=
  2 * p.cypress_price + 3 * p.pine_price = 850 ∧
  3 * p.cypress_price + 2 * p.pine_price = 900 ∧
  p.cypress_count + p.pine_count = 80 ∧
  p.cypress_count ≥ 2 * p.pine_count

/-- The total cost of a tree purchase --/
def total_cost (p : TreePurchase) : ℕ :=
  p.cypress_price * p.cypress_count + p.pine_price * p.pine_count

/-- The theorem stating the minimum cost and optimal purchase --/
theorem min_cost_tree_purchase :
  ∃ (p : TreePurchase), tree_problem p ∧
    total_cost p = 14700 ∧
    p.cypress_count = 54 ∧
    p.pine_count = 26 ∧
    (∀ (q : TreePurchase), tree_problem q → total_cost q ≥ total_cost p) :=
by sorry

end min_cost_tree_purchase_l1785_178560


namespace sqrt_meaningful_range_l1785_178505

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by
  sorry

end sqrt_meaningful_range_l1785_178505


namespace odd_square_plus_two_divisor_congruence_l1785_178574

theorem odd_square_plus_two_divisor_congruence (a d : ℤ) : 
  Odd a → a > 0 → d ∣ (a^2 + 2) → d % 8 = 1 ∨ d % 8 = 3 := by
  sorry

end odd_square_plus_two_divisor_congruence_l1785_178574


namespace worker_savings_fraction_l1785_178520

/-- A worker saves a constant fraction of her constant monthly take-home pay. -/
structure Worker where
  /-- Monthly take-home pay -/
  P : ℝ
  /-- Fraction of monthly take-home pay saved -/
  f : ℝ
  /-- Monthly take-home pay is positive -/
  P_pos : P > 0
  /-- Savings fraction is between 0 and 1 -/
  f_range : 0 ≤ f ∧ f ≤ 1

/-- The theorem stating that if a worker's yearly savings equals 8 times
    her monthly non-savings, then she saves 2/5 of her income. -/
theorem worker_savings_fraction (w : Worker) 
    (h : 12 * w.f * w.P = 8 * (1 - w.f) * w.P) : 
    w.f = 2 / 5 := by
  sorry

end worker_savings_fraction_l1785_178520


namespace clock_cost_price_l1785_178525

theorem clock_cost_price (total_clocks : ℕ) (clocks_profit1 : ℕ) (clocks_profit2 : ℕ)
  (profit1 : ℚ) (profit2 : ℚ) (uniform_profit : ℚ) (revenue_difference : ℚ) :
  total_clocks = 200 →
  clocks_profit1 = 80 →
  clocks_profit2 = 120 →
  profit1 = 5 / 25 →
  profit2 = 7 / 25 →
  uniform_profit = 6 / 25 →
  revenue_difference = 200 →
  ∃ (cost_price : ℚ),
    cost_price * (clocks_profit1 * (1 + profit1) + clocks_profit2 * (1 + profit2)) -
    cost_price * (total_clocks * (1 + uniform_profit)) = revenue_difference ∧
    cost_price = 125 :=
by sorry

end clock_cost_price_l1785_178525


namespace kevin_max_sum_l1785_178550

def kevin_process (S : Finset ℕ) : Finset ℕ :=
  sorry

theorem kevin_max_sum :
  let initial_set : Finset ℕ := Finset.range 15
  let final_set := kevin_process initial_set
  Finset.sum final_set id = 360864 :=
sorry

end kevin_max_sum_l1785_178550


namespace separation_sister_chromatids_not_in_first_division_l1785_178572

-- Define the events
inductive MeioticEvent
| PairingHomologousChromosomes
| CrossingOver
| SeparationSisterChromatids
| SeparationHomologousChromosomes

-- Define the property of occurring during the first meiotic division
def occursInFirstMeioticDivision : MeioticEvent → Prop :=
  fun event =>
    match event with
    | MeioticEvent.PairingHomologousChromosomes => True
    | MeioticEvent.CrossingOver => True
    | MeioticEvent.SeparationSisterChromatids => False
    | MeioticEvent.SeparationHomologousChromosomes => True

-- Theorem stating that separation of sister chromatids is the only event
-- that does not occur during the first meiotic division
theorem separation_sister_chromatids_not_in_first_division :
  ∀ (e : MeioticEvent),
    ¬occursInFirstMeioticDivision e ↔ e = MeioticEvent.SeparationSisterChromatids :=
by sorry

end separation_sister_chromatids_not_in_first_division_l1785_178572


namespace two_hour_charge_l1785_178566

/-- Represents the pricing scheme of a psychologist's therapy sessions. -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  hourDifference : firstHourCharge = additionalHourCharge + 35

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  if hours = 0 then 0
  else pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

theorem two_hour_charge (pricing : TherapyPricing) 
  (h : totalCharge pricing 5 = 350) : totalCharge pricing 2 = 161 := by
  sorry

#check two_hour_charge

end two_hour_charge_l1785_178566


namespace ben_joe_shirt_difference_l1785_178530

/-- The number of new shirts Alex has -/
def alex_shirts : ℕ := 4

/-- The number of additional shirts Joe has compared to Alex -/
def joe_extra_shirts : ℕ := 3

/-- The number of new shirts Ben has -/
def ben_shirts : ℕ := 15

/-- The number of new shirts Joe has -/
def joe_shirts : ℕ := alex_shirts + joe_extra_shirts

theorem ben_joe_shirt_difference : ben_shirts - joe_shirts = 8 := by
  sorry

end ben_joe_shirt_difference_l1785_178530


namespace right_triangle_legs_l1785_178526

/-- A right triangle with specific median and altitude properties -/
structure RightTriangle where
  -- The length of the median from the right angle vertex
  median : ℝ
  -- The length of the altitude from the right angle vertex
  altitude : ℝ
  -- Condition that the median is 5
  median_is_five : median = 5
  -- Condition that the altitude is 4
  altitude_is_four : altitude = 4

/-- The legs of a right triangle -/
structure TriangleLegs where
  -- The length of the first leg
  leg1 : ℝ
  -- The length of the second leg
  leg2 : ℝ

/-- Theorem stating the legs of the triangle given the median and altitude -/
theorem right_triangle_legs (t : RightTriangle) : 
  ∃ (legs : TriangleLegs), legs.leg1 = 2 * Real.sqrt 5 ∧ legs.leg2 = 4 * Real.sqrt 5 := by
  sorry

end right_triangle_legs_l1785_178526


namespace one_box_can_be_emptied_l1785_178512

/-- Represents a state of three boxes with balls -/
structure BoxState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents an operation of doubling balls in one box by transferring from another -/
inductive DoubleOperation
  | DoubleAFromB
  | DoubleAFromC
  | DoubleBFromA
  | DoubleBFromC
  | DoubleCFromA
  | DoubleCFromB

/-- Applies a single doubling operation to a BoxState -/
def applyOperation (state : BoxState) (op : DoubleOperation) : BoxState :=
  match op with
  | DoubleOperation.DoubleAFromB => ⟨state.a * 2, state.b - state.a, state.c⟩
  | DoubleOperation.DoubleAFromC => ⟨state.a * 2, state.b, state.c - state.a⟩
  | DoubleOperation.DoubleBFromA => ⟨state.a - state.b, state.b * 2, state.c⟩
  | DoubleOperation.DoubleBFromC => ⟨state.a, state.b * 2, state.c - state.b⟩
  | DoubleOperation.DoubleCFromA => ⟨state.a - state.c, state.b, state.c * 2⟩
  | DoubleOperation.DoubleCFromB => ⟨state.a, state.b - state.c, state.c * 2⟩

/-- Applies a sequence of doubling operations to a BoxState -/
def applyOperations (state : BoxState) (ops : List DoubleOperation) : BoxState :=
  ops.foldl applyOperation state

/-- Predicate to check if any box is empty -/
def isAnyBoxEmpty (state : BoxState) : Prop :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- The main theorem stating that one box can be emptied -/
theorem one_box_can_be_emptied (initial : BoxState) :
  ∃ (ops : List DoubleOperation), isAnyBoxEmpty (applyOperations initial ops) :=
sorry

end one_box_can_be_emptied_l1785_178512


namespace smallest_positive_m_for_symmetry_l1785_178548

open Real

/-- The smallest positive value of m for which the function 
    y = sin(2(x-m) + π/6) is symmetric about the y-axis -/
theorem smallest_positive_m_for_symmetry : 
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x : ℝ), sin (2*(x-m) + π/6) = sin (2*(-x-m) + π/6)) ∧
  (∀ (m' : ℝ), 0 < m' ∧ m' < m → 
    ∃ (x : ℝ), sin (2*(x-m') + π/6) ≠ sin (2*(-x-m') + π/6)) ∧
  m = π/3 :=
sorry

end smallest_positive_m_for_symmetry_l1785_178548


namespace second_catch_up_l1785_178540

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ

/-- Represents the state of the race -/
structure RaceState where
  runner1 : Runner
  runner2 : Runner
  laps_completed : ℝ

/-- Defines the initial state of the race -/
def initial_state : RaceState :=
  { runner1 := { speed := 3 },
    runner2 := { speed := 1 },
    laps_completed := 0 }

/-- Defines the state after the second runner doubles their speed -/
def intermediate_state : RaceState :=
  { runner1 := { speed := 3 },
    runner2 := { speed := 2 },
    laps_completed := 0.5 }

/-- Theorem stating that the first runner will catch up again when the second runner has completed 2.5 laps -/
theorem second_catch_up (state : RaceState) :
  state.runner1.speed > state.runner2.speed →
  ∃ t : ℝ, t > 0 ∧ 
    state.runner1.speed * t = (state.laps_completed + 2.5) ∧
    state.runner2.speed * t = 2 :=
  sorry

end second_catch_up_l1785_178540


namespace measure_10_liters_l1785_178596

/-- Represents the state of water in two containers -/
structure WaterState :=
  (container1 : ℕ)  -- Amount of water in container 1 (11-liter container)
  (container2 : ℕ)  -- Amount of water in container 2 (9-liter container)

/-- Defines the possible operations on the water containers -/
inductive WaterOperation
  | Fill1      -- Fill container 1
  | Fill2      -- Fill container 2
  | Empty1     -- Empty container 1
  | Empty2     -- Empty container 2
  | Pour1to2   -- Pour from container 1 to container 2
  | Pour2to1   -- Pour from container 2 to container 1

/-- Applies a single operation to a water state -/
def applyOperation (state : WaterState) (op : WaterOperation) : WaterState :=
  match op with
  | WaterOperation.Fill1    => { container1 := 11, container2 := state.container2 }
  | WaterOperation.Fill2    => { container1 := state.container1, container2 := 9 }
  | WaterOperation.Empty1   => { container1 := 0,  container2 := state.container2 }
  | WaterOperation.Empty2   => { container1 := state.container1, container2 := 0 }
  | WaterOperation.Pour1to2 => 
      let amount := min state.container1 (9 - state.container2)
      { container1 := state.container1 - amount, container2 := state.container2 + amount }
  | WaterOperation.Pour2to1 => 
      let amount := min state.container2 (11 - state.container1)
      { container1 := state.container1 + amount, container2 := state.container2 - amount }

/-- Theorem: It is possible to measure out exactly 10 liters of water -/
theorem measure_10_liters : ∃ (ops : List WaterOperation), 
  (ops.foldl applyOperation { container1 := 0, container2 := 0 }).container1 = 10 ∨
  (ops.foldl applyOperation { container1 := 0, container2 := 0 }).container2 = 10 :=
sorry

end measure_10_liters_l1785_178596


namespace c_share_l1785_178591

def total_amount : ℕ := 880

def share_ratio (a b c : ℕ) : Prop :=
  4 * a = 5 * b ∧ 5 * b = 10 * c

theorem c_share (a b c : ℕ) (h1 : share_ratio a b c) (h2 : a + b + c = total_amount) :
  c = 160 := by
  sorry

end c_share_l1785_178591


namespace veggies_expense_correct_l1785_178584

/-- Calculates the amount spent on veggies given the total amount brought,
    expenses on other items, and the amount left after shopping. -/
def amount_spent_on_veggies (total_brought : ℕ) (meat_expense : ℕ) (chicken_expense : ℕ)
                             (eggs_expense : ℕ) (dog_food_expense : ℕ) (amount_left : ℕ) : ℕ :=
  total_brought - (meat_expense + chicken_expense + eggs_expense + dog_food_expense + amount_left)

/-- Proves that the amount Trisha spent on veggies is correct given the problem conditions. -/
theorem veggies_expense_correct (total_brought : ℕ) (meat_expense : ℕ) (chicken_expense : ℕ)
                                 (eggs_expense : ℕ) (dog_food_expense : ℕ) (amount_left : ℕ)
                                 (h1 : total_brought = 167)
                                 (h2 : meat_expense = 17)
                                 (h3 : chicken_expense = 22)
                                 (h4 : eggs_expense = 5)
                                 (h5 : dog_food_expense = 45)
                                 (h6 : amount_left = 35) :
  amount_spent_on_veggies total_brought meat_expense chicken_expense eggs_expense dog_food_expense amount_left = 43 :=
by
  sorry

#eval amount_spent_on_veggies 167 17 22 5 45 35

end veggies_expense_correct_l1785_178584


namespace p_sufficient_not_necessary_for_q_l1785_178513

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a > 0

def q (a : ℝ) : Prop := a < 0

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end p_sufficient_not_necessary_for_q_l1785_178513


namespace constant_term_zero_implies_m_zero_l1785_178561

theorem constant_term_zero_implies_m_zero :
  ∀ m : ℝ, (m^2 - m = 0) → (m = 0) :=
by sorry

end constant_term_zero_implies_m_zero_l1785_178561


namespace ratio_bounds_l1785_178582

theorem ratio_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  2 / 3 ≤ b / a ∧ b / a ≤ 3 / 2 := by
  sorry

end ratio_bounds_l1785_178582


namespace sum_of_coefficients_quadratic_l1785_178599

theorem sum_of_coefficients_quadratic (x : ℝ) : 
  (∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ x * (x + 1) = 4) → 
  (∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ x * (x + 1) = 4 ∧ a + b + c = -2) := by
  sorry

end sum_of_coefficients_quadratic_l1785_178599


namespace yoongi_multiplication_l1785_178575

theorem yoongi_multiplication (n : ℚ) : n * 15 = 45 → n - 1 = 2 := by
  sorry

end yoongi_multiplication_l1785_178575


namespace cheapest_caterer_l1785_178564

def first_caterer_cost (people : ℕ) : ℚ := 120 + 18 * people
def second_caterer_cost (people : ℕ) : ℚ := 250 + 15 * people

theorem cheapest_caterer (people : ℕ) :
  (people ≥ 44 → second_caterer_cost people ≤ first_caterer_cost people) ∧
  (people < 44 → second_caterer_cost people > first_caterer_cost people) :=
sorry

end cheapest_caterer_l1785_178564


namespace max_sum_problem_l1785_178524

def is_valid_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

theorem max_sum_problem (A B C D : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_valid : is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C ∧ is_valid_digit D)
  (h_integer : ∃ k : ℕ, k * (C + D) = A + B + 1)
  (h_max : ∀ A' B' C' D' : ℕ, 
    A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' →
    is_valid_digit A' ∧ is_valid_digit B' ∧ is_valid_digit C' ∧ is_valid_digit D' →
    (∃ k' : ℕ, k' * (C' + D') = A' + B' + 1) →
    (A' + B' + 1) / (C' + D') ≤ (A + B + 1) / (C + D)) :
  A + B + 1 = 18 := by
sorry

end max_sum_problem_l1785_178524


namespace fib_F15_units_digit_l1785_178594

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The period of the units digit in the Fibonacci sequence -/
def fib_units_period : ℕ := 60

/-- Theorem: The units digit of F_{F_15} is 5 -/
theorem fib_F15_units_digit : fib (fib 15) % 10 = 5 := by
  sorry

end fib_F15_units_digit_l1785_178594


namespace modular_inverse_7_mod_29_l1785_178529

theorem modular_inverse_7_mod_29 :
  ∃ x : ℕ, x < 29 ∧ (7 * x) % 29 = 1 ∧ x = 25 := by
  sorry

end modular_inverse_7_mod_29_l1785_178529


namespace number_sum_problem_l1785_178554

theorem number_sum_problem (x : ℝ) (h : 20 + x = 30) : x = 10 := by
  sorry

end number_sum_problem_l1785_178554


namespace min_occupied_seats_l1785_178579

/-- Represents a row of seats -/
structure SeatRow :=
  (total : ℕ)
  (occupied : Finset ℕ)
  (h_occupied : occupied.card ≤ total)

/-- Predicts if a new person must sit next to someone -/
def mustSitNext (row : SeatRow) : Prop :=
  ∀ n : ℕ, n ≤ row.total → n ∉ row.occupied →
    (n > 1 ∧ n - 1 ∈ row.occupied) ∨ (n < row.total ∧ n + 1 ∈ row.occupied)

/-- The theorem to be proved -/
theorem min_occupied_seats :
  ∃ (row : SeatRow),
    row.total = 120 ∧
    row.occupied.card = 40 ∧
    mustSitNext row ∧
    ∀ (row' : SeatRow),
      row'.total = 120 →
      row'.occupied.card < 40 →
      ¬mustSitNext row' :=
by sorry

end min_occupied_seats_l1785_178579


namespace sum_of_symmetric_points_coords_l1785_178573

/-- Two points P₁ and P₂ are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are the same. -/
def symmetric_wrt_y_axis (P₁ P₂ : ℝ × ℝ) : Prop :=
  P₁.1 = -P₂.1 ∧ P₁.2 = P₂.2

/-- Given two points P₁(a,-5) and P₂(3,b) that are symmetric with respect to the y-axis,
    prove that a + b = -8. -/
theorem sum_of_symmetric_points_coords (a b : ℝ) 
    (h : symmetric_wrt_y_axis (a, -5) (3, b)) : 
  a + b = -8 := by
  sorry

end sum_of_symmetric_points_coords_l1785_178573


namespace probability_of_at_least_one_of_each_color_l1785_178581

-- Define the number of marbles of each color
def red_marbles : Nat := 3
def blue_marbles : Nat := 3
def green_marbles : Nat := 3

-- Define the total number of marbles
def total_marbles : Nat := red_marbles + blue_marbles + green_marbles

-- Define the number of marbles to be selected
def selected_marbles : Nat := 4

-- Define the probability of selecting at least one marble of each color
def prob_at_least_one_of_each : Rat := 9/14

-- Theorem statement
theorem probability_of_at_least_one_of_each_color :
  prob_at_least_one_of_each = 
    (Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 2 +
     Nat.choose red_marbles 1 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 1 +
     Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1) /
    Nat.choose total_marbles selected_marbles := by
  sorry

end probability_of_at_least_one_of_each_color_l1785_178581


namespace gas_station_candy_boxes_l1785_178562

/-- Given a gas station that sold 2 boxes of chocolate candy, 5 boxes of sugar candy,
    and some boxes of gum, with a total of 9 boxes sold, prove that 2 boxes of gum were sold. -/
theorem gas_station_candy_boxes : 
  let chocolate_boxes : ℕ := 2
  let sugar_boxes : ℕ := 5
  let total_boxes : ℕ := 9
  let gum_boxes : ℕ := total_boxes - chocolate_boxes - sugar_boxes
  gum_boxes = 2 := by sorry

end gas_station_candy_boxes_l1785_178562


namespace cone_volume_from_circle_sector_l1785_178597

/-- The volume of a right circular cone formed by rolling up a five-sixth sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 5 / 6
  let base_radius : ℝ := sector_fraction * r
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let volume : ℝ := (1 / 3) * Real.pi * base_radius^2 * height
  volume = (25 / 3) * Real.pi * Real.sqrt 11 := by
  sorry


end cone_volume_from_circle_sector_l1785_178597


namespace regular_tetrahedron_vertices_and_edges_l1785_178547

/-- A regular tetrahedron is a regular triangular pyramid -/
structure RegularTetrahedron where
  is_regular_triangular_pyramid : Bool

/-- The number of vertices in a regular tetrahedron -/
def num_vertices (t : RegularTetrahedron) : ℕ := 4

/-- The number of edges in a regular tetrahedron -/
def num_edges (t : RegularTetrahedron) : ℕ := 6

/-- Theorem stating that a regular tetrahedron has 4 vertices and 6 edges -/
theorem regular_tetrahedron_vertices_and_edges (t : RegularTetrahedron) :
  num_vertices t = 4 ∧ num_edges t = 6 := by
  sorry

end regular_tetrahedron_vertices_and_edges_l1785_178547


namespace max_weight_theorem_l1785_178537

def weight_set : Set ℕ := {2, 5, 10}

def is_measurable (w : ℕ) : Prop :=
  ∃ (a b c : ℕ), w = 2*a + 5*b + 10*c

def max_measurable : ℕ := 17

theorem max_weight_theorem :
  (∀ w : ℕ, is_measurable w → w ≤ max_measurable) ∧
  is_measurable max_measurable :=
sorry

end max_weight_theorem_l1785_178537


namespace three_people_on_third_stop_l1785_178518

/-- Represents the number of people on a bus and its changes at stops -/
structure BusRide where
  initial : ℕ
  first_off : ℕ
  second_off : ℕ
  second_on : ℕ
  third_off : ℕ
  final : ℕ

/-- Calculates the number of people who got on at the third stop -/
def people_on_third_stop (ride : BusRide) : ℕ :=
  ride.final - (ride.initial - ride.first_off - ride.second_off + ride.second_on - ride.third_off)

/-- Theorem stating that 3 people got on at the third stop -/
theorem three_people_on_third_stop (ride : BusRide) 
  (h_initial : ride.initial = 50)
  (h_first_off : ride.first_off = 15)
  (h_second_off : ride.second_off = 8)
  (h_second_on : ride.second_on = 2)
  (h_third_off : ride.third_off = 4)
  (h_final : ride.final = 28) :
  people_on_third_stop ride = 3 := by
  sorry

#eval people_on_third_stop { initial := 50, first_off := 15, second_off := 8, second_on := 2, third_off := 4, final := 28 }

end three_people_on_third_stop_l1785_178518


namespace other_root_of_quadratic_l1785_178557

/-- Given a quadratic equation x^2 + kx - 2 = 0 where x = 1 is one root,
    prove that x = -2 is the other root. -/
theorem other_root_of_quadratic (k : ℝ) : 
  (1 : ℝ)^2 + k * 1 - 2 = 0 → -2^2 + k * (-2) - 2 = 0 := by
  sorry

end other_root_of_quadratic_l1785_178557


namespace square_difference_l1785_178553

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end square_difference_l1785_178553


namespace triangle_rotation_path_length_l1785_178590

/-- The path length of a vertex of an equilateral triangle rotating around a square --/
theorem triangle_rotation_path_length 
  (square_side : ℝ) 
  (triangle_side : ℝ) 
  (h_square : square_side = 6) 
  (h_triangle : triangle_side = 3) : 
  let path_length := 4 * 3 * (2 * π * triangle_side / 3)
  path_length = 24 * π := by sorry

end triangle_rotation_path_length_l1785_178590


namespace inequality_solution_l1785_178535

def inequality (x : ℝ) : Prop :=
  1 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 1 / (x - 4) < 1 / 24

theorem inequality_solution (x : ℝ) :
  inequality x → (x > -7 ∧ x < 1) ∨ (x > 3 ∧ x < 4) := by
  sorry

end inequality_solution_l1785_178535


namespace village_population_l1785_178598

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  percentage = 60 / 100 →
  partial_population = 23040 →
  (percentage : ℚ) * total_population = partial_population →
  total_population = 38400 := by
sorry

end village_population_l1785_178598


namespace quadratic_maximum_l1785_178585

-- Define the quadratic function
def quadratic (p r s x : ℝ) : ℝ := x^2 + p*x + r + s

-- State the theorem
theorem quadratic_maximum (p s : ℝ) :
  let r : ℝ := 10 - s + p^2/4
  (∀ x, quadratic p r s x ≤ 10) ∧ 
  (quadratic p r s (-p/2) = 10) :=
by sorry

end quadratic_maximum_l1785_178585


namespace reflected_ray_equation_l1785_178552

/-- The equation of a reflected ray given an incident ray and a reflecting line. -/
theorem reflected_ray_equation (x y : ℝ) :
  (y = 2 * x + 1) →  -- incident ray
  (y = x) →          -- reflecting line
  (x - 2 * y - 1 = 0) -- reflected ray
  := by sorry

end reflected_ray_equation_l1785_178552


namespace runner_speed_l1785_178567

/-- Given a runner who runs 5 days a week, 1.5 hours each day, and covers 60 miles in a week,
    prove that their running speed is 8 mph. -/
theorem runner_speed (days_per_week : ℕ) (hours_per_day : ℝ) (miles_per_week : ℝ) :
  days_per_week = 5 →
  hours_per_day = 1.5 →
  miles_per_week = 60 →
  miles_per_week / (days_per_week * hours_per_day) = 8 := by
  sorry

end runner_speed_l1785_178567


namespace tangent_slope_at_one_l1785_178543

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_one :
  (f' 1) = 1 :=
sorry

end tangent_slope_at_one_l1785_178543


namespace max_stores_visited_l1785_178533

theorem max_stores_visited (
  num_stores : ℕ) 
  (total_visits : ℕ) 
  (num_shoppers : ℕ) 
  (num_two_store_visitors : ℕ) 
  (h1 : num_stores = 8)
  (h2 : total_visits = 21)
  (h3 : num_shoppers = 12)
  (h4 : num_two_store_visitors = 8)
  (h5 : num_two_store_visitors ≤ num_shoppers)
  (h6 : ∀ n : ℕ, n ≤ num_shoppers → n > 0) :
  ∃ max_visits : ℕ, max_visits = 3 ∧ 
  ∀ n : ℕ, n ≤ num_shoppers → ∃ k : ℕ, k ≤ max_visits ∧ 
  (num_two_store_visitors * 2 + (num_shoppers - num_two_store_visitors) * k = total_visits) :=
by sorry

end max_stores_visited_l1785_178533


namespace tower_combinations_l1785_178506

def red_cubes : ℕ := 2
def blue_cubes : ℕ := 4
def green_cubes : ℕ := 5
def tower_height : ℕ := 7

/-- The number of different towers with a height of 7 cubes that can be built
    with 2 red cubes, 4 blue cubes, and 5 green cubes. -/
def number_of_towers : ℕ := 420

theorem tower_combinations :
  (red_cubes + blue_cubes + green_cubes - tower_height = 2) →
  number_of_towers = 420 :=
by sorry

end tower_combinations_l1785_178506


namespace two_digit_multiple_of_eight_l1785_178504

theorem two_digit_multiple_of_eight (A : Nat) : 
  (30 ≤ 10 * 3 + A) ∧ (10 * 3 + A < 40) ∧ (10 * 3 + A) % 8 = 0 → A = 2 := by
  sorry

end two_digit_multiple_of_eight_l1785_178504


namespace mark_friends_percentage_l1785_178544

/-- Calculates the percentage of friends kept initially -/
def friendsKeptPercentage (initialFriends : ℕ) (finalFriends : ℕ) (responseRate : ℚ) : ℚ :=
  let keptPercentage : ℚ := (2 * finalFriends - initialFriends : ℚ) / initialFriends
  keptPercentage * 100

/-- Proves that the percentage of friends Mark kept initially is 40% -/
theorem mark_friends_percentage :
  friendsKeptPercentage 100 70 (1/2) = 40 := by
  sorry

#eval friendsKeptPercentage 100 70 (1/2)

end mark_friends_percentage_l1785_178544


namespace tangent_length_is_three_l1785_178558

/-- The equation of a circle in the xy-plane -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + 12 = 0

/-- The point P -/
def P : ℝ × ℝ := (-1, 4)

/-- The length of the tangent line from a point to a circle -/
noncomputable def tangentLength (p : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem: The length of the tangent line from P to the circle is 3 -/
theorem tangent_length_is_three :
  tangentLength P = 3 := by sorry

end tangent_length_is_three_l1785_178558


namespace vector_equation_solution_l1785_178559

/-- Given four distinct points P, A, B, C on a plane, prove that if 
    PA + PB + PC = 0 and AB + AC + m * AP = 0, then m = -3 -/
theorem vector_equation_solution (P A B C : EuclideanSpace ℝ (Fin 2)) 
    (h1 : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
    (h2 : (A - P) + (B - P) + (C - P) = 0)
    (h3 : ∃ m : ℝ, (B - A) + (C - A) + m • (P - A) = 0) : 
  ∃ m : ℝ, (B - A) + (C - A) + m • (P - A) = 0 ∧ m = -3 := by
  sorry

end vector_equation_solution_l1785_178559


namespace james_and_louise_ages_l1785_178527

theorem james_and_louise_ages :
  ∀ (james louise : ℕ),
  james = louise + 7 →
  james + 10 = 3 * (louise - 3) →
  james + louise = 33 :=
by
  sorry

end james_and_louise_ages_l1785_178527


namespace profit_percentage_l1785_178556

/-- If selling an article at 2/3 of a certain price results in a 15% loss,
    then selling at the full certain price results in a 27.5% profit. -/
theorem profit_percentage (certain_price : ℝ) (cost_price : ℝ) :
  certain_price > 0 →
  cost_price > 0 →
  (2 / 3 : ℝ) * certain_price = 0.85 * cost_price →
  (certain_price - cost_price) / cost_price = 0.275 :=
by sorry

end profit_percentage_l1785_178556


namespace log_equation_proof_l1785_178511

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem log_equation_proof :
  (ln 5) ^ 2 + ln 2 * ln 50 = 1 := by sorry

end log_equation_proof_l1785_178511


namespace f_range_l1785_178545

def f (x : ℕ) : ℤ := Int.floor ((x + 1) / 2 : ℚ) - Int.floor (x / 2 : ℚ)

theorem f_range : ∀ x : ℕ, f x = 0 ∨ f x = 1 ∧ ∃ a b : ℕ, f a = 0 ∧ f b = 1 := by
  sorry

end f_range_l1785_178545


namespace difference_of_squares_l1785_178586

theorem difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end difference_of_squares_l1785_178586


namespace ma_xiaotiao_rank_l1785_178571

theorem ma_xiaotiao_rank (total_participants : ℕ) (ma_rank : ℕ) : 
  total_participants = 34 →
  ma_rank > 0 →
  ma_rank ≤ total_participants →
  total_participants - ma_rank = 2 * (ma_rank - 1) →
  ma_rank = 12 := by
  sorry

end ma_xiaotiao_rank_l1785_178571


namespace candy_sampling_percentage_l1785_178502

theorem candy_sampling_percentage : 
  ∀ (total_customers : ℝ) (caught_percent : ℝ) (not_caught_percent : ℝ),
  caught_percent = 22 →
  not_caught_percent = 12 →
  ∃ (total_sampling_percent : ℝ),
    total_sampling_percent = caught_percent + (not_caught_percent / 100) * total_sampling_percent ∧
    total_sampling_percent = 25 := by
  sorry

end candy_sampling_percentage_l1785_178502


namespace limit_of_rational_function_l1785_178500

theorem limit_of_rational_function (f : ℝ → ℝ) (h : ∀ x ≠ 1, f x = (x^4 - 1) / (2*x^4 - x^2 - 1)) :
  ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |f x - 2/3| < ε :=
sorry

end limit_of_rational_function_l1785_178500


namespace product_representation_l1785_178536

theorem product_representation (a : ℝ) (p : ℕ+) 
  (h1 : 12345 * 6789 = a * (10 : ℝ)^(p : ℝ))
  (h2 : 1 ≤ a ∧ a < 10) :
  p = 7 := by
  sorry

end product_representation_l1785_178536


namespace fraction_inequality_solution_set_l1785_178531

theorem fraction_inequality_solution_set (x : ℝ) :
  (x - 2) / (x - 1) > 0 ↔ x < 1 ∨ x > 2 :=
by sorry

end fraction_inequality_solution_set_l1785_178531


namespace evaluate_expression_l1785_178542

theorem evaluate_expression : 
  (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 := by
  sorry

end evaluate_expression_l1785_178542


namespace least_reducible_fraction_l1785_178538

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 27) ∧ k ∣ (7 * m + 4))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 27) ∧ k ∣ (7 * n + 4)) ∧
  n = 220 :=
sorry

end least_reducible_fraction_l1785_178538


namespace gcf_of_60_and_90_l1785_178508

theorem gcf_of_60_and_90 : Nat.gcd 60 90 = 30 := by
  sorry

end gcf_of_60_and_90_l1785_178508


namespace spherical_segment_angle_l1785_178517

theorem spherical_segment_angle (r : ℝ) (α : ℝ) (h : r > 0) :
  (2 * π * r * (r * (1 - Real.cos (α / 2))) + π * (r * Real.sin (α / 2))^2 = π * r^2) →
  (Real.cos (α / 2))^2 + 2 * Real.cos (α / 2) - 2 = 0 :=
by sorry

end spherical_segment_angle_l1785_178517


namespace routes_in_3x3_grid_l1785_178519

/-- The number of different routes in a 3x3 grid from top-left to bottom-right -/
def numRoutes : ℕ := 20

/-- The size of the grid -/
def gridSize : ℕ := 3

/-- The total number of moves required to reach the destination -/
def totalMoves : ℕ := gridSize * 2

/-- The number of moves in one direction (either right or down) -/
def movesInOneDirection : ℕ := gridSize

theorem routes_in_3x3_grid :
  numRoutes = Nat.choose totalMoves movesInOneDirection := by sorry

end routes_in_3x3_grid_l1785_178519


namespace fourth_term_is_negative_24_l1785_178588

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n-1)

-- Define the conditions of our specific sequence
def sequence_conditions (x : ℝ) : Prop :=
  ∃ (r : ℝ), 
    geometric_sequence x r 2 = 3*x + 3 ∧
    geometric_sequence x r 3 = 6*x + 6

-- Theorem statement
theorem fourth_term_is_negative_24 :
  ∀ x : ℝ, sequence_conditions x → geometric_sequence x 2 4 = -24 :=
by sorry

end fourth_term_is_negative_24_l1785_178588


namespace calculation_proof_l1785_178510

theorem calculation_proof : 5^2 * 7 + 9 * 4 - 35 / 5 = 204 := by
  sorry

end calculation_proof_l1785_178510


namespace simplify_nested_roots_l1785_178576

theorem simplify_nested_roots (x : ℝ) :
  (((x^16)^(1/8))^(1/4))^3 * (((x^16)^(1/4))^(1/8))^5 = x^4 := by
  sorry

end simplify_nested_roots_l1785_178576


namespace arithmetic_sequence_range_of_d_l1785_178569

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_range_of_d (d : ℝ) :
  (arithmetic_sequence 24 d 9 ≥ 0 ∧ arithmetic_sequence 24 d 10 < 0) →
  -3 ≤ d ∧ d < -8/3 :=
sorry

end arithmetic_sequence_range_of_d_l1785_178569


namespace range_of_k_l1785_178522

def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}
def B (k : ℝ) : Set ℝ := {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

theorem range_of_k (k : ℝ) : A ⊇ B k ↔ -1 ≤ k ∧ k ≤ 1/2 := by
  sorry

end range_of_k_l1785_178522


namespace max_m_value_min_weighted_sum_of_squares_l1785_178593

-- Part 1
theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - m| ≥ 2*m) → m ≤ 1 :=
sorry

-- Part 2
theorem min_weighted_sum_of_squares :
  let f (a b c : ℝ) := 4*a^2 + 9*b^2 + c^2
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    f a b c ≥ 36/49 ∧
    (f a b c = 36/49 ↔ a = 9/49 ∧ b = 4/49 ∧ c = 36/49) :=
sorry

end max_m_value_min_weighted_sum_of_squares_l1785_178593


namespace movie_ticket_cost_l1785_178555

theorem movie_ticket_cost (x : ℝ) : 
  (2 * x + 3 * (x - 2) = 39) →
  x = 9 := by sorry

end movie_ticket_cost_l1785_178555


namespace halfway_between_fractions_l1785_178501

theorem halfway_between_fractions : 
  let a := (1 : ℚ) / 6
  let b := (1 : ℚ) / 12
  let midpoint := (a + b) / 2
  midpoint = (1 : ℚ) / 8 := by sorry

end halfway_between_fractions_l1785_178501


namespace sqrt_fraction_difference_l1785_178577

theorem sqrt_fraction_difference : Real.sqrt (9/4) - Real.sqrt (4/9) = 5/6 := by
  sorry

end sqrt_fraction_difference_l1785_178577


namespace largest_good_set_size_l1785_178523

/-- A set of positive integers is "good" if there exists a coloring with 2008 colors
    of all positive integers such that no number in the set is the sum of two
    different positive integers of the same color. -/
def isGoodSet (S : Set ℕ) : Prop :=
  ∃ (f : ℕ → Fin 2008), ∀ n ∈ S, ∀ x y : ℕ, x ≠ y → f x = f y → n ≠ x + y

/-- The set S(a, t) = {a+1, a+2, ..., a+t} for a positive integer a and natural number t. -/
def S (a t : ℕ) : Set ℕ := {n : ℕ | a + 1 ≤ n ∧ n ≤ a + t}

/-- The largest value of t for which S(a, t) is "good" for any positive integer a is 4014. -/
theorem largest_good_set_size :
  (∀ a : ℕ, a > 0 → isGoodSet (S a 4014)) ∧
  (∀ t : ℕ, t > 4014 → ∃ a : ℕ, a > 0 ∧ ¬isGoodSet (S a t)) :=
sorry

end largest_good_set_size_l1785_178523


namespace consecutive_arithmetic_geometric_equality_l1785_178580

theorem consecutive_arithmetic_geometric_equality (a b c : ℝ) : 
  (∃ r : ℝ, b - a = r ∧ c - b = r) →  -- arithmetic progression condition
  (∃ q : ℝ, b / a = q ∧ c / b = q) →  -- geometric progression condition
  a = b ∧ b = c := by
sorry

end consecutive_arithmetic_geometric_equality_l1785_178580


namespace river_speed_theorem_l1785_178515

/-- Represents the equation for a ship traveling upstream and downstream -/
def river_equation (s v d1 d2 : ℝ) : Prop :=
  d1 / (s + v) = d2 / (s - v)

/-- Theorem stating that the river equation holds for the given conditions -/
theorem river_speed_theorem (s v d1 d2 : ℝ) 
  (h_s : s > 0)
  (h_v : 0 < v ∧ v < s)
  (h_d1 : d1 > 0)
  (h_d2 : d2 > 0)
  (h_s_still : s = 30)
  (h_d1 : d1 = 144)
  (h_d2 : d2 = 96) :
  river_equation s v d1 d2 :=
sorry

end river_speed_theorem_l1785_178515


namespace number_of_cows_l1785_178507

-- Define the types for animals
inductive Animal : Type
| Cow : Animal
| Chicken : Animal
| Pig : Animal

-- Define the farm
def Farm : Type := Animal → ℕ

-- Define the number of legs for each animal
def legs : Animal → ℕ
| Animal.Cow => 4
| Animal.Chicken => 2
| Animal.Pig => 4

-- Define the total number of animals
def total_animals (farm : Farm) : ℕ :=
  farm Animal.Cow + farm Animal.Chicken + farm Animal.Pig

-- Define the total number of legs
def total_legs (farm : Farm) : ℕ :=
  farm Animal.Cow * legs Animal.Cow +
  farm Animal.Chicken * legs Animal.Chicken +
  farm Animal.Pig * legs Animal.Pig

-- State the theorem
theorem number_of_cows (farm : Farm) : 
  farm Animal.Chicken = 6 ∧ 
  total_legs farm = 20 + 2 * total_animals farm → 
  farm Animal.Cow = 6 :=
sorry

end number_of_cows_l1785_178507


namespace sum_of_interchanged_digits_divisible_by_11_l1785_178546

theorem sum_of_interchanged_digits_divisible_by_11 (a b : ℕ) 
  (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0) : 
  ∃ k : ℕ, (10 * a + b) + (10 * b + a) = 11 * k := by
  sorry

end sum_of_interchanged_digits_divisible_by_11_l1785_178546
