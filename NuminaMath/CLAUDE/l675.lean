import Mathlib

namespace NUMINAMATH_CALUDE_half_angle_quadrant_l675_67593

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3/2) * Real.pi

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  (∃ k : Int, k * Real.pi + Real.pi/2 < α ∧ α < k * Real.pi + Real.pi) ∨
  (∃ k : Int, k * Real.pi + (3/2) * Real.pi < α ∧ α < (k + 1) * Real.pi)

theorem half_angle_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_or_fourth_quadrant (α/2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l675_67593


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l675_67598

theorem sock_pair_combinations (black green red : ℕ) 
  (h_black : black = 5) 
  (h_green : green = 3) 
  (h_red : red = 4) : 
  black * green + black * red + green * red = 47 := by
sorry

end NUMINAMATH_CALUDE_sock_pair_combinations_l675_67598


namespace NUMINAMATH_CALUDE_terms_before_negative_23_l675_67542

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem terms_before_negative_23 :
  let a₁ := 101
  let d := -4
  ∃ n : ℕ, 
    (arithmetic_sequence a₁ d n = -23) ∧ 
    (∀ k : ℕ, k < n → arithmetic_sequence a₁ d k > -23) ∧
    n - 1 = 31 :=
by sorry

end NUMINAMATH_CALUDE_terms_before_negative_23_l675_67542


namespace NUMINAMATH_CALUDE_arithmetic_relations_l675_67545

theorem arithmetic_relations : 
  (10 * 100 = 1000) ∧ 
  (10 * 1000 = 10000) ∧ 
  (10000 / 100 = 100) ∧ 
  (1000 / 10 = 100) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_relations_l675_67545


namespace NUMINAMATH_CALUDE_exists_islands_with_inverse_area_relation_l675_67507

/-- Represents a rectangular island with length and width in kilometers. -/
structure Island where
  length : ℝ
  width : ℝ

/-- Calculates the area of an island in square kilometers. -/
def islandArea (i : Island) : ℝ :=
  i.length * i.width

/-- Calculates the coastal water area of an island in square kilometers. 
    Coastal water is defined as the area within 50 km of the shore. -/
def coastalWaterArea (i : Island) : ℝ :=
  (i.length + 100) * (i.width + 100) - islandArea i

/-- Theorem stating that there exist two islands where the first has smaller area
    but larger coastal water area compared to the second. -/
theorem exists_islands_with_inverse_area_relation : 
  ∃ (i1 i2 : Island), 
    islandArea i1 < islandArea i2 ∧ 
    coastalWaterArea i1 > coastalWaterArea i2 :=
sorry

end NUMINAMATH_CALUDE_exists_islands_with_inverse_area_relation_l675_67507


namespace NUMINAMATH_CALUDE_jacket_dimes_count_l675_67582

/-- The value of a single dime in dollars -/
def dime_value : ℚ := 1 / 10

/-- The total amount of money found in dollars -/
def total_money : ℚ := 19 / 10

/-- The number of dimes found in the shorts -/
def dimes_in_shorts : ℕ := 4

/-- The number of dimes found in the jacket -/
def dimes_in_jacket : ℕ := 15

theorem jacket_dimes_count :
  dimes_in_jacket * dime_value + dimes_in_shorts * dime_value = total_money :=
by sorry

end NUMINAMATH_CALUDE_jacket_dimes_count_l675_67582


namespace NUMINAMATH_CALUDE_daughter_weight_l675_67519

/-- Represents the weights of family members in kilograms -/
structure FamilyWeights where
  mother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- Conditions for the family weights problem -/
def FamilyWeightsProblem (w : FamilyWeights) : Prop :=
  w.mother + w.daughter + w.grandchild = 150 ∧
  w.daughter + w.grandchild = 60 ∧
  w.grandchild = (1 / 5) * w.mother

/-- The weight of the daughter is 42 kg given the conditions -/
theorem daughter_weight (w : FamilyWeights) 
  (h : FamilyWeightsProblem w) : w.daughter = 42 := by
  sorry

end NUMINAMATH_CALUDE_daughter_weight_l675_67519


namespace NUMINAMATH_CALUDE_athletes_count_is_ten_l675_67585

-- Define the types for our counts
def TotalLegs : ℕ := 108
def TotalHeads : ℕ := 32

-- Define a structure to represent the counts of each animal type
structure AnimalCounts where
  athletes : ℕ
  elephants : ℕ
  monkeys : ℕ

-- Define the property that the counts satisfy the given conditions
def satisfiesConditions (counts : AnimalCounts) : Prop :=
  2 * counts.athletes + 4 * counts.elephants + 2 * counts.monkeys = TotalLegs ∧
  counts.athletes + counts.elephants + counts.monkeys = TotalHeads

-- The theorem to prove
theorem athletes_count_is_ten :
  ∃ (counts : AnimalCounts), satisfiesConditions counts ∧ counts.athletes = 10 :=
by sorry

end NUMINAMATH_CALUDE_athletes_count_is_ten_l675_67585


namespace NUMINAMATH_CALUDE_range_of_m_l675_67508

def p (m : ℝ) : Prop := ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 - 2 * x + 1 = 0

theorem range_of_m (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬¬(q m)) : m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l675_67508


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l675_67532

theorem largest_stamps_per_page (stamps_book1 stamps_book2 : ℕ) 
  (h1 : stamps_book1 = 960) 
  (h2 : stamps_book2 = 1200) 
  (h3 : stamps_book1 > 0) 
  (h4 : stamps_book2 > 0) : 
  ∃ (stamps_per_page : ℕ), 
    stamps_per_page > 0 ∧ 
    stamps_book1 % stamps_per_page = 0 ∧ 
    stamps_book2 % stamps_per_page = 0 ∧ 
    stamps_book1 / stamps_per_page ≥ 2 ∧ 
    stamps_book2 / stamps_per_page ≥ 2 ∧ 
    ∀ (n : ℕ), n > stamps_per_page → 
      (stamps_book1 % n ≠ 0 ∨ 
       stamps_book2 % n ≠ 0 ∨ 
       stamps_book1 / n < 2 ∨ 
       stamps_book2 / n < 2) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l675_67532


namespace NUMINAMATH_CALUDE_equation_solutions_l675_67556

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.log (x^2 - 5*x + 10) = 2

-- State the theorem
theorem equation_solutions :
  ∃ (x₁ x₂ : ℝ), equation x₁ ∧ equation x₂ ∧ 
  (abs (x₁ - 4.4) < 0.01) ∧ (abs (x₂ - 0.6) < 0.01) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l675_67556


namespace NUMINAMATH_CALUDE_matches_per_box_l675_67570

/-- Given 5 dozen boxes containing a total of 1200 matches, prove that each box contains 20 matches. -/
theorem matches_per_box (dozen_boxes : ℕ) (total_matches : ℕ) : 
  dozen_boxes = 5 → total_matches = 1200 → (dozen_boxes * 12) * 20 = total_matches := by
  sorry

end NUMINAMATH_CALUDE_matches_per_box_l675_67570


namespace NUMINAMATH_CALUDE_initial_bunnies_l675_67528

theorem initial_bunnies (initial : ℕ) : 
  (3 / 5 : ℚ) * initial + 2 * ((3 / 5 : ℚ) * initial) = 54 → initial = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_bunnies_l675_67528


namespace NUMINAMATH_CALUDE_sandra_savings_l675_67522

-- Define the number of notepads
def num_notepads : ℕ := 8

-- Define the original price per notepad
def original_price : ℚ := 375 / 100

-- Define the discount rate
def discount_rate : ℚ := 25 / 100

-- Define the savings calculation
def savings : ℚ :=
  num_notepads * original_price - num_notepads * (original_price * (1 - discount_rate))

-- Theorem to prove
theorem sandra_savings : savings = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sandra_savings_l675_67522


namespace NUMINAMATH_CALUDE_expressions_not_equivalent_l675_67547

theorem expressions_not_equivalent :
  ∃ x : ℝ, (x^2 + 1 ≠ 0 ∧ x^2 + 2*x + 1 ≠ 0) →
    (x^2 + x + 1) / (x^2 + 1) ≠ ((x + 1)^2) / (x^2 + 2*x + 1) :=
by sorry

end NUMINAMATH_CALUDE_expressions_not_equivalent_l675_67547


namespace NUMINAMATH_CALUDE_max_utility_problem_l675_67584

theorem max_utility_problem (s : ℝ) : 
  s ≥ 0 ∧ s * (10 - s) = (3 - s) * (s + 4) → s = 0 := by
  sorry

end NUMINAMATH_CALUDE_max_utility_problem_l675_67584


namespace NUMINAMATH_CALUDE_triangle_property_l675_67575

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) ∧
  ((c - 2*a) * Real.cos B + b * Real.cos C = 0) →
  (B = π/3) ∧
  (a + b + c = 6 ∧ b = 2 → 
    1/2 * a * c * Real.sin B = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l675_67575


namespace NUMINAMATH_CALUDE_statue_weight_calculation_l675_67576

/-- The weight of a marble statue after three weeks of carving --/
def final_statue_weight (initial_weight : ℝ) (cut_week1 : ℝ) (cut_week2 : ℝ) (cut_week3 : ℝ) : ℝ :=
  initial_weight * (1 - cut_week1) * (1 - cut_week2) * (1 - cut_week3)

/-- Theorem stating the final weight of the statue --/
theorem statue_weight_calculation :
  let initial_weight : ℝ := 180
  let cut_week1 : ℝ := 0.28
  let cut_week2 : ℝ := 0.18
  let cut_week3 : ℝ := 0.20
  final_statue_weight initial_weight cut_week1 cut_week2 cut_week3 = 85.0176 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_calculation_l675_67576


namespace NUMINAMATH_CALUDE_runner_speed_increase_l675_67533

theorem runner_speed_increase (v : ℝ) (h : v > 0) : 
  (v + 2) / v = 2.5 → (v + 4) / v = 4 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_increase_l675_67533


namespace NUMINAMATH_CALUDE_equation_solutions_l675_67501

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0) ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ y₁ y₂ : ℝ, (3*y₁^2 + 2*y₁ - 1 = 0 ∧ 3*y₂^2 + 2*y₂ - 1 = 0) ∧ y₁ = 1/3 ∧ y₂ = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l675_67501


namespace NUMINAMATH_CALUDE_insurance_cost_decade_l675_67504

/-- Benjamin's yearly car insurance cost in dollars -/
def yearly_cost : ℕ := 3000

/-- Number of years in a decade -/
def decade : ℕ := 10

/-- Theorem: Benjamin's car insurance cost over a decade -/
theorem insurance_cost_decade : yearly_cost * decade = 30000 := by
  sorry

end NUMINAMATH_CALUDE_insurance_cost_decade_l675_67504


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l675_67525

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric progression -/
def geometric_prog (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a →
  geometric_prog (a 1) (a 2) (a 5) →
  a 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l675_67525


namespace NUMINAMATH_CALUDE_four_diamonds_balance_four_bullets_l675_67561

/-- Represents the balance of symbols in a weighing system -/
structure SymbolBalance where
  delta : ℚ      -- Represents Δ
  diamond : ℚ    -- Represents ♢
  bullet : ℚ     -- Represents •

/-- The balance equations given in the problem -/
def balance_equations (sb : SymbolBalance) : Prop :=
  (2 * sb.delta + 3 * sb.diamond = 12 * sb.bullet) ∧
  (sb.delta = 3 * sb.diamond + 2 * sb.bullet)

/-- The theorem to be proved -/
theorem four_diamonds_balance_four_bullets (sb : SymbolBalance) :
  balance_equations sb → 4 * sb.diamond = 4 * sb.bullet :=
by sorry

end NUMINAMATH_CALUDE_four_diamonds_balance_four_bullets_l675_67561


namespace NUMINAMATH_CALUDE_airplane_seats_l675_67520

/-- Represents the total number of seats on an airplane -/
def total_seats : ℕ := 216

/-- Represents the number of seats in First Class -/
def first_class_seats : ℕ := 36

/-- Theorem stating that the total number of seats on the airplane is 216 -/
theorem airplane_seats :
  (first_class_seats : ℚ) + (1 : ℚ) / 3 * total_seats + (1 : ℚ) / 2 * total_seats = total_seats :=
by sorry

end NUMINAMATH_CALUDE_airplane_seats_l675_67520


namespace NUMINAMATH_CALUDE_abs_negative_2035_l675_67568

theorem abs_negative_2035 : |(-2035 : ℤ)| = 2035 := by sorry

end NUMINAMATH_CALUDE_abs_negative_2035_l675_67568


namespace NUMINAMATH_CALUDE_teacher_selection_probability_l675_67592

/-- Represents a university department -/
structure Department where
  name : String
  teachers : ℕ

/-- Represents a university -/
structure University where
  departments : List Department

/-- Calculates the total number of teachers in a university -/
def totalTeachers (u : University) : ℕ :=
  u.departments.map (·.teachers) |>.sum

/-- Calculates the probability of selecting an individual teacher -/
def selectionProbability (u : University) (numSelected : ℕ) : ℚ :=
  numSelected / (totalTeachers u)

/-- Theorem stating the probability of selecting an individual teacher -/
theorem teacher_selection_probability
  (u : University)
  (hDepartments : u.departments = [
    ⟨"A", 10⟩,
    ⟨"B", 20⟩,
    ⟨"C", 30⟩
  ])
  (hNumSelected : numSelected = 6) :
  selectionProbability u numSelected = 1 / 10 := by
  sorry


end NUMINAMATH_CALUDE_teacher_selection_probability_l675_67592


namespace NUMINAMATH_CALUDE_function_decomposition_l675_67596

/-- Given a function φ: ℝ³ → ℝ and two functions f, g: ℝ² → ℝ satisfying certain conditions,
    prove the existence of a function h: ℝ → ℝ with a specific property. -/
theorem function_decomposition
  (φ : ℝ → ℝ → ℝ → ℝ)
  (f g : ℝ → ℝ → ℝ)
  (h1 : ∀ x y z, φ x y z = f (x + y) z)
  (h2 : ∀ x y z, φ x y z = g x (y + z)) :
  ∃ h : ℝ → ℝ, ∀ x y z, φ x y z = h (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l675_67596


namespace NUMINAMATH_CALUDE_triangle_problem_l675_67572

/-- 
Given an acute triangle ABC where a, b, c are sides opposite to angles A, B, C respectively,
if √3a = 2c sin A, a = 2, and the area of triangle ABC is 3√3/2,
then the measure of angle C is π/3 and c = √7.
-/
theorem triangle_problem (a b c A B C : Real) : 
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 → -- acute triangle
  Real.sqrt 3 * a = 2 * c * Real.sin A →
  a = 2 →
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 →
  C = π/3 ∧ c = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l675_67572


namespace NUMINAMATH_CALUDE_wong_valentines_l675_67574

/-- Mrs. Wong's Valentine problem -/
theorem wong_valentines (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 30 → given_away = 8 → remaining = initial - given_away → remaining = 22 := by
  sorry

end NUMINAMATH_CALUDE_wong_valentines_l675_67574


namespace NUMINAMATH_CALUDE_coeff_x_neg_one_proof_l675_67521

/-- The coefficient of x^(-1) in the expansion of (√x - 2/x)^7 -/
def coeff_x_neg_one : ℤ := -280

/-- The binomial coefficient (7 choose 3) -/
def binom_7_3 : ℕ := Nat.choose 7 3

theorem coeff_x_neg_one_proof :
  coeff_x_neg_one = binom_7_3 * (-8) :=
sorry

end NUMINAMATH_CALUDE_coeff_x_neg_one_proof_l675_67521


namespace NUMINAMATH_CALUDE_total_cost_of_clothing_l675_67555

/-- The total cost of a shirt, pants, and shoes given specific pricing conditions -/
theorem total_cost_of_clothing (pants_price : ℝ) : 
  pants_price = 120 →
  let shirt_price := (3/4) * pants_price
  let shoes_price := pants_price + 10
  shirt_price + pants_price + shoes_price = 340 := by
sorry

end NUMINAMATH_CALUDE_total_cost_of_clothing_l675_67555


namespace NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_l675_67562

theorem infinitely_many_primes_3_mod_4 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 4 = 3} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_l675_67562


namespace NUMINAMATH_CALUDE_total_blocks_is_2250_l675_67586

/-- Represents the size of a dog -/
inductive DogSize
  | Small
  | Medium
  | Large

/-- Represents the walking speed of a dog in blocks per 10 minutes -/
def walkingSpeed (size : DogSize) : ℕ :=
  match size with
  | .Small => 3
  | .Medium => 4
  | .Large => 2

/-- Represents the number of dogs of each size -/
def dogCounts : DogSize → ℕ
  | .Small => 10
  | .Medium => 8
  | .Large => 7

/-- The total vacation cost in dollars -/
def vacationCost : ℕ := 1200

/-- The number of family members -/
def familyMembers : ℕ := 5

/-- The total available time in minutes -/
def totalAvailableTime : ℕ := 8 * 60

/-- The break time in minutes -/
def breakTime : ℕ := 30

/-- Calculates the total number of blocks Jules has to walk -/
def totalBlocks : ℕ :=
  let availableTime := totalAvailableTime - breakTime
  let slowestSpeed := walkingSpeed DogSize.Large
  let blocksPerDog := (availableTime / 10) * slowestSpeed
  (dogCounts DogSize.Small + dogCounts DogSize.Medium + dogCounts DogSize.Large) * blocksPerDog

theorem total_blocks_is_2250 : totalBlocks = 2250 := by
  sorry

#eval totalBlocks

end NUMINAMATH_CALUDE_total_blocks_is_2250_l675_67586


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l675_67516

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_difference
  (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 7 - a 3 = 20) :
  a 2008 - a 2000 = 40 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l675_67516


namespace NUMINAMATH_CALUDE_car_mpg_difference_l675_67549

/-- Proves that the difference between highway and city miles per gallon is 9 --/
theorem car_mpg_difference (highway_miles : ℕ) (city_miles : ℕ) (city_mpg : ℕ) :
  highway_miles = 462 →
  city_miles = 336 →
  city_mpg = 24 →
  (highway_miles / (city_miles / city_mpg)) - city_mpg = 9 := by
  sorry

#check car_mpg_difference

end NUMINAMATH_CALUDE_car_mpg_difference_l675_67549


namespace NUMINAMATH_CALUDE_inequalities_not_equivalent_l675_67512

-- Define the two inequalities
def inequality1 (x : ℝ) : Prop := x + 3 - 1 / (x - 1) > -x + 2 - 1 / (x - 1)
def inequality2 (x : ℝ) : Prop := x + 3 > -x + 2

-- Theorem stating that the inequalities are not equivalent
theorem inequalities_not_equivalent : ¬(∀ x : ℝ, inequality1 x ↔ inequality2 x) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_not_equivalent_l675_67512


namespace NUMINAMATH_CALUDE_all_graphs_different_l675_67506

-- Define the equations
def eq1 (x y : ℝ) : Prop := y = 2 * x - 1
def eq2 (x y : ℝ) : Prop := y = (4 * x^2 - 1) / (2 * x + 1)
def eq3 (x y : ℝ) : Prop := (2 * x + 1) * y = 4 * x^2 - 1

-- Define the graph of an equation as the set of points (x, y) that satisfy it
def graph (eq : ℝ → ℝ → Prop) : Set (ℝ × ℝ) := {p : ℝ × ℝ | eq p.1 p.2}

-- Theorem stating that all graphs are different
theorem all_graphs_different :
  graph eq1 ≠ graph eq2 ∧ graph eq1 ≠ graph eq3 ∧ graph eq2 ≠ graph eq3 :=
sorry

end NUMINAMATH_CALUDE_all_graphs_different_l675_67506


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bound_l675_67518

theorem arithmetic_geometric_mean_difference_bound (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a < b) : 
  (a + b) / 2 - Real.sqrt (a * b) < (b - a)^2 / (8 * a) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bound_l675_67518


namespace NUMINAMATH_CALUDE_quadratic_minimum_l675_67505

/-- Given a quadratic function f(x) = x^2 - 2x + m with a minimum value of 1 
    on the interval [3, +∞), prove that m = -2. -/
theorem quadratic_minimum (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x : ℝ, x ≥ 3 → f x = x^2 - 2*x + m) →
  (∀ x : ℝ, x ≥ 3 → f x ≥ 1) →
  (∃ x : ℝ, x ≥ 3 ∧ f x = 1) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l675_67505


namespace NUMINAMATH_CALUDE_sum_of_specific_S_values_l675_67543

-- Define the sequence Sn
def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

-- State the theorem
theorem sum_of_specific_S_values : S 19 + S 37 + S 52 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_values_l675_67543


namespace NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l675_67502

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Define what it means for a function to be differentiable
def IsDifferentiable (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f

-- Define what it means for a point to be an extremum
def IsExtremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- Theorem stating that f'(x) = 0 is necessary but not sufficient for x to be an extremum
theorem derivative_zero_necessary_not_sufficient (h : IsDifferentiable f) :
  (IsExtremum f x → deriv f x = 0) ∧
  ¬(deriv f x = 0 → IsExtremum f x) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l675_67502


namespace NUMINAMATH_CALUDE_problem_solution_l675_67526

theorem problem_solution (x : ℝ) 
  (h : x - Real.sqrt (x^2 - 4) + 1 / (x + Real.sqrt (x^2 - 4)) = 10) :
  x^2 - Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 225/16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l675_67526


namespace NUMINAMATH_CALUDE_raul_remaining_money_l675_67566

def initial_amount : ℕ := 87
def number_of_comics : ℕ := 8
def cost_per_comic : ℕ := 4

theorem raul_remaining_money : 
  initial_amount - (number_of_comics * cost_per_comic) = 55 := by
  sorry

end NUMINAMATH_CALUDE_raul_remaining_money_l675_67566


namespace NUMINAMATH_CALUDE_garbage_collection_total_l675_67500

/-- The total amount of garbage collected by two groups, where one group collected 387 pounds and the other collected 39 pounds less. -/
theorem garbage_collection_total : 
  let lizzie_group := 387
  let other_group := lizzie_group - 39
  lizzie_group + other_group = 735 := by
  sorry

end NUMINAMATH_CALUDE_garbage_collection_total_l675_67500


namespace NUMINAMATH_CALUDE_expression_evaluation_l675_67517

theorem expression_evaluation :
  (2^1000 + 5^1001)^2 - (2^1000 - 5^1001)^2 = 20 * 10^1000 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l675_67517


namespace NUMINAMATH_CALUDE_local_max_implies_local_min_l675_67546

-- Define the function f
variable (f : ℝ → ℝ)

-- Define x₀ as a real number not equal to 0
variable (x₀ : ℝ)
variable (h₁ : x₀ ≠ 0)

-- Define that x₀ is a local maximum point of f
def isLocalMax (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀

variable (h₂ : isLocalMax f x₀)

-- Define what it means for a point to be a local minimum
def isLocalMin (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x₀ ≤ f x

-- Theorem statement
theorem local_max_implies_local_min :
  isLocalMin (fun x => -f (-x)) (-x₀) := by sorry

end NUMINAMATH_CALUDE_local_max_implies_local_min_l675_67546


namespace NUMINAMATH_CALUDE_max_tickets_purchasable_l675_67550

def ticket_price : ℚ := 15.75
def processing_fee : ℚ := 1.25
def budget : ℚ := 150.00

theorem max_tickets_purchasable :
  ∀ n : ℕ, n * (ticket_price + processing_fee) ≤ budget ↔ n ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_tickets_purchasable_l675_67550


namespace NUMINAMATH_CALUDE_divisibility_of_polynomial_l675_67595

theorem divisibility_of_polynomial (n : ℤ) : 
  120 ∣ (n^5 - 5*n^3 + 4*n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_polynomial_l675_67595


namespace NUMINAMATH_CALUDE_folded_paper_length_l675_67538

def paper_length : ℝ := 12

theorem folded_paper_length : 
  paper_length / 2 = 6 := by sorry

end NUMINAMATH_CALUDE_folded_paper_length_l675_67538


namespace NUMINAMATH_CALUDE_donut_selection_count_donut_problem_l675_67591

theorem donut_selection_count : Nat → Nat → Nat
  | n, k => Nat.choose (n + k - 1) (k - 1)

theorem donut_problem : donut_selection_count 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_donut_selection_count_donut_problem_l675_67591


namespace NUMINAMATH_CALUDE_measure_one_kg_possible_l675_67534

/-- Represents a balance scale with two pans -/
structure BalanceScale :=
  (left_pan : ℝ)
  (right_pan : ℝ)

/-- Represents the state of the weighing process -/
structure WeighingState :=
  (scale : BalanceScale)
  (remaining_grain : ℝ)
  (weighings_left : ℕ)

/-- Performs a single weighing operation -/
def perform_weighing (state : WeighingState) : WeighingState :=
  sorry

/-- Checks if the current state has isolated 1 kg of grain -/
def has_isolated_one_kg (state : WeighingState) : Prop :=
  sorry

/-- Theorem stating that it's possible to measure 1 kg of grain under the given conditions -/
theorem measure_one_kg_possible :
  ∃ (initial_state : WeighingState),
    initial_state.scale.left_pan = 0 ∧
    initial_state.scale.right_pan = 0 ∧
    initial_state.remaining_grain = 19 ∧
    initial_state.weighings_left = 3 ∧
    ∃ (final_state : WeighingState),
      final_state = (perform_weighing ∘ perform_weighing ∘ perform_weighing) initial_state ∧
      has_isolated_one_kg final_state :=
by
  sorry

end NUMINAMATH_CALUDE_measure_one_kg_possible_l675_67534


namespace NUMINAMATH_CALUDE_discriminant_of_quadratic_equation_l675_67560

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic equation 6x² + (6 + 1/6)x + 1/6 -/
def quadratic_equation (x : ℚ) : ℚ := 6*x^2 + (6 + 1/6)*x + 1/6

theorem discriminant_of_quadratic_equation : 
  discriminant 6 (6 + 1/6) (1/6) = 1225/36 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_quadratic_equation_l675_67560


namespace NUMINAMATH_CALUDE_largest_n_inequality_l675_67541

theorem largest_n_inequality : 
  ∀ n : ℕ, (1/4 : ℚ) + (n/8 : ℚ) < (3/2 : ℚ) ↔ n ≤ 9 := by sorry

end NUMINAMATH_CALUDE_largest_n_inequality_l675_67541


namespace NUMINAMATH_CALUDE_binary_search_sixteen_people_l675_67535

/-- The number of tests required to identify one infected person in a group of size n using binary search. -/
def numTestsBinarySearch (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else 1 + numTestsBinarySearch (n / 2)

/-- Theorem: For a group of 16 people with one infected person, 4 tests are required using binary search. -/
theorem binary_search_sixteen_people :
  numTestsBinarySearch 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_binary_search_sixteen_people_l675_67535


namespace NUMINAMATH_CALUDE_lcm_of_36_and_105_l675_67563

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_105_l675_67563


namespace NUMINAMATH_CALUDE_sufficient_material_for_box_l675_67503

/-- A rectangular box with integer dimensions -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculate the volume of a box -/
def volume (b : Box) : ℕ :=
  b.length * b.width * b.height

/-- Calculate the surface area of a box -/
def surface_area (b : Box) : ℕ :=
  2 * (b.length * b.width + b.length * b.height + b.width * b.height)

/-- Theorem: There exists a box with volume at least 1995 and surface area exactly 958 -/
theorem sufficient_material_for_box : 
  ∃ (b : Box), volume b ≥ 1995 ∧ surface_area b = 958 :=
by
  sorry

end NUMINAMATH_CALUDE_sufficient_material_for_box_l675_67503


namespace NUMINAMATH_CALUDE_harkamal_fruit_purchase_cost_l675_67544

/-- The total cost of fruits purchased by Harkamal -/
def total_cost (grapes_kg : ℕ) (grapes_price : ℕ) 
               (mangoes_kg : ℕ) (mangoes_price : ℕ)
               (apples_kg : ℕ) (apples_price : ℕ)
               (strawberries_kg : ℕ) (strawberries_price : ℕ) : ℕ :=
  grapes_kg * grapes_price + 
  mangoes_kg * mangoes_price + 
  apples_kg * apples_price + 
  strawberries_kg * strawberries_price

/-- Theorem stating the total cost of fruits purchased by Harkamal -/
theorem harkamal_fruit_purchase_cost : 
  total_cost 8 70 9 45 5 30 3 100 = 1415 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_fruit_purchase_cost_l675_67544


namespace NUMINAMATH_CALUDE_solid_volume_l675_67553

/-- A solid with specific face dimensions -/
structure Solid where
  square_side : ℝ
  rect_length : ℝ
  rect_width : ℝ
  trapezoid_leg : ℝ

/-- The volume of the solid -/
noncomputable def volume (s : Solid) : ℝ := sorry

/-- Theorem stating that the volume of the specified solid is 552 dm³ -/
theorem solid_volume (s : Solid) 
  (h1 : s.square_side = 1) 
  (h2 : s.rect_length = 0.4)
  (h3 : s.rect_width = 0.2)
  (h4 : s.trapezoid_leg = 1.3) :
  volume s = 0.552 := by sorry

end NUMINAMATH_CALUDE_solid_volume_l675_67553


namespace NUMINAMATH_CALUDE_bus_wheel_radius_l675_67569

/-- The radius of a bus wheel given its speed and revolutions per minute -/
theorem bus_wheel_radius 
  (speed_kmh : ℝ) 
  (rpm : ℝ) 
  (h1 : speed_kmh = 66) 
  (h2 : rpm = 175.15923566878982) : 
  ∃ (r : ℝ), abs (r - 99.89) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_bus_wheel_radius_l675_67569


namespace NUMINAMATH_CALUDE_a_less_than_one_l675_67524

/-- The sequence a_n defined recursively -/
def a (k : ℕ) : ℕ → ℚ
  | 0 => 1 / k
  | n + 1 => a k n + (1 / (n + 1)^2) * (a k n)^2

/-- The theorem stating the condition for a_n < 1 for all n -/
theorem a_less_than_one (k : ℕ) : (∀ n : ℕ, a k n < 1) ↔ k ≥ 3 := by sorry

end NUMINAMATH_CALUDE_a_less_than_one_l675_67524


namespace NUMINAMATH_CALUDE_ABCD_equals_one_l675_67548

theorem ABCD_equals_one :
  let A := Real.sqrt 3003 + Real.sqrt 3004
  let B := -Real.sqrt 3003 - Real.sqrt 3004
  let C := Real.sqrt 3003 - Real.sqrt 3004
  let D := Real.sqrt 3004 - Real.sqrt 3003
  A * B * C * D = 1 := by
  sorry

end NUMINAMATH_CALUDE_ABCD_equals_one_l675_67548


namespace NUMINAMATH_CALUDE_function_equality_implies_a_value_l675_67523

open Real

theorem function_equality_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, x₀ + exp (x₀ - a) - (log (x₀ + 2) - 4 * exp (a - x₀)) = 3) →
  a = -log 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_function_equality_implies_a_value_l675_67523


namespace NUMINAMATH_CALUDE_possible_values_of_c_l675_67513

theorem possible_values_of_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (h : a^3 - b^3 = a^2 - b^2) :
  {c : ℤ | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ a^3 - b^3 = a^2 - b^2 ∧ c = ⌊9 * a * b⌋} = {1, 2, 3} :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_c_l675_67513


namespace NUMINAMATH_CALUDE_sphere_surface_volume_relation_l675_67571

theorem sphere_surface_volume_relation : 
  ∀ (r : ℝ) (S V S' V' : ℝ), 
    r > 0 →
    S = 4 * Real.pi * r^2 →
    V = (4/3) * Real.pi * r^3 →
    S' = 4 * S →
    V' = (4/3) * Real.pi * (2*r)^3 →
    V' = 8 * V := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_volume_relation_l675_67571


namespace NUMINAMATH_CALUDE_circles_intersect_l675_67565

theorem circles_intersect (r₁ r₂ d : ℝ) (hr₁ : r₁ = 4) (hr₂ : r₂ = 5) (hd : d = 8) :
  (r₂ - r₁ < d) ∧ (d < r₁ + r₂) := by sorry

end NUMINAMATH_CALUDE_circles_intersect_l675_67565


namespace NUMINAMATH_CALUDE_circle_trajectory_l675_67577

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 77 = 0

-- Define the property of being externally tangent
def externally_tangent (P C : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), P x y ∧ C x y ∧ ∀ (x' y' : ℝ), P x' y' → C x' y' → (x = x' ∧ y = y')

-- Define the property of being internally tangent
def internally_tangent (P C : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), P x y ∧ C x y ∧ ∀ (x' y' : ℝ), P x' y' → C x' y' → (x = x' ∧ y = y')

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop := x^2 / 25 + y^2 / 21 = 1

-- State the theorem
theorem circle_trajectory :
  ∀ (P : ℝ → ℝ → Prop),
  (externally_tangent P C₁ ∧ internally_tangent P C₂) →
  (∀ (x y : ℝ), P x y → trajectory x y) :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_l675_67577


namespace NUMINAMATH_CALUDE_fifth_month_sales_l675_67567

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_3 : ℕ := 6855
def sales_4 : ℕ := 7230
def sales_6 : ℕ := 4991
def average_sale : ℕ := 6500
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    sales_5 = average_sale * num_months - (sales_1 + sales_2 + sales_3 + sales_4 + sales_6) ∧
    sales_5 = 6562 :=
by sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l675_67567


namespace NUMINAMATH_CALUDE_jasmine_needs_seven_cans_l675_67588

/-- Represents the paint coverage problem for Jasmine --/
def paint_coverage_problem (initial_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) (rooms_per_new_can : ℕ) (total_rooms : ℕ) : Prop :=
  ∃ (additional_cans : ℕ),
    remaining_rooms + additional_cans * rooms_per_new_can = total_rooms

/-- Theorem stating that 7 additional cans are needed to cover all rooms --/
theorem jasmine_needs_seven_cans :
  paint_coverage_problem 50 4 36 2 50 →
  ∃ (additional_cans : ℕ), additional_cans = 7 ∧ 36 + additional_cans * 2 = 50 := by
  sorry

#check jasmine_needs_seven_cans

end NUMINAMATH_CALUDE_jasmine_needs_seven_cans_l675_67588


namespace NUMINAMATH_CALUDE_dandelion_picking_average_l675_67536

/-- Represents the number of dandelions picked by Billy and George -/
structure DandelionPicks where
  billy_initial : ℕ
  george_initial : ℕ
  billy_additional : ℕ
  george_additional : ℕ

/-- Calculates the average number of dandelions picked -/
def average_picks (d : DandelionPicks) : ℚ :=
  (d.billy_initial + d.george_initial + d.billy_additional + d.george_additional : ℚ) / 2

/-- Theorem stating the average number of dandelions picked by Billy and George -/
theorem dandelion_picking_average :
  ∃ d : DandelionPicks,
    d.billy_initial = 36 ∧
    d.george_initial = (2 * d.billy_initial) / 5 ∧
    d.billy_additional = (5 * d.billy_initial) / 3 ∧
    d.george_additional = (7 * d.george_initial) / 2 ∧
    average_picks d = 79.5 :=
by
  sorry


end NUMINAMATH_CALUDE_dandelion_picking_average_l675_67536


namespace NUMINAMATH_CALUDE_xyz_value_l675_67509

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + x * y * z = 15) :
  x * y * z = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l675_67509


namespace NUMINAMATH_CALUDE_birthday_candles_distribution_l675_67511

/-- The number of people sharing the candles -/
def num_people : ℕ := 4

/-- The number of candles Ambika has -/
def ambika_candles : ℕ := 4

/-- The ratio of Aniyah's candles to Ambika's candles -/
def aniyah_ratio : ℕ := 6

/-- The total number of candles -/
def total_candles : ℕ := aniyah_ratio * ambika_candles + ambika_candles

/-- The number of candles each person gets when shared equally -/
def candles_per_person : ℕ := total_candles / num_people

theorem birthday_candles_distribution :
  candles_per_person = 7 :=
sorry

end NUMINAMATH_CALUDE_birthday_candles_distribution_l675_67511


namespace NUMINAMATH_CALUDE_count_distinct_digit_numbers_l675_67578

/-- The number of four-digit numbers with distinct digits, including numbers beginning with zero -/
def distinctDigitNumbers : ℕ :=
  10 * 9 * 8 * 7

/-- Theorem stating that the number of four-digit numbers with distinct digits,
    including numbers beginning with zero, is equal to 5040 -/
theorem count_distinct_digit_numbers :
  distinctDigitNumbers = 5040 := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_digit_numbers_l675_67578


namespace NUMINAMATH_CALUDE_income_increase_percentage_l675_67590

theorem income_increase_percentage 
  (initial_income : ℝ)
  (initial_expenditure_ratio : ℝ)
  (expenditure_increase_ratio : ℝ)
  (savings_increase_ratio : ℝ)
  (income_increase_ratio : ℝ)
  (h1 : initial_expenditure_ratio = 0.75)
  (h2 : expenditure_increase_ratio = 0.1)
  (h3 : savings_increase_ratio = 0.5)
  (h4 : initial_income > 0) :
  let initial_expenditure := initial_income * initial_expenditure_ratio
  let initial_savings := initial_income - initial_expenditure
  let new_income := initial_income * (1 + income_increase_ratio)
  let new_expenditure := initial_expenditure * (1 + expenditure_increase_ratio)
  let new_savings := new_income - new_expenditure
  (new_savings = initial_savings * (1 + savings_increase_ratio)) →
  (income_increase_ratio = 0.2) :=
by sorry

end NUMINAMATH_CALUDE_income_increase_percentage_l675_67590


namespace NUMINAMATH_CALUDE_rectangular_field_posts_l675_67552

/-- Calculates the number of posts needed for a rectangular fence -/
def num_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let num_sections := perimeter / post_spacing
  num_sections

theorem rectangular_field_posts :
  num_posts 6 8 2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_posts_l675_67552


namespace NUMINAMATH_CALUDE_T_10_mod_5_l675_67551

def T : ℕ → ℕ
  | 0 => 0
  | 1 => 2
  | (n+2) =>
    let c₁ := T n
    let c₂ := T (n+1)
    c₁ + c₂

theorem T_10_mod_5 : T 10 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_T_10_mod_5_l675_67551


namespace NUMINAMATH_CALUDE_permutation_inequality_l675_67597

theorem permutation_inequality (n : ℕ) : 
  (Nat.factorial (n + 1)).choose n ≠ Nat.factorial n := by
  sorry

end NUMINAMATH_CALUDE_permutation_inequality_l675_67597


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l675_67599

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l675_67599


namespace NUMINAMATH_CALUDE_thirtieth_roots_with_real_fifth_power_l675_67559

theorem thirtieth_roots_with_real_fifth_power (ω : ℂ) (h : ω^3 = 1 ∧ ω ≠ 1) :
  ∃! (s : Finset ℂ), 
    (∀ z ∈ s, z^30 = 1) ∧ 
    (∀ z ∈ s, ∃ r : ℝ, z^5 = r) ∧
    s.card = 10 :=
sorry

end NUMINAMATH_CALUDE_thirtieth_roots_with_real_fifth_power_l675_67559


namespace NUMINAMATH_CALUDE_final_book_count_is_1160_l675_67564

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSequenceSum (a1 n d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

/-- Represents Tracy's book store -/
structure BookStore where
  initialBooks : ℕ
  donators : ℕ
  firstDonation : ℕ
  donationIncrement : ℕ
  borrowedBooks : ℕ
  returnedBooks : ℕ

/-- Calculates the final number of books in the store -/
def finalBookCount (store : BookStore) : ℕ :=
  store.initialBooks +
  arithmeticSequenceSum store.firstDonation store.donators store.donationIncrement -
  store.borrowedBooks +
  store.returnedBooks

/-- Theorem stating that the final book count is 1160 -/
theorem final_book_count_is_1160 (store : BookStore)
  (h1 : store.initialBooks = 1000)
  (h2 : store.donators = 15)
  (h3 : store.firstDonation = 2)
  (h4 : store.donationIncrement = 2)
  (h5 : store.borrowedBooks = 350)
  (h6 : store.returnedBooks = 270) :
  finalBookCount store = 1160 := by
  sorry


end NUMINAMATH_CALUDE_final_book_count_is_1160_l675_67564


namespace NUMINAMATH_CALUDE_prime_triplet_equation_l675_67583

theorem prime_triplet_equation :
  ∀ p q r : ℕ,
  Prime p ∧ Prime q ∧ Prime r →
  p * (p - 7) + q * (q - 7) = r * (r - 7) →
  ((p = 2 ∧ q = 5 ∧ r = 7) ∨
   (p = 2 ∧ q = 5 ∧ r = 7) ∨
   (p = 7 ∧ q = 5 ∧ r = 5) ∨
   (p = 5 ∧ q = 7 ∧ r = 5) ∨
   (p = 5 ∧ q = 7 ∧ r = 2) ∨
   (p = 7 ∧ q = 5 ∧ r = 2) ∨
   (p = 7 ∧ q = 3 ∧ r = 3) ∨
   (p = 3 ∧ q = 7 ∧ r = 3) ∨
   (Prime p ∧ q = 7 ∧ r = p) ∨
   (p = 7 ∧ Prime q ∧ r = q)) :=
by sorry

end NUMINAMATH_CALUDE_prime_triplet_equation_l675_67583


namespace NUMINAMATH_CALUDE_visual_range_increase_l675_67589

theorem visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 100)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_visual_range_increase_l675_67589


namespace NUMINAMATH_CALUDE_trajectory_equation_l675_67587

/-- Given m ∈ ℝ, vector a = (mx, y+1), vector b = (x, y-1), and a ⊥ b,
    the equation of trajectory E for moving point M(x,y) is mx² + y² = 1 -/
theorem trajectory_equation (m : ℝ) (x y : ℝ) 
    (h : (m * x) * x + (y + 1) * (y - 1) = 0) : 
  m * x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l675_67587


namespace NUMINAMATH_CALUDE_three_numbers_problem_l675_67515

theorem three_numbers_problem (a b c : ℝ) :
  (a + 1) * (b + 1) * (c + 1) = a * b * c + 1 ∧
  (a + 2) * (b + 2) * (c + 2) = a * b * c + 2 →
  a = -1 ∧ b = -1 ∧ c = -1 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l675_67515


namespace NUMINAMATH_CALUDE_equation_solution_l675_67510

theorem equation_solution (n : ℝ) : 
  (1 / (n + 1) + 2 / (n + 1) + n / (n + 1) + 1 / (n + 2) = 4) ↔ 
  (n = (-3 + Real.sqrt 6) / 3 ∨ n = (-3 - Real.sqrt 6) / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l675_67510


namespace NUMINAMATH_CALUDE_farmer_apples_l675_67540

theorem farmer_apples (initial_apples given_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : given_apples = 88) :
  initial_apples - given_apples = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l675_67540


namespace NUMINAMATH_CALUDE_expand_product_l675_67539

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l675_67539


namespace NUMINAMATH_CALUDE_tree_height_after_three_years_l675_67531

/-- The height of a tree that doubles every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (2 ^ years)

theorem tree_height_after_three_years :
  ∃ (initial_height : ℝ),
    tree_height initial_height 6 = 32 ∧
    tree_height initial_height 3 = 4 := by
  sorry

#check tree_height_after_three_years

end NUMINAMATH_CALUDE_tree_height_after_three_years_l675_67531


namespace NUMINAMATH_CALUDE_valid_selections_eq_sixteen_l675_67514

/-- The number of people to choose from -/
def n : ℕ := 5

/-- The number of positions to fill (team leader and deputy team leader) -/
def k : ℕ := 2

/-- The number of ways to select k people from n people, where order matters -/
def permutations (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

/-- The number of ways to select a team leader and deputy team leader
    when one specific person cannot be the deputy team leader -/
def valid_selections (n k : ℕ) : ℕ :=
  permutations n k - permutations (n - 1) (k - 1)

/-- The main theorem: prove that the number of valid selections is 16 -/
theorem valid_selections_eq_sixteen : valid_selections n k = 16 := by
  sorry

end NUMINAMATH_CALUDE_valid_selections_eq_sixteen_l675_67514


namespace NUMINAMATH_CALUDE_q_polynomial_form_l675_67529

theorem q_polynomial_form (q : ℝ → ℝ) 
  (h : ∀ x, q x + (2 * x^5 + 5 * x^4 + 4 * x^3 + 12 * x) = 3 * x^4 + 14 * x^3 + 32 * x^2 + 17 * x + 3) :
  ∀ x, q x = -2 * x^5 - 2 * x^4 + 10 * x^3 + 32 * x^2 + 5 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l675_67529


namespace NUMINAMATH_CALUDE_sin_angle_through_point_l675_67594

theorem sin_angle_through_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 2 ∧ r * Real.sin α = -1) →
  Real.sin α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_angle_through_point_l675_67594


namespace NUMINAMATH_CALUDE_problem_statement_l675_67558

theorem problem_statement : 
  (3 * (0.6 * 40) - (4/5 * 25) / 2) * (Real.sqrt 16 - 3) = 62 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l675_67558


namespace NUMINAMATH_CALUDE_initial_milk_amount_l675_67580

/-- Proves that the initial amount of milk is 10 liters given the conditions of the problem -/
theorem initial_milk_amount (initial_water_content : Real) 
                            (final_water_content : Real)
                            (pure_milk_added : Real) :
  initial_water_content = 0.05 →
  final_water_content = 0.02 →
  pure_milk_added = 15 →
  ∃ (initial_milk : Real),
    initial_milk = 10 ∧
    initial_water_content * initial_milk = 
    final_water_content * (initial_milk + pure_milk_added) :=
by
  sorry


end NUMINAMATH_CALUDE_initial_milk_amount_l675_67580


namespace NUMINAMATH_CALUDE_warehouse_construction_l675_67581

/-- Warehouse construction problem -/
theorem warehouse_construction (investment : ℝ) (front_cost side_cost top_cost : ℝ) 
  (h_investment : investment = 3200)
  (h_front_cost : front_cost = 40)
  (h_side_cost : side_cost = 45)
  (h_top_cost : top_cost = 20) :
  ∃ (x y : ℝ),
    0 < x ∧ x < 80 ∧
    y = (320 - 4*x) / (9 + 2*x) ∧
    x * y ≤ 100 ∧
    (∀ x' y' : ℝ, 0 < x' ∧ x' < 80 ∧ y' = (320 - 4*x') / (9 + 2*x') → x' * y' ≤ x * y) ∧
    x = 15 ∧ y = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_construction_l675_67581


namespace NUMINAMATH_CALUDE_quadratic_sum_l675_67557

theorem quadratic_sum (x y : ℝ) 
  (h1 : 4 * x + 2 * y = 12) 
  (h2 : 2 * x + 4 * y = 20) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 544 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l675_67557


namespace NUMINAMATH_CALUDE_road_trip_speed_l675_67579

/-- Road trip problem -/
theorem road_trip_speed (total_distance : ℝ) (jenna_distance : ℝ) (friend_distance : ℝ)
  (jenna_speed : ℝ) (total_time : ℝ) (num_breaks : ℕ) (break_duration : ℝ) :
  total_distance = jenna_distance + friend_distance →
  jenna_distance = 200 →
  friend_distance = 100 →
  jenna_speed = 50 →
  total_time = 10 →
  num_breaks = 2 →
  break_duration = 0.5 →
  ∃ (friend_speed : ℝ), friend_speed = 20 ∧ 
    total_time = jenna_distance / jenna_speed + friend_distance / friend_speed + num_breaks * break_duration :=
by sorry


end NUMINAMATH_CALUDE_road_trip_speed_l675_67579


namespace NUMINAMATH_CALUDE_no_valid_n_l675_67573

/-- Represents the number of matches won by women -/
def women_wins (n : ℕ) : ℚ := 3 * (n * (4 * n - 1) / 8)

/-- Represents the number of matches won by men -/
def men_wins (n : ℕ) : ℚ := 5 * (n * (4 * n - 1) / 8)

/-- Represents the total number of matches played -/
def total_matches (n : ℕ) : ℕ := n * (4 * n - 1) / 2

theorem no_valid_n : ∀ n : ℕ, n > 0 →
  (women_wins n + men_wins n = total_matches n) →
  (3 * men_wins n = 5 * women_wins n) →
  False :=
sorry

end NUMINAMATH_CALUDE_no_valid_n_l675_67573


namespace NUMINAMATH_CALUDE_max_sum_abs_on_circle_l675_67537

theorem max_sum_abs_on_circle : 
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ 
  (∀ x y : ℝ, x^2 + y^2 = 4 → |x| + |y| ≤ M) ∧
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = M) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_abs_on_circle_l675_67537


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l675_67530

theorem square_plus_reciprocal_square (x : ℝ) (h : x ≠ 0) :
  x^4 + (1/x^4) = 47 → x^2 + (1/x^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l675_67530


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l675_67554

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem digit_sum_theorem (a b c d : ℕ) (square : ℕ) :
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit square →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * 100 + 60 + b - (400 + c * 10 + d) = 2 →
  a + b + c + d = 10 ∨ a + b + c + d = 18 ∨ a + b + c + d = 19 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l675_67554


namespace NUMINAMATH_CALUDE_min_value_3x_plus_4y_l675_67527

theorem min_value_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_4y_l675_67527
