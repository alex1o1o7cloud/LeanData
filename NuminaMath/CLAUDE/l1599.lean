import Mathlib

namespace distribute_balls_into_boxes_l1599_159980

theorem distribute_balls_into_boxes (n : ℕ) (k : ℕ) : 
  n = 5 → k = 4 → (Nat.choose (n + k - 1) (k - 1)) = 56 := by
  sorry

end distribute_balls_into_boxes_l1599_159980


namespace min_perimeter_triangle_l1599_159955

theorem min_perimeter_triangle (d e f : ℕ) : 
  d > 0 → e > 0 → f > 0 →
  (d^2 + e^2 - f^2 : ℚ) / (2 * d * e) = 3 / 5 →
  (d^2 + f^2 - e^2 : ℚ) / (2 * d * f) = 9 / 10 →
  (e^2 + f^2 - d^2 : ℚ) / (2 * e * f) = -1 / 3 →
  d + e + f ≥ 50 :=
by sorry

end min_perimeter_triangle_l1599_159955


namespace range_of_m_l1599_159989

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 4 * x + y - x * y = 0) (h2 : x * y ≥ m^2 - 6*m) : 
  -2 ≤ m ∧ m ≤ 8 := by
  sorry

end range_of_m_l1599_159989


namespace johns_age_l1599_159981

/-- John's age in years -/
def john_age : ℕ := sorry

/-- John's dad's age in years -/
def dad_age : ℕ := sorry

/-- John is 18 years younger than his dad -/
axiom age_difference : john_age = dad_age - 18

/-- The sum of John's and his dad's ages is 74 years -/
axiom age_sum : john_age + dad_age = 74

/-- Theorem: John's age is 28 years -/
theorem johns_age : john_age = 28 := by sorry

end johns_age_l1599_159981


namespace total_wheels_calculation_l1599_159906

/-- The number of wheels on a four-wheeler -/
def wheels_per_four_wheeler : ℕ := 4

/-- The number of four-wheelers parked -/
def num_four_wheelers : ℕ := 13

/-- The total number of wheels for all four-wheelers -/
def total_wheels : ℕ := num_four_wheelers * wheels_per_four_wheeler

theorem total_wheels_calculation :
  total_wheels = 52 := by sorry

end total_wheels_calculation_l1599_159906


namespace triangle_angles_l1599_159941

theorem triangle_angles (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) * (a + b - c) = 3 * a * b →
  Real.sin A ^ 2 = Real.sin B ^ 2 + Real.sin C ^ 2 →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  A = π / 6 ∧ B = π / 3 ∧ C = π / 2 := by
  sorry

end triangle_angles_l1599_159941


namespace approximateValuesOfSqrt3_cannot_form_set_l1599_159947

-- Define a type for the concept of "group of objects"
structure GroupOfObjects where
  elements : Set ℝ
  description : String

-- Define the properties required for a set
def hasDeterminacy (g : GroupOfObjects) : Prop :=
  ∀ x, x ∈ g.elements → (∃ y, y = x)

def hasDistinctness (g : GroupOfObjects) : Prop :=
  ∀ x y, x ∈ g.elements → y ∈ g.elements → x = y → x = y

def hasUnorderedness (g : GroupOfObjects) : Prop :=
  ∀ x y, x ∈ g.elements → y ∈ g.elements → x ≠ y → y ∈ g.elements

-- Define what it means for a group of objects to be able to form a set
def canFormSet (g : GroupOfObjects) : Prop :=
  hasDeterminacy g ∧ hasDistinctness g ∧ hasUnorderedness g

-- Define the group of all approximate values of √3
def approximateValuesOfSqrt3 : GroupOfObjects :=
  { elements := {x : ℝ | ∃ ε > 0, |x^2 - 3| < ε},
    description := "All approximate values of √3" }

-- The theorem to prove
theorem approximateValuesOfSqrt3_cannot_form_set :
  ¬(canFormSet approximateValuesOfSqrt3) :=
sorry

end approximateValuesOfSqrt3_cannot_form_set_l1599_159947


namespace angle_extrema_l1599_159926

/-- The angle formed by the construction described in the problem -/
def constructionAngle (x : Fin n → ℝ) : ℝ :=
  sorry

/-- The theorem stating that the angle is minimal for descending sequences and maximal for ascending sequences -/
theorem angle_extrema (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, x i > 0) :
  (∀ i j, i < j → x i ≥ x j) →
  (∀ y : Fin n → ℝ, (∀ i, y i > 0) → constructionAngle x ≤ constructionAngle y) ∧
  (∀ i j, i < j → x i ≤ x j) →
  (∀ y : Fin n → ℝ, (∀ i, y i > 0) → constructionAngle x ≥ constructionAngle y) :=
sorry

end angle_extrema_l1599_159926


namespace frog_jumps_equivalence_l1599_159921

/-- Represents a frog's position on an integer line -/
def FrogPosition := ℤ

/-- Represents a configuration of frogs on the line -/
def FrogConfiguration := List FrogPosition

/-- Represents a direction of movement -/
inductive Direction
| Left : Direction
| Right : Direction

/-- Represents a sequence of n moves -/
def MoveSequence (n : ℕ) := Vector Direction n

/-- Predicate to check if a configuration has distinct positions -/
def HasDistinctPositions (config : FrogConfiguration) : Prop :=
  config.Nodup

/-- Function to count valid move sequences -/
def CountValidMoveSequences (n : ℕ) (initialConfig : FrogConfiguration) (dir : Direction) : ℕ :=
  sorry  -- Implementation details omitted

theorem frog_jumps_equivalence 
  (n : ℕ) 
  (initialConfig : FrogConfiguration) 
  (h : HasDistinctPositions initialConfig) :
  CountValidMoveSequences n initialConfig Direction.Right = 
  CountValidMoveSequences n initialConfig Direction.Left :=
sorry

end frog_jumps_equivalence_l1599_159921


namespace square_difference_306_294_l1599_159963

theorem square_difference_306_294 : 306^2 - 294^2 = 7200 := by
  sorry

end square_difference_306_294_l1599_159963


namespace line_up_count_proof_l1599_159950

/-- The number of ways to arrange 5 people in a line with the youngest not first or last -/
def lineUpCount : ℕ := 72

/-- The number of possible positions for the youngest person -/
def youngestPositions : ℕ := 3

/-- The number of ways to arrange the other 4 people -/
def otherArrangements : ℕ := 24

theorem line_up_count_proof :
  lineUpCount = youngestPositions * otherArrangements :=
by sorry

end line_up_count_proof_l1599_159950


namespace toothpick_pattern_l1599_159900

/-- 
Given a sequence where:
- The first term is 6
- Each successive term increases by 5 more than the previous increase
Prove that the 150th term is equal to 751
-/
theorem toothpick_pattern (n : ℕ) (a : ℕ → ℕ) : 
  a 1 = 6 ∧ 
  (∀ k, k ≥ 1 → a (k + 1) - a k = a k - a (k - 1) + 5) →
  a 150 = 751 :=
sorry

end toothpick_pattern_l1599_159900


namespace black_cards_remaining_l1599_159949

/-- Represents a standard deck of playing cards -/
structure Deck where
  total_cards : Nat
  black_cards : Nat
  removed_cards : Nat

/-- Theorem: The number of black cards remaining in a standard deck after removing 4 black cards is 22 -/
theorem black_cards_remaining (d : Deck) 
  (h1 : d.total_cards = 52)
  (h2 : d.black_cards = 26)
  (h3 : d.removed_cards = 4) :
  d.black_cards - d.removed_cards = 22 := by
  sorry

end black_cards_remaining_l1599_159949


namespace meaningful_fraction_range_l1599_159907

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end meaningful_fraction_range_l1599_159907


namespace binomial_8_choose_3_l1599_159958

theorem binomial_8_choose_3 : Nat.choose 8 3 = 56 := by
  sorry

end binomial_8_choose_3_l1599_159958


namespace count_valid_pairs_l1599_159962

/-- The number of ordered pairs (m,n) of positive integers satisfying the given conditions -/
def solution_count : ℕ := 3

/-- Predicate defining the conditions for valid pairs -/
def is_valid_pair (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 128

theorem count_valid_pairs :
  (∃! (s : Finset (ℕ × ℕ)), ∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_pair p.1 p.2) ∧
  (∃ (s : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_pair p.1 p.2) ∧ s.card = solution_count) :=
sorry

end count_valid_pairs_l1599_159962


namespace geometric_sequence_sum_ratio_l1599_159998

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 1 / a 0

/-- Sum of the first n terms of a geometric sequence -/
def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ := sorry

/-- The main theorem -/
theorem geometric_sequence_sum_ratio 
  (seq : GeometricSequence) 
  (h : seq.a 6 = 8 * seq.a 3) : 
  sum_n seq 6 / sum_n seq 3 = 9 := by sorry

end geometric_sequence_sum_ratio_l1599_159998


namespace trapezoid_segment_length_l1599_159961

/-- Represents a trapezoid PQRU -/
structure Trapezoid where
  PQ : ℝ
  RU : ℝ

/-- The theorem stating the length of PQ in the given trapezoid -/
theorem trapezoid_segment_length (PQRU : Trapezoid) 
  (h1 : PQRU.PQ / PQRU.RU = 5 / 2)
  (h2 : PQRU.PQ + PQRU.RU = 180) : 
  PQRU.PQ = 900 / 7 := by
  sorry

end trapezoid_segment_length_l1599_159961


namespace line_y_axis_intersection_l1599_159953

/-- A line passing through two given points intersects the y-axis at a specific point -/
theorem line_y_axis_intersection (x₁ y₁ x₂ y₂ : ℚ) 
  (h₁ : x₁ = 3) (h₂ : y₁ = 20) (h₃ : x₂ = -9) (h₄ : y₂ = -6) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0, b) = (0, 27/2) :=
by sorry

end line_y_axis_intersection_l1599_159953


namespace gold_distribution_l1599_159942

/-- Given an arithmetic sequence with 10 terms, if the sum of the first 3 terms
    is 4 and the sum of the last 4 terms is 3, then the common difference
    of the sequence is 7/78. -/
theorem gold_distribution (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
  (∀ n, n ≥ 10 → a n = 0) →             -- 10 terms
  a 9 + a 8 + a 7 = 4 →                 -- sum of first 3 terms is 4
  a 0 + a 1 + a 2 + a 3 = 3 →           -- sum of last 4 terms is 3
  a 1 - a 0 = 7 / 78 :=                 -- common difference is 7/78
by sorry

end gold_distribution_l1599_159942


namespace project_distribution_count_l1599_159999

/-- The number of districts --/
def num_districts : ℕ := 4

/-- The number of projects to be sponsored --/
def num_projects : ℕ := 3

/-- The maximum number of projects allowed in a single district --/
def max_projects_per_district : ℕ := 2

/-- The total number of possible distributions of projects among districts --/
def total_distributions : ℕ := num_districts ^ num_projects

/-- The number of invalid distributions (more than 2 projects in a district) --/
def invalid_distributions : ℕ := num_districts

theorem project_distribution_count :
  (total_distributions - invalid_distributions) = 60 := by
  sorry

end project_distribution_count_l1599_159999


namespace dealership_van_sales_l1599_159930

/-- Calculates the expected number of vans to be sold given the truck-to-van ratio and the number of trucks expected to be sold. -/
def expected_vans (truck_ratio : ℕ) (van_ratio : ℕ) (trucks_sold : ℕ) : ℕ :=
  (van_ratio * trucks_sold) / truck_ratio

/-- Theorem stating that given a 3:5 ratio of trucks to vans and an expected sale of 45 trucks, 
    the expected number of vans to be sold is 75. -/
theorem dealership_van_sales : expected_vans 3 5 45 = 75 := by
  sorry

end dealership_van_sales_l1599_159930


namespace base6_addition_problem_l1599_159933

/-- Converts a base 6 number to its decimal representation -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Checks if the given digit satisfies the base 6 addition problem -/
def satisfiesAdditionProblem (digit : Nat) : Prop :=
  let num1 := base6ToDecimal [4, 3, 2, digit]
  let num2 := base6ToDecimal [digit, 5, 1]
  let num3 := base6ToDecimal [digit, 3]
  let sum := base6ToDecimal [5, 3, digit, 0]
  num1 + num2 + num3 = sum

theorem base6_addition_problem :
  ∃! (digit : Nat), digit < 6 ∧ satisfiesAdditionProblem digit :=
sorry

end base6_addition_problem_l1599_159933


namespace alok_chapati_order_l1599_159959

-- Define the variables
def rice_plates : ℕ := 5
def vegetable_plates : ℕ := 7
def ice_cream_cups : ℕ := 6
def chapati_cost : ℕ := 6
def rice_cost : ℕ := 45
def vegetable_cost : ℕ := 70
def total_paid : ℕ := 1111

-- Define the theorem
theorem alok_chapati_order :
  ∃ (chapatis : ℕ), 
    chapatis * chapati_cost + 
    rice_plates * rice_cost + 
    vegetable_plates * vegetable_cost + 
    ice_cream_cups * (total_paid - (chapatis * chapati_cost + rice_plates * rice_cost + vegetable_plates * vegetable_cost)) / ice_cream_cups = 
    total_paid ∧ 
    chapatis = 66 := by
  sorry

end alok_chapati_order_l1599_159959


namespace first_hour_distance_l1599_159920

/-- A structure representing a family road trip -/
structure RoadTrip where
  firstHourDistance : ℝ
  remainingDistance : ℝ
  totalTime : ℝ
  speed : ℝ

/-- Theorem: Given the conditions of the road trip, the distance traveled in the first hour is 100 miles -/
theorem first_hour_distance (trip : RoadTrip) 
  (h1 : trip.remainingDistance = 300)
  (h2 : trip.totalTime = 4)
  (h3 : trip.speed * 1 = trip.firstHourDistance)
  (h4 : trip.speed * 3 = trip.remainingDistance) : 
  trip.firstHourDistance = 100 := by
  sorry

#check first_hour_distance

end first_hour_distance_l1599_159920


namespace next_signal_time_l1599_159901

def factory_interval : ℕ := 18
def train_interval : ℕ := 24
def lighthouse_interval : ℕ := 36
def start_time : ℕ := 480  -- 8:00 AM in minutes since midnight

def next_simultaneous_signal (f t l s : ℕ) : ℕ :=
  s + Nat.lcm (Nat.lcm f t) l

theorem next_signal_time :
  next_simultaneous_signal factory_interval train_interval lighthouse_interval start_time = 552 := by
  sorry

#eval next_simultaneous_signal factory_interval train_interval lighthouse_interval start_time

end next_signal_time_l1599_159901


namespace opposite_silver_is_orange_l1599_159977

-- Define the colors
inductive Color
| Orange | Blue | Yellow | Black | Silver | Pink

-- Define the cube faces
inductive Face
| Top | Bottom | Front | Back | Left | Right

-- Define the cube structure
structure Cube where
  color : Face → Color

-- Define the three views of the cube
def view1 (c : Cube) : Prop :=
  c.color Face.Top = Color.Black ∧
  c.color Face.Front = Color.Blue ∧
  c.color Face.Right = Color.Yellow

def view2 (c : Cube) : Prop :=
  c.color Face.Top = Color.Black ∧
  c.color Face.Front = Color.Pink ∧
  c.color Face.Right = Color.Yellow

def view3 (c : Cube) : Prop :=
  c.color Face.Top = Color.Black ∧
  c.color Face.Front = Color.Silver ∧
  c.color Face.Right = Color.Yellow

-- Define the theorem
theorem opposite_silver_is_orange (c : Cube) :
  view1 c → view2 c → view3 c →
  c.color Face.Back = Color.Orange :=
sorry

end opposite_silver_is_orange_l1599_159977


namespace equation_one_solution_equation_two_no_solution_l1599_159952

-- Equation 1
theorem equation_one_solution (x : ℚ) : 
  (3/2 - 1/(3*x - 1) = 5/(6*x - 2)) ↔ (x = 10/9) :=
sorry

-- Equation 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℚ), (5*x - 4)/(x - 2) = (4*x + 10)/(3*x - 6) - 1 :=
sorry

end equation_one_solution_equation_two_no_solution_l1599_159952


namespace fraction_sum_equality_l1599_159918

theorem fraction_sum_equality (n : ℕ) (hn : n > 1) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x ≤ y ∧ (1 : ℚ) / n = 1 / x - 1 / (y + 1) := by
  sorry

end fraction_sum_equality_l1599_159918


namespace max_m_value_l1599_159944

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_eq : 2/a + 1/b = 1/4) :
  (∀ m : ℝ, 2*a + b ≥ 9*m) → (∃ m_max : ℝ, m_max = 4 ∧ ∀ m : ℝ, (2*a + b ≥ 9*m → m ≤ m_max)) :=
by sorry

end max_m_value_l1599_159944


namespace range_of_a_l1599_159971

/-- The function f(x) = x|x^2 - 12| -/
def f (x : ℝ) : ℝ := x * abs (x^2 - 12)

theorem range_of_a (m : ℝ) (h_m : m > 0) :
  (∃ (a : ℝ), ∀ (y : ℝ), y ∈ Set.range (fun x => f x) ↔ y ∈ Set.Icc 0 (a * m^2)) →
  (∃ (a : ℝ), a ≥ 1 ∧ ∀ (b : ℝ), b ≥ 1 → ∃ (m : ℝ), m > 0 ∧
    (∀ (y : ℝ), y ∈ Set.range (fun x => f x) ↔ y ∈ Set.Icc 0 (b * m^2))) :=
by sorry

end range_of_a_l1599_159971


namespace scientific_notation_700_3_l1599_159925

/-- Definition of scientific notation -/
def is_scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

/-- Theorem: 700.3 in scientific notation -/
theorem scientific_notation_700_3 :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation 700.3 a n ∧ a = 7.003 ∧ n = 2 := by
  sorry

end scientific_notation_700_3_l1599_159925


namespace males_band_not_orchestra_l1599_159986

/-- Represents the school band and orchestra at West Valley High -/
structure MusicGroups where
  band_females : ℕ
  band_males : ℕ
  orchestra_females : ℕ
  orchestra_males : ℕ
  both_females : ℕ
  total_students : ℕ

/-- The specific music groups at West Valley High -/
def westValleyHigh : MusicGroups :=
  { band_females := 120
  , band_males := 100
  , orchestra_females := 90
  , orchestra_males := 110
  , both_females := 70
  , total_students := 250 }

/-- The number of males in the band who are not in the orchestra is 0 -/
theorem males_band_not_orchestra (g : MusicGroups) (h : g = westValleyHigh) :
  g.band_males - (g.band_males + g.orchestra_males - (g.total_students - (g.band_females + g.orchestra_females - g.both_females))) = 0 := by
  sorry


end males_band_not_orchestra_l1599_159986


namespace train_length_train_length_approx_l1599_159990

/-- The length of a train given its speed and time to cross a post -/
theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  speed_m_s * time_seconds

/-- Theorem stating that a train with speed 40 km/hr crossing a post in 25.2 seconds has a length of approximately 280 meters -/
theorem train_length_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |train_length 40 25.2 - 280| < ε :=
sorry

end train_length_train_length_approx_l1599_159990


namespace inequality_proof_l1599_159995

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
sorry

end inequality_proof_l1599_159995


namespace linda_savings_fraction_l1599_159951

theorem linda_savings_fraction (original_savings : ℚ) (tv_cost : ℚ) 
  (h1 : original_savings = 880)
  (h2 : tv_cost = 220) :
  (original_savings - tv_cost) / original_savings = 3 / 4 := by
  sorry

end linda_savings_fraction_l1599_159951


namespace A_and_C_work_time_l1599_159991

-- Define work rates
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12
def work_rate_BC : ℚ := 1 / 3

-- Define the theorem
theorem A_and_C_work_time :
  let work_rate_C : ℚ := work_rate_BC - work_rate_B
  let work_rate_AC : ℚ := work_rate_A + work_rate_C
  (1 : ℚ) / work_rate_AC = 2 := by sorry

end A_and_C_work_time_l1599_159991


namespace min_sum_squares_l1599_159940

theorem min_sum_squares (x y z t : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : 0 ≤ t)
  (h5 : |x - y| + |y - z| + |z - t| + |t - x| = 4) :
  2 ≤ x^2 + y^2 + z^2 + t^2 ∧ ∃ (a b c d : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    |a - b| + |b - c| + |c - d| + |d - a| = 4 ∧ a^2 + b^2 + c^2 + d^2 = 2 :=
by sorry

end min_sum_squares_l1599_159940


namespace encyclopedia_chapters_l1599_159983

theorem encyclopedia_chapters (total_pages : ℕ) (pages_per_chapter : ℕ) (h1 : total_pages = 3962) (h2 : pages_per_chapter = 566) :
  total_pages / pages_per_chapter = 7 := by
sorry

end encyclopedia_chapters_l1599_159983


namespace fraction_equality_l1599_159908

theorem fraction_equality (P Q M N X : ℚ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.3 * P)
  (hN : N = 0.6 * P)
  (hX : X = 0.25 * M)
  (hP : P ≠ 0) : 
  X / N = 1 / 20 := by
  sorry

end fraction_equality_l1599_159908


namespace mod_difference_equals_negative_four_l1599_159967

-- Define the % operation
def mod (x y : ℤ) : ℤ := x * y - 3 * x - y

-- State the theorem
theorem mod_difference_equals_negative_four : 
  (mod 6 4) - (mod 4 6) = -4 := by
  sorry

end mod_difference_equals_negative_four_l1599_159967


namespace product_of_numbers_with_sum_and_difference_l1599_159994

theorem product_of_numbers_with_sum_and_difference 
  (x y : ℝ) (sum_eq : x + y = 60) (diff_eq : x - y = 10) : x * y = 875 := by
  sorry

end product_of_numbers_with_sum_and_difference_l1599_159994


namespace cubic_expression_value_l1599_159910

theorem cubic_expression_value (a b : ℝ) :
  (a * 1^3 + b * 1 + 1 = 5) → (a * (-1)^3 + b * (-1) + 1 = -3) :=
by sorry

end cubic_expression_value_l1599_159910


namespace ocean_area_scientific_notation_l1599_159960

/-- The total area of the global ocean in million square kilometers -/
def ocean_area : ℝ := 36200

/-- The conversion factor from million to scientific notation -/
def million_to_scientific : ℝ := 10^6

theorem ocean_area_scientific_notation :
  ocean_area * million_to_scientific = 3.62 * 10^8 := by
  sorry

end ocean_area_scientific_notation_l1599_159960


namespace ratio_problem_l1599_159909

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 2) : 
  (a + b) / (b + c) = 4/9 := by
sorry

end ratio_problem_l1599_159909


namespace h_nonzero_l1599_159987

/-- A polynomial of degree 4 with four distinct roots, one of which is 0 -/
structure QuarticPolynomial where
  f : ℝ
  g : ℝ
  h : ℝ
  roots : Finset ℝ
  distinct_roots : roots.card = 4
  zero_root : (0 : ℝ) ∈ roots
  is_root (x : ℝ) : x ∈ roots → x^4 + f*x^3 + g*x^2 + h*x = 0

theorem h_nonzero (Q : QuarticPolynomial) : Q.h ≠ 0 := by
  sorry

end h_nonzero_l1599_159987


namespace kim_payment_share_l1599_159976

/-- Represents the time (in days) it takes a person to complete the work alone -/
structure WorkTime where
  days : ℚ
  days_positive : days > 0

/-- Calculates the work rate (portion of work done per day) given the work time -/
def work_rate (wt : WorkTime) : ℚ := 1 / wt.days

/-- Calculates the share of payment for a person given their work rate and the total work rate -/
def payment_share (individual_rate total_rate : ℚ) : ℚ := individual_rate / total_rate

theorem kim_payment_share 
  (kim : WorkTime)
  (david : WorkTime)
  (lisa : WorkTime)
  (h_kim : kim.days = 3)
  (h_david : david.days = 2)
  (h_lisa : lisa.days = 4)
  (total_payment : ℚ)
  (h_total_payment : total_payment = 200) :
  payment_share (work_rate kim) (work_rate kim + work_rate david + work_rate lisa) * total_payment = 800 / 13 :=
sorry

end kim_payment_share_l1599_159976


namespace dog_sled_race_l1599_159927

theorem dog_sled_race (total_sleds : ℕ) (pairs : ℕ) (triples : ℕ) 
  (h1 : total_sleds = 315)
  (h2 : pairs + triples = total_sleds)
  (h3 : (6 * pairs + 2 * triples) * 10 = (2 * pairs + 3 * triples) * 5) :
  pairs = 225 ∧ triples = 90 := by
  sorry

end dog_sled_race_l1599_159927


namespace a_eq_one_sufficient_not_necessary_l1599_159973

theorem a_eq_one_sufficient_not_necessary :
  ∃ (a : ℝ), a ^ 2 = a ∧ a ≠ 1 ∧
  ∀ (b : ℝ), b = 1 → b ^ 2 = b :=
by sorry

end a_eq_one_sufficient_not_necessary_l1599_159973


namespace max_value_sin_cos_function_l1599_159923

theorem max_value_sin_cos_function :
  let f : ℝ → ℝ := λ x => Real.sin (π / 2 + x) * Real.cos (π / 6 - x)
  ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = (2 + Real.sqrt 3) / 4 := by
  sorry

end max_value_sin_cos_function_l1599_159923


namespace c_profit_share_l1599_159972

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_profit_share (total_investment : ℕ) (partner_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (partner_investment * total_profit) / total_investment

theorem c_profit_share :
  let a_investment : ℕ := 5000
  let b_investment : ℕ := 8000
  let c_investment : ℕ := 9000
  let total_investment : ℕ := a_investment + b_investment + c_investment
  let total_profit : ℕ := 88000
  calculate_profit_share total_investment c_investment total_profit = 36000 := by
  sorry

end c_profit_share_l1599_159972


namespace variance_of_defective_parts_l1599_159937

def defective_parts : List ℕ := [3, 3, 0, 2, 3, 0, 3]

def mean (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

def variance (list : List ℕ) : ℚ :=
  let μ := mean list
  (list.map (fun x => ((x : ℚ) - μ) ^ 2)).sum / list.length

theorem variance_of_defective_parts :
  variance defective_parts = 12 / 7 := by
  sorry

end variance_of_defective_parts_l1599_159937


namespace arithmetic_sequence_property_l1599_159948

/-- Given an arithmetic sequence with 3n terms, prove that t₁ = t₂ -/
theorem arithmetic_sequence_property (n : ℕ) (s₁ s₂ s₃ : ℝ) :
  let t₁ := s₂^2 - s₁*s₃
  let t₂ := ((s₁ - s₃)/2)^2
  (s₁ + s₃ = 2*s₂) → t₁ = t₂ := by
sorry

end arithmetic_sequence_property_l1599_159948


namespace frequency_distribution_necessary_sufficient_l1599_159982

/-- Represents a student's test score -/
def TestScore := ℕ

/-- Represents a group of students who took the test -/
def StudentGroup := List TestScore

/-- Represents the different score ranges -/
inductive ScoreRange
  | AboveOrEqual120
  | Between90And120
  | Between75And90
  | Between60And75
  | Below60

/-- Function to calculate the proportion of students in each score range -/
def calculateProportions (students : StudentGroup) : ScoreRange → ℚ :=
  sorry

/-- Function to perform frequency distribution -/
def frequencyDistribution (students : StudentGroup) : ScoreRange → ℕ :=
  sorry

/-- Theorem stating that frequency distribution is necessary and sufficient
    to determine the proportions of students in different score ranges -/
theorem frequency_distribution_necessary_sufficient
  (students : StudentGroup)
  (h : students.length = 800) :
  (∀ range, calculateProportions students range =
    (frequencyDistribution students range : ℚ) / 800) :=
sorry

end frequency_distribution_necessary_sufficient_l1599_159982


namespace english_chinese_difference_l1599_159997

def hours_english : ℕ := 6
def hours_chinese : ℕ := 3

theorem english_chinese_difference : hours_english - hours_chinese = 3 := by
  sorry

end english_chinese_difference_l1599_159997


namespace isosceles_triangle_most_stable_l1599_159943

-- Define the shapes
inductive Shape
  | RegularHexagon
  | Square
  | Pentagon
  | IsoscelesTriangle

-- Define a function to get the number of sides for each shape
def numSides (s : Shape) : Nat :=
  match s with
  | .RegularHexagon => 6
  | .Square => 4
  | .Pentagon => 5
  | .IsoscelesTriangle => 3

-- Define stability as inversely proportional to the number of sides
def stability (s : Shape) : Nat := 7 - numSides s

-- Theorem: Isosceles triangle is the most stable shape
theorem isosceles_triangle_most_stable :
  ∀ s : Shape, s ≠ Shape.IsoscelesTriangle → 
    stability Shape.IsoscelesTriangle > stability s :=
by sorry

#check isosceles_triangle_most_stable

end isosceles_triangle_most_stable_l1599_159943


namespace elizabeth_position_l1599_159968

theorem elizabeth_position (total_distance : ℝ) (total_steps : ℕ) (steps_taken : ℕ) : 
  total_distance = 24 → 
  total_steps = 6 → 
  steps_taken = 4 → 
  (total_distance / total_steps) * steps_taken = 16 := by
sorry

end elizabeth_position_l1599_159968


namespace lock_and_key_theorem_l1599_159928

/-- The number of scientists in the team -/
def n : ℕ := 7

/-- The minimum number of scientists required to open the door -/
def k : ℕ := 4

/-- The number of scientists that can be absent -/
def m : ℕ := n - k

/-- The number of unique locks required -/
def num_locks : ℕ := Nat.choose n m

/-- The number of keys each scientist must have -/
def num_keys : ℕ := Nat.choose (n - 1) m

theorem lock_and_key_theorem :
  (num_locks = 35) ∧ (num_keys = 20) :=
sorry

end lock_and_key_theorem_l1599_159928


namespace line_plane_perpendicularity_l1599_159978

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l m n : Line) (α : Plane) :
  parallel l m → parallel m n → perpendicular l α → perpendicular n α :=
sorry

end line_plane_perpendicularity_l1599_159978


namespace smallest_number_l1599_159916

-- Define the numbers in their respective bases
def num_decimal : ℕ := 75
def num_binary : ℕ := 63  -- 111111₍₂₎ in decimal
def num_base_6 : ℕ := 2 * 6^2 + 1 * 6  -- 210₍₆₎
def num_base_9 : ℕ := 8 * 9 + 5  -- 85₍₉₎

-- Theorem statement
theorem smallest_number :
  num_binary < num_decimal ∧
  num_binary < num_base_6 ∧
  num_binary < num_base_9 :=
by sorry

end smallest_number_l1599_159916


namespace matthews_friends_l1599_159905

theorem matthews_friends (total_crackers : ℕ) (crackers_per_friend : ℕ) 
  (h1 : total_crackers = 36) (h2 : crackers_per_friend = 6) : 
  total_crackers / crackers_per_friend = 6 := by
  sorry

end matthews_friends_l1599_159905


namespace oldest_child_age_oldest_child_age_proof_l1599_159966

/-- Proves that given four children with an average age of 8 years, 
    and three of them being 5, 7, and 10 years old, 
    the age of the fourth child is 10 years. -/
theorem oldest_child_age 
  (total_children : Nat)
  (average_age : ℚ)
  (younger_children_ages : List Nat)
  (h1 : total_children = 4)
  (h2 : average_age = 8)
  (h3 : younger_children_ages = [5, 7, 10])
  : Nat :=
10

theorem oldest_child_age_proof : oldest_child_age 4 8 [5, 7, 10] rfl rfl rfl = 10 := by
  sorry

end oldest_child_age_oldest_child_age_proof_l1599_159966


namespace smallest_separating_degree_l1599_159904

/-- A point on the coordinate plane with a color -/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Bool  -- True for red, False for blue

/-- A set of N points is permissible if their x-coordinates are distinct -/
def isPermissible (points : Finset ColoredPoint) : Prop :=
  ∀ p q : ColoredPoint, p ∈ points → q ∈ points → p ≠ q → p.x ≠ q.x

/-- A polynomial P separates a set of points if no red points are above
    and no blue points below its graph, or vice versa -/
def separates (P : ℝ → ℝ) (points : Finset ColoredPoint) : Prop :=
  (∀ p ∈ points, p.color = true → P p.x ≥ p.y) ∧
  (∀ p ∈ points, p.color = false → P p.x ≤ p.y) ∨
  (∀ p ∈ points, p.color = true → P p.x ≤ p.y) ∧
  (∀ p ∈ points, p.color = false → P p.x ≥ p.y)

/-- The main theorem: For any N ≥ 3, the smallest degree k of a polynomial
    that can separate any permissible set of N points is N-2 -/
theorem smallest_separating_degree (N : ℕ) (h : N ≥ 3) :
  ∃ k : ℕ, (∀ points : Finset ColoredPoint, points.card = N → isPermissible points →
    ∃ P : ℝ → ℝ, (∃ coeffs : Finset ℝ, coeffs.card ≤ k + 1 ∧
      P = fun x ↦ (coeffs.toList.enum.map (fun (i, a) ↦ a * x ^ i)).sum) ∧
    separates P points) ∧
  (∀ k' : ℕ, k' < k →
    ∃ points : Finset ColoredPoint, points.card = N ∧ isPermissible points ∧
    ∀ P : ℝ → ℝ, (∃ coeffs : Finset ℝ, coeffs.card ≤ k' + 1 ∧
      P = fun x ↦ (coeffs.toList.enum.map (fun (i, a) ↦ a * x ^ i)).sum) →
    ¬separates P points) ∧
  k = N - 2 := by
  sorry

end smallest_separating_degree_l1599_159904


namespace new_solutions_introduced_l1599_159914

variables {α : Type*} [LinearOrder α]
variable (x : α)
variable (F₁ F₂ f : α → ℝ)

theorem new_solutions_introduced (h : F₁ x > F₂ x) :
  (f x < 0 ∧ F₁ x < F₂ x) ↔ (f x * F₁ x < f x * F₂ x ∧ ¬(F₁ x > F₂ x)) :=
by sorry

end new_solutions_introduced_l1599_159914


namespace log_sqrt8_512sqrt8_equals_7_l1599_159902

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sqrt8_512sqrt8_equals_7 :
  log (Real.sqrt 8) (512 * Real.sqrt 8) = 7 := by sorry

end log_sqrt8_512sqrt8_equals_7_l1599_159902


namespace tan_alpha_2_implies_l1599_159969

theorem tan_alpha_2_implies (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6/13 ∧
  3 * Real.sin α ^ 2 + 3 * Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 16/5 := by
  sorry

end tan_alpha_2_implies_l1599_159969


namespace smallest_fourth_lucky_number_l1599_159934

theorem smallest_fourth_lucky_number :
  let first_three : List Nat := [68, 24, 85]
  let sum_first_three := first_three.sum
  let sum_digits_first_three := (first_three.map (fun n => n / 10 + n % 10)).sum
  ∀ x : Nat,
    x ≥ 10 ∧ x < 100 →
    (sum_first_three + x) * 1/4 = sum_digits_first_three + x / 10 + x % 10 →
    x ≥ 93 :=
by sorry

end smallest_fourth_lucky_number_l1599_159934


namespace product_of_roots_l1599_159929

theorem product_of_roots (x : ℝ) : 
  (25 * x^2 + 60 * x - 375 = 0) → 
  (∃ r₁ r₂ : ℝ, (25 * r₁^2 + 60 * r₁ - 375 = 0) ∧ 
                (25 * r₂^2 + 60 * r₂ - 375 = 0) ∧ 
                (r₁ * r₂ = -15)) := by
  sorry

end product_of_roots_l1599_159929


namespace quadratic_real_roots_l1599_159993

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0) ↔ m ≤ 1/4 := by
  sorry

end quadratic_real_roots_l1599_159993


namespace theater_ticket_price_l1599_159988

/-- Calculates the ticket price for a theater performance --/
theorem theater_ticket_price
  (capacity : ℕ)
  (fill_rate : ℚ)
  (num_performances : ℕ)
  (total_earnings : ℕ)
  (h1 : capacity = 400)
  (h2 : fill_rate = 4/5)
  (h3 : num_performances = 3)
  (h4 : total_earnings = 28800) :
  (total_earnings : ℚ) / ((capacity : ℚ) * fill_rate * num_performances) = 30 :=
by
  sorry

#check theater_ticket_price

end theater_ticket_price_l1599_159988


namespace total_tickets_is_91_l1599_159939

/-- The total number of tickets needed for Janet's family's amusement park visits -/
def total_tickets : ℕ :=
  let family_size : ℕ := 4
  let adults : ℕ := 2
  let children : ℕ := 2
  let roller_coaster_adult : ℕ := 7
  let roller_coaster_child : ℕ := 5
  let giant_slide_adult : ℕ := 4
  let giant_slide_child : ℕ := 3
  let adult_roller_coaster_rides : ℕ := 3
  let child_roller_coaster_rides : ℕ := 2
  let adult_giant_slide_rides : ℕ := 5
  let child_giant_slide_rides : ℕ := 3

  let roller_coaster_tickets := 
    adults * roller_coaster_adult * adult_roller_coaster_rides +
    children * roller_coaster_child * child_roller_coaster_rides
  
  let giant_slide_tickets :=
    1 * giant_slide_adult * adult_giant_slide_rides +
    1 * giant_slide_child * child_giant_slide_rides

  roller_coaster_tickets + giant_slide_tickets

theorem total_tickets_is_91 : total_tickets = 91 := by
  sorry

end total_tickets_is_91_l1599_159939


namespace line_bisecting_circle_min_value_l1599_159992

/-- Given a line that always bisects the circumference of a circle, 
    prove the minimum value of 1/a + 1/b -/
theorem line_bisecting_circle_min_value (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, 2*a*x - b*y + 2 = 0 → 
    x^2 + y^2 + 2*x - 4*y + 1 = 0 → 
    -- The line bisects the circle (implicit condition)
    True) → 
  (1/a + 1/b) ≥ 4 := by
sorry

end line_bisecting_circle_min_value_l1599_159992


namespace class_size_is_50_l1599_159956

/-- The number of students in class 4(1) -/
def class_size : ℕ := 50

/-- The number of students in the basketball group -/
def basketball_group : ℕ := class_size / 2 + 1

/-- The number of students in the table tennis group -/
def table_tennis_group : ℕ := (class_size - basketball_group) / 2 + 2

/-- The number of students in the chess group -/
def chess_group : ℕ := (class_size - basketball_group - table_tennis_group) / 2 + 3

/-- The number of students in the broadcasting group -/
def broadcasting_group : ℕ := 2

theorem class_size_is_50 :
  class_size = 50 ∧
  basketball_group > class_size / 2 ∧
  table_tennis_group = (class_size - basketball_group) / 2 + 2 ∧
  chess_group = (class_size - basketball_group - table_tennis_group) / 2 + 3 ∧
  broadcasting_group = 2 ∧
  class_size = basketball_group + table_tennis_group + chess_group + broadcasting_group :=
by sorry

end class_size_is_50_l1599_159956


namespace cd_store_problem_l1599_159913

/-- Represents the total number of CDs in the store -/
def total_cds : ℕ := sorry

/-- Represents the price of expensive CDs -/
def expensive_price : ℕ := 10

/-- Represents the price of cheap CDs -/
def cheap_price : ℕ := 5

/-- Represents the proportion of expensive CDs -/
def expensive_proportion : ℚ := 2/5

/-- Represents the proportion of cheap CDs -/
def cheap_proportion : ℚ := 3/5

/-- Represents the proportion of expensive CDs bought by Prince -/
def expensive_bought_proportion : ℚ := 1/2

/-- Represents the total amount spent by Prince -/
def total_spent : ℕ := 1000

theorem cd_store_problem :
  (expensive_proportion * expensive_bought_proportion * (total_cds : ℚ) * expensive_price) +
  (cheap_proportion * (total_cds : ℚ) * cheap_price) = total_spent ∧
  total_cds = 200 := by sorry

end cd_store_problem_l1599_159913


namespace z_range_l1599_159946

theorem z_range (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x + y = x * y) (hxyz : x + y + z = x * y * z) :
  1 < z ∧ z ≤ Real.sqrt 3 := by
  sorry

end z_range_l1599_159946


namespace football_team_right_handed_players_l1599_159979

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 31)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0)
  : total_players - (total_players - throwers) / 3 = 57 := by
  sorry

end football_team_right_handed_players_l1599_159979


namespace arithmetic_sequence_odd_numbers_l1599_159922

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_odd_numbers :
  ∀ n : ℕ, n > 0 → arithmetic_sequence 1 2 n = 2 * n - 1 :=
by
  sorry

end arithmetic_sequence_odd_numbers_l1599_159922


namespace hyperbola_condition_l1599_159984

/-- The equation represents a hyperbola with foci on the x-axis -/
def is_hyperbola_x_axis (k : ℝ) : Prop :=
  ∃ (x y : ℝ → ℝ), ∀ t : ℝ, (x t)^2 / (k + 3) + (y t)^2 / (k + 2) = 1

theorem hyperbola_condition (k : ℝ) :
  is_hyperbola_x_axis k ↔ -3 < k ∧ k < -2 :=
sorry

end hyperbola_condition_l1599_159984


namespace square_root_of_sixteen_l1599_159957

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end square_root_of_sixteen_l1599_159957


namespace abs_f_range_l1599_159903

/-- A function whose range is [-2, 3] -/
def f : ℝ → ℝ :=
  sorry

/-- The range of f is [-2, 3] -/
axiom f_range : Set.range f = Set.Icc (-2) 3

/-- Theorem: If the range of f(x) is [-2, 3], then the range of |f(x)| is [0, 3] -/
theorem abs_f_range :
  Set.range (fun x ↦ |f x|) = Set.Icc 0 3 :=
sorry

end abs_f_range_l1599_159903


namespace milk_replacement_amount_l1599_159945

/-- Represents the amount of milk removed and replaced with water in each operation -/
def x : ℝ := 9

/-- The capacity of the vessel in litres -/
def vessel_capacity : ℝ := 90

/-- The amount of pure milk remaining after the operations in litres -/
def final_pure_milk : ℝ := 72.9

/-- Theorem stating that the amount of milk removed and replaced with water in each operation is correct -/
theorem milk_replacement_amount : 
  vessel_capacity - x - (vessel_capacity - x) * x / vessel_capacity = final_pure_milk := by
  sorry

end milk_replacement_amount_l1599_159945


namespace final_racers_count_l1599_159970

/-- Calculates the number of racers remaining after each elimination round -/
def remaining_racers (initial : ℕ) (first_elim : ℕ) (second_elim_frac : ℚ) (third_elim_frac : ℚ) : ℕ :=
  let after_first := initial - first_elim
  let after_second := after_first - (after_first * second_elim_frac).floor
  (after_second - (after_second * third_elim_frac).floor).toNat

/-- Theorem stating that given the initial conditions, 30 racers remain for the final section -/
theorem final_racers_count :
  remaining_racers 100 10 (1/3) (1/2) = 30 := by
  sorry


end final_racers_count_l1599_159970


namespace aye_aye_friendship_l1599_159935

theorem aye_aye_friendship (n : ℕ) (h : n = 23) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧ 
  (∃ (f : Fin n → Finset (Fin n)), 
    (∀ x, x ∉ f x) ∧
    (∀ x y, y ∈ f x ↔ x ∈ f y) ∧
    (f i).card = k ∧ (f j).card = k) :=
by sorry


end aye_aye_friendship_l1599_159935


namespace vowel_sequences_count_l1599_159912

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The length of each sequence -/
def sequence_length : ℕ := 5

/-- Calculates the number of five-letter sequences containing at least one of each vowel -/
def vowel_sequences : ℕ :=
  sequence_length^num_vowels - 
  (Nat.choose num_vowels 1) * (num_vowels - 1)^sequence_length +
  (Nat.choose num_vowels 2) * (num_vowels - 2)^sequence_length -
  (Nat.choose num_vowels 3) * (num_vowels - 3)^sequence_length +
  (Nat.choose num_vowels 4) * (num_vowels - 4)^sequence_length

theorem vowel_sequences_count : vowel_sequences = 120 := by
  sorry

end vowel_sequences_count_l1599_159912


namespace polynomial_divisibility_l1599_159924

theorem polynomial_divisibility (n : ℕ) (hn : n > 0) :
  ∃ Q : Polynomial ℚ, (n^2 * X^(n+2) - (2*n^2 + 2*n - 1) * X^(n+1) + (n+1)^2 * X^n - X - 1) = (X - 1)^3 * Q := by
  sorry

end polynomial_divisibility_l1599_159924


namespace point_on_same_side_l1599_159931

def sameSideOfLine (p1 p2 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x1 + y1 - 1) * (x2 + y2 - 1) > 0

def referencePt : ℝ × ℝ := (1, 2)

theorem point_on_same_side : 
  sameSideOfLine (-1, 3) referencePt ∧ 
  ¬sameSideOfLine (0, 0) referencePt ∧ 
  ¬sameSideOfLine (-1, 1) referencePt ∧ 
  ¬sameSideOfLine (2, -3) referencePt :=
by sorry

end point_on_same_side_l1599_159931


namespace max_sum_on_circle_l1599_159915

theorem max_sum_on_circle (x y : ℤ) : 
  x > 0 → y > 0 → x^2 + y^2 = 49 → x + y ≤ 7 :=
by sorry

end max_sum_on_circle_l1599_159915


namespace least_subtraction_for_divisibility_l1599_159954

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 4 ∧ (5026 - k) % 5 = 0 ∧ ∀ (m : ℕ), m < k → (5026 - m) % 5 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_l1599_159954


namespace anns_age_is_30_l1599_159985

/-- Represents the ages of Ann and Barbara at different points in time. -/
structure AgeRelation where
  a : ℕ  -- Ann's current age
  b : ℕ  -- Barbara's current age

/-- The condition that the sum of their present ages is 50 years. -/
def sum_of_ages (ages : AgeRelation) : Prop :=
  ages.a + ages.b = 50

/-- The complex age relation described in the problem. -/
def age_relation (ages : AgeRelation) : Prop :=
  ∃ (y : ℕ), 
    ages.b = ages.a / 2 + 2 * y ∧
    ages.a - ages.b = y

/-- The theorem stating that given the conditions, Ann's age is 30 years. -/
theorem anns_age_is_30 (ages : AgeRelation) 
  (h1 : sum_of_ages ages) 
  (h2 : age_relation ages) : 
  ages.a = 30 := by sorry

end anns_age_is_30_l1599_159985


namespace andy_remaining_demerits_l1599_159932

/-- The number of additional demerits Andy can get before being fired -/
def additional_demerits (max_demerits : ℕ) (lateness_instances : ℕ) (demerits_per_lateness : ℕ) (joke_demerits : ℕ) : ℕ :=
  max_demerits - (lateness_instances * demerits_per_lateness + joke_demerits)

/-- Theorem stating that Andy can get 23 more demerits before being fired -/
theorem andy_remaining_demerits :
  additional_demerits 50 6 2 15 = 23 := by
  sorry

end andy_remaining_demerits_l1599_159932


namespace ab_value_l1599_159917

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 := by
  sorry

end ab_value_l1599_159917


namespace triangle_area_l1599_159996

theorem triangle_area (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sin_sum : Real.sin (A + B) = 3/5)
  (h_sin_diff : Real.sin (A - B) = 1/5)
  (h_AB : 3 = 3) :
  (1/2) * 3 * (2 * Real.sqrt 6 - 2) = (6 + 3 * Real.sqrt 6) / 2 := by
sorry

end triangle_area_l1599_159996


namespace a_capital_is_15000_l1599_159965

/-- The amount of money partner a put into the business -/
def a_capital : ℝ := sorry

/-- The amount of money partner b put into the business -/
def b_capital : ℝ := 25000

/-- The total profit of the business -/
def total_profit : ℝ := 9600

/-- The percentage of profit a receives for managing the business -/
def management_fee_percentage : ℝ := 0.1

/-- The total amount a receives -/
def a_total_received : ℝ := 4200

theorem a_capital_is_15000 :
  a_capital = 15000 :=
by
  sorry

#check a_capital_is_15000

end a_capital_is_15000_l1599_159965


namespace logarithmic_function_value_l1599_159964

noncomputable def f (a : ℝ) (x : ℝ) := (a^2 + a - 5) * Real.log x / Real.log a

theorem logarithmic_function_value (a : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (a^2 + a - 5 = 1) →
  f a (1/8) = -3 :=
by sorry

end logarithmic_function_value_l1599_159964


namespace ratio_of_numbers_l1599_159919

theorem ratio_of_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (sum_diff : a + b = 7 * (a - b)) (product : a * b = 50) :
  max a b / min a b = 4 / 3 := by
sorry

end ratio_of_numbers_l1599_159919


namespace f_increasing_interval_f_not_increasing_below_one_l1599_159938

/-- The function f(x) = |x-1| + |x+1| -/
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

/-- The interval of increase for f(x) -/
def interval_of_increase : Set ℝ := { x | x ≥ 1 }

/-- Theorem stating that the interval of increase for f(x) is [1, +∞) -/
theorem f_increasing_interval :
  ∀ x y, x ∈ interval_of_increase → y ∈ interval_of_increase → x < y → f x < f y :=
by sorry

/-- Theorem stating that f(x) is not increasing for x < 1 -/
theorem f_not_increasing_below_one :
  ∃ x y, x < 1 ∧ y < 1 ∧ x < y ∧ f x ≥ f y :=
by sorry

end f_increasing_interval_f_not_increasing_below_one_l1599_159938


namespace prism_with_18_edges_has_8_faces_l1599_159975

/-- The number of faces in a prism given the number of edges -/
def prism_faces (edges : ℕ) : ℕ :=
  (edges / 3) + 2

theorem prism_with_18_edges_has_8_faces :
  prism_faces 18 = 8 := by
  sorry

end prism_with_18_edges_has_8_faces_l1599_159975


namespace local_minimum_at_two_l1599_159911

/-- The function f(x) = x^3 - 12x --/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 12

theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 := by
  sorry

end local_minimum_at_two_l1599_159911


namespace sum_of_integers_l1599_159936

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 7)
  (eq2 : q - r + s = 8)
  (eq3 : r - s + p = 4)
  (eq4 : s - p + q = 3) :
  p + q + r + s = 22 := by
sorry

end sum_of_integers_l1599_159936


namespace corresponding_angles_equal_l1599_159974

theorem corresponding_angles_equal (α β γ : ℝ) :
  α + β + γ = 180 ∧ (180 - α) + β + γ = 180 →
  α = 180 - α ∧ β = β ∧ γ = γ := by
  sorry

end corresponding_angles_equal_l1599_159974
