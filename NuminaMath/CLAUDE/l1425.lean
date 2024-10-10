import Mathlib

namespace units_digit_sum_factorials_9_l1425_142597

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_factorials (n : ℕ) : ℕ :=
  (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_9 :
  units_digit (sum_factorials 9) = 3 := by
  sorry

end units_digit_sum_factorials_9_l1425_142597


namespace least_positive_four_digit_octal_l1425_142561

/-- The number of digits required to represent a positive integer in a given base -/
def numDigits (n : ℕ+) (base : ℕ) : ℕ :=
  Nat.log base n.val + 1

/-- Checks if a number requires at least four digits in base 8 -/
def requiresFourDigitsOctal (n : ℕ+) : Prop :=
  numDigits n 8 ≥ 4

theorem least_positive_four_digit_octal :
  ∃ (n : ℕ+), requiresFourDigitsOctal n ∧
    ∀ (m : ℕ+), m < n → ¬requiresFourDigitsOctal m ∧
    n = 512 := by
  sorry

end least_positive_four_digit_octal_l1425_142561


namespace city_inhabitants_problem_l1425_142536

theorem city_inhabitants_problem :
  ∃ (n : ℕ), 
    (∃ (m : ℕ), n^2 + 100 = m^2 + 1) ∧ 
    (∃ (k : ℕ), n^2 + 200 = k^2) ∧ 
    n = 49 := by
  sorry

end city_inhabitants_problem_l1425_142536


namespace inequality_proof_l1425_142521

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end inequality_proof_l1425_142521


namespace bar_and_line_charts_represent_amount_l1425_142541

-- Define bar charts and line charts as types that can represent data
def BarChart : Type := Unit
def LineChart : Type := Unit

-- Define a property for charts that can represent amount
def CanRepresentAmount (chart : Type) : Prop := True

-- State the theorem
theorem bar_and_line_charts_represent_amount :
  CanRepresentAmount BarChart ∧ CanRepresentAmount LineChart := by
  sorry

end bar_and_line_charts_represent_amount_l1425_142541


namespace four_objects_three_containers_l1425_142547

/-- The number of ways to distribute n distinct objects into k distinct containers --/
def distributionWays (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 4 distinct objects into 3 distinct containers is 81 --/
theorem four_objects_three_containers : distributionWays 4 3 = 81 := by
  sorry

end four_objects_three_containers_l1425_142547


namespace fraction_equality_implies_five_l1425_142508

theorem fraction_equality_implies_five (a b : ℕ) (h : (a + 30) / b = a / (b - 6)) : 
  (a + 30) / b = 5 := by
  sorry

end fraction_equality_implies_five_l1425_142508


namespace dice_product_120_probability_l1425_142533

/-- A function representing a standard die roll --/
def standardDie : ℕ → Prop :=
  λ n => 1 ≤ n ∧ n ≤ 6

/-- The probability of a specific outcome when rolling three dice --/
def tripleRollProb : ℚ := (1 : ℚ) / 216

/-- The number of favorable outcomes --/
def favorableOutcomes : ℕ := 6

/-- The probability that the product of three dice rolls equals 120 --/
theorem dice_product_120_probability :
  (favorableOutcomes : ℚ) * tripleRollProb = (1 : ℚ) / 36 :=
sorry

end dice_product_120_probability_l1425_142533


namespace tim_and_donna_dating_years_l1425_142540

/-- Represents the timeline of Tim and Donna's relationship -/
structure Relationship where
  meetYear : ℕ
  weddingYear : ℕ
  anniversaryYear : ℕ
  yearsBetweenMeetingAndDating : ℕ

/-- Calculate the number of years Tim and Donna dated before marriage -/
def yearsDatingBeforeMarriage (r : Relationship) : ℕ :=
  r.weddingYear - r.meetYear - r.yearsBetweenMeetingAndDating

/-- The main theorem stating that Tim and Donna dated for 3 years before marriage -/
theorem tim_and_donna_dating_years (r : Relationship) 
  (h1 : r.meetYear = 2000)
  (h2 : r.anniversaryYear = 2025)
  (h3 : r.anniversaryYear - r.weddingYear = 20)
  (h4 : r.yearsBetweenMeetingAndDating = 2) : 
  yearsDatingBeforeMarriage r = 3 := by
  sorry


end tim_and_donna_dating_years_l1425_142540


namespace projection_sum_bound_l1425_142504

/-- A segment in the plane represented by its length and angle -/
structure Segment where
  length : ℝ
  angle : ℝ

/-- The theorem statement -/
theorem projection_sum_bound (segments : List Segment) 
  (total_length : (segments.map (λ s => s.length)).sum = 1) :
  ∃ θ : ℝ, (segments.map (λ s => s.length * |Real.cos (θ - s.angle)|)).sum < 2 / Real.pi := by
  sorry

end projection_sum_bound_l1425_142504


namespace angle_B_measure_l1425_142599

-- Define the triangles and angles
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define congruence between triangles
def congruent (t1 t2 : Triangle) : Prop :=
  t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

-- Theorem statement
theorem angle_B_measure (ABC DEF : Triangle) :
  congruent ABC DEF →
  ABC.A = 30 →
  DEF.C = 85 →
  ABC.B = 65 :=
by
  sorry

end angle_B_measure_l1425_142599


namespace some_zims_not_cims_l1425_142565

variable (U : Type) -- Universe set

-- Define the sets
variable (Zim Bim Cim : Set U)

-- Define the conditions
variable (h1 : Zim ⊆ Bim)  -- All Zims are Bims
variable (h2 : ∃ x, x ∈ Bim ∧ x ∉ Cim)  -- Some (at least one) Bims are not Cims

-- Theorem to prove
theorem some_zims_not_cims : ∃ x, x ∈ Zim ∧ x ∉ Cim :=
sorry

end some_zims_not_cims_l1425_142565


namespace abcd_product_l1425_142528

theorem abcd_product (a b c d : ℝ) 
  (ha : a = Real.sqrt (4 + Real.sqrt (5 - a)))
  (hb : b = Real.sqrt (4 + Real.sqrt (5 + b)))
  (hc : c = Real.sqrt (4 - Real.sqrt (5 - c)))
  (hd : d = Real.sqrt (4 - Real.sqrt (5 + d))) :
  a * b * c * d = 11 := by
sorry

end abcd_product_l1425_142528


namespace more_males_than_females_difference_in_population_l1425_142538

theorem more_males_than_females : Int → Int → Int
  | num_males, num_females =>
    num_males - num_females

theorem difference_in_population (num_males num_females : Int) 
  (h1 : num_males = 23) 
  (h2 : num_females = 9) : 
  more_males_than_females num_males num_females = 14 := by
  sorry

end more_males_than_females_difference_in_population_l1425_142538


namespace constant_value_l1425_142584

theorem constant_value (t : ℝ) (x y : ℝ → ℝ) (constant : ℝ) :
  (∀ t, x t = constant - 3 * t) →
  (∀ t, y t = 2 * t - 3) →
  x 0.8 = y 0.8 →
  constant = 1 :=
by
  sorry

end constant_value_l1425_142584


namespace marbles_remaining_example_l1425_142577

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

end marbles_remaining_example_l1425_142577


namespace multiply_is_enlarge_l1425_142570

-- Define the concept of enlarging a number
def enlarge (n : ℕ) (times : ℕ) : ℕ := n * times

-- State the theorem
theorem multiply_is_enlarge :
  ∀ (n : ℕ), 28 * 5 = enlarge 28 5 :=
by
  sorry

end multiply_is_enlarge_l1425_142570


namespace assign_25_to_4_l1425_142512

/-- The number of ways to assign different service providers to children -/
def assignProviders (n m : ℕ) : ℕ :=
  (n - 0) * (n - 1) * (n - 2) * (n - 3)

/-- Theorem: Assigning 25 service providers to 4 children results in 303600 possibilities -/
theorem assign_25_to_4 : assignProviders 25 4 = 303600 := by
  sorry

#eval assignProviders 25 4

end assign_25_to_4_l1425_142512


namespace hall_volume_l1425_142587

/-- The volume of a rectangular hall with given dimensions and area constraint -/
theorem hall_volume (length width : ℝ) (h : length = 15 ∧ width = 12) 
  (area_constraint : 2 * (length * width) = 2 * (length + width) * ((2 * length * width) / (2 * (length + width)))) :
  length * width * ((2 * length * width) / (2 * (length + width))) = 8004 := by
sorry

end hall_volume_l1425_142587


namespace selection_methods_count_l1425_142575

def num_male_students : ℕ := 5
def num_female_students : ℕ := 4
def num_representatives : ℕ := 4
def min_female_representatives : ℕ := 2

theorem selection_methods_count :
  (Finset.sum (Finset.range (num_representatives - min_female_representatives + 1))
    (λ k => Nat.choose num_female_students (min_female_representatives + k) *
            Nat.choose num_male_students (num_representatives - (min_female_representatives + k))))
  = 81 := by
  sorry

end selection_methods_count_l1425_142575


namespace percentage_of_14_to_70_l1425_142557

theorem percentage_of_14_to_70 : ∀ (x : ℚ), x = 14 / 70 * 100 → x = 20 := by
  sorry

end percentage_of_14_to_70_l1425_142557


namespace equal_intercept_line_equation_l1425_142579

/-- A line passing through (1,2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The slope of the line
  k : ℝ
  -- The line passes through (1,2)
  point_condition : 2 = k * (1 - 1) + 2
  -- The line has equal intercepts on both axes
  equal_intercepts : 2 - k = 1 - 2 / k

/-- The equation of the line is either x+y-3=0 or 2x-y=0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, x + y - 3 = 0 ↔ y = l.k * (x - 1) + 2) ∨
  (∀ x y, 2 * x - y = 0 ↔ y = l.k * (x - 1) + 2) :=
sorry

end equal_intercept_line_equation_l1425_142579


namespace gift_distribution_theorem_l1425_142581

/-- The number of ways to choose and distribute gifts -/
def giftDistributionWays (totalGifts classmates chosenGifts : ℕ) : ℕ :=
  (totalGifts.choose chosenGifts) * chosenGifts.factorial

/-- Theorem stating that choosing 3 out of 5 gifts and distributing to 3 classmates results in 60 ways -/
theorem gift_distribution_theorem :
  giftDistributionWays 5 3 3 = 60 := by
  sorry

end gift_distribution_theorem_l1425_142581


namespace area_common_part_squares_l1425_142590

/-- The area of the common part of two squares -/
theorem area_common_part_squares (small_side : ℝ) (large_side : ℝ) 
  (h1 : small_side = 1)
  (h2 : large_side = 4 * small_side)
  (h3 : small_side > 0)
  (h4 : large_side > small_side) : 
  large_side^2 - (1/2 * small_side^2 + 1/2 * large_side^2) = 13.5 := by
  sorry

end area_common_part_squares_l1425_142590


namespace smallest_multiple_ten_satisfies_ten_is_smallest_l1425_142544

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 500 ∣ (450 * x) → x ≥ 10 := by
  sorry

theorem ten_satisfies : 500 ∣ (450 * 10) := by
  sorry

theorem ten_is_smallest : ∀ y : ℕ, y > 0 ∧ 500 ∣ (450 * y) → y ≥ 10 := by
  sorry

end smallest_multiple_ten_satisfies_ten_is_smallest_l1425_142544


namespace total_dress_designs_l1425_142539

/-- The number of fabric colors available --/
def num_colors : ℕ := 5

/-- The number of patterns available --/
def num_patterns : ℕ := 6

/-- The number of fabric types available --/
def num_fabric_types : ℕ := 2

/-- Theorem stating the total number of possible dress designs --/
theorem total_dress_designs : num_colors * num_patterns * num_fabric_types = 60 := by
  sorry

end total_dress_designs_l1425_142539


namespace wednesday_sales_l1425_142527

/-- Represents the number of crates of eggs sold on each day of the week -/
structure EggSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Calculates the total number of crates sold over 4 days -/
def total_sales (sales : EggSales) : ℕ :=
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday

/-- Theorem stating the number of crates sold on Wednesday -/
theorem wednesday_sales (sales : EggSales) 
  (h1 : sales.monday = 5)
  (h2 : sales.tuesday = 2 * sales.monday)
  (h3 : sales.thursday = sales.tuesday / 2)
  (h4 : total_sales sales = 28) :
  sales.wednesday = 8 := by
  sorry

end wednesday_sales_l1425_142527


namespace ice_cream_fraction_l1425_142516

theorem ice_cream_fraction (initial_amount : ℚ) (lunch_cost : ℚ) (ice_cream_cost : ℚ) : 
  initial_amount = 30 →
  lunch_cost = 10 →
  ice_cream_cost = 5 →
  ice_cream_cost / (initial_amount - lunch_cost) = 1 / 4 :=
by sorry

end ice_cream_fraction_l1425_142516


namespace set_problems_l1425_142518

def U : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {1, 4}

def A (x : ℕ) : Set ℕ := {1, 2, x^2}

theorem set_problems (x : ℕ) (hx : x ∈ U) :
  (U \ B = {2, 3}) ∧ 
  (A x ∩ B = B → x = 1) ∧
  ¬∃ (y : ℕ), y ∈ U ∧ A y ∪ B = U :=
by sorry

end set_problems_l1425_142518


namespace vector_inequality_l1425_142592

theorem vector_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end vector_inequality_l1425_142592


namespace special_quadrilateral_is_kite_l1425_142502

/-- A quadrilateral with perpendicular diagonals and equal adjacent sides, but not all sides equal -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perpendicular_diagonals : Bool
  /-- Each pair of adjacent sides is equal -/
  adjacent_sides_equal : Bool
  /-- Not all sides are equal -/
  not_all_sides_equal : Bool

/-- Definition of a kite -/
def is_kite (q : SpecialQuadrilateral) : Prop :=
  q.perpendicular_diagonals ∧ q.adjacent_sides_equal ∧ q.not_all_sides_equal

/-- Theorem stating that the given quadrilateral is a kite -/
theorem special_quadrilateral_is_kite (q : SpecialQuadrilateral) 
  (h1 : q.perpendicular_diagonals = true) 
  (h2 : q.adjacent_sides_equal = true) 
  (h3 : q.not_all_sides_equal = true) : 
  is_kite q :=
sorry

end special_quadrilateral_is_kite_l1425_142502


namespace oldest_babysat_current_age_l1425_142537

/- Define the parameters of the problem -/
def jane_start_age : ℕ := 18
def jane_current_age : ℕ := 34
def years_since_stopped : ℕ := 10

/- Define the function to calculate the maximum age of a child Jane could baby-sit at a given age -/
def max_child_age (jane_age : ℕ) : ℕ :=
  jane_age / 2

/- Theorem statement -/
theorem oldest_babysat_current_age :
  let jane_stop_age : ℕ := jane_current_age - years_since_stopped
  let max_child_age_when_stopped : ℕ := max_child_age jane_stop_age
  let oldest_babysat_age : ℕ := max_child_age_when_stopped + years_since_stopped
  oldest_babysat_age = 22 := by sorry

end oldest_babysat_current_age_l1425_142537


namespace complement_intersection_theorem_l1425_142510

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 2, 3, 4}

-- Define set B
def B : Set Nat := {1, 3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {2, 4, 5, 6} := by
  sorry

end complement_intersection_theorem_l1425_142510


namespace power_of_81_l1425_142589

theorem power_of_81 : (81 : ℝ) ^ (5/4) = 243 := by
  sorry

end power_of_81_l1425_142589


namespace max_handshakes_l1425_142513

def number_of_people : ℕ := 30

theorem max_handshakes (n : ℕ) (h : n = number_of_people) : 
  Nat.choose n 2 = 435 := by
  sorry

end max_handshakes_l1425_142513


namespace map_distance_between_mountains_l1425_142525

/-- Given a map with a known scale factor, this theorem proves that the distance
    between two mountains on the map is 312 inches, given their actual distance
    and a reference point. -/
theorem map_distance_between_mountains
  (actual_distance : ℝ)
  (map_reference : ℝ)
  (actual_reference : ℝ)
  (h1 : actual_distance = 136)
  (h2 : map_reference = 28)
  (h3 : actual_reference = 12.205128205128204)
  : (actual_distance / (actual_reference / map_reference)) = 312 := by
  sorry

end map_distance_between_mountains_l1425_142525


namespace even_function_derivative_at_zero_l1425_142522

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_derivative_at_zero
  (f : ℝ → ℝ)
  (hf : EvenFunction f)
  (hf' : Differentiable ℝ f) :
  deriv f 0 = 0 :=
sorry

end even_function_derivative_at_zero_l1425_142522


namespace exists_2008_acquaintances_l1425_142506

/-- Represents a gathering of people -/
structure Gathering where
  people : Finset Nat
  acquaintances : Nat → Finset Nat
  no_common_acquaintances : ∀ x y, x ∈ people → y ∈ people →
    (acquaintances x).card = (acquaintances y).card →
    (acquaintances x ∩ acquaintances y).card ≤ 1

/-- Main theorem: If there's someone with at least 2008 acquaintances,
    then there's someone with exactly 2008 acquaintances -/
theorem exists_2008_acquaintances (g : Gathering) :
  (∃ x ∈ g.people, (g.acquaintances x).card ≥ 2008) →
  (∃ y ∈ g.people, (g.acquaintances y).card = 2008) := by
  sorry

end exists_2008_acquaintances_l1425_142506


namespace monotonic_decreasing_interval_l1425_142593

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem monotonic_decreasing_interval :
  ∀ x y : ℝ, x < y → x < 0 → f x > f y :=
by sorry

end monotonic_decreasing_interval_l1425_142593


namespace functional_equation_solution_l1425_142515

/-- A function satisfying f(a+b) = f(a) * f(b) for all real a and b, 
    and f(x) > 0 for all real x, with f(1) = 1/3 -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, f (a + b) = f a * f b) ∧ 
  (∀ x : ℝ, f x > 0) ∧
  (f 1 = 1/3)

/-- If f satisfies the functional equation, then f(-2) = 9 -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : FunctionalEquation f) : f (-2) = 9 := by
  sorry

end functional_equation_solution_l1425_142515


namespace third_number_tenth_row_l1425_142588

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of integers in the first n rows of the triangular array -/
def numbers_in_rows (n : ℕ) : ℕ := triangular_number (n - 1)

/-- The kth number from the left on the nth row of the triangular array -/
def number_at_position (n : ℕ) (k : ℕ) : ℕ := 
  numbers_in_rows n + k

theorem third_number_tenth_row : 
  number_at_position 10 3 = 48 := by sorry

end third_number_tenth_row_l1425_142588


namespace sum_floor_equals_n_l1425_142550

/-- For any natural number n, the sum of floor((n+2^k)/(2^(k+1))) from k=0 to infinity equals n -/
theorem sum_floor_equals_n (n : ℕ) :
  (∑' k, ⌊(n + 2^k : ℝ) / (2^(k+1) : ℝ)⌋) = n :=
sorry

end sum_floor_equals_n_l1425_142550


namespace r₂_bound_r₂_bound_tight_l1425_142594

/-- A function f(x) = x² - r₂x + r₃ -/
def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂*x + r₃

/-- Sequence g_n defined recursively -/
def g (r₂ r₃ : ℝ) : ℕ → ℝ
| 0 => 0
| n + 1 => f r₂ r₃ (g r₂ r₃ n)

/-- The statement that needs to be proved -/
theorem r₂_bound (r₂ r₃ : ℝ) :
  (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i+1) ∧ g r₂ r₃ (2*i+1) > g r₂ r₃ (2*i+2)) →
  (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i+1) > g r₂ r₃ i) →
  (∀ M : ℝ, ∃ n : ℕ, g r₂ r₃ n > M) →
  abs r₂ ≥ 2 :=
by sorry

/-- The bound is tight -/
theorem r₂_bound_tight : ∀ ε > 0, ∃ r₂ r₃ : ℝ,
  (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i+1) ∧ g r₂ r₃ (2*i+1) > g r₂ r₃ (2*i+2)) ∧
  (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i+1) > g r₂ r₃ i) ∧
  (∀ M : ℝ, ∃ n : ℕ, g r₂ r₃ n > M) ∧
  abs r₂ < 2 + ε :=
by sorry

end r₂_bound_r₂_bound_tight_l1425_142594


namespace fraction_sum_equality_l1425_142531

theorem fraction_sum_equality : (3 : ℚ) / 10 + 5 / 100 - 1 / 1000 = 349 / 1000 := by
  sorry

end fraction_sum_equality_l1425_142531


namespace bike_journey_l1425_142523

theorem bike_journey (v d : ℝ) 
  (h1 : d / (v - 4) - d / v = 1.2)
  (h2 : d / v - d / (v + 4) = 2) :
  d = 160 := by
  sorry

end bike_journey_l1425_142523


namespace sum_of_sign_ratios_l1425_142503

theorem sum_of_sign_ratios (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a / |a| + b / |b| + (a * b) / |a * b| = 3 ∨ a / |a| + b / |b| + (a * b) / |a * b| = -1 := by
  sorry

end sum_of_sign_ratios_l1425_142503


namespace expression_evaluation_l1425_142598

theorem expression_evaluation :
  let d : ℕ := 4
  (d^d - d*(d-2)^d + d^2)^(d-1) = 9004736 := by sorry

end expression_evaluation_l1425_142598


namespace quadratic_minimum_value_l1425_142569

theorem quadratic_minimum_value (x m : ℝ) : 
  (∀ x, x^2 - 4*x + m ≥ 4) → m = 8 := by
  sorry

end quadratic_minimum_value_l1425_142569


namespace light_beam_reflection_l1425_142543

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given two points, returns the line passing through them -/
def line_through_points (p1 p2 : Point) : Line :=
  { a := p1.y - p2.y
    b := p2.x - p1.x
    c := p1.x * p2.y - p2.x * p1.y }

/-- Checks if a point lies on a given line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The x-axis -/
def x_axis : Line :=
  { a := 0, b := 1, c := 0 }

/-- Reflects a point across the x-axis -/
def reflect_point_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem light_beam_reflection (M N : Point) 
    (h1 : M = { x := 4, y := 5 })
    (h2 : N = { x := 2, y := 0 })
    (h3 : point_on_line N x_axis) :
  ∃ (l : Line), 
    l = { a := 5, b := -2, c := -10 } ∧ 
    point_on_line M l ∧ 
    point_on_line N l ∧
    point_on_line (reflect_point_x_axis M) l :=
  sorry

end light_beam_reflection_l1425_142543


namespace solve_equation_l1425_142578

theorem solve_equation (x : ℝ) : (x / 5) + 3 = 4 → x = 5 := by
  sorry

end solve_equation_l1425_142578


namespace fraction_squared_l1425_142572

theorem fraction_squared (x : ℚ) : x^2 = 0.0625 → x = 0.25 := by
  sorry

end fraction_squared_l1425_142572


namespace equation_solutions_l1425_142519

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -1 ∧ 
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solutions_l1425_142519


namespace quadratic_root_difference_l1425_142542

theorem quadratic_root_difference (p : ℝ) : 
  p > 0 → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + 1 = 0 ∧ 
    x₂^2 + p*x₂ + 1 = 0 ∧ 
    |x₁ - x₂| = 1) → 
  p = Real.sqrt 5 := by
sorry

end quadratic_root_difference_l1425_142542


namespace a_range_l1425_142534

-- Define the function f(x) piecewise
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x / Real.log a
  else (6 - a) * x - 4 * a

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) →
  1 < a ∧ a < 6 := by
  sorry

end a_range_l1425_142534


namespace inequality_proof_l1425_142501

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) : 
  a / (a + 2*b)^(1/3) + b / (b + 2*c)^(1/3) + c / (c + 2*a)^(1/3) ≥ 1 := by
  sorry

end inequality_proof_l1425_142501


namespace limit_proof_l1425_142546

def a_n (n : ℕ) : ℚ := (7 * n - 1) / (n + 1)

theorem limit_proof (ε : ℚ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |a_n n - 7| < ε := by
  sorry

end limit_proof_l1425_142546


namespace smallest_constant_inequality_l1425_142526

theorem smallest_constant_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hx₃ : x₃ > 0) (hx₄ : x₄ > 0) (hx₅ : x₅ > 0) :
  ∃ C : ℝ, C = 5^15 ∧ 
  (∀ D : ℝ, D < C → ∃ y₁ y₂ y₃ y₄ y₅ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ y₄ > 0 ∧ y₅ > 0 ∧
    D * (y₁^2005 + y₂^2005 + y₃^2005 + y₄^2005 + y₅^2005) < 
    y₁*y₂*y₃*y₄*y₅ * (y₁^125 + y₂^125 + y₃^125 + y₄^125 + y₅^125)^16) ∧
  C * (x₁^2005 + x₂^2005 + x₃^2005 + x₄^2005 + x₅^2005) ≥ 
  x₁*x₂*x₃*x₄*x₅ * (x₁^125 + x₂^125 + x₃^125 + x₄^125 + x₅^125)^16 := by
sorry

end smallest_constant_inequality_l1425_142526


namespace product_of_sums_and_differences_l1425_142566

theorem product_of_sums_and_differences (P Q R S : ℝ) : 
  P = Real.sqrt 2010 + Real.sqrt 2011 →
  Q = -(Real.sqrt 2010) - Real.sqrt 2011 →
  R = Real.sqrt 2010 - Real.sqrt 2011 →
  S = Real.sqrt 2011 - Real.sqrt 2010 →
  P * Q * R * S = 1 := by
  sorry

end product_of_sums_and_differences_l1425_142566


namespace smallest_positive_value_l1425_142568

theorem smallest_positive_value : 
  let S : Set ℝ := {12 - 4 * Real.sqrt 7, 4 * Real.sqrt 7 - 12, 25 - 6 * Real.sqrt 19, 65 - 15 * Real.sqrt 17, 15 * Real.sqrt 17 - 65}
  ∀ x ∈ S, x > 0 → 12 - 4 * Real.sqrt 7 ≤ x :=
by sorry

end smallest_positive_value_l1425_142568


namespace intersection_complement_A_and_B_l1425_142563

def A : Set ℝ := {x : ℝ | |x| > 1}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem intersection_complement_A_and_B :
  (Aᶜ ∩ B) = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end intersection_complement_A_and_B_l1425_142563


namespace triangle_base_length_l1425_142595

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- Perimeter of the triangle -/
  perimeter : ℝ
  /-- Length of the segment of the tangent to the inscribed circle, drawn parallel to the base and contained between the sides of the triangle -/
  tangent_segment : ℝ

/-- Theorem stating that for a triangle with perimeter 20 cm and an inscribed circle, 
    if the segment of the tangent to the circle drawn parallel to the base and 
    contained between the sides of the triangle is 2.4 cm, 
    then the base of the triangle is either 4 cm or 6 cm -/
theorem triangle_base_length (t : TriangleWithInscribedCircle) 
  (h_perimeter : t.perimeter = 20)
  (h_tangent : t.tangent_segment = 2.4) :
  ∃ (base : ℝ), (base = 4 ∨ base = 6) ∧ 
  (∃ (side1 side2 : ℝ), side1 + side2 + base = t.perimeter) :=
sorry

end triangle_base_length_l1425_142595


namespace weight_loss_challenge_l1425_142552

theorem weight_loss_challenge (initial_loss : ℝ) (measured_loss : ℝ) (clothes_addition : ℝ) :
  initial_loss = 0.14 →
  measured_loss = 0.1228 →
  (1 - measured_loss) * (1 - initial_loss) = 1 + clothes_addition →
  clothes_addition = 0.02 := by
sorry

end weight_loss_challenge_l1425_142552


namespace f_sum_constant_l1425_142562

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem f_sum_constant (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 2 / 2 := by
  sorry

end f_sum_constant_l1425_142562


namespace pens_left_for_lenny_l1425_142529

def total_pens : ℕ := 75 * 15

def friends_percentage : ℚ := 30 / 100
def classmates_percentage : ℚ := 20 / 100
def coworkers_percentage : ℚ := 25 / 100
def neighbors_percentage : ℚ := 15 / 100

def pens_after_friends : ℕ := total_pens - (Nat.floor (friends_percentage * total_pens))
def pens_after_classmates : ℕ := pens_after_friends - (Nat.floor (classmates_percentage * pens_after_friends))
def pens_after_coworkers : ℕ := pens_after_classmates - (Nat.floor (coworkers_percentage * pens_after_classmates))
def pens_after_neighbors : ℕ := pens_after_coworkers - (Nat.floor (neighbors_percentage * pens_after_coworkers))

theorem pens_left_for_lenny : pens_after_neighbors = 403 := by
  sorry

end pens_left_for_lenny_l1425_142529


namespace f_one_equals_neg_log_four_l1425_142553

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≤ 0 then -x * Real.log (3 - x) else -(-x * Real.log (3 + x))

-- State the theorem
theorem f_one_equals_neg_log_four :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≤ 0, f x = -x * Real.log (3 - x)) →  -- definition for x ≤ 0
  f 1 = -Real.log 4 := by
sorry

end f_one_equals_neg_log_four_l1425_142553


namespace conditional_equivalence_l1425_142586

theorem conditional_equivalence (R S : Prop) :
  (¬R → S) ↔ (¬S → R) := by sorry

end conditional_equivalence_l1425_142586


namespace train_crossing_time_l1425_142530

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 50 →
  train_speed_kmh = 60 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 3 := by
  sorry

end train_crossing_time_l1425_142530


namespace fritz_money_l1425_142532

theorem fritz_money (fritz sean rick : ℝ) : 
  sean = fritz / 2 + 4 →
  rick = 3 * sean →
  rick + sean = 96 →
  fritz = 40 := by
sorry

end fritz_money_l1425_142532


namespace optimal_optimism_coefficient_l1425_142511

theorem optimal_optimism_coefficient 
  (a b c x : ℝ) 
  (h1 : b > a) 
  (h2 : 0 < x ∧ x < 1) 
  (h3 : c = a + x * (b - a)) 
  (h4 : (c - a)^2 = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 := by
sorry

end optimal_optimism_coefficient_l1425_142511


namespace sphere_volume_derivative_l1425_142573

noncomputable section

-- Define the volume function for a sphere
def sphere_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

-- Define the surface area function for a sphere
def sphere_surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

-- State the theorem
theorem sphere_volume_derivative (R : ℝ) (h : R > 0) :
  deriv sphere_volume R = sphere_surface_area R := by
  sorry

end

end sphere_volume_derivative_l1425_142573


namespace jens_birds_multiple_l1425_142500

theorem jens_birds_multiple (ducks chickens total_birds M : ℕ) : 
  ducks = 150 →
  total_birds = 185 →
  ducks = M * chickens + 10 →
  total_birds = ducks + chickens →
  M = 4 := by
sorry

end jens_birds_multiple_l1425_142500


namespace two_x_minus_one_gt_zero_is_linear_inequality_l1425_142596

/-- Definition of a linear inequality in one variable -/
def is_linear_inequality_one_var (f : ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x ↔ a * x + b > 0 ∨ a * x + b < 0 ∨ a * x + b = 0

/-- The inequality 2x - 1 > 0 is a linear inequality in one variable -/
theorem two_x_minus_one_gt_zero_is_linear_inequality :
  is_linear_inequality_one_var (λ x : ℝ => 2 * x - 1 > 0) :=
sorry

end two_x_minus_one_gt_zero_is_linear_inequality_l1425_142596


namespace third_side_length_l1425_142582

/-- Two similar triangles with given side lengths -/
structure SimilarTriangles where
  -- Larger triangle
  a : ℝ
  b : ℝ
  c : ℝ
  angle : ℝ
  -- Smaller triangle
  d : ℝ
  e : ℝ
  -- Conditions
  ha : a = 16
  hb : b = 20
  hc : c = 24
  hangle : angle = 30 * π / 180
  hd : d = 8
  he : e = 12
  -- Similarity condition
  similar : a / d = b / e

/-- The third side of the smaller triangle is 12 cm -/
theorem third_side_length (t : SimilarTriangles) : t.c / t.d = 12 := by
  sorry

end third_side_length_l1425_142582


namespace box_volume_increase_l1425_142559

theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 4860)
  (hs : 2 * (l * w + w * h + l * h) = 1860)
  (he : 4 * (l + w + h) = 224) :
  (l + 2) * (w + 3) * (h + 1) = 5964 := by
  sorry

end box_volume_increase_l1425_142559


namespace symmetric_point_reciprocal_function_l1425_142564

theorem symmetric_point_reciprocal_function (k : ℝ) : 
  let B : ℝ × ℝ := (Real.cos (π / 3), -Real.sqrt 3)
  let A : ℝ × ℝ := (B.1, -B.2)
  (A.2 = k / A.1) → k = Real.sqrt 3 / 2 := by
sorry

end symmetric_point_reciprocal_function_l1425_142564


namespace parabola_focus_coordinates_l1425_142509

/-- Given a parabola with equation y = (1/m)x^2 where m < 0, 
    its focus has coordinates (0, m/4) -/
theorem parabola_focus_coordinates (m : ℝ) (hm : m < 0) :
  let parabola := {(x, y) : ℝ × ℝ | y = (1/m) * x^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (0, m/4) := by
  sorry

end parabola_focus_coordinates_l1425_142509


namespace tessellation_with_squares_and_triangles_l1425_142545

theorem tessellation_with_squares_and_triangles :
  ∀ m n : ℕ,
  (60 * m + 90 * n = 360) →
  (m = 3 ∧ n = 2) :=
by
  sorry

end tessellation_with_squares_and_triangles_l1425_142545


namespace functional_equation_solution_l1425_142558

/-- The functional equation that f must satisfy for all real a, b, c -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, (f a - f b) * (f b - f c) * (f c - f a) = f (a*b^2 + b*c^2 + c*a^2) - f (a^2*b + b^2*c + c^2*a)

/-- The set of possible functions that satisfy the equation -/
def PossibleFunctions (f : ℝ → ℝ) : Prop :=
  (∃ α β : ℝ, α ∈ ({-1, 0, 1} : Set ℝ) ∧ (∀ x, f x = α * x + β)) ∨
  (∃ α β : ℝ, α ∈ ({-1, 0, 1} : Set ℝ) ∧ (∀ x, f x = α * x^3 + β))

/-- The main theorem stating that any function satisfying the equation must be of the specified form -/
theorem functional_equation_solution (f : ℝ → ℝ) :
  FunctionalEquation f → PossibleFunctions f := by
  sorry

end functional_equation_solution_l1425_142558


namespace quarters_found_l1425_142520

/-- The number of quarters Alyssa found in her couch -/
def num_quarters : ℕ := sorry

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The number of pennies Alyssa found -/
def num_pennies : ℕ := 7

/-- The total amount Alyssa found in cents -/
def total_amount : ℕ := 307

theorem quarters_found :
  num_quarters * quarter_value + num_pennies * penny_value = total_amount ∧
  num_quarters = 12 := by sorry

end quarters_found_l1425_142520


namespace implication_proof_l1425_142549

theorem implication_proof (p q r : Prop) : 
  ((p ∧ ¬q ∧ r) → ((p → q) → r)) ∧
  ((¬p ∧ ¬q ∧ r) → ((p → q) → r)) ∧
  ((p ∧ ¬q ∧ ¬r) → ((p → q) → r)) ∧
  ((¬p ∧ q ∧ r) → ((p → q) → r)) := by
  sorry

end implication_proof_l1425_142549


namespace ned_gave_away_13_games_l1425_142535

/-- The number of games Ned originally had -/
def original_games : ℕ := 19

/-- The number of games Ned currently has -/
def current_games : ℕ := 6

/-- The number of games Ned gave away -/
def games_given_away : ℕ := original_games - current_games

theorem ned_gave_away_13_games : games_given_away = 13 := by
  sorry

end ned_gave_away_13_games_l1425_142535


namespace new_student_weight_l1425_142583

/-- Given a group of students and their weights, calculate the weight of a new student
    that changes the average weight of the group. -/
theorem new_student_weight
  (initial_count : ℕ)
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (h1 : initial_count = 19)
  (h2 : initial_avg = 15)
  (h3 : new_avg = 14.8) :
  (initial_count + 1) * new_avg - initial_count * initial_avg = 11 :=
by sorry

end new_student_weight_l1425_142583


namespace multiplication_of_powers_l1425_142548

theorem multiplication_of_powers (a : ℝ) : a * a^2 = a^3 := by
  sorry

end multiplication_of_powers_l1425_142548


namespace point_P_satisfies_conditions_l1425_142560

def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (-9, 2)
def P : ℝ × ℝ := (-3, -1)

def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def lies_on_line (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

theorem point_P_satisfies_conditions :
  lies_on_line A B P ∧ vector A P = (1/2 : ℝ) • vector A B := by sorry

end point_P_satisfies_conditions_l1425_142560


namespace cube_side_length_l1425_142505

theorem cube_side_length (volume : ℝ) (x : ℝ) : volume = 8 → x^3 = volume → x = 2 := by
  sorry

end cube_side_length_l1425_142505


namespace min_value_and_inequality_l1425_142507

theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ((2 / a + 8 / b) ≥ 9) ∧
  (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 2 / a' + 8 / b' = 9 ∧ a' = 2 / 3 ∧ b' = 4 / 3) ∧
  (a^2 + b^2 ≥ 2) :=
by sorry

end min_value_and_inequality_l1425_142507


namespace allen_age_difference_l1425_142514

/-- Allen's age problem -/
theorem allen_age_difference :
  ∀ (allen_age mother_age : ℕ),
  mother_age = 30 →
  allen_age + 3 + mother_age + 3 = 41 →
  mother_age - allen_age = 25 :=
by sorry

end allen_age_difference_l1425_142514


namespace exists_central_island_l1425_142571

/-- A type representing the islands -/
def Island : Type := ℕ

/-- A structure representing the City of Islands -/
structure CityOfIslands (n : ℕ) where
  /-- The set of islands -/
  islands : Finset Island
  /-- The number of islands is n -/
  island_count : islands.card = n
  /-- Connectivity relation between islands -/
  connected : Island → Island → Prop
  /-- Any two islands are connected (directly or indirectly) -/
  all_connected : ∀ (a b : Island), a ∈ islands → b ∈ islands → connected a b
  /-- The special connectivity property for four islands -/
  four_island_property : ∀ (a b c d : Island), 
    a ∈ islands → b ∈ islands → c ∈ islands → d ∈ islands →
    connected a b → connected b c → connected c d →
    (connected a c ∨ connected b d)

/-- The main theorem: there exists an island connected to all others -/
theorem exists_central_island {n : ℕ} (h : n ≥ 1) (city : CityOfIslands n) : 
  ∃ (central : Island), central ∈ city.islands ∧ 
    ∀ (other : Island), other ∈ city.islands → city.connected central other :=
sorry

end exists_central_island_l1425_142571


namespace bookmarked_pages_march_end_l1425_142551

/-- Represents the number of bookmarked pages at the end of a month -/
def bookmarked_pages_at_month_end (
  pages_per_day : ℕ
) (initial_pages : ℕ) (days_in_month : ℕ) : ℕ :=
  initial_pages + pages_per_day * days_in_month

/-- Theorem: Given the conditions, prove that the total bookmarked pages at the end of March is 1330 -/
theorem bookmarked_pages_march_end :
  bookmarked_pages_at_month_end 30 400 31 = 1330 := by
  sorry

end bookmarked_pages_march_end_l1425_142551


namespace equation_solutions_count_l1425_142574

theorem equation_solutions_count :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (m n : ℤ), (m, n) ∈ s ↔ m^4 + 8*n^2 + 425 = n^4 + 42*m^2) ∧
    s.card = 16 :=
sorry

end equation_solutions_count_l1425_142574


namespace binomial_expansion_sum_l1425_142517

theorem binomial_expansion_sum : 
  let f : ℕ → ℕ → ℕ := λ m n => (Nat.choose 6 m) * (Nat.choose 4 n)
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by sorry

end binomial_expansion_sum_l1425_142517


namespace cube_tetrahedrons_l1425_142556

/-- A cube is a three-dimensional shape with 8 vertices. -/
structure Cube where
  vertices : Finset (Fin 8)
  vertex_count : vertices.card = 8

/-- A tetrahedron is a three-dimensional shape with 4 vertices. -/
structure Tetrahedron where
  vertices : Finset (Fin 8)
  vertex_count : vertices.card = 4

/-- The number of ways to choose 4 vertices from 8 vertices. -/
def total_choices : ℕ := Nat.choose 8 4

/-- The number of sets of 4 vertices that cannot form a tetrahedron. -/
def invalid_choices : ℕ := 12

/-- The function that calculates the number of valid tetrahedrons. -/
def valid_tetrahedrons (c : Cube) : ℕ := total_choices - invalid_choices

/-- Theorem: The number of distinct tetrahedrons that can be formed using the vertices of a cube is 58. -/
theorem cube_tetrahedrons (c : Cube) : valid_tetrahedrons c = 58 := by
  sorry

end cube_tetrahedrons_l1425_142556


namespace congruence_problem_l1425_142591

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 18 = 3 → (3 * x + 14) % 18 = 14 := by
  sorry

end congruence_problem_l1425_142591


namespace average_score_two_classes_l1425_142576

theorem average_score_two_classes (students1 students2 : ℕ) (avg1 avg2 : ℚ) :
  students1 = 20 →
  students2 = 30 →
  avg1 = 80 →
  avg2 = 70 →
  (students1 * avg1 + students2 * avg2) / (students1 + students2 : ℚ) = 74 :=
by sorry

end average_score_two_classes_l1425_142576


namespace new_distance_segment_l1425_142555

/-- New distance function between two points -/
def new_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₂ - x₁| + |y₂ - y₁|

/-- Predicate to check if a point is on a line segment -/
def on_segment (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁)

/-- Theorem: If C is on segment AB, then |AC| + |BC| = |AB| -/
theorem new_distance_segment (x₁ y₁ x₂ y₂ x y : ℝ) :
  on_segment x₁ y₁ x₂ y₂ x y →
  new_distance x₁ y₁ x y + new_distance x y x₂ y₂ = new_distance x₁ y₁ x₂ y₂ :=
by sorry

end new_distance_segment_l1425_142555


namespace min_cost_closed_chain_l1425_142554

/-- Represents the cost in cents to separate one link -/
def separation_cost : ℕ := 1

/-- Represents the cost in cents to attach one link -/
def attachment_cost : ℕ := 2

/-- Represents the number of pieces in the gold chain -/
def num_pieces : ℕ := 13

/-- Represents the number of links in each piece of the chain -/
def links_per_piece : ℕ := 80

/-- Calculates the total cost to separate and reattach one link -/
def link_operation_cost : ℕ := separation_cost + attachment_cost

/-- Theorem stating the minimum cost to form a closed chain -/
theorem min_cost_closed_chain : 
  ∃ (cost : ℕ), cost = (num_pieces - 1) * link_operation_cost ∧ 
  ∀ (other_cost : ℕ), other_cost ≥ cost := by sorry

end min_cost_closed_chain_l1425_142554


namespace circle_symmetry_implies_m_equals_one_l1425_142585

/-- A circle with equation x^2 + y^2 + 2x - 4y = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 = 0}

/-- A line with equation 3x + y + m = 0 -/
def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3*p.1 + p.2 + m = 0}

/-- The center of a circle -/
def center (c : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Symmetry of a circle about a line -/
def isSymmetric (c : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop := sorry

theorem circle_symmetry_implies_m_equals_one :
  isSymmetric Circle (Line m) → m = 1 := by
  sorry

end circle_symmetry_implies_m_equals_one_l1425_142585


namespace election_winner_votes_l1425_142524

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (runner_up_percentage : ℚ) 
  (other_candidates_percentage : ℚ) 
  (invalid_votes_percentage : ℚ) 
  (vote_difference : ℕ) :
  winner_percentage = 48/100 →
  runner_up_percentage = 34/100 →
  other_candidates_percentage = 15/100 →
  invalid_votes_percentage = 3/100 →
  (winner_percentage - runner_up_percentage) * total_votes = vote_difference →
  vote_difference = 2112 →
  winner_percentage * total_votes = 7241 :=
sorry

end election_winner_votes_l1425_142524


namespace star_square_sum_l1425_142567

/-- The ★ operation for real numbers -/
def star (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating that (x + y)^2 ★ (y + x)^2 = 4(x + y)^4 -/
theorem star_square_sum (x y : ℝ) : star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end star_square_sum_l1425_142567


namespace single_point_implies_d_eq_seven_l1425_142580

/-- The equation of the graph -/
def equation (x y d : ℝ) : ℝ := 3 * x^2 + 4 * y^2 + 6 * x - 8 * y + d

/-- The graph consists of a single point -/
def is_single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, equation p.1 p.2 d = 0

/-- Theorem: If the equation represents a graph that consists of a single point, then d = 7 -/
theorem single_point_implies_d_eq_seven :
  ∃ d : ℝ, is_single_point d → d = 7 :=
sorry

end single_point_implies_d_eq_seven_l1425_142580
