import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_prime_sum_of_digits_l1106_110661

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem quadratic_roots_prime_sum_of_digits (c : ℕ) :
  (∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ 
    p ≠ q ∧
    p * q = c ∧
    p + q = 85 ∧
    ∀ x : ℝ, x^2 - 85*x + c = 0 ↔ x = p ∨ x = q) →
  sum_of_digits c = 13 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_prime_sum_of_digits_l1106_110661


namespace NUMINAMATH_CALUDE_banana_bread_theorem_l1106_110653

def bananas_per_loaf : ℕ := 4
def monday_loaves : ℕ := 3
def tuesday_loaves : ℕ := 2 * monday_loaves

def total_loaves : ℕ := monday_loaves + tuesday_loaves
def total_bananas : ℕ := total_loaves * bananas_per_loaf

theorem banana_bread_theorem : total_bananas = 36 := by
  sorry

end NUMINAMATH_CALUDE_banana_bread_theorem_l1106_110653


namespace NUMINAMATH_CALUDE_mass_percentage_N_is_9_66_l1106_110694

/-- The mass percentage of N in a certain compound -/
def mass_percentage_N : ℝ := 9.66

/-- Theorem stating that the mass percentage of N in the compound is 9.66% -/
theorem mass_percentage_N_is_9_66 : mass_percentage_N = 9.66 := by
  sorry

end NUMINAMATH_CALUDE_mass_percentage_N_is_9_66_l1106_110694


namespace NUMINAMATH_CALUDE_total_birds_l1106_110636

def geese : ℕ := 58
def ducks : ℕ := 37

theorem total_birds : geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_l1106_110636


namespace NUMINAMATH_CALUDE_min_n_for_inequality_l1106_110666

theorem min_n_for_inequality : 
  ∃ (n : ℕ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 ≤ n * (x^4 + y^4 + z^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), x^2 + y^2 + z^2 > m * (x^4 + y^4 + z^4)) :=
by
  use 3
  sorry

end NUMINAMATH_CALUDE_min_n_for_inequality_l1106_110666


namespace NUMINAMATH_CALUDE_advantage_is_most_appropriate_l1106_110625

/-- Represents the beneficial aspect of language skills in a job context -/
def BeneficialAspect : Type := String

/-- The set of possible words to fill in the blank -/
def WordChoices : Set String := {"chance", "ability", "possibility", "advantage"}

/-- Predicate to check if a word appropriately describes the beneficial aspect of language skills -/
def IsAppropriateWord (word : String) : Prop :=
  word ∈ WordChoices ∧ 
  word = "advantage"

/-- Theorem stating that "advantage" is the most appropriate word -/
theorem advantage_is_most_appropriate : 
  ∃ (word : String), IsAppropriateWord word ∧ 
  ∀ (other : String), IsAppropriateWord other → other = word :=
sorry

end NUMINAMATH_CALUDE_advantage_is_most_appropriate_l1106_110625


namespace NUMINAMATH_CALUDE_three_petal_percentage_is_75_l1106_110667

/-- The percentage of clovers with three petals -/
def three_petal_percentage : ℝ := 100 - 24 - 1

/-- The total percentage of clovers with two, three, and four petals -/
def total_percentage : ℝ := 100

/-- The percentage of clovers with two petals -/
def two_petal_percentage : ℝ := 24

/-- The percentage of clovers with four petals -/
def four_petal_percentage : ℝ := 1

theorem three_petal_percentage_is_75 :
  three_petal_percentage = 75 :=
by sorry

end NUMINAMATH_CALUDE_three_petal_percentage_is_75_l1106_110667


namespace NUMINAMATH_CALUDE_sequence_bounded_l1106_110615

/-- Given a sequence of nonnegative real numbers satisfying certain conditions, prove that it is bounded -/
theorem sequence_bounded (c : ℝ) (a : ℕ → ℝ) (hc : c > 2)
  (h1 : ∀ m n : ℕ, m ≥ 1 → n ≥ 1 → a (m + n) ≤ 2 * a m + 2 * a n)
  (h2 : ∀ k : ℕ, a (2^k) ≤ 1 / ((k : ℝ) + 1)^c)
  (h3 : ∀ n : ℕ, a n ≥ 0) :
  ∃ M : ℝ, ∀ n : ℕ, n ≥ 1 → a n ≤ M :=
sorry

end NUMINAMATH_CALUDE_sequence_bounded_l1106_110615


namespace NUMINAMATH_CALUDE_total_turnips_count_l1106_110675

/-- The number of turnips grown by Sally -/
def sally_turnips : ℕ := 113

/-- The number of turnips grown by Mary -/
def mary_turnips : ℕ := 129

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := sally_turnips + mary_turnips

theorem total_turnips_count : total_turnips = 242 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_count_l1106_110675


namespace NUMINAMATH_CALUDE_smallest_c_for_inequality_l1106_110681

theorem smallest_c_for_inequality (m n : ℕ) : 
  (∀ c : ℕ, (27 ^ c) * (2 ^ (24 - n)) > (3 ^ (24 + m)) * (5 ^ n) → c ≥ 9) ∧ 
  ((27 ^ 9) * (2 ^ (24 - n)) > (3 ^ (24 + m)) * (5 ^ n)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_inequality_l1106_110681


namespace NUMINAMATH_CALUDE_tensor_identity_l1106_110610

/-- Define a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Define the ⊗ operation -/
def tensor (m n : Vector2D) : Vector2D :=
  ⟨m.x * n.x + m.y * n.y, m.x * n.y + m.y * n.x⟩

theorem tensor_identity (p : Vector2D) : 
  (∀ m : Vector2D, tensor m p = m) → p = ⟨1, 0⟩ := by
  sorry

end NUMINAMATH_CALUDE_tensor_identity_l1106_110610


namespace NUMINAMATH_CALUDE_fraction_equality_l1106_110637

theorem fraction_equality (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (1 / (y + 1)) / (1 / (x + 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1106_110637


namespace NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l1106_110673

/-- The minimum number of socks needed to ensure at least n pairs of the same color
    when randomly picking from a set of socks with m different colors. -/
def min_socks (n : ℕ) (m : ℕ) : ℕ :=
  m + 1 + 2 * (n - 1)

/-- Theorem: Given a set of socks with 4 different colors, 
    the minimum number of socks that must be randomly picked 
    to ensure at least 15 pairs of the same color is 33. -/
theorem min_socks_for_fifteen_pairs : min_socks 15 4 = 33 := by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l1106_110673


namespace NUMINAMATH_CALUDE_horse_rider_ratio_l1106_110696

theorem horse_rider_ratio :
  ∀ (total_horses : ℕ) (total_legs_walking : ℕ),
    total_horses = 12 →
    total_legs_walking = 60 →
    ∃ (riding_owners walking_owners : ℕ),
      riding_owners + walking_owners = total_horses ∧
      walking_owners * 6 = total_legs_walking ∧
      riding_owners * 6 = total_horses := by
  sorry

end NUMINAMATH_CALUDE_horse_rider_ratio_l1106_110696


namespace NUMINAMATH_CALUDE_number_thought_of_l1106_110692

theorem number_thought_of (x : ℝ) : x / 5 + 23 = 42 → x = 95 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l1106_110692


namespace NUMINAMATH_CALUDE_student_rank_theorem_l1106_110643

/-- Calculates the rank from the last given the total number of students and rank from the top -/
def rankFromLast (totalStudents : ℕ) (rankFromTop : ℕ) : ℕ :=
  totalStudents - rankFromTop + 1

/-- Theorem stating that in a class of 35 students, if a student ranks 14th from the top, their rank from the last is 22nd -/
theorem student_rank_theorem (totalStudents : ℕ) (rankFromTop : ℕ) 
  (h1 : totalStudents = 35) (h2 : rankFromTop = 14) : 
  rankFromLast totalStudents rankFromTop = 22 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_theorem_l1106_110643


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1106_110665

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1106_110665


namespace NUMINAMATH_CALUDE_f_2015_equals_one_l1106_110613

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2015_equals_one (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : ∀ x, f (x + 2) * f x = 1)
  (h3 : ∀ x, f x > 0) : 
  f 2015 = 1 := by sorry

end NUMINAMATH_CALUDE_f_2015_equals_one_l1106_110613


namespace NUMINAMATH_CALUDE_largest_quantity_l1106_110660

def A : ℚ := 3003 / 3002 + 3003 / 3004
def B : ℚ := 3003 / 3004 + 3005 / 3004
def C : ℚ := 3004 / 3003 + 3004 / 3005

theorem largest_quantity : A > B ∧ A > C := by sorry

end NUMINAMATH_CALUDE_largest_quantity_l1106_110660


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1106_110626

/-- Given a geometric sequence, returns the sum of the first n terms -/
noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Proves that for a geometric sequence with specific properties, 
    the sum of the first 9000 terms is 1355 -/
theorem geometric_sequence_sum 
  (a r : ℝ) 
  (h1 : geometricSum a r 3000 = 500)
  (h2 : geometricSum a r 6000 = 950) :
  geometricSum a r 9000 = 1355 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1106_110626


namespace NUMINAMATH_CALUDE_min_value_a_l1106_110629

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (a - 1) * x^2 - 2 * Real.sqrt 2 * x * y + a * y^2 ≥ 0) →
  a ≥ 2 ∧ ∀ b : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → (b - 1) * x^2 - 2 * Real.sqrt 2 * x * y + b * y^2 ≥ 0) → b ≥ a :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l1106_110629


namespace NUMINAMATH_CALUDE_f_has_zero_in_interval_l1106_110671

/-- The function f(x) = x^3 + x - 8 -/
def f (x : ℝ) : ℝ := x^3 + x - 8

/-- Theorem: f(x) has a zero in the interval (1, 2) -/
theorem f_has_zero_in_interval :
  ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0 :=
by
  have h1 : f 1 < 0 := by sorry
  have h2 : f 2 > 0 := by sorry
  sorry

#check f_has_zero_in_interval

end NUMINAMATH_CALUDE_f_has_zero_in_interval_l1106_110671


namespace NUMINAMATH_CALUDE_min_value_fraction_l1106_110608

theorem min_value_fraction (x y : ℝ) (hx : 1/2 ≤ x ∧ x ≤ 2) (hy : 4/3 ≤ y ∧ y ≤ 3/2) :
  (x^3 * y^3) / (x^6 + 3*x^4*y^2 + 3*x^3*y^3 + 3*x^2*y^4 + y^6) ≥ 27/1081 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1106_110608


namespace NUMINAMATH_CALUDE_remaining_work_for_x_l1106_110609

/-- The number of days x needs to finish the remaining work after y worked for 5 days --/
def remaining_days_for_x (x_days y_days : ℚ) : ℚ :=
  (1 - 5 / y_days) * x_days

theorem remaining_work_for_x :
  remaining_days_for_x 21 15 = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_work_for_x_l1106_110609


namespace NUMINAMATH_CALUDE_bella_friends_count_l1106_110647

/-- The number of beads needed per bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of beads Bella currently has -/
def current_beads : ℕ := 36

/-- The number of additional beads Bella needs -/
def additional_beads : ℕ := 12

/-- The number of friends Bella is making bracelets for -/
def num_friends : ℕ := (current_beads + additional_beads) / beads_per_bracelet

theorem bella_friends_count : num_friends = 6 := by
  sorry

end NUMINAMATH_CALUDE_bella_friends_count_l1106_110647


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1106_110679

theorem inequality_solution_set :
  {x : ℝ | |x - 1| + |2*x + 5| < 8} = Set.Ioo (-4 : ℝ) (4/3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1106_110679


namespace NUMINAMATH_CALUDE_tank_filling_time_l1106_110670

/-- The time (in hours) it takes to fill the tank without a leak -/
def fill_time : ℝ := 5

/-- The time (in hours) it takes for the leak to empty a full tank -/
def empty_time : ℝ := 30

/-- The extra time (in hours) it takes to fill the tank due to the leak -/
def extra_time : ℝ := 1

theorem tank_filling_time :
  extra_time = (1 / ((1 / fill_time) - (1 / empty_time))) - fill_time :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l1106_110670


namespace NUMINAMATH_CALUDE_smallest_common_difference_l1106_110644

/-- Represents a quadratic equation ax^2 + bx + c = 0 --/
structure QuadraticEquation where
  a : Int
  b : Int
  c : Int

/-- Checks if a quadratic equation has two distinct roots --/
def hasTwoDistinctRoots (eq : QuadraticEquation) : Prop :=
  eq.b * eq.b - 4 * eq.a * eq.c > 0

/-- Generates all possible quadratic equations with coefficients a, b, 2c --/
def generateQuadraticEquations (a b c : Int) : List QuadraticEquation :=
  [
    ⟨a, b, 2*c⟩, ⟨a, 2*c, b⟩, ⟨b, a, 2*c⟩,
    ⟨b, 2*c, a⟩, ⟨2*c, a, b⟩, ⟨2*c, b, a⟩
  ]

theorem smallest_common_difference
  (a b c : Int)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_arithmetic : ∃ d : Int, b = a + d ∧ c = a + 2*d)
  (h_increasing : a < b ∧ b < c)
  (h_distinct_roots : ∀ eq ∈ generateQuadraticEquations a b c, hasTwoDistinctRoots eq) :
  ∃ d : Int, d = 4 ∧ a = -5 ∧ b = -1 ∧ c = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_difference_l1106_110644


namespace NUMINAMATH_CALUDE_polygon_sides_range_l1106_110686

/-- Represents the count of vertices with different internal angles -/
structure VertexCounts where
  a : ℕ  -- Count of 60° angles
  b : ℕ  -- Count of 90° angles
  c : ℕ  -- Count of 120° angles
  d : ℕ  -- Count of 150° angles

/-- Theorem stating the possible values of n for a convex n-sided polygon 
    formed by combining equilateral triangles and squares -/
theorem polygon_sides_range (n : ℕ) : 
  (∃ v : VertexCounts, 
    v.a + v.b + v.c + v.d = n ∧ 
    4 * v.a + 3 * v.b + 2 * v.c + v.d = 12 ∧
    v.a + v.b > 0 ∧ v.c + v.d > 0) ↔ 
  5 ≤ n ∧ n ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_range_l1106_110686


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1106_110685

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 11 = 0 →
  ∃ (h k r : ℝ), h = -1 ∧ k = 2 ∧ r = 2 ∧
  (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1106_110685


namespace NUMINAMATH_CALUDE_sqrt_pattern_l1106_110645

theorem sqrt_pattern (n : ℕ+) : 
  Real.sqrt (n + 1 / (n + 2)) = ((n + 1) * Real.sqrt (n + 2)) / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l1106_110645


namespace NUMINAMATH_CALUDE_abc_inequality_l1106_110690

theorem abc_inequality (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l1106_110690


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l1106_110656

/-- Given three concentric circles with radii r, s, and t, where r > s > t,
    and p, q as defined in the problem, prove that the area between the
    largest and smallest circles is π(p² + q²). -/
theorem area_between_concentric_circles
  (r s t p q : ℝ)
  (h_order : r > s ∧ s > t)
  (h_p : p = (r^2 - s^2).sqrt)
  (h_q : q = (s^2 - t^2).sqrt) :
  π * (r^2 - t^2) = π * (p^2 + q^2) := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l1106_110656


namespace NUMINAMATH_CALUDE_divisor_problem_l1106_110699

theorem divisor_problem (x k m y : ℤ) 
  (h1 : x = 62 * k + 7)
  (h2 : x + 11 = y * m + 18)
  (h3 : y > 18)
  (h4 : 62 % y = 0) :
  y = 31 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1106_110699


namespace NUMINAMATH_CALUDE_smallest_divisor_of_427395_l1106_110688

theorem smallest_divisor_of_427395 : 
  ∀ d : ℕ, d > 0 ∧ d < 5 → ¬(427395 % d = 0) ∧ 427395 % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_427395_l1106_110688


namespace NUMINAMATH_CALUDE_sine_cosine_sum_l1106_110658

theorem sine_cosine_sum (α : Real) (h : Real.sin (α - π/6) = 1/3) :
  Real.sin (2*α - π/6) + Real.cos (2*α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_l1106_110658


namespace NUMINAMATH_CALUDE_village_population_l1106_110634

theorem village_population (P : ℕ) : 
  (((((P * 95 / 100) * 85 / 100) * 93 / 100) * 80 / 100) * 90 / 100) * 75 / 100 = 3553 →
  P = 9262 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1106_110634


namespace NUMINAMATH_CALUDE_correct_donations_l1106_110642

/-- Represents the donations to five orphanages -/
structure OrphanageDonations where
  total : ℝ
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ
  fifth : ℝ

/-- Checks if the donations satisfy the given conditions -/
def validDonations (d : OrphanageDonations) : Prop :=
  d.total = 1300 ∧
  d.first = 0.2 * d.total ∧
  d.second = d.first / 2 ∧
  d.third = 2 * d.second ∧
  d.fourth = d.fifth ∧
  d.fourth + d.fifth = d.third

/-- Theorem stating that the given donations satisfy all conditions -/
theorem correct_donations :
  ∃ d : OrphanageDonations,
    validDonations d ∧
    d.first = 260 ∧
    d.second = 130 ∧
    d.third = 260 ∧
    d.fourth = 130 ∧
    d.fifth = 130 :=
sorry

end NUMINAMATH_CALUDE_correct_donations_l1106_110642


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l1106_110691

theorem floor_plus_self_eq_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l1106_110691


namespace NUMINAMATH_CALUDE_fashion_show_runway_time_l1106_110630

/-- The fashion show runway problem -/
theorem fashion_show_runway_time :
  let num_models : ℕ := 6
  let bathing_suits_per_model : ℕ := 2
  let evening_wear_per_model : ℕ := 3
  let time_per_trip : ℕ := 2

  let total_trips_per_model : ℕ := bathing_suits_per_model + evening_wear_per_model
  let total_trips : ℕ := num_models * total_trips_per_model
  let total_time : ℕ := total_trips * time_per_trip

  total_time = 60
  := by sorry

end NUMINAMATH_CALUDE_fashion_show_runway_time_l1106_110630


namespace NUMINAMATH_CALUDE_bc_length_l1106_110622

-- Define the triangle
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 0) -- Exact coordinates don't matter for this proof
def C : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (0, 0)

-- Define the given lengths
def AD : ℝ := 47
def CD : ℝ := 25
def AC : ℝ := 24

-- Define the theorem
theorem bc_length :
  Triangle A B C →
  Triangle A B D →
  D.1 < C.1 →
  D.2 = B.2 →
  C.2 = B.2 →
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = AD^2 →
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = CD^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 20.16^2 := by
  sorry

end NUMINAMATH_CALUDE_bc_length_l1106_110622


namespace NUMINAMATH_CALUDE_museum_trip_total_l1106_110684

/-- The number of people on the first bus -/
def first_bus : ℕ := 12

/-- The number of people on the second bus -/
def second_bus : ℕ := 2 * first_bus

/-- The number of people on the third bus -/
def third_bus : ℕ := second_bus - 6

/-- The number of people on the fourth bus -/
def fourth_bus : ℕ := first_bus + 9

/-- The total number of people going to the museum -/
def total_people : ℕ := first_bus + second_bus + third_bus + fourth_bus

theorem museum_trip_total : total_people = 75 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_total_l1106_110684


namespace NUMINAMATH_CALUDE_marble_ratio_proof_l1106_110638

def marble_problem (initial_marbles : ℕ) (lost_through_hole : ℕ) (final_marbles : ℕ) : Prop :=
  let dog_eaten : ℕ := lost_through_hole / 2
  let before_giving_away : ℕ := initial_marbles - lost_through_hole - dog_eaten
  let given_away : ℕ := before_giving_away - final_marbles
  (given_away : ℚ) / lost_through_hole = 2

theorem marble_ratio_proof :
  marble_problem 24 4 10 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_proof_l1106_110638


namespace NUMINAMATH_CALUDE_orthogonal_trajectories_and_intersection_angle_l1106_110641

-- Define the family of conics
def conic (a : ℝ) (x y : ℝ) : Prop :=
  (x + 2*y)^2 = a*(x + y)

-- Define the orthogonal trajectory
def orthogonal_trajectory (c : ℝ) (x y : ℝ) : Prop :=
  y = c*x^2 - 3*x

-- Theorem statement
theorem orthogonal_trajectories_and_intersection_angle :
  ∀ (a c : ℝ),
  (∃ (x y : ℝ), conic a x y ∧ orthogonal_trajectory c x y) ∧
  (∃ (x y : ℝ), conic a x y ∧ x = 0 ∧ y = 0 ∧
    ∃ (x' y' : ℝ), orthogonal_trajectory c x' y' ∧ x' = 0 ∧ y' = 0 ∧
    Real.arctan ((y' - y) / (x' - x)) = π / 4) :=
by sorry


end NUMINAMATH_CALUDE_orthogonal_trajectories_and_intersection_angle_l1106_110641


namespace NUMINAMATH_CALUDE_door_open_probability_l1106_110604

def num_keys : ℕ := 5

def probability_open_on_third_attempt : ℚ := 1 / 5

theorem door_open_probability :
  probability_open_on_third_attempt = 0.2 := by sorry

end NUMINAMATH_CALUDE_door_open_probability_l1106_110604


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1106_110639

/-- The line equation passes through the point (2,2) for all values of k -/
theorem line_passes_through_point :
  ∀ (k : ℝ), (1 + 4*k) * 2 - (2 - 3*k) * 2 + 2 - 14*k = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1106_110639


namespace NUMINAMATH_CALUDE_arrangements_of_six_acts_l1106_110687

/-- The number of ways to insert two distinguishable items into a sequence of n fixed items -/
def insert_two_items (n : ℕ) : ℕ :=
  (n + 1) * (n + 2)

/-- Theorem stating that inserting 2 items into a sequence of 4 fixed items results in 30 arrangements -/
theorem arrangements_of_six_acts : insert_two_items 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_of_six_acts_l1106_110687


namespace NUMINAMATH_CALUDE_max_cookies_andy_l1106_110662

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_cookies_andy (total : ℕ) (x : ℕ) (p : ℕ) 
  (h_total : total = 30)
  (h_prime : is_prime p)
  (h_all_eaten : x + p * x = total) :
  x ≤ 10 ∧ ∃ (x₀ : ℕ) (p₀ : ℕ), x₀ = 10 ∧ is_prime p₀ ∧ x₀ + p₀ * x₀ = total :=
sorry

end NUMINAMATH_CALUDE_max_cookies_andy_l1106_110662


namespace NUMINAMATH_CALUDE_equal_water_after_operations_l1106_110633

theorem equal_water_after_operations (x : ℝ) (h : x > 0) :
  let barrel1 := x * 0.9 * 1.1
  let barrel2 := x * 1.1 * 0.9
  barrel1 = barrel2 := by sorry

end NUMINAMATH_CALUDE_equal_water_after_operations_l1106_110633


namespace NUMINAMATH_CALUDE_value_of_b_l1106_110619

theorem value_of_b (m a b d : ℝ) (h : m = (d * a * b) / (a + b)) :
  b = (m * a) / (d * a - m) := by sorry

end NUMINAMATH_CALUDE_value_of_b_l1106_110619


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_l1106_110683

theorem recurring_decimal_fraction :
  (5 : ℚ) / 33 / ((2401 : ℚ) / 999) = 4995 / 79233 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_l1106_110683


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1106_110631

-- Define the inverse proportionality constant
def k : ℝ → ℝ → ℝ := λ x y => x * y

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  ∃ (c : ℝ), k x y = c ∧ x + y = 30 ∧ x - y = 10

-- Theorem statement
theorem inverse_proportion_problem :
  ∀ x y : ℝ, conditions x y → (x = 4 → y = 50) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1106_110631


namespace NUMINAMATH_CALUDE_sector_angle_l1106_110669

/-- Given a sector with radius 1 and area 3π/8, its central angle is 3π/4 -/
theorem sector_angle (r : ℝ) (A : ℝ) (α : ℝ) : 
  r = 1 → A = (3 * π) / 8 → A = (1 / 2) * α * r^2 → α = (3 * π) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1106_110669


namespace NUMINAMATH_CALUDE_hare_run_distance_l1106_110651

/-- The distance between trees in meters -/
def tree_distance : ℕ := 5

/-- The number of the first tree -/
def first_tree : ℕ := 1

/-- The number of the last tree -/
def last_tree : ℕ := 10

/-- The total distance between the first and last tree -/
def total_distance : ℕ := tree_distance * (last_tree - first_tree)

theorem hare_run_distance :
  total_distance = 45 := by sorry

end NUMINAMATH_CALUDE_hare_run_distance_l1106_110651


namespace NUMINAMATH_CALUDE_square_minus_three_product_plus_square_l1106_110628

theorem square_minus_three_product_plus_square (a b : ℝ) 
  (sum_eq : a + b = 8) 
  (product_eq : a * b = 9) : 
  a^2 - 3*a*b + b^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_square_minus_three_product_plus_square_l1106_110628


namespace NUMINAMATH_CALUDE_max_value_of_function_l1106_110689

theorem max_value_of_function (x : ℝ) (h : x < 5/4) :
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 ∧ 
  ∃ y : ℝ, y < 5/4 ∧ 4 * y - 2 + 1 / (4 * y - 5) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1106_110689


namespace NUMINAMATH_CALUDE_multiplicative_inverse_problem_l1106_110624

theorem multiplicative_inverse_problem :
  let A : ℕ := 111112
  let B : ℕ := 142858
  let M : ℕ := 1000003
  let N : ℕ := 513487
  (A * B * N) % M = 1 := by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_problem_l1106_110624


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1106_110682

/-- Given a cube with surface area 6x^2, where x is the length of one side,
    prove that the volume of the cube is x^3. -/
theorem cube_volume_from_surface_area (x : ℝ) (h : x > 0) :
  let surface_area := 6 * x^2
  let side_length := x
  let volume := side_length^3
  surface_area = 6 * side_length^2 → volume = x^3 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1106_110682


namespace NUMINAMATH_CALUDE_point_transformation_l1106_110663

/-- Rotation of 90 degrees around the z-axis -/
def rotateZ90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

/-- Reflection through the xy-plane -/
def reflectXY (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

/-- Reflection through the yz-plane -/
def reflectYZ (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

/-- The sequence of transformations applied to the point -/
def transformPoint (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotateZ90 |> reflectXY |> reflectYZ |> rotateZ90 |> reflectYZ

theorem point_transformation :
  transformPoint (2, 3, 4) = (2, 3, -4) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1106_110663


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l1106_110607

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  3230000 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.23 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l1106_110607


namespace NUMINAMATH_CALUDE_scaled_variance_l1106_110640

def variance (data : List ℝ) : ℝ := sorry

theorem scaled_variance (data : List ℝ) (h : variance data = 3) :
  variance (List.map (· * 2) data) = 12 := by sorry

end NUMINAMATH_CALUDE_scaled_variance_l1106_110640


namespace NUMINAMATH_CALUDE_sprint_competition_races_l1106_110668

/-- The number of races needed to determine a champion in a sprint competition --/
def racesNeeded (totalSprinters : ℕ) (lanesPerRace : ℕ) (eliminatedPerRace : ℕ) : ℕ :=
  Nat.ceil ((totalSprinters - 1) / eliminatedPerRace)

/-- Theorem stating that 46 races are needed for the given conditions --/
theorem sprint_competition_races : 
  racesNeeded 320 8 7 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sprint_competition_races_l1106_110668


namespace NUMINAMATH_CALUDE_unique_projection_l1106_110657

def vector_projection (a b s p : ℝ × ℝ) : Prop :=
  let line_dir := (b.1 - a.1, b.2 - a.2)
  let shifted_line (t : ℝ) := (a.1 + s.1 + t * line_dir.1, a.2 + s.2 + t * line_dir.2)
  ∃ t : ℝ, 
    p = shifted_line t ∧ 
    line_dir.1 * (p.1 - (a.1 + s.1)) + line_dir.2 * (p.2 - (a.2 + s.2)) = 0

theorem unique_projection :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (-1, 4)
  let s : ℝ × ℝ := (1, 1)
  let p : ℝ × ℝ := (16/13, 41/26)
  vector_projection a b s p ∧ 
  ∀ q : ℝ × ℝ, vector_projection a b s q → q = p := by sorry

end NUMINAMATH_CALUDE_unique_projection_l1106_110657


namespace NUMINAMATH_CALUDE_coordinates_of_point_A_l1106_110614

def point_A (a : ℝ) : ℝ × ℝ := (a - 1, 3 * a - 2)

theorem coordinates_of_point_A :
  ∀ a : ℝ, (point_A a).1 = (point_A a).2 + 3 → point_A a = (-2, -5) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_point_A_l1106_110614


namespace NUMINAMATH_CALUDE_y_value_l1106_110649

theorem y_value (y : ℝ) (h : (9 : ℝ) / y^2 = y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1106_110649


namespace NUMINAMATH_CALUDE_sum_of_bounds_l1106_110602

def U : Type := ℝ

def A (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

def complement_A : Set ℝ := {x | x > 4 ∨ x < 3}

theorem sum_of_bounds (a b : ℝ) :
  A a b = (Set.univ \ complement_A) → a + b = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_bounds_l1106_110602


namespace NUMINAMATH_CALUDE_simplify_T_l1106_110605

theorem simplify_T (x : ℝ) : 
  (x + 2)^6 + 6*(x + 2)^5 + 15*(x + 2)^4 + 20*(x + 2)^3 + 15*(x + 2)^2 + 6*(x + 2) + 1 = (x + 3)^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_T_l1106_110605


namespace NUMINAMATH_CALUDE_find_number_to_multiply_l1106_110652

theorem find_number_to_multiply : ∃ x : ℕ, 
  (43 * x) - (34 * x) = 1224 ∧ x = 136 := by
  sorry

end NUMINAMATH_CALUDE_find_number_to_multiply_l1106_110652


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1106_110654

/-- A quadratic function with vertex form (x + h)^2 passing through a specific point -/
def QuadraticFunction (a : ℝ) (h : ℝ) (x₀ : ℝ) (y₀ : ℝ) : Prop :=
  y₀ = a * (x₀ + h)^2

theorem quadratic_coefficient (a : ℝ) :
  QuadraticFunction a 3 2 (-50) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1106_110654


namespace NUMINAMATH_CALUDE_index_card_area_l1106_110664

theorem index_card_area (length width : ℝ) (h1 : length = 5) (h2 : width = 7) : 
  (∃ side, (side - 2) * width = 21 ∨ length * (side - 2) = 21) →
  (length * (width - 2) = 25 ∨ (length - 2) * width = 25) := by
sorry

end NUMINAMATH_CALUDE_index_card_area_l1106_110664


namespace NUMINAMATH_CALUDE_four_solutions_l1106_110611

/-- The number of solutions to the equation 4/m + 2/n = 1 where m and n are positive integers -/
def num_solutions : ℕ := 4

/-- A function that checks if a pair of positive integers satisfies the equation 4/m + 2/n = 1 -/
def satisfies_equation (m n : ℕ+) : Prop :=
  (4 : ℚ) / m.val + (2 : ℚ) / n.val = 1

/-- The theorem stating that there are exactly 4 solutions to the equation -/
theorem four_solutions :
  ∃! (solutions : Finset (ℕ+ × ℕ+)),
    solutions.card = num_solutions ∧
    ∀ (pair : ℕ+ × ℕ+), pair ∈ solutions ↔ satisfies_equation pair.1 pair.2 :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l1106_110611


namespace NUMINAMATH_CALUDE_perimeter_difference_l1106_110697

/-- The perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ := 2 * (length + width)

/-- The perimeter of a stack of rectangles --/
def stackedRectanglesPerimeter (length width count : ℕ) : ℕ :=
  2 * length + 2 * (width * count)

/-- The difference in perimeters between a 6x1 rectangle and three 3x1 rectangles stacked vertically --/
theorem perimeter_difference :
  rectanglePerimeter 6 1 - stackedRectanglesPerimeter 3 1 3 = 2 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_difference_l1106_110697


namespace NUMINAMATH_CALUDE_rectangular_triangle_condition_l1106_110674

theorem rectangular_triangle_condition (A B C : Real) 
  (h : (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 
       2 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2)) 
  (triangle_angles : A + B + C = Real.pi) :
  A = Real.pi/2 ∨ B = Real.pi/2 ∨ C = Real.pi/2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_triangle_condition_l1106_110674


namespace NUMINAMATH_CALUDE_min_handshakes_35_people_l1106_110648

/-- Represents a gathering of people and their handshakes -/
structure Gathering where
  people : ℕ
  handshakes_per_person : ℕ

/-- Calculates the total number of handshakes in a gathering -/
def total_handshakes (g : Gathering) : ℕ :=
  g.people * g.handshakes_per_person / 2

/-- Theorem: In a gathering of 35 people where each person shakes hands with 
    exactly 3 others, the minimum possible number of handshakes is 105 -/
theorem min_handshakes_35_people : 
  ∃ (g : Gathering), g.people = 35 ∧ g.handshakes_per_person = 6 ∧ total_handshakes g = 105 := by
  sorry

#check min_handshakes_35_people

end NUMINAMATH_CALUDE_min_handshakes_35_people_l1106_110648


namespace NUMINAMATH_CALUDE_quadratic_root_in_arithmetic_sequence_l1106_110678

/-- Given real numbers x, y, z forming an arithmetic sequence with x ≥ y ≥ z ≥ 0,
    if the quadratic zx^2 + yx + x has exactly one root, then this root is -3/4. -/
theorem quadratic_root_in_arithmetic_sequence (x y z : ℝ) :
  (∃ d : ℝ, y = x - d ∧ z = x - 2*d) →  -- arithmetic sequence condition
  x ≥ y →
  y ≥ z →
  z ≥ 0 →
  (∃! r : ℝ, z*r^2 + y*r + x = 0) →  -- exactly one root condition
  (∃ r : ℝ, z*r^2 + y*r + x = 0 ∧ r = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_arithmetic_sequence_l1106_110678


namespace NUMINAMATH_CALUDE_piggy_bank_savings_l1106_110655

theorem piggy_bank_savings (first_year : ℝ) : 
  first_year + 2 * first_year + 4 * first_year + 8 * first_year = 450 →
  first_year = 30 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_savings_l1106_110655


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1106_110623

theorem absolute_value_inequality (x : ℝ) :
  (abs (x + 2) + abs (x - 1) ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1106_110623


namespace NUMINAMATH_CALUDE_min_m_plus_n_for_1978_power_divisibility_l1106_110677

theorem min_m_plus_n_for_1978_power_divisibility (m n : ℕ) : 
  m > n → n ≥ 1 → (1000 ∣ 1978^m - 1978^n) → m + n ≥ 106 ∧ ∃ (m₀ n₀ : ℕ), m₀ > n₀ ∧ n₀ ≥ 1 ∧ (1000 ∣ 1978^m₀ - 1978^n₀) ∧ m₀ + n₀ = 106 :=
by sorry

end NUMINAMATH_CALUDE_min_m_plus_n_for_1978_power_divisibility_l1106_110677


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1106_110601

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the theorem
theorem sin_2alpha_value (α : ℝ) :
  (Real.cos α * P.1 = Real.sin α * P.2) →  -- Terminal side passes through P
  Real.sin (2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1106_110601


namespace NUMINAMATH_CALUDE_integral_sqrt_plus_x_l1106_110650

theorem integral_sqrt_plus_x :
  ∫ (x : ℝ) in (0)..(1), (Real.sqrt (1 - x^2) + x) = π / 4 + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_plus_x_l1106_110650


namespace NUMINAMATH_CALUDE_min_value_f_inequality_abc_l1106_110600

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∀ x : ℝ, f x ≥ 3 := by sorry

-- Theorem for the inequality
theorem inequality_abc (a b c m : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hm : a + b + c = m) :
  b^2 / a + c^2 / b + a^2 / c ≥ 3 := by sorry

end NUMINAMATH_CALUDE_min_value_f_inequality_abc_l1106_110600


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_9_l1106_110680

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

theorem five_digit_multiple_of_9 (d : ℕ) (h1 : d < 10) :
  is_multiple_of_9 (63470 + d) ↔ d = 7 := by sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_9_l1106_110680


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1106_110627

/-- The solution set of the system of equations {x - 2y = 1, x^3 - 6xy - 8y^3 = 1} 
    is equivalent to the line y = (x-1)/2 -/
theorem solution_set_equivalence (x y : ℝ) : 
  (x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1) ↔ y = (x - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1106_110627


namespace NUMINAMATH_CALUDE_number_comparison_l1106_110632

theorem number_comparison : 22^44 > 33^33 ∧ 33^33 > 44^22 := by sorry

end NUMINAMATH_CALUDE_number_comparison_l1106_110632


namespace NUMINAMATH_CALUDE_ninth_root_unity_product_l1106_110612

theorem ninth_root_unity_product : 
  let x : ℂ := Complex.exp (2 * π * I / 9)
  (3 * x + x^2) * (3 * x^3 + x^6) * (3 * x^4 + x^8) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ninth_root_unity_product_l1106_110612


namespace NUMINAMATH_CALUDE_exponential_inequality_l1106_110698

theorem exponential_inequality (a b m : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : a ≠ 1) 
  (h4 : b ≠ 1) 
  (h5 : 0 < m) 
  (h6 : m < 1) : 
  m^a < m^b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1106_110698


namespace NUMINAMATH_CALUDE_square_of_cube_of_fourth_smallest_prime_l1106_110695

def fourth_smallest_prime : ℕ := 7

theorem square_of_cube_of_fourth_smallest_prime :
  (fourth_smallest_prime ^ 3) ^ 2 = 117649 := by
  sorry

end NUMINAMATH_CALUDE_square_of_cube_of_fourth_smallest_prime_l1106_110695


namespace NUMINAMATH_CALUDE_largest_possible_median_l1106_110646

def number_set (x : ℤ) : Finset ℤ := {x, 2*x, 3, 2, 5}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (λ y => y ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ y => y ≥ m)).card ≥ s.card

theorem largest_possible_median (x : ℤ) :
  ∃ m : ℤ, is_median m (number_set x) ∧ ∀ n : ℤ, is_median n (number_set x) → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_largest_possible_median_l1106_110646


namespace NUMINAMATH_CALUDE_min_value_function_compare_squares_min_value_M_l1106_110620

-- Part 1
theorem min_value_function (x : ℝ) (h : x > -1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 2 + 3 ∧
  ∀ (y : ℝ), y = ((x + 2) * (x + 3)) / (x + 1) → y ≥ min_val :=
sorry

-- Part 2
theorem compare_squares (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1)
  (h : x^2 / a^2 - y^2 / b^2 = 1) :
  a^2 - b^2 ≤ (x - y)^2 :=
sorry

-- Part 3
theorem min_value_M (m : ℝ) (hm : m ≥ 1) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 3 / 2 ∧
  ∀ (M : ℝ), M = Real.sqrt (4 * m - 3) - Real.sqrt (m - 1) → M ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_function_compare_squares_min_value_M_l1106_110620


namespace NUMINAMATH_CALUDE_lychees_remaining_l1106_110693

theorem lychees_remaining (initial : ℕ) (sold_fraction : ℚ) (eaten_fraction : ℚ) : 
  initial = 500 → 
  sold_fraction = 1/2 → 
  eaten_fraction = 3/5 → 
  (initial - initial * sold_fraction - (initial - initial * sold_fraction) * eaten_fraction : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_lychees_remaining_l1106_110693


namespace NUMINAMATH_CALUDE_sequence_matches_first_10_terms_l1106_110618

/-- The sequence defined by a(n) = n(n-1) -/
def a (n : ℕ) : ℕ := n * (n - 1)

/-- The first 10 terms of the sequence -/
def first_10_terms : List ℕ := [0, 2, 6, 12, 20, 30, 42, 56, 72, 90]

theorem sequence_matches_first_10_terms :
  (List.range 10).map (fun i => a (i + 1)) = first_10_terms := by sorry

end NUMINAMATH_CALUDE_sequence_matches_first_10_terms_l1106_110618


namespace NUMINAMATH_CALUDE_truncated_pyramid_overlap_l1106_110676

/-- Regular triangular pyramid with planar angle α at the vertex -/
structure RegularTriangularPyramid where
  α : ℝ  -- Planar angle at the vertex

/-- Regular truncated pyramid cut from a regular triangular pyramid -/
structure RegularTruncatedPyramid (p : RegularTriangularPyramid) where

/-- Unfolded development of a regular truncated pyramid -/
def UnfoldedDevelopment (t : RegularTruncatedPyramid p) : Type := sorry

/-- Predicate to check if an unfolded development overlaps itself -/
def is_self_overlapping (d : UnfoldedDevelopment t) : Prop := sorry

theorem truncated_pyramid_overlap (p : RegularTriangularPyramid) 
  (t : RegularTruncatedPyramid p) (d : UnfoldedDevelopment t) :
  is_self_overlapping d ↔ 100 * π / 180 < p.α ∧ p.α < 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_truncated_pyramid_overlap_l1106_110676


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l1106_110616

theorem multiplication_puzzle (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → (30 + a) * (10 * b + 4) = 142 → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l1106_110616


namespace NUMINAMATH_CALUDE_shifted_function_passes_through_origin_l1106_110606

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Represents a vertical shift of a function -/
structure VerticalShift where
  shift : ℝ

/-- Checks if a linear function passes through the origin -/
def passes_through_origin (f : LinearFunction) : Prop :=
  f.slope * 0 + f.intercept = 0

/-- Applies a vertical shift to a linear function -/
def apply_shift (f : LinearFunction) (s : VerticalShift) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept - s.shift }

/-- The original linear function y = 3x + 5 -/
def original_function : LinearFunction :=
  { slope := 3, intercept := 5 }

/-- The vertical shift of 5 units down -/
def shift_down : VerticalShift :=
  { shift := 5 }

theorem shifted_function_passes_through_origin :
  passes_through_origin (apply_shift original_function shift_down) := by
  sorry

end NUMINAMATH_CALUDE_shifted_function_passes_through_origin_l1106_110606


namespace NUMINAMATH_CALUDE_plane_perpendicular_deduction_l1106_110621

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_deduction 
  (m n : Line) (α β : Plane) 
  (h1 : parallel m n) 
  (h2 : subset m α) 
  (h3 : perp_line_plane n β) : 
  perp_plane α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_deduction_l1106_110621


namespace NUMINAMATH_CALUDE_not_divisible_by_two_2013_l1106_110603

-- Define a property for odd numbers
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Define what it means for a number to be not divisible by 2
def NotDivisibleByTwo (n : ℤ) : Prop := ¬ (∃ k : ℤ, n = 2 * k)

-- State the theorem
theorem not_divisible_by_two_2013 :
  (∀ n : ℤ, IsOdd n → NotDivisibleByTwo n) →
  IsOdd 2013 →
  NotDivisibleByTwo 2013 := by sorry

end NUMINAMATH_CALUDE_not_divisible_by_two_2013_l1106_110603


namespace NUMINAMATH_CALUDE_jakes_snake_length_l1106_110659

theorem jakes_snake_length (j p : ℕ) : 
  j = p + 12 →  -- Jake's snake is 12 inches longer than Penny's snake
  j + p = 70 →  -- The combined length of the two snakes is 70 inches
  j = 41        -- Jake's snake is 41 inches long
:= by sorry

end NUMINAMATH_CALUDE_jakes_snake_length_l1106_110659


namespace NUMINAMATH_CALUDE_magnitude_of_difference_vector_l1106_110672

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 4)

theorem magnitude_of_difference_vector :
  let dot_product := a.1 * b.1 + a.2 * b.2
  dot_product = 10 →
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_difference_vector_l1106_110672


namespace NUMINAMATH_CALUDE_largest_valid_code_l1106_110617

def is_power_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5^k

def is_power_of_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def digits_to_nat (a b c d e : ℕ) : ℕ := a * 10000 + b * 1000 + c * 100 + d * 10 + e

def is_valid_code (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ,
    n = digits_to_nat a b c d e ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    is_power_of_5 (a * 10 + b) ∧
    is_power_of_2 (d * 10 + e) ∧
    ∃ k : ℕ, c = 3 * k ∧
    (a + b + c + d + e) % 2 = 1

theorem largest_valid_code :
  ∀ n : ℕ, is_valid_code n → n ≤ 25916 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_code_l1106_110617


namespace NUMINAMATH_CALUDE_whale_prediction_correct_l1106_110635

/-- The number of whales predicted for next year -/
def whales_next_year : ℕ := 8800

/-- The number of whales last year -/
def whales_last_year : ℕ := 4000

/-- The number of whales this year -/
def whales_this_year : ℕ := 2 * whales_last_year

/-- The predicted increase in the number of whales for next year -/
def predicted_increase : ℕ := whales_next_year - whales_this_year

theorem whale_prediction_correct : predicted_increase = 800 := by
  sorry

end NUMINAMATH_CALUDE_whale_prediction_correct_l1106_110635
