import Mathlib

namespace NUMINAMATH_CALUDE_line_l_equation_no_symmetric_points_l3487_348764

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2*x + y + 1 = 0
def l₂ (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define line l
def l (x y : ℝ) : Prop := x + y = 0

-- Define the parabola
def parabola (a x y : ℝ) : Prop := y = a*x^2 - 1

-- Theorem 1: Prove that l is the correct line given the midpoint condition
theorem line_l_equation : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ (x₁ + x₂)/2 = 0 ∧ (y₁ + y₂)/2 = 0) →
  (∀ x y : ℝ, l x y ↔ x + y = 0) :=
sorry

-- Theorem 2: Prove the condition for non-existence of symmetric points
theorem no_symmetric_points (a : ℝ) :
  (a ≠ 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (parabola a x₁ y₁ ∧ parabola a x₂ y₂ ∧ 
     (x₁ + x₂)/2 + (y₁ + y₂)/2 = 0) → x₁ = x₂ ∧ y₁ = y₂) ↔ 
  (a ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_no_symmetric_points_l3487_348764


namespace NUMINAMATH_CALUDE_total_photos_lisa_robert_l3487_348768

def claire_photos : ℕ := 8
def lisa_photos : ℕ := 3 * claire_photos
def robert_photos : ℕ := claire_photos + 16

theorem total_photos_lisa_robert : lisa_photos + robert_photos = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_photos_lisa_robert_l3487_348768


namespace NUMINAMATH_CALUDE_ellipse_intercept_inequality_l3487_348732

-- Define the ellipse E
def E (m : ℝ) (x y : ℝ) : Prop := x^2 / m + y^2 / 4 = 1

-- Define the discriminant for the line y = kx + 1
def discriminant1 (m k : ℝ) : ℝ := 16 * m^2 * k^2 + 48 * m

-- Define the discriminant for the line kx + y - 2 = 0
def discriminant2 (m k : ℝ) : ℝ := 16 * m^2 * k^2

-- Theorem statement
theorem ellipse_intercept_inequality (m : ℝ) (h : m > 0) :
  ∀ k : ℝ, discriminant1 m k ≠ discriminant2 m k :=
sorry

end NUMINAMATH_CALUDE_ellipse_intercept_inequality_l3487_348732


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l3487_348791

/-- Given two people moving in opposite directions for 45 minutes,
    with one moving at 30 kmph and ending up 60 km apart,
    prove that the speed of the other person is 50 kmph. -/
theorem opposite_direction_speed 
  (riya_speed : ℝ) 
  (time : ℝ) 
  (total_distance : ℝ) 
  (h1 : riya_speed = 30) 
  (h2 : time = 45 / 60) 
  (h3 : total_distance = 60) : 
  ∃ (priya_speed : ℝ), priya_speed = 50 ∧ 
    riya_speed * time + priya_speed * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l3487_348791


namespace NUMINAMATH_CALUDE_worker_payment_l3487_348719

/-- The daily wage in rupees -/
def daily_wage : ℚ := 20

/-- The number of days worked in a week -/
def days_worked : ℚ := 11/3 + 2/3 + 1/8 + 3/4

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem worker_payment :
  round_to_nearest (daily_wage * days_worked) = 104 := by
  sorry

end NUMINAMATH_CALUDE_worker_payment_l3487_348719


namespace NUMINAMATH_CALUDE_smallest_share_is_five_thirds_l3487_348701

/-- Represents the shares of bread in an arithmetic sequence -/
structure BreadShares where
  a : ℚ  -- The middle term of the arithmetic sequence
  d : ℚ  -- The common difference of the arithmetic sequence
  sum_equals_100 : 5 * a = 100
  larger_three_seventh_smaller_two : 3 * a + 3 * d = 7 * (2 * a - 3 * d)
  d_positive : d > 0

/-- The smallest share of bread -/
def smallest_share (shares : BreadShares) : ℚ :=
  shares.a - 2 * shares.d

/-- Theorem stating that the smallest share is 5/3 -/
theorem smallest_share_is_five_thirds (shares : BreadShares) :
  smallest_share shares = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_share_is_five_thirds_l3487_348701


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3487_348749

theorem average_speed_calculation (distance1 distance2 speed1 speed2 : ℝ) 
  (h1 : distance1 = 20)
  (h2 : distance2 = 40)
  (h3 : speed1 = 8)
  (h4 : speed2 = 20) :
  (distance1 + distance2) / (distance1 / speed1 + distance2 / speed2) = 40/3 :=
sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3487_348749


namespace NUMINAMATH_CALUDE_problem_statement_l3487_348772

theorem problem_statement (a b c d : ℤ) 
  (h1 : a - b - c + d = 13) 
  (h2 : a + b - c - d = 5) : 
  (b - d)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3487_348772


namespace NUMINAMATH_CALUDE_set_a_contains_one_l3487_348729

theorem set_a_contains_one (a : ℝ) : 
  1 ∈ ({a + 2, (a + 1)^2, a^2 + 3*a + 3} : Set ℝ) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_a_contains_one_l3487_348729


namespace NUMINAMATH_CALUDE_existence_of_many_prime_factors_l3487_348755

theorem existence_of_many_prime_factors (N : ℕ+) :
  ∃ n : ℕ+, ∃ p : Finset ℕ,
    (∀ q ∈ p, Nat.Prime q) ∧
    (Finset.card p ≥ N) ∧
    (∀ q ∈ p, q ∣ (n^2013 - n^20 + n^13 - 2013)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_many_prime_factors_l3487_348755


namespace NUMINAMATH_CALUDE_student_line_count_l3487_348761

/-- The number of students in the line -/
def num_students : ℕ := 26

/-- The counting cycle -/
def cycle_length : ℕ := 4

/-- The last number called -/
def last_number : ℕ := 2

theorem student_line_count :
  num_students % cycle_length = last_number :=
by sorry

end NUMINAMATH_CALUDE_student_line_count_l3487_348761


namespace NUMINAMATH_CALUDE_gcd_143_144_l3487_348726

theorem gcd_143_144 : Nat.gcd 143 144 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_143_144_l3487_348726


namespace NUMINAMATH_CALUDE_expression_evaluation_l3487_348797

theorem expression_evaluation : -6 * 3 - (-8 * -2) + (-7 * -5) - 10 = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3487_348797


namespace NUMINAMATH_CALUDE_raccoon_carrots_l3487_348739

theorem raccoon_carrots (raccoon_per_hole rabbit_per_hole : ℕ) 
  (hole_difference : ℕ) (total_carrots : ℕ) : 
  raccoon_per_hole = 5 →
  rabbit_per_hole = 8 →
  hole_difference = 3 →
  raccoon_per_hole * (hole_difference + total_carrots / rabbit_per_hole) = total_carrots →
  total_carrots = 40 :=
by
  sorry

#check raccoon_carrots

end NUMINAMATH_CALUDE_raccoon_carrots_l3487_348739


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3487_348702

theorem framed_painting_ratio : 
  let painting_width : ℝ := 28
  let painting_height : ℝ := 32
  let frame_side_width : ℝ := 10/3
  let frame_top_bottom_width : ℝ := 3 * frame_side_width
  let framed_width : ℝ := painting_width + 2 * frame_side_width
  let framed_height : ℝ := painting_height + 2 * frame_top_bottom_width
  let frame_area : ℝ := framed_width * framed_height - painting_width * painting_height
  frame_area = painting_width * painting_height →
  framed_width / framed_height = 26 / 35 :=
by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3487_348702


namespace NUMINAMATH_CALUDE_high_octane_half_cost_l3487_348769

/-- Represents the composition and cost of a fuel mixture -/
structure FuelMixture where
  high_octane_units : ℕ
  regular_octane_units : ℕ
  high_octane_cost_multiplier : ℕ

/-- Calculates the fraction of the total cost due to high octane fuel -/
def high_octane_cost_fraction (fuel : FuelMixture) : ℚ :=
  let high_octane_cost := fuel.high_octane_units * fuel.high_octane_cost_multiplier
  let regular_octane_cost := fuel.regular_octane_units
  let total_cost := high_octane_cost + regular_octane_cost
  high_octane_cost / total_cost

/-- Theorem: The fraction of the cost due to high octane is 1/2 for the given fuel mixture -/
theorem high_octane_half_cost (fuel : FuelMixture) 
    (h1 : fuel.high_octane_units = 1515)
    (h2 : fuel.regular_octane_units = 4545)
    (h3 : fuel.high_octane_cost_multiplier = 3) :
  high_octane_cost_fraction fuel = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_high_octane_half_cost_l3487_348769


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3487_348721

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x - 5| = 3*x + 6 :=
by
  -- The unique solution is x = -1/4
  use (-1/4 : ℚ)
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3487_348721


namespace NUMINAMATH_CALUDE_average_comparisons_equals_size_l3487_348774

/-- Represents a sequential search on an unordered array -/
structure SequentialSearch where
  /-- The number of elements in the array -/
  size : ℕ
  /-- Predicate indicating if the array is unordered -/
  unordered : Prop
  /-- Predicate indicating if the searched element is not in the array -/
  element_not_present : Prop

/-- The average number of comparisons needed in a sequential search -/
def average_comparisons (search : SequentialSearch) : ℕ := sorry

/-- Theorem stating that the average number of comparisons is equal to the array size 
    when the element is not present in an unordered array -/
theorem average_comparisons_equals_size (search : SequentialSearch) 
  (h_size : search.size = 100)
  (h_unordered : search.unordered)
  (h_not_present : search.element_not_present) :
  average_comparisons search = search.size := by sorry

end NUMINAMATH_CALUDE_average_comparisons_equals_size_l3487_348774


namespace NUMINAMATH_CALUDE_faye_coloring_books_l3487_348752

def coloring_books_problem (initial_books : ℝ) (first_giveaway : ℝ) (second_giveaway : ℝ) : Prop :=
  initial_books - first_giveaway - second_giveaway = 11.0

theorem faye_coloring_books :
  coloring_books_problem 48.0 34.0 3.0 := by
  sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l3487_348752


namespace NUMINAMATH_CALUDE_solution_set_correct_l3487_348760

/-- The solution set of the system of equations y² = x and y = x -/
def solution_set : Set (ℝ × ℝ) := {(1, 1), (0, 0)}

/-- The system of equations y² = x and y = x -/
def system_equations (p : ℝ × ℝ) : Prop :=
  p.2 ^ 2 = p.1 ∧ p.2 = p.1

theorem solution_set_correct :
  ∀ p : ℝ × ℝ, p ∈ solution_set ↔ system_equations p := by
  sorry

end NUMINAMATH_CALUDE_solution_set_correct_l3487_348760


namespace NUMINAMATH_CALUDE_range_of_m_l3487_348762

-- Define the ellipse C
def ellipse_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m + y^2 / (8 - m) = 1

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x - y + m = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

-- Define proposition p
def prop_p (m : ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ m = 4 + c ∧ 8 - m = 4 - c

-- Define proposition q
def prop_q (m : ℝ) : Prop :=
  abs m ≤ 3 * Real.sqrt 2

-- Main theorem
theorem range_of_m :
  ∀ m : ℝ, (prop_p m ∧ ¬prop_q m) ∨ (¬prop_p m ∧ prop_q m) →
    (3 * Real.sqrt 2 < m ∧ m < 8) ∨ (-3 * Real.sqrt 2 ≤ m ∧ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3487_348762


namespace NUMINAMATH_CALUDE_railway_theorem_l3487_348708

structure City where
  id : Nat

structure DirectedGraph where
  cities : Set City
  connections : City → City → Prop

def reachable (g : DirectedGraph) (a b : City) : Prop :=
  g.connections a b ∨ ∃ c, g.connections a c ∧ g.connections c b

theorem railway_theorem (g : DirectedGraph) 
  (h₁ : ∀ a b : City, a ∈ g.cities → b ∈ g.cities → a ≠ b → (g.connections a b ∨ g.connections b a)) :
  ∃ n : City, n ∈ g.cities ∧ ∀ m : City, m ∈ g.cities → m ≠ n → reachable g m n :=
sorry

end NUMINAMATH_CALUDE_railway_theorem_l3487_348708


namespace NUMINAMATH_CALUDE_hill_climb_speed_l3487_348754

/-- Proves that given a journey with an uphill climb taking 4 hours and a downhill descent
    taking 2 hours, if the average speed for the entire journey is 1.5 km/h,
    then the average speed for the uphill climb is 1.125 km/h. -/
theorem hill_climb_speed (distance : ℝ) (climb_time : ℝ) (descent_time : ℝ) 
    (average_speed : ℝ) (h1 : climb_time = 4) (h2 : descent_time = 2) 
    (h3 : average_speed = 1.5) :
  distance / climb_time = 1.125 := by
  sorry

end NUMINAMATH_CALUDE_hill_climb_speed_l3487_348754


namespace NUMINAMATH_CALUDE_expression_simplification_l3487_348731

theorem expression_simplification :
  let a : ℝ := Real.sqrt 3
  let b : ℝ := Real.sqrt 2
  let c : ℝ := Real.sqrt 5
  6 * 37 * (a + b) ^ (2 * (Real.log c / Real.log (a - b))) = 1110 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3487_348731


namespace NUMINAMATH_CALUDE_fitness_center_membership_ratio_l3487_348733

theorem fitness_center_membership_ratio 
  (f m : ℕ) -- f: number of female members, m: number of male members
  (hf : f > 0) -- ensure f is positive
  (hm : m > 0) -- ensure m is positive
  (h_avg : (45 * f + 20 * m) / (f + m) = 25) : -- condition for overall average age
  f / m = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_fitness_center_membership_ratio_l3487_348733


namespace NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l3487_348783

theorem coefficient_m5n5_in_expansion : (Nat.choose 10 5) = 252 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l3487_348783


namespace NUMINAMATH_CALUDE_side_view_area_is_four_l3487_348713

/-- Represents a triangular prism -/
structure TriangularPrism where
  lateral_edge_length : ℝ
  base_side_length : ℝ
  main_view_side_length : ℝ

/-- The area of the side view of a triangular prism -/
def side_view_area (prism : TriangularPrism) : ℝ :=
  prism.lateral_edge_length * prism.base_side_length

/-- Theorem: The area of the side view of a specific triangular prism is 4 -/
theorem side_view_area_is_four :
  ∀ (prism : TriangularPrism),
    prism.lateral_edge_length = 2 →
    prism.base_side_length = 2 →
    prism.main_view_side_length = 2 →
    side_view_area prism = 4 := by
  sorry

end NUMINAMATH_CALUDE_side_view_area_is_four_l3487_348713


namespace NUMINAMATH_CALUDE_triangle_theorem_l3487_348744

theorem triangle_theorem (a b c A B C : ℝ) (h1 : a * Real.cos C + (1/2) * c = b)
                         (h2 : b = 4) (h3 : c = 6) : 
  A = π/3 ∧ Real.cos B = 2/Real.sqrt 7 ∧ Real.cos (A + 2*B) = -11/14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3487_348744


namespace NUMINAMATH_CALUDE_meeting_arrangements_presidency_meeting_arrangements_l3487_348792

/-- Represents a school in the club --/
structure School :=
  (members : Nat)

/-- Represents the club --/
structure Club :=
  (schools : Finset School)
  (total_members : Nat)

/-- Represents a meeting arrangement --/
structure MeetingArrangement :=
  (host : School)
  (host_representatives : Nat)
  (other_representatives : Nat)

/-- The number of ways to choose k items from n items --/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

/-- Theorem: Number of possible meeting arrangements --/
theorem meeting_arrangements (club : Club) (arrangement : MeetingArrangement) : Nat :=
  let num_schools := Finset.card club.schools
  let host_choices := choose num_schools 1
  let host_rep_choices := choose arrangement.host.members arrangement.host_representatives
  let other_rep_choices := (choose arrangement.host.members arrangement.other_representatives) ^ (num_schools - 1)
  host_choices * host_rep_choices * other_rep_choices

/-- Main theorem: Prove the number of possible arrangements is 40,000 --/
theorem presidency_meeting_arrangements :
  ∀ (club : Club) (arrangement : MeetingArrangement),
    Finset.card club.schools = 4 →
    (∀ s ∈ club.schools, s.members = 5) →
    club.total_members = 20 →
    arrangement.host_representatives = 3 →
    arrangement.other_representatives = 2 →
    meeting_arrangements club arrangement = 40000 :=
sorry

end NUMINAMATH_CALUDE_meeting_arrangements_presidency_meeting_arrangements_l3487_348792


namespace NUMINAMATH_CALUDE_division_problem_l3487_348786

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 190 →
  quotient = 9 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 21 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3487_348786


namespace NUMINAMATH_CALUDE_prime_solution_equation_l3487_348720

theorem prime_solution_equation : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 ∧ p = 17 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_solution_equation_l3487_348720


namespace NUMINAMATH_CALUDE_jenn_savings_l3487_348775

/-- Represents the value of a coin in cents -/
def coinValue (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | "penny" => 1
  | _ => 0

/-- Calculates the total value of coins in a jar -/
def jarValue (coin : String) (count : ℕ) : ℚ :=
  (coinValue coin * count : ℚ) / 100

/-- Calculates the available amount after applying the usage constraint -/
def availableAmount (amount : ℚ) (constraint : ℚ) : ℚ :=
  amount * constraint

/-- Represents Jenn's saving scenario -/
structure SavingScenario where
  quarterJars : ℕ
  quarterCount : ℕ
  dimeJars : ℕ
  dimeCount : ℕ
  nickelJars : ℕ
  nickelCount : ℕ
  monthlyPennies : ℕ
  months : ℕ
  usageConstraint : ℚ
  bikeCost : ℚ

/-- Theorem stating that Jenn will have $24.57 left after buying the bike -/
theorem jenn_savings (scenario : SavingScenario) : 
  scenario.quarterJars = 4 ∧ 
  scenario.quarterCount = 160 ∧
  scenario.dimeJars = 4 ∧
  scenario.dimeCount = 300 ∧
  scenario.nickelJars = 2 ∧
  scenario.nickelCount = 500 ∧
  scenario.monthlyPennies = 12 ∧
  scenario.months = 6 ∧
  scenario.usageConstraint = 4/5 ∧
  scenario.bikeCost = 240 →
  let totalQuarters := jarValue "quarter" (scenario.quarterJars * scenario.quarterCount)
  let totalDimes := jarValue "dime" (scenario.dimeJars * scenario.dimeCount)
  let totalNickels := jarValue "nickel" (scenario.nickelJars * scenario.nickelCount)
  let totalPennies := jarValue "penny" (scenario.monthlyPennies * scenario.months)
  let availableQuarters := availableAmount totalQuarters scenario.usageConstraint
  let availableDimes := availableAmount totalDimes scenario.usageConstraint
  let availableNickels := availableAmount totalNickels scenario.usageConstraint
  let availablePennies := availableAmount totalPennies scenario.usageConstraint
  let totalAvailable := availableQuarters + availableDimes + availableNickels + availablePennies
  totalAvailable - scenario.bikeCost = 24.57 := by
  sorry

end NUMINAMATH_CALUDE_jenn_savings_l3487_348775


namespace NUMINAMATH_CALUDE_common_chord_length_l3487_348705

/-- Given two intersecting circles with radii in ratio 4:3, prove that the length of their common chord is 2√2 when the segment connecting their centers is divided into parts of length 5 and 2 by the common chord. -/
theorem common_chord_length (r₁ r₂ : ℝ) (h_ratio : r₁ = (4/3) * r₂) 
  (center_distance : ℝ) (h_center_distance : center_distance = 7)
  (segment_1 segment_2 : ℝ) (h_segment_1 : segment_1 = 5) (h_segment_2 : segment_2 = 2)
  (h_segments_sum : segment_1 + segment_2 = center_distance) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l3487_348705


namespace NUMINAMATH_CALUDE_koala_bear_ratio_is_one_half_l3487_348785

/-- Represents the number of tickets spent on different items -/
structure TicketSpending where
  total : ℕ
  earbuds : ℕ
  glowBracelets : ℕ
  koalaBear : ℕ

/-- The ratio of tickets spent on the koala bear to the total number of tickets -/
def koalaBearRatio (ts : TicketSpending) : Rat :=
  ts.koalaBear / ts.total

theorem koala_bear_ratio_is_one_half (ts : TicketSpending) 
  (h_total : ts.total = 50)
  (h_earbuds : ts.earbuds = 10)
  (h_glow : ts.glowBracelets = 15)
  (h_koala : ts.koalaBear = ts.total - ts.earbuds - ts.glowBracelets) :
  koalaBearRatio ts = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_koala_bear_ratio_is_one_half_l3487_348785


namespace NUMINAMATH_CALUDE_initial_sum_is_500_l3487_348759

/-- Prove that the initial sum of money is $500 given the conditions of the problem. -/
theorem initial_sum_is_500 
  (sum_after_2_years : ℝ → ℝ → ℝ → ℝ) -- Function for final amount after 2 years
  (initial_sum : ℝ)  -- Initial sum of money
  (interest_rate : ℝ) -- Original interest rate
  (h1 : sum_after_2_years initial_sum interest_rate 2 = 600) -- First condition
  (h2 : sum_after_2_years initial_sum (interest_rate + 0.1) 2 = 700) -- Second condition
  : initial_sum = 500 := by
  sorry

end NUMINAMATH_CALUDE_initial_sum_is_500_l3487_348759


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l3487_348787

theorem system_of_equations_solutions :
  -- System (1)
  let x₁ := -1
  let y₁ := 1
  -- System (2)
  let x₂ := 5 / 2
  let y₂ := -2
  -- Proof statements
  (x₁ = y₁ - 2 ∧ 3 * x₁ + 2 * y₁ = -1) ∧
  (2 * x₂ - 3 * y₂ = 11 ∧ 4 * x₂ + 5 * y₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l3487_348787


namespace NUMINAMATH_CALUDE_current_rate_l3487_348727

/-- Given a man's rowing speeds, calculate the rate of the current -/
theorem current_rate (downstream_speed upstream_speed still_water_speed : ℝ)
  (h1 : downstream_speed = 30)
  (h2 : upstream_speed = 10)
  (h3 : still_water_speed = 20) :
  downstream_speed - still_water_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_current_rate_l3487_348727


namespace NUMINAMATH_CALUDE_P_complement_subset_Q_l3487_348750

def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > -1}
def P_complement : Set ℝ := {x | x ≥ 1}

theorem P_complement_subset_Q : P_complement ⊆ Q := by
  sorry

end NUMINAMATH_CALUDE_P_complement_subset_Q_l3487_348750


namespace NUMINAMATH_CALUDE_circuit_equation_l3487_348707

/-- Given voltage and impedance, prove the current satisfies the equation V = IZ -/
theorem circuit_equation (V Z I : ℂ) (hV : V = 2 + 3*I) (hZ : Z = 2 - I) : 
  V = I * Z ↔ I = (1 : ℝ)/5 + (8 : ℝ)/5 * I :=
sorry

end NUMINAMATH_CALUDE_circuit_equation_l3487_348707


namespace NUMINAMATH_CALUDE_perpendicular_vectors_a_equals_two_l3487_348712

-- Define the vectors m and n
def m : ℝ × ℝ := (1, 2)
def n (a : ℝ) : ℝ × ℝ := (a, -1)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem perpendicular_vectors_a_equals_two :
  ∀ a : ℝ, dot_product m (n a) = 0 → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_vectors_a_equals_two_l3487_348712


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3487_348714

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ) = Real.sqrt (3^a * 3^b) → 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3487_348714


namespace NUMINAMATH_CALUDE_sum_of_non_solutions_l3487_348741

/-- Given an equation with infinitely many solutions, prove the sum of non-solutions -/
theorem sum_of_non_solutions (A B C : ℚ) : 
  (∀ x, (x + B) * (A * x + 16) = 3 * (x + C) * (x + 5)) →
  (∃ x₁ x₂, ∀ x, x ≠ x₁ ∧ x ≠ x₂ → (x + B) * (A * x + 16) = 3 * (x + C) * (x + 5)) →
  (∃ x₁ x₂, ∀ x, (x + B) * (A * x + 16) ≠ 3 * (x + C) * (x + 5) ↔ x = x₁ ∨ x = x₂) →
  (∃ x₁ x₂, x₁ + x₂ = -31/3 ∧ 
    ∀ x, (x + B) * (A * x + 16) ≠ 3 * (x + C) * (x + 5) ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_non_solutions_l3487_348741


namespace NUMINAMATH_CALUDE_f_major_premise_incorrect_l3487_348756

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- State that f'(0) = 0
theorem f'_zero : f' 0 = 0 := by sorry

-- Define what it means for a point to be an extremum
def is_extremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≥ f x ∨ f x₀ ≤ f x

-- Theorem stating that the major premise is incorrect
theorem major_premise_incorrect :
  ¬(∀ x₀ : ℝ, f' x₀ = 0 → is_extremum f x₀) := by sorry

end NUMINAMATH_CALUDE_f_major_premise_incorrect_l3487_348756


namespace NUMINAMATH_CALUDE_sword_length_difference_main_result_l3487_348734

/-- Proves that Jameson's sword is 3 inches longer than twice Christopher's sword length -/
theorem sword_length_difference : ℕ → ℕ → ℕ → Prop :=
  fun christopher_length june_christopher_diff jameson_june_diff =>
    let christopher_length : ℕ := 15
    let june_christopher_diff : ℕ := 23
    let jameson_june_diff : ℕ := 5
    let june_length : ℕ := christopher_length + june_christopher_diff
    let jameson_length : ℕ := june_length - jameson_june_diff
    let twice_christopher_length : ℕ := 2 * christopher_length
    jameson_length - twice_christopher_length = 3

/-- Main theorem stating the result -/
theorem main_result : sword_length_difference 15 23 5 := by
  sorry

end NUMINAMATH_CALUDE_sword_length_difference_main_result_l3487_348734


namespace NUMINAMATH_CALUDE_sachins_age_l3487_348718

theorem sachins_age (sachin rahul : ℕ) : 
  rahul = sachin + 18 →
  sachin * 9 = rahul * 7 →
  sachin = 63 := by sorry

end NUMINAMATH_CALUDE_sachins_age_l3487_348718


namespace NUMINAMATH_CALUDE_solution_count_l3487_348788

/-- The number of solutions to the system of equations y = (x+1)^3 and xy + y = 1 -/
def num_solutions : ℕ := 4

/-- The number of real solutions to the system of equations y = (x+1)^3 and xy + y = 1 -/
def num_real_solutions : ℕ := 2

/-- The number of complex solutions to the system of equations y = (x+1)^3 and xy + y = 1 -/
def num_complex_solutions : ℕ := 2

/-- Definition of the first equation: y = (x+1)^3 -/
def equation1 (x y : ℂ) : Prop := y = (x + 1)^3

/-- Definition of the second equation: xy + y = 1 -/
def equation2 (x y : ℂ) : Prop := x * y + y = 1

/-- A solution is a pair (x, y) that satisfies both equations -/
def is_solution (x y : ℂ) : Prop := equation1 x y ∧ equation2 x y

/-- The main theorem stating the number and nature of solutions -/
theorem solution_count :
  (∃ (s : Finset (ℂ × ℂ)), s.card = num_solutions ∧
    (∀ (p : ℂ × ℂ), p ∈ s ↔ is_solution p.1 p.2) ∧
    (∃ (sr : Finset (ℝ × ℝ)), sr.card = num_real_solutions ∧
      (∀ (p : ℝ × ℝ), p ∈ sr ↔ is_solution p.1 p.2)) ∧
    (∃ (sc : Finset (ℂ × ℂ)), sc.card = num_complex_solutions ∧
      (∀ (p : ℂ × ℂ), p ∈ sc ↔ (is_solution p.1 p.2 ∧ ¬(p.1.im = 0 ∧ p.2.im = 0))))) :=
sorry

end NUMINAMATH_CALUDE_solution_count_l3487_348788


namespace NUMINAMATH_CALUDE_exists_m_intersecting_line_and_circle_l3487_348730

/-- A line intersects a circle if and only if the distance from the center of the circle to the line is less than the radius of the circle. -/
axiom line_intersects_circle_iff_distance_lt_radius {a b c x₀ y₀ r : ℝ} :
  (∃ x y, a * x + b * y + c = 0 ∧ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2) < r

/-- The theorem stating that there exists an integer m between 2 and 7 (exclusive) such that the line 4x + 3y + 2m = 0 intersects with the circle (x + 3)² + (y - 1)² = 1. -/
theorem exists_m_intersecting_line_and_circle :
  ∃ m : ℤ, 2 < m ∧ m < 7 ∧
  (∃ x y : ℝ, 4 * x + 3 * y + 2 * (m : ℝ) = 0 ∧ (x + 3)^2 + (y - 1)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_intersecting_line_and_circle_l3487_348730


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l3487_348766

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l3487_348766


namespace NUMINAMATH_CALUDE_lcm_and_sum_of_numbers_l3487_348757

def numbers : List Nat := [14, 21, 35]

theorem lcm_and_sum_of_numbers :
  (Nat.lcm (Nat.lcm 14 21) 35 = 210) ∧ (numbers.sum = 70) := by
  sorry

end NUMINAMATH_CALUDE_lcm_and_sum_of_numbers_l3487_348757


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l3487_348795

def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x + 1

theorem max_value_of_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 3 ∧ 
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f c) ∧
  f c = 10 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l3487_348795


namespace NUMINAMATH_CALUDE_abs_diff_lt_abs_one_minus_prod_l3487_348793

theorem abs_diff_lt_abs_one_minus_prod {x y : ℝ} (hx : |x| < 1) (hy : |y| < 1) :
  |x - y| < |1 - x * y| := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_lt_abs_one_minus_prod_l3487_348793


namespace NUMINAMATH_CALUDE_M_equals_set_l3487_348784

def M : Set ℕ := {m | m > 0 ∧ ∃ k : ℤ, (10 : ℤ) = k * (m + 1)}

theorem M_equals_set : M = {1, 4, 9} := by sorry

end NUMINAMATH_CALUDE_M_equals_set_l3487_348784


namespace NUMINAMATH_CALUDE_max_square_side_length_l3487_348717

theorem max_square_side_length (length width : ℕ) (h1 : length = 54) (h2 : width = 24) :
  Nat.gcd length width = 6 :=
sorry

end NUMINAMATH_CALUDE_max_square_side_length_l3487_348717


namespace NUMINAMATH_CALUDE_periodic_function_property_l3487_348738

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β), if f(4) = 3, then f(2017) = -3 -/
theorem periodic_function_property (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 4 = 3 → f 2017 = -3 := by
  sorry


end NUMINAMATH_CALUDE_periodic_function_property_l3487_348738


namespace NUMINAMATH_CALUDE_merchant_profit_calculation_l3487_348740

theorem merchant_profit_calculation (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  markup_percentage = 20 →
  discount_percentage = 5 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 14 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_calculation_l3487_348740


namespace NUMINAMATH_CALUDE_amelias_dinner_l3487_348746

/-- Amelia's dinner problem -/
theorem amelias_dinner (first_course second_course dessert remaining_money : ℝ) 
  (h1 : first_course = 15)
  (h2 : second_course = first_course + 5)
  (h3 : dessert = 0.25 * second_course)
  (h4 : remaining_money = 20) : 
  first_course + second_course + dessert + remaining_money = 60 := by
  sorry

end NUMINAMATH_CALUDE_amelias_dinner_l3487_348746


namespace NUMINAMATH_CALUDE_livestock_theorem_l3487_348799

/-- Represents the value of livestock in taels of silver -/
structure LivestockValue where
  cow : ℕ
  sheep : ℕ
  total : ℕ

/-- Represents a purchase of livestock -/
structure Purchase where
  cows : ℕ
  sheep : ℕ

/-- The main theorem about livestock values and purchases -/
theorem livestock_theorem 
  (eq1 : LivestockValue)
  (eq2 : LivestockValue)
  (h1 : eq1.cow = 5 ∧ eq1.sheep = 2 ∧ eq1.total = 19)
  (h2 : eq2.cow = 2 ∧ eq2.sheep = 5 ∧ eq2.total = 16) :
  (∃ (cow_value sheep_value : ℕ),
    cow_value = 3 ∧ sheep_value = 2 ∧
    eq1.cow * cow_value + eq1.sheep * sheep_value = eq1.total ∧
    eq2.cow * cow_value + eq2.sheep * sheep_value = eq2.total) ∧
  (∃ (purchases : List Purchase),
    purchases.length = 3 ∧
    purchases.all (λ p => p.cows > 0 ∧ p.sheep > 0 ∧ p.cows * 3 + p.sheep * 2 = 20) ∧
    ∀ p : Purchase, p.cows > 0 → p.sheep > 0 → p.cows * 3 + p.sheep * 2 = 20 → p ∈ purchases) :=
by sorry

end NUMINAMATH_CALUDE_livestock_theorem_l3487_348799


namespace NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l3487_348723

theorem min_value_x_plus_four_over_x :
  ∀ x : ℝ, x > 0 → x + 4 / x ≥ 4 ∧ ∃ y : ℝ, y > 0 ∧ y + 4 / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l3487_348723


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_2005_l3487_348700

/-- The units digit of 3^n for n ≥ 1 -/
def units_digit_of_3_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 1

/-- The units digit of 3^2005 is 3 -/
theorem units_digit_of_3_pow_2005 : units_digit_of_3_pow 2005 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_2005_l3487_348700


namespace NUMINAMATH_CALUDE_prime_divisors_theorem_l3487_348709

def f (p : ℕ) : ℕ := 3^p + 4^p + 5^p + 9^p - 98

theorem prime_divisors_theorem (p : ℕ) :
  Prime p ↔ (Nat.card (Nat.divisors (f p)) ≤ 6 ↔ p = 2 ∨ p = 3) := by sorry

end NUMINAMATH_CALUDE_prime_divisors_theorem_l3487_348709


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3487_348725

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) + Real.sqrt (x - 2) > Real.sqrt (x - 4) + Real.sqrt (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3487_348725


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3487_348722

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  q > 1 →
  (4 * (a 2016)^2 - 8 * (a 2016) + 3 = 0) →
  (4 * (a 2017)^2 - 8 * (a 2017) + 3 = 0) →
  a 2018 + a 2019 = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3487_348722


namespace NUMINAMATH_CALUDE_swap_digits_two_digit_number_l3487_348777

theorem swap_digits_two_digit_number (x : ℕ) (h : 9 < x ∧ x < 100) :
  let a : ℕ := x / 10
  let b : ℕ := x % 10
  10 * b + a = 10 * (x % 10) + (x / 10) :=
by
  sorry

end NUMINAMATH_CALUDE_swap_digits_two_digit_number_l3487_348777


namespace NUMINAMATH_CALUDE_inequality_proof_l3487_348745

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3487_348745


namespace NUMINAMATH_CALUDE_margarets_mean_score_l3487_348710

def scores : List ℝ := [86, 88, 91, 93, 95, 97, 99, 100]

theorem margarets_mean_score 
  (h1 : scores.length = 8)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    margaret_scores.length = 4 ∧ 
    cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    cyprian_scores.sum / cyprian_scores.length = 92) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 4 ∧ 
    margaret_scores.sum / margaret_scores.length = 95.25 := by
  sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l3487_348710


namespace NUMINAMATH_CALUDE_rectangular_yard_area_l3487_348773

theorem rectangular_yard_area (L W : ℝ) : 
  L = 40 →  -- One full side (length) is 40 feet
  2 * W + L = 52 →  -- Total fencing for three sides is 52 feet
  L * W = 240 :=  -- Area of the yard is 240 square feet
by
  sorry

end NUMINAMATH_CALUDE_rectangular_yard_area_l3487_348773


namespace NUMINAMATH_CALUDE_cake_recipe_salt_l3487_348765

theorem cake_recipe_salt (sugar_total : ℕ) (salt : ℕ) : 
  sugar_total = 8 → 
  sugar_total = salt + 1 → 
  salt = 7 := by
sorry

end NUMINAMATH_CALUDE_cake_recipe_salt_l3487_348765


namespace NUMINAMATH_CALUDE_notebook_purchase_savings_l3487_348703

theorem notebook_purchase_savings (s : ℚ) (n : ℚ) (p : ℚ) 
  (h1 : s > 0) (h2 : n > 0) (h3 : p > 0) 
  (h4 : (1/4) * s = (1/2) * n * p) : 
  s - n * p = (1/2) * s := by
sorry

end NUMINAMATH_CALUDE_notebook_purchase_savings_l3487_348703


namespace NUMINAMATH_CALUDE_quadratic_properties_l3487_348771

/-- A quadratic function with a < 0, f(-1) = 0, and axis of symmetry x = 1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_neg : a < 0
  root_neg_one : a * (-1)^2 + b * (-1) + c = 0
  axis_sym : -b / (2 * a) = 1

/-- Properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (f.a - f.b + f.c = 0) ∧
  (∀ m : ℝ, f.a * m^2 + f.b * m + f.c ≤ -4 * f.a) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → 
    f.a * x₁^2 + f.b * x₁ + f.c + 1 = 0 → 
    f.a * x₂^2 + f.b * x₂ + f.c + 1 = 0 → 
    x₁ < -1 ∧ x₂ > 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3487_348771


namespace NUMINAMATH_CALUDE_subsets_sum_to_negative_eight_l3487_348747

def S : Finset Int := {-6, -4, -2, -1, 1, 2, 3, 4, 6}

theorem subsets_sum_to_negative_eight :
  ∃! (subsets : Finset (Finset Int)),
    (∀ subset ∈ subsets, subset ⊆ S ∧ (subset.sum id = -8)) ∧
    subsets.card = 6 :=
by sorry

end NUMINAMATH_CALUDE_subsets_sum_to_negative_eight_l3487_348747


namespace NUMINAMATH_CALUDE_monotonic_quadratic_l3487_348794

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 3

theorem monotonic_quadratic (m : ℝ) :
  (∀ x₁ x₂, 1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 → (f m x₁ < f m x₂ ∨ f m x₁ > f m x₂)) →
  m ≤ 1 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_l3487_348794


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_l3487_348753

def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

def carl_cubes : ℕ := 6
def carl_side_length : ℝ := 1

def kate_cubes : ℕ := 4
def kate_side_length : ℝ := 3

theorem total_volume_of_cubes :
  (carl_cubes : ℝ) * cube_volume carl_side_length +
  (kate_cubes : ℝ) * cube_volume kate_side_length = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_cubes_l3487_348753


namespace NUMINAMATH_CALUDE_goats_in_field_l3487_348743

theorem goats_in_field (total_animals cows sheep : ℕ) 
  (h1 : total_animals = 200)
  (h2 : cows = 40)
  (h3 : sheep = 56) : 
  total_animals - (cows + sheep) = 104 := by
  sorry

end NUMINAMATH_CALUDE_goats_in_field_l3487_348743


namespace NUMINAMATH_CALUDE_transport_probabilities_l3487_348748

structure TransportProbabilities where
  train : ℝ
  ship : ℝ
  car : ℝ
  airplane : ℝ
  mutually_exclusive : train + ship + car + airplane = 1
  going_probability : ℝ

def prob : TransportProbabilities :=
  { train := 0.3
  , ship := 0.2
  , car := 0.1
  , airplane := 0.4
  , mutually_exclusive := by sorry
  , going_probability := 0.5
  }

theorem transport_probabilities (p : TransportProbabilities) :
  (p.train + p.airplane = 0.7) ∧
  (1 - p.ship = 0.8) ∧
  ((p.train + p.ship = p.going_probability) ∨ (p.car + p.airplane = p.going_probability)) :=
by sorry

end NUMINAMATH_CALUDE_transport_probabilities_l3487_348748


namespace NUMINAMATH_CALUDE_min_value_theorem_l3487_348770

theorem min_value_theorem (x : ℝ) (h : x > 1) : x + 4 / (x - 1) ≥ 5 ∧ ∃ y > 1, y + 4 / (y - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3487_348770


namespace NUMINAMATH_CALUDE_intersection_when_a_half_subset_iff_a_range_l3487_348736

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- Theorem 1: When a = 1/2, A ∩ B = B
theorem intersection_when_a_half : A (1/2) ∩ B = B := by sorry

-- Theorem 2: B ⊆ A if and only if 0 ≤ a ≤ 1
theorem subset_iff_a_range : B ⊆ A a ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_half_subset_iff_a_range_l3487_348736


namespace NUMINAMATH_CALUDE_even_mono_decreasing_relation_l3487_348706

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f is monotonically decreasing on [0, +∞) if
    for all x, y ≥ 0, x < y implies f(x) > f(y) -/
def IsMonoDecreasingOnNonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x > f y

theorem even_mono_decreasing_relation
    (f : ℝ → ℝ)
    (h_even : IsEven f)
    (h_mono : IsMonoDecreasingOnNonnegatives f) :
    f 1 > f (-6) := by
  sorry

end NUMINAMATH_CALUDE_even_mono_decreasing_relation_l3487_348706


namespace NUMINAMATH_CALUDE_store_sales_increase_l3487_348728

/-- Represents a store's sales performance --/
structure StoreSales where
  original_price : ℝ
  original_quantity : ℝ
  discount_rate : ℝ
  quantity_increase_rate : ℝ

/-- Calculates the percentage change in gross income --/
def gross_income_change (s : StoreSales) : ℝ :=
  ((1 + s.quantity_increase_rate) * (1 - s.discount_rate) - 1) * 100

/-- Theorem: If a store applies a 10% discount and experiences a 15% increase in sales quantity,
    then the gross income increases by 3.5% --/
theorem store_sales_increase (s : StoreSales) 
  (h1 : s.discount_rate = 0.1)
  (h2 : s.quantity_increase_rate = 0.15) :
  gross_income_change s = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_store_sales_increase_l3487_348728


namespace NUMINAMATH_CALUDE_cars_added_during_play_l3487_348779

/-- The number of cars added during a play, given initial car counts and final total. -/
def cars_added (front_initial : ℕ) (back_multiplier : ℕ) (total_final : ℕ) : ℕ :=
  total_final - (front_initial + back_multiplier * front_initial)

/-- Theorem stating that 400 cars were added during the play. -/
theorem cars_added_during_play :
  cars_added 100 2 700 = 400 := by sorry

end NUMINAMATH_CALUDE_cars_added_during_play_l3487_348779


namespace NUMINAMATH_CALUDE_aquaflow_pumping_time_l3487_348711

/-- Aquaflow system pumping problem -/
theorem aquaflow_pumping_time 
  (initial_rate : ℝ) 
  (increased_rate : ℝ) 
  (initial_time : ℝ) 
  (target_volume : ℝ) : 
  initial_rate = 360 →
  increased_rate = 480 →
  initial_time = 0.5 →
  target_volume = 540 →
  ∃ (total_time : ℝ), 
    total_time = 75/60 ∧ 
    initial_rate * initial_time + 
    increased_rate * (total_time - initial_time) = target_volume :=
by sorry

end NUMINAMATH_CALUDE_aquaflow_pumping_time_l3487_348711


namespace NUMINAMATH_CALUDE_inequality_solution_l3487_348715

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3487_348715


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l3487_348758

/-- Given two parabolas that intersect the coordinate axes in four points forming a kite -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∀ x y : ℝ, (y = a * x^2 + 4 ∨ y = 6 - b * x^2) → 
    (x = 0 ∨ y = 0)) →  -- intersect coordinate axes
  (∃! p q r s : ℝ × ℝ, 
    (p.2 = a * p.1^2 + 4 ∨ p.2 = 6 - b * p.1^2) ∧
    (q.2 = a * q.1^2 + 4 ∨ q.2 = 6 - b * q.1^2) ∧
    (r.2 = a * r.1^2 + 4 ∨ r.2 = 6 - b * r.1^2) ∧
    (s.2 = a * s.1^2 + 4 ∨ s.2 = 6 - b * s.1^2) ∧
    (p.1 = 0 ∨ p.2 = 0) ∧ (q.1 = 0 ∨ q.2 = 0) ∧
    (r.1 = 0 ∨ r.2 = 0) ∧ (s.1 = 0 ∨ s.2 = 0)) →  -- exactly four intersection points
  (∃ d₁ d₂ : ℝ, d₁ * d₂ / 2 = 18) →  -- kite area is 18
  a + b = 2/81 :=
by sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l3487_348758


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l3487_348776

/-- Represents a complex figure made of toothpicks forming triangles -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  max_triangle_side : ℕ
  min_triangle_side : ℕ

/-- Calculates the minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ :=
  20

/-- Theorem stating that for the given figure, 20 toothpicks must be removed to eliminate all triangles -/
theorem min_toothpicks_removal (figure : ToothpickFigure) 
  (h1 : figure.total_toothpicks = 60)
  (h2 : figure.max_triangle_side = 3)
  (h3 : figure.min_triangle_side = 1) :
  min_toothpicks_to_remove figure = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_l3487_348776


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3487_348789

/-- Given vectors a and b in ℝ², prove that if a + b is perpendicular to b, then the second component of a is 8. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 = 1 ∧ b = (3, -2)) :
  (a + b) • b = 0 → a.2 = 8 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3487_348789


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3487_348767

/-- Given a point P in polar coordinates, find its symmetric point with respect to the pole -/
theorem symmetric_point_coordinates (r : ℝ) (θ : ℝ) :
  let P : ℝ × ℝ := (r, θ)
  let symmetric_polar : ℝ × ℝ := (r, θ + π)
  let symmetric_cartesian : ℝ × ℝ := (r * Real.cos (θ + π), r * Real.sin (θ + π))
  P = (2, -5 * π / 3) →
  symmetric_polar = (2, -2 * π / 3) ∧
  symmetric_cartesian = (-1, -Real.sqrt 3) := by
  sorry

#check symmetric_point_coordinates

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3487_348767


namespace NUMINAMATH_CALUDE_tunnel_length_proof_l3487_348780

/-- The length of a train in miles -/
def train_length : ℝ := 1.5

/-- The time difference in minutes between the front of the train entering the tunnel and the tail exiting -/
def time_difference : ℝ := 4

/-- The speed of the train in miles per hour -/
def train_speed : ℝ := 45

/-- The length of the tunnel in miles -/
def tunnel_length : ℝ := 1.5

theorem tunnel_length_proof :
  tunnel_length = train_speed * (time_difference / 60) - train_length :=
by sorry

end NUMINAMATH_CALUDE_tunnel_length_proof_l3487_348780


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3487_348751

/-- The roots of the quadratic equation ax² + 4ax + c = 0 are equal if and only if c = 4a, given that a ≠ 0 -/
theorem quadratic_equal_roots (a c : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, ∀ y : ℝ, a * y^2 + 4 * a * y + c = 0 ↔ y = x) ↔ c = 4 * a :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3487_348751


namespace NUMINAMATH_CALUDE_system_solution_l3487_348798

theorem system_solution (x y k : ℝ) : 
  (x + 2*y = k - 1) →
  (2*x + y = 5*k + 4) →
  (x + y = 5) →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3487_348798


namespace NUMINAMATH_CALUDE_rational_function_sum_l3487_348704

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_quadratic : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c
  q_cubic : ∃ a b c d : ℝ, ∀ x, q x = a * x^3 + b * x^2 + c * x + d
  p_cond : p 4 = 4
  q_cond1 : q 1 = 0
  q_cond2 : q 3 = 3
  q_factor : ∃ r : ℝ → ℝ, ∀ x, q x = (x - 2) * r x

/-- The main theorem -/
theorem rational_function_sum (f : RationalFunction) :
  ∃ p q : ℝ → ℝ, (∀ x, f.p x = p x ∧ f.q x = q x) ∧
  (∀ x, p x + q x = (1/2) * x^3 - (5/4) * x^2 + (17/4) * x) := by
  sorry

end NUMINAMATH_CALUDE_rational_function_sum_l3487_348704


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l3487_348778

/-- The original quadratic function -/
def original_quadratic (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The converted quadratic function -/
def converted_quadratic (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating the equivalence of the two quadratic functions -/
theorem quadratic_equivalence :
  ∀ x : ℝ, original_quadratic x = converted_quadratic x :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l3487_348778


namespace NUMINAMATH_CALUDE_power_of_product_equality_l3487_348724

theorem power_of_product_equality (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equality_l3487_348724


namespace NUMINAMATH_CALUDE_max_card_arrangement_l3487_348735

/-- A type representing the cards with numbers from 1 to 9 -/
inductive Card : Type
| one : Card
| two : Card
| three : Card
| four : Card
| five : Card
| six : Card
| seven : Card
| eight : Card
| nine : Card

/-- Convert a Card to its corresponding natural number -/
def card_to_nat (c : Card) : Nat :=
  match c with
  | Card.one => 1
  | Card.two => 2
  | Card.three => 3
  | Card.four => 4
  | Card.five => 5
  | Card.six => 6
  | Card.seven => 7
  | Card.eight => 8
  | Card.nine => 9

/-- Check if one card is divisible by another -/
def is_divisible (a b : Card) : Prop :=
  (card_to_nat a) % (card_to_nat b) = 0 ∨ (card_to_nat b) % (card_to_nat a) = 0

/-- A valid arrangement of cards -/
def valid_arrangement (arr : List Card) : Prop :=
  ∀ i, i + 1 < arr.length → is_divisible (arr.get ⟨i, by sorry⟩) (arr.get ⟨i + 1, by sorry⟩)

/-- The main theorem -/
theorem max_card_arrangement :
  ∃ (arr : List Card), arr.length = 8 ∧ valid_arrangement arr ∧
  ∀ (arr' : List Card), valid_arrangement arr' → arr'.length ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_card_arrangement_l3487_348735


namespace NUMINAMATH_CALUDE_zoo_animals_count_l3487_348790

theorem zoo_animals_count (penguins : ℕ) (polar_bears : ℕ) : 
  penguins = 21 → polar_bears = 2 * penguins → penguins + polar_bears = 63 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l3487_348790


namespace NUMINAMATH_CALUDE_work_completion_time_l3487_348763

theorem work_completion_time 
  (total_work : ℝ) 
  (a_rate : ℝ) 
  (ab_rate : ℝ) 
  (h1 : a_rate = total_work / 12) 
  (h2 : 10 * ab_rate + 9 * a_rate = total_work) :
  ab_rate = total_work / 40 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3487_348763


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3487_348742

theorem geometric_sequence_seventh_term 
  (a : ℝ) (r : ℝ) (h1 : r ≠ 0) 
  (h2 : a * r^3 = 16) 
  (h3 : a * r^8 = 2) : 
  a * r^6 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3487_348742


namespace NUMINAMATH_CALUDE_trig_inequality_l3487_348737

theorem trig_inequality (x : ℝ) : 
  2 * (Real.sin x)^4 + 3 * (Real.sin x)^2 * (Real.cos x)^2 + 5 * (Real.cos x)^4 ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l3487_348737


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l3487_348716

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 20)
  (h2 : selling_price = 35) :
  (selling_price - cost_price) / cost_price * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l3487_348716


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3487_348796

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) ↔
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3487_348796


namespace NUMINAMATH_CALUDE_square_of_integer_ending_in_five_l3487_348782

theorem square_of_integer_ending_in_five (a : ℤ) : (10 * a + 5)^2 = 100 * a * (a + 1) + 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_integer_ending_in_five_l3487_348782


namespace NUMINAMATH_CALUDE_senior_sports_solution_l3487_348781

def senior_sports_problem (total_seniors : ℕ) 
  (football : Finset ℕ) (baseball : Finset ℕ) (lacrosse : Finset ℕ) : Prop :=
  (total_seniors = 85) ∧
  (football.card = 74) ∧
  (baseball.card = 26) ∧
  ((football ∩ lacrosse).card = 17) ∧
  ((baseball ∩ football).card = 18) ∧
  ((baseball ∩ lacrosse).card = 13) ∧
  (lacrosse.card = 2 * (football ∩ baseball ∩ lacrosse).card) ∧
  (∀ s, s ∈ football ∪ baseball ∪ lacrosse) ∧
  ((football ∪ baseball ∪ lacrosse).card = total_seniors)

theorem senior_sports_solution 
  {total_seniors : ℕ} {football baseball lacrosse : Finset ℕ} 
  (h : senior_sports_problem total_seniors football baseball lacrosse) :
  (football ∩ baseball ∩ lacrosse).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_senior_sports_solution_l3487_348781
