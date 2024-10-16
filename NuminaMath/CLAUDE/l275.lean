import Mathlib

namespace NUMINAMATH_CALUDE_percentage_not_french_l275_27557

def total_students : ℕ := 200
def french_and_english : ℕ := 25
def french_not_english : ℕ := 65

theorem percentage_not_french : 
  (total_students - (french_and_english + french_not_english)) * 100 / total_students = 55 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_french_l275_27557


namespace NUMINAMATH_CALUDE_odd_sum_not_divisible_by_three_l275_27558

theorem odd_sum_not_divisible_by_three (x y z : ℕ) 
  (h_odd_x : Odd x) (h_odd_y : Odd y) (h_odd_z : Odd z)
  (h_positive_x : x > 0) (h_positive_y : y > 0) (h_positive_z : z > 0)
  (h_gcd : Nat.gcd x (Nat.gcd y z) = 1)
  (h_divisible : (x^2 + y^2 + z^2) % (x + y + z) = 0) :
  ¬(((x + y + z) - 2) % 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_not_divisible_by_three_l275_27558


namespace NUMINAMATH_CALUDE_mollys_current_age_l275_27500

/-- Given the ratio of Sandy's age to Molly's age and Sandy's age after 6 years, 
    calculate Molly's current age. -/
theorem mollys_current_age 
  (sandy_age : ℕ) 
  (molly_age : ℕ) 
  (h1 : sandy_age / molly_age = 4 / 3)  -- Ratio of ages
  (h2 : sandy_age + 6 = 30)             -- Sandy's age after 6 years
  : molly_age = 18 :=
by sorry

end NUMINAMATH_CALUDE_mollys_current_age_l275_27500


namespace NUMINAMATH_CALUDE_commission_increase_l275_27570

theorem commission_increase (total_sales : ℕ) (big_sale_commission : ℝ) (new_average : ℝ) :
  total_sales = 6 ∧ big_sale_commission = 1000 ∧ new_average = 250 →
  (new_average * total_sales - big_sale_commission) / (total_sales - 1) = 100 ∧
  new_average - (new_average * total_sales - big_sale_commission) / (total_sales - 1) = 150 := by
sorry

end NUMINAMATH_CALUDE_commission_increase_l275_27570


namespace NUMINAMATH_CALUDE_jesse_room_area_l275_27510

/-- The area of a rectangular room -/
def room_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of Jesse's room is 96 square feet -/
theorem jesse_room_area :
  room_area 12 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_jesse_room_area_l275_27510


namespace NUMINAMATH_CALUDE_largest_int_less_100_rem_5_div_8_l275_27506

theorem largest_int_less_100_rem_5_div_8 : 
  ∃ (n : ℕ), n < 100 ∧ n % 8 = 5 ∧ ∀ (m : ℕ), m < 100 ∧ m % 8 = 5 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_int_less_100_rem_5_div_8_l275_27506


namespace NUMINAMATH_CALUDE_unfolded_paper_has_four_crosses_l275_27579

/-- Represents a square piece of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents a fold on the paper -/
inductive Fold
  | LeftRight
  | TopBottom

/-- Represents a cross pattern of holes -/
structure Cross :=
  (center : ℝ × ℝ)
  (size : ℝ)

/-- Represents the state of the paper after folding and punching -/
structure FoldedPaper :=
  (paper : Paper)
  (folds : List Fold)
  (cross : Cross)

/-- Represents the unfolded paper with crosses -/
structure UnfoldedPaper :=
  (paper : Paper)
  (crosses : List Cross)

/-- Function to unfold the paper -/
def unfold (fp : FoldedPaper) : UnfoldedPaper :=
  sorry

/-- Main theorem: Unfolding results in four crosses, one in each quadrant -/
theorem unfolded_paper_has_four_crosses (fp : FoldedPaper) 
  (h1 : fp.folds = [Fold.LeftRight, Fold.TopBottom])
  (h2 : fp.cross.center.1 > fp.paper.side / 2 ∧ fp.cross.center.2 > fp.paper.side / 2) :
  let up := unfold fp
  (up.crosses.length = 4) ∧ 
  (∀ q : ℕ, q < 4 → ∃ c ∈ up.crosses, 
    (c.center.1 < up.paper.side / 2 ↔ q % 2 = 0) ∧
    (c.center.2 < up.paper.side / 2 ↔ q < 2)) :=
  sorry

end NUMINAMATH_CALUDE_unfolded_paper_has_four_crosses_l275_27579


namespace NUMINAMATH_CALUDE_blue_paint_cans_l275_27530

/-- Given a paint mixture with a blue to green ratio of 4:1 and a total of 40 cans,
    prove that 32 cans of blue paint are required. -/
theorem blue_paint_cans (total_cans : ℕ) (blue_ratio green_ratio : ℕ) 
  (h1 : total_cans = 40)
  (h2 : blue_ratio = 4)
  (h3 : green_ratio = 1) :
  (blue_ratio * total_cans) / (blue_ratio + green_ratio) = 32 := by
sorry


end NUMINAMATH_CALUDE_blue_paint_cans_l275_27530


namespace NUMINAMATH_CALUDE_sum_of_distinct_divisors_of_2000_l275_27526

def divisors_of_2000 : List ℕ := [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000]

def is_sum_of_distinct_divisors (n : ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.Nodup ∧ subset.Subset divisors_of_2000 ∧ subset.sum = n

theorem sum_of_distinct_divisors_of_2000 :
  ∀ n : ℕ, n > 0 ∧ n < 2000 → is_sum_of_distinct_divisors n :=
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_divisors_of_2000_l275_27526


namespace NUMINAMATH_CALUDE_jason_seashells_l275_27574

/-- Given that Jason initially had 49 seashells and gave away 13 seashells,
    prove that he now has 36 seashells. -/
theorem jason_seashells (initial : ℕ) (given_away : ℕ) (remaining : ℕ) 
    (h1 : initial = 49)
    (h2 : given_away = 13)
    (h3 : remaining = initial - given_away) :
  remaining = 36 := by
  sorry

end NUMINAMATH_CALUDE_jason_seashells_l275_27574


namespace NUMINAMATH_CALUDE_cherries_eaten_l275_27576

theorem cherries_eaten (initial : ℝ) (remaining : ℝ) (eaten : ℝ)
  (h1 : initial = 67.5)
  (h2 : remaining = 42.25)
  (h3 : eaten = initial - remaining) :
  eaten = 25.25 := by sorry

end NUMINAMATH_CALUDE_cherries_eaten_l275_27576


namespace NUMINAMATH_CALUDE_youtube_ad_time_l275_27511

/-- Calculates the time spent watching ads on Youtube --/
def time_watching_ads (videos_per_day : ℕ) (video_duration : ℕ) (total_time : ℕ) : ℕ :=
  total_time - (videos_per_day * video_duration)

/-- Theorem: The time spent watching ads is 3 minutes --/
theorem youtube_ad_time :
  time_watching_ads 2 7 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_youtube_ad_time_l275_27511


namespace NUMINAMATH_CALUDE_bruno_coconut_capacity_l275_27533

theorem bruno_coconut_capacity (total_coconuts : ℕ) (barbie_capacity : ℕ) (total_trips : ℕ) 
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : total_trips = 12) :
  (total_coconuts - barbie_capacity * total_trips) / total_trips = 8 := by
  sorry

end NUMINAMATH_CALUDE_bruno_coconut_capacity_l275_27533


namespace NUMINAMATH_CALUDE_part_one_part_two_l275_27543

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem part_one : ∀ x : ℝ, f x - f (x + 2) < 1 ↔ x > -1/2 := by sorry

-- Theorem for part II
theorem part_two : (∀ x : ℝ, x ∈ Set.Icc 1 2 → x - f (x + 1 - a) ≤ 1) → (a ≤ 1 ∨ a ≥ 3) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l275_27543


namespace NUMINAMATH_CALUDE_bank_depositors_bound_l275_27504

theorem bank_depositors_bound (total_deposits : ℝ) (probability_less_100 : ℝ) 
  (h_total : total_deposits = 20000)
  (h_prob : probability_less_100 = 0.8) :
  ∃ n : ℕ, n ≤ 1000 ∧ (total_deposits / n : ℝ) ≤ 100 / (1 - probability_less_100) := by
  sorry

end NUMINAMATH_CALUDE_bank_depositors_bound_l275_27504


namespace NUMINAMATH_CALUDE_triangle_side_length_l275_27552

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 → a = Real.sqrt 3 → b = 1 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  c ^ 2 = a ^ 2 + b ^ 2 →
  c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l275_27552


namespace NUMINAMATH_CALUDE_black_balls_count_l275_27541

theorem black_balls_count (total : ℕ) (red : ℕ) (prob_white : ℚ) 
  (h_total : total = 100)
  (h_red : red = 30)
  (h_prob_white : prob_white = 47/100) :
  total - red - (total * prob_white).num = 23 := by
sorry

end NUMINAMATH_CALUDE_black_balls_count_l275_27541


namespace NUMINAMATH_CALUDE_circle_c_equation_l275_27539

/-- A circle C with center on y = x^2, passing through origin, and intercepting 8 units on y-axis -/
structure CircleC where
  a : ℝ
  center : ℝ × ℝ
  center_on_parabola : center.2 = center.1^2
  passes_through_origin : (0 - center.1)^2 + (0 - center.2)^2 = (4 + center.1)^2
  intercepts_8_on_yaxis : (0 - center.1)^2 + (4 - center.2)^2 = (4 + center.1)^2

/-- The equation of circle C is either (x-2)^2 + (y-4)^2 = 20 or (x+2)^2 + (y-4)^2 = 20 -/
theorem circle_c_equation (c : CircleC) :
  ((λ (x y : ℝ) => (x - 2)^2 + (y - 4)^2 = 20) = λ (x y : ℝ) => (x - c.center.1)^2 + (y - c.center.2)^2 = (4 + c.a)^2) ∨
  ((λ (x y : ℝ) => (x + 2)^2 + (y - 4)^2 = 20) = λ (x y : ℝ) => (x - c.center.1)^2 + (y - c.center.2)^2 = (4 + c.a)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_c_equation_l275_27539


namespace NUMINAMATH_CALUDE_floor_sum_example_l275_27585

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l275_27585


namespace NUMINAMATH_CALUDE_twelve_switches_four_connections_l275_27582

/-- The number of connections in a network of switches where each switch connects to a fixed number of others. -/
def connections (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a network of 12 switches, where each switch is directly connected to exactly 4 other switches, the total number of connections is 24. -/
theorem twelve_switches_four_connections :
  connections 12 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_twelve_switches_four_connections_l275_27582


namespace NUMINAMATH_CALUDE_expression_evaluation_l275_27531

theorem expression_evaluation (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h : y - z / x ≠ 0) : 
  (x - z / y) / (y - z / x) = x / y := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l275_27531


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_twenty_thirds_l275_27550

theorem sum_abcd_equals_negative_twenty_thirds 
  (y a b c d : ℚ) 
  (h1 : y = a + 2)
  (h2 : y = b + 4)
  (h3 : y = c + 6)
  (h4 : y = d + 8)
  (h5 : y = a + b + c + d + 10) :
  a + b + c + d = -20 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_twenty_thirds_l275_27550


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l275_27566

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + 2

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | 2/a ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) (h : a < 0) :
  {x : ℝ | f a x ≥ 0} = solution_set a :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l275_27566


namespace NUMINAMATH_CALUDE_bus_speed_proof_l275_27508

/-- The speed of Bus A in miles per hour -/
def speed_A : ℝ := 45

/-- The speed of Bus B in miles per hour -/
def speed_B : ℝ := speed_A - 15

/-- The initial distance between Bus A and Bus B in miles -/
def initial_distance : ℝ := 150

/-- The time it takes for Bus A to overtake Bus B when driving in the same direction, in hours -/
def overtake_time : ℝ := 10

/-- The time it would take for the buses to meet if driving towards each other, in hours -/
def meet_time : ℝ := 2

theorem bus_speed_proof :
  (speed_A - speed_B) * overtake_time = initial_distance ∧
  (speed_A + speed_B) * meet_time = initial_distance ∧
  speed_A = 45 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_proof_l275_27508


namespace NUMINAMATH_CALUDE_fuel_mixture_proof_l275_27567

def tank_capacity : ℝ := 200
def ethanol_percentage_A : ℝ := 0.12
def ethanol_percentage_B : ℝ := 0.16
def total_ethanol : ℝ := 30

theorem fuel_mixture_proof (x : ℝ) 
  (hx : x ≥ 0 ∧ x ≤ 100) 
  (h_ethanol : ethanol_percentage_A * x + ethanol_percentage_B * (tank_capacity - x) = total_ethanol) :
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_fuel_mixture_proof_l275_27567


namespace NUMINAMATH_CALUDE_angle_A_value_side_a_range_l275_27564

/-- Represents an acute triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

theorem angle_A_value (t : AcuteTriangle) :
  Real.cos (2 * t.A) - Real.cos (2 * t.B) + 2 * Real.cos (π/6 - t.B) * Real.cos (π/6 + t.B) = 0 →
  t.A = π/3 := by sorry

theorem side_a_range (t : AcuteTriangle) :
  t.b = Real.sqrt 3 → t.b ≤ t.a → t.A = π/3 →
  t.a ≥ Real.sqrt 3 ∧ t.a < 3 := by sorry

end NUMINAMATH_CALUDE_angle_A_value_side_a_range_l275_27564


namespace NUMINAMATH_CALUDE_normal_dist_prob_ge_six_l275_27561

/-- A random variable following a normal distribution -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The probability that a random variable falls within one standard deviation of the mean -/
def prob_within_one_std (X : NormalDistribution) : ℝ := 0.6826

/-- The probability that a random variable is greater than or equal to a given value -/
noncomputable def prob_ge (X : NormalDistribution) (x : ℝ) : ℝ :=
  1 - (prob_within_one_std X) / 2

/-- Theorem: For a normal distribution N(5, 1), P(X ≥ 6) = 0.1587 -/
theorem normal_dist_prob_ge_six (X : NormalDistribution) 
  (h1 : X.μ = 5) (h2 : X.σ = 1) : 
  prob_ge X 6 = 0.1587 := by
  sorry


end NUMINAMATH_CALUDE_normal_dist_prob_ge_six_l275_27561


namespace NUMINAMATH_CALUDE_is_circle_center_l275_27519

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, 2)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l275_27519


namespace NUMINAMATH_CALUDE_square_perimeter_division_l275_27521

/-- Represents a division of a square's perimeter into two groups of segments -/
structure SquarePerimeterDivision where
  side_length : ℝ
  group1_count : ℕ
  group2_count : ℕ
  group1_segment_length : ℝ
  group2_segment_length : ℝ

/-- Checks if the given division is valid for the square's perimeter -/
def is_valid_division (d : SquarePerimeterDivision) : Prop :=
  d.group1_count * d.group1_segment_length + d.group2_count * d.group2_segment_length = 4 * d.side_length

/-- The specific division of a square with side length 20 cm into 3 and 4 segments -/
def specific_division : SquarePerimeterDivision :=
  { side_length := 20
  , group1_count := 3
  , group2_count := 4
  , group1_segment_length := 20
  , group2_segment_length := 5 }

theorem square_perimeter_division :
  is_valid_division specific_division ∧
  specific_division.group1_segment_length = 20 ∧
  specific_division.group2_segment_length = 5 := by
  sorry

#check square_perimeter_division

end NUMINAMATH_CALUDE_square_perimeter_division_l275_27521


namespace NUMINAMATH_CALUDE_nicholas_crackers_l275_27571

theorem nicholas_crackers (marcus_crackers : ℕ) (mona_crackers : ℕ) (nicholas_crackers : ℕ) : 
  marcus_crackers = 27 →
  marcus_crackers = 3 * mona_crackers →
  nicholas_crackers = mona_crackers + 6 →
  nicholas_crackers = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_nicholas_crackers_l275_27571


namespace NUMINAMATH_CALUDE_second_number_is_13_l275_27592

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ        -- Total number of items
  sampleSize : ℕ   -- Number of items to be sampled
  first : ℕ        -- First number drawn

/-- Calculates the nth number in a systematic sample -/
def nthNumber (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first + (s.total / s.sampleSize) * (n - 1)

/-- Theorem stating that the second number drawn is 13 -/
theorem second_number_is_13 (s : SystematicSample) 
  (h1 : s.total = 500) 
  (h2 : s.sampleSize = 50) 
  (h3 : s.first = 3) : 
  nthNumber s 2 = 13 := by
  sorry

#check second_number_is_13

end NUMINAMATH_CALUDE_second_number_is_13_l275_27592


namespace NUMINAMATH_CALUDE_hash_two_three_four_l275_27584

-- Define the # operation
def hash (r s t : ℝ) : ℝ := r + s + t + r*s + r*t + s*t + r*s*t

-- State the theorem
theorem hash_two_three_four : hash 2 3 4 = 59 := by
  sorry

end NUMINAMATH_CALUDE_hash_two_three_four_l275_27584


namespace NUMINAMATH_CALUDE_ratio_of_segments_l275_27565

-- Define the right triangle
def right_triangle (a b c r s : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 ∧
  c^2 = a^2 + b^2 ∧
  c = r + s ∧
  a^2 = r * c ∧
  b^2 = s * c

-- Theorem statement
theorem ratio_of_segments (a b c r s : ℝ) :
  right_triangle a b c r s →
  a / b = 2 / 5 →
  r / s = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l275_27565


namespace NUMINAMATH_CALUDE_sqrt_450_equals_15_l275_27535

theorem sqrt_450_equals_15 : Real.sqrt 450 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_equals_15_l275_27535


namespace NUMINAMATH_CALUDE_surface_area_increase_l275_27593

/-- Represents a rectangular solid with given dimensions -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the original solid -/
def originalSolid : RectangularSolid :=
  { length := 4, width := 3, height := 2 }

/-- Represents the size of the removed cube -/
def cubeSize : ℝ := 1

/-- Theorem stating that removing a 1-foot cube from the center of the original solid
    increases its surface area by 6 square feet -/
theorem surface_area_increase :
  surfaceArea originalSolid + 6 = surfaceArea originalSolid + 6 * cubeSize^2 := by
  sorry

#check surface_area_increase

end NUMINAMATH_CALUDE_surface_area_increase_l275_27593


namespace NUMINAMATH_CALUDE_positive_solution_range_l275_27527

theorem positive_solution_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ a / (x + 3) = 1 / 2 ∧ x = a) → a > 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_range_l275_27527


namespace NUMINAMATH_CALUDE_horners_rule_polynomial_l275_27515

theorem horners_rule_polynomial (x : ℝ) : 
  x^3 + 2*x^2 + x - 1 = ((x + 2)*x + 1)*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_horners_rule_polynomial_l275_27515


namespace NUMINAMATH_CALUDE_platform_length_l275_27536

/-- Given a train that passes a pole and a platform, prove the length of the platform. -/
theorem platform_length
  (train_length : ℝ)
  (pole_time : ℝ)
  (platform_time : ℝ)
  (h1 : train_length = 120)
  (h2 : pole_time = 11)
  (h3 : platform_time = 22) :
  (train_length * platform_time / pole_time) - train_length = 120 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l275_27536


namespace NUMINAMATH_CALUDE_sum_of_digits_multiple_of_990_l275_27540

/-- Given a six-digit number 123abc that is a multiple of 990, 
    prove that the sum of its hundreds, tens, and units digits (a + b + c) is 12 -/
theorem sum_of_digits_multiple_of_990 (a b c : ℕ) : 
  (0 < a) → (a < 10) →
  (0 ≤ b) → (b < 10) →
  (0 ≤ c) → (c < 10) →
  (123000 + 100 * a + 10 * b + c) % 990 = 0 →
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_multiple_of_990_l275_27540


namespace NUMINAMATH_CALUDE_choose_captains_l275_27513

theorem choose_captains (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_captains_l275_27513


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l275_27586

theorem quadratic_is_perfect_square (x : ℝ) : 
  ∃ (a : ℝ), x^2 - 20*x + 100 = (x + a)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l275_27586


namespace NUMINAMATH_CALUDE_flour_amount_second_combination_l275_27537

/-- The cost per pound of sugar and flour -/
def cost_per_pound : ℝ := 0.45

/-- The total cost of both combinations -/
def total_cost : ℝ := 26

/-- The amount of sugar in the first combination -/
def sugar_amount_1 : ℝ := 40

/-- The amount of flour in the first combination -/
def flour_amount_1 : ℝ := 16

/-- The amount of sugar in the second combination -/
def sugar_amount_2 : ℝ := 30

/-- The amount of flour in the second combination -/
def flour_amount_2 : ℝ := 28

theorem flour_amount_second_combination :
  sugar_amount_1 * cost_per_pound + flour_amount_1 * cost_per_pound = total_cost ∧
  sugar_amount_2 * cost_per_pound + flour_amount_2 * cost_per_pound = total_cost :=
by sorry

end NUMINAMATH_CALUDE_flour_amount_second_combination_l275_27537


namespace NUMINAMATH_CALUDE_peggy_record_count_l275_27559

/-- The number of records Peggy has -/
def num_records : ℕ := 200

/-- The price Sammy offers for each record -/
def sammy_price : ℚ := 4

/-- The price Bryan offers for each record he's interested in -/
def bryan_interested_price : ℚ := 6

/-- The price Bryan offers for each record he's not interested in -/
def bryan_not_interested_price : ℚ := 1

/-- The difference in profit between Sammy's and Bryan's deals -/
def profit_difference : ℚ := 100

theorem peggy_record_count :
  (sammy_price * num_records) - 
  ((bryan_interested_price * (num_records / 2)) + (bryan_not_interested_price * (num_records / 2))) = 
  profit_difference :=
sorry

end NUMINAMATH_CALUDE_peggy_record_count_l275_27559


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l275_27581

/-- Given a geometric sequence with positive terms and common ratio q,
    if 3a₁, (1/2)a₃, 2a₂ form an arithmetic sequence, then q = 3 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence with ratio q
  q > 0 →  -- q is positive
  2 * ((1/2) * a 3) = 3 * a 1 + 2 * a 2 →  -- arithmetic sequence condition
  q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l275_27581


namespace NUMINAMATH_CALUDE_acid_dilution_l275_27545

/-- Given m ounces of m% acid solution, adding x ounces of water yields (m-10)% solution -/
theorem acid_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m^2 / 100 = (m - 10) / 100 * (m + x)) → x = 10 * m / (m - 10) := by sorry

end NUMINAMATH_CALUDE_acid_dilution_l275_27545


namespace NUMINAMATH_CALUDE_even_function_implies_m_eq_neg_one_l275_27529

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = (m - 1)x² - (m² - 1)x + m + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 - (m^2 - 1) * x + m + 2

theorem even_function_implies_m_eq_neg_one :
  ∀ m : ℝ, EvenFunction (f m) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_eq_neg_one_l275_27529


namespace NUMINAMATH_CALUDE_find_y_l275_27553

theorem find_y : ∃ y : ℚ, 3 + 1 / (2 - y) = 2 * (1 / (2 - y)) → y = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l275_27553


namespace NUMINAMATH_CALUDE_f_pow_ten_l275_27524

/-- f(n) is the number of ones that occur in the decimal representations of all the numbers from 1 to n -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem: For any natural number k, f(10^k) = k * 10^(k-1) + 1 -/
theorem f_pow_ten (k : ℕ) : f (10^k) = k * 10^(k-1) + 1 := by sorry

end NUMINAMATH_CALUDE_f_pow_ten_l275_27524


namespace NUMINAMATH_CALUDE_triangle_perimeter_with_inscribed_circles_triangle_perimeter_l275_27532

/-- Represents an equilateral triangle with inscribed circles -/
structure TriangleWithCircles where
  /-- The side length of the equilateral triangle -/
  side_length : ℝ
  /-- The radius of each inscribed circle -/
  circle_radius : ℝ
  /-- Assumption that the circles touch two sides of the triangle and each other -/
  circles_touch_sides_and_each_other : True

/-- Theorem stating the perimeter of the triangle given the inscribed circles -/
theorem triangle_perimeter_with_inscribed_circles
  (t : TriangleWithCircles)
  (h : t.circle_radius = 2) :
  t.side_length = 2 * Real.sqrt 3 + 4 :=
sorry

/-- Corollary calculating the perimeter of the triangle -/
theorem triangle_perimeter
  (t : TriangleWithCircles)
  (h : t.circle_radius = 2) :
  3 * t.side_length = 6 * Real.sqrt 3 + 12 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_with_inscribed_circles_triangle_perimeter_l275_27532


namespace NUMINAMATH_CALUDE_ln_is_elite_elite_bound_exists_nonincreasing_elite_sufficient_condition_elite_l275_27512

/-- Definition of an "elite" function -/
def IsElite (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ + x₂) < f x₁ + f x₂

/-- Statement 1: ln(1+x) is an "elite" function -/
theorem ln_is_elite : IsElite (fun x => Real.log (1 + x)) := sorry

/-- Statement 2: For "elite" functions, f(n) < nf(1) for n ≥ 2 -/
theorem elite_bound (f : ℝ → ℝ) (hf : IsElite f) :
  ∀ n : ℕ, n ≥ 2 → f n < n * f 1 := sorry

/-- Statement 3: Existence of an "elite" function that is not strictly increasing -/
theorem exists_nonincreasing_elite :
  ∃ f : ℝ → ℝ, IsElite f ∧ ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ (f x₁ - f x₂) / (x₁ - x₂) ≤ 0 := sorry

/-- Statement 4: A sufficient condition for a function to be "elite" -/
theorem sufficient_condition_elite (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ > x₂ → x₂ > 0 → x₂ * f x₁ < x₁ * f x₂) : 
  IsElite f := sorry

end NUMINAMATH_CALUDE_ln_is_elite_elite_bound_exists_nonincreasing_elite_sufficient_condition_elite_l275_27512


namespace NUMINAMATH_CALUDE_cube_root_three_equation_l275_27577

theorem cube_root_three_equation (s : ℝ) : s = 1 / (2 - (3 : ℝ)^(1/3)) → s = 2 + (3 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_three_equation_l275_27577


namespace NUMINAMATH_CALUDE_distance_swam_against_current_l275_27596

/-- Calculates the distance swam against a current given swimming speed, current speed, and time taken. -/
def distance_against_current (swimming_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  (swimming_speed - current_speed) * time

theorem distance_swam_against_current 
  (swimming_speed : ℝ) (current_speed : ℝ) (time : ℝ)
  (h1 : swimming_speed = 4)
  (h2 : current_speed = 2)
  (h3 : time = 5) :
  distance_against_current swimming_speed current_speed time = 10 := by
sorry

end NUMINAMATH_CALUDE_distance_swam_against_current_l275_27596


namespace NUMINAMATH_CALUDE_tan_beta_rationality_l275_27502

theorem tan_beta_rationality (p q : ℤ) (α β : ℝ) (h1 : q ≠ 0) (h2 : Real.tan α = p / q) (h3 : Real.tan (2 * β) = Real.tan (3 * α)) :
  (∃ (r s : ℤ), s ≠ 0 ∧ Real.tan β = r / s) ↔ ∃ (n : ℤ), p^2 + q^2 = n^2 :=
sorry

end NUMINAMATH_CALUDE_tan_beta_rationality_l275_27502


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l275_27573

def f (x : ℝ) : ℝ := x^2 + 22*x + 105

theorem quadratic_function_properties :
  (∀ x, f x = x^2 + 22*x + 105) ∧
  (∃ a b : ℤ, ∀ x, f x = x^2 + a*x + b) ∧
  (∃ r₁ r₂ r₃ r₄ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
    f (f r₁) = 0 ∧ f (f r₂) = 0 ∧ f (f r₃) = 0 ∧ f (f r₄) = 0) ∧
  (∃ d : ℝ, d ≠ 0 ∧ ∃ r₁ r₂ r₃ r₄ : ℝ,
    f (f r₁) = 0 ∧ f (f r₂) = 0 ∧ f (f r₃) = 0 ∧ f (f r₄) = 0 ∧
    r₂ = r₁ + d ∧ r₃ = r₂ + d ∧ r₄ = r₃ + d) ∧
  (∀ g : ℝ → ℝ, (∃ a b : ℤ, ∀ x, g x = x^2 + a*x + b) →
    (∃ r₁ r₂ r₃ r₄ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
      g (g r₁) = 0 ∧ g (g r₂) = 0 ∧ g (g r₃) = 0 ∧ g (g r₄) = 0) →
    (∃ d : ℝ, d ≠ 0 ∧ ∃ r₁ r₂ r₃ r₄ : ℝ,
      g (g r₁) = 0 ∧ g (g r₂) = 0 ∧ g (g r₃) = 0 ∧ g (g r₄) = 0 ∧
      r₂ = r₁ + d ∧ r₃ = r₂ + d ∧ r₄ = r₃ + d) →
    (1 + a + b ≥ 1 + 22 + 105)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l275_27573


namespace NUMINAMATH_CALUDE_volleyball_tournament_wins_l275_27517

theorem volleyball_tournament_wins (n m : ℕ) : 
  (∃ (x : ℕ), 0 < x ∧ x < 73 ∧ x * n + (73 - x) * m = 36 * 73) →
  n = m :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_wins_l275_27517


namespace NUMINAMATH_CALUDE_average_fee_is_4_6_l275_27503

/-- Represents the delivery statistics for a delivery person in December -/
structure DeliveryStats where
  short_distance_percent : ℝ  -- Percentage of deliveries ≤ 3 km
  long_distance_percent : ℝ   -- Percentage of deliveries > 3 km
  short_distance_fee : ℝ      -- Fee for deliveries ≤ 3 km
  long_distance_fee : ℝ       -- Fee for deliveries > 3 km

/-- Calculates the average delivery fee per order -/
def average_delivery_fee (stats : DeliveryStats) : ℝ :=
  stats.short_distance_percent * stats.short_distance_fee +
  stats.long_distance_percent * stats.long_distance_fee

/-- Theorem stating that the average delivery fee is 4.6 yuan for the given statistics -/
theorem average_fee_is_4_6 (stats : DeliveryStats) 
  (h1 : stats.short_distance_percent = 0.7)
  (h2 : stats.long_distance_percent = 0.3)
  (h3 : stats.short_distance_fee = 4)
  (h4 : stats.long_distance_fee = 6) :
  average_delivery_fee stats = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_average_fee_is_4_6_l275_27503


namespace NUMINAMATH_CALUDE_new_years_party_assignments_l275_27525

/-- The number of ways to assign teachers to classes -/
def assignTeachers (totalTeachers : ℕ) (numClasses : ℕ) (maxPerClass : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of assignments for the given conditions -/
theorem new_years_party_assignments :
  assignTeachers 6 2 4 = 50 := by sorry

end NUMINAMATH_CALUDE_new_years_party_assignments_l275_27525


namespace NUMINAMATH_CALUDE_function_form_l275_27563

/-- Given a function g: ℝ → ℝ satisfying certain conditions, prove it has a specific form. -/
theorem function_form (g : ℝ → ℝ) 
  (h1 : g 2 = 2)
  (h2 : ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y) :
  ∀ x : ℝ, g x = (5^x - 3^x) / 8 := by
  sorry

end NUMINAMATH_CALUDE_function_form_l275_27563


namespace NUMINAMATH_CALUDE_smallest_distance_complex_circles_l275_27538

/-- The smallest possible distance between two complex numbers on given circles -/
theorem smallest_distance_complex_circles :
  ∀ (z w : ℂ),
  Complex.abs (z - (2 + 4*Complex.I)) = 2 →
  Complex.abs (w - (8 + 6*Complex.I)) = 4 →
  ∀ (d : ℝ),
  d = Complex.abs (z - w) →
  d ≥ Real.sqrt 10 - 6 ∧
  ∃ (z₀ w₀ : ℂ),
    Complex.abs (z₀ - (2 + 4*Complex.I)) = 2 ∧
    Complex.abs (w₀ - (8 + 6*Complex.I)) = 4 ∧
    Complex.abs (z₀ - w₀) = Real.sqrt 10 - 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_complex_circles_l275_27538


namespace NUMINAMATH_CALUDE_decimal_equivalences_l275_27507

-- Define the decimal number
def decimal_number : ℚ := 209 / 100

-- Theorem to prove the equivalence
theorem decimal_equivalences :
  -- Percentage equivalence
  (decimal_number * 100 : ℚ) = 209 ∧
  -- Simplified fraction equivalence
  decimal_number = 209 / 100 ∧
  -- Mixed number equivalence
  ∃ (whole : ℕ) (numerator : ℕ) (denominator : ℕ),
    whole = 2 ∧
    numerator = 9 ∧
    denominator = 100 ∧
    decimal_number = whole + (numerator : ℚ) / denominator :=
by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalences_l275_27507


namespace NUMINAMATH_CALUDE_brenda_baking_days_l275_27554

/-- Represents the number of cakes Brenda bakes per day -/
def cakes_per_day : ℕ := 20

/-- Represents the number of cakes Brenda has left after selling -/
def cakes_left : ℕ := 90

/-- Theorem: The number of days Brenda baked cakes is 9 -/
theorem brenda_baking_days : 
  ∃ (days : ℕ), 
    (cakes_per_day * days) / 2 = cakes_left ∧ 
    days = 9 := by
  sorry

end NUMINAMATH_CALUDE_brenda_baking_days_l275_27554


namespace NUMINAMATH_CALUDE_negation_p_sufficient_not_necessary_for_q_l275_27588

theorem negation_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x^2 + x - 6 < 0) ∧
  (∃ x : ℝ, x^2 + x - 6 < 0 ∧ (x < -1 ∨ x > 1)) := by sorry

end NUMINAMATH_CALUDE_negation_p_sufficient_not_necessary_for_q_l275_27588


namespace NUMINAMATH_CALUDE_parallelogram_division_slope_l275_27568

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Checks if a line with given slope passes through the origin and divides the parallelogram into two congruent polygons -/
def dividesParallelogramEqually (p : Parallelogram) (slope : ℚ) : Prop :=
  ∃ (a : ℝ),
    (p.v1.y + a) / p.v1.x = slope ∧
    (p.v3.y - a) / p.v3.x = slope ∧
    0 < a ∧ a < p.v2.y - p.v1.y

/-- The main theorem stating the slope of the line dividing the parallelogram equally -/
theorem parallelogram_division_slope :
  let p : Parallelogram := {
    v1 := { x := 12, y := 60 },
    v2 := { x := 12, y := 152 },
    v3 := { x := 32, y := 204 },
    v4 := { x := 32, y := 112 }
  }
  dividesParallelogramEqually p 16 := by sorry

end NUMINAMATH_CALUDE_parallelogram_division_slope_l275_27568


namespace NUMINAMATH_CALUDE_reciprocal_problem_l275_27534

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 4) : 200 * (1 / x) = 400 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l275_27534


namespace NUMINAMATH_CALUDE_erik_money_left_l275_27599

/-- The amount of money Erik started with -/
def initial_money : ℕ := 86

/-- The number of loaves of bread Erik bought -/
def bread_quantity : ℕ := 3

/-- The cost of each loaf of bread -/
def bread_cost : ℕ := 3

/-- The number of cartons of orange juice Erik bought -/
def juice_quantity : ℕ := 3

/-- The cost of each carton of orange juice -/
def juice_cost : ℕ := 6

/-- The theorem stating how much money Erik has left -/
theorem erik_money_left : 
  initial_money - (bread_quantity * bread_cost + juice_quantity * juice_cost) = 59 := by
  sorry

end NUMINAMATH_CALUDE_erik_money_left_l275_27599


namespace NUMINAMATH_CALUDE_cone_volume_and_surface_area_l275_27520

/-- Given a cone with slant height 17 cm and height 15 cm, prove its volume and lateral surface area -/
theorem cone_volume_and_surface_area :
  let slant_height : ℝ := 17
  let height : ℝ := 15
  let radius : ℝ := Real.sqrt (slant_height ^ 2 - height ^ 2)
  let volume : ℝ := (1 / 3) * π * radius ^ 2 * height
  let lateral_surface_area : ℝ := π * radius * slant_height
  volume = 320 * π ∧ lateral_surface_area = 136 * π := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_and_surface_area_l275_27520


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l275_27505

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 + 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l275_27505


namespace NUMINAMATH_CALUDE_second_term_is_twelve_l275_27575

/-- A geometric sequence with a sum formula -/
structure GeometricSequence where
  a : ℝ  -- The common ratio multiplier
  sequence : ℕ → ℝ
  sum : ℕ → ℝ
  sum_formula : ∀ n : ℕ, sum n = a * 3^n - 2
  is_geometric : ∀ n : ℕ, sequence (n + 2) * sequence n = (sequence (n + 1))^2

/-- The second term of the geometric sequence is 12 -/
theorem second_term_is_twelve (seq : GeometricSequence) : seq.sequence 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_term_is_twelve_l275_27575


namespace NUMINAMATH_CALUDE_candy_problem_l275_27555

/-- Given a number of students and pieces per student, calculates the total number of candy pieces. -/
def total_candy (num_students : ℕ) (pieces_per_student : ℕ) : ℕ :=
  num_students * pieces_per_student

/-- Proves that the total number of candy pieces is 344, given 43 students and 8 pieces per student. -/
theorem candy_problem :
  total_candy 43 8 = 344 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l275_27555


namespace NUMINAMATH_CALUDE_married_men_fraction_l275_27556

theorem married_men_fraction (total_women : ℕ) (h_total_women : total_women > 0) :
  let single_women := (3 : ℚ) / 7 * total_women
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  married_men / total_people = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l275_27556


namespace NUMINAMATH_CALUDE_bill_amount_correct_l275_27549

/-- The amount of the bill in dollars -/
def bill_amount : ℝ := 26

/-- The percentage a bad tipper tips -/
def bad_tip_percent : ℝ := 0.05

/-- The percentage a good tipper tips -/
def good_tip_percent : ℝ := 0.20

/-- The difference between a good tip and a bad tip in dollars -/
def tip_difference : ℝ := 3.90

theorem bill_amount_correct : 
  (good_tip_percent - bad_tip_percent) * bill_amount = tip_difference := by
  sorry

end NUMINAMATH_CALUDE_bill_amount_correct_l275_27549


namespace NUMINAMATH_CALUDE_bert_sale_earnings_l275_27547

/-- Calculates Bert's earnings from a sale given the selling price, markup, and tax rate. -/
def bertEarnings (sellingPrice markup taxRate : ℚ) : ℚ :=
  let purchasePrice := sellingPrice - markup
  let tax := taxRate * sellingPrice
  sellingPrice - tax - purchasePrice

/-- Theorem: Given a selling price of $90, a markup of $10, and a tax rate of 10%, Bert's earnings are $1. -/
theorem bert_sale_earnings :
  bertEarnings 90 10 (1/10) = 1 := by
  sorry

#eval bertEarnings 90 10 (1/10)

end NUMINAMATH_CALUDE_bert_sale_earnings_l275_27547


namespace NUMINAMATH_CALUDE_cube_root_of_three_times_two_to_seven_l275_27572

theorem cube_root_of_three_times_two_to_seven (x : ℝ) :
  x = Real.rpow 2 7 + Real.rpow 2 7 + Real.rpow 2 7 →
  Real.rpow x (1/3) = 4 * Real.rpow 6 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_three_times_two_to_seven_l275_27572


namespace NUMINAMATH_CALUDE_unique_value_of_expression_l275_27597

theorem unique_value_of_expression (x y : ℝ) 
  (h : x * y - 3 * x / (y^2) - 3 * y / (x^2) = 7) : 
  (x - 2) * (y - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_value_of_expression_l275_27597


namespace NUMINAMATH_CALUDE_factor_expression_l275_27542

theorem factor_expression (y : ℝ) : 5 * y * (y - 2) + 11 * (y - 2) = (y - 2) * (5 * y + 11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l275_27542


namespace NUMINAMATH_CALUDE_simple_annual_interest_rate_l275_27516

/-- Simple annual interest rate calculation -/
theorem simple_annual_interest_rate 
  (monthly_interest : ℝ) 
  (principal : ℝ) 
  (h1 : monthly_interest = 216)
  (h2 : principal = 28800) :
  (monthly_interest * 12) / principal = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_simple_annual_interest_rate_l275_27516


namespace NUMINAMATH_CALUDE_sams_book_count_l275_27501

theorem sams_book_count :
  let used_adventure_books : ℝ := 13.0
  let used_mystery_books : ℝ := 17.0
  let new_crime_books : ℝ := 15.0
  let total_books := used_adventure_books + used_mystery_books + new_crime_books
  total_books = 45.0 := by sorry

end NUMINAMATH_CALUDE_sams_book_count_l275_27501


namespace NUMINAMATH_CALUDE_system_solution_l275_27509

def solution_set : Set (ℝ × ℝ) :=
  {(-3/Real.sqrt 5, 1/Real.sqrt 5), (-3/Real.sqrt 5, -1/Real.sqrt 5),
   (3/Real.sqrt 5, -1/Real.sqrt 5), (3/Real.sqrt 5, 1/Real.sqrt 5)}

theorem system_solution :
  ∀ x y : ℝ, (x^2 + y^2 ≤ 2 ∧
    81*x^4 - 18*x^2*y^2 + y^4 - 360*x^2 - 40*y^2 + 400 = 0) ↔
  (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l275_27509


namespace NUMINAMATH_CALUDE_first_division_percentage_l275_27594

theorem first_division_percentage (total_students : ℕ) 
  (second_division_percent : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  second_division_percent = 54 / 100 →
  just_passed = 54 →
  (just_passed : ℚ) / total_students + second_division_percent + 28 / 100 = 1 :=
by sorry

end NUMINAMATH_CALUDE_first_division_percentage_l275_27594


namespace NUMINAMATH_CALUDE_omelet_problem_l275_27591

/-- The number of people that can be served omelets given the conditions -/
def number_of_people (eggs_per_dozen : ℕ) (eggs_per_omelet : ℕ) (omelets_per_person : ℕ) : ℕ :=
  let total_eggs := 3 * eggs_per_dozen
  let total_omelets := total_eggs / eggs_per_omelet
  total_omelets / omelets_per_person

/-- Theorem stating that under the given conditions, the number of people is 3 -/
theorem omelet_problem : number_of_people 12 4 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_omelet_problem_l275_27591


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l275_27522

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ), 
    (x₁ * (5 * x₁ - 9) = -4) ∧
    (x₂ * (5 * x₂ - 9) = -4) ∧
    (x₁ = (9 + Real.sqrt 1) / 10) ∧
    (x₂ = (9 - Real.sqrt 1) / 10) ∧
    (9 + 1 + 10 = 20) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l275_27522


namespace NUMINAMATH_CALUDE_total_flowers_in_gardens_l275_27589

/-- Given 10 gardens, each with 544 pots, and 32 flowers per pot,
    prove that the total number of flowers in all gardens is 174,080. -/
theorem total_flowers_in_gardens : 
  let num_gardens : ℕ := 10
  let pots_per_garden : ℕ := 544
  let flowers_per_pot : ℕ := 32
  num_gardens * pots_per_garden * flowers_per_pot = 174080 :=
by sorry

end NUMINAMATH_CALUDE_total_flowers_in_gardens_l275_27589


namespace NUMINAMATH_CALUDE_exists_real_a_sqrt3_minus_a_real_l275_27544

theorem exists_real_a_sqrt3_minus_a_real : ∃ a : ℝ, ∃ b : ℝ, b = Real.sqrt 3 - a := by
  sorry

end NUMINAMATH_CALUDE_exists_real_a_sqrt3_minus_a_real_l275_27544


namespace NUMINAMATH_CALUDE_frustum_sphere_equal_volume_l275_27528

/-- Given a frustum of a cone with small radius 2 inches, large radius 3 inches,
    and height 5 inches, the radius of a sphere with the same volume is ∛(95/4) inches. -/
theorem frustum_sphere_equal_volume :
  let r₁ : ℝ := 2  -- small radius of frustum
  let r₂ : ℝ := 3  -- large radius of frustum
  let h : ℝ := 5   -- height of frustum
  let V_frustum := (1/3) * π * h * (r₁^2 + r₁*r₂ + r₂^2)
  let r_sphere := (95/4)^(1/3 : ℝ)
  let V_sphere := (4/3) * π * r_sphere^3
  V_frustum = V_sphere := by sorry

end NUMINAMATH_CALUDE_frustum_sphere_equal_volume_l275_27528


namespace NUMINAMATH_CALUDE_tennis_tournament_l275_27514

theorem tennis_tournament (n : ℕ) : 
  (∃ (total_matches : ℕ) (women_wins men_wins : ℕ),
    -- Total number of players
    (n + (2*n + 1) = 3*n + 1) ∧
    -- Total matches calculation
    (total_matches = (3*n + 1) * (3*n) / 2 + 2*n) ∧
    -- Ratio of wins
    (3 * men_wins = 2 * women_wins) ∧
    -- Total wins equal total matches
    (women_wins + men_wins = total_matches) ∧
    -- n is a positive integer
    (n > 0)) →
  n = 2 :=
by sorry

end NUMINAMATH_CALUDE_tennis_tournament_l275_27514


namespace NUMINAMATH_CALUDE_complex_equation_solution_l275_27598

def complex_i : ℂ := Complex.I

theorem complex_equation_solution (a : ℝ) :
  (2 : ℂ) / (a + complex_i) = 1 - complex_i → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l275_27598


namespace NUMINAMATH_CALUDE_problem_statement_l275_27590

theorem problem_statement (a b : ℝ) : 
  |a - 3| + (b + 4)^2 = 0 → (a + b)^2003 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l275_27590


namespace NUMINAMATH_CALUDE_emma_bank_account_l275_27548

/-- Calculates the final amount in a bank account after a withdrawal and deposit -/
def final_amount (initial_savings withdrawal : ℕ) : ℕ :=
  let remaining := initial_savings - withdrawal
  let deposit := 2 * withdrawal
  remaining + deposit

/-- Proves that given the specific conditions, the final amount is $290 -/
theorem emma_bank_account : final_amount 230 60 = 290 := by
  sorry

end NUMINAMATH_CALUDE_emma_bank_account_l275_27548


namespace NUMINAMATH_CALUDE_password_combinations_l275_27580

/-- A digit is either odd or even -/
inductive Digit
| odd
| even

/-- The set of possible digits -/
def digit_set : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- A valid password is a list of four digits satisfying the given conditions -/
def ValidPassword : Type := List Digit

/-- The number of odd digits in the digit set -/
def num_odd_digits : Nat := (digit_set.filter (fun n => n % 2 = 1)).card

/-- The number of even digits in the digit set -/
def num_even_digits : Nat := (digit_set.filter (fun n => n % 2 = 0)).card

/-- The total number of digits in the digit set -/
def total_digits : Nat := digit_set.card

/-- The number of valid passwords -/
def num_valid_passwords : Nat := 
  (num_odd_digits * num_even_digits * total_digits * total_digits) +
  (num_even_digits * num_odd_digits * total_digits * total_digits)

theorem password_combinations : num_valid_passwords = 648 := by
  sorry

end NUMINAMATH_CALUDE_password_combinations_l275_27580


namespace NUMINAMATH_CALUDE_distinct_primes_count_l275_27551

theorem distinct_primes_count (n : ℕ) : n = 95 * 97 * 99 * 101 * 103 → 
  (Finset.card (Nat.factors n).toFinset) = 7 := by
sorry

end NUMINAMATH_CALUDE_distinct_primes_count_l275_27551


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l275_27546

def f (x : ℝ) : ℝ := 3 * x^2 + 8 * x - 10

theorem f_increasing_on_interval : 
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l275_27546


namespace NUMINAMATH_CALUDE_die_roll_probabilities_l275_27595

-- Define the sample space for rolling a fair six-sided die twice
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events
def A : Set Ω := {ω | ω.1 + ω.2 = 4}
def B : Set Ω := {ω | ω.2 % 2 = 0}
def C : Set Ω := {ω | ω.1 = ω.2}
def D : Set Ω := {ω | ω.1 % 2 = 1 ∨ ω.2 % 2 = 1}

-- Theorem statement
theorem die_roll_probabilities :
  P D = 3/4 ∧
  P (B ∩ D) = 1/4 ∧
  P (B ∩ C) = P B * P C := by sorry

end NUMINAMATH_CALUDE_die_roll_probabilities_l275_27595


namespace NUMINAMATH_CALUDE_range_of_a_l275_27560

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 2) ↔ ((x - a)^2 < 1)) ↔ 
  (1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l275_27560


namespace NUMINAMATH_CALUDE_different_color_pairings_l275_27518

/-- The number of distinct colors for bowls and glasses -/
def num_colors : ℕ := 5

/-- The number of pairings where the bowl and glass colors are different -/
def num_different_pairings : ℕ := num_colors * (num_colors - 1)

/-- Theorem stating that the number of different color pairings is 20 -/
theorem different_color_pairings :
  num_different_pairings = 20 := by
  sorry

#eval num_different_pairings -- This should output 20

end NUMINAMATH_CALUDE_different_color_pairings_l275_27518


namespace NUMINAMATH_CALUDE_mrs_copper_class_size_l275_27578

theorem mrs_copper_class_size :
  ∀ (initial_jellybeans : ℕ) 
    (absent_children : ℕ) 
    (jellybeans_per_child : ℕ) 
    (remaining_jellybeans : ℕ),
  initial_jellybeans = 100 →
  absent_children = 2 →
  jellybeans_per_child = 3 →
  remaining_jellybeans = 34 →
  ∃ (total_children : ℕ),
    total_children = 
      (initial_jellybeans - remaining_jellybeans) / jellybeans_per_child + absent_children ∧
    total_children = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_copper_class_size_l275_27578


namespace NUMINAMATH_CALUDE_factorization_of_36x_squared_minus_4_l275_27587

theorem factorization_of_36x_squared_minus_4 (x : ℝ) :
  36 * x^2 - 4 = 4 * (3*x + 1) * (3*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_36x_squared_minus_4_l275_27587


namespace NUMINAMATH_CALUDE_smallest_digit_count_l275_27562

theorem smallest_digit_count (a n : ℕ) (h : (Nat.log 10 (a^n) + 1 = 2014)) :
  ∀ k < 2014, ∃ a' : ℕ, 10^(k-1) ≤ a' ∧ a' < 10^k ∧ (Nat.log 10 (a'^n) + 1 = 2014) ∧
  ¬∃ a' : ℕ, 10^2013 ≤ a' ∧ a' < 10^2014 ∧ (Nat.log 10 (a'^n) + 1 = 2014) :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_count_l275_27562


namespace NUMINAMATH_CALUDE_quadratic_solution_l275_27523

theorem quadratic_solution (c : ℝ) : 
  ((-9 : ℝ)^2 + c * (-9 : ℝ) + 45 = 0) → c = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l275_27523


namespace NUMINAMATH_CALUDE_count_solutions_l275_27569

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem count_solutions : 
  (Finset.filter (fun n => n + S n + S (S n) = 2007) (Finset.range 2008)).card = 4 := by sorry

end NUMINAMATH_CALUDE_count_solutions_l275_27569


namespace NUMINAMATH_CALUDE_days_for_other_books_l275_27583

-- Define the daily charge for a book
def daily_charge : ℚ := 0.5

-- Define the number of books borrowed
def total_books : ℕ := 3

-- Define the number of days for the first book
def days_first_book : ℕ := 20

-- Define the total amount paid
def total_paid : ℚ := 41

-- Define a function to calculate the cost for a given number of books and days
def calculate_cost (books : ℕ) (days : ℕ) : ℚ :=
  (books : ℚ) * (days : ℚ) * daily_charge

-- Theorem to prove
theorem days_for_other_books : 
  ∃ (days : ℕ), 
    calculate_cost 1 days_first_book + calculate_cost 2 days = total_paid ∧ 
    days = 31 := by
  sorry

end NUMINAMATH_CALUDE_days_for_other_books_l275_27583
