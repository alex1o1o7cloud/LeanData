import Mathlib

namespace NUMINAMATH_CALUDE_omega_value_l1326_132615

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)

theorem omega_value (ω : ℝ) (α β : ℝ) 
  (h_pos : ω > 0)
  (h_f_alpha : f ω α = -2)
  (h_f_beta : f ω β = 0)
  (h_min_diff : ∀ γ δ : ℝ, f ω γ = -2 → f ω δ = 0 → |α - β| ≤ |γ - δ|)
  (h_diff : |α - β| = 3 * Real.pi / 4) :
  ω = 2/3 := by
sorry

end NUMINAMATH_CALUDE_omega_value_l1326_132615


namespace NUMINAMATH_CALUDE_sphere_radius_is_one_l1326_132628

/-- Represents a cone with a given base radius -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents the configuration of three cones and a sphere -/
structure ConeSphereProblem where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  sphere : Sphere
  sameHeight : cone1.height = cone2.height ∧ cone2.height = cone3.height
  baseRadii : cone1.baseRadius = 1 ∧ cone2.baseRadius = 2 ∧ cone3.baseRadius = 3
  touching : True  -- Cones are touching each other
  sphereTouchingCones : True  -- Sphere touches all cones
  sphereTouchingTable : True  -- Sphere touches the table
  centerEquidistant : True  -- Center of sphere is equidistant from all points of contact with cones

/-- The theorem stating that the radius of the sphere is 1 -/
theorem sphere_radius_is_one (problem : ConeSphereProblem) : problem.sphere.radius = 1 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_one_l1326_132628


namespace NUMINAMATH_CALUDE_average_scores_is_68_l1326_132668

def scores : List ℝ := [50, 60, 70, 80, 80]

theorem average_scores_is_68 : (scores.sum / scores.length) = 68 := by
  sorry

end NUMINAMATH_CALUDE_average_scores_is_68_l1326_132668


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_roots_l1326_132673

def fibonacci_like_sequence (F : ℕ → ℝ) : Prop :=
  F 0 = 2 ∧ F 1 = 3 ∧ ∀ n, F (n + 1) * F (n - 1) - F n ^ 2 = (-1) ^ n * 2

def has_exponential_form (F : ℕ → ℝ) (r₁ r₂ : ℝ) : Prop :=
  ∃ a b : ℝ, ∀ n, F n = a * r₁ ^ n + b * r₂ ^ n

theorem fibonacci_like_sequence_roots 
  (F : ℕ → ℝ) (r₁ r₂ : ℝ) 
  (h₁ : fibonacci_like_sequence F) 
  (h₂ : has_exponential_form F r₁ r₂) : 
  |r₁ - r₂| = Real.sqrt 17 / 2 := by sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_roots_l1326_132673


namespace NUMINAMATH_CALUDE_continuous_at_five_l1326_132625

def f (x : ℝ) : ℝ := 4 * x^2 - 2

theorem continuous_at_five :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 5| < δ → |f x - f 5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_continuous_at_five_l1326_132625


namespace NUMINAMATH_CALUDE_june1st_is_tuesday_l1326_132635

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific year -/
structure Year where
  febHasFiveSundays : Bool
  febHas29Days : Bool
  feb1stIsSunday : Bool

/-- Function to calculate the day of the week for June 1st -/
def june1stDayOfWeek (y : Year) : DayOfWeek :=
  sorry

/-- Theorem stating that June 1st is a Tuesday in the specified year -/
theorem june1st_is_tuesday (y : Year) 
  (h1 : y.febHasFiveSundays = true) 
  (h2 : y.febHas29Days = true) 
  (h3 : y.feb1stIsSunday = true) : 
  june1stDayOfWeek y = DayOfWeek.Tuesday :=
  sorry

end NUMINAMATH_CALUDE_june1st_is_tuesday_l1326_132635


namespace NUMINAMATH_CALUDE_difference_of_fractions_of_6000_l1326_132669

theorem difference_of_fractions_of_6000 : 
  (1 / 10 : ℚ) * 6000 - (1 / 1000 : ℚ) * 6000 = 594 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_fractions_of_6000_l1326_132669


namespace NUMINAMATH_CALUDE_distance_origin_to_line_l1326_132622

/-- The distance from the origin to the line x + 2y - 5 = 0 is √5 -/
theorem distance_origin_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + 2*y - 5 = 0}
  abs (5) / Real.sqrt (1^2 + 2^2) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_l1326_132622


namespace NUMINAMATH_CALUDE_fraction_problem_l1326_132633

theorem fraction_problem (x y : ℚ) 
  (h1 : y / (x - 1) = 1 / 3)
  (h2 : (y + 4) / x = 1 / 2) :
  y / x = 7 / 22 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1326_132633


namespace NUMINAMATH_CALUDE_probability_of_drawing_two_l1326_132689

def card_set : Finset ℕ := {1, 2, 2, 3, 5}

theorem probability_of_drawing_two (s : Finset ℕ := card_set) :
  (s.filter (· = 2)).card / s.card = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_two_l1326_132689


namespace NUMINAMATH_CALUDE_tangent_fraction_equals_one_l1326_132641

theorem tangent_fraction_equals_one (θ : Real) (h : Real.tan θ = -2 * Real.sqrt 2) :
  (2 * (Real.cos (θ / 2))^2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + π/4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_fraction_equals_one_l1326_132641


namespace NUMINAMATH_CALUDE_min_value_theorem_l1326_132623

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 4) :
  (1 / (4 - x) + 2 / x) ≥ (3 + 2 * Real.sqrt 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1326_132623


namespace NUMINAMATH_CALUDE_hiker_speed_day3_l1326_132629

/-- A hiker's three-day journey --/
structure HikerJourney where
  day1_distance : ℝ
  day1_speed : ℝ
  day2_hours_reduction : ℝ
  day2_speed_increase : ℝ
  day3_hours : ℝ
  total_distance : ℝ

/-- Theorem about the hiker's speed on the third day --/
theorem hiker_speed_day3 (journey : HikerJourney)
  (h1 : journey.day1_distance = 18)
  (h2 : journey.day1_speed = 3)
  (h3 : journey.day2_hours_reduction = 1)
  (h4 : journey.day2_speed_increase = 1)
  (h5 : journey.day3_hours = 3)
  (h6 : journey.total_distance = 53) :
  (journey.total_distance
    - journey.day1_distance
    - (journey.day1_distance / journey.day1_speed - journey.day2_hours_reduction)
      * (journey.day1_speed + journey.day2_speed_increase))
  / journey.day3_hours = 5 := by
  sorry


end NUMINAMATH_CALUDE_hiker_speed_day3_l1326_132629


namespace NUMINAMATH_CALUDE_largest_odd_factor_sum_difference_l1326_132614

/-- f(n) represents the largest odd factor of a positive integer n -/
def f (n : ℕ+) : ℕ+ :=
  sorry

/-- The sum of f(i) from a to b -/
def sum_f (a b : ℕ+) : ℕ :=
  sorry

theorem largest_odd_factor_sum_difference :
  sum_f 51 100 - sum_f 1 50 = 1656 :=
sorry

end NUMINAMATH_CALUDE_largest_odd_factor_sum_difference_l1326_132614


namespace NUMINAMATH_CALUDE_factorial_division_l1326_132620

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 5 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1326_132620


namespace NUMINAMATH_CALUDE_count_non_divisors_is_33_l1326_132607

/-- g(n) is the product of the proper positive integer divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- The number of integers n between 2 and 100 (inclusive) that do not divide g(n) -/
def count_non_divisors : ℕ := sorry

/-- Theorem stating that the count of non-divisors is 33 -/
theorem count_non_divisors_is_33 : count_non_divisors = 33 := by sorry

end NUMINAMATH_CALUDE_count_non_divisors_is_33_l1326_132607


namespace NUMINAMATH_CALUDE_robert_nickel_chocolate_difference_l1326_132605

theorem robert_nickel_chocolate_difference :
  let robert_chocolates : ℕ := 9
  let nickel_chocolates : ℕ := 2
  robert_chocolates - nickel_chocolates = 7 := by
sorry

end NUMINAMATH_CALUDE_robert_nickel_chocolate_difference_l1326_132605


namespace NUMINAMATH_CALUDE_complex_function_chain_l1326_132666

theorem complex_function_chain (x y u : ℂ) : 
  u = 2 * x - 5 → (y = (2 * x - 5)^10 ↔ y = u^10) := by
  sorry

end NUMINAMATH_CALUDE_complex_function_chain_l1326_132666


namespace NUMINAMATH_CALUDE_dog_cost_l1326_132667

/-- The cost of a dog given the current money and additional money needed -/
theorem dog_cost (current_money additional_money : ℕ) :
  current_money = 34 →
  additional_money = 13 →
  current_money + additional_money = 47 :=
by sorry

end NUMINAMATH_CALUDE_dog_cost_l1326_132667


namespace NUMINAMATH_CALUDE_only_2222_cannot_form_24_l1326_132685

/-- A hand is a list of four natural numbers representing card values. -/
def Hand := List Nat

/-- Possible arithmetic operations -/
inductive Operation
| Add
| Sub
| Mul
| Div

/-- Apply an operation to two natural numbers -/
def applyOp (op : Operation) (a b : Nat) : Option Nat :=
  match op with
  | Operation.Add => some (a + b)
  | Operation.Sub => if a ≥ b then some (a - b) else none
  | Operation.Mul => some (a * b)
  | Operation.Div => if b ≠ 0 && a % b = 0 then some (a / b) else none

/-- Check if a hand can form 24 using the given operations and rules -/
def canForm24 (hand : Hand) : Prop :=
  ∃ (op1 op2 op3 : Operation) (perm : List Nat),
    perm.length = 4 ∧
    perm.toFinset = hand.toFinset ∧
    (∃ (x y z : Nat),
      applyOp op1 perm[0]! perm[1]! = some x ∧
      applyOp op2 x perm[2]! = some y ∧
      applyOp op3 y perm[3]! = some 24)

theorem only_2222_cannot_form_24 :
  canForm24 [1, 2, 3, 3] ∧
  canForm24 [1, 5, 5, 5] ∧
  canForm24 [3, 3, 3, 3] ∧
  ¬canForm24 [2, 2, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_only_2222_cannot_form_24_l1326_132685


namespace NUMINAMATH_CALUDE_different_testing_methods_part1_different_testing_methods_part2_l1326_132655

/-- The number of products -/
def n : ℕ := 10

/-- The number of defective products -/
def d : ℕ := 4

/-- The position of the first defective product in part 1 -/
def first_defective : ℕ := 5

/-- The position of the last defective product in part 1 -/
def last_defective : ℕ := 10

/-- The number of different testing methods in part 1 -/
def methods_part1 : ℕ := 103680

/-- The number of different testing methods in part 2 -/
def methods_part2 : ℕ := 576

/-- Theorem for part 1 -/
theorem different_testing_methods_part1 :
  (n = 10) → (d = 4) → (first_defective = 5) → (last_defective = 10) →
  methods_part1 = 103680 := by sorry

/-- Theorem for part 2 -/
theorem different_testing_methods_part2 :
  (n = 10) → (d = 4) → methods_part2 = 576 := by sorry

end NUMINAMATH_CALUDE_different_testing_methods_part1_different_testing_methods_part2_l1326_132655


namespace NUMINAMATH_CALUDE_train_passing_time_l1326_132679

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 80 → 
  train_speed_kmph = 36 → 
  (train_length / (train_speed_kmph * 1000 / 3600)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1326_132679


namespace NUMINAMATH_CALUDE_juan_lunch_time_l1326_132683

/-- The number of pages in Juan's book -/
def book_pages : ℕ := 4000

/-- The number of pages Juan reads per hour -/
def pages_per_hour : ℕ := 250

/-- The time it takes Juan to read the entire book, in hours -/
def reading_time : ℚ := book_pages / pages_per_hour

/-- The time it takes Juan to grab lunch from his office and back, in hours -/
def lunch_time : ℚ := reading_time / 2

theorem juan_lunch_time : lunch_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_juan_lunch_time_l1326_132683


namespace NUMINAMATH_CALUDE_yellow_parrot_count_l1326_132681

theorem yellow_parrot_count (total : ℕ) (red_fraction : ℚ) : 
  total = 120 → red_fraction = 5/8 → (total : ℚ) * (1 - red_fraction) = 45 := by
  sorry

end NUMINAMATH_CALUDE_yellow_parrot_count_l1326_132681


namespace NUMINAMATH_CALUDE_discount_difference_l1326_132624

/-- Proves that the difference between the claimed discount and the true discount is 9% -/
theorem discount_difference (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) :
  initial_discount = 0.4 →
  additional_discount = 0.1 →
  claimed_discount = 0.55 →
  claimed_discount - (1 - (1 - initial_discount) * (1 - additional_discount)) = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l1326_132624


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l1326_132695

/-- Given two cylinders A and B with base areas S₁ and S₂, volumes V₁ and V₂,
    if S₁/S₂ = 9/4 and their lateral surface areas are equal, then V₁/V₂ = 3/2 -/
theorem cylinder_volume_ratio (S₁ S₂ V₁ V₂ R r H h : ℝ) 
    (h_base_ratio : S₁ / S₂ = 9 / 4)
    (h_S₁ : S₁ = π * R^2)
    (h_S₂ : S₂ = π * r^2)
    (h_V₁ : V₁ = S₁ * H)
    (h_V₂ : V₂ = S₂ * h)
    (h_lateral_area : 2 * π * R * H = 2 * π * r * h) : 
  V₁ / V₂ = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l1326_132695


namespace NUMINAMATH_CALUDE_no_prime_roots_for_equation_l1326_132672

theorem no_prime_roots_for_equation (k : ℤ) : 
  ¬∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (p : ℤ) + q = 101 ∧
    (p : ℤ) * q = k ∧
    p ≠ q :=
by sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_equation_l1326_132672


namespace NUMINAMATH_CALUDE_inverse_square_inequality_l1326_132660

theorem inverse_square_inequality {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x ≤ y) :
  1 / y ^ 2 ≤ 1 / x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_inequality_l1326_132660


namespace NUMINAMATH_CALUDE_area_of_triangle_FOH_l1326_132670

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Theorem about the area of triangle FOH in a trapezoid -/
theorem area_of_triangle_FOH (t : Trapezoid) 
  (h1 : t.base1 = 40)
  (h2 : t.base2 = 50)
  (h3 : t.area = 900) : 
  ∃ (area_FOH : ℝ), abs (area_FOH - 400/9) < 0.01 := by
  sorry

#check area_of_triangle_FOH

end NUMINAMATH_CALUDE_area_of_triangle_FOH_l1326_132670


namespace NUMINAMATH_CALUDE_school_distance_l1326_132674

/-- The distance between a student's house and school, given travel time conditions. -/
theorem school_distance (t : ℝ) : 
  (t + 1/3 = 24/9) → (t - 1/3 = 24/12) → 24 = 24 := by
  sorry

end NUMINAMATH_CALUDE_school_distance_l1326_132674


namespace NUMINAMATH_CALUDE_mirror_frame_areas_l1326_132658

/-- Represents the dimensions and properties of a rectangular mirror frame -/
structure MirrorFrame where
  outer_width : ℝ
  outer_length : ℝ
  frame_width : ℝ

/-- Calculates the area of the frame alone -/
def frame_area (frame : MirrorFrame) : ℝ :=
  frame.outer_width * frame.outer_length - (frame.outer_width - 2 * frame.frame_width) * (frame.outer_length - 2 * frame.frame_width)

/-- Calculates the area of the mirror inside the frame -/
def mirror_area (frame : MirrorFrame) : ℝ :=
  (frame.outer_width - 2 * frame.frame_width) * (frame.outer_length - 2 * frame.frame_width)

theorem mirror_frame_areas (frame : MirrorFrame) 
  (h1 : frame.outer_width = 100)
  (h2 : frame.outer_length = 120)
  (h3 : frame.frame_width = 15) :
  frame_area frame = 5700 ∧ mirror_area frame = 6300 := by
  sorry

end NUMINAMATH_CALUDE_mirror_frame_areas_l1326_132658


namespace NUMINAMATH_CALUDE_min_value_is_nine_l1326_132601

/-- Two circles C₁ and C₂ with centers and radii -/
structure TwoCircles where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0

/-- The circles have only one common tangent -/
axiom one_common_tangent (c : TwoCircles) : 4 * c.a^2 + c.b^2 = 1

/-- The minimum value of 1/a² + 1/b² is 9 -/
theorem min_value_is_nine (c : TwoCircles) : 
  ∀ ε > 0, (1 / c.a^2 + 1 / c.b^2) > 9 - ε :=
sorry

end NUMINAMATH_CALUDE_min_value_is_nine_l1326_132601


namespace NUMINAMATH_CALUDE_range_of_function_1_range_of_function_1_supremum_l1326_132643

theorem range_of_function_1 (x : ℝ) :
  ∃ y : ℝ, y = x + Real.sqrt (1 - 2 * x) ∧ y ≤ 1 :=
sorry

theorem range_of_function_1_supremum :
  ∀ ε > 0, ∃ x y : ℝ, y = x + Real.sqrt (1 - 2 * x) ∧ y > 1 - ε :=
sorry

end NUMINAMATH_CALUDE_range_of_function_1_range_of_function_1_supremum_l1326_132643


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l1326_132612

/-- Calculates the average speed of a cyclist who drives four laps of equal distance
    at different speeds. -/
theorem cyclist_average_speed (d : ℝ) (h : d > 0) :
  let speeds := [6, 12, 18, 24]
  let total_distance := 4 * d
  let total_time := d / 6 + d / 12 + d / 18 + d / 24
  total_distance / total_time = 288 / 25 := by
sorry

#eval (288 : ℚ) / 25  -- To verify the result is approximately 11.52

end NUMINAMATH_CALUDE_cyclist_average_speed_l1326_132612


namespace NUMINAMATH_CALUDE_equal_cost_at_20_teacups_l1326_132671

/-- The price of a teapot in yuan -/
def teapot_price : ℝ := 30

/-- The price of a teacup in yuan -/
def teacup_price : ℝ := 5

/-- The number of teapots bought -/
def num_teapots : ℕ := 5

/-- The cost function for Store A -/
def cost_A (x : ℝ) : ℝ := 5 * x + 125

/-- The cost function for Store B -/
def cost_B (x : ℝ) : ℝ := 4.5 * x + 135

/-- The number of teacups at which the costs are equal -/
def equal_cost_teacups : ℝ := 20

theorem equal_cost_at_20_teacups :
  cost_A equal_cost_teacups = cost_B equal_cost_teacups :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_20_teacups_l1326_132671


namespace NUMINAMATH_CALUDE_train_speed_problem_l1326_132630

/-- Proves that the original speed of a train is 60 km/h given the specified conditions -/
theorem train_speed_problem (delay : Real) (distance : Real) (speed_increase : Real) :
  delay = 0.2 ∧ distance = 60 ∧ speed_increase = 15 →
  ∃ original_speed : Real,
    original_speed > 0 ∧
    distance / original_speed - distance / (original_speed + speed_increase) = delay ∧
    original_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1326_132630


namespace NUMINAMATH_CALUDE_lily_painting_rate_l1326_132631

/-- Represents the number of cups Gina can paint per hour -/
structure PaintingRate where
  roses : ℕ
  lilies : ℕ

/-- Represents an order of cups -/
structure Order where
  roses : ℕ
  lilies : ℕ

theorem lily_painting_rate 
  (gina_rate : PaintingRate)
  (order : Order)
  (total_payment : ℕ)
  (hourly_rate : ℕ)
  (h1 : gina_rate.roses = 6)
  (h2 : order.roses = 6)
  (h3 : order.lilies = 14)
  (h4 : total_payment = 90)
  (h5 : hourly_rate = 30) :
  gina_rate.lilies = 7 := by
  sorry

end NUMINAMATH_CALUDE_lily_painting_rate_l1326_132631


namespace NUMINAMATH_CALUDE_zoes_purchase_cost_l1326_132611

/-- The total cost of soda and pizza for a group, given the cost per item and number of people -/
def totalCost (sodaCost pizzaCost : ℚ) (numPeople : ℕ) : ℚ :=
  numPeople * (sodaCost + pizzaCost)

/-- Theorem: The total cost for soda and pizza for 6 people is $9.00 -/
theorem zoes_purchase_cost :
  totalCost (1/2) 1 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_zoes_purchase_cost_l1326_132611


namespace NUMINAMATH_CALUDE_chocolate_manufacturer_cost_l1326_132656

/-- Proves that the cost per unit must be ≤ £340 given the problem conditions -/
theorem chocolate_manufacturer_cost (
  monthly_production : ℕ)
  (selling_price : ℝ)
  (minimum_profit : ℝ)
  (cost_per_unit : ℝ)
  (h1 : monthly_production = 400)
  (h2 : selling_price = 440)
  (h3 : minimum_profit = 40000)
  (h4 : monthly_production * selling_price - monthly_production * cost_per_unit ≥ minimum_profit) :
  cost_per_unit ≤ 340 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_manufacturer_cost_l1326_132656


namespace NUMINAMATH_CALUDE_calculate_3Y5_l1326_132686

-- Define the operation Y
def Y (a b : ℤ) : ℤ := b + 12 * a - a^3

-- Theorem to prove
theorem calculate_3Y5 : Y 3 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_calculate_3Y5_l1326_132686


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_distinct_roots_l1326_132609

theorem at_least_one_equation_has_two_distinct_roots
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  ¬(4*b^2 - 4*a*c ≤ 0 ∧ 4*c^2 - 4*a*b ≤ 0 ∧ 4*a^2 - 4*b*c ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_distinct_roots_l1326_132609


namespace NUMINAMATH_CALUDE_balanced_quadruple_theorem_l1326_132646

def is_balanced (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem balanced_quadruple_theorem :
  ∀ x : ℝ, x > 0 →
  (∀ a b c d : ℝ, is_balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔
  x ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_balanced_quadruple_theorem_l1326_132646


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l1326_132696

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  5 * x^2 = 15 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (x + 3)^3 = -64 ↔ x = -7 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l1326_132696


namespace NUMINAMATH_CALUDE_weeks_per_season_l1326_132697

def weekly_earnings : ℕ := 1357
def num_seasons : ℕ := 73
def total_earnings : ℕ := 22090603

theorem weeks_per_season : 
  (total_earnings / weekly_earnings) / num_seasons = 223 :=
sorry

end NUMINAMATH_CALUDE_weeks_per_season_l1326_132697


namespace NUMINAMATH_CALUDE_quadratic_equation_magnitude_l1326_132647

theorem quadratic_equation_magnitude (z : ℂ) : 
  z^2 - 10*z + 28 = 0 → ∃! m : ℝ, ∃ z : ℂ, z^2 - 10*z + 28 = 0 ∧ Complex.abs z = m :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_magnitude_l1326_132647


namespace NUMINAMATH_CALUDE_total_boxes_in_cases_l1326_132694

/-- The number of cases Jenny needs to deliver -/
def num_cases : ℕ := 3

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 8

/-- Theorem: The total number of boxes in the cases Jenny needs to deliver is 24 -/
theorem total_boxes_in_cases : num_cases * boxes_per_case = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_in_cases_l1326_132694


namespace NUMINAMATH_CALUDE_sum_of_roots_l1326_132652

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1326_132652


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1326_132636

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) : 
  2*x^2*y - 8*x*y + 8*y = 2*y*(x-2)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (x : ℝ) :
  x^4 - 81 = (x^2 + 9)*(x - 3)*(x + 3) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1326_132636


namespace NUMINAMATH_CALUDE_union_of_sets_l1326_132663

theorem union_of_sets : 
  let M : Set ℤ := {-1, 0, 1}
  let N : Set ℤ := {0, 1, 2}
  M ∪ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l1326_132663


namespace NUMINAMATH_CALUDE_remainder_problem_l1326_132661

theorem remainder_problem (n : ℕ) : 
  n % 68 = 0 ∧ n / 68 = 269 → n % 67 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1326_132661


namespace NUMINAMATH_CALUDE_sample_size_correct_l1326_132617

/-- Represents a population of students -/
structure Population where
  size : ℕ

/-- Represents a sample of students -/
structure Sample where
  size : ℕ

/-- Theorem stating that the sample size is correct -/
theorem sample_size_correct (pop : Population) (samp : Sample) : 
  pop.size = 8000 → samp.size = 400 → samp.size = 400 := by sorry

end NUMINAMATH_CALUDE_sample_size_correct_l1326_132617


namespace NUMINAMATH_CALUDE_charity_ticket_sales_l1326_132632

theorem charity_ticket_sales (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 140)
  (h_total_revenue : total_revenue = 2001) :
  ∃ (full_price : ℕ) (half_price : ℕ) (full_price_tickets : ℕ) (half_price_tickets : ℕ),
    full_price > 0 ∧
    half_price = full_price / 2 ∧
    full_price_tickets + half_price_tickets = total_tickets ∧
    full_price_tickets * full_price + half_price_tickets * half_price = total_revenue ∧
    full_price_tickets * full_price = 782 :=
by sorry

end NUMINAMATH_CALUDE_charity_ticket_sales_l1326_132632


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_l1326_132639

theorem no_solution_to_inequality :
  ¬∃ x : ℝ, (4 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 5) := by
sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_l1326_132639


namespace NUMINAMATH_CALUDE_journey_speed_theorem_l1326_132618

/-- Given a journey with the following parameters:
  * total_distance: The total distance traveled in miles
  * total_time: The total time of the journey in minutes
  * speed_first_30: The average speed during the first 30 minutes in mph
  * speed_second_30: The average speed during the second 30 minutes in mph

  This function calculates the average speed during the last 60 minutes of the journey. -/
def average_speed_last_60 (total_distance : ℝ) (total_time : ℝ) (speed_first_30 : ℝ) (speed_second_30 : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, 
    the average speed during the last 60 minutes is 77.5 mph -/
theorem journey_speed_theorem :
  average_speed_last_60 150 120 75 70 = 77.5 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_theorem_l1326_132618


namespace NUMINAMATH_CALUDE_complex_rearrangements_with_vowels_first_l1326_132637

def word : String := "COMPLEX"

def is_vowel (c : Char) : Bool :=
  c = 'O' || c = 'E'

def vowels : List Char :=
  word.data.filter is_vowel

def consonants : List Char :=
  word.data.filter (fun c => !is_vowel c)

theorem complex_rearrangements_with_vowels_first :
  (vowels.permutations.length) * (consonants.permutations.length) = 240 := by
  sorry

end NUMINAMATH_CALUDE_complex_rearrangements_with_vowels_first_l1326_132637


namespace NUMINAMATH_CALUDE_tank_solution_volume_l1326_132619

theorem tank_solution_volume 
  (V : ℝ) 
  (h1 : V > 0) 
  (h2 : 0.05 * V / (V - 5500) = 1 / 9) : 
  V = 10000 := by
sorry

end NUMINAMATH_CALUDE_tank_solution_volume_l1326_132619


namespace NUMINAMATH_CALUDE_sandy_initial_money_l1326_132699

/-- Sandy's initial amount of money before buying the pie -/
def initial_money : ℕ := sorry

/-- The cost of the pie -/
def pie_cost : ℕ := 6

/-- The amount of money Sandy has left after buying the pie -/
def remaining_money : ℕ := 57

/-- Theorem stating that Sandy's initial amount of money was 63 dollars -/
theorem sandy_initial_money : initial_money = 63 := by sorry

end NUMINAMATH_CALUDE_sandy_initial_money_l1326_132699


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l1326_132675

theorem exactly_one_positive_integer_solution : 
  ∃! (n : ℕ+), 24 - 6 * (n : ℝ) > 12 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l1326_132675


namespace NUMINAMATH_CALUDE_sweetsies_remainder_l1326_132606

theorem sweetsies_remainder (m : ℕ) (h : m % 7 = 5) : (2 * m) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sweetsies_remainder_l1326_132606


namespace NUMINAMATH_CALUDE_students_like_both_correct_l1326_132616

/-- The number of students who like both apple pie and chocolate cake -/
def students_like_both (total : ℕ) (apple : ℕ) (chocolate : ℕ) (pumpkin : ℕ) (none : ℕ) : ℕ := 
  apple + chocolate - (total - none)

theorem students_like_both_correct (total : ℕ) (apple : ℕ) (chocolate : ℕ) (pumpkin : ℕ) (none : ℕ) 
  (h1 : total = 50)
  (h2 : apple = 22)
  (h3 : chocolate = 20)
  (h4 : pumpkin = 17)
  (h5 : none = 15) :
  students_like_both total apple chocolate pumpkin none = 7 := by
  sorry

#eval students_like_both 50 22 20 17 15

end NUMINAMATH_CALUDE_students_like_both_correct_l1326_132616


namespace NUMINAMATH_CALUDE_solve_for_y_l1326_132665

theorem solve_for_y (x y : ℝ) (h1 : x^(2*y) = 64) (h2 : x = 8) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1326_132665


namespace NUMINAMATH_CALUDE_even_numbers_average_21_l1326_132603

/-- The sum of the first n even numbers -/
def sumFirstEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The average of the first n even numbers -/
def averageFirstEvenNumbers (n : ℕ) : ℚ := (sumFirstEvenNumbers n : ℚ) / n

theorem even_numbers_average_21 :
  ∃ n : ℕ, n > 0 ∧ averageFirstEvenNumbers n = 21 :=
sorry

end NUMINAMATH_CALUDE_even_numbers_average_21_l1326_132603


namespace NUMINAMATH_CALUDE_regular_polygon_center_containment_l1326_132648

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  sideLength : ℝ
  center : ℝ × ℝ

/-- M1 is situated inside M2 -/
def isInside (M1 M2 : RegularPolygon n) : Prop :=
  sorry

/-- The center of a polygon -/
def centerOf (M : RegularPolygon n) : ℝ × ℝ :=
  M.center

/-- A point is contained in a polygon -/
def contains (M : RegularPolygon n) (p : ℝ × ℝ) : Prop :=
  sorry

theorem regular_polygon_center_containment (n : ℕ) (M1 M2 : RegularPolygon n) 
  (h1 : M1.sideLength = a)
  (h2 : M2.sideLength = 2 * a)
  (h3 : isInside M1 M2)
  : contains M1 (centerOf M2) :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_center_containment_l1326_132648


namespace NUMINAMATH_CALUDE_debt_ratio_proof_l1326_132678

/-- Proves that the ratio of Aryan's debt to Kyro's debt is 2:1 given the problem conditions --/
theorem debt_ratio_proof (aryan_debt kyro_debt : ℝ) 
  (h1 : aryan_debt = 1200)
  (h2 : 0.6 * aryan_debt + 0.8 * kyro_debt + 300 = 1500) :
  aryan_debt / kyro_debt = 2 := by
  sorry


end NUMINAMATH_CALUDE_debt_ratio_proof_l1326_132678


namespace NUMINAMATH_CALUDE_teenas_speed_l1326_132649

theorem teenas_speed (initial_distance : ℝ) (poes_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 7.5 →
  poes_speed = 40 →
  time = 1.5 →
  final_distance = 15 →
  (initial_distance + poes_speed * time + final_distance) / time = 55 := by
sorry

end NUMINAMATH_CALUDE_teenas_speed_l1326_132649


namespace NUMINAMATH_CALUDE_tangent_slope_implies_function_value_l1326_132682

open Real

theorem tangent_slope_implies_function_value (x₀ : ℝ) (h : x₀ > 0) : 
  let f : ℝ → ℝ := λ x ↦ log x + 2 * x
  (deriv f x₀ = 3) → f x₀ = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_function_value_l1326_132682


namespace NUMINAMATH_CALUDE_complex_inequality_l1326_132680

theorem complex_inequality (z : ℂ) (n : ℕ) (h1 : z.re ≥ 1) (h2 : n ≥ 4) :
  Complex.abs (z^(n+1) - 1) ≥ Complex.abs (z^n) * Complex.abs (z - 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l1326_132680


namespace NUMINAMATH_CALUDE_f_max_value_l1326_132650

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - x^3

-- State the theorem
theorem f_max_value :
  ∃ (c : ℝ), c > 0 ∧ f c = 2 ∧ ∀ x > 0, f x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l1326_132650


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1326_132644

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1326_132644


namespace NUMINAMATH_CALUDE_parabola_properties_l1326_132677

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 4 * x * y + y^2 - 10 * y - 15 = 0

-- Define the axis of symmetry
def axis_of_symmetry (x y : ℝ) : Prop :=
  2 * x + y - 1 = 0

-- Define the directrix
def directrix (x y : ℝ) : Prop :=
  x - 2 * y - 5 = 0

-- Define the tangent line
def tangent_line (y : ℝ) : Prop :=
  2 * y + 3 = 0

-- Theorem statement
theorem parabola_properties :
  ∀ (x y : ℝ),
    parabola_equation x y →
    (∃ (x₀ y₀ : ℝ), axis_of_symmetry x₀ y₀ ∧
                     directrix x₀ y₀ ∧
                     tangent_line y₀) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1326_132677


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1326_132676

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = 4/5 + 3/5 * Complex.I →
  Complex.exp (φ * Complex.I) = -5/13 + 12/13 * Complex.I →
  Real.sin (θ + φ) = 84/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1326_132676


namespace NUMINAMATH_CALUDE_alex_jane_pen_difference_l1326_132640

/-- The number of pens Alex has after n weeks, given she starts with 4 pens and triples her collection each week -/
def alexPens (n : ℕ) : ℕ := 4 * 3^(n - 1)

/-- The number of pens Jane has after a month -/
def janePens : ℕ := 50

/-- The number of weeks in a month -/
def weeksInMonth : ℕ := 4

theorem alex_jane_pen_difference :
  alexPens weeksInMonth - janePens = 58 := by sorry

end NUMINAMATH_CALUDE_alex_jane_pen_difference_l1326_132640


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l1326_132664

theorem det_2x2_matrix (x : ℝ) : 
  Matrix.det !![5, x; 4, 3] = 15 - 4 * x := by sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l1326_132664


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l1326_132642

/-- For an infinite geometric series with common ratio 1/4 and sum 40, the first term is 30 -/
theorem geometric_series_first_term (a : ℝ) : 
  (∀ n : ℕ, ∑' k, a * (1/4)^k = 40) → a = 30 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l1326_132642


namespace NUMINAMATH_CALUDE_inequality_solution_l1326_132684

theorem inequality_solution (x : ℝ) : 
  (202 * Real.sqrt (x^3 - 2*x - 2/x + 1/x^3 + 4) ≤ 0) ↔ 
  (x = (-1 - Real.sqrt 17 + Real.sqrt (2 * Real.sqrt 17 + 2)) / 4 ∨ 
   x = (-1 - Real.sqrt 17 - Real.sqrt (2 * Real.sqrt 17 + 2)) / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1326_132684


namespace NUMINAMATH_CALUDE_small_paintings_completed_l1326_132657

/-- Represents the number of ounces of paint used for a large canvas --/
def paint_per_large_canvas : ℕ := 3

/-- Represents the number of ounces of paint used for a small canvas --/
def paint_per_small_canvas : ℕ := 2

/-- Represents the number of large paintings completed --/
def large_paintings_completed : ℕ := 3

/-- Represents the total amount of paint used in ounces --/
def total_paint_used : ℕ := 17

/-- Proves that the number of small paintings completed is 4 --/
theorem small_paintings_completed :
  (total_paint_used - large_paintings_completed * paint_per_large_canvas) / paint_per_small_canvas = 4 :=
by sorry

end NUMINAMATH_CALUDE_small_paintings_completed_l1326_132657


namespace NUMINAMATH_CALUDE_cylinder_volume_unit_dimensions_l1326_132621

/-- The volume of a cylinder with base radius 1 and height 1 is π. -/
theorem cylinder_volume_unit_dimensions : 
  let r : ℝ := 1
  let h : ℝ := 1
  let V := π * r^2 * h
  V = π := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_unit_dimensions_l1326_132621


namespace NUMINAMATH_CALUDE_natural_number_pairs_equality_l1326_132638

theorem natural_number_pairs_equality (m n : ℕ) : 
  n * (n - 1) * (n - 2) * (n - 3) = m * (m - 1) ↔ 
  ((m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 1 ∧ n = 3)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_equality_l1326_132638


namespace NUMINAMATH_CALUDE_boys_candies_order_independent_l1326_132604

/-- Represents a child's gender -/
inductive Gender
| Boy
| Girl

/-- Represents a child with their gender -/
structure Child where
  gender : Gender

/-- Represents the state of the candy distribution process -/
structure CandyState where
  remaining_candies : ℕ
  remaining_children : List Child

/-- Represents the result of a candy distribution process -/
structure DistributionResult where
  boys_candies : ℕ
  girls_candies : ℕ

/-- Function to distribute candies according to the rules -/
def distributeCandies (initial_state : CandyState) : DistributionResult :=
  sorry

/-- Theorem stating that the number of candies taken by boys is independent of the order -/
theorem boys_candies_order_independent
  (children : List Child)
  (perm : List Child)
  (h : perm.Perm children) :
  (distributeCandies { remaining_candies := 2021, remaining_children := children }).boys_candies =
  (distributeCandies { remaining_candies := 2021, remaining_children := perm }).boys_candies :=
  sorry

end NUMINAMATH_CALUDE_boys_candies_order_independent_l1326_132604


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1326_132602

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 = 2)
  (h3 : d = 1) : 
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1326_132602


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l1326_132690

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point on a circle. -/
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

/-- A coloring function that assigns either red or blue to each point on the circle. -/
def Coloring (c : Circle) := PointOnCircle c → Bool

/-- Predicate to check if three points form a right-angled triangle. -/
def IsRightAngledTriangle (c : Circle) (p1 p2 p3 : PointOnCircle c) : Prop :=
  ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    let points := [p1, p2, p3]
    let a := points[i].point
    let b := points[j].point
    let c := points[k].point
    (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0

/-- The main theorem: there exists a coloring such that no inscribed right-angled triangle
    has all vertices of the same color. -/
theorem exists_valid_coloring (c : Circle) :
  ∃ (coloring : Coloring c),
    ∀ (p1 p2 p3 : PointOnCircle c),
      IsRightAngledTriangle c p1 p2 p3 →
        coloring p1 ≠ coloring p2 ∨ coloring p2 ≠ coloring p3 ∨ coloring p1 ≠ coloring p3 :=
by sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l1326_132690


namespace NUMINAMATH_CALUDE_grid_paths_eq_binom_binom_eq_factorial_div_grid_paths_eq_factorial_div_l1326_132600

/-- The number of paths from (0,0) to (n,m) on a grid where only north and east movements are allowed -/
def grid_paths (m n : ℕ) : ℕ := sorry

/-- The binomial coefficient -/
def binom (n k : ℕ) : ℕ := sorry

theorem grid_paths_eq_binom (m n : ℕ) :
  grid_paths m n = binom (m + n) m :=
sorry

theorem binom_eq_factorial_div (n k : ℕ) :
  binom n k = (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k))) :=
sorry

theorem grid_paths_eq_factorial_div (m n : ℕ) :
  grid_paths m n = (Nat.factorial (m + n)) / ((Nat.factorial m) * (Nat.factorial n)) :=
sorry

end NUMINAMATH_CALUDE_grid_paths_eq_binom_binom_eq_factorial_div_grid_paths_eq_factorial_div_l1326_132600


namespace NUMINAMATH_CALUDE_only_two_solutions_l1326_132688

/-- Represents a solution of steers and cows --/
structure Solution :=
  (s : ℕ+)
  (c : ℕ+)

/-- Checks if a solution is valid given the budget constraint --/
def is_valid_solution (sol : Solution) : Prop :=
  30 * sol.s.val + 35 * sol.c.val = 1500

/-- The set of all valid solutions --/
def valid_solutions : Set Solution :=
  {sol : Solution | is_valid_solution sol}

/-- The theorem stating that there are only two valid solutions --/
theorem only_two_solutions :
  valid_solutions = {⟨1, 42⟩, ⟨36, 12⟩} :=
sorry

end NUMINAMATH_CALUDE_only_two_solutions_l1326_132688


namespace NUMINAMATH_CALUDE_inequality_holds_l1326_132613

theorem inequality_holds (a b c : ℝ) (h : a > b) : a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1326_132613


namespace NUMINAMATH_CALUDE_arccos_one_half_l1326_132634

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by sorry

end NUMINAMATH_CALUDE_arccos_one_half_l1326_132634


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1326_132626

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b : ℝ, a < b → a < b + 1) ∧
  (∃ a b : ℝ, a < b + 1 ∧ ¬(a < b)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1326_132626


namespace NUMINAMATH_CALUDE_cubic_function_property_l1326_132693

/-- A cubic function g(x) with coefficients p, q, r, and s. -/
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

/-- Theorem stating that for a cubic function g(x) = px³ + qx² + rx + s,
    if g(-3) = 4, then 10p - 5q + 3r - 2s = 40. -/
theorem cubic_function_property (p q r s : ℝ) : 
  g p q r s (-3) = 4 → 10*p - 5*q + 3*r - 2*s = 40 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1326_132693


namespace NUMINAMATH_CALUDE_max_value_theorem_l1326_132654

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h2 : a + b + c = 3) (h3 : a = 1) :
  (∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z → x + y + z = 3 → x = 1 →
    (a*b)/(a+b) + (a*c)/(a+c) + (b*c)/(b+c) ≥ (x*y)/(x+y) + (x*z)/(x+z) + (y*z)/(y+z)) ∧
  (∃ b' c' : ℝ, 0 ≤ b' ∧ 0 ≤ c' ∧ a + b' + c' = 3 ∧
    (a*b')/(a+b') + (a*c')/(a+c') + (b'*c')/(b'+c') = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1326_132654


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_in_expansion_fourth_term_has_largest_coefficient_l1326_132627

theorem largest_binomial_coefficient_in_expansion :
  ∀ k : ℕ, k ≤ 6 → Nat.choose 6 3 ≥ Nat.choose 6 k :=
by sorry

theorem fourth_term_has_largest_coefficient :
  ∃ k : ℕ, k = 4 ∧
  ∀ j : ℕ, j ≤ 6 → Nat.choose 6 (k - 1) ≥ Nat.choose 6 j :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_in_expansion_fourth_term_has_largest_coefficient_l1326_132627


namespace NUMINAMATH_CALUDE_systematic_sampling_prob_example_l1326_132645

/-- Represents the probability of selection in systematic sampling -/
def systematic_sampling_probability (sample_size : ℕ) (population_size : ℕ) : ℚ :=
  sample_size / population_size

/-- Theorem: In systematic sampling with a sample size of 15 and a population size of 152,
    the probability of each person being selected is 15/152 -/
theorem systematic_sampling_prob_example :
  systematic_sampling_probability 15 152 = 15 / 152 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_prob_example_l1326_132645


namespace NUMINAMATH_CALUDE_min_value_of_f_l1326_132653

theorem min_value_of_f (x : ℝ) (hx : x > 0) : x + 1/x - 2 ≥ 0 ∧ (x + 1/x - 2 = 0 ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1326_132653


namespace NUMINAMATH_CALUDE_max_nine_letter_palindromes_l1326_132608

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the palindromes we're considering -/
def palindrome_length : ℕ := 9

/-- A palindrome is a word that reads the same forward and backward -/
def is_palindrome (word : List Char) : Prop :=
  word = word.reverse

/-- The maximum number of 9-letter palindromes using the English alphabet -/
theorem max_nine_letter_palindromes :
  (alphabet_size ^ ((palindrome_length - 1) / 2 + 1) : ℕ) = 11881376 :=
sorry

end NUMINAMATH_CALUDE_max_nine_letter_palindromes_l1326_132608


namespace NUMINAMATH_CALUDE_power_of_power_l1326_132698

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l1326_132698


namespace NUMINAMATH_CALUDE_smallest_number_in_specific_integer_set_l1326_132662

theorem smallest_number_in_specific_integer_set :
  ∀ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    (a + b + c : ℚ) / 3 = 30 →
    b = 29 →
    max a (max b c) = b + 4 →
    min a (min b c) = 28 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_in_specific_integer_set_l1326_132662


namespace NUMINAMATH_CALUDE_cabbage_price_calculation_l1326_132659

/-- Represents the price of the cabbage in Janet's grocery purchase. -/
def cabbage_price : ℝ := sorry

/-- Represents Janet's total grocery budget. -/
def total_budget : ℝ := sorry

theorem cabbage_price_calculation :
  let broccoli_cost : ℝ := 3 * 4
  let oranges_cost : ℝ := 3 * 0.75
  let bacon_cost : ℝ := 1 * 3
  let chicken_cost : ℝ := 2 * 3
  let meat_cost : ℝ := bacon_cost + chicken_cost
  let known_items_cost : ℝ := broccoli_cost + oranges_cost + bacon_cost + chicken_cost
  meat_cost = 0.33 * total_budget ∧
  cabbage_price = total_budget - known_items_cost →
  cabbage_price = 4.02 := by sorry

end NUMINAMATH_CALUDE_cabbage_price_calculation_l1326_132659


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_l1326_132687

theorem pencil_eraser_cost :
  ∀ (p e : ℕ),
  15 * p + 5 * e = 125 →
  p > e →
  p > 0 →
  e > 0 →
  p + e = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_l1326_132687


namespace NUMINAMATH_CALUDE_rhombus_area_l1326_132691

/-- The area of a rhombus with side length 10 and angle 60 degrees between sides is 50√3 -/
theorem rhombus_area (side_length : ℝ) (angle : ℝ) :
  side_length = 10 →
  angle = 60 * π / 180 →
  side_length * side_length * Real.sin angle = 50 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l1326_132691


namespace NUMINAMATH_CALUDE_max_d_value_l1326_132692

def a (n : ℕ+) : ℕ := 100 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (m : ℕ+), ∀ (n : ℕ+), d n ≤ d m ∧ d m = 401 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1326_132692


namespace NUMINAMATH_CALUDE_system_solution_l1326_132610

theorem system_solution (x y k : ℝ) : 
  x - y = k - 3 →
  3 * x + 5 * y = 2 * k + 8 →
  x + y = 2 →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1326_132610


namespace NUMINAMATH_CALUDE_gift_shop_combinations_l1326_132651

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 5

/-- The number of gift card types -/
def gift_card_types : ℕ := 6

/-- The number of required ribbon colors (silver and gold) -/
def required_ribbon_colors : ℕ := 2

/-- The total number of possible combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * required_ribbon_colors * gift_card_types

theorem gift_shop_combinations :
  total_combinations = 120 :=
by sorry

end NUMINAMATH_CALUDE_gift_shop_combinations_l1326_132651
