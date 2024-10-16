import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l1033_103375

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arith_prop : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The main theorem -/
theorem arithmetic_sequence_m_value (seq : ArithmeticSequence) (m : ℕ) 
    (h1 : seq.S (m - 1) = -2)
    (h2 : seq.S m = 0)
    (h3 : seq.S (m + 1) = 3) :
    m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l1033_103375


namespace NUMINAMATH_CALUDE_xiaolis_estimate_l1033_103355

theorem xiaolis_estimate (p q a b : ℝ) 
  (h1 : p > q) (h2 : q > 0) (h3 : a > b) (h4 : b > 0) : 
  (p + a) - (q + b) > p - q := by
  sorry

end NUMINAMATH_CALUDE_xiaolis_estimate_l1033_103355


namespace NUMINAMATH_CALUDE_integer_solution_fifth_power_minus_three_times_square_l1033_103317

theorem integer_solution_fifth_power_minus_three_times_square : ∃ x : ℤ, x^5 - 3*x^2 = 216 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_fifth_power_minus_three_times_square_l1033_103317


namespace NUMINAMATH_CALUDE_distance_A_to_C_l1033_103305

/-- Prove the distance between cities A and C given travel conditions -/
theorem distance_A_to_C (time_E time_F : ℝ) (distance_AB : ℝ) (speed_ratio : ℝ) :
  time_E = 3 →
  time_F = 4 →
  distance_AB = 900 →
  speed_ratio = 4 →
  let speed_E := distance_AB / time_E
  let speed_F := speed_E / speed_ratio
  distance_AB / time_E = 4 * (distance_AB / time_E / speed_ratio) →
  speed_F * time_F = 300 :=
by sorry

end NUMINAMATH_CALUDE_distance_A_to_C_l1033_103305


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1033_103327

/-- The number of schools -/
def num_schools : ℕ := 3

/-- The number of members per school -/
def members_per_school : ℕ := 6

/-- The number of representatives from the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives from each non-host school -/
def non_host_representatives : ℕ := 1

/-- The total number of ways to arrange the presidency meeting -/
def total_arrangements : ℕ := num_schools * (Nat.choose members_per_school host_representatives) * (Nat.choose members_per_school non_host_representatives)^2

theorem presidency_meeting_arrangements :
  total_arrangements = 2160 :=
sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1033_103327


namespace NUMINAMATH_CALUDE_book_arrangements_eq_126_l1033_103330

/-- The number of ways to arrange 4 indistinguishable objects and 5 other indistinguishable objects in a row of 9 positions -/
def book_arrangements : ℕ := Nat.choose 9 4

/-- Theorem stating that the number of book arrangements is 126 -/
theorem book_arrangements_eq_126 : book_arrangements = 126 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_eq_126_l1033_103330


namespace NUMINAMATH_CALUDE_inequality_solution_l1033_103307

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_increasing : ∀ x y, x < y → f x < f y
axiom f_point1 : f 0 = -2
axiom f_point2 : f 3 = 2

-- Define the solution set
def solution_set : Set ℝ := {x | x < -1 ∨ x ≥ 2}

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | |f (x + 1)| ≥ 2} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1033_103307


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1033_103384

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_remainder (a₁ d aₙ : ℕ) (h1 : a₁ = 3) (h2 : d = 6) (h3 : aₙ = 309) :
  arithmetic_sequence_sum a₁ d aₙ % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1033_103384


namespace NUMINAMATH_CALUDE_integer_root_implies_a_value_l1033_103376

theorem integer_root_implies_a_value (a : ℕ) : 
  (∃ x : ℤ, a^2 * x^2 - (3 * a^2 - 8 * a) * x + 2 * a^2 - 13 * a + 15 = 0) →
  (a = 1 ∨ a = 3 ∨ a = 5) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_implies_a_value_l1033_103376


namespace NUMINAMATH_CALUDE_complex_modulus_l1033_103328

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1033_103328


namespace NUMINAMATH_CALUDE_tank_dimension_l1033_103331

theorem tank_dimension (cost_per_sqft : ℝ) (total_cost : ℝ) (length : ℝ) (width : ℝ) :
  cost_per_sqft = 20 →
  total_cost = 1440 →
  length = 3 →
  width = 6 →
  ∃ height : ℝ, 
    height = 2 ∧ 
    total_cost = cost_per_sqft * (2 * (length * width + length * height + width * height)) :=
by sorry

end NUMINAMATH_CALUDE_tank_dimension_l1033_103331


namespace NUMINAMATH_CALUDE_employee_pay_theorem_l1033_103391

def employee_pay (total : ℚ) (x_ratio : ℚ) (z_ratio : ℚ) :
  (ℚ × ℚ × ℚ) :=
  let y := total / (1 + x_ratio + z_ratio)
  let x := x_ratio * y
  let z := z_ratio * y
  (x, y, z)

theorem employee_pay_theorem (total : ℚ) (x_ratio : ℚ) (z_ratio : ℚ) :
  let (x, y, z) := employee_pay total x_ratio z_ratio
  (x + y + z = total) ∧ (x = x_ratio * y) ∧ (z = z_ratio * y) :=
by sorry

#eval employee_pay 934 1.2 0.8

end NUMINAMATH_CALUDE_employee_pay_theorem_l1033_103391


namespace NUMINAMATH_CALUDE_wheel_of_fraction_probability_l1033_103398

/-- Represents the possible outcomes of a single spin --/
inductive SpinOutcome
  | Bankrupt
  | Thousand
  | TwoHundred
  | SevenHundred
  | FiveHundred
  | FourHundred

/-- The total number of possible outcomes for three spins --/
def totalOutcomes : ℕ := 6^3

/-- The number of ways to earn exactly $1600 in three spins --/
def favorableOutcomes : ℕ := 9

/-- The probability of earning exactly $1600 in three spins --/
def probability : ℚ := favorableOutcomes / totalOutcomes

theorem wheel_of_fraction_probability :
  probability = 1 / 24 := by sorry

end NUMINAMATH_CALUDE_wheel_of_fraction_probability_l1033_103398


namespace NUMINAMATH_CALUDE_silver_medals_count_l1033_103308

theorem silver_medals_count (total_medals gold_medals bronze_medals : ℕ) 
  (h1 : total_medals = 67)
  (h2 : gold_medals = 19)
  (h3 : bronze_medals = 16) :
  total_medals - gold_medals - bronze_medals = 32 := by
sorry

end NUMINAMATH_CALUDE_silver_medals_count_l1033_103308


namespace NUMINAMATH_CALUDE_max_gcd_sum_l1033_103322

theorem max_gcd_sum (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a < b) (h3 : b < c) (h4 : c ≤ 3000) :
  (∃ (x y z : ℕ), 1 ≤ x ∧ x < y ∧ y < z ∧ z ≤ 3000 ∧
    Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 3000) ∧
  (∀ (x y z : ℕ), 1 ≤ x → x < y → y < z → z ≤ 3000 →
    Nat.gcd x y + Nat.gcd y z + Nat.gcd z x ≤ 3000) :=
by
  sorry

end NUMINAMATH_CALUDE_max_gcd_sum_l1033_103322


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l1033_103347

theorem number_exceeding_fraction : ∃ x : ℚ, x = (3/8) * x + 30 ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l1033_103347


namespace NUMINAMATH_CALUDE_triangle_area_calculation_l1033_103313

theorem triangle_area_calculation (a b : Real) (C : Real) :
  a = 45 ∧ b = 60 ∧ C = 37 →
  abs ((1/2) * a * b * Real.sin (C * Real.pi / 180) - 812.45) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_calculation_l1033_103313


namespace NUMINAMATH_CALUDE_barons_claim_l1033_103332

/-- Define the type of weight sets -/
def WeightSet := Fin 1000 → ℕ

/-- The condition that all weights are different -/
def all_different (w : WeightSet) : Prop :=
  ∀ i j, i ≠ j → w i ≠ w j

/-- The sum of one of each weight -/
def sum_of_weights (w : WeightSet) : ℕ :=
  Finset.sum Finset.univ (λ i => w i)

/-- The uniqueness of the sum -/
def unique_sum (w : WeightSet) : Prop :=
  ∀ s : Finset (Fin 1000), s.card < 1000 → Finset.sum s (λ i => w i) ≠ sum_of_weights w

/-- The main theorem -/
theorem barons_claim :
  ∃ w : WeightSet,
    all_different w ∧
    sum_of_weights w < 2^1010 ∧
    unique_sum w :=
  sorry

end NUMINAMATH_CALUDE_barons_claim_l1033_103332


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_e_l1033_103377

open Real

theorem extremum_implies_a_equals_e (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = exp x - a * x) →
  (∃ ε > 0, ∀ h ∈ Set.Ioo (-ε) ε, f (1 + h) ≤ f 1) ∨
  (∃ ε > 0, ∀ h ∈ Set.Ioo (-ε) ε, f (1 + h) ≥ f 1) →
  a = exp 1 := by
sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_e_l1033_103377


namespace NUMINAMATH_CALUDE_sequence_problem_l1033_103369

theorem sequence_problem (a : ℕ → ℕ) (n : ℕ) :
  a 1 = 1 ∧
  (∀ k, a (k + 1) = a k + 3) ∧
  a n = 2014 →
  n = 672 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1033_103369


namespace NUMINAMATH_CALUDE_frustum_cross_section_area_l1033_103351

theorem frustum_cross_section_area 
  (S' S Q : ℝ) 
  (n m : ℝ) 
  (h1 : S' > 0) 
  (h2 : S > 0) 
  (h3 : Q > 0) 
  (h4 : n > 0) 
  (h5 : m > 0) :
  Real.sqrt Q = (n * Real.sqrt S + m * Real.sqrt S') / (n + m) := by
sorry

end NUMINAMATH_CALUDE_frustum_cross_section_area_l1033_103351


namespace NUMINAMATH_CALUDE_cheryl_strawberries_l1033_103350

theorem cheryl_strawberries (total : ℕ) (buckets : ℕ) (left_in_each : ℕ) 
  (h1 : total = 300)
  (h2 : buckets = 5)
  (h3 : left_in_each = 40) :
  total / buckets - left_in_each = 20 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_strawberries_l1033_103350


namespace NUMINAMATH_CALUDE_quadrilateral_d_coordinates_l1033_103371

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop :=
  (q.B.x - q.A.x = q.C.x - q.D.x) ∧ 
  (q.B.y - q.A.y = q.C.y - q.D.y) ∧
  (q.D.x - q.A.x = q.C.x - q.B.x) ∧
  (q.D.y - q.A.y = q.C.y - q.B.y)

theorem quadrilateral_d_coordinates :
  ∀ (q : Quadrilateral),
    q.A = Point.mk (-1) (-2) →
    q.B = Point.mk 3 1 →
    q.C = Point.mk 0 2 →
    isParallelogram q →
    q.D = Point.mk (-4) (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_d_coordinates_l1033_103371


namespace NUMINAMATH_CALUDE_rachel_milk_consumption_l1033_103309

theorem rachel_milk_consumption (don_milk : ℚ) (rachel_fraction : ℚ) : 
  don_milk = 1/5 → rachel_fraction = 2/3 → rachel_fraction * don_milk = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_rachel_milk_consumption_l1033_103309


namespace NUMINAMATH_CALUDE_prism_volume_l1033_103374

/-- The volume of a right rectangular prism with face areas 30, 50, and 75 -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1033_103374


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1033_103324

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -8
  let c : ℝ := 5
  let x₁ : ℝ := 2 + Real.sqrt 6 / 2
  let x₂ : ℝ := 2 - Real.sqrt 6 / 2
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1033_103324


namespace NUMINAMATH_CALUDE_snow_probability_in_week_l1033_103354

theorem snow_probability_in_week (p1 p2 : ℝ) : 
  p1 = 1/2 → p2 = 1/3 → 
  (1 - (1 - p1)^4 * (1 - p2)^3) = 53/54 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_in_week_l1033_103354


namespace NUMINAMATH_CALUDE_base_conversion_equality_l1033_103345

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number in base 10 -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_equality : 
  toBase7 ((107 + 93) - 47) = [3, 0, 6] :=
sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l1033_103345


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1033_103306

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1033_103306


namespace NUMINAMATH_CALUDE_puppies_per_cage_l1033_103323

theorem puppies_per_cage (initial_puppies : Nat) (sold_puppies : Nat) (num_cages : Nat)
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : num_cages = 3)
  (h4 : initial_puppies > sold_puppies) :
  (initial_puppies - sold_puppies) / num_cages = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l1033_103323


namespace NUMINAMATH_CALUDE_nickel_ate_two_chocolates_l1033_103353

-- Define the number of chocolates Robert ate
def robert_chocolates : ℕ := 9

-- Define the difference between Robert's and Nickel's chocolates
def chocolate_difference : ℕ := 7

-- Define Nickel's chocolates
def nickel_chocolates : ℕ := robert_chocolates - chocolate_difference

-- Theorem to prove
theorem nickel_ate_two_chocolates : nickel_chocolates = 2 := by
  sorry

end NUMINAMATH_CALUDE_nickel_ate_two_chocolates_l1033_103353


namespace NUMINAMATH_CALUDE_locus_of_Q_l1033_103361

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/24 + y^2/16 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x/12 + y/8 = 1

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the intersection point R of OP and ellipse C
def point_R (x y : ℝ) : Prop := ellipse_C x y ∧ ∃ t : ℝ, x = t * (x - 0) ∧ y = t * (y - 0)

-- Define point Q on OP satisfying |OQ| * |OP| = |OR|^2
def point_Q (x y : ℝ) (xp yp xr yr : ℝ) : Prop :=
  ∃ t : ℝ, x = t * xp ∧ y = t * yp ∧ 
  (x^2 + y^2) * (xp^2 + yp^2) = (xr^2 + yr^2)^2

-- Theorem statement
theorem locus_of_Q (x y : ℝ) : 
  (∃ xp yp xr yr : ℝ, 
    point_P xp yp ∧ 
    point_R xr yr ∧ 
    point_Q x y xp yp xr yr) → 
  (x - 1)^2 / (5/2) + (y - 1)^2 / (5/3) = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_Q_l1033_103361


namespace NUMINAMATH_CALUDE_projection_theorem_l1033_103367

/-- Given vectors a and b in R², prove that the projection of a onto b is -3/5 -/
theorem projection_theorem (a b : ℝ × ℝ) : 
  b = (3, 4) → (a.1 * b.1 + a.2 * b.2 = -3) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l1033_103367


namespace NUMINAMATH_CALUDE_apple_cost_l1033_103383

/-- The cost of an item given the amount paid and change received -/
def itemCost (amountPaid changeReceived : ℚ) : ℚ :=
  amountPaid - changeReceived

/-- Proof that the apple costs $0.75 -/
theorem apple_cost (amountPaid changeReceived : ℚ) 
  (h1 : amountPaid = 5)
  (h2 : changeReceived = 4.25) : 
  itemCost amountPaid changeReceived = 0.75 := by
  sorry

#check apple_cost

end NUMINAMATH_CALUDE_apple_cost_l1033_103383


namespace NUMINAMATH_CALUDE_largest_c_for_3_in_range_l1033_103300

def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

theorem largest_c_for_3_in_range : 
  (∃ (c : ℝ), ∀ (c' : ℝ), 
    (∃ (x : ℝ), f c' x = 3) → c' ≤ c ∧ 
    (∃ (x : ℝ), f c x = 3) ∧
    c = 12) := by sorry

end NUMINAMATH_CALUDE_largest_c_for_3_in_range_l1033_103300


namespace NUMINAMATH_CALUDE_sum_fourth_fifth_sixth_l1033_103340

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_second_third : a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem sum_fourth_fifth_sixth (seq : ArithmeticSequence) :
  seq.a 4 + seq.a 5 + seq.a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_fourth_fifth_sixth_l1033_103340


namespace NUMINAMATH_CALUDE_cork_price_calculation_l1033_103336

/-- The price of a bottle of wine with a cork -/
def bottle_with_cork : ℚ := 2.10

/-- The additional cost of a bottle without a cork compared to the cork price -/
def additional_cost : ℚ := 2.00

/-- The price of the cork -/
def cork_price : ℚ := 0.05

theorem cork_price_calculation :
  cork_price + (cork_price + additional_cost) = bottle_with_cork :=
by sorry

end NUMINAMATH_CALUDE_cork_price_calculation_l1033_103336


namespace NUMINAMATH_CALUDE_phone_call_duration_l1033_103343

/-- Calculates the duration of a phone call given the initial credit, cost per minute, and remaining credit -/
theorem phone_call_duration (initial_credit remaining_credit cost_per_minute : ℚ) : 
  initial_credit = 30 ∧ 
  cost_per_minute = 16/100 ∧ 
  remaining_credit = 264/10 →
  (initial_credit - remaining_credit) / cost_per_minute = 22 := by
  sorry

end NUMINAMATH_CALUDE_phone_call_duration_l1033_103343


namespace NUMINAMATH_CALUDE_marquita_garden_length_l1033_103303

-- Define the number of gardens for each person
def mancino_gardens : ℕ := 3
def marquita_gardens : ℕ := 2

-- Define the dimensions of Mancino's gardens
def mancino_garden_length : ℕ := 16
def mancino_garden_width : ℕ := 5

-- Define the width of Marquita's gardens
def marquita_garden_width : ℕ := 4

-- Define the total area of all gardens
def total_area : ℕ := 304

-- Theorem to prove
theorem marquita_garden_length :
  ∃ (l : ℕ), 
    mancino_gardens * mancino_garden_length * mancino_garden_width +
    marquita_gardens * l * marquita_garden_width = total_area ∧
    l = 8 := by
  sorry

end NUMINAMATH_CALUDE_marquita_garden_length_l1033_103303


namespace NUMINAMATH_CALUDE_sum_and_average_of_squares_of_multiples_of_7_l1033_103311

def multiples_of_7 (n : ℕ) : List ℕ :=
  List.range n |>.map (· * 7 + 7)

def sum_of_squares (lst : List ℕ) : ℕ :=
  lst.map (· ^ 2) |>.sum

theorem sum_and_average_of_squares_of_multiples_of_7 :
  let lst := multiples_of_7 10
  let sum := sum_of_squares lst
  let avg := (sum : ℚ) / 10
  sum = 16865 ∧ avg = 1686.5 := by sorry

end NUMINAMATH_CALUDE_sum_and_average_of_squares_of_multiples_of_7_l1033_103311


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_theorem_l1033_103326

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  /-- The altitude to the base -/
  altitude : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The triangle is isosceles -/
  is_isosceles : True

/-- Calculate the area of an isosceles triangle -/
def triangle_area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area_theorem :
  ∀ (t : IsoscelesTriangle), t.altitude = 10 ∧ t.perimeter = 40 → triangle_area t = 75 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_theorem_l1033_103326


namespace NUMINAMATH_CALUDE_point_movement_l1033_103394

/-- Given a point P in a 2D Cartesian coordinate system, moving it right and down
    results in a new point Q with the expected coordinates. -/
theorem point_movement (P : ℝ × ℝ) (right down : ℝ) (Q : ℝ × ℝ) :
  P = (-1, 2) →
  right = 2 →
  down = 3 →
  Q.1 = P.1 + right →
  Q.2 = P.2 - down →
  Q = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l1033_103394


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l1033_103373

/-- An arithmetic sequence with a_1 = 3 and a_2 + a_3 = 12 has a_2 = 5 -/
theorem arithmetic_sequence_a2 (a : ℕ → ℝ) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 = 3 →                                 -- a_1 = 3
  a 2 + a 3 = 12 →                          -- a_2 + a_3 = 12
  a 2 = 5 :=                                -- conclusion: a_2 = 5
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l1033_103373


namespace NUMINAMATH_CALUDE_third_student_weight_l1033_103344

theorem third_student_weight (original_count : ℕ) (original_avg : ℝ) 
  (new_count : ℕ) (new_avg : ℝ) (first_weight : ℝ) (second_weight : ℝ) :
  original_count = 29 →
  original_avg = 28 →
  new_count = original_count + 3 →
  new_avg = 27.3 →
  first_weight = 20 →
  second_weight = 30 →
  ∃ (third_weight : ℝ),
    third_weight = new_count * new_avg - original_count * original_avg - first_weight - second_weight ∧
    third_weight = 11.6 := by
  sorry

end NUMINAMATH_CALUDE_third_student_weight_l1033_103344


namespace NUMINAMATH_CALUDE_expression_evaluation_l1033_103399

theorem expression_evaluation (a : ℝ) : 
  let x : ℝ := 2*a + 6
  (x - 2*a + 4) = 10 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1033_103399


namespace NUMINAMATH_CALUDE_smallest_section_area_l1033_103388

/-- The area of the smallest circular section of a sphere circumscribed around a cube --/
theorem smallest_section_area (cube_edge : ℝ) (h : cube_edge = 4) : 
  let sphere_radius : ℝ := cube_edge * Real.sqrt 3 / 2
  let midpoint_to_center : ℝ := cube_edge * Real.sqrt 2 / 2
  let section_radius : ℝ := Real.sqrt (sphere_radius^2 - midpoint_to_center^2)
  π * section_radius^2 = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_smallest_section_area_l1033_103388


namespace NUMINAMATH_CALUDE_nancy_folders_l1033_103310

theorem nancy_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 43 → 
  deleted_files = 31 → 
  files_per_folder = 6 → 
  (initial_files - deleted_files) / files_per_folder = 2 := by
  sorry

end NUMINAMATH_CALUDE_nancy_folders_l1033_103310


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1033_103387

theorem complex_number_quadrant (z : ℂ) (h : z * (2 + Complex.I) = 3 - Complex.I) :
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1033_103387


namespace NUMINAMATH_CALUDE_cindys_calculation_l1033_103302

theorem cindys_calculation (x : ℤ) : (x - 7) / 5 = 37 → (x - 5) / 7 = 26 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l1033_103302


namespace NUMINAMATH_CALUDE_greatest_four_digit_number_with_remainders_l1033_103337

theorem greatest_four_digit_number_with_remainders :
  ∃ n : ℕ,
    n ≤ 9999 ∧
    n > 999 ∧
    n % 15 = 2 ∧
    n % 24 = 8 ∧
    (∀ m : ℕ, m ≤ 9999 ∧ m > 999 ∧ m % 15 = 2 → m ≤ n) ∧
    n = 9992 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_number_with_remainders_l1033_103337


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l1033_103397

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  (n / 2) * (a₁ + aₙ)

theorem arithmetic_sequences_ratio : 
  (arithmetic_sum 4 4 72) / (arithmetic_sum 5 5 90) = 76 / 95 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l1033_103397


namespace NUMINAMATH_CALUDE_find_x_l1033_103335

theorem find_x : ∃ x : ℚ, (3 * x - 5) / 7 = 15 ∧ x = 110 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1033_103335


namespace NUMINAMATH_CALUDE_extreme_values_and_range_of_a_l1033_103359

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * x^2 - x

theorem extreme_values_and_range_of_a :
  (∀ x : ℝ, f (1/4) x ≤ f (1/4) 0) ∧
  (∀ x : ℝ, f (1/4) x ≥ f (1/4) 1) ∧
  (f (1/4) 0 = 0) ∧
  (f (1/4) 1 = Real.log 2 - 3/4) ∧
  (∀ a : ℝ, (∀ b : ℝ, 1 < b → b < 2 → 
    (∀ x : ℝ, -1 < x → x ≤ b → f a x ≤ f a b)) →
    a ≥ 1 - Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_range_of_a_l1033_103359


namespace NUMINAMATH_CALUDE_science_study_time_l1033_103318

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The total time Sam spends studying in hours -/
def total_study_time_hours : ℕ := 3

/-- The time Sam spends studying Math in minutes -/
def math_study_time : ℕ := 80

/-- The time Sam spends studying Literature in minutes -/
def literature_study_time : ℕ := 40

/-- Theorem: Sam spends 60 minutes studying Science -/
theorem science_study_time : ℕ := by
  sorry

end NUMINAMATH_CALUDE_science_study_time_l1033_103318


namespace NUMINAMATH_CALUDE_inequality_proof_l1033_103333

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  1 / (a^2 + b^2) ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1033_103333


namespace NUMINAMATH_CALUDE_smallest_angle_SQR_l1033_103370

-- Define the angles
def angle_PQR : ℝ := 40
def angle_PQS : ℝ := 28

-- Define the theorem
theorem smallest_angle_SQR : 
  let angle_SQR := angle_PQR - angle_PQS
  angle_SQR = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_SQR_l1033_103370


namespace NUMINAMATH_CALUDE_triangle_problem_l1033_103348

/-- Given a triangle ABC with circumradius 1 and the relation between sides, prove the value of a and the area when b = 1. -/
theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (2 * Real.sin A) = 1 ∧  -- circumradius = 1
  b = a * Real.cos C - (Real.sqrt 3 / 6) * a * c →
  -- Conclusions
  a = Real.sqrt 3 ∧
  (b = 1 → Real.sqrt 3 / 4 = 1/2 * b * c * Real.sin A) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1033_103348


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1033_103319

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1033_103319


namespace NUMINAMATH_CALUDE_f_positive_on_interval_l1033_103360

open Real

noncomputable def f (a x : ℝ) : ℝ := a * log x - x - a / x + 2 * a

theorem f_positive_on_interval (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), f a x > 0) ↔ a > (exp 2) / (3 * exp 1 - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_positive_on_interval_l1033_103360


namespace NUMINAMATH_CALUDE_classroom_notebooks_l1033_103316

theorem classroom_notebooks (total_students : ℕ) 
  (h1 : total_students = 28)
  (group1_notebooks : ℕ) (h2 : group1_notebooks = 5)
  (group2_notebooks : ℕ) (h3 : group2_notebooks = 3)
  (group3_notebooks : ℕ) (h4 : group3_notebooks = 7) :
  (total_students / 3) * group1_notebooks +
  (total_students / 3) * group2_notebooks +
  (total_students - 2 * (total_students / 3)) * group3_notebooks = 142 :=
by sorry

end NUMINAMATH_CALUDE_classroom_notebooks_l1033_103316


namespace NUMINAMATH_CALUDE_journey_time_change_l1033_103363

/-- Proves that for a journey of 40 km, if increasing the speed by 3 kmph reduces
    the time by 40 minutes, then decreasing the speed by 2 kmph from the original
    speed increases the time by 40 minutes. -/
theorem journey_time_change (v : ℝ) (h1 : v > 0) : 
  (40 / v - 40 / (v + 3) = 2 / 3) → 
  (40 / (v - 2) - 40 / v = 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_journey_time_change_l1033_103363


namespace NUMINAMATH_CALUDE_tomatoes_picked_l1033_103381

theorem tomatoes_picked (initial_tomatoes : ℕ) (initial_potatoes : ℕ) (final_total : ℕ) : 
  initial_tomatoes = 177 → 
  initial_potatoes = 12 → 
  final_total = 136 → 
  initial_tomatoes - (final_total - initial_potatoes) = 53 := by
sorry

end NUMINAMATH_CALUDE_tomatoes_picked_l1033_103381


namespace NUMINAMATH_CALUDE_nine_nines_squared_zeros_l1033_103364

/-- The number of nines in 9,999,999 -/
def n : ℕ := 7

/-- The number 9,999,999 -/
def x : ℕ := 10^n - 1

/-- The number of zeros at the end of x^2 -/
def num_zeros (x : ℕ) : ℕ := n - 1

theorem nine_nines_squared_zeros :
  num_zeros x = 6 :=
sorry

end NUMINAMATH_CALUDE_nine_nines_squared_zeros_l1033_103364


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1033_103358

def p (x y : ℝ) : Prop := x > 1 ∨ y > 2

def q (x y : ℝ) : Prop := x + y > 3

theorem p_necessary_not_sufficient :
  (∀ x y : ℝ, q x y → p x y) ∧ 
  (∃ x y : ℝ, p x y ∧ ¬(q x y)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1033_103358


namespace NUMINAMATH_CALUDE_person_a_age_l1033_103349

/-- The ages of two people, A and B, satisfy certain conditions. -/
structure AgeProblem where
  /-- Age of Person A this year -/
  a : ℕ
  /-- Age of Person B this year -/
  b : ℕ
  /-- The sum of their ages this year is 43 -/
  sum_constraint : a + b = 43
  /-- In 4 years, A will be 3 years older than B -/
  future_constraint : a + 4 = (b + 4) + 3

/-- Given the age constraints, Person A's age this year is 23 -/
theorem person_a_age (p : AgeProblem) : p.a = 23 := by
  sorry

end NUMINAMATH_CALUDE_person_a_age_l1033_103349


namespace NUMINAMATH_CALUDE_sqrt_500_simplification_l1033_103314

theorem sqrt_500_simplification : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_500_simplification_l1033_103314


namespace NUMINAMATH_CALUDE_mod_pow_diff_zero_l1033_103380

theorem mod_pow_diff_zero (n : ℕ) : (51 ^ n - 27 ^ n) % 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_pow_diff_zero_l1033_103380


namespace NUMINAMATH_CALUDE_custom_op_seven_five_l1033_103378

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a * b : ℚ) / ((a : ℤ) - (b : ℤ) + 8 : ℚ)

/-- Theorem stating that 7 @ 5 = 7/2 -/
theorem custom_op_seven_five :
  custom_op 7 5 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_seven_five_l1033_103378


namespace NUMINAMATH_CALUDE_gcd_75_360_l1033_103341

theorem gcd_75_360 : Nat.gcd 75 360 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_75_360_l1033_103341


namespace NUMINAMATH_CALUDE_find_c_l1033_103385

def f (x : ℝ) : ℝ := x - 2

def F (x y : ℝ) : ℝ := y^2 + x

theorem find_c (b : ℝ) : ∃ c : ℝ, c = F 3 (f b) ∧ c = 199 := by
  sorry

end NUMINAMATH_CALUDE_find_c_l1033_103385


namespace NUMINAMATH_CALUDE_point_on_ln_curve_with_specific_tangent_l1033_103379

open Real

/-- Proves that a point on y = ln(x) with tangent line through (-e, -1) has coordinates (e, 1) -/
theorem point_on_ln_curve_with_specific_tangent (x₀ : ℝ) :
  (∃ (A : ℝ × ℝ), 
    A.1 = x₀ ∧ 
    A.2 = log x₀ ∧ 
    (log x₀ - (-1)) / (x₀ - (-Real.exp 1)) = 1 / x₀) →
  x₀ = Real.exp 1 ∧ log x₀ = 1 := by
  sorry


end NUMINAMATH_CALUDE_point_on_ln_curve_with_specific_tangent_l1033_103379


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l1033_103338

theorem part_to_whole_ratio (N P : ℝ) 
  (h1 : (1/4) * (1/3) * P = 15) 
  (h2 : 0.40 * N = 180) : 
  P / N = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l1033_103338


namespace NUMINAMATH_CALUDE_painting_efficiency_theorem_l1033_103315

/-- Represents the efficiency of painting classrooms -/
structure PaintingEfficiency where
  groups : ℕ
  workers_per_group : ℕ
  hours_per_day : ℕ
  classrooms : ℕ
  days : ℚ

/-- The initial painting scenario -/
def initial_scenario : PaintingEfficiency :=
  { groups := 6
  , workers_per_group := 6
  , hours_per_day := 6
  , classrooms := 6
  , days := 6 }

/-- The new painting scenario -/
def new_scenario : PaintingEfficiency :=
  { groups := 8
  , workers_per_group := 8
  , hours_per_day := 8
  , classrooms := 8
  , days := 27/8 }

/-- Calculates the painting rate (classrooms per worker-hour) -/
def painting_rate (p : PaintingEfficiency) : ℚ :=
  p.classrooms / (p.groups * p.workers_per_group * p.hours_per_day * p.days)

theorem painting_efficiency_theorem :
  painting_rate initial_scenario = painting_rate new_scenario := by
  sorry

#check painting_efficiency_theorem

end NUMINAMATH_CALUDE_painting_efficiency_theorem_l1033_103315


namespace NUMINAMATH_CALUDE_luke_paula_commute_l1033_103312

/-- The problem of Luke and Paula's commute times -/
theorem luke_paula_commute :
  -- Luke's bus time to work
  ∀ (luke_bus : ℕ),
  -- Paula's bus time as a fraction of Luke's
  ∀ (paula_fraction : ℚ),
  -- Total travel time for both
  ∀ (total_time : ℕ),
  -- Conditions
  luke_bus = 70 →
  paula_fraction = 3/5 →
  total_time = 504 →
  -- Conclusion
  ∃ (bike_multiple : ℚ),
    -- Luke's bike time = bus time * bike_multiple
    (luke_bus * bike_multiple).floor +
    -- Luke's bus time to work
    luke_bus +
    -- Paula's bus time to work
    (paula_fraction * luke_bus).floor +
    -- Paula's bus time back home
    (paula_fraction * luke_bus).floor = total_time ∧
    bike_multiple = 5 :=
by sorry

end NUMINAMATH_CALUDE_luke_paula_commute_l1033_103312


namespace NUMINAMATH_CALUDE_graduating_class_boys_count_l1033_103334

theorem graduating_class_boys_count (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 466 → diff = 212 → boys + (boys + diff) = total → boys = 127 := by
  sorry

end NUMINAMATH_CALUDE_graduating_class_boys_count_l1033_103334


namespace NUMINAMATH_CALUDE_largest_m_is_138_l1033_103389

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_valid_pair (x y : ℕ) : Prop :=
  x < 15 ∧ y < 15 ∧ x ≠ y ∧ is_prime x ∧ is_prime y ∧ is_prime (x + y) ∧ is_prime (10 * x + y)

def m (x y : ℕ) : ℕ := x * y * (10 * x + y)

theorem largest_m_is_138 :
  ∀ x y : ℕ, is_valid_pair x y → m x y ≤ 138 :=
sorry

end NUMINAMATH_CALUDE_largest_m_is_138_l1033_103389


namespace NUMINAMATH_CALUDE_total_bottles_is_255_l1033_103368

-- Define the number of bottles in each box
def boxA_water : ℕ := 24
def boxA_orange : ℕ := 21
def boxA_apple : ℕ := boxA_water + 6

def boxB_water : ℕ := boxA_water + boxA_water / 4
def boxB_orange : ℕ := boxA_orange - boxA_orange * 3 / 10
def boxB_apple : ℕ := boxA_apple

def boxC_water : ℕ := 2 * boxB_water
def boxC_apple : ℕ := (3 * boxB_apple) / 2
def boxC_orange : ℕ := 0

-- Define the total number of bottles
def total_bottles : ℕ := 
  boxA_water + boxA_orange + boxA_apple + 
  boxB_water + boxB_orange + boxB_apple + 
  boxC_water + boxC_orange + boxC_apple

-- Theorem to prove
theorem total_bottles_is_255 : total_bottles = 255 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_is_255_l1033_103368


namespace NUMINAMATH_CALUDE_dirichlet_approximation_l1033_103395

theorem dirichlet_approximation (N : ℕ) (hN : N > 0) :
  ∃ (a b : ℕ), 1 ≤ b ∧ b ≤ N ∧ |a - b * Real.sqrt 2| ≤ 1 / N :=
by sorry

end NUMINAMATH_CALUDE_dirichlet_approximation_l1033_103395


namespace NUMINAMATH_CALUDE_inequality_proof_l1033_103386

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1033_103386


namespace NUMINAMATH_CALUDE_root_square_value_l1033_103342

theorem root_square_value (x₁ x₂ : ℂ) : 
  x₁ ≠ x₂ →
  (x₁ - 1)^2 = -3 →
  (x₂ - 1)^2 = -3 →
  x₁ = 1 - Complex.I * Real.sqrt 3 →
  x₂^2 = -2 + 2 * Complex.I * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_root_square_value_l1033_103342


namespace NUMINAMATH_CALUDE_correct_multiplication_l1033_103362

theorem correct_multiplication (x : ℕ) : 
  987 * x = 559981 → 987 * x = 559989 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplication_l1033_103362


namespace NUMINAMATH_CALUDE_x_value_l1033_103357

theorem x_value : ∃ x : ℝ, (3 * x = (16 - x) + 4) ∧ (x = 5) := by sorry

end NUMINAMATH_CALUDE_x_value_l1033_103357


namespace NUMINAMATH_CALUDE_function_symmetry_l1033_103325

/-- Given a function f(x) = ax + b*sin(x) + 1 where f(2017) = 7, prove that f(-2017) = -5 -/
theorem function_symmetry (a b : ℝ) :
  (let f := fun x => a * x + b * Real.sin x + 1
   f 2017 = 7) →
  (fun x => a * x + b * Real.sin x + 1) (-2017) = -5 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l1033_103325


namespace NUMINAMATH_CALUDE_inequality_proof_l1033_103392

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1033_103392


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1033_103301

-- Problem 1
theorem problem_one : -9 + 5 - (-12) + (-3) = 5 := by
  sorry

-- Problem 2
theorem problem_two : -(1.5) - (-4.25) + 3.75 - 8.5 = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1033_103301


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1033_103393

theorem diophantine_equation_solutions : 
  {(x, y) : ℕ × ℕ | 2 * x^2 + 2 * x * y - x + y = 2020} = {(0, 2020), (1, 673)} := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1033_103393


namespace NUMINAMATH_CALUDE_seventh_term_is_eight_l1033_103365

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, sum n = (n : ℝ) * (a 1 + a n) / 2
  third_eighth_sum : a 3 + a 8 = 13
  seventh_sum : sum 7 = 35

/-- The main theorem stating that the 7th term of the sequence is 8 -/
theorem seventh_term_is_eight (seq : ArithmeticSequence) : seq.a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_eight_l1033_103365


namespace NUMINAMATH_CALUDE_diagonal_passes_900_cubes_l1033_103320

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: An internal diagonal of a 200 × 400 × 500 rectangular solid
    passes through 900 cubes -/
theorem diagonal_passes_900_cubes :
  cubes_passed 200 400 500 = 900 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_900_cubes_l1033_103320


namespace NUMINAMATH_CALUDE_equality_of_fractions_implies_equality_of_products_l1033_103304

theorem equality_of_fractions_implies_equality_of_products 
  (x y z t : ℝ) (h : (x + y) / (y + z) = (z + t) / (t + x)) : 
  x * (z + t + y) = z * (x + y + t) := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_implies_equality_of_products_l1033_103304


namespace NUMINAMATH_CALUDE_halloween_candy_weight_l1033_103321

/-- The combined weight of candy for Frank and Gwen -/
theorem halloween_candy_weight (frank_candy : ℕ) (gwen_candy : ℕ) 
  (h1 : frank_candy = 10) (h2 : gwen_candy = 7) : 
  frank_candy + gwen_candy = 17 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_weight_l1033_103321


namespace NUMINAMATH_CALUDE_dart_game_equations_correct_l1033_103382

/-- Represents the dart throwing game scenario -/
structure DartGame where
  x : ℕ  -- number of times Xiao Hua hits the target
  y : ℕ  -- number of times the father hits the target

/-- The conditions of the dart throwing game -/
def validGame (game : DartGame) : Prop :=
  game.x + game.y = 30 ∧  -- total number of hits
  5 * game.x + 2 = 3 * game.y  -- score difference condition

/-- Theorem stating that the system of equations correctly represents the game -/
theorem dart_game_equations_correct (game : DartGame) :
  validGame game ↔ 
    (game.x + game.y = 30 ∧ 5 * game.x + 2 = 3 * game.y) :=
by sorry

end NUMINAMATH_CALUDE_dart_game_equations_correct_l1033_103382


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1033_103356

theorem expression_simplification_and_evaluation :
  ∀ a : ℚ, a + 1 ≠ 0 → a + 2 ≠ 0 →
  (a + 1 - (5 + 2*a) / (a + 1)) / ((a^2 + 4*a + 4) / (a + 1)) = (a - 2) / (a + 2) ∧
  (let simplified := (a - 2) / (a + 2);
   a = -3 → simplified = 5) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1033_103356


namespace NUMINAMATH_CALUDE_value_of_a_l1033_103390

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 9) 
  (eq3 : c = 4) : 
  a = -1 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l1033_103390


namespace NUMINAMATH_CALUDE_fraction_decimal_places_l1033_103352

/-- The number of decimal places when converting the fraction 123456789 / (2^26 * 5^4) to a decimal -/
def decimal_places : ℕ :=
  let numerator : ℕ := 123456789
  let denominator : ℕ := 2^26 * 5^4
  26

theorem fraction_decimal_places :
  decimal_places = 26 :=
sorry

end NUMINAMATH_CALUDE_fraction_decimal_places_l1033_103352


namespace NUMINAMATH_CALUDE_fraction_equality_l1033_103329

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x - 2 * y) / (2 * x + 5 * y) = 3) : 
  (2 * x - 5 * y) / (x + 2 * y) = 13 / 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1033_103329


namespace NUMINAMATH_CALUDE_average_of_combined_results_l1033_103366

theorem average_of_combined_results (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 45) (h₂ : n₂ = 25) (h₃ : avg₁ = 25) (h₄ : avg₂ = 45) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = 2250 / 70 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l1033_103366


namespace NUMINAMATH_CALUDE_orchids_planted_tomorrow_l1033_103346

/-- Proves that the number of orchid bushes to be planted tomorrow is 25 --/
theorem orchids_planted_tomorrow
  (initial : ℕ) -- Initial number of orchid bushes
  (planted_today : ℕ) -- Number of orchid bushes planted today
  (final : ℕ) -- Final number of orchid bushes
  (h1 : initial = 47)
  (h2 : planted_today = 37)
  (h3 : final = 109) :
  final - (initial + planted_today) = 25 := by
  sorry

#check orchids_planted_tomorrow

end NUMINAMATH_CALUDE_orchids_planted_tomorrow_l1033_103346


namespace NUMINAMATH_CALUDE_complementary_not_supplementary_l1033_103339

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- Two angles are supplementary if their sum is 180 degrees -/
def supplementary (a b : ℝ) : Prop := a + b = 180

/-- Theorem: It is impossible for two angles to be both complementary and supplementary -/
theorem complementary_not_supplementary : ¬ ∃ (a b : ℝ), complementary a b ∧ supplementary a b := by
  sorry

end NUMINAMATH_CALUDE_complementary_not_supplementary_l1033_103339


namespace NUMINAMATH_CALUDE_floor_is_linear_periodic_linear_periodic_minus_x_sin_plus_kx_linear_periodic_iff_l1033_103372

/-- Definition of a linear periodic function -/
def IsLinearPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x + T

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℝ := ⌊x⌋

/-- Theorem: The floor function is linear periodic -/
theorem floor_is_linear_periodic : IsLinearPeriodic floor := sorry

/-- Theorem: If g is linear periodic, then g(x) - x is also linear periodic -/
theorem linear_periodic_minus_x {g : ℝ → ℝ} (h : IsLinearPeriodic g) :
  IsLinearPeriodic (fun x ↦ g x - x) := sorry

/-- Theorem: sin(x) + kx is linear periodic if and only if k = 1 -/
theorem sin_plus_kx_linear_periodic_iff (k : ℝ) :
  IsLinearPeriodic (fun x ↦ Real.sin x + k * x) ↔ k = 1 := sorry

end NUMINAMATH_CALUDE_floor_is_linear_periodic_linear_periodic_minus_x_sin_plus_kx_linear_periodic_iff_l1033_103372


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l1033_103396

-- Define the # operation
def hash (a b : ℝ) : ℝ := a + 2 * b

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  ¬(∀ (x y z : ℝ), hash x (y + z) = hash x y + hash x z) ∧
  ¬(∀ (x y z : ℝ), x + hash y z = hash (x + y) (x + z)) ∧
  ¬(∀ (x y z : ℝ), hash x (hash y z) = hash (hash x y) (hash x z)) :=
sorry

end NUMINAMATH_CALUDE_no_distributive_laws_hold_l1033_103396
