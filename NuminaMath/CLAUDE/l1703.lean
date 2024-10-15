import Mathlib

namespace NUMINAMATH_CALUDE_jenny_house_worth_l1703_170339

/-- The current worth of Jenny's house -/
def house_worth : ℝ := 500000

/-- Jenny's property tax rate -/
def tax_rate : ℝ := 0.02

/-- The increase in house value due to the high-speed rail project -/
def value_increase : ℝ := 0.25

/-- The maximum amount Jenny can spend on property tax per year -/
def max_tax : ℝ := 15000

/-- The value of improvements Jenny can make to her house -/
def improvements : ℝ := 250000

theorem jenny_house_worth :
  tax_rate * (house_worth * (1 + value_increase) + improvements) = max_tax := by
  sorry

#check jenny_house_worth

end NUMINAMATH_CALUDE_jenny_house_worth_l1703_170339


namespace NUMINAMATH_CALUDE_scientific_notation_of_132000000_l1703_170329

theorem scientific_notation_of_132000000 :
  (132000000 : ℝ) = 1.32 * (10 : ℝ) ^ 8 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_132000000_l1703_170329


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l1703_170348

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 1}

-- State the theorem
theorem set_operations_and_subset :
  (A ∩ B = {x | 2 < x ∧ x < 6}) ∧
  (A ∪ (Set.univ \ B) = {x | x < 6 ∨ 9 ≤ x}) ∧
  (∀ a : ℝ, C a ⊆ A → a ≤ 5/2) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l1703_170348


namespace NUMINAMATH_CALUDE_min_disks_needed_l1703_170318

/-- Represents the capacity of each disk in MB -/
def diskCapacity : ℚ := 1.44

/-- Represents the file sizes in MB -/
def fileSizes : List ℚ := [0.9, 0.6, 0.45, 0.3]

/-- Represents the quantity of each file size -/
def fileQuantities : List ℕ := [5, 10, 10, 5]

/-- Calculates the total storage required for all files in MB -/
def totalStorage : ℚ :=
  List.sum (List.zipWith (· * ·) (List.map (λ x => (x : ℚ)) fileQuantities) fileSizes)

/-- Theorem: The minimum number of disks needed is 15 -/
theorem min_disks_needed : 
  ∃ (n : ℕ), n = 15 ∧ 
  n * diskCapacity ≥ totalStorage ∧
  ∀ m : ℕ, m * diskCapacity ≥ totalStorage → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_disks_needed_l1703_170318


namespace NUMINAMATH_CALUDE_equality_of_sides_from_equal_angles_l1703_170337

-- Define a structure for 3D points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a function to calculate the angle between three points
def angle (p1 p2 p3 : Point3D) : ℝ := sorry

-- Define a function to calculate the distance between two points
def distance (p1 p2 : Point3D) : ℝ := sorry

-- Define a predicate to check if four points are non-coplanar
def nonCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

theorem equality_of_sides_from_equal_angles 
  (A B C D : Point3D) 
  (h1 : nonCoplanar A B C D)
  (h2 : angle A B C = angle A D C)
  (h3 : angle B A D = angle B C D) :
  distance A B = distance C D ∧ distance B C = distance A D := by
  sorry

end NUMINAMATH_CALUDE_equality_of_sides_from_equal_angles_l1703_170337


namespace NUMINAMATH_CALUDE_partition_of_positive_integers_l1703_170357

def nth_prime (n : ℕ) : ℕ := sorry

def count_primes (n : ℕ) : ℕ := sorry

def set_A : Set ℕ := {m | ∃ n : ℕ, n > 0 ∧ m = n + nth_prime n - 1}

def set_B : Set ℕ := {m | ∃ n : ℕ, n > 0 ∧ m = n + count_primes n}

theorem partition_of_positive_integers : 
  ∀ m : ℕ, m > 0 → (m ∈ set_A ∧ m ∉ set_B) ∨ (m ∉ set_A ∧ m ∈ set_B) :=
sorry

end NUMINAMATH_CALUDE_partition_of_positive_integers_l1703_170357


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l1703_170332

/-- The measure of each interior angle of a regular octagon -/
def regular_octagon_interior_angle : ℝ := 135

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def polygon_interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

theorem regular_octagon_interior_angle_measure :
  regular_octagon_interior_angle = polygon_interior_angle_sum octagon_sides / octagon_sides := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l1703_170332


namespace NUMINAMATH_CALUDE_second_meal_cost_l1703_170342

/-- The cost of a meal consisting of burgers, shakes, and cola. -/
structure MealCost where
  burger : ℝ
  shake : ℝ
  cola : ℝ

/-- The theorem stating the cost of the second meal given the costs of two other meals. -/
theorem second_meal_cost 
  (meal1 : MealCost) 
  (meal2 : MealCost) 
  (h1 : 3 * meal1.burger + 7 * meal1.shake + meal1.cola = 120)
  (h2 : meal2.burger + meal2.shake + meal2.cola = 39)
  (h3 : meal1 = meal2) :
  4 * meal1.burger + 10 * meal1.shake + meal1.cola = 160.5 := by
  sorry

end NUMINAMATH_CALUDE_second_meal_cost_l1703_170342


namespace NUMINAMATH_CALUDE_four_square_product_l1703_170345

theorem four_square_product (p q r s p₁ q₁ r₁ s₁ : ℝ) :
  ∃ A B C D : ℝ, (p^2 + q^2 + r^2 + s^2) * (p₁^2 + q₁^2 + r₁^2 + s₁^2) = A^2 + B^2 + C^2 + D^2 := by
  sorry

end NUMINAMATH_CALUDE_four_square_product_l1703_170345


namespace NUMINAMATH_CALUDE_thirteenth_most_likely_friday_l1703_170385

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents the Gregorian calendar -/
structure GregorianCalendar where
  /-- The current year in the 400-year cycle -/
  year : Nat
  /-- Whether the current year is a leap year -/
  is_leap_year : Bool
  /-- The day of the week for the 1st of January of the current year -/
  first_day : DayOfWeek

/-- Counts the occurrences of the 13th falling on each day of the week in a 400-year cycle -/
def count_13ths (calendar : GregorianCalendar) : DayOfWeek → Nat
  | _ => sorry

/-- Theorem: The 13th day of the month falls on Friday more often than on any other day
    in a complete 400-year cycle of the Gregorian calendar -/
theorem thirteenth_most_likely_friday (calendar : GregorianCalendar) :
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Friday → count_13ths calendar DayOfWeek.Friday > count_13ths calendar d := by
  sorry

#check thirteenth_most_likely_friday

end NUMINAMATH_CALUDE_thirteenth_most_likely_friday_l1703_170385


namespace NUMINAMATH_CALUDE_modulus_constraint_implies_range_l1703_170393

theorem modulus_constraint_implies_range (a : ℝ) :
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) →
  a ∈ Set.Icc (-Real.sqrt 5 / 5) (Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_modulus_constraint_implies_range_l1703_170393


namespace NUMINAMATH_CALUDE_lucky_sum_equality_l1703_170352

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to select k distinct numbers from 1 to n with sum s -/
def sumCombinations (n k s : ℕ) : ℕ := sorry

/-- The probability of selecting k balls from n balls with sum s -/
def probability (n k s : ℕ) : ℚ :=
  (sumCombinations n k s : ℚ) / (choose n k : ℚ)

theorem lucky_sum_equality (N : ℕ) :
  probability N 10 63 = probability N 8 44 ↔ N = 18 := by
  sorry

end NUMINAMATH_CALUDE_lucky_sum_equality_l1703_170352


namespace NUMINAMATH_CALUDE_pineapple_cost_proof_l1703_170379

/-- Given the cost of pineapples and shipping, prove the total cost per pineapple -/
theorem pineapple_cost_proof (pineapple_cost : ℚ) (num_pineapples : ℕ) (shipping_cost : ℚ) 
  (h1 : pineapple_cost = 5/4)  -- $1.25 represented as a rational number
  (h2 : num_pineapples = 12)
  (h3 : shipping_cost = 21) :
  (pineapple_cost * num_pineapples + shipping_cost) / num_pineapples = 3 := by
  sorry

end NUMINAMATH_CALUDE_pineapple_cost_proof_l1703_170379


namespace NUMINAMATH_CALUDE_scientific_notation_18_million_l1703_170341

theorem scientific_notation_18_million :
  (18000000 : ℝ) = 1.8 * (10 : ℝ) ^ 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_18_million_l1703_170341


namespace NUMINAMATH_CALUDE_county_population_distribution_l1703_170376

theorem county_population_distribution (less_than_10k : ℝ) (between_10k_and_100k : ℝ) :
  less_than_10k = 25 →
  between_10k_and_100k = 59 →
  less_than_10k + between_10k_and_100k = 84 :=
by sorry

end NUMINAMATH_CALUDE_county_population_distribution_l1703_170376


namespace NUMINAMATH_CALUDE_exponent_of_nine_in_nine_to_seven_l1703_170310

theorem exponent_of_nine_in_nine_to_seven (h : ∀ y : ℕ, y > 14 → ¬(3^y ∣ 9^7)) :
  ∃ n : ℕ, 9^7 = 9^n ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_exponent_of_nine_in_nine_to_seven_l1703_170310


namespace NUMINAMATH_CALUDE_parabola_point_ordinate_l1703_170353

/-- Represents a parabola y = ax^2 with a > 0 -/
structure Parabola where
  a : ℝ
  a_pos : a > 0

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.a * x^2

theorem parabola_point_ordinate (p : Parabola) (M : PointOnParabola p) 
    (focus_directrix_dist : (1 : ℝ) / (2 * p.a) = 1)
    (M_to_focus_dist : Real.sqrt ((M.x - 0)^2 + (M.y - 1 / (4 * p.a))^2) = 5) :
    M.y = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordinate_l1703_170353


namespace NUMINAMATH_CALUDE_entrance_charge_is_twelve_l1703_170302

/-- The entrance charge for the strawberry fields -/
def entrance_charge (standard_price : ℕ) (paid_amount : ℕ) (picked_amount : ℕ) : ℕ :=
  standard_price * picked_amount - paid_amount

/-- Proof that the entrance charge is $12 -/
theorem entrance_charge_is_twelve :
  entrance_charge 20 128 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_entrance_charge_is_twelve_l1703_170302


namespace NUMINAMATH_CALUDE_unattainable_y_value_l1703_170365

theorem unattainable_y_value (x : ℝ) (h : x ≠ -4/3) :
  ¬∃ y : ℝ, y = -1/3 ∧ y = (2 - x) / (3*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l1703_170365


namespace NUMINAMATH_CALUDE_combinatorics_problem_l1703_170383

theorem combinatorics_problem :
  (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial (15 - 6))) = 5005 ∧
  Nat.factorial 6 = 720 := by
sorry

end NUMINAMATH_CALUDE_combinatorics_problem_l1703_170383


namespace NUMINAMATH_CALUDE_leftmost_row_tiles_l1703_170326

/-- Represents the number of tiles in each row of the floor -/
def tileSequence (firstRow : ℕ) : ℕ → ℕ
  | 0 => firstRow
  | n + 1 => tileSequence firstRow n - 2

/-- The sum of tiles in all rows -/
def totalTiles (firstRow : ℕ) : ℕ :=
  (List.range 9).map (tileSequence firstRow) |>.sum

theorem leftmost_row_tiles :
  ∃ (firstRow : ℕ), totalTiles firstRow = 405 ∧ firstRow = 53 := by
  sorry

end NUMINAMATH_CALUDE_leftmost_row_tiles_l1703_170326


namespace NUMINAMATH_CALUDE_tangent_slope_is_e_l1703_170344

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- A line passing through the origin -/
def line_through_origin (k : ℝ) (x : ℝ) : ℝ := k * x

/-- Tangent condition: The line touches the curve at exactly one point -/
def is_tangent (k : ℝ) : Prop :=
  ∃ x₀ : ℝ, 
    f x₀ = line_through_origin k x₀ ∧
    ∀ x ≠ x₀, f x ≠ line_through_origin k x

theorem tangent_slope_is_e :
  ∃ k : ℝ, is_tangent k ∧ k = Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_is_e_l1703_170344


namespace NUMINAMATH_CALUDE_dogs_groomed_l1703_170377

/-- Proves that the number of dogs groomed is 5, given the grooming times for dogs and cats,
    and the total time spent grooming dogs and 3 cats. -/
theorem dogs_groomed (dog_time : ℝ) (cat_time : ℝ) (total_time : ℝ) :
  dog_time = 2.5 →
  cat_time = 0.5 →
  total_time = 840 / 60 →
  (dog_time * ⌊(total_time - 3 * cat_time) / dog_time⌋ + 3 * cat_time) = total_time →
  ⌊(total_time - 3 * cat_time) / dog_time⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_dogs_groomed_l1703_170377


namespace NUMINAMATH_CALUDE_sum_of_roots_l1703_170381

-- Define the cubic equation
def cubic_equation (p q d x : ℝ) : Prop := 2 * x^3 - p * x^2 + q * x - d = 0

-- Define the theorem
theorem sum_of_roots (p q d x₁ x₂ x₃ : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)
  (h_roots : cubic_equation p q d x₁ ∧ cubic_equation p q d x₂ ∧ cubic_equation p q d x₃)
  (h_positive : p > 0 ∧ q > 0 ∧ d > 0)
  (h_relation : q = 2 * d) :
  x₁ + x₂ + x₃ = p / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1703_170381


namespace NUMINAMATH_CALUDE_mr_mcpherson_contribution_l1703_170394

def total_rent : ℝ := 1200
def mrs_mcpherson_percentage : ℝ := 30

theorem mr_mcpherson_contribution :
  total_rent - (mrs_mcpherson_percentage / 100 * total_rent) = 840 := by
  sorry

end NUMINAMATH_CALUDE_mr_mcpherson_contribution_l1703_170394


namespace NUMINAMATH_CALUDE_intersecting_circles_distance_l1703_170346

theorem intersecting_circles_distance (R r d : ℝ) : 
  R > 0 → r > 0 → R > r → 
  (∃ (x y : ℝ × ℝ), (x.1 - y.1)^2 + (x.2 - y.2)^2 = d^2 ∧ 
    ∃ (p : ℝ × ℝ), (p.1 - x.1)^2 + (p.2 - x.2)^2 = R^2 ∧ 
                   (p.1 - y.1)^2 + (p.2 - y.2)^2 = r^2) →
  R - r < d ∧ d < R + r :=
by sorry

end NUMINAMATH_CALUDE_intersecting_circles_distance_l1703_170346


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l1703_170360

theorem unique_solution_floor_equation :
  ∃! c : ℝ, c + ⌊c⌋ = 25.6 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l1703_170360


namespace NUMINAMATH_CALUDE_house_rent_fraction_l1703_170399

theorem house_rent_fraction (salary : ℕ) (food_fraction : ℚ) (clothes_fraction : ℚ) (remaining : ℕ) 
  (h1 : salary = 160000)
  (h2 : food_fraction = 1/5)
  (h3 : clothes_fraction = 3/5)
  (h4 : remaining = 16000)
  (h5 : ∃ (house_rent_fraction : ℚ), salary * (1 - food_fraction - clothes_fraction - house_rent_fraction) = remaining) :
  ∃ (house_rent_fraction : ℚ), house_rent_fraction = 1/10 := by
sorry

end NUMINAMATH_CALUDE_house_rent_fraction_l1703_170399


namespace NUMINAMATH_CALUDE_eliminate_y_implies_opposite_coefficients_l1703_170308

/-- Given a system of linear equations in two variables x and y,
    prove that if the sum of the equations directly eliminates y,
    then the coefficients of y in the two equations are opposite numbers. -/
theorem eliminate_y_implies_opposite_coefficients 
  (a b c d : ℝ) (k₁ k₂ : ℝ) : 
  (∀ x y : ℝ, a * x + b * y = k₁ ∧ c * x + d * y = k₂) →
  (∀ x : ℝ, (a + c) * x = k₁ + k₂) →
  b + d = 0 :=
sorry

end NUMINAMATH_CALUDE_eliminate_y_implies_opposite_coefficients_l1703_170308


namespace NUMINAMATH_CALUDE_leaky_cistern_fill_time_l1703_170321

/-- Calculates the additional time needed to fill a leaky cistern -/
theorem leaky_cistern_fill_time 
  (fill_time : ℝ) 
  (empty_time : ℝ) 
  (h1 : fill_time = 4) 
  (h2 : empty_time = 20 / 3) : 
  (1 / ((1 / fill_time) - (1 / empty_time))) - fill_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_leaky_cistern_fill_time_l1703_170321


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1703_170304

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_3 : a 3 = 9)
  (h_9 : a 9 = 3) :
  (∀ n : ℕ, a n = 12 - n) ∧ 
  (∀ n : ℕ, n ≥ 13 → a n < 0) ∧
  (∀ n : ℕ, n < 13 → a n ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1703_170304


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1703_170359

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 3}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1703_170359


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1703_170361

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (x - 5) + 1 = 10 → x = 86 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1703_170361


namespace NUMINAMATH_CALUDE_binomial_expectation_l1703_170323

/-- The number of trials -/
def n : ℕ := 3

/-- The probability of drawing a red ball -/
def p : ℚ := 3/5

/-- The expected value of a binomial distribution -/
def expected_value (n : ℕ) (p : ℚ) : ℚ := n * p

theorem binomial_expectation :
  expected_value n p = 9/5 := by sorry

end NUMINAMATH_CALUDE_binomial_expectation_l1703_170323


namespace NUMINAMATH_CALUDE_sum_of_factors_36_l1703_170362

theorem sum_of_factors_36 : (List.sum (List.filter (λ x => 36 % x = 0) (List.range 37))) = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_36_l1703_170362


namespace NUMINAMATH_CALUDE_probability_same_fruit_choices_l1703_170356

/-- The number of fruit types available -/
def num_fruits : ℕ := 4

/-- The number of fruit types each student must choose -/
def num_choices : ℕ := 2

/-- The probability that two students choose the same two types of fruits -/
def probability_same_choice : ℚ := 1 / 6

/-- Theorem stating the probability of two students choosing the same fruits -/
theorem probability_same_fruit_choices :
  (Nat.choose num_fruits num_choices : ℚ) / ((Nat.choose num_fruits num_choices : ℚ) ^ 2) = probability_same_choice :=
sorry

end NUMINAMATH_CALUDE_probability_same_fruit_choices_l1703_170356


namespace NUMINAMATH_CALUDE_complex_equation_proof_l1703_170371

theorem complex_equation_proof (z : ℂ) (h : z = -1/2 + (Real.sqrt 3 / 2) * Complex.I) : 
  z^2 + z + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l1703_170371


namespace NUMINAMATH_CALUDE_probability_three_integer_points_l1703_170364

/-- Square with diagonal endpoints (1/4, 3/4) and (-1/4, -3/4) -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    p.1 = t/4 - (1-t)/4 ∧ p.2 = 3*t/4 - 3*(1-t)/4}

/-- Random point v = (x, y) where 0 ≤ x ≤ 100 and 0 ≤ y ≤ 100 -/
def V : Set (ℝ × ℝ) :=
  {v : ℝ × ℝ | 0 ≤ v.1 ∧ v.1 ≤ 100 ∧ 0 ≤ v.2 ∧ v.2 ≤ 100}

/-- Translated copy of S centered at v -/
def T (v : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (q : ℝ × ℝ), q ∈ S ∧ p.1 = q.1 + v.1 ∧ p.2 = q.2 + v.2}

/-- Set of integer points -/
def IntegerPoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (m n : ℤ), p.1 = m ∧ p.2 = n}

/-- Probability measure on V -/
noncomputable def P : (Set (ℝ × ℝ)) → ℝ := sorry

theorem probability_three_integer_points :
  P {v ∈ V | (T v ∩ IntegerPoints).ncard = 3} = 3/100 := sorry

end NUMINAMATH_CALUDE_probability_three_integer_points_l1703_170364


namespace NUMINAMATH_CALUDE_unique_a_value_l1703_170389

theorem unique_a_value (a : ℝ) : 3 ∈ ({1, a, a - 2} : Set ℝ) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1703_170389


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1703_170317

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 + x^4 + 1 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1703_170317


namespace NUMINAMATH_CALUDE_three_zero_points_implies_k_leq_neg_two_l1703_170300

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then k * x + 2 else Real.log x

theorem three_zero_points_implies_k_leq_neg_two (k : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    |f k x₁| + k = 0 ∧ |f k x₂| + k = 0 ∧ |f k x₃| + k = 0) →
  k ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_three_zero_points_implies_k_leq_neg_two_l1703_170300


namespace NUMINAMATH_CALUDE_intersection_points_sum_greater_than_two_l1703_170324

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - a * x^2 + (2 * a - 1) * x

theorem intersection_points_sum_greater_than_two (a t : ℝ) (x₁ x₂ : ℝ) :
  a ≤ 0 →
  -1 < t →
  t < 0 →
  x₁ < x₂ →
  f a x₁ = t →
  f a x₂ = t →
  x₁ + x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_sum_greater_than_two_l1703_170324


namespace NUMINAMATH_CALUDE_sum_of_powers_mod_five_l1703_170375

theorem sum_of_powers_mod_five (n : ℕ) (hn : n > 0) : 
  (1^n + 2^n + 3^n + 4^n + 5^n) % 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_powers_mod_five_l1703_170375


namespace NUMINAMATH_CALUDE_class_size_proof_l1703_170319

theorem class_size_proof (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 60) 
  (h3 : remaining_average = 90) (h4 : excluded_count = 5) : 
  ∃ (n : ℕ), n = 15 ∧ 
  (n : ℝ) * total_average = 
    ((n : ℝ) - excluded_count) * remaining_average + (excluded_count : ℝ) * excluded_average :=
by
  sorry

#check class_size_proof

end NUMINAMATH_CALUDE_class_size_proof_l1703_170319


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1703_170355

/-- Given a geometric sequence of 10 terms, prove that if the sum of these terms is 18
    and the sum of their reciprocals is 6, then the product of these terms is (1/6)^55 -/
theorem geometric_sequence_product (a r : ℝ) (h1 : a ≠ 0) (h2 : r ≠ 0) (h3 : r ≠ 1) :
  (a * r * (r^10 - 1) / (r - 1) = 18) →
  (1 / (a * r) * (1 - 1/r^10) / (1 - 1/r) = 6) →
  (a * r)^55 = (1/6)^55 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1703_170355


namespace NUMINAMATH_CALUDE_petrol_price_equation_l1703_170343

/-- The original price of petrol per gallon -/
def P : ℝ := 2.11

/-- The reduction rate in price -/
def reduction_rate : ℝ := 0.1

/-- The additional gallons that can be bought after price reduction -/
def additional_gallons : ℝ := 5

/-- The fixed amount of money spent -/
def fixed_amount : ℝ := 200

theorem petrol_price_equation :
  fixed_amount / ((1 - reduction_rate) * P) - fixed_amount / P = additional_gallons := by
sorry

end NUMINAMATH_CALUDE_petrol_price_equation_l1703_170343


namespace NUMINAMATH_CALUDE_pets_count_l1703_170314

/-- The total number of pets owned by Teddy, Ben, and Dave -/
def total_pets (x y z a b c d e f : ℕ) : ℕ := x + y + z + a + b + c + d + e + f

/-- Theorem stating the total number of pets is 118 -/
theorem pets_count (x y z a b c d e f : ℕ) 
  (eq1 : x = 9)
  (eq2 : y = 8)
  (eq3 : z = 10)
  (eq4 : a = 21)
  (eq5 : b = 2 * y)
  (eq6 : c = z)
  (eq7 : d = x - 4)
  (eq8 : e = y + 13)
  (eq9 : f = 18) :
  total_pets x y z a b c d e f = 118 := by
  sorry


end NUMINAMATH_CALUDE_pets_count_l1703_170314


namespace NUMINAMATH_CALUDE_encryption_proof_l1703_170366

def encrypt (x : ℕ) : ℕ :=
  if x % 2 = 1 ∧ 1 ≤ x ∧ x ≤ 26 then
    (x + 1) / 2
  else if x % 2 = 0 ∧ 1 ≤ x ∧ x ≤ 26 then
    x / 2 + 13
  else
    0

def letter_to_num (c : Char) : ℕ :=
  match c with
  | 'a' => 1 | 'b' => 2 | 'c' => 3 | 'd' => 4 | 'e' => 5
  | 'f' => 6 | 'g' => 7 | 'h' => 8 | 'i' => 9 | 'j' => 10
  | 'k' => 11 | 'l' => 12 | 'm' => 13 | 'n' => 14 | 'o' => 15
  | 'p' => 16 | 'q' => 17 | 'r' => 18 | 's' => 19 | 't' => 20
  | 'u' => 21 | 'v' => 22 | 'w' => 23 | 'x' => 24 | 'y' => 25
  | 'z' => 26
  | _ => 0

def num_to_letter (n : ℕ) : Char :=
  match n with
  | 1 => 'a' | 2 => 'b' | 3 => 'c' | 4 => 'd' | 5 => 'e'
  | 6 => 'f' | 7 => 'g' | 8 => 'h' | 9 => 'i' | 10 => 'j'
  | 11 => 'k' | 12 => 'l' | 13 => 'm' | 14 => 'n' | 15 => 'o'
  | 16 => 'p' | 17 => 'q' | 18 => 'r' | 19 => 's' | 20 => 't'
  | 21 => 'u' | 22 => 'v' | 23 => 'w' | 24 => 'x' | 25 => 'y'
  | 26 => 'z'
  | _ => ' '

theorem encryption_proof :
  (encrypt (letter_to_num 'l'), 
   encrypt (letter_to_num 'o'), 
   encrypt (letter_to_num 'v'), 
   encrypt (letter_to_num 'e')) = 
  (letter_to_num 's', 
   letter_to_num 'h', 
   letter_to_num 'x', 
   letter_to_num 'c') := by
  sorry

end NUMINAMATH_CALUDE_encryption_proof_l1703_170366


namespace NUMINAMATH_CALUDE_abs_three_plus_one_l1703_170311

theorem abs_three_plus_one (a : ℝ) : 
  (|a| = 3) → (a + 1 = 4 ∨ a + 1 = -2) := by sorry

end NUMINAMATH_CALUDE_abs_three_plus_one_l1703_170311


namespace NUMINAMATH_CALUDE_triangle_problem_l1703_170325

theorem triangle_problem (A B C : Real) (a b c : Real) :
  let m : Real × Real := (Real.sqrt 3, 1 - Real.cos A)
  let n : Real × Real := (Real.sin A, -1)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⊥ n
  (a = 2) →
  (Real.cos B = Real.sqrt 3 / 3) →
  (A = 2 * Real.pi / 3 ∧ b = 4 * Real.sqrt 2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1703_170325


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l1703_170372

theorem euler_family_mean_age :
  let ages : List ℕ := [6, 6, 6, 6, 10, 10, 16]
  (List.sum ages) / (List.length ages) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l1703_170372


namespace NUMINAMATH_CALUDE_expression_equals_m_times_ten_to_1006_l1703_170336

theorem expression_equals_m_times_ten_to_1006 : 
  (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = 114337548 * 10^1006 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_m_times_ten_to_1006_l1703_170336


namespace NUMINAMATH_CALUDE_treasure_chest_coins_l1703_170387

theorem treasure_chest_coins : ∃ n : ℕ, 
  (n > 0) ∧
  (n % 8 = 2) ∧ 
  (n % 7 = 6) ∧ 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 2 ∨ m % 7 ≠ 6)) →
  (n % 9 = 7) := by
sorry

end NUMINAMATH_CALUDE_treasure_chest_coins_l1703_170387


namespace NUMINAMATH_CALUDE_frequency_of_boys_born_l1703_170390

theorem frequency_of_boys_born (total : ℕ) (boys : ℕ) (h1 : total = 1000) (h2 : boys = 515) :
  (boys : ℚ) / total = 0.515 := by
sorry

end NUMINAMATH_CALUDE_frequency_of_boys_born_l1703_170390


namespace NUMINAMATH_CALUDE_cubic_root_sum_of_eighth_powers_l1703_170327

theorem cubic_root_sum_of_eighth_powers (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  a^8 + b^8 + c^8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_of_eighth_powers_l1703_170327


namespace NUMINAMATH_CALUDE_fibonacci_congruence_existence_and_uniqueness_l1703_170335

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_congruence_existence_and_uniqueness :
  ∃! (a b m : ℕ), 0 < a ∧ a < m ∧ 0 < b ∧ b < m ∧
    (∀ n : ℕ, n > 0 → (fibonacci n - a * n * (b ^ n)) % m = 0) ∧
    a = 2 ∧ b = 3 ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_congruence_existence_and_uniqueness_l1703_170335


namespace NUMINAMATH_CALUDE_polynomial_real_root_condition_l1703_170378

theorem polynomial_real_root_condition (a : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 - x^2 + a^2*x + 1 = 0) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_condition_l1703_170378


namespace NUMINAMATH_CALUDE_extracted_25_30_is_120_l1703_170382

/-- Represents the number of questionnaires collected for each age group -/
structure QuestionnaireCount where
  group_8_12 : ℕ
  group_13_18 : ℕ
  group_19_24 : ℕ
  group_25_30 : ℕ

/-- Represents the sample extracted from the collected questionnaires -/
structure SampleCount where
  total : ℕ
  group_13_18 : ℕ

/-- Calculates the number of questionnaires extracted from the 25-30 age group -/
def extracted_25_30 (collected : QuestionnaireCount) (sample : SampleCount) : ℕ :=
  (collected.group_25_30 * sample.group_13_18) / collected.group_13_18

theorem extracted_25_30_is_120 (collected : QuestionnaireCount) (sample : SampleCount) :
  collected.group_8_12 = 120 →
  collected.group_13_18 = 180 →
  collected.group_19_24 = 240 →
  sample.total = 300 →
  sample.group_13_18 = 60 →
  extracted_25_30 collected sample = 120 := by
  sorry

#check extracted_25_30_is_120

end NUMINAMATH_CALUDE_extracted_25_30_is_120_l1703_170382


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1703_170330

/-- Simplification of a polynomial expression -/
theorem polynomial_simplification (x : ℝ) :
  3 * x + 10 * x^2 + 5 * x^3 + 15 - (7 - 3 * x - 10 * x^2 - 5 * x^3) =
  10 * x^3 + 20 * x^2 + 6 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1703_170330


namespace NUMINAMATH_CALUDE_right_triangle_345_l1703_170315

/-- A triangle with side lengths 3, 4, and 5 is a right triangle. -/
theorem right_triangle_345 :
  ∀ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 →
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_345_l1703_170315


namespace NUMINAMATH_CALUDE_mrs_sheridan_cats_l1703_170338

theorem mrs_sheridan_cats (initial_cats : ℕ) : 
  initial_cats + 14 = 31 → initial_cats = 17 := by
  sorry

end NUMINAMATH_CALUDE_mrs_sheridan_cats_l1703_170338


namespace NUMINAMATH_CALUDE_power_sum_seven_l1703_170328

theorem power_sum_seven (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 2)
  (h2 : α₁^2 + α₂^2 + α₃^2 = 6)
  (h3 : α₁^3 + α₂^3 + α₃^3 = 14) :
  α₁^7 + α₂^7 + α₃^7 = 46 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_seven_l1703_170328


namespace NUMINAMATH_CALUDE_max_abs_cexp_minus_two_l1703_170313

-- Define the complex exponential function
noncomputable def cexp (x : ℝ) : ℂ := Complex.exp (Complex.I * x)

-- State Euler's formula
axiom euler_formula (x : ℝ) : cexp x = Complex.cos x + Complex.I * Complex.sin x

-- State the theorem
theorem max_abs_cexp_minus_two :
  ∃ (M : ℝ), M = 3 ∧ ∀ (x : ℝ), Complex.abs (cexp x - 2) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_abs_cexp_minus_two_l1703_170313


namespace NUMINAMATH_CALUDE_total_watermelon_seeds_l1703_170388

/-- The number of watermelon seeds each person has -/
structure WatermelonSeeds where
  bom : ℕ
  gwi : ℕ
  yeon : ℕ
  eun : ℕ

/-- Given conditions about watermelon seeds -/
def watermelon_seed_conditions (w : WatermelonSeeds) : Prop :=
  w.yeon = 3 * w.gwi ∧
  w.gwi = w.bom + 40 ∧
  w.eun = 2 * w.gwi ∧
  w.bom = 300

/-- Theorem stating the total number of watermelon seeds -/
theorem total_watermelon_seeds (w : WatermelonSeeds) 
  (h : watermelon_seed_conditions w) : 
  w.bom + w.gwi + w.yeon + w.eun = 2340 := by
  sorry

end NUMINAMATH_CALUDE_total_watermelon_seeds_l1703_170388


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1703_170369

/-- Given a geometric sequence {a_n}, prove that if a_2 * a_6 = 36, then a_4 = ±6 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_prod : a 2 * a 6 = 36) : a 4 = 6 ∨ a 4 = -6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1703_170369


namespace NUMINAMATH_CALUDE_rectangular_field_shortcut_l1703_170322

theorem rectangular_field_shortcut (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x / y = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_shortcut_l1703_170322


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l1703_170303

/-- The number of aquariums Tyler has -/
def num_aquariums : ℕ := 8

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 64

/-- The total number of saltwater animals Tyler has -/
def total_animals : ℕ := num_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_animals = 512 :=
sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l1703_170303


namespace NUMINAMATH_CALUDE_bicycle_speed_problem_l1703_170333

/-- Proves that given a distance of 12 km, if person A's speed is 1.2 times person B's speed,
    and A arrives 1/6 hour earlier than B, then B's speed is 12 km/h. -/
theorem bicycle_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
    (h1 : distance = 12)
    (h2 : speed_ratio = 1.2)
    (h3 : time_difference = 1/6) : 
  let speed_B := distance / (distance / (speed_ratio * (distance / time_difference)) + time_difference)
  speed_B = 12 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_speed_problem_l1703_170333


namespace NUMINAMATH_CALUDE_restaurant_donates_24_l1703_170367

/-- The restaurant's donation policy -/
def donation_rate : ℚ := 2 / 10

/-- The average customer donation -/
def avg_customer_donation : ℚ := 3

/-- The number of customers -/
def num_customers : ℕ := 40

/-- The restaurant's donation function -/
def restaurant_donation (customer_total : ℚ) : ℚ :=
  (customer_total / 10) * 2

/-- Theorem: The restaurant donates $24 given the conditions -/
theorem restaurant_donates_24 :
  restaurant_donation (avg_customer_donation * num_customers) = 24 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_donates_24_l1703_170367


namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l1703_170320

theorem bumper_car_line_problem (initial_people : ℕ) (joined : ℕ) (final_people : ℕ) :
  initial_people = 12 →
  joined = 15 →
  final_people = 17 →
  ∃ (left : ℕ), initial_people - left + joined = final_people ∧ left = 10 :=
by sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l1703_170320


namespace NUMINAMATH_CALUDE_three_digit_two_digit_operations_l1703_170398

theorem three_digit_two_digit_operations (a b : ℕ) 
  (ha : 100 ≤ a ∧ a ≤ 999) (hb : 10 ≤ b ∧ b ≤ 99) : 
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x + y ≥ a + b) → a + b = 110 ∧
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x + y ≤ a + b) → a + b = 1098 ∧
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x - y ≥ a - b) → a - b = 1 ∧
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x - y ≤ a - b) → a - b = 989 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_two_digit_operations_l1703_170398


namespace NUMINAMATH_CALUDE_min_correct_problems_is_16_l1703_170347

/-- AMC 10 scoring system and John's strategy -/
structure AMC10 where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Nat
  unanswered_points : Nat
  min_total_score : Nat

/-- Calculate the minimum number of correctly solved problems -/
def min_correct_problems (test : AMC10) : Nat :=
  let unanswered := test.total_problems - test.attempted_problems
  let unanswered_score := unanswered * test.unanswered_points
  let required_score := test.min_total_score - unanswered_score
  (required_score + test.correct_points - 1) / test.correct_points

/-- Theorem: The minimum number of correctly solved problems is 16 -/
theorem min_correct_problems_is_16 (test : AMC10) 
  (h1 : test.total_problems = 25)
  (h2 : test.attempted_problems = 20)
  (h3 : test.correct_points = 7)
  (h4 : test.unanswered_points = 2)
  (h5 : test.min_total_score = 120) :
  min_correct_problems test = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_correct_problems_is_16_l1703_170347


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l1703_170316

/-- The number of y-intercepts of the parabola x = 3y^2 - 5y - 2 -/
def num_y_intercepts : ℕ := 2

/-- The equation of the parabola -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 5 * y - 2

theorem parabola_y_intercepts :
  (∃ (s : Finset ℝ), s.card = num_y_intercepts ∧
    ∀ y ∈ s, parabola_equation y = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l1703_170316


namespace NUMINAMATH_CALUDE_abs_z_equals_10_l1703_170358

def z : ℂ := (3 + Complex.I)^2 * Complex.I

theorem abs_z_equals_10 : Complex.abs z = 10 := by sorry

end NUMINAMATH_CALUDE_abs_z_equals_10_l1703_170358


namespace NUMINAMATH_CALUDE_election_result_l1703_170312

theorem election_result (total_votes : ℕ) (invalid_percentage : ℚ) (second_candidate_votes : ℕ) : 
  total_votes = 7000 →
  invalid_percentage = 1/5 →
  second_candidate_votes = 2520 →
  (((1 - invalid_percentage) * total_votes - second_candidate_votes) / ((1 - invalid_percentage) * total_votes) : ℚ) = 11/20 := by
sorry

end NUMINAMATH_CALUDE_election_result_l1703_170312


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1703_170363

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, ∃ r : ℝ, r ≠ 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1703_170363


namespace NUMINAMATH_CALUDE_circles_intersect_l1703_170397

/-- Two circles are intersecting if the distance between their centers is greater than the absolute
    difference of their radii and less than the sum of their radii. -/
def are_circles_intersecting (r1 r2 d : ℝ) : Prop :=
  abs (r1 - r2) < d ∧ d < r1 + r2

/-- Given two circles with radii 4 and 3, and a distance of 5 between their centers,
    prove that they are intersecting. -/
theorem circles_intersect : are_circles_intersecting 4 3 5 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l1703_170397


namespace NUMINAMATH_CALUDE_cos_neg_three_pi_half_l1703_170349

theorem cos_neg_three_pi_half : Real.cos (-3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_three_pi_half_l1703_170349


namespace NUMINAMATH_CALUDE_square_difference_of_solutions_l1703_170309

theorem square_difference_of_solutions (α β : ℝ) : 
  (α^2 = 2*α + 1) → (β^2 = 2*β + 1) → (α ≠ β) → (α - β)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_square_difference_of_solutions_l1703_170309


namespace NUMINAMATH_CALUDE_nathan_air_hockey_games_l1703_170370

/-- The number of times Nathan played basketball -/
def basketball_games : ℕ := 4

/-- The cost of each game in tokens -/
def tokens_per_game : ℕ := 3

/-- The total number of tokens Nathan used -/
def total_tokens : ℕ := 18

/-- The number of times Nathan played air hockey -/
def air_hockey_games : ℕ := 2

theorem nathan_air_hockey_games :
  air_hockey_games = (total_tokens - basketball_games * tokens_per_game) / tokens_per_game :=
by sorry

end NUMINAMATH_CALUDE_nathan_air_hockey_games_l1703_170370


namespace NUMINAMATH_CALUDE_max_profit_at_180_l1703_170392

/-- The total cost function for a certain product -/
def total_cost (x : ℝ) : ℝ := 0.1 * x^2 - 11 * x + 3000

/-- The selling price per unit in ten thousand yuan -/
def selling_price : ℝ := 25

/-- The profit function -/
def profit (x : ℝ) : ℝ := selling_price * x - total_cost x

/-- Theorem: The production volume that maximizes profit is 180 units -/
theorem max_profit_at_180 : 
  ∃ (max_x : ℝ), (∀ x : ℝ, profit x ≤ profit max_x) ∧ max_x = 180 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_180_l1703_170392


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l1703_170386

theorem students_playing_both_sports (total : ℕ) (hockey : ℕ) (basketball : ℕ) (neither : ℕ) :
  total = 25 →
  hockey = 15 →
  basketball = 16 →
  neither = 4 →
  hockey + basketball - (total - neither) = 10 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l1703_170386


namespace NUMINAMATH_CALUDE_cymbal_triangle_sync_l1703_170396

theorem cymbal_triangle_sync (cymbal_beats triangle_beats : ℕ) 
  (h1 : cymbal_beats = 7) (h2 : triangle_beats = 2) : 
  Nat.lcm cymbal_beats triangle_beats = 14 := by
  sorry

end NUMINAMATH_CALUDE_cymbal_triangle_sync_l1703_170396


namespace NUMINAMATH_CALUDE_number_equality_l1703_170380

theorem number_equality (x : ℝ) : (0.4 * x = 0.3 * 50) → x = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l1703_170380


namespace NUMINAMATH_CALUDE_problem_solution_l1703_170374

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- State the theorem
theorem problem_solution :
  ∀ m : ℝ,
  (∀ x : ℝ, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-1) 1) →
  (m = 1 ∧
   {x : ℝ | |x + 1| + |x - 2| > 4 * m} = {x : ℝ | x < -3/2 ∨ x > 5/2}) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1703_170374


namespace NUMINAMATH_CALUDE_line_circle_properties_l1703_170395

-- Define the line l and circle C
def line_l (m x y : ℝ) : Prop := (m + 2) * x + (1 - 2 * m) * y + 4 * m - 2 = 0
def circle_C (x y : ℝ) : Prop := x^2 - 2 * x + y^2 = 0

-- Define the intersection points M and N
def intersect_points (m : ℝ) : Prop := ∃ x_M y_M x_N y_N : ℝ,
  line_l m x_M y_M ∧ circle_C x_M y_M ∧
  line_l m x_N y_N ∧ circle_C x_N y_N ∧
  (x_M ≠ x_N ∨ y_M ≠ y_N)

-- Define the slopes of OM and ON
def slope_OM_ON (m : ℝ) : Prop := ∃ k₁ k₂ x_M y_M x_N y_N : ℝ,
  line_l m x_M y_M ∧ circle_C x_M y_M ∧
  line_l m x_N y_N ∧ circle_C x_N y_N ∧
  k₁ = y_M / x_M ∧ k₂ = y_N / x_N

-- Theorem statement
theorem line_circle_properties :
  (∀ m : ℝ, line_l m 0 2) ∧
  (∀ m : ℝ, intersect_points m → -(m + 2) / (1 - 2 * m) < -3/4) ∧
  (∀ m : ℝ, slope_OM_ON m → ∃ k₁ k₂ : ℝ, k₁ + k₂ = 1) :=
sorry

end NUMINAMATH_CALUDE_line_circle_properties_l1703_170395


namespace NUMINAMATH_CALUDE_tangent_line_property_l1703_170306

/-- Given a function f: ℝ → ℝ, if the tangent line to the graph of f at the point (2, f(2))
    has the equation 2x - y - 3 = 0, then f(2) + f'(2) = 3. -/
theorem tangent_line_property (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x y, y = f 2 → 2 * x - y - 3 = 0 ↔ y = 2 * x - 3) →
  f 2 + deriv f 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_property_l1703_170306


namespace NUMINAMATH_CALUDE_problem_statement_l1703_170373

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : a * b = 1) : 
  (a + b ≥ 2) ∧ (a^3 + b^3 ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1703_170373


namespace NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l1703_170301

def is_17_digit (n : ℕ) : Prop := 10^16 ≤ n ∧ n < 10^17

def reverse_number (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n)
  List.foldl (λ acc d => acc * 10 + d) 0 digits

def has_even_digit (n : ℕ) : Prop :=
  ∃ d, d ∈ Nat.digits 10 n ∧ Even d

theorem sum_with_reverse_has_even_digit (n : ℕ) (h : is_17_digit n) :
  has_even_digit (n + reverse_number n) := by
  sorry

end NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l1703_170301


namespace NUMINAMATH_CALUDE_shelby_rain_time_l1703_170307

/-- Represents the speed of Shelby's scooter in miles per hour -/
structure ScooterSpeed where
  normal : ℝ  -- Speed when not raining
  rain : ℝ    -- Speed when raining

/-- Represents Shelby's journey -/
structure Journey where
  total_distance : ℝ  -- Total distance covered in miles
  total_time : ℝ      -- Total time taken in minutes
  rain_time : ℝ       -- Time driven in rain in minutes

/-- Checks if the given journey satisfies the conditions of Shelby's ride -/
def is_valid_journey (speed : ScooterSpeed) (j : Journey) : Prop :=
  speed.normal = 40 ∧
  speed.rain = 25 ∧
  j.total_distance = 20 ∧
  j.total_time = 40 ∧
  j.total_distance = (speed.normal / 60) * (j.total_time - j.rain_time) + (speed.rain / 60) * j.rain_time

theorem shelby_rain_time (speed : ScooterSpeed) (j : Journey) 
  (h : is_valid_journey speed j) : j.rain_time = 27 := by
  sorry

end NUMINAMATH_CALUDE_shelby_rain_time_l1703_170307


namespace NUMINAMATH_CALUDE_domain_transformation_l1703_170368

-- Define a real-valued function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x²)
def domain_f_squared : Set ℝ := Set.Ioc (-3) 1

-- Define the domain of f(x-1)
def domain_f_shifted : Set ℝ := Set.Ico 1 10

-- Theorem statement
theorem domain_transformation (h : ∀ x, x ∈ domain_f_squared ↔ f (x^2) ∈ Set.range f) :
  ∀ x, x ∈ domain_f_shifted ↔ f (x - 1) ∈ Set.range f :=
sorry

end NUMINAMATH_CALUDE_domain_transformation_l1703_170368


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l1703_170340

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (r : ℝ) : 
  (0 < n) →
  (0 < r) →
  (r < 1 / 500) →
  (m : ℝ)^(1/3) = n + r →
  (∀ k < n, ¬∃ s, (0 < s) ∧ (s < 1 / 500) ∧ (∃ l : ℕ, (l : ℝ)^(1/3) = k + s)) →
  n = 13 := by
  sorry

#check smallest_cube_root_with_small_fraction

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l1703_170340


namespace NUMINAMATH_CALUDE_sqrt_minus_three_minus_m_real_l1703_170305

theorem sqrt_minus_three_minus_m_real (m : ℝ) :
  (∃ (x : ℝ), x ^ 2 = -3 - m) ↔ m ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_minus_three_minus_m_real_l1703_170305


namespace NUMINAMATH_CALUDE_nh4_2so4_weight_l1703_170384

/-- Atomic weight of Nitrogen in g/mol -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Atomic weight of Sulfur in g/mol -/
def S_weight : ℝ := 32.07

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Number of Nitrogen atoms in (NH4)2SO4 -/
def N_count : ℕ := 2

/-- Number of Hydrogen atoms in (NH4)2SO4 -/
def H_count : ℕ := 8

/-- Number of Sulfur atoms in (NH4)2SO4 -/
def S_count : ℕ := 1

/-- Number of Oxygen atoms in (NH4)2SO4 -/
def O_count : ℕ := 4

/-- Number of moles of (NH4)2SO4 -/
def moles : ℝ := 7

/-- Molecular weight of (NH4)2SO4 in g/mol -/
def molecular_weight : ℝ := N_weight * N_count + H_weight * H_count + S_weight * S_count + O_weight * O_count

theorem nh4_2so4_weight : moles * molecular_weight = 924.19 := by
  sorry

end NUMINAMATH_CALUDE_nh4_2so4_weight_l1703_170384


namespace NUMINAMATH_CALUDE_new_person_weight_l1703_170334

/-- Given a group of 8 people where one person weighing 45 kg is replaced by a new person,
    and the average weight increases by 6 kg, the weight of the new person is 93 kg. -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 45 →
  avg_increase = 6 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 93 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1703_170334


namespace NUMINAMATH_CALUDE_existence_of_monotonic_tail_l1703_170331

def IsMonotonicSegment (a : ℕ → ℝ) (i m : ℕ) : Prop :=
  (∀ j ∈ Finset.range (m - 1), a (i + j) < a (i + j + 1)) ∨
  (∀ j ∈ Finset.range (m - 1), a (i + j) > a (i + j + 1))

theorem existence_of_monotonic_tail
  (a : ℕ → ℝ)
  (distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (monotonic_segment : ∀ k, ∃ i m, k ∈ Finset.range m ∧ IsMonotonicSegment a i (k + 1)) :
  ∃ N, (∀ i j, N ≤ i → i < j → a i < a j) ∨ (∀ i j, N ≤ i → i < j → a i > a j) :=
sorry

end NUMINAMATH_CALUDE_existence_of_monotonic_tail_l1703_170331


namespace NUMINAMATH_CALUDE_calculation_difference_l1703_170350

theorem calculation_difference : ∀ x : ℝ, (x - 3) + 49 = 66 → (3 * x + 49) - 66 = 43 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l1703_170350


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1703_170351

theorem min_value_quadratic (x : ℝ) : x^2 - 3*x + 2023 ≥ 2020 + 3/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1703_170351


namespace NUMINAMATH_CALUDE_angle_z_is_90_l1703_170354

-- Define a triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)

-- Define the property that the sum of angles in a triangle is 180°
axiom triangle_angle_sum (t : Triangle) : t.X + t.Y + t.Z = 180

-- Theorem: If the sum of angles X and Y is 90°, then angle Z is 90°
theorem angle_z_is_90 (t : Triangle) (h : t.X + t.Y = 90) : t.Z = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_z_is_90_l1703_170354


namespace NUMINAMATH_CALUDE_optimal_square_perimeter_l1703_170391

/-- Given a wire of length 1 cut into two pieces to form a square and a circle,
    the perimeter of the square that minimizes the sum of their areas is π / (π + 4) -/
theorem optimal_square_perimeter :
  ∃ (x : ℝ), x > 0 ∧ x < 1 ∧
  (∀ (y : ℝ), y > 0 → y < 1 →
    x^2 / 16 + (1 - x)^2 / (4 * π) ≤ y^2 / 16 + (1 - y)^2 / (4 * π)) ∧
  x = π / (π + 4) := by
  sorry

end NUMINAMATH_CALUDE_optimal_square_perimeter_l1703_170391
