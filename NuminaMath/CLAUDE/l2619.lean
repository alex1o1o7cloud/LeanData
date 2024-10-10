import Mathlib

namespace juice_problem_l2619_261900

theorem juice_problem (J : ℝ) : 
  J > 0 →
  (1/6 : ℝ) * J + (2/5 : ℝ) * (5/6 : ℝ) * J + (2/3 : ℝ) * (1/2 : ℝ) * J + 120 = J →
  J = 720 := by
sorry

end juice_problem_l2619_261900


namespace equal_segments_l2619_261949

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (incenter : Point → Point → Point → Point)
variable (intersect_circle_line : Circle → Point → Point → Point)
variable (circle_through : Point → Point → Point → Circle)
variable (length : Point → Point → ℝ)

-- State the theorem
theorem equal_segments 
  (A B C I X Y : Point) :
  I = incenter A B C →
  X = intersect_circle_line (circle_through A C I) B C →
  Y = intersect_circle_line (circle_through B C I) A C →
  length A Y = length B X :=
sorry

end equal_segments_l2619_261949


namespace train_speed_proof_l2619_261948

-- Define the given parameters
def train_length : ℝ := 155
def bridge_length : ℝ := 220
def crossing_time : ℝ := 30

-- Define the conversion factor from m/s to km/hr
def m_s_to_km_hr : ℝ := 3.6

-- Theorem statement
theorem train_speed_proof :
  let total_distance := train_length + bridge_length
  let speed_m_s := total_distance / crossing_time
  let speed_km_hr := speed_m_s * m_s_to_km_hr
  speed_km_hr = 45 := by sorry

end train_speed_proof_l2619_261948


namespace complex_inequality_l2619_261982

theorem complex_inequality (z₁ z₂ z₃ z₄ : ℂ) :
  ‖z₁ - z₃‖^2 + ‖z₂ - z₄‖^2 ≤ ‖z₁ - z₂‖^2 + ‖z₂ - z₃‖^2 + ‖z₃ - z₄‖^2 + ‖z₄ - z₁‖^2 ∧
  (‖z₁ - z₃‖^2 + ‖z₂ - z₄‖^2 = ‖z₁ - z₂‖^2 + ‖z₂ - z₃‖^2 + ‖z₃ - z₄‖^2 + ‖z₄ - z₁‖^2 ↔ z₁ + z₃ = z₂ + z₄) :=
by sorry

end complex_inequality_l2619_261982


namespace sum_of_distinct_divisors_of_2000_l2619_261912

def divisors_of_2000 : List ℕ := [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000]

def is_sum_of_distinct_divisors (n : ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.Nodup ∧ subset.Subset divisors_of_2000 ∧ subset.sum = n

theorem sum_of_distinct_divisors_of_2000 :
  ∀ n : ℕ, n > 0 ∧ n < 2000 → is_sum_of_distinct_divisors n :=
sorry

end sum_of_distinct_divisors_of_2000_l2619_261912


namespace donation_ratio_l2619_261969

theorem donation_ratio (shirts pants shorts : ℕ) : 
  shirts = 4 →
  pants = 2 * shirts →
  shirts + pants + shorts = 16 →
  shorts * 2 = pants :=
by
  sorry

end donation_ratio_l2619_261969


namespace cone_height_from_lateral_surface_l2619_261978

/-- If the lateral surface of a cone, when unfolded, forms a semicircle with an area of 2π,
    then the height of the cone is √3. -/
theorem cone_height_from_lateral_surface (r h : ℝ) : 
  r > 0 → h > 0 → 2 * π = π * (r^2 + h^2) → h = Real.sqrt 3 := by
  sorry

end cone_height_from_lateral_surface_l2619_261978


namespace interest_calculation_l2619_261950

/-- Calculates the total interest earned after 4 years given an initial investment
    and annual interest rates for each year. -/
def total_interest (initial_investment : ℝ) (rate1 rate2 rate3 rate4 : ℝ) : ℝ :=
  let final_amount := initial_investment * (1 + rate1) * (1 + rate2) * (1 + rate3) * (1 + rate4)
  final_amount - initial_investment

/-- Proves that the total interest earned after 4 years is approximately $572.36416
    given the specified initial investment and interest rates. -/
theorem interest_calculation :
  let initial_investment := 2000
  let rate1 := 0.05
  let rate2 := 0.06
  let rate3 := 0.07
  let rate4 := 0.08
  abs (total_interest initial_investment rate1 rate2 rate3 rate4 - 572.36416) < 0.00001 := by
  sorry

end interest_calculation_l2619_261950


namespace expression_equals_95_l2619_261943

theorem expression_equals_95 : 
  let some_number := -5765435
  7 ^ 8 - 6 / 2 + 9 ^ 3 + 3 + some_number = 95 := by
sorry

end expression_equals_95_l2619_261943


namespace ladies_walking_distance_l2619_261985

theorem ladies_walking_distance (x y : ℝ) (h1 : x = 2 * y) (h2 : y = 4) :
  x + y = 12 := by sorry

end ladies_walking_distance_l2619_261985


namespace jason_born_1981_l2619_261934

/-- The year of the first AMC 8 competition -/
def first_amc8_year : ℕ := 1985

/-- The number of the AMC 8 competition Jason participated in -/
def jason_amc8_number : ℕ := 10

/-- Jason's age when he participated in the AMC 8 -/
def jason_age : ℕ := 13

/-- Calculates the year of a given AMC 8 competition -/
def amc8_year (n : ℕ) : ℕ := first_amc8_year + n - 1

/-- Jason's birth year -/
def jason_birth_year : ℕ := amc8_year jason_amc8_number - jason_age

theorem jason_born_1981 : jason_birth_year = 1981 := by
  sorry

end jason_born_1981_l2619_261934


namespace integral_of_improper_rational_function_l2619_261977

noncomputable def F (x : ℝ) : ℝ :=
  x^3 / 3 + x^2 - x + 
  (1 / (4 * Real.sqrt 2)) * Real.log ((x^2 - Real.sqrt 2 * x + 1) / (x^2 + Real.sqrt 2 * x + 1)) + 
  (1 / (2 * Real.sqrt 2)) * (Real.arctan (Real.sqrt 2 * x + 1) + Real.arctan (Real.sqrt 2 * x - 1))

theorem integral_of_improper_rational_function (x : ℝ) :
  deriv F x = (x^6 + 2*x^5 - x^4 + x^2 + 2*x) / (x^4 + 1) := by sorry

end integral_of_improper_rational_function_l2619_261977


namespace gcd_of_1975_and_2625_l2619_261931

theorem gcd_of_1975_and_2625 : Nat.gcd 1975 2625 = 25 := by
  sorry

end gcd_of_1975_and_2625_l2619_261931


namespace desired_depth_calculation_l2619_261908

/-- Calculates the desired depth to be dug given initial and new working conditions -/
theorem desired_depth_calculation
  (initial_men : ℕ)
  (initial_hours : ℕ)
  (initial_depth : ℝ)
  (new_hours : ℕ)
  (extra_men : ℕ)
  (h1 : initial_men = 72)
  (h2 : initial_hours = 8)
  (h3 : initial_depth = 30)
  (h4 : new_hours = 6)
  (h5 : extra_men = 88)
  : ∃ (desired_depth : ℝ), desired_depth = 50 := by
  sorry


end desired_depth_calculation_l2619_261908


namespace different_color_pairings_l2619_261904

/-- The number of distinct colors for bowls and glasses -/
def num_colors : ℕ := 5

/-- The number of pairings where the bowl and glass colors are different -/
def num_different_pairings : ℕ := num_colors * (num_colors - 1)

/-- Theorem stating that the number of different color pairings is 20 -/
theorem different_color_pairings :
  num_different_pairings = 20 := by
  sorry

#eval num_different_pairings -- This should output 20

end different_color_pairings_l2619_261904


namespace geometry_book_pages_l2619_261915

theorem geometry_book_pages :
  let new_edition : ℕ := 450
  let old_edition : ℕ := 340
  let deluxe_edition : ℕ := new_edition + old_edition + 125
  (2 * old_edition - 230 = new_edition) ∧
  (deluxe_edition ≥ old_edition + (old_edition / 10)) →
  old_edition = 340 :=
by sorry

end geometry_book_pages_l2619_261915


namespace sum_digits_greatest_prime_divisor_59048_l2619_261951

/-- The greatest prime divisor of a natural number -/
def greatest_prime_divisor (n : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the digits of the greatest prime divisor of 59048 is 7 -/
theorem sum_digits_greatest_prime_divisor_59048 :
  sum_of_digits (greatest_prime_divisor 59048) = 7 := by sorry

end sum_digits_greatest_prime_divisor_59048_l2619_261951


namespace invisible_dots_count_l2619_261966

/-- The sum of numbers on a single die -/
def dieFaceSum : ℕ := 21

/-- The number of dice -/
def numDice : ℕ := 5

/-- The visible numbers on the dice -/
def visibleNumbers : List ℕ := [1, 1, 2, 2, 3, 3, 4, 5, 6, 6]

/-- The theorem stating that the total number of dots not visible is 72 -/
theorem invisible_dots_count : 
  numDice * dieFaceSum - visibleNumbers.sum = 72 := by sorry

end invisible_dots_count_l2619_261966


namespace trigonometric_expression_value_l2619_261937

theorem trigonometric_expression_value (α : Real) (h : Real.tan α = 3) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / (Real.cos (3 * π / 2 - α) + 2 * Real.cos (-π + α)) = -2 / 5 := by
sorry

end trigonometric_expression_value_l2619_261937


namespace floor_sum_example_l2619_261925

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l2619_261925


namespace geometric_mean_of_two_and_six_l2619_261988

theorem geometric_mean_of_two_and_six :
  ∃ (x : ℝ), x^2 = 2 * 6 ∧ (x = 2 * Real.sqrt 3 ∨ x = -2 * Real.sqrt 3) :=
by sorry

end geometric_mean_of_two_and_six_l2619_261988


namespace half_plus_seven_equals_seventeen_l2619_261907

theorem half_plus_seven_equals_seventeen (n : ℝ) : (1/2 * n + 7 = 17) → n = 20 := by
  sorry

end half_plus_seven_equals_seventeen_l2619_261907


namespace apple_orange_cost_l2619_261922

theorem apple_orange_cost (apple_cost orange_cost : ℚ) 
  (eq1 : 2 * apple_cost + 3 * orange_cost = 6)
  (eq2 : 4 * apple_cost + 7 * orange_cost = 13) :
  16 * apple_cost + 23 * orange_cost = 47 := by
  sorry

end apple_orange_cost_l2619_261922


namespace reflection_across_y_axis_l2619_261936

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (-1, 3)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (1, 3)

theorem reflection_across_y_axis :
  reflect_y original_point = reflected_point := by sorry

end reflection_across_y_axis_l2619_261936


namespace student_proportion_is_frequency_rate_l2619_261970

/-- Represents the total number of people in the population -/
def total_population : ℕ := 10

/-- Represents the number of students in the population -/
def number_of_students : ℕ := 4

/-- Represents the proportion of students in the population -/
def student_proportion : ℚ := 2 / 5

/-- Defines what a frequency rate is in this context -/
def is_frequency_rate (proportion : ℚ) (num : ℕ) (total : ℕ) : Prop :=
  proportion = num / total

/-- Theorem stating that the given proportion is a frequency rate -/
theorem student_proportion_is_frequency_rate :
  is_frequency_rate student_proportion number_of_students total_population := by
  sorry

end student_proportion_is_frequency_rate_l2619_261970


namespace attendance_ratio_l2619_261929

/-- Proves that given the charges on three days and the average charge, 
    the ratio of attendance on these days is 4:1:5. -/
theorem attendance_ratio 
  (charge1 charge2 charge3 avg_charge : ℚ)
  (h1 : charge1 = 15)
  (h2 : charge2 = 15/2)
  (h3 : charge3 = 5/2)
  (h4 : avg_charge = 5)
  (x y z : ℚ) -- attendance on day 1, 2, and 3 respectively
  (h5 : (charge1 * x + charge2 * y + charge3 * z) / (x + y + z) = avg_charge) :
  ∃ (k : ℚ), k > 0 ∧ x = 4*k ∧ y = k ∧ z = 5*k := by
sorry


end attendance_ratio_l2619_261929


namespace count_numbers_with_seven_equals_133_l2619_261930

/-- Returns true if the given natural number contains the digit 7 at least once -/
def contains_seven (n : ℕ) : Bool := sorry

/-- Counts the number of natural numbers from 1 to 700 (inclusive) that contain the digit 7 at least once -/
def count_numbers_with_seven : ℕ := sorry

theorem count_numbers_with_seven_equals_133 : count_numbers_with_seven = 133 := by sorry

end count_numbers_with_seven_equals_133_l2619_261930


namespace soda_cost_is_one_l2619_261921

/-- The cost of one can of soda -/
def soda_cost : ℝ := 1

/-- The cost of one soup -/
def soup_cost : ℝ := 3 * soda_cost

/-- The cost of one sandwich -/
def sandwich_cost : ℝ := 3 * soup_cost

/-- The total cost of Sean's purchase -/
def total_cost : ℝ := 3 * soda_cost + 2 * soup_cost + sandwich_cost

theorem soda_cost_is_one :
  soda_cost = 1 ∧ total_cost = 18 := by sorry

end soda_cost_is_one_l2619_261921


namespace one_third_of_36_l2619_261965

theorem one_third_of_36 : (1 / 3 : ℚ) * 36 = 12 := by sorry

end one_third_of_36_l2619_261965


namespace lost_money_proof_l2619_261995

def money_lost (initial_amount spent_amount remaining_amount : ℕ) : ℕ :=
  (initial_amount - spent_amount) - remaining_amount

theorem lost_money_proof (initial_amount spent_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  money_lost initial_amount spent_amount remaining_amount = 6 := by
  sorry

#eval money_lost 11 2 3

end lost_money_proof_l2619_261995


namespace halloween_cleanup_time_l2619_261924

theorem halloween_cleanup_time
  (egg_cleanup_time : ℕ)
  (tp_cleanup_time : ℕ)
  (egg_count : ℕ)
  (tp_count : ℕ)
  (h1 : egg_cleanup_time = 15)  -- 15 seconds per egg
  (h2 : tp_cleanup_time = 30)   -- 30 minutes per roll of toilet paper
  (h3 : egg_count = 60)         -- 60 eggs
  (h4 : tp_count = 7)           -- 7 rolls of toilet paper
  : (egg_count * egg_cleanup_time) / 60 + tp_count * tp_cleanup_time = 225 := by
  sorry

#check halloween_cleanup_time

end halloween_cleanup_time_l2619_261924


namespace floor_sqrt_18_squared_l2619_261973

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by
  sorry

end floor_sqrt_18_squared_l2619_261973


namespace sheet_area_difference_l2619_261947

/-- The combined area (front and back) of a rectangular sheet -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets -/
def areaDifference (l1 w1 l2 w2 : ℝ) : ℝ :=
  combinedArea l1 w1 - combinedArea l2 w2

theorem sheet_area_difference :
  areaDifference 11 19 9.5 11 = 209 := by
  sorry

end sheet_area_difference_l2619_261947


namespace fraction_equality_l2619_261989

theorem fraction_equality (a b : ℚ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 2 := by
  sorry

end fraction_equality_l2619_261989


namespace unique_prime_generator_l2619_261987

theorem unique_prime_generator : ∃! p : ℕ, Prime (p + 10) ∧ Prime (p + 14) :=
  ⟨3, 
    by {
      sorry -- Proof that 3 satisfies the conditions
    },
    by {
      sorry -- Proof that 3 is the only natural number satisfying the conditions
    }
  ⟩

end unique_prime_generator_l2619_261987


namespace largest_fraction_l2619_261913

theorem largest_fraction : 
  let fractions := [2/5, 3/7, 4/9, 3/8, 9/20]
  ∀ f ∈ fractions, (9:ℚ)/20 ≥ f := by sorry

end largest_fraction_l2619_261913


namespace fraction_inequality_l2619_261953

theorem fraction_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end fraction_inequality_l2619_261953


namespace division_problem_l2619_261911

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 55053 → 
  divisor = 456 → 
  remainder = 333 → 
  dividend = divisor * quotient + remainder → 
  quotient = 120 := by
sorry

end division_problem_l2619_261911


namespace ellipse_perpendicular_distance_l2619_261997

/-- The ellipse with equation 9x² + 16y² = 114 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | 9 * p.1^2 + 16 * p.2^2 = 114}

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The distance from a point to a line defined by two points -/
noncomputable def distanceToLine (p q r : ℝ × ℝ) : ℝ :=
  sorry

theorem ellipse_perpendicular_distance :
  ∀ (P Q : ℝ × ℝ),
  P ∈ Ellipse →
  Q ∈ Ellipse →
  (P.1 - O.1) * (Q.1 - O.1) + (P.2 - O.2) * (Q.2 - O.2) = 0 →
  distanceToLine O P Q = 12/5 := by
    sorry

end ellipse_perpendicular_distance_l2619_261997


namespace hyperbola_sufficient_condition_l2619_261991

-- Define the equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / (4 - m) = 1

-- Define the condition for a hyperbola with foci on the x-axis
def is_hyperbola_x_axis (m : ℝ) : Prop :=
  m - 1 > 0 ∧ 4 - m < 0

-- The theorem to prove
theorem hyperbola_sufficient_condition :
  ∃ (m : ℝ), m > 5 → is_hyperbola_x_axis m ∧
  ∃ (m' : ℝ), is_hyperbola_x_axis m' ∧ m' ≤ 5 :=
sorry

end hyperbola_sufficient_condition_l2619_261991


namespace new_customers_calculation_l2619_261933

theorem new_customers_calculation (initial_customers final_customers : ℕ) :
  initial_customers = 3 →
  final_customers = 8 →
  final_customers - initial_customers = 5 := by
  sorry

end new_customers_calculation_l2619_261933


namespace max_value_of_trigonometric_function_l2619_261941

theorem max_value_of_trigonometric_function :
  let f : ℝ → ℝ := λ x => Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)
  let S : Set ℝ := {x | -5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 3}
  ∃ x₀ ∈ S, ∀ x ∈ S, f x ≤ f x₀ ∧ f x₀ = 11 * Real.sqrt 3 / 6 := by
  sorry

end max_value_of_trigonometric_function_l2619_261941


namespace abs_positive_for_nonzero_l2619_261946

theorem abs_positive_for_nonzero (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end abs_positive_for_nonzero_l2619_261946


namespace invalid_vote_percentage_l2619_261935

theorem invalid_vote_percentage
  (total_votes : ℕ)
  (candidate_A_share : ℚ)
  (candidate_A_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_A_share = 60 / 100)
  (h3 : candidate_A_votes = 285600) :
  (total_votes - (candidate_A_votes / candidate_A_share : ℚ)) / total_votes = 15 / 100 :=
by sorry

end invalid_vote_percentage_l2619_261935


namespace fixed_point_range_l2619_261967

/-- The problem statement translated to Lean 4 --/
theorem fixed_point_range (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) (hm1 : m ≠ 1) :
  (∃ (x y : ℝ), (2 * a * x - b * y + 14 = 0) ∧ 
                (y = m^(x + 1) + 1) ∧ 
                ((x - a + 1)^2 + (y + b - 2)^2 ≤ 25)) →
  (3 / 4 : ℝ) ≤ b / a ∧ b / a ≤ (4 / 3 : ℝ) := by
  sorry

end fixed_point_range_l2619_261967


namespace stock_percentage_value_l2619_261956

/-- Calculates the percentage value of a stock given its yield and price. -/
def percentageValue (yield : ℝ) (price : ℝ) : ℝ :=
  yield * 100

theorem stock_percentage_value :
  let yield : ℝ := 0.10
  let price : ℝ := 80
  percentageValue yield price = 10 := by
  sorry

end stock_percentage_value_l2619_261956


namespace vegetarian_eaters_count_family_total_check_l2619_261927

/-- Represents the eating habits distribution in a family -/
structure FamilyEatingHabits where
  total : Nat
  onlyVegetarian : Nat
  onlyNonVegetarian : Nat
  both : Nat
  pescatarian : Nat
  vegan : Nat

/-- Calculates the number of people eating vegetarian food -/
def vegetarianEaters (habits : FamilyEatingHabits) : Nat :=
  habits.onlyVegetarian + habits.both + habits.vegan

/-- The given family's eating habits -/
def familyHabits : FamilyEatingHabits := {
  total := 40
  onlyVegetarian := 16
  onlyNonVegetarian := 12
  both := 8
  pescatarian := 3
  vegan := 1
}

/-- Theorem: The number of vegetarian eaters in the family is 25 -/
theorem vegetarian_eaters_count :
  vegetarianEaters familyHabits = 25 := by
  sorry

/-- Theorem: The sum of all eating habit categories equals the total family members -/
theorem family_total_check :
  familyHabits.onlyVegetarian + familyHabits.onlyNonVegetarian + familyHabits.both +
  familyHabits.pescatarian + familyHabits.vegan = familyHabits.total := by
  sorry

end vegetarian_eaters_count_family_total_check_l2619_261927


namespace arithmetic_sequence_property_l2619_261992

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) (m : ℕ) :
  arithmetic_sequence a d →
  d ≠ 0 →
  a 3 + a 6 + a 10 + a 13 = 32 →
  a m = 8 →
  m = 8 :=
by
  sorry

end arithmetic_sequence_property_l2619_261992


namespace merry_go_round_cost_per_child_l2619_261998

/-- The cost of a merry-go-round ride per child given the following conditions:
  - There are 5 children
  - 3 children rode the Ferris wheel
  - Ferris wheel cost is $5 per child
  - Everyone rode the merry-go-round
  - Each child bought 2 ice cream cones
  - Each ice cream cone costs $8
  - Total spent is $110
-/
theorem merry_go_round_cost_per_child 
  (num_children : ℕ)
  (ferris_wheel_riders : ℕ)
  (ferris_wheel_cost : ℚ)
  (ice_cream_cones_per_child : ℕ)
  (ice_cream_cone_cost : ℚ)
  (total_spent : ℚ)
  (h1 : num_children = 5)
  (h2 : ferris_wheel_riders = 3)
  (h3 : ferris_wheel_cost = 5)
  (h4 : ice_cream_cones_per_child = 2)
  (h5 : ice_cream_cone_cost = 8)
  (h6 : total_spent = 110) :
  (total_spent - (ferris_wheel_riders * ferris_wheel_cost) - (num_children * ice_cream_cones_per_child * ice_cream_cone_cost)) / num_children = 3 :=
by sorry

end merry_go_round_cost_per_child_l2619_261998


namespace triangle_max_area_l2619_261940

/-- Given a triangle ABC with side lengths a, b, c, where c = 1,
    and area S = (a^2 + b^2 - 1) / 4,
    prove that the maximum value of S is (√2 + 1) / 4 -/
theorem triangle_max_area (a b : ℝ) (h_c : c = 1) 
  (h_area : (a^2 + b^2 - 1) / 4 = (1/2) * a * b * Real.sin C) :
  (∃ (S : ℝ), S = (a^2 + b^2 - 1) / 4 ∧ 
    (∀ (S' : ℝ), S' = (a'^2 + b'^2 - 1) / 4 → S' ≤ S)) →
  (a^2 + b^2 - 1) / 4 ≤ (Real.sqrt 2 + 1) / 4 :=
sorry

end triangle_max_area_l2619_261940


namespace quadratic_equations_root_difference_l2619_261984

theorem quadratic_equations_root_difference (k : ℝ) : 
  (∀ x, x^2 + k*x + 6 = 0 → ∃ y, y^2 - k*y + 6 = 0 ∧ y = x + 5) →
  (∀ y, y^2 - k*y + 6 = 0 → ∃ x, x^2 + k*x + 6 = 0 ∧ y = x + 5) →
  k = 5 := by
sorry

end quadratic_equations_root_difference_l2619_261984


namespace calculation_proof_l2619_261942

theorem calculation_proof : 
  Real.sqrt 3 * (Real.sqrt 3 + 2) - 2 * Real.tan (60 * π / 180) + (-1) ^ 2023 = 2 + Real.sqrt 3 := by
sorry

end calculation_proof_l2619_261942


namespace second_term_is_negative_x_cubed_l2619_261903

/-- A line on a two-dimensional coordinate plane defined by a = x^2 - x^3 -/
def line (x : ℝ) : ℝ := x^2 - x^3

/-- The line touches the x-axis in 2 places -/
axiom touches_x_axis_twice : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ line x₁ = 0 ∧ line x₂ = 0

/-- The second term of the equation representing the line is -x^3 -/
theorem second_term_is_negative_x_cubed :
  ∃ f : ℝ → ℝ, (∀ x, line x = f x - x^3) ∧ (∀ x, f x = x^2) :=
sorry

end second_term_is_negative_x_cubed_l2619_261903


namespace elephant_weight_l2619_261916

theorem elephant_weight (elephant_weight : ℝ) (donkey_weight : ℝ) : 
  elephant_weight * 2000 + donkey_weight = 6600 →
  donkey_weight = 0.1 * (elephant_weight * 2000) →
  elephant_weight = 3 := by
sorry

end elephant_weight_l2619_261916


namespace tan_sqrt_three_solution_l2619_261909

theorem tan_sqrt_three_solution (x : ℝ) : 
  Real.tan x = Real.sqrt 3 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 3 := by
sorry

end tan_sqrt_three_solution_l2619_261909


namespace line_perpendicular_condition_l2619_261975

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)
variable (linePerpPlane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_condition
  (a b : Line) (α β : Plane)
  (h1 : lineInPlane a α)
  (h2 : linePerpPlane b β)
  (h3 : parallel α β) :
  perpendicular a b :=
sorry

end line_perpendicular_condition_l2619_261975


namespace art_group_size_l2619_261981

/-- The number of students in the art interest group -/
def num_students : ℕ := 6

/-- The total number of colored papers when each student cuts 10 pieces -/
def total_papers_10 (x : ℕ) : ℕ := 10 * x + 6

/-- The total number of colored papers when each student cuts 12 pieces -/
def total_papers_12 (x : ℕ) : ℕ := 12 * x - 6

/-- Theorem stating that the number of students satisfies the given conditions -/
theorem art_group_size :
  total_papers_10 num_students = total_papers_12 num_students :=
by sorry

end art_group_size_l2619_261981


namespace repeating_decimal_sum_difference_l2619_261963

theorem repeating_decimal_sum_difference (x y z : ℚ) : 
  (x = 246 / 999) → 
  (y = 135 / 999) → 
  (z = 579 / 999) → 
  x - y + z = 230 / 333 := by
sorry

end repeating_decimal_sum_difference_l2619_261963


namespace min_reciprocal_sum_l2619_261994

theorem min_reciprocal_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 3) :
  (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x + y + z = 3 → 1/x + 1/y + 1/z ≥ 1/a + 1/b + 1/c) →
  1/a + 1/b + 1/c = 3 :=
sorry

end min_reciprocal_sum_l2619_261994


namespace cylinder_radius_problem_l2619_261918

/-- Given a cylinder with original height 3 inches, if increasing the radius by 4 inches
    and the height by 6 inches results in the same new volume, then the original radius
    is 2 + 2√3 inches. -/
theorem cylinder_radius_problem (r : ℝ) : 
  (3 * π * (r + 4)^2 = 9 * π * r^2) → r = 2 + 2 * Real.sqrt 3 := by
  sorry

end cylinder_radius_problem_l2619_261918


namespace mode_of_student_dishes_l2619_261902

def student_dishes : List ℕ := [3, 5, 4, 6, 3, 3, 4]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_student_dishes :
  mode student_dishes = 3 := by sorry

end mode_of_student_dishes_l2619_261902


namespace square_sum_reciprocal_l2619_261917

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 4) : x^2 + (1 / x^2) = 14 := by
  sorry

end square_sum_reciprocal_l2619_261917


namespace complex_fraction_simplification_l2619_261972

theorem complex_fraction_simplification : 
  ((12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) * (72^4 + 324)) / 
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) * (66^4 + 324)) = 313 := by
  sorry

end complex_fraction_simplification_l2619_261972


namespace selection_methods_five_three_two_l2619_261971

/-- The number of ways to select 3 students out of 5 for 3 different language majors,
    where 2 specific students cannot be selected for one particular major -/
def selection_methods (n : ℕ) (k : ℕ) (excluded : ℕ) : ℕ :=
  Nat.choose (n - excluded) 1 * (n - 1).factorial / (n - k).factorial

theorem selection_methods_five_three_two :
  selection_methods 5 3 2 = 36 := by
  sorry

end selection_methods_five_three_two_l2619_261971


namespace joes_haircuts_l2619_261980

/-- The number of women's haircuts Joe did -/
def womens_haircuts : ℕ := sorry

/-- The time it takes to cut a woman's hair in minutes -/
def womens_haircut_time : ℕ := 50

/-- The time it takes to cut a man's hair in minutes -/
def mens_haircut_time : ℕ := 15

/-- The time it takes to cut a kid's hair in minutes -/
def kids_haircut_time : ℕ := 25

/-- The number of men's haircuts Joe did -/
def mens_haircuts : ℕ := 2

/-- The number of kids' haircuts Joe did -/
def kids_haircuts : ℕ := 3

/-- The total time Joe spent cutting hair in minutes -/
def total_time : ℕ := 255

theorem joes_haircuts : womens_haircuts = 3 := by sorry

end joes_haircuts_l2619_261980


namespace max_difference_two_digit_numbers_l2619_261976

theorem max_difference_two_digit_numbers :
  ∀ (A B : ℕ),
  (10 ≤ A ∧ A ≤ 99) →
  (10 ≤ B ∧ B ≤ 99) →
  (2 * A = 7 * B / 3) →
  (∀ (C D : ℕ), (10 ≤ C ∧ C ≤ 99) → (10 ≤ D ∧ D ≤ 99) → (2 * C = 7 * D / 3) → (C - D ≤ A - B)) →
  A - B = 56 :=
by sorry

end max_difference_two_digit_numbers_l2619_261976


namespace geometric_sequence_n_l2619_261974

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem geometric_sequence_n (a₁ q : ℝ) (n : ℕ) :
  a₁ = 1 → q = 2 → geometric_sequence a₁ q n = 64 → n = 7 := by
  sorry

end geometric_sequence_n_l2619_261974


namespace solution_set_inequality_l2619_261901

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) / (x + 2) ≤ 0 ↔ -2 < x ∧ x ≤ 3 :=
by sorry

end solution_set_inequality_l2619_261901


namespace ratio_equality_l2619_261914

theorem ratio_equality (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) :
  (x - 2*y + 3*z) / (x + y + z) = 8 / 9 := by
  sorry

end ratio_equality_l2619_261914


namespace log_equality_implies_ratio_l2619_261920

-- Define the conditions
theorem log_equality_implies_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 18) ∧
  (Real.log p / Real.log 8 = Real.log (p + q) / Real.log 32) →
  q / p = (4 + Real.sqrt 41) / 5 := by
  sorry

-- The proof is omitted as per instructions

end log_equality_implies_ratio_l2619_261920


namespace modified_cube_surface_area_l2619_261919

/-- Represents the structure of the cube after modifications -/
structure ModifiedCube where
  initialSize : Nat
  smallCubeSize : Nat
  removedCubes : Nat
  remainingCubes : Nat

/-- Calculates the surface area of the modified cube structure -/
def surfaceArea (cube : ModifiedCube) : Nat :=
  sorry

/-- Theorem stating that the surface area of the specific modified cube is 2820 -/
theorem modified_cube_surface_area :
  let cube : ModifiedCube := {
    initialSize := 12,
    smallCubeSize := 3,
    removedCubes := 14,
    remainingCubes := 50
  }
  surfaceArea cube = 2820 := by
  sorry

end modified_cube_surface_area_l2619_261919


namespace kendall_driving_distance_l2619_261923

/-- The distance Kendall drove with her mother in miles -/
def mother_distance : ℝ := 0.17

/-- The distance Kendall drove with her father in miles -/
def father_distance : ℝ := 0.5

/-- The distance Kendall drove with her friend in miles -/
def friend_distance : ℝ := 0.68

/-- The conversion factor from miles to kilometers -/
def mile_to_km : ℝ := 1.60934

/-- The total distance Kendall drove in kilometers -/
def total_distance_km : ℝ := (mother_distance + father_distance + friend_distance) * mile_to_km

theorem kendall_driving_distance :
  ∃ ε > 0, |total_distance_km - 2.17| < ε :=
sorry

end kendall_driving_distance_l2619_261923


namespace find_y_l2619_261938

theorem find_y : ∃ y : ℚ, 3 + 1 / (2 - y) = 2 * (1 / (2 - y)) → y = 5 / 3 := by
  sorry

end find_y_l2619_261938


namespace number_of_cartons_l2619_261944

/-- Represents the number of boxes in a carton -/
def boxes_per_carton : ℕ := 12

/-- Represents the number of packs in a box -/
def packs_per_box : ℕ := 10

/-- Represents the price of a pack in dollars -/
def price_per_pack : ℕ := 1

/-- Represents the total cost for all cartons in dollars -/
def total_cost : ℕ := 1440

/-- Theorem stating that the number of cartons is 12 -/
theorem number_of_cartons : 
  (total_cost : ℚ) / (boxes_per_carton * packs_per_box * price_per_pack) = 12 := by
  sorry

end number_of_cartons_l2619_261944


namespace line_equation_proof_l2619_261990

/-- A line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if a vector is normal to a line -/
def isNormalVector (l : Line2D) (v : Vector2D) : Prop :=
  l.a * v.x + l.b * v.y = 0

/-- The main theorem -/
theorem line_equation_proof (l : Line2D) (A : Point2D) (n : Vector2D) :
  l.a = 2 ∧ l.b = 1 ∧ l.c = 2 →
  A.x = 1 ∧ A.y = 0 →
  n.x = 2 ∧ n.y = -1 →
  pointOnLine l A ∧ isNormalVector l n := by
  sorry

#check line_equation_proof

end line_equation_proof_l2619_261990


namespace three_digit_sum_not_always_three_digits_l2619_261999

theorem three_digit_sum_not_always_three_digits : ∃ (a b : ℕ), 
  100 ≤ a ∧ a ≤ 999 ∧ 100 ≤ b ∧ b ≤ 999 ∧ 1000 ≤ a + b :=
by sorry

end three_digit_sum_not_always_three_digits_l2619_261999


namespace brenda_baking_days_l2619_261939

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

end brenda_baking_days_l2619_261939


namespace max_self_intersections_specific_cases_max_self_intersections_formula_l2619_261905

/-- Maximum number of self-intersection points for a closed polygonal chain -/
def max_self_intersections (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    n * (n - 3) / 2
  else
    n * (n - 4) / 2 + 1

/-- Theorem stating the maximum number of self-intersection points for specific cases -/
theorem max_self_intersections_specific_cases :
  (max_self_intersections 13 = 65) ∧ (max_self_intersections 1950 = 1898851) := by
  sorry

/-- Theorem for the general formula of maximum self-intersection points -/
theorem max_self_intersections_formula (n : ℕ) (h : n > 2) :
  max_self_intersections n = 
    if n % 2 = 1 then
      n * (n - 3) / 2
    else
      n * (n - 4) / 2 + 1 := by
  sorry

end max_self_intersections_specific_cases_max_self_intersections_formula_l2619_261905


namespace circumcircle_radius_right_triangle_l2619_261962

/-- The radius of the circumcircle of a triangle with side lengths 8, 15, and 17 is 17/2 -/
theorem circumcircle_radius_right_triangle : 
  ∀ (a b c : ℝ), 
  a = 8 → b = 15 → c = 17 →
  a^2 + b^2 = c^2 →
  (∃ (r : ℝ), r = c / 2 ∧ r = 17 / 2) :=
by sorry

end circumcircle_radius_right_triangle_l2619_261962


namespace constant_expression_l2619_261979

theorem constant_expression (x y : ℝ) (h : x + y = 1) :
  let a := Real.sqrt (1 + x^2)
  let b := Real.sqrt (1 + y^2)
  (a + b + 1) * (a + b - 1) * (a - b + 1) * (-a + b + 1) = 4 := by
  sorry

end constant_expression_l2619_261979


namespace largest_perimeter_l2619_261957

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℤ

/-- Checks if the given lengths can form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℝ :=
  t.side1 + t.side2 + t.side3

/-- Theorem stating the largest possible perimeter for the given triangle --/
theorem largest_perimeter :
  ∃ (t : Triangle), t.side1 = 8 ∧ t.side2 = 12 ∧ is_valid_triangle t ∧
  ∀ (t' : Triangle), t'.side1 = 8 ∧ t'.side2 = 12 ∧ is_valid_triangle t' →
  perimeter t ≥ perimeter t' ∧ perimeter t = 39 :=
sorry

end largest_perimeter_l2619_261957


namespace rationalize_denominator_l2619_261928

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) : ℝ) =
    (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = 3 ∧ B = -9 ∧ C = -9 ∧ D = 9 ∧ E = 165 ∧ F = 51 :=
by sorry

end rationalize_denominator_l2619_261928


namespace peggy_record_count_l2619_261960

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

end peggy_record_count_l2619_261960


namespace unique_solution_system_l2619_261993

theorem unique_solution_system :
  ∃! (x y z : ℝ), x + 3 * y = 10 ∧ y = 3 ∧ 2 * x - y + z = 7 := by
  sorry

end unique_solution_system_l2619_261993


namespace first_quartile_of_list_l2619_261996

def number_list : List ℝ := [42, 24, 30, 22, 26, 27, 33, 35]

def median (l : List ℝ) : ℝ := sorry

def first_quartile (l : List ℝ) : ℝ :=
  let m := median l
  median (l.filter (λ x => x < m))

theorem first_quartile_of_list :
  first_quartile number_list = 25 := by sorry

end first_quartile_of_list_l2619_261996


namespace smallest_special_number_l2619_261926

theorem smallest_special_number : ∃ (n : ℕ), 
  (100 ≤ n ∧ n < 1000) ∧ 
  (∃ (k : ℕ), n = 2 * k) ∧
  (∃ (k : ℕ), n + 1 = 3 * k) ∧
  (∃ (k : ℕ), n + 2 = 4 * k) ∧
  (∃ (k : ℕ), n + 3 = 5 * k) ∧
  (∃ (k : ℕ), n + 4 = 6 * k) ∧
  (∀ m : ℕ, m < n → 
    ¬((100 ≤ m ∧ m < 1000) ∧ 
      (∃ (k : ℕ), m = 2 * k) ∧
      (∃ (k : ℕ), m + 1 = 3 * k) ∧
      (∃ (k : ℕ), m + 2 = 4 * k) ∧
      (∃ (k : ℕ), m + 3 = 5 * k) ∧
      (∃ (k : ℕ), m + 4 = 6 * k))) ∧
  n = 122 :=
by sorry

end smallest_special_number_l2619_261926


namespace diego_fruit_weight_l2619_261952

/-- The total weight of fruit Diego can carry in his bookbag -/
def total_weight (watermelon grapes oranges apples : ℕ) : ℕ :=
  watermelon + grapes + oranges + apples

/-- Theorem stating the total weight of fruit Diego can carry -/
theorem diego_fruit_weight :
  ∃ (watermelon grapes oranges apples : ℕ),
    watermelon = 1 ∧ grapes = 1 ∧ oranges = 1 ∧ apples = 17 ∧
    total_weight watermelon grapes oranges apples = 20 := by
  sorry

end diego_fruit_weight_l2619_261952


namespace exists_real_a_sqrt3_minus_a_real_l2619_261958

theorem exists_real_a_sqrt3_minus_a_real : ∃ a : ℝ, ∃ b : ℝ, b = Real.sqrt 3 - a := by
  sorry

end exists_real_a_sqrt3_minus_a_real_l2619_261958


namespace tangent_line_quadratic_l2619_261968

/-- Given a quadratic function f(x) = x² + ax + b, if the tangent line
    to f at x = 0 is x - y + 1 = 0, then a = 1 and b = 1 -/
theorem tangent_line_quadratic (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  let f' : ℝ → ℝ := λ x ↦ 2*x + a
  (∀ x, f' x = (deriv f) x) →
  (f' 0 = 1) →
  (f 0 = 1) →
  a = 1 ∧ b = 1 := by
sorry

end tangent_line_quadratic_l2619_261968


namespace coin_selection_probability_l2619_261906

/-- Represents the placement of boxes in drawers -/
inductive BoxPlacement
  | AloneInDrawer
  | WithOneOther
  | Random

/-- Probability of selecting the coin-containing box given a placement -/
def probability (placement : BoxPlacement) : ℚ :=
  match placement with
  | BoxPlacement.AloneInDrawer => 1/2
  | BoxPlacement.WithOneOther => 1/4
  | BoxPlacement.Random => 1/3

theorem coin_selection_probability 
  (boxes : Nat) 
  (drawers : Nat) 
  (coin_box : Nat) 
  (h1 : boxes = 3) 
  (h2 : drawers = 2) 
  (h3 : coin_box = 1) 
  (h4 : ∀ d, d ≤ drawers → d > 0 → ∃ b, b ≤ boxes ∧ b > 0) :
  (probability BoxPlacement.AloneInDrawer = 1/2) ∧
  (probability BoxPlacement.WithOneOther = 1/4) ∧
  (probability BoxPlacement.Random = 1/3) := by
  sorry

end coin_selection_probability_l2619_261906


namespace polynomial_expansion_l2619_261983

theorem polynomial_expansion (t : ℝ) : 
  (3 * t^3 - 4 * t + 1) * (4 * t^2 - 5 * t + 3) = 
  12 * t^5 - 15 * t^4 - 7 * t^3 + 24 * t^2 - 17 * t + 3 := by
sorry

end polynomial_expansion_l2619_261983


namespace sqrt_product_plus_one_equals_869_l2619_261945

theorem sqrt_product_plus_one_equals_869 : 
  Real.sqrt ((31 * 30 * 29 * 28) + 1) = 869 := by
  sorry

end sqrt_product_plus_one_equals_869_l2619_261945


namespace distance_swam_against_current_l2619_261954

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

end distance_swam_against_current_l2619_261954


namespace range_of_a_l2619_261961

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 2) ↔ ((x - a)^2 < 1)) ↔ 
  (1 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l2619_261961


namespace train_A_speed_l2619_261910

/-- The speed of Train A in miles per hour -/
def speed_A : ℝ := 30

/-- The speed of Train B in miles per hour -/
def speed_B : ℝ := 38

/-- The time difference between Train A and Train B departures in hours -/
def time_diff : ℝ := 2

/-- The distance from the station where Train B overtakes Train A in miles -/
def overtake_distance : ℝ := 285

/-- Theorem stating that the speed of Train A is 30 miles per hour -/
theorem train_A_speed :
  speed_A = 30 ∧
  speed_A * (overtake_distance / speed_B + time_diff) = overtake_distance :=
sorry

end train_A_speed_l2619_261910


namespace acid_dilution_l2619_261959

/-- Given m ounces of m% acid solution, adding x ounces of water yields (m-10)% solution -/
theorem acid_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m^2 / 100 = (m - 10) / 100 * (m + x)) → x = 10 * m / (m - 10) := by sorry

end acid_dilution_l2619_261959


namespace expression_value_at_two_l2619_261986

theorem expression_value_at_two :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 4)
  f 2 = 6 := by sorry

end expression_value_at_two_l2619_261986


namespace x_plus_y_value_l2619_261932

theorem x_plus_y_value (x y : Real) 
  (eq1 : x + Real.sin y = 2023)
  (eq2 : x + 2023 * Real.cos y = 2022)
  (y_range : π/2 ≤ y ∧ y ≤ π) :
  x + y = 2022 + π/2 := by
  sorry

end x_plus_y_value_l2619_261932


namespace intersection_implies_a_value_l2619_261964

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 2*a-1, a^2+1}
  A ∩ B = {-3} → a = -1 := by
sorry

end intersection_implies_a_value_l2619_261964


namespace unique_value_of_expression_l2619_261955

theorem unique_value_of_expression (x y : ℝ) 
  (h : x * y - 3 * x / (y^2) - 3 * y / (x^2) = 7) : 
  (x - 2) * (y - 2) = 1 := by
  sorry

end unique_value_of_expression_l2619_261955
