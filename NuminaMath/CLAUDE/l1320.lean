import Mathlib

namespace grade_distribution_l1320_132003

theorem grade_distribution (total_students : ℝ) (prob_A prob_B prob_C : ℝ) :
  total_students = 40 →
  prob_A = 0.6 * prob_B →
  prob_C = 1.5 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  prob_B * total_students = 40 / 3.1 :=
by sorry

end grade_distribution_l1320_132003


namespace difference_given_sum_and_difference_of_squares_l1320_132018

theorem difference_given_sum_and_difference_of_squares (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end difference_given_sum_and_difference_of_squares_l1320_132018


namespace john_total_height_climbed_l1320_132028

/-- Calculates the total height climbed by John given the number of flights, 
    height per flight, and additional climbing information. -/
def totalHeightClimbed (numFlights : ℕ) (heightPerFlight : ℕ) : ℕ :=
  let stairsHeight := numFlights * heightPerFlight
  let ropeHeight := stairsHeight / 2
  let ladderHeight := ropeHeight + 10
  stairsHeight + ropeHeight + ladderHeight

/-- Theorem stating that the total height climbed by John is 70 feet. -/
theorem john_total_height_climbed :
  totalHeightClimbed 3 10 = 70 := by
  sorry

end john_total_height_climbed_l1320_132028


namespace golden_man_poem_analysis_correct_l1320_132073

/-- Represents a poem --/
structure Poem where
  content : String
  deriving Repr

/-- Represents the analysis of a poem --/
structure PoemAnalysis where
  sentimentality_reasons : List String
  artistic_techniques : List String
  deriving Repr

/-- Function to analyze a poem --/
def analyze_poem (p : Poem) : PoemAnalysis :=
  { sentimentality_reasons := ["humiliating mission", "decline of homeland", "aging"],
    artistic_techniques := ["using scenery to express emotions"] }

/-- The poem in question --/
def golden_man_poem : Poem :=
  { content := "Recalling the divine capital, a bustling place, where I once roamed..." }

/-- Theorem stating that the analysis of the golden_man_poem is correct --/
theorem golden_man_poem_analysis_correct :
  analyze_poem golden_man_poem =
    { sentimentality_reasons := ["humiliating mission", "decline of homeland", "aging"],
      artistic_techniques := ["using scenery to express emotions"] } := by
  sorry


end golden_man_poem_analysis_correct_l1320_132073


namespace negation_of_existence_l1320_132010

theorem negation_of_existence (x : ℝ) :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by sorry

end negation_of_existence_l1320_132010


namespace x_value_l1320_132067

theorem x_value (x y : ℚ) (h1 : x / y = 5 / 2) (h2 : y = 30) : x = 75 := by
  sorry

end x_value_l1320_132067


namespace walmart_shelving_problem_l1320_132048

/-- Given a total number of pots and the capacity of each shelf,
    calculate the number of shelves needed to stock all pots. -/
def shelves_needed (total_pots : ℕ) (vertical_capacity : ℕ) (horizontal_capacity : ℕ) : ℕ :=
  (total_pots + vertical_capacity * horizontal_capacity - 1) / (vertical_capacity * horizontal_capacity)

/-- Proof that 4 shelves are needed to stock 60 pots when each shelf can hold 
    5 vertically stacked pots in 3 side-by-side sets. -/
theorem walmart_shelving_problem : shelves_needed 60 5 3 = 4 := by
  sorry

end walmart_shelving_problem_l1320_132048


namespace complex_fraction_pure_imaginary_l1320_132058

theorem complex_fraction_pure_imaginary (a : ℝ) : 
  (Complex.I * (Complex.I * (a + 1) + (1 - a)) = Complex.I * (a + Complex.I) / (1 + Complex.I)) → 
  a = -1 :=
by sorry

end complex_fraction_pure_imaginary_l1320_132058


namespace problem_solution_l1320_132097

theorem problem_solution : (-1 : ℚ)^51 + 2^(4^2 + 5^2 - 7^2) = -127/128 := by
  sorry

end problem_solution_l1320_132097


namespace greatest_divisor_with_remainders_l1320_132006

theorem greatest_divisor_with_remainders : 
  ∃ (n : ℕ), n > 0 ∧ 
  (178340 % n = 20) ∧ 
  (253785 % n = 35) ∧ 
  (375690 % n = 50) ∧ 
  (∀ m : ℕ, m > 0 → 
    (178340 % m = 20) → 
    (253785 % m = 35) → 
    (375690 % m = 50) → 
    m ≤ n) ∧
  n = 10 := by
sorry

end greatest_divisor_with_remainders_l1320_132006


namespace train_passing_station_time_l1320_132031

/-- The time taken for a train to pass a station -/
theorem train_passing_station_time
  (train_length : Real)
  (train_speed_kmh : Real)
  (station_length : Real)
  (h1 : train_length = 250)
  (h2 : train_speed_kmh = 36)
  (h3 : station_length = 200) :
  (train_length + station_length) / (train_speed_kmh * 1000 / 3600) = 45 := by
  sorry

#check train_passing_station_time

end train_passing_station_time_l1320_132031


namespace triangle_area_l1320_132082

theorem triangle_area (A B C : Real) (a b c : Real) :
  (b = c * (2 * Real.sin A + Real.cos A)) →
  (a = Real.sqrt 2) →
  (B = 3 * Real.pi / 4) →
  (∃ (S : Real), S = (1 / 2) * a * c * Real.sin B ∧ S = 1) :=
by sorry

end triangle_area_l1320_132082


namespace first_discount_percentage_l1320_132027

theorem first_discount_percentage 
  (original_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) : 
  original_price = 400 →
  final_price = 342 →
  second_discount = 5 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (100 - first_discount) / 100 * (100 - second_discount) / 100 ∧
    first_discount = 10 := by
  sorry

end first_discount_percentage_l1320_132027


namespace count_special_quadrilaterals_l1320_132057

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  ab : ℕ+
  bc : ℕ+
  cd : ℕ+
  ad : ℕ+
  right_angle_b : True  -- Represents the right angle at B
  right_angle_c : True  -- Represents the right angle at C
  ab_eq_two : ab = 2
  cd_eq_ad : cd = ad

/-- The perimeter of a SpecialQuadrilateral -/
def perimeter (q : SpecialQuadrilateral) : ℕ :=
  q.ab + q.bc + q.cd + q.ad

/-- The theorem statement -/
theorem count_special_quadrilaterals :
  (∃ (s : Finset ℕ), s.card = 31 ∧
    (∀ p ∈ s, p < 2015 ∧ ∃ q : SpecialQuadrilateral, perimeter q = p) ∧
    (∀ p < 2015, (∃ q : SpecialQuadrilateral, perimeter q = p) → p ∈ s)) :=
sorry

end count_special_quadrilaterals_l1320_132057


namespace egg_weight_probability_l1320_132016

/-- Given that the probability of an egg's weight being less than 30 grams is 0.30,
    prove that the probability of its weight being not less than 30 grams is 0.70. -/
theorem egg_weight_probability (p_less_than_30 : ℝ) (h1 : p_less_than_30 = 0.30) :
  1 - p_less_than_30 = 0.70 := by
  sorry

end egg_weight_probability_l1320_132016


namespace f_is_even_iff_l1320_132034

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = ax^2 + (2a+1)x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + (2*a + 1) * x - 1

/-- Theorem: The function f is even if and only if a = -1/2 -/
theorem f_is_even_iff (a : ℝ) : IsEven (f a) ↔ a = -1/2 := by
  sorry


end f_is_even_iff_l1320_132034


namespace range_of_a_l1320_132030

def f (x : ℝ) := x^2 - 4*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4) a, f x ∈ Set.Icc (-4) 32) →
  (Set.Icc (-4) a = f ⁻¹' (Set.Icc (-4) 32)) →
  a ∈ Set.Icc 2 8 := by
  sorry

end range_of_a_l1320_132030


namespace blonde_to_total_ratio_l1320_132042

/-- Given a class with a specific hair color ratio and number of students, 
    prove the ratio of blonde-haired children to total children -/
theorem blonde_to_total_ratio 
  (red_ratio : ℕ) (blonde_ratio : ℕ) (black_ratio : ℕ)
  (red_count : ℕ) (total_count : ℕ)
  (h1 : red_ratio = 3)
  (h2 : blonde_ratio = 6)
  (h3 : black_ratio = 7)
  (h4 : red_count = 9)
  (h5 : total_count = 48)
  : (blonde_ratio * red_count / red_ratio) / total_count = 3 / 8 := by
  sorry

#check blonde_to_total_ratio

end blonde_to_total_ratio_l1320_132042


namespace absolute_value_inequality_solution_set_l1320_132036

theorem absolute_value_inequality_solution_set (x : ℝ) :
  (|x - 1| < 1) ↔ (x ∈ Set.Ioo 0 2) :=
sorry

end absolute_value_inequality_solution_set_l1320_132036


namespace curve_symmetrical_y_axis_l1320_132052

-- Define a function to represent the left-hand side of the equation
def f (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem stating that the curve is symmetrical with respect to the y-axis
theorem curve_symmetrical_y_axis : ∀ x y : ℝ, f x y = 1 ↔ f (-x) y = 1 := by
  sorry

end curve_symmetrical_y_axis_l1320_132052


namespace divisibility_by_fifteen_l1320_132050

theorem divisibility_by_fifteen (a : ℤ) :
  15 ∣ ((5 * a + 1) * (3 * a + 2)) ↔ a % 15 = 4 := by
  sorry

end divisibility_by_fifteen_l1320_132050


namespace max_value_x_plus_inverse_l1320_132081

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 := by
  sorry

end max_value_x_plus_inverse_l1320_132081


namespace complex_number_calculation_l1320_132019

theorem complex_number_calculation : 
  let z : ℂ := 1 + Complex.I * Real.sqrt 2
  z^2 - 2*z = -3 := by sorry

end complex_number_calculation_l1320_132019


namespace smallest_multiple_of_1_to_10_l1320_132092

def is_multiple_of_all (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i → i ≤ 10 → n % i = 0

theorem smallest_multiple_of_1_to_10 :
  ∃ (n : ℕ), n > 0 ∧ is_multiple_of_all n ∧ ∀ m : ℕ, m > 0 → is_multiple_of_all m → n ≤ m :=
by
  use 2520
  sorry

end smallest_multiple_of_1_to_10_l1320_132092


namespace chris_pears_equal_lily_apples_l1320_132079

/-- Represents the number of fruits in the box -/
structure FruitBox where
  apples : ℕ
  pears : ℕ
  apples_twice_pears : apples = 2 * pears

/-- Represents the distribution of fruits between Chris and Lily -/
structure FruitDistribution where
  box : FruitBox
  chris_apples : ℕ
  chris_pears : ℕ
  lily_apples : ℕ
  lily_pears : ℕ
  total_distributed : chris_apples + chris_pears + lily_apples + lily_pears = box.apples + box.pears
  chris_twice_lily : chris_apples + chris_pears = 2 * (lily_apples + lily_pears)

/-- Theorem stating that Chris took as many pears as Lily took apples -/
theorem chris_pears_equal_lily_apples (dist : FruitDistribution) : 
  dist.chris_pears = dist.lily_apples := by sorry

end chris_pears_equal_lily_apples_l1320_132079


namespace cab_driver_income_l1320_132099

theorem cab_driver_income (income_day1 income_day2 income_day3 income_day4 : ℕ)
  (average_income : ℕ) (total_days : ℕ) :
  income_day1 = 200 →
  income_day2 = 150 →
  income_day3 = 750 →
  income_day4 = 400 →
  average_income = 400 →
  total_days = 5 →
  (income_day1 + income_day2 + income_day3 + income_day4 + 
    (average_income * total_days - (income_day1 + income_day2 + income_day3 + income_day4))) / total_days = average_income →
  average_income * total_days - (income_day1 + income_day2 + income_day3 + income_day4) = 500 :=
by sorry

end cab_driver_income_l1320_132099


namespace lcm_of_25_35_50_l1320_132000

theorem lcm_of_25_35_50 : Nat.lcm 25 (Nat.lcm 35 50) = 350 := by sorry

end lcm_of_25_35_50_l1320_132000


namespace curve_intersects_median_unique_point_l1320_132088

/-- Given non-collinear points A, B, C with complex coordinates, 
    prove that the curve intersects the median of triangle ABC at a unique point. -/
theorem curve_intersects_median_unique_point 
  (a b c : ℝ) 
  (h_non_collinear : a + c ≠ 2*b) : 
  ∃! p : ℂ, 
    (∃ t : ℝ, p = Complex.I * a * (Real.cos t)^4 + 
               (1/2 + Complex.I * b) * 2 * (Real.cos t)^2 * (Real.sin t)^2 + 
               (1 + Complex.I * c) * (Real.sin t)^4) ∧ 
    (p.re = 1/2 ∧ p.im = (a + 2*b + c) / 4) := by
  sorry


end curve_intersects_median_unique_point_l1320_132088


namespace range_of_a_when_proposition_false_l1320_132011

theorem range_of_a_when_proposition_false :
  (¬ ∃ x₀ : ℝ, ∃ a : ℝ, a * x₀^2 - 2 * a * x₀ - 3 > 0) →
  (∀ a : ℝ, a ∈ Set.Icc (-3 : ℝ) 0) :=
by sorry

end range_of_a_when_proposition_false_l1320_132011


namespace shooting_stars_count_difference_l1320_132012

/-- The number of shooting stars counted by Bridget -/
def bridget_count : ℕ := 14

/-- The number of shooting stars counted by Reginald -/
def reginald_count : ℕ := 12

/-- The number of shooting stars counted by Sam -/
def sam_count : ℕ := reginald_count + 4

/-- The average number of shooting stars counted by all three -/
def average_count : ℚ := (bridget_count + reginald_count + sam_count) / 3

theorem shooting_stars_count_difference :
  sam_count = average_count + 2 →
  bridget_count - reginald_count = 2 := by
  sorry

#eval bridget_count - reginald_count

end shooting_stars_count_difference_l1320_132012


namespace min_sum_squares_l1320_132072

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) 
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0)
  (sum_cond : x₁ + 2*x₂ + 3*x₃ = 60) :
  x₁^2 + x₂^2 + x₃^2 ≥ 1800/7 ∧ 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 
    y₁ + 2*y₂ + 3*y₃ = 60 ∧ 
    y₁^2 + y₂^2 + y₃^2 = 1800/7 :=
by sorry

end min_sum_squares_l1320_132072


namespace midpoint_specific_segment_l1320_132017

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ := sorry

theorem midpoint_specific_segment :
  let p1 : ℝ × ℝ := (6, π/4)
  let p2 : ℝ × ℝ := (6, 3*π/4)
  let (r, θ) := polar_midpoint p1.1 p1.2 p2.1 p2.2
  r = 3 * Real.sqrt 2 ∧ θ = π/2 :=
sorry

end midpoint_specific_segment_l1320_132017


namespace consecutive_numbers_digit_sum_exists_l1320_132071

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Theorem statement
theorem consecutive_numbers_digit_sum_exists :
  ∃ n : ℕ, sumOfDigits n = 52 ∧ sumOfDigits (n + 4) = 20 :=
sorry

end consecutive_numbers_digit_sum_exists_l1320_132071


namespace loss_equals_five_balls_l1320_132002

/-- Prove that the number of balls the loss equates to is 5 -/
theorem loss_equals_five_balls 
  (cost_price : ℕ) 
  (num_balls_sold : ℕ) 
  (selling_price : ℕ) 
  (h1 : cost_price = 72)
  (h2 : num_balls_sold = 15)
  (h3 : selling_price = 720) :
  (num_balls_sold * cost_price - selling_price) / cost_price = 5 := by
  sorry

#check loss_equals_five_balls

end loss_equals_five_balls_l1320_132002


namespace day_of_week_p_minus_one_l1320_132053

-- Define a type for days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the day of the week for a given day number
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

-- Define the theorem
theorem day_of_week_p_minus_one (P : Nat) :
  dayOfWeek 250 = DayOfWeek.Sunday →
  dayOfWeek 150 = DayOfWeek.Sunday →
  dayOfWeek 50 = DayOfWeek.Sunday :=
by
  sorry

-- The proof is omitted as per instructions

end day_of_week_p_minus_one_l1320_132053


namespace ratio_p_to_q_l1320_132086

def total_ways : ℕ := 6^24

def ways_p : ℕ := Nat.choose 6 2 * Nat.choose 24 2 * Nat.choose 22 6 * 
                  Nat.choose 16 4 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4

def ways_q : ℕ := Nat.choose 6 2 * Nat.choose 24 3 * Nat.choose 21 3 * 
                  Nat.choose 18 4 * Nat.choose 14 4 * Nat.choose 10 4 * Nat.choose 6 4

def p : ℚ := ways_p / total_ways
def q : ℚ := ways_q / total_ways

theorem ratio_p_to_q : p / q = ways_p / ways_q := by
  sorry

end ratio_p_to_q_l1320_132086


namespace camp_girls_count_l1320_132047

theorem camp_girls_count (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 133 → difference = 33 → girls + (girls + difference) = total → girls = 50 := by
sorry

end camp_girls_count_l1320_132047


namespace football_team_numbers_l1320_132013

theorem football_team_numbers (x : ℕ) (n : ℕ) : 
  (n * (n + 1)) / 2 - x = 100 → x = 5 ∧ n = 14 := by
  sorry

end football_team_numbers_l1320_132013


namespace children_neither_happy_nor_sad_l1320_132038

theorem children_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neutral_boys : ℕ) : 
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  boys = 17 →
  girls = 43 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 5 →
  total_children - (happy_children + sad_children) = 20 := by
sorry

end children_neither_happy_nor_sad_l1320_132038


namespace f_odd_and_decreasing_l1320_132039

def f (x : ℝ) : ℝ := -3 * x

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
sorry

end f_odd_and_decreasing_l1320_132039


namespace percent_of_y_l1320_132090

theorem percent_of_y (y : ℝ) (h : y > 0) : ((7 * y) / 20 + (3 * y) / 10) / y = 0.65 := by
  sorry

end percent_of_y_l1320_132090


namespace inscribed_circle_radius_l1320_132023

/-- The radius of the inscribed circle of a triangle with sides 15, 16, and 17 is √21 -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 15) (hb : b = 16) (hc : c = 17) :
  let s := (a + b + c) / 2
  let r := Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s
  r = Real.sqrt 21 := by sorry

end inscribed_circle_radius_l1320_132023


namespace cube_pyramid_sum_is_34_l1320_132094

/-- Represents a three-dimensional shape --/
structure Shape3D where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- A cube --/
def cube : Shape3D :=
  { faces := 6, edges := 12, vertices := 8 }

/-- Adds a pyramid to one face of a given shape --/
def addPyramid (shape : Shape3D) : Shape3D :=
  { faces := shape.faces + 3,  -- One face is covered, 4 new faces added
    edges := shape.edges + 4,  -- 4 new edges from apex to base
    vertices := shape.vertices + 1 }  -- 1 new vertex (apex)

/-- Calculates the sum of faces, edges, and vertices --/
def sumComponents (shape : Shape3D) : Nat :=
  shape.faces + shape.edges + shape.vertices

/-- Theorem: The maximum sum of exterior faces, vertices, and edges
    of a shape formed by adding a pyramid to one face of a cube is 34 --/
theorem cube_pyramid_sum_is_34 :
  sumComponents (addPyramid cube) = 34 := by
  sorry

end cube_pyramid_sum_is_34_l1320_132094


namespace solution_set_is_two_lines_l1320_132075

/-- The solution set of the equation (2x - y)^2 = 4x^2 - y^2 -/
def SolutionSet : Set (ℝ × ℝ) :=
  {(x, y) | (2*x - y)^2 = 4*x^2 - y^2}

/-- The set consisting of two lines: y = 0 and y = 2x -/
def TwoLines : Set (ℝ × ℝ) :=
  {(x, y) | y = 0 ∨ y = 2*x}

/-- Theorem stating that the solution set of the equation is equivalent to two lines -/
theorem solution_set_is_two_lines : SolutionSet = TwoLines := by
  sorry

end solution_set_is_two_lines_l1320_132075


namespace power_of_two_problem_l1320_132007

theorem power_of_two_problem (a b : ℕ+) 
  (h1 : (2 ^ a.val) ^ b.val = 2 ^ 2)
  (h2 : 2 ^ a.val * 2 ^ b.val = 8) :
  2 ^ b.val = 4 := by
  sorry

end power_of_two_problem_l1320_132007


namespace greatest_prime_divisor_l1320_132078

def ribbon_lengths : List ℕ := [8, 16, 20, 28]

theorem greatest_prime_divisor (lengths : List ℕ) : 
  ∃ (n : ℕ), n.Prime ∧ 
  (∀ m : ℕ, m.Prime → (∀ l ∈ lengths, l % m = 0) → m ≤ n) ∧
  (∀ l ∈ lengths, l % n = 0) := by
  sorry

end greatest_prime_divisor_l1320_132078


namespace exists_prime_with_integer_roots_l1320_132043

theorem exists_prime_with_integer_roots :
  ∃ p : ℕ, Prime p ∧ 1 < p ∧ p ≤ 11 ∧
  ∃ x y : ℤ, x^2 + p*x - 720*p = 0 ∧ y^2 + p*y - 720*p = 0 :=
by sorry

end exists_prime_with_integer_roots_l1320_132043


namespace final_face_is_four_l1320_132022

/-- Represents a standard 6-sided die where opposite faces sum to 7 -/
structure StandardDie where
  faces : Fin 6 → Nat
  opposite_sum_seven : ∀ (f : Fin 6), faces f + faces (5 - f) = 7

/-- Represents a move direction -/
inductive Move
| Left
| Forward
| Right
| Back

/-- The sequence of moves in the path -/
def path : List Move := [Move.Left, Move.Forward, Move.Right, Move.Back, Move.Forward, Move.Back]

/-- Simulates rolling the die in a given direction -/
def roll (d : StandardDie) (m : Move) (top : Fin 6) : Fin 6 :=
  sorry

/-- Simulates rolling the die along the entire path -/
def rollPath (d : StandardDie) (initial : Fin 6) : Fin 6 :=
  sorry

/-- Theorem stating that the final top face is 4 regardless of initial state -/
theorem final_face_is_four (d : StandardDie) (initial : Fin 6) :
  d.faces (rollPath d initial) = 4 := by sorry

end final_face_is_four_l1320_132022


namespace scientific_notation_450_million_l1320_132065

theorem scientific_notation_450_million :
  (450000000 : ℝ) = 4.5 * (10 : ℝ)^8 := by
  sorry

end scientific_notation_450_million_l1320_132065


namespace magic_king_episodes_l1320_132014

theorem magic_king_episodes (total_seasons : ℕ) 
  (first_half_episodes : ℕ) (second_half_episodes : ℕ) : 
  total_seasons = 10 ∧ 
  first_half_episodes = 20 ∧ 
  second_half_episodes = 25 →
  (total_seasons / 2 * first_half_episodes) + 
  (total_seasons / 2 * second_half_episodes) = 225 := by
  sorry

end magic_king_episodes_l1320_132014


namespace inequality_system_solution_expression_factorization_l1320_132045

-- Part 1: System of inequalities
theorem inequality_system_solution (x : ℝ) :
  (2 * x + 1 ≤ 4 - x ∧ x - 1 < 3 * x / 2) ↔ (-2 < x ∧ x ≤ 1) := by sorry

-- Part 2: Expression factorization
theorem expression_factorization (a x y : ℝ) :
  a^2 * (x - y) + 4 * (y - x) = (x - y) * (a + 2) * (a - 2) := by sorry

end inequality_system_solution_expression_factorization_l1320_132045


namespace prob_odd_sum_is_four_ninths_l1320_132068

/-- Represents a skewed six-sided die where rolling an odd number is twice as likely as rolling an even number. -/
structure SkewedDie :=
  (prob_even : ℝ)
  (prob_odd : ℝ)
  (six_sided : Nat)
  (skew_condition : prob_odd = 2 * prob_even)
  (probability_sum : prob_even + prob_odd = 1)
  (six_sided_condition : six_sided = 6)

/-- The probability of rolling an odd sum when rolling the skewed die twice. -/
def prob_odd_sum (d : SkewedDie) : ℝ :=
  2 * d.prob_even * d.prob_odd

/-- Theorem stating that the probability of rolling an odd sum with the skewed die is 4/9. -/
theorem prob_odd_sum_is_four_ninths (d : SkewedDie) : 
  prob_odd_sum d = 4/9 := by
  sorry

end prob_odd_sum_is_four_ninths_l1320_132068


namespace exists_acute_triangle_l1320_132063

/-- Given five positive real numbers that can form triangles in any combination of three,
    there exists at least one acute-angled triangle among them. -/
theorem exists_acute_triangle
  (a b c d e : ℝ)
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) (pos_e : e > 0)
  (triangle_abc : a + b > c ∧ b + c > a ∧ c + a > b)
  (triangle_abd : a + b > d ∧ b + d > a ∧ d + a > b)
  (triangle_abe : a + b > e ∧ b + e > a ∧ e + a > b)
  (triangle_acd : a + c > d ∧ c + d > a ∧ d + a > c)
  (triangle_ace : a + c > e ∧ c + e > a ∧ e + a > c)
  (triangle_ade : a + d > e ∧ d + e > a ∧ e + a > d)
  (triangle_bcd : b + c > d ∧ c + d > b ∧ d + b > c)
  (triangle_bce : b + c > e ∧ c + e > b ∧ e + b > c)
  (triangle_bde : b + d > e ∧ d + e > b ∧ e + b > d)
  (triangle_cde : c + d > e ∧ d + e > c ∧ e + c > d) :
  ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                 (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                 (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
                 x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                 x^2 + y^2 > z^2 ∧ y^2 + z^2 > x^2 ∧ z^2 + x^2 > y^2 :=
sorry

end exists_acute_triangle_l1320_132063


namespace empty_solution_set_has_solutions_l1320_132049

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |3 - x| < a

-- Theorem 1: The solution set is empty iff a ≤ 1
theorem empty_solution_set (a : ℝ) :
  (∀ x : ℝ, ¬ inequality x a) ↔ a ≤ 1 := by sorry

-- Theorem 2: The inequality has solutions iff a > 1
theorem has_solutions (a : ℝ) :
  (∃ x : ℝ, inequality x a) ↔ a > 1 := by sorry

end empty_solution_set_has_solutions_l1320_132049


namespace additional_time_is_24_minutes_l1320_132026

/-- Time to fill one barrel normally (in minutes) -/
def normal_time : ℕ := 3

/-- Time to fill one barrel with leak (in minutes) -/
def leak_time : ℕ := 5

/-- Number of barrels to fill -/
def num_barrels : ℕ := 12

/-- Additional time required to fill barrels with leak -/
def additional_time : ℕ := (leak_time * num_barrels) - (normal_time * num_barrels)

theorem additional_time_is_24_minutes : additional_time = 24 := by
  sorry

end additional_time_is_24_minutes_l1320_132026


namespace original_number_proof_l1320_132056

theorem original_number_proof : ∃ n : ℕ, n + 1 = 30 ∧ n < 30 := by
  sorry

end original_number_proof_l1320_132056


namespace rachel_day_visitor_count_l1320_132029

/-- The number of visitors to Buckingham Palace over two days -/
def total_visitors : ℕ := 829

/-- The number of visitors to Buckingham Palace on the day before Rachel's visit -/
def previous_day_visitors : ℕ := 246

/-- The number of visitors to Buckingham Palace on the day of Rachel's visit -/
def rachel_day_visitors : ℕ := total_visitors - previous_day_visitors

theorem rachel_day_visitor_count : rachel_day_visitors = 583 := by
  sorry

end rachel_day_visitor_count_l1320_132029


namespace carol_peanuts_l1320_132015

theorem carol_peanuts (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 2 → received = 5 → total = initial + received → total = 7 := by
sorry

end carol_peanuts_l1320_132015


namespace inequality_holds_l1320_132037

theorem inequality_holds (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^3 * (y + 1) + y^3 * (x + 1) ≥ x^2 * (y + y^2) + y^2 * (x + x^2) := by
  sorry

end inequality_holds_l1320_132037


namespace polygon_with_120_degree_angles_is_hexagon_l1320_132070

theorem polygon_with_120_degree_angles_is_hexagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 120 →
    (n - 2) * 180 = n * interior_angle →
    n = 6 :=
by
  sorry

end polygon_with_120_degree_angles_is_hexagon_l1320_132070


namespace line_intersects_circle_l1320_132096

/-- The line y - 1 = k(x - 1) intersects the circle x^2 + y^2 - 2y = 0 for any real number k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  (y - 1 = k * (x - 1)) ∧ (x^2 + y^2 - 2*y = 0) := by
  sorry

end line_intersects_circle_l1320_132096


namespace geometric_sequence_common_ratio_l1320_132093

/-- Given a geometric sequence {a_n} with sum S_n, prove that if S_3 = 39 and a_2 = 9,
    then the common ratio q satisfies q^2 - (10/3)q + 1 = 0 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : S 3 = 39) 
  (h2 : a 2 = 9) 
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * q) 
  (h4 : ∀ n : ℕ, n ≥ 1 → S n = a 1 * (1 - q^n) / (1 - q)) 
  : q^2 - (10/3) * q + 1 = 0 :=
sorry

end geometric_sequence_common_ratio_l1320_132093


namespace regular_soda_count_l1320_132020

/-- The number of bottles of regular soda in a grocery store -/
def regular_soda_bottles : ℕ := 83

/-- The number of bottles of diet soda in the grocery store -/
def diet_soda_bottles : ℕ := 4

/-- The difference between the number of regular soda bottles and diet soda bottles -/
def soda_difference : ℕ := 79

theorem regular_soda_count :
  regular_soda_bottles = diet_soda_bottles + soda_difference := by
  sorry

end regular_soda_count_l1320_132020


namespace inequality_system_solution_l1320_132024

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 2 > 2*x - 6 ∧ x < m) ↔ x < 8) → m ≥ 8 := by
  sorry

end inequality_system_solution_l1320_132024


namespace original_denominator_problem_l1320_132041

theorem original_denominator_problem (d : ℚ) : 
  (4 : ℚ) / d ≠ 0 →
  (4 + 3 : ℚ) / (d + 3) = 2 / 3 →
  d = 7.5 := by
sorry

end original_denominator_problem_l1320_132041


namespace star_calculation_l1320_132046

-- Define the ☆ operation
def star (a b : ℚ) : ℚ := a - b + 1

-- Theorem to prove
theorem star_calculation : (star (star 2 3) 2) = -1 := by
  sorry

end star_calculation_l1320_132046


namespace boys_count_l1320_132064

/-- Represents the number of boys on the chess team -/
def boys : ℕ := sorry

/-- Represents the number of girls on the chess team -/
def girls : ℕ := sorry

/-- The total number of team members is 30 -/
axiom total_members : boys + girls = 30

/-- 18 members attended the last meeting -/
axiom attendees : (2 * girls / 3 : ℚ) + boys = 18

/-- Proves that the number of boys on the chess team is 6 -/
theorem boys_count : boys = 6 := by sorry

end boys_count_l1320_132064


namespace coefficient_of_x_squared_l1320_132087

def p (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 3 * x - 1
def q (x : ℝ) : ℝ := 2 * x^3 - x^2 + 5 * x - 4

theorem coefficient_of_x_squared :
  (∃ a b c d e : ℝ, ∀ x, p x * q x = a * x^5 + b * x^4 + c * x^3 - 31 * x^2 + d * x + e) :=
sorry

end coefficient_of_x_squared_l1320_132087


namespace bennetts_brothers_l1320_132083

/-- Given that Aaron has four brothers and Bennett's number of brothers is two less than twice
    the number of Aaron's brothers, prove that Bennett has 6 brothers. -/
theorem bennetts_brothers (aaron_brothers : ℕ) (bennett_brothers : ℕ) 
    (h1 : aaron_brothers = 4)
    (h2 : bennett_brothers = 2 * aaron_brothers - 2) : 
  bennett_brothers = 6 := by
  sorry

end bennetts_brothers_l1320_132083


namespace unique_integral_root_l1320_132084

theorem unique_integral_root :
  ∃! (x : ℤ), x - 8 / (x - 4 : ℚ) = 2 - 8 / (x - 4 : ℚ) :=
by
  -- The proof would go here
  sorry

end unique_integral_root_l1320_132084


namespace xy_value_l1320_132077

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 35/12 := by
  sorry

end xy_value_l1320_132077


namespace root_equation_r_values_l1320_132025

theorem root_equation_r_values (r : ℤ) : 
  (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ 
    r * x^2 - (2*r + 7) * x + r + 7 = 0 ∧
    r * y^2 - (2*r + 7) * y + r + 7 = 0) →
  r = 7 ∨ r = 0 ∨ r = 1 :=
by sorry

end root_equation_r_values_l1320_132025


namespace worker_distance_at_explosion_l1320_132098

/-- The time in seconds when the bomb explodes -/
def bomb_time : ℝ := 45

/-- The speed of the worker in yards per second -/
def worker_speed : ℝ := 6

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1100

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- The distance run by the worker after t seconds, in feet -/
def worker_distance (t : ℝ) : ℝ := worker_speed * yards_to_feet * t

/-- The distance traveled by sound after the bomb explodes, in feet -/
def sound_distance (t : ℝ) : ℝ := sound_speed * (t - bomb_time)

/-- The time when the worker hears the explosion -/
noncomputable def explosion_time : ℝ := 
  (sound_speed * bomb_time) / (sound_speed - worker_speed * yards_to_feet)

/-- The theorem stating that the worker runs approximately 275 yards when he hears the explosion -/
theorem worker_distance_at_explosion : 
  ∃ ε > 0, abs (worker_distance explosion_time / yards_to_feet - 275) < ε :=
sorry

end worker_distance_at_explosion_l1320_132098


namespace paper_area_problem_l1320_132089

theorem paper_area_problem (x : ℕ) : 
  (2 * 11 * 11 = 2 * x * 11 + 100) → x = 6 := by
  sorry

end paper_area_problem_l1320_132089


namespace dog_age_64_human_years_l1320_132080

/-- Calculates the age of a dog in dog years given its age in human years -/
def dogAge (humanYears : ℕ) : ℕ :=
  if humanYears ≤ 15 then 1
  else if humanYears ≤ 24 then 2
  else 2 + (humanYears - 24) / 5

/-- Theorem stating that a dog that has lived 64 human years is 10 years old in dog years -/
theorem dog_age_64_human_years : dogAge 64 = 10 := by
  sorry

end dog_age_64_human_years_l1320_132080


namespace smallest_square_count_is_minimal_l1320_132001

/-- The smallest positive integer n such that n * (1² + 2² + 3²) is a perfect square,
    where n represents the number of squares of each size (1x1, 2x2, 3x3) needed to form a larger square. -/
def smallest_square_count : ℕ := 14

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- Theorem stating that smallest_square_count is the smallest positive integer satisfying the conditions -/
theorem smallest_square_count_is_minimal :
  (is_perfect_square (smallest_square_count * (1 * 1 + 2 * 2 + 3 * 3))) ∧
  (∀ m : ℕ, m > 0 ∧ m < smallest_square_count →
    ¬(is_perfect_square (m * (1 * 1 + 2 * 2 + 3 * 3)))) :=
sorry

end smallest_square_count_is_minimal_l1320_132001


namespace geometric_sum_first_8_terms_l1320_132085

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_8_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by
  sorry

end geometric_sum_first_8_terms_l1320_132085


namespace binomial_square_condition_l1320_132074

/-- If 9x^2 - 18x + a is the square of a binomial, then a = 9 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end binomial_square_condition_l1320_132074


namespace billy_younger_than_gladys_l1320_132004

def billy_age : ℕ := sorry
def lucas_age : ℕ := sorry
def gladys_age : ℕ := 30

axiom lucas_future_age : lucas_age + 3 = 8
axiom gladys_age_relation : gladys_age = 2 * (billy_age + lucas_age)

theorem billy_younger_than_gladys : gladys_age / billy_age = 3 := by sorry

end billy_younger_than_gladys_l1320_132004


namespace range_of_f_l1320_132066

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 1 ∨ y > 1} :=
sorry

end range_of_f_l1320_132066


namespace total_marbles_in_jar_l1320_132059

def ben_marbles : ℕ := 56
def leo_marbles_difference : ℕ := 20

theorem total_marbles_in_jar : 
  ben_marbles + (ben_marbles + leo_marbles_difference) = 132 := by
sorry

end total_marbles_in_jar_l1320_132059


namespace cube_cut_volume_ratio_l1320_132040

theorem cube_cut_volume_ratio (x y : ℝ) (h_positive : x > 0 ∧ y > 0) 
  (h_cut : y < x) (h_surface_ratio : 2 * (x^2 + 2*x*y) = x^2 + 2*x*(x-y)) : 
  (x^2 * y) / (x^2 * (x - y)) = 1 / 5 := by
sorry

end cube_cut_volume_ratio_l1320_132040


namespace factor_implies_d_value_l1320_132051

-- Define the polynomial Q(x)
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

-- Theorem statement
theorem factor_implies_d_value :
  ∀ d : ℝ, (∀ x : ℝ, (x - 3) ∣ Q d x) → d = 8/3 := by
  sorry

end factor_implies_d_value_l1320_132051


namespace quadruplet_babies_l1320_132044

theorem quadruplet_babies (total : ℕ) (twins triplets quadruplets : ℕ) : 
  total = 1500 →
  2 * twins + 3 * triplets + 4 * quadruplets = total →
  triplets = 3 * quadruplets →
  twins = 2 * triplets →
  4 * quadruplets = 240 := by
sorry

end quadruplet_babies_l1320_132044


namespace sum_of_features_l1320_132060

/-- A rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- The number of edges in a rectangular prism -/
def num_edges (prism : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def num_corners (prism : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (prism : RectangularPrism) : ℕ := 6

/-- The theorem stating that the sum of edges, corners, and faces is 26 -/
theorem sum_of_features (prism : RectangularPrism) :
  num_edges prism + num_corners prism + num_faces prism = 26 := by
  sorry

end sum_of_features_l1320_132060


namespace eldora_paper_clips_count_l1320_132095

/-- The cost of one box of paper clips in dollars -/
def paper_clip_cost : ℚ := 185 / 100

/-- The total cost of Eldora's purchase in dollars -/
def eldora_total : ℚ := 5540 / 100

/-- The number of packages of index cards Eldora bought -/
def eldora_index_cards : ℕ := 7

/-- The total cost of Finn's purchase in dollars -/
def finn_total : ℚ := 6170 / 100

/-- The number of boxes of paper clips Finn bought -/
def finn_paper_clips : ℕ := 12

/-- The number of packages of index cards Finn bought -/
def finn_index_cards : ℕ := 10

/-- The number of boxes of paper clips Eldora bought -/
def eldora_paper_clips : ℕ := 15

theorem eldora_paper_clips_count :
  ∃ (index_card_cost : ℚ),
    index_card_cost * finn_index_cards + paper_clip_cost * finn_paper_clips = finn_total ∧
    index_card_cost * eldora_index_cards + paper_clip_cost * eldora_paper_clips = eldora_total :=
by sorry

end eldora_paper_clips_count_l1320_132095


namespace tea_mixture_ratio_l1320_132008

theorem tea_mixture_ratio (price_tea1 price_tea2 price_mixture : ℚ) 
  (h1 : price_tea1 = 62)
  (h2 : price_tea2 = 72)
  (h3 : price_mixture = 64.5) :
  ∃ (x y : ℚ), x > 0 ∧ y > 0 ∧ x / y = 3 ∧
  (x * price_tea1 + y * price_tea2) / (x + y) = price_mixture :=
by sorry

end tea_mixture_ratio_l1320_132008


namespace isosceles_triangle_base_length_l1320_132069

/-- An isosceles triangle with two sides of length 6 cm and perimeter 20 cm has a base of length 8 cm. -/
theorem isosceles_triangle_base_length 
  (side_length : ℝ) 
  (perimeter : ℝ) 
  (h1 : side_length = 6) 
  (h2 : perimeter = 20) : 
  perimeter - 2 * side_length = 8 := by
  sorry

end isosceles_triangle_base_length_l1320_132069


namespace paths_from_A_to_C_l1320_132035

/-- The number of paths between two points -/
def num_paths (start finish : Point) : ℕ := sorry

/-- A point in the graph -/
inductive Point
| A
| B
| C

/-- The total number of paths from A to C -/
def total_paths : ℕ := sorry

theorem paths_from_A_to_C :
  (num_paths Point.A Point.B = 3) →
  (num_paths Point.B Point.C = 1) →
  (num_paths Point.A Point.C = 1) →
  total_paths = 4 := by sorry

end paths_from_A_to_C_l1320_132035


namespace line_translation_l1320_132033

/-- Given a line y = 2x translated by vector (m, n) to y = 2x + 5, 
    prove the relationship between m and n. -/
theorem line_translation (m n : ℝ) : 
  (∀ x y : ℝ, y = 2*x + 5 ↔ y - n = 2*(x - m)) → n = 2*m + 5 := by
sorry

end line_translation_l1320_132033


namespace consecutive_multiples_of_three_l1320_132061

theorem consecutive_multiples_of_three (n : ℕ) : 
  3 * (n - 1) + 3 * (n + 1) = 150 → 3 * n = 75 := by
  sorry

end consecutive_multiples_of_three_l1320_132061


namespace most_appropriate_survey_method_l1320_132062

/-- Represents different survey methods -/
inductive SurveyMethod
| Census
| Sampling

/-- Represents different survey scenarios -/
inductive SurveyScenario
| CityFloatingPopulation
| AirplaneSecurityCheck
| ShellKillingRadius
| ClassMathScores

/-- Determines if a survey method is appropriate for a given scenario -/
def is_appropriate (method : SurveyMethod) (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.CityFloatingPopulation => method = SurveyMethod.Sampling
  | SurveyScenario.AirplaneSecurityCheck => method = SurveyMethod.Census
  | SurveyScenario.ShellKillingRadius => method = SurveyMethod.Sampling
  | SurveyScenario.ClassMathScores => method = SurveyMethod.Census

/-- Theorem stating that using a census method for class math scores is the most appropriate -/
theorem most_appropriate_survey_method :
  is_appropriate SurveyMethod.Census SurveyScenario.ClassMathScores ∧
  ¬(is_appropriate SurveyMethod.Census SurveyScenario.CityFloatingPopulation) ∧
  ¬(is_appropriate SurveyMethod.Sampling SurveyScenario.AirplaneSecurityCheck) ∧
  ¬(is_appropriate SurveyMethod.Census SurveyScenario.ShellKillingRadius) :=
by sorry

end most_appropriate_survey_method_l1320_132062


namespace intersection_point_l1320_132009

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The y-intercept of the first line -/
def b₁ : ℚ := -4

/-- The x-coordinate of the given point -/
def x₀ : ℚ := 3

/-- The y-coordinate of the given point -/
def y₀ : ℚ := 2

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The y-intercept of the perpendicular line -/
def b₂ : ℚ := y₀ - m₂ * x₀

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := (b₂ - b₁) / (m₁ - m₂)

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := m₁ * x_intersect + b₁

theorem intersection_point :
  (x_intersect = 27 / 10) ∧ (y_intersect = 41 / 10) := by
  sorry

end intersection_point_l1320_132009


namespace allocation_schemes_count_l1320_132055

/-- The number of ways to allocate three distinct individuals to seven laboratories,
    where each laboratory can hold at most two people. -/
def allocationSchemes : ℕ := 336

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of permutations of n distinct objects. -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem allocation_schemes_count :
  allocationSchemes = 
    choose 7 3 * factorial 3 + choose 7 2 * choose 3 2 * 2 :=
by sorry

end allocation_schemes_count_l1320_132055


namespace arithmetic_sequence_problem_l1320_132021

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 4 + a 10 + a 16 = 30) : 
  a 18 - 2 * a 14 = -10 := by
  sorry

end arithmetic_sequence_problem_l1320_132021


namespace fence_price_per_foot_l1320_132005

theorem fence_price_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 289) 
  (h2 : total_cost = 3672) : 
  total_cost / (4 * Real.sqrt area) = 54 := by
  sorry

end fence_price_per_foot_l1320_132005


namespace total_questions_formula_l1320_132076

/-- Represents the number of questions completed by three girls in 2 hours -/
def total_questions (fiona_questions : ℕ) (r : ℚ) : ℚ :=
  let shirley_questions := r * fiona_questions
  let kiana_questions := (fiona_questions + shirley_questions) / 2
  2 * (fiona_questions + shirley_questions + kiana_questions)

/-- Theorem stating the total number of questions completed by three girls in 2 hours -/
theorem total_questions_formula (r : ℚ) : 
  total_questions 36 r = 108 + 108 * r := by
  sorry

end total_questions_formula_l1320_132076


namespace english_only_students_l1320_132032

theorem english_only_students (total : ℕ) (eg ef gf egf g f : ℕ) : 
  total = 50 ∧ 
  eg = 12 ∧ 
  g = 22 ∧ 
  f = 18 ∧ 
  ef = 10 ∧ 
  gf = 8 ∧ 
  egf = 4 ∧ 
  (∃ (e g_only f_only : ℕ), 
    e + g_only + f_only + eg + ef + gf - egf = total) →
  ∃ (e : ℕ), e = 14 ∧ 
    e + (g - (eg + gf - egf)) + (f - (ef + gf - egf)) + eg + ef + gf - egf = total :=
by sorry

end english_only_students_l1320_132032


namespace perpendicular_lines_relationship_l1320_132054

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a vector
def perpendicular (l : Line3D) (v : ℝ × ℝ × ℝ) : Prop :=
  let (dx, dy, dz) := l.direction
  let (vx, vy, vz) := v
  dx * vx + dy * vy + dz * vz = 0

-- Define the relationships between two lines
inductive LineRelationship
  | Parallel
  | Intersecting
  | Skew

-- State the theorem
theorem perpendicular_lines_relationship (l1 l2 l3 : Line3D) 
  (h1 : perpendicular l1 l3.direction) 
  (h2 : perpendicular l2 l3.direction) :
  ∃ r : LineRelationship, true :=
sorry

end perpendicular_lines_relationship_l1320_132054


namespace greatest_power_of_two_factor_of_20_factorial_l1320_132091

-- Define n as 20!
def n : ℕ := (List.range 20).foldl (· * ·) 1

-- Define the property of k being the greatest integer for which 2^k divides n
def is_greatest_power_of_two_factor (k : ℕ) : Prop :=
  2^k ∣ n ∧ ∀ m : ℕ, 2^m ∣ n → m ≤ k

-- Theorem statement
theorem greatest_power_of_two_factor_of_20_factorial :
  is_greatest_power_of_two_factor 18 :=
sorry

end greatest_power_of_two_factor_of_20_factorial_l1320_132091
