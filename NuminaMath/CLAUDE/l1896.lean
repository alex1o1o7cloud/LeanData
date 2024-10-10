import Mathlib

namespace tyler_saltwater_animals_l1896_189643

/-- The number of saltwater aquariums Tyler has -/
def num_saltwater_aquariums : ℕ := 56

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 39

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := num_saltwater_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_saltwater_animals = 2184 := by
  sorry

end tyler_saltwater_animals_l1896_189643


namespace gcd_inequality_l1896_189620

theorem gcd_inequality (n d₁ d₂ : ℕ+) : 
  (Nat.gcd n (d₁ + d₂) : ℚ) / (Nat.gcd n d₁ * Nat.gcd n d₂) ≥ 1 / n.val :=
by sorry

end gcd_inequality_l1896_189620


namespace archibald_win_percentage_l1896_189628

theorem archibald_win_percentage (archibald_wins brother_wins : ℕ) : 
  archibald_wins = 12 → brother_wins = 18 → 
  (archibald_wins : ℚ) / (archibald_wins + brother_wins : ℚ) * 100 = 40 := by
  sorry

end archibald_win_percentage_l1896_189628


namespace five_saturdays_in_august_l1896_189685

/-- Represents days of the week --/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month --/
structure Month where
  days : Nat
  first_day : DayOfWeek

/-- July of the given year --/
def july : Month := sorry

/-- August of the given year --/
def august : Month := sorry

/-- Counts the occurrences of a specific day in a month --/
def count_day_occurrences (m : Month) (d : DayOfWeek) : Nat := sorry

/-- The theorem to prove --/
theorem five_saturdays_in_august (h : count_day_occurrences july DayOfWeek.Wednesday = 5) :
  count_day_occurrences august DayOfWeek.Saturday = 5 := by sorry

end five_saturdays_in_august_l1896_189685


namespace divisor_sum_ratio_l1896_189610

def N : ℕ := 48 * 49 * 75 * 343

def sum_of_divisors (n : ℕ) : ℕ := sorry

def sum_of_divisors_multiple_of_three (n : ℕ) : ℕ := sorry

def sum_of_divisors_not_multiple_of_three (n : ℕ) : ℕ := sorry

theorem divisor_sum_ratio :
  ∃ (a b : ℕ), 
    (sum_of_divisors_multiple_of_three N) * b = (sum_of_divisors_not_multiple_of_three N) * a ∧
    a ≠ 0 ∧ b ≠ 0 ∧ 
    (∀ (c d : ℕ), c ≠ 0 → d ≠ 0 → 
      (sum_of_divisors_multiple_of_three N) * d = (sum_of_divisors_not_multiple_of_three N) * c →
      a ≤ c ∧ b ≤ d) :=
by sorry

end divisor_sum_ratio_l1896_189610


namespace even_function_domain_l1896_189612

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the domain of the function
def Domain (a : ℝ) : Set ℝ := {x : ℝ | |x + 2 - a| < a}

-- Theorem statement
theorem even_function_domain (f : ℝ → ℝ) (a : ℝ) 
  (h_even : EvenFunction f) 
  (h_domain : Set.range f = Domain a) 
  (h_positive : a > 0) : 
  a = 2 := by sorry

end even_function_domain_l1896_189612


namespace locus_equation_l1896_189624

/-- Circle C₁ with equation x² + y² = 4 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Circle C₂ with equation (x - 3)² + y² = 81 -/
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + p.2^2 = 81}

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  center.1^2 + center.2^2 = (radius + 2)^2

/-- A circle is internally tangent to C₂ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₂ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - 3)^2 + center.2^2 = (9 - radius)^2

/-- The locus of centers of circles externally tangent to C₁ and internally tangent to C₂ -/
def locus : Set (ℝ × ℝ) :=
  {p | ∃ r : ℝ, externally_tangent_C₁ p r ∧ internally_tangent_C₂ p r}

theorem locus_equation : locus = {p : ℝ × ℝ | 12 * p.1^2 + 169 * p.2^2 - 36 * p.1 - 1584 = 0} := by
  sorry

end locus_equation_l1896_189624


namespace miriam_flower_care_l1896_189625

/-- Calculates the number of flowers Miriam can take care of in a given number of days -/
def flowers_cared_for (hours_per_day : ℕ) (flowers_per_day : ℕ) (num_days : ℕ) : ℕ :=
  (hours_per_day * num_days) * (flowers_per_day / hours_per_day)

/-- Proves that Miriam can take care of 360 flowers in 6 days -/
theorem miriam_flower_care :
  flowers_cared_for 5 60 6 = 360 := by
  sorry

#eval flowers_cared_for 5 60 6

end miriam_flower_care_l1896_189625


namespace paper_cranes_problem_l1896_189679

/-- The number of paper cranes folded by student A -/
def cranes_A (x : ℤ) : ℤ := 3 * x - 100

/-- The number of paper cranes folded by student C -/
def cranes_C (x : ℤ) : ℤ := cranes_A x - 67

theorem paper_cranes_problem (x : ℤ) 
  (h1 : cranes_A x + x + cranes_C x = 1000) : 
  cranes_A x = 443 := by
  sorry

end paper_cranes_problem_l1896_189679


namespace sequence_non_positive_l1896_189671

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k : ℕ, 1 ≤ k ∧ k < n → a (k-1) - 2 * a k + a (k+1) ≥ 0) :
  ∀ k : ℕ, k ≤ n → a k ≤ 0 := by
  sorry

end sequence_non_positive_l1896_189671


namespace inequality1_solution_inequality2_solution_l1896_189696

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x * (7 - x) ≥ 12
def inequality2 (x : ℝ) : Prop := x^2 > 2 * (x - 1)

-- Define the solution sets
def solution_set1 : Set ℝ := {x | 3 ≤ x ∧ x ≤ 4}
def solution_set2 : Set ℝ := Set.univ

-- Theorem statements
theorem inequality1_solution : {x : ℝ | inequality1 x} = solution_set1 := by sorry

theorem inequality2_solution : {x : ℝ | inequality2 x} = solution_set2 := by sorry

end inequality1_solution_inequality2_solution_l1896_189696


namespace square_root_sum_l1896_189608

theorem square_root_sum (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end square_root_sum_l1896_189608


namespace total_students_in_halls_l1896_189622

theorem total_students_in_halls (general : ℕ) (biology : ℕ) (math : ℕ) : 
  general = 30 →
  biology = 2 * general →
  math = (3 * (general + biology)) / 5 →
  general + biology + math = 144 := by
sorry

end total_students_in_halls_l1896_189622


namespace function_characterization_l1896_189626

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ a b : ℕ, f (a * b) = f a + f b - f (Nat.gcd a b)) ∧
  (∀ p a : ℕ, Nat.Prime p → (f a ≥ f (a * p) → f a + f p ≥ f a * f p + 1))

theorem function_characterization (f : ℕ → ℕ) (h : is_valid_function f) :
  (∀ n : ℕ, f n = n) ∨ (∀ n : ℕ, f n = 1) :=
sorry

end function_characterization_l1896_189626


namespace apple_pie_calculation_l1896_189670

theorem apple_pie_calculation (total_apples : ℕ) (unripe_apples : ℕ) (apples_per_pie : ℕ) 
  (h1 : total_apples = 34) 
  (h2 : unripe_apples = 6) 
  (h3 : apples_per_pie = 4) :
  (total_apples - unripe_apples) / apples_per_pie = 7 :=
by sorry

end apple_pie_calculation_l1896_189670


namespace line_graph_most_suitable_for_aqi_l1896_189684

/-- Represents types of statistical graphs -/
inductive StatGraph
  | LineGraph
  | Histogram
  | BarGraph
  | PieChart

/-- Represents a series of daily AQI values -/
def AQISeries := List Nat

/-- Determines if a graph type is suitable for showing time-based trends -/
def shows_time_trends (graph : StatGraph) : Prop :=
  match graph with
  | StatGraph.LineGraph => true
  | _ => false

/-- Determines if a graph type is suitable for showing continuous data -/
def shows_continuous_data (graph : StatGraph) : Prop :=
  match graph with
  | StatGraph.LineGraph => true
  | _ => false

/-- Theorem: A line graph is the most suitable for describing AQI changes over time -/
theorem line_graph_most_suitable_for_aqi (aqi_data : AQISeries) :
  aqi_data.length = 10 →
  ∃ (g : StatGraph), shows_time_trends g ∧ shows_continuous_data g ∧
  ∀ (g' : StatGraph), (shows_time_trends g' ∧ shows_continuous_data g') → g = g' :=
by sorry

end line_graph_most_suitable_for_aqi_l1896_189684


namespace width_length_ratio_l1896_189651

/-- A rectangle with given length and perimeter -/
structure Rectangle where
  length : ℝ
  perimeter : ℝ
  width : ℝ
  length_pos : length > 0
  perimeter_pos : perimeter > 0
  width_pos : width > 0
  perimeter_eq : perimeter = 2 * (length + width)

/-- The ratio of width to length for a rectangle with length 10 and perimeter 30 is 1:2 -/
theorem width_length_ratio (rect : Rectangle) 
    (h1 : rect.length = 10) 
    (h2 : rect.perimeter = 30) : 
    rect.width / rect.length = 1 / 2 := by
  sorry


end width_length_ratio_l1896_189651


namespace square_sum_inequality_l1896_189687

theorem square_sum_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end square_sum_inequality_l1896_189687


namespace parallel_lines_a_value_l1896_189639

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal and not undefined -/
def parallel (l1 l2 : Line) : Prop :=
  l1.b ≠ 0 ∧ l2.b ≠ 0 ∧ l1.a / l1.b = l2.a / l2.b

theorem parallel_lines_a_value :
  ∃ (a : ℝ), parallel (Line.mk 2 a (-2)) (Line.mk a (a + 4) (-4)) ↔ a = -2 :=
sorry

end parallel_lines_a_value_l1896_189639


namespace salary_comparison_l1896_189633

theorem salary_comparison (raja_salary : ℝ) (ram_salary : ℝ) 
  (h : ram_salary = raja_salary * 1.25) : 
  (raja_salary / ram_salary) = 0.8 := by
sorry

end salary_comparison_l1896_189633


namespace integer_solutions_of_equation_l1896_189693

theorem integer_solutions_of_equation :
  ∀ x m : ℤ, (|x^2 - 1| + |x^2 - 4| = m * x) ↔ ((x = -1 ∧ m = 3) ∨ (x = 1 ∧ m = 3)) :=
by sorry

end integer_solutions_of_equation_l1896_189693


namespace julia_bill_ratio_l1896_189602

/-- Proves the ratio of Julia's Sunday miles to Bill's Sunday miles -/
theorem julia_bill_ratio (bill_sunday : ℕ) (bill_saturday : ℕ) (julia_sunday : ℕ) :
  bill_sunday = 10 →
  bill_sunday = bill_saturday + 4 →
  bill_sunday + bill_saturday + julia_sunday = 36 →
  julia_sunday = 2 * bill_sunday :=
by sorry

end julia_bill_ratio_l1896_189602


namespace common_difference_is_two_l1896_189629

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem common_difference_is_two (seq : ArithmeticSequence) 
    (h : seq.S 3 / 3 - seq.S 2 / 2 = 1) : seq.d = 2 := by
  sorry

end common_difference_is_two_l1896_189629


namespace unique_prime_generating_number_l1896_189623

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem unique_prime_generating_number :
  ∃! n : ℕ, n > 0 ∧ is_prime (n^n + 1) ∧ is_prime ((2*n)^(2*n) + 1) ∧ n = 2 :=
sorry

end unique_prime_generating_number_l1896_189623


namespace dandelion_survival_l1896_189660

/-- The number of seeds produced by each dandelion -/
def seeds_per_dandelion : ℕ := 300

/-- The fraction of seeds that land in water and die -/
def water_death_fraction : ℚ := 1/3

/-- The fraction of starting seeds eaten by insects -/
def insect_eaten_fraction : ℚ := 1/6

/-- The fraction of remaining seeds that sprout and are immediately eaten -/
def sprout_eaten_fraction : ℚ := 1/2

/-- The number of dandelions that survive long enough to flower -/
def surviving_dandelions : ℕ := 75

theorem dandelion_survival :
  (seeds_per_dandelion : ℚ) * (1 - water_death_fraction) * (1 - insect_eaten_fraction) * (1 - sprout_eaten_fraction) = surviving_dandelions := by
  sorry

end dandelion_survival_l1896_189660


namespace age_difference_is_zero_l1896_189638

/-- Given that Carlos and David were born on the same day in different years,
    prove that the age difference between them is 0 years. -/
theorem age_difference_is_zero (C D m : ℕ) : 
  C = D + m →
  C - 1 = 6 * (D - 1) →
  C = D^3 →
  m = 0 := by
  sorry

end age_difference_is_zero_l1896_189638


namespace no_infinite_sequence_with_sqrt_property_l1896_189673

theorem no_infinite_sequence_with_sqrt_property :
  ¬ (∃ (a : ℕ → ℕ), ∀ (n : ℕ), a (n + 2) = a (n + 1) + Real.sqrt (a (n + 1) + a n)) :=
by sorry

end no_infinite_sequence_with_sqrt_property_l1896_189673


namespace linear_decreasing_implies_second_or_third_quadrant_l1896_189698

/-- A linear function f(x) = kx + b is monotonically decreasing on ℝ -/
def MonotonicallyDecreasing (k b : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → k * x + b > k * y + b

/-- The point (x, y) is in the second or third quadrant -/
def InSecondOrThirdQuadrant (x y : ℝ) : Prop :=
  x < 0

theorem linear_decreasing_implies_second_or_third_quadrant
  (k b : ℝ) (h : MonotonicallyDecreasing k b) :
  InSecondOrThirdQuadrant k b :=
sorry

end linear_decreasing_implies_second_or_third_quadrant_l1896_189698


namespace candy_distribution_l1896_189614

theorem candy_distribution (x y n : ℕ) : 
  y + n = 4 * (x - n) →
  x + 90 = 5 * (y - 90) →
  y ≥ 115 →
  (∀ y' : ℕ, y' ≥ 115 → y' + n = 4 * (x - n) → x + 90 = 5 * (y' - 90) → y ≤ y') →
  y = 115 ∧ x = 35 ∧ n = 5 := by
  sorry

end candy_distribution_l1896_189614


namespace age_problem_l1896_189662

theorem age_problem (a b : ℕ) : 
  (a : ℚ) / b = 5 / 3 →
  ((a + 2) : ℚ) / (b + 2) = 3 / 2 →
  b = 6 :=
by
  sorry

end age_problem_l1896_189662


namespace star_operation_result_l1896_189637

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.three
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.one
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem star_operation_result :
  star (star Element.three Element.one) (star Element.four Element.two) = Element.one := by
  sorry

end star_operation_result_l1896_189637


namespace dianes_gambling_problem_l1896_189653

/-- Diane's gambling problem -/
theorem dianes_gambling_problem 
  (x y a b : ℝ) 
  (h1 : x * a = 65)
  (h2 : y * b = 150)
  (h3 : x * a - y * b = -50) :
  y * b - x * a = 50 := by
  sorry

end dianes_gambling_problem_l1896_189653


namespace k_range_l1896_189688

/-- The ellipse equation -/
def ellipse_eq (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

/-- The origin (0,0) is inside the ellipse -/
def origin_inside (k : ℝ) : Prop :=
  ∃ ε > 0, ∀ x y : ℝ, x^2 + y^2 < ε^2 → ellipse_eq k x y

/-- The theorem stating the range of k -/
theorem k_range (k : ℝ) (h : origin_inside k) : 0 < |k| ∧ |k| < 1 :=
sorry

end k_range_l1896_189688


namespace max_candies_eaten_l1896_189654

theorem max_candies_eaten (n : ℕ) (h : n = 28) : 
  (n * (n - 1)) / 2 = 378 := by
  sorry

end max_candies_eaten_l1896_189654


namespace decimal_arithmetic_l1896_189606

theorem decimal_arithmetic : 25.3 - 0.432 + 1.25 = 26.118 := by
  sorry

end decimal_arithmetic_l1896_189606


namespace smallest_number_of_points_l1896_189640

/-- The length of the circle -/
def circleLength : ℕ := 1956

/-- The distance between adjacent points in the sequence -/
def distanceStep : ℕ := 3

/-- The number of points required -/
def numPoints : ℕ := 2 * (circleLength / distanceStep)

/-- Theorem stating the smallest number of points satisfying the conditions -/
theorem smallest_number_of_points :
  numPoints = 1304 ∧
  ∀ n : ℕ, n < numPoints →
    ¬(∀ i : Fin n,
      ∃! j : Fin n, i ≠ j ∧ (circleLength * (i.val - j.val : ℤ) / n).natAbs % circleLength = 1 ∧
      ∃! k : Fin n, i ≠ k ∧ (circleLength * (i.val - k.val : ℤ) / n).natAbs % circleLength = 2) :=
by sorry

end smallest_number_of_points_l1896_189640


namespace perpendicular_slope_l1896_189663

theorem perpendicular_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let m₁ := a / b
  let m₂ := -1 / m₁
  (a * x - b * y = c) → (m₂ = -b / a) :=
sorry

end perpendicular_slope_l1896_189663


namespace triangle_area_l1896_189664

/-- Given a triangle ABC where BC = 12 cm, AC = 5 cm, and the angle between BC and AC is 30°,
    prove that the area of the triangle is 15 square centimeters. -/
theorem triangle_area (BC AC : ℝ) (angle : Real) (h : BC = 12 ∧ AC = 5 ∧ angle = 30 * Real.pi / 180) :
  (1 / 2 : ℝ) * BC * (AC * Real.sin angle) = 15 := by
  sorry

end triangle_area_l1896_189664


namespace fruit_cost_theorem_l1896_189619

/-- Given the prices of fruits satisfying certain conditions, prove the cost of a specific combination. -/
theorem fruit_cost_theorem (x y z : ℝ) 
  (h1 : 2 * x + y + 4 * z = 6) 
  (h2 : 4 * x + 2 * y + 2 * z = 4) : 
  4 * x + 2 * y + 5 * z = 8 := by
  sorry

end fruit_cost_theorem_l1896_189619


namespace lab_budget_theorem_l1896_189601

def lab_budget_problem (total_budget flask_cost test_tube_cost safety_gear_cost chemical_cost min_instrument_cost : ℚ) 
  (min_instruments : ℕ) : Prop :=
  let total_spent := flask_cost + test_tube_cost + safety_gear_cost + chemical_cost + min_instrument_cost
  total_budget = 750 ∧
  flask_cost = 200 ∧
  test_tube_cost = 2/3 * flask_cost ∧
  safety_gear_cost = 1/2 * test_tube_cost ∧
  chemical_cost = 3/4 * flask_cost ∧
  min_instrument_cost ≥ 50 ∧
  min_instruments ≥ 10 ∧
  total_budget - total_spent = 150

theorem lab_budget_theorem :
  ∃ (total_budget flask_cost test_tube_cost safety_gear_cost chemical_cost min_instrument_cost : ℚ) 
    (min_instruments : ℕ),
  lab_budget_problem total_budget flask_cost test_tube_cost safety_gear_cost chemical_cost min_instrument_cost min_instruments :=
by
  sorry

end lab_budget_theorem_l1896_189601


namespace problem_1_problem_2_l1896_189695

-- Problem 1
theorem problem_1 : |(-2023 : ℤ)| + π^(0 : ℝ) - (1/6)⁻¹ + Real.sqrt 16 = 2022 := by sorry

-- Problem 2
theorem problem_2 (m : ℝ) (hm : m ≠ 1) :
  (1 + 1/m) / ((m^2 - 1) / m) = 1 / (m - 1) := by sorry

end problem_1_problem_2_l1896_189695


namespace unit_circle_point_x_coordinate_l1896_189613

theorem unit_circle_point_x_coordinate 
  (P : ℝ × ℝ) (α : ℝ) 
  (h1 : P.1^2 + P.2^2 = 1) 
  (h2 : P.1 = Real.cos α) 
  (h3 : P.2 = Real.sin α) 
  (h4 : π/3 < α ∧ α < 5*π/6) 
  (h5 : Real.sin (α + π/6) = 3/5) : 
  P.1 = (3 - 4*Real.sqrt 3) / 10 := by
sorry

end unit_circle_point_x_coordinate_l1896_189613


namespace sqrt_difference_equals_4sqrt2_l1896_189672

theorem sqrt_difference_equals_4sqrt2 :
  Real.sqrt (5 + 6 * Real.sqrt 2) - Real.sqrt (5 - 6 * Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end sqrt_difference_equals_4sqrt2_l1896_189672


namespace sports_club_membership_l1896_189650

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 30 →
  badminton = 17 →
  tennis = 21 →
  both = 10 →
  total - (badminton + tennis - both) = 2 :=
by sorry

end sports_club_membership_l1896_189650


namespace different_color_picks_count_l1896_189661

/-- Represents a card color -/
inductive CardColor
| Red
| Black
| Colorless

/-- Represents the deck composition -/
structure Deck :=
  (red_cards : Nat)
  (black_cards : Nat)
  (jokers : Nat)

/-- The number of ways to pick two different cards of different colors -/
def different_color_picks (d : Deck) : Nat :=
  -- Red-Black or Black-Red
  2 * d.red_cards * d.black_cards +
  -- Colorless-Red or Colorless-Black
  2 * d.jokers * (d.red_cards + d.black_cards) +
  -- Red-Colorless or Black-Colorless
  2 * (d.red_cards + d.black_cards) * d.jokers

/-- The theorem to be proved -/
theorem different_color_picks_count :
  let d : Deck := { red_cards := 26, black_cards := 26, jokers := 2 }
  different_color_picks d = 1508 := by
  sorry

end different_color_picks_count_l1896_189661


namespace symmetric_point_about_line_l1896_189646

/-- The symmetric point of (x₁, y₁) about the line ax + by + c = 0 is (x₂, y₂) -/
def is_symmetric_point (x₁ y₁ x₂ y₂ a b c : ℝ) : Prop :=
  -- The line connecting the points is perpendicular to the line of symmetry
  (y₂ - y₁) * a = -(x₂ - x₁) * b ∧
  -- The midpoint of the two points lies on the line of symmetry
  (a * ((x₁ + x₂) / 2) + b * ((y₁ + y₂) / 2) + c = 0)

theorem symmetric_point_about_line :
  is_symmetric_point (-1) 2 (-6) (-3) 1 1 4 := by
  sorry

end symmetric_point_about_line_l1896_189646


namespace surrounding_circles_radius_l1896_189649

theorem surrounding_circles_radius (r : ℝ) : r = 4 := by
  -- Given a central circle of radius 2
  -- Surrounded by 4 circles of radius r
  -- The surrounding circles touch the central circle and each other
  -- We need to prove that r = 4
  sorry

end surrounding_circles_radius_l1896_189649


namespace inverse_proportion_y_relationship_l1896_189631

/-- Given three points on the graph of y = -4/x, prove their y-coordinates' relationship -/
theorem inverse_proportion_y_relationship 
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -4 / x₁)
  (h2 : y₂ = -4 / x₂)
  (h3 : y₃ = -4 / x₃)
  (hx : x₁ < 0 ∧ 0 < x₂ ∧ x₂ < x₃) :
  y₁ > y₃ ∧ y₃ > y₂ :=
by sorry

end inverse_proportion_y_relationship_l1896_189631


namespace sqrt_sum_simplification_l1896_189694

theorem sqrt_sum_simplification :
  ∃ (a b c : ℕ+),
    (Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) = 
     (a.val * Real.sqrt 6 + b.val * Real.sqrt 8) / c.val) ∧
    (∀ (a' b' c' : ℕ+),
      (Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) = 
       (a'.val * Real.sqrt 6 + b'.val * Real.sqrt 8) / c'.val) →
      c'.val ≥ c.val) ∧
    (a.val + b.val + c.val = 19) := by
  sorry

end sqrt_sum_simplification_l1896_189694


namespace greatest_n_value_l1896_189682

theorem greatest_n_value (n : ℤ) (h : 102 * n^2 ≤ 8100) : n ≤ 8 ∧ ∃ (m : ℤ), m = 8 ∧ 102 * m^2 ≤ 8100 := by
  sorry

end greatest_n_value_l1896_189682


namespace max_discount_theorem_l1896_189681

theorem max_discount_theorem (C : ℝ) : 
  C > 0 →                           -- Cost price is positive
  1.8 * C = 360 →                   -- Selling price is 80% above cost price and equals 360
  ∀ x : ℝ, 
    360 - x ≥ 1.3 * C →             -- Price after discount is at least 130% of cost price
    x ≤ 100 :=                      -- Maximum discount is 100
by
  sorry

end max_discount_theorem_l1896_189681


namespace second_player_winning_strategy_l1896_189665

/-- Represents a domino tile with two numbers -/
structure Domino :=
  (upper : Nat)
  (lower : Nat)
  (upper_bound : upper ≤ 6)
  (lower_bound : lower ≤ 6)

/-- The set of all possible domino tiles -/
def dominoSet : Finset Domino := sorry

/-- The game state, including the numbers written on the blackboard and remaining tiles -/
structure GameState :=
  (written : Finset Nat)
  (remaining : Finset Domino)

/-- A player's strategy for selecting a domino -/
def Strategy := GameState → Option Domino

/-- Determines if a strategy is winning for the second player -/
def isWinningStrategy (s : Strategy) : Prop := sorry

/-- The main theorem stating that there exists a winning strategy for the second player -/
theorem second_player_winning_strategy :
  ∃ (s : Strategy), isWinningStrategy s := by sorry

end second_player_winning_strategy_l1896_189665


namespace function_range_l1896_189615

theorem function_range (x : ℝ) (h : x > 1) : 
  let y := x + 1 / (x - 1)
  (∀ x > 1, y ≥ 3) ∧ (∃ x > 1, y = 3) :=
by sorry

end function_range_l1896_189615


namespace comparison_of_special_angles_l1896_189690

open Real

theorem comparison_of_special_angles (a b c : ℝ) 
  (ha : 0 < a ∧ a < π/2)
  (hb : 0 < b ∧ b < π/2)
  (hc : 0 < c ∧ c < π/2)
  (eq_a : cos a = a)
  (eq_b : sin (cos b) = b)
  (eq_c : cos (sin c) = c) :
  c > a ∧ a > b :=
sorry

end comparison_of_special_angles_l1896_189690


namespace james_oranges_l1896_189680

theorem james_oranges :
  ∀ (o : ℕ),
    o ≤ 7 →
    (∃ (a : ℕ), a + o = 7 ∧ (65 * o + 40 * a) % 100 = 0) →
    o = 4 :=
by sorry

end james_oranges_l1896_189680


namespace average_speed_calculation_l1896_189630

/-- Given a distance of 10000 meters and a time of 28 minutes, 
    prove that the average speed is approximately 595.24 cm/s. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) : 
  distance = 10000 ∧ time = 28 → 
  ∃ (speed : ℝ), abs (speed - 595.24) < 0.01 ∧ 
  speed = (distance * 100) / (time * 60) := by
  sorry

#check average_speed_calculation

end average_speed_calculation_l1896_189630


namespace jenny_money_problem_l1896_189627

theorem jenny_money_problem (original : ℚ) : 
  (4/7 : ℚ) * original = 24 → (1/2 : ℚ) * original = 21 := by
  sorry

end jenny_money_problem_l1896_189627


namespace average_youtube_viewer_videos_l1896_189645

theorem average_youtube_viewer_videos (video_length : ℕ) (ad_time : ℕ) (total_time : ℕ) :
  video_length = 7 →
  ad_time = 3 →
  total_time = 17 →
  ∃ (num_videos : ℕ), num_videos * video_length + ad_time = total_time ∧ num_videos = 2 :=
by sorry

end average_youtube_viewer_videos_l1896_189645


namespace find_m_value_l1896_189655

theorem find_m_value (x y m : ℝ) 
  (eq1 : 3 * x + 7 * y = 5 * m - 3)
  (eq2 : 2 * x + 3 * y = 8)
  (eq3 : x + 2 * y = 5) : 
  m = 4 := by
sorry

end find_m_value_l1896_189655


namespace sum_of_coordinates_zero_l1896_189692

/-- For all points (x, y) in the real plane where x + y = 0, prove that y = -x -/
theorem sum_of_coordinates_zero (x y : ℝ) (h : x + y = 0) : y = -x := by
  sorry

end sum_of_coordinates_zero_l1896_189692


namespace chord_line_equation_l1896_189699

/-- Given an ellipse and a chord midpoint, prove the equation of the line containing the chord -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 / 16 + y^2 / 9 = 1) →  -- Ellipse equation
  (∃ (x1 y1 x2 y2 : ℝ),       -- Existence of chord endpoints
    x1^2 / 16 + y1^2 / 9 = 1 ∧
    x2^2 / 16 + y2^2 / 9 = 1 ∧
    (x1 + x2) / 2 = 2 ∧       -- Midpoint x-coordinate
    (y1 + y2) / 2 = 3/2) →    -- Midpoint y-coordinate
  (∃ (a b c : ℝ),             -- Existence of line equation
    a*x + b*y + c = 0 ∧       -- General form of line equation
    a = 3 ∧ b = 4 ∧ c = -12)  -- Specific coefficients
  := by sorry

end chord_line_equation_l1896_189699


namespace min_tablets_extraction_l1896_189652

/-- Represents the number of tablets for each medicine type -/
structure MedicineCount where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Represents the minimum number of tablets required for each medicine type -/
structure RequiredCount where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Calculates the minimum number of tablets to be extracted -/
def minTablets (total : MedicineCount) (required : RequiredCount) : Nat :=
  sorry

theorem min_tablets_extraction (total : MedicineCount) (required : RequiredCount) :
  total.A = 10 →
  total.B = 14 →
  total.C = 18 →
  total.D = 20 →
  required.A = 3 →
  required.B = 4 →
  required.C = 3 →
  required.D = 2 →
  minTablets total required = 55 := by
  sorry

end min_tablets_extraction_l1896_189652


namespace solve_equation_l1896_189678

theorem solve_equation (y : ℝ) : (7 - y = 4) → y = 3 := by
  sorry

end solve_equation_l1896_189678


namespace expected_socks_theorem_l1896_189642

/-- The expected number of socks taken until a pair is found -/
def expected_socks (n : ℕ) : ℝ := 2 * n

/-- Theorem: For n pairs of distinct socks arranged randomly, 
    the expected number of socks taken until a pair is found is 2n -/
theorem expected_socks_theorem (n : ℕ) : 
  expected_socks n = 2 * n := by sorry

end expected_socks_theorem_l1896_189642


namespace smallest_block_volume_l1896_189636

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Checks if the given dimensions satisfy the problem conditions. -/
def satisfiesConditions (d : BlockDimensions) : Prop :=
  (d.length - 1) * (d.width - 1) * (d.height - 1) = 288 ∧
  (d.length + d.width + d.height) % 10 = 0

/-- The volume of the block given its dimensions. -/
def blockVolume (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The theorem stating the smallest possible value of N. -/
theorem smallest_block_volume :
  ∃ (d : BlockDimensions), satisfiesConditions d ∧
    blockVolume d = 455 ∧
    ∀ (d' : BlockDimensions), satisfiesConditions d' → blockVolume d' ≥ 455 := by
  sorry

end smallest_block_volume_l1896_189636


namespace divisor_property_l1896_189634

theorem divisor_property (x y : ℕ) (h1 : x % 63 = 11) (h2 : x % y = 2) :
  ∃ (k : ℕ), y ∣ (63 * k + 9) := by
  sorry

end divisor_property_l1896_189634


namespace silva_family_zoo_cost_l1896_189667

/-- Calculates the total cost of zoo tickets for a family group -/
def total_zoo_cost (senior_ticket_cost : ℚ) (child_discount : ℚ) (senior_discount : ℚ) : ℚ :=
  let full_price := senior_ticket_cost / (1 - senior_discount)
  let child_price := full_price * (1 - child_discount)
  3 * senior_ticket_cost + 3 * full_price + 3 * child_price

/-- Theorem stating the total cost for the Silva family zoo trip -/
theorem silva_family_zoo_cost :
  total_zoo_cost 7 (4/10) (3/10) = 69 := by
  sorry

#eval total_zoo_cost 7 (4/10) (3/10)

end silva_family_zoo_cost_l1896_189667


namespace gcd_143_98_l1896_189674

theorem gcd_143_98 : Nat.gcd 143 98 = 1 := by
  sorry

end gcd_143_98_l1896_189674


namespace reimbursement_calculation_l1896_189605

/-- Calculates the total reimbursement for a sales rep based on daily mileage -/
def total_reimbursement (rate : ℚ) (miles : List ℚ) : ℚ :=
  (miles.map (· * rate)).sum

/-- Proves that the total reimbursement for the given mileage and rate is $36.00 -/
theorem reimbursement_calculation : 
  let rate : ℚ := 36 / 100
  let daily_miles : List ℚ := [18, 26, 20, 20, 16]
  total_reimbursement rate daily_miles = 36 := by
  sorry

#eval total_reimbursement (36 / 100) [18, 26, 20, 20, 16]

end reimbursement_calculation_l1896_189605


namespace quadratic_root_sum_l1896_189641

theorem quadratic_root_sum (p q : ℝ) : 
  (∃ (x : ℂ), x^2 + p*x + q = 0 ∧ x = 1 + I) → p + q = 0 := by
  sorry

end quadratic_root_sum_l1896_189641


namespace sum_edge_lengths_truncated_octahedron_l1896_189644

/-- A polyhedron with 24 vertices and all edges of length 5 cm -/
structure Polyhedron where
  vertices : ℕ
  edge_length : ℝ
  h_vertices : vertices = 24
  h_edge_length : edge_length = 5

/-- A truncated octahedron is a polyhedron with 36 edges -/
def is_truncated_octahedron (p : Polyhedron) : Prop :=
  ∃ (edges : ℕ), edges = 36

/-- The sum of edge lengths for a polyhedron -/
def sum_edge_lengths (p : Polyhedron) (edges : ℕ) : ℝ :=
  p.edge_length * edges

/-- Theorem: If the polyhedron is a truncated octahedron, 
    then the sum of edge lengths is 180 cm -/
theorem sum_edge_lengths_truncated_octahedron (p : Polyhedron) 
  (h : is_truncated_octahedron p) : 
  ∃ (edges : ℕ), sum_edge_lengths p edges = 180 := by
  sorry


end sum_edge_lengths_truncated_octahedron_l1896_189644


namespace bounded_expression_l1896_189669

theorem bounded_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end bounded_expression_l1896_189669


namespace functions_equal_at_three_l1896_189635

open Set

-- Define the open interval (2, 4)
def OpenInterval : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- Define the properties of functions f and g
def SatisfiesConditions (f g : ℝ → ℝ) : Prop :=
  ∀ x ∈ OpenInterval,
    (2 < f x ∧ f x < 4) ∧
    (2 < g x ∧ g x < 4) ∧
    (f (g x) = x) ∧
    (g (f x) = x) ∧
    (f x * g x = x^2)

-- Theorem statement
theorem functions_equal_at_three
  (f g : ℝ → ℝ)
  (h : SatisfiesConditions f g) :
  f 3 = g 3 := by
  sorry

end functions_equal_at_three_l1896_189635


namespace intersection_empty_implies_a_range_l1896_189686

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 2| ≤ a}
def B : Set ℝ := {x : ℝ | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_empty_implies_a_range (a : ℝ) :
  A a ∩ B = ∅ → a ∈ Set.Iio 1 := by
  sorry

end intersection_empty_implies_a_range_l1896_189686


namespace perfect_square_sum_l1896_189621

theorem perfect_square_sum (x : ℕ) : x = 12 → ∃ y : ℕ, 2^x + 2^8 + 2^11 = y^2 := by
  sorry

end perfect_square_sum_l1896_189621


namespace triangle_angle_measures_l1896_189691

theorem triangle_angle_measures :
  ∀ (A B C : ℝ),
  (A + B + C = 180) →
  (B = 2 * A) →
  (C + A + B = 180) →
  ∃ (x : ℝ),
    A = x ∧
    B = 2 * x ∧
    C = 180 - 3 * x :=
by sorry

end triangle_angle_measures_l1896_189691


namespace expression_factorization_l1896_189657

theorem expression_factorization (x y z : ℝ) :
  29.52 * x^2 * y - y^2 * z + z^2 * x - x^2 * z + y^2 * x + z^2 * y - 2 * x * y * z =
  (y - z) * (x + y) * (x - z) := by sorry

end expression_factorization_l1896_189657


namespace complex_additive_inverse_l1896_189658

theorem complex_additive_inverse (b : ℝ) : 
  let z : ℂ := (4 + b * Complex.I) / (1 + Complex.I)
  (z.re = -z.im) → b = 0 := by
sorry

end complex_additive_inverse_l1896_189658


namespace at_least_one_geq_two_l1896_189677

theorem at_least_one_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) :=
by sorry

end at_least_one_geq_two_l1896_189677


namespace urn_probability_l1896_189617

theorem urn_probability (N : ℚ) : 
  let urn1_green : ℚ := 5
  let urn1_blue : ℚ := 7
  let urn2_green : ℚ := 20
  let urn2_blue : ℚ := N
  let total_probability : ℚ := 65/100
  (urn1_green / (urn1_green + urn1_blue)) * (urn2_green / (urn2_green + urn2_blue)) +
  (urn1_blue / (urn1_green + urn1_blue)) * (urn2_blue / (urn2_green + urn2_blue)) = total_probability →
  N = 280/311 := by
sorry

end urn_probability_l1896_189617


namespace three_values_of_sum_l1896_189656

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Define the property that both domain and range are [a, b]
def domain_range_equal (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → a ≤ f x ∧ f x ≤ b) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

-- Theorem stating that there are exactly 3 different values of a+b
theorem three_values_of_sum :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
  (∀ x, x ∈ s ↔ ∃ a b, domain_range_equal a b ∧ a + b = x) :=
sorry

end three_values_of_sum_l1896_189656


namespace brick_surface_area_l1896_189616

theorem brick_surface_area :
  let length : ℝ := 8
  let width : ℝ := 4
  let height : ℝ := 2
  let surface_area := 2 * (length * width + length * height + width * height)
  surface_area = 112 :=
by sorry

end brick_surface_area_l1896_189616


namespace coconut_grove_problem_l1896_189648

theorem coconut_grove_problem (x : ℕ) : 
  (3 * 60 + 2 * 120 + x * 180 = 100 * (3 + 2 + x)) → x = 1 := by
  sorry

end coconut_grove_problem_l1896_189648


namespace intersection_P_Q_l1896_189668

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | ∃ y, Real.log (2 - x) = y}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = Set.Ioo 1 2 := by
  sorry

end intersection_P_Q_l1896_189668


namespace quadratic_no_roots_l1896_189618

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a given x -/
def QuadraticPolynomial.eval (f : QuadraticPolynomial) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The discriminant of a quadratic polynomial -/
def QuadraticPolynomial.discriminant (f : QuadraticPolynomial) : ℝ :=
  f.b^2 - 4 * f.a * f.c

/-- A function has exactly one solution when equal to a linear function -/
def has_exactly_one_solution (f : QuadraticPolynomial) (m : ℝ) (k : ℝ) : Prop :=
  ∃! x : ℝ, f.eval x = m * x + k

theorem quadratic_no_roots (f : QuadraticPolynomial) 
    (h1 : has_exactly_one_solution f 1 (-1))
    (h2 : has_exactly_one_solution f (-2) 2) :
    f.discriminant < 0 := by
  sorry

#check quadratic_no_roots

end quadratic_no_roots_l1896_189618


namespace well_depth_specific_well_depth_l1896_189683

/-- The depth of a cylindrical well given its diameter, cost per cubic meter, and total cost -/
theorem well_depth (diameter : ℝ) (cost_per_cubic_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let radius := diameter / 2
  let volume := total_cost / cost_per_cubic_meter
  let depth := volume / (Real.pi * radius^2)
  depth

/-- The depth of a specific well with given parameters is approximately 14 meters -/
theorem specific_well_depth : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000001 ∧ 
  |well_depth 3 16 1583.3626974092558 - 14| < ε :=
sorry

end well_depth_specific_well_depth_l1896_189683


namespace percentage_relation_l1896_189647

theorem percentage_relation (x y z : ℝ) 
  (h1 : x = 1.3 * y) 
  (h2 : y = 0.5 * z) : 
  x = 0.65 * z := by
sorry

end percentage_relation_l1896_189647


namespace angle_bisector_length_formulas_l1896_189666

theorem angle_bisector_length_formulas (a b c : ℝ) (α β γ : ℝ) (p R : ℝ) (l_a : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  α > 0 ∧ β > 0 ∧ γ > 0 ∧
  α + β + γ = π ∧
  p = (a + b + c) / 2 ∧
  R > 0 →
  (l_a = Real.sqrt (4 * p * (p - a) * b * c / ((b + c)^2))) ∧
  (l_a = 2 * b * c * Real.cos (α / 2) / (b + c)) ∧
  (l_a = 2 * R * Real.sin β * Real.sin γ / Real.cos ((β - γ) / 2)) ∧
  (l_a = 4 * p * Real.sin (β / 2) * Real.sin (γ / 2) / (Real.sin β + Real.sin γ)) :=
by sorry

end angle_bisector_length_formulas_l1896_189666


namespace salary_increase_percentage_l1896_189632

theorem salary_increase_percentage (S : ℝ) : 
  S * 1.1 = 770.0000000000001 → 
  S * (1 + 16 / 100) = 812 := by
sorry

end salary_increase_percentage_l1896_189632


namespace max_value_of_expression_l1896_189689

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^10 + 3*x^8 - 5*x^6 + 15*x^4 + 25) ≤ 1/17 ∧
  ∃ y : ℝ, y^6 / (y^10 + 3*y^8 - 5*y^6 + 15*y^4 + 25) = 1/17 :=
by sorry

end max_value_of_expression_l1896_189689


namespace patsy_appetizers_l1896_189697

def guests : ℕ := 30
def appetizers_per_guest : ℕ := 6
def deviled_eggs_dozens : ℕ := 3
def pigs_in_blanket_dozens : ℕ := 2
def kebabs_dozens : ℕ := 2
def appetizers_per_dozen : ℕ := 12

theorem patsy_appetizers : 
  (guests * appetizers_per_guest - 
   (deviled_eggs_dozens + pigs_in_blanket_dozens + kebabs_dozens) * appetizers_per_dozen) / 
   appetizers_per_dozen = 8 := by
sorry

end patsy_appetizers_l1896_189697


namespace chicken_entree_cost_l1896_189607

/-- Calculates the cost of each chicken entree given the wedding catering constraints. -/
theorem chicken_entree_cost
  (total_guests : ℕ)
  (steak_to_chicken_ratio : ℕ)
  (steak_cost : ℕ)
  (total_budget : ℕ)
  (h_total_guests : total_guests = 80)
  (h_ratio : steak_to_chicken_ratio = 3)
  (h_steak_cost : steak_cost = 25)
  (h_total_budget : total_budget = 1860) :
  (total_budget - steak_cost * (steak_to_chicken_ratio * total_guests / (steak_to_chicken_ratio + 1))) /
  (total_guests / (steak_to_chicken_ratio + 1)) = 18 := by
  sorry

#check chicken_entree_cost

end chicken_entree_cost_l1896_189607


namespace close_numbers_properties_l1896_189611

/-- A set of close numbers -/
structure CloseNumbers where
  n : ℕ
  numbers : Fin n → ℝ
  sum : ℝ
  n_gt_one : n > 1
  close : ∀ i, numbers i < sum / (n - 1)

/-- Theorems about close numbers -/
theorem close_numbers_properties (cn : CloseNumbers) :
  (∀ i, cn.numbers i > 0) ∧
  (∀ i j k, cn.numbers i + cn.numbers j > cn.numbers k) ∧
  (∀ i j, cn.numbers i + cn.numbers j > cn.sum / (cn.n - 1)) :=
by sorry

end close_numbers_properties_l1896_189611


namespace candy_container_volume_l1896_189675

theorem candy_container_volume (a b c : ℕ) (h : a * b * c = 216) :
  (3 * a) * (2 * b) * (4 * c) = 5184 := by
  sorry

end candy_container_volume_l1896_189675


namespace number_ordering_eight_ten_equals_four_fifteen_l1896_189603

theorem number_ordering : 8^10 < 3^20 ∧ 3^20 < 4^15 := by
  sorry

-- Additional theorem to establish the given condition
theorem eight_ten_equals_four_fifteen : 8^10 = 4^15 := by
  sorry

end number_ordering_eight_ten_equals_four_fifteen_l1896_189603


namespace twin_prime_divisibility_l1896_189659

theorem twin_prime_divisibility (p q : ℕ) : 
  Prime p → Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
sorry

end twin_prime_divisibility_l1896_189659


namespace textbook_packing_probability_l1896_189609

/-- Represents the problem of packing textbooks into boxes -/
structure TextbookPacking where
  total_books : Nat
  math_books : Nat
  box_sizes : Finset Nat

/-- The probability of all math books ending up in the same box -/
def probability_all_math_in_same_box (p : TextbookPacking) : ℚ :=
  sorry

/-- The main theorem stating the probability for the given problem -/
theorem textbook_packing_probability :
  let p := TextbookPacking.mk 15 4 {4, 5, 6}
  probability_all_math_in_same_box p = 27 / 1759 :=
sorry

end textbook_packing_probability_l1896_189609


namespace roses_in_vase_l1896_189604

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 6

/-- The number of roses Mary added to the vase -/
def added_roses : ℕ := 16

/-- The total number of roses in the vase after Mary added more -/
def total_roses : ℕ := initial_roses + added_roses

theorem roses_in_vase : total_roses = 22 := by
  sorry

end roses_in_vase_l1896_189604


namespace solution_set_characterization_l1896_189676

-- Define the set M
def M (a : ℝ) := {x : ℝ | x^2 + (a-4)*x - (a+1)*(2*a-3) < 0}

-- State the theorem
theorem solution_set_characterization (a : ℝ) :
  (0 ∈ M a) →
  ((a < -1 ∨ a > 3/2) ∧
   (a < -1 → M a = Set.Ioo (a+1) (3-2*a)) ∧
   (a > 3/2 → M a = Set.Ioo (3-2*a) (a+1))) :=
by sorry

end solution_set_characterization_l1896_189676


namespace festival_end_day_l1896_189600

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Theorem stating that 45 days after a 5-day festival starting on Tuesday is Wednesday -/
theorem festival_end_day (startDay : DayOfWeek) 
  (h : startDay = DayOfWeek.Tuesday) : 
  advanceDays startDay (5 + 45) = DayOfWeek.Wednesday := by
  sorry


end festival_end_day_l1896_189600
