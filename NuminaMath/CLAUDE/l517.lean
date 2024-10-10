import Mathlib

namespace a_value_l517_51777

def round_down_tens (n : ℕ) : ℕ :=
  (n / 10) * 10

theorem a_value (A : ℕ) : 
  A < 10 → 
  round_down_tens (A * 1000 + 567) = 2560 → 
  A = 2 := by
sorry

end a_value_l517_51777


namespace ant_count_approximation_l517_51799

/-- Represents the dimensions of a rectangular field in feet -/
structure FieldDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the number of ants in a rectangular field given the specified conditions -/
def calculateAnts (field : FieldDimensions) (antDensity : ℝ) (rockCoverage : ℝ) : ℝ :=
  let inchesPerFoot : ℝ := 12
  let fieldAreaInches : ℝ := field.width * field.length * inchesPerFoot * inchesPerFoot
  let antHabitatArea : ℝ := fieldAreaInches * (1 - rockCoverage)
  antHabitatArea * antDensity

/-- Theorem stating that the number of ants in the field is approximately 26 million -/
theorem ant_count_approximation :
  let field : FieldDimensions := { width := 200, length := 500 }
  let antDensity : ℝ := 2  -- ants per square inch
  let rockCoverage : ℝ := 0.1  -- 10% of the field covered by rocks
  abs (calculateAnts field antDensity rockCoverage - 26000000) ≤ 500000 := by
  sorry


end ant_count_approximation_l517_51799


namespace different_color_chips_probability_l517_51788

/-- The probability of drawing two chips of different colors from a bag containing
    7 blue chips, 5 red chips, 4 yellow chips, and 3 green chips, when drawing
    with replacement. -/
theorem different_color_chips_probability
  (blue : ℕ) (red : ℕ) (yellow : ℕ) (green : ℕ)
  (h_blue : blue = 7)
  (h_red : red = 5)
  (h_yellow : yellow = 4)
  (h_green : green = 3) :
  let total := blue + red + yellow + green
  (blue * (total - blue) + red * (total - red) + yellow * (total - yellow) + green * (total - green)) / (total * total) = 262 / 361 :=
by sorry

end different_color_chips_probability_l517_51788


namespace even_student_schools_count_l517_51735

/-- Represents a school with its student count -/
structure School where
  name : String
  students : ℕ

/-- Checks if a number is even -/
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

/-- Counts the number of schools with an even number of students -/
def countEvenStudentSchools (schools : List School) : ℕ :=
  (schools.filter (fun s => isEven s.students)).length

/-- The main theorem -/
theorem even_student_schools_count :
  let schools : List School := [
    ⟨"A", 786⟩,
    ⟨"B", 777⟩,
    ⟨"C", 762⟩,
    ⟨"D", 819⟩,
    ⟨"E", 493⟩
  ]
  countEvenStudentSchools schools = 2 := by
  sorry

end even_student_schools_count_l517_51735


namespace ab_squared_nonpositive_l517_51765

theorem ab_squared_nonpositive (a b : ℝ) (h : 7 * a + 9 * |b| = 0) : a * b^2 ≤ 0 := by
  sorry

end ab_squared_nonpositive_l517_51765


namespace total_cats_is_31_l517_51718

/-- The number of cats owned by Jamie, Gordon, Hawkeye, and Natasha -/
def total_cats : ℕ :=
  let jamie_persian := 4
  let jamie_maine_coon := 2
  let gordon_persian := jamie_persian / 2
  let gordon_maine_coon := jamie_maine_coon + 1
  let hawkeye_persian := 0
  let hawkeye_maine_coon := gordon_maine_coon * 2
  let natasha_persian := 3
  let natasha_maine_coon := jamie_maine_coon + gordon_maine_coon + hawkeye_maine_coon
  jamie_persian + jamie_maine_coon +
  gordon_persian + gordon_maine_coon +
  hawkeye_persian + hawkeye_maine_coon +
  natasha_persian + natasha_maine_coon

theorem total_cats_is_31 : total_cats = 31 := by
  sorry

end total_cats_is_31_l517_51718


namespace optimal_strategy_with_budget_optimal_strategy_without_budget_l517_51797

/-- Revenue function -/
def R (x₁ x₂ : ℝ) : ℝ := -2 * x₁^2 - x₂^2 + 13 * x₁ + 11 * x₂ - 28

/-- Profit function -/
def profit (x₁ x₂ : ℝ) : ℝ := R x₁ x₂ - (x₁ + x₂)

/-- Theorem for part 1 -/
theorem optimal_strategy_with_budget :
  ∀ x₁ x₂ : ℝ, x₁ + x₂ = 5 → profit x₁ x₂ ≤ profit 2 3 :=
sorry

/-- Theorem for part 2 -/
theorem optimal_strategy_without_budget :
  ∀ x₁ x₂ : ℝ, profit x₁ x₂ ≤ profit 3 5 :=
sorry

end optimal_strategy_with_budget_optimal_strategy_without_budget_l517_51797


namespace largest_value_l517_51705

theorem largest_value (x y z w : ℝ) (h : x + 3 = y - 1 ∧ y - 1 = z + 5 ∧ z + 5 = w - 2) :
  w ≥ x ∧ w ≥ y ∧ w ≥ z := by sorry

end largest_value_l517_51705


namespace quadratic_real_equal_roots_l517_51711

theorem quadratic_real_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 6 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 6 = 0 → y = x) ↔ 
  (m = 2 - 6 * Real.sqrt 2 ∨ m = 2 + 6 * Real.sqrt 2) :=
by sorry

end quadratic_real_equal_roots_l517_51711


namespace solve_widgets_problem_l517_51753

def widgets_problem (initial_widgets : ℕ) (total_money : ℕ) (price_reduction : ℕ) : Prop :=
  let initial_price := total_money / initial_widgets
  let new_price := initial_price - price_reduction
  let new_widgets := total_money / new_price
  new_widgets = 8

theorem solve_widgets_problem :
  widgets_problem 6 48 2 := by
  sorry

end solve_widgets_problem_l517_51753


namespace prob_odd_add_only_prob_odd_with_multiply_l517_51761

-- Define the calculator operations
inductive Operation
| Add
| Multiply

-- Define the calculator state
structure CalculatorState where
  display : ℕ
  lastOp : Option Operation

-- Define the probability of getting an odd number
def probOdd (ops : List Operation) : ℚ :=
  sorry

-- Theorem for part (a)
theorem prob_odd_add_only :
  ∀ (n : ℕ), probOdd (List.replicate n Operation.Add) = 1/2 :=
sorry

-- Theorem for part (b)
theorem prob_odd_with_multiply (n : ℕ) :
  probOdd (List.cons Operation.Multiply (List.replicate n Operation.Add)) < 1/2 :=
sorry

end prob_odd_add_only_prob_odd_with_multiply_l517_51761


namespace cyclic_sum_square_inequality_l517_51775

theorem cyclic_sum_square_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

end cyclic_sum_square_inequality_l517_51775


namespace preimage_of_four_l517_51703

def f (x : ℝ) : ℝ := x^2

theorem preimage_of_four (x : ℝ) : f x = 4 ↔ x = 2 ∨ x = -2 := by
  sorry

end preimage_of_four_l517_51703


namespace divisibility_property_l517_51751

theorem divisibility_property (a b c : ℤ) (h : 13 ∣ (a + b + c)) :
  13 ∣ (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) :=
by sorry

end divisibility_property_l517_51751


namespace bus_arrival_probability_l517_51787

/-- The probability of a bus arriving on time for a single ride -/
def p : ℝ := 0.9

/-- The number of total rides -/
def n : ℕ := 5

/-- The number of on-time arrivals we're interested in -/
def k : ℕ := 4

/-- The binomial probability of k successes in n trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)

theorem bus_arrival_probability :
  binomial_probability n k p = 0.328 := by
  sorry

end bus_arrival_probability_l517_51787


namespace kids_in_movie_l517_51768

theorem kids_in_movie (riverside_total : ℕ) (westside_total : ℕ) (mountaintop_total : ℕ)
  (riverside_denied_percent : ℚ) (westside_denied_percent : ℚ) (mountaintop_denied_percent : ℚ)
  (h1 : riverside_total = 120)
  (h2 : westside_total = 90)
  (h3 : mountaintop_total = 50)
  (h4 : riverside_denied_percent = 20/100)
  (h5 : westside_denied_percent = 70/100)
  (h6 : mountaintop_denied_percent = 1/2) :
  ↑riverside_total - ↑riverside_total * riverside_denied_percent +
  ↑westside_total - ↑westside_total * westside_denied_percent +
  ↑mountaintop_total - ↑mountaintop_total * mountaintop_denied_percent = 148 := by
  sorry

end kids_in_movie_l517_51768


namespace geometric_sequence_transformation_l517_51719

/-- Given a geometric sequence {a_n} with common ratio q (q ≠ 1),
    prove that the sequence {b_n} defined as b_n = a_{3n-2} + a_{3n-1} + a_{3n}
    is a geometric sequence with common ratio q^3. -/
theorem geometric_sequence_transformation (q : ℝ) (hq : q ≠ 1) (a : ℕ → ℝ) 
    (h_geom : ∀ n : ℕ, a (n + 1) = q * a n) :
  let b : ℕ → ℝ := λ n ↦ a (3 * n - 2) + a (3 * n - 1) + a (3 * n)
  ∀ n : ℕ, b (n + 1) = q^3 * b n := by
  sorry

end geometric_sequence_transformation_l517_51719


namespace volunteers_distribution_l517_51750

/-- The number of ways to distribute n volunteers into k schools,
    with each school receiving at least one volunteer. -/
def distribute_volunteers (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 75 volunteers into 3 schools,
    with each school receiving at least one volunteer, is equal to 150. -/
theorem volunteers_distribution :
  distribute_volunteers 75 3 = 150 := by sorry

end volunteers_distribution_l517_51750


namespace age_ratio_l517_51756

theorem age_ratio (current_age : ℕ) (years_ago : ℕ) : 
  current_age = 10 → 
  years_ago = 5 → 
  (current_age : ℚ) / ((current_age - years_ago) : ℚ) = 2 := by
sorry

end age_ratio_l517_51756


namespace math_department_candidates_l517_51720

theorem math_department_candidates :
  ∀ (m : ℕ),
    (∃ (cs_candidates : ℕ),
      cs_candidates = 7 ∧
      (Nat.choose cs_candidates 2) * m = 84) →
    m = 4 :=
by sorry

end math_department_candidates_l517_51720


namespace sum_of_powers_of_i_l517_51728

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i :
  1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i :=
by sorry

end sum_of_powers_of_i_l517_51728


namespace square_sum_from_means_l517_51700

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 110) : 
  x^2 + y^2 = 1380 := by
sorry

end square_sum_from_means_l517_51700


namespace power_23_mod_5_l517_51717

theorem power_23_mod_5 : 2^23 % 5 = 3 := by
  sorry

end power_23_mod_5_l517_51717


namespace imaginary_part_of_complex_fraction_l517_51715

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + 2*I) / (2 - I) → z.im = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l517_51715


namespace marks_speed_l517_51749

/-- Given a distance of 24 miles and a time of 4 hours, prove that the speed is 6 miles per hour. -/
theorem marks_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 24 ∧ time = 4 ∧ speed = distance / time → speed = 6 := by
  sorry

end marks_speed_l517_51749


namespace kishore_savings_l517_51759

def monthly_salary (expenses : ℕ) (savings_rate : ℚ) : ℚ :=
  expenses / (1 - savings_rate)

def savings (salary : ℚ) (savings_rate : ℚ) : ℚ :=
  salary * savings_rate

theorem kishore_savings (expenses : ℕ) (savings_rate : ℚ) :
  expenses = 18000 →
  savings_rate = 1/10 →
  savings (monthly_salary expenses savings_rate) savings_rate = 2000 := by
sorry

end kishore_savings_l517_51759


namespace juggling_balls_count_l517_51793

theorem juggling_balls_count (balls_per_juggler : ℕ) (number_of_jugglers : ℕ) (total_balls : ℕ) : 
  balls_per_juggler = 6 → 
  number_of_jugglers = 378 → 
  total_balls = balls_per_juggler * number_of_jugglers → 
  total_balls = 2268 := by
sorry

end juggling_balls_count_l517_51793


namespace series_sum_equals_257_l517_51736

def series_sum : ℕ := by
  -- Define the ranges for n, m, and p
  let n_range := Finset.range 12
  let m_range := Finset.range 3
  let p_range := Finset.range 2

  -- Define the summation functions
  let f_n (n : ℕ) := 2 * (n + 1)
  let f_m (m : ℕ) := 3 * (2 * m + 3)
  let f_p (p : ℕ) := 4 * (4 * p + 2)

  -- Calculate the sum
  exact (n_range.sum f_n) + (m_range.sum f_m) + (p_range.sum f_p)

theorem series_sum_equals_257 : series_sum = 257 := by
  sorry

end series_sum_equals_257_l517_51736


namespace jerry_can_carry_l517_51754

/-- Given the following conditions:
  * There are 28 cans to be recycled
  * The total time taken is 350 seconds
  * It takes 30 seconds to drain the cans
  * It takes 10 seconds to walk each way (to and from the sink/recycling bin)
  Prove that Jerry can carry 4 cans at once. -/
theorem jerry_can_carry (total_cans : ℕ) (total_time : ℕ) (drain_time : ℕ) (walk_time : ℕ) :
  total_cans = 28 →
  total_time = 350 →
  drain_time = 30 →
  walk_time = 10 →
  (total_time / (drain_time + 2 * walk_time) : ℚ) * (total_cans / (total_time / (drain_time + 2 * walk_time)) : ℚ) = 4 :=
by sorry

end jerry_can_carry_l517_51754


namespace power_calculation_l517_51713

theorem power_calculation : 4^2009 * (-0.25)^2008 - 1 = 3 := by
  sorry

end power_calculation_l517_51713


namespace sqrt_ratio_equality_l517_51722

theorem sqrt_ratio_equality : 
  (Real.sqrt (3^2 + 4^2)) / (Real.sqrt (25 + 16)) = (5 * Real.sqrt 41) / 41 := by
sorry

end sqrt_ratio_equality_l517_51722


namespace rational_roots_count_l517_51731

/-- The set of factors of a natural number -/
def factors (n : ℕ) : Finset ℤ :=
  sorry

/-- The set of possible rational roots for a polynomial with given leading coefficient and constant term -/
def possibleRationalRoots (leadingCoeff constTerm : ℤ) : Finset ℚ :=
  sorry

/-- Theorem stating that the number of different possible rational roots for the given polynomial form is 20 -/
theorem rational_roots_count :
  let leadingCoeff := 4
  let constTerm := 18
  (possibleRationalRoots leadingCoeff constTerm).card = 20 := by
  sorry

end rational_roots_count_l517_51731


namespace multiple_of_six_as_sum_of_four_cubes_l517_51767

theorem multiple_of_six_as_sum_of_four_cubes (k : ℤ) :
  ∃ (a b c d : ℤ), 6 * k = a^3 + b^3 + c^3 + d^3 :=
sorry

end multiple_of_six_as_sum_of_four_cubes_l517_51767


namespace exam_duration_l517_51781

/-- Proves that the examination time is 30 hours given the specified conditions -/
theorem exam_duration (total_questions : ℕ) (type_a_questions : ℕ) (type_a_time : ℝ) :
  total_questions = 200 →
  type_a_questions = 10 →
  type_a_time = 17.142857142857142 →
  (total_questions - type_a_questions) * (type_a_time / 2) + type_a_questions * type_a_time = 30 * 60 := by
  sorry

end exam_duration_l517_51781


namespace power_inequality_l517_51744

theorem power_inequality (a b : ℝ) (n : ℕ+) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : Odd b) 
  (h4 : (b ^ n.val) ∣ (a ^ n.val - 1)) : 
  a ^ (⌊b⌋) > 3 ^ n.val / n.val :=
sorry

end power_inequality_l517_51744


namespace projection_implies_y_value_l517_51796

/-- Given two vectors v and w in ℝ², prove that if the projection of v onto w
    is [-8, -12], then the y-coordinate of v must be -56/3. -/
theorem projection_implies_y_value (v w : ℝ × ℝ) (y : ℝ) 
    (h1 : v = (2, y))
    (h2 : w = (4, 6))
    (h3 : (v • w / (w • w)) • w = (-8, -12)) :
  y = -56/3 := by
  sorry

end projection_implies_y_value_l517_51796


namespace parallelogram_area_l517_51791

/-- A parallelogram bounded by lines y = a, y = -b, x = -c + 2y, and x = d - 2y -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d

/-- The area of the parallelogram -/
def area (p : Parallelogram) : ℝ :=
  p.a * p.d + p.a * p.c + p.b * p.d + p.b * p.c

theorem parallelogram_area (p : Parallelogram) :
  area p = p.a * p.d + p.a * p.c + p.b * p.d + p.b * p.c := by
  sorry

end parallelogram_area_l517_51791


namespace solution_value_l517_51758

theorem solution_value (x y m : ℝ) : 
  x = 2 ∧ y = -1 ∧ 2*x - 3*y = m → m = 7 := by sorry

end solution_value_l517_51758


namespace shobhas_current_age_l517_51712

/-- Given the ratio of Shekhar's age to Shobha's age and Shekhar's future age, 
    prove Shobha's current age -/
theorem shobhas_current_age 
  (shekhar_age shobha_age : ℕ) 
  (ratio : shekhar_age / shobha_age = 4 / 3)
  (future_age : shekhar_age + 6 = 26) : 
  shobha_age = 15 := by
  sorry

end shobhas_current_age_l517_51712


namespace sum_of_roots_cubic_sum_of_roots_specific_cubic_l517_51772

theorem sum_of_roots_cubic : ∀ (a b c d : ℝ),
  (∃ x y z : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ∧
                a * y^3 + b * y^2 + c * y + d = 0 ∧
                a * z^3 + b * z^2 + c * z + d = 0 ∧
                (∀ w : ℝ, a * w^3 + b * w^2 + c * w + d = 0 → w = x ∨ w = y ∨ w = z)) →
  x + y + z = -b / a :=
by sorry

theorem sum_of_roots_specific_cubic :
  ∃ x y z : ℝ, x^3 - 3*x^2 - 12*x - 7 = 0 ∧
              y^3 - 3*y^2 - 12*y - 7 = 0 ∧
              z^3 - 3*z^2 - 12*z - 7 = 0 ∧
              (∀ w : ℝ, w^3 - 3*w^2 - 12*w - 7 = 0 → w = x ∨ w = y ∨ w = z) ∧
              x + y + z = 3 :=
by sorry

end sum_of_roots_cubic_sum_of_roots_specific_cubic_l517_51772


namespace at_least_one_angle_le_30_deg_l517_51763

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define a point P
variable (P : Point)

-- Define that P is inside the triangle
def isInside (P : Point) (t : Triangle) : Prop := sorry

-- Define the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Theorem statement
theorem at_least_one_angle_le_30_deg (t : Triangle) (P : Point) 
  (h : isInside P t) : 
  (angle P t.A t.B ≤ 30) ∨ (angle P t.B t.C ≤ 30) ∨ (angle P t.C t.A ≤ 30) := by
  sorry

end at_least_one_angle_le_30_deg_l517_51763


namespace quarter_circles_sum_limit_l517_51742

theorem quarter_circles_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |2 * n * (π * D / (8 * n)) - (π * D / 4)| < ε :=
sorry

end quarter_circles_sum_limit_l517_51742


namespace hyperbola_focal_length_l517_51714

/-- The hyperbola C with parameter m -/
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

/-- The asymptote of the hyperbola C -/
def asymptote (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

/-- The focal length of a hyperbola -/
def focal_length (m : ℝ) : ℝ := sorry

theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) :
  (∀ x y : ℝ, hyperbola m x y ↔ asymptote m x y) →
  focal_length m = 4 := by sorry

end hyperbola_focal_length_l517_51714


namespace shopping_price_difference_l517_51783

/-- Proves that the difference between shoe price and bag price is $17 --/
theorem shopping_price_difference 
  (initial_amount : ℕ) 
  (shoe_price : ℕ) 
  (remaining_amount : ℕ) 
  (bag_price : ℕ) 
  (lunch_price : ℕ) 
  (h1 : initial_amount = 158)
  (h2 : shoe_price = 45)
  (h3 : remaining_amount = 78)
  (h4 : lunch_price = bag_price / 4)
  (h5 : initial_amount = shoe_price + bag_price + lunch_price + remaining_amount) :
  shoe_price - bag_price = 17 := by
  sorry

end shopping_price_difference_l517_51783


namespace negative_root_condition_l517_51709

theorem negative_root_condition (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 7^(x+1) - 7^x * a - a - 5 = 0) ↔ -5 < a ∧ a < 1 :=
by sorry

end negative_root_condition_l517_51709


namespace remaining_money_is_130_l517_51747

/-- Given an initial amount of money, calculate the remaining amount after spending on books and DVDs -/
def remaining_money (initial : ℚ) : ℚ :=
  let after_books := initial - (1/4 * initial + 10)
  let after_dvds := after_books - (2/5 * after_books + 8)
  after_dvds

/-- Theorem: Given $320 initially, the remaining money after buying books and DVDs is $130 -/
theorem remaining_money_is_130 : remaining_money 320 = 130 := by
  sorry

#eval remaining_money 320

end remaining_money_is_130_l517_51747


namespace max_area_rectangle_l517_51780

def is_valid_rectangle (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ x ≥ y

def cost (x y : ℕ) : ℕ :=
  2 * (3 * x + 5 * y)

def area (x y : ℕ) : ℕ :=
  x * y

theorem max_area_rectangle :
  ∃ (x y : ℕ), is_valid_rectangle x y ∧ cost x y ≤ 100 ∧
  area x y = 40 ∧
  ∀ (a b : ℕ), is_valid_rectangle a b → cost a b ≤ 100 → area a b ≤ 40 :=
by sorry

end max_area_rectangle_l517_51780


namespace sum_of_solutions_l517_51760

theorem sum_of_solutions (x y : ℝ) 
  (hx : (x - 1)^3 + 2015*(x - 1) = -1) 
  (hy : (y - 1)^3 + 2015*(y - 1) = 1) : 
  x + y = 2 := by
sorry

end sum_of_solutions_l517_51760


namespace cubic_roots_sum_l517_51730

/-- A cubic polynomial with rational coefficients -/
def CubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + p*x^2 + q*x + r
  where
  p := -(a + b + c)
  q := a*b + b*c + c*a
  r := -a*b*c

theorem cubic_roots_sum (a b c : ℝ) :
  (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c < 1) →
  CubicPolynomial a b c 0 = -1/8 →
  (∃ r : ℝ, b = a*r ∧ c = a*r^2) →
  (∑' k, (a^k + b^k + c^k)) = 9/2 →
  a + b + c = 19/12 := by
sorry

end cubic_roots_sum_l517_51730


namespace extreme_value_condition_l517_51792

/-- The function f(x) = ax + ln(x) has an extreme value -/
def has_extreme_value (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → a * x + Real.log x ≥ a * y + Real.log y) ∨
                   (∀ y : ℝ, y > 0 → a * x + Real.log x ≤ a * y + Real.log y)

/-- a ≤ 0 is a necessary but not sufficient condition for f(x) = ax + ln(x) to have an extreme value -/
theorem extreme_value_condition (a : ℝ) :
  has_extreme_value a → a ≤ 0 ∧ ∃ b : ℝ, b ≤ 0 ∧ ¬has_extreme_value b :=
sorry

end extreme_value_condition_l517_51792


namespace folded_paper_distance_l517_51702

/-- Given a square sheet of paper with area 18 cm², when folded so that a corner point A
    rests on the diagonal making the visible black area equal to the visible white area,
    the distance from A to its original position is 2√6 cm. -/
theorem folded_paper_distance (s : ℝ) (x : ℝ) :
  s^2 = 18 →
  (1/2) * x^2 = 18 - x^2 →
  Real.sqrt (2 * x^2) = 2 * Real.sqrt 6 := by
  sorry

end folded_paper_distance_l517_51702


namespace red_balls_count_l517_51726

theorem red_balls_count (total : ℕ) (p_black : ℚ) (p_at_least_one_white : ℚ) :
  total = 10 ∧ 
  p_black = 2/5 ∧ 
  p_at_least_one_white = 7/9 →
  ∃ (black white red : ℕ), 
    black + white + red = total ∧
    black = 4 ∧
    red = 1 ∧
    (black : ℚ) / total = p_black ∧
    1 - (Nat.choose (black + red) 2 : ℚ) / (Nat.choose total 2) = p_at_least_one_white :=
by
  sorry

#check red_balls_count

end red_balls_count_l517_51726


namespace jakes_weight_l517_51771

theorem jakes_weight (jake_weight sister_weight : ℝ) : 
  jake_weight - 12 = 2 * sister_weight →
  jake_weight + sister_weight = 156 →
  jake_weight = 108 := by
sorry

end jakes_weight_l517_51771


namespace parking_lot_wheels_l517_51774

/-- Calculates the total number of wheels in a parking lot with specific vehicles and conditions -/
theorem parking_lot_wheels : 
  let cars := 14
  let bikes := 5
  let unicycles := 3
  let twelve_wheeler_trucks := 2
  let eighteen_wheeler_truck := 1
  let cars_with_missing_wheel := 2
  let truck_with_damaged_wheels := 1
  let damaged_wheels := 3

  let car_wheels := cars * 4 - cars_with_missing_wheel * 1
  let bike_wheels := bikes * 2
  let unicycle_wheels := unicycles * 1
  let twelve_wheeler_truck_wheels := twelve_wheeler_trucks * 12 - damaged_wheels
  let eighteen_wheeler_truck_wheels := eighteen_wheeler_truck * 18

  car_wheels + bike_wheels + unicycle_wheels + twelve_wheeler_truck_wheels + eighteen_wheeler_truck_wheels = 106 := by
  sorry

end parking_lot_wheels_l517_51774


namespace ines_initial_amount_l517_51790

/-- The amount of money Ines had in her purse initially -/
def initial_amount : ℕ := 20

/-- The number of pounds of peaches Ines bought -/
def peaches_bought : ℕ := 3

/-- The cost per pound of peaches -/
def cost_per_pound : ℕ := 2

/-- The amount of money Ines had left after buying peaches -/
def amount_left : ℕ := 14

/-- Theorem stating that Ines had $20 in her purse initially -/
theorem ines_initial_amount :
  initial_amount = peaches_bought * cost_per_pound + amount_left :=
by sorry

end ines_initial_amount_l517_51790


namespace gcd_lcm_pairs_l517_51707

theorem gcd_lcm_pairs : 
  ∀ a b : ℕ, 
    a > 0 ∧ b > 0 →
    Nat.gcd a b = 24 ∧ Nat.lcm a b = 360 → 
    ((a = 24 ∧ b = 360) ∨ (a = 360 ∧ b = 24) ∨ (a = 72 ∧ b = 120) ∨ (a = 120 ∧ b = 72)) :=
by sorry

end gcd_lcm_pairs_l517_51707


namespace angle_B_range_l517_51778

theorem angle_B_range (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = 180) (h5 : A ≤ B) (h6 : B ≤ C) (h7 : 2 * B = 5 * A) :
  0 < B ∧ B ≤ 75 := by
sorry

end angle_B_range_l517_51778


namespace vector_subtraction_l517_51739

theorem vector_subtraction (u v : Fin 3 → ℝ) 
  (hu : u = ![-3, 5, 2]) 
  (hv : v = ![1, -1, 3]) : 
  u - 2 • v = ![-5, 7, -4] := by
  sorry

end vector_subtraction_l517_51739


namespace jessie_points_l517_51725

def total_points : ℕ := 311
def other_players_points : ℕ := 188
def num_equal_scorers : ℕ := 3

theorem jessie_points : 
  (total_points - other_players_points) / num_equal_scorers = 41 := by
  sorry

end jessie_points_l517_51725


namespace salary_restoration_l517_51733

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : 0 < original_salary) :
  let reduced_salary := original_salary * (1 - 0.25)
  let increase_factor := 1 + (1 / 3)
  reduced_salary * increase_factor = original_salary :=
by sorry

end salary_restoration_l517_51733


namespace time_saved_by_bike_l517_51710

/-- Given that it takes Mike 98 minutes to walk to school and riding a bicycle saves him 64 minutes,
    prove that the time saved by Mike when riding a bicycle is 64 minutes. -/
theorem time_saved_by_bike (walking_time : ℕ) (time_saved : ℕ) 
  (h1 : walking_time = 98) 
  (h2 : time_saved = 64) : 
  time_saved = 64 := by
  sorry

end time_saved_by_bike_l517_51710


namespace expression_simplification_l517_51762

theorem expression_simplification (a : ℝ) (h : a = 4) :
  (1 - (a + 1) / a) / ((a^2 - 1) / (a^2 - a)) = -1/5 := by
  sorry

end expression_simplification_l517_51762


namespace solve_for_y_l517_51708

theorem solve_for_y (x y : ℤ) (h1 : x^2 + 5 = y - 8) (h2 : x = -7) : y = 62 := by
  sorry

end solve_for_y_l517_51708


namespace journey_distance_l517_51740

/-- Proves that a journey with given conditions results in a total distance of 560 km -/
theorem journey_distance (total_time : ℝ) (speed_first_half : ℝ) (speed_second_half : ℝ) 
  (h1 : total_time = 25)
  (h2 : speed_first_half = 21)
  (h3 : speed_second_half = 24) :
  let total_distance := total_time * (speed_first_half + speed_second_half) / 2
  total_distance = 560 := by
  sorry

end journey_distance_l517_51740


namespace min_value_when_a_is_one_a_range_when_f_2_gt_5_l517_51701

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - a|

-- Part I
theorem min_value_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 2 :=
sorry

-- Part II
theorem a_range_when_f_2_gt_5 :
  ∀ a : ℝ, f a 2 > 5 → a < -5/2 ∨ a > 5/2 :=
sorry

end min_value_when_a_is_one_a_range_when_f_2_gt_5_l517_51701


namespace garden_perimeter_l517_51776

/-- A rectangular garden with given diagonal and area has a specific perimeter. -/
theorem garden_perimeter (a b : ℝ) : 
  a > 0 → b > 0 → -- Positive side lengths
  a^2 + b^2 = 15^2 → -- Diagonal condition
  a * b = 54 → -- Area condition
  2 * (a + b) = 2 * Real.sqrt 333 := by
  sorry

end garden_perimeter_l517_51776


namespace marias_green_towels_l517_51784

theorem marias_green_towels :
  ∀ (green_towels : ℕ),
  (green_towels + 21 : ℕ) - 34 = 22 →
  green_towels = 35 :=
by sorry

end marias_green_towels_l517_51784


namespace f_three_equals_three_l517_51743

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_three_equals_three :
  (∀ x, f (2 * x - 1) = x + 1) → f 3 = 3 := by
  sorry

end f_three_equals_three_l517_51743


namespace largest_angle_in_special_quadrilateral_l517_51789

/-- The measure of the largest angle in a quadrilateral with angles in the ratio 3:4:5:6 -/
theorem largest_angle_in_special_quadrilateral : 
  ∀ (a b c d : ℝ), 
  (a + b + c + d = 360) →  -- Sum of angles in a quadrilateral is 360°
  (∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k ∧ d = 6*k) →  -- Angles are in the ratio 3:4:5:6
  max a (max b (max c d)) = 120  -- The largest angle is 120°
:= by sorry

end largest_angle_in_special_quadrilateral_l517_51789


namespace captain_smollett_problem_l517_51766

theorem captain_smollett_problem :
  ∃! (a c l : ℕ), 
    0 < a ∧ a < 100 ∧
    c > 3 ∧
    l > 0 ∧
    a * c * l = 32118 ∧
    a = 53 ∧ c = 6 ∧ l = 101 := by
  sorry

end captain_smollett_problem_l517_51766


namespace problem_solution_l517_51741

theorem problem_solution (m n : ℝ) (h1 : m - n = 6) (h2 : m * n = 4) : 
  (m^2 + n^2 = 44) ∧ ((m + 2) * (n - 2) = -12) := by sorry

end problem_solution_l517_51741


namespace hyperbola_other_asymptote_l517_51724

/-- A hyperbola with given properties -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- The x-coordinate of the foci -/
  foci_x : ℝ
  /-- The hyperbola has a horizontal axis -/
  horizontal_axis : Prop

/-- The other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ := 
  fun x ↦ -2 * x + 16

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ 2 * x) 
  (h2 : h.foci_x = 4)
  (h3 : h.horizontal_axis) :
  other_asymptote h = fun x ↦ -2 * x + 16 := by
  sorry

end hyperbola_other_asymptote_l517_51724


namespace modulo_seventeen_residue_l517_51727

theorem modulo_seventeen_residue : (352 + 6 * 68 + 8 * 221 + 3 * 34 + 5 * 17) % 17 = 0 := by
  sorry

end modulo_seventeen_residue_l517_51727


namespace equation_solutions_l517_51746

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 - 3*x₁ - 4 = 0 ∧ x₂^2 - 3*x₂ - 4 = 0) ∧ x₁ = 4 ∧ x₂ = -1) ∧
  (∃ y₁ y₂ : ℝ, (y₁*(y₁ - 2) = 1 ∧ y₂*(y₂ - 2) = 1) ∧ y₁ = 1 + Real.sqrt 2 ∧ y₂ = 1 - Real.sqrt 2) :=
by sorry

end equation_solutions_l517_51746


namespace digit_2457_is_5_l517_51773

/-- The decimal number constructed by concatenating integers from 1 to 999 -/
def x : ℝ := sorry

/-- The nth digit after the decimal point in the number x -/
def digit_at (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 2457th digit of x is 5 -/
theorem digit_2457_is_5 : digit_at 2457 = 5 := by sorry

end digit_2457_is_5_l517_51773


namespace candy_difference_l517_51734

def frankie_candy : ℕ := 74
def max_candy : ℕ := 92

theorem candy_difference : max_candy - frankie_candy = 18 := by
  sorry

end candy_difference_l517_51734


namespace janet_snowball_percentage_l517_51752

/-- Given that Janet makes 50 snowballs and her brother makes 150 snowballs,
    prove that Janet made 25% of the total snowballs. -/
theorem janet_snowball_percentage
  (janet_snowballs : ℕ)
  (brother_snowballs : ℕ)
  (h1 : janet_snowballs = 50)
  (h2 : brother_snowballs = 150) :
  (janet_snowballs : ℚ) / (janet_snowballs + brother_snowballs) * 100 = 25 := by
  sorry

end janet_snowball_percentage_l517_51752


namespace dice_coloring_probability_l517_51769

/-- Represents the number of faces on a die -/
def numFaces : ℕ := 6

/-- Represents the number of color options for each face -/
def numColors : ℕ := 3

/-- Represents a coloring of a die -/
def DieColoring := Fin numFaces → Fin numColors

/-- Represents whether two die colorings are equivalent under rotation -/
def areEquivalentUnderRotation (d1 d2 : DieColoring) : Prop :=
  ∃ (rotation : Equiv.Perm (Fin numFaces)), ∀ i, d1 i = d2 (rotation i)

/-- The total number of ways to color two dice -/
def totalColorings : ℕ := numColors^numFaces * numColors^numFaces

/-- The number of ways to color two dice that are equivalent under rotation -/
def equivalentColorings : ℕ := 8425

theorem dice_coloring_probability :
  (equivalentColorings : ℚ) / totalColorings = 8425 / 531441 := by sorry

end dice_coloring_probability_l517_51769


namespace banana_orange_relation_bananas_to_oranges_l517_51757

/-- The value of one banana in terms of oranges -/
def banana_value : ℚ := 1

/-- The given relationship between bananas and oranges -/
theorem banana_orange_relation : (3/4 : ℚ) * 16 * banana_value = 12 := by sorry

/-- Theorem to prove: If 3/4 of 16 bananas are worth 12 oranges, 
    then 1/3 of 9 bananas are worth 3 oranges -/
theorem bananas_to_oranges : 
  ((1/3 : ℚ) * 9 * banana_value = 3) := by sorry

end banana_orange_relation_bananas_to_oranges_l517_51757


namespace mei_wendin_equation_theory_l517_51779

theorem mei_wendin_equation_theory :
  ∀ x y : ℚ,
  (3 * x + 6 * y = 47/10) →
  (5 * x + 3 * y = 11/2) →
  (x = 9/10 ∧ y = 1/3) :=
by
  sorry

end mei_wendin_equation_theory_l517_51779


namespace composition_fraction_l517_51706

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composition_fraction : f (g (f 3)) / g (f (g 3)) = 59 / 35 := by
  sorry

end composition_fraction_l517_51706


namespace pizza_toppings_theorem_l517_51721

/-- Given a number of pizza flavors and total pizza varieties (including pizzas with and without additional toppings), 
    calculate the number of possible additional toppings. -/
def calculate_toppings (flavors : ℕ) (total_varieties : ℕ) : ℕ :=
  (total_varieties / flavors) - 1

/-- Theorem stating that with 4 pizza flavors and 16 total pizza varieties, 
    there are 3 possible additional toppings. -/
theorem pizza_toppings_theorem :
  calculate_toppings 4 16 = 3 := by
  sorry

#eval calculate_toppings 4 16

end pizza_toppings_theorem_l517_51721


namespace extremum_at_one_decreasing_when_a_geq_two_monotonicity_when_a_lt_two_l517_51737

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := (2 - a) * x - 2 * Real.log x

def f_deriv (x : ℝ) : ℝ := 2 - a - 2 / x

theorem extremum_at_one (h : f_deriv a 1 = 0) : a = 0 := by sorry

theorem decreasing_when_a_geq_two (h : a ≥ 2) : 
  ∀ x > 0, f_deriv a x < 0 := by sorry

theorem monotonicity_when_a_lt_two (h : a < 2) :
  (∀ x ∈ Set.Ioo 0 (2 / (2 - a)), f_deriv a x < 0) ∧
  (∀ x ∈ Set.Ioi (2 / (2 - a)), f_deriv a x > 0) := by sorry

end extremum_at_one_decreasing_when_a_geq_two_monotonicity_when_a_lt_two_l517_51737


namespace min_value_of_x_plus_y_l517_51745

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y + 9 * x = x * y) :
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ y₀ + 9 * x₀ = x₀ * y₀ ∧ x₀ + y₀ = 16 :=
sorry

end min_value_of_x_plus_y_l517_51745


namespace sum_of_largest_and_smallest_prime_factors_of_1242_l517_51704

theorem sum_of_largest_and_smallest_prime_factors_of_1242 :
  ∃ (smallest largest : Nat),
    smallest.Prime ∧
    largest.Prime ∧
    smallest ∣ 1242 ∧
    largest ∣ 1242 ∧
    (∀ p : Nat, p.Prime → p ∣ 1242 → p ≤ largest) ∧
    (∀ p : Nat, p.Prime → p ∣ 1242 → p ≥ smallest) ∧
    smallest + largest = 25 :=
by sorry

end sum_of_largest_and_smallest_prime_factors_of_1242_l517_51704


namespace intersection_line_slope_l517_51786

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 11 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 14*x + 12*y + 60 = 0

-- Define the line passing through the intersection points
def intersectionLine (x y : ℝ) : Prop := 10*x - 10*y - 71 = 0

-- Theorem statement
theorem intersection_line_slope :
  ∀ (x1 y1 x2 y2 : ℝ),
  circle1 x1 y1 ∧ circle1 x2 y2 ∧
  circle2 x1 y1 ∧ circle2 x2 y2 ∧
  intersectionLine x1 y1 ∧ intersectionLine x2 y2 ∧
  x1 ≠ x2 →
  (y2 - y1) / (x2 - x1) = 1 :=
by sorry

end intersection_line_slope_l517_51786


namespace harriet_drive_time_l517_51782

theorem harriet_drive_time (total_time : ℝ) (outbound_speed return_speed : ℝ) 
  (h1 : total_time = 5)
  (h2 : outbound_speed = 100)
  (h3 : return_speed = 150) :
  let distance := (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed)
  let outbound_time := distance / outbound_speed
  outbound_time * 60 = 180 := by
sorry

end harriet_drive_time_l517_51782


namespace catholic_tower_height_l517_51785

/-- Given two towers and a grain between them, prove the height of the second tower --/
theorem catholic_tower_height 
  (church_height : ℝ) 
  (total_distance : ℝ) 
  (grain_distance : ℝ) 
  (h : ℝ → church_height = 150 ∧ total_distance = 350 ∧ grain_distance = 150) :
  ∃ (catholic_height : ℝ), 
    catholic_height = 50 * Real.sqrt 5 ∧ 
    (church_height^2 + grain_distance^2 = 
     catholic_height^2 + (total_distance - grain_distance)^2) :=
by sorry

end catholic_tower_height_l517_51785


namespace f_2016_value_l517_51723

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem f_2016_value (f : ℝ → ℝ) 
  (h1 : f 1 = 1/4)
  (h2 : functional_equation f) : 
  f 2016 = 1/2 := by
sorry

end f_2016_value_l517_51723


namespace simplify_expression_l517_51716

theorem simplify_expression : 8 * (15 / 4) * (-56 / 45) = -112 / 3 := by
  sorry

end simplify_expression_l517_51716


namespace log_equation_solution_l517_51794

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x^3 / Real.log 3 + Real.log x / Real.log (1/3) = 6 → x = 27 := by
  sorry

end log_equation_solution_l517_51794


namespace quadratic_root_bound_l517_51798

theorem quadratic_root_bound (a b c x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hx : a * x^2 + b * x + c = 0) : 
  |x| ≤ (2 * |a * c| + b^2) / (|a * b|) := by
  sorry

end quadratic_root_bound_l517_51798


namespace square_side_length_l517_51732

theorem square_side_length (d : ℝ) (h : d = Real.sqrt 8) :
  ∃ s : ℝ, s > 0 ∧ s * Real.sqrt 2 = d ∧ s = 2 := by
  sorry

end square_side_length_l517_51732


namespace decimal_to_percentage_l517_51770

theorem decimal_to_percentage (x : ℝ) : x = 1.20 → (x * 100 : ℝ) = 120 := by
  sorry

end decimal_to_percentage_l517_51770


namespace integers_abs_leq_three_l517_51795

theorem integers_abs_leq_three :
  {x : ℤ | |x| ≤ 3} = {-3, -2, -1, 0, 1, 2, 3} := by
sorry

end integers_abs_leq_three_l517_51795


namespace touching_circle_exists_l517_51729

-- Define the rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the configuration of circles
structure CircleConfiguration where
  rect : Rectangle
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle

-- Define the property that circles touch each other and the rectangle sides
def circlesValidConfiguration (config : CircleConfiguration) : Prop :=
  -- Circles touch each other
  (config.circle1.center.1 + config.circle1.radius = config.circle2.center.1 - config.circle2.radius) ∧
  (config.circle2.center.1 + config.circle2.radius = config.circle3.center.1 - config.circle3.radius) ∧
  -- Circles touch the rectangle sides
  (config.circle1.center.2 = config.circle1.radius) ∧
  (config.circle2.center.2 = config.rect.height - config.circle2.radius) ∧
  (config.circle3.center.2 = config.circle3.radius)

-- Define the existence of a circle touching all three circles and one side of the rectangle
def existsTouchingCircle (config : CircleConfiguration) : Prop :=
  ∃ (x : ℝ), x > 0 ∧
    -- The new circle touches circle1 and circle2
    (x + config.circle1.radius)^2 + config.circle1.radius^2 = (x + config.circle2.radius)^2 + (config.circle2.center.2 - config.circle1.center.2)^2 ∧
    -- The new circle touches circle2 and circle3
    (x + config.circle2.radius)^2 + (config.rect.height - config.circle2.center.2 - x)^2 = (x + config.circle3.radius)^2 + (config.circle3.center.2 - x)^2

-- The theorem to be proved
theorem touching_circle_exists (config : CircleConfiguration) 
  (h1 : config.circle1.radius = 1)
  (h2 : config.circle2.radius = 3)
  (h3 : config.circle3.radius = 4)
  (h4 : circlesValidConfiguration config) :
  existsTouchingCircle config :=
sorry

end touching_circle_exists_l517_51729


namespace optimal_purchase_l517_51738

def budget : ℕ := 100
def basic_calc_cost : ℕ := 8
def battery_cost : ℕ := 2
def scientific_calc_cost : ℕ := 2 * basic_calc_cost
def graphing_calc_cost : ℕ := 3 * scientific_calc_cost

def total_basic_cost : ℕ := basic_calc_cost + battery_cost
def total_scientific_cost : ℕ := scientific_calc_cost + battery_cost
def total_graphing_cost : ℕ := graphing_calc_cost + battery_cost

def one_of_each_cost : ℕ := total_basic_cost + total_scientific_cost + total_graphing_cost

theorem optimal_purchase :
  ∀ (b s g : ℕ),
    b ≥ 1 → s ≥ 1 → g ≥ 1 →
    (b + s + g) % 3 = 0 →
    b * total_basic_cost + s * total_scientific_cost + g * total_graphing_cost ≤ budget →
    b + s + g ≤ 3 ∧
    budget - (b * total_basic_cost + s * total_scientific_cost + g * total_graphing_cost) ≤ budget - one_of_each_cost :=
by sorry

end optimal_purchase_l517_51738


namespace parallel_vectors_condition_l517_51748

/-- Given plane vectors a and b, if a + b is parallel to a - b, then the second component of b is -2√3. -/
theorem parallel_vectors_condition (a b : ℝ × ℝ) :
  a = (1, -Real.sqrt 3) →
  b.1 = 2 →
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (a - b)) →
  b.2 = -2 * Real.sqrt 3 := by
  sorry

end parallel_vectors_condition_l517_51748


namespace vector_projection_l517_51764

/-- Given vectors a and b in ℝ², prove that the projection of a onto 2√3b is √65/5 -/
theorem vector_projection (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b = (-4, 7)) :
  let proj := (a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2) * Real.sqrt 3 * 2
  proj = Real.sqrt 65 / 5 := by
  sorry

end vector_projection_l517_51764


namespace garden_area_increase_l517_51755

/-- Proves that changing a 60-foot by 20-foot rectangular garden to a square garden 
    with the same perimeter results in an increase of 400 square feet in area. -/
theorem garden_area_increase : 
  let rectangle_length : ℝ := 60
  let rectangle_width : ℝ := 20
  let rectangle_area := rectangle_length * rectangle_width
  let perimeter := 2 * (rectangle_length + rectangle_width)
  let square_side := perimeter / 4
  let square_area := square_side * square_side
  square_area - rectangle_area = 400 := by
  sorry

end garden_area_increase_l517_51755
