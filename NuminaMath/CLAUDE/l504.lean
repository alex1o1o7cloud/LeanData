import Mathlib

namespace find_b_l504_50409

def gcd_notation (x y : ℕ) : ℕ := x * y

theorem find_b : ∃ b : ℕ, gcd_notation (gcd_notation (16 * b) (18 * 24)) 2 = 2 ∧ b = 1 := by sorry

end find_b_l504_50409


namespace salary_calculation_l504_50408

theorem salary_calculation (salary : ℚ) 
  (food : ℚ) (rent : ℚ) (clothes : ℚ) (transport : ℚ) (personal_care : ℚ) 
  (remaining : ℚ) :
  food = 1/4 * salary →
  rent = 1/6 * salary →
  clothes = 3/8 * salary →
  transport = 1/12 * salary →
  personal_care = 1/24 * salary →
  remaining = 45000 →
  salary - (food + rent + clothes + transport + personal_care) = remaining →
  salary = 540000 := by
sorry

end salary_calculation_l504_50408


namespace fluffy_spotted_cats_ratio_l504_50441

theorem fluffy_spotted_cats_ratio (total_cats : ℕ) (fluffy_spotted_cats : ℕ) :
  total_cats = 120 →
  fluffy_spotted_cats = 10 →
  (total_cats / 3 : ℚ) = (total_cats / 3 : ℕ) →
  (fluffy_spotted_cats : ℚ) / (total_cats / 3 : ℚ) = 1 / 4 := by
  sorry

end fluffy_spotted_cats_ratio_l504_50441


namespace cannot_form_square_l504_50474

/-- Represents the collection of sticks --/
structure StickCollection where
  twoLengthCount : Nat
  threeLengthCount : Nat
  sevenLengthCount : Nat

/-- Checks if it's possible to form a square with given sticks --/
def canFormSquare (sticks : StickCollection) : Prop :=
  ∃ (side : ℕ), 
    4 * side = 2 * sticks.twoLengthCount + 
               3 * sticks.threeLengthCount + 
               7 * sticks.sevenLengthCount ∧
    ∃ (a b c : ℕ), 
      a + b + c = 4 ∧
      a * 2 + b * 3 + c * 7 = 4 * side ∧
      a ≤ sticks.twoLengthCount ∧
      b ≤ sticks.threeLengthCount ∧
      c ≤ sticks.sevenLengthCount

/-- The given collection of sticks --/
def givenSticks : StickCollection :=
  { twoLengthCount := 5
    threeLengthCount := 5
    sevenLengthCount := 1 }

theorem cannot_form_square : ¬(canFormSquare givenSticks) := by
  sorry

end cannot_form_square_l504_50474


namespace min_plus_arg_is_pi_third_l504_50433

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * Real.cos (2 * x) + 16) - Real.sin x ^ 2

def has_min (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x ≥ m

def is_smallest_positive_min (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  has_min f m ∧ f n = m ∧ n > 0 ∧ ∀ x, 0 < x ∧ x < n → f x > m

theorem min_plus_arg_is_pi_third :
  ∃ (m n : ℝ), is_smallest_positive_min f m n ∧ m + n = Real.pi / 3 :=
sorry

end min_plus_arg_is_pi_third_l504_50433


namespace egg_problem_l504_50413

theorem egg_problem (x : ℕ) : x > 0 ∧ 
  x % 2 = 1 ∧ 
  x % 3 = 1 ∧ 
  x % 4 = 1 ∧ 
  x % 5 = 1 ∧ 
  x % 6 = 1 ∧ 
  x % 7 = 0 → 
  x ≥ 301 :=
by sorry

end egg_problem_l504_50413


namespace curve_properties_l504_50459

structure Curve where
  m : ℝ
  n : ℝ
  equation : ℝ → ℝ → Prop

def isEllipse (C : Curve) : Prop := sorry

def hasYAxisFoci (C : Curve) : Prop := sorry

def isHyperbola (C : Curve) : Prop := sorry

def hasAsymptotes (C : Curve) (f : ℝ → ℝ) : Prop := sorry

def isTwoLines (C : Curve) : Prop := sorry

theorem curve_properties (C : Curve) 
  (h_eq : C.equation = fun x y ↦ C.m * x^2 + C.n * y^2 = 1) :
  (C.m > C.n ∧ C.n > 0 → isEllipse C ∧ hasYAxisFoci C) ∧
  (C.m * C.n < 0 → isHyperbola C ∧ hasAsymptotes C (fun x ↦ Real.sqrt (-C.m / C.n) * x)) ∧
  (C.m = 0 ∧ C.n > 0 → isTwoLines C) := by
  sorry

end curve_properties_l504_50459


namespace function_form_l504_50483

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem function_form (f : ℕ → ℕ) 
  (h1 : StrictlyIncreasing f)
  (h2 : ∀ x y : ℕ, ∃ k : ℕ+, (f x + f y) / (1 + f (x + y)) = k) :
  ∃ a : ℕ+, ∀ x : ℕ, f x = a * x + 1 :=
sorry

end function_form_l504_50483


namespace positive_polynomial_fraction_representation_l504_50497

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- A polynomial with non-negative coefficients -/
def NonNegativePolynomial (p : RealPolynomial) : Prop :=
  ∀ i, (p.coeff i) ≥ 0

/-- The theorem statement -/
theorem positive_polynomial_fraction_representation
  (P : RealPolynomial) (h : ∀ x : ℝ, x > 0 → P.eval x > 0) :
  ∃ (Q R : RealPolynomial), NonNegativePolynomial Q ∧ NonNegativePolynomial R ∧
    ∀ x : ℝ, x ≠ 0 → P.eval x = (Q.eval x) / (R.eval x) := by
  sorry

end positive_polynomial_fraction_representation_l504_50497


namespace sqrt_meaningful_range_l504_50431

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 3) → x ≥ -3 := by
  sorry

end sqrt_meaningful_range_l504_50431


namespace min_value_a1_plus_a7_l504_50489

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a n > 0 ∧ a (n + 1) = r * a n

/-- The theorem stating the minimum value of a₁ + a₇ in a positive geometric sequence where a₃ * a₅ = 64 -/
theorem min_value_a1_plus_a7 (a : ℕ → ℝ) 
    (h_geom : is_positive_geometric_sequence a) 
    (h_prod : a 3 * a 5 = 64) : 
  (∀ b : ℕ → ℝ, is_positive_geometric_sequence b → b 3 * b 5 = 64 → a 1 + a 7 ≤ b 1 + b 7) → 
  a 1 + a 7 = 16 := by
sorry

end min_value_a1_plus_a7_l504_50489


namespace cost_for_23_days_l504_50430

/-- Calculates the total cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 14
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- Theorem stating that the cost for a 23-day stay is $350.00 -/
theorem cost_for_23_days :
  hostelCost 23 = 350 := by
  sorry

#eval hostelCost 23

end cost_for_23_days_l504_50430


namespace total_jellybeans_proof_l504_50422

/-- The number of jellybeans needed to fill a large drinking glass -/
def large_glass_beans : ℕ := 50

/-- The number of large drinking glasses -/
def num_large_glasses : ℕ := 5

/-- The number of small drinking glasses -/
def num_small_glasses : ℕ := 3

/-- The number of jellybeans needed to fill a small drinking glass -/
def small_glass_beans : ℕ := large_glass_beans / 2

/-- The total number of jellybeans needed to fill all glasses -/
def total_beans : ℕ := large_glass_beans * num_large_glasses + small_glass_beans * num_small_glasses

theorem total_jellybeans_proof : total_beans = 325 := by
  sorry

end total_jellybeans_proof_l504_50422


namespace selena_bashar_passes_l504_50485

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the lane in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def number_of_passes (runner1 runner2 : Runner) (total_time : ℝ) (delay : ℝ) : ℕ :=
  sorry

theorem selena_bashar_passes : 
  let selena : Runner := ⟨200, 70, 1⟩
  let bashar : Runner := ⟨240, 80, -1⟩
  let total_time : ℝ := 35
  let delay : ℝ := 5
  number_of_passes selena bashar total_time delay = 21 := by
  sorry

end selena_bashar_passes_l504_50485


namespace stratified_sampling_possible_after_adjustment_l504_50421

/-- Represents the population sizes of different age groups -/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Represents the sampling parameters -/
structure SamplingParams where
  population : Population
  sampleSize : Nat

/-- Checks if stratified sampling is possible with equal sampling fractions -/
def canStratifySample (p : SamplingParams) : Prop :=
  ∃ (k : Nat), k > 0 ∧
    k ∣ p.sampleSize ∧
    k ∣ p.population.elderly ∧
    k ∣ p.population.middleAged ∧
    k ∣ p.population.young

/-- The given population and sample size -/
def givenParams : SamplingParams :=
  { population := { elderly := 28, middleAged := 54, young := 81 },
    sampleSize := 36 }

/-- The adjusted parameters after removing one elderly person -/
def adjustedParams : SamplingParams :=
  { population := { elderly := 27, middleAged := 54, young := 81 },
    sampleSize := 36 }

/-- Theorem stating that stratified sampling becomes possible after adjustment -/
theorem stratified_sampling_possible_after_adjustment :
  ¬canStratifySample givenParams ∧ canStratifySample adjustedParams :=
sorry


end stratified_sampling_possible_after_adjustment_l504_50421


namespace largest_angle_in_3_4_5_ratio_triangle_l504_50411

theorem largest_angle_in_3_4_5_ratio_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  (a + b + c = 180) →
  (b = (4/3) * a) →
  (c = (5/3) * a) →
  c = 75 := by
sorry

end largest_angle_in_3_4_5_ratio_triangle_l504_50411


namespace inequality_solution_set_l504_50424

theorem inequality_solution_set (x : ℝ) : 
  |5*x - x^2| < 6 ↔ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) := by
  sorry

end inequality_solution_set_l504_50424


namespace diophantine_equation_solutions_l504_50491

theorem diophantine_equation_solutions :
  ∀ a b c d : ℕ, 2^a * 3^b - 5^c * 7^d = 1 ↔
    (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
    (a = 3 ∧ b = 0 ∧ c = 0 ∧ d = 1) ∨
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 0) ∨
    (a = 2 ∧ b = 2 ∧ c = 1 ∧ d = 1) :=
by sorry


end diophantine_equation_solutions_l504_50491


namespace miles_reading_pages_l504_50463

/-- Calculates the total number of pages read by Miles --/
def total_pages_read (hours_in_day : ℝ) (reading_fraction : ℝ) 
  (novel_pages_per_hour : ℝ) (graphic_novel_pages_per_hour : ℝ) (comic_book_pages_per_hour : ℝ) 
  (fraction_per_book_type : ℝ) : ℝ :=
  let total_reading_hours := hours_in_day * reading_fraction
  let hours_per_book_type := total_reading_hours * fraction_per_book_type
  let novel_pages := novel_pages_per_hour * hours_per_book_type
  let graphic_novel_pages := graphic_novel_pages_per_hour * hours_per_book_type
  let comic_book_pages := comic_book_pages_per_hour * hours_per_book_type
  novel_pages + graphic_novel_pages + comic_book_pages

/-- Theorem stating that Miles reads 128 pages given the problem conditions --/
theorem miles_reading_pages : 
  total_pages_read 24 (1/6) 21 30 45 (1/3) = 128 := by
  sorry

end miles_reading_pages_l504_50463


namespace tony_puzzle_solution_l504_50439

/-- The number of puzzles Tony solved after the warm-up puzzle -/
def puzzles_after_warmup : ℕ := 2

/-- The time taken for the warm-up puzzle in minutes -/
def warmup_time : ℕ := 10

/-- The total time Tony spent solving puzzles in minutes -/
def total_time : ℕ := 70

/-- Each puzzle after the warm-up takes this many times longer than the warm-up -/
def puzzle_time_multiplier : ℕ := 3

theorem tony_puzzle_solution :
  warmup_time + puzzles_after_warmup * (puzzle_time_multiplier * warmup_time) = total_time :=
by sorry

end tony_puzzle_solution_l504_50439


namespace f_is_even_l504_50488

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end f_is_even_l504_50488


namespace evaluate_expression_l504_50482

theorem evaluate_expression : 6 - 5 * (9 - 2^3) * 3 = -9 := by
  sorry

end evaluate_expression_l504_50482


namespace divisor_problem_l504_50466

theorem divisor_problem (n d k m : ℤ) : 
  n = k * d + 4 → 
  n + 15 = 5 * m + 4 → 
  d = 5 := by sorry

end divisor_problem_l504_50466


namespace dodecagon_diagonal_intersection_probability_l504_50487

/-- A regular dodecagon is a 12-sided polygon with all sides equal and all angles equal. -/
def RegularDodecagon : Type := Unit

/-- A diagonal of a regular dodecagon is a line segment connecting two non-adjacent vertices. -/
def Diagonal (d : RegularDodecagon) : Type := Unit

/-- The probability that two randomly chosen diagonals of a regular dodecagon intersect inside the polygon. -/
def intersectionProbability (d : RegularDodecagon) : ℚ :=
  165 / 287

/-- Theorem: The probability that the intersection of two randomly chosen diagonals 
    of a regular dodecagon lies inside the polygon is 165/287. -/
theorem dodecagon_diagonal_intersection_probability (d : RegularDodecagon) :
  intersectionProbability d = 165 / 287 :=
by
  sorry

end dodecagon_diagonal_intersection_probability_l504_50487


namespace divisibility_condition_l504_50444

theorem divisibility_condition (m n : ℕ+) :
  (∃ k : ℤ, 4 * (m.val * n.val + 1) = k * (m.val + n.val)^2) ↔ m = n :=
sorry

end divisibility_condition_l504_50444


namespace cos_sum_diff_product_leq_cos_sq_l504_50486

theorem cos_sum_diff_product_leq_cos_sq (x y : ℝ) :
  Real.cos (x + y) * Real.cos (x - y) ≤ Real.cos x ^ 2 := by
  sorry

end cos_sum_diff_product_leq_cos_sq_l504_50486


namespace geese_percentage_among_non_swans_l504_50490

theorem geese_percentage_among_non_swans 
  (total_percentage : ℝ) 
  (geese_percentage : ℝ) 
  (swan_percentage : ℝ) 
  (h1 : total_percentage = 100) 
  (h2 : geese_percentage = 20) 
  (h3 : swan_percentage = 25) : 
  (geese_percentage / (total_percentage - swan_percentage)) * 100 = 26.67 := by
sorry

end geese_percentage_among_non_swans_l504_50490


namespace oil_spend_is_500_l504_50458

/-- Represents the price reduction, amount difference, and reduced price of oil --/
structure OilPriceData where
  reduction_percent : ℚ
  amount_difference : ℚ
  reduced_price : ℚ

/-- Calculates the amount spent on oil given the price reduction data --/
def calculate_oil_spend (data : OilPriceData) : ℚ :=
  let original_price := data.reduced_price / (1 - data.reduction_percent)
  let m := data.amount_difference * (data.reduced_price * original_price) / (original_price - data.reduced_price)
  m

/-- Theorem stating that given the specific conditions, the amount spent on oil is 500 --/
theorem oil_spend_is_500 (data : OilPriceData) 
  (h1 : data.reduction_percent = 1/4)
  (h2 : data.amount_difference = 5)
  (h3 : data.reduced_price = 25) : 
  calculate_oil_spend data = 500 := by
  sorry

end oil_spend_is_500_l504_50458


namespace floor_fraction_equals_eight_l504_50471

theorem floor_fraction_equals_eight (n : ℕ) (h : n = 2006) : 
  ⌊(8 * (n^2 + 1 : ℝ)) / (n^2 - 1 : ℝ)⌋ = 8 := by
  sorry

end floor_fraction_equals_eight_l504_50471


namespace arithmetic_sequence_sum_l504_50449

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a₁ + a₂ = 5 and a₃ + a₄ = 7,
    prove that a₅ + a₆ = 9 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 = 5)
  (h_sum2 : a 3 + a 4 = 7) :
  a 5 + a 6 = 9 := by
  sorry


end arithmetic_sequence_sum_l504_50449


namespace four_balls_four_boxes_l504_50403

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls_boxes (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 5 ways to distribute 4 indistinguishable balls into 4 indistinguishable boxes -/
theorem four_balls_four_boxes : distribute_balls_boxes 4 4 = 5 := by
  sorry

end four_balls_four_boxes_l504_50403


namespace arithmetic_geometric_sequence_l504_50447

/-- An arithmetic sequence with common difference 2 -/
def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Condition that a_1, a_3, and a_4 form a geometric sequence -/
def geometricSubsequence (a : ℕ → ℤ) : Prop :=
  (a 3) ^ 2 = a 1 * a 4

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) 
  (h_arith : arithmeticSequence a) 
  (h_geom : geometricSubsequence a) : 
  a 2 = -6 := by sorry

end arithmetic_geometric_sequence_l504_50447


namespace boys_to_total_ratio_l504_50451

theorem boys_to_total_ratio 
  (b g : ℕ) -- number of boys and girls
  (h1 : b > 0 ∧ g > 0) -- ensure non-empty class
  (h2 : (b : ℚ) / (b + g) = 4/5 * (g : ℚ) / (b + g)) -- probability condition
  : (b : ℚ) / (b + g) = 4/9 := by
  sorry

end boys_to_total_ratio_l504_50451


namespace chemistry_lab_workstations_l504_50484

theorem chemistry_lab_workstations (total_capacity : ℕ) (total_workstations : ℕ) 
  (three_student_stations : ℕ) (remaining_stations : ℕ) 
  (h1 : total_capacity = 38)
  (h2 : total_workstations = 16)
  (h3 : three_student_stations = 6)
  (h4 : remaining_stations = 10)
  (h5 : total_workstations = three_student_stations + remaining_stations) :
  ∃ (students_per_remaining : ℕ),
    students_per_remaining * remaining_stations + 3 * three_student_stations = total_capacity ∧
    students_per_remaining * remaining_stations = 20 :=
by sorry

end chemistry_lab_workstations_l504_50484


namespace correct_probability_l504_50443

/-- The number of options for the first three digits -/
def first_three_options : ℕ := 3

/-- The number of remaining digits to arrange -/
def remaining_digits : ℕ := 5

/-- The probability of correctly guessing the phone number -/
def probability_correct_guess : ℚ := 1 / (first_three_options * remaining_digits.factorial)

theorem correct_probability :
  probability_correct_guess = 1 / 360 := by
  sorry

end correct_probability_l504_50443


namespace stratified_sampling_theorem_l504_50423

theorem stratified_sampling_theorem (total_population : ℕ) (category_size : ℕ) (sample_size : ℕ) 
  (h1 : total_population = 100)
  (h2 : category_size = 30)
  (h3 : sample_size = 20) :
  (category_size : ℚ) / total_population * sample_size = 6 := by
  sorry

end stratified_sampling_theorem_l504_50423


namespace abs_is_even_l504_50453

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def abs_function (x : ℝ) : ℝ := |x|

theorem abs_is_even : is_even_function abs_function := by
  sorry

end abs_is_even_l504_50453


namespace oil_price_reduction_l504_50404

/-- Proves that given a 35% reduction in oil price allowing 5 kg more for Rs. 800, the reduced price is Rs. 36.4 per kg -/
theorem oil_price_reduction (original_price : ℝ) : 
  (800 / (0.65 * original_price) - 800 / original_price = 5) →
  (0.65 * original_price = 36.4) := by
sorry

end oil_price_reduction_l504_50404


namespace power_2017_mod_11_l504_50415

theorem power_2017_mod_11 : 2^2017 % 11 = 7 := by
  sorry

end power_2017_mod_11_l504_50415


namespace ellipse_point_Q_l504_50410

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the condition for point M
def M_condition (M : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  (mx - 2) * (mx + 2) + my^2 = 0  -- MB ⊥ AB

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  let (px, py) := P
  ellipse px py  -- P is on the ellipse

-- Define the condition for point Q
def Q_condition (Q : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  qy = 0 ∧ qx ≠ -2 ∧ qx ≠ 2  -- Q is on x-axis and distinct from A and B

-- Define the circle condition
def circle_condition (M P Q : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  let (px, py) := P
  let (qx, qy) := Q
  ∃ (I : ℝ × ℝ), 
    (I.1 - px) * (mx - px) + (I.2 - py) * (my - py) = 0 ∧  -- I is on BP
    (I.1 - mx) * (qx - mx) + (I.2 - my) * (qy - my) = 0 ∧  -- I is on MQ
    (I.1 - (mx + px) / 2)^2 + (I.2 - (my + py) / 2)^2 = ((mx - px)^2 + (my - py)^2) / 4  -- I is on the circle

theorem ellipse_point_Q : 
  ∀ (M P Q : ℝ × ℝ),
    M_condition M →
    P_condition P →
    Q_condition Q →
    circle_condition M P Q →
    Q = (0, 0) := by sorry

end ellipse_point_Q_l504_50410


namespace equation_solutions_l504_50473

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, (x₁ + 5)^2 = 16 ∧ (x₂ + 5)^2 = 16 ∧ x₁ = -9 ∧ x₂ = -1) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ - 12 = 0 ∧ y₂^2 - 4*y₂ - 12 = 0 ∧ y₁ = 6 ∧ y₂ = -2) :=
by
  sorry

#check equation_solutions

end equation_solutions_l504_50473


namespace innocent_statement_l504_50400

/-- Represents the type of person making a statement --/
inductive PersonType
| Knight
| Liar
| Normal

/-- Represents a statement that can be made --/
inductive Statement
| IAmALiar

/-- Defines whether a statement is true or false --/
def isTrue : PersonType → Statement → Prop
| PersonType.Knight, Statement.IAmALiar => False
| PersonType.Liar, Statement.IAmALiar => False
| PersonType.Normal, Statement.IAmALiar => True

theorem innocent_statement :
  ∀ (p : PersonType), p ≠ PersonType.Normal → ¬(isTrue p Statement.IAmALiar) := by
  sorry

end innocent_statement_l504_50400


namespace pages_copied_for_fifty_dollars_l504_50417

/-- Given that 4 pages can be copied for 10 cents, prove that $50 allows for copying 2000 pages. -/
theorem pages_copied_for_fifty_dollars (cost_per_four_pages : ℚ) (pages_per_fifty_dollars : ℕ) :
  cost_per_four_pages = 10 / 100 →
  pages_per_fifty_dollars = 2000 :=
by sorry

end pages_copied_for_fifty_dollars_l504_50417


namespace father_twice_son_age_l504_50461

/-- Represents the ages of a father and son --/
structure Ages where
  sonPast : ℕ
  fatherPast : ℕ
  sonNow : ℕ
  fatherNow : ℕ

/-- The conditions of the problem --/
def ageConditions (a : Ages) : Prop :=
  a.fatherPast = 3 * a.sonPast ∧
  a.sonNow = a.sonPast + 18 ∧
  a.fatherNow = a.fatherPast + 18 ∧
  a.sonNow + a.fatherNow = 108 ∧
  ∃ k : ℕ, a.fatherNow = k * a.sonNow

/-- The theorem to be proved --/
theorem father_twice_son_age (a : Ages) (h : ageConditions a) : a.fatherNow = 2 * a.sonNow := by
  sorry

end father_twice_son_age_l504_50461


namespace johns_raise_l504_50418

/-- Proves that if an amount x is increased by 9.090909090909092% to reach $60, then x is equal to $55 -/
theorem johns_raise (x : ℝ) : 
  x * (1 + 0.09090909090909092) = 60 → x = 55 := by
  sorry

end johns_raise_l504_50418


namespace min_sticks_removal_part_a_result_part_b_result_l504_50445

/-- Represents a rectangular fence made of sticks -/
structure Fence where
  m : Nat
  n : Nat
  sticks : Nat

/-- The number of ants in a fence is equal to the number of 1x1 squares -/
def num_ants (f : Fence) : Nat := f.m * f.n

/-- The minimum number of sticks to remove for all ants to escape -/
def min_sticks_to_remove (f : Fence) : Nat := num_ants f

/-- Theorem: The minimum number of sticks to remove for all ants to escape
    is equal to the number of ants in the fence -/
theorem min_sticks_removal (f : Fence) :
  min_sticks_to_remove f = num_ants f :=
by sorry

/-- Corollary: For a 1x4 fence with 13 sticks, 4 sticks need to be removed -/
theorem part_a_result :
  min_sticks_to_remove ⟨1, 4, 13⟩ = 4 :=
by sorry

/-- Corollary: For a 4x4 fence with 24 sticks, 9 sticks need to be removed -/
theorem part_b_result :
  min_sticks_to_remove ⟨4, 4, 24⟩ = 9 :=
by sorry

end min_sticks_removal_part_a_result_part_b_result_l504_50445


namespace f_ln_2_equals_3_l504_50436

-- Define a monotonically increasing function
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_ln_2_equals_3 
  (f : ℝ → ℝ)
  (h_mono : MonoIncreasing f)
  (h_prop : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1) : 
  f (Real.log 2) = 3 := by
sorry


end f_ln_2_equals_3_l504_50436


namespace l_triomino_division_l504_50450

/-- An L-triomino is a shape with 3 squares formed by removing one square from a 2x2 grid. -/
def L_triomino_area : ℕ := 3

/-- Theorem: A 1961 × 1963 grid rectangle cannot be exactly divided into L-triominoes,
    but a 1963 × 1965 rectangle can be exactly divided into L-triominoes. -/
theorem l_triomino_division :
  (¬ (1961 * 1963) % L_triomino_area = 0) ∧
  ((1963 * 1965) % L_triomino_area = 0) := by
  sorry

end l_triomino_division_l504_50450


namespace complex_fraction_equality_l504_50498

/-- Given that i is the imaginary unit, prove that (1+3i)/(1+i) = 2+i -/
theorem complex_fraction_equality : (1 + 3 * I) / (1 + I) = 2 + I := by
  sorry

end complex_fraction_equality_l504_50498


namespace sum_of_f_values_negative_l504_50438

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem sum_of_f_values_negative
  (f : ℝ → ℝ)
  (h_decreasing : is_monotonically_decreasing f)
  (h_odd : is_odd_function f)
  (x₁ x₂ x₃ : ℝ)
  (h₁₂ : x₁ + x₂ > 0)
  (h₂₃ : x₂ + x₃ > 0)
  (h₃₁ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
sorry

end sum_of_f_values_negative_l504_50438


namespace interest_rate_proof_l504_50414

/-- Proves that the annual interest rate is 5% given the specified conditions -/
theorem interest_rate_proof (principal : ℝ) (time : ℕ) (amount : ℝ) :
  principal = 973.913043478261 →
  time = 3 →
  amount = 1120 →
  (amount - principal) / (principal * time) = 0.05 := by
  sorry

end interest_rate_proof_l504_50414


namespace rosie_lou_speed_ratio_l504_50475

/-- The ratio of Rosie's speed to Lou's speed on a circular track -/
theorem rosie_lou_speed_ratio :
  let track_length : ℚ := 1/4  -- Length of the track in miles
  let lou_distance : ℚ := 3    -- Lou's total distance in miles
  let rosie_laps : ℕ := 24     -- Number of laps Rosie completes
  let rosie_distance : ℚ := rosie_laps * track_length  -- Rosie's total distance in miles
  ∀ (lou_speed rosie_speed : ℚ),
    lou_speed > 0 →  -- Lou's speed is positive
    rosie_speed > 0 →  -- Rosie's speed is positive
    lou_speed * lou_distance = rosie_speed * rosie_distance →  -- They run for the same duration
    rosie_speed / lou_speed = 2/1 :=
by sorry

end rosie_lou_speed_ratio_l504_50475


namespace six_bottle_caps_cost_l504_50481

/-- The cost of a given number of bottle caps -/
def bottle_cap_cost (num_caps : ℕ) (cost_per_cap : ℕ) : ℕ :=
  num_caps * cost_per_cap

/-- Theorem: The cost of 6 bottle caps at $2 each is $12 -/
theorem six_bottle_caps_cost : bottle_cap_cost 6 2 = 12 := by
  sorry

end six_bottle_caps_cost_l504_50481


namespace min_value_of_expression_min_value_attained_l504_50440

theorem min_value_of_expression (x : ℝ) : 
  (14 - x) * (9 - x) * (14 + x) * (9 + x) ≥ -1156.25 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (14 - x) * (9 - x) * (14 + x) * (9 + x) = -1156.25 :=
by sorry

end min_value_of_expression_min_value_attained_l504_50440


namespace manuscript_revision_cost_l504_50495

/-- The cost per page for revision in a manuscript typing service --/
def revision_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) 
  (initial_cost : ℕ) (total_cost : ℕ) : ℚ :=
  let pages_not_revised := total_pages - revised_once - revised_twice
  let initial_typing_cost := total_pages * initial_cost
  let revision_pages := revised_once + 2 * revised_twice
  (total_cost - initial_typing_cost : ℚ) / revision_pages

/-- Theorem stating the revision cost for the given manuscript --/
theorem manuscript_revision_cost :
  revision_cost 200 80 20 5 1360 = 3 := by
  sorry

end manuscript_revision_cost_l504_50495


namespace tank_volume_l504_50479

-- Define the rates of the pipes
def inlet_rate : ℝ := 3
def outlet_rate_1 : ℝ := 9
def outlet_rate_2 : ℝ := 6

-- Define the time it takes to empty the tank
def emptying_time : ℝ := 4320

-- Define the conversion factor from cubic inches to cubic feet
def cubic_inches_per_cubic_foot : ℝ := 1728

-- State the theorem
theorem tank_volume (net_rate : ℝ) (volume_cubic_inches : ℝ) (volume_cubic_feet : ℝ) 
  (h1 : net_rate = outlet_rate_1 + outlet_rate_2 - inlet_rate)
  (h2 : volume_cubic_inches = net_rate * emptying_time)
  (h3 : volume_cubic_feet = volume_cubic_inches / cubic_inches_per_cubic_foot) :
  volume_cubic_feet = 30 := by
  sorry

end tank_volume_l504_50479


namespace dozen_pens_cost_is_600_l504_50494

/-- The cost of a pen in rupees -/
def pen_cost : ℚ := sorry

/-- The cost of a pencil in rupees -/
def pencil_cost : ℚ := sorry

/-- The cost ratio of a pen to a pencil -/
def cost_ratio : ℚ := 5

/-- The total cost of 3 pens and 5 pencils in rupees -/
def total_cost : ℚ := 200

/-- The cost of one dozen pens in rupees -/
def dozen_pens_cost : ℚ := 12 * pen_cost

theorem dozen_pens_cost_is_600 :
  pen_cost = 5 * pencil_cost ∧
  3 * pen_cost + 5 * pencil_cost = total_cost →
  dozen_pens_cost = 600 := by
  sorry

end dozen_pens_cost_is_600_l504_50494


namespace intersection_max_value_l504_50452

def P (a b : ℝ) (x : ℝ) : ℝ := x^6 - 8*x^5 + 24*x^4 - 37*x^3 + a*x^2 + b*x - 6

def L (d : ℝ) (x : ℝ) : ℝ := d*x + 2

theorem intersection_max_value (a b d : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ 
    P a b x = L d x ∧
    P a b y = L d y ∧
    P a b z = L d z ∧
    (∀ t : ℝ, t ≠ z → (P a b t - L d t) / (t - z) ≠ 0)) →
  (∃ w : ℝ, P a b w = L d w ∧ ∀ v : ℝ, P a b v = L d v → v ≤ w ∧ w = 5) :=
by sorry

end intersection_max_value_l504_50452


namespace three_inscribed_circles_exist_l504_50468

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Three circles are inscribed in a larger circle --/
structure InscribedCircles where
  outer : Circle
  inner1 : Circle
  inner2 : Circle
  inner3 : Circle

/-- The property of three circles being equal --/
def equal_circles (c1 c2 c3 : Circle) : Prop :=
  c1.radius = c2.radius ∧ c2.radius = c3.radius

/-- The property of two circles being tangent --/
def tangent_circles (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- The property of a circle being inscribed in another circle --/
def inscribed_circle (outer inner : Circle) : Prop :=
  let (x1, y1) := outer.center
  let (x2, y2) := inner.center
  (x2 - x1)^2 + (y2 - y1)^2 = (outer.radius - inner.radius)^2

/-- Theorem: Three equal circles can be inscribed in a larger circle,
    such that they are tangent to each other and to the larger circle --/
theorem three_inscribed_circles_exist (outer : Circle) :
  ∃ (ic : InscribedCircles),
    ic.outer = outer ∧
    equal_circles ic.inner1 ic.inner2 ic.inner3 ∧
    tangent_circles ic.inner1 ic.inner2 ∧
    tangent_circles ic.inner2 ic.inner3 ∧
    tangent_circles ic.inner3 ic.inner1 ∧
    inscribed_circle outer ic.inner1 ∧
    inscribed_circle outer ic.inner2 ∧
    inscribed_circle outer ic.inner3 :=
  sorry

end three_inscribed_circles_exist_l504_50468


namespace median_is_212_l504_50420

/-- The sum of integers from 1 to n -/
def triangularSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total count of numbers in our special list up to n -/
def cumulativeCount (n : ℕ) : ℕ := triangularSum n

/-- The total length of our special list -/
def totalLength : ℕ := triangularSum 300

/-- The position of the lower median element -/
def lowerMedianPos : ℕ := totalLength / 2

/-- The position of the upper median element -/
def upperMedianPos : ℕ := lowerMedianPos + 1

theorem median_is_212 : 
  ∃ (n : ℕ), n = 212 ∧ 
  cumulativeCount (n - 1) < lowerMedianPos ∧
  cumulativeCount n ≥ upperMedianPos :=
sorry

end median_is_212_l504_50420


namespace early_finish_hours_l504_50470

/-- Represents the number of workers -/
def num_workers : ℕ := 3

/-- Represents the normal working hours per day -/
def normal_hours : ℕ := 8

/-- Represents the number of customers served per hour by each worker -/
def customers_per_hour : ℕ := 7

/-- Represents the total number of customers served that day -/
def total_customers : ℕ := 154

/-- Theorem stating that the worker who finished early worked for 6 hours -/
theorem early_finish_hours :
  ∃ (h : ℕ),
    h < normal_hours ∧
    (2 * normal_hours * customers_per_hour + h * customers_per_hour = total_customers) ∧
    h = 6 :=
sorry

end early_finish_hours_l504_50470


namespace company_results_l504_50427

structure Company where
  team_a_success_prob : ℚ
  team_b_success_prob : ℚ
  profit_a_success : ℤ
  loss_a_failure : ℤ
  profit_b_success : ℤ
  loss_b_failure : ℤ

def company : Company := {
  team_a_success_prob := 3/4,
  team_b_success_prob := 3/5,
  profit_a_success := 120,
  loss_a_failure := 50,
  profit_b_success := 100,
  loss_b_failure := 40
}

def exactly_one_success_prob (c : Company) : ℚ :=
  (1 - c.team_a_success_prob) * c.team_b_success_prob +
  c.team_a_success_prob * (1 - c.team_b_success_prob)

def profit_distribution (c : Company) : List (ℤ × ℚ) :=
  [(-90, (1 - c.team_a_success_prob) * (1 - c.team_b_success_prob)),
   (50, (1 - c.team_a_success_prob) * c.team_b_success_prob),
   (80, c.team_a_success_prob * (1 - c.team_b_success_prob)),
   (220, c.team_a_success_prob * c.team_b_success_prob)]

theorem company_results :
  exactly_one_success_prob company = 9/20 ∧
  profit_distribution company = [(-90, 1/10), (50, 3/20), (80, 3/10), (220, 9/20)] := by
  sorry

end company_results_l504_50427


namespace tobias_driveways_shoveled_l504_50419

/-- Calculates the number of driveways Tobias shoveled given his earnings and expenses. -/
theorem tobias_driveways_shoveled (shoe_cost allowance_per_month lawn_mowing_charge driveway_shoveling_charge change_after_purchase : ℕ) (months_saved lawns_mowed : ℕ) : 
  shoe_cost = 95 →
  allowance_per_month = 5 →
  months_saved = 3 →
  lawn_mowing_charge = 15 →
  driveway_shoveling_charge = 7 →
  change_after_purchase = 15 →
  lawns_mowed = 4 →
  (shoe_cost + change_after_purchase - months_saved * allowance_per_month - lawns_mowed * lawn_mowing_charge) / driveway_shoveling_charge = 5 := by
sorry

end tobias_driveways_shoveled_l504_50419


namespace west_is_negative_of_east_l504_50401

/-- Represents distance and direction, where positive values indicate east and negative values indicate west. -/
def Distance := ℤ

/-- Converts a distance in kilometers to the corresponding Distance representation. -/
def km_to_distance (x : ℤ) : Distance := x

/-- The distance representation for 2km east. -/
def two_km_east : Distance := km_to_distance 2

/-- The distance representation for 1km west. -/
def one_km_west : Distance := km_to_distance (-1)

theorem west_is_negative_of_east (h : two_km_east = km_to_distance 2) :
  one_km_west = km_to_distance (-1) := by sorry

end west_is_negative_of_east_l504_50401


namespace joan_gemstone_count_l504_50496

/-- Proves that Joan has 21 gemstone samples given the conditions of the problem -/
theorem joan_gemstone_count :
  ∀ (minerals_yesterday minerals_today gemstones : ℕ),
    gemstones = minerals_yesterday / 2 →
    minerals_today = minerals_yesterday + 6 →
    minerals_today = 48 →
    gemstones = 21 := by
  sorry

end joan_gemstone_count_l504_50496


namespace min_value_of_f_l504_50437

def f (x a : ℝ) : ℝ := 2 * x^2 - 4 * a * x + a^2 + 2 * a + 2

theorem min_value_of_f (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → f x a ≥ 2) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f x a = 2) →
  a = -3 - Real.sqrt 7 ∨ a = 0 ∨ a = 2 ∨ a = 4 :=
sorry

end min_value_of_f_l504_50437


namespace distance_to_x_axis_reflection_triangle_DEF_reflection_distance_l504_50426

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_x_axis_reflection (x y : ℝ) : 
  Real.sqrt ((x - x)^2 + ((-y) - y)^2) = 2 * |y| := by
  sorry

/-- The specific case for the triangle DEF --/
theorem triangle_DEF_reflection_distance : 
  Real.sqrt ((2 - 2)^2 + ((-1) - 1)^2) = 2 := by
  sorry

end distance_to_x_axis_reflection_triangle_DEF_reflection_distance_l504_50426


namespace chess_tournament_success_ratio_l504_50429

theorem chess_tournament_success_ratio (charlie_day1_score charlie_day1_attempted charlie_day2_score charlie_day2_attempted : ℕ) : 
  -- Total points for both players
  charlie_day1_attempted + charlie_day2_attempted = 600 →
  -- Charlie's scores are positive integers
  charlie_day1_score > 0 →
  charlie_day2_score > 0 →
  -- Charlie's daily success ratios are less than Alpha's
  charlie_day1_score * 360 < 180 * charlie_day1_attempted →
  charlie_day2_score * 240 < 120 * charlie_day2_attempted →
  -- Charlie did not attempt 360 points on day 1
  charlie_day1_attempted ≠ 360 →
  -- The maximum two-day success ratio for Charlie
  (charlie_day1_score + charlie_day2_score : ℚ) / 600 ≤ 299 / 600 :=
by sorry

end chess_tournament_success_ratio_l504_50429


namespace consecutive_integers_sum_of_cubes_l504_50467

theorem consecutive_integers_sum_of_cubes (a b c : ℕ) : 
  (a > 0) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (a^2 + b^2 + c^2 = 2450) → 
  (a^3 + b^3 + c^3 = 73341) :=
by sorry

end consecutive_integers_sum_of_cubes_l504_50467


namespace martinez_family_height_l504_50469

def chiquita_height : ℝ := 5

def mr_martinez_height : ℝ := chiquita_height + 2

def mrs_martinez_height : ℝ := chiquita_height - 1

def son_height : ℝ := chiquita_height + 3

def combined_height : ℝ := chiquita_height + mr_martinez_height + mrs_martinez_height + son_height

theorem martinez_family_height : combined_height = 24 := by
  sorry

end martinez_family_height_l504_50469


namespace special_line_equation_l504_50412

/-- A line passing through point (2, 3) with intercepts on the coordinate axes that are opposite numbers -/
structure SpecialLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2, 3)
  passes_through : m * 2 + b = 3
  -- The intercepts are opposite numbers
  opposite_intercepts : b = m * b

theorem special_line_equation (L : SpecialLine) :
  (L.m = 3/2 ∧ L.b = 0) ∨ (L.m = 1 ∧ L.b = -1) :=
sorry

end special_line_equation_l504_50412


namespace largest_number_value_l504_50446

theorem largest_number_value (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 100)
  (h_diff_large : c - b = 10)
  (h_diff_small : b - a = 5) :
  c = 41.67 := by
  sorry

end largest_number_value_l504_50446


namespace joan_balloons_l504_50465

theorem joan_balloons (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 8 → received = 2 → total = initial + received → total = 10 := by
  sorry

end joan_balloons_l504_50465


namespace complex_equation_solution_l504_50476

theorem complex_equation_solution (x : ℝ) : 
  (1 - 2*Complex.I) * (x + Complex.I) = 4 - 3*Complex.I → x = 2 := by
  sorry

end complex_equation_solution_l504_50476


namespace unique_integer_solution_l504_50402

theorem unique_integer_solution :
  ∃! (x : ℤ), x - 8 / (x - 2) = 5 - 8 / (x - 2) :=
by
  -- Proof goes here
  sorry

end unique_integer_solution_l504_50402


namespace trig_identity_l504_50477

theorem trig_identity (α : Real) (h : Real.tan α = 2) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 11 / 5 := by
  sorry

end trig_identity_l504_50477


namespace imaginary_part_of_i_times_one_minus_i_l504_50432

theorem imaginary_part_of_i_times_one_minus_i (i : ℂ) : 
  (Complex.I * (1 - Complex.I)).im = 1 := by sorry

end imaginary_part_of_i_times_one_minus_i_l504_50432


namespace tom_dancing_hours_l504_50472

/-- Calculates the total dancing hours over multiple years -/
def total_dancing_hours (sessions_per_week : ℕ) (hours_per_session : ℕ) (years : ℕ) (weeks_per_year : ℕ) : ℕ :=
  sessions_per_week * hours_per_session * years * weeks_per_year

/-- Proves that Tom's total dancing hours over 10 years is 4160 -/
theorem tom_dancing_hours : 
  total_dancing_hours 4 2 10 52 = 4160 := by
  sorry

#eval total_dancing_hours 4 2 10 52

end tom_dancing_hours_l504_50472


namespace unique_solution_l504_50428

def equation (y : ℝ) : Prop :=
  y ≠ 0 ∧ y ≠ 3 ∧ (3 * y^2 - 15 * y) / (y^2 - 3 * y) = y + 1

theorem unique_solution :
  ∃! y : ℝ, equation y :=
sorry

end unique_solution_l504_50428


namespace inequalities_hold_l504_50493

theorem inequalities_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := by
  sorry

end inequalities_hold_l504_50493


namespace find_A_l504_50407

theorem find_A : ∃ A B : ℚ, A - 3 * B = 303.1 ∧ A = 10 * B → A = 433 := by
  sorry

end find_A_l504_50407


namespace union_of_A_and_B_l504_50448

def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

end union_of_A_and_B_l504_50448


namespace sara_pumpkins_l504_50492

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem sara_pumpkins : pumpkins_eaten 43 20 = 23 := by
  sorry

end sara_pumpkins_l504_50492


namespace sum_product_equality_l504_50499

theorem sum_product_equality : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := by
  sorry

end sum_product_equality_l504_50499


namespace composition_of_even_is_even_l504_50480

/-- A function f is even if f(-x) = f(x) for all x. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- Given an even function f, prove that f(f(x)) is also even. -/
theorem composition_of_even_is_even (f : ℝ → ℝ) (hf : IsEven f) : IsEven (f ∘ f) := by
  sorry

end composition_of_even_is_even_l504_50480


namespace square_of_101_l504_50478

theorem square_of_101 : 101 * 101 = 10201 := by
  sorry

end square_of_101_l504_50478


namespace marys_story_characters_l504_50457

theorem marys_story_characters (total : ℕ) (init_a init_c init_d init_e : ℕ) : 
  total = 60 →
  init_a = total / 2 →
  init_c = init_a / 2 →
  init_d + init_e = total - init_a - init_c →
  init_d = 2 * init_e →
  init_d = 10 := by
  sorry

end marys_story_characters_l504_50457


namespace addition_preserves_inequality_l504_50434

theorem addition_preserves_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 := by
  sorry

end addition_preserves_inequality_l504_50434


namespace min_queries_needed_l504_50416

/-- Represents a quadratic polynomial ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates a quadratic polynomial at a given point -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Represents Petia's strategy of choosing which polynomial value to return -/
def PetiaStrategy := ℕ → Bool

/-- Represents Vasya's strategy of choosing query points -/
def VasyaStrategy := ℕ → ℝ

/-- Determines if Vasya can identify one of Petia's polynomials after n queries -/
def canIdentifyPolynomial (f g : QuadraticPolynomial) (petiaStrat : PetiaStrategy) (vasyaStrat : VasyaStrategy) (n : ℕ) : Prop :=
  ∃ (i : Fin n), 
    let x := vasyaStrat i
    let y := if petiaStrat i then evaluate f x else evaluate g x
    ∀ (f' g' : QuadraticPolynomial), 
      (∀ (j : Fin n), 
        let x' := vasyaStrat j
        let y' := if petiaStrat j then evaluate f' x' else evaluate g' x'
        y' = if petiaStrat j then evaluate f x' else evaluate g x') →
      f' = f ∨ g' = g

/-- The main theorem: 8 is the smallest number of queries needed -/
theorem min_queries_needed : 
  (∃ (vasyaStrat : VasyaStrategy), ∀ (f g : QuadraticPolynomial) (petiaStrat : PetiaStrategy), 
    canIdentifyPolynomial f g petiaStrat vasyaStrat 8) ∧ 
  (∀ (n : ℕ), n < 8 → 
    ∀ (vasyaStrat : VasyaStrategy), ∃ (f g : QuadraticPolynomial) (petiaStrat : PetiaStrategy), 
      ¬canIdentifyPolynomial f g petiaStrat vasyaStrat n) := by
  sorry

end min_queries_needed_l504_50416


namespace white_marble_probability_l504_50405

theorem white_marble_probability (total_marbles : ℕ) 
  (p_green p_red_or_blue : ℝ) : 
  total_marbles = 84 →
  p_green = 2 / 7 →
  p_red_or_blue = 0.4642857142857143 →
  1 - (p_green + p_red_or_blue) = 0.25 := by
  sorry

end white_marble_probability_l504_50405


namespace correct_seating_arrangements_l504_50456

-- Define the number of students and rows
def num_students : ℕ := 12
def num_rows : ℕ := 2
def students_per_row : ℕ := num_students / num_rows

-- Define the number of test versions
def num_versions : ℕ := 2

-- Define the function to calculate the number of seating arrangements
def seating_arrangements : ℕ := 2 * (Nat.factorial students_per_row)^2

-- Theorem statement
theorem correct_seating_arrangements :
  seating_arrangements = 1036800 :=
sorry

end correct_seating_arrangements_l504_50456


namespace solution_to_equation_l504_50406

theorem solution_to_equation : ∃! x : ℤ, (2008 + x)^2 = x^2 ∧ x = -1004 := by sorry

end solution_to_equation_l504_50406


namespace parabola_focus_distance_l504_50455

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -32*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (-8, 0)

-- Define a point on the parabola
def point_on_parabola (x₀ : ℝ) : ℝ × ℝ := (x₀, 4)

-- State the theorem
theorem parabola_focus_distance (x₀ : ℝ) :
  parabola x₀ 4 →
  let P := point_on_parabola x₀
  let F := focus
  dist P F = 17/2 := by sorry

end parabola_focus_distance_l504_50455


namespace symmetric_circles_line_l504_50464

-- Define the circles
def C₁ (x y a : ℝ) : Prop := x^2 + y^2 - a = 0
def C₂ (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 2*a*y + 3 = 0

-- Define the line
def line_l (x y : ℝ) : Prop := 2*x - 4*y + 5 = 0

-- State the theorem
theorem symmetric_circles_line (a : ℝ) :
  (∀ x y, C₁ x y a ↔ C₂ (2*x + 1) (2*y - a) a) →
  (∀ x y, line_l x y ↔ (∃ x₀ y₀, C₁ x₀ y₀ a ∧ C₂ (2*x - x₀) (2*y - y₀) a)) :=
sorry

end symmetric_circles_line_l504_50464


namespace probability_below_curve_probability_is_one_third_l504_50460

/-- The probability that a randomly chosen point in the unit square falls below the curve y = x^2 is 1/3 -/
theorem probability_below_curve : Real → Prop := λ p =>
  let curve := λ x : Real => x^2
  let unit_square_area := 1
  let area_below_curve := ∫ x in (0 : Real)..1, curve x
  p = area_below_curve / unit_square_area ∧ p = 1/3

/-- The main theorem stating the probability is 1/3 -/
theorem probability_is_one_third : ∃ p : Real, probability_below_curve p := by
  sorry

end probability_below_curve_probability_is_one_third_l504_50460


namespace brick_length_is_correct_l504_50442

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 25

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 11.25

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 6

/-- The length of the wall in centimeters -/
def wall_length : ℝ := 700

/-- The height of the wall in centimeters -/
def wall_height : ℝ := 600

/-- The width of the wall in centimeters -/
def wall_width : ℝ := 22.5

/-- The number of bricks needed to build the wall -/
def num_bricks : ℕ := 5600

/-- Theorem stating that the brick length is correct given the wall and brick dimensions -/
theorem brick_length_is_correct :
  brick_length * brick_width * brick_height * num_bricks = wall_length * wall_height * wall_width := by
  sorry

end brick_length_is_correct_l504_50442


namespace moving_circle_trajectory_l504_50462

/-- The trajectory equation of the center of a moving circle that is tangent to the x-axis
    and internally tangent to the semicircle x^2 + y^2 = 4 (0 ≤ y ≤ 2) -/
theorem moving_circle_trajectory (x y : ℝ) : 
  (0 < y ∧ y ≤ 1) →
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x' y' : ℝ), x'^2 + y'^2 = 4 ∧ 0 ≤ y' ∧ y' ≤ 2 → 
      (x - x')^2 + (y - y')^2 = (2 - r)^2) ∧
    y = r) →
  x^2 = -4*(y - 1) := by
  sorry

end moving_circle_trajectory_l504_50462


namespace lake_distance_proof_l504_50425

def lake_distance : Set ℝ := {d | d > 9 ∧ d < 10}

theorem lake_distance_proof (d : ℝ) :
  (¬ (d ≥ 10)) ∧ (¬ (d ≤ 9)) ∧ (d ≠ 7) ↔ d ∈ lake_distance := by
  sorry

end lake_distance_proof_l504_50425


namespace xiao_ming_score_l504_50454

/-- Calculates the weighted score for a given component -/
def weightedScore (score : ℝ) (weight : ℝ) : ℝ := score * weight

/-- Calculates the total score based on individual scores and weights -/
def totalScore (regularScore midtermScore finalScore : ℝ) 
               (regularWeight midtermWeight finalWeight : ℝ) : ℝ :=
  weightedScore regularScore regularWeight + 
  weightedScore midtermScore midtermWeight + 
  weightedScore finalScore finalWeight

theorem xiao_ming_score : 
  let regularScore : ℝ := 70
  let midtermScore : ℝ := 80
  let finalScore : ℝ := 85
  let totalWeight : ℝ := 3 + 3 + 4
  let regularWeight : ℝ := 3 / totalWeight
  let midtermWeight : ℝ := 3 / totalWeight
  let finalWeight : ℝ := 4 / totalWeight
  totalScore regularScore midtermScore finalScore 
             regularWeight midtermWeight finalWeight = 79 := by
  sorry

end xiao_ming_score_l504_50454


namespace linear_function_not_in_first_quadrant_l504_50435

/-- A linear function y = mx + b, where m is the slope and b is the y-intercept -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The first quadrant of the Cartesian plane -/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 > 0}

/-- Theorem: The linear function y = -2x - 3 does not pass through the first quadrant -/
theorem linear_function_not_in_first_quadrant :
  let f : LinearFunction := ⟨-2, -3⟩
  ∀ x y : ℝ, y = f.m * x + f.b → (x, y) ∉ FirstQuadrant :=
by
  sorry


end linear_function_not_in_first_quadrant_l504_50435
