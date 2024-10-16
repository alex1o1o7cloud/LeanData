import Mathlib

namespace NUMINAMATH_CALUDE_train_crossing_time_l236_23625

/-- Proves that a train 75 meters long, traveling at 54 km/hr, will take 5 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 75 →
  train_speed_kmh = 54 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 5 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l236_23625


namespace NUMINAMATH_CALUDE_f_2019_eq_zero_l236_23674

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_3 (f : ℝ → ℝ) : Prop := ∀ x, f (3 - x) = f x

theorem f_2019_eq_zero 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period_3 f) : 
  f 2019 = 0 := by sorry

end NUMINAMATH_CALUDE_f_2019_eq_zero_l236_23674


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l236_23697

theorem trig_expression_simplification (α : ℝ) :
  (Real.tan (2 * Real.pi + α)) / (Real.tan (α + Real.pi) - Real.cos (-α) + Real.sin (Real.pi / 2 - α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l236_23697


namespace NUMINAMATH_CALUDE_student_marks_l236_23631

/-- Calculate the total marks secured in an exam given the following conditions:
  * total_questions: The total number of questions in the exam
  * correct_answers: The number of questions answered correctly
  * marks_per_correct: The number of marks awarded for each correct answer
  * marks_lost_per_wrong: The number of marks lost for each wrong answer
-/
def calculate_total_marks (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : ℤ :=
  (correct_answers * marks_per_correct : ℤ) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong)

/-- Theorem stating that under the given conditions, the student secures 160 marks -/
theorem student_marks : 
  calculate_total_marks 60 44 4 1 = 160 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_l236_23631


namespace NUMINAMATH_CALUDE_jessica_quarters_l236_23628

/-- Calculates the number of quarters Jessica has after her sister borrows some. -/
def quarters_remaining (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Jessica has 5 quarters remaining. -/
theorem jessica_quarters : quarters_remaining 8 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_quarters_l236_23628


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l236_23679

/-- The decimal representation of the repeating decimal 0.565656... -/
def repeating_decimal : ℚ := 56 / 99

/-- The reciprocal of the repeating decimal 0.565656... -/
def reciprocal : ℚ := 99 / 56

/-- Theorem: The reciprocal of the common fraction form of 0.565656... is 99/56 -/
theorem reciprocal_of_repeating_decimal : (repeating_decimal)⁻¹ = reciprocal := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l236_23679


namespace NUMINAMATH_CALUDE_product_mod_eleven_l236_23607

theorem product_mod_eleven : (103 * 107) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_eleven_l236_23607


namespace NUMINAMATH_CALUDE_horner_method_proof_l236_23618

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^7 + 2x^5 + 4x^3 + x -/
def f_coeffs : List ℤ := [3, 0, 2, 0, 4, 0, 1, 0]

theorem horner_method_proof :
  horner f_coeffs 3 = 7158 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l236_23618


namespace NUMINAMATH_CALUDE_girls_combined_avg_is_76_l236_23623

-- Define the schools
inductive School
| Cedar
| Dale

-- Define the student types
inductive StudentType
| Boy
| Girl

-- Define the average score function
def avg_score (s : School) (st : StudentType) : ℝ :=
  match s, st with
  | School.Cedar, StudentType.Boy => 65
  | School.Cedar, StudentType.Girl => 70
  | School.Dale, StudentType.Boy => 75
  | School.Dale, StudentType.Girl => 82

-- Define the combined average score function for each school
def combined_avg_score (s : School) : ℝ :=
  match s with
  | School.Cedar => 68
  | School.Dale => 78

-- Define the combined average score for boys at both schools
def combined_boys_avg : ℝ := 73

-- Theorem to prove
theorem girls_combined_avg_is_76 :
  ∃ (c d : ℝ), c > 0 ∧ d > 0 ∧
  (c * avg_score School.Cedar StudentType.Boy + d * avg_score School.Dale StudentType.Boy) / (c + d) = combined_boys_avg ∧
  (c * combined_avg_score School.Cedar + d * combined_avg_score School.Dale) / (c + d) = (c * avg_score School.Cedar StudentType.Girl + d * avg_score School.Dale StudentType.Girl) / (c + d) ∧
  (avg_score School.Cedar StudentType.Girl + avg_score School.Dale StudentType.Girl) / 2 = 76 :=
sorry

end NUMINAMATH_CALUDE_girls_combined_avg_is_76_l236_23623


namespace NUMINAMATH_CALUDE_max_xy_value_l236_23666

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x/3 + y/4 = 1) :
  ∃ (M : ℝ), M = 3 ∧ xy ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀/3 + y₀/4 = 1 ∧ x₀*y₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l236_23666


namespace NUMINAMATH_CALUDE_angle_measure_l236_23675

theorem angle_measure (A : ℝ) : 
  (90 - A = (180 - A) / 3 - 10) → A = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l236_23675


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l236_23616

theorem rectangular_prism_width (l h d : ℝ) (hl : l = 5) (hh : h = 8) (hd : d = 17) :
  ∃ w : ℝ, w > 0 ∧ w^2 + l^2 + h^2 = d^2 ∧ w = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l236_23616


namespace NUMINAMATH_CALUDE_appliance_pricing_l236_23611

/-- Represents the cost price of an electrical appliance in yuan -/
def cost_price : ℝ := sorry

/-- The markup percentage as a decimal -/
def markup : ℝ := 0.30

/-- The discount percentage as a decimal -/
def discount : ℝ := 0.20

/-- The final selling price in yuan -/
def selling_price : ℝ := 2080

theorem appliance_pricing :
  cost_price * (1 + markup) * (1 - discount) = selling_price := by sorry

end NUMINAMATH_CALUDE_appliance_pricing_l236_23611


namespace NUMINAMATH_CALUDE_cyclist_heartbeats_l236_23657

/-- Calculates the total number of heartbeats during a cycling race -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that the total number of heartbeats is 24000 for the given conditions -/
theorem cyclist_heartbeats :
  let heart_rate : ℕ := 120  -- beats per minute
  let race_distance : ℕ := 50  -- miles
  let pace : ℕ := 4  -- minutes per mile
  total_heartbeats heart_rate race_distance pace = 24000 := by
sorry


end NUMINAMATH_CALUDE_cyclist_heartbeats_l236_23657


namespace NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l236_23606

theorem no_bounded_function_satisfying_inequality :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, ∃ M : ℝ, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y : ℝ, f (x + y)^2 ≥ f x^2 + 2 * f (x * y) + f y^2) :=
by sorry

end NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l236_23606


namespace NUMINAMATH_CALUDE_campaign_fliers_l236_23619

theorem campaign_fliers (initial_fliers : ℕ) : 
  (initial_fliers : ℚ) * (4/5) * (3/4) = 600 → initial_fliers = 1000 := by
  sorry

end NUMINAMATH_CALUDE_campaign_fliers_l236_23619


namespace NUMINAMATH_CALUDE_hockey_league_games_l236_23663

theorem hockey_league_games (n : ℕ) (total_games : ℕ) 
  (hn : n = 17) (htotal : total_games = 1360) : 
  (total_games * 2) / (n * (n - 1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l236_23663


namespace NUMINAMATH_CALUDE_quadratic_radicals_sum_product_l236_23601

theorem quadratic_radicals_sum_product (a b c d e : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (he : 0 ≤ e)
  (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) (h4 : d = 9) (h5 : e = 11) :
  (Real.sqrt a - 1 + Real.sqrt b - Real.sqrt a + Real.sqrt c - Real.sqrt b + 
   Real.sqrt d - Real.sqrt c + Real.sqrt e - Real.sqrt d) * 
  (Real.sqrt e + 1) = 10 := by
  sorry

-- Additional lemmas to represent the given conditions
lemma quadratic_radical_diff (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  2 / (Real.sqrt a + Real.sqrt b) = Real.sqrt a - Real.sqrt b := by
  sorry

lemma quadratic_radical_sum (a : ℝ) (ha : 0 ≤ a) :
  Real.sqrt (a + 2 * Real.sqrt (a - 1)) = Real.sqrt a + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_sum_product_l236_23601


namespace NUMINAMATH_CALUDE_goats_bought_l236_23672

theorem goats_bought (total_cost : ℕ) (cow_price goat_price : ℕ) (num_cows : ℕ) :
  total_cost = 1400 →
  cow_price = 460 →
  goat_price = 60 →
  num_cows = 2 →
  ∃ (num_goats : ℕ), num_goats = 8 ∧ total_cost = num_cows * cow_price + num_goats * goat_price :=
by sorry

end NUMINAMATH_CALUDE_goats_bought_l236_23672


namespace NUMINAMATH_CALUDE_unique_three_digit_odd_l236_23646

/-- A function that returns true if a number is a three-digit odd number -/
def isThreeDigitOdd (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 2 = 1

/-- A function that returns the sum of squares of digits of a number -/
def sumOfSquaresOfDigits (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a * a + b * b + c * c

/-- The main theorem stating that 803 is the only three-digit odd number
    satisfying the given condition -/
theorem unique_three_digit_odd : ∀ n : ℕ, 
  isThreeDigitOdd n ∧ (n / 11 : ℚ) = (sumOfSquaresOfDigits n : ℚ) → n = 803 :=
by
  sorry

#check unique_three_digit_odd

end NUMINAMATH_CALUDE_unique_three_digit_odd_l236_23646


namespace NUMINAMATH_CALUDE_double_acute_angle_less_than_180_l236_23688

theorem double_acute_angle_less_than_180 (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  2 * α < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_double_acute_angle_less_than_180_l236_23688


namespace NUMINAMATH_CALUDE_cubic_function_properties_l236_23680

/-- A cubic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + b

/-- The derivative of f -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem cubic_function_properties (a b : ℝ) (h_a : a ≠ 0) :
  (f' a 2 = 0 ∧ f a b 2 = 8) →
  (a = 4 ∧ b = 24) ∧
  (∀ x, x < -2 → (f' a x > 0)) ∧
  (∀ x, x > 2 → (f' a x > 0)) ∧
  (∀ x, -2 < x ∧ x < 2 → (f' a x < 0)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - (-2)| ∧ |x - (-2)| < δ → f a b x < f a b (-2)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f a b x > f a b 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l236_23680


namespace NUMINAMATH_CALUDE_geometric_series_sum_l236_23604

/-- The sum of the infinite geometric series 4/3 - 1/2 + 3/32 - 9/256 + ... -/
theorem geometric_series_sum : 
  let a : ℚ := 4/3  -- first term
  let r : ℚ := -3/8 -- common ratio
  let S := Nat → ℚ  -- sequence type
  let series : S := fun n => a * r^n  -- geometric series
  ∑' n, series n = 32/33 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l236_23604


namespace NUMINAMATH_CALUDE_orthocenter_on_line_l236_23694

/-
  Define the necessary geometric objects and properties
-/

-- Define a Point type
structure Point := (x y : ℝ)

-- Define a Line type
structure Line := (a b c : ℝ)

-- Define a Circle type
structure Circle := (center : Point) (radius : ℝ)

-- Define a Triangle type
structure Triangle := (A B C : Point)

-- Function to check if a triangle is acute-angled
def is_acute_triangle (t : Triangle) : Prop := sorry

-- Function to get the circumcenter of a triangle
def circumcenter (t : Triangle) : Point := sorry

-- Function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Function to get the orthocenter of a triangle
def orthocenter (t : Triangle) : Point := sorry

-- Function to check if a circle passes through a point
def circle_passes_through (c : Circle) (p : Point) : Prop := sorry

-- Function to get the intersection points of a circle and a line segment
def circle_line_intersection (c : Circle) (l : Line) : List Point := sorry

-- Main theorem
theorem orthocenter_on_line 
  (A B C : Point) 
  (O : Point) 
  (c : Circle) 
  (P Q : Point) :
  is_acute_triangle (Triangle.mk A B C) →
  O = circumcenter (Triangle.mk A B C) →
  circle_passes_through c B →
  circle_passes_through c O →
  P ∈ circle_line_intersection c (Line.mk 0 1 0) → -- Assuming BC is on y-axis
  Q ∈ circle_line_intersection c (Line.mk 1 0 0) → -- Assuming BA is on x-axis
  point_on_line (orthocenter (Triangle.mk P O Q)) (Line.mk 1 1 0) -- Assuming AC is y = x
  := by sorry

end NUMINAMATH_CALUDE_orthocenter_on_line_l236_23694


namespace NUMINAMATH_CALUDE_two_digit_number_property_l236_23673

theorem two_digit_number_property (a b j m : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : 10 * a + b = j * (a^2 + b^2)) (h4 : 10 * b + a = m * (a^2 + b^2)) : m = j := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l236_23673


namespace NUMINAMATH_CALUDE_problem_statement_l236_23600

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : 
  (a + b)^2021 + a^2022 = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l236_23600


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l236_23665

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), r > 0 ∧ a₁ > 0 ∧ ∀ n, a n = a₁ * r ^ (n - 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 3 + a 2 * a 6 + 2 * a 3 ^ 2 = 36) →
  (a 2 + a 4 = 6) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l236_23665


namespace NUMINAMATH_CALUDE_polar_equation_pi_over_four_is_line_l236_23613

/-- The set of points (x, y) satisfying the polar equation θ = π/4 forms a line in the Cartesian plane. -/
theorem polar_equation_pi_over_four_is_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  {p : ℝ × ℝ | ∃ (r : ℝ), p.1 = r * Real.cos (π/4) ∧ p.2 = r * Real.sin (π/4)} =
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0} :=
sorry

end NUMINAMATH_CALUDE_polar_equation_pi_over_four_is_line_l236_23613


namespace NUMINAMATH_CALUDE_seating_arrangement_solution_l236_23649

/-- Represents a seating arrangement with rows of 7 or 9 seats. -/
structure SeatingArrangement where
  rows_of_nine : ℕ
  rows_of_seven : ℕ

/-- 
  Theorem: Given a seating arrangement where each row seats either 7 or 9 people, 
  and 61 people are to be seated with every seat occupied, 
  the number of rows seating exactly 9 people is 6.
-/
theorem seating_arrangement_solution : 
  ∃ (arrangement : SeatingArrangement),
    arrangement.rows_of_nine * 9 + arrangement.rows_of_seven * 7 = 61 ∧
    arrangement.rows_of_nine = 6 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_solution_l236_23649


namespace NUMINAMATH_CALUDE_city_mpg_calculation_l236_23689

/-- The average miles per gallon (mpg) for an SUV in the city -/
def city_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 25 gallons of gasoline -/
def max_distance : ℝ := 305

/-- The amount of gasoline in gallons used for the maximum distance -/
def gasoline_amount : ℝ := 25

/-- Theorem stating that the average mpg in the city is 12.2 -/
theorem city_mpg_calculation : city_mpg = max_distance / gasoline_amount := by
  sorry

end NUMINAMATH_CALUDE_city_mpg_calculation_l236_23689


namespace NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l236_23692

/-- The area of the largest equilateral triangle inscribed in a circle with radius 10 cm,
    where one side of the triangle is a diameter of the circle. -/
theorem largest_inscribed_equilateral_triangle_area :
  let r : ℝ := 10  -- radius of the circle in cm
  let d : ℝ := 2 * r  -- diameter of the circle in cm
  let h : ℝ := r * Real.sqrt 3  -- height of the equilateral triangle
  let area : ℝ := (1 / 2) * d * h  -- area of the triangle
  area = 100 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l236_23692


namespace NUMINAMATH_CALUDE_common_roots_product_l236_23608

theorem common_roots_product (C : ℝ) : 
  ∃ (u v : ℝ), 
    (u^3 - 5*u + 20 = 0) ∧ 
    (v^3 - 5*v + 20 = 0) ∧ 
    (u^3 + C*u^2 + 80 = 0) ∧ 
    (v^3 + C*v^2 + 80 = 0) ∧ 
    (u * v = 10 * (4^(1/3))) :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l236_23608


namespace NUMINAMATH_CALUDE_unique_line_divides_triangle_l236_23639

/-- A triangle in a 2D plane --/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- A line in the form y = mx --/
structure Line where
  m : ℝ

/-- Checks if a line divides a triangle into two equal areas --/
def dividesEqualArea (t : Triangle) (l : Line) : Prop :=
  sorry

/-- The specific triangle in the problem --/
def specificTriangle : Triangle :=
  { v1 := (0, 0),
    v2 := (4, 4),
    v3 := (12, 0) }

theorem unique_line_divides_triangle :
  ∃! m : ℝ, dividesEqualArea specificTriangle { m := m } ∧ m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_line_divides_triangle_l236_23639


namespace NUMINAMATH_CALUDE_function_identity_l236_23690

theorem function_identity (f : ℝ → ℝ) 
  (h_bounded : ∃ a b : ℝ, ∃ M : ℝ, ∀ x ∈ Set.Icc a b, |f x| ≤ M)
  (h_additive : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂)
  (h_one : f 1 = 1) :
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_identity_l236_23690


namespace NUMINAMATH_CALUDE_crossout_theorem_l236_23638

/-- The process of crossing out numbers and writing sums -/
def crossOutProcess (n : ℕ) : ℕ → ℕ
| 0 => n
| (m + 1) => let prev := crossOutProcess n m
             if prev > 4 then prev - 3 else prev

/-- The condition for n to be reduced to one number -/
def reducesToOne (n : ℕ) : Prop :=
  ∃ k, crossOutProcess n k = 1

/-- The sum of all numbers written during the process -/
def totalSum (n : ℕ) : ℕ :=
  sorry  -- Definition of totalSum would go here

/-- Main theorem combining both parts of the problem -/
theorem crossout_theorem :
  (∀ n : ℕ, reducesToOne n ↔ n % 3 = 1) ∧
  totalSum 2002 = 12881478 :=
sorry

end NUMINAMATH_CALUDE_crossout_theorem_l236_23638


namespace NUMINAMATH_CALUDE_binomial_60_3_l236_23643

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l236_23643


namespace NUMINAMATH_CALUDE_line_passes_through_point_l236_23637

/-- The line equation mx - y + 1 - m = 0 passes through the point (1,1) for all real m -/
theorem line_passes_through_point (m : ℝ) : m * 1 - 1 + 1 - m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l236_23637


namespace NUMINAMATH_CALUDE_number_ratio_problem_l236_23677

theorem number_ratio_problem (N : ℝ) (x : ℝ) 
  (h1 : N = 280) 
  (h2 : (1/5) * N + 4 = x * N - 10) : 
  x = 1/4 := by sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l236_23677


namespace NUMINAMATH_CALUDE_ladybugs_on_tuesday_l236_23650

theorem ladybugs_on_tuesday (monday_ladybugs : ℕ) (dots_per_ladybug : ℕ) (total_dots : ℕ) :
  monday_ladybugs = 8 →
  dots_per_ladybug = 6 →
  total_dots = 78 →
  ∃ tuesday_ladybugs : ℕ, 
    tuesday_ladybugs = 5 ∧
    total_dots = monday_ladybugs * dots_per_ladybug + tuesday_ladybugs * dots_per_ladybug :=
by sorry

end NUMINAMATH_CALUDE_ladybugs_on_tuesday_l236_23650


namespace NUMINAMATH_CALUDE_difference_numbers_between_500_and_600_l236_23603

def is_difference_number (n : ℕ) : Prop :=
  n % 7 = 6 ∧ n % 5 = 4

theorem difference_numbers_between_500_and_600 :
  {n : ℕ | 500 < n ∧ n < 600 ∧ is_difference_number n} = {524, 559, 594} := by
  sorry

end NUMINAMATH_CALUDE_difference_numbers_between_500_and_600_l236_23603


namespace NUMINAMATH_CALUDE_train_speed_l236_23686

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 357) 
  (h2 : bridge_length = 137) (h3 : time = 42.34285714285714) : 
  (train_length + bridge_length) / time = 11.66666666666667 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l236_23686


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l236_23617

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 4) (Fin 4) ℤ := !![3, 0, 2, 0;
                                       2, 3, -1, 4;
                                       0, 4, -2, 3;
                                       5, 2, 0, 1]
  Matrix.det A = -84 := by sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l236_23617


namespace NUMINAMATH_CALUDE_perpendicular_point_k_range_l236_23696

/-- Given points A(1,0) and B(3,0), if there exists a point P on the line y = kx + 1
    such that PA ⊥ PB, then -4/3 ≤ k ≤ 0. -/
theorem perpendicular_point_k_range (k : ℝ) :
  (∃ P : ℝ × ℝ, P.2 = k * P.1 + 1 ∧
    ((P.1 - 1) * (P.1 - 3) + P.2^2 = 0)) →
  -4/3 ≤ k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_point_k_range_l236_23696


namespace NUMINAMATH_CALUDE_basketballs_in_boxes_l236_23612

theorem basketballs_in_boxes 
  (total_basketballs : ℕ) 
  (basketballs_per_bag : ℕ) 
  (bags_per_box : ℕ) 
  (h1 : total_basketballs = 720) 
  (h2 : basketballs_per_bag = 8) 
  (h3 : bags_per_box = 6) : 
  (total_basketballs / (basketballs_per_bag * bags_per_box)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_basketballs_in_boxes_l236_23612


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l236_23685

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 30) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l236_23685


namespace NUMINAMATH_CALUDE_solution_difference_l236_23602

theorem solution_difference (r s : ℝ) : 
  (∀ x, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) →
  r ≠ s →
  r > s →
  r - s = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l236_23602


namespace NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l236_23653

def first_four_composite_numbers : List Nat := [4, 6, 8, 9]

theorem units_digit_of_product_first_four_composites :
  (first_four_composite_numbers.prod % 10) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l236_23653


namespace NUMINAMATH_CALUDE_symmetry_implies_m_equals_4_l236_23655

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Checks if two points are symmetric with respect to the y-axis -/
def symmetric_y_axis (a b : Point2D) : Prop :=
  a.x = -b.x ∧ a.y = b.y

theorem symmetry_implies_m_equals_4 (m : ℝ) :
  let a := Point2D.mk (-3) m
  let b := Point2D.mk 3 4
  symmetric_y_axis a b → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_m_equals_4_l236_23655


namespace NUMINAMATH_CALUDE_proposition_counterexample_l236_23654

theorem proposition_counterexample : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_proposition_counterexample_l236_23654


namespace NUMINAMATH_CALUDE_triangle_side_values_l236_23682

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_side_values :
  ∀ x : ℕ+, 
    (triangle_exists 5 (x.val ^ 2) 12) ↔ (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l236_23682


namespace NUMINAMATH_CALUDE_right_triangle_special_divisibility_l236_23652

theorem right_triangle_special_divisibility :
  ∃ (a b c : ℕ), 
    a^2 + b^2 = c^2 ∧  -- right-angled triangle
    (4 ∣ a ∨ 4 ∣ b) ∧  -- one leg is multiple of 4
    (3 ∣ a ∨ 3 ∣ b) ∧  -- one leg is multiple of 3
    (5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c) -- one side is multiple of 5
    := by sorry

end NUMINAMATH_CALUDE_right_triangle_special_divisibility_l236_23652


namespace NUMINAMATH_CALUDE_base_number_inequality_l236_23659

theorem base_number_inequality (x : ℝ) : 64^8 > x^22 ↔ x = 2^(24/11) := by
  sorry

end NUMINAMATH_CALUDE_base_number_inequality_l236_23659


namespace NUMINAMATH_CALUDE_polynomial_factorization_l236_23633

theorem polynomial_factorization (x : ℤ) :
  9 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 5 * x^2 =
  (3 * x^2 + 59 * x + 231) * (x + 7) * (3 * x + 11) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l236_23633


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l236_23699

theorem arithmetic_simplification : 4 * (8 - 3 + 2) / 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l236_23699


namespace NUMINAMATH_CALUDE_chocolate_percentage_proof_l236_23656

/-- Represents the number of each type of chocolate bar -/
def chocolate_count : ℕ := 25

/-- Represents the number of different types of chocolate bars -/
def chocolate_types : ℕ := 4

/-- Calculates the total number of chocolate bars -/
def total_chocolates : ℕ := chocolate_count * chocolate_types

/-- Represents the percentage as a rational number -/
def percentage_per_type : ℚ := chocolate_count / total_chocolates

theorem chocolate_percentage_proof :
  percentage_per_type = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_chocolate_percentage_proof_l236_23656


namespace NUMINAMATH_CALUDE_triangle_area_l236_23615

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 + c^2 = a^2 + b*c →
  b * c * Real.cos A = 4 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l236_23615


namespace NUMINAMATH_CALUDE_daisy_toys_theorem_l236_23667

/-- The number of dog toys Daisy's owner bought on Wednesday -/
def wednesday_toys (monday_toys tuesday_left tuesday_bought total_if_found : ℕ) : ℕ :=
  total_if_found - (tuesday_left + tuesday_bought)

theorem daisy_toys_theorem (monday_toys tuesday_left tuesday_bought total_if_found : ℕ) 
  (h1 : monday_toys = 5)
  (h2 : tuesday_left = 3)
  (h3 : tuesday_bought = 3)
  (h4 : total_if_found = 13) :
  wednesday_toys monday_toys tuesday_left tuesday_bought total_if_found = 7 := by
  sorry

#eval wednesday_toys 5 3 3 13

end NUMINAMATH_CALUDE_daisy_toys_theorem_l236_23667


namespace NUMINAMATH_CALUDE_cube_root_minus_square_root_plus_abs_l236_23647

theorem cube_root_minus_square_root_plus_abs : 
  ((-8 : ℝ) ^ (1/3 : ℝ)) - Real.sqrt ((-3 : ℝ)^2) + |Real.sqrt 2 - 1| = Real.sqrt 2 - 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_minus_square_root_plus_abs_l236_23647


namespace NUMINAMATH_CALUDE_inequalities_solution_l236_23658

theorem inequalities_solution (x : ℝ) : 
  (((2*x - 4)*(x - 5) < 0) ↔ (x > 2 ∧ x < 5)) ∧
  ((3*x^2 + 5*x + 1 > 0) ↔ (x < (-5 - Real.sqrt 13) / 6 ∨ x > (-5 + Real.sqrt 13) / 6)) ∧
  (∀ x, -x^2 + x < 2) ∧
  (¬∃ x, 7*x^2 + 5*x + 1 ≤ 0) ∧
  ((4*x ≥ 4*x^2 + 1) ↔ (x = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_solution_l236_23658


namespace NUMINAMATH_CALUDE_mrs_hilt_money_left_l236_23684

theorem mrs_hilt_money_left (initial_amount : ℕ) (pencil_cost : ℕ) (h1 : initial_amount = 15) (h2 : pencil_cost = 11) :
  initial_amount - pencil_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_mrs_hilt_money_left_l236_23684


namespace NUMINAMATH_CALUDE_monthly_salary_is_7600_l236_23636

/-- Represents the monthly salary allocation problem --/
def SalaryAllocation (x : ℝ) : Prop :=
  let bank := x / 2
  let remaining := x / 2
  let mortgage := remaining / 2 - 300
  let meals := (remaining - mortgage) / 2 + 300
  let leftover := remaining - mortgage - meals
  (bank = x / 2) ∧
  (mortgage = remaining / 2 - 300) ∧
  (meals = (remaining - mortgage) / 2 + 300) ∧
  (leftover = 800)

/-- Theorem stating that the monthly salary satisfying the given conditions is 7600 --/
theorem monthly_salary_is_7600 :
  ∃ x : ℝ, SalaryAllocation x ∧ x = 7600 :=
sorry

end NUMINAMATH_CALUDE_monthly_salary_is_7600_l236_23636


namespace NUMINAMATH_CALUDE_factorization_equality_l236_23620

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l236_23620


namespace NUMINAMATH_CALUDE_inequality_proof_l236_23687

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x + 5*y)) + (y / (y + 5*x)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l236_23687


namespace NUMINAMATH_CALUDE_shorts_savings_l236_23698

/-- Calculates the savings when buying shorts with a discount compared to buying individually -/
def savings (price : ℝ) (quantity : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_cost := price * quantity
  let discounted_cost := total_cost * (1 - discount_rate)
  total_cost - discounted_cost

/-- Proves that the savings when buying 3 pairs of shorts at $10 each with a 10% discount is $3 -/
theorem shorts_savings : savings 10 3 0.1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_shorts_savings_l236_23698


namespace NUMINAMATH_CALUDE_cube_properties_l236_23622

/-- Given a cube with surface area 864 square units, prove its volume and diagonal length -/
theorem cube_properties (s : ℝ) (h : 6 * s^2 = 864) : 
  s^3 = 1728 ∧ s * Real.sqrt 3 = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_properties_l236_23622


namespace NUMINAMATH_CALUDE_smallest_sum_five_consecutive_odd_integers_l236_23621

theorem smallest_sum_five_consecutive_odd_integers : 
  ∀ n : ℕ, n ≥ 35 → 
  ∃ k : ℤ, (k % 2 ≠ 0) ∧ 
  (n = k + (k + 2) + (k + 4) + (k + 6) + (k + 8)) ∧
  (∀ m : ℕ, m < 35 → 
    ¬∃ j : ℤ, (j % 2 ≠ 0) ∧ 
    (m = j + (j + 2) + (j + 4) + (j + 6) + (j + 8))) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_five_consecutive_odd_integers_l236_23621


namespace NUMINAMATH_CALUDE_sphere_only_identical_views_l236_23614

-- Define the set of common 3D solids
inductive Solid
  | Sphere
  | Cylinder
  | TriangularPrism
  | Cone

-- Define a function to check if all views are identical
def allViewsIdentical (s : Solid) : Prop :=
  match s with
  | Solid.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_only_identical_views :
  ∀ s : Solid, allViewsIdentical s ↔ s = Solid.Sphere :=
sorry

end NUMINAMATH_CALUDE_sphere_only_identical_views_l236_23614


namespace NUMINAMATH_CALUDE_divisibility_in_sequence_l236_23662

theorem divisibility_in_sequence (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ (i j : Fin (n + 1)), i ≠ j ∧ (n : ℤ) ∣ (a i - a j) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_in_sequence_l236_23662


namespace NUMINAMATH_CALUDE_boundary_length_special_square_l236_23676

/-- The length of the boundary of a special figure constructed from a square --/
theorem boundary_length_special_square : 
  ∀ (s : Real) (a : Real),
    s * s = 64 →  -- area of the square is 64
    a = s / 4 →   -- length of each arc segment
    (16 : Real) + 14 * Real.pi = 
      4 * s +     -- sum of straight segments
      12 * (a * Real.pi / 2) +  -- sum of side arcs
      4 * (a * Real.pi / 2)     -- sum of corner arcs
    := by sorry

end NUMINAMATH_CALUDE_boundary_length_special_square_l236_23676


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l236_23630

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 4 then a * x - 8 else x^2 - 2 * a * x

-- Define what it means for f to be increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_increasing_iff_a_in_range (a : ℝ) :
  is_increasing (f a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l236_23630


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l236_23661

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | (x + 2)*(4 - x) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a + 1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

-- Theorem for the range of a when B ∪ C = B
theorem range_of_a (a : ℝ) (h : B ∪ C a = B) : -2 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l236_23661


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l236_23626

theorem min_value_of_function (x : ℝ) (h : x > 1) : 1 / (x - 1) + x ≥ 3 := by
  sorry

theorem equality_condition (x : ℝ) (h : x > 1) : 1 / (x - 1) + x = 3 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l236_23626


namespace NUMINAMATH_CALUDE_apples_eaten_proof_l236_23669

-- Define the daily apple consumption for each person
def simone_daily : ℚ := 1/2
def lauri_daily : ℚ := 1/3
def alex_daily : ℚ := 1/4

-- Define the number of days each person ate apples
def simone_days : ℕ := 16
def lauri_days : ℕ := 15
def alex_days : ℕ := 20

-- Define the total number of apples eaten by all three
def total_apples : ℚ := simone_daily * simone_days + lauri_daily * lauri_days + alex_daily * alex_days

-- Theorem statement
theorem apples_eaten_proof : total_apples = 18 := by
  sorry

end NUMINAMATH_CALUDE_apples_eaten_proof_l236_23669


namespace NUMINAMATH_CALUDE_exp_sum_gt_two_l236_23644

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.exp (a * x) - a * (x + 2)

theorem exp_sum_gt_two (ha : a ≠ 0) (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) (hf₂ : f a x₂ = 0) : 
  Real.exp (a * x₁) + Real.exp (a * x₂) > 2 :=
by sorry

end

end NUMINAMATH_CALUDE_exp_sum_gt_two_l236_23644


namespace NUMINAMATH_CALUDE_common_chord_length_l236_23624

theorem common_chord_length (r₁ r₂ d : ℝ) (h₁ : r₁ = 8) (h₂ : r₂ = 12) (h₃ : d = 20) :
  let chord_length := 2 * Real.sqrt (r₂^2 - (d/2)^2)
  chord_length = 4 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_l236_23624


namespace NUMINAMATH_CALUDE_jerry_money_left_l236_23610

/-- Calculates the amount of money Jerry has left after grocery shopping --/
def money_left (budget : ℚ) (mustard_oil_price : ℚ) (mustard_oil_quantity : ℚ) 
  (mustard_oil_discount : ℚ) (pasta_price : ℚ) (pasta_quantity : ℚ) 
  (pasta_sauce_price : ℚ) (pasta_sauce_quantity : ℚ) : ℚ :=
  let mustard_oil_cost := mustard_oil_price * mustard_oil_quantity * (1 - mustard_oil_discount)
  let pasta_cost := pasta_price * (pasta_quantity - 1)  -- Buy 2, Get the 3rd one free
  let pasta_sauce_cost := pasta_sauce_price * pasta_sauce_quantity
  budget - (mustard_oil_cost + pasta_cost + pasta_sauce_cost)

theorem jerry_money_left :
  money_left 100 13 2 0.1 4 3 5 1 = 63.6 := by
  sorry

end NUMINAMATH_CALUDE_jerry_money_left_l236_23610


namespace NUMINAMATH_CALUDE_find_c_l236_23660

theorem find_c : ∃ c : ℝ, 
  (∃ n : ℤ, Int.floor c = n ∧ 3 * (n : ℝ)^2 + 12 * (n : ℝ) - 27 = 0) ∧ 
  (let frac := c - Int.floor c
   4 * frac^2 - 12 * frac + 5 = 0) ∧
  (0 ≤ c - Int.floor c ∧ c - Int.floor c < 1) ∧
  c = -8.5 := by
  sorry

end NUMINAMATH_CALUDE_find_c_l236_23660


namespace NUMINAMATH_CALUDE_chichikov_game_l236_23681

theorem chichikov_game (total_nuts : ℕ) (box1 box2 : ℕ) : total_nuts = 222 → box1 + box2 = total_nuts →
  ∃ N : ℕ, 1 ≤ N ∧ N ≤ 222 ∧
  (∀ move : ℕ, move < 37 →
    ¬(∃ new_box1 new_box2 new_box3 : ℕ,
      new_box1 + new_box2 + new_box3 = total_nuts ∧
      (new_box1 = N ∨ new_box2 = N ∨ new_box3 = N ∨ new_box1 + new_box2 = N ∨ new_box1 + new_box3 = N ∨ new_box2 + new_box3 = N) ∧
      new_box1 + new_box2 + move = box1 + box2)) ∧
  (∀ N : ℕ, 1 ≤ N ∧ N ≤ 222 →
    ∃ new_box1 new_box2 new_box3 : ℕ,
      new_box1 + new_box2 + new_box3 = total_nuts ∧
      (new_box1 = N ∨ new_box2 = N ∨ new_box3 = N ∨ new_box1 + new_box2 = N ∨ new_box1 + new_box3 = N ∨ new_box2 + new_box3 = N) ∧
      new_box1 + new_box2 + 37 ≥ box1 + box2) :=
by
  sorry

end NUMINAMATH_CALUDE_chichikov_game_l236_23681


namespace NUMINAMATH_CALUDE_problem_statement_l236_23664

theorem problem_statement :
  (¬(∀ x : ℝ, x > 0 → Real.log x ≥ 0)) ∧ (∃ x₀ : ℝ, Real.sin x₀ = Real.cos x₀) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l236_23664


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l236_23695

theorem geometric_sequence_third_term (a : ℕ → ℝ) (q : ℝ) (S₄ : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence condition
  q = 2 →                       -- Common ratio
  S₄ = 60 →                     -- Sum of first 4 terms
  (a 0 * (1 - q^4)) / (1 - q) = S₄ →  -- Sum formula for geometric sequence
  a 2 = 16 := by               -- Third term (index 2 in 0-based indexing)
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l236_23695


namespace NUMINAMATH_CALUDE_factorial_divisibility_l236_23691

theorem factorial_divisibility (k n : ℕ) (hk : 0 < k ∧ k ≤ 2020) (hn : 0 < n) :
  ¬ (3^((k-1)*n+1) ∣ ((Nat.factorial (k*n) / Nat.factorial n)^2)) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l236_23691


namespace NUMINAMATH_CALUDE_point_symmetry_l236_23627

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the x-axis -/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- Symmetry about the y-axis -/
def symmetricAboutYAxis (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = q.y

/-- The main theorem -/
theorem point_symmetry (M N P : Point) :
  symmetricAboutXAxis M P →
  symmetricAboutYAxis N M →
  N = Point.mk 1 2 →
  P = Point.mk (-1) (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l236_23627


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_squares_max_value_achieved_l236_23640

theorem max_value_of_sum_of_squares (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y ≥ x^3 + y^2) : 
  x^2 + y^2 ≤ 2 := by
  sorry

-- The maximum value is indeed achieved
theorem max_value_achieved : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y ≥ x^3 + y^2 ∧ x^2 + y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_squares_max_value_achieved_l236_23640


namespace NUMINAMATH_CALUDE_power_of_three_difference_l236_23605

theorem power_of_three_difference : 3^(2+3+4) - (3^2 + 3^3 + 3^4) = 19566 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_difference_l236_23605


namespace NUMINAMATH_CALUDE_inequality_proof_l236_23635

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : 1/m < 1/n) : m < 0 ∧ 0 < n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l236_23635


namespace NUMINAMATH_CALUDE_integer_triplet_solution_l236_23629

theorem integer_triplet_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 - 2*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_triplet_solution_l236_23629


namespace NUMINAMATH_CALUDE_smith_children_age_l236_23671

theorem smith_children_age (age1 age2 age3 age4 : ℕ) 
  (h1 : age1 = 5)
  (h2 : age2 = 7)
  (h3 : age3 = 10)
  (h_avg : (age1 + age2 + age3 + age4) / 4 = 8) :
  age4 = 10 := by
sorry

end NUMINAMATH_CALUDE_smith_children_age_l236_23671


namespace NUMINAMATH_CALUDE_birthday_party_ratio_l236_23683

theorem birthday_party_ratio (total_guests : ℕ) (men : ℕ) (stayed : ℕ) : 
  total_guests = 60 →
  men = 15 →
  stayed = 50 →
  (total_guests / 2 : ℕ) + men + (total_guests - (total_guests / 2 + men)) = total_guests →
  (total_guests - stayed - 5 : ℕ) / men = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_birthday_party_ratio_l236_23683


namespace NUMINAMATH_CALUDE_intersection_M_N_l236_23693

noncomputable def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (p.1 - 1)}

noncomputable def N : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1}

theorem intersection_M_N :
  ∃! a : ℝ, a > 1 ∧ Real.sqrt (a - 1) = Real.log a ∧
  M ∩ N = {(a, Real.log a)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l236_23693


namespace NUMINAMATH_CALUDE_skating_speed_ratio_l236_23651

theorem skating_speed_ratio (v_f v_s : ℝ) (h1 : v_f > v_s) (h2 : v_f > 0) (h3 : v_s > 0) :
  (v_f + v_s) / (v_f - v_s) = 5 → v_f / v_s = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_skating_speed_ratio_l236_23651


namespace NUMINAMATH_CALUDE_shortest_perpendicular_best_measurement_l236_23634

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a long jump measurement method -/
inductive LongJumpMeasurement
  | Vertical
  | ShortestLineSegment
  | TwoPointLine
  | ShortestPerpendicular

/-- Defines the accuracy of a measurement method -/
def isAccurate (method : LongJumpMeasurement) : Prop :=
  match method with
  | LongJumpMeasurement.ShortestPerpendicular => true
  | _ => false

/-- Defines the consistency of a measurement method -/
def isConsistent (method : LongJumpMeasurement) : Prop :=
  match method with
  | LongJumpMeasurement.ShortestPerpendicular => true
  | _ => false

/-- Theorem: The shortest perpendicular line segment is the most accurate and consistent method for measuring long jump performance -/
theorem shortest_perpendicular_best_measurement :
  ∀ (method : LongJumpMeasurement),
    isAccurate method ∧ isConsistent method ↔ method = LongJumpMeasurement.ShortestPerpendicular :=
by sorry

end NUMINAMATH_CALUDE_shortest_perpendicular_best_measurement_l236_23634


namespace NUMINAMATH_CALUDE_second_movie_length_second_movie_is_one_and_half_hours_l236_23632

/-- Calculates the length of the second movie given Henri's schedule --/
theorem second_movie_length 
  (total_time : ℝ) 
  (first_movie : ℝ) 
  (reading_rate : ℝ) 
  (words_read : ℝ) : ℝ :=
  let reading_time : ℝ := words_read / (reading_rate * 60)
  let second_movie : ℝ := total_time - first_movie - reading_time
  second_movie

/-- Proves that the length of the second movie is 1.5 hours --/
theorem second_movie_is_one_and_half_hours :
  second_movie_length 8 3.5 10 1800 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_second_movie_length_second_movie_is_one_and_half_hours_l236_23632


namespace NUMINAMATH_CALUDE_f_equals_x_l236_23648

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - b*x + c

-- State the theorem
theorem f_equals_x (a b c : ℝ) :
  (∀ x, f a b c x + f a b c (-x) = 0) →  -- f is odd
  (∀ x ≥ 1, ∀ y ≥ 1, x < y → f a b c x < f a b c y) →  -- f is strictly increasing on [1, +∞)
  (a = 0 ∧ c = 0 ∧ b ≤ 3) →  -- conditions on a, b, c
  ∀ x ≥ 1, f a b c x ≥ 1 →  -- f(x) ≥ 1 for x ≥ 1
  (∀ x ≥ 1, f a b c (f a b c x) = x) →  -- f(f(x)) = x for x ≥ 1
  ∀ x ≥ 1, f a b c x = x :=  -- conclusion: f(x) = x for x ≥ 1
by sorry


end NUMINAMATH_CALUDE_f_equals_x_l236_23648


namespace NUMINAMATH_CALUDE_only_one_equals_sum_of_squares_of_digits_l236_23670

/-- Sum of squares of digits of a natural number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The only positive integer n such that s(n) = n is 1 -/
theorem only_one_equals_sum_of_squares_of_digits :
  ∀ n : ℕ, n > 0 → (sum_of_squares_of_digits n = n ↔ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_only_one_equals_sum_of_squares_of_digits_l236_23670


namespace NUMINAMATH_CALUDE_balls_in_boxes_l236_23678

theorem balls_in_boxes (x y z : ℕ) : 
  x + y + z = 320 →
  x > 0 ∧ y > 0 ∧ z > 0 →
  ∃ (a b c : ℕ), a ≤ x ∧ b ≤ y ∧ c ≤ z ∧ 6*a + 11*b + 15*c = 1001 :=
by sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l236_23678


namespace NUMINAMATH_CALUDE_largest_square_area_l236_23668

theorem largest_square_area (side_length : ℝ) (corner_size : ℝ) : 
  side_length = 5 → 
  corner_size = 1 → 
  (side_length - 2 * corner_size)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l236_23668


namespace NUMINAMATH_CALUDE_shorter_lateral_side_length_l236_23641

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- One angle of the trapezoid -/
  angle : ℝ
  /-- The midline (median) of the trapezoid -/
  midline : ℝ
  /-- One of the bases of the trapezoid -/
  base : ℝ
  /-- The angle is 30 degrees -/
  angle_is_30 : angle = 30
  /-- The lines containing the lateral sides intersect at a right angle -/
  lateral_sides_right_angle : True
  /-- The midline is 10 -/
  midline_is_10 : midline = 10
  /-- One base is 8 -/
  base_is_8 : base = 8

/-- The theorem stating the length of the shorter lateral side -/
theorem shorter_lateral_side_length (t : SpecialTrapezoid) : 
  ∃ (shorter_side : ℝ), shorter_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_shorter_lateral_side_length_l236_23641


namespace NUMINAMATH_CALUDE_bookstore_profit_rate_l236_23642

/-- Calculates the overall rate of profit for three books given their cost and selling prices -/
theorem bookstore_profit_rate 
  (cost_A selling_A cost_B selling_B cost_C selling_C : ℚ) 
  (h1 : cost_A = 50) (h2 : selling_A = 70)
  (h3 : cost_B = 80) (h4 : selling_B = 100)
  (h5 : cost_C = 150) (h6 : selling_C = 180) :
  (selling_A - cost_A + selling_B - cost_B + selling_C - cost_C) / 
  (cost_A + cost_B + cost_C) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_profit_rate_l236_23642


namespace NUMINAMATH_CALUDE_unanswered_test_completion_ways_l236_23609

/-- Represents a multiple-choice test -/
structure MultipleChoiceTest where
  num_questions : ℕ
  choices_per_question : ℕ

/-- Calculates the number of ways to complete the test with all questions unanswered -/
def ways_to_leave_unanswered (test : MultipleChoiceTest) : ℕ := 1

/-- Theorem: For a test with 4 questions and 5 choices per question,
    there is only one way to leave all questions unanswered -/
theorem unanswered_test_completion_ways
  (test : MultipleChoiceTest)
  (h1 : test.num_questions = 4)
  (h2 : test.choices_per_question = 5) :
  ways_to_leave_unanswered test = 1 := by
  sorry


end NUMINAMATH_CALUDE_unanswered_test_completion_ways_l236_23609


namespace NUMINAMATH_CALUDE_number_plus_five_equals_six_l236_23645

theorem number_plus_five_equals_six : ∃ x : ℝ, x + 5 = 6 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_five_equals_six_l236_23645
