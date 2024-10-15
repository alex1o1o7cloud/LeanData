import Mathlib

namespace NUMINAMATH_CALUDE_slope_theorem_l1843_184338

/-- Given two points A(-3,5) and B(x,2) in a coordinate plane, 
    if the slope of the line through A and B is -1/4, then x = 9. -/
theorem slope_theorem (x : ℝ) : 
  let A : ℝ × ℝ := (-3, 5)
  let B : ℝ × ℝ := (x, 2)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -1/4 → x = 9 := by
sorry

end NUMINAMATH_CALUDE_slope_theorem_l1843_184338


namespace NUMINAMATH_CALUDE_license_plate_combinations_l1843_184347

/-- The number of possible letter combinations in the license plate -/
def letter_combinations : ℕ := Nat.choose 26 2 * 3

/-- The number of possible digit combinations in the license plate -/
def digit_combinations : ℕ := 10 * 9 * 3

/-- The total number of possible license plate combinations -/
def total_combinations : ℕ := letter_combinations * digit_combinations

theorem license_plate_combinations :
  total_combinations = 877500 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l1843_184347


namespace NUMINAMATH_CALUDE_cosine_rationality_l1843_184331

theorem cosine_rationality (k : ℤ) (θ : ℝ) 
  (h1 : k ≥ 3)
  (h2 : ∃ q₁ : ℚ, (↑q₁ : ℝ) = Real.cos ((k - 1) * θ))
  (h3 : ∃ q₂ : ℚ, (↑q₂ : ℝ) = Real.cos (k * θ)) :
  ∃ (n : ℕ), n > k ∧ 
    (∃ q₃ : ℚ, (↑q₃ : ℝ) = Real.cos ((n - 1) * θ)) ∧ 
    (∃ q₄ : ℚ, (↑q₄ : ℝ) = Real.cos (n * θ)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_rationality_l1843_184331


namespace NUMINAMATH_CALUDE_maple_logs_solution_l1843_184394

/-- The number of logs each maple tree makes -/
def maple_logs : ℕ := 60

theorem maple_logs_solution : 
  ∃ (x : ℕ), x > 0 ∧ 8 * 80 + 3 * x + 4 * 100 = 1220 → x = maple_logs :=
by sorry

end NUMINAMATH_CALUDE_maple_logs_solution_l1843_184394


namespace NUMINAMATH_CALUDE_total_sleep_time_in_week_l1843_184315

/-- The number of hours a cougar sleeps per night -/
def cougar_sleep : ℕ := 4

/-- The additional hours a zebra sleeps compared to a cougar -/
def zebra_extra_sleep : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total sleep time for both a cougar and a zebra in one week is 70 hours -/
theorem total_sleep_time_in_week : 
  (cougar_sleep * days_in_week) + ((cougar_sleep + zebra_extra_sleep) * days_in_week) = 70 := by
  sorry


end NUMINAMATH_CALUDE_total_sleep_time_in_week_l1843_184315


namespace NUMINAMATH_CALUDE_combined_tax_rate_l1843_184300

/-- Combined tax rate calculation -/
theorem combined_tax_rate 
  (john_tax_rate : ℝ) 
  (ingrid_tax_rate : ℝ) 
  (john_income : ℝ) 
  (ingrid_income : ℝ) 
  (h1 : john_tax_rate = 0.3)
  (h2 : ingrid_tax_rate = 0.4)
  (h3 : john_income = 58000)
  (h4 : ingrid_income = 72000) :
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = 
  (0.3 * 58000 + 0.4 * 72000) / (58000 + 72000) := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l1843_184300


namespace NUMINAMATH_CALUDE_unique_base_l1843_184325

/-- Converts a number from base h to base 10 --/
def to_base_10 (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

/-- The equation in base h --/
def equation_holds (h : Nat) : Prop :=
  h > 9 ∧ 
  to_base_10 [8, 3, 2, 7] h + to_base_10 [9, 4, 6, 1] h = to_base_10 [1, 9, 2, 8, 8] h

theorem unique_base : ∃! h, equation_holds h :=
  sorry

end NUMINAMATH_CALUDE_unique_base_l1843_184325


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1843_184367

theorem fraction_equation_solution (x : ℚ) :
  (x + 10) / (x - 4) = (x + 3) / (x - 6) → x = 48 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1843_184367


namespace NUMINAMATH_CALUDE_expression_bounds_l1843_184307

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 + Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (1-b)^2 + 1) + Real.sqrt (b^2 + (1-c)^2 + 1) + 
    Real.sqrt (c^2 + (1-d)^2 + 1) + Real.sqrt (d^2 + (1-a)^2 + 1) ∧
  Real.sqrt (a^2 + (1-b)^2 + 1) + Real.sqrt (b^2 + (1-c)^2 + 1) + 
  Real.sqrt (c^2 + (1-d)^2 + 1) + Real.sqrt (d^2 + (1-a)^2 + 1) ≤ 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_expression_bounds_l1843_184307


namespace NUMINAMATH_CALUDE_triangle_longest_side_l1843_184371

theorem triangle_longest_side (x : ℚ) :
  9 + (x + 5) + (2 * x + 2) = 42 →
  max 9 (max (x + 5) (2 * x + 2)) = 58 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l1843_184371


namespace NUMINAMATH_CALUDE_high_school_students_l1843_184333

/-- The number of students in a high school, given information about music and art classes -/
theorem high_school_students (music_students : ℕ) (art_students : ℕ) (both_students : ℕ) (neither_students : ℕ)
  (h1 : music_students = 20)
  (h2 : art_students = 20)
  (h3 : both_students = 10)
  (h4 : neither_students = 470) :
  music_students + art_students - both_students + neither_students = 500 :=
by sorry

end NUMINAMATH_CALUDE_high_school_students_l1843_184333


namespace NUMINAMATH_CALUDE_square_difference_theorem_l1843_184365

theorem square_difference_theorem : ∃ (n m : ℕ),
  (∀ k : ℕ, k^2 < 2018 → k ≤ n) ∧
  (n^2 < 2018) ∧
  (∀ k : ℕ, 2018 < k^2 → m ≤ k) ∧
  (2018 < m^2) ∧
  (m^2 - n^2 = 89) := by
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l1843_184365


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1843_184385

/-- The perimeter of a rectangle with a long side of 1 meter and a short side
    that is 2/8 meter shorter than the long side is 3.5 meters. -/
theorem rectangle_perimeter : 
  let long_side : ℝ := 1
  let short_side : ℝ := long_side - 2/8
  let perimeter : ℝ := 2 * long_side + 2 * short_side
  perimeter = 3.5 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1843_184385


namespace NUMINAMATH_CALUDE_boat_current_rate_l1843_184377

/-- Proves that given a boat with a speed of 22 km/hr in still water,
    traveling 10.4 km downstream in 24 minutes, the rate of the current is 4 km/hr. -/
theorem boat_current_rate
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : boat_speed = 22)
  (h2 : downstream_distance = 10.4)
  (h3 : downstream_time = 24 / 60) :
  ∃ current_rate : ℝ,
    current_rate = 4 ∧
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_current_rate_l1843_184377


namespace NUMINAMATH_CALUDE_average_difference_implies_unknown_l1843_184395

theorem average_difference_implies_unknown (x : ℝ) : 
  let set1 := [20, 40, 60]
  let set2 := [10, x, 15]
  (set1.sum / set1.length) = (set2.sum / set2.length) + 5 →
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_average_difference_implies_unknown_l1843_184395


namespace NUMINAMATH_CALUDE_horner_rule_f_at_2_f_2_equals_62_l1843_184344

/-- Horner's Rule evaluation of a polynomial -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_f_at_2 :
  horner_eval [2, 3, 0, 5, -4] 2 = f 2 := by sorry

theorem f_2_equals_62 : f 2 = 62 := by sorry

end NUMINAMATH_CALUDE_horner_rule_f_at_2_f_2_equals_62_l1843_184344


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1843_184349

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 72 ∧ 
  (∀ (m : ℕ), m < n → ¬(127 ∣ (100203 - m))) ∧ 
  (127 ∣ (100203 - n)) := by
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1843_184349


namespace NUMINAMATH_CALUDE_cube_root_8000_simplification_l1843_184304

theorem cube_root_8000_simplification :
  ∃ (a b : ℕ+), (a : ℝ) * ((b : ℝ) ^ (1/3 : ℝ)) = (8000 : ℝ) ^ (1/3 : ℝ) ∧ 
  (∀ (c d : ℕ+), (c : ℝ) * ((d : ℝ) ^ (1/3 : ℝ)) = (8000 : ℝ) ^ (1/3 : ℝ) → d ≥ b) ∧
  a = 20 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_8000_simplification_l1843_184304


namespace NUMINAMATH_CALUDE_best_sampling_methods_l1843_184364

/-- Represents different income levels of families -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Structure representing a community -/
structure Community where
  total_families : ℕ
  high_income : ℕ
  middle_income : ℕ
  low_income : ℕ
  sample_size : ℕ

/-- Structure representing a group of student-athletes -/
structure StudentAthleteGroup where
  total_athletes : ℕ
  sample_size : ℕ

/-- Function to determine the best sampling method for a community survey -/
def best_community_sampling_method (c : Community) : SamplingMethod := sorry

/-- Function to determine the best sampling method for a student-athlete survey -/
def best_student_athlete_sampling_method (g : StudentAthleteGroup) : SamplingMethod := sorry

/-- Theorem stating the best sampling methods for the given scenarios -/
theorem best_sampling_methods 
  (community : Community) 
  (student_athletes : StudentAthleteGroup) : 
  community.total_families = 500 ∧ 
  community.high_income = 125 ∧ 
  community.middle_income = 280 ∧ 
  community.low_income = 95 ∧ 
  community.sample_size = 100 ∧
  student_athletes.total_athletes = 12 ∧ 
  student_athletes.sample_size = 3 →
  best_community_sampling_method community = SamplingMethod.Stratified ∧
  best_student_athlete_sampling_method student_athletes = SamplingMethod.SimpleRandom := by
  sorry

end NUMINAMATH_CALUDE_best_sampling_methods_l1843_184364


namespace NUMINAMATH_CALUDE_river_depth_l1843_184339

/-- Proves that given a river with specified width, flow rate, and volume of water flowing into the sea per minute, the depth of the river is 5 meters. -/
theorem river_depth 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (volume_per_minute : ℝ) 
  (h1 : width = 35) 
  (h2 : flow_rate_kmph = 2) 
  (h3 : volume_per_minute = 5833.333333333333) : 
  (volume_per_minute / (flow_rate_kmph * 1000 / 60 * width)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_river_depth_l1843_184339


namespace NUMINAMATH_CALUDE_faster_train_speed_l1843_184353

/-- Proves that the speed of the faster train is 50 km/hr given the conditions of the problem -/
theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 70)
  (h2 : slower_speed = 36)
  (h3 : passing_time = 36)
  : ∃ (faster_speed : ℝ), faster_speed = 50 ∧ 
    (faster_speed - slower_speed) * (1000 / 3600) * passing_time = 2 * train_length :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l1843_184353


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1843_184301

/-- An arithmetic sequence with first term a₁ > 0 and common ratio q -/
structure ArithmeticSequence where
  a₁ : ℝ
  q : ℝ
  h₁ : a₁ > 0

/-- Sum of first n terms of an arithmetic sequence -/
def S (as : ArithmeticSequence) (n : ℕ) : ℝ := sorry

/-- Statement: q > 1 is sufficient but not necessary for S₃ + S₅ > 2S₄ -/
theorem sufficient_not_necessary (as : ArithmeticSequence) :
  (∀ as, as.q > 1 → S as 3 + S as 5 > 2 * S as 4) ∧
  ¬(∀ as, S as 3 + S as 5 > 2 * S as 4 → as.q > 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1843_184301


namespace NUMINAMATH_CALUDE_smallest_k_multiple_of_200_l1843_184317

theorem smallest_k_multiple_of_200 : ∃ (k : ℕ), k > 0 ∧ 
  (∀ (n : ℕ), n > 0 → n < k → ¬(200 ∣ (n * (n + 1) * (2 * n + 1)) / 6)) ∧ 
  (200 ∣ (k * (k + 1) * (2 * k + 1)) / 6) ∧
  k = 31 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_multiple_of_200_l1843_184317


namespace NUMINAMATH_CALUDE_triangle_inradius_circumradius_inequality_l1843_184318

/-- For any triangle with inradius r, circumradius R, and an angle α, 
    the inequality r / R ≤ 2 sin(α / 2)(1 - sin(α / 2)) holds. -/
theorem triangle_inradius_circumradius_inequality 
  (r R α : ℝ) 
  (hr : r > 0) 
  (hR : R > 0) 
  (hα : 0 < α ∧ α < π) : 
  r / R ≤ 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) :=
sorry

end NUMINAMATH_CALUDE_triangle_inradius_circumradius_inequality_l1843_184318


namespace NUMINAMATH_CALUDE_kaiden_first_week_cans_l1843_184311

/-- The number of cans collected in the first week of Kaiden's soup can collection -/
def cans_first_week (goal : ℕ) (cans_second_week : ℕ) (cans_needed : ℕ) : ℕ :=
  goal - cans_needed - cans_second_week

/-- Theorem stating that Kaiden collected 158 cans in the first week -/
theorem kaiden_first_week_cans :
  cans_first_week 500 259 83 = 158 := by sorry

end NUMINAMATH_CALUDE_kaiden_first_week_cans_l1843_184311


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1843_184326

/-- Prove that the polar equation ρ = 4cosθ is equivalent to the Cartesian equation (x - 2)² + y² = 4 -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.cos θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1843_184326


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l1843_184360

/-- Calculates the local tax deduction in cents given an hourly wage in dollars and a tax rate as a percentage. -/
def localTaxDeduction (hourlyWage : ℚ) (taxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (taxRate / 100)

/-- Theorem stating that for an hourly wage of $25 and a tax rate of 2.2%, the local tax deduction is 55 cents. -/
theorem alicia_tax_deduction :
  localTaxDeduction 25 2.2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_alicia_tax_deduction_l1843_184360


namespace NUMINAMATH_CALUDE_drama_ticket_revenue_l1843_184369

theorem drama_ticket_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 160)
  (h_total_revenue : total_revenue = 2400) : ∃ (full_price : ℕ) (half_price : ℕ) (price : ℕ),
  full_price + half_price = total_tickets ∧
  full_price * price + half_price * (price / 2) = total_revenue ∧
  full_price * price = 1600 :=
sorry

end NUMINAMATH_CALUDE_drama_ticket_revenue_l1843_184369


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1843_184328

theorem simplify_and_evaluate :
  (∀ x y : ℚ, x = -2 ∧ y = -3 → 6*x - 5*y + 3*y - 2*x = -2) ∧
  (∀ a : ℚ, a = -1/2 → 1/4*(-4*a^2 + 2*a - 8) - (1/2*a - 2) = -1/4) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1843_184328


namespace NUMINAMATH_CALUDE_circle_fit_theorem_l1843_184324

/-- Represents a square with unit side length -/
structure UnitSquare where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The main theorem statement -/
theorem circle_fit_theorem (rect : Rectangle) (squares : Finset UnitSquare) :
  rect.width = 20 ∧ rect.height = 25 ∧ squares.card = 120 →
  ∃ (cx cy : ℝ), cx ∈ Set.Icc 0.5 19.5 ∧ cy ∈ Set.Icc 0.5 24.5 ∧
    ∀ (s : UnitSquare), s ∈ squares →
      (cx - s.x) ^ 2 + (cy - s.y) ^ 2 > 0.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_fit_theorem_l1843_184324


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1843_184316

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  first : ℝ  -- First term of the sequence
  d : ℝ       -- Common difference
  seq_def : ∀ n, a n = first + (n - 1) * d
  sum_def : ∀ n, S n = n * (2 * first + (n - 1) * d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_difference 
  (seq : ArithmeticSequence)
  (h : seq.S 2016 / 2016 = seq.S 2015 / 2015 + 2) :
  seq.d = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1843_184316


namespace NUMINAMATH_CALUDE_parallel_line_l1843_184329

/-- A linear function in two variables -/
def LinearFunction (α : Type) [Ring α] := α → α → α

/-- A point in 2D space -/
structure Point (α : Type) [Ring α] where
  x : α
  y : α

/-- Theorem stating that the given equation represents a line parallel to l -/
theorem parallel_line
  {α : Type} [Field α]
  (f : LinearFunction α)
  (M N : Point α)
  (h1 : f M.x M.y = 0)
  (h2 : f N.x N.y ≠ 0) :
  ∃ (k : α), ∀ (P : Point α),
    f P.x P.y - f M.x M.y - f N.x N.y = 0 ↔ f P.x P.y = k :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_l1843_184329


namespace NUMINAMATH_CALUDE_wall_length_calculation_l1843_184354

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the wall's area,
    prove that the length of the wall is approximately 43 inches. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) : 
  mirror_side = 34 →
  wall_width = 54 →
  (mirror_side ^ 2) * 2 = wall_width * (round ((mirror_side ^ 2) * 2 / wall_width)) :=
by sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l1843_184354


namespace NUMINAMATH_CALUDE_corresponding_angles_are_equal_l1843_184336

-- Define the concept of angles
def Angle : Type := sorry

-- Define the property of being corresponding angles
def are_corresponding (a b : Angle) : Prop := sorry

-- Theorem statement
theorem corresponding_angles_are_equal (a b : Angle) : 
  are_corresponding a b → a = b := by
  sorry

end NUMINAMATH_CALUDE_corresponding_angles_are_equal_l1843_184336


namespace NUMINAMATH_CALUDE_S_equals_formula_S_2k_minus_1_is_polynomial_l1843_184391

-- Define S as a function of n and k
def S (n k : ℕ) : ℚ := sorry

-- Define S_{2k-1}(n) as a function
def S_2k_minus_1 (n k : ℕ) : ℚ := sorry

-- Theorem 1: S equals (n^k * (n+1)^k) / 2
theorem S_equals_formula (n k : ℕ) : 
  S n k = (n^k * (n+1)^k : ℚ) / 2 := by sorry

-- Theorem 2: S_{2k-1}(n) is a polynomial of degree k in (n(n+1))/2
theorem S_2k_minus_1_is_polynomial (n k : ℕ) :
  ∃ (p : Polynomial ℚ), 
    (S_2k_minus_1 n k = p.eval ((n * (n+1) : ℕ) / 2 : ℚ)) ∧ 
    (p.degree = k) := by sorry

end NUMINAMATH_CALUDE_S_equals_formula_S_2k_minus_1_is_polynomial_l1843_184391


namespace NUMINAMATH_CALUDE_second_quadrant_complex_number_range_l1843_184319

theorem second_quadrant_complex_number_range (m : ℝ) : 
  let z : ℂ := m - 1 + (m + 2) * I
  (z.re < 0 ∧ z.im > 0) ↔ -2 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_second_quadrant_complex_number_range_l1843_184319


namespace NUMINAMATH_CALUDE_rabbit_position_after_ten_exchanges_l1843_184337

-- Define the seats
inductive Seat
| one
| two
| three
| four

-- Define the animals
inductive Animal
| mouse
| monkey
| rabbit
| cat

-- Define the seating arrangement
def Arrangement := Seat → Animal

-- Define the initial arrangement
def initial_arrangement : Arrangement := fun seat =>
  match seat with
  | Seat.one => Animal.mouse
  | Seat.two => Animal.monkey
  | Seat.three => Animal.rabbit
  | Seat.four => Animal.cat

-- Define a single exchange operation
def exchange (arr : Arrangement) (n : ℕ) : Arrangement := 
  if n % 2 = 0 then
    fun seat =>
      match seat with
      | Seat.one => arr Seat.three
      | Seat.two => arr Seat.four
      | Seat.three => arr Seat.one
      | Seat.four => arr Seat.two
  else
    fun seat =>
      match seat with
      | Seat.one => arr Seat.two
      | Seat.two => arr Seat.one
      | Seat.three => arr Seat.four
      | Seat.four => arr Seat.three

-- Define multiple exchanges
def multiple_exchanges (arr : Arrangement) (n : ℕ) : Arrangement :=
  match n with
  | 0 => arr
  | n+1 => exchange (multiple_exchanges arr n) n

-- Theorem statement
theorem rabbit_position_after_ten_exchanges :
  ∃ (seat : Seat), (multiple_exchanges initial_arrangement 10) seat = Animal.rabbit ∧ seat = Seat.two :=
sorry

end NUMINAMATH_CALUDE_rabbit_position_after_ten_exchanges_l1843_184337


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l1843_184342

theorem inserted_numbers_sum (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  (∃ d : ℝ, a = 2 + d ∧ b = 2 + 2*d) ∧ 
  (∃ r : ℝ, b = a * r ∧ 18 = b * r) →
  a + b = 16 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l1843_184342


namespace NUMINAMATH_CALUDE_one_fourth_of_8_4_l1843_184330

theorem one_fourth_of_8_4 : (8.4 : ℚ) / 4 = 21 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_4_l1843_184330


namespace NUMINAMATH_CALUDE_sum_of_first_53_odd_numbers_l1843_184386

theorem sum_of_first_53_odd_numbers : 
  (Finset.range 53).sum (fun n => 2 * n + 1) = 2809 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_53_odd_numbers_l1843_184386


namespace NUMINAMATH_CALUDE_annulus_area_l1843_184351

/-- The area of an annulus with outer radius 8 feet and inner radius 2 feet is 60π square feet. -/
theorem annulus_area : ∀ (π : ℝ), π > 0 → π * (8^2 - 2^2) = 60 * π := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l1843_184351


namespace NUMINAMATH_CALUDE_empty_set_subset_of_all_l1843_184384

theorem empty_set_subset_of_all (A : Set α) : ∅ ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_empty_set_subset_of_all_l1843_184384


namespace NUMINAMATH_CALUDE_fair_products_l1843_184348

/-- The number of recycled materials made by the group -/
def group_materials : ℕ := 65

/-- The number of recycled materials made by the teachers -/
def teacher_materials : ℕ := 28

/-- The total number of recycled products to sell at the fair -/
def total_products : ℕ := group_materials + teacher_materials

theorem fair_products : total_products = 93 := by
  sorry

end NUMINAMATH_CALUDE_fair_products_l1843_184348


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1843_184309

theorem quadratic_inequality (x : ℝ) : x^2 - 9*x + 18 ≤ 0 ↔ 3 ≤ x ∧ x ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1843_184309


namespace NUMINAMATH_CALUDE_pascal_triangle_prob_one_or_two_l1843_184383

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Type := Unit

/-- Total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (pt : PascalTriangle n) : ℕ := n * (n + 1) / 2

/-- Number of elements equal to 1 in the first n rows of Pascal's Triangle -/
def countOnes (pt : PascalTriangle n) : ℕ := 1 + 2 * (n - 1)

/-- Number of elements equal to 2 in the first n rows of Pascal's Triangle -/
def countTwos (pt : PascalTriangle n) : ℕ := 2 * (n - 3)

/-- Probability of selecting 1 or 2 from the first n rows of Pascal's Triangle -/
def probOneOrTwo (pt : PascalTriangle n) : ℚ :=
  (countOnes pt + countTwos pt : ℚ) / totalElements pt

theorem pascal_triangle_prob_one_or_two :
  ∃ (pt : PascalTriangle 20), probOneOrTwo pt = 73 / 210 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_prob_one_or_two_l1843_184383


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achieved_l1843_184362

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z) ≥ 36 := by
sorry

theorem min_value_achieved (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) : 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
  x₀ + y₀ + z₀ = 1 ∧ 
  (1 / x₀ + 4 / y₀ + 9 / z₀) = 36 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achieved_l1843_184362


namespace NUMINAMATH_CALUDE_base_five_digits_of_1234_l1843_184358

theorem base_five_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 5^(n-1) ≤ 1234 ∧ 1234 < 5^n :=
  sorry

end NUMINAMATH_CALUDE_base_five_digits_of_1234_l1843_184358


namespace NUMINAMATH_CALUDE_solve_equation_l1843_184355

/-- Define the determinant-like operation for four rational numbers -/
def det (a b c d : ℚ) : ℚ := a * d - b * c

/-- Theorem stating that given the condition, x must equal 3 -/
theorem solve_equation : ∃ x : ℚ, det (2*x) (-4) x 1 = 18 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1843_184355


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_range_l1843_184376

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ↔ -3 ≤ a ∧ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_range_l1843_184376


namespace NUMINAMATH_CALUDE_quadratic_roots_and_specific_case_l1843_184380

/-- The quadratic equation x^2 - (m-1)x = 3 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - (m-1)*x = 3

theorem quadratic_roots_and_specific_case :
  (∀ m : ℝ, ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y) ∧
  (∃ m : ℝ, quadratic_equation m 2 ∧ quadratic_equation m (-3/2) ∧ m = 5/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_specific_case_l1843_184380


namespace NUMINAMATH_CALUDE_max_profit_selling_price_l1843_184323

-- Define the profit function
def profit (x : ℝ) : ℝ := 10 * (-x^2 + 140*x - 4000)

-- Define the theorem
theorem max_profit_selling_price :
  -- Given conditions
  let cost_price : ℝ := 40
  let initial_price : ℝ := 50
  let initial_sales : ℝ := 500
  let price_sensitivity : ℝ := 10

  -- Theorem statement
  ∃ (max_price max_profit : ℝ),
    -- The maximum price is greater than the cost price
    max_price > cost_price ∧
    -- The maximum profit occurs at the maximum price
    profit max_price = max_profit ∧
    -- The maximum profit is indeed the maximum
    ∀ x > cost_price, profit x ≤ max_profit ∧
    -- The specific values for maximum price and profit
    max_price = 70 ∧ max_profit = 9000 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_selling_price_l1843_184323


namespace NUMINAMATH_CALUDE_smaller_circle_circumference_l1843_184373

theorem smaller_circle_circumference 
  (R : ℝ) 
  (h1 : R > 0) 
  (h2 : π * (3*R)^2 - π * R^2 = 32 / π) : 
  2 * π * R = 4 := by
sorry

end NUMINAMATH_CALUDE_smaller_circle_circumference_l1843_184373


namespace NUMINAMATH_CALUDE_remainder_property_l1843_184381

/-- A polynomial of the form Dx^6 + Ex^4 + Fx^2 + 7 -/
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^6 + E * x^4 + F * x^2 + 7

/-- The remainder theorem -/
def remainder_theorem (p : ℝ → ℝ) (a : ℝ) : ℝ := p a

theorem remainder_property (D E F : ℝ) :
  remainder_theorem (q D E F) 2 = 17 →
  remainder_theorem (q D E F) (-2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_property_l1843_184381


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l1843_184356

def v : ℕ := (List.range 30).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : (3^k ∣ v) ↔ k ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l1843_184356


namespace NUMINAMATH_CALUDE_left_of_kolya_l1843_184350

/-- The number of people in a class lineup -/
structure ClassLineup where
  total : ℕ
  leftOfSasha : ℕ
  rightOfSasha : ℕ
  rightOfKolya : ℕ
  leftOfKolya : ℕ

/-- Theorem stating the number of people to the left of Kolya -/
theorem left_of_kolya (c : ClassLineup)
  (h1 : c.leftOfSasha = 20)
  (h2 : c.rightOfSasha = 8)
  (h3 : c.rightOfKolya = 12)
  (h4 : c.total = c.leftOfSasha + c.rightOfSasha + 1)
  (h5 : c.total = c.leftOfKolya + c.rightOfKolya + 1) :
  c.leftOfKolya = 16 := by
  sorry

end NUMINAMATH_CALUDE_left_of_kolya_l1843_184350


namespace NUMINAMATH_CALUDE_womans_swimming_speed_l1843_184302

/-- Given a woman who swims downstream and upstream with specific distances and times,
    this theorem proves her speed in still water. -/
theorem womans_swimming_speed
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (h_downstream : downstream_distance = 54)
  (h_upstream : upstream_distance = 6)
  (h_time : downstream_time = 6 ∧ upstream_time = 6)
  : ∃ (speed_still_water : ℝ) (stream_speed : ℝ),
    speed_still_water = 5 ∧
    downstream_distance / downstream_time = speed_still_water + stream_speed ∧
    upstream_distance / upstream_time = speed_still_water - stream_speed :=
by sorry

end NUMINAMATH_CALUDE_womans_swimming_speed_l1843_184302


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l1843_184392

/-- Given a cone with an acute triangular cross-section and slant height 4,
    where the maximum area of cross-sections passing through the vertex is 4√3,
    prove that the central angle of the sector in the lateral surface development is π. -/
theorem cone_lateral_surface_angle (h : ℝ) (θ : ℝ) (r : ℝ) (α : ℝ) : 
  h = 4 → 
  θ < π / 2 →
  (1 / 2) * h * h * Real.sin θ = 4 * Real.sqrt 3 →
  r = 2 →
  α = 2 * π * r / h →
  α = π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l1843_184392


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1843_184389

theorem trigonometric_identity : 
  Real.sin (315 * π / 180) - Real.cos (135 * π / 180) + 2 * Real.sin (570 * π / 180) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1843_184389


namespace NUMINAMATH_CALUDE_perfect_squares_between_50_and_200_l1843_184340

theorem perfect_squares_between_50_and_200 : 
  (Finset.filter (fun n : ℕ => 50 < n^2 ∧ n^2 ≤ 200) (Finset.range 201)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_50_and_200_l1843_184340


namespace NUMINAMATH_CALUDE_tan_360_minus_45_l1843_184352

theorem tan_360_minus_45 : Real.tan (360 * π / 180 - 45 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_360_minus_45_l1843_184352


namespace NUMINAMATH_CALUDE_curve_C_equation_min_area_QAB_l1843_184312

-- Define the parabola E
def parabola_E (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point N on parabola E
def point_N (x y : ℝ) : Prop := parabola_E x y

-- Define point O as the origin
def point_O : ℝ × ℝ := (0, 0)

-- Define point P as the midpoint of ON
def point_P (x y : ℝ) : Prop := ∃ (nx ny : ℝ), point_N nx ny ∧ x = nx / 2 ∧ y = ny / 2

-- Define curve C as the trajectory of point P
def curve_C (x y : ℝ) : Prop := point_P x y

-- Define point Q on curve C with x₀ ≥ 5
def point_Q (x₀ y₀ : ℝ) : Prop := curve_C x₀ y₀ ∧ x₀ ≥ 5

-- Theorem for the equation of curve C
theorem curve_C_equation (x y : ℝ) : curve_C x y → y^2 = 4 * x := by sorry

-- Theorem for the minimum area of △QAB
theorem min_area_QAB (x₀ y₀ : ℝ) (hQ : point_Q x₀ y₀) : 
  ∃ (A B : ℝ × ℝ), (∀ (area : ℝ), area ≥ 25/2) := by sorry

end NUMINAMATH_CALUDE_curve_C_equation_min_area_QAB_l1843_184312


namespace NUMINAMATH_CALUDE_parallel_lines_point_on_circle_l1843_184310

def line1 (a b x y : ℝ) : Prop := (b + 2) * x + a * y + 4 = 0

def line2 (a b x y : ℝ) : Prop := a * x + (2 - b) * y - 3 = 0

def parallel (f g : ℝ → ℝ → Prop) : Prop := 
  ∀ x₁ y₁ x₂ y₂, f x₁ y₁ ∧ f x₂ y₂ → (x₁ ≠ x₂ → (y₁ - y₂) / (x₁ - x₂) = (y₂ - y₁) / (x₂ - x₁))

theorem parallel_lines_point_on_circle (a b : ℝ) :
  parallel (line1 a b) (line2 a b) → a^2 + b^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_point_on_circle_l1843_184310


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1843_184359

theorem algebraic_expression_equality (x y : ℝ) (h : x - 2*y + 8 = 18) :
  3*x - 6*y + 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1843_184359


namespace NUMINAMATH_CALUDE_expected_value_max_two_dice_rolls_expected_value_max_two_dice_rolls_eq_l1843_184396

/-- The expected value of the maximum of two independent rolls of a fair six-sided die -/
theorem expected_value_max_two_dice_rolls : ℝ :=
  let X : Fin 6 → ℝ := λ i => (i : ℝ) + 1
  let P : Fin 6 → ℝ := λ i =>
    match i with
    | 0 => 1 / 36
    | 1 => 3 / 36
    | 2 => 5 / 36
    | 3 => 7 / 36
    | 4 => 9 / 36
    | 5 => 11 / 36
  161 / 36

/-- The expected value of the maximum of two independent rolls of a fair six-sided die is 161/36 -/
theorem expected_value_max_two_dice_rolls_eq : expected_value_max_two_dice_rolls = 161 / 36 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_max_two_dice_rolls_expected_value_max_two_dice_rolls_eq_l1843_184396


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1843_184313

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 5 * z) = 7 :=
by
  -- The unique solution is z = -44/5
  use -44/5
  constructor
  -- Prove that -44/5 satisfies the equation
  · sorry
  -- Prove uniqueness
  · sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1843_184313


namespace NUMINAMATH_CALUDE_bucket_weight_l1843_184346

/-- Given a bucket with weight c when 1/4 full and weight d when 3/4 full,
    prove that its weight when full is (3d - 3c)/2 -/
theorem bucket_weight (c d : ℝ) 
  (h1 : ∃ x y : ℝ, x + (1/4) * y = c ∧ x + (3/4) * y = d) : 
  ∃ w : ℝ, w = (3*d - 3*c)/2 ∧ 
  (∃ x y : ℝ, x + y = w ∧ x + (1/4) * y = c ∧ x + (3/4) * y = d) :=
by sorry

end NUMINAMATH_CALUDE_bucket_weight_l1843_184346


namespace NUMINAMATH_CALUDE_gum_distribution_l1843_184368

theorem gum_distribution (cousins : ℕ) (total_gum : ℕ) (gum_per_cousin : ℕ) : 
  cousins = 4 → total_gum = 20 → gum_per_cousin = total_gum / cousins → gum_per_cousin = 5 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l1843_184368


namespace NUMINAMATH_CALUDE_probability_chords_intersect_2000_probability_chords_intersect_general_l1843_184390

/-- Given a circle with evenly spaced points, this function calculates the probability
    that chord AB intersects chord CD when five distinct points are randomly selected. -/
def probability_chords_intersect (n : ℕ) : ℚ :=
  if n < 5 then 0
  else 1 / 15

/-- Theorem stating that the probability of chord AB intersecting chord CD
    when five distinct points are randomly selected from 2000 evenly spaced
    points on a circle is 1/15. -/
theorem probability_chords_intersect_2000 :
  probability_chords_intersect 2000 = 1 / 15 := by
  sorry

/-- Theorem stating that the probability of chord AB intersecting chord CD
    is 1/15 for any number of evenly spaced points on a circle, as long as
    there are at least 5 points. -/
theorem probability_chords_intersect_general (n : ℕ) (h : n ≥ 5) :
  probability_chords_intersect n = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_chords_intersect_2000_probability_chords_intersect_general_l1843_184390


namespace NUMINAMATH_CALUDE_inequality_proof_l1843_184398

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a^5 + b^5 ≤ 1) (h6 : c^5 + d^5 ≤ 1) : 
  a^2 * c^3 + b^2 * d^3 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1843_184398


namespace NUMINAMATH_CALUDE_tv_purchase_price_l1843_184303

/-- Proves that the purchase price of a TV is 1200 yuan given the markup, promotion, and profit conditions. -/
theorem tv_purchase_price (x : ℝ) 
  (markup : ℝ → ℝ) 
  (promotion : ℝ → ℝ) 
  (profit : ℝ) 
  (h1 : markup x = 1.35 * x) 
  (h2 : promotion (markup x) = 0.9 * markup x - 50)
  (h3 : promotion (markup x) - x = profit)
  (h4 : profit = 208) : 
  x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_tv_purchase_price_l1843_184303


namespace NUMINAMATH_CALUDE_height_to_hypotenuse_l1843_184357

theorem height_to_hypotenuse (a b c h : ℝ) : 
  a = 6 → b = 8 → c = 10 → a^2 + b^2 = c^2 → (a * b) / 2 = (c * h) / 2 → h = 4.8 :=
by sorry

end NUMINAMATH_CALUDE_height_to_hypotenuse_l1843_184357


namespace NUMINAMATH_CALUDE_percentage_difference_l1843_184332

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.45)) :
  y = x * (1 + 0.45) := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1843_184332


namespace NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_five_squared_l1843_184305

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_five_squared_l1843_184305


namespace NUMINAMATH_CALUDE_cosine_ratio_equals_negative_sqrt_three_l1843_184387

theorem cosine_ratio_equals_negative_sqrt_three : 
  (2 * Real.cos (80 * π / 180) + Real.cos (160 * π / 180)) / Real.cos (70 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_ratio_equals_negative_sqrt_three_l1843_184387


namespace NUMINAMATH_CALUDE_imaginary_part_of_symmetrical_complex_ratio_l1843_184322

theorem imaginary_part_of_symmetrical_complex_ratio :
  ∀ (z₁ z₂ : ℂ),
  z₁ = 1 - 2*I →
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) →
  Complex.im (z₂ / z₁) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_symmetrical_complex_ratio_l1843_184322


namespace NUMINAMATH_CALUDE_parabola_vertex_l1843_184308

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = (x - 6)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (6, 3)

/-- Theorem: The vertex of the parabola y = (x - 6)^2 + 3 is at (6, 3) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1843_184308


namespace NUMINAMATH_CALUDE_shortest_diagonal_probability_l1843_184375

/-- The number of sides in the regular polygon -/
def n : ℕ := 11

/-- The total number of diagonals in the polygon -/
def total_diagonals : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in the polygon -/
def shortest_diagonals : ℕ := n / 2

/-- The probability of selecting a shortest diagonal -/
def probability : ℚ := shortest_diagonals / total_diagonals

theorem shortest_diagonal_probability :
  probability = 5 / 44 := by sorry

end NUMINAMATH_CALUDE_shortest_diagonal_probability_l1843_184375


namespace NUMINAMATH_CALUDE_ellipse_condition_l1843_184393

def is_ellipse_equation (m : ℝ) : Prop :=
  m > 2 ∧ m < 5 ∧ m ≠ 7/2

theorem ellipse_condition (m : ℝ) :
  (2 < m ∧ m < 5) → (is_ellipse_equation m) ∧
  ∃ m', is_ellipse_equation m' ∧ ¬(2 < m' ∧ m' < 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1843_184393


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l1843_184378

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 39 →
  football = 26 →
  tennis = 20 →
  neither = 10 →
  ∃ (both : ℕ), both = 17 ∧
    total = football + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l1843_184378


namespace NUMINAMATH_CALUDE_jims_age_l1843_184321

theorem jims_age (j t : ℕ) (h1 : j = 3 * t + 10) (h2 : j + t = 70) : j = 55 := by
  sorry

end NUMINAMATH_CALUDE_jims_age_l1843_184321


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l1843_184343

/-- The probability of drawing a white ball from a bag containing white and red balls -/
theorem probability_of_white_ball (num_white : ℕ) (num_red : ℕ) : 
  num_white = 6 → num_red = 14 → (num_white : ℚ) / (num_white + num_red : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l1843_184343


namespace NUMINAMATH_CALUDE_fraction_sum_problem_l1843_184382

theorem fraction_sum_problem (x y : ℚ) (h : x / y = 2 / 7) : (x + y) / y = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_problem_l1843_184382


namespace NUMINAMATH_CALUDE_seating_arrangements_l1843_184397

-- Define the number of people excluding the fixed person
def n : ℕ := 4

-- Define the function to calculate the total number of permutations
def total_permutations (n : ℕ) : ℕ := n.factorial

-- Define the function to calculate the number of permutations where two specific people are adjacent
def adjacent_permutations (n : ℕ) : ℕ := 2 * (n - 1).factorial

-- Theorem statement
theorem seating_arrangements :
  total_permutations n - adjacent_permutations n = 12 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1843_184397


namespace NUMINAMATH_CALUDE_total_spent_calculation_l1843_184379

def lunch_cost : ℝ := 50.20
def tip_rate : ℝ := 0.20

theorem total_spent_calculation :
  lunch_cost * (1 + tip_rate) = 60.24 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_calculation_l1843_184379


namespace NUMINAMATH_CALUDE_boat_fuel_cost_is_50_l1843_184363

/-- The boat fuel cost per hour for Pat's shark hunting -/
def boat_fuel_cost_per_hour : ℚ :=
  let photo_earning : ℚ := 15
  let shark_interval : ℚ := 10 / 60  -- 10 minutes in hours
  let hunting_duration : ℚ := 5
  let expected_profit : ℚ := 200
  let total_sharks : ℚ := hunting_duration / shark_interval
  let total_earnings : ℚ := total_sharks * photo_earning
  let total_fuel_cost : ℚ := total_earnings - expected_profit
  total_fuel_cost / hunting_duration

/-- Theorem stating that the boat fuel cost per hour is $50 -/
theorem boat_fuel_cost_is_50 : boat_fuel_cost_per_hour = 50 := by
  sorry

end NUMINAMATH_CALUDE_boat_fuel_cost_is_50_l1843_184363


namespace NUMINAMATH_CALUDE_wine_price_increase_l1843_184320

/-- The additional cost for 5 bottles of wine after a 25% price increase -/
theorem wine_price_increase (current_price : ℝ) (num_bottles : ℕ) (price_increase_percent : ℝ) :
  current_price = 20 →
  num_bottles = 5 →
  price_increase_percent = 0.25 →
  num_bottles * current_price * price_increase_percent = 25 :=
by sorry

end NUMINAMATH_CALUDE_wine_price_increase_l1843_184320


namespace NUMINAMATH_CALUDE_square_weight_calculation_l1843_184341

theorem square_weight_calculation (density : ℝ) (thickness : ℝ) 
  (side_length1 : ℝ) (weight1 : ℝ) (side_length2 : ℝ) 
  (h1 : density > 0) (h2 : thickness > 0) 
  (h3 : side_length1 = 4) (h4 : weight1 = 16) (h5 : side_length2 = 6) :
  let weight2 := density * thickness * side_length2^2
  weight2 = 36 := by
  sorry

#check square_weight_calculation

end NUMINAMATH_CALUDE_square_weight_calculation_l1843_184341


namespace NUMINAMATH_CALUDE_angle_bisector_length_l1843_184327

-- Define the triangle DEF
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (DF : ℝ)

-- Define the angle bisector EG
def angleBisector (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_bisector_length (t : Triangle) 
  (h1 : t.DE = 4)
  (h2 : t.EF = 5)
  (h3 : t.DF = 6) :
  angleBisector t = 3 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l1843_184327


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1843_184374

theorem unique_integer_solution : 
  ∃! x : ℤ, (12*x - 1) * (6*x - 1) * (4*x - 1) * (3*x - 1) = 330 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1843_184374


namespace NUMINAMATH_CALUDE_jason_shopping_total_l1843_184370

theorem jason_shopping_total (jacket_cost shorts_cost : ℚ) 
  (h1 : jacket_cost = 4.74)
  (h2 : shorts_cost = 9.54) :
  jacket_cost + shorts_cost = 14.28 := by
  sorry

end NUMINAMATH_CALUDE_jason_shopping_total_l1843_184370


namespace NUMINAMATH_CALUDE_rectangle_perimeter_ratio_l1843_184372

theorem rectangle_perimeter_ratio :
  let original_width : ℚ := 6
  let original_height : ℚ := 8
  let folded_height : ℚ := original_height / 2
  let small_width : ℚ := original_width / 2
  let small_height : ℚ := folded_height
  let original_perimeter : ℚ := 2 * (original_width + original_height)
  let small_perimeter : ℚ := 2 * (small_width + small_height)
  small_perimeter / original_perimeter = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_ratio_l1843_184372


namespace NUMINAMATH_CALUDE_notebook_distribution_l1843_184334

theorem notebook_distribution (total notebooks k v y s se : ℕ) : 
  notebooks = 100 ∧
  k + v = 52 ∧
  v + y = 43 ∧
  y + s = 34 ∧
  s + se = 30 ∧
  k + v + y + s + se = notebooks →
  k = 27 ∧ v = 25 ∧ y = 18 ∧ s = 16 ∧ se = 14 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l1843_184334


namespace NUMINAMATH_CALUDE_ratio_calculation_l1843_184345

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by sorry

end NUMINAMATH_CALUDE_ratio_calculation_l1843_184345


namespace NUMINAMATH_CALUDE_fence_whitewashing_fence_theorem_l1843_184366

theorem fence_whitewashing (total_fence : ℝ) (ben_amount : ℝ) 
  (billy_fraction : ℝ) (johnny_fraction : ℝ) : ℝ :=
  let remaining_after_ben := total_fence - ben_amount
  let billy_amount := billy_fraction * remaining_after_ben
  let remaining_after_billy := remaining_after_ben - billy_amount
  let johnny_amount := johnny_fraction * remaining_after_billy
  let final_remaining := remaining_after_billy - johnny_amount
  final_remaining

theorem fence_theorem : 
  fence_whitewashing 100 10 (1/5) (1/3) = 48 := by
  sorry

end NUMINAMATH_CALUDE_fence_whitewashing_fence_theorem_l1843_184366


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1843_184361

theorem quadratic_inequality_solution_sets (a : ℝ) :
  let f := fun x => x^2 - (1 + a) * x + a
  (a = 2 → {x | f x > 0} = {x | x > 2 ∨ x < 1}) ∧
  (a > 1 → {x | f x > 0} = {x | x > a ∨ x < 1}) ∧
  (a = 1 → {x | f x > 0} = {x | x ≠ 1}) ∧
  (a < 1 → {x | f x > 0} = {x | x > 1 ∨ x < a}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1843_184361


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l1843_184314

theorem mystery_book_shelves (books_per_shelf : ℕ) (picture_book_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 9 →
  picture_book_shelves = 2 →
  total_books = 72 →
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 6 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l1843_184314


namespace NUMINAMATH_CALUDE_expected_yield_for_80kg_fertilizer_l1843_184388

/-- Represents the regression line equation for rice yield based on fertilizer amount -/
def regression_line (x : ℝ) : ℝ := 5 * x + 250

/-- Theorem stating that the expected rice yield is 650 kg when 80 kg of fertilizer is applied -/
theorem expected_yield_for_80kg_fertilizer : 
  regression_line 80 = 650 := by sorry

end NUMINAMATH_CALUDE_expected_yield_for_80kg_fertilizer_l1843_184388


namespace NUMINAMATH_CALUDE_sewage_treatment_equipment_costs_l1843_184306

theorem sewage_treatment_equipment_costs (a b : ℝ) : 
  (a - b = 3) → (3 * b - 2 * a = 3) → (a = 12 ∧ b = 9) :=
by sorry

end NUMINAMATH_CALUDE_sewage_treatment_equipment_costs_l1843_184306


namespace NUMINAMATH_CALUDE_product_remainder_remainder_1287_1499_300_l1843_184335

theorem product_remainder (a b m : ℕ) (h : m > 0) : (a * b) % m = ((a % m) * (b % m)) % m := by sorry

theorem remainder_1287_1499_300 : (1287 * 1499) % 300 = 213 := by sorry

end NUMINAMATH_CALUDE_product_remainder_remainder_1287_1499_300_l1843_184335


namespace NUMINAMATH_CALUDE_pythagorean_triples_l1843_184399

theorem pythagorean_triples (n m : ℕ) : 
  (n ≥ 3 ∧ Odd n) → 
  ((n^2 - 1) / 2)^2 + n^2 = ((n^2 + 1) / 2)^2 ∧
  (m > 1) →
  (m^2 - 1)^2 + (2*m)^2 = (m^2 + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_triples_l1843_184399
