import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_l1732_173206

def is_valid (n : ℕ) : Prop :=
  n ≥ 6 ∧ n.choose 4 * 24 ≤ n.factorial / ((n - 4).factorial)

theorem solution_set :
  {n : ℕ | is_valid n} = {6, 7, 8, 9} := by sorry

end NUMINAMATH_CALUDE_solution_set_l1732_173206


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1732_173243

theorem quadratic_no_real_roots (a b c : ℝ) (h : b^2 = a*c) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1732_173243


namespace NUMINAMATH_CALUDE_average_age_proof_l1732_173218

def john_age (mary_age : ℕ) : ℕ := 2 * mary_age

def tonya_age : ℕ := 60

theorem average_age_proof (mary_age : ℕ) (h1 : john_age mary_age = tonya_age / 2) :
  (mary_age + john_age mary_age + tonya_age) / 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_age_proof_l1732_173218


namespace NUMINAMATH_CALUDE_expected_rolls_in_year_l1732_173240

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
  | composite
  | prime
  | reroll

/-- Represents the rules for Bob's breakfast die -/
def breakfastDie : Fin 8 → DieOutcome
  | 1 => DieOutcome.reroll
  | 2 => DieOutcome.prime
  | 3 => DieOutcome.prime
  | 4 => DieOutcome.composite
  | 5 => DieOutcome.prime
  | 6 => DieOutcome.composite
  | 7 => DieOutcome.reroll
  | 8 => DieOutcome.reroll

/-- The probability of getting each outcome -/
def outcomeProb : DieOutcome → Rat
  | DieOutcome.composite => 2/8
  | DieOutcome.prime => 3/8
  | DieOutcome.reroll => 3/8

/-- The number of days in a non-leap year -/
def daysInYear : Nat := 365

/-- Theorem stating the expected number of rolls in a non-leap year -/
theorem expected_rolls_in_year :
  let expectedRollsPerDay := 8/5
  (expectedRollsPerDay * daysInYear : Rat) = 584 := by
  sorry

end NUMINAMATH_CALUDE_expected_rolls_in_year_l1732_173240


namespace NUMINAMATH_CALUDE_log_one_over_81_base_3_l1732_173256

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_over_81_base_3 : log 3 (1/81) = -4 := by
  sorry

end NUMINAMATH_CALUDE_log_one_over_81_base_3_l1732_173256


namespace NUMINAMATH_CALUDE_gcd_problem_l1732_173266

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 714 * k) :
  Int.gcd (5 * b^3 + 2 * b^2 + 6 * b + 102) b = 102 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1732_173266


namespace NUMINAMATH_CALUDE_steven_needs_three_more_seeds_l1732_173253

/-- Represents the number of seeds Steven needs to collect for his assignment -/
def total_seeds_required : ℕ := 60

/-- Represents the average number of seeds in an apple -/
def apple_seeds : ℕ := 6

/-- Represents the average number of seeds in a pear -/
def pear_seeds : ℕ := 2

/-- Represents the average number of seeds in a grape -/
def grape_seeds : ℕ := 3

/-- Represents the number of apples Steven has -/
def steven_apples : ℕ := 4

/-- Represents the number of pears Steven has -/
def steven_pears : ℕ := 3

/-- Represents the number of grapes Steven has -/
def steven_grapes : ℕ := 9

/-- Theorem stating that Steven needs 3 more seeds to fulfill his assignment -/
theorem steven_needs_three_more_seeds :
  total_seeds_required - (steven_apples * apple_seeds + steven_pears * pear_seeds + steven_grapes * grape_seeds) = 3 := by
  sorry

end NUMINAMATH_CALUDE_steven_needs_three_more_seeds_l1732_173253


namespace NUMINAMATH_CALUDE_frank_has_twelve_cookies_l1732_173295

-- Define the number of cookies each person has
def lucy_cookies : ℕ := 5
def millie_cookies : ℕ := 2 * lucy_cookies
def mike_cookies : ℕ := 3 * millie_cookies
def frank_cookies : ℕ := mike_cookies / 2 - 3

-- Theorem to prove
theorem frank_has_twelve_cookies : frank_cookies = 12 := by
  sorry

end NUMINAMATH_CALUDE_frank_has_twelve_cookies_l1732_173295


namespace NUMINAMATH_CALUDE_translation_right_4_units_l1732_173247

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in the x-direction -/
def translateX (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

theorem translation_right_4_units (P : Point) (P' : Point) :
  P.x = -2 ∧ P.y = 3 →
  P' = translateX P 4 →
  P'.x = 2 ∧ P'.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_translation_right_4_units_l1732_173247


namespace NUMINAMATH_CALUDE_system_solution_l1732_173246

theorem system_solution (a b c A B C x y z : ℝ) : 
  (x + a * y + a^2 * z = A) ∧ 
  (x + b * y + b^2 * z = B) ∧ 
  (x + c * y + c^2 * z = C) →
  ((A = b + c ∧ B = c + a ∧ C = a + b) → 
    (z = 0 ∧ y = -1 ∧ x = A + b)) ∧
  ((A = b * c ∧ B = c * a ∧ C = a * b) → 
    (z = 1 ∧ y = -(a + b + c) ∧ x = a * b * c)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1732_173246


namespace NUMINAMATH_CALUDE_factorization_proof_l1732_173286

theorem factorization_proof (b : ℝ) : 2 * b^2 - 8 * b + 8 = 2 * (b - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1732_173286


namespace NUMINAMATH_CALUDE_businessmen_who_drank_nothing_l1732_173241

/-- The number of businessmen who drank neither coffee, tea, nor soda -/
theorem businessmen_who_drank_nothing (total : ℕ) (coffee tea soda : ℕ) 
  (coffee_and_tea tea_and_soda coffee_and_soda : ℕ) (all_three : ℕ) : 
  total = 40 →
  coffee = 20 →
  tea = 15 →
  soda = 10 →
  coffee_and_tea = 8 →
  tea_and_soda = 4 →
  coffee_and_soda = 3 →
  all_three = 2 →
  total - (coffee + tea + soda - coffee_and_tea - tea_and_soda - coffee_and_soda + all_three) = 8 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_who_drank_nothing_l1732_173241


namespace NUMINAMATH_CALUDE_subtraction_in_third_quadrant_l1732_173287

/-- Given complex numbers z₁ and z₂, prove that z₁ - z₂ is in the third quadrant -/
theorem subtraction_in_third_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = -2 + I) 
  (h₂ : z₂ = 1 + 2*I) : 
  let z := z₁ - z₂
  (z.re < 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_in_third_quadrant_l1732_173287


namespace NUMINAMATH_CALUDE_stratified_sampling_l1732_173254

/-- Represents the number of questionnaires for each unit -/
structure Questionnaires :=
  (a b c d : ℕ)

/-- Represents the sample sizes for each unit -/
structure SampleSizes :=
  (a b c d : ℕ)

/-- Checks if the given questionnaire counts form an arithmetic sequence -/
def is_arithmetic_sequence (q : Questionnaires) : Prop :=
  q.b - q.a = q.c - q.b ∧ q.c - q.b = q.d - q.c

/-- The main theorem statement -/
theorem stratified_sampling
  (q : Questionnaires)
  (s : SampleSizes)
  (h1 : q.a + q.b + q.c + q.d = 1000)
  (h2 : is_arithmetic_sequence q)
  (h3 : s.a + s.b + s.c + s.d = 150)
  (h4 : s.b = 30)
  (h5 : s.b * 1000 = q.b * 150) :
  s.d = 60 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l1732_173254


namespace NUMINAMATH_CALUDE_horner_method_v2_l1732_173258

def f (x : ℝ) : ℝ := x^6 - 8*x^5 + 60*x^4 + 16*x^3 + 96*x^2 + 240*x + 64

def horner_v2 (a : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * a - 8
  v1 * a + 60

theorem horner_method_v2 :
  horner_v2 2 = 48 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l1732_173258


namespace NUMINAMATH_CALUDE_fourth_grade_students_l1732_173248

theorem fourth_grade_students (initial_students leaving_students new_students : ℕ) :
  initial_students = 35 →
  leaving_students = 10 →
  new_students = 10 →
  initial_students - leaving_students + new_students = 35 := by
sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l1732_173248


namespace NUMINAMATH_CALUDE_simplification_and_constant_coefficient_l1732_173293

-- Define the expression as a function of x and square
def expression (x : ℝ) (square : ℝ) : ℝ :=
  (square * x^2 + 6*x + 8) - (6*x + 5*x^2 + 2)

theorem simplification_and_constant_coefficient :
  (∀ x : ℝ, expression x 3 = -2 * x^2 + 6) ∧
  (∃! square : ℝ, ∀ x : ℝ, expression x square = (expression 0 square)) :=
by sorry

end NUMINAMATH_CALUDE_simplification_and_constant_coefficient_l1732_173293


namespace NUMINAMATH_CALUDE_vertical_angles_are_congruent_l1732_173215

-- Define what it means for two angles to be vertical
def are_vertical_angles (α β : Angle) : Prop := sorry

-- Define what it means for two angles to be congruent
def are_congruent (α β : Angle) : Prop := sorry

-- Theorem statement
theorem vertical_angles_are_congruent (α β : Angle) : 
  are_vertical_angles α β → are_congruent α β := by
  sorry

end NUMINAMATH_CALUDE_vertical_angles_are_congruent_l1732_173215


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l1732_173272

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- State the theorem
theorem hyperbola_triangle_perimeter :
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  (A.1 < 0 ∧ B.1 < 0) →  -- A and B are on the left branch
  F₁ ∈ Set.Icc A B →     -- F₁ is on the line segment AB
  dist A B = 6 →
  dist A F₂ + dist B F₂ + dist A B = 28 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l1732_173272


namespace NUMINAMATH_CALUDE_notebooks_left_l1732_173290

theorem notebooks_left (notebooks_per_bundle : ℕ) (num_bundles : ℕ) (num_groups : ℕ) (students_per_group : ℕ) : 
  notebooks_per_bundle = 25 →
  num_bundles = 5 →
  num_groups = 8 →
  students_per_group = 13 →
  num_bundles * notebooks_per_bundle - num_groups * students_per_group = 21 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_left_l1732_173290


namespace NUMINAMATH_CALUDE_square_sum_not_equal_sum_squares_l1732_173227

theorem square_sum_not_equal_sum_squares : ∃ (a b : ℝ), a^2 + b^2 ≠ (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_not_equal_sum_squares_l1732_173227


namespace NUMINAMATH_CALUDE_annie_gives_mary_25_crayons_l1732_173228

/-- The number of crayons Annie gives to Mary -/
def crayons_given_to_mary (pack : ℕ) (locker : ℕ) : ℕ :=
  let initial_total := pack + locker
  let from_bobby := locker / 2
  let final_total := initial_total + from_bobby
  final_total / 3

/-- Theorem stating that Annie gives 25 crayons to Mary -/
theorem annie_gives_mary_25_crayons :
  crayons_given_to_mary 21 36 = 25 := by
  sorry

#eval crayons_given_to_mary 21 36

end NUMINAMATH_CALUDE_annie_gives_mary_25_crayons_l1732_173228


namespace NUMINAMATH_CALUDE_tina_pens_theorem_l1732_173267

def pink_pens : ℕ := 15
def green_pens : ℕ := pink_pens - 9
def blue_pens : ℕ := green_pens + 3
def yellow_pens : ℕ := pink_pens + green_pens - 5
def pens_used_per_day : ℕ := 4

theorem tina_pens_theorem :
  let total_pens := pink_pens + green_pens + blue_pens + yellow_pens
  let days_to_use_pink := (pink_pens + pens_used_per_day - 1) / pens_used_per_day
  total_pens = 46 ∧ days_to_use_pink = 4 := by
  sorry

end NUMINAMATH_CALUDE_tina_pens_theorem_l1732_173267


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1732_173260

theorem arithmetic_equality : 1234562 - 12 * 3 * 2 = 1234490 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1732_173260


namespace NUMINAMATH_CALUDE_factorial_base_700_a4_l1732_173205

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Coefficient in factorial base representation -/
def factorial_base_coeff (n k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

/-- Theorem: The coefficient a₄ in the factorial base representation of 700 is 4 -/
theorem factorial_base_700_a4 : factorial_base_coeff 700 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_base_700_a4_l1732_173205


namespace NUMINAMATH_CALUDE_three_digit_swap_solution_l1732_173220

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def swap_digits (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  100 * c + 10 * b + a

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_swap_solution :
  ∀ A B : ℕ,
    is_three_digit A →
    is_three_digit B →
    B = swap_digits A →
    A / B = 3 →
    A % B = 7 * sum_of_digits A →
    ((A = 421 ∧ B = 124) ∨ (A = 842 ∧ B = 248)) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_swap_solution_l1732_173220


namespace NUMINAMATH_CALUDE_alpha_integer_and_nonnegative_l1732_173229

theorem alpha_integer_and_nonnegative (α : ℝ) 
  (h : ∀ n : ℕ+, ∃ k : ℤ, (n : ℝ) / α = k) : 
  0 ≤ α ∧ ∃ m : ℤ, α = m := by sorry

end NUMINAMATH_CALUDE_alpha_integer_and_nonnegative_l1732_173229


namespace NUMINAMATH_CALUDE_min_sum_on_unit_circle_l1732_173292

theorem min_sum_on_unit_circle (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = 1) :
  ∃ (m : ℝ), m = Real.sqrt 2 ∧ ∀ (a b : ℝ), 0 < a → 0 < b → a^2 + b^2 = 1 → m ≤ a + b :=
sorry

end NUMINAMATH_CALUDE_min_sum_on_unit_circle_l1732_173292


namespace NUMINAMATH_CALUDE_continuity_at_one_l1732_173223

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / (x^3 - 1)

theorem continuity_at_one :
  ∃ (L : ℝ), ContinuousAt (fun x => if x = 1 then L else f x) 1 ↔ L = 2/3 :=
sorry

end NUMINAMATH_CALUDE_continuity_at_one_l1732_173223


namespace NUMINAMATH_CALUDE_shirts_sold_l1732_173201

/-- The number of shirts sold in a store -/
theorem shirts_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 49 → remaining = 28 → sold = initial - remaining → sold = 21 :=
by sorry

end NUMINAMATH_CALUDE_shirts_sold_l1732_173201


namespace NUMINAMATH_CALUDE_range_of_special_set_l1732_173251

def is_valid_set (a b c : ℝ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ c = 10 ∧ (a + b + c) / 3 = 6 ∧ b = 6

theorem range_of_special_set :
  ∀ a b c : ℝ, is_valid_set a b c → c - a = 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_special_set_l1732_173251


namespace NUMINAMATH_CALUDE_equation_solutions_l1732_173273

theorem equation_solutions (b c : ℝ) : 
  (∀ x : ℝ, (|x - 4| = 3) ↔ (x^2 + b*x + c = 0)) → 
  (b = -8 ∧ c = 7) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1732_173273


namespace NUMINAMATH_CALUDE_unique_solution_iff_in_set_l1732_173202

/-- The set of real numbers m for which the equation 2√(1-m(x+2)) = x+4 has exactly one solution -/
def solution_set : Set ℝ :=
  {m : ℝ | m > -1/2 ∨ m = -1}

/-- The equation 2√(1-m(x+2)) = x+4 -/
def equation (m : ℝ) (x : ℝ) : Prop :=
  2 * Real.sqrt (1 - m * (x + 2)) = x + 4

theorem unique_solution_iff_in_set (m : ℝ) :
  (∃! x, equation m x) ↔ m ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_in_set_l1732_173202


namespace NUMINAMATH_CALUDE_north_distance_calculation_l1732_173276

/-- Represents a trip with distances driven north and west. -/
structure Trip where
  total_distance : ℝ
  distance_west : ℝ
  distance_north : ℝ

/-- Calculates the distance driven north given the total distance and distance driven west. -/
def calculate_north_distance (total : ℝ) (west : ℝ) : ℝ :=
  total - west

/-- Theorem stating that for a trip with a total distance of 150 miles and 95 miles driven west,
    the distance driven north is 55 miles. -/
theorem north_distance_calculation (trip : Trip) 
    (h1 : trip.total_distance = 150)
    (h2 : trip.distance_west = 95)
    (h3 : trip.distance_north = calculate_north_distance trip.total_distance trip.distance_west) :
  trip.distance_north = 55 := by
  sorry

end NUMINAMATH_CALUDE_north_distance_calculation_l1732_173276


namespace NUMINAMATH_CALUDE_problem_1_l1732_173208

theorem problem_1 (x : ℝ) : 4 * x^2 * (x - 1/4) = 4 * x^3 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1732_173208


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l1732_173296

def polynomial (x : ℝ) : ℝ := 3 * (x^5 + 5*x^3 + 2)

theorem sum_of_squares_of_coefficients :
  (3^2 : ℝ) + 0^2 + 15^2 + 0^2 + 0^2 + 6^2 = 270 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l1732_173296


namespace NUMINAMATH_CALUDE_project_work_difference_l1732_173288

/-- Represents the work times of four people on a project -/
structure ProjectWork where
  time1 : ℕ
  time2 : ℕ
  time3 : ℕ
  time4 : ℕ

/-- The total work time of the project -/
def totalTime (pw : ProjectWork) : ℕ :=
  pw.time1 + pw.time2 + pw.time3 + pw.time4

/-- The work times are in the ratio 1:2:3:4 -/
def validRatio (pw : ProjectWork) : Prop :=
  2 * pw.time1 = pw.time2 ∧
  3 * pw.time1 = pw.time3 ∧
  4 * pw.time1 = pw.time4

theorem project_work_difference (pw : ProjectWork) 
  (h1 : totalTime pw = 240)
  (h2 : validRatio pw) :
  pw.time4 - pw.time1 = 72 := by
  sorry

end NUMINAMATH_CALUDE_project_work_difference_l1732_173288


namespace NUMINAMATH_CALUDE_ball_max_height_l1732_173204

/-- The height function of the ball's trajectory -/
def h (t : ℝ) : ℝ := -16 * t^2 + 32 * t + 15

/-- Theorem stating that the maximum height of the ball is 31 feet -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 31 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l1732_173204


namespace NUMINAMATH_CALUDE_tony_squat_weight_l1732_173207

-- Define Tony's lifting capabilities
def curl_weight : ℕ := 90
def military_press_weight : ℕ := 2 * curl_weight
def squat_weight : ℕ := 5 * military_press_weight

-- Theorem to prove
theorem tony_squat_weight : squat_weight = 900 := by
  sorry

end NUMINAMATH_CALUDE_tony_squat_weight_l1732_173207


namespace NUMINAMATH_CALUDE_convex_polygon_as_intersection_of_halfplanes_l1732_173234

-- Define a convex polygon
def ConvexPolygon (P : Set (ℝ × ℝ)) : Prop := sorry

-- Define a half-plane
def HalfPlane (H : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem convex_polygon_as_intersection_of_halfplanes 
  (P : Set (ℝ × ℝ)) (h : ConvexPolygon P) :
  ∃ (n : ℕ) (H : Fin n → Set (ℝ × ℝ)), 
    (∀ i, HalfPlane (H i)) ∧ 
    P = ⋂ i, H i :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_as_intersection_of_halfplanes_l1732_173234


namespace NUMINAMATH_CALUDE_concert_processing_fee_percentage_l1732_173216

theorem concert_processing_fee_percentage
  (ticket_price : ℝ)
  (parking_fee : ℝ)
  (entrance_fee : ℝ)
  (total_cost : ℝ)
  (h1 : ticket_price = 50)
  (h2 : parking_fee = 10)
  (h3 : entrance_fee = 5)
  (h4 : total_cost = 135)
  : (total_cost - (2 * ticket_price + 2 * entrance_fee + parking_fee)) / (2 * ticket_price) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_concert_processing_fee_percentage_l1732_173216


namespace NUMINAMATH_CALUDE_city_mpg_equals_highway_mpg_l1732_173235

/-- The average miles per gallon (mpg) for an SUV on the highway -/
def highway_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 21 gallons of gasoline -/
def max_distance : ℝ := 256.2

/-- The amount of gasoline in gallons used for the maximum distance -/
def gasoline_amount : ℝ := 21

/-- Theorem: The average mpg in the city is equal to the average mpg on the highway -/
theorem city_mpg_equals_highway_mpg :
  max_distance / gasoline_amount = highway_mpg := by
  sorry


end NUMINAMATH_CALUDE_city_mpg_equals_highway_mpg_l1732_173235


namespace NUMINAMATH_CALUDE_hotel_pricing_theorem_l1732_173268

/-- Hotel pricing model -/
structure HotelPricing where
  flatFee : ℝ  -- Flat fee for the first night
  nightlyFee : ℝ  -- Fixed amount for each additional night

/-- Calculate total cost for a stay -/
def totalCost (pricing : HotelPricing) (nights : ℕ) : ℝ :=
  pricing.flatFee + pricing.nightlyFee * (nights - 1)

/-- The hotel pricing theorem -/
theorem hotel_pricing_theorem (pricing : HotelPricing) :
  totalCost pricing 4 = 200 ∧ totalCost pricing 7 = 350 → pricing.flatFee = 50 := by
  sorry

#check hotel_pricing_theorem

end NUMINAMATH_CALUDE_hotel_pricing_theorem_l1732_173268


namespace NUMINAMATH_CALUDE_reciprocal_root_property_l1732_173210

theorem reciprocal_root_property (c : ℝ) : 
  c^3 - c + 1 = 0 → (1/c)^5 + (1/c) + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_root_property_l1732_173210


namespace NUMINAMATH_CALUDE_product_inequality_l1732_173274

theorem product_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_condition : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (z + x) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1732_173274


namespace NUMINAMATH_CALUDE_sufficient_unnecessary_condition_for_hyperbola_l1732_173252

/-- The equation of a conic section -/
def conic_equation (k x y : ℝ) : Prop :=
  x^2 / (k - 2) + y^2 / (5 - k) = 1

/-- Condition for the equation to represent a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  k < 2 ∨ k > 5

/-- Statement that k < 1 is a sufficient and unnecessary condition for a hyperbola -/
theorem sufficient_unnecessary_condition_for_hyperbola :
  (∀ k, k < 1 → is_hyperbola k) ∧
  ∃ k, is_hyperbola k ∧ ¬(k < 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_unnecessary_condition_for_hyperbola_l1732_173252


namespace NUMINAMATH_CALUDE_zoo_trip_money_left_l1732_173264

/-- The amount of money left for lunch and snacks after a zoo trip -/
def money_left_for_lunch_and_snacks (
  zoo_ticket_price : ℚ)
  (bus_fare_one_way : ℚ)
  (total_money : ℚ)
  (num_people : ℕ) : ℚ :=
  total_money - (num_people * zoo_ticket_price + 2 * num_people * bus_fare_one_way)

/-- Theorem: Noah and Ava have $24 left for lunch and snacks after their zoo trip -/
theorem zoo_trip_money_left :
  money_left_for_lunch_and_snacks 5 (3/2) 40 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_money_left_l1732_173264


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l1732_173244

/-- 
Given three terms in the form (10 + x), (40 + x), and (90 + x),
prove that x = 35 is the unique solution for which these terms form a geometric progression.
-/
theorem geometric_progression_solution : 
  ∃! x : ℝ, (∃ r : ℝ, r ≠ 0 ∧ (40 + x) = (10 + x) * r ∧ (90 + x) = (40 + x) * r) ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l1732_173244


namespace NUMINAMATH_CALUDE_circle_radius_exists_l1732_173285

theorem circle_radius_exists : ∃ r : ℝ, r > 0 ∧ π * r^2 + 2 * r - 2 * π * r = 12 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_exists_l1732_173285


namespace NUMINAMATH_CALUDE_cone_volume_relation_l1732_173236

/-- Represents a cone with given dimensions and properties -/
structure Cone where
  r : ℝ  -- base radius
  h : ℝ  -- height
  l : ℝ  -- slant height
  d : ℝ  -- distance from center of base to slant height
  S : ℝ  -- lateral surface area
  V : ℝ  -- volume
  r_pos : 0 < r
  h_pos : 0 < h
  l_pos : 0 < l
  d_pos : 0 < d
  S_pos : 0 < S
  V_pos : 0 < V
  S_eq : S = π * r * l
  V_eq : V = (1/3) * π * r^2 * h

/-- The volume of a cone is one-third of the product of its lateral surface area and the distance from the center of the base to the slant height -/
theorem cone_volume_relation (c : Cone) : c.V = (1/3) * c.d * c.S := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_relation_l1732_173236


namespace NUMINAMATH_CALUDE_alan_collected_48_shells_l1732_173278

/-- Given the number of shells collected by Laurie, calculate the number of shells collected by Alan. -/
def alan_shells (laurie_shells : ℕ) : ℕ :=
  let ben_shells := laurie_shells / 3
  4 * ben_shells

/-- Theorem stating that if Laurie collected 36 shells, Alan collected 48 shells. -/
theorem alan_collected_48_shells :
  alan_shells 36 = 48 := by
  sorry

end NUMINAMATH_CALUDE_alan_collected_48_shells_l1732_173278


namespace NUMINAMATH_CALUDE_cube_collinear_groups_l1732_173238

/-- Represents a point in a cube structure -/
inductive CubePoint
  | Vertex
  | EdgeMidpoint
  | FaceCenter
  | Center

/-- Represents a group of three collinear points in the cube -/
structure CollinearGroup :=
  (points : Fin 3 → CubePoint)

/-- The cube structure with its points -/
structure Cube :=
  (vertices : Fin 8 → CubePoint)
  (edgeMidpoints : Fin 12 → CubePoint)
  (faceCenters : Fin 6 → CubePoint)
  (center : CubePoint)

/-- Function to count collinear groups in the cube -/
def countCollinearGroups (c : Cube) : Nat :=
  sorry

theorem cube_collinear_groups :
  ∀ c : Cube, countCollinearGroups c = 49 :=
sorry

end NUMINAMATH_CALUDE_cube_collinear_groups_l1732_173238


namespace NUMINAMATH_CALUDE_mary_current_books_l1732_173282

/-- Calculates the number of books Mary has checked out after a series of library transactions. -/
def marysBooks (initialBooks : ℕ) (firstReturn : ℕ) (firstCheckout : ℕ) (secondReturn : ℕ) (secondCheckout : ℕ) : ℕ :=
  (((initialBooks - firstReturn) + firstCheckout) - secondReturn) + secondCheckout

/-- Proves that Mary currently has 12 books checked out from the library. -/
theorem mary_current_books :
  marysBooks 5 3 5 2 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_current_books_l1732_173282


namespace NUMINAMATH_CALUDE_exists_divisible_by_two_not_four_l1732_173294

theorem exists_divisible_by_two_not_four : ∃ m : ℕ, (2 ∣ m) ∧ ¬(4 ∣ m) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_two_not_four_l1732_173294


namespace NUMINAMATH_CALUDE_inscribed_rectangle_sides_l1732_173217

/-- A triangle with sides 3, 4, and 5 -/
structure Triangle345 where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 3
  hb : b = 4
  hc : c = 5

/-- A rectangle inscribed in a Triangle345 -/
structure InscribedRectangle (t : Triangle345) where
  short_side : ℝ
  long_side : ℝ
  h_double : long_side = 2 * short_side
  h_inscribed : short_side > 0 ∧ long_side > 0 ∧ long_side ≤ t.c

theorem inscribed_rectangle_sides (t : Triangle345) (r : InscribedRectangle t) :
  r.short_side = 48 / 67 ∧ r.long_side = 96 / 67 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_sides_l1732_173217


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1732_173297

theorem shopkeeper_profit (cost_price : ℝ) : cost_price > 0 →
  let marked_price := cost_price * 1.2
  let selling_price := marked_price * 0.85
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 2 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1732_173297


namespace NUMINAMATH_CALUDE_power_of_product_l1732_173291

theorem power_of_product (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1732_173291


namespace NUMINAMATH_CALUDE_percent_increase_decrease_l1732_173242

theorem percent_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 50) (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) < M ↔ p < 100 * q / (100 - q) := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_decrease_l1732_173242


namespace NUMINAMATH_CALUDE_teacher_problem_l1732_173263

theorem teacher_problem (x : ℤ) : 4 * (3 * (x + 3) - 2) = 4 * (3 * x + 9 - 2) := by
  sorry

#check teacher_problem

end NUMINAMATH_CALUDE_teacher_problem_l1732_173263


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l1732_173262

/-- Calculates the time required for a train to pass a platform -/
theorem train_platform_passing_time
  (train_length : ℝ)
  (tree_passing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 1100)
  (h2 : tree_passing_time = 110)
  (h3 : platform_length = 700) :
  (train_length + platform_length) / (train_length / tree_passing_time) = 180 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l1732_173262


namespace NUMINAMATH_CALUDE_base_to_lateral_area_ratio_l1732_173249

/-- A cone whose lateral surface unfolds into a semicircle -/
structure SemicircleCone where
  r : ℝ  -- radius of the base
  l : ℝ  -- slant height
  h : 2 * π * r = π * l  -- condition for unfolding into a semicircle

theorem base_to_lateral_area_ratio (cone : SemicircleCone) :
  (π * cone.r^2) / ((1/2) * π * cone.l^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_base_to_lateral_area_ratio_l1732_173249


namespace NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_per_day_l1732_173245

/-- Represents the number of blocks in each segment of Ray's walk --/
structure WalkSegments where
  toPark : ℕ
  toHighSchool : ℕ
  toHome : ℕ

/-- Calculates the total blocks walked in one trip --/
def totalBlocksPerTrip (w : WalkSegments) : ℕ :=
  w.toPark + w.toHighSchool + w.toHome

/-- Represents Ray's daily dog walking routine --/
structure DailyWalk where
  segments : WalkSegments
  tripsPerDay : ℕ

/-- Calculates the total blocks walked per day --/
def totalBlocksPerDay (d : DailyWalk) : ℕ :=
  (totalBlocksPerTrip d.segments) * d.tripsPerDay

/-- Theorem stating that Ray's dog walks 66 blocks each day --/
theorem rays_dog_walks_66_blocks_per_day :
  ∀ (d : DailyWalk),
    d.segments.toPark = 4 →
    d.segments.toHighSchool = 7 →
    d.segments.toHome = 11 →
    d.tripsPerDay = 3 →
    totalBlocksPerDay d = 66 :=
by
  sorry


end NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_per_day_l1732_173245


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1732_173270

theorem diophantine_equation_solutions (x y : ℕ+) :
  x^(y : ℕ) = y^(x : ℕ) + 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1732_173270


namespace NUMINAMATH_CALUDE_rowing_current_rate_l1732_173237

/-- Proves that the current rate is 1.1 km/hr given the conditions of the rowing problem -/
theorem rowing_current_rate (man_speed : ℝ) (upstream_time_ratio : ℝ) :
  man_speed = 3.3 →
  upstream_time_ratio = 2 →
  ∃ (current_rate : ℝ),
    current_rate = 1.1 ∧
    (man_speed + current_rate) * upstream_time_ratio = man_speed - current_rate :=
by sorry

end NUMINAMATH_CALUDE_rowing_current_rate_l1732_173237


namespace NUMINAMATH_CALUDE_reflection_and_shift_theorem_l1732_173257

-- Define the transformation properties
def is_reflection_and_shift (f h : ℝ → ℝ) : Prop :=
  ∀ x, h x = f (3 - x)

-- State the theorem
theorem reflection_and_shift_theorem (f h : ℝ → ℝ) 
  (h_def : is_reflection_and_shift f h) : 
  ∀ x, h x = f (3 - x) := by
  sorry

end NUMINAMATH_CALUDE_reflection_and_shift_theorem_l1732_173257


namespace NUMINAMATH_CALUDE_square_difference_theorem_l1732_173259

theorem square_difference_theorem :
  (41 : ℕ)^2 = 40^2 + 81 ∧ 39^2 = 40^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l1732_173259


namespace NUMINAMATH_CALUDE_cube_edge_length_is_15_l1732_173200

/-- The edge length of a cube that displaces a specific volume of water -/
def cube_edge_length (base_length base_width water_rise : ℝ) : ℝ :=
  (base_length * base_width * water_rise) ^ (1/3)

/-- Theorem stating that a cube with the given specifications has an edge length of 15 cm -/
theorem cube_edge_length_is_15 :
  cube_edge_length 20 14 12.053571428571429 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_is_15_l1732_173200


namespace NUMINAMATH_CALUDE_imaginary_part_of_1_minus_i_l1732_173261

theorem imaginary_part_of_1_minus_i :
  let z : ℂ := 1 - I
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_1_minus_i_l1732_173261


namespace NUMINAMATH_CALUDE_cos_theta_equals_three_fifths_l1732_173222

/-- Given that the terminal side of angle θ passes through the point (3, -4), prove that cos θ = 3/5 -/
theorem cos_theta_equals_three_fifths (θ : Real) (h : ∃ (r : Real), r > 0 ∧ r * Real.cos θ = 3 ∧ r * Real.sin θ = -4) : 
  Real.cos θ = 3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_equals_three_fifths_l1732_173222


namespace NUMINAMATH_CALUDE_range_of_a_l1732_173255

-- Define the conditions p and q as functions of x and a
def p (x : ℝ) : Prop := abs (4 * x - 1) ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the property that ¬p is a necessary but not sufficient condition for ¬q
def neg_p_necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ ∃ x, ¬(p x) ∧ (q x a)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, neg_p_necessary_not_sufficient a ↔ -1/2 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1732_173255


namespace NUMINAMATH_CALUDE_garden_internal_boundary_length_l1732_173230

/-- Represents a square plot in the garden -/
structure Plot where
  side : ℕ
  deriving Repr

/-- Represents the garden configuration -/
structure Garden where
  width : ℕ
  height : ℕ
  plots : List Plot
  deriving Repr

/-- Calculates the total area of the garden -/
def gardenArea (g : Garden) : ℕ := g.width * g.height

/-- Calculates the area of a single plot -/
def plotArea (p : Plot) : ℕ := p.side * p.side

/-- Calculates the perimeter of a single plot -/
def plotPerimeter (p : Plot) : ℕ := 4 * p.side

/-- Calculates the external boundary of the garden -/
def externalBoundary (g : Garden) : ℕ := 2 * (g.width + g.height)

/-- The main theorem to prove -/
theorem garden_internal_boundary_length 
  (g : Garden) 
  (h1 : g.width = 6) 
  (h2 : g.height = 7) 
  (h3 : g.plots.length = 5) 
  (h4 : ∀ p ∈ g.plots, ∃ n : ℕ, p.side = n) 
  (h5 : (g.plots.map plotArea).sum = gardenArea g) : 
  ((g.plots.map plotPerimeter).sum - externalBoundary g) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_garden_internal_boundary_length_l1732_173230


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1732_173214

theorem simplify_square_roots : 
  (Real.sqrt 648 / Real.sqrt 81) - (Real.sqrt 245 / Real.sqrt 49) = 2 * Real.sqrt 2 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1732_173214


namespace NUMINAMATH_CALUDE_product_of_powers_l1732_173265

theorem product_of_powers (m : ℝ) : 2 * m^3 * (3 * m^4) = 6 * m^7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_l1732_173265


namespace NUMINAMATH_CALUDE_g_domain_all_reals_l1732_173231

/-- The function g(x) = 1 / ((x-2)^2 + (x+2)^2 + 1) is defined for all real numbers. -/
theorem g_domain_all_reals :
  ∀ x : ℝ, (((x - 2)^2 + (x + 2)^2 + 1) ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_g_domain_all_reals_l1732_173231


namespace NUMINAMATH_CALUDE_spiral_config_399_400_401_l1732_173250

/-- A function representing the spiral number sequence -/
def spiral_sequence : ℕ → ℕ := sorry

/-- Perfect squares are positioned at the center-bottom of their spiral layers -/
axiom perfect_square_position (n : ℕ) :
  ∃ (k : ℕ), k^2 = n → spiral_sequence n = spiral_sequence (n-1) + 1

/-- The vertical configuration of three consecutive numbers -/
def vertical_config (a b c : ℕ) : Prop :=
  spiral_sequence b = spiral_sequence a + 1 ∧
  spiral_sequence c = spiral_sequence b + 1

/-- Theorem stating the configuration of 399, 400, and 401 in the spiral -/
theorem spiral_config_399_400_401 :
  vertical_config 399 400 401 := by sorry

end NUMINAMATH_CALUDE_spiral_config_399_400_401_l1732_173250


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1732_173233

theorem complex_modulus_problem (z : ℂ) (x : ℝ) 
  (h1 : z * Complex.I = 2 * Complex.I + x)
  (h2 : z.im = 2) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1732_173233


namespace NUMINAMATH_CALUDE_bipyramid_volume_l1732_173271

/-- A bipyramid with square bases -/
structure Bipyramid :=
  (side : ℝ)
  (apex_angle : ℝ)

/-- The volume of a bipyramid -/
noncomputable def volume (b : Bipyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of a specific bipyramid -/
theorem bipyramid_volume (b : Bipyramid) (h1 : b.side = 2) (h2 : b.apex_angle = π / 3) :
  volume b = 16 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_bipyramid_volume_l1732_173271


namespace NUMINAMATH_CALUDE_max_inequality_value_zero_is_max_l1732_173289

theorem max_inequality_value (x : ℝ) (h : x > -1) : x + 1 / (x + 1) - 1 ≥ 0 :=
sorry

theorem zero_is_max (a : ℝ) (h : ∀ x > -1, x + 1 / (x + 1) - 1 ≥ a) : a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_max_inequality_value_zero_is_max_l1732_173289


namespace NUMINAMATH_CALUDE_square_1444_product_l1732_173284

theorem square_1444_product (x : ℤ) (h : x^2 = 1444) : (x + 1) * (x - 1) = 1443 := by
  sorry

end NUMINAMATH_CALUDE_square_1444_product_l1732_173284


namespace NUMINAMATH_CALUDE_failed_implies_no_perfect_essay_l1732_173299

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (wrote_perfect_essay : Student → Prop)
variable (passed_course : Student → Prop)

-- Define the given condition
axiom perfect_essay_implies_pass :
  ∀ (s : Student), wrote_perfect_essay s → passed_course s

-- The statement to prove
theorem failed_implies_no_perfect_essay :
  ∀ (s : Student), ¬(passed_course s) → ¬(wrote_perfect_essay s) :=
sorry

end NUMINAMATH_CALUDE_failed_implies_no_perfect_essay_l1732_173299


namespace NUMINAMATH_CALUDE_valid_star_arrangement_exists_l1732_173277

/-- Represents a domino piece with two sides -/
structure Domino :=
  (side1 : Nat)
  (side2 : Nat)

/-- Represents a ray in the star arrangement -/
structure Ray :=
  (pieces : List Domino)
  (length : Nat)
  (sum : Nat)

/-- Represents the center of the star -/
structure Center :=
  (tiles : List Nat)

/-- Represents the entire star arrangement -/
structure StarArrangement :=
  (rays : List Ray)
  (center : Center)

/-- Checks if a domino arrangement is valid according to domino rules -/
def isValidDominoArrangement (arrangement : List Domino) : Prop :=
  sorry

/-- Checks if a ray is valid (correct length and sum) -/
def isValidRay (ray : Ray) : Prop :=
  ray.length ∈ [3, 4] ∧ ray.sum = 21 ∧ isValidDominoArrangement ray.pieces

/-- Checks if the center is valid -/
def isValidCenter (center : Center) : Prop :=
  center.tiles.length = 8 ∧
  (∀ n, n ∈ [1, 2, 3, 4, 5, 6] → n ∈ center.tiles) ∧
  (center.tiles.filter (· = 0)).length = 2

/-- Checks if the entire star arrangement is valid -/
def isValidStarArrangement (star : StarArrangement) : Prop :=
  star.rays.length = 8 ∧
  (∀ ray ∈ star.rays, isValidRay ray) ∧
  isValidCenter star.center

/-- The main theorem stating that a valid star arrangement exists -/
theorem valid_star_arrangement_exists : ∃ (star : StarArrangement), isValidStarArrangement star :=
  sorry

end NUMINAMATH_CALUDE_valid_star_arrangement_exists_l1732_173277


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l1732_173209

/-- A geometric sequence with common ratio r -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_value (a : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  a 4 + a 8 = -2 →
  a 6 * (a 2 + 2 * a 6 + a 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_value_l1732_173209


namespace NUMINAMATH_CALUDE_sqrt_144_div_6_l1732_173280

theorem sqrt_144_div_6 : Real.sqrt 144 / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144_div_6_l1732_173280


namespace NUMINAMATH_CALUDE_right_triangle_angle_A_l1732_173281

theorem right_triangle_angle_A (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : C = Real.pi / 2) (h3 : Real.cos B = Real.sqrt 3 / 2) : A = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_A_l1732_173281


namespace NUMINAMATH_CALUDE_sin_2x_value_l1732_173239

theorem sin_2x_value (x : ℝ) (h : Real.sin (π + x) + Real.cos (π + x) = 1/2) : 
  Real.sin (2 * x) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_2x_value_l1732_173239


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l1732_173269

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + k - 1 ≠ 0) → k > 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l1732_173269


namespace NUMINAMATH_CALUDE_tetrahedron_triangles_l1732_173226

/-- The number of vertices in a regular tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a regular tetrahedron -/
def distinct_triangles : ℕ := Nat.choose tetrahedron_vertices triangle_vertices

theorem tetrahedron_triangles : distinct_triangles = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_triangles_l1732_173226


namespace NUMINAMATH_CALUDE_b_age_l1732_173298

-- Define variables for ages
variable (a b c : ℕ)

-- Define the conditions from the problem
axiom age_relation : a = b + 2
axiom b_twice_c : b = 2 * c
axiom total_age : a + b + c = 27

-- Theorem to prove
theorem b_age : b = 10 := by
  sorry

end NUMINAMATH_CALUDE_b_age_l1732_173298


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1732_173232

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1732_173232


namespace NUMINAMATH_CALUDE_sequence_properties_l1732_173212

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = b n - 1

theorem sequence_properties
  (a b : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : sum_of_terms b S)
  (h_a2b1 : a 2 = b 1)
  (h_a5b2 : a 5 = b 2) :
  (∀ n : ℕ, a n = 2 * n - 6) ∧
  (∀ n : ℕ, S n = (-2)^n - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1732_173212


namespace NUMINAMATH_CALUDE_spatial_diagonals_count_l1732_173211

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of spatial diagonals in a convex polyhedron -/
def spatial_diagonals (P : ConvexPolyhedron) : ℕ :=
  Nat.choose P.vertices 2 - P.edges - 2 * P.quadrilateral_faces

/-- Theorem stating the number of spatial diagonals in the given polyhedron -/
theorem spatial_diagonals_count (P : ConvexPolyhedron) 
  (h1 : P.vertices = 26)
  (h2 : P.edges = 60)
  (h3 : P.faces = 36)
  (h4 : P.triangular_faces = 24)
  (h5 : P.quadrilateral_faces = 12)
  (h6 : P.triangular_faces + P.quadrilateral_faces = P.faces) :
  spatial_diagonals P = 241 := by
  sorry

#eval spatial_diagonals ⟨26, 60, 36, 24, 12⟩

end NUMINAMATH_CALUDE_spatial_diagonals_count_l1732_173211


namespace NUMINAMATH_CALUDE_salt_solution_concentration_l1732_173221

/-- Proves that the concentration of a salt solution is 75% when mixed with pure water to form a 15% solution -/
theorem salt_solution_concentration 
  (water_volume : ℝ) 
  (salt_solution_volume : ℝ) 
  (mixture_concentration : ℝ) 
  (h1 : water_volume = 1) 
  (h2 : salt_solution_volume = 0.25) 
  (h3 : mixture_concentration = 15) : 
  (mixture_concentration * (water_volume + salt_solution_volume)) / salt_solution_volume = 75 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_concentration_l1732_173221


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1732_173224

/-- The curve function f(x) = x^2 - ln x -/
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line function g(x) = x - 2 -/
def g (x : ℝ) : ℝ := x - 2

/-- A point P on the curve -/
structure PointOnCurve where
  x : ℝ
  y : ℝ
  h : y = f x

/-- Theorem: The minimum distance from any point on the curve to the line is 1 -/
theorem min_distance_to_line (P : PointOnCurve) : 
  ∃ (d : ℝ), d = 1 ∧ ∀ (Q : ℝ × ℝ), Q.2 = g Q.1 → Real.sqrt ((P.x - Q.1)^2 + (P.y - Q.2)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1732_173224


namespace NUMINAMATH_CALUDE_system_solution_l1732_173283

theorem system_solution (x y m : ℚ) : 
  x + 3 * y = 7 ∧ 
  x - 3 * y + m * x + 3 = 0 ∧ 
  2 * x - 3 * y = 2 → 
  m = -2/3 := by sorry

end NUMINAMATH_CALUDE_system_solution_l1732_173283


namespace NUMINAMATH_CALUDE_race_track_length_l1732_173213

/-- Represents a runner in the race --/
structure Runner where
  position : ℝ
  velocity : ℝ

/-- Represents the race --/
structure Race where
  track_length : ℝ
  alberto : Runner
  bernardo : Runner
  carlos : Runner

/-- The conditions of the race --/
def race_conditions (r : Race) : Prop :=
  r.alberto.velocity > 0 ∧
  r.bernardo.velocity > 0 ∧
  r.carlos.velocity > 0 ∧
  r.alberto.position = r.track_length ∧
  r.bernardo.position = r.track_length - 36 ∧
  r.carlos.position = r.track_length - 46 ∧
  (r.track_length / r.bernardo.velocity) * r.carlos.velocity = r.track_length - 16

theorem race_track_length (r : Race) (h : race_conditions r) : r.track_length = 96 := by
  sorry

#check race_track_length

end NUMINAMATH_CALUDE_race_track_length_l1732_173213


namespace NUMINAMATH_CALUDE_two_intersection_points_l1732_173275

/-- A line in the plane represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of intersection points between at least two of three given lines -/
def intersection_count (l1 l2 l3 : Line) : ℕ :=
  sorry

/-- The three lines from the problem -/
def line1 : Line := { a := -2, b := 3, c := 1 }
def line2 : Line := { a := 1, b := 2, c := 2 }
def line3 : Line := { a := 4, b := -6, c := 5 }

theorem two_intersection_points : intersection_count line1 line2 line3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_intersection_points_l1732_173275


namespace NUMINAMATH_CALUDE_average_MTWT_is_48_l1732_173279

/-- The average temperature for some days -/
def average_some_days : ℝ := 48

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday -/
def average_TWTF : ℝ := 46

/-- The temperature on Monday -/
def temp_Monday : ℝ := 43

/-- The temperature on Friday -/
def temp_Friday : ℝ := 35

/-- The number of days in the TWTF group -/
def num_days_TWTF : ℕ := 4

/-- The number of days in the MTWT group -/
def num_days_MTWT : ℕ := 4

/-- Theorem: The average temperature for Monday, Tuesday, Wednesday, and Thursday is 48 degrees -/
theorem average_MTWT_is_48 : 
  (temp_Monday + (average_TWTF * num_days_TWTF - temp_Friday)) / num_days_MTWT = average_some_days :=
by sorry

end NUMINAMATH_CALUDE_average_MTWT_is_48_l1732_173279


namespace NUMINAMATH_CALUDE_function_upper_bound_l1732_173219

theorem function_upper_bound 
  (a r : ℝ) 
  (ha : a > 1) 
  (hr : r > 1) 
  (f : ℝ → ℝ) 
  (hf : ∀ x > 0, f x ^ 2 ≤ a * x * f (x / a))
  (hf_small : ∀ x, 0 < x → x < 1 / 2^2005 → f x < 2^2005) :
  ∀ x > 0, f x ≤ a^(1 - r) * x^r := by
sorry

end NUMINAMATH_CALUDE_function_upper_bound_l1732_173219


namespace NUMINAMATH_CALUDE_seating_theorem_standing_theorem_distribution_theorem_l1732_173225

/- Problem 1 -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  sorry

theorem seating_theorem : seating_arrangements 8 3 = 24 := by
  sorry

/- Problem 2 -/
def standing_arrangements (total_people : ℕ) (condition : Bool) : ℕ :=
  sorry

theorem standing_theorem : standing_arrangements 5 true = 60 := by
  sorry

/- Problem 3 -/
def distribute_spots (total_spots : ℕ) (schools : ℕ) : ℕ :=
  sorry

theorem distribution_theorem : distribute_spots 10 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_standing_theorem_distribution_theorem_l1732_173225


namespace NUMINAMATH_CALUDE_linear_dependence_condition_l1732_173203

/-- Two 2D vectors are linearly dependent -/
def linearlyDependent (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), (a, b) ≠ (0, 0) ∧ a • v1 + b • v2 = (0, 0)

/-- The main theorem: vectors (2, 4) and (5, p) are linearly dependent iff p = 10 -/
theorem linear_dependence_condition (p : ℝ) :
  linearlyDependent (2, 4) (5, p) ↔ p = 10 := by
  sorry


end NUMINAMATH_CALUDE_linear_dependence_condition_l1732_173203
