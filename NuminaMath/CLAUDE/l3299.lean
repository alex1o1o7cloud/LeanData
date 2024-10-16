import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_first_grade_l3299_329968

theorem stratified_sampling_first_grade (total_students : ℕ) (sampled_students : ℕ) (first_grade_students : ℕ) :
  total_students = 2400 →
  sampled_students = 100 →
  first_grade_students = 840 →
  (first_grade_students * sampled_students) / total_students = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_first_grade_l3299_329968


namespace NUMINAMATH_CALUDE_function_is_linear_l3299_329919

-- Define the function f and constants a and b
variable (f : ℝ → ℝ) (a b : ℝ)

-- State the theorem
theorem function_is_linear
  (h_continuous : Continuous f)
  (h_a : 0 < a ∧ a < 1/2)
  (h_b : 0 < b ∧ b < 1/2)
  (h_functional : ∀ x, f (f x) = a * f x + b * x) :
  ∃ k : ℝ, ∀ x, f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_function_is_linear_l3299_329919


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_32_plus_sqrt_2_sqrt_27_minus_sqrt_12_div_sqrt_3_sqrt_one_sixth_plus_sqrt_24_minus_sqrt_600_sqrt_3_plus_1_times_sqrt_3_minus_1_l3299_329910

-- (1)
theorem sqrt_18_minus_sqrt_32_plus_sqrt_2 : 
  Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 := by sorry

-- (2)
theorem sqrt_27_minus_sqrt_12_div_sqrt_3 : 
  (Real.sqrt 27 - Real.sqrt 12) / Real.sqrt 3 = 1 := by sorry

-- (3)
theorem sqrt_one_sixth_plus_sqrt_24_minus_sqrt_600 : 
  Real.sqrt (1/6) + Real.sqrt 24 - Real.sqrt 600 = -(43/6) * Real.sqrt 6 := by sorry

-- (4)
theorem sqrt_3_plus_1_times_sqrt_3_minus_1 : 
  (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_32_plus_sqrt_2_sqrt_27_minus_sqrt_12_div_sqrt_3_sqrt_one_sixth_plus_sqrt_24_minus_sqrt_600_sqrt_3_plus_1_times_sqrt_3_minus_1_l3299_329910


namespace NUMINAMATH_CALUDE_inequality_range_l3299_329930

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3299_329930


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l3299_329936

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 38 →
  football = 26 →
  tennis = 20 →
  neither = 9 →
  football + tennis - (total - neither) = 17 := by
sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l3299_329936


namespace NUMINAMATH_CALUDE_parallel_statements_l3299_329923

-- Define the concept of parallel lines
def parallel_lines (l1 l2 : Line) : Prop := sorry

-- Define the concept of parallel planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Define a line being parallel to a plane
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

theorem parallel_statements :
  -- Statement 1
  (∀ l1 l2 l3 : Line, parallel_lines l1 l3 → parallel_lines l2 l3 → parallel_lines l1 l2) ∧
  -- Statement 2
  (∀ p1 p2 p3 : Plane, parallel_planes p1 p3 → parallel_planes p2 p3 → parallel_planes p1 p2) ∧
  -- Statement 3 (negation)
  (∃ l1 l2 : Line, ∃ p : Plane, 
    parallel_lines l1 l2 ∧ line_parallel_to_plane l1 p ∧ ¬line_parallel_to_plane l2 p) ∧
  -- Statement 4 (negation)
  (∃ l : Line, ∃ p1 p2 : Plane,
    parallel_planes p1 p2 ∧ line_parallel_to_plane l p1 ∧ ¬line_parallel_to_plane l p2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_statements_l3299_329923


namespace NUMINAMATH_CALUDE_probability_of_humanities_is_two_thirds_l3299_329917

/-- Represents a school subject -/
inductive Subject
| Mathematics
| Chinese
| Politics
| Geography
| English
| History
| PhysicalEducation

/-- Represents the time of day for a class -/
inductive TimeOfDay
| Morning
| Afternoon

/-- Defines whether a subject is considered a humanities subject -/
def isHumanities (s : Subject) : Bool :=
  match s with
  | Subject.Politics | Subject.History | Subject.Geography => true
  | _ => false

/-- Returns the list of subjects for a given time of day -/
def subjectsForTime (t : TimeOfDay) : List Subject :=
  match t with
  | TimeOfDay.Morning => [Subject.Mathematics, Subject.Chinese, Subject.Politics, Subject.Geography]
  | TimeOfDay.Afternoon => [Subject.English, Subject.History, Subject.PhysicalEducation]

/-- Calculates the probability of selecting at least one humanities class -/
def probabilityOfHumanities : ℚ :=
  let morningSubjects := subjectsForTime TimeOfDay.Morning
  let afternoonSubjects := subjectsForTime TimeOfDay.Afternoon
  let totalCombinations := morningSubjects.length * afternoonSubjects.length
  let humanitiesCombinations := 
    (morningSubjects.filter isHumanities).length * afternoonSubjects.length +
    (morningSubjects.filter (not ∘ isHumanities)).length * (afternoonSubjects.filter isHumanities).length
  humanitiesCombinations / totalCombinations

theorem probability_of_humanities_is_two_thirds :
  probabilityOfHumanities = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_humanities_is_two_thirds_l3299_329917


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3299_329990

/-- Function to replace 2s with 5s and 5s with 2s in a number -/
def replaceDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is a 5-digit odd number -/
def isFiveDigitOdd (n : ℕ) : Prop := sorry

theorem unique_number_satisfying_conditions :
  ∀ x y : ℕ,
    isFiveDigitOdd x →
    y = replaceDigits x →
    y = 2 * (x + 1) →
    x = 29995 := by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3299_329990


namespace NUMINAMATH_CALUDE_equation_solutions_l3299_329939

theorem equation_solutions : 
  {x : ℝ | Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6} = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3299_329939


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l3299_329993

theorem cow_chicken_problem (cows chickens : ℕ) : 
  4 * cows + 2 * chickens = 14 + 2 * (cows + chickens) → cows = 7 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l3299_329993


namespace NUMINAMATH_CALUDE_loss_equals_twenty_pencils_l3299_329969

/-- The number of pencils purchased -/
def num_pencils : ℕ := 70

/-- The ratio of cost to selling price for the total purchase -/
def cost_to_sell_ratio : ℚ := 1.2857142857142856

/-- The number of pencils whose selling price equals the total loss -/
def loss_in_pencils : ℕ := 20

theorem loss_equals_twenty_pencils :
  ∀ (cost_per_pencil sell_per_pencil : ℚ),
  cost_per_pencil = cost_to_sell_ratio * sell_per_pencil →
  (num_pencils : ℚ) * (cost_per_pencil - sell_per_pencil) = (loss_in_pencils : ℚ) * sell_per_pencil :=
by sorry

end NUMINAMATH_CALUDE_loss_equals_twenty_pencils_l3299_329969


namespace NUMINAMATH_CALUDE_solution_set_properties_l3299_329905

def M : Set ℝ := {x : ℝ | 3 - 2*x < 0}

theorem solution_set_properties : (0 ∉ M) ∧ (2 ∈ M) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_properties_l3299_329905


namespace NUMINAMATH_CALUDE_annual_growth_rate_l3299_329949

/-- Given a monthly average growth rate, calculate the annual average growth rate -/
theorem annual_growth_rate (P : ℝ) :
  let monthly_rate := P
  let annual_rate := (1 + P)^12 - 1
  annual_rate = ((1 + monthly_rate)^12 - 1) :=
by sorry

end NUMINAMATH_CALUDE_annual_growth_rate_l3299_329949


namespace NUMINAMATH_CALUDE_marble_color_convergence_l3299_329981

/-- Represents the number of marbles of each color -/
structure MarbleState :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- The total number of marbles -/
def totalMarbles : ℕ := 2015

/-- Possible operations on the marble state -/
inductive MarbleOperation
  | RedGreenToBlue
  | RedBlueToGreen
  | GreenBlueToRed

/-- Apply a marble operation to a state -/
def applyOperation (state : MarbleState) (op : MarbleOperation) : MarbleState :=
  match op with
  | MarbleOperation.RedGreenToBlue => 
      { red := state.red - 1, green := state.green - 1, blue := state.blue + 2 }
  | MarbleOperation.RedBlueToGreen => 
      { red := state.red - 1, green := state.green + 2, blue := state.blue - 1 }
  | MarbleOperation.GreenBlueToRed => 
      { red := state.red + 2, green := state.green - 1, blue := state.blue - 1 }

/-- Check if all marbles are the same color -/
def allSameColor (state : MarbleState) : Prop :=
  (state.red = totalMarbles ∧ state.green = 0 ∧ state.blue = 0) ∨
  (state.red = 0 ∧ state.green = totalMarbles ∧ state.blue = 0) ∨
  (state.red = 0 ∧ state.green = 0 ∧ state.blue = totalMarbles)

/-- The main theorem to prove -/
theorem marble_color_convergence 
  (initial : MarbleState) 
  (h_total : initial.red + initial.green + initial.blue = totalMarbles) :
  ∃ (operations : List MarbleOperation), 
    allSameColor (operations.foldl applyOperation initial) :=
sorry

end NUMINAMATH_CALUDE_marble_color_convergence_l3299_329981


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3299_329914

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3299_329914


namespace NUMINAMATH_CALUDE_yellow_ball_count_l3299_329953

/-- Given a bag with red and yellow balls, this theorem proves the number of yellow balls
    when the number of red balls and the probability of drawing a red ball are known. -/
theorem yellow_ball_count (total : ℕ) (red : ℕ) (p : ℚ) : 
  red = 8 →
  p = 1/3 →
  p = red / total →
  total - red = 16 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_count_l3299_329953


namespace NUMINAMATH_CALUDE_ellipse_equation_l3299_329957

/-- Given an ellipse with the equation x²/a² + y²/b² = 1 where a > b > 0,
    with right focus F, and a line passing through F intersecting the ellipse at A and B,
    if the midpoint of AB is (1, -1/2) and the angle of inclination of AB is 45°,
    then the equation of the ellipse is 2x²/9 + 4y²/9 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (F A B : ℝ × ℝ) (h3 : F.1 > 0) (h4 : F.2 = 0)
  (h5 : (A.1 + B.1) / 2 = 1) (h6 : (A.2 + B.2) / 2 = -1/2)
  (h7 : (B.2 - A.2) / (B.1 - A.1) = 1) :
  ∃ (x y : ℝ), 2 * x^2 / 9 + 4 * y^2 / 9 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3299_329957


namespace NUMINAMATH_CALUDE_prob_third_draw_exactly_l3299_329976

/-- Simple random sampling without replacement from a finite population -/
structure SimpleRandomSampling where
  population_size : ℕ
  sample_size : ℕ
  h_sample_size : sample_size ≤ population_size

/-- The probability of drawing a specific individual on the nth draw,
    given they were not drawn in the previous n-1 draws -/
def prob_draw_on_nth (srs : SimpleRandomSampling) (n : ℕ) : ℚ :=
  if n ≤ srs.sample_size
  then 1 / (srs.population_size - n + 1)
  else 0

/-- The probability of not drawing a specific individual on the nth draw,
    given they were not drawn in the previous n-1 draws -/
def prob_not_draw_on_nth (srs : SimpleRandomSampling) (n : ℕ) : ℚ :=
  if n ≤ srs.sample_size
  then (srs.population_size - n) / (srs.population_size - n + 1)
  else 1

theorem prob_third_draw_exactly
  (srs : SimpleRandomSampling)
  (h : srs.population_size = 6 ∧ srs.sample_size = 3) :
  prob_not_draw_on_nth srs 1 * prob_not_draw_on_nth srs 2 * prob_draw_on_nth srs 3 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_third_draw_exactly_l3299_329976


namespace NUMINAMATH_CALUDE_complement_of_B_l3299_329902

-- Define the set B
def B : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem complement_of_B : 
  (Set.univ : Set ℝ) \ B = {x | x < -2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_l3299_329902


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3299_329942

theorem quadratic_equation_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ↔ 
  m < -2 ∨ m > 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3299_329942


namespace NUMINAMATH_CALUDE_angle_measure_l3299_329918

theorem angle_measure : ∃ (x : ℝ), 
  (180 - x = 7 * (90 - x)) ∧ 
  (0 < x) ∧ 
  (x < 180) ∧ 
  (x = 75) := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3299_329918


namespace NUMINAMATH_CALUDE_clothing_tax_rate_l3299_329935

theorem clothing_tax_rate
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (other_tax_rate : ℝ)
  (total_tax_rate : ℝ)
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.2)
  (h3 : other_percent = 0.3)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : other_tax_rate = 0.1)
  (h6 : total_tax_rate = 0.055) :
  ∃ (clothing_tax_rate : ℝ),
    clothing_tax_rate * clothing_percent + other_tax_rate * other_percent = total_tax_rate ∧
    clothing_tax_rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_clothing_tax_rate_l3299_329935


namespace NUMINAMATH_CALUDE_circle_configuration_radius_l3299_329909

/-- Given a configuration of three circles C, D, and E, prove that the radius of circle D is 4√15 - 14 -/
theorem circle_configuration_radius (C D E : ℝ → ℝ → Prop) (A B F : ℝ × ℝ) :
  (∀ x y, C x y ↔ (x - 0)^2 + (y - 0)^2 = 4) →  -- Circle C with radius 2 centered at origin
  (∃ x y, C x y ∧ D x y) →  -- D is internally tangent to C
  (∃ x y, C x y ∧ E x y) →  -- E is tangent to C
  (∃ x y, D x y ∧ E x y) →  -- E is externally tangent to D
  (∃ t, 0 ≤ t ∧ t ≤ 1 ∧ F = (2*t - 1, 0) ∧ E (2*t - 1) 0) →  -- E is tangent to AB at F
  (∀ x y z w, D x y ∧ E z w → (x - z)^2 + (y - w)^2 = (3*r)^2 - r^2) →  -- Radius of D is 3 times radius of E
  (∃ r_D, ∀ x y, D x y ↔ (x - 0)^2 + (y - 0)^2 = r_D^2 ∧ r_D = 4*Real.sqrt 15 - 14) :=
sorry

end NUMINAMATH_CALUDE_circle_configuration_radius_l3299_329909


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l3299_329997

theorem tan_double_angle_special_case (x : ℝ) 
  (h : Real.sin x - 3 * Real.cos x = Real.sqrt 5) : 
  Real.tan (2 * x) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l3299_329997


namespace NUMINAMATH_CALUDE_probability_problems_l3299_329995

def bag : Finset ℕ := {1, 2, 3, 4, 5, 6}

def isPrime (n : ℕ) : Prop := Nat.Prime n

def sumIs6 (a b : ℕ) : Prop := a + b = 6

theorem probability_problems :
  (∃ (S : Finset ℕ), S ⊆ bag ∧ (∀ n ∈ S, isPrime n) ∧ S.card / bag.card = 1 / 2) ∧
  (∃ (T : Finset (ℕ × ℕ)), T ⊆ bag.product bag ∧ 
    (∀ p ∈ T, sumIs6 p.1 p.2) ∧ 
    T.card / (bag.card * bag.card) = 5 / 36) := by
  sorry

end NUMINAMATH_CALUDE_probability_problems_l3299_329995


namespace NUMINAMATH_CALUDE_quadratic_shift_and_roots_l3299_329980

/-- A quadratic function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_shift_and_roots (a b c : ℝ) (h : a > 0) :
  (∀ k > 0, ∀ x, quadratic a b (c - k) x < quadratic a b c x) ∧
  (∀ x, quadratic a b c x ≠ 0 →
    ∃ k > 0, ∃ x, quadratic a b (c - k) x = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_and_roots_l3299_329980


namespace NUMINAMATH_CALUDE_price_decrease_l3299_329933

theorem price_decrease (original_price : ℝ) : 
  (original_price * (1 - 0.24) = 836) → original_price = 1100 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_l3299_329933


namespace NUMINAMATH_CALUDE_house_sale_profit_l3299_329934

theorem house_sale_profit (market_value : ℝ) (over_market_percentage : ℝ) 
  (tax_rate : ℝ) (num_people : ℕ) : 
  market_value = 500000 ∧ 
  over_market_percentage = 0.2 ∧ 
  tax_rate = 0.1 ∧ 
  num_people = 4 → 
  (market_value * (1 + over_market_percentage) * (1 - tax_rate)) / num_people = 135000 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_profit_l3299_329934


namespace NUMINAMATH_CALUDE_inequality_proof_l3299_329911

theorem inequality_proof (a b c : ℝ) : a^2 + 4*b^2 + 8*c^2 - 3*a*b - 4*b*c - 2*c*a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3299_329911


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3299_329999

theorem simplify_and_evaluate (a b : ℝ) (h : a = -b) :
  2 * (3 * a^2 + a - 2*b) - 6 * (a^2 - b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3299_329999


namespace NUMINAMATH_CALUDE_min_distance_theorem_l3299_329913

/-- Given a line segment AB of length 2 with midpoint C, where A moves on the x-axis and B moves on the y-axis. -/
def line_segment (A B C : ℝ × ℝ) : Prop :=
  norm (A - B) = 2 ∧ C = (A + B) / 2 ∧ A.2 = 0 ∧ B.1 = 0

/-- The trajectory of point C is a circle with equation x² + y² = 1 -/
def trajectory (C : ℝ × ℝ) : Prop :=
  C.1^2 + C.2^2 = 1

/-- The line √2ax + by = 1 intersects the trajectory at points C and D -/
def intersecting_line (a b : ℝ) (C D : ℝ × ℝ) : Prop :=
  trajectory C ∧ trajectory D ∧ 
  Real.sqrt 2 * a * C.1 + b * C.2 = 1 ∧
  Real.sqrt 2 * a * D.1 + b * D.2 = 1

/-- Triangle COD is a right-angled triangle with O as the origin -/
def right_triangle (C D : ℝ × ℝ) : Prop :=
  (C.1 * D.1 + C.2 * D.2) = 0

/-- Point P has coordinates (a, b) -/
def point_P (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  P = (a, b)

/-- The main theorem: The minimum distance between P(a, b) and (0, 1) is √2 - 1 -/
theorem min_distance_theorem (A B C D P : ℝ × ℝ) (a b : ℝ) :
  line_segment A B C →
  trajectory C →
  intersecting_line a b C D →
  right_triangle C D →
  point_P P a b →
  (∃ (min_dist : ℝ), ∀ (a' b' : ℝ), 
    norm ((a', b') - (0, 1)) ≥ min_dist ∧
    min_dist = Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_min_distance_theorem_l3299_329913


namespace NUMINAMATH_CALUDE_score_ordering_l3299_329987

structure Participant where
  score : ℕ

def Leonard : Participant := sorry
def Nina : Participant := sorry
def Oscar : Participant := sorry
def Paula : Participant := sorry

theorem score_ordering :
  (Oscar.score = Leonard.score) →
  (Nina.score < max Oscar.score Paula.score) →
  (Paula.score > Leonard.score) →
  (Oscar.score < Nina.score) ∧ (Nina.score < Paula.score) := by
  sorry

end NUMINAMATH_CALUDE_score_ordering_l3299_329987


namespace NUMINAMATH_CALUDE_coordinate_translation_l3299_329977

/-- Given a translation of the coordinate system where point A moves from (-1, 3) to (-3, -1),
    prove that the new origin O' has coordinates (2, 4). -/
theorem coordinate_translation (A_old A_new O'_new : ℝ × ℝ) : 
  A_old = (-1, 3) → A_new = (-3, -1) → O'_new = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_coordinate_translation_l3299_329977


namespace NUMINAMATH_CALUDE_second_smallest_three_digit_in_pascal_l3299_329982

/-- Pascal's Triangle coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Checks if a number is three digits -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- The smallest three-digit number in Pascal's Triangle -/
def smallestThreeDigit : ℕ := 100

/-- The row where the smallest three-digit number first appears -/
def smallestThreeDigitRow : ℕ := 100

theorem second_smallest_three_digit_in_pascal :
  ∃ (n : ℕ), isThreeDigit n ∧
    (∀ (m : ℕ), isThreeDigit m → m < n → m = smallestThreeDigit) ∧
    (∃ (row : ℕ), binomial row 1 = n ∧
      ∀ (r : ℕ), r < row → ¬(∃ (k : ℕ), isThreeDigit (binomial r k) ∧ binomial r k = n)) ∧
    n = 101 ∧ row = 101 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_three_digit_in_pascal_l3299_329982


namespace NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l3299_329900

theorem longest_segment_in_quarter_circle (d : ℝ) (h : d = 18) :
  let r := d / 2
  let m := (2 * r^2)^(1/2)
  m^2 = 162 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l3299_329900


namespace NUMINAMATH_CALUDE_min_nSn_l3299_329983

/-- Represents an arithmetic sequence and its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  m : ℕ      -- Given index
  h_m : m ≥ 2
  h_sum_pred : S (m - 1) = -2
  h_sum : S m = 0
  h_sum_succ : S (m + 1) = 3

/-- The minimum value of nS_n for the given arithmetic sequence -/
theorem min_nSn (seq : ArithmeticSequence) : 
  ∃ (k : ℝ), k = -9 ∧ ∀ (n : ℕ), n * seq.S n ≥ k :=
sorry

end NUMINAMATH_CALUDE_min_nSn_l3299_329983


namespace NUMINAMATH_CALUDE_prime_n_l3299_329973

theorem prime_n (p h n : ℕ) : 
  Nat.Prime p → 
  h < p → 
  n = p * h + 1 → 
  (2^(n-1) - 1) % n = 0 → 
  (2^h - 1) % n ≠ 0 → 
  Nat.Prime n := by
sorry

end NUMINAMATH_CALUDE_prime_n_l3299_329973


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3299_329951

theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + a*x + 81 = (x + b)^2) → (a = 18 ∨ a = -18) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3299_329951


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l3299_329916

theorem stamp_collection_problem (C K A : ℕ) : 
  C > 2 * K ∧ 
  K = A / 2 ∧ 
  C + K + A = 930 ∧ 
  A = 370 → 
  C - 2 * K = 5 := by
sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l3299_329916


namespace NUMINAMATH_CALUDE_journalist_selection_theorem_l3299_329964

-- Define the number of domestic and foreign journalists
def domestic_journalists : ℕ := 5
def foreign_journalists : ℕ := 4

-- Define the total number of journalists to be selected
def selected_journalists : ℕ := 3

-- Function to calculate the number of ways to select and arrange journalists
def select_and_arrange_journalists : ℕ := sorry

-- Theorem stating the correct number of ways
theorem journalist_selection_theorem : 
  select_and_arrange_journalists = 260 := by sorry

end NUMINAMATH_CALUDE_journalist_selection_theorem_l3299_329964


namespace NUMINAMATH_CALUDE_crate_middle_dimension_l3299_329941

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder fits upright in a crate -/
def cylinderFitsUpright (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.height ≤ crate.width) ∨
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.height ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.height ≤ crate.length) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.height ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.length) ∨
  (cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.width)

theorem crate_middle_dimension (x : ℝ) :
  let crate := CrateDimensions.mk 5 x 12
  let cylinder := Cylinder.mk 5 12
  cylinderFitsUpright crate cylinder → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_crate_middle_dimension_l3299_329941


namespace NUMINAMATH_CALUDE_solution_added_mass_l3299_329944

/-- Represents the composition and manipulation of a solution --/
structure Solution :=
  (total_mass : ℝ)
  (liquid_x_percentage : ℝ)

/-- Calculates the mass of liquid x in a solution --/
def liquid_x_mass (s : Solution) : ℝ :=
  s.total_mass * s.liquid_x_percentage

/-- Represents the problem scenario --/
def solution_problem (initial_solution : Solution) 
  (evaporated_water : ℝ) (added_solution : Solution) : Prop :=
  let remaining_solution : Solution := {
    total_mass := initial_solution.total_mass - evaporated_water,
    liquid_x_percentage := 
      liquid_x_mass initial_solution / (initial_solution.total_mass - evaporated_water)
  }
  let final_solution : Solution := {
    total_mass := remaining_solution.total_mass + added_solution.total_mass,
    liquid_x_percentage := 0.4
  }
  liquid_x_mass remaining_solution + liquid_x_mass added_solution = 
    liquid_x_mass final_solution

/-- The theorem to be proved --/
theorem solution_added_mass : 
  let initial_solution : Solution := { total_mass := 6, liquid_x_percentage := 0.3 }
  let evaporated_water : ℝ := 2
  let added_solution : Solution := { total_mass := 2, liquid_x_percentage := 0.3 }
  solution_problem initial_solution evaporated_water added_solution := by
  sorry

end NUMINAMATH_CALUDE_solution_added_mass_l3299_329944


namespace NUMINAMATH_CALUDE_reflected_ray_tangent_to_circle_l3299_329906

-- Define the initial ray of light
def initial_ray (x y : ℝ) : Prop := x + 2*y + 2 + Real.sqrt 5 = 0 ∧ y ≥ 0

-- Define the x-axis
def x_axis (y : ℝ) : Prop := y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 2)

-- Define a function to check if a point is on the circle
def on_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

-- Theorem statement
theorem reflected_ray_tangent_to_circle :
  ∃ (x y : ℝ), initial_ray x y ∧ 
               x_axis y ∧
               on_circle x y ∧
               ∀ (x' y' : ℝ), on_circle x' y' → 
                 ((x' - x)^2 + (y' - y)^2 ≥ 1 ∨ (x' = x ∧ y' = y)) :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_tangent_to_circle_l3299_329906


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l3299_329948

theorem aquarium_fish_count (total : ℕ) : 
  (total : ℚ) / 3 = 60 ∧  -- One third of fish are blue
  (total : ℚ) / 4 ≤ (total : ℚ) / 3 ∧  -- One fourth of fish are yellow
  (total : ℚ) - ((total : ℚ) / 3 + (total : ℚ) / 4) = 45 ∧  -- The rest are red
  (60 : ℚ) / 2 = 30 ∧  -- 50% of blue fish have spots
  30 * (100 : ℚ) / 60 = 50 ∧  -- Verify 50% of blue fish have spots
  9 * (100 : ℚ) / 45 = 20  -- Verify 20% of red fish have spots
  → total = 140 := by
sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l3299_329948


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_negative_l3299_329966

theorem consecutive_integers_sum_negative : ∃ n : ℤ, 
  (n^2 - 13*n + 36) + ((n+1)^2 - 13*(n+1) + 36) < 0 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_negative_l3299_329966


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l3299_329991

/-- Reflects a point over the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- Calculates the area of a triangle given three points -/
def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  let (x1, y1) := a
  let (x2, y2) := b
  let (x3, y3) := c
  0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC' :
  let A : ℝ × ℝ := (3, 4)
  let B' := reflect_y_axis A
  let C' := reflect_y_eq_x B'
  triangle_area A B' C' = 21 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l3299_329991


namespace NUMINAMATH_CALUDE_root_comparison_l3299_329978

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem root_comparison (x₀ : ℝ) (hx₀ : f x₀ = 0) :
  Real.log (Real.log x₀) < Real.log (Real.sqrt x₀) ∧
  Real.log (Real.sqrt x₀) < Real.log x₀ ∧
  Real.log x₀ < (Real.log x₀)^2 := by
  sorry

end NUMINAMATH_CALUDE_root_comparison_l3299_329978


namespace NUMINAMATH_CALUDE_square_with_seven_in_tens_place_l3299_329975

theorem square_with_seven_in_tens_place (a : ℕ) (b : Fin 10) :
  ∃ k : ℕ, ((10 * a + b) ^ 2) % 100 = 70 + k ∧ k < 10 →
  (b = 4 ∨ b = 6) := by
sorry

end NUMINAMATH_CALUDE_square_with_seven_in_tens_place_l3299_329975


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3299_329989

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = -5 ∧ ∀ x, x^2 + 2*x - 4 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3299_329989


namespace NUMINAMATH_CALUDE_percentage_of_women_in_parent_group_l3299_329907

theorem percentage_of_women_in_parent_group (women_fulltime : Real) 
  (men_fulltime : Real) (total_not_fulltime : Real) :
  women_fulltime = 0.9 →
  men_fulltime = 0.75 →
  total_not_fulltime = 0.19 →
  ∃ (w : Real), w ≥ 0 ∧ w ≤ 1 ∧
    w * (1 - women_fulltime) + (1 - w) * (1 - men_fulltime) = total_not_fulltime ∧
    w = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_parent_group_l3299_329907


namespace NUMINAMATH_CALUDE_equal_money_after_transfer_l3299_329901

/-- Represents the amount of gold coins each merchant has -/
structure Merchants where
  foma : ℤ
  ierema : ℤ
  yuliy : ℤ

/-- The conditions of the problem -/
def satisfies_conditions (m : Merchants) : Prop :=
  (m.ierema + 70 = m.yuliy) ∧ (m.foma - 40 = m.yuliy)

/-- The theorem to prove -/
theorem equal_money_after_transfer (m : Merchants) 
  (h : satisfies_conditions m) : 
  m.foma - 55 = m.ierema + 55 := by
  sorry

#check equal_money_after_transfer

end NUMINAMATH_CALUDE_equal_money_after_transfer_l3299_329901


namespace NUMINAMATH_CALUDE_m_intersect_n_equals_open_one_closed_three_l3299_329959

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 9}

-- State the theorem
theorem m_intersect_n_equals_open_one_closed_three : M ∩ N = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_m_intersect_n_equals_open_one_closed_three_l3299_329959


namespace NUMINAMATH_CALUDE_triangle_existence_l3299_329932

theorem triangle_existence (x : ℝ) (h : x > 1) :
  let a := x^4 + x^3 + 2*x^2 + x + 1
  let b := 2*x^3 + x^2 + 2*x + 1
  let c := x^4 - 1
  (a > c) ∧ (a > b) ∧ (a < b + c) := by sorry

end NUMINAMATH_CALUDE_triangle_existence_l3299_329932


namespace NUMINAMATH_CALUDE_union_dues_deduction_l3299_329996

def weekly_hours : ℕ := 42
def hourly_rate : ℚ := 10
def tax_rate : ℚ := 1/5
def insurance_rate : ℚ := 1/20
def take_home_pay : ℚ := 310

theorem union_dues_deduction :
  let gross_earnings := weekly_hours * hourly_rate
  let tax_deduction := tax_rate * gross_earnings
  let insurance_deduction := insurance_rate * gross_earnings
  let total_deductions := tax_deduction + insurance_deduction
  let net_before_union := gross_earnings - total_deductions
  net_before_union - take_home_pay = 5 := by sorry

end NUMINAMATH_CALUDE_union_dues_deduction_l3299_329996


namespace NUMINAMATH_CALUDE_vector_b_values_l3299_329985

/-- Given two vectors a and b in ℝ², where a = (2,1), |b| = 2√5, and a is parallel to b,
    prove that b is either (-4,-2) or (4,2) -/
theorem vector_b_values (a b : ℝ × ℝ) : 
  a = (2, 1) → 
  ‖b‖ = 2 * Real.sqrt 5 →
  ∃ (k : ℝ), b = k • a →
  b = (-4, -2) ∨ b = (4, 2) := by
sorry

end NUMINAMATH_CALUDE_vector_b_values_l3299_329985


namespace NUMINAMATH_CALUDE_quadratic_roots_value_l3299_329940

theorem quadratic_roots_value (x₁ x₂ m : ℝ) : 
  (∀ x, x^2 - 8*x + m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ = 3*x₂ →
  m = 12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_value_l3299_329940


namespace NUMINAMATH_CALUDE_cubic_function_equality_l3299_329915

/-- Given two cubic functions f and g, prove that f(g(x)) = g(f(x)) for all x if and only if d = ±a -/
theorem cubic_function_equality (a b c d e f : ℝ) :
  (∀ x : ℝ, (a * (d * x^3 + e * x + f)^3 + b * (d * x^3 + e * x + f) + c) = 
            (d * (a * x^3 + b * x + c)^3 + e * (a * x^3 + b * x + c) + f)) ↔ 
  (d = a ∨ d = -a) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_equality_l3299_329915


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3299_329938

theorem line_passes_through_point (a : ℝ) : (a + 2) * 1 + a * (-1) - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3299_329938


namespace NUMINAMATH_CALUDE_negative_six_divided_by_three_l3299_329946

theorem negative_six_divided_by_three : (-6) / 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_six_divided_by_three_l3299_329946


namespace NUMINAMATH_CALUDE_quadrilaterals_with_same_lengths_not_necessarily_congruent_l3299_329956

/-- Represents a convex quadrilateral -/
structure ConvexQuadrilateral where
  -- Define the necessary properties of a convex quadrilateral
  -- (We don't need to fully define it for this statement)

/-- Returns a list of side lengths and diagonal lengths of a quadrilateral in ascending order -/
def lengthsList (q : ConvexQuadrilateral) : List ℝ :=
  sorry -- Implementation details not needed for the statement

/-- Two quadrilaterals are congruent if they have the same shape and size -/
def areCongruent (q1 q2 : ConvexQuadrilateral) : Prop :=
  sorry -- Implementation details not needed for the statement

theorem quadrilaterals_with_same_lengths_not_necessarily_congruent :
  ∃ (q1 q2 : ConvexQuadrilateral),
    lengthsList q1 = lengthsList q2 ∧ ¬(areCongruent q1 q2) :=
  sorry

end NUMINAMATH_CALUDE_quadrilaterals_with_same_lengths_not_necessarily_congruent_l3299_329956


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3299_329988

theorem quadratic_roots_sum_of_squares (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - m*x + (2*m - 1) = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁^2 + x₂^2 = 7 →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3299_329988


namespace NUMINAMATH_CALUDE_circle_center_sum_l3299_329955

/-- Given a circle with equation x^2 + y^2 = 6x + 8y + 2, 
    the sum of the coordinates of its center is 7. -/
theorem circle_center_sum : ∃ (h k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 6*x + 8*y + 2 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 2)) ∧
  h + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3299_329955


namespace NUMINAMATH_CALUDE_sqrt_equation_implies_difference_l3299_329958

theorem sqrt_equation_implies_difference (m n : ℕ) : 
  (Real.sqrt (9 - m / n) = 9 * Real.sqrt (m / n)) → (n - m = 73) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_implies_difference_l3299_329958


namespace NUMINAMATH_CALUDE_football_field_area_l3299_329994

-- Define the football field and fertilizer properties
def total_fertilizer : ℝ := 800
def partial_fertilizer : ℝ := 300
def partial_area : ℝ := 3600

-- Define the theorem
theorem football_field_area :
  (total_fertilizer * partial_area) / partial_fertilizer = 9600 := by
  sorry

end NUMINAMATH_CALUDE_football_field_area_l3299_329994


namespace NUMINAMATH_CALUDE_pond_draining_time_l3299_329945

theorem pond_draining_time 
  (pump1_half_time : ℝ) 
  (pump2_full_time : ℝ) 
  (combined_half_time : ℝ) 
  (h1 : pump2_full_time = 1.25) 
  (h2 : combined_half_time = 0.5) :
  pump1_half_time = 5/12 := by
sorry

end NUMINAMATH_CALUDE_pond_draining_time_l3299_329945


namespace NUMINAMATH_CALUDE_min_value_of_function_l3299_329947

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3299_329947


namespace NUMINAMATH_CALUDE_paper_strip_dimensions_l3299_329924

theorem paper_strip_dimensions (a b c : ℕ+) (h : 2 * a * b + 2 * a * c - a * a = 43) :
  a = 1 ∧ b + c = 22 := by
  sorry

end NUMINAMATH_CALUDE_paper_strip_dimensions_l3299_329924


namespace NUMINAMATH_CALUDE_exterior_angle_decreases_l3299_329984

theorem exterior_angle_decreases (n : ℕ) (h : n > 2) :
  (360 : ℝ) / (n + 1) < 360 / n := by
sorry

end NUMINAMATH_CALUDE_exterior_angle_decreases_l3299_329984


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_negative_a_abs_function_solutions_l3299_329912

-- Define the quadratic equation
def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 3) * x + a

-- Define the absolute value function
def abs_function (x : ℝ) : ℝ := |3 - x^2|

theorem quadratic_roots_imply_negative_a (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ quadratic a x = 0 ∧ quadratic a y = 0) → a < 0 :=
sorry

theorem abs_function_solutions (a : ℝ) :
  ¬(∃! x : ℝ, abs_function x = a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_negative_a_abs_function_solutions_l3299_329912


namespace NUMINAMATH_CALUDE_meeting_selection_ways_l3299_329967

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of managers -/
def total_managers : ℕ := 7

/-- The number of managers needed for the meeting -/
def meeting_size : ℕ := 4

/-- The number of managers who cannot attend together -/
def incompatible_managers : ℕ := 2

/-- The number of ways to select managers for the meeting -/
def select_managers : ℕ :=
  choose (total_managers - incompatible_managers) meeting_size +
  incompatible_managers * choose (total_managers - 1) (meeting_size - 1)

theorem meeting_selection_ways :
  select_managers = 25 := by sorry

end NUMINAMATH_CALUDE_meeting_selection_ways_l3299_329967


namespace NUMINAMATH_CALUDE_line_circle_min_value_l3299_329954

/-- Given a line ax + by + 1 = 0 that divides a circle into two equal areas, 
    prove that the minimum value of 1/(2a) + 2/b is 8 -/
theorem line_circle_min_value (a b : ℝ) : 
  a > 0 → 
  b > 0 → 
  (∀ x y : ℝ, a * x + b * y + 1 = 0 → (x + 4)^2 + (y + 1)^2 = 16 → 
    (∃ k : ℝ, k > 0 ∧ k * ((x + 4)^2 + (y + 1)^2) = 16 ∧ 
    k * (a * x + b * y + 1) = 0)) → 
  (∀ x y : ℝ, (1 / (2 * a) + 2 / b) ≥ 8) ∧ 
  (∃ x y : ℝ, 1 / (2 * a) + 2 / b = 8) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_min_value_l3299_329954


namespace NUMINAMATH_CALUDE_quadratic_polynomial_remainders_l3299_329922

theorem quadratic_polynomial_remainders (m n : ℚ) : 
  (∀ x, (x^2 + m*x + n) % (x - m) = m ∧ (x^2 + m*x + n) % (x - n) = n) ↔ 
  ((m = 0 ∧ n = 0) ∨ (m = 1/2 ∧ n = 0) ∨ (m = 1 ∧ n = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_remainders_l3299_329922


namespace NUMINAMATH_CALUDE_age_difference_is_32_l3299_329998

/-- The age difference between Mrs Bai and her daughter Jenni -/
def age_difference : ℕ :=
  let jenni_age : ℕ := 19
  let sum_of_ages : ℕ := 70
  sum_of_ages - 2 * jenni_age

/-- Theorem stating that the age difference between Mrs Bai and Jenni is 32 years -/
theorem age_difference_is_32 : age_difference = 32 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_32_l3299_329998


namespace NUMINAMATH_CALUDE_inverse_of_two_mod_185_l3299_329979

theorem inverse_of_two_mod_185 : Int.ModEq 1 185 (2 * 93) := by sorry

end NUMINAMATH_CALUDE_inverse_of_two_mod_185_l3299_329979


namespace NUMINAMATH_CALUDE_bananas_in_basket_E_l3299_329926

def num_baskets : ℕ := 5
def average_fruits_per_basket : ℕ := 25
def fruits_in_basket_A : ℕ := 15
def fruits_in_basket_B : ℕ := 30
def fruits_in_basket_C : ℕ := 20
def fruits_in_basket_D : ℕ := 25

theorem bananas_in_basket_E : 
  (num_baskets * average_fruits_per_basket) - 
  (fruits_in_basket_A + fruits_in_basket_B + fruits_in_basket_C + fruits_in_basket_D) = 35 := by
  sorry

end NUMINAMATH_CALUDE_bananas_in_basket_E_l3299_329926


namespace NUMINAMATH_CALUDE_coin_probability_l3299_329908

theorem coin_probability (p q : ℝ) : 
  q = 1 - p →
  (Nat.choose 10 5 : ℝ) * p^5 * q^5 = (Nat.choose 10 6 : ℝ) * p^6 * q^4 →
  p = 6/11 := by
sorry

end NUMINAMATH_CALUDE_coin_probability_l3299_329908


namespace NUMINAMATH_CALUDE_smallest_cube_ending_368_l3299_329965

theorem smallest_cube_ending_368 :
  ∀ n : ℕ+, n.val^3 ≡ 368 [MOD 1000] → n.val ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_368_l3299_329965


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l3299_329929

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def units_digit (n : ℕ) : ℕ := n % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

theorem unique_three_digit_divisible_by_11 :
  ∃! n : ℕ, is_three_digit n ∧ 
             units_digit n = 5 ∧ 
             hundreds_digit n = 6 ∧ 
             n % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l3299_329929


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l3299_329903

/-- Represents the number of eggs found by each group or individual -/
structure EggCounts where
  kevin : ℕ
  someChildren : ℕ
  george : ℕ
  cheryl : ℕ

/-- The Easter egg hunt problem -/
theorem easter_egg_hunt (counts : EggCounts) 
  (h1 : counts.kevin = 5)
  (h2 : counts.george = 9)
  (h3 : counts.cheryl = 56)
  (h4 : counts.cheryl = counts.kevin + counts.someChildren + counts.george + 29) :
  counts.someChildren = 13 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l3299_329903


namespace NUMINAMATH_CALUDE_regular_polygon_with_160_degree_angles_l3299_329972

theorem regular_polygon_with_160_degree_angles (n : ℕ) : 
  (n ≥ 3) →  -- A polygon must have at least 3 sides
  (∀ i : ℕ, i < n → 160 = (n - 2) * 180 / n) →  -- Each interior angle is 160°
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_160_degree_angles_l3299_329972


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3299_329960

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3299_329960


namespace NUMINAMATH_CALUDE_polynomial_identity_l3299_329970

theorem polynomial_identity (x : ℝ) : 
  (x - 2)^4 + 5*(x - 2)^3 + 10*(x - 2)^2 + 10*(x - 2) + 5 = (x - 2 + Real.sqrt 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3299_329970


namespace NUMINAMATH_CALUDE_jacob_pencils_l3299_329974

theorem jacob_pencils (total : ℕ) (zain_monday : ℕ) (zain_tuesday : ℕ) : 
  total = 21 →
  zain_monday + zain_tuesday + (2 * zain_monday + zain_tuesday) / 3 = total →
  (2 * zain_monday + zain_tuesday) / 3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_jacob_pencils_l3299_329974


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_5000_l3299_329986

theorem greatest_multiple_of_four_cubed_less_than_5000 :
  ∀ x : ℕ, x > 0 → x % 4 = 0 → x^3 < 5000 → x ≤ 16 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_5000_l3299_329986


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_value_l3299_329937

theorem quadratic_root_implies_u_value (u : ℝ) : 
  (3 * (((-15 - Real.sqrt 205) / 6) ^ 2) + 15 * ((-15 - Real.sqrt 205) / 6) + u = 0) → 
  u = 5/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_value_l3299_329937


namespace NUMINAMATH_CALUDE_knight_same_color_probability_l3299_329962

/-- Represents the colors of the chessboard squares -/
inductive ChessColor
| Red
| Green
| Blue

/-- Represents a square on the chessboard -/
structure ChessSquare where
  row : Fin 8
  col : Fin 8
  color : ChessColor

/-- The chessboard with its colored squares -/
def chessboard : Array ChessSquare := sorry

/-- Determines if a knight's move is legal -/
def isLegalKnightMove (start finish : ChessSquare) : Bool := sorry

/-- Calculates the probability of a knight landing on the same color after one move -/
def knightSameColorProbability (board : Array ChessSquare) : ℚ := sorry

/-- The main theorem to prove -/
theorem knight_same_color_probability :
  knightSameColorProbability chessboard = 1/2 := by sorry

end NUMINAMATH_CALUDE_knight_same_color_probability_l3299_329962


namespace NUMINAMATH_CALUDE_base_4_divisibility_l3299_329928

def base_4_to_decimal (a b c d : ℕ) : ℕ :=
  a * 4^3 + b * 4^2 + c * 4 + d

def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

theorem base_4_divisibility :
  ∀ x : ℕ, x < 4 →
    is_divisible_by_13 (base_4_to_decimal 2 3 1 x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_4_divisibility_l3299_329928


namespace NUMINAMATH_CALUDE_battery_usage_difference_l3299_329921

theorem battery_usage_difference (flashlights remote_controllers wall_clock wireless_mouse toys : ℝ) 
  (h1 : flashlights = 3.5)
  (h2 : remote_controllers = 7.25)
  (h3 : wall_clock = 4.8)
  (h4 : wireless_mouse = 3.4)
  (h5 : toys = 15.75) :
  toys - (flashlights + remote_controllers + wall_clock + wireless_mouse) = -3.2 := by
  sorry

end NUMINAMATH_CALUDE_battery_usage_difference_l3299_329921


namespace NUMINAMATH_CALUDE_min_sticks_to_break_12_can_form_square_15_l3299_329952

-- Define a function to calculate the sum of integers from 1 to n
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define a function to check if it's possible to form a square without breaking sticks
def can_form_square (n : ℕ) : Bool :=
  sum_to_n n % 4 = 0

-- Define a function to find the minimum number of sticks to break
def min_sticks_to_break (n : ℕ) : ℕ :=
  if can_form_square n then 0
  else if n = 12 then 2
  else sorry  -- We don't have a general formula for other cases

-- Theorem for n = 12
theorem min_sticks_to_break_12 :
  min_sticks_to_break 12 = 2 :=
by sorry

-- Theorem for n = 15
theorem can_form_square_15 :
  can_form_square 15 = true :=
by sorry

end NUMINAMATH_CALUDE_min_sticks_to_break_12_can_form_square_15_l3299_329952


namespace NUMINAMATH_CALUDE_container_weight_container_weight_proof_l3299_329931

/-- Given a container with weights p and q when three-quarters and one-third full respectively,
    the total weight when completely full is (8p - 3q) / 5 -/
theorem container_weight (p q : ℝ) : ℝ :=
  let three_quarters_weight := p
  let one_third_weight := q
  let full_weight := (8 * p - 3 * q) / 5
  full_weight

/-- Proof of the container weight theorem -/
theorem container_weight_proof (p q : ℝ) :
  container_weight p q = (8 * p - 3 * q) / 5 := by
  sorry

end NUMINAMATH_CALUDE_container_weight_container_weight_proof_l3299_329931


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l3299_329904

/-- The total amount of ice cream eaten over two nights -/
def total_ice_cream (friday_amount saturday_amount : Real) : Real :=
  friday_amount + saturday_amount

/-- Theorem stating the total amount of ice cream eaten -/
theorem ice_cream_consumption : 
  total_ice_cream 3.25 0.25 = 3.50 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l3299_329904


namespace NUMINAMATH_CALUDE_least_four_divisors_sum_of_squares_l3299_329971

theorem least_four_divisors_sum_of_squares (n : ℕ+) 
  (h1 : ∃ (d1 d2 d3 d4 : ℕ+), d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ 
    (∀ m : ℕ+, m ∣ n → m = d1 ∨ m = d2 ∨ m = d3 ∨ m = d4 ∨ m > d4))
  (h2 : ∃ (d1 d2 d3 d4 : ℕ+), d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ 
    n = d1^2 + d2^2 + d3^2 + d4^2) : 
  n = 130 := by
sorry

end NUMINAMATH_CALUDE_least_four_divisors_sum_of_squares_l3299_329971


namespace NUMINAMATH_CALUDE_box_area_product_equals_volume_squared_l3299_329927

/-- Given a rectangular box with dimensions x, y, and z, 
    prove that the product of the areas of its three pairs of opposite faces 
    is equal to the square of its volume. -/
theorem box_area_product_equals_volume_squared 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^2 := by
  sorry

end NUMINAMATH_CALUDE_box_area_product_equals_volume_squared_l3299_329927


namespace NUMINAMATH_CALUDE_distance_between_cities_l3299_329920

/-- The distance between city A and city B in miles -/
def distance : ℝ := sorry

/-- The time taken for the trip from A to B in hours -/
def time_AB : ℝ := 3

/-- The time taken for the trip from B to A in hours -/
def time_BA : ℝ := 2.5

/-- The time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- The speed for the round trip if time was saved, in miles per hour -/
def speed_with_savings : ℝ := 80

theorem distance_between_cities :
  distance = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l3299_329920


namespace NUMINAMATH_CALUDE_factor_expression_l3299_329963

theorem factor_expression (b : ℝ) : 56 * b^2 + 168 * b = 56 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3299_329963


namespace NUMINAMATH_CALUDE_fence_length_not_eighteen_l3299_329925

theorem fence_length_not_eighteen (length width : ℝ) : 
  length = 6 → width = 3 → 
  ¬(length + 2 * width = 18 ∨ 2 * length + width = 18) := by
sorry

end NUMINAMATH_CALUDE_fence_length_not_eighteen_l3299_329925


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l3299_329961

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l3299_329961


namespace NUMINAMATH_CALUDE_initial_puppies_count_l3299_329950

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has now -/
def puppies_remaining : ℕ := 5

/-- The initial number of puppies Alyssa had -/
def initial_puppies : ℕ := puppies_given_away + puppies_remaining

theorem initial_puppies_count : initial_puppies = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l3299_329950


namespace NUMINAMATH_CALUDE_five_eighths_of_twelve_fifths_l3299_329992

theorem five_eighths_of_twelve_fifths : (5 / 8 : ℚ) * (12 / 5 : ℚ) = (3 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_five_eighths_of_twelve_fifths_l3299_329992


namespace NUMINAMATH_CALUDE_expression_value_l3299_329943

theorem expression_value : 
  let x : ℝ := 3
  5 * 7 + 9 * 4 - 35 / 5 + x * 2 = 70 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3299_329943
