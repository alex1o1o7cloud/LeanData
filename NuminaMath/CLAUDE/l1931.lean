import Mathlib

namespace NUMINAMATH_CALUDE_sugar_amount_theorem_l1931_193196

def sugar_amount (sugar flour baking_soda chocolate_chips : ℚ) : Prop :=
  -- Ratio of sugar to flour is 5:4
  sugar / flour = 5 / 4 ∧
  -- Ratio of flour to baking soda is 10:1
  flour / baking_soda = 10 / 1 ∧
  -- Ratio of baking soda to chocolate chips is 3:2
  baking_soda / chocolate_chips = 3 / 2 ∧
  -- New ratio after adding 120 pounds of baking soda and 50 pounds of chocolate chips
  flour / (baking_soda + 120) = 16 / 3 ∧
  flour / (chocolate_chips + 50) = 16 / 2 ∧
  -- The amount of sugar is 1714 pounds
  sugar = 1714

theorem sugar_amount_theorem :
  ∃ sugar flour baking_soda chocolate_chips : ℚ,
    sugar_amount sugar flour baking_soda chocolate_chips :=
by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_theorem_l1931_193196


namespace NUMINAMATH_CALUDE_smallest_class_size_l1931_193186

theorem smallest_class_size (n : ℕ) : 
  n > 0 ∧ 
  (6 * 120 + (n - 6) * 70 : ℝ) ≤ (n * 85 : ℝ) ∧ 
  (∀ m : ℕ, m > 0 → m < n → (6 * 120 + (m - 6) * 70 : ℝ) > (m * 85 : ℝ)) → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l1931_193186


namespace NUMINAMATH_CALUDE_triangle_altitude_segment_length_l1931_193183

theorem triangle_altitude_segment_length 
  (a b c h d : ℝ) 
  (triangle_sides : a = 30 ∧ b = 70 ∧ c = 80) 
  (altitude_condition : h^2 = b^2 - d^2) 
  (segment_condition : a^2 = h^2 + (c - d)^2) : 
  d = 65 := by sorry

end NUMINAMATH_CALUDE_triangle_altitude_segment_length_l1931_193183


namespace NUMINAMATH_CALUDE_domino_less_than_trimino_l1931_193136

/-- A domino tiling of a 2n × 2n grid -/
def DominoTiling (n : ℕ) := Fin (2*n) → Fin (2*n) → Bool

/-- A trimino tiling of a 3n × 3n grid -/
def TriminoTiling (n : ℕ) := Fin (3*n) → Fin (3*n) → Bool

/-- The number of domino tilings of a 2n × 2n grid -/
def numDominoTilings (n : ℕ) : ℕ := sorry

/-- The number of trimino tilings of a 3n × 3n grid -/
def numTriminoTilings (n : ℕ) : ℕ := sorry

/-- Theorem: The number of domino tilings of a 2n × 2n grid is less than
    the number of trimino tilings of a 3n × 3n grid for all positive n -/
theorem domino_less_than_trimino (n : ℕ) (h : n > 0) : 
  numDominoTilings n < numTriminoTilings n := by
  sorry

end NUMINAMATH_CALUDE_domino_less_than_trimino_l1931_193136


namespace NUMINAMATH_CALUDE_red_square_area_equals_cross_area_l1931_193192

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  /-- Side length of the flag -/
  side : ℝ
  /-- Ratio of the cross arm width to the flag side length -/
  arm_ratio : ℝ
  /-- The cross (arms + center) occupies 49% of the flag area -/
  cross_area_constraint : 4 * arm_ratio * (1 - arm_ratio) = 0.49

theorem red_square_area_equals_cross_area (flag : CrossFlag) :
  4 * flag.arm_ratio^2 = 4 * flag.arm_ratio * (1 - flag.arm_ratio) := by
  sorry

#check red_square_area_equals_cross_area

end NUMINAMATH_CALUDE_red_square_area_equals_cross_area_l1931_193192


namespace NUMINAMATH_CALUDE_negation_of_odd_function_implication_l1931_193156

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem negation_of_odd_function_implication :
  (¬ (is_odd f → is_odd (λ x => f (-x)))) ↔ (is_odd f → ¬ is_odd (λ x => f (-x))) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_odd_function_implication_l1931_193156


namespace NUMINAMATH_CALUDE_license_plate_count_l1931_193197

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 14

/-- The length of the license plate -/
def plate_length : ℕ := 6

/-- The number of possible first letters (B or C) -/
def first_letter_choices : ℕ := 2

/-- The number of possible last letters (N) -/
def last_letter_choices : ℕ := 1

/-- The number of letters that cannot be used in the middle (B, C, M, N) -/
def excluded_middle_letters : ℕ := 4

theorem license_plate_count :
  (first_letter_choices * (alphabet_size - excluded_middle_letters) *
   (alphabet_size - excluded_middle_letters - 1) *
   (alphabet_size - excluded_middle_letters - 2) *
   (alphabet_size - excluded_middle_letters - 3) *
   last_letter_choices) = 15840 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l1931_193197


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_a_theorem_l1931_193132

def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

theorem solution_set_theorem (x : ℝ) :
  f x ≥ -2 ↔ -2/3 ≤ x ∧ x ≤ 6 :=
sorry

theorem range_of_a_theorem :
  (∀ x : ℝ, f x ≤ x - a) ↔ a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_a_theorem_l1931_193132


namespace NUMINAMATH_CALUDE_max_volume_at_one_cm_l1931_193122

/-- The side length of the original square sheet -/
def sheet_side : ℝ := 6

/-- The side length of the small square cut from each corner -/
def cut_side : ℝ := 1

/-- The volume of the box as a function of the cut side length -/
def box_volume (x : ℝ) : ℝ := x * (sheet_side - 2 * x)^2

theorem max_volume_at_one_cm :
  ∀ x, 0 < x → x < sheet_side / 2 → box_volume cut_side ≥ box_volume x :=
sorry

end NUMINAMATH_CALUDE_max_volume_at_one_cm_l1931_193122


namespace NUMINAMATH_CALUDE_average_after_removal_l1931_193133

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) : 
  Finset.card numbers = 12 →
  sum = Finset.sum numbers id →
  sum / 12 = 90 →
  65 ∈ numbers →
  75 ∈ numbers →
  85 ∈ numbers →
  (sum - 65 - 75 - 85) / 9 = 95 :=
by sorry

end NUMINAMATH_CALUDE_average_after_removal_l1931_193133


namespace NUMINAMATH_CALUDE_sqrt_real_range_l1931_193199

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 6 - 2 * x) ↔ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l1931_193199


namespace NUMINAMATH_CALUDE_complement_of_A_l1931_193124

def A : Set ℝ := {x | |x - 1| > 2}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1931_193124


namespace NUMINAMATH_CALUDE_inverse_parallel_corresponding_angles_true_l1931_193123

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of corresponding angles
def correspondingAngles (a1 a2 : Angle) : Prop := sorry

-- Define the concept of equal angles
def equalAngles (a1 a2 : Angle) : Prop := sorry

-- Theorem statement
theorem inverse_parallel_corresponding_angles_true :
  ∀ (l1 l2 : Line) (a1 a2 : Angle),
    (correspondingAngles a1 a2 ∧ equalAngles a1 a2) → parallel l1 l2 :=
by sorry

end NUMINAMATH_CALUDE_inverse_parallel_corresponding_angles_true_l1931_193123


namespace NUMINAMATH_CALUDE_ravi_money_l1931_193106

theorem ravi_money (ravi giri kiran : ℚ) : 
  (ravi / giri = 6 / 7) →
  (giri / kiran = 6 / 15) →
  (kiran = 105) →
  ravi = 36 := by
sorry

end NUMINAMATH_CALUDE_ravi_money_l1931_193106


namespace NUMINAMATH_CALUDE_cell_phone_call_cost_l1931_193163

/-- Given a constant rate per minute where a 3-minute call costs $0.18, 
    prove that a 10-minute call will cost $0.60. -/
theorem cell_phone_call_cost 
  (rate : ℝ) 
  (h1 : rate * 3 = 0.18) -- Cost of 3-minute call
  : rate * 10 = 0.60 := by 
  sorry

end NUMINAMATH_CALUDE_cell_phone_call_cost_l1931_193163


namespace NUMINAMATH_CALUDE_tangent_line_smallest_slope_l1931_193121

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem tangent_line_smallest_slope :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, f x = y → a*x + b*y + c = 0) ∧ 
    (∀ x₀ y₀ : ℝ, f x₀ = y₀ → ∀ m : ℝ, (∃ x y : ℝ, f x = y ∧ m = f' x) → m ≥ a) ∧
    a = 3 ∧ b = -1 ∧ c = -11 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_smallest_slope_l1931_193121


namespace NUMINAMATH_CALUDE_sum_sqrt_inequality_l1931_193140

theorem sum_sqrt_inequality (a b c : ℝ) 
  (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) 
  (sum_eq : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_inequality_l1931_193140


namespace NUMINAMATH_CALUDE_sum_of_roots_of_equation_l1931_193187

theorem sum_of_roots_of_equation (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, (r₁ - 7)^2 = 16 ∧ (r₂ - 7)^2 = 16 ∧ r₁ + r₂ = 14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_equation_l1931_193187


namespace NUMINAMATH_CALUDE_klinker_age_relation_l1931_193117

/-- Represents the ages of Mr. Klinker, Julie, and Tim -/
structure Ages where
  klinker : ℕ
  julie : ℕ
  tim : ℕ

/-- The current ages -/
def currentAges : Ages := { klinker := 48, julie := 12, tim := 8 }

/-- The number of years to pass -/
def yearsLater : ℕ := 12

/-- Calculates the ages after a given number of years -/
def agesAfter (initial : Ages) (years : ℕ) : Ages :=
  { klinker := initial.klinker + years
  , julie := initial.julie + years
  , tim := initial.tim + years }

/-- Theorem stating that after 12 years, Mr. Klinker will be twice as old as Julie and thrice as old as Tim -/
theorem klinker_age_relation :
  let futureAges := agesAfter currentAges yearsLater
  futureAges.klinker = 2 * futureAges.julie ∧ futureAges.klinker = 3 * futureAges.tim :=
by sorry

end NUMINAMATH_CALUDE_klinker_age_relation_l1931_193117


namespace NUMINAMATH_CALUDE_division_problem_l1931_193134

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 15 ∧ quotient = 4 ∧ remainder = 3 ∧ 
  dividend = divisor * quotient + remainder → 
  divisor = 3 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1931_193134


namespace NUMINAMATH_CALUDE_bridge_lamps_l1931_193105

/-- The number of lamps on a bridge -/
def numLamps (bridgeLength : ℕ) (lampSpacing : ℕ) : ℕ :=
  bridgeLength / lampSpacing + 1

theorem bridge_lamps :
  let bridgeLength : ℕ := 30
  let lampSpacing : ℕ := 5
  numLamps bridgeLength lampSpacing = 7 := by
  sorry

end NUMINAMATH_CALUDE_bridge_lamps_l1931_193105


namespace NUMINAMATH_CALUDE_max_side_length_l1931_193139

/-- A triangle with three different integer side lengths and perimeter 30 -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter : a + b + c = 30

/-- The maximum length of any side in a triangle with perimeter 30 and different integer side lengths is 14 -/
theorem max_side_length (t : Triangle) : t.a ≤ 14 ∧ t.b ≤ 14 ∧ t.c ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_max_side_length_l1931_193139


namespace NUMINAMATH_CALUDE_special_function_unique_special_function_at_3_l1931_193116

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y)

/-- Theorem stating that any function satisfying the conditions must be f(x) = 2x -/
theorem special_function_unique (f : ℝ → ℝ) (h : special_function f) : 
  ∀ x : ℝ, f x = 2 * x :=
sorry

/-- Corollary: f(3) = 6 for any function satisfying the conditions -/
theorem special_function_at_3 (f : ℝ → ℝ) (h : special_function f) : 
  f 3 = 6 :=
sorry

end NUMINAMATH_CALUDE_special_function_unique_special_function_at_3_l1931_193116


namespace NUMINAMATH_CALUDE_alissa_presents_count_l1931_193168

/-- The number of presents Ethan has -/
def ethan_presents : ℝ := 31.0

/-- The difference between Ethan's and Alissa's presents -/
def difference : ℝ := 22.0

/-- The number of presents Alissa has -/
def alissa_presents : ℝ := ethan_presents - difference

theorem alissa_presents_count : alissa_presents = 9.0 := by sorry

end NUMINAMATH_CALUDE_alissa_presents_count_l1931_193168


namespace NUMINAMATH_CALUDE_sara_bought_fifteen_cards_l1931_193119

/-- Calculates the number of baseball cards Sara bought from Sally -/
def cards_bought_by_sara (initial_cards torn_cards remaining_cards : ℕ) : ℕ :=
  initial_cards - torn_cards - remaining_cards

/-- Theorem stating that Sara bought 15 cards from Sally -/
theorem sara_bought_fifteen_cards :
  let initial_cards := 39
  let torn_cards := 9
  let remaining_cards := 15
  cards_bought_by_sara initial_cards torn_cards remaining_cards = 15 := by
  sorry

end NUMINAMATH_CALUDE_sara_bought_fifteen_cards_l1931_193119


namespace NUMINAMATH_CALUDE_upgrade_ways_count_l1931_193159

/-- Represents the number of levels in the game -/
def totalLevels : ℕ := 16

/-- Represents the level at which the special ability can first be upgraded -/
def firstSpecialLevel : ℕ := 6

/-- Represents the level at which the special ability can be upgraded for the second time -/
def secondSpecialLevel : ℕ := 11

/-- Represents the number of times the special ability must be upgraded -/
def specialUpgrades : ℕ := 2

/-- Represents the number of choices for upgrading regular abilities at each level -/
def regularChoices : ℕ := 3

/-- The function that calculates the number of ways to upgrade abilities -/
def upgradeWays : ℕ := 5 * (regularChoices ^ totalLevels)

/-- Theorem stating that the number of ways to upgrade abilities is 5 · 3^16 -/
theorem upgrade_ways_count : upgradeWays = 5 * (3 ^ 16) := by
  sorry

end NUMINAMATH_CALUDE_upgrade_ways_count_l1931_193159


namespace NUMINAMATH_CALUDE_xy_sum_squared_l1931_193177

theorem xy_sum_squared (x y : ℝ) (h1 : x * y = -3) (h2 : x + y = -4) :
  x^2 + 3*x*y + y^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_squared_l1931_193177


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1931_193154

-- Define set M
def M : Set ℝ := {x | x * (x - 5) ≤ 6}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1931_193154


namespace NUMINAMATH_CALUDE_sine_identity_l1931_193146

theorem sine_identity (α : Real) (h : α = π / 7) :
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := by
  sorry

end NUMINAMATH_CALUDE_sine_identity_l1931_193146


namespace NUMINAMATH_CALUDE_windows_preference_l1931_193107

/-- Given a survey of college students about computer brand preferences,
    this theorem proves the number of students preferring Windows. -/
theorem windows_preference (total : ℕ) (mac : ℕ) (no_pref : ℕ) : 
  total = 210 →
  mac = 60 →
  no_pref = 90 →
  ∃ (windows : ℕ), 
    windows = total - (mac + mac / 3 + no_pref) ∧
    windows = 40 :=
by sorry

end NUMINAMATH_CALUDE_windows_preference_l1931_193107


namespace NUMINAMATH_CALUDE_student_number_problem_l1931_193151

theorem student_number_problem (x : ℝ) : (7 * x - 150 = 130) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1931_193151


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l1931_193101

theorem tic_tac_toe_tie_probability (amy_win : ℚ) (lily_win : ℚ) :
  amy_win = 5/12 → lily_win = 1/4 → 1 - (amy_win + lily_win) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l1931_193101


namespace NUMINAMATH_CALUDE_kids_savings_l1931_193193

-- Define the number of coins each child has
def teagan_pennies : ℕ := 200
def rex_nickels : ℕ := 100
def toni_dimes : ℕ := 330

-- Define the value of each coin type in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10

-- Define the total savings in cents
def total_savings : ℕ := 
  teagan_pennies * penny_value + 
  rex_nickels * nickel_value + 
  toni_dimes * dime_value

-- Theorem to prove
theorem kids_savings : total_savings = 4000 := by
  sorry

end NUMINAMATH_CALUDE_kids_savings_l1931_193193


namespace NUMINAMATH_CALUDE_parabola_sum_l1931_193173

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℚ) : ℚ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) :
  p.x_coord (-6) = 7 →  -- vertex condition
  p.x_coord 0 = 1 →     -- point condition
  p.a + p.b + p.c = -43/6 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l1931_193173


namespace NUMINAMATH_CALUDE_price_reduction_doubles_profit_l1931_193104

-- Define the initial conditions
def initial_purchase_price : ℝ := 210
def initial_selling_price : ℝ := 270
def initial_daily_sales : ℝ := 30
def sales_increase_per_yuan : ℝ := 3

-- Define the profit function
def profit (price_reduction : ℝ) : ℝ :=
  let new_price := initial_selling_price - price_reduction
  let new_sales := initial_daily_sales + sales_increase_per_yuan * price_reduction
  (new_price - initial_purchase_price) * new_sales

-- Theorem statement
theorem price_reduction_doubles_profit :
  ∃ (price_reduction : ℝ), 
    price_reduction = 30 ∧ 
    profit price_reduction = 2 * profit 0 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_doubles_profit_l1931_193104


namespace NUMINAMATH_CALUDE_trivia_team_score_l1931_193126

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_score : ℕ) :
  total_members = 7 →
  absent_members = 2 →
  total_score = 20 →
  (total_score / (total_members - absent_members) : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_score_l1931_193126


namespace NUMINAMATH_CALUDE_power_three_twenty_mod_five_l1931_193180

theorem power_three_twenty_mod_five : 3^20 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_twenty_mod_five_l1931_193180


namespace NUMINAMATH_CALUDE_average_age_of_five_students_l1931_193147

/-- Proves that the average age of 5 students is 14 years given the conditions of the problem -/
theorem average_age_of_five_students
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_students_with_known_average : Nat)
  (average_age_known : ℝ)
  (age_of_twelfth_student : ℕ)
  (h1 : total_students = 16)
  (h2 : average_age_all = 16)
  (h3 : num_students_with_known_average = 9)
  (h4 : average_age_known = 16)
  (h5 : age_of_twelfth_student = 42)
  : (total_students * average_age_all - num_students_with_known_average * average_age_known - age_of_twelfth_student) / (total_students - num_students_with_known_average - 1) = 14 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_five_students_l1931_193147


namespace NUMINAMATH_CALUDE_grants_room_count_l1931_193113

def danielles_rooms : ℕ := 6

def heidis_rooms (danielles_rooms : ℕ) : ℕ := 3 * danielles_rooms

def grants_rooms (heidis_rooms : ℕ) : ℚ := (1 : ℚ) / 9 * heidis_rooms

theorem grants_room_count :
  grants_rooms (heidis_rooms danielles_rooms) = 2 := by
  sorry

end NUMINAMATH_CALUDE_grants_room_count_l1931_193113


namespace NUMINAMATH_CALUDE_inequality_proof_l1931_193178

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : 0 < d ∧ 0 < c ∧ 0 < b ∧ 0 < a) 
  (h_order : d < c ∧ c < b ∧ b < a) : 
  (a + b + c + d)^2 > a^2 + 3*b^2 + 5*c^2 + 7*d^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1931_193178


namespace NUMINAMATH_CALUDE_trailing_zeros_of_nine_to_999_plus_one_l1931_193194

theorem trailing_zeros_of_nine_to_999_plus_one :
  ∃ n : ℕ, (9^999 + 1 : ℕ) = 10 * n ∧ (9^999 + 1 : ℕ) % 100 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_nine_to_999_plus_one_l1931_193194


namespace NUMINAMATH_CALUDE_expression_decrease_value_decrease_l1931_193188

theorem expression_decrease (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  (3/4 * x) * (3/4 * y)^2 = (27/64) * (x * y^2) := by
  sorry

theorem value_decrease (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  1 - (3/4 * x) * (3/4 * y)^2 / (x * y^2) = 37/64 := by
  sorry

end NUMINAMATH_CALUDE_expression_decrease_value_decrease_l1931_193188


namespace NUMINAMATH_CALUDE_f_properties_l1931_193166

noncomputable def f (x : ℝ) : ℝ := ((Real.sin x - Real.cos x) * Real.sin (2 * x)) / Real.sin x

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 ↔ x ∉ {y | ∃ k : ℤ, y = k * Real.pi}) ∧
  (∃ T : ℝ, T > 0 ∧ is_periodic f T ∧ ∀ S, (S > 0 ∧ is_periodic f S) → T ≤ S) ∧
  (∀ x : ℝ, f x ≥ 0 ↔ ∃ k : ℤ, x ∈ Set.Icc (Real.pi / 4 + k * Real.pi) (Real.pi / 2 + k * Real.pi)) ∧
  (∃ m : ℝ, m > 0 ∧ is_even (fun x ↦ f (x + m)) ∧
    ∀ n : ℝ, (n > 0 ∧ is_even (fun x ↦ f (x + n))) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1931_193166


namespace NUMINAMATH_CALUDE_matching_color_probability_l1931_193172

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.yellow

/-- Abe's jelly bean distribution -/
def abe : JellyBeans :=
  { green := 2, red := 2, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob : JellyBeans :=
  { green := 2, red := 3, yellow := 2 }

/-- Calculates the probability of selecting a specific color -/
def probColor (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / jb.total

/-- Calculates the probability of both selecting the same color -/
def probMatchingColor (jb1 jb2 : JellyBeans) : ℚ :=
  probColor jb1 jb1.green * probColor jb2 jb2.green +
  probColor jb1 jb1.red * probColor jb2 jb2.red

theorem matching_color_probability :
  probMatchingColor abe bob = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_matching_color_probability_l1931_193172


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_count_l1931_193138

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_ratio_count
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 2 * a 8 = 36)
  (h_sum : a 3 + a 7 = 15) :
  ∃ (S : Finset ℝ), (∀ q ∈ S, ∃ (a : ℕ → ℝ), is_geometric_sequence a ∧ a 2 * a 8 = 36 ∧ a 3 + a 7 = 15) ∧ S.card = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_count_l1931_193138


namespace NUMINAMATH_CALUDE_marble_difference_l1931_193179

/-- Represents a jar of marbles -/
structure Jar :=
  (blue : ℕ)
  (green : ℕ)

/-- The problem statement -/
theorem marble_difference (jar1 jar2 : Jar) : 
  jar1.blue + jar1.green = jar2.blue + jar2.green →
  7 * jar1.green = 3 * jar1.blue →
  9 * jar2.green = jar2.blue →
  jar1.green + jar2.green = 80 →
  jar2.blue - jar1.blue = 40 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l1931_193179


namespace NUMINAMATH_CALUDE_tree_planting_event_l1931_193149

theorem tree_planting_event (boys : ℕ) (girls : ℕ) 
  (h1 : boys = 600)
  (h2 : girls > boys)
  (h3 : (boys + girls) * 60 / 100 = 960) :
  girls - boys = 400 := by
sorry

end NUMINAMATH_CALUDE_tree_planting_event_l1931_193149


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1931_193153

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The arithmetic sequence condition -/
def ArithmeticCondition (a : ℕ → ℝ) : Prop :=
  2 * ((1/2) * a 3) = a 1 + 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticCondition a →
  (∀ n : ℕ, a n > 0) →
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1931_193153


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l1931_193182

theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) : 
  small_side = 1 →
  large_side = 15 →
  (large_side / small_side) ^ 2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l1931_193182


namespace NUMINAMATH_CALUDE_first_day_is_friday_l1931_193170

/-- Days of the week -/
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

/-- Function to get the day of the week after n days -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (dayAfter d m)

/-- Theorem: If the 25th day of a month is a Monday, then the 1st day of that month is a Friday -/
theorem first_day_is_friday (d : DayOfWeek) : 
  dayAfter d 24 = DayOfWeek.Monday → d = DayOfWeek.Friday :=
by
  sorry


end NUMINAMATH_CALUDE_first_day_is_friday_l1931_193170


namespace NUMINAMATH_CALUDE_hexagon_intersection_area_l1931_193128

-- Define the hexagon
structure Hexagon where
  area : ℝ
  is_regular : Prop

-- Define a function to calculate the expected value
def expected_intersection_area (H : Hexagon) : ℝ :=
  -- The actual calculation of the expected value
  12

-- The theorem to be proved
theorem hexagon_intersection_area (H : Hexagon) 
  (h1 : H.area = 360) 
  (h2 : H.is_regular) : 
  expected_intersection_area H = 12 := by
  sorry


end NUMINAMATH_CALUDE_hexagon_intersection_area_l1931_193128


namespace NUMINAMATH_CALUDE_min_distance_theorem_l1931_193142

theorem min_distance_theorem (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ - a)^2 + (Real.log (x₀^2) - 2*a)^2 ≤ 4/5) →
  a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_theorem_l1931_193142


namespace NUMINAMATH_CALUDE_problem_solution_l1931_193114

theorem problem_solution (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -1) : 
  (3 * a + a * b + 3 * b = 5) ∧ (a^2 + b^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1931_193114


namespace NUMINAMATH_CALUDE_tangent_fifteen_degree_ratio_l1931_193110

theorem tangent_fifteen_degree_ratio :
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_fifteen_degree_ratio_l1931_193110


namespace NUMINAMATH_CALUDE_parity_of_cube_plus_multiple_l1931_193135

theorem parity_of_cube_plus_multiple (o n : ℤ) (h_odd : Odd o) :
  Odd (o^3 + n*o) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_parity_of_cube_plus_multiple_l1931_193135


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l1931_193120

-- Define the set A (condition p)
def A (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a^2 + 1}

-- Define the set B (condition q)
def B (a : ℝ) : Set ℝ := {x | x^2 - 3 * (a + 1) * x + 6 * a + 2 ≤ 0}

-- Define the range of a
def RangeOfA : Set ℝ := {a | 1 ≤ a ∧ a ≤ 3 ∨ a = -1}

-- Statement of the theorem
theorem range_of_a_theorem :
  (∀ a : ℝ, A a ⊆ B a) → 
  (∀ a : ℝ, a ∈ RangeOfA ↔ (A a ⊆ B a)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l1931_193120


namespace NUMINAMATH_CALUDE_jacob_tank_fill_time_l1931_193176

/-- Represents the number of days needed to fill a water tank -/
def days_to_fill_tank (tank_capacity_liters : ℕ) (rain_collection_ml : ℕ) (river_collection_ml : ℕ) : ℕ :=
  (tank_capacity_liters * 1000) / (rain_collection_ml + river_collection_ml)

/-- Theorem stating that it takes 20 days to fill Jacob's water tank -/
theorem jacob_tank_fill_time :
  days_to_fill_tank 50 800 1700 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jacob_tank_fill_time_l1931_193176


namespace NUMINAMATH_CALUDE_P_n_roots_P_2018_roots_l1931_193130

-- Define the sequence of polynomials
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | 1, x => x
  | (n + 2), x => x * P (n + 1) x - P n x

-- Define a function to count distinct real roots
noncomputable def count_distinct_real_roots (f : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem P_n_roots (n : ℕ) : count_distinct_real_roots (P n) = n := by
  sorry

-- Specific case for P_2018
theorem P_2018_roots : count_distinct_real_roots (P 2018) = 2018 := by
  sorry

end NUMINAMATH_CALUDE_P_n_roots_P_2018_roots_l1931_193130


namespace NUMINAMATH_CALUDE_work_rate_increase_l1931_193109

theorem work_rate_increase (total_time hours_worked : ℝ)
  (original_items additional_items : ℕ) :
  total_time = 10 ∧ 
  hours_worked = 6 ∧ 
  original_items = 1250 ∧ 
  additional_items = 150 →
  let original_rate := original_items / total_time
  let items_processed := original_rate * hours_worked
  let remaining_items := original_items - items_processed + additional_items
  let remaining_time := total_time - hours_worked
  let new_rate := remaining_items / remaining_time
  (new_rate - original_rate) / original_rate * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_work_rate_increase_l1931_193109


namespace NUMINAMATH_CALUDE_investment_problem_l1931_193127

/-- Calculates an investor's share of the profit based on their investment, duration, and the total profit --/
def calculate_share (investment : ℕ) (duration : ℕ) (total_investment_time : ℕ) (total_profit : ℕ) : ℚ :=
  (investment * duration : ℚ) / total_investment_time * total_profit

/-- Represents the investment problem with four investors --/
theorem investment_problem (tom_investment : ℕ) (jose_investment : ℕ) (anil_investment : ℕ) (maya_investment : ℕ)
  (tom_duration : ℕ) (jose_duration : ℕ) (anil_duration : ℕ) (maya_duration : ℕ) (total_profit : ℕ) :
  tom_investment = 30000 →
  jose_investment = 45000 →
  anil_investment = 50000 →
  maya_investment = 70000 →
  tom_duration = 12 →
  jose_duration = 10 →
  anil_duration = 7 →
  maya_duration = 1 →
  total_profit = 108000 →
  let total_investment_time := tom_investment * tom_duration + jose_investment * jose_duration +
                               anil_investment * anil_duration + maya_investment * maya_duration
  abs (calculate_share jose_investment jose_duration total_investment_time total_profit - 39512.20) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l1931_193127


namespace NUMINAMATH_CALUDE_two_digit_sum_doubled_l1931_193157

theorem two_digit_sum_doubled (J L M K : ℕ) 
  (h_digits : J < 10 ∧ L < 10 ∧ M < 10 ∧ K < 10)
  (h_sum : (10 * J + M) + (10 * L + K) = 79) :
  2 * ((10 * J + M) + (10 * L + K)) = 158 := by
sorry

end NUMINAMATH_CALUDE_two_digit_sum_doubled_l1931_193157


namespace NUMINAMATH_CALUDE_owner_short_percentage_l1931_193195

/-- Calculates the percentage of tank price the owner is short of after selling goldfish --/
def percentage_short_of_tank_price (goldfish_buy_price goldfish_sell_price tank_cost : ℚ) 
                                   (goldfish_sold : ℕ) : ℚ :=
  let profit_per_goldfish := goldfish_sell_price - goldfish_buy_price
  let total_profit := profit_per_goldfish * goldfish_sold
  let amount_short := tank_cost - total_profit
  (amount_short / tank_cost) * 100

/-- Proves that the owner is short of 45% of the tank price --/
theorem owner_short_percentage (goldfish_buy_price goldfish_sell_price tank_cost : ℚ) 
                               (goldfish_sold : ℕ) :
  goldfish_buy_price = 25/100 →
  goldfish_sell_price = 75/100 →
  tank_cost = 100 →
  goldfish_sold = 110 →
  percentage_short_of_tank_price goldfish_buy_price goldfish_sell_price tank_cost goldfish_sold = 45 :=
by
  sorry

#eval percentage_short_of_tank_price (25/100) (75/100) 100 110

end NUMINAMATH_CALUDE_owner_short_percentage_l1931_193195


namespace NUMINAMATH_CALUDE_population_growth_inequality_l1931_193141

theorem population_growth_inequality (m n p : ℝ) 
  (h1 : (1 + p / 100)^2 = (1 + m / 100) * (1 + n / 100)) : 
  p ≤ (m + n) / 2 := by
sorry

end NUMINAMATH_CALUDE_population_growth_inequality_l1931_193141


namespace NUMINAMATH_CALUDE_mod_seventeen_problem_l1931_193103

theorem mod_seventeen_problem (n : ℕ) (h1 : n < 17) (h2 : (2 * n) % 17 = 1) :
  (3^n)^2 % 17 - 3 % 17 = 13 % 17 := by
  sorry

end NUMINAMATH_CALUDE_mod_seventeen_problem_l1931_193103


namespace NUMINAMATH_CALUDE_square_sum_value_l1931_193115

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 15) : a^2 + b^2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l1931_193115


namespace NUMINAMATH_CALUDE_sector_area_l1931_193162

/-- Given a sector with central angle 135° and arc length 3π cm, its area is 6π cm² -/
theorem sector_area (θ : ℝ) (arc_length : ℝ) (area : ℝ) :
  θ = 135 ∧ arc_length = 3 * Real.pi → area = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1931_193162


namespace NUMINAMATH_CALUDE_deck_width_l1931_193155

/-- Proves that for a rectangular pool of 20 feet by 22 feet, surrounded by a deck of uniform width,
    if the total area of the pool and deck is 728 square feet, then the width of the deck is 3 feet. -/
theorem deck_width (w : ℝ) : 
  (20 + 2*w) * (22 + 2*w) = 728 → w = 3 := by sorry

end NUMINAMATH_CALUDE_deck_width_l1931_193155


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1931_193131

-- Define what a quadratic equation is
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function we're checking
def f (x : ℝ) : ℝ := x^2 - 2

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1931_193131


namespace NUMINAMATH_CALUDE_average_of_combined_sets_l1931_193125

theorem average_of_combined_sets :
  ∀ (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ),
    n₁ = 30 →
    n₂ = 20 →
    avg₁ = 20 →
    avg₂ = 30 →
    (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = 24 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_sets_l1931_193125


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1931_193169

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1931_193169


namespace NUMINAMATH_CALUDE_factorization_example_l1931_193144

theorem factorization_example (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

#check factorization_example

end NUMINAMATH_CALUDE_factorization_example_l1931_193144


namespace NUMINAMATH_CALUDE_third_side_length_l1931_193100

-- Define a triangle with two known sides
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem third_side_length (x : ℝ) :
  Triangle 3 7 x ↔ x = 7 :=
sorry

end NUMINAMATH_CALUDE_third_side_length_l1931_193100


namespace NUMINAMATH_CALUDE_ae_length_l1931_193129

-- Define the points
variable (A B C D E : Point)

-- Define the shapes
def is_isosceles_trapezoid (A B C E : Point) : Prop := sorry

def is_rectangle (A C D E : Point) : Prop := sorry

-- Define the lengths
def length (P Q : Point) : ℝ := sorry

-- State the theorem
theorem ae_length 
  (h1 : is_isosceles_trapezoid A B C E)
  (h2 : is_rectangle A C D E)
  (h3 : length A B = 10)
  (h4 : length E C = 20) :
  length A E = 20 := by sorry

end NUMINAMATH_CALUDE_ae_length_l1931_193129


namespace NUMINAMATH_CALUDE_current_year_is_2021_l1931_193137

-- Define the given conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := kelsey_birth_year - 3
def sister_current_age : ℕ := 50

-- Define the theorem
theorem current_year_is_2021 :
  sister_birth_year + sister_current_age = 2021 :=
sorry

end NUMINAMATH_CALUDE_current_year_is_2021_l1931_193137


namespace NUMINAMATH_CALUDE_min_value_in_intersection_l1931_193150

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | let (x, y) := p; (y - x) * (y - 18 / (25 * x)) ≥ 0}
def B : Set (ℝ × ℝ) := {p | let (x, y) := p; (x - 1)^2 + (y - 1)^2 ≤ 1}

-- Define the objective function
def f (p : ℝ × ℝ) : ℝ := let (x, y) := p; 2 * x - y

-- Theorem statement
theorem min_value_in_intersection :
  (∀ p ∈ A ∩ B, f p ≥ -1) ∧ (∃ p ∈ A ∩ B, f p = -1) :=
sorry

end NUMINAMATH_CALUDE_min_value_in_intersection_l1931_193150


namespace NUMINAMATH_CALUDE_condition_analysis_l1931_193102

theorem condition_analysis (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a + Real.log b > b + Real.log a) ∧
  (∃ a b : ℝ, a + Real.log b > b + Real.log a ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l1931_193102


namespace NUMINAMATH_CALUDE_cricket_average_l1931_193108

theorem cricket_average (initial_average : ℝ) : 
  (8 * initial_average + 90) / 9 = initial_average + 6 → 
  initial_average + 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l1931_193108


namespace NUMINAMATH_CALUDE_find_b_value_l1931_193165

theorem find_b_value (a b c : ℝ) 
  (sum_eq : a + b + c = 99)
  (equal_after_change : a + 6 = b - 6 ∧ b - 6 = 5 * c) : 
  b = 51 := by
sorry

end NUMINAMATH_CALUDE_find_b_value_l1931_193165


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1931_193184

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_conversion_sum : 
  let base9 := toBase10 [1, 2, 3] 9
  let base8 := toBase10 [6, 5, 2] 8
  let base7 := toBase10 [4, 3, 1] 7
  base9 - base8 + base7 = 162 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1931_193184


namespace NUMINAMATH_CALUDE_smallest_square_area_l1931_193112

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A square defined by its side length -/
structure Square where
  sideLength : ℕ

/-- Predicate to check if a point is inside or on the boundary of a square -/
def isInSquare (s : Square) (p : LatticePoint) : Prop :=
  0 ≤ p.x ∧ p.x ≤ s.sideLength ∧ 0 ≤ p.y ∧ p.y ≤ s.sideLength

/-- Predicate to check if a point is a vertex of a square -/
def isVertex (s : Square) (p : LatticePoint) : Prop :=
  (p.x = 0 ∨ p.x = s.sideLength) ∧ (p.y = 0 ∨ p.y = s.sideLength)

/-- The main theorem -/
theorem smallest_square_area :
  ∃ (s : Square),
    (∃ (v1 v2 v3 v4 : LatticePoint),
      v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧
      isVertex s v1 ∧ isVertex s v2 ∧ isVertex s v3 ∧ isVertex s v4) ∧
    (∀ (p : LatticePoint),
      isInSquare s p → (isVertex s p ∨ p.x = 0 ∨ p.x = s.sideLength ∨ p.y = 0 ∨ p.y = s.sideLength)) ∧
    (∀ (s' : Square),
      (∃ (v1' v2' v3' v4' : LatticePoint),
        v1' ≠ v2' ∧ v1' ≠ v3' ∧ v1' ≠ v4' ∧ v2' ≠ v3' ∧ v2' ≠ v4' ∧ v3' ≠ v4' ∧
        isVertex s' v1' ∧ isVertex s' v2' ∧ isVertex s' v3' ∧ isVertex s' v4') →
      (∀ (p : LatticePoint),
        isInSquare s' p → (isVertex s' p ∨ p.x = 0 ∨ p.x = s'.sideLength ∨ p.y = 0 ∨ p.y = s'.sideLength)) →
      s.sideLength ≤ s'.sideLength) ∧
    s.sideLength * s.sideLength = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l1931_193112


namespace NUMINAMATH_CALUDE_inverse_composition_equality_l1931_193152

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the condition f⁻¹ ∘ g = λ x, 2*x - 4
variable (h : ∀ x, f⁻¹ (g x) = 2 * x - 4)

-- State the theorem
theorem inverse_composition_equality : g⁻¹ (f (-3)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_equality_l1931_193152


namespace NUMINAMATH_CALUDE_pizza_dough_milk_calculation_l1931_193160

/-- Calculates the amount of milk needed for pizza dough given the flour quantity and milk-to-flour ratio -/
def milk_needed (flour_quantity : ℕ) (milk_per_flour_unit : ℚ) : ℚ :=
  (flour_quantity : ℚ) * milk_per_flour_unit

/-- Proves the correct amount of milk for one and two batches of pizza dough -/
theorem pizza_dough_milk_calculation :
  let flour_quantity : ℕ := 1200
  let milk_per_flour_unit : ℚ := 60 / 300
  let milk_for_one_batch : ℚ := milk_needed flour_quantity milk_per_flour_unit
  let milk_for_two_batches : ℚ := 2 * milk_for_one_batch
  milk_for_one_batch = 240 ∧ milk_for_two_batches = 480 := by
  sorry

#check pizza_dough_milk_calculation

end NUMINAMATH_CALUDE_pizza_dough_milk_calculation_l1931_193160


namespace NUMINAMATH_CALUDE_inequality_count_l1931_193145

theorem inequality_count (x y a b : ℝ) (hx : |x| > a) (hy : |y| > b) :
  ∃! n : ℕ, n = (Bool.toNat (|x + y| > a + b)) +
               (Bool.toNat (|x - y| > |a - b|)) +
               (Bool.toNat (x * y > a * b)) +
               (Bool.toNat (|x / y| > |a / b|)) ∧
  n = 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_count_l1931_193145


namespace NUMINAMATH_CALUDE_factor_expression_l1931_193198

theorem factor_expression (x : ℝ) : 9*x^2 + 3*x = 3*x*(3*x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1931_193198


namespace NUMINAMATH_CALUDE_circle_area_equals_rectangle_area_l1931_193185

theorem circle_area_equals_rectangle_area (R : ℝ) (h : R = 4) :
  π * R^2 = (2 * π * R) * (R / 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equals_rectangle_area_l1931_193185


namespace NUMINAMATH_CALUDE_sum_of_roots_of_special_quadratic_l1931_193191

/-- A real quadratic trinomial -/
def QuadraticTrinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_roots_of_special_quadratic 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x : ℝ, QuadraticTrinomial a b c (x^3 + x) ≥ QuadraticTrinomial a b c (x^2 + 1)) →
  (-b / a = 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_special_quadratic_l1931_193191


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1931_193174

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![-2, 3]

theorem vector_sum_magnitude :
  ‖vector_a + vector_b‖ = Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1931_193174


namespace NUMINAMATH_CALUDE_gmat_question_percentages_l1931_193190

theorem gmat_question_percentages
  (first_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : first_correct = 85)
  (h2 : neither_correct = 5)
  (h3 : both_correct = 60)
  : ∃ (second_correct : ℝ), second_correct = 70 :=
by sorry

end NUMINAMATH_CALUDE_gmat_question_percentages_l1931_193190


namespace NUMINAMATH_CALUDE_line_perpendicular_theorem_l1931_193111

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_theorem
  (a b : Line) (α β : Plane)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : contained a α)
  (h4 : perpendicularLP b β)
  (h5 : parallel α β) :
  perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_theorem_l1931_193111


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1931_193158

theorem arithmetic_sequence_sum (a₁ d : ℝ) (h₁ : d ≠ 0) :
  let a : ℕ → ℝ := λ n => a₁ + (n - 1) * d
  let S : ℕ → ℝ := λ n => n * a₁ + n * (n - 1) / 2 * d
  (a 4)^2 = (a 3) * (a 7) ∧ S 8 = 32 → S 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1931_193158


namespace NUMINAMATH_CALUDE_right_triangle_area_perimeter_l1931_193161

theorem right_triangle_area_perimeter : 
  ∀ (a b c : ℝ),
  a = 5 →
  c = 13 →
  a^2 + b^2 = c^2 →
  (1/2 * a * b = 30 ∧ a + b + c = 30) :=
λ a b c h1 h2 h3 =>
  sorry

#check right_triangle_area_perimeter

end NUMINAMATH_CALUDE_right_triangle_area_perimeter_l1931_193161


namespace NUMINAMATH_CALUDE_painted_cubes_count_l1931_193189

/-- Represents the number of smaller cubes with a given number of painted faces -/
structure PaintedCubes :=
  (three : ℕ)
  (two : ℕ)
  (one : ℕ)

/-- Calculates the number of smaller cubes with different numbers of painted faces
    when a large cube is cut into smaller cubes -/
def countPaintedCubes (large_edge : ℕ) (small_edge : ℕ) : PaintedCubes :=
  sorry

/-- Theorem stating the correct number of painted smaller cubes for the given problem -/
theorem painted_cubes_count :
  countPaintedCubes 8 2 = PaintedCubes.mk 8 24 24 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l1931_193189


namespace NUMINAMATH_CALUDE_quadratic_equations_count_l1931_193171

variable (p : ℕ) [Fact (Nat.Prime p)]

/-- The number of quadratic equations with two distinct roots in p-arithmetic -/
def two_roots (p : ℕ) : ℕ := p * (p - 1) / 2

/-- The number of quadratic equations with exactly one root in p-arithmetic -/
def one_root (p : ℕ) : ℕ := p

/-- The number of quadratic equations with no roots in p-arithmetic -/
def no_roots (p : ℕ) : ℕ := p * (p - 1) / 2

/-- The total number of distinct quadratic equations in p-arithmetic -/
def total_equations (p : ℕ) : ℕ := p^2

theorem quadratic_equations_count (p : ℕ) [Fact (Nat.Prime p)] :
  two_roots p + one_root p + no_roots p = total_equations p :=
sorry

end NUMINAMATH_CALUDE_quadratic_equations_count_l1931_193171


namespace NUMINAMATH_CALUDE_expand_expression_l1931_193118

theorem expand_expression (x : ℝ) : (13 * x + 15) * (2 * x) = 26 * x^2 + 30 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1931_193118


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l1931_193175

/-- Jake's current weight in pounds -/
def jakes_weight : ℕ := 156

/-- Jake's weight after losing 20 pounds -/
def jakes_reduced_weight : ℕ := jakes_weight - 20

/-- Jake's sister's weight in pounds -/
def sisters_weight : ℕ := jakes_reduced_weight / 2

/-- The combined weight of Jake and his sister -/
def combined_weight : ℕ := jakes_weight + sisters_weight

theorem jake_and_sister_weight : combined_weight = 224 := by
  sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l1931_193175


namespace NUMINAMATH_CALUDE_max_men_with_all_attributes_l1931_193164

/-- Represents the population of men in the city with various attributes -/
structure CityPopulation where
  total : ℕ
  married : ℕ
  withTV : ℕ
  withRadio : ℕ
  withAC : ℕ
  withCar : ℕ
  withSmartphone : ℕ

/-- The given population data for the city -/
def cityData : CityPopulation := {
  total := 3000,
  married := 2300,
  withTV := 2100,
  withRadio := 2600,
  withAC := 1800,
  withCar := 2500,
  withSmartphone := 2200
}

/-- Theorem stating that the maximum number of men with all attributes is at most 1800 -/
theorem max_men_with_all_attributes (p : CityPopulation) (h : p = cityData) :
  ∃ n : ℕ, n ≤ 1800 ∧ n ≤ p.married ∧ n ≤ p.withTV ∧ n ≤ p.withRadio ∧
           n ≤ p.withAC ∧ n ≤ p.withCar ∧ n ≤ p.withSmartphone :=
by sorry

end NUMINAMATH_CALUDE_max_men_with_all_attributes_l1931_193164


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1931_193148

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 3 * c)) + (b / (8 * c + 4 * a)) + (9 * c / (3 * a + 2 * b)) ≥ 47 / 48 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1931_193148


namespace NUMINAMATH_CALUDE_hash_four_neg_three_l1931_193143

-- Define the # operation
def hash (x y : Int) : Int := x * (y + 2) + 2 * x * y

-- Theorem statement
theorem hash_four_neg_three : hash 4 (-3) = -28 := by
  sorry

end NUMINAMATH_CALUDE_hash_four_neg_three_l1931_193143


namespace NUMINAMATH_CALUDE_four_even_numbers_sum_100_l1931_193167

theorem four_even_numbers_sum_100 :
  ∃ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∃ (k₁ k₂ k₃ k₄ : ℕ), a = 2 * k₁ ∧ b = 2 * k₂ ∧ c = 2 * k₃ ∧ d = 2 * k₄) ∧
    a + b + c + d = 50 ∧
    c < 10 ∧ d < 10 ∧
    6 * a + 15 * b + 26 * c = 500 ∧
    ((a = 31 ∧ b = 14 ∧ c = 4 ∧ d = 1) ∨ (a = 24 ∧ b = 22 ∧ c = 1 ∧ d = 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_four_even_numbers_sum_100_l1931_193167


namespace NUMINAMATH_CALUDE_complex_subtraction_imaginary_part_l1931_193181

theorem complex_subtraction_imaginary_part : 
  (Complex.im ((2 + Complex.I) / (1 - Complex.I) - (2 - Complex.I) / (1 + Complex.I)) = 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_imaginary_part_l1931_193181
