import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2576_257613

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 40) 
  (eq2 : 3 * a + 4 * b = 38) : 
  a + b = 74 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2576_257613


namespace NUMINAMATH_CALUDE_tan_function_property_l2576_257660

/-- 
Given a function f(x) = a * tan(b * x) where a and b are positive constants,
if the function has a period of 2π/5 and passes through the point (π/10, 1),
then the product ab equals 5/2.
-/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + 2 * π / 5))) → 
  (a * Real.tan (b * π / 10) = 1) → 
  a * b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_function_property_l2576_257660


namespace NUMINAMATH_CALUDE_y_relationship_l2576_257691

/-- A linear function with slope -2 and y-intercept 5 -/
def f (x : ℝ) : ℝ := -2 * x + 5

/-- Theorem stating the relationship between y-values for specific x-values in the linear function f -/
theorem y_relationship (x₁ y₁ y₂ y₃ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f (x₁ - 2) = y₂) 
  (h3 : f (x₁ + 3) = y₃) : 
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_relationship_l2576_257691


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2576_257651

/-- Given a line with equation 3x + 6y = -24, prove that the slope of any parallel line is -1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x + 6 * y = -24) → (slope_of_parallel_line : ℝ) = -1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_line_slope_l2576_257651


namespace NUMINAMATH_CALUDE_converse_of_quadratic_eq_l2576_257639

theorem converse_of_quadratic_eq (x : ℝ) : x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0 := by sorry

end NUMINAMATH_CALUDE_converse_of_quadratic_eq_l2576_257639


namespace NUMINAMATH_CALUDE_cheaper_candy_price_l2576_257680

/-- Proves that the price of the cheaper candy is $2.00 per pound given the conditions of the mixture problem. -/
theorem cheaper_candy_price (total_weight : ℝ) (mixture_price : ℝ) (cheaper_weight : ℝ) (expensive_price : ℝ) 
  (h1 : total_weight = 80)
  (h2 : mixture_price = 2.20)
  (h3 : cheaper_weight = 64)
  (h4 : expensive_price = 3.00) :
  ∃ (cheaper_price : ℝ), 
    cheaper_price * cheaper_weight + expensive_price * (total_weight - cheaper_weight) = 
    mixture_price * total_weight ∧ cheaper_price = 2.00 := by
  sorry

end NUMINAMATH_CALUDE_cheaper_candy_price_l2576_257680


namespace NUMINAMATH_CALUDE_function_passes_through_point_one_one_l2576_257656

/-- The function f(x) = a^(x-1) always passes through the point (1, 1) for any a > 0 and a ≠ 1 -/
theorem function_passes_through_point_one_one (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1)
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_one_one_l2576_257656


namespace NUMINAMATH_CALUDE_chair_cost_l2576_257623

theorem chair_cost (total_cost : ℝ) (table_cost : ℝ) (num_chairs : ℕ) :
  total_cost = 135 →
  table_cost = 55 →
  num_chairs = 4 →
  ∃ (chair_cost : ℝ),
    chair_cost * num_chairs = total_cost - table_cost ∧
    chair_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_chair_cost_l2576_257623


namespace NUMINAMATH_CALUDE_peanut_ball_probability_l2576_257645

/-- The probability of selecting at least one peanut filling glutinous rice ball -/
theorem peanut_ball_probability : 
  let total_balls : ℕ := 6
  let peanut_balls : ℕ := 2
  let selected_balls : ℕ := 2
  Nat.choose total_balls selected_balls ≠ 0 →
  (Nat.choose peanut_balls selected_balls + 
   peanut_balls * (total_balls - peanut_balls)) / 
  Nat.choose total_balls selected_balls = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_peanut_ball_probability_l2576_257645


namespace NUMINAMATH_CALUDE_racing_game_cost_l2576_257621

/-- Given that Joan spent $9.43 on video games in total and
    purchased a basketball game for $5.2, prove that the
    cost of the racing game is $9.43 - $5.2. -/
theorem racing_game_cost (total_spent : ℝ) (basketball_cost : ℝ)
    (h1 : total_spent = 9.43)
    (h2 : basketball_cost = 5.2) :
    total_spent - basketball_cost = 9.43 - 5.2 := by
  sorry

end NUMINAMATH_CALUDE_racing_game_cost_l2576_257621


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l2576_257692

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℝ := 21

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℝ := green_pill_cost - 3

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 819

theorem green_pill_cost_proof :
  green_pill_cost = 21 ∧
  pink_pill_cost = green_pill_cost - 3 ∧
  treatment_days = 21 ∧
  total_cost = 819 ∧
  treatment_days * (green_pill_cost + pink_pill_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_green_pill_cost_proof_l2576_257692


namespace NUMINAMATH_CALUDE_library_books_remaining_l2576_257683

theorem library_books_remaining (initial_books : ℕ) (given_away : ℕ) (donated : ℕ) : 
  initial_books = 125 → given_away = 42 → donated = 31 → 
  initial_books - given_away - donated = 52 := by
sorry

end NUMINAMATH_CALUDE_library_books_remaining_l2576_257683


namespace NUMINAMATH_CALUDE_village_population_theorem_l2576_257678

theorem village_population_theorem (total_population : ℕ) 
  (h1 : total_population = 800) 
  (h2 : ∃ (part : ℕ), 4 * part = total_population) 
  (h3 : ∃ (male_population : ℕ), male_population = 2 * (total_population / 4)) :
  ∃ (male_population : ℕ), male_population = 400 := by
sorry

end NUMINAMATH_CALUDE_village_population_theorem_l2576_257678


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2576_257659

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧
  ∀ x y : ℝ, (x - y) * (f (f (x^2)) - f (f (y^2))) = (f x + f y) * (f x - f y)^2

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = cx for some constant c -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2576_257659


namespace NUMINAMATH_CALUDE_shrimp_earnings_l2576_257662

theorem shrimp_earnings (victor_shrimp : ℕ) (austin_less : ℕ) (price : ℚ) (tails_per_set : ℕ) : 
  victor_shrimp = 26 →
  austin_less = 8 →
  price = 7 →
  tails_per_set = 11 →
  let austin_shrimp := victor_shrimp - austin_less
  let total_victor_austin := victor_shrimp + austin_shrimp
  let brian_shrimp := total_victor_austin / 2
  let total_shrimp := victor_shrimp + austin_shrimp + brian_shrimp
  let sets_sold := total_shrimp / tails_per_set
  let total_earnings := price * (sets_sold : ℚ)
  let each_boy_earnings := total_earnings / 3
  each_boy_earnings = 14 := by
sorry

end NUMINAMATH_CALUDE_shrimp_earnings_l2576_257662


namespace NUMINAMATH_CALUDE_son_age_l2576_257684

theorem son_age (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_son_age_l2576_257684


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_2_range_of_m_for_inequality_l2576_257625

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - 2*m| - |x + m|

-- Part 1
theorem solution_set_when_m_is_2 :
  let m : ℝ := 2
  ∀ x : ℝ, f m x ≥ 1 ↔ -2 < x ∧ x ≤ 1/2 :=
sorry

-- Part 2
theorem range_of_m_for_inequality :
  ∀ m : ℝ, m > 0 →
  (∀ x t : ℝ, f m x ≤ |t + 3| + |t - 2|) ↔
  (0 < m ∧ m ≤ 5/3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_2_range_of_m_for_inequality_l2576_257625


namespace NUMINAMATH_CALUDE_brians_age_in_eight_years_l2576_257672

/-- Given that Christian is twice as old as Brian and Christian will be 72 years old in eight years,
    prove that Brian will be 40 years old in eight years. -/
theorem brians_age_in_eight_years (christian_age : ℕ) (brian_age : ℕ) : 
  christian_age = 2 * brian_age →
  christian_age + 8 = 72 →
  brian_age + 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_brians_age_in_eight_years_l2576_257672


namespace NUMINAMATH_CALUDE_no_roots_implies_negative_a_l2576_257607

theorem no_roots_implies_negative_a :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x - 1/x + a ≠ 0) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_roots_implies_negative_a_l2576_257607


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2576_257606

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2576_257606


namespace NUMINAMATH_CALUDE_bucket_water_volume_l2576_257637

theorem bucket_water_volume (initial_volume : ℝ) (additional_volume : ℝ) : 
  initial_volume = 2 → 
  additional_volume = 460 → 
  (initial_volume * 1000 + additional_volume : ℝ) = 2460 := by
sorry

end NUMINAMATH_CALUDE_bucket_water_volume_l2576_257637


namespace NUMINAMATH_CALUDE_equation_solutions_count_l2576_257667

open Real

theorem equation_solutions_count (a : ℝ) (h : a < 0) :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x ∈ s, -π < x ∧ x < π ∧ 
    (a - 1) * (sin (2 * x) + cos x) + (a - 1) * (sin x - cos (2 * x)) = 0) ∧
  (∀ x, -π < x → x < π → 
    (a - 1) * (sin (2 * x) + cos x) + (a - 1) * (sin x - cos (2 * x)) = 0 → x ∈ s) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l2576_257667


namespace NUMINAMATH_CALUDE_race_catch_up_time_l2576_257652

/-- Proves that Nicky runs for 30 seconds before Cristina catches up in a 300-meter race --/
theorem race_catch_up_time 
  (race_distance : ℝ) 
  (head_start : ℝ) 
  (cristina_speed : ℝ) 
  (nicky_speed : ℝ) 
  (h1 : race_distance = 300)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) : 
  ∃ (t : ℝ), t = 30 ∧ 
  cristina_speed * (t - head_start) = nicky_speed * t := by
  sorry


end NUMINAMATH_CALUDE_race_catch_up_time_l2576_257652


namespace NUMINAMATH_CALUDE_tree_break_height_l2576_257608

/-- Given a tree of height 36 meters that breaks and falls across a road of width 12 meters,
    touching the opposite edge, the height at which the tree broke is 16 meters. -/
theorem tree_break_height :
  ∀ (h : ℝ), 
  h > 0 →
  h < 36 →
  (36 - h)^2 = h^2 + 12^2 →
  h = 16 :=
by sorry

end NUMINAMATH_CALUDE_tree_break_height_l2576_257608


namespace NUMINAMATH_CALUDE_gcd_repeated_digits_l2576_257615

def repeat_digit (n : ℕ) : ℕ := n + 1000 * n + 1000000 * n

theorem gcd_repeated_digits :
  ∃ (g : ℕ), g > 0 ∧ 
  (∀ (n : ℕ), 100 ≤ n → n < 1000 → g ∣ repeat_digit n) ∧
  (∀ (d : ℕ), d > 0 → 
    (∀ (n : ℕ), 100 ≤ n → n < 1000 → d ∣ repeat_digit n) → 
    d ∣ g) ∧
  g = 1001001 := by
sorry

#eval 1001001

end NUMINAMATH_CALUDE_gcd_repeated_digits_l2576_257615


namespace NUMINAMATH_CALUDE_triangle_side_length_l2576_257657

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = 10 →
  a / Real.sin A = b / Real.sin B →
  b = 5 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2576_257657


namespace NUMINAMATH_CALUDE_circle_with_chords_theorem_l2576_257635

/-- Represents a circle with two intersecting chords --/
structure CircleWithChords where
  radius : ℝ
  chord_length : ℝ
  intersection_distance : ℝ

/-- Represents the area of a region in the form mπ - n√d --/
structure RegionArea where
  m : ℕ
  n : ℕ
  d : ℕ

/-- Checks if a number is square-free (not divisible by the square of any prime) --/
def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p ∣ n) → p = 1

/-- Main theorem about the circle with intersecting chords --/
theorem circle_with_chords_theorem (circle : CircleWithChords) 
  (h1 : circle.radius = 36)
  (h2 : circle.chord_length = 66)
  (h3 : circle.intersection_distance = 12) :
  ∃ (area : RegionArea), 
    (area.m : ℝ) * Real.pi - (area.n : ℝ) * Real.sqrt (area.d : ℝ) > 0 ∧
    is_square_free area.d ∧
    area.m + area.n + area.d = 378 :=
  sorry

end NUMINAMATH_CALUDE_circle_with_chords_theorem_l2576_257635


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2576_257688

theorem sum_of_roots_quadratic (x : ℝ) : 
  (4 * x + 3) * (3 * x - 8) = 0 → 
  ∃ x₁ x₂ : ℝ, x₁ + x₂ = 23 / 12 ∧ 
    ((4 * x₁ + 3) * (3 * x₁ - 8) = 0) ∧ 
    ((4 * x₂ + 3) * (3 * x₂ - 8) = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2576_257688


namespace NUMINAMATH_CALUDE_floor_function_unique_l2576_257634

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the conditions
def Condition1 (f : RealFunction) : Prop :=
  ∀ x y : ℝ, f x + f y + 1 ≥ f (x + y) ∧ f (x + y) ≥ f x + f y

def Condition2 (f : RealFunction) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f 0 ≥ f x

def Condition3 (f : RealFunction) : Prop :=
  f (-1) = -1 ∧ f 1 = 1

-- Main theorem
theorem floor_function_unique (f : RealFunction)
  (h1 : Condition1 f) (h2 : Condition2 f) (h3 : Condition3 f) :
  ∀ x : ℝ, f x = ⌊x⌋ := by sorry

end NUMINAMATH_CALUDE_floor_function_unique_l2576_257634


namespace NUMINAMATH_CALUDE_bryden_receives_correct_amount_l2576_257666

/-- The amount Bryden receives for his state quarters -/
def bryden_amount (num_quarters : ℕ) (face_value : ℚ) (percentage : ℕ) (bonus_per_five : ℚ) : ℚ :=
  let base_amount := (num_quarters : ℚ) * face_value * (percentage : ℚ) / 100
  let num_bonuses := num_quarters / 5
  base_amount + (num_bonuses : ℚ) * bonus_per_five

/-- Theorem stating that Bryden will receive $45.75 for his seven state quarters -/
theorem bryden_receives_correct_amount :
  bryden_amount 7 0.25 2500 2 = 45.75 := by
  sorry

end NUMINAMATH_CALUDE_bryden_receives_correct_amount_l2576_257666


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2009_l2576_257674

/-- Given an arithmetic sequence {a_n} with common difference d and a_k, 
    this function returns a_n -/
def arithmeticSequence (d : ℤ) (k : ℕ) (a_k : ℤ) (n : ℕ) : ℤ :=
  a_k + d * (n - k)

theorem arithmetic_sequence_2009 :
  let d := 2
  let k := 2007
  let a_k := 2007
  let n := 2009
  arithmeticSequence d k a_k n = 2011 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2009_l2576_257674


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2576_257626

open Real

theorem inequality_solution_set (x : ℝ) :
  x ∈ Set.Ioo (-1 : ℝ) 1 →
  (abs (sin x) + abs (log (1 - x^2)) > abs (sin x + log (1 - x^2))) ↔ x ∈ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2576_257626


namespace NUMINAMATH_CALUDE_cylinder_volume_theorem_l2576_257671

/-- Represents the dimensions of a rectangle formed by unrolling a cylinder's lateral surface -/
structure UnrolledCylinder where
  side1 : ℝ
  side2 : ℝ

/-- Calculates the possible volumes of a cylinder given its unrolled lateral surface dimensions -/
def possible_cylinder_volumes (uc : UnrolledCylinder) : Set ℝ :=
  let v1 := (uc.side1 / (2 * Real.pi)) ^ 2 * Real.pi * uc.side2
  let v2 := (uc.side2 / (2 * Real.pi)) ^ 2 * Real.pi * uc.side1
  {v1, v2}

/-- Theorem stating that a cylinder with unrolled lateral surface of 8π and 4π has volume 32π² or 64π² -/
theorem cylinder_volume_theorem (uc : UnrolledCylinder) 
    (h1 : uc.side1 = 8 * Real.pi) (h2 : uc.side2 = 4 * Real.pi) : 
    possible_cylinder_volumes uc = {32 * Real.pi ^ 2, 64 * Real.pi ^ 2} := by
  sorry

#check cylinder_volume_theorem

end NUMINAMATH_CALUDE_cylinder_volume_theorem_l2576_257671


namespace NUMINAMATH_CALUDE_math_club_team_selection_l2576_257698

def boys : ℕ := 10
def girls : ℕ := 12
def team_size : ℕ := 8
def boys_in_team : ℕ := 5
def girls_in_team : ℕ := 3

theorem math_club_team_selection :
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 55440 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l2576_257698


namespace NUMINAMATH_CALUDE_function_property_l2576_257687

theorem function_property (f : ℕ+ → ℕ) 
  (h1 : ∀ (x y : ℕ+), f (x * y) = f x + f y)
  (h2 : f 10 = 16)
  (h3 : f 40 = 26)
  (h4 : f 8 = 12) :
  f 1000 = 48 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2576_257687


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2576_257612

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 3*X + 1 = (X - 3)^2 * q + (81*X - 161) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2576_257612


namespace NUMINAMATH_CALUDE_fans_per_bleacher_set_l2576_257617

theorem fans_per_bleacher_set (total_fans : ℕ) (num_bleacher_sets : ℕ) 
  (h1 : total_fans = 2436) (h2 : num_bleacher_sets = 3) :
  total_fans / num_bleacher_sets = 812 := by
  sorry

end NUMINAMATH_CALUDE_fans_per_bleacher_set_l2576_257617


namespace NUMINAMATH_CALUDE_unique_solution_system_l2576_257614

theorem unique_solution_system (x y z t : ℝ) :
  (x * y + z + t = 1) ∧
  (y * z + t + x = 3) ∧
  (z * t + x + y = -1) ∧
  (t * x + y + z = 1) →
  (x = 1 ∧ y = 0 ∧ z = -1 ∧ t = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2576_257614


namespace NUMINAMATH_CALUDE_ten_caterpillars_left_l2576_257620

/-- The number of caterpillars left on a tree after some changes -/
def caterpillars_left (initial : ℕ) (hatched : ℕ) (left : ℕ) : ℕ :=
  initial + hatched - left

/-- Theorem: Given the initial conditions, prove that 10 caterpillars are left on the tree -/
theorem ten_caterpillars_left : 
  caterpillars_left 14 4 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_caterpillars_left_l2576_257620


namespace NUMINAMATH_CALUDE_inequality_and_system_solution_l2576_257673

theorem inequality_and_system_solution :
  (∀ x : ℝ, 2 * (-3 + x) > 3 * (x + 2) ↔ x < -12) ∧
  (∀ x : ℝ, (1/2 * (x + 1) < 2 ∧ (x + 2)/2 ≥ (x + 3)/3) ↔ 0 ≤ x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_system_solution_l2576_257673


namespace NUMINAMATH_CALUDE_cricketer_average_score_l2576_257653

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (three_match_avg : ℝ) 
  (total_avg : ℝ) 
  (h1 : total_matches = 5)
  (h2 : three_match_avg = 40)
  (h3 : total_avg = 36) :
  (5 * total_avg - 3 * three_match_avg) / 2 = 30 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l2576_257653


namespace NUMINAMATH_CALUDE_exists_spanish_couple_l2576_257663

-- Define the set S
def S : Set ℝ := {x | ∃ (a b : ℕ), x = (a - 1) / b}

-- Define the property of being strictly increasing
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the Spanish Couple property
def SpanishCouple (f g : ℝ → ℝ) : Prop :=
  (∀ x ∈ S, f x ∈ S) ∧
  (∀ x ∈ S, g x ∈ S) ∧
  StrictlyIncreasing f ∧
  StrictlyIncreasing g ∧
  ∀ x ∈ S, f (g (g x)) < g (f x)

-- Theorem statement
theorem exists_spanish_couple : ∃ f g, SpanishCouple f g := by
  sorry

end NUMINAMATH_CALUDE_exists_spanish_couple_l2576_257663


namespace NUMINAMATH_CALUDE_calculation_problem_1_calculation_problem_2_l2576_257694

-- Question 1
theorem calculation_problem_1 :
  (-1/4)⁻¹ - |Real.sqrt 3 - 1| + 3 * Real.tan (30 * π / 180) + (2017 - π) = -2 := by sorry

-- Question 2
theorem calculation_problem_2 (x : ℝ) (h : x = 2) :
  (2 * x^2) / (x^2 - 2*x + 1) / ((2*x + 1) / (x + 1) + 1 / (x - 1)) = 3 := by sorry

end NUMINAMATH_CALUDE_calculation_problem_1_calculation_problem_2_l2576_257694


namespace NUMINAMATH_CALUDE_freeway_to_traffic_ratio_l2576_257661

def total_time : ℝ := 10
def traffic_time : ℝ := 2

theorem freeway_to_traffic_ratio :
  (total_time - traffic_time) / traffic_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_freeway_to_traffic_ratio_l2576_257661


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_squared_l2576_257641

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ

/-- The conditions of the trapezoid as described in the problem -/
def trapezoid_conditions (t : Trapezoid) : Prop :=
  -- The longer base is 150 units longer than the shorter base
  ∃ (longer_base : ℝ), longer_base = t.shorter_base + 150
  -- The midline divides the trapezoid into regions with area ratio 3:4
  ∧ (t.shorter_base + t.shorter_base + 75) / (t.shorter_base + 75 + t.shorter_base + 150) = 3 / 4
  -- t.equal_area_segment divides the trapezoid into two equal-area regions
  ∧ ∃ (h₁ : ℝ), 2 * (1/2 * h₁ * (t.shorter_base + t.equal_area_segment)) = 
                 1/2 * t.height * (t.shorter_base + t.shorter_base + 150)

/-- The theorem to be proved -/
theorem trapezoid_segment_length_squared (t : Trapezoid) 
  (h : trapezoid_conditions t) : 
  ⌊t.equal_area_segment^2 / 150⌋ = 300 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_squared_l2576_257641


namespace NUMINAMATH_CALUDE_solution_implies_m_equals_one_l2576_257665

theorem solution_implies_m_equals_one (x y m : ℝ) : 
  x = 2 → y = -1 → m * x - y = 3 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_equals_one_l2576_257665


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2576_257628

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 5 * x) = 3 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2576_257628


namespace NUMINAMATH_CALUDE_sufficient_condition_for_sum_of_roots_l2576_257664

theorem sufficient_condition_for_sum_of_roots 
  (a b c x₁ x₂ : ℝ) (ha : a ≠ 0) 
  (hroots : x₁ * x₁ + a⁻¹ * b * x₁ + a⁻¹ * c = 0 ∧ 
            x₂ * x₂ + a⁻¹ * b * x₂ + a⁻¹ * c = 0) :
  x₁ + x₂ = -b / a := by
  sorry


end NUMINAMATH_CALUDE_sufficient_condition_for_sum_of_roots_l2576_257664


namespace NUMINAMATH_CALUDE_problem_statement_l2576_257690

theorem problem_statement (x y : ℝ) (h1 : x - y = -2) (h2 : x * y = 3) :
  x^2 * y - x * y^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2576_257690


namespace NUMINAMATH_CALUDE_smallest_expression_l2576_257654

theorem smallest_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) ≤ min ((a + b) / 2) (min (Real.sqrt (a * b)) (Real.sqrt ((a^2 + b^2) / 2))) :=
sorry

end NUMINAMATH_CALUDE_smallest_expression_l2576_257654


namespace NUMINAMATH_CALUDE_equation_solution_l2576_257699

theorem equation_solution :
  ∃ x : ℚ, (4 * x^2 + 3 * x + 2) / (x + 2) = 4 * x + 3 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2576_257699


namespace NUMINAMATH_CALUDE_cd_purchase_remaining_money_l2576_257676

theorem cd_purchase_remaining_money 
  (total_money : ℚ) 
  (total_cds : ℚ) 
  (cd_price : ℚ) 
  (h1 : cd_price > 0) 
  (h2 : total_money > 0) 
  (h3 : total_cds > 0) 
  (h4 : total_money / 5 = cd_price * total_cds / 3) :
  total_money - cd_price * total_cds = 2 * total_money / 5 := by
sorry

end NUMINAMATH_CALUDE_cd_purchase_remaining_money_l2576_257676


namespace NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l2576_257618

theorem infinitely_many_primes_mod_3_eq_2 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l2576_257618


namespace NUMINAMATH_CALUDE_percentage_problem_l2576_257679

theorem percentage_problem (x : ℝ) (h : (32 / 100) * x = 115.2) : x = 360 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2576_257679


namespace NUMINAMATH_CALUDE_jasmine_cut_length_l2576_257644

def ribbon_length : ℕ := 10
def janice_cut_length : ℕ := 2

theorem jasmine_cut_length :
  ∀ (jasmine_cut : ℕ),
    jasmine_cut ≠ janice_cut_length →
    ribbon_length % jasmine_cut = 0 →
    ribbon_length % janice_cut_length = 0 →
    jasmine_cut = 5 :=
by sorry

end NUMINAMATH_CALUDE_jasmine_cut_length_l2576_257644


namespace NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l2576_257638

def number_of_divisors (n : ℕ) : ℕ :=
  (Nat.factors n).map (· + 1) |>.prod

def is_smallest_with_2020_divisors (n : ℕ) : Prop :=
  number_of_divisors n = 2020 ∧
  ∀ m < n, number_of_divisors m ≠ 2020

theorem smallest_number_with_2020_divisors :
  is_smallest_with_2020_divisors (2^100 * 3^4 * 5 * 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l2576_257638


namespace NUMINAMATH_CALUDE_abs_equation_solution_l2576_257686

theorem abs_equation_solution :
  ∀ x : ℝ, |2*x + 6| = 3*x - 1 ↔ x = 7 := by sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l2576_257686


namespace NUMINAMATH_CALUDE_vodka_alcohol_consumption_l2576_257602

/-- Calculates the amount of pure alcohol consumed by one person when splitting vodka shots. -/
theorem vodka_alcohol_consumption
  (total_shots : ℕ)
  (ounces_per_shot : ℚ)
  (alcohol_percentage : ℚ)
  (h1 : total_shots = 8)
  (h2 : ounces_per_shot = 3/2)
  (h3 : alcohol_percentage = 1/2) :
  (((total_shots : ℚ) / 2) * ounces_per_shot) * alcohol_percentage = 3 := by
  sorry

end NUMINAMATH_CALUDE_vodka_alcohol_consumption_l2576_257602


namespace NUMINAMATH_CALUDE_certain_number_problem_l2576_257619

theorem certain_number_problem (x y : ℤ) : x = 15 ∧ 2 * x = (y - x) + 19 → y = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2576_257619


namespace NUMINAMATH_CALUDE_opposite_of_negative_nine_l2576_257695

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_nine : opposite (-9) = 9 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_nine_l2576_257695


namespace NUMINAMATH_CALUDE_part_one_part_two_l2576_257630

/-- Definition of a difference solution equation -/
def is_difference_solution_equation (a b : ℝ) : Prop :=
  (b / a) = b - a

/-- Part 1: Prove that 3x = 4.5 is a difference solution equation -/
theorem part_one : is_difference_solution_equation 3 4.5 := by sorry

/-- Part 2: Prove that 5x - m = 1 is a difference solution equation when m = 21/4 -/
theorem part_two : is_difference_solution_equation 5 ((21/4) + 1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2576_257630


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_city_stratified_sampling_l2576_257627

/-- Represents the types of schools in the city -/
inductive SchoolType
  | University
  | MiddleSchool
  | PrimarySchool

/-- Represents the distribution of schools in the city -/
structure SchoolDistribution where
  total : ℕ
  universities : ℕ
  middleSchools : ℕ
  primarySchools : ℕ

/-- Represents the sample size and distribution in stratified sampling -/
structure StratifiedSample where
  sampleSize : ℕ
  universitiesSample : ℕ
  middleSchoolsSample : ℕ
  primarySchoolsSample : ℕ

def citySchools : SchoolDistribution :=
  { total := 500
  , universities := 10
  , middleSchools := 200
  , primarySchools := 290 }

def sampleSize : ℕ := 50

theorem stratified_sampling_theorem (d : SchoolDistribution) (s : ℕ) :
  d.total = d.universities + d.middleSchools + d.primarySchools →
  s ≤ d.total →
  ∃ (sample : StratifiedSample),
    sample.sampleSize = s ∧
    sample.universitiesSample = (s * d.universities) / d.total ∧
    sample.middleSchoolsSample = (s * d.middleSchools) / d.total ∧
    sample.primarySchoolsSample = (s * d.primarySchools) / d.total ∧
    sample.sampleSize = sample.universitiesSample + sample.middleSchoolsSample + sample.primarySchoolsSample :=
by sorry

theorem city_stratified_sampling :
  ∃ (sample : StratifiedSample),
    sample.sampleSize = sampleSize ∧
    sample.universitiesSample = 1 ∧
    sample.middleSchoolsSample = 20 ∧
    sample.primarySchoolsSample = 29 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_city_stratified_sampling_l2576_257627


namespace NUMINAMATH_CALUDE_random_course_selection_probability_l2576_257616

theorem random_course_selection_probability 
  (courses : Finset String) 
  (h1 : courses.card = 4) 
  (h2 : courses.Nonempty) 
  (selected_course : String) 
  (h3 : selected_course ∈ courses) :
  (Finset.filter (· = selected_course) courses).card / courses.card = (1 : ℚ) / 4 :=
sorry

end NUMINAMATH_CALUDE_random_course_selection_probability_l2576_257616


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l2576_257632

theorem sum_of_coefficients_is_one : 
  let p (x : ℝ) := 3*(x^8 - 2*x^5 + 4*x^3 - 6) - 5*(2*x^4 + 3*x - 7) + 6*(x^6 - x^2 + 1)
  p 1 = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l2576_257632


namespace NUMINAMATH_CALUDE_hawkeye_remaining_money_l2576_257629

/-- Calculates the remaining money after battery charges. -/
def remaining_money (cost_per_charge : ℚ) (num_charges : ℕ) (budget : ℚ) : ℚ :=
  budget - (cost_per_charge * num_charges)

/-- Theorem: Given the specified conditions, the remaining money is $6. -/
theorem hawkeye_remaining_money :
  remaining_money (35/10) 4 20 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_remaining_money_l2576_257629


namespace NUMINAMATH_CALUDE_monomial_type_sum_l2576_257609

/-- Two monomials are of the same type if they have the same variables raised to the same powers -/
def same_type_monomials (m n : ℕ) : Prop :=
  m - 1 = 2 ∧ n + 1 = 2

theorem monomial_type_sum (m n : ℕ) :
  same_type_monomials m n → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_monomial_type_sum_l2576_257609


namespace NUMINAMATH_CALUDE_max_a_for_increasing_f_l2576_257675

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

-- State the theorem
theorem max_a_for_increasing_f :
  ∃ (a : ℝ), a = 1 ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ a → f x₁ < f x₂) ∧
  (∀ a' : ℝ, a' > a → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ a' ∧ f x₁ ≥ f x₂) :=
sorry

end NUMINAMATH_CALUDE_max_a_for_increasing_f_l2576_257675


namespace NUMINAMATH_CALUDE_polar_to_circle_l2576_257636

/-- The polar equation r = 1 / (1 - sin θ) represents a circle. -/
theorem polar_to_circle : ∃ (h k R : ℝ), ∀ (x y : ℝ),
  (∃ (r θ : ℝ), r = 1 / (1 - Real.sin θ) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  (x - h)^2 + (y - k)^2 = R^2 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_circle_l2576_257636


namespace NUMINAMATH_CALUDE_haley_trees_l2576_257697

theorem haley_trees (initial_trees : ℕ) : 
  (((initial_trees - 5) - 8) - 3 = 12) → initial_trees = 28 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_l2576_257697


namespace NUMINAMATH_CALUDE_smaller_factor_of_4582_l2576_257696

theorem smaller_factor_of_4582 :
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    a * b = 4582 ∧
    (∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 4582 → min x y = 21) :=
by sorry

end NUMINAMATH_CALUDE_smaller_factor_of_4582_l2576_257696


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2576_257601

/-- Proves that a boat's speed in still water is 2.5 km/hr given its downstream and upstream travel times -/
theorem boat_speed_in_still_water 
  (distance : ℝ) 
  (downstream_time upstream_time : ℝ) 
  (h1 : distance = 10) 
  (h2 : downstream_time = 3) 
  (h3 : upstream_time = 6) : 
  ∃ (boat_speed : ℝ), boat_speed = 2.5 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2576_257601


namespace NUMINAMATH_CALUDE_cereal_cost_l2576_257689

theorem cereal_cost (total_spent groceries_cost milk_cost cereal_boxes banana_cost banana_count
                     apple_cost apple_count cookie_cost_multiplier cookie_boxes : ℚ) :
  groceries_cost = 25 →
  milk_cost = 3 →
  cereal_boxes = 2 →
  banana_cost = 0.25 →
  banana_count = 4 →
  apple_cost = 0.5 →
  apple_count = 4 →
  cookie_cost_multiplier = 2 →
  cookie_boxes = 2 →
  (groceries_cost - (milk_cost + banana_cost * banana_count + apple_cost * apple_count +
   cookie_cost_multiplier * milk_cost * cookie_boxes)) / cereal_boxes = 3.5 := by
sorry

end NUMINAMATH_CALUDE_cereal_cost_l2576_257689


namespace NUMINAMATH_CALUDE_barbara_butcher_cost_l2576_257647

/-- The total cost of Barbara's purchase at the butcher's -/
def total_cost (steak_weight : ℝ) (steak_price : ℝ) (chicken_weight : ℝ) (chicken_price : ℝ) : ℝ :=
  steak_weight * steak_price + chicken_weight * chicken_price

/-- Theorem: Barbara's total cost at the butcher's is $79.50 -/
theorem barbara_butcher_cost : 
  total_cost 4.5 15 1.5 8 = 79.5 := by
  sorry

#eval total_cost 4.5 15 1.5 8

end NUMINAMATH_CALUDE_barbara_butcher_cost_l2576_257647


namespace NUMINAMATH_CALUDE_smallest_product_sum_l2576_257603

def digits : List Nat := [3, 4, 5, 6, 7]

def is_valid_configuration (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product_sum (a b c d e : Nat) : Nat :=
  (10 * a + b) * (10 * c + d) + e * (10 * a + b)

theorem smallest_product_sum :
  ∀ a b c d e : Nat,
    is_valid_configuration a b c d e →
    product_sum a b c d e ≥ 2448 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_sum_l2576_257603


namespace NUMINAMATH_CALUDE_not_always_equal_l2576_257642

theorem not_always_equal (a b c : ℝ) (h1 : a = b - c) :
  (a - b/2)^2 = (c - b/2)^2 → a = c ∨ a + c = b := by sorry

end NUMINAMATH_CALUDE_not_always_equal_l2576_257642


namespace NUMINAMATH_CALUDE_parabola_directrix_l2576_257681

/-- The equation of the directrix of a parabola with equation x² = 4y and focus at (0, 1) -/
theorem parabola_directrix : ∃ (l : ℝ → ℝ), 
  (∀ x y : ℝ, x^2 = 4*y → (∀ t : ℝ, (x - 0)^2 + (y - 1)^2 = (x - t)^2 + (y - l t)^2)) → 
  (∀ t : ℝ, l t = -1) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2576_257681


namespace NUMINAMATH_CALUDE_velocity_zero_at_two_l2576_257631

-- Define the motion equation
def s (t : ℝ) : ℝ := -4 * t^3 + 48 * t

-- Define the velocity function (derivative of s)
def v (t : ℝ) : ℝ := -12 * t^2 + 48

-- Theorem stating that the positive time when velocity is zero is 2
theorem velocity_zero_at_two :
  ∃ (t : ℝ), t > 0 ∧ v t = 0 ∧ t = 2 := by
  sorry

end NUMINAMATH_CALUDE_velocity_zero_at_two_l2576_257631


namespace NUMINAMATH_CALUDE_bill_experience_l2576_257655

/-- Represents the work experience of a person -/
structure Experience where
  current : ℕ
  fiveYearsAgo : ℕ

/-- The problem setup -/
def libraryProblem : Prop := ∃ (bill joan : Experience),
  -- Bill's current age
  40 = bill.current + bill.fiveYearsAgo
  -- Joan's current age
  ∧ 50 = joan.current + joan.fiveYearsAgo
  -- 5 years ago, Joan had 3 times as much experience as Bill
  ∧ joan.fiveYearsAgo = 3 * bill.fiveYearsAgo
  -- Now, Joan has twice as much experience as Bill
  ∧ joan.current = 2 * bill.current
  -- Bill's current experience is 10 years
  ∧ bill.current = 10

/-- The theorem to prove -/
theorem bill_experience : libraryProblem := by sorry

end NUMINAMATH_CALUDE_bill_experience_l2576_257655


namespace NUMINAMATH_CALUDE_chord_squares_difference_l2576_257600

/-- Given a circle with a chord at distance h from the center, and two squares inscribed in the 
segments subtended by the chord (with two adjacent vertices on the arc and two on the chord or 
its extension), the difference in the side lengths of these squares is 8h/5. -/
theorem chord_squares_difference (h : ℝ) (h_pos : h > 0) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_chord_squares_difference_l2576_257600


namespace NUMINAMATH_CALUDE_valid_solutions_are_only_solutions_l2576_257669

/-- A structure representing a solution to the system of equations -/
structure Solution :=
  (x y z t : ℕ)

/-- The set of all valid solutions -/
def valid_solutions : Set Solution :=
  { ⟨1,1,2,3⟩, ⟨3,2,1,1⟩, ⟨4,1,3,1⟩, ⟨1,3,4,1⟩ }

/-- Predicate to check if a solution satisfies the equations -/
def satisfies_equations (s : Solution) : Prop :=
  ∃ a b : ℕ,
    s.x^2 + s.y^2 = a ∧
    s.z^2 + s.t^2 = b ∧
    (s.x^2 + s.t^2) * (s.z^2 + s.y^2) = 50

/-- Theorem stating that the valid solutions are the only ones satisfying the equations -/
theorem valid_solutions_are_only_solutions :
  ∀ s : Solution, satisfies_equations s ↔ s ∈ valid_solutions :=
sorry

end NUMINAMATH_CALUDE_valid_solutions_are_only_solutions_l2576_257669


namespace NUMINAMATH_CALUDE_percent_relation_l2576_257640

theorem percent_relation (P Q : ℝ) (h : (1/2) * P = (1/5) * Q) :
  P = (2/5) * Q := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2576_257640


namespace NUMINAMATH_CALUDE_negation_of_statement_l2576_257605

theorem negation_of_statement :
  ¬(∀ x : ℝ, x ≠ 0 → x^2 - 4 > 0) ↔ 
  (∃ x : ℝ, x ≠ 0 ∧ x^2 - 4 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_statement_l2576_257605


namespace NUMINAMATH_CALUDE_factorization_equality_l2576_257646

theorem factorization_equality (x : ℝ) :
  (4 * x^3 + 100 * x^2 - 28) - (-9 * x^3 + 2 * x^2 - 28) = 13 * x^2 * (x + 7) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2576_257646


namespace NUMINAMATH_CALUDE_trig_problem_l2576_257610

theorem trig_problem (α β : ℝ) 
  (h1 : Real.sin α - Real.sin β = -1/3)
  (h2 : Real.cos α - Real.cos β = 1/2)
  (h3 : Real.tan (α + β) = 2/5)
  (h4 : Real.tan (β - π/4) = 1/4) :
  Real.cos (α - β) = 59/72 ∧ Real.tan (α + π/4) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l2576_257610


namespace NUMINAMATH_CALUDE_negation_of_forall_even_square_plus_n_l2576_257643

theorem negation_of_forall_even_square_plus_n :
  (¬ ∀ n : ℕ, Even (n^2 + n)) ↔ (∃ n : ℕ, ¬ Even (n^2 + n)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_even_square_plus_n_l2576_257643


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2576_257650

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 
  a^2 - 3*a + 1 - a^2 + 6*a - 7 = 3*a - 6 := by sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℝ) : 
  (3*m^2*n - 5*m*n) - 3*(4*m^2*n - 5*m*n) = -9*m^2*n + 10*m*n := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2576_257650


namespace NUMINAMATH_CALUDE_equidistant_function_l2576_257648

def f (a b : ℝ) (z : ℂ) : ℂ := (a + b * Complex.I) * z

theorem equidistant_function (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_equidistant : ∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z - 1))
  (h_norm : Complex.abs (a + b * Complex.I) = 5) :
  b^2 = 99/4 := by
sorry

end NUMINAMATH_CALUDE_equidistant_function_l2576_257648


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2576_257611

theorem algebraic_expression_value (y : ℝ) : 
  3 * y^2 - 2 * y + 6 = 8 → (3/2) * y^2 - y + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2576_257611


namespace NUMINAMATH_CALUDE_common_solution_proof_l2576_257649

theorem common_solution_proof :
  let x : ℝ := Real.rpow 2 (1/3) - 2
  let y : ℝ := Real.rpow 2 (2/3)
  (y = (x + 2)^2) ∧ (x * y + y = 2) := by
  sorry

end NUMINAMATH_CALUDE_common_solution_proof_l2576_257649


namespace NUMINAMATH_CALUDE_min_fence_length_l2576_257670

theorem min_fence_length (area : ℝ) (h : area = 64) :
  ∃ (length width : ℝ), length > 0 ∧ width > 0 ∧
  length * width = area ∧
  ∀ (l w : ℝ), l > 0 → w > 0 → l * w = area →
  2 * (length + width) ≤ 2 * (l + w) ∧
  2 * (length + width) = 32 :=
sorry

end NUMINAMATH_CALUDE_min_fence_length_l2576_257670


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2576_257658

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {3, 4, 5, 6}

theorem intersection_of_M_and_N : M ∩ N = {3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2576_257658


namespace NUMINAMATH_CALUDE_max_min_product_l2576_257685

theorem max_min_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 12) (prod_sum_eq : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 9 ∧ 
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l2576_257685


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2576_257604

theorem expand_and_simplify (x y : ℝ) : 
  (2*x + 3*y)^2 - 2*x*(2*x - 3*y) = 18*x*y + 9*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2576_257604


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2576_257624

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of six coins -/
structure SixCoins where
  penny : CoinFlip
  nickel : CoinFlip
  dime : CoinFlip
  quarter : CoinFlip
  halfDollar : CoinFlip
  dollar : CoinFlip

/-- The total number of possible outcomes when flipping six coins -/
def totalOutcomes : Nat := 64

/-- Checks if the penny and dime have different outcomes -/
def pennyDimeDifferent (coins : SixCoins) : Prop :=
  coins.penny ≠ coins.dime

/-- Checks if the nickel and quarter have the same outcome -/
def nickelQuarterSame (coins : SixCoins) : Prop :=
  coins.nickel = coins.quarter

/-- Counts the number of favorable outcomes -/
def favorableOutcomes : Nat := 16

/-- The probability of the specified event -/
def probability : ℚ := 1 / 4

theorem coin_flip_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = probability :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2576_257624


namespace NUMINAMATH_CALUDE_parallelogram_area_l2576_257693

/-- The area of a parallelogram with one angle of 100 degrees and two consecutive sides of lengths 10 and 20 is approximately 196.96 -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h₁ : a = 10) (h₂ : b = 20) (h₃ : θ = 100 * π / 180) :
  abs (a * b * Real.sin θ - 196.96) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2576_257693


namespace NUMINAMATH_CALUDE_circles_common_chord_l2576_257677

/-- Two circles with equations x² + y² - 2x + 2y - 2 = 0 and x² + y² - 2mx = 0 (m > 0) 
    have a common chord of length 2 if and only if m = √6/2 -/
theorem circles_common_chord (m : ℝ) (hm : m > 0) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + 2*y - 2 = 0 ∧ x^2 + y^2 - 2*m*x = 0) ∧ 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 - 2*x₁ + 2*y₁ - 2 = 0 ∧
    x₁^2 + y₁^2 - 2*m*x₁ = 0 ∧
    x₂^2 + y₂^2 - 2*x₂ + 2*y₂ - 2 = 0 ∧
    x₂^2 + y₂^2 - 2*m*x₂ = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) ↔ 
  m = Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_circles_common_chord_l2576_257677


namespace NUMINAMATH_CALUDE_tan_product_simplification_l2576_257682

theorem tan_product_simplification :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_simplification_l2576_257682


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2576_257668

/-- The eccentricity of the hyperbola 9y² - 16x² = 144 is 5/4 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 5/4 ∧ 
  ∀ (x y : ℝ), 9*y^2 - 16*x^2 = 144 → 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
    y^2/a^2 - x^2/b^2 = 1 ∧
    c^2 = a^2 + b^2 ∧
    e = c/a :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2576_257668


namespace NUMINAMATH_CALUDE_casino_table_ratio_l2576_257622

/-- Proves that the ratio of money on table B to table C is 2 given the casino table conditions -/
theorem casino_table_ratio : 
  ∀ (A B C : ℝ),
  A = 40 →
  C = A + 20 →
  A + B + C = 220 →
  B / C = 2 := by
sorry

end NUMINAMATH_CALUDE_casino_table_ratio_l2576_257622


namespace NUMINAMATH_CALUDE_hendrix_class_size_l2576_257633

theorem hendrix_class_size :
  ∀ (initial_students : ℕ),
    (initial_students + 20 : ℚ) * (2/3) = 120 →
    initial_students = 160 := by
  sorry

end NUMINAMATH_CALUDE_hendrix_class_size_l2576_257633
