import Mathlib

namespace NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l2809_280933

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x+1) + 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) + 2

-- Theorem statement
theorem fixed_point_of_logarithmic_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l2809_280933


namespace NUMINAMATH_CALUDE_trivia_team_score_l2809_280943

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_points : ℕ) 
  (h1 : total_members = 15)
  (h2 : absent_members = 6)
  (h3 : total_points = 27) :
  total_points / (total_members - absent_members) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_score_l2809_280943


namespace NUMINAMATH_CALUDE_blue_easter_eggs_fraction_l2809_280951

theorem blue_easter_eggs_fraction 
  (purple_fraction : ℚ) 
  (purple_five_candy_ratio : ℚ) 
  (blue_five_candy_ratio : ℚ) 
  (five_candy_probability : ℚ) :
  purple_fraction = 1/5 →
  purple_five_candy_ratio = 1/2 →
  blue_five_candy_ratio = 1/4 →
  five_candy_probability = 3/10 →
  ∃ blue_fraction : ℚ, 
    blue_fraction = 4/5 ∧ 
    purple_fraction * purple_five_candy_ratio + blue_fraction * blue_five_candy_ratio = five_candy_probability :=
by sorry

end NUMINAMATH_CALUDE_blue_easter_eggs_fraction_l2809_280951


namespace NUMINAMATH_CALUDE_total_cleanings_is_777_l2809_280964

/-- Calculates the total number of times Michael, Angela, and Lucy clean themselves in 52 weeks --/
def total_cleanings : ℕ :=
  let weeks_in_year : ℕ := 52
  let days_in_week : ℕ := 7
  let month_in_weeks : ℕ := 4

  -- Michael's cleaning schedule
  let michael_baths_per_week : ℕ := 2
  let michael_showers_per_week : ℕ := 1
  let michael_vacation_weeks : ℕ := 3

  -- Angela's cleaning schedule
  let angela_showers_per_day : ℕ := 1
  let angela_vacation_weeks : ℕ := 2

  -- Lucy's regular cleaning schedule
  let lucy_baths_per_week : ℕ := 3
  let lucy_showers_per_week : ℕ := 2

  -- Lucy's modified schedule for one month
  let lucy_modified_baths_per_week : ℕ := 1
  let lucy_modified_showers_per_day : ℕ := 1

  -- Calculate total cleanings
  let michael_total := (michael_baths_per_week + michael_showers_per_week) * weeks_in_year - 
                       (michael_baths_per_week + michael_showers_per_week) * michael_vacation_weeks

  let angela_total := angela_showers_per_day * days_in_week * weeks_in_year - 
                      angela_showers_per_day * days_in_week * angela_vacation_weeks

  let lucy_regular_weeks := weeks_in_year - month_in_weeks
  let lucy_total := (lucy_baths_per_week + lucy_showers_per_week) * lucy_regular_weeks +
                    (lucy_modified_baths_per_week + lucy_modified_showers_per_day * days_in_week) * month_in_weeks

  michael_total + angela_total + lucy_total

theorem total_cleanings_is_777 : total_cleanings = 777 := by
  sorry

end NUMINAMATH_CALUDE_total_cleanings_is_777_l2809_280964


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l2809_280965

/-- A pentagon formed by 11 segments of length 2 -/
structure Pentagon where
  /-- The number of segments forming the pentagon -/
  num_segments : ℕ
  /-- The length of each segment -/
  segment_length : ℝ
  /-- The area of the pentagon -/
  area : ℝ
  /-- The first positive integer in the area expression -/
  m : ℕ
  /-- The second positive integer in the area expression -/
  n : ℕ
  /-- Condition: The number of segments is 11 -/
  h_num_segments : num_segments = 11
  /-- Condition: The length of each segment is 2 -/
  h_segment_length : segment_length = 2
  /-- Condition: The area is expressed as √m + √n -/
  h_area : area = Real.sqrt m + Real.sqrt n
  /-- Condition: m is positive -/
  h_m_pos : m > 0
  /-- Condition: n is positive -/
  h_n_pos : n > 0

/-- Theorem: For the given pentagon, m + n = 23 -/
theorem pentagon_area_sum (p : Pentagon) : p.m + p.n = 23 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l2809_280965


namespace NUMINAMATH_CALUDE_jacqueline_erasers_l2809_280992

/-- The number of boxes of erasers Jacqueline has -/
def num_boxes : ℕ := 4

/-- The number of erasers in each box -/
def erasers_per_box : ℕ := 10

/-- The total number of erasers Jacqueline has -/
def total_erasers : ℕ := num_boxes * erasers_per_box

theorem jacqueline_erasers : total_erasers = 40 :=
by sorry

end NUMINAMATH_CALUDE_jacqueline_erasers_l2809_280992


namespace NUMINAMATH_CALUDE_cats_not_eating_l2809_280968

theorem cats_not_eating (total : ℕ) (likes_apples : ℕ) (likes_fish : ℕ) (likes_both : ℕ) 
  (h1 : total = 75)
  (h2 : likes_apples = 15)
  (h3 : likes_fish = 55)
  (h4 : likes_both = 8) :
  total - (likes_apples + likes_fish - likes_both) = 13 :=
by sorry

end NUMINAMATH_CALUDE_cats_not_eating_l2809_280968


namespace NUMINAMATH_CALUDE_flea_jump_rational_angle_l2809_280915

/-- Represents a flea jumping between two intersecting lines -/
structure FleaJump where
  α : ℝ  -- Angle between the lines in radians
  jump_length : ℝ  -- Length of each jump
  returns_to_start : Prop  -- Flea eventually returns to the starting point

/-- Main theorem: If a flea jumps between two intersecting lines and returns to the start,
    the angle between the lines is a rational multiple of π -/
theorem flea_jump_rational_angle (fj : FleaJump) 
  (h1 : fj.jump_length = 1)
  (h2 : fj.returns_to_start)
  (h3 : fj.α > 0)
  (h4 : fj.α < π) :
  ∃ q : ℚ, fj.α = q * π :=
sorry

end NUMINAMATH_CALUDE_flea_jump_rational_angle_l2809_280915


namespace NUMINAMATH_CALUDE_initial_men_count_l2809_280953

theorem initial_men_count (provisions : ℕ) : ∃ (initial_men : ℕ),
  (provisions / (initial_men * 20) = provisions / ((initial_men + 200) * 15)) ∧
  initial_men = 600 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l2809_280953


namespace NUMINAMATH_CALUDE_angle_A_value_min_value_expression_l2809_280978

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  R : Real -- circumradius

-- Define the given condition
def triangle_condition (t : Triangle) : Prop :=
  2 * t.R - t.a = (t.a * (t.b^2 + t.c^2 - t.a^2)) / (t.a^2 + t.c^2 - t.b^2)

-- Theorem 1
theorem angle_A_value (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.A ≠ π/2) 
  (h3 : t.B = π/6) : 
  t.A = π/6 := by sorry

-- Theorem 2
theorem min_value_expression (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.A ≠ π/2) : 
  ∃ (min : Real), (∀ (t' : Triangle), triangle_condition t' → t'.A ≠ π/2 → 
    (2 * t'.a^2 - t'.c^2) / t'.b^2 ≥ min) ∧ 
  min = 4 * Real.sqrt 2 - 7 := by sorry

end NUMINAMATH_CALUDE_angle_A_value_min_value_expression_l2809_280978


namespace NUMINAMATH_CALUDE_find_divisor_l2809_280920

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 139 →
  quotient = 7 →
  remainder = 6 →
  dividend = divisor * quotient + remainder →
  divisor = 19 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2809_280920


namespace NUMINAMATH_CALUDE_fraction_sum_to_ratio_proof_l2809_280962

theorem fraction_sum_to_ratio_proof (x y : ℝ) 
  (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_to_ratio_proof_l2809_280962


namespace NUMINAMATH_CALUDE_point_P_in_fourth_quadrant_iff_a_in_range_l2809_280979

/-- A point in the fourth quadrant has a positive x-coordinate and a negative y-coordinate -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The coordinates of point P are defined in terms of parameter a -/
def point_P (a : ℝ) : ℝ × ℝ := (2*a + 4, 3*a - 6)

/-- Theorem stating the range of a for point P to be in the fourth quadrant -/
theorem point_P_in_fourth_quadrant_iff_a_in_range :
  ∀ a : ℝ, fourth_quadrant (point_P a).1 (point_P a).2 ↔ -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_P_in_fourth_quadrant_iff_a_in_range_l2809_280979


namespace NUMINAMATH_CALUDE_probability_both_selected_l2809_280946

theorem probability_both_selected (prob_ram : ℚ) (prob_ravi : ℚ) 
  (h1 : prob_ram = 5 / 7) 
  (h2 : prob_ravi = 1 / 5) : 
  prob_ram * prob_ravi = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l2809_280946


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2809_280941

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Generates the nth element of a systematic sample -/
def SystematicSample.nth_element (s : SystematicSample) (n : ℕ) : ℕ :=
  s.start + (n - 1) * s.interval

/-- Theorem: In a systematic sample of size 4 from a population of 50,
    if students with ID numbers 6, 30, and 42 are included,
    then the fourth student in the sample must have ID number 18 -/
theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop : s.population_size = 50)
  (h_sample : s.sample_size = 4)
  (h_start : s.start = 6)
  (h_interval : s.interval = 12)
  (h_30 : s.nth_element 3 = 30)
  (h_42 : s.nth_element 4 = 42) :
  s.nth_element 2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2809_280941


namespace NUMINAMATH_CALUDE_quadrilateral_rhombus_l2809_280969

-- Define the points
variable (A B C D P Q R S : ℝ × ℝ)

-- Define the properties of the quadrilaterals
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def are_external_similar_isosceles_triangles (A B C D P Q R S : ℝ × ℝ) : Prop := sorry

def is_rectangle (P Q R S : ℝ × ℝ) : Prop := sorry

def sides_not_equal (P Q R S : ℝ × ℝ) : Prop := sorry

def is_rhombus (A B C D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_rhombus 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : are_external_similar_isosceles_triangles A B C D P Q R S)
  (h3 : is_rectangle P Q R S)
  (h4 : sides_not_equal P Q R S) :
  is_rhombus A B C D :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_rhombus_l2809_280969


namespace NUMINAMATH_CALUDE_complex_numbers_on_circle_l2809_280973

/-- Given non-zero complex numbers a₁, a₂, a₃, a₄, a₅ satisfying certain conditions,
    prove that they lie on the same circle in the complex plane. -/
theorem complex_numbers_on_circle (a₁ a₂ a₃ a₄ a₅ : ℂ) (S : ℝ) 
    (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0)
    (h_ratio : a₂ / a₁ = a₃ / a₂ ∧ a₃ / a₂ = a₄ / a₃ ∧ a₄ / a₃ = a₅ / a₄)
    (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅))
    (h_sum_real : a₁ + a₂ + a₃ + a₄ + a₅ = S)
    (h_S_bound : abs S ≤ 2) :
  ∃ r : ℝ, r > 0 ∧ Complex.abs a₁ = r ∧ Complex.abs a₂ = r ∧ 
    Complex.abs a₃ = r ∧ Complex.abs a₄ = r ∧ Complex.abs a₅ = r :=
by sorry

end NUMINAMATH_CALUDE_complex_numbers_on_circle_l2809_280973


namespace NUMINAMATH_CALUDE_jacob_shooting_improvement_l2809_280931

/-- Represents the number of shots Jacob made in the fourth game -/
def shots_made_fourth_game : ℕ := 9

/-- Represents Jacob's initial number of shots -/
def initial_shots : ℕ := 45

/-- Represents Jacob's initial number of successful shots -/
def initial_successful_shots : ℕ := 18

/-- Represents the number of shots Jacob attempted in the fourth game -/
def fourth_game_attempts : ℕ := 15

/-- Represents Jacob's initial shooting average as a rational number -/
def initial_average : ℚ := 2/5

/-- Represents Jacob's final shooting average as a rational number -/
def final_average : ℚ := 9/20

theorem jacob_shooting_improvement :
  (initial_successful_shots + shots_made_fourth_game : ℚ) / (initial_shots + fourth_game_attempts) = final_average :=
sorry

end NUMINAMATH_CALUDE_jacob_shooting_improvement_l2809_280931


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l2809_280907

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ (3 * x^2 + 1 = 4) ↔ (x = -1 ∧ y = -4) ∨ (x = 1 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l2809_280907


namespace NUMINAMATH_CALUDE_time_after_elapsed_minutes_l2809_280910

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting DateTime -/
def startTime : DateTime :=
  { year := 2015, month := 3, day := 3, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def elapsedMinutes : Nat := 4350

/-- The expected result DateTime -/
def expectedResult : DateTime :=
  { year := 2015, month := 3, day := 6, hour := 0, minute := 30 }

theorem time_after_elapsed_minutes :
  addMinutes startTime elapsedMinutes = expectedResult := by
  sorry

end NUMINAMATH_CALUDE_time_after_elapsed_minutes_l2809_280910


namespace NUMINAMATH_CALUDE_expression_evaluation_l2809_280909

theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = 1) :
  2 * (x - y) - 3 * (2 * x - y) + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2809_280909


namespace NUMINAMATH_CALUDE_unobserved_planet_exists_l2809_280954

/-- A planet in the Zoolander system -/
structure Planet where
  id : Nat

/-- The planetary system of star Zoolander -/
structure ZoolanderSystem where
  planets : Finset Planet
  distance : Planet → Planet → ℝ
  closest_planet : Planet → Planet
  num_planets : planets.card = 2015
  different_distances : ∀ p q r s : Planet, p ≠ q → r ≠ s → (p, q) ≠ (r, s) → distance p q ≠ distance r s
  closest_is_closest : ∀ p q : Planet, p ≠ q → p ∈ planets → q ∈ planets → 
    distance p (closest_planet p) ≤ distance p q

/-- There exists a planet that is not observed by any astronomer -/
theorem unobserved_planet_exists (z : ZoolanderSystem) : 
  ∃ p : Planet, p ∈ z.planets ∧ ∀ q : Planet, q ∈ z.planets → z.closest_planet q ≠ p :=
sorry

end NUMINAMATH_CALUDE_unobserved_planet_exists_l2809_280954


namespace NUMINAMATH_CALUDE_ben_hit_seven_l2809_280925

-- Define the set of friends
inductive Friend
| Alice | Ben | Cindy | Dave | Ellen | Frank

-- Define the scores for each friend
def score (f : Friend) : ℕ :=
  match f with
  | Friend.Alice => 18
  | Friend.Ben => 13
  | Friend.Cindy => 19
  | Friend.Dave => 16
  | Friend.Ellen => 20
  | Friend.Frank => 5

-- Define the set of possible target scores
def targetScores : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

-- Define a function to check if a pair of scores is valid
def validPair (a b : ℕ) : Prop :=
  a ∈ targetScores ∧ b ∈ targetScores ∧ a ≠ b ∧ a + b = score Friend.Ben

-- Theorem statement
theorem ben_hit_seven :
  ∃ (a b : ℕ), validPair a b ∧ (a = 7 ∨ b = 7) ∧
  (∀ (f : Friend), f ≠ Friend.Ben → ¬∃ (x y : ℕ), validPair x y ∧ (x = 7 ∨ y = 7)) :=
sorry

end NUMINAMATH_CALUDE_ben_hit_seven_l2809_280925


namespace NUMINAMATH_CALUDE_lizas_account_balance_l2809_280934

/-- Calculates the final balance in Liza's account after all transactions -/
def final_balance (initial_balance rent paycheck electricity internet phone : ℤ) : ℤ :=
  initial_balance - rent + paycheck - electricity - internet - phone

/-- Theorem stating that Liza's final account balance is correct -/
theorem lizas_account_balance :
  final_balance 800 450 1500 117 100 70 = 1563 := by
  sorry

end NUMINAMATH_CALUDE_lizas_account_balance_l2809_280934


namespace NUMINAMATH_CALUDE_rectangle_diagonal_pythagorean_l2809_280944

/-- A rectangle with side lengths a and b, and diagonal c -/
structure Rectangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- The Pythagorean theorem holds for the rectangle's diagonal -/
theorem rectangle_diagonal_pythagorean (rect : Rectangle) : 
  rect.c^2 = rect.a^2 + rect.b^2 := by
  sorry

#check rectangle_diagonal_pythagorean

end NUMINAMATH_CALUDE_rectangle_diagonal_pythagorean_l2809_280944


namespace NUMINAMATH_CALUDE_pages_read_l2809_280938

/-- Given a book with a total number of pages and the number of pages left to read,
    calculate the number of pages already read. -/
theorem pages_read (total_pages left_to_read : ℕ) : 
  total_pages = 17 → left_to_read = 6 → total_pages - left_to_read = 11 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_l2809_280938


namespace NUMINAMATH_CALUDE_trapezoid_area_l2809_280916

/-- Trapezoid ABCD with diagonals AC and BD, and midpoints P and S of AD and BC respectively -/
structure Trapezoid :=
  (A B C D P S : ℝ × ℝ)
  (is_trapezoid : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1))
  (diag_AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 8)
  (diag_BD_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 6)
  (P_midpoint : P = ((A.1 + D.1) / 2, (A.2 + D.2) / 2))
  (S_midpoint : S = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (PS_length : Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) = 5)

/-- The area of the trapezoid ABCD is 24 -/
theorem trapezoid_area (t : Trapezoid) : 
  (1 / 2) * Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) * 
  Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2809_280916


namespace NUMINAMATH_CALUDE_octal_to_decimal_conversion_coral_age_conversion_l2809_280952

-- Define the octal age
def octal_age : ℕ := 753

-- Define the decimal age
def decimal_age : ℕ := 491

-- Theorem to prove the equivalence
theorem octal_to_decimal_conversion :
  (3 * 8^0 + 5 * 8^1 + 7 * 8^2 : ℕ) = decimal_age :=
by sorry

-- Theorem to prove that octal_age in decimal is equal to decimal_age
theorem coral_age_conversion :
  octal_age.digits 8 = [3, 5, 7] ∧
  (3 * 8^0 + 5 * 8^1 + 7 * 8^2 : ℕ) = decimal_age :=
by sorry

end NUMINAMATH_CALUDE_octal_to_decimal_conversion_coral_age_conversion_l2809_280952


namespace NUMINAMATH_CALUDE_negation_of_all_divisible_by_five_are_even_l2809_280971

theorem negation_of_all_divisible_by_five_are_even :
  (¬ ∀ n : ℤ, 5 ∣ n → Even n) ↔ (∃ n : ℤ, 5 ∣ n ∧ ¬Even n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_divisible_by_five_are_even_l2809_280971


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l2809_280982

theorem smallest_number_divisibility : ∃! n : ℕ, 
  (∀ m : ℕ, m < n → ¬(((m + 7) % 8 = 0) ∧ ((m + 7) % 11 = 0) ∧ ((m + 7) % 24 = 0))) ∧
  ((n + 7) % 8 = 0) ∧ ((n + 7) % 11 = 0) ∧ ((n + 7) % 24 = 0) ∧
  n = 257 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l2809_280982


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2809_280999

theorem sum_of_solutions_is_zero :
  let f : ℝ → ℝ := λ x => Real.sqrt (9 - x^2 / 4)
  (∀ x, f x = 3 → x = 0) ∧ (∃ x, f x = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2809_280999


namespace NUMINAMATH_CALUDE_chipped_marbles_count_l2809_280937

/-- Represents the number of marbles in each bag -/
def bags : List Nat := [20, 22, 25, 30, 32, 34, 36]

/-- Represents the number of bags Jane takes -/
def jane_bags : Nat := 3

/-- Represents the number of bags George takes -/
def george_bags : Nat := 3

/-- The number of chipped marbles -/
def chipped_marbles : Nat := 22

theorem chipped_marbles_count :
  ∃ (jane_selection george_selection : List Nat),
    jane_selection.length = jane_bags ∧
    george_selection.length = george_bags ∧
    (∀ x, x ∈ jane_selection ∨ x ∈ george_selection → x ∈ bags) ∧
    (∀ x, x ∈ jane_selection → x ∉ george_selection) ∧
    (∀ x, x ∈ george_selection → x ∉ jane_selection) ∧
    (∃ remaining, remaining ∈ bags ∧
      remaining ∉ jane_selection ∧
      remaining ∉ george_selection ∧
      remaining = chipped_marbles ∧
      (jane_selection.sum + george_selection.sum = 3 * remaining)) :=
sorry

end NUMINAMATH_CALUDE_chipped_marbles_count_l2809_280937


namespace NUMINAMATH_CALUDE_race_heartbeats_l2809_280960

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  let race_time := pace * race_distance
  race_time * heart_rate

/-- Proves that the total number of heartbeats during a 26-mile race is 19500,
    given the specified heart rate and pace. -/
theorem race_heartbeats :
  total_heartbeats 150 5 26 = 19500 :=
by sorry

end NUMINAMATH_CALUDE_race_heartbeats_l2809_280960


namespace NUMINAMATH_CALUDE_scientific_notation_of_6_1757_million_l2809_280932

theorem scientific_notation_of_6_1757_million :
  let original_number : ℝ := 6.1757 * 1000000
  original_number = 6.1757 * (10 ^ 6) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_6_1757_million_l2809_280932


namespace NUMINAMATH_CALUDE_school_ratio_problem_l2809_280901

theorem school_ratio_problem (S T : ℕ) : 
  S / T = 50 →
  (S + 50) / (T + 5) = 25 →
  T = 3 :=
by sorry

end NUMINAMATH_CALUDE_school_ratio_problem_l2809_280901


namespace NUMINAMATH_CALUDE_solve_equation_l2809_280974

theorem solve_equation : ∃ x : ℚ, (40 / 60 : ℚ) = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2809_280974


namespace NUMINAMATH_CALUDE_petya_wins_l2809_280913

/-- Represents the game between Petya and Vasya -/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game with the given conditions -/
def game : CandyGame :=
  { total_candies := 25,
    prob_two_caramels := 0.54 }

/-- Theorem: Petya has a higher chance of winning -/
theorem petya_wins (g : CandyGame) 
  (h1 : g.total_candies = 25)
  (h2 : g.prob_two_caramels = 0.54) :
  g.prob_two_caramels > 1 - g.prob_two_caramels := by
  sorry

#check petya_wins game

end NUMINAMATH_CALUDE_petya_wins_l2809_280913


namespace NUMINAMATH_CALUDE_original_mixture_composition_l2809_280927

def original_mixture (acid water : ℝ) : Prop :=
  acid > 0 ∧ water > 0

def after_adding_water (acid water : ℝ) : Prop :=
  acid / (acid + water + 2) = 1/4

def after_adding_acid (acid water : ℝ) : Prop :=
  (acid + 3) / (acid + water + 5) = 2/5

theorem original_mixture_composition (acid water : ℝ) :
  original_mixture acid water →
  after_adding_water acid water →
  after_adding_acid acid water →
  acid / (acid + water) = 3/10 :=
by sorry

end NUMINAMATH_CALUDE_original_mixture_composition_l2809_280927


namespace NUMINAMATH_CALUDE_ball_probability_l2809_280986

theorem ball_probability (red green yellow total : ℕ) (p_green : ℚ) :
  red = 8 →
  green = 10 →
  total = red + green + yellow →
  p_green = 1 / 4 →
  p_green = green / total →
  yellow / total = 11 / 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l2809_280986


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l2809_280949

theorem base_2_representation_of_123 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1) ∧
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l2809_280949


namespace NUMINAMATH_CALUDE_last_term_value_l2809_280922

-- Define the arithmetic sequence
def arithmetic_sequence (a b : ℝ) : ℕ → ℝ
  | 0 => a
  | 1 => b
  | 2 => 5 * a
  | 3 => 7
  | 4 => 3 * b
  | n + 5 => arithmetic_sequence a b n

-- Define the sum of the sequence
def sequence_sum (a b : ℝ) (n : ℕ) : ℝ :=
  (List.range n).map (arithmetic_sequence a b) |>.sum

-- Theorem statement
theorem last_term_value (a b : ℝ) (n : ℕ) :
  sequence_sum a b n = 2500 →
  ∃ c, arithmetic_sequence a b (n - 1) = c ∧ c = 99 := by
  sorry

end NUMINAMATH_CALUDE_last_term_value_l2809_280922


namespace NUMINAMATH_CALUDE_log_base_32_integer_count_l2809_280987

theorem log_base_32_integer_count : 
  ∃! (n : ℕ), n > 0 ∧ (∃ (S : Finset ℕ), 
    (∀ b ∈ S, b > 0 ∧ ∃ (k : ℕ), k > 0 ∧ (↑b : ℝ) ^ k = 32) ∧
    S.card = n) :=
by sorry

end NUMINAMATH_CALUDE_log_base_32_integer_count_l2809_280987


namespace NUMINAMATH_CALUDE_janine_reading_theorem_l2809_280918

/-- The number of books Janine read last month -/
def books_last_month : ℕ := 5

/-- The number of books Janine read this month -/
def books_this_month : ℕ := 2 * books_last_month

/-- The number of pages in each book -/
def pages_per_book : ℕ := 10

/-- The total number of pages Janine read in two months -/
def total_pages : ℕ := (books_last_month + books_this_month) * pages_per_book

theorem janine_reading_theorem : total_pages = 150 := by
  sorry

end NUMINAMATH_CALUDE_janine_reading_theorem_l2809_280918


namespace NUMINAMATH_CALUDE_probability_at_least_one_green_l2809_280957

theorem probability_at_least_one_green (total : ℕ) (red : ℕ) (green : ℕ) (choose : ℕ) :
  total = red + green →
  total = 10 →
  red = 6 →
  green = 4 →
  choose = 3 →
  (1 : ℚ) - (Nat.choose red choose : ℚ) / (Nat.choose total choose : ℚ) = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_green_l2809_280957


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l2809_280905

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ m / (x - 1) + 3 / (1 - x) = 1) ↔ (m ≥ 2 ∧ m ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l2809_280905


namespace NUMINAMATH_CALUDE_cost_per_meter_l2809_280928

def total_cost : ℝ := 416.25
def total_length : ℝ := 9.25

theorem cost_per_meter : total_cost / total_length = 45 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_meter_l2809_280928


namespace NUMINAMATH_CALUDE_sams_carrots_l2809_280903

theorem sams_carrots (sandy_carrots : ℕ) (total_carrots : ℕ) (h1 : sandy_carrots = 6) (h2 : total_carrots = 9) : 
  total_carrots - sandy_carrots = 3 := by
  sorry

end NUMINAMATH_CALUDE_sams_carrots_l2809_280903


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2809_280981

/-- Given a rectangle with dimensions 12 and 18, prove that the fraction of the rectangle
    that is shaded is 1/12, where the shaded region is 1/3 of a quarter of the rectangle. -/
theorem shaded_fraction_of_rectangle (width : ℕ) (height : ℕ) (shaded_area : ℚ) :
  width = 12 →
  height = 18 →
  shaded_area = (1 / 3) * (1 / 4) * (width * height) →
  shaded_area / (width * height) = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2809_280981


namespace NUMINAMATH_CALUDE_min_value_of_f_l2809_280976

/-- The function f(x) = 3x^2 - 12x + 7 + 749 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 7 + 749

theorem min_value_of_f :
  ∃ (m : ℝ), m = 744 ∧ ∀ (x : ℝ), f x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2809_280976


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l2809_280917

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (boat_speed : ℝ) (stream_speed : ℝ) : ℝ :=
  boat_speed + stream_speed

theorem boat_upstream_distance 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : boat_speed = 11)
  (h2 : boat_distance boat_speed (downstream_distance - boat_speed) = 13) :
  boat_distance boat_speed (boat_speed - downstream_distance) = 9 := by
sorry

end NUMINAMATH_CALUDE_boat_upstream_distance_l2809_280917


namespace NUMINAMATH_CALUDE_number_equation_proof_l2809_280988

theorem number_equation_proof : ∃ x : ℝ, 5020 - (x / 100.4) = 5015 ∧ x = 502 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l2809_280988


namespace NUMINAMATH_CALUDE_impossible_equal_side_sums_l2809_280940

/-- Represents the pattern of squares in the problem -/
structure SquarePattern :=
  (vertices : Fin 24 → ℕ)
  (is_consecutive : ∀ i : Fin 23, vertices i.succ = vertices i + 1)
  (is_bijective : Function.Bijective vertices)

/-- Represents a side of a square in the pattern -/
inductive Side : Type
| Top : Side
| Right : Side
| Bottom : Side
| Left : Side

/-- Gets the vertices on a given side of a square -/
def side_vertices (square : Fin 4) (side : Side) : Fin 24 → Prop :=
  sorry

/-- The sum of numbers on a side of a square -/
def side_sum (p : SquarePattern) (square : Fin 4) (side : Side) : ℕ :=
  sorry

/-- The theorem stating the impossibility of the required arrangement -/
theorem impossible_equal_side_sums :
  ¬ ∃ (p : SquarePattern),
    ∀ (s1 s2 : Fin 4) (side1 side2 : Side),
      side_sum p s1 side1 = side_sum p s2 side2 :=
sorry

end NUMINAMATH_CALUDE_impossible_equal_side_sums_l2809_280940


namespace NUMINAMATH_CALUDE_sequence_product_theorem_l2809_280980

def arithmetic_sequence (n : ℕ) : ℕ :=
  2 * n - 1

def geometric_sequence (n : ℕ) : ℕ :=
  2^(n - 1)

theorem sequence_product_theorem :
  let a := arithmetic_sequence
  let b := geometric_sequence
  b (a 1) * b (a 3) * b (a 5) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_theorem_l2809_280980


namespace NUMINAMATH_CALUDE_proposition_relationship_l2809_280924

theorem proposition_relationship :
  (∀ a b : ℝ, (a > b ∧ a⁻¹ > b⁻¹) → a > 0) ∧
  ¬(∀ a b : ℝ, a > 0 → (a > b ∧ a⁻¹ > b⁻¹)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l2809_280924


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2809_280961

theorem smallest_prime_divisor_of_sum : 
  ∃ (n : ℕ), n = 6^15 + 9^11 ∧ (∀ p : ℕ, Prime p → p ∣ n → p ≥ 3) ∧ 3 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2809_280961


namespace NUMINAMATH_CALUDE_problem_solution_l2809_280906

theorem problem_solution (x y : ℝ) 
  (sum_eq : x + y = 360)
  (ratio_eq : x / y = 3 / 5) : 
  y - x = 90 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2809_280906


namespace NUMINAMATH_CALUDE_jack_christina_lindy_problem_l2809_280929

/-- The problem setup and solution for Jack, Christina, and Lindy's movement --/
theorem jack_christina_lindy_problem (
  initial_distance : ℝ) 
  (jack_speed christina_speed lindy_speed : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : jack_speed = 7)
  (h3 : christina_speed = 8)
  (h4 : lindy_speed = 10) :
  let meeting_time := initial_distance / (jack_speed + christina_speed)
  lindy_speed * meeting_time = 100 := by
  sorry


end NUMINAMATH_CALUDE_jack_christina_lindy_problem_l2809_280929


namespace NUMINAMATH_CALUDE_m_value_l2809_280914

theorem m_value (a b : ℝ) (m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = 10 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l2809_280914


namespace NUMINAMATH_CALUDE_line_opposite_sides_m_range_l2809_280998

/-- A line in 2D space defined by the equation 3x - 2y + m = 0 -/
structure Line (m : ℝ) where
  equation : ℝ → ℝ → ℝ
  eq_def : equation = fun x y => 3 * x - 2 * y + m

/-- Determines if two points are on opposite sides of a line -/
def opposite_sides (l : Line m) (p1 p2 : ℝ × ℝ) : Prop :=
  l.equation p1.1 p1.2 * l.equation p2.1 p2.2 < 0

theorem line_opposite_sides_m_range (m : ℝ) (l : Line m) :
  opposite_sides l (3, 1) (-4, 6) → -7 < m ∧ m < 24 := by
  sorry


end NUMINAMATH_CALUDE_line_opposite_sides_m_range_l2809_280998


namespace NUMINAMATH_CALUDE_bread_price_is_two_l2809_280994

/-- The price of a can of spam in dollars -/
def spam_price : ℚ := 3

/-- The price of a jar of peanut butter in dollars -/
def peanut_butter_price : ℚ := 5

/-- The number of cans of spam bought -/
def spam_quantity : ℕ := 12

/-- The number of jars of peanut butter bought -/
def peanut_butter_quantity : ℕ := 3

/-- The number of loaves of bread bought -/
def bread_quantity : ℕ := 4

/-- The total amount paid in dollars -/
def total_paid : ℚ := 59

/-- The price of a loaf of bread in dollars -/
def bread_price : ℚ := 2

theorem bread_price_is_two :
  spam_price * spam_quantity +
  peanut_butter_price * peanut_butter_quantity +
  bread_price * bread_quantity = total_paid := by
  sorry

end NUMINAMATH_CALUDE_bread_price_is_two_l2809_280994


namespace NUMINAMATH_CALUDE_tshirt_sale_duration_l2809_280930

/-- Calculates the duration of a t-shirt sale given the number of shirts sold,
    their prices, and the revenue rate per minute. -/
theorem tshirt_sale_duration
  (total_shirts : ℕ)
  (black_shirts : ℕ)
  (white_shirts : ℕ)
  (black_price : ℚ)
  (white_price : ℚ)
  (revenue_rate : ℚ)
  (h1 : total_shirts = 200)
  (h2 : black_shirts = total_shirts / 2)
  (h3 : white_shirts = total_shirts / 2)
  (h4 : black_price = 30)
  (h5 : white_price = 25)
  (h6 : revenue_rate = 220) :
  (black_shirts * black_price + white_shirts * white_price) / revenue_rate = 25 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_sale_duration_l2809_280930


namespace NUMINAMATH_CALUDE_dice_product_six_prob_l2809_280956

/-- The probability of rolling a specific number on a standard die -/
def die_prob : ℚ := 1 / 6

/-- The set of all possible outcomes when rolling three dice -/
def all_outcomes : Finset (ℕ × ℕ × ℕ) := sorry

/-- The set of favorable outcomes where the product of the three numbers is 6 -/
def favorable_outcomes : Finset (ℕ × ℕ × ℕ) := sorry

/-- The probability of rolling three dice such that their product is 6 -/
theorem dice_product_six_prob : 
  (Finset.card favorable_outcomes : ℚ) / (Finset.card all_outcomes : ℚ) = 1 / 24 := by sorry

end NUMINAMATH_CALUDE_dice_product_six_prob_l2809_280956


namespace NUMINAMATH_CALUDE_concert_admission_revenue_l2809_280939

theorem concert_admission_revenue :
  let total_attendance : ℕ := 578
  let adult_price : ℚ := 2
  let child_price : ℚ := (3/2)
  let num_adults : ℕ := 342
  let num_children : ℕ := total_attendance - num_adults
  let total_revenue : ℚ := (num_adults : ℚ) * adult_price + (num_children : ℚ) * child_price
  total_revenue = 1038 :=
by sorry

end NUMINAMATH_CALUDE_concert_admission_revenue_l2809_280939


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2809_280935

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.cos (α + π / 4) = -3 / 5) : Real.tan α = 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2809_280935


namespace NUMINAMATH_CALUDE_toaster_cost_l2809_280945

def amazon_purchase : ℝ := 3000
def tv_cost : ℝ := 700
def returned_bike_cost : ℝ := 500
def sold_bike_cost : ℝ := returned_bike_cost * 1.2
def sold_bike_price : ℝ := sold_bike_cost * 0.8
def total_out_of_pocket : ℝ := 2020

theorem toaster_cost :
  let total_return := tv_cost + returned_bike_cost
  let out_of_pocket_before_toaster := amazon_purchase - total_return + sold_bike_price
  let toaster_cost := out_of_pocket_before_toaster - total_out_of_pocket
  toaster_cost = 260 :=
by sorry

end NUMINAMATH_CALUDE_toaster_cost_l2809_280945


namespace NUMINAMATH_CALUDE_train_length_proof_l2809_280966

/-- Given two trains running in opposite directions with the same speed,
    prove that their length is 120 meters. -/
theorem train_length_proof (speed : ℝ) (crossing_time : ℝ) :
  speed = 36 → crossing_time = 12 → 
  ∃ (train_length : ℝ), train_length = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l2809_280966


namespace NUMINAMATH_CALUDE_jelly_beans_remaining_l2809_280991

/-- The number of jelly beans remaining in a container after distribution -/
def remaining_jelly_beans (total : ℕ) (num_people : ℕ) (first_group : ℕ) (second_group : ℕ) (second_group_beans : ℕ) : ℕ :=
  total - (first_group * (2 * second_group_beans) + second_group * second_group_beans)

/-- Proof that 1600 jelly beans remain in the container -/
theorem jelly_beans_remaining :
  remaining_jelly_beans 8000 10 6 4 400 = 1600 := by
  sorry

#eval remaining_jelly_beans 8000 10 6 4 400

end NUMINAMATH_CALUDE_jelly_beans_remaining_l2809_280991


namespace NUMINAMATH_CALUDE_megan_shirt_payment_l2809_280911

/-- The amount Megan pays for a shirt after discount -/
def shirt_price (original_price discount : ℕ) : ℕ :=
  original_price - discount

/-- Theorem: Megan pays $16 for the shirt -/
theorem megan_shirt_payment : shirt_price 22 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_megan_shirt_payment_l2809_280911


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l2809_280950

theorem smallest_n_divisible (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < n → ¬(20 ∣ (25 * m) ∧ 18 ∣ (25 * m) ∧ 24 ∣ (25 * m))) →
  (20 ∣ (25 * n) ∧ 18 ∣ (25 * n) ∧ 24 ∣ (25 * n)) →
  n = 36 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l2809_280950


namespace NUMINAMATH_CALUDE_exists_parallel_line_l2809_280902

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersects : Plane → Plane → Prop)
variable (not_perpendicular : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem exists_parallel_line 
  (α β γ : Plane)
  (h1 : perpendicular β γ)
  (h2 : intersects α γ)
  (h3 : not_perpendicular α γ) :
  ∃ (a : Line), in_plane a α ∧ parallel a γ :=
sorry

end NUMINAMATH_CALUDE_exists_parallel_line_l2809_280902


namespace NUMINAMATH_CALUDE_min_toothpicks_theorem_l2809_280983

/-- A geometric figure made of toothpicks -/
structure ToothpickFigure where
  upward_triangles : ℕ
  downward_triangles : ℕ
  horizontal_toothpicks : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ :=
  figure.horizontal_toothpicks

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_theorem (figure : ToothpickFigure) 
  (h1 : figure.upward_triangles = 15)
  (h2 : figure.downward_triangles = 10)
  (h3 : figure.horizontal_toothpicks = 15) :
  min_toothpicks_to_remove figure = 15 := by
  sorry

#check min_toothpicks_theorem

end NUMINAMATH_CALUDE_min_toothpicks_theorem_l2809_280983


namespace NUMINAMATH_CALUDE_correct_investment_structure_l2809_280948

/-- Represents the investment structure of a company --/
structure InvestmentStructure where
  initial_investors : ℕ
  initial_contribution : ℕ

/-- Checks if the investment structure satisfies the given conditions --/
def satisfies_conditions (s : InvestmentStructure) : Prop :=
  let contribution_1 := s.initial_contribution + 10000
  let contribution_2 := s.initial_contribution + 30000
  (s.initial_investors - 10) * contribution_1 = s.initial_investors * s.initial_contribution ∧
  (s.initial_investors - 25) * contribution_2 = s.initial_investors * s.initial_contribution

/-- Theorem stating the correct investment structure --/
theorem correct_investment_structure :
  ∃ (s : InvestmentStructure), s.initial_investors = 100 ∧ s.initial_contribution = 90000 ∧ satisfies_conditions s := by
  sorry

end NUMINAMATH_CALUDE_correct_investment_structure_l2809_280948


namespace NUMINAMATH_CALUDE_election_votes_theorem_l2809_280936

theorem election_votes_theorem (total_votes : ℕ) : 
  (∃ (candidate_votes rival_votes : ℕ),
    candidate_votes = total_votes / 10 ∧
    rival_votes = candidate_votes + 16000 ∧
    candidate_votes + rival_votes = total_votes) →
  total_votes = 20000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l2809_280936


namespace NUMINAMATH_CALUDE_sphere_radius_l2809_280995

/-- The radius of a sphere that forms a quarter-sphere with radius 4∛4 cm is 4 cm. -/
theorem sphere_radius (r : ℝ) : r = 4 * Real.rpow 4 (1/3) → 4 = (1/4)^(1/3) * r := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_l2809_280995


namespace NUMINAMATH_CALUDE_peanuts_remaining_l2809_280989

theorem peanuts_remaining (initial_peanuts : ℕ) (brock_fraction : ℚ) (bonita_peanuts : ℕ) : 
  initial_peanuts = 148 →
  brock_fraction = 1/4 →
  bonita_peanuts = 29 →
  initial_peanuts - (initial_peanuts * brock_fraction).floor - bonita_peanuts = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_peanuts_remaining_l2809_280989


namespace NUMINAMATH_CALUDE_quarter_power_inequality_l2809_280912

theorem quarter_power_inequality (x y : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : y < 1) :
  (1/4 : ℝ)^x > (1/4 : ℝ)^y := by
  sorry

end NUMINAMATH_CALUDE_quarter_power_inequality_l2809_280912


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l2809_280919

/-- A triangle is obtuse if one of its angles is greater than 90 degrees -/
def IsObtuseTriangle (a b c : ℝ) : Prop :=
  a > 90 ∨ b > 90 ∨ c > 90

/-- Theorem: If A, B, and C are the interior angles of a triangle, 
    and A > 3B and C < 2B, then the triangle is obtuse -/
theorem triangle_is_obtuse (a b c : ℝ) 
    (angle_sum : a + b + c = 180)
    (h1 : a > 3 * b) 
    (h2 : c < 2 * b) : 
  IsObtuseTriangle a b c := by
sorry

end NUMINAMATH_CALUDE_triangle_is_obtuse_l2809_280919


namespace NUMINAMATH_CALUDE_fraction_arrangement_l2809_280921

theorem fraction_arrangement :
  (1 / 8 * 1 / 9 * 1 / 28 : ℚ) = 1 / 2016 ∨ ((1 / 8 - 1 / 9) * 1 / 28 : ℚ) = 1 / 2016 :=
by sorry

end NUMINAMATH_CALUDE_fraction_arrangement_l2809_280921


namespace NUMINAMATH_CALUDE_division_problem_l2809_280955

theorem division_problem (dividend divisor : ℕ) (h1 : dividend + divisor = 136) (h2 : dividend / divisor = 7) : divisor = 17 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2809_280955


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2809_280970

theorem right_triangle_sets :
  -- Set A
  (5^2 + 12^2 = 13^2) ∧
  -- Set B
  ((Real.sqrt 2)^2 + (Real.sqrt 3)^2 = (Real.sqrt 5)^2) ∧
  -- Set C
  (3^2 + (Real.sqrt 7)^2 = 4^2) ∧
  -- Set D
  (2^2 + 3^2 ≠ 4^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2809_280970


namespace NUMINAMATH_CALUDE_equation_solution_l2809_280985

theorem equation_solution : ∃ x : ℚ, (2 / 5 : ℚ) - (1 / 7 : ℚ) = 1 / x ∧ x = 35 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2809_280985


namespace NUMINAMATH_CALUDE_curve_tangent_to_line_l2809_280942

/-- A curve y = e^x + a is tangent to the line y = x if and only if a = -1 -/
theorem curve_tangent_to_line (a : ℝ) : 
  (∃ x₀ : ℝ, (Real.exp x₀ + a = x₀) ∧ (Real.exp x₀ = 1)) ↔ a = -1 :=
sorry

end NUMINAMATH_CALUDE_curve_tangent_to_line_l2809_280942


namespace NUMINAMATH_CALUDE_prob_twice_daughters_is_37_256_l2809_280984

-- Define the number of children
def num_children : ℕ := 8

-- Define the probability of having a daughter (equal to the probability of having a son)
def p_daughter : ℚ := 1/2

-- Define the function to calculate the probability of having exactly k daughters out of n children
def prob_k_daughters (n k : ℕ) : ℚ :=
  (n.choose k) * (p_daughter ^ k) * ((1 - p_daughter) ^ (n - k))

-- Define the probability of having at least twice as many daughters as sons
def prob_twice_daughters : ℚ :=
  prob_k_daughters num_children num_children +
  prob_k_daughters num_children (num_children - 1) +
  prob_k_daughters num_children (num_children - 2)

-- Theorem statement
theorem prob_twice_daughters_is_37_256 : prob_twice_daughters = 37/256 := by
  sorry

end NUMINAMATH_CALUDE_prob_twice_daughters_is_37_256_l2809_280984


namespace NUMINAMATH_CALUDE_arithmetic_sequence_25th_term_l2809_280904

/-- An arithmetic sequence with first term 100 and common difference -4 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  100 - 4 * (n - 1)

theorem arithmetic_sequence_25th_term :
  arithmetic_sequence 25 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_25th_term_l2809_280904


namespace NUMINAMATH_CALUDE_median_longest_side_right_triangle_l2809_280926

theorem median_longest_side_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let median := (max a (max b c)) / 2
  median = 5 := by
  sorry

end NUMINAMATH_CALUDE_median_longest_side_right_triangle_l2809_280926


namespace NUMINAMATH_CALUDE_tan_75_degrees_l2809_280959

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_75_degrees_l2809_280959


namespace NUMINAMATH_CALUDE_two_solutions_only_l2809_280908

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem two_solutions_only : 
  {k : ℕ | k > 0 ∧ digit_product k = (25 * k) / 8 - 211} = {72, 88} :=
by sorry

end NUMINAMATH_CALUDE_two_solutions_only_l2809_280908


namespace NUMINAMATH_CALUDE_solution_to_congruence_l2809_280993

theorem solution_to_congruence (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (100 * n) % 103 = 34 % 103 → n = 52 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_congruence_l2809_280993


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_inequality_l2809_280977

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 2
def g (a x : ℝ) : ℝ := |x - a| - |x - 1|

-- Theorem for part 1
theorem solution_set_when_a_is_4 :
  {x : ℝ | f x > g 4 x} = {x : ℝ | x > 1 ∨ x ≤ -1} := by sorry

-- Theorem for part 2
theorem range_of_a_for_inequality :
  (∀ x₁ x₂ : ℝ, f x₁ ≥ g a x₂) ↔ -1 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_inequality_l2809_280977


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l2809_280972

def y : ℕ := 2^3^4^5^6^7^8^9

theorem smallest_multiplier_for_perfect_square (k : ℕ) : 
  k > 0 ∧ 
  ∃ m : ℕ, k * y = m^2 ∧ 
  ∀ l : ℕ, l > 0 ∧ l < k → ¬∃ n : ℕ, l * y = n^2 
  ↔ 
  k = 10 := by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l2809_280972


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l2809_280997

/-- Represents a square grid with shaded squares -/
structure SquareGrid :=
  (size : ℕ)
  (shaded : Set (ℕ × ℕ))

/-- Checks if a SquareGrid has at least one line of symmetry -/
def has_line_symmetry (grid : SquareGrid) : Prop :=
  sorry

/-- Checks if a SquareGrid has rotational symmetry of order 2 -/
def has_rotational_symmetry_order_2 (grid : SquareGrid) : Prop :=
  sorry

/-- Counts the number of additional squares shaded -/
def count_additional_shaded (initial grid : SquareGrid) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of additional squares to be shaded -/
theorem min_additional_squares_for_symmetry (initial : SquareGrid) :
  ∃ (final : SquareGrid),
    (has_line_symmetry final ∧ has_rotational_symmetry_order_2 final) ∧
    (count_additional_shaded initial final = 3) ∧
    (∀ (other : SquareGrid),
      (has_line_symmetry other ∧ has_rotational_symmetry_order_2 other) →
      count_additional_shaded initial other ≥ 3) :=
  sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l2809_280997


namespace NUMINAMATH_CALUDE_sets_and_range_theorem_l2809_280990

-- Define the sets A and B
def A : Set ℝ := {x | -x^2 + 3*x + 10 ≥ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem sets_and_range_theorem (m : ℝ) (h : B m ⊆ A) : 
  A = {x | -2 ≤ x ∧ x ≤ 5} ∧ m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sets_and_range_theorem_l2809_280990


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2809_280900

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : ¬(a = b ∧ b = c)) : 
  (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ≥ 
    (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ∧
  ((a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 = 
    (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ↔ 
    ((a = 0 ∧ b = 0 ∧ c > 0) ∨ (a = 0 ∧ c = 0 ∧ b > 0) ∨ (b = 0 ∧ c = 0 ∧ a > 0))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2809_280900


namespace NUMINAMATH_CALUDE_faye_pencil_count_l2809_280958

/-- Given that Faye arranges her pencils in rows of 5 and can make 7 full rows, 
    prove that she has 35 pencils. -/
theorem faye_pencil_count (pencils_per_row : ℕ) (num_rows : ℕ) 
  (h1 : pencils_per_row = 5)
  (h2 : num_rows = 7) : 
  pencils_per_row * num_rows = 35 := by
  sorry

#check faye_pencil_count

end NUMINAMATH_CALUDE_faye_pencil_count_l2809_280958


namespace NUMINAMATH_CALUDE_lcm_of_20_25_30_l2809_280947

theorem lcm_of_20_25_30 : Nat.lcm (Nat.lcm 20 25) 30 = 300 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_25_30_l2809_280947


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l2809_280975

/-- An arithmetic sequence with a given first term and second term -/
def arithmeticSequence (a₁ : ℚ) (a₂ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

/-- Theorem: The 12th term of the arithmetic sequence with first term 1/2 and second term 5/6 is 25/6 -/
theorem twelfth_term_of_specific_arithmetic_sequence :
  arithmeticSequence (1/2) (5/6) 12 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l2809_280975


namespace NUMINAMATH_CALUDE_grid_sum_bottom_corners_l2809_280963

/-- Represents a 3x3 grid where each cell contains a number -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Checks if a given number appears exactly once in each row -/
def rowValid (g : Grid) (n : Nat) : Prop :=
  ∀ i : Fin 3, ∃! j : Fin 3, g i j = n

/-- Checks if a given number appears exactly once in each column -/
def colValid (g : Grid) (n : Nat) : Prop :=
  ∀ j : Fin 3, ∃! i : Fin 3, g i j = n

/-- Checks if the grid contains only the numbers 4, 5, and 6 -/
def gridContainsOnly456 (g : Grid) : Prop :=
  ∀ i j : Fin 3, g i j = 4 ∨ g i j = 5 ∨ g i j = 6

/-- The main theorem statement -/
theorem grid_sum_bottom_corners (g : Grid) :
  rowValid g 4 ∧ rowValid g 5 ∧ rowValid g 6 ∧
  colValid g 4 ∧ colValid g 5 ∧ colValid g 6 ∧
  gridContainsOnly456 g ∧
  g 0 0 = 5 ∧ g 1 1 = 4 →
  g 2 0 + g 2 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_grid_sum_bottom_corners_l2809_280963


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2809_280923

theorem parallel_vectors_magnitude (k : ℝ) : 
  let a : Fin 2 → ℝ := ![(-1), 2]
  let b : Fin 2 → ℝ := ![2, k]
  (∃ (c : ℝ), a = c • b) →
  ‖2 • a - b‖ = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2809_280923


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_three_l2809_280996

theorem smallest_four_digit_divisible_by_three :
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ 
           n % 3 = 0 ∧
           (∀ m : ℕ, (1000 ≤ m ∧ m < n) → m % 3 ≠ 0) ∧
           n = 1002 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_three_l2809_280996


namespace NUMINAMATH_CALUDE_consumption_increase_l2809_280967

theorem consumption_increase (original_tax original_consumption : ℝ) 
  (h_tax_positive : original_tax > 0) 
  (h_consumption_positive : original_consumption > 0) : 
  ∃ (increase_percentage : ℝ),
    (original_tax * 0.8 * (original_consumption * (1 + increase_percentage / 100)) = 
     original_tax * original_consumption * 0.96) ∧
    increase_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_consumption_increase_l2809_280967
