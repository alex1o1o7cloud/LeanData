import Mathlib

namespace NUMINAMATH_CALUDE_supplement_of_forty_degrees_l1013_101353

/-- Given a system of parallel lines where an angle of 40° is formed, 
    prove that its supplement measures 140°. -/
theorem supplement_of_forty_degrees (α : Real) (h1 : α = 40) : 180 - α = 140 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_forty_degrees_l1013_101353


namespace NUMINAMATH_CALUDE_three_chapters_eight_pages_l1013_101382

/-- Calculates the total number of pages read given the number of chapters and pages per chapter -/
def pages_read (chapters : ℕ) (pages_per_chapter : ℕ) : ℕ :=
  chapters * pages_per_chapter

/-- Proves that reading 3 chapters of 8 pages each results in 24 pages read -/
theorem three_chapters_eight_pages :
  pages_read 3 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_three_chapters_eight_pages_l1013_101382


namespace NUMINAMATH_CALUDE_smallest_a1_l1013_101339

/-- A sequence of positive real numbers satisfying aₙ = 7aₙ₋₁ - n for n > 1 -/
def ValidSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 7 * a (n - 1) - n)

/-- The smallest possible value of a₁ in a valid sequence is 13/36 -/
theorem smallest_a1 :
    ∀ a : ℕ → ℝ, ValidSequence a → a 1 ≥ 13/36 ∧ ∃ a', ValidSequence a' ∧ a' 1 = 13/36 :=
  sorry

end NUMINAMATH_CALUDE_smallest_a1_l1013_101339


namespace NUMINAMATH_CALUDE_remainder_theorem_l1013_101340

theorem remainder_theorem : (2^210 + 210) % (2^105 + 2^63 + 1) = 210 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1013_101340


namespace NUMINAMATH_CALUDE_first_day_rain_l1013_101370

/-- The amount of rain Greg experienced while camping, given the known conditions -/
def camping_rain (first_day : ℝ) : ℝ := first_day + 6 + 5

/-- The amount of rain at Greg's house during the same week -/
def house_rain : ℝ := 26

/-- The difference in rain between Greg's house and his camping experience -/
def rain_difference : ℝ := 12

theorem first_day_rain : 
  ∃ (x : ℝ), camping_rain x = house_rain - rain_difference ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_first_day_rain_l1013_101370


namespace NUMINAMATH_CALUDE_dumplings_remaining_l1013_101328

theorem dumplings_remaining (cooked : ℕ) (eaten : ℕ) (h1 : cooked = 14) (h2 : eaten = 7) :
  cooked - eaten = 7 := by
  sorry

end NUMINAMATH_CALUDE_dumplings_remaining_l1013_101328


namespace NUMINAMATH_CALUDE_literary_readers_count_l1013_101335

theorem literary_readers_count (total : ℕ) (science_fiction : ℕ) (both : ℕ) 
  (h1 : total = 650) 
  (h2 : science_fiction = 250) 
  (h3 : both = 150) : 
  total = science_fiction + (550 : ℕ) - both :=
by sorry

end NUMINAMATH_CALUDE_literary_readers_count_l1013_101335


namespace NUMINAMATH_CALUDE_book_pages_count_l1013_101326

theorem book_pages_count : 
  let total_chapters : ℕ := 31
  let first_ten_pages : ℕ := 61
  let middle_ten_pages : ℕ := 59
  let last_eleven_pages : List ℕ := [58, 65, 62, 63, 64, 57, 66, 60, 59, 67]
  
  (10 * first_ten_pages) + 
  (10 * middle_ten_pages) + 
  (last_eleven_pages.sum) = 1821 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l1013_101326


namespace NUMINAMATH_CALUDE_kot_ycehyj_inequality_l1013_101385

theorem kot_ycehyj_inequality : 
  ∀ (K O T Y C E H J : ℕ),
    K ∈ Finset.range 9 ∧ 
    O ∈ Finset.range 9 ∧ 
    T ∈ Finset.range 9 ∧ 
    Y ∈ Finset.range 9 ∧ 
    C ∈ Finset.range 9 ∧ 
    E ∈ Finset.range 9 ∧ 
    H ∈ Finset.range 9 ∧ 
    J ∈ Finset.range 9 ∧
    K ≠ O ∧ K ≠ T ∧ K ≠ Y ∧ K ≠ C ∧ K ≠ E ∧ K ≠ H ∧ K ≠ J ∧
    O ≠ T ∧ O ≠ Y ∧ O ≠ C ∧ O ≠ E ∧ O ≠ H ∧ O ≠ J ∧
    T ≠ Y ∧ T ≠ C ∧ T ≠ E ∧ T ≠ H ∧ T ≠ J ∧
    Y ≠ C ∧ Y ≠ E ∧ Y ≠ H ∧ Y ≠ J ∧
    C ≠ E ∧ C ≠ H ∧ C ≠ J ∧
    E ≠ H ∧ E ≠ J ∧
    H ≠ J →
    K * O * T < Y * C * E * H * Y * J :=
by sorry

end NUMINAMATH_CALUDE_kot_ycehyj_inequality_l1013_101385


namespace NUMINAMATH_CALUDE_curve_tangent_problem_l1013_101307

/-- The curve C is defined by the equation y = 2x³ + ax + a -/
def C (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x + a

/-- The derivative of C with respect to x -/
def C_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 + a

theorem curve_tangent_problem (a : ℝ) :
  (C a (-1) = 0) →  -- C passes through point M(-1, 0)
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    C_derivative a t₁ + C_derivative a t₂ = 0 ∧  -- |MA| = |MB| condition
    4 * t₁^3 + 6 * t₁^2 = 0 ∧                   -- Tangent line condition for t₁
    4 * t₂^3 + 6 * t₂^2 = 0) →                  -- Tangent line condition for t₂
  a = -27/4 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_problem_l1013_101307


namespace NUMINAMATH_CALUDE_product_markup_rate_l1013_101398

theorem product_markup_rate (selling_price : ℝ) (profit_rate : ℝ) (expense_rate : ℝ) (fixed_cost : ℝ) :
  selling_price = 10 ∧ 
  profit_rate = 0.20 ∧ 
  expense_rate = 0.30 ∧ 
  fixed_cost = 1 →
  let variable_cost := selling_price * (1 - profit_rate - expense_rate) - fixed_cost
  (selling_price - variable_cost) / variable_cost = 1.5 := by sorry

end NUMINAMATH_CALUDE_product_markup_rate_l1013_101398


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1013_101389

-- Define the conditions
def condition_p (m : ℝ) : Prop := -1 < m ∧ m < 5

def condition_q (m : ℝ) : Prop :=
  ∀ x, x^2 - 2*m*x + m^2 - 1 = 0 → -2 < x ∧ x < 4

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ m, condition_q m → condition_p m) ∧
  (∃ m, condition_p m ∧ ¬condition_q m) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1013_101389


namespace NUMINAMATH_CALUDE_savannah_gift_wrapping_l1013_101304

/-- Given the conditions of Savannah's gift wrapping, prove that the first roll wraps 3 gifts -/
theorem savannah_gift_wrapping (total_rolls : ℕ) (total_gifts : ℕ) (second_roll_gifts : ℕ) (third_roll_gifts : ℕ) 
  (h1 : total_rolls = 3)
  (h2 : total_gifts = 12)
  (h3 : second_roll_gifts = 5)
  (h4 : third_roll_gifts = 4) :
  total_gifts - (second_roll_gifts + third_roll_gifts) = 3 := by
  sorry

end NUMINAMATH_CALUDE_savannah_gift_wrapping_l1013_101304


namespace NUMINAMATH_CALUDE_equation_solution_l1013_101388

theorem equation_solution : 
  ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 57 ∧ x = 92 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1013_101388


namespace NUMINAMATH_CALUDE_uphill_distance_is_six_l1013_101303

/-- Represents the travel problem with given conditions -/
structure TravelProblem where
  total_time : ℚ
  total_distance : ℕ
  speed_uphill : ℕ
  speed_flat : ℕ
  speed_downhill : ℕ

/-- Checks if a solution satisfies the problem conditions -/
def is_valid_solution (problem : TravelProblem) (uphill_distance : ℕ) (flat_distance : ℕ) : Prop :=
  let downhill_distance := problem.total_distance - uphill_distance - flat_distance
  uphill_distance + flat_distance ≤ problem.total_distance ∧
  (uphill_distance : ℚ) / problem.speed_uphill +
  (flat_distance : ℚ) / problem.speed_flat +
  (downhill_distance : ℚ) / problem.speed_downhill = problem.total_time

/-- The main theorem stating that 6 km is the correct uphill distance -/
theorem uphill_distance_is_six (problem : TravelProblem) 
  (h1 : problem.total_time = 67 / 30)
  (h2 : problem.total_distance = 10)
  (h3 : problem.speed_uphill = 4)
  (h4 : problem.speed_flat = 5)
  (h5 : problem.speed_downhill = 6) :
  ∃ (flat_distance : ℕ), is_valid_solution problem 6 flat_distance ∧
  ∀ (other_uphill : ℕ) (other_flat : ℕ),
    other_uphill ≠ 6 → ¬ is_valid_solution problem other_uphill other_flat :=
by sorry


end NUMINAMATH_CALUDE_uphill_distance_is_six_l1013_101303


namespace NUMINAMATH_CALUDE_circles_intersecting_parallel_lines_l1013_101352

-- Define the types for our objects
variable (Point Circle Line : Type)

-- Define the necessary relations and properties
variable (onCircle : Point → Circle → Prop)
variable (intersectsAt : Circle → Circle → Point → Prop)
variable (passesThrough : Line → Point → Prop)
variable (intersectsCircleAt : Line → Circle → Point → Prop)
variable (parallel : Line → Line → Prop)
variable (lineThroughPoints : Point → Point → Line)

-- State the theorem
theorem circles_intersecting_parallel_lines
  (Γ₁ Γ₂ : Circle)
  (P Q A A' B B' : Point) :
  intersectsAt Γ₁ Γ₂ P →
  intersectsAt Γ₁ Γ₂ Q →
  (∃ l : Line, passesThrough l P ∧ intersectsCircleAt l Γ₁ A ∧ intersectsCircleAt l Γ₂ A') →
  (∃ m : Line, passesThrough m Q ∧ intersectsCircleAt m Γ₁ B ∧ intersectsCircleAt m Γ₂ B') →
  A ≠ P →
  A' ≠ P →
  B ≠ Q →
  B' ≠ Q →
  parallel (lineThroughPoints A B) (lineThroughPoints A' B') :=
sorry

end NUMINAMATH_CALUDE_circles_intersecting_parallel_lines_l1013_101352


namespace NUMINAMATH_CALUDE_construct_equilateral_from_given_l1013_101333

-- Define the given triangle
structure GivenTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_angles : angle1 + angle2 + angle3 = 180
  angle_values : angle1 = 40 ∧ angle2 = 70 ∧ angle3 = 70

-- Define an equilateral triangle
def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

-- Theorem statement
theorem construct_equilateral_from_given (t : GivenTriangle) :
  ∃ (a b c : ℝ), is_equilateral a b c :=
sorry


end NUMINAMATH_CALUDE_construct_equilateral_from_given_l1013_101333


namespace NUMINAMATH_CALUDE_daughters_age_in_three_years_l1013_101317

/-- Given that 5 years ago, a mother was twice as old as her daughter, and the mother is 41 years old now,
    prove that the daughter will be 26 years old in 3 years. -/
theorem daughters_age_in_three_years 
  (mother_age_now : ℕ) 
  (mother_daughter_age_relation : ℕ → ℕ → Prop) 
  (h1 : mother_age_now = 41)
  (h2 : mother_daughter_age_relation (mother_age_now - 5) ((mother_age_now - 5) / 2)) :
  ((mother_age_now - 5) / 2) + 8 = 26 := by
  sorry

#check daughters_age_in_three_years

end NUMINAMATH_CALUDE_daughters_age_in_three_years_l1013_101317


namespace NUMINAMATH_CALUDE_circle_center_l1013_101301

/-- The center of a circle given by the equation x^2 - 4x + y^2 - 6y - 12 = 0 is (2, 3) -/
theorem circle_center (x y : ℝ) : 
  x^2 - 4*x + y^2 - 6*y - 12 = 0 → (2, 3) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l1013_101301


namespace NUMINAMATH_CALUDE_postage_for_5_5_ounces_l1013_101373

/-- Calculates the postage for a letter given its weight and the rate structure -/
def calculatePostage (weight : ℚ) : ℚ :=
  let baseRate : ℚ := 25 / 100  -- 25 cents
  let additionalRate : ℚ := 18 / 100  -- 18 cents
  let overweightSurcharge : ℚ := 10 / 100  -- 10 cents
  let overweightThreshold : ℚ := 3  -- 3 ounces
  
  let additionalWeight := max (weight - 1) 0
  let additionalCharges := ⌈additionalWeight⌉
  
  let cost := baseRate + additionalRate * additionalCharges
  if weight > overweightThreshold then
    cost + overweightSurcharge
  else
    cost

/-- Theorem stating that the postage for a 5.5 ounce letter is $1.25 -/
theorem postage_for_5_5_ounces :
  calculatePostage (11/2) = 5/4 := by sorry

end NUMINAMATH_CALUDE_postage_for_5_5_ounces_l1013_101373


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l1013_101338

theorem quadratic_integer_roots (a : ℤ) :
  (∃ x y : ℤ, (a + 1) * x^2 - (a^2 + 1) * x + 2 * a^3 - 6 = 0 ∧
               (a + 1) * y^2 - (a^2 + 1) * y + 2 * a^3 - 6 = 0 ∧
               x ≠ y) ↔
  (a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l1013_101338


namespace NUMINAMATH_CALUDE_power_equality_l1013_101322

theorem power_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1013_101322


namespace NUMINAMATH_CALUDE_series_sum_8_eq_43690_l1013_101375

def series_sum : ℕ → ℕ 
  | 0 => 2
  | n + 1 => 2 * (1 + 4 * series_sum n)

theorem series_sum_8_eq_43690 : series_sum 7 = 43690 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_8_eq_43690_l1013_101375


namespace NUMINAMATH_CALUDE_age_difference_l1013_101309

/-- Given that the total age of A and B is 13 years more than the total age of B and C,
    prove that C is 13 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 13) : A = C + 13 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1013_101309


namespace NUMINAMATH_CALUDE_line_segment_point_sum_l1013_101315

/-- Given a line y = -2/3x + 6, prove that the sum of coordinates of point T
    satisfies r + s = 8.25, where T(r,s) is on PQ, P and Q are x and y intercepts,
    and area of POQ is 4 times area of TOP. -/
theorem line_segment_point_sum (x₁ y₁ r s : ℝ) : 
  y₁ = 6 ∧                        -- Q is (0, y₁)
  x₁ = 9 ∧                        -- P is (x₁, 0)
  s = -2/3 * r + 6 ∧              -- T(r,s) is on the line
  0 ≤ r ∧ r ≤ x₁ ∧                -- T is between P and Q
  1/2 * x₁ * y₁ = 4 * (1/2 * r * s) -- Area POQ = 4 * Area TOP
  → r + s = 8.25 := by
    sorry

end NUMINAMATH_CALUDE_line_segment_point_sum_l1013_101315


namespace NUMINAMATH_CALUDE_min_values_theorem_l1013_101384

theorem min_values_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : Real.log a + Real.log b = Real.log (a + 9*b)) : 
  (a * b ≥ 36) ∧ ((81 / a^2) + (1 / b^2) ≥ 1/2) ∧ (a + b ≥ 16) := by
  sorry

end NUMINAMATH_CALUDE_min_values_theorem_l1013_101384


namespace NUMINAMATH_CALUDE_f_derivative_l1013_101365

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem f_derivative (x : ℝ) (hx : x ≠ 0) :
  deriv f x = (Real.exp x * (x - 1)) / (x^2) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_l1013_101365


namespace NUMINAMATH_CALUDE_complex_number_equality_l1013_101351

theorem complex_number_equality (z : ℂ) : z = 2 - (13 / 6) * I →
  Complex.abs (z - 2) = Complex.abs (z + 2) ∧
  Complex.abs (z - 2) = Complex.abs (z - 3 * I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1013_101351


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l1013_101360

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l1013_101360


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1013_101371

/-- The perimeter of a rhombus with diagonals of 12 inches and 16 inches is 40 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 40 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1013_101371


namespace NUMINAMATH_CALUDE_power_equation_solution_l1013_101399

theorem power_equation_solution (n : ℕ) : 2^n = 8^20 → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1013_101399


namespace NUMINAMATH_CALUDE_complex_number_equality_l1013_101314

theorem complex_number_equality : Complex.I - (1 / Complex.I) = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1013_101314


namespace NUMINAMATH_CALUDE_first_movie_length_proof_l1013_101363

/-- Represents the length of the first movie in hours -/
def first_movie_length : ℝ := 3.5

/-- Represents the length of the second movie in hours -/
def second_movie_length : ℝ := 1.5

/-- Represents the total available time in hours -/
def total_time : ℝ := 8

/-- Represents the reading rate in words per minute -/
def reading_rate : ℝ := 10

/-- Represents the total number of words read -/
def total_words_read : ℝ := 1800

/-- Proves that given the conditions, the length of the first movie must be 3.5 hours -/
theorem first_movie_length_proof :
  first_movie_length + second_movie_length + (total_words_read / reading_rate / 60) = total_time :=
by sorry

end NUMINAMATH_CALUDE_first_movie_length_proof_l1013_101363


namespace NUMINAMATH_CALUDE_semicircle_area_theorem_l1013_101350

noncomputable def semicircle_area (P Q R S T U : Point) : ℝ :=
  let PQ_radius := 2
  let PS_length := Real.sqrt 2
  let QS_length := Real.sqrt 2
  let PT_radius := PQ_radius / 2
  let QU_radius := PQ_radius / 2
  let TU_radius := PS_length
  let triangle_PQS_area := PQ_radius * (Real.sqrt 2) / 2
  (PT_radius^2 * Real.pi / 2) + (QU_radius^2 * Real.pi / 2) + (TU_radius^2 * Real.pi / 2) - triangle_PQS_area

theorem semicircle_area_theorem (P Q R S T U : Point) :
  semicircle_area P Q R S T U = 9 * Real.pi - 2 :=
sorry

end NUMINAMATH_CALUDE_semicircle_area_theorem_l1013_101350


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1013_101356

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(10 ∣ (724946 - y))) ∧ 
  (10 ∣ (724946 - x)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1013_101356


namespace NUMINAMATH_CALUDE_three_eighths_percent_of_160_l1013_101381

theorem three_eighths_percent_of_160 : (3 / 8 / 100) * 160 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_three_eighths_percent_of_160_l1013_101381


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1013_101387

def average_age_of_team : ℝ := 25

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (wicket_keeper_age : ℝ) 
  (remaining_players_average_age : ℝ) :
  team_size = 11 →
  wicket_keeper_age = average_age_of_team + 3 →
  remaining_players_average_age = average_age_of_team - 1 →
  average_age_of_team * team_size = 
    wicket_keeper_age + 
    (team_size - 2) * remaining_players_average_age + 
    (average_age_of_team * team_size - wicket_keeper_age - (team_size - 2) * remaining_players_average_age) →
  average_age_of_team = 25 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l1013_101387


namespace NUMINAMATH_CALUDE_profit_percentage_15_20_l1013_101308

/-- Represents the profit percentage when selling articles -/
def profit_percentage (sold : ℕ) (cost_equivalent : ℕ) : ℚ :=
  (cost_equivalent - sold) / sold

/-- Theorem: The profit percentage when selling 15 articles at the cost of 20 is 1/3 -/
theorem profit_percentage_15_20 : profit_percentage 15 20 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_15_20_l1013_101308


namespace NUMINAMATH_CALUDE_marys_age_l1013_101362

/-- Given that Suzy is 20 years old now and in four years she will be twice Mary's age,
    prove that Mary is currently 8 years old. -/
theorem marys_age (suzy_age : ℕ) (mary_age : ℕ) : 
  suzy_age = 20 → 
  (suzy_age + 4 = 2 * (mary_age + 4)) → 
  mary_age = 8 := by
sorry

end NUMINAMATH_CALUDE_marys_age_l1013_101362


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1013_101336

theorem decimal_to_fraction : 
  (3.36 : ℚ) = 84 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1013_101336


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l1013_101395

theorem no_positive_integer_solution :
  ¬ ∃ (x y : ℕ+), x^2006 - 4*y^2006 - 2006 = 4*y^2007 + 2007*y := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l1013_101395


namespace NUMINAMATH_CALUDE_field_trip_capacity_l1013_101325

theorem field_trip_capacity (seats_per_bus : ℕ) (num_buses : ℕ) : 
  let max_students := seats_per_bus * num_buses
  seats_per_bus = 60 → num_buses = 3 → max_students = 180 := by
sorry

end NUMINAMATH_CALUDE_field_trip_capacity_l1013_101325


namespace NUMINAMATH_CALUDE_cost_not_proportional_cost_increases_linearly_l1013_101347

/-- Represents the cost of a telegram -/
def telegram_cost (a b n : ℝ) : ℝ := a + b * n

/-- The cost is not proportional to the number of words -/
theorem cost_not_proportional (a b : ℝ) (h : a ≠ 0) :
  ¬∃ k : ℝ, ∀ n : ℝ, telegram_cost a b n = k * n :=
sorry

/-- The cost increases linearly with the number of words -/
theorem cost_increases_linearly (a b : ℝ) (h : b > 0) :
  ∀ n₁ n₂ : ℝ, n₁ < n₂ → telegram_cost a b n₁ < telegram_cost a b n₂ :=
sorry

end NUMINAMATH_CALUDE_cost_not_proportional_cost_increases_linearly_l1013_101347


namespace NUMINAMATH_CALUDE_building_height_l1013_101366

/-- Given a flagpole and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_shadow_pos : 0 < building_shadow)
  (h : flagpole_height = 18)
  (s1 : flagpole_shadow = 45)
  (s2 : building_shadow = 60) :
  flagpole_height / flagpole_shadow * building_shadow = 24 := by
sorry


end NUMINAMATH_CALUDE_building_height_l1013_101366


namespace NUMINAMATH_CALUDE_triangle_properties_l1013_101380

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that under certain conditions, we can determine the values of A, b, and c. -/
theorem triangle_properties (t : Triangle) 
  (h1 : ∃ (k : ℝ), k * (Real.sqrt 3 * t.a) = t.c ∧ k * (1 + Real.cos t.A) = Real.sin t.C) 
  (h2 : 3 * t.b * t.c = 16 - t.a^2)
  (h3 : t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1013_101380


namespace NUMINAMATH_CALUDE_school_vote_total_l1013_101386

theorem school_vote_total (x : ℝ) : 
  (0.35 * x = 0.65 * x) ∧ 
  (0.45 * (x + 80) = 0.65 * x) →
  x + 80 = 260 := by
sorry

end NUMINAMATH_CALUDE_school_vote_total_l1013_101386


namespace NUMINAMATH_CALUDE_red_marbles_after_replacement_l1013_101342

theorem red_marbles_after_replacement (total : ℕ) (blue green red : ℕ) : 
  total > 0 →
  blue = (40 * total + 99) / 100 →
  green = 20 →
  red = (10 * total + 99) / 100 →
  (15 * total + 99) / 100 + (5 * total + 99) / 100 + blue + green + red = total →
  (16 : ℕ) = red + blue / 3 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_after_replacement_l1013_101342


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1013_101354

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 5) * (Real.sqrt 7 / Real.sqrt 11) * (Real.sqrt 15 / Real.sqrt 2) = 
  (3 * Real.sqrt 154) / 22 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1013_101354


namespace NUMINAMATH_CALUDE_coin_toss_probability_l1013_101343

theorem coin_toss_probability (p : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : p ^ 5 = 0.0625) :
  p = 0.5 := by
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l1013_101343


namespace NUMINAMATH_CALUDE_union_of_sets_l1013_101349

theorem union_of_sets : 
  let M : Set ℤ := {4, -3}
  let N : Set ℤ := {0, -3}
  M ∪ N = {0, -3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l1013_101349


namespace NUMINAMATH_CALUDE_cooler_capacity_l1013_101390

theorem cooler_capacity (c1 c2 c3 : ℝ) : 
  c1 = 100 → 
  c2 = c1 * 1.5 → 
  c3 = c2 / 2 → 
  c1 + c2 + c3 = 325 := by
sorry

end NUMINAMATH_CALUDE_cooler_capacity_l1013_101390


namespace NUMINAMATH_CALUDE_cos_theta_minus_phi_l1013_101324

theorem cos_theta_minus_phi (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I →
  Complex.exp (φ * Complex.I) = (5 / 13 : ℂ) - (12 / 13 : ℂ) * Complex.I →
  Real.cos (θ - φ) = -16 / 65 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_minus_phi_l1013_101324


namespace NUMINAMATH_CALUDE_solve_equation_l1013_101337

theorem solve_equation (x : ℝ) : 3 * x + 20 = (1/3) * (7 * x + 45) → x = -7.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1013_101337


namespace NUMINAMATH_CALUDE_greatest_x_value_l1013_101372

theorem greatest_x_value (x : ℤ) (h : 2.134 * (10 : ℝ) ^ (x : ℝ) < 220000) :
  x ≤ 5 ∧ 2.134 * (10 : ℝ) ^ (5 : ℝ) < 220000 := by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1013_101372


namespace NUMINAMATH_CALUDE_unique_digit_system_solution_l1013_101396

theorem unique_digit_system_solution (a b c t x y : ℕ) 
  (unique_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ t ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧
                    a ≠ b ∧ a ≠ c ∧ a ≠ t ∧ a ≠ x ∧ a ≠ y ∧
                    b ≠ c ∧ b ≠ t ∧ b ≠ x ∧ b ≠ y ∧
                    c ≠ t ∧ c ≠ x ∧ c ≠ y ∧
                    t ≠ x ∧ t ≠ y ∧
                    x ≠ y)
  (eq1 : a + b = x)
  (eq2 : x + c = t)
  (eq3 : t + a = y)
  (eq4 : b + c + y = 20) :
  t = 10 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_system_solution_l1013_101396


namespace NUMINAMATH_CALUDE_orange_stack_sum_l1013_101393

def pyramid_stack (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let layers := min base_width base_length
  List.range layers
    |> List.map (λ i => (base_width - i) * (base_length - i))
    |> List.sum

theorem orange_stack_sum :
  pyramid_stack 6 9 = 155 := by
  sorry

end NUMINAMATH_CALUDE_orange_stack_sum_l1013_101393


namespace NUMINAMATH_CALUDE_largest_common_value_l1013_101377

/-- The first arithmetic progression -/
def progression1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression -/
def progression2 (n : ℕ) : ℕ := 5 + 9 * n

/-- A common term of both progressions -/
def commonTerm (m : ℕ) : ℕ := 14 + 45 * m

theorem largest_common_value :
  (∃ n1 n2 : ℕ, progression1 n1 = 959 ∧ progression2 n2 = 959) ∧ 
  (∀ k : ℕ, k < 1000 → k > 959 → 
    (∀ n1 n2 : ℕ, progression1 n1 ≠ k ∨ progression2 n2 ≠ k)) :=
sorry

end NUMINAMATH_CALUDE_largest_common_value_l1013_101377


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l1013_101357

/-- Given real numbers a and b where ab ≠ 0, prove that ax - y + b = 0 represents a line
    and bx² + ay² = ab represents an ellipse -/
theorem line_intersects_ellipse (a b : ℝ) (h : a * b ≠ 0) :
  ∃ (line : ℝ → ℝ) (ellipse : Set (ℝ × ℝ)),
    (∀ x y, ax - y + b = 0 ↔ y = line x) ∧
    (∀ x y, (x, y) ∈ ellipse ↔ b * x^2 + a * y^2 = a * b) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l1013_101357


namespace NUMINAMATH_CALUDE_system_solutions_l1013_101327

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x + y = (5 * x * y) / (1 + x * y) ∧
  y + z = (6 * y * z) / (1 + y * z) ∧
  z + x = (7 * z * x) / (1 + z * x)

-- Define the set of solutions
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0),
   ((3 + Real.sqrt 5) / 2, 1, 2 + Real.sqrt 3),
   ((3 + Real.sqrt 5) / 2, 1, 2 - Real.sqrt 3),
   ((3 - Real.sqrt 5) / 2, 1, 2 + Real.sqrt 3),
   ((3 - Real.sqrt 5) / 2, 1, 2 - Real.sqrt 3)}

-- Theorem statement
theorem system_solutions :
  ∀ x y z, system x y z ↔ (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l1013_101327


namespace NUMINAMATH_CALUDE_triangle_side_length_range_l1013_101364

theorem triangle_side_length_range (b : ℝ) (B : ℝ) :
  b = 2 →
  B = π / 3 →
  ∃ (a : ℝ), 2 < a ∧ a < 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_range_l1013_101364


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1013_101383

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → f (-x) = f x

theorem solution_set_of_inequality
  (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_deriv : ∀ x > 0, x * f' x > -2 * f x)
  (g : ℝ → ℝ) (h_g : ∀ x, g x = x^2 * f x) :
  {x : ℝ | g x < g (1 - x)} = {x : ℝ | x < 0 ∨ (0 < x ∧ x < 1/2)} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1013_101383


namespace NUMINAMATH_CALUDE_dany_farm_bushels_l1013_101319

/-- The number of bushels needed for Dany's farm animals for one day -/
def bushels_needed (cows sheep : ℕ) (cow_sheep_bushels : ℕ) (chickens : ℕ) (chicken_bushels : ℕ) : ℕ :=
  (cows + sheep) * cow_sheep_bushels + chicken_bushels

/-- Theorem: Dany needs 17 bushels for his animals for one day -/
theorem dany_farm_bushels :
  bushels_needed 4 3 2 7 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_dany_farm_bushels_l1013_101319


namespace NUMINAMATH_CALUDE_phone_time_proof_l1013_101320

/-- 
Given a person who spends time on the phone for 5 days, 
doubling the time each day after the first, 
and spending a total of 155 minutes,
prove that they spent 5 minutes on the first day.
-/
theorem phone_time_proof (x : ℝ) : 
  x + 2*x + 4*x + 8*x + 16*x = 155 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_phone_time_proof_l1013_101320


namespace NUMINAMATH_CALUDE_expected_digits_is_correct_l1013_101318

/-- A fair 20-sided die with numbers 1 to 20 -/
def icosahedral_die : Finset ℕ := Finset.range 20

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the die -/
def expected_digits : ℚ :=
  (icosahedral_die.sum (λ i => num_digits (i + 1))) / icosahedral_die.card

/-- Theorem: The expected number of digits is 1.55 -/
theorem expected_digits_is_correct :
  expected_digits = 31 / 20 := by sorry

end NUMINAMATH_CALUDE_expected_digits_is_correct_l1013_101318


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_three_l1013_101331

/-- A quadratic function f(x) = x^2 - 2ax + a - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a - 3

/-- The theorem stating that if f(x) is monotonically decreasing on (-∞, -1/4),
    then a ≤ -3 -/
theorem monotone_decreasing_implies_a_leq_neg_three (a : ℝ) :
  (∀ x y, x < y → x < -1/4 → f a x > f a y) → a ≤ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_three_l1013_101331


namespace NUMINAMATH_CALUDE_jenny_sleep_hours_l1013_101323

theorem jenny_sleep_hours (minutes_per_hour : ℕ) (total_sleep_minutes : ℕ) 
  (h1 : minutes_per_hour = 60) 
  (h2 : total_sleep_minutes = 480) : 
  total_sleep_minutes / minutes_per_hour = 8 := by
sorry

end NUMINAMATH_CALUDE_jenny_sleep_hours_l1013_101323


namespace NUMINAMATH_CALUDE_price_per_large_bottle_l1013_101341

/-- The price per large bottle, given the number of large and small bottles,
    the price of small bottles, and the average price of all bottles. -/
theorem price_per_large_bottle (large_count small_count : ℕ)
                                (small_price avg_price : ℚ) :
  large_count = 1325 →
  small_count = 750 →
  small_price = 138/100 →
  avg_price = 17057/10000 →
  ∃ (large_price : ℚ), 
    (large_count * large_price + small_count * small_price) / (large_count + small_count) = avg_price ∧
    abs (large_price - 189/100) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_price_per_large_bottle_l1013_101341


namespace NUMINAMATH_CALUDE_fraction_equality_l1013_101321

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 8)
  (h2 : c / b = 4)
  (h3 : c / d = 1 / 3) :
  d / a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1013_101321


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1013_101369

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point :
  let l1 : Line := { a := 2, b := -3, c := 9 }
  let l2 : Line := { a := 3, b := 2, c := -1 }
  let p : Point := { x := -1, y := 2 }
  perpendicular l1 l2 ∧ pointOnLine p l2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1013_101369


namespace NUMINAMATH_CALUDE_parabola_properties_l1013_101361

-- Define the parabola C
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (1, 0)

-- Define the directrix of the parabola
def Directrix (x : ℝ) : Prop := x = -1

-- Define a point on the parabola
def PointOnParabola (p : ℝ × ℝ) : Prop := Parabola p.1 p.2

-- Define a line passing through two points
def Line (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Theorem statement
theorem parabola_properties :
  ∀ (M N : ℝ × ℝ),
  Directrix M.1 ∧ Directrix N.1 →
  M.2 * N.2 = -4 →
  ∃ (A B : ℝ × ℝ) (F : ℝ × ℝ → ℝ × ℝ),
    PointOnParabola A ∧ PointOnParabola B ∧
    Line (0, 0) M A.1 A.2 ∧
    Line (0, 0) N B.1 B.2 ∧
    (∀ (x y : ℝ), Line A B x y → Line A B (F A).1 (F A).2) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1013_101361


namespace NUMINAMATH_CALUDE_polynomial_divisibility_implies_specific_coefficients_l1013_101310

theorem polynomial_divisibility_implies_specific_coefficients :
  ∀ (p q : ℝ),
  (∀ x : ℝ, (x + 3) * (x - 2) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x + 9)) →
  p = -19.5 ∧ q = -55.5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_implies_specific_coefficients_l1013_101310


namespace NUMINAMATH_CALUDE_second_number_proof_l1013_101330

theorem second_number_proof (x : ℤ) (h1 : x + (x + 4) = 56) : x + 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l1013_101330


namespace NUMINAMATH_CALUDE_mixed_fraction_decimal_calculation_l1013_101334

theorem mixed_fraction_decimal_calculation :
  let a : ℚ := 84 + 4 / 19
  let b : ℚ := 105 + 5 / 19
  let c : ℚ := 1.375
  let d : ℚ := 0.8
  a * c + b * d = 200 := by sorry

end NUMINAMATH_CALUDE_mixed_fraction_decimal_calculation_l1013_101334


namespace NUMINAMATH_CALUDE_slope_range_l1013_101312

theorem slope_range (a : ℝ) : 
  (∃ x y : ℝ, (a^2 + 2*a)*x - y + 1 = 0 ∧ a^2 + 2*a < 0) ↔ 
  -2 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_slope_range_l1013_101312


namespace NUMINAMATH_CALUDE_yasna_reading_schedule_l1013_101345

/-- The number of pages Yasna needs to read daily to finish two books in two weeks -/
def pages_per_day (book1_pages book2_pages : ℕ) (days : ℕ) : ℕ :=
  (book1_pages + book2_pages) / days

theorem yasna_reading_schedule :
  pages_per_day 180 100 14 = 20 := by
  sorry

end NUMINAMATH_CALUDE_yasna_reading_schedule_l1013_101345


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1013_101376

/-- Configuration of squares and rectangle forming a large square -/
structure SquareConfiguration where
  s : ℝ  -- Side length of small squares
  large_square_side : ℝ  -- Side length of the large square
  rectangle_length : ℝ  -- Length of the rectangle
  rectangle_width : ℝ  -- Width of the rectangle

/-- Properties of the square configuration -/
def valid_configuration (config : SquareConfiguration) : Prop :=
  config.s > 0 ∧
  config.large_square_side = 3 * config.s ∧
  config.rectangle_length = config.large_square_side ∧
  config.rectangle_width = config.s

/-- Theorem stating the ratio of rectangle's length to width -/
theorem rectangle_ratio (config : SquareConfiguration) 
  (h : valid_configuration config) : 
  config.rectangle_length / config.rectangle_width = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1013_101376


namespace NUMINAMATH_CALUDE_complex_number_sum_parts_l1013_101367

theorem complex_number_sum_parts (a : ℝ) : 
  let z : ℂ := a / (2 - Complex.I) + (3 - 4 * Complex.I) / 5
  (z.re + z.im = 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_parts_l1013_101367


namespace NUMINAMATH_CALUDE_f_range_l1013_101359

def f (x : ℝ) := 2 * x^2 + 4 * x + 1

theorem f_range : ∀ x ∈ Set.Icc (-2 : ℝ) 4, 
  -1 ≤ f x ∧ f x ≤ 49 ∧ 
  (∃ x₁ ∈ Set.Icc (-2 : ℝ) 4, f x₁ = -1) ∧
  (∃ x₂ ∈ Set.Icc (-2 : ℝ) 4, f x₂ = 49) :=
by sorry

end NUMINAMATH_CALUDE_f_range_l1013_101359


namespace NUMINAMATH_CALUDE_original_machines_work_hours_l1013_101311

/-- The number of original machines in the factory -/
def original_machines : ℕ := 3

/-- The number of hours the new machine works per day -/
def new_machine_hours : ℕ := 12

/-- The production rate of each machine in kg per hour -/
def production_rate : ℕ := 2

/-- The selling price of the material in dollars per kg -/
def selling_price : ℕ := 50

/-- The total earnings of the factory in one day in dollars -/
def total_earnings : ℕ := 8100

/-- Theorem stating that the original machines work 23 hours a day -/
theorem original_machines_work_hours : 
  ∃ h : ℕ, 
    (original_machines * production_rate * h + new_machine_hours * production_rate) * selling_price = total_earnings ∧ 
    h = 23 := by
  sorry

end NUMINAMATH_CALUDE_original_machines_work_hours_l1013_101311


namespace NUMINAMATH_CALUDE_sledding_time_difference_l1013_101379

/-- Given the conditions of Mary and Ann's sledding trip, prove that Ann's trip takes 13 minutes longer than Mary's. -/
theorem sledding_time_difference 
  (mary_hill_length : ℝ) 
  (mary_speed : ℝ) 
  (ann_hill_length : ℝ) 
  (ann_speed : ℝ) 
  (h1 : mary_hill_length = 630)
  (h2 : mary_speed = 90)
  (h3 : ann_hill_length = 800)
  (h4 : ann_speed = 40) :
  ann_hill_length / ann_speed - mary_hill_length / mary_speed = 13 := by
  sorry

end NUMINAMATH_CALUDE_sledding_time_difference_l1013_101379


namespace NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l1013_101348

-- Define a quadratic function
def quadratic (m n t : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + t

-- Define a fixed point
def is_fixed_point (m n t : ℝ) (x : ℝ) : Prop := quadratic m n t x = x

theorem fixed_points_of_specific_quadratic :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point 1 (-1) (-3) x1 ∧ is_fixed_point 1 (-1) (-3) x2 ∧ x1 = -1 ∧ x2 = 3 := by sorry

theorem min_value_of_ratio_sum :
  ∀ a : ℝ, a > 1 →
  ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧
  is_fixed_point 2 (-(3+a)) (a-1) x1 ∧
  is_fixed_point 2 (-(3+a)) (a-1) x2 →
  (x1 / x2 + x2 / x1 ≥ 8) ∧ (∃ a0 : ℝ, a0 > 1 ∧ ∃ x3 x4 : ℝ, x3 / x4 + x4 / x3 = 8) := by sorry

theorem range_of_a_for_always_fixed_point :
  ∀ a : ℝ, a ≠ 0 →
  (∀ b : ℝ, ∃ x : ℝ, is_fixed_point a (b+1) (b-1) x) ↔
  (a > 0 ∧ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l1013_101348


namespace NUMINAMATH_CALUDE_no_integer_tangent_length_l1013_101358

-- Define the circle and point P
def Circle : Type := Unit
def Point : Type := Unit

-- Define the tangent and secant
def tangent (p : Point) (c : Circle) : ℝ := sorry
def secant (p : Point) (c : Circle) : ℝ × ℝ := sorry

-- State the theorem
theorem no_integer_tangent_length 
  (P : Point) (C : Circle) 
  (h1 : (secant P C).1 = 2 * (secant P C).2)  -- m = 2n
  (h2 : (secant P C).1 + (secant P C).2 = 18) -- circumference = 18
  (h3 : (tangent P C)^2 = (secant P C).1 * (secant P C).2) -- t^2 = mn
  : ¬ ∃ (t : ℕ), (tangent P C) = t := by
  sorry


end NUMINAMATH_CALUDE_no_integer_tangent_length_l1013_101358


namespace NUMINAMATH_CALUDE_equation_solution_l1013_101344

theorem equation_solution (x y c : ℝ) : 
  7^(3*x - 1) * 3^(4*y - 3) = c^x * 27^y ∧ x + y = 4 → c = 49 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1013_101344


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l1013_101306

/-- 
Given a quadratic expression of the form 6x^2 + nx + 72, 
this theorem states that the largest value of n for which 
the expression can be factored as the product of two linear 
factors with integer coefficients is 433.
-/
theorem largest_n_for_factorization : 
  (∀ n : ℤ, ∃ a b c d : ℤ, 
    (6 * x^2 + n * x + 72 = (a * x + b) * (c * x + d)) → n ≤ 433) ∧ 
  (∃ a b c d : ℤ, 6 * x^2 + 433 * x + 72 = (a * x + b) * (c * x + d)) := by
sorry


end NUMINAMATH_CALUDE_largest_n_for_factorization_l1013_101306


namespace NUMINAMATH_CALUDE_fruit_selection_ways_l1013_101313

/-- The number of ways to choose k items from n distinct items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of fruits in the basket -/
def num_fruits : ℕ := 5

/-- The number of fruits to be selected -/
def num_selected : ℕ := 2

/-- Theorem: There are 10 ways to select 2 fruits from a basket of 5 fruits -/
theorem fruit_selection_ways : choose num_fruits num_selected = 10 := by
  sorry

end NUMINAMATH_CALUDE_fruit_selection_ways_l1013_101313


namespace NUMINAMATH_CALUDE_newton_family_mean_age_l1013_101378

theorem newton_family_mean_age :
  let ages : List ℝ := [6, 6, 9, 12]
  let mean := (ages.sum) / (ages.length)
  mean = 8.25 := by
sorry

end NUMINAMATH_CALUDE_newton_family_mean_age_l1013_101378


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l1013_101368

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n*(x^4 + y^4 + z^4 + w^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 > m*(x^4 + y^4 + z^4 + w^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l1013_101368


namespace NUMINAMATH_CALUDE_main_theorem_l1013_101397

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f (f y))^2)

/-- The main theorem to prove -/
theorem main_theorem (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
sorry

end NUMINAMATH_CALUDE_main_theorem_l1013_101397


namespace NUMINAMATH_CALUDE_intersection_inequality_solution_l1013_101302

/-- Given two linear functions y₁ = ax + b and y₂ = cx + d with a > c > 0,
    intersecting at the point (2, m), prove that the solution set of
    the inequality (a-c)x ≤ d-b is x ≤ 2. -/
theorem intersection_inequality_solution
  (a b c d m : ℝ)
  (h1 : a > c)
  (h2 : c > 0)
  (h3 : a * 2 + b = c * 2 + d)
  (h4 : a * 2 + b = m) :
  ∀ x, (a - c) * x ≤ d - b ↔ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_inequality_solution_l1013_101302


namespace NUMINAMATH_CALUDE_cafeteria_bags_l1013_101300

theorem cafeteria_bags (total : ℕ) (x : ℕ) : 
  total = 351 → 
  (x + 20) - 3 * ((total - x) - 50) = 1 → 
  x = 221 ∧ (total - x) = 130 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_bags_l1013_101300


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l1013_101374

theorem no_perfect_square_in_range : 
  ¬ ∃ (n : ℕ), 4 ≤ n ∧ n ≤ 13 ∧ ∃ (m : ℕ), n^2 + n + 1 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l1013_101374


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1013_101305

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * l + 2 * w = 2 * Real.pi * r) : 
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1013_101305


namespace NUMINAMATH_CALUDE_sum_of_absolute_roots_l1013_101392

theorem sum_of_absolute_roots (m : ℤ) (a b c d : ℤ) : 
  (∀ x : ℤ, x^4 - x^3 - 4023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) →
  |a| + |b| + |c| + |d| = 621 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_roots_l1013_101392


namespace NUMINAMATH_CALUDE_equilateral_triangle_vertices_product_l1013_101355

theorem equilateral_triangle_vertices_product (a b : ℝ) : 
  (∀ z : ℂ, z^3 = 1 ∧ z ≠ 1 → (a + 18 * I) * z = b + 42 * I) →
  a * b = -2652 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_vertices_product_l1013_101355


namespace NUMINAMATH_CALUDE_root_product_theorem_l1013_101329

theorem root_product_theorem (x₁ x₂ x₃ : ℝ) : 
  x₃ < x₂ ∧ x₂ < x₁ →
  (Real.sqrt 120 * x₁^3 - 480 * x₁^2 + 8 * x₁ + 1 = 0) →
  (Real.sqrt 120 * x₂^3 - 480 * x₂^2 + 8 * x₂ + 1 = 0) →
  (Real.sqrt 120 * x₃^3 - 480 * x₃^2 + 8 * x₃ + 1 = 0) →
  x₂ * (x₁ + x₃) = -1/120 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1013_101329


namespace NUMINAMATH_CALUDE_prop_equivalence_l1013_101332

theorem prop_equivalence (p q : Prop) : (p ∧ q) ↔ ¬(¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_prop_equivalence_l1013_101332


namespace NUMINAMATH_CALUDE_triangle_base_length_l1013_101346

theorem triangle_base_length (square_side : ℝ) (triangle_altitude : ℝ) 
  (h1 : square_side = 6)
  (h2 : triangle_altitude = 12)
  (h3 : (square_side * square_side) = (triangle_altitude * triangle_base) / 2) :
  triangle_base = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l1013_101346


namespace NUMINAMATH_CALUDE_stone_transport_impossible_l1013_101394

/-- The number of stone blocks -/
def n : ℕ := 50

/-- The weight of the first stone block in kg -/
def first_weight : ℕ := 370

/-- The weight increase for each subsequent block in kg -/
def weight_increase : ℕ := 2

/-- The number of available trucks -/
def num_trucks : ℕ := 7

/-- The capacity of each truck in kg -/
def truck_capacity : ℕ := 3000

/-- The total weight of n stone blocks -/
def total_weight (n : ℕ) : ℕ :=
  n * first_weight + (n * (n - 1) / 2) * weight_increase

/-- The total capacity of all trucks -/
def total_capacity : ℕ := num_trucks * truck_capacity

theorem stone_transport_impossible : total_weight n > total_capacity := by
  sorry

end NUMINAMATH_CALUDE_stone_transport_impossible_l1013_101394


namespace NUMINAMATH_CALUDE_poached_percentage_less_than_sold_l1013_101316

def total_pears : ℕ := 42
def sold_pears : ℕ := 20

def canned_pears (poached : ℕ) : ℕ := poached + poached / 5

theorem poached_percentage_less_than_sold :
  ∃ (poached : ℕ),
    poached > 0 ∧
    poached < sold_pears ∧
    total_pears = sold_pears + canned_pears poached + poached ∧
    (sold_pears - poached) * 100 / sold_pears = 50 := by
  sorry

end NUMINAMATH_CALUDE_poached_percentage_less_than_sold_l1013_101316


namespace NUMINAMATH_CALUDE_max_value_of_f_l1013_101391

def f (x a : ℝ) : ℝ := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1013_101391
