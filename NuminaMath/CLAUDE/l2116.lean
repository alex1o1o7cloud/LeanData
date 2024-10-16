import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l2116_211632

theorem min_sum_with_constraint (x y z : ℝ) (h : (4 / x) + (2 / y) + (1 / z) = 1) :
  x + 8 * y + 4 * z ≥ 64 ∧ ∃ (x₀ y₀ z₀ : ℝ), (4 / x₀) + (2 / y₀) + (1 / z₀) = 1 ∧ x₀ + 8 * y₀ + 4 * z₀ = 64 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l2116_211632


namespace NUMINAMATH_CALUDE_trader_gain_l2116_211682

theorem trader_gain (cost selling_price : ℝ) (h1 : selling_price = 1.25 * cost) : 
  (80 * selling_price - 80 * cost) / cost = 20 := by
  sorry

#check trader_gain

end NUMINAMATH_CALUDE_trader_gain_l2116_211682


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2116_211622

/-- Given that x and y are inversely proportional, prove that if x = 3y when x + y = 60, then y = 45 when x = 15. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ x₀ y₀ : ℝ, x₀ = 3 * y₀ ∧ x₀ + y₀ = 60 ∧ x₀ * y₀ = k) :
  x = 15 → y = 45 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2116_211622


namespace NUMINAMATH_CALUDE_class_average_mark_l2116_211646

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_avg : ℝ) (remaining_avg : ℝ) :
  total_students = 25 →
  excluded_students = 5 →
  excluded_avg = 20 →
  remaining_avg = 95 →
  (total_students * (total_students * remaining_avg - excluded_students * remaining_avg + excluded_students * excluded_avg)) / (total_students * total_students) = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l2116_211646


namespace NUMINAMATH_CALUDE_cubic_equation_one_positive_root_l2116_211605

theorem cubic_equation_one_positive_root (a b : ℝ) (hb : b > 0) :
  ∃! x : ℝ, x > 0 ∧ x^3 + a*x^2 - b = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_one_positive_root_l2116_211605


namespace NUMINAMATH_CALUDE_problem_solution_l2116_211665

/-- Proposition A: The solution set of x^2 + (a-1)x + a^2 ≤ 0 with respect to x is empty -/
def proposition_a (a : ℝ) : Prop :=
  ∀ x, x^2 + (a-1)*x + a^2 > 0

/-- Proposition B: The function y = (2a^2 - a)^x is increasing -/
def proposition_b (a : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → (2*a^2 - a)^x₁ < (2*a^2 - a)^x₂

/-- The set of real numbers a for which at least one of A or B is true -/
def at_least_one_true (a : ℝ) : Prop :=
  proposition_a a ∨ proposition_b a

/-- The set of real numbers a for which exactly one of A or B is true -/
def exactly_one_true (a : ℝ) : Prop :=
  (proposition_a a ∧ ¬proposition_b a) ∨ (¬proposition_a a ∧ proposition_b a)

theorem problem_solution :
  (∀ a, at_least_one_true a ↔ (a < -1/2 ∨ a > 1/3)) ∧
  (∀ a, exactly_one_true a ↔ (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2116_211665


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2116_211668

/-- Represents a systematic sample from a population --/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Checks if a number is in the systematic sample --/
def inSample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

/-- The theorem to be proved --/
theorem systematic_sample_fourth_element :
  ∀ s : SystematicSample,
    s.population_size = 48 →
    s.sample_size = 4 →
    inSample s 5 →
    inSample s 29 →
    inSample s 41 →
    inSample s 17 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2116_211668


namespace NUMINAMATH_CALUDE_triangle_theorem_l2116_211645

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  cosine_law_a : a^2 = b^2 + c^2 - 2*b*c*Real.cos A
  cosine_law_b : b^2 = a^2 + c^2 - 2*a*c*Real.cos B
  cosine_law_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) (h : 2*t.a*Real.cos t.C = 2*t.b - t.c) :
  /- Part 1 -/
  t.A = π/3 ∧
  /- Part 2 -/
  (t.A < π/2 ∧ t.B < π/2 ∧ t.C < π/2 → 
    3/2 < Real.sin t.B + Real.sin t.C ∧ Real.sin t.B + Real.sin t.C ≤ Real.sqrt 3) ∧
  /- Part 3 -/
  (t.a = 2*Real.sqrt 3 ∧ 1/2*t.b*t.c*Real.sin t.A = 2*Real.sqrt 3 →
    Real.cos (2*t.B) + Real.cos (2*t.C) = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2116_211645


namespace NUMINAMATH_CALUDE_nine_rings_puzzle_5_l2116_211614

def nine_rings_puzzle (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Define for 0 to satisfy recursion
  | 1 => 1
  | n + 1 =>
    if n % 2 = 0 then
      2 * nine_rings_puzzle n + 2
    else
      2 * nine_rings_puzzle n - 1

theorem nine_rings_puzzle_5 :
  nine_rings_puzzle 5 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_nine_rings_puzzle_5_l2116_211614


namespace NUMINAMATH_CALUDE_probability_point_not_in_square_l2116_211613

/-- Given a rectangle A and a square B, where B is placed within A, 
    calculate the probability that a random point in A is not in B. -/
theorem probability_point_not_in_square (area_A perimeter_B : ℝ) : 
  area_A = 30 →
  perimeter_B = 16 →
  (area_A - (perimeter_B / 4)^2) / area_A = 7/15 :=
by sorry

end NUMINAMATH_CALUDE_probability_point_not_in_square_l2116_211613


namespace NUMINAMATH_CALUDE_expression_value_l2116_211696

theorem expression_value (a b : ℝ) (h : a * 1 + b * 2 = 3) : 2 * a + 4 * b - 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2116_211696


namespace NUMINAMATH_CALUDE_broken_bowls_l2116_211670

theorem broken_bowls (total_bowls : ℕ) (lost_bowls : ℕ) (fee : ℕ) (safe_payment : ℕ) (penalty : ℕ) (total_payment : ℕ) :
  total_bowls = 638 →
  lost_bowls = 12 →
  fee = 100 →
  safe_payment = 3 →
  penalty = 4 →
  total_payment = 1825 →
  ∃ (broken_bowls : ℕ),
    fee + safe_payment * (total_bowls - lost_bowls - broken_bowls) - 
    (penalty * lost_bowls + penalty * broken_bowls) = total_payment ∧
    broken_bowls = 29 :=
sorry

end NUMINAMATH_CALUDE_broken_bowls_l2116_211670


namespace NUMINAMATH_CALUDE_truncated_pyramid_volume_division_l2116_211679

/-- Represents a truncated triangular pyramid -/
structure TruncatedPyramid where
  upperBaseArea : ℝ
  lowerBaseArea : ℝ
  height : ℝ
  baseRatio : upperBaseArea / lowerBaseArea = 1 / 4

/-- Represents the volumes of the two parts created by the plane -/
structure DividedVolumes where
  v1 : ℝ
  v2 : ℝ

/-- 
  Given a truncated triangular pyramid where the corresponding sides of the upper and lower 
  bases are in the ratio 1:2, if a plane is drawn through a side of the upper base parallel 
  to the opposite lateral edge, it divides the volume of the truncated pyramid in the ratio 3:4.
-/
theorem truncated_pyramid_volume_division (p : TruncatedPyramid) : 
  ∃ (v : DividedVolumes), v.v1 / v.v2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_truncated_pyramid_volume_division_l2116_211679


namespace NUMINAMATH_CALUDE_units_produced_today_l2116_211691

theorem units_produced_today (past_average : ℝ) (new_average : ℝ) (past_days : ℕ) :
  past_average = 40 →
  new_average = 45 →
  past_days = 9 →
  (past_days + 1) * new_average - past_days * past_average = 90 := by
  sorry

end NUMINAMATH_CALUDE_units_produced_today_l2116_211691


namespace NUMINAMATH_CALUDE_root_equation_implies_d_equals_eight_l2116_211692

theorem root_equation_implies_d_equals_eight 
  (a b c d : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) (hd : d > 1) 
  (h : ∀ (M : ℝ), M ≠ 1 → (M^(1/a + 1/(a*b) + 1/(a*b*c) + 1/(a*b*c*d))) = M^(17/24)) : 
  d = 8 := by
sorry

end NUMINAMATH_CALUDE_root_equation_implies_d_equals_eight_l2116_211692


namespace NUMINAMATH_CALUDE_cubic_expression_value_l2116_211698

theorem cubic_expression_value (α : ℝ) (h1 : α > 0) (h2 : α^2 - 8*α - 5 = 0) :
  α^3 - 7*α^2 - 13*α + 6 = 11 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l2116_211698


namespace NUMINAMATH_CALUDE_gcd_of_100_and_250_l2116_211651

theorem gcd_of_100_and_250 : Nat.gcd 100 250 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_100_and_250_l2116_211651


namespace NUMINAMATH_CALUDE_seven_patients_three_doctors_l2116_211688

/-- The number of ways to assign n distinct objects to k distinct categories,
    where each object is assigned to exactly one category and
    each category receives at least one object. -/
def assignments (n k : ℕ) : ℕ :=
  k^n - (k * (k-1)^n - k * (k-1) * (k-2)^n)

/-- There are 7 patients and 3 doctors -/
theorem seven_patients_three_doctors :
  assignments 7 3 = 1806 := by
  sorry

end NUMINAMATH_CALUDE_seven_patients_three_doctors_l2116_211688


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2116_211660

theorem imaginary_part_of_complex_expression :
  let z : ℂ := (1 + I) / (1 - I) + (1 - I)^2
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2116_211660


namespace NUMINAMATH_CALUDE_track_circumference_jogging_track_circumference_l2116_211617

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (speed1 speed2 : ℝ) (meeting_time : ℝ) (h1 : speed1 = 4.2)
    (h2 : speed2 = 3.8) (h3 : meeting_time = 4.8 / 60) : ℝ :=
  let distance1 := speed1 * meeting_time
  let distance2 := speed2 * meeting_time
  let total_distance := distance1 + distance2
  total_distance

/-- The circumference of the jogging track is 0.63984 km -/
theorem jogging_track_circumference :
    track_circumference 4.2 3.8 (4.8 / 60) rfl rfl rfl = 0.63984 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_jogging_track_circumference_l2116_211617


namespace NUMINAMATH_CALUDE_fruit_basket_theorem_l2116_211663

/-- Calculates the number of possible fruit baskets given a number of apples and oranges. -/
def fruitBasketCount (apples : ℕ) (oranges : ℕ) : ℕ :=
  (apples + 1) * (oranges + 1) - 1

/-- Theorem stating that the number of fruit baskets with 4 apples and 8 oranges is 44. -/
theorem fruit_basket_theorem :
  fruitBasketCount 4 8 = 44 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_theorem_l2116_211663


namespace NUMINAMATH_CALUDE_distance_center_to_point_l2116_211601

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 10*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ :=
  let a := 3
  let b := 5
  (a, b)

-- Define the given point
def given_point : ℝ × ℝ := (-4, -2)

-- Theorem statement
theorem distance_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = 7 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_point_l2116_211601


namespace NUMINAMATH_CALUDE_final_black_goats_count_l2116_211648

theorem final_black_goats_count (total : ℕ) (initial_black : ℕ) (new_black : ℕ) :
  total = 93 →
  initial_black = 66 →
  new_black = 21 →
  initial_black ≤ total →
  let initial_white := total - initial_black
  let new_total_black := initial_black + new_black
  let deaths := min initial_white new_total_black
  new_total_black - deaths = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_final_black_goats_count_l2116_211648


namespace NUMINAMATH_CALUDE_ian_hourly_rate_l2116_211672

/-- Represents Ian's survey work and earnings -/
structure SurveyWork where
  hours_worked : ℕ
  money_left : ℕ
  spend_ratio : ℚ

/-- Calculates Ian's hourly rate given his survey work details -/
def hourly_rate (work : SurveyWork) : ℚ :=
  (work.money_left / (1 - work.spend_ratio)) / work.hours_worked

/-- Theorem stating that Ian's hourly rate is $18 -/
theorem ian_hourly_rate :
  let work : SurveyWork := {
    hours_worked := 8,
    money_left := 72,
    spend_ratio := 1/2
  }
  hourly_rate work = 18 := by sorry

end NUMINAMATH_CALUDE_ian_hourly_rate_l2116_211672


namespace NUMINAMATH_CALUDE_square_side_length_l2116_211678

theorem square_side_length : 
  ∃ (x : ℝ), x > 0 ∧ 4 * x = 2 * (x ^ 2) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2116_211678


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2116_211609

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
    a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + a₉*(x+2)^9 + 
    a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2116_211609


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2116_211628

theorem sin_cos_identity (x : ℝ) (h : 3 * Real.sin x + Real.cos x = 0) :
  Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + Real.cos x ^ 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2116_211628


namespace NUMINAMATH_CALUDE_unique_modular_solution_l2116_211633

theorem unique_modular_solution : ∃! n : ℕ, n < 251 ∧ (250 * n) % 251 = 123 % 251 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l2116_211633


namespace NUMINAMATH_CALUDE_infections_exceed_threshold_l2116_211690

/-- The number of people infected after two rounds of infection -/
def infected_after_two_rounds : ℕ := 81

/-- The average number of people infected by one person in each round -/
def average_infections_per_round : ℕ := 8

/-- The threshold number of infections we want to exceed after three rounds -/
def infection_threshold : ℕ := 700

/-- Theorem stating that the number of infected people after three rounds exceeds the threshold -/
theorem infections_exceed_threshold : 
  infected_after_two_rounds * (1 + average_infections_per_round) > infection_threshold := by
  sorry


end NUMINAMATH_CALUDE_infections_exceed_threshold_l2116_211690


namespace NUMINAMATH_CALUDE_master_craftsman_production_l2116_211600

/-- The number of parts manufactured by a master craftsman during a shift -/
def total_parts : ℕ := 210

/-- The number of parts manufactured in the first hour -/
def first_hour_parts : ℕ := 35

/-- The increase in production rate (parts per hour) -/
def rate_increase : ℕ := 15

/-- The time saved by increasing the production rate (in hours) -/
def time_saved : ℚ := 1.5

theorem master_craftsman_production :
  ∃ (N : ℕ),
    (N : ℚ) / first_hour_parts - (N : ℚ) / (first_hour_parts + rate_increase) = time_saved ∧
    total_parts = first_hour_parts + N :=
  sorry

end NUMINAMATH_CALUDE_master_craftsman_production_l2116_211600


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2116_211652

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2116_211652


namespace NUMINAMATH_CALUDE_quadratic_intersection_l2116_211627

/-- A quadratic function of the form y = x^2 + px + q where p + q = 2002 -/
def QuadraticFunction (p q : ℝ) : ℝ → ℝ := fun x ↦ x^2 + p*x + q

/-- The theorem stating that all quadratic functions satisfying the condition
    p + q = 2002 intersect at the point (1, 2003) -/
theorem quadratic_intersection (p q : ℝ) (h : p + q = 2002) :
  QuadraticFunction p q 1 = 2003 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l2116_211627


namespace NUMINAMATH_CALUDE_unique_p_q_for_f_bounded_l2116_211611

def f (p q x : ℝ) := x^2 + p*x + q

theorem unique_p_q_for_f_bounded :
  ∃! p q : ℝ, ∀ x ∈ Set.Icc 1 5, |f p q x| ≤ 2 := by sorry

end NUMINAMATH_CALUDE_unique_p_q_for_f_bounded_l2116_211611


namespace NUMINAMATH_CALUDE_polygon_count_l2116_211667

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The number of distinct convex polygons with 3 or more sides 
    that can be drawn using some or all of n points marked on a circle as vertices -/
def num_polygons (n : ℕ) : ℕ := 2^n - (n.choose 0 + n.choose 1 + n.choose 2)

theorem polygon_count : num_polygons n = 4017 := by
  sorry

end NUMINAMATH_CALUDE_polygon_count_l2116_211667


namespace NUMINAMATH_CALUDE_prob_different_colors_specific_l2116_211634

/-- The probability of drawing two chips of different colors -/
def prob_different_colors (blue yellow red : ℕ) : ℚ :=
  let total := blue + yellow + red
  let p_blue := blue / total
  let p_yellow := yellow / total
  let p_red := red / total
  p_blue * (p_yellow + p_red) + p_yellow * (p_blue + p_red) + p_red * (p_blue + p_yellow)

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_specific : prob_different_colors 6 4 2 = 11 / 18 := by
  sorry

#eval prob_different_colors 6 4 2

end NUMINAMATH_CALUDE_prob_different_colors_specific_l2116_211634


namespace NUMINAMATH_CALUDE_longest_crafting_pattern_length_l2116_211680

/-- Represents the lengths of ribbons in inches -/
structure RibbonLengths where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ
  purple : ℕ

/-- Calculates the remaining lengths of ribbons -/
def remainingLengths (initial used : RibbonLengths) : RibbonLengths :=
  { red := initial.red - used.red,
    blue := initial.blue - used.blue,
    green := initial.green - used.green,
    yellow := initial.yellow - used.yellow,
    purple := initial.purple - used.purple }

/-- Finds the minimum length among all ribbon colors -/
def minLength (lengths : RibbonLengths) : ℕ :=
  min lengths.red (min lengths.blue (min lengths.green (min lengths.yellow lengths.purple)))

/-- The initial lengths of ribbons -/
def initialLengths : RibbonLengths :=
  { red := 84, blue := 96, green := 112, yellow := 54, purple := 120 }

/-- The used lengths of ribbons -/
def usedLengths : RibbonLengths :=
  { red := 46, blue := 58, green := 72, yellow := 30, purple := 90 }

theorem longest_crafting_pattern_length :
  minLength (remainingLengths initialLengths usedLengths) = 24 := by
  sorry

#eval minLength (remainingLengths initialLengths usedLengths)

end NUMINAMATH_CALUDE_longest_crafting_pattern_length_l2116_211680


namespace NUMINAMATH_CALUDE_power_multiplication_correct_equation_l2116_211610

theorem power_multiplication (a b : ℕ) : 2^a * 2^b = 2^(a + b) := by sorry

theorem correct_equation : 2^2 * 2^3 = 2^5 := by
  apply power_multiplication

end NUMINAMATH_CALUDE_power_multiplication_correct_equation_l2116_211610


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2116_211656

-- Define the set of people
inductive Person : Type
  | A
  | B
  | C

-- Define the set of cards
inductive Card : Type
  | Red
  | Yellow
  | Blue

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person B gets the red card"
def event_B (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A d ∧ event_B d)) ∧
  (∃ d : Distribution, ¬event_A d ∧ ¬event_B d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2116_211656


namespace NUMINAMATH_CALUDE_coffee_blend_cost_l2116_211699

/-- The cost of the first blend of coffee in dollars per pound -/
def first_blend_cost : ℝ := 9

/-- The cost of the second blend of coffee in dollars per pound -/
def second_blend_cost : ℝ := 8

/-- The total weight of the mixed blend in pounds -/
def total_blend_weight : ℝ := 20

/-- The selling price of the mixed blend in dollars per pound -/
def mixed_blend_price : ℝ := 8.4

/-- The weight of the first blend used in the mixture in pounds -/
def first_blend_weight : ℝ := 8

theorem coffee_blend_cost : 
  first_blend_cost * first_blend_weight + 
  second_blend_cost * (total_blend_weight - first_blend_weight) = 
  mixed_blend_price * total_blend_weight := by
  sorry

#check coffee_blend_cost

end NUMINAMATH_CALUDE_coffee_blend_cost_l2116_211699


namespace NUMINAMATH_CALUDE_investor_profit_l2116_211697

def total_investment : ℝ := 1900
def investment_fund1 : ℝ := 1700
def profit_rate_fund1 : ℝ := 0.09
def profit_rate_fund2 : ℝ := 0.02

def investment_fund2 : ℝ := total_investment - investment_fund1

def profit_fund1 : ℝ := investment_fund1 * profit_rate_fund1
def profit_fund2 : ℝ := investment_fund2 * profit_rate_fund2

def total_profit : ℝ := profit_fund1 + profit_fund2

theorem investor_profit : total_profit = 157 := by
  sorry

end NUMINAMATH_CALUDE_investor_profit_l2116_211697


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2116_211685

/-- The volume of a rectangular prism with face areas √2, √3, and √6 is √6 -/
theorem rectangular_prism_volume (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : a * c = Real.sqrt 6) :
  a * b * c = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2116_211685


namespace NUMINAMATH_CALUDE_well_depth_l2116_211639

/-- Represents a circular well -/
structure CircularWell where
  diameter : ℝ
  volume : ℝ
  depth : ℝ

/-- Theorem stating the depth of a specific circular well -/
theorem well_depth (w : CircularWell) 
  (h1 : w.diameter = 4)
  (h2 : w.volume = 175.92918860102841) :
  w.depth = 14 := by
  sorry

end NUMINAMATH_CALUDE_well_depth_l2116_211639


namespace NUMINAMATH_CALUDE_fishing_line_sections_l2116_211673

/-- The number of reels of fishing line John buys -/
def num_reels : ℕ := 3

/-- The length of fishing line in each reel (in meters) -/
def reel_length : ℕ := 100

/-- The length of each section John cuts the fishing line into (in meters) -/
def section_length : ℕ := 10

/-- The total number of sections John gets from cutting all the fishing line -/
def total_sections : ℕ := (num_reels * reel_length) / section_length

theorem fishing_line_sections :
  total_sections = 30 := by sorry

end NUMINAMATH_CALUDE_fishing_line_sections_l2116_211673


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2116_211636

universe u

def U : Set Nat := {0, 1, 2, 3}
def A : Set Nat := {0, 1}
def B : Set Nat := {1, 2, 3}

theorem complement_intersection_theorem :
  (Set.compl A ∩ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2116_211636


namespace NUMINAMATH_CALUDE_circular_rug_middle_ring_area_l2116_211629

theorem circular_rug_middle_ring_area :
  ∀ (inner_radius middle_radius outer_radius : ℝ)
    (inner_area middle_area outer_area : ℝ),
  inner_radius = 1 →
  middle_radius = inner_radius + 1 →
  outer_radius = middle_radius + 1 →
  inner_area = π * inner_radius^2 →
  middle_area = π * middle_radius^2 →
  outer_area = π * outer_radius^2 →
  middle_area - inner_area = 3 * π := by
sorry

end NUMINAMATH_CALUDE_circular_rug_middle_ring_area_l2116_211629


namespace NUMINAMATH_CALUDE_origin_on_circle_circle_through_P_l2116_211683

-- Define the parabola C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 2 * p.1}

-- Define the point (2, 0)
def point_2_0 : ℝ × ℝ := (2, 0)

-- Define the line l passing through (2, 0)
def l (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * (p.1 - 2)}

-- Define the intersection points A and B
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

-- Define the circle M with diameter AB
def M (k : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define point P
def P : ℝ × ℝ := (4, -2)

-- Theorem 1: The origin O is on circle M
theorem origin_on_circle (k : ℝ) : O ∈ M k := sorry

-- Theorem 2: If M passes through P, then l and M have specific equations
theorem circle_through_P (k : ℝ) (h : P ∈ M k) :
  (k = -2 ∧ l k = {p : ℝ × ℝ | p.2 = -2 * p.1 + 4} ∧ 
   M k = {p : ℝ × ℝ | (p.1 - 9/4)^2 + (p.2 + 1/2)^2 = 85/16}) ∨
  (k = 1 ∧ l k = {p : ℝ × ℝ | p.2 = p.1 - 2} ∧ 
   M k = {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 10}) := sorry

end NUMINAMATH_CALUDE_origin_on_circle_circle_through_P_l2116_211683


namespace NUMINAMATH_CALUDE_problem_proof_l2116_211626

theorem problem_proof : -1^2023 + (Real.pi - 3.14)^0 + |-2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2116_211626


namespace NUMINAMATH_CALUDE_turtleneck_profit_percentage_l2116_211603

/-- Calculates the profit percentage on turtleneck sweaters sold in February 
    given specific markup and discount conditions. -/
theorem turtleneck_profit_percentage :
  let initial_markup : ℝ := 0.20
  let new_year_markup : ℝ := 0.25
  let february_discount : ℝ := 0.09
  let first_price := 1 + initial_markup
  let second_price := first_price + new_year_markup * first_price
  let final_price := second_price * (1 - february_discount)
  let profit_percentage := final_price - 1
  profit_percentage = 0.365 := by sorry

end NUMINAMATH_CALUDE_turtleneck_profit_percentage_l2116_211603


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_less_than_150_l2116_211616

theorem sum_of_fourth_powers_less_than_150 : 
  (Finset.filter (fun n : ℕ => n^4 < 150) (Finset.range 150)).sum id = 98 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_less_than_150_l2116_211616


namespace NUMINAMATH_CALUDE_silas_payment_ratio_l2116_211602

theorem silas_payment_ratio (total_bill : ℚ) (friend_payment : ℚ) 
  (h1 : total_bill = 150)
  (h2 : friend_payment = 18)
  (h3 : (5 : ℚ) * friend_payment + (total_bill / 10) = total_bill + (total_bill / 10) - (total_bill / 2)) :
  (total_bill / 2) / total_bill = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_silas_payment_ratio_l2116_211602


namespace NUMINAMATH_CALUDE_f_properties_l2116_211676

noncomputable def f (x : ℝ) : ℝ := Real.log (x / (x^2 + 1))

theorem f_properties :
  (∀ x : ℝ, x > 0 → f x ≠ 0) ∧
  (∀ x : ℝ, 0 < x → x < 1 → ∀ y : ℝ, x < y → y < 1 → f x < f y) ∧
  (∀ x : ℝ, x > 1 → ∀ y : ℝ, y > x → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2116_211676


namespace NUMINAMATH_CALUDE_distance_after_three_minutes_l2116_211625

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

/-- Theorem: The distance between two vehicles with speeds 65 km/h and 85 km/h after 3 minutes is 1 km -/
theorem distance_after_three_minutes :
  let v1 : ℝ := 65  -- Speed of the truck in km/h
  let v2 : ℝ := 85  -- Speed of the car in km/h
  let t : ℝ := 3 / 60  -- 3 minutes converted to hours
  distance_between_vehicles v1 v2 t = 1 := by
  sorry


end NUMINAMATH_CALUDE_distance_after_three_minutes_l2116_211625


namespace NUMINAMATH_CALUDE_first_divisor_problem_l2116_211662

theorem first_divisor_problem (x : ℚ) : 
  (((377 / x) / 29) * (1/4)) / 2 = 0.125 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l2116_211662


namespace NUMINAMATH_CALUDE_inequality_proof_l2116_211661

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) + 
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) + 
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2116_211661


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_ratio_l2116_211615

/-- 
Given three distinct real numbers a, b, c forming an arithmetic sequence with a < b < c,
if swapping two of these numbers results in a geometric sequence,
then (a² + c²) / b² = 20.
-/
theorem arithmetic_to_geometric_sequence_ratio (a b c : ℝ) : 
  a < b → b < c → 
  (∃ d : ℝ, c - b = b - a ∧ d = b - a) →
  (∃ (x y z : ℝ) (σ : Equiv.Perm (Fin 3)), 
    ({x, y, z} : Finset ℝ) = {a, b, c} ∧ 
    (y * y = x * z)) →
  (a * a + c * c) / (b * b) = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_ratio_l2116_211615


namespace NUMINAMATH_CALUDE_negation_of_exists_square_nonpositive_l2116_211607

theorem negation_of_exists_square_nonpositive :
  (¬ ∃ x : ℝ, x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_square_nonpositive_l2116_211607


namespace NUMINAMATH_CALUDE_complement_of_A_range_of_c_l2116_211624

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x - 6 ≥ 0}

-- Define set B
def B (c : ℝ) : Set ℝ := {x : ℝ | x > c}

-- Theorem for the complement of A
theorem complement_of_A : 
  {x : ℝ | x ∉ A} = {x : ℝ | -2 < x ∧ x < 3} := by sorry

-- Theorem for the range of c
theorem range_of_c :
  (∀ x : ℝ, x ∈ A ∨ x ∈ B c) → c ∈ Set.Iic (-2) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_range_of_c_l2116_211624


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2116_211658

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2116_211658


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_six_l2116_211649

theorem sum_of_roots_eq_six : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_six_l2116_211649


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2116_211671

/-- In a right triangle LMN, given cos M and the length of LM, we can determine the length of LN. -/
theorem right_triangle_side_length 
  (L M N : ℝ × ℝ) 
  (right_angle_M : (N.1 - M.1) * (L.2 - M.2) = (L.1 - M.1) * (N.2 - M.2)) 
  (cos_M : Real.cos (Real.arctan ((L.2 - M.2) / (L.1 - M.1))) = 3/5) 
  (LM_length : Real.sqrt ((L.1 - M.1)^2 + (L.2 - M.2)^2) = 15) :
  Real.sqrt ((L.1 - N.1)^2 + (L.2 - N.2)^2) = 9 := by
    sorry


end NUMINAMATH_CALUDE_right_triangle_side_length_l2116_211671


namespace NUMINAMATH_CALUDE_total_pushups_is_53_l2116_211641

/-- The number of push-ups David did -/
def david_pushups : ℕ := 51

/-- The difference between David's and Zachary's push-ups -/
def pushup_difference : ℕ := 49

/-- Calculates the total number of push-ups done by David and Zachary -/
def total_pushups : ℕ := david_pushups + (david_pushups - pushup_difference)

theorem total_pushups_is_53 : total_pushups = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_pushups_is_53_l2116_211641


namespace NUMINAMATH_CALUDE_part1_part2_l2116_211681

-- Define the quadratic inequality
def quadratic_inequality (a b x : ℝ) : Prop := a * x^2 + b * x - 1 ≥ 0

-- Part 1
theorem part1 (a b : ℝ) :
  (∀ x, quadratic_inequality a b x ↔ (3 ≤ x ∧ x ≤ 4)) →
  a + b = 1/2 := by sorry

-- Part 2
theorem part2 (b : ℝ) :
  let solution_set := {x : ℝ | quadratic_inequality (-1) b x}
  if -2 < b ∧ b < 2 then
    solution_set = ∅
  else if b = -2 then
    solution_set = {-1}
  else if b = 2 then
    solution_set = {1}
  else
    ∃ (l u : ℝ), l = (b - Real.sqrt (b^2 - 4)) / 2 ∧
                 u = (b + Real.sqrt (b^2 - 4)) / 2 ∧
                 solution_set = {x | l ≤ x ∧ x ≤ u} := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2116_211681


namespace NUMINAMATH_CALUDE_seven_non_drinkers_l2116_211631

/-- Represents the number of businessmen who drank a specific beverage or combination of beverages -/
structure BeverageCounts where
  total : Nat
  coffee : Nat
  tea : Nat
  water : Nat
  coffeeAndTea : Nat
  teaAndWater : Nat
  coffeeAndWater : Nat
  allThree : Nat

/-- Calculates the number of businessmen who drank none of the beverages -/
def nonDrinkers (counts : BeverageCounts) : Nat :=
  counts.total - (counts.coffee + counts.tea + counts.water
                  - counts.coffeeAndTea - counts.teaAndWater - counts.coffeeAndWater
                  + counts.allThree)

/-- Theorem stating that given the conditions, 7 businessmen drank none of the beverages -/
theorem seven_non_drinkers (counts : BeverageCounts)
  (h1 : counts.total = 30)
  (h2 : counts.coffee = 15)
  (h3 : counts.tea = 13)
  (h4 : counts.water = 6)
  (h5 : counts.coffeeAndTea = 7)
  (h6 : counts.teaAndWater = 3)
  (h7 : counts.coffeeAndWater = 2)
  (h8 : counts.allThree = 1) :
  nonDrinkers counts = 7 := by
  sorry

#eval nonDrinkers { total := 30, coffee := 15, tea := 13, water := 6,
                    coffeeAndTea := 7, teaAndWater := 3, coffeeAndWater := 2, allThree := 1 }

end NUMINAMATH_CALUDE_seven_non_drinkers_l2116_211631


namespace NUMINAMATH_CALUDE_dog_catches_fox_l2116_211642

/-- The speed of the dog in meters per second -/
def dog_speed : ℝ := 2

/-- The time the dog runs in each unit of time, in seconds -/
def dog_time : ℝ := 2

/-- The speed of the fox in meters per second -/
def fox_speed : ℝ := 3

/-- The time the fox runs in each unit of time, in seconds -/
def fox_time : ℝ := 1

/-- The initial distance between the dog and the fox in meters -/
def initial_distance : ℝ := 30

/-- The total distance the dog runs before catching the fox -/
def total_distance : ℝ := 120

theorem dog_catches_fox : 
  let dog_distance_per_unit := dog_speed * dog_time
  let fox_distance_per_unit := fox_speed * fox_time
  let distance_gained_per_unit := dog_distance_per_unit - fox_distance_per_unit
  let units_to_catch := initial_distance / distance_gained_per_unit
  dog_distance_per_unit * units_to_catch = total_distance := by
sorry

end NUMINAMATH_CALUDE_dog_catches_fox_l2116_211642


namespace NUMINAMATH_CALUDE_volume_S_form_prism_ratio_l2116_211618

/-- A right rectangular prism with given edge lengths -/
structure RectPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The set of points within distance r of a point in the prism -/
def S (B : RectPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume_S (B : RectPrism) (r : ℝ) : ℝ := sorry

theorem volume_S_form (B : RectPrism) :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  ∀ r : ℝ, r ≥ 0 → volume_S B r = a * r^3 + b * r^2 + c * r + d :=
sorry

theorem prism_ratio (B : RectPrism) (a b c d : ℝ) :
  B.length = 2 ∧ B.width = 3 ∧ B.height = 5 →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  (∀ r : ℝ, r ≥ 0 → volume_S B r = a * r^3 + b * r^2 + c * r + d) →
  b * c / (a * d) = 15.5 :=
sorry

end NUMINAMATH_CALUDE_volume_S_form_prism_ratio_l2116_211618


namespace NUMINAMATH_CALUDE_kopeck_ruble_equivalence_l2116_211664

/-- Represents the denominations of coins available in kopecks -/
def coin_denominations : List ℕ := [1, 2, 5, 10, 20, 50, 100]

/-- Represents a collection of coins, where each natural number is the count of coins for the corresponding denomination -/
def Coins := List ℕ

/-- Calculates the total value of a collection of coins in kopecks -/
def total_value (coins : Coins) : ℕ :=
  List.sum (List.zipWith (· * ·) coins coin_denominations)

/-- Calculates the total number of coins in a collection -/
def total_count (coins : Coins) : ℕ :=
  List.sum coins

theorem kopeck_ruble_equivalence (k m : ℕ) (coins : Coins) 
    (h1 : total_count coins = k)
    (h2 : total_value coins = m) :
  ∃ (new_coins : Coins), total_count new_coins = m ∧ total_value new_coins = k * 100 := by
  sorry

end NUMINAMATH_CALUDE_kopeck_ruble_equivalence_l2116_211664


namespace NUMINAMATH_CALUDE_triangle_inequality_l2116_211623

theorem triangle_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum : a + b + c = 1) : 
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2116_211623


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l2116_211689

theorem floor_plus_self_eq_seventeen_fourths :
  ∃! (y : ℚ), ⌊y⌋ + y = 17 / 4 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l2116_211689


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2116_211637

/-- An odd function from ℝ to ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem sufficient_not_necessary_condition
  (f : ℝ → ℝ) (hf : OddFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧
  (∃ x₁ x₂ : ℝ, f x₁ + f x₂ = 0 ∧ x₁ + x₂ ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2116_211637


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l2116_211619

-- Define a complex number
def complex (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem a_zero_necessary_not_sufficient :
  (∀ z : ℂ, is_purely_imaginary z → z.re = 0) ∧
  ¬(∀ z : ℂ, z.re = 0 → is_purely_imaginary z) :=
by sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l2116_211619


namespace NUMINAMATH_CALUDE_alcohol_dilution_l2116_211657

theorem alcohol_dilution (original_volume : ℝ) (original_percentage : ℝ) 
  (added_water : ℝ) (new_percentage : ℝ) :
  original_volume = 15 →
  original_percentage = 0.2 →
  added_water = 3 →
  new_percentage = 1/6 →
  (original_volume * original_percentage) / (original_volume + added_water) = new_percentage := by
  sorry

#check alcohol_dilution

end NUMINAMATH_CALUDE_alcohol_dilution_l2116_211657


namespace NUMINAMATH_CALUDE_circle_arrangement_divisible_by_three_l2116_211677

/-- A type representing the arrangement of numbers in a circle. -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- Predicate to check if two numbers differ by 1, 2, or factor of two. -/
def ValidDifference (a b : ℕ) : Prop :=
  (a = b + 1) ∨ (b = a + 1) ∨ (a = b + 2) ∨ (b = a + 2) ∨ (a = 2 * b) ∨ (b = 2 * a)

/-- Theorem stating that in any arrangement of 99 natural numbers in a circle
    where any two neighboring numbers differ either by 1, or by 2,
    or by a factor of two, at least one of these numbers is divisible by 3. -/
theorem circle_arrangement_divisible_by_three
  (arr : CircularArrangement 99)
  (h : ∀ i : Fin 99, ValidDifference (arr i) (arr (i + 1))) :
  ∃ i : Fin 99, 3 ∣ arr i :=
sorry

end NUMINAMATH_CALUDE_circle_arrangement_divisible_by_three_l2116_211677


namespace NUMINAMATH_CALUDE_calculator_trick_l2116_211694

theorem calculator_trick (a b c : ℕ) (h1 : 100 ≤ a * 100 + b * 10 + c) (h2 : a * 100 + b * 10 + c < 1000) :
  let abc := a * 100 + b * 10 + c
  let abcabc := abc * 1000 + abc
  (((abcabc / 7) / 11) / 13) = abc :=
sorry

end NUMINAMATH_CALUDE_calculator_trick_l2116_211694


namespace NUMINAMATH_CALUDE_solve_lollipops_problem_l2116_211666

def lollipops_problem (alison_lollipops henry_lollipops diane_lollipops days : ℕ) : Prop :=
  alison_lollipops = 60 ∧
  henry_lollipops = alison_lollipops + 30 ∧
  diane_lollipops = 2 * alison_lollipops ∧
  days = 6 ∧
  (alison_lollipops + henry_lollipops + diane_lollipops) / days = 45

theorem solve_lollipops_problem :
  ∃ (alison_lollipops henry_lollipops diane_lollipops days : ℕ),
    lollipops_problem alison_lollipops henry_lollipops diane_lollipops days :=
by
  sorry

end NUMINAMATH_CALUDE_solve_lollipops_problem_l2116_211666


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l2116_211650

-- Define the curve and line
def curve (x b c : ℝ) : ℝ := x^3 + b*x^2 + c
def line (x k : ℝ) : ℝ := k*x + 1

-- Define the derivative of the curve
def curve_derivative (x b : ℝ) : ℝ := 3*x^2 + 2*b*x

theorem tangent_line_b_value :
  ∀ (k b c : ℝ),
  -- The line passes through the point (1, 2)
  (line 1 k = 2) →
  -- The curve passes through the point (1, 2)
  (curve 1 b c = 2) →
  -- The slope of the line equals the derivative of the curve at x = 1
  (k = curve_derivative 1 b) →
  -- The value of b is -1
  (b = -1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l2116_211650


namespace NUMINAMATH_CALUDE_largest_s_value_l2116_211655

/-- The largest possible value of s for regular polygons satisfying given conditions -/
theorem largest_s_value : ∃ (s : ℕ), s = 121 ∧ 
  (∀ (r s' : ℕ), r ≥ s' ∧ s' ≥ 3 →
    (r - 2 : ℚ) / r * 60 = (s' - 2 : ℚ) / s' * 61 →
    s' ≤ s) :=
sorry

end NUMINAMATH_CALUDE_largest_s_value_l2116_211655


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2116_211640

def n : ℕ := 12
def k : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (n.choose k : ℚ) / (2 ^ n : ℚ) = 220 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2116_211640


namespace NUMINAMATH_CALUDE_s_tends_to_infinity_l2116_211687

/-- Sum of digits in the decimal expansion of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- s_n is the sum of digits in the decimal expansion of 2^n -/
def s (n : ℕ) : ℕ := sum_of_digits (2^n)

/-- The sequence (s_n) tends to infinity -/
theorem s_tends_to_infinity : ∀ k : ℕ, ∃ N : ℕ, ∀ n ≥ N, s n ≥ k := by
  sorry

end NUMINAMATH_CALUDE_s_tends_to_infinity_l2116_211687


namespace NUMINAMATH_CALUDE_unique_balance_l2116_211695

def weights : List ℕ := [1, 2, 4, 8, 16, 32]
def candy_weight : ℕ := 25

def is_valid_partition (partition : List ℕ × List ℕ) : Prop :=
  partition.1.length = 3 ∧ 
  partition.2.length = 3 ∧
  (partition.1 ++ partition.2).toFinset = weights.toFinset

def is_balanced (partition : List ℕ × List ℕ) : Prop :=
  (partition.1.sum + candy_weight = partition.2.sum) ∧
  is_valid_partition partition

theorem unique_balance :
  ∃! partition : List ℕ × List ℕ, 
    is_balanced partition ∧ 
    partition.2.toFinset = {4, 8, 32} := by sorry

end NUMINAMATH_CALUDE_unique_balance_l2116_211695


namespace NUMINAMATH_CALUDE_sarah_picked_45_apples_l2116_211606

/-- The number of apples Sarah's brother picked -/
def brother_apples : ℝ := 9.0

/-- The factor by which Sarah picked more apples than her brother -/
def sarah_factor : ℕ := 5

/-- The number of apples Sarah picked -/
def sarah_apples : ℝ := brother_apples * sarah_factor

theorem sarah_picked_45_apples : sarah_apples = 45 := by
  sorry

end NUMINAMATH_CALUDE_sarah_picked_45_apples_l2116_211606


namespace NUMINAMATH_CALUDE_hotdogs_served_today_l2116_211684

/-- The number of hot dogs served during lunch today -/
def lunch_hotdogs : ℕ := 9

/-- The number of hot dogs served during dinner today -/
def dinner_hotdogs : ℕ := 2

/-- The total number of hot dogs served today -/
def total_hotdogs : ℕ := lunch_hotdogs + dinner_hotdogs

theorem hotdogs_served_today : total_hotdogs = 11 := by
  sorry

end NUMINAMATH_CALUDE_hotdogs_served_today_l2116_211684


namespace NUMINAMATH_CALUDE_unique_solution_l2116_211630

theorem unique_solution (x y z : ℝ) :
  (Real.sqrt (x^3 - y) = z - 1) ∧
  (Real.sqrt (y^3 - z) = x - 1) ∧
  (Real.sqrt (z^3 - x) = y - 1) →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2116_211630


namespace NUMINAMATH_CALUDE_min_omega_value_l2116_211669

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem min_omega_value (ω φ : ℝ) (h_ω_pos : ω > 0) 
  (h_exists : ∃ x₀ : ℝ, f ω φ (x₀ + 2) - f ω φ x₀ = 4) :
  ω ≥ Real.pi / 2 ∧ ∀ ω' > 0, (∃ x₀' : ℝ, f ω' φ (x₀' + 2) - f ω' φ x₀' = 4) → ω' ≥ Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_value_l2116_211669


namespace NUMINAMATH_CALUDE_triangle_not_right_angle_l2116_211675

theorem triangle_not_right_angle (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : A / 3 = B / 4) (h3 : A / 3 = C / 5) : 
  A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_not_right_angle_l2116_211675


namespace NUMINAMATH_CALUDE_phi_value_l2116_211635

theorem phi_value : ∃ (Φ : ℕ), Φ < 10 ∧ (220 : ℚ) / Φ = 40 + 3 * Φ := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l2116_211635


namespace NUMINAMATH_CALUDE_new_person_weight_l2116_211620

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 45 →
  avg_increase = 2.5 →
  ∃ (new_weight : ℝ), new_weight = replaced_weight + (initial_count * avg_increase) :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2116_211620


namespace NUMINAMATH_CALUDE_negative_difference_l2116_211653

theorem negative_difference (P Q R S T : ℝ) 
  (h1 : P < Q) (h2 : Q < R) (h3 : R < S) (h4 : S < T) 
  (h5 : P ≠ 0) (h6 : Q ≠ 0) (h7 : R ≠ 0) (h8 : S ≠ 0) (h9 : T ≠ 0) : 
  P - Q < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l2116_211653


namespace NUMINAMATH_CALUDE_problem_solution_l2116_211604

theorem problem_solution (x y : ℝ) (h1 : x^(3*y) = 16) (h2 : x = 16) : y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2116_211604


namespace NUMINAMATH_CALUDE_hen_count_l2116_211644

theorem hen_count (total_heads : ℕ) (total_feet : ℕ) 
  (h_heads : total_heads = 48)
  (h_feet : total_feet = 140) :
  ∃ (hens cows : ℕ),
    hens + cows = total_heads ∧
    2 * hens + 4 * cows = total_feet ∧
    hens = 26 := by sorry

end NUMINAMATH_CALUDE_hen_count_l2116_211644


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2116_211612

theorem sum_of_decimals : 0.3 + 0.08 + 0.007 = 0.387 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2116_211612


namespace NUMINAMATH_CALUDE_base5_division_theorem_l2116_211674

/-- Converts a base-5 number to decimal --/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base-5 --/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Represents a number in base-5 --/
structure Base5Number where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 5

/-- Division operation for Base5Number --/
def base5Div (a b : Base5Number) : Base5Number :=
  { digits := toBase5 ((toDecimal a.digits) / (toDecimal b.digits))
    valid := sorry }

theorem base5_division_theorem :
  let a : Base5Number := ⟨[1, 0, 3, 2], sorry⟩  -- 2301 in base 5
  let b : Base5Number := ⟨[2, 2], sorry⟩        -- 22 in base 5
  let result : Base5Number := ⟨[2, 0, 1], sorry⟩  -- 102 in base 5
  base5Div a b = result := by sorry

end NUMINAMATH_CALUDE_base5_division_theorem_l2116_211674


namespace NUMINAMATH_CALUDE_fraction_simplification_l2116_211693

theorem fraction_simplification :
  (1 - 2 - 4 + 8 + 16 + 32 - 64 + 128 - 256) /
  (2 - 4 - 8 + 16 + 32 + 64 - 128 + 256) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2116_211693


namespace NUMINAMATH_CALUDE_unique_solution_equation_l2116_211621

theorem unique_solution_equation :
  ∃! x : ℝ, (x^18 + 1) * (x^16 + x^14 + x^12 + x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 18 * x^9 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l2116_211621


namespace NUMINAMATH_CALUDE_a_is_guilty_l2116_211638

-- Define the set of suspects
inductive Suspect : Type
| A : Suspect
| B : Suspect
| C : Suspect

-- Define the properties of the crime and suspects
class CrimeScene where
  involved : Suspect → Prop
  canDrive : Suspect → Prop
  usedCar : Prop

-- Define the specific conditions of this crime
axiom crime_conditions (cs : CrimeScene) :
  -- The crime was committed using a car
  cs.usedCar ∧
  -- At least one suspect was involved
  (cs.involved Suspect.A ∨ cs.involved Suspect.B ∨ cs.involved Suspect.C) ∧
  -- C never commits a crime without A
  (cs.involved Suspect.C → cs.involved Suspect.A) ∧
  -- B knows how to drive
  cs.canDrive Suspect.B

-- Theorem: A is guilty
theorem a_is_guilty (cs : CrimeScene) : cs.involved Suspect.A :=
sorry

end NUMINAMATH_CALUDE_a_is_guilty_l2116_211638


namespace NUMINAMATH_CALUDE_ab_value_l2116_211686

theorem ab_value (a b : ℝ) (h : Real.sqrt (a - 1) + b^2 - 4*b + 4 = 0) : a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2116_211686


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2116_211608

theorem sqrt_equation_solution (t : ℝ) : 
  Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4) → t = 37/10 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2116_211608


namespace NUMINAMATH_CALUDE_apple_basket_problem_l2116_211647

theorem apple_basket_problem (n : ℕ) : 
  (27 * n > 25 * n) ∧                  -- A has more apples than B
  (27 * n - 4 < 25 * n + 4) ∧          -- Moving 4 apples makes B have more
  (27 * n - 3 ≥ 25 * n + 3) →          -- Moving 3 apples doesn't make B have more
  27 * n + 25 * n = 156 :=              -- Total number of apples
by sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l2116_211647


namespace NUMINAMATH_CALUDE_coefficient_third_term_binomial_expansion_l2116_211654

theorem coefficient_third_term_binomial_expansion :
  let n : ℕ := 3
  let a : ℝ := 2
  let b : ℝ := 1
  let k : ℕ := 2
  (n.choose k) * a^(n - k) * b^k = 6 := by
sorry

end NUMINAMATH_CALUDE_coefficient_third_term_binomial_expansion_l2116_211654


namespace NUMINAMATH_CALUDE_probability_sum_nine_l2116_211643

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're looking for -/
def targetSum : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of ways to roll a sum of 9 with three dice -/
def favorableOutcomes : ℕ := 25

/-- The probability of rolling a sum of 9 with three standard six-faced dice -/
theorem probability_sum_nine :
  (favorableOutcomes : ℚ) / totalOutcomes = 25 / 216 := by sorry

end NUMINAMATH_CALUDE_probability_sum_nine_l2116_211643


namespace NUMINAMATH_CALUDE_triangle_radius_inequality_l2116_211659

/-- A structure representing a triangle with its circumradius and inradius -/
structure Triangle where
  R : ℝ  -- circumradius
  r : ℝ  -- inradius

/-- The theorem stating the relationship between circumradius and inradius of a triangle -/
theorem triangle_radius_inequality (t : Triangle) : 
  t.R ≥ 2 * t.r ∧ 
  (t.R = 2 * t.r ↔ ∃ (s : ℝ), s > 0 ∧ t.R = s * Real.sqrt 3 / 3 ∧ t.r = s / 3) ∧
  ∀ (R r : ℝ), R ≥ 2 * r → R > 0 → r > 0 → ∃ (t : Triangle), t.R = R ∧ t.r = r :=
sorry


end NUMINAMATH_CALUDE_triangle_radius_inequality_l2116_211659
