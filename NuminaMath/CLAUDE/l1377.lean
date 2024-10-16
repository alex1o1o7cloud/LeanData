import Mathlib

namespace NUMINAMATH_CALUDE_average_of_next_ten_l1377_137772

def consecutive_integers_average (c d : ℤ) : Prop :=
  (7 * d = c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) ∧
  (c > 0)

theorem average_of_next_ten (c d : ℤ) 
  (h : consecutive_integers_average c d) : 
  (((d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + 
    (d + 6) + (d + 7) + (d + 8) + (d + 9) + (d + 10)) / 10) = c + 9 :=
by sorry

end NUMINAMATH_CALUDE_average_of_next_ten_l1377_137772


namespace NUMINAMATH_CALUDE_greatest_a_value_l1377_137702

theorem greatest_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = -12) → 
  (a > 0) → 
  (∀ b : ℤ, b > a → ¬(∃ y : ℤ, y^2 + b*y = -12)) → 
  a = 13 := by
  sorry

end NUMINAMATH_CALUDE_greatest_a_value_l1377_137702


namespace NUMINAMATH_CALUDE_optimal_choice_is_104_l1377_137798

/-- Counts the number of distinct rectangles with integer sides for a given perimeter --/
def countRectangles (perimeter : ℕ) : ℕ :=
  if perimeter % 2 = 0 then
    (perimeter / 4 : ℕ)
  else
    0

/-- Checks if a number is a valid choice in the game --/
def isValidChoice (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 105

/-- Theorem stating that 104 is the optimal choice for Grisha --/
theorem optimal_choice_is_104 :
  ∀ n, isValidChoice n → countRectangles 104 ≥ countRectangles n :=
by sorry

end NUMINAMATH_CALUDE_optimal_choice_is_104_l1377_137798


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l1377_137719

theorem sin_cos_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.sin (13 * π / 180) * Real.cos (43 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l1377_137719


namespace NUMINAMATH_CALUDE_wire_necklace_length_l1377_137777

def wire_problem (num_spools : ℕ) (spool_length : ℕ) (total_necklaces : ℕ) : ℕ :=
  (num_spools * spool_length) / total_necklaces

theorem wire_necklace_length :
  wire_problem 3 20 15 = 4 :=
by sorry

end NUMINAMATH_CALUDE_wire_necklace_length_l1377_137777


namespace NUMINAMATH_CALUDE_one_quarter_of_6_75_l1377_137748

theorem one_quarter_of_6_75 : (6.75 : ℚ) / 4 = 27 / 16 := by sorry

end NUMINAMATH_CALUDE_one_quarter_of_6_75_l1377_137748


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l1377_137766

/-- Represents a club with members of different handedness and music preferences -/
structure Club where
  total : Nat
  leftHanded : Nat
  jazzLovers : Nat
  rightHandedNonJazz : Nat

/-- The number of left-handed jazz lovers in the club -/
def leftHandedJazzLovers (c : Club) : Nat :=
  c.leftHanded + c.jazzLovers - (c.total - c.rightHandedNonJazz)

/-- Theorem stating the number of left-handed jazz lovers in the specific club scenario -/
theorem left_handed_jazz_lovers_count (c : Club) 
  (h1 : c.total = 25)
  (h2 : c.leftHanded = 10)
  (h3 : c.jazzLovers = 18)
  (h4 : c.rightHandedNonJazz = 4) :
  leftHandedJazzLovers c = 7 := by
  sorry

#check left_handed_jazz_lovers_count

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l1377_137766


namespace NUMINAMATH_CALUDE_correct_log_values_l1377_137727

/-- The logarithm base 10 function -/
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- Representation of logarithmic values in terms of a, b, and c -/
structure LogValues where
  a : ℝ
  b : ℝ
  c : ℝ

theorem correct_log_values (v : LogValues) :
  (log10 0.021 = 2 * v.a + v.b + v.c - 3) →
  (log10 0.27 = 6 * v.a - 3 * v.b - 2) →
  (log10 2.8 = 1 - 2 * v.a + 2 * v.b - v.c) →
  (log10 3 = 2 * v.a - v.b) →
  (log10 5 = v.a + v.c) →
  (log10 6 = 1 + v.a - v.b - v.c) →
  (log10 8 = 3 - 3 * v.a - 3 * v.c) →
  (log10 9 = 4 * v.a - 2 * v.b) →
  (log10 14 = 1 - v.c + 2 * v.b) →
  (log10 1.5 = 3 * v.a - v.b + v.c - 1) ∧
  (log10 7 = 2 * v.b + v.c) := by
  sorry


end NUMINAMATH_CALUDE_correct_log_values_l1377_137727


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1377_137731

/-- Given two vectors a and b in ℝ², prove that if a = (1, 2) and b = (x, 4) are perpendicular, then x = -8 -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 4]
  (∀ i : Fin 2, a i * b i = 0) → x = -8 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1377_137731


namespace NUMINAMATH_CALUDE_dog_kennel_theorem_l1377_137776

theorem dog_kennel_theorem (total : ℕ) (long_fur : ℕ) (brown : ℕ) (long_fur_and_brown : ℕ) 
  (h1 : total = 45)
  (h2 : long_fur = 29)
  (h3 : brown = 17)
  (h4 : long_fur_and_brown = 9) :
  total - (long_fur + brown - long_fur_and_brown) = 8 := by
  sorry

end NUMINAMATH_CALUDE_dog_kennel_theorem_l1377_137776


namespace NUMINAMATH_CALUDE_platyfish_count_l1377_137742

/-- The number of goldfish in the tank -/
def num_goldfish : ℕ := 3

/-- The number of red balls each goldfish plays with -/
def red_balls_per_goldfish : ℕ := 10

/-- The number of white balls each platyfish plays with -/
def white_balls_per_platyfish : ℕ := 5

/-- The total number of balls in the fish tank -/
def total_balls : ℕ := 80

/-- The number of platyfish in the tank -/
def num_platyfish : ℕ := (total_balls - num_goldfish * red_balls_per_goldfish) / white_balls_per_platyfish

theorem platyfish_count : num_platyfish = 10 := by
  sorry

end NUMINAMATH_CALUDE_platyfish_count_l1377_137742


namespace NUMINAMATH_CALUDE_slope_characterization_l1377_137765

/-- The set of possible slopes for a line with y-intercept (0,3) intersecting the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (16/405) ∨ m ≥ Real.sqrt (16/405)}

/-- The equation of the line with slope m and y-intercept (0,3) -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

/-- Theorem stating that the set of possible slopes for a line with y-intercept (0,3) 
    intersecting the ellipse 4x^2 + 25y^2 = 100 is (-∞, -√(16/405)] ∪ [√(16/405), ∞) -/
theorem slope_characterization :
  ∀ m : ℝ, (∃ x : ℝ, ellipse_equation x (line_equation m x)) ↔ m ∈ possible_slopes := by
  sorry

end NUMINAMATH_CALUDE_slope_characterization_l1377_137765


namespace NUMINAMATH_CALUDE_mixture_capacity_l1377_137747

/-- Represents a vessel containing a mixture of alcohol and water -/
structure Vessel where
  capacity : ℝ
  alcohol_percentage : ℝ

/-- Represents the final mixture -/
structure FinalMixture where
  total_volume : ℝ
  vessel_capacity : ℝ

def mixture_problem (vessel1 vessel2 : Vessel) (final : FinalMixture) : Prop :=
  vessel1.capacity = 2 ∧
  vessel1.alcohol_percentage = 0.35 ∧
  vessel2.capacity = 6 ∧
  vessel2.alcohol_percentage = 0.50 ∧
  final.total_volume = 8 ∧
  final.vessel_capacity = 10 ∧
  vessel1.capacity + vessel2.capacity = final.total_volume

theorem mixture_capacity (vessel1 vessel2 : Vessel) (final : FinalMixture) 
  (h : mixture_problem vessel1 vessel2 final) : 
  final.vessel_capacity = 10 := by
  sorry

#check mixture_capacity

end NUMINAMATH_CALUDE_mixture_capacity_l1377_137747


namespace NUMINAMATH_CALUDE_circle_tangent_and_passes_through_l1377_137767

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : Prop := 4 * x - 3 * y + 6 = 0

/-- The point of tangency -/
def point_A : ℝ × ℝ := (3, 6)

/-- The point through which the circle passes -/
def point_B : ℝ × ℝ := (5, 2)

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x - 5)^2 + (y - 9/2)^2 = 25/4

/-- Theorem stating that the given circle equation represents the circle
    that is tangent to the line at point A and passes through point B -/
theorem circle_tangent_and_passes_through :
  (∀ x y, tangent_line x y → circle_equation x y → (x, y) = point_A) ∧
  circle_equation point_B.1 point_B.2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_and_passes_through_l1377_137767


namespace NUMINAMATH_CALUDE_room_length_calculation_l1377_137763

/-- Given a room with width 2.75 m and a floor paving cost of 600 per sq. metre
    resulting in a total cost of 10725, the length of the room is 6.5 meters. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  width = 2.75 ∧ cost_per_sqm = 600 ∧ total_cost = 10725 →
  total_cost = (6.5 * width * cost_per_sqm) :=
by sorry

end NUMINAMATH_CALUDE_room_length_calculation_l1377_137763


namespace NUMINAMATH_CALUDE_second_meeting_time_l1377_137790

-- Define the number of rounds Charging Bull completes in an hour
def charging_bull_rounds_per_hour : ℕ := 40

-- Define the time Racing Magic takes to complete one round (in seconds)
def racing_magic_time_per_round : ℕ := 150

-- Define the number of seconds in an hour
def seconds_per_hour : ℕ := 3600

-- Define the function to calculate the meeting time in minutes
def meeting_time : ℕ :=
  let racing_magic_rounds_per_hour := seconds_per_hour / racing_magic_time_per_round
  let lcm_rounds := Nat.lcm racing_magic_rounds_per_hour charging_bull_rounds_per_hour
  let hours_to_meet := lcm_rounds / racing_magic_rounds_per_hour
  hours_to_meet * 60

-- Theorem statement
theorem second_meeting_time :
  meeting_time = 300 := by sorry

end NUMINAMATH_CALUDE_second_meeting_time_l1377_137790


namespace NUMINAMATH_CALUDE_range_of_f_l1377_137756

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1377_137756


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1377_137733

/-- A quadratic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + b + 2

/-- The theorem statement -/
theorem quadratic_function_properties (a b : ℝ) :
  a > 0 →
  (∀ x ∈ Set.Icc 0 1, f a b x ≤ f a b 0) →
  (∀ x ∈ Set.Icc 0 1, f a b x ≥ f a b 1) →
  f a b 0 - f a b 1 = 3 →
  f a b 1 = 0 →
  a = 3 ∧ b = 1 ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc (1/3) 2, f a b x < m * x^2 + 1) ↔ m > 3) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1377_137733


namespace NUMINAMATH_CALUDE_theresa_final_week_hours_l1377_137728

/-- The number of weeks Theresa needs to work -/
def total_weeks : ℕ := 6

/-- The required average hours per week -/
def required_average : ℕ := 12

/-- The hours worked in the first five weeks -/
def first_five_weeks : List ℕ := [10, 13, 9, 14, 11]

/-- The sum of hours worked in the first five weeks -/
def sum_first_five : ℕ := first_five_weeks.sum

/-- The number of hours Theresa needs to work in the final week -/
def final_week_hours : ℕ := 15

theorem theresa_final_week_hours :
  (sum_first_five + final_week_hours) / total_weeks = required_average :=
sorry

end NUMINAMATH_CALUDE_theresa_final_week_hours_l1377_137728


namespace NUMINAMATH_CALUDE_tea_leaves_problem_l1377_137758

theorem tea_leaves_problem (num_plants : ℕ) (initial_leaves : ℕ) (fall_fraction : ℚ) : 
  num_plants = 3 → 
  initial_leaves = 18 → 
  fall_fraction = 1/3 → 
  (num_plants * initial_leaves * (1 - fall_fraction) : ℚ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_tea_leaves_problem_l1377_137758


namespace NUMINAMATH_CALUDE_average_equation_solution_l1377_137770

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 12) + (12*x + 4) + (4*x + 14)) = 8*x - 14 → x = 12 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1377_137770


namespace NUMINAMATH_CALUDE_inequality_solution_l1377_137703

theorem inequality_solution (x : ℝ) : 
  x > 11.57 → -2 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1377_137703


namespace NUMINAMATH_CALUDE_haley_trees_died_l1377_137745

/-- The number of trees that died due to a typhoon -/
def trees_died (initial_trees : ℕ) (remaining_trees : ℕ) : ℕ :=
  initial_trees - remaining_trees

/-- Proof that 2 trees died in Haley's backyard after the typhoon -/
theorem haley_trees_died : trees_died 12 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_died_l1377_137745


namespace NUMINAMATH_CALUDE_max_baggies_of_cookies_l1377_137715

def chocolateChipCookies : ℕ := 23
def oatmealCookies : ℕ := 25
def cookiesPerBaggie : ℕ := 6

theorem max_baggies_of_cookies : 
  (chocolateChipCookies + oatmealCookies) / cookiesPerBaggie = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_baggies_of_cookies_l1377_137715


namespace NUMINAMATH_CALUDE_min_value_x_one_minus_y_l1377_137761

theorem min_value_x_one_minus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 4 * x^2 + 4 * x * y + y^2 + 2 * x + y - 6 = 0) :
  ∀ z : ℝ, z > 0 → ∀ w : ℝ, w > 0 →
  4 * z^2 + 4 * z * w + w^2 + 2 * z + w - 6 = 0 →
  x * (1 - y) ≤ z * (1 - w) ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  4 * a^2 + 4 * a * b + b^2 + 2 * a + b - 6 = 0 ∧
  a * (1 - b) = -1/8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_one_minus_y_l1377_137761


namespace NUMINAMATH_CALUDE_min_correct_answers_for_john_l1377_137716

/-- Represents a mathematics competition with specific scoring rules. -/
structure MathCompetition where
  total_questions : ℕ
  correct_points : ℕ
  incorrect_deduction : ℕ
  unanswered_points : ℕ

/-- Represents a participant's strategy in the competition. -/
structure ParticipantStrategy where
  questions_attempted : ℕ
  questions_unanswered : ℕ

/-- Calculates the minimum number of correct answers needed to achieve a target score. -/
def min_correct_answers (comp : MathCompetition) (strategy : ParticipantStrategy) (target_score : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of correct answers needed in the given scenario. -/
theorem min_correct_answers_for_john (comp : MathCompetition) (strategy : ParticipantStrategy) :
    comp.total_questions = 30 →
    comp.correct_points = 8 →
    comp.incorrect_deduction = 2 →
    comp.unanswered_points = 2 →
    strategy.questions_attempted = 25 →
    strategy.questions_unanswered = 5 →
    min_correct_answers comp strategy 160 = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_john_l1377_137716


namespace NUMINAMATH_CALUDE_book_arrangement_l1377_137718

theorem book_arrangement (k m n : ℕ) :
  (∃ (f : ℕ → ℕ), f 0 = 3 * k.factorial * m.factorial * n.factorial) ∧
  (∃ (g : ℕ → ℕ), g 0 = (m + n).factorial * (m + n + 1) * k.factorial) := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_l1377_137718


namespace NUMINAMATH_CALUDE_julia_watch_collection_l1377_137760

/-- Proves that the percentage of gold watches in Julia's collection is 9.09% -/
theorem julia_watch_collection (silver : ℕ) (bronze : ℕ) (gold : ℕ) (total : ℕ) : 
  silver = 20 →
  bronze = 3 * silver →
  total = silver + bronze + gold →
  total = 88 →
  (gold : ℝ) / (total : ℝ) * 100 = 9.09 := by
  sorry

end NUMINAMATH_CALUDE_julia_watch_collection_l1377_137760


namespace NUMINAMATH_CALUDE_crescent_area_equals_rectangle_area_l1377_137774

theorem crescent_area_equals_rectangle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let rectangle_area := 4 * a * b
  let circle_area := π * (a^2 + b^2)
  let semicircles_area := π * a^2 + π * b^2
  let crescent_area := semicircles_area + rectangle_area - circle_area
  crescent_area = rectangle_area := by
  sorry

end NUMINAMATH_CALUDE_crescent_area_equals_rectangle_area_l1377_137774


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1377_137723

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 + Complex.I → z = 1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1377_137723


namespace NUMINAMATH_CALUDE_coefficient_x5_expansion_l1377_137792

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^5 in the expansion of (1-x^3)(1+x)^10
def coefficient_x5 : ℕ := binomial 10 5 - binomial 10 2

-- Theorem statement
theorem coefficient_x5_expansion :
  coefficient_x5 = 207 := by sorry

end NUMINAMATH_CALUDE_coefficient_x5_expansion_l1377_137792


namespace NUMINAMATH_CALUDE_jeep_distance_calculation_l1377_137704

theorem jeep_distance_calculation (initial_time : ℝ) (speed : ℝ) (time_factor : ℝ) :
  initial_time = 7 →
  speed = 40 →
  time_factor = 3 / 2 →
  (speed * (time_factor * initial_time)) = 420 :=
by sorry

end NUMINAMATH_CALUDE_jeep_distance_calculation_l1377_137704


namespace NUMINAMATH_CALUDE_common_root_divisibility_l1377_137787

theorem common_root_divisibility (a b c : ℤ) (h1 : c ≠ b) 
  (h2 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0) : 
  3 ∣ (a + b + 2*c) := by
sorry

end NUMINAMATH_CALUDE_common_root_divisibility_l1377_137787


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l1377_137768

theorem quadratic_equation_two_distinct_roots :
  let a : ℝ := 1
  let b : ℝ := 0
  let c : ℝ := -2
  let Δ : ℝ := b^2 - 4*a*c
  (∀ (a b c : ℝ), (b^2 - 4*a*c > 0) ↔ (∃ (x y : ℝ), x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0)) →
  ∃ (x y : ℝ), x ≠ y ∧ x^2 - 2 = 0 ∧ y^2 - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l1377_137768


namespace NUMINAMATH_CALUDE_fifth_degree_monomial_n_value_l1377_137793

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree (n : ℕ) : ℕ := n + 2 + 1

/-- A monomial 4a^nb^2c is a fifth-degree monomial if its degree is 5 -/
def is_fifth_degree (n : ℕ) : Prop := degree n = 5

theorem fifth_degree_monomial_n_value :
  ∀ n : ℕ, is_fifth_degree n → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_fifth_degree_monomial_n_value_l1377_137793


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1377_137701

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 4 + a 9 = 24) 
  (h_sixth : a 6 = 11) : 
  a 7 = 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1377_137701


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1377_137735

theorem min_value_on_circle (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → a^2 + b^2 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1377_137735


namespace NUMINAMATH_CALUDE_product_nonnegative_implies_lower_bound_l1377_137799

open Real

theorem product_nonnegative_implies_lower_bound (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (∀ x : ℝ, x > 0 → (log (a * x) - 1) * (exp x - b) ≥ 0) →
  a * b ≥ exp 2 :=
by sorry

end NUMINAMATH_CALUDE_product_nonnegative_implies_lower_bound_l1377_137799


namespace NUMINAMATH_CALUDE_probability_not_face_card_l1377_137784

theorem probability_not_face_card (total_cards : ℕ) (red_cards : ℕ) (spades_cards : ℕ)
  (red_face_cards : ℕ) (spades_face_cards : ℕ) :
  total_cards = 52 →
  red_cards = 26 →
  spades_cards = 13 →
  red_face_cards = 6 →
  spades_face_cards = 3 →
  (red_cards + spades_cards - (red_face_cards + spades_face_cards)) / (red_cards + spades_cards) = 10 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_face_card_l1377_137784


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1729_l1377_137743

theorem smallest_prime_factor_of_1729 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1729_l1377_137743


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1377_137749

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → y = 2*x) →
  (∃ x₀ : ℝ, x₀ = 5 ∧ (∀ y : ℝ, y^2 = 20*x₀ → (x₀^2 / a^2 - y^2 / b^2 = 1))) →
  a^2 = 5 ∧ b^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1377_137749


namespace NUMINAMATH_CALUDE_solution_interval_l1377_137781

theorem solution_interval (x : ℝ) : (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Ici (-4) ∩ Set.Iio (-3/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l1377_137781


namespace NUMINAMATH_CALUDE_no_triangle_cosine_sum_one_l1377_137739

theorem no_triangle_cosine_sum_one :
  ¬ ∃ (A B C : ℝ), 
    (0 < A ∧ A < π) ∧ 
    (0 < B ∧ B < π) ∧ 
    (0 < C ∧ C < π) ∧ 
    (A + B + C = π) ∧
    (Real.cos A + Real.cos B + Real.cos C = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_cosine_sum_one_l1377_137739


namespace NUMINAMATH_CALUDE_paint_for_large_cube_l1377_137752

-- Define the surface area of a cube
def surface_area (edge : ℝ) : ℝ := 6 * edge ^ 2

-- Define the paint required for a cube with edge 2 cm
def paint_for_2cm : ℝ := 1

-- Define the edge length of the larger cube
def large_cube_edge : ℝ := 6

-- Theorem to prove
theorem paint_for_large_cube : 
  (surface_area large_cube_edge / surface_area 2) * paint_for_2cm = 9 := by
  sorry

end NUMINAMATH_CALUDE_paint_for_large_cube_l1377_137752


namespace NUMINAMATH_CALUDE_triangular_array_coins_l1377_137779

theorem triangular_array_coins (N : ℕ) : 
  (N * (N + 1)) / 2 = 2010 → N = 63 ∧ (N / 10 + N % 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_coins_l1377_137779


namespace NUMINAMATH_CALUDE_vertex_of_given_function_l1377_137762

/-- A quadratic function of the form y = a(x - h)^2 + k -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The vertex of a quadratic function -/
def vertex (f : QuadraticFunction) : ℝ × ℝ := (f.h, f.k)

/-- The given quadratic function y = -2(x+1)^2 + 5 -/
def given_function : QuadraticFunction := ⟨-2, -1, 5⟩

theorem vertex_of_given_function :
  vertex given_function = (-1, 5) := by sorry

end NUMINAMATH_CALUDE_vertex_of_given_function_l1377_137762


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l1377_137721

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 4 →
  set1_mean = 10 →
  set2_count = 8 →
  set2_mean = 21 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 52 / 3 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l1377_137721


namespace NUMINAMATH_CALUDE_second_smallest_hot_dog_packs_l1377_137750

theorem second_smallest_hot_dog_packs : 
  (∃ n : ℕ, n > 0 ∧ (12 * n) % 8 = (8 - 7) % 8 ∧ 
   (∀ m : ℕ, m > 0 ∧ m < n → (12 * m) % 8 ≠ (8 - 7) % 8)) → 
  (∃ n : ℕ, n > 0 ∧ (12 * n) % 8 = (8 - 7) % 8 ∧ 
   (∃! m : ℕ, m > 0 ∧ m < n ∧ (12 * m) % 8 = (8 - 7) % 8) ∧ n = 5) :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_hot_dog_packs_l1377_137750


namespace NUMINAMATH_CALUDE_three_digit_sum_not_2021_l1377_137753

theorem three_digit_sum_not_2021 (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  a ≠ b → b ≠ c → a ≠ c → 
  a < 10 → b < 10 → c < 10 → 
  222 * (a + b + c) ≠ 2021 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_sum_not_2021_l1377_137753


namespace NUMINAMATH_CALUDE_households_without_car_or_bike_l1377_137759

/-- Prove that the number of households without either a car or a bike is 11 -/
theorem households_without_car_or_bike (total : ℕ) (car_and_bike : ℕ) (car : ℕ) (bike_only : ℕ)
  (h_total : total = 90)
  (h_car_and_bike : car_and_bike = 18)
  (h_car : car = 44)
  (h_bike_only : bike_only = 35) :
  total - (car + bike_only) = 11 := by
  sorry

end NUMINAMATH_CALUDE_households_without_car_or_bike_l1377_137759


namespace NUMINAMATH_CALUDE_saturday_duty_probability_is_one_sixth_l1377_137744

/-- A person's weekly night duty schedule -/
structure DutySchedule where
  total_duties : ℕ
  sunday_duty : Bool
  h_total : total_duties = 2
  h_sunday : sunday_duty = true

/-- The probability of being on duty on Saturday night given the duty schedule -/
def saturday_duty_probability (schedule : DutySchedule) : ℚ :=
  1 / 6

/-- Theorem stating that the probability of Saturday duty is 1/6 -/
theorem saturday_duty_probability_is_one_sixth (schedule : DutySchedule) :
  saturday_duty_probability schedule = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_saturday_duty_probability_is_one_sixth_l1377_137744


namespace NUMINAMATH_CALUDE_integral_x_exp_x_squared_l1377_137712

theorem integral_x_exp_x_squared (x : ℝ) :
  (deriv (fun x => (1/2) * Real.exp (x^2))) x = x * Real.exp (x^2) := by
  sorry

end NUMINAMATH_CALUDE_integral_x_exp_x_squared_l1377_137712


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l1377_137789

theorem system_of_equations_solutions :
  (∃ x y : ℝ, y = 2*x - 3 ∧ 2*x + y = 5 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℝ, 3*x + 4*y = 5 ∧ 5*x - 2*y = 17 ∧ x = 3 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l1377_137789


namespace NUMINAMATH_CALUDE_triploid_oyster_principle_is_chromosome_variation_l1377_137736

/-- Represents the principle underlying oyster cultivation methods -/
inductive CultivationPrinciple
  | GeneticMutation
  | ChromosomeNumberVariation
  | GeneRecombination
  | ChromosomeStructureVariation

/-- Represents the ploidy level of an oyster -/
inductive Ploidy
  | Diploid
  | Triploid

/-- Represents the state of a cell during oyster reproduction -/
structure CellState where
  chromosomeSets : ℕ
  polarBodyReleased : Bool

/-- Represents the cultivation method for oysters -/
structure CultivationMethod where
  chemicalTreatment : Bool
  preventPolarBodyRelease : Bool
  solveFleshQualityDecline : Bool

/-- The principle of triploid oyster cultivation -/
def triploidOysterPrinciple (method : CultivationMethod) : CultivationPrinciple :=
  sorry

/-- Theorem stating that the principle of triploid oyster cultivation
    is chromosome number variation -/
theorem triploid_oyster_principle_is_chromosome_variation
  (method : CultivationMethod)
  (h1 : method.chemicalTreatment = true)
  (h2 : method.preventPolarBodyRelease = true)
  (h3 : method.solveFleshQualityDecline = true) :
  triploidOysterPrinciple method = CultivationPrinciple.ChromosomeNumberVariation :=
  sorry

end NUMINAMATH_CALUDE_triploid_oyster_principle_is_chromosome_variation_l1377_137736


namespace NUMINAMATH_CALUDE_count_random_events_l1377_137725

-- Define the set of events
inductive Event
| DiceRoll
| Rain
| Lottery
| SumGreaterThanTwo
| WaterBoiling

-- Define a function to check if an event is random
def isRandomEvent : Event → Bool
| Event.DiceRoll => true
| Event.Rain => true
| Event.Lottery => true
| Event.SumGreaterThanTwo => false
| Event.WaterBoiling => false

-- Theorem: The number of random events is 3
theorem count_random_events :
  (List.filter isRandomEvent [Event.DiceRoll, Event.Rain, Event.Lottery, Event.SumGreaterThanTwo, Event.WaterBoiling]).length = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_random_events_l1377_137725


namespace NUMINAMATH_CALUDE_max_candies_eaten_l1377_137722

theorem max_candies_eaten (n : ℕ) (h : n = 25) : 
  (n.choose 2) = 300 := by
  sorry

#check max_candies_eaten

end NUMINAMATH_CALUDE_max_candies_eaten_l1377_137722


namespace NUMINAMATH_CALUDE_viewers_scientific_notation_l1377_137794

/-- Represents 1 billion -/
def billion : ℝ := 1000000000

/-- The number of viewers who watched the Spring Festival Gala live broadcast -/
def viewers : ℝ := 1.173 * billion

/-- Theorem stating that the number of viewers in billions is equal to its scientific notation -/
theorem viewers_scientific_notation : viewers = 1.173 * (10 : ℝ)^9 := by
  sorry

end NUMINAMATH_CALUDE_viewers_scientific_notation_l1377_137794


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l1377_137705

/-- The number of people sitting around the table -/
def total_people : ℕ := 11

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 3

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 3

/-- The probability of math majors sitting consecutively -/
def consecutive_math_prob : ℚ := 1 / 42

theorem math_majors_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := Nat.factorial (total_people - math_majors) * Nat.factorial math_majors
  (favorable_arrangements : ℚ) / total_arrangements = consecutive_math_prob := by
  sorry

#check math_majors_consecutive_probability

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l1377_137705


namespace NUMINAMATH_CALUDE_circle_ray_angle_l1377_137726

/-- In a circle with twelve evenly spaced rays, where one ray points north,
    the smaller angle between the north-pointing ray and the southeast-pointing ray is 90°. -/
theorem circle_ray_angle (n : ℕ) (θ : ℝ) : 
  n = 12 → θ = 360 / n → θ * 3 = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_ray_angle_l1377_137726


namespace NUMINAMATH_CALUDE_trolley_passengers_l1377_137720

/-- The number of people on a trolley after three stops -/
def people_on_trolley (initial_pickup : ℕ) (second_stop_off : ℕ) (second_stop_on : ℕ) 
  (third_stop_off : ℕ) (third_stop_on : ℕ) : ℕ :=
  initial_pickup - second_stop_off + second_stop_on - third_stop_off + third_stop_on

/-- Theorem stating the number of people on the trolley after three stops -/
theorem trolley_passengers : 
  people_on_trolley 10 3 20 18 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_trolley_passengers_l1377_137720


namespace NUMINAMATH_CALUDE_exists_divisible_by_five_l1377_137730

def T : Set ℤ := {s | ∃ a : ℤ, s = a^2 + (a+1)^2 + (a+2)^2 + (a+3)^2}

theorem exists_divisible_by_five : ∃ s ∈ T, 5 ∣ s := by sorry

end NUMINAMATH_CALUDE_exists_divisible_by_five_l1377_137730


namespace NUMINAMATH_CALUDE_factorial_difference_l1377_137797

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1377_137797


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1377_137773

theorem complex_fraction_equality : (3 - Complex.I) / (1 - Complex.I) = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1377_137773


namespace NUMINAMATH_CALUDE_lori_marble_sharing_l1377_137738

theorem lori_marble_sharing :
  ∀ (total_marbles : ℕ) (share_percent : ℚ) (num_friends : ℕ),
    total_marbles = 60 →
    share_percent = 75 / 100 →
    num_friends = 5 →
    (total_marbles : ℚ) * share_percent / num_friends = 9 := by
  sorry

end NUMINAMATH_CALUDE_lori_marble_sharing_l1377_137738


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l1377_137796

/-- Given two rectangles with equal areas, where one rectangle has dimensions 5 by 24 inches
    and the other has a length of 3 inches, prove that the width of the second rectangle is 40 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length : ℝ)
    (carol_area jordan_area : ℝ) (h1 : carol_length = 5)
    (h2 : carol_width = 24) (h3 : jordan_length = 3)
    (h4 : carol_area = carol_length * carol_width)
    (h5 : jordan_area = jordan_length * (jordan_area / jordan_length))
    (h6 : carol_area = jordan_area) :
  jordan_area / jordan_length = 40 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l1377_137796


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1377_137754

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4
def circle_N (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 100

-- Define the property of being externally tangent
def externally_tangent (x y R : ℝ) : Prop :=
  ∃ (x_m y_m : ℝ), circle_M x_m y_m ∧ (x - x_m)^2 + (y - y_m)^2 = (R + 2)^2

-- Define the property of being internally tangent
def internally_tangent (x y R : ℝ) : Prop :=
  ∃ (x_n y_n : ℝ), circle_N x_n y_n ∧ (x - x_n)^2 + (y - y_n)^2 = (10 - R)^2

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (x y R : ℝ),
    externally_tangent x y R →
    internally_tangent x y R →
    x^2 / 36 + y^2 / 27 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1377_137754


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_range_l1377_137755

/-- Given real numbers a, b, c forming a geometric sequence with sum 1,
    prove that a + c is non-negative and unbounded above. -/
theorem geometric_sequence_sum_range (a b c : ℝ) : 
  (∃ r : ℝ, a = r ∧ b = r^2 ∧ c = r^3) →  -- geometric sequence condition
  a + b + c = 1 →                        -- sum condition
  (a + c ≥ 0 ∧ ∀ M : ℝ, ∃ x y z : ℝ, 
    (∃ r : ℝ, x = r ∧ y = r^2 ∧ z = r^3) ∧ 
    x + y + z = 1 ∧ 
    x + z > M) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_range_l1377_137755


namespace NUMINAMATH_CALUDE_average_weight_a_b_l1377_137771

/-- Given the weights of three people a, b, and c, prove that the average weight of a and b is 40 kg -/
theorem average_weight_a_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 ∧ 
  (b + c) / 2 = 43 ∧ 
  b = 31 → 
  (a + b) / 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_weight_a_b_l1377_137771


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1377_137734

theorem quadratic_equation_roots (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (2*k + 1)*x + k^2 + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (k > 3/4) ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ + x₂ = -x₁ * x₂ → k = 2) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equation_roots_l1377_137734


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_eq_zero_l1377_137764

theorem cos_alpha_minus_pi_sixth_eq_zero (α : Real)
  (h1 : 2 * Real.tan α * Real.sin α = 3)
  (h2 : -Real.pi/2 < α)
  (h3 : α < 0) :
  Real.cos (α - Real.pi/6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_eq_zero_l1377_137764


namespace NUMINAMATH_CALUDE_negation_equivalence_l1377_137717

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1377_137717


namespace NUMINAMATH_CALUDE_order_of_numbers_l1377_137786

def Ψ : ℤ := -1006

def Ω : ℤ := -1007

def Θ : ℤ := -1008

theorem order_of_numbers : Θ < Ω ∧ Ω < Ψ := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l1377_137786


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l1377_137780

-- Define the function f
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x, f x = f (-x)) ∧                   -- f is an even function
  (∀ x y, 0 ≤ x ∧ x ≤ y → f y ≤ f x) ∧    -- f is monotonically decreasing on [0,+∞)
  (∀ y, ∃ x, f x = y ↔ y ≤ 0) :=          -- Range of f is (-∞,0]
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l1377_137780


namespace NUMINAMATH_CALUDE_people_per_column_second_arrangement_l1377_137700

/-- 
Given a group of people that can be arranged in two ways:
1. 16 columns with 30 people per column
2. 8 columns with an unknown number of people per column

This theorem proves that the number of people per column in the second arrangement is 60.
-/
theorem people_per_column_second_arrangement 
  (total_people : ℕ) 
  (columns_first : ℕ) 
  (people_per_column_first : ℕ) 
  (columns_second : ℕ) : 
  columns_first = 16 → 
  people_per_column_first = 30 → 
  columns_second = 8 → 
  total_people = columns_first * people_per_column_first → 
  total_people / columns_second = 60 := by
  sorry

end NUMINAMATH_CALUDE_people_per_column_second_arrangement_l1377_137700


namespace NUMINAMATH_CALUDE_remainder_of_M_div_500_l1377_137788

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def product_of_factorials : ℕ := (List.range 50).foldl (fun acc n => acc * factorial (n + 1)) 1

def trailing_zeros (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n.digits 10).takeWhile (· = 0) |>.length

def M : ℕ := trailing_zeros product_of_factorials

theorem remainder_of_M_div_500 : M % 500 = 21 := by sorry

end NUMINAMATH_CALUDE_remainder_of_M_div_500_l1377_137788


namespace NUMINAMATH_CALUDE_total_learning_time_is_19_l1377_137782

/-- Represents the learning time for each vowel -/
def vowel_time : Fin 5 → ℕ
  | 0 => 4  -- A
  | 1 => 6  -- E
  | 2 => 5  -- I
  | 3 => 3  -- O
  | 4 => 4  -- U

/-- The break time between learning pairs -/
def break_time : ℕ := 2

/-- Calculates the total learning time for all vowels -/
def total_learning_time : ℕ :=
  let pair1 := max (vowel_time 1) (vowel_time 3)  -- E and O
  let pair2 := max (vowel_time 2) (vowel_time 4)  -- I and U
  let single := vowel_time 0  -- A
  pair1 + break_time + pair2 + break_time + single

/-- Theorem stating that the total learning time is 19 days -/
theorem total_learning_time_is_19 : total_learning_time = 19 := by
  sorry

#eval total_learning_time

end NUMINAMATH_CALUDE_total_learning_time_is_19_l1377_137782


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l1377_137778

theorem complex_in_second_quadrant :
  let z : ℂ := Complex.mk (Real.cos 2) (Real.sin 3)
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l1377_137778


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1377_137757

theorem absolute_value_simplification : |(-5^2 + 7 - 3)| = 21 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1377_137757


namespace NUMINAMATH_CALUDE_pyramid_volume_l1377_137724

/-- The volume of a pyramid with a rectangular base, lateral edges of length l,
    and angles α and β between the lateral edges and adjacent sides of the base. -/
theorem pyramid_volume (l α β : ℝ) (hl : l > 0) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ∃ V : ℝ, V = (4 / 3) * l^3 * Real.cos α * Real.cos β * Real.sqrt (-Real.cos (α + β) * Real.cos (α - β)) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1377_137724


namespace NUMINAMATH_CALUDE_f_monotonicity_l1377_137706

-- Define the function
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- State the theorem
theorem f_monotonicity :
  (∀ x y, x < y ∧ y < -1/3 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x y, -1/3 < x ∧ x < y ∧ y < 1 → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_l1377_137706


namespace NUMINAMATH_CALUDE_bank_profit_maximization_l1377_137710

/-- The bank's profit maximization problem -/
theorem bank_profit_maximization
  (k : ℝ) -- Proportionality constant
  (h_k_pos : k > 0) -- k is positive
  (loan_rate : ℝ := 0.048) -- Loan interest rate
  (deposit_rate : ℝ) -- Deposit interest rate
  (h_deposit_rate : deposit_rate > 0 ∧ deposit_rate < loan_rate) -- Deposit rate is between 0 and loan rate
  (deposit_amount : ℝ := k * deposit_rate^2) -- Deposit amount formula
  (profit : ℝ → ℝ := λ x => loan_rate * k * x^2 - k * x^3) -- Profit function
  : (∀ x, x > 0 ∧ x < loan_rate → profit x ≤ profit 0.032) :=
by sorry

end NUMINAMATH_CALUDE_bank_profit_maximization_l1377_137710


namespace NUMINAMATH_CALUDE_lisa_photos_l1377_137729

/-- The number of photos Lisa took this weekend -/
def total_photos (animal_photos flower_photos scenery_photos : ℕ) : ℕ :=
  animal_photos + flower_photos + scenery_photos

theorem lisa_photos : 
  ∀ (animal_photos flower_photos scenery_photos : ℕ),
    animal_photos = 10 →
    flower_photos = 3 * animal_photos →
    scenery_photos = flower_photos - 10 →
    total_photos animal_photos flower_photos scenery_photos = 60 := by
  sorry

#check lisa_photos

end NUMINAMATH_CALUDE_lisa_photos_l1377_137729


namespace NUMINAMATH_CALUDE_games_for_23_teams_l1377_137746

/-- A single-elimination tournament where teams are eliminated after one loss and no ties are possible. -/
structure Tournament :=
  (num_teams : ℕ)

/-- The number of games needed to declare a champion in a single-elimination tournament. -/
def games_to_champion (t : Tournament) : ℕ := t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams, 22 games are needed to declare a champion. -/
theorem games_for_23_teams :
  ∀ t : Tournament, t.num_teams = 23 → games_to_champion t = 22 := by
  sorry


end NUMINAMATH_CALUDE_games_for_23_teams_l1377_137746


namespace NUMINAMATH_CALUDE_diagonal_length_is_2_8_l1377_137751

/-- Represents a quadrilateral with given side lengths and a diagonal -/
structure Quadrilateral :=
  (side1 side2 side3 side4 diagonal : ℝ)

/-- Checks if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Checks if the diagonal forms valid triangles with all possible combinations of sides -/
def diagonal_forms_valid_triangles (q : Quadrilateral) : Prop :=
  is_valid_triangle q.diagonal q.side1 q.side2 ∧
  is_valid_triangle q.diagonal q.side1 q.side3 ∧
  is_valid_triangle q.diagonal q.side1 q.side4 ∧
  is_valid_triangle q.diagonal q.side2 q.side3 ∧
  is_valid_triangle q.diagonal q.side2 q.side4 ∧
  is_valid_triangle q.diagonal q.side3 q.side4

theorem diagonal_length_is_2_8 (q : Quadrilateral) 
  (h1 : q.side1 = 1) (h2 : q.side2 = 2) (h3 : q.side3 = 5) (h4 : q.side4 = 7.5) (h5 : q.diagonal = 2.8) :
  diagonal_forms_valid_triangles q :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_is_2_8_l1377_137751


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1377_137713

theorem other_root_of_quadratic (m : ℝ) :
  (3 * (1 : ℝ)^2 + m * 1 = 5) →
  (3 * (-5/3 : ℝ)^2 + m * (-5/3) = 5) ∧
  (∀ x : ℝ, 3 * x^2 + m * x = 5 → x = 1 ∨ x = -5/3) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1377_137713


namespace NUMINAMATH_CALUDE_coronavirus_cases_difference_l1377_137791

theorem coronavirus_cases_difference (new_york california texas : ℕ) : 
  new_york = 2000 →
  california = new_york / 2 →
  new_york + california + texas = 3600 →
  texas < california →
  california - texas = 400 :=
by sorry

end NUMINAMATH_CALUDE_coronavirus_cases_difference_l1377_137791


namespace NUMINAMATH_CALUDE_exam_comparison_l1377_137711

theorem exam_comparison (total_items : ℕ) (lyssa_incorrect_percent : ℚ) (precious_mistakes : ℕ)
  (h1 : total_items = 120)
  (h2 : lyssa_incorrect_percent = 25 / 100)
  (h3 : precious_mistakes = 17) :
  (total_items - (lyssa_incorrect_percent * total_items).num) - (total_items - precious_mistakes) = -13 :=
by sorry

end NUMINAMATH_CALUDE_exam_comparison_l1377_137711


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1377_137714

theorem arithmetic_calculation : 1984 + 180 / 60 - 284 = 1703 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1377_137714


namespace NUMINAMATH_CALUDE_pascal_triangle_15th_row_5th_number_l1377_137769

theorem pascal_triangle_15th_row_5th_number : Nat.choose 15 4 = 1365 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_15th_row_5th_number_l1377_137769


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1377_137737

theorem max_value_of_expression (t : ℝ) :
  (∃ (c : ℝ), ∀ (t : ℝ), (3^t - 4*t)*t / 9^t ≤ c) ∧
  (∃ (t : ℝ), (3^t - 4*t)*t / 9^t = 1/16) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1377_137737


namespace NUMINAMATH_CALUDE_volume_surface_area_ratio_l1377_137775

/-- Represents a shape created by joining unit cubes -/
structure CubeShape where
  /-- The number of unit cubes in the shape -/
  num_cubes : ℕ
  /-- The number of cubes surrounding the center cube -/
  surrounding_cubes : ℕ
  /-- Whether there's an additional cube on top -/
  has_top_cube : Bool

/-- Calculates the volume of the shape -/
def volume (shape : CubeShape) : ℕ := shape.num_cubes

/-- Calculates the surface area of the shape -/
def surface_area (shape : CubeShape) : ℕ :=
  shape.surrounding_cubes * 4 + (if shape.has_top_cube then 5 else 0)

/-- The specific shape described in the problem -/
def problem_shape : CubeShape :=
  { num_cubes := 8
  , surrounding_cubes := 6
  , has_top_cube := true }

theorem volume_surface_area_ratio :
  (volume problem_shape : ℚ) / (surface_area problem_shape : ℚ) = 8 / 29 := by sorry

end NUMINAMATH_CALUDE_volume_surface_area_ratio_l1377_137775


namespace NUMINAMATH_CALUDE_sum_of_coordinates_symmetric_points_l1377_137707

/-- Two points A(a, 2022) and A'(-2023, b) are symmetric with respect to the origin if and only if
    their coordinates satisfy the given conditions. -/
def symmetric_points (a b : ℝ) : Prop :=
  a = 2023 ∧ b = -2022

/-- The sum of a and b is 1 when A(a, 2022) and A'(-2023, b) are symmetric with respect to the origin. -/
theorem sum_of_coordinates_symmetric_points (a b : ℝ) 
    (h : symmetric_points a b) : a + b = 1 := by
  sorry

#check sum_of_coordinates_symmetric_points

end NUMINAMATH_CALUDE_sum_of_coordinates_symmetric_points_l1377_137707


namespace NUMINAMATH_CALUDE_initial_balloons_eq_sum_l1377_137732

/-- The number of balloons Tom initially had -/
def initial_balloons : ℕ := 30

/-- The number of balloons Tom gave to Fred -/
def balloons_given : ℕ := 16

/-- The number of balloons Tom has left -/
def balloons_left : ℕ := 14

/-- Theorem stating that the initial number of balloons is equal to
    the sum of balloons given away and balloons left -/
theorem initial_balloons_eq_sum :
  initial_balloons = balloons_given + balloons_left := by
  sorry

end NUMINAMATH_CALUDE_initial_balloons_eq_sum_l1377_137732


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l1377_137785

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = !![5, -2; 1, 3] →
  (A^3)⁻¹ = !![111, -34; 47, 5] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l1377_137785


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1377_137741

theorem sufficient_but_not_necessary (a b : ℝ) :
  (∀ a b, (a + b)/2 < Real.sqrt (a * b) → |a + b| = |a| + |b|) ∧
  (∃ a b, |a + b| = |a| + |b| ∧ (a + b)/2 ≥ Real.sqrt (a * b)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1377_137741


namespace NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l1377_137708

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a^2 - |b|

-- Theorem to prove
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l1377_137708


namespace NUMINAMATH_CALUDE_power_six_sum_l1377_137709

theorem power_six_sum (x : ℝ) (h : x + 1/x = 4) : x^6 + 1/x^6 = 2702 := by
  sorry

end NUMINAMATH_CALUDE_power_six_sum_l1377_137709


namespace NUMINAMATH_CALUDE_tan_3_expression_zero_l1377_137795

theorem tan_3_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_3_expression_zero_l1377_137795


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l1377_137740

/-- An equilateral hexagon with three nonadjacent 120° angles -/
structure SpecialHexagon where
  -- Side length
  s : ℝ
  -- Condition that s is positive
  s_pos : s > 0
  -- Area of the hexagon
  area : ℝ
  -- Condition that area is 12 square units
  area_eq : area = 12

/-- The perimeter of a SpecialHexagon is 24 units -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : 
  6 * h.s = 24 := by sorry

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l1377_137740


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1377_137783

/-- The length of the major axis of the ellipse 2x^2 + y^2 = 8 is 4√2 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | 2 * x^2 + y^2 = 8}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ ellipse ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    2 * a = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1377_137783
