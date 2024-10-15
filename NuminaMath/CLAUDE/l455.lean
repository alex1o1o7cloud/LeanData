import Mathlib

namespace NUMINAMATH_CALUDE_grant_coverage_percentage_l455_45526

def total_cost : ℝ := 30000
def savings : ℝ := 10000
def loan_amount : ℝ := 12000

theorem grant_coverage_percentage : 
  let remainder := total_cost - savings
  let grant_amount := remainder - loan_amount
  (grant_amount / remainder) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_grant_coverage_percentage_l455_45526


namespace NUMINAMATH_CALUDE_complex_number_equality_l455_45542

theorem complex_number_equality : ∀ z : ℂ, z = (Complex.I ^ 3) / (1 + Complex.I) → z = (-1 - Complex.I) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l455_45542


namespace NUMINAMATH_CALUDE_complement_of_M_l455_45541

-- Define the universal set U
def U : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define the set M
def M : Set ℝ := {x | x^2 - x ≤ 0}

-- Theorem statement
theorem complement_of_M (x : ℝ) : 
  x ∈ (U \ M) ↔ (1 < x ∧ x ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_complement_of_M_l455_45541


namespace NUMINAMATH_CALUDE_correct_articles_for_newton_discovery_l455_45523

/-- Represents the possible article choices for each blank --/
inductive Article
  | A : Article  -- represents "a"
  | The : Article  -- represents "the"
  | None : Article  -- represents no article

/-- Represents the context of the discovery --/
structure DiscoveryContext where
  is_specific : Bool
  is_previously_mentioned : Bool

/-- Represents the usage of "man" in the sentence --/
structure ManUsage where
  represents_mankind : Bool

/-- Determines the correct article choice given the context --/
def correct_article (context : DiscoveryContext) (man_usage : ManUsage) : Article × Article :=
  sorry

/-- Theorem stating the correct article choice for the given sentence --/
theorem correct_articles_for_newton_discovery 
  (context : DiscoveryContext)
  (man_usage : ManUsage)
  (h1 : context.is_specific = true)
  (h2 : context.is_previously_mentioned = false)
  (h3 : man_usage.represents_mankind = true) :
  correct_article context man_usage = (Article.A, Article.The) :=
sorry

end NUMINAMATH_CALUDE_correct_articles_for_newton_discovery_l455_45523


namespace NUMINAMATH_CALUDE_snow_volume_calculation_l455_45591

/-- Calculates the volume of snow to be shoveled from a partially melted rectangular pathway -/
theorem snow_volume_calculation (length width : ℝ) (depth_full depth_half : ℝ) 
  (h_length : length = 30)
  (h_width : width = 4)
  (h_depth_full : depth_full = 1)
  (h_depth_half : depth_half = 1/2) :
  length * width * depth_full / 2 + length * width * depth_half / 2 = 90 :=
by sorry

end NUMINAMATH_CALUDE_snow_volume_calculation_l455_45591


namespace NUMINAMATH_CALUDE_average_people_moving_per_hour_l455_45510

/-- The number of people moving to Florida -/
def people_moving : ℕ := 3000

/-- The number of days -/
def days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculate the average number of people moving per hour -/
def average_per_hour : ℚ :=
  people_moving / (days * hours_per_day)

/-- Round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem average_people_moving_per_hour :
  round_to_nearest average_per_hour = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_people_moving_per_hour_l455_45510


namespace NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_l455_45590

theorem not_p_or_q_false_implies_p_or_q (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
sorry

end NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_l455_45590


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l455_45522

def U : Set ℕ := {x | x^2 - 4*x - 5 ≤ 0}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3, 5}

theorem intersection_complement_equals_set : A ∩ (U \ B) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l455_45522


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l455_45593

/-- The line equation is 5y + 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line 5y + 3x = 15 with the y-axis is (0, 3) -/
theorem line_y_axis_intersection :
  ∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l455_45593


namespace NUMINAMATH_CALUDE_rectangular_field_length_l455_45549

theorem rectangular_field_length : 
  ∀ (w : ℝ), 
    w > 0 → 
    w^2 + (w + 10)^2 = 22^2 → 
    w + 10 = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l455_45549


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l455_45556

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 1 ∧ p.b = -2 ∧ p.c = -5 →
  let p_shifted := shift_parabola p 2 3
  p_shifted.a = 1 ∧ p_shifted.b = 2 ∧ p_shifted.c = -3 := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l455_45556


namespace NUMINAMATH_CALUDE_games_before_third_l455_45529

theorem games_before_third (average_score : ℝ) (third_game_score : ℝ) (points_needed : ℝ) :
  average_score = 61.5 →
  third_game_score = 47 →
  points_needed = 330 →
  (∃ n : ℕ, n * average_score + third_game_score + points_needed = 500 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_games_before_third_l455_45529


namespace NUMINAMATH_CALUDE_probability_not_above_x_axis_is_half_l455_45511

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Defines the specific parallelogram ABCD from the problem -/
def ABCD : Parallelogram := {
  A := { x := 3, y := 3 }
  B := { x := -3, y := -3 }
  C := { x := -9, y := -3 }
  D := { x := -3, y := 3 }
}

/-- Probability of a point being not above the x-axis in the parallelogram -/
def probability_not_above_x_axis (p : Parallelogram) : ℚ :=
  1/2

/-- Theorem stating that the probability of a randomly selected point 
    from the region determined by parallelogram ABCD being not above 
    the x-axis is 1/2 -/
theorem probability_not_above_x_axis_is_half : 
  probability_not_above_x_axis ABCD = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_probability_not_above_x_axis_is_half_l455_45511


namespace NUMINAMATH_CALUDE_max_base_eight_digit_sum_l455_45505

def base_eight_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem max_base_eight_digit_sum (n : ℕ) (h : n < 1728) :
  (base_eight_digits n).sum ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_base_eight_digit_sum_l455_45505


namespace NUMINAMATH_CALUDE_hibiscus_flowers_solution_l455_45540

/-- The number of flowers on Mario's first hibiscus plant -/
def first_plant_flowers : ℕ := 2

/-- The number of flowers on Mario's second hibiscus plant -/
def second_plant_flowers : ℕ := 2 * first_plant_flowers

/-- The number of flowers on Mario's third hibiscus plant -/
def third_plant_flowers : ℕ := 4 * second_plant_flowers

/-- The total number of flowers on all three plants -/
def total_flowers : ℕ := 22

theorem hibiscus_flowers_solution :
  first_plant_flowers + second_plant_flowers + third_plant_flowers = total_flowers ∧
  first_plant_flowers = 2 := by
  sorry

end NUMINAMATH_CALUDE_hibiscus_flowers_solution_l455_45540


namespace NUMINAMATH_CALUDE_max_a_items_eleven_a_items_possible_l455_45501

/-- Represents the number of items purchased for each stationery type -/
structure Stationery where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total cost of the stationery purchase -/
def totalCost (s : Stationery) : ℕ :=
  3 * s.a + 2 * s.b + s.c

/-- Checks if the purchase satisfies all conditions -/
def isValidPurchase (s : Stationery) : Prop :=
  s.b = s.a - 2 ∧
  3 * s.a ≤ 33 ∧
  totalCost s = 66

/-- Theorem stating that the maximum number of A items that can be purchased is 11 -/
theorem max_a_items : ∀ s : Stationery, isValidPurchase s → s.a ≤ 11 :=
  sorry

/-- Theorem stating that 11 A items can actually be purchased -/
theorem eleven_a_items_possible : ∃ s : Stationery, isValidPurchase s ∧ s.a = 11 :=
  sorry

end NUMINAMATH_CALUDE_max_a_items_eleven_a_items_possible_l455_45501


namespace NUMINAMATH_CALUDE_sufficient_condition_for_a_gt_b_l455_45518

theorem sufficient_condition_for_a_gt_b (a b : ℝ) : 
  (1 / a < 1 / b) ∧ (1 / b < 0) → a > b := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_a_gt_b_l455_45518


namespace NUMINAMATH_CALUDE_freshman_count_proof_l455_45570

theorem freshman_count_proof :
  ∃! n : ℕ, n < 600 ∧ n % 25 = 24 ∧ n % 19 = 10 ∧ n = 574 := by
  sorry

end NUMINAMATH_CALUDE_freshman_count_proof_l455_45570


namespace NUMINAMATH_CALUDE_parabola_vertex_l455_45569

/-- A quadratic function f(x) = -x^2 + ax + b where f(x) ≤ 0 
    has the solution (-∞,-3] ∪ [5,∞) -/
def f (a b x : ℝ) : ℝ := -x^2 + a*x + b

/-- The solution set of f(x) ≤ 0 -/
def solution_set (a b : ℝ) : Set ℝ :=
  {x | x ≤ -3 ∨ x ≥ 5}

/-- The vertex of the parabola -/
def vertex (a b : ℝ) : ℝ × ℝ := (1, 16)

theorem parabola_vertex (a b : ℝ) 
  (h : ∀ x, f a b x ≤ 0 ↔ x ∈ solution_set a b) :
  vertex a b = (1, 16) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l455_45569


namespace NUMINAMATH_CALUDE_function_composition_equality_l455_45515

/-- Given a function f(x) = a x^2 - √2, where a is a constant,
    prove that if f(f(√2)) = -√2, then a = √2 / 2 -/
theorem function_composition_equality (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 - Real.sqrt 2
  f (f (Real.sqrt 2)) = -Real.sqrt 2 → a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l455_45515


namespace NUMINAMATH_CALUDE_inequality_proof_l455_45514

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1 / x + 1 / y ≤ 1 / x^2 + 1 / y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l455_45514


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l455_45509

theorem cyclist_speed_ratio :
  ∀ (v_A v_B : ℝ),
    v_A > 0 →
    v_B > 0 →
    v_A < v_B →
    (v_B - v_A) * 4.5 = 10 →
    v_A + v_B = 10 →
    v_A / v_B = 61 / 29 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l455_45509


namespace NUMINAMATH_CALUDE_number_line_steps_l455_45567

theorem number_line_steps (total_distance : ℝ) (num_steps : ℕ) (step_to_x : ℕ) : 
  total_distance = 32 →
  num_steps = 8 →
  step_to_x = 6 →
  (total_distance / num_steps) * step_to_x = 24 := by
sorry

end NUMINAMATH_CALUDE_number_line_steps_l455_45567


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l455_45564

/-- Given an initial angle of 45 degrees that is rotated 510 degrees clockwise,
    the resulting new acute angle measures 75 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 45 → 
  rotation = 510 → 
  (((rotation % 360) - initial_angle) % 180) = 75 :=
by sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l455_45564


namespace NUMINAMATH_CALUDE_trivia_contest_probability_l455_45506

def num_questions : ℕ := 4
def num_choices : ℕ := 4
def min_correct : ℕ := 3

def probability_correct_guess : ℚ := 1 / num_choices

def probability_winning : ℚ :=
  (num_questions.choose min_correct) * (probability_correct_guess ^ min_correct) * ((1 - probability_correct_guess) ^ (num_questions - min_correct)) +
  (num_questions.choose (min_correct + 1)) * (probability_correct_guess ^ (min_correct + 1)) * ((1 - probability_correct_guess) ^ (num_questions - (min_correct + 1)))

theorem trivia_contest_probability : probability_winning = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_trivia_contest_probability_l455_45506


namespace NUMINAMATH_CALUDE_dog_travel_time_l455_45574

theorem dog_travel_time (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 20)
  (h2 : speed1 = 10)
  (h3 : speed2 = 5) :
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_dog_travel_time_l455_45574


namespace NUMINAMATH_CALUDE_cone_volume_approximation_l455_45559

theorem cone_volume_approximation (L h : ℝ) (h1 : L > 0) (h2 : h > 0) :
  (1 / 75 : ℝ) * L^2 * h = (1 / 3 : ℝ) * ((25 / 4) / 4) * L^2 * h := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_approximation_l455_45559


namespace NUMINAMATH_CALUDE_parallel_line_slope_l455_45528

/-- The slope of any line parallel to the line containing points (3, -2) and (1, 5) is -7/2 -/
theorem parallel_line_slope : ∀ (m : ℚ), 
  (∃ (b : ℚ), ∀ (x y : ℚ), y = m * x + b → 
    (∃ (k : ℚ), y - (-2) = m * (x - 3) ∧ y - 5 = m * (x - 1))) → 
  m = -7/2 := by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l455_45528


namespace NUMINAMATH_CALUDE_commodity_price_difference_l455_45561

theorem commodity_price_difference (total_cost first_price : ℕ) 
  (h1 : total_cost = 827)
  (h2 : first_price = 477)
  (h3 : first_price > total_cost - first_price) : 
  first_price - (total_cost - first_price) = 127 := by
  sorry

end NUMINAMATH_CALUDE_commodity_price_difference_l455_45561


namespace NUMINAMATH_CALUDE_student_count_l455_45579

theorem student_count (stars_per_student : ℕ) (total_stars : ℕ) (h1 : stars_per_student = 3) (h2 : total_stars = 372) :
  total_stars / stars_per_student = 124 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l455_45579


namespace NUMINAMATH_CALUDE_percentage_calculation_l455_45575

theorem percentage_calculation (N I P : ℝ) : 
  N = 93.75 →
  I = 0.4 * N →
  (P / 100) * I = 6 →
  P = 16 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l455_45575


namespace NUMINAMATH_CALUDE_range_of_a_l455_45503

/-- The set of real numbers x satisfying the condition p -/
def set_p (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- The set of real numbers x satisfying the condition q -/
def set_q : Set ℝ := {x | x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0}

/-- The theorem stating the range of values for a -/
theorem range_of_a (a : ℝ) : 
  (a < 0) → 
  (set_p a)ᶜ ⊂ (set_q)ᶜ → 
  (set_p a)ᶜ ≠ (set_q)ᶜ → 
  (-4 ≤ a ∧ a < 0) ∨ (a ≤ -4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l455_45503


namespace NUMINAMATH_CALUDE_monday_temperature_l455_45551

theorem monday_temperature
  (temp : Fin 5 → ℝ)
  (avg_mon_to_thu : (temp 0 + temp 1 + temp 2 + temp 3) / 4 = 48)
  (avg_tue_to_fri : (temp 1 + temp 2 + temp 3 + temp 4) / 4 = 40)
  (some_day_42 : ∃ i, temp i = 42)
  (friday_10 : temp 4 = 10) :
  temp 0 = 42 := by
sorry

end NUMINAMATH_CALUDE_monday_temperature_l455_45551


namespace NUMINAMATH_CALUDE_simple_interest_doubling_l455_45560

/-- The factor by which a sum of money increases under simple interest -/
def simple_interest_factor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

/-- Theorem: Given a simple interest rate of 25% per annum over 4 years, 
    the factor by which an initial sum of money increases is 2 -/
theorem simple_interest_doubling : 
  simple_interest_factor 0.25 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_doubling_l455_45560


namespace NUMINAMATH_CALUDE_largest_multiple_of_60_with_7_and_0_l455_45557

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def consists_of_7_and_0 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7 ∨ d = 0

theorem largest_multiple_of_60_with_7_and_0 :
  ∃ n : ℕ,
    is_multiple_of n 60 ∧
    consists_of_7_and_0 n ∧
    (∀ m : ℕ, m > n → ¬(is_multiple_of m 60 ∧ consists_of_7_and_0 m)) ∧
    n / 15 = 518 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_60_with_7_and_0_l455_45557


namespace NUMINAMATH_CALUDE_triangle_existence_l455_45588

theorem triangle_existence (n : ℕ) (h : n ≥ 2) : 
  ∃ (points : Finset (Fin (2*n))) (segments : Finset (Fin (2*n) × Fin (2*n))),
    Finset.card segments = n^2 + 1 →
    ∃ (a b c : Fin (2*n)), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (a, b) ∈ segments ∧ (b, c) ∈ segments ∧ (a, c) ∈ segments :=
by sorry


end NUMINAMATH_CALUDE_triangle_existence_l455_45588


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l455_45544

/-- The number of Oxygen atoms in a compound with given properties -/
def oxygenAtoms (molecularWeight : ℕ) (hydrogenAtoms carbonAtoms : ℕ) 
  (atomicWeightH atomicWeightC atomicWeightO : ℕ) : ℕ :=
  (molecularWeight - (hydrogenAtoms * atomicWeightH + carbonAtoms * atomicWeightC)) / atomicWeightO

/-- Theorem stating the number of Oxygen atoms in the compound -/
theorem compound_oxygen_atoms :
  oxygenAtoms 62 2 1 1 12 16 = 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l455_45544


namespace NUMINAMATH_CALUDE_line_slope_l455_45530

/-- The slope of the line x + √3y + 2 = 0 is -1/√3 -/
theorem line_slope (x y : ℝ) : x + Real.sqrt 3 * y + 2 = 0 → 
  (y - (-2/Real.sqrt 3)) / (x - 0) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l455_45530


namespace NUMINAMATH_CALUDE_rectangle_property_l455_45597

/-- Represents a complex number --/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- Represents a rectangle in the complex plane --/
structure ComplexRectangle where
  A : ComplexNumber
  B : ComplexNumber
  C : ComplexNumber
  D : ComplexNumber

/-- The theorem stating the properties of the given rectangle --/
theorem rectangle_property (rect : ComplexRectangle) :
  rect.A = ComplexNumber.mk 2 3 →
  rect.B = ComplexNumber.mk 3 2 →
  rect.C = ComplexNumber.mk (-2) (-3) →
  rect.D = ComplexNumber.mk (-3) (-2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_property_l455_45597


namespace NUMINAMATH_CALUDE_mikis_sandcastle_height_l455_45596

/-- The height of Miki's sandcastle given the height of her sister's sandcastle and the difference in height -/
theorem mikis_sandcastle_height 
  (sisters_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : sisters_height = 0.5)
  (h2 : height_difference = 0.3333333333333333) : 
  sisters_height + height_difference = 0.8333333333333333 :=
by sorry

end NUMINAMATH_CALUDE_mikis_sandcastle_height_l455_45596


namespace NUMINAMATH_CALUDE_repeating_digit_equality_l455_45520

/-- Represents a repeating digit number -/
def repeatingDigit (d : ℕ) (n : ℕ) : ℕ := d * (10^n - 1) / 9

/-- The main theorem -/
theorem repeating_digit_equality (x y z : ℕ) (h : x < 10 ∧ y < 10 ∧ z < 10) :
  (∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧
    (repeatingDigit x (2 * n₁) - repeatingDigit y n₁).sqrt = repeatingDigit z n₁ ∧
    (repeatingDigit x (2 * n₂) - repeatingDigit y n₂).sqrt = repeatingDigit z n₂) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 9 ∧ y = 8 ∧ z = 9)) ∧
  (∀ n : ℕ, (repeatingDigit x (2 * n) - repeatingDigit y n).sqrt = repeatingDigit z n) :=
sorry

end NUMINAMATH_CALUDE_repeating_digit_equality_l455_45520


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_100111011_base6_l455_45533

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Finds all divisors of a number -/
def divisors (n : ℕ) : List ℕ := sorry

theorem largest_prime_divisor_of_100111011_base6 :
  let n := base6ToBase10 100111011
  ∃ (d : ℕ), d ∈ divisors n ∧ isPrime d ∧ d = 181 ∧ ∀ (p : ℕ), p ∈ divisors n → isPrime p → p ≤ d :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_100111011_base6_l455_45533


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l455_45565

theorem arithmetic_sequence_sixth_term 
  (a : ℕ → ℚ)  -- a is a sequence of rational numbers
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence condition
  (h_first : a 1 = 3/8)  -- first term is 3/8
  (h_eleventh : a 11 = 5/6)  -- eleventh term is 5/6
  : a 6 = 29/48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l455_45565


namespace NUMINAMATH_CALUDE_lowest_dropped_score_l455_45594

theorem lowest_dropped_score (scores : Fin 4 → ℕ) 
  (avg_all : (scores 0 + scores 1 + scores 2 + scores 3) / 4 = 45)
  (avg_after_drop : ∃ i, (scores 0 + scores 1 + scores 2 + scores 3 - scores i) / 3 = 50) :
  ∃ i, scores i = 30 ∧ ∀ j, scores j ≥ scores i := by
  sorry

end NUMINAMATH_CALUDE_lowest_dropped_score_l455_45594


namespace NUMINAMATH_CALUDE_representatives_selection_l455_45554

/-- The number of ways to select representatives from a group of male and female students. -/
def select_representatives (num_male num_female num_total num_min_female : ℕ) : ℕ :=
  (Nat.choose num_female 2 * Nat.choose num_male 2) +
  (Nat.choose num_female 3 * Nat.choose num_male 1) +
  (Nat.choose num_female 4 * Nat.choose num_male 0)

/-- Theorem stating that selecting 4 representatives from 5 male and 4 female students,
    with at least 2 females, can be done in 81 ways. -/
theorem representatives_selection :
  select_representatives 5 4 4 2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_representatives_selection_l455_45554


namespace NUMINAMATH_CALUDE_area_fraction_on_7x7_grid_l455_45589

/-- Represents a square grid of points -/
structure PointGrid :=
  (size : ℕ)

/-- Represents a square on the grid -/
structure GridSquare :=
  (sideLength : ℕ)

/-- The larger square formed by the outer points of the grid -/
def outerSquare (grid : PointGrid) : GridSquare :=
  { sideLength := grid.size - 1 }

/-- The shaded square inside the grid -/
def innerSquare : GridSquare :=
  { sideLength := 2 }

/-- Calculate the area of a square -/
def area (square : GridSquare) : ℕ :=
  square.sideLength * square.sideLength

/-- The fraction of the outer square's area occupied by the inner square -/
def areaFraction (grid : PointGrid) : ℚ :=
  (area innerSquare : ℚ) / (area (outerSquare grid))

theorem area_fraction_on_7x7_grid :
  areaFraction { size := 7 } = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_fraction_on_7x7_grid_l455_45589


namespace NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_l455_45558

/-- Sum of digits of a three-digit number -/
def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- Theorem: Among any 18 consecutive three-digit numbers, there exists one divisible by its sum of digits -/
theorem exists_divisible_by_sum_of_digits (n : ℕ) (h : 100 ≤ n ∧ n ≤ 982) :
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % sumOfDigits k = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_l455_45558


namespace NUMINAMATH_CALUDE_prob_a_before_b_is_one_third_l455_45545

-- Define the set of people
inductive Person : Type
  | A
  | B
  | C

-- Define a duty arrangement as a list of people
def DutyArrangement := List Person

-- Define the set of all possible duty arrangements
def allArrangements : List DutyArrangement :=
  [[Person.A, Person.B, Person.C],
   [Person.A, Person.C, Person.B],
   [Person.C, Person.A, Person.B]]

-- Define a function to check if A is immediately before B in an arrangement
def isABeforeB (arrangement : DutyArrangement) : Bool :=
  match arrangement with
  | [Person.A, Person.B, _] => true
  | _ => false

-- Define the probability of A being immediately before B
def probABeforeB : ℚ :=
  (allArrangements.filter isABeforeB).length / allArrangements.length

-- Theorem statement
theorem prob_a_before_b_is_one_third :
  probABeforeB = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_a_before_b_is_one_third_l455_45545


namespace NUMINAMATH_CALUDE_arithmetic_computation_l455_45576

theorem arithmetic_computation : 12 + 4 * (2 * 3 - 8 + 1)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l455_45576


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l455_45512

theorem consecutive_squares_sum (x : ℤ) :
  (x + 1)^2 - x^2 = 199 → x^2 + (x + 1)^2 = 19801 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l455_45512


namespace NUMINAMATH_CALUDE_triangle_problem_l455_45581

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sin (t.A + t.C) = 8 * (Real.sin (t.B / 2))^2)
  (h2 : t.a + t.c = 6)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = 2) : 
  Real.cos t.B = 15/17 ∧ t.b = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l455_45581


namespace NUMINAMATH_CALUDE_f_properties_l455_45521

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi ∧
  (∀ (x y : ℝ), x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) →
    y ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l455_45521


namespace NUMINAMATH_CALUDE_circumradii_ratio_eq_side_ratio_l455_45599

/-- Represents the properties of an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  perimeter : ℝ
  area : ℝ
  circumradius : ℝ
  perimeter_eq : perimeter = 3 * side
  area_eq : area = (side^2 * Real.sqrt 3) / 4
  circumradius_eq : circumradius = (side * Real.sqrt 3) / 3

/-- Theorem stating the relationship between circumradii of two equilateral triangles -/
theorem circumradii_ratio_eq_side_ratio 
  (n m : ℝ) 
  (fore back : EquilateralTriangle) 
  (h_perimeter_ratio : fore.perimeter / back.perimeter = n / m)
  (h_area_ratio : fore.area / back.area = n / m) :
  fore.circumradius / back.circumradius = fore.side / back.side := by
  sorry

#check circumradii_ratio_eq_side_ratio

end NUMINAMATH_CALUDE_circumradii_ratio_eq_side_ratio_l455_45599


namespace NUMINAMATH_CALUDE_alexa_first_day_pages_l455_45566

/-- The number of pages Alexa read on the first day of reading a Nancy Drew mystery. -/
def pages_read_first_day (total_pages second_day_pages pages_left : ℕ) : ℕ :=
  total_pages - second_day_pages - pages_left

/-- Theorem stating that Alexa read 18 pages on the first day. -/
theorem alexa_first_day_pages :
  pages_read_first_day 95 58 19 = 18 := by
  sorry

end NUMINAMATH_CALUDE_alexa_first_day_pages_l455_45566


namespace NUMINAMATH_CALUDE_angus_token_count_l455_45502

/-- The number of tokens Elsa has -/
def elsa_tokens : ℕ := 60

/-- The value of each token in dollars -/
def token_value : ℕ := 4

/-- The difference in token value between Elsa and Angus in dollars -/
def token_value_difference : ℕ := 20

/-- The number of tokens Angus has -/
def angus_tokens : ℕ := elsa_tokens - (token_value_difference / token_value)

theorem angus_token_count : angus_tokens = 55 := by
  sorry

end NUMINAMATH_CALUDE_angus_token_count_l455_45502


namespace NUMINAMATH_CALUDE_self_checkout_increase_is_20_percent_l455_45548

/-- The percentage increase in complaints when the self-checkout is broken -/
def self_checkout_increase (normal_complaints : ℕ) (short_staffed_increase : ℚ) (total_complaints : ℕ) : ℚ :=
  let short_staffed_complaints := normal_complaints * (1 + short_staffed_increase)
  let daily_complaints_both := total_complaints / 3
  (daily_complaints_both - short_staffed_complaints) / short_staffed_complaints * 100

/-- Theorem stating that the percentage increase when self-checkout is broken is 20% -/
theorem self_checkout_increase_is_20_percent :
  self_checkout_increase 120 (1/3) 576 = 20 := by
  sorry

end NUMINAMATH_CALUDE_self_checkout_increase_is_20_percent_l455_45548


namespace NUMINAMATH_CALUDE_perpendicular_vectors_sum_l455_45519

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (2, -m)

theorem perpendicular_vectors_sum (m : ℝ) 
  (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  (3 * a.1 + 2 * (b m).1, 3 * a.2 + 2 * (b m).2) = (7, 4) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_sum_l455_45519


namespace NUMINAMATH_CALUDE_horner_method_v3_equals_55_l455_45582

def horner_polynomial (x : ℝ) : ℝ := 3*x^5 + 8*x^4 - 3*x^3 + 5*x^2 + 12*x - 6

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 8
  let v2 := v1 * x - 3
  v2 * x + 5

theorem horner_method_v3_equals_55 :
  horner_v3 2 = 55 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_equals_55_l455_45582


namespace NUMINAMATH_CALUDE_find_number_l455_45537

theorem find_number : ∃ x : ℝ, (0.8 * x - 20 = 60) ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l455_45537


namespace NUMINAMATH_CALUDE_hiker_speed_l455_45555

/-- Proves that given a cyclist traveling at 10 miles per hour who passes a hiker, 
    stops 5 minutes later, and waits 7.5 minutes for the hiker to catch up, 
    the hiker's constant speed is 50/7.5 miles per hour. -/
theorem hiker_speed (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) :
  cyclist_speed = 10 →
  cyclist_travel_time = 5 / 60 →
  hiker_catch_up_time = 7.5 / 60 →
  (cyclist_speed * cyclist_travel_time) / hiker_catch_up_time = 50 / 7.5 := by
  sorry

#eval (50 : ℚ) / 7.5

end NUMINAMATH_CALUDE_hiker_speed_l455_45555


namespace NUMINAMATH_CALUDE_range_of_function_l455_45538

theorem range_of_function (f : ℝ → ℝ) (h : ∀ x, f x ∈ Set.Icc (3/8) (4/9)) :
  ∀ x, f x + Real.sqrt (1 - 2 * f x) ∈ Set.Icc (7/9) (7/8) := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l455_45538


namespace NUMINAMATH_CALUDE_altitude_from_C_to_AB_l455_45534

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 5)
def B : ℝ × ℝ := (1, -2)
def C : ℝ × ℝ := (-6, 4)

-- Define the equation of the altitude
def altitude_equation (x y : ℝ) : Prop := 7 * x - 6 * y + 30 = 0

-- Theorem statement
theorem altitude_from_C_to_AB :
  ∀ x y : ℝ, altitude_equation x y ↔ 
  (x - C.1) * (B.1 - A.1) + (y - C.2) * (B.2 - A.2) = 0 ∧
  (x, y) ≠ C :=
sorry

end NUMINAMATH_CALUDE_altitude_from_C_to_AB_l455_45534


namespace NUMINAMATH_CALUDE_complex_power_difference_l455_45524

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + i)^20 - (1 - i)^20 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_power_difference_l455_45524


namespace NUMINAMATH_CALUDE_function_identity_proof_l455_45535

theorem function_identity_proof (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, (n - 1)^2 < f n * f (f n) ∧ f n * f (f n) < n^2 + n) : 
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_proof_l455_45535


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l455_45546

def M : Set ℝ := {x : ℝ | x^2 + 2*x = 0}
def N : Set ℝ := {x : ℝ | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l455_45546


namespace NUMINAMATH_CALUDE_complex_power_sum_l455_45500

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^2021 + 1/z^2021 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l455_45500


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l455_45587

/-- Two real numbers are inversely proportional if their product is constant. -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_ratio
  (x y : ℝ → ℝ)
  (h_inv_prop : InverselyProportional x y)
  (x₁ x₂ y₁ y₂ : ℝ)
  (h_x_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0)
  (h_y_nonzero : y₁ ≠ 0 ∧ y₂ ≠ 0)
  (h_x_ratio : x₁ / x₂ = 4 / 5)
  (h_y_corr : y₁ = y (x.invFun x₁) ∧ y₂ = y (x.invFun x₂)) :
  y₁ / y₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l455_45587


namespace NUMINAMATH_CALUDE_bank_account_transfer_l455_45550

/-- Represents a bank account transfer operation that doubles the amount in one account. -/
inductive Transfer : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop
| double12 : ∀ a b c, Transfer (a, b, c) (a + b, 0, c)
| double13 : ∀ a b c, Transfer (a, b, c) (a + c, b, 0)
| double21 : ∀ a b c, Transfer (a, b, c) (0, a + b, c)
| double23 : ∀ a b c, Transfer (a, b, c) (a, b + c, 0)
| double31 : ∀ a b c, Transfer (a, b, c) (0, b, a + c)
| double32 : ∀ a b c, Transfer (a, b, c) (a, 0, b + c)

/-- Represents a sequence of transfers. -/
def TransferSeq : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop :=
  Relation.ReflTransGen Transfer

theorem bank_account_transfer :
  (∀ a b c : ℕ, ∃ a' b' c', TransferSeq (a, b, c) (a', b', c') ∧ (a' = 0 ∨ b' = 0 ∨ c' = 0)) ∧
  (∃ a b c : ℕ, ∀ a' b' c', TransferSeq (a, b, c) (a', b', c') → ¬(a' = 0 ∧ b' = 0) ∧ ¬(a' = 0 ∧ c' = 0) ∧ ¬(b' = 0 ∧ c' = 0)) :=
by sorry

end NUMINAMATH_CALUDE_bank_account_transfer_l455_45550


namespace NUMINAMATH_CALUDE_simplify_expression_l455_45553

theorem simplify_expression (a : ℝ) : 2*a*(2*a^2 + a) - a^2 = 4*a^3 + a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l455_45553


namespace NUMINAMATH_CALUDE_cos_a_plus_beta_half_l455_45585

theorem cos_a_plus_beta_half (a β : ℝ) 
  (h1 : 0 < a ∧ a < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (a + π / 4) = 1 / 3)
  (h4 : Real.sin (π / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (a + β / 2) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_a_plus_beta_half_l455_45585


namespace NUMINAMATH_CALUDE_smallest_ending_number_l455_45586

/-- A function that returns the number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- A function that checks if a number has an even number of factors -/
def has_even_factors (n : ℕ+) : Prop :=
  Even (num_factors n)

/-- A function that counts the number of even integers with an even number of factors
    in the range from 1 to n (inclusive) -/
def count_even_with_even_factors (n : ℕ) : ℕ := sorry

theorem smallest_ending_number :
  ∀ k : ℕ, k < 14 → count_even_with_even_factors k < 5 ∧
  count_even_with_even_factors 14 ≥ 5 := by sorry

end NUMINAMATH_CALUDE_smallest_ending_number_l455_45586


namespace NUMINAMATH_CALUDE_female_democrat_ratio_is_half_l455_45507

/-- Represents the number of participants in a meeting with given conditions -/
structure Meeting where
  total : ℕ
  maleDemocratRatio : ℚ
  totalDemocratRatio : ℚ
  femaleDemocrats : ℕ
  male : ℕ
  female : ℕ

/-- The ratio of female democrats to total female participants -/
def femaleDemocratRatio (m : Meeting) : ℚ :=
  m.femaleDemocrats / m.female

theorem female_democrat_ratio_is_half (m : Meeting) 
  (h1 : m.total = 660)
  (h2 : m.maleDemocratRatio = 1/4)
  (h3 : m.totalDemocratRatio = 1/3)
  (h4 : m.femaleDemocrats = 110)
  (h5 : m.male + m.female = m.total)
  (h6 : m.maleDemocratRatio * m.male + m.femaleDemocrats = m.totalDemocratRatio * m.total) :
  femaleDemocratRatio m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_female_democrat_ratio_is_half_l455_45507


namespace NUMINAMATH_CALUDE_license_plate_count_l455_45508

def alphabet_size : ℕ := 26
def digit_count : ℕ := 10

def license_plate_combinations : ℕ :=
  -- Choose first repeated letter
  alphabet_size *
  -- Choose second repeated letter
  (alphabet_size - 1) *
  -- Choose two other unique letters
  (Nat.choose (alphabet_size - 2) 2) *
  -- Positions for first repeated letter
  (Nat.choose 6 2) *
  -- Positions for second repeated letter
  (Nat.choose 4 2) *
  -- Arrange two unique letters
  2 *
  -- Choose first digit
  digit_count *
  -- Choose second digit
  (digit_count - 1) *
  -- Choose third digit
  (digit_count - 2)

theorem license_plate_count :
  license_plate_combinations = 241164000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l455_45508


namespace NUMINAMATH_CALUDE_reading_time_calculation_l455_45531

def total_time : ℕ := 120
def piano_time : ℕ := 30
def writing_time : ℕ := 25
def exerciser_time : ℕ := 27

theorem reading_time_calculation :
  total_time - piano_time - writing_time - exerciser_time = 38 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l455_45531


namespace NUMINAMATH_CALUDE_outfits_count_l455_45571

theorem outfits_count (shirts : ℕ) (hats : ℕ) : shirts = 5 → hats = 3 → shirts * hats = 15 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l455_45571


namespace NUMINAMATH_CALUDE_ice_cream_cost_is_734_l455_45583

/-- The cost of Mrs. Hilt's ice cream purchase -/
def ice_cream_cost : ℚ :=
  let vanilla_price : ℚ := 99 / 100
  let chocolate_price : ℚ := 129 / 100
  let strawberry_price : ℚ := 149 / 100
  let vanilla_quantity : ℕ := 2
  let chocolate_quantity : ℕ := 3
  let strawberry_quantity : ℕ := 1
  vanilla_price * vanilla_quantity +
  chocolate_price * chocolate_quantity +
  strawberry_price * strawberry_quantity

theorem ice_cream_cost_is_734 : ice_cream_cost = 734 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_is_734_l455_45583


namespace NUMINAMATH_CALUDE_currency_multiplication_invalid_l455_45595

-- Define currency types
inductive Currency
| Ruble
| Kopeck

-- Define a structure for money
structure Money where
  amount : ℚ
  currency : Currency

-- Define conversion rate
def conversionRate : ℚ := 100

-- Define equality for Money
def Money.eq (a b : Money) : Prop :=
  (a.currency = b.currency ∧ a.amount = b.amount) ∨
  (a.currency = Currency.Ruble ∧ b.currency = Currency.Kopeck ∧ a.amount * conversionRate = b.amount) ∨
  (a.currency = Currency.Kopeck ∧ b.currency = Currency.Ruble ∧ a.amount = b.amount * conversionRate)

-- Define multiplication for Money (this operation is not well-defined for real currencies)
def Money.mul (a b : Money) : Money :=
  { amount := a.amount * b.amount,
    currency := 
      match a.currency, b.currency with
      | Currency.Ruble, Currency.Ruble => Currency.Ruble
      | Currency.Kopeck, Currency.Kopeck => Currency.Kopeck
      | _, _ => Currency.Ruble }

-- Theorem statement
theorem currency_multiplication_invalid :
  ∃ (a b c d : Money),
    Money.eq a b ∧ Money.eq c d ∧
    ¬(Money.eq (Money.mul a c) (Money.mul b d)) := by
  sorry

end NUMINAMATH_CALUDE_currency_multiplication_invalid_l455_45595


namespace NUMINAMATH_CALUDE_shirt_coat_ratio_l455_45584

/-- Given a shirt costing $150 and a total cost of $600 for the shirt and coat,
    prove that the ratio of the cost of the shirt to the cost of the coat is 1:3. -/
theorem shirt_coat_ratio (shirt_cost coat_cost total_cost : ℕ) : 
  shirt_cost = 150 → 
  total_cost = 600 → 
  total_cost = shirt_cost + coat_cost →
  (shirt_cost : ℚ) / coat_cost = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shirt_coat_ratio_l455_45584


namespace NUMINAMATH_CALUDE_parabola_through_point_standard_form_l455_45562

/-- A parabola is defined by its equation and the point it passes through. -/
structure Parabola where
  /-- The point that the parabola passes through -/
  point : ℝ × ℝ
  /-- The equation of the parabola, represented as a function -/
  equation : (ℝ × ℝ) → Prop

/-- The standard form of a parabola's equation -/
inductive StandardForm
  | VerticalAxis (p : ℝ) : StandardForm  -- y² = -2px
  | HorizontalAxis (p : ℝ) : StandardForm  -- x² = 2py

/-- Theorem: If a parabola passes through the point (-2, 3), then its standard equation
    must be either y² = -9/2x or x² = 4/3y -/
theorem parabola_through_point_standard_form (P : Parabola) 
    (h : P.point = (-2, 3)) :
    (∃ (sf : StandardForm), 
      (sf = StandardForm.VerticalAxis (9/4) ∨ 
       sf = StandardForm.HorizontalAxis (2/3)) ∧
      (∀ (x y : ℝ), P.equation (x, y) ↔ 
        (sf = StandardForm.VerticalAxis (9/4) → y^2 = -9/2 * x) ∧
        (sf = StandardForm.HorizontalAxis (2/3) → x^2 = 4/3 * y))) :=
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_standard_form_l455_45562


namespace NUMINAMATH_CALUDE_solve_for_k_l455_45552

theorem solve_for_k (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) →
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_solve_for_k_l455_45552


namespace NUMINAMATH_CALUDE_quadratic_sum_l455_45539

/-- A quadratic function g(x) = ax^2 + bx + c satisfying g(1) = 2 and g(2) = 3 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- Theorem: For a quadratic function g(x) = ax^2 + bx + c, if g(1) = 2 and g(2) = 3, then a + 2b + 3c = 7 -/
theorem quadratic_sum (a b c : ℝ) :
  (QuadraticFunction a b c 1 = 2) →
  (QuadraticFunction a b c 2 = 3) →
  a + 2 * b + 3 * c = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l455_45539


namespace NUMINAMATH_CALUDE_integers_less_than_four_abs_l455_45516

theorem integers_less_than_four_abs : 
  {n : ℤ | |n| < 4} = {-3, -2, -1, 0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_integers_less_than_four_abs_l455_45516


namespace NUMINAMATH_CALUDE_chess_positions_l455_45573

/-- The number of different positions on a chessboard after both players make one move each -/
def num_positions : ℕ :=
  let pawns_per_player := 8
  let knights_per_player := 2
  let pawn_moves := 2
  let knight_moves := 2
  let moves_per_player := pawns_per_player * pawn_moves + knights_per_player * knight_moves
  moves_per_player * moves_per_player

theorem chess_positions : num_positions = 400 := by
  sorry

end NUMINAMATH_CALUDE_chess_positions_l455_45573


namespace NUMINAMATH_CALUDE_unique_solution_l455_45517

def is_solution (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ Nat.gcd a b = 1 ∧ (a + 12) * b = 3 * a * (b + 12)

theorem unique_solution : ∀ a b : ℕ, is_solution a b ↔ a = 2 ∧ b = 9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l455_45517


namespace NUMINAMATH_CALUDE_tangent_line_m_value_l455_45532

/-- The curve function f(x) = x^3 + x - 1 -/
def f (x : ℝ) : ℝ := x^3 + x - 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_line_m_value :
  let p₁ : ℝ × ℝ := (1, f 1)
  let slope : ℝ := f' 1
  let p₂ : ℝ × ℝ := (2, m)
  (∀ m : ℝ, (m - p₁.2) = slope * (p₂.1 - p₁.1) → m = 5) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_m_value_l455_45532


namespace NUMINAMATH_CALUDE_aisha_mp3_song_count_l455_45527

/-- The number of songs on Aisha's mp3 player after a series of additions and removals -/
def final_song_count (initial : ℕ) (first_addition : ℕ) (removed : ℕ) : ℕ :=
  let after_first_addition := initial + first_addition
  let doubled := after_first_addition * 2
  let before_removal := after_first_addition + doubled
  before_removal - removed

/-- Theorem stating that given the initial conditions, the final number of songs is 2950 -/
theorem aisha_mp3_song_count :
  final_song_count 500 500 50 = 2950 := by
  sorry

end NUMINAMATH_CALUDE_aisha_mp3_song_count_l455_45527


namespace NUMINAMATH_CALUDE_sin_alpha_value_l455_45536

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 3) = 1 / 5) : 
  Real.sin α = (2 * Real.sqrt 6 - Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l455_45536


namespace NUMINAMATH_CALUDE_optimal_schedule_l455_45592

/-- Represents the construction teams -/
inductive Team
| A
| B

/-- Represents the construction schedule -/
structure Schedule where
  teamA_months : ℕ
  teamB_months : ℕ

/-- Calculates the total work done by a team given the months worked and their efficiency -/
def work_done (months : ℕ) (efficiency : ℚ) : ℚ :=
  months * efficiency

/-- Calculates the cost of a schedule given the monthly rates -/
def schedule_cost (s : Schedule) (rateA rateB : ℕ) : ℕ :=
  s.teamA_months * rateA + s.teamB_months * rateB

/-- Checks if a schedule is valid according to the given constraints -/
def is_valid_schedule (s : Schedule) : Prop :=
  s.teamA_months > 0 ∧ s.teamA_months ≤ 6 ∧
  s.teamB_months > 0 ∧ s.teamB_months ≤ 24 ∧
  s.teamA_months + s.teamB_months ≤ 24

/-- The main theorem to be proved -/
theorem optimal_schedule :
  ∃ (s : Schedule),
    is_valid_schedule s ∧
    work_done s.teamA_months (1 / 18) + work_done s.teamB_months (1 / 27) = 1 ∧
    ∀ (s' : Schedule),
      is_valid_schedule s' ∧
      work_done s'.teamA_months (1 / 18) + work_done s'.teamB_months (1 / 27) = 1 →
      schedule_cost s 80000 50000 ≤ schedule_cost s' 80000 50000 ∧
    s.teamA_months = 2 ∧ s.teamB_months = 24 :=
  sorry

end NUMINAMATH_CALUDE_optimal_schedule_l455_45592


namespace NUMINAMATH_CALUDE_F_max_value_l455_45580

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def f_derivative (x : ℝ) : ℝ := Real.cos x - Real.sin x

noncomputable def F (x : ℝ) : ℝ := f x * f_derivative x + f x ^ 2

theorem F_max_value :
  ∃ (M : ℝ), (∀ (x : ℝ), F x ≤ M) ∧ (∃ (x : ℝ), F x = M) ∧ M = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_F_max_value_l455_45580


namespace NUMINAMATH_CALUDE_amy_video_files_l455_45578

/-- Represents the number of video files Amy had initially -/
def initial_video_files : ℕ := 36

theorem amy_video_files :
  let initial_music_files : ℕ := 26
  let deleted_files : ℕ := 48
  let remaining_files : ℕ := 14
  initial_video_files + initial_music_files - deleted_files = remaining_files :=
by sorry

end NUMINAMATH_CALUDE_amy_video_files_l455_45578


namespace NUMINAMATH_CALUDE_f_properties_l455_45598

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := sin x + m * x

theorem f_properties (m : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ (deriv (f m)) x₁ = (deriv (f m)) x₂) ∧
  (∃ s : ℕ → ℝ, ∀ i j, i ≠ j → (deriv (f m)) (s i) = (deriv (f m)) (s j)) ∧
  (∃ t : ℕ → ℝ, ∀ i j, (deriv (f m)) (t i) = (deriv (f m)) (t j)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l455_45598


namespace NUMINAMATH_CALUDE_right_triangle_roots_l455_45525

theorem right_triangle_roots (a b z₁ z₂ : ℂ) : 
  (z₁^2 + a*z₁ + b = 0) →
  (z₂^2 + a*z₂ + b = 0) →
  (z₂ = Complex.I * z₁) →
  a^2 / b = 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_roots_l455_45525


namespace NUMINAMATH_CALUDE_cookie_milk_calculation_l455_45543

/-- Given that 12 cookies require 2 quarts of milk and 1 quart equals 2 pints,
    prove that 3 cookies require 1 pint of milk. -/
theorem cookie_milk_calculation 
  (cookies_per_recipe : ℕ := 12)
  (quarts_per_recipe : ℚ := 2)
  (pints_per_quart : ℕ := 2)
  (target_cookies : ℕ := 3) :
  let pints_per_recipe := quarts_per_recipe * pints_per_quart
  let pints_per_cookie := pints_per_recipe / cookies_per_recipe
  target_cookies * pints_per_cookie = 1 := by
sorry

end NUMINAMATH_CALUDE_cookie_milk_calculation_l455_45543


namespace NUMINAMATH_CALUDE_one_third_1206_is_100_5_percent_of_400_l455_45563

theorem one_third_1206_is_100_5_percent_of_400 :
  (1206 / 3) / 400 = 1.005 := by
  sorry

end NUMINAMATH_CALUDE_one_third_1206_is_100_5_percent_of_400_l455_45563


namespace NUMINAMATH_CALUDE_sue_driving_days_l455_45504

theorem sue_driving_days (total_cost : ℚ) (sister_days : ℚ) (sue_payment : ℚ) :
  total_cost = 2100 →
  sister_days = 4 →
  sue_payment = 900 →
  ∃ (sue_days : ℚ), sue_days + sister_days = 7 ∧ sue_days / (7 - sue_days) = sue_payment / (total_cost - sue_payment) ∧ sue_days = 3 :=
by sorry

end NUMINAMATH_CALUDE_sue_driving_days_l455_45504


namespace NUMINAMATH_CALUDE_fred_balloons_l455_45572

theorem fred_balloons (initial_balloons given_balloons : ℕ) 
  (h1 : initial_balloons = 709)
  (h2 : given_balloons = 221) :
  initial_balloons - given_balloons = 488 :=
by sorry

end NUMINAMATH_CALUDE_fred_balloons_l455_45572


namespace NUMINAMATH_CALUDE_complex_equation_solution_l455_45568

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l455_45568


namespace NUMINAMATH_CALUDE_older_sibling_age_l455_45513

/-- Given two siblings with a two-year age gap and their combined age, 
    prove the older sibling's age -/
theorem older_sibling_age 
  (h : ℕ) -- Hyeongjun's age
  (s : ℕ) -- Older sister's age
  (age_gap : s = h + 2) -- Two-year age gap condition
  (total_age : h + s = 26) -- Sum of ages condition
  : s = 14 := by
  sorry

end NUMINAMATH_CALUDE_older_sibling_age_l455_45513


namespace NUMINAMATH_CALUDE_sam_final_marbles_l455_45547

/-- Represents the number of marbles each person has -/
structure Marbles where
  steve : ℕ
  sam : ℕ
  sally : ℕ

/-- Represents the initial distribution of marbles -/
def initial_marbles (steve_marbles : ℕ) : Marbles :=
  { steve := steve_marbles,
    sam := 2 * steve_marbles,
    sally := 2 * steve_marbles - 5 }

/-- Represents the distribution of marbles after the exchange -/
def final_marbles (m : Marbles) : Marbles :=
  { steve := m.steve + 3,
    sam := m.sam - 6,
    sally := m.sally + 3 }

/-- Theorem stating that Sam ends up with 8 marbles -/
theorem sam_final_marbles :
  ∀ (initial : Marbles),
    initial.sam = 2 * initial.steve →
    initial.sally = initial.sam - 5 →
    (final_marbles initial).steve = 10 →
    (final_marbles initial).sam = 8 :=
by sorry

end NUMINAMATH_CALUDE_sam_final_marbles_l455_45547


namespace NUMINAMATH_CALUDE_square_number_problem_l455_45577

theorem square_number_problem : ∃ x : ℤ, 
  (∃ m : ℤ, x + 15 = m^2) ∧ 
  (∃ n : ℤ, x - 74 = n^2) ∧ 
  x = 2010 := by
  sorry

end NUMINAMATH_CALUDE_square_number_problem_l455_45577
