import Mathlib

namespace NUMINAMATH_CALUDE_car_distance_traveled_l362_36276

theorem car_distance_traveled (time : ℝ) (speed : ℝ) (distance : ℝ) : 
  time = 11 → speed = 65 → distance = time * speed → distance = 715 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l362_36276


namespace NUMINAMATH_CALUDE_hyperbola_sum_l362_36274

/-- Represents a hyperbola with center (h, k), focus (h + c, k), and vertex (h - a, k) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  a : ℝ
  c : ℝ
  vertex_x : ℝ
  focus_x : ℝ
  h_pos : 0 < a
  h_c_gt_a : c > a
  h_vertex : vertex_x = h - a
  h_focus : focus_x = h + c

/-- The theorem stating the sum of h, k, a, and b for the given hyperbola -/
theorem hyperbola_sum (H : Hyperbola) (h_center : H.h = 1 ∧ H.k = -1)
    (h_vertex : H.vertex_x = -2) (h_focus : H.focus_x = 1 + Real.sqrt 41) :
    H.h + H.k + H.a + Real.sqrt (H.c^2 - H.a^2) = 3 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l362_36274


namespace NUMINAMATH_CALUDE_police_officer_ratio_l362_36208

/-- Proves that the ratio of female officers to male officers on duty is 1:1 -/
theorem police_officer_ratio (total_on_duty : ℕ) (female_officers : ℕ) (female_on_duty_percent : ℚ) :
  total_on_duty = 204 →
  female_officers = 600 →
  female_on_duty_percent = 17 / 100 →
  ∃ (female_on_duty male_on_duty : ℕ),
    female_on_duty = female_on_duty_percent * female_officers ∧
    male_on_duty = total_on_duty - female_on_duty ∧
    female_on_duty = male_on_duty :=
by sorry

end NUMINAMATH_CALUDE_police_officer_ratio_l362_36208


namespace NUMINAMATH_CALUDE_regression_line_estimate_l362_36229

/-- Represents a linear regression line y = bx + a -/
structure RegressionLine where
  b : ℝ  -- slope
  a : ℝ  -- y-intercept

/-- Calculates the y-value for a given x on the regression line -/
def RegressionLine.yValue (line : RegressionLine) (x : ℝ) : ℝ :=
  line.b * x + line.a

theorem regression_line_estimate :
  ∀ (line : RegressionLine),
    line.b = 1.23 →
    line.yValue 4 = 5 →
    line.yValue 2 = 2.54 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_estimate_l362_36229


namespace NUMINAMATH_CALUDE_translation_correctness_given_translations_correct_l362_36264

/-- Represents a word in either Russian or Kurdish -/
structure Word :=
  (value : String)

/-- Represents a sentence in either Russian or Kurdish -/
structure Sentence :=
  (words : List Word)

/-- Defines the rules for Kurdish sentence structure -/
def kurdishSentenceStructure (s : Sentence) : Prop :=
  -- The predicate is at the end of the sentence
  -- The subject starts the sentence, followed by the object
  -- Noun-adjective constructions follow the "S (adjective determinant) O (determined word)" structure
  -- The determined word takes the suffix "e"
  sorry

/-- Translates a Russian sentence to Kurdish -/
def translateToKurdish (russianSentence : Sentence) : Sentence :=
  sorry

/-- Verifies that the translated sentence follows Kurdish sentence structure -/
theorem translation_correctness (russianSentence : Sentence) :
  kurdishSentenceStructure (translateToKurdish russianSentence) :=
  sorry

/-- Specific sentences from the problem -/
def sentence1 : Sentence := sorry -- "The lazy lion eats meat"
def sentence2 : Sentence := sorry -- "The healthy poor man carries the burden"
def sentence3 : Sentence := sorry -- "The bull of the poor man does not understand the poor man"

/-- Verifies the correctness of the given translations -/
theorem given_translations_correct :
  kurdishSentenceStructure (translateToKurdish sentence1) ∧
  kurdishSentenceStructure (translateToKurdish sentence2) ∧
  kurdishSentenceStructure (translateToKurdish sentence3) :=
  sorry

end NUMINAMATH_CALUDE_translation_correctness_given_translations_correct_l362_36264


namespace NUMINAMATH_CALUDE_football_player_average_increase_l362_36297

theorem football_player_average_increase (goals_fifth_match : ℕ) (total_goals : ℕ) :
  goals_fifth_match = 2 →
  total_goals = 8 →
  (total_goals / 5 : ℚ) - ((total_goals - goals_fifth_match) / 4 : ℚ) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_football_player_average_increase_l362_36297


namespace NUMINAMATH_CALUDE_circles_intersection_m_range_l362_36289

/-- Circle C₁ with equation x² + y² - 2mx + m² - 4 = 0 -/
def C₁ (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*m*p.1 + m^2 - 4 = 0}

/-- Circle C₂ with equation x² + y² + 2x - 4my + 4m² - 8 = 0 -/
def C₂ (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 4*m*p.2 + 4*m^2 - 8 = 0}

/-- The theorem stating that if C₁ and C₂ intersect, then m is in the specified range -/
theorem circles_intersection_m_range (m : ℝ) :
  (C₁ m ∩ C₂ m).Nonempty →
  m ∈ Set.Ioo (-12/5) (-2/5) ∪ Set.Ioo (3/5) 2 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersection_m_range_l362_36289


namespace NUMINAMATH_CALUDE_coordinate_plane_conditions_l362_36265

-- Define a point on the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions and their corresponding geometric interpretations
theorem coordinate_plane_conditions (p : Point) :
  (p.x = 3 → p ∈ {q : Point | q.x = 3}) ∧
  (p.x < 3 → p ∈ {q : Point | q.x < 3}) ∧
  (p.x > 3 → p ∈ {q : Point | q.x > 3}) ∧
  (p.y = 2 → p ∈ {q : Point | q.y = 2}) ∧
  (p.y > 2 → p ∈ {q : Point | q.y > 2}) := by
  sorry

end NUMINAMATH_CALUDE_coordinate_plane_conditions_l362_36265


namespace NUMINAMATH_CALUDE_min_value_problem_l362_36220

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b - a - 2 * b = 0) :
  ∃ (min : ℝ), min = 7 ∧ 
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y - x - 2 * y = 0 → 
    x^2 / 4 - 2 / x + y^2 - 1 / y ≥ min) ∧
  (a^2 / 4 - 2 / a + b^2 - 1 / b = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l362_36220


namespace NUMINAMATH_CALUDE_erased_number_proof_l362_36242

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  n ≥ 3 →
  x ≥ 3 →
  x ≤ n →
  (n * (n + 1) / 2 - 3 - x) / (n - 2 : ℚ) = 151 / 3 →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l362_36242


namespace NUMINAMATH_CALUDE_measure_nine_kg_from_twentyfour_l362_36294

/-- Represents a pile of nails with a given weight in kg. -/
structure NailPile :=
  (weight : ℚ)

/-- Represents the state of our nails, divided into at most four piles. -/
structure NailState :=
  (pile1 : NailPile)
  (pile2 : Option NailPile)
  (pile3 : Option NailPile)
  (pile4 : Option NailPile)

/-- Divides a pile into two equal piles. -/
def dividePile (p : NailPile) : NailPile × NailPile :=
  (⟨p.weight / 2⟩, ⟨p.weight / 2⟩)

/-- Combines two piles into one. -/
def combinePiles (p1 p2 : NailPile) : NailPile :=
  ⟨p1.weight + p2.weight⟩

/-- The theorem stating that we can measure out 9 kg from 24 kg using only division. -/
theorem measure_nine_kg_from_twentyfour :
  ∃ (final : NailState),
    (final.pile1.weight = 9 ∨ 
     (∃ p, final.pile2 = some p ∧ p.weight = 9) ∨
     (∃ p, final.pile3 = some p ∧ p.weight = 9) ∨
     (∃ p, final.pile4 = some p ∧ p.weight = 9)) ∧
    final.pile1.weight + 
    (final.pile2.map (λ p => p.weight) |>.getD 0) +
    (final.pile3.map (λ p => p.weight) |>.getD 0) +
    (final.pile4.map (λ p => p.weight) |>.getD 0) = 24 :=
sorry

end NUMINAMATH_CALUDE_measure_nine_kg_from_twentyfour_l362_36294


namespace NUMINAMATH_CALUDE_investment_return_rate_l362_36262

theorem investment_return_rate 
  (total_investment : ℝ) 
  (total_interest : ℝ) 
  (known_rate : ℝ) 
  (known_investment : ℝ) 
  (h1 : total_investment = 33000)
  (h2 : total_interest = 970)
  (h3 : known_rate = 0.0225)
  (h4 : known_investment = 13000)
  : ∃ r : ℝ, 
    r * known_investment + known_rate * (total_investment - known_investment) = total_interest ∧ 
    r = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_investment_return_rate_l362_36262


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l362_36299

/-- The line equation ax + y + a + 1 = 0 always passes through the point (-1, -1) for all values of a. -/
theorem line_passes_through_fixed_point (a : ℝ) : a * (-1) + (-1) + a + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l362_36299


namespace NUMINAMATH_CALUDE_rectangle_to_square_l362_36272

/-- A rectangle can be divided into two parts that form a square -/
theorem rectangle_to_square (length width : ℝ) (h1 : length = 9) (h2 : width = 4) :
  ∃ (side : ℝ), side^2 = length * width ∧ side = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l362_36272


namespace NUMINAMATH_CALUDE_independence_test_smoking_lung_disease_l362_36293

-- Define the variables and constants
variable (K : ℝ)
variable (confidence_level : ℝ)
variable (error_rate : ℝ)

-- Define the relationship between smoking and lung disease
def smoking_related_to_lung_disease : Prop := sorry

-- Define the critical value for K^2
def critical_value : ℝ := 6.635

-- Define the theorem
theorem independence_test_smoking_lung_disease :
  K ≥ critical_value →
  confidence_level = 0.99 →
  error_rate = 1 - confidence_level →
  smoking_related_to_lung_disease ∧
  (smoking_related_to_lung_disease → error_rate = 0.01) :=
by sorry

end NUMINAMATH_CALUDE_independence_test_smoking_lung_disease_l362_36293


namespace NUMINAMATH_CALUDE_percentage_calculation_l362_36281

theorem percentage_calculation : 
  (0.47 * 1442 - 0.36 * 1412) + 63 = 232.42 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l362_36281


namespace NUMINAMATH_CALUDE_complex_symmetry_ratio_imag_part_l362_36231

theorem complex_symmetry_ratio_imag_part (z₁ z₂ : ℂ) :
  z₁ = 1 - 2*I →
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) →
  (z₂ / z₁).im = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_symmetry_ratio_imag_part_l362_36231


namespace NUMINAMATH_CALUDE_water_width_after_drop_l362_36271

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -2*y

-- Define the point that the parabola passes through
def parabola_point : ℝ × ℝ := (2, -2)

-- Theorem to prove
theorem water_width_after_drop :
  parabola parabola_point.1 parabola_point.2 →
  ∀ x : ℝ, parabola x (-3) → 2 * |x| = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_water_width_after_drop_l362_36271


namespace NUMINAMATH_CALUDE_geometric_mean_point_existence_l362_36210

/-- In a triangle ABC, point D on BC exists such that AD is the geometric mean of BD and DC
    if and only if b + c ≤ a√2, where a = BC, b = AC, and c = AB. -/
theorem geometric_mean_point_existence (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) :
  (∃ (t : ℝ), 0 < t ∧ t < a ∧ 
    (b^2 * t * (a - t) = a * (a - t) * t)) ↔ b + c ≤ a * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_point_existence_l362_36210


namespace NUMINAMATH_CALUDE_orchid_painting_time_l362_36221

/-- The time it takes Ellen to paint various flowers and vines -/
structure PaintingTimes where
  lily : ℕ
  rose : ℕ
  vine : ℕ
  total : ℕ
  lilies : ℕ
  roses : ℕ
  orchids : ℕ
  vines : ℕ

/-- Theorem stating that the time to paint an orchid is 3 minutes -/
theorem orchid_painting_time (pt : PaintingTimes)
  (h1 : pt.lily = 5)
  (h2 : pt.rose = 7)
  (h3 : pt.vine = 2)
  (h4 : pt.total = 213)
  (h5 : pt.lilies = 17)
  (h6 : pt.roses = 10)
  (h7 : pt.orchids = 6)
  (h8 : pt.vines = 20) :
  (pt.total - (pt.lily * pt.lilies + pt.rose * pt.roses + pt.vine * pt.vines)) / pt.orchids = 3 :=
by sorry

end NUMINAMATH_CALUDE_orchid_painting_time_l362_36221


namespace NUMINAMATH_CALUDE_least_common_denominator_l362_36251

theorem least_common_denominator : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l362_36251


namespace NUMINAMATH_CALUDE_jensens_inequality_l362_36248

/-- Jensen's inequality for convex functions -/
theorem jensens_inequality (f : ℝ → ℝ) (hf : ConvexOn ℝ Set.univ f) 
  (x₁ x₂ q₁ q₂ : ℝ) (hq₁ : q₁ > 0) (hq₂ : q₂ > 0) (hsum : q₁ + q₂ = 1) :
  f (q₁ * x₁ + q₂ * x₂) ≤ q₁ * f x₁ + q₂ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_jensens_inequality_l362_36248


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l362_36236

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l362_36236


namespace NUMINAMATH_CALUDE_find_other_number_l362_36280

theorem find_other_number (a b : ℤ) (h1 : 3 * a + 2 * b = 120) (h2 : a = 26 ∨ b = 26) : 
  (a ≠ 26 → a = 21) ∧ (b ≠ 26 → b = 21) :=
sorry

end NUMINAMATH_CALUDE_find_other_number_l362_36280


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l362_36287

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l362_36287


namespace NUMINAMATH_CALUDE_toms_robot_collection_l362_36215

/-- Represents the number of robots of each type for a person -/
structure RobotCollection where
  animal : ℕ
  humanoid : ℕ
  vehicle : ℕ

/-- Given the conditions of the problem, prove that Tom's robot collection matches the expected values -/
theorem toms_robot_collection (michael : RobotCollection) (tom : RobotCollection) : 
  michael.animal = 8 ∧ 
  michael.humanoid = 12 ∧ 
  michael.vehicle = 20 ∧
  tom.animal = 2 * michael.animal ∧
  tom.humanoid = (3 : ℕ) / 2 * michael.humanoid ∧
  michael.vehicle = (5 : ℕ) / 4 * tom.vehicle →
  tom.animal = 16 ∧ tom.humanoid = 18 ∧ tom.vehicle = 16 := by
  sorry

end NUMINAMATH_CALUDE_toms_robot_collection_l362_36215


namespace NUMINAMATH_CALUDE_total_weight_loss_l362_36225

theorem total_weight_loss (first_person_loss second_person_loss third_person_loss fourth_person_loss : ℕ) :
  first_person_loss = 27 →
  second_person_loss = first_person_loss - 7 →
  third_person_loss = 28 →
  fourth_person_loss = 28 →
  first_person_loss + second_person_loss + third_person_loss + fourth_person_loss = 103 :=
by sorry

end NUMINAMATH_CALUDE_total_weight_loss_l362_36225


namespace NUMINAMATH_CALUDE_flu_infection_rate_l362_36249

theorem flu_infection_rate : 
  ∀ (x : ℝ), 
  (1 : ℝ) + x + x * ((1 : ℝ) + x) = 144 → 
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_flu_infection_rate_l362_36249


namespace NUMINAMATH_CALUDE_percentage_decrease_of_b_l362_36273

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- a and b are positive
  a / b = 4 / 5 ∧  -- ratio of a to b is 4 to 5
  x = 1.25 * a ∧  -- x equals a increased by 25 percent of a
  m = b * (1 - p / 100) ∧  -- m equals b decreased by p percent of b
  m / x = 0.4  -- m / x is 0.4
  → p = 60 :=  -- The percentage decrease of b to get m is 60%
by sorry

end NUMINAMATH_CALUDE_percentage_decrease_of_b_l362_36273


namespace NUMINAMATH_CALUDE_sequence_sum_l362_36219

theorem sequence_sum (P Q R S T U V : ℝ) : 
  R = 7 ∧
  P + Q + R = 36 ∧
  Q + R + S = 36 ∧
  R + S + T = 36 ∧
  S + T + U = 36 ∧
  T + U + V = 36 →
  P + V = 29 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l362_36219


namespace NUMINAMATH_CALUDE_scheduled_halt_duration_l362_36234

def average_speed : ℝ := 87
def total_distance : ℝ := 348
def scheduled_start_time : ℝ := 9
def scheduled_end_time : ℝ := 13.75  -- 1:45 PM in decimal hours

theorem scheduled_halt_duration :
  let travel_time_without_halt := total_distance / average_speed
  let scheduled_travel_time := scheduled_end_time - scheduled_start_time
  scheduled_travel_time - travel_time_without_halt = 0.75 := by sorry

end NUMINAMATH_CALUDE_scheduled_halt_duration_l362_36234


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l362_36256

universe u

def M : Set ℤ := {-1, 0, 1, 2, 3, 4}
def A : Set ℤ := {2, 3}
def AUnionB : Set ℤ := {1, 2, 3, 4}

theorem intersection_complement_theorem :
  ∃ B : Set ℤ, A ∪ B = AUnionB ∧ B ∩ (M \ A) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l362_36256


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l362_36268

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 2 ∧ (1 / (x - 2) + (1 - x) / (2 - x) = 3) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l362_36268


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l362_36238

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := -25 * i / (3 + 4 * i)
  Complex.im z = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l362_36238


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l362_36258

theorem complex_modulus_problem (z : ℂ) (h : (8 + 6*I)*z = 5 + 12*I) : 
  Complex.abs z = 13/10 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l362_36258


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l362_36252

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationWindow (cycle : TrafficLightCycle) : ℕ :=
  3 * 5  -- 5 seconds before each color change, and there are 3 changes

/-- Theorem: The probability of observing a color change in a 5-second interval
    for the given traffic light cycle is 3/20 -/
theorem traffic_light_change_probability 
  (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 50) 
  (h2 : cycle.yellow = 5) 
  (h3 : cycle.red = 45) :
  (changeObservationWindow cycle : ℚ) / (cycleDuration cycle) = 3 / 20 := by
  sorry


end NUMINAMATH_CALUDE_traffic_light_change_probability_l362_36252


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l362_36205

theorem arithmetic_sequence_problem (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 4 - a 2 = -2 →                                      -- given condition
  a 7 = -3 →                                            -- given condition
  a 9 = -5 := by                                        -- conclusion to prove
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l362_36205


namespace NUMINAMATH_CALUDE_sweet_shop_inventory_l362_36223

/-- The Sweet Shop inventory problem -/
theorem sweet_shop_inventory (total_cases : ℕ) (chocolate_cases : ℕ) (lollipop_cases : ℕ) :
  total_cases = 80 →
  chocolate_cases = 25 →
  lollipop_cases = total_cases - chocolate_cases →
  lollipop_cases = 55 := by
  sorry

#check sweet_shop_inventory

end NUMINAMATH_CALUDE_sweet_shop_inventory_l362_36223


namespace NUMINAMATH_CALUDE_algebra_test_male_students_l362_36204

theorem algebra_test_male_students (M : ℕ) : 
  (90 * (M + 32) = 82 * M + 92 * 32) → M = 8 := by
sorry

end NUMINAMATH_CALUDE_algebra_test_male_students_l362_36204


namespace NUMINAMATH_CALUDE_maple_grove_elementary_difference_l362_36230

theorem maple_grove_elementary_difference : 
  let classrooms : ℕ := 5
  let students_per_classroom : ℕ := 22
  let hamsters_per_classroom : ℕ := 3
  let total_students : ℕ := classrooms * students_per_classroom
  let total_hamsters : ℕ := classrooms * hamsters_per_classroom
  total_students - total_hamsters = 95 := by
  sorry

end NUMINAMATH_CALUDE_maple_grove_elementary_difference_l362_36230


namespace NUMINAMATH_CALUDE_fifteen_equation_system_solution_l362_36286

theorem fifteen_equation_system_solution (x : Fin 15 → ℝ) :
  (∀ i : Fin 14, 1 - x i * x (i + 1) = 0) ∧
  (1 - x 15 * x 1 = 0) →
  (∀ i : Fin 15, x i = 1) ∨ (∀ i : Fin 15, x i = -1) := by
  sorry

end NUMINAMATH_CALUDE_fifteen_equation_system_solution_l362_36286


namespace NUMINAMATH_CALUDE_shorter_diagonal_of_parallelepiped_l362_36296

/-- Represents a parallelepiped with a rhombus base -/
structure Parallelepiped where
  base_side : ℝ
  lateral_edge : ℝ
  lateral_angle : ℝ
  section_area : ℝ

/-- Theorem: The shorter diagonal of the base rhombus in the given parallelepiped is 60 -/
theorem shorter_diagonal_of_parallelepiped (p : Parallelepiped) 
  (h1 : p.base_side = 60)
  (h2 : p.lateral_edge = 80)
  (h3 : p.lateral_angle = Real.pi / 3)  -- 60 degrees in radians
  (h4 : p.section_area = 7200) :
  ∃ (shorter_diagonal : ℝ), shorter_diagonal = 60 :=
by sorry

end NUMINAMATH_CALUDE_shorter_diagonal_of_parallelepiped_l362_36296


namespace NUMINAMATH_CALUDE_spoon_fork_sale_price_comparison_l362_36270

theorem spoon_fork_sale_price_comparison :
  ∃ (initial_price : ℕ),
    initial_price % 10 = 0 ∧
    initial_price > 100 ∧
    initial_price - 100 < initial_price / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_spoon_fork_sale_price_comparison_l362_36270


namespace NUMINAMATH_CALUDE_smallest_result_is_zero_l362_36227

def S : Set ℕ := {2, 4, 6, 8, 10, 12}

def operation (a b c : ℕ) : ℕ := ((a + b - c) * c)

theorem smallest_result_is_zero :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  operation a b c = 0 ∧
  ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  operation x y z ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_result_is_zero_l362_36227


namespace NUMINAMATH_CALUDE_smallest_of_three_l362_36218

theorem smallest_of_three : 
  ∀ (x y z : ℝ), x = -Real.sqrt 2 ∧ y = 0 ∧ z = -1 → 
  x < y ∧ x < z := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_l362_36218


namespace NUMINAMATH_CALUDE_line_properties_l362_36228

-- Define the lines l₁ and l₂
def l₁ (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1) + 2
def l₂ (k : ℝ) (x y : ℝ) : Prop := 3 * x - (k - 2) * y + 5 = 0

-- Define point P
def P : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem line_properties :
  ∀ k : ℝ,
  (∀ x y : ℝ, l₁ k x y → (x, y) = P) ∧
  (∀ x y : ℝ, l₁ k x y ∧ l₂ k x y → k = -1) ∧
  (k = -1 →
    let d := (4 * Real.sqrt 2) / 3
    ∀ x₁ y₁ x₂ y₂ : ℝ,
    l₁ k x₁ y₁ → l₂ k x₂ y₂ →
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = d) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l362_36228


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l362_36235

theorem exponential_equation_solution :
  ∃ x : ℝ, (16 : ℝ) ^ x * (16 : ℝ) ^ x * (16 : ℝ) ^ x = (256 : ℝ) ^ 3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l362_36235


namespace NUMINAMATH_CALUDE_existence_of_set_B_l362_36263

theorem existence_of_set_B : ∃ (a : ℝ), 
  let A : Set ℝ := {1, 3, a^2 + 3*a - 4}
  let B : Set ℝ := {0, 6, a^2 + 4*a - 2, a + 3}
  (A ∩ B = {3}) ∧ (a = 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_set_B_l362_36263


namespace NUMINAMATH_CALUDE_angle_theorem_l362_36260

theorem angle_theorem (α β θ : Real) 
  (h1 : 0 < α ∧ α < 60)
  (h2 : 0 < β ∧ β < 60)
  (h3 : 0 < θ ∧ θ < 60)
  (h4 : α + β = 2 * θ)
  (h5 : Real.sin α * Real.sin β * Real.sin θ = 
        Real.sin (60 - α) * Real.sin (60 - β) * Real.sin (60 - θ)) :
  θ = 30 := by sorry

end NUMINAMATH_CALUDE_angle_theorem_l362_36260


namespace NUMINAMATH_CALUDE_function_form_l362_36298

theorem function_form (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x + Real.cos x ^ 2| ≤ 3/4)
  (h2 : ∀ x, |f x - Real.sin x ^ 2| ≤ 1/4) :
  ∀ x, f x = Real.sin x ^ 2 - 1/4 := by
  sorry

end NUMINAMATH_CALUDE_function_form_l362_36298


namespace NUMINAMATH_CALUDE_media_group_arrangement_count_l362_36283

/-- Represents the number of domestic media groups -/
def domestic_groups : ℕ := 6

/-- Represents the number of foreign media groups -/
def foreign_groups : ℕ := 3

/-- Represents the total number of media groups to be selected -/
def selected_groups : ℕ := 4

/-- Calculates the number of ways to select and arrange media groups -/
def media_group_arrangements (d : ℕ) (f : ℕ) (s : ℕ) : ℕ :=
  -- Implementation details are omitted as per the instructions
  sorry

/-- Theorem stating that the number of valid arrangements is 684 -/
theorem media_group_arrangement_count :
  media_group_arrangements domestic_groups foreign_groups selected_groups = 684 := by
  sorry

end NUMINAMATH_CALUDE_media_group_arrangement_count_l362_36283


namespace NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l362_36203

theorem common_root_of_quadratic_equations (p q : ℝ) (x : ℝ) :
  (2017 * x^2 + p * x + q = 0) ∧ 
  (p * x^2 + q * x + 2017 = 0) →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l362_36203


namespace NUMINAMATH_CALUDE_range_of_a_l362_36285

-- Define the complex number z
def z (a : ℝ) : ℂ := (2 + Complex.I) * (a + 2 * Complex.I^3)

-- Define the condition for z to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (in_fourth_quadrant (z a)) ↔ (-1 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l362_36285


namespace NUMINAMATH_CALUDE_line_vector_proof_l362_36224

def line_vector (t : ℝ) : ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 0 = (2, 3)) →
  (line_vector 5 = (12, -37)) →
  (line_vector (-3) = (-4, 27)) :=
by sorry

end NUMINAMATH_CALUDE_line_vector_proof_l362_36224


namespace NUMINAMATH_CALUDE_chromium_percentage_in_first_alloy_l362_36277

/-- Given two alloys, proves that the percentage of chromium in the first alloy is 12% -/
theorem chromium_percentage_in_first_alloy :
  let weight_first_alloy : ℝ := 15
  let weight_second_alloy : ℝ := 35
  let chromium_percentage_second_alloy : ℝ := 10
  let chromium_percentage_new_alloy : ℝ := 10.6
  let total_weight : ℝ := weight_first_alloy + weight_second_alloy
  ∃ (chromium_percentage_first_alloy : ℝ),
    chromium_percentage_first_alloy * weight_first_alloy / 100 +
    chromium_percentage_second_alloy * weight_second_alloy / 100 =
    chromium_percentage_new_alloy * total_weight / 100 ∧
    chromium_percentage_first_alloy = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_chromium_percentage_in_first_alloy_l362_36277


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_divisors_of_2520_l362_36257

theorem sum_of_distinct_prime_divisors_of_2520 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.divisors 2520)) id) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_divisors_of_2520_l362_36257


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l362_36216

/-- A three-digit number is between 100 and 999 inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem three_digit_number_proof :
  ∃ (x : ℕ), is_three_digit x ∧ (7000 + x) - (10 * x + 7) = 3555 ∧ x = 382 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l362_36216


namespace NUMINAMATH_CALUDE_power_seven_mod_eight_l362_36295

theorem power_seven_mod_eight : 7^135 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_eight_l362_36295


namespace NUMINAMATH_CALUDE_equal_distance_to_line_l362_36291

theorem equal_distance_to_line (a : ℝ) : 
  let A : ℝ × ℝ := (-2, 4)
  let B : ℝ × ℝ := (-4, 6)
  let distance_to_line (x y : ℝ) := |a * x + y + 1| / Real.sqrt (a^2 + 1)
  distance_to_line A.1 A.2 = distance_to_line B.1 B.2 → a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_equal_distance_to_line_l362_36291


namespace NUMINAMATH_CALUDE_larger_number_proof_l362_36237

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l362_36237


namespace NUMINAMATH_CALUDE_cheryl_mms_l362_36250

/-- The number of m&m's Cheryl ate after lunch -/
def lunch_mms : ℕ := 7

/-- The number of m&m's Cheryl ate after dinner -/
def dinner_mms : ℕ := 5

/-- The number of m&m's Cheryl gave to her sister -/
def sister_mms : ℕ := 13

/-- The total number of m&m's Cheryl had at the beginning -/
def total_mms : ℕ := lunch_mms + dinner_mms + sister_mms

theorem cheryl_mms : total_mms = 25 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_mms_l362_36250


namespace NUMINAMATH_CALUDE_markus_family_ages_l362_36246

/-- Given a family where:
  * Markus is twice the age of his son
  * Markus's son is twice the age of Markus's grandson
  * Markus's grandson is three times the age of Markus's great-grandson
  * The sum of their ages is 140 years
Prove that Markus's great-grandson's age is 140/22 years. -/
theorem markus_family_ages (markus son grandson great_grandson : ℚ)
  (h1 : markus = 2 * son)
  (h2 : son = 2 * grandson)
  (h3 : grandson = 3 * great_grandson)
  (h4 : markus + son + grandson + great_grandson = 140) :
  great_grandson = 140 / 22 := by
  sorry

end NUMINAMATH_CALUDE_markus_family_ages_l362_36246


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l362_36217

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 5*x + 6) / ((x - 1)*(x - 4)*(x - 6)) =
      A / (x - 1) + B / (x - 4) + C / (x - 6) ∧
      A = 2/15 ∧ B = -1/3 ∧ C = 3/5 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l362_36217


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l362_36222

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l362_36222


namespace NUMINAMATH_CALUDE_no_valid_propositions_l362_36255

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the given conditions
variable (m n : Line) (α β : Plane)
variable (h1 : perpendicular m α)
variable (h2 : contains β n)

-- State the theorem
theorem no_valid_propositions :
  ¬(∀ (m n : Line) (α β : Plane), 
    perpendicular m α → contains β n → 
    ((parallel_planes α β → perpendicular_lines m n) ∧
     (perpendicular_lines m n → parallel_planes α β) ∧
     (parallel_lines m n → perpendicular_planes α β) ∧
     (perpendicular_planes α β → parallel_lines m n))) :=
sorry

end NUMINAMATH_CALUDE_no_valid_propositions_l362_36255


namespace NUMINAMATH_CALUDE_remainder_twice_sum_first_150_l362_36266

theorem remainder_twice_sum_first_150 : 
  (2 * (List.range 150).sum) % 10000 = 2650 := by
  sorry

end NUMINAMATH_CALUDE_remainder_twice_sum_first_150_l362_36266


namespace NUMINAMATH_CALUDE_sum_of_rationals_l362_36292

theorem sum_of_rationals (a b : ℚ) (h : a + Real.sqrt 3 * b = Real.sqrt (4 + 2 * Real.sqrt 3)) : 
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rationals_l362_36292


namespace NUMINAMATH_CALUDE_sum_of_21st_set_l362_36213

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The first element of the nth set -/
def first_element (n : ℕ) : ℕ := 1 + sum_first_n (n - 1)

/-- The last element of the nth set -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- The theorem to prove -/
theorem sum_of_21st_set : S 21 = 4641 := by sorry

end NUMINAMATH_CALUDE_sum_of_21st_set_l362_36213


namespace NUMINAMATH_CALUDE_stratified_sample_third_group_l362_36290

-- Define the total ratio
def total_ratio : ℕ := 2 + 5 + 3

-- Define the ratio of the third group (model C)
def third_group_ratio : ℕ := 3

-- Define the sample size
def sample_size : ℕ := 120

-- Theorem statement
theorem stratified_sample_third_group :
  (sample_size * third_group_ratio) / total_ratio = 36 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_third_group_l362_36290


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l362_36206

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2 - I) / (1 - 3*I) = a + b*I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l362_36206


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l362_36214

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 12*x + a = (3*x + b)^2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l362_36214


namespace NUMINAMATH_CALUDE_g_zero_l362_36202

/-- The function g(x) = 5x - 6 -/
def g (x : ℝ) : ℝ := 5 * x - 6

/-- Theorem: g(6/5) = 0 -/
theorem g_zero : g (6 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_l362_36202


namespace NUMINAMATH_CALUDE_ace_then_diamond_probability_l362_36207

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumberOfAces : ℕ := 4

/-- Number of diamonds in a standard deck -/
def NumberOfDiamonds : ℕ := 13

/-- Probability of drawing an Ace as the first card and a diamond as the second card -/
def ProbabilityAceThenDiamond : ℚ := 1 / StandardDeck

theorem ace_then_diamond_probability :
  ProbabilityAceThenDiamond = 1 / StandardDeck := by
  sorry

end NUMINAMATH_CALUDE_ace_then_diamond_probability_l362_36207


namespace NUMINAMATH_CALUDE_max_valid_arrangement_l362_36232

/-- A type representing the cards with numbers 1 to 9 -/
inductive Card : Type
  | one | two | three | four | five | six | seven | eight | nine

/-- A function that returns the numerical value of a card -/
def cardValue : Card → Nat
  | Card.one => 1
  | Card.two => 2
  | Card.three => 3
  | Card.four => 4
  | Card.five => 5
  | Card.six => 6
  | Card.seven => 7
  | Card.eight => 8
  | Card.nine => 9

/-- A predicate that checks if two cards satisfy the adjacency condition -/
def validAdjacent (c1 c2 : Card) : Prop :=
  (cardValue c1 ∣ cardValue c2) ∨ (cardValue c2 ∣ cardValue c1)

/-- A type representing a valid arrangement of cards -/
def ValidArrangement := List Card

/-- A predicate that checks if an arrangement is valid -/
def isValidArrangement : ValidArrangement → Prop
  | [] => True
  | [_] => True
  | (c1 :: c2 :: rest) => validAdjacent c1 c2 ∧ isValidArrangement (c2 :: rest)

/-- The main theorem stating that the maximum number of cards in a valid arrangement is 8 -/
theorem max_valid_arrangement :
  (∃ (arr : ValidArrangement), isValidArrangement arr ∧ arr.length = 8) ∧
  (∀ (arr : ValidArrangement), isValidArrangement arr → arr.length ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_max_valid_arrangement_l362_36232


namespace NUMINAMATH_CALUDE_garden_area_is_400_l362_36244

/-- A rectangular garden with specific walking distances -/
structure Garden where
  length : ℝ
  width : ℝ
  length_total : ℝ
  perimeter_total : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ

/-- The garden satisfies the given conditions -/
def garden_satisfies_conditions (g : Garden) : Prop :=
  g.length_total = 2000 ∧
  g.perimeter_total = 2000 ∧
  g.length_walks = 50 ∧
  g.perimeter_walks = 20 ∧
  g.length_total = g.length * g.length_walks ∧
  g.perimeter_total = g.perimeter_walks * (2 * g.length + 2 * g.width)

/-- The theorem stating that a garden satisfying the conditions has an area of 400 square meters -/
theorem garden_area_is_400 (g : Garden) (h : garden_satisfies_conditions g) : 
  g.length * g.width = 400 := by
  sorry

#check garden_area_is_400

end NUMINAMATH_CALUDE_garden_area_is_400_l362_36244


namespace NUMINAMATH_CALUDE_gcd_840_1764_l362_36259

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l362_36259


namespace NUMINAMATH_CALUDE_curve_k_range_l362_36269

theorem curve_k_range (a : ℝ) (k : ℝ) : 
  ((-a)^2 - a*(-a) + 2*a + k = 0) → k ≤ (1/2 : ℝ) ∧ ∀ (ε : ℝ), ε > 0 → ∃ (k' : ℝ), k' < -ε ∧ ∃ (a' : ℝ), ((-a')^2 - a'*(-a') + 2*a' + k' = 0) :=
by sorry

end NUMINAMATH_CALUDE_curve_k_range_l362_36269


namespace NUMINAMATH_CALUDE_yujin_wire_length_l362_36211

/-- The length of Yujin's wire given Junhoe's wire length and the ratio --/
theorem yujin_wire_length (junhoe_length : ℝ) (ratio : ℝ) (h1 : junhoe_length = 134.5) (h2 : ratio = 1.06) :
  junhoe_length * ratio = 142.57 := by
  sorry

end NUMINAMATH_CALUDE_yujin_wire_length_l362_36211


namespace NUMINAMATH_CALUDE_percentage_problem_l362_36226

theorem percentage_problem : 
  ∃ P : ℝ, (0.45 * 60 = P * 40 + 13) ∧ P = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l362_36226


namespace NUMINAMATH_CALUDE_bus_driver_overtime_limit_l362_36284

/-- Represents the problem of determining overtime limit for a bus driver --/
theorem bus_driver_overtime_limit 
  (regular_rate : ℝ) 
  (overtime_rate : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) 
  (h1 : regular_rate = 16)
  (h2 : overtime_rate = regular_rate * 1.75)
  (h3 : total_compensation = 864)
  (h4 : total_hours = 48) :
  ∃ (limit : ℝ), 
    limit = 40 ∧ 
    total_compensation = limit * regular_rate + (total_hours - limit) * overtime_rate :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_overtime_limit_l362_36284


namespace NUMINAMATH_CALUDE_craigs_commission_problem_l362_36240

/-- Craig's appliance sales commission problem -/
theorem craigs_commission_problem 
  (fixed_amount : ℝ) 
  (num_appliances : ℕ) 
  (total_selling_price : ℝ) 
  (total_commission : ℝ) 
  (h1 : num_appliances = 6)
  (h2 : total_selling_price = 3620)
  (h3 : total_commission = 662)
  (h4 : total_commission = num_appliances * fixed_amount + 0.1 * total_selling_price) :
  fixed_amount = 50 := by
sorry

end NUMINAMATH_CALUDE_craigs_commission_problem_l362_36240


namespace NUMINAMATH_CALUDE_trig_identity_l362_36253

theorem trig_identity (α : ℝ) : 
  Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l362_36253


namespace NUMINAMATH_CALUDE_john_pays_21_l362_36212

/-- The amount John pays for candy bars -/
def john_payment (total_bars : ℕ) (dave_bars : ℕ) (price_per_bar : ℚ) : ℚ :=
  (total_bars - dave_bars : ℚ) * price_per_bar

/-- Theorem: John pays $21 for candy bars -/
theorem john_pays_21 :
  john_payment 20 6 (3/2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_john_pays_21_l362_36212


namespace NUMINAMATH_CALUDE_even_function_range_l362_36200

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_range (f : ℝ → ℝ) (h_even : IsEven f)
  (h_cond : ∀ x₁ x₂, x₁ ∈ Set.Ici 0 ∧ x₂ ∈ Set.Ici 0 ∧ x₁ ≠ x₂ → 
    (x₁ - x₂) * (f x₁ - f x₂) > 0)
  (m : ℝ) (h_ineq : f (m + 1) ≥ f 2) :
  m ∈ Set.Iic (-3) ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_range_l362_36200


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l362_36245

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the stack configuration -/
structure StackConfiguration where
  num_3ft : ℕ
  num_4ft : ℕ
  num_5ft : ℕ

def crate_dimensions : CrateDimensions :=
  { length := 3, width := 4, height := 5 }

def num_crates : ℕ := 12

def target_height : ℕ := 50

def valid_configuration (config : StackConfiguration) : Prop :=
  config.num_3ft + config.num_4ft + config.num_5ft = num_crates ∧
  3 * config.num_3ft + 4 * config.num_4ft + 5 * config.num_5ft = target_height

def num_valid_configurations : ℕ := 33616

def total_possible_configurations : ℕ := 3^num_crates

theorem crate_stacking_probability :
  (num_valid_configurations : ℚ) / (total_possible_configurations : ℚ) = 80 / 1593 := by
  sorry

#check crate_stacking_probability

end NUMINAMATH_CALUDE_crate_stacking_probability_l362_36245


namespace NUMINAMATH_CALUDE_no_heptagon_intersection_l362_36254

/-- A cube in 3D space -/
structure Cube where
  -- Add necessary fields for a cube

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields for a plane

/-- Represents the intersection of a plane and a cube -/
def Intersection (c : Cube) (p : Plane) : Set Point := sorry

/-- The number of edges of the cube that the plane intersects -/
def numIntersectedEdges (c : Cube) (p : Plane) : ℕ := sorry

/-- Predicate to check if a plane passes through a vertex more than once -/
def passesVertexTwice (c : Cube) (p : Plane) : Prop := sorry

/-- Theorem: A plane intersecting a cube cannot form a heptagon -/
theorem no_heptagon_intersection (c : Cube) (p : Plane) : 
  ¬(numIntersectedEdges c p = 7 ∧ ¬passesVertexTwice c p) := by
  sorry

end NUMINAMATH_CALUDE_no_heptagon_intersection_l362_36254


namespace NUMINAMATH_CALUDE_pentagonal_to_triangular_prism_l362_36279

/-- The number of cans in a pentagonal pyramid with l layers -/
def T (l : ℕ) : ℕ := l * (3 * l^2 - l) / 2

/-- A triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem pentagonal_to_triangular_prism (l : ℕ) (h : l ≥ 2) :
  ∃ n : ℕ, T l = l * triangular_number n :=
by
  sorry

end NUMINAMATH_CALUDE_pentagonal_to_triangular_prism_l362_36279


namespace NUMINAMATH_CALUDE_land_plot_side_length_l362_36275

/-- For a square-shaped land plot with an area of 100 square units, 
    the length of one side is 10 units. -/
theorem land_plot_side_length (area : ℝ) (side : ℝ) : 
  area = 100 → side * side = area → side = 10 := by
  sorry

end NUMINAMATH_CALUDE_land_plot_side_length_l362_36275


namespace NUMINAMATH_CALUDE_bottle_production_l362_36233

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 12 such machines will produce 2160 bottles in 4 minutes. -/
theorem bottle_production
  (machines : ℕ)
  (bottles_per_minute : ℕ)
  (h1 : machines = 6)
  (h2 : bottles_per_minute = 270)
  (time : ℕ)
  (h3 : time = 4) :
  (12 * bottles_per_minute * time) / machines = 2160 :=
sorry

end NUMINAMATH_CALUDE_bottle_production_l362_36233


namespace NUMINAMATH_CALUDE_optimal_cutting_l362_36209

/-- Represents a rectangular piece of cardboard -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  r.length * r.width

/-- Represents the problem of cutting small rectangles from a large rectangle -/
structure CuttingProblem :=
  (large : Rectangle)
  (small : Rectangle)

/-- Calculates the maximum number of small rectangles that can be cut from a large rectangle -/
def maxPieces (p : CuttingProblem) : ℕ :=
  sorry

theorem optimal_cutting (p : CuttingProblem) 
  (h1 : p.large = ⟨17, 22⟩) 
  (h2 : p.small = ⟨3, 5⟩) : 
  maxPieces p = 24 :=
sorry

end NUMINAMATH_CALUDE_optimal_cutting_l362_36209


namespace NUMINAMATH_CALUDE_fraction_subtraction_theorem_l362_36239

theorem fraction_subtraction_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x := a * b / (a + b)
  let y := a * b * (b + a) / (b^2 + a*b + a^2)
  ((a - x) / (b - x) = (a / b)^2) ∧ ((a - y) / (b - y) = (a / b)^3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_theorem_l362_36239


namespace NUMINAMATH_CALUDE_pipe_filling_speed_l362_36261

/-- Proves that if Pipe A fills a tank in 24 minutes, and both Pipe A and Pipe B together fill the tank in 3 minutes, then Pipe B fills the tank 7 times faster than Pipe A. -/
theorem pipe_filling_speed (fill_time_A : ℝ) (fill_time_both : ℝ) (speed_ratio : ℝ) : 
  fill_time_A = 24 → 
  fill_time_both = 3 → 
  (1 / fill_time_A + speed_ratio / fill_time_A) * fill_time_both = 1 →
  speed_ratio = 7 := by
sorry

end NUMINAMATH_CALUDE_pipe_filling_speed_l362_36261


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l362_36267

theorem nearest_integer_to_power : 
  ∃ n : ℤ, n = 3936 ∧ ∀ m : ℤ, |((3:ℝ) + Real.sqrt 5)^5 - (n:ℝ)| ≤ |((3:ℝ) + Real.sqrt 5)^5 - (m:ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l362_36267


namespace NUMINAMATH_CALUDE_derivative_ln_2x_plus_1_l362_36241

open Real

theorem derivative_ln_2x_plus_1 (x : ℝ) :
  deriv (fun x => Real.log (2 * x + 1)) x = 2 / (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_2x_plus_1_l362_36241


namespace NUMINAMATH_CALUDE_five_sixteenths_decimal_l362_36282

theorem five_sixteenths_decimal : (5 : ℚ) / 16 = (3125 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_five_sixteenths_decimal_l362_36282


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l362_36278

/-- Given an isosceles triangle with equal sides of length x and base of length y,
    if a median to one of the equal sides divides the perimeter into parts of 15 cm and 6 cm,
    then the length of the base (y) is 1 cm. -/
theorem isosceles_triangle_base_length
  (x y : ℝ)
  (isosceles : x > 0)
  (perimeter_division : x + x/2 = 15 ∧ y + x/2 = 6 ∨ x + x/2 = 6 ∧ y + x/2 = 15)
  (triangle_inequality : x + x > y ∧ x + y > x ∧ x + y > x) :
  y = 1 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l362_36278


namespace NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l362_36288

noncomputable section

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := y^2 / 3 + x^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x - y = 4

-- Define the distance function between a point (x, y) and the line C₂
def distance_to_C₂ (x y : ℝ) : ℝ := |x - y - 4| / Real.sqrt 2

-- State the theorem
theorem min_distance_ellipse_to_line :
  ∃ (α : ℝ), 
    let x := Real.sin α
    let y := Real.sqrt 3 * Real.cos α
    C₁ x y ∧ 
    (∀ β : ℝ, distance_to_C₂ (Real.sin β) (Real.sqrt 3 * Real.cos β) ≥ Real.sqrt 2) ∧
    distance_to_C₂ x y = Real.sqrt 2 ∧
    x = 1/2 ∧ y = -3/2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l362_36288


namespace NUMINAMATH_CALUDE_equation_equivalence_l362_36201

theorem equation_equivalence (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 3*x^2 + x + 2 = 0 ↔ x^2 * (y^2 + y - 5) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l362_36201


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l362_36247

def f (a : ℤ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + 1

theorem quadratic_inequality_solution_set 
  (a : ℤ) 
  (h1 : ∃! x : ℝ, -2 < x ∧ x < -1 ∧ f a x = 0) :
  {x : ℝ | f a x > 1} = {x : ℝ | -1 < x ∧ x < 0} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l362_36247


namespace NUMINAMATH_CALUDE_correct_result_l362_36243

theorem correct_result (x : ℝ) : (-1.25 * x) - 0.25 = 1.25 * x → -1.25 * x = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l362_36243
