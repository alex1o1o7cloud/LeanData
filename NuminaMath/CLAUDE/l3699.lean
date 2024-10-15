import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3699_369922

theorem inequality_proof (x y z : ℝ) (h : x^4 + y^4 + z^4 + x*y*z = 4) :
  x ≤ 2 ∧ Real.sqrt (2 - x) ≥ (y + z) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3699_369922


namespace NUMINAMATH_CALUDE_smallest_power_l3699_369992

theorem smallest_power (a b c d : ℕ) : 
  a = 2 → b = 3 → c = 5 → d = 6 → 
  a^55 < b^44 ∧ a^55 < c^33 ∧ a^55 < d^22 := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_l3699_369992


namespace NUMINAMATH_CALUDE_demand_decrease_with_price_increase_l3699_369969

theorem demand_decrease_with_price_increase (P Q : ℝ) (P_new Q_new : ℝ) :
  P > 0 → Q > 0 →
  P_new = 1.5 * P →
  P_new * Q_new = 1.2 * P * Q →
  Q_new = 0.8 * Q :=
by
  sorry

#check demand_decrease_with_price_increase

end NUMINAMATH_CALUDE_demand_decrease_with_price_increase_l3699_369969


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3699_369921

theorem integer_roots_of_polynomial (a b c : ℚ) : 
  ∃ (p q : ℤ), p ≠ q ∧ 
    (∀ x : ℂ, x^4 + a*x^2 + b*x + c = 0 ↔ 
      (x = 2 - Real.sqrt 3 ∨ x = p ∨ x = q ∨ x = 2 + Real.sqrt 3)) ∧
    p = -1 ∧ q = -3 := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3699_369921


namespace NUMINAMATH_CALUDE_equation_represents_parabola_l3699_369914

/-- The equation represents a parabola if it can be transformed into the form x² + bx + c = Ay + B --/
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c A B : ℝ, a ≠ 0 ∧ 
  ∀ x y : ℝ, f x y ↔ a * x^2 + b * x + c = A * y + B

/-- The given equation |y-3| = √((x+4)² + y²) --/
def given_equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 4)^2 + y^2)

theorem equation_represents_parabola : is_parabola given_equation := by
  sorry

end NUMINAMATH_CALUDE_equation_represents_parabola_l3699_369914


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3699_369957

/-- Given a cubic function f with three distinct roots, prove properties about its values at 0, 1, and 3 -/
theorem cubic_function_properties (a b c : ℝ) (abc : ℝ) :
  a < b → b < c →
  let f : ℝ → ℝ := fun x ↦ x^3 - 6*x^2 + 9*x - abc
  f a = 0 → f b = 0 → f c = 0 →
  (f 0) * (f 1) < 0 ∧ (f 0) * (f 3) > 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3699_369957


namespace NUMINAMATH_CALUDE_function_equality_l3699_369989

theorem function_equality (f g : ℕ+ → ℕ+) 
  (h1 : ∀ n : ℕ+, f (g n) = f n + 1) 
  (h2 : ∀ n : ℕ+, g (f n) = g n + 1) : 
  ∀ n : ℕ+, f n = g n := by
sorry

end NUMINAMATH_CALUDE_function_equality_l3699_369989


namespace NUMINAMATH_CALUDE_vector_sum_of_squares_l3699_369919

/-- Given vectors a and b, with n as their midpoint, prove that ‖a‖² + ‖b‖² = 48 -/
theorem vector_sum_of_squares (a b : ℝ × ℝ) (n : ℝ × ℝ) : 
  n = (4, -1) → n = (a + b) / 2 → a • b = 10 → ‖a‖^2 + ‖b‖^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_of_squares_l3699_369919


namespace NUMINAMATH_CALUDE_fish_after_ten_years_l3699_369938

def initial_fish : ℕ := 6

def fish_added (year : ℕ) : ℕ :=
  if year ≤ 10 then year + 1 else 0

def fish_died (year : ℕ) : ℕ :=
  if year ≤ 10 then
    if year ≤ 4 then 5 - year
    else year - 3
  else 0

def fish_count (year : ℕ) : ℕ :=
  if year = 0 then initial_fish
  else (fish_count (year - 1) + fish_added year - fish_died year)

theorem fish_after_ten_years :
  fish_count 10 = 34 := by sorry

end NUMINAMATH_CALUDE_fish_after_ten_years_l3699_369938


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l3699_369947

/-- Given two lines l₁ and l₂, prove that their distance is √10/5 -/
theorem distance_between_parallel_lines (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | 2*x + 3*m*y - m + 2 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | m*x + 6*y - 4 = 0}
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ → (x₂, y₂) ∈ l₂ → 
    (2 * (y₂ - y₁) = 3*m * (x₂ - x₁))) →  -- parallel condition
  (∃ (d : ℝ), d = Real.sqrt 10 / 5 ∧
    ∀ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ l₁ → p₂ ∈ l₂ →
      d ≤ Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_parallel_lines_l3699_369947


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3699_369981

/-- A rectangular field with length double its width and perimeter 180 meters has an area of 1800 square meters. -/
theorem rectangular_field_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  width > 0 →
  length = 2 * width →
  perimeter = 2 * (length + width) →
  perimeter = 180 →
  area = length * width →
  area = 1800 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l3699_369981


namespace NUMINAMATH_CALUDE_base13_representation_of_234_l3699_369913

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a natural number to its base 13 representation -/
def toBase13 (n : ℕ) : List Base13Digit := sorry

/-- Converts a list of Base13Digits to its decimal (base 10) value -/
def fromBase13 (digits : List Base13Digit) : ℕ := sorry

theorem base13_representation_of_234 :
  toBase13 234 = [Base13Digit.D1, Base13Digit.D5] := by sorry

end NUMINAMATH_CALUDE_base13_representation_of_234_l3699_369913


namespace NUMINAMATH_CALUDE_conference_hall_tables_l3699_369955

/-- Given a conference hall with tables and chairs, prove the number of tables. -/
theorem conference_hall_tables (total_legs : ℕ) (chairs_per_table : ℕ) (chair_legs : ℕ) (table_legs : ℕ)
  (h1 : chairs_per_table = 8)
  (h2 : chair_legs = 4)
  (h3 : table_legs = 4)
  (h4 : total_legs = 648) :
  ∃ (num_tables : ℕ), num_tables = 18 ∧ 
    total_legs = num_tables * table_legs + num_tables * chairs_per_table * chair_legs :=
by sorry

end NUMINAMATH_CALUDE_conference_hall_tables_l3699_369955


namespace NUMINAMATH_CALUDE_rocket_components_most_suitable_for_comprehensive_survey_l3699_369973

/-- Represents the characteristics of a scenario that can be surveyed -/
structure SurveyScenario where
  population : Type
  countable : Bool
  criticalImportance : Bool
  requiresCompleteExamination : Bool

/-- Defines what makes a scenario suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (scenario : SurveyScenario) : Prop :=
  scenario.countable ∧ scenario.criticalImportance ∧ scenario.requiresCompleteExamination

/-- Represents the Long March II-F Y17 rocket components scenario -/
def rocketComponentsScenario : SurveyScenario :=
  { population := Unit,  -- The type doesn't matter for this example
    countable := true,
    criticalImportance := true,
    requiresCompleteExamination := true }

/-- Represents all other given scenarios -/
def otherScenarios : List SurveyScenario :=
  [ { population := Unit, countable := false, criticalImportance := false, requiresCompleteExamination := false },
    { population := Unit, countable := false, criticalImportance := true, requiresCompleteExamination := false },
    { population := Unit, countable := false, criticalImportance := false, requiresCompleteExamination := false } ]

theorem rocket_components_most_suitable_for_comprehensive_survey :
  isSuitableForComprehensiveSurvey rocketComponentsScenario ∧
  (∀ scenario ∈ otherScenarios, ¬(isSuitableForComprehensiveSurvey scenario)) :=
sorry

end NUMINAMATH_CALUDE_rocket_components_most_suitable_for_comprehensive_survey_l3699_369973


namespace NUMINAMATH_CALUDE_mrs_hilt_markers_l3699_369979

theorem mrs_hilt_markers (num_packages : ℕ) (markers_per_package : ℕ) 
  (h1 : num_packages = 7) 
  (h2 : markers_per_package = 5) : 
  num_packages * markers_per_package = 35 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_markers_l3699_369979


namespace NUMINAMATH_CALUDE_spotlight_illumination_theorem_l3699_369999

/-- Represents a point on a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction (North, South, East, West) --/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a spotlight that illuminates a right angle --/
structure Spotlight where
  position : Point
  direction1 : Direction
  direction2 : Direction

/-- Represents the configuration of four spotlights --/
structure SpotlightConfiguration where
  spotlights : Fin 4 → Spotlight

/-- Predicate to check if a configuration illuminates the entire plane --/
def illuminatesEntirePlane (config : SpotlightConfiguration) : Prop := sorry

/-- The main theorem stating that there exists a configuration of spotlights that illuminates the entire plane --/
theorem spotlight_illumination_theorem :
  ∃ (config : SpotlightConfiguration), illuminatesEntirePlane config :=
sorry

end NUMINAMATH_CALUDE_spotlight_illumination_theorem_l3699_369999


namespace NUMINAMATH_CALUDE_campaign_fundraising_l3699_369962

-- Define the problem parameters
def max_donation : ℕ := 1200
def max_donors : ℕ := 500
def half_donors_multiplier : ℕ := 3
def donation_percentage : ℚ := 40 / 100

-- Define the total money raised
def total_money_raised : ℚ := 3750000

-- Theorem statement
theorem campaign_fundraising :
  let max_donation_total := max_donation * max_donors
  let half_donation_total := (max_donation / 2) * (max_donors * half_donors_multiplier)
  let total_donations := max_donation_total + half_donation_total
  total_donations = donation_percentage * total_money_raised := by
  sorry


end NUMINAMATH_CALUDE_campaign_fundraising_l3699_369962


namespace NUMINAMATH_CALUDE_seventeen_meter_rod_pieces_l3699_369980

/-- The number of pieces of a given length that can be cut from a rod --/
def num_pieces (rod_length : ℕ) (piece_length : ℕ) : ℕ :=
  rod_length / piece_length

/-- Theorem: The number of 85 cm pieces that can be cut from a 17-meter rod is 20 --/
theorem seventeen_meter_rod_pieces : num_pieces (17 * 100) 85 = 20 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_meter_rod_pieces_l3699_369980


namespace NUMINAMATH_CALUDE_oil_quantity_function_correct_l3699_369996

/-- Represents the remaining oil quantity in liters -/
def Q (t : ℝ) : ℝ := 40 - 0.2 * t

/-- The initial oil quantity in liters -/
def initial_quantity : ℝ := 40

/-- The oil flow rate in liters per minute -/
def flow_rate : ℝ := 0.2

theorem oil_quantity_function_correct (t : ℝ) :
  Q t = initial_quantity - flow_rate * t :=
by sorry

end NUMINAMATH_CALUDE_oil_quantity_function_correct_l3699_369996


namespace NUMINAMATH_CALUDE_like_terms_exponent_relation_l3699_369963

theorem like_terms_exponent_relation (x y : ℤ) : 
  (∃ (m n : ℝ), -0.5 * m^x * n^3 = 5 * m^4 * n^y) → (y - x)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_relation_l3699_369963


namespace NUMINAMATH_CALUDE_complex_equidistant_points_l3699_369972

theorem complex_equidistant_points (z : ℂ) : 
  (Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
   Complex.abs (z - 2) = Complex.abs (z + 2*I)) ↔ 
  z = -1 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_equidistant_points_l3699_369972


namespace NUMINAMATH_CALUDE_ice_pop_probability_l3699_369994

/-- Represents the number of ice pops of each flavor --/
structure IcePops where
  cherry : ℕ
  orange : ℕ
  lemonLime : ℕ

/-- Calculates the probability of selecting two ice pops of different flavors --/
def probDifferentFlavors (pops : IcePops) : ℚ :=
  let total := pops.cherry + pops.orange + pops.lemonLime
  1 - (pops.cherry * (pops.cherry - 1) + pops.orange * (pops.orange - 1) + pops.lemonLime * (pops.lemonLime - 1)) / (total * (total - 1))

/-- The main theorem stating that for the given ice pop distribution, 
    the probability of selecting two different flavors is 8/11 --/
theorem ice_pop_probability : 
  let pops : IcePops := ⟨4, 3, 4⟩
  probDifferentFlavors pops = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_ice_pop_probability_l3699_369994


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3699_369961

/-- Represents a hyperbola with the equation (x^2 / a^2) - (y^2 / b^2) = 1 -/
structure Hyperbola (a b : ℝ) where
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- Represents an asymptote of a hyperbola -/
structure Asymptote (m : ℝ) where
  equation : ∀ x y : ℝ, y = m * x

/-- Represents a focus point of a hyperbola -/
structure Focus (x y : ℝ) where
  coordinates : ℝ × ℝ := (x, y)

theorem hyperbola_properties (h : Hyperbola 12 9) :
  (∃ a₁ : Asymptote (3/4), True) ∧
  (∃ a₂ : Asymptote (-3/4), True) ∧
  (∃ f₁ : Focus 15 0, True) ∧
  (∃ f₂ : Focus (-15) 0, True) := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_properties_l3699_369961


namespace NUMINAMATH_CALUDE_f_le_g_l3699_369931

def f (n : ℕ+) : ℚ :=
  (Finset.range n).sum (fun i => 1 / ((i + 1) : ℚ) ^ 2) + 1

def g (n : ℕ+) : ℚ :=
  1/2 * (3 - 1 / (n : ℚ) ^ 2)

theorem f_le_g : ∀ n : ℕ+, f n ≤ g n := by
  sorry

end NUMINAMATH_CALUDE_f_le_g_l3699_369931


namespace NUMINAMATH_CALUDE_blue_chip_fraction_l3699_369941

theorem blue_chip_fraction (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 60)
  (h2 : red = 34)
  (h3 : green = 16) :
  (total - red - green : ℚ) / total = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_blue_chip_fraction_l3699_369941


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l3699_369927

theorem roots_of_quadratic_equation (θ : Real) (x₁ x₂ : ℂ) :
  θ ∈ Set.Icc 0 π ∧
  x₁^2 - 3 * Real.sin θ * x₁ + Real.sin θ^2 + 1 = 0 ∧
  x₂^2 - 3 * Real.sin θ * x₂ + Real.sin θ^2 + 1 = 0 ∧
  Complex.abs x₁ + Complex.abs x₂ = 2 →
  θ = 0 := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l3699_369927


namespace NUMINAMATH_CALUDE_problem_solution_l3699_369937

theorem problem_solution :
  (∀ x : ℝ, x^2 - x ≥ x - 1) ∧
  (∃ x : ℝ, x > 1 ∧ x + 4 / (x - 1) = 6) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → b / a < (b + 1) / (a + 1)) ∧
  (∀ x : ℝ, (x^2 + 10) / Real.sqrt (x^2 + 9) > 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3699_369937


namespace NUMINAMATH_CALUDE_a_ge_one_l3699_369991

open Real

/-- The function f(x) = a * ln(x) + (1/2) * x^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + (1/2) * x^2

/-- Theorem stating that if f satisfies the given condition, then a ≥ 1 -/
theorem a_ge_one (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 2) →
  a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_a_ge_one_l3699_369991


namespace NUMINAMATH_CALUDE_missing_chess_pieces_l3699_369978

/-- The number of pieces in a standard chess set -/
def standard_chess_set_pieces : ℕ := 32

/-- The number of pieces present -/
def present_pieces : ℕ := 28

/-- The number of missing pieces -/
def missing_pieces : ℕ := standard_chess_set_pieces - present_pieces

theorem missing_chess_pieces : missing_pieces = 4 := by
  sorry

end NUMINAMATH_CALUDE_missing_chess_pieces_l3699_369978


namespace NUMINAMATH_CALUDE_pakistan_traditional_model_l3699_369918

-- Define the population growth models
inductive PopulationModel
  | Primitive
  | Traditional
  | Modern

-- Define a function that assigns a population model to a country
def countryModel : String → PopulationModel
  | "Nigeria" => PopulationModel.Traditional
  | "China" => PopulationModel.Modern
  | "India" => PopulationModel.Traditional
  | "Pakistan" => PopulationModel.Traditional
  | _ => PopulationModel.Traditional  -- Default case

-- Theorem stating that Pakistan follows the Traditional model
theorem pakistan_traditional_model :
  countryModel "Pakistan" = PopulationModel.Traditional := by
  sorry

end NUMINAMATH_CALUDE_pakistan_traditional_model_l3699_369918


namespace NUMINAMATH_CALUDE_best_candidate_is_C_l3699_369977

structure Participant where
  name : String
  average_score : Float
  variance : Float

def participants : List Participant := [
  { name := "A", average_score := 8.5, variance := 1.7 },
  { name := "B", average_score := 8.8, variance := 2.1 },
  { name := "C", average_score := 9.1, variance := 1.7 },
  { name := "D", average_score := 9.1, variance := 2.5 }
]

def is_best_candidate (p : Participant) : Prop :=
  ∀ q ∈ participants,
    (p.average_score > q.average_score ∨
    (p.average_score = q.average_score ∧ p.variance ≤ q.variance))

theorem best_candidate_is_C :
  ∃ p ∈ participants, p.name = "C" ∧ is_best_candidate p :=
by sorry

end NUMINAMATH_CALUDE_best_candidate_is_C_l3699_369977


namespace NUMINAMATH_CALUDE_hamburger_combinations_count_l3699_369946

/-- The number of condiments available for hamburgers -/
def num_condiments : ℕ := 8

/-- The number of patty options available for hamburgers -/
def num_patty_options : ℕ := 4

/-- Calculates the number of different hamburger combinations -/
def num_hamburger_combinations : ℕ := 2^num_condiments * num_patty_options

theorem hamburger_combinations_count :
  num_hamburger_combinations = 1024 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_count_l3699_369946


namespace NUMINAMATH_CALUDE_square_sum_from_means_l3699_369940

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 150) : 
  x^2 + y^2 = 1300 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l3699_369940


namespace NUMINAMATH_CALUDE_washing_machine_loads_l3699_369970

/-- Calculate the minimum number of loads required to wash a given number of items with a fixed machine capacity -/
def minimum_loads (total_items : ℕ) (machine_capacity : ℕ) : ℕ :=
  (total_items + machine_capacity - 1) / machine_capacity

/-- The washing machine capacity -/
def machine_capacity : ℕ := 12

/-- The total number of items to wash -/
def total_items : ℕ := 19 + 8 + 15 + 10

theorem washing_machine_loads :
  minimum_loads total_items machine_capacity = 5 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_loads_l3699_369970


namespace NUMINAMATH_CALUDE_shadow_arrangements_l3699_369942

def word_length : Nat := 6
def selection_size : Nat := 4
def remaining_letters : Nat := word_length - 1  -- excluding 'a'
def letters_to_choose : Nat := selection_size - 1  -- excluding 'a'

theorem shadow_arrangements : 
  (Nat.choose remaining_letters letters_to_choose) * 
  (Nat.factorial selection_size) = 240 := by
sorry

end NUMINAMATH_CALUDE_shadow_arrangements_l3699_369942


namespace NUMINAMATH_CALUDE_area_ratio_value_l3699_369905

/-- Represents a sequence of circles touching a right angle -/
structure CircleSequence where
  -- The ratio of radii between consecutive circles
  radius_ratio : ℝ
  -- Assumption that the ratio is equal to (√2 - 1)²
  h_ratio : radius_ratio = (Real.sqrt 2 - 1)^2

/-- The ratio of the area of the first circle to the sum of areas of all subsequent circles -/
def area_ratio (seq : CircleSequence) : ℝ := sorry

/-- Theorem stating the area ratio for the given circle sequence -/
theorem area_ratio_value (seq : CircleSequence) :
  area_ratio seq = 16 + 12 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_area_ratio_value_l3699_369905


namespace NUMINAMATH_CALUDE_sequential_draw_probability_l3699_369956

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of cards of each suit in a standard deck -/
def cardsPerSuit : ℕ := 13

/-- The probability of drawing a club, then a diamond, then a heart in order from a standard deck -/
def sequentialDrawProbability : ℚ :=
  (cardsPerSuit : ℚ) / standardDeckSize *
  (cardsPerSuit : ℚ) / (standardDeckSize - 1) *
  (cardsPerSuit : ℚ) / (standardDeckSize - 2)

theorem sequential_draw_probability :
  sequentialDrawProbability = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_sequential_draw_probability_l3699_369956


namespace NUMINAMATH_CALUDE_work_completion_time_l3699_369976

theorem work_completion_time 
  (efficiency_ratio : ℝ) 
  (combined_time : ℝ) 
  (a_efficiency : ℝ) 
  (b_efficiency : ℝ) :
  efficiency_ratio = 2 →
  combined_time = 6 →
  a_efficiency = efficiency_ratio * b_efficiency →
  (a_efficiency + b_efficiency) * combined_time = b_efficiency * 18 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3699_369976


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l3699_369943

theorem unique_two_digit_integer (u : ℕ) : 
  (10 ≤ u ∧ u < 100) →
  (15 * u) % 100 = 45 →
  u % 17 = 7 →
  u = 43 :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l3699_369943


namespace NUMINAMATH_CALUDE_triangle_ratio_range_l3699_369936

theorem triangle_ratio_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  b / Real.sin B = c / Real.sin C ∧  -- Law of sines
  -Real.cos B / Real.cos C = (2 * a + b) / c  -- Given condition
  →
  1 < (a + b) / c ∧ (a + b) / c ≤ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_range_l3699_369936


namespace NUMINAMATH_CALUDE_expansion_terms_abcd_efghi_l3699_369971

/-- The number of terms in the expansion of a product of two sums -/
def expansion_terms (n m : ℕ) : ℕ := n * m

/-- The first group (a+b+c+d) has 4 terms -/
def first_group_terms : ℕ := 4

/-- The second group (e+f+g+h+i) has 5 terms -/
def second_group_terms : ℕ := 5

/-- Theorem: The number of terms in the expansion of (a+b+c+d)(e+f+g+h+i) is 20 -/
theorem expansion_terms_abcd_efghi :
  expansion_terms first_group_terms second_group_terms = 20 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_abcd_efghi_l3699_369971


namespace NUMINAMATH_CALUDE_judy_school_week_days_l3699_369998

/-- The number of pencils Judy uses during her school week. -/
def pencils_per_week : ℕ := 10

/-- The number of pencils in a pack. -/
def pencils_per_pack : ℕ := 30

/-- The cost of a pack of pencils in dollars. -/
def cost_per_pack : ℚ := 4

/-- The amount Judy spends on pencils in dollars. -/
def total_spent : ℚ := 12

/-- The number of days over which Judy spends the total amount. -/
def total_days : ℕ := 45

/-- The number of days in Judy's school week. -/
def school_week_days : ℕ := 5

/-- Theorem stating that the number of days in Judy's school week is 5. -/
theorem judy_school_week_days :
  (pencils_per_week : ℚ) * total_days * cost_per_pack =
  pencils_per_pack * total_spent * (school_week_days : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_judy_school_week_days_l3699_369998


namespace NUMINAMATH_CALUDE_sqrt_comparison_l3699_369904

theorem sqrt_comparison : Real.sqrt 11 - 3 < Real.sqrt 7 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l3699_369904


namespace NUMINAMATH_CALUDE_samson_age_relation_l3699_369928

/-- Samson's current age in years -/
def samsonAge : ℝ := 6.25

/-- Samson's mother's current age in years -/
def motherAge : ℝ := 30.65

/-- The age Samson will be when his mother is exactly 4 times his age -/
def targetAge : ℝ := 8.1333

theorem samson_age_relation :
  ∃ (T : ℝ), 
    (samsonAge + T = targetAge) ∧ 
    (motherAge + T = 4 * (samsonAge + T)) := by
  sorry

end NUMINAMATH_CALUDE_samson_age_relation_l3699_369928


namespace NUMINAMATH_CALUDE_pipe_equivalence_l3699_369926

/-- The number of smaller pipes needed to match the water-carrying capacity of a larger pipe -/
theorem pipe_equivalence (r_large r_small : ℝ) (h_large : r_large = 4) (h_small : r_small = 1) :
  (π * r_large ^ 2) / (π * r_small ^ 2) = 16 := by
  sorry

#check pipe_equivalence

end NUMINAMATH_CALUDE_pipe_equivalence_l3699_369926


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3699_369975

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 1 = 0 → x₂^2 - 2*x₂ - 1 = 0 → x₁^2 + x₂^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3699_369975


namespace NUMINAMATH_CALUDE_equal_fractions_l3699_369915

theorem equal_fractions (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  let f1 := (x + y) / (x^2 + x*y + y^2)
  let f2 := (y + z) / (y^2 + y*z + z^2)
  let f3 := (z + x) / (z^2 + z*x + x^2)
  (f1 = f2 ∨ f2 = f3 ∨ f3 = f1) → (f1 = f2 ∧ f2 = f3) :=
by sorry

end NUMINAMATH_CALUDE_equal_fractions_l3699_369915


namespace NUMINAMATH_CALUDE_W_lower_bound_l3699_369920

/-- W(k,2) is the smallest number such that if n ≥ W(k,2), 
    for each coloring of the set {1,2,...,n} with two colors, 
    there exists a monochromatic arithmetic progression of length k -/
def W (k : ℕ) : ℕ := sorry

/-- The main theorem stating that W(k,2) = Ω(2^(k/2)) -/
theorem W_lower_bound : ∃ (c : ℝ) (k₀ : ℕ), c > 0 ∧ ∀ k ≥ k₀, (W k : ℝ) ≥ c * 2^(k/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_W_lower_bound_l3699_369920


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l3699_369909

/-- Represents a parabola in the form y² = -2px --/
structure Parabola where
  p : ℝ

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about a specific parabola and line --/
theorem parabola_and_line_properties
  (C : Parabola)
  (A : Point)
  (h1 : A.y^2 = -2 * C.p * A.x) -- A lies on the parabola
  (h2 : A.x = -1 ∧ A.y = -2) -- A is (-1, -2)
  (h3 : ∃ (B : Point), B ≠ A ∧ 
    (B.y - A.y) / (B.x - A.x) = -Real.sqrt 3 ∧ -- Line AB has slope -√3
    B.y^2 = -2 * C.p * B.x) -- B also lies on the parabola
  : 
  (C.p = -2) ∧ -- Equation of parabola is y² = -4x
  (∀ (x y : ℝ), y^2 = -4*x ↔ y^2 = -2 * C.p * x) ∧ -- Equivalent form of parabola equation
  (1 = -C.p/2) ∧ -- Axis of symmetry is x = 1
  (∃ (B : Point), B ≠ A ∧
    (B.y - A.y)^2 + (B.x - A.x)^2 = (16/3)^2) -- Length of AB is 16/3
  := by sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l3699_369909


namespace NUMINAMATH_CALUDE_tan_sum_identity_l3699_369911

theorem tan_sum_identity (A B : Real) (hA : A = 10 * π / 180) (hB : B = 20 * π / 180) :
  (1 + Real.tan A) * (1 + Real.tan B) = 1 + Real.sqrt 3 * (Real.tan A + Real.tan B) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l3699_369911


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_chord_length_l3699_369910

/-- Given a hyperbola and a circle, prove the length of the chord formed by their intersection -/
theorem hyperbola_circle_intersection_chord_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola : ℝ → ℝ → Prop) 
  (asymptote : ℝ → ℝ → Prop)
  (circle : ℝ → ℝ → Prop) :
  (∀ x y, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1) →
  (asymptote 1 2) →
  (∀ x y, circle x y ↔ (x + 1)^2 + (y - 2)^2 = 4) →
  ∃ chord_length, chord_length = 4 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_chord_length_l3699_369910


namespace NUMINAMATH_CALUDE_max_value_implies_b_equals_two_l3699_369988

def is_valid_triple (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a ∈ ({2, 3, 6} : Set ℕ) ∧ 
  b ∈ ({2, 3, 6} : Set ℕ) ∧ 
  c ∈ ({2, 3, 6} : Set ℕ)

theorem max_value_implies_b_equals_two (a b c : ℕ) :
  is_valid_triple a b c →
  (a : ℚ) / (b / c) ≤ 9 →
  (∀ x y z : ℕ, is_valid_triple x y z → (x : ℚ) / (y / z) ≤ 9) →
  b = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_b_equals_two_l3699_369988


namespace NUMINAMATH_CALUDE_root_in_interval_l3699_369903

-- Define the function g(x) = lg x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 10 + x - 2

-- State the theorem
theorem root_in_interval :
  ∃ x₀ : ℝ, g x₀ = 0 ∧ 1 < x₀ ∧ x₀ < 2 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l3699_369903


namespace NUMINAMATH_CALUDE_crayons_left_l3699_369901

theorem crayons_left (initial_crayons : ℕ) (crayons_taken : ℕ) : 
  initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_l3699_369901


namespace NUMINAMATH_CALUDE_difference_of_powers_l3699_369958

theorem difference_of_powers (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4) 
  (h2 : c ^ 3 = d ^ 2) 
  (h3 : c - a = 19) : 
  d - b = 757 := by
sorry

end NUMINAMATH_CALUDE_difference_of_powers_l3699_369958


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l3699_369930

/-- The locus of points (x, y) in the complex plane satisfying the given equation is an ellipse -/
theorem locus_is_ellipse (x y : ℝ) : 
  let z : ℂ := x + y * Complex.I
  (Complex.abs (z - (2 - Complex.I)) + Complex.abs (z - (-3 + Complex.I)) = 6) →
  ∃ (a b c d e f : ℝ), 
    a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0 ∧ 
    b^2 - 4*a*c < 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l3699_369930


namespace NUMINAMATH_CALUDE_three_digit_sum_problem_l3699_369964

/-- Represents a three-digit number in the form abc -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem three_digit_sum_problem (a c : Nat) :
  let num1 := ThreeDigitNumber.mk 3 a 7 (by sorry)
  let num2 := ThreeDigitNumber.mk 2 1 4 (by sorry)
  let sum := ThreeDigitNumber.mk 5 c 1 (by sorry)
  (num1.toNat + num2.toNat = sum.toNat) →
  (sum.toNat % 3 = 0) →
  a + c = 4 := by
  sorry

#check three_digit_sum_problem

end NUMINAMATH_CALUDE_three_digit_sum_problem_l3699_369964


namespace NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l3699_369953

theorem proof_by_contradiction_assumption (a b : ℕ) (h : 5 ∣ (a * b)) :
  (¬ (5 ∣ a) ∧ ¬ (5 ∣ b)) ↔ 
  ¬ (5 ∣ a ∨ 5 ∣ b) :=
by sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l3699_369953


namespace NUMINAMATH_CALUDE_least_integer_square_52_more_than_triple_l3699_369995

theorem least_integer_square_52_more_than_triple : 
  ∃ x : ℤ, x^2 = 3*x + 52 ∧ ∀ y : ℤ, y^2 = 3*y + 52 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_52_more_than_triple_l3699_369995


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3699_369939

theorem inequality_equivalence (x : ℝ) :
  (3 ≤ |(x - 3)^2 - 4| ∧ |(x - 3)^2 - 4| ≤ 7) ↔ (3 - Real.sqrt 11 ≤ x ∧ x ≤ 3 + Real.sqrt 11) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3699_369939


namespace NUMINAMATH_CALUDE_total_team_combinations_l3699_369982

/-- The number of ways to choose k items from n items without replacement and without regard to order. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of people in each group. -/
def group_size : ℕ := 6

/-- The number of people to be selected from each group. -/
def team_size : ℕ := 3

/-- The number of groups. -/
def num_groups : ℕ := 2

theorem total_team_combinations : 
  (choose group_size team_size) ^ num_groups = 400 := by sorry

end NUMINAMATH_CALUDE_total_team_combinations_l3699_369982


namespace NUMINAMATH_CALUDE_total_hours_played_l3699_369951

def football_minutes : ℕ := 60
def basketball_minutes : ℕ := 30

def total_minutes : ℕ := football_minutes + basketball_minutes

def minutes_per_hour : ℕ := 60

theorem total_hours_played :
  (total_minutes : ℚ) / minutes_per_hour = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_played_l3699_369951


namespace NUMINAMATH_CALUDE_imaginary_unit_power_2018_l3699_369985

theorem imaginary_unit_power_2018 (i : ℂ) (hi : i^2 = -1) : i^2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_2018_l3699_369985


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3699_369993

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (5 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) = A * Real.sqrt B + C * Real.sqrt D ∧
    B < D ∧
    A = -4 ∧
    B = 7 ∧
    C = 3 ∧
    D = 13 ∧
    E = 1 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3699_369993


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3699_369932

theorem average_of_remaining_numbers
  (total : ℝ)
  (avg_all : ℝ)
  (avg_group1 : ℝ)
  (avg_group2 : ℝ)
  (h1 : total = 6 * avg_all)
  (h2 : avg_all = 3.95)
  (h3 : avg_group1 = 3.6)
  (h4 : avg_group2 = 3.85) :
  (total - 2 * avg_group1 - 2 * avg_group2) / 2 = 4.4 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3699_369932


namespace NUMINAMATH_CALUDE_zeta_sum_sixth_power_l3699_369960

theorem zeta_sum_sixth_power (ζ₁ ζ₂ ζ₃ : ℂ)
  (sum_condition : ζ₁ + ζ₂ + ζ₃ = 2)
  (sum_squares : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (sum_fourth_powers : ζ₁^4 + ζ₂^4 + ζ₃^4 = 29) :
  ζ₁^6 + ζ₂^6 + ζ₃^6 = 101.40625 := by
sorry

end NUMINAMATH_CALUDE_zeta_sum_sixth_power_l3699_369960


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equals_area_l3699_369917

theorem rectangle_perimeter_equals_area (x y : ℕ) : 
  x ≠ y →
  x > 0 →
  y > 0 →
  2 * (x + y) = x * y →
  ((x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_equals_area_l3699_369917


namespace NUMINAMATH_CALUDE_right_triangular_prism_volume_l3699_369968

theorem right_triangular_prism_volume 
  (a b h : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 2) 
  (hh : h = 3) 
  (right_triangle : a * a + b * b = (a + b) * (a + b) / 2) :
  (1 / 2) * a * b * h = 3 := by sorry

end NUMINAMATH_CALUDE_right_triangular_prism_volume_l3699_369968


namespace NUMINAMATH_CALUDE_solve_for_b_l3699_369984

theorem solve_for_b (a b : ℚ) (h1 : a = 5) (h2 : b - a + (2 * b / 3) = 7) : b = 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3699_369984


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3699_369944

theorem polynomial_division_remainder :
  ∃ (q : Polynomial ℝ), x^4 + x^3 - 4*x + 1 = (x^3 - 1) * q + (-3*x + 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3699_369944


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_logic_l3699_369948

theorem quadratic_inequality_and_logic :
  (∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  ¬(∃ x : ℝ, x^2 + x + 1 < 0) ∧
  ((∀ x : ℝ, x^2 + x + 1 ≥ 0) → 
   ((∃ x : ℝ, x^2 + x + 1 < 0) ∨ (∀ x : ℝ, x^2 + x + 1 ≥ 0))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_logic_l3699_369948


namespace NUMINAMATH_CALUDE_total_layers_is_112_l3699_369949

/-- Represents an artist working on a multi-layered painting project -/
structure Artist where
  hours_per_week : ℕ
  hours_per_layer : ℕ

/-- Calculates the number of layers an artist can complete in a given number of weeks -/
def layers_completed (artist : Artist) (weeks : ℕ) : ℕ :=
  (artist.hours_per_week * weeks) / artist.hours_per_layer

/-- The duration of the project in weeks -/
def project_duration : ℕ := 4

/-- The team of artists working on the project -/
def artist_team : List Artist := [
  { hours_per_week := 30, hours_per_layer := 3 },
  { hours_per_week := 40, hours_per_layer := 5 },
  { hours_per_week := 20, hours_per_layer := 2 }
]

/-- Theorem: The total number of layers completed by all artists in the project is 112 -/
theorem total_layers_is_112 : 
  (artist_team.map (λ a => layers_completed a project_duration)).sum = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_layers_is_112_l3699_369949


namespace NUMINAMATH_CALUDE_passage_uses_deductive_reasoning_l3699_369916

/-- Represents a statement in the chain of reasoning --/
inductive Statement
| NamesNotCorrect
| LanguageNotCorrect
| ThingsNotDoneSuccessfully
| RitualsAndMusicNotFlourish
| PunishmentsNotProper
| PeopleConfused

/-- Represents the chain of reasoning in the passage --/
def reasoning_chain : List (Statement × Statement) :=
  [(Statement.NamesNotCorrect, Statement.LanguageNotCorrect),
   (Statement.LanguageNotCorrect, Statement.ThingsNotDoneSuccessfully),
   (Statement.ThingsNotDoneSuccessfully, Statement.RitualsAndMusicNotFlourish),
   (Statement.RitualsAndMusicNotFlourish, Statement.PunishmentsNotProper),
   (Statement.PunishmentsNotProper, Statement.PeopleConfused)]

/-- Definition of deductive reasoning --/
def is_deductive_reasoning (chain : List (Statement × Statement)) : Prop :=
  ∀ (premise conclusion : Statement), 
    (premise, conclusion) ∈ chain → 
    (∃ (general_premise : Statement), 
      (general_premise, premise) ∈ chain ∧ (general_premise, conclusion) ∈ chain)

/-- Theorem stating that the reasoning in the passage is deductive --/
theorem passage_uses_deductive_reasoning : 
  is_deductive_reasoning reasoning_chain :=
sorry

end NUMINAMATH_CALUDE_passage_uses_deductive_reasoning_l3699_369916


namespace NUMINAMATH_CALUDE_expression_simplification_l3699_369959

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) : 
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3699_369959


namespace NUMINAMATH_CALUDE_brittany_age_theorem_l3699_369967

/-- Brittany's age after returning from vacation -/
def brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) : ℕ :=
  rebecca_age + age_difference + vacation_duration

theorem brittany_age_theorem (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) 
  (h1 : rebecca_age = 25)
  (h2 : age_difference = 3)
  (h3 : vacation_duration = 4) :
  brittany_age_after_vacation rebecca_age age_difference vacation_duration = 32 := by
  sorry

end NUMINAMATH_CALUDE_brittany_age_theorem_l3699_369967


namespace NUMINAMATH_CALUDE_triangle_in_circle_and_polygon_l3699_369912

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the regular polygon
structure RegularPolygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ

def is_inscribed (t : Triangle) (c : Circle) : Prop :=
  sorry

def angle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry

def are_adjacent_vertices (p1 p2 : ℝ × ℝ) (poly : RegularPolygon) : Prop :=
  sorry

theorem triangle_in_circle_and_polygon (t : Triangle) (c : Circle) (poly : RegularPolygon) :
  is_inscribed t c →
  angle t.B t.A t.C = angle t.C t.A t.B →
  angle t.B t.A t.C = 3 * angle t.A t.B t.C →
  are_adjacent_vertices t.B t.C poly →
  is_inscribed (Triangle.mk t.A t.B t.C) c →
  poly.n = 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_in_circle_and_polygon_l3699_369912


namespace NUMINAMATH_CALUDE_leading_coefficient_of_P_l3699_369986

/-- The polynomial in question -/
def P (x : ℝ) : ℝ := -5 * (x^4 - 2*x^3 + 3*x) + 8 * (x^4 - x^2 + 1) - 3 * (3*x^4 + x^3 + x)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (p : ℝ → ℝ) : ℝ :=
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_P :
  leadingCoefficient P = -6 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_P_l3699_369986


namespace NUMINAMATH_CALUDE_type_q_machine_time_l3699_369966

theorem type_q_machine_time (q : ℝ) (h1 : q > 0) 
  (h2 : 2 / q + 3 / 7 = 1 / 1.2) : q = 84 / 17 := by
  sorry

end NUMINAMATH_CALUDE_type_q_machine_time_l3699_369966


namespace NUMINAMATH_CALUDE_labeling_periodic_l3699_369974

/-- Represents the labeling of vertices at a given time -/
def Labeling := Fin 1993 → Int

/-- The rule for updating labels -/
def update_label (l : Labeling) (n : Fin 1993) : Int :=
  if l (n - 1) = l (n + 1) then 1 else -1

/-- The next labeling based on the current one -/
def next_labeling (l : Labeling) : Labeling :=
  fun n => update_label l n

/-- The labeling after t steps -/
def labeling_at_time (initial : Labeling) : ℕ → Labeling
  | 0 => initial
  | t + 1 => next_labeling (labeling_at_time initial t)

theorem labeling_periodic (initial : Labeling) :
  ∃ n : ℕ, n > 1 ∧ labeling_at_time initial n = labeling_at_time initial 1 := by
  sorry

end NUMINAMATH_CALUDE_labeling_periodic_l3699_369974


namespace NUMINAMATH_CALUDE_fraction_addition_l3699_369987

theorem fraction_addition (d : ℝ) : (3 + 4 * d) / 5 + 3 = (18 + 4 * d) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3699_369987


namespace NUMINAMATH_CALUDE_expression_evaluation_l3699_369965

theorem expression_evaluation : -25 + 12 * (8 / (2 + 2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3699_369965


namespace NUMINAMATH_CALUDE_smaller_rectangle_area_l3699_369997

/-- The area of a rectangle with half the length and half the width of a 40m by 20m rectangle is 200 square meters. -/
theorem smaller_rectangle_area (big_length big_width : ℝ) 
  (h_big_length : big_length = 40)
  (h_big_width : big_width = 20)
  (small_length small_width : ℝ)
  (h_small_length : small_length = big_length / 2)
  (h_small_width : small_width = big_width / 2) :
  small_length * small_width = 200 := by
  sorry

end NUMINAMATH_CALUDE_smaller_rectangle_area_l3699_369997


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3699_369924

-- Define the curve
def f (x : ℝ) : ℝ := -x^2 + 4

-- Define the point of interest
def x₀ : ℝ := -1

-- Define the slope of the tangent line
def k : ℝ := -2 * x₀

-- Define the y-coordinate of the point on the curve
def y₀ : ℝ := f x₀

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y = k * (x - x₀) + y₀ ↔ y = 2*x + 5 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3699_369924


namespace NUMINAMATH_CALUDE_candy_distribution_l3699_369906

theorem candy_distribution (x : ℕ) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3699_369906


namespace NUMINAMATH_CALUDE_product_of_decimals_l3699_369983

theorem product_of_decimals (h : 268 * 74 = 19832) :
  2.68 * 0.74 = 1.9832 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l3699_369983


namespace NUMINAMATH_CALUDE_det_evaluation_l3699_369925

theorem det_evaluation (x z : ℝ) : 
  Matrix.det !![1, x, z; 1, x + z, 2*z; 1, x, x + 2*z] = z * (3*x + z) := by
  sorry

end NUMINAMATH_CALUDE_det_evaluation_l3699_369925


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3699_369935

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) :
  min x y = 15 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3699_369935


namespace NUMINAMATH_CALUDE_flower_bed_width_l3699_369934

theorem flower_bed_width (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 35 →
  length = 7 →
  area = length * width →
  width = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_flower_bed_width_l3699_369934


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3699_369900

-- Define the sets M and N
def M : Set ℝ := {x | 3 * x - x^2 > 0}
def N : Set ℝ := {x | x^2 - 4 * x + 3 > 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3699_369900


namespace NUMINAMATH_CALUDE_f_2004_value_l3699_369933

/-- A function with the property that f(a) + f(b) = n^3 when a + b = 2^(n+1) -/
def special_function (f : ℕ → ℕ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a + b = 2^(n+1) → f a + f b = n^3

theorem f_2004_value (f : ℕ → ℕ) (h : special_function f) : f 2004 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_f_2004_value_l3699_369933


namespace NUMINAMATH_CALUDE_first_year_balance_l3699_369945

/-- Proves that the total balance at the end of the first year is $5500,
    given the initial deposit of $5000 and the interest accrued in the first year of $500. -/
theorem first_year_balance (initial_deposit : ℝ) (interest_first_year : ℝ) 
  (h1 : initial_deposit = 5000)
  (h2 : interest_first_year = 500) :
  initial_deposit + interest_first_year = 5500 := by
  sorry

end NUMINAMATH_CALUDE_first_year_balance_l3699_369945


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3699_369952

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2 * Complex.I) * z = 5) : 
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3699_369952


namespace NUMINAMATH_CALUDE_calculator_sum_l3699_369990

/-- The number of participants in the circle. -/
def n : ℕ := 44

/-- The operation performed on the first calculator (squaring). -/
def op1 (x : ℕ) : ℕ := x ^ 2

/-- The operation performed on the second calculator (squaring). -/
def op2 (x : ℕ) : ℕ := x ^ 2

/-- The operation performed on the third calculator (negation). -/
def op3 (x : ℤ) : ℤ := -x

/-- The final value of the first calculator after n iterations. -/
def final1 : ℕ := 2 ^ (2 ^ n)

/-- The final value of the second calculator after n iterations. -/
def final2 : ℕ := 0

/-- The final value of the third calculator after n iterations. -/
def final3 : ℤ := (-1) ^ n

/-- The theorem stating the final sum of the calculators. -/
theorem calculator_sum :
  (final1 : ℤ) + final2 + final3 = 2 ^ (2 ^ n) + 1 := by sorry

end NUMINAMATH_CALUDE_calculator_sum_l3699_369990


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3699_369923

theorem sum_with_radical_conjugate :
  let x : ℝ := 12 - Real.sqrt 2023
  let y : ℝ := 12 + Real.sqrt 2023
  x + y = 24 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3699_369923


namespace NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l3699_369929

/-- Represents an alloy with its chromium percentage and weight -/
structure Alloy where
  chromium_percentage : Float
  weight : Float

/-- Calculates the total chromium weight in an alloy -/
def chromium_weight (a : Alloy) : Float :=
  a.chromium_percentage / 100 * a.weight

/-- Calculates the percentage of chromium in a new alloy formed by combining multiple alloys -/
def new_alloy_chromium_percentage (alloys : List Alloy) : Float :=
  let total_chromium : Float := (alloys.map chromium_weight).sum
  let total_weight : Float := (alloys.map (·.weight)).sum
  total_chromium / total_weight * 100

theorem chromium_percentage_in_new_alloy : 
  let a1 : Alloy := { chromium_percentage := 12, weight := 15 }
  let a2 : Alloy := { chromium_percentage := 10, weight := 35 }
  let a3 : Alloy := { chromium_percentage := 8, weight := 25 }
  let a4 : Alloy := { chromium_percentage := 15, weight := 10 }
  let alloys : List Alloy := [a1, a2, a3, a4]
  new_alloy_chromium_percentage alloys = 10.35 := by
  sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l3699_369929


namespace NUMINAMATH_CALUDE_haleys_concert_tickets_l3699_369908

theorem haleys_concert_tickets (ticket_price : ℕ) (extra_tickets : ℕ) (total_spent : ℕ) : 
  ticket_price = 4 → 
  extra_tickets = 5 → 
  total_spent = 32 → 
  ∃ (tickets_for_friends : ℕ), 
    ticket_price * (tickets_for_friends + extra_tickets) = total_spent ∧ 
    tickets_for_friends = 3 :=
by sorry

end NUMINAMATH_CALUDE_haleys_concert_tickets_l3699_369908


namespace NUMINAMATH_CALUDE_fifth_teapot_volume_l3699_369954

theorem fifth_teapot_volume
  (a : ℕ → ℚ)  -- arithmetic sequence of rational numbers
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence condition
  (h_length : ∀ n, n ≥ 9 → a n = a 9)  -- sequence has 9 terms
  (h_sum_first_three : a 1 + a 2 + a 3 = 1/2)  -- sum of first three terms
  (h_sum_last_three : a 7 + a 8 + a 9 = 5/2)  -- sum of last three terms
  : a 5 = 1/2 := by sorry

end NUMINAMATH_CALUDE_fifth_teapot_volume_l3699_369954


namespace NUMINAMATH_CALUDE_parabola_coeff_sum_l3699_369950

/-- A parabola with equation y = ax^2 + bx + c, vertex at (-3, 2), and passing through (1, 6) -/
def Parabola (a b c : ℝ) : Prop :=
  (∀ x y : ℝ, y = a * x^2 + b * x + c) ∧
  (2 = a * (-3)^2 + b * (-3) + c) ∧
  (6 = a * 1^2 + b * 1 + c)

/-- The sum of coefficients a, b, and c equals 6 -/
theorem parabola_coeff_sum (a b c : ℝ) (h : Parabola a b c) : a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coeff_sum_l3699_369950


namespace NUMINAMATH_CALUDE_triangle_side_length_l3699_369907

/-- Represents a triangle with sides a, b, c and median ma from vertex A to midpoint of side BC. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ma : ℝ

/-- The theorem states that for a triangle with sides 6 and 9, and a median of 5,
    the third side has length √134. -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 6)
  (h2 : t.b = 9) 
  (h3 : t.ma = 5) : 
  t.c = Real.sqrt 134 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3699_369907


namespace NUMINAMATH_CALUDE_continued_fraction_sum_l3699_369902

theorem continued_fraction_sum (a b c d : ℕ+) :
  (147 : ℚ) / 340 = 1 / (a + 1 / (b + 1 / (c + 1 / d))) →
  (a : ℕ) + b + c + d = 19 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_sum_l3699_369902
