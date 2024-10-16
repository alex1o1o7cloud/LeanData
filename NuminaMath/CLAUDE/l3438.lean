import Mathlib

namespace NUMINAMATH_CALUDE_largest_multiple_18_with_9_0_m_div_18_eq_555_l3438_343838

/-- A function that checks if a natural number consists only of digits 9 and 0 -/
def onlyNineAndZero (n : ℕ) : Prop := sorry

/-- The largest positive multiple of 18 consisting only of digits 9 and 0 -/
def m : ℕ := sorry

theorem largest_multiple_18_with_9_0 :
  m > 0 ∧
  m % 18 = 0 ∧
  onlyNineAndZero m ∧
  ∀ k : ℕ, k > m → (k % 18 = 0 → ¬onlyNineAndZero k) :=
sorry

theorem m_div_18_eq_555 : m / 18 = 555 := sorry

end NUMINAMATH_CALUDE_largest_multiple_18_with_9_0_m_div_18_eq_555_l3438_343838


namespace NUMINAMATH_CALUDE_height_ratio_of_cones_l3438_343854

/-- The ratio of heights of two right circular cones with the same base circumference -/
theorem height_ratio_of_cones (r : ℝ) (h₁ h₂ : ℝ) :
  r > 0 → h₁ > 0 → h₂ > 0 →
  (2 * Real.pi * r = 20 * Real.pi) →
  ((1/3) * Real.pi * r^2 * h₁ = 400 * Real.pi) →
  h₂ = 40 →
  h₁ / h₂ = 3/10 := by
sorry

end NUMINAMATH_CALUDE_height_ratio_of_cones_l3438_343854


namespace NUMINAMATH_CALUDE_birdhouse_distance_l3438_343888

/-- The distance flown by objects in a tornado scenario -/
def tornado_scenario (car_distance : ℕ) : Prop :=
  let lawn_chair_distance := 2 * car_distance
  let birdhouse_distance := 3 * lawn_chair_distance
  car_distance = 200 ∧ birdhouse_distance = 1200

/-- Theorem stating that in the given scenario, the birdhouse flew 1200 feet -/
theorem birdhouse_distance : tornado_scenario 200 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_distance_l3438_343888


namespace NUMINAMATH_CALUDE_kitty_cleanup_time_l3438_343859

/-- Represents the weekly cleaning routine for a living room -/
structure CleaningRoutine where
  pickup_time : ℕ  -- Time spent picking up toys and straightening
  vacuum_time : ℕ  -- Time spent vacuuming
  window_time : ℕ  -- Time spent cleaning windows
  dusting_time : ℕ  -- Time spent dusting furniture

/-- Calculates the total cleaning time for a given number of weeks -/
def total_cleaning_time (routine : CleaningRoutine) (weeks : ℕ) : ℕ :=
  weeks * (routine.pickup_time + routine.vacuum_time + routine.window_time + routine.dusting_time)

theorem kitty_cleanup_time :
  ∃ (routine : CleaningRoutine),
    routine.vacuum_time = 20 ∧
    routine.window_time = 15 ∧
    routine.dusting_time = 10 ∧
    total_cleaning_time routine 4 = 200 ∧
    routine.pickup_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_kitty_cleanup_time_l3438_343859


namespace NUMINAMATH_CALUDE_exists_m_for_even_f_l3438_343813

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = x^2 + mx for some m ∈ ℝ -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

/-- There exists an m ∈ ℝ such that f(x) = x^2 + mx is an even function -/
theorem exists_m_for_even_f : ∃ m : ℝ, IsEven (f m) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_for_even_f_l3438_343813


namespace NUMINAMATH_CALUDE_modular_congruence_in_range_l3438_343811

theorem modular_congruence_in_range : ∃ n : ℤ, 5 ≤ n ∧ n ≤ 12 ∧ n ≡ 10569 [ZMOD 7] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_in_range_l3438_343811


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l3438_343853

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 → 
  Nat.gcd A B = 30 → 
  A = 231 → 
  B = 300 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l3438_343853


namespace NUMINAMATH_CALUDE_solve_journey_problem_l3438_343804

def journey_problem (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : Prop :=
  let half_distance := (total_time * speed1 * speed2) / (speed1 + speed2)
  total_time = half_distance / speed1 + half_distance / speed2 →
  2 * half_distance = 240

theorem solve_journey_problem :
  journey_problem 20 10 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_journey_problem_l3438_343804


namespace NUMINAMATH_CALUDE_min_equal_number_example_min_equal_number_is_minimum_l3438_343833

/-- Given three initial numbers on a blackboard, this function represents
    the minimum number to which all three can be made equal by repeatedly
    selecting two numbers and adding 1 to each. -/
def min_equal_number (a b c : ℕ) : ℕ :=
  (a + b + c + 2 * ((a + b + c) % 3)) / 3

/-- Theorem stating that 747 is the minimum number to which 20, 201, and 2016
    can be made equal using the described operation. -/
theorem min_equal_number_example : min_equal_number 20 201 2016 = 747 := by
  sorry

/-- Theorem stating that the result of min_equal_number is indeed the minimum
    possible number to which the initial numbers can be made equal. -/
theorem min_equal_number_is_minimum (a b c : ℕ) :
  ∀ n : ℕ, (∃ k : ℕ, a + k ≤ n ∧ b + k ≤ n ∧ c + k ≤ n) →
  min_equal_number a b c ≤ n := by
  sorry

end NUMINAMATH_CALUDE_min_equal_number_example_min_equal_number_is_minimum_l3438_343833


namespace NUMINAMATH_CALUDE_perpendicular_vectors_collinear_vectors_l3438_343851

def vector_a (x : ℝ) : ℝ × ℝ := (3, x)
def vector_b : ℝ × ℝ := (-2, 2)

theorem perpendicular_vectors (x : ℝ) :
  (vector_a x).1 * vector_b.1 + (vector_a x).2 * vector_b.2 = 0 → x = 3 := by sorry

theorem collinear_vectors (x : ℝ) :
  ∃ (k : ℝ), k ≠ 0 ∧ 
  (vector_b.1 - (vector_a x).1, vector_b.2 - (vector_a x).2) = 
  k • (3 * (vector_a x).1 + 2 * vector_b.1, 3 * (vector_a x).2 + 2 * vector_b.2) 
  → x = -3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_collinear_vectors_l3438_343851


namespace NUMINAMATH_CALUDE_decimal_to_binary_38_l3438_343868

theorem decimal_to_binary_38 : 
  (38 : ℕ).digits 2 = [0, 1, 1, 0, 0, 1] :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_38_l3438_343868


namespace NUMINAMATH_CALUDE_radical_simplification_l3438_343866

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (20 * q^3) * Real.sqrt (12 * q^5) = 60 * q^4 * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l3438_343866


namespace NUMINAMATH_CALUDE_max_non_multiples_of_three_l3438_343885

/-- Given a list of 6 positive integers whose product is a multiple of 3,
    the maximum number of integers in the list that are not multiples of 3 is 5. -/
theorem max_non_multiples_of_three (integers : List ℕ+) : 
  integers.length = 6 → 
  integers.prod.val % 3 = 0 → 
  (integers.filter (fun x => x.val % 3 ≠ 0)).length ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_non_multiples_of_three_l3438_343885


namespace NUMINAMATH_CALUDE_ball_placement_events_l3438_343835

structure Ball :=
  (number : Nat)

structure Box :=
  (number : Nat)

def Placement := Ball → Box

def event_ball1_in_box1 (p : Placement) : Prop :=
  p ⟨1⟩ = ⟨1⟩

def event_ball1_in_box2 (p : Placement) : Prop :=
  p ⟨1⟩ = ⟨2⟩

def mutually_exclusive (e1 e2 : Placement → Prop) : Prop :=
  ∀ p : Placement, ¬(e1 p ∧ e2 p)

def complementary (e1 e2 : Placement → Prop) : Prop :=
  ∀ p : Placement, e1 p ↔ ¬(e2 p)

theorem ball_placement_events :
  (mutually_exclusive event_ball1_in_box1 event_ball1_in_box2) ∧
  ¬(complementary event_ball1_in_box1 event_ball1_in_box2) :=
sorry

end NUMINAMATH_CALUDE_ball_placement_events_l3438_343835


namespace NUMINAMATH_CALUDE_package_not_qualified_l3438_343898

/-- The standard net weight of a biscuit in grams -/
def standard_weight : ℝ := 350

/-- The acceptable deviation from the standard weight in grams -/
def acceptable_deviation : ℝ := 5

/-- The weight of the package in question in grams -/
def package_weight : ℝ := 358

/-- A package is qualified if its weight is within the acceptable range -/
def is_qualified (weight : ℝ) : Prop :=
  (standard_weight - acceptable_deviation ≤ weight) ∧
  (weight ≤ standard_weight + acceptable_deviation)

/-- Theorem stating that the package with weight 358 grams is not qualified -/
theorem package_not_qualified : ¬(is_qualified package_weight) := by
  sorry

end NUMINAMATH_CALUDE_package_not_qualified_l3438_343898


namespace NUMINAMATH_CALUDE_pencil_count_l3438_343895

/-- The total number of pencils in a drawer after adding more pencils -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: Given 27 initial pencils and 45 added pencils, the total is 72 -/
theorem pencil_count : total_pencils 27 45 = 72 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3438_343895


namespace NUMINAMATH_CALUDE_candle_flower_groupings_l3438_343832

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem candle_flower_groupings :
  (choose 4 2) * (choose 9 8) = 54 := by
  sorry

end NUMINAMATH_CALUDE_candle_flower_groupings_l3438_343832


namespace NUMINAMATH_CALUDE_rectangular_field_shortcut_l3438_343869

theorem rectangular_field_shortcut (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x < y) :
  x + y - Real.sqrt (x^2 + y^2) = x →
  y / x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_shortcut_l3438_343869


namespace NUMINAMATH_CALUDE_utopia_national_park_elephant_rate_l3438_343872

/-- Proves that the rate of new elephants entering Utopia National Park is 1500 per hour --/
theorem utopia_national_park_elephant_rate : 
  let initial_elephants : ℕ := 30000
  let exodus_duration : ℕ := 4
  let exodus_rate : ℕ := 2880
  let new_elephants_duration : ℕ := 7
  let final_elephants : ℕ := 28980
  
  let elephants_after_exodus := initial_elephants - exodus_duration * exodus_rate
  let new_elephants := final_elephants - elephants_after_exodus
  let new_elephants_rate := new_elephants / new_elephants_duration
  
  new_elephants_rate = 1500 := by
  sorry

end NUMINAMATH_CALUDE_utopia_national_park_elephant_rate_l3438_343872


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l3438_343826

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  (2 * a 3 = a 1 + a 2) →           -- arithmetic sequence condition
  (q = 1 ∨ q = -1/2) :=             -- conclusion
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l3438_343826


namespace NUMINAMATH_CALUDE_regression_sum_of_squares_l3438_343821

theorem regression_sum_of_squares 
  (SST : ℝ) (SSE : ℝ) (SSR : ℝ) 
  (h1 : SST = 256) 
  (h2 : SSE = 32) 
  (h3 : SSR = SST - SSE) : 
  SSR = 224 := by
  sorry

end NUMINAMATH_CALUDE_regression_sum_of_squares_l3438_343821


namespace NUMINAMATH_CALUDE_appropriate_citizen_actions_l3438_343823

/-- Represents the current state of the cultural market -/
structure CulturalMarket where
  entertainment_trend : Bool
  vulgarity_trend : Bool

/-- Represents possible actions citizens can take -/
inductive CitizenAction
  | choose_personality_trends
  | improve_distinction_ability
  | enhance_aesthetic_taste
  | pursue_high_end_culture

/-- Determines if an action is appropriate given the cultural market state -/
def is_appropriate_action (market : CulturalMarket) (action : CitizenAction) : Prop :=
  match action with
  | CitizenAction.improve_distinction_ability => true
  | CitizenAction.enhance_aesthetic_taste => true
  | _ => false

/-- Theorem stating the most appropriate actions for citizens -/
theorem appropriate_citizen_actions (market : CulturalMarket) 
  (h_entertainment : market.entertainment_trend = true)
  (h_vulgarity : market.vulgarity_trend = true) :
  (∀ action : CitizenAction, is_appropriate_action market action ↔ 
    (action = CitizenAction.improve_distinction_ability ∨ 
     action = CitizenAction.enhance_aesthetic_taste)) :=
by sorry

end NUMINAMATH_CALUDE_appropriate_citizen_actions_l3438_343823


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3438_343837

/-- Given a circle with equation x^2 - 2x + y^2 = 3, its symmetric circle
    with respect to the y-axis has the equation x^2 + 2x + y^2 = 3 -/
theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 - 2*x + y^2 = 3) →
  ((-x)^2 - 2*(-x) + y^2 = 3) →
  (x^2 + 2*x + y^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3438_343837


namespace NUMINAMATH_CALUDE_toilet_paper_squares_per_roll_l3438_343850

theorem toilet_paper_squares_per_roll 
  (daily_visits : ℕ) 
  (squares_per_visit : ℕ) 
  (total_rolls : ℕ) 
  (days_supply_lasts : ℕ) 
  (h1 : daily_visits = 3) 
  (h2 : squares_per_visit = 5) 
  (h3 : total_rolls = 1000) 
  (h4 : days_supply_lasts = 20000) :
  (daily_visits * squares_per_visit * days_supply_lasts) / total_rolls = 300 := by
  sorry

end NUMINAMATH_CALUDE_toilet_paper_squares_per_roll_l3438_343850


namespace NUMINAMATH_CALUDE_perimeter_invariant_under_translation_l3438_343816

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a hexagon formed by the intersection of two equilateral triangles -/
structure IntersectionHexagon where
  triangle1 : EquilateralTriangle
  triangle2 : EquilateralTriangle

/-- Calculates the perimeter of the intersection hexagon -/
def perimeter (h : IntersectionHexagon) : ℝ :=
  sorry

/-- Represents a parallel translation of a triangle -/
def parallelTranslation (t : EquilateralTriangle) (v : ℝ × ℝ) : EquilateralTriangle :=
  sorry

/-- The theorem stating that the perimeter remains constant under parallel translation -/
theorem perimeter_invariant_under_translation 
  (h : IntersectionHexagon) 
  (v : ℝ × ℝ) 
  (h' : IntersectionHexagon := ⟨h.triangle1, parallelTranslation h.triangle2 v⟩) : 
  perimeter h = perimeter h' :=
sorry

end NUMINAMATH_CALUDE_perimeter_invariant_under_translation_l3438_343816


namespace NUMINAMATH_CALUDE_extended_triangle_pc_length_l3438_343897

/-- Triangle ABC with sides AB, BC, CA, extended to point P -/
structure ExtendedTriangle where
  AB : ℝ
  BC : ℝ
  CA : ℝ
  PC : ℝ

/-- Similarity of triangles PAB and PCA -/
def similar_triangles (t : ExtendedTriangle) : Prop :=
  t.PC / (t.PC + t.BC) = t.CA / t.AB

theorem extended_triangle_pc_length 
  (t : ExtendedTriangle) 
  (h1 : t.AB = 10) 
  (h2 : t.BC = 9) 
  (h3 : t.CA = 7) 
  (h4 : similar_triangles t) : 
  t.PC = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_extended_triangle_pc_length_l3438_343897


namespace NUMINAMATH_CALUDE_coin_division_problem_l3438_343870

theorem coin_division_problem :
  ∃ n : ℕ, 
    n > 0 ∧
    n % 8 = 5 ∧
    n % 7 = 4 ∧
    (∀ m : ℕ, m > 0 → m % 8 = 5 → m % 7 = 4 → n ≤ m) ∧
    n % 9 = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_coin_division_problem_l3438_343870


namespace NUMINAMATH_CALUDE_product_of_fractions_l3438_343857

theorem product_of_fractions : (1 : ℚ) / 3 * 3 / 5 * 5 / 7 * 7 / 9 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3438_343857


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l3438_343890

theorem quadratic_equation_condition (a : ℝ) :
  (∀ x, (a - 2) * x^2 + a * x - 3 = 0 → (a - 2) ≠ 0) ↔ a ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l3438_343890


namespace NUMINAMATH_CALUDE_no_real_solutions_l3438_343878

theorem no_real_solutions : ¬ ∃ x : ℝ, Real.sqrt ((x^2 - 2*x + 1) + 1) = -x := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3438_343878


namespace NUMINAMATH_CALUDE_only_paint_worthy_is_204_l3438_343893

/-- Represents a painting configuration for the fence. -/
structure PaintConfig where
  h : ℕ+  -- Harold's interval
  t : ℕ+  -- Tanya's interval
  u : ℕ+  -- Ulysses' interval

/-- Checks if a painting configuration is valid (covers all pickets exactly once). -/
def isValidConfig (config : PaintConfig) : Prop :=
  -- Harold starts from second picket
  -- Tanya starts from first picket
  -- Ulysses starts from fourth picket
  -- Each picket is painted exactly once
  sorry

/-- Calculates the paint-worthy number for a given configuration. -/
def paintWorthy (config : PaintConfig) : ℕ :=
  100 * config.h.val + 10 * config.t.val + config.u.val

/-- The main theorem stating that 204 is the only paint-worthy number. -/
theorem only_paint_worthy_is_204 :
  ∀ config : PaintConfig, isValidConfig config → paintWorthy config = 204 := by
  sorry

end NUMINAMATH_CALUDE_only_paint_worthy_is_204_l3438_343893


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3438_343815

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => x * (2 * x + 4) - (10 + 5 * x)
  ∀ x : ℝ, f x = 0 ↔ x = -2 ∨ x = 5/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3438_343815


namespace NUMINAMATH_CALUDE_square_sum_and_difference_l3438_343822

/-- Given a = √3 - 2 and b = √3 + 2, prove that (a + b)² = 12 and a² - b² = -8√3 -/
theorem square_sum_and_difference (a b : ℝ) (ha : a = Real.sqrt 3 - 2) (hb : b = Real.sqrt 3 + 2) :
  (a + b)^2 = 12 ∧ a^2 - b^2 = -8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_square_sum_and_difference_l3438_343822


namespace NUMINAMATH_CALUDE_three_pump_fill_time_l3438_343825

/-- Represents the time taken (in hours) for three pumps to fill a tank when working together. -/
def combined_fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating that three pumps with given rates will fill a tank in 6/29 hours. -/
theorem three_pump_fill_time :
  combined_fill_time (1/3) 4 (1/2) = 6/29 := by
  sorry

#eval combined_fill_time (1/3) 4 (1/2)

end NUMINAMATH_CALUDE_three_pump_fill_time_l3438_343825


namespace NUMINAMATH_CALUDE_geometric_series_solution_l3438_343889

theorem geometric_series_solution (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (6 * n - 2) / k^n = 3) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_solution_l3438_343889


namespace NUMINAMATH_CALUDE_tylers_sanctuary_pairs_l3438_343802

/-- Represents the animal sanctuary with three regions -/
structure AnimalSanctuary where
  bird_species : ℕ
  bird_pairs_per_species : ℕ
  marine_species : ℕ
  marine_pairs_per_species : ℕ
  mammal_species : ℕ
  mammal_pairs_per_species : ℕ

/-- Calculates the total number of pairs in the sanctuary -/
def total_pairs (sanctuary : AnimalSanctuary) : ℕ :=
  sanctuary.bird_species * sanctuary.bird_pairs_per_species +
  sanctuary.marine_species * sanctuary.marine_pairs_per_species +
  sanctuary.mammal_species * sanctuary.mammal_pairs_per_species

/-- Theorem stating that the total number of pairs in Tyler's sanctuary is 470 -/
theorem tylers_sanctuary_pairs :
  let tyler_sanctuary : AnimalSanctuary := {
    bird_species := 29,
    bird_pairs_per_species := 7,
    marine_species := 15,
    marine_pairs_per_species := 9,
    mammal_species := 22,
    mammal_pairs_per_species := 6
  }
  total_pairs tyler_sanctuary = 470 := by
  sorry

end NUMINAMATH_CALUDE_tylers_sanctuary_pairs_l3438_343802


namespace NUMINAMATH_CALUDE_f_property_P_implies_m_range_l3438_343805

/-- Property P(a) for a function f on domain D -/
def property_P (f : ℝ → ℝ) (D : Set ℝ) (a : ℝ) : Prop :=
  ∀ x₁ ∈ D, ∃ x₂ ∈ D, (x₁ + f x₂) / 2 = a

/-- The function f(x) = -x² + mx - 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - 3

/-- The domain of f(x) -/
def D : Set ℝ := {x : ℝ | x > 0}

theorem f_property_P_implies_m_range :
  ∀ m : ℝ, property_P (f m) D (1/2) → m ∈ {m : ℝ | m ≥ 4} := by sorry

end NUMINAMATH_CALUDE_f_property_P_implies_m_range_l3438_343805


namespace NUMINAMATH_CALUDE_square_difference_65_35_l3438_343877

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_65_35_l3438_343877


namespace NUMINAMATH_CALUDE_exists_dry_student_l3438_343899

/-- A student in the water gun game -/
structure Student where
  id : ℕ
  position : ℝ × ℝ

/-- The state of the water gun game -/
structure WaterGunGame where
  n : ℕ
  students : Fin (2 * n + 1) → Student
  distinct_distances : ∀ i j k l : Fin (2 * n + 1), 
    i ≠ j → k ≠ l → (i, j) ≠ (k, l) → 
    dist (students i).position (students j).position ≠ 
    dist (students k).position (students l).position

/-- The shooting function: each student shoots their closest neighbor -/
def shoot (game : WaterGunGame) (shooter : Fin (2 * game.n + 1)) : Fin (2 * game.n + 1) :=
  sorry

/-- The main theorem: there exists a dry student -/
theorem exists_dry_student (game : WaterGunGame) : 
  ∃ s : Fin (2 * game.n + 1), ∀ t : Fin (2 * game.n + 1), shoot game t ≠ s :=
sorry

end NUMINAMATH_CALUDE_exists_dry_student_l3438_343899


namespace NUMINAMATH_CALUDE_garden_tulips_calculation_l3438_343894

-- Define the initial state of the garden
def initial_daisies : ℕ := 32
def ratio_tulips : ℕ := 3
def ratio_daisies : ℕ := 4
def added_daisies : ℕ := 8

-- Define the function to calculate tulips based on daisies and ratio
def calculate_tulips (daisies : ℕ) : ℕ :=
  (daisies * ratio_tulips) / ratio_daisies

-- Theorem statement
theorem garden_tulips_calculation :
  let initial_tulips := calculate_tulips initial_daisies
  let final_daisies := initial_daisies + added_daisies
  let final_tulips := calculate_tulips final_daisies
  let additional_tulips := final_tulips - initial_tulips
  (additional_tulips = 6) ∧ (final_tulips = 30) := by
  sorry

end NUMINAMATH_CALUDE_garden_tulips_calculation_l3438_343894


namespace NUMINAMATH_CALUDE_cookies_left_for_sonny_l3438_343855

theorem cookies_left_for_sonny (total : ℕ) (brother sister cousin : ℕ) 
  (h1 : total = 45)
  (h2 : brother = 12)
  (h3 : sister = 9)
  (h4 : cousin = 7) :
  total - (brother + sister + cousin) = 17 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_for_sonny_l3438_343855


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3438_343817

/-- The equation of a potential ellipse -/
def is_potential_ellipse (m n x y : ℝ) : Prop :=
  x^2 / m + y^2 / n = 1

/-- The condition mn > 0 -/
def condition_mn_positive (m n : ℝ) : Prop :=
  m * n > 0

/-- Definition of an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

theorem condition_necessary_not_sufficient :
  (∀ m n : ℝ, is_ellipse m n → condition_mn_positive m n) ∧
  (∃ m n : ℝ, condition_mn_positive m n ∧ ¬is_ellipse m n) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3438_343817


namespace NUMINAMATH_CALUDE_adrian_days_off_l3438_343887

/-- The number of days Adrian takes off per month for personal reasons -/
def personal_days_per_month : ℕ := 4

/-- The number of days Adrian takes off per month for professional development -/
def professional_days_per_month : ℕ := 2

/-- The number of days Adrian takes off per quarter for team-building events -/
def team_building_days_per_quarter : ℕ := 1

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The number of quarters in a year -/
def quarters_per_year : ℕ := 4

/-- The total number of days Adrian takes off in a year -/
def total_days_off : ℕ :=
  personal_days_per_month * months_per_year +
  professional_days_per_month * months_per_year +
  team_building_days_per_quarter * quarters_per_year

theorem adrian_days_off : total_days_off = 76 := by
  sorry

end NUMINAMATH_CALUDE_adrian_days_off_l3438_343887


namespace NUMINAMATH_CALUDE_volunteer_schedule_lcm_l3438_343882

theorem volunteer_schedule_lcm : Nat.lcm 2 (Nat.lcm 5 (Nat.lcm 9 11)) = 990 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_lcm_l3438_343882


namespace NUMINAMATH_CALUDE_circle_radii_ratio_l3438_343867

theorem circle_radii_ratio (A₁ A₂ r₁ r₂ : ℝ) (h_area_ratio : A₁ / A₂ = 98 / 63)
  (h_area_formula₁ : A₁ = π * r₁^2) (h_area_formula₂ : A₂ = π * r₂^2) :
  ∃ (x y z : ℕ), (r₁ / r₂ = x * Real.sqrt y / z) ∧ (x * Real.sqrt y / z = Real.sqrt 14 / 3) ∧ x + y + z = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_radii_ratio_l3438_343867


namespace NUMINAMATH_CALUDE_area_between_tangent_circles_l3438_343883

/-- The area of the region between two tangent circles -/
theorem area_between_tangent_circles 
  (r₁ : ℝ) -- radius of the inner circle
  (d : ℝ)  -- distance between the centers of the circles
  (h₁ : r₁ = 5) -- given radius of inner circle
  (h₂ : d = 3)  -- given distance between centers
  : (π * ((r₁ + d)^2 - r₁^2) : ℝ) = 39 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_tangent_circles_l3438_343883


namespace NUMINAMATH_CALUDE_student_weight_loss_l3438_343874

/-- The amount of weight a student needs to lose to weigh twice as much as his sister. -/
def weight_to_lose (total_weight sister_weight : ℝ) : ℝ :=
  total_weight - sister_weight - 2 * sister_weight

theorem student_weight_loss (total_weight student_weight : ℝ) 
  (h1 : total_weight = 104)
  (h2 : student_weight = 71) :
  weight_to_lose total_weight (total_weight - student_weight) = 5 := by
  sorry

#eval weight_to_lose 104 33

end NUMINAMATH_CALUDE_student_weight_loss_l3438_343874


namespace NUMINAMATH_CALUDE_valentines_to_cinco_l3438_343812

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a year -/
structure Date where
  month : Nat
  day : Nat

def valentinesDay : Date := ⟨2, 14⟩
def cincoMayo : Date := ⟨5, 5⟩

/-- Given that February 14 is a Tuesday, calculate the day of the week for any date -/
def dayOfWeek (d : Date) : DayOfWeek := sorry

/-- Calculate the number of days between two dates, inclusive -/
def daysBetween (d1 d2 : Date) : Nat := sorry

theorem valentines_to_cinco : 
  dayOfWeek valentinesDay = DayOfWeek.Tuesday →
  (dayOfWeek cincoMayo = DayOfWeek.Friday ∧ 
   daysBetween valentinesDay cincoMayo = 81) := by
  sorry

end NUMINAMATH_CALUDE_valentines_to_cinco_l3438_343812


namespace NUMINAMATH_CALUDE_ice_cream_line_problem_l3438_343818

def Line := Fin 5 → Fin 5

def is_valid_line (l : Line) : Prop :=
  (∀ i j, i ≠ j → l i ≠ l j) ∧
  (∃ i, l i = 0) ∧
  (∃ i, l i = 1) ∧
  (∃ i, l i = 2) ∧
  (∃ i, l i = 3) ∧
  (∃ i, l i = 4)

theorem ice_cream_line_problem (l : Line) 
  (h_valid : is_valid_line l)
  (h_A_first : ∃ i, l i = 0)
  (h_B_next_to_A : ∃ i j, l i = 0 ∧ l j = 1 ∧ (i.val + 1 = j.val ∨ j.val + 1 = i.val))
  (h_C_second_to_last : ∃ i, l i = 3)
  (h_D_last : ∃ i, l i = 4)
  (h_E_remaining : ∃ i, l i = 2) :
  ∃ i j, l i = 2 ∧ l j = 1 ∧ (i.val = j.val + 1 ∨ j.val = i.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_line_problem_l3438_343818


namespace NUMINAMATH_CALUDE_average_playtime_l3438_343852

def wednesday_hours : ℝ := 2
def thursday_hours : ℝ := 2
def friday_additional_hours : ℝ := 3
def total_days : ℕ := 3

theorem average_playtime :
  let total_hours := wednesday_hours + thursday_hours + (wednesday_hours + friday_additional_hours)
  total_hours / total_days = 3 := by
sorry

end NUMINAMATH_CALUDE_average_playtime_l3438_343852


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l3438_343848

theorem cube_plus_reciprocal_cube (r : ℝ) (hr : r ≠ 0) 
  (h : (r + 1/r)^2 = 5) : 
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l3438_343848


namespace NUMINAMATH_CALUDE_trigonometric_sum_trigonometric_fraction_l3438_343886

-- Part 1
theorem trigonometric_sum : 
  Real.cos (9 * Real.pi / 4) + Real.tan (-Real.pi / 4) + Real.sin (21 * Real.pi) = Real.sqrt 2 / 2 - 1 :=
by sorry

-- Part 2
theorem trigonometric_fraction (θ : Real) (h : Real.sin θ = 2 * Real.cos θ) : 
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_sum_trigonometric_fraction_l3438_343886


namespace NUMINAMATH_CALUDE_lukes_coin_piles_l3438_343873

/-- Given that Luke has an equal number of piles of quarters and dimes,
    each pile contains 3 coins, and the total number of coins is 30,
    prove that the number of piles of quarters is 5. -/
theorem lukes_coin_piles (num_quarter_piles num_dime_piles : ℕ)
  (h1 : num_quarter_piles = num_dime_piles)
  (h2 : ∀ pile, pile = num_quarter_piles ∨ pile = num_dime_piles → 3 * pile = num_quarter_piles * 3 + num_dime_piles * 3)
  (h3 : num_quarter_piles * 3 + num_dime_piles * 3 = 30) :
  num_quarter_piles = 5 := by
  sorry

end NUMINAMATH_CALUDE_lukes_coin_piles_l3438_343873


namespace NUMINAMATH_CALUDE_problem_solution_l3438_343803

theorem problem_solution (a : ℝ) : 3 ∈ ({1, -a^2, a-1} : Set ℝ) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3438_343803


namespace NUMINAMATH_CALUDE_fertilizer_calculation_l3438_343827

theorem fertilizer_calculation (total_area : ℝ) (partial_area : ℝ) (partial_fertilizer : ℝ) 
  (h1 : total_area = 7200)
  (h2 : partial_area = 3600)
  (h3 : partial_fertilizer = 600) :
  (total_area / partial_area) * partial_fertilizer = 1200 := by
sorry

end NUMINAMATH_CALUDE_fertilizer_calculation_l3438_343827


namespace NUMINAMATH_CALUDE_range_of_a_when_union_is_reals_l3438_343800

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | -4 + a < x ∧ x < 4 + a}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 5}

-- Theorem statement
theorem range_of_a_when_union_is_reals :
  ∀ a : ℝ, (A a ∪ B = Set.univ) ↔ (1 < a ∧ a < 3) := by sorry

end NUMINAMATH_CALUDE_range_of_a_when_union_is_reals_l3438_343800


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3438_343820

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3438_343820


namespace NUMINAMATH_CALUDE_function_is_identity_l3438_343844

def is_positive (n : ℕ) : Prop := n > 0

def satisfies_functional_equation (f : ℕ → ℕ) : Prop :=
  ∀ m n, is_positive m → is_positive n → f (f m + f n) = m + n

theorem function_is_identity 
  (f : ℕ → ℕ) 
  (h : satisfies_functional_equation f) :
  ∀ x, is_positive x → f x = x :=
sorry

end NUMINAMATH_CALUDE_function_is_identity_l3438_343844


namespace NUMINAMATH_CALUDE_ellipse_equation_l3438_343810

/-- An ellipse with center at the origin, right focus at (1,0), and eccentricity 1/2 -/
structure Ellipse where
  /-- The x-coordinate of a point on the ellipse -/
  x : ℝ
  /-- The y-coordinate of a point on the ellipse -/
  y : ℝ
  /-- The distance from the center to the focus -/
  c : ℝ
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The semi-major axis of the ellipse -/
  a : ℝ
  /-- The semi-minor axis of the ellipse -/
  b : ℝ
  /-- The center is at the origin -/
  center_origin : c = 1
  /-- The eccentricity is 1/2 -/
  eccentricity_half : e = 1/2
  /-- The relation between eccentricity, c, and a -/
  eccentricity_def : e = c / a
  /-- The relation between a, b, and c -/
  axis_relation : b^2 = a^2 - c^2

/-- The equation of the ellipse is x^2/4 + y^2/3 = 1 -/
theorem ellipse_equation (C : Ellipse) : C.x^2 / 4 + C.y^2 / 3 = 1 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3438_343810


namespace NUMINAMATH_CALUDE_min_value_g_l3438_343864

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points -/
def distance (p q : Point3D) : ℝ := sorry

/-- Definition of the tetrahedron EFGH -/
def Tetrahedron (E F G H : Point3D) : Prop :=
  distance E H = 30 ∧
  distance F G = 30 ∧
  distance E G = 40 ∧
  distance F H = 40 ∧
  distance E F = 48 ∧
  distance G H = 48

/-- Function g(Y) as defined in the problem -/
def g (E F G H Y : Point3D) : ℝ :=
  distance E Y + distance F Y + distance G Y + distance H Y

/-- Theorem stating the minimum value of g(Y) -/
theorem min_value_g (E F G H : Point3D) :
  Tetrahedron E F G H →
  ∃ (min : ℝ), min = 4 * Real.sqrt 578 ∧
    ∀ (Y : Point3D), g E F G H Y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_g_l3438_343864


namespace NUMINAMATH_CALUDE_smallest_cube_for_pyramid_l3438_343847

/-- Represents a triangular pyramid with a given height and base side length -/
structure TriangularPyramid where
  height : ℝ
  baseSide : ℝ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ :=
  c.sideLength ^ 3

/-- Checks if a cube can contain a triangular pyramid upright -/
def canContainPyramid (c : Cube) (p : TriangularPyramid) : Prop :=
  c.sideLength ≥ p.height ∧ c.sideLength ≥ p.baseSide

/-- The main theorem statement -/
theorem smallest_cube_for_pyramid (p : TriangularPyramid)
    (h1 : p.height = 15)
    (h2 : p.baseSide = 12) :
    ∃ (c : Cube), canContainPyramid c p ∧
    cubeVolume c = 3375 ∧
    ∀ (c' : Cube), canContainPyramid c' p → cubeVolume c' ≥ cubeVolume c := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_for_pyramid_l3438_343847


namespace NUMINAMATH_CALUDE_problem_solution_l3438_343856

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem problem_solution (n : ℕ) (h1 : n = 1221) :
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n} ∧ d.card = 8) ∧
  (∃ (d1 d2 d3 : ℕ), d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧
    d1 < d2 ∧ d2 < d3 ∧
    ∀ (x : ℕ), x ∣ n → x ≤ d1 ∨ x = d2 ∨ x ≥ d3 ∧
    d1 + d2 + d3 = 15) ∧
  is_four_digit n ∧
  (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    n = p * q * r ∧ p - 5 * q = 2 * r) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3438_343856


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3438_343849

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define vectors a and b
variable (a b : V)

-- State the theorem
theorem angle_between_vectors 
  (h1 : ‖a‖ = Real.sqrt 3)
  (h2 : ‖b‖ = 1)
  (h3 : ‖a - 2 • b‖ = 1) :
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3438_343849


namespace NUMINAMATH_CALUDE_rectangle_count_l3438_343876

/-- The number of ways to choose 2 items from n items -/
def choose (n : ℕ) (k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of parallel lines in each direction -/
def num_lines : ℕ := 6

/-- The number of rectangles formed by the intersection of parallel lines -/
def num_rectangles : ℕ := (choose num_lines 2) * (choose num_lines 2)

theorem rectangle_count : num_rectangles = 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_l3438_343876


namespace NUMINAMATH_CALUDE_fifteen_ones_sum_multiple_of_30_l3438_343845

theorem fifteen_ones_sum_multiple_of_30 : 
  (Nat.choose 14 9 : ℕ) = 2002 := by sorry

end NUMINAMATH_CALUDE_fifteen_ones_sum_multiple_of_30_l3438_343845


namespace NUMINAMATH_CALUDE_intersection_equals_M_l3438_343858

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {x | x*(x-1)*(x-2) = 0}

theorem intersection_equals_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_intersection_equals_M_l3438_343858


namespace NUMINAMATH_CALUDE_fish_birth_calculation_l3438_343840

theorem fish_birth_calculation (num_tanks : ℕ) (fish_per_tank : ℕ) (total_young : ℕ) :
  num_tanks = 3 →
  fish_per_tank = 4 →
  total_young = 240 →
  total_young / (num_tanks * fish_per_tank) = 20 :=
by sorry

end NUMINAMATH_CALUDE_fish_birth_calculation_l3438_343840


namespace NUMINAMATH_CALUDE_polynomial_real_root_l3438_343896

/-- The polynomial for which we're finding real roots -/
def P (a x : ℝ) : ℝ := x^5 + a*x^4 - x^3 + a*x^2 + x + 1

/-- The set of values for a where the polynomial has at least one real root -/
def A : Set ℝ := { a | a ≤ -1/2 ∨ a ≥ 1/2 }

/-- Theorem stating that the polynomial has at least one real root if and only if a is in set A -/
theorem polynomial_real_root (a : ℝ) : (∃ x : ℝ, P a x = 0) ↔ a ∈ A := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l3438_343896


namespace NUMINAMATH_CALUDE_tan_product_lower_bound_l3438_343871

theorem tan_product_lower_bound (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < Real.pi / 2)
  (h_acute_β : 0 < β ∧ β < Real.pi / 2)
  (h_acute_γ : 0 < γ ∧ γ < Real.pi / 2)
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_lower_bound_l3438_343871


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l3438_343892

/-- Triangle ABC with given properties --/
structure Triangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  angle_ABC : Real

/-- Angle bisector AD in triangle ABC --/
structure AngleBisector (T : Triangle) where
  AD : ℝ
  is_bisector : Bool

/-- Area of triangle ADC --/
def area_ADC (T : Triangle) (AB : AngleBisector T) : ℝ := sorry

/-- Main theorem --/
theorem triangle_area_theorem (T : Triangle) (AB : AngleBisector T) :
  T.angle_ABC = 90 ∧ T.AB = 90 ∧ T.BC = 56 ∧ T.AC = 2 * T.BC - 6 ∧ AB.is_bisector = true →
  abs (area_ADC T AB - 1363) < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l3438_343892


namespace NUMINAMATH_CALUDE_circular_table_dice_probability_l3438_343806

/-- The number of people sitting around the circular table -/
def num_people : ℕ := 5

/-- The number of sides on each die -/
def die_sides : ℕ := 8

/-- The probability that two adjacent people roll different numbers -/
def prob_different_adjacent : ℚ := 7 / 8

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ := (prob_different_adjacent) ^ num_people

theorem circular_table_dice_probability :
  prob_no_adjacent_same = (7 / 8) ^ 5 :=
sorry

end NUMINAMATH_CALUDE_circular_table_dice_probability_l3438_343806


namespace NUMINAMATH_CALUDE_smallest_divisible_by_8_13_14_l3438_343836

theorem smallest_divisible_by_8_13_14 :
  ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 13 ∣ n ∧ 14 ∣ n ∧ ∀ (m : ℕ), m > 0 → 8 ∣ m → 13 ∣ m → 14 ∣ m → n ≤ m :=
by
  use 728
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_8_13_14_l3438_343836


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3438_343865

/-- Given a bus with an average speed including stoppages and the time it stops per hour,
    calculate the average speed excluding stoppages. -/
theorem bus_speed_excluding_stoppages
  (speed_with_stops : ℝ)
  (stop_time : ℝ)
  (h1 : speed_with_stops = 20)
  (h2 : stop_time = 40) :
  speed_with_stops * (60 / (60 - stop_time)) = 60 :=
sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3438_343865


namespace NUMINAMATH_CALUDE_initial_deposit_calculation_l3438_343863

-- Define the initial deposit
variable (P : ℝ)

-- Define the interest rates
def first_year_rate : ℝ := 0.20
def second_year_rate : ℝ := 0.15

-- Define the final amount
def final_amount : ℝ := 690

-- Theorem statement
theorem initial_deposit_calculation :
  (P * (1 + first_year_rate) / 2) * (1 + second_year_rate) = final_amount →
  P = 1000 := by
  sorry


end NUMINAMATH_CALUDE_initial_deposit_calculation_l3438_343863


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l3438_343808

theorem cos_sin_sum_equals_half : 
  Real.cos (25 * π / 180) * Real.cos (85 * π / 180) + 
  Real.sin (25 * π / 180) * Real.sin (85 * π / 180) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l3438_343808


namespace NUMINAMATH_CALUDE_universal_set_intersection_l3438_343841

-- Define the universe
variable (U : Type)

-- Define sets A and B
variable (A B : Set U)

-- Define S as the universal set
variable (S : Set U)

-- Theorem statement
theorem universal_set_intersection (h1 : S = Set.univ) (h2 : A ∩ B = S) : A = S ∧ B = S := by
  sorry

end NUMINAMATH_CALUDE_universal_set_intersection_l3438_343841


namespace NUMINAMATH_CALUDE_no_common_roots_l3438_343881

theorem no_common_roots (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) :
  ¬∃ x : ℝ, (x^4 + b*x + c = 0) ∧ (x^4 + a*x + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_common_roots_l3438_343881


namespace NUMINAMATH_CALUDE_probability_seven_heads_in_ten_flips_l3438_343824

-- Define the number of coins and the number of heads
def n : ℕ := 10
def k : ℕ := 7

-- Define the probability of getting heads on a single flip
def p : ℚ := 1/2

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- State the theorem
theorem probability_seven_heads_in_ten_flips :
  (binomial_coeff n k : ℚ) * p^n = 15/128 := by sorry

end NUMINAMATH_CALUDE_probability_seven_heads_in_ten_flips_l3438_343824


namespace NUMINAMATH_CALUDE_vertex_on_parabola_and_line_intersection_l3438_343860

/-- The quadratic function -/
def f (m x : ℝ) : ℝ := x^2 + 2*(m + 1)*x - m + 1

/-- The vertex of the quadratic function -/
def vertex (m : ℝ) : ℝ × ℝ := (-m - 1, -m^2 - 3*m)

/-- The parabola on which the vertex lies -/
def parabola (x : ℝ) : ℝ := -x^2 + x + 2

/-- The line that may pass through the vertex -/
def line (x : ℝ) : ℝ := x + 1

theorem vertex_on_parabola_and_line_intersection (m : ℝ) :
  (∀ m, parabola (vertex m).1 = (vertex m).2) ∧
  (line (vertex m).1 = (vertex m).2 ↔ m = -2 ∨ m = 0) :=
by sorry

end NUMINAMATH_CALUDE_vertex_on_parabola_and_line_intersection_l3438_343860


namespace NUMINAMATH_CALUDE_rational_cube_sum_representation_l3438_343814

theorem rational_cube_sum_representation (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
    r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_sum_representation_l3438_343814


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3438_343843

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3438_343843


namespace NUMINAMATH_CALUDE_max_min_difference_f_l3438_343819

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem max_min_difference_f : 
  (⨆ x ∈ (Set.Icc (-3) 0), f x) - (⨅ x ∈ (Set.Icc (-3) 0), f x) = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_min_difference_f_l3438_343819


namespace NUMINAMATH_CALUDE_concert_ticket_prices_l3438_343884

theorem concert_ticket_prices (x : ℕ) : 
  (∃ a b : ℕ, a * x = 80 ∧ b * x = 100) → 
  (Finset.filter (fun d => d ∣ 80 ∧ d ∣ 100) (Finset.range 101)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_prices_l3438_343884


namespace NUMINAMATH_CALUDE_simon_blueberries_l3438_343831

def blueberry_problem (own_bushes nearby_bushes pies_made blueberries_per_pie : ℕ) : Prop :=
  own_bushes + nearby_bushes = pies_made * blueberries_per_pie

theorem simon_blueberries : 
  ∃ (own_bushes : ℕ), 
    blueberry_problem own_bushes 200 3 100 ∧ 
    own_bushes = 100 := by sorry

end NUMINAMATH_CALUDE_simon_blueberries_l3438_343831


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3438_343875

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3438_343875


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l3438_343834

theorem tangent_sum_simplification :
  (Real.tan (10 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (50 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (40 * π / 180) = -Real.sin (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l3438_343834


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l3438_343842

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (initial_average : ℝ) 
  (wrong_mark correct_mark : ℝ) :
  n = 25 → 
  initial_average = 100 → 
  wrong_mark = 60 → 
  correct_mark = 10 → 
  (n * initial_average - wrong_mark + correct_mark) / n = 98 := by
sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l3438_343842


namespace NUMINAMATH_CALUDE_log_equation_solution_l3438_343862

theorem log_equation_solution (x : ℝ) (h : Real.log x / Real.log 3 * Real.log 3 / Real.log 4 = 4) : x = 256 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3438_343862


namespace NUMINAMATH_CALUDE_percentage_non_defective_m3_l3438_343846

theorem percentage_non_defective_m3 (m1_percentage : Real) (m2_percentage : Real)
  (m1_defective : Real) (m2_defective : Real) (total_defective : Real) :
  m1_percentage = 0.4 →
  m2_percentage = 0.3 →
  m1_defective = 0.03 →
  m2_defective = 0.01 →
  total_defective = 0.036 →
  ∃ (m3_non_defective : Real),
    m3_non_defective = 0.93 ∧
    m1_percentage * m1_defective + m2_percentage * m2_defective +
    (1 - m1_percentage - m2_percentage) * (1 - m3_non_defective) = total_defective :=
by sorry

end NUMINAMATH_CALUDE_percentage_non_defective_m3_l3438_343846


namespace NUMINAMATH_CALUDE_greatest_common_factor_3465_10780_l3438_343801

theorem greatest_common_factor_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_3465_10780_l3438_343801


namespace NUMINAMATH_CALUDE_correct_height_l3438_343809

theorem correct_height (n : ℕ) (initial_avg actual_avg wrong_height : ℝ) :
  n = 35 →
  initial_avg = 180 →
  actual_avg = 178 →
  wrong_height = 156 →
  ∃ (correct_height : ℝ),
    correct_height = n * actual_avg - (n * initial_avg - wrong_height) := by
  sorry

end NUMINAMATH_CALUDE_correct_height_l3438_343809


namespace NUMINAMATH_CALUDE_B_power_101_l3438_343839

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  !![0, 1, 0;
     0, 0, 1;
     1, 0, 0]

theorem B_power_101 :
  B ^ 101 = !![0, 0, 1;
                1, 0, 0;
                0, 1, 0] := by sorry

end NUMINAMATH_CALUDE_B_power_101_l3438_343839


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l3438_343891

/-- Represents the number of students in each grade -/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- Represents the number of students to be sampled from each grade -/
structure SampleSize where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- Calculates the stratified sample size for each grade -/
def stratifiedSample (population : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPopulation := population.freshmen + population.sophomores + population.juniors
  let ratio := totalSample / totalPopulation
  { freshmen := population.freshmen * ratio,
    sophomores := population.sophomores * ratio,
    juniors := population.juniors * ratio }

theorem correct_stratified_sample :
  let population := GradePopulation.mk 560 540 520
  let sample := stratifiedSample population 81
  sample = SampleSize.mk 28 27 26 := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l3438_343891


namespace NUMINAMATH_CALUDE_basketball_handshakes_l3438_343861

/-- The number of handshakes in a basketball game -/
def total_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let inter_team := team_size * team_size * (num_teams - 1) / 2
  let intra_team := num_teams * (team_size * (team_size - 1) / 2)
  let with_referees := num_teams * team_size * num_referees
  inter_team + intra_team + with_referees

/-- Theorem stating the total number of handshakes in the specific basketball game scenario -/
theorem basketball_handshakes :
  total_handshakes 6 2 3 = 102 := by
  sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l3438_343861


namespace NUMINAMATH_CALUDE_kim_shoes_problem_l3438_343829

theorem kim_shoes_problem (num_pairs : ℕ) (prob_same_color : ℚ) : 
  num_pairs = 7 →
  prob_same_color = 7692307692307693 / 100000000000000000 →
  (1 : ℚ) / (num_pairs * 2 - 1) = prob_same_color →
  num_pairs * 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_kim_shoes_problem_l3438_343829


namespace NUMINAMATH_CALUDE_line_through_points_l3438_343828

/-- The general form equation of a line passing through two points -/
def general_form_equation (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧
    a * x₁ + b * y₁ + c = 0 ∧
    a * x₂ + b * y₂ + c = 0 ∧
    (a ≠ 0 ∨ b ≠ 0)}

/-- Theorem: The general form equation of the line passing through (1, 1) and (-2, 4) is x + y - 2 = 0 -/
theorem line_through_points : 
  general_form_equation 1 1 (-2) 4 = {(x, y) | x + y - 2 = 0} := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3438_343828


namespace NUMINAMATH_CALUDE_ten_not_diff_of_squares_five_is_diff_of_squares_seven_is_diff_of_squares_eight_is_diff_of_squares_nine_is_diff_of_squares_l3438_343830

-- Define a function to check if a number is a difference of two squares
def is_diff_of_squares (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 - b^2

-- Theorem stating that 10 cannot be expressed as the difference of two squares
theorem ten_not_diff_of_squares : ¬ is_diff_of_squares 10 :=
sorry

-- Theorems stating that 5, 7, 8, and 9 can be expressed as the difference of two squares
theorem five_is_diff_of_squares : is_diff_of_squares 5 :=
sorry

theorem seven_is_diff_of_squares : is_diff_of_squares 7 :=
sorry

theorem eight_is_diff_of_squares : is_diff_of_squares 8 :=
sorry

theorem nine_is_diff_of_squares : is_diff_of_squares 9 :=
sorry

end NUMINAMATH_CALUDE_ten_not_diff_of_squares_five_is_diff_of_squares_seven_is_diff_of_squares_eight_is_diff_of_squares_nine_is_diff_of_squares_l3438_343830


namespace NUMINAMATH_CALUDE_triple_composition_even_l3438_343879

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem triple_composition_even
  (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l3438_343879


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3438_343807

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_formula
  (a : ℕ → ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_geometric : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_relation : ∀ n : ℕ, (a (n + 2))^2 + 4*(a n)^2 = 4*(a (n + 1))^2) :
  ∀ n : ℕ, a n = 2^((n + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3438_343807


namespace NUMINAMATH_CALUDE_solve_pencil_problem_l3438_343880

def pencil_problem (anna_pencils : ℕ) (harry_multiplier : ℕ) (harry_lost : ℕ) : Prop :=
  let harry_initial := anna_pencils * harry_multiplier
  harry_initial - harry_lost = 81

theorem solve_pencil_problem :
  pencil_problem 50 2 19 := by sorry

end NUMINAMATH_CALUDE_solve_pencil_problem_l3438_343880
