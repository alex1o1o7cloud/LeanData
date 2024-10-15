import Mathlib

namespace NUMINAMATH_CALUDE_supplementary_angles_difference_l377_37737

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 7 / 2 →  -- The ratio of the measures is 7:2
  max a b - min a b = 100 :=  -- The positive difference is 100°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_difference_l377_37737


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l377_37761

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_condition : a + b + c = 15)
  (product_sum_condition : a * b + a * c + b * c = 50) :
  a^3 + b^3 + c^3 - 3*a*b*c = 1125 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l377_37761


namespace NUMINAMATH_CALUDE_product_equation_solution_l377_37790

theorem product_equation_solution : ∃! (B : ℕ), 
  B < 10 ∧ (10 * B + 2) * (90 + B) = 8016 := by sorry

end NUMINAMATH_CALUDE_product_equation_solution_l377_37790


namespace NUMINAMATH_CALUDE_time_for_one_toy_l377_37775

/-- Represents the time (in hours) it takes to make a certain number of toys -/
structure ToyProduction where
  hours : ℝ
  toys : ℝ

/-- Given that 50 toys are made in 100 hours, prove that it takes 2 hours to make one toy -/
theorem time_for_one_toy (prod : ToyProduction) 
  (h1 : prod.hours = 100) 
  (h2 : prod.toys = 50) : 
  prod.hours / prod.toys = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_for_one_toy_l377_37775


namespace NUMINAMATH_CALUDE_exists_good_submatrix_l377_37755

/-- Definition of a binary matrix -/
def BinaryMatrix (n : ℕ) := Matrix (Fin n) (Fin n) Bool

/-- Definition of a good matrix -/
def IsGoodMatrix {n : ℕ} (A : BinaryMatrix n) : Prop :=
  ∃ (x y : Bool), ∀ (i j : Fin n),
    (i < j → A i j = x) ∧
    (j < i → A i j = y)

/-- Main theorem -/
theorem exists_good_submatrix :
  ∃ (M : ℕ), ∀ (n : ℕ) (A : BinaryMatrix n),
    n > M →
    ∃ (m : ℕ) (indices : Fin m → Fin n),
      Function.Injective indices ∧
      IsGoodMatrix (Matrix.submatrix A indices indices) :=
by sorry

end NUMINAMATH_CALUDE_exists_good_submatrix_l377_37755


namespace NUMINAMATH_CALUDE_sum_of_opposite_sign_l377_37778

/-- Two real numbers are opposite in sign if their product is less than or equal to zero -/
def opposite_sign (a b : ℝ) : Prop := a * b ≤ 0

/-- If two real numbers are opposite in sign, then their sum is zero -/
theorem sum_of_opposite_sign (a b : ℝ) : opposite_sign a b → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_opposite_sign_l377_37778


namespace NUMINAMATH_CALUDE_quadrant_line_relationships_l377_37722

/-- A line passing through the first, second, and fourth quadrants -/
structure QuadrantLine where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_quadrants : 
    ∃ (x₁ y₁ x₂ y₂ x₄ y₄ : ℝ),
      (x₁ > 0 ∧ y₁ > 0 ∧ a * x₁ + b * y₁ + c = 0) ∧
      (x₂ < 0 ∧ y₂ > 0 ∧ a * x₂ + b * y₂ + c = 0) ∧
      (x₄ > 0 ∧ y₄ < 0 ∧ a * x₄ + b * y₄ + c = 0)

/-- The relationships between a, b, and c for a line passing through the first, second, and fourth quadrants -/
theorem quadrant_line_relationships (l : QuadrantLine) : l.a * l.b > 0 ∧ l.b * l.c < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_line_relationships_l377_37722


namespace NUMINAMATH_CALUDE_tangent_intersection_points_l377_37768

/-- Given a function f(x) = x^3 - x^2 + ax + 1, prove that the tangent line passing through
    the origin intersects the curve y = f(x) at the points (1, a + 1) and (-1, -a - 1). -/
theorem tangent_intersection_points (a : ℝ) :
  let f := λ x : ℝ => x^3 - x^2 + a*x + 1
  let tangent_line := λ x : ℝ => (a + 1) * x
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -1 ∧
    f x₁ = tangent_line x₁ ∧
    f x₂ = tangent_line x₂ ∧
    (∀ x : ℝ, f x = tangent_line x → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_tangent_intersection_points_l377_37768


namespace NUMINAMATH_CALUDE_players_who_quit_video_game_problem_l377_37733

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  initial_players - (total_lives / lives_per_player)

theorem video_game_problem :
  players_who_quit 13 6 30 = 8 := by
  sorry

end NUMINAMATH_CALUDE_players_who_quit_video_game_problem_l377_37733


namespace NUMINAMATH_CALUDE_macks_travel_problem_l377_37743

/-- Mack's travel problem -/
theorem macks_travel_problem (speed_to_office : ℝ) (total_time : ℝ) (time_to_office : ℝ) 
  (h1 : speed_to_office = 58)
  (h2 : total_time = 3)
  (h3 : time_to_office = 1.4) :
  (speed_to_office * time_to_office) / (total_time - time_to_office) = 50.75 := by
  sorry

end NUMINAMATH_CALUDE_macks_travel_problem_l377_37743


namespace NUMINAMATH_CALUDE_square_wire_length_l377_37736

theorem square_wire_length (area : ℝ) (side_length : ℝ) (wire_length : ℝ) : 
  area = 324 → 
  area = side_length ^ 2 → 
  wire_length = 4 * side_length → 
  wire_length = 72 := by
sorry

end NUMINAMATH_CALUDE_square_wire_length_l377_37736


namespace NUMINAMATH_CALUDE_equation_solutions_l377_37708

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 49 = 0 ↔ x = 7 ∨ x = -7) ∧
  (∀ x : ℝ, 2*(x+1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l377_37708


namespace NUMINAMATH_CALUDE_final_value_after_percentage_changes_l377_37751

theorem final_value_after_percentage_changes (initial_value : ℝ) 
  (increase_percent : ℝ) (decrease_percent : ℝ) : 
  initial_value = 1500 → 
  increase_percent = 20 → 
  decrease_percent = 40 → 
  let increased_value := initial_value * (1 + increase_percent / 100)
  let final_value := increased_value * (1 - decrease_percent / 100)
  final_value = 1080 := by
  sorry

end NUMINAMATH_CALUDE_final_value_after_percentage_changes_l377_37751


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l377_37772

/-- Two lines are parallel if their coefficients are proportional -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- The first line l₁: (m+3)x + 4y + 3m - 5 = 0 -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 3) * x + 4 * y + 3 * m - 5 = 0

/-- The second line l₂: 2x + (m+5)y - 8 = 0 -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 5) * y - 8 = 0

/-- Theorem: If l₁ and l₂ are parallel, then m = -7 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel (m + 3) 4 2 (m + 5) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l377_37772


namespace NUMINAMATH_CALUDE_probability_non_defective_pencils_l377_37723

/-- The probability of selecting 3 non-defective pencils from a box of 10 pencils with 2 defective pencils -/
theorem probability_non_defective_pencils :
  let total_pencils : ℕ := 10
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations := Nat.choose total_pencils selected_pencils
  let non_defective_combinations := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_defective_pencils_l377_37723


namespace NUMINAMATH_CALUDE_desk_chair_prices_l377_37731

/-- Proves that given a desk and chair set with a total price of 115 yuan,
    where the chair is 45 yuan cheaper than the desk,
    the price of the chair is 35 yuan and the price of the desk is 80 yuan. -/
theorem desk_chair_prices (total_price : ℕ) (price_difference : ℕ)
    (h1 : total_price = 115)
    (h2 : price_difference = 45) :
    ∃ (chair_price desk_price : ℕ),
      chair_price = 35 ∧
      desk_price = 80 ∧
      chair_price + desk_price = total_price ∧
      desk_price = chair_price + price_difference :=
by
  sorry

end NUMINAMATH_CALUDE_desk_chair_prices_l377_37731


namespace NUMINAMATH_CALUDE_sample_xy_value_l377_37704

theorem sample_xy_value (x y : ℝ) : 
  (x + 1 + y + 5) / 4 = 2 →
  ((x - 2)^2 + (1 - 2)^2 + (y - 2)^2 + (5 - 2)^2) / 4 = 5 →
  x * y = -4 := by
sorry

end NUMINAMATH_CALUDE_sample_xy_value_l377_37704


namespace NUMINAMATH_CALUDE_max_min_difference_z_l377_37700

theorem max_min_difference_z (x y z : ℝ) 
  (sum_condition : x + y + z = 5)
  (sum_squares_condition : x^2 + y^2 + z^2 = 29) :
  ∃ (z_max z_min : ℝ),
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 29) → z' ≤ z_max) ∧
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 29) → z' ≥ z_min) ∧
    z_max - z_min = 20/3 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l377_37700


namespace NUMINAMATH_CALUDE_profit_calculation_l377_37714

def trees : ℕ := 30
def planks_per_tree : ℕ := 25
def planks_per_table : ℕ := 15
def selling_price : ℕ := 300
def labor_cost : ℕ := 3000

theorem profit_calculation :
  let total_planks := trees * planks_per_tree
  let tables_made := total_planks / planks_per_table
  let revenue := tables_made * selling_price
  revenue - labor_cost = 12000 := by
sorry

end NUMINAMATH_CALUDE_profit_calculation_l377_37714


namespace NUMINAMATH_CALUDE_dice_probabilities_l377_37769

/-- Represents the probabilities of an unfair 6-sided dice -/
structure DiceProbabilities where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  sum_one : a + b + c + d + e + f = 1
  all_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f

/-- The probability of rolling the same number twice -/
def P (probs : DiceProbabilities) : ℝ :=
  probs.a^2 + probs.b^2 + probs.c^2 + probs.d^2 + probs.e^2 + probs.f^2

/-- The probability of rolling an odd number first and an even number second -/
def Q (probs : DiceProbabilities) : ℝ :=
  (probs.a + probs.c + probs.e) * (probs.b + probs.d + probs.f)

theorem dice_probabilities (probs : DiceProbabilities) :
  P probs ≥ 1/6 ∧ Q probs ≤ 1/4 ∧ Q probs ≥ 1/2 - 3/2 * P probs := by
  sorry

end NUMINAMATH_CALUDE_dice_probabilities_l377_37769


namespace NUMINAMATH_CALUDE_percentage_loss_l377_37759

/-- Calculate the percentage of loss in a sale transaction -/
theorem percentage_loss (cost_price selling_price : ℚ) (h1 : cost_price = 1800) (h2 : selling_price = 1620) :
  (cost_price - selling_price) / cost_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_loss_l377_37759


namespace NUMINAMATH_CALUDE_swimming_passings_l377_37795

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  poolLength : ℝ
  swimmerASpeed : ℝ
  swimmerBSpeed : ℝ
  duration : ℝ

/-- Calculates the number of times swimmers pass each other -/
def calculatePassings (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- Theorem stating the number of passings in the given scenario -/
theorem swimming_passings :
  let scenario : SwimmingScenario := {
    poolLength := 100,
    swimmerASpeed := 4,
    swimmerBSpeed := 5,
    duration := 30 * 60  -- 30 minutes in seconds
  }
  calculatePassings scenario = 54 := by sorry

end NUMINAMATH_CALUDE_swimming_passings_l377_37795


namespace NUMINAMATH_CALUDE_smallest_value_of_floor_sum_l377_37793

theorem smallest_value_of_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_floor_sum_l377_37793


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l377_37794

theorem quadratic_always_positive (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) ↔ m > 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l377_37794


namespace NUMINAMATH_CALUDE_number_equals_scientific_notation_l377_37701

-- Define the number we want to represent in scientific notation
def number : ℕ := 11700000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.17 * (10 ^ 7)

-- Theorem stating that the number is equal to its scientific notation representation
theorem number_equals_scientific_notation : (number : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_number_equals_scientific_notation_l377_37701


namespace NUMINAMATH_CALUDE_exactlyOneBlack_bothBlack_mutuallyExclusive_atLeastOneBlack_bothRed_complementary_l377_37764

-- Define the sample space
def SampleSpace := Fin 4 × Fin 4

-- Define the events
def exactlyOneBlack (outcome : SampleSpace) : Prop :=
  (outcome.1 = 2 ∧ outcome.2 = 0) ∨ (outcome.1 = 2 ∧ outcome.2 = 1) ∨
  (outcome.1 = 3 ∧ outcome.2 = 0) ∨ (outcome.1 = 3 ∧ outcome.2 = 1)

def bothBlack (outcome : SampleSpace) : Prop :=
  outcome.1 = 2 ∧ outcome.2 = 3

def atLeastOneBlack (outcome : SampleSpace) : Prop :=
  outcome.1 = 2 ∨ outcome.1 = 3 ∨ outcome.2 = 2 ∨ outcome.2 = 3

def bothRed (outcome : SampleSpace) : Prop :=
  outcome.1 = 0 ∧ outcome.2 = 1

-- Theorem statements
theorem exactlyOneBlack_bothBlack_mutuallyExclusive :
  ∀ (outcome : SampleSpace), ¬(exactlyOneBlack outcome ∧ bothBlack outcome) :=
sorry

theorem atLeastOneBlack_bothRed_complementary :
  ∀ (outcome : SampleSpace), atLeastOneBlack outcome ↔ ¬(bothRed outcome) :=
sorry

end NUMINAMATH_CALUDE_exactlyOneBlack_bothBlack_mutuallyExclusive_atLeastOneBlack_bothRed_complementary_l377_37764


namespace NUMINAMATH_CALUDE_unique_prime_power_of_four_minus_one_l377_37748

theorem unique_prime_power_of_four_minus_one :
  ∃! (n : ℕ), n > 0 ∧ Nat.Prime (4^n - 1) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_power_of_four_minus_one_l377_37748


namespace NUMINAMATH_CALUDE_minimum_value_range_l377_37780

noncomputable def f (x : ℝ) := x^3 - 3*x

def has_minimum_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ (c : ℝ), a < c ∧ c < b ∧ ∀ (x : ℝ), a < x ∧ x < b → f c ≤ f x

theorem minimum_value_range (a : ℝ) :
  has_minimum_on_interval f a (10 + 2*a^2) ↔ -2 ≤ a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_range_l377_37780


namespace NUMINAMATH_CALUDE_confidence_interval_for_population_mean_l377_37729

-- Define the sample data
def sample_data : List (Float × Nat) := [(-2, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 1)]

-- Define the sample size
def n : Nat := 10

-- Define the confidence level
def confidence_level : Float := 0.95

-- Define the critical t-value for 9 degrees of freedom and 95% confidence
def t_critical : Float := 2.262

-- State the theorem
theorem confidence_interval_for_population_mean :
  let sample_mean := (sample_data.map (λ (x, freq) => x * freq.toFloat)).sum / n.toFloat
  let sample_variance := (sample_data.map (λ (x, freq) => freq.toFloat * (x - sample_mean)^2)).sum / (n.toFloat - 1)
  let sample_std_dev := sample_variance.sqrt
  let margin_of_error := t_critical * (sample_std_dev / (n.toFloat.sqrt))
  0.363 < sample_mean - margin_of_error ∧ sample_mean + margin_of_error < 3.837 := by
  sorry


end NUMINAMATH_CALUDE_confidence_interval_for_population_mean_l377_37729


namespace NUMINAMATH_CALUDE_cone_surface_area_l377_37744

/-- The surface area of a cone given its slant height and angle between slant height and axis -/
theorem cone_surface_area (slant_height : ℝ) (angle : ℝ) : 
  slant_height = 20 →
  angle = 30 * π / 180 →
  ∃ (surface_area : ℝ), surface_area = 300 * π := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_l377_37744


namespace NUMINAMATH_CALUDE_negation_existence_positive_real_l377_37739

theorem negation_existence_positive_real (R_plus : Set ℝ) :
  (¬ ∃ x ∈ R_plus, x > x^2) ↔ (∀ x ∈ R_plus, x ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_positive_real_l377_37739


namespace NUMINAMATH_CALUDE_sugar_water_concentration_l377_37706

theorem sugar_water_concentration (a b m : ℝ) 
  (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : 
  a / b < (a + m) / (b + m) := by
sorry

end NUMINAMATH_CALUDE_sugar_water_concentration_l377_37706


namespace NUMINAMATH_CALUDE_alternate_arrangement_count_l377_37705

def number_of_men : ℕ := 2
def number_of_women : ℕ := 2

theorem alternate_arrangement_count :
  (number_of_men = 2 ∧ number_of_women = 2) →
  (∃ (count : ℕ), count = 8 ∧
    count = (number_of_men * number_of_women * 1 * 1) +
            (number_of_women * number_of_men * 1 * 1)) :=
by sorry

end NUMINAMATH_CALUDE_alternate_arrangement_count_l377_37705


namespace NUMINAMATH_CALUDE_total_people_in_line_l377_37791

/-- Given a line of people at an amusement park ride, this theorem proves
    the total number of people in line based on Eunji's position. -/
theorem total_people_in_line (eunji_position : ℕ) (people_behind_eunji : ℕ) :
  eunji_position = 6 →
  people_behind_eunji = 7 →
  eunji_position + people_behind_eunji = 13 := by
  sorry

#check total_people_in_line

end NUMINAMATH_CALUDE_total_people_in_line_l377_37791


namespace NUMINAMATH_CALUDE_arc_measure_is_sixty_l377_37756

/-- An equilateral triangle with a circle rolling along its side -/
structure TriangleWithCircle where
  /-- Side length of the equilateral triangle -/
  a : ℝ
  /-- Assumption that the side length is positive -/
  a_pos : 0 < a

/-- The angular measure of the arc intercepted on the circle -/
def arcMeasure (t : TriangleWithCircle) : ℝ := 60

/-- Theorem stating that the arc measure is always 60 degrees -/
theorem arc_measure_is_sixty (t : TriangleWithCircle) : arcMeasure t = 60 := by
  sorry

end NUMINAMATH_CALUDE_arc_measure_is_sixty_l377_37756


namespace NUMINAMATH_CALUDE_treadmill_theorem_l377_37725

def treadmill_problem (days : Nat) (distance_per_day : Real) 
  (speeds : List Real) (constant_speed : Real) : Prop :=
  days = 4 ∧
  distance_per_day = 3 ∧
  speeds = [6, 4, 3, 5] ∧
  constant_speed = 5 ∧
  let actual_time := (List.map (fun s => distance_per_day / s) speeds).sum
  let constant_time := (days * distance_per_day) / constant_speed
  (actual_time - constant_time) * 60 = 27

theorem treadmill_theorem : 
  ∃ (days : Nat) (distance_per_day : Real) (speeds : List Real) (constant_speed : Real),
  treadmill_problem days distance_per_day speeds constant_speed :=
sorry

end NUMINAMATH_CALUDE_treadmill_theorem_l377_37725


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l377_37797

theorem complex_fraction_simplification :
  let a := (5 + 4/45) - (4 + 1/6)
  let b := 5 + 8/15
  let c := (4 + 2/3) + 0.75
  let d := 3 + 9/13
  let e := 34 + 2/7
  let f := 0.3
  let g := 0.01
  let h := 70
  (a / b) / (c * d) * e + (f / g) / h + 2/7 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l377_37797


namespace NUMINAMATH_CALUDE_common_point_properties_l377_37798

open Real

noncomputable section

variables (a b : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 + 2*a*x

def g (a b : ℝ) (x : ℝ) : ℝ := 3*a^2 * log x + b

def common_point (a b : ℝ) : Prop :=
  ∃ x > 0, f a x = g a b x ∧ (deriv (f a)) x = (deriv (g a b)) x

theorem common_point_properties (h : a > 0) (h_common : common_point a b) :
  (a = 1 → b = 5/2) ∧
  (b = 5/2 * a^2 - 3*a^2 * log a) ∧
  (b ≤ 3/2 * exp (2/3)) :=
sorry

end

end NUMINAMATH_CALUDE_common_point_properties_l377_37798


namespace NUMINAMATH_CALUDE_outfit_combinations_l377_37792

theorem outfit_combinations (shirts : ℕ) (ties : ℕ) (belts : ℕ) : 
  shirts = 8 → ties = 6 → belts = 4 → shirts * ties * belts = 192 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l377_37792


namespace NUMINAMATH_CALUDE_best_of_three_win_probability_l377_37740

/-- The probability of winning a single set -/
def p : ℝ := 0.6

/-- The probability of winning a best-of-three match given the probability of winning each set -/
def win_probability (p : ℝ) : ℝ := p^2 + 3 * p^2 * (1 - p)

/-- Theorem: The probability of winning a best-of-three match when p = 0.6 is 0.648 -/
theorem best_of_three_win_probability :
  win_probability p = 0.648 := by sorry

end NUMINAMATH_CALUDE_best_of_three_win_probability_l377_37740


namespace NUMINAMATH_CALUDE_circumcircle_equation_l377_37726

-- Define the points and line
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)
def line_C (x y : ℝ) : Prop := 3 * x - y + 3 = 0

-- Define the area of the triangle
def area_ABC : ℝ := 10

-- Define the possible equations of the circumcircle
def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 - 1/2 * x - 5 * y - 3/2 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 25/6 * x - 89/9 * y + 347/18 = 0

-- Theorem statement
theorem circumcircle_equation :
  ∃ (C : ℝ × ℝ), line_C C.1 C.2 ∧
  (∀ (x y : ℝ), circle_eq1 x y ∨ circle_eq2 x y) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l377_37726


namespace NUMINAMATH_CALUDE_square_park_circumference_l377_37734

/-- The circumference of a square park with side length 5 kilometers is 20 kilometers. -/
theorem square_park_circumference :
  ∀ (side_length circumference : ℝ),
  side_length = 5 →
  circumference = 4 * side_length →
  circumference = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_square_park_circumference_l377_37734


namespace NUMINAMATH_CALUDE_inequality_proof_l377_37709

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a * b + b * c + c * a = 1) :
  3 * Real.rpow (1 / (a * b * c) + 6 * (a + b + c)) (1/3) ≤ Real.rpow 3 (1/3) / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l377_37709


namespace NUMINAMATH_CALUDE_abs_zero_iff_eq_l377_37732

theorem abs_zero_iff_eq (y : ℚ) : |5 * y - 7| = 0 ↔ y = 7 / 5 := by sorry

end NUMINAMATH_CALUDE_abs_zero_iff_eq_l377_37732


namespace NUMINAMATH_CALUDE_max_constant_quadratic_real_roots_l377_37712

theorem max_constant_quadratic_real_roots :
  ∀ c : ℝ, (∃ x : ℝ, x^2 - 6*x + c = 0) → c ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_constant_quadratic_real_roots_l377_37712


namespace NUMINAMATH_CALUDE_geometry_biology_overlap_difference_l377_37724

theorem geometry_biology_overlap_difference (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h1 : total = 232)
  (h2 : geometry = 144)
  (h3 : biology = 119) :
  let max_overlap := min geometry biology
  let min_overlap := geometry + biology - total
  max_overlap - min_overlap = 88 := by
sorry

end NUMINAMATH_CALUDE_geometry_biology_overlap_difference_l377_37724


namespace NUMINAMATH_CALUDE_sandy_initial_books_l377_37727

/-- The number of books Sandy had initially -/
def sandy_books : ℕ := 10

/-- The number of books Tim has -/
def tim_books : ℕ := 33

/-- The number of books Benny lost -/
def benny_lost : ℕ := 24

/-- The number of books Sandy and Tim have together after Benny lost some -/
def remaining_books : ℕ := 19

/-- Theorem stating that Sandy had 10 books initially -/
theorem sandy_initial_books : 
  sandy_books + tim_books = remaining_books + benny_lost := by sorry

end NUMINAMATH_CALUDE_sandy_initial_books_l377_37727


namespace NUMINAMATH_CALUDE_andy_distance_to_market_l377_37719

/-- The distance between Andy's house and the market -/
def distance_to_market (distance_to_school : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance - 2 * distance_to_school

theorem andy_distance_to_market :
  let distance_to_school : ℕ := 50
  let total_distance : ℕ := 140
  distance_to_market distance_to_school total_distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_andy_distance_to_market_l377_37719


namespace NUMINAMATH_CALUDE_interior_cubes_6_5_4_l377_37735

/-- Represents a rectangular prism -/
structure RectangularPrism where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of interior cubes in a rectangular prism -/
def interiorCubes (prism : RectangularPrism) : ℕ :=
  (prism.width - 2) * (prism.length - 2) * (prism.height - 2)

/-- Theorem: A 6x5x4 rectangular prism cut into 1x1x1 cubes has 24 interior cubes -/
theorem interior_cubes_6_5_4 :
  interiorCubes { width := 6, length := 5, height := 4 } = 24 := by
  sorry

#eval interiorCubes { width := 6, length := 5, height := 4 }

end NUMINAMATH_CALUDE_interior_cubes_6_5_4_l377_37735


namespace NUMINAMATH_CALUDE_f_value_at_neg_five_halves_l377_37770

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_neg_five_halves :
  (∀ x, f x = f (-x)) →                     -- f is even
  (∀ x, f (x + 2) = f x) →                  -- f has period 2
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2*x*(1 - x)) → -- f definition for 0 ≤ x ≤ 1
  f (-5/2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_f_value_at_neg_five_halves_l377_37770


namespace NUMINAMATH_CALUDE_merchant_profit_theorem_l377_37762

/-- Calculate the profit for a single item -/
def calculate_profit (purchase_price markup_percent discount_percent : ℚ) : ℚ :=
  let selling_price := purchase_price * (1 + markup_percent / 100)
  let discounted_price := selling_price * (1 - discount_percent / 100)
  discounted_price - purchase_price

/-- Calculate the total gross profit for three items -/
def total_gross_profit (
  jacket_price jeans_price shirt_price : ℚ)
  (jacket_markup jeans_markup shirt_markup : ℚ)
  (jacket_discount jeans_discount shirt_discount : ℚ) : ℚ :=
  calculate_profit jacket_price jacket_markup jacket_discount +
  calculate_profit jeans_price jeans_markup jeans_discount +
  calculate_profit shirt_price shirt_markup shirt_discount

theorem merchant_profit_theorem :
  total_gross_profit 60 45 30 25 30 15 20 10 5 = 10.43 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_theorem_l377_37762


namespace NUMINAMATH_CALUDE_cylinder_base_area_at_different_heights_l377_37758

/-- Represents the properties of a cylinder with constant volume -/
structure Cylinder where
  volume : ℝ
  height : ℝ
  base_area : ℝ
  height_positive : height > 0
  volume_eq : volume = height * base_area

/-- Theorem about the base area of a cylinder with constant volume -/
theorem cylinder_base_area_at_different_heights
  (c : Cylinder)
  (h_initial : c.height = 12)
  (s_initial : c.base_area = 2)
  (h_final : ℝ)
  (h_final_positive : h_final > 0)
  (h_final_value : h_final = 4.8) :
  let s_final := c.volume / h_final
  s_final = 5 := by sorry

end NUMINAMATH_CALUDE_cylinder_base_area_at_different_heights_l377_37758


namespace NUMINAMATH_CALUDE_presidency_meeting_combinations_l377_37785

/-- The number of schools participating in the conference -/
def num_schools : ℕ := 4

/-- The number of members in each school -/
def members_per_school : ℕ := 5

/-- The number of representatives sent by the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives sent by each non-host school -/
def non_host_representatives : ℕ := 1

/-- The total number of ways to choose representatives for the presidency meeting -/
def total_ways : ℕ := num_schools * (members_per_school.choose host_representatives) * 
  (members_per_school ^ (num_schools - 1))

theorem presidency_meeting_combinations : total_ways = 5000 := by
  sorry

end NUMINAMATH_CALUDE_presidency_meeting_combinations_l377_37785


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l377_37713

/-- Given a line passing through points (-3,1) and (1,3), prove that the sum of its slope and y-intercept is 3. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → (x = -3 ∧ y = 1) ∨ (x = 1 ∧ y = 3)) → 
  m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l377_37713


namespace NUMINAMATH_CALUDE_geometric_series_double_sum_l377_37799

/-- Given two infinite geometric series with the following properties:
    - First series: first term = 20, second term = 5
    - Second series: first term = 20, second term = 5+n
    - Sum of second series is double the sum of first series
    This theorem proves that n = 7.5 -/
theorem geometric_series_double_sum (n : ℝ) : 
  let a₁ : ℝ := 20
  let r₁ : ℝ := 5 / 20
  let r₂ : ℝ := (5 + n) / 20
  let sum₁ : ℝ := a₁ / (1 - r₁)
  let sum₂ : ℝ := a₁ / (1 - r₂)
  sum₂ = 2 * sum₁ → n = 7.5 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_double_sum_l377_37799


namespace NUMINAMATH_CALUDE_article_sale_profit_loss_l377_37728

theorem article_sale_profit_loss (cost_price selling_price_profit selling_price_25_percent : ℕ)
  (h1 : cost_price = 1400)
  (h2 : selling_price_profit = 1520)
  (h3 : selling_price_25_percent = 1750)
  (h4 : selling_price_25_percent = cost_price + cost_price / 4) :
  ∃ (selling_price_loss : ℕ),
    selling_price_loss = 1280 ∧
    (selling_price_profit - cost_price) / cost_price =
    (cost_price - selling_price_loss) / cost_price :=
by sorry

end NUMINAMATH_CALUDE_article_sale_profit_loss_l377_37728


namespace NUMINAMATH_CALUDE_multiply_both_sides_by_x_minus_3_l377_37702

variable (f g : ℝ → ℝ)
variable (x : ℝ)

theorem multiply_both_sides_by_x_minus_3 :
  f x = g x → (x - 3) * f x = (x - 3) * g x := by
  sorry

end NUMINAMATH_CALUDE_multiply_both_sides_by_x_minus_3_l377_37702


namespace NUMINAMATH_CALUDE_reflection_symmetry_l377_37752

/-- Represents an L-like shape with two segments --/
structure LShape :=
  (top_segment : ℝ)
  (bottom_segment : ℝ)

/-- Reflects an L-shape over a horizontal line --/
def reflect (shape : LShape) : LShape :=
  { top_segment := shape.bottom_segment,
    bottom_segment := shape.top_segment }

/-- Checks if two L-shapes are equal --/
def is_equal (shape1 shape2 : LShape) : Prop :=
  shape1.top_segment = shape2.top_segment ∧ shape1.bottom_segment = shape2.bottom_segment

theorem reflection_symmetry (original : LShape) :
  original.top_segment > original.bottom_segment →
  is_equal (reflect original) { top_segment := original.bottom_segment, bottom_segment := original.top_segment } :=
by
  sorry

#check reflection_symmetry

end NUMINAMATH_CALUDE_reflection_symmetry_l377_37752


namespace NUMINAMATH_CALUDE_star_operation_result_l377_37750

-- Define the operation *
def star : Fin 4 → Fin 4 → Fin 4
| 1, 1 => 1 | 1, 2 => 2 | 1, 3 => 3 | 1, 4 => 4
| 2, 1 => 2 | 2, 2 => 4 | 2, 3 => 1 | 2, 4 => 3
| 3, 1 => 3 | 3, 2 => 1 | 3, 3 => 4 | 3, 4 => 2
| 4, 1 => 4 | 4, 2 => 3 | 4, 3 => 2 | 4, 4 => 1

-- State the theorem
theorem star_operation_result : star 4 (star 3 2) = 4 := by sorry

end NUMINAMATH_CALUDE_star_operation_result_l377_37750


namespace NUMINAMATH_CALUDE_square_lake_side_length_l377_37707

/-- Proves the length of each side of a square lake given Jake's swimming and rowing speeds and the time it takes to row around the lake. -/
theorem square_lake_side_length 
  (swimming_speed : ℝ) 
  (rowing_speed : ℝ) 
  (rowing_time : ℝ) 
  (h1 : swimming_speed = 3) 
  (h2 : rowing_speed = 2 * swimming_speed) 
  (h3 : rowing_time = 10) : 
  (rowing_speed * rowing_time) / 4 = 15 := by
  sorry

#check square_lake_side_length

end NUMINAMATH_CALUDE_square_lake_side_length_l377_37707


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l377_37738

def downstream_distance : ℝ := 24
def upstream_distance : ℝ := 18
def time : ℝ := 3
def current_speed : ℝ := 2

def man_speed : ℝ := 6

theorem man_speed_in_still_water :
  (downstream_distance / time = man_speed + current_speed) ∧
  (upstream_distance / time = man_speed - current_speed) :=
by sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l377_37738


namespace NUMINAMATH_CALUDE_train_length_l377_37783

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → 
  time_s = 2.49980001599872 → 
  ∃ (length_m : ℝ), abs (length_m - 99.992) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l377_37783


namespace NUMINAMATH_CALUDE_wrong_observation_value_l377_37720

theorem wrong_observation_value 
  (n : ℕ) 
  (initial_mean correct_value new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 32)
  (h3 : correct_value = 48)
  (h4 : new_mean = 32.5) :
  ∃ wrong_value : ℝ,
    (n : ℝ) * new_mean = (n : ℝ) * initial_mean - wrong_value + correct_value ∧
    wrong_value = 23 := by
sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l377_37720


namespace NUMINAMATH_CALUDE_mel_weight_proof_l377_37753

/-- Mel's weight in pounds -/
def mels_weight : ℝ := 70

/-- Brenda's weight in pounds -/
def brendas_weight : ℝ := 220

/-- Relationship between Brenda's and Mel's weights -/
def weight_relationship (m : ℝ) : Prop := brendas_weight = 3 * m + 10

theorem mel_weight_proof : 
  weight_relationship mels_weight ∧ mels_weight = 70 := by
  sorry

end NUMINAMATH_CALUDE_mel_weight_proof_l377_37753


namespace NUMINAMATH_CALUDE_gis_may_lead_to_overfishing_l377_37789

/-- Represents the use of GIS technology in fishery production -/
structure GISTechnology where
  locateSchools : Bool
  widelyIntroduced : Bool

/-- Represents the state of fishery resources -/
structure FisheryResources where
  overfishing : Bool
  exhausted : Bool

/-- The impact of GIS technology on fishery resources -/
def gisImpact (tech : GISTechnology) : FisheryResources :=
  { overfishing := tech.locateSchools ∧ tech.widelyIntroduced,
    exhausted := tech.locateSchools ∧ tech.widelyIntroduced }

theorem gis_may_lead_to_overfishing (tech : GISTechnology) 
  (h1 : tech.locateSchools = true) 
  (h2 : tech.widelyIntroduced = true) : 
  (gisImpact tech).overfishing = true ∧ (gisImpact tech).exhausted = true :=
by sorry

end NUMINAMATH_CALUDE_gis_may_lead_to_overfishing_l377_37789


namespace NUMINAMATH_CALUDE_gcd_divisibility_l377_37749

theorem gcd_divisibility (p q r s : ℕ+) : 
  (Nat.gcd p.val q.val = 21) →
  (Nat.gcd q.val r.val = 45) →
  (Nat.gcd r.val s.val = 75) →
  (120 < Nat.gcd s.val p.val) →
  (Nat.gcd s.val p.val < 180) →
  9 ∣ p.val :=
by sorry

end NUMINAMATH_CALUDE_gcd_divisibility_l377_37749


namespace NUMINAMATH_CALUDE_problem_statement_l377_37767

theorem problem_statement (a b : ℝ) (h1 : a^2 + b^2 = 1) :
  (|a - b| / |1 - a*b| ≤ 1) ∧
  (a*b > 0 → (a + b)*(a^3 + b^3) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l377_37767


namespace NUMINAMATH_CALUDE_employee_pay_l377_37760

theorem employee_pay (total_pay x y : ℝ) : 
  total_pay = 638 →
  x = 1.2 * y →
  total_pay = x + y →
  y = 290 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_l377_37760


namespace NUMINAMATH_CALUDE_seeds_planted_equals_85_l377_37746

/-- Calculates the total number of seeds planted given the number of seeds per bed,
    flowers per bed, and total flowers grown. -/
def total_seeds_planted (seeds_per_bed : ℕ) (flowers_per_bed : ℕ) (total_flowers : ℕ) : ℕ :=
  let full_beds := total_flowers / flowers_per_bed
  let seeds_in_full_beds := full_beds * seeds_per_bed
  let flowers_in_partial_bed := total_flowers % flowers_per_bed
  seeds_in_full_beds + flowers_in_partial_bed

/-- Theorem stating that given the specific conditions, the total seeds planted is 85. -/
theorem seeds_planted_equals_85 :
  total_seeds_planted 15 60 220 = 85 := by
  sorry

end NUMINAMATH_CALUDE_seeds_planted_equals_85_l377_37746


namespace NUMINAMATH_CALUDE_other_number_is_three_l377_37721

theorem other_number_is_three (x y : ℝ) : 
  x + y = 10 → 
  2 * x = 3 * y + 5 → 
  (x = 7 ∨ y = 7) → 
  (x = 3 ∨ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_other_number_is_three_l377_37721


namespace NUMINAMATH_CALUDE_fraction_difference_simplification_l377_37703

theorem fraction_difference_simplification : 
  ∃ q : ℕ+, (1011 : ℚ) / 1010 - 1010 / 1011 = (2021 : ℚ) / q ∧ Nat.gcd 2021 q.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_simplification_l377_37703


namespace NUMINAMATH_CALUDE_hyperbola_equation_l377_37765

/-- Given a hyperbola and conditions on its asymptote and focus, prove its equation --/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ k : ℝ, (a / b = Real.sqrt 3) ∧ 
   (∃ x y : ℝ, x^2 = 24*y ∧ (y^2 / a^2 - x^2 / b^2 = 1))) →
  (∃ x y : ℝ, y^2 / 27 - x^2 / 9 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l377_37765


namespace NUMINAMATH_CALUDE_eight_person_arrangement_l377_37788

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def arrangements (n : ℕ) (a b c : ℕ) : ℕ :=
  factorial n - (factorial (n-1) * 2) - (factorial (n-2) * 6 - factorial (n-1) * 2)

theorem eight_person_arrangement : arrangements 8 1 1 1 = 36000 := by
  sorry

end NUMINAMATH_CALUDE_eight_person_arrangement_l377_37788


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_two_three_l377_37776

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}

-- Theorem to prove
theorem A_intersect_B_equals_two_three : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_two_three_l377_37776


namespace NUMINAMATH_CALUDE_recurring_decimal_one_zero_six_l377_37757

-- Define the recurring decimal notation
def recurring_decimal (whole : ℕ) (recurring : ℕ) : ℚ :=
  whole + (recurring : ℚ) / 99

-- State the theorem
theorem recurring_decimal_one_zero_six :
  recurring_decimal 1 6 = 35 / 33 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_one_zero_six_l377_37757


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l377_37711

theorem infinitely_many_solutions : 
  ∃ (S : Set (ℤ × ℤ × ℤ)), Set.Infinite S ∧ 
  ∀ (x y z : ℤ), (x, y, z) ∈ S → x^2 + y^2 + z^2 = x^3 + y^3 + z^3 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l377_37711


namespace NUMINAMATH_CALUDE_bear_ratio_l377_37730

theorem bear_ratio (black_bears : ℕ) (brown_bears : ℕ) (white_bears : ℕ) :
  black_bears = 60 →
  brown_bears = black_bears + 40 →
  black_bears + brown_bears + white_bears = 190 →
  (black_bears : ℚ) / white_bears = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_bear_ratio_l377_37730


namespace NUMINAMATH_CALUDE_water_consumption_theorem_l377_37779

/-- The amount of water drunk by the traveler and his camel in gallons -/
def total_water_gallons (traveler_ounces : ℕ) (camel_multiplier : ℕ) (ounces_per_gallon : ℕ) : ℚ :=
  (traveler_ounces + traveler_ounces * camel_multiplier) / ounces_per_gallon

/-- Theorem stating that the total water drunk is 2 gallons -/
theorem water_consumption_theorem :
  total_water_gallons 32 7 128 = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_theorem_l377_37779


namespace NUMINAMATH_CALUDE_min_flowers_for_outstanding_pioneer_l377_37763

/-- Represents the number of small red flowers needed for one small red flag -/
def flowers_per_flag : ℕ := 5

/-- Represents the number of small red flags needed for one badge -/
def flags_per_badge : ℕ := 4

/-- Represents the number of badges needed for one small gold cup -/
def badges_per_cup : ℕ := 3

/-- Represents the number of small gold cups needed to be an outstanding Young Pioneer -/
def cups_needed : ℕ := 2

/-- Theorem stating the minimum number of small red flowers needed to be an outstanding Young Pioneer -/
theorem min_flowers_for_outstanding_pioneer : 
  cups_needed * badges_per_cup * flags_per_badge * flowers_per_flag = 120 := by
  sorry

end NUMINAMATH_CALUDE_min_flowers_for_outstanding_pioneer_l377_37763


namespace NUMINAMATH_CALUDE_min_perimeter_noncongruent_isosceles_triangles_l377_37796

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ  -- Length of equal sides
  base : ℕ  -- Length of the base
  is_isosceles : side > base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt ((t.side : ℝ)^2 - ((t.base : ℝ) / 2)^2) / 2

/-- Theorem: Minimum perimeter of two noncongruent integer-sided isosceles triangles -/
theorem min_perimeter_noncongruent_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    9 * t2.base = 8 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      9 * s2.base = 8 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 842 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_noncongruent_isosceles_triangles_l377_37796


namespace NUMINAMATH_CALUDE_prob_both_paper_is_one_ninth_l377_37782

/-- Represents the possible choices in rock-paper-scissors -/
inductive Choice
| Rock
| Paper
| Scissors

/-- Represents the outcome of a rock-paper-scissors game -/
structure GameOutcome :=
  (player1 : Choice)
  (player2 : Choice)

/-- The set of all possible game outcomes -/
def allOutcomes : Finset GameOutcome :=
  sorry

/-- The set of outcomes where both players choose paper -/
def bothPaperOutcomes : Finset GameOutcome :=
  sorry

/-- The probability of both players choosing paper -/
def probBothPaper : ℚ :=
  (bothPaperOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

theorem prob_both_paper_is_one_ninth :
  probBothPaper = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_paper_is_one_ninth_l377_37782


namespace NUMINAMATH_CALUDE_min_value_expression_l377_37717

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c = Real.sqrt 6) :
  ∃ (min_val : ℝ), min_val = 8 * Real.sqrt 2 - 4 ∧
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → y + z = Real.sqrt 6 →
    (x * z^2 + 2 * x) / (y * z) + 16 / (x + 2) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l377_37717


namespace NUMINAMATH_CALUDE_e_recursive_relation_l377_37786

def e (n : ℕ) : ℕ := n^5

theorem e_recursive_relation (n : ℕ) :
  e (n + 6) = 6 * e (n + 5) - 15 * e (n + 4) + 20 * e (n + 3) - 15 * e (n + 2) + 6 * e (n + 1) - e n :=
by sorry

end NUMINAMATH_CALUDE_e_recursive_relation_l377_37786


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l377_37710

theorem min_value_reciprocal_sum (x y : ℝ) (h1 : x * y > 0) (h2 : x + 4 * y = 3) :
  ∀ z w : ℝ, z * w > 0 → z + 4 * w = 3 → (1 / x + 1 / y) ≤ (1 / z + 1 / w) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l377_37710


namespace NUMINAMATH_CALUDE_porridge_eaters_today_l377_37741

/-- Represents the number of children eating porridge daily -/
def daily_eaters : ℕ := 5

/-- Represents the number of children eating porridge every other day -/
def alternate_eaters : ℕ := 7

/-- Represents the number of children who ate porridge yesterday -/
def yesterday_eaters : ℕ := 9

/-- Calculates the number of children eating porridge today -/
def today_eaters : ℕ := daily_eaters + (alternate_eaters - (yesterday_eaters - daily_eaters))

/-- Theorem stating that the number of children eating porridge today is 8 -/
theorem porridge_eaters_today : today_eaters = 8 := by
  sorry

end NUMINAMATH_CALUDE_porridge_eaters_today_l377_37741


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_2_unique_a_for_integer_solution_set_l377_37716

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2| + |x - a|

-- Theorem for part 1
theorem solution_set_when_a_eq_2 :
  ∀ x : ℝ, f x 2 ≥ 4 ↔ x ≤ 0 ∨ x ≥ 4 := by sorry

-- Theorem for part 2
theorem unique_a_for_integer_solution_set :
  (∃! a : ℝ, ∀ x : ℤ, f (x : ℝ) a < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) ∧
  (∀ a : ℝ, (∀ x : ℤ, f (x : ℝ) a < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → a = 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_2_unique_a_for_integer_solution_set_l377_37716


namespace NUMINAMATH_CALUDE_inscribed_circle_total_area_l377_37766

/-- The total area of a figure consisting of a circle inscribed in a square, 
    where the circle has a diameter of 6 meters. -/
theorem inscribed_circle_total_area :
  let circle_diameter : ℝ := 6
  let square_side : ℝ := circle_diameter
  let circle_radius : ℝ := circle_diameter / 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let square_area : ℝ := square_side ^ 2
  let total_area : ℝ := circle_area + square_area
  total_area = 36 + 9 * π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_total_area_l377_37766


namespace NUMINAMATH_CALUDE_tangent_line_equation_l377_37718

/-- The equation of the tangent line to y = x³ + x + 1 at (1, 3) is 4x - y - 1 = 0 -/
theorem tangent_line_equation : 
  let f (x : ℝ) := x^3 + x + 1
  let point : ℝ × ℝ := (1, 3)
  let tangent_line (x y : ℝ) := 4*x - y - 1 = 0
  (∀ x, tangent_line x (f x)) ∧ tangent_line point.1 point.2 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_equation_l377_37718


namespace NUMINAMATH_CALUDE_complex_distance_l377_37784

theorem complex_distance (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 3)
  (h2 : Complex.abs z₂ = 4)
  (h3 : Complex.abs (z₁ + z₂) = 5) :
  Complex.abs (z₁ - z₂) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_l377_37784


namespace NUMINAMATH_CALUDE_min_value_abc_l377_37742

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 27 → a + 3 * b + 9 * c ≤ x + 3 * y + 9 * z :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l377_37742


namespace NUMINAMATH_CALUDE_quadrilateral_on_exponential_curve_l377_37773

theorem quadrilateral_on_exponential_curve (e : ℝ) (h_e : e > 0) :
  ∃ m : ℕ+, 
    (1/2 * (e^(m : ℝ) - e^((m : ℝ) + 3)) = (e^2 - 1) / e) ∧ 
    (∀ k : ℕ+, k < m → 1/2 * (e^(k : ℝ) - e^((k : ℝ) + 3)) ≠ (e^2 - 1) / e) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_on_exponential_curve_l377_37773


namespace NUMINAMATH_CALUDE_unique_sum_with_identical_digits_l377_37774

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Checks if a number is a three-digit number with identical digits -/
def is_three_identical_digits (m : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ Finset.range 10 ∧ m = 111 * d

theorem unique_sum_with_identical_digits :
  ∃! (n : ℕ), is_three_identical_digits (sum_of_first_n n) :=
sorry

end NUMINAMATH_CALUDE_unique_sum_with_identical_digits_l377_37774


namespace NUMINAMATH_CALUDE_evaluate_expression_l377_37781

theorem evaluate_expression : (-3)^7 / 3^5 + 2^6 - 4^2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l377_37781


namespace NUMINAMATH_CALUDE_exponential_inequality_l377_37747

theorem exponential_inequality (a x y : ℝ) :
  (a > 1 ∧ x > y → a^x > a^y) ∧ (a < 1 ∧ x > y → a^x < a^y) := by
sorry

end NUMINAMATH_CALUDE_exponential_inequality_l377_37747


namespace NUMINAMATH_CALUDE_inequality_proof_l377_37745

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l377_37745


namespace NUMINAMATH_CALUDE_andy_inappropriate_joke_demerits_l377_37777

/-- Represents the number of demerits Andy got for making an inappropriate joke -/
def inappropriate_joke_demerits : ℕ := sorry

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def max_demerits : ℕ := 50

/-- The number of demerits Andy gets per instance of being late -/
def late_demerits_per_instance : ℕ := 2

/-- The number of times Andy was late -/
def late_instances : ℕ := 6

/-- The number of additional demerits Andy can get this month before getting fired -/
def remaining_demerits : ℕ := 23

theorem andy_inappropriate_joke_demerits :
  inappropriate_joke_demerits = 
    max_demerits - remaining_demerits - (late_demerits_per_instance * late_instances) :=
by sorry

end NUMINAMATH_CALUDE_andy_inappropriate_joke_demerits_l377_37777


namespace NUMINAMATH_CALUDE_complex_power_patterns_l377_37787

theorem complex_power_patterns (i : ℂ) (h : i^2 = -1) :
  ∀ n : ℕ,
    i^(4*n + 1) = i ∧
    i^(4*n + 2) = -1 ∧
    i^(4*n + 3) = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_power_patterns_l377_37787


namespace NUMINAMATH_CALUDE_largest_consecutive_composites_l377_37715

theorem largest_consecutive_composites : ∃ (n : ℕ), 
  (n ≤ 36) ∧ 
  (∀ i ∈ Finset.range 7, 30 ≤ n - i ∧ n - i < 40 ∧ ¬(Nat.Prime (n - i))) ∧
  (∀ m : ℕ, m > n → 
    ¬(∀ i ∈ Finset.range 7, 30 ≤ m - i ∧ m - i < 40 ∧ ¬(Nat.Prime (m - i)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_composites_l377_37715


namespace NUMINAMATH_CALUDE_three_divisions_not_imply_symmetry_l377_37754

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  -- We don't need to define the full structure of a polygon for this problem
  mk :: 

/-- A division of a polygon is a way to split it into two equal parts. -/
structure Division (P : Polygon) where
  -- We don't need to define the full structure of a division for this problem
  mk ::

/-- A symmetry of a polygon is either a center of symmetry or an axis of symmetry. -/
inductive Symmetry (P : Polygon)
  | Center : Symmetry P
  | Axis : Symmetry P

/-- A polygon has three divisions if there exist three distinct ways to split it into two equal parts. -/
def has_three_divisions (P : Polygon) : Prop :=
  ∃ (d1 d2 d3 : Division P), d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

/-- A polygon has a symmetry if it has either a center of symmetry or an axis of symmetry. -/
def has_symmetry (P : Polygon) : Prop :=
  ∃ (s : Symmetry P), true

/-- 
The existence of three ways to divide a polygon into two equal parts 
does not necessarily imply the existence of a center or axis of symmetry for that polygon.
-/
theorem three_divisions_not_imply_symmetry :
  ∃ (P : Polygon), has_three_divisions P ∧ ¬has_symmetry P :=
sorry

end NUMINAMATH_CALUDE_three_divisions_not_imply_symmetry_l377_37754


namespace NUMINAMATH_CALUDE_smallest_n_with_partial_divisibility_l377_37771

theorem smallest_n_with_partial_divisibility : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - 2*n + 1) % k = 0) ∧ 
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - 2*n + 1) % k ≠ 0) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - 2*m + 1) % k = 0) ∨ 
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - 2*m + 1) % k ≠ 0)) ∧
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_with_partial_divisibility_l377_37771
