import Mathlib

namespace NUMINAMATH_CALUDE_streamer_earnings_l1709_170979

/-- Calculates the weekly earnings of a streamer given their schedule and hourly rate. -/
def weekly_earnings (days_off : ℕ) (hours_per_stream : ℕ) (hourly_rate : ℕ) : ℕ :=
  (7 - days_off) * hours_per_stream * hourly_rate

/-- Theorem stating that a streamer with the given schedule earns $160 per week. -/
theorem streamer_earnings :
  weekly_earnings 3 4 10 = 160 := by
  sorry

end NUMINAMATH_CALUDE_streamer_earnings_l1709_170979


namespace NUMINAMATH_CALUDE_logarithmic_function_problem_l1709_170971

open Real

theorem logarithmic_function_problem (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  let f := fun x => |log x|
  (f a = f b) →
  (∀ x ∈ Set.Icc (a^2) b, f x ≤ 2) →
  (∃ x ∈ Set.Icc (a^2) b, f x = 2) →
  2 * a + b = 2 / Real.exp 1 + Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_logarithmic_function_problem_l1709_170971


namespace NUMINAMATH_CALUDE_root_implies_q_value_l1709_170903

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def is_root (x p q : ℂ) : Prop := x^2 + p*x + q = 0

-- State the theorem
theorem root_implies_q_value (p q : ℝ) :
  is_root (2 + 3*i) p q → q = 13 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_q_value_l1709_170903


namespace NUMINAMATH_CALUDE_star_equality_implies_power_equality_l1709_170930

/-- The k-th smallest positive integer not in X -/
def f (X : Finset Nat) (k : Nat) : Nat :=
  (Finset.range k.succ \ X).min' sorry

/-- The * operation for finite sets of positive integers -/
def star (X Y : Finset Nat) : Finset Nat :=
  X ∪ (Y.image (f X))

/-- Repeated application of star operation -/
def repeat_star (X : Finset Nat) : Nat → Finset Nat
  | 0 => X
  | n + 1 => star X (repeat_star X n)

theorem star_equality_implies_power_equality
  (A B : Finset Nat) (a b : Nat) (ha : a > 0) (hb : b > 0) :
  star A B = star B A →
  repeat_star A b = repeat_star B a :=
sorry

end NUMINAMATH_CALUDE_star_equality_implies_power_equality_l1709_170930


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_l1709_170916

theorem purely_imaginary_complex (a : ℝ) :
  let z : ℂ := Complex.mk (a + 1) (1 + a^2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_l1709_170916


namespace NUMINAMATH_CALUDE_middle_card_is_four_l1709_170923

/-- Represents the three cards with positive integers -/
structure Cards where
  left : ℕ+
  middle : ℕ+
  right : ℕ+
  different : left ≠ middle ∧ middle ≠ right ∧ left ≠ right
  increasing : left < middle ∧ middle < right
  sum_15 : left + middle + right = 15

/-- Casey cannot determine the other two numbers given the leftmost card -/
def casey_statement (cards : Cards) : Prop :=
  ∀ (other_cards : Cards), other_cards.left = cards.left → 
    (other_cards.middle ≠ cards.middle ∨ other_cards.right ≠ cards.right)

/-- Tracy cannot determine the other two numbers given the rightmost card -/
def tracy_statement (cards : Cards) : Prop :=
  ∀ (other_cards : Cards), other_cards.right = cards.right → 
    (other_cards.left ≠ cards.left ∨ other_cards.middle ≠ cards.middle)

/-- Stacy cannot determine the other two numbers given the middle card -/
def stacy_statement (cards : Cards) : Prop :=
  ∀ (other_cards : Cards), other_cards.middle = cards.middle → 
    (other_cards.left ≠ cards.left ∨ other_cards.right ≠ cards.right)

/-- The main theorem stating that the middle card must be 4 -/
theorem middle_card_is_four (cards : Cards) 
  (h_casey : casey_statement cards)
  (h_tracy : tracy_statement cards)
  (h_stacy : stacy_statement cards) : 
  cards.middle = 4 := by
  sorry

end NUMINAMATH_CALUDE_middle_card_is_four_l1709_170923


namespace NUMINAMATH_CALUDE_common_tangents_count_l1709_170960

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 9 = 0

-- Define the function to count common tangent lines
def count_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle1 circle2 = 3 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l1709_170960


namespace NUMINAMATH_CALUDE_sample_grade_10_is_15_l1709_170966

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  grade_10_students : ℕ
  sample_size : ℕ

/-- Calculates the number of Grade 10 students to be sampled -/
def sample_grade_10 (school : School) : ℕ :=
  (school.sample_size * school.grade_10_students) / school.total_students

/-- Theorem stating that for the given school parameters, 
    the number of Grade 10 students to be sampled is 15 -/
theorem sample_grade_10_is_15 (school : School) 
  (h1 : school.total_students = 2000)
  (h2 : school.grade_10_students = 600)
  (h3 : school.sample_size = 50) :
  sample_grade_10 school = 15 := by
  sorry

#eval sample_grade_10 ⟨2000, 600, 50⟩

end NUMINAMATH_CALUDE_sample_grade_10_is_15_l1709_170966


namespace NUMINAMATH_CALUDE_james_weekly_nut_spending_l1709_170967

/-- Represents the cost and consumption of nuts -/
structure NutInfo where
  price : ℚ
  weight : ℚ
  consumption : ℚ
  days : ℕ

/-- Calculates the weekly cost for a type of nut -/
def weeklyCost (nut : NutInfo) : ℚ :=
  (nut.consumption / nut.days) * 7 * (nut.price / nut.weight)

/-- Theorem stating James' weekly spending on nuts -/
theorem james_weekly_nut_spending :
  let pistachios : NutInfo := ⟨10, 5, 30, 5⟩
  let almonds : NutInfo := ⟨8, 4, 24, 4⟩
  let walnuts : NutInfo := ⟨12, 6, 18, 3⟩
  weeklyCost pistachios + weeklyCost almonds + weeklyCost walnuts = 252 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_nut_spending_l1709_170967


namespace NUMINAMATH_CALUDE_equation_solution_l1709_170984

theorem equation_solution (a b : ℤ) : 
  (((a + 2 : ℚ) / (b + 1) + (a + 1 : ℚ) / (b + 2) = 1 + 6 / (a + b + 1)) ∧ 
   (b + 1 ≠ 0) ∧ (b + 2 ≠ 0) ∧ (a + b + 1 ≠ 0)) ↔ 
  ((∃ t : ℤ, t ≠ 0 ∧ t ≠ -1 ∧ a = -3 - t ∧ b = t) ∨ (a = 1 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1709_170984


namespace NUMINAMATH_CALUDE_min_rectangles_theorem_l1709_170956

/-- The minimum number of rectangles that can be placed on an n × n grid -/
def min_rectangles (k n : ℕ) : ℕ :=
  if n = k then k
  else min n (2*n - 2*k + 2)

/-- Theorem stating the minimum number of rectangles that can be placed -/
theorem min_rectangles_theorem (k n : ℕ) (h1 : k ≥ 2) (h2 : k ≤ n) (h3 : n ≤ 2*k - 1) :
  min_rectangles k n = 
    if n = k then k
    else min n (2*n - 2*k + 2) := by
  sorry

#check min_rectangles_theorem

end NUMINAMATH_CALUDE_min_rectangles_theorem_l1709_170956


namespace NUMINAMATH_CALUDE_product_selection_theorem_l1709_170995

def total_products : ℕ := 10
def defective_products : ℕ := 3
def good_products : ℕ := 7
def products_drawn : ℕ := 5

theorem product_selection_theorem :
  (∃ (no_defective : ℕ) (exactly_two_defective : ℕ) (at_least_one_defective : ℕ),
    -- No defective products
    no_defective = Nat.choose good_products products_drawn ∧
    -- Exactly 2 defective products
    exactly_two_defective = Nat.choose defective_products 2 * Nat.choose good_products 3 ∧
    -- At least 1 defective product
    at_least_one_defective = Nat.choose total_products products_drawn - Nat.choose good_products products_drawn) :=
by
  sorry

end NUMINAMATH_CALUDE_product_selection_theorem_l1709_170995


namespace NUMINAMATH_CALUDE_sum_of_squares_l1709_170997

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 131)
  (h2 : a + b + c = 21) : 
  a^2 + b^2 + c^2 = 179 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1709_170997


namespace NUMINAMATH_CALUDE_sean_initial_apples_l1709_170946

theorem sean_initial_apples (initial : ℕ) (received : ℕ) (total : ℕ) : 
  received = 8 → total = 17 → initial + received = total → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_sean_initial_apples_l1709_170946


namespace NUMINAMATH_CALUDE_pizza_division_l1709_170973

theorem pizza_division (total_pizza : ℚ) (num_friends : ℕ) : 
  total_pizza = 5/6 ∧ num_friends = 4 → 
  total_pizza / num_friends = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_pizza_division_l1709_170973


namespace NUMINAMATH_CALUDE_class_height_most_suitable_for_census_l1709_170945

/-- Represents a scenario that could be investigated --/
inductive Scenario
| WaterQuality
| StudentMentalHealth
| ClassHeight
| TVRatings

/-- Characteristics of a scenario --/
structure ScenarioCharacteristics where
  population_size : ℕ
  accessibility : Bool
  feasibility : Bool

/-- Defines what makes a scenario suitable for a census --/
def suitable_for_census (c : ScenarioCharacteristics) : Prop :=
  c.population_size ≤ 100 ∧ c.accessibility ∧ c.feasibility

/-- Assigns characteristics to each scenario --/
def scenario_characteristics : Scenario → ScenarioCharacteristics
| Scenario.WaterQuality => ⟨1000, false, false⟩
| Scenario.StudentMentalHealth => ⟨1000000, false, false⟩
| Scenario.ClassHeight => ⟨30, true, true⟩
| Scenario.TVRatings => ⟨10000000, false, false⟩

theorem class_height_most_suitable_for_census :
  ∀ s : Scenario, s ≠ Scenario.ClassHeight →
    ¬(suitable_for_census (scenario_characteristics s)) ∧
    suitable_for_census (scenario_characteristics Scenario.ClassHeight) :=
by sorry

end NUMINAMATH_CALUDE_class_height_most_suitable_for_census_l1709_170945


namespace NUMINAMATH_CALUDE_fraction_comparison_l1709_170994

theorem fraction_comparison : (5555553 : ℚ) / 5555557 > (6666664 : ℚ) / 6666669 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1709_170994


namespace NUMINAMATH_CALUDE_adams_shelves_l1709_170950

theorem adams_shelves (action_figures_per_shelf : ℕ) (num_shelves : ℕ) (total_capacity : ℕ) :
  action_figures_per_shelf = 11 →
  num_shelves = 4 →
  total_capacity = action_figures_per_shelf * num_shelves →
  total_capacity = 44 :=
by sorry

end NUMINAMATH_CALUDE_adams_shelves_l1709_170950


namespace NUMINAMATH_CALUDE_power_function_satisfies_no_equation_l1709_170929

theorem power_function_satisfies_no_equation (a : ℝ) :
  ¬(∀ x y : ℝ, (x*y)^a = x^a + y^a) ∧
  ¬(∀ x y : ℝ, (x+y)^a = x^a * y^a) ∧
  ¬(∀ x y : ℝ, (x+y)^a = x^a + y^a) :=
by sorry

end NUMINAMATH_CALUDE_power_function_satisfies_no_equation_l1709_170929


namespace NUMINAMATH_CALUDE_calculator_prices_and_relations_l1709_170963

/-- The price of two A-brand and three B-brand calculators -/
def total_price_1 : ℝ := 156

/-- The price of three A-brand and one B-brand calculator -/
def total_price_2 : ℝ := 122

/-- The discount rate for A-brand calculators during promotion -/
def discount_rate_A : ℝ := 0.8

/-- The discount rate for B-brand calculators during promotion -/
def discount_rate_B : ℝ := 0.875

/-- The unit price of A-brand calculators -/
def price_A : ℝ := 30

/-- The unit price of B-brand calculators -/
def price_B : ℝ := 32

/-- The function relation for A-brand calculators during promotion -/
def y1 (x : ℝ) : ℝ := 24 * x

/-- The function relation for B-brand calculators during promotion -/
def y2 (x : ℝ) : ℝ := 28 * x

theorem calculator_prices_and_relations :
  (2 * price_A + 3 * price_B = total_price_1) ∧
  (3 * price_A + price_B = total_price_2) ∧
  (∀ x, y1 x = discount_rate_A * price_A * x) ∧
  (∀ x, y2 x = discount_rate_B * price_B * x) :=
sorry

end NUMINAMATH_CALUDE_calculator_prices_and_relations_l1709_170963


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1709_170910

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (ha : a = 1657) (hb : b = 2037) (hr1 : r1 = 6) (hr2 : r2 = 5) :
  Nat.gcd (a - r1) (b - r2) = 127 :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1709_170910


namespace NUMINAMATH_CALUDE_max_area_rectangle_l1709_170932

/-- A rectangle with non-negative length and width. -/
structure Rectangle where
  length : ℝ
  width : ℝ
  length_nonneg : 0 ≤ length
  width_nonneg : 0 ≤ width

/-- The perimeter of a rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- The area of a rectangle. -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- A rectangle with perimeter at least 80. -/
def RectangleWithLargePerimeter := {r : Rectangle // r.perimeter ≥ 80}

theorem max_area_rectangle (r : RectangleWithLargePerimeter) :
  r.val.area ≤ 400 ∧ 
  (r.val.area = 400 ↔ r.val.length = 20 ∧ r.val.width = 20) := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l1709_170932


namespace NUMINAMATH_CALUDE_angle_with_special_complement_supplement_l1709_170980

theorem angle_with_special_complement_supplement : 
  ∀ x : ℝ, 
  (0 ≤ x) ∧ (x ≤ 180) →
  (180 - x = 3 * (90 - x)) →
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_angle_with_special_complement_supplement_l1709_170980


namespace NUMINAMATH_CALUDE_octagon_area_l1709_170972

/-- Given two concentric squares with side length 2 and a line segment AB of length 3/4 between the squares,
    the area of the resulting octagon ABCDEFGH is 6. -/
theorem octagon_area (square_side : ℝ) (AB_length : ℝ) (h1 : square_side = 2) (h2 : AB_length = 3/4) :
  let triangle_area := (1/2) * square_side * AB_length
  let octagon_area := 8 * triangle_area
  octagon_area = 6 := by sorry

end NUMINAMATH_CALUDE_octagon_area_l1709_170972


namespace NUMINAMATH_CALUDE_three_integers_sum_and_ratio_l1709_170941

theorem three_integers_sum_and_ratio : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 90 ∧
  2 * b = 3 * a ∧
  2 * c = 5 * a ∧
  a = 18 ∧ b = 27 ∧ c = 45 := by
sorry

end NUMINAMATH_CALUDE_three_integers_sum_and_ratio_l1709_170941


namespace NUMINAMATH_CALUDE_chips_left_uneaten_l1709_170958

/-- Calculates the number of chips left uneaten when half of a batch of cookies is consumed. -/
theorem chips_left_uneaten (chips_per_cookie : ℕ) (dozens : ℕ) : 
  chips_per_cookie = 7 → dozens = 4 → (dozens * 12 / 2) * chips_per_cookie = 168 := by
  sorry

#check chips_left_uneaten

end NUMINAMATH_CALUDE_chips_left_uneaten_l1709_170958


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1709_170955

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 32 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 3 + a 10 + a 11 = 32

/-- Theorem: If a is an arithmetic sequence satisfying the sum condition,
    then the sum of the 6th and 7th terms is 16 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : a 6 + a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1709_170955


namespace NUMINAMATH_CALUDE_problem_statement_l1709_170944

theorem problem_statement (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) :
  3 * a^2008 - 5 * b^2008 = -5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1709_170944


namespace NUMINAMATH_CALUDE_podcast_storage_theorem_l1709_170924

def podcast_duration : ℕ := 837
def cd_capacity : ℕ := 75

theorem podcast_storage_theorem :
  let num_cds : ℕ := (podcast_duration + cd_capacity - 1) / cd_capacity
  let audio_per_cd : ℚ := podcast_duration / num_cds
  audio_per_cd = 69.75 := by sorry

end NUMINAMATH_CALUDE_podcast_storage_theorem_l1709_170924


namespace NUMINAMATH_CALUDE_quadratic_solution_l1709_170908

theorem quadratic_solution : ∃ x : ℝ, x^2 - 2*x + 1 = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1709_170908


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_eight_consecutive_l1709_170961

theorem three_digit_divisible_by_eight_consecutive : ∃ n : ℕ, 
  100 ≤ n ∧ n ≤ 999 ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 8 → k ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_eight_consecutive_l1709_170961


namespace NUMINAMATH_CALUDE_triangle_number_assignment_l1709_170976

theorem triangle_number_assignment :
  ∀ (A B C D E F : ℕ),
    ({A, B, C, D, E, F} : Finset ℕ) = {1, 2, 3, 4, 5, 6} →
    B + D + E = 14 →
    C + E + F = 12 →
    A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_number_assignment_l1709_170976


namespace NUMINAMATH_CALUDE_sum_of_squares_with_inequality_l1709_170913

theorem sum_of_squares_with_inequality (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ (k : ℕ), n = 5 * k) 
  (h3 : ∃ (a b : ℤ), n = a^2 + b^2) : 
  ∃ (x y : ℤ), n = x^2 + y^2 ∧ x^2 ≥ 4 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_with_inequality_l1709_170913


namespace NUMINAMATH_CALUDE_sin_180_degrees_l1709_170999

theorem sin_180_degrees : Real.sin (180 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_180_degrees_l1709_170999


namespace NUMINAMATH_CALUDE_min_n_value_l1709_170959

theorem min_n_value (m n : ℝ) : 
  (∃ x : ℝ, x^2 + (m - 2023) * x + (n - 1) = 0 ∧ 
   ∀ y : ℝ, y^2 + (m - 2023) * y + (n - 1) = 0 → y = x) → 
  n ≥ 1 ∧ ∃ m₀ : ℝ, ∃ x₀ : ℝ, x₀^2 + (m₀ - 2023) * x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_n_value_l1709_170959


namespace NUMINAMATH_CALUDE_function_value_at_pi_over_four_l1709_170953

theorem function_value_at_pi_over_four (φ : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (x + 2 * φ) - 2 * Real.sin φ * Real.cos (x + φ)
  f (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_pi_over_four_l1709_170953


namespace NUMINAMATH_CALUDE_parallelogram_base_l1709_170933

/-- Given a parallelogram with area 864 square cm and height 24 cm, its base is 36 cm. -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 864 ∧ height = 24 ∧ area = base * height → base = 36 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l1709_170933


namespace NUMINAMATH_CALUDE_root_sum_squares_l1709_170931

theorem root_sum_squares (a : ℝ) : 
  (∃ x y : ℝ, x^2 - 3*a*x + a^2 = 0 ∧ y^2 - 3*a*y + a^2 = 0 ∧ x^2 + y^2 = 1.75) → 
  a = 0.5 ∨ a = -0.5 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l1709_170931


namespace NUMINAMATH_CALUDE_calculate_ants_monroe_ants_l1709_170937

/-- Given a collection of spiders and ants, calculate the number of ants -/
theorem calculate_ants (num_spiders : ℕ) (total_legs : ℕ) (spider_legs : ℕ) (ant_legs : ℕ) : ℕ :=
  let num_ants := (total_legs - num_spiders * spider_legs) / ant_legs
  num_ants

/-- Prove that Monroe has 12 ants in his collection -/
theorem monroe_ants : 
  let num_spiders : ℕ := 8
  let total_legs : ℕ := 136
  let spider_legs : ℕ := 8
  let ant_legs : ℕ := 6
  calculate_ants num_spiders total_legs spider_legs ant_legs = 12 := by
  sorry

end NUMINAMATH_CALUDE_calculate_ants_monroe_ants_l1709_170937


namespace NUMINAMATH_CALUDE_basketball_score_difference_l1709_170904

theorem basketball_score_difference 
  (tim joe ken : ℕ) 
  (h1 : tim > joe)
  (h2 : tim = ken / 2)
  (h3 : tim + joe + ken = 100)
  (h4 : tim = 30) :
  tim - joe = 20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_difference_l1709_170904


namespace NUMINAMATH_CALUDE_working_light_bulbs_l1709_170925

theorem working_light_bulbs (total_lamps : ℕ) (bulbs_per_lamp : ℕ) 
  (lamps_with_two_burnt : ℕ) (lamps_with_one_burnt : ℕ) (lamps_with_three_burnt : ℕ) :
  total_lamps = 60 →
  bulbs_per_lamp = 7 →
  lamps_with_two_burnt = total_lamps / 3 →
  lamps_with_one_burnt = total_lamps / 4 →
  lamps_with_three_burnt = total_lamps / 5 →
  (total_lamps - (lamps_with_two_burnt + lamps_with_one_burnt + lamps_with_three_burnt)) * bulbs_per_lamp +
  lamps_with_two_burnt * (bulbs_per_lamp - 2) +
  lamps_with_one_burnt * (bulbs_per_lamp - 1) +
  lamps_with_three_burnt * (bulbs_per_lamp - 3) = 329 :=
by
  sorry


end NUMINAMATH_CALUDE_working_light_bulbs_l1709_170925


namespace NUMINAMATH_CALUDE_lucy_popsicles_l1709_170909

/-- The maximum number of popsicles Lucy can buy given her funds and the pricing structure -/
def max_popsicles (total_funds : ℚ) (first_tier_price : ℚ) (second_tier_price : ℚ) (first_tier_limit : ℕ) : ℕ :=
  let first_tier_cost := first_tier_limit * first_tier_price
  let remaining_funds := total_funds - first_tier_cost
  let additional_popsicles := (remaining_funds / second_tier_price).floor
  first_tier_limit + additional_popsicles.toNat

/-- Theorem stating that Lucy can buy 15 popsicles -/
theorem lucy_popsicles :
  max_popsicles 25.5 1.75 1.5 8 = 15 := by
  sorry

#eval max_popsicles 25.5 1.75 1.5 8

end NUMINAMATH_CALUDE_lucy_popsicles_l1709_170909


namespace NUMINAMATH_CALUDE_inequality_proof_l1709_170906

theorem inequality_proof (x : ℝ) (hx : x > 0) : (x + 1) * Real.sqrt (x + 1) ≥ Real.sqrt 2 * (x + Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1709_170906


namespace NUMINAMATH_CALUDE_solution_set_is_correct_l1709_170975

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x : ℝ, (f.deriv.deriv) x < f x)
variable (h2 : f 2 = 1)

-- Define the solution set
def solution_set := {x : ℝ | f x > Real.exp (x - 2)}

-- State the theorem
theorem solution_set_is_correct : solution_set f = Set.Iio 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_correct_l1709_170975


namespace NUMINAMATH_CALUDE_mall_walking_methods_l1709_170901

/-- The number of entrances in the mall -/
def num_entrances : ℕ := 4

/-- The number of different walking methods through the mall -/
def num_walking_methods : ℕ := num_entrances * (num_entrances - 1)

/-- Theorem stating the number of different walking methods -/
theorem mall_walking_methods :
  num_walking_methods = 12 :=
sorry

end NUMINAMATH_CALUDE_mall_walking_methods_l1709_170901


namespace NUMINAMATH_CALUDE_initial_tomatoes_count_l1709_170970

def tomatoes_picked_yesterday : ℕ := 56
def tomatoes_picked_today : ℕ := 41
def tomatoes_remaining_after_yesterday : ℕ := 104

theorem initial_tomatoes_count :
  tomatoes_picked_yesterday + tomatoes_picked_today + tomatoes_remaining_after_yesterday = 201 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_tomatoes_count_l1709_170970


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1709_170989

/-- A polynomial with real coefficients -/
def g (p q r s : ℝ) (x : ℂ) : ℂ := x^4 + p*x^3 + q*x^2 + r*x + s

/-- Theorem stating that if g(1+i) = 0 and g(3i) = 0, then p + q + r + s = 9 -/
theorem sum_of_coefficients (p q r s : ℝ) :
  g p q r s (1 + Complex.I) = 0 →
  g p q r s (3 * Complex.I) = 0 →
  p + q + r + s = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1709_170989


namespace NUMINAMATH_CALUDE_modulus_of_i_times_one_plus_i_l1709_170952

theorem modulus_of_i_times_one_plus_i : Complex.abs (Complex.I * (1 + Complex.I)) = 1 := by sorry

end NUMINAMATH_CALUDE_modulus_of_i_times_one_plus_i_l1709_170952


namespace NUMINAMATH_CALUDE_minimum_at_one_positive_when_minimum_less_than_one_l1709_170991

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (2 * x + 1) * (a * x - 1) / x

-- Theorem 1: If the minimum point of f(x) is at x_0 = 1, then a = 1
theorem minimum_at_one (a : ℝ) :
  (∀ x > 0, f a x ≥ f a 1) → a = 1 := by sorry

-- Theorem 2: If 0 < x_0 < 1, where x_0 is the minimum point of f(x), then f(x) > 0 for all x > 0
theorem positive_when_minimum_less_than_one (a : ℝ) (x_0 : ℝ) :
  (0 < x_0 ∧ x_0 < 1) →
  (∀ x > 0, f a x ≥ f a x_0) →
  (∀ x > 0, f a x > 0) := by sorry

end NUMINAMATH_CALUDE_minimum_at_one_positive_when_minimum_less_than_one_l1709_170991


namespace NUMINAMATH_CALUDE_full_price_revenue_is_1250_l1709_170942

/-- Represents the revenue from full-price tickets in a school club's ticket sale. -/
def revenue_full_price (full_price : ℚ) (num_full_price : ℕ) : ℚ :=
  full_price * num_full_price

/-- Represents the total revenue from all tickets sold. -/
def total_revenue (full_price : ℚ) (num_full_price : ℕ) (num_discount_price : ℕ) : ℚ :=
  revenue_full_price full_price num_full_price + (full_price / 3) * num_discount_price

/-- Theorem stating that the revenue from full-price tickets is $1250. -/
theorem full_price_revenue_is_1250 :
  ∃ (full_price : ℚ) (num_full_price num_discount_price : ℕ),
    num_full_price + num_discount_price = 200 ∧
    total_revenue full_price num_full_price num_discount_price = 2500 ∧
    revenue_full_price full_price num_full_price = 1250 :=
sorry

end NUMINAMATH_CALUDE_full_price_revenue_is_1250_l1709_170942


namespace NUMINAMATH_CALUDE_product_of_primes_l1709_170905

theorem product_of_primes (p q : ℕ) (hp : Prime p) (hq : Prime q)
  (h_range : 15 < p * q ∧ p * q < 36)
  (hp_range : 2 < p ∧ p < 6)
  (hq_range : 8 < q ∧ q < 24) :
  p * q = 33 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_l1709_170905


namespace NUMINAMATH_CALUDE_interest_difference_proof_l1709_170918

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates the difference between principal and interest -/
def principalInterestDifference (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal - simpleInterest principal rate time

theorem interest_difference_proof :
  let principal : ℝ := 1100
  let rate : ℝ := 0.06
  let time : ℝ := 8
  principalInterestDifference principal rate time = 572 := by
sorry

end NUMINAMATH_CALUDE_interest_difference_proof_l1709_170918


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l1709_170914

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l1709_170914


namespace NUMINAMATH_CALUDE_alternating_fraction_value_l1709_170985

theorem alternating_fraction_value :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_alternating_fraction_value_l1709_170985


namespace NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l1709_170920

/-- Given a 50% profit percent, prove that the ratio of the cost price to the selling price is 2:3 -/
theorem cost_to_selling_price_ratio (cost_price selling_price : ℝ) 
  (h_positive : cost_price > 0 ∧ selling_price > 0)
  (h_profit : selling_price = cost_price * 1.5) : 
  cost_price / selling_price = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l1709_170920


namespace NUMINAMATH_CALUDE_perfect_square_15AB9_l1709_170968

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def has_form_15AB9 (n : ℕ) : Prop :=
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ n = 15000 + A * 100 + B * 10 + 9

theorem perfect_square_15AB9 (n : ℕ) (h1 : is_five_digit n) (h2 : has_form_15AB9 n) (h3 : is_perfect_square n) :
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ n = 15000 + A * 100 + B * 10 + 9 ∧ A + B = 3 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_15AB9_l1709_170968


namespace NUMINAMATH_CALUDE_ten_thousandths_digit_of_437_div_128_l1709_170962

theorem ten_thousandths_digit_of_437_div_128 :
  (437 : ℚ) / 128 = 3 + 4/10 + 1/100 + 4/1000 + 6/10000 + 8/100000 + 7/1000000 + 5/10000000 :=
by sorry

end NUMINAMATH_CALUDE_ten_thousandths_digit_of_437_div_128_l1709_170962


namespace NUMINAMATH_CALUDE_sodium_reduction_is_one_third_l1709_170996

def sodium_reduction_fraction (salt_teaspoons : ℕ) (parmesan_oz : ℕ) 
  (salt_sodium_per_tsp : ℕ) (parmesan_sodium_per_oz : ℕ) 
  (parmesan_reduction_oz : ℕ) : ℚ :=
  let original_sodium := salt_teaspoons * salt_sodium_per_tsp + parmesan_oz * parmesan_sodium_per_oz
  let reduced_sodium := salt_teaspoons * salt_sodium_per_tsp + (parmesan_oz - parmesan_reduction_oz) * parmesan_sodium_per_oz
  (original_sodium - reduced_sodium : ℚ) / original_sodium

theorem sodium_reduction_is_one_third :
  sodium_reduction_fraction 2 8 50 25 4 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sodium_reduction_is_one_third_l1709_170996


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l1709_170954

def polynomial (x : ℝ) := 3 * (x^4 + 2*x^3 + 5*x^2 + 2)

theorem sum_of_squares_of_coefficients : 
  (3^2 : ℝ) + (6^2 : ℝ) + (15^2 : ℝ) + (6^2 : ℝ) = 306 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l1709_170954


namespace NUMINAMATH_CALUDE_power_of_three_equivalence_l1709_170983

theorem power_of_three_equivalence : 
  (1 / 2 : ℝ) * (3 : ℝ)^21 - (1 / 3 : ℝ) * (3 : ℝ)^20 = (7 / 6 : ℝ) * (3 : ℝ)^20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equivalence_l1709_170983


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l1709_170977

-- Define the function f
def f (x : ℝ) : ℝ := x

-- Theorem stating that f satisfies all conditions
theorem f_satisfies_conditions :
  (∀ x : ℝ, f (-x) + f x = 0) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) :=
by sorry

#check f_satisfies_conditions

end NUMINAMATH_CALUDE_f_satisfies_conditions_l1709_170977


namespace NUMINAMATH_CALUDE_trapezoid_area_l1709_170951

/-- Represents a triangle in the diagram -/
structure Triangle where
  area : ℝ

/-- Represents the isosceles triangle PQR -/
structure IsoscelesTriangle extends Triangle

/-- Represents the trapezoid TQRS -/
structure Trapezoid where
  area : ℝ

/-- The problem setup -/
axiom smallest_triangle : Triangle
axiom smallest_triangle_area : smallest_triangle.area = 2

axiom PQR : IsoscelesTriangle
axiom PQR_area : PQR.area = 72

axiom PTQ : Triangle
axiom PTQ_composition : PTQ.area = 5 * smallest_triangle.area

axiom TQRS : Trapezoid
axiom TQRS_formation : TQRS.area = PQR.area - PTQ.area

/-- The theorem to prove -/
theorem trapezoid_area : TQRS.area = 62 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1709_170951


namespace NUMINAMATH_CALUDE_students_walking_home_l1709_170992

theorem students_walking_home (bus auto bike skate : ℚ) 
  (h_bus : bus = 1/3)
  (h_auto : auto = 1/5)
  (h_bike : bike = 1/8)
  (h_skate : skate = 1/15) :
  1 - (bus + auto + bike + skate) = 11/40 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l1709_170992


namespace NUMINAMATH_CALUDE_arrival_time_difference_l1709_170982

/-- The distance to the campsite in miles -/
def distance : ℝ := 3

/-- Jill's hiking speed in miles per hour -/
def jill_speed : ℝ := 6

/-- Jack's hiking speed in miles per hour -/
def jack_speed : ℝ := 3

/-- Conversion factor from hours to minutes -/
def minutes_per_hour : ℝ := 60

/-- The time difference in minutes between Jill and Jack's arrival at the campsite -/
theorem arrival_time_difference : 
  (distance / jack_speed - distance / jill_speed) * minutes_per_hour = 30 := by
  sorry

end NUMINAMATH_CALUDE_arrival_time_difference_l1709_170982


namespace NUMINAMATH_CALUDE_distribution_ways_l1709_170912

theorem distribution_ways (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  (k : ℕ) ^ n = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribution_ways_l1709_170912


namespace NUMINAMATH_CALUDE_solve_equation_l1709_170907

theorem solve_equation (x : ℤ) (h : 9773 + x = 13200) : x = 3427 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1709_170907


namespace NUMINAMATH_CALUDE_tomatoes_left_after_yesterday_l1709_170915

theorem tomatoes_left_after_yesterday (initial_tomatoes picked_yesterday : ℕ) : 
  initial_tomatoes = 160 → picked_yesterday = 56 → initial_tomatoes - picked_yesterday = 104 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_after_yesterday_l1709_170915


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1709_170948

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) : 
  Real.tan α = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1709_170948


namespace NUMINAMATH_CALUDE_even_mono_decreasing_range_l1709_170965

/-- A function f: ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is monotonically decreasing on [0,+∞) -/
def IsMonoDecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem even_mono_decreasing_range (f : ℝ → ℝ) (m : ℝ) 
    (h_even : IsEven f) 
    (h_mono : IsMonoDecreasingOnNonnegative f) 
    (h_ineq : f m > f (1 - m)) : 
  m < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_even_mono_decreasing_range_l1709_170965


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l1709_170981

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_expression :
  6 * (2 - i) + 4 * i * (6 - i) = 16 + 18 * i :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l1709_170981


namespace NUMINAMATH_CALUDE_sum_of_digits_difference_l1709_170934

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for all numbers in a list -/
def sumOfDigitsForList (list : List ℕ) : ℕ := sorry

/-- List of odd numbers from 1 to 99 -/
def oddNumbers : List ℕ := sorry

/-- List of even numbers from 2 to 100 -/
def evenNumbers : List ℕ := sorry

theorem sum_of_digits_difference : 
  sumOfDigitsForList oddNumbers - sumOfDigitsForList evenNumbers = 49 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_difference_l1709_170934


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1709_170947

/-- The function f(x) = x^2 + bx + 3 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- Theorem: -3 is not in the range of f(x) = x^2 + bx + 3 if and only if b ∈ (-2√6, 2√6) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, f b x ≠ -3) ↔ b ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1709_170947


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l1709_170919

theorem min_x_prime_factorization (x y : ℕ+) (h : 13 * x^7 = 17 * y^11) :
  ∃ (a b c d : ℕ),
    x = a^c * b^d ∧
    x ≥ 17^15 * 13^2 ∧
    (∀ x' : ℕ+, 13 * x'^7 = 17 * y^11 → x' ≥ x) ∧
    a + b + c + d = 47 := by
  sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l1709_170919


namespace NUMINAMATH_CALUDE_compound_composition_l1709_170949

/-- Represents the number of atoms of each element in a compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem compound_composition :
  ∃ (c : Compound),
    c.hydrogen = 8 ∧
    c.oxygen = 7 ∧
    molecularWeight c 12.01 1.01 16.00 = 192 ∧
    c.carbon = 6 := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l1709_170949


namespace NUMINAMATH_CALUDE_smallest_seating_l1709_170939

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  chairs : Nat
  seated : Nat

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone already seated. -/
def satisfiesCondition (table : CircularTable) : Prop :=
  ∀ (new_seat : Nat), new_seat < table.chairs → 
    ∃ (adjacent_seat : Nat), adjacent_seat < table.chairs ∧ 
      (adjacent_seat = (new_seat + 1) % table.chairs ∨ 
       adjacent_seat = (new_seat + table.chairs - 1) % table.chairs)

/-- Theorem stating the smallest number of people that can be seated to satisfy the condition. -/
theorem smallest_seating (table : CircularTable) : 
  table.chairs = 90 → 
  (∀ n < 23, ¬(satisfiesCondition ⟨90, n⟩)) ∧ 
  satisfiesCondition ⟨90, 23⟩ := by
  sorry

end NUMINAMATH_CALUDE_smallest_seating_l1709_170939


namespace NUMINAMATH_CALUDE_problem_solution_l1709_170922

/-- Proposition p -/
def p (m x : ℝ) : Prop := m * x + 1 > 0

/-- Proposition q -/
def q (x : ℝ) : Prop := (3 * x - 1) * (x + 2) < 0

theorem problem_solution (m : ℝ) (hm : m > 0) :
  (∃ a b : ℝ, a < b ∧ 
    (m = 1 → 
      (∀ x : ℝ, p m x ∧ q x ↔ a < x ∧ x < b) ∧
      a = -1 ∧ b = 1/3)) ∧
  (∀ x : ℝ, q x → p m x) ∧ 
  (∃ x : ℝ, p m x ∧ ¬q x) →
  0 < m ∧ m ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1709_170922


namespace NUMINAMATH_CALUDE_f_composition_half_equals_one_l1709_170921

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x|

-- State the theorem
theorem f_composition_half_equals_one : f (f (1/2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_half_equals_one_l1709_170921


namespace NUMINAMATH_CALUDE_gmat_question_percentage_l1709_170902

theorem gmat_question_percentage
  (second_correct : Real)
  (neither_correct : Real)
  (both_correct : Real)
  (h1 : second_correct = 0.8)
  (h2 : neither_correct = 0.05)
  (h3 : both_correct = 0.7) :
  ∃ (first_correct : Real),
    first_correct = 0.85 ∧
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by
  sorry

end NUMINAMATH_CALUDE_gmat_question_percentage_l1709_170902


namespace NUMINAMATH_CALUDE_second_player_can_win_l1709_170988

/-- A function representing a player's strategy for choosing digits. -/
def Strategy := Nat → Fin 5

/-- The result of a game where two players alternate choosing digits. -/
def GameResult (s1 s2 : Strategy) : Fin 9 :=
  (List.range 30).foldl
    (λ acc i => (acc + if i % 2 = 0 then s1 i else s2 i) % 9)
    0

/-- Theorem stating that the second player can always ensure divisibility by 9. -/
theorem second_player_can_win :
  ∀ s1 : Strategy, ∃ s2 : Strategy, GameResult s1 s2 = 0 :=
sorry

end NUMINAMATH_CALUDE_second_player_can_win_l1709_170988


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1709_170917

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1709_170917


namespace NUMINAMATH_CALUDE_unique_three_digit_odd_sum_27_l1709_170969

/-- A three-digit number is a natural number between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The digit sum of a natural number is the sum of its digits. -/
def DigitSum (n : ℕ) : ℕ := sorry

/-- A number is odd if it leaves a remainder of 1 when divided by 2. -/
def IsOdd (n : ℕ) : Prop := n % 2 = 1

/-- There is exactly one three-digit number with a digit sum of 27 that is odd. -/
theorem unique_three_digit_odd_sum_27 : 
  ∃! n : ℕ, ThreeDigitNumber n ∧ DigitSum n = 27 ∧ IsOdd n := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_odd_sum_27_l1709_170969


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1709_170936

/-- A geometric sequence with positive terms where a_1 and a_{99} are roots of x^2 - 10x + 16 = 0 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) ∧
  a 1 * a 99 = 16 ∧
  a 1 + a 99 = 10

theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 20 * a 50 * a 80 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1709_170936


namespace NUMINAMATH_CALUDE_picture_on_wall_l1709_170900

theorem picture_on_wall (wall_width picture_width : ℝ) 
  (hw : wall_width = 22) 
  (hp : picture_width = 4) : 
  (wall_width - picture_width) / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_picture_on_wall_l1709_170900


namespace NUMINAMATH_CALUDE_no_x_term_condition_l1709_170978

theorem no_x_term_condition (a : ℝ) : 
  (∀ x, (-2*x + a)*(x - 1) = -2*x^2 - a) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_x_term_condition_l1709_170978


namespace NUMINAMATH_CALUDE_circle_radius_relation_l1709_170926

theorem circle_radius_relation (square_area : ℝ) (small_circle_circumference : ℝ) :
  square_area = 784 →
  small_circle_circumference = 8 →
  ∃ (x : ℝ) (r_s r_l : ℝ),
    r_s = 4 / π ∧
    r_l = 14 ∧
    r_l = x - (1/3) * r_s ∧
    x = 14 + 4 / (3 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_relation_l1709_170926


namespace NUMINAMATH_CALUDE_ninth_term_is_512_l1709_170911

/-- Given a geometric sequence where:
  * The first term is 2
  * The common ratio is 2
  * n is the term number
  This function calculates the nth term of the sequence -/
def geometricSequenceTerm (n : ℕ) : ℕ := 2 * 2^(n - 1)

/-- Theorem stating that the 9th term of the geometric sequence is 512 -/
theorem ninth_term_is_512 : geometricSequenceTerm 9 = 512 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_512_l1709_170911


namespace NUMINAMATH_CALUDE_min_value_theorem_l1709_170993

theorem min_value_theorem (a : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  (4 / a) + (1 / (2 - a)) ≥ (9 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1709_170993


namespace NUMINAMATH_CALUDE_complementary_angle_of_25_l1709_170987

def complementary_angle (x : ℝ) : ℝ := 90 - x

theorem complementary_angle_of_25 :
  complementary_angle 25 = 65 :=
by sorry

end NUMINAMATH_CALUDE_complementary_angle_of_25_l1709_170987


namespace NUMINAMATH_CALUDE_specific_jump_record_l1709_170928

/-- The standard distance for the long jump competition -/
def standard_distance : ℝ := 4.00

/-- Calculate the recorded result for a given jump distance -/
def record_jump (jump_distance : ℝ) : ℝ :=
  jump_distance - standard_distance

/-- The specific jump distance we want to prove about -/
def specific_jump : ℝ := 3.85

/-- Theorem stating that the record for the specific jump should be -0.15 -/
theorem specific_jump_record :
  record_jump specific_jump = -0.15 := by sorry

end NUMINAMATH_CALUDE_specific_jump_record_l1709_170928


namespace NUMINAMATH_CALUDE_chairs_to_remove_proof_l1709_170943

/-- Calculates the number of chairs to remove given the initial setup and expected attendance --/
def chairs_to_remove (chairs_per_row : ℕ) (total_chairs : ℕ) (expected_attendees : ℕ) : ℕ :=
  let rows_needed := (expected_attendees + chairs_per_row - 1) / chairs_per_row
  let chairs_needed := rows_needed * chairs_per_row
  total_chairs - chairs_needed

/-- Proves that given the specific conditions, 105 chairs should be removed --/
theorem chairs_to_remove_proof :
  chairs_to_remove 15 300 180 = 105 := by
  sorry

#eval chairs_to_remove 15 300 180

end NUMINAMATH_CALUDE_chairs_to_remove_proof_l1709_170943


namespace NUMINAMATH_CALUDE_melissa_banana_count_l1709_170990

/-- Calculates the final number of bananas Melissa has -/
def melissasFinalBananas (initialBananas buyMultiplier sharedBananas : ℕ) : ℕ :=
  let remainingBananas := initialBananas - sharedBananas
  let boughtBananas := buyMultiplier * remainingBananas
  remainingBananas + boughtBananas

theorem melissa_banana_count :
  melissasFinalBananas 88 3 4 = 336 := by
  sorry

end NUMINAMATH_CALUDE_melissa_banana_count_l1709_170990


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1709_170927

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 2 * x) = 8 → x = -30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1709_170927


namespace NUMINAMATH_CALUDE_continuous_fraction_solution_l1709_170964

theorem continuous_fraction_solution :
  ∃ y : ℝ, y > 0 ∧ y = 3 + 3 / (2 + 3 / y) ∧ y = (3 + 3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continuous_fraction_solution_l1709_170964


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1709_170938

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 3*x < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1709_170938


namespace NUMINAMATH_CALUDE_min_cuts_for_100_pieces_l1709_170935

/-- Represents the number of pieces a cube is divided into after making cuts -/
def num_pieces (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)

/-- Theorem stating that 11 is the minimum number of cuts needed to divide a cube into 100 pieces -/
theorem min_cuts_for_100_pieces :
  ∃ (a b c : ℕ), num_pieces a b c = 100 ∧ a + b + c = 11 ∧
  (∀ (x y z : ℕ), num_pieces x y z ≥ 100 → x + y + z ≥ 11) :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_100_pieces_l1709_170935


namespace NUMINAMATH_CALUDE_cosine_sine_sum_equals_half_l1709_170986

theorem cosine_sine_sum_equals_half : 
  Real.cos (80 * π / 180) * Real.cos (20 * π / 180) + 
  Real.sin (100 * π / 180) * Real.sin (380 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_equals_half_l1709_170986


namespace NUMINAMATH_CALUDE_francis_fruit_cups_l1709_170940

/-- The cost of a breakfast given the number of muffins and fruit cups -/
def breakfast_cost (muffins fruit_cups : ℕ) : ℕ := 2 * muffins + 3 * fruit_cups

/-- The problem statement -/
theorem francis_fruit_cups : ∃ f : ℕ, 
  breakfast_cost 2 f + breakfast_cost 2 1 = 17 ∧ f = 2 := by sorry

end NUMINAMATH_CALUDE_francis_fruit_cups_l1709_170940


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1709_170998

/-- A geometric sequence where the sum of every two consecutive terms forms another geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 2) + a (n + 3) = r * (a n + a (n + 1))

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 = 1 →
  a 3 + a 4 = 2 →
  a 9 + a 10 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1709_170998


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1709_170974

/-- The line x + ay = 3 is tangent to the circle (x-1)² + y² = 2 if and only if a = ±1 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, (x + a * y = 3) → ((x - 1)^2 + y^2 = 2) → 
   (∀ x' y' : ℝ, (x' + a * y' = 3) → ((x' - 1)^2 + y'^2 ≥ 2))) ↔ 
  (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1709_170974


namespace NUMINAMATH_CALUDE_initial_erasers_eq_taken_plus_left_l1709_170957

/-- The initial number of erasers in the box -/
def initial_erasers : ℕ := 69

/-- The number of erasers Doris took out of the box -/
def erasers_taken : ℕ := 54

/-- The number of erasers left in the box -/
def erasers_left : ℕ := 15

/-- Theorem stating that the initial number of erasers is equal to
    the sum of erasers taken and erasers left -/
theorem initial_erasers_eq_taken_plus_left :
  initial_erasers = erasers_taken + erasers_left := by
  sorry

end NUMINAMATH_CALUDE_initial_erasers_eq_taken_plus_left_l1709_170957
