import Mathlib

namespace NUMINAMATH_CALUDE_interior_angle_regular_hexagon_l81_8127

/-- The measure of an interior angle of a regular hexagon -/
def interior_angle_measure_hexagon : ℝ := 120

/-- A regular hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The measure of each interior angle in a regular hexagon is 120° -/
theorem interior_angle_regular_hexagon :
  interior_angle_measure_hexagon = (((hexagon_sides - 2 : ℕ) * 180) / hexagon_sides : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_interior_angle_regular_hexagon_l81_8127


namespace NUMINAMATH_CALUDE_stick_markings_l81_8113

theorem stick_markings (stick_length : ℝ) (red_mark : ℝ) (blue_mark : ℝ) : 
  stick_length = 12 →
  red_mark = stick_length / 2 →
  blue_mark = red_mark / 2 →
  red_mark - blue_mark = 3 := by
sorry

end NUMINAMATH_CALUDE_stick_markings_l81_8113


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l81_8125

theorem gasoline_price_increase (original_price original_quantity : ℝ) 
  (h1 : original_price > 0) (h2 : original_quantity > 0) : 
  let new_quantity := original_quantity * (1 - 0.1)
  let new_total_cost := original_price * original_quantity * (1 + 0.08)
  let price_increase_factor := new_total_cost / (new_quantity * original_price)
  price_increase_factor = 1.2 := by sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l81_8125


namespace NUMINAMATH_CALUDE_miles_driven_with_budget_l81_8134

-- Define the given conditions
def miles_per_gallon : ℝ := 32
def cost_per_gallon : ℝ := 4
def budget : ℝ := 20

-- Define the theorem
theorem miles_driven_with_budget :
  (budget / cost_per_gallon) * miles_per_gallon = 160 := by
  sorry

end NUMINAMATH_CALUDE_miles_driven_with_budget_l81_8134


namespace NUMINAMATH_CALUDE_difference_of_squares_l81_8162

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 10) : x^2 - y^2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l81_8162


namespace NUMINAMATH_CALUDE_last_two_digits_product_l81_8189

/-- Given an integer n, returns the tens digit of n. -/
def tens_digit (n : ℤ) : ℤ := (n / 10) % 10

/-- Given an integer n, returns the units digit of n. -/
def units_digit (n : ℤ) : ℤ := n % 10

/-- Theorem: For any integer n divisible by 6 with the sum of its last two digits being 15,
    the product of its last two digits is either 56 or 54. -/
theorem last_two_digits_product (n : ℤ) 
  (div_by_6 : n % 6 = 0)
  (sum_15 : tens_digit n + units_digit n = 15) :
  tens_digit n * units_digit n = 56 ∨ tens_digit n * units_digit n = 54 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l81_8189


namespace NUMINAMATH_CALUDE_courageous_iff_coprime_l81_8128

/-- A function is courageous if it and its 100 shifts are all bijections -/
def IsCourageous (n : ℕ) (g : ZMod n → ZMod n) : Prop :=
  Function.Bijective g ∧
  ∀ k : Fin 101, Function.Bijective (λ x => g x + k * x)

/-- The main theorem: existence of a courageous function is equivalent to n being coprime to 101! -/
theorem courageous_iff_coprime (n : ℕ) :
  (∃ g : ZMod n → ZMod n, IsCourageous n g) ↔ Nat.Coprime n (Nat.factorial 101) := by
  sorry

end NUMINAMATH_CALUDE_courageous_iff_coprime_l81_8128


namespace NUMINAMATH_CALUDE_camp_kids_count_camp_kids_count_proof_l81_8184

theorem camp_kids_count : ℕ → Prop :=
  fun total_kids =>
    let soccer_kids := total_kids / 2
    let morning_soccer_kids := soccer_kids / 4
    let afternoon_soccer_kids := soccer_kids - morning_soccer_kids
    afternoon_soccer_kids = 750 ∧ total_kids = 2000

-- The proof goes here
theorem camp_kids_count_proof : ∃ n : ℕ, camp_kids_count n := by
  sorry

end NUMINAMATH_CALUDE_camp_kids_count_camp_kids_count_proof_l81_8184


namespace NUMINAMATH_CALUDE_range_of_f_l81_8179

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -3 ≤ y ∧ y ≤ 14 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l81_8179


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l81_8154

theorem imaginary_part_of_complex_expression : 
  Complex.im ((2 * Complex.I) / (1 - Complex.I) + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l81_8154


namespace NUMINAMATH_CALUDE_smallest_common_nondivisor_l81_8148

theorem smallest_common_nondivisor : 
  ∃ (a : ℕ), a > 0 ∧ 
  (∀ (k : ℕ), 0 < k ∧ k < a → (Nat.gcd k 77 = 1 ∨ Nat.gcd k 66 = 1)) ∧ 
  Nat.gcd a 77 > 1 ∧ Nat.gcd a 66 > 1 ∧ 
  a = 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_nondivisor_l81_8148


namespace NUMINAMATH_CALUDE_basketball_five_bounces_l81_8172

/-- Calculates the total distance traveled by a basketball dropped from a given height,
    rebounding to a fraction of its previous height, for a given number of bounces. -/
def basketballDistance (initialHeight : ℝ) (reboundFraction : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- Theorem stating that a basketball dropped from 80 feet, rebounding to three-quarters
    of its previous height each time, will have traveled 408.125 feet when it hits the
    ground for the fifth time. -/
theorem basketball_five_bounces :
  basketballDistance 80 0.75 5 = 408.125 := by sorry

end NUMINAMATH_CALUDE_basketball_five_bounces_l81_8172


namespace NUMINAMATH_CALUDE_quadratic_composition_roots_l81_8164

/-- Given two quadratic trinomials f and g such that f(g(x)) = 0 and g(f(x)) = 0 have no real roots,
    at least one of f(f(x)) = 0 or g(g(x)) = 0 has no real roots. -/
theorem quadratic_composition_roots
  (f g : ℝ → ℝ)
  (hf : ∀ x, ∃ a b c : ℝ, f x = a * x^2 + b * x + c)
  (hg : ∀ x, ∃ d e f : ℝ, g x = d * x^2 + e * x + f)
  (hfg : ¬∃ x, f (g x) = 0)
  (hgf : ¬∃ x, g (f x) = 0) :
  (¬∃ x, f (f x) = 0) ∨ (¬∃ x, g (g x) = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_composition_roots_l81_8164


namespace NUMINAMATH_CALUDE_probability_at_least_two_successes_probability_one_from_pair_contest_probabilities_l81_8101

/-- The probability of getting at least two successes in three independent trials 
    with a 50% success rate for each trial -/
theorem probability_at_least_two_successes : ℝ := by sorry

/-- The probability of selecting exactly one item from a specific pair 
    when selecting 2 out of 4 items -/
theorem probability_one_from_pair : ℝ := by sorry

/-- Proof of the probabilities for the contest scenario -/
theorem contest_probabilities : 
  probability_at_least_two_successes = 1/2 ∧ 
  probability_one_from_pair = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_successes_probability_one_from_pair_contest_probabilities_l81_8101


namespace NUMINAMATH_CALUDE_simplify_fraction_l81_8111

theorem simplify_fraction : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l81_8111


namespace NUMINAMATH_CALUDE_lines_parallel_when_perpendicular_to_parallel_planes_l81_8124

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem lines_parallel_when_perpendicular_to_parallel_planes
  (a b : Line) (α β : Plane)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : perpendicular a α)
  (h4 : perpendicular b β)
  (h5 : planeParallel α β) :
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_when_perpendicular_to_parallel_planes_l81_8124


namespace NUMINAMATH_CALUDE_safe_plucking_percentage_is_correct_l81_8194

/-- The number of tail feathers each flamingo has -/
def feathers_per_flamingo : ℕ := 20

/-- The number of boas Milly needs to make -/
def number_of_boas : ℕ := 12

/-- The number of feathers needed for each boa -/
def feathers_per_boa : ℕ := 200

/-- The number of flamingoes Milly needs to harvest -/
def flamingoes_to_harvest : ℕ := 480

/-- The percentage of tail feathers Milly can safely pluck from each flamingo -/
def safe_plucking_percentage : ℚ := 25 / 100

theorem safe_plucking_percentage_is_correct :
  safe_plucking_percentage = 
    (number_of_boas * feathers_per_boa) / 
    (flamingoes_to_harvest * feathers_per_flamingo) := by
  sorry

end NUMINAMATH_CALUDE_safe_plucking_percentage_is_correct_l81_8194


namespace NUMINAMATH_CALUDE_cloth_sold_proof_l81_8120

/-- Represents the profit per meter of cloth in Rs. -/
def profit_per_meter : ℕ := 35

/-- Represents the total profit earned in Rs. -/
def total_profit : ℕ := 1400

/-- Represents the number of meters of cloth sold -/
def meters_sold : ℕ := total_profit / profit_per_meter

theorem cloth_sold_proof : meters_sold = 40 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sold_proof_l81_8120


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l81_8195

/-- The parabola defined by y = 2x^2 + 3 intersects the y-axis at the point (0, 3) -/
theorem parabola_y_axis_intersection :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 + 3
  (0, f 0) = (0, 3) := by sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l81_8195


namespace NUMINAMATH_CALUDE_school_survey_sample_size_l81_8156

/-- Represents a survey conducted in a school -/
structure SchoolSurvey where
  total_students : ℕ
  selected_students : ℕ

/-- The sample size of a school survey is the number of selected students -/
def sample_size (survey : SchoolSurvey) : ℕ := survey.selected_students

/-- Theorem: For a school with 3600 students and 200 randomly selected for a survey,
    the sample size is 200 -/
theorem school_survey_sample_size :
  let survey := SchoolSurvey.mk 3600 200
  sample_size survey = 200 := by
  sorry

end NUMINAMATH_CALUDE_school_survey_sample_size_l81_8156


namespace NUMINAMATH_CALUDE_islander_liar_count_l81_8131

/-- Represents the type of islander: knight or liar -/
inductive IslanderType
| Knight
| Liar

/-- Represents a group of islanders making a statement -/
structure IslanderGroup where
  count : Nat
  statement : Nat

/-- The main theorem to prove -/
theorem islander_liar_count 
  (total_islanders : Nat)
  (group1 group2 group3 : IslanderGroup)
  (h1 : total_islanders = 19)
  (h2 : group1.count = 3 ∧ group1.statement = 3)
  (h3 : group2.count = 6 ∧ group2.statement = 6)
  (h4 : group3.count = 9 ∧ group3.statement = 9)
  (h5 : group1.count + group2.count + group3.count = total_islanders) :
  ∃ (liar_count : Nat), (liar_count = 9 ∨ liar_count = 18 ∨ liar_count = 19) ∧
    (∀ (x : Nat), x ≠ 9 ∧ x ≠ 18 ∧ x ≠ 19 → x ≠ liar_count) :=
by sorry

end NUMINAMATH_CALUDE_islander_liar_count_l81_8131


namespace NUMINAMATH_CALUDE_inequality_solution_set_l81_8133

theorem inequality_solution_set (x : ℝ) : 
  (x ≠ 0 ∧ (x - 1) / x ≤ 0) ↔ (0 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l81_8133


namespace NUMINAMATH_CALUDE_function_property_implications_l81_8169

def FunctionProperty (f : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, f x = f (x^2 + x + 1)

theorem function_property_implications
  (f : ℤ → ℤ) (h : FunctionProperty f) :
  ((∀ x : ℤ, f x = f (-x)) → (∃ c : ℤ, ∀ x : ℤ, f x = c)) ∧
  ((∀ x : ℤ, f (-x) = -f x) → (∀ x : ℤ, f x = 0)) := by
  sorry

end NUMINAMATH_CALUDE_function_property_implications_l81_8169


namespace NUMINAMATH_CALUDE_total_money_of_three_people_l81_8121

/-- Given three people A, B, and C with some money between them, prove that their total amount is 400. -/
theorem total_money_of_three_people (a b c : ℕ) : 
  a + c = 300 →
  b + c = 150 →
  c = 50 →
  a + b + c = 400 := by
sorry

end NUMINAMATH_CALUDE_total_money_of_three_people_l81_8121


namespace NUMINAMATH_CALUDE_cubes_not_touching_foil_l81_8103

/-- Represents the dimensions of a rectangular prism --/
structure PrismDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of cubes in a rectangular prism --/
def cubeCount (d : PrismDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Theorem: The number of cubes not touching tin foil in the described prism is 128 --/
theorem cubes_not_touching_foil : ∃ (inner outer : PrismDimensions),
  -- The width of the foil-covered prism is 10 inches
  outer.width = 10 ∧
  -- The width of the inner figure is twice its length and height
  inner.width = 2 * inner.length ∧
  inner.width = 2 * inner.height ∧
  -- There is a 1-inch layer of cubes touching the foil on all sides
  outer.length = inner.length + 2 ∧
  outer.width = inner.width + 2 ∧
  outer.height = inner.height + 2 ∧
  -- The number of cubes not touching any tin foil is 128
  cubeCount inner = 128 := by
  sorry


end NUMINAMATH_CALUDE_cubes_not_touching_foil_l81_8103


namespace NUMINAMATH_CALUDE_function_property_l81_8183

/-- Iterated function application -/
def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (iterate f n x)

/-- The main theorem -/
theorem function_property (f : ℕ → ℕ) 
  (h : ∀ x y, iterate f (x + 1) y + iterate f (y + 1) x = 2 * f (x + y)) :
  ∀ n, f (f n) = f (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l81_8183


namespace NUMINAMATH_CALUDE_seventh_person_weight_l81_8126

def elevator_problem (initial_people : ℕ) (initial_avg_weight : ℚ) (new_avg_weight : ℚ) : ℚ :=
  let total_initial_weight := initial_people * initial_avg_weight
  let total_new_weight := (initial_people + 1) * new_avg_weight
  total_new_weight - total_initial_weight

theorem seventh_person_weight :
  elevator_problem 6 160 151 = 97 := by sorry

end NUMINAMATH_CALUDE_seventh_person_weight_l81_8126


namespace NUMINAMATH_CALUDE_wood_wasted_percentage_l81_8109

/-- The percentage of wood wasted when carving a cone from a sphere -/
theorem wood_wasted_percentage (sphere_radius cone_height cone_base_diameter : ℝ) :
  sphere_radius = 9 →
  cone_height = 9 →
  cone_base_diameter = 18 →
  let cone_base_radius := cone_base_diameter / 2
  let cone_volume := (1 / 3) * Real.pi * cone_base_radius^2 * cone_height
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius^3
  let percentage_wasted := (cone_volume / sphere_volume) * 100
  percentage_wasted = 25 := by sorry

end NUMINAMATH_CALUDE_wood_wasted_percentage_l81_8109


namespace NUMINAMATH_CALUDE_y_one_gt_y_two_l81_8177

/-- Two points on a line with negative slope -/
structure PointsOnLine where
  y₁ : ℝ
  y₂ : ℝ
  h₁ : y₁ = -1/2 * (-5)
  h₂ : y₂ = -1/2 * (-2)

/-- Theorem: For two points A(-5, y₁) and B(-2, y₂) on the line y = -1/2x, y₁ > y₂ -/
theorem y_one_gt_y_two (p : PointsOnLine) : p.y₁ > p.y₂ := by
  sorry

end NUMINAMATH_CALUDE_y_one_gt_y_two_l81_8177


namespace NUMINAMATH_CALUDE_total_weekly_sleep_is_123_l81_8118

/-- Represents the type of day (odd or even) -/
inductive DayType
| odd
| even

/-- Calculates the sleep time for a cougar based on the day type -/
def cougarSleep (day : DayType) : ℕ :=
  match day with
  | DayType.odd => 6
  | DayType.even => 4

/-- Calculates the sleep time for a zebra based on the cougar's sleep time -/
def zebraSleep (cougarSleepTime : ℕ) : ℕ :=
  cougarSleepTime + 2

/-- Calculates the sleep time for a lion based on the day type and other animals' sleep times -/
def lionSleep (day : DayType) (cougarSleepTime zebraSleepTime : ℕ) : ℕ :=
  match day with
  | DayType.odd => cougarSleepTime + 1
  | DayType.even => zebraSleepTime - 3

/-- Calculates the total weekly sleep time for all three animals -/
def totalWeeklySleep : ℕ :=
  let oddDays := 4
  let evenDays := 3
  let cougarTotal := oddDays * cougarSleep DayType.odd + evenDays * cougarSleep DayType.even
  let zebraTotal := oddDays * zebraSleep (cougarSleep DayType.odd) + evenDays * zebraSleep (cougarSleep DayType.even)
  let lionTotal := oddDays * lionSleep DayType.odd (cougarSleep DayType.odd) (zebraSleep (cougarSleep DayType.odd)) +
                   evenDays * lionSleep DayType.even (cougarSleep DayType.even) (zebraSleep (cougarSleep DayType.even))
  cougarTotal + zebraTotal + lionTotal

theorem total_weekly_sleep_is_123 : totalWeeklySleep = 123 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_sleep_is_123_l81_8118


namespace NUMINAMATH_CALUDE_probability_shaded_isosceles_triangle_l81_8139

/-- Represents a game board shaped like an isosceles triangle -/
structure GameBoard where
  regions : ℕ
  shaded_regions : ℕ

/-- Calculates the probability of landing in a shaded region -/
def probability_shaded (board : GameBoard) : ℚ :=
  board.shaded_regions / board.regions

theorem probability_shaded_isosceles_triangle :
  ∀ (board : GameBoard),
    board.regions = 7 →
    board.shaded_regions = 3 →
    probability_shaded board = 3 / 7 := by
  sorry

#eval probability_shaded { regions := 7, shaded_regions := 3 }

end NUMINAMATH_CALUDE_probability_shaded_isosceles_triangle_l81_8139


namespace NUMINAMATH_CALUDE_total_miles_run_l81_8198

/-- Given that Sam runs 12 miles and Harvey runs 8 miles more than Sam,
    prove that the total distance run by both friends is 32 miles. -/
theorem total_miles_run (sam_miles harvey_miles total_miles : ℕ) : 
  sam_miles = 12 →
  harvey_miles = sam_miles + 8 →
  total_miles = sam_miles + harvey_miles →
  total_miles = 32 := by
sorry

end NUMINAMATH_CALUDE_total_miles_run_l81_8198


namespace NUMINAMATH_CALUDE_book_distribution_ways_l81_8140

theorem book_distribution_ways (n m : ℕ) (h1 : n = 3) (h2 : m = 2) : 
  n * (n - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_ways_l81_8140


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l81_8112

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l81_8112


namespace NUMINAMATH_CALUDE_lcm_of_primes_l81_8163

theorem lcm_of_primes (p₁ p₂ p₃ p₄ : Nat) (h₁ : p₁ = 97) (h₂ : p₂ = 193) (h₃ : p₃ = 419) (h₄ : p₄ = 673) :
  Nat.lcm p₁ (Nat.lcm p₂ (Nat.lcm p₃ p₄)) = 5280671387 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_primes_l81_8163


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_product_l81_8153

/-- Represents a point on a 2D grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a rectangle on a 2D grid --/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomRight : Point
  bottomLeft : Point

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℕ :=
  (r.topRight.x - r.topLeft.x) * (r.topLeft.y - r.bottomLeft.y)

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℕ :=
  2 * ((r.topRight.x - r.topLeft.x) + (r.topLeft.y - r.bottomLeft.y))

/-- The main theorem to prove --/
theorem rectangle_area_perimeter_product :
  let r := Rectangle.mk
    (Point.mk 1 5) (Point.mk 5 5)
    (Point.mk 5 2) (Point.mk 1 2)
  area r * perimeter r = 168 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_product_l81_8153


namespace NUMINAMATH_CALUDE_train_speed_problem_l81_8107

theorem train_speed_problem (x : ℝ) (V : ℝ) (h1 : x > 0) (h2 : V > 0) :
  (3 * x) / ((x / V) + ((2 * x) / 20)) = 25 →
  V = 50 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l81_8107


namespace NUMINAMATH_CALUDE_cubic_roots_from_quadratic_l81_8178

theorem cubic_roots_from_quadratic (b c : ℝ) :
  let x₁ := b + c
  let x₂ := b - c
  (∀ x, x^2 - 2*b*x + b^2 - c^2 = 0 ↔ x = x₁ ∨ x = x₂) →
  (∀ x, x^2 - 2*b*(b^2 + 3*c^2)*x + (b^2 - c^2)^3 = 0 ↔ x = x₁^3 ∨ x = x₂^3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_from_quadratic_l81_8178


namespace NUMINAMATH_CALUDE_proportion_equality_l81_8174

theorem proportion_equality (a b c d : ℝ) (h : a / b = c / d) :
  (a + c) / c = (b + d) / d := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l81_8174


namespace NUMINAMATH_CALUDE_marge_garden_weeds_l81_8145

def garden_problem (total_seeds planted_seeds non_growing_seeds eaten_fraction 
                    strangled_fraction kept_weeds final_plants : ℕ) : Prop :=
  let grown_plants := planted_seeds - non_growing_seeds
  let eaten_plants := (grown_plants / 3 : ℕ)
  let uneaten_plants := grown_plants - eaten_plants
  let strangled_plants := (uneaten_plants / 3 : ℕ)
  let healthy_plants := uneaten_plants - strangled_plants
  healthy_plants + kept_weeds = final_plants

theorem marge_garden_weeds : 
  ∃ (pulled_weeds : ℕ), 
    garden_problem 23 23 5 (1/3) (1/3) 1 9 ∧ 
    pulled_weeds = 3 := by sorry

end NUMINAMATH_CALUDE_marge_garden_weeds_l81_8145


namespace NUMINAMATH_CALUDE_chess_group_size_l81_8175

/-- The number of games played when n players each play every other player once -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that 14 players result in 91 games when each player plays every other player once -/
theorem chess_group_size :
  ∃ (n : ℕ), n > 0 ∧ gamesPlayed n = 91 ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_chess_group_size_l81_8175


namespace NUMINAMATH_CALUDE_digit_2015_is_zero_l81_8142

/-- A sequence formed by arranging all positive integers in increasing order -/
def integer_sequence : ℕ → ℕ := sorry

/-- The nth digit in the integer sequence -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Theorem: If the 11th digit in the integer sequence is 0, then the 2015th digit is also 0 -/
theorem digit_2015_is_zero (h : nth_digit 11 = 0) : nth_digit 2015 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_2015_is_zero_l81_8142


namespace NUMINAMATH_CALUDE_bernardo_wins_smallest_number_l81_8158

theorem bernardo_wins_smallest_number : ∃ M : ℕ, 
  M ≤ 799 ∧ 
  (∀ k : ℕ, k < M → 
    (2 * k ≤ 800 ∧ 
     2 * k + 70 ≤ 800 ∧ 
     4 * k + 140 ≤ 800 ∧ 
     4 * k + 210 ≤ 800 ∧ 
     8 * k + 420 ≤ 800) → 
    8 * k + 490 ≤ 800) ∧
  2 * M ≤ 800 ∧ 
  2 * M + 70 ≤ 800 ∧ 
  4 * M + 140 ≤ 800 ∧ 
  4 * M + 210 ≤ 800 ∧ 
  8 * M + 420 ≤ 800 ∧ 
  8 * M + 490 > 800 ∧
  M = 37 :=
by sorry

end NUMINAMATH_CALUDE_bernardo_wins_smallest_number_l81_8158


namespace NUMINAMATH_CALUDE_max_angle_MPN_x_coordinate_l81_8181

/-- The x-coordinate of point P when angle MPN is maximum -/
def max_angle_x_coordinate : ℝ := 1

/-- Point M with coordinates (-1, 2) -/
def M : ℝ × ℝ := (-1, 2)

/-- Point N with coordinates (1, 4) -/
def N : ℝ × ℝ := (1, 4)

/-- Point P moves on the positive half of the x-axis -/
def P (x : ℝ) : ℝ × ℝ := (x, 0)

/-- The angle MPN as a function of the x-coordinate of P -/
noncomputable def angle_MPN (x : ℝ) : ℝ := sorry

theorem max_angle_MPN_x_coordinate :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 → angle_MPN y ≤ angle_MPN x) ∧
  x = max_angle_x_coordinate := by sorry

end NUMINAMATH_CALUDE_max_angle_MPN_x_coordinate_l81_8181


namespace NUMINAMATH_CALUDE_a_works_friday_50th_week_l81_8108

/-- Represents the days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the people working night shifts -/
inductive Person
  | A
  | B
  | C
  | D
  | E
  | F

/-- Returns the next day in the week -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

/-- Returns the next person in the rotation -/
def nextPerson (p : Person) : Person :=
  match p with
  | Person.A => Person.B
  | Person.B => Person.C
  | Person.C => Person.D
  | Person.D => Person.E
  | Person.E => Person.F
  | Person.F => Person.A

/-- Returns the person working on a given day number -/
def personOnDay (dayNumber : Nat) : Person :=
  match dayNumber % 6 with
  | 0 => Person.F
  | 1 => Person.A
  | 2 => Person.B
  | 3 => Person.C
  | 4 => Person.D
  | 5 => Person.E
  | _ => Person.A  -- This case should never occur

/-- Returns the day of the week for a given day number -/
def dayOfWeek (dayNumber : Nat) : Day :=
  match dayNumber % 7 with
  | 0 => Day.Saturday
  | 1 => Day.Sunday
  | 2 => Day.Monday
  | 3 => Day.Tuesday
  | 4 => Day.Wednesday
  | 5 => Day.Thursday
  | 6 => Day.Friday
  | _ => Day.Sunday  -- This case should never occur

theorem a_works_friday_50th_week :
  personOnDay (50 * 7 - 2) = Person.A ∧ dayOfWeek (50 * 7 - 2) = Day.Friday :=
by sorry

end NUMINAMATH_CALUDE_a_works_friday_50th_week_l81_8108


namespace NUMINAMATH_CALUDE_nephews_count_l81_8104

/-- The number of nephews Alden and Vihaan have together -/
def total_nephews (alden_past : ℕ) (alden_ratio : ℕ) (vihaan_diff : ℕ) : ℕ :=
  let alden_current := alden_past * alden_ratio
  let vihaan_current := alden_current + vihaan_diff
  alden_current + vihaan_current

/-- Proof that Alden and Vihaan have 600 nephews together -/
theorem nephews_count : total_nephews 80 3 120 = 600 := by
  sorry

end NUMINAMATH_CALUDE_nephews_count_l81_8104


namespace NUMINAMATH_CALUDE_subtraction_result_l81_8117

theorem subtraction_result : (1000000000000 : ℕ) - 777777777777 = 222222222223 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l81_8117


namespace NUMINAMATH_CALUDE_fraction_equivalence_l81_8135

theorem fraction_equivalence : (15 : ℝ) / (4 * 63) = 1.5 / (0.4 * 63) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l81_8135


namespace NUMINAMATH_CALUDE_fourth_power_representation_l81_8100

/-- For any base N ≥ 6, (N-1)^4 in base N can be represented as (N-4)5(N-4)1 -/
theorem fourth_power_representation (N : ℕ) (h : N ≥ 6) :
  ∃ (a b c d : ℕ), (N - 1)^4 = a * N^3 + b * N^2 + c * N + d ∧
                    a = N - 4 ∧
                    b = 5 ∧
                    c = N - 4 ∧
                    d = 1 :=
by sorry

end NUMINAMATH_CALUDE_fourth_power_representation_l81_8100


namespace NUMINAMATH_CALUDE_weight_replacement_l81_8170

theorem weight_replacement (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 6 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l81_8170


namespace NUMINAMATH_CALUDE_pentagon_vertex_c_y_coordinate_l81_8157

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The area of a pentagon -/
def area (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a vertical line of symmetry -/
def has_vertical_symmetry (p : Pentagon) : Prop := sorry

/-- The main theorem -/
theorem pentagon_vertex_c_y_coordinate
  (p : Pentagon)
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 5))
  (h3 : ∃ y, p.C = (2.5, y))
  (h4 : p.D = (5, 5))
  (h5 : p.E = (5, 0))
  (h6 : has_vertical_symmetry p)
  (h7 : area p = 50)
  : p.C.2 = 15 := by sorry


end NUMINAMATH_CALUDE_pentagon_vertex_c_y_coordinate_l81_8157


namespace NUMINAMATH_CALUDE_isosceles_triangle_l81_8193

theorem isosceles_triangle (A B C : ℝ) (h1 : 0 < A ∧ A < π)
    (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π)
    (h5 : 2 * Real.cos B * Real.sin A = Real.sin C) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l81_8193


namespace NUMINAMATH_CALUDE_triangle_intersection_coord_diff_l81_8167

/-- Triangle ABC with vertices A(0,10), B(3,-1), C(9,-1) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Point R on line AC and S on line BC -/
structure IntersectionPoints (T : Triangle) :=
  (R : ℝ × ℝ)
  (S : ℝ × ℝ)

/-- The area of triangle RSC -/
def areaRSC (T : Triangle) (I : IntersectionPoints T) : ℝ := sorry

/-- The positive difference between x and y coordinates of R -/
def coordDiffR (I : IntersectionPoints T) : ℝ :=
  |I.R.1 - I.R.2|

theorem triangle_intersection_coord_diff 
  (T : Triangle) 
  (hA : T.A = (0, 10)) 
  (hB : T.B = (3, -1)) 
  (hC : T.C = (9, -1)) 
  (I : IntersectionPoints T) 
  (hvert : I.R.1 = I.S.1) -- R and S on the same vertical line
  (harea : areaRSC T I = 20) :
  coordDiffR I = 50/9 := by sorry

end NUMINAMATH_CALUDE_triangle_intersection_coord_diff_l81_8167


namespace NUMINAMATH_CALUDE_sin_transform_l81_8132

/-- Given a function f(x) = sin(x - π/3), prove that after stretching the x-coordinates
    to twice their original length and shifting the resulting graph to the left by π/3 units,
    the resulting function is g(x) = sin(1/2x - π/6) -/
theorem sin_transform (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.sin (x - π/3)
  let g : ℝ → ℝ := fun x => Real.sin (x/2 - π/6)
  let h : ℝ → ℝ := fun x => f (x/2 + π/3)
  h = g := by sorry

end NUMINAMATH_CALUDE_sin_transform_l81_8132


namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l81_8151

/-- Represents a vertical shift of a parabola -/
def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := 
  λ x => f x + shift

/-- The original parabola function -/
def original_parabola : ℝ → ℝ := 
  λ x => x^2 - 1

/-- The shifted parabola G -/
def G : ℝ → ℝ := vertical_shift original_parabola 3

/-- Theorem stating that G is equivalent to x^2 + 2 -/
theorem shifted_parabola_equation : G = λ x => x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l81_8151


namespace NUMINAMATH_CALUDE_ducks_in_marsh_l81_8186

theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) :
  total_birds - geese = 37 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_marsh_l81_8186


namespace NUMINAMATH_CALUDE_cooking_and_yoga_count_l81_8102

/-- Represents the number of people in various curriculum groups -/
structure CurriculumGroups where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- Theorem stating the number of people studying both cooking and yoga -/
theorem cooking_and_yoga_count (g : CurriculumGroups) 
  (h1 : g.yoga = 25)
  (h2 : g.cooking = 18)
  (h3 : g.weaving = 10)
  (h4 : g.cookingOnly = 4)
  (h5 : g.allCurriculums = 4)
  (h6 : g.cookingAndWeaving = 5) :
  g.cooking - g.cookingOnly - g.cookingAndWeaving + g.allCurriculums = 9 :=
by sorry

end NUMINAMATH_CALUDE_cooking_and_yoga_count_l81_8102


namespace NUMINAMATH_CALUDE_sandwich_combinations_l81_8168

def lunch_meat : ℕ := 12
def cheese : ℕ := 8

theorem sandwich_combinations : 
  (lunch_meat.choose 1) * (cheese.choose 2) = 336 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l81_8168


namespace NUMINAMATH_CALUDE_sum_removal_proof_l81_8114

theorem sum_removal_proof : 
  let original_sum := (1 : ℚ) / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18
  let removed_sum := (1 : ℚ) / 12 + 1 / 15
  original_sum - removed_sum = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_removal_proof_l81_8114


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l81_8144

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l81_8144


namespace NUMINAMATH_CALUDE_equation_solution_l81_8176

theorem equation_solution (x : ℝ) :
  (Real.sqrt (2 * x + 7)) / (Real.sqrt (8 * x + 10)) = 2 / Real.sqrt 5 →
  x = -5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l81_8176


namespace NUMINAMATH_CALUDE_total_flowers_ratio_l81_8105

/-- Represents the number of pots in the garden -/
def num_pots : ℕ := 350

/-- Represents the ratio of flowers to total items in a pot -/
def flower_ratio : ℚ := 3 / 5

/-- Represents the number of flowers in a single pot -/
def flowers_per_pot (total_items : ℕ) : ℚ := flower_ratio * total_items

theorem total_flowers_ratio (total_items_per_pot : ℕ) :
  (num_pots : ℚ) * flowers_per_pot total_items_per_pot = 
  flower_ratio * ((num_pots : ℚ) * total_items_per_pot) := by sorry

end NUMINAMATH_CALUDE_total_flowers_ratio_l81_8105


namespace NUMINAMATH_CALUDE_grid_sum_puzzle_l81_8143

theorem grid_sum_puzzle :
  ∃ (a b c d e f g : ℤ),
    a = 3 ∧ b = 0 ∧ c = 5 ∧ d = -2 ∧ e = -9 ∧ f = -5 ∧ g = 1 ∧
    a + (-1) + 2 = 4 ∧
    2 + 1 + b = 3 ∧
    c + (-4) + (-3) = -2 ∧
    b - 5 - 4 = e ∧
    f = d - 3 ∧
    g = d + 3 ∧
    -8 = 4 + 3 - 9 - 2 + f + g :=
by sorry

end NUMINAMATH_CALUDE_grid_sum_puzzle_l81_8143


namespace NUMINAMATH_CALUDE_infinitely_many_primes_dividing_n_squared_plus_n_plus_one_l81_8149

theorem infinitely_many_primes_dividing_n_squared_plus_n_plus_one :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ n^2 + n + 1} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_dividing_n_squared_plus_n_plus_one_l81_8149


namespace NUMINAMATH_CALUDE_smallest_value_when_x_is_9_l81_8130

theorem smallest_value_when_x_is_9 (x : ℝ) (h : x = 9) :
  min (9/x) (min (9/(x+1)) (min (9/(x-2)) (min (9/(6-x)) ((x-2)/9)))) = 9/(x+1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_when_x_is_9_l81_8130


namespace NUMINAMATH_CALUDE_probability_at_2_3_after_5_moves_l81_8188

/-- Represents the probability of a particle reaching a specific point after a number of moves -/
def particle_probability (x y n : ℕ) : ℚ :=
  if x + y = n then
    (n.choose y : ℚ) * (1/2)^n
  else
    0

/-- Theorem stating the probability of reaching (2,3) after 5 moves -/
theorem probability_at_2_3_after_5_moves :
  particle_probability 2 3 5 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_2_3_after_5_moves_l81_8188


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_9240_l81_8185

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem largest_perfect_square_factor_of_9240 :
  ∃ (n : ℕ), is_perfect_square n ∧ 
             is_factor n 9240 ∧ 
             (∀ m : ℕ, is_perfect_square m → is_factor m 9240 → m ≤ n) ∧
             n = 36 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_9240_l81_8185


namespace NUMINAMATH_CALUDE_line_perp_plane_contained_in_plane_implies_planes_perp_l81_8136

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_contained_in_plane_implies_planes_perp
  (a : Line) (M N : Plane) :
  perpendicular a M → contained_in a N → planes_perpendicular M N :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_contained_in_plane_implies_planes_perp_l81_8136


namespace NUMINAMATH_CALUDE_exact_time_solution_l81_8192

def minute_hand_speed : ℝ := 6
def hour_hand_speed : ℝ := 0.5
def hour_hand_start : ℝ := 270

def clock_problem (t : ℝ) : Prop :=
  0 ≤ t ∧ t < 60 ∧
  |minute_hand_speed * (t + 5) - (hour_hand_start + hour_hand_speed * (t - 2))| = 180

theorem exact_time_solution :
  ∃ t : ℝ, clock_problem t ∧ (21 : ℝ) < t ∧ t < 22 :=
sorry

end NUMINAMATH_CALUDE_exact_time_solution_l81_8192


namespace NUMINAMATH_CALUDE_min_area_triangle_l81_8199

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := (1 / a) * Real.log x

theorem min_area_triangle (a : ℝ) (h : a ≠ 0) :
  let P := (0, a)
  let Q := (Real.exp (a^2), a)
  let R := (0, a - 1/a)
  let area := (Real.exp (a^2)) / (2 * |a|)
  ∃ (min_area : ℝ), min_area = Real.exp (1/2) / Real.sqrt 2 ∧
    ∀ a' : ℝ, a' ≠ 0 → area ≥ min_area := by sorry

end NUMINAMATH_CALUDE_min_area_triangle_l81_8199


namespace NUMINAMATH_CALUDE_cats_on_ship_l81_8160

/-- Represents the number of cats on the ship -/
def num_cats : ℕ := 7

/-- Represents the number of humans on the ship -/
def num_humans : ℕ := 14 - num_cats

/-- The total number of heads on the ship -/
def total_heads : ℕ := 14

/-- The total number of legs on the ship -/
def total_legs : ℕ := 41

theorem cats_on_ship :
  (num_cats + num_humans = total_heads) ∧
  (4 * num_cats + 2 * num_humans - 1 = total_legs) :=
sorry

end NUMINAMATH_CALUDE_cats_on_ship_l81_8160


namespace NUMINAMATH_CALUDE_count_numbers_with_three_ones_l81_8196

/-- Recursive function to count numbers without three consecutive 1's -/
def count_without_three_ones (n : ℕ) : ℕ :=
  if n ≤ 3 then
    match n with
    | 1 => 2
    | 2 => 4
    | 3 => 7
    | _ => 0
  else
    count_without_three_ones (n - 1) + count_without_three_ones (n - 2) + count_without_three_ones (n - 3)

/-- Theorem stating the count of 12-digit numbers with three consecutive 1's -/
theorem count_numbers_with_three_ones : 
  (2^12 : ℕ) - count_without_three_ones 12 = 3592 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_three_ones_l81_8196


namespace NUMINAMATH_CALUDE_surviving_trees_count_l81_8171

/-- Calculates the number of surviving trees after two months given initial conditions --/
theorem surviving_trees_count
  (tree_A_plants tree_B_plants tree_C_plants : ℕ)
  (tree_A_seeds_per_plant tree_B_seeds_per_plant tree_C_seeds_per_plant : ℕ)
  (tree_A_plant_rate tree_B_plant_rate tree_C_plant_rate : ℚ)
  (tree_A_first_month_survival_rate tree_B_first_month_survival_rate tree_C_first_month_survival_rate : ℚ)
  (second_month_survival_rate : ℚ)
  (h1 : tree_A_plants = 25)
  (h2 : tree_B_plants = 20)
  (h3 : tree_C_plants = 10)
  (h4 : tree_A_seeds_per_plant = 1)
  (h5 : tree_B_seeds_per_plant = 2)
  (h6 : tree_C_seeds_per_plant = 3)
  (h7 : tree_A_plant_rate = 3/5)
  (h8 : tree_B_plant_rate = 4/5)
  (h9 : tree_C_plant_rate = 1/2)
  (h10 : tree_A_first_month_survival_rate = 3/4)
  (h11 : tree_B_first_month_survival_rate = 9/10)
  (h12 : tree_C_first_month_survival_rate = 7/10)
  (h13 : second_month_survival_rate = 9/10) :
  ⌊(⌊tree_A_plants * tree_A_seeds_per_plant * tree_A_plant_rate * tree_A_first_month_survival_rate⌋ : ℚ) * second_month_survival_rate⌋ +
  ⌊(⌊tree_B_plants * tree_B_seeds_per_plant * tree_B_plant_rate * tree_B_first_month_survival_rate⌋ : ℚ) * second_month_survival_rate⌋ +
  ⌊(⌊tree_C_plants * tree_C_seeds_per_plant * tree_C_plant_rate * tree_C_first_month_survival_rate⌋ : ℚ) * second_month_survival_rate⌋ = 43 := by
  sorry


end NUMINAMATH_CALUDE_surviving_trees_count_l81_8171


namespace NUMINAMATH_CALUDE_melanie_breadcrumbs_count_l81_8197

/-- Represents the number of pieces a bread slice is divided into -/
structure BreadDivision where
  firstHalf : Nat
  secondHalf : Nat

/-- Calculates the total number of pieces for a bread slice -/
def totalPieces (division : BreadDivision) : Nat :=
  division.firstHalf + division.secondHalf

/-- Represents Melanie's bread slicing method -/
def melanieBreadSlicing : List BreadDivision :=
  [{ firstHalf := 3, secondHalf := 4 },  -- First slice
   { firstHalf := 2, secondHalf := 10 }] -- Second slice

/-- Theorem: Melanie's bread slicing method results in 19 total pieces -/
theorem melanie_breadcrumbs_count :
  (melanieBreadSlicing.map totalPieces).sum = 19 := by
  sorry

#eval (melanieBreadSlicing.map totalPieces).sum

end NUMINAMATH_CALUDE_melanie_breadcrumbs_count_l81_8197


namespace NUMINAMATH_CALUDE_pizza_fraction_l81_8159

theorem pizza_fraction (total_slices : ℕ) (whole_slice : ℚ) (shared_slice : ℚ) : 
  total_slices = 16 → whole_slice = 1 → shared_slice = 1/3 → 
  whole_slice / total_slices + (shared_slice / total_slices) = 1/12 := by
sorry

end NUMINAMATH_CALUDE_pizza_fraction_l81_8159


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l81_8129

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x + 1

-- State the theorem
theorem increasing_function_a_range :
  (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) →
  a ∈ Set.Ici (-(3/2)) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l81_8129


namespace NUMINAMATH_CALUDE_optimal_inequality_l81_8122

theorem optimal_inequality (a b c d : ℝ) 
  (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) (hd : d ≥ -1) :
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d) ∧ 
  ∀ k > 3/4, ∃ x y z w : ℝ, x ≥ -1 ∧ y ≥ -1 ∧ z ≥ -1 ∧ w ≥ -1 ∧ 
    x^3 + y^3 + z^3 + w^3 + 1 < k * (x + y + z + w) :=
by sorry

end NUMINAMATH_CALUDE_optimal_inequality_l81_8122


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l81_8110

-- Define the arithmetic sequence
def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmeticSequence a d →
  d > 0 →
  a 1 + a 2 + a 3 = 15 →
  a 1 * a 2 * a 3 = 80 →
  a 11 + a 12 + a 13 = 105 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l81_8110


namespace NUMINAMATH_CALUDE_russia_canada_size_comparison_l81_8116

theorem russia_canada_size_comparison 
  (us canada russia : ℝ) 
  (h1 : canada = 1.5 * us) 
  (h2 : russia = 2 * us) : 
  (russia - canada) / canada = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_russia_canada_size_comparison_l81_8116


namespace NUMINAMATH_CALUDE_least_with_four_prime_factors_l81_8106

/-- A function that returns the number of prime factors (counting multiplicity) of a positive integer -/
def num_prime_factors (n : ℕ+) : ℕ := sorry

/-- The property that both n and n+1 have exactly four prime factors -/
def has_four_prime_factors (n : ℕ+) : Prop :=
  num_prime_factors n = 4 ∧ num_prime_factors (n + 1) = 4

theorem least_with_four_prime_factors :
  ∀ n : ℕ+, n < 1155 → ¬(has_four_prime_factors n) ∧ has_four_prime_factors 1155 := by
  sorry

end NUMINAMATH_CALUDE_least_with_four_prime_factors_l81_8106


namespace NUMINAMATH_CALUDE_car_speed_problem_l81_8123

/-- Proves that the speed in the first hour is 90 km/h given the conditions -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 50 →
  average_speed = 70 →
  (speed_first_hour : ℝ) →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_first_hour = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l81_8123


namespace NUMINAMATH_CALUDE_inequality_range_l81_8173

theorem inequality_range (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_mn : m + n + 3 = m * n) :
  (∀ x : ℝ, (m + n) * x^2 + 2 * x + m * n - 13 ≥ 0) ↔ 
  (∀ x : ℝ, x ≤ -1 ∨ x ≥ 2/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l81_8173


namespace NUMINAMATH_CALUDE_prime_square_minus_one_remainder_l81_8141

theorem prime_square_minus_one_remainder (p : ℕ) (hp : Nat.Prime p) :
  ∃ r ∈ ({0, 3, 8} : Set ℕ), (p^2 - 1) % 12 = r :=
sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_remainder_l81_8141


namespace NUMINAMATH_CALUDE_total_fruits_eaten_l81_8146

theorem total_fruits_eaten (sophie_oranges_per_day : ℕ) (hannah_grapes_per_day : ℕ) (days : ℕ) :
  sophie_oranges_per_day = 20 →
  hannah_grapes_per_day = 40 →
  days = 30 →
  sophie_oranges_per_day * days + hannah_grapes_per_day * days = 1800 :=
by sorry

end NUMINAMATH_CALUDE_total_fruits_eaten_l81_8146


namespace NUMINAMATH_CALUDE_milk_consumption_l81_8138

theorem milk_consumption (total_monitors : ℕ) (monitors_per_group : ℕ) (students_per_group : ℕ)
  (girl_percentage : ℚ) (boy_milk : ℕ) (girl_milk : ℕ) :
  total_monitors = 8 →
  monitors_per_group = 2 →
  students_per_group = 15 →
  girl_percentage = 2/5 →
  boy_milk = 1 →
  girl_milk = 2 →
  (total_monitors / monitors_per_group * students_per_group *
    ((1 - girl_percentage) * boy_milk + girl_percentage * girl_milk) : ℚ) = 84 :=
by sorry

end NUMINAMATH_CALUDE_milk_consumption_l81_8138


namespace NUMINAMATH_CALUDE_nathaniel_best_friends_l81_8187

theorem nathaniel_best_friends (total_tickets : ℕ) (tickets_per_friend : ℕ) (tickets_left : ℕ) :
  total_tickets = 11 →
  tickets_per_friend = 2 →
  tickets_left = 3 →
  (total_tickets - tickets_left) / tickets_per_friend = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_nathaniel_best_friends_l81_8187


namespace NUMINAMATH_CALUDE_smallest_radius_is_one_l81_8166

/-- Triangle ABC with a circle inscribed on side AB -/
structure TriangleWithInscribedCircle where
  /-- Length of side AC -/
  ac : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The circle's center is on side AB -/
  center_on_ab : Bool
  /-- The circle is tangent to sides AC and BC -/
  tangent_to_ac_bc : Bool

/-- The smallest positive integer radius for the given triangle configuration -/
def smallest_integer_radius (t : TriangleWithInscribedCircle) : ℕ :=
  sorry

/-- Theorem stating that the smallest positive integer radius is 1 -/
theorem smallest_radius_is_one :
  ∀ t : TriangleWithInscribedCircle,
    t.ac = 5 ∧ t.bc = 3 ∧ t.center_on_ab ∧ t.tangent_to_ac_bc →
    smallest_integer_radius t = 1 :=
  sorry

end NUMINAMATH_CALUDE_smallest_radius_is_one_l81_8166


namespace NUMINAMATH_CALUDE_logical_equivalence_l81_8150

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ ¬R) → ¬Q) ↔ (Q → (¬P ∨ R)) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l81_8150


namespace NUMINAMATH_CALUDE_max_candies_eaten_l81_8182

/-- Represents the state of the board and the total candies eaten -/
structure BoardState where
  numbers : List Nat
  candies : Nat

/-- The process of combining two numbers on the board -/
def combineNumbers (state : BoardState) : BoardState :=
  match state.numbers with
  | x :: y :: rest => {
      numbers := (x + y) :: rest,
      candies := state.candies + x * y
    }
  | _ => state

/-- Theorem stating the maximum number of candies that can be eaten -/
theorem max_candies_eaten :
  ∃ (final : BoardState),
    (combineNumbers^[48] {numbers := List.replicate 49 1, candies := 0}) = final ∧
    final.candies = 1176 := by
  sorry

#check max_candies_eaten

end NUMINAMATH_CALUDE_max_candies_eaten_l81_8182


namespace NUMINAMATH_CALUDE_intersection_circle_regions_l81_8161

/-- The maximum number of regions in the intersection of n circles -/
def max_regions (n : ℕ) : ℕ :=
  2 * n - 2

/-- Theorem stating the maximum number of regions in the intersection of n circles -/
theorem intersection_circle_regions (n : ℕ) (h : n ≥ 2) :
  max_regions n = 2 * n - 2 := by
  sorry

#check intersection_circle_regions

end NUMINAMATH_CALUDE_intersection_circle_regions_l81_8161


namespace NUMINAMATH_CALUDE_intersection_sum_l81_8180

/-- Given two functions f and g where
    f(x) = -|x - a| + b
    g(x) = |x - c| + d
    If f and g intersect at points (2, 5) and (8, 3), then a + c = 10 -/
theorem intersection_sum (a b c d : ℝ) : 
  (∀ x, -|x - a| + b = |x - c| + d → x = 2 ∨ x = 8) →
  -|2 - a| + b = 5 →
  -|8 - a| + b = 3 →
  |2 - c| + d = 5 →
  |8 - c| + d = 3 →
  a + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l81_8180


namespace NUMINAMATH_CALUDE_red_balls_count_l81_8115

theorem red_balls_count (total : ℕ) (red : ℕ) (h1 : total = 15) 
  (h2 : red ≤ total) 
  (h3 : (red * (red - 1)) / (total * (total - 1)) = 1 / 21) : 
  red = 4 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l81_8115


namespace NUMINAMATH_CALUDE_exist_same_color_parallel_triangle_l81_8147

/-- Represents a color --/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a point in the triangle --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the coloring of vertices --/
def Coloring := Point → Color

/-- Represents the large equilateral triangle --/
structure LargeTriangle where
  side_length : ℝ
  division_count : ℕ
  coloring : Coloring

/-- Checks if three points form a triangle parallel to the original triangle --/
def is_parallel_triangle (p1 p2 p3 : Point) : Prop := sorry

/-- Main theorem --/
theorem exist_same_color_parallel_triangle (T : LargeTriangle) 
  (h1 : T.division_count = 3000) -- 9000000 small triangles means 3000 divisions per side
  (h2 : T.side_length > 0) :
  ∃ (c : Color) (p1 p2 p3 : Point),
    T.coloring p1 = c ∧
    T.coloring p2 = c ∧
    T.coloring p3 = c ∧
    is_parallel_triangle p1 p2 p3 :=
  sorry

end NUMINAMATH_CALUDE_exist_same_color_parallel_triangle_l81_8147


namespace NUMINAMATH_CALUDE_inequality_solution_set_l81_8190

theorem inequality_solution_set (x : ℝ) : 
  (Set.Ioo (-3 : ℝ) 1 : Set ℝ) = {x | (1 - x) * (3 + x) > 0} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l81_8190


namespace NUMINAMATH_CALUDE_polygon_sides_count_l81_8155

theorem polygon_sides_count (n : ℕ) (exterior_angle : ℝ) : 
  (n ≥ 2) →
  (exterior_angle > 0) →
  (exterior_angle < 45) →
  (n * exterior_angle = 360) →
  (n ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l81_8155


namespace NUMINAMATH_CALUDE_john_money_left_l81_8152

/-- The amount of money John has left after shopping --/
def money_left (initial_amount : ℝ) (roast_cost : ℝ) (vegetable_cost : ℝ) (wine_cost : ℝ) (dessert_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  let total_cost := roast_cost + vegetable_cost + wine_cost + dessert_cost
  let discounted_cost := total_cost * (1 - discount_rate)
  initial_amount - discounted_cost

/-- Theorem stating that John has €56.8 left after shopping --/
theorem john_money_left :
  money_left 100 17 11 12 8 0.1 = 56.8 := by
  sorry

end NUMINAMATH_CALUDE_john_money_left_l81_8152


namespace NUMINAMATH_CALUDE_liquid_x_percentage_in_solution_b_l81_8119

/-- Given two solutions A and B, where liquid X makes up 0.8% of solution A,
    and a mixture of 300g of A and 700g of B results in a solution with 1.5% of liquid X,
    prove that liquid X makes up 1.8% of solution B. -/
theorem liquid_x_percentage_in_solution_b : 
  let percent_x_in_a : ℝ := 0.008
  let mass_a : ℝ := 300
  let mass_b : ℝ := 700
  let percent_x_in_mixture : ℝ := 0.015
  let percent_x_in_b : ℝ := (percent_x_in_mixture * (mass_a + mass_b) - percent_x_in_a * mass_a) / mass_b
  percent_x_in_b = 0.018 := by
  sorry

end NUMINAMATH_CALUDE_liquid_x_percentage_in_solution_b_l81_8119


namespace NUMINAMATH_CALUDE_cafeteria_optimal_location_l81_8137

-- Define the offices and their employee counts
structure Office where
  location : ℝ × ℝ
  employees : ℕ

-- Define the triangle formed by the offices
def office_triangle (A B C : Office) : Prop :=
  A.location ≠ B.location ∧ B.location ≠ C.location ∧ C.location ≠ A.location

-- Define the total distance function
def total_distance (cafeteria : ℝ × ℝ) (A B C : Office) : ℝ :=
  A.employees * dist cafeteria A.location +
  B.employees * dist cafeteria B.location +
  C.employees * dist cafeteria C.location

-- State the theorem
theorem cafeteria_optimal_location (A B C : Office) 
  (h_triangle : office_triangle A B C)
  (h_employees : A.employees = 10 ∧ B.employees = 20 ∧ C.employees = 30) :
  ∀ cafeteria : ℝ × ℝ, total_distance C.location A B C ≤ total_distance cafeteria A B C :=
sorry

end NUMINAMATH_CALUDE_cafeteria_optimal_location_l81_8137


namespace NUMINAMATH_CALUDE_number_wall_top_l81_8165

/-- Represents a number wall with 5 base numbers -/
structure NumberWall (a b c d e : ℕ) where
  level1 : Fin 5 → ℕ
  level2 : Fin 4 → ℕ
  level3 : Fin 3 → ℕ
  level4 : Fin 2 → ℕ
  top : ℕ
  base_correct : level1 = ![a, b, c, d, e]
  level2_correct : ∀ i : Fin 4, level2 i = level1 i + level1 (i.succ)
  level3_correct : ∀ i : Fin 3, level3 i = level2 i + level2 (i.succ)
  level4_correct : ∀ i : Fin 2, level4 i = level3 i + level3 (i.succ)
  top_correct : top = level4 0 + level4 1

/-- The theorem stating that the top of the number wall is x + 103 -/
theorem number_wall_top (x : ℕ) : 
  ∀ (w : NumberWall x 4 8 7 11), w.top = x + 103 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_top_l81_8165


namespace NUMINAMATH_CALUDE_explosion_hyperbola_eccentricity_l81_8191

/-- The eccentricity of a hyperbola formed by an explosion point, given two sentry posts
    1400m apart, a time difference of 3s in hearing the explosion, and a speed of sound of 340m/s -/
theorem explosion_hyperbola_eccentricity
  (distance_between_posts : ℝ)
  (time_difference : ℝ)
  (speed_of_sound : ℝ)
  (h_distance : distance_between_posts = 1400)
  (h_time : time_difference = 3)
  (h_speed : speed_of_sound = 340) :
  let c : ℝ := distance_between_posts / 2
  let a : ℝ := time_difference * speed_of_sound / 2
  c / a = 70 / 51 :=
by sorry

end NUMINAMATH_CALUDE_explosion_hyperbola_eccentricity_l81_8191
