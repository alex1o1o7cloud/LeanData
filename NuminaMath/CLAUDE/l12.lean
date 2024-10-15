import Mathlib

namespace NUMINAMATH_CALUDE_max_value_P_l12_1268

theorem max_value_P (a b x₁ x₂ x₃ : ℝ) 
  (h1 : a = x₁ + x₂ + x₃)
  (h2 : a = x₁ * x₂ * x₃)
  (h3 : a * b = x₁ * x₂ + x₂ * x₃ + x₃ * x₁)
  (h4 : x₁ > 0)
  (h5 : x₂ > 0)
  (h6 : x₃ > 0) :
  let P := (a^2 + 6*b + 1) / (a^2 + a)
  ∃ (max_P : ℝ), ∀ (P_val : ℝ), P ≤ P_val → max_P ≥ P_val ∧ max_P = (9 + Real.sqrt 3) / 9 := by
  sorry


end NUMINAMATH_CALUDE_max_value_P_l12_1268


namespace NUMINAMATH_CALUDE_symmetric_line_passes_through_fixed_point_l12_1274

/-- A line in 2D space represented by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The point of symmetry -/
def symmetryPoint : ℝ × ℝ := (2, 1)

/-- Line l₁ defined as y = k(x-4) -/
def l₁ (k : ℝ) : Line :=
  { slope := k, point := (4, 0) }

/-- Line l₂ symmetric to l₁ with respect to the symmetry point -/
def l₂ (k : ℝ) : Line :=
  sorry -- definition omitted as it's not directly given in the problem

theorem symmetric_line_passes_through_fixed_point (k : ℝ) :
  (0, 2) ∈ {p : ℝ × ℝ | p.2 = (l₂ k).slope * (p.1 - (l₂ k).point.1) + (l₂ k).point.2} :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_passes_through_fixed_point_l12_1274


namespace NUMINAMATH_CALUDE_c_share_is_45_l12_1211

/-- Represents the rent share calculation for a pasture -/
structure PastureRent where
  total_rent : ℕ
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ

/-- Calculates C's share of the rent -/
def calculate_c_share (p : PastureRent) : ℕ :=
  let total_ox_months := p.a_oxen * p.a_months + p.b_oxen * p.b_months + p.c_oxen * p.c_months
  let c_ox_months := p.c_oxen * p.c_months
  (c_ox_months * p.total_rent) / total_ox_months

/-- Theorem stating that C's share of the rent is 45 -/
theorem c_share_is_45 (p : PastureRent) 
  (h1 : p.total_rent = 175)
  (h2 : p.a_oxen = 10) (h3 : p.a_months = 7)
  (h4 : p.b_oxen = 12) (h5 : p.b_months = 5)
  (h6 : p.c_oxen = 15) (h7 : p.c_months = 3) :
  calculate_c_share p = 45 := by
  sorry

end NUMINAMATH_CALUDE_c_share_is_45_l12_1211


namespace NUMINAMATH_CALUDE_equation_system_equivalence_l12_1233

theorem equation_system_equivalence (x y : ℝ) :
  (3 * x^2 + 9 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 4 = 0) →
  4 * y^2 + 19 * y - 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_equivalence_l12_1233


namespace NUMINAMATH_CALUDE_power_of_three_mod_seven_l12_1227

theorem power_of_three_mod_seven : 3^20 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_seven_l12_1227


namespace NUMINAMATH_CALUDE_lamp_post_break_height_l12_1269

/-- Given a 6-meter tall lamp post that breaks and hits the ground 2 meters away from its base,
    the breaking point is √10 meters above the ground. -/
theorem lamp_post_break_height :
  ∀ (x : ℝ),
  x > 0 →
  x < 6 →
  x * x + 2 * 2 = (6 - x) * (6 - x) →
  x = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_lamp_post_break_height_l12_1269


namespace NUMINAMATH_CALUDE_park_benches_l12_1271

theorem park_benches (bench_capacity : ℕ) (people_sitting : ℕ) (spaces_available : ℕ) : 
  bench_capacity = 4 →
  people_sitting = 80 →
  spaces_available = 120 →
  (people_sitting + spaces_available) / bench_capacity = 50 := by
  sorry

end NUMINAMATH_CALUDE_park_benches_l12_1271


namespace NUMINAMATH_CALUDE_marias_green_beans_l12_1282

/-- Given Maria's vegetable cutting preferences and the number of potatoes,
    calculate the number of green beans she needs to cut. -/
theorem marias_green_beans (potatoes : ℕ) : potatoes = 2 → 8 = (potatoes * 6 * 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_marias_green_beans_l12_1282


namespace NUMINAMATH_CALUDE_max_time_digit_sum_l12_1203

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours ≤ 23
  minutes_valid : minutes ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum possible sum of digits for any Time24 -/
def maxTimeDigitSum : Nat := 24

theorem max_time_digit_sum :
  ∀ t : Time24, timeDigitSum t ≤ maxTimeDigitSum :=
by sorry

end NUMINAMATH_CALUDE_max_time_digit_sum_l12_1203


namespace NUMINAMATH_CALUDE_renovation_cost_calculation_l12_1292

def renovation_cost (hourly_rates : List ℝ) (hours_per_day : ℝ) (days : ℕ) 
  (meal_cost : ℝ) (material_cost : ℝ) (unexpected_costs : List ℝ) : ℝ :=
  let daily_labor_cost := hourly_rates.sum * hours_per_day
  let total_labor_cost := daily_labor_cost * days
  let total_meal_cost := meal_cost * hourly_rates.length * days
  let total_unexpected_cost := unexpected_costs.sum
  total_labor_cost + total_meal_cost + material_cost + total_unexpected_cost

theorem renovation_cost_calculation : 
  renovation_cost [15, 20, 18, 22] 8 10 10 2500 [750, 500, 400] = 10550 := by
  sorry

end NUMINAMATH_CALUDE_renovation_cost_calculation_l12_1292


namespace NUMINAMATH_CALUDE_trig_identities_l12_1276

/-- Proof of trigonometric identities -/
theorem trig_identities :
  (Real.cos (π / 3) + Real.sin (π / 4) - Real.tan (π / 4) = (-1 + Real.sqrt 2) / 2) ∧
  (6 * (Real.tan (π / 6))^2 - Real.sqrt 3 * Real.sin (π / 3) - 2 * Real.cos (π / 4) = 1 / 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_trig_identities_l12_1276


namespace NUMINAMATH_CALUDE_no_natural_solution_l12_1264

theorem no_natural_solution :
  ¬∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 18 * m^2 := by
sorry

end NUMINAMATH_CALUDE_no_natural_solution_l12_1264


namespace NUMINAMATH_CALUDE_boat_travel_theorem_l12_1215

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (boat_speed stream_speed : ℝ) : ℝ := boat_speed - stream_speed

/-- Proves that a boat traveling 11 km along the stream in one hour will travel 7 km against the stream in one hour, given its still water speed is 9 km/hr -/
theorem boat_travel_theorem (boat_speed : ℝ) (h1 : boat_speed = 9) 
  (h2 : boat_speed + (11 - boat_speed) = 11) : 
  boat_distance boat_speed (11 - boat_speed) = 7 := by
  sorry

#check boat_travel_theorem

end NUMINAMATH_CALUDE_boat_travel_theorem_l12_1215


namespace NUMINAMATH_CALUDE_hyperbola_foci_l12_1223

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

/-- The foci of the hyperbola -/
def foci : Set (ℝ × ℝ) :=
  {(-5, 0), (5, 0)}

/-- Theorem: The given foci are the correct foci of the hyperbola -/
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y →
  ∃ (f : ℝ × ℝ), f ∈ foci ∧
  (x - f.1)^2 + y^2 = (x + f.1)^2 + y^2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l12_1223


namespace NUMINAMATH_CALUDE_tetrahedron_triangles_l12_1275

/-- The number of vertices in a regular tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a regular tetrahedron -/
def distinct_triangles : ℕ := Nat.choose tetrahedron_vertices triangle_vertices

theorem tetrahedron_triangles : distinct_triangles = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_triangles_l12_1275


namespace NUMINAMATH_CALUDE_lisa_candy_consumption_l12_1224

/-- The number of candies Lisa eats on other days of the week -/
def candies_on_other_days (total_candies : ℕ) (candies_on_mon_wed : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℚ :=
  let total_days := days_per_week * num_weeks
  let mon_wed_days := 2 * num_weeks
  let other_days := total_days - mon_wed_days
  let candies_on_mon_wed_total := candies_on_mon_wed * mon_wed_days
  let remaining_candies := total_candies - candies_on_mon_wed_total
  (remaining_candies : ℚ) / other_days

theorem lisa_candy_consumption :
  candies_on_other_days 36 2 7 4 = 1 := by sorry

end NUMINAMATH_CALUDE_lisa_candy_consumption_l12_1224


namespace NUMINAMATH_CALUDE_product_expansion_sum_l12_1281

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 14 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l12_1281


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_8_l12_1221

theorem no_solution_implies_a_leq_8 (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_8_l12_1221


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l12_1265

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote x - 2y = 0 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b/a = 1/2) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l12_1265


namespace NUMINAMATH_CALUDE_fourth_root_of_sum_of_cubes_l12_1222

theorem fourth_root_of_sum_of_cubes : ∃ n : ℕ, n > 0 ∧ n^4 = 5508^3 + 5625^3 + 5742^3 ∧ n = 855 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_sum_of_cubes_l12_1222


namespace NUMINAMATH_CALUDE_sum_of_roots_special_quadratic_l12_1206

theorem sum_of_roots_special_quadratic :
  let f : ℝ → ℝ := λ x ↦ (x - 7)^2 - 16
  ∃ r₁ r₂ : ℝ, (f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂) ∧ r₁ + r₂ = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_special_quadratic_l12_1206


namespace NUMINAMATH_CALUDE_wife_departure_time_l12_1214

/-- Proves that given the conditions of the problem, the wife left 24 minutes after the man -/
theorem wife_departure_time (man_speed wife_speed : ℝ) (meeting_time : ℝ) :
  man_speed = 40 →
  wife_speed = 50 →
  meeting_time = 2 →
  ∃ (t : ℝ), t * wife_speed = meeting_time * man_speed ∧ t = 24 / 60 :=
by sorry

end NUMINAMATH_CALUDE_wife_departure_time_l12_1214


namespace NUMINAMATH_CALUDE_counterexample_exists_l12_1287

theorem counterexample_exists : ∃ (a b c : ℝ), a > b ∧ a * c ≤ b * c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l12_1287


namespace NUMINAMATH_CALUDE_identity_function_unique_l12_1244

theorem identity_function_unique (f : ℤ → ℤ) 
  (h1 : ∀ x : ℤ, f (f x) = x)
  (h2 : ∀ x y : ℤ, Odd (x + y) → f x + f y ≥ x + y) :
  ∀ x : ℤ, f x = x := by sorry

end NUMINAMATH_CALUDE_identity_function_unique_l12_1244


namespace NUMINAMATH_CALUDE_f_has_minimum_at_negative_four_l12_1247

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 8*x + 2

-- Theorem stating that f has a minimum at x = -4
theorem f_has_minimum_at_negative_four :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ x₀ = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_f_has_minimum_at_negative_four_l12_1247


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l12_1288

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) + 1
  f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l12_1288


namespace NUMINAMATH_CALUDE_otimes_inequality_implies_a_unrestricted_l12_1255

/-- Custom operation ⊗ defined on real numbers -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating that if (x-a) ⊗ (x+a) < 1 holds for all real x, then a can be any real number -/
theorem otimes_inequality_implies_a_unrestricted :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → a ∈ Set.univ :=
by sorry

end NUMINAMATH_CALUDE_otimes_inequality_implies_a_unrestricted_l12_1255


namespace NUMINAMATH_CALUDE_largest_lower_bound_area_l12_1285

/-- A point in the Euclidean plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The convex hull of a set of points -/
def convex_hull (points : Set Point) : Set Point := sorry

/-- The area of a set of points -/
def area (s : Set Point) : ℝ := sorry

/-- A convex set of points -/
def is_convex (s : Set Point) : Prop := sorry

theorem largest_lower_bound_area (points : Set Point) :
  ∀ (s : Set Point), (is_convex s ∧ points ⊆ s) →
    area (convex_hull points) ≤ area s :=
by sorry

end NUMINAMATH_CALUDE_largest_lower_bound_area_l12_1285


namespace NUMINAMATH_CALUDE_new_individuals_weight_l12_1258

/-- The total weight of three new individuals joining a group, given specific conditions -/
theorem new_individuals_weight (W : ℝ) : 
  let initial_group_size : ℕ := 10
  let leaving_weights : List ℝ := [75, 80, 90]
  let average_weight_increase : ℝ := 6.5
  let new_individuals_count : ℕ := 3
  W - (initial_group_size : ℝ) * average_weight_increase = 
    (W - leaving_weights.sum) + (new_individuals_count : ℝ) * average_weight_increase →
  (∃ X : ℝ, X = (new_individuals_count : ℝ) * average_weight_increase ∧ X = 65) := by
sorry


end NUMINAMATH_CALUDE_new_individuals_weight_l12_1258


namespace NUMINAMATH_CALUDE_dictionary_correct_and_complete_l12_1204

-- Define the types for words and sentences
def Word : Type := String
def Sentence : Type := List Word

-- Define the type for a dictionary
def Dictionary : Type := List (Word × Word)

-- Define the Russian sentences
def russian_sentences : List Sentence := [
  ["Мышка", "ночью", "пошла", "гулять"],
  ["Кошка", "ночью", "видит", "мышка"],
  ["Мышку", "кошка", "пошла", "поймать"]
]

-- Define the Am-Yam sentences
def amyam_sentences : List Sentence := [
  ["ту", "ам", "ям", "му"],
  ["ля", "ам", "бу", "ту"],
  ["ту", "ля", "ям", "ям"]
]

-- Define the correct dictionary fragment
def correct_dictionary : Dictionary := [
  ("гулять", "му"),
  ("видит", "бу"),
  ("поймать", "ям"),
  ("мышка", "ту"),
  ("ночью", "ам"),
  ("пошла", "ям"),
  ("кошка", "ля")
]

-- Function to create dictionary from sentence pairs
def create_dictionary (russian : List Sentence) (amyam : List Sentence) : Dictionary :=
  sorry

-- Theorem statement
theorem dictionary_correct_and_complete 
  (russian : List Sentence := russian_sentences)
  (amyam : List Sentence := amyam_sentences)
  (correct : Dictionary := correct_dictionary) :
  create_dictionary russian amyam = correct :=
sorry

end NUMINAMATH_CALUDE_dictionary_correct_and_complete_l12_1204


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l12_1257

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.im (2 / (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l12_1257


namespace NUMINAMATH_CALUDE_solve_nested_equation_l12_1250

theorem solve_nested_equation : 
  ∃ x : ℤ, 45 - (28 - (x - (15 - 16))) = 55 ∧ x = 37 :=
by sorry

end NUMINAMATH_CALUDE_solve_nested_equation_l12_1250


namespace NUMINAMATH_CALUDE_unanswered_questions_count_l12_1277

/-- Represents the scoring system for a math test. -/
structure ScoringSystem where
  correct : Int
  wrong : Int
  unanswered : Int
  initial : Int

/-- Represents the result of a math test. -/
structure TestResult where
  correct : Nat
  wrong : Nat
  unanswered : Nat
  total_score : Int

/-- Calculates the score based on a given scoring system and test result. -/
def calculate_score (system : ScoringSystem) (result : TestResult) : Int :=
  system.initial +
  system.correct * result.correct +
  system.wrong * result.wrong +
  system.unanswered * result.unanswered

theorem unanswered_questions_count
  (new_system : ScoringSystem)
  (old_system : ScoringSystem)
  (result : TestResult)
  (h1 : new_system = { correct := 6, wrong := 0, unanswered := 3, initial := 0 })
  (h2 : old_system = { correct := 4, wrong := -1, unanswered := 0, initial := 40 })
  (h3 : result.correct + result.wrong + result.unanswered = 35)
  (h4 : calculate_score new_system result = 120)
  (h5 : calculate_score old_system result = 100) :
  result.unanswered = 5 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_questions_count_l12_1277


namespace NUMINAMATH_CALUDE_min_wings_theorem_l12_1286

/-- Represents the number and cost of birds John bought -/
structure BirdPurchase where
  parrots : Nat
  pigeons : Nat
  canaries : Nat
  total_cost : Nat

/-- Calculates the total number of wings for a given bird purchase -/
def total_wings (purchase : BirdPurchase) : Nat :=
  2 * (purchase.parrots + purchase.pigeons + purchase.canaries)

/-- Checks if the purchase satisfies all conditions -/
def is_valid_purchase (purchase : BirdPurchase) : Prop :=
  purchase.parrots ≥ 1 ∧
  purchase.pigeons ≥ 1 ∧
  purchase.canaries ≥ 1 ∧
  purchase.total_cost = 200 ∧
  purchase.total_cost = 30 * purchase.parrots + 20 * purchase.pigeons + 15 * purchase.canaries

theorem min_wings_theorem :
  ∃ (purchase : BirdPurchase),
    is_valid_purchase purchase ∧
    (∀ (other : BirdPurchase), is_valid_purchase other → total_wings purchase ≤ total_wings other) ∧
    total_wings purchase = 24 := by
  sorry

end NUMINAMATH_CALUDE_min_wings_theorem_l12_1286


namespace NUMINAMATH_CALUDE_paper_strip_dimensions_l12_1246

theorem paper_strip_dimensions 
  (a b c : ℕ) 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) :
  a = 1 ∧ b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_paper_strip_dimensions_l12_1246


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_solution_l12_1297

theorem cubic_equation_one_real_solution :
  ∀ (a : ℝ), ∃! (x : ℝ), x^3 - a*x^2 - 3*a*x + a^2 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_solution_l12_1297


namespace NUMINAMATH_CALUDE_distribute_balls_to_bags_correct_l12_1208

/-- The number of ways to distribute n identical balls into m numbered bags, such that no bag is empty -/
def distribute_balls_to_bags (n m : ℕ) : ℕ :=
  Nat.choose (n - 1) (m - 1)

/-- Theorem: The number of ways to distribute n identical balls into m numbered bags, 
    such that no bag is empty, is equal to (n-1) choose (m-1) -/
theorem distribute_balls_to_bags_correct (n m : ℕ) (h1 : n > m) (h2 : m > 0) : 
  distribute_balls_to_bags n m = Nat.choose (n - 1) (m - 1) := by
  sorry

#check distribute_balls_to_bags_correct

end NUMINAMATH_CALUDE_distribute_balls_to_bags_correct_l12_1208


namespace NUMINAMATH_CALUDE_base5_44_to_decimal_l12_1278

/-- Converts a base-5 number to its decimal equivalent -/
def base5ToDecimal (d₁ d₀ : ℕ) : ℕ := d₁ * 5^1 + d₀ * 5^0

/-- The base-5 number 44₅ -/
def base5_44 : ℕ × ℕ := (4, 4)

theorem base5_44_to_decimal :
  base5ToDecimal base5_44.1 base5_44.2 = 24 := by sorry

end NUMINAMATH_CALUDE_base5_44_to_decimal_l12_1278


namespace NUMINAMATH_CALUDE_football_shape_area_l12_1253

/-- The area of the football-shaped region formed by two circular sectors -/
theorem football_shape_area 
  (r1 : ℝ) 
  (r2 : ℝ) 
  (h1 : r1 = 2 * Real.sqrt 2) 
  (h2 : r2 = 2) 
  (θ : ℝ) 
  (h3 : θ = π / 2) : 
  (θ / (2 * π)) * π * r1^2 - (θ / (2 * π)) * π * r2^2 = π := by
  sorry

end NUMINAMATH_CALUDE_football_shape_area_l12_1253


namespace NUMINAMATH_CALUDE_system_solution_l12_1216

theorem system_solution (a b c : ℝ) 
  (eq1 : b + c = 10 - 4*a)
  (eq2 : a + c = -16 - 4*b)
  (eq3 : a + b = 9 - 4*c) :
  2*a + 2*b + 2*c = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l12_1216


namespace NUMINAMATH_CALUDE_x_value_proof_l12_1218

theorem x_value_proof (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l12_1218


namespace NUMINAMATH_CALUDE_valentines_theorem_l12_1213

theorem valentines_theorem (n x y : ℕ+) (h : x * y = x + y + 36) : x * y = 76 := by
  sorry

end NUMINAMATH_CALUDE_valentines_theorem_l12_1213


namespace NUMINAMATH_CALUDE_yellow_tiles_count_l12_1299

theorem yellow_tiles_count (total : ℕ) (purple : ℕ) (white : ℕ) 
  (h1 : total = 20)
  (h2 : purple = 6)
  (h3 : white = 7)
  : ∃ (yellow : ℕ), 
    yellow + (yellow + 1) + purple + white = total ∧ 
    yellow = 3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_tiles_count_l12_1299


namespace NUMINAMATH_CALUDE_tailor_buttons_total_l12_1252

theorem tailor_buttons_total (green yellow blue total : ℕ) : 
  green = 90 →
  yellow = green + 10 →
  blue = green - 5 →
  total = green + yellow + blue →
  total = 275 := by
sorry

end NUMINAMATH_CALUDE_tailor_buttons_total_l12_1252


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l12_1263

theorem tangent_line_to_circle (r : ℝ) (hr : r > 0) : 
  (∀ x y : ℝ, 2*x + y = r → x^2 + y^2 = 2*r → 
   ∃ x₀ y₀ : ℝ, 2*x₀ + y₀ = r ∧ x₀^2 + y₀^2 = 2*r ∧ 
   ∀ x₁ y₁ : ℝ, 2*x₁ + y₁ = r → x₁^2 + y₁^2 ≥ 2*r) →
  r = 10 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l12_1263


namespace NUMINAMATH_CALUDE_running_speed_is_six_l12_1260

/-- Calculates the running speed given swimming speed and average speed -/
def calculate_running_speed (swimming_speed average_speed : ℝ) : ℝ :=
  2 * average_speed - swimming_speed

/-- Proves that given a swimming speed of 1 mph and an average speed of 3.5 mph
    for equal time spent swimming and running, the running speed is 6 mph -/
theorem running_speed_is_six :
  let swimming_speed : ℝ := 1
  let average_speed : ℝ := 3.5
  calculate_running_speed swimming_speed average_speed = 6 := by
  sorry

#eval calculate_running_speed 1 3.5

end NUMINAMATH_CALUDE_running_speed_is_six_l12_1260


namespace NUMINAMATH_CALUDE_prob_draw_queen_l12_1289

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Represents the number of a specific card in the deck -/
def cardsOfType (d : Deck) : Nat := d.suits

/-- A standard deck of cards -/
def standardDeck : Deck :=
  { cards := 52
  , ranks := 13
  , suits := 4 }

/-- The probability of drawing a specific card from the deck -/
def probDraw (d : Deck) : ℚ := (cardsOfType d : ℚ) / (d.cards : ℚ)

theorem prob_draw_queen (d : Deck := standardDeck) :
  probDraw d = 1 / 13 := by
  sorry

#eval probDraw standardDeck

end NUMINAMATH_CALUDE_prob_draw_queen_l12_1289


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l12_1262

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - p - 1 = 0) → 
  (q^3 - q - 1 = 0) → 
  (r^3 - r - 1 = 0) → 
  (1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 11 / 7) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l12_1262


namespace NUMINAMATH_CALUDE_excluded_students_count_l12_1228

theorem excluded_students_count (total_students : ℕ) (initial_avg : ℚ) 
  (excluded_avg : ℚ) (new_avg : ℚ) (h1 : total_students = 10) 
  (h2 : initial_avg = 80) (h3 : excluded_avg = 70) (h4 : new_avg = 90) :
  ∃ (excluded : ℕ), 
    excluded = 5 ∧ 
    (initial_avg * total_students : ℚ) = 
      excluded_avg * excluded + new_avg * (total_students - excluded) :=
by sorry

end NUMINAMATH_CALUDE_excluded_students_count_l12_1228


namespace NUMINAMATH_CALUDE_max_books_borrowed_l12_1259

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℕ) (h1 : total_students = 20) (h2 : zero_books = 3) (h3 : one_book = 9) 
  (h4 : two_books = 4) (h5 : avg_books = 2) : 
  ∃ (max_books : ℕ), max_books = 14 ∧ 
  ∀ (student_books : ℕ), student_books ≤ max_books ∧
  (zero_books * 0 + one_book * 1 + two_books * 2 + 
   (total_students - zero_books - one_book - two_books) * 3 + 
   (max_books - 3) ≤ total_students * avg_books) :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l12_1259


namespace NUMINAMATH_CALUDE_third_derivative_y_l12_1238

open Real

noncomputable def y (x : ℝ) : ℝ := (log (2 * x + 5)) / (2 * x + 5)

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = (88 - 48 * log (2 * x + 5)) / (2 * x + 5)^4 :=
by sorry

end NUMINAMATH_CALUDE_third_derivative_y_l12_1238


namespace NUMINAMATH_CALUDE_nancy_carrots_l12_1210

def initial_carrots : ℕ → ℕ → ℕ → Prop :=
  fun x thrown_out additional =>
    x - thrown_out + additional = 31

theorem nancy_carrots : initial_carrots 12 2 21 := by sorry

end NUMINAMATH_CALUDE_nancy_carrots_l12_1210


namespace NUMINAMATH_CALUDE_banknote_replacement_theorem_l12_1254

/-- Represents the banknote replacement problem in the Magical Kingdom treasury --/
structure BanknoteReplacement where
  total_banknotes : ℕ
  machine_startup_cost : ℕ
  major_repair_cost : ℕ
  post_repair_capacity : ℕ
  budget : ℕ

/-- Calculates the number of banknotes replaced in a given number of days --/
def banknotes_replaced (br : BanknoteReplacement) (days : ℕ) : ℕ :=
  sorry

/-- Checks if all banknotes can be replaced within the budget --/
def can_replace_all (br : BanknoteReplacement) : Prop :=
  sorry

/-- The main theorem about banknote replacement --/
theorem banknote_replacement_theorem (br : BanknoteReplacement) 
  (h1 : br.total_banknotes = 3628800)
  (h2 : br.machine_startup_cost = 90000)
  (h3 : br.major_repair_cost = 700000)
  (h4 : br.post_repair_capacity = 1000000)
  (h5 : br.budget = 1000000) :
  (banknotes_replaced br 3 ≥ br.total_banknotes * 9 / 10) ∧
  (can_replace_all br) :=
sorry

end NUMINAMATH_CALUDE_banknote_replacement_theorem_l12_1254


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l12_1291

/-- Given a hyperbola with equation x²/9 - y² = 1, its asymptotes are y = x/3 and y = -x/3 -/
theorem hyperbola_asymptotes :
  let hyperbola := fun (x y : ℝ) => x^2 / 9 - y^2 = 1
  let asymptote1 := fun (x y : ℝ) => y = x / 3
  let asymptote2 := fun (x y : ℝ) => y = -x / 3
  (∀ x y, hyperbola x y → (asymptote1 x y ∨ asymptote2 x y)) :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l12_1291


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l12_1284

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a partition of a rectangle into four smaller rectangles -/
structure RectanglePartition where
  total : Rectangle
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ

/-- The theorem statement -/
theorem fourth_rectangle_area 
  (partition : RectanglePartition)
  (h1 : partition.total.length = 20)
  (h2 : partition.total.width = 12)
  (h3 : partition.area1 = 24)
  (h4 : partition.area2 = 48)
  (h5 : partition.area3 = 36) :
  partition.total.length * partition.total.width - (partition.area1 + partition.area2 + partition.area3) = 112 :=
by sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l12_1284


namespace NUMINAMATH_CALUDE_car_speed_adjustment_l12_1267

theorem car_speed_adjustment (distance : ℝ) (original_time : ℝ) (time_factor : ℝ) :
  distance = 324 →
  original_time = 6 →
  time_factor = 3 / 2 →
  (distance / (original_time * time_factor)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_adjustment_l12_1267


namespace NUMINAMATH_CALUDE_square_sum_solution_l12_1231

theorem square_sum_solution (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 130) : 
  x^2 + y^2 = 3049 / 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_solution_l12_1231


namespace NUMINAMATH_CALUDE_initial_number_theorem_l12_1217

theorem initial_number_theorem (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_theorem_l12_1217


namespace NUMINAMATH_CALUDE_rectangle_quadrilateral_inequality_l12_1240

theorem rectangle_quadrilateral_inequality 
  (m n a b c d : ℝ) 
  (hm : m > 0) 
  (hn : n > 0) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (h_rectangle : ∃ (x y z s t u v w : ℝ), 
    x + w = m ∧ y + z = n ∧ s + t = n ∧ u + v = m ∧
    a^2 = x^2 + y^2 ∧ b^2 = z^2 + s^2 ∧ c^2 = t^2 + u^2 ∧ d^2 = v^2 + w^2) :
  1 ≤ (a^2 + b^2 + c^2 + d^2) / (m^2 + n^2) ∧ (a^2 + b^2 + c^2 + d^2) / (m^2 + n^2) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_quadrilateral_inequality_l12_1240


namespace NUMINAMATH_CALUDE_beautiful_points_of_A_beautiful_points_coincide_original_point_C_l12_1245

-- Define the type for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the beautiful points of a given point
def beautifulPoints (p : Point2D) : (Point2D × Point2D) :=
  let a := -p.x
  let b := p.x - p.y
  ({x := a, y := b}, {x := b, y := a})

-- Theorem 1: Beautiful points of A(4,1)
theorem beautiful_points_of_A :
  let A : Point2D := {x := 4, y := 1}
  let (M, N) := beautifulPoints A
  M = {x := -4, y := 3} ∧ N = {x := 3, y := -4} := by sorry

-- Theorem 2: When beautiful points of B(2,y) coincide
theorem beautiful_points_coincide :
  ∀ y : ℝ, let B : Point2D := {x := 2, y := y}
  let (M, N) := beautifulPoints B
  M = N → y = 4 := by sorry

-- Theorem 3: Original point C given a beautiful point (-2,7)
theorem original_point_C :
  ∀ C : Point2D, let (M, N) := beautifulPoints C
  (M = {x := -2, y := 7} ∨ N = {x := -2, y := 7}) →
  (C = {x := 2, y := -5} ∨ C = {x := -7, y := -5}) := by sorry

end NUMINAMATH_CALUDE_beautiful_points_of_A_beautiful_points_coincide_original_point_C_l12_1245


namespace NUMINAMATH_CALUDE_spruce_tree_height_l12_1232

theorem spruce_tree_height 
  (height_maple : ℝ) 
  (height_pine : ℝ) 
  (height_spruce : ℝ) 
  (h1 : height_maple = height_pine + 1)
  (h2 : height_pine = height_spruce - 4)
  (h3 : height_maple / height_spruce = 25 / 64) :
  height_spruce = 64 / 13 := by
  sorry

end NUMINAMATH_CALUDE_spruce_tree_height_l12_1232


namespace NUMINAMATH_CALUDE_swimming_scenario_l12_1272

/-- The number of weeks in the swimming scenario -/
def weeks : ℕ := 4

/-- Camden's total number of swims -/
def camden_total : ℕ := 16

/-- Susannah's total number of swims -/
def susannah_total : ℕ := 24

/-- Camden's swims per week -/
def camden_per_week : ℚ := camden_total / weeks

/-- Susannah's swims per week -/
def susannah_per_week : ℚ := susannah_total / weeks

theorem swimming_scenario :
  (susannah_per_week = camden_per_week + 2) ∧
  (camden_per_week * weeks = camden_total) ∧
  (susannah_per_week * weeks = susannah_total) :=
by sorry

end NUMINAMATH_CALUDE_swimming_scenario_l12_1272


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l12_1212

/-- The line equation mx+(1-m)y+m-2=0 always passes through the point (1,2) for all real m. -/
theorem fixed_point_on_line (m : ℝ) : m * 1 + (1 - m) * 2 + m - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l12_1212


namespace NUMINAMATH_CALUDE_square_root_of_25_l12_1295

theorem square_root_of_25 : Real.sqrt 25 = 5 ∨ Real.sqrt 25 = -5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_25_l12_1295


namespace NUMINAMATH_CALUDE_inequality_proof_l12_1205

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 1) : 
  x * y + y * z + z * x ≤ 2 / 7 + 9 * x * y * z / 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l12_1205


namespace NUMINAMATH_CALUDE_discounted_cd_cost_l12_1241

/-- The cost of five CDs with a 10% discount, given the cost of two CDs and the discount condition -/
theorem discounted_cd_cost (cost_of_two : ℝ) (discount_rate : ℝ) : 
  cost_of_two = 40 →
  discount_rate = 0.1 →
  (5 : ℝ) * (cost_of_two / 2) * (1 - discount_rate) = 90 := by
sorry

end NUMINAMATH_CALUDE_discounted_cd_cost_l12_1241


namespace NUMINAMATH_CALUDE_distance_ratio_cars_l12_1283

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (car : Car) : ℝ := car.speed * car.time

/-- Theorem: The ratio of distances covered by Car A and Car B is 2:1 -/
theorem distance_ratio_cars (carA carB : Car)
  (hA_speed : carA.speed = 80)
  (hA_time : carA.time = 5)
  (hB_speed : carB.speed = 100)
  (hB_time : carB.time = 2) :
  distance carA / distance carB = 2 := by
  sorry

#check distance_ratio_cars

end NUMINAMATH_CALUDE_distance_ratio_cars_l12_1283


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l12_1280

theorem cos_alpha_plus_pi_fourth (α : Real) :
  (∃ (x y : Real), x = 4 ∧ y = -3 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos (α + π/4) = 7 * Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l12_1280


namespace NUMINAMATH_CALUDE_num_routes_eq_factorial_power_l12_1296

/-- Represents the number of southern cities -/
def num_southern_cities : ℕ := 4

/-- Represents the number of northern cities -/
def num_northern_cities : ℕ := 5

/-- Represents the number of transfers between southern cities -/
def num_transfers : ℕ := num_southern_cities

/-- Calculates the number of different routes for the traveler -/
def num_routes : ℕ := (Nat.factorial (num_southern_cities - 1)) * (num_northern_cities ^ num_transfers)

/-- Theorem stating that the number of routes is equal to 3! × 5^4 -/
theorem num_routes_eq_factorial_power : num_routes = 3750 := by sorry

end NUMINAMATH_CALUDE_num_routes_eq_factorial_power_l12_1296


namespace NUMINAMATH_CALUDE_ellipse_sum_l12_1237

/-- Represents an ellipse with center (h, k) and semi-axes lengths a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

theorem ellipse_sum (e : Ellipse) :
  e.h = 3 ∧ e.k = -5 ∧ e.a = 7 ∧ e.b = 4 →
  e.h + e.k + e.a + e.b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l12_1237


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l12_1242

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 35)
  (h2 : x^2 * y + x * y^2 = 306) : 
  x^2 + y^2 = 290 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l12_1242


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l12_1236

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + a * c) = 34)
  (edge_length : 4 * (a + b + c) = 40) :
  ∃ d : ℝ, d^2 = 66 ∧ d^2 = a^2 + b^2 + c^2 := by sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l12_1236


namespace NUMINAMATH_CALUDE_integers_less_than_four_abs_value_l12_1207

theorem integers_less_than_four_abs_value :
  {x : ℤ | |x| < 4} = {-3, -2, -1, 0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_integers_less_than_four_abs_value_l12_1207


namespace NUMINAMATH_CALUDE_sum_of_digits_of_2012_power_l12_1273

def A : ℕ := 2012^2012

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def B : ℕ := sum_of_digits A
def C : ℕ := sum_of_digits B
def D : ℕ := sum_of_digits C

theorem sum_of_digits_of_2012_power : D = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_2012_power_l12_1273


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l12_1266

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x > 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ -1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l12_1266


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l12_1243

theorem min_value_of_expression (x y : ℝ) : (x * y + 1)^2 + (x - y)^2 ≥ 1 := by
  sorry

theorem lower_bound_achievable : ∃ x y : ℝ, (x * y + 1)^2 + (x - y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l12_1243


namespace NUMINAMATH_CALUDE_count_hyperbola_integer_tangent_points_l12_1298

/-- The number of points on the hyperbola y = 2013/x where the tangent line
    intersects both coordinate axes at integer points -/
def hyperbola_integer_tangent_points : ℕ := 48

/-- The hyperbola equation y = 2013/x -/
def hyperbola (x y : ℝ) : Prop := y = 2013 / x

/-- Predicate for a point (x, y) on the hyperbola having a tangent line
    that intersects both axes at integer coordinates -/
def has_integer_intercepts (x y : ℝ) : Prop :=
  hyperbola x y ∧
  ∃ (x_int y_int : ℤ),
    (x_int ≠ 0 ∧ y_int ≠ 0) ∧
    (y - 2013 / x = -(2013 / x^2) * (x_int - x)) ∧
    (0 = -(2013 / x^2) * x_int + 2 * 2013 / x)

theorem count_hyperbola_integer_tangent_points :
  (∑' p : {p : ℝ × ℝ // has_integer_intercepts p.1 p.2}, 1) =
    hyperbola_integer_tangent_points :=
sorry

end NUMINAMATH_CALUDE_count_hyperbola_integer_tangent_points_l12_1298


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l12_1235

theorem largest_prime_factor_of_1729 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l12_1235


namespace NUMINAMATH_CALUDE_apple_tree_production_l12_1220

/-- Apple tree production over three years -/
theorem apple_tree_production : 
  let first_year : ℕ := 40
  let second_year : ℕ := 8 + 2 * first_year
  let third_year : ℕ := second_year - second_year / 4
  first_year + second_year + third_year = 194 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_production_l12_1220


namespace NUMINAMATH_CALUDE_square_diagonal_perimeter_l12_1229

theorem square_diagonal_perimeter (d : ℝ) (s : ℝ) (P : ℝ) :
  d = 2 * Real.sqrt 2 →  -- diagonal length
  d = s * Real.sqrt 2 →  -- relation between diagonal and side length
  P = 4 * s →           -- perimeter definition
  P = 8 := by
sorry

end NUMINAMATH_CALUDE_square_diagonal_perimeter_l12_1229


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l12_1251

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n

theorem minimum_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  GeometricSequence a →
  a 3 = a 2 + 2 * a 1 →
  a m * a n = 64 * (a 1)^2 →
  (∀ k l : ℕ, a k * a l = 64 * (a 1)^2 → 1 / k + 9 / l ≥ 1 / m + 9 / n) →
  1 / m + 9 / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l12_1251


namespace NUMINAMATH_CALUDE_larry_channels_l12_1294

/-- Calculates the final number of channels for Larry given the initial count and subsequent changes. -/
def final_channels (initial : ℕ) (removed : ℕ) (replaced : ℕ) (reduced : ℕ) (sports : ℕ) (supreme : ℕ) : ℕ :=
  initial - removed + replaced - reduced + sports + supreme

/-- Theorem stating that Larry's final channel count is 147 given the specific changes. -/
theorem larry_channels : 
  final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end NUMINAMATH_CALUDE_larry_channels_l12_1294


namespace NUMINAMATH_CALUDE_xiao_ming_envelopes_l12_1293

def red_envelopes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def total_sum : ℕ := red_envelopes.sum

def each_person_sum : ℕ := total_sum / 3

def father_envelopes : List ℕ := [1, 3]
def mother_envelopes : List ℕ := [8, 9]

theorem xiao_ming_envelopes :
  ∀ (xm : List ℕ),
    xm.length = 4 →
    father_envelopes.length = 4 →
    mother_envelopes.length = 4 →
    xm.sum = each_person_sum →
    father_envelopes.sum = each_person_sum →
    mother_envelopes.sum = each_person_sum →
    (∀ x ∈ xm, x ∈ red_envelopes) →
    (∀ x ∈ father_envelopes, x ∈ red_envelopes) →
    (∀ x ∈ mother_envelopes, x ∈ red_envelopes) →
    (∀ x ∈ red_envelopes, x ∈ xm ∨ x ∈ father_envelopes ∨ x ∈ mother_envelopes) →
    6 ∈ xm ∧ 11 ∈ xm :=
by
  sorry

#check xiao_ming_envelopes

end NUMINAMATH_CALUDE_xiao_ming_envelopes_l12_1293


namespace NUMINAMATH_CALUDE_greatest_difference_of_unit_digits_l12_1261

def is_multiple_of_four (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k

def three_digit_72X (n : ℕ) : Prop := ∃ x : ℕ, n = 720 + x ∧ x < 10

def possible_unit_digit (x : ℕ) : Prop :=
  ∃ n : ℕ, three_digit_72X n ∧ is_multiple_of_four n ∧ n % 10 = x

theorem greatest_difference_of_unit_digits :
  (∃ x y : ℕ, possible_unit_digit x ∧ possible_unit_digit y ∧ x - y = 8) ∧
  (∀ a b : ℕ, possible_unit_digit a → possible_unit_digit b → a - b ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_of_unit_digits_l12_1261


namespace NUMINAMATH_CALUDE_line_L_equation_trajectory_Q_equation_l12_1201

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line L
def LineL (x y : ℝ) : Prop := x = 1 ∨ 3*x - 4*y + 5 = 0

-- Define the trajectory of Q
def TrajectoryQ (x y : ℝ) : Prop := x^2/4 + y^2/16 = 1

-- Theorem for Part I
theorem line_L_equation : 
  ∃ (A B : ℝ × ℝ), 
  Circle A.1 A.2 ∧ Circle B.1 B.2 ∧
  LineL A.1 A.2 ∧ LineL B.1 B.2 ∧
  LineL 1 2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
sorry

-- Theorem for Part II
theorem trajectory_Q_equation :
  ∀ (M : ℝ × ℝ), Circle M.1 M.2 →
  ∃ (Q : ℝ × ℝ), 
  Q.1 = M.1 ∧ Q.2 = 2 * M.2 ∧
  TrajectoryQ Q.1 Q.2 :=
sorry

end NUMINAMATH_CALUDE_line_L_equation_trajectory_Q_equation_l12_1201


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l12_1239

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 11))) →
  11 ∣ x →
  x = 59048 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l12_1239


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l12_1219

/-- The perimeter of a regular hexagon with side length 5 cm is 30 cm. -/
theorem regular_hexagon_perimeter :
  ∀ (side_length : ℝ), side_length = 5 →
  (6 : ℝ) * side_length = 30 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l12_1219


namespace NUMINAMATH_CALUDE_f_properties_g_property_l12_1209

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) + 6 * Real.cos (ω * x / 2)^2 - 3

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem f_properties (ω : ℝ) (θ : ℝ) (h_ω : ω > 0) (h_θ : 0 < θ ∧ θ < Real.pi / 2) :
  (is_even (fun x ↦ f ω (x + θ)) ∧ 
   has_period (fun x ↦ f ω (x + θ)) Real.pi ∧
   ∀ p, has_period (fun x ↦ f ω (x + θ)) p → p ≥ Real.pi) →
  ω = 2 ∧ θ = Real.pi / 12 :=
sorry

theorem g_property (ω : ℝ) (h_ω : ω > 0) :
  is_increasing_on (fun x ↦ f ω (3 * x)) 0 (Real.pi / 3) →
  ω ≤ 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_f_properties_g_property_l12_1209


namespace NUMINAMATH_CALUDE_nine_candidates_l12_1200

/- Define the number of ways to select president and vice president -/
def selection_ways : ℕ := 72

/- Define the property that determines the number of candidates -/
def candidate_count (n : ℕ) : Prop :=
  n * (n - 1) = selection_ways

/- Theorem statement -/
theorem nine_candidates : 
  ∃ (n : ℕ), candidate_count n ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_nine_candidates_l12_1200


namespace NUMINAMATH_CALUDE_integers_between_cubes_l12_1256

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(10.3 : ℝ)^3⌋ - ⌈(10.2 : ℝ)^3⌉ + 1) ∧ n = 155 := by
  sorry

end NUMINAMATH_CALUDE_integers_between_cubes_l12_1256


namespace NUMINAMATH_CALUDE_computer_cost_computer_cost_proof_l12_1226

theorem computer_cost (accessories_cost : ℕ) (playstation_worth : ℕ) (discount_percent : ℕ) (out_of_pocket : ℕ) : ℕ :=
  let playstation_sold := playstation_worth - (playstation_worth * discount_percent / 100)
  let total_paid := playstation_sold + out_of_pocket
  total_paid - accessories_cost

#check computer_cost 200 400 20 580 = 700

theorem computer_cost_proof :
  computer_cost 200 400 20 580 = 700 := by
  sorry

end NUMINAMATH_CALUDE_computer_cost_computer_cost_proof_l12_1226


namespace NUMINAMATH_CALUDE_number_added_l12_1270

theorem number_added (x y : ℝ) (h1 : x = 55) (h2 : (x / 5) + y = 21) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_added_l12_1270


namespace NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l12_1248

/-- The total number of boxes in the game -/
def total_boxes : ℕ := 26

/-- The number of boxes containing at least $200,000 -/
def high_value_boxes : ℕ := 9

/-- The probability threshold for holding a high-value box -/
def probability_threshold : ℚ := 1/2

/-- The minimum number of boxes that need to be eliminated -/
def boxes_to_eliminate : ℕ := 9

theorem deal_or_no_deal_elimination :
  boxes_to_eliminate = total_boxes - high_value_boxes - (total_boxes - high_value_boxes) / 2 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l12_1248


namespace NUMINAMATH_CALUDE_wood_sawing_time_l12_1279

/-- Time to saw wood into segments -/
def saw_time (segments : ℕ) (time : ℕ) : Prop :=
  segments > 1 ∧ time = (segments - 1) * (12 / 3)

theorem wood_sawing_time :
  saw_time 4 12 →
  saw_time 8 28 ∧ ¬saw_time 8 24 := by
sorry

end NUMINAMATH_CALUDE_wood_sawing_time_l12_1279


namespace NUMINAMATH_CALUDE_minimum_team_size_l12_1225

theorem minimum_team_size : ∃ n : ℕ, n > 0 ∧ n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ ∀ m : ℕ, m > 0 → m % 8 = 0 → m % 9 = 0 → m % 10 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_minimum_team_size_l12_1225


namespace NUMINAMATH_CALUDE_acorn_theorem_l12_1202

/-- The number of acorns Shawna, Sheila, and Danny have altogether -/
theorem acorn_theorem (shawna sheila danny : ℕ) : 
  shawna = 7 →
  sheila = 5 * shawna →
  danny = sheila + 3 →
  shawna + sheila + danny = 80 := by
  sorry

end NUMINAMATH_CALUDE_acorn_theorem_l12_1202


namespace NUMINAMATH_CALUDE_chess_tournament_draw_fraction_l12_1249

theorem chess_tournament_draw_fraction 
  (peter_wins : Rat) 
  (marc_wins : Rat) 
  (h1 : peter_wins = 2 / 5)
  (h2 : marc_wins = 1 / 4)
  : 1 - (peter_wins + marc_wins) = 7 / 20 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_draw_fraction_l12_1249


namespace NUMINAMATH_CALUDE_joshua_total_cars_l12_1234

/-- The total number of toy cars Joshua has -/
def total_cars (box1 box2 box3 : ℕ) : ℕ := box1 + box2 + box3

/-- Theorem: Joshua has 71 toy cars in total -/
theorem joshua_total_cars :
  total_cars 21 31 19 = 71 := by
  sorry

end NUMINAMATH_CALUDE_joshua_total_cars_l12_1234


namespace NUMINAMATH_CALUDE_chicken_duck_difference_l12_1290

theorem chicken_duck_difference (total_birds ducks : ℕ) 
  (h1 : total_birds = 95) 
  (h2 : ducks = 32) : 
  total_birds - ducks - ducks = 31 := by
  sorry

end NUMINAMATH_CALUDE_chicken_duck_difference_l12_1290


namespace NUMINAMATH_CALUDE_ellipse_and_intersection_properties_l12_1230

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Definition of the intersection line -/
def intersection_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 2

/-- Theorem stating the properties of the ellipse and the range of k -/
theorem ellipse_and_intersection_properties :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), ellipse_C x y ∧ x = 1 ∧ y = 3/2) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    intersection_line k x₁ y₁ ∧ intersection_line k x₂ y₂ ∧
    x₁ ≠ x₂) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    intersection_line k x₁ y₁ ∧ intersection_line k x₂ y₂ ∧
    x₁ ≠ x₂ →
    (1/3 * x₁) * (2/3 * x₂) + (1/3 * y₁) * (2/3 * y₂) < 
    ((1/3 * x₁)^2 + (1/3 * y₁)^2 + (2/3 * x₂)^2 + (2/3 * y₂)^2) / 2) →
  (k > 1/2 ∧ k < 2 * Real.sqrt 3 / 3) ∨ (k < -1/2 ∧ k > -2 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_intersection_properties_l12_1230
