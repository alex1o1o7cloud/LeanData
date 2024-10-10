import Mathlib

namespace necessary_condition_for_inequality_l1692_169248

theorem necessary_condition_for_inequality (a b : ℝ) :
  a * Real.sqrt a + b * Real.sqrt b > a * Real.sqrt b + b * Real.sqrt a →
  a ≥ 0 ∧ b ≥ 0 ∧ a ≠ b :=
by sorry

end necessary_condition_for_inequality_l1692_169248


namespace three_more_than_twice_x_l1692_169240

/-- The algebraic expression for a number that is 3 more than twice x is 2x + 3. -/
theorem three_more_than_twice_x (x : ℝ) : 2 * x + 3 = 2 * x + 3 := by
  sorry

end three_more_than_twice_x_l1692_169240


namespace odd_function_sum_l1692_169238

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_sum (a b c : ℝ) (f : ℝ → ℝ) :
  IsOdd f →
  (∀ x, f x = x^2 * Real.sin x + c - 3) →
  (∀ x, x ∈ Set.Icc (a + 2) b → f x ≠ 0) →
  b > a + 2 →
  a + b + c = 1 := by sorry

end odd_function_sum_l1692_169238


namespace negative_half_times_negative_two_l1692_169291

theorem negative_half_times_negative_two : (-1/2 : ℚ) * (-2 : ℚ) = 1 := by
  sorry

end negative_half_times_negative_two_l1692_169291


namespace three_digit_permutations_l1692_169256

/-- The set of digits used in the problem -/
def digits : Finset Nat := {1, 2, 3}

/-- The number of digits used -/
def n : Nat := Finset.card digits

/-- The length of each permutation -/
def k : Nat := 3

/-- The number of permutations of the digits -/
def num_permutations : Nat := Nat.factorial n

theorem three_digit_permutations : num_permutations = 6 := by
  sorry

end three_digit_permutations_l1692_169256


namespace book_cost_price_l1692_169220

theorem book_cost_price (final_price : ℝ) (profit_rate : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) 
  (h1 : final_price = 250)
  (h2 : profit_rate = 0.20)
  (h3 : tax_rate = 0.10)
  (h4 : discount_rate = 0.05) : 
  ∃ (cost_price : ℝ), cost_price = final_price / ((1 + profit_rate) * (1 - discount_rate) * (1 + tax_rate)) :=
by sorry

end book_cost_price_l1692_169220


namespace right_triangle_vector_relation_l1692_169277

/-- Given a right triangle ABC with ∠C = 90°, vector AB = (t, 1), and vector AC = (2, 3), prove that t = 5 -/
theorem right_triangle_vector_relation (t : ℝ) : 
  let A : ℝ × ℝ := (0, 0)  -- Assuming A is at the origin for simplicity
  let B : ℝ × ℝ := (t, 1)
  let C : ℝ × ℝ := (2, 3)
  let AB : ℝ × ℝ := (t - 0, 1 - 0)  -- Vector from A to B
  let AC : ℝ × ℝ := (2 - 0, 3 - 0)  -- Vector from A to C
  let BC : ℝ × ℝ := (2 - t, 3 - 1)  -- Vector from B to C
  (AC.1 * BC.1 + AC.2 * BC.2 = 0) →  -- Dot product of AC and BC is 0 (perpendicular)
  t = 5 := by
sorry


end right_triangle_vector_relation_l1692_169277


namespace larger_number_is_ten_l1692_169264

theorem larger_number_is_ten (x y : ℕ) (h1 : x * y = 30) (h2 : x + y = 13) : 
  max x y = 10 := by
sorry

end larger_number_is_ten_l1692_169264


namespace water_added_to_container_l1692_169262

theorem water_added_to_container (capacity initial_percentage final_fraction : ℝ) 
  (h1 : capacity = 80)
  (h2 : initial_percentage = 0.30)
  (h3 : final_fraction = 3/4) : 
  capacity * (final_fraction - initial_percentage) = 36 := by
sorry

end water_added_to_container_l1692_169262


namespace platform_walk_probability_l1692_169273

/-- The number of platforms at the train station -/
def num_platforms : ℕ := 16

/-- The distance between adjacent platforms in feet -/
def platform_distance : ℕ := 200

/-- The maximum walking distance we're interested in -/
def max_walk_distance : ℕ := 800

/-- The probability of walking 800 feet or less between two randomly assigned platforms -/
theorem platform_walk_probability : 
  let total_assignments := num_platforms * (num_platforms - 1)
  let favorable_assignments := 
    (2 * 4 * 8) +  -- Edge platforms (1-4 and 13-16) have 8 choices each
    (8 * 10)       -- Central platforms (5-12) have 10 choices each
  (favorable_assignments : ℚ) / total_assignments = 3 / 5 := by
  sorry

end platform_walk_probability_l1692_169273


namespace empty_solution_set_range_l1692_169200

theorem empty_solution_set_range (k : ℝ) : 
  (∀ x : ℝ, ¬(k * x^2 + 2 * k * x + 2 < 0)) ↔ (0 ≤ k ∧ k ≤ 2) := by
  sorry

end empty_solution_set_range_l1692_169200


namespace triangle_side_difference_l1692_169255

theorem triangle_side_difference (x : ℕ) : 
  (x > 0) →
  (x + 10 > 8) →
  (x + 8 > 10) →
  (10 + 8 > x) →
  (∃ (max min : ℕ), 
    (∀ y : ℕ, (y > 0 ∧ y + 10 > 8 ∧ y + 8 > 10 ∧ 10 + 8 > y) → y ≤ max ∧ y ≥ min) ∧
    (max - min = 14)) :=
by sorry

end triangle_side_difference_l1692_169255


namespace lisa_income_percentage_l1692_169251

-- Define variables for incomes
variable (T : ℝ) -- Tim's income
variable (M : ℝ) -- Mary's income
variable (J : ℝ) -- Juan's income
variable (L : ℝ) -- Lisa's income

-- Define the conditions
variable (h1 : M = 1.60 * T) -- Mary's income is 60% more than Tim's
variable (h2 : T = 0.50 * J) -- Tim's income is 50% less than Juan's
variable (h3 : L = 1.30 * M) -- Lisa's income is 30% more than Mary's
variable (h4 : L = 0.75 * J) -- Lisa's income is 25% less than Juan's

-- Define the theorem
theorem lisa_income_percentage :
  (L / (M + J)) * 100 = 41.67 :=
sorry

end lisa_income_percentage_l1692_169251


namespace problem_statement_l1692_169241

theorem problem_statement (x y z t : ℝ) 
  (eq1 : 3 * x^2 + 3 * x * z + z^2 = 1)
  (eq2 : 3 * y^2 + 3 * y * z + z^2 = 4)
  (eq3 : x^2 - x * y + y^2 = t) :
  t ≤ 10 := by
  sorry

end problem_statement_l1692_169241


namespace angelina_walking_speed_l1692_169247

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (distance_home_grocery : ℝ) 
  (distance_grocery_gym : ℝ) 
  (speed_home_grocery : ℝ) 
  (speed_grocery_gym : ℝ) 
  (time_difference : ℝ) :
  distance_home_grocery = 150 →
  distance_grocery_gym = 200 →
  speed_grocery_gym = 2 * speed_home_grocery →
  distance_home_grocery / speed_home_grocery - 
    distance_grocery_gym / speed_grocery_gym = time_difference →
  time_difference = 10 →
  speed_grocery_gym = 10 := by
sorry


end angelina_walking_speed_l1692_169247


namespace symmetry_implies_difference_l1692_169299

/-- Two points are symmetric with respect to the origin if the sum of their coordinates is (0,0) -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

theorem symmetry_implies_difference (a b : ℝ) :
  symmetric_wrt_origin a (-2) 4 b → a - b = -6 := by
  sorry

end symmetry_implies_difference_l1692_169299


namespace left_handed_jazz_lovers_count_l1692_169280

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

end left_handed_jazz_lovers_count_l1692_169280


namespace parallel_perpendicular_implication_l1692_169242

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (α : Plane) : Prop := sorry

/-- Theorem: If two lines are parallel and one is perpendicular to a plane, 
    then the other is also perpendicular to that plane -/
theorem parallel_perpendicular_implication 
  (m n : Line) (α : Plane) : 
  parallel m n → perpendicular m α → perpendicular n α := by
  sorry

end parallel_perpendicular_implication_l1692_169242


namespace jessie_weight_l1692_169239

/-- Jessie's weight problem -/
theorem jessie_weight (initial_weight lost_weight : ℕ) (h1 : initial_weight = 74) (h2 : lost_weight = 7) :
  initial_weight - lost_weight = 67 := by
  sorry

end jessie_weight_l1692_169239


namespace log_equality_implies_product_l1692_169228

theorem log_equality_implies_product (x y : ℝ) :
  x > 1 →
  y > 1 →
  (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5) →
  x^2 * y^2 = 225^Real.sqrt 2 := by
sorry

end log_equality_implies_product_l1692_169228


namespace crayons_count_l1692_169215

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 41

/-- The number of crayons Sam added to the drawer -/
def added_crayons : ℕ := 12

/-- The total number of crayons after Sam's addition -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem crayons_count : total_crayons = 53 := by
  sorry

end crayons_count_l1692_169215


namespace largest_k_for_inequality_l1692_169204

theorem largest_k_for_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  (∀ k : ℝ, 0 < k → k ≤ 5 → a^3 + b^3 + c^3 - 3 ≥ k * (3 - a*b - b*c - c*a)) ∧
  (∃ a₀ b₀ c₀ : ℝ, 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ + b₀ + c₀ = 3 ∧
    a₀^3 + b₀^3 + c₀^3 - 3 = 5 * (3 - a₀*b₀ - b₀*c₀ - c₀*a₀)) :=
by sorry

end largest_k_for_inequality_l1692_169204


namespace triangle_properties_l1692_169257

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c / (t.b - t.a) = (Real.sin t.A + Real.sin t.B) / (Real.sin t.A + Real.sin t.C))
  (h2 : Real.sin t.C = 2 * Real.sin t.A)
  (h3 : 1/2 * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3)
  (h4 : t.b = Real.sqrt 3)
  (h5 : t.a * t.c = 1) : 
  t.B = 2 * Real.pi / 3 ∧ 
  t.a = 2 ∧ 
  t.c = 4 ∧ 
  t.a + t.b + t.c = 2 + Real.sqrt 3 := by
  sorry


end triangle_properties_l1692_169257


namespace difference_of_roots_quadratic_l1692_169243

theorem difference_of_roots_quadratic (x : ℝ) : 
  let roots := {r : ℝ | r^2 - 9*r + 14 = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 5 :=
by
  sorry

end difference_of_roots_quadratic_l1692_169243


namespace monas_weekly_miles_l1692_169235

/-- Represents the days of the week Mona bikes --/
inductive BikingDay
| Monday
| Wednesday
| Saturday

/-- Represents Mona's biking schedule --/
structure BikingSchedule where
  monday_miles : ℕ
  wednesday_miles : ℕ
  saturday_miles : ℕ

/-- Mona's actual biking schedule --/
def monas_schedule : BikingSchedule :=
  { monday_miles := 6
  , wednesday_miles := 12
  , saturday_miles := 12 }

/-- The total miles Mona bikes in a week --/
def total_miles (schedule : BikingSchedule) : ℕ :=
  schedule.monday_miles + schedule.wednesday_miles + schedule.saturday_miles

/-- Theorem stating that Mona bikes 30 miles each week --/
theorem monas_weekly_miles :
  total_miles monas_schedule = 30 ∧
  monas_schedule.wednesday_miles = 12 ∧
  monas_schedule.saturday_miles = 2 * monas_schedule.monday_miles ∧
  monas_schedule.monday_miles = 6 :=
by
  sorry


end monas_weekly_miles_l1692_169235


namespace square_binomial_equality_l1692_169269

theorem square_binomial_equality (a b : ℝ) : 
  (-1 + a * b^2)^2 = 1 - 2 * a * b^2 + a^2 * b^4 := by
  sorry

end square_binomial_equality_l1692_169269


namespace quilt_block_shaded_fraction_l1692_169226

/-- Represents a quilt block -/
structure QuiltBlock where
  size : ℕ
  totalSquares : ℕ
  dividedSquares : ℕ
  shadedTrianglesPerSquare : ℕ

/-- The fraction of a quilt block that is shaded -/
def shadedFraction (q : QuiltBlock) : ℚ :=
  (q.dividedSquares * q.shadedTrianglesPerSquare : ℚ) / (2 * q.totalSquares : ℚ)

/-- Theorem: The shaded fraction of the specified quilt block is 1/8 -/
theorem quilt_block_shaded_fraction :
  let q : QuiltBlock := ⟨4, 16, 4, 1⟩
  shadedFraction q = 1 / 8 := by
  sorry

end quilt_block_shaded_fraction_l1692_169226


namespace circle_radius_from_distances_l1692_169272

theorem circle_radius_from_distances (max_distance min_distance : ℝ) 
  (h1 : max_distance = 11)
  (h2 : min_distance = 5) :
  ∃ (r : ℝ), (r = 3 ∨ r = 8) ∧ 
  ((max_distance - min_distance = 2 * r) ∨ (max_distance + min_distance = 2 * r)) := by
sorry

end circle_radius_from_distances_l1692_169272


namespace product_inequality_l1692_169205

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end product_inequality_l1692_169205


namespace difference_of_sums_l1692_169274

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_5 (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def sum_rounded_to_5 (n : ℕ) : ℕ :=
  (n / 5) * (0 + 5 + 5 + 5 + 10)

theorem difference_of_sums (n : ℕ) (h : n = 200) : 
  (sum_to_n n) - (sum_rounded_to_5 n) = 19100 := by
  sorry

#check difference_of_sums

end difference_of_sums_l1692_169274


namespace midpoint_after_translation_l1692_169283

/-- Given points A, J, and H in a 2D coordinate system, and a translation vector,
    prove that the midpoint of A'H' after translation is as specified. -/
theorem midpoint_after_translation (A J H : ℝ × ℝ) (translation : ℝ × ℝ) :
  A = (3, 3) →
  J = (4, 8) →
  H = (7, 3) →
  translation = (-6, 3) →
  let A' := (A.1 + translation.1, A.2 + translation.2)
  let H' := (H.1 + translation.1, H.2 + translation.2)
  ((A'.1 + H'.1) / 2, (A'.2 + H'.2) / 2) = (-1, 6) := by
  sorry

end midpoint_after_translation_l1692_169283


namespace fourth_root_equation_solutions_l1692_169276

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (((43 - 2*x) ^ (1/4) : ℝ) + ((37 + 2*x) ^ (1/4) : ℝ) = 4) ↔ (x = -19 ∨ x = 21) :=
by sorry

end fourth_root_equation_solutions_l1692_169276


namespace dormitory_allocation_l1692_169210

/-- The number of ways to assign n students to two dormitories with at least k students in each -/
def allocation_schemes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to arrange r items from n items -/
def arrange (n : ℕ) (r : ℕ) : ℕ := sorry

theorem dormitory_allocation :
  allocation_schemes 7 2 = 112 :=
by
  sorry

#check dormitory_allocation

end dormitory_allocation_l1692_169210


namespace alligator_count_l1692_169254

theorem alligator_count (crocodiles vipers total : ℕ) 
  (h1 : crocodiles = 22)
  (h2 : vipers = 5)
  (h3 : total = 50)
  (h4 : ∃ alligators : ℕ, crocodiles + alligators + vipers = total) :
  ∃ alligators : ℕ, alligators = 23 ∧ crocodiles + alligators + vipers = total :=
by sorry

end alligator_count_l1692_169254


namespace largest_centrally_symmetric_polygon_area_l1692_169232

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A polygon in a 2D plane --/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Checks if a polygon is centrally symmetric --/
def isCentrallySymmetric (p : Polygon) : Prop := sorry

/-- Checks if a polygon is inside a triangle --/
def isInsideTriangle (p : Polygon) (t : Triangle) : Prop := sorry

/-- Calculates the area of a polygon --/
def area (p : Polygon) : ℝ := sorry

/-- Calculates the area of a triangle --/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The theorem to be proved --/
theorem largest_centrally_symmetric_polygon_area (t : Triangle) : 
  ∃ (p : Polygon), 
    isCentrallySymmetric p ∧ 
    isInsideTriangle p t ∧ 
    (∀ (q : Polygon), isCentrallySymmetric q → isInsideTriangle q t → area q ≤ area p) ∧
    area p = (2/3) * triangleArea t := by
  sorry

end largest_centrally_symmetric_polygon_area_l1692_169232


namespace sufficient_not_necessary_condition_l1692_169252

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 1 → log a 3 < log b 3) ∧
  (∃ a b, log a 3 < log b 3 ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end sufficient_not_necessary_condition_l1692_169252


namespace school_population_theorem_l1692_169295

theorem school_population_theorem (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 400 →
  boys + girls = total →
  girls = (boys * 100) / total →
  boys = 320 := by
sorry

end school_population_theorem_l1692_169295


namespace beach_house_pool_problem_l1692_169234

theorem beach_house_pool_problem (total_people : ℕ) (legs_in_pool : ℕ) (legs_per_person : ℕ) :
  total_people = 26 →
  legs_in_pool = 34 →
  legs_per_person = 2 →
  total_people - (legs_in_pool / legs_per_person) = 9 := by
sorry

end beach_house_pool_problem_l1692_169234


namespace xy_value_l1692_169271

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = -6 := by
  sorry

end xy_value_l1692_169271


namespace gcd_of_powers_of_79_l1692_169212

theorem gcd_of_powers_of_79 : 
  Nat.Prime 79 → Nat.gcd (79^7 + 1) (79^7 + 79^2 + 1) = 1 := by
  sorry

end gcd_of_powers_of_79_l1692_169212


namespace sugar_amount_in_new_recipe_l1692_169267

/-- Represents a ratio of three ingredients -/
structure Ratio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original ratio of ingredients -/
def original_ratio : Ratio :=
  { flour := 10, water := 6, sugar := 3 }

/-- The new ratio after adjusting flour to water and flour to sugar -/
def new_ratio : Ratio :=
  { flour := 20, water := 6, sugar := 12 }

/-- The amount of water in the new recipe -/
def new_water_amount : ℚ := 2

theorem sugar_amount_in_new_recipe :
  let sugar_amount := (new_ratio.sugar / new_ratio.water) * new_water_amount
  sugar_amount = 4 := by sorry

end sugar_amount_in_new_recipe_l1692_169267


namespace repeating_decimal_fraction_equality_l1692_169293

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + d.repeatingPart / (99 : ℚ)

/-- The main theorem stating that the given fraction of repeating decimals equals the specified rational number -/
theorem repeating_decimal_fraction_equality : 
  let a := RepeatingDecimal.mk 0 75
  let b := RepeatingDecimal.mk 2 25
  (toRational a) / (toRational b) = 2475 / 7339 := by
  sorry

end repeating_decimal_fraction_equality_l1692_169293


namespace gym_attendance_proof_l1692_169207

theorem gym_attendance_proof (W A S : ℕ) : 
  (W + A + S) + 8 = 30 → W + A + S = 22 := by
  sorry

end gym_attendance_proof_l1692_169207


namespace almond_walnut_ratio_l1692_169231

/-- Given a mixture of almonds and walnuts, prove the ratio of almonds to walnuts -/
theorem almond_walnut_ratio 
  (total_weight : ℝ) 
  (almond_weight : ℝ) 
  (almond_parts : ℕ) 
  (h1 : total_weight = 280) 
  (h2 : almond_weight = 200) 
  (h3 : almond_parts = 5) :
  ∃ (walnut_parts : ℕ), 
    (almond_parts : ℝ) / walnut_parts = 5 / 2 :=
by sorry

end almond_walnut_ratio_l1692_169231


namespace total_books_l1692_169266

-- Define the number of books for each person
def betty_books (x : ℚ) : ℚ := x

def sister_books (x : ℚ) : ℚ := x + (1/4) * x

def cousin_books (x : ℚ) : ℚ := 2 * (sister_books x)

def friend_books (x y : ℚ) : ℚ := 
  betty_books x + sister_books x + cousin_books x - y

-- Theorem statement
theorem total_books (x y : ℚ) : 
  betty_books x + sister_books x + cousin_books x + friend_books x y = (19/2) * x - y := by
  sorry

end total_books_l1692_169266


namespace weight_of_barium_iodide_l1692_169260

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of moles of Barium iodide -/
def moles_BaI2 : ℝ := 4

/-- The molecular weight of Barium iodide (BaI2) in g/mol -/
def molecular_weight_BaI2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_I

/-- The total weight of Barium iodide in grams -/
def total_weight_BaI2 : ℝ := moles_BaI2 * molecular_weight_BaI2

theorem weight_of_barium_iodide :
  total_weight_BaI2 = 1564.52 := by sorry

end weight_of_barium_iodide_l1692_169260


namespace vitamin_a_weekly_pills_l1692_169209

/-- The number of pills needed to meet the weekly recommended amount of Vitamin A -/
def pills_needed (vitamin_per_pill : ℕ) (daily_recommended : ℕ) (days_per_week : ℕ) : ℕ :=
  (daily_recommended * days_per_week) / vitamin_per_pill

/-- Proof that 28 pills are needed per week to meet the recommended Vitamin A intake -/
theorem vitamin_a_weekly_pills : pills_needed 50 200 7 = 28 := by
  sorry

end vitamin_a_weekly_pills_l1692_169209


namespace problem_1_problem_2_l1692_169294

theorem problem_1 : 2 * Real.cos (45 * π / 180) + (π - Real.sqrt 3) ^ 0 - Real.sqrt 8 = 1 - Real.sqrt 2 := by
  sorry

theorem problem_2 (m : ℝ) (h : m ≠ 1) : 
  ((2 / (m - 1) + 1) / ((2 * m + 2) / (m^2 - 2 * m + 1))) = (m - 1) / 2 := by
  sorry

end problem_1_problem_2_l1692_169294


namespace sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l1692_169297

theorem sum_of_solutions_quadratic (a b c : ℚ) (h : a ≠ 0) :
  let eq := fun x => a * x^2 + b * x + c
  (∀ x, eq x = 0 → x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) ∨ 
                    x = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)) →
  (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) + (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a) = -b / a :=
by sorry

theorem sum_of_solutions_specific_quadratic :
  let eq := fun x : ℚ => -48 * x^2 + 100 * x + 200
  (∀ x, eq x = 0 → x = (-100 + Real.sqrt (100^2 - 4*(-48)*200)) / (2*(-48)) ∨ 
                    x = (-100 - Real.sqrt (100^2 - 4*(-48)*200)) / (2*(-48))) →
  (-100 + Real.sqrt (100^2 - 4*(-48)*200)) / (2*(-48)) + 
  (-100 - Real.sqrt (100^2 - 4*(-48)*200)) / (2*(-48)) = 25 / 12 :=
by sorry

end sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l1692_169297


namespace car_repair_cost_l1692_169261

theorem car_repair_cost (total_cost : ℝ) (num_parts : ℕ) (labor_rate : ℝ) (work_hours : ℝ)
  (h1 : total_cost = 220)
  (h2 : num_parts = 2)
  (h3 : labor_rate = 0.5)
  (h4 : work_hours = 6) :
  (total_cost - labor_rate * work_hours * 60) / num_parts = 20 := by
  sorry

end car_repair_cost_l1692_169261


namespace induction_step_for_even_numbers_l1692_169245

theorem induction_step_for_even_numbers (k : ℕ) (h_k_even : Even k) (h_k_ge_2 : k ≥ 2) :
  let n := k + 2
  Even n ∧ n > k :=
by sorry

end induction_step_for_even_numbers_l1692_169245


namespace solve_system_of_equations_l1692_169206

theorem solve_system_of_equations (x y : ℝ) : 
  (2 * x - y = 12) → (x = 5) → (y = -2) := by
  sorry

end solve_system_of_equations_l1692_169206


namespace ratio_composition_l1692_169288

theorem ratio_composition (a b c : ℚ) 
  (h1 : a / b = 2 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 2 / 15 := by
  sorry

end ratio_composition_l1692_169288


namespace min_distance_to_line_l1692_169225

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + y = 4}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧ 
    ∀ (P : ℝ × ℝ), P ∈ line → 
      Real.sqrt (P.1^2 + P.2^2) ≥ d ∧ 
      ∃ (Q : ℝ × ℝ), Q ∈ line ∧ Real.sqrt (Q.1^2 + Q.2^2) = d :=
by sorry

end min_distance_to_line_l1692_169225


namespace redo_profit_is_5000_l1692_169229

/-- Calculates the profit for Redo's horseshoe manufacturing --/
def calculate_profit (initial_outlay : ℕ) (cost_per_set : ℕ) (selling_price : ℕ) (num_sets : ℕ) : ℤ :=
  let revenue := num_sets * selling_price
  let manufacturing_costs := initial_outlay + (cost_per_set * num_sets)
  (revenue : ℤ) - manufacturing_costs

/-- Proves that the profit for Redo's horseshoe manufacturing is $5,000 --/
theorem redo_profit_is_5000 :
  calculate_profit 10000 20 50 500 = 5000 := by
  sorry

end redo_profit_is_5000_l1692_169229


namespace perp_to_same_plane_implies_parallel_perp_to_two_planes_implies_parallel_l1692_169214

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define non-coincidence
variable (non_coincident_lines : Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem perp_to_same_plane_implies_parallel 
  (a b : Line) (α : Plane) 
  (h1 : non_coincident_lines a b) 
  (h2 : perp a α) (h3 : perp b α) : 
  parallel a b := by sorry

-- Theorem 2: If a line is perpendicular to two planes, then those planes are parallel
theorem perp_to_two_planes_implies_parallel 
  (a : Line) (α β : Plane) 
  (h1 : non_coincident_planes α β) 
  (h2 : perp a α) (h3 : perp a β) : 
  plane_parallel α β := by sorry

end perp_to_same_plane_implies_parallel_perp_to_two_planes_implies_parallel_l1692_169214


namespace no_duplicates_on_diagonal_l1692_169201

/-- Represents a symmetric table with specific properties -/
structure SymmetricTable :=
  (size : Nat)
  (values : Fin size → Fin size → Fin size)
  (symmetric : ∀ i j, values i j = values j i)
  (distinct_rows : ∀ i j k, j ≠ k → values i j ≠ values i k)

/-- The main theorem stating that there are no duplicate numbers on the diagonal of symmetry -/
theorem no_duplicates_on_diagonal (t : SymmetricTable) (h : t.size = 101) :
  ∀ i j, i ≠ j → t.values i i ≠ t.values j j := by
  sorry

end no_duplicates_on_diagonal_l1692_169201


namespace speed_conversion_l1692_169258

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The speed of the train in meters per second -/
def train_speed_mps : ℝ := 20.0016

/-- The speed of the train in kilometers per hour -/
def train_speed_kmph : ℝ := train_speed_mps * mps_to_kmph

theorem speed_conversion :
  train_speed_kmph = 72.00576 := by
  sorry

end speed_conversion_l1692_169258


namespace exhibition_arrangements_l1692_169203

def total_arrangements (n : ℕ) : ℕ := n.factorial

def adjacent_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

theorem exhibition_arrangements :
  let n := 4
  let total := total_arrangements n
  let adjacent := adjacent_arrangements n
  total - adjacent = 12 := by sorry

end exhibition_arrangements_l1692_169203


namespace one_minus_repeating_third_equals_two_thirds_l1692_169249

def repeating_decimal_one_third : ℚ := 1/3

theorem one_minus_repeating_third_equals_two_thirds :
  1 - repeating_decimal_one_third = 2/3 := by sorry

end one_minus_repeating_third_equals_two_thirds_l1692_169249


namespace fencing_calculation_l1692_169218

/-- The total fencing length for a square playground and a rectangular garden -/
def total_fencing (playground_side : ℝ) (garden_length garden_width : ℝ) : ℝ :=
  4 * playground_side + 2 * (garden_length + garden_width)

/-- Theorem: The total fencing for a playground with side 27 yards and a garden of 12 by 9 yards is 150 yards -/
theorem fencing_calculation :
  total_fencing 27 12 9 = 150 := by
  sorry

end fencing_calculation_l1692_169218


namespace school_averages_l1692_169202

theorem school_averages 
  (J L : ℕ) -- Number of boys at Jefferson and Lincoln
  (j l : ℕ) -- Number of girls at Jefferson and Lincoln
  (h1 : (68 * J + 73 * j) / (J + j) = 70) -- Jefferson combined average
  (h2 : (68 * J + 78 * L) / (J + L) = 76) -- Boys combined average
  (h3 : J = (3 * j) / 2) -- Derived from h1
  (h4 : J = L) -- Derived from h2
  (h5 : l = j) -- Assumption of equal girls at both schools
  : ((73 * j + 85 * l) / (j + l) = 79) ∧ 
    ((78 * L + 85 * l) / (L + l) = 808/10) :=
by sorry

end school_averages_l1692_169202


namespace prism_with_hole_volume_formula_l1692_169268

/-- The volume of a rectangular prism with a hole running through it -/
def prism_with_hole_volume (x : ℝ) : ℝ :=
  let large_prism_volume := (x + 8) * (x + 6) * 4
  let hole_volume := (2*x - 4) * (x - 3) * 4
  large_prism_volume - hole_volume

/-- Theorem stating the volume of the prism with a hole -/
theorem prism_with_hole_volume_formula (x : ℝ) :
  prism_with_hole_volume x = -4*x^2 + 96*x + 144 :=
by sorry

end prism_with_hole_volume_formula_l1692_169268


namespace sphere_surface_area_l1692_169244

theorem sphere_surface_area (d : ℝ) (h : d = 4) : 
  (π * d^2) = 16 * π := by
  sorry

end sphere_surface_area_l1692_169244


namespace cistern_filling_time_l1692_169233

theorem cistern_filling_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) : 
  empty_rate = 1 / 6 →
  combined_fill_time = 30 →
  1 / fill_time - empty_rate = 1 / combined_fill_time →
  fill_time = 5 := by
sorry

end cistern_filling_time_l1692_169233


namespace x_value_is_five_l1692_169223

theorem x_value_is_five (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : 5 * x^2 + 10 * x * y = x^3 + 2 * x^2 * y) : x = 5 := by
  sorry

end x_value_is_five_l1692_169223


namespace double_sum_equals_one_point_five_l1692_169292

/-- The double sum of 1/(mn(m+n+2)) from m=1 to ∞ and n=1 to ∞ equals 1.5 -/
theorem double_sum_equals_one_point_five :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 2))) = (3 / 2 : ℝ) := by
  sorry

end double_sum_equals_one_point_five_l1692_169292


namespace local_max_at_two_l1692_169221

def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

theorem local_max_at_two (c : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), f c x ≤ f c 2) →
  c = 6 := by
sorry

end local_max_at_two_l1692_169221


namespace tan_315_degrees_l1692_169217

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by sorry

end tan_315_degrees_l1692_169217


namespace rational_expression_equality_inequality_system_solution_l1692_169227

-- Part 1
theorem rational_expression_equality (x : ℝ) (h : x ≠ 3) :
  (2*x + 4) / (x^2 - 6*x + 9) / ((2*x - 1) / (x - 3) - 1) = 2 / (x - 3) :=
sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (5*x - 2 > 3*(x + 1) ∧ (1/2)*x - 1 ≥ 7 - (3/2)*x) ↔ x ≥ 4 :=
sorry

end rational_expression_equality_inequality_system_solution_l1692_169227


namespace circle_radius_with_inscribed_dodecagon_l1692_169237

theorem circle_radius_with_inscribed_dodecagon (Q : ℝ) (R : ℝ) : 
  (R > 0) → 
  (π * R^2 = Q + 3 * R^2) → 
  R = Real.sqrt (Q / (π - 3)) := by
  sorry

end circle_radius_with_inscribed_dodecagon_l1692_169237


namespace jungkook_money_l1692_169224

def initial_amount (notebook_cost pencil_cost remaining : ℕ) : Prop :=
  ∃ (total : ℕ),
    notebook_cost = total / 2 ∧
    pencil_cost = (total - notebook_cost) / 2 ∧
    remaining = total - notebook_cost - pencil_cost ∧
    remaining = 750

theorem jungkook_money : 
  ∀ (notebook_cost pencil_cost remaining : ℕ),
    initial_amount notebook_cost pencil_cost remaining →
    ∃ (total : ℕ), total = 3000 :=
by
  sorry

end jungkook_money_l1692_169224


namespace mikes_books_l1692_169259

/-- Mike's book counting problem -/
theorem mikes_books (initial_books bought_books : ℕ) :
  initial_books = 35 →
  bought_books = 56 →
  initial_books + bought_books = 91 := by
  sorry

end mikes_books_l1692_169259


namespace min_boxes_for_cube_l1692_169275

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of boxes needed to form a cube -/
def boxesNeededForCube (box : BoxDimensions) : ℕ :=
  let lcm := Nat.lcm (Nat.lcm box.width box.length) box.height
  (lcm / box.width) * (lcm / box.length) * (lcm / box.height)

/-- The main theorem stating that 24 boxes are needed to form a cube -/
theorem min_boxes_for_cube :
  let box : BoxDimensions := ⟨18, 12, 9⟩
  boxesNeededForCube box = 24 := by
  sorry

#eval boxesNeededForCube ⟨18, 12, 9⟩

end min_boxes_for_cube_l1692_169275


namespace conic_section_is_hyperbola_l1692_169208

/-- The equation (x+7)^2 = (5y-6)^2 + 125 defines a hyperbola -/
theorem conic_section_is_hyperbola :
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
  ∀ (x y : ℝ), (x + 7)^2 = (5*y - 6)^2 + 125 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0 :=
by sorry

end conic_section_is_hyperbola_l1692_169208


namespace lcm_24_36_45_l1692_169216

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end lcm_24_36_45_l1692_169216


namespace carla_initial_marbles_l1692_169284

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The number of marbles Carla has now -/
def marbles_now : ℕ := 187

/-- The number of marbles Carla started with -/
def marbles_start : ℕ := marbles_now - marbles_bought

theorem carla_initial_marbles : marbles_start = 53 := by
  sorry

end carla_initial_marbles_l1692_169284


namespace slope_characterization_l1692_169279

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

end slope_characterization_l1692_169279


namespace inscribed_circle_square_area_l1692_169298

theorem inscribed_circle_square_area : 
  ∀ (r : ℝ) (s : ℝ),
  r > 0 →
  s > 0 →
  π * r^2 = 9 * π →
  2 * r = s →
  s^2 = 36 :=
by sorry

end inscribed_circle_square_area_l1692_169298


namespace area_of_union_rectangle_circle_l1692_169285

def rectangle_width : ℝ := 8
def rectangle_height : ℝ := 12
def circle_radius : ℝ := 10

theorem area_of_union_rectangle_circle :
  let rectangle_area := rectangle_width * rectangle_height
  let circle_area := π * circle_radius^2
  let overlap_area := (π * circle_radius^2) / 4
  rectangle_area + circle_area - overlap_area = 96 + 75 * π := by
sorry

end area_of_union_rectangle_circle_l1692_169285


namespace air_density_scientific_notation_l1692_169282

/-- The mass per unit volume of air in grams per cubic centimeter -/
def air_density : ℝ := 0.00124

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem air_density_scientific_notation :
  to_scientific_notation air_density = ScientificNotation.mk 1.24 (-3) sorry :=
sorry

end air_density_scientific_notation_l1692_169282


namespace astroid_length_l1692_169222

/-- The astroid curve -/
def astroid (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^(2/3) + p.2^(2/3) = a^(2/3)) ∧ a > 0}

/-- The length of a curve -/
noncomputable def curveLength (C : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The length of the astroid x^(2/3) + y^(2/3) = a^(2/3) is 6a -/
theorem astroid_length (a : ℝ) (h : a > 0) : 
  curveLength (astroid a) = 6 * a := by sorry

end astroid_length_l1692_169222


namespace inequality_solution_range_l1692_169278

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → 
  a ∈ Set.Iio 1 ∪ Set.Ioi 3 := by
sorry

end inequality_solution_range_l1692_169278


namespace fraction_integrality_l1692_169250

theorem fraction_integrality (a b c : ℤ) 
  (h : ∃ (n : ℤ), (a * b : ℚ) / c + (a * c : ℚ) / b + (b * c : ℚ) / a = n) :
  (∃ (n1 : ℤ), (a * b : ℚ) / c = n1) ∧ 
  (∃ (n2 : ℤ), (a * c : ℚ) / b = n2) ∧ 
  (∃ (n3 : ℤ), (b * c : ℚ) / a = n3) := by
sorry

end fraction_integrality_l1692_169250


namespace lineup_combinations_l1692_169213

def total_players : ℕ := 15
def selected_players : ℕ := 2
def lineup_size : ℕ := 5

theorem lineup_combinations :
  Nat.choose (total_players - selected_players) (lineup_size - selected_players) = 286 := by
  sorry

end lineup_combinations_l1692_169213


namespace no_prime_sum_72_l1692_169263

theorem no_prime_sum_72 : ¬∃ (p q k : ℕ), Prime p ∧ Prime q ∧ p + q = 72 ∧ p * q = k := by
  sorry

end no_prime_sum_72_l1692_169263


namespace prime_divisor_form_l1692_169219

theorem prime_divisor_form (p q : ℕ) (hp : Prime p) (hp2 : p > 2) (hq : Prime q) 
  (hdiv : q ∣ (2^p - 1)) : ∃ k : ℕ, q = 2*k*p + 1 := by
  sorry

end prime_divisor_form_l1692_169219


namespace parallelogram_area_18_10_l1692_169286

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 10 cm is 180 cm² -/
theorem parallelogram_area_18_10 : 
  parallelogram_area 18 10 = 180 := by sorry

end parallelogram_area_18_10_l1692_169286


namespace div_exp_eq_pow_reciprocal_l1692_169290

/-- Division exponentiation for rational numbers -/
def div_exp (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then a
  else a / (div_exp a (n - 1))

/-- Theorem: Division exponentiation equals power of reciprocal -/
theorem div_exp_eq_pow_reciprocal (a : ℚ) (n : ℕ) (h1 : a ≠ 0) (h2 : n ≥ 3) :
  div_exp a n = (1 / a) ^ (n - 2) := by
  sorry

end div_exp_eq_pow_reciprocal_l1692_169290


namespace y_value_at_x_2_l1692_169287

theorem y_value_at_x_2 :
  let y₁ := λ x : ℝ => x^2 - 7*x + 6
  let y₂ := λ x : ℝ => 7*x - 3
  let y := λ x : ℝ => y₁ x + x * y₂ x
  y 2 = 18 := by sorry

end y_value_at_x_2_l1692_169287


namespace smallest_valid_n_l1692_169270

def is_valid (n : ℕ) : Prop :=
  Real.sqrt (n : ℝ) - Real.sqrt ((n - 1) : ℝ) < 0.01

theorem smallest_valid_n :
  (∀ m : ℕ, m < 2501 → ¬(is_valid m)) ∧ is_valid 2501 := by
  sorry

end smallest_valid_n_l1692_169270


namespace motel_rent_theorem_l1692_169265

/-- Represents the total rent charged by a motel on a Saturday night. -/
def TotalRent : ℕ → ℕ → ℕ 
  | r50, r60 => 50 * r50 + 60 * r60

/-- Represents the condition that changing 10 rooms from $60 to $50 reduces the rent by 25%. -/
def RentReductionCondition (r50 r60 : ℕ) : Prop :=
  4 * (TotalRent (r50 + 10) (r60 - 10)) = 3 * (TotalRent r50 r60)

theorem motel_rent_theorem :
  ∃ (r50 r60 : ℕ), RentReductionCondition r50 r60 ∧ TotalRent r50 r60 = 400 :=
sorry

end motel_rent_theorem_l1692_169265


namespace cargo_passenger_relationship_l1692_169296

/-- Represents a train with passenger cars and cargo cars. -/
structure Train where
  total_cars : ℕ
  passenger_cars : ℕ
  cargo_cars : ℕ

/-- Defines the properties of our specific train. -/
def our_train : Train where
  total_cars := 71
  passenger_cars := 44
  cargo_cars := 25

/-- Theorem stating the relationship between cargo cars and passenger cars. -/
theorem cargo_passenger_relationship (t : Train) 
  (h1 : t.total_cars = t.passenger_cars + t.cargo_cars + 2) 
  (h2 : t.cargo_cars = t.passenger_cars / 2 + (t.cargo_cars - t.passenger_cars / 2)) : 
  t.cargo_cars - t.passenger_cars / 2 = 3 :=
by sorry

end cargo_passenger_relationship_l1692_169296


namespace simplify_trig_expression_l1692_169246

theorem simplify_trig_expression :
  let tan_45 : ℝ := 1
  let cot_45 : ℝ := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 := by
  sorry

end simplify_trig_expression_l1692_169246


namespace ice_cream_flavors_l1692_169230

theorem ice_cream_flavors (n k : ℕ) (hn : n = 4) (hk : k = 4) :
  (n + k - 1).choose (k - 1) = 35 := by
  sorry

end ice_cream_flavors_l1692_169230


namespace sum_of_eight_smallest_multiples_of_12_l1692_169281

theorem sum_of_eight_smallest_multiples_of_12 : 
  (Finset.range 8).sum (λ i => 12 * (i + 1)) = 432 := by
  sorry

end sum_of_eight_smallest_multiples_of_12_l1692_169281


namespace consecutive_even_integers_l1692_169211

theorem consecutive_even_integers (a b c d e : ℤ) : 
  (∃ n : ℤ, a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8) →  -- five consecutive even integers
  (a + e = 204) →  -- sum of first and last is 204
  (a + b + c + d + e = 510) ∧ (a = 98)  -- sum is 510 and smallest is 98
  := by sorry

end consecutive_even_integers_l1692_169211


namespace batsman_running_percentage_l1692_169236

def batsman_score_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℚ :=
  let boundary_runs := 4 * boundaries
  let six_runs := 6 * sixes
  let runs_without_running := boundary_runs + six_runs
  let runs_by_running := total_runs - runs_without_running
  (runs_by_running : ℚ) / total_runs * 100

theorem batsman_running_percentage :
  batsman_score_percentage 120 3 8 = 50 := by sorry

end batsman_running_percentage_l1692_169236


namespace joyce_farmland_l1692_169253

/-- Calculates the area of land suitable for growing vegetables given the size of the previous property, 
    the factor by which the new property is larger, and the size of a pond on the new property. -/
def land_for_vegetables (prev_property : ℝ) (size_factor : ℝ) (pond_size : ℝ) : ℝ :=
  prev_property * size_factor - pond_size

/-- Theorem stating that given a previous property of 2 acres, a new property 10 times larger, 
    and a 1-acre pond, the land suitable for growing vegetables is 19 acres. -/
theorem joyce_farmland : land_for_vegetables 2 10 1 = 19 := by
  sorry

end joyce_farmland_l1692_169253


namespace garden_area_l1692_169289

/-- A rectangular garden with perimeter 36 feet and one side 10 feet has an area of 80 square feet. -/
theorem garden_area (perimeter : ℝ) (side : ℝ) (h1 : perimeter = 36) (h2 : side = 10) :
  let other_side := (perimeter - 2 * side) / 2
  side * other_side = 80 :=
by sorry

end garden_area_l1692_169289
