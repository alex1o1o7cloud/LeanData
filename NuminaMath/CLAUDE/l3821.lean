import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_function_from_roots_and_point_l3821_382143

theorem quadratic_function_from_roots_and_point (f : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →  -- f is a quadratic function
  f 0 = 2 →                                       -- f(0) = 2
  (∃ x, f x = 0 ∧ x = -2) →                       -- -2 is a root
  (∃ x, f x = 0 ∧ x = 1) →                        -- 1 is a root
  ∀ x, f x = -x^2 - x + 2 :=                      -- Conclusion: f(x) = -x^2 - x + 2
by sorry

end NUMINAMATH_CALUDE_quadratic_function_from_roots_and_point_l3821_382143


namespace NUMINAMATH_CALUDE_fourth_ball_black_prob_l3821_382144

/-- A box containing colored balls. -/
structure Box where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball from a box. -/
def prob_black_ball (box : Box) : ℚ :=
  box.black_balls / (box.red_balls + box.black_balls)

/-- The theorem stating that the probability of the fourth ball being black
    is equal to the probability of selecting a black ball from the box. -/
theorem fourth_ball_black_prob (box : Box) (h1 : box.red_balls = 2) (h2 : box.black_balls = 5) :
  prob_black_ball box = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_ball_black_prob_l3821_382144


namespace NUMINAMATH_CALUDE_hyperbola_intersection_midpoint_l3821_382115

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the line
def line (x y : ℝ) : Prop := 4*x - y - 7 = 0

-- Theorem statement
theorem hyperbola_intersection_midpoint :
  ∃ (A B : ℝ × ℝ),
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    line A.1 A.2 ∧
    line B.1 B.2 ∧
    line P.1 P.2 ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_midpoint_l3821_382115


namespace NUMINAMATH_CALUDE_largest_fraction_l3821_382153

theorem largest_fraction : 
  let a := (5 : ℚ) / 12
  let b := (7 : ℚ) / 16
  let c := (23 : ℚ) / 48
  let d := (99 : ℚ) / 200
  let e := (201 : ℚ) / 400
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l3821_382153


namespace NUMINAMATH_CALUDE_best_calorie_deal_l3821_382187

-- Define the food options
structure FoodOption where
  name : String
  quantity : Nat
  price : Nat
  caloriesPerItem : Nat

-- Define the function to calculate calories per dollar
def caloriesPerDollar (option : FoodOption) : Rat :=
  (option.quantity * option.caloriesPerItem : Rat) / option.price

-- Define the food options
def burritos : FoodOption := ⟨"Burritos", 10, 6, 120⟩
def burgers : FoodOption := ⟨"Burgers", 5, 8, 400⟩
def pizza : FoodOption := ⟨"Pizza", 8, 10, 300⟩
def donuts : FoodOption := ⟨"Donuts", 15, 12, 250⟩

-- Define the list of food options
def foodOptions : List FoodOption := [burritos, burgers, pizza, donuts]

-- Theorem statement
theorem best_calorie_deal :
  (caloriesPerDollar donuts = 312.5) ∧
  (∀ option ∈ foodOptions, caloriesPerDollar option ≤ caloriesPerDollar donuts) ∧
  (caloriesPerDollar donuts - caloriesPerDollar burgers = 62.5) :=
sorry

end NUMINAMATH_CALUDE_best_calorie_deal_l3821_382187


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3821_382158

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3821_382158


namespace NUMINAMATH_CALUDE_integer_division_property_l3821_382191

theorem integer_division_property (n : ℕ) : 
  100 ≤ n ∧ n ≤ 1997 →
  (∃ k : ℕ, (2^n + 2 : ℕ) = k * n) ↔ n ∈ ({66, 198, 398, 798} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_integer_division_property_l3821_382191


namespace NUMINAMATH_CALUDE_orange_picking_ratio_l3821_382139

/-- Proves that the ratio of oranges picked on Tuesday to Monday is 3:1 --/
theorem orange_picking_ratio :
  let monday_oranges : ℕ := 100
  let wednesday_oranges : ℕ := 70
  let total_oranges : ℕ := 470
  let tuesday_oranges : ℕ := total_oranges - monday_oranges - wednesday_oranges
  tuesday_oranges / monday_oranges = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_picking_ratio_l3821_382139


namespace NUMINAMATH_CALUDE_women_per_table_l3821_382174

theorem women_per_table (tables : Nat) (men_per_table : Nat) (total_customers : Nat) :
  tables = 9 →
  men_per_table = 3 →
  total_customers = 90 →
  (total_customers - tables * men_per_table) / tables = 7 := by
  sorry

end NUMINAMATH_CALUDE_women_per_table_l3821_382174


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l3821_382172

theorem smallest_divisible_by_1_to_12 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → m ≥ 27720) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l3821_382172


namespace NUMINAMATH_CALUDE_binomial_probability_two_successes_l3821_382159

/-- The probability mass function for a Binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- Theorem: For a random variable ξ following Binomial distribution B(3, 1/3), P(ξ=2) = 2/9 -/
theorem binomial_probability_two_successes :
  binomial_pmf 3 (1/3 : ℝ) 2 = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_two_successes_l3821_382159


namespace NUMINAMATH_CALUDE_some_number_value_l3821_382168

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3821_382168


namespace NUMINAMATH_CALUDE_positive_solution_x_l3821_382111

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 12 - 3 * x - 4 * y)
  (eq2 : y * z = 8 - 2 * y - 3 * z)
  (eq3 : x * z = 42 - 5 * x - 6 * z)
  (h_positive : x > 0) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_x_l3821_382111


namespace NUMINAMATH_CALUDE_alvez_family_has_three_children_l3821_382142

/-- Represents the Alvez family structure -/
structure AlvezFamily where
  num_children : ℕ
  mother_age : ℕ
  children_ages : Fin num_children → ℕ

/-- The average age of a list of ages -/
def average_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

/-- The Alvez family satisfies the given conditions -/
def satisfies_conditions (family : AlvezFamily) : Prop :=
  let total_members := family.num_children + 2
  let all_ages := family.mother_age :: 50 :: (List.ofFn family.children_ages)
  average_age all_ages = 22 ∧
  average_age (family.mother_age :: (List.ofFn family.children_ages)) = 15

/-- The main theorem: There are exactly 3 children in the Alvez family -/
theorem alvez_family_has_three_children :
  ∃ (family : AlvezFamily), satisfies_conditions family ∧ family.num_children = 3 :=
sorry

end NUMINAMATH_CALUDE_alvez_family_has_three_children_l3821_382142


namespace NUMINAMATH_CALUDE_used_books_count_l3821_382156

def total_books : ℕ := 30
def new_books : ℕ := 15

theorem used_books_count : total_books - new_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_used_books_count_l3821_382156


namespace NUMINAMATH_CALUDE_lychee_production_increase_l3821_382105

theorem lychee_production_increase (x : ℝ) : 
  let increase_factor := 1 + x / 100
  let two_year_increase := increase_factor ^ 2 - 1
  two_year_increase = ((1 + x / 100) ^ 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_lychee_production_increase_l3821_382105


namespace NUMINAMATH_CALUDE_dads_first_half_speed_is_28_l3821_382180

/-- The speed of Jake's dad during the first half of the journey to the water park -/
def dads_first_half_speed : ℝ := by sorry

/-- The total journey time for Jake's dad in hours -/
def total_journey_time : ℝ := 0.5

/-- Jake's biking speed in miles per hour -/
def jake_bike_speed : ℝ := 11

/-- Time it takes Jake to bike to the water park in hours -/
def jake_bike_time : ℝ := 2

/-- Jake's dad's speed during the second half of the journey in miles per hour -/
def dads_second_half_speed : ℝ := 60

theorem dads_first_half_speed_is_28 :
  dads_first_half_speed = 28 := by sorry

end NUMINAMATH_CALUDE_dads_first_half_speed_is_28_l3821_382180


namespace NUMINAMATH_CALUDE_five_people_handshakes_l3821_382173

/-- The number of handshakes when n people meet, where each pair shakes hands exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: When 5 people meet, they shake hands a total of 10 times -/
theorem five_people_handshakes : handshakes 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_five_people_handshakes_l3821_382173


namespace NUMINAMATH_CALUDE_abel_overtake_distance_l3821_382185

/-- Represents the race scenario between Abel and Kelly -/
structure RaceScenario where
  totalDistance : ℝ
  headStart : ℝ
  lossDistance : ℝ

/-- Calculates the distance Abel needs to run to overtake Kelly -/
def distanceToOvertake (race : RaceScenario) : ℝ :=
  race.totalDistance - (race.totalDistance - race.headStart + race.lossDistance)

/-- Theorem stating that Abel needs to run 98 meters to overtake Kelly -/
theorem abel_overtake_distance (race : RaceScenario) 
  (h1 : race.totalDistance = 100)
  (h2 : race.headStart = 3)
  (h3 : race.lossDistance = 0.5) :
  distanceToOvertake race = 98 := by
  sorry

#eval distanceToOvertake { totalDistance := 100, headStart := 3, lossDistance := 0.5 }

end NUMINAMATH_CALUDE_abel_overtake_distance_l3821_382185


namespace NUMINAMATH_CALUDE_complex_number_product_l3821_382117

theorem complex_number_product (a b c d : ℂ) : 
  (a + b + c + d = 5) →
  ((5 - a)^4 + (5 - b)^4 + (5 - c)^4 + (5 - d)^4 = 125) →
  ((a + b)^4 + (b + c)^4 + (c + d)^4 + (d + a)^4 + (a + c)^4 + (b + d)^4 = 1205) →
  (a^4 + b^4 + c^4 + d^4 = 25) →
  a * b * c * d = 70 := by
sorry

end NUMINAMATH_CALUDE_complex_number_product_l3821_382117


namespace NUMINAMATH_CALUDE_angle_relationship_l3821_382110

theorem angle_relationship (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : 2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β) : 
  α < β := by
sorry

end NUMINAMATH_CALUDE_angle_relationship_l3821_382110


namespace NUMINAMATH_CALUDE_mikey_has_125_jelly_beans_l3821_382165

/-- The number of jelly beans each person has -/
structure JellyBeans where
  napoleon : ℕ
  sedrich : ℕ
  daphne : ℕ
  alondra : ℕ
  mikey : ℕ

/-- The conditions of the jelly bean problem -/
def jelly_bean_conditions (jb : JellyBeans) : Prop :=
  jb.napoleon = 56 ∧
  jb.sedrich = 3 * jb.napoleon + 9 ∧
  jb.daphne = 2 * (jb.sedrich - jb.napoleon) ∧
  jb.alondra = (jb.napoleon + jb.sedrich + jb.daphne) / 3 - 8 ∧
  jb.napoleon + jb.sedrich + jb.daphne + jb.alondra = 5 * jb.mikey

/-- The theorem stating that under the given conditions, Mikey has 125 jelly beans -/
theorem mikey_has_125_jelly_beans (jb : JellyBeans) 
  (h : jelly_bean_conditions jb) : jb.mikey = 125 := by
  sorry


end NUMINAMATH_CALUDE_mikey_has_125_jelly_beans_l3821_382165


namespace NUMINAMATH_CALUDE_solution_set_equality_l3821_382155

theorem solution_set_equality : 
  {x : ℝ | x^2 - 2*x ≤ 0} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3821_382155


namespace NUMINAMATH_CALUDE_min_value_problem_l3821_382102

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : Real.log x + Real.log y = 2) :
  5 * x + 2 * y ≥ 20 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3821_382102


namespace NUMINAMATH_CALUDE_log_product_equals_five_l3821_382132

theorem log_product_equals_five :
  (Real.log 4 / Real.log 2) * (Real.log 8 / Real.log 4) *
  (Real.log 16 / Real.log 8) * (Real.log 32 / Real.log 16) = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_five_l3821_382132


namespace NUMINAMATH_CALUDE_smallest_b_value_l3821_382193

theorem smallest_b_value : ∃ (b : ℝ), b > 0 ∧
  (∀ (x : ℝ), x > 0 →
    (9 * Real.sqrt ((3 * x)^2 + 2^2) - 6 * x^2 - 4) / (Real.sqrt (4 + 6 * x^2) + 5) = 3 →
    b ≤ x) ∧
  (9 * Real.sqrt ((3 * b)^2 + 2^2) - 6 * b^2 - 4) / (Real.sqrt (4 + 6 * b^2) + 5) = 3 ∧
  b = Real.sqrt (11 / 30) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3821_382193


namespace NUMINAMATH_CALUDE_division_multiplication_negatives_l3821_382170

theorem division_multiplication_negatives : (-100) / (-25) * (-6) = -24 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_negatives_l3821_382170


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3821_382184

/-- The minimum distance from the origin to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + y = 4}
  ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧ 
    ∀ P ∈ line, Real.sqrt (P.1^2 + P.2^2) ≥ d ∧
    ∃ Q ∈ line, Real.sqrt (Q.1^2 + Q.2^2) = d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3821_382184


namespace NUMINAMATH_CALUDE_coin_array_problem_l3821_382183

/-- The number of coins in a triangular array with n rows -/
def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem coin_array_problem :
  ∃ N : ℕ, triangle_sum N = 2080 ∧ sum_of_digits N = 10 := by
  sorry

end NUMINAMATH_CALUDE_coin_array_problem_l3821_382183


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3821_382120

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℝ),
  (x₁ = Real.sqrt 3 ∧ x₂ = (Real.sqrt 3) / 3 ∧ x₃ = -(2 * Real.sqrt 3)) ∧
  (x₁ * x₂ = 1) ∧
  (3 * x₁^3 + 2 * Real.sqrt 3 * x₁^2 - 21 * x₁ + 6 * Real.sqrt 3 = 0) ∧
  (3 * x₂^3 + 2 * Real.sqrt 3 * x₂^2 - 21 * x₂ + 6 * Real.sqrt 3 = 0) ∧
  (3 * x₃^3 + 2 * Real.sqrt 3 * x₃^2 - 21 * x₃ + 6 * Real.sqrt 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3821_382120


namespace NUMINAMATH_CALUDE_five_spheres_configuration_exists_l3821_382179

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Checks if a plane is tangent to a sphere -/
def isTangent (p : Plane) (s : Sphere) : Prop := sorry

/-- Checks if a plane passes through a point -/
def passesThrough (p : Plane) (point : ℝ × ℝ × ℝ) : Prop := sorry

/-- Theorem stating the existence of a configuration of five spheres with the required property -/
theorem five_spheres_configuration_exists : 
  ∃ (s₁ s₂ s₃ s₄ s₅ : Sphere),
    (∃ (p₁ : Plane), passesThrough p₁ s₁.center ∧ 
      isTangent p₁ s₂ ∧ isTangent p₁ s₃ ∧ isTangent p₁ s₄ ∧ isTangent p₁ s₅) ∧
    (∃ (p₂ : Plane), passesThrough p₂ s₂.center ∧ 
      isTangent p₂ s₁ ∧ isTangent p₂ s₃ ∧ isTangent p₂ s₄ ∧ isTangent p₂ s₅) ∧
    (∃ (p₃ : Plane), passesThrough p₃ s₃.center ∧ 
      isTangent p₃ s₁ ∧ isTangent p₃ s₂ ∧ isTangent p₃ s₄ ∧ isTangent p₃ s₅) ∧
    (∃ (p₄ : Plane), passesThrough p₄ s₄.center ∧ 
      isTangent p₄ s₁ ∧ isTangent p₄ s₂ ∧ isTangent p₄ s₃ ∧ isTangent p₄ s₅) ∧
    (∃ (p₅ : Plane), passesThrough p₅ s₅.center ∧ 
      isTangent p₅ s₁ ∧ isTangent p₅ s₂ ∧ isTangent p₅ s₃ ∧ isTangent p₅ s₄) :=
by
  sorry

end NUMINAMATH_CALUDE_five_spheres_configuration_exists_l3821_382179


namespace NUMINAMATH_CALUDE_det_equals_xy_l3821_382113

/-- The determinant of the matrix
    [1, x, y]
    [1, x+y, y]
    [1, x, x+y]
    is equal to xy -/
theorem det_equals_xy (x y : ℝ) : 
  Matrix.det !![1, x, y; 1, x+y, y; 1, x, x+y] = x * y := by
  sorry

end NUMINAMATH_CALUDE_det_equals_xy_l3821_382113


namespace NUMINAMATH_CALUDE_scientific_notation_of_3185800_l3821_382108

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Rounds a ScientificNotation to a specified number of decimal places -/
def roundToDecimalPlaces (sn : ScientificNotation) (places : ℕ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_3185800 :
  let original := 3185800
  let scientificForm := toScientificNotation original
  let rounded := roundToDecimalPlaces scientificForm 1
  rounded.coefficient = 3.2 ∧ rounded.exponent = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_3185800_l3821_382108


namespace NUMINAMATH_CALUDE_coefficient_of_x_term_l3821_382162

theorem coefficient_of_x_term (x : ℝ) : 
  let expansion := (Real.sqrt x - 2 / x) ^ 5
  ∃ c : ℝ, c = -10 ∧ 
    ∃ t : ℝ → ℝ, (expansion = c * x + t x) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → |t h / h| < ε) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_term_l3821_382162


namespace NUMINAMATH_CALUDE_f_simplification_inverse_sum_value_l3821_382195

noncomputable def f (α : Real) : Real :=
  (Real.sin (2 * Real.pi - α) * Real.cos (Real.pi + α) * Real.cos (Real.pi / 2 - α) * Real.cos (11 * Real.pi / 2 - α)) /
  (Real.sin (3 * Real.pi - α) * Real.cos (Real.pi / 2 + α) * Real.sin (9 * Real.pi / 2 + α)) +
  Real.cos (2 * Real.pi - α)

theorem f_simplification (α : Real) : f α = Real.sin α + Real.cos α := by sorry

theorem inverse_sum_value (α : Real) (h : f α = Real.sqrt 10 / 5) :
  1 / Real.sin α + 1 / Real.cos α = -4 * Real.sqrt 10 / 3 := by sorry

end NUMINAMATH_CALUDE_f_simplification_inverse_sum_value_l3821_382195


namespace NUMINAMATH_CALUDE_watermelon_seeds_l3821_382176

/-- Given 4 watermelons with a total of 400 seeds, prove that each watermelon has 100 seeds. -/
theorem watermelon_seeds (num_watermelons : ℕ) (total_seeds : ℕ) 
  (h1 : num_watermelons = 4) 
  (h2 : total_seeds = 400) : 
  total_seeds / num_watermelons = 100 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_seeds_l3821_382176


namespace NUMINAMATH_CALUDE_complex_division_l3821_382112

theorem complex_division (i : ℂ) (h : i * i = -1) : 
  (2 - i) / (1 + i) = 1/2 - 3/2 * i := by sorry

end NUMINAMATH_CALUDE_complex_division_l3821_382112


namespace NUMINAMATH_CALUDE_max_value_a_l3821_382160

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 2 * d)
  (h4 : d < 100) :
  a ≤ 1179 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 1179 ∧
    a' < 2 * b' ∧
    b' < 3 * c' ∧
    c' < 2 * d' ∧
    d' < 100 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l3821_382160


namespace NUMINAMATH_CALUDE_rachel_homework_pages_l3821_382192

/-- The total number of pages for math and biology homework -/
def total_math_biology_pages (math_pages biology_pages : ℕ) : ℕ :=
  math_pages + biology_pages

/-- Theorem: Given Rachel has 8 pages of math homework and 3 pages of biology homework,
    the total number of pages for math and biology homework is 11. -/
theorem rachel_homework_pages : total_math_biology_pages 8 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_pages_l3821_382192


namespace NUMINAMATH_CALUDE_floral_shop_sale_l3821_382150

/-- Represents the number of bouquets sold on each day of a three-day sale. -/
structure SaleData where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Theorem stating the conditions of the sale and the result to be proven. -/
theorem floral_shop_sale (sale : SaleData) : 
  sale.tuesday = 3 * sale.monday ∧ 
  sale.wednesday = sale.tuesday / 3 ∧
  sale.monday + sale.tuesday + sale.wednesday = 60 →
  sale.monday = 12 := by
  sorry

end NUMINAMATH_CALUDE_floral_shop_sale_l3821_382150


namespace NUMINAMATH_CALUDE_seventh_term_of_sequence_l3821_382123

def geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ := a * r^(n - 1)

theorem seventh_term_of_sequence (a₁ a₅ : ℕ) (h₁ : a₁ = 3) (h₅ : a₅ = 243) :
  ∃ r : ℕ, 
    (geometric_sequence a₁ r 5 = a₅) ∧ 
    (geometric_sequence a₁ r 7 = 2187) := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_sequence_l3821_382123


namespace NUMINAMATH_CALUDE_complement_of_B_l3821_382126

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def B : Set Nat := {1, 3, 5, 7}

theorem complement_of_B :
  (U \ B) = {2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_B_l3821_382126


namespace NUMINAMATH_CALUDE_blue_marble_probability_l3821_382124

theorem blue_marble_probability (total : ℕ) (yellow : ℕ) :
  total = 120 →
  yellow = 30 →
  let green := yellow / 3
  let red := 2 * yellow
  let blue := total - (yellow + green + red)
  (blue : ℚ) / total = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_blue_marble_probability_l3821_382124


namespace NUMINAMATH_CALUDE_withdrawal_theorem_l3821_382127

/-- Calculates the number of bills received when withdrawing from two banks -/
def number_of_bills (withdrawal_per_bank : ℕ) (bill_denomination : ℕ) : ℕ :=
  (2 * withdrawal_per_bank) / bill_denomination

/-- Theorem: Withdrawing $300 from each of two banks in $20 bills results in 30 bills -/
theorem withdrawal_theorem : number_of_bills 300 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_withdrawal_theorem_l3821_382127


namespace NUMINAMATH_CALUDE_house_construction_bricks_house_construction_bricks_specific_l3821_382104

/-- Calculates the number of bricks needed for house construction given specific costs and requirements. -/
theorem house_construction_bricks (land_cost_per_sqm : ℕ) (brick_cost_per_thousand : ℕ) 
  (roof_tile_cost : ℕ) (land_area : ℕ) (roof_tiles : ℕ) (total_cost : ℕ) : ℕ :=
  let land_cost := land_cost_per_sqm * land_area
  let roof_cost := roof_tile_cost * roof_tiles
  let brick_budget := total_cost - land_cost - roof_cost
  let bricks_thousands := brick_budget / brick_cost_per_thousand
  bricks_thousands * 1000

/-- Proves that given the specific conditions, the number of bricks needed is 10,000. -/
theorem house_construction_bricks_specific : 
  house_construction_bricks 50 100 10 2000 500 106000 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_house_construction_bricks_house_construction_bricks_specific_l3821_382104


namespace NUMINAMATH_CALUDE_luncheon_table_capacity_l3821_382128

/-- Given a luncheon where 24 people were invited, 10 didn't show up, and 2 tables were needed,
    prove that each table could hold 7 people. -/
theorem luncheon_table_capacity :
  ∀ (invited : ℕ) (no_show : ℕ) (tables : ℕ) (capacity : ℕ),
    invited = 24 →
    no_show = 10 →
    tables = 2 →
    capacity = (invited - no_show) / tables →
    capacity = 7 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_table_capacity_l3821_382128


namespace NUMINAMATH_CALUDE_sum_squared_expression_lower_bound_l3821_382194

theorem sum_squared_expression_lower_bound 
  (x y z : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (h_sum : x + y + z = x * y * z) : 
  ((x^2 - 1) / x)^2 + ((y^2 - 1) / y)^2 + ((z^2 - 1) / z)^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_expression_lower_bound_l3821_382194


namespace NUMINAMATH_CALUDE_movie_collection_average_usage_l3821_382133

/-- Given a movie collection that occupies 27,000 megabytes of disk space and lasts for 15 days
    of continuous viewing, the average megabyte usage per hour is 75 megabytes. -/
theorem movie_collection_average_usage
  (total_megabytes : ℕ)
  (total_days : ℕ)
  (h_megabytes : total_megabytes = 27000)
  (h_days : total_days = 15) :
  (total_megabytes : ℚ) / (total_days * 24 : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_movie_collection_average_usage_l3821_382133


namespace NUMINAMATH_CALUDE_quadratic_trinomials_equal_sum_squares_l3821_382141

/-- 
Given two quadratic trinomials f(x) = x^2 - 6x + 4a and g(x) = x^2 + ax + 6,
prove that a = -12 is the only value for which both trinomials have two roots
and the sum of the squares of the roots of f(x) equals the sum of the squares
of the roots of g(x).
-/
theorem quadratic_trinomials_equal_sum_squares (a : ℝ) : 
  (∃ x y : ℝ, x^2 - 6*x + 4*a = 0 ∧ y^2 + a*y + 6 = 0) ∧ 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁^2 - 6*x₁ + 4*a = 0 ∧ 
    x₂^2 - 6*x₂ + 4*a = 0 ∧ 
    y₁^2 + a*y₁ + 6 = 0 ∧ 
    y₂^2 + a*y₂ + 6 = 0 ∧ 
    x₁^2 + x₂^2 = y₁^2 + y₂^2) →
  a = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomials_equal_sum_squares_l3821_382141


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3821_382136

/-- Two triangles are similar -/
structure SimilarTriangles (T1 T2 : Type) :=
  (ratio : ℝ)
  (similar : T1 → T2 → Prop)

/-- Triangle XYZ -/
structure TriangleXYZ :=
  (X Y Z : ℝ × ℝ)
  (YZ : ℝ)
  (XZ : ℝ)

/-- Triangle MNP -/
structure TriangleMNP :=
  (M N P : ℝ × ℝ)
  (MN : ℝ)
  (NP : ℝ)

theorem similar_triangles_side_length 
  (XYZ : TriangleXYZ) 
  (MNP : TriangleMNP) 
  (sim : SimilarTriangles TriangleXYZ TriangleMNP) 
  (h_sim : sim.similar XYZ MNP) 
  (h_YZ : XYZ.YZ = 10) 
  (h_XZ : XYZ.XZ = 7) 
  (h_MN : MNP.MN = 4.2) : 
  MNP.NP = 6 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3821_382136


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3821_382188

/-- The y-coordinate of the vertex of the parabola y = -3x^2 - 30x - 81 is -6 -/
theorem parabola_vertex_y_coordinate :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 - 30 * x - 81
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≤ f x₀ ∧ f x₀ = -6 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3821_382188


namespace NUMINAMATH_CALUDE_three_day_trip_mileage_l3821_382140

theorem three_day_trip_mileage (total_miles : ℕ) (day1_miles : ℕ) (day2_miles : ℕ) 
  (h1 : total_miles = 493) 
  (h2 : day1_miles = 125) 
  (h3 : day2_miles = 223) : 
  total_miles - (day1_miles + day2_miles) = 145 := by
  sorry

end NUMINAMATH_CALUDE_three_day_trip_mileage_l3821_382140


namespace NUMINAMATH_CALUDE_yoongis_rank_l3821_382118

theorem yoongis_rank (namjoons_rank yoongis_rank : ℕ) : 
  namjoons_rank = 2 →
  yoongis_rank = namjoons_rank + 10 →
  yoongis_rank = 12 :=
by sorry

end NUMINAMATH_CALUDE_yoongis_rank_l3821_382118


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_max_is_six_l3821_382137

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x < 1} = {x : ℝ | x < 1/2} := by sorry

-- Part II
theorem a_value_when_max_is_six :
  (∃ (x : ℝ), f a x = 6) ∧ (∀ (x : ℝ), f a x ≤ 6) → a = 5 ∨ a = -7 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_max_is_six_l3821_382137


namespace NUMINAMATH_CALUDE_x_divisibility_l3821_382148

def x : ℕ := 48 + 64 + 192 + 256 + 384 + 768 + 1024

theorem x_divisibility :
  (∃ k : ℕ, x = 4 * k) ∧
  (∃ k : ℕ, x = 16 * k) ∧
  ¬(∀ k : ℕ, x = 64 * k) ∧
  ¬(∀ k : ℕ, x = 128 * k) := by
  sorry

end NUMINAMATH_CALUDE_x_divisibility_l3821_382148


namespace NUMINAMATH_CALUDE_perfect_squares_condition_l3821_382135

theorem perfect_squares_condition (n : ℕ) : 
  (∃ k m : ℕ, 2 * n + 1 = k^2 ∧ 3 * n + 1 = m^2) ↔ 
  (∃ a b : ℕ, n + 1 = a^2 + (a + 1)^2 ∧ ∃ c : ℕ, n + 1 = c^2 + 2 * (c + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_condition_l3821_382135


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_8_equals_sqrt_2_l3821_382169

theorem sqrt_18_minus_sqrt_8_equals_sqrt_2 : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_8_equals_sqrt_2_l3821_382169


namespace NUMINAMATH_CALUDE_tan_double_angle_l3821_382114

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/2) : 
  Real.tan (2 * α) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3821_382114


namespace NUMINAMATH_CALUDE_typists_productivity_l3821_382131

/-- Given that 20 typists can type 42 letters in 20 minutes, 
    prove that 30 typists can type 189 letters in 1 hour at the same rate. -/
theorem typists_productivity (typists_20 : ℕ) (letters_20 : ℕ) (minutes_20 : ℕ) 
  (typists_30 : ℕ) (minutes_60 : ℕ) :
  typists_20 = 20 →
  letters_20 = 42 →
  minutes_20 = 20 →
  typists_30 = 30 →
  minutes_60 = 60 →
  (typists_30 : ℚ) * (letters_20 : ℚ) / (typists_20 : ℚ) * (minutes_60 : ℚ) / (minutes_20 : ℚ) = 189 :=
by sorry

end NUMINAMATH_CALUDE_typists_productivity_l3821_382131


namespace NUMINAMATH_CALUDE_inequality_proof_l3821_382103

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a < b + c) :
  a / (1 + a) < b / (1 + b) + c / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3821_382103


namespace NUMINAMATH_CALUDE_complex_cube_root_l3821_382161

theorem complex_cube_root (a b : ℕ+) :
  (↑a + ↑b * Complex.I) ^ 3 = 2 + 11 * Complex.I →
  ↑a + ↑b * Complex.I = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l3821_382161


namespace NUMINAMATH_CALUDE_inequality_proof_l3821_382197

theorem inequality_proof (n : ℕ) (x : ℝ) (h1 : n ≥ 2) (h2 : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3821_382197


namespace NUMINAMATH_CALUDE_max_prism_pyramid_elements_l3821_382100

/-- A shape formed by fusing a rectangular prism with a pyramid on one of its faces -/
structure PrismPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_new_faces : ℕ
  pyramid_new_edges : ℕ
  pyramid_new_vertex : ℕ

/-- The sum of exterior faces, vertices, and edges of a PrismPyramid -/
def total_elements (pp : PrismPyramid) : ℕ :=
  (pp.prism_faces - 1 + pp.pyramid_new_faces) +
  (pp.prism_edges + pp.pyramid_new_edges) +
  (pp.prism_vertices + pp.pyramid_new_vertex)

/-- Theorem stating that the maximum sum of exterior faces, vertices, and edges is 34 -/
theorem max_prism_pyramid_elements :
  ∃ (pp : PrismPyramid), 
    pp.prism_faces = 6 ∧
    pp.prism_edges = 12 ∧
    pp.prism_vertices = 8 ∧
    pp.pyramid_new_faces = 4 ∧
    pp.pyramid_new_edges = 4 ∧
    pp.pyramid_new_vertex = 1 ∧
    total_elements pp = 34 ∧
    ∀ (pp' : PrismPyramid), total_elements pp' ≤ 34 :=
  sorry

end NUMINAMATH_CALUDE_max_prism_pyramid_elements_l3821_382100


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3821_382186

theorem unique_solution_equation : ∃! x : ℝ, 3 * x + 3 * 15 + 3 * 18 + 11 = 152 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3821_382186


namespace NUMINAMATH_CALUDE_fence_cost_l3821_382125

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 60) :
  4 * Real.sqrt area * price_per_foot = 4080 := by
  sorry

#check fence_cost

end NUMINAMATH_CALUDE_fence_cost_l3821_382125


namespace NUMINAMATH_CALUDE_point_division_theorem_l3821_382198

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points A, B, and P
variable (A B P : V)

-- Define the condition that P is on the line segment AB with the given ratio
def on_segment_with_ratio (A B P : V) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B ∧ t = 5 / 8

-- Theorem statement
theorem point_division_theorem (h : on_segment_with_ratio A B P) :
  P = (3 / 8) • A + (5 / 8) • B := by sorry

end NUMINAMATH_CALUDE_point_division_theorem_l3821_382198


namespace NUMINAMATH_CALUDE_retirement_sum_is_70_l3821_382175

/-- Represents the retirement policy of a company -/
structure RetirementPolicy where
  hireYear : Nat
  hireAge : Nat
  retirementYear : Nat
  retirementSum : Nat

/-- Theorem: The required total of age and years of employment for retirement is 70 -/
theorem retirement_sum_is_70 (policy : RetirementPolicy) 
  (h1 : policy.hireYear = 1987)
  (h2 : policy.hireAge = 32)
  (h3 : policy.retirementYear = 2006) :
  policy.retirementSum = 70 := by
  sorry

#check retirement_sum_is_70

end NUMINAMATH_CALUDE_retirement_sum_is_70_l3821_382175


namespace NUMINAMATH_CALUDE_max_sphere_volume_in_prism_l3821_382134

/-- The maximum volume of a sphere inscribed in a right triangular prism -/
theorem max_sphere_volume_in_prism (a b h : ℝ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) :
  let r := min (h / 2) (a * b / (a + b + (a^2 + b^2).sqrt))
  (4 / 3) * π * r^3 = (9 * π) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_volume_in_prism_l3821_382134


namespace NUMINAMATH_CALUDE_set_operations_l3821_382130

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5,6,7,8}

-- Define set A
def A : Finset Nat := {5,6,7,8}

-- Define set B
def B : Finset Nat := {2,4,6,8}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {6,8}) ∧
  (U \ A = {1,2,3,4}) ∧
  (U \ B = {1,3,5,7}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3821_382130


namespace NUMINAMATH_CALUDE_parabola_equation_l3821_382164

-- Define the parabola type
structure Parabola where
  -- The equation of the parabola is either y² = ax or x² = by
  a : ℝ
  b : ℝ
  along_x_axis : Bool

-- Define the properties of the parabola
def satisfies_conditions (p : Parabola) : Prop :=
  -- Vertex at origin (implied by the standard form of equation)
  -- Axis of symmetry along one of the coordinate axes (implied by the structure)
  -- Passes through the point (-2, 3)
  (p.along_x_axis ∧ 3^2 = -p.a * (-2)) ∨
  (¬p.along_x_axis ∧ (-2)^2 = p.b * 3)

-- Theorem statement
theorem parabola_equation :
  ∀ p : Parabola, satisfies_conditions p →
    (p.along_x_axis ∧ p.a = -9/2) ∨ (¬p.along_x_axis ∧ p.b = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3821_382164


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3821_382166

theorem inequality_solution_set (x : ℝ) :
  x * (2 * x^2 - 3 * x + 1) ≤ 0 ↔ x ≤ 0 ∨ (1/2 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3821_382166


namespace NUMINAMATH_CALUDE_assignment_plans_count_l3821_382154

/-- The number of students --/
def total_students : ℕ := 6

/-- The number of tasks --/
def total_tasks : ℕ := 4

/-- The number of students to be selected --/
def selected_students : ℕ := 4

/-- The number of students who cannot be assigned to a specific task --/
def restricted_students : ℕ := 2

/-- Calculates the total number of different assignment plans --/
def total_assignment_plans : ℕ := 
  (total_students.factorial / (total_students - selected_students).factorial) - 
  2 * ((total_students - 1).factorial / (total_students - selected_students).factorial)

/-- Theorem stating the total number of different assignment plans --/
theorem assignment_plans_count : total_assignment_plans = 240 := by
  sorry

end NUMINAMATH_CALUDE_assignment_plans_count_l3821_382154


namespace NUMINAMATH_CALUDE_parabola_sum_l3821_382101

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate stating that a point (x, y) is on the parabola -/
def on_parabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Predicate stating that (h, k) is the vertex of the parabola -/
def has_vertex (p : Parabola) (h k : ℝ) : Prop :=
  ∀ x, p.a * (x - h)^2 + k = p.a * x^2 + p.b * x + p.c

/-- The axis of symmetry is vertical when x = h, where (h, k) is the vertex -/
def has_vertical_axis_of_symmetry (p : Parabola) (h : ℝ) : Prop :=
  ∀ x y, on_parabola p x y ↔ on_parabola p (2*h - x) y

theorem parabola_sum (p : Parabola) :
  has_vertex p 4 4 →
  has_vertical_axis_of_symmetry p 4 →
  on_parabola p 3 0 →
  p.a + p.b + p.c = -32 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l3821_382101


namespace NUMINAMATH_CALUDE_camila_weeks_to_match_steven_l3821_382190

-- Define the initial number of hikes for Camila
def camila_initial_hikes : ℕ := 7

-- Define Amanda's hikes in terms of Camila's
def amanda_hikes : ℕ := 8 * camila_initial_hikes

-- Define Steven's hikes in terms of Amanda's
def steven_hikes : ℕ := amanda_hikes + 15

-- Define Camila's planned hikes per week
def camila_weekly_hikes : ℕ := 4

-- Theorem to prove
theorem camila_weeks_to_match_steven :
  (steven_hikes - camila_initial_hikes) / camila_weekly_hikes = 16 := by
  sorry

end NUMINAMATH_CALUDE_camila_weeks_to_match_steven_l3821_382190


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l3821_382171

-- Define the binary number 1101₂
def binary_num : ℕ := 13

-- Define the ternary number 102₃
def ternary_num : ℕ := 11

-- Theorem statement
theorem product_of_binary_and_ternary :
  binary_num * ternary_num = 143 := by sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l3821_382171


namespace NUMINAMATH_CALUDE_tom_fruit_purchase_total_l3821_382181

/-- Calculate the total amount Tom paid for fruits with discount and tax --/
theorem tom_fruit_purchase_total : 
  let apple_cost : ℝ := 8 * 70
  let mango_cost : ℝ := 9 * 90
  let grape_cost : ℝ := 5 * 150
  let total_before_discount : ℝ := apple_cost + mango_cost + grape_cost
  let discount_rate : ℝ := 0.10
  let tax_rate : ℝ := 0.05
  let discounted_amount : ℝ := total_before_discount * (1 - discount_rate)
  let final_amount : ℝ := discounted_amount * (1 + tax_rate)
  final_amount = 2003.4 := by sorry

end NUMINAMATH_CALUDE_tom_fruit_purchase_total_l3821_382181


namespace NUMINAMATH_CALUDE_boy_walking_time_l3821_382178

/-- Given a boy who walks at 6/7 of his usual rate and reaches school 4 minutes early, 
    his usual time to reach the school is 24 minutes. -/
theorem boy_walking_time (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0) (h2 : usual_time > 0) : 
  (6 / 7 * usual_rate) * (usual_time - 4) = usual_rate * usual_time → 
  usual_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_boy_walking_time_l3821_382178


namespace NUMINAMATH_CALUDE_function_properties_l3821_382106

def f (a b c x : ℝ) := a * x^4 + b * x^2 + c

theorem function_properties (a b c : ℝ) :
  f a b c 0 = 1 ∧
  (∀ x, x = 1 → f a b c x + 2 = x) ∧
  f a b c 1 = -1 →
  a = 5/2 ∧ c = 1 ∧
  ∀ x, (- (3 * Real.sqrt 10) / 10 < x ∧ x < 0) ∨ (3 * Real.sqrt 10 / 10 < x) →
    ∀ y, x < y → f a b c x < f a b c y :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3821_382106


namespace NUMINAMATH_CALUDE_sum_greater_than_product_l3821_382163

theorem sum_greater_than_product (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : Real.arctan x + Real.arctan y + Real.arctan z < π) : 
  x + y + z > x * y * z := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_product_l3821_382163


namespace NUMINAMATH_CALUDE_df_length_is_six_l3821_382138

/-- Represents a triangle with side lengths and an angle --/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle : ℝ

/-- The perimeter of a triangle --/
def perimeter (t : Triangle) : ℝ := t.side1 + t.side2 + t.side3

/-- Given two triangles ABC and DEF with the specified properties,
    prove that the length of DF is 6 cm --/
theorem df_length_is_six 
  (ABC : Triangle)
  (DEF : Triangle)
  (angle_relation : ABC.angle = 2 * DEF.angle)
  (ab_length : ABC.side1 = 4)
  (ac_length : ABC.side2 = 6)
  (de_length : DEF.side1 = 2)
  (perimeter_relation : perimeter ABC = 2 * perimeter DEF) :
  DEF.side2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_df_length_is_six_l3821_382138


namespace NUMINAMATH_CALUDE_four_digit_integers_with_one_or_seven_l3821_382122

/-- The number of four-digit positive integers -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit positive integers without 1 or 7 -/
def four_digit_integers_without_one_or_seven : ℕ := 3584

/-- Theorem: The number of four-digit positive integers with at least one digit as 1 or 7 -/
theorem four_digit_integers_with_one_or_seven :
  total_four_digit_integers - four_digit_integers_without_one_or_seven = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_integers_with_one_or_seven_l3821_382122


namespace NUMINAMATH_CALUDE_count_D_eq_3_is_13_l3821_382182

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D_eq_3 : ℕ := sorry

theorem count_D_eq_3_is_13 : count_D_eq_3 = 13 := by sorry

end NUMINAMATH_CALUDE_count_D_eq_3_is_13_l3821_382182


namespace NUMINAMATH_CALUDE_sqrt_neg_one_is_plus_minus_i_l3821_382147

theorem sqrt_neg_one_is_plus_minus_i :
  ∃ (z : ℂ), z * z = -1 ∧ (z = Complex.I ∨ z = -Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_neg_one_is_plus_minus_i_l3821_382147


namespace NUMINAMATH_CALUDE_inconsistent_statistics_l3821_382129

theorem inconsistent_statistics (x_bar m S_squared : ℝ) 
  (h1 : x_bar = 0)
  (h2 : m = 4)
  (h3 : S_squared = 15.917)
  : ¬ (|x_bar - m| ≤ Real.sqrt S_squared) := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_statistics_l3821_382129


namespace NUMINAMATH_CALUDE_problem_statement_l3821_382121

/-- Given that a² + ab = -2 and b² - 3ab = -3, prove that a² + 4ab - b² = 1 -/
theorem problem_statement (a b : ℝ) (h1 : a^2 + a*b = -2) (h2 : b^2 - 3*a*b = -3) :
  a^2 + 4*a*b - b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3821_382121


namespace NUMINAMATH_CALUDE_total_players_is_60_l3821_382107

/-- Represents the total number of players in each sport and their intersections --/
structure SportPlayers where
  cricket : ℕ
  hockey : ℕ
  football : ℕ
  softball : ℕ
  cricket_hockey : ℕ
  cricket_football : ℕ
  cricket_softball : ℕ
  hockey_football : ℕ
  hockey_softball : ℕ
  football_softball : ℕ
  cricket_hockey_football : ℕ

/-- Calculate the total number of unique players given the sport participation data --/
def totalUniquePlayers (sp : SportPlayers) : ℕ :=
  sp.cricket + sp.hockey + sp.football + sp.softball
  - sp.cricket_hockey - sp.cricket_football - sp.cricket_softball
  - sp.hockey_football - sp.hockey_softball - sp.football_softball
  + sp.cricket_hockey_football

/-- The main theorem stating that given the specific sport participation data,
    the total number of unique players is 60 --/
theorem total_players_is_60 (sp : SportPlayers)
  (h1 : sp.cricket = 25)
  (h2 : sp.hockey = 20)
  (h3 : sp.football = 30)
  (h4 : sp.softball = 18)
  (h5 : sp.cricket_hockey = 5)
  (h6 : sp.cricket_football = 8)
  (h7 : sp.cricket_softball = 3)
  (h8 : sp.hockey_football = 4)
  (h9 : sp.hockey_softball = 6)
  (h10 : sp.football_softball = 9)
  (h11 : sp.cricket_hockey_football = 2) :
  totalUniquePlayers sp = 60 := by
  sorry


end NUMINAMATH_CALUDE_total_players_is_60_l3821_382107


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l3821_382196

theorem inserted_numbers_sum (a b : ℝ) : 
  (∃ d : ℝ, a = 4 + d ∧ b = 4 + 2*d) →  -- arithmetic progression condition
  (∃ r : ℝ, b = a*r ∧ 16 = b*r) →       -- geometric progression condition
  a + b = 6*Real.sqrt 3 + 8 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l3821_382196


namespace NUMINAMATH_CALUDE_bread_and_ham_percentage_l3821_382109

def bread_cost : ℚ := 50
def ham_cost : ℚ := 150
def cake_cost : ℚ := 200

theorem bread_and_ham_percentage : 
  (bread_cost + ham_cost) / (bread_cost + ham_cost + cake_cost) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_bread_and_ham_percentage_l3821_382109


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l3821_382157

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l3821_382157


namespace NUMINAMATH_CALUDE_complex_number_problem_l3821_382189

theorem complex_number_problem (z : ℂ) : 
  z + Complex.abs z = 5 + Complex.I * Real.sqrt 3 → z = 11/5 + Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3821_382189


namespace NUMINAMATH_CALUDE_gcd_of_36_45_495_l3821_382146

theorem gcd_of_36_45_495 : Nat.gcd 36 (Nat.gcd 45 495) = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_36_45_495_l3821_382146


namespace NUMINAMATH_CALUDE_max_remaining_pairs_l3821_382145

def original_total_pairs : ℕ := 20
def original_high_heeled_pairs : ℕ := 4
def original_flat_pairs : ℕ := 16
def lost_high_heeled_shoes : ℕ := 5
def lost_flat_shoes : ℕ := 11

def shoes_per_pair : ℕ := 2

theorem max_remaining_pairs : 
  let original_high_heeled_shoes := original_high_heeled_pairs * shoes_per_pair
  let original_flat_shoes := original_flat_pairs * shoes_per_pair
  let remaining_high_heeled_shoes := original_high_heeled_shoes - lost_high_heeled_shoes
  let remaining_flat_shoes := original_flat_shoes - lost_flat_shoes
  let remaining_high_heeled_pairs := remaining_high_heeled_shoes / shoes_per_pair
  let remaining_flat_pairs := remaining_flat_shoes / shoes_per_pair
  remaining_high_heeled_pairs + remaining_flat_pairs = 11 :=
by sorry

end NUMINAMATH_CALUDE_max_remaining_pairs_l3821_382145


namespace NUMINAMATH_CALUDE_jacket_cost_ratio_l3821_382151

theorem jacket_cost_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_rate : ℝ := 4/5
  let cost : ℝ := selling_price * cost_rate
  cost / marked_price = 3/5 := by
sorry

end NUMINAMATH_CALUDE_jacket_cost_ratio_l3821_382151


namespace NUMINAMATH_CALUDE_parking_spaces_on_first_level_l3821_382177

/-- Represents a 4-level parking garage -/
structure ParkingGarage where
  level1 : ℕ
  level2 : ℕ
  level3 : ℕ
  level4 : ℕ

/-- The conditions of the parking garage problem -/
def validParkingGarage (g : ParkingGarage) : Prop :=
  g.level2 = g.level1 + 8 ∧
  g.level3 = g.level2 + 12 ∧
  g.level4 = g.level3 - 9 ∧
  g.level1 + g.level2 + g.level3 + g.level4 = 299 - 100

theorem parking_spaces_on_first_level (g : ParkingGarage) 
  (h : validParkingGarage g) : g.level1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_parking_spaces_on_first_level_l3821_382177


namespace NUMINAMATH_CALUDE_father_son_age_sum_l3821_382199

/-- Represents the ages of a father and son pair -/
structure FatherSonAges where
  father : ℕ
  son : ℕ

/-- The sum of father's and son's ages after a given number of years -/
def ageSum (ages : FatherSonAges) (years : ℕ) : ℕ :=
  ages.father + ages.son + 2 * years

theorem father_son_age_sum :
  ∀ (ages : FatherSonAges),
    ages.father + ages.son = 55 →
    ages.father = 37 →
    ages.son = 18 →
    ageSum ages (ages.father - ages.son) = 93 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_l3821_382199


namespace NUMINAMATH_CALUDE_max_parts_properties_l3821_382167

/-- The maximum number of parts that can be produced from n blanks -/
def max_parts (n : ℕ) : ℕ :=
  let rec aux (blanks remaining : ℕ) : ℕ :=
    if remaining = 0 then blanks
    else
      let new_blanks := remaining / 3
      aux (blanks + remaining) new_blanks
  aux 0 n

theorem max_parts_properties :
  (max_parts 9 = 13) ∧
  (max_parts 14 = 20) ∧
  (max_parts 27 = 40 ∧ ∀ m < 27, max_parts m < 40) := by
  sorry

end NUMINAMATH_CALUDE_max_parts_properties_l3821_382167


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3821_382149

def f (a c x : ℝ) : ℝ := x^2 - a*x + c

theorem quadratic_function_properties (a c : ℝ) :
  (∀ x, f a c x > 1 ↔ x < -1 ∨ x > 3) →
  (∀ x m, m^2 - 4*m < f a c (2^x)) →
  (∀ x₁ x₂, x₁ ∈ [-1, 5] → x₂ ∈ [-1, 5] → |f a c x₁ - f a c x₂| ≤ 10) →
  (a = 2 ∧ c = -2) ∧
  (∀ m, m > 1 ∧ m < 3) ∧
  (a ≥ 10 - 2*Real.sqrt 10 ∧ a ≤ -2 + 2*Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3821_382149


namespace NUMINAMATH_CALUDE_function_symmetry_l3821_382152

/-- Given a polynomial function f(x) = ax^7 + bx^5 + cx^3 + dx + 5 where a, b, c, d are constants,
    if f(-7) = -7, then f(7) = 17 -/
theorem function_symmetry (a b c d : ℝ) :
  let f := fun x : ℝ => a * x^7 + b * x^5 + c * x^3 + d * x + 5
  (f (-7) = -7) → (f 7 = 17) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l3821_382152


namespace NUMINAMATH_CALUDE_impossible_single_piece_on_center_l3821_382116

/-- Represents a square on the solitaire board -/
inductive Square
| One
| Two
| Three

/-- Represents the state of the solitaire board -/
structure BoardState where
  occupied_ones : Nat
  occupied_twos : Nat

/-- Represents a valid move in the solitaire game -/
inductive Move
| HorizontalMove
| VerticalMove

/-- Defines K as the sum of occupied 1-squares and 2-squares -/
def K (state : BoardState) : Nat :=
  state.occupied_ones + state.occupied_twos

/-- The initial state of the board -/
def initial_state : BoardState :=
  { occupied_ones := 15, occupied_twos := 15 }

/-- Applies a move to the board state -/
def apply_move (state : BoardState) (move : Move) : BoardState :=
  sorry

/-- Theorem stating that it's impossible to end with a single piece on the central square -/
theorem impossible_single_piece_on_center :
  ∀ (moves : List Move),
    let final_state := moves.foldl apply_move initial_state
    ¬(K final_state = 1 ∧ final_state.occupied_ones + final_state.occupied_twos = 1) :=
  sorry

end NUMINAMATH_CALUDE_impossible_single_piece_on_center_l3821_382116


namespace NUMINAMATH_CALUDE_problem_solution_l3821_382119

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m / x - 3 / (x^2) - 1

theorem problem_solution :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 → f x₁ < f x₂) ∧
  (∀ x₁ x₂, Real.exp 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ m, (∀ x, 0 < x → 2 * f x ≥ g m x) ↔ m ≤ 4) ∧
  (∀ x, 0 < x → Real.log x < (2 * x / Real.exp 1) - (x^2 / Real.exp x)) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3821_382119
