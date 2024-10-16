import Mathlib

namespace NUMINAMATH_CALUDE_sarah_hair_product_usage_l3022_302229

/-- Calculates the total volume of hair care products used over a given number of days -/
def total_hair_product_usage (shampoo_daily : ℝ) (conditioner_ratio : ℝ) (days : ℕ) : ℝ :=
  let conditioner_daily := shampoo_daily * conditioner_ratio
  let total_daily := shampoo_daily + conditioner_daily
  total_daily * days

/-- Proves that Sarah's total hair product usage over two weeks is 21 ounces -/
theorem sarah_hair_product_usage : 
  total_hair_product_usage 1 0.5 14 = 21 := by
sorry

#eval total_hair_product_usage 1 0.5 14

end NUMINAMATH_CALUDE_sarah_hair_product_usage_l3022_302229


namespace NUMINAMATH_CALUDE_claudia_weekend_earnings_l3022_302220

-- Define the charge per class
def charge_per_class : ℝ := 10.00

-- Define the number of kids in Saturday's class
def saturday_attendance : ℕ := 20

-- Define the number of kids in Sunday's class
def sunday_attendance : ℕ := saturday_attendance / 2

-- Define the total attendance for the weekend
def total_attendance : ℕ := saturday_attendance + sunday_attendance

-- Theorem to prove
theorem claudia_weekend_earnings :
  (total_attendance : ℝ) * charge_per_class = 300.00 := by
  sorry

end NUMINAMATH_CALUDE_claudia_weekend_earnings_l3022_302220


namespace NUMINAMATH_CALUDE_factor_polynomial_l3022_302233

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3022_302233


namespace NUMINAMATH_CALUDE_plane_relation_l3022_302213

-- Define the concept of a plane
class Plane

-- Define the concept of a line
class Line

-- Define the parallelism relation between planes
def parallel (α β : Plane) : Prop := sorry

-- Define the intersection relation between planes
def intersects (α β : Plane) : Prop := sorry

-- Define the relation of a line being parallel to a plane
def line_parallel_to_plane (l : Line) (β : Plane) : Prop := sorry

-- Define the property of having infinitely many parallel lines
def has_infinitely_many_parallel_lines (α β : Plane) : Prop :=
  ∃ (S : Set Line), Set.Infinite S ∧ ∀ l ∈ S, line_parallel_to_plane l β

-- State the theorem
theorem plane_relation (α β : Plane) :
  has_infinitely_many_parallel_lines α β → parallel α β ∨ intersects α β :=
sorry

end NUMINAMATH_CALUDE_plane_relation_l3022_302213


namespace NUMINAMATH_CALUDE_november_to_december_ratio_l3022_302250

/-- Represents the revenue of a toy store in a given month -/
structure Revenue where
  amount : ℝ
  amount_pos : amount > 0

/-- The toy store's revenues for November, December, and January -/
structure StoreRevenue where
  november : Revenue
  december : Revenue
  january : Revenue
  january_is_third_of_november : january.amount = (1/3) * november.amount
  december_is_average_multiple : december.amount = 2.5 * ((november.amount + january.amount) / 2)

/-- The ratio of November's revenue to December's revenue is 3:5 -/
theorem november_to_december_ratio (s : StoreRevenue) :
  s.november.amount / s.december.amount = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_november_to_december_ratio_l3022_302250


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_composite_l3022_302248

theorem quadratic_roots_imply_composite (m n : ℤ) :
  (∃ x₁ x₂ : ℤ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ ∧ 2 * x₁^2 + m * x₁ + 2 - n = 0 ∧ 2 * x₂^2 + m * x₂ + 2 - n = 0) →
  ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ (m^2 + n^2) / 4 = k * l :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_composite_l3022_302248


namespace NUMINAMATH_CALUDE_greatest_multiple_under_1000_l3022_302296

theorem greatest_multiple_under_1000 : 
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 6 = 0 → n ≤ 990 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_under_1000_l3022_302296


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3022_302260

theorem ratio_x_to_y (x y : ℝ) (h : y = 0.25 * x) : x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3022_302260


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l3022_302218

theorem exterior_angle_measure (a b : ℝ) (ha : a = 70) (hb : b = 40) :
  180 - a = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l3022_302218


namespace NUMINAMATH_CALUDE_set_operations_l3022_302253

-- Define the universal set U
def U : Set Int := {-3, -1, 0, 1, 2, 3, 4, 6}

-- Define set A
def A : Set Int := {0, 2, 4, 6}

-- Define the complement of A in U
def C_UA : Set Int := {-1, -3, 1, 3}

-- Define the complement of B in U
def C_UB : Set Int := {-1, 0, 2}

-- Define set B
def B : Set Int := U \ C_UB

-- Theorem to prove
theorem set_operations :
  (A ∩ B = {4, 6}) ∧ (A ∪ B = {-3, 0, 1, 2, 3, 4, 6}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3022_302253


namespace NUMINAMATH_CALUDE_cost_of_roses_shoes_l3022_302292

/-- The cost of Rose's shoes given Mary and Rose's shopping details -/
theorem cost_of_roses_shoes 
  (mary_rose_total : ℝ → ℝ → Prop)  -- Mary and Rose spent the same total amount
  (mary_sunglasses_cost : ℝ)        -- Cost of each pair of Mary's sunglasses
  (mary_sunglasses_quantity : ℕ)    -- Number of pairs of sunglasses Mary bought
  (mary_jeans_cost : ℝ)             -- Cost of Mary's jeans
  (rose_cards_cost : ℝ)             -- Cost of each deck of Rose's basketball cards
  (rose_cards_quantity : ℕ)         -- Number of decks of basketball cards Rose bought
  (h1 : mary_sunglasses_cost = 50)
  (h2 : mary_sunglasses_quantity = 2)
  (h3 : mary_jeans_cost = 100)
  (h4 : rose_cards_cost = 25)
  (h5 : rose_cards_quantity = 2)
  (h6 : mary_rose_total (mary_sunglasses_cost * mary_sunglasses_quantity + mary_jeans_cost) 
                        (rose_cards_cost * rose_cards_quantity + rose_shoes_cost))
  : rose_shoes_cost = 150 := by
  sorry


end NUMINAMATH_CALUDE_cost_of_roses_shoes_l3022_302292


namespace NUMINAMATH_CALUDE_time_to_school_building_l3022_302226

/-- Proves that the time to get from the school gate to the school building is 6 minutes -/
theorem time_to_school_building 
  (total_time : ℕ) 
  (time_to_gate : ℕ) 
  (time_to_room : ℕ) 
  (h1 : total_time = 30) 
  (h2 : time_to_gate = 15) 
  (h3 : time_to_room = 9) : 
  total_time - time_to_gate - time_to_room = 6 := by
  sorry

#check time_to_school_building

end NUMINAMATH_CALUDE_time_to_school_building_l3022_302226


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l3022_302286

def a : ℝ × ℝ := (-1, 1)
def b (m : ℝ) : ℝ × ℝ := (1, m)

theorem vector_dot_product_problem (m : ℝ) : 
  (2 * a - b m) • a = 4 → m = 1 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l3022_302286


namespace NUMINAMATH_CALUDE_cos_270_degrees_l3022_302278

theorem cos_270_degrees : Real.cos (270 * π / 180) = 0 := by sorry

end NUMINAMATH_CALUDE_cos_270_degrees_l3022_302278


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3022_302255

/-- Represents a hyperbola with equation x²/m - y²/3 = 1 -/
structure Hyperbola where
  m : ℝ
  focus : ℝ × ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  h.focus = (2, 0) → eccentricity h = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3022_302255


namespace NUMINAMATH_CALUDE_power_sum_l3022_302273

theorem power_sum (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l3022_302273


namespace NUMINAMATH_CALUDE_situp_rate_difference_l3022_302276

-- Define the given conditions
def diana_rate : ℕ := 4
def diana_situps : ℕ := 40
def total_situps : ℕ := 110

-- Define the theorem
theorem situp_rate_difference : ℕ := by
  -- The difference between Hani's and Diana's situp rates is 3
  sorry

end NUMINAMATH_CALUDE_situp_rate_difference_l3022_302276


namespace NUMINAMATH_CALUDE_train_length_l3022_302295

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 5 → ∃ (length : ℝ), abs (length - 83.35) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3022_302295


namespace NUMINAMATH_CALUDE_optionC_is_most_suitable_l3022_302252

structure SamplingMethod where
  method : String
  representativeOfAllStudents : Bool
  includesAllGrades : Bool
  unbiased : Bool

def cityJuniorHighSchools : Set String := sorry

def isMostSuitableSamplingMethod (m : SamplingMethod) : Prop :=
  m.representativeOfAllStudents ∧ m.includesAllGrades ∧ m.unbiased

def optionC : SamplingMethod := {
  method := "Randomly select 1000 students from each of the three grades in junior high schools in the city",
  representativeOfAllStudents := true,
  includesAllGrades := true,
  unbiased := true
}

theorem optionC_is_most_suitable :
  isMostSuitableSamplingMethod optionC :=
sorry

end NUMINAMATH_CALUDE_optionC_is_most_suitable_l3022_302252


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3022_302210

/-- An ellipse with given properties -/
structure Ellipse where
  /-- The distance between the foci -/
  focal_distance : ℝ
  /-- The distance from the center to the line connecting a focus and the endpoint of the minor axis -/
  center_to_focus_minor_line : ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Theorem: The eccentricity of the ellipse with given properties is √5/3 -/
theorem ellipse_eccentricity (e : Ellipse) 
    (h1 : e.focal_distance = 3) 
    (h2 : e.center_to_focus_minor_line = 1) : 
  eccentricity e = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3022_302210


namespace NUMINAMATH_CALUDE_jerrys_breakfast_theorem_l3022_302244

/-- Calculates the total calories in Jerry's breakfast -/
def jerrys_breakfast_calories : ℕ :=
  let pancake_calories : ℕ := 120
  let bacon_calories : ℕ := 100
  let cereal_calories : ℕ := 200
  let num_pancakes : ℕ := 6
  let num_bacon_strips : ℕ := 2
  let num_cereal_bowls : ℕ := 1
  (pancake_calories * num_pancakes) + (bacon_calories * num_bacon_strips) + (cereal_calories * num_cereal_bowls)

/-- Proves that Jerry's breakfast contains 1120 calories -/
theorem jerrys_breakfast_theorem : jerrys_breakfast_calories = 1120 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_breakfast_theorem_l3022_302244


namespace NUMINAMATH_CALUDE_max_value_of_f_l3022_302269

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt 2 * Real.sin x + Real.cos x) / (Real.sin x + Real.sqrt (1 - Real.sin x))

theorem max_value_of_f :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f x ≤ M) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ f x = M) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3022_302269


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3022_302287

/-- A cubic function with specific properties -/
structure CubicFunction where
  b : ℝ
  c : ℝ
  d : ℝ
  f : ℝ → ℝ
  f_def : ∀ x, f x = x^3 + 3*b*x^2 + c*x + d
  increasing_neg : ∀ x y, x < y → y < 0 → f x < f y
  decreasing_pos : ∀ x y, 0 < x → x < y → y < 2 → f y < f x
  root_neg_b : f (-b) = 0

/-- Main theorem about the cubic function -/
theorem cubic_function_properties (cf : CubicFunction) :
  cf.c = 0 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ -cf.b ∧ x₂ ≠ -cf.b ∧ cf.f x₁ = 0 ∧ cf.f x₂ = 0 ∧ x₂ - (-cf.b) = (-cf.b) - x₁) ∧
  (0 ≤ cf.f 1 ∧ cf.f 1 < 11) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3022_302287


namespace NUMINAMATH_CALUDE_solution_set_inequality1_solution_set_inequality2_l3022_302211

-- Problem 1
def inequality1 (x : ℝ) : Prop := abs (x - 2) + abs (2 * x - 3) < 4

theorem solution_set_inequality1 :
  {x : ℝ | inequality1 x} = {x : ℝ | 1/3 < x ∧ x < 3} :=
by sorry

-- Problem 2
def inequality2 (x : ℝ) : Prop := (x^2 - 3*x) / (x^2 - x - 2) ≤ x

theorem solution_set_inequality2 :
  {x : ℝ | inequality2 x} = 
    {x : ℝ | -1 < x ∧ x ≤ 0} ∪ {1} ∪ {x : ℝ | x > 2} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality1_solution_set_inequality2_l3022_302211


namespace NUMINAMATH_CALUDE_sum_difference_is_4750_l3022_302290

/-- Rounds a number to the nearest multiple of 5, rounding 2.5s up -/
def roundToNearestFive (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

/-- Sums all integers from 1 to n -/
def sumToN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Sums all integers from 1 to n after rounding each to the nearest multiple of 5 -/
def sumRoundedToN (n : ℕ) : ℕ :=
  (n / 5) * (2 * 0 + 3 * 5)

theorem sum_difference_is_4750 :
  sumToN 100 - sumRoundedToN 100 = 4750 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_is_4750_l3022_302290


namespace NUMINAMATH_CALUDE_rectangle_length_l3022_302238

/-- Given a rectangle and a square, prove that the length of the rectangle is 15 cm. -/
theorem rectangle_length (w l : ℝ) (square_side : ℝ) : 
  w = 9 → 
  square_side = 12 → 
  4 * square_side = 2 * w + 2 * l → 
  l = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l3022_302238


namespace NUMINAMATH_CALUDE_morgan_pens_l3022_302257

theorem morgan_pens (total red blue : ℕ) (h1 : total = 168) (h2 : red = 65) (h3 : blue = 45) :
  total - red - blue = 58 := by
  sorry

end NUMINAMATH_CALUDE_morgan_pens_l3022_302257


namespace NUMINAMATH_CALUDE_smallest_result_l3022_302270

def S : Finset Nat := {3, 5, 7, 11, 13, 17}

def process (a b c : Nat) : Nat :=
  max (max ((a + b) * c) ((a + c) * b)) ((b + c) * a)

def valid_selection (a b c : Nat) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∃ (a b c : Nat), valid_selection a b c ∧
    process a b c = 36 ∧
    ∀ (x y z : Nat), valid_selection x y z → process x y z ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_result_l3022_302270


namespace NUMINAMATH_CALUDE_min_value_a_l3022_302265

theorem min_value_a (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, -1 < x₀ ∧ x₀ ≤ 2 ∧ x₀ - a > 0) → 
  (∀ b : ℝ, (¬ ∃ x₀ : ℝ, -1 < x₀ ∧ x₀ ≤ 2 ∧ x₀ - b > 0) → a ≤ b) → 
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l3022_302265


namespace NUMINAMATH_CALUDE_journey_proportions_l3022_302200

theorem journey_proportions 
  (total_distance : ℝ) 
  (rail_proportion bus_proportion : ℝ) 
  (h1 : rail_proportion > 0)
  (h2 : bus_proportion > 0)
  (h3 : rail_proportion + bus_proportion < 1) :
  ∃ (foot_proportion : ℝ),
    foot_proportion > 0 ∧ 
    rail_proportion + bus_proportion + foot_proportion = 1 := by
  sorry

end NUMINAMATH_CALUDE_journey_proportions_l3022_302200


namespace NUMINAMATH_CALUDE_multiply_by_hundred_l3022_302228

theorem multiply_by_hundred (x : ℝ) : x = 15.46 → x * 100 = 1546 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_hundred_l3022_302228


namespace NUMINAMATH_CALUDE_leila_cake_consumption_l3022_302258

def monday_cakes : ℕ := 6
def friday_cakes : ℕ := 9
def saturday_cakes : ℕ := 3 * monday_cakes

theorem leila_cake_consumption : 
  monday_cakes + friday_cakes + saturday_cakes = 33 := by
  sorry

end NUMINAMATH_CALUDE_leila_cake_consumption_l3022_302258


namespace NUMINAMATH_CALUDE_pages_written_first_week_pages_written_first_week_proof_l3022_302268

/-- Calculates the number of pages written in the first week of a 500-page book -/
theorem pages_written_first_week : ℕ :=
  let total_pages : ℕ := 500
  let second_week_write_ratio : ℚ := 30 / 100
  let coffee_damage_ratio : ℚ := 20 / 100
  let remaining_empty_pages : ℕ := 196
  
  -- Define a function to calculate pages written in first week
  let pages_written (x : ℕ) : Prop :=
    let remaining_after_first := total_pages - x
    let remaining_after_second := remaining_after_first - (second_week_write_ratio * remaining_after_first).floor
    let damaged_pages := (coffee_damage_ratio * remaining_after_second).floor
    remaining_after_second - damaged_pages = remaining_empty_pages

  -- The theorem states that 150 satisfies the conditions
  150

/-- Proof of the theorem -/
theorem pages_written_first_week_proof : pages_written_first_week = 150 := by
  sorry

end NUMINAMATH_CALUDE_pages_written_first_week_pages_written_first_week_proof_l3022_302268


namespace NUMINAMATH_CALUDE_prime_between_squares_l3022_302298

/-- A number is a perfect square if it's the square of some integer. -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- A number is prime if it's greater than 1 and its only divisors are 1 and itself. -/
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

theorem prime_between_squares : 
  ∃! p : ℕ, is_prime p ∧ 
    (∃ n : ℕ, is_perfect_square n ∧ p = n + 12) ∧
    (∃ m : ℕ, is_perfect_square m ∧ p + 9 = m) :=
sorry

end NUMINAMATH_CALUDE_prime_between_squares_l3022_302298


namespace NUMINAMATH_CALUDE_total_protest_days_l3022_302294

theorem total_protest_days (first_protest : ℕ) (second_protest_percentage : ℚ) : 
  first_protest = 4 →
  second_protest_percentage = 25 / 100 →
  first_protest + (first_protest + first_protest * second_protest_percentage) = 9 := by
sorry

end NUMINAMATH_CALUDE_total_protest_days_l3022_302294


namespace NUMINAMATH_CALUDE_parabola_equation_l3022_302205

/-- Given a point M(5,3) and a parabola y=ax^2 where the distance from M to the axis of symmetry is 6,
    prove that the equation of the parabola is either y = 1/12 x^2 or y = -1/36 x^2 -/
theorem parabola_equation (a : ℝ) (h : |5 + 1/(4*a)| = 6) :
  a = 1/12 ∨ a = -1/36 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3022_302205


namespace NUMINAMATH_CALUDE_unique_coprime_solution_l3022_302217

theorem unique_coprime_solution (n : ℕ+) :
  ∀ p q : ℤ,
  p > 0 ∧ q > 0 ∧
  Int.gcd p q = 1 ∧
  p + q^2 = (n.val^2 + 1) * p^2 + q →
  p = n.val + 1 ∧ q = n.val^2 + n.val + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_coprime_solution_l3022_302217


namespace NUMINAMATH_CALUDE_jim_gave_away_900_cards_l3022_302206

/-- The number of cards Jim gave away -/
def cards_given_away (initial_cards : ℕ) (set_size : ℕ) (sets_to_brother sets_to_sister sets_to_friend sets_to_cousin sets_to_classmate : ℕ) : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend + sets_to_cousin + sets_to_classmate) * set_size

/-- Proof that Jim gave away 900 cards -/
theorem jim_gave_away_900_cards :
  cards_given_away 1500 25 15 8 4 6 3 = 900 := by
  sorry

end NUMINAMATH_CALUDE_jim_gave_away_900_cards_l3022_302206


namespace NUMINAMATH_CALUDE_theodore_wooden_statues_l3022_302232

/-- Theodore's monthly statue production and earnings --/
structure StatueProduction where
  stone_statues : ℕ
  wooden_statues : ℕ
  stone_price : ℚ
  wooden_price : ℚ
  tax_rate : ℚ
  total_earnings_after_tax : ℚ

/-- Theorem: Theodore crafts 20 wooden statues per month --/
theorem theodore_wooden_statues (p : StatueProduction) 
  (h1 : p.stone_statues = 10)
  (h2 : p.stone_price = 20)
  (h3 : p.wooden_price = 5)
  (h4 : p.tax_rate = 1/10)
  (h5 : p.total_earnings_after_tax = 270) :
  p.wooden_statues = 20 := by
  sorry

#check theodore_wooden_statues

end NUMINAMATH_CALUDE_theodore_wooden_statues_l3022_302232


namespace NUMINAMATH_CALUDE_platform_length_platform_length_proof_l3022_302225

/-- Calculates the length of a platform given train parameters -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time
  total_distance - train_length

/-- Proves that the platform length is 50 meters given the specific conditions -/
theorem platform_length_proof : 
  platform_length 250 72 15 = 50 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_proof_l3022_302225


namespace NUMINAMATH_CALUDE_class_fraction_problem_l3022_302282

theorem class_fraction_problem (G : ℕ) (B : ℕ) :
  B = (5 * G) / 3 →
  (2 * G) / 3 = (1 / 4) * (B + G) :=
by sorry

end NUMINAMATH_CALUDE_class_fraction_problem_l3022_302282


namespace NUMINAMATH_CALUDE_equation_graph_is_two_lines_l3022_302201

/-- The set of points (x, y) satisfying the equation (x + y - 1)^2 = x^2 + y^2 - 1 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + p.2 - 1)^2 = p.1^2 + p.2^2 - 1}

/-- The union of the lines x = 1 and y = 1 -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 1 ∨ p.2 = 1}

/-- Theorem stating that the sets S and T are equal -/
theorem equation_graph_is_two_lines : S = T := by
  sorry

end NUMINAMATH_CALUDE_equation_graph_is_two_lines_l3022_302201


namespace NUMINAMATH_CALUDE_list_property_l3022_302279

theorem list_property (S : ℝ) (n : ℝ) (list_size : ℕ) (h1 : list_size = 21) 
  (h2 : n = 4 * ((S - n) / (list_size - 1))) 
  (h3 : n = S / 6) : 
  list_size - 1 = 20 := by
sorry

end NUMINAMATH_CALUDE_list_property_l3022_302279


namespace NUMINAMATH_CALUDE_lion_meeting_day_l3022_302246

/-- Represents days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the day after a given day -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

/-- Returns true if the lion lies on the given day according to his pattern -/
def lionLies (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Friday ∨ d = Day.Saturday

/-- The day Alice met the lion -/
def meetingDay : Day := Day.Monday

theorem lion_meeting_day :
  (lionLies (nextDay (nextDay meetingDay)) ∧
   lionLies (nextDay (nextDay (nextDay meetingDay))) ∧
   ¬lionLies (nextDay meetingDay)) ∧
  ¬(lionLies (nextDay (nextDay (nextDay meetingDay))) ∧
    lionLies (nextDay (nextDay (nextDay (nextDay meetingDay)))) ∧
    ¬lionLies (nextDay (nextDay (nextDay (nextDay (nextDay meetingDay))))) ∧
    meetingDay ≠ Day.Monday) :=
by sorry


end NUMINAMATH_CALUDE_lion_meeting_day_l3022_302246


namespace NUMINAMATH_CALUDE_infinite_power_tower_equals_four_l3022_302299

-- Define the infinite power tower function
noncomputable def powerTower (x : ℝ) : ℝ := Real.sqrt (4 : ℝ)

-- State the theorem
theorem infinite_power_tower_equals_four (x : ℝ) (h₁ : x > 0) :
  powerTower x = 4 → x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_power_tower_equals_four_l3022_302299


namespace NUMINAMATH_CALUDE_tangent_sine_equality_l3022_302256

open Real

theorem tangent_sine_equality (α : ℝ) :
  (∃ k : ℤ, -π/2 + 2*π*(k : ℝ) < α ∧ α < π/2 + 2*π*(k : ℝ)) ↔
  Real.sqrt ((tan α)^2 - (sin α)^2) = tan α * sin α :=
sorry

end NUMINAMATH_CALUDE_tangent_sine_equality_l3022_302256


namespace NUMINAMATH_CALUDE_window_purchase_savings_l3022_302209

/-- Calculates the cost of windows given the number of windows and the discount rule -/
def calculateCost (windowCount : ℕ) (windowPrice : ℕ) : ℕ :=
  (windowCount - windowCount / 3) * windowPrice

/-- Represents the window purchase scenario -/
theorem window_purchase_savings
  (windowPrice : ℕ)
  (daveWindowCount : ℕ)
  (dougWindowCount : ℕ)
  (h1 : windowPrice = 100)
  (h2 : daveWindowCount = 10)
  (h3 : dougWindowCount = 12) :
  calculateCost (daveWindowCount + dougWindowCount) windowPrice =
  calculateCost daveWindowCount windowPrice + calculateCost dougWindowCount windowPrice :=
by sorry

#eval calculateCost 22 100 -- Joint purchase
#eval calculateCost 10 100 + calculateCost 12 100 -- Separate purchases

end NUMINAMATH_CALUDE_window_purchase_savings_l3022_302209


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3022_302237

theorem smaller_number_proof (x y : ℝ) : 
  x - y = 9 → x + y = 46 → min x y = 18.5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3022_302237


namespace NUMINAMATH_CALUDE_democrat_ratio_l3022_302291

/-- Represents the number of participants in a meeting with democrats -/
structure Meeting where
  total : ℕ
  female : ℕ
  male : ℕ
  femaleDemocrats : ℕ
  maleDemocrats : ℕ

/-- The properties of the meeting as described in the problem -/
def meetingProperties (m : Meeting) : Prop :=
  m.total = 750 ∧
  m.female + m.male = m.total ∧
  m.femaleDemocrats = m.female / 2 ∧
  m.maleDemocrats = m.male / 4 ∧
  m.femaleDemocrats = 125

/-- The theorem stating that the ratio of democrats to total participants is 1:3 -/
theorem democrat_ratio (m : Meeting) (h : meetingProperties m) :
  (m.femaleDemocrats + m.maleDemocrats) * 3 = m.total := by
  sorry


end NUMINAMATH_CALUDE_democrat_ratio_l3022_302291


namespace NUMINAMATH_CALUDE_stone_splitting_properties_l3022_302224

/-- Represents the state of stone piles -/
structure PileState :=
  (piles : List Nat)
  (valid : piles.sum = 100)

/-- Represents a single move in the stone-splitting process -/
def split_move (s : PileState) : PileState → Prop :=
  sorry

/-- Represents the complete process of splitting stones -/
def splitting_process (initial : PileState) (final : PileState) : Prop :=
  sorry

theorem stone_splitting_properties 
  (initial : PileState)
  (final : PileState)
  (h_initial : initial.piles = [100])
  (h_final : final.piles.all (· = 1) ∧ final.piles.length = 100)
  (h_process : splitting_process initial final) :
  (∃ s : PileState, splitting_process initial s ∧ 
    (∃ sub : List Nat, sub ⊆ s.piles ∧ sub.length = 30 ∧ sub.sum = 60)) ∧
  (∃ s : PileState, splitting_process initial s ∧ 
    (∃ sub : List Nat, sub ⊆ s.piles ∧ sub.length = 20 ∧ sub.sum = 60)) ∧
  (∃ f : PileState → PileState, 
    splitting_process initial (f final) ∧
    ∀ s, splitting_process initial s → splitting_process s (f final) →
      ¬∃ sub : List Nat, sub ⊆ s.piles ∧ sub.length = 19 ∧ sub.sum = 60) :=
sorry

end NUMINAMATH_CALUDE_stone_splitting_properties_l3022_302224


namespace NUMINAMATH_CALUDE_olivias_albums_l3022_302249

def number_of_albums (pictures_from_phone : ℕ) (pictures_from_camera : ℕ) (pictures_per_album : ℕ) : ℕ :=
  (pictures_from_phone + pictures_from_camera) / pictures_per_album

theorem olivias_albums :
  let pictures_from_phone : ℕ := 5
  let pictures_from_camera : ℕ := 35
  let pictures_per_album : ℕ := 5
  number_of_albums pictures_from_phone pictures_from_camera pictures_per_album = 8 := by
  sorry

end NUMINAMATH_CALUDE_olivias_albums_l3022_302249


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3022_302264

theorem inequality_equivalence (x : ℝ) : 4 * x - 1 < 0 ↔ x < 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3022_302264


namespace NUMINAMATH_CALUDE_largest_fourth_number_l3022_302251

/-- Represents a two-digit number -/
def TwoDigitNumber := {n : ℕ // 10 ≤ n ∧ n < 100}

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The problem setup -/
def fourthNumberProblem (a b c d : TwoDigitNumber) : Prop :=
  a.val = 34 ∧ b.val = 21 ∧ c.val = 65 ∧ 
  (∃ (x : ℕ), d.val = 40 + x ∧ x < 10) ∧
  4 * (sumOfDigits a.val + sumOfDigits b.val + sumOfDigits c.val + sumOfDigits d.val) = 
    a.val + b.val + c.val + d.val

/-- The theorem to be proved -/
theorem largest_fourth_number : 
  ∀ (a b c d : TwoDigitNumber), 
    fourthNumberProblem a b c d → d.val ≤ 49 := by sorry

end NUMINAMATH_CALUDE_largest_fourth_number_l3022_302251


namespace NUMINAMATH_CALUDE_line_point_k_value_l3022_302266

/-- Given a line containing points (3,5), (-1,k), and (-7,2), prove that k = 3.8 -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (line : ℝ → ℝ), line 3 = 5 ∧ line (-1) = k ∧ line (-7) = 2) → k = 3.8 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_value_l3022_302266


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l3022_302222

theorem arithmetic_series_sum (k : ℕ) : 
  let a₁ : ℕ := k^2 + k + 1
  let d : ℕ := 1
  let n : ℕ := 2*k + 3
  let S := n * (2*a₁ + (n-1)*d) / 2
  S = 2*k^3 + 7*k^2 + 10*k + 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l3022_302222


namespace NUMINAMATH_CALUDE_quadratic_b_value_l3022_302207

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 - b * x - 1

-- Define the property of passing through two points with the same y-coordinate
def passes_through (b : ℝ) : Prop :=
  ∃ y₀ : ℝ, f b 3 = y₀ ∧ f b 9 = y₀

-- Theorem statement
theorem quadratic_b_value :
  ∀ b : ℝ, passes_through b → b = 24 := by sorry

end NUMINAMATH_CALUDE_quadratic_b_value_l3022_302207


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l3022_302223

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l3022_302223


namespace NUMINAMATH_CALUDE_no_zero_points_when_k_is_one_exactly_one_zero_point_when_k_is_negative_exists_k_with_two_zero_points_l3022_302297

-- Define the piecewise function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - k * x else k * x^2 - x + 1

-- Statement 1: When k = 1, f(x) has no zero points
theorem no_zero_points_when_k_is_one :
  ∀ x : ℝ, f 1 x ≠ 0 := by sorry

-- Statement 2: When k < 0, f(x) has exactly one zero point
theorem exactly_one_zero_point_when_k_is_negative :
  ∀ k : ℝ, k < 0 → ∃! x : ℝ, f k x = 0 := by sorry

-- Statement 3: There exists a k such that f(x) has two zero points
theorem exists_k_with_two_zero_points :
  ∃ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ f k x = 0 ∧ f k y = 0 := by sorry

end NUMINAMATH_CALUDE_no_zero_points_when_k_is_one_exactly_one_zero_point_when_k_is_negative_exists_k_with_two_zero_points_l3022_302297


namespace NUMINAMATH_CALUDE_river_lengths_theorem_l3022_302231

/-- The lengths of the Danube, Dnieper, and Don rivers satisfy the given conditions -/
theorem river_lengths_theorem (danube dnieper don : ℝ) : 
  (dnieper / danube = 5 / (19 / 3)) →
  (don / danube = 6.5 / 9.5) →
  (dnieper - don = 300) →
  (danube = 2850 ∧ dnieper = 2250 ∧ don = 1950) :=
by sorry

end NUMINAMATH_CALUDE_river_lengths_theorem_l3022_302231


namespace NUMINAMATH_CALUDE_quadratic_solution_l3022_302277

theorem quadratic_solution : ∃ x : ℚ, x > 0 ∧ 5 * x^2 + 9 * x - 18 = 0 ∧ x = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3022_302277


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l3022_302212

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * I) = (1 / 5 : ℂ) + (2 * Real.sqrt 6 / 5 : ℂ) * I ∧
  Complex.exp (φ * I) = (-5 / 13 : ℂ) - (12 / 13 : ℂ) * I →
  Real.sin (θ + φ) = -(12 - 10 * Real.sqrt 6) / 65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l3022_302212


namespace NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l3022_302280

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem statement
theorem root_in_interval_implies_m_range :
  ∀ m : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ f x + m = 0) → -2 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l3022_302280


namespace NUMINAMATH_CALUDE_absolute_value_fraction_l3022_302235

theorem absolute_value_fraction (i : ℂ) : i * i = -1 → Complex.abs ((3 - i) / (i + 2)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_l3022_302235


namespace NUMINAMATH_CALUDE_roses_apples_l3022_302234

/-- Rose's apple distribution problem -/
theorem roses_apples (num_friends : ℕ) (apples_per_friend : ℕ) : 
  num_friends = 3 → apples_per_friend = 3 → num_friends * apples_per_friend = 9 :=
by sorry

end NUMINAMATH_CALUDE_roses_apples_l3022_302234


namespace NUMINAMATH_CALUDE_x_in_interval_l3022_302236

theorem x_in_interval (x : ℝ) (hx : x ≠ 0) : x = 2 * (1 / x) * (-x) → -4 < x ∧ x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_x_in_interval_l3022_302236


namespace NUMINAMATH_CALUDE_arccos_difference_equals_negative_pi_over_six_l3022_302243

theorem arccos_difference_equals_negative_pi_over_six :
  Real.arccos ((Real.sqrt 6 + 1) / (2 * Real.sqrt 3)) - Real.arccos (Real.sqrt (2/3)) = -π/6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_difference_equals_negative_pi_over_six_l3022_302243


namespace NUMINAMATH_CALUDE_percentage_of_800_l3022_302261

theorem percentage_of_800 : (25 / 100) * 800 = 200 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_800_l3022_302261


namespace NUMINAMATH_CALUDE_sqrt_calculations_l3022_302227

theorem sqrt_calculations :
  (∃ (x y : ℝ), x = Real.sqrt 3 ∧ y = Real.sqrt 2 ∧
    x * y - Real.sqrt 12 / Real.sqrt 8 = Real.sqrt 6 / 2) ∧
  ((Real.sqrt 2 - 3)^2 - Real.sqrt 2^2 - Real.sqrt (2^2) - Real.sqrt 2 = 7 - 7 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l3022_302227


namespace NUMINAMATH_CALUDE_second_number_value_l3022_302221

theorem second_number_value (x y z : ℚ) : 
  x + y + z = 120 ∧ 
  x / y = 3 / 4 ∧ 
  y / z = 4 / 7 →
  y = 240 / 7 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l3022_302221


namespace NUMINAMATH_CALUDE_race_graph_representation_l3022_302271

-- Define the types of contestants
inductive Contestant
| Snail
| Horse

-- Define the movement pattern
structure MovementPattern where
  contestant : Contestant
  isConsistent : Bool
  hasRest : Bool
  initialSpeed : ℕ
  finalPosition : ℕ

-- Define the graph types
inductive GraphType
| FluctuatingSpeed
| SteadySlowWinnerVsFastStartStop
| ConsistentlyIncreasing

-- Define the race outcome
def raceOutcome (snailPattern : MovementPattern) (horsePattern : MovementPattern) : GraphType :=
  if snailPattern.isConsistent ∧ 
     snailPattern.initialSpeed < horsePattern.initialSpeed ∧ 
     horsePattern.hasRest ∧ 
     snailPattern.finalPosition > horsePattern.finalPosition
  then GraphType.SteadySlowWinnerVsFastStartStop
  else GraphType.FluctuatingSpeed

-- Theorem statement
theorem race_graph_representation 
  (snail : MovementPattern) 
  (horse : MovementPattern) 
  (h_snail_contestant : snail.contestant = Contestant.Snail)
  (h_horse_contestant : horse.contestant = Contestant.Horse)
  (h_snail_consistent : snail.isConsistent = true)
  (h_snail_slow : snail.initialSpeed < horse.initialSpeed)
  (h_horse_rest : horse.hasRest = true)
  (h_snail_wins : snail.finalPosition > horse.finalPosition) :
  raceOutcome snail horse = GraphType.SteadySlowWinnerVsFastStartStop :=
by sorry

end NUMINAMATH_CALUDE_race_graph_representation_l3022_302271


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3022_302203

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + 2*x + 3 > 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3022_302203


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3022_302204

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculate the distance from a point to a line segment -/
def distanceToLineSegment (p : Point) (a : Point) (b : Point) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem parallelogram_vertex_sum (ABCD : Parallelogram) : 
  ABCD.A = Point.mk (-1) 2 →
  ABCD.B = Point.mk 3 (-6) →
  ABCD.D = Point.mk 7 0 →
  distanceToLineSegment ABCD.C ABCD.A ABCD.D = 3 * distanceToLineSegment ABCD.B ABCD.A ABCD.D →
  ABCD.C.x + ABCD.C.y = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3022_302204


namespace NUMINAMATH_CALUDE_third_bed_theorem_l3022_302239

/-- Represents the number of carrots harvested from each bed and the total harvest weight -/
structure CarrotHarvest where
  first_bed : ℕ
  second_bed : ℕ
  total_weight : ℕ
  carrots_per_pound : ℕ

/-- Calculates the number of carrots in the third bed given the harvest information -/
def third_bed_carrots (harvest : CarrotHarvest) : ℕ :=
  harvest.total_weight * harvest.carrots_per_pound - (harvest.first_bed + harvest.second_bed)

/-- Theorem stating that given the specific harvest conditions, the third bed contains 78 carrots -/
theorem third_bed_theorem (harvest : CarrotHarvest)
  (h1 : harvest.first_bed = 55)
  (h2 : harvest.second_bed = 101)
  (h3 : harvest.total_weight = 39)
  (h4 : harvest.carrots_per_pound = 6) :
  third_bed_carrots harvest = 78 := by
  sorry

#eval third_bed_carrots { first_bed := 55, second_bed := 101, total_weight := 39, carrots_per_pound := 6 }

end NUMINAMATH_CALUDE_third_bed_theorem_l3022_302239


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l3022_302247

theorem quadratic_equation_k_value :
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := 3
  let k : ℝ := 16/3
  (4 * b^2 - k * a * c = 0) ∧
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 4 * b * x + c = 0 ∧ a * y^2 + 4 * b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l3022_302247


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3022_302230

theorem fraction_to_decimal : (47 : ℚ) / (2 * 5^3) = 0.188 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3022_302230


namespace NUMINAMATH_CALUDE_athletes_same_first_digit_know_each_other_l3022_302242

/-- Represents an athlete with an assigned number -/
structure Athlete where
  id : Nat
  number : Nat

/-- Represents the relation of two athletes knowing each other -/
def knows (a b : Athlete) : Prop := sorry

/-- Returns the first digit of a natural number -/
def firstDigit (n : Nat) : Nat := sorry

/-- Theorem: Given 19100 athletes, where among any 12 athletes at least 2 know each other,
    there exist 2 athletes who know each other and whose assigned numbers start with the same digit -/
theorem athletes_same_first_digit_know_each_other 
  (athletes : Finset Athlete) 
  (h1 : athletes.card = 19100) 
  (h2 : ∀ s : Finset Athlete, s ⊆ athletes → s.card = 12 → 
        ∃ a b : Athlete, a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ knows a b) :
  ∃ a b : Athlete, a ∈ athletes ∧ b ∈ athletes ∧ a ≠ b ∧ 
    knows a b ∧ firstDigit a.number = firstDigit b.number := by
  sorry

end NUMINAMATH_CALUDE_athletes_same_first_digit_know_each_other_l3022_302242


namespace NUMINAMATH_CALUDE_n_solution_approx_l3022_302245

def n_equation (n : ℝ) : Prop :=
  (n + 2 * 1.5) ^ 5 = (1 + 3 * 1.5) ^ 4

theorem n_solution_approx : ∃ n : ℝ, n_equation n ∧ abs (n - 0.72) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_n_solution_approx_l3022_302245


namespace NUMINAMATH_CALUDE_existence_of_n_l3022_302272

theorem existence_of_n (p a k : ℕ) (h_prime : Nat.Prime p) (h_pos_a : a > 0) (h_pos_k : k > 0)
  (h_lower : p^a < k) (h_upper : k < 2*p^a) :
  ∃ n : ℕ, n < p^(2*a) ∧ (Nat.choose n k : ZMod (p^a)) = n ∧ (n : ZMod (p^a)) = k := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l3022_302272


namespace NUMINAMATH_CALUDE_earth_habitable_surface_l3022_302284

theorem earth_habitable_surface (total_surface : ℝ) 
  (h1 : total_surface > 0)
  (land_fraction : ℝ) 
  (h2 : land_fraction = 1 / 3)
  (habitable_land_fraction : ℝ) 
  (h3 : habitable_land_fraction = 2 / 3) : 
  (land_fraction * habitable_land_fraction) * total_surface = (2 / 9) * total_surface :=
by sorry

end NUMINAMATH_CALUDE_earth_habitable_surface_l3022_302284


namespace NUMINAMATH_CALUDE_sqrt_5_greater_than_2_l3022_302281

theorem sqrt_5_greater_than_2 : Real.sqrt 5 > 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_greater_than_2_l3022_302281


namespace NUMINAMATH_CALUDE_curve_classification_l3022_302293

-- Define the curve equation
def curve_equation (x y m : ℝ) : Prop := 3 * x^2 + m * y^2 = 1

-- Define the possible curve types
inductive CurveType
  | TwoLines
  | Ellipse
  | Circle
  | Hyperbola

-- Theorem statement
theorem curve_classification (m : ℝ) : 
  ∃ (t : CurveType), ∀ (x y : ℝ), curve_equation x y m → 
    (t = CurveType.TwoLines ∨ 
     t = CurveType.Ellipse ∨ 
     t = CurveType.Circle ∨ 
     t = CurveType.Hyperbola) :=
sorry

end NUMINAMATH_CALUDE_curve_classification_l3022_302293


namespace NUMINAMATH_CALUDE_store_inventory_difference_l3022_302215

theorem store_inventory_difference : 
  ∀ (apples regular_soda diet_soda : ℕ),
    apples = 36 →
    regular_soda = 80 →
    diet_soda = 54 →
    regular_soda + diet_soda - apples = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_store_inventory_difference_l3022_302215


namespace NUMINAMATH_CALUDE_expression_evaluation_l3022_302240

theorem expression_evaluation : (35 * 100) / (0.07 * 100) = 500 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3022_302240


namespace NUMINAMATH_CALUDE_dilution_calculation_l3022_302289

/-- Calculates the amount of water needed to dilute a shaving lotion to a desired alcohol concentration -/
theorem dilution_calculation (initial_volume : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 12 →
  initial_concentration = 0.6 →
  final_concentration = 0.45 →
  ∃ (water_volume : ℝ),
    water_volume = 4 ∧
    (initial_volume * initial_concentration) / (initial_volume + water_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_dilution_calculation_l3022_302289


namespace NUMINAMATH_CALUDE_least_upper_bound_inequality_inequality_holds_l3022_302285

theorem least_upper_bound_inequality (a b c : ℝ) : 
  let M : ℝ := (9 * Real.sqrt 2) / 32
  ∀ N : ℝ, (∀ x y z : ℝ, 
    |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N*(x^2 + y^2 + z^2)^2) →
  N ≥ M :=
by sorry

theorem inequality_holds (a b c : ℝ) : 
  let M : ℝ := (9 * Real.sqrt 2) / 32
  |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M*(a^2 + b^2 + c^2)^2 :=
by sorry

end NUMINAMATH_CALUDE_least_upper_bound_inequality_inequality_holds_l3022_302285


namespace NUMINAMATH_CALUDE_stadium_length_feet_l3022_302274

/-- Converts yards to feet -/
def yards_to_feet (yards : ℕ) : ℕ := yards * 3

/-- The length of the sports stadium in yards -/
def stadium_length_yards : ℕ := 80

theorem stadium_length_feet :
  yards_to_feet stadium_length_yards = 240 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_feet_l3022_302274


namespace NUMINAMATH_CALUDE_bullets_shot_l3022_302202

theorem bullets_shot (n : ℕ) (initial_bullets : ℕ) (total_guys : ℕ) 
  (h1 : total_guys = 5)
  (h2 : initial_bullets = 25)
  (h3 : n ≤ initial_bullets)
  (h4 : (total_guys * initial_bullets) - (total_guys * n) = initial_bullets) :
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_bullets_shot_l3022_302202


namespace NUMINAMATH_CALUDE_number_equation_proof_l3022_302219

theorem number_equation_proof : 
  ∃ x : ℝ, (3034 - (x / 20.04) = 2984) ∧ (x = 1002) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l3022_302219


namespace NUMINAMATH_CALUDE_fraction_difference_l3022_302263

theorem fraction_difference (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : m^2 - n^2 = m*n) : 
  n/m - m/n = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l3022_302263


namespace NUMINAMATH_CALUDE_triple_hash_100_l3022_302267

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 3

-- State the theorem
theorem triple_hash_100 : hash (hash (hash 100)) = 11.08 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_100_l3022_302267


namespace NUMINAMATH_CALUDE_triangle_angle_and_side_relations_l3022_302288

open Real

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sine_rule : a / sin A = b / sin B ∧ b / sin B = c / sin C
  cosine_rule : a^2 + b^2 - c^2 = 2 * a * b * cos C

theorem triangle_angle_and_side_relations (t : Triangle) 
  (h : (t.a^2 + t.b^2 - t.c^2) * sin t.C = Real.sqrt 3 * t.a * t.b * cos t.C) :
  t.C = π/3 ∧ (t.c = Real.sqrt 3 → -3 < t.b - 2*t.a ∧ t.b - 2*t.a < 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_and_side_relations_l3022_302288


namespace NUMINAMATH_CALUDE_earth_total_area_l3022_302254

/-- The ocean area on Earth's surface in million square kilometers -/
def ocean_area : ℝ := 361

/-- The difference between ocean and land area in million square kilometers -/
def area_difference : ℝ := 2.12

/-- The total area of the Earth in million square kilometers -/
def total_area : ℝ := ocean_area + (ocean_area - area_difference)

theorem earth_total_area :
  total_area = 5.10 := by
  sorry

end NUMINAMATH_CALUDE_earth_total_area_l3022_302254


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3022_302208

/-- Definition of a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point -/
def given_point : Point2D :=
  { x := 3, y := -4 }

/-- Theorem: The given point lies in the fourth quadrant -/
theorem point_in_fourth_quadrant :
  fourth_quadrant given_point := by
  sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3022_302208


namespace NUMINAMATH_CALUDE_root_sum_problem_l3022_302275

theorem root_sum_problem (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → 
  b^2 - 5*b + 6 = 0 → 
  a^3 + a^4*b^2 + a^2*b^4 + b^3 + a*b*(a+b) = 533 := by
sorry

end NUMINAMATH_CALUDE_root_sum_problem_l3022_302275


namespace NUMINAMATH_CALUDE_sum_of_ages_l3022_302214

-- Define Rose's age
def rose_age : ℕ := 25

-- Define Rose's mother's age
def mother_age : ℕ := 75

-- Theorem: The sum of Rose's age and her mother's age is 100
theorem sum_of_ages : rose_age + mother_age = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3022_302214


namespace NUMINAMATH_CALUDE_sparrow_distribution_l3022_302259

theorem sparrow_distribution (total : ℕ) (moved : ℕ) (flew_away : ℕ) :
  total = 25 →
  moved = 5 →
  flew_away = 7 →
  (∃ (first second : ℕ),
    first + second = total ∧
    first - moved = 2 * (second + moved - flew_away) ∧
    first = 17 ∧
    second = 8) :=
by sorry

end NUMINAMATH_CALUDE_sparrow_distribution_l3022_302259


namespace NUMINAMATH_CALUDE_correct_dot_counts_l3022_302216

/-- Represents a single die face -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the four visible faces of the dice configuration -/
structure VisibleFaces :=
  (A : DieFace)
  (B : DieFace)
  (C : DieFace)
  (D : DieFace)

/-- Counts the number of dots on a die face -/
def dotCount (face : DieFace) : Nat :=
  match face with
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

/-- The configuration of dice as described in the problem -/
def diceConfiguration : VisibleFaces :=
  { A := DieFace.three
  , B := DieFace.five
  , C := DieFace.six
  , D := DieFace.five }

/-- Theorem stating the correct number of dots on each visible face -/
theorem correct_dot_counts :
  dotCount diceConfiguration.A = 3 ∧
  dotCount diceConfiguration.B = 5 ∧
  dotCount diceConfiguration.C = 6 ∧
  dotCount diceConfiguration.D = 5 :=
sorry

end NUMINAMATH_CALUDE_correct_dot_counts_l3022_302216


namespace NUMINAMATH_CALUDE_A_power_98_l3022_302262

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_98 : A ^ 98 = !![0, 0, 0; 0, -1, 0; 0, 0, -1] := by sorry

end NUMINAMATH_CALUDE_A_power_98_l3022_302262


namespace NUMINAMATH_CALUDE_prob_three_heads_in_eight_tosses_l3022_302283

/-- A fair coin is tossed 8 times. -/
def num_tosses : ℕ := 8

/-- The number of heads we're looking for. -/
def target_heads : ℕ := 3

/-- The probability of getting heads on a single toss of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The probability of getting exactly 'target_heads' heads in 'num_tosses' tosses of a fair coin. -/
def probability_exact_heads : ℚ :=
  (Nat.choose num_tosses target_heads : ℚ) * prob_heads^target_heads * (1 - prob_heads)^(num_tosses - target_heads)

/-- Theorem stating that the probability of getting exactly 3 heads in 8 tosses of a fair coin is 7/32. -/
theorem prob_three_heads_in_eight_tosses : probability_exact_heads = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_heads_in_eight_tosses_l3022_302283


namespace NUMINAMATH_CALUDE_sequence_with_special_sums_l3022_302241

theorem sequence_with_special_sums : ∃ (seq : Fin 20 → ℝ),
  (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧
  (Finset.sum Finset.univ seq < 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_special_sums_l3022_302241
