import Mathlib

namespace NUMINAMATH_CALUDE_max_value_implies_A_l666_66648

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x, 1)

noncomputable def n (A x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * A * Real.cos x, A / 2 * Real.cos (2 * x))

noncomputable def f (A x : ℝ) : ℝ := (m x).1 * (n A x).1 + (m x).2 * (n A x).2

theorem max_value_implies_A (A : ℝ) (h1 : A > 0) (h2 : ∀ x, f A x ≤ 6) (h3 : ∃ x, f A x = 6) : A = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_A_l666_66648


namespace NUMINAMATH_CALUDE_exists_k_for_A_l666_66627

theorem exists_k_for_A (M : ℕ) (hM : M > 2) :
  ∃ k : ℕ, ((M + Real.sqrt (M^2 - 4 : ℝ)) / 2)^5 = (k + Real.sqrt (k^2 - 4 : ℝ)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_exists_k_for_A_l666_66627


namespace NUMINAMATH_CALUDE_hyperbola_sequence_fixed_point_l666_66693

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 2

/-- Definition of the line with slope 2 passing through a point -/
def line_slope_2 (x₀ y₀ x y : ℝ) : Prop := y - y₀ = 2 * (x - x₀)

/-- Definition of the next point in the sequence -/
def next_point (x₀ x₁ : ℝ) : Prop :=
  ∃ y₁, hyperbola x₁ y₁ ∧ line_slope_2 x₀ 0 x₁ y₁

/-- Definition of the sequence of points -/
def point_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n, next_point (x n) (x (n+1)) ∨ x n = 0

/-- The main theorem -/
theorem hyperbola_sequence_fixed_point :
  ∃! k : ℕ, k = (2^2048 - 2) ∧
  ∃ x : ℕ → ℝ, point_sequence x ∧ x 0 = x 2048 ∧ x 0 ≠ 0 ∧
  ∀ y : ℕ → ℝ, point_sequence y ∧ y 0 = y 2048 ∧ y 0 ≠ 0 →
    ∃! i : ℕ, i < k ∧ x 0 = y 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_sequence_fixed_point_l666_66693


namespace NUMINAMATH_CALUDE_negative_x_squared_to_fourth_negative_x_squared_y_cubed_l666_66670

-- Problem 1
theorem negative_x_squared_to_fourth (x : ℝ) : (-x^2)^4 = x^8 := by sorry

-- Problem 2
theorem negative_x_squared_y_cubed (x y : ℝ) : (-x^2*y)^3 = -x^6*y^3 := by sorry

end NUMINAMATH_CALUDE_negative_x_squared_to_fourth_negative_x_squared_y_cubed_l666_66670


namespace NUMINAMATH_CALUDE_connie_markers_total_l666_66690

/-- The total number of markers Connie has is 3343, given that she has 2315 red markers and 1028 blue markers. -/
theorem connie_markers_total : 
  let red_markers : ℕ := 2315
  let blue_markers : ℕ := 1028
  red_markers + blue_markers = 3343 := by sorry

end NUMINAMATH_CALUDE_connie_markers_total_l666_66690


namespace NUMINAMATH_CALUDE_u_equals_fib_squared_l666_66691

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

def u : ℕ → ℤ
  | 0 => 4
  | 1 => 9
  | n + 2 => 3 * u (n + 1) - u n - 2 * (-1 : ℤ) ^ (n + 2)

theorem u_equals_fib_squared (n : ℕ) : u n = (fib (n + 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_u_equals_fib_squared_l666_66691


namespace NUMINAMATH_CALUDE_arithmetic_vector_sequence_sum_parallel_l666_66657

/-- An arithmetic vector sequence in 2D space -/
def ArithmeticVectorSequence (a : ℕ → ℝ × ℝ) : Prop :=
  ∃ d : ℝ × ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first n vectors in a sequence -/
def VectorSum (a : ℕ → ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  (List.range n).map a |>.sum

/-- Two vectors are parallel -/
def Parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = k • w

theorem arithmetic_vector_sequence_sum_parallel
  (a : ℕ → ℝ × ℝ) (h : ArithmeticVectorSequence a) :
  Parallel (VectorSum a 21) (a 11) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_vector_sequence_sum_parallel_l666_66657


namespace NUMINAMATH_CALUDE_goose_eggs_count_l666_66610

/-- The number of goose eggs laid at a pond -/
def total_eggs : ℕ := 400

/-- The fraction of eggs that hatched -/
def hatch_rate : ℚ := 1/2

/-- The fraction of hatched geese that survived the first month -/
def first_month_survival_rate : ℚ := 3/4

/-- The fraction of geese that survived the first month but did not survive the first year -/
def first_year_death_rate : ℚ := 3/5

/-- The number of geese that survived the first year -/
def survived_first_year : ℕ := 120

theorem goose_eggs_count :
  (total_eggs : ℚ) * hatch_rate * first_month_survival_rate * (1 - first_year_death_rate) = survived_first_year :=
sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l666_66610


namespace NUMINAMATH_CALUDE_school_study_sample_size_l666_66695

/-- Represents a collection of student report cards -/
structure ReportCardCollection where
  total : Nat
  selected : Nat
  h_selected_le_total : selected ≤ total

/-- Defines the sample size of a report card collection -/
def sampleSize (collection : ReportCardCollection) : Nat :=
  collection.selected

/-- Theorem stating that for the given scenario, the sample size is 100 -/
theorem school_study_sample_size :
  ∀ (collection : ReportCardCollection),
    collection.total = 1000 →
    collection.selected = 100 →
    sampleSize collection = 100 := by
  sorry

end NUMINAMATH_CALUDE_school_study_sample_size_l666_66695


namespace NUMINAMATH_CALUDE_max_piles_count_l666_66679

/-- The total number of stones --/
def total_stones : ℕ := 660

/-- A function to check if two pile sizes are similar (differ by strictly less than 2 times) --/
def similar_sizes (a b : ℕ) : Prop := a < 2 * b ∧ b < 2 * a

/-- A type representing a valid division of stones into piles --/
def valid_division (piles : List ℕ) : Prop :=
  piles.sum = total_stones ∧
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → similar_sizes a b

/-- The theorem stating that the maximum number of piles is 30 --/
theorem max_piles_count :
  (∃ (piles : List ℕ), valid_division piles ∧ piles.length = 30) ∧
  (∀ (piles : List ℕ), valid_division piles → piles.length ≤ 30) :=
sorry

end NUMINAMATH_CALUDE_max_piles_count_l666_66679


namespace NUMINAMATH_CALUDE_valid_pairings_count_l666_66694

def num_colors : ℕ := 5

def total_pairings (n : ℕ) : ℕ := n * n

def same_color_pairings (n : ℕ) : ℕ := n

theorem valid_pairings_count :
  total_pairings num_colors - same_color_pairings num_colors = 20 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairings_count_l666_66694


namespace NUMINAMATH_CALUDE_converse_of_negative_square_positive_l666_66641

theorem converse_of_negative_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, x^2 > 0 → x < 0) :=
sorry

end NUMINAMATH_CALUDE_converse_of_negative_square_positive_l666_66641


namespace NUMINAMATH_CALUDE_average_age_campo_verde_l666_66633

/-- Proves that the average age of a population is 40 years, given the specified conditions -/
theorem average_age_campo_verde (H : ℕ) (h_positive : H > 0) : 
  let M := (3 / 2 : ℚ) * H
  let total_population := H + M
  let men_age_sum := 37 * H
  let women_age_sum := 42 * M
  let total_age_sum := men_age_sum + women_age_sum
  (total_age_sum / total_population : ℚ) = 40 := by
sorry


end NUMINAMATH_CALUDE_average_age_campo_verde_l666_66633


namespace NUMINAMATH_CALUDE_kaleb_allowance_l666_66683

theorem kaleb_allowance (savings : ℕ) (toy_cost : ℕ) (num_toys : ℕ) (allowance : ℕ) : 
  savings = 21 → 
  toy_cost = 6 → 
  num_toys = 6 → 
  allowance = num_toys * toy_cost - savings → 
  allowance = 15 := by
sorry

end NUMINAMATH_CALUDE_kaleb_allowance_l666_66683


namespace NUMINAMATH_CALUDE_place_values_and_names_l666_66630

/-- Represents a place in a base-10 positional number system -/
inductive Place : Nat → Type where
  | units : Place 1
  | next (n : Nat) : Place n → Place (n + 1)

/-- The value of a place in a base-10 positional number system -/
def placeValue : ∀ n, Place n → Nat
  | _, Place.units => 1
  | _, Place.next _ p => 10 * placeValue _ p

/-- The name of a place in a base-10 positional number system -/
def placeName : ∀ n, Place n → String
  | _, Place.units => "units"
  | _, Place.next _ p =>
    let prev := placeName _ p
    if prev = "units" then "tens"
    else if prev = "tens" then "hundreds"
    else if prev = "hundreds" then "thousands"
    else if prev = "thousands" then "ten thousands"
    else if prev = "ten thousands" then "hundred thousands"
    else if prev = "hundred thousands" then "millions"
    else if prev = "millions" then "ten millions"
    else if prev = "ten millions" then "hundred millions"
    else "billion"

theorem place_values_and_names :
  ∃ (fifth tenth : Nat) (p5 : Place fifth) (p10 : Place tenth),
    fifth = 5 ∧
    tenth = 10 ∧
    placeName _ p5 = "ten thousands" ∧
    placeName _ p10 = "billion" ∧
    placeValue _ p5 = 10000 ∧
    placeValue _ p10 = 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_place_values_and_names_l666_66630


namespace NUMINAMATH_CALUDE_digit_sum_problem_l666_66661

/-- Given that P, Q, and R are single digits and PQR + QR = 1012, prove that P + Q + R = 20 -/
theorem digit_sum_problem (P Q R : ℕ) : 
  P < 10 → Q < 10 → R < 10 → 
  100 * P + 10 * Q + R + 10 * Q + R = 1012 →
  P + Q + R = 20 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l666_66661


namespace NUMINAMATH_CALUDE_condition_relations_l666_66677

-- Define the propositions
variable (A B C D : Prop)

-- Define the given conditions
axiom A_sufficient_D : A → D
axiom B_sufficient_C : B → C
axiom D_necessary_C : C → D
axiom D_sufficient_B : D → B

-- Theorem to prove
theorem condition_relations :
  (C → D) ∧ (A → B) := by sorry

end NUMINAMATH_CALUDE_condition_relations_l666_66677


namespace NUMINAMATH_CALUDE_odd_prime_expression_factors_l666_66609

theorem odd_prime_expression_factors (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (hodd_a : Odd a) (hodd_b : Odd b) (hab : a < b) : 
  (Finset.filter (· ∣ a^3 * b) (Finset.range (a^3 * b + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_expression_factors_l666_66609


namespace NUMINAMATH_CALUDE_ice_cream_price_l666_66685

theorem ice_cream_price (game_cost : ℚ) (num_ice_creams : ℕ) (h1 : game_cost = 60) (h2 : num_ice_creams = 24) :
  game_cost / num_ice_creams = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_price_l666_66685


namespace NUMINAMATH_CALUDE_pa_distance_bounds_l666_66640

/-- Given a segment AB of length 2 and a point P satisfying |PA| + |PB| = 8,
    prove that the distance |PA| is bounded by 3 ≤ |PA| ≤ 5. -/
theorem pa_distance_bounds (A B P : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist A B = 2)
  (h2 : dist P A + dist P B = 8) :
  3 ≤ dist P A ∧ dist P A ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_pa_distance_bounds_l666_66640


namespace NUMINAMATH_CALUDE_factor_sum_l666_66646

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 54 := by sorry

end NUMINAMATH_CALUDE_factor_sum_l666_66646


namespace NUMINAMATH_CALUDE_modulus_of_z_l666_66688

theorem modulus_of_z (z : ℂ) (h : z / (1 - z) = Complex.I) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l666_66688


namespace NUMINAMATH_CALUDE_division_multiplication_order_matters_l666_66618

theorem division_multiplication_order_matters : (32 / 0.25) * 4 ≠ 32 / (0.25 * 4) := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_order_matters_l666_66618


namespace NUMINAMATH_CALUDE_age_sum_is_23_l666_66681

/-- The ages of Al, Bob, and Carl satisfy the given conditions and their sum is 23 -/
theorem age_sum_is_23 (a b c : ℕ) : 
  a = 10 * b * c ∧ 
  a^3 = 8000 + 8 * b^3 * c^3 → 
  a + b + c = 23 := by
sorry

end NUMINAMATH_CALUDE_age_sum_is_23_l666_66681


namespace NUMINAMATH_CALUDE_append_nine_to_two_digit_number_l666_66601

/-- Given a two-digit number with tens digit t and units digit u,
    appending 9 to the end results in the number 100t + 10u + 9 -/
theorem append_nine_to_two_digit_number (t u : ℕ) (h : t ≤ 9 ∧ u ≤ 9) :
  (10 * t + u) * 10 + 9 = 100 * t + 10 * u + 9 := by
  sorry

end NUMINAMATH_CALUDE_append_nine_to_two_digit_number_l666_66601


namespace NUMINAMATH_CALUDE_reflection_maps_correctly_l666_66652

-- Define points in 2D space
def C : Prod ℝ ℝ := (-3, 2)
def D : Prod ℝ ℝ := (-2, 5)
def C' : Prod ℝ ℝ := (3, -2)
def D' : Prod ℝ ℝ := (2, -5)

-- Define reflection across y = -x
def reflect_across_y_eq_neg_x (p : Prod ℝ ℝ) : Prod ℝ ℝ :=
  (-p.2, -p.1)

-- Theorem statement
theorem reflection_maps_correctly :
  reflect_across_y_eq_neg_x C = C' ∧
  reflect_across_y_eq_neg_x D = D' := by
  sorry

end NUMINAMATH_CALUDE_reflection_maps_correctly_l666_66652


namespace NUMINAMATH_CALUDE_car_distance_ratio_l666_66608

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

/-- Theorem stating the ratio of distances covered by Car A and Car B -/
theorem car_distance_ratio (carA carB : Car)
    (hA : carA = { speed := 80, time := 5 })
    (hB : carB = { speed := 100, time := 2 }) :
    distance carA / distance carB = 2 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_ratio_l666_66608


namespace NUMINAMATH_CALUDE_probability_ratio_l666_66612

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The probability of drawing all five slips with the same number -/
def p : ℚ := (distinct_numbers * 1) / Nat.choose total_slips drawn_slips

/-- The probability of drawing three slips with one number and two with another -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

theorem probability_ratio :
  q / p = 450 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l666_66612


namespace NUMINAMATH_CALUDE_triangle_side_length_l666_66600

theorem triangle_side_length (D E F : ℝ) : 
  -- Triangle DEF exists
  (0 < D) → (0 < E) → (0 < F) → 
  (D + E > F) → (D + F > E) → (E + F > D) →
  -- Given conditions
  (E = 45 * π / 180) →  -- Convert 45° to radians
  (D = 100) →
  (F = 100 * Real.sqrt 2) →
  -- Conclusion
  (E = Real.sqrt (30000 + 5000 * (Real.sqrt 6 - Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l666_66600


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l666_66604

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a survey with its characteristics -/
structure Survey where
  totalUnits : ℕ
  sampleSize : ℕ
  hasSignificantDifferences : Bool

/-- Determines the most appropriate sampling method for a given survey -/
def mostAppropriateSamplingMethod (s : Survey) : SamplingMethod :=
  if s.hasSignificantDifferences then
    SamplingMethod.Stratified
  else
    SamplingMethod.SimpleRandom

/-- The first survey of high school classes -/
def survey1 : Survey :=
  { totalUnits := 15
  , sampleSize := 2
  , hasSignificantDifferences := false }

/-- The second survey of stores in the city -/
def survey2 : Survey :=
  { totalUnits := 1500
  , sampleSize := 15
  , hasSignificantDifferences := true }

theorem appropriate_sampling_methods :
  (mostAppropriateSamplingMethod survey1 = SamplingMethod.SimpleRandom) ∧
  (mostAppropriateSamplingMethod survey2 = SamplingMethod.Stratified) := by
  sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l666_66604


namespace NUMINAMATH_CALUDE_number_ratio_l666_66649

theorem number_ratio (first second third : ℚ) : 
  first + second + third = 220 →
  second = 60 →
  third = (1 / 3) * first →
  first / second = 2 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l666_66649


namespace NUMINAMATH_CALUDE_max_boxes_in_wooden_box_l666_66673

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of boxes that can fit in the wooden box -/
def max_boxes_by_volume (wooden_box : BoxDimensions) (small_box : BoxDimensions) : ℕ :=
  (wooden_box.length / small_box.length) *
  (wooden_box.width / small_box.width) *
  (wooden_box.height / small_box.height)

/-- Calculates the maximum number of boxes based on weight constraints -/
def max_boxes_by_weight (max_weight : ℕ) (box_weight : ℕ) : ℕ :=
  max_weight / box_weight

/-- Main theorem stating the maximum number of boxes that can be carried -/
theorem max_boxes_in_wooden_box :
  let wooden_box : BoxDimensions := ⟨800, 1000, 600⟩
  let small_box : BoxDimensions := ⟨4, 5, 6⟩
  let max_weight : ℕ := 3000
  let box_weight : ℕ := 500 / 1000  -- 500 grams converted to kilograms
  min (max_boxes_by_volume wooden_box small_box) (max_boxes_by_weight max_weight box_weight) = 6000 := by
  sorry


end NUMINAMATH_CALUDE_max_boxes_in_wooden_box_l666_66673


namespace NUMINAMATH_CALUDE_wayne_shrimp_guests_l666_66662

/-- Given Wayne's shrimp appetizer scenario, prove the number of guests he can serve. -/
theorem wayne_shrimp_guests :
  ∀ (shrimp_per_guest : ℕ) 
    (cost_per_pound : ℚ) 
    (shrimp_per_pound : ℕ) 
    (total_spent : ℚ),
  shrimp_per_guest = 5 →
  cost_per_pound = 17 →
  shrimp_per_pound = 20 →
  total_spent = 170 →
  (total_spent / cost_per_pound * shrimp_per_pound) / shrimp_per_guest = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_wayne_shrimp_guests_l666_66662


namespace NUMINAMATH_CALUDE_max_a_value_l666_66638

-- Define the function f(x) = -x^3 + ax
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x

-- Define the property of f being monotonically decreasing on [1, +∞)
def is_monotone_decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x ≤ y → f y ≤ f x

-- Theorem statement
theorem max_a_value (a : ℝ) :
  is_monotone_decreasing_on_interval (f a) → a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l666_66638


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l666_66663

def is_necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem quadratic_roots_condition :
  ∀ m n : ℝ,
  let roots := {x : ℝ | x^2 - m*x + n = 0}
  is_necessary_not_sufficient
    (m > 2 ∧ n > 1)
    (∀ x ∈ roots, x > 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l666_66663


namespace NUMINAMATH_CALUDE_total_revenue_is_4586_80_l666_66684

structure PhoneModel where
  name : String
  initialInventory : ℕ
  price : ℚ
  discountRate : ℚ
  taxRate : ℚ
  damaged : ℕ
  finalInventory : ℕ

def calculateRevenue (model : PhoneModel) : ℚ :=
  let discountedPrice := model.price * (1 - model.discountRate)
  let priceAfterTax := discountedPrice * (1 + model.taxRate)
  let soldUnits := model.initialInventory - model.finalInventory - model.damaged
  soldUnits * priceAfterTax

def totalRevenue (models : List PhoneModel) : ℚ :=
  models.map calculateRevenue |>.sum

def phoneModels : List PhoneModel := [
  { name := "Samsung Galaxy S20", initialInventory := 14, price := 800, discountRate := 0.1, taxRate := 0.12, damaged := 2, finalInventory := 10 },
  { name := "iPhone 12", initialInventory := 8, price := 1000, discountRate := 0.15, taxRate := 0.1, damaged := 1, finalInventory := 5 },
  { name := "Google Pixel 5", initialInventory := 7, price := 700, discountRate := 0.05, taxRate := 0.08, damaged := 0, finalInventory := 8 },
  { name := "OnePlus 8T", initialInventory := 6, price := 600, discountRate := 0.2, taxRate := 0.15, damaged := 1, finalInventory := 3 }
]

theorem total_revenue_is_4586_80 :
  totalRevenue phoneModels = 4586.8 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_4586_80_l666_66684


namespace NUMINAMATH_CALUDE_child_growth_l666_66655

theorem child_growth (current_height previous_height : ℝ) 
  (h1 : current_height = 41.5)
  (h2 : previous_height = 38.5) : 
  current_height - previous_height = 3 := by
sorry

end NUMINAMATH_CALUDE_child_growth_l666_66655


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l666_66625

def M : Set ℕ := {0, 1, 2, 3}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l666_66625


namespace NUMINAMATH_CALUDE_total_marbles_l666_66615

theorem total_marbles (jars : ℕ) (clay_pots : ℕ) (marbles_per_jar : ℕ) :
  jars = 16 →
  jars = 2 * clay_pots →
  marbles_per_jar = 5 →
  jars * marbles_per_jar + clay_pots * (3 * marbles_per_jar) = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l666_66615


namespace NUMINAMATH_CALUDE_steve_pie_ratio_l666_66660

/-- Steve's weekly pie baking schedule -/
structure PieSchedule where
  monday_apple : ℕ
  monday_blueberry : ℕ
  tuesday_cherry : ℕ
  tuesday_blueberry : ℕ
  wednesday_apple : ℕ
  wednesday_blueberry : ℕ
  thursday_cherry : ℕ
  thursday_blueberry : ℕ
  friday_apple : ℕ
  friday_blueberry : ℕ
  saturday_apple : ℕ
  saturday_cherry : ℕ
  saturday_blueberry : ℕ
  sunday_apple : ℕ
  sunday_cherry : ℕ
  sunday_blueberry : ℕ

/-- Calculate the total number of each type of pie baked in a week -/
def total_pies (schedule : PieSchedule) : ℕ × ℕ × ℕ :=
  let apple := schedule.monday_apple + schedule.wednesday_apple + schedule.friday_apple + 
                schedule.saturday_apple + schedule.sunday_apple
  let cherry := schedule.tuesday_cherry + schedule.thursday_cherry + 
                 schedule.saturday_cherry + schedule.sunday_cherry
  let blueberry := schedule.monday_blueberry + schedule.tuesday_blueberry + 
                   schedule.wednesday_blueberry + schedule.thursday_blueberry + 
                   schedule.friday_blueberry + schedule.saturday_blueberry + 
                   schedule.sunday_blueberry
  (apple, cherry, blueberry)

/-- Steve's actual weekly pie baking schedule -/
def steve_schedule : PieSchedule := {
  monday_apple := 16, monday_blueberry := 10,
  tuesday_cherry := 14, tuesday_blueberry := 8,
  wednesday_apple := 20, wednesday_blueberry := 12,
  thursday_cherry := 18, thursday_blueberry := 10,
  friday_apple := 16, friday_blueberry := 10,
  saturday_apple := 10, saturday_cherry := 8, saturday_blueberry := 6,
  sunday_apple := 6, sunday_cherry := 12, sunday_blueberry := 4
}

theorem steve_pie_ratio : 
  ∃ (k : ℕ), k > 0 ∧ total_pies steve_schedule = (17 * k, 13 * k, 15 * k) := by
  sorry

end NUMINAMATH_CALUDE_steve_pie_ratio_l666_66660


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l666_66607

-- Define the surface area of the cube
def surface_area : ℝ := 864

-- Theorem stating the relationship between surface area and volume
theorem cube_volume_from_surface_area :
  ∃ (side_length : ℝ), 
    side_length > 0 ∧ 
    6 * side_length^2 = surface_area ∧ 
    side_length^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l666_66607


namespace NUMINAMATH_CALUDE_jeremy_age_l666_66653

theorem jeremy_age (amy jeremy chris : ℕ) 
  (h1 : amy + jeremy + chris = 132)
  (h2 : amy = jeremy / 3)
  (h3 : chris = 2 * amy) :
  jeremy = 66 := by
sorry

end NUMINAMATH_CALUDE_jeremy_age_l666_66653


namespace NUMINAMATH_CALUDE_fudge_difference_is_14_ounces_l666_66650

/-- Conversion factor from pounds to ounces -/
def poundsToOunces : ℚ := 16

/-- Marina's fudge in pounds -/
def marinaFudgePounds : ℚ := 4.5

/-- Amount of fudge Lazlo has less than 4 pounds, in ounces -/
def lazloFudgeDifference : ℚ := 6

/-- Calculates the difference in ounces of fudge between Marina and Lazlo -/
def fudgeDifferenceInOunces : ℚ :=
  marinaFudgePounds * poundsToOunces - (4 * poundsToOunces - lazloFudgeDifference)

theorem fudge_difference_is_14_ounces :
  fudgeDifferenceInOunces = 14 := by
  sorry

end NUMINAMATH_CALUDE_fudge_difference_is_14_ounces_l666_66650


namespace NUMINAMATH_CALUDE_range_of_f_l666_66613

def f (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ f x = y) ↔ y ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l666_66613


namespace NUMINAMATH_CALUDE_percentage_of_five_digit_numbers_with_repeats_l666_66616

def five_digit_numbers : ℕ := 90000

def numbers_without_repeats : ℕ := 9 * 9 * 8 * 7 * 6

def numbers_with_repeats : ℕ := five_digit_numbers - numbers_without_repeats

def percentage_with_repeats : ℚ := numbers_with_repeats / five_digit_numbers

theorem percentage_of_five_digit_numbers_with_repeats :
  (percentage_with_repeats * 100).floor / 10 = 698 / 10 := by sorry

end NUMINAMATH_CALUDE_percentage_of_five_digit_numbers_with_repeats_l666_66616


namespace NUMINAMATH_CALUDE_expression_value_l666_66656

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = -4) :
  5 * (x - y)^2 - x * y = 28 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l666_66656


namespace NUMINAMATH_CALUDE_donut_purchase_proof_l666_66668

/-- Represents the number of items purchased over the week -/
def total_items : ℕ := 4

/-- Price of a croissant in cents -/
def croissant_price : ℕ := 60

/-- Price of a donut in cents -/
def donut_price : ℕ := 90

/-- Represents the number of donuts purchased -/
def num_donuts : ℕ := sorry

/-- Represents the number of croissants purchased -/
def num_croissants : ℕ := total_items - num_donuts

/-- Total cost in cents -/
def total_cost : ℕ := num_donuts * donut_price + num_croissants * croissant_price

theorem donut_purchase_proof : 
  (num_donuts + num_croissants = total_items) ∧ 
  (total_cost % 100 = 0) ∧ 
  (num_donuts = 2) := by sorry

end NUMINAMATH_CALUDE_donut_purchase_proof_l666_66668


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l666_66698

theorem complex_fraction_sum : (2 + 2 * Complex.I) / Complex.I + (1 + Complex.I) / (1 - Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l666_66698


namespace NUMINAMATH_CALUDE_rock_cd_price_l666_66605

/-- The price of a rock and roll CD -/
def rock_price : ℝ := sorry

/-- The price of a pop CD -/
def pop_price : ℝ := 10

/-- The price of a dance CD -/
def dance_price : ℝ := 3

/-- The price of a country CD -/
def country_price : ℝ := 7

/-- The number of each type of CD Julia wants to buy -/
def quantity : ℕ := 4

/-- The amount of money Julia has -/
def julia_money : ℝ := 75

/-- The amount Julia is short by -/
def short_amount : ℝ := 25

theorem rock_cd_price : rock_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_rock_cd_price_l666_66605


namespace NUMINAMATH_CALUDE_distance_to_focus_is_13_l666_66624

/-- Parabola with equation y^2 = 16x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  axis_of_symmetry : ℝ → ℝ

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

theorem distance_to_focus_is_13 (p : Parabola) (P : PointOnParabola p) 
  (h_equation : p.equation = fun x y => y^2 = 16*x)
  (h_distance : abs P.point.2 = 12) :
  dist P.point p.focus = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_is_13_l666_66624


namespace NUMINAMATH_CALUDE_milk_production_per_cow_l666_66678

theorem milk_production_per_cow 
  (num_cows : ℕ) 
  (milk_price : ℚ) 
  (butter_ratio : ℕ) 
  (butter_price : ℚ) 
  (num_customers : ℕ) 
  (milk_per_customer : ℕ) 
  (total_earnings : ℚ) 
  (h1 : num_cows = 12)
  (h2 : milk_price = 3)
  (h3 : butter_ratio = 2)
  (h4 : butter_price = 3/2)
  (h5 : num_customers = 6)
  (h6 : milk_per_customer = 6)
  (h7 : total_earnings = 144) :
  (total_earnings / num_cows : ℚ) / milk_price = 4 := by
sorry

end NUMINAMATH_CALUDE_milk_production_per_cow_l666_66678


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l666_66687

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) (h_sum : sum_condition a) :
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l666_66687


namespace NUMINAMATH_CALUDE_remainder_three_pow_2040_mod_5_l666_66620

theorem remainder_three_pow_2040_mod_5 : 3^2040 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_pow_2040_mod_5_l666_66620


namespace NUMINAMATH_CALUDE_heartsuit_five_three_l666_66628

def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

theorem heartsuit_five_three : heartsuit 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_five_three_l666_66628


namespace NUMINAMATH_CALUDE_nonIntersectingPolylines_correct_l666_66651

/-- The number of ways to connect n points on a circle with a non-self-intersecting polyline -/
def nonIntersectingPolylines (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n ≥ 3 then n * 2^(n-3)
  else 0

theorem nonIntersectingPolylines_correct (n : ℕ) (h : n > 1) :
  nonIntersectingPolylines n =
    if n = 2 then 1
    else n * 2^(n-3) := by
  sorry

end NUMINAMATH_CALUDE_nonIntersectingPolylines_correct_l666_66651


namespace NUMINAMATH_CALUDE_ordering_of_expressions_l666_66697

theorem ordering_of_expressions : e^(0.11 : ℝ) > (1.1 : ℝ)^(1.1 : ℝ) ∧ (1.1 : ℝ)^(1.1 : ℝ) > 1.11 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_expressions_l666_66697


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l666_66680

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_expression : 3 * (2 - i) + i * (3 + 2 * i) = (4 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l666_66680


namespace NUMINAMATH_CALUDE_jake_work_hours_l666_66606

/-- Calculates the number of hours needed to work off a debt -/
def hoursToWorkOff (initialDebt : ℚ) (amountPaid : ℚ) (hourlyRate : ℚ) : ℚ :=
  (initialDebt - amountPaid) / hourlyRate

/-- Proves that Jake needs to work 4 hours to pay off his debt -/
theorem jake_work_hours :
  let initialDebt : ℚ := 100
  let amountPaid : ℚ := 40
  let hourlyRate : ℚ := 15
  hoursToWorkOff initialDebt amountPaid hourlyRate = 4 := by
  sorry

end NUMINAMATH_CALUDE_jake_work_hours_l666_66606


namespace NUMINAMATH_CALUDE_margin_selling_price_relation_l666_66623

/-- Proof of the relationship between margin, cost, and selling price -/
theorem margin_selling_price_relation (n : ℝ) (C S M : ℝ) 
  (h1 : n > 2) 
  (h2 : M = (2/n) * C) 
  (h3 : S = C + M) : 
  M = (2/(n+2)) * S := by
  sorry

end NUMINAMATH_CALUDE_margin_selling_price_relation_l666_66623


namespace NUMINAMATH_CALUDE_interview_pass_probability_l666_66622

/-- Represents a job interview with three questions and three chances to answer. -/
structure JobInterview where
  num_questions : ℕ
  num_chances : ℕ
  prob_correct : ℝ

/-- The probability of passing the given job interview. -/
def pass_probability (interview : JobInterview) : ℝ :=
  interview.prob_correct +
  (1 - interview.prob_correct) * interview.prob_correct +
  (1 - interview.prob_correct) * (1 - interview.prob_correct) * interview.prob_correct

/-- Theorem stating that for the specific interview conditions, 
    the probability of passing is 0.973. -/
theorem interview_pass_probability :
  let interview : JobInterview := {
    num_questions := 3,
    num_chances := 3,
    prob_correct := 0.7
  }
  pass_probability interview = 0.973 := by
  sorry


end NUMINAMATH_CALUDE_interview_pass_probability_l666_66622


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l666_66629

theorem pure_imaginary_condition (a : ℝ) : 
  (a = 1 ↔ ∃ (b : ℝ), Complex.mk (a^2 - 1) (a + 1) = Complex.I * b) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l666_66629


namespace NUMINAMATH_CALUDE_tomato_crate_capacity_l666_66647

/-- Represents the problem of determining the capacity of a tomato crate. -/
theorem tomato_crate_capacity 
  (num_crates : ℕ)
  (crates_cost : ℝ)
  (selling_price_per_kg : ℝ)
  (rotten_tomatoes : ℝ)
  (profit : ℝ)
  (h1 : num_crates = 3)
  (h2 : crates_cost = 330)
  (h3 : selling_price_per_kg = 6)
  (h4 : rotten_tomatoes = 3)
  (h5 : profit = 12) :
  ∃ (crate_capacity : ℝ),
    crate_capacity = 20 ∧ 
    (num_crates * crate_capacity - rotten_tomatoes) * selling_price_per_kg - crates_cost = profit :=
by sorry

end NUMINAMATH_CALUDE_tomato_crate_capacity_l666_66647


namespace NUMINAMATH_CALUDE_geometric_series_sum_l666_66611

/-- The limiting sum of a geometric series with first term 6 and common ratio -2/5 is 30/7 -/
theorem geometric_series_sum : 
  let a : ℚ := 6
  let r : ℚ := -2/5
  let s : ℚ := a / (1 - r)
  s = 30/7 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l666_66611


namespace NUMINAMATH_CALUDE_dye_jobs_calculation_l666_66637

def haircut_price : ℕ := 30
def perm_price : ℕ := 40
def dye_job_price : ℕ := 60
def dye_cost : ℕ := 10
def haircuts_scheduled : ℕ := 4
def perms_scheduled : ℕ := 1
def tips : ℕ := 50
def total_revenue : ℕ := 310

def dye_jobs_scheduled : ℕ := 2

theorem dye_jobs_calculation :
  (haircuts_scheduled * haircut_price + 
   perms_scheduled * perm_price + 
   dye_jobs_scheduled * (dye_job_price - dye_cost) + 
   tips) = total_revenue := by sorry

end NUMINAMATH_CALUDE_dye_jobs_calculation_l666_66637


namespace NUMINAMATH_CALUDE_range_of_a_l666_66664

/-- The solution set of the inequality |x+a|+|2x-1| ≤ |2x+1| with respect to x -/
def A (a : ℝ) : Set ℝ :=
  {x : ℝ | |x + a| + |2*x - 1| ≤ |2*x + 1|}

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x ∈ A a) → a ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l666_66664


namespace NUMINAMATH_CALUDE_golden_rectangle_ratio_l666_66667

theorem golden_rectangle_ratio (x y : ℝ) (h1 : x > y) (h2 : y > 0) : 
  (y / x = (x - y) / y) → (x / y = (1 + Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_golden_rectangle_ratio_l666_66667


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l666_66659

/-- The perimeter of a rectangle with a long side of 1 meter and a short side
    that is 2/8 meter shorter than the long side is 3.5 meters. -/
theorem rectangle_perimeter : 
  let long_side : ℝ := 1
  let short_side : ℝ := long_side - 2/8
  let perimeter : ℝ := 2 * long_side + 2 * short_side
  perimeter = 3.5 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l666_66659


namespace NUMINAMATH_CALUDE_apple_price_calculation_l666_66602

/-- Calculates the price of each apple given the produce inventory and total worth -/
theorem apple_price_calculation (asparagus_bundles : ℕ) (asparagus_price : ℚ)
  (grape_boxes : ℕ) (grape_price : ℚ) (apple_count : ℕ) (total_worth : ℚ) :
  asparagus_bundles = 60 →
  asparagus_price = 3 →
  grape_boxes = 40 →
  grape_price = 5/2 →
  apple_count = 700 →
  total_worth = 630 →
  (total_worth - (asparagus_bundles * asparagus_price + grape_boxes * grape_price)) / apple_count = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_calculation_l666_66602


namespace NUMINAMATH_CALUDE_complex_modulus_l666_66658

theorem complex_modulus (z : ℂ) : (z - 1) * I = I - 1 → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l666_66658


namespace NUMINAMATH_CALUDE_modified_prism_edge_count_l666_66689

/-- Represents a modified rectangular prism with intersecting corner cuts -/
structure ModifiedPrism where
  original_edges : Nat
  vertex_count : Nat
  new_edges_per_vertex : Nat
  intersections_per_vertex : Nat
  additional_edges_per_intersection : Nat

/-- Calculates the total number of edges in the modified prism -/
def total_edges (p : ModifiedPrism) : Nat :=
  p.original_edges + 
  (p.vertex_count * p.new_edges_per_vertex) + 
  (p.vertex_count * p.intersections_per_vertex * p.additional_edges_per_intersection)

/-- Theorem stating that the modified prism has 52 edges -/
theorem modified_prism_edge_count :
  ∃ (p : ModifiedPrism), total_edges p = 52 :=
sorry

end NUMINAMATH_CALUDE_modified_prism_edge_count_l666_66689


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l666_66626

/-- The minimum value of 2x + y given the constraints |y| ≤ 2 - x and x ≥ -1 is -5 -/
theorem min_value_2x_plus_y (x y : ℝ) (h1 : |y| ≤ 2 - x) (h2 : x ≥ -1) : 
  ∃ (m : ℝ), m = -5 ∧ ∀ (x' y' : ℝ), |y'| ≤ 2 - x' → x' ≥ -1 → 2*x' + y' ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l666_66626


namespace NUMINAMATH_CALUDE_max_value_of_function_l666_66699

theorem max_value_of_function :
  let f : ℝ → ℝ := λ x => (Real.sqrt 3 / 2) * Real.sin (x + Real.pi / 2) + Real.cos (Real.pi / 6 - x)
  ∃ (M : ℝ), M = Real.sqrt 13 / 2 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l666_66699


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l666_66686

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (∀ x : ℂ, (3 : ℂ) * x^2 + (a : ℂ) * x + (b : ℂ) = 0 ↔ x = 4 + 2*I ∨ x = 4 - 2*I) ∧
    c = 3 ∧
    a = -24 ∧
    b = 60 :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l666_66686


namespace NUMINAMATH_CALUDE_seating_arrangement_with_constraint_l666_66643

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def arrangements_with_pair_together (n : ℕ) : ℕ :=
  Nat.factorial (n - 1) * Nat.factorial 2

theorem seating_arrangement_with_constraint :
  total_arrangements 8 - arrangements_with_pair_together 8 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_with_constraint_l666_66643


namespace NUMINAMATH_CALUDE_train_speed_problem_l666_66634

theorem train_speed_problem (initial_distance : ℝ) (speed_train1 : ℝ) (distance_before_meet : ℝ) (time_before_meet : ℝ) :
  initial_distance = 120 →
  speed_train1 = 40 →
  distance_before_meet = 70 →
  time_before_meet = 1 →
  ∃ speed_train2 : ℝ,
    speed_train2 = 30 ∧
    initial_distance - (speed_train1 + speed_train2) * time_before_meet = distance_before_meet :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l666_66634


namespace NUMINAMATH_CALUDE_chad_pet_food_difference_l666_66645

theorem chad_pet_food_difference :
  let cat_packages : ℕ := 6
  let dog_packages : ℕ := 2
  let cat_cans_per_package : ℕ := 9
  let dog_cans_per_package : ℕ := 3
  let total_cat_cans := cat_packages * cat_cans_per_package
  let total_dog_cans := dog_packages * dog_cans_per_package
  total_cat_cans - total_dog_cans = 48 :=
by sorry

end NUMINAMATH_CALUDE_chad_pet_food_difference_l666_66645


namespace NUMINAMATH_CALUDE_find_number_l666_66603

theorem find_number : ∃ x : ℝ, 0.123 + 0.321 + x = 1.794 ∧ x = 1.350 := by sorry

end NUMINAMATH_CALUDE_find_number_l666_66603


namespace NUMINAMATH_CALUDE_sin_graph_shift_l666_66642

theorem sin_graph_shift (x : ℝ) :
  3 * Real.sin (2 * (x + π/8) - π/4) = 3 * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l666_66642


namespace NUMINAMATH_CALUDE_school_sample_size_l666_66654

theorem school_sample_size (n : ℕ) : 
  (6 : ℚ) / 11 * n / 10 - (5 : ℚ) / 11 * n / 10 = 12 → n = 1320 := by
  sorry

end NUMINAMATH_CALUDE_school_sample_size_l666_66654


namespace NUMINAMATH_CALUDE_equal_money_after_transfer_l666_66671

/-- Given that Ann has $777 and Bill has $1,111, prove that if Bill gives $167 to Ann, 
    they will have equal amounts of money. -/
theorem equal_money_after_transfer (ann_initial : ℕ) (bill_initial : ℕ) (transfer : ℕ) : 
  ann_initial = 777 →
  bill_initial = 1111 →
  transfer = 167 →
  ann_initial + transfer = bill_initial - transfer :=
by
  sorry

#check equal_money_after_transfer

end NUMINAMATH_CALUDE_equal_money_after_transfer_l666_66671


namespace NUMINAMATH_CALUDE_correspondence_count_l666_66636

-- Define the sets and correspondences
def Triangle : Type := sorry
def Circle : Type := sorry
def RealNumber : Type := ℝ

-- Define the correspondences
def correspondence1 : Triangle → Circle := sorry
def correspondence2 : Triangle → RealNumber := sorry
def correspondence3 : RealNumber → RealNumber := sorry
def correspondence4 : RealNumber → RealNumber := sorry

-- Define what it means to be a mapping
def is_mapping (f : α → β) : Prop := ∀ x : α, ∃! y : β, f x = y

-- Define what it means to be a function
def is_function (f : α → β) : Prop := ∀ x : α, ∃ y : β, f x = y

-- The main theorem
theorem correspondence_count :
  (is_mapping correspondence1 ∧
   is_mapping correspondence2 ∧
   is_mapping correspondence3 ∧
   ¬is_mapping correspondence4) ∧
  (¬is_function correspondence1 ∧
   is_function correspondence2 ∧
   is_function correspondence3 ∧
   ¬is_function correspondence4) :=
sorry

end NUMINAMATH_CALUDE_correspondence_count_l666_66636


namespace NUMINAMATH_CALUDE_specific_pyramid_surface_area_l666_66669

/-- Represents a pyramid with a parallelogram base -/
structure Pyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_diagonal : ℝ
  height : ℝ

/-- Calculates the total surface area of the pyramid -/
def totalSurfaceArea (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific pyramid -/
theorem specific_pyramid_surface_area :
  let p : Pyramid := { base_side1 := 10, base_side2 := 8, base_diagonal := 6, height := 4 }
  totalSurfaceArea p = 8 * (11 + Real.sqrt 34) := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_surface_area_l666_66669


namespace NUMINAMATH_CALUDE_intersection_A_B_l666_66676

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = 2^x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ico 1 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l666_66676


namespace NUMINAMATH_CALUDE_existence_of_product_one_derivatives_l666_66696

theorem existence_of_product_one_derivatives 
  (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_diff : DifferentiableOn ℝ f (Set.Ioo 0 1))
  (h_range : Set.range f ⊆ Set.Icc 0 1)
  (h_zero : f 0 = 0)
  (h_one : f 1 = 1) :
  ∃ a b : ℝ, a ∈ Set.Ioo 0 1 ∧ b ∈ Set.Ioo 0 1 ∧ a ≠ b ∧ 
    deriv f a * deriv f b = 1 :=
sorry

end NUMINAMATH_CALUDE_existence_of_product_one_derivatives_l666_66696


namespace NUMINAMATH_CALUDE_min_value_a_min_value_a_tight_l666_66635

theorem min_value_a (a : ℝ) : 
  (∀ x > 0, x^2 + a*x + 1 ≥ 0) → a ≥ -2 :=
by sorry

theorem min_value_a_tight : 
  ∃ a : ℝ, (∀ x > 0, x^2 + a*x + 1 ≥ 0) ∧ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_min_value_a_tight_l666_66635


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l666_66675

/-- Represents the time for a train to cross a bridge given its parameters -/
theorem train_bridge_crossing_time 
  (L : ℝ) -- Length of the train
  (u : ℝ) -- Initial speed of the train
  (a : ℝ) -- Constant acceleration of the train
  (t : ℝ) -- Time to cross the signal post
  (B : ℝ) -- Length of the bridge
  (h1 : L > 0) -- Train has positive length
  (h2 : u ≥ 0) -- Initial speed is non-negative
  (h3 : a > 0) -- Acceleration is positive
  (h4 : t > 0) -- Time to cross signal post is positive
  (h5 : B > 0) -- Bridge has positive length
  (h6 : L = u * t + (1/2) * a * t^2) -- Equation for crossing signal post
  : ∃ T, T > 0 ∧ B + L = u * T + (1/2) * a * T^2 :=
sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l666_66675


namespace NUMINAMATH_CALUDE_rug_area_is_48_l666_66692

/-- Calculates the area of a rug with specific dimensions -/
def rugArea (rect_length rect_width para_base para_height : ℝ) : ℝ :=
  let rect_area := rect_length * rect_width
  let para_area := para_base * para_height
  rect_area + 2 * para_area

/-- Theorem stating that a rug with given dimensions has an area of 48 square meters -/
theorem rug_area_is_48 :
  rugArea 6 4 3 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_rug_area_is_48_l666_66692


namespace NUMINAMATH_CALUDE_isosceles_triangle_l666_66621

/-- A triangle is isosceles if it has at least two equal sides -/
def IsIsosceles (A B C : ℝ × ℝ) : Prop :=
  (dist A B = dist B C) ∨ (dist B C = dist A C) ∨ (dist A C = dist A B)

/-- The perimeter of a triangle is the sum of the lengths of its sides -/
def Perimeter (A B C : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist A C

theorem isosceles_triangle (A B C M N : ℝ × ℝ) :
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = (1 - t) • A + t • B) →  -- M is on AB
  (∃ s : ℝ, 0 < s ∧ s < 1 ∧ N = (1 - s) • B + s • C) →  -- N is on BC
  Perimeter A M C = Perimeter C N A →                  -- Perimeter condition 1
  Perimeter A N B = Perimeter C M B →                  -- Perimeter condition 2
  IsIsosceles A B C :=                                 -- Conclusion
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l666_66621


namespace NUMINAMATH_CALUDE_team_capacity_ratio_l666_66666

/-- The working capacity ratio of two teams -/
def working_capacity_ratio (p_engineers q_engineers : ℕ) (p_days q_days : ℕ) : ℚ × ℚ :=
  let p_capacity := p_engineers * p_days / p_engineers
  let q_capacity := q_engineers * q_days / q_engineers
  (p_capacity, q_capacity)

/-- Theorem: The ratio of working capacity for the given teams is 16:15 -/
theorem team_capacity_ratio :
  let (p_cap, q_cap) := working_capacity_ratio 20 16 32 30
  p_cap / q_cap = 16 / 15 := by
  sorry

end NUMINAMATH_CALUDE_team_capacity_ratio_l666_66666


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l666_66644

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_product_pure_imaginary (b : ℝ) :
  is_pure_imaginary ((1 + b * Complex.I) * (2 + Complex.I)) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l666_66644


namespace NUMINAMATH_CALUDE_root_in_interval_l666_66674

theorem root_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (-4 : ℝ) (-3 : ℝ) ∧ x^3 + 3*x^2 - x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l666_66674


namespace NUMINAMATH_CALUDE_equation_solution_l666_66682

theorem equation_solution :
  ∃! y : ℝ, 7 * (4 * y + 3) - 5 = -3 * (2 - 8 * y) + 5 * y ∧ y = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l666_66682


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l666_66619

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l666_66619


namespace NUMINAMATH_CALUDE_total_pizza_combinations_l666_66665

def num_toppings : ℕ := 8

def num_one_topping (n : ℕ) : ℕ := n

def num_two_toppings (n : ℕ) : ℕ := n.choose 2

def num_three_toppings (n : ℕ) : ℕ := n.choose 3

theorem total_pizza_combinations :
  num_one_topping num_toppings + num_two_toppings num_toppings + num_three_toppings num_toppings = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_combinations_l666_66665


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_l666_66672

def digit_sum (n : ℕ) : ℕ := sorry

def A : ℕ := digit_sum (4444^4444)

def B : ℕ := digit_sum A

theorem sum_of_digits_of_B : digit_sum B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_l666_66672


namespace NUMINAMATH_CALUDE_fraction_zero_l666_66614

theorem fraction_zero (x : ℝ) (h : x ≠ 3) : 
  x = 0 ↔ (2 * x^2 - 6 * x) / (x - 3) = 0 := by sorry

end NUMINAMATH_CALUDE_fraction_zero_l666_66614


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l666_66639

theorem interest_rate_calculation (P r : ℝ) 
  (h1 : P * (1 + 4 * r) = 400)
  (h2 : P * (1 + 6 * r) = 500) :
  r = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l666_66639


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l666_66632

theorem polynomial_divisibility (a b c d : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d = 5 * k) →
  (∃ ka kb kc kd : ℤ, a = 5 * ka ∧ b = 5 * kb ∧ c = 5 * kc ∧ d = 5 * kd) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l666_66632


namespace NUMINAMATH_CALUDE_octal_726_to_binary_l666_66617

/-- Converts a single digit from base 8 to its 3-digit binary representation -/
def octalToBinary (digit : Nat) : Fin 8 → Fin 2 × Fin 2 × Fin 2 := sorry

/-- Converts a 3-digit octal number to its 9-digit binary representation -/
def octalToBinaryThreeDigits (a b c : Fin 8) : Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 := sorry

theorem octal_726_to_binary :
  octalToBinaryThreeDigits 7 2 6 = (1, 1, 1, 0, 1, 0, 1, 1, 0) := by sorry

end NUMINAMATH_CALUDE_octal_726_to_binary_l666_66617


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l666_66631

theorem inverse_proposition_false : ¬(∀ a b : ℝ, a + b > 0 → a > 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l666_66631
