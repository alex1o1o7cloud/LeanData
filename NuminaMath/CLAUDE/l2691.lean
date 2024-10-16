import Mathlib

namespace NUMINAMATH_CALUDE_mn_equals_six_l2691_269163

/-- Given that -x³yⁿ and 3xᵐy² are like terms, prove that mn = 6 -/
theorem mn_equals_six (x y : ℝ) (m n : ℕ) 
  (h : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → -x^3 * y^n = 3 * x^m * y^2) : 
  m * n = 6 := by
  sorry

end NUMINAMATH_CALUDE_mn_equals_six_l2691_269163


namespace NUMINAMATH_CALUDE_inequality_proof_l2691_269153

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^4 ≥ x^2*y^2 + y^2*z^2 + z^2*x^2 ∧ 
  x^2*y^2 + y^2*z^2 + z^2*x^2 ≥ x*y*z*(x+y+z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2691_269153


namespace NUMINAMATH_CALUDE_prove_january_salary_l2691_269192

def january_salary (feb mar apr may : ℕ) : Prop :=
  let jan := 32000 - (feb + mar + apr)
  (feb + mar + apr + may) / 4 = 8100 ∧
  (jan + feb + mar + apr) / 4 = 8000 ∧
  may = 6500 →
  jan = 6100

theorem prove_january_salary :
  ∀ (feb mar apr may : ℕ),
  january_salary feb mar apr may :=
by
  sorry

end NUMINAMATH_CALUDE_prove_january_salary_l2691_269192


namespace NUMINAMATH_CALUDE_total_turtles_count_l2691_269121

/-- Represents the total number of turtles in the lake -/
def total_turtles : ℕ := sorry

/-- Represents the number of striped male adult common turtles -/
def striped_male_adult_common : ℕ := 70

/-- Percentage of common turtles in the lake -/
def common_percentage : ℚ := 1/2

/-- Percentage of female common turtles -/
def common_female_percentage : ℚ := 3/5

/-- Percentage of striped male common turtles among male common turtles -/
def striped_male_common_percentage : ℚ := 1/4

/-- Percentage of adult striped male common turtles among striped male common turtles -/
def adult_striped_male_common_percentage : ℚ := 4/5

theorem total_turtles_count : total_turtles = 1760 := by sorry

end NUMINAMATH_CALUDE_total_turtles_count_l2691_269121


namespace NUMINAMATH_CALUDE_exists_m_with_infinite_solutions_l2691_269117

/-- The equation we're considering -/
def equation (m a b c : ℕ+) : Prop :=
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / (a * b * c) = m / (a + b + c)

/-- The existence of m with infinitely many solutions -/
theorem exists_m_with_infinite_solutions :
  ∃ m : ℕ+, ∀ n : ℕ, ∃ a b c : ℕ+, a > n ∧ b > n ∧ c > n ∧ equation m a b c :=
sorry

end NUMINAMATH_CALUDE_exists_m_with_infinite_solutions_l2691_269117


namespace NUMINAMATH_CALUDE_polar_to_hyperbola_l2691_269102

/-- Theorem: The polar equation ρ² cos(2θ) = 1 represents a hyperbola in Cartesian coordinates -/
theorem polar_to_hyperbola (ρ θ x y : ℝ) : 
  (ρ^2 * (Real.cos (2 * θ)) = 1) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) → 
  (x^2 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_hyperbola_l2691_269102


namespace NUMINAMATH_CALUDE_rectangle_perimeter_and_area_l2691_269165

theorem rectangle_perimeter_and_area :
  ∀ (length width perimeter area : ℝ),
    length = 10 →
    width = length - 3 →
    perimeter = 2 * (length + width) →
    area = length * width →
    perimeter = 34 ∧ area = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_and_area_l2691_269165


namespace NUMINAMATH_CALUDE_k_value_l2691_269172

theorem k_value (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) : 
  k = Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_k_value_l2691_269172


namespace NUMINAMATH_CALUDE_rotate_point_A_about_C_l2691_269164

-- Define the rotation function
def rotate90ClockwiseAboutPoint (p center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (cx + (y - cy), cy - (x - cx))

-- Theorem statement
theorem rotate_point_A_about_C :
  let A : ℝ × ℝ := (-3, 2)
  let C : ℝ × ℝ := (-2, 2)
  rotate90ClockwiseAboutPoint A C = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_rotate_point_A_about_C_l2691_269164


namespace NUMINAMATH_CALUDE_ursula_shopping_cost_l2691_269166

/-- Represents the prices of items in Ursula's shopping trip -/
structure ShoppingPrices where
  butter : ℝ
  bread : ℝ
  cheese : ℝ
  tea : ℝ
  eggs : ℝ
  honey : ℝ

/-- Calculates the total cost of all items -/
def totalCost (prices : ShoppingPrices) : ℝ :=
  prices.butter + prices.bread + prices.cheese + prices.tea + prices.eggs + prices.honey

/-- Theorem stating the conditions and the result of Ursula's shopping trip -/
theorem ursula_shopping_cost (prices : ShoppingPrices) : 
  prices.bread = prices.butter / 2 →
  prices.butter = 0.8 * prices.cheese →
  prices.tea = 1.5 * (prices.bread + prices.butter + prices.cheese) →
  prices.tea = 10 →
  prices.eggs = prices.bread / 2 →
  prices.honey = prices.eggs + 3 →
  abs (totalCost prices - 20.87) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ursula_shopping_cost_l2691_269166


namespace NUMINAMATH_CALUDE_interest_problem_l2691_269182

/-- Given a sum P at simple interest rate R for 3 years, if increasing the rate by 8%
    results in Rs. 120 more interest, then P = 500. -/
theorem interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  (P * (R + 8) * 3) / 100 = (P * R * 3) / 100 + 120 →
  P = 500 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l2691_269182


namespace NUMINAMATH_CALUDE_john_used_four_quarters_l2691_269154

/-- The number of quarters John used to pay for a candy bar -/
def quarters_used (candy_cost dime_value nickel_value quarter_value : ℕ) 
  (num_dimes : ℕ) (change : ℕ) : ℕ :=
  ((candy_cost + change) - (num_dimes * dime_value + nickel_value)) / quarter_value

/-- Theorem stating that John used 4 quarters to pay for the candy bar -/
theorem john_used_four_quarters :
  quarters_used 131 10 5 25 3 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_john_used_four_quarters_l2691_269154


namespace NUMINAMATH_CALUDE_percentage_of_red_shirts_l2691_269179

theorem percentage_of_red_shirts 
  (total_students : ℕ) 
  (blue_percentage : ℚ) 
  (green_percentage : ℚ) 
  (other_colors : ℕ) 
  (h1 : total_students = 600) 
  (h2 : blue_percentage = 45/100) 
  (h3 : green_percentage = 15/100) 
  (h4 : other_colors = 102) :
  (total_students - (blue_percentage * total_students + green_percentage * total_students + other_colors)) / total_students = 23/100 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_red_shirts_l2691_269179


namespace NUMINAMATH_CALUDE_tim_weekly_reading_time_l2691_269110

/-- Tim's daily meditation time in hours -/
def daily_meditation_time : ℝ := 1

/-- Tim's daily reading time in hours -/
def daily_reading_time : ℝ := 2 * daily_meditation_time

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Tim's weekly reading time in hours -/
def weekly_reading_time : ℝ := daily_reading_time * days_in_week

theorem tim_weekly_reading_time :
  weekly_reading_time = 14 := by sorry

end NUMINAMATH_CALUDE_tim_weekly_reading_time_l2691_269110


namespace NUMINAMATH_CALUDE_zero_point_existence_l2691_269109

def f (x : ℝ) : ℝ := -x^3 - 3*x + 5

theorem zero_point_existence :
  ∃ x₀ ∈ Set.Ioo 1 2, f x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_point_existence_l2691_269109


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l2691_269189

theorem complex_roots_theorem (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (a + 5 * Complex.I) * (b + 6 * Complex.I) = 9 + 61 * Complex.I →
  (a + 5 * Complex.I) + (b + 6 * Complex.I) = 12 + 11 * Complex.I →
  (a, b) = (9, 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l2691_269189


namespace NUMINAMATH_CALUDE_largest_perimeter_incenter_l2691_269106

/-- A triangle in a plane with a fixed point P --/
structure TriangleWithFixedPoint where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ

/-- Distance between two points in a plane --/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Perimeter of a triangle --/
def perimeter (t : TriangleWithFixedPoint) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Predicate to check if a point is the incenter of a triangle --/
def is_incenter (t : TriangleWithFixedPoint) : Prop := sorry

/-- Theorem: Triangles with largest perimeter have P as incenter --/
theorem largest_perimeter_incenter (t : TriangleWithFixedPoint) 
  (h1 : distance t.P t.A = 3)
  (h2 : distance t.P t.B = 5)
  (h3 : distance t.P t.C = 7) :
  (∀ t' : TriangleWithFixedPoint, 
    distance t'.P t'.A = 3 → 
    distance t'.P t'.B = 5 → 
    distance t'.P t'.C = 7 → 
    perimeter t ≥ perimeter t') ↔ 
  is_incenter t := by sorry

end NUMINAMATH_CALUDE_largest_perimeter_incenter_l2691_269106


namespace NUMINAMATH_CALUDE_baseball_card_value_l2691_269125

def initialValue : ℝ := 100

def yearlyChanges : List ℝ := [-0.10, 0.12, -0.08, 0.05, -0.07]

def applyChange (value : ℝ) (change : ℝ) : ℝ := value * (1 + change)

def finalValue : ℝ := yearlyChanges.foldl applyChange initialValue

theorem baseball_card_value : 
  ∃ ε > 0, |finalValue - 90.56| < ε :=
sorry

end NUMINAMATH_CALUDE_baseball_card_value_l2691_269125


namespace NUMINAMATH_CALUDE_heather_oranges_l2691_269152

theorem heather_oranges (initial : Real) (received : Real) :
  initial = 60.0 → received = 35.0 → initial + received = 95.0 := by
  sorry

end NUMINAMATH_CALUDE_heather_oranges_l2691_269152


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2691_269188

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x + y + z = 3) :
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) > 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2691_269188


namespace NUMINAMATH_CALUDE_reciprocal_of_2022_l2691_269194

theorem reciprocal_of_2022 : (2022⁻¹ : ℝ) = 1 / 2022 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_2022_l2691_269194


namespace NUMINAMATH_CALUDE_expected_value_is_correct_l2691_269150

def number_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_colored (n : ℕ) : Bool := sorry

def is_red (n : ℕ) : Bool := sorry

def is_blue (n : ℕ) : Bool := sorry

def probability_red : ℚ := 1/2

def probability_blue : ℚ := 1/2

def is_sum_of_red_and_blue (n : ℕ) : Bool := sorry

def expected_value : ℚ := sorry

theorem expected_value_is_correct : expected_value = 423/32 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_correct_l2691_269150


namespace NUMINAMATH_CALUDE_boys_from_beethoven_l2691_269171

/-- Given the following conditions about a music camp:
  * There are 120 total students
  * There are 65 boys and 55 girls
  * 50 students are from Mozart Middle School
  * 70 students are from Beethoven Middle School
  * 17 girls are from Mozart Middle School
  This theorem proves that there are 32 boys from Beethoven Middle School -/
theorem boys_from_beethoven (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (mozart_students : ℕ) (beethoven_students : ℕ) (mozart_girls : ℕ) :
  total_students = 120 →
  total_boys = 65 →
  total_girls = 55 →
  mozart_students = 50 →
  beethoven_students = 70 →
  mozart_girls = 17 →
  beethoven_students - (beethoven_students - total_boys + mozart_students - mozart_girls) = 32 :=
by sorry

end NUMINAMATH_CALUDE_boys_from_beethoven_l2691_269171


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2691_269123

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 6*x + k = (x + a)^2) → k = 9 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2691_269123


namespace NUMINAMATH_CALUDE_student_count_l2691_269141

theorem student_count (initial_avg : ℝ) (incorrect_height : ℝ) (actual_height : ℝ) (actual_avg : ℝ) :
  initial_avg = 175 →
  incorrect_height = 151 →
  actual_height = 136 →
  actual_avg = 174.5 →
  ∃ n : ℕ, n = 30 ∧ n * actual_avg = n * initial_avg - (incorrect_height - actual_height) :=
by sorry

end NUMINAMATH_CALUDE_student_count_l2691_269141


namespace NUMINAMATH_CALUDE_roots_sum_greater_than_a_l2691_269101

noncomputable section

-- Define the function f(x) = x ln x
def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := Real.log x + 1

-- Define the function F(x) = x^2 - a[x + f'(x)] + 2x
def F (a : ℝ) (x : ℝ) : ℝ := x^2 - a * (x + f' x) + 2 * x

-- Theorem statement
theorem roots_sum_greater_than_a (a m x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂)
  (h₂ : F a x₁ = m)
  (h₃ : F a x₂ = m)
  : x₁ + x₂ > a :=
sorry

end

end NUMINAMATH_CALUDE_roots_sum_greater_than_a_l2691_269101


namespace NUMINAMATH_CALUDE_bicycle_price_after_discounts_l2691_269127

def original_price : ℝ := 200
def first_discount : ℝ := 0.4
def second_discount : ℝ := 0.25

theorem bicycle_price_after_discounts :
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 90 := by sorry

end NUMINAMATH_CALUDE_bicycle_price_after_discounts_l2691_269127


namespace NUMINAMATH_CALUDE_library_book_count_l2691_269134

/-- Given a library with shelves that each hold a fixed number of books,
    calculate the total number of books in the library. -/
def total_books (num_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  num_shelves * books_per_shelf

/-- Theorem stating that a library with 1780 shelves, each holding 8 books,
    contains 14240 books in total. -/
theorem library_book_count : total_books 1780 8 = 14240 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l2691_269134


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2691_269143

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) + (x - 3)^2 = 3 + k * x) ↔ 
  (k = -3 + 2 * Real.sqrt 10 ∨ k = -3 - 2 * Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2691_269143


namespace NUMINAMATH_CALUDE_waiter_net_earnings_waiter_earnings_result_l2691_269148

/-- Calculates the waiter's net earnings from tips after commission --/
theorem waiter_net_earnings (customers : Nat) 
  (tipping_customers : Nat)
  (bill1 bill2 bill3 bill4 : ℝ)
  (tip_percent1 tip_percent2 tip_percent3 tip_percent4 : ℝ)
  (commission_rate : ℝ) : ℝ :=
  let total_tips := 
    bill1 * tip_percent1 + 
    bill2 * tip_percent2 + 
    bill3 * tip_percent3 + 
    bill4 * tip_percent4
  let commission := total_tips * commission_rate
  let net_earnings := total_tips - commission
  net_earnings

/-- The waiter's net earnings are approximately $16.82 --/
theorem waiter_earnings_result : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |waiter_net_earnings 9 4 25 22 35 30 0.15 0.18 0.20 0.10 0.05 - 16.82| < ε :=
sorry

end NUMINAMATH_CALUDE_waiter_net_earnings_waiter_earnings_result_l2691_269148


namespace NUMINAMATH_CALUDE_trig_identity_l2691_269128

theorem trig_identity : Real.cos (70 * π / 180) * Real.sin (50 * π / 180) - 
                        Real.cos (200 * π / 180) * Real.sin (40 * π / 180) = 
                        Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2691_269128


namespace NUMINAMATH_CALUDE_vessel_volume_ratio_l2691_269113

theorem vessel_volume_ratio : 
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (5/8 : ℝ) * V₂ →
  V₁ / V₂ = 5/6 := by
sorry

end NUMINAMATH_CALUDE_vessel_volume_ratio_l2691_269113


namespace NUMINAMATH_CALUDE_item_list_price_equality_l2691_269176

theorem item_list_price_equality (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) → list_price = 40 := by
  sorry

#check item_list_price_equality

end NUMINAMATH_CALUDE_item_list_price_equality_l2691_269176


namespace NUMINAMATH_CALUDE_inequality_proof_l2691_269139

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : 1/b - 1/a > 1) :
  Real.sqrt (1 + a) > 1 / Real.sqrt (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2691_269139


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2691_269183

/-- Given a quadratic function f(x) = ax² + bx + c passing through specific points,
    prove that g(x) = cx² + 2bx + a has a specific vertex form -/
theorem quadratic_transformation (a b c : ℝ) 
  (h1 : c = 1)
  (h2 : a + b + c = -2)
  (h3 : a - b + c = 2) :
  let f := fun x => a * x^2 + b * x + c
  let g := fun x => c * x^2 + 2 * b * x + a
  ∀ x, g x = (x - 2)^2 - 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2691_269183


namespace NUMINAMATH_CALUDE_f_13_value_l2691_269191

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_13_value (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 4)
  (h_f_neg_one : f (-1) = 2) :
  f 13 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_13_value_l2691_269191


namespace NUMINAMATH_CALUDE_greatest_color_count_l2691_269137

theorem greatest_color_count (α β : ℝ) (h1 : 1 < α) (h2 : α < β) : 
  (∀ (r : ℕ), r > 2 → 
    ∃ (f : ℕ+ → Fin r), ∀ (x y : ℕ+), 
      f x = f y → (α : ℝ) ≤ (x : ℝ) / (y : ℝ) → (x : ℝ) / (y : ℝ) ≤ β → False) ∧
  (∀ (f : ℕ+ → Fin 2), ∃ (x y : ℕ+), 
    f x = f y ∧ (α : ℝ) ≤ (x : ℝ) / (y : ℝ) ∧ (x : ℝ) / (y : ℝ) ≤ β) :=
by sorry

end NUMINAMATH_CALUDE_greatest_color_count_l2691_269137


namespace NUMINAMATH_CALUDE_bacteria_in_seventh_generation_l2691_269132

/-- The number of bacteria in a given generation -/
def bacteria_count (generation : ℕ) : ℕ :=
  match generation with
  | 0 => 1  -- First generation
  | n + 1 => 4 * bacteria_count n  -- Subsequent generations

/-- Theorem stating the number of bacteria in the seventh generation -/
theorem bacteria_in_seventh_generation :
  bacteria_count 6 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_in_seventh_generation_l2691_269132


namespace NUMINAMATH_CALUDE_conference_married_men_fraction_l2691_269100

theorem conference_married_men_fraction 
  (total_women : ℕ) 
  (single_women : ℕ) 
  (married_women : ℕ) 
  (married_men : ℕ) 
  (h1 : single_women + married_women = total_women)
  (h2 : married_women = married_men)
  (h3 : (single_women : ℚ) / total_women = 3 / 7) :
  (married_men : ℚ) / (total_women + married_men) = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_conference_married_men_fraction_l2691_269100


namespace NUMINAMATH_CALUDE_class_vision_only_comprehensive_l2691_269186

-- Define the concept of a survey
structure Survey where
  population : Type
  data_collection : population → Bool

-- Define what makes a survey comprehensive
def is_comprehensive (s : Survey) : Prop :=
  ∀ x : s.population, s.data_collection x

-- Define the specific surveys
def bulb_survey : Survey := sorry
def class_vision_survey : Survey := sorry
def food_preservative_survey : Survey := sorry
def river_water_quality_survey : Survey := sorry

-- State the theorem
theorem class_vision_only_comprehensive :
  is_comprehensive class_vision_survey ∧
  ¬is_comprehensive bulb_survey ∧
  ¬is_comprehensive food_preservative_survey ∧
  ¬is_comprehensive river_water_quality_survey :=
sorry

end NUMINAMATH_CALUDE_class_vision_only_comprehensive_l2691_269186


namespace NUMINAMATH_CALUDE_speed_conversion_correct_l2691_269168

/-- Conversion factor from km/h to m/s -/
def kmh_to_ms : ℝ := 0.277778

/-- Given speed in km/h -/
def speed_kmh : ℝ := 84

/-- Equivalent speed in m/s -/
def speed_ms : ℝ := speed_kmh * kmh_to_ms

theorem speed_conversion_correct : 
  ∃ ε > 0, |speed_ms - 23.33| < ε :=
sorry

end NUMINAMATH_CALUDE_speed_conversion_correct_l2691_269168


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2691_269174

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 4*x + 3 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 - x - 3 = 0
  let sol1 : Set ℝ := {3, 1}
  let sol2 : Set ℝ := {(1 + Real.sqrt 13) / 2, (1 - Real.sqrt 13) / 2}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2691_269174


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2691_269147

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, (a * x) / (x - 1) < 1 ↔ (x < b ∨ x > 3)) →
  (a * 3) / (3 - 1) = 1 →
  a - b = -1/3 := by
    sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2691_269147


namespace NUMINAMATH_CALUDE_marys_income_percentage_l2691_269104

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.5) 
  (h2 : mary = tim * 1.6) : 
  mary = juan * 0.8 := by
sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l2691_269104


namespace NUMINAMATH_CALUDE_positive_integer_M_satisfying_equation_l2691_269162

theorem positive_integer_M_satisfying_equation : ∃ M : ℕ+, (12^2 * 60^2 : ℕ) = 30^2 * M^2 ∧ M = 12 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_M_satisfying_equation_l2691_269162


namespace NUMINAMATH_CALUDE_second_replaced_man_age_is_23_l2691_269122

/-- The age of the second replaced man in a group where:
  * There are 8 men initially
  * Two men are replaced
  * The average age increases by 2 years after replacement
  * One of the replaced men is 21 years old
  * The average age of the two new men is 30 years
-/
def second_replaced_man_age : ℕ := by
  -- Define the initial number of men
  let initial_count : ℕ := 8
  -- Define the age increase after replacement
  let age_increase : ℕ := 2
  -- Define the age of the first replaced man
  let first_replaced_age : ℕ := 21
  -- Define the average age of the new men
  let new_men_avg_age : ℕ := 30

  -- The actual proof would go here
  sorry

theorem second_replaced_man_age_is_23 : second_replaced_man_age = 23 := by
  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_second_replaced_man_age_is_23_l2691_269122


namespace NUMINAMATH_CALUDE_initial_solution_strength_l2691_269146

/-- Proves that the initial solution strength is 60% given the problem conditions --/
theorem initial_solution_strength 
  (initial_volume : ℝ)
  (drained_volume : ℝ)
  (replacement_strength : ℝ)
  (final_strength : ℝ)
  (h1 : initial_volume = 50)
  (h2 : drained_volume = 35)
  (h3 : replacement_strength = 40)
  (h4 : final_strength = 46)
  (h5 : initial_volume - drained_volume + drained_volume = initial_volume)
  (h6 : (initial_volume - drained_volume) * (initial_strength / 100) + 
        drained_volume * (replacement_strength / 100) = 
        initial_volume * (final_strength / 100)) :
  initial_strength = 60 := by
  sorry

#check initial_solution_strength

end NUMINAMATH_CALUDE_initial_solution_strength_l2691_269146


namespace NUMINAMATH_CALUDE_negation_is_true_l2691_269197

theorem negation_is_true : 
  (∀ x : ℝ, x^2 ≥ 1 → (x ≤ -1 ∨ x ≥ 1)) := by sorry

end NUMINAMATH_CALUDE_negation_is_true_l2691_269197


namespace NUMINAMATH_CALUDE_monotone_sine_function_l2691_269160

/-- The function f(x) = x + t*sin(2x) is monotonically increasing on ℝ if and only if t ∈ [-1/2, 1/2] -/
theorem monotone_sine_function (t : ℝ) :
  (∀ x : ℝ, Monotone (λ x => x + t * Real.sin (2 * x))) ↔ t ∈ Set.Icc (-1/2) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_monotone_sine_function_l2691_269160


namespace NUMINAMATH_CALUDE_route_count_is_70_l2691_269138

-- Define the grid structure
structure Grid :=
  (levels : Nat)
  (segments_between_levels : List Nat)

-- Define a route
def Route := List (Nat × Nat)

-- Function to check if a route is valid (doesn't intersect itself)
def is_valid_route (g : Grid) (r : Route) : Bool := sorry

-- Function to generate all possible routes
def all_routes (g : Grid) : List Route := sorry

-- Function to count valid routes
def count_valid_routes (g : Grid) : Nat :=
  (all_routes g).filter (is_valid_route g) |>.length

-- Define our specific grid
def our_grid : Grid :=
  { levels := 4,
    segments_between_levels := [3, 5, 3] }

-- Theorem statement
theorem route_count_is_70 :
  count_valid_routes our_grid = 70 := by sorry

end NUMINAMATH_CALUDE_route_count_is_70_l2691_269138


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l2691_269142

theorem opposite_sides_line_range (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ = -1 ∧ y₁ = 0 ∧ x₂ = 2 ∧ y₂ = -1 ∧
    (2 * x₁ + y₁ + a) * (2 * x₂ + y₂ + a) < 0) →
  -3 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l2691_269142


namespace NUMINAMATH_CALUDE_eliana_steps_theorem_l2691_269159

/-- The number of steps Eliana walked on the first day -/
def first_day_steps : ℕ := 200 + 300

/-- The number of steps Eliana walked on the second day -/
def second_day_steps : ℕ := (3 * first_day_steps) / 2

/-- The number of steps Eliana walked on the third day -/
def third_day_steps : ℕ := 2 * second_day_steps

/-- The total number of steps Eliana walked during the three days -/
def total_steps : ℕ := first_day_steps + second_day_steps + third_day_steps

theorem eliana_steps_theorem : total_steps = 2750 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_theorem_l2691_269159


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_700_by_75_percent_l2691_269120

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_700_by_75_percent :
  700 * (1 + 75 / 100) = 1225 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_700_by_75_percent_l2691_269120


namespace NUMINAMATH_CALUDE_valid_coloring_iff_odd_l2691_269158

/-- A coloring of edges and diagonals of an n-gon -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin n

/-- Predicate for a valid coloring -/
def is_valid_coloring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ i →
    ∃ (x y z : Fin n), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
      c x y = i ∧ c y z = j ∧ c z x = k

/-- Theorem: A valid coloring exists if and only if n is odd -/
theorem valid_coloring_iff_odd (n : ℕ) :
  (∃ c : Coloring n, is_valid_coloring n c) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_valid_coloring_iff_odd_l2691_269158


namespace NUMINAMATH_CALUDE_base_five_of_232_l2691_269185

def base_five_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_five_of_232 :
  base_five_repr 232 = [1, 4, 1, 2] := by
sorry

end NUMINAMATH_CALUDE_base_five_of_232_l2691_269185


namespace NUMINAMATH_CALUDE_fish_value_in_dragon_scales_l2691_269129

/-- In a magical kingdom with given exchange rates, prove the value of a fish in dragon scales -/
theorem fish_value_in_dragon_scales 
  (fish_to_bread : ℚ) -- Exchange rate of fish to bread
  (bread_to_scales : ℚ) -- Exchange rate of bread to dragon scales
  (h1 : 2 * fish_to_bread = 3) -- Two fish can be exchanged for three loaves of bread
  (h2 : bread_to_scales = 2) -- One loaf of bread can be traded for two dragon scales
  : fish_to_bread * bread_to_scales = 3 := by sorry

end NUMINAMATH_CALUDE_fish_value_in_dragon_scales_l2691_269129


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2691_269111

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 40) (h2 : x = 32) : x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2691_269111


namespace NUMINAMATH_CALUDE_largest_negative_congruent_to_one_mod_23_l2691_269157

theorem largest_negative_congruent_to_one_mod_23 : 
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 1 [ZMOD 23] → n ≤ -9994 :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_congruent_to_one_mod_23_l2691_269157


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l2691_269173

theorem triangle_area_inequality (a b c α β γ : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : α > 0) (h5 : β > 0) (h6 : γ > 0)
  (h7 : α = 2 * Real.sqrt (b * c))
  (h8 : β = 2 * Real.sqrt (c * a))
  (h9 : γ = 2 * Real.sqrt (a * b)) :
  a / α + b / β + c / γ ≥ 3 / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l2691_269173


namespace NUMINAMATH_CALUDE_packaging_combinations_l2691_269115

/-- The number of wrapping paper designs --/
def num_wrapping_paper : ℕ := 10

/-- The number of ribbon colors --/
def num_ribbons : ℕ := 4

/-- The number of gift card varieties --/
def num_gift_cards : ℕ := 5

/-- The number of decorative sticker styles --/
def num_stickers : ℕ := 6

/-- The total number of unique packaging combinations --/
def total_combinations : ℕ := num_wrapping_paper * num_ribbons * num_gift_cards * num_stickers

/-- Theorem stating that the total number of unique packaging combinations is 1200 --/
theorem packaging_combinations : total_combinations = 1200 := by
  sorry

end NUMINAMATH_CALUDE_packaging_combinations_l2691_269115


namespace NUMINAMATH_CALUDE_range_of_m_l2691_269161

-- Define the propositions p and q
def p (m : ℝ) : Prop := (1 + 1 - 2*m + 2*m + 2*m^2 - 4) < 0

def q (m : ℝ) : Prop := m ≥ 0 ∧ 2*m + 1 ≥ 0

-- Define the theorem
theorem range_of_m :
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (∀ m : ℝ, (-1 < m ∧ m < 0) ∨ m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2691_269161


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l2691_269144

theorem cost_increase_percentage (cost selling_price : ℝ) (increase_factor : ℝ) : 
  cost > 0 →
  selling_price = cost * 2.6 →
  (selling_price - cost * (1 + increase_factor)) / selling_price = 0.5692307692307692 →
  increase_factor = 0.12 := by
sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l2691_269144


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_is_one_over_sqrt_two_l2691_269103

/-- A right-angled triangle with its height and inscribed circles -/
structure RightTriangleWithInscribedCircles where
  /-- The original right-angled triangle -/
  originalTriangle : Set (ℝ × ℝ)
  /-- The two triangles formed by the height -/
  subTriangle1 : Set (ℝ × ℝ)
  subTriangle2 : Set (ℝ × ℝ)
  /-- The center of the inscribed circle of subTriangle1 -/
  center1 : ℝ × ℝ
  /-- The center of the inscribed circle of subTriangle2 -/
  center2 : ℝ × ℝ
  /-- The distance between center1 and center2 is 1 -/
  centers_distance : Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2) = 1
  /-- The height divides the original triangle into subTriangle1 and subTriangle2 -/
  height_divides : originalTriangle = subTriangle1 ∪ subTriangle2
  /-- The original triangle is right-angled -/
  is_right_angled : ∃ (a b c : ℝ × ℝ), a ∈ originalTriangle ∧ b ∈ originalTriangle ∧ c ∈ originalTriangle ∧
    (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0

/-- The radius of the inscribed circle of the original triangle -/
def inscribed_circle_radius (t : RightTriangleWithInscribedCircles) : ℝ :=
  sorry

/-- Theorem: The radius of the inscribed circle of the original triangle is 1/√2 -/
theorem inscribed_circle_radius_is_one_over_sqrt_two (t : RightTriangleWithInscribedCircles) :
  inscribed_circle_radius t = 1 / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_is_one_over_sqrt_two_l2691_269103


namespace NUMINAMATH_CALUDE_simplify_expression_l2691_269169

theorem simplify_expression : 1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2691_269169


namespace NUMINAMATH_CALUDE_country_z_diploma_percentage_l2691_269116

theorem country_z_diploma_percentage :
  let total_population : ℝ := 100
  let no_diploma_job_choice : ℝ := 18
  let diploma_no_job_choice_ratio : ℝ := 0.25
  let job_choice : ℝ := 40

  let diploma_job_choice : ℝ := job_choice - no_diploma_job_choice
  let no_job_choice : ℝ := total_population - job_choice
  let diploma_no_job_choice : ℝ := diploma_no_job_choice_ratio * no_job_choice

  diploma_job_choice + diploma_no_job_choice = 37 :=
by sorry

end NUMINAMATH_CALUDE_country_z_diploma_percentage_l2691_269116


namespace NUMINAMATH_CALUDE_smallest_number_l2691_269187

theorem smallest_number : 
  ∀ (a b c : ℝ), a = -Real.sqrt 2 ∧ b = 3.14 ∧ c = 2021 → 
    a < 0 ∧ a < b ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2691_269187


namespace NUMINAMATH_CALUDE_adrian_days_off_l2691_269180

/-- The number of days Adrian took off in a year -/
def total_holidays : ℕ := 48

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of days Adrian took off each month -/
def days_off_per_month : ℕ := total_holidays / months_in_year

theorem adrian_days_off :
  days_off_per_month = 4 :=
by sorry

end NUMINAMATH_CALUDE_adrian_days_off_l2691_269180


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_values_solution_set_correct_l2691_269195

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = 3 ∧ p.1 ≠ 2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + 2 * p.2 + a = 0}

-- State the theorem
theorem intersection_empty_implies_a_values (a : ℝ) :
  M ∩ N a = ∅ → a = -6 ∨ a = -2 := by
  sorry

-- Define the solution set
def solution_set : Set ℝ := {-6, -2}

-- State the theorem for the solution set
theorem solution_set_correct :
  ∀ a : ℝ, (M ∩ N a = ∅) → a ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_values_solution_set_correct_l2691_269195


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l2691_269156

/-- Given the teaching years of Virginia, Adrienne, and Dennis, prove that Dennis has taught for 46 years. -/
theorem dennis_teaching_years 
  (total : ℕ) 
  (h_total : total = 102)
  (h_virginia_adrienne : ∃ (a : ℕ), virginia = a + 9)
  (h_virginia_dennis : ∃ (d : ℕ), virginia = d - 9)
  (h_sum : virginia + adrienne + dennis = total)
  : dennis = 46 := by
  sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l2691_269156


namespace NUMINAMATH_CALUDE_nina_weekend_earnings_l2691_269140

/-- Calculates the total money made from jewelry sales --/
def total_money_made (necklace_price bracelet_price earring_pair_price ensemble_price : ℚ)
                     (necklaces_sold bracelets_sold earrings_sold ensembles_sold : ℕ) : ℚ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * (earrings_sold / 2) +
  ensemble_price * ensembles_sold

/-- Theorem: Nina's weekend earnings --/
theorem nina_weekend_earnings :
  let necklace_price : ℚ := 25
  let bracelet_price : ℚ := 15
  let earring_pair_price : ℚ := 10
  let ensemble_price : ℚ := 45
  let necklaces_sold : ℕ := 5
  let bracelets_sold : ℕ := 10
  let earrings_sold : ℕ := 20
  let ensembles_sold : ℕ := 2
  total_money_made necklace_price bracelet_price earring_pair_price ensemble_price
                    necklaces_sold bracelets_sold earrings_sold ensembles_sold = 465 :=
by
  sorry

end NUMINAMATH_CALUDE_nina_weekend_earnings_l2691_269140


namespace NUMINAMATH_CALUDE_keychain_manufacturing_cost_l2691_269108

theorem keychain_manufacturing_cost 
  (P : ℝ) -- Selling price
  (h1 : P > 0) -- Selling price is positive
  (h2 : P - 0.5 * P = 50) -- New manufacturing cost is $50
  : P - 0.4 * P = 60 := by
  sorry

end NUMINAMATH_CALUDE_keychain_manufacturing_cost_l2691_269108


namespace NUMINAMATH_CALUDE_intersection_and_system_solution_l2691_269131

theorem intersection_and_system_solution :
  ∀ (m n : ℝ),
  (∃ (x y : ℝ), y = -x + 4 ∧ y = 2*x + m ∧ x = 3 ∧ y = n) →
  (∀ (x y : ℝ), x + y - 4 = 0 ∧ 2*x - y + m = 0 ↔ x = 3 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_system_solution_l2691_269131


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2691_269118

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set
def solution_set (b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

-- Main theorem
theorem quadratic_inequality_theorem (a b : ℝ) (h : ∀ x, f a x > 0 ↔ x ∈ solution_set b) :
  a = 1 ∧ b = 2 ∧
  (∀ c : ℝ, 
    (c > 2 → {x | x^2 - (c+2)*x + 2*c < 0} = {x | 2 < x ∧ x < c}) ∧
    (c < 2 → {x | x^2 - (c+2)*x + 2*c < 0} = {x | c < x ∧ x < 2}) ∧
    (c = 2 → {x | x^2 - (c+2)*x + 2*c < 0} = ∅)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2691_269118


namespace NUMINAMATH_CALUDE_trigonometric_equation_equivalence_l2691_269196

theorem trigonometric_equation_equivalence (α : ℝ) : 
  (1 - 2 * (Real.cos α) ^ 2) / (2 * Real.tan (2 * α - π / 4) * (Real.sin (π / 4 + 2 * α)) ^ 2) = 
  -(Real.cos (2 * α)) / ((Real.cos (2 * α - π / 4) + Real.sin (2 * α - π / 4)) ^ 2) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_equivalence_l2691_269196


namespace NUMINAMATH_CALUDE_jerry_piercing_earnings_l2691_269178

theorem jerry_piercing_earnings :
  let nose_price : ℚ := 20
  let ear_price : ℚ := nose_price * (1 + 1/2)
  let nose_count : ℕ := 6
  let ear_count : ℕ := 9
  nose_price * nose_count + ear_price * ear_count = 390 := by
  sorry

end NUMINAMATH_CALUDE_jerry_piercing_earnings_l2691_269178


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l2691_269149

theorem smallest_of_five_consecutive_even_numbers (x : ℤ) : 
  (∀ i : ℕ, i < 5 → 2 ∣ (x + 2*i)) →  -- x and the next 4 numbers are even
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 200) →  -- sum is 200
  x = 36 :=  -- smallest number is 36
by
  sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l2691_269149


namespace NUMINAMATH_CALUDE_sine_cosine_acute_less_than_one_l2691_269112

-- Define an acute angle
def is_acute_angle (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define sine and cosine for an acute angle in a right-angled triangle
def sine_acute (α : Real) (h : is_acute_angle α) : Real := sorry
def cosine_acute (α : Real) (h : is_acute_angle α) : Real := sorry

-- Theorem statement
theorem sine_cosine_acute_less_than_one (α : Real) (h : is_acute_angle α) :
  sine_acute α h < 1 ∧ cosine_acute α h < 1 := by sorry

end NUMINAMATH_CALUDE_sine_cosine_acute_less_than_one_l2691_269112


namespace NUMINAMATH_CALUDE_f_derivative_l2691_269155

noncomputable def f (x : ℝ) : ℝ := x * Real.cos (2 * x)

theorem f_derivative : 
  deriv f = fun x => Real.cos (2 * x) - 2 * x * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l2691_269155


namespace NUMINAMATH_CALUDE_angle_ratio_equality_l2691_269184

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define a point P inside the triangle
def PointInside (t : Triangle) (P : Point) : Prop := sorry

-- Define angle measure
def AngleMeasure (A B C : Point) : ℝ := sorry

-- Theorem statement
theorem angle_ratio_equality (t : Triangle) (P : Point) (x : ℝ) 
  (h_inside : PointInside t P)
  (h_ratio_AB_AC : AngleMeasure t.A P t.B / AngleMeasure t.A P t.C = x)
  (h_ratio_CA_CB : AngleMeasure t.C P t.A / AngleMeasure t.C P t.B = x)
  (h_ratio_BC_BA : AngleMeasure t.B P t.C / AngleMeasure t.B P t.A = x) :
  x = 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_ratio_equality_l2691_269184


namespace NUMINAMATH_CALUDE_cos_sum_min_value_l2691_269133

theorem cos_sum_min_value (x : ℝ) : |Real.cos x| + |Real.cos (2 * x)| ≥ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_min_value_l2691_269133


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l2691_269193

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) :
  let original_area := 6 * s^2
  let new_edge := 1.6 * s
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 1.56 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l2691_269193


namespace NUMINAMATH_CALUDE_min_quotient_value_l2691_269167

def is_valid_number (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧
  1 ≤ b ∧ b ≤ 9 ∧
  1 ≤ c ∧ c ≤ 9 ∧
  1 ≤ d ∧ d ≤ 9 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a = c + 1 ∧
  b = d + 1

def number_value (a b c d : ℕ) : ℕ :=
  1000 * a + 100 * b + 10 * c + d

def digit_sum (a b c d : ℕ) : ℕ :=
  a + b + c + d

def quotient (a b c d : ℕ) : ℚ :=
  (number_value a b c d : ℚ) / (digit_sum a b c d : ℚ)

theorem min_quotient_value :
  ∀ a b c d : ℕ, is_valid_number a b c d →
  quotient a b c d ≥ 192.67 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_value_l2691_269167


namespace NUMINAMATH_CALUDE_fermat_number_prime_factor_l2691_269151

def F (n : ℕ) : ℕ := 2^(2^n) + 1

theorem fermat_number_prime_factor (n : ℕ) (h : n ≥ 3) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ F n ∧ p > 2^(n+2) * (n+1) := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_prime_factor_l2691_269151


namespace NUMINAMATH_CALUDE_largest_whole_number_solution_l2691_269126

theorem largest_whole_number_solution : 
  (∀ n : ℕ, n > 3 → ¬(1/4 + n/5 < 9/10)) ∧ 
  (1/4 + 3/5 < 9/10) := by
sorry

end NUMINAMATH_CALUDE_largest_whole_number_solution_l2691_269126


namespace NUMINAMATH_CALUDE_installation_cost_is_6255_l2691_269177

/-- Calculates the installation cost for a refrigerator purchase --/
def calculate_installation_cost (purchase_price : ℚ) (discount_rate : ℚ) 
  (transport_cost : ℚ) (profit_rate : ℚ) (selling_price : ℚ) : ℚ :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let total_cost := selling_price / (1 + profit_rate)
  total_cost - purchase_price - transport_cost

/-- Proves that the installation cost is 6255, given the problem conditions --/
theorem installation_cost_is_6255 :
  calculate_installation_cost 12500 0.20 125 0.18 18880 = 6255 := by
  sorry

end NUMINAMATH_CALUDE_installation_cost_is_6255_l2691_269177


namespace NUMINAMATH_CALUDE_value_of_expression_l2691_269107

theorem value_of_expression (a b c d : ℝ) 
  (h1 : a - b = 3) 
  (h2 : c + d = 2) : 
  (a + c) - (b - d) = 5 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l2691_269107


namespace NUMINAMATH_CALUDE_profit_maximized_at_100_l2691_269135

/-- The profit function L(x) for annual production x (in thousand units) -/
noncomputable def L (x : ℝ) : ℝ :=
  if x < 80 then
    -1/3 * x^2 + 40 * x - 250
  else
    1200 - (x + 10000 / x)

/-- Annual fixed cost in ten thousand yuan -/
def annual_fixed_cost : ℝ := 250

/-- Price per unit in ten thousand yuan -/
def price_per_unit : ℝ := 50

theorem profit_maximized_at_100 :
  ∀ x > 0, L x ≤ L 100 :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_100_l2691_269135


namespace NUMINAMATH_CALUDE_parallel_planes_distance_equivalence_l2691_269114

-- Define the types for planes and points
variable (Plane Point : Type)

-- Define the distance function between planes
variable (distance : Plane → Plane → ℝ)

-- Define the length function between points
variable (length : Point → Point → ℝ)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the intersection of a line with a plane
variable (intersect : Plane → Point)

-- Given three parallel planes
variable (α₁ α₂ α₃ : Plane)
variable (h_parallel : parallel α₁ α₂ ∧ parallel α₂ α₃ ∧ parallel α₁ α₃)

-- Define the distances between planes
variable (d₁ d₂ : ℝ)
variable (h_d₁ : distance α₁ α₂ = d₁)
variable (h_d₂ : distance α₂ α₃ = d₂)

-- Define the intersection points
variable (P₁ P₂ P₃ : Point)
variable (h_P₁ : P₁ = intersect α₁)
variable (h_P₂ : P₂ = intersect α₂)
variable (h_P₃ : P₃ = intersect α₃)

-- State the theorem
theorem parallel_planes_distance_equivalence :
  (length P₁ P₂ = length P₂ P₃) ↔ (d₁ = d₂) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_distance_equivalence_l2691_269114


namespace NUMINAMATH_CALUDE_sum_of_roots_l2691_269136

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1716 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2691_269136


namespace NUMINAMATH_CALUDE_spherical_to_cartesian_coordinates_l2691_269181

/-- Given a point M with spherical coordinates (1, π/3, π/6), 
    prove that its Cartesian coordinates are (3/4, √3/4, 1/2). -/
theorem spherical_to_cartesian_coordinates :
  let r : ℝ := 1
  let θ : ℝ := π / 3
  let φ : ℝ := π / 6
  let x : ℝ := r * Real.sin θ * Real.cos φ
  let y : ℝ := r * Real.sin θ * Real.sin φ
  let z : ℝ := r * Real.cos θ
  (x = 3/4) ∧ (y = Real.sqrt 3 / 4) ∧ (z = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_spherical_to_cartesian_coordinates_l2691_269181


namespace NUMINAMATH_CALUDE_equation_solution_l2691_269119

theorem equation_solution : ∃ x : ℝ, (2 * x + 6) / (x - 3) = 4 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2691_269119


namespace NUMINAMATH_CALUDE_smokers_percentage_is_five_percent_l2691_269105

/-- Represents the survey setup and results -/
structure SurveyData where
  total_students : ℕ
  white_balls : ℕ
  red_balls : ℕ
  stones_in_box : ℕ

/-- Calculates the estimated percentage of smokers based on the survey data -/
def estimate_smokers_percentage (data : SurveyData) : ℚ :=
  let total_balls := data.white_balls + data.red_balls
  let prob_question1 := data.white_balls / total_balls
  let expected_yes_question1 := data.total_students * prob_question1 * (1 / 2)
  let smokers := data.stones_in_box - expected_yes_question1
  let students_answering_question2 := data.total_students * (data.red_balls / total_balls)
  (smokers / students_answering_question2) * 100

/-- The main theorem stating that given the survey conditions, 
    the estimated percentage of smokers is 5% -/
theorem smokers_percentage_is_five_percent 
  (data : SurveyData) 
  (h1 : data.total_students = 200)
  (h2 : data.white_balls = 5)
  (h3 : data.red_balls = 5)
  (h4 : data.stones_in_box = 55) :
  estimate_smokers_percentage data = 5 := by
  sorry


end NUMINAMATH_CALUDE_smokers_percentage_is_five_percent_l2691_269105


namespace NUMINAMATH_CALUDE_function_value_at_two_l2691_269145

/-- Given a function f(x) = ax^5 + bx^3 - x + 2 where a and b are constants,
    and f(-2) = 5, prove that f(2) = -1 -/
theorem function_value_at_two
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^5 + b * x^3 - x + 2)
  (h2 : f (-2) = 5) :
  f 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l2691_269145


namespace NUMINAMATH_CALUDE_impossible_d_greater_than_c_l2691_269170

/-- A decreasing function on positive reals -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f y < f x

theorem impossible_d_greater_than_c
  (f : ℝ → ℝ) (a b c d : ℝ)
  (h_dec : DecreasingFunction f)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_prod : f a * f b * f c < 0)
  (h_d : f d = 0) :
  ¬(d > c) := by
sorry

end NUMINAMATH_CALUDE_impossible_d_greater_than_c_l2691_269170


namespace NUMINAMATH_CALUDE_marks_trees_l2691_269124

theorem marks_trees (initial_trees : ℕ) (planted_trees : ℕ) : 
  initial_trees = 13 → planted_trees = 12 → initial_trees + planted_trees = 25 := by
  sorry

end NUMINAMATH_CALUDE_marks_trees_l2691_269124


namespace NUMINAMATH_CALUDE_total_apples_buyable_l2691_269175

def apple_cost : ℕ := 2
def emmy_money : ℕ := 200
def gerry_money : ℕ := 100

theorem total_apples_buyable : 
  (emmy_money + gerry_money) / apple_cost = 150 := by
sorry

end NUMINAMATH_CALUDE_total_apples_buyable_l2691_269175


namespace NUMINAMATH_CALUDE_prob_at_least_three_heads_is_half_l2691_269199

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The probability of getting at least three heads when flipping five coins -/
def prob_at_least_three_heads : ℚ := 1/2

/-- Theorem stating that the probability of getting at least three heads 
    when flipping five coins simultaneously is 1/2 -/
theorem prob_at_least_three_heads_is_half : 
  prob_at_least_three_heads = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_three_heads_is_half_l2691_269199


namespace NUMINAMATH_CALUDE_expression_equals_six_l2691_269190

-- Define the expression
def expression : ℚ := 3 * (3 + 3) / 3

-- Theorem statement
theorem expression_equals_six : expression = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_six_l2691_269190


namespace NUMINAMATH_CALUDE_maurice_job_search_l2691_269198

/-- The probability of a single application being accepted -/
def p_accept : ℚ := 1 / 5

/-- The probability threshold for stopping -/
def p_threshold : ℚ := 3 / 4

/-- The number of letters Maurice needs to write -/
def num_letters : ℕ := 7

theorem maurice_job_search :
  (1 - (1 - p_accept) ^ num_letters) ≥ p_threshold ∧
  ∀ n : ℕ, n < num_letters → (1 - (1 - p_accept) ^ n) < p_threshold :=
by sorry

end NUMINAMATH_CALUDE_maurice_job_search_l2691_269198


namespace NUMINAMATH_CALUDE_product_in_N_l2691_269130

-- Define set M
def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n + 1}

-- Define set N
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n - 1}

-- Theorem statement
theorem product_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  sorry

end NUMINAMATH_CALUDE_product_in_N_l2691_269130
