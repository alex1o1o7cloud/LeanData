import Mathlib

namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_l242_24234

/-- Given a cubic polynomial Q with specific values at 1, -1, and 0,
    prove that Q(3) + Q(-3) = 47m -/
theorem cubic_polynomial_sum (m : ℝ) (Q : ℝ → ℝ) 
  (h_cubic : ∃ (a b c : ℝ), ∀ x, Q x = a * x^3 + b * x^2 + c * x + m)
  (h_1 : Q 1 = 3 * m)
  (h_neg1 : Q (-1) = 4 * m)
  (h_0 : Q 0 = m) :
  Q 3 + Q (-3) = 47 * m := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_l242_24234


namespace NUMINAMATH_CALUDE_x4_coefficient_zero_l242_24206

theorem x4_coefficient_zero (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x, (x^2 + a*x + 1) * (-6*x^3) = -6*x^5 + f x * x^4 + -6*x^3) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_x4_coefficient_zero_l242_24206


namespace NUMINAMATH_CALUDE_probability_one_match_l242_24228

/-- Represents the two topics that can be chosen. -/
inductive Topic : Type
  | A : Topic
  | B : Topic

/-- Represents a selection of topics by the three teachers. -/
def Selection := Topic × Topic × Topic

/-- The set of all possible selections. -/
def allSelections : Finset Selection := sorry

/-- Predicate for selections where exactly one male and the female choose the same topic. -/
def exactlyOneMatch (s : Selection) : Prop := sorry

/-- The set of selections where exactly one male and the female choose the same topic. -/
def matchingSelections : Finset Selection := sorry

/-- Theorem stating that the probability of exactly one male and the female choosing the same topic is 1/2. -/
theorem probability_one_match :
  (matchingSelections.card : ℚ) / allSelections.card = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_one_match_l242_24228


namespace NUMINAMATH_CALUDE_book_price_increase_l242_24242

theorem book_price_increase (original_price : ℝ) (increase_percentage : ℝ) (new_price : ℝ) :
  original_price = 300 →
  increase_percentage = 50 →
  new_price = original_price + (increase_percentage / 100) * original_price →
  new_price = 450 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l242_24242


namespace NUMINAMATH_CALUDE_f_monotone_increasing_implies_a_bound_l242_24272

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 2 * Real.sin x * Real.cos x + a * Real.cos x

def monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem f_monotone_increasing_implies_a_bound :
  ∀ a : ℝ, monotone_increasing (f a) (π/4) (3*π/4) → a ≤ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_implies_a_bound_l242_24272


namespace NUMINAMATH_CALUDE_vector_addition_l242_24276

/-- Given two vectors AB and BC in 2D space, prove that AC is their sum. -/
theorem vector_addition (AB BC : ℝ × ℝ) (h1 : AB = (2, 3)) (h2 : BC = (1, -4)) :
  AB.1 + BC.1 = 3 ∧ AB.2 + BC.2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l242_24276


namespace NUMINAMATH_CALUDE_max_n_value_l242_24291

theorem max_n_value (a b c : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c) 
  (h3 : (a - b)⁻¹ + (b - c)⁻¹ ≥ n / (a - c)) : n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_n_value_l242_24291


namespace NUMINAMATH_CALUDE_num_clips_property_l242_24251

/-- The number of clips on a curtain rod after k halving steps -/
def num_clips (k : ℕ) : ℕ :=
  2^(k-1) + 1

/-- The property that each interval has a middle clip -/
def has_middle_clip (n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → ∃ j : ℕ, j < n ∧ j > i ∧ j - i = (n - i) / 2

/-- The theorem stating that num_clips satisfies the middle clip property for all steps -/
theorem num_clips_property (k : ℕ) : 
  k > 0 → has_middle_clip (num_clips k) :=
sorry

end NUMINAMATH_CALUDE_num_clips_property_l242_24251


namespace NUMINAMATH_CALUDE_count_cow_herds_l242_24256

/-- Given a farm with cows organized into herds, this theorem proves
    the number of herds given the total number of cows and the number
    of cows per herd. -/
theorem count_cow_herds (total_cows : ℕ) (cows_per_herd : ℕ) 
    (h1 : total_cows = 320) (h2 : cows_per_herd = 40) :
    total_cows / cows_per_herd = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_cow_herds_l242_24256


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l242_24290

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) * (1 - 2 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l242_24290


namespace NUMINAMATH_CALUDE_total_books_read_is_72sc_l242_24211

/-- Calculates the total number of books read by a school's student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month : ℕ := 6
  let months_per_year : ℕ := 12
  let books_per_student_per_year : ℕ := books_per_month * months_per_year
  let total_students : ℕ := c * s
  books_per_student_per_year * total_students

/-- Theorem stating that the total number of books read by the entire student body in one year is 72sc -/
theorem total_books_read_is_72sc (c s : ℕ) : total_books_read c s = 72 * c * s := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_is_72sc_l242_24211


namespace NUMINAMATH_CALUDE_champion_is_team_d_l242_24257

-- Define the teams
inductive Team : Type
| A | B | C | D

-- Define the rankings
structure Ranking :=
(first : Team)
(second : Team)
(third : Team)
(fourth : Team)

-- Define the predictions
structure Prediction :=
(first : Option Team)
(second : Option Team)
(third : Option Team)
(fourth : Option Team)

-- Define the function to check if a prediction is half correct
def isHalfCorrect (pred : Prediction) (actual : Ranking) : Prop :=
  (pred.first = some actual.first ∨ pred.second = some actual.second ∨ 
   pred.third = some actual.third ∨ pred.fourth = some actual.fourth) ∧
  (pred.first ≠ some actual.first ∨ pred.second ≠ some actual.second ∨
   pred.third ≠ some actual.third ∨ pred.fourth ≠ some actual.fourth)

-- Theorem statement
theorem champion_is_team_d :
  ∀ (actual : Ranking),
    let wang_pred : Prediction := ⟨some Team.D, some Team.B, none, none⟩
    let li_pred : Prediction := ⟨none, some Team.A, none, some Team.C⟩
    let zhang_pred : Prediction := ⟨none, some Team.D, some Team.C, none⟩
    isHalfCorrect wang_pred actual ∧
    isHalfCorrect li_pred actual ∧
    isHalfCorrect zhang_pred actual →
    actual.first = Team.D :=
by sorry


end NUMINAMATH_CALUDE_champion_is_team_d_l242_24257


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l242_24269

/-- Given an arithmetic sequence with first term 5/8 and eleventh term 3/4,
    the sixth term is 11/16. -/
theorem sixth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℚ), 
    (∀ n m, a (n + m) - a n = m * (a 2 - a 1)) →  -- arithmetic sequence condition
    a 1 = 5/8 →                                   -- first term
    a 11 = 3/4 →                                  -- eleventh term
    a 6 = 11/16 :=                                -- sixth term
by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l242_24269


namespace NUMINAMATH_CALUDE_no_valid_a_for_quadratic_l242_24240

theorem no_valid_a_for_quadratic : ¬∃ (a : ℝ), 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + 2*(a+1)*x₁ - (a-1) = 0) ∧
  (x₂^2 + 2*(a+1)*x₂ - (a-1) = 0) ∧
  ((x₁ > 1 ∧ x₂ < 1) ∨ (x₁ < 1 ∧ x₂ > 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_a_for_quadratic_l242_24240


namespace NUMINAMATH_CALUDE_clothes_pricing_l242_24274

/-- Given a total spend and a price relation between shirt and trousers,
    prove the individual costs of the shirt and trousers. -/
theorem clothes_pricing (total : ℕ) (shirt_price trousers_price : ℕ) 
    (h1 : total = 185)
    (h2 : shirt_price = 2 * trousers_price + 5)
    (h3 : total = shirt_price + trousers_price) :
    trousers_price = 60 ∧ shirt_price = 125 := by
  sorry

end NUMINAMATH_CALUDE_clothes_pricing_l242_24274


namespace NUMINAMATH_CALUDE_vector_linear_combination_l242_24210

/-- Given vectors a, b, and c in R², prove that c can be expressed as a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) (hb : b = (1, -1)) (hc : c = (-1, 2)) :
  c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l242_24210


namespace NUMINAMATH_CALUDE_solid_figures_count_l242_24233

-- Define the list of shapes
inductive Shape
  | Circle
  | Square
  | Cone
  | Cuboid
  | LineSegment
  | Sphere
  | TriangularPrism
  | RightAngledTriangle

-- Define a function to determine if a shape is solid
def isSolid (s : Shape) : Bool :=
  match s with
  | Shape.Cone => true
  | Shape.Cuboid => true
  | Shape.Sphere => true
  | Shape.TriangularPrism => true
  | _ => false

-- Define the list of shapes
def shapeList : List Shape := [
  Shape.Circle,
  Shape.Square,
  Shape.Cone,
  Shape.Cuboid,
  Shape.LineSegment,
  Shape.Sphere,
  Shape.TriangularPrism,
  Shape.RightAngledTriangle
]

-- Theorem: The number of solid figures in the list is 4
theorem solid_figures_count :
  (shapeList.filter isSolid).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_solid_figures_count_l242_24233


namespace NUMINAMATH_CALUDE_count_ways_theorem_l242_24284

/-- Represents the arrangement of the sentence --/
structure Arrangement :=
  (layout : String)

/-- Calculates the number of ways to read a given arrangement --/
def count_ways_to_read (a : Arrangement) : ℕ :=
  sorry

/-- The specific arrangement given in the problem --/
def given_arrangement : Arrangement :=
  { layout := "Q\"otviaelegenditotper\n10+viacl pja\nitviarle ran\ntviaeleg num\nriaelege umv\nnaitotperannumvolvanturhoraefe\nρl\nlices\nc⇔s\nex\ns." }

/-- Theorem stating that the number of ways to read the given arrangement is 8784 --/
theorem count_ways_theorem : count_ways_to_read given_arrangement = 8784 :=
  sorry

end NUMINAMATH_CALUDE_count_ways_theorem_l242_24284


namespace NUMINAMATH_CALUDE_number_of_spinsters_l242_24297

-- Define the number of spinsters and cats
def spinsters : ℕ := sorry
def cats : ℕ := sorry

-- State the theorem
theorem number_of_spinsters :
  -- Condition 1: The ratio of spinsters to cats is 2:7
  (spinsters : ℚ) / cats = 2 / 7 →
  -- Condition 2: There are 55 more cats than spinsters
  cats = spinsters + 55 →
  -- Conclusion: The number of spinsters is 22
  spinsters = 22 := by
  sorry

end NUMINAMATH_CALUDE_number_of_spinsters_l242_24297


namespace NUMINAMATH_CALUDE_water_jars_count_l242_24254

theorem water_jars_count (total_water : ℚ) (quart_jars half_gal_jars one_gal_jars two_gal_jars : ℕ) : 
  total_water = 56 →
  quart_jars = 16 →
  half_gal_jars = 12 →
  one_gal_jars = 8 →
  two_gal_jars = 4 →
  ∃ (three_gal_jars : ℕ), 
    (quart_jars : ℚ) * (1/4) + 
    (half_gal_jars : ℚ) * (1/2) + 
    (one_gal_jars : ℚ) + 
    (two_gal_jars : ℚ) * 2 + 
    (three_gal_jars : ℚ) * 3 = total_water ∧
    quart_jars + half_gal_jars + one_gal_jars + two_gal_jars + three_gal_jars = 50 :=
by sorry

end NUMINAMATH_CALUDE_water_jars_count_l242_24254


namespace NUMINAMATH_CALUDE_log_function_domain_l242_24247

open Real

theorem log_function_domain (f : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Icc 1 2, ∃ y, f x = log y) →
  (∀ x ∈ Set.Icc 1 2, f x = log (x + 2^x - m)) →
  m < 3 :=
by sorry

end NUMINAMATH_CALUDE_log_function_domain_l242_24247


namespace NUMINAMATH_CALUDE_expression_simplification_l242_24204

theorem expression_simplification (a : ℝ) (h : a ≠ 0) :
  (a * (a + 1) + (a - 1)^2 - 1) / (-a) = -2 * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l242_24204


namespace NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l242_24261

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem f_geq_a_iff_a_in_range (a : ℝ) :
  (∀ x ≥ (1/2 : ℝ), f a x ≥ a) ↔ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l242_24261


namespace NUMINAMATH_CALUDE_diego_fruit_problem_l242_24296

/-- Given a bag with capacity for fruit and some fruits already in the bag,
    calculate the remaining capacity for additional fruit. -/
def remaining_capacity (bag_capacity : ℕ) (occupied_capacity : ℕ) : ℕ :=
  bag_capacity - occupied_capacity

/-- Diego's fruit buying problem -/
theorem diego_fruit_problem (bag_capacity : ℕ) (watermelon_weight : ℕ) (grapes_weight : ℕ) (oranges_weight : ℕ) 
  (h1 : bag_capacity = 20)
  (h2 : watermelon_weight = 1)
  (h3 : grapes_weight = 1)
  (h4 : oranges_weight = 1) :
  remaining_capacity bag_capacity (watermelon_weight + grapes_weight + oranges_weight) = 17 :=
sorry

end NUMINAMATH_CALUDE_diego_fruit_problem_l242_24296


namespace NUMINAMATH_CALUDE_fred_earnings_l242_24253

/-- Represents Fred's chore earnings --/
def chore_earnings (initial_amount final_amount : ℕ) 
  (car_wash_price lawn_mow_price dog_walk_price : ℕ)
  (cars_washed lawns_mowed dogs_walked : ℕ) : Prop :=
  final_amount - initial_amount = 
    car_wash_price * cars_washed + 
    lawn_mow_price * lawns_mowed + 
    dog_walk_price * dogs_walked

/-- Theorem stating that Fred's earnings from chores match the difference in his money --/
theorem fred_earnings :
  chore_earnings 23 86 5 10 3 4 3 7 := by
  sorry

end NUMINAMATH_CALUDE_fred_earnings_l242_24253


namespace NUMINAMATH_CALUDE_expression_nonnegative_l242_24201

theorem expression_nonnegative (x : ℝ) : 
  (x - 20*x^2 + 100*x^3) / (16 - 2*x^3) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonnegative_l242_24201


namespace NUMINAMATH_CALUDE_average_apple_weight_l242_24214

def apple_weights : List ℝ := [120, 150, 180, 200, 220]

theorem average_apple_weight :
  (apple_weights.sum / apple_weights.length : ℝ) = 174 := by
  sorry

end NUMINAMATH_CALUDE_average_apple_weight_l242_24214


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l242_24241

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - (4*k - 2) * x + k^2 = 0) → k ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l242_24241


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_plus_i_l242_24265

theorem imaginary_part_of_one_over_one_plus_i :
  Complex.im (1 / (1 + Complex.I)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_plus_i_l242_24265


namespace NUMINAMATH_CALUDE_sum_of_digits_base_8_of_888_l242_24259

/-- The sum of the digits of the base 8 representation of 888₁₀ is 13. -/
theorem sum_of_digits_base_8_of_888 : 
  (Nat.digits 8 888).sum = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_base_8_of_888_l242_24259


namespace NUMINAMATH_CALUDE_vector_calculation_l242_24246

/-- Given two plane vectors a and b, prove that (1/2)a - (3/2)b equals (-1, 2) -/
theorem vector_calculation (a b : ℝ × ℝ) (ha : a = (1, 1)) (hb : b = (1, -1)) :
  (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l242_24246


namespace NUMINAMATH_CALUDE_distinct_paths_count_l242_24244

/-- The number of floors in the building -/
def num_floors : ℕ := 5

/-- The number of staircases between each consecutive floor -/
def staircases_per_floor : ℕ := 2

/-- The number of floors to descend -/
def floors_to_descend : ℕ := num_floors - 1

/-- The number of distinct paths from the top floor to the bottom floor -/
def num_paths : ℕ := staircases_per_floor ^ floors_to_descend

theorem distinct_paths_count :
  num_paths = 16 := by sorry

end NUMINAMATH_CALUDE_distinct_paths_count_l242_24244


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l242_24224

def book_cost : ℚ := 46.25

def five_dollar_bills : ℕ := 4
def one_dollar_bills : ℕ := 5
def quarters : ℕ := 10

def nickel_value : ℚ := 0.05

theorem minimum_nickels_needed :
  ∃ n : ℕ,
    (n : ℚ) * nickel_value +
    (five_dollar_bills : ℚ) * 5 +
    (one_dollar_bills : ℚ) * 1 +
    (quarters : ℚ) * 0.25 ≥ book_cost ∧
    ∀ m : ℕ, m < n →
      (m : ℚ) * nickel_value +
      (five_dollar_bills : ℚ) * 5 +
      (one_dollar_bills : ℚ) * 1 +
      (quarters : ℚ) * 0.25 < book_cost :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l242_24224


namespace NUMINAMATH_CALUDE_total_unique_plants_l242_24260

-- Define the flower beds as finite sets
variable (A B C : Finset ℕ)

-- Define the cardinalities of the sets
variable (card_A : Finset.card A = 600)
variable (card_B : Finset.card B = 500)
variable (card_C : Finset.card C = 400)

-- Define the intersections
variable (card_AB : Finset.card (A ∩ B) = 60)
variable (card_AC : Finset.card (A ∩ C) = 80)
variable (card_BC : Finset.card (B ∩ C) = 40)
variable (card_ABC : Finset.card (A ∩ B ∩ C) = 20)

-- Theorem statement
theorem total_unique_plants :
  Finset.card (A ∪ B ∪ C) = 1340 := by
  sorry

end NUMINAMATH_CALUDE_total_unique_plants_l242_24260


namespace NUMINAMATH_CALUDE_proposition_implication_l242_24202

theorem proposition_implication (p q : Prop) 
  (h1 : ¬p) 
  (h2 : p ∨ q) : 
  q := by sorry

end NUMINAMATH_CALUDE_proposition_implication_l242_24202


namespace NUMINAMATH_CALUDE_fraction_division_simplification_l242_24283

theorem fraction_division_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ 0) (h4 : x ≠ 3) : 
  ((x^2 - 5*x + 6) / (x^2 - 1)) / ((x - 3) / (x^2 + x)) = (x * (x - 2)) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_simplification_l242_24283


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l242_24250

theorem circle_area_from_circumference (r : ℝ) (k : ℝ) : 
  (2 * π * r = 36 * π) → (π * r^2 = k * π) → k = 324 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l242_24250


namespace NUMINAMATH_CALUDE_complex_expression_equality_l242_24277

theorem complex_expression_equality : 
  let a : ℂ := 3 + 2*I
  let b : ℂ := 2 - I
  3*a + 4*b = 17 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l242_24277


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l242_24279

theorem cone_lateral_surface_area 
  (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 10) : 
  π * r * l = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l242_24279


namespace NUMINAMATH_CALUDE_tan_beta_plus_pi_fourth_l242_24221

theorem tan_beta_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3/5) 
  (h2 : Real.tan α = 1/3) : 
  Real.tan (β + π/4) = 11/7 := by sorry

end NUMINAMATH_CALUDE_tan_beta_plus_pi_fourth_l242_24221


namespace NUMINAMATH_CALUDE_friday_temperature_l242_24213

/-- Temperatures for each day of the week -/
structure WeekTemperatures where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The theorem stating the temperature on Friday given the conditions -/
theorem friday_temperature (temps : WeekTemperatures)
  (h1 : (temps.monday + temps.tuesday + temps.wednesday + temps.thursday) / 4 = 48)
  (h2 : (temps.tuesday + temps.wednesday + temps.thursday + temps.friday) / 4 = 46)
  (h3 : temps.monday = 41) :
  temps.friday = 33 := by
  sorry

#check friday_temperature

end NUMINAMATH_CALUDE_friday_temperature_l242_24213


namespace NUMINAMATH_CALUDE_total_rainfall_three_years_l242_24220

def average_rainfall_2003 : ℝ := 50
def rainfall_increase_2004 : ℝ := 3
def rainfall_increase_2005 : ℝ := 5
def months_per_year : ℕ := 12

theorem total_rainfall_three_years : 
  (average_rainfall_2003 * months_per_year) +
  ((average_rainfall_2003 + rainfall_increase_2004) * months_per_year) +
  ((average_rainfall_2003 + rainfall_increase_2004 + rainfall_increase_2005) * months_per_year) = 1932 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_three_years_l242_24220


namespace NUMINAMATH_CALUDE_parallel_line_plane_not_imply_parallel_lines_l242_24226

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallelLine : Line → Line → Prop)
variable (parallelLineToPlane : Line → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)

-- Define the specific objects
variable (a b : Line)
variable (α : Plane)

-- State the theorem
theorem parallel_line_plane_not_imply_parallel_lines
  (h1 : parallelLineToPlane a α)
  (h2 : lineInPlane b α) :
  ¬ (parallelLine a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_not_imply_parallel_lines_l242_24226


namespace NUMINAMATH_CALUDE_real_m_values_l242_24218

theorem real_m_values (m : ℝ) : 
  let z : ℂ := m^2 * (1 + Complex.I) - m * (m + Complex.I)
  Complex.im z = 0 → m = 0 ∨ m = 1 := by
sorry

end NUMINAMATH_CALUDE_real_m_values_l242_24218


namespace NUMINAMATH_CALUDE_inverse_proposition_geometric_sequence_l242_24252

theorem inverse_proposition_geometric_sequence (a b c : ℝ) :
  (∀ {a b c : ℝ}, (∃ r : ℝ, b = a * r ∧ c = b * r) → b^2 = a * c) →
  (b^2 = a * c → ∃ r : ℝ, b = a * r ∧ c = b * r) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_geometric_sequence_l242_24252


namespace NUMINAMATH_CALUDE_unique_congruence_l242_24267

theorem unique_congruence (n : ℤ) : 
  12 ≤ n ∧ n ≤ 18 ∧ n ≡ 9001 [ZMOD 7] → n = 13 := by
sorry

end NUMINAMATH_CALUDE_unique_congruence_l242_24267


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_l242_24243

/-- A regular six-pointed star -/
structure SixPointedStar :=
  (points : Fin 12)

/-- The symmetry group of a regular six-pointed star -/
def starSymmetryGroup : ℕ := 12

/-- The number of distinct arrangements of 12 different objects on a regular six-pointed star,
    considering reflections and rotations as equivalent -/
def distinctArrangements (star : SixPointedStar) : ℕ :=
  Nat.factorial 12 / starSymmetryGroup

theorem distinct_arrangements_count (star : SixPointedStar) :
  distinctArrangements star = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_l242_24243


namespace NUMINAMATH_CALUDE_simplified_multiplication_l242_24231

def factor1 : Nat := 20213
def factor2 : Nat := 732575

theorem simplified_multiplication (f1 f2 : Nat) (h1 : f1 = factor1) (h2 : f2 = factor2) :
  ∃ (partial_products : List Nat),
    f1 * f2 = partial_products.sum ∧
    partial_products.length < 5 :=
sorry

end NUMINAMATH_CALUDE_simplified_multiplication_l242_24231


namespace NUMINAMATH_CALUDE_probability_all_players_have_initial_coins_l242_24255

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Charlie : Player
| Dana : Player

/-- Represents a ball color -/
inductive BallColor : Type
| Blue : BallColor
| Red : BallColor
| White : BallColor
| Yellow : BallColor

/-- Represents the state of the game -/
structure GameState :=
  (coins : Player → ℕ)
  (round : ℕ)

/-- Represents a single round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- Probability of a specific outcome in a single round -/
def round_probability : ℚ :=
  12 / 120

/-- The game consists of 5 rounds -/
def num_rounds : ℕ := 5

/-- The initial number of coins for each player -/
def initial_coins : ℕ := 5

/-- Theorem stating the probability of all players having the initial number of coins after the game -/
theorem probability_all_players_have_initial_coins :
  (round_probability ^ num_rounds : ℚ) = 1 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_players_have_initial_coins_l242_24255


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l242_24262

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l242_24262


namespace NUMINAMATH_CALUDE_total_notes_count_l242_24299

theorem total_notes_count (total_amount : ℕ) (note_50_count : ℕ) (note_50_value : ℕ) (note_500_value : ℕ) 
  (h1 : total_amount = 10350)
  (h2 : note_50_count = 97)
  (h3 : note_50_value = 50)
  (h4 : note_500_value = 500)
  (h5 : ∃ (note_500_count : ℕ), total_amount = note_50_count * note_50_value + note_500_count * note_500_value) :
  ∃ (total_notes : ℕ), total_notes = note_50_count + (total_amount - note_50_count * note_50_value) / note_500_value ∧ total_notes = 108 := by
sorry

end NUMINAMATH_CALUDE_total_notes_count_l242_24299


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l242_24212

theorem cubic_equation_one_real_root (a : ℝ) : 
  (∃! x : ℝ, x^3 - a*x^2 - 2*a*x + a^2 - 1 = 0) → a < 3/4 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l242_24212


namespace NUMINAMATH_CALUDE_green_toads_count_l242_24282

/-- The number of green toads per acre -/
def green_toads_per_acre : ℕ := 8

/-- The ratio of green toads to brown toads -/
def green_to_brown_ratio : ℚ := 1 / 25

/-- The fraction of brown toads that are spotted -/
def spotted_brown_fraction : ℚ := 1 / 4

/-- The number of spotted brown toads per acre -/
def spotted_brown_per_acre : ℕ := 50

theorem green_toads_count :
  green_toads_per_acre = 8 :=
sorry

end NUMINAMATH_CALUDE_green_toads_count_l242_24282


namespace NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l242_24264

theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) :
  Real.cos (π + 2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l242_24264


namespace NUMINAMATH_CALUDE_non_similar_500_pointed_stars_l242_24205

/-- A regular n-pointed star is the union of n line segments. -/
def RegularStar (n : ℕ) (m : ℕ) : Prop :=
  (n > 0) ∧ (m > 0) ∧ (m < n) ∧ (Nat.gcd m n = 1)

/-- Two stars are similar if they have the same number of points and
    their m values are either equal or complementary modulo n. -/
def SimilarStars (n : ℕ) (m1 m2 : ℕ) : Prop :=
  RegularStar n m1 ∧ RegularStar n m2 ∧ (m1 = m2 ∨ m1 + m2 = n)

/-- The number of non-similar regular n-pointed stars -/
def NonSimilarStarCount (n : ℕ) : ℕ :=
  (Nat.totient n - 2) / 2 + 1

theorem non_similar_500_pointed_stars :
  NonSimilarStarCount 500 = 99 := by
  sorry

#eval NonSimilarStarCount 500  -- This should output 99

end NUMINAMATH_CALUDE_non_similar_500_pointed_stars_l242_24205


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l242_24275

/-- Given an ellipse with equation x²/m² + y²/n² = 1, where m > 0 and n > 0,
    whose right focus coincides with the focus of the parabola y² = 8x,
    and has an eccentricity of 1/2, prove that its standard equation is
    x²/16 + y²/12 = 1. -/
theorem ellipse_standard_equation
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_focus : m^2 - n^2 = 4)  -- Right focus coincides with parabola focus (2, 0)
  (h_eccentricity : 2 / m = 1 / 2)  -- Eccentricity is 1/2
  : ∃ (x y : ℝ), x^2/16 + y^2/12 = 1 ∧ x^2/m^2 + y^2/n^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l242_24275


namespace NUMINAMATH_CALUDE_milton_pies_sold_l242_24280

/-- Calculates the total number of pies sold given the number of slices ordered and slices per pie -/
def total_pies_sold (apple_slices_ordered : ℕ) (peach_slices_ordered : ℕ) 
                    (slices_per_apple_pie : ℕ) (slices_per_peach_pie : ℕ) : ℕ :=
  (apple_slices_ordered / slices_per_apple_pie) + (peach_slices_ordered / slices_per_peach_pie)

/-- Theorem stating that given the specific conditions, Milton sold 15 pies -/
theorem milton_pies_sold : 
  total_pies_sold 56 48 8 6 = 15 := by
  sorry

#eval total_pies_sold 56 48 8 6

end NUMINAMATH_CALUDE_milton_pies_sold_l242_24280


namespace NUMINAMATH_CALUDE_work_completion_time_l242_24287

/-- The efficiency of worker q -/
def q_efficiency : ℝ := 1

/-- The efficiency of worker p relative to q -/
def p_efficiency : ℝ := 1.6

/-- The efficiency of worker r relative to q -/
def r_efficiency : ℝ := 1.4

/-- The time taken by p alone to complete the work -/
def p_time : ℝ := 26

/-- The total amount of work to be done -/
def total_work : ℝ := p_efficiency * p_time

/-- The combined efficiency of p, q, and r -/
def combined_efficiency : ℝ := p_efficiency + q_efficiency + r_efficiency

/-- The theorem stating the time taken for p, q, and r to complete the work together -/
theorem work_completion_time : 
  total_work / combined_efficiency = 10.4 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l242_24287


namespace NUMINAMATH_CALUDE_inequality_solution_set_l242_24285

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | 12 * x^2 - a * x > a^2}
  S = if a > 0 then
        {x : ℝ | x < -a/4 ∨ x > a/3}
      else if a = 0 then
        {x : ℝ | x ≠ 0}
      else
        {x : ℝ | x > -a/4 ∨ x < a/3} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l242_24285


namespace NUMINAMATH_CALUDE_honey_jars_needed_l242_24200

theorem honey_jars_needed (num_hives : ℕ) (honey_per_hive : ℝ) (jar_capacity : ℝ) 
  (h1 : num_hives = 5)
  (h2 : honey_per_hive = 20)
  (h3 : jar_capacity = 0.5)
  (h4 : jar_capacity > 0) :
  ⌈(↑num_hives * honey_per_hive / 2) / jar_capacity⌉ = 100 := by
  sorry

end NUMINAMATH_CALUDE_honey_jars_needed_l242_24200


namespace NUMINAMATH_CALUDE_garden_usable_area_l242_24236

/-- Calculate the usable area of a rectangular garden with a square pond in one corner -/
theorem garden_usable_area 
  (garden_length : ℝ) 
  (garden_width : ℝ) 
  (pond_side : ℝ) 
  (h1 : garden_length = 20) 
  (h2 : garden_width = 18) 
  (h3 : pond_side = 4) : 
  garden_length * garden_width - pond_side * pond_side = 344 := by
  sorry

#check garden_usable_area

end NUMINAMATH_CALUDE_garden_usable_area_l242_24236


namespace NUMINAMATH_CALUDE_root_sum_squared_plus_triple_plus_other_root_l242_24237

theorem root_sum_squared_plus_triple_plus_other_root (α β : ℝ) : 
  α^2 + 2*α - 2024 = 0 → β^2 + 2*β - 2024 = 0 → α^2 + 3*α + β = 2022 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_squared_plus_triple_plus_other_root_l242_24237


namespace NUMINAMATH_CALUDE_abs_x_minus_log_y_equals_x_plus_2log_y_implies_x_0_y_1_l242_24216

theorem abs_x_minus_log_y_equals_x_plus_2log_y_implies_x_0_y_1
  (x : ℝ) (y : ℝ) (h : y > 0) :
  |x - Real.log y| = x + 2 * Real.log y → x = 0 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_log_y_equals_x_plus_2log_y_implies_x_0_y_1_l242_24216


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l242_24270

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l242_24270


namespace NUMINAMATH_CALUDE_pentagon_area_in_16_sided_polygon_l242_24227

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sideLength : ℝ

/-- Represents a pentagon in a regular polygon -/
structure Pentagon (n : ℕ) where
  polygon : RegularPolygon n
  vertices : Fin 5 → Fin n

/-- Calculates the area of a pentagon in a regular polygon -/
def pentagonArea (n : ℕ) (p : Pentagon n) : ℝ := sorry

theorem pentagon_area_in_16_sided_polygon :
  ∀ (p : Pentagon 16),
    p.polygon.sideLength = 3 →
    (∀ i : Fin 5, (p.vertices i + 4) % 16 = p.vertices ((i + 1) % 5)) →
    pentagonArea 16 p = 198 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_in_16_sided_polygon_l242_24227


namespace NUMINAMATH_CALUDE_train_length_calculation_l242_24298

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length_calculation (speed : Real) (time : Real) : 
  speed = 144 ∧ time = 1.24990000799936 → 
  ∃ (length : Real), abs (length - 50) < 0.01 ∧ length = speed * time * (5 / 18) := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l242_24298


namespace NUMINAMATH_CALUDE_minimum_area_of_reported_tile_l242_24263

/-- Represents the reported dimension of a side of a tile -/
structure ReportedDimension where
  value : ℝ
  lower_bound : ℝ := value - 0.7
  upper_bound : ℝ := value + 0.7

/-- Represents a rectangular tile with reported dimensions -/
structure ReportedTile where
  length : ReportedDimension
  width : ReportedDimension

def minimum_area (tile : ReportedTile) : ℝ :=
  tile.length.lower_bound * tile.width.lower_bound

theorem minimum_area_of_reported_tile (tile : ReportedTile) 
  (h1 : tile.length.value = 3) 
  (h2 : tile.width.value = 4) : 
  minimum_area tile = 7.59 := by
  sorry

#eval minimum_area { length := { value := 3 }, width := { value := 4 } }

end NUMINAMATH_CALUDE_minimum_area_of_reported_tile_l242_24263


namespace NUMINAMATH_CALUDE_nectar_water_percentage_l242_24223

/-- The ratio of flower-nectar to honey produced -/
def nectarToHoneyRatio : ℝ := 1.6

/-- The percentage of water in the produced honey -/
def honeyWaterPercentage : ℝ := 20

/-- The percentage of water in flower-nectar -/
def nectarWaterPercentage : ℝ := 50

theorem nectar_water_percentage :
  nectarWaterPercentage = 100 * (nectarToHoneyRatio - (1 - honeyWaterPercentage / 100)) / nectarToHoneyRatio :=
by sorry

end NUMINAMATH_CALUDE_nectar_water_percentage_l242_24223


namespace NUMINAMATH_CALUDE_profit_40_percent_l242_24249

/-- Calculates the profit percentage when selling a certain number of articles at a price equal to the cost of a different number of articles. -/
def profit_percentage (sold : ℕ) (cost_equivalent : ℕ) : ℚ :=
  ((cost_equivalent - sold) / sold) * 100

/-- Theorem stating that selling 50 articles at the cost price of 70 articles results in a 40% profit. -/
theorem profit_40_percent :
  profit_percentage 50 70 = 40 := by
  sorry

end NUMINAMATH_CALUDE_profit_40_percent_l242_24249


namespace NUMINAMATH_CALUDE_min_value_theorem_l242_24230

theorem min_value_theorem (c : ℝ) (hc : c > 0) (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (heq : a^2 - 2*a*b + 2*b^2 - c = 0) (hmax : ∀ a' b' : ℝ, a'^2 - 2*a'*b' + 2*b'^2 - c = 0 → a' + b' ≤ a + b) :
  ∃ (m : ℝ), m = -1/4 ∧ ∀ a' b' : ℝ, a' ≠ 0 → b' ≠ 0 → a'^2 - 2*a'*b' + 2*b'^2 - c = 0 →
    m ≤ (3/a' - 4/b' + 5/c) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l242_24230


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_l242_24207

/-- Calculates the profit percentage for a dishonest dealer --/
theorem dishonest_dealer_profit (real_weight : ℝ) (cost_price : ℝ) 
  (h1 : real_weight > 0) (h2 : cost_price > 0) : 
  let counterfeit_weight := 0.8 * real_weight
  let impure_weight := counterfeit_weight * 1.15
  let selling_price := cost_price * (real_weight / impure_weight)
  let profit := selling_price - cost_price
  profit / cost_price = 0.25 := by sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_l242_24207


namespace NUMINAMATH_CALUDE_box_height_is_eight_inches_l242_24209

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.height * d.width * d.length

theorem box_height_is_eight_inches
  (box : Dimensions)
  (block : Dimensions)
  (h1 : box.width = 10)
  (h2 : box.length = 12)
  (h3 : block.height = 3)
  (h4 : block.width = 2)
  (h5 : block.length = 4)
  (h6 : volume box = 40 * volume block) :
  box.height = 8 := by
  sorry

end NUMINAMATH_CALUDE_box_height_is_eight_inches_l242_24209


namespace NUMINAMATH_CALUDE_least_with_four_prime_factors_l242_24293

/-- A function that returns the number of prime factors (counting multiplicity) of a positive integer -/
def num_prime_factors (n : ℕ+) : ℕ := sorry

/-- The property that both n and n+1 have exactly four prime factors -/
def has_four_prime_factors (n : ℕ+) : Prop :=
  num_prime_factors n = 4 ∧ num_prime_factors (n + 1) = 4

theorem least_with_four_prime_factors :
  ∀ n : ℕ+, n < 1155 → ¬(has_four_prime_factors n) ∧ has_four_prime_factors 1155 := by
  sorry

end NUMINAMATH_CALUDE_least_with_four_prime_factors_l242_24293


namespace NUMINAMATH_CALUDE_series_diverges_l242_24271

/-- Ceiling function -/
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

/-- The general term of the series -/
noncomputable def a (n : ℕ) : ℝ :=
  1 / (n : ℝ) ^ (1 + ceiling (Real.sin n))

/-- The series is divergent -/
theorem series_diverges : ¬ (Summable a) := by
  sorry

end NUMINAMATH_CALUDE_series_diverges_l242_24271


namespace NUMINAMATH_CALUDE_class_average_age_l242_24258

theorem class_average_age (original_students : ℕ) (new_students : ℕ) (new_average_age : ℕ) (average_decrease : ℕ) :
  original_students = 12 →
  new_students = 12 →
  new_average_age = 32 →
  average_decrease = 4 →
  ∃ (original_average : ℕ),
    (original_students * original_average + new_students * new_average_age) / (original_students + new_students) = original_average - average_decrease ∧
    original_average = 40 :=
by sorry

end NUMINAMATH_CALUDE_class_average_age_l242_24258


namespace NUMINAMATH_CALUDE_pregnant_cow_percentage_l242_24268

theorem pregnant_cow_percentage (total_cows : ℕ) (female_percentage : ℚ) (pregnant_cows : ℕ) : 
  total_cows = 44 →
  female_percentage = 1/2 →
  pregnant_cows = 11 →
  (pregnant_cows : ℚ) / (female_percentage * total_cows) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pregnant_cow_percentage_l242_24268


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l242_24229

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 := by sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l242_24229


namespace NUMINAMATH_CALUDE_shaded_area_circles_l242_24238

theorem shaded_area_circles (R : ℝ) (h : R = 10) : 
  let r : ℝ := R / 3
  let larger_area : ℝ := π * R^2
  let smaller_area : ℝ := 2 * π * r^2
  let shaded_area : ℝ := larger_area - smaller_area
  shaded_area = (700 / 9) * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_circles_l242_24238


namespace NUMINAMATH_CALUDE_four_to_fourth_sum_l242_24278

theorem four_to_fourth_sum : (4 : ℕ) ^ 4 + (4 : ℕ) ^ 4 + (4 : ℕ) ^ 4 + (4 : ℕ) ^ 4 = (4 : ℕ) ^ 5 := by
  sorry

end NUMINAMATH_CALUDE_four_to_fourth_sum_l242_24278


namespace NUMINAMATH_CALUDE_ellipse_properties_l242_24239

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  eccentricity : ℝ
  eccentricity_eq : eccentricity = Real.sqrt 6 / 3
  equation : ℝ → ℝ → Prop
  equation_def : equation = λ x y => x^2 / a^2 + y^2 / b^2 = 1
  focal_line_length : ℝ
  focal_line_length_eq : focal_line_length = 2 * Real.sqrt 3 / 3

/-- The main theorem about the ellipse -/
theorem ellipse_properties (e : Ellipse) :
  e.equation = λ x y => x^2 / 3 + y^2 = 1 ∧
  ∃ k : ℝ, k = 7 / 6 ∧
    ∀ C D : ℝ × ℝ,
      (e.equation C.1 C.2 ∧ e.equation D.1 D.2) →
      (C.2 = k * C.1 + 2 ∧ D.2 = k * D.1 + 2) →
      (C.1 - (-1))^2 + (C.2 - 0)^2 = (D.1 - (-1))^2 + (D.2 - 0)^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l242_24239


namespace NUMINAMATH_CALUDE_jesses_room_dimension_difference_l242_24245

/-- Given Jesse's room dimensions, prove the difference between length and width --/
theorem jesses_room_dimension_difference :
  let room_length : ℕ := 20
  let room_width : ℕ := 19
  room_length - room_width = 1 := by
  sorry

end NUMINAMATH_CALUDE_jesses_room_dimension_difference_l242_24245


namespace NUMINAMATH_CALUDE_number_of_sodas_bought_l242_24288

/-- Given the total cost, sandwich cost, and soda cost, calculate the number of sodas bought -/
theorem number_of_sodas_bought (total_cost sandwich_cost soda_cost : ℚ) 
  (h_total : total_cost = 8.36)
  (h_sandwich : sandwich_cost = 2.44)
  (h_soda : soda_cost = 0.87) :
  (total_cost - 2 * sandwich_cost) / soda_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_number_of_sodas_bought_l242_24288


namespace NUMINAMATH_CALUDE_complex_equation_solution_l242_24225

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) - 2 * Complex.I * z = 1 + 5 * Complex.I * z ∧ z = -((4 : ℂ) * Complex.I / 7) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l242_24225


namespace NUMINAMATH_CALUDE_cloth_sold_proof_l242_24295

/-- Represents the profit per meter of cloth in Rs. -/
def profit_per_meter : ℕ := 35

/-- Represents the total profit earned in Rs. -/
def total_profit : ℕ := 1400

/-- Represents the number of meters of cloth sold -/
def meters_sold : ℕ := total_profit / profit_per_meter

theorem cloth_sold_proof : meters_sold = 40 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sold_proof_l242_24295


namespace NUMINAMATH_CALUDE_rectangle_ratio_l242_24215

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if the ratio of their areas is 0.16 and a/c = b/d,
    then a/c = b/d = 0.4 -/
theorem rectangle_ratio (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : (a * b) / (c * d) = 0.16) (h6 : a / c = b / d) :
  a / c = 0.4 ∧ b / d = 0.4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l242_24215


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l242_24203

/-- Proves that the repeating decimal 7.832̅ is equal to the fraction 70/9 -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 7 + 832 / 999 ∧ x = 70 / 9 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l242_24203


namespace NUMINAMATH_CALUDE_largest_package_size_l242_24292

theorem largest_package_size (liam_markers zoe_markers : ℕ) 
  (h1 : liam_markers = 60) 
  (h2 : zoe_markers = 36) : 
  Nat.gcd liam_markers zoe_markers = 12 := by
sorry

end NUMINAMATH_CALUDE_largest_package_size_l242_24292


namespace NUMINAMATH_CALUDE_train_speed_problem_l242_24294

theorem train_speed_problem (x : ℝ) (V : ℝ) (h1 : x > 0) (h2 : V > 0) :
  (3 * x) / ((x / V) + ((2 * x) / 20)) = 25 →
  V = 50 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l242_24294


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_396_l242_24235

theorem six_digit_divisible_by_396 (n : ℕ) :
  (∃ x y z : ℕ, 
    0 ≤ x ∧ x ≤ 9 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    0 ≤ z ∧ z ≤ 9 ∧
    n = 100000 * x + 10000 * y + 3420 + z) →
  n % 396 = 0 →
  n = 453420 ∨ n = 413424 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_396_l242_24235


namespace NUMINAMATH_CALUDE_greatest_divisor_of_product_l242_24248

def S : Set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) :=
  {t | let (a, b, c, d, e, f) := t
       a^2 + b^2 + c^2 + d^2 + e^2 = f^2}

theorem greatest_divisor_of_product (k : ℕ) : k = 24 ↔ 
  (∀ t ∈ S, let (a, b, c, d, e, f) := t
            (k : ℤ) ∣ a * b * c * d * e * f) ∧
  (∀ m > k, ∃ t ∈ S, let (a, b, c, d, e, f) := t
            ¬((m : ℤ) ∣ a * b * c * d * e * f)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_product_l242_24248


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l242_24232

/-- Definition of the ⋄ operation -/
noncomputable def diamond (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem stating that if 3 ⋄ y = 12, then y = 72 -/
theorem diamond_equation_solution :
  ∃ y : ℝ, diamond 3 y = 12 ∧ y = 72 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l242_24232


namespace NUMINAMATH_CALUDE_power_inequality_specific_power_inequality_l242_24208

theorem power_inequality (a : ℕ) (h : a ≥ 3) : a^(a+1) > (a+1)^a := by sorry

theorem specific_power_inequality : (2023 : ℕ)^2024 > 2024^2023 := by sorry

end NUMINAMATH_CALUDE_power_inequality_specific_power_inequality_l242_24208


namespace NUMINAMATH_CALUDE_base7_4513_equals_1627_l242_24266

/-- Converts a base-7 digit to its base-10 equivalent --/
def base7ToBase10Digit (d : ℕ) : ℕ := d

/-- Converts a list of base-7 digits to a base-10 number --/
def base7ToBase10 (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

theorem base7_4513_equals_1627 :
  base7ToBase10 [3, 1, 5, 4] = 1627 := by sorry

end NUMINAMATH_CALUDE_base7_4513_equals_1627_l242_24266


namespace NUMINAMATH_CALUDE_percentage_difference_l242_24289

theorem percentage_difference (A B C y : ℝ) : 
  A = B + C → 
  B > C → 
  C > 0 → 
  B = C * (1 + y / 100) → 
  y = 100 * ((B - C) / C) :=
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l242_24289


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l242_24219

theorem complementary_angles_ratio (x y : ℝ) : 
  x + y = 90 →  -- The angles are complementary (sum to 90°)
  x = 4 * y →   -- The ratio of the angles is 4:1
  y = 18 :=     -- The smaller angle is 18°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l242_24219


namespace NUMINAMATH_CALUDE_max_value_sum_fractions_l242_24286

theorem max_value_sum_fractions (a b c : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c)
  (sum_eq_two : a + b + c = 2) :
  (a * b / (a + b + 1)) + (b * c / (b + c + 1)) + (c * a / (c + a + 1)) ≤ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_fractions_l242_24286


namespace NUMINAMATH_CALUDE_tangent_point_determines_b_l242_24273

-- Define the curve and line
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b
def line (x k : ℝ) : ℝ := k*x + 1

-- Define the tangent condition
def is_tangent (a b k : ℝ) : Prop :=
  ∃ x, curve x a b = line x k ∧ 
       (deriv (fun x => curve x a b)) x = k

theorem tangent_point_determines_b :
  ∀ a b k : ℝ, 
    is_tangent a b k →  -- The line is tangent to the curve
    curve 1 a b = 3 →   -- The point of tangency is (1, 3)
    b = 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_determines_b_l242_24273


namespace NUMINAMATH_CALUDE_city_pairing_equality_l242_24217

/-- The number of ways to form r pairs in City A -/
def A (n r : ℕ) : ℕ := sorry

/-- The number of ways to form r pairs in City B -/
def B (n r : ℕ) : ℕ := sorry

/-- Girls in City B know a specific number of boys -/
def girls_know_boys (i : ℕ) : ℕ := 2 * i - 1

theorem city_pairing_equality (n : ℕ) (hn : n ≥ 1) :
  ∀ r : ℕ, 1 ≤ r ∧ r ≤ n → A n r = B n r := by
  sorry

end NUMINAMATH_CALUDE_city_pairing_equality_l242_24217


namespace NUMINAMATH_CALUDE_min_x_value_and_factor_sum_l242_24222

theorem min_x_value_and_factor_sum (x y : ℕ+) (h : 3 * x^7 = 17 * y^11) :
  ∃ (a b c d : ℕ),
    x = a^c * b^d ∧
    a = 17 ∧ b = 3 ∧ c = 6 ∧ d = 4 ∧
    a + b + c + d = 30 ∧
    (∀ (x' : ℕ+), 3 * x'^7 = 17 * y^11 → x ≤ x') := by
sorry

end NUMINAMATH_CALUDE_min_x_value_and_factor_sum_l242_24222


namespace NUMINAMATH_CALUDE_second_player_wins_l242_24281

/-- Represents a grid in the domino gluing game -/
structure Grid :=
  (size : Nat)
  (is_cut_into_dominoes : Bool)

/-- Represents a move in the domino gluing game -/
structure Move :=
  (x1 y1 x2 y2 : Nat)

/-- Represents the state of the game -/
structure GameState :=
  (grid : Grid)
  (current_player : Nat)
  (moves : List Move)

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : Move) : Bool :=
  sorry

/-- Checks if the game is over (i.e., the figure is connected) -/
def is_game_over (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for a player -/
def is_winning_strategy (player : Nat) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins (grid : Grid) 
    (h1 : grid.size = 100) 
    (h2 : grid.is_cut_into_dominoes = true) : 
  ∃ (strategy : Strategy), is_winning_strategy 2 strategy :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l242_24281
