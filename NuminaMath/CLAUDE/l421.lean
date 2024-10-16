import Mathlib

namespace NUMINAMATH_CALUDE_cards_given_to_jeff_main_theorem_l421_42187

/-- Proves that the number of cards Nell gave to Jeff is 276 --/
theorem cards_given_to_jeff : ℕ → ℕ → ℕ → Prop :=
  fun nell_initial nell_remaining cards_given =>
    nell_initial = 528 →
    nell_remaining = 252 →
    cards_given = nell_initial - nell_remaining →
    cards_given = 276

/-- The main theorem --/
theorem main_theorem : ∃ (cards_given : ℕ), cards_given_to_jeff 528 252 cards_given :=
  sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_main_theorem_l421_42187


namespace NUMINAMATH_CALUDE_hidden_dots_on_dice_l421_42172

theorem hidden_dots_on_dice (dice_count : Nat) (face_count : Nat) (visible_faces : Nat) (visible_sum : Nat) : 
  dice_count = 3 →
  face_count = 6 →
  visible_faces = 7 →
  visible_sum = 22 →
  (dice_count * face_count * (face_count + 1) / 2) - visible_sum = 41 := by
sorry

end NUMINAMATH_CALUDE_hidden_dots_on_dice_l421_42172


namespace NUMINAMATH_CALUDE_binomial_10_3_l421_42116

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l421_42116


namespace NUMINAMATH_CALUDE_book_pages_calculation_l421_42188

theorem book_pages_calculation (chapters : ℕ) (days : ℕ) (pages_per_day : ℕ) 
  (h1 : chapters = 41)
  (h2 : days = 30)
  (h3 : pages_per_day = 15) :
  chapters * (days * pages_per_day / chapters) = days * pages_per_day :=
by
  sorry

#check book_pages_calculation

end NUMINAMATH_CALUDE_book_pages_calculation_l421_42188


namespace NUMINAMATH_CALUDE_iggy_ran_four_miles_on_tuesday_l421_42190

/-- Represents the days of the week Iggy runs --/
inductive RunDay
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents Iggy's running data --/
structure RunningData where
  miles : RunDay → ℕ
  pace : ℕ  -- minutes per mile
  totalTime : ℕ  -- total running time in minutes

/-- Theorem stating that Iggy ran 4 miles on Tuesday --/
theorem iggy_ran_four_miles_on_tuesday (data : RunningData) : data.miles RunDay.Tuesday = 4 :=
  by
  have h1 : data.miles RunDay.Monday = 3 := by sorry
  have h2 : data.miles RunDay.Wednesday = 6 := by sorry
  have h3 : data.miles RunDay.Thursday = 8 := by sorry
  have h4 : data.miles RunDay.Friday = 3 := by sorry
  have h5 : data.pace = 10 := by sorry
  have h6 : data.totalTime = 4 * 60 := by sorry
  
  sorry


end NUMINAMATH_CALUDE_iggy_ran_four_miles_on_tuesday_l421_42190


namespace NUMINAMATH_CALUDE_f_2007_values_l421_42168

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1

theorem f_2007_values (f : ℕ → ℕ) (h : is_valid_f f) : 
  f 2007 ∈ Finset.range 2009 ∧ 
  ∀ k ∈ Finset.range 2009, ∃ g : ℕ → ℕ, is_valid_f g ∧ g 2007 = k :=
sorry

end NUMINAMATH_CALUDE_f_2007_values_l421_42168


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l421_42113

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 2 * a^2 - 3*a - 5*a^2 + 6*a = -3*a^2 + 3*a := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : 2*(a-1) - (2*a-3) + 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l421_42113


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l421_42101

/-- Given a line y = x + m intersecting the ellipse 4x^2 + y^2 = 1 and forming a chord of length 2√2/5, prove that m = ± √5/2 -/
theorem line_ellipse_intersection (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    4 * x₁^2 + (x₁ + m)^2 = 1 ∧ 
    4 * x₂^2 + (x₂ + m)^2 = 1 ∧ 
    (x₂ - x₁)^2 + ((x₂ + m) - (x₁ + m))^2 = (2 * Real.sqrt 2 / 5)^2) → 
  m = Real.sqrt 5 / 2 ∨ m = -Real.sqrt 5 / 2 :=
by sorry


end NUMINAMATH_CALUDE_line_ellipse_intersection_l421_42101


namespace NUMINAMATH_CALUDE_hot_dog_price_l421_42171

-- Define variables for hamburger and hot dog prices
variable (h d : ℚ)

-- Define the equations based on the given conditions
def day1_equation : Prop := 3 * h + 4 * d = 10
def day2_equation : Prop := 2 * h + 3 * d = 7

-- Theorem statement
theorem hot_dog_price 
  (eq1 : day1_equation h d) 
  (eq2 : day2_equation h d) : 
  d = 1 := by sorry

end NUMINAMATH_CALUDE_hot_dog_price_l421_42171


namespace NUMINAMATH_CALUDE_probability_at_least_three_speak_l421_42163

def probability_of_success : ℚ := 1 / 3

def number_of_trials : ℕ := 7

def minimum_successes : ℕ := 3

theorem probability_at_least_three_speak :
  (1 : ℚ) - (Finset.sum (Finset.range minimum_successes) (λ k =>
    (Nat.choose number_of_trials k : ℚ) *
    probability_of_success ^ k *
    (1 - probability_of_success) ^ (number_of_trials - k)))
  = 939 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_three_speak_l421_42163


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l421_42177

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4 / (1 - r))) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l421_42177


namespace NUMINAMATH_CALUDE_shop_b_better_l421_42136

/-- Represents a costume rental shop -/
structure Shop where
  name : String
  base_price : ℕ
  discount_rate : ℚ
  discount_threshold : ℕ
  additional_discount : ℕ

/-- Calculates the number of sets that can be rented from a shop given a budget -/
def sets_rentable (shop : Shop) (budget : ℕ) : ℚ :=
  if budget / shop.base_price > shop.discount_threshold
  then (budget + shop.additional_discount) / (shop.base_price * (1 - shop.discount_rate))
  else budget / shop.base_price

/-- The main theorem proving Shop B offers more sets than Shop A -/
theorem shop_b_better (shop_a shop_b : Shop) (budget : ℕ) :
  shop_a.name = "A" →
  shop_b.name = "B" →
  shop_b.base_price = shop_a.base_price + 10 →
  400 / shop_a.base_price = 500 / shop_b.base_price →
  shop_b.discount_rate = 1/5 →
  shop_b.discount_threshold = 100 →
  shop_b.additional_discount = 200 →
  budget = 5000 →
  sets_rentable shop_b budget > sets_rentable shop_a budget :=
by
  sorry

end NUMINAMATH_CALUDE_shop_b_better_l421_42136


namespace NUMINAMATH_CALUDE_angle_conversion_l421_42120

theorem angle_conversion (angle : Real) : ∃ (k : Int) (α : Real), 
  angle = k * (2 * Real.pi) + α ∧ 
  0 ≤ α ∧ 
  α < 2 * Real.pi ∧ 
  angle = -1125 * (Real.pi / 180) ∧ 
  angle = -8 * Real.pi + 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_conversion_l421_42120


namespace NUMINAMATH_CALUDE_money_ratio_l421_42127

theorem money_ratio (j : ℝ) (k : ℝ) : 
  (j + (2 * j - 7) + 60 = 113) →  -- Sum of all money
  (60 = k * j) →                  -- Patricia's money is a multiple of Jethro's
  (60 : ℝ) / j = 3 :=             -- Ratio of Patricia's to Jethro's money
by
  sorry

end NUMINAMATH_CALUDE_money_ratio_l421_42127


namespace NUMINAMATH_CALUDE_exponent_simplification_l421_42135

theorem exponent_simplification :
  3^12 * 8^12 * 3^3 * 8^8 = 24^15 * 32768 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l421_42135


namespace NUMINAMATH_CALUDE_staircase_arrangement_7_steps_l421_42182

/-- The number of ways 3 people can stand on a staircase with n steps,
    where each step can accommodate at most 2 people and
    the positions of people on the same step are not distinguished. -/
def staircase_arrangements (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The number of ways 3 people can stand on a 7-step staircase is 336,
    given that each step can accommodate at most 2 people and
    the positions of people on the same step are not distinguished. -/
theorem staircase_arrangement_7_steps :
  staircase_arrangements 7 = 336 := by
  sorry

end NUMINAMATH_CALUDE_staircase_arrangement_7_steps_l421_42182


namespace NUMINAMATH_CALUDE_pump_fill_time_l421_42104

/-- The time it takes for the pump to fill the tank without the leak -/
def pump_time : ℝ := 2

/-- The time it takes for the pump and leak together to fill the tank -/
def combined_time : ℝ := 2.8

/-- The time it takes for the leak to empty the full tank -/
def leak_time : ℝ := 7

theorem pump_fill_time :
  (1 / pump_time) - (1 / leak_time) = (1 / combined_time) :=
by sorry

end NUMINAMATH_CALUDE_pump_fill_time_l421_42104


namespace NUMINAMATH_CALUDE_molecular_weight_one_mole_l421_42166

/-- The molecular weight of Aluminium hydroxide for a given number of moles. -/
def molecular_weight (moles : ℝ) : ℝ := sorry

/-- The number of moles for which we know the molecular weight. -/
def known_moles : ℝ := 4

/-- The known molecular weight for the given number of moles. -/
def known_weight : ℝ := 312

/-- Theorem stating that the molecular weight of one mole of Aluminium hydroxide is 78 g/mol. -/
theorem molecular_weight_one_mole :
  molecular_weight 1 = 78 :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_one_mole_l421_42166


namespace NUMINAMATH_CALUDE_range_of_a_l421_42165

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + 1 > 0) ∨ (∃ x : ℝ, a*x^2 + 2 ≤ 0) = False →
  a ∈ Set.Ici 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l421_42165


namespace NUMINAMATH_CALUDE_minutes_to_hours_l421_42103

-- Define the number of minutes Marcia spent
def minutes_spent : ℕ := 300

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem: 300 minutes is equal to 5 hours
theorem minutes_to_hours : 
  (minutes_spent : ℚ) / minutes_per_hour = 5 := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_hours_l421_42103


namespace NUMINAMATH_CALUDE_officer_selection_ways_l421_42180

theorem officer_selection_ways (n : ℕ) (h : n = 8) : 
  (n.factorial / (n - 3).factorial) = 336 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_ways_l421_42180


namespace NUMINAMATH_CALUDE_difference_5321_1234_base7_l421_42111

/-- Converts a base 7 number to decimal --/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a decimal number to base 7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- The difference between two numbers in base 7 --/
def diffBase7 (a b : List Nat) : List Nat :=
  toBase7 (toDecimal a - toDecimal b)

theorem difference_5321_1234_base7 :
  diffBase7 [5, 3, 2, 1] [1, 2, 3, 4] = [4, 0, 5, 4] := by
  sorry

end NUMINAMATH_CALUDE_difference_5321_1234_base7_l421_42111


namespace NUMINAMATH_CALUDE_ellipse_geometric_sequence_l421_42161

/-- Given an ellipse E with equation x²/a² + y²/b² = 1 (a > b > 0),
    eccentricity e = √2/2, and left vertex at (-2,0),
    prove that for points B and C on E, where AB is parallel to OC
    and AB intersects the y-axis at D, |AB|, √2|OC|, and |AD|
    form a geometric sequence. -/
theorem ellipse_geometric_sequence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (E : Set (ℝ × ℝ))
  (hE : E = {(x, y) | x^2/a^2 + y^2/b^2 = 1})
  (he : (a^2 - b^2)/a^2 = 1/2)
  (hA : (-2, 0) ∈ E)
  (B C : ℝ × ℝ) (hB : B ∈ E) (hC : C ∈ E)
  (hparallel : ∃ (k : ℝ), (B.2 + 2*k = B.1 ∧ C.2 = k*C.1))
  (D : ℝ × ℝ) (hD : D.1 = 0 ∧ D.2 = B.2 - B.1/2*B.2) :
  ∃ (r : ℝ), abs (B.1 - (-2)) * abs (D.2) = r * (abs (C.1) * abs (C.1) + abs (C.2) * abs (C.2))
    ∧ abs (D.2)^2 = r * (abs (B.1 - (-2)) * abs (D.2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_geometric_sequence_l421_42161


namespace NUMINAMATH_CALUDE_max_value_of_a_l421_42122

theorem max_value_of_a (a b c : ℝ) 
  (sum_eq : a + b + c = 6)
  (prod_sum_eq : a * b + a * c + b * c = 11) :
  a ≤ 2 + 2 * Real.sqrt 3 / 3 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 6 ∧ 
                    a₀ * b₀ + a₀ * c₀ + b₀ * c₀ = 11 ∧ 
                    a₀ = 2 + 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l421_42122


namespace NUMINAMATH_CALUDE_inequality_proof_l421_42162

theorem inequality_proof (n : ℕ) (hn : n > 1) :
  1 / Real.exp 1 - 1 / (n * Real.exp 1) < (1 - 1 / n : ℝ)^n ∧
  (1 - 1 / n : ℝ)^n < 1 / Real.exp 1 - 1 / (2 * n * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l421_42162


namespace NUMINAMATH_CALUDE_circle_representation_l421_42174

theorem circle_representation (a : ℝ) :
  ∃ h k r, ∀ x y : ℝ,
    x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1 = 0 ↔
    (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_representation_l421_42174


namespace NUMINAMATH_CALUDE_proportion_of_dogs_l421_42159

theorem proportion_of_dogs (C G : ℝ) 
  (h1 : 0.8 * G + 0.25 * C = 0.3 * (G + C)) 
  (h2 : C > 0) 
  (h3 : G > 0) : 
  C / (C + G) = 10 / 11 := by
  sorry

end NUMINAMATH_CALUDE_proportion_of_dogs_l421_42159


namespace NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l421_42198

theorem conditional_probability_rain_given_east_wind 
  (p_east_wind : ℝ) 
  (p_east_wind_and_rain : ℝ) 
  (h1 : p_east_wind = 8/30) 
  (h2 : p_east_wind_and_rain = 7/30) : 
  p_east_wind_and_rain / p_east_wind = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l421_42198


namespace NUMINAMATH_CALUDE_construction_delay_l421_42199

/-- Represents the construction project with given parameters -/
structure ConstructionProject where
  initialWorkers : ℕ
  additionalWorkers : ℕ
  daysBeforeAddingWorkers : ℕ
  totalDays : ℕ

/-- Calculates the total man-days for the project -/
def totalManDays (project : ConstructionProject) : ℕ :=
  (project.initialWorkers * project.daysBeforeAddingWorkers) +
  ((project.initialWorkers + project.additionalWorkers) * (project.totalDays - project.daysBeforeAddingWorkers))

/-- Calculates the number of days needed with only initial workers -/
def daysWithInitialWorkersOnly (project : ConstructionProject) : ℕ :=
  (totalManDays project) / project.initialWorkers

/-- Theorem stating the delay in construction without additional workers -/
theorem construction_delay (project : ConstructionProject) 
  (h1 : project.initialWorkers = 100)
  (h2 : project.additionalWorkers = 100)
  (h3 : project.daysBeforeAddingWorkers = 10)
  (h4 : project.totalDays = 100) :
  daysWithInitialWorkersOnly project - project.totalDays = 90 := by
  sorry


end NUMINAMATH_CALUDE_construction_delay_l421_42199


namespace NUMINAMATH_CALUDE_simplify_expression_l421_42185

theorem simplify_expression (x y : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 20 + 4*y = 45*x + 20 + 4*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l421_42185


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l421_42102

theorem arithmetic_calculation : 15 * 20 - 25 * 15 + 10 * 25 = 175 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l421_42102


namespace NUMINAMATH_CALUDE_equilateral_triangle_count_l421_42108

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Counts distinct equilateral triangles with at least two vertices from the polygon -/
def count_equilateral_triangles (p : RegularPolygon 11) : ℕ :=
  sorry

/-- The main theorem stating the count of distinct equilateral triangles -/
theorem equilateral_triangle_count (p : RegularPolygon 11) :
  count_equilateral_triangles p = 92 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_count_l421_42108


namespace NUMINAMATH_CALUDE_cards_ratio_l421_42143

/-- Prove the ratio of cards given to initial cards is 1:2 -/
theorem cards_ratio (brandon_cards : ℕ) (malcom_extra : ℕ) (malcom_left : ℕ)
  (h1 : brandon_cards = 20)
  (h2 : malcom_extra = 8)
  (h3 : malcom_left = 14) :
  (brandon_cards + malcom_extra - malcom_left) / (brandon_cards + malcom_extra) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cards_ratio_l421_42143


namespace NUMINAMATH_CALUDE_total_rats_l421_42191

/-- The number of rats each person has -/
structure RatCounts where
  elodie : ℕ
  hunter : ℕ
  kenia : ℕ
  teagan : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (rc : RatCounts) : Prop :=
  rc.elodie = 30 ∧
  rc.hunter = rc.elodie - 10 ∧
  rc.kenia = 3 * (rc.hunter + rc.elodie) ∧
  rc.teagan = 2 * rc.elodie ∧
  rc.teagan = rc.kenia - 5

/-- The theorem stating that the total number of rats is 260 -/
theorem total_rats (rc : RatCounts) (h : satisfiesConditions rc) :
  rc.elodie + rc.hunter + rc.kenia + rc.teagan = 260 :=
by sorry

end NUMINAMATH_CALUDE_total_rats_l421_42191


namespace NUMINAMATH_CALUDE_f_properties_l421_42164

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) * (1 / x - 2)

theorem f_properties :
  ∀ a : ℝ,
  (∀ x : ℝ, x > 0 → f a x = a * Real.log x + (a + 1) * (1 / x - 2)) →
  (a < -1 → 
    (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ = (a + 1) / a ∧ 
      (∀ x : ℝ, x > 0 → f a x ≤ f a x₀) ∧
      (∀ ε : ℝ, ε > 0 → ∃ x : ℝ, x > 0 ∧ f a x > f a x₀ - ε))) ∧
  (-1 ≤ a ∧ a ≤ 0 → 
    (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 → f a x₁ ≠ f a x₂ ∨ x₁ = x₂)) ∧
  (a > 0 → 
    (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ = (a + 1) / a ∧ 
      (∀ x : ℝ, x > 0 → f a x ≥ f a x₀) ∧
      (∀ ε : ℝ, ε > 0 → ∃ x : ℝ, x > 0 ∧ f a x < f a x₀ + ε))) ∧
  (a > 0 → ∀ x : ℝ, x > 0 → f a x > -a^2 / (a + 1) - 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l421_42164


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l421_42112

theorem square_difference_of_integers (a b : ℕ+) 
  (sum_eq : a + b = 40)
  (diff_eq : a - b = 8) :
  a^2 - b^2 = 320 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l421_42112


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l421_42146

theorem tens_digit_of_8_pow_2023 : ∃ k : ℕ, 8^2023 ≡ 10 * k + 1 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l421_42146


namespace NUMINAMATH_CALUDE_tree_spacing_l421_42106

theorem tree_spacing (yard_length : ℕ) (num_trees : ℕ) (spacing : ℕ) :
  yard_length = 434 →
  num_trees = 32 →
  spacing * (num_trees - 1) = yard_length →
  spacing = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l421_42106


namespace NUMINAMATH_CALUDE_tower_heights_count_l421_42141

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of different tower heights achievable -/
def calculateTowerHeights (numBricks : ℕ) (dimensions : BrickDimensions) : ℕ :=
  let minHeight := numBricks * dimensions.length
  let maxAdditionalHeight := numBricks * (dimensions.height - dimensions.length)
  (maxAdditionalHeight / 5 + 1 : ℕ)

/-- Theorem stating the number of different tower heights achievable -/
theorem tower_heights_count (numBricks : ℕ) (dimensions : BrickDimensions) :
  numBricks = 78 →
  dimensions = { length := 3, width := 8, height := 20 } →
  calculateTowerHeights numBricks dimensions = 266 := by
  sorry

end NUMINAMATH_CALUDE_tower_heights_count_l421_42141


namespace NUMINAMATH_CALUDE_cherry_tomato_jars_l421_42155

theorem cherry_tomato_jars (total_tomatoes : ℕ) (tomatoes_per_jar : ℕ) (h1 : total_tomatoes = 56) (h2 : tomatoes_per_jar = 8) :
  (total_tomatoes / tomatoes_per_jar : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tomato_jars_l421_42155


namespace NUMINAMATH_CALUDE_octal_to_decimal_l421_42195

theorem octal_to_decimal (n : ℕ) (h : n = 246) : 
  2 * 8^2 + 4 * 8^1 + 6 * 8^0 = 166 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_l421_42195


namespace NUMINAMATH_CALUDE_terrys_total_spending_l421_42169

/-- Terry's spending over three days --/
def terrys_spending (monday_amount : ℝ) : ℝ :=
  let tuesday_amount := 2 * monday_amount
  let wednesday_amount := 2 * (monday_amount + tuesday_amount)
  monday_amount + tuesday_amount + wednesday_amount

/-- Theorem: Terry's total spending is $54 --/
theorem terrys_total_spending : terrys_spending 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_terrys_total_spending_l421_42169


namespace NUMINAMATH_CALUDE_xy_neq_one_condition_l421_42138

theorem xy_neq_one_condition (x y : ℝ) :
  (∃ x y : ℝ, (x ≠ 1 ∨ y ≠ 1) ∧ x * y = 1) ∧
  (x * y ≠ 1 → (x ≠ 1 ∨ y ≠ 1)) :=
by sorry

end NUMINAMATH_CALUDE_xy_neq_one_condition_l421_42138


namespace NUMINAMATH_CALUDE_butterfat_mixture_proof_l421_42156

/-- Proves that mixing 8 gallons of 35% butterfat milk with 12 gallons of 10% butterfat milk
    results in a mixture that is 20% butterfat. -/
theorem butterfat_mixture_proof :
  let x : ℝ := 8 -- Amount of 35% butterfat milk in gallons
  let y : ℝ := 12 -- Amount of 10% butterfat milk in gallons
  let butterfat_high : ℝ := 0.35 -- Percentage of butterfat in high-fat milk
  let butterfat_low : ℝ := 0.10 -- Percentage of butterfat in low-fat milk
  let butterfat_target : ℝ := 0.20 -- Target percentage of butterfat in mixture
  (butterfat_high * x + butterfat_low * y) / (x + y) = butterfat_target :=
by sorry

end NUMINAMATH_CALUDE_butterfat_mixture_proof_l421_42156


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_l421_42176

theorem quadratic_roots_imply_m (m : ℝ) : 
  (∃ x : ℂ, 5 * x^2 - 2 * x + m = 0 ∧ x = (2 + Complex.I * Real.sqrt 78) / 10) ∧
  (∃ x : ℂ, 5 * x^2 - 2 * x + m = 0 ∧ x = (2 - Complex.I * Real.sqrt 78) / 10) →
  m = 41 / 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_l421_42176


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l421_42110

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 3) ↔ x ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l421_42110


namespace NUMINAMATH_CALUDE_negation_of_proposition_l421_42119

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x > 0 → ¬(x > 0)) ↔ ∃ x : ℝ, x ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l421_42119


namespace NUMINAMATH_CALUDE_hiking_team_gloves_l421_42153

/-- The minimum number of gloves needed for a hiking team -/
theorem hiking_team_gloves (num_participants : ℕ) (gloves_per_participant : ℕ) 
  (h1 : num_participants = 63)
  (h2 : gloves_per_participant = 3) : 
  num_participants * gloves_per_participant = 189 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_gloves_l421_42153


namespace NUMINAMATH_CALUDE_min_value_quadratic_root_condition_l421_42117

/-- Given a quadratic equation x^2 + ax + b - 3 = 0 with a real root in [1,2],
    the minimum value of a^2 + (b-4)^2 is 2 -/
theorem min_value_quadratic_root_condition (a b : ℝ) :
  (∃ x : ℝ, x^2 + a*x + b - 3 = 0 ∧ 1 ≤ x ∧ x ≤ 2) →
  (∀ a' b' : ℝ, (∃ x : ℝ, x^2 + a'*x + b' - 3 = 0 ∧ 1 ≤ x ∧ x ≤ 2) →
    a^2 + (b-4)^2 ≤ a'^2 + (b'-4)^2) →
  a^2 + (b-4)^2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_min_value_quadratic_root_condition_l421_42117


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l421_42183

theorem complex_number_quadrant : ∃ (z : ℂ), z = 2 / (1 + Complex.I) ∧ Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l421_42183


namespace NUMINAMATH_CALUDE_ratio_problem_l421_42154

theorem ratio_problem (a b : ℝ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l421_42154


namespace NUMINAMATH_CALUDE_average_of_multiples_of_10_l421_42123

theorem average_of_multiples_of_10 : 
  let multiples := List.filter (fun n => n % 10 = 0) (List.range 201)
  (List.sum multiples) / multiples.length = 105 := by
sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_10_l421_42123


namespace NUMINAMATH_CALUDE_blue_chairs_count_l421_42149

/-- Represents the number of chairs of each color in a classroom --/
structure Classroom where
  blue : ℕ
  green : ℕ
  white : ℕ

/-- Defines the conditions for the classroom chair problem --/
def validClassroom (c : Classroom) : Prop :=
  c.green = 3 * c.blue ∧
  c.white = c.blue + c.green - 13 ∧
  c.blue + c.green + c.white = 67

/-- Theorem stating that in a valid classroom, there are 10 blue chairs --/
theorem blue_chairs_count (c : Classroom) (h : validClassroom c) : c.blue = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_chairs_count_l421_42149


namespace NUMINAMATH_CALUDE_journey_length_l421_42125

theorem journey_length :
  ∀ (total : ℚ),
  (1 / 4 : ℚ) * total +  -- First part
  30 +                   -- Second part (city)
  (1 / 7 : ℚ) * total    -- Third part
  = total                -- Sum of all parts equals total
  →
  total = 840 / 17 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_length_l421_42125


namespace NUMINAMATH_CALUDE_store_inventory_count_l421_42128

theorem store_inventory_count : 
  ∀ (original_price : ℝ) (discount_rate : ℝ) (sold_percentage : ℝ) 
    (debt : ℝ) (remaining : ℝ),
  original_price = 50 →
  discount_rate = 0.8 →
  sold_percentage = 0.9 →
  debt = 15000 →
  remaining = 3000 →
  (((1 - discount_rate) * original_price * sold_percentage) * 
    (debt + remaining) / ((1 - discount_rate) * original_price * sold_percentage)) = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_store_inventory_count_l421_42128


namespace NUMINAMATH_CALUDE_practice_hours_until_game_l421_42173

/-- Calculates the total practice hours for a given number of weeks -/
def total_practice_hours (weeks : ℕ) : ℕ :=
  let weekday_hours := 3
  let weekday_count := 5
  let saturday_hours := 5
  let weekly_hours := weekday_hours * weekday_count + saturday_hours
  weekly_hours * weeks

/-- The number of weeks until the next game -/
def weeks_until_game : ℕ := 3

theorem practice_hours_until_game :
  total_practice_hours weeks_until_game = 60 := by
  sorry

end NUMINAMATH_CALUDE_practice_hours_until_game_l421_42173


namespace NUMINAMATH_CALUDE_inequality_holds_l421_42137

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^3 + k * log x

theorem inequality_holds (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : k ≥ -3) 
  (h2 : x₁ ≥ 1) 
  (h3 : x₂ ≥ 1) 
  (h4 : x₁ > x₂) : 
  (deriv (f k) x₁ + deriv (f k) x₂) / 2 > (f k x₁ - f k x₂) / (x₁ - x₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_l421_42137


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l421_42132

theorem quadratic_equation_solution (y : ℝ) : 
  y^2 + 6*y + 8 = -(y + 4)*(y + 6) ↔ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l421_42132


namespace NUMINAMATH_CALUDE_hcf_of_two_numbers_l421_42194

theorem hcf_of_two_numbers (a b : ℕ) : 
  (a > 0) → 
  (b > 0) → 
  (a ≤ b) → 
  (b = 1071) → 
  (∃ k : ℕ, Nat.lcm a b = k * 11 * 17) → 
  Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_two_numbers_l421_42194


namespace NUMINAMATH_CALUDE_correct_article_usage_l421_42148

-- Define the possible article types
inductive Article
  | Definite
  | Indefinite
  | NoArticle

-- Define the context of a noun
structure NounContext where
  isSpecific : Bool

-- Define the function to determine the correct article
def correctArticle (context : NounContext) : Article :=
  if context.isSpecific then Article.Definite else Article.Indefinite

-- Theorem statement
theorem correct_article_usage 
  (keyboard_context : NounContext)
  (computer_context : NounContext)
  (h1 : keyboard_context.isSpecific = true)
  (h2 : computer_context.isSpecific = false) :
  (correctArticle keyboard_context = Article.Definite) ∧
  (correctArticle computer_context = Article.Indefinite) := by
  sorry


end NUMINAMATH_CALUDE_correct_article_usage_l421_42148


namespace NUMINAMATH_CALUDE_constant_d_value_l421_42124

theorem constant_d_value (e c f : ℝ) : 
  (∃ d : ℝ, ∀ x : ℝ, 
    (3 * x^3 - 2 * x^2 + x - 5/4) * (e * x^3 + d * x^2 + c * x + f) = 
    9 * x^6 - 5 * x^5 - x^4 + 20 * x^3 - 25/4 * x^2 + 15/4 * x - 5/2) →
  (∃ d : ℝ, d = 1/3) := by
sorry

end NUMINAMATH_CALUDE_constant_d_value_l421_42124


namespace NUMINAMATH_CALUDE_percentage_difference_l421_42157

theorem percentage_difference (p j t : ℝ) 
  (hj : j = 0.75 * p) 
  (ht : t = 0.9375 * p) : 
  (t - j) / t = 0.2 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l421_42157


namespace NUMINAMATH_CALUDE_ten_factorial_minus_nine_factorial_l421_42134

-- Define factorial function
def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

-- State the theorem
theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_ten_factorial_minus_nine_factorial_l421_42134


namespace NUMINAMATH_CALUDE_puzzle_solution_l421_42193

/-- Represents the animals in the puzzle -/
inductive Animal : Type
  | Cat | Chicken | Crab | Bear | Goat

/-- Represents the puzzle grid -/
def Grid := Animal → Nat

/-- Checks if the grid satisfies the sum conditions -/
def satisfies_sums (g : Grid) : Prop :=
  g Animal.Crab + g Animal.Crab + g Animal.Crab + g Animal.Crab + g Animal.Crab = 10 ∧
  g Animal.Goat + g Animal.Goat + g Animal.Crab + g Animal.Bear + g Animal.Bear = 16 ∧
  g Animal.Crab + g Animal.Chicken + g Animal.Chicken + g Animal.Goat + g Animal.Crab = 17 ∧
  g Animal.Cat + g Animal.Bear + g Animal.Goat + g Animal.Goat + g Animal.Crab = 13

/-- Checks if all animals have different values -/
def all_different (g : Grid) : Prop :=
  ∀ a b : Animal, a ≠ b → g a ≠ g b

/-- The main theorem stating the unique solution -/
theorem puzzle_solution :
  ∃! g : Grid, satisfies_sums g ∧ all_different g ∧
    g Animal.Cat = 1 ∧ g Animal.Chicken = 5 ∧ g Animal.Crab = 2 ∧
    g Animal.Bear = 4 ∧ g Animal.Goat = 3 :=
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l421_42193


namespace NUMINAMATH_CALUDE_fraction_calculation_l421_42186

theorem fraction_calculation :
  (1 / 5 + 1 / 7) / (3 / 8 + 2 / 9) = 864 / 1505 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l421_42186


namespace NUMINAMATH_CALUDE_solutions_of_f_f_x_eq_x_l421_42118

def f (x : ℝ) : ℝ := x^2 - 4*x - 5

theorem solutions_of_f_f_x_eq_x :
  let s₁ := (5 + 3*Real.sqrt 5) / 2
  let s₂ := (5 - 3*Real.sqrt 5) / 2
  let s₃ := (3 + Real.sqrt 41) / 2
  let s₄ := (3 - Real.sqrt 41) / 2
  (∀ x : ℝ, f (f x) = x ↔ x = s₁ ∨ x = s₂ ∨ x = s₃ ∨ x = s₄) :=
by sorry

end NUMINAMATH_CALUDE_solutions_of_f_f_x_eq_x_l421_42118


namespace NUMINAMATH_CALUDE_donut_combinations_l421_42121

/-- The number of ways to distribute n identical objects into k distinct boxes -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The total number of donuts Josh needs to buy -/
def total_donuts : ℕ := 8

/-- The number of different types of donuts -/
def donut_types : ℕ := 5

/-- The number of donuts Josh must buy of the first type -/
def first_type_min : ℕ := 2

/-- The number of donuts Josh must buy of each other type -/
def other_types_min : ℕ := 1

/-- The number of remaining donuts to distribute after meeting minimum requirements -/
def remaining_donuts : ℕ := total_donuts - (first_type_min + (donut_types - 1) * other_types_min)

theorem donut_combinations : stars_and_bars remaining_donuts donut_types = 15 := by
  sorry

end NUMINAMATH_CALUDE_donut_combinations_l421_42121


namespace NUMINAMATH_CALUDE_missy_claims_count_l421_42150

/-- The number of insurance claims that can be handled by three agents --/
def insurance_claims (jan_claims : ℕ) : ℕ × ℕ × ℕ :=
  let john_claims := jan_claims + (jan_claims * 30 / 100)
  let missy_claims := john_claims + 15
  (jan_claims, john_claims, missy_claims)

/-- Theorem stating that Missy can handle 41 claims given the conditions --/
theorem missy_claims_count :
  let (jan, john, missy) := insurance_claims 20
  missy = 41 := by sorry

end NUMINAMATH_CALUDE_missy_claims_count_l421_42150


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l421_42189

def Rectangle (length width : ℝ) : Prop :=
  length > 0 ∧ width > 0

def Area (length width : ℝ) : ℝ :=
  length * width

def Perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

theorem rectangle_dimensions :
  ∀ length width : ℝ,
    Rectangle length width →
    Area length width = 12 →
    Perimeter length width = 26 →
    (length = 1 ∧ width = 12) ∨ (length = 12 ∧ width = 1) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l421_42189


namespace NUMINAMATH_CALUDE_subtraction_difference_l421_42130

theorem subtraction_difference : 
  let total : ℝ := 7000
  let one_tenth : ℝ := 1 / 10
  let one_tenth_percent : ℝ := 1 / 1000
  (one_tenth * total) - (one_tenth_percent * total) = 693 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_difference_l421_42130


namespace NUMINAMATH_CALUDE_original_average_calculation_l421_42115

theorem original_average_calculation (S : Finset ℝ) (A : ℝ) :
  Finset.card S = 10 →
  (Finset.sum S id + 8) / 10 = 7 →
  Finset.sum S id = 10 * A →
  A = 6.2 := by
  sorry

end NUMINAMATH_CALUDE_original_average_calculation_l421_42115


namespace NUMINAMATH_CALUDE_simplify_expression_l421_42175

theorem simplify_expression (x : ℝ) : (3 * x^4)^2 * (2 * x^2)^3 = 72 * x^14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l421_42175


namespace NUMINAMATH_CALUDE_savings_ratio_first_year_l421_42133

/-- Represents the financial situation of a person over two years -/
structure FinancialSituation where
  firstYearIncome : ℝ
  firstYearSavingsRatio : ℝ
  incomeIncrease : ℝ
  savingsIncrease : ℝ

/-- The theorem stating the savings ratio in the first year -/
theorem savings_ratio_first_year 
  (fs : FinancialSituation)
  (h1 : fs.incomeIncrease = 0.3)
  (h2 : fs.savingsIncrease = 1.0)
  (h3 : fs.firstYearIncome > 0)
  (h4 : 0 ≤ fs.firstYearSavingsRatio ∧ fs.firstYearSavingsRatio ≤ 1) :
  let firstYearExpenditure := fs.firstYearIncome * (1 - fs.firstYearSavingsRatio)
  let secondYearIncome := fs.firstYearIncome * (1 + fs.incomeIncrease)
  let secondYearSavings := fs.firstYearIncome * fs.firstYearSavingsRatio * (1 + fs.savingsIncrease)
  let secondYearExpenditure := secondYearIncome - secondYearSavings
  firstYearExpenditure + secondYearExpenditure = 2 * firstYearExpenditure →
  fs.firstYearSavingsRatio = 0.3 := by
sorry

end NUMINAMATH_CALUDE_savings_ratio_first_year_l421_42133


namespace NUMINAMATH_CALUDE_inequality_system_solution_l421_42179

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 ≥ 1 ∧ x ≥ a) ↔ x ≥ 2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l421_42179


namespace NUMINAMATH_CALUDE_ninth_grade_basketball_tournament_l421_42151

theorem ninth_grade_basketball_tournament (n : ℕ) : 
  (n * (n - 1)) / 2 = 45 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_basketball_tournament_l421_42151


namespace NUMINAMATH_CALUDE_folded_carbon_copies_l421_42181

/-- Represents the number of carbon copies produced given the initial number of sheets,
    carbon papers, and whether the setup is folded or not -/
def carbonCopies (sheets : ℕ) (carbons : ℕ) (folded : Bool) : ℕ :=
  if folded then
    2 * (sheets - 1)
  else
    carbons

/-- Theorem stating that with 3 sheets, 2 carbons, and folded setup, 4 carbon copies are produced -/
theorem folded_carbon_copies :
  carbonCopies 3 2 true = 4 := by sorry

end NUMINAMATH_CALUDE_folded_carbon_copies_l421_42181


namespace NUMINAMATH_CALUDE_original_denominator_proof_l421_42139

theorem original_denominator_proof (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = (1 : ℚ) / 3 →
  d = 23 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l421_42139


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l421_42126

theorem no_prime_roots_for_quadratic : ¬∃ (k : ℤ), ∃ (p q : ℕ), 
  Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p : ℤ) + q = 57 ∧ (p : ℤ) * q = k ∧
  ∀ (x : ℤ), x^2 - 57*x + k = 0 ↔ x = p ∨ x = q := by
  sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l421_42126


namespace NUMINAMATH_CALUDE_exists_self_appended_perfect_square_l421_42167

theorem exists_self_appended_perfect_square :
  ∃ (A : ℕ), ∃ (n : ℕ), ∃ (B : ℕ),
    A > 0 ∧ 
    10^n ≤ A ∧ A < 10^(n+1) ∧
    A * (10^n + 1) = B^2 :=
sorry

end NUMINAMATH_CALUDE_exists_self_appended_perfect_square_l421_42167


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l421_42158

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence with given conditions,
    the general term formula is a_n = 4 - 2n. -/
theorem arithmetic_sequence_formula 
  (a : ℕ → ℤ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_3 : a 3 = -2) 
  (h_7 : a 7 = -10) :
  ∃ c : ℤ, ∀ n : ℕ, a n = 4 - 2 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l421_42158


namespace NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l421_42197

-- System 1
theorem system_1_solution :
  ∃ (x y : ℝ), 2 * x + 3 * y = 9 ∧ x = 2 * y + 1 ∧ x = 3 ∧ y = 1 := by sorry

-- System 2
theorem system_2_solution :
  ∃ (x y : ℝ), 2 * x - y = 6 ∧ 3 * x + 2 * y = 2 ∧ x = 2 ∧ y = -2 := by sorry

end NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l421_42197


namespace NUMINAMATH_CALUDE_can_cut_one_more_square_l421_42142

/-- Represents a grid of cells -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square region in the grid -/
structure Square :=
  (size : ℕ)

/-- The number of 2x2 squares that can fit in a grid -/
def count_2x2_squares (g : Grid) : ℕ :=
  ((g.rows - 1) / 2) * ((g.cols - 1) / 2)

theorem can_cut_one_more_square (g : Grid) (s : Square) (n : ℕ) :
  g.rows = 29 →
  g.cols = 29 →
  s.size = 2 →
  n = 99 →
  n < count_2x2_squares g →
  ∃ (remaining : ℕ), remaining > 0 ∧ remaining = count_2x2_squares g - n :=
by sorry

end NUMINAMATH_CALUDE_can_cut_one_more_square_l421_42142


namespace NUMINAMATH_CALUDE_milk_packet_content_l421_42131

theorem milk_packet_content 
  (num_packets : ℕ) 
  (oz_to_ml : ℝ) 
  (total_oz : ℝ) 
  (h1 : num_packets = 150)
  (h2 : oz_to_ml = 30)
  (h3 : total_oz = 1250) :
  (total_oz * oz_to_ml) / num_packets = 250 := by
sorry

end NUMINAMATH_CALUDE_milk_packet_content_l421_42131


namespace NUMINAMATH_CALUDE_five_card_draw_probability_l421_42152

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards to be drawn -/
def CardsDrawn : ℕ := 5

/-- Represents the number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- The probability of drawing 5 cards from a standard 52-card deck without replacement,
    such that there is one card from each suit and the fifth card is from hearts -/
theorem five_card_draw_probability :
  (1 : ℚ) * (CardsPerSuit : ℚ) / (StandardDeck - 1 : ℚ) *
  (CardsPerSuit : ℚ) / (StandardDeck - 2 : ℚ) *
  (CardsPerSuit : ℚ) / (StandardDeck - 3 : ℚ) *
  (CardsPerSuit - 1 : ℚ) / (StandardDeck - 4 : ℚ) =
  2197 / 83300 :=
by sorry

end NUMINAMATH_CALUDE_five_card_draw_probability_l421_42152


namespace NUMINAMATH_CALUDE_two_by_one_cuboid_net_l421_42140

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of unit squares in the net of a cuboid -/
def net_squares (c : Cuboid) : ℕ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Theorem: A 2x1x1 cuboid's net has 10 unit squares, and removing any one leaves 9 -/
theorem two_by_one_cuboid_net :
  let c : Cuboid := ⟨2, 1, 1⟩
  net_squares c = 10 ∧ net_squares c - 1 = 9 := by
  sorry

#eval net_squares ⟨2, 1, 1⟩

end NUMINAMATH_CALUDE_two_by_one_cuboid_net_l421_42140


namespace NUMINAMATH_CALUDE_missing_number_proof_l421_42192

theorem missing_number_proof : ∃ x : ℚ, (306 / 34) * 15 + x = 405 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l421_42192


namespace NUMINAMATH_CALUDE_money_sum_l421_42170

theorem money_sum (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 320) (h3 : C = 20) :
  A + B + C = 500 := by
  sorry

end NUMINAMATH_CALUDE_money_sum_l421_42170


namespace NUMINAMATH_CALUDE_inverse_of_A_zero_matrix_if_not_invertible_l421_42109

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 7; 2, 6]

theorem inverse_of_A :
  let inv_A := !![0.6, -0.7; -0.2, 0.4]
  A.det ≠ 0 → A * inv_A = 1 ∧ inv_A * A = 1 :=
by sorry

theorem zero_matrix_if_not_invertible :
  A.det = 0 → A⁻¹ = 0 :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_A_zero_matrix_if_not_invertible_l421_42109


namespace NUMINAMATH_CALUDE_distance_AB_is_40_l421_42147

/-- The distance between two points A and B, where two cyclists start simultaneously. -/
def distance_AB : ℝ := 40

/-- The remaining distance for the second cyclist when the first cyclist has traveled half the total distance. -/
def remaining_distance_second : ℝ := 24

/-- The remaining distance for the first cyclist when the second cyclist has traveled half the total distance. -/
def remaining_distance_first : ℝ := 15

/-- Theorem stating that the distance between points A and B is 40 km, given the conditions of the cycling problem. -/
theorem distance_AB_is_40 :
  (distance_AB / 2 + remaining_distance_second = distance_AB) ∧
  (distance_AB / 2 + remaining_distance_first = distance_AB) →
  distance_AB = 40 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_40_l421_42147


namespace NUMINAMATH_CALUDE_cuts_through_examples_l421_42105

/-- A line cuts through a curve at a point if it's tangent to the curve at that point
    and the curve lies on both sides of the line near that point. -/
def cuts_through (l : ℝ → ℝ) (c : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  (∀ x, l x = c x → x = p.1) ∧  -- l is tangent to c at p
  (∃ ε > 0, ∀ x, |x - p.1| < ε → 
    ((c x > l x ∧ x < p.1) ∨ (c x < l x ∧ x > p.1) ∨
     (c x < l x ∧ x < p.1) ∨ (c x > l x ∧ x > p.1)))

theorem cuts_through_examples :
  (cuts_through (λ _ ↦ 0) (λ x ↦ x^3) (0, 0)) ∧
  (cuts_through (λ x ↦ x) Real.sin (0, 0)) ∧
  (cuts_through (λ x ↦ x) Real.tan (0, 0)) :=
sorry

end NUMINAMATH_CALUDE_cuts_through_examples_l421_42105


namespace NUMINAMATH_CALUDE_measure_eight_liters_possible_l421_42129

/-- Represents the state of the buckets -/
structure BucketState where
  b10 : ℕ  -- Amount of water in the 10-liter bucket
  b6 : ℕ   -- Amount of water in the 6-liter bucket

/-- Represents a single operation on the buckets -/
inductive BucketOperation
  | FillFromRiver (bucket : ℕ)  -- Fill a bucket from the river
  | EmptyToRiver (bucket : ℕ)   -- Empty a bucket to the river
  | PourBetweenBuckets          -- Pour from one bucket to another

/-- Applies a single operation to the current state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  sorry

/-- Checks if the given sequence of operations results in 8 liters in one bucket -/
def isValidSolution (operations : List BucketOperation) : Bool :=
  sorry

/-- Theorem stating that it's possible to measure 8 liters using the given buckets -/
theorem measure_eight_liters_possible :
  ∃ (operations : List BucketOperation), isValidSolution operations :=
  sorry

end NUMINAMATH_CALUDE_measure_eight_liters_possible_l421_42129


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l421_42100

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l421_42100


namespace NUMINAMATH_CALUDE_distance_between_points_l421_42145

/-- The distance between two points given rowing speed, stream speed, and round trip time -/
theorem distance_between_points (rowing_speed stream_speed : ℝ) (round_trip_time : ℝ) :
  rowing_speed = 10 →
  stream_speed = 2 →
  round_trip_time = 5 →
  ∃ (distance : ℝ),
    distance / (rowing_speed + stream_speed) + distance / (rowing_speed - stream_speed) = round_trip_time ∧
    distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l421_42145


namespace NUMINAMATH_CALUDE_square_root_fraction_simplification_l421_42144

theorem square_root_fraction_simplification :
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (25 + 16)) = (17 * Real.sqrt 41) / 41 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_simplification_l421_42144


namespace NUMINAMATH_CALUDE_f_min_value_l421_42160

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

-- State the theorem
theorem f_min_value (a b : ℝ) :
  (∀ x > 0, f a b x ≤ 5) ∧ (∃ x > 0, f a b x = 5) →
  (∀ x < 0, f a b x ≥ -1) ∧ (∃ x < 0, f a b x = -1) :=
by sorry

end NUMINAMATH_CALUDE_f_min_value_l421_42160


namespace NUMINAMATH_CALUDE_stack_height_l421_42178

/-- Calculates the vertical distance of a stack of linked rings -/
def verticalDistance (topDiameter : ℕ) (bottomDiameter : ℕ) (thickness : ℕ) : ℕ :=
  let numberOfRings := (topDiameter - bottomDiameter) / 2 + 1
  let innerDiameterSum := (numberOfRings * (topDiameter - thickness * 2 + bottomDiameter - thickness * 2)) / 2
  innerDiameterSum + thickness * 2

/-- The vertical distance of the stack of rings is 76 cm -/
theorem stack_height : verticalDistance 20 4 2 = 76 := by
  sorry

end NUMINAMATH_CALUDE_stack_height_l421_42178


namespace NUMINAMATH_CALUDE_teal_color_survey_l421_42114

theorem teal_color_survey (total : ℕ) (green : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  green = 90 →
  both = 40 →
  neither = 20 →
  ∃ blue : ℕ, blue = 80 ∧ blue + green - both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l421_42114


namespace NUMINAMATH_CALUDE_paint_usage_l421_42184

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ)
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1 / 9)
  (h3 : second_week_fraction = 1 / 5) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  let total_usage := first_week_usage + second_week_usage
  total_usage = 104 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_l421_42184


namespace NUMINAMATH_CALUDE_fraction_comparison_l421_42196

theorem fraction_comparison : 
  (14 / 10 : ℚ) = 7 / 5 ∧ 
  (1 + 2 / 5 : ℚ) = 7 / 5 ∧ 
  (1 + 4 / 20 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 2 / 6 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 28 / 20 : ℚ) ≠ 7 / 5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l421_42196


namespace NUMINAMATH_CALUDE_equation_roots_l421_42107

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => 3*x^4 - 2*x^3 - 7*x^2 - 2*x + 3
  ∃ (a b c d : ℝ), 
    (a = (1 + Real.sqrt 5) / 2) ∧
    (b = (1 - Real.sqrt 5) / 2) ∧
    (c = (-1 + Real.sqrt 37) / 6) ∧
    (d = (-1 - Real.sqrt 37) / 6) ∧
    (∀ x : ℝ, f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l421_42107
