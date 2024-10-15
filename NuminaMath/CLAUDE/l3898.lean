import Mathlib

namespace NUMINAMATH_CALUDE_fourth_power_sum_l3898_389883

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 3)
  (sum_cubes : a^3 + b^3 + c^3 = 6) :
  a^4 + b^4 + c^4 = 34/3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l3898_389883


namespace NUMINAMATH_CALUDE_remaining_apples_l3898_389821

def initial_apples : ℕ := 150

def sold_to_jill (apples : ℕ) : ℕ :=
  apples - (apples * 30 / 100)

def sold_to_june (apples : ℕ) : ℕ :=
  apples - (apples * 20 / 100)

def give_to_teacher (apples : ℕ) : ℕ :=
  apples - 1

theorem remaining_apples :
  give_to_teacher (sold_to_june (sold_to_jill initial_apples)) = 83 := by
  sorry

end NUMINAMATH_CALUDE_remaining_apples_l3898_389821


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l3898_389866

/-- The height of a square-based pyramid with the same volume as a cube -/
theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (h : ℝ) : 
  cube_edge = 5 →
  pyramid_base = 10 →
  (1 / 3) * pyramid_base^2 * h = cube_edge^3 →
  h = 3.75 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l3898_389866


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_equals_one_l3898_389812

theorem intersection_equality_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {1, 2, 5}
  let B : Set ℝ := {a + 4, a}
  A ∩ B = B → a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_equals_one_l3898_389812


namespace NUMINAMATH_CALUDE_dice_probability_l3898_389873

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability of a single die showing an even number -/
def p_even : ℚ := 1/2

/-- The probability of a single die showing an odd number -/
def p_odd : ℚ := 1/2

/-- The number of dice required to show even numbers -/
def num_even : ℕ := 4

/-- The number of dice required to show odd numbers -/
def num_odd : ℕ := 4

theorem dice_probability : 
  (Nat.choose num_dice num_even : ℚ) * p_even ^ num_dice = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l3898_389873


namespace NUMINAMATH_CALUDE_triangle_max_area_l3898_389869

noncomputable def triangle_area (x : Real) : Real :=
  4 * Real.sqrt 3 * Real.sin x * Real.sin ((2 * Real.pi / 3) - x)

theorem triangle_max_area :
  ∀ x : Real, 0 < x → x < 2 * Real.pi / 3 →
    triangle_area x ≤ triangle_area (Real.pi / 3) ∧
    triangle_area (Real.pi / 3) = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3898_389869


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3898_389857

theorem max_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0 →
  x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0 →
  x₁ ≠ x₂ →
  ∃ (M : ℝ), M = 18 ∧ x₁^2 + x₂^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3898_389857


namespace NUMINAMATH_CALUDE_number_solution_l3898_389870

theorem number_solution : ∃ x : ℝ, (35 - 3 * x = 14) ∧ (x = 7) := by sorry

end NUMINAMATH_CALUDE_number_solution_l3898_389870


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3898_389832

theorem ratio_x_to_y (x y : ℝ) (h : 0.8 * x = 0.2 * y) : x / y = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3898_389832


namespace NUMINAMATH_CALUDE_ellipse_property_l3898_389891

/-- Definition of an ellipse with foci F₁ and F₂ -/
def Ellipse (F₁ F₂ : ℝ × ℝ) (a b : ℝ) :=
  {P : ℝ × ℝ | (P.1^2 / a^2) + (P.2^2 / b^2) = 1 ∧ a > b ∧ b > 0}

/-- The angle between two vectors -/
def angle (v₁ v₂ : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its vertices -/
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_property (F₁ F₂ : ℝ × ℝ) (a b : ℝ) (P : ℝ × ℝ) :
  P ∈ Ellipse F₁ F₂ a b →
  angle (P.1 - F₁.1, P.2 - F₁.2) (P.1 - F₂.1, P.2 - F₂.2) = π / 3 →
  triangle_area P F₁ F₂ = 3 * Real.sqrt 3 →
  b = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_property_l3898_389891


namespace NUMINAMATH_CALUDE_smallest_positive_a_l3898_389843

-- Define the equation
def equation (x a : ℚ) : Prop :=
  (((x - a) / 2 + (x - 2*a) / 3) / ((x + 4*a) / 5 - (x + 3*a) / 4)) =
  (((x - 3*a) / 4 + (x - 4*a) / 5) / ((x + 2*a) / 3 - (x + a) / 2))

-- Define what it means for the equation to have an integer root
def has_integer_root (a : ℚ) : Prop :=
  ∃ x : ℤ, equation x a

-- State the theorem
theorem smallest_positive_a : 
  (∀ a : ℚ, 0 < a ∧ a < 419/421 → ¬ has_integer_root a) ∧ 
  has_integer_root (419/421) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_a_l3898_389843


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l3898_389849

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (25 - x) = 9) →
  ((10 + x) * (25 - x) = 529) := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l3898_389849


namespace NUMINAMATH_CALUDE_largest_prime_to_test_for_500_to_550_l3898_389863

theorem largest_prime_to_test_for_500_to_550 (n : ℕ) :
  500 ≤ n ∧ n ≤ 550 →
  (∀ p : ℕ, Prime p ∧ p ≤ Real.sqrt n → p ≤ 23) ∧
  Prime 23 ∧ 23 ≤ Real.sqrt n :=
sorry

end NUMINAMATH_CALUDE_largest_prime_to_test_for_500_to_550_l3898_389863


namespace NUMINAMATH_CALUDE_reading_time_is_fifty_l3898_389819

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of paragraphs per page -/
def paragraphs_per_page : ℕ := 20

/-- Represents the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ := 10

/-- Represents the total number of pages in the book -/
def total_pages : ℕ := 50

/-- Calculates the total time in hours needed to read the book -/
def reading_time : ℚ :=
  (total_pages * paragraphs_per_page * sentences_per_paragraph) / reading_speed

theorem reading_time_is_fifty : reading_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_is_fifty_l3898_389819


namespace NUMINAMATH_CALUDE_edwin_alvin_age_fraction_l3898_389811

/-- The fraction of Alvin's age that Edwin will be 20 more than in two years -/
def fraction_of_alvins_age : ℚ := 1 / 29

theorem edwin_alvin_age_fraction :
  let alvin_current_age : ℚ := (30.99999999 - 6) / 2
  let edwin_current_age : ℚ := alvin_current_age + 6
  let alvin_future_age : ℚ := alvin_current_age + 2
  let edwin_future_age : ℚ := edwin_current_age + 2
  edwin_future_age = fraction_of_alvins_age * alvin_future_age + 20 :=
by sorry

end NUMINAMATH_CALUDE_edwin_alvin_age_fraction_l3898_389811


namespace NUMINAMATH_CALUDE_seven_items_ten_people_distribution_l3898_389887

/-- The number of ways to distribute n unique items among m people,
    where no more than k people receive at least one item. -/
def distribution_ways (n m k : ℕ) : ℕ :=
  (Nat.choose m k) * (k^n)

/-- Theorem stating the correct number of ways to distribute
    7 unique items among 10 people, where no more than 3 people
    receive at least one item. -/
theorem seven_items_ten_people_distribution :
  distribution_ways 7 10 3 = 262440 := by
  sorry

end NUMINAMATH_CALUDE_seven_items_ten_people_distribution_l3898_389887


namespace NUMINAMATH_CALUDE_cyclists_initial_distance_l3898_389835

/-- The initial distance between two cyclists -/
def initial_distance : ℝ := 50

/-- The speed of each cyclist -/
def cyclist_speed : ℝ := 10

/-- The speed of the fly -/
def fly_speed : ℝ := 15

/-- The total distance covered by the fly -/
def fly_distance : ℝ := 37.5

/-- Theorem stating that the initial distance between the cyclists is 50 miles -/
theorem cyclists_initial_distance :
  initial_distance = 
    (2 * cyclist_speed * fly_distance) / fly_speed :=
by sorry

end NUMINAMATH_CALUDE_cyclists_initial_distance_l3898_389835


namespace NUMINAMATH_CALUDE_same_day_ticket_cost_l3898_389864

/-- Proves that the cost of same-day tickets is $30 given the specified conditions -/
theorem same_day_ticket_cost
  (total_tickets : ℕ)
  (total_receipts : ℕ)
  (advance_ticket_cost : ℕ)
  (advance_tickets_sold : ℕ)
  (h1 : total_tickets = 60)
  (h2 : total_receipts = 1600)
  (h3 : advance_ticket_cost = 20)
  (h4 : advance_tickets_sold = 20) :
  (total_receipts - advance_ticket_cost * advance_tickets_sold) / (total_tickets - advance_tickets_sold) = 30 :=
by sorry

end NUMINAMATH_CALUDE_same_day_ticket_cost_l3898_389864


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3898_389879

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_4 + a_8 = 16, a_6 = 8 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 4 + a 8 = 16) : 
  a 6 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3898_389879


namespace NUMINAMATH_CALUDE_divisibility_condition_l3898_389813

theorem divisibility_condition (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3898_389813


namespace NUMINAMATH_CALUDE_f_upper_bound_f_negative_l3898_389861

/-- The function f(x) = ax^2 - (a+1)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

/-- Theorem stating the range of a for which f(x) ≤ 2 for all x in ℝ -/
theorem f_upper_bound (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 2) ↔ -3 - 2 * Real.sqrt 2 ≤ a ∧ a ≤ -3 + 2 * Real.sqrt 2 :=
sorry

/-- Theorem describing the solution set of f(x) < 0 for different ranges of a -/
theorem f_negative (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f a x < 0 ↔ 
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨
     (a > 1 ∧ 1/a < x ∧ x < 1) ∨
     (a < 0 ∧ ((x < 1/a) ∨ (x > 1))))) :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_f_negative_l3898_389861


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3898_389840

theorem min_value_quadratic (x : ℝ) : 
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 4 * x^2 + 8 * x + 16 → y ≥ y_min ∧ y_min = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3898_389840


namespace NUMINAMATH_CALUDE_sticker_collection_l3898_389801

theorem sticker_collection (karl_stickers : ℕ) : 
  (∃ (ryan_stickers ben_stickers : ℕ),
    ryan_stickers = karl_stickers + 20 ∧
    ben_stickers = ryan_stickers - 10 ∧
    karl_stickers + ryan_stickers + ben_stickers = 105) →
  karl_stickers = 25 := by
sorry

end NUMINAMATH_CALUDE_sticker_collection_l3898_389801


namespace NUMINAMATH_CALUDE_farm_egg_yolks_l3898_389875

/-- Represents the number of yolks in an egg carton -/
def yolks_in_carton (total_eggs : ℕ) (double_yolk_eggs : ℕ) : ℕ :=
  2 * double_yolk_eggs + (total_eggs - double_yolk_eggs)

/-- Theorem: A carton of 12 eggs with 5 double-yolk eggs has 17 yolks in total -/
theorem farm_egg_yolks : yolks_in_carton 12 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_farm_egg_yolks_l3898_389875


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3898_389822

theorem min_value_of_sum (a b : ℝ) : 
  a > 0 → b > 0 → (1 / a + 1 / b = 1) → (∀ x y : ℝ, a > 0 ∧ b > 0 ∧ x / a + y / b = 1 → a + 4 * b ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3898_389822


namespace NUMINAMATH_CALUDE_alf3_weight_calculation_l3898_389815

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_F : ℝ := 19.00

-- Define the number of moles
def num_moles : ℝ := 7

-- Define the molecular weight calculation function
def molecular_weight (al_weight f_weight : ℝ) : ℝ :=
  al_weight + 3 * f_weight

-- Define the total weight calculation function
def total_weight (mol_weight num_mol : ℝ) : ℝ :=
  mol_weight * num_mol

-- Theorem statement
theorem alf3_weight_calculation :
  total_weight (molecular_weight atomic_weight_Al atomic_weight_F) num_moles = 587.86 := by
  sorry


end NUMINAMATH_CALUDE_alf3_weight_calculation_l3898_389815


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l3898_389833

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^2 * x^(1/2))^(1/4) = x^(5/8) := by
sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l3898_389833


namespace NUMINAMATH_CALUDE_jake_balloons_l3898_389882

def total_balloons : ℕ := 3
def allan_balloons : ℕ := 2

theorem jake_balloons : total_balloons - allan_balloons = 1 := by
  sorry

end NUMINAMATH_CALUDE_jake_balloons_l3898_389882


namespace NUMINAMATH_CALUDE_roots_of_transformed_equation_l3898_389827

theorem roots_of_transformed_equation
  (p q : ℝ) (x₁ x₂ : ℝ)
  (h1 : x₁^2 + p*x₁ + q = 0)
  (h2 : x₂^2 + p*x₂ + q = 0)
  : (-x₁)^2 - p*(-x₁) + q = 0 ∧ (-x₂)^2 - p*(-x₂) + q = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_transformed_equation_l3898_389827


namespace NUMINAMATH_CALUDE_combined_storage_temperature_l3898_389810

-- Define the temperature ranges for each type of vegetable
def type_A_range : Set ℝ := {x | 3 ≤ x ∧ x ≤ 8}
def type_B_range : Set ℝ := {x | 5 ≤ x ∧ x ≤ 10}

-- Define the combined suitable temperature range
def combined_range : Set ℝ := type_A_range ∩ type_B_range

-- Theorem to prove
theorem combined_storage_temperature :
  combined_range = {x | 5 ≤ x ∧ x ≤ 8} := by
  sorry

end NUMINAMATH_CALUDE_combined_storage_temperature_l3898_389810


namespace NUMINAMATH_CALUDE_colored_isosceles_triangle_l3898_389877

/-- A regular polygon with 4n + 1 vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (4 * n + 1) → ℝ × ℝ

/-- A coloring of 2n vertices in a (4n + 1)-gon -/
def Coloring (n : ℕ) := Fin (4 * n + 1) → Bool

/-- Three vertices form an isosceles triangle -/
def IsIsosceles (p : RegularPolygon n) (v1 v2 v3 : Fin (4 * n + 1)) : Prop :=
  let d12 := dist (p.vertices v1) (p.vertices v2)
  let d23 := dist (p.vertices v2) (p.vertices v3)
  let d31 := dist (p.vertices v3) (p.vertices v1)
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- Main theorem -/
theorem colored_isosceles_triangle (n : ℕ) (h : n ≥ 3) (p : RegularPolygon n) (c : Coloring n) :
  ∃ v1 v2 v3 : Fin (4 * n + 1), c v1 ∧ c v2 ∧ c v3 ∧ IsIsosceles p v1 v2 v3 :=
sorry


end NUMINAMATH_CALUDE_colored_isosceles_triangle_l3898_389877


namespace NUMINAMATH_CALUDE_bench_cost_l3898_389842

theorem bench_cost (total_cost bench_cost table_cost : ℝ) : 
  total_cost = 450 →
  table_cost = 2 * bench_cost →
  total_cost = bench_cost + table_cost →
  bench_cost = 150 := by
sorry

end NUMINAMATH_CALUDE_bench_cost_l3898_389842


namespace NUMINAMATH_CALUDE_work_completion_time_l3898_389888

/-- The number of days it takes for worker A to complete the work alone -/
def days_A : ℝ := 6

/-- The number of days it takes for worker B to complete the work alone -/
def days_B : ℝ := 5

/-- The number of days it takes for workers A, B, and C to complete the work together -/
def days_ABC : ℝ := 2

/-- The number of days it takes for worker C to complete the work alone -/
def days_C : ℝ := 7.5

theorem work_completion_time :
  (1 / days_A) + (1 / days_B) + (1 / days_C) = (1 / days_ABC) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3898_389888


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3898_389871

theorem inequality_solution_set (x : ℝ) :
  (x^2 + x - 6 ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3898_389871


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3898_389841

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3898_389841


namespace NUMINAMATH_CALUDE_equation_positive_root_l3898_389856

theorem equation_positive_root (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x + 1) - m / (x + 1) = 1) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l3898_389856


namespace NUMINAMATH_CALUDE_park_walking_area_l3898_389885

/-- The area available for walking in a rectangular park with a centered circular fountain -/
theorem park_walking_area (park_length park_width fountain_radius : ℝ) 
  (h1 : park_length = 50)
  (h2 : park_width = 30)
  (h3 : fountain_radius = 5) : 
  park_length * park_width - Real.pi * fountain_radius^2 = 1500 - 25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_park_walking_area_l3898_389885


namespace NUMINAMATH_CALUDE_married_men_fraction_l3898_389844

theorem married_men_fraction (total_women : ℕ) (h_total_women_pos : total_women > 0) :
  let single_women : ℕ := (3 * total_women) / 7
  let married_women : ℕ := total_women - single_women
  let married_men : ℕ := married_women
  let total_people : ℕ := total_women + married_men
  (↑single_women : ℚ) / total_women = 3 / 7 →
  (↑married_men : ℚ) / total_people = 4 / 11 :=
by sorry

end NUMINAMATH_CALUDE_married_men_fraction_l3898_389844


namespace NUMINAMATH_CALUDE_equal_one_two_digit_prob_l3898_389816

-- Define a 12-sided die
def twelveSidedDie : Finset ℕ := Finset.range 12

-- Define one-digit numbers on the die
def oneDigitNumbers : Finset ℕ := Finset.filter (λ n => n < 10) twelveSidedDie

-- Define two-digit numbers on the die
def twoDigitNumbers : Finset ℕ := Finset.filter (λ n => n ≥ 10) twelveSidedDie

-- Define the probability of rolling a one-digit number
def probOneDigit : ℚ := (oneDigitNumbers.card : ℚ) / (twelveSidedDie.card : ℚ)

-- Define the probability of rolling a two-digit number
def probTwoDigit : ℚ := (twoDigitNumbers.card : ℚ) / (twelveSidedDie.card : ℚ)

-- Theorem stating the probability of rolling 4 dice and getting an equal number of one-digit and two-digit numbers
theorem equal_one_two_digit_prob : 
  (Finset.card oneDigitNumbers * Finset.card twoDigitNumbers * 6 : ℚ) / (twelveSidedDie.card ^ 4 : ℚ) = 27 / 128 :=
by sorry

end NUMINAMATH_CALUDE_equal_one_two_digit_prob_l3898_389816


namespace NUMINAMATH_CALUDE_percentage_of_women_in_study_group_l3898_389878

theorem percentage_of_women_in_study_group :
  let percentage_women_lawyers : ℝ := 0.4
  let prob_woman_lawyer : ℝ := 0.32
  let percentage_women : ℝ := prob_woman_lawyer / percentage_women_lawyers
  percentage_women = 0.8 := by sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_study_group_l3898_389878


namespace NUMINAMATH_CALUDE_rectangle_side_length_l3898_389897

/-- Given three rectangles with equal areas and integer sides, where one side is 29, prove that another side is 870 -/
theorem rectangle_side_length (a b k l : ℕ) : 
  let S := 29 * (a + b)
  a * k = S ∧ 
  b * l = S ∧ 
  k * l = 29 * (k + l) →
  k = 870 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l3898_389897


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3898_389847

def sequence_u (n : ℕ) : ℝ :=
  sorry

theorem sum_of_coefficients :
  (∃ (a b c : ℝ), ∀ (n : ℕ), sequence_u n = a * n^2 + b * n + c) →
  (sequence_u 1 = 7) →
  (∀ (n : ℕ), sequence_u (n + 1) - sequence_u n = 5 + 3 * (n - 1)) →
  (∃ (a b c : ℝ), 
    (∀ (n : ℕ), sequence_u n = a * n^2 + b * n + c) ∧
    (a + b + c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3898_389847


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_for_inequality_l3898_389820

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part I
theorem solution_set_when_a_eq_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part II
theorem range_of_a_for_inequality :
  (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_for_inequality_l3898_389820


namespace NUMINAMATH_CALUDE_h_domain_l3898_389860

def f_domain : Set ℝ := Set.Icc (-3) 6

def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-3 * x)

theorem h_domain (f : ℝ → ℝ) : 
  {x : ℝ | ∃ y ∈ f_domain, y = -3 * x} = Set.Icc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_h_domain_l3898_389860


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l3898_389838

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) :
  x^2 + y^2 ≥ 229 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l3898_389838


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_98_l3898_389867

theorem largest_four_digit_divisible_by_98 : 
  ∀ n : ℕ, n ≤ 9998 ∧ n ≥ 1000 ∧ n % 98 = 0 → n ≤ 9998 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_98_l3898_389867


namespace NUMINAMATH_CALUDE_first_player_can_force_draw_l3898_389824

/-- Represents the state of a square on the game board -/
inductive Square
| Empty : Square
| A : Square
| B : Square

/-- Represents the game board as a list of squares -/
def Board := List Square

/-- Checks if a given board contains the winning sequence ABA -/
def hasWinningSequence (board : Board) : Bool :=
  sorry

/-- Represents a player's move -/
structure Move where
  position : Nat
  letter : Square

/-- Applies a move to the board -/
def applyMove (board : Board) (move : Move) : Board :=
  sorry

/-- Checks if a move is valid on the given board -/
def isValidMove (board : Board) (move : Move) : Bool :=
  sorry

/-- Represents the game state -/
structure GameState where
  board : Board
  currentPlayer : Bool  -- True for first player, False for second player

/-- The main theorem stating that the first player can force a draw -/
theorem first_player_can_force_draw :
  ∃ (strategy : GameState → Move),
    ∀ (game : GameState),
      game.board.length = 14 →
      game.currentPlayer = true →
      ¬(hasWinningSequence (applyMove game.board (strategy game))) :=
sorry

end NUMINAMATH_CALUDE_first_player_can_force_draw_l3898_389824


namespace NUMINAMATH_CALUDE_parkway_elementary_girls_not_playing_soccer_l3898_389814

theorem parkway_elementary_girls_not_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (playing_soccer : ℕ)
  (boys_playing_soccer_percentage : ℚ)
  (h1 : total_students = 470)
  (h2 : boys = 300)
  (h3 : playing_soccer = 250)
  (h4 : boys_playing_soccer_percentage = 86 / 100)
  : ℕ := by
  sorry

#check parkway_elementary_girls_not_playing_soccer

end NUMINAMATH_CALUDE_parkway_elementary_girls_not_playing_soccer_l3898_389814


namespace NUMINAMATH_CALUDE_statement_independent_of_parallel_postulate_l3898_389886

-- Define a geometry
class Geometry where
  -- Define the concept of a line
  Line : Type
  -- Define the concept of a point
  Point : Type
  -- Define the concept of parallelism
  parallel : Line → Line → Prop
  -- Define the concept of intersection
  intersects : Line → Line → Prop

-- Define the statement to be proven
def statement (G : Geometry) : Prop :=
  ∀ (l₁ l₂ l₃ : G.Line),
    G.parallel l₁ l₂ → G.intersects l₃ l₁ → G.intersects l₃ l₂

-- Define the parallel postulate
def parallel_postulate (G : Geometry) : Prop :=
  ∀ (p : G.Point) (l : G.Line),
    ∃! (m : G.Line), G.parallel l m

-- Theorem: The statement is independent of the parallel postulate
theorem statement_independent_of_parallel_postulate :
  ∀ (G : Geometry),
    (statement G ↔ statement G) ∧ 
    (¬(parallel_postulate G → statement G)) ∧
    (¬(statement G → parallel_postulate G)) :=
sorry

end NUMINAMATH_CALUDE_statement_independent_of_parallel_postulate_l3898_389886


namespace NUMINAMATH_CALUDE_roots_sum_powers_l3898_389829

theorem roots_sum_powers (a b : ℝ) : 
  a + b = 6 → ab = 8 → a^2 + a^5 * b^3 + a^3 * b^5 + b^2 = 10260 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l3898_389829


namespace NUMINAMATH_CALUDE_limit_of_a_is_2_l3898_389852

def a (n : ℕ) : ℚ := (4 * n - 3) / (2 * n + 1)

theorem limit_of_a_is_2 : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_of_a_is_2_l3898_389852


namespace NUMINAMATH_CALUDE_expand_expression_l3898_389806

theorem expand_expression (x : ℝ) : (16*x + 18 - 4*x^2) * 3*x = -12*x^3 + 48*x^2 + 54*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3898_389806


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l3898_389825

theorem simplify_sqrt_difference (x : ℝ) (h : x ≤ 2) : 
  Real.sqrt (x^2 - 4*x + 4) - Real.sqrt (x^2 - 6*x + 9) = -1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l3898_389825


namespace NUMINAMATH_CALUDE_return_trip_time_l3898_389859

/-- Given a route with uphill and downhill sections, prove the return trip time -/
theorem return_trip_time (total_distance : ℝ) (uphill_speed downhill_speed : ℝ) 
  (time_ab : ℝ) (h1 : total_distance = 21) (h2 : uphill_speed = 4) 
  (h3 : downhill_speed = 6) (h4 : time_ab = 4.25) : ∃ (time_ba : ℝ), time_ba = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_return_trip_time_l3898_389859


namespace NUMINAMATH_CALUDE_parameterization_validity_l3898_389854

def line (x : ℝ) : ℝ := -3 * x + 4

def is_valid_parameterization (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line (p.1 + t * v.1) = p.2 + t * v.2

theorem parameterization_validity :
  (is_valid_parameterization (0, 4) (1, -3)) ∧
  (is_valid_parameterization (-4, 16) (1/3, -1)) ∧
  ¬(is_valid_parameterization (1/3, 0) (3, -1)) ∧
  ¬(is_valid_parameterization (2, -2) (4, -12)) ∧
  ¬(is_valid_parameterization (1, 1) (0.5, -1.5)) :=
sorry

end NUMINAMATH_CALUDE_parameterization_validity_l3898_389854


namespace NUMINAMATH_CALUDE_alexandra_magazines_l3898_389808

theorem alexandra_magazines : 
  let friday_magazines : ℕ := 8
  let saturday_magazines : ℕ := 12
  let sunday_magazines : ℕ := 4 * friday_magazines
  let chewed_magazines : ℕ := 4
  friday_magazines + saturday_magazines + sunday_magazines - chewed_magazines = 48
  := by sorry

end NUMINAMATH_CALUDE_alexandra_magazines_l3898_389808


namespace NUMINAMATH_CALUDE_student_scores_l3898_389830

theorem student_scores (math physics chemistry : ℕ) : 
  math + physics = 32 →
  (math + chemistry) / 2 = 26 →
  ∃ x : ℕ, chemistry = physics + x ∧ x = 20 := by
sorry

end NUMINAMATH_CALUDE_student_scores_l3898_389830


namespace NUMINAMATH_CALUDE_chord_segment_lengths_l3898_389850

theorem chord_segment_lengths (R : ℝ) (OM : ℝ) (AB : ℝ) (AM MB : ℝ) : 
  R = 15 →
  OM = 13 →
  AB = 18 →
  AM + MB = AB →
  OM^2 = R^2 - (AB/2)^2 + ((AM - MB)/2)^2 →
  AM = 14 ∧ MB = 4 :=
by sorry

end NUMINAMATH_CALUDE_chord_segment_lengths_l3898_389850


namespace NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l3898_389853

/-- The number of rectangles that can be formed on a 4x4 grid --/
def num_rectangles_4x4 : ℕ := 36

/-- The size of the grid --/
def grid_size : ℕ := 4

/-- Theorem: The number of rectangles on a 4x4 grid is 36 --/
theorem rectangles_on_4x4_grid :
  num_rectangles_4x4 = (grid_size.choose 2) * (grid_size.choose 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l3898_389853


namespace NUMINAMATH_CALUDE_factorial_ratio_problem_l3898_389837

theorem factorial_ratio_problem (m n : ℕ) : 
  m > 1 → n > 1 → (Nat.factorial (n + m)) / (Nat.factorial n) = 17297280 → 
  n / m = 1 ∨ n / m = 31 / 2 := by
sorry

end NUMINAMATH_CALUDE_factorial_ratio_problem_l3898_389837


namespace NUMINAMATH_CALUDE_tangent_perpendicular_condition_l3898_389804

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0
def circle2 (a x y : ℝ) : Prop := x^2 + y^2 + 2*(a-1)*x + 2*y + a^2 = 0

-- Define the condition for perpendicular tangent lines
def perpendicular_tangents (a m n : ℝ) : Prop :=
  (n + 2) / m * (n + 1) / (m - (1 - a)) = -1

-- Define the theorem
theorem tangent_perpendicular_condition :
  ∃ (a : ℝ), a = -2 ∧
  ∀ (m n : ℝ), circle1 m n → circle2 a m n → perpendicular_tangents a m n :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_condition_l3898_389804


namespace NUMINAMATH_CALUDE_x_value_proof_l3898_389826

theorem x_value_proof (x : ℝ) 
  (h : (x^2 - x - 6) / (x + 1) = (x^2 - 2*x - 3)*Complex.I) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l3898_389826


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3898_389896

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 3) ↔ x ≥ -3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3898_389896


namespace NUMINAMATH_CALUDE_complex_magnitude_l3898_389855

/-- Given a real number a, if (a^2 * i) / (1 + i) is imaginary, then |a + i| = √5 -/
theorem complex_magnitude (a : ℝ) : 
  (((a^2 * Complex.I) / (1 + Complex.I)).im ≠ 0 ∧ ((a^2 * Complex.I) / (1 + Complex.I)).re = 0) → 
  Complex.abs (a + Complex.I) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3898_389855


namespace NUMINAMATH_CALUDE_solution_set_l3898_389818

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := {x | x > 0}

-- State the theorem
theorem solution_set (hf : StrictMono f) (hd : ∀ x ∈ domain, f x ≠ 0) :
  {x : ℝ | f x > f (8 * (x - 2))} = {x : ℝ | 2 < x ∧ x < 16/7} := by sorry

end NUMINAMATH_CALUDE_solution_set_l3898_389818


namespace NUMINAMATH_CALUDE_candy_division_l3898_389889

theorem candy_division (total_candy : ℕ) (num_students : ℕ) (candy_per_student : ℕ) 
  (h1 : total_candy = 344) 
  (h2 : num_students = 43) 
  (h3 : candy_per_student = total_candy / num_students) :
  candy_per_student = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_l3898_389889


namespace NUMINAMATH_CALUDE_two_thirds_plus_six_l3898_389848

theorem two_thirds_plus_six (x : ℝ) : x = 6 → (2 / 3 * x) + 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_plus_six_l3898_389848


namespace NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l3898_389817

/-- Calculate the cost per quart of ratatouille --/
theorem ratatouille_cost_per_quart :
  let eggplant_weight : ℝ := 5.5
  let eggplant_price : ℝ := 2.20
  let zucchini_weight : ℝ := 3.8
  let zucchini_price : ℝ := 1.85
  let tomato_weight : ℝ := 4.6
  let tomato_price : ℝ := 3.75
  let onion_weight : ℝ := 2.7
  let onion_price : ℝ := 1.10
  let basil_weight : ℝ := 1.0
  let basil_price : ℝ := 2.70 * 4  -- Price per pound (4 quarters)
  let pepper_weight : ℝ := 0.75
  let pepper_price : ℝ := 3.15
  let total_yield : ℝ := 4.5

  let total_cost : ℝ := 
    eggplant_weight * eggplant_price +
    zucchini_weight * zucchini_price +
    tomato_weight * tomato_price +
    onion_weight * onion_price +
    basil_weight * basil_price +
    pepper_weight * pepper_price

  let cost_per_quart : ℝ := total_cost / total_yield

  cost_per_quart = 11.67 := by sorry

end NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l3898_389817


namespace NUMINAMATH_CALUDE_total_weight_sold_l3898_389831

/-- Calculates the total weight of bags sold in a day given the sales data and weight per bag -/
theorem total_weight_sold (morning_potatoes afternoon_potatoes morning_onions afternoon_onions
  morning_carrots afternoon_carrots potato_weight onion_weight carrot_weight : ℕ) :
  morning_potatoes = 29 →
  afternoon_potatoes = 17 →
  morning_onions = 15 →
  afternoon_onions = 22 →
  morning_carrots = 12 →
  afternoon_carrots = 9 →
  potato_weight = 7 →
  onion_weight = 5 →
  carrot_weight = 4 →
  (morning_potatoes + afternoon_potatoes) * potato_weight +
  (morning_onions + afternoon_onions) * onion_weight +
  (morning_carrots + afternoon_carrots) * carrot_weight = 591 :=
by
  sorry

end NUMINAMATH_CALUDE_total_weight_sold_l3898_389831


namespace NUMINAMATH_CALUDE_three_circles_sum_l3898_389884

theorem three_circles_sum (triangle circle : ℚ) 
  (eq1 : 5 * triangle + 2 * circle = 27)
  (eq2 : 2 * triangle + 5 * circle = 29) :
  3 * circle = 13 := by
sorry

end NUMINAMATH_CALUDE_three_circles_sum_l3898_389884


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l3898_389895

/-- The coefficient of the linear term in the quadratic equation x^2 - x = 0 is -1 -/
theorem linear_coefficient_of_quadratic (x : ℝ) : 
  (fun x => x^2 - x) = (fun x => x^2 - 1*x) :=
by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l3898_389895


namespace NUMINAMATH_CALUDE_wilsons_theorem_l3898_389823

theorem wilsons_theorem (p : ℕ) (hp : Prime p) :
  (((Nat.factorial (p - 1)) : ℤ) % p = -1) ∧
  (p^2 ∣ ((Nat.factorial (p - 1)) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l3898_389823


namespace NUMINAMATH_CALUDE_game_points_total_l3898_389881

theorem game_points_total (eric_points mark_points samanta_points : ℕ) : 
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points = mark_points + 8 →
  samanta_points + mark_points + eric_points = 32 := by
sorry

end NUMINAMATH_CALUDE_game_points_total_l3898_389881


namespace NUMINAMATH_CALUDE_valid_triangulations_are_4_7_19_l3898_389890

/-- Represents a triangulation of a triangle. -/
structure Triangulation where
  num_triangles : ℕ
  sides_per_vertex : ℕ

/-- Predicate to check if a triangulation is valid according to the problem conditions. -/
def is_valid_triangulation (t : Triangulation) : Prop :=
  t.num_triangles ≤ 19 ∧
  t.num_triangles > 0 ∧
  t.sides_per_vertex > 2

/-- The set of all valid triangulations. -/
def valid_triangulations : Set Triangulation :=
  {t : Triangulation | is_valid_triangulation t}

/-- Theorem stating that the only valid triangulations have 4, 7, or 19 triangles. -/
theorem valid_triangulations_are_4_7_19 :
  ∀ t ∈ valid_triangulations, t.num_triangles = 4 ∨ t.num_triangles = 7 ∨ t.num_triangles = 19 := by
  sorry

end NUMINAMATH_CALUDE_valid_triangulations_are_4_7_19_l3898_389890


namespace NUMINAMATH_CALUDE_fifth_power_sum_l3898_389846

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a*x + b*y = 2)
  (h2 : a*x^2 + b*y^2 = 5)
  (h3 : a*x^3 + b*y^3 = 15)
  (h4 : a*x^4 + b*y^4 = 35) :
  a*x^5 + b*y^5 = 10 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l3898_389846


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l3898_389872

theorem second_term_of_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) : 
  r = (1 : ℝ) / 4 →
  S = 16 →
  S = a / (1 - r) →
  a * r = 3 :=
by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l3898_389872


namespace NUMINAMATH_CALUDE_total_crayons_calculation_l3898_389874

/-- Given a group of children where each child has a certain number of crayons,
    calculate the total number of crayons. -/
def total_crayons (crayons_per_child : ℕ) (num_children : ℕ) : ℕ :=
  crayons_per_child * num_children

/-- Theorem stating that the total number of crayons is 648 when each child has 18 crayons
    and there are 36 children. -/
theorem total_crayons_calculation :
  total_crayons 18 36 = 648 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_calculation_l3898_389874


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l3898_389803

theorem similar_triangles_leg_sum (a b c d : ℕ) : 
  a * b = 18 →  -- area of smaller triangle is 9
  a^2 + b^2 = 25 →  -- hypotenuse of smaller triangle is 5
  a ≠ 3 ∨ b ≠ 4 →  -- not a 3-4-5 triangle
  c * d = 450 →  -- area of larger triangle is 225
  (c : ℝ) / a = (d : ℝ) / b →  -- triangles are similar
  (c + d : ℝ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l3898_389803


namespace NUMINAMATH_CALUDE_marias_piggy_bank_l3898_389845

/-- Represents the number of coins of each type -/
structure CoinCount where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- Calculates the total value of coins in dollars -/
def totalValue (coins : CoinCount) : ℚ :=
  0.10 * coins.dimes + 0.25 * coins.quarters + 0.05 * coins.nickels

/-- The problem statement -/
theorem marias_piggy_bank (initialCoins : CoinCount) :
  initialCoins.dimes = 4 →
  initialCoins.quarters = 4 →
  totalValue { dimes := initialCoins.dimes,
               quarters := initialCoins.quarters + 5,
               nickels := initialCoins.nickels } = 3 →
  initialCoins.nickels = 7 := by
  sorry

end NUMINAMATH_CALUDE_marias_piggy_bank_l3898_389845


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l3898_389862

theorem trader_gain_percentage (cost : ℝ) (h : cost > 0) : 
  (22 * cost) / (88 * cost) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l3898_389862


namespace NUMINAMATH_CALUDE_total_fish_count_l3898_389892

-- Define the number of fish for each person
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish
def jenny_fish : ℕ := bobby_fish - 4

-- Theorem to prove
theorem total_fish_count : billy_fish + tony_fish + sarah_fish + bobby_fish + jenny_fish = 211 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l3898_389892


namespace NUMINAMATH_CALUDE_marbles_left_l3898_389839

def initial_marbles : ℕ := 38
def lost_marbles : ℕ := 15

theorem marbles_left : initial_marbles - lost_marbles = 23 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_l3898_389839


namespace NUMINAMATH_CALUDE_enlarged_poster_height_l3898_389834

/-- Calculates the new height of a proportionally enlarged poster -/
def new_poster_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Theorem: The new height of the enlarged poster is 10 inches -/
theorem enlarged_poster_height :
  new_poster_height 3 2 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_enlarged_poster_height_l3898_389834


namespace NUMINAMATH_CALUDE_expansion_properties_l3898_389805

/-- Given the expansion of (2x + 3/∛x)^n, where the ratio of the binomial coefficient
    of the third term to that of the second term is 5:2, prove the following: -/
theorem expansion_properties (n : ℕ) (x : ℝ) :
  (Nat.choose n 2 : ℚ) / (Nat.choose n 1 : ℚ) = 5 / 2 →
  (n = 6 ∧
   (∃ (r : ℕ), Nat.choose 6 r * 2^(6-r) * 3^r * x^(6 - 4/3*r) = 4320 * x^2) ∧
   (∃ (k : ℕ), Nat.choose 6 k * 2^(6-k) * 3^k * x^((2:ℝ)/3) = 4860 * x^((2:ℝ)/3) ∧
               ∀ (j : ℕ), j ≠ k → Nat.choose 6 j * 2^(6-j) * 3^j ≤ Nat.choose 6 k * 2^(6-k) * 3^k)) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l3898_389805


namespace NUMINAMATH_CALUDE_seven_valid_triples_l3898_389893

/-- The number of valid triples (a, b, c) for the prism cutting problem -/
def count_valid_triples : ℕ :=
  let b := 2023
  (Finset.filter (fun p : ℕ × ℕ =>
    let a := p.1
    let c := p.2
    a ≤ b ∧ b ≤ c ∧ a * c = b * b
  ) (Finset.product (Finset.range (b + 1)) (Finset.range (b * b + 1)))).card

/-- The main theorem stating there are exactly 7 valid triples -/
theorem seven_valid_triples : count_valid_triples = 7 := by
  sorry


end NUMINAMATH_CALUDE_seven_valid_triples_l3898_389893


namespace NUMINAMATH_CALUDE_polynomial_characterization_l3898_389898

variable (f g : ℝ → ℝ)

def IsConcave (f : ℝ → ℝ) : Prop :=
  ∀ x y t : ℝ, 0 ≤ t ∧ t ≤ 1 → f (t * x + (1 - t) * y) ≥ t * f x + (1 - t) * f y

theorem polynomial_characterization
  (hf_concave : IsConcave f)
  (hg_continuous : Continuous g)
  (h_equality : ∀ x y : ℝ, f (x + y) + f (x - y) - 2 * f x = g x * y^2) :
  ∃ A B C : ℝ, ∀ x : ℝ, f x = A * x + B * x^2 + C :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l3898_389898


namespace NUMINAMATH_CALUDE_complex_power_sum_l3898_389807

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^500 + 1/(z^500) = 2 * Real.cos (100 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3898_389807


namespace NUMINAMATH_CALUDE_angle_negative_2015_in_second_quadrant_l3898_389836

/-- The quadrant of an angle in degrees -/
inductive Quadrant
| first
| second
| third
| fourth

/-- Determine the quadrant of an angle in degrees -/
def angleQuadrant (angle : ℤ) : Quadrant :=
  let normalizedAngle := angle % 360
  if 0 ≤ normalizedAngle && normalizedAngle < 90 then Quadrant.first
  else if 90 ≤ normalizedAngle && normalizedAngle < 180 then Quadrant.second
  else if 180 ≤ normalizedAngle && normalizedAngle < 270 then Quadrant.third
  else Quadrant.fourth

theorem angle_negative_2015_in_second_quadrant :
  angleQuadrant (-2015) = Quadrant.second := by
  sorry

end NUMINAMATH_CALUDE_angle_negative_2015_in_second_quadrant_l3898_389836


namespace NUMINAMATH_CALUDE_train_journey_time_l3898_389828

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (4 / 5 * usual_speed) * (usual_time + 1 / 2) = usual_speed * usual_time → 
  usual_time = 2 := by
sorry

end NUMINAMATH_CALUDE_train_journey_time_l3898_389828


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_3_and_4_l3898_389894

theorem largest_five_digit_divisible_by_3_and_4 : ∃ n : ℕ, 
  (n ≤ 99999) ∧ 
  (n ≥ 10000) ∧ 
  (n % 3 = 0) ∧ 
  (n % 4 = 0) ∧ 
  (∀ m : ℕ, m ≤ 99999 ∧ m ≥ 10000 ∧ m % 3 = 0 ∧ m % 4 = 0 → m ≤ n) ∧
  n = 99996 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_3_and_4_l3898_389894


namespace NUMINAMATH_CALUDE_remainder_3_pow_19_mod_10_l3898_389858

theorem remainder_3_pow_19_mod_10 : (3^19) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_19_mod_10_l3898_389858


namespace NUMINAMATH_CALUDE_cubic_sum_l3898_389876

theorem cubic_sum (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a * b + a * c + b * c = 7) 
  (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 29 := by
sorry


end NUMINAMATH_CALUDE_cubic_sum_l3898_389876


namespace NUMINAMATH_CALUDE_fish_problem_l3898_389865

theorem fish_problem (ken_fish : ℕ) (kendra_fish : ℕ) :
  ken_fish = 2 * kendra_fish - 3 →
  ken_fish + kendra_fish = 87 →
  kendra_fish = 30 := by
sorry

end NUMINAMATH_CALUDE_fish_problem_l3898_389865


namespace NUMINAMATH_CALUDE_sum_of_two_equals_third_l3898_389802

theorem sum_of_two_equals_third (x y z : ℤ) 
  (h1 : x + y = z) (h2 : y + z = x) (h3 : z + x = y) : 
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_equals_third_l3898_389802


namespace NUMINAMATH_CALUDE_seeds_per_flowerbed_l3898_389809

theorem seeds_per_flowerbed 
  (total_seeds : ℕ) 
  (num_flowerbeds : ℕ) 
  (h1 : total_seeds = 45) 
  (h2 : num_flowerbeds = 9) 
  (h3 : total_seeds % num_flowerbeds = 0) :
  total_seeds / num_flowerbeds = 5 := by
sorry

end NUMINAMATH_CALUDE_seeds_per_flowerbed_l3898_389809


namespace NUMINAMATH_CALUDE_alien_number_conversion_l3898_389851

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The base-6 representation of the number --/
def alienNumber : List Nat := [4, 5, 1, 2]

theorem alien_number_conversion :
  base6ToBase10 alienNumber = 502 := by
  sorry

#eval base6ToBase10 alienNumber

end NUMINAMATH_CALUDE_alien_number_conversion_l3898_389851


namespace NUMINAMATH_CALUDE_sally_has_more_cards_l3898_389880

theorem sally_has_more_cards (sally_initial : ℕ) (sally_bought : ℕ) (dan_cards : ℕ)
  (h1 : sally_initial = 27)
  (h2 : sally_bought = 20)
  (h3 : dan_cards = 41) :
  sally_initial + sally_bought - dan_cards = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_has_more_cards_l3898_389880


namespace NUMINAMATH_CALUDE_broken_flagpole_theorem_l3898_389800

/-- Represents a broken flagpole -/
structure BrokenFlagpole where
  initial_height : ℝ
  tip_height : ℝ
  break_point : ℝ

/-- The condition for a valid broken flagpole configuration -/
def is_valid_broken_flagpole (f : BrokenFlagpole) : Prop :=
  f.initial_height > 0 ∧
  f.tip_height > 0 ∧
  f.tip_height < f.initial_height ∧
  f.break_point > 0 ∧
  f.break_point < f.initial_height ∧
  (f.initial_height - f.break_point) * 2 = f.initial_height - f.tip_height

theorem broken_flagpole_theorem (f : BrokenFlagpole)
  (h_valid : is_valid_broken_flagpole f)
  (h_initial : f.initial_height = 12)
  (h_tip : f.tip_height = 2) :
  f.break_point = 7 := by
sorry

end NUMINAMATH_CALUDE_broken_flagpole_theorem_l3898_389800


namespace NUMINAMATH_CALUDE_money_problem_l3898_389899

theorem money_problem (a b : ℝ) 
  (h1 : 4 * a - b > 32)
  (h2 : 2 * a + b = 26) : 
  a > 9.67 ∧ b < 6.66 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3898_389899


namespace NUMINAMATH_CALUDE_line_y_intercept_l3898_389868

/-- A straight line in the xy-plane with slope 2 passing through (269, 540) has y-intercept 2 -/
theorem line_y_intercept (m slope : ℝ) (x₀ y₀ : ℝ) :
  slope = 2 →
  x₀ = 269 →
  y₀ = 540 →
  y₀ = slope * x₀ + m →
  m = 2 := by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l3898_389868
