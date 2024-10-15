import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l316_31652

theorem arithmetic_progression_common_difference 
  (a₁ : ℝ) (a₂₁ : ℝ) (d : ℝ) :
  a₁ = 3 → a₂₁ = 103 → a₂₁ = a₁ + 20 * d → d = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l316_31652


namespace NUMINAMATH_CALUDE_peters_remaining_money_is_304_50_l316_31669

/-- Represents Peter's shopping trips and calculates his remaining money. -/
def petersRemainingMoney : ℝ :=
  let initialAmount : ℝ := 500
  let firstTripPurchases : List (ℝ × ℝ) := [
    (6, 2),    -- potatoes
    (9, 3),    -- tomatoes
    (5, 4),    -- cucumbers
    (3, 5),    -- bananas
    (2, 3.5),  -- apples
    (7, 4.25), -- oranges
    (4, 6),    -- grapes
    (8, 5.5)   -- strawberries
  ]
  let secondTripPurchases : List (ℝ × ℝ) := [
    (2, 1.5),  -- additional potatoes
    (5, 2.75)  -- additional tomatoes
  ]
  let totalCost := (firstTripPurchases ++ secondTripPurchases).foldl
    (fun acc (quantity, price) => acc + quantity * price) 0
  initialAmount - totalCost

/-- Theorem stating that Peter's remaining money is $304.50 -/
theorem peters_remaining_money_is_304_50 :
  petersRemainingMoney = 304.50 := by
  sorry

#eval petersRemainingMoney

end NUMINAMATH_CALUDE_peters_remaining_money_is_304_50_l316_31669


namespace NUMINAMATH_CALUDE_work_earnings_problem_l316_31647

theorem work_earnings_problem (t : ℝ) : 
  (t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 3) + 3 → t = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_problem_l316_31647


namespace NUMINAMATH_CALUDE_pencils_per_row_l316_31677

/-- Given a total of 720 pencils arranged in 30 rows, prove that there are 24 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (total_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 720) 
  (h2 : total_rows = 30) 
  (h3 : total_pencils = total_rows * pencils_per_row) : 
  pencils_per_row = 24 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l316_31677


namespace NUMINAMATH_CALUDE_points_earned_is_thirteen_l316_31619

/-- VideoGame represents the state of the game --/
structure VideoGame where
  totalEnemies : Nat
  redEnemies : Nat
  blueEnemies : Nat
  defeatedEnemies : Nat
  hits : Nat
  pointsPerEnemy : Nat
  bonusPoints : Nat
  pointsLostPerHit : Nat

/-- Calculate the total points earned in the game --/
def calculatePoints (game : VideoGame) : Int :=
  let basePoints := game.defeatedEnemies * game.pointsPerEnemy
  let bonusEarned := if (game.redEnemies - 1 > 0) && (game.blueEnemies - 1 > 0) then game.bonusPoints else 0
  let totalEarned := basePoints + bonusEarned
  let pointsLost := game.hits * game.pointsLostPerHit
  totalEarned - pointsLost

/-- Theorem stating that given the game conditions, the total points earned is 13 --/
theorem points_earned_is_thirteen :
  ∀ (game : VideoGame),
    game.totalEnemies = 6 →
    game.redEnemies = 3 →
    game.blueEnemies = 3 →
    game.defeatedEnemies = 4 →
    game.hits = 2 →
    game.pointsPerEnemy = 3 →
    game.bonusPoints = 5 →
    game.pointsLostPerHit = 2 →
    calculatePoints game = 13 := by
  sorry

end NUMINAMATH_CALUDE_points_earned_is_thirteen_l316_31619


namespace NUMINAMATH_CALUDE_binomial_expansion_alternating_sum_l316_31642

theorem binomial_expansion_alternating_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ - a₀ + a₃ - a₂ + a₅ - a₄ = -1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_alternating_sum_l316_31642


namespace NUMINAMATH_CALUDE_chocolate_boxes_pieces_per_box_l316_31636

theorem chocolate_boxes_pieces_per_box 
  (initial_boxes : ℕ) 
  (given_away_boxes : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : initial_boxes = 14)
  (h2 : given_away_boxes = 5)
  (h3 : remaining_pieces = 54)
  (h4 : initial_boxes > given_away_boxes) :
  (remaining_pieces / (initial_boxes - given_away_boxes) = 6) :=
by sorry

end NUMINAMATH_CALUDE_chocolate_boxes_pieces_per_box_l316_31636


namespace NUMINAMATH_CALUDE_table_tennis_choices_l316_31685

theorem table_tennis_choices (rackets balls nets : ℕ) 
  (h_rackets : rackets = 7)
  (h_balls : balls = 7)
  (h_nets : nets = 3) :
  rackets * balls * nets = 147 := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_choices_l316_31685


namespace NUMINAMATH_CALUDE_conditional_probability_rain_given_wind_l316_31615

/-- Given probabilities of events A and B, and their intersection, prove the conditional probability P(A|B) -/
theorem conditional_probability_rain_given_wind 
  (P_A : ℚ) (P_B : ℚ) (P_A_and_B : ℚ)
  (h1 : P_A = 4/15)
  (h2 : P_B = 2/15)
  (h3 : P_A_and_B = 1/10)
  : P_A_and_B / P_B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_given_wind_l316_31615


namespace NUMINAMATH_CALUDE_power_of_product_l316_31649

theorem power_of_product (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l316_31649


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_new_mixture_l316_31660

def original_volume : ℝ := 24
def water_volume : ℝ := 16

def alcohol_A_fraction : ℝ := 0.3
def alcohol_B_fraction : ℝ := 0.4
def alcohol_C_fraction : ℝ := 0.3

def alcohol_A_purity : ℝ := 0.8
def alcohol_B_purity : ℝ := 0.9
def alcohol_C_purity : ℝ := 0.95

def new_mixture_volume : ℝ := original_volume + water_volume

def total_pure_alcohol : ℝ :=
  original_volume * (
    alcohol_A_fraction * alcohol_A_purity +
    alcohol_B_fraction * alcohol_B_purity +
    alcohol_C_fraction * alcohol_C_purity
  )

theorem alcohol_percentage_in_new_mixture :
  (total_pure_alcohol / new_mixture_volume) * 100 = 53.1 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_new_mixture_l316_31660


namespace NUMINAMATH_CALUDE_final_answer_calculation_l316_31645

theorem final_answer_calculation (chosen_number : ℤ) (h : chosen_number = 848) : 
  (chosen_number / 8 : ℚ) - 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_final_answer_calculation_l316_31645


namespace NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_is_11920_l316_31627

/-- The sum of all integers between 100 and 500 which end in 3 -/
def sum_of_integers_ending_in_3 : ℕ :=
  let first_term := 103
  let last_term := 493
  let num_terms := (last_term - first_term) / 10 + 1
  num_terms * (first_term + last_term) / 2

theorem sum_of_integers_ending_in_3_is_11920 :
  sum_of_integers_ending_in_3 = 11920 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_is_11920_l316_31627


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l316_31693

theorem division_multiplication_equality : (120 / 4 / 2 * 3 : ℚ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l316_31693


namespace NUMINAMATH_CALUDE_particle_probability_l316_31673

def probability (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * probability (x-1) y + (1/3) * probability x (y-1) + (1/3) * probability (x-1) (y-1)

theorem particle_probability :
  probability 3 3 = 7/81 :=
sorry

end NUMINAMATH_CALUDE_particle_probability_l316_31673


namespace NUMINAMATH_CALUDE_largest_sum_l316_31687

/-- A digit is a natural number between 0 and 9 inclusive. -/
def Digit : Type := {n : ℕ // n ≤ 9}

/-- The sum function for the given problem. -/
def sum (A B C : Digit) : ℕ := 111 * A.val + 10 * C.val + 2 * B.val

/-- The theorem stating that 976 is the largest possible 3-digit sum. -/
theorem largest_sum :
  ∀ A B C : Digit,
    A ≠ B → A ≠ C → B ≠ C →
    sum A B C ≤ 976 ∧
    (∃ A' B' C' : Digit, A' ≠ B' ∧ A' ≠ C' ∧ B' ≠ C' ∧ sum A' B' C' = 976) :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_l316_31687


namespace NUMINAMATH_CALUDE_ceiling_negative_fraction_cubed_l316_31601

theorem ceiling_negative_fraction_cubed : ⌈(-7/4)^3⌉ = -5 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_fraction_cubed_l316_31601


namespace NUMINAMATH_CALUDE_inequality_solution_set_l316_31628

-- Define a monotonically increasing function on [0, +∞)
def monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Define the set of x that satisfies the inequality
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (2 * x - 1) < f (1 / 3)}

-- Theorem statement
theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_monotone : monotone_increasing_on_nonneg f) :
  solution_set f = Set.Ici (1 / 2) ∩ Set.Iio (2 / 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l316_31628


namespace NUMINAMATH_CALUDE_abs_difference_inequality_l316_31639

theorem abs_difference_inequality (x : ℝ) : |x + 1| - |x - 5| < 4 ↔ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_difference_inequality_l316_31639


namespace NUMINAMATH_CALUDE_find_coefficient_a_l316_31613

theorem find_coefficient_a (f' : ℝ → ℝ) (a : ℝ) :
  (∀ x, f' x = 2 * x^3 + a * x^2 + x) →
  f' 1 = 9 →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_find_coefficient_a_l316_31613


namespace NUMINAMATH_CALUDE_line_equation_transformation_l316_31624

/-- Given a line L with equation y = (2/3)x + 4, prove that a line M with twice the slope
    and half the y-intercept of L has the equation y = (4/3)x + 2 -/
theorem line_equation_transformation (x y : ℝ) :
  let L : ℝ → ℝ := λ x => (2/3) * x + 4
  let M : ℝ → ℝ := λ x => (4/3) * x + 2
  (∀ x, M x = 2 * ((2/3) * x) + (1/2) * 4) → (∀ x, M x = (4/3) * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_transformation_l316_31624


namespace NUMINAMATH_CALUDE_angle_A_magnitude_max_area_l316_31622

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.cos t.C = (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A

-- Theorem for part I
theorem angle_A_magnitude (t : Triangle) (h : given_condition t) : t.A = π / 6 :=
sorry

-- Theorem for part II
theorem max_area (t : Triangle) (h1 : given_condition t) (h2 : t.a = 2) :
  ∃ (area : ℝ), area ≤ 2 + Real.sqrt 3 ∧
  ∀ (other_area : ℝ), (∃ (t' : Triangle), t'.a = 2 ∧ given_condition t' ∧ 
    other_area = (1 / 2) * t'.b * t'.c * Real.sin t'.A) → other_area ≤ area :=
sorry

end NUMINAMATH_CALUDE_angle_A_magnitude_max_area_l316_31622


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l316_31683

theorem fixed_point_parabola (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 9 * x^2 + 3 * k * x - 6 * k
  f 2 = 36 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l316_31683


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l316_31629

theorem expression_simplification_and_evaluation (a : ℤ) 
  (h1 : -2 < a) (h2 : a ≤ 2) (h3 : a ≠ 0) (h4 : a ≠ 1) :
  (a - (2 * a - 1) / a) / ((a - 1) / a) = a - 1 ∧
  (a = -1 ∨ a = 2) ∧
  (a = -1 → (a - (2 * a - 1) / a) / ((a - 1) / a) = -2) ∧
  (a = 2 → (a - (2 * a - 1) / a) / ((a - 1) / a) = 1) :=
by sorry

#check expression_simplification_and_evaluation

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l316_31629


namespace NUMINAMATH_CALUDE_polynomial_simplification_l316_31625

theorem polynomial_simplification (x : ℝ) :
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l316_31625


namespace NUMINAMATH_CALUDE_unknown_number_value_l316_31631

theorem unknown_number_value (x n : ℤ) : 
  x = 88320 →
  x + n + 9211 - 1569 = 11901 →
  n = -84061 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_value_l316_31631


namespace NUMINAMATH_CALUDE_rahims_average_book_price_l316_31661

/-- Calculates the average price per book given two purchases -/
def average_price_per_book (books1 : ℕ) (price1 : ℕ) (books2 : ℕ) (price2 : ℕ) : ℚ :=
  (price1 + price2 : ℚ) / (books1 + books2 : ℚ)

/-- Proves that Rahim's average price per book is 20 rupees -/
theorem rahims_average_book_price :
  average_price_per_book 50 1000 40 800 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rahims_average_book_price_l316_31661


namespace NUMINAMATH_CALUDE_f_properties_l316_31680

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 - 4*x + 1

-- State the theorem
theorem f_properties :
  (∃ (max_value : ℝ), max_value = 5 ∧ ∀ x, f x ≤ max_value) ∧
  (∀ x y, x < y ∧ y < -2 → f x < f y) ∧
  (∀ x y, -2 < x ∧ x < y → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l316_31680


namespace NUMINAMATH_CALUDE_sequence_relation_l316_31641

theorem sequence_relation (a b : ℕ+ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ+, a n + b n = 1)
  (h3 : ∀ n : ℕ+, b (n + 1) = b n / (1 - (a n)^2)) :
  ∀ n : ℕ+, b n = n / (n + 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_relation_l316_31641


namespace NUMINAMATH_CALUDE_percentage_of_m1_products_l316_31671

theorem percentage_of_m1_products (m1_defective : Real) (m2_defective : Real)
  (m3_non_defective : Real) (m2_percentage : Real) (total_defective : Real) :
  m1_defective = 0.03 →
  m2_defective = 0.01 →
  m3_non_defective = 0.93 →
  m2_percentage = 0.3 →
  total_defective = 0.036 →
  ∃ (m1_percentage : Real),
    m1_percentage = 0.4 ∧
    m1_percentage + m2_percentage + (1 - m1_percentage - m2_percentage) = 1 ∧
    m1_percentage * m1_defective +
    m2_percentage * m2_defective +
    (1 - m1_percentage - m2_percentage) * (1 - m3_non_defective) = total_defective :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_m1_products_l316_31671


namespace NUMINAMATH_CALUDE_ice_cream_yogurt_cost_difference_l316_31635

def ice_cream_cartons : ℕ := 20
def yogurt_cartons : ℕ := 2
def ice_cream_price : ℕ := 6
def yogurt_price : ℕ := 1

theorem ice_cream_yogurt_cost_difference :
  ice_cream_cartons * ice_cream_price - yogurt_cartons * yogurt_price = 118 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_yogurt_cost_difference_l316_31635


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l316_31692

theorem quadratic_root_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ x₂ > 1 ∧ 
   (∀ x : ℝ, x^2 + 2*(m-1)*x - 5*m - 2 = 0 ↔ (x = x₁ ∨ x = x₂))) 
  ↔ m > 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l316_31692


namespace NUMINAMATH_CALUDE_triangular_weight_is_60_l316_31617

/-- The weight of a rectangular weight in grams -/
def rectangular_weight : ℝ := 90

/-- The weight of a round weight in grams -/
def round_weight : ℝ := 30

/-- The weight of a triangular weight in grams -/
def triangular_weight : ℝ := 60

/-- First balance condition: 1 round + 1 triangular = 3 round -/
axiom balance1 : round_weight + triangular_weight = 3 * round_weight

/-- Second balance condition: 4 round + 1 triangular = 1 triangular + 1 round + 1 rectangular -/
axiom balance2 : 4 * round_weight + triangular_weight = triangular_weight + round_weight + rectangular_weight

theorem triangular_weight_is_60 : triangular_weight = 60 := by sorry

end NUMINAMATH_CALUDE_triangular_weight_is_60_l316_31617


namespace NUMINAMATH_CALUDE_finance_class_competition_l316_31667

theorem finance_class_competition (total : ℕ) (abacus : ℕ) (cash_counting : ℕ) (neither : ℕ) :
  total = 48 →
  abacus = 28 →
  cash_counting = 23 →
  neither = 5 →
  ∃ n : ℕ, n = 8 ∧ 
    total = abacus + cash_counting - n + neither :=
by sorry

end NUMINAMATH_CALUDE_finance_class_competition_l316_31667


namespace NUMINAMATH_CALUDE_fraction_quadrupled_l316_31696

theorem fraction_quadrupled (a b : ℚ) (h : a ≠ 0) :
  (2 * b) / (a / 2) = 4 * (b / a) := by sorry

end NUMINAMATH_CALUDE_fraction_quadrupled_l316_31696


namespace NUMINAMATH_CALUDE_total_liquid_proof_l316_31623

/-- The amount of oil used in cups -/
def oil_amount : ℝ := 0.17

/-- The amount of water used in cups -/
def water_amount : ℝ := 1.17

/-- The total amount of liquid used in cups -/
def total_liquid : ℝ := oil_amount + water_amount

theorem total_liquid_proof : total_liquid = 1.34 := by
  sorry

end NUMINAMATH_CALUDE_total_liquid_proof_l316_31623


namespace NUMINAMATH_CALUDE_min_distance_point_is_diagonal_intersection_l316_31689

/-- Given a quadrilateral ABCD in a plane, the point that minimizes the sum of
    distances to all vertices is the intersection of its diagonals. -/
theorem min_distance_point_is_diagonal_intersection
  (A B C D : EuclideanSpace ℝ (Fin 2)) :
  ∃ O : EuclideanSpace ℝ (Fin 2),
    (∀ P : EuclideanSpace ℝ (Fin 2),
      dist O A + dist O B + dist O C + dist O D ≤
      dist P A + dist P B + dist P C + dist P D) ∧
    (∃ t s : ℝ, O = (1 - t) • A + t • C ∧ O = (1 - s) • B + s • D) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_point_is_diagonal_intersection_l316_31689


namespace NUMINAMATH_CALUDE_new_edition_has_450_pages_l316_31697

/-- The number of pages in the old edition of the Geometry book. -/
def old_edition_pages : ℕ := 340

/-- The difference in pages between twice the old edition and the new edition. -/
def page_difference : ℕ := 230

/-- The number of pages in the new edition of the Geometry book. -/
def new_edition_pages : ℕ := 2 * old_edition_pages - page_difference

/-- Theorem stating that the new edition of the Geometry book has 450 pages. -/
theorem new_edition_has_450_pages : new_edition_pages = 450 := by
  sorry

end NUMINAMATH_CALUDE_new_edition_has_450_pages_l316_31697


namespace NUMINAMATH_CALUDE_bakery_children_count_l316_31618

theorem bakery_children_count (initial_count : ℕ) : 
  initial_count + 24 - 31 = 78 → initial_count = 85 := by
  sorry

end NUMINAMATH_CALUDE_bakery_children_count_l316_31618


namespace NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l316_31605

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 - 2*a)^x else Real.log x / Real.log a + 1/3

theorem function_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) ↔ 0 < a ∧ a ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l316_31605


namespace NUMINAMATH_CALUDE_min_value_theorem_l316_31686

/-- Given positive real numbers x and y satisfying x + y = 1,
    if the minimum value of 1/x + a/y is 9, then a = 4 -/
theorem min_value_theorem (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1)
    (hmin : ∀ (u v : ℝ), u > 0 → v > 0 → u + v = 1 → 1/u + a/v ≥ 9) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l316_31686


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l316_31650

theorem christmas_tree_lights (total : ℕ) (red : ℕ) (yellow : ℕ) (green : ℕ) 
  (h1 : total = 350) (h2 : red = 85) (h3 : yellow = 112) (h4 : green = 65) :
  total - (red + yellow + green) = 88 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l316_31650


namespace NUMINAMATH_CALUDE_equation_solution_l316_31698

theorem equation_solution :
  ∃! x : ℝ, (5 : ℝ)^x * 125^(3*x) = 625^7 ∧ x = 2.8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l316_31698


namespace NUMINAMATH_CALUDE_rectangle_grid_40_squares_l316_31688

/-- Represents a rectangle divided into squares -/
structure RectangleGrid where
  rows : ℕ
  cols : ℕ
  total_squares : ℕ
  h_total : rows * cols = total_squares
  h_more_than_one_row : rows > 1
  h_odd_rows : Odd rows

/-- The number of squares not in the middle row of a rectangle grid -/
def squares_not_in_middle_row (r : RectangleGrid) : ℕ :=
  r.total_squares - r.cols

theorem rectangle_grid_40_squares (r : RectangleGrid) 
  (h_40_squares : r.total_squares = 40) :
  squares_not_in_middle_row r = 32 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_grid_40_squares_l316_31688


namespace NUMINAMATH_CALUDE_ratio_e_to_f_l316_31670

theorem ratio_e_to_f (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 3 / 4) :
  e / f = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_e_to_f_l316_31670


namespace NUMINAMATH_CALUDE_workshop_workers_l316_31658

theorem workshop_workers (avg_salary : ℝ) (tech_count : ℕ) (tech_avg_salary : ℝ) (non_tech_avg_salary : ℝ)
  (h1 : avg_salary = 8000)
  (h2 : tech_count = 7)
  (h3 : tech_avg_salary = 10000)
  (h4 : non_tech_avg_salary = 6000) :
  ∃ (total_workers : ℕ), total_workers = 14 ∧
  (tech_count * tech_avg_salary + (total_workers - tech_count) * non_tech_avg_salary) / total_workers = avg_salary :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l316_31658


namespace NUMINAMATH_CALUDE_subset_relation_l316_31643

theorem subset_relation (A B C : Set α) (h : A ∪ B = B ∩ C) : A ⊆ C := by
  sorry

end NUMINAMATH_CALUDE_subset_relation_l316_31643


namespace NUMINAMATH_CALUDE_tiffany_cans_l316_31602

theorem tiffany_cans (initial_bags : ℕ) (next_day_bags : ℕ) (total_bags : ℕ) :
  initial_bags = 10 →
  next_day_bags = 3 →
  total_bags = 20 →
  total_bags - (initial_bags + next_day_bags) = 7 :=
by sorry

end NUMINAMATH_CALUDE_tiffany_cans_l316_31602


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l316_31666

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 - 6*x + 4*x^2 - 5*x^3 + 7*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 4 - 3*x - 7*x^3 + 11*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The coefficient of x^4 in h(x) -/
def coeff_x4 (c : ℝ) : ℝ := 7 + 11*c

/-- The coefficient of x^3 in h(x) -/
def coeff_x3 (c : ℝ) : ℝ := -5 - 7*c

theorem degree_three_polynomial :
  ∃ c : ℝ, coeff_x4 c = 0 ∧ coeff_x3 c ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l316_31666


namespace NUMINAMATH_CALUDE_impossible_closed_line_1989_sticks_l316_31630

theorem impossible_closed_line_1989_sticks : ¬ ∃ (a b : ℕ), 2 * (a + b) = 1989 := by
  sorry

end NUMINAMATH_CALUDE_impossible_closed_line_1989_sticks_l316_31630


namespace NUMINAMATH_CALUDE_coin_collection_dime_difference_l316_31679

theorem coin_collection_dime_difference :
  ∀ (nickels dimes quarters : ℕ),
  nickels + dimes + quarters = 120 →
  5 * nickels + 10 * dimes + 25 * quarters = 1265 →
  quarters ≥ 10 →
  ∃ (min_dimes max_dimes : ℕ),
    (∀ d : ℕ, 
      nickels + d + quarters = 120 ∧ 
      5 * nickels + 10 * d + 25 * quarters = 1265 →
      min_dimes ≤ d ∧ d ≤ max_dimes) ∧
    max_dimes - min_dimes = 92 :=
by sorry

end NUMINAMATH_CALUDE_coin_collection_dime_difference_l316_31679


namespace NUMINAMATH_CALUDE_house_representatives_difference_l316_31682

theorem house_representatives_difference (total : Nat) (democrats : Nat) :
  total = 434 →
  democrats = 202 →
  democrats < total - democrats →
  total - 2 * democrats = 30 := by
sorry

end NUMINAMATH_CALUDE_house_representatives_difference_l316_31682


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_00000428_l316_31638

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_bounds : 1 ≤ |mantissa| ∧ |mantissa| < 10

/-- Conversion function from a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00000428 :
  toScientificNotation 0.00000428 = ScientificNotation.mk 4.28 (-6) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_00000428_l316_31638


namespace NUMINAMATH_CALUDE_wayne_shrimp_appetizer_cost_l316_31699

/-- Calculates the total cost of shrimp for an appetizer given the number of shrimp per guest,
    number of guests, cost per pound of shrimp, and number of shrimp per pound. -/
def shrimp_appetizer_cost (shrimp_per_guest : ℕ) (num_guests : ℕ) (cost_per_pound : ℚ) (shrimp_per_pound : ℕ) : ℚ :=
  (shrimp_per_guest * num_guests : ℚ) / shrimp_per_pound * cost_per_pound

/-- Proves that Wayne's shrimp appetizer cost is $170.00 given the specified conditions. -/
theorem wayne_shrimp_appetizer_cost :
  shrimp_appetizer_cost 5 40 17 20 = 170 :=
by sorry

end NUMINAMATH_CALUDE_wayne_shrimp_appetizer_cost_l316_31699


namespace NUMINAMATH_CALUDE_inequality_equivalence_l316_31634

theorem inequality_equivalence (x : ℝ) (h : x > 0) :
  x^(0.5 * Real.log x / Real.log 0.5 - 3) ≥ 0.5^(3 - 2.5 * Real.log x / Real.log 0.5) ↔ 
  0.125 ≤ x ∧ x ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l316_31634


namespace NUMINAMATH_CALUDE_max_area_triangle_AOB_l316_31637

/-- The maximum area of triangle AOB formed by the intersection points of
    two lines and a curve in polar coordinates. -/
theorem max_area_triangle_AOB :
  ∀ α : Real,
  0 < α →
  α < π / 2 →
  let C₁ := {θ : Real | θ = α}
  let C₂ := {θ : Real | θ = α + π / 2}
  let C₃ := {(ρ, θ) : Real × Real | ρ = 8 * Real.sin θ}
  let A := (8 * Real.sin α, α)
  let B := (8 * Real.cos α, α + π / 2)
  A.1 ≠ 0 ∨ A.2 ≠ 0 →
  B.1 ≠ 0 ∨ B.2 ≠ 0 →
  (∃ (S : Real → Real),
    (∀ α, S α = (1/2) * 8 * Real.sin α * 8 * Real.cos α) ∧
    (∀ α, S α ≤ 16)) :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_AOB_l316_31637


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l316_31651

/-- The surface area of a cylinder with height 2 and base circumference 2π is 6π. -/
theorem cylinder_surface_area : 
  ∀ (h c : ℝ), h = 2 → c = 2 * Real.pi → 
  2 * Real.pi * (c / (2 * Real.pi)) * (c / (2 * Real.pi)) + c * h = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l316_31651


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l316_31632

/-- The longest segment in a cylinder with radius 5 cm and height 6 cm is √136 cm. -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 6) :
  Real.sqrt ((2 * r)^2 + h^2) = Real.sqrt 136 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l316_31632


namespace NUMINAMATH_CALUDE_base12_addition_l316_31606

/-- Converts a base 12 number to base 10 --/
def toBase10 (x : ℕ) (y : ℕ) (z : ℕ) : ℕ := x * 144 + y * 12 + z

/-- Converts a base 10 number to base 12 --/
def toBase12 (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 144
  let b := (n % 144) / 12
  let c := n % 12
  (a, b, c)

theorem base12_addition : 
  let x := toBase10 11 4 8  -- B48 in base 12
  let y := toBase10 5 7 10  -- 57A in base 12
  toBase12 (x + y) = (5, 11, 6) := by sorry

end NUMINAMATH_CALUDE_base12_addition_l316_31606


namespace NUMINAMATH_CALUDE_oldest_sibling_age_l316_31614

theorem oldest_sibling_age (average_age : ℝ) (age1 age2 age3 : ℕ) :
  average_age = 9 ∧ age1 = 5 ∧ age2 = 8 ∧ age3 = 7 →
  ∃ (oldest_age : ℕ), (age1 + age2 + age3 + oldest_age) / 4 = average_age ∧ oldest_age = 16 :=
by sorry

end NUMINAMATH_CALUDE_oldest_sibling_age_l316_31614


namespace NUMINAMATH_CALUDE_smart_mart_puzzles_l316_31664

/-- The number of science kits sold by Smart Mart last week -/
def science_kits : ℕ := 45

/-- The difference between science kits and puzzles sold -/
def difference : ℕ := 9

/-- The number of puzzles sold by Smart Mart last week -/
def puzzles : ℕ := science_kits - difference

/-- Theorem stating that the number of puzzles sold is 36 -/
theorem smart_mart_puzzles : puzzles = 36 := by
  sorry

end NUMINAMATH_CALUDE_smart_mart_puzzles_l316_31664


namespace NUMINAMATH_CALUDE_cos_2alpha_on_unit_circle_l316_31653

theorem cos_2alpha_on_unit_circle (α : Real) :
  (Real.cos α = -Real.sqrt 5 / 5 ∧ Real.sin α = 2 * Real.sqrt 5 / 5) →
  Real.cos (2 * α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_on_unit_circle_l316_31653


namespace NUMINAMATH_CALUDE_intersection_with_complement_l316_31646

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set P
def P : Set Nat := {1, 2, 3, 4}

-- Define set Q
def Q : Set Nat := {3, 4, 5}

-- Theorem statement
theorem intersection_with_complement : P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l316_31646


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l316_31626

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assumption that the hexagon is equilateral
  is_equilateral : True
  -- Assumption that three nonadjacent acute interior angles measure 45°
  has_45_degree_angles : True
  -- The enclosed area of the hexagon
  area : ℝ
  -- The area is 12√2
  area_eq : area = 12 * Real.sqrt 2

/-- The perimeter of a hexagon is 6 times its side length -/
def perimeter (h : SpecialHexagon) : ℝ := 6 * h.side

/-- Theorem: The perimeter of the special hexagon is 24√2 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : perimeter h = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l316_31626


namespace NUMINAMATH_CALUDE_coronavirus_survey_census_l316_31600

/-- A survey type -/
inductive SurveyType
| HeightOfStudents
| LightBulbLifespan
| GlobalGenderRatio
| CoronavirusExposure

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  smallGroup : Bool
  specificGroup : Bool
  completeDataNecessary : Bool

/-- Define what makes a survey suitable for a census -/
def suitableForCensus (c : SurveyCharacteristics) : Prop :=
  c.smallGroup ∧ c.specificGroup ∧ c.completeDataNecessary

/-- Assign characteristics to each survey type -/
def surveyCharacteristics : SurveyType → SurveyCharacteristics
| SurveyType.HeightOfStudents => ⟨false, true, false⟩
| SurveyType.LightBulbLifespan => ⟨true, true, false⟩
| SurveyType.GlobalGenderRatio => ⟨false, false, false⟩
| SurveyType.CoronavirusExposure => ⟨true, true, true⟩

/-- Theorem: The coronavirus exposure survey is the only one suitable for a census -/
theorem coronavirus_survey_census :
  ∀ (s : SurveyType), suitableForCensus (surveyCharacteristics s) ↔ s = SurveyType.CoronavirusExposure :=
sorry

end NUMINAMATH_CALUDE_coronavirus_survey_census_l316_31600


namespace NUMINAMATH_CALUDE_custom_mul_four_three_l316_31644

/-- Custom multiplication operation -/
def custom_mul (a b : ℤ) : ℤ := a^2 + a*b + a - b^2

/-- Theorem stating that 4 * 3 = 23 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_four_three_l316_31644


namespace NUMINAMATH_CALUDE_expand_expression_l316_31608

theorem expand_expression (x : ℝ) : (5 * x^2 - 3) * 4 * x^3 = 20 * x^5 - 12 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l316_31608


namespace NUMINAMATH_CALUDE_length_AE_is_5_sqrt_5_div_3_l316_31616

-- Define the points
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (4, 2)
def D : ℝ × ℝ := (2, 0)

-- Define E as the intersection point of AB and CD
def E : ℝ × ℝ := sorry

-- Theorem statement
theorem length_AE_is_5_sqrt_5_div_3 :
  let dist := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A E = (5 * Real.sqrt 5) / 3 := by sorry

end NUMINAMATH_CALUDE_length_AE_is_5_sqrt_5_div_3_l316_31616


namespace NUMINAMATH_CALUDE_light_ray_exits_l316_31620

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ
  length_pos : length > 0

/-- Represents an angle formed by two segments with a common vertex -/
structure Angle where
  seg1 : Segment
  seg2 : Segment

/-- Represents a light ray traveling inside an angle -/
structure LightRay where
  angle : Angle
  position : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a light ray has exited an angle -/
def has_exited (ray : LightRay) : Prop :=
  -- Implementation details omitted
  sorry

/-- Function to update the light ray's position and direction after a reflection -/
def reflect (ray : LightRay) : LightRay :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that a light ray will eventually exit the angle -/
theorem light_ray_exits (angle : Angle) :
  ∃ (n : ℕ), ∀ (ray : LightRay), ray.angle = angle →
    has_exited (n.iterate reflect ray) :=
  sorry

end NUMINAMATH_CALUDE_light_ray_exits_l316_31620


namespace NUMINAMATH_CALUDE_floor_greater_than_fraction_l316_31633

theorem floor_greater_than_fraction (a : ℝ) (n : ℤ) 
  (h1 : a ≥ 1) (h2 : 0 ≤ n) (h3 : n ≤ a) :
  Int.floor a > (n / (n + 1 : ℝ)) * a := by
  sorry

end NUMINAMATH_CALUDE_floor_greater_than_fraction_l316_31633


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l316_31659

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : A = 4 * Real.pi → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l316_31659


namespace NUMINAMATH_CALUDE_farm_cows_l316_31656

theorem farm_cows (milk_per_week : ℝ) (total_milk : ℝ) (num_weeks : ℕ) :
  milk_per_week = 108 →
  total_milk = 2160 →
  num_weeks = 5 →
  (total_milk / (milk_per_week / 6 * num_weeks) : ℝ) = 24 :=
by sorry

end NUMINAMATH_CALUDE_farm_cows_l316_31656


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l316_31676

/-- The polynomial x^2 + ax + 2a has two distinct integer roots if and only if a = -1 or a = 9 -/
theorem quadratic_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + a*x + 2*a = 0 ∧ y^2 + a*y + 2*a = 0) ↔ (a = -1 ∨ a = 9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l316_31676


namespace NUMINAMATH_CALUDE_cricket_score_problem_l316_31611

theorem cricket_score_problem :
  ∀ (a b c d e : ℕ),
    -- Average score is 36
    a + b + c + d + e = 36 * 5 →
    -- D scored 5 more than E
    d = e + 5 →
    -- E scored 8 fewer than A
    e = a - 8 →
    -- B scored as many as D and E combined
    b = d + e →
    -- E scored 20 runs
    e = 20 →
    -- Prove that B and C scored 107 runs between them
    b + c = 107 := by
  sorry

end NUMINAMATH_CALUDE_cricket_score_problem_l316_31611


namespace NUMINAMATH_CALUDE_problem_statement_l316_31640

theorem problem_statement (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l316_31640


namespace NUMINAMATH_CALUDE_cos_sin_10_deg_equality_l316_31609

theorem cos_sin_10_deg_equality : 
  4 * Real.cos (10 * π / 180) - Real.cos (10 * π / 180) / Real.sin (10 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_10_deg_equality_l316_31609


namespace NUMINAMATH_CALUDE_sally_sock_order_l316_31690

/-- The ratio of black socks to blue socks in Sally's original order -/
def sock_ratio : ℚ := 5

theorem sally_sock_order :
  ∀ (x : ℝ) (b : ℕ),
  x > 0 →  -- Price of black socks is positive
  b > 0 →  -- Number of blue socks is positive
  (5 * x + 3 * b * x) * 2 = b * x + 15 * x →  -- Doubled bill condition
  sock_ratio = 5 := by
sorry

end NUMINAMATH_CALUDE_sally_sock_order_l316_31690


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l316_31674

/-- The number of sheep on Stewart farm given the ratio of sheep to horses and food consumption. -/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ) (food_per_horse total_food : ℕ),
  sheep * 7 = horses * 6 →  -- Ratio of sheep to horses is 6:7
  food_per_horse = 230 →    -- Each horse eats 230 ounces per day
  horses * food_per_horse = total_food →  -- Total food consumed by horses
  total_food = 12880 →      -- Total food needed is 12,880 ounces
  sheep = 48 := by
sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l316_31674


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l316_31663

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = 5) : a^2 + 1/a^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l316_31663


namespace NUMINAMATH_CALUDE_bacon_students_count_l316_31694

-- Define the total number of students
def total_students : ℕ := 310

-- Define the number of students who suggested mashed potatoes
def mashed_potatoes_students : ℕ := 185

-- Theorem to prove
theorem bacon_students_count : total_students - mashed_potatoes_students = 125 := by
  sorry

end NUMINAMATH_CALUDE_bacon_students_count_l316_31694


namespace NUMINAMATH_CALUDE_complex_fraction_value_l316_31612

theorem complex_fraction_value : 
  let i : ℂ := Complex.I
  (3 + i) / (1 - i) = 1 + 2*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l316_31612


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l316_31678

theorem gcd_of_three_numbers : Nat.gcd 13456 (Nat.gcd 25345 15840) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l316_31678


namespace NUMINAMATH_CALUDE_polynomial_factorization_l316_31657

theorem polynomial_factorization (a b c : ℝ) : 
  a*(b - c)^4 + b*(c - a)^4 + c*(a - b)^4 = (a - b)*(b - c)*(c - a)*(a*b^2 + a*c^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l316_31657


namespace NUMINAMATH_CALUDE_remainder_problem_l316_31603

theorem remainder_problem (N : ℤ) : N % 1927 = 131 → (3 * N) % 43 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l316_31603


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l316_31610

/-- Given a car price, total amount to pay, and loan amount, calculate the interest rate -/
theorem calculate_interest_rate 
  (car_price : ℝ) 
  (total_amount : ℝ) 
  (loan_amount : ℝ) 
  (h1 : car_price = 35000)
  (h2 : total_amount = 38000)
  (h3 : loan_amount = 20000) :
  (total_amount - loan_amount) / loan_amount * 100 = 90 := by
  sorry

#check calculate_interest_rate

end NUMINAMATH_CALUDE_calculate_interest_rate_l316_31610


namespace NUMINAMATH_CALUDE_average_string_length_l316_31684

theorem average_string_length : 
  let string1 : ℝ := 2.5
  let string2 : ℝ := 3.5
  let string3 : ℝ := 4.5
  let total_length : ℝ := string1 + string2 + string3
  let num_strings : ℕ := 3
  (total_length / num_strings) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_average_string_length_l316_31684


namespace NUMINAMATH_CALUDE_total_daily_salary_l316_31655

def grocery_store_salaries (manager_salary clerk_salary : ℕ) (num_managers num_clerks : ℕ) : ℕ :=
  manager_salary * num_managers + clerk_salary * num_clerks

theorem total_daily_salary :
  grocery_store_salaries 5 2 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_daily_salary_l316_31655


namespace NUMINAMATH_CALUDE_fib_fraction_numerator_l316_31668

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: The simplified fraction of (F₂₀₀₃/F₂₀₀₂ - F₂₀₀₄/F₂₀₀₃) has numerator 1 -/
theorem fib_fraction_numerator :
  (fib 2003 : ℚ) / fib 2002 - (fib 2004 : ℚ) / fib 2003 = 1 / (fib 2002 * fib 2003) :=
by sorry

end NUMINAMATH_CALUDE_fib_fraction_numerator_l316_31668


namespace NUMINAMATH_CALUDE_achieve_any_distribution_l316_31675

-- Define the Student type
def Student : Type := ℕ

-- Define the friendship relation
def IsFriend (s1 s2 : Student) : Prop := sorry

-- Define the candy distribution
def CandyCount : Student → Fin 7 := sorry

-- Define the property of friendship for the set of students
def FriendshipProperty (students : Set Student) : Prop :=
  ∀ s1 s2 : Student, s1 ∈ students → s2 ∈ students → s1 ≠ s2 →
    ∃ s3 ∈ students, (IsFriend s3 s1 ∧ ¬IsFriend s3 s2) ∨ (IsFriend s3 s2 ∧ ¬IsFriend s3 s1)

-- Define a step in the candy distribution process
def DistributionStep (students : Set Student) (initial : Student → Fin 7) : Student → Fin 7 := sorry

-- Theorem: Any desired candy distribution can be achieved in finitely many steps
theorem achieve_any_distribution 
  (students : Set Student) 
  (h_finite : Finite students) 
  (h_friendship : FriendshipProperty students) 
  (initial : Student → Fin 7) 
  (target : Student → Fin 7) :
  ∃ n : ℕ, ∃ steps : Fin n → (Set Student), 
    (DistributionStep students)^[n] initial = target := by
  sorry

end NUMINAMATH_CALUDE_achieve_any_distribution_l316_31675


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l316_31607

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - 2*x*y = 0) :
  ∀ z, z = 2*x + y → z ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l316_31607


namespace NUMINAMATH_CALUDE_regular_adult_ticket_price_correct_l316_31604

/-- The regular price of an adult movie ticket given the following conditions:
  * There are 5 adults and 2 children.
  * Children's concessions cost $3 each.
  * Adults' concessions cost $5, $6, $7, $4, and $9 respectively.
  * Total cost of the trip is $139.
  * Each child's ticket costs $7.
  * Three adults have discounts of $3, $2, and $1 on their tickets.
-/
def regular_adult_ticket_price : ℚ :=
  let num_adults : ℕ := 5
  let num_children : ℕ := 2
  let child_concession_cost : ℚ := 3
  let adult_concession_costs : List ℚ := [5, 6, 7, 4, 9]
  let total_trip_cost : ℚ := 139
  let child_ticket_cost : ℚ := 7
  let adult_ticket_discounts : List ℚ := [3, 2, 1]
  18.8

theorem regular_adult_ticket_price_correct :
  let num_adults : ℕ := 5
  let num_children : ℕ := 2
  let child_concession_cost : ℚ := 3
  let adult_concession_costs : List ℚ := [5, 6, 7, 4, 9]
  let total_trip_cost : ℚ := 139
  let child_ticket_cost : ℚ := 7
  let adult_ticket_discounts : List ℚ := [3, 2, 1]
  regular_adult_ticket_price = 18.8 := by
  sorry

#eval regular_adult_ticket_price

end NUMINAMATH_CALUDE_regular_adult_ticket_price_correct_l316_31604


namespace NUMINAMATH_CALUDE_probability_neither_prime_nor_composite_l316_31691

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

theorem probability_neither_prime_nor_composite :
  let S := Finset.range 97
  let E := {1}
  (Finset.card E : ℚ) / (Finset.card S : ℚ) = 1 / 97 :=
by sorry

end NUMINAMATH_CALUDE_probability_neither_prime_nor_composite_l316_31691


namespace NUMINAMATH_CALUDE_train_length_l316_31654

/-- Calculates the length of a train given its speed and the time it takes to cross a platform of known length. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * 1000 / 3600 →
  platform_length = 250 →
  crossing_time = 15 →
  (train_speed * crossing_time) - platform_length = 50 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l316_31654


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_9_l316_31695

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1 : ℤ) * d) / 2

theorem arithmetic_sequence_sum_9 (a₁ d : ℤ) :
  a₁ = 2 →
  arithmetic_sequence a₁ d 5 = 3 * arithmetic_sequence a₁ d 3 →
  sum_arithmetic_sequence a₁ d 9 = -54 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_9_l316_31695


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2013_l316_31665

theorem tens_digit_of_8_pow_2013 : ∃ n : ℕ, 8^2013 ≡ 88 + 100*n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2013_l316_31665


namespace NUMINAMATH_CALUDE_calculate_expression_l316_31648

theorem calculate_expression : 
  (8/27)^(2/3) + Real.log 3 / Real.log 12 + 2 * Real.log 2 / Real.log 12 = 13/9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l316_31648


namespace NUMINAMATH_CALUDE_journeymen_ratio_after_layoff_journeymen_fraction_is_two_thirds_l316_31662

/-- The total number of employees in the anvil factory -/
def total_employees : ℕ := 20210

/-- The fraction of employees who are journeymen -/
def journeymen_fraction : ℚ := sorry

/-- The number of journeymen after laying off half of them -/
def remaining_journeymen : ℚ := journeymen_fraction * (total_employees : ℚ) / 2

/-- The total number of employees after laying off half of the journeymen -/
def remaining_employees : ℚ := (total_employees : ℚ) - remaining_journeymen

/-- The condition that after laying off half of the journeymen, they constitute 50% of the remaining workforce -/
theorem journeymen_ratio_after_layoff : remaining_journeymen / remaining_employees = 1 / 2 := sorry

/-- The main theorem: proving that the fraction of employees who are journeymen is 2/3 -/
theorem journeymen_fraction_is_two_thirds : journeymen_fraction = 2 / 3 := sorry

end NUMINAMATH_CALUDE_journeymen_ratio_after_layoff_journeymen_fraction_is_two_thirds_l316_31662


namespace NUMINAMATH_CALUDE_martha_clothes_count_l316_31621

/-- Calculates the total number of clothes Martha takes home given the number of jackets and t-shirts bought -/
def total_clothes (jackets_bought : ℕ) (tshirts_bought : ℕ) : ℕ :=
  let free_jackets := jackets_bought / 2
  let free_tshirts := tshirts_bought / 3
  jackets_bought + free_jackets + tshirts_bought + free_tshirts

/-- Proves that Martha takes home 18 clothes when buying 4 jackets and 9 t-shirts -/
theorem martha_clothes_count : total_clothes 4 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_martha_clothes_count_l316_31621


namespace NUMINAMATH_CALUDE_fraction_unchanged_l316_31681

theorem fraction_unchanged (x y : ℝ) : (5 * x) / (x + y) = (5 * (10 * x)) / ((10 * x) + (10 * y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l316_31681


namespace NUMINAMATH_CALUDE_solve_system_l316_31672

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 8) (eq2 : x + 3 * y = 7) : x = 38 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l316_31672
