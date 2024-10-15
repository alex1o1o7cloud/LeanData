import Mathlib

namespace NUMINAMATH_CALUDE_no_rational_roots_l1177_117735

theorem no_rational_roots : ∀ (p q : ℤ), q ≠ 0 → 3 * (p / q)^4 - 2 * (p / q)^3 - 8 * (p / q)^2 + (p / q) + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l1177_117735


namespace NUMINAMATH_CALUDE_min_value_expression_l1177_117734

theorem min_value_expression (x y : ℝ) (h : x^2 + y^2 ≤ 1) :
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 + b^2 ≤ 1 → |2*a + b - 2| + |6 - a - 3*b| ≥ m) ∧
             (∃ (c d : ℝ), c^2 + d^2 ≤ 1 ∧ |2*c + d - 2| + |6 - c - 3*d| = m) ∧
             m = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1177_117734


namespace NUMINAMATH_CALUDE_ac_lt_zero_sufficient_not_necessary_l1177_117733

theorem ac_lt_zero_sufficient_not_necessary (a b c : ℝ) (h : c < b ∧ b < a) :
  (∀ x y z : ℝ, x < y ∧ y < z ∧ x*z < 0 → z*y > z*x) ∧
  (∃ x y z : ℝ, x < y ∧ y < z ∧ z*y > z*x ∧ x*z ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ac_lt_zero_sufficient_not_necessary_l1177_117733


namespace NUMINAMATH_CALUDE_largest_value_is_2_pow_35_l1177_117721

theorem largest_value_is_2_pow_35 : 
  (2 ^ 35 : ℕ) > 26 ∧ (2 ^ 35 : ℕ) > 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_is_2_pow_35_l1177_117721


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_even_numbers_l1177_117774

theorem sum_of_three_consecutive_even_numbers (m : ℤ) : 
  m % 2 = 0 → (m + (m + 2) + (m + 4)) = 3 * m + 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_even_numbers_l1177_117774


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l1177_117791

def n : ℕ := 240360

theorem sum_of_distinct_prime_factors :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id) = 62 :=
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l1177_117791


namespace NUMINAMATH_CALUDE_pet_store_ratio_l1177_117726

theorem pet_store_ratio (dogs : ℕ) (total : ℕ) : 
  dogs = 6 → 
  total = 39 → 
  (dogs + dogs / 2 + 2 * dogs + (total - (dogs + dogs / 2 + 2 * dogs))) / dogs = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_ratio_l1177_117726


namespace NUMINAMATH_CALUDE_andrew_cookie_expenditure_l1177_117714

/-- The number of days in May -/
def days_in_may : ℕ := 31

/-- The number of cookies Andrew purchases each day -/
def cookies_per_day : ℕ := 3

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 15

/-- The total amount Andrew spent on cookies in May -/
def total_spent : ℕ := days_in_may * cookies_per_day * cost_per_cookie

/-- Theorem stating that Andrew spent 1395 dollars on cookies in May -/
theorem andrew_cookie_expenditure : total_spent = 1395 := by
  sorry

end NUMINAMATH_CALUDE_andrew_cookie_expenditure_l1177_117714


namespace NUMINAMATH_CALUDE_rod_cutting_l1177_117745

theorem rod_cutting (rod_length : Real) (num_pieces : Nat) (piece_length_cm : Real) : 
  rod_length = 29.75 ∧ num_pieces = 35 → piece_length_cm = 85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l1177_117745


namespace NUMINAMATH_CALUDE_ratio_to_percent_l1177_117718

theorem ratio_to_percent (a b : ℕ) (h : a = 2 ∧ b = 3) : (a : ℚ) / (a + b : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percent_l1177_117718


namespace NUMINAMATH_CALUDE_parallelogram_with_equal_vector_sums_is_rectangle_l1177_117737

/-- A parallelogram ABCD with vertices A, B, C, and D. -/
structure Parallelogram (V : Type*) [NormedAddCommGroup V] :=
  (A B C D : V)
  (is_parallelogram : (B - A) = (C - D) ∧ (D - A) = (C - B))

/-- Definition of a rectangle as a parallelogram with equal diagonals. -/
def is_rectangle {V : Type*} [NormedAddCommGroup V] (p : Parallelogram V) : Prop :=
  ‖p.C - p.A‖ = ‖p.D - p.B‖

theorem parallelogram_with_equal_vector_sums_is_rectangle
  {V : Type*} [NormedAddCommGroup V] (p : Parallelogram V) :
  ‖p.B - p.A + (p.D - p.A)‖ = ‖p.B - p.A - (p.D - p.A)‖ →
  is_rectangle p :=
sorry

end NUMINAMATH_CALUDE_parallelogram_with_equal_vector_sums_is_rectangle_l1177_117737


namespace NUMINAMATH_CALUDE_magic_square_b_plus_c_l1177_117770

/-- Represents a 3x3 magic square with the given layout -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  S : ℕ
  row1_sum : 30 + b + 18 = S
  row2_sum : 15 + c + d = S
  row3_sum : a + 33 + e = S
  col1_sum : 30 + 15 + a = S
  col2_sum : b + c + 33 = S
  col3_sum : 18 + d + e = S
  diag1_sum : 30 + c + e = S
  diag2_sum : 18 + c + a = S

/-- The sum of b and c in a magic square is 33 -/
theorem magic_square_b_plus_c (ms : MagicSquare) : ms.b + ms.c = 33 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_b_plus_c_l1177_117770


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1177_117703

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the perpendicular bisector of a line with slope m
def perpendicular_bisector (m : ℝ) (x y : ℝ) : Prop := x = -1/m * y + (2*m^2 + 3)

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

theorem parabola_intersection_theorem (m : ℝ) :
  ∃ (xA yA xB yB xC yC xD yD : ℝ),
    -- A and B are on the parabola and the line through focus
    parabola xA yA ∧ parabola xB yB ∧
    line_through_focus m xA yA ∧ line_through_focus m xB yB ∧
    -- C and D are on the parabola and the perpendicular bisector
    parabola xC yC ∧ parabola xD yD ∧
    perpendicular_bisector m xC yC ∧ perpendicular_bisector m xD yD ∧
    -- AC is perpendicular to AD
    dot_product (xC - xA) (yC - yA) (xD - xA) (yD - yA) = 0 →
    m = 1 ∨ m = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1177_117703


namespace NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l1177_117702

theorem consecutive_integers_fourth_power_sum (a b c : ℤ) : 
  (b = a + 1) →
  (c = b + 1) →
  (a^2 + b^2 + c^2 = 12246) →
  (a^4 + b^4 + c^4 = 50380802) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l1177_117702


namespace NUMINAMATH_CALUDE_james_profit_l1177_117799

def total_toys : ℕ := 200
def buy_price : ℕ := 20
def sell_price : ℕ := 30
def sell_percentage : ℚ := 80 / 100

theorem james_profit :
  (↑total_toys * sell_percentage * sell_price : ℚ) -
  (↑total_toys * sell_percentage * buy_price : ℚ) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_james_profit_l1177_117799


namespace NUMINAMATH_CALUDE_satisfying_function_characterization_l1177_117709

/-- A function from positive reals to reals satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 →
    (f x + f y ≤ f (x + y) / 2) ∧
    (f x / x + f y / y ≥ f (x + y) / (x + y))

/-- The theorem stating that any satisfying function must be of the form f(x) = ax² where a ≤ 0 -/
theorem satisfying_function_characterization (f : ℝ → ℝ) :
  SatisfyingFunction f →
  ∃ a : ℝ, a ≤ 0 ∧ ∀ x : ℝ, x > 0 → f x = a * x^2 :=
sorry

end NUMINAMATH_CALUDE_satisfying_function_characterization_l1177_117709


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1177_117711

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 24)
  (area2 : w * h = 16)
  (area3 : l * h = 6) :
  l * w * h = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1177_117711


namespace NUMINAMATH_CALUDE_reflection_theorem_l1177_117785

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a given line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Checks if two points are symmetric with respect to the line x + y = 0 --/
def symmetricPoints (p1 p2 : Point) : Prop :=
  p2.x = -p1.y ∧ p2.y = -p1.x

/-- The main theorem to prove --/
theorem reflection_theorem :
  ∀ (b : ℝ),
  let incident_ray : Line := { slope := -3, intercept := b }
  let reflected_ray : Line := { slope := -1/3, intercept := 2 }
  let incident_point : Point := { x := 1, y := b - 3 }
  let reflected_point : Point := { x := -b + 3, y := -1 }
  pointOnLine incident_point incident_ray ∧
  pointOnLine reflected_point reflected_ray ∧
  symmetricPoints incident_point reflected_point →
  b = -6 := by
sorry

end NUMINAMATH_CALUDE_reflection_theorem_l1177_117785


namespace NUMINAMATH_CALUDE_six_digit_concatenation_divisibility_l1177_117712

theorem six_digit_concatenation_divisibility :
  ∀ a b : ℕ,
    100000 ≤ a ∧ a < 1000000 →
    100000 ≤ b ∧ b < 1000000 →
    (∃ k : ℕ, 1000000 * a + b = k * a * b) →
    (a = 166667 ∧ b = 333334) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_concatenation_divisibility_l1177_117712


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1177_117723

theorem point_in_fourth_quadrant :
  let P : ℝ × ℝ := (Real.tan (549 * π / 180), Real.cos (549 * π / 180))
  (P.1 > 0) ∧ (P.2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1177_117723


namespace NUMINAMATH_CALUDE_evaluate_expression_l1177_117756

theorem evaluate_expression : -(18 / 3 * 8 - 40 + 5^2) = -33 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1177_117756


namespace NUMINAMATH_CALUDE_jose_weekly_earnings_l1177_117779

/-- Calculates Jose's weekly earnings from his swimming pool. -/
theorem jose_weekly_earnings :
  let kid_price : ℕ := 3
  let adult_price : ℕ := 2 * kid_price
  let kids_per_day : ℕ := 8
  let adults_per_day : ℕ := 10
  let days_per_week : ℕ := 7
  
  (kid_price * kids_per_day + adult_price * adults_per_day) * days_per_week = 588 :=
by sorry

end NUMINAMATH_CALUDE_jose_weekly_earnings_l1177_117779


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l1177_117724

theorem percentage_of_sikh_boys (total_boys : ℕ) (muslim_percentage hindu_percentage : ℚ) 
  (other_boys : ℕ) (h1 : total_boys = 850) (h2 : muslim_percentage = 40/100) 
  (h3 : hindu_percentage = 28/100) (h4 : other_boys = 187) : 
  (total_boys - (muslim_percentage * total_boys + hindu_percentage * total_boys + other_boys)) / total_boys = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l1177_117724


namespace NUMINAMATH_CALUDE_outfit_combinations_l1177_117758

/-- Represents the number of shirts Li Fang has -/
def num_shirts : ℕ := 4

/-- Represents the number of skirts Li Fang has -/
def num_skirts : ℕ := 3

/-- Represents the number of dresses Li Fang has -/
def num_dresses : ℕ := 2

/-- Calculates the total number of outfit combinations -/
def total_outfits : ℕ := num_shirts * num_skirts + num_dresses

/-- Theorem stating that the total number of outfit combinations is 14 -/
theorem outfit_combinations : total_outfits = 14 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1177_117758


namespace NUMINAMATH_CALUDE_sum_of_powers_l1177_117753

theorem sum_of_powers (ω : ℂ) (h1 : ω^11 = 1) (h2 : ω ≠ 1) :
  ω^10 + ω^14 + ω^18 + ω^22 + ω^26 + ω^30 + ω^34 + ω^38 + ω^42 + ω^46 + ω^50 + ω^54 + ω^58 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1177_117753


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1177_117769

theorem complex_equation_solution (z : ℂ) : (z - Complex.I) * (2 - Complex.I) = 5 → z = 2 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1177_117769


namespace NUMINAMATH_CALUDE_card_probability_ratio_l1177_117704

def num_cards : ℕ := 40
def num_numbers : ℕ := 10
def cards_per_number : ℕ := 4
def cards_drawn : ℕ := 4

def p : ℚ := num_numbers / (num_cards.choose cards_drawn)
def q : ℚ := (num_numbers * (num_numbers - 1) * (cards_per_number.choose 3) * (cards_per_number.choose 1)) / (num_cards.choose cards_drawn)

theorem card_probability_ratio : q / p = 144 := by
  sorry

end NUMINAMATH_CALUDE_card_probability_ratio_l1177_117704


namespace NUMINAMATH_CALUDE_senior_tickets_sold_l1177_117787

/-- Proves the number of senior citizen tickets sold given the total tickets,
    ticket prices, and total receipts -/
theorem senior_tickets_sold
  (total_tickets : ℕ)
  (adult_price senior_price : ℕ)
  (total_receipts : ℕ)
  (h1 : total_tickets = 510)
  (h2 : adult_price = 21)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 8748) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end NUMINAMATH_CALUDE_senior_tickets_sold_l1177_117787


namespace NUMINAMATH_CALUDE_x_varies_as_z_l1177_117717

-- Define the variables and constants
variable (x y z : ℝ)
variable (k j : ℝ)

-- Define the conditions
axiom x_varies_as_y : ∃ k, x = k * y^3
axiom y_varies_as_z : ∃ j, y = j * z^(1/4)

-- Define the theorem to prove
theorem x_varies_as_z : ∃ m, x = m * z^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_x_varies_as_z_l1177_117717


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l1177_117765

def q (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 4

theorem cubic_polynomial_satisfies_conditions :
  q 1 = -8 ∧ q 2 = -10 ∧ q 3 = -16 ∧ q 4 = -32 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l1177_117765


namespace NUMINAMATH_CALUDE_dandelion_counts_l1177_117708

/-- Represents the state of dandelions in the meadow on a given day -/
structure DandelionState where
  yellow : ℕ
  white : ℕ

/-- The lifecycle of dandelions -/
def dandelionLifecycle : Prop :=
  ∀ d : DandelionState, d.yellow = d.white

/-- Yesterday's dandelion state -/
def yesterday : DandelionState :=
  { yellow := 20, white := 14 }

/-- Today's dandelion state -/
def today : DandelionState :=
  { yellow := 15, white := 11 }

/-- Theorem: Given the dandelion lifecycle and the counts for yesterday and today,
    the number of yellow dandelions the day before yesterday was 25, and
    the number of white dandelions tomorrow will be 9. -/
theorem dandelion_counts
  (h : dandelionLifecycle)
  (hy : yesterday.yellow = 20 ∧ yesterday.white = 14)
  (ht : today.yellow = 15 ∧ today.white = 11) :
  (yesterday.white + today.white = 25) ∧
  (yesterday.yellow - today.white = 9) :=
by sorry

end NUMINAMATH_CALUDE_dandelion_counts_l1177_117708


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1177_117731

theorem quadratic_root_relation (a b : ℝ) : 
  (3 : ℝ)^2 + 2*a*3 + 3*b = 0 → 2*a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1177_117731


namespace NUMINAMATH_CALUDE_corner_subset_exists_l1177_117773

/-- A corner is a finite set of n-tuples of positive integers with a specific property. -/
def Corner (n : ℕ) : Type :=
  {S : Set (Fin n → ℕ+) // S.Finite ∧
    ∀ a b : Fin n → ℕ+, a ∈ S → (∀ k, b k ≤ a k) → b ∈ S}

/-- The theorem states that in any infinite collection of corners,
    there exist two corners where one is a subset of the other. -/
theorem corner_subset_exists {n : ℕ} (h : n > 0) (S : Set (Corner n)) (hS : Set.Infinite S) :
  ∃ C₁ C₂ : Corner n, C₁ ∈ S ∧ C₂ ∈ S ∧ C₁.1 ⊆ C₂.1 :=
sorry

end NUMINAMATH_CALUDE_corner_subset_exists_l1177_117773


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l1177_117751

theorem parakeets_per_cage (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ) :
  num_cages = 6 →
  parrots_per_cage = 6 →
  total_birds = 48 →
  (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := by
  sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l1177_117751


namespace NUMINAMATH_CALUDE_zachary_pushups_l1177_117746

/-- Given that Zachary did 14 crunches and a total of 67 push-ups and crunches,
    prove that Zachary did 53 push-ups. -/
theorem zachary_pushups :
  ∀ (zachary_pushups zachary_crunches : ℕ),
    zachary_crunches = 14 →
    zachary_pushups + zachary_crunches = 67 →
    zachary_pushups = 53 :=
by
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_l1177_117746


namespace NUMINAMATH_CALUDE_cost_of_cakes_l1177_117722

/-- The cost of cakes problem -/
theorem cost_of_cakes (num_cakes : ℕ) (johns_share : ℚ) (cost_per_cake : ℚ) 
  (h1 : num_cakes = 3)
  (h2 : johns_share = 18)
  (h3 : johns_share * 2 = num_cakes * cost_per_cake) :
  cost_per_cake = 12 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_cakes_l1177_117722


namespace NUMINAMATH_CALUDE_complex_power_difference_l1177_117783

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^40 - (1 - i)^40 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1177_117783


namespace NUMINAMATH_CALUDE_strawberry_price_difference_l1177_117719

/-- Proves that the difference in price per pint between the regular price and the sale price is $2 --/
theorem strawberry_price_difference
  (pints_sold : ℕ)
  (sale_revenue : ℚ)
  (revenue_difference : ℚ)
  (h1 : pints_sold = 54)
  (h2 : sale_revenue = 216)
  (h3 : revenue_difference = 108)
  : (sale_revenue + revenue_difference) / pints_sold - sale_revenue / pints_sold = 2 := by
  sorry

#check strawberry_price_difference

end NUMINAMATH_CALUDE_strawberry_price_difference_l1177_117719


namespace NUMINAMATH_CALUDE_fifteen_consecutive_naturals_l1177_117797

theorem fifteen_consecutive_naturals (N : ℕ) : 
  (N < 81 ∧ 
   ∀ k : ℕ, (N < k ∧ k < 81) → (k - N ≤ 15)) ∧ 
  (∃ m : ℕ, N < m ∧ m < 81 ∧ m - N = 15) →
  N = 66 := by
sorry

end NUMINAMATH_CALUDE_fifteen_consecutive_naturals_l1177_117797


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l1177_117777

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line passing through the intersection points of two circles -/
def intersectionLine (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = 6.5

theorem intersection_line_of_circles :
  let c1 : Circle := { center := (5, -2), radius := 7 }
  let c2 : Circle := { center := (-1, 5), radius := 5 }
  ∃ (p1 p2 : ℝ × ℝ),
    (p1.1 - c1.center.1)^2 + (p1.2 - c1.center.2)^2 = c1.radius^2 ∧
    (p1.1 - c2.center.1)^2 + (p1.2 - c2.center.2)^2 = c2.radius^2 ∧
    (p2.1 - c1.center.1)^2 + (p2.2 - c1.center.2)^2 = c1.radius^2 ∧
    (p2.1 - c2.center.1)^2 + (p2.2 - c2.center.2)^2 = c2.radius^2 ∧
    p1 ≠ p2 ∧
    intersectionLine c1 c2 p1.1 p1.2 ∧
    intersectionLine c1 c2 p2.1 p2.2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l1177_117777


namespace NUMINAMATH_CALUDE_sum_first_seven_odd_numbers_l1177_117701

def sum_odd_numbers (n : ℕ) : ℕ := (2 * n - 1) * n

theorem sum_first_seven_odd_numbers :
  (sum_odd_numbers 2 = 2^2) →
  (sum_odd_numbers 5 = 5^2) →
  (sum_odd_numbers 7 = 7^2) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_first_seven_odd_numbers_l1177_117701


namespace NUMINAMATH_CALUDE_solution_difference_l1177_117707

theorem solution_difference (r s : ℝ) : 
  (r - 5) * (r + 5) = 25 * r - 125 →
  (s - 5) * (s + 5) = 25 * s - 125 →
  r ≠ s →
  r > s →
  r - s = 15 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1177_117707


namespace NUMINAMATH_CALUDE_point_division_theorem_l1177_117754

/-- Given a line segment AB and a point P on AB such that AP:PB = 3:4,
    prove that P = (4/7)*A + (3/7)*B -/
theorem point_division_theorem (A B P : ℝ × ℝ) :
  (P.1 - A.1) / (B.1 - P.1) = 3 / 4 ∧
  (P.2 - A.2) / (B.2 - P.2) = 3 / 4 →
  P = ((4:ℝ)/7) • A + ((3:ℝ)/7) • B :=
sorry

end NUMINAMATH_CALUDE_point_division_theorem_l1177_117754


namespace NUMINAMATH_CALUDE_simplify_expression_proof_l1177_117710

noncomputable def simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : ℝ :=
  (a / b * (b - 4 * a^6 / b^3)^(1/3) - a^2 * (b / a^6 - 4 / b^3)^(1/3) + 2 / (a * b) * (a^3 * b^4 - 4 * a^9)^(1/3)) / ((b^2 - 2 * a^3)^(1/3) / b^2)

theorem simplify_expression_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  simplify_expression a b ha hb = (a + b) * (b^2 + 2 * a^3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_proof_l1177_117710


namespace NUMINAMATH_CALUDE_fruit_vendor_lemons_sold_l1177_117755

/-- Proves that a fruit vendor who sold 5 dozens of avocados and a total of 90 fruits sold 2.5 dozens of lemons -/
theorem fruit_vendor_lemons_sold (total_fruits : ℕ) (avocado_dozens : ℕ) (lemon_dozens : ℚ) : 
  total_fruits = 90 → avocado_dozens = 5 → lemon_dozens = 2.5 → 
  total_fruits = 12 * avocado_dozens + 12 * lemon_dozens := by
  sorry

#check fruit_vendor_lemons_sold

end NUMINAMATH_CALUDE_fruit_vendor_lemons_sold_l1177_117755


namespace NUMINAMATH_CALUDE_trapezium_side_length_l1177_117776

/-- Proves that given a trapezium with specified dimensions, the length of the unknown parallel side is 28 cm. -/
theorem trapezium_side_length 
  (known_side : ℝ)
  (height : ℝ)
  (area : ℝ)
  (h1 : known_side = 20)
  (h2 : height = 21)
  (h3 : area = 504)
  (h4 : area = (1/2) * (known_side + unknown_side) * height) :
  unknown_side = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l1177_117776


namespace NUMINAMATH_CALUDE_inverse_proportion_quadrants_l1177_117730

/-- An inverse proportion function passing through a specific point -/
structure InverseProportionFunction where
  k : ℝ
  a : ℝ
  point_condition : k / (3 * a) = a

/-- The quadrants where the graph of an inverse proportion function lies -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- The set of quadrants where the graph lies -/
def graph_quadrants (f : InverseProportionFunction) : Set Quadrant :=
  {Quadrant.I, Quadrant.III}

/-- Theorem: The graph of the inverse proportion function lies in Quadrants I and III -/
theorem inverse_proportion_quadrants (f : InverseProportionFunction) :
  graph_quadrants f = {Quadrant.I, Quadrant.III} := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_quadrants_l1177_117730


namespace NUMINAMATH_CALUDE_range_of_a_l1177_117720

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 - x + a

-- Define the range of f as set B
def B (a : ℝ) : Set ℝ := {y : ℝ | ∃ x ∈ A, f a x = y}

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ A, f a x ∈ A) → a = -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1177_117720


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1177_117750

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1177_117750


namespace NUMINAMATH_CALUDE_minAbsNegativeFractions_minAbsNegativeTwo_solveMinAbsEquation_l1177_117761

-- Define the min|a,b| operation for rational numbers
def minAbs (a b : ℚ) : ℚ := min a b

-- Theorem 1
theorem minAbsNegativeFractions : minAbs (-5/2) (-4/3) = -5/2 := by sorry

-- Theorem 2
theorem minAbsNegativeTwo (y : ℚ) (h : y < -2) : minAbs (-2) y = y := by sorry

-- Theorem 3
theorem solveMinAbsEquation : 
  ∃ x : ℚ, (minAbs (-x) 0 = -5 + 2*x) ∧ (x = 5/3) := by sorry

end NUMINAMATH_CALUDE_minAbsNegativeFractions_minAbsNegativeTwo_solveMinAbsEquation_l1177_117761


namespace NUMINAMATH_CALUDE_m_eq_two_iff_parallel_l1177_117725

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The lines l1 and l2 with parameter m -/
def l1 (m : ℝ) : Line := ⟨2, -m, -1⟩
def l2 (m : ℝ) : Line := ⟨m-1, -1, 1⟩

/-- The theorem stating that m=2 is a necessary and sufficient condition for l1 ∥ l2 -/
theorem m_eq_two_iff_parallel :
  ∀ m : ℝ, parallel (l1 m) (l2 m) ↔ m = 2 :=
sorry

end NUMINAMATH_CALUDE_m_eq_two_iff_parallel_l1177_117725


namespace NUMINAMATH_CALUDE_add_1876_minutes_to_6am_l1177_117795

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem add_1876_minutes_to_6am (start : Time) 
  (h_start : start.hours = 6 ∧ start.minutes = 0) :
  addMinutes start 1876 = Time.mk 13 16 sorry sorry :=
sorry

end NUMINAMATH_CALUDE_add_1876_minutes_to_6am_l1177_117795


namespace NUMINAMATH_CALUDE_people_born_in_country_l1177_117778

/-- The number of people who immigrated to the country last year -/
def immigrants : ℕ := 16320

/-- The total number of new people who began living in the country last year -/
def new_residents : ℕ := 106491

/-- The number of people born in the country last year -/
def births : ℕ := new_residents - immigrants

theorem people_born_in_country : births = 90171 := by
  sorry

end NUMINAMATH_CALUDE_people_born_in_country_l1177_117778


namespace NUMINAMATH_CALUDE_cos_equality_proof_l1177_117782

theorem cos_equality_proof (n : ℤ) : 
  n = 43 ∧ -180 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (317 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_proof_l1177_117782


namespace NUMINAMATH_CALUDE_circle_equation_l1177_117772

theorem circle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (4, 6)
  let center_line (x y : ℝ) := x - 2*y - 1 = 0
  ∃ (h k : ℝ), 
    center_line h k ∧ 
    (h - A.1)^2 + (k - A.2)^2 = (h - B.1)^2 + (k - B.2)^2 ∧
    (x - h)^2 + (y - k)^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1177_117772


namespace NUMINAMATH_CALUDE_julia_birth_year_l1177_117706

/-- Given that Wayne is 37 years old in 2021, Peter is 3 years older than Wayne,
    and Julia is 2 years older than Peter, prove that Julia was born in 1979. -/
theorem julia_birth_year (wayne_age : ℕ) (peter_age_diff : ℕ) (julia_age_diff : ℕ) :
  wayne_age = 37 →
  peter_age_diff = 3 →
  julia_age_diff = 2 →
  2021 - wayne_age - peter_age_diff - julia_age_diff = 1979 := by
  sorry

end NUMINAMATH_CALUDE_julia_birth_year_l1177_117706


namespace NUMINAMATH_CALUDE_log_equation_solution_l1177_117741

theorem log_equation_solution (x : ℝ) :
  x > 0 →
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 →
  x = 4 ∨ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1177_117741


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_4_l1177_117752

theorem smallest_lcm_with_gcd_4 :
  ∃ (m n : ℕ),
    1000 ≤ m ∧ m < 10000 ∧
    1000 ≤ n ∧ n < 10000 ∧
    Nat.gcd m n = 4 ∧
    Nat.lcm m n = 252912 ∧
    ∀ (a b : ℕ),
      1000 ≤ a ∧ a < 10000 ∧
      1000 ≤ b ∧ b < 10000 ∧
      Nat.gcd a b = 4 →
      Nat.lcm a b ≥ 252912 :=
sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_4_l1177_117752


namespace NUMINAMATH_CALUDE_trajectory_theorem_l1177_117781

/-- The trajectory of point M -/
def trajectory_M (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 4 * ((x - 4)^2 + y^2)

/-- The trajectory of point P -/
def trajectory_P (x y : ℝ) : Prop :=
  (x - 1/2)^2 + y^2 = 1

/-- The main theorem -/
theorem trajectory_theorem :
  (∀ x y : ℝ, trajectory_M x y ↔ x^2 + y^2 = 4) ∧
  (∀ x y : ℝ, (∃ a b : ℝ, trajectory_M a b ∧ x = (a + 1) / 2 ∧ y = b / 2) → trajectory_P x y) :=
sorry

end NUMINAMATH_CALUDE_trajectory_theorem_l1177_117781


namespace NUMINAMATH_CALUDE_complex_number_property_l1177_117798

theorem complex_number_property (b : ℝ) : 
  let z : ℂ := (2 - b * I) / (1 + 2 * I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_property_l1177_117798


namespace NUMINAMATH_CALUDE_square_roots_problem_l1177_117729

theorem square_roots_problem (n : ℝ) (a : ℝ) : 
  n > 0 ∧ (a - 7)^2 = n ∧ (2*a + 1)^2 = n → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1177_117729


namespace NUMINAMATH_CALUDE_limit_f_at_zero_l1177_117764

open Real
open Filter
open Topology

noncomputable def f (x : ℝ) : ℝ := Real.log ((Real.exp (x^2) - Real.cos x) * Real.cos (1/x) + Real.tan (x + π/3))

theorem limit_f_at_zero : 
  Tendsto f (𝓝 0) (𝓝 ((1/2) * Real.log 3)) := by sorry

end NUMINAMATH_CALUDE_limit_f_at_zero_l1177_117764


namespace NUMINAMATH_CALUDE_broadway_ticket_sales_l1177_117780

theorem broadway_ticket_sales
  (num_adults : ℕ)
  (num_children : ℕ)
  (adult_ticket_price : ℝ)
  (h1 : num_adults = 400)
  (h2 : num_children = 200)
  (h3 : adult_ticket_price = 32)
  (h4 : adult_ticket_price = 2 * (adult_ticket_price / 2)) :
  num_adults * adult_ticket_price + num_children * (adult_ticket_price / 2) = 16000 := by
sorry

end NUMINAMATH_CALUDE_broadway_ticket_sales_l1177_117780


namespace NUMINAMATH_CALUDE_order_combination_savings_l1177_117792

/-- Calculates the discount percentage based on the number of photocopies -/
def discount_percentage (n : ℕ) : ℚ :=
  if n ≤ 50 then 0
  else if n ≤ 100 then 1/10
  else if n ≤ 200 then 1/4
  else 7/20

/-- Calculates the discounted cost for a given number of photocopies -/
def discounted_cost (n : ℕ) : ℚ :=
  let base_cost : ℚ := (n : ℚ) * 2/100
  base_cost * (1 - discount_percentage n)

/-- Theorem: The savings from combining orders is $0.225 -/
theorem order_combination_savings :
  discounted_cost 75 + discounted_cost 105 - discounted_cost 180 = 9/40 := by
  sorry

end NUMINAMATH_CALUDE_order_combination_savings_l1177_117792


namespace NUMINAMATH_CALUDE_total_poultry_count_l1177_117715

def poultry_farm (num_hens num_ducks num_geese : ℕ) 
                 (male_female_ratio : ℚ) 
                 (chicks_per_hen ducklings_per_duck goslings_per_goose : ℕ) : ℕ :=
  let female_hens := (num_hens * 4) / 5
  let female_ducks := (num_ducks * 4) / 5
  let female_geese := (num_geese * 4) / 5
  let total_chicks := female_hens * chicks_per_hen
  let total_ducklings := female_ducks * ducklings_per_duck
  let total_goslings := female_geese * goslings_per_goose
  num_hens + num_ducks + num_geese + total_chicks + total_ducklings + total_goslings

theorem total_poultry_count : 
  poultry_farm 25 10 5 (1/4) 6 8 3 = 236 := by
  sorry

end NUMINAMATH_CALUDE_total_poultry_count_l1177_117715


namespace NUMINAMATH_CALUDE_square_perimeter_contradiction_l1177_117762

theorem square_perimeter_contradiction (perimeter : ℝ) (side_length : ℝ) : 
  perimeter = 4 → side_length = 2 → perimeter ≠ 4 * side_length :=
by
  sorry

#check square_perimeter_contradiction

end NUMINAMATH_CALUDE_square_perimeter_contradiction_l1177_117762


namespace NUMINAMATH_CALUDE_surface_area_of_sliced_solid_l1177_117768

/-- Represents a right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Represents the sliced-off solid CPQR -/
structure SlicedSolid where
  prism : RightPrism

/-- Calculates the surface area of the sliced-off solid CPQR -/
def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the sliced-off solid CPQR -/
theorem surface_area_of_sliced_solid (solid : SlicedSolid) 
  (h1 : solid.prism.height = 18)
  (h2 : solid.prism.base_side = 14) :
  surface_area solid = 63 + (49 * Real.sqrt 3 + Real.sqrt 521) / 4 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_sliced_solid_l1177_117768


namespace NUMINAMATH_CALUDE_percentage_calculation_l1177_117760

theorem percentage_calculation (P : ℝ) : 
  (0.47 * 1442 - P / 100 * 1412) + 65 = 5 → P = 52.24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1177_117760


namespace NUMINAMATH_CALUDE_petya_friends_count_l1177_117732

/-- The number of friends Petya has -/
def num_friends : ℕ := 19

/-- The number of stickers Petya has -/
def total_stickers : ℕ := num_friends * 5 + 8

theorem petya_friends_count :
  (total_stickers = num_friends * 5 + 8) ∧
  (total_stickers = num_friends * 6 - 11) →
  num_friends = 19 := by
sorry

end NUMINAMATH_CALUDE_petya_friends_count_l1177_117732


namespace NUMINAMATH_CALUDE_elevatorProblem_l1177_117716

-- Define the type of elevator move
inductive Move
| Up7 : Move
| Up10 : Move
| Down7 : Move
| Down10 : Move

-- Function to apply a move to a floor number
def applyMove (floor : ℕ) (move : Move) : ℕ :=
  match move with
  | Move.Up7 => floor + 7
  | Move.Up10 => floor + 10
  | Move.Down7 => if floor ≥ 7 then floor - 7 else floor
  | Move.Down10 => if floor ≥ 10 then floor - 10 else floor

-- Function to check if a floor is visited in a sequence of moves
def isVisited (startFloor : ℕ) (moves : List Move) (targetFloor : ℕ) : Prop :=
  targetFloor ∈ List.scanl applyMove startFloor moves

-- Theorem stating the existence of a valid sequence of moves
theorem elevatorProblem : 
  ∃ (moves : List Move), 
    moves.length ≤ 10 ∧ 
    isVisited 1 moves 13 ∧ 
    isVisited 1 moves 16 ∧ 
    isVisited 1 moves 24 :=
by
  sorry


end NUMINAMATH_CALUDE_elevatorProblem_l1177_117716


namespace NUMINAMATH_CALUDE_quadratic_decreasing_iff_a_in_range_l1177_117736

/-- A quadratic function f(x) = ax^2 + 2(a-3)x + 1 is decreasing on [-2, +∞) if and only if a ∈ [-3, 0] -/
theorem quadratic_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, -2 ≤ x ∧ x < y → (a*x^2 + 2*(a-3)*x + 1) > (a*y^2 + 2*(a-3)*y + 1)) ↔ 
  -3 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_iff_a_in_range_l1177_117736


namespace NUMINAMATH_CALUDE_subtraction_of_reciprocals_l1177_117713

theorem subtraction_of_reciprocals (p q : ℝ) : 
  3 / p = 6 → 3 / q = 15 → p - q = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_reciprocals_l1177_117713


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1177_117742

/-- The function f(x) = x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The function g(x) = e^x -/
noncomputable def g (x : ℝ) : ℝ := Real.exp x

/-- The theorem statement -/
theorem function_inequality_implies_a_range :
  ∀ a : ℝ, 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ → 
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) →
  a ∈ Set.Icc (-1) (2 - 2 * Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1177_117742


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l1177_117738

/-- A circle tangent to the y-axis and a line, passing through a specific point -/
structure TangentCircle where
  -- The slope of the line the circle is tangent to
  slope : ℝ
  -- The point the circle passes through
  point : ℝ × ℝ

/-- The radii of a circle satisfying the given conditions -/
def circle_radii (c : TangentCircle) : Set ℝ :=
  {r : ℝ | r = 1 ∨ r = 7/3}

/-- Theorem stating that a circle satisfying the given conditions has radius 1 or 7/3 -/
theorem tangent_circle_radius 
  (c : TangentCircle) 
  (h1 : c.slope = Real.sqrt 3 / 3) 
  (h2 : c.point = (2, Real.sqrt 3)) : 
  ∀ r ∈ circle_radii c, r = 1 ∨ r = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l1177_117738


namespace NUMINAMATH_CALUDE_sticks_in_yard_l1177_117763

theorem sticks_in_yard (picked_up left : ℕ) 
  (h1 : picked_up = 38) 
  (h2 : left = 61) : 
  picked_up + left = 99 := by
  sorry

end NUMINAMATH_CALUDE_sticks_in_yard_l1177_117763


namespace NUMINAMATH_CALUDE_rectangle_width_calculation_l1177_117796

theorem rectangle_width_calculation (big_length : ℝ) (small_area : ℝ) :
  big_length = 40 →
  small_area = 200 →
  ∃ (big_width : ℝ),
    big_width = 20 ∧
    small_area = (big_length / 2) * (big_width / 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_calculation_l1177_117796


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1177_117771

theorem quadratic_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) →
  a < 1 ∧ a ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1177_117771


namespace NUMINAMATH_CALUDE_set_equality_l1177_117727

def M : Set ℝ := {x | ∃ n : ℤ, x = n}
def N : Set ℝ := {x | ∃ n : ℤ, x = n / 2}
def P : Set ℝ := {x | ∃ n : ℤ, x = n + 1 / 2}

theorem set_equality : N = M ∪ P := by sorry

end NUMINAMATH_CALUDE_set_equality_l1177_117727


namespace NUMINAMATH_CALUDE_class_average_score_l1177_117790

theorem class_average_score (total_students : ℕ) 
  (score_95_count score_0_count score_65_count score_80_count : ℕ)
  (remaining_avg : ℚ) :
  total_students = 40 →
  score_95_count = 5 →
  score_0_count = 3 →
  score_65_count = 6 →
  score_80_count = 8 →
  remaining_avg = 45 →
  (2000 : ℚ) ≤ (score_95_count * 95 + score_0_count * 0 + score_65_count * 65 + 
    score_80_count * 80 + (total_students - score_95_count - score_0_count - 
    score_65_count - score_80_count) * remaining_avg) →
  (score_95_count * 95 + score_0_count * 0 + score_65_count * 65 + 
    score_80_count * 80 + (total_students - score_95_count - score_0_count - 
    score_65_count - score_80_count) * remaining_avg) ≤ (2400 : ℚ) →
  (score_95_count * 95 + score_0_count * 0 + score_65_count * 65 + 
    score_80_count * 80 + (total_students - score_95_count - score_0_count - 
    score_65_count - score_80_count) * remaining_avg) / total_students = (57875 : ℚ) / 1000 :=
by sorry

end NUMINAMATH_CALUDE_class_average_score_l1177_117790


namespace NUMINAMATH_CALUDE_x_value_l1177_117744

theorem x_value : ∀ (x y z w : ℤ), 
  x = y + 5 →
  y = z + 10 →
  z = w + 20 →
  w = 80 →
  x = 115 := by
sorry

end NUMINAMATH_CALUDE_x_value_l1177_117744


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_in_range_l1177_117786

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

-- State the theorem
theorem monotonic_f_implies_a_in_range (a : ℝ) :
  (∀ x y, x ≤ y ∧ y ≤ -2 → f a x ≤ f a y) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f a x ≤ f a y) →
  -2 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_in_range_l1177_117786


namespace NUMINAMATH_CALUDE_carrot_weight_problem_l1177_117794

/-- Prove that given 20 carrots weighing 3.64 kg in total, and 4 carrots with an average weight of 190 grams are removed, the average weight of the remaining 16 carrots is 180 grams. -/
theorem carrot_weight_problem (total_weight : ℝ) (removed_avg : ℝ) :
  total_weight = 3.64 →
  removed_avg = 190 →
  (total_weight * 1000 - 4 * removed_avg) / 16 = 180 := by
sorry

end NUMINAMATH_CALUDE_carrot_weight_problem_l1177_117794


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_eleven_l1177_117759

theorem six_digit_divisible_by_eleven (d : Nat) : 
  d < 10 → (67890 * 10 + d) % 11 = 0 ↔ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_eleven_l1177_117759


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1177_117766

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1177_117766


namespace NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_specific_solution_3003m_55555n_l1177_117700

theorem smallest_positive_integer_3003m_55555n :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 1 ∧
  ∀ (k l : ℤ), 3003 * k + 55555 * l > 0 → 3003 * k + 55555 * l ≥ 1 :=
by sorry

theorem specific_solution_3003m_55555n :
  3003 * 37 + 55555 * (-2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_specific_solution_3003m_55555n_l1177_117700


namespace NUMINAMATH_CALUDE_workshop_attendance_prove_workshop_attendance_l1177_117793

theorem workshop_attendance : ℕ → Prop :=
  fun total_scientists =>
    ∃ (wolf_laureates nobel_laureates wolf_and_nobel non_wolf_non_nobel : ℕ),
      wolf_laureates = 31 ∧
      wolf_and_nobel = 14 ∧
      nobel_laureates = 25 ∧
      nobel_laureates - wolf_and_nobel = non_wolf_non_nobel + 3 ∧
      total_scientists = wolf_laureates + (nobel_laureates - wolf_and_nobel) + non_wolf_non_nobel ∧
      total_scientists = 50

theorem prove_workshop_attendance : workshop_attendance 50 := by
  sorry

end NUMINAMATH_CALUDE_workshop_attendance_prove_workshop_attendance_l1177_117793


namespace NUMINAMATH_CALUDE_tree_planting_l1177_117757

theorem tree_planting (road_length : ℕ) (tree_spacing : ℕ) (h1 : road_length = 42) (h2 : tree_spacing = 7) : 
  road_length / tree_spacing + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_l1177_117757


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1177_117749

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| = 81 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1177_117749


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_nineteen_fifteenths_l1177_117748

theorem sqrt_sum_equals_nineteen_fifteenths (w x z : ℝ) 
  (hw : w = 4) (hx : x = 9) (hz : z = 25) : 
  Real.sqrt (w / x) + Real.sqrt (x / z) = 19 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_nineteen_fifteenths_l1177_117748


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1177_117775

theorem contrapositive_equivalence : 
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ↔ 
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) := by
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1177_117775


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1177_117789

theorem polynomial_factorization (x : ℝ) : x^2 - x = x * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1177_117789


namespace NUMINAMATH_CALUDE_interior_perimeter_is_14_l1177_117784

/-- Represents a rectangular picture frame -/
structure PictureFrame where
  outerWidth : ℝ
  outerHeight : ℝ
  frameWidth : ℝ

/-- Calculates the area of just the frame -/
def frameArea (frame : PictureFrame) : ℝ :=
  frame.outerWidth * frame.outerHeight - (frame.outerWidth - 2 * frame.frameWidth) * (frame.outerHeight - 2 * frame.frameWidth)

/-- Calculates the sum of the lengths of the four interior edges -/
def interiorPerimeter (frame : PictureFrame) : ℝ :=
  2 * (frame.outerWidth - 2 * frame.frameWidth) + 2 * (frame.outerHeight - 2 * frame.frameWidth)

/-- Theorem: Given the conditions, the sum of interior edges is 14 inches -/
theorem interior_perimeter_is_14 (frame : PictureFrame) 
  (h1 : frame.frameWidth = 1)
  (h2 : frameArea frame = 18)
  (h3 : frame.outerWidth = 5) :
  interiorPerimeter frame = 14 := by
  sorry

end NUMINAMATH_CALUDE_interior_perimeter_is_14_l1177_117784


namespace NUMINAMATH_CALUDE_basketball_free_throws_l1177_117739

theorem basketball_free_throws (total_players : Nat) (goalkeepers : Nat) : 
  total_players = 18 → goalkeepers = 2 → 
  (total_players - goalkeepers) * goalkeepers = 34 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l1177_117739


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l1177_117743

/-- The area of a right-angled triangle inscribed in a circle of radius 100, 
    with acute angles α and β satisfying tan α = 4 tan β, is equal to 8000. -/
theorem inscribed_triangle_area (α β : Real) (h1 : α > 0) (h2 : β > 0) (h3 : α + β = Real.pi / 2) 
  (h4 : Real.tan α = 4 * Real.tan β) : 
  let r : Real := 100
  let area := r^2 * Real.sin α * Real.sin β
  area = 8000 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l1177_117743


namespace NUMINAMATH_CALUDE_prob_same_color_specific_l1177_117705

/-- Probability of drawing two marbles of the same color -/
def prob_same_color (red white blue green : ℕ) : ℚ :=
  let total := red + white + blue + green
  let prob_red := (red * (red - 1)) / (total * (total - 1))
  let prob_white := (white * (white - 1)) / (total * (total - 1))
  let prob_blue := (blue * (blue - 1)) / (total * (total - 1))
  let prob_green := (green * (green - 1)) / (total * (total - 1))
  prob_red + prob_white + prob_blue + prob_green

theorem prob_same_color_specific : prob_same_color 5 6 7 3 = 7 / 30 := by
  sorry

#eval prob_same_color 5 6 7 3

end NUMINAMATH_CALUDE_prob_same_color_specific_l1177_117705


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l1177_117728

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  ∀ a : ℝ,
  (∀ a : ℝ, a = 1 → |a| = 1) ∧
  (∃ a : ℝ, |a| = 1 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l1177_117728


namespace NUMINAMATH_CALUDE_shaded_percentage_of_grid_l1177_117788

theorem shaded_percentage_of_grid (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 25 →
  shaded_squares = 13 →
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 52 := by
  sorry

end NUMINAMATH_CALUDE_shaded_percentage_of_grid_l1177_117788


namespace NUMINAMATH_CALUDE_total_palm_trees_l1177_117767

theorem total_palm_trees (forest_trees : ℕ) (desert_reduction : ℚ) (river_trees : ℕ)
  (h1 : forest_trees = 5000)
  (h2 : desert_reduction = 3 / 5)
  (h3 : river_trees = 1200) :
  forest_trees + (forest_trees - desert_reduction * forest_trees) + river_trees = 8200 :=
by sorry

end NUMINAMATH_CALUDE_total_palm_trees_l1177_117767


namespace NUMINAMATH_CALUDE_negation_of_forall_square_ge_self_l1177_117740

theorem negation_of_forall_square_ge_self :
  (¬ ∀ x : ℕ, x^2 ≥ x) ↔ (∃ x : ℕ, x^2 < x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_square_ge_self_l1177_117740


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l1177_117747

theorem factorial_equation_solution : ∃ k : ℕ, (4 * 3 * 2 * 1) * (2 * 1) = 2 * k * (3 * 2 * 1) ∧ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l1177_117747
