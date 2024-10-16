import Mathlib

namespace NUMINAMATH_CALUDE_prob_king_or_queen_l3025_302546

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (kings : ℕ)
  (queens : ℕ)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52
  , ranks := 13
  , suits := 4
  , kings := 4
  , queens := 4 }

/-- The probability of drawing a King or Queen from a standard deck -/
theorem prob_king_or_queen (d : Deck) (h : d = standard_deck) :
  (d.kings + d.queens : ℚ) / d.total_cards = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_or_queen_l3025_302546


namespace NUMINAMATH_CALUDE_power_of_point_theorem_l3025_302561

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a plane -/
def Point := ℝ × ℝ

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Determine if a point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop := sorry

theorem power_of_point_theorem 
  (c : Circle) (A B C D E : Point) 
  (hB : onCircle B c) (hC : onCircle C c) (hD : onCircle D c) (hE : onCircle E c)
  (hAB : distance A B = 7)
  (hBC : distance B C = 7)
  (hAD : distance A D = 10) :
  distance D E = 0.2 := by sorry

end NUMINAMATH_CALUDE_power_of_point_theorem_l3025_302561


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3025_302584

theorem arithmetic_calculation : 2 + 3 * 4^2 - 5 + 6 = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3025_302584


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3025_302556

theorem sqrt_sum_inequality (a b c d : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_a : a ≤ 1)
  (h_ab : a + b ≤ 5)
  (h_abc : a + b + c ≤ 14)
  (h_abcd : a + b + c + d ≤ 30) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3025_302556


namespace NUMINAMATH_CALUDE_intersection_equals_S_l3025_302582

def S : Set ℝ := {y | ∃ x : ℝ, y = 3 * x}
def T : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

theorem intersection_equals_S : S ∩ T = S := by sorry

end NUMINAMATH_CALUDE_intersection_equals_S_l3025_302582


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3025_302514

/-- A random variable with normal distribution -/
def normal_dist (μ σ : ℝ) : Type := ℝ

/-- Probability measure for the normal distribution -/
noncomputable def P (ξ : normal_dist (-1) σ) (s : Set ℝ) : ℝ := sorry

/-- The statement of the problem -/
theorem normal_distribution_probability (σ : ℝ) (ξ : normal_dist (-1) σ) 
  (h : P ξ {x | -3 ≤ x ∧ x ≤ -1} = 0.4) : 
  P ξ {x | x ≥ 1} = 0.1 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3025_302514


namespace NUMINAMATH_CALUDE_sibling_pairs_count_l3025_302562

theorem sibling_pairs_count 
  (business_students : ℕ) 
  (law_students : ℕ) 
  (sibling_pair_probability : ℝ) 
  (h1 : business_students = 500) 
  (h2 : law_students = 800) 
  (h3 : sibling_pair_probability = 7.500000000000001e-05) : 
  ℕ := 
by
  sorry

#check sibling_pairs_count

end NUMINAMATH_CALUDE_sibling_pairs_count_l3025_302562


namespace NUMINAMATH_CALUDE_nail_size_fraction_l3025_302526

theorem nail_size_fraction (x : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 1) 
  (h2 : x + 0.5 = 0.75) : 
  x = 0.25 := by
sorry

end NUMINAMATH_CALUDE_nail_size_fraction_l3025_302526


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_diff_odd_l3025_302597

theorem consecutive_squares_sum_diff_odd (n : ℕ) : 
  Odd (n^2 + (n+1)^2) ∧ Odd ((n+1)^2 - n^2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_diff_odd_l3025_302597


namespace NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l3025_302510

theorem sqrt_198_between_14_and_15 : 14 < Real.sqrt 198 ∧ Real.sqrt 198 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l3025_302510


namespace NUMINAMATH_CALUDE_pilot_weeks_flown_l3025_302522

def miles_tuesday : ℕ := 1134
def miles_thursday : ℕ := 1475
def total_miles : ℕ := 7827

theorem pilot_weeks_flown : 
  (total_miles : ℚ) / (miles_tuesday + miles_thursday : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_pilot_weeks_flown_l3025_302522


namespace NUMINAMATH_CALUDE_probability_theorem_l3025_302542

def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def green_marbles : ℕ := 5
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles
def marbles_selected : ℕ := 4

def probability_one_red_two_blue_one_green : ℚ :=
  (red_marbles.choose 1 * blue_marbles.choose 2 * green_marbles.choose 1) /
  (total_marbles.choose marbles_selected)

theorem probability_theorem :
  probability_one_red_two_blue_one_green = 411 / 4200 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3025_302542


namespace NUMINAMATH_CALUDE_witch_clock_theorem_l3025_302518

def clock_cycle (t : ℕ) : ℕ :=
  (5 * (t / 8 + 1) - 3 * (t / 8)) % 60

theorem witch_clock_theorem (t : ℕ) (h : t = 2022) :
  clock_cycle t = 28 := by
  sorry

end NUMINAMATH_CALUDE_witch_clock_theorem_l3025_302518


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3025_302573

/-- Given a square with one side on the line y = 7 and endpoints on the parabola y = x^2 + 4x + 3,
    prove that its area is 32. -/
theorem square_area_on_parabola : 
  ∀ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) →
  (x₂^2 + 4*x₂ + 3 = 7) →
  x₁ ≠ x₂ →
  (x₂ - x₁)^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3025_302573


namespace NUMINAMATH_CALUDE_three_points_in_circle_l3025_302544

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane. -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point is inside or on the boundary of a circle. -/
def isInCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- The theorem to be proved. -/
theorem three_points_in_circle (points : Finset Point) 
    (h1 : points.card = 51)
    (h2 : ∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1) :
  ∃ (c : Circle), c.radius = 1/7 ∧ (∃ (p1 p2 p3 : Point), 
    p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    isInCircle p1 c ∧ isInCircle p2 c ∧ isInCircle p3 c) :=
by
  sorry


end NUMINAMATH_CALUDE_three_points_in_circle_l3025_302544


namespace NUMINAMATH_CALUDE_bill_selling_price_l3025_302509

theorem bill_selling_price (original_purchase_price : ℝ) : 
  let original_selling_price := 1.1 * original_purchase_price
  let new_selling_price := 1.17 * original_purchase_price
  new_selling_price = original_selling_price + 28 →
  original_selling_price = 440 := by
sorry

end NUMINAMATH_CALUDE_bill_selling_price_l3025_302509


namespace NUMINAMATH_CALUDE_subtraction_result_l3025_302535

theorem subtraction_result (x : ℝ) (h : x / 10 = 6) : x - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l3025_302535


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3025_302577

/-- The number of ways to distribute n distinguishable objects into k boxes,
    where m boxes are distinguishable and (k-m) boxes are indistinguishable. -/
def distribution_count (n k m : ℕ) : ℕ :=
  k^n - (k-m)^n + ((k-m)^n / 2)

/-- The number of ways to place 5 distinguishable balls into 3 boxes,
    where one box is distinguishable (red) and the other two are indistinguishable. -/
theorem five_balls_three_boxes :
  distribution_count 5 3 1 = 227 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3025_302577


namespace NUMINAMATH_CALUDE_union_when_a_is_two_intersection_empty_iff_l3025_302565

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 3 ∧ a > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- Theorem 1: When a = 2, A ∪ B = {x | -2 < x < 7}
theorem union_when_a_is_two : 
  A 2 ∪ B = {x : ℝ | -2 < x ∧ x < 7} := by sorry

-- Theorem 2: A ∩ B = ∅ if and only if a ≥ 5
theorem intersection_empty_iff : 
  ∀ a : ℝ, A a ∩ B = ∅ ↔ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_union_when_a_is_two_intersection_empty_iff_l3025_302565


namespace NUMINAMATH_CALUDE_divisibility_by_x_squared_plus_x_plus_one_l3025_302595

theorem divisibility_by_x_squared_plus_x_plus_one (n : ℕ) :
  ∃ q : Polynomial ℤ, (X + 1 : Polynomial ℤ)^(2*n + 1) + X^(n + 2) = (X^2 + X + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_x_squared_plus_x_plus_one_l3025_302595


namespace NUMINAMATH_CALUDE_f_properties_l3025_302520

-- Define the function f(x) = x^2 + ln|x|
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log (abs x)

-- State the theorem
theorem f_properties :
  -- f is defined for all non-zero real numbers
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) ∧
  -- f is an even function
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  -- f is increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3025_302520


namespace NUMINAMATH_CALUDE_cost_difference_between_cars_l3025_302537

/-- Represents a car with its associated costs and characteristics -/
structure Car where
  initialCost : ℕ
  fuelConsumption : ℕ
  annualInsurance : ℕ
  annualMaintenance : ℕ
  resaleValue : ℕ

/-- Calculates the total cost of owning a car for 5 years -/
def totalCost (c : Car) (annualDistance : ℕ) (fuelPrice : ℕ) (years : ℕ) : ℕ :=
  c.initialCost +
  (c.fuelConsumption * annualDistance / 100 * fuelPrice * years) +
  (c.annualInsurance * years) +
  (c.annualMaintenance * years) -
  c.resaleValue

/-- Theorem stating the difference in total cost between two cars -/
theorem cost_difference_between_cars :
  let carA : Car := {
    initialCost := 900000,
    fuelConsumption := 9,
    annualInsurance := 35000,
    annualMaintenance := 25000,
    resaleValue := 500000
  }
  let carB : Car := {
    initialCost := 600000,
    fuelConsumption := 10,
    annualInsurance := 32000,
    annualMaintenance := 20000,
    resaleValue := 350000
  }
  let annualDistance := 15000
  let fuelPrice := 40
  let years := 5

  totalCost carA annualDistance fuelPrice years -
  totalCost carB annualDistance fuelPrice years = 160000 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_between_cars_l3025_302537


namespace NUMINAMATH_CALUDE_consecutive_triangular_not_square_infinitely_many_square_products_l3025_302587

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Statement: The product of two consecutive triangular numbers is not a perfect square -/
theorem consecutive_triangular_not_square (n : ℕ) (h : n > 1) :
  ¬ ∃ m : ℕ, triangular_number (n - 1) * triangular_number n = m^2 := by sorry

/-- Statement: For each triangular number, there exist infinitely many larger triangular numbers
    such that their product is a perfect square -/
theorem infinitely_many_square_products (n : ℕ) :
  ∃ f : ℕ → ℕ, Monotone f ∧ (∀ k : ℕ, f k > n) ∧
  (∀ k : ℕ, ∃ m : ℕ, triangular_number n * triangular_number (f k) = m^2) := by sorry

end NUMINAMATH_CALUDE_consecutive_triangular_not_square_infinitely_many_square_products_l3025_302587


namespace NUMINAMATH_CALUDE_infinite_essentially_different_solutions_l3025_302578

/-- Two integer triples are essentially different if they are not scalar multiples of each other -/
def EssentiallyDifferent (a b c a₁ b₁ c₁ : ℤ) : Prop :=
  ∀ r : ℚ, ¬(a₁ = r * a ∧ b₁ = r * b ∧ c₁ = r * c)

/-- The set of solutions for the equation x^2 = y^2 + k·z^2 -/
def SolutionSet (k : ℕ) : Set (ℤ × ℤ × ℤ) :=
  {(x, y, z) | x^2 = y^2 + k * z^2}

/-- The theorem stating that there are infinitely many essentially different solutions -/
theorem infinite_essentially_different_solutions (k : ℕ) :
  ∃ S : Set (ℤ × ℤ × ℤ),
    (∀ (x y z : ℤ), (x, y, z) ∈ S → (x, y, z) ∈ SolutionSet k) ∧
    (∀ (x y z x₁ y₁ z₁ : ℤ), (x, y, z) ∈ S → (x₁, y₁, z₁) ∈ S → (x, y, z) ≠ (x₁, y₁, z₁) →
      EssentiallyDifferent x y z x₁ y₁ z₁) ∧
    Set.Infinite S :=
  sorry

end NUMINAMATH_CALUDE_infinite_essentially_different_solutions_l3025_302578


namespace NUMINAMATH_CALUDE_dice_roll_probability_l3025_302586

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of rolling a 2 on the first die -/
def prob_first_die_2 : ℚ := 1 / num_sides

/-- The probability of rolling a 5 or 6 on the second die -/
def prob_second_die_5_or_6 : ℚ := 2 / num_sides

/-- The probability of the combined event -/
def prob_combined : ℚ := prob_first_die_2 * prob_second_die_5_or_6

theorem dice_roll_probability : prob_combined = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l3025_302586


namespace NUMINAMATH_CALUDE_sandys_comic_books_l3025_302534

/-- Sandy's comic book problem -/
theorem sandys_comic_books (initial : ℕ) : 
  (initial / 2 : ℚ) + 6 = 13 → initial = 14 := by
  sorry

end NUMINAMATH_CALUDE_sandys_comic_books_l3025_302534


namespace NUMINAMATH_CALUDE_present_worth_calculation_present_worth_approximation_l3025_302598

/-- Calculates the present worth of an investment given specific interest rates and banker's gain --/
theorem present_worth_calculation (banker_gain : ℝ) : ∃ P : ℝ,
  P * (1.05 * 1.1025 * 1.1255 - 1) = banker_gain :=
by
  sorry

/-- Verifies that the calculated present worth is approximately 114.94 --/
theorem present_worth_approximation (P : ℝ) 
  (h : P * (1.05 * 1.1025 * 1.1255 - 1) = 36) : 
  114.9 < P ∧ P < 115 :=
by
  sorry

end NUMINAMATH_CALUDE_present_worth_calculation_present_worth_approximation_l3025_302598


namespace NUMINAMATH_CALUDE_parabola_sum_l3025_302580

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_sum (p : Parabola) 
  (vertex_condition : p.y_at 1 = 3 ∧ (- p.b / (2 * p.a)) = 1)
  (point_condition : p.y_at 0 = 2) :
  p.a + p.b + p.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l3025_302580


namespace NUMINAMATH_CALUDE_percentage_problem_l3025_302505

/-- The percentage that, when applied to 12356, results in 6.178 is 0.05% -/
theorem percentage_problem : ∃ p : ℝ, p * 12356 = 6.178 ∧ p = 0.0005 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3025_302505


namespace NUMINAMATH_CALUDE_equation_solution_l3025_302517

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 1/3 ∧ 
  (∀ x : ℝ, (x - 1)^2 + 2*x*(x - 1) = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3025_302517


namespace NUMINAMATH_CALUDE_function_property_l3025_302549

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y^3 * f x = x^3 * f y

theorem function_property (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 3 ≠ 0) :
  (f 20 - f 2) / f 3 = 296 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3025_302549


namespace NUMINAMATH_CALUDE_cubic_root_sum_log_l3025_302531

theorem cubic_root_sum_log (a b : ℝ) : 
  (∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧ 
   r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
   16 * r^3 + 7 * a * r^2 + 6 * b * r + 2 * a = 0 ∧
   16 * s^3 + 7 * a * s^2 + 6 * b * s + 2 * a = 0 ∧
   16 * t^3 + 7 * a * t^2 + 6 * b * t + 2 * a = 0 ∧
   Real.log r / Real.log 4 + Real.log s / Real.log 4 + Real.log t / Real.log 4 = 3) →
  a = -512 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_log_l3025_302531


namespace NUMINAMATH_CALUDE_purely_imaginary_x_eq_neg_one_l3025_302591

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given a real number x, define z as (x^2 - 1) + (x - 1)i. -/
def z (x : ℝ) : ℂ :=
  ⟨x^2 - 1, x - 1⟩

theorem purely_imaginary_x_eq_neg_one :
  ∀ x : ℝ, IsPurelyImaginary (z x) → x = -1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_x_eq_neg_one_l3025_302591


namespace NUMINAMATH_CALUDE_curve_intersection_minimum_a_l3025_302554

theorem curve_intersection_minimum_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ a * x^2 = Real.exp x) →
  a ≥ Real.exp 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_curve_intersection_minimum_a_l3025_302554


namespace NUMINAMATH_CALUDE_molly_gift_cost_l3025_302571

/-- Represents the cost structure and family composition for Molly's gift-sending scenario -/
structure GiftSendingScenario where
  cost_per_package : ℕ
  num_parents : ℕ
  num_brothers : ℕ
  num_children_per_brother : ℕ

/-- Calculates the total number of relatives Molly needs to send gifts to -/
def total_relatives (scenario : GiftSendingScenario) : ℕ :=
  scenario.num_parents + 
  scenario.num_brothers + 
  scenario.num_brothers + -- for sisters-in-law
  scenario.num_brothers * scenario.num_children_per_brother

/-- Calculates the total cost of sending gifts to all relatives -/
def total_cost (scenario : GiftSendingScenario) : ℕ :=
  scenario.cost_per_package * total_relatives scenario

/-- Theorem stating that Molly's total cost for sending gifts is $70 -/
theorem molly_gift_cost : 
  ∀ (scenario : GiftSendingScenario), 
  scenario.cost_per_package = 5 ∧ 
  scenario.num_parents = 2 ∧ 
  scenario.num_brothers = 3 ∧ 
  scenario.num_children_per_brother = 2 → 
  total_cost scenario = 70 := by
  sorry

end NUMINAMATH_CALUDE_molly_gift_cost_l3025_302571


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3025_302569

/-- Given a quadratic equation x^2 - 2mx + 4 = 0 where m is a real number,
    if both of its real roots are greater than 1, then m is in the range [2, 5/2). -/
theorem quadratic_roots_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*m*x + 4 = 0 → x > 1) → 
  m ∈ Set.Icc 2 (5/2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3025_302569


namespace NUMINAMATH_CALUDE_complex_equality_l3025_302536

theorem complex_equality (x y z : ℝ) (α β γ : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hα : Complex.abs α = 1) (hβ : Complex.abs β = 1) (hγ : Complex.abs γ = 1)
  (h_sum : x + y + z = 0) (h_complex_sum : α * x + β * y + γ * z = 0) :
  α = β ∧ β = γ := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3025_302536


namespace NUMINAMATH_CALUDE_successive_integers_product_l3025_302530

theorem successive_integers_product (n : ℤ) : n * (n + 1) = 9506 → n = 97 := by
  sorry

end NUMINAMATH_CALUDE_successive_integers_product_l3025_302530


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l3025_302596

theorem sum_of_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ 1 + i + i^2 + i^3 = i := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l3025_302596


namespace NUMINAMATH_CALUDE_sequence_term_equals_three_l3025_302567

def a (n : ℝ) : ℝ := n^2 - 8*n + 15

theorem sequence_term_equals_three :
  ∃! (s : Set ℝ), s = {n : ℝ | a n = 3} ∧ s = {2, 6} :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_equals_three_l3025_302567


namespace NUMINAMATH_CALUDE_selection_theorem_l3025_302568

/-- The probability of a student being selected for a visiting group -/
def selection_probability (total : ℕ) (eliminated : ℕ) (group_size : ℕ) : ℚ :=
  group_size / (total - eliminated)

/-- The properties of the selection process -/
theorem selection_theorem (total : ℕ) (eliminated : ℕ) (group_size : ℕ) 
  (h1 : total = 2004) 
  (h2 : eliminated = 4) 
  (h3 : group_size = 50) :
  selection_probability total eliminated group_size = 1 / 40 := by
  sorry

#eval selection_probability 2004 4 50

end NUMINAMATH_CALUDE_selection_theorem_l3025_302568


namespace NUMINAMATH_CALUDE_fraction_power_equality_l3025_302559

theorem fraction_power_equality : (123456 / 41152)^5 = 243 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l3025_302559


namespace NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l3025_302581

def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_polynomial_inequality 
  (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (x y : ℝ) : 
  (quadratic_polynomial a b c (x * y))^2 ≤ 
  (quadratic_polynomial a b c (x^2)) * (quadratic_polynomial a b c (y^2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l3025_302581


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3025_302513

theorem trigonometric_simplification :
  let numerator := Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + Real.sin (45 * π / 180) + 
                   Real.sin (60 * π / 180) + Real.sin (75 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)
  numerator / denominator = 
    (Real.sqrt 2 * (4 * Real.cos (22.5 * π / 180) * Real.cos (7.5 * π / 180) + 1)) / 
    (2 * Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3025_302513


namespace NUMINAMATH_CALUDE_amy_garden_problem_l3025_302511

/-- Amy's gardening problem -/
theorem amy_garden_problem (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ)
  (h1 : total_seeds = 101)
  (h2 : small_gardens = 9)
  (h3 : seeds_per_small_garden = 6) :
  total_seeds - (small_gardens * seeds_per_small_garden) = 47 := by
  sorry

end NUMINAMATH_CALUDE_amy_garden_problem_l3025_302511


namespace NUMINAMATH_CALUDE_antoinette_weight_l3025_302589

theorem antoinette_weight (rupert_weight : ℝ) : 
  let antoinette_weight := 2 * rupert_weight - 7
  (antoinette_weight + rupert_weight = 98) → antoinette_weight = 63 := by
sorry

end NUMINAMATH_CALUDE_antoinette_weight_l3025_302589


namespace NUMINAMATH_CALUDE_upper_bound_of_expression_l3025_302555

theorem upper_bound_of_expression (n : ℤ) (U : ℤ) : 
  (∃ (S : Finset ℤ), 
    (∀ m ∈ S, (4 * m + 7 > 1 ∧ 4 * m + 7 < U)) ∧ 
    S.card = 15 ∧
    (∀ m : ℤ, 4 * m + 7 > 1 ∧ 4 * m + 7 < U → m ∈ S)) →
  U ≥ 64 :=
sorry

end NUMINAMATH_CALUDE_upper_bound_of_expression_l3025_302555


namespace NUMINAMATH_CALUDE_marble_probability_theorem_l3025_302503

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  sum_constraint : red + blue = total

/-- Represents the probability of drawing two marbles of the same color -/
def drawProbability (box1 box2 : MarbleBox) (color : ℕ → ℕ) : ℚ :=
  (color box1.red / box1.total) * (color box2.red / box2.total)

theorem marble_probability_theorem (box1 box2 : MarbleBox) :
  box1.total + box2.total = 34 →
  drawProbability box1 box2 (fun x => x) = 19/34 →
  drawProbability box1 box2 (fun x => box1.total - x) = 64/289 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_theorem_l3025_302503


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l3025_302516

/-- Calculates the time for a train to pass a platform given its length, time to cross a tree, and platform length -/
theorem train_platform_passing_time
  (train_length : ℝ)
  (tree_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 2000)
  (h2 : tree_crossing_time = 80)
  (h3 : platform_length = 1200) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 128 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l3025_302516


namespace NUMINAMATH_CALUDE_sequence_a_10_l3025_302525

/-- A sequence satisfying the given properties -/
def Sequence (a : ℕ+ → ℤ) : Prop :=
  (∀ p q : ℕ+, a (p + q) = a p + a q) ∧ (a 2 = -6)

/-- The theorem to be proved -/
theorem sequence_a_10 (a : ℕ+ → ℤ) (h : Sequence a) : a 10 = -30 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_10_l3025_302525


namespace NUMINAMATH_CALUDE_sin_squared_sum_less_than_one_l3025_302533

theorem sin_squared_sum_less_than_one (x y z : ℝ) 
  (h1 : Real.tan x + Real.tan y + Real.tan z = 2)
  (h2 : 0 < x ∧ x < Real.pi / 2)
  (h3 : 0 < y ∧ y < Real.pi / 2)
  (h4 : 0 < z ∧ z < Real.pi / 2) :
  Real.sin x ^ 2 + Real.sin y ^ 2 + Real.sin z ^ 2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_less_than_one_l3025_302533


namespace NUMINAMATH_CALUDE_solution_system_trigonometric_equations_l3025_302524

theorem solution_system_trigonometric_equations :
  ∀ x y : ℝ,
  (Real.sin x)^2 = Real.sin y ∧ (Real.cos x)^4 = Real.cos y →
  (∃ l m : ℤ, x = l * Real.pi ∧ y = 2 * m * Real.pi) ∨
  (∃ l m : ℤ, x = l * Real.pi + Real.pi / 2 ∧ y = 2 * m * Real.pi + Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_trigonometric_equations_l3025_302524


namespace NUMINAMATH_CALUDE_max_inscribed_rectangle_area_l3025_302507

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-coordinate for a given x on the parabola -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Theorem: Maximum inscribed rectangle area in a parabola -/
theorem max_inscribed_rectangle_area
  (p : Parabola)
  (vertex_x : p.y_at 3 = -5)
  (point_on_parabola : p.y_at 5 = 15) :
  ∃ (area : ℝ), area = 10 ∧ 
  ∀ (rect_area : ℝ), 
    (∃ (x1 x2 : ℝ), 
      x1 < x2 ∧ 
      p.y_at x1 = 0 ∧ 
      p.y_at x2 = 0 ∧ 
      rect_area = (x2 - x1) * min (p.y_at ((x1 + x2) / 2)) 0) →
    rect_area ≤ area :=
by sorry

end NUMINAMATH_CALUDE_max_inscribed_rectangle_area_l3025_302507


namespace NUMINAMATH_CALUDE_bird_flight_problem_l3025_302523

theorem bird_flight_problem (h₁ h₂ w : ℝ) (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (w_pos : w > 0)
  (h₁_val : h₁ = 20) (h₂_val : h₂ = 30) (w_val : w = 50) :
  ∃ (d x : ℝ),
    d = 10 * Real.sqrt 13 ∧
    x = 20 ∧
    d = Real.sqrt (x^2 + h₂^2) ∧
    d = Real.sqrt ((w - x)^2 + h₁^2) := by
  sorry

#check bird_flight_problem

end NUMINAMATH_CALUDE_bird_flight_problem_l3025_302523


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3025_302575

theorem polynomial_factorization (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3025_302575


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3025_302563

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : x + |y| - y = 12) : 
  x + y = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3025_302563


namespace NUMINAMATH_CALUDE_inequality_proof_l3025_302572

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3025_302572


namespace NUMINAMATH_CALUDE_all_cells_colored_l3025_302504

/-- Represents a 6x6 grid where cells can be colored -/
structure Grid :=
  (colored : Fin 6 → Fin 6 → Bool)

/-- Returns the number of colored cells in a 2x2 square starting at (i, j) -/
def count_2x2 (g : Grid) (i j : Fin 4) : Nat :=
  (g.colored i j).toNat + (g.colored i (j + 1)).toNat +
  (g.colored (i + 1) j).toNat + (g.colored (i + 1) (j + 1)).toNat

/-- Returns the number of colored cells in a 1x3 stripe starting at (i, j) -/
def count_1x3 (g : Grid) (i : Fin 6) (j : Fin 4) : Nat :=
  (g.colored i j).toNat + (g.colored i (j + 1)).toNat + (g.colored i (j + 2)).toNat

/-- The main theorem -/
theorem all_cells_colored (g : Grid) 
  (h1 : ∀ i j : Fin 4, count_2x2 g i j = count_2x2 g 0 0)
  (h2 : ∀ i : Fin 6, ∀ j : Fin 4, count_1x3 g i j = count_1x3 g 0 0) :
  ∀ i j : Fin 6, g.colored i j = true := by
  sorry

end NUMINAMATH_CALUDE_all_cells_colored_l3025_302504


namespace NUMINAMATH_CALUDE_physics_marks_l3025_302583

def marks_english : ℕ := 81
def marks_mathematics : ℕ := 65
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def average_marks : ℕ := 76
def total_subjects : ℕ := 5

theorem physics_marks :
  let total_marks := average_marks * total_subjects
  let known_marks := marks_english + marks_mathematics + marks_chemistry + marks_biology
  total_marks - known_marks = 82 := by sorry

end NUMINAMATH_CALUDE_physics_marks_l3025_302583


namespace NUMINAMATH_CALUDE_derivative_problems_l3025_302551

open Real

theorem derivative_problems :
  (∀ x : ℝ, x > 0 → deriv (λ x => x * log x) x = log x + 1) ∧
  (∀ x : ℝ, x ≠ 0 → deriv (λ x => sin x / x) x = (x * cos x - sin x) / x^2) :=
by sorry

end NUMINAMATH_CALUDE_derivative_problems_l3025_302551


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l3025_302550

/-- Given a drawer with initial pencils and some taken out, calculate the remaining pencils -/
def remaining_pencils (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Theorem: If there were 9 pencils initially and 4 were taken out, 5 pencils remain -/
theorem pencils_in_drawer : remaining_pencils 9 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l3025_302550


namespace NUMINAMATH_CALUDE_max_distance_complex_l3025_302532

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (max_dist : ℝ), max_dist = 729 + 162 * Real.sqrt 13 ∧
  ∀ (w : ℂ), Complex.abs w = 3 →
    Complex.abs ((2 + 3*Complex.I)*(w^4) - w^6) ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_l3025_302532


namespace NUMINAMATH_CALUDE_unique_sum_of_three_squares_l3025_302566

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def sum_of_three_squares (a b c : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧ a + b + c = 100

def distinct_combinations (a b c : ℕ) : Prop :=
  sum_of_three_squares a b c ∧ 
  (a ≤ b ∧ b ≤ c)

theorem unique_sum_of_three_squares : 
  ∃! (abc : ℕ × ℕ × ℕ), distinct_combinations abc.1 abc.2.1 abc.2.2 :=
sorry

end NUMINAMATH_CALUDE_unique_sum_of_three_squares_l3025_302566


namespace NUMINAMATH_CALUDE_probability_all_truth_l3025_302529

theorem probability_all_truth (pA pB pC : ℝ) 
  (hA : 0 ≤ pA ∧ pA ≤ 1) 
  (hB : 0 ≤ pB ∧ pB ≤ 1) 
  (hC : 0 ≤ pC ∧ pC ≤ 1) 
  (hpA : pA = 0.8) 
  (hpB : pB = 0.6) 
  (hpC : pC = 0.75) : 
  pA * pB * pC = 0.27 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_truth_l3025_302529


namespace NUMINAMATH_CALUDE_product_closest_to_315_l3025_302599

def product : ℝ := 3.57 * 9.052 * (6.18 + 3.821)

def options : List ℝ := [200, 300, 315, 400, 500]

theorem product_closest_to_315 :
  ∀ x ∈ options, |product - 315| ≤ |product - x| :=
sorry

end NUMINAMATH_CALUDE_product_closest_to_315_l3025_302599


namespace NUMINAMATH_CALUDE_license_plate_count_l3025_302506

/-- The number of consonants available for the license plate. -/
def num_consonants : ℕ := 20

/-- The number of vowels available for the license plate. -/
def num_vowels : ℕ := 6

/-- The number of digits available for the license plate. -/
def num_digits : ℕ := 10

/-- The number of special symbols available for the license plate. -/
def num_symbols : ℕ := 3

/-- The total number of possible license plates. -/
def total_license_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits * num_symbols

/-- Theorem stating that the total number of license plates is 72,000. -/
theorem license_plate_count : total_license_plates = 72000 := by
  sorry


end NUMINAMATH_CALUDE_license_plate_count_l3025_302506


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l3025_302574

theorem point_in_first_quadrant (a : ℕ+) :
  (4 > 0 ∧ 2 - a.val > 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l3025_302574


namespace NUMINAMATH_CALUDE_comparison_theorem_l3025_302543

theorem comparison_theorem :
  (3 * 10^5 < 2 * 10^6) ∧ (-2 - 1/3 > -3 - 1/2) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l3025_302543


namespace NUMINAMATH_CALUDE_max_value_theorem_l3025_302547

theorem max_value_theorem (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 4) :
  (6 * a + 3 * b + 10 * c) ≤ Real.sqrt 41 ∧
  ∃ a₀ b₀ c₀ : ℝ, 9 * a₀^2 + 4 * b₀^2 + 25 * c₀^2 = 4 ∧ 6 * a₀ + 3 * b₀ + 10 * c₀ = Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3025_302547


namespace NUMINAMATH_CALUDE_rabbit_groupings_count_l3025_302541

/-- The number of ways to divide 12 rabbits into specific groups -/
def rabbit_groupings : ℕ :=
  let total_rabbits : ℕ := 12
  let group1_size : ℕ := 4
  let group2_size : ℕ := 6
  let group3_size : ℕ := 2
  let remaining_rabbits : ℕ := total_rabbits - 2  -- BunBun and Thumper are already placed
  Nat.choose remaining_rabbits (group1_size - 1) * Nat.choose (remaining_rabbits - (group1_size - 1)) (group2_size - 1)

/-- Theorem stating the number of ways to divide the rabbits -/
theorem rabbit_groupings_count : rabbit_groupings = 2520 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_groupings_count_l3025_302541


namespace NUMINAMATH_CALUDE_three_fourths_of_48_plus_5_l3025_302521

theorem three_fourths_of_48_plus_5 : (3 / 4 : ℚ) * 48 + 5 = 41 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_of_48_plus_5_l3025_302521


namespace NUMINAMATH_CALUDE_constant_b_value_l3025_302502

theorem constant_b_value (x y : ℝ) (b : ℝ) 
  (h1 : (7 * x + b * y) / (x - 2 * y) = 29)
  (h2 : x / (2 * y) = 3 / 2) : 
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_constant_b_value_l3025_302502


namespace NUMINAMATH_CALUDE_isosceles_triangle_n_value_l3025_302508

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : (side1 = side2 ∧ side3 ≠ side1) ∨ (side1 = side3 ∧ side2 ≠ side1) ∨ (side2 = side3 ∧ side1 ≠ side2)

/-- The quadratic equation x^2 - 8x + n = 0 -/
def quadratic_equation (x n : ℝ) : Prop :=
  x^2 - 8*x + n = 0

/-- Theorem statement -/
theorem isosceles_triangle_n_value :
  ∀ (t : IsoscelesTriangle) (n : ℝ),
    ((t.side1 = 3 ∨ t.side2 = 3 ∨ t.side3 = 3) ∧
     (quadratic_equation t.side1 n ∧ quadratic_equation t.side2 n) ∨
     (quadratic_equation t.side1 n ∧ quadratic_equation t.side3 n) ∨
     (quadratic_equation t.side2 n ∧ quadratic_equation t.side3 n)) →
    n = 15 ∨ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_n_value_l3025_302508


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3025_302585

theorem quadratic_inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | x^2 - 2*a*x - 3*a^2 < 0} = {x : ℝ | 3*a < x ∧ x < -a} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3025_302585


namespace NUMINAMATH_CALUDE_compound_proposition_true_l3025_302500

theorem compound_proposition_true : 
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∨ (∀ n : ℝ, ∃ m : ℝ, m^2 < n) :=
by sorry

end NUMINAMATH_CALUDE_compound_proposition_true_l3025_302500


namespace NUMINAMATH_CALUDE_equal_group_formation_l3025_302545

-- Define the total number of people
def total_people : ℕ := 20

-- Define the number of boys
def num_boys : ℕ := 10

-- Define the number of girls
def num_girls : ℕ := 10

-- Define the size of the group to be formed
def group_size : ℕ := 10

-- Theorem statement
theorem equal_group_formation :
  Nat.choose total_people group_size = 184756 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_group_formation_l3025_302545


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_negative_three_l3025_302512

theorem no_linear_term_implies_m_equals_negative_three (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 3) = a * x^2 + b) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_negative_three_l3025_302512


namespace NUMINAMATH_CALUDE_hundredth_term_is_14_l3025_302593

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def is_nth_term (x n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ S (x - 1) < k ∧ k ≤ S x

theorem hundredth_term_is_14 : is_nth_term 14 100 := by sorry

end NUMINAMATH_CALUDE_hundredth_term_is_14_l3025_302593


namespace NUMINAMATH_CALUDE_volume_equality_l3025_302592

-- Define the region S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |5 - p.1| + p.2 ≤ 8 ∧ 4 * p.2 - p.1 ≥ 10}

-- Define the line y = x
def line_y_eq_x (x : ℝ) : ℝ := x

-- Define the volume of the solid obtained by revolving S around y = x
noncomputable def volume_of_solid : ℝ := sorry

-- Define the volume calculated using the cone formula
noncomputable def volume_by_cones : ℝ := sorry

-- Theorem statement
theorem volume_equality : volume_of_solid = volume_by_cones := by sorry

end NUMINAMATH_CALUDE_volume_equality_l3025_302592


namespace NUMINAMATH_CALUDE_max_player_salary_l3025_302564

theorem max_player_salary (n : ℕ) (min_salary max_total : ℝ) :
  n = 15 →
  min_salary = 20000 →
  max_total = 800000 →
  (∃ (salaries : Fin n → ℝ),
    (∀ i, salaries i ≥ min_salary) ∧
    (Finset.sum Finset.univ salaries ≤ max_total) ∧
    (∀ i, salaries i ≤ 520000)) ∧
  ¬(∃ (salaries : Fin n → ℝ),
    (∀ i, salaries i ≥ min_salary) ∧
    (Finset.sum Finset.univ salaries ≤ max_total) ∧
    (∃ i, salaries i > 520000)) :=
by sorry

end NUMINAMATH_CALUDE_max_player_salary_l3025_302564


namespace NUMINAMATH_CALUDE_vector_subtraction_l3025_302538

theorem vector_subtraction (c d : Fin 3 → ℝ) 
  (hc : c = ![5, -3, 2])
  (hd : d = ![-2, 1, 5]) :
  c - 4 • d = ![13, -7, -18] := by
sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3025_302538


namespace NUMINAMATH_CALUDE_dividend_calculation_l3025_302552

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 8) :
  divisor * quotient + remainder = 161 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3025_302552


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3025_302570

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  a = 4 → ∃ (r s : ℝ), a * x^2 + 16 * x + 16 = (r * x + s)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3025_302570


namespace NUMINAMATH_CALUDE_max_distance_complex_l3025_302527

theorem max_distance_complex (z : ℂ) (h : Complex.abs (z - 1) = 1) :
  ∃ (max_val : ℝ), max_val = 3 ∧ ∀ w, Complex.abs (w - 1) = 1 → Complex.abs (w - (2 * Complex.I + 1)) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l3025_302527


namespace NUMINAMATH_CALUDE_sum_zero_ratio_theorem_l3025_302590

theorem sum_zero_ratio_theorem (x y z w : ℝ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) 
  (h_sum_zero : x + y + z + w = 0) : 
  (x*y + y*z + z*x + w*x + w*y + w*z) / (x^2 + y^2 + z^2 + w^2) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_zero_ratio_theorem_l3025_302590


namespace NUMINAMATH_CALUDE_janes_reading_speed_l3025_302519

theorem janes_reading_speed (total_pages : ℕ) (first_half_speed : ℕ) (total_days : ℕ) 
  (h1 : total_pages = 500)
  (h2 : first_half_speed = 10)
  (h3 : total_days = 75) :
  (total_pages / 2) / (total_days - (total_pages / 2) / first_half_speed) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_reading_speed_l3025_302519


namespace NUMINAMATH_CALUDE_factorization_theorem_l3025_302557

/-- The polynomial to be factored -/
def p (x : ℝ) : ℝ := x^2 + 6*x + 9 - 64*x^4

/-- The first factor of the factorization -/
def f1 (x : ℝ) : ℝ := -8*x^2 + x + 3

/-- The second factor of the factorization -/
def f2 (x : ℝ) : ℝ := 8*x^2 + x + 3

/-- Theorem stating that p(x) is equal to the product of f1(x) and f2(x) for all real x -/
theorem factorization_theorem : ∀ x : ℝ, p x = f1 x * f2 x := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l3025_302557


namespace NUMINAMATH_CALUDE_base4_calculation_l3025_302594

/-- Converts a number from base 4 to base 10 -/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Performs division in base 4 -/
def divBase4 (a b : ℕ) : ℕ := sorry

/-- Performs multiplication in base 4 -/
def mulBase4 (a b : ℕ) : ℕ := sorry

theorem base4_calculation :
  mulBase4 (divBase4 130 3) 14 = 1200 := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l3025_302594


namespace NUMINAMATH_CALUDE_perpendicular_circle_radius_l3025_302579

/-- Given two perpendicular lines and a circle of radius R tangent to these lines,
    the radius of a circle that is tangent to the same lines and intersects
    the given circle at a right angle is R(2 ± √3). -/
theorem perpendicular_circle_radius (R : ℝ) (R_pos : R > 0) :
  ∃ x : ℝ, x > 0 ∧ 
  (x = R * (2 + Real.sqrt 3) ∨ x = R * (2 - Real.sqrt 3)) ∧
  (∃ (C C₁ : ℝ × ℝ),
    (C.1 = R ∧ C.2 = R) ∧  -- Center of the given circle
    (C₁.1 > 0 ∧ C₁.2 > 0) ∧  -- Center of the new circle in the first quadrant
    ((C₁.1 - C.1)^2 + (C₁.2 - C.2)^2 = (x + R)^2) ∧  -- Circles intersect at right angle
    (C₁.1 = x ∧ C₁.2 = x))  -- New circle is tangent to the perpendicular lines
:= by sorry

end NUMINAMATH_CALUDE_perpendicular_circle_radius_l3025_302579


namespace NUMINAMATH_CALUDE_survey_income_problem_l3025_302540

/-- Proves that given the conditions from the survey, the average income of the other 40 customers is $42,500 -/
theorem survey_income_problem (total_customers : ℕ) (wealthy_customers : ℕ) 
  (total_avg_income : ℝ) (wealthy_avg_income : ℝ) :
  total_customers = 50 →
  wealthy_customers = 10 →
  total_avg_income = 45000 →
  wealthy_avg_income = 55000 →
  let other_customers := total_customers - wealthy_customers
  let total_income := total_avg_income * total_customers
  let wealthy_income := wealthy_avg_income * wealthy_customers
  let other_income := total_income - wealthy_income
  other_income / other_customers = 42500 := by
sorry

end NUMINAMATH_CALUDE_survey_income_problem_l3025_302540


namespace NUMINAMATH_CALUDE_image_and_preimage_l3025_302539

def f (x : ℝ) : ℝ := x^2 - 2*x - 1

theorem image_and_preimage :
  (f (1 + Real.sqrt 2) = 0) ∧
  ({x : ℝ | f x = -1} = {0, 2}) := by
sorry

end NUMINAMATH_CALUDE_image_and_preimage_l3025_302539


namespace NUMINAMATH_CALUDE_sequence_value_l3025_302576

/-- Given a sequence {aₙ} satisfying a₁ = 1 and aₙ - aₙ₋₁ = 2ⁿ⁻¹ for n ≥ 2, prove that a₈ = 255 -/
theorem sequence_value (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n - a (n-1) = 2^(n-1)) : 
  a 8 = 255 := by
sorry

end NUMINAMATH_CALUDE_sequence_value_l3025_302576


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3025_302560

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (3, 2)
  are_parallel a b → m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3025_302560


namespace NUMINAMATH_CALUDE_smallest_possible_d_l3025_302515

theorem smallest_possible_d : 
  ∀ c d : ℝ, 
  (2 < c) → 
  (c < d) → 
  (2 + c ≤ d) → 
  (1 / c + 1 / d ≤ 2) → 
  (∀ d' : ℝ, 
    (∃ c' : ℝ, (2 < c') ∧ (c' < d') ∧ (2 + c' ≤ d') ∧ (1 / c' + 1 / d' ≤ 2)) → 
    d' ≥ 2 + Real.sqrt 3) → 
  d = 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l3025_302515


namespace NUMINAMATH_CALUDE_teaching_years_sum_l3025_302588

/-- The combined years of teaching for Virginia, Adrienne, and Dennis -/
def combined_years (virginia adrienne dennis : ℕ) : ℕ := virginia + adrienne + dennis

/-- Theorem stating the combined years of teaching given the conditions -/
theorem teaching_years_sum :
  ∀ (virginia adrienne dennis : ℕ),
  virginia = adrienne + 9 →
  virginia = dennis - 9 →
  dennis = 40 →
  combined_years virginia adrienne dennis = 93 := by
sorry

end NUMINAMATH_CALUDE_teaching_years_sum_l3025_302588


namespace NUMINAMATH_CALUDE_library_code_probability_l3025_302548

/-- The number of digits in the code -/
def code_length : ℕ := 6

/-- The total number of possible digits -/
def total_digits : ℕ := 10

/-- The probability of selecting a code with all different digits and not starting with 0 -/
def probability : ℚ := 1496880 / 1000000

/-- Theorem stating the probability of selecting a code with all different digits 
    and not starting with 0 is 0.13608 -/
theorem library_code_probability : 
  probability = 1496880 / 1000000 ∧ 
  (1496880 : ℚ) / 1000000 = 0.13608 := by sorry

end NUMINAMATH_CALUDE_library_code_probability_l3025_302548


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3025_302528

theorem quadratic_roots_problem (p : ℤ) : 
  (∃ u v : ℤ, u > 0 ∧ v > 0 ∧ 
   5 * u^2 - 5 * p * u + (66 * p - 1) = 0 ∧
   5 * v^2 - 5 * p * v + (66 * p - 1) = 0) →
  p = 76 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3025_302528


namespace NUMINAMATH_CALUDE_xyz_product_magnitude_l3025_302558

theorem xyz_product_magnitude (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x)
  (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) (h3 : z + 1/x = 2) :
  |x * y * z| = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_magnitude_l3025_302558


namespace NUMINAMATH_CALUDE_min_sum_equal_last_three_digits_l3025_302501

theorem min_sum_equal_last_three_digits (m n : ℕ) : 
  m ≥ 1 → n > m → 
  (1978^n - 1978^m) % 1000 = 0 → 
  (∀ k l : ℕ, k ≥ 1 → l > k → (1978^l - 1978^k) % 1000 = 0 → m + n ≤ k + l) → 
  m + n = 106 := by
sorry

end NUMINAMATH_CALUDE_min_sum_equal_last_three_digits_l3025_302501


namespace NUMINAMATH_CALUDE_card_draw_not_algorithm_l3025_302553

/-- Represents an algorithm in our discussion -/
structure Algorithm where
  steps : List String
  rules : List String
  problem_type : String
  computable : Bool

/-- Represents the operation of calculating the possibility of reaching 24 by randomly drawing 4 playing cards -/
def card_draw_operation : Algorithm := sorry

/-- The definition of an algorithm in our discussion -/
def is_valid_algorithm (a : Algorithm) : Prop :=
  a.steps.length > 0 ∧ 
  a.steps.all (λ s => s.length > 0) ∧
  a.rules.length > 0 ∧
  a.problem_type.length > 0 ∧
  a.computable

/-- Theorem stating that the card draw operation is not a valid algorithm -/
theorem card_draw_not_algorithm : ¬(is_valid_algorithm card_draw_operation) := by
  sorry

end NUMINAMATH_CALUDE_card_draw_not_algorithm_l3025_302553
