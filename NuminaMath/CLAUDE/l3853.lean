import Mathlib

namespace NUMINAMATH_CALUDE_area_ratio_of_similar_triangles_l3853_385303

-- Define two triangles
variable (T1 T2 : Set (ℝ × ℝ))

-- Define similarity ratio
variable (k : ℝ)

-- Define the property of similarity
def are_similar (T1 T2 : Set (ℝ × ℝ)) (k : ℝ) : Prop := sorry

-- Define the area of a triangle
def area (T : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_ratio_of_similar_triangles 
  (h_similar : are_similar T1 T2 k) 
  (h_k_pos : k > 0) :
  area T2 / area T1 = k^2 := sorry

end NUMINAMATH_CALUDE_area_ratio_of_similar_triangles_l3853_385303


namespace NUMINAMATH_CALUDE_product_of_sqrt5_plus_minus_2_l3853_385353

theorem product_of_sqrt5_plus_minus_2 :
  let a := Real.sqrt 5 + 2
  let b := Real.sqrt 5 - 2
  a * b = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_sqrt5_plus_minus_2_l3853_385353


namespace NUMINAMATH_CALUDE_power_equality_l3853_385390

theorem power_equality (x : ℝ) (h : (2 : ℝ) ^ (3 * x) = 7) : (8 : ℝ) ^ (x + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3853_385390


namespace NUMINAMATH_CALUDE_sqrt_square_eq_x_for_nonneg_l3853_385326

theorem sqrt_square_eq_x_for_nonneg (x : ℝ) (h : x ≥ 0) : (Real.sqrt x)^2 = x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_x_for_nonneg_l3853_385326


namespace NUMINAMATH_CALUDE_f_properties_l3853_385356

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, deriv f x > 0) ∧
  (∀ k : ℝ, (∀ x : ℝ, x > 0 → f x > k * x) ↔ k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3853_385356


namespace NUMINAMATH_CALUDE_unique_prime_double_squares_l3853_385322

theorem unique_prime_double_squares : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x : ℕ), p + 7 = 2 * x^2) ∧ 
    (∃ (y : ℕ), p^2 + 7 = 2 * y^2) ∧ 
    p = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_double_squares_l3853_385322


namespace NUMINAMATH_CALUDE_girls_joined_correct_l3853_385329

/-- The number of girls who joined the school -/
def girls_joined : ℕ := 465

/-- The initial number of girls in the school -/
def initial_girls : ℕ := 632

/-- The initial number of boys in the school -/
def initial_boys : ℕ := 410

/-- The difference between the number of girls and boys after some girls joined -/
def girl_boy_difference : ℕ := 687

theorem girls_joined_correct :
  initial_girls + girls_joined = initial_boys + girl_boy_difference :=
by sorry

end NUMINAMATH_CALUDE_girls_joined_correct_l3853_385329


namespace NUMINAMATH_CALUDE_pi_half_irrational_l3853_385337

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l3853_385337


namespace NUMINAMATH_CALUDE_speed_AB_is_60_l3853_385387

-- Define the distances and speeds
def distance_BC : ℝ := 1  -- We can use any positive real number as the base distance
def distance_AB : ℝ := 2 * distance_BC
def speed_BC : ℝ := 20
def average_speed : ℝ := 36

-- Define the speed from A to B as a variable we want to solve for
def speed_AB : ℝ := sorry

-- Theorem statement
theorem speed_AB_is_60 : speed_AB = 60 := by
  sorry

end NUMINAMATH_CALUDE_speed_AB_is_60_l3853_385387


namespace NUMINAMATH_CALUDE_employee_payment_l3853_385394

theorem employee_payment (total_payment x y z : ℝ) : 
  total_payment = 1000 →
  x = 1.2 * y →
  z = 0.8 * y →
  x + z = 600 →
  y = 300 := by sorry

end NUMINAMATH_CALUDE_employee_payment_l3853_385394


namespace NUMINAMATH_CALUDE_mixed_oil_rate_l3853_385336

/-- The rate of mixed oil per litre given specific quantities and prices of three types of oil -/
theorem mixed_oil_rate (quantity1 quantity2 quantity3 : ℚ) (price1 price2 price3 : ℚ) : 
  quantity1 = 12 ∧ quantity2 = 8 ∧ quantity3 = 4 ∧
  price1 = 55 ∧ price2 = 70 ∧ price3 = 82 →
  (quantity1 * price1 + quantity2 * price2 + quantity3 * price3) / (quantity1 + quantity2 + quantity3) = 64.5 := by
  sorry

#check mixed_oil_rate

end NUMINAMATH_CALUDE_mixed_oil_rate_l3853_385336


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_contradictory_l3853_385375

-- Define the bag contents
def total_balls : ℕ := 4
def red_balls : ℕ := 2
def black_balls : ℕ := 2

-- Define the events
def exactly_one_black (drawn : ℕ) : Prop := drawn = 1
def exactly_two_black (drawn : ℕ) : Prop := drawn = 2

-- Define mutual exclusivity
def mutually_exclusive (event1 event2 : ℕ → Prop) : Prop :=
  ∀ n, ¬(event1 n ∧ event2 n)

-- Define non-contradictory
def non_contradictory (event1 event2 : ℕ → Prop) : Prop :=
  ∃ n, event1 n ∨ event2 n

-- Theorem statement
theorem events_mutually_exclusive_but_not_contradictory :
  mutually_exclusive exactly_one_black exactly_two_black ∧
  non_contradictory exactly_one_black exactly_two_black :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_contradictory_l3853_385375


namespace NUMINAMATH_CALUDE_vector_identity_l3853_385331

variable {V : Type*} [AddCommGroup V]

/-- For any four points A, B, C, and D in a vector space, 
    CB + AD - AB = CD -/
theorem vector_identity (A B C D : V) : C - B + (D - A) - (B - A) = D - C := by
  sorry

end NUMINAMATH_CALUDE_vector_identity_l3853_385331


namespace NUMINAMATH_CALUDE_dot_product_constant_l3853_385334

/-- Definition of ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Definition of curve E -/
def curve_E (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Definition of point D -/
def point_D : ℝ × ℝ := (-2, 0)

/-- Definition of line passing through origin -/
def line_through_origin (k : ℝ) (x y : ℝ) : Prop := y = k * x

/-- Theorem: Dot product of DA and DB is constant -/
theorem dot_product_constant (A B : ℝ × ℝ) :
  curve_E A.1 A.2 →
  curve_E B.1 B.2 →
  (∃ k : ℝ, line_through_origin k A.1 A.2 ∧ line_through_origin k B.1 B.2) →
  ((A.1 + 2) * (B.1 + 2) + (A.2 * B.2) = 3) :=
sorry

end NUMINAMATH_CALUDE_dot_product_constant_l3853_385334


namespace NUMINAMATH_CALUDE_oxygen_weight_in_N2O_l3853_385349

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in N2O -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in N2O -/
def num_O : ℕ := 1

/-- The molecular weight of the oxygen part in N2O -/
def molecular_weight_O_part : ℝ := num_O * atomic_weight_O

theorem oxygen_weight_in_N2O : 
  molecular_weight_O_part = 16.00 := by sorry

end NUMINAMATH_CALUDE_oxygen_weight_in_N2O_l3853_385349


namespace NUMINAMATH_CALUDE_initial_peaches_proof_l3853_385379

/-- The number of peaches Mike picked from the orchard -/
def peaches_picked : ℕ := 52

/-- The total number of peaches Mike has now -/
def total_peaches : ℕ := 86

/-- The initial number of peaches at Mike's roadside fruit dish -/
def initial_peaches : ℕ := total_peaches - peaches_picked

theorem initial_peaches_proof : initial_peaches = 34 := by
  sorry

end NUMINAMATH_CALUDE_initial_peaches_proof_l3853_385379


namespace NUMINAMATH_CALUDE_square_difference_l3853_385328

theorem square_difference (a b : ℝ) (h1 : a + b = 2) (h2 : a - b = 1) : a^2 - b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3853_385328


namespace NUMINAMATH_CALUDE_parabola_tangent_intercept_l3853_385382

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line on a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of the specific parabola x² = 4y -/
def parabola_C : Parabola :=
  { equation := fun x y => x^2 = 4*y }

/-- Focus of the parabola -/
def F : Point :=
  { x := 0, y := 1 }

/-- Point E on the y-axis -/
def E : Point :=
  { x := 0, y := 3 }

/-- Origin -/
def O : Point :=
  { x := 0, y := 0 }

/-- Theorem statement -/
theorem parabola_tangent_intercept :
  ∀ (M : Point),
    parabola_C.equation M.x M.y →
    M.x ≠ 0 →
    (∃ (l : Line),
      -- l is tangent to the parabola at M
      (∀ (P : Point), P.y = l.slope * P.x + l.intercept → parabola_C.equation P.x P.y) →
      -- l passes through M
      M.y = l.slope * M.x + l.intercept →
      -- l is perpendicular to ME
      l.slope * ((M.y - E.y) / (M.x - E.x)) = -1 →
      -- y-intercept of l is -1
      l.intercept = -1) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_intercept_l3853_385382


namespace NUMINAMATH_CALUDE_probability_sum_to_15_l3853_385374

/-- Represents a standard playing card --/
inductive Card
| Number (n : Nat)
| Face
| Ace

/-- A standard 52-card deck --/
def Deck : Finset Card := sorry

/-- The set of number cards (2 through 10) in a standard deck --/
def NumberCards : Finset Card := sorry

/-- The number of ways to choose two cards that sum to 15 from number cards --/
def SumTo15Ways : Nat := sorry

/-- The total number of ways to choose two cards from a 52-card deck --/
def TotalWays : Nat := sorry

/-- The probability of selecting two number cards that sum to 15 --/
theorem probability_sum_to_15 :
  (SumTo15Ways : ℚ) / TotalWays = 16 / 884 := by sorry

end NUMINAMATH_CALUDE_probability_sum_to_15_l3853_385374


namespace NUMINAMATH_CALUDE_polygon_sides_l3853_385305

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 = 3 * 360) → 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l3853_385305


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3853_385358

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 300 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 7 + a 8 + a 9 = 300

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  a 2 + a 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3853_385358


namespace NUMINAMATH_CALUDE_math_books_count_l3853_385395

theorem math_books_count (total_books : ℕ) (math_book_price history_book_price : ℚ) (total_price : ℚ) :
  total_books = 80 →
  math_book_price = 4 →
  history_book_price = 5 →
  total_price = 368 →
  ∃ (math_books : ℕ), 
    math_books ≤ total_books ∧
    math_book_price * math_books + history_book_price * (total_books - math_books) = total_price ∧
    math_books = 32 :=
by
  sorry


end NUMINAMATH_CALUDE_math_books_count_l3853_385395


namespace NUMINAMATH_CALUDE_quadratic_function_characterization_l3853_385385

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ y → (deriv f) ((x + y) / 2) = (f y - f x) / (y - x)

/-- The theorem stating that any differentiable function satisfying the functional equation
    is a quadratic function -/
theorem quadratic_function_characterization (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : SatisfiesFunctionalEquation f) :
    ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_characterization_l3853_385385


namespace NUMINAMATH_CALUDE_bob_anne_distance_difference_l3853_385389

/-- Represents the dimensions of a rectangular block in Geometrytown --/
structure BlockDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle --/
def rectanglePerimeter (d : BlockDimensions) : ℕ :=
  2 * (d.length + d.width)

/-- Represents the street width in Geometrytown --/
def streetWidth : ℕ := 30

/-- Calculates Bob's running distance around the block --/
def bobDistance (d : BlockDimensions) : ℕ :=
  rectanglePerimeter { length := d.length + 2 * streetWidth, width := d.width + 2 * streetWidth }

/-- Calculates Anne's running distance around the block --/
def anneDistance (d : BlockDimensions) : ℕ :=
  rectanglePerimeter d

/-- The main theorem stating the difference between Bob's and Anne's running distances --/
theorem bob_anne_distance_difference (d : BlockDimensions) 
    (h1 : d.length = 300) 
    (h2 : d.width = 500) : 
    bobDistance d - anneDistance d = 240 := by
  sorry

end NUMINAMATH_CALUDE_bob_anne_distance_difference_l3853_385389


namespace NUMINAMATH_CALUDE_numerator_increase_percentage_l3853_385340

theorem numerator_increase_percentage (original_fraction : ℚ) 
  (denominator_decrease : ℚ) (new_fraction : ℚ) : 
  original_fraction = 3/4 →
  denominator_decrease = 8/100 →
  new_fraction = 15/16 →
  ∃ numerator_increase : ℚ, 
    (original_fraction * (1 + numerator_increase)) / (1 - denominator_decrease) = new_fraction ∧
    numerator_increase = 15/100 := by
  sorry

end NUMINAMATH_CALUDE_numerator_increase_percentage_l3853_385340


namespace NUMINAMATH_CALUDE_birds_on_fence_l3853_385383

theorem birds_on_fence (initial_birds joining_birds : ℕ) : 
  initial_birds = 2 → joining_birds = 4 → initial_birds + joining_birds = 6 :=
by sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3853_385383


namespace NUMINAMATH_CALUDE_luke_stickers_l3853_385381

theorem luke_stickers (initial bought birthday given_away used : ℕ) :
  initial = 20 →
  bought = 12 →
  birthday = 20 →
  given_away = 5 →
  used = 8 →
  initial + bought + birthday - given_away - used = 39 := by
  sorry

end NUMINAMATH_CALUDE_luke_stickers_l3853_385381


namespace NUMINAMATH_CALUDE_chocolate_bars_per_box_l3853_385346

theorem chocolate_bars_per_box (total_bars : ℕ) (total_boxes : ℕ) 
  (h1 : total_bars = 849) (h2 : total_boxes = 170) :
  total_bars / total_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_per_box_l3853_385346


namespace NUMINAMATH_CALUDE_timothy_total_cost_l3853_385396

/-- Calculates the total cost of Timothy's purchases --/
def total_cost (land_acres : ℕ) (land_price_per_acre : ℕ) (house_price : ℕ) 
  (num_cows : ℕ) (cow_price : ℕ) (num_chickens : ℕ) (chicken_price : ℕ)
  (solar_install_hours : ℕ) (solar_install_rate : ℕ) (solar_equipment_price : ℕ) : ℕ :=
  land_acres * land_price_per_acre +
  house_price +
  num_cows * cow_price +
  num_chickens * chicken_price +
  solar_install_hours * solar_install_rate + solar_equipment_price

/-- Theorem stating that the total cost of Timothy's purchases is $147,700 --/
theorem timothy_total_cost :
  total_cost 30 20 120000 20 1000 100 5 6 100 6000 = 147700 := by
  sorry

end NUMINAMATH_CALUDE_timothy_total_cost_l3853_385396


namespace NUMINAMATH_CALUDE_netball_points_calculation_l3853_385348

theorem netball_points_calculation 
  (w d : ℕ) 
  (h1 : w > d) 
  (h2 : 7 * w + 3 * d = 44) : 
  5 * w + 2 * d = 31 := by
sorry

end NUMINAMATH_CALUDE_netball_points_calculation_l3853_385348


namespace NUMINAMATH_CALUDE_negative_exponent_equality_l3853_385392

theorem negative_exponent_equality : -5^3 = -(5^3) := by
  sorry

end NUMINAMATH_CALUDE_negative_exponent_equality_l3853_385392


namespace NUMINAMATH_CALUDE_circle_center_point_satisfies_center_circle_center_is_4_2_l3853_385324

/-- The center of a circle given by the equation x^2 - 8x + y^2 - 4y = 16 is (4, 2) -/
theorem circle_center (x y : ℝ) : 
  x^2 - 8*x + y^2 - 4*y = 16 → (x - 4)^2 + (y - 2)^2 = 36 := by
  sorry

/-- The point (4, 2) satisfies the center condition of the circle -/
theorem point_satisfies_center : 
  (4 : ℝ)^2 - 8*4 + (2 : ℝ)^2 - 4*2 = 16 := by
  sorry

/-- The center of the circle with equation x^2 - 8x + y^2 - 4y = 16 is (4, 2) -/
theorem circle_center_is_4_2 : 
  ∃! (c : ℝ × ℝ), ∀ (x y : ℝ), x^2 - 8*x + y^2 - 4*y = 16 → (x - c.1)^2 + (y - c.2)^2 = 36 ∧ c = (4, 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_point_satisfies_center_circle_center_is_4_2_l3853_385324


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_total_oak_trees_after_planting_l3853_385302

/-- The number of oak trees in a park after planting new trees is equal to the sum of the initial number of trees and the number of newly planted trees. -/
theorem oak_trees_after_planting (initial_trees newly_planted_trees : ℕ) :
  initial_trees + newly_planted_trees = initial_trees + newly_planted_trees :=
by sorry

/-- The park initially has 5 oak trees. -/
def initial_oak_trees : ℕ := 5

/-- The number of oak trees to be planted is 4. -/
def oak_trees_to_plant : ℕ := 4

/-- The total number of oak trees after planting is 9. -/
theorem total_oak_trees_after_planting :
  initial_oak_trees + oak_trees_to_plant = 9 :=
by sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_total_oak_trees_after_planting_l3853_385302


namespace NUMINAMATH_CALUDE_largest_inexpressible_number_l3853_385323

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

theorem largest_inexpressible_number : 
  (∀ k : ℕ, k > 19 ∧ k ≤ 50 → is_expressible k) ∧
  ¬is_expressible 19 :=
sorry

end NUMINAMATH_CALUDE_largest_inexpressible_number_l3853_385323


namespace NUMINAMATH_CALUDE_sector_shape_area_l3853_385309

theorem sector_shape_area (r : ℝ) (h : r = 12) : 
  let circle_area := π * r^2
  let sector_90 := (90 / 360) * circle_area
  let sector_120 := (120 / 360) * circle_area
  sector_90 + sector_120 = 84 * π := by
sorry

end NUMINAMATH_CALUDE_sector_shape_area_l3853_385309


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3853_385311

-- Problem 1
theorem problem_1 : -1.5 + 1.4 - (-3.6) - 4.3 + (-5.2) = -6 := by
  sorry

-- Problem 2
theorem problem_2 : 17 - 2^3 / (-2) * 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3853_385311


namespace NUMINAMATH_CALUDE_jump_rope_cost_l3853_385373

/-- The cost of Dalton's desired items --/
structure ItemCosts where
  board_game : ℕ
  playground_ball : ℕ
  jump_rope : ℕ

/-- Dalton's available money and additional need --/
structure DaltonMoney where
  allowance : ℕ
  uncle_gift : ℕ
  additional_need : ℕ

/-- Theorem: Given the costs of items and Dalton's available money, 
    prove that the jump rope costs $7 --/
theorem jump_rope_cost 
  (costs : ItemCosts) 
  (money : DaltonMoney) 
  (h1 : costs.board_game = 12)
  (h2 : costs.playground_ball = 4)
  (h3 : money.allowance = 6)
  (h4 : money.uncle_gift = 13)
  (h5 : money.additional_need = 4)
  (h6 : costs.board_game + costs.playground_ball + costs.jump_rope = 
        money.allowance + money.uncle_gift + money.additional_need) :
  costs.jump_rope = 7 := by
  sorry


end NUMINAMATH_CALUDE_jump_rope_cost_l3853_385373


namespace NUMINAMATH_CALUDE_parabola_equation_l3853_385362

/-- Given a parabola with focus F(0, p/2) where p > 0, if its directrix intersects 
    the hyperbola x^2 - y^2 = 6 at points M and N such that triangle MNF is 
    a right-angled triangle, then the equation of the parabola is x^2 = 4√2y -/
theorem parabola_equation (p : ℝ) (M N : ℝ × ℝ) 
  (h_p : p > 0)
  (h_hyperbola : M.1^2 - M.2^2 = 6 ∧ N.1^2 - N.2^2 = 6)
  (h_right_triangle : (M.1 - 0)^2 + (M.2 - p/2)^2 = p^2 ∧ 
                      (N.1 - 0)^2 + (N.2 - p/2)^2 = p^2) :
  ∃ (x y : ℝ), x^2 = 4 * Real.sqrt 2 * y := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3853_385362


namespace NUMINAMATH_CALUDE_sum_of_valid_m_is_three_l3853_385365

-- Define the linear function
def linear_function (m : ℤ) (x : ℝ) : ℝ := (4 - m) * x - 3

-- Define the fractional equation
def fractional_equation (m : ℤ) (z : ℤ) : Prop :=
  m / (z - 1 : ℝ) - 2 = 3 / (1 - z : ℝ)

-- Main theorem
theorem sum_of_valid_m_is_three :
  ∃ (S : Finset ℤ),
    (∀ m ∈ S,
      (∀ x y : ℝ, x < y → linear_function m x < linear_function m y) ∧
      (∃ z : ℤ, z > 1 ∧ fractional_equation m z)) ∧
    (∀ m : ℤ,
      (∀ x y : ℝ, x < y → linear_function m x < linear_function m y) ∧
      (∃ z : ℤ, z > 1 ∧ fractional_equation m z) →
      m ∈ S) ∧
    (S.sum id = 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_valid_m_is_three_l3853_385365


namespace NUMINAMATH_CALUDE_sandy_final_position_l3853_385339

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define Sandy's walk
def sandy_walk (start : Point) : Point :=
  let p1 : Point := ⟨start.x, start.y - 20⟩  -- 20 meters south
  let p2 : Point := ⟨p1.x + 20, p1.y⟩        -- 20 meters east
  let p3 : Point := ⟨p2.x, p2.y + 20⟩        -- 20 meters north
  let p4 : Point := ⟨p3.x + 10, p3.y⟩        -- 10 meters east
  p4

-- Theorem stating that Sandy ends up 10 meters east of her starting point
theorem sandy_final_position (start : Point) : 
  (sandy_walk start).x - start.x = 10 ∧ (sandy_walk start).y = start.y :=
by sorry

end NUMINAMATH_CALUDE_sandy_final_position_l3853_385339


namespace NUMINAMATH_CALUDE_hyperbola_dot_product_bound_l3853_385307

/-- The hyperbola with center at origin, left focus at (-2,0), and equation x²/a² - y² = 1 where a > 0 -/
structure Hyperbola where
  a : ℝ
  h_a_pos : a > 0

/-- A point on the right branch of the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 = 1
  h_right_branch : x ≥ h.a

/-- The theorem stating that the dot product of OP and FP is bounded below -/
theorem hyperbola_dot_product_bound (h : Hyperbola) (p : HyperbolaPoint h) :
  p.x * (p.x + 2) + p.y * p.y ≥ 3 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_dot_product_bound_l3853_385307


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3853_385360

def M (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def P (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, M a ∩ P a = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3853_385360


namespace NUMINAMATH_CALUDE_fourth_task_completion_time_l3853_385354

-- Define the start and end times
def start_time : ℕ := 12 * 60  -- 12:00 PM in minutes
def end_time : ℕ := 15 * 60    -- 3:00 PM in minutes

-- Define the number of tasks completed
def num_tasks : ℕ := 3

-- Theorem to prove
theorem fourth_task_completion_time 
  (h1 : end_time - start_time = num_tasks * (end_time - start_time) / num_tasks) -- Tasks are equally time-consuming
  (h2 : (end_time - start_time) % num_tasks = 0) -- Ensures division is exact
  : end_time + (end_time - start_time) / num_tasks = 16 * 60 := -- 4:00 PM in minutes
by
  sorry

end NUMINAMATH_CALUDE_fourth_task_completion_time_l3853_385354


namespace NUMINAMATH_CALUDE_area_triangle_ABC_l3853_385312

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line m
def line_m : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}

-- Define the circles A, B, and C
def circle_A : Circle := ⟨(-7, 3), 3⟩
def circle_B : Circle := ⟨(0, -4), 4⟩
def circle_C : Circle := ⟨(9, 5), 5⟩

-- Define the tangent points
def point_A' : ℝ × ℝ := (-7, 0)
def point_B' : ℝ × ℝ := (0, 0)
def point_C' : ℝ × ℝ := (9, 0)

-- Define the properties of the circles and their arrangement
axiom tangent_to_m : 
  circle_A.center.2 = circle_A.radius ∧
  circle_B.center.2 = -circle_B.radius ∧
  circle_C.center.2 = circle_C.radius

axiom external_tangency :
  (circle_A.center.1 - circle_B.center.1)^2 + (circle_A.center.2 - circle_B.center.2)^2 
    = (circle_A.radius + circle_B.radius)^2 ∧
  (circle_C.center.1 - circle_B.center.1)^2 + (circle_C.center.2 - circle_B.center.2)^2 
    = (circle_C.radius + circle_B.radius)^2

axiom B'_between_A'_C' :
  point_A'.1 < point_B'.1 ∧ point_B'.1 < point_C'.1

-- Theorem to prove
theorem area_triangle_ABC : 
  let A := circle_A.center
  let B := circle_B.center
  let C := circle_C.center
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = 63 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_ABC_l3853_385312


namespace NUMINAMATH_CALUDE_not_all_equilateral_triangles_congruent_l3853_385325

/-- An equilateral triangle is a triangle where all three sides are of equal length. -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Two triangles are congruent if they have the same size and shape. -/
def congruent (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side_length = t2.side_length

theorem not_all_equilateral_triangles_congruent :
  ∃ t1 t2 : EquilateralTriangle, ¬(congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_not_all_equilateral_triangles_congruent_l3853_385325


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3853_385327

-- Define the inequality function
def inequality (m : ℝ) (x : ℝ) : Prop :=
  m * (x - 1) > x^2 - x

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ :=
  {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_implies_m_value :
  ∀ m : ℝ, (∀ x : ℝ, inequality m x ↔ x ∈ solution_set m) → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3853_385327


namespace NUMINAMATH_CALUDE_complex_square_sum_l3853_385342

theorem complex_square_sum (a b : ℝ) : (1 + Complex.I) ^ 2 = a + b * Complex.I → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l3853_385342


namespace NUMINAMATH_CALUDE_smallest_odd_angle_in_right_triangle_l3853_385399

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem smallest_odd_angle_in_right_triangle :
  ∀ y : ℕ, 
    (is_odd y) →
    (∃ x : ℕ, 
      (is_even x) ∧ 
      (x + y = 90) ∧ 
      (x > y)) →
    y ≥ 31 :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_angle_in_right_triangle_l3853_385399


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l3853_385371

theorem inequality_holds_iff (m : ℝ) :
  (∀ x : ℝ, (x^2 + m*x - 1) / (2*x^2 - 2*x + 3) < 1) ↔ m > -6 ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l3853_385371


namespace NUMINAMATH_CALUDE_train_speed_problem_l3853_385314

/-- The speed of the second train given the conditions of the problem -/
theorem train_speed_problem (initial_distance : ℝ) (speed_train1 : ℝ) (distance_before_meet : ℝ) :
  initial_distance = 120 →
  speed_train1 = 30 →
  distance_before_meet = 70 →
  ∃ (speed_train2 : ℝ), 
    speed_train2 = 40 ∧ 
    (speed_train1 + speed_train2) * 1 = distance_before_meet ∧
    initial_distance - distance_before_meet = (speed_train1 + speed_train2) * 1 :=
by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_train_speed_problem_l3853_385314


namespace NUMINAMATH_CALUDE_gcd_1515_600_l3853_385366

theorem gcd_1515_600 : Nat.gcd 1515 600 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1515_600_l3853_385366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3853_385320

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1/3)
  (h3 : a 2 + a 5 = 4)
  (h4 : ∃ n : ℕ, a n = 33) :
  (∃ n : ℕ, a n = 33 ∧ n = 50) ∧
  (∃ S : ℚ, S = (50 * 51) / 3 ∧ S = 850) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3853_385320


namespace NUMINAMATH_CALUDE_greatest_integer_in_set_l3853_385343

/-- A set of consecutive even integers -/
def ConsecutiveEvenSet : Type := List Nat

/-- The median of a list of numbers -/
def median (l : List Nat) : Nat :=
  sorry

/-- Check if a list contains only even numbers -/
def allEven (l : List Nat) : Prop :=
  sorry

/-- Check if a list contains consecutive even integers -/
def isConsecutiveEven (l : List Nat) : Prop :=
  sorry

theorem greatest_integer_in_set (s : ConsecutiveEvenSet) 
  (h1 : median s = 150)
  (h2 : s.head! = 140)
  (h3 : allEven s)
  (h4 : isConsecutiveEven s) :
  s.getLast! = 152 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_in_set_l3853_385343


namespace NUMINAMATH_CALUDE_machine_production_rate_l3853_385377

/-- The number of shirts a machine can make in one minute, given the total number of shirts and total time -/
def shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) : ℚ :=
  total_shirts / total_minutes

/-- Theorem stating that the machine makes 7 shirts per minute -/
theorem machine_production_rate :
  shirts_per_minute 196 28 = 7 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_rate_l3853_385377


namespace NUMINAMATH_CALUDE_work_completion_time_l3853_385333

/-- The time taken for three workers to complete a task together,
    given their individual completion times. -/
theorem work_completion_time
  (x_time y_time z_time : ℝ)
  (hx : x_time = 10)
  (hy : y_time = 15)
  (hz : z_time = 20)
  : (1 : ℝ) / ((1 / x_time) + (1 / y_time) + (1 / z_time)) = 60 / 13 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3853_385333


namespace NUMINAMATH_CALUDE_plan_A_first_9_minutes_charge_l3853_385391

/- Define the charge per minute after the first 9 minutes for plan A -/
def plan_A_rate : ℚ := 6 / 100

/- Define the charge per minute for plan B -/
def plan_B_rate : ℚ := 8 / 100

/- Define the duration at which both plans charge the same amount -/
def equal_duration : ℚ := 3

/- Theorem statement -/
theorem plan_A_first_9_minutes_charge : 
  ∃ (charge : ℚ), 
    charge = plan_B_rate * equal_duration ∧ 
    charge = 24 / 100 := by
  sorry

end NUMINAMATH_CALUDE_plan_A_first_9_minutes_charge_l3853_385391


namespace NUMINAMATH_CALUDE_class_size_from_marking_error_l3853_385372

/-- The number of pupils in a class where a marking error occurred. -/
def num_pupils : ℕ := by sorry

/-- The difference between the incorrectly entered mark and the correct mark. -/
def mark_difference : ℚ := 73 - 65

/-- The increase in class average due to the marking error. -/
def average_increase : ℚ := 1/2

theorem class_size_from_marking_error :
  num_pupils = 16 := by sorry

end NUMINAMATH_CALUDE_class_size_from_marking_error_l3853_385372


namespace NUMINAMATH_CALUDE_group_size_proof_l3853_385355

theorem group_size_proof (total : ℕ) (older : ℕ) (prob : ℚ) 
  (h1 : older = 90)
  (h2 : prob = 40/130)
  (h3 : prob = (total - older) / total) :
  total = 130 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l3853_385355


namespace NUMINAMATH_CALUDE_carol_nickels_l3853_385380

/-- Represents the contents of Carol's piggy bank -/
structure PiggyBank where
  quarters : ℕ
  nickels : ℕ
  total_cents : ℕ
  nickel_quarter_diff : nickels = quarters + 7
  total_value : total_cents = 5 * nickels + 25 * quarters

/-- Theorem stating that Carol has 21 nickels in her piggy bank -/
theorem carol_nickels (bank : PiggyBank) (h : bank.total_cents = 455) : bank.nickels = 21 := by
  sorry

end NUMINAMATH_CALUDE_carol_nickels_l3853_385380


namespace NUMINAMATH_CALUDE_hexagonal_solid_volume_l3853_385361

/-- The volume of a solid with a hexagonal base and scaled, rotated upper face -/
theorem hexagonal_solid_volume : 
  let s : ℝ := 4  -- side length of base
  let h : ℝ := 9  -- height of solid
  let base_area : ℝ := (3 * Real.sqrt 3 / 2) * s^2
  let upper_area : ℝ := (3 * Real.sqrt 3 / 2) * (1.5 * s)^2
  let avg_area : ℝ := (base_area + upper_area) / 2
  let volume : ℝ := avg_area * h
  volume = 351 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagonal_solid_volume_l3853_385361


namespace NUMINAMATH_CALUDE_reflection_line_sum_l3853_385386

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,6),
    then m + b = 107/6 -/
theorem reflection_line_sum (m b : ℚ) : 
  (∃ (x y : ℚ), 
    -- Midpoint of original and reflected point lies on the line
    (x + 2)/2 = 6 ∧ (y + 3)/2 = 9/2 ∧ y = m*x + b ∧
    -- Perpendicular slope condition
    (x - 2)*(10 - 2) + (y - 3)*(6 - 3) = 0 ∧
    -- Distance equality condition
    (x - 2)^2 + (y - 3)^2 = (10 - x)^2 + (6 - y)^2) →
  m + b = 107/6 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l3853_385386


namespace NUMINAMATH_CALUDE_base8_531_to_base7_l3853_385319

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def base10ToBase7 (n : ℕ) : List ℕ := sorry

/-- Checks if a list of digits represents a valid base 7 number --/
def isValidBase7 (digits : List ℕ) : Prop :=
  digits.all (· < 7)

theorem base8_531_to_base7 :
  let base10 := base8ToBase10 531
  let base7 := base10ToBase7 base10
  isValidBase7 base7 ∧ base7 = [1, 0, 0, 2] :=
by sorry

end NUMINAMATH_CALUDE_base8_531_to_base7_l3853_385319


namespace NUMINAMATH_CALUDE_expression_value_at_three_l3853_385388

theorem expression_value_at_three : 
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 5)
  f 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l3853_385388


namespace NUMINAMATH_CALUDE_triangle_side_length_proof_l3853_385317

noncomputable def triangle_side_length (A B C : Real) : Real :=
  let AB : Real := 1
  let AC : Real := 2
  let cos_B_plus_sin_C : Real := 1
  (2 * Real.sqrt 21 + 3) / 5

theorem triangle_side_length_proof (A B C : Real) :
  let AB : Real := 1
  let AC : Real := 2
  let cos_B_plus_sin_C : Real := 1
  triangle_side_length A B C = (2 * Real.sqrt 21 + 3) / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_proof_l3853_385317


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l3853_385364

theorem smallest_number_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < 1572 → ¬((y + 3) % 9 = 0 ∧ (y + 3) % 35 = 0 ∧ (y + 3) % 25 = 0 ∧ (y + 3) % 21 = 0)) ∧
  ((1572 + 3) % 9 = 0 ∧ (1572 + 3) % 35 = 0 ∧ (1572 + 3) % 25 = 0 ∧ (1572 + 3) % 21 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l3853_385364


namespace NUMINAMATH_CALUDE_expected_pairs_eq_63_l3853_385332

/-- The number of students in the gathering -/
def n : ℕ := 15

/-- The probability of any pair of students liking each other -/
def p : ℚ := 3/5

/-- The expected number of pairs that like each other -/
def expected_pairs : ℚ := p * (n.choose 2)

theorem expected_pairs_eq_63 : expected_pairs = 63 := by sorry

end NUMINAMATH_CALUDE_expected_pairs_eq_63_l3853_385332


namespace NUMINAMATH_CALUDE_both_selected_probability_l3853_385300

theorem both_selected_probability (ram_prob ravi_prob : ℚ) 
  (h1 : ram_prob = 2/7) 
  (h2 : ravi_prob = 1/5) : 
  ram_prob * ravi_prob = 2/35 := by
sorry

end NUMINAMATH_CALUDE_both_selected_probability_l3853_385300


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l3853_385306

theorem quadratic_inequality_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - m > 0) → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l3853_385306


namespace NUMINAMATH_CALUDE_sarah_connor_wage_ratio_l3853_385357

def connors_hourly_wage : ℝ := 7.20
def sarahs_daily_wage : ℝ := 288
def work_hours : ℕ := 8

theorem sarah_connor_wage_ratio :
  (sarahs_daily_wage / work_hours) / connors_hourly_wage = 5 := by sorry

end NUMINAMATH_CALUDE_sarah_connor_wage_ratio_l3853_385357


namespace NUMINAMATH_CALUDE_trigonometric_equalities_l3853_385397

theorem trigonometric_equalities (α : ℝ) (h : 2 * Real.sin α + Real.cos α = 0) :
  (2 * Real.cos α - Real.sin α) / (Real.sin α + Real.cos α) = 5 ∧
  Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equalities_l3853_385397


namespace NUMINAMATH_CALUDE_multiply_powers_l3853_385367

theorem multiply_powers (a : ℝ) : 6 * a^2 * (1/2 * a^3) = 3 * a^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_l3853_385367


namespace NUMINAMATH_CALUDE_system_and_expression_solution_l3853_385352

theorem system_and_expression_solution :
  -- System of equations
  (∃ (x y : ℝ), 2*x + y = 4 ∧ x + 2*y = 5 ∧ x = 1 ∧ y = 2) ∧
  -- Simplified expression evaluation
  (let x : ℝ := -2; (x^2 + 1) / x = -5/2) :=
sorry

end NUMINAMATH_CALUDE_system_and_expression_solution_l3853_385352


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l3853_385318

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1 : ℤ) * d

theorem twentieth_term_of_sequence (a₁ d : ℤ) (h₁ : a₁ = 8) (h₂ : d = -3) :
  arithmeticSequenceTerm a₁ d 20 = -49 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l3853_385318


namespace NUMINAMATH_CALUDE_central_cell_value_l3853_385301

/-- Represents a 3x3 table of real numbers -/
def Table := Fin 3 → Fin 3 → ℝ

/-- The product of numbers in a row equals 10 -/
def row_product (t : Table) : Prop :=
  ∀ i : Fin 3, (t i 0) * (t i 1) * (t i 2) = 10

/-- The product of numbers in a column equals 10 -/
def col_product (t : Table) : Prop :=
  ∀ j : Fin 3, (t 0 j) * (t 1 j) * (t 2 j) = 10

/-- The product of numbers in any 2x2 square equals 3 -/
def square_product (t : Table) : Prop :=
  ∀ i j : Fin 2, (t i j) * (t i (j+1)) * (t (i+1) j) * (t (i+1) (j+1)) = 3

theorem central_cell_value (t : Table) 
  (h_row : row_product t) 
  (h_col : col_product t) 
  (h_square : square_product t) : 
  t 1 1 = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l3853_385301


namespace NUMINAMATH_CALUDE_prob_at_least_one_value_l3853_385347

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.4

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.5

/-- Events A and B are independent -/
axiom events_independent : True

/-- The probability of at least one of the events A or B occurring -/
def prob_at_least_one : ℝ := 1 - (1 - prob_A) * (1 - prob_B)

/-- Theorem: The probability of at least one of the events A or B occurring is 0.7 -/
theorem prob_at_least_one_value : prob_at_least_one = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_value_l3853_385347


namespace NUMINAMATH_CALUDE_charlottes_distance_l3853_385368

/-- The distance between Charlotte's home and school -/
def distance : ℝ := 60

/-- The time taken for Charlotte's one-way journey in hours -/
def journey_time : ℝ := 6

/-- Charlotte's average speed in miles per hour -/
def average_speed : ℝ := 10

/-- Theorem stating that the distance is equal to the product of average speed and journey time -/
theorem charlottes_distance : distance = average_speed * journey_time := by
  sorry

end NUMINAMATH_CALUDE_charlottes_distance_l3853_385368


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3853_385315

theorem quadratic_inequality_solution_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) ↔ k < -Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3853_385315


namespace NUMINAMATH_CALUDE_remaining_potatoes_l3853_385345

def initial_potatoes : ℕ := 8
def eaten_potatoes : ℕ := 3

theorem remaining_potatoes : initial_potatoes - eaten_potatoes = 5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_potatoes_l3853_385345


namespace NUMINAMATH_CALUDE_marble_probability_l3853_385330

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 20) (h2 : blue = 6) (h3 : red = 9) :
  let white := total - (blue + red)
  (red + white : ℚ) / total = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l3853_385330


namespace NUMINAMATH_CALUDE_problem_statement_l3853_385335

theorem problem_statement (a b : ℝ) (h1 : a * b = 2) (h2 : a - b = 3) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 18 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3853_385335


namespace NUMINAMATH_CALUDE_min_large_buses_correct_l3853_385350

/-- The minimum number of large buses required to transport students --/
def min_large_buses (total_students : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) (min_small_buses : ℕ) : ℕ :=
  let remaining_students := total_students - min_small_buses * small_capacity
  (remaining_students + large_capacity - 1) / large_capacity

theorem min_large_buses_correct (total_students : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) (min_small_buses : ℕ)
  (h1 : total_students = 523)
  (h2 : large_capacity = 45)
  (h3 : small_capacity = 30)
  (h4 : min_small_buses = 5) :
  min_large_buses total_students large_capacity small_capacity min_small_buses = 9 := by
  sorry

#eval min_large_buses 523 45 30 5

end NUMINAMATH_CALUDE_min_large_buses_correct_l3853_385350


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3853_385369

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (3 - 2*I) / I
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3853_385369


namespace NUMINAMATH_CALUDE_tangent_sum_inequality_l3853_385359

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π
  
-- Define the perimeter and inradius
def perimeter (t : AcuteTriangle) : Real := sorry
def inradius (t : AcuteTriangle) : Real := sorry

-- State the theorem
theorem tangent_sum_inequality (t : AcuteTriangle) :
  Real.tan t.A + Real.tan t.B + Real.tan t.C ≥ perimeter t / (2 * inradius t) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_inequality_l3853_385359


namespace NUMINAMATH_CALUDE_product_divisible_by_five_l3853_385393

theorem product_divisible_by_five :
  ∃ k : ℤ, 1495 * 1781 * 1815 * 1999 = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_five_l3853_385393


namespace NUMINAMATH_CALUDE_cloak_change_theorem_l3853_385398

/-- Represents the cost and change for buying an invisibility cloak -/
structure CloakTransaction where
  silver_paid : ℕ
  gold_change : ℕ

/-- Calculates the number of silver coins received as change when buying a cloak with gold coins -/
def silver_change_for_gold_payment (t1 t2 : CloakTransaction) (gold_paid : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct change in silver coins when buying a cloak with 14 gold coins -/
theorem cloak_change_theorem (t1 t2 : CloakTransaction) 
  (h1 : t1 = { silver_paid := 20, gold_change := 4 })
  (h2 : t2 = { silver_paid := 15, gold_change := 1 }) :
  silver_change_for_gold_payment t1 t2 14 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloak_change_theorem_l3853_385398


namespace NUMINAMATH_CALUDE_apple_price_theorem_l3853_385378

/-- The relationship between the selling price and quantity of apples -/
def apple_price_relation (x y : ℝ) : Prop :=
  y = 8 * x

/-- The price increase per kg of apples -/
def price_increase_per_kg : ℝ := 8

theorem apple_price_theorem (x y : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₂ - x₁ = 1 → apple_price_relation x₂ y₂ → apple_price_relation x₁ y₁ → y₂ - y₁ = price_increase_per_kg) →
  apple_price_relation x y :=
sorry

end NUMINAMATH_CALUDE_apple_price_theorem_l3853_385378


namespace NUMINAMATH_CALUDE_stock_price_drop_l3853_385376

theorem stock_price_drop (initial_price : ℝ) (x : ℝ) 
  (h1 : initial_price > 0)
  (h2 : x > 0 ∧ x < 100)
  (h3 : (1 + 0.3) * (1 - x / 100) * (1 + 0.2) * initial_price = 1.17 * initial_price) :
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_stock_price_drop_l3853_385376


namespace NUMINAMATH_CALUDE_game_result_l3853_385370

def game_operation (n : ℕ) : ℕ :=
  if n % 2 = 1 then n + 3 else n / 2

def reaches_one_in (n : ℕ) (steps : ℕ) : Prop :=
  ∃ (seq : Fin steps.succ → ℕ), 
    seq 0 = n ∧ 
    seq steps = 1 ∧ 
    ∀ i : Fin steps, seq (i.succ) = game_operation (seq i)

theorem game_result :
  {n : ℕ | reaches_one_in n 5} = {1, 8, 16, 10, 13} := by sorry

end NUMINAMATH_CALUDE_game_result_l3853_385370


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3853_385310

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, -3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 8 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  ∀ (x y : ℝ),
    (perpendicular_line x y ∧ (x, y) = point_A) →
    (∀ (x' y' : ℝ), given_line x' y' → (x - x') * (x' - x) + (y - y') * (y' - y) = 0) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3853_385310


namespace NUMINAMATH_CALUDE_shirts_sold_proof_l3853_385341

/-- The number of shirts sold by Sab and Dane -/
def num_shirts : ℕ := 18

/-- The number of pairs of shoes sold -/
def num_shoes : ℕ := 6

/-- The price of each pair of shoes in dollars -/
def price_shoes : ℕ := 3

/-- The price of each shirt in dollars -/
def price_shirts : ℕ := 2

/-- The earnings of each person (Sab and Dane) in dollars -/
def earnings_per_person : ℕ := 27

theorem shirts_sold_proof : 
  num_shirts = 18 ∧ 
  num_shoes * price_shoes + num_shirts * price_shirts = 2 * earnings_per_person := by
  sorry

end NUMINAMATH_CALUDE_shirts_sold_proof_l3853_385341


namespace NUMINAMATH_CALUDE_pen_price_calculation_l3853_385363

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 450 ∧ num_pens = 30 ∧ num_pencils = 75 ∧ pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 10 := by
sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l3853_385363


namespace NUMINAMATH_CALUDE_fencing_requirement_l3853_385308

theorem fencing_requirement (area : ℝ) (uncovered_side : ℝ) (fencing : ℝ) : 
  area = 880 →
  uncovered_side = 25 →
  fencing = uncovered_side + 2 * (area / uncovered_side) →
  fencing = 95.4 := by
sorry

end NUMINAMATH_CALUDE_fencing_requirement_l3853_385308


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3853_385338

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → 
  a + h + k = -6 := by sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3853_385338


namespace NUMINAMATH_CALUDE_type_B_completion_time_l3853_385384

/-- The time (in hours) it takes for a type R machine to complete the job -/
def time_R : ℝ := 5

/-- The time (in hours) it takes for 2 type R machines and 3 type B machines working together to complete the job -/
def time_combined : ℝ := 1.2068965517241381

/-- The time (in hours) it takes for a type B machine to complete the job -/
def time_B : ℝ := 7

theorem type_B_completion_time : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |time_B - (3 * time_combined) / (1 / time_combined - 2 / time_R)| < ε :=
sorry

end NUMINAMATH_CALUDE_type_B_completion_time_l3853_385384


namespace NUMINAMATH_CALUDE_remainder_4063_div_97_l3853_385313

theorem remainder_4063_div_97 : 4063 % 97 = 86 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4063_div_97_l3853_385313


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3853_385344

/-- An equilateral triangle with inscribed circles and square -/
structure TriangleWithInscriptions where
  /-- Side length of the equilateral triangle -/
  side : ℝ
  /-- Radius of each inscribed circle -/
  circle_radius : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The circle radius is 4 -/
  h_circle_radius : circle_radius = 4
  /-- The square side is equal to the triangle side minus twice the diameter of two circles -/
  h_square_side : square_side = side - 4 * circle_radius
  /-- The triangle side is composed of two parts touching the circles and the diameter of two circles -/
  h_side : side = 2 * (circle_radius * Real.sqrt 3) + 2 * circle_radius

/-- The perimeter of the triangle is 24 + 24√3 -/
theorem triangle_perimeter (t : TriangleWithInscriptions) : 
  3 * t.side = 24 + 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3853_385344


namespace NUMINAMATH_CALUDE_function_identity_implies_constant_relation_l3853_385316

-- Define the functions and constants
variable (f g : ℝ → ℝ)
variable (a b c : ℝ)

-- State the theorem
theorem function_identity_implies_constant_relation 
  (h : ∀ (x y : ℝ), f x * g y = a * x * y + b * x + c * y + 1) : 
  a = b * c := by sorry

end NUMINAMATH_CALUDE_function_identity_implies_constant_relation_l3853_385316


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3853_385351

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_z : ℝ), min_z = 12 ∧ ∀ z : ℝ, z = 4*x^2 + 8*x + 16 → z ≥ min_z :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3853_385351


namespace NUMINAMATH_CALUDE_compound_interest_doubling_l3853_385321

/-- The annual interest rate as a decimal -/
def r : ℝ := 0.15

/-- The compound interest factor for one year -/
def factor : ℝ := 1 + r

/-- The number of years we're proving about -/
def years : ℕ := 5

theorem compound_interest_doubling :
  (∀ n : ℕ, n < years → factor ^ n ≤ 2) ∧
  factor ^ years > 2 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_doubling_l3853_385321


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3853_385304

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : arithmeticSequence a) 
  (h_sum : a 2 + a 12 = 32) : 
  2 * a 3 + a 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3853_385304
