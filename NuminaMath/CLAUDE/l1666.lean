import Mathlib

namespace NUMINAMATH_CALUDE_max_sum_of_seventh_powers_l1666_166608

theorem max_sum_of_seventh_powers (a b c d : ℝ) 
  (h : a^6 + b^6 + c^6 + d^6 = 64) : 
  ∃ (M : ℝ), M = 128 ∧ a^7 + b^7 + c^7 + d^7 ≤ M ∧ 
  ∃ (a' b' c' d' : ℝ), a'^6 + b'^6 + c'^6 + d'^6 = 64 ∧ 
                        a'^7 + b'^7 + c'^7 + d'^7 = M :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_seventh_powers_l1666_166608


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l1666_166609

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 12) (h2 : n2 = 28) (h3 : avg1 = 40) (h4 : avg2 = 60) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 54 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l1666_166609


namespace NUMINAMATH_CALUDE_power_product_equality_l1666_166602

theorem power_product_equality : (3^3 * 5^3) * (3^8 * 5^8) = 15^11 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1666_166602


namespace NUMINAMATH_CALUDE_pythagorean_fraction_bound_l1666_166635

theorem pythagorean_fraction_bound (m n t : ℝ) (h1 : m^2 + n^2 = t^2) (h2 : t ≠ 0) :
  -Real.sqrt 3 / 3 ≤ n / (m - 2 * t) ∧ n / (m - 2 * t) ≤ Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_fraction_bound_l1666_166635


namespace NUMINAMATH_CALUDE_chord_equation_l1666_166663

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 3

-- Define the midpoint M
def midpoint_M : ℝ × ℝ := (1, 0)

-- Define the chord AB
def chord_AB (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem statement
theorem chord_equation :
  ∀ (x y : ℝ),
  circle_C x y →
  chord_AB x y →
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    midpoint_M = ((x1 + x2) / 2, (y1 + y2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l1666_166663


namespace NUMINAMATH_CALUDE_rotation_of_point_A_l1666_166628

/-- Represents a 2D point or vector -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Rotates a point clockwise by π/2 around the origin -/
def rotate_clockwise_90 (p : Point2D) : Point2D :=
  ⟨p.y, -p.x⟩

theorem rotation_of_point_A : 
  let A : Point2D := ⟨2, 1⟩
  let B : Point2D := rotate_clockwise_90 A
  B.x = 1 ∧ B.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_rotation_of_point_A_l1666_166628


namespace NUMINAMATH_CALUDE_max_colored_cells_l1666_166666

/-- Represents a cell in the 8x8 square --/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Predicate to check if four cells form a rectangle with sides parallel to the edges --/
def formsRectangle (c1 c2 c3 c4 : Cell) : Prop :=
  (c1.row = c2.row ∧ c3.row = c4.row ∧ c1.col = c3.col ∧ c2.col = c4.col) ∨
  (c1.row = c3.row ∧ c2.row = c4.row ∧ c1.col = c2.col ∧ c3.col = c4.col)

/-- The main theorem --/
theorem max_colored_cells :
  ∃ (S : Finset Cell),
    S.card = 24 ∧
    (∀ (c1 c2 c3 c4 : Cell),
      c1 ∈ S → c2 ∈ S → c3 ∈ S → c4 ∈ S →
      c1 ≠ c2 → c1 ≠ c3 → c1 ≠ c4 → c2 ≠ c3 → c2 ≠ c4 → c3 ≠ c4 →
      ¬formsRectangle c1 c2 c3 c4) ∧
    (∀ (T : Finset Cell),
      T.card > 24 →
      ∃ (c1 c2 c3 c4 : Cell),
        c1 ∈ T ∧ c2 ∈ T ∧ c3 ∈ T ∧ c4 ∈ T ∧
        c1 ≠ c2 ∧ c1 ≠ c3 ∧ c1 ≠ c4 ∧ c2 ≠ c3 ∧ c2 ≠ c4 ∧ c3 ≠ c4 ∧
        formsRectangle c1 c2 c3 c4) :=
by sorry


end NUMINAMATH_CALUDE_max_colored_cells_l1666_166666


namespace NUMINAMATH_CALUDE_initial_rate_is_three_l1666_166611

/-- Calculates the initial consumption rate per soldier per day -/
def initial_consumption_rate (initial_soldiers : ℕ) (initial_duration : ℕ) 
  (additional_soldiers : ℕ) (new_consumption_rate : ℚ) (new_duration : ℕ) : ℚ :=
  (((initial_soldiers + additional_soldiers) * new_consumption_rate * new_duration) / 
   (initial_soldiers * initial_duration))

/-- Theorem stating that the initial consumption rate is 3 kg per soldier per day -/
theorem initial_rate_is_three :
  initial_consumption_rate 1200 30 528 (5/2) 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_rate_is_three_l1666_166611


namespace NUMINAMATH_CALUDE_rescue_possible_l1666_166689

/-- Represents the rescue mission parameters --/
structure RescueMission where
  distance : ℝ
  rover_air : ℝ
  ponchik_extra_air : ℝ
  dunno_tank_air : ℝ
  max_tanks : ℕ
  speed : ℝ

/-- Represents a rescue strategy --/
structure RescueStrategy where
  trips : ℕ
  air_drops : List ℝ
  meeting_point : ℝ

/-- Checks if a rescue strategy is valid for a given mission --/
def is_valid_strategy (mission : RescueMission) (strategy : RescueStrategy) : Prop :=
  -- Define the conditions for a valid strategy
  sorry

/-- Theorem stating that a valid rescue strategy exists --/
theorem rescue_possible (mission : RescueMission) 
  (h1 : mission.distance = 18)
  (h2 : mission.rover_air = 3)
  (h3 : mission.ponchik_extra_air = 1)
  (h4 : mission.dunno_tank_air = 2)
  (h5 : mission.max_tanks = 2)
  (h6 : mission.speed = 6) :
  ∃ (strategy : RescueStrategy), is_valid_strategy mission strategy :=
sorry

end NUMINAMATH_CALUDE_rescue_possible_l1666_166689


namespace NUMINAMATH_CALUDE_cos_arctan_squared_l1666_166645

theorem cos_arctan_squared (x : ℝ) (h1 : x > 0) (h2 : Real.cos (Real.arctan x) = x) :
  x^2 = (Real.sqrt 5 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_cos_arctan_squared_l1666_166645


namespace NUMINAMATH_CALUDE_shaded_area_is_thirty_l1666_166690

/-- An isosceles right triangle with legs of length 10 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  is_ten : leg_length = 10

/-- The large triangle partitioned into 25 congruent smaller triangles -/
def num_partitions : ℕ := 25

/-- The number of shaded smaller triangles -/
def num_shaded : ℕ := 15

/-- The theorem to be proved -/
theorem shaded_area_is_thirty 
  (t : IsoscelesRightTriangle) 
  (h_partitions : num_partitions = 25) 
  (h_shaded : num_shaded = 15) : 
  (t.leg_length * t.leg_length / 2) * (num_shaded / num_partitions) = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_thirty_l1666_166690


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_165_l1666_166639

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

theorem arithmetic_progression_sum_165 :
  ∃ ap : ArithmeticProgression,
    sum_n_terms ap 15 = 200 ∧
    sum_n_terms ap 150 = 150 ∧
    sum_n_terms ap 165 = -3064 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_165_l1666_166639


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l1666_166668

-- Define the lines L1 and L2
def L1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def L2 (x y : ℝ) : Prop := x - 3 * y + 8 = 0

-- Define the reference line
def ref_line (x y : ℝ) : Prop := 2 * x + y + 5 = 0

-- Define the intersection point M
def M : ℝ × ℝ := ((-2 : ℝ), (2 : ℝ))

-- Theorem for the parallel line
theorem parallel_line_equation :
  ∀ (x y : ℝ), 2 * x + y + 2 = 0 →
  (L1 (M.1) (M.2) ∧ L2 (M.1) (M.2)) ∧
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), 2 * x + y + 2 = 0 ↔ k * (2 * x + y + 5) = 0 :=
sorry

-- Theorem for the perpendicular line
theorem perpendicular_line_equation :
  ∀ (x y : ℝ), x - 2 * y + 6 = 0 →
  (L1 (M.1) (M.2) ∧ L2 (M.1) (M.2)) ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (x - 2 * y + 6 = 0 → x₁ = x ∧ y₁ = y) →
    (2 * x + y + 5 = 0 → x₂ = x ∧ y₂ = y) →
    (x₂ - x₁) * (x - x₁) + (y₂ - y₁) * (y - y₁) = 0 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l1666_166668


namespace NUMINAMATH_CALUDE_cookie_price_calculation_l1666_166634

theorem cookie_price_calculation (cupcake_price : ℚ) (doughnut_price : ℚ) (pie_slice_price : ℚ) 
  (num_cupcakes : ℕ) (num_doughnuts : ℕ) (num_pie_slices : ℕ) (num_cookies : ℕ) (total_spent : ℚ) :
  cupcake_price = 2 →
  doughnut_price = 1 →
  pie_slice_price = 2 →
  num_cupcakes = 5 →
  num_doughnuts = 6 →
  num_pie_slices = 4 →
  num_cookies = 15 →
  total_spent = 33 →
  (total_spent - (num_cupcakes * cupcake_price + num_doughnuts * doughnut_price + num_pie_slices * pie_slice_price)) / num_cookies = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_cookie_price_calculation_l1666_166634


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1666_166624

-- Define sets A and B
def A : Set ℝ := {x | 2 + x ≥ 4}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1666_166624


namespace NUMINAMATH_CALUDE_fraction_nonnegative_l1666_166649

theorem fraction_nonnegative (x : ℝ) (h : x ≠ -2) : x^2 / (x + 2)^2 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_fraction_nonnegative_l1666_166649


namespace NUMINAMATH_CALUDE_sam_mystery_books_l1666_166652

/-- Represents the number of books in each category --/
structure BookCount where
  adventure : ℕ
  mystery : ℕ
  used : ℕ
  new : ℕ

/-- The total number of books is the sum of used and new books --/
def total_books (b : BookCount) : ℕ := b.used + b.new

/-- Theorem stating the number of mystery books Sam bought --/
theorem sam_mystery_books :
  ∃ (b : BookCount),
    b.adventure = 13 ∧
    b.used = 15 ∧
    b.new = 15 ∧
    total_books b = b.adventure + b.mystery ∧
    b.mystery = 17 := by
  sorry

end NUMINAMATH_CALUDE_sam_mystery_books_l1666_166652


namespace NUMINAMATH_CALUDE_shirt_markup_price_l1666_166692

/-- Given a wholesale price of a shirt, prove that the initial price after 80% markup is $27 -/
theorem shirt_markup_price (P : ℝ) 
  (h1 : 1.80 * P = 1.80 * P) -- Initial price after 80% markup
  (h2 : 2.00 * P = 2.00 * P) -- Price for 100% markup
  (h3 : 2.00 * P - 1.80 * P = 3) -- Difference between 100% and 80% markup is $3
  : 1.80 * P = 27 := by
  sorry

end NUMINAMATH_CALUDE_shirt_markup_price_l1666_166692


namespace NUMINAMATH_CALUDE_max_value_of_largest_integer_l1666_166679

theorem max_value_of_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 60 →
  e.val - a.val = 10 →
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e →
  e.val ≤ 290 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_largest_integer_l1666_166679


namespace NUMINAMATH_CALUDE_michael_matchsticks_l1666_166686

theorem michael_matchsticks (total : ℕ) (houses : ℕ) (sticks_per_house : ℕ) : 
  houses = 30 →
  sticks_per_house = 10 →
  houses * sticks_per_house = total / 2 →
  total = 600 := by
  sorry

end NUMINAMATH_CALUDE_michael_matchsticks_l1666_166686


namespace NUMINAMATH_CALUDE_parabola_directrix_tangent_to_circle_l1666_166648

/-- The value of p for which the directrix of the parabola y² = 2px (p > 0) 
    is tangent to the circle x² + y² - 4x + 2y - 4 = 0 -/
theorem parabola_directrix_tangent_to_circle :
  ∀ p : ℝ, p > 0 →
  (∀ x y : ℝ, y^2 = 2*p*x) →
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y - 4 = 0) →
  (∃ x y : ℝ, x = -p/2 ∧ (x - 2)^2 + (y + 1)^2 = 9) →
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_tangent_to_circle_l1666_166648


namespace NUMINAMATH_CALUDE_families_with_items_l1666_166688

theorem families_with_items (total_telephone : ℕ) (total_tricycle : ℕ) (both : ℕ)
  (h1 : total_telephone = 35)
  (h2 : total_tricycle = 65)
  (h3 : both = 20) :
  total_telephone + total_tricycle - both = 80 := by
  sorry

end NUMINAMATH_CALUDE_families_with_items_l1666_166688


namespace NUMINAMATH_CALUDE_f_value_at_3_l1666_166662

-- Define the function f
def f (a c : ℝ) (x : ℝ) : ℝ := a * x^3 + c * x + 5

-- State the theorem
theorem f_value_at_3 (a c : ℝ) : f a c (-3) = -3 → f a c 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l1666_166662


namespace NUMINAMATH_CALUDE_joker_king_probability_l1666_166641

/-- A deck of cards with jokers -/
structure Deck :=
  (total_cards : ℕ)
  (num_jokers : ℕ)
  (num_kings : ℕ)

/-- The probability of drawing a joker first and a king second -/
def joker_king_prob (d : Deck) : ℚ :=
  (d.num_jokers : ℚ) / d.total_cards * (d.num_kings : ℚ) / (d.total_cards - 1)

/-- The modified 54-card deck -/
def modified_deck : Deck :=
  { total_cards := 54,
    num_jokers := 2,
    num_kings := 4 }

theorem joker_king_probability :
  joker_king_prob modified_deck = 8 / 1431 := by
  sorry

end NUMINAMATH_CALUDE_joker_king_probability_l1666_166641


namespace NUMINAMATH_CALUDE_max_x_value_l1666_166607

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 9) 
  (prod_sum_eq : x*y + x*z + y*z = 20) : 
  x ≤ (18 + Real.sqrt 312) / 6 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l1666_166607


namespace NUMINAMATH_CALUDE_divisibility_property_l1666_166677

theorem divisibility_property (n : ℕ) (a : Fin n → ℕ+) 
  (h_n : n ≥ 3)
  (h_gcd : Nat.gcd (Finset.univ.prod (fun i => (a i).val)) = 1)
  (h_div : ∀ i : Fin n, (a i).val ∣ (Finset.univ.sum (fun j => (a j).val))) :
  (Finset.univ.prod (fun i => (a i).val)) ∣ (Finset.univ.sum (fun i => (a i).val))^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1666_166677


namespace NUMINAMATH_CALUDE_supermarket_spending_l1666_166613

theorem supermarket_spending (total : ℝ) (category1 : ℝ) (category2 : ℝ) (category3 : ℝ) (category4 : ℝ) :
  total = 120 →
  category1 = (1 / 2) * total →
  category2 = (1 / 10) * total →
  category3 = 8 →
  category1 + category2 + category3 + category4 = total →
  category4 / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l1666_166613


namespace NUMINAMATH_CALUDE_no_real_roots_l1666_166627

-- Define the sequence of polynomials P_n(x)
def P : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | n + 1 => λ x => x^(17*(n+1)) - P n x

-- Theorem statement
theorem no_real_roots : ∀ n : ℕ, ∀ x : ℝ, P n x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l1666_166627


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1666_166678

theorem complex_modulus_equality (n : ℝ) :
  n > 0 → (Complex.abs (4 + n * Complex.I) = 4 * Real.sqrt 13 ↔ n = 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1666_166678


namespace NUMINAMATH_CALUDE_sqrt_72_plus_24sqrt6_l1666_166615

theorem sqrt_72_plus_24sqrt6 :
  ∃ (a b c : ℤ), (c > 0) ∧ 
  (∀ (n : ℕ), n > 1 → ¬(∃ (k : ℕ), c = n^2 * k)) ∧
  Real.sqrt (72 + 24 * Real.sqrt 6) = a + b * Real.sqrt c ∧
  a = 6 ∧ b = 3 ∧ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_72_plus_24sqrt6_l1666_166615


namespace NUMINAMATH_CALUDE_multiplier_problem_l1666_166695

theorem multiplier_problem (a b : ℝ) (h1 : 4 * a = b) (h2 : b = 30) (h3 : 40 * a * b = 1800) :
  ∃ m : ℝ, m * b = 30 ∧ m = 5 := by
sorry

end NUMINAMATH_CALUDE_multiplier_problem_l1666_166695


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l1666_166601

/-- Represents a 9x9 checkerboard filled with numbers 1 through 81 in order across rows. -/
def Checkerboard := Fin 9 → Fin 9 → Nat

/-- The value at position (i, j) on the checkerboard. -/
def checkerboardValue (i j : Fin 9) : Nat :=
  i.val * 9 + j.val + 1

/-- The sum of the values in the four corners of the checkerboard. -/
def cornerSum (board : Checkerboard) : Nat :=
  board 0 0 + board 0 8 + board 8 0 + board 8 8

/-- Theorem stating that the sum of the numbers in the four corners of the checkerboard is 164. -/
theorem corner_sum_is_164 (board : Checkerboard) :
  (∀ i j : Fin 9, board i j = checkerboardValue i j) →
  cornerSum board = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l1666_166601


namespace NUMINAMATH_CALUDE_system_solution_l1666_166687

theorem system_solution (x y z w : ℝ) : 
  (x - y + z - w = 2) ∧
  (x^2 - y^2 + z^2 - w^2 = 6) ∧
  (x^3 - y^3 + z^3 - w^3 = 20) ∧
  (x^4 - y^4 + z^4 - w^4 = 60) →
  ((x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1666_166687


namespace NUMINAMATH_CALUDE_frank_breakfast_shopping_l1666_166650

/-- The cost of one bun in dollars -/
def bun_cost : ℚ := 1/10

/-- The cost of one bottle of milk in dollars -/
def milk_cost : ℚ := 2

/-- The number of bottles of milk Frank bought -/
def milk_bottles : ℕ := 2

/-- The cost of a carton of eggs in dollars -/
def egg_cost : ℚ := 3 * milk_cost

/-- The total amount Frank paid in dollars -/
def total_paid : ℚ := 11

/-- The number of buns Frank bought -/
def buns_bought : ℕ := 10

theorem frank_breakfast_shopping :
  buns_bought * bun_cost + milk_bottles * milk_cost + egg_cost = total_paid :=
by sorry

end NUMINAMATH_CALUDE_frank_breakfast_shopping_l1666_166650


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1666_166665

theorem unique_triple_solution (a b p : ℕ) (h_prime : Nat.Prime p) :
  (a + b)^p = p^a + p^b ↔ a = 1 ∧ b = 1 ∧ p = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1666_166665


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1666_166655

/-- Given a line in vector form, prove it's equivalent to slope-intercept form -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 4) + (5 : ℝ) * (y - 1) = 0 ↔ 
  y = -(2/5 : ℝ) * x + (13/5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1666_166655


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l1666_166694

theorem complex_magnitude_proof : Complex.abs (8/7 + 3*I) = Real.sqrt 505 / 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l1666_166694


namespace NUMINAMATH_CALUDE_real_roots_condition_l1666_166626

theorem real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - x + 3 = 0) ↔ (k ≤ 1/12 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_l1666_166626


namespace NUMINAMATH_CALUDE_intersection_range_length_AB_l1666_166699

-- Define the hyperbola C: x^2 - y^2 = 1
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the line l: y = kx + 1
def line (k x y : ℝ) : Prop := y = k * x + 1

-- Define the condition for two distinct intersection points
def has_two_intersections (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂

-- Theorem for the range of k
theorem intersection_range :
  ∀ k : ℝ, has_two_intersections k ↔ 
    (k > -Real.sqrt 2 ∧ k < -1) ∨ 
    (k > -1 ∧ k < 1) ∨ 
    (k > 1 ∧ k < Real.sqrt 2) :=
sorry

-- Define the condition for the midpoint x-coordinate
def midpoint_x_is_sqrt2 (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    (x₁ + x₂) / 2 = Real.sqrt 2

-- Theorem for the length of AB
theorem length_AB (k : ℝ) :
  midpoint_x_is_sqrt2 k → 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_length_AB_l1666_166699


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_l1666_166673

theorem largest_of_three_consecutive_integers (n : ℤ) 
  (h : (n - 1) + n + (n + 1) = 90) : 
  max (n - 1) (max n (n + 1)) = 31 := by
sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_l1666_166673


namespace NUMINAMATH_CALUDE_both_systematic_sampling_l1666_166676

/-- Represents a sampling method --/
inductive SamplingMethod
| Systematic
| SimpleRandom
| Stratified

/-- Represents a reporter conducting interviews --/
structure Reporter where
  name : String
  interval : Nat
  intervalType : String

/-- Represents the interview setup at the train station --/
structure InterviewSetup where
  reporterA : Reporter
  reporterB : Reporter
  constantFlow : Bool

/-- Determines the sampling method based on the interview setup --/
def determineSamplingMethod (reporter : Reporter) (setup : InterviewSetup) : SamplingMethod :=
  if setup.constantFlow && (reporter.intervalType = "time" || reporter.intervalType = "people") then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

/-- Theorem: Both reporters are using systematic sampling --/
theorem both_systematic_sampling (setup : InterviewSetup) 
  (h1 : setup.reporterA = { name := "A", interval := 10, intervalType := "time" })
  (h2 : setup.reporterB = { name := "B", interval := 1000, intervalType := "people" })
  (h3 : setup.constantFlow = true) :
  determineSamplingMethod setup.reporterA setup = SamplingMethod.Systematic ∧
  determineSamplingMethod setup.reporterB setup = SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_both_systematic_sampling_l1666_166676


namespace NUMINAMATH_CALUDE_triangle_area_product_l1666_166660

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → (1/2) * (8/a) * (8/b) = 8 → a * b = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_product_l1666_166660


namespace NUMINAMATH_CALUDE_braiding_time_for_dance_team_l1666_166638

/-- Calculates the time in minutes to braid dancers' hair -/
def braiding_time (num_dancers : ℕ) (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) : ℕ :=
  (num_dancers * braids_per_dancer * seconds_per_braid) / 60

/-- Proves that braiding 8 dancers' hair with 5 braids each, taking 30 seconds per braid, results in 20 minutes total -/
theorem braiding_time_for_dance_team : braiding_time 8 5 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_braiding_time_for_dance_team_l1666_166638


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_difference_56_l1666_166637

-- Define the property for two numbers to have the same last two digits in their squares
def SameLastTwoDigitsSquared (a b : ℕ) : Prop :=
  a ^ 2 % 100 = b ^ 2 % 100

-- Main theorem
theorem two_digit_numbers_with_difference_56 :
  ∀ x y : ℕ,
    10 ≤ x ∧ x < 100 →  -- x is a two-digit number
    10 ≤ y ∧ y < 100 →  -- y is a two-digit number
    x - y = 56 →        -- their difference is 56
    SameLastTwoDigitsSquared x y →  -- last two digits of their squares are the same
    (x = 78 ∧ y = 22) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_difference_56_l1666_166637


namespace NUMINAMATH_CALUDE_cannot_determine_read_sonnets_l1666_166616

/-- Represents the number of lines in a sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of unread lines -/
def unread_lines : ℕ := 70

/-- Represents the number of sonnets not read -/
def unread_sonnets : ℕ := unread_lines / lines_per_sonnet

theorem cannot_determine_read_sonnets (total_sonnets : ℕ) :
  ∀ n : ℕ, n < total_sonnets → n ≥ unread_sonnets →
  ∃ m : ℕ, m ≠ n ∧ m < total_sonnets ∧ m ≥ unread_sonnets :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_read_sonnets_l1666_166616


namespace NUMINAMATH_CALUDE_sons_age_l1666_166669

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1666_166669


namespace NUMINAMATH_CALUDE_betty_nuts_purchase_l1666_166636

/-- The number of packs of nuts Betty wants to buy -/
def num_packs : ℕ := 20

/-- Betty's age -/
def betty_age : ℕ := 50

/-- Doug's age -/
def doug_age : ℕ := 40

/-- Cost of one pack of nuts -/
def pack_cost : ℕ := 100

/-- Total cost Betty wants to spend on nuts -/
def total_cost : ℕ := 2000

theorem betty_nuts_purchase :
  (2 * betty_age = pack_cost) ∧
  (betty_age + doug_age = 90) ∧
  (num_packs * pack_cost = total_cost) →
  num_packs = 20 := by sorry

end NUMINAMATH_CALUDE_betty_nuts_purchase_l1666_166636


namespace NUMINAMATH_CALUDE_wire_length_from_sphere_l1666_166623

/-- The length of a wire formed by melting a sphere -/
theorem wire_length_from_sphere (r : ℝ) (h : r > 0) : 
  (4 / 3 * π * 12^3) = (π * r^2 * ((4 * 12^3) / (3 * r^2))) := by
  sorry

#check wire_length_from_sphere

end NUMINAMATH_CALUDE_wire_length_from_sphere_l1666_166623


namespace NUMINAMATH_CALUDE_dartboard_double_score_angle_l1666_166680

theorem dartboard_double_score_angle (num_regions : ℕ) (probability : ℚ) :
  num_regions = 6 →
  probability = 1 / 8 →
  (360 : ℚ) * probability = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_dartboard_double_score_angle_l1666_166680


namespace NUMINAMATH_CALUDE_charity_dinner_cost_l1666_166675

/-- The total cost of dinners given the number of plates and the cost of rice and chicken per plate -/
def total_cost (num_plates : ℕ) (rice_cost chicken_cost : ℚ) : ℚ :=
  num_plates * (rice_cost + chicken_cost)

/-- Theorem stating that the total cost for 100 plates with rice costing $0.10 and chicken costing $0.40 per plate is $50.00 -/
theorem charity_dinner_cost :
  total_cost 100 (10 / 100) (40 / 100) = 50 := by
  sorry

end NUMINAMATH_CALUDE_charity_dinner_cost_l1666_166675


namespace NUMINAMATH_CALUDE_radical_simplification_l1666_166670

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^2) * Real.sqrt (8 * p) * Real.sqrt (27 * p^5) = 18 * p^4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l1666_166670


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1666_166642

def A : Set ℝ := {x | x^2 - 1 ≥ 0}
def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1666_166642


namespace NUMINAMATH_CALUDE_highway_traffic_l1666_166664

/-- The number of vehicles involved in accidents per 100 million vehicles -/
def accident_rate : ℕ := 96

/-- The total number of vehicles involved in accidents last year -/
def total_accidents : ℕ := 2880

/-- The number of vehicles (in billions) that traveled on the highway last year -/
def vehicles_traveled : ℕ := 3

theorem highway_traffic :
  vehicles_traveled * 1000000000 = (total_accidents * 100000000) / accident_rate := by
  sorry

end NUMINAMATH_CALUDE_highway_traffic_l1666_166664


namespace NUMINAMATH_CALUDE_correct_proposition_l1666_166651

theorem correct_proposition :
  ∀ (p q : Prop),
    (p ∨ q) →
    ¬(p ∧ q) →
    ¬p →
    (p ↔ (5 + 2 = 6)) →
    (q ↔ (6 > 2)) →
    (¬p ∧ q) :=
by sorry

end NUMINAMATH_CALUDE_correct_proposition_l1666_166651


namespace NUMINAMATH_CALUDE_sugar_amount_in_recipe_l1666_166654

/-- A recipe with specified amounts of ingredients -/
structure Recipe where
  flour : ℕ
  salt : ℕ
  sugar : ℕ

/-- The condition that sugar is one more cup than salt -/
def sugar_salt_relation (r : Recipe) : Prop :=
  r.sugar = r.salt + 1

theorem sugar_amount_in_recipe (r : Recipe) 
  (h1 : r.flour = 6)
  (h2 : r.salt = 7)
  (h3 : sugar_salt_relation r) :
  r.sugar = 8 := by
sorry

end NUMINAMATH_CALUDE_sugar_amount_in_recipe_l1666_166654


namespace NUMINAMATH_CALUDE_rectangular_plot_perimeter_l1666_166621

theorem rectangular_plot_perimeter 
  (width : ℝ) 
  (length : ℝ) 
  (perimeter : ℝ) 
  (cost_per_meter : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = width + 10) 
  (h2 : perimeter = 2 * (length + width)) 
  (h3 : cost_per_meter = 6.5) 
  (h4 : total_cost = 2210) 
  (h5 : total_cost = cost_per_meter * perimeter) : 
  perimeter = 340 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_perimeter_l1666_166621


namespace NUMINAMATH_CALUDE_dancing_preference_theorem_l1666_166619

structure DancingPreference where
  like : Rat
  neutral : Rat
  dislike : Rat
  likeSayLike : Rat
  likeSayDislike : Rat
  dislikeSayLike : Rat
  dislikeSayDislike : Rat
  neutralSayLike : Rat
  neutralSayDislike : Rat

/-- The fraction of students who say they dislike dancing but actually like it -/
def fractionLikeSayDislike (pref : DancingPreference) : Rat :=
  (pref.like * pref.likeSayDislike) /
  (pref.like * pref.likeSayDislike + pref.dislike * pref.dislikeSayDislike + pref.neutral * pref.neutralSayDislike)

theorem dancing_preference_theorem (pref : DancingPreference) 
  (h1 : pref.like = 1/2)
  (h2 : pref.neutral = 3/10)
  (h3 : pref.dislike = 1/5)
  (h4 : pref.likeSayLike = 7/10)
  (h5 : pref.likeSayDislike = 3/10)
  (h6 : pref.dislikeSayLike = 1/5)
  (h7 : pref.dislikeSayDislike = 4/5)
  (h8 : pref.neutralSayLike = 2/5)
  (h9 : pref.neutralSayDislike = 3/5)
  : fractionLikeSayDislike pref = 15/49 := by
  sorry

end NUMINAMATH_CALUDE_dancing_preference_theorem_l1666_166619


namespace NUMINAMATH_CALUDE_tonys_normal_temp_l1666_166683

/-- Tony's normal body temperature -/
def normal_temp : ℝ := 95

/-- The fever threshold temperature -/
def fever_threshold : ℝ := 100

/-- Tony's current temperature -/
def current_temp : ℝ := normal_temp + 10

theorem tonys_normal_temp :
  (current_temp = fever_threshold + 5) →
  (fever_threshold = 100) →
  (normal_temp = 95) := by sorry

end NUMINAMATH_CALUDE_tonys_normal_temp_l1666_166683


namespace NUMINAMATH_CALUDE_number_2009_in_group_31_l1666_166618

/-- The sum of squares of the first n odd numbers -/
def O (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 3

/-- The number we're looking for -/
def target : ℕ := 2009

/-- The group number we're proving -/
def group_number : ℕ := 31

theorem number_2009_in_group_31 :
  O (group_number - 1) < target ∧ target ≤ O group_number :=
sorry

end NUMINAMATH_CALUDE_number_2009_in_group_31_l1666_166618


namespace NUMINAMATH_CALUDE_not_perfect_square_l1666_166612

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, 7 * n + 3 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1666_166612


namespace NUMINAMATH_CALUDE_star_six_three_l1666_166696

-- Define the binary operation *
def star (x y : ℝ) : ℝ := 4*x + 5*y - x*y

-- Theorem statement
theorem star_six_three : star 6 3 = 21 := by sorry

end NUMINAMATH_CALUDE_star_six_three_l1666_166696


namespace NUMINAMATH_CALUDE_chocolate_chip_cookie_batches_l1666_166610

/-- Given:
  - Each batch of chocolate chip cookies contains 3 cookies.
  - There are 4 oatmeal cookies.
  - The total number of cookies is 10.
Prove that the number of batches of chocolate chip cookies is 2. -/
theorem chocolate_chip_cookie_batches :
  ∀ (batch_size : ℕ) (oatmeal_cookies : ℕ) (total_cookies : ℕ),
    batch_size = 3 →
    oatmeal_cookies = 4 →
    total_cookies = 10 →
    (total_cookies - oatmeal_cookies) / batch_size = 2 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookie_batches_l1666_166610


namespace NUMINAMATH_CALUDE_perpendicular_iff_m_eq_half_l1666_166698

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line: 2x - y - 1 = 0 -/
def l1 : Line :=
  { a := 2, b := -1, c := -1 }

/-- The second line: mx + y + 1 = 0 -/
def l2 (m : ℝ) : Line :=
  { a := m, b := 1, c := 1 }

/-- The theorem stating the necessary and sufficient condition for perpendicularity -/
theorem perpendicular_iff_m_eq_half :
  ∀ m : ℝ, perpendicular l1 (l2 m) ↔ m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_iff_m_eq_half_l1666_166698


namespace NUMINAMATH_CALUDE_complex_sum_powers_l1666_166643

theorem complex_sum_powers (z : ℂ) (h : z = (1 - Complex.I) / (1 + Complex.I)) :
  z^2 + z^4 + z^6 + z^8 + z^10 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l1666_166643


namespace NUMINAMATH_CALUDE_total_friends_l1666_166659

def friends_in_line (front : ℕ) (back : ℕ) : ℕ :=
  (front - 1) + 1 + (back - 1)

theorem total_friends (seokjin_front : ℕ) (seokjin_back : ℕ) 
  (h1 : seokjin_front = 8) (h2 : seokjin_back = 6) : 
  friends_in_line seokjin_front seokjin_back = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_friends_l1666_166659


namespace NUMINAMATH_CALUDE_line_parametrization_l1666_166697

/-- The slope of the line -/
def m : ℚ := 3/4

/-- The y-intercept of the line -/
def b : ℚ := 3

/-- The x-coordinate of the point on the line when t = 0 -/
def x₀ : ℚ := -9

/-- The y-coordinate of the direction vector -/
def v : ℚ := -7

/-- The equation of the line -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- The parametric equations of the line -/
def param_eq (x y s l t : ℚ) : Prop :=
  x = x₀ + t * l ∧ y = s + t * v

theorem line_parametrization (s l : ℚ) : 
  (∀ t, line_eq (x₀ + t * l) (s + t * v)) ↔ s = -15/4 ∧ l = -28/3 :=
sorry

end NUMINAMATH_CALUDE_line_parametrization_l1666_166697


namespace NUMINAMATH_CALUDE_david_did_more_pushups_l1666_166684

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 35

/-- The difference in push-ups between David and Zachary -/
def pushup_difference : ℕ := david_pushups - zachary_pushups

theorem david_did_more_pushups : pushup_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_david_did_more_pushups_l1666_166684


namespace NUMINAMATH_CALUDE_max_value_is_80_l1666_166620

structure Rock :=
  (weight : ℕ)
  (value : ℕ)

def rock_types : List Rock := [
  ⟨6, 20⟩,
  ⟨3, 9⟩,
  ⟨2, 4⟩
]

def max_weight : ℕ := 24

def min_available : ℕ := 10

def optimal_value (rocks : List Rock) (max_w : ℕ) (min_avail : ℕ) : ℕ :=
  sorry

theorem max_value_is_80 :
  optimal_value rock_types max_weight min_available = 80 :=
sorry

end NUMINAMATH_CALUDE_max_value_is_80_l1666_166620


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l1666_166682

theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 432 → 
  width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l1666_166682


namespace NUMINAMATH_CALUDE_stating_equation_satisfied_l1666_166630

/-- 
Represents a number system with base X.
-/
structure BaseX where
  X : ℕ
  X_ge_two : X ≥ 2

/-- 
Represents a digit in the number system with base X.
-/
def Digit (b : BaseX) := {d : ℕ // d < b.X}

/--
Converts a number represented by digits in base X to its decimal value.
-/
def to_decimal (b : BaseX) (digits : List (Digit b)) : ℕ :=
  digits.foldr (fun d acc => acc * b.X + d.val) 0

/--
Theorem stating that the equation ABBC * CCA = CCCCAC is satisfied
in any base X ≥ 2 when A = 1, B = 0, and C = X - 1 or C = 1.
-/
theorem equation_satisfied (b : BaseX) :
  let A : Digit b := ⟨1, by sorry⟩
  let B : Digit b := ⟨0, by sorry⟩
  let C₁ : Digit b := ⟨b.X - 1, by sorry⟩
  let C₂ : Digit b := ⟨1, by sorry⟩
  (to_decimal b [A, B, B, C₁] * to_decimal b [C₁, C₁, A] = to_decimal b [C₁, C₁, C₁, C₁, A, C₁]) ∧
  (to_decimal b [A, B, B, C₂] * to_decimal b [C₂, C₂, A] = to_decimal b [C₂, C₂, C₂, C₂, A, C₂]) :=
by
  sorry


end NUMINAMATH_CALUDE_stating_equation_satisfied_l1666_166630


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solutions_l1666_166603

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_fourth_sum : a 1 + a 4 = 4
  second_third_product : a 2 * a 3 = 3
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_solutions (seq : ArithmeticSequence) :
  (seq.a 1 = -1 ∧ seq.d = 2 ∧ (∀ n, seq.a n = 2 * n - 3) ∧ (∀ n, S seq n = n^2 - 2*n)) ∨
  (seq.a 1 = 5 ∧ seq.d = -2 ∧ (∀ n, seq.a n = 7 - 2 * n) ∧ (∀ n, S seq n = 6*n - n^2)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solutions_l1666_166603


namespace NUMINAMATH_CALUDE_codecracker_codes_count_l1666_166633

/-- The number of available colors in the CodeCracker game -/
def num_colors : ℕ := 6

/-- The number of slots in a CodeCracker code -/
def code_length : ℕ := 5

/-- The number of possible secret codes in the CodeCracker game -/
def num_codes : ℕ := num_colors * (num_colors - 1)^(code_length - 1)

theorem codecracker_codes_count :
  num_codes = 3750 :=
by sorry

end NUMINAMATH_CALUDE_codecracker_codes_count_l1666_166633


namespace NUMINAMATH_CALUDE_min_value_x2_minus_xy_plus_y2_l1666_166693

theorem min_value_x2_minus_xy_plus_y2 :
  ∀ x y : ℝ, x^2 - x*y + y^2 ≥ 0 ∧ (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_min_value_x2_minus_xy_plus_y2_l1666_166693


namespace NUMINAMATH_CALUDE_arithmetic_progression_reciprocal_l1666_166629

/-- If a, b, and c form an arithmetic progression, and their reciprocals also form an arithmetic progression, then a = b = c. -/
theorem arithmetic_progression_reciprocal (a b c : ℝ) 
  (h1 : b - a = c - b)  -- a, b, c form an arithmetic progression
  (h2 : 1/b - 1/a = 1/c - 1/b)  -- reciprocals form an arithmetic progression
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_reciprocal_l1666_166629


namespace NUMINAMATH_CALUDE_jack_euros_calculation_l1666_166691

/-- Calculates the number of euros Jack has given his total currency amounts -/
theorem jack_euros_calculation (pounds : ℕ) (yen : ℕ) (total_yen : ℕ) 
  (h1 : pounds = 42)
  (h2 : yen = 3000)
  (h3 : total_yen = 9400)
  (h4 : ∀ (e : ℕ), e * 2 * 100 + yen + pounds * 100 = total_yen) :
  ∃ (euros : ℕ), euros = 11 ∧ euros * 2 * 100 + yen + pounds * 100 = total_yen :=
by sorry

end NUMINAMATH_CALUDE_jack_euros_calculation_l1666_166691


namespace NUMINAMATH_CALUDE_complex_power_sum_l1666_166614

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^100 + 1/z^100 = -2 * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1666_166614


namespace NUMINAMATH_CALUDE_twentieth_even_after_15_prove_twentieth_even_after_15_l1666_166661

theorem twentieth_even_after_15 : ℕ → Prop :=
  fun n => (∃ k : ℕ, n = 15 + 2 * k + 2) ∧ 
           (∀ m : ℕ, m > 15 ∧ m < n ∧ Even m → 
             (∃ j : ℕ, j < 20 ∧ m = 15 + 2 * j + 2)) ∧
           (n = 54)

theorem prove_twentieth_even_after_15 : twentieth_even_after_15 54 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_even_after_15_prove_twentieth_even_after_15_l1666_166661


namespace NUMINAMATH_CALUDE_basketball_scores_l1666_166646

theorem basketball_scores (total_players : ℕ) (less_than_yoongi : ℕ) (h1 : total_players = 21) (h2 : less_than_yoongi = 11) :
  total_players - less_than_yoongi - 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_basketball_scores_l1666_166646


namespace NUMINAMATH_CALUDE_square_to_rectangle_area_ratio_l1666_166653

/-- The ratio of the area of a square with side length 30 cm to the area of a rectangle with dimensions 28 cm by 45 cm is 5/7. -/
theorem square_to_rectangle_area_ratio : 
  let square_side : ℝ := 30
  let rect_length : ℝ := 28
  let rect_width : ℝ := 45
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area / rect_area = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_square_to_rectangle_area_ratio_l1666_166653


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l1666_166685

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l1666_166685


namespace NUMINAMATH_CALUDE_triangle_side_difference_l1666_166667

theorem triangle_side_difference (x : ℕ) : 
  (x + 8 > 10) ∧ (x + 10 > 8) ∧ (8 + 10 > x) →
  (∃ (max min : ℕ), 
    (∀ y : ℕ, (y + 8 > 10) ∧ (y + 10 > 8) ∧ (8 + 10 > y) → y ≤ max ∧ y ≥ min) ∧
    (max - min = 14)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l1666_166667


namespace NUMINAMATH_CALUDE_quadratic_equation_has_two_distinct_real_roots_l1666_166644

/-- Proves that the quadratic equation (2kx^2 + 5kx + 2) = 0, where k = 0.64, has two distinct real roots -/
theorem quadratic_equation_has_two_distinct_real_roots :
  let k : ℝ := 0.64
  let a : ℝ := 2 * k
  let b : ℝ := 5 * k
  let c : ℝ := 2
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_has_two_distinct_real_roots_l1666_166644


namespace NUMINAMATH_CALUDE_concentric_circles_equal_areas_l1666_166640

/-- Given a circle of radius R divided by two concentric circles into three equal areas,
    prove that the radii of the concentric circles are R/√3 and R√(2/3) -/
theorem concentric_circles_equal_areas (R : ℝ) (R₁ R₂ : ℝ) (h₁ : R > 0) :
  (π * R₁^2 = π * R^2 / 3) ∧ 
  (π * R₂^2 - π * R₁^2 = π * R^2 / 3) ∧ 
  (π * R^2 - π * R₂^2 = π * R^2 / 3) →
  (R₁ = R / Real.sqrt 3) ∧ (R₂ = R * Real.sqrt (2/3)) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_equal_areas_l1666_166640


namespace NUMINAMATH_CALUDE_line_perpendicular_slope_l1666_166632

/-- Given a line l passing through points (a-2, -1) and (-a-2, 1), perpendicular to a line
    through (-2, 1) with slope -2/3, prove that a = -2/3 -/
theorem line_perpendicular_slope (a : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = ((1 - t) * (a - 2) + t * (-a - 2), (1 - t) * (-1) + t * 1)}
  let m : ℝ := (1 - (-1)) / ((-a - 2) - (a - 2))
  (∀ p ∈ l, (p.2 - 1) = -2/3 * (p.1 - (-2))) → m * (-2/3) = -1 → a = -2/3 := by
sorry

end NUMINAMATH_CALUDE_line_perpendicular_slope_l1666_166632


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1666_166600

/-- Given a line passing through points (1, 2) and (3, 0), 
    prove that the sum of its slope and y-intercept is 2. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (2 = m * 1 + b) → (0 = m * 3 + b) → m + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1666_166600


namespace NUMINAMATH_CALUDE_x_minus_y_equals_ten_l1666_166617

theorem x_minus_y_equals_ten (x y : ℝ) 
  (h1 : 2 = 0.10 * x) 
  (h2 : 2 = 0.20 * y) : 
  x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_ten_l1666_166617


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1666_166658

theorem circle_area_ratio (Q P R : ℝ) (hP : P = 0.5 * Q) (hR : R = 0.75 * Q) :
  (π * (R / 2)^2) / (π * (Q / 2)^2) = 0.140625 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1666_166658


namespace NUMINAMATH_CALUDE_production_problem_l1666_166625

/-- Calculates the production for today given the number of past days, 
    past average production, and new average including today's production. -/
def todaysProduction (n : ℕ) (pastAvg newAvg : ℚ) : ℚ :=
  (n + 1) * newAvg - n * pastAvg

/-- Proves that given the conditions, today's production is 105 units. -/
theorem production_problem (n : ℕ) (pastAvg newAvg : ℚ) 
  (h1 : n = 10)
  (h2 : pastAvg = 50)
  (h3 : newAvg = 55) :
  todaysProduction n pastAvg newAvg = 105 := by
  sorry

#eval todaysProduction 10 50 55

end NUMINAMATH_CALUDE_production_problem_l1666_166625


namespace NUMINAMATH_CALUDE_tennis_balls_cost_l1666_166671

/-- The number of packs of tennis balls Melissa bought -/
def num_packs : ℕ := 4

/-- The number of balls in each pack -/
def balls_per_pack : ℕ := 3

/-- The cost of each tennis ball in dollars -/
def cost_per_ball : ℕ := 2

/-- The total cost of the tennis balls -/
def total_cost : ℕ := num_packs * balls_per_pack * cost_per_ball

theorem tennis_balls_cost : total_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_tennis_balls_cost_l1666_166671


namespace NUMINAMATH_CALUDE_day_318_is_monday_l1666_166674

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a specific day in a year -/
structure DayInYear where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Given that the 45th day of 2003 is a Monday, 
    prove that the 318th day of 2003 is also a Monday -/
theorem day_318_is_monday (d45 d318 : DayInYear) 
  (h1 : d45.dayNumber = 45)
  (h2 : d45.dayOfWeek = DayOfWeek.Monday)
  (h3 : d318.dayNumber = 318) :
  d318.dayOfWeek = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_day_318_is_monday_l1666_166674


namespace NUMINAMATH_CALUDE_smallest_triangle_angle_function_range_l1666_166681

theorem smallest_triangle_angle_function_range :
  ∀ x : Real,
  0 < x → x ≤ Real.pi / 3 →
  let y := (Real.sin x * Real.cos x + 1) / (Real.sin x + Real.cos x)
  ∃ (a b : Real), a = 3/2 ∧ b = 3 * Real.sqrt 2 / 4 ∧
  (∀ z, y = z → a < z ∧ z ≤ b) ∧
  (∀ ε > 0, ∃ z, y = z ∧ z < a + ε) ∧
  (∃ z, y = z ∧ z = b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_triangle_angle_function_range_l1666_166681


namespace NUMINAMATH_CALUDE_two_distinct_roots_for_all_m_m_value_when_root_sum_condition_l1666_166606

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4*eq.a*eq.c

/-- Checks if a quadratic equation has two distinct real roots -/
def has_two_distinct_real_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq > 0

/-- Our specific quadratic equation x^2 - 2x - 3m^2 = 0 -/
def our_equation (m : ℝ) : QuadraticEquation :=
  { a := 1, b := -2, c := -3*m^2 }

theorem two_distinct_roots_for_all_m (m : ℝ) :
  has_two_distinct_real_roots (our_equation m) := by
  sorry

theorem m_value_when_root_sum_condition (m : ℝ) (α β : ℝ)
  (h1 : α + β = 2)
  (h2 : α + 2*β = 5)
  (h3 : α * β = -(-3*m^2)) :
  m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_roots_for_all_m_m_value_when_root_sum_condition_l1666_166606


namespace NUMINAMATH_CALUDE_max_min_values_l1666_166656

theorem max_min_values (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (∀ x y, 0 < x ∧ 0 < y ∧ x + y = 1 → Real.sqrt x + Real.sqrt y ≤ Real.sqrt 2) ∧
  (a^2 + b^2 ≥ 1/2) ∧
  (1 / (a + 2*b) + 1 / (2*a + b) ≥ 4/3) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_l1666_166656


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_19_l1666_166647

theorem remainder_of_3_pow_19 : 3^19 % 1162261460 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_19_l1666_166647


namespace NUMINAMATH_CALUDE_pieces_per_package_calculation_l1666_166604

/-- Given the number of gum packages, candy packages, and total pieces,
    calculate the number of pieces per package. -/
def pieces_per_package (gum_packages : ℕ) (candy_packages : ℕ) (total_pieces : ℕ) : ℚ :=
  total_pieces / (gum_packages + candy_packages)

/-- Theorem stating that with 28 gum packages, 14 candy packages, and 7 total pieces,
    the number of pieces per package is 1/6. -/
theorem pieces_per_package_calculation :
  pieces_per_package 28 14 7 = 1/6 := by
  sorry

#eval pieces_per_package 28 14 7

end NUMINAMATH_CALUDE_pieces_per_package_calculation_l1666_166604


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1666_166622

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The arithmetic sequence condition -/
def ArithmeticCondition (a : ℕ → ℝ) : Prop :=
  2 * a 1 - a 3 / 2 = a 2 - (a 3 / 2)

theorem geometric_sequence_ratio (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_arith : ArithmeticCondition a) 
  (h_pos : ∀ n, a n > 0) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1666_166622


namespace NUMINAMATH_CALUDE_subsequence_theorem_l1666_166657

theorem subsequence_theorem (seq : List ℕ) (h1 : seq.length = 101) 
  (h2 : ∀ n, n ∈ seq → 1 ≤ n ∧ n ≤ 101) 
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 101 → n ∈ seq) :
  ∃ subseq : List ℕ, subseq.length = 11 ∧ 
    (∀ i j, i < j → subseq.get ⟨i, by sorry⟩ < subseq.get ⟨j, by sorry⟩) ∨
    (∀ i j, i < j → subseq.get ⟨i, by sorry⟩ > subseq.get ⟨j, by sorry⟩) :=
sorry

end NUMINAMATH_CALUDE_subsequence_theorem_l1666_166657


namespace NUMINAMATH_CALUDE_race_completion_time_l1666_166605

/-- Given a 1000-meter race where runner A beats runner B by either 60 meters or 10 seconds,
    this theorem proves that runner A completes the race in 156.67 seconds. -/
theorem race_completion_time :
  ∀ (speed_A speed_B : ℝ),
  speed_A > 0 ∧ speed_B > 0 →
  1000 / speed_A = 940 / speed_B →
  1000 / speed_A = (1000 / speed_B) - 10 →
  1000 / speed_A = 156.67 :=
by sorry

end NUMINAMATH_CALUDE_race_completion_time_l1666_166605


namespace NUMINAMATH_CALUDE_sodium_bicarbonate_moles_l1666_166631

-- Define the chemical reaction
structure Reaction where
  hcl : ℝ
  nahco3 : ℝ
  nacl : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.hcl = r.nahco3 ∧ r.hcl = r.nacl

-- Theorem statement
theorem sodium_bicarbonate_moles 
  (r : Reaction) 
  (h1 : r.hcl = 1) 
  (h2 : r.nacl = 1) 
  (h3 : balanced_equation r) : 
  r.nahco3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sodium_bicarbonate_moles_l1666_166631


namespace NUMINAMATH_CALUDE_tabitha_honey_nights_l1666_166672

/-- The number of nights Tabitha can enjoy honey in her tea before bed -/
def honey_nights (servings_per_cup : ℕ) (cups_per_night : ℕ) (container_size : ℕ) (servings_per_ounce : ℕ) : ℕ :=
  (container_size * servings_per_ounce) / (servings_per_cup * cups_per_night)

/-- Theorem stating that Tabitha can enjoy honey in her tea for 48 nights before bed -/
theorem tabitha_honey_nights :
  honey_nights 1 2 16 6 = 48 := by sorry

end NUMINAMATH_CALUDE_tabitha_honey_nights_l1666_166672
