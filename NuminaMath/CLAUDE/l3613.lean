import Mathlib

namespace NUMINAMATH_CALUDE_decagon_vertex_sum_l3613_361375

theorem decagon_vertex_sum (π : Fin 10 → Fin 10) 
  (hπ : Function.Bijective π) :
  ∃ k : Fin 10, 
    π k + π ((k + 9) % 10) + π ((k + 1) % 10) ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_decagon_vertex_sum_l3613_361375


namespace NUMINAMATH_CALUDE_min_value_theorem_l3613_361344

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  (2 * x^2 - x + 1) / (x * y) ≥ 2 * Real.sqrt 2 + 1 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧
    (2 * x₀^2 - x₀ + 1) / (x₀ * y₀) = 2 * Real.sqrt 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3613_361344


namespace NUMINAMATH_CALUDE_expression_value_l3613_361336

theorem expression_value (x y : ℤ) (hx : x = -5) (hy : y = 8) :
  2 * (x - y)^2 - x * y = 378 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3613_361336


namespace NUMINAMATH_CALUDE_substitution_insufficient_for_identity_proof_l3613_361327

/-- A mathematical identity is an equality that holds for all values of the variables involved. -/
def MathematicalIdentity (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x

/-- Substitution method verifies if an expression holds true for particular values. -/
def SubstitutionMethod (f g : ℝ → ℝ) (values : Set ℝ) : Prop :=
  ∀ x ∈ values, f x = g x

/-- Theorem: Substituting numerical values is insufficient to conclusively prove an identity. -/
theorem substitution_insufficient_for_identity_proof :
  ∃ (f g : ℝ → ℝ) (values : Set ℝ), 
    SubstitutionMethod f g values ∧ ¬MathematicalIdentity f g :=
  sorry

#check substitution_insufficient_for_identity_proof

end NUMINAMATH_CALUDE_substitution_insufficient_for_identity_proof_l3613_361327


namespace NUMINAMATH_CALUDE_dance_team_quitters_l3613_361361

theorem dance_team_quitters (initial_members : ℕ) (new_members : ℕ) (final_members : ℕ) 
  (h1 : initial_members = 25)
  (h2 : new_members = 13)
  (h3 : final_members = 30)
  : initial_members - (initial_members - final_members + new_members) = 8 := by
  sorry

end NUMINAMATH_CALUDE_dance_team_quitters_l3613_361361


namespace NUMINAMATH_CALUDE_lcm_14_25_l3613_361340

theorem lcm_14_25 : Nat.lcm 14 25 = 350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_14_25_l3613_361340


namespace NUMINAMATH_CALUDE_max_squares_is_seven_l3613_361373

/-- A shape formed by unit-length sticks on a plane -/
structure StickShape where
  sticks : ℕ
  squares : ℕ
  rows : ℕ
  first_row_squares : ℕ

/-- Predicate to check if a shape is valid according to the problem constraints -/
def is_valid_shape (s : StickShape) : Prop :=
  s.sticks = 20 ∧
  s.rows ≥ 1 ∧
  s.first_row_squares ≥ 1 ∧
  s.first_row_squares ≤ s.squares ∧
  (s.squares - s.first_row_squares) % (s.rows - 1) = 0

/-- The maximum number of squares that can be formed -/
def max_squares : ℕ := 7

/-- Theorem stating that the maximum number of squares is 7 -/
theorem max_squares_is_seven :
  ∀ s : StickShape, is_valid_shape s → s.squares ≤ max_squares :=
sorry

end NUMINAMATH_CALUDE_max_squares_is_seven_l3613_361373


namespace NUMINAMATH_CALUDE_max_pieces_is_seven_l3613_361386

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.length digits = List.length (List.dedup digits)

theorem max_pieces_is_seven :
  (∃ (max : ℕ), 
    (∀ (n : ℕ), ∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * n → n ≤ max) ∧
    (∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * max)) ∧
  (∀ (m : ℕ), 
    (∀ (n : ℕ), ∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * n → n ≤ m) ∧
    (∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * m) → 
    m ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_is_seven_l3613_361386


namespace NUMINAMATH_CALUDE_average_of_ABC_l3613_361349

theorem average_of_ABC (A B C : ℝ) 
  (eq1 : 501 * C - 1002 * A = 2002)
  (eq2 : 501 * B + 2002 * A = 2505) :
  (A + B + C) / 3 = -A / 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_ABC_l3613_361349


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3613_361338

-- Define sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2*x > 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x < 3/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3613_361338


namespace NUMINAMATH_CALUDE_probability_sum_10_l3613_361365

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're looking for -/
def targetSum : ℕ := 10

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (sum of 10) -/
def favorableOutcomes : ℕ := 24

/-- The probability of rolling a sum of 10 with three standard six-sided dice -/
theorem probability_sum_10 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_probability_sum_10_l3613_361365


namespace NUMINAMATH_CALUDE_woods_width_l3613_361304

theorem woods_width (area : ℝ) (length : ℝ) (width : ℝ) 
  (h1 : area = 24) 
  (h2 : length = 3) 
  (h3 : area = length * width) : width = 8 := by
sorry

end NUMINAMATH_CALUDE_woods_width_l3613_361304


namespace NUMINAMATH_CALUDE_problem_solution_l3613_361351

theorem problem_solution : ∃ x : ℝ, 550 - (x / 20.8) = 545 ∧ x = 104 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3613_361351


namespace NUMINAMATH_CALUDE_difference_of_squares_l3613_361313

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3613_361313


namespace NUMINAMATH_CALUDE_daisy_count_proof_l3613_361334

theorem daisy_count_proof (white : ℕ) (pink : ℕ) (red : ℕ) 
  (h1 : white = 6)
  (h2 : pink = 9 * white)
  (h3 : red = 4 * pink - 3) :
  white + pink + red = 273 := by
  sorry

end NUMINAMATH_CALUDE_daisy_count_proof_l3613_361334


namespace NUMINAMATH_CALUDE_shaded_area_problem_l3613_361332

/-- Given a square FGHI with area 80 and points J, K, L, M on its sides
    such that FK = GL = HM = IJ and FK = 3KG, 
    the area of the quadrilateral JKLM is 50. -/
theorem shaded_area_problem (F G H I J K L M : ℝ × ℝ) : 
  (∃ s : ℝ, s > 0 ∧ (G.1 - F.1)^2 + (G.2 - F.2)^2 = s^2 ∧ s^2 = 80) →
  (K.1 - F.1)^2 + (K.2 - F.2)^2 = (L.1 - G.1)^2 + (L.2 - G.2)^2 ∧
   (L.1 - G.1)^2 + (L.2 - G.2)^2 = (M.1 - H.1)^2 + (M.2 - H.2)^2 ∧
   (M.1 - H.1)^2 + (M.2 - H.2)^2 = (J.1 - I.1)^2 + (J.2 - I.2)^2 →
  (K.1 - F.1)^2 + (K.2 - F.2)^2 = 9 * ((G.1 - K.1)^2 + (G.2 - K.2)^2) →
  (K.1 - J.1)^2 + (K.2 - J.2)^2 = 50 :=
by sorry


end NUMINAMATH_CALUDE_shaded_area_problem_l3613_361332


namespace NUMINAMATH_CALUDE_seahorse_penguin_ratio_l3613_361370

theorem seahorse_penguin_ratio :
  let seahorses : ℕ := 70
  let penguins : ℕ := seahorses + 85
  (seahorses : ℚ) / penguins = 14 / 31 := by
  sorry

end NUMINAMATH_CALUDE_seahorse_penguin_ratio_l3613_361370


namespace NUMINAMATH_CALUDE_unique_solution_l3613_361345

theorem unique_solution : ∃! (x : ℝ), 
  x > 0 ∧ 
  (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ∧ 
  x^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3613_361345


namespace NUMINAMATH_CALUDE_drum_capacity_ratio_l3613_361314

theorem drum_capacity_ratio (capacity_x capacity_y : ℝ) 
  (h1 : capacity_x > 0) 
  (h2 : capacity_y > 0) 
  (h3 : (1/2 : ℝ) * capacity_x + (2/5 : ℝ) * capacity_y = (65/100 : ℝ) * capacity_y) : 
  capacity_y / capacity_x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_drum_capacity_ratio_l3613_361314


namespace NUMINAMATH_CALUDE_pushup_sequence_sum_l3613_361356

theorem pushup_sequence_sum (a : ℕ → ℕ) :
  (a 0 = 10) →
  (∀ n : ℕ, a (n + 1) = a n + 5) →
  (a 0 + a 1 + a 2 = 45) := by
  sorry

end NUMINAMATH_CALUDE_pushup_sequence_sum_l3613_361356


namespace NUMINAMATH_CALUDE_point_relationship_l3613_361374

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 5

-- Define the points
def point1 : ℝ × ℝ := (-4, f (-4))
def point2 : ℝ × ℝ := (-1, f (-1))
def point3 : ℝ × ℝ := (2, f 2)

-- Theorem statement
theorem point_relationship :
  let y₁ := point1.2
  let y₂ := point2.2
  let y₃ := point3.2
  y₂ > y₃ ∧ y₃ > y₁ := by sorry

end NUMINAMATH_CALUDE_point_relationship_l3613_361374


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3613_361355

-- Define the quadratic function
def f (x : ℝ) := -2 * x^2 + x + 1

-- Define the solution set
def solution_set := {x : ℝ | -1/2 < x ∧ x < 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3613_361355


namespace NUMINAMATH_CALUDE_pizza_combinations_l3613_361395

theorem pizza_combinations (n m : ℕ) (h1 : n = 8) (h2 : m = 5) : 
  Nat.choose n m = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3613_361395


namespace NUMINAMATH_CALUDE_negation_equivalence_angle_sine_equivalence_l3613_361391

-- Define the proposition for the first part
def P (x : ℝ) : Prop := x^2 - x > 0

-- Theorem for the first part
theorem negation_equivalence : (¬ ∃ x, P x) ↔ (∀ x, ¬(P x)) := by sorry

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Theorem for the second part
theorem angle_sine_equivalence (t : Triangle) : t.A > t.B ↔ Real.sin t.A > Real.sin t.B := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_angle_sine_equivalence_l3613_361391


namespace NUMINAMATH_CALUDE_matrix_multiplication_l3613_361341

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -2; -1, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 3; 2, -2]

theorem matrix_multiplication :
  A * B = !![(-4), 16; 10, (-13)] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l3613_361341


namespace NUMINAMATH_CALUDE_divisor_calculation_l3613_361381

theorem divisor_calculation (quotient dividend : ℚ) (h1 : quotient = -5/16) (h2 : dividend = -5/2) :
  dividend / quotient = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisor_calculation_l3613_361381


namespace NUMINAMATH_CALUDE_profit_without_discount_l3613_361335

theorem profit_without_discount (discount_percent : ℝ) (profit_with_discount_percent : ℝ) :
  discount_percent = 5 →
  profit_with_discount_percent = 20.65 →
  let cost_price := 100
  let selling_price_with_discount := cost_price * (1 - discount_percent / 100)
  let profit := cost_price * profit_with_discount_percent / 100
  let selling_price_without_discount := cost_price + profit
  profit / cost_price * 100 = 20.65 :=
by sorry

end NUMINAMATH_CALUDE_profit_without_discount_l3613_361335


namespace NUMINAMATH_CALUDE_min_fence_posts_for_grazing_area_l3613_361390

/-- Calculates the number of fence posts required for a rectangular grazing area -/
def fence_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let long_side_posts := length / post_spacing + 1
  let short_side_posts := 2 * (width / post_spacing)
  long_side_posts + short_side_posts

/-- Theorem stating the minimum number of fence posts required for the given conditions -/
theorem min_fence_posts_for_grazing_area :
  fence_posts 80 40 10 = 17 :=
sorry

end NUMINAMATH_CALUDE_min_fence_posts_for_grazing_area_l3613_361390


namespace NUMINAMATH_CALUDE_train_passing_time_train_passing_man_time_l3613_361347

/-- The time it takes for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_ms := relative_speed * (1000 / 3600)
  train_length / relative_speed_ms

/-- Proof that the time for a 110m train moving at 40 km/h to pass a man moving at 4 km/h in the opposite direction is approximately 8.99 seconds -/
theorem train_passing_man_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_passing_time 110 40 4 - 8.99| < ε :=
sorry

end NUMINAMATH_CALUDE_train_passing_time_train_passing_man_time_l3613_361347


namespace NUMINAMATH_CALUDE_a_10_value_l3613_361337

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_10_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  (a 1)^2 - 10 * (a 1) + 16 = 0 →
  (a 19)^2 - 10 * (a 19) + 16 = 0 →
  a 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l3613_361337


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3613_361300

theorem simplify_trig_expression :
  (Real.sin (30 * π / 180) + Real.sin (50 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (50 * π / 180)) =
  Real.tan (40 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3613_361300


namespace NUMINAMATH_CALUDE_unique_solution_is_five_l3613_361366

/-- The function f(x) = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x - 3

/-- The theorem stating that x = 5 is the unique solution to the equation -/
theorem unique_solution_is_five :
  ∃! x : ℝ, 2 * (f x) - 11 = f (x - 2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_five_l3613_361366


namespace NUMINAMATH_CALUDE_sum_of_four_twos_to_fourth_l3613_361326

theorem sum_of_four_twos_to_fourth (n : ℕ) : 
  (2^4 : ℕ) + (2^4 : ℕ) + (2^4 : ℕ) + (2^4 : ℕ) = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_twos_to_fourth_l3613_361326


namespace NUMINAMATH_CALUDE_texas_tech_game_profit_l3613_361379

/-- Represents the discount tiers for t-shirt sales -/
inductive DiscountTier
  | NoDiscount
  | MediumDiscount
  | HighDiscount

/-- Calculates the discount tier based on the number of t-shirts sold -/
def getDiscountTier (numSold : ℕ) : DiscountTier :=
  if numSold ≤ 50 then DiscountTier.NoDiscount
  else if numSold ≤ 100 then DiscountTier.MediumDiscount
  else DiscountTier.HighDiscount

/-- Calculates the profit per t-shirt based on the discount tier -/
def getProfitPerShirt (tier : DiscountTier) (fullPrice : ℕ) : ℕ :=
  match tier with
  | DiscountTier.NoDiscount => fullPrice
  | DiscountTier.MediumDiscount => fullPrice - 5
  | DiscountTier.HighDiscount => fullPrice - 10

/-- Theorem: The money made from selling t-shirts during the Texas Tech game is $1092 -/
theorem texas_tech_game_profit (totalSold arkansasSold fullPrice : ℕ) 
    (h1 : totalSold = 186)
    (h2 : arkansasSold = 172)
    (h3 : fullPrice = 78) :
    let texasTechSold := totalSold - arkansasSold
    let tier := getDiscountTier texasTechSold
    let profitPerShirt := getProfitPerShirt tier fullPrice
    texasTechSold * profitPerShirt = 1092 := by
  sorry

end NUMINAMATH_CALUDE_texas_tech_game_profit_l3613_361379


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l3613_361323

/-- Given a quadratic equation 4x^2 - 8x - 320 = 0, prove that when transformed
    into the form (x+p)^2 = q by completing the square, the value of q is 81. -/
theorem complete_square_quadratic :
  ∃ (p : ℝ), ∀ (x : ℝ),
    (4 * x^2 - 8 * x - 320 = 0) ↔ ((x + p)^2 = 81) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l3613_361323


namespace NUMINAMATH_CALUDE_cats_remaining_l3613_361333

theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 13 → house = 5 → sold = 10 → siamese + house - sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l3613_361333


namespace NUMINAMATH_CALUDE_locus_is_conic_locus_degenerate_line_locus_circle_l3613_361319

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in the first quadrant -/
structure Square where
  a : ℝ
  A : Point
  B : Point

/-- Defines the locus of a point P relative to the square -/
def locus (s : Square) (P : Point) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ θ : ℝ, 
    x = P.x * Real.sin θ + (s.a - P.x) * Real.cos θ ∧
    y = (s.a - P.y) * Real.sin θ + P.y * Real.cos θ}

theorem locus_is_conic (s : Square) (P : Point) 
  (h1 : s.A.y = 0 ∧ s.B.x = 0)  -- A is on x-axis, B is on y-axis
  (h2 : 0 ≤ P.x ∧ P.x ≤ 2*s.a ∧ 0 ≤ P.y ∧ P.y ≤ 2*s.a)  -- P is inside or on the square
  : ∃ (A B C D E F : ℝ), 
    A * P.x^2 + B * P.x * P.y + C * P.y^2 + D * P.x + E * P.y + F = 0 :=
sorry

theorem locus_degenerate_line (s : Square) (P : Point)
  (h : P.y = P.x)  -- P is on the diagonal
  : ∃ (m b : ℝ), ∀ (x y : ℝ), (x, y) ∈ locus s P → y = m * x + b :=
sorry

theorem locus_circle (s : Square) (P : Point)
  (h : P.x = s.a ∧ P.y = 0)  -- P is at midpoint of AB
  : ∃ (c : Point) (r : ℝ), ∀ (x y : ℝ), 
    (x, y) ∈ locus s P → (x - c.x)^2 + (y - c.y)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_locus_is_conic_locus_degenerate_line_locus_circle_l3613_361319


namespace NUMINAMATH_CALUDE_abc_remainder_mod_9_l3613_361371

theorem abc_remainder_mod_9 (a b c : ℕ) : 
  a < 9 → b < 9 → c < 9 →
  (a + 2*b + 3*c) % 9 = 1 →
  (2*a + 3*b + c) % 9 = 2 →
  (3*a + b + 2*c) % 9 = 3 →
  (a * b * c) % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_remainder_mod_9_l3613_361371


namespace NUMINAMATH_CALUDE_value_of_expression_l3613_361346

theorem value_of_expression (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (x + y) = 95 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l3613_361346


namespace NUMINAMATH_CALUDE_alice_unanswered_questions_l3613_361396

/-- Represents the scoring systems and Alice's results in a math competition. -/
structure MathCompetition where
  total_questions : ℕ
  new_correct_points : ℕ
  new_incorrect_points : ℕ
  new_unanswered_points : ℕ
  old_start_points : ℕ
  old_correct_points : ℕ
  old_incorrect_points : Int
  old_unanswered_points : ℕ
  new_score : ℕ
  old_score : ℕ

/-- Calculates the number of unanswered questions in the math competition. -/
def calculate_unanswered_questions (comp : MathCompetition) : ℕ :=
  sorry

/-- Theorem stating that Alice left 2 questions unanswered. -/
theorem alice_unanswered_questions (comp : MathCompetition)
  (h1 : comp.total_questions = 30)
  (h2 : comp.new_correct_points = 4)
  (h3 : comp.new_incorrect_points = 0)
  (h4 : comp.new_unanswered_points = 1)
  (h5 : comp.old_start_points = 20)
  (h6 : comp.old_correct_points = 3)
  (h7 : comp.old_incorrect_points = -1)
  (h8 : comp.old_unanswered_points = 0)
  (h9 : comp.new_score = 87)
  (h10 : comp.old_score = 75) :
  calculate_unanswered_questions comp = 2 := by
  sorry

end NUMINAMATH_CALUDE_alice_unanswered_questions_l3613_361396


namespace NUMINAMATH_CALUDE_additional_land_cost_l3613_361321

/-- Calculates the cost of additional land purchased by Carlson -/
theorem additional_land_cost (initial_area : ℝ) (final_area : ℝ) (cost_per_sqm : ℝ) :
  initial_area = 300 →
  final_area = 900 →
  cost_per_sqm = 20 →
  (final_area - initial_area) * cost_per_sqm = 12000 := by
  sorry

#check additional_land_cost

end NUMINAMATH_CALUDE_additional_land_cost_l3613_361321


namespace NUMINAMATH_CALUDE_money_difference_l3613_361301

/-- The problem statement about Isabella, Sam, and Giselle's money --/
theorem money_difference (isabella sam giselle : ℕ) : 
  isabella = sam + 45 →  -- Isabella has $45 more than Sam
  giselle = 120 →  -- Giselle has $120
  isabella + sam + giselle = 3 * 115 →  -- Total money shared equally among 3 shoppers
  isabella - giselle = 15 :=  -- Isabella has $15 more than Giselle
by sorry

end NUMINAMATH_CALUDE_money_difference_l3613_361301


namespace NUMINAMATH_CALUDE_completing_square_transformation_l3613_361320

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l3613_361320


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3613_361307

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 6

-- Define the solution set condition
def solution_set (a b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

-- Define the theorem
theorem quadratic_inequality_theorem (a b : ℝ) :
  (∀ x, f a x > 4 ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = 2) ∧
  (∀ c, 
    let g (x : ℝ) := a * x^2 - (a * c + b) * x + b * c
    if c > 2 then
      {x | g x < 0} = {x | 2 < x ∧ x < c}
    else if c < 2 then
      {x | g x < 0} = {x | c < x ∧ x < 2}
    else
      {x | g x < 0} = ∅) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3613_361307


namespace NUMINAMATH_CALUDE_special_polyhedron_properties_l3613_361348

/-- A convex polyhedron with triangular and hexagonal faces -/
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex

/-- The properties of our specific polyhedron -/
def special_polyhedron : Polyhedron where
  V := 50
  E := 78
  F := 30
  t := 8
  h := 22
  T := 2
  H := 2

/-- Theorem stating the properties of the special polyhedron -/
theorem special_polyhedron_properties (p : Polyhedron) 
  (h1 : p.V - p.E + p.F = 2)  -- Euler's formula
  (h2 : p.F = 30)
  (h3 : p.F = p.t + p.h)
  (h4 : p.T = 2)
  (h5 : p.H = 2)
  (h6 : p.t = 8)
  (h7 : p.h = 22)
  (h8 : p.E = (3 * p.t + 6 * p.h) / 2) :
  100 * p.H + 10 * p.T + p.V = 270 := by
  sorry

#check special_polyhedron_properties

end NUMINAMATH_CALUDE_special_polyhedron_properties_l3613_361348


namespace NUMINAMATH_CALUDE_general_term_k_n_l3613_361383

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  a2_geometric_mean : a 2 ^ 2 = a 1 * a 4
  geometric_subseq : ∀ n, a (3^n) / a (3^(n-1)) = a 3 / a 1

/-- The theorem stating the general term of k_n -/
theorem general_term_k_n (seq : ArithmeticSequence) : 
  ∀ n : ℕ, ∃ k_n : ℕ, seq.a k_n = seq.a 1 * (3 : ℝ)^(n-1) ∧ k_n = 3^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_general_term_k_n_l3613_361383


namespace NUMINAMATH_CALUDE_summer_camp_duration_l3613_361352

def summer_camp (n : ℕ) (k : ℕ) (d : ℕ) : Prop :=
  -- n is the number of participants
  -- k is the number of participants chosen each day
  -- d is the number of days
  n = 15 ∧ 
  k = 3 ∧
  Nat.choose n 2 = d * Nat.choose k 2

theorem summer_camp_duration : 
  ∃ d : ℕ, summer_camp 15 3 d ∧ d = 35 := by
  sorry

end NUMINAMATH_CALUDE_summer_camp_duration_l3613_361352


namespace NUMINAMATH_CALUDE_cristobal_beatrix_pages_difference_l3613_361392

theorem cristobal_beatrix_pages_difference (beatrix_pages cristobal_extra_pages : ℕ) 
  (h1 : beatrix_pages = 704)
  (h2 : cristobal_extra_pages = 1423) :
  (beatrix_pages + cristobal_extra_pages) - (3 * beatrix_pages) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cristobal_beatrix_pages_difference_l3613_361392


namespace NUMINAMATH_CALUDE_coin_toss_experiment_l3613_361363

theorem coin_toss_experiment (total_tosses : ℕ) (heads_frequency : ℚ) (tails_count : ℕ) :
  total_tosses = 100 →
  heads_frequency = 49/100 →
  tails_count = total_tosses - (total_tosses * heads_frequency).num →
  tails_count = 51 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_experiment_l3613_361363


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l3613_361353

theorem quadratic_completion_of_square (x : ℝ) : 
  2 * x^2 - 4 * x - 3 = 0 ↔ (x - 1)^2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l3613_361353


namespace NUMINAMATH_CALUDE_surface_area_ratio_l3613_361325

-- Define the side length of the cube
variable (s : ℝ)

-- Define the surface area of a cube
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the surface area of the rectangular solid
def rectangular_solid_surface_area (s : ℝ) : ℝ := 2 * (2*s*s + 2*s*s + s*s)

-- Theorem statement
theorem surface_area_ratio :
  (cube_surface_area s) / (rectangular_solid_surface_area s) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_ratio_l3613_361325


namespace NUMINAMATH_CALUDE_flooring_boxes_needed_l3613_361393

/-- Calculates the number of flooring boxes needed to complete a room -/
theorem flooring_boxes_needed
  (room_length : ℝ)
  (room_width : ℝ)
  (area_covered : ℝ)
  (area_per_box : ℝ)
  (h1 : room_length = 16)
  (h2 : room_width = 20)
  (h3 : area_covered = 250)
  (h4 : area_per_box = 10)
  : ⌈(room_length * room_width - area_covered) / area_per_box⌉ = 7 := by
  sorry

end NUMINAMATH_CALUDE_flooring_boxes_needed_l3613_361393


namespace NUMINAMATH_CALUDE_cubic_equation_one_root_l3613_361315

theorem cubic_equation_one_root (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ x^3 - a*x^2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_one_root_l3613_361315


namespace NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l3613_361354

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4 * x^2 + 5 * x

-- Define the three points
def p1 : ℝ × ℝ := (0, 0)
def p2 : ℝ × ℝ := (-1, -1)
def p3 : ℝ × ℝ := (1, 9)

-- Theorem statement
theorem quadratic_function_passes_through_points :
  f p1.1 = p1.2 ∧ f p2.1 = p2.2 ∧ f p3.1 = p3.2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l3613_361354


namespace NUMINAMATH_CALUDE_patrol_results_l3613_361310

/-- Represents the patrol records of the police car --/
def patrol_records : List Int := [6, -8, 9, -5, 4, -3]

/-- Fuel consumption rate in liters per kilometer --/
def fuel_consumption_rate : ℚ := 0.2

/-- Initial fuel in the tank in liters --/
def initial_fuel : ℚ := 5

/-- Calculates the final position of the police car --/
def final_position (records : List Int) : Int :=
  records.sum

/-- Calculates the total distance traveled --/
def total_distance (records : List Int) : Int :=
  records.map (abs) |>.sum

/-- Calculates the total fuel consumed --/
def total_fuel_consumed (distance : Int) (rate : ℚ) : ℚ :=
  (distance : ℚ) * rate

/-- Calculates the additional fuel needed --/
def additional_fuel_needed (consumed : ℚ) (initial : ℚ) : ℚ :=
  max (consumed - initial) 0

theorem patrol_results :
  (final_position patrol_records = 3) ∧
  (total_fuel_consumed (total_distance patrol_records) fuel_consumption_rate = 7) ∧
  (additional_fuel_needed (total_fuel_consumed (total_distance patrol_records) fuel_consumption_rate) initial_fuel = 2) :=
by sorry

end NUMINAMATH_CALUDE_patrol_results_l3613_361310


namespace NUMINAMATH_CALUDE_equation_solution_l3613_361318

theorem equation_solution : ∃ x : ℚ, (x - 1) / 2 = 1 - (3 * x + 2) / 5 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3613_361318


namespace NUMINAMATH_CALUDE_wrapping_paper_area_correct_l3613_361309

/-- Represents a box with square base -/
structure Box where
  w : ℝ  -- width of the base
  h : ℝ  -- height of the box

/-- Calculates the area of the wrapping paper for a given box -/
def wrappingPaperArea (box : Box) : ℝ :=
  6 * box.w * box.h + box.h^2

/-- Theorem stating that the area of the wrapping paper is correct -/
theorem wrapping_paper_area_correct (box : Box) :
  wrappingPaperArea box = 6 * box.w * box.h + box.h^2 :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_correct_l3613_361309


namespace NUMINAMATH_CALUDE_tip_percentage_calculation_l3613_361312

theorem tip_percentage_calculation (total_bill : ℝ) (food_price : ℝ) (tax_rate : ℝ) : 
  total_bill = 198 →
  food_price = 150 →
  tax_rate = 0.1 →
  (total_bill - food_price * (1 + tax_rate)) / (food_price * (1 + tax_rate)) = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_tip_percentage_calculation_l3613_361312


namespace NUMINAMATH_CALUDE_residue_of_15_power_1234_mod_19_l3613_361394

theorem residue_of_15_power_1234_mod_19 :
  (15 : ℤ)^1234 ≡ 6 [ZMOD 19] := by sorry

end NUMINAMATH_CALUDE_residue_of_15_power_1234_mod_19_l3613_361394


namespace NUMINAMATH_CALUDE_percentage_relation_l3613_361360

theorem percentage_relation (x y z : ℝ) 
  (h1 : x = 0.2 * y) 
  (h2 : x = 0.5 * z) : 
  z = 0.4 * y := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3613_361360


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3613_361330

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (1 + 2*i) / (1 - i)
  Complex.im z = 3/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3613_361330


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l3613_361384

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_15 :
  last_two_digits (sum_factorials 15) = 13 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l3613_361384


namespace NUMINAMATH_CALUDE_tournament_matches_divisible_by_two_l3613_361324

/-- Represents a single elimination tennis tournament -/
structure TennisTournament where
  total_players : ℕ
  bye_players : ℕ
  first_round_players : ℕ

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : TennisTournament) : ℕ :=
  t.total_players - 1

/-- Theorem: The total number of matches in the specified tournament is divisible by 2 -/
theorem tournament_matches_divisible_by_two :
  ∃ (t : TennisTournament), 
    t.total_players = 128 ∧ 
    t.bye_players = 32 ∧ 
    t.first_round_players = 96 ∧ 
    ∃ (k : ℕ), total_matches t = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_tournament_matches_divisible_by_two_l3613_361324


namespace NUMINAMATH_CALUDE_weather_period_days_l3613_361359

/-- Represents the weather conditions over a period of time. -/
structure WeatherPeriod where
  totalRainyDays : ℕ
  clearEvenings : ℕ
  clearMornings : ℕ
  morningRainImpliesClearEvening : Unit
  eveningRainImpliesClearMorning : Unit

/-- Calculates the total number of days in the weather period. -/
def totalDays (w : WeatherPeriod) : ℕ :=
  w.totalRainyDays + (w.clearEvenings + w.clearMornings - w.totalRainyDays) / 2

/-- Theorem stating that given the specific weather conditions, the total period is 11 days. -/
theorem weather_period_days (w : WeatherPeriod)
  (h1 : w.totalRainyDays = 9)
  (h2 : w.clearEvenings = 6)
  (h3 : w.clearMornings = 7) :
  totalDays w = 11 := by
  sorry

end NUMINAMATH_CALUDE_weather_period_days_l3613_361359


namespace NUMINAMATH_CALUDE_base9_to_base3_7254_l3613_361322

/-- Converts a single digit from base 9 to its two-digit representation in base 3 -/
def base9_to_base3_digit (d : Nat) : Nat := sorry

/-- Converts a number from base 9 to base 3 -/
def base9_to_base3 (n : Nat) : Nat := sorry

theorem base9_to_base3_7254 :
  base9_to_base3 7254 = 210212113 := by sorry

end NUMINAMATH_CALUDE_base9_to_base3_7254_l3613_361322


namespace NUMINAMATH_CALUDE_track_circumference_l3613_361369

/-- Represents the circumference of the circular track -/
def circumference : ℝ := 720

/-- Represents the distance B has traveled at the first meeting -/
def first_meeting_distance : ℝ := 150

/-- Represents the distance A has left to complete one lap at the second meeting -/
def second_meeting_remaining : ℝ := 90

/-- Represents the number of laps A has completed at the third meeting -/
def third_meeting_laps : ℝ := 1.5

theorem track_circumference :
  (first_meeting_distance + (circumference - first_meeting_distance) = circumference) ∧
  (circumference - second_meeting_remaining + (circumference / 2 + second_meeting_remaining) = circumference) ∧
  (third_meeting_laps * circumference + (circumference + first_meeting_distance) = 2 * circumference) :=
by sorry

end NUMINAMATH_CALUDE_track_circumference_l3613_361369


namespace NUMINAMATH_CALUDE_part_time_employees_l3613_361350

theorem part_time_employees (total_employees full_time_employees : ℕ) 
  (h1 : total_employees = 65134)
  (h2 : full_time_employees = 63093)
  (h3 : total_employees ≥ full_time_employees) :
  total_employees - full_time_employees = 2041 := by
  sorry

end NUMINAMATH_CALUDE_part_time_employees_l3613_361350


namespace NUMINAMATH_CALUDE_six_last_digit_to_appear_l3613_361385

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppeared (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ unitsDigit (fib k) = d

-- Theorem statement
theorem six_last_digit_to_appear :
  ∀ d : ℕ, d < 10 → d ≠ 6 →
    ∃ n : ℕ, digitAppeared d n ∧ ¬digitAppeared 6 n :=
by sorry

end NUMINAMATH_CALUDE_six_last_digit_to_appear_l3613_361385


namespace NUMINAMATH_CALUDE_root_difference_squared_l3613_361303

theorem root_difference_squared (p q : ℚ) : 
  (6 * p^2 - 7 * p - 20 = 0) → 
  (6 * q^2 - 7 * q - 20 = 0) → 
  (p - q)^2 = 529 / 36 := by
sorry

end NUMINAMATH_CALUDE_root_difference_squared_l3613_361303


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3613_361368

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {aₙ} is a geometric sequence with common ratio q
  a 1 = 1 / 4 →                 -- a₁ = 1/4
  a 3 * a 5 = 4 * (a 4 - 1) →   -- a₃a₅ = 4(a₄ - 1)
  a 2 = 1 / 2 := by             -- a₂ = 1/2
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3613_361368


namespace NUMINAMATH_CALUDE_sum_of_monomials_is_monomial_l3613_361399

/-- 
Given two monomials 2x^3y^n and -6x^(m+5)y, if their sum is still a monomial,
then m + n = -1.
-/
theorem sum_of_monomials_is_monomial (m n : ℤ) : 
  (∃ (x y : ℝ), ∀ (a b : ℝ), 2 * (x^3) * (y^n) + (-6) * (x^(m+5)) * y = a * (x^b) * y) → 
  m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_monomials_is_monomial_l3613_361399


namespace NUMINAMATH_CALUDE_right_triangle_area_and_hypotenuse_l3613_361311

theorem right_triangle_area_and_hypotenuse 
  (leg1 leg2 : ℝ) 
  (h_leg1 : leg1 = 30) 
  (h_leg2 : leg2 = 45) : 
  (1/2 * leg1 * leg2 = 675) ∧ 
  (Real.sqrt (leg1^2 + leg2^2) = 54) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_and_hypotenuse_l3613_361311


namespace NUMINAMATH_CALUDE_mountain_speed_decrease_l3613_361308

/-- The problem of finding the percentage decrease in vehicle speed when ascending a mountain. -/
theorem mountain_speed_decrease (initial_speed : ℝ) (ascend_distance descend_distance : ℝ) 
  (total_time : ℝ) (descend_increase : ℝ) :
  initial_speed = 30 →
  ascend_distance = 60 →
  descend_distance = 72 →
  total_time = 6 →
  descend_increase = 0.2 →
  ∃ (x : ℝ),
    x = 0.5 ∧
    (ascend_distance / (initial_speed * (1 - x))) + 
    (descend_distance / (initial_speed * (1 + descend_increase))) = total_time :=
by sorry

end NUMINAMATH_CALUDE_mountain_speed_decrease_l3613_361308


namespace NUMINAMATH_CALUDE_open_box_volume_l3613_361358

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume 
  (sheet_length sheet_width cut_size : ℝ)
  (h_length : sheet_length = 52)
  (h_width : sheet_width = 36)
  (h_cut : cut_size = 8) :
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 5760 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l3613_361358


namespace NUMINAMATH_CALUDE_four_square_prod_inequality_l3613_361357

theorem four_square_prod_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + d^2) * (d^2 + a^2) ≥ 64 * a * b * c * d * |((a - b) * (b - c) * (c - d) * (d - a))| := by
  sorry

end NUMINAMATH_CALUDE_four_square_prod_inequality_l3613_361357


namespace NUMINAMATH_CALUDE_height_opposite_y_is_8_l3613_361382

/-- Regular triangle XYZ with pillars -/
structure Triangle where
  /-- Side length of the triangle -/
  side : ℝ
  /-- Height of pillar at X -/
  height_x : ℝ
  /-- Height of pillar at Y -/
  height_y : ℝ
  /-- Height of pillar at Z -/
  height_z : ℝ

/-- Calculate the height of the pillar opposite to Y -/
def height_opposite_y (t : Triangle) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the height of the pillar opposite Y is 8m -/
theorem height_opposite_y_is_8 (t : Triangle) 
  (h_regular : t.side > 0)
  (h_x : t.height_x = 8)
  (h_y : t.height_y = 5)
  (h_z : t.height_z = 7) : 
  height_opposite_y t = 8 :=
sorry

end NUMINAMATH_CALUDE_height_opposite_y_is_8_l3613_361382


namespace NUMINAMATH_CALUDE_jerk_tuna_fish_count_l3613_361398

theorem jerk_tuna_fish_count (jerk_tuna : ℕ) (tall_tuna : ℕ) : 
  tall_tuna = 2 * jerk_tuna → 
  jerk_tuna + tall_tuna = 432 → 
  jerk_tuna = 144 := by
sorry

end NUMINAMATH_CALUDE_jerk_tuna_fish_count_l3613_361398


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l3613_361367

theorem least_three_digit_multiple_of_eight : 
  (∀ n : ℕ, 100 ≤ n ∧ n < 104 → n % 8 ≠ 0) ∧ 104 % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l3613_361367


namespace NUMINAMATH_CALUDE_fewer_football_boxes_l3613_361329

theorem fewer_football_boxes (total_cards : ℕ) (basketball_boxes : ℕ) (cards_per_basketball_box : ℕ) (cards_per_football_box : ℕ) 
  (h1 : total_cards = 255)
  (h2 : basketball_boxes = 9)
  (h3 : cards_per_basketball_box = 15)
  (h4 : cards_per_football_box = 20)
  (h5 : basketball_boxes * cards_per_basketball_box + (total_cards - basketball_boxes * cards_per_basketball_box) = total_cards)
  (h6 : (total_cards - basketball_boxes * cards_per_basketball_box) % cards_per_football_box = 0) :
  basketball_boxes - (total_cards - basketball_boxes * cards_per_basketball_box) / cards_per_football_box = 3 := by
  sorry

end NUMINAMATH_CALUDE_fewer_football_boxes_l3613_361329


namespace NUMINAMATH_CALUDE_tax_discount_order_invariance_l3613_361376

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_order_invariance 
  (original_price tax_rate discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_price : 0 < original_price) :
  original_price * (1 + tax_rate) * (1 - discount_rate) = 
  original_price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

end NUMINAMATH_CALUDE_tax_discount_order_invariance_l3613_361376


namespace NUMINAMATH_CALUDE_vlads_pen_price_ratio_l3613_361328

/-- The ratio of gel pen price to ballpoint pen price given the conditions in Vlad's pen purchase problem -/
theorem vlads_pen_price_ratio :
  ∀ (x y : ℕ) (b g : ℝ),
  x > 0 → y > 0 → b > 0 → g > 0 →
  (x + y) * g = 4 * (x * b + y * g) →
  (x + y) * b = (1 / 2) * (x * b + y * g) →
  g = 8 * b := by
sorry

end NUMINAMATH_CALUDE_vlads_pen_price_ratio_l3613_361328


namespace NUMINAMATH_CALUDE_inequality_proof_l3613_361389

/-- Given positive real numbers a, b, c, and the function f(x) = |x+a| * |x+b|,
    prove that f(1)f(c) ≥ 16abc -/
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let f := fun x => |x + a| * |x + b|
  f 1 * f c ≥ 16 * a * b * c := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3613_361389


namespace NUMINAMATH_CALUDE_quadratic_root_bound_l3613_361302

theorem quadratic_root_bound (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_real_roots : ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ |x| ≤ 2 * |c / b| := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_bound_l3613_361302


namespace NUMINAMATH_CALUDE_inequality_comparison_l3613_361331

theorem inequality_comparison : 
  (-0.1 < -0.01) ∧ ¬(-1 > 0) ∧ ¬((1:ℚ)/2 < (1:ℚ)/3) ∧ ¬(-5 > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_comparison_l3613_361331


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3613_361378

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3613_361378


namespace NUMINAMATH_CALUDE_total_books_l3613_361343

theorem total_books (joan_books tom_books : ℕ) 
  (h1 : joan_books = 10) 
  (h2 : tom_books = 38) : 
  joan_books + tom_books = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l3613_361343


namespace NUMINAMATH_CALUDE_no_cube_sum_equals_cube_l3613_361342

theorem no_cube_sum_equals_cube : ∀ m n : ℕ+, m^3 + 11^3 ≠ n^3 := by
  sorry

end NUMINAMATH_CALUDE_no_cube_sum_equals_cube_l3613_361342


namespace NUMINAMATH_CALUDE_common_difference_is_negative_two_l3613_361377

def arithmetic_sequence (n : ℕ) : ℤ := 3 - 2 * n

theorem common_difference_is_negative_two :
  ∀ n : ℕ, arithmetic_sequence (n + 1) - arithmetic_sequence n = -2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_negative_two_l3613_361377


namespace NUMINAMATH_CALUDE_solution_count_l3613_361387

/-- The greatest integer function (floor function) -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The number of solutions to x^2 - ⌊x⌋^2 = (x - ⌊x⌋)^2 in [1, n] -/
def num_solutions (n : ℕ) : ℕ :=
  n^2 - n + 1

/-- Theorem stating the number of solutions to the equation -/
theorem solution_count (n : ℕ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ n →
    x^2 - (floor x)^2 = (x - floor x)^2) →
  num_solutions n = n^2 - n + 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_count_l3613_361387


namespace NUMINAMATH_CALUDE_positive_integer_triplets_equation_l3613_361372

theorem positive_integer_triplets_equation :
  ∀ a b c : ℕ+,
    (6 : ℕ) ^ a.val = 1 + 2 ^ b.val + 3 ^ c.val ↔
    ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 3, 3) ∨ (a, b, c) = (2, 5, 1)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_triplets_equation_l3613_361372


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3613_361364

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4/3))
  A ≤ 3 ∧ (A = 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3613_361364


namespace NUMINAMATH_CALUDE_clothing_sales_theorem_l3613_361388

/-- Represents the sales data for a clothing store --/
structure SalesData where
  typeA_sold : ℕ
  typeB_sold : ℕ
  total_sales : ℕ

/-- Represents the pricing and sales increase data --/
structure ClothingData where
  typeA_price : ℕ
  typeB_price : ℕ
  typeA_increase : ℚ
  typeB_increase : ℚ

def store_A : SalesData := ⟨60, 15, 3600⟩
def store_B : SalesData := ⟨40, 60, 4400⟩

theorem clothing_sales_theorem (d : ClothingData) :
  d.typeA_price = 50 ∧ 
  d.typeB_price = 40 ∧ 
  d.typeA_increase = 1/5 ∧
  d.typeB_increase = 1/2 →
  (store_A.typeA_sold * d.typeA_price + store_A.typeB_sold * d.typeB_price = store_A.total_sales) ∧
  (store_B.typeA_sold * d.typeA_price + store_B.typeB_sold * d.typeB_price = store_B.total_sales) ∧
  ((store_A.typeA_sold + store_B.typeA_sold) * d.typeA_price * (1 + d.typeA_increase) : ℚ) / 
  ((store_A.typeB_sold + store_B.typeB_sold) * d.typeB_price * (1 + d.typeB_increase) : ℚ) = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_clothing_sales_theorem_l3613_361388


namespace NUMINAMATH_CALUDE_production_days_calculation_l3613_361380

/-- Given the average daily production for n days and the production on an additional day,
    calculate the number of days n. -/
theorem production_days_calculation (n : ℕ) : 
  (n * 70 + 90) / (n + 1) = 75 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l3613_361380


namespace NUMINAMATH_CALUDE_max_value_x_y4_z5_l3613_361317

theorem max_value_x_y4_z5 (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  ∃ (max : ℝ), max = 243 ∧ x + y^4 + z^5 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_y4_z5_l3613_361317


namespace NUMINAMATH_CALUDE_corrected_mean_l3613_361339

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 34 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * 36.22 :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l3613_361339


namespace NUMINAMATH_CALUDE_hexagonal_board_cell_count_l3613_361362

/-- The number of cells in a hexagonal board with side length m -/
def hexagonal_board_cells (m : ℕ) : ℕ := 3 * m^2 - 3 * m + 1

/-- Theorem: The number of cells in a hexagonal board with side length m is 3m^2 - 3m + 1 -/
theorem hexagonal_board_cell_count (m : ℕ) :
  hexagonal_board_cells m = 3 * m^2 - 3 * m + 1 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_board_cell_count_l3613_361362


namespace NUMINAMATH_CALUDE_all_statements_false_l3613_361397

theorem all_statements_false :
  (¬ (∀ x : ℝ, x^(1/3) = x → x = 0 ∨ x = 1)) ∧
  (¬ (∀ a : ℝ, Real.sqrt (a^2) = a)) ∧
  (¬ ((-8 : ℝ)^(1/3) = 2 ∨ (-8 : ℝ)^(1/3) = -2)) ∧
  (¬ (Real.sqrt (Real.sqrt 81) = 9)) :=
sorry

end NUMINAMATH_CALUDE_all_statements_false_l3613_361397


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_bound_l3613_361316

variable (a : ℝ)

def f (x : ℝ) := (x - 1)^2 + 2*a*x + 1

theorem decreasing_function_implies_a_bound :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 4 → f a x₁ > f a x₂) →
  a ≤ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_bound_l3613_361316


namespace NUMINAMATH_CALUDE_tennis_balls_per_pack_l3613_361306

theorem tennis_balls_per_pack (num_packs : ℕ) (total_cost : ℕ) (cost_per_ball : ℕ) : 
  num_packs = 4 → total_cost = 24 → cost_per_ball = 2 → 
  (total_cost / cost_per_ball) / num_packs = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tennis_balls_per_pack_l3613_361306


namespace NUMINAMATH_CALUDE_snake_toy_cost_l3613_361305

theorem snake_toy_cost (cage_cost total_cost : ℚ) (found_money : ℚ) : 
  cage_cost = 14.54 → 
  found_money = 1 → 
  total_cost = 26.3 → 
  total_cost = cage_cost + (12.76 : ℚ) - found_money := by sorry

end NUMINAMATH_CALUDE_snake_toy_cost_l3613_361305
