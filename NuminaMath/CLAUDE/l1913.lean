import Mathlib

namespace NUMINAMATH_CALUDE_square_partition_exists_equilateral_triangle_partition_exists_l1913_191311

-- Define a structure for a triangle
structure Triangle :=
  (a b c : ℝ)

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define what it means for two triangles to be congruent
def areCongruent (t1 t2 : Triangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

-- Define a structure for a square
structure Square :=
  (side : ℝ)

-- Define a structure for an equilateral triangle
structure EquilateralTriangle :=
  (side : ℝ)

-- Theorem for the square partition
theorem square_partition_exists (s : Square) : 
  ∃ (t1 t2 t3 t4 : Triangle), 
    (isIsosceles t1 ∧ isIsosceles t2 ∧ isIsosceles t3 ∧ isIsosceles t4) ∧
    (¬ areCongruent t1 t2 ∧ ¬ areCongruent t1 t3 ∧ ¬ areCongruent t1 t4 ∧
     ¬ areCongruent t2 t3 ∧ ¬ areCongruent t2 t4 ∧ ¬ areCongruent t3 t4) :=
sorry

-- Theorem for the equilateral triangle partition
theorem equilateral_triangle_partition_exists (et : EquilateralTriangle) : 
  ∃ (t1 t2 t3 t4 : Triangle), 
    (isIsosceles t1 ∧ isIsosceles t2 ∧ isIsosceles t3 ∧ isIsosceles t4) ∧
    (¬ areCongruent t1 t2 ∧ ¬ areCongruent t1 t3 ∧ ¬ areCongruent t1 t4 ∧
     ¬ areCongruent t2 t3 ∧ ¬ areCongruent t2 t4 ∧ ¬ areCongruent t3 t4) :=
sorry

end NUMINAMATH_CALUDE_square_partition_exists_equilateral_triangle_partition_exists_l1913_191311


namespace NUMINAMATH_CALUDE_perfect_square_sums_l1913_191384

theorem perfect_square_sums : ∃ (x y : ℕ+), 
  (∃ (a : ℕ+), (x + y : ℕ) = a^2) ∧ 
  (∃ (b : ℕ+), (x^2 + y^2 : ℕ) = b^2) ∧ 
  (∃ (c : ℕ+), (x^3 + y^3 : ℕ) = c^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sums_l1913_191384


namespace NUMINAMATH_CALUDE_book_price_increase_l1913_191335

/-- Given a book with an original price and a percentage increase, 
    calculate the new price after the increase. -/
theorem book_price_increase (original_price : ℝ) (percent_increase : ℝ) 
  (h1 : original_price = 300)
  (h2 : percent_increase = 10) : 
  original_price * (1 + percent_increase / 100) = 330 := by
  sorry

end NUMINAMATH_CALUDE_book_price_increase_l1913_191335


namespace NUMINAMATH_CALUDE_sally_weekday_pages_l1913_191352

/-- The number of pages Sally reads on weekdays -/
def weekday_pages : ℕ := sorry

/-- The number of pages Sally reads on weekends -/
def weekend_pages : ℕ := 20

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- The total number of pages in the book -/
def total_pages : ℕ := 180

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

theorem sally_weekday_pages :
  weekday_pages = 10 :=
by sorry

end NUMINAMATH_CALUDE_sally_weekday_pages_l1913_191352


namespace NUMINAMATH_CALUDE_paper_length_calculation_l1913_191391

/-- The length of a rectangular sheet of paper satisfying specific area conditions -/
theorem paper_length_calculation (L : ℝ) : 
  (2 * 11 * L = 2 * 9.5 * 11 + 100) → L = 14 := by
  sorry

end NUMINAMATH_CALUDE_paper_length_calculation_l1913_191391


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l1913_191339

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = π) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![Real.cos A ^ 2, Real.tan A, 1],
    ![Real.cos B ^ 2, Real.tan B, 1],
    ![Real.cos C ^ 2, Real.tan C, 1]
  ]
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l1913_191339


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1913_191390

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I)/z = I) : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1913_191390


namespace NUMINAMATH_CALUDE_heather_block_distribution_l1913_191395

/-- Given an initial number of blocks, the number of blocks shared, and the number of friends,
    calculate the number of blocks each friend receives when distributing the remaining blocks equally. -/
def blocks_per_friend (initial_blocks : ℕ) (shared_blocks : ℕ) (num_friends : ℕ) : ℕ :=
  (initial_blocks - shared_blocks) / num_friends

/-- Theorem stating that given 258 initial blocks, after sharing 129 blocks and
    distributing the remainder equally among 6 friends, each friend receives 21 blocks. -/
theorem heather_block_distribution :
  blocks_per_friend 258 129 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_heather_block_distribution_l1913_191395


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1913_191374

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 100) 
  (h2 : bridge_length = 170) 
  (h3 : train_speed_kmph = 36) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 27 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1913_191374


namespace NUMINAMATH_CALUDE_incorrect_inequality_l1913_191326

theorem incorrect_inequality (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  ¬(abs a + abs b > abs (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l1913_191326


namespace NUMINAMATH_CALUDE_all_statements_imply_negation_l1913_191367

theorem all_statements_imply_negation (p q r : Prop) :
  (p ∧ q ∧ ¬r) → (¬p ∨ ¬q ∨ ¬r) ∧
  (¬p ∧ q ∧ r) → (¬p ∨ ¬q ∨ ¬r) ∧
  (p ∧ ¬q ∧ r) → (¬p ∨ ¬q ∨ ¬r) ∧
  (¬p ∧ ¬q ∧ r) → (¬p ∨ ¬q ∨ ¬r) :=
by sorry

#check all_statements_imply_negation

end NUMINAMATH_CALUDE_all_statements_imply_negation_l1913_191367


namespace NUMINAMATH_CALUDE_cheryl_basil_harvest_l1913_191337

-- Define the variables and constants
def basil_per_pesto : ℝ := 4
def harvest_weeks : ℕ := 8
def total_pesto : ℝ := 32

-- Define the theorem
theorem cheryl_basil_harvest :
  (basil_per_pesto * total_pesto) / harvest_weeks = 16 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_basil_harvest_l1913_191337


namespace NUMINAMATH_CALUDE_article_cost_changes_l1913_191376

theorem article_cost_changes (initial_cost : ℝ) : 
  initial_cost = 75 →
  (initial_cost * (1 + 0.2) * (1 - 0.2) * (1 + 0.3) * (1 - 0.25)) = 70.2 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_changes_l1913_191376


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l1913_191382

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  ¬(7 ∣ (a.val * b.val * (a.val + b.val))) ∧ 
  (7^7 ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l1913_191382


namespace NUMINAMATH_CALUDE_subset_condition_l1913_191357

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | x > a}

theorem subset_condition (a : ℝ) : A ⊆ B a → a < -2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l1913_191357


namespace NUMINAMATH_CALUDE_function_properties_l1913_191314

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * a * x)
def g (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem function_properties (a k b : ℝ) (h_k : k ≠ 0) :
  -- Part 1
  (f a 1 = Real.exp 1 ∧ ∀ x, g k b x = -g k b (-x)) →
  a = 1/2 ∧ b = 0 ∧
  -- Part 2
  (∀ x > 0, f (1/2) x > g k 0 x) →
  k < Real.exp 1 ∧
  -- Part 3
  (∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ f (1/2) x₁ = g k 0 x₁ ∧ f (1/2) x₂ = g k 0 x₂) →
  x₁ * x₂ < 1 :=
by sorry

end

end NUMINAMATH_CALUDE_function_properties_l1913_191314


namespace NUMINAMATH_CALUDE_draw_probability_modified_deck_l1913_191341

/-- A modified deck of cards -/
structure ModifiedDeck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (cards_per_rank : ℕ)

/-- The probability of drawing a specific sequence of cards -/
def draw_probability (deck : ModifiedDeck) (heart_cards : ℕ) (spade_cards : ℕ) (king_cards : ℕ) : ℚ :=
  (heart_cards * spade_cards * king_cards : ℚ) / 
  (deck.total_cards * (deck.total_cards - 1) * (deck.total_cards - 2))

/-- The main theorem -/
theorem draw_probability_modified_deck :
  let deck := ModifiedDeck.mk 104 26 4 26 8
  draw_probability deck 26 26 8 = 169 / 34102 := by sorry

end NUMINAMATH_CALUDE_draw_probability_modified_deck_l1913_191341


namespace NUMINAMATH_CALUDE_fans_with_all_items_l1913_191375

def total_fans : ℕ := 5000
def tshirt_interval : ℕ := 90
def cap_interval : ℕ := 45
def scarf_interval : ℕ := 60

theorem fans_with_all_items :
  (total_fans / (Nat.lcm (Nat.lcm tshirt_interval cap_interval) scarf_interval)) = 27 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l1913_191375


namespace NUMINAMATH_CALUDE_octadecagon_diagonals_l1913_191306

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octadecagon has 18 sides -/
def octadecagon_sides : ℕ := 18

theorem octadecagon_diagonals :
  num_diagonals octadecagon_sides = 135 := by
  sorry

end NUMINAMATH_CALUDE_octadecagon_diagonals_l1913_191306


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1913_191359

theorem unique_positive_solution : 
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 - 7 * x - 6 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1913_191359


namespace NUMINAMATH_CALUDE_park_area_l1913_191329

/-- The area of a rectangular park with perimeter 120 feet and length three times the width is 675 square feet. -/
theorem park_area (length width : ℝ) : 
  (2 * length + 2 * width = 120) →
  (length = 3 * width) →
  (length * width = 675) :=
by
  sorry

end NUMINAMATH_CALUDE_park_area_l1913_191329


namespace NUMINAMATH_CALUDE_room_height_proof_l1913_191336

/-- Proves that the height of a room with given dimensions and openings is 6 feet -/
theorem room_height_proof (width length : ℝ) (doorway1_width doorway1_height : ℝ)
  (window_width window_height : ℝ) (doorway2_width doorway2_height : ℝ)
  (total_paint_area : ℝ) (h : ℝ) :
  width = 20 ∧ length = 20 ∧
  doorway1_width = 3 ∧ doorway1_height = 7 ∧
  window_width = 6 ∧ window_height = 4 ∧
  doorway2_width = 5 ∧ doorway2_height = 7 ∧
  total_paint_area = 560 ∧
  total_paint_area = 4 * width * h - (doorway1_width * doorway1_height + window_width * window_height + doorway2_width * doorway2_height) →
  h = 6 := by
  sorry

#check room_height_proof

end NUMINAMATH_CALUDE_room_height_proof_l1913_191336


namespace NUMINAMATH_CALUDE_units_digit_of_n_l1913_191397

/-- Given two natural numbers m and n, returns true if m has a units digit of 9 -/
def hasUnitsDigitOf9 (m : ℕ) : Prop :=
  m % 10 = 9

/-- Given a natural number n, returns its units digit -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 31^6) (h2 : hasUnitsDigitOf9 m) :
  unitsDigit n = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l1913_191397


namespace NUMINAMATH_CALUDE_fifth_root_of_unity_l1913_191325

theorem fifth_root_of_unity (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + 1 = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + m + p = 0) :
  m^5 = 1 := by sorry

end NUMINAMATH_CALUDE_fifth_root_of_unity_l1913_191325


namespace NUMINAMATH_CALUDE_remainder_relationship_l1913_191398

theorem remainder_relationship (P P' D R R' C : ℕ) (h1 : P > P') (h2 : P % D = R) (h3 : P' % D = R') : 
  ∃ (s r : ℕ), ((P + C) * P') % D = s ∧ (P * P') % D = r ∧ 
  (∃ (C1 D1 : ℕ), s > r) ∧ (∃ (C2 D2 : ℕ), s < r) :=
sorry

end NUMINAMATH_CALUDE_remainder_relationship_l1913_191398


namespace NUMINAMATH_CALUDE_student_congress_size_l1913_191323

/-- The number of classes in the school -/
def num_classes : ℕ := 40

/-- The number of representatives sent from each class -/
def representatives_per_class : ℕ := 3

/-- The sample size (number of students in the "Student Congress") -/
def sample_size : ℕ := num_classes * representatives_per_class

theorem student_congress_size :
  sample_size = 120 :=
by sorry

end NUMINAMATH_CALUDE_student_congress_size_l1913_191323


namespace NUMINAMATH_CALUDE_problem_1_l1913_191317

theorem problem_1 : (2 * Real.sqrt 12 - 3 * Real.sqrt (1/3)) * Real.sqrt 6 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1913_191317


namespace NUMINAMATH_CALUDE_cricket_match_average_l1913_191324

/-- Represents the runs scored by each batsman -/
structure BatsmanScores where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The conditions of the cricket match -/
def cricket_match_conditions (scores : BatsmanScores) : Prop :=
  scores.d = scores.e + 5 ∧
  scores.e = scores.a - 8 ∧
  scores.b = scores.d + scores.e ∧
  scores.b + scores.c = 107 ∧
  scores.e = 20

/-- The theorem stating that the average score is 36 -/
theorem cricket_match_average (scores : BatsmanScores) 
  (h : cricket_match_conditions scores) : 
  (scores.a + scores.b + scores.c + scores.d + scores.e) / 5 = 36 := by
  sorry

#check cricket_match_average

end NUMINAMATH_CALUDE_cricket_match_average_l1913_191324


namespace NUMINAMATH_CALUDE_ladder_problem_l1913_191362

theorem ladder_problem (ladder_length height base : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12)
  (h3 : ladder_length^2 = height^2 + base^2) : 
  base = 5 := by sorry

end NUMINAMATH_CALUDE_ladder_problem_l1913_191362


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_at_10_l1913_191310

-- Define a quadratic polynomial
def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the divisibility condition
def divisibility_condition (q : ℝ → ℝ) : Prop :=
  ∃ (p : ℝ → ℝ), ∀ (x : ℝ), q x^3 - 3*x = p x * (x - 2) * (x + 2) * (x - 5)

theorem quadratic_polynomial_value_at_10 
  (a b c : ℝ) 
  (h : divisibility_condition (quadratic_polynomial a b c)) :
  quadratic_polynomial a b c 10 = (96 * Real.rpow 15 (1/3) - 135 * Real.rpow 6 (1/3)) / 21 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_polynomial_value_at_10_l1913_191310


namespace NUMINAMATH_CALUDE_sin_transformation_equivalence_l1913_191315

theorem sin_transformation_equivalence (x : ℝ) :
  let f (x : ℝ) := Real.sin x
  let g (x : ℝ) := Real.sin (2*x - π/5)
  let transform1 (x : ℝ) := Real.sin (2*(x - π/5))
  let transform2 (x : ℝ) := Real.sin (2*(x - π/10))
  (∀ x, g x = transform1 x) ∧ (∀ x, g x = transform2 x) :=
by sorry

end NUMINAMATH_CALUDE_sin_transformation_equivalence_l1913_191315


namespace NUMINAMATH_CALUDE_bridge_length_l1913_191393

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 255 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1913_191393


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1913_191313

theorem inequality_solution_set (x : ℝ) : 
  (1 / (x^2 + 1) > 4 / x + 25 / 10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1913_191313


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l1913_191363

-- Define the set of integers
def U : Set Int := Set.univ

-- Define set A
def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}

-- Define set B
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}

-- Define the set of integers divisible by 3
def DivisibleBy3 : Set Int := {x | ∃ k : Int, x = 3 * k}

-- Theorem statement
theorem complement_of_A_union_B (x : Int) : 
  x ∈ (U \ (A ∪ B)) ↔ x ∈ DivisibleBy3 :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l1913_191363


namespace NUMINAMATH_CALUDE_problem_solution_l1913_191370

theorem problem_solution (m : ℝ) (h : m + 1/m = 10) : 
  m^2 + 1/m^2 + m + 1/m = 108 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1913_191370


namespace NUMINAMATH_CALUDE_solve_linear_system_l1913_191378

theorem solve_linear_system (a b : ℤ) 
  (eq1 : 2009 * a + 2013 * b = 2021)
  (eq2 : 2011 * a + 2015 * b = 2023) :
  a - b = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_system_l1913_191378


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l1913_191316

theorem min_value_of_sum_of_squares (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 8) 
  (h2 : t * u * v * w = 16) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l1913_191316


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1913_191321

theorem complex_number_modulus (z : ℂ) : z = (1 - Complex.I) / (1 + Complex.I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1913_191321


namespace NUMINAMATH_CALUDE_polynomial_equality_l1913_191342

/-- Given a polynomial q(x) satisfying the equation
    q(x) + (2x^6 + 4x^4 + 5x^3 + 11x) = (10x^4 + 30x^3 + 40x^2 + 8x + 3),
    prove that q(x) = -2x^6 + 6x^4 + 25x^3 + 40x^2 - 3x + 3 -/
theorem polynomial_equality (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 4 * x^4 + 5 * x^3 + 11 * x) = 
       (10 * x^4 + 30 * x^3 + 40 * x^2 + 8 * x + 3)) →
  (∀ x, q x = -2 * x^6 + 6 * x^4 + 25 * x^3 + 40 * x^2 - 3 * x + 3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1913_191342


namespace NUMINAMATH_CALUDE_dog_collar_nylon_l1913_191379

/-- The number of inches of nylon needed for one cat collar -/
def cat_collar_nylon : ℕ := 10

/-- The total number of inches of nylon needed for 9 dog collars and 3 cat collars -/
def total_nylon : ℕ := 192

/-- The number of dog collars made -/
def num_dog_collars : ℕ := 9

/-- The number of cat collars made -/
def num_cat_collars : ℕ := 3

/-- Theorem stating that 18 inches of nylon are needed for one dog collar -/
theorem dog_collar_nylon : 
  ∃ (x : ℕ), x * num_dog_collars + cat_collar_nylon * num_cat_collars = total_nylon ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_dog_collar_nylon_l1913_191379


namespace NUMINAMATH_CALUDE_largest_fraction_l1913_191369

theorem largest_fraction (a b c d : ℝ) 
  (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d)
  (h5 : a = 2) (h6 : b = 3) (h7 : c = 5) (h8 : d = 8) :
  (c + d) / (a + b) = max ((a + b) / (c + d)) 
                         (max ((a + d) / (b + c)) 
                              (max ((b + c) / (a + d)) 
                                   ((b + d) / (a + c)))) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l1913_191369


namespace NUMINAMATH_CALUDE_power_five_mod_hundred_l1913_191392

theorem power_five_mod_hundred : 5^2023 % 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_hundred_l1913_191392


namespace NUMINAMATH_CALUDE_mustard_at_first_table_l1913_191328

-- Define the amount of mustard at each table
def mustard_table1 : ℝ := sorry
def mustard_table2 : ℝ := 0.25
def mustard_table3 : ℝ := 0.38

-- Define the total amount of mustard
def total_mustard : ℝ := 0.88

-- Theorem statement
theorem mustard_at_first_table :
  mustard_table1 + mustard_table2 + mustard_table3 = total_mustard →
  mustard_table1 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_mustard_at_first_table_l1913_191328


namespace NUMINAMATH_CALUDE_no_overlap_l1913_191320

theorem no_overlap (x : ℝ) (h1 : 150 ≤ x ∧ x ≤ 300) (h2 : ⌊Real.sqrt x⌋ = 16) : 
  ⌊Real.sqrt (10 * x)⌋ ≠ 160 := by
  sorry

end NUMINAMATH_CALUDE_no_overlap_l1913_191320


namespace NUMINAMATH_CALUDE_inequality_proof_l1913_191361

theorem inequality_proof (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  Real.sin (Real.cos x) < Real.cos x ∧ Real.cos x < Real.cos (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1913_191361


namespace NUMINAMATH_CALUDE_new_years_day_theorem_l1913_191305

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a year -/
structure Date where
  month : Nat
  day : Nat

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

theorem new_years_day_theorem 
  (february_has_29_days : Nat)
  (february_has_four_mondays : Nat)
  (february_has_five_sundays : Nat)
  (february_13_is_friday : DayOfWeek)
  : (february_has_29_days = 29) →
    (february_has_four_mondays = 4) →
    (february_has_five_sundays = 5) →
    (february_13_is_friday = DayOfWeek.Friday) →
    (∃ (new_years_day : DayOfWeek), 
      new_years_day = DayOfWeek.Thursday ∧ 
      advanceDay new_years_day 366 = DayOfWeek.Saturday) :=
by sorry


end NUMINAMATH_CALUDE_new_years_day_theorem_l1913_191305


namespace NUMINAMATH_CALUDE_function_domain_range_implies_a_value_l1913_191380

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

-- State the theorem
theorem function_domain_range_implies_a_value :
  (∀ x, ∃ y, f a x = y) → -- Domain is ℝ
  (∀ y, ∃ x, f a x = y) → -- Range is ℝ
  a = -1 := by sorry

end NUMINAMATH_CALUDE_function_domain_range_implies_a_value_l1913_191380


namespace NUMINAMATH_CALUDE_tiling_ways_eq_fib_l1913_191351

/-- The number of ways to tile a 2 × n rectangle with 2 × 1 dominoes -/
def tiling_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => tiling_ways k + tiling_ways (k + 1)

/-- The Fibonacci sequence -/
def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | k + 2 => fib k + fib (k + 1)

theorem tiling_ways_eq_fib (n : ℕ) : tiling_ways n = fib (n + 1) := by
  sorry


end NUMINAMATH_CALUDE_tiling_ways_eq_fib_l1913_191351


namespace NUMINAMATH_CALUDE_barbed_wire_rate_l1913_191307

/-- Given a square field with area 3136 sq m and a total cost of 865.80 for barbed wire
    (excluding two 1 m wide gates), the rate per meter of barbed wire is 3.90. -/
theorem barbed_wire_rate (area : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  area = 3136 →
  total_cost = 865.80 →
  gate_width = 1 →
  num_gates = 2 →
  (total_cost / (4 * Real.sqrt area - gate_width * num_gates)) = 3.90 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_l1913_191307


namespace NUMINAMATH_CALUDE_z_value_and_quadrant_l1913_191385

def z : ℂ := (1 + Complex.I) * (3 - 2 * Complex.I)

theorem z_value_and_quadrant :
  z = 5 + Complex.I ∧ Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_value_and_quadrant_l1913_191385


namespace NUMINAMATH_CALUDE_pirates_walking_distance_l1913_191303

/-- The number of miles walked per day on the first two islands -/
def miles_per_day_first_two_islands (
  num_islands : ℕ)
  (days_per_island : ℚ)
  (miles_per_day_last_two : ℕ)
  (total_miles : ℕ) : ℚ :=
  (total_miles - 2 * (miles_per_day_last_two * days_per_island)) /
  (2 * days_per_island)

/-- Theorem stating that the miles walked per day on the first two islands is 20 -/
theorem pirates_walking_distance :
  miles_per_day_first_two_islands 4 (3/2) 25 135 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pirates_walking_distance_l1913_191303


namespace NUMINAMATH_CALUDE_tank_capacity_l1913_191358

theorem tank_capacity
  (bucket1_capacity bucket2_capacity : ℕ)
  (bucket1_uses bucket2_uses : ℕ)
  (h1 : bucket1_capacity = 4)
  (h2 : bucket2_capacity = 3)
  (h3 : bucket2_uses = bucket1_uses + 4)
  (h4 : bucket1_capacity * bucket1_uses = bucket2_capacity * bucket2_uses) :
  bucket1_capacity * bucket1_uses = 48 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l1913_191358


namespace NUMINAMATH_CALUDE_power_equality_implies_m_value_l1913_191350

theorem power_equality_implies_m_value (m : ℕ) : 8^4 = 4^m → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_m_value_l1913_191350


namespace NUMINAMATH_CALUDE_range_of_m_satisfying_conditions_l1913_191301

def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

theorem range_of_m_satisfying_conditions : 
  {m : ℝ | (∀ a ∈ Set.Icc 1 2, |m - 5| ≤ Real.sqrt (a^2 + 8)) ∧ 
           ¬(∃ (max min : ℝ), ∀ x, f m x ≤ max ∧ f m x ≥ min)} = 
  Set.Icc 2 6 := by sorry

end NUMINAMATH_CALUDE_range_of_m_satisfying_conditions_l1913_191301


namespace NUMINAMATH_CALUDE_x_value_l1913_191309

def M (x : ℝ) : Set ℝ := {2, 0, x}
def N : Set ℝ := {0, 1}

theorem x_value : ∀ x : ℝ, N ⊆ M x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1913_191309


namespace NUMINAMATH_CALUDE_range_of_x_when_f_positive_l1913_191334

/-- A linear function obtained by translating y = x upwards by 2 units -/
def f (x : ℝ) : ℝ := x + 2

/-- The range of x when f(x) > 0 -/
theorem range_of_x_when_f_positive : 
  {x : ℝ | f x > 0} = {x : ℝ | x > -2} := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_when_f_positive_l1913_191334


namespace NUMINAMATH_CALUDE_minimize_function_l1913_191386

theorem minimize_function (x y z : ℝ) : 
  (3 * x + 2 * y + z = 3) →
  (x^2 + y^2 + 2 * z^2 ≥ 2/3) →
  (x^2 + y^2 + 2 * z^2 = 2/3 → x * y / z = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_minimize_function_l1913_191386


namespace NUMINAMATH_CALUDE_pears_eaten_by_mike_l1913_191300

theorem pears_eaten_by_mike (jason_pears keith_pears remaining_pears : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : remaining_pears = 81) :
  jason_pears + keith_pears - remaining_pears = 12 := by
  sorry

end NUMINAMATH_CALUDE_pears_eaten_by_mike_l1913_191300


namespace NUMINAMATH_CALUDE_joys_remaining_tape_l1913_191366

/-- Calculates the remaining tape after wrapping a rectangular field once. -/
def remaining_tape (total_tape : ℝ) (width : ℝ) (length : ℝ) : ℝ :=
  total_tape - (2 * (width + length))

/-- Theorem stating the remaining tape for Joy's specific problem. -/
theorem joys_remaining_tape :
  remaining_tape 250 20 60 = 90 := by
  sorry

end NUMINAMATH_CALUDE_joys_remaining_tape_l1913_191366


namespace NUMINAMATH_CALUDE_total_herd_count_l1913_191377

/-- Represents the number of animals in a shepherd's herd -/
structure Herd where
  count : ℕ

/-- Represents a shepherd with their herd -/
structure Shepherd where
  name : String
  herd : Herd

/-- The conditions of the problem -/
def exchange_conditions (jack jim dan : Shepherd) : Prop :=
  (jim.herd.count + 6 = 2 * (jack.herd.count - 1)) ∧
  (jack.herd.count + 14 = 3 * (dan.herd.count - 1)) ∧
  (dan.herd.count + 4 = 6 * (jim.herd.count - 1))

/-- The theorem to be proved -/
theorem total_herd_count (jack jim dan : Shepherd) :
  exchange_conditions jack jim dan →
  jack.herd.count + jim.herd.count + dan.herd.count = 39 := by
  sorry


end NUMINAMATH_CALUDE_total_herd_count_l1913_191377


namespace NUMINAMATH_CALUDE_sum_of_first_four_powers_of_i_is_zero_l1913_191331

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The property that i^2 = -1 -/
axiom i_squared : i^2 = -1

/-- Theorem: The sum of the first four powers of i equals 0 -/
theorem sum_of_first_four_powers_of_i_is_zero : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_four_powers_of_i_is_zero_l1913_191331


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1913_191322

/-- Given an examination with the following conditions:
  - Total number of questions is 120
  - Each correct answer scores 3 marks
  - Each wrong answer loses 1 mark
  - The total score is 180 marks
  This theorem proves that the number of correctly answered questions is 75. -/
theorem exam_score_calculation (total_questions : ℕ) (correct_score wrong_score total_score : ℤ) 
  (h1 : total_questions = 120)
  (h2 : correct_score = 3)
  (h3 : wrong_score = -1)
  (h4 : total_score = 180) :
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_score + (total_questions - correct_answers) * wrong_score = total_score ∧ 
    correct_answers = 75 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1913_191322


namespace NUMINAMATH_CALUDE_value_set_of_m_l1913_191372

def A : Set ℝ := {x : ℝ | x^2 + 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem value_set_of_m : 
  ∀ m : ℝ, (A ∪ B m = A) ↔ m ∈ ({0, 1/2, 1/3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_value_set_of_m_l1913_191372


namespace NUMINAMATH_CALUDE_PQ_length_l1913_191387

def triangle_PQR (P Q R : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xq, yq) := Q
  let (xr, yr) := R
  -- R is a right angle
  (xq - xr) * (xp - xr) + (yq - yr) * (yp - yr) = 0 ∧
  -- RP = 2.4
  Real.sqrt ((xp - xr)^2 + (yp - yr)^2) = 2.4 ∧
  -- RQ = 1.8
  Real.sqrt ((xq - xr)^2 + (yq - yr)^2) = 1.8

theorem PQ_length (P Q R : ℝ × ℝ) (h : triangle_PQR P Q R) :
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_PQ_length_l1913_191387


namespace NUMINAMATH_CALUDE_sqrt_five_squared_times_seven_sixth_power_l1913_191354

theorem sqrt_five_squared_times_seven_sixth_power : 
  Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_times_seven_sixth_power_l1913_191354


namespace NUMINAMATH_CALUDE_todd_snow_cone_stand_l1913_191304

/-- Todd's snow-cone stand problem -/
theorem todd_snow_cone_stand (borrowed : ℝ) (repay : ℝ) (equipment : ℝ) (ingredients : ℝ) 
  (marketing : ℝ) (snow_cones : ℕ) (price : ℝ) : 
  borrowed = 200 →
  repay = 220 →
  equipment = 100 →
  ingredients = 45 →
  marketing = 30 →
  snow_cones = 350 →
  price = 1.5 →
  snow_cones * price - (equipment + ingredients + marketing) - repay = 130 := by
  sorry

end NUMINAMATH_CALUDE_todd_snow_cone_stand_l1913_191304


namespace NUMINAMATH_CALUDE_alex_last_five_shots_l1913_191371

/-- Represents the number of shots made by Alex -/
structure ShotsMade where
  initial : ℕ
  after_60 : ℕ
  final : ℕ

/-- Represents the shooting percentages at different stages -/
structure ShootingPercentages where
  initial : ℚ
  after_60 : ℚ
  final : ℚ

/-- Theorem stating the number of shots Alex made in the last 5 attempts -/
theorem alex_last_five_shots 
  (shots : ShotsMade)
  (percentages : ShootingPercentages)
  (h1 : shots.initial = 30)
  (h2 : shots.after_60 = 37)
  (h3 : shots.final = 39)
  (h4 : percentages.initial = 3/5)
  (h5 : percentages.after_60 = 31/50)
  (h6 : percentages.final = 3/5) :
  shots.final - shots.after_60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alex_last_five_shots_l1913_191371


namespace NUMINAMATH_CALUDE_salt_trade_initial_investment_l1913_191355

/-- Represents the merchant's salt trading scenario -/
structure SaltTrade where
  initial_investment : ℕ  -- Initial investment in rubles
  first_profit : ℕ        -- Profit from first sale in rubles
  second_profit : ℕ       -- Profit from second sale in rubles

/-- Theorem stating the initial investment in the salt trade scenario -/
theorem salt_trade_initial_investment (trade : SaltTrade) 
  (h1 : trade.first_profit = 100)
  (h2 : trade.second_profit = 120)
  (h3 : (trade.initial_investment + trade.first_profit + trade.second_profit) = 
        (trade.initial_investment + trade.first_profit) * 
        (trade.initial_investment + trade.first_profit) / trade.initial_investment) :
  trade.initial_investment = 500 := by
  sorry

end NUMINAMATH_CALUDE_salt_trade_initial_investment_l1913_191355


namespace NUMINAMATH_CALUDE_smallest_number_l1913_191388

theorem smallest_number : min (-5 : ℝ) (min (-0.8) (min 0 (abs (-6)))) = -5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1913_191388


namespace NUMINAMATH_CALUDE_arm_wrestling_tournament_l1913_191340

/-- The number of participants with k points after m rounds in a tournament with 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * Nat.choose m k

theorem arm_wrestling_tournament (n : ℕ) (h1 : n > 7) (h2 : f n 7 5 = 42) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arm_wrestling_tournament_l1913_191340


namespace NUMINAMATH_CALUDE_animal_shelter_problem_l1913_191394

/-- Represents the animal shelter problem --/
theorem animal_shelter_problem
  (initial_dogs : ℕ) (initial_cats : ℕ) (new_pets : ℕ) (total_after_month : ℕ)
  (dog_adoption_rate : ℚ) (cat_adoption_rate : ℚ) (lizard_adoption_rate : ℚ)
  (h1 : initial_dogs = 30)
  (h2 : initial_cats = 28)
  (h3 : new_pets = 13)
  (h4 : total_after_month = 65)
  (h5 : dog_adoption_rate = 1/2)
  (h6 : cat_adoption_rate = 1/4)
  (h7 : lizard_adoption_rate = 1/5) :
  ∃ (initial_lizards : ℕ),
    initial_lizards = 20 ∧
    (↑initial_dogs * (1 - dog_adoption_rate) +
     ↑initial_cats * (1 - cat_adoption_rate) +
     ↑initial_lizards * (1 - lizard_adoption_rate) +
     ↑new_pets : ℚ) = total_after_month :=
by sorry

end NUMINAMATH_CALUDE_animal_shelter_problem_l1913_191394


namespace NUMINAMATH_CALUDE_largest_number_l1913_191389

/-- Represents a repeating decimal number -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : List ℕ
  repeatingPart : List ℕ

/-- Convert a RepeatingDecimal to a rational number -/
def toRational (r : RepeatingDecimal) : ℚ :=
  sorry

/-- The number 5.14322 -/
def a : ℚ := 5.14322

/-- The number 5.143̅2 -/
def b : RepeatingDecimal := ⟨5, [1, 4, 3], [2]⟩

/-- The number 5.14̅32 -/
def c : RepeatingDecimal := ⟨5, [1, 4], [3, 2]⟩

/-- The number 5.1̅432 -/
def d : RepeatingDecimal := ⟨5, [1], [4, 3, 2]⟩

/-- The number 5.̅4321 -/
def e : RepeatingDecimal := ⟨5, [], [4, 3, 2, 1]⟩

theorem largest_number : 
  toRational d > a ∧ 
  toRational d > toRational b ∧ 
  toRational d > toRational c ∧ 
  toRational d > toRational e :=
sorry

end NUMINAMATH_CALUDE_largest_number_l1913_191389


namespace NUMINAMATH_CALUDE_troy_vegetable_purchase_l1913_191346

/-- The number of pounds of vegetables Troy buys -/
def vegetable_pounds : ℝ := 6

/-- The number of pounds of beef Troy buys -/
def beef_pounds : ℝ := 4

/-- The cost of vegetables per pound in dollars -/
def vegetable_cost_per_pound : ℝ := 2

/-- The total cost of Troy's purchase in dollars -/
def total_cost : ℝ := 36

/-- Theorem stating that the number of pounds of vegetables Troy buys is 6 -/
theorem troy_vegetable_purchase :
  vegetable_pounds = 6 ∧
  beef_pounds = 4 ∧
  vegetable_cost_per_pound = 2 ∧
  total_cost = 36 ∧
  (3 * vegetable_cost_per_pound * beef_pounds + vegetable_cost_per_pound * vegetable_pounds = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_troy_vegetable_purchase_l1913_191346


namespace NUMINAMATH_CALUDE_butter_left_is_two_l1913_191338

/-- Calculates the amount of butter left after making three types of cookies. -/
def butter_left (total : ℚ) (choc_chip_frac : ℚ) (peanut_butter_frac : ℚ) (sugar_frac : ℚ) : ℚ :=
  let remaining_after_two := total - (choc_chip_frac * total) - (peanut_butter_frac * total)
  remaining_after_two - (sugar_frac * remaining_after_two)

/-- Proves that given the specified conditions, the amount of butter left is 2 kilograms. -/
theorem butter_left_is_two :
  butter_left 10 (1/2) (1/5) (1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_butter_left_is_two_l1913_191338


namespace NUMINAMATH_CALUDE_drink_volume_theorem_l1913_191318

/-- Represents the parts of each ingredient in the drink recipe. -/
structure DrinkRecipe where
  coke : ℕ
  sprite : ℕ
  mountainDew : ℕ
  drPepper : ℕ
  fanta : ℕ

/-- Calculates the total parts in a drink recipe. -/
def totalParts (recipe : DrinkRecipe) : ℕ :=
  recipe.coke + recipe.sprite + recipe.mountainDew + recipe.drPepper + recipe.fanta

/-- Theorem stating that given the specific drink recipe and the amount of Coke,
    the total volume of the drink is 48 ounces. -/
theorem drink_volume_theorem (recipe : DrinkRecipe)
    (h1 : recipe.coke = 4)
    (h2 : recipe.sprite = 2)
    (h3 : recipe.mountainDew = 5)
    (h4 : recipe.drPepper = 3)
    (h5 : recipe.fanta = 2)
    (h6 : 12 = recipe.coke * 3) :
    (totalParts recipe) * 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_drink_volume_theorem_l1913_191318


namespace NUMINAMATH_CALUDE_negation_of_existence_sqrt_gt_three_l1913_191332

theorem negation_of_existence_sqrt_gt_three : 
  (¬ ∃ x : ℝ, Real.sqrt x > 3) ↔ (∀ x : ℝ, Real.sqrt x ≤ 3 ∨ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_sqrt_gt_three_l1913_191332


namespace NUMINAMATH_CALUDE_train_length_l1913_191353

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (bridge_length : ℝ) :
  train_speed = 45 →
  time_to_pass = 36 →
  bridge_length = 140 →
  (train_speed * 1000 / 3600) * time_to_pass - bridge_length = 310 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1913_191353


namespace NUMINAMATH_CALUDE_total_seashells_l1913_191364

/-- The number of seashells found by Mary -/
def x : ℝ := 2

/-- The number of seashells found by Keith -/
def y : ℝ := 5

/-- The percentage of cracked seashells found by Mary -/
def m : ℝ := 0.5

/-- The percentage of cracked seashells found by Keith -/
def k : ℝ := 0.6

/-- The total number of seashells found by Mary and Keith -/
def T : ℝ := x + y

/-- The total number of cracked seashells -/
def z : ℝ := m * x + k * y

theorem total_seashells : T = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l1913_191364


namespace NUMINAMATH_CALUDE_cos_2α_value_l1913_191343

theorem cos_2α_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/4) = 1/4) : 
  Real.cos (2*α) = -(Real.sqrt 15)/8 := by
sorry

end NUMINAMATH_CALUDE_cos_2α_value_l1913_191343


namespace NUMINAMATH_CALUDE_sin_arctan_reciprocal_square_l1913_191381

theorem sin_arctan_reciprocal_square (x : ℝ) (h_pos : x > 0) (h_eq : Real.sin (Real.arctan x) = 1 / x) : x^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_arctan_reciprocal_square_l1913_191381


namespace NUMINAMATH_CALUDE_remainder_theorem_l1913_191383

theorem remainder_theorem (n : ℤ) : (5 * n^2 + 7) - (3 * n - 2) ≡ 2 * n + 4 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1913_191383


namespace NUMINAMATH_CALUDE_textbook_weight_difference_l1913_191347

theorem textbook_weight_difference :
  let chemistry_weight : ℝ := 7.12
  let geometry_weight : ℝ := 0.62
  let history_weight : ℝ := 4.25
  let literature_weight : ℝ := 3.8
  let chem_geo_combined : ℝ := chemistry_weight + geometry_weight
  let hist_lit_combined : ℝ := history_weight + literature_weight
  chem_geo_combined - hist_lit_combined = -0.31 :=
by
  sorry

end NUMINAMATH_CALUDE_textbook_weight_difference_l1913_191347


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1913_191345

theorem arithmetic_computation : 5 * 7 + 6 * 12 + 10 * 4 + 7 * 6 + 30 / 5 = 195 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1913_191345


namespace NUMINAMATH_CALUDE_expression_value_l1913_191396

theorem expression_value : 3 * (24 + 7)^2 - (24^2 + 7^2) = 2258 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1913_191396


namespace NUMINAMATH_CALUDE_smallest_of_three_powers_l1913_191302

theorem smallest_of_three_powers : 127^8 < 63^10 ∧ 63^10 < 33^12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_powers_l1913_191302


namespace NUMINAMATH_CALUDE_total_pebbles_after_fifteen_days_l1913_191327

/-- The number of pebbles collected on the first day -/
def initial_pebbles : ℕ := 3

/-- The daily increase in the number of pebbles collected -/
def daily_increase : ℕ := 2

/-- The number of days Murtha collects pebbles -/
def collection_days : ℕ := 15

/-- The arithmetic sequence of daily pebble collections -/
def pebble_sequence (n : ℕ) : ℕ := initial_pebbles + (n - 1) * daily_increase

/-- The total number of pebbles collected after a given number of days -/
def total_pebbles (n : ℕ) : ℕ := n * (initial_pebbles + pebble_sequence n) / 2

theorem total_pebbles_after_fifteen_days :
  total_pebbles collection_days = 255 := by sorry

end NUMINAMATH_CALUDE_total_pebbles_after_fifteen_days_l1913_191327


namespace NUMINAMATH_CALUDE_smallest_n_has_9_digits_l1913_191319

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def has_9_digits (n : ℕ) : Prop := n ≥ 100000000 ∧ n < 1000000000

theorem smallest_n_has_9_digits :
  ∃ n : ℕ, 
    (∀ m : ℕ, m < n → ¬(is_divisible_by m 30 ∧ is_perfect_cube (m^2) ∧ is_perfect_square (m^5))) ∧
    is_divisible_by n 30 ∧
    is_perfect_cube (n^2) ∧
    is_perfect_square (n^5) ∧
    has_9_digits n :=
sorry

end NUMINAMATH_CALUDE_smallest_n_has_9_digits_l1913_191319


namespace NUMINAMATH_CALUDE_giraffe_difference_l1913_191333

/-- In a zoo with giraffes and other animals, where the number of giraffes
    is 3 times the number of all other animals, prove that there are 200
    more giraffes than other animals. -/
theorem giraffe_difference (total_giraffes : ℕ) (other_animals : ℕ) : 
  total_giraffes = 300 →
  total_giraffes = 3 * other_animals →
  total_giraffes - other_animals = 200 :=
by sorry

end NUMINAMATH_CALUDE_giraffe_difference_l1913_191333


namespace NUMINAMATH_CALUDE_book_purchase_theorem_l1913_191360

/-- The number of people who purchased only book A -/
def Z : ℕ := 1000

/-- The number of people who purchased only book B -/
def X : ℕ := 250

/-- The number of people who purchased both books A and B -/
def Y : ℕ := 500

/-- The total number of people who purchased book A -/
def A : ℕ := Z + Y

/-- The total number of people who purchased book B -/
def B : ℕ := X + Y

theorem book_purchase_theorem :
  (A = 2 * B) ∧             -- The number of people who purchased book A is twice the number of people who purchased book B
  (Y = 500) ∧               -- The number of people who purchased both books A and B is 500
  (Y = 2 * X) ∧             -- The number of people who purchased both books A and B is twice the number of people who purchased only book B
  (Z = 1000) :=             -- The number of people who purchased only book A is 1000
by sorry

end NUMINAMATH_CALUDE_book_purchase_theorem_l1913_191360


namespace NUMINAMATH_CALUDE_equation_solutions_l1913_191349

theorem equation_solutions : 
  {x : ℝ | (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) = 1 / 6)} = {7, -2} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1913_191349


namespace NUMINAMATH_CALUDE_candies_distribution_proof_l1913_191312

def least_candies_to_remove (total_candies : ℕ) (num_friends : ℕ) : ℕ :=
  total_candies % num_friends

theorem candies_distribution_proof (total_candies : ℕ) (num_friends : ℕ) 
  (h1 : total_candies = 25) (h2 : num_friends = 4) :
  least_candies_to_remove total_candies num_friends = 1 := by
  sorry

end NUMINAMATH_CALUDE_candies_distribution_proof_l1913_191312


namespace NUMINAMATH_CALUDE_total_legs_of_daniels_animals_l1913_191373

/-- The number of legs an animal has -/
def legs (animal : String) : ℕ :=
  match animal with
  | "horse" => 4
  | "dog" => 4
  | "cat" => 4
  | "turtle" => 4
  | "goat" => 4
  | _ => 0

/-- Daniel's collection of animals -/
def daniels_animals : List (String × ℕ) :=
  [("horse", 2), ("dog", 5), ("cat", 7), ("turtle", 3), ("goat", 1)]

/-- Theorem: The total number of legs of Daniel's animals is 72 -/
theorem total_legs_of_daniels_animals :
  (daniels_animals.map (fun (animal, count) => count * legs animal)).sum = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_of_daniels_animals_l1913_191373


namespace NUMINAMATH_CALUDE_tan_2alpha_proof_l1913_191365

theorem tan_2alpha_proof (α : Real) (h : Real.sin α + 2 * Real.cos α = Real.sqrt 10 / 2) :
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_2alpha_proof_l1913_191365


namespace NUMINAMATH_CALUDE_high_school_students_l1913_191356

theorem high_school_students (high_school middle_school lower_school : ℕ) : 
  high_school = 4 * lower_school →
  high_school + lower_school = 7 * middle_school →
  middle_school = 300 →
  high_school = 1680 := by
sorry

end NUMINAMATH_CALUDE_high_school_students_l1913_191356


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l1913_191399

def first_four_primes : List Nat := [2, 3, 5, 7]

theorem arithmetic_mean_reciprocals_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length : ℚ) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l1913_191399


namespace NUMINAMATH_CALUDE_percentage_both_correct_l1913_191368

/-- Given a class of students taking a test with two questions, this theorem proves
    the percentage of students who answered both questions correctly. -/
theorem percentage_both_correct
  (p_first : ℝ)  -- Probability of answering the first question correctly
  (p_second : ℝ) -- Probability of answering the second question correctly
  (p_neither : ℝ) -- Probability of answering neither question correctly
  (h1 : p_first = 0.65)  -- 65% answered the first question correctly
  (h2 : p_second = 0.55) -- 55% answered the second question correctly
  (h3 : p_neither = 0.20) -- 20% answered neither question correctly
  : p_first + p_second - (1 - p_neither) = 0.40 := by
  sorry

#check percentage_both_correct

end NUMINAMATH_CALUDE_percentage_both_correct_l1913_191368


namespace NUMINAMATH_CALUDE_t_shirts_per_package_l1913_191308

theorem t_shirts_per_package (total_shirts : ℕ) (num_packages : ℕ) 
  (h1 : total_shirts = 51) (h2 : num_packages = 17) : 
  total_shirts / num_packages = 3 := by
  sorry

end NUMINAMATH_CALUDE_t_shirts_per_package_l1913_191308


namespace NUMINAMATH_CALUDE_total_jumps_eq_308_l1913_191348

/-- The total number of times Joonyoung and Namyoung jumped rope --/
def total_jumps (joonyoung_freq : ℕ) (joonyoung_months : ℕ) (namyoung_freq : ℕ) (namyoung_months : ℕ) : ℕ :=
  joonyoung_freq * joonyoung_months + namyoung_freq * namyoung_months

/-- Theorem stating that the total jumps for Joonyoung and Namyoung is 308 --/
theorem total_jumps_eq_308 :
  total_jumps 56 3 35 4 = 308 := by
  sorry

end NUMINAMATH_CALUDE_total_jumps_eq_308_l1913_191348


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l1913_191330

theorem sphere_volume_ratio (r : ℝ) (hr : r > 0) : 
  (4 / 3 * Real.pi * (3 * r)^3) = 3 * ((4 / 3 * Real.pi * r^3) + (4 / 3 * Real.pi * (2 * r)^3)) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l1913_191330


namespace NUMINAMATH_CALUDE_problem_solution_l1913_191344

noncomputable def θ : ℝ := sorry

-- The terminal side of angle θ lies on the ray y = 2x (x ≥ 0)
axiom h : ∀ x : ℝ, x ≥ 0 → Real.tan θ * x = 2 * x

theorem problem_solution :
  (Real.tan θ = 2) ∧
  ((2 * Real.cos θ + 3 * Real.sin θ) / (Real.cos θ - 3 * Real.sin θ) + Real.sin θ * Real.cos θ = -6/5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1913_191344
