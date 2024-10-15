import Mathlib

namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2197_219740

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals :
  diagonals_in_nonagon = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2197_219740


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l2197_219701

/-- Given a rhombus with an area equal to that of a square with side length 8,
    and one diagonal of length 8, prove that the other diagonal has length 16. -/
theorem rhombus_diagonal_length :
  ∀ (d1 : ℝ),
  (d1 * 8 / 2 = 8 * 8) →
  d1 = 16 := by
sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l2197_219701


namespace NUMINAMATH_CALUDE_adam_gave_seven_boxes_l2197_219775

/-- The number of boxes Adam gave to his little brother -/
def boxes_given (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) : ℕ :=
  (total_boxes * pieces_per_box - pieces_left) / pieces_per_box

/-- Proof that Adam gave 7 boxes to his little brother -/
theorem adam_gave_seven_boxes :
  boxes_given 13 6 36 = 7 := by
  sorry

end NUMINAMATH_CALUDE_adam_gave_seven_boxes_l2197_219775


namespace NUMINAMATH_CALUDE_election_results_l2197_219742

theorem election_results (total_votes : ℕ) 
  (votes_A votes_B votes_C : ℕ) : 
  votes_A = (35 : ℕ) * total_votes / 100 →
  votes_B = votes_A + 1800 →
  votes_C = votes_A / 2 →
  total_votes = votes_A + votes_B + votes_C →
  total_votes = 14400 ∧
  (votes_A : ℚ) / total_votes = 35 / 100 ∧
  (votes_B : ℚ) / total_votes = 475 / 1000 ∧
  (votes_C : ℚ) / total_votes = 175 / 1000 :=
by
  sorry

#check election_results

end NUMINAMATH_CALUDE_election_results_l2197_219742


namespace NUMINAMATH_CALUDE_product_of_sines_equality_l2197_219789

theorem product_of_sines_equality : 
  (1 + Real.sin (π/12)) * (1 + Real.sin (5*π/12)) * (1 + Real.sin (7*π/12)) * (1 + Real.sin (11*π/12)) = 
  (1 + Real.sin (π/12))^2 * (1 + Real.sin (5*π/12))^2 := by
sorry

end NUMINAMATH_CALUDE_product_of_sines_equality_l2197_219789


namespace NUMINAMATH_CALUDE_five_distinct_values_of_triple_exponentiation_l2197_219785

def exponentiateThree (n : ℕ) : ℕ := 3^n

theorem five_distinct_values_of_triple_exponentiation :
  ∃! (s : Finset ℕ), 
    (∀ x ∈ s, ∃ f : ℕ → ℕ → ℕ, x = f (exponentiateThree 3) (exponentiateThree (exponentiateThree 3))) ∧ 
    s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_distinct_values_of_triple_exponentiation_l2197_219785


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l2197_219761

theorem log_sum_equals_two (a b : ℝ) (h1 : 2^a = Real.sqrt 10) (h2 : 5^b = Real.sqrt 10) :
  1/a + 1/b = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l2197_219761


namespace NUMINAMATH_CALUDE_computer_purchase_cost_l2197_219786

/-- Calculates the total cost of John's computer purchase --/
theorem computer_purchase_cost (computer_cost : ℝ) (base_video_card_cost : ℝ) 
  (monitor_foreign_cost : ℝ) (exchange_rate : ℝ) :
  computer_cost = 1500 →
  base_video_card_cost = 300 →
  monitor_foreign_cost = 200 →
  exchange_rate = 1.25 →
  ∃ total_cost : ℝ,
    total_cost = 
      (computer_cost + 
       (0.25 * computer_cost) + 
       (2.5 * base_video_card_cost * 0.88) + 
       ((0.25 * computer_cost) * 1.05) - 
       (0.07 * (computer_cost + (0.25 * computer_cost) + (2.5 * base_video_card_cost * 0.88))) + 
       (monitor_foreign_cost / exchange_rate)) ∧
    total_cost = 2536.30 := by
  sorry

end NUMINAMATH_CALUDE_computer_purchase_cost_l2197_219786


namespace NUMINAMATH_CALUDE_place_mat_length_l2197_219746

theorem place_mat_length (r : ℝ) (n : ℕ) (h_r : r = 5) (h_n : n = 8) : 
  let x := 2 * r * Real.sin (π / (2 * n))
  x = r * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

#check place_mat_length

end NUMINAMATH_CALUDE_place_mat_length_l2197_219746


namespace NUMINAMATH_CALUDE_equal_utility_implies_u_equals_four_l2197_219729

def sunday_utility (u : ℝ) : ℝ := 2 * u * (10 - 2 * u)
def monday_utility (u : ℝ) : ℝ := 2 * (4 - 2 * u) * (2 * u + 4)

theorem equal_utility_implies_u_equals_four :
  ∀ u : ℝ, sunday_utility u = monday_utility u → u = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_utility_implies_u_equals_four_l2197_219729


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l2197_219796

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 77 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l2197_219796


namespace NUMINAMATH_CALUDE_stock_shares_calculation_l2197_219709

/-- Represents the number of shares for each stock --/
structure StockShares where
  v : ℕ
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the range of shares --/
def calculateRange (shares : StockShares) : ℕ :=
  max shares.v (max shares.w (max shares.x (max shares.y shares.z))) -
  min shares.v (min shares.w (min shares.x (min shares.y shares.z)))

/-- The main theorem to prove --/
theorem stock_shares_calculation (initial : StockShares) (y : ℕ) :
  initial.v = 68 →
  initial.w = 112 →
  initial.x = 56 →
  initial.z = 45 →
  initial.y = y →
  let final : StockShares := {
    v := initial.v,
    w := initial.w,
    x := initial.x - 20,
    y := initial.y + 23,
    z := initial.z
  }
  calculateRange final - calculateRange initial = 14 →
  y = 50 := by
  sorry

#check stock_shares_calculation

end NUMINAMATH_CALUDE_stock_shares_calculation_l2197_219709


namespace NUMINAMATH_CALUDE_matrix_power_difference_l2197_219753

/-- Given a 2x2 matrix B, prove that B^30 - 3B^29 equals the specified result -/
theorem matrix_power_difference (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B = ![![2, 4], ![0, 1]]) : 
  B^30 - 3 * B^29 = ![![-2, 0], ![0, 2]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_difference_l2197_219753


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_l2197_219772

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/6 - y^2/3 = 1

-- Define the right focus of the hyperbola
def right_focus_hyperbola (x y : ℝ) : Prop := x = 3 ∧ y = 0

-- Theorem statement
theorem parabola_hyperbola_focus (p : ℝ) : 
  (∃ x y : ℝ, parabola p x y ∧ right_focus_hyperbola x y) → p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_l2197_219772


namespace NUMINAMATH_CALUDE_rectangles_in_7x4_grid_l2197_219736

/-- The number of rectangles in a grid -/
def num_rectangles (columns rows : ℕ) : ℕ :=
  (columns + 1).choose 2 * (rows + 1).choose 2

/-- Theorem: In a 7x4 grid, the number of rectangles is 280 -/
theorem rectangles_in_7x4_grid :
  num_rectangles 7 4 = 280 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_7x4_grid_l2197_219736


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l2197_219748

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 7

theorem probability_all_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 8 / 6435 :=
sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l2197_219748


namespace NUMINAMATH_CALUDE_alyssas_allowance_l2197_219791

theorem alyssas_allowance (allowance : ℝ) : 
  (allowance / 2 + 8 = 12) → allowance = 8 := by
  sorry

end NUMINAMATH_CALUDE_alyssas_allowance_l2197_219791


namespace NUMINAMATH_CALUDE_f_value_theorem_l2197_219755

def is_prime (p : ℕ) : Prop := ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def f_property (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 1 → ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

theorem f_value_theorem (f : ℕ → ℝ) (h1 : f_property f) 
  (h2 : f (2^2007) + f (3^2008) + f (5^2009) = 2006) :
  f (2007^2) + f (2008^3) + f (2009^5) = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_value_theorem_l2197_219755


namespace NUMINAMATH_CALUDE_power_sum_equals_six_l2197_219781

theorem power_sum_equals_six : 2 - 2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 + 2^10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_six_l2197_219781


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2197_219737

theorem inscribed_cube_volume (large_cube_side : ℝ) (sphere_diameter : ℝ) 
  (small_cube_diagonal : ℝ) (small_cube_side : ℝ) (small_cube_volume : ℝ) :
  large_cube_side = 12 →
  sphere_diameter = large_cube_side →
  small_cube_diagonal = sphere_diameter →
  small_cube_diagonal = small_cube_side * Real.sqrt 3 →
  small_cube_side = 12 / Real.sqrt 3 →
  small_cube_volume = small_cube_side ^ 3 →
  small_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2197_219737


namespace NUMINAMATH_CALUDE_journey_takes_four_days_l2197_219745

/-- Represents the journey of a young man returning home from vacation. -/
structure Journey where
  totalDistance : ℕ
  firstLegDistance : ℕ
  secondLegDistance : ℕ
  totalDays : ℕ
  remainingDays : ℕ

/-- Checks if the journey satisfies the given conditions. -/
def isValidJourney (j : Journey) : Prop :=
  j.totalDistance = j.firstLegDistance + j.secondLegDistance ∧
  j.firstLegDistance = 246 ∧
  j.secondLegDistance = 276 ∧
  j.totalDays - j.remainingDays = j.remainingDays / 2 + 1 ∧
  j.totalDays > 0 ∧
  j.remainingDays > 0

/-- Theorem stating that the journey takes 4 days in total. -/
theorem journey_takes_four_days :
  ∃ (j : Journey), isValidJourney j ∧ j.totalDays = 4 :=
sorry


end NUMINAMATH_CALUDE_journey_takes_four_days_l2197_219745


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2197_219724

theorem complex_fraction_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2197_219724


namespace NUMINAMATH_CALUDE_complex_simplification_l2197_219752

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The property of the imaginary unit -/
axiom i_squared : i * i = -1

/-- The theorem to prove -/
theorem complex_simplification :
  3 * (2 - 2 * i) + 2 * i * (3 + i) = (4 : ℂ) := by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2197_219752


namespace NUMINAMATH_CALUDE_max_sum_product_l2197_219788

theorem max_sum_product (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_eq_200 : a + b + c + d = 200) : 
  a * b + b * c + c * d + d * a ≤ 10000 := by
sorry

end NUMINAMATH_CALUDE_max_sum_product_l2197_219788


namespace NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l2197_219792

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 100 → xy = 40 → x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l2197_219792


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2197_219756

theorem arithmetic_calculation : 8 / 2 + (-3) * 4 - (-10) + 6 * (-2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2197_219756


namespace NUMINAMATH_CALUDE_exists_quadratic_with_two_n_roots_l2197_219700

-- Define a quadratic polynomial
def QuadraticPolynomial : Type := ℝ → ℝ

-- Define the property of having 2n distinct real roots for n-fold composition
def HasTwoNRoots (f : QuadraticPolynomial) : Prop :=
  ∀ n : ℕ, ∃! (roots : Finset ℝ), (roots.card = 2 * n) ∧ 
    (∀ x ∈ roots, (f^[n]) x = 0) ∧
    (∀ x : ℝ, (f^[n]) x = 0 → x ∈ roots)

-- The theorem to be proved
theorem exists_quadratic_with_two_n_roots :
  ∃ f : QuadraticPolynomial, HasTwoNRoots f :=
sorry


end NUMINAMATH_CALUDE_exists_quadratic_with_two_n_roots_l2197_219700


namespace NUMINAMATH_CALUDE_sin_shift_l2197_219764

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l2197_219764


namespace NUMINAMATH_CALUDE_total_cost_proof_l2197_219704

def hand_mitts_cost : ℚ := 14
def apron_cost : ℚ := 16
def utensils_cost : ℚ := 10
def knife_cost : ℚ := 2 * utensils_cost
def discount_rate : ℚ := 0.25
def tax_rate : ℚ := 0.08
def num_recipients : ℕ := 8

def total_cost : ℚ :=
  let set_cost := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let total_before_discount := num_recipients * set_cost
  let discounted_total := total_before_discount * (1 - discount_rate)
  discounted_total * (1 + tax_rate)

theorem total_cost_proof : total_cost = 388.8 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l2197_219704


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2197_219716

theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ x ∈ Set.Icc (-1/2) (-1/3)) → 
  (∀ x, ax^2 - b*x - 1 < 0 ↔ x ∈ Set.Ioo 2 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2197_219716


namespace NUMINAMATH_CALUDE_inequality_problem_l2197_219708

theorem inequality_problem :
  -- Part 1: Maximum value of m
  (∃ M : ℝ, (∀ m : ℝ, (∃ x : ℝ, |x - 2| - |x + 3| ≥ |m + 1|) → m ≤ M) ∧
    (∃ x : ℝ, |x - 2| - |x + 3| ≥ |M + 1|) ∧
    M = 4) ∧
  -- Part 2: Inequality for positive a, b, c
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + 2*b + c = 4 →
    1 / (a + b) + 1 / (b + c) ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l2197_219708


namespace NUMINAMATH_CALUDE_divisible_by_24_l2197_219780

theorem divisible_by_24 (n : ℕ) : ∃ k : ℤ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l2197_219780


namespace NUMINAMATH_CALUDE_parabola_c_value_l2197_219747

-- Define the parabola equation
def parabola (x b c : ℝ) : ℝ := x^2 + b*x + c

-- Theorem statement
theorem parabola_c_value :
  ∀ b c : ℝ,
  (parabola 2 b c = 12) ∧ (parabola (-2) b c = 8) →
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2197_219747


namespace NUMINAMATH_CALUDE_valid_lineup_count_l2197_219767

/-- The total number of players in the basketball team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def quadruplets : ℕ := 4

/-- The size of the starting lineup -/
def lineup_size : ℕ := 6

/-- The maximum number of quadruplets allowed in the starting lineup -/
def max_quadruplets : ℕ := 2

/-- The number of ways to choose the starting lineup with the given restrictions -/
def valid_lineups : ℕ := 7062

theorem valid_lineup_count : 
  (Nat.choose total_players lineup_size) - 
  (Nat.choose quadruplets 3 * Nat.choose (total_players - quadruplets) (lineup_size - 3) +
   Nat.choose quadruplets 4 * Nat.choose (total_players - quadruplets) (lineup_size - 4)) = 
  valid_lineups :=
sorry

end NUMINAMATH_CALUDE_valid_lineup_count_l2197_219767


namespace NUMINAMATH_CALUDE_card_draw_probability_l2197_219762

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)

/-- The probability of drawing a specific sequence of cards -/
def draw_probability (d : Deck) (spades : Nat) (tens : Nat) (queens : Nat) : Rat :=
  let p1 := spades / d.cards
  let p2 := tens / (d.cards - 1)
  let p3 := queens / (d.cards - 2)
  p1 * p2 * p3

/-- The theorem to prove -/
theorem card_draw_probability :
  let d := Deck.mk 52
  let spades := 13
  let tens := 4
  let queens := 4
  draw_probability d spades tens queens = 17 / 11050 :=
by
  sorry


end NUMINAMATH_CALUDE_card_draw_probability_l2197_219762


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l2197_219793

/-- Given an arithmetic sequence with first term a₁ = -3, last term aₙ = 45, 
    and common difference d = 3, prove that the number of terms n is 17. -/
theorem arithmetic_sequence_count : 
  ∀ (n : ℕ) (a : ℕ → ℤ), 
    a 1 = -3 ∧ 
    (∀ k, a (k + 1) = a k + 3) ∧ 
    a n = 45 → 
    n = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l2197_219793


namespace NUMINAMATH_CALUDE_bath_frequency_l2197_219787

/-- 
Given a person who takes a bath B times per week and a shower once per week,
prove that if they clean themselves 156 times in 52 weeks, then B = 2.
-/
theorem bath_frequency (B : ℕ) 
  (h1 : B + 1 = (156 : ℕ) / 52) : B = 2 := by
  sorry

#check bath_frequency

end NUMINAMATH_CALUDE_bath_frequency_l2197_219787


namespace NUMINAMATH_CALUDE_prime_sum_square_l2197_219799

theorem prime_sum_square (p q r : ℕ) (n : ℕ+) :
  Prime p → Prime q → Prime r → p^(n:ℕ) + q^(n:ℕ) = r^2 → n = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_square_l2197_219799


namespace NUMINAMATH_CALUDE_calculation_difference_l2197_219766

def correct_calculation : ℤ := 12 - (3 * 4)

def incorrect_calculation : ℤ := 12 - 3 * 4

theorem calculation_difference :
  correct_calculation - incorrect_calculation = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l2197_219766


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2197_219763

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2197_219763


namespace NUMINAMATH_CALUDE_geometric_sequence_roots_l2197_219718

theorem geometric_sequence_roots (m n : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^2 - m*a + 2) * (a^2 - n*a + 2) = 0 ∧
    (b^2 - m*b + 2) * (b^2 - n*b + 2) = 0 ∧
    (c^2 - m*c + 2) * (c^2 - n*c + 2) = 0 ∧
    (d^2 - m*d + 2) * (d^2 - n*d + 2) = 0 ∧
    a = 1/2 ∧
    ∃ r : ℝ, b = a*r ∧ c = b*r ∧ d = c*r) →
  |m - n| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_roots_l2197_219718


namespace NUMINAMATH_CALUDE_geometric_jump_sequence_ratio_range_l2197_219759

/-- A sequence is a jump sequence if for any three consecutive terms,
    the product (a_i - a_i+2)(a_i+2 - a_i+1) is positive. -/
def is_jump_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, (a i - a (i + 2)) * (a (i + 2) - a (i + 1)) > 0

/-- A sequence is geometric with ratio q if each term is q times the previous term. -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_jump_sequence_ratio_range
  (a : ℕ → ℝ) (q : ℝ) (h_geom : is_geometric_sequence a q) (h_jump : is_jump_sequence a) :
  q ∈ Set.Ioo (-1 : ℝ) 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_jump_sequence_ratio_range_l2197_219759


namespace NUMINAMATH_CALUDE_pizza_combinations_l2197_219705

def num_toppings : ℕ := 8

theorem pizza_combinations (n : ℕ) (h : n = num_toppings) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

#eval num_toppings.choose 1 + num_toppings.choose 2 + num_toppings.choose 3

end NUMINAMATH_CALUDE_pizza_combinations_l2197_219705


namespace NUMINAMATH_CALUDE_song_listens_proof_l2197_219703

/-- Given a song with an initial number of listens that doubles each month for 3 months,
    resulting in a total of 900,000 listens, prove that the initial number of listens is 60,000. -/
theorem song_listens_proof (L : ℕ) : 
  (L + 2*L + 4*L + 8*L = 900000) → L = 60000 := by
  sorry

end NUMINAMATH_CALUDE_song_listens_proof_l2197_219703


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l2197_219706

theorem binomial_coefficient_divisible_by_prime (p k : ℕ) : 
  Prime p → 0 < k → k < p → ∃ m : ℕ, Nat.choose p k = p * m := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l2197_219706


namespace NUMINAMATH_CALUDE_strawberry_jelly_sales_l2197_219790

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ
  apricot : ℕ
  mixedFruit : ℕ

/-- Conditions for jelly sales -/
def validJellySales (s : JellySales) : Prop :=
  s.grape = 4 * s.strawberry ∧
  s.raspberry = 2 * s.plum ∧
  s.apricot = s.grape / 2 ∧
  s.mixedFruit = 3 * s.raspberry ∧
  s.raspberry = s.grape / 3 ∧
  s.plum = 8

theorem strawberry_jelly_sales (s : JellySales) (h : validJellySales s) : s.strawberry = 12 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_jelly_sales_l2197_219790


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2197_219735

/-- Given a geometric sequence {a_n} with positive terms, where 4a_3, a_5, and 2a_4 form an arithmetic sequence, and a_1 = 1, prove that S_4 = 15 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence
  4 * a 3 + a 5 = 2 * (2 * a 4) →  -- Arithmetic sequence condition
  a 1 = 1 →  -- First term is 1
  a 1 + a 2 + a 3 + a 4 = 15 :=  -- S_4 = 15
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2197_219735


namespace NUMINAMATH_CALUDE_negation_equivalence_l2197_219774

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 5^x + Real.sin x ≤ 0) ↔ (∀ x : ℝ, 5^x + Real.sin x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2197_219774


namespace NUMINAMATH_CALUDE_percentage_needed_is_35_l2197_219734

/-- The percentage of total marks needed to pass, given Pradeep's score, 
    the marks he fell short by, and the maximum marks. -/
def percentage_to_pass (pradeep_score : ℕ) (marks_short : ℕ) (max_marks : ℕ) : ℚ :=
  ((pradeep_score + marks_short : ℚ) / max_marks) * 100

/-- Theorem stating that the percentage needed to pass is 35% -/
theorem percentage_needed_is_35 (pradeep_score marks_short max_marks : ℕ) 
  (h1 : pradeep_score = 185)
  (h2 : marks_short = 25)
  (h3 : max_marks = 600) :
  percentage_to_pass pradeep_score marks_short max_marks = 35 := by
  sorry

#eval percentage_to_pass 185 25 600

end NUMINAMATH_CALUDE_percentage_needed_is_35_l2197_219734


namespace NUMINAMATH_CALUDE_systematic_sampling_probabilities_l2197_219769

/-- Systematic sampling probabilities -/
theorem systematic_sampling_probabilities
  (population : ℕ)
  (sample_size : ℕ)
  (removed : ℕ)
  (h_pop : population = 1005)
  (h_sample : sample_size = 50)
  (h_removed : removed = 5) :
  (removed : ℚ) / population = 5 / 1005 ∧
  (sample_size : ℚ) / population = 50 / 1005 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_probabilities_l2197_219769


namespace NUMINAMATH_CALUDE_another_rational_right_triangle_with_same_area_l2197_219778

/-- Given a right triangle with rational sides and area S, 
    there exists another right triangle with rational sides and area S -/
theorem another_rational_right_triangle_with_same_area 
  (a b c S : ℚ) : 
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem for right triangle
  (S = (1/2) * a * b) →  -- Area formula
  (∃ (a' b' c' : ℚ), 
    a'^2 + b'^2 = c'^2 ∧  -- New triangle is right-angled
    (1/2) * a' * b' = S ∧  -- New triangle has the same area
    (a' ≠ a ∨ b' ≠ b ∨ c' ≠ c))  -- New triangle is different from the original
  := by sorry

end NUMINAMATH_CALUDE_another_rational_right_triangle_with_same_area_l2197_219778


namespace NUMINAMATH_CALUDE_doctor_nurse_ratio_l2197_219777

theorem doctor_nurse_ratio (total : ℕ) (nurses : ℕ) (h1 : total = 280) (h2 : nurses = 180) :
  (total - nurses) / (Nat.gcd (total - nurses) nurses) = 5 ∧
  nurses / (Nat.gcd (total - nurses) nurses) = 9 :=
by sorry

end NUMINAMATH_CALUDE_doctor_nurse_ratio_l2197_219777


namespace NUMINAMATH_CALUDE_slower_train_speed_l2197_219738

/-- Proves the speed of a slower train given specific conditions --/
theorem slower_train_speed
  (train_length : ℝ)
  (faster_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 60)
  (h2 : faster_speed = 48)
  (h3 : passing_time = 36)
  : ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * (5 / 18) * passing_time = 2 * train_length :=
by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l2197_219738


namespace NUMINAMATH_CALUDE_coin_flips_count_l2197_219765

theorem coin_flips_count (heads : ℕ) (tails : ℕ) : heads = 65 → tails = heads + 81 → heads + tails = 211 := by
  sorry

end NUMINAMATH_CALUDE_coin_flips_count_l2197_219765


namespace NUMINAMATH_CALUDE_cube_of_ten_expansion_l2197_219771

theorem cube_of_ten_expansion : 9^3 + 3*(9^2) + 3*9 + 1 = 1000 := by sorry

end NUMINAMATH_CALUDE_cube_of_ten_expansion_l2197_219771


namespace NUMINAMATH_CALUDE_remaining_fabric_area_l2197_219795

/-- Calculates the remaining fabric area after cutting curtains -/
theorem remaining_fabric_area (bolt_length bolt_width living_room_length living_room_width bedroom_length bedroom_width : ℝ) 
  (h1 : bolt_length = 16)
  (h2 : bolt_width = 12)
  (h3 : living_room_length = 4)
  (h4 : living_room_width = 6)
  (h5 : bedroom_length = 2)
  (h6 : bedroom_width = 4) :
  bolt_length * bolt_width - (living_room_length * living_room_width + bedroom_length * bedroom_width) = 160 := by
  sorry

#check remaining_fabric_area

end NUMINAMATH_CALUDE_remaining_fabric_area_l2197_219795


namespace NUMINAMATH_CALUDE_fraction_to_whole_number_l2197_219721

theorem fraction_to_whole_number : 
  (∃ n : ℤ, (12 : ℚ) / 2 = n) ∧
  (∀ n : ℤ, (8 : ℚ) / 6 ≠ n) ∧
  (∀ n : ℤ, (9 : ℚ) / 5 ≠ n) ∧
  (∀ n : ℤ, (10 : ℚ) / 4 ≠ n) ∧
  (∀ n : ℤ, (11 : ℚ) / 3 ≠ n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_whole_number_l2197_219721


namespace NUMINAMATH_CALUDE_mode_median_constant_l2197_219712

/-- Represents the age distribution of a club --/
structure AgeDistribution where
  age13 : ℕ
  age14 : ℕ
  age15 : ℕ
  age16 : ℕ
  age17 : ℕ
  total : ℕ
  sum_eq_total : age13 + age14 + age15 + age16 + age17 = total

/-- The age distribution of the club --/
def clubDistribution (x : ℕ) : AgeDistribution where
  age13 := 5
  age14 := 12
  age15 := x
  age16 := 11 - x
  age17 := 2
  total := 30
  sum_eq_total := by sorry

/-- The mode of the age distribution --/
def mode (d : AgeDistribution) : ℕ := 
  max d.age13 (max d.age14 (max d.age15 (max d.age16 d.age17)))

/-- The median of the age distribution --/
def median (d : AgeDistribution) : ℚ := 14

theorem mode_median_constant (x : ℕ) : 
  mode (clubDistribution x) = 14 ∧ median (clubDistribution x) = 14 := by sorry

end NUMINAMATH_CALUDE_mode_median_constant_l2197_219712


namespace NUMINAMATH_CALUDE_calculation_proof_l2197_219702

theorem calculation_proof : (10^8 / (2 * 10^5)) - 50 = 450 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2197_219702


namespace NUMINAMATH_CALUDE_infinite_partition_numbers_l2197_219732

theorem infinite_partition_numbers : ∃ (f : ℕ → ℕ), Infinite {n : ℕ | ∃ k, n = f k ∧ n % 4 = 1 ∧ (3 * n * (3 * n + 1) / 2) % (6 * n) = 0} :=
sorry

end NUMINAMATH_CALUDE_infinite_partition_numbers_l2197_219732


namespace NUMINAMATH_CALUDE_prob_heads_tails_heads_l2197_219797

/-- A fair coin has equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of a sequence of independent events is the product of their individual probabilities -/
def prob_independent_events (p q r : ℝ) : ℝ := p * q * r

/-- The probability of getting heads, tails, then heads when flipping a fair coin three times is 1/8 -/
theorem prob_heads_tails_heads :
  ∀ (p : ℝ), fair_coin p →
  prob_independent_events p p p = 1/8 :=
sorry

end NUMINAMATH_CALUDE_prob_heads_tails_heads_l2197_219797


namespace NUMINAMATH_CALUDE_log_equation_holds_l2197_219773

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 3) * (Real.log 5 / Real.log x) = Real.log 5 / Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_holds_l2197_219773


namespace NUMINAMATH_CALUDE_female_officers_count_l2197_219744

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percentage : ℚ) 
  (h1 : total_on_duty = 144)
  (h2 : female_on_duty_percentage = 18 / 100)
  (h3 : (total_on_duty / 2 : ℚ) = female_on_duty_percentage * female_total) :
  female_total = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2197_219744


namespace NUMINAMATH_CALUDE_max_late_all_days_l2197_219717

theorem max_late_all_days (total_late : ℕ) (late_monday : ℕ) (late_tuesday : ℕ) (late_wednesday : ℕ)
  (h_total : total_late = 30)
  (h_monday : late_monday = 20)
  (h_tuesday : late_tuesday = 13)
  (h_wednesday : late_wednesday = 7) :
  ∃ (x : ℕ), x ≤ 5 ∧ 
    x ≤ late_monday ∧ 
    x ≤ late_tuesday ∧ 
    x ≤ late_wednesday ∧
    (late_monday - x) + (late_tuesday - x) + (late_wednesday - x) + x ≤ total_late ∧
    ∀ (y : ℕ), y > x → 
      (y > late_monday ∨ y > late_tuesday ∨ y > late_wednesday ∨
       (late_monday - y) + (late_tuesday - y) + (late_wednesday - y) + y > total_late) :=
by sorry

end NUMINAMATH_CALUDE_max_late_all_days_l2197_219717


namespace NUMINAMATH_CALUDE_meal_cost_theorem_l2197_219783

theorem meal_cost_theorem (initial_people : ℕ) (additional_people : ℕ) (share_decrease : ℚ) :
  initial_people = 5 →
  additional_people = 3 →
  share_decrease = 15 →
  let total_people := initial_people + additional_people
  let total_cost := (initial_people * share_decrease * total_people) / (total_people - initial_people)
  total_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_meal_cost_theorem_l2197_219783


namespace NUMINAMATH_CALUDE_complex_power_equality_smallest_power_is_minimal_l2197_219760

/-- The smallest positive integer n for which (a+bi)^(n+1) = (a-bi)^(n+1) holds for some positive real a and b -/
def smallest_power : ℕ := 3

theorem complex_power_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.mk a b)^4 = (Complex.mk a (-b))^4 → b / a = 1 :=
by sorry

theorem smallest_power_is_minimal (n : ℕ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  n < smallest_power →
  (Complex.mk a b)^(n + 1) ≠ (Complex.mk a (-b))^(n + 1) :=
by sorry

#check smallest_power
#check complex_power_equality
#check smallest_power_is_minimal

end NUMINAMATH_CALUDE_complex_power_equality_smallest_power_is_minimal_l2197_219760


namespace NUMINAMATH_CALUDE_series_sum_equals_35_over_13_l2197_219722

/-- Definition of the sequence G_n -/
def G : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => G (n + 1) + 2 * G n

/-- The sum of the series -/
noncomputable def seriesSum : ℚ := ∑' n, G n / 5^n

/-- Theorem stating that the sum of the series equals 35/13 -/
theorem series_sum_equals_35_over_13 : seriesSum = 35/13 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_35_over_13_l2197_219722


namespace NUMINAMATH_CALUDE_disinfectant_sales_analysis_l2197_219723

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (x : ℤ) : ℤ := -5 * x + 150

/-- Represents the daily profit as a function of selling price -/
def profit (x : ℤ) : ℤ := (x - 8) * sales_quantity x

theorem disinfectant_sales_analysis 
  (h1 : ∀ x : ℤ, 8 ≤ x → x ≤ 15 → sales_quantity x = -5 * x + 150)
  (h2 : sales_quantity 9 = 105)
  (h3 : sales_quantity 11 = 95)
  (h4 : sales_quantity 13 = 85) :
  (∀ x : ℤ, 8 ≤ x → x ≤ 15 → sales_quantity x = -5 * x + 150) ∧
  (profit 13 = 425) ∧
  (∀ x : ℤ, 8 ≤ x → x ≤ 15 → profit x ≤ 525) ∧
  (profit 15 = 525) := by
sorry


end NUMINAMATH_CALUDE_disinfectant_sales_analysis_l2197_219723


namespace NUMINAMATH_CALUDE_volunteer_selection_count_l2197_219782

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of students to be selected -/
def num_selected : ℕ := 4

/-- The total number of students -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to select 4 students from 7 students -/
def total_selections : ℕ := Nat.choose total_students num_selected

/-- The number of ways to select 4 boys from 4 boys -/
def all_boys_selections : ℕ := Nat.choose num_boys num_selected

theorem volunteer_selection_count :
  total_selections - all_boys_selections = 34 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_count_l2197_219782


namespace NUMINAMATH_CALUDE_savings_theorem_l2197_219741

/-- Represents the savings and interest calculations for Dick and Jane --/
structure Savings where
  dick_1989 : ℝ
  jane_1989 : ℝ
  dick_increase_rate : ℝ
  interest_rate : ℝ

/-- Calculates the total savings of Dick and Jane in 1990 --/
def total_savings_1990 (s : Savings) : ℝ :=
  (s.dick_1989 * (1 + s.dick_increase_rate) + s.dick_1989) * (1 + s.interest_rate) +
  s.jane_1989 * (1 + s.interest_rate)

/-- Calculates the percent change in Jane's savings from 1989 to 1990 --/
def jane_savings_percent_change (s : Savings) : ℝ := 0

/-- Theorem stating the total savings in 1990 and Jane's savings percent change --/
theorem savings_theorem (s : Savings) 
  (h1 : s.dick_1989 = 5000)
  (h2 : s.jane_1989 = 3000)
  (h3 : s.dick_increase_rate = 0.1)
  (h4 : s.interest_rate = 0.03) :
  total_savings_1990 s = 8740 ∧ jane_savings_percent_change s = 0 := by
  sorry

end NUMINAMATH_CALUDE_savings_theorem_l2197_219741


namespace NUMINAMATH_CALUDE_rhombus_properties_l2197_219710

structure Rhombus (O : ℝ × ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  side_length : ℝ
  OB_length : ℝ
  OD_length : ℝ
  is_rhombus : side_length = 4 ∧ OB_length = 6 ∧ OD_length = 6

def on_semicircle (A : ℝ × ℝ) : Prop :=
  let (x, y) := A
  (x - 2)^2 + y^2 = 4 ∧ 2 ≤ x ∧ x ≤ 4

theorem rhombus_properties (r : Rhombus (0, 0)) :
  (|r.A.1 * r.C.1 + r.A.2 * r.C.2| = 36) ∧
  (∀ A', on_semicircle A' → 
    ∃ C', r.C = C' → (C' = (5, 5) ∨ C' = (5, -5))) :=
sorry

end NUMINAMATH_CALUDE_rhombus_properties_l2197_219710


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l2197_219758

theorem consecutive_numbers_sum (n : ℕ) : 
  n + (n + 1) + (n + 2) = 60 → (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l2197_219758


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2197_219730

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_sequence :
  let a₁ := 2
  let a₂ := 7
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 20 = 97 := by sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2197_219730


namespace NUMINAMATH_CALUDE_cindy_calculation_l2197_219727

theorem cindy_calculation (x : ℝ) : 
  ((x - 10) / 5 = 40) → ((x - 4) / 10 = 20.6) := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l2197_219727


namespace NUMINAMATH_CALUDE_staircase_polygon_perimeter_staircase_polygon_area_l2197_219715

/-- A polygonal region formed by removing a 3x4 rectangle from an 8x12 rectangle -/
structure StaircasePolygon where
  width : ℕ := 12
  height : ℕ := 8
  small_width : ℕ := 3
  small_height : ℕ := 4
  area : ℕ := 86
  stair_side_length : ℕ := 1
  stair_side_count : ℕ := 12

/-- The perimeter of a StaircasePolygon is 44 -/
theorem staircase_polygon_perimeter (p : StaircasePolygon) : 
  p.width + p.height + (p.width - p.small_width) + (p.height - p.small_height) + p.stair_side_count * p.stair_side_length = 44 := by
  sorry

/-- The area of a StaircasePolygon is consistent with its dimensions -/
theorem staircase_polygon_area (p : StaircasePolygon) :
  p.area = p.width * p.height - p.small_width * p.small_height := by
  sorry

end NUMINAMATH_CALUDE_staircase_polygon_perimeter_staircase_polygon_area_l2197_219715


namespace NUMINAMATH_CALUDE_x_value_l2197_219749

theorem x_value : ∃ x : ℝ, (3 * x = (16 - x) + 4) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2197_219749


namespace NUMINAMATH_CALUDE_average_speed_two_segments_l2197_219754

/-- Calculate the average speed of a two-segment journey -/
theorem average_speed_two_segments 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 50) 
  (h2 : speed1 = 20) 
  (h3 : distance2 = 20) 
  (h4 : speed2 = 40) : 
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 70 / 3 := by
  sorry

#eval (70 : ℚ) / 3

end NUMINAMATH_CALUDE_average_speed_two_segments_l2197_219754


namespace NUMINAMATH_CALUDE_cos_shift_right_l2197_219779

theorem cos_shift_right (x : ℝ) :
  2 * Real.cos (2 * (x - π/8)) = 2 * Real.cos (2 * x - π/4) :=
by sorry

end NUMINAMATH_CALUDE_cos_shift_right_l2197_219779


namespace NUMINAMATH_CALUDE_sum_of_radii_is_eight_l2197_219731

/-- A circle with center C that is tangent to the positive x and y-axes
    and externally tangent to a circle centered at (3,0) with radius 1 -/
def CircleC (r : ℝ) : Prop :=
  (∃ C : ℝ × ℝ, C.1 = r ∧ C.2 = r) ∧  -- Center of circle C is at (r,r)
  ((r - 3)^2 + r^2 = (r + 1)^2)  -- External tangency condition

/-- The theorem stating that the sum of all possible radii of CircleC is 8 -/
theorem sum_of_radii_is_eight :
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ CircleC r₁ ∧ CircleC r₂ ∧ r₁ + r₂ = 8 :=
sorry

end NUMINAMATH_CALUDE_sum_of_radii_is_eight_l2197_219731


namespace NUMINAMATH_CALUDE_train_speed_l2197_219768

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 2500) (h2 : time = 100) :
  length / time = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2197_219768


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2197_219725

-- Define the total number of balls
def total_balls : ℕ := 7

-- Define the number of red balls
def red_balls : ℕ := 2

-- Define the number of black balls
def black_balls : ℕ := 4

-- Define the number of white balls
def white_balls : ℕ := 1

-- Define the probability of drawing a red ball
def prob_red_ball : ℚ := red_balls / total_balls

-- Theorem statement
theorem probability_of_red_ball :
  prob_red_ball = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2197_219725


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l2197_219750

theorem weight_loss_challenge (W : ℝ) (x : ℝ) (h : x > 0) :
  W * (1 - x / 100 + 2 / 100) = W * (100 - 12.28) / 100 →
  x = 14.28 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l2197_219750


namespace NUMINAMATH_CALUDE_largest_circle_area_l2197_219794

/-- The area of the largest circle formed from a string with length equal to the perimeter of a 15x9 rectangle is 576/π. -/
theorem largest_circle_area (length width : ℝ) (h1 : length = 15) (h2 : width = 9) :
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * π)
  π * radius^2 = 576 / π :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_area_l2197_219794


namespace NUMINAMATH_CALUDE_box_surface_areas_and_cost_l2197_219714

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular box -/
def surfaceArea (box : BoxDimensions) : ℝ :=
  2 * (box.length * box.width + box.length * box.height + box.width * box.height)

/-- Theorem about the surface areas of two boxes and their cost -/
theorem box_surface_areas_and_cost 
  (a b c : ℝ) 
  (small_box : BoxDimensions := ⟨a, b, c⟩)
  (large_box : BoxDimensions := ⟨2*a, 2*b, 1.5*c⟩)
  (cardboard_cost_per_sqm : ℝ := 15) : 
  (surfaceArea small_box + surfaceArea large_box = 10*a*b + 8*b*c + 8*a*c) ∧ 
  (surfaceArea large_box - surfaceArea small_box = 6*a*b + 4*b*c + 4*a*c) ∧
  (a = 20 → b = 10 → c = 15 → 
    cardboard_cost_per_sqm * (surfaceArea small_box + surfaceArea large_box) / 10000 = 8.4) :=
by sorry


end NUMINAMATH_CALUDE_box_surface_areas_and_cost_l2197_219714


namespace NUMINAMATH_CALUDE_polygon_sides_count_polygon_sides_count_proof_l2197_219713

theorem polygon_sides_count : ℕ → Prop :=
  fun n =>
    (n - 2) * 180 = 2 * 360 →
    n = 6

-- The proof is omitted
theorem polygon_sides_count_proof : ∃ n : ℕ, polygon_sides_count n :=
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_polygon_sides_count_proof_l2197_219713


namespace NUMINAMATH_CALUDE_mistaken_addition_problem_l2197_219757

/-- Given a two-digit number and conditions from the problem, prove it equals 49. -/
theorem mistaken_addition_problem (A B : ℕ) : 
  B = 9 →
  A * 10 + 6 + 253 = 299 →
  A * 10 + B = 49 :=
by sorry

end NUMINAMATH_CALUDE_mistaken_addition_problem_l2197_219757


namespace NUMINAMATH_CALUDE_problem_statement_l2197_219733

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = ({0, a^2, a+b} : Set ℝ) → a^2005 + b^2005 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2197_219733


namespace NUMINAMATH_CALUDE_sara_final_quarters_l2197_219728

/-- Calculates the final number of quarters Sara has after a series of transactions -/
def sara_quarters (initial : ℕ) (from_dad : ℕ) (spent : ℕ) (dollars_from_mom : ℕ) (quarters_per_dollar : ℕ) : ℕ :=
  initial + from_dad - spent + dollars_from_mom * quarters_per_dollar

/-- Theorem stating that Sara ends up with 63 quarters -/
theorem sara_final_quarters : 
  sara_quarters 21 49 15 2 4 = 63 := by
  sorry

end NUMINAMATH_CALUDE_sara_final_quarters_l2197_219728


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l2197_219743

def is_valid_digit (d : ℕ) : Prop := d = 4 ∨ d = 5

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def all_digits_valid (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

theorem largest_number_with_conditions :
  ∀ n : ℕ,
    all_digits_valid n →
    digit_sum n = 17 →
    n ≤ 5444 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l2197_219743


namespace NUMINAMATH_CALUDE_correct_reasoning_statements_l2197_219711

/-- Represents different types of reasoning -/
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
  | PartToWhole
  | GeneralToGeneral
  | GeneralToSpecific
  | SpecificToGeneral
  | SpecificToSpecific

/-- Defines the correct reasoning direction for each reasoning type -/
def correct_reasoning (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

/-- Theorem stating the correct reasoning directions for each type -/
theorem correct_reasoning_statements :
  (correct_reasoning ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (correct_reasoning ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (correct_reasoning ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end NUMINAMATH_CALUDE_correct_reasoning_statements_l2197_219711


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l2197_219751

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℝ) + ⌈x⌉ = 7 ↔ 3 < x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l2197_219751


namespace NUMINAMATH_CALUDE_goods_train_speed_is_36_l2197_219776

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 36

/-- The speed of the express train in km/h -/
def express_train_speed : ℝ := 90

/-- The time difference between the departure of the two trains in hours -/
def time_difference : ℝ := 6

/-- The time it takes for the express train to catch up with the goods train in hours -/
def catch_up_time : ℝ := 4

/-- Theorem stating that the speed of the goods train is 36 km/h -/
theorem goods_train_speed_is_36 :
  goods_train_speed * (time_difference + catch_up_time) = express_train_speed * catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_is_36_l2197_219776


namespace NUMINAMATH_CALUDE_sqrt_square_abs_l2197_219798

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_l2197_219798


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2197_219739

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l2197_219739


namespace NUMINAMATH_CALUDE_new_person_weight_l2197_219719

theorem new_person_weight (n : ℕ) (initial_avg weight_replaced increase : ℝ) :
  n = 8 →
  initial_avg = 57 →
  weight_replaced = 55 →
  increase = 1.5 →
  (n * initial_avg + (weight_replaced + increase * n) - weight_replaced) / n = initial_avg + increase →
  weight_replaced + increase * n = 67 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2197_219719


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l2197_219726

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 8) 
  (h2 : sum_first_two = 5) :
  ∃ a : ℝ, (a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6)) ∧ 
    (∃ r : ℝ, a / (1 - r) = S ∧ a + a * r = sum_first_two) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l2197_219726


namespace NUMINAMATH_CALUDE_manufacturer_profit_is_18_percent_l2197_219784

/- Define the given values -/
def customer_price : ℚ := 30.09
def retailer_profit_percentage : ℚ := 25
def wholesaler_profit_percentage : ℚ := 20
def manufacturer_cost : ℚ := 17

/- Define the calculation steps -/
def retailer_cost : ℚ := customer_price / (1 + retailer_profit_percentage / 100)
def wholesaler_price : ℚ := retailer_cost
def wholesaler_cost : ℚ := wholesaler_price / (1 + wholesaler_profit_percentage / 100)
def manufacturer_price : ℚ := wholesaler_cost

/- Define the manufacturer's profit percentage calculation -/
def manufacturer_profit_percentage : ℚ := 
  (manufacturer_price - manufacturer_cost) / manufacturer_cost * 100

/- The theorem to prove -/
theorem manufacturer_profit_is_18_percent : 
  manufacturer_profit_percentage = 18 := by sorry

end NUMINAMATH_CALUDE_manufacturer_profit_is_18_percent_l2197_219784


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2197_219720

open Set

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2197_219720


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l2197_219707

theorem root_equation_implies_expression_value (m : ℝ) : 
  m^2 + 2*m - 1 = 0 → 2*m^2 + 4*m + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l2197_219707


namespace NUMINAMATH_CALUDE_min_value_fraction_l2197_219770

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 5 → 1/x + 4/(y+1) ≤ 1/a + 4/(b+1)) ∧
  (1/x + 4/(y+1) = 3/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2197_219770
