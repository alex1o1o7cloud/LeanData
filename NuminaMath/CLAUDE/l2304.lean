import Mathlib

namespace NUMINAMATH_CALUDE_shaded_square_area_l2304_230453

/-- A configuration of four unit squares arranged in a 2x2 grid, each containing an inscribed equilateral triangle sharing an edge with the square. -/
structure SquareTriangleConfig where
  /-- The side length of each unit square -/
  unit_square_side : ℝ
  /-- The side length of each equilateral triangle -/
  triangle_side : ℝ
  /-- The side length of the larger square formed by the four unit squares -/
  large_square_side : ℝ
  /-- The side length of the shaded square formed by connecting triangle vertices -/
  shaded_square_side : ℝ
  /-- Condition: Each unit square has side length 1 -/
  unit_square_cond : unit_square_side = 1
  /-- Condition: The triangle side is equal to the unit square side -/
  triangle_side_cond : triangle_side = unit_square_side
  /-- Condition: The larger square has side length 2 -/
  large_square_cond : large_square_side = 2 * unit_square_side
  /-- Condition: The diagonal of the shaded square equals the side of the larger square -/
  shaded_square_diag_cond : shaded_square_side * Real.sqrt 2 = large_square_side

/-- The theorem stating that the area of the shaded square is 2 square units -/
theorem shaded_square_area (config : SquareTriangleConfig) : 
  config.shaded_square_side ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_square_area_l2304_230453


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2304_230473

theorem complex_modulus_problem (z : ℂ) (h : (z + 2) / (z - 2) = Complex.I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2304_230473


namespace NUMINAMATH_CALUDE_range_of_f_l2304_230485

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem range_of_f :
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, ∃ y ∈ Set.Icc 0 25, f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc 0 25 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2304_230485


namespace NUMINAMATH_CALUDE_distance_to_origin_l2304_230404

/-- The distance from the point corresponding to the complex number 2i/(1+i) to the origin is √2. -/
theorem distance_to_origin : ∃ (z : ℂ), z = (2 * Complex.I) / (1 + Complex.I) ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l2304_230404


namespace NUMINAMATH_CALUDE_tan_75_deg_l2304_230482

/-- Tangent of angle addition formula -/
axiom tan_add (a b : ℝ) : Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)

/-- Proof that tan 75° = 2 + √3 -/
theorem tan_75_deg : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  have h1 : 75 * π / 180 = 60 * π / 180 + 15 * π / 180 := by sorry
  have h2 : Real.tan (60 * π / 180) = Real.sqrt 3 := by sorry
  have h3 : Real.tan (15 * π / 180) = 2 - Real.sqrt 3 := by sorry
  sorry


end NUMINAMATH_CALUDE_tan_75_deg_l2304_230482


namespace NUMINAMATH_CALUDE_same_color_probability_l2304_230489

/-- The probability of drawing two balls of the same color from a bag containing green and white balls. -/
theorem same_color_probability (green white : ℕ) (h : green = 9 ∧ white = 8) :
  let total := green + white
  let p_green := green * (green - 1) / (total * (total - 1))
  let p_white := white * (white - 1) / (total * (total - 1))
  p_green + p_white = 8 / 17 := by
  sorry

#check same_color_probability

end NUMINAMATH_CALUDE_same_color_probability_l2304_230489


namespace NUMINAMATH_CALUDE_priyas_trip_l2304_230432

/-- Priya's trip between towns X, Y, and Z -/
theorem priyas_trip (time_x_to_z : ℝ) (speed_x_to_z : ℝ) (time_z_to_y : ℝ) :
  time_x_to_z = 5 →
  speed_x_to_z = 50 →
  time_z_to_y = 2.0833333333333335 →
  let distance_x_to_z := time_x_to_z * speed_x_to_z
  let distance_z_to_y := distance_x_to_z / 2
  let speed_z_to_y := distance_z_to_y / time_z_to_y
  speed_z_to_y = 60 := by
sorry


end NUMINAMATH_CALUDE_priyas_trip_l2304_230432


namespace NUMINAMATH_CALUDE_remainder_3_88_plus_5_mod_7_l2304_230457

theorem remainder_3_88_plus_5_mod_7 : (3^88 + 5) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_88_plus_5_mod_7_l2304_230457


namespace NUMINAMATH_CALUDE_angle_phi_value_l2304_230445

theorem angle_phi_value (φ : Real) (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : Real.sqrt 3 * Real.sin (20 * Real.pi / 180) = Real.cos φ - Real.sin φ) : 
  φ = 25 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_phi_value_l2304_230445


namespace NUMINAMATH_CALUDE_exists_all_cards_moved_no_guarantee_ace_not_next_to_empty_l2304_230418

/-- Represents a deck of cards arranged in a circle with one empty spot. -/
structure CircularDeck :=
  (cards : Fin 52 → Option (Fin 52))
  (empty_spot : Fin 53)
  (injective : ∀ i j, i ≠ j → cards i ≠ cards j)
  (surjective : ∀ c, c ≠ empty_spot → ∃ i, cards i = some c)

/-- Represents a sequence of card namings. -/
def NamingSequence := List (Fin 52)

/-- Predicate to check if a card has moved from its original position. -/
def has_moved (deck : CircularDeck) (card : Fin 52) : Prop :=
  deck.cards card ≠ some card

/-- Predicate to check if the Ace of Spades is next to the empty spot. -/
def ace_next_to_empty (deck : CircularDeck) : Prop :=
  ∃ i, deck.cards i = some 0 ∧ 
    ((i + 1) % 53 = deck.empty_spot ∨ (i - 1 + 53) % 53 = deck.empty_spot)

/-- Theorem stating that there exists a naming sequence that moves all cards. -/
theorem exists_all_cards_moved :
  ∃ (seq : NamingSequence), ∀ (initial_deck : CircularDeck),
    ∃ (final_deck : CircularDeck),
      ∀ (card : Fin 52), has_moved final_deck card :=
sorry

/-- Theorem stating that no naming sequence can guarantee the Ace of Spades
    is not next to the empty spot. -/
theorem no_guarantee_ace_not_next_to_empty :
  ∀ (seq : NamingSequence), ∃ (initial_deck : CircularDeck),
    ∃ (final_deck : CircularDeck),
      ace_next_to_empty final_deck :=
sorry

end NUMINAMATH_CALUDE_exists_all_cards_moved_no_guarantee_ace_not_next_to_empty_l2304_230418


namespace NUMINAMATH_CALUDE_logarithm_equation_l2304_230447

theorem logarithm_equation (a b c : ℝ) 
  (eq1 : Real.log 3 = 2*a - b)
  (eq2 : Real.log 5 = a + c)
  (eq3 : Real.log 8 = 3 - 3*a - 3*c)
  (eq4 : Real.log 9 = 4*a - 2*b) :
  Real.log 15 = 3*a - b + c := by sorry

end NUMINAMATH_CALUDE_logarithm_equation_l2304_230447


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2304_230401

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_positive : ∀ n, a n > 0) 
  (h_fifth : a 5 = 16) 
  (h_ninth : a 9 = 4) : 
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2304_230401


namespace NUMINAMATH_CALUDE_circumference_diameter_ratio_l2304_230425

/-- The ratio of circumference to diameter for a ring with radius 15 cm and circumference 90 cm is 3. -/
theorem circumference_diameter_ratio :
  let radius : ℝ := 15
  let circumference : ℝ := 90
  let diameter : ℝ := 2 * radius
  circumference / diameter = 3 := by
  sorry

end NUMINAMATH_CALUDE_circumference_diameter_ratio_l2304_230425


namespace NUMINAMATH_CALUDE_inequality_proof_l2304_230427

theorem inequality_proof (a b c d : ℝ) 
  (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2304_230427


namespace NUMINAMATH_CALUDE_inscribed_circles_radii_equal_l2304_230426

theorem inscribed_circles_radii_equal (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let r₁ := a * b / (a + b)
  let r₂ := a * b / (a + b)
  r₁ = r₂ := by sorry

end NUMINAMATH_CALUDE_inscribed_circles_radii_equal_l2304_230426


namespace NUMINAMATH_CALUDE_arithmetic_sequence_minimum_value_l2304_230472

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem arithmetic_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_special : a 7 = a 6 + 2 * a 5)
  (h_exists : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) :
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  (∀ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1 → 1 / m + 4 / n ≥ 3 / 2) ∧
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1 ∧ 1 / m + 4 / n = 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_minimum_value_l2304_230472


namespace NUMINAMATH_CALUDE_odd_function_property_l2304_230460

-- Define the function f on the interval [-1, 1]
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property :
  (∀ x ∈ Set.Icc (-1) 1, f x = f x) →  -- f is defined on [-1, 1]
  isOdd f →  -- f is an odd function
  (∀ x ∈ Set.Ioo 0 1, f x = x * (x - 1)) →  -- f(x) = x(x-1) for 0 < x ≤ 1
  (∀ x ∈ Set.Ioc (-1) 0, f x = -x^2 - x) :=  -- f(x) = -x^2 - x for -1 ≤ x < 0
by sorry

end NUMINAMATH_CALUDE_odd_function_property_l2304_230460


namespace NUMINAMATH_CALUDE_factor_expression_l2304_230442

theorem factor_expression (x : ℝ) : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2304_230442


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2304_230465

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 --/
def CircleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The given equation of the circle --/
def GivenEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 4*y = 16

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ x y : ℝ, GivenEquation x y ↔ CircleEquation c x y) ∧
                  c.center = (-4, 2) ∧
                  c.radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2304_230465


namespace NUMINAMATH_CALUDE_solution_value_l2304_230408

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 6*x + 11 = 24

-- Define a and b as solutions to the equation
def a_b_solutions (a b : ℝ) : Prop :=
  quadratic_equation a ∧ quadratic_equation b ∧ a ≥ b

-- Theorem statement
theorem solution_value (a b : ℝ) (h : a_b_solutions a b) :
  3*a - b = 6 + 4*Real.sqrt 22 :=
by sorry

end NUMINAMATH_CALUDE_solution_value_l2304_230408


namespace NUMINAMATH_CALUDE_max_consecutive_sum_36_l2304_230490

/-- The sum of consecutive integers from a to (a + n - 1) -/
def sum_consecutive (a : ℤ) (n : ℕ) : ℤ := n * a + (n * (n - 1)) / 2

/-- The proposition that 72 is the maximum number of consecutive integers summing to 36 -/
theorem max_consecutive_sum_36 :
  (∃ a : ℤ, sum_consecutive a 72 = 36) ∧
  (∀ n : ℕ, n > 72 → ∀ a : ℤ, sum_consecutive a n ≠ 36) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_36_l2304_230490


namespace NUMINAMATH_CALUDE_track_width_l2304_230493

theorem track_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 10 * Real.pi) : 
  r₁ - r₂ = 5 := by
  sorry

end NUMINAMATH_CALUDE_track_width_l2304_230493


namespace NUMINAMATH_CALUDE_equation_solution_l2304_230417

theorem equation_solution (a : ℝ) : (a + 3) ^ (a + 1) = 1 ↔ a = -1 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2304_230417


namespace NUMINAMATH_CALUDE_initial_kittens_initial_kittens_is_18_l2304_230464

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica : ℕ := 3

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara : ℕ := 6

/-- The number of kittens Tim has left -/
def kittens_left : ℕ := 9

/-- Theorem stating the initial number of kittens Tim had -/
theorem initial_kittens : ℕ :=
  kittens_to_jessica + kittens_to_sara + kittens_left

/-- Proof that the initial number of kittens is 18 -/
theorem initial_kittens_is_18 : initial_kittens = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_kittens_initial_kittens_is_18_l2304_230464


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l2304_230429

/-- The imaginary part of (1 - i)(2 + 4i) is 2, where i is the imaginary unit. -/
theorem imaginary_part_of_product : Complex.im ((1 - Complex.I) * (2 + 4 * Complex.I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l2304_230429


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l2304_230423

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_example : parallelogramArea 12 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l2304_230423


namespace NUMINAMATH_CALUDE_equation_solution_l2304_230461

theorem equation_solution (z : ℝ) (hz : z ≠ 0) :
  (5 * z)^10 = (20 * z)^5 ↔ z = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2304_230461


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2304_230411

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (1/x + 1/y) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2304_230411


namespace NUMINAMATH_CALUDE_circle_tangent_to_ellipse_l2304_230463

/-- Two circles of radius s are externally tangent to each other and internally tangent to the ellipse x^2 + 4y^2 = 8. The radius s of the circles is √(3/2). -/
theorem circle_tangent_to_ellipse (s : ℝ) : 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 8 ∧ (x - s)^2 + y^2 = s^2) → s = Real.sqrt (3/2) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_ellipse_l2304_230463


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l2304_230479

theorem cube_sum_inequality (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1)
  (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 ≤ 4 ∧ 
  (a + b + c + a^2 + b^2 + c^2 = 4 ↔ 
    ((a = 1 ∧ b = 1 ∧ c = -1) ∨
     (a = 1 ∧ b = -1 ∧ c = 1) ∨
     (a = -1 ∧ b = 1 ∧ c = 1))) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l2304_230479


namespace NUMINAMATH_CALUDE_factorization_equality_l2304_230476

theorem factorization_equality (m n : ℝ) : -8*m^2 + 2*m*n = -2*m*(4*m - n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2304_230476


namespace NUMINAMATH_CALUDE_solution_count_implies_n_l2304_230470

/-- The number of solutions to the equation 3x + 2y + 4z = n in positive integers x, y, and z -/
def num_solutions (n : ℕ+) : ℕ :=
  (Finset.filter (fun (x, y, z) => 3 * x + 2 * y + 4 * z = n) (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card

/-- Theorem stating that if the equation 3x + 2y + 4z = n has exactly 30 solutions in positive integers,
    then n must be either 22 or 23 -/
theorem solution_count_implies_n (n : ℕ+) :
  num_solutions n = 30 → n = 22 ∨ n = 23 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_implies_n_l2304_230470


namespace NUMINAMATH_CALUDE_functions_properties_l2304_230478

/-- Given functions f and g with parameter a, prove monotonicity of g and range of b -/
theorem functions_properties (a : ℝ) (h : a < -1) :
  let f := fun (x : ℝ) ↦ x^3 / 3 - x^2 / 2 + a^2 / 2 - 1 / 3
  let g := fun (x : ℝ) ↦ a * Real.log (x + 1) - x^2 / 2 - a * x
  let g_deriv := fun (x : ℝ) ↦ a / (x + 1) - x - a
  let monotonic_intervals := (Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (-a - 1), Set.Ioo 0 (-a - 1))
  let b := g (-a - 1) - f 1
  (∀ x ∈ monotonic_intervals.1, g_deriv x < 0) ∧
  (∀ x ∈ monotonic_intervals.2, g_deriv x > 0) ∧
  (∀ y : ℝ, y < 0 → ∃ x : ℝ, b = y) := by
  sorry


end NUMINAMATH_CALUDE_functions_properties_l2304_230478


namespace NUMINAMATH_CALUDE_new_tires_cost_l2304_230437

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def total_spent : ℝ := 387.85

theorem new_tires_cost (new_tires_cost : ℝ) : 
  new_tires_cost = total_spent - (speakers_cost + cd_player_cost) :=
by sorry

end NUMINAMATH_CALUDE_new_tires_cost_l2304_230437


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l2304_230468

/-- A line ax + by = c passes through the first, third, and fourth quadrants
    given that ab < 0 and bc < 0 -/
theorem line_passes_through_quadrants
  (a b c : ℝ) 
  (hab : a * b < 0) 
  (hbc : b * c < 0) : 
  ∃ (x y : ℝ), 
    (a * x + b * y = c) ∧ 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l2304_230468


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2304_230483

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2304_230483


namespace NUMINAMATH_CALUDE_bennys_total_work_hours_l2304_230496

/-- Calculates the total hours worked given the hours per day and number of days -/
def total_hours (hours_per_day : ℕ) (days : ℕ) : ℕ :=
  hours_per_day * days

/-- Theorem: Benny's total work hours -/
theorem bennys_total_work_hours :
  let hours_per_day : ℕ := 3
  let days : ℕ := 6
  total_hours hours_per_day days = 18 := by
  sorry

end NUMINAMATH_CALUDE_bennys_total_work_hours_l2304_230496


namespace NUMINAMATH_CALUDE_clock_hand_positions_l2304_230424

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hours : ℕ)
  (minutes : ℕ)

/-- The number of times the hour and minute hands coincide in 24 hours -/
def coincidences : ℕ := 22

/-- The number of times the hour and minute hands form a straight angle in 24 hours -/
def straight_angles : ℕ := 24

/-- The number of times the hour and minute hands form a right angle in 24 hours -/
def right_angles : ℕ := 48

/-- The number of full rotations the minute hand makes in 24 hours -/
def minute_rotations : ℕ := 24

/-- The number of full rotations the hour hand makes in 24 hours -/
def hour_rotations : ℕ := 2

theorem clock_hand_positions (c : Clock) :
  coincidences = 22 ∧
  straight_angles = 24 ∧
  right_angles = 48 :=
sorry

end NUMINAMATH_CALUDE_clock_hand_positions_l2304_230424


namespace NUMINAMATH_CALUDE_squirrel_nuts_theorem_l2304_230443

/-- The number of consecutive days the squirrel eats nuts -/
def total_days : Nat := 19

/-- The number of nuts eaten on a diet day -/
def diet_day_nuts : Nat := 1

/-- The number of additional nuts eaten on a normal day compared to a diet day -/
def normal_day_additional_nuts : Nat := 2

/-- The number of nuts eaten on a normal day -/
def normal_day_nuts : Nat := diet_day_nuts + normal_day_additional_nuts

/-- The number of diet days when starting with a diet day -/
def diet_days_start_diet : Nat := (total_days + 1) / 2

/-- The number of normal days when starting with a diet day -/
def normal_days_start_diet : Nat := total_days - diet_days_start_diet

/-- The number of diet days when starting with a normal day -/
def diet_days_start_normal : Nat := total_days - diet_days_start_diet

/-- The number of normal days when starting with a normal day -/
def normal_days_start_normal : Nat := diet_days_start_diet

/-- The total number of nuts eaten when starting with a diet day -/
def total_nuts_start_diet : Nat :=
  diet_days_start_diet * diet_day_nuts + normal_days_start_diet * normal_day_nuts

/-- The total number of nuts eaten when starting with a normal day -/
def total_nuts_start_normal : Nat :=
  diet_days_start_normal * diet_day_nuts + normal_days_start_normal * normal_day_nuts

theorem squirrel_nuts_theorem :
  (min total_nuts_start_diet total_nuts_start_normal = 37) ∧
  (max total_nuts_start_diet total_nuts_start_normal = 39) := by
  sorry

end NUMINAMATH_CALUDE_squirrel_nuts_theorem_l2304_230443


namespace NUMINAMATH_CALUDE_simplify_expression_l2304_230449

theorem simplify_expression : (27 * (10 ^ 12)) / (9 * (10 ^ 4)) = 300000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2304_230449


namespace NUMINAMATH_CALUDE_frank_average_reading_rate_l2304_230494

/-- Calculates the average number of pages read per day given the number of pages and days for each book --/
def average_pages_per_day (book1_pages book1_days book2_pages book2_days book3_pages book3_days : ℕ) : ℚ :=
  (book1_pages + book2_pages + book3_pages : ℚ) / (book1_days + book2_days + book3_days)

/-- Theorem stating that given Frank's reading data, his average pages per day is approximately 79.14 --/
theorem frank_average_reading_rate : 
  let avg := average_pages_per_day 249 3 379 5 480 6
  ∃ ε > 0, abs (avg - 79.14) < ε :=
by sorry

end NUMINAMATH_CALUDE_frank_average_reading_rate_l2304_230494


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2304_230487

/-- The minimum distance between two points on given curves with the same y-coordinate -/
theorem min_distance_between_curves : ∃ (d : ℝ), d = (5 + Real.log 2) / 4 ∧
  ∀ (x₁ x₂ y : ℝ), 
    y = Real.exp (2 * x₁ + 1) → 
    y = Real.sqrt (2 * x₂ - 1) → 
    d ≤ |x₂ - x₁| := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2304_230487


namespace NUMINAMATH_CALUDE_white_squares_95th_figure_l2304_230486

/-- The number of white squares in the nth figure of the sequence -/
def white_squares (n : ℕ) : ℕ := 8 + 5 * (n - 1)

/-- Theorem: The 95th figure in the sequence has 478 white squares -/
theorem white_squares_95th_figure : white_squares 95 = 478 := by
  sorry

end NUMINAMATH_CALUDE_white_squares_95th_figure_l2304_230486


namespace NUMINAMATH_CALUDE_two_layer_coverage_is_zero_l2304_230435

/-- Represents the area covered by rugs with different layers of overlap -/
structure RugCoverage where
  total_rug_area : ℝ
  total_floor_coverage : ℝ
  multilayer_coverage : ℝ
  three_layer_coverage : ℝ

/-- Calculates the area covered by exactly two layers of rug -/
def two_layer_coverage (rc : RugCoverage) : ℝ :=
  rc.multilayer_coverage - rc.three_layer_coverage

/-- Theorem stating that under the given conditions, the area covered by exactly two layers of rug is 0 -/
theorem two_layer_coverage_is_zero (rc : RugCoverage)
  (h1 : rc.total_rug_area = 212)
  (h2 : rc.total_floor_coverage = 140)
  (h3 : rc.multilayer_coverage = 24)
  (h4 : rc.three_layer_coverage = 24) :
  two_layer_coverage rc = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_layer_coverage_is_zero_l2304_230435


namespace NUMINAMATH_CALUDE_not_always_zero_l2304_230412

-- Define the heart operation
def heart (x y : ℝ) : ℝ := |x + y|

-- Theorem stating that the statement is false
theorem not_always_zero : ¬ ∀ x : ℝ, heart x x = 0 := by
  sorry

end NUMINAMATH_CALUDE_not_always_zero_l2304_230412


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2304_230471

theorem necessary_and_sufficient_condition : 
  (∀ x : ℝ, x = 1 ↔ x^2 - 2*x + 1 = 0) := by sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2304_230471


namespace NUMINAMATH_CALUDE_drinks_preparation_l2304_230438

/-- Given a number of pitchers and the capacity of each pitcher in glasses,
    calculate the total number of glasses that can be filled. -/
def total_glasses (num_pitchers : ℕ) (glasses_per_pitcher : ℕ) : ℕ :=
  num_pitchers * glasses_per_pitcher

/-- Theorem stating that 9 pitchers, each filling 6 glasses, results in 54 glasses total. -/
theorem drinks_preparation :
  total_glasses 9 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_drinks_preparation_l2304_230438


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l2304_230484

noncomputable def f (x : ℝ) := (x + 1) * Real.exp x

theorem f_monotone_decreasing :
  ∀ x y, x < y → x < -2 → y < -2 → f y < f x := by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l2304_230484


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2304_230407

theorem trig_identity_proof : 
  Real.sin (15 * π / 180) * Real.cos (45 * π / 180) + 
  Real.sin (75 * π / 180) * Real.sin (135 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2304_230407


namespace NUMINAMATH_CALUDE_gumdrop_replacement_l2304_230497

theorem gumdrop_replacement (blue_percent : Real) (brown_percent : Real) 
  (red_percent : Real) (yellow_percent : Real) (green_count : Nat) :
  blue_percent = 0.3 →
  brown_percent = 0.2 →
  red_percent = 0.15 →
  yellow_percent = 0.1 →
  green_count = 30 →
  let total := green_count / (1 - (blue_percent + brown_percent + red_percent + yellow_percent))
  let blue_count := blue_percent * total
  let brown_count := brown_percent * total
  let new_brown_count := brown_count + blue_count / 2
  new_brown_count = 42 := by
  sorry

end NUMINAMATH_CALUDE_gumdrop_replacement_l2304_230497


namespace NUMINAMATH_CALUDE_santa_candy_remainders_l2304_230474

theorem santa_candy_remainders (N : ℕ) (x : ℕ) (h : N = 35 * x + 7) :
  N % 15 ∈ ({2, 7, 12} : Finset ℕ) := by
  sorry

end NUMINAMATH_CALUDE_santa_candy_remainders_l2304_230474


namespace NUMINAMATH_CALUDE_class_closure_theorem_l2304_230491

theorem class_closure_theorem (n : ℕ) (M : Matrix (Fin n) (Fin n) Bool) 
  (h_distinct_rows : ∀ i j, i ≠ j → ∃ k, M i k ≠ M j k) :
  ∃ j : Fin n, ∀ i₁ i₂, i₁ ≠ i₂ → 
    ∃ k : Fin n, k ≠ j ∧ M i₁ k ≠ M i₂ k :=
sorry

end NUMINAMATH_CALUDE_class_closure_theorem_l2304_230491


namespace NUMINAMATH_CALUDE_reflection_sum_l2304_230451

/-- Reflects a point over the y-axis -/
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

/-- Reflects a point over the x-axis -/
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)

/-- Sums the coordinates of a point -/
def sum_coordinates (p : ℝ × ℝ) : ℝ := p.1 + p.2

theorem reflection_sum (y : ℝ) :
  let C : ℝ × ℝ := (3, y)
  let D := reflect_y C.1 C.2
  let E := reflect_x D.1 D.2
  sum_coordinates C + sum_coordinates E = -6 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_l2304_230451


namespace NUMINAMATH_CALUDE_fraction_1800_1809_is_7_30_l2304_230456

/-- The number of states that joined the union from 1800 to 1809 -/
def states_1800_1809 : ℕ := 7

/-- The total number of states considered (first 30 states) -/
def total_states : ℕ := 30

/-- The fraction of states that joined from 1800 to 1809 out of the first 30 states -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_states

theorem fraction_1800_1809_is_7_30 : fraction_1800_1809 = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_1800_1809_is_7_30_l2304_230456


namespace NUMINAMATH_CALUDE_pollution_filtering_l2304_230419

/-- Given a pollution filtering process where P = P₀e^(-kt),
    if 10% of pollutants are eliminated in 5 hours,
    then 81% of pollutants remain after 10 hours. -/
theorem pollution_filtering (P₀ k : ℝ) (h : P₀ > 0) :
  P₀ * Real.exp (-5 * k) = P₀ * 0.9 →
  P₀ * Real.exp (-10 * k) = P₀ * 0.81 := by
sorry

end NUMINAMATH_CALUDE_pollution_filtering_l2304_230419


namespace NUMINAMATH_CALUDE_alternative_plan_cost_is_eleven_l2304_230405

/-- The cost of Darnell's current unlimited plan -/
def current_plan_cost : ℕ := 12

/-- The difference in cost between the current plan and the alternative plan -/
def cost_difference : ℕ := 1

/-- The number of texts Darnell sends per month -/
def texts_per_month : ℕ := 60

/-- The number of minutes Darnell spends on calls per month -/
def call_minutes_per_month : ℕ := 60

/-- The cost of the alternative plan -/
def alternative_plan_cost : ℕ := current_plan_cost - cost_difference

theorem alternative_plan_cost_is_eleven :
  alternative_plan_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_alternative_plan_cost_is_eleven_l2304_230405


namespace NUMINAMATH_CALUDE_perfume_production_l2304_230499

/-- The number of rose petals required to make an ounce of perfume -/
def petals_per_ounce (petals_per_rose : ℕ) (roses_per_bush : ℕ) (bushes_harvested : ℕ) (bottles : ℕ) (ounces_per_bottle : ℕ) : ℕ :=
  (petals_per_rose * roses_per_bush * bushes_harvested) / (bottles * ounces_per_bottle)

/-- Theorem stating the number of rose petals required to make an ounce of perfume under given conditions -/
theorem perfume_production (petals_per_rose roses_per_bush bushes_harvested bottles ounces_per_bottle : ℕ) 
  (h1 : petals_per_rose = 8)
  (h2 : roses_per_bush = 12)
  (h3 : bushes_harvested = 800)
  (h4 : bottles = 20)
  (h5 : ounces_per_bottle = 12) :
  petals_per_ounce petals_per_rose roses_per_bush bushes_harvested bottles ounces_per_bottle = 320 := by
  sorry

end NUMINAMATH_CALUDE_perfume_production_l2304_230499


namespace NUMINAMATH_CALUDE_percentage_difference_l2304_230495

theorem percentage_difference (x y : ℝ) (h : x = 8 * y) :
  (x - y) / x * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2304_230495


namespace NUMINAMATH_CALUDE_expression_value_polynomial_simplification_l2304_230409

-- Part 1
theorem expression_value : (1/2)^(-2) - 0.01^(-1) + (-1 - 1/7)^0 = -95 := by sorry

-- Part 2
theorem polynomial_simplification (x : ℝ) : (x-2)*(x+1) - (x-1)^2 = x - 3 := by sorry

end NUMINAMATH_CALUDE_expression_value_polynomial_simplification_l2304_230409


namespace NUMINAMATH_CALUDE_average_of_remaining_digits_l2304_230455

theorem average_of_remaining_digits
  (total_count : Nat)
  (total_avg : ℚ)
  (subset_count : Nat)
  (subset_avg : ℚ)
  (h_total_count : total_count = 10)
  (h_total_avg : total_avg = 80)
  (h_subset_count : subset_count = 6)
  (h_subset_avg : subset_avg = 58)
  : (total_count * total_avg - subset_count * subset_avg) / (total_count - subset_count) = 113 := by
  sorry

end NUMINAMATH_CALUDE_average_of_remaining_digits_l2304_230455


namespace NUMINAMATH_CALUDE_cos_negative_twentythree_fourths_pi_l2304_230469

theorem cos_negative_twentythree_fourths_pi :
  Real.cos (-23 / 4 * Real.pi) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_twentythree_fourths_pi_l2304_230469


namespace NUMINAMATH_CALUDE_final_score_for_five_hours_l2304_230444

/-- Represents a student's test performance -/
structure TestPerformance where
  maxPoints : ℝ
  preparationTime : ℝ
  score : ℝ
  effortBonus : ℝ

/-- Calculates the final score given a TestPerformance -/
def finalScore (tp : TestPerformance) : ℝ :=
  tp.score * (1 + tp.effortBonus)

/-- Theorem stating the final score for 5 hours of preparation -/
theorem final_score_for_five_hours 
  (tp : TestPerformance)
  (h1 : tp.maxPoints = 150)
  (h2 : tp.preparationTime = 5)
  (h3 : tp.effortBonus = 0.1)
  (h4 : ∃ (t : TestPerformance), t.preparationTime = 2 ∧ t.score = 90 ∧ 
        tp.score / tp.preparationTime = t.score / t.preparationTime) :
  finalScore tp = 247.5 := by
sorry


end NUMINAMATH_CALUDE_final_score_for_five_hours_l2304_230444


namespace NUMINAMATH_CALUDE_interior_triangle_area_l2304_230400

theorem interior_triangle_area (a b c : ℝ) (ha : a = 64) (hb : b = 225) (hc : c = 289)
  (h_right_triangle : a + b = c) : (1/2) * Real.sqrt a * Real.sqrt b = 60 := by
  sorry

end NUMINAMATH_CALUDE_interior_triangle_area_l2304_230400


namespace NUMINAMATH_CALUDE_x_intercepts_count_l2304_230498

theorem x_intercepts_count : Nat.card { k : ℤ | 100 < k * Real.pi ∧ k * Real.pi < 1000 } = 286 := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l2304_230498


namespace NUMINAMATH_CALUDE_clothes_cost_l2304_230458

def total_spent : ℕ := 8000
def adidas_cost : ℕ := 600

theorem clothes_cost (nike_cost : ℕ) (skechers_cost : ℕ) :
  nike_cost = 3 * adidas_cost →
  skechers_cost = 5 * adidas_cost →
  total_spent - (adidas_cost + nike_cost + skechers_cost) = 2600 :=
by sorry

end NUMINAMATH_CALUDE_clothes_cost_l2304_230458


namespace NUMINAMATH_CALUDE_unique_m_for_inequality_l2304_230420

/-- The approximate value of log_10(2) -/
def log10_2 : ℝ := 0.3010

/-- The theorem stating that 155 is the unique positive integer m satisfying the inequality -/
theorem unique_m_for_inequality : ∃! (m : ℕ), m > 0 ∧ (10 : ℝ)^(m - 1) < 2^512 ∧ 2^512 < 10^m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_m_for_inequality_l2304_230420


namespace NUMINAMATH_CALUDE_probability_divisible_by_20_l2304_230462

/-- The set of digits used to form the six-digit number -/
def digits : Finset Nat := {1, 2, 3, 4, 5, 8}

/-- The total number of possible six-digit arrangements -/
def total_arrangements : Nat := 720

/-- Predicate to check if a number is divisible by 20 -/
def is_divisible_by_20 (n : Nat) : Prop := n % 20 = 0

/-- The number of arrangements divisible by 20 -/
def arrangements_divisible_by_20 : Nat := 576

theorem probability_divisible_by_20 :
  (arrangements_divisible_by_20 : ℚ) / total_arrangements = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisible_by_20_l2304_230462


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l2304_230488

/-- The inradius of a right triangle with side lengths 5, 12, and 13 is 2. -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l2304_230488


namespace NUMINAMATH_CALUDE_pilot_miles_theorem_l2304_230436

theorem pilot_miles_theorem (tuesday_miles : ℕ) (total_miles : ℕ) :
  tuesday_miles = 1134 →
  total_miles = 7827 →
  ∃ (thursday_miles : ℕ),
    3 * (tuesday_miles + thursday_miles) = total_miles ∧
    thursday_miles = 1475 :=
by
  sorry

end NUMINAMATH_CALUDE_pilot_miles_theorem_l2304_230436


namespace NUMINAMATH_CALUDE_smallest_interesting_rectangle_area_l2304_230410

/-- A rectangle is interesting if it has integer side lengths and contains
    exactly four lattice points strictly in its interior. -/
def is_interesting (a b : ℕ) : Prop :=
  (a - 1) * (b - 1) = 4

/-- The area of the smallest interesting rectangle is 10. -/
theorem smallest_interesting_rectangle_area : 
  (∃ a b : ℕ, is_interesting a b ∧ a * b = 10) ∧ 
  (∀ a b : ℕ, is_interesting a b → a * b ≥ 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_interesting_rectangle_area_l2304_230410


namespace NUMINAMATH_CALUDE_star_three_neg_two_l2304_230431

/-- Definition of the ☆ operation for rational numbers -/
def star (a b : ℚ) : ℚ := b^3 - abs (b - a)

/-- Theorem stating that 3☆(-2) = -13 -/
theorem star_three_neg_two : star 3 (-2) = -13 := by sorry

end NUMINAMATH_CALUDE_star_three_neg_two_l2304_230431


namespace NUMINAMATH_CALUDE_existence_of_larger_element_l2304_230452

/-- A doubly infinite array of positive integers -/
def InfiniteArray := ℕ+ → ℕ+ → ℕ+

/-- The property that each positive integer appears exactly eight times in the array -/
def EightOccurrences (a : InfiniteArray) : Prop :=
  ∀ n : ℕ+, (∃ (s : Finset (ℕ+ × ℕ+)), s.card = 8 ∧ (∀ (p : ℕ+ × ℕ+), p ∈ s ↔ a p.1 p.2 = n))

/-- The main theorem -/
theorem existence_of_larger_element (a : InfiniteArray) (h : EightOccurrences a) :
  ∃ (m n : ℕ+), a m n > m * n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_larger_element_l2304_230452


namespace NUMINAMATH_CALUDE_reading_time_for_18_pages_l2304_230475

-- Define the reading rate (pages per minute)
def reading_rate : ℚ := 4 / 2

-- Define the number of pages to read
def pages_to_read : ℕ := 18

-- Theorem: It takes 9 minutes to read 18 pages at the given rate
theorem reading_time_for_18_pages :
  (pages_to_read : ℚ) / reading_rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_for_18_pages_l2304_230475


namespace NUMINAMATH_CALUDE_cube_surface_area_l2304_230414

/-- The surface area of a cube with edge length 2a cm is 24a² cm² -/
theorem cube_surface_area (a : ℝ) (h : a > 0) : 
  6 * (2 * a) ^ 2 = 24 * a ^ 2 := by
  sorry

#check cube_surface_area

end NUMINAMATH_CALUDE_cube_surface_area_l2304_230414


namespace NUMINAMATH_CALUDE_ab_leq_one_l2304_230454

theorem ab_leq_one (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b = 2) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_leq_one_l2304_230454


namespace NUMINAMATH_CALUDE_liquid_ratio_after_replacement_l2304_230459

def container_capacity : ℝ := 37.5

def liquid_replaced : ℝ := 15

def final_ratio_A : ℝ := 9

def final_ratio_B : ℝ := 16

theorem liquid_ratio_after_replacement :
  let initial_A := container_capacity
  let first_step_A := initial_A - liquid_replaced
  let first_step_B := liquid_replaced
  let second_step_A := first_step_A * (1 - liquid_replaced / container_capacity)
  let second_step_B := container_capacity - second_step_A
  (second_step_A / final_ratio_A = second_step_B / final_ratio_B) ∧
  (second_step_A + second_step_B = container_capacity) := by
  sorry

end NUMINAMATH_CALUDE_liquid_ratio_after_replacement_l2304_230459


namespace NUMINAMATH_CALUDE_certain_number_problem_l2304_230446

theorem certain_number_problem : 
  (∃ n : ℕ, (∀ m > n, ¬∃ p q : ℕ+, 
    p > m ∧ 
    q > m ∧ 
    17 * (p + 1) = 28 * (q + 1) ∧ 
    p + q = 43) ∧
  (∃ p q : ℕ+, 
    p > n ∧ 
    q > n ∧ 
    17 * (p + 1) = 28 * (q + 1) ∧ 
    p + q = 43)) ∧
  (∀ n' > n, ¬∃ p q : ℕ+, 
    p > n' ∧ 
    q > n' ∧ 
    17 * (p + 1) = 28 * (q + 1) ∧ 
    p + q = 43) →
  n = 15 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2304_230446


namespace NUMINAMATH_CALUDE_janes_daily_vase_arrangement_l2304_230450

def total_vases : ℕ := 248
def last_day_vases : ℕ := 8

theorem janes_daily_vase_arrangement :
  ∃ (daily_vases : ℕ),
    daily_vases > 0 ∧
    daily_vases = last_day_vases ∧
    (total_vases - last_day_vases) % daily_vases = 0 :=
by sorry

end NUMINAMATH_CALUDE_janes_daily_vase_arrangement_l2304_230450


namespace NUMINAMATH_CALUDE_exists_composite_prime_product_plus_one_l2304_230492

/-- pₖ denotes the k-th prime number -/
def nth_prime (k : ℕ) : ℕ := sorry

/-- Product of first n prime numbers plus 1 -/
def prime_product_plus_one (n : ℕ) : ℕ := 
  (List.range n).foldl (λ acc i => acc * nth_prime (i + 1)) 1 + 1

theorem exists_composite_prime_product_plus_one :
  ∃ n : ℕ, ¬ Nat.Prime (prime_product_plus_one n) := by sorry

end NUMINAMATH_CALUDE_exists_composite_prime_product_plus_one_l2304_230492


namespace NUMINAMATH_CALUDE_rectangle_area_difference_main_theorem_l2304_230440

theorem rectangle_area_difference : ℕ → Prop :=
fun diff =>
  (∃ (l w : ℕ), l + w = 30 ∧ l * w = 225) ∧  -- Largest area
  (∃ (l w : ℕ), l + w = 30 ∧ l * w = 29) ∧  -- Smallest area
  diff = 225 - 29

theorem main_theorem : rectangle_area_difference 196 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_main_theorem_l2304_230440


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2304_230413

/-- Given a circle C with equation x^2 - 8y - 7 = -y^2 - 6x, 
    prove that the sum of its center coordinates and radius is 1 + 4√2 -/
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), 
    (∀ (x y : ℝ), x^2 - 8*y - 7 = -y^2 - 6*x → (x - a)^2 + (y - b)^2 = r^2) →
    a + b + r = 1 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2304_230413


namespace NUMINAMATH_CALUDE_fib_100_102_minus_101_squared_l2304_230467

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Determinant property of Fibonacci relationship -/
axiom fib_det_property (n : ℕ) : 
  fib (n + 1) * fib (n - 1) - fib n ^ 2 = (-1) ^ n

/-- Theorem: F₁₀₀ F₁₀₂ - F₁₀₁² = -1 -/
theorem fib_100_102_minus_101_squared : 
  fib 100 * fib 102 - fib 101 ^ 2 = -1 := by sorry

end NUMINAMATH_CALUDE_fib_100_102_minus_101_squared_l2304_230467


namespace NUMINAMATH_CALUDE_only_valid_quadruples_l2304_230428

/-- A quadruple of non-negative integers satisfying the given conditions -/
structure ValidQuadruple where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  eq : a * b = 2 * (1 + c * d)
  triangle : (a - c) + (b - d) > c + d ∧ 
             (a - c) + (c + d) > b - d ∧ 
             (b - d) + (c + d) > a - c

/-- The theorem stating that only two specific quadruples satisfy the conditions -/
theorem only_valid_quadruples : 
  ∀ q : ValidQuadruple, (q.a = 1 ∧ q.b = 2 ∧ q.c = 0 ∧ q.d = 1) ∨ 
                        (q.a = 2 ∧ q.b = 1 ∧ q.c = 1 ∧ q.d = 0) := by
  sorry

end NUMINAMATH_CALUDE_only_valid_quadruples_l2304_230428


namespace NUMINAMATH_CALUDE_juice_subtraction_l2304_230402

theorem juice_subtraction (initial_juice : ℚ) (given_away : ℚ) :
  initial_juice = 5 →
  given_away = 16 / 3 →
  initial_juice - given_away = -1 / 3 := by
sorry


end NUMINAMATH_CALUDE_juice_subtraction_l2304_230402


namespace NUMINAMATH_CALUDE_recurrence_sequence_general_term_l2304_230481

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  a 1 = 1 ∧
  ∀ n, (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0

/-- The theorem stating that the sequence satisfying the recurrence relation
    has the general term a_n = 1/(2^(n-1)) -/
theorem recurrence_sequence_general_term (a : ℕ → ℝ) 
    (h : RecurrenceSequence a) : 
    ∀ n, a n = 1 / (2 ^ (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_general_term_l2304_230481


namespace NUMINAMATH_CALUDE_probability_even_product_l2304_230421

/-- Spinner C with numbers 1 through 6 -/
def spinner_C : Finset ℕ := Finset.range 6

/-- Spinner D with numbers 1 through 4 -/
def spinner_D : Finset ℕ := Finset.range 4

/-- Function to check if a number is even -/
def is_even (n : ℕ) : Bool := n % 2 = 0

/-- Function to check if the product of two numbers is even -/
def product_is_even (x y : ℕ) : Bool := is_even (x * y)

/-- Total number of possible outcomes -/
def total_outcomes : ℕ := (Finset.card spinner_C) * (Finset.card spinner_D)

/-- Number of outcomes where the product is even -/
def even_product_outcomes : ℕ := Finset.card (Finset.filter (λ (pair : ℕ × ℕ) => product_is_even pair.1 pair.2) (spinner_C.product spinner_D))

/-- Theorem stating the probability of getting an even product -/
theorem probability_even_product :
  (even_product_outcomes : ℚ) / total_outcomes = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_probability_even_product_l2304_230421


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l2304_230480

theorem sum_of_roots_cubic (x : ℝ) : 
  (x + 2)^2 * (x - 3) = 40 → 
  ∃ (r₁ r₂ r₃ : ℝ), r₁ + r₂ + r₃ = -1 ∧ 
    ((x = r₁) ∨ (x = r₂) ∨ (x = r₃)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l2304_230480


namespace NUMINAMATH_CALUDE_stationery_cost_is_18300_l2304_230433

/-- Calculates the total amount paid for stationery given the number of pencil boxes,
    pencils per box, and the costs of pens and pencils. -/
def total_stationery_cost (pencil_boxes : ℕ) (pencils_per_box : ℕ) 
                          (pen_cost : ℕ) (pencil_cost : ℕ) : ℕ :=
  let total_pencils := pencil_boxes * pencils_per_box
  let total_pens := 2 * total_pencils + 300
  total_pens * pen_cost + total_pencils * pencil_cost

/-- Proves that the total amount paid for the stationery is $18300 -/
theorem stationery_cost_is_18300 :
  total_stationery_cost 15 80 5 4 = 18300 := by
  sorry

#eval total_stationery_cost 15 80 5 4

end NUMINAMATH_CALUDE_stationery_cost_is_18300_l2304_230433


namespace NUMINAMATH_CALUDE_three_person_subcommittees_l2304_230422

theorem three_person_subcommittees (n : ℕ) (k : ℕ) : n = 8 → k = 3 →
  Nat.choose n k = 56 := by sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_l2304_230422


namespace NUMINAMATH_CALUDE_expand_expression_l2304_230416

theorem expand_expression (x y : ℝ) : (2*x - 3*y + 1) * (2*x + 3*y - 1) = 4*x^2 - 9*y^2 + 6*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2304_230416


namespace NUMINAMATH_CALUDE_smallest_n_for_radio_profit_l2304_230466

theorem smallest_n_for_radio_profit (n d : ℕ) : 
  d > 0 → 
  (n : ℚ) * ((n : ℚ) - 11) = (d : ℚ) / 8 → 
  (∃ k : ℕ, d = 8 * k) →
  n ≥ 11 →
  (∀ m : ℕ, m < n → m ≥ 11 → (m : ℚ) * ((m : ℚ) - 11) ≠ (d : ℚ) / 8 ∨ ¬(∃ k : ℕ, d = 8 * k)) →
  n = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_radio_profit_l2304_230466


namespace NUMINAMATH_CALUDE_square_and_cube_roots_l2304_230415

-- Define square root
def is_square_root (x y : ℝ) : Prop := y^2 = x

-- Define cube root
def is_cube_root (x y : ℝ) : Prop := y^3 = x

-- Define self square root
def is_self_square_root (x : ℝ) : Prop := x^2 = x

theorem square_and_cube_roots :
  (∃ y : ℝ, y < 0 ∧ is_square_root 2 y) ∧
  (is_cube_root (-1) (-1)) ∧
  (is_square_root 100 10) ∧
  (∀ x : ℝ, is_self_square_root x ↔ (x = 0 ∨ x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_l2304_230415


namespace NUMINAMATH_CALUDE_solution_conditions_l2304_230439

/-- Represents the system of equations and conditions for the problem -/
structure EquationSystem where
  x : ℝ
  y : ℝ
  z : ℝ
  a : ℝ
  b : ℝ
  eq1 : 52 * x - 34 * y - z + 9 * a = 0
  eq2 : 49 * x - 31 * y - 3 * z - b = 0
  eq3 : 36 * x - 24 * y + z + 3 * a + 2 * b = 0
  a_pos : a > 0
  b_pos : b > 0

/-- The main theorem stating the conditions for positivity and minimal x -/
theorem solution_conditions (sys : EquationSystem) :
  (sys.a > (2/9) * sys.b → sys.x > 0 ∧ sys.y > 0 ∧ sys.z > 0) ∧
  (sys.x = 9 ∧ sys.y = 14 ∧ sys.z = 1 ∧ sys.a = 1 ∧ sys.b = 4 →
   ∀ (a' b' : ℝ), a' > 0 → b' > 0 → sys.x ≤ 17 * a' - 2 * b') :=
by sorry

end NUMINAMATH_CALUDE_solution_conditions_l2304_230439


namespace NUMINAMATH_CALUDE_response_rate_percentage_l2304_230406

theorem response_rate_percentage 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 240) 
  (h2 : questionnaires_mailed = 400) : 
  (responses_needed : ℝ) / questionnaires_mailed * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_response_rate_percentage_l2304_230406


namespace NUMINAMATH_CALUDE_larger_gate_width_l2304_230448

/-- Calculates the width of the larger gate for a rectangular garden. -/
theorem larger_gate_width
  (length : ℝ)
  (width : ℝ)
  (small_gate_width : ℝ)
  (total_fencing : ℝ)
  (h1 : length = 225)
  (h2 : width = 125)
  (h3 : small_gate_width = 3)
  (h4 : total_fencing = 687) :
  2 * (length + width) - (small_gate_width + total_fencing) = 10 :=
by sorry

end NUMINAMATH_CALUDE_larger_gate_width_l2304_230448


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_smallest_n_is_626_l2304_230477

theorem smallest_n_for_sqrt_difference (n : ℕ) : n ≥ 626 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.02 := by
  sorry

theorem smallest_n_is_626 : ∀ k : ℕ, k < 626 → Real.sqrt k - Real.sqrt (k - 1) ≥ 0.02 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_smallest_n_is_626_l2304_230477


namespace NUMINAMATH_CALUDE_square_sum_of_reciprocal_sum_and_sum_l2304_230403

theorem square_sum_of_reciprocal_sum_and_sum (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) (h2 : x + y = 5) : x^2 + y^2 = 35/2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_reciprocal_sum_and_sum_l2304_230403


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2304_230430

theorem arithmetic_calculations :
  (5 + (-6) + 3 - (-4) = 6) ∧
  (-1^2024 - (2 - (-2)^3) / (-2/5) * 5/2 = 123/2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2304_230430


namespace NUMINAMATH_CALUDE_peters_age_fraction_l2304_230434

/-- Proves that Peter's current age is 1/2 of his mother's age -/
theorem peters_age_fraction (peter_age harriet_age mother_age : ℕ) : 
  harriet_age = 13 →
  mother_age = 60 →
  peter_age + 4 = 2 * (harriet_age + 4) →
  peter_age = mother_age / 2 := by
sorry

end NUMINAMATH_CALUDE_peters_age_fraction_l2304_230434


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2304_230441

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 3 * x^2 - 5 * x + 20 = 0 ↔ x = a + b * I ∨ x = a - b * I) → 
  a + b^2 = 245/36 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2304_230441
