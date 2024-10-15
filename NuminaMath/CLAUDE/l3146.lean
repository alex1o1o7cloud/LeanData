import Mathlib

namespace NUMINAMATH_CALUDE_parabola_triangle_area_l3146_314632

-- Define the parabolas
def M1 (a c x : ℝ) : ℝ := a * x^2 + c
def M2 (a c x : ℝ) : ℝ := a * (x - 2)^2 + c - 5

-- Define the theorem
theorem parabola_triangle_area (a c : ℝ) :
  -- M2 passes through the vertex of M1
  (M2 a c 0 = M1 a c 0) →
  -- Point C on M2 has coordinates (2, c-5)
  (M2 a c 2 = c - 5) →
  -- The area of triangle ABC is 10
  ∃ (x_B y_B : ℝ), 
    x_B = 2 ∧ 
    y_B = M1 a c x_B ∧ 
    (1/2 * |x_B - 0| * |y_B - (c - 5)| = 10) :=
by sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l3146_314632


namespace NUMINAMATH_CALUDE_M_mod_45_l3146_314673

def M : ℕ := sorry

theorem M_mod_45 : M % 45 = 15 := by sorry

end NUMINAMATH_CALUDE_M_mod_45_l3146_314673


namespace NUMINAMATH_CALUDE_exists_all_cards_moved_no_guarantee_ace_not_next_to_empty_l3146_314676

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

end NUMINAMATH_CALUDE_exists_all_cards_moved_no_guarantee_ace_not_next_to_empty_l3146_314676


namespace NUMINAMATH_CALUDE_surface_area_of_S_l3146_314610

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents the solid S' formed by removing a tunnel from the cube -/
structure Solid where
  cube : Cube
  tunnelStart : Point3D
  tunnelEnd : Point3D

/-- Calculate the surface area of the solid S' -/
def surfaceAreaS' (s : Solid) : ℝ :=
  sorry

theorem surface_area_of_S' (c : Cube) (e i j k : Point3D) :
  c.sideLength = 12 ∧
  e.x = 12 ∧ e.y = 12 ∧ e.z = 12 ∧
  i.x = 9 ∧ i.y = 12 ∧ i.z = 12 ∧
  j.x = 12 ∧ j.y = 9 ∧ j.z = 12 ∧
  k.x = 12 ∧ k.y = 12 ∧ k.z = 9 →
  surfaceAreaS' { cube := c, tunnelStart := i, tunnelEnd := k } = 840 + 45 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_S_l3146_314610


namespace NUMINAMATH_CALUDE_quadratic_equation_one_l3146_314602

theorem quadratic_equation_one (x : ℝ) :
  (x + 2) * (x - 2) - 2 * (x - 3) = 3 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_l3146_314602


namespace NUMINAMATH_CALUDE_ant_movement_l3146_314665

theorem ant_movement (initial_position : Int) (move_right1 move_left move_right2 : Int) :
  initial_position = -3 →
  move_right1 = 5 →
  move_left = 9 →
  move_right2 = 1 →
  initial_position + move_right1 - move_left + move_right2 = -6 :=
by sorry

end NUMINAMATH_CALUDE_ant_movement_l3146_314665


namespace NUMINAMATH_CALUDE_right_triangle_area_l3146_314612

/-- Given a right triangle with perimeter 4 + √26 and median length 2 on the hypotenuse, its area is 5/2 -/
theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Positive side lengths
  a^2 + b^2 = c^2 →  -- Right triangle (Pythagorean theorem)
  a + b + c = 4 + Real.sqrt 26 →  -- Perimeter condition
  c / 2 = 2 →  -- Median length condition
  (1/2) * a * b = 5/2 := by  -- Area of the triangle
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3146_314612


namespace NUMINAMATH_CALUDE_paths_to_2005_l3146_314686

/-- Represents the number of choices for each step in forming the number 2005 -/
structure PathChoices where
  first_zero : Nat
  second_zero : Nat
  final_five : Nat

/-- Calculates the total number of paths to form 2005 -/
def total_paths (choices : PathChoices) : Nat :=
  choices.first_zero * choices.second_zero * choices.final_five

/-- The given choices for each step in forming 2005 -/
def given_choices : PathChoices :=
  { first_zero := 6
  , second_zero := 2
  , final_five := 3 }

/-- Theorem stating that there are 36 different paths to form 2005 -/
theorem paths_to_2005 : total_paths given_choices = 36 := by
  sorry

#eval total_paths given_choices

end NUMINAMATH_CALUDE_paths_to_2005_l3146_314686


namespace NUMINAMATH_CALUDE_school_attendance_problem_l3146_314660

theorem school_attendance_problem (girls : ℕ) (percentage_increase : ℚ) (boys : ℕ) :
  girls = 5000 →
  percentage_increase = 40 / 100 →
  (boys : ℚ) + percentage_increase * (boys : ℚ) = (boys : ℚ) + (girls : ℚ) →
  boys = 12500 := by
sorry

end NUMINAMATH_CALUDE_school_attendance_problem_l3146_314660


namespace NUMINAMATH_CALUDE_n_pointed_star_angle_sum_l3146_314637

/-- Represents an n-pointed star created from a convex n-gon. -/
structure NPointedStar where
  n : ℕ
  n_ge_7 : n ≥ 7

/-- The sum of interior angles at the n intersection points of an n-pointed star. -/
def interior_angle_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem stating that the sum of interior angles at the n intersection points
    of an n-pointed star is 180°(n-2). -/
theorem n_pointed_star_angle_sum (star : NPointedStar) :
  interior_angle_sum star = 180 * (star.n - 2) := by
  sorry

end NUMINAMATH_CALUDE_n_pointed_star_angle_sum_l3146_314637


namespace NUMINAMATH_CALUDE_three_person_subcommittees_l3146_314696

theorem three_person_subcommittees (n : ℕ) (k : ℕ) : n = 8 → k = 3 →
  Nat.choose n k = 56 := by sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_l3146_314696


namespace NUMINAMATH_CALUDE_parametric_eq_line_l3146_314657

/-- Prove that the parametric equations x = t - 1 and y = 2t - 1 represent the line y = 2x + 1 for all real values of t. -/
theorem parametric_eq_line (t : ℝ) : 
  let x := t - 1
  let y := 2*t - 1
  y = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_parametric_eq_line_l3146_314657


namespace NUMINAMATH_CALUDE_pollution_filtering_l3146_314677

/-- Given a pollution filtering process where P = P₀e^(-kt),
    if 10% of pollutants are eliminated in 5 hours,
    then 81% of pollutants remain after 10 hours. -/
theorem pollution_filtering (P₀ k : ℝ) (h : P₀ > 0) :
  P₀ * Real.exp (-5 * k) = P₀ * 0.9 →
  P₀ * Real.exp (-10 * k) = P₀ * 0.81 := by
sorry

end NUMINAMATH_CALUDE_pollution_filtering_l3146_314677


namespace NUMINAMATH_CALUDE_matrix_N_computation_l3146_314646

theorem matrix_N_computation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec (![3, -2]) = ![5, 1])
  (h2 : N.mulVec (![(-2), 4]) = ![0, -2]) :
  N.mulVec (![7, 0]) = ![17.5, 0] := by
sorry

end NUMINAMATH_CALUDE_matrix_N_computation_l3146_314646


namespace NUMINAMATH_CALUDE_magpie_porridge_l3146_314674

/-- The amount of porridge given to each chick -/
def PorridgeDistribution (p₁ p₂ p₃ p₄ p₅ p₆ : ℝ) : Prop :=
  p₃ = p₁ + p₂ ∧
  p₄ = p₂ + p₃ ∧
  p₅ = p₃ + p₄ ∧
  p₆ = p₄ + p₅ ∧
  p₅ = 10

theorem magpie_porridge : 
  ∀ p₁ p₂ p₃ p₄ p₅ p₆ : ℝ, 
  PorridgeDistribution p₁ p₂ p₃ p₄ p₅ p₆ → 
  p₁ + p₂ + p₃ + p₄ + p₅ + p₆ = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_magpie_porridge_l3146_314674


namespace NUMINAMATH_CALUDE_library_visitors_l3146_314698

theorem library_visitors (sunday_visitors : ℕ) (total_days : ℕ) (sundays : ℕ) (avg_visitors : ℕ) :
  sunday_visitors = 510 →
  total_days = 30 →
  sundays = 5 →
  avg_visitors = 285 →
  (sunday_visitors * sundays + (total_days - sundays) * 
    ((total_days * avg_visitors - sunday_visitors * sundays) / (total_days - sundays))) / total_days = avg_visitors :=
by sorry

end NUMINAMATH_CALUDE_library_visitors_l3146_314698


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3146_314691

/-- 
Theorem: The number of terms in an arithmetic sequence 
starting with 2, ending with 2014, and having a common difference of 4 
is equal to 504.
-/
theorem arithmetic_sequence_length : 
  let a₁ : ℕ := 2  -- First term
  let d : ℕ := 4   -- Common difference
  let aₙ : ℕ := 2014  -- Last term
  ∃ n : ℕ, n > 0 ∧ aₙ = a₁ + (n - 1) * d ∧ n = 504
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3146_314691


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l3146_314613

theorem students_taking_one_subject (total_geometry : ℕ) (both_subjects : ℕ) (science_only : ℕ)
  (h1 : both_subjects = 15)
  (h2 : total_geometry = 30)
  (h3 : science_only = 18) :
  total_geometry - both_subjects + science_only = 33 := by
sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l3146_314613


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3146_314651

/-- The equation of the tangent line to the curve y = x³ + 2x at the point (1, 3) is 5x - y - 2 = 0 -/
theorem tangent_line_equation (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x = x^3 + 2*x →
  f x₀ = y₀ →
  x₀ = 1 →
  y₀ = 3 →
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 5*x - y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3146_314651


namespace NUMINAMATH_CALUDE_no_common_solution_l3146_314644

theorem no_common_solution : ¬∃ (x y : ℝ), x^2 + y^2 = 16 ∧ x^2 + 3*y + 30 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l3146_314644


namespace NUMINAMATH_CALUDE_smallest_solution_equation_smallest_solution_l3146_314681

theorem smallest_solution_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ↔ (x = 4 - Real.sqrt 2 ∨ x = 4 + Real.sqrt 2) :=
sorry

theorem smallest_solution (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ (∀ y, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x) →
  x = 4 - Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_equation_smallest_solution_l3146_314681


namespace NUMINAMATH_CALUDE_no_intersection_points_l3146_314683

theorem no_intersection_points : 
  ¬∃ (z : ℂ), z^4 + z = 1 ∧ Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_no_intersection_points_l3146_314683


namespace NUMINAMATH_CALUDE_expression_bounds_l3146_314656

theorem expression_bounds (a b c d e : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  let expr := Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
              Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + 
              Real.sqrt (e^2 + (1-a)^2)
  5 / Real.sqrt 2 ≤ expr ∧ expr ≤ 5 ∧ 
  ∃ (a' b' c' d' e' : Real), (0 ≤ a' ∧ a' ≤ 1) ∧ (0 ≤ b' ∧ b' ≤ 1) ∧ 
    (0 ≤ c' ∧ c' ≤ 1) ∧ (0 ≤ d' ∧ d' ≤ 1) ∧ (0 ≤ e' ∧ e' ≤ 1) ∧
    let expr' := Real.sqrt (a'^2 + (1-b')^2) + Real.sqrt (b'^2 + (1-c')^2) + 
                 Real.sqrt (c'^2 + (1-d')^2) + Real.sqrt (d'^2 + (1-e')^2) + 
                 Real.sqrt (e'^2 + (1-a')^2)
    expr' = 5 / Real.sqrt 2 ∧
  ∃ (a'' b'' c'' d'' e'' : Real), (0 ≤ a'' ∧ a'' ≤ 1) ∧ (0 ≤ b'' ∧ b'' ≤ 1) ∧ 
    (0 ≤ c'' ∧ c'' ≤ 1) ∧ (0 ≤ d'' ∧ d'' ≤ 1) ∧ (0 ≤ e'' ∧ e'' ≤ 1) ∧
    let expr'' := Real.sqrt (a''^2 + (1-b'')^2) + Real.sqrt (b''^2 + (1-c'')^2) + 
                  Real.sqrt (c''^2 + (1-d'')^2) + Real.sqrt (d''^2 + (1-e'')^2) + 
                  Real.sqrt (e''^2 + (1-a'')^2)
    expr'' = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l3146_314656


namespace NUMINAMATH_CALUDE_expand_expression_l3146_314635

theorem expand_expression (x y : ℝ) :
  5 * (3 * x^3 - 4 * x * y + x^2 - y^2) = 15 * x^3 - 20 * x * y + 5 * x^2 - 5 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3146_314635


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l3146_314682

theorem complex_equation_solutions (x y : ℝ) :
  x^2 - y^2 + (2*x*y : ℂ)*I = (2 : ℂ)*I →
  ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l3146_314682


namespace NUMINAMATH_CALUDE_gcd_of_319_377_116_l3146_314658

theorem gcd_of_319_377_116 : Nat.gcd 319 (Nat.gcd 377 116) = 29 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_319_377_116_l3146_314658


namespace NUMINAMATH_CALUDE_chocolate_bars_distribution_l3146_314633

theorem chocolate_bars_distribution (large_box_total : ℕ) (small_boxes : ℕ) (bars_per_small_box : ℕ) :
  large_box_total = 375 →
  small_boxes = 15 →
  large_box_total = small_boxes * bars_per_small_box →
  bars_per_small_box = 25 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_distribution_l3146_314633


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3146_314611

theorem product_of_sums_equals_difference_of_powers : 
  (5 + 2) * (5^3 + 2^3) * (5^9 + 2^9) * (5^27 + 2^27) * (5^81 + 2^81) = 5^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3146_314611


namespace NUMINAMATH_CALUDE_B_equals_set_l3146_314647

def A : Set ℤ := {-1, 2, 3, 4}

def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2*x + 2}

theorem B_equals_set : B = {2, 5, 10} := by sorry

end NUMINAMATH_CALUDE_B_equals_set_l3146_314647


namespace NUMINAMATH_CALUDE_border_area_l3146_314670

def photo_height : ℝ := 9
def photo_width : ℝ := 12
def frame_border : ℝ := 3

theorem border_area : 
  let framed_height := photo_height + 2 * frame_border
  let framed_width := photo_width + 2 * frame_border
  let photo_area := photo_height * photo_width
  let framed_area := framed_height * framed_width
  framed_area - photo_area = 162 := by sorry

end NUMINAMATH_CALUDE_border_area_l3146_314670


namespace NUMINAMATH_CALUDE_john_good_games_l3146_314655

/-- 
Given:
- John bought 21 games from a friend
- John bought 8 games at a garage sale
- 23 of the games didn't work

Prove that John ended up with 6 good games.
-/
theorem john_good_games : 
  let games_from_friend : ℕ := 21
  let games_from_garage_sale : ℕ := 8
  let non_working_games : ℕ := 23
  let total_games := games_from_friend + games_from_garage_sale
  let good_games := total_games - non_working_games
  good_games = 6 := by
  sorry

end NUMINAMATH_CALUDE_john_good_games_l3146_314655


namespace NUMINAMATH_CALUDE_square_2209_identity_l3146_314689

theorem square_2209_identity (x : ℤ) (h : x^2 = 2209) : (2*x + 1) * (2*x - 1) = 8835 := by
  sorry

end NUMINAMATH_CALUDE_square_2209_identity_l3146_314689


namespace NUMINAMATH_CALUDE_horner_V₁_value_l3146_314608

/-- Horner's method for polynomial evaluation -/
def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = 3x⁴ + 2x² + x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

/-- V₁ in Horner's method for f(x) at x = 10 -/
def V₁ : ℝ := horner_step 2 10 3

theorem horner_V₁_value : V₁ = 32 := by sorry

end NUMINAMATH_CALUDE_horner_V₁_value_l3146_314608


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3146_314606

theorem sum_of_coefficients (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 15625 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3146_314606


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3146_314615

/-- Given an arithmetic sequence with first term -3 and second term 5,
    the positive difference between the 1010th and 1000th terms is 80. -/
theorem arithmetic_sequence_difference : ∀ (a : ℕ → ℤ),
  (a 1 = -3) →
  (a 2 = 5) →
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
  |a 1010 - a 1000| = 80 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3146_314615


namespace NUMINAMATH_CALUDE_symmetric_points_range_l3146_314699

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then 2 * x^2 - 3 * x else a / Real.exp x

theorem symmetric_points_range (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ f a x = f a (-y)) →
  -Real.exp (-1/2) ≤ a ∧ a ≤ 9 * Real.exp (-3) :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_range_l3146_314699


namespace NUMINAMATH_CALUDE_sequence_sum_equals_29_l3146_314622

def sequence_term (n : ℕ) : ℤ :=
  if n % 2 = 0 then 2 + 3 * (n - 1) else -(5 + 3 * (n - 2))

def sequence_length : ℕ := 19

theorem sequence_sum_equals_29 :
  (Finset.range sequence_length).sum (λ i => sequence_term i) = 29 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_29_l3146_314622


namespace NUMINAMATH_CALUDE_nine_fifteen_div_fifty_four_five_l3146_314680

theorem nine_fifteen_div_fifty_four_five :
  (9 : ℝ)^15 / 54^5 = 1594323 * (3 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_nine_fifteen_div_fifty_four_five_l3146_314680


namespace NUMINAMATH_CALUDE_exactly_two_classical_models_l3146_314631

/-- Represents a random event model -/
structure RandomEventModel where
  is_finite : Bool
  has_equal_likelihood : Bool

/-- Checks if a random event model is a classical probability model -/
def is_classical_probability_model (model : RandomEventModel) : Bool :=
  model.is_finite && model.has_equal_likelihood

/-- The list of random event models given in the problem -/
def models : List RandomEventModel := [
  ⟨false, true⟩,   -- Model 1
  ⟨true, false⟩,   -- Model 2
  ⟨true, true⟩,    -- Model 3
  ⟨false, false⟩,  -- Model 4
  ⟨true, true⟩     -- Model 5
]

theorem exactly_two_classical_models : 
  (models.filter is_classical_probability_model).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_classical_models_l3146_314631


namespace NUMINAMATH_CALUDE_smallest_regiment_size_exact_smallest_regiment_size_new_uniforms_condition_l3146_314678

theorem smallest_regiment_size (m n : ℕ) (h1 : m ≥ 40) (h2 : n ≥ 30) : m * n ≥ 1200 := by
  sorry

theorem exact_smallest_regiment_size : ∃ m n : ℕ, m ≥ 40 ∧ n ≥ 30 ∧ m * n = 1200 := by
  sorry

theorem new_uniforms_condition (m n : ℕ) (h1 : m ≥ 40) (h2 : n ≥ 30) :
  (m * n : ℚ) / 100 ≥ (0.3 : ℚ) * m ∧ (m * n : ℚ) / 100 ≥ (0.4 : ℚ) * n := by
  sorry

end NUMINAMATH_CALUDE_smallest_regiment_size_exact_smallest_regiment_size_new_uniforms_condition_l3146_314678


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l3146_314671

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_english = 42)
  (h2 : failed_both = 28)
  (h3 : passed_both = 56) :
  ∃ (failed_hindi : ℝ), failed_hindi = 30 ∧
    failed_hindi + failed_english - failed_both = 100 - passed_both :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l3146_314671


namespace NUMINAMATH_CALUDE_cylinder_radius_ratio_l3146_314693

/-- Given a right circular cylinder with initial volume 6 and final volume 186,
    prove that the ratio of the new radius to the original radius is √31. -/
theorem cylinder_radius_ratio (r R h : ℝ) : 
  r > 0 → h > 0 → 
  π * r^2 * h = 6 → 
  π * R^2 * h = 186 → 
  R / r = Real.sqrt 31 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_ratio_l3146_314693


namespace NUMINAMATH_CALUDE_solution_difference_l3146_314654

theorem solution_difference (r s : ℝ) : 
  (r - 4) * (r + 4) = 24 * r - 96 →
  (s - 4) * (s + 4) = 24 * s - 96 →
  r ≠ s →
  r > s →
  r - s = 16 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l3146_314654


namespace NUMINAMATH_CALUDE_pencil_problem_l3146_314600

theorem pencil_problem (s p t : ℚ) : 
  (6 * s = 12) →
  (t = 8 * s) →
  (p = 2.5 * s + 3) →
  (t = 16 ∧ p = 8) := by
sorry

end NUMINAMATH_CALUDE_pencil_problem_l3146_314600


namespace NUMINAMATH_CALUDE_max_correct_answers_l3146_314620

theorem max_correct_answers (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) :
  total_questions = 25 →
  correct_score = 4 →
  incorrect_score = -3 →
  total_score = 57 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct ≤ 18 ∧
    ∀ c i u : ℕ,
      c + i + u = total_questions →
      correct_score * c + incorrect_score * i = total_score →
      c ≤ correct :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l3146_314620


namespace NUMINAMATH_CALUDE_cupcake_combinations_l3146_314668

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The total number of cupcakes to be purchased -/
def total_cupcakes : ℕ := 7

/-- The number of cupcake types available -/
def cupcake_types : ℕ := 5

/-- The number of cupcake types that must have at least one selected -/
def required_types : ℕ := 4

/-- The number of remaining cupcakes after selecting one of each required type -/
def remaining_cupcakes : ℕ := total_cupcakes - required_types

theorem cupcake_combinations : 
  stars_and_bars remaining_cupcakes cupcake_types = 35 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_combinations_l3146_314668


namespace NUMINAMATH_CALUDE_trig_identity_l3146_314623

theorem trig_identity : Real.sin (63 * π / 180) * Real.cos (18 * π / 180) + 
  Real.cos (63 * π / 180) * Real.cos (108 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3146_314623


namespace NUMINAMATH_CALUDE_number_division_problem_l3146_314621

theorem number_division_problem (x y : ℝ) 
  (h1 : (x - 5) / y = 7)
  (h2 : (x - 4) / 10 = 5) : 
  y = 7 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l3146_314621


namespace NUMINAMATH_CALUDE_max_trailing_zeros_l3146_314652

/-- Given three natural numbers whose sum is 1003, the maximum number of trailing zeros in their product is 7 -/
theorem max_trailing_zeros (a b c : ℕ) (h_sum : a + b + c = 1003) : 
  (∃ (n : ℕ), a * b * c = n * 10^7 ∧ n % 10 ≠ 0) ∧ 
  ¬(∃ (m : ℕ), a * b * c = m * 10^8) :=
sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_l3146_314652


namespace NUMINAMATH_CALUDE_inequality_solution_l3146_314601

theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(6 - x) > a^(2 + 3*x) ↔ 
    ((0 < a ∧ a < 1 ∧ x > 1) ∨ (a > 1 ∧ x < 1))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3146_314601


namespace NUMINAMATH_CALUDE_fifth_term_value_l3146_314625

/-- A geometric sequence with common ratio 2 and positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem fifth_term_value (a : ℕ → ℝ) :
  GeometricSequence a → a 3 * a 11 = 16 → a 5 = 1 := by
  sorry


end NUMINAMATH_CALUDE_fifth_term_value_l3146_314625


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l3146_314638

theorem salary_increase_percentage (x : ℝ) : 
  (((100 + x) / 100) * 0.8 = 1.04) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l3146_314638


namespace NUMINAMATH_CALUDE_distance_to_origin_l3146_314687

/-- The distance from the point corresponding to the complex number 2i/(1+i) to the origin is √2. -/
theorem distance_to_origin : ∃ (z : ℂ), z = (2 * Complex.I) / (1 + Complex.I) ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3146_314687


namespace NUMINAMATH_CALUDE_smallest_integer_l3146_314675

theorem smallest_integer (a b : ℕ) (ha : a = 36) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 20) :
  b ≥ 45 ∧ ∃ (b' : ℕ), b' = 45 ∧ Nat.lcm a b' / Nat.gcd a b' = 20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l3146_314675


namespace NUMINAMATH_CALUDE_fifteen_blue_points_l3146_314653

/-- Represents the configuration of points on a line -/
structure LineConfiguration where
  red_points : Fin 2 → ℕ
  blue_left : Fin 2 → ℕ
  blue_right : Fin 2 → ℕ

/-- The number of segments containing a red point with blue endpoints -/
def segments_count (config : LineConfiguration) (i : Fin 2) : ℕ :=
  config.blue_left i * config.blue_right i

/-- The total number of blue points -/
def total_blue_points (config : LineConfiguration) : ℕ :=
  config.blue_left 0 + config.blue_right 0

/-- Theorem stating that there are exactly 15 blue points -/
theorem fifteen_blue_points (config : LineConfiguration) 
  (h1 : segments_count config 0 = 56)
  (h2 : segments_count config 1 = 50)
  (h3 : config.blue_left 0 + config.blue_right 0 = config.blue_left 1 + config.blue_right 1) :
  total_blue_points config = 15 := by
  sorry


end NUMINAMATH_CALUDE_fifteen_blue_points_l3146_314653


namespace NUMINAMATH_CALUDE_probability_statements_l3146_314690

-- Define a type for days in a year
def Day := Fin 365

-- Define a type for numbers in a drawing
def DrawNumber := Fin 10

-- Function to calculate birthday probability
def birthday_probability : ℚ :=
  1 / 365

-- Function to check if drawing method is fair
def is_fair_drawing_method (draw : DrawNumber → DrawNumber → Bool) : Prop :=
  ∀ a b : DrawNumber, (draw a b) = ¬(draw b a)

-- Theorem statement
theorem probability_statements :
  (birthday_probability = 1 / 365) ∧
  (∃ draw : DrawNumber → DrawNumber → Bool, is_fair_drawing_method draw) :=
sorry

end NUMINAMATH_CALUDE_probability_statements_l3146_314690


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l3146_314628

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

/-- The theorem stating that if a geometric sequence satisfies the given condition,
    then it is a constant sequence. -/
theorem geometric_sequence_constant
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_condition : (a 3 + a 11) / a 7 ≤ 2) :
  ∃ c : ℝ, ∀ n : ℕ, a n = c :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l3146_314628


namespace NUMINAMATH_CALUDE_squirrels_and_nuts_l3146_314692

theorem squirrels_and_nuts (squirrels : ℕ) (nuts : ℕ) : 
  squirrels = 4 → squirrels - nuts = 2 → nuts = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_and_nuts_l3146_314692


namespace NUMINAMATH_CALUDE_first_ten_digits_of_expression_l3146_314641

theorem first_ten_digits_of_expression (ε : ℝ) (h : ε > 0) :
  ∃ n : ℤ, (5 + Real.sqrt 26) ^ 100 = n - ε ∧ 0 < ε ∧ ε < 1e-10 :=
sorry

end NUMINAMATH_CALUDE_first_ten_digits_of_expression_l3146_314641


namespace NUMINAMATH_CALUDE_circled_plus_two_three_four_l3146_314616

/-- The operation ⊕ is defined for real numbers a, b, and c. -/
def CircledPlus (a b c : ℝ) : ℝ := b^2 - 3*a*c

/-- Theorem: The value of ⊕(2, 3, 4) is -15. -/
theorem circled_plus_two_three_four :
  CircledPlus 2 3 4 = -15 := by
  sorry

end NUMINAMATH_CALUDE_circled_plus_two_three_four_l3146_314616


namespace NUMINAMATH_CALUDE_house_sale_profit_rate_l3146_314604

/-- The profit rate calculation for a house sale with discount, price increase, and inflation -/
theorem house_sale_profit_rate 
  (list_price : ℝ) 
  (discount_rate : ℝ) 
  (price_increase_rate : ℝ) 
  (inflation_rate : ℝ) 
  (h1 : discount_rate = 0.05)
  (h2 : price_increase_rate = 0.60)
  (h3 : inflation_rate = 0.40) : 
  ∃ (profit_rate : ℝ), 
    abs (profit_rate - ((1 + price_increase_rate) / ((1 - discount_rate) * (1 + inflation_rate)) - 1)) < 0.001 ∧ 
    abs (profit_rate - 0.203) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_profit_rate_l3146_314604


namespace NUMINAMATH_CALUDE_sqrt_66_greater_than_8_l3146_314630

theorem sqrt_66_greater_than_8 : Real.sqrt 66 > 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_66_greater_than_8_l3146_314630


namespace NUMINAMATH_CALUDE_restock_is_mode_l3146_314639

def shoe_sizes : List ℝ := [22, 22.5, 23, 23.5, 24, 24.5, 25]
def quantities : List ℕ := [3, 5, 10, 15, 8, 3, 2]
def restock_size : ℝ := 23.5

def mode (sizes : List ℝ) (quants : List ℕ) : ℝ :=
  let paired := List.zip sizes quants
  let max_quant := paired.map (λ p => p.2) |>.maximum?
  match paired.find? (λ p => p.2 = max_quant) with
  | some (size, _) => size
  | none => 0  -- This case should not occur if the lists are non-empty

theorem restock_is_mode :
  mode shoe_sizes quantities = restock_size :=
sorry

end NUMINAMATH_CALUDE_restock_is_mode_l3146_314639


namespace NUMINAMATH_CALUDE_boric_acid_weight_l3146_314659

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of Boron in g/mol -/
def atomic_weight_B : ℝ := 10.81

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Hydrogen atoms in Boric acid -/
def num_H : ℕ := 3

/-- The number of Boron atoms in Boric acid -/
def num_B : ℕ := 1

/-- The number of Oxygen atoms in Boric acid -/
def num_O : ℕ := 3

/-- The molecular weight of Boric acid (H3BO3) in g/mol -/
def molecular_weight_boric_acid : ℝ :=
  num_H * atomic_weight_H + num_B * atomic_weight_B + num_O * atomic_weight_O

theorem boric_acid_weight :
  molecular_weight_boric_acid = 61.834 := by sorry

end NUMINAMATH_CALUDE_boric_acid_weight_l3146_314659


namespace NUMINAMATH_CALUDE_new_average_after_joining_l3146_314617

theorem new_average_after_joining (initial_count : ℕ) (initial_average : ℚ) (new_member_amount : ℚ) :
  initial_count = 7 →
  initial_average = 14 →
  new_member_amount = 56 →
  (initial_count * initial_average + new_member_amount) / (initial_count + 1) = 19.25 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_joining_l3146_314617


namespace NUMINAMATH_CALUDE_half_dollar_and_dollar_heads_probability_l3146_314685

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the result of flipping four coins -/
structure FourCoinFlip :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (halfDollar : CoinFlip)
  (oneDollar : CoinFlip)

/-- The set of all possible outcomes when flipping four coins -/
def allOutcomes : Finset FourCoinFlip := sorry

/-- The set of favorable outcomes (half-dollar and one-dollar are both heads) -/
def favorableOutcomes : Finset FourCoinFlip := sorry

/-- The probability of an event occurring -/
def probability (event : Finset FourCoinFlip) : ℚ :=
  (event.card : ℚ) / (allOutcomes.card : ℚ)

theorem half_dollar_and_dollar_heads_probability :
  probability favorableOutcomes = 1/4 := by sorry

end NUMINAMATH_CALUDE_half_dollar_and_dollar_heads_probability_l3146_314685


namespace NUMINAMATH_CALUDE_square_area_given_circle_l3146_314643

-- Define the circle's area
def circle_area : ℝ := 39424

-- Define the relationship between square perimeter and circle radius
def square_perimeter_equals_circle_radius (square_side : ℝ) (circle_radius : ℝ) : Prop :=
  4 * square_side = circle_radius

-- Theorem statement
theorem square_area_given_circle (square_side : ℝ) (circle_radius : ℝ) :
  circle_area = Real.pi * circle_radius^2 →
  square_perimeter_equals_circle_radius square_side circle_radius →
  square_side^2 = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_given_circle_l3146_314643


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l3146_314627

/-- 
Given that (m+2)x^(m^2-2) + 2x + 1 = 0 is a quadratic equation in x and m+2 ≠ 0, 
prove that m = 2.
-/
theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 2) * x^(m^2 - 2) + 2*x + 1 = a*x^2 + b*x + c) →
  (m + 2 ≠ 0) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l3146_314627


namespace NUMINAMATH_CALUDE_students_in_same_group_l3146_314661

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 2

/-- The probability of a student joining any specific group -/
def prob_join_group : ℚ := 1 / num_groups

/-- The probability that both students are in the same group -/
def prob_same_group : ℚ := num_groups * (prob_join_group * prob_join_group)

theorem students_in_same_group :
  prob_same_group = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_students_in_same_group_l3146_314661


namespace NUMINAMATH_CALUDE_min_value_expression_l3146_314642

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 4 / y) ≥ 9 ∧
  ((x + y) * (1 / x + 4 / y) = 9 ↔ y = 2 * x) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3146_314642


namespace NUMINAMATH_CALUDE_not_suff_not_nec_condition_l3146_314619

theorem not_suff_not_nec_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) := by
  sorry

end NUMINAMATH_CALUDE_not_suff_not_nec_condition_l3146_314619


namespace NUMINAMATH_CALUDE_alternative_plan_cost_is_eleven_l3146_314688

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

end NUMINAMATH_CALUDE_alternative_plan_cost_is_eleven_l3146_314688


namespace NUMINAMATH_CALUDE_creeping_jennies_count_l3146_314629

/-- The number of creeping jennies per planter -/
def creeping_jennies : ℕ := sorry

/-- The cost of a palm fern -/
def palm_fern_cost : ℚ := 15

/-- The cost of a creeping jenny -/
def creeping_jenny_cost : ℚ := 4

/-- The cost of a geranium -/
def geranium_cost : ℚ := 3.5

/-- The number of geraniums per planter -/
def geraniums_per_planter : ℕ := 4

/-- The number of planters -/
def num_planters : ℕ := 4

/-- The total cost for all planters -/
def total_cost : ℚ := 180

theorem creeping_jennies_count : 
  creeping_jennies = 4 ∧ 
  (num_planters : ℚ) * (palm_fern_cost + creeping_jenny_cost * (creeping_jennies : ℚ) + 
    geranium_cost * (geraniums_per_planter : ℚ)) = total_cost :=
sorry

end NUMINAMATH_CALUDE_creeping_jennies_count_l3146_314629


namespace NUMINAMATH_CALUDE_ellipse_properties_l3146_314694

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/5 + y^2 = 1

-- Define the right focus F
def right_focus : ℝ × ℝ := (2, 0)

-- Define the line l passing through F
def line_l (k : ℝ) (x : ℝ) : ℝ := k*(x - 2)

-- Define points A and B on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse_C x y ∧ ∃ k, y = line_l k x

-- Define point M on y-axis
def point_M (y : ℝ) : ℝ × ℝ := (0, y)

-- Define vectors MA, MB, FA, and FB
def vector_MA (x y y0 : ℝ) : ℝ × ℝ := (x, y - y0)
def vector_MB (x y y0 : ℝ) : ℝ × ℝ := (x, y - y0)
def vector_FA (x y : ℝ) : ℝ × ℝ := (x - 2, y)
def vector_FB (x y : ℝ) : ℝ × ℝ := (x - 2, y)

theorem ellipse_properties :
  ∀ (x1 y1 x2 y2 y0 m n : ℝ),
  point_on_ellipse x1 y1 →
  point_on_ellipse x2 y2 →
  vector_MA x1 y1 y0 = m • (vector_FA x1 y1) →
  vector_MB x2 y2 y0 = n • (vector_FB x2 y2) →
  m + n = 10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3146_314694


namespace NUMINAMATH_CALUDE_largest_base_sum_not_sixteen_l3146_314634

/-- Represents a number in a given base --/
structure BaseNumber (base : ℕ) where
  digits : List ℕ
  valid : ∀ d ∈ digits, d < base

/-- Computes the sum of digits of a BaseNumber --/
def sumOfDigits {base : ℕ} (n : BaseNumber base) : ℕ :=
  n.digits.sum

/-- Represents 11^4 in different bases --/
def elevenFourth (base : ℕ) : BaseNumber base :=
  if base ≥ 7 then
    ⟨[1, 4, 6, 4, 1], sorry⟩
  else if base = 6 then
    ⟨[1, 5, 0, 4, 1], sorry⟩
  else
    ⟨[], sorry⟩  -- Undefined for bases less than 6

/-- The theorem to be proved --/
theorem largest_base_sum_not_sixteen :
  (∃ b : ℕ, b > 0 ∧ sumOfDigits (elevenFourth b) ≠ 16) ∧
  (∀ b : ℕ, b > 6 → sumOfDigits (elevenFourth b) = 16) :=
sorry

end NUMINAMATH_CALUDE_largest_base_sum_not_sixteen_l3146_314634


namespace NUMINAMATH_CALUDE_smallest_n_doughnuts_l3146_314609

theorem smallest_n_doughnuts : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → (15 * m - 1) % 5 = 0 → m ≥ n) ∧
  (15 * n - 1) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_doughnuts_l3146_314609


namespace NUMINAMATH_CALUDE_fixed_root_quadratic_l3146_314645

theorem fixed_root_quadratic (k : ℝ) : 
  ∃ x : ℝ, x^2 + (k + 3) * x + k + 2 = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_root_quadratic_l3146_314645


namespace NUMINAMATH_CALUDE_shop_owner_profit_l3146_314649

/-- Calculates the percentage profit of a shop owner who cheats while buying and selling -/
theorem shop_owner_profit (buy_cheat : ℝ) (sell_cheat : ℝ) : 
  buy_cheat = 0.12 → sell_cheat = 0.3 → 
  (((1 + buy_cheat) / (1 - sell_cheat) - 1) * 100 : ℝ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_shop_owner_profit_l3146_314649


namespace NUMINAMATH_CALUDE_polynomial_sum_l3146_314697

/-- Given a polynomial P such that P + (x^2 - y^2) = x^2 + y^2, then P = 2y^2 -/
theorem polynomial_sum (x y : ℝ) (P : ℝ → ℝ) :
  (∀ x, P x + (x^2 - y^2) = x^2 + y^2) → P x = 2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3146_314697


namespace NUMINAMATH_CALUDE_trapezoid_median_l3146_314672

/-- Given a triangle and a trapezoid with equal areas and the same altitude,
    where the base of the triangle is 24 inches and the sum of the bases of
    the trapezoid is 40 inches, the median of the trapezoid is 20 inches. -/
theorem trapezoid_median (h : ℝ) (triangle_area trapezoid_area : ℝ) 
  (triangle_base trapezoid_base_sum : ℝ) (trapezoid_median : ℝ) :
  h > 0 →
  triangle_area = trapezoid_area →
  triangle_base = 24 →
  trapezoid_base_sum = 40 →
  triangle_area = (1 / 2) * triangle_base * h →
  trapezoid_area = trapezoid_median * h →
  trapezoid_median = trapezoid_base_sum / 2 →
  trapezoid_median = 20 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_median_l3146_314672


namespace NUMINAMATH_CALUDE_min_sum_squares_l3146_314695

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4.8 ∧ x^2 + y^2 + z^2 ≥ m ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^3 + y₀^3 + z₀^3 - 3*x₀*y₀*z₀ = 8 ∧ x₀^2 + y₀^2 + z₀^2 = m :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3146_314695


namespace NUMINAMATH_CALUDE_yadav_savings_l3146_314663

/-- Mr. Yadav's monthly savings calculation --/
def monthly_savings (salary : ℝ) : ℝ :=
  salary * (1 - 0.6 - 0.5 * (1 - 0.6))

/-- Mr. Yadav's yearly savings calculation --/
def yearly_savings (salary : ℝ) : ℝ :=
  12 * monthly_savings salary

/-- Theorem: Mr. Yadav's yearly savings are 46800 --/
theorem yadav_savings :
  ∃ (salary : ℝ),
    salary > 0 ∧
    0.5 * (1 - 0.6) * salary = 3900 ∧
    yearly_savings salary = 46800 :=
sorry

end NUMINAMATH_CALUDE_yadav_savings_l3146_314663


namespace NUMINAMATH_CALUDE_similar_triangles_height_l3146_314614

/-- Given two similar triangles with an area ratio of 1:9 and the height of the smaller triangle is 5 cm,
    prove that the corresponding height of the larger triangle is 15 cm. -/
theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 5 →
  area_ratio = 9 →
  ∃ h_large : ℝ, h_large = 15 ∧ h_large / h_small = Real.sqrt area_ratio :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l3146_314614


namespace NUMINAMATH_CALUDE_ellipse_equation_correct_l3146_314636

/-- An ellipse with foci (-1,0) and (1,0), passing through (2,0) -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- The foci of the ellipse -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, 0), (1, 0))

/-- A point on the ellipse -/
def P : ℝ × ℝ := (2, 0)

theorem ellipse_equation_correct :
  (∀ p ∈ Ellipse, 
    (Real.sqrt ((p.1 - foci.1.1)^2 + (p.2 - foci.1.2)^2) + 
     Real.sqrt ((p.1 - foci.2.1)^2 + (p.2 - foci.2.2)^2)) = 
    (Real.sqrt ((P.1 - foci.1.1)^2 + (P.2 - foci.1.2)^2) + 
     Real.sqrt ((P.1 - foci.2.1)^2 + (P.2 - foci.2.2)^2))) ∧
  P ∈ Ellipse := by
  sorry

#check ellipse_equation_correct

end NUMINAMATH_CALUDE_ellipse_equation_correct_l3146_314636


namespace NUMINAMATH_CALUDE_fraction_division_addition_l3146_314607

theorem fraction_division_addition : (3 / 7 : ℚ) / 4 + 2 / 7 = 11 / 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_addition_l3146_314607


namespace NUMINAMATH_CALUDE_sum_of_roots_l3146_314650

theorem sum_of_roots (r s : ℝ) : 
  (r ≠ s) → 
  (2 * (r^2 + 1/r^2) - 3 * (r + 1/r) = 1) → 
  (2 * (s^2 + 1/s^2) - 3 * (s + 1/s) = 1) → 
  (r + s = -5/2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3146_314650


namespace NUMINAMATH_CALUDE_base_conversion_l3146_314640

/-- Given that 132 in base k is equal to 42 in base 10, prove that k = 5 -/
theorem base_conversion (k : ℕ) : k ^ 2 + 3 * k + 2 = 42 → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l3146_314640


namespace NUMINAMATH_CALUDE_two_books_different_genres_l3146_314648

/-- Represents the number of books in each genre -/
def books_per_genre : ℕ := 3

/-- Represents the number of genres -/
def num_genres : ℕ := 3

/-- Represents the total number of books -/
def total_books : ℕ := books_per_genre * num_genres

/-- Calculates the number of ways to choose two books of different genres -/
def choose_two_different_genres : ℕ := 
  (total_books * (total_books - books_per_genre)) / 2

theorem two_books_different_genres : 
  choose_two_different_genres = 27 := by sorry

end NUMINAMATH_CALUDE_two_books_different_genres_l3146_314648


namespace NUMINAMATH_CALUDE_linear_function_properties_l3146_314624

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 3

-- Theorem statement
theorem linear_function_properties :
  (f 1 = 1) ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ f x = y) ∧
  (f (3/2) = 0) ∧
  (∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l3146_314624


namespace NUMINAMATH_CALUDE_double_discount_l3146_314605

/-- Calculates the final price as a percentage of the original price after applying two consecutive discounts -/
theorem double_discount (initial_discount coupon_discount : ℝ) :
  initial_discount = 0.4 →
  coupon_discount = 0.25 →
  (1 - initial_discount) * (1 - coupon_discount) = 0.45 :=
by
  sorry

#check double_discount

end NUMINAMATH_CALUDE_double_discount_l3146_314605


namespace NUMINAMATH_CALUDE_jasons_shopping_l3146_314626

theorem jasons_shopping (total_spent jacket_cost : ℝ) 
  (h1 : total_spent = 14.28)
  (h2 : jacket_cost = 4.74) :
  total_spent - jacket_cost = 9.54 := by
sorry

end NUMINAMATH_CALUDE_jasons_shopping_l3146_314626


namespace NUMINAMATH_CALUDE_magic_square_solution_l3146_314679

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℚ)
  (row_sum : ℚ)
  (magic_property : 
    a11 + a12 + a13 = row_sum ∧
    a21 + a22 + a23 = row_sum ∧
    a31 + a32 + a33 = row_sum ∧
    a11 + a21 + a31 = row_sum ∧
    a12 + a22 + a32 = row_sum ∧
    a13 + a23 + a33 = row_sum ∧
    a11 + a22 + a33 = row_sum ∧
    a13 + a22 + a31 = row_sum)

/-- The theorem stating the solution to the magic square problem -/
theorem magic_square_solution (ms : MagicSquare) 
  (h1 : ms.a12 = 25)
  (h2 : ms.a13 = 64)
  (h3 : ms.a21 = 3) :
  ms.a11 = 272 / 3 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_solution_l3146_314679


namespace NUMINAMATH_CALUDE_ellipse_inscribed_circle_radius_specific_ellipse_inscribed_circle_radius_l3146_314666

/-- The radius of the largest circle inside an ellipse with its center at a focus --/
theorem ellipse_inscribed_circle_radius (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let c := Real.sqrt (a^2 - b^2)
  let r := c - b
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x - c)^2 + y^2 ≥ r^2) ∧
  (∃ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ∧ (x - c)^2 + y^2 = r^2) :=
by sorry

/-- The specific case for the given ellipse --/
theorem specific_ellipse_inscribed_circle_radius :
  let a : ℝ := 6
  let b : ℝ := 5
  let c := Real.sqrt (a^2 - b^2)
  let r := c - a
  r = Real.sqrt 11 - 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_inscribed_circle_radius_specific_ellipse_inscribed_circle_radius_l3146_314666


namespace NUMINAMATH_CALUDE_only_set_C_is_right_triangle_l3146_314603

-- Define a function to check if three numbers satisfy the Pythagorean theorem
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Theorem statement
theorem only_set_C_is_right_triangle :
  (¬ isPythagoreanTriple 3 4 2) ∧
  (¬ isPythagoreanTriple 5 12 15) ∧
  (isPythagoreanTriple 8 15 17) ∧
  (¬ isPythagoreanTriple 9 16 25) :=
by sorry


end NUMINAMATH_CALUDE_only_set_C_is_right_triangle_l3146_314603


namespace NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l3146_314618

theorem at_least_one_fraction_less_than_two (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l3146_314618


namespace NUMINAMATH_CALUDE_constant_pace_jogging_l3146_314664

/-- Represents the time taken to jog a certain distance at a constant pace -/
structure JoggingTime where
  distance : ℝ
  time : ℝ

/-- Given a constant jogging pace, if it takes 24 minutes to jog 3 miles,
    then it will take 12 minutes to jog 1.5 miles -/
theorem constant_pace_jogging 
  (pace : ℝ) 
  (gym : JoggingTime) 
  (park : JoggingTime) 
  (h1 : gym.distance = 3) 
  (h2 : gym.time = 24) 
  (h3 : park.distance = 1.5) 
  (h4 : pace > 0) 
  (h5 : ∀ j : JoggingTime, j.time = j.distance / pace) : 
  park.time = 12 :=
sorry

end NUMINAMATH_CALUDE_constant_pace_jogging_l3146_314664


namespace NUMINAMATH_CALUDE_p_minus_q_plus_r_equals_two_thirds_l3146_314667

theorem p_minus_q_plus_r_equals_two_thirds
  (p q r : ℚ)
  (hp : 3 / p = 6)
  (hq : 3 / q = 18)
  (hr : 5 / r = 15) :
  p - q + r = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_p_minus_q_plus_r_equals_two_thirds_l3146_314667


namespace NUMINAMATH_CALUDE_polynomial_shift_root_existence_l3146_314669

/-- A polynomial of degree 10 with leading coefficient 1 -/
def Polynomial10 := {p : Polynomial ℝ // p.degree = 10 ∧ p.leadingCoeff = 1}

theorem polynomial_shift_root_existence (P Q : Polynomial10) 
  (h : ∀ x : ℝ, P.val.eval x ≠ Q.val.eval x) :
  ∃ x : ℝ, (P.val.eval (x + 1)) = (Q.val.eval (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_shift_root_existence_l3146_314669


namespace NUMINAMATH_CALUDE_least_common_multiple_of_first_ten_l3146_314684

/-- The least positive integer divisible by each of the first ten positive integers -/
def leastCommonMultiple : ℕ := 2520

/-- The set of the first ten positive integers -/
def firstTenIntegers : Finset ℕ := Finset.range 10

theorem least_common_multiple_of_first_ten :
  (∀ n ∈ firstTenIntegers, leastCommonMultiple % (n + 1) = 0) ∧
  (∀ m : ℕ, m > 0 → m < leastCommonMultiple →
    ∃ k ∈ firstTenIntegers, m % (k + 1) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_first_ten_l3146_314684


namespace NUMINAMATH_CALUDE_wheel_revolutions_l3146_314662

-- Define constants
def wheel_diameter : ℝ := 8
def distance_km : ℝ := 2
def km_to_feet : ℝ := 3280.84

-- Define the theorem
theorem wheel_revolutions :
  let wheel_circumference := π * wheel_diameter
  let distance_feet := distance_km * km_to_feet
  let revolutions := distance_feet / wheel_circumference
  revolutions = 820.21 / π := by
sorry

end NUMINAMATH_CALUDE_wheel_revolutions_l3146_314662
