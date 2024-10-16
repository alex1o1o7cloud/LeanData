import Mathlib

namespace NUMINAMATH_CALUDE_fence_painting_l2516_251696

theorem fence_painting (total_length : ℝ) (percentage_difference : ℝ) : 
  total_length = 792 → percentage_difference = 0.2 → 
  ∃ (x : ℝ), x + (1 + percentage_difference) * x = total_length ∧ 
  (1 + percentage_difference) * x = 432 :=
by
  sorry

end NUMINAMATH_CALUDE_fence_painting_l2516_251696


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_negative_one_l2516_251672

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)

-- State the theorem
theorem even_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_negative_one_l2516_251672


namespace NUMINAMATH_CALUDE_line_through_points_l2516_251650

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_through_points : 
  let p1 : Point := ⟨2, 5⟩
  let p2 : Point := ⟨6, 17⟩
  let p3 : Point := ⟨10, 29⟩
  let p4 : Point := ⟨34, s⟩
  collinear p1 p2 p3 ∧ collinear p1 p2 p4 → s = 101 := by
  sorry


end NUMINAMATH_CALUDE_line_through_points_l2516_251650


namespace NUMINAMATH_CALUDE_expression_simplification_redundant_condition_l2516_251605

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) = x :=
by sorry

theorem redundant_condition (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (z : ℝ), z ≠ y ∧
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) =
  (2 * x * (x^2 * z - x * z^2) + x * z * (2 * x * z - x^2)) / (x^2 * z) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_redundant_condition_l2516_251605


namespace NUMINAMATH_CALUDE_line_AB_equation_line_l_equations_l2516_251697

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the midpoint of AB
def midpoint_AB (x y : ℝ) : Prop := x = 3 ∧ y = 2

-- Define the point (2, 0) on line l
def point_on_l (x y : ℝ) : Prop := x = 2 ∧ y = 0

-- Define the area of triangle OMN
def area_OMN : ℝ := 6

-- Theorem for the equation of line AB
theorem line_AB_equation (A B : ℝ × ℝ) :
  parabola A.1 A.2 → parabola B.1 B.2 → midpoint_AB ((A.1 + B.1)/2) ((A.2 + B.2)/2) →
  ∃ (k : ℝ), A.1 - A.2 - k = 0 ∧ B.1 - B.2 - k = 0 :=
sorry

-- Theorem for the equations of line l
theorem line_l_equations :
  ∃ (M N : ℝ × ℝ), parabola M.1 M.2 ∧ parabola N.1 N.2 ∧
  point_on_l 2 0 ∧
  (∃ (m : ℝ), (M.2 = m*M.1 - 2 ∧ N.2 = m*N.1 - 2) ∨
              (M.2 = -m*M.1 - 2 ∧ N.2 = -m*N.1 - 2)) ∧
  area_OMN = 6 :=
sorry

end NUMINAMATH_CALUDE_line_AB_equation_line_l_equations_l2516_251697


namespace NUMINAMATH_CALUDE_people_dislike_radio_and_music_l2516_251632

theorem people_dislike_radio_and_music
  (total_people : ℕ)
  (radio_dislike_percent : ℚ)
  (music_dislike_percent : ℚ)
  (h_total : total_people = 1500)
  (h_radio : radio_dislike_percent = 40 / 100)
  (h_music : music_dislike_percent = 15 / 100) :
  (total_people : ℚ) * radio_dislike_percent * music_dislike_percent = 90 := by
  sorry

end NUMINAMATH_CALUDE_people_dislike_radio_and_music_l2516_251632


namespace NUMINAMATH_CALUDE_ad_probability_is_one_third_l2516_251645

/-- The duration of advertisements per hour in minutes -/
def ad_duration : ℕ := 20

/-- The total duration of an hour in minutes -/
def hour_duration : ℕ := 60

/-- The probability of seeing an advertisement when turning on the TV -/
def ad_probability : ℚ := ad_duration / hour_duration

theorem ad_probability_is_one_third : ad_probability = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ad_probability_is_one_third_l2516_251645


namespace NUMINAMATH_CALUDE_trig_identity_l2516_251652

theorem trig_identity (α : Real) (h : Real.sin (2 * Real.pi / 3 + α) = 1 / 3) :
  Real.cos (5 * Real.pi / 6 - α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2516_251652


namespace NUMINAMATH_CALUDE_point_inside_ellipse_l2516_251638

/-- A point A(a, 1) is inside the ellipse x²/4 + y²/2 = 1 if and only if -√2 < a < √2 -/
theorem point_inside_ellipse (a : ℝ) : 
  (a^2 / 4 + 1 / 2 < 1) ↔ (-Real.sqrt 2 < a ∧ a < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_point_inside_ellipse_l2516_251638


namespace NUMINAMATH_CALUDE_eight_fifteen_divided_by_sixty_four_six_l2516_251629

theorem eight_fifteen_divided_by_sixty_four_six : 8^15 / 64^6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_eight_fifteen_divided_by_sixty_four_six_l2516_251629


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2516_251647

def sequence_a (n : ℕ) : ℝ := 2^n

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - 2

def sequence_T (n : ℕ) : ℝ := n * 2^(n+1) - 2^(n+1) + 2

theorem smallest_n_for_inequality :
  ∀ n : ℕ, (∀ k < n, sequence_T k - k * 2^(k+1) + 50 ≥ 0) ∧
           (sequence_T n - n * 2^(n+1) + 50 < 0) →
  n = 5 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2516_251647


namespace NUMINAMATH_CALUDE_sqrt2_fractional_part_bounds_l2516_251618

theorem sqrt2_fractional_part_bounds :
  (∀ n : ℕ, n * Real.sqrt 2 - ⌊n * Real.sqrt 2⌋ > 1 / (2 * n * Real.sqrt 2)) ∧
  (∀ ε > 0, ∃ n : ℕ, n * Real.sqrt 2 - ⌊n * Real.sqrt 2⌋ < 1 / (2 * n * Real.sqrt 2) + ε) := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_fractional_part_bounds_l2516_251618


namespace NUMINAMATH_CALUDE_vector_calculation_l2516_251668

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ m ↦ (4, m)

theorem vector_calculation (m : ℝ) (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  (5 * a.1 - 3 * (b m).1, 5 * a.2 - 3 * (b m).2) = (-7, -16) := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l2516_251668


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l2516_251636

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 20 [ZMOD 25]) : 
  x^2 ≡ 4 [ZMOD 25] := by
sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l2516_251636


namespace NUMINAMATH_CALUDE_probability_diamond_ace_face_card_l2516_251653

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (favorable_cards : ℕ)
  (h_total : total_cards = 54)
  (h_favorable : favorable_cards = 26)

/-- The probability of selecting at least one favorable card in two draws with replacement -/
def probability_favorable_card (d : Deck) : ℚ :=
  1 - (↑(d.total_cards - d.favorable_cards) / ↑d.total_cards) ^ 2

theorem probability_diamond_ace_face_card :
  ∃ d : Deck, probability_favorable_card d = 533 / 729 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_ace_face_card_l2516_251653


namespace NUMINAMATH_CALUDE_safari_animal_count_l2516_251662

theorem safari_animal_count (total animals : ℕ) (antelopes rabbits hyenas wild_dogs leopards : ℕ) :
  total = 605 →
  antelopes = 80 →
  rabbits = antelopes + 34 →
  hyenas = antelopes + rabbits - 42 →
  wild_dogs > hyenas →
  leopards * 2 = rabbits →
  total = antelopes + rabbits + hyenas + wild_dogs + leopards →
  wild_dogs - hyenas = 50 := by
  sorry

end NUMINAMATH_CALUDE_safari_animal_count_l2516_251662


namespace NUMINAMATH_CALUDE_tom_neither_soccer_nor_test_l2516_251693

theorem tom_neither_soccer_nor_test (soccer_prob : ℚ) (test_prob : ℚ) 
  (h_soccer : soccer_prob = 5 / 8)
  (h_test : test_prob = 1 / 4)
  (h_independent : True) -- Assumption of independence
  : (1 - soccer_prob) * (1 - test_prob) = 9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_tom_neither_soccer_nor_test_l2516_251693


namespace NUMINAMATH_CALUDE_club_members_theorem_l2516_251631

theorem club_members_theorem (total : ℕ) (left_handed : ℕ) (rock_fans : ℕ) (right_handed_non_fans : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : rock_fans = 18)
  (h4 : right_handed_non_fans = 3)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ (left_handed_rock_fans : ℕ),
    left_handed_rock_fans = 6 ∧
    left_handed_rock_fans ≤ left_handed ∧
    left_handed_rock_fans ≤ rock_fans ∧
    left_handed_rock_fans + (left_handed - left_handed_rock_fans) + 
    (rock_fans - left_handed_rock_fans) + right_handed_non_fans = total :=
by
  sorry

end NUMINAMATH_CALUDE_club_members_theorem_l2516_251631


namespace NUMINAMATH_CALUDE_last_three_digits_of_11_pow_210_l2516_251669

theorem last_three_digits_of_11_pow_210 : 11^210 ≡ 601 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_11_pow_210_l2516_251669


namespace NUMINAMATH_CALUDE_octal_55_to_binary_l2516_251616

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := 
  (n / 10) * 8 + (n % 10)

/-- Converts a decimal number to binary --/
def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

/-- Represents a binary number as a natural number --/
def binary_to_nat (l : List ℕ) : ℕ :=
  l.foldl (fun acc d => 2 * acc + d) 0

theorem octal_55_to_binary : 
  binary_to_nat (decimal_to_binary (octal_to_decimal 55)) = binary_to_nat [1,0,1,1,0,1] := by
  sorry

end NUMINAMATH_CALUDE_octal_55_to_binary_l2516_251616


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2516_251657

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, |x + a| < b ↔ 2 < x ∧ x < 4) → a * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2516_251657


namespace NUMINAMATH_CALUDE_only_setC_forms_right_triangle_l2516_251635

-- Define a function to check if three numbers can form a right triangle
def canFormRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the sets of line segments
def setA : List ℕ := [4, 5, 6]
def setB : List ℕ := [5, 7, 9]
def setC : List ℕ := [6, 8, 10]
def setD : List ℕ := [7, 8, 9]

-- Theorem stating that only set C can form a right triangle
theorem only_setC_forms_right_triangle :
  (¬ canFormRightTriangle setA[0] setA[1] setA[2]) ∧
  (¬ canFormRightTriangle setB[0] setB[1] setB[2]) ∧
  (canFormRightTriangle setC[0] setC[1] setC[2]) ∧
  (¬ canFormRightTriangle setD[0] setD[1] setD[2]) :=
by
  sorry

#check only_setC_forms_right_triangle

end NUMINAMATH_CALUDE_only_setC_forms_right_triangle_l2516_251635


namespace NUMINAMATH_CALUDE_todays_production_l2516_251690

def average_production (total_production : ℕ) (days : ℕ) : ℚ :=
  (total_production : ℚ) / (days : ℚ)

theorem todays_production
  (h1 : average_production (9 * 50) 9 = 50)
  (h2 : average_production ((9 * 50) + x) 10 = 55)
  : x = 100 := by
  sorry

end NUMINAMATH_CALUDE_todays_production_l2516_251690


namespace NUMINAMATH_CALUDE_A_inter_B_l2516_251628

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

theorem A_inter_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_A_inter_B_l2516_251628


namespace NUMINAMATH_CALUDE_athlete_score_comparison_l2516_251640

theorem athlete_score_comparison 
  (p₁ p₂ p₃ : ℝ) 
  (hp₁ : p₁ > 0) 
  (hp₂ : p₂ > 0) 
  (hp₃ : p₃ > 0) : 
  (16/25) * p₁ + (9/25) * p₂ + (4/15) * p₃ > 
  (16/25) * p₁ + (1/4) * p₂ + (27/128) * p₃ :=
sorry

end NUMINAMATH_CALUDE_athlete_score_comparison_l2516_251640


namespace NUMINAMATH_CALUDE_marlas_grid_squares_per_row_l2516_251677

/-- Represents a grid with colored squares -/
structure ColoredGrid where
  rows : ℕ
  squaresPerRow : ℕ
  redSquares : ℕ
  blueRows : ℕ
  greenSquares : ℕ

/-- The number of squares in each row of Marla's grid -/
def marlasGridSquaresPerRow : ℕ := 15

/-- Theorem stating that Marla's grid has 15 squares per row -/
theorem marlas_grid_squares_per_row :
  ∃ (g : ColoredGrid),
    g.rows = 10 ∧
    g.redSquares = 24 ∧
    g.blueRows = 4 ∧
    g.greenSquares = 66 ∧
    g.squaresPerRow = marlasGridSquaresPerRow :=
by sorry


end NUMINAMATH_CALUDE_marlas_grid_squares_per_row_l2516_251677


namespace NUMINAMATH_CALUDE_circle_center_from_equation_l2516_251659

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in standard form --/
def CircleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_center_from_equation :
  ∃ (c : Circle), (∀ x y : ℝ, CircleEquation c x y ↔ (x - 1)^2 + (y - 2)^2 = 5) ∧ c.center = (1, 2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_from_equation_l2516_251659


namespace NUMINAMATH_CALUDE_program_count_proof_l2516_251684

/-- The number of thirty-minute programs in a television schedule where
    one-fourth of the airing time is spent on commercials and
    45 minutes are spent on commercials for the whole duration of these programs. -/
def number_of_programs : ℕ := 6

/-- The duration of each program in minutes. -/
def program_duration : ℕ := 30

/-- The fraction of airing time spent on commercials. -/
def commercial_fraction : ℚ := 1/4

/-- The total time spent on commercials for all programs in minutes. -/
def total_commercial_time : ℕ := 45

theorem program_count_proof :
  number_of_programs = total_commercial_time / (commercial_fraction * program_duration) :=
by sorry

end NUMINAMATH_CALUDE_program_count_proof_l2516_251684


namespace NUMINAMATH_CALUDE_sandbox_fill_cost_l2516_251681

/-- The cost to fill a square sandbox with sand -/
theorem sandbox_fill_cost : 
  ∀ (length width depth bag_coverage bag_cost : ℝ),
    length = 3 →
    width = 3 →
    depth = 1 →
    bag_coverage = 3 →
    bag_cost = 4 →
    (length * width * depth / bag_coverage) * bag_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_fill_cost_l2516_251681


namespace NUMINAMATH_CALUDE_average_weight_abc_l2516_251620

theorem average_weight_abc (a b c : ℝ) 
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 43)
  (h3 : b = 31) :
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_abc_l2516_251620


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l2516_251622

theorem parabola_intersection_distance : 
  ∀ (p q r s : ℝ), 
  (∃ x y : ℝ, y = 3*x^2 - 6*x + 3 ∧ y = -x^2 - 3*x + 3 ∧ ((x = p ∧ y = q) ∨ (x = r ∧ y = s))) → 
  r ≥ p → 
  r - p = 3/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l2516_251622


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l2516_251623

/-- The height of the tree that the monkey climbs -/
def tree_height : ℕ := 20

/-- The height the monkey climbs in one hour during the first 17 hours -/
def hourly_climb : ℕ := 3

/-- The height the monkey slips back in one hour during the first 17 hours -/
def hourly_slip : ℕ := 2

/-- The number of hours it takes the monkey to reach the top of the tree -/
def total_hours : ℕ := 18

/-- The height the monkey climbs in the last hour -/
def final_climb : ℕ := 3

theorem monkey_climb_theorem :
  tree_height = (total_hours - 1) * (hourly_climb - hourly_slip) + final_climb :=
by sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l2516_251623


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2516_251655

/-- The remainder when x^3 + 3 is divided by x^2 + 2 is -2x + 3 -/
theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, x^3 + 3 = (x^2 + 2) * q + (-2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2516_251655


namespace NUMINAMATH_CALUDE_triangle_problem_l2516_251619

/-- Triangle with side lengths a, b, c and inradius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ

/-- The main theorem stating the properties of the three triangles -/
theorem triangle_problem (H₁ H₂ H₃ : Triangle) : 
  (∃ (d : ℝ), H₁.b = (H₁.a + H₁.c) / 2 ∧ 
               H₁.a = H₁.b - d ∧ 
               H₁.c = H₁.b + d) →
  (H₂.a = H₁.a - 10 ∧ H₂.b = H₁.b - 10 ∧ H₂.c = H₁.c - 10) →
  (H₃.a = H₁.a + 14 ∧ H₃.b = H₁.b + 14 ∧ H₃.c = H₁.c + 14) →
  (H₂.r = H₁.r - 5) →
  (H₃.r = H₁.r + 5) →
  (H₁.a = 25 ∧ H₁.b = 38 ∧ H₁.c = 51) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2516_251619


namespace NUMINAMATH_CALUDE_extra_crayons_l2516_251612

theorem extra_crayons (num_packs : ℕ) (crayons_per_pack : ℕ) (total_crayons : ℕ) : 
  num_packs = 4 →
  crayons_per_pack = 10 →
  total_crayons = 40 →
  total_crayons - (num_packs * crayons_per_pack) = 0 := by
  sorry

end NUMINAMATH_CALUDE_extra_crayons_l2516_251612


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2516_251614

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {(x, y) | y = 2 * x - 1}
def B : Set (ℝ × ℝ) := {(x, y) | y = x + 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {(4, 7)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2516_251614


namespace NUMINAMATH_CALUDE_nested_radical_solution_l2516_251698

theorem nested_radical_solution :
  ∃ x : ℝ, x > 0 ∧ x^2 = 6 + x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_nested_radical_solution_l2516_251698


namespace NUMINAMATH_CALUDE_inheritance_tax_calculation_l2516_251678

theorem inheritance_tax_calculation (x : ℝ) : 
  (0.2 * x + 0.1 * (x - 0.2 * x) = 10500) → x = 37500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_tax_calculation_l2516_251678


namespace NUMINAMATH_CALUDE_complex_quadratic_roots_l2516_251663

theorem complex_quadratic_roots : 
  ∃ (z₁ z₂ : ℂ), z₁ = Complex.I ∧ z₂ = -3 - 2*Complex.I ∧
  (∀ z : ℂ, z^2 + 2*z = -3 + 4*Complex.I ↔ z = z₁ ∨ z = z₂) :=
by sorry

end NUMINAMATH_CALUDE_complex_quadratic_roots_l2516_251663


namespace NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l2516_251685

theorem conditional_probability_rain_given_east_wind
  (p_east_wind : ℚ)
  (p_rain : ℚ)
  (p_both : ℚ)
  (h1 : p_east_wind = 3 / 10)
  (h2 : p_rain = 11 / 30)
  (h3 : p_both = 8 / 30) :
  p_both / p_east_wind = 8 / 9 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l2516_251685


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2516_251686

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2516_251686


namespace NUMINAMATH_CALUDE_intersection_M_N_l2516_251689

def M : Set ℤ := {x | ∃ a : ℤ, x = a^2 + 1}
def N : Set ℤ := {y | 1 ≤ y ∧ y ≤ 6}

theorem intersection_M_N : M ∩ N = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2516_251689


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l2516_251615

theorem parallel_vectors_tan_theta (θ : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (a : Fin 2 → ℝ) (b : Fin 2 → ℝ)
  (h_a : a = ![1 - Real.sin θ, 1])
  (h_b : b = ![1 / 2, 1 + Real.sin θ])
  (h_parallel : ∃ (k : ℝ), a = k • b) :
  Real.tan θ = 1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l2516_251615


namespace NUMINAMATH_CALUDE_randy_pictures_l2516_251633

/-- Given that Peter drew 8 pictures, Quincy drew 20 more pictures than Peter,
    and they drew 41 pictures altogether, prove that Randy drew 5 pictures. -/
theorem randy_pictures (peter_pictures : ℕ) (quincy_pictures : ℕ) (total_pictures : ℕ) :
  peter_pictures = 8 →
  quincy_pictures = peter_pictures + 20 →
  total_pictures = 41 →
  total_pictures = peter_pictures + quincy_pictures + 5 := by
  sorry

end NUMINAMATH_CALUDE_randy_pictures_l2516_251633


namespace NUMINAMATH_CALUDE_sam_distance_l2516_251670

/-- Proves that Sam drove 200 miles given the conditions of the problem -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) : 
  (marguerite_distance / marguerite_time) * sam_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_l2516_251670


namespace NUMINAMATH_CALUDE_negative_a_exponent_division_l2516_251613

theorem negative_a_exponent_division (a : ℝ) : (-a)^6 / (-a)^3 = -a^3 := by sorry

end NUMINAMATH_CALUDE_negative_a_exponent_division_l2516_251613


namespace NUMINAMATH_CALUDE_floor_equality_l2516_251674

theorem floor_equality (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b > 1) :
  ⌊((a - b)^2 - 1 : ℚ) / (a * b)⌋ = ⌊((a - b)^2 - 1 : ℚ) / (a * b - 1)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_l2516_251674


namespace NUMINAMATH_CALUDE_equation_is_linear_one_var_l2516_251627

/-- Predicate to check if an expression is linear in one variable -/
def IsLinearOneVar (e : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, e x = a * x + b ∧ a ≠ 0

/-- The specific equation we're checking -/
def equation (x : ℝ) : ℝ := 3 - 2*x

/-- Theorem stating that our equation is linear in one variable -/
theorem equation_is_linear_one_var : IsLinearOneVar equation :=
sorry

end NUMINAMATH_CALUDE_equation_is_linear_one_var_l2516_251627


namespace NUMINAMATH_CALUDE_ratio_of_fifth_terms_in_arithmetic_sequences_l2516_251680

/-- Given two arithmetic sequences, prove the ratio of their 5th terms -/
theorem ratio_of_fifth_terms_in_arithmetic_sequences 
  (a b : ℕ → ℚ) 
  (h_arithmetic_a : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_arithmetic_b : ∀ n, b (n + 1) - b n = b 1 - b 0)
  (h_ratio : ∀ n, (n * (a 0 + a n)) / (n * (b 0 + b n)) = (3 * n) / (2 * n + 9)) :
  a 5 / b 5 = 15 / 19 := by
sorry


end NUMINAMATH_CALUDE_ratio_of_fifth_terms_in_arithmetic_sequences_l2516_251680


namespace NUMINAMATH_CALUDE_school_population_l2516_251600

theorem school_population (boys : ℕ) (girls : ℕ) : 
  (boys : ℚ) / girls = 8 / 5 → 
  boys = 128 → 
  boys + girls = 208 := by
sorry

end NUMINAMATH_CALUDE_school_population_l2516_251600


namespace NUMINAMATH_CALUDE_train_length_l2516_251602

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length (speed : Real) (time : Real) (bridge_length : Real) :
  speed = 10 → -- 36 kmph converted to m/s
  time = 29.997600191984642 →
  bridge_length = 150 →
  speed * time - bridge_length = 149.97600191984642 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2516_251602


namespace NUMINAMATH_CALUDE_problem_statement_l2516_251626

theorem problem_statement (x y z : ℝ) 
  (h1 : 2 * x - y - 2 * z - 6 = 0) 
  (h2 : x^2 + y^2 + z^2 ≤ 4) : 
  2 * x + y + z = 2/3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2516_251626


namespace NUMINAMATH_CALUDE_davids_math_marks_l2516_251617

theorem davids_math_marks (english physics chemistry biology average : ℕ) 
  (h1 : english = 86)
  (h2 : physics = 92)
  (h3 : chemistry = 87)
  (h4 : biology = 95)
  (h5 : average = 89)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_davids_math_marks_l2516_251617


namespace NUMINAMATH_CALUDE_davids_math_marks_l2516_251699

theorem davids_math_marks
  (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℚ)
  (h1 : english = 45)
  (h2 : physics = 52)
  (h3 : chemistry = 47)
  (h4 : biology = 55)
  (h5 : average = 46.8)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 35 := by
  sorry

end NUMINAMATH_CALUDE_davids_math_marks_l2516_251699


namespace NUMINAMATH_CALUDE_equivalent_operation_l2516_251601

theorem equivalent_operation (x : ℝ) : (x * (2/3)) / (5/6) = x * (4/5) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l2516_251601


namespace NUMINAMATH_CALUDE_min_value_expression_l2516_251610

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 2) :
  (4 / (x + 3 * y)) + (1 / (x - y)) ≥ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2516_251610


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2516_251651

theorem gcd_of_specific_numbers : Nat.gcd 123456789 987654321 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2516_251651


namespace NUMINAMATH_CALUDE_age_problem_solution_l2516_251637

/-- Represents the ages of two people --/
structure Ages where
  your_age : ℕ
  my_age : ℕ

/-- The conditions of the age problem --/
def age_conditions (ages : Ages) : Prop :=
  -- Condition 1: I am twice as old as you were when I was as old as you are now
  ages.your_age = 2 * (2 * ages.my_age - ages.your_age) ∧
  -- Condition 2: When you are as old as I am now, the sum of our ages will be 140 years
  ages.my_age + (2 * ages.my_age - ages.your_age) = 140

/-- The theorem stating the solution to the age problem --/
theorem age_problem_solution :
  ∃ (ages : Ages), age_conditions ages ∧ ages.your_age = 112 ∧ ages.my_age = 84 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_solution_l2516_251637


namespace NUMINAMATH_CALUDE_digit_sum_equation_l2516_251643

-- Define the digits as natural numbers
def X : ℕ := sorry
def Y : ℕ := sorry
def M : ℕ := sorry
def Z : ℕ := sorry
def F : ℕ := sorry

-- Define the two-digit numbers
def XY : ℕ := 10 * X + Y
def MZ : ℕ := 10 * M + Z

-- Define the three-digit number FFF
def FFF : ℕ := 100 * F + 10 * F + F

-- Theorem statement
theorem digit_sum_equation : 
  (X ≠ 0) ∧ (Y ≠ 0) ∧ (M ≠ 0) ∧ (Z ≠ 0) ∧ (F ≠ 0) ∧  -- non-zero digits
  (X ≠ Y) ∧ (X ≠ M) ∧ (X ≠ Z) ∧ (X ≠ F) ∧
  (Y ≠ M) ∧ (Y ≠ Z) ∧ (Y ≠ F) ∧
  (M ≠ Z) ∧ (M ≠ F) ∧
  (Z ≠ F) ∧  -- unique digits
  (X < 10) ∧ (Y < 10) ∧ (M < 10) ∧ (Z < 10) ∧ (F < 10) ∧  -- single digits
  (XY * MZ = FFF) →  -- equation condition
  X + Y + M + Z + F = 28 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_equation_l2516_251643


namespace NUMINAMATH_CALUDE_line_CD_passes_through_fixed_point_l2516_251666

-- Define the Cartesian plane
variable (x y : ℝ)

-- Define points E and F
def E : ℝ × ℝ := (0, 1)
def F : ℝ × ℝ := (0, -1)

-- Define the trajectory of point G
def trajectory (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define line l
def line_l (x : ℝ) : Prop := x = 4

-- Define point P on line l
def P (y₀ : ℝ) : ℝ × ℝ := (4, y₀)

-- Define points A and B (vertices of the trajectory)
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem line_CD_passes_through_fixed_point (x y y₀ : ℝ) 
  (h1 : trajectory x y) 
  (h2 : y₀ ≠ 0) :
  ∃ (xc yc xd yd : ℝ), 
    trajectory xc yc ∧ 
    trajectory xd yd ∧ 
    (yc - yd) / (xc - xd) * (1 - xc) + yc = 0 := 
sorry


end NUMINAMATH_CALUDE_line_CD_passes_through_fixed_point_l2516_251666


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l2516_251625

theorem triangle_angle_ratio (A B C : ℝ) (x : ℝ) : 
  B = x * A →
  C = A + 12 →
  A = 24 →
  A + B + C = 180 →
  x = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l2516_251625


namespace NUMINAMATH_CALUDE_inequality_proof_l2516_251675

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (4 / (c - a)) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2516_251675


namespace NUMINAMATH_CALUDE_broker_investment_l2516_251660

theorem broker_investment (P : ℝ) (x : ℝ) (h : P > 0) :
  (P + x / 100 * P) * (1 - 30 / 100) = P * (1 + 26 / 100) →
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_broker_investment_l2516_251660


namespace NUMINAMATH_CALUDE_product_mod_seven_l2516_251644

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026 * 2027) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2516_251644


namespace NUMINAMATH_CALUDE_root_of_two_quadratics_l2516_251682

theorem root_of_two_quadratics (a b c d : ℂ) (k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hk1 : a * k^2 + b * k + c = 0)
  (hk2 : b * k^2 + c * k + d = 0) :
  k = 1 ∨ k = (-1 + Complex.I * Real.sqrt 3) / 2 ∨ k = (-1 - Complex.I * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_root_of_two_quadratics_l2516_251682


namespace NUMINAMATH_CALUDE_count_divisors_3240_multiple_of_three_l2516_251641

/-- The number of positive divisors of 3240 that are multiples of 3 -/
def num_divisors_multiple_of_three : ℕ := 32

/-- The prime factorization of 3240 -/
def factorization_3240 : List (ℕ × ℕ) := [(2, 3), (3, 4), (5, 1)]

/-- A function to count the number of positive divisors of 3240 that are multiples of 3 -/
def count_divisors_multiple_of_three (factorization : List (ℕ × ℕ)) : ℕ :=
  sorry

theorem count_divisors_3240_multiple_of_three :
  count_divisors_multiple_of_three factorization_3240 = num_divisors_multiple_of_three :=
sorry

end NUMINAMATH_CALUDE_count_divisors_3240_multiple_of_three_l2516_251641


namespace NUMINAMATH_CALUDE_chads_cat_food_packages_l2516_251664

/-- Chad's pet food purchase problem -/
theorem chads_cat_food_packages :
  ∀ (c : ℕ), -- c represents the number of packages of cat food
  (9 * c = 2 * 3 + 48) → -- Equation representing the difference in cans
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_chads_cat_food_packages_l2516_251664


namespace NUMINAMATH_CALUDE_same_point_on_bisector_l2516_251688

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The angle bisector of the first and third quadrants -/
def firstThirdQuadrantBisector : Set Point2D :=
  {p : Point2D | p.x = p.y}

/-- Theorem: If A(a, b) and B(b, a) represent the same point, 
    then this point lies on the angle bisector of the first and third quadrants -/
theorem same_point_on_bisector (a b : ℝ) :
  Point2D.mk a b = Point2D.mk b a → 
  Point2D.mk a b ∈ firstThirdQuadrantBisector := by
  sorry

end NUMINAMATH_CALUDE_same_point_on_bisector_l2516_251688


namespace NUMINAMATH_CALUDE_equation_solution_l2516_251648

theorem equation_solution :
  {x : ℂ | x^4 - 81 = 0} = {3, -3, 3*I, -3*I} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2516_251648


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2516_251621

/-- Represents a right circular cylinder inscribed in a right circular cone. -/
structure InscribedCylinder where
  /-- Radius of the inscribed cylinder -/
  radius : ℝ
  /-- Height of the inscribed cylinder -/
  height : ℝ
  /-- Diameter of the cone -/
  cone_diameter : ℝ
  /-- Altitude of the cone -/
  cone_altitude : ℝ
  /-- The cylinder's diameter is equal to its height -/
  cylinder_property : height = 2 * radius
  /-- The cone has a diameter of 20 -/
  cone_diameter_value : cone_diameter = 20
  /-- The cone has an altitude of 24 -/
  cone_altitude_value : cone_altitude = 24

/-- Theorem stating that the radius of the inscribed cylinder is 60/11 -/
theorem inscribed_cylinder_radius (c : InscribedCylinder) : c.radius = 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2516_251621


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l2516_251634

/-- Represents the probability of drawing specific colored balls from an urn --/
def draw_probability (red white green total : ℕ) : ℚ :=
  (red : ℚ) / total * (white : ℚ) / (total - 1) * (green : ℚ) / (total - 2)

/-- Represents the probability of drawing specific colored balls in any order --/
def draw_probability_any_order (red white green total : ℕ) : ℚ :=
  6 * draw_probability red white green total

theorem urn_probability_theorem (red white green : ℕ) 
  (h_red : red = 15) (h_white : white = 9) (h_green : green = 4) :
  let total := red + white + green
  draw_probability red white green total = 5 / 182 ∧
  draw_probability_any_order red white green total = 15 / 91 := by
  sorry


end NUMINAMATH_CALUDE_urn_probability_theorem_l2516_251634


namespace NUMINAMATH_CALUDE_abs_c_value_l2516_251691

def f (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem abs_c_value (a b c : ℤ) (h1 : Int.gcd a (Int.gcd b c) = 1) 
  (h2 : f a b c (2 + Complex.I) = 0) : 
  Int.natAbs c = 42 := by
  sorry

end NUMINAMATH_CALUDE_abs_c_value_l2516_251691


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2516_251687

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - 2 * m * x - 3 < 0) ↔ -3 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2516_251687


namespace NUMINAMATH_CALUDE_factorial_sum_division_l2516_251661

theorem factorial_sum_division : (Nat.factorial 8 + Nat.factorial 9) / Nat.factorial 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_division_l2516_251661


namespace NUMINAMATH_CALUDE_discount_calculation_l2516_251646

theorem discount_calculation (CP : ℝ) (CP_pos : CP > 0) : 
  let MP := 1.12 * CP
  let SP := 0.99 * CP
  MP - SP = 0.13 * CP := by sorry

end NUMINAMATH_CALUDE_discount_calculation_l2516_251646


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l2516_251695

theorem sum_of_squares_theorem (x y z a b c k : ℝ) 
  (h1 : x * y = k * a) 
  (h2 : x * z = k * b) 
  (h3 : y * z = k * c) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hk : k ≠ 0) : 
  x^2 + y^2 + z^2 = k * (a * b / c + a * c / b + b * c / a) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l2516_251695


namespace NUMINAMATH_CALUDE_max_value_theorem_l2516_251642

theorem max_value_theorem (x : ℝ) (h : x < -3) :
  x + 2 / (x + 3) ≤ -2 * Real.sqrt 2 - 3 ∧
  ∃ y, y < -3 ∧ y + 2 / (y + 3) = -2 * Real.sqrt 2 - 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2516_251642


namespace NUMINAMATH_CALUDE_scallop_cost_per_pound_l2516_251604

/-- The cost per pound of jumbo scallops -/
def cost_per_pound (scallops_per_pound : ℕ) (num_people : ℕ) (scallops_per_person : ℕ) (total_cost : ℚ) : ℚ :=
  let total_scallops := num_people * scallops_per_person
  let pounds_needed := total_scallops / scallops_per_pound
  total_cost / pounds_needed

/-- Theorem stating the cost per pound of jumbo scallops is $24 -/
theorem scallop_cost_per_pound :
  cost_per_pound 8 8 2 48 = 24 := by
  sorry

end NUMINAMATH_CALUDE_scallop_cost_per_pound_l2516_251604


namespace NUMINAMATH_CALUDE_abs_sum_greater_than_one_necessary_not_sufficient_l2516_251665

theorem abs_sum_greater_than_one_necessary_not_sufficient (a b : ℝ) :
  (∀ b, b < -1 → ∀ a, |a| + |b| > 1) ∧
  (∃ a b, |a| + |b| > 1 ∧ b ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_greater_than_one_necessary_not_sufficient_l2516_251665


namespace NUMINAMATH_CALUDE_min_m_plus_n_l2516_251609

theorem min_m_plus_n (m n : ℕ+) (h : 75 * m = n^3) : 
  ∀ (m' n' : ℕ+), 75 * m' = n'^3 → m + n ≤ m' + n' :=
by sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l2516_251609


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2516_251676

theorem quadratic_inequality_range (p : ℝ) (α : ℝ) (h1 : p = 4 * (Real.sin α) ^ 4)
  (h2 : α ∈ Set.Icc (π / 6) (5 * π / 6)) :
  (∀ x : ℝ, x^2 + p*x + 1 > 2*x + p) ↔ (∀ x : ℝ, x > 1 ∨ x < -3) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2516_251676


namespace NUMINAMATH_CALUDE_quadrilaterals_in_100gon_l2516_251679

/-- A regular polygon with n vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A coloring of vertices of a regular polygon -/
def Coloring (n : ℕ) := Fin n → Bool

/-- A convex quadrilateral formed by four vertices of a regular polygon -/
structure Quadrilateral (n : ℕ) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n
  v4 : Fin n

/-- Check if two quadrilaterals are disjoint -/
def are_disjoint (n : ℕ) (q1 q2 : Quadrilateral n) : Prop :=
  q1.v1 ≠ q2.v1 ∧ q1.v1 ≠ q2.v2 ∧ q1.v1 ≠ q2.v3 ∧ q1.v1 ≠ q2.v4 ∧
  q1.v2 ≠ q2.v1 ∧ q1.v2 ≠ q2.v2 ∧ q1.v2 ≠ q2.v3 ∧ q1.v2 ≠ q2.v4 ∧
  q1.v3 ≠ q2.v1 ∧ q1.v3 ≠ q2.v2 ∧ q1.v3 ≠ q2.v3 ∧ q1.v3 ≠ q2.v4 ∧
  q1.v4 ≠ q2.v1 ∧ q1.v4 ≠ q2.v2 ∧ q1.v4 ≠ q2.v3 ∧ q1.v4 ≠ q2.v4

/-- Check if a quadrilateral has three corners of one color and one of the other -/
def has_three_one_coloring (n : ℕ) (q : Quadrilateral n) (c : Coloring n) : Prop :=
  (c q.v1 = c q.v2 ∧ c q.v2 = c q.v3 ∧ c q.v3 ≠ c q.v4) ∨
  (c q.v1 = c q.v2 ∧ c q.v2 = c q.v4 ∧ c q.v4 ≠ c q.v3) ∨
  (c q.v1 = c q.v3 ∧ c q.v3 = c q.v4 ∧ c q.v4 ≠ c q.v2) ∨
  (c q.v2 = c q.v3 ∧ c q.v3 = c q.v4 ∧ c q.v4 ≠ c q.v1)

/-- The main theorem -/
theorem quadrilaterals_in_100gon :
  ∃ (p : RegularPolygon 100) (c : Coloring 100) (qs : Fin 24 → Quadrilateral 100),
    (∀ i : Fin 100, c i = true → (∃ j : Fin 41, true)) ∧  -- 41 black vertices
    (∀ i : Fin 100, c i = false → (∃ j : Fin 59, true)) ∧  -- 59 white vertices
    (∀ i j : Fin 24, i ≠ j → are_disjoint 100 (qs i) (qs j)) ∧
    (∀ i : Fin 24, has_three_one_coloring 100 (qs i) c) :=
by sorry

end NUMINAMATH_CALUDE_quadrilaterals_in_100gon_l2516_251679


namespace NUMINAMATH_CALUDE_sqrt_16_divided_by_2_l2516_251673

theorem sqrt_16_divided_by_2 : Real.sqrt 16 / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_divided_by_2_l2516_251673


namespace NUMINAMATH_CALUDE_quadrilateral_sine_equality_l2516_251667

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)  -- Interior angles
  (AB BC CD DA : ℝ)  -- Side lengths

-- Define the property of being convex
def is_convex (q : Quadrilateral) : Prop :=
  q.A > 0 ∧ q.B > 0 ∧ q.C > 0 ∧ q.D > 0 ∧ q.A + q.B + q.C + q.D = 2 * Real.pi

-- State the theorem
theorem quadrilateral_sine_equality (q : Quadrilateral) (h : is_convex q) :
  (Real.sin q.A) / (q.BC * q.CD) + (Real.sin q.C) / (q.DA * q.AB) =
  (Real.sin q.B) / (q.CD * q.DA) + (Real.sin q.D) / (q.AB * q.BC) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_sine_equality_l2516_251667


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l2516_251656

-- Define the repeating decimal 0.333...
def repeating_third : ℚ := 1 / 3

-- State the theorem
theorem eight_divided_by_repeating_third : 8 / repeating_third = 24 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l2516_251656


namespace NUMINAMATH_CALUDE_cos_sum_thirteenth_l2516_251658

theorem cos_sum_thirteenth : 
  Real.cos (3 * Real.pi / 13) + Real.cos (5 * Real.pi / 13) + Real.cos (7 * Real.pi / 13) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_thirteenth_l2516_251658


namespace NUMINAMATH_CALUDE_f_properties_l2516_251692

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x / 2) * cos (x / 2) - 2 * Real.sqrt 3 * sin (x / 2) ^ 2 + Real.sqrt 3

theorem f_properties (α : ℝ) (h1 : α ∈ Set.Ioo (π / 6) (2 * π / 3)) (h2 : f α = 6 / 5) :
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (2 * k * π + π / 6) (2 * k * π + 7 * π / 6))) ∧
  f (α - π / 6) = (4 + 3 * Real.sqrt 3) / 5 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2516_251692


namespace NUMINAMATH_CALUDE_park_visit_cost_l2516_251608

def park_entrance_fee : ℕ := 5
def attraction_fee_child : ℕ := 2
def attraction_fee_adult : ℕ := 4
def num_children : ℕ := 4
def num_adults : ℕ := 3
def total_people : ℕ := num_children + num_adults

def total_cost : ℕ :=
  park_entrance_fee * total_people +
  attraction_fee_child * num_children +
  attraction_fee_adult * num_adults

theorem park_visit_cost :
  total_cost = 55 := by sorry

end NUMINAMATH_CALUDE_park_visit_cost_l2516_251608


namespace NUMINAMATH_CALUDE_troy_needs_ten_more_l2516_251606

def new_computer_cost : ℕ := 80
def initial_savings : ℕ := 50
def old_computer_value : ℕ := 20

theorem troy_needs_ten_more :
  new_computer_cost - (initial_savings + old_computer_value) = 10 := by
  sorry

end NUMINAMATH_CALUDE_troy_needs_ten_more_l2516_251606


namespace NUMINAMATH_CALUDE_percentage_increase_l2516_251603

theorem percentage_increase (x : ℝ) (base : ℝ) (percentage : ℝ) : 
  x = base + (percentage / 100) * base →
  x = 110 →
  base = 88 →
  percentage = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l2516_251603


namespace NUMINAMATH_CALUDE_house_work_payment_l2516_251611

/-- Represents the total payment for work on a house --/
def total_payment (bricklayer_rate : ℝ) (electrician_rate : ℝ) (hours_worked : ℝ) : ℝ :=
  bricklayer_rate * hours_worked + electrician_rate * hours_worked

/-- Proves that the total payment for the work is $630 --/
theorem house_work_payment : 
  let bricklayer_rate : ℝ := 12
  let electrician_rate : ℝ := 16
  let hours_worked : ℝ := 22.5
  total_payment bricklayer_rate electrician_rate hours_worked = 630 := by
  sorry

#eval total_payment 12 16 22.5

end NUMINAMATH_CALUDE_house_work_payment_l2516_251611


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2516_251649

theorem child_ticket_cost (adult_price : ℕ) (num_adults num_children : ℕ) (total_price : ℕ) :
  adult_price = 22 →
  num_adults = 2 →
  num_children = 2 →
  total_price = 58 →
  (total_price - num_adults * adult_price) / num_children = 7 := by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2516_251649


namespace NUMINAMATH_CALUDE_probability_20th_to_30th_l2516_251694

/-- A sequence of 40 distinct real numbers -/
def Sequence := Fin 40 → ℝ

/-- Predicate to check if a sequence contains distinct elements -/
def IsDistinct (s : Sequence) : Prop :=
  ∀ i j, i ≠ j → s i ≠ s j

/-- The probability that the 20th number ends up in the 30th position after one bubble pass -/
def ProbabilityOf20thTo30th (s : Sequence) : ℚ :=
  1 / 930

/-- Theorem stating the probability of the 20th number ending up in the 30th position -/
theorem probability_20th_to_30th (s : Sequence) (h : IsDistinct s) :
    ProbabilityOf20thTo30th s = 1 / 930 := by
  sorry

end NUMINAMATH_CALUDE_probability_20th_to_30th_l2516_251694


namespace NUMINAMATH_CALUDE_proof_by_contradiction_on_incorrect_statement_l2516_251671

-- Define a proposition
variable (P : Prop)

-- Define the property of being an incorrect statement
def is_incorrect (S : Prop) : Prop := ¬S

-- Define the process of attempting proof by contradiction
def attempt_proof_by_contradiction (S : Prop) : Prop :=
  ∃ (proof : ¬S → False), True

-- Define what it means for a proof method to fail to produce a useful conclusion
def fails_to_produce_useful_conclusion (S : Prop) : Prop :=
  ¬(S ∨ ¬S)

-- Theorem statement
theorem proof_by_contradiction_on_incorrect_statement
  (h : is_incorrect P) :
  attempt_proof_by_contradiction P →
  fails_to_produce_useful_conclusion P :=
by sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_on_incorrect_statement_l2516_251671


namespace NUMINAMATH_CALUDE_matematika_arrangements_l2516_251683

/-- The number of distinct letters in "MATEMATIKA" excluding "A" -/
def n : ℕ := 7

/-- The number of repeated letters (M and T) -/
def r : ℕ := 2

/-- The number of "A"s in "MATEMATIKA" -/
def a : ℕ := 3

/-- The number of positions to place "A"s -/
def p : ℕ := n + 1

theorem matematika_arrangements : 
  (n.factorial / (r.factorial * r.factorial)) * Nat.choose p a = 70560 := by
  sorry

end NUMINAMATH_CALUDE_matematika_arrangements_l2516_251683


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2516_251654

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (9 + 3 * z) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2516_251654


namespace NUMINAMATH_CALUDE_apple_distribution_l2516_251630

theorem apple_distribution (total_apples : ℕ) (num_people : ℕ) (min_apples : ℕ) :
  total_apples = 30 →
  num_people = 3 →
  min_apples = 3 →
  (Nat.choose (total_apples - num_people * min_apples + num_people - 1) (num_people - 1)) = 253 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l2516_251630


namespace NUMINAMATH_CALUDE_custom_mult_solution_l2516_251607

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := 2 * a - b^2

/-- Theorem stating that if a * 3 = 15 under the custom multiplication, then a = 12 -/
theorem custom_mult_solution :
  ∀ a : ℝ, custom_mult a 3 = 15 → a = 12 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_solution_l2516_251607


namespace NUMINAMATH_CALUDE_max_value_of_f_l2516_251639

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2516_251639


namespace NUMINAMATH_CALUDE_belle_weekly_treat_cost_l2516_251624

/-- The cost to feed Belle treats for a week -/
def weekly_treat_cost (dog_biscuits_per_day : ℕ) (rawhide_bones_per_day : ℕ) 
  (dog_biscuit_cost : ℚ) (rawhide_bone_cost : ℚ) (days_per_week : ℕ) : ℚ :=
  (dog_biscuits_per_day * dog_biscuit_cost + rawhide_bones_per_day * rawhide_bone_cost) * days_per_week

/-- Proof that Belle's weekly treat cost is $21.00 -/
theorem belle_weekly_treat_cost :
  weekly_treat_cost 4 2 0.25 1 7 = 21 := by
  sorry

#eval weekly_treat_cost 4 2 (1/4) 1 7

end NUMINAMATH_CALUDE_belle_weekly_treat_cost_l2516_251624
