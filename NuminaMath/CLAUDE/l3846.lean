import Mathlib

namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3846_384640

theorem gcd_lcm_sum : Nat.gcd 42 98 + Nat.lcm 60 15 = 74 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3846_384640


namespace NUMINAMATH_CALUDE_exists_valid_cylinder_arrangement_l3846_384610

/-- Represents a straight circular cylinder in 3D space -/
structure Cylinder where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  height : ℝ

/-- Checks if two cylinders have a common boundary point -/
def havesCommonPoint (c1 c2 : Cylinder) : Prop := sorry

/-- Represents an arrangement of six cylinders -/
def CylinderArrangement := Fin 6 → Cylinder

/-- Checks if a given arrangement satisfies the condition that each cylinder
    has a common point with every other cylinder -/
def isValidArrangement (arr : CylinderArrangement) : Prop :=
  ∀ i j, i ≠ j → havesCommonPoint (arr i) (arr j)

/-- The main theorem stating that there exists a valid arrangement of six cylinders -/
theorem exists_valid_cylinder_arrangement :
  ∃ (arr : CylinderArrangement), isValidArrangement arr := by sorry

end NUMINAMATH_CALUDE_exists_valid_cylinder_arrangement_l3846_384610


namespace NUMINAMATH_CALUDE_systematic_sampling_l3846_384695

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  groups : Nat
  first_group_number : Nat
  sixteenth_group_number : Nat

/-- The systematic sampling theorem -/
theorem systematic_sampling
  (s : SystematicSample)
  (h1 : s.total_students = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.groups = 20)
  (h4 : s.sixteenth_group_number = 126) :
  s.first_group_number = 6 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_l3846_384695


namespace NUMINAMATH_CALUDE_valid_numbers_l3846_384617

def isValidNumber (n : ℕ) : Prop :=
  n ≥ 500 ∧ n < 1000 ∧
  (n / 100 % 2 = 1) ∧
  ((n / 10) % 10 % 2 = 0) ∧
  (n % 10 % 2 = 0) ∧
  (n / 100 % 3 = 0) ∧
  ((n / 10) % 10 % 3 = 0) ∧
  (n % 10 % 3 ≠ 0)

theorem valid_numbers :
  {n : ℕ | isValidNumber n} = {902, 904, 908, 962, 964, 968} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3846_384617


namespace NUMINAMATH_CALUDE_percentage_problem_l3846_384661

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 2 * x) : x = 200 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3846_384661


namespace NUMINAMATH_CALUDE_rhombus_area_l3846_384694

/-- The area of a rhombus with diagonals satisfying a specific equation --/
theorem rhombus_area (a b : ℝ) (h : (a - 1)^2 + Real.sqrt (b - 4) = 0) :
  (1/2 : ℝ) * a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3846_384694


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3846_384667

theorem max_candy_leftover (x : ℕ) : ∃ (q : ℕ), x = 11 * q + 10 ∧ ∀ (r : ℕ), r < 11 → x ≠ 11 * q + r + 1 :=
sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3846_384667


namespace NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l3846_384602

theorem consecutive_integer_product_divisibility (j : ℤ) : 
  let m := j * (j + 1) * (j + 2) * (j + 3)
  (∃ k : ℤ, m = 11 * k) →
  (∃ k : ℤ, m = 12 * k) ∧
  (∃ k : ℤ, m = 33 * k) ∧
  (∃ k : ℤ, m = 44 * k) ∧
  (∃ k : ℤ, m = 66 * k) ∧
  ¬(∀ j : ℤ, ∃ k : ℤ, m = 24 * k) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l3846_384602


namespace NUMINAMATH_CALUDE_book_pages_l3846_384649

/-- Given a book where the total number of digits used in numbering its pages is 930,
    prove that the book has 346 pages. -/
theorem book_pages (total_digits : ℕ) (h : total_digits = 930) : ∃ (pages : ℕ), pages = 346 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l3846_384649


namespace NUMINAMATH_CALUDE_football_games_this_year_l3846_384684

theorem football_games_this_year 
  (total_games : ℕ) 
  (last_year_games : ℕ) 
  (h1 : total_games = 9)
  (h2 : last_year_games = 5) :
  total_games - last_year_games = 4 :=
by sorry

end NUMINAMATH_CALUDE_football_games_this_year_l3846_384684


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l3846_384619

-- Define the functions f and g
def f (A B C x : ℝ) : ℝ := A * x + B + C
def g (A B C x : ℝ) : ℝ := B * x + A - C

-- State the theorem
theorem sum_of_coefficients_is_zero 
  (A B C : ℝ) 
  (h1 : A ≠ B) 
  (h2 : C ≠ 0) 
  (h3 : ∀ x, f A B C (g A B C x) - g A B C (f A B C x) = 2 * C) : 
  A + B = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l3846_384619


namespace NUMINAMATH_CALUDE_root_equation_r_value_l3846_384603

theorem root_equation_r_value (a b m p r : ℝ) : 
  (a^2 - m*a + 6 = 0) → 
  (b^2 - m*b + 6 = 0) → 
  ((a + 2/b)^2 - p*(a + 2/b) + r = 0) → 
  ((b + 2/a)^2 - p*(b + 2/a) + r = 0) → 
  r = 32/3 := by sorry

end NUMINAMATH_CALUDE_root_equation_r_value_l3846_384603


namespace NUMINAMATH_CALUDE_base_conversion_equality_l3846_384632

/-- Given that 10b1₍₂₎ = a02₍₃₎, b ∈ {0, 1}, and a ∈ {0, 1, 2}, prove that a = 1 and b = 1 -/
theorem base_conversion_equality (a b : ℕ) : 
  (1 + 2 * b + 8 = 2 + 9 * a) → 
  (b = 0 ∨ b = 1) → 
  (a = 0 ∨ a = 1 ∨ a = 2) → 
  (a = 1 ∧ b = 1) := by
sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l3846_384632


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_attained_l3846_384693

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 4*x₂ + 6*x₃ = 120) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 350 := by
  sorry

theorem min_sum_squares_attained (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 4*x₂ + 6*x₃ = 120) : 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 2*y₁ + 4*y₂ + 6*y₃ = 120 ∧ 
  y₁^2 + y₂^2 + y₃^2 = 350 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_attained_l3846_384693


namespace NUMINAMATH_CALUDE_exponential_function_property_l3846_384606

theorem exponential_function_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 1 2, a^x ≤ a^2) ∧ 
  (∀ x ∈ Set.Icc 1 2, a^x ≥ a^1) ∧
  (a^2 - a^1 = a / 2) →
  a = 1/2 ∨ a = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l3846_384606


namespace NUMINAMATH_CALUDE_num_monomials_for_problem_exponent_l3846_384692

/-- The number of monomials with nonzero coefficients in the expansion of (x+y+z)^n + (x-y-z)^n -/
def num_monomials (n : ℕ) : ℕ :=
  (n / 2 + 1) ^ 2

/-- The given exponent in the problem -/
def problem_exponent : ℕ := 2032

theorem num_monomials_for_problem_exponent :
  num_monomials problem_exponent = 1034289 := by
  sorry

end NUMINAMATH_CALUDE_num_monomials_for_problem_exponent_l3846_384692


namespace NUMINAMATH_CALUDE_trig_identity_l3846_384690

theorem trig_identity (α : Real) (h : Real.cos α ^ 2 = Real.sin α) :
  1 / Real.sin α + Real.cos α ^ 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3846_384690


namespace NUMINAMATH_CALUDE_faye_coloring_books_faye_coloring_books_proof_l3846_384679

theorem faye_coloring_books : ℕ :=
  let initial : ℕ := sorry
  let given_away : ℕ := 3
  let bought : ℕ := 48
  let final : ℕ := 79
  have h : initial - given_away + bought = final := by sorry
  initial

-- The proof
theorem faye_coloring_books_proof : faye_coloring_books = 34 := by sorry

end NUMINAMATH_CALUDE_faye_coloring_books_faye_coloring_books_proof_l3846_384679


namespace NUMINAMATH_CALUDE_deer_families_stayed_l3846_384614

theorem deer_families_stayed (total : ℕ) (moved_out : ℕ) (h1 : total = 79) (h2 : moved_out = 34) :
  total - moved_out = 45 := by
  sorry

end NUMINAMATH_CALUDE_deer_families_stayed_l3846_384614


namespace NUMINAMATH_CALUDE_smallest_multiple_l3846_384659

theorem smallest_multiple (x : ℕ) : x = 32 ↔ 
  (x > 0 ∧ 900 * x % 1152 = 0 ∧ ∀ y : ℕ, (y > 0 ∧ y < x) → 900 * y % 1152 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3846_384659


namespace NUMINAMATH_CALUDE_larger_number_is_fifty_l3846_384611

theorem larger_number_is_fifty (a b : ℝ) : 
  (4 * b = 5 * a) → (b - a = 10) → b = 50 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_fifty_l3846_384611


namespace NUMINAMATH_CALUDE_correct_average_l3846_384624

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 46 ∧ incorrect_num = 25 ∧ correct_num = 75 →
  (n * initial_avg + (correct_num - incorrect_num)) / n = 51 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l3846_384624


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3846_384629

theorem sum_of_x_and_y (a x y : ℝ) (hx : a / x = 1 / 3) (hy : a / y = 1 / 4) :
  x + y = 7 * a := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3846_384629


namespace NUMINAMATH_CALUDE_real_part_of_z_l3846_384604

theorem real_part_of_z (i : ℂ) (h : i * i = -1) :
  (i * (1 - 2 * i)).re = 2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3846_384604


namespace NUMINAMATH_CALUDE_root_sum_squares_l3846_384686

theorem root_sum_squares (r s : ℝ) (α β γ δ : ℂ) : 
  (α^2 - r*α - 2 = 0) → 
  (β^2 - r*β - 2 = 0) → 
  (γ^2 + s*γ - 2 = 0) → 
  (δ^2 + s*δ - 2 = 0) → 
  (α - γ)^2 + (β - γ)^2 + (α + δ)^2 + (β + δ)^2 = 4*s*(r - s) + 8 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_squares_l3846_384686


namespace NUMINAMATH_CALUDE_commercial_time_l3846_384645

theorem commercial_time (p : ℝ) (h : p = 0.9) : (1 - p) * 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_commercial_time_l3846_384645


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l3846_384647

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l3846_384647


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_four_l3846_384608

theorem angle_sum_is_pi_over_four (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan α = 1/7 →
  Real.sin β = Real.sqrt 10/10 →
  α + 2*β = π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_four_l3846_384608


namespace NUMINAMATH_CALUDE_galaxy_planets_l3846_384635

theorem galaxy_planets (total : ℕ) (ratio : ℕ) (h1 : total = 200) (h2 : ratio = 8) : 
  ∃ (planets : ℕ), planets * (ratio + 1) = total ∧ planets = 22 := by
  sorry

end NUMINAMATH_CALUDE_galaxy_planets_l3846_384635


namespace NUMINAMATH_CALUDE_sixth_term_value_l3846_384618

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem sixth_term_value (seq : ArithmeticSequence) 
    (first_term : seq.a 1 = 1)
    (sum_5 : seq.S 5 = 15) :
  seq.a 6 = 6 := by
  sorry


end NUMINAMATH_CALUDE_sixth_term_value_l3846_384618


namespace NUMINAMATH_CALUDE_y_value_l3846_384688

theorem y_value (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = -5) : y = 28 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3846_384688


namespace NUMINAMATH_CALUDE_rectangle_area_l3846_384682

theorem rectangle_area (width length : ℝ) (h1 : length = width + 6) (h2 : 2 * (width + length) = 68) :
  width * length = 280 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3846_384682


namespace NUMINAMATH_CALUDE_infinite_power_tower_four_implies_sqrt_two_l3846_384633

/-- The limit of the infinite power tower x^(x^(x^...)) -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ :=
  Real.log x / Real.log (Real.log x)

/-- Theorem: If the infinite power tower of x equals 4, then x equals √2 -/
theorem infinite_power_tower_four_implies_sqrt_two (x : ℝ) 
  (h_pos : x > 0) 
  (h_converge : infinitePowerTower x = 4) : 
  x = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_infinite_power_tower_four_implies_sqrt_two_l3846_384633


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l3846_384683

theorem sqrt_sum_equality : Real.sqrt 50 + Real.sqrt 72 = 11 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l3846_384683


namespace NUMINAMATH_CALUDE_minervas_stamps_l3846_384620

/-- Given that Lizette has 813 stamps and 125 more stamps than Minerva,
    prove that Minerva has 688 stamps. -/
theorem minervas_stamps :
  let lizette_stamps : ℕ := 813
  let difference : ℕ := 125
  let minerva_stamps : ℕ := lizette_stamps - difference
  minerva_stamps = 688 := by
sorry

end NUMINAMATH_CALUDE_minervas_stamps_l3846_384620


namespace NUMINAMATH_CALUDE_tech_club_theorem_l3846_384670

/-- The number of students in the tech club who take neither coding nor robotics -/
def students_taking_neither (total : ℕ) (coding : ℕ) (robotics : ℕ) (both : ℕ) : ℕ :=
  total - (coding + robotics - both)

/-- Theorem: Given the conditions from the problem, 20 students take neither coding nor robotics -/
theorem tech_club_theorem :
  students_taking_neither 150 80 70 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tech_club_theorem_l3846_384670


namespace NUMINAMATH_CALUDE_blackjack_bet_l3846_384689

theorem blackjack_bet (payout_ratio : Rat) (received_amount : ℚ) (original_bet : ℚ) : 
  payout_ratio = 3/2 →
  received_amount = 60 →
  received_amount = payout_ratio * original_bet →
  original_bet = 40 := by
sorry

end NUMINAMATH_CALUDE_blackjack_bet_l3846_384689


namespace NUMINAMATH_CALUDE_tim_running_hours_l3846_384616

def days_per_week : ℕ := 7

def previous_running_days : ℕ := 3
def added_running_days : ℕ := 2
def hours_per_run : ℕ := 2

def total_running_days : ℕ := previous_running_days + added_running_days
def total_running_hours : ℕ := total_running_days * hours_per_run

theorem tim_running_hours : total_running_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_tim_running_hours_l3846_384616


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3846_384609

theorem tan_alpha_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan β = 1/3) : 
  Real.tan (α + π/4) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3846_384609


namespace NUMINAMATH_CALUDE_min_value_function_l3846_384656

theorem min_value_function (x : ℝ) (h : x > -1) :
  (x^2 + 3*x + 4) / (x + 1) ≥ 2*Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l3846_384656


namespace NUMINAMATH_CALUDE_max_value_of_g_l3846_384641

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) : ℝ := 4 * x - x^4

/-- The interval [0, 2] -/
def I : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

theorem max_value_of_g :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), x ∈ I → g x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3846_384641


namespace NUMINAMATH_CALUDE_sine_value_from_tangent_cosine_relation_l3846_384638

theorem sine_value_from_tangent_cosine_relation (θ : Real) 
  (h1 : 8 * Real.tan θ = 3 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sine_value_from_tangent_cosine_relation_l3846_384638


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_3_area_is_three_sqrt_three_over_two_l3846_384628

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 + t.b * t.c

-- Theorem 1
theorem angle_A_is_pi_over_3 (t : Triangle) (h : satisfiesCondition t) :
  t.A = π / 3 := by sorry

-- Theorem 2
theorem area_is_three_sqrt_three_over_two (t : Triangle) 
  (h1 : satisfiesCondition t) (h2 : t.a = Real.sqrt 7) (h3 : t.b = 2) :
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_3_area_is_three_sqrt_three_over_two_l3846_384628


namespace NUMINAMATH_CALUDE_line_angle_of_inclination_l3846_384627

/-- The angle of inclination of the line 2x + 2y - 5 = 0 is 135° -/
theorem line_angle_of_inclination :
  let line := {(x, y) : ℝ × ℝ | 2*x + 2*y - 5 = 0}
  ∃ α : Real, α = 135 * (π / 180) ∧ 
    ∀ (x y : ℝ), (x, y) ∈ line → (Real.tan α = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_angle_of_inclination_l3846_384627


namespace NUMINAMATH_CALUDE_edges_after_cutting_l3846_384646

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  edges : ℕ

/-- Result of cutting off pyramids from a convex polyhedron. -/
def cutOffPyramids (P : ConvexPolyhedron) : ConvexPolyhedron :=
  ConvexPolyhedron.mk (3 * P.edges)

/-- Theorem stating the number of edges in the new polyhedron after cutting off pyramids. -/
theorem edges_after_cutting (P : ConvexPolyhedron) 
  (h : P.edges = 2021) : 
  (cutOffPyramids P).edges = 6063 := by
  sorry

end NUMINAMATH_CALUDE_edges_after_cutting_l3846_384646


namespace NUMINAMATH_CALUDE_probability_of_four_ones_in_twelve_dice_l3846_384674

def number_of_dice : ℕ := 12
def sides_per_die : ℕ := 6
def desired_ones : ℕ := 4

def probability_of_one : ℚ := 1 / sides_per_die
def probability_of_not_one : ℚ := 1 - probability_of_one

def binomial_coefficient (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

def probability_exact_ones : ℚ :=
  (binomial_coefficient number_of_dice desired_ones : ℚ) *
  (probability_of_one ^ desired_ones) *
  (probability_of_not_one ^ (number_of_dice - desired_ones))

theorem probability_of_four_ones_in_twelve_dice :
  probability_exact_ones = 495 * 390625 / 2176782336 :=
sorry

-- The following line is to show the approximate decimal value
#eval (495 * 390625 : ℚ) / 2176782336

end NUMINAMATH_CALUDE_probability_of_four_ones_in_twelve_dice_l3846_384674


namespace NUMINAMATH_CALUDE_unique_solution_is_zero_l3846_384699

theorem unique_solution_is_zero : 
  ∃! x : ℝ, (3 : ℝ) / (x - 3) = (5 : ℝ) / (x - 5) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_zero_l3846_384699


namespace NUMINAMATH_CALUDE_triangle_theorem_cosine_rule_sine_rule_l3846_384675

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the main theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : 3 * t.a * Real.cos t.A = t.c * Real.cos t.B + t.b * Real.cos t.C) :
  Real.cos t.A = 1/3 ∧ 
  (t.a = 1 ∧ Real.cos t.B + Real.cos t.C = 2 * Real.sqrt 3 / 3 → t.c = Real.sqrt 3 / 2) := by
  sorry

-- Define helper theorems for cosine and sine rules
theorem cosine_rule (t : Triangle) :
  2 * t.a * t.c * Real.cos t.B = t.a^2 + t.c^2 - t.b^2 := by
  sorry

theorem sine_rule (t : Triangle) :
  t.a / Real.sin t.A = t.b / Real.sin t.B := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_cosine_rule_sine_rule_l3846_384675


namespace NUMINAMATH_CALUDE_min_cubes_for_given_box_l3846_384642

/-- Calculates the minimum number of cubes required to build a box -/
def min_cubes_for_box (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

/-- Theorem stating the minimum number of cubes required for the given box -/
theorem min_cubes_for_given_box :
  min_cubes_for_box 8 15 5 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_given_box_l3846_384642


namespace NUMINAMATH_CALUDE_keiko_walking_speed_l3846_384636

/-- Keiko's walking speed around two rectangular tracks with semicircular ends -/
theorem keiko_walking_speed :
  ∀ (speed : ℝ) (width_A width_B time_diff_A time_diff_B : ℝ),
  width_A = 4 →
  width_B = 8 →
  time_diff_A = 48 →
  time_diff_B = 72 →
  (2 * π * width_A) / speed = time_diff_A →
  (2 * π * width_B) / speed = time_diff_B →
  speed = 2 * π / 5 :=
by sorry

end NUMINAMATH_CALUDE_keiko_walking_speed_l3846_384636


namespace NUMINAMATH_CALUDE_number_of_students_l3846_384680

/-- Proves the number of students in a class given average ages and teacher's age -/
theorem number_of_students (avg_age : ℝ) (teacher_age : ℝ) (new_avg_age : ℝ) :
  avg_age = 22 →
  teacher_age = 46 →
  new_avg_age = 23 →
  (avg_age * n + teacher_age) / (n + 1) = new_avg_age →
  n = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l3846_384680


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3846_384691

theorem algebraic_expression_value (p q : ℝ) :
  (2^3 * p + 2 * q + 1 = -2022) → ((-2)^3 * p + (-2) * q + 1 = 2024) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3846_384691


namespace NUMINAMATH_CALUDE_max_and_dog_same_age_in_dog_years_l3846_384660

/-- Conversion rate from human years to dog years -/
def human_to_dog_years : ℕ → ℕ := (· * 7)

/-- Max's age in human years -/
def max_age : ℕ := 3

/-- Max's dog's age in human years -/
def dog_age : ℕ := 3

/-- Theorem: Max and his dog have the same age when expressed in dog years -/
theorem max_and_dog_same_age_in_dog_years :
  human_to_dog_years max_age = human_to_dog_years dog_age :=
by sorry

end NUMINAMATH_CALUDE_max_and_dog_same_age_in_dog_years_l3846_384660


namespace NUMINAMATH_CALUDE_chime_1500_date_l3846_384673

/-- Represents a date with year, month, and day. -/
structure Date :=
  (year : Nat) (month : Nat) (day : Nat)

/-- Represents a time with hour and minute. -/
structure Time :=
  (hour : Nat) (minute : Nat)

/-- Represents the chiming pattern of the clock. -/
def chime_pattern (hour : Nat) (minute : Nat) : Nat :=
  if minute == 0 then hour
  else if minute == 15 || minute == 30 then 1
  else 0

/-- Calculates the number of chimes from a given start date and time to an end date and time. -/
def count_chimes (start_date : Date) (start_time : Time) (end_date : Date) (end_time : Time) : Nat :=
  sorry

/-- The theorem to be proved. -/
theorem chime_1500_date :
  let start_date := Date.mk 2003 2 28
  let start_time := Time.mk 18 30
  let end_date := Date.mk 2003 3 13
  count_chimes start_date start_time end_date (Time.mk 23 59) ≥ 1500 ∧
  count_chimes start_date start_time end_date (Time.mk 0 0) < 1500 :=
sorry

end NUMINAMATH_CALUDE_chime_1500_date_l3846_384673


namespace NUMINAMATH_CALUDE_marc_watching_friends_l3846_384648

theorem marc_watching_friends (total_episodes : ℕ) (watch_fraction : ℚ) (days : ℕ) : 
  total_episodes = 50 → 
  watch_fraction = 1 / 10 → 
  (total_episodes : ℚ) * watch_fraction = (days : ℚ) → 
  days = 10 := by
sorry

end NUMINAMATH_CALUDE_marc_watching_friends_l3846_384648


namespace NUMINAMATH_CALUDE_logarithmic_expression_equality_algebraic_expression_equality_l3846_384600

-- Part 1
theorem logarithmic_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - 25 ^ (Real.log 3 / Real.log 5) = -7 := by sorry

-- Part 2
theorem algebraic_expression_equality : 
  (9 / 4) ^ (1 / 2) - (-7.8) ^ 0 - (27 / 8) ^ (2 / 3) + (2 / 3) ^ (-2) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_logarithmic_expression_equality_algebraic_expression_equality_l3846_384600


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l3846_384626

theorem quadratic_inequality_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → x^2 + a*x + 9 ≥ 0) → a ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l3846_384626


namespace NUMINAMATH_CALUDE_roller_coaster_rides_l3846_384625

def initial_tickets : ℕ := 287
def spent_tickets : ℕ := 134
def earned_tickets : ℕ := 32
def cost_per_ride : ℕ := 17

theorem roller_coaster_rides : 
  (initial_tickets - spent_tickets + earned_tickets) / cost_per_ride = 10 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_rides_l3846_384625


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l3846_384668

/-- The problem of Jerry's action figures --/
theorem jerrys_action_figures 
  (final_count : ℕ) 
  (added_count : ℕ) 
  (h1 : final_count = 10) 
  (h2 : added_count = 2) :
  final_count - added_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l3846_384668


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3846_384623

theorem average_speed_calculation (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  total_distance = 80 ∧
  distance1 = 30 ∧
  speed1 = 30 ∧
  distance2 = 50 ∧
  speed2 = 50 →
  (total_distance / (distance1 / speed1 + distance2 / speed2)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3846_384623


namespace NUMINAMATH_CALUDE_backyard_max_area_l3846_384658

theorem backyard_max_area (P : ℝ) (h : P > 0) :
  let A : ℝ → ℝ → ℝ := λ l w => l * w
  let perimeter : ℝ → ℝ → ℝ := λ l w => l + 2 * w
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ perimeter l w = P ∧
    ∀ (l' w' : ℝ), l' > 0 → w' > 0 → perimeter l' w' = P →
      A l w ≥ A l' w' ∧
      A l w = (P / 4) ^ 2 ∧
      w = P / 4 :=
by sorry

end NUMINAMATH_CALUDE_backyard_max_area_l3846_384658


namespace NUMINAMATH_CALUDE_triangle_side_sum_max_l3846_384677

/-- In a triangle ABC, prove that given certain conditions, b + c has a maximum value of 6 --/
theorem triangle_side_sum_max (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧ 
  a = 3 ∧ 
  1 + (Real.tan A / Real.tan B) = (2 * c / b) ∧ 
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A → 
  b + c ≤ 6 ∧ ∃ b c, b + c = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_max_l3846_384677


namespace NUMINAMATH_CALUDE_bank_deposit_duration_l3846_384601

theorem bank_deposit_duration (initial_deposit : ℝ) (interest_rate : ℝ) (final_amount : ℝ) :
  initial_deposit = 5600 →
  interest_rate = 0.07 →
  final_amount = 6384 →
  (final_amount - initial_deposit) / (interest_rate * initial_deposit) = 2 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_duration_l3846_384601


namespace NUMINAMATH_CALUDE_units_digit_of_power_product_l3846_384664

theorem units_digit_of_power_product : 2^1201 * 4^1302 * 6^1403 ≡ 2 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_units_digit_of_power_product_l3846_384664


namespace NUMINAMATH_CALUDE_scientific_notation_1300000_l3846_384687

theorem scientific_notation_1300000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1300000 = a * (10 : ℝ) ^ n ∧ a = 1.3 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_1300000_l3846_384687


namespace NUMINAMATH_CALUDE_square_root_three_expansion_l3846_384666

theorem square_root_three_expansion {a b m n : ℕ+} :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2 →
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
by sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_l3846_384666


namespace NUMINAMATH_CALUDE_sourball_theorem_l3846_384655

/-- The number of sourball candies Nellie can eat before crying -/
def nellies_candies : ℕ := 12

/-- The initial number of candies in the bucket -/
def initial_candies : ℕ := 30

/-- The number of candies each person gets after dividing the remaining candies -/
def remaining_candies_per_person : ℕ := 3

/-- The number of sourball candies Jacob can eat before crying -/
def jacobs_candies (n : ℕ) : ℕ := n / 2

/-- The number of sourball candies Lana can eat before crying -/
def lanas_candies (n : ℕ) : ℕ := jacobs_candies n - 3

theorem sourball_theorem : 
  nellies_candies + jacobs_candies nellies_candies + lanas_candies nellies_candies = 
  initial_candies - 3 * remaining_candies_per_person := by
  sorry

end NUMINAMATH_CALUDE_sourball_theorem_l3846_384655


namespace NUMINAMATH_CALUDE_sum_of_triangle_ops_equals_21_l3846_384652

-- Define the triangle operation
def triangle_op (a b c : ℕ) : ℕ := a + b + c

-- Define the two triangles
def triangle1 : (ℕ × ℕ × ℕ) := (2, 4, 3)
def triangle2 : (ℕ × ℕ × ℕ) := (1, 6, 5)

-- Theorem statement
theorem sum_of_triangle_ops_equals_21 :
  triangle_op triangle1.1 triangle1.2.1 triangle1.2.2 +
  triangle_op triangle2.1 triangle2.2.1 triangle2.2.2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangle_ops_equals_21_l3846_384652


namespace NUMINAMATH_CALUDE_shaded_area_semicircle_arrangement_l3846_384678

/-- The area of the shaded region in a semicircle arrangement -/
theorem shaded_area_semicircle_arrangement (n : ℕ) (d : ℝ) (h : n = 8 ∧ d = 5) :
  let large_diameter := n * d
  let large_semicircle_area := π * (large_diameter / 2)^2 / 2
  let small_semicircle_area := n * (π * d^2 / 8)
  large_semicircle_area - small_semicircle_area = 175 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_semicircle_arrangement_l3846_384678


namespace NUMINAMATH_CALUDE_OPSQ_configurations_l3846_384676

structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩

def isCollinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

def isParallelogram (p q r s : Point) : Prop :=
  (q.x - p.x = s.x - r.x) ∧ (q.y - p.y = s.y - r.y)

theorem OPSQ_configurations (x₁ y₁ x₂ y₂ : ℝ) :
  let P : Point := ⟨x₁, y₁⟩
  let Q : Point := ⟨x₂, y₂⟩
  let S : Point := ⟨2*x₁, 2*y₁⟩
  (isCollinear O P Q ∨ 
   ¬(isCollinear O P Q) ∨ 
   isParallelogram O P S Q) := by sorry

end NUMINAMATH_CALUDE_OPSQ_configurations_l3846_384676


namespace NUMINAMATH_CALUDE_photo_arrangements_eq_288_l3846_384615

/-- The number of ways to arrange teachers and students in a photo. -/
def photoArrangements (numTeachers numMaleStudents numFemaleStudents : ℕ) : ℕ :=
  2 * (numMaleStudents.factorial * numFemaleStudents.factorial * (numFemaleStudents + 1).choose numMaleStudents)

/-- Theorem stating the number of photo arrangements under given conditions. -/
theorem photo_arrangements_eq_288 :
  photoArrangements 2 3 3 = 288 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_eq_288_l3846_384615


namespace NUMINAMATH_CALUDE_basketball_activity_results_l3846_384669

/-- Represents the outcome of a shot -/
inductive ShotResult
| Hit
| Miss

/-- Represents the game state -/
inductive GameState
| InProgress
| Cleared
| Failed

/-- Represents the possible coupon amounts -/
inductive CouponAmount
| Three
| Six
| Nine

/-- The shooting accuracy of Xiao Ming -/
def accuracy : ℚ := 2/3

/-- Updates the game state based on the current state and the new shot result -/
def updateGameState (state : GameState) (shot : ShotResult) : GameState :=
  sorry

/-- Simulates the game for a given number of shots -/
def simulateGame (n : ℕ) : GameState :=
  sorry

/-- Calculates the probability of ending the game after exactly 5 shots -/
def probEndAfterFiveShots : ℚ :=
  sorry

/-- Represents the distribution of the coupon amount -/
def couponDistribution : CouponAmount → ℚ :=
  sorry

/-- Calculates the expectation of the coupon amount -/
def expectedCouponAmount : ℚ :=
  sorry

theorem basketball_activity_results :
  probEndAfterFiveShots = 8/81 ∧
  couponDistribution CouponAmount.Three = 233/729 ∧
  couponDistribution CouponAmount.Six = 112/729 ∧
  couponDistribution CouponAmount.Nine = 128/243 ∧
  expectedCouponAmount = 1609/243 :=
sorry

end NUMINAMATH_CALUDE_basketball_activity_results_l3846_384669


namespace NUMINAMATH_CALUDE_complex_magnitude_l3846_384654

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 4 - 2 * Complex.I) : 
  Complex.abs z = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3846_384654


namespace NUMINAMATH_CALUDE_max_sum_of_squares_of_roots_l3846_384672

/-- The quadratic equation in question -/
def quadratic (a x : ℝ) : ℝ := x^2 + 2*a*x + 2*a^2 + 4*a + 3

/-- The sum of squares of roots of the quadratic equation -/
def sumOfSquaresOfRoots (a : ℝ) : ℝ := -8*a - 6

/-- The theorem stating the maximum sum of squares of roots and when it occurs -/
theorem max_sum_of_squares_of_roots :
  (∃ (a : ℝ), ∀ (b : ℝ), sumOfSquaresOfRoots b ≤ sumOfSquaresOfRoots a) ∧
  (sumOfSquaresOfRoots (-3) = 18) := by
  sorry

#check max_sum_of_squares_of_roots

end NUMINAMATH_CALUDE_max_sum_of_squares_of_roots_l3846_384672


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3846_384605

theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  (a = 4 ∧ b = 8) ∨ (a = 8 ∧ b = 4) →  -- two sides are 4cm and 8cm
  (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  a + b + c = 20 :=  -- perimeter is 20cm
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3846_384605


namespace NUMINAMATH_CALUDE_cube_root_problem_l3846_384622

theorem cube_root_problem : (0.07 : ℝ)^3 = 0.000343 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l3846_384622


namespace NUMINAMATH_CALUDE_squirrel_walnut_theorem_l3846_384657

/-- Calculates the final number of walnuts after squirrel activities -/
def final_walnut_count (initial : ℕ) (boy_gathered : ℕ) (boy_dropped : ℕ) (girl_brought : ℕ) (girl_ate : ℕ) : ℕ :=
  initial + boy_gathered - boy_dropped + girl_brought - girl_ate

/-- Theorem stating that given the squirrel activities, the final walnut count is 20 -/
theorem squirrel_walnut_theorem : 
  final_walnut_count 12 6 1 5 2 = 20 := by
  sorry

#eval final_walnut_count 12 6 1 5 2

end NUMINAMATH_CALUDE_squirrel_walnut_theorem_l3846_384657


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3846_384644

theorem line_segment_endpoint (x : ℝ) : x > 0 ∧ 
  Real.sqrt ((x - 2)^2 + (5 - 2)^2) = 8 → x = 2 + Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3846_384644


namespace NUMINAMATH_CALUDE_q4_value_l3846_384643

def sequence_a : ℕ → ℝ
| 0 => 1  -- We define a₁ = 1 based on the solution
| n + 1 => 2 * sequence_a n + 4

def sequence_q : ℕ → ℝ
| 0 => 17  -- We define q₁ = 17 to satisfy q₄ = 76
| n + 1 => 4 * sequence_q n + 8

theorem q4_value :
  sequence_a 4 = sequence_q 3 ∧ 
  sequence_a 6 = 316 → 
  sequence_q 3 = 76 := by
sorry

end NUMINAMATH_CALUDE_q4_value_l3846_384643


namespace NUMINAMATH_CALUDE_sets_equality_l3846_384639

def M : Set ℤ := {u : ℤ | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u : ℤ | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem sets_equality : M = N :=
sorry

end NUMINAMATH_CALUDE_sets_equality_l3846_384639


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3846_384637

theorem inequality_solution_set (x : ℝ) : 
  (1 - 2*x) / (x + 3) ≥ 1 ↔ -3 < x ∧ x ≤ -2/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3846_384637


namespace NUMINAMATH_CALUDE_system_solution_1_l3846_384696

theorem system_solution_1 (x y : ℝ) :
  x + y = 10^20 ∧ x - y = 10^19 → x = 55 * 10^18 ∧ y = 45 * 10^18 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_1_l3846_384696


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_eighth_l3846_384685

theorem fraction_sum_equals_one_eighth :
  (1 : ℚ) / 6 - 5 / 12 + 3 / 8 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_eighth_l3846_384685


namespace NUMINAMATH_CALUDE_energy_increase_with_center_charge_l3846_384630

/-- Represents the energy stored between two charges -/
structure EnergyBetweenCharges where
  charge1 : ℝ
  charge2 : ℝ
  distance : ℝ
  energy : ℝ
  proportionality : energy = (charge1 * charge2) / distance

/-- Configuration of charges on a square -/
structure SquareChargeConfiguration where
  sideLength : ℝ
  chargeValue : ℝ
  totalEnergy : ℝ

/-- Configuration with one charge moved to the center -/
structure CenterChargeConfiguration where
  sideLength : ℝ
  chargeValue : ℝ
  totalEnergy : ℝ

theorem energy_increase_with_center_charge 
  (initial : SquareChargeConfiguration)
  (final : CenterChargeConfiguration)
  (h1 : initial.totalEnergy = 20)
  (h2 : initial.sideLength = final.sideLength)
  (h3 : initial.chargeValue = final.chargeValue)
  : final.totalEnergy - initial.totalEnergy = 40 := by
  sorry

end NUMINAMATH_CALUDE_energy_increase_with_center_charge_l3846_384630


namespace NUMINAMATH_CALUDE_triangle_properties_l3846_384671

/-- Triangle ABC with vertices A(3,0), B(4,6), and C(0,8) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from point B to side AC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  fun p => 2 * p.1 - p.2 - 6 = 0

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := 13

theorem triangle_properties :
  let t : Triangle := { A := (3, 0), B := (4, 6), C := (0, 8) }
  (∀ p, altitude t p ↔ 2 * p.1 - p.2 - 6 = 0) ∧
  area t = 13 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3846_384671


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l3846_384697

/-- A hexagon formed by cutting a triangular corner from a square -/
structure CornerCutHexagon where
  sides : Fin 6 → ℕ
  is_valid_sides : (sides 0) + (sides 1) + (sides 2) + (sides 3) + (sides 4) + (sides 5) = 11 + 17 + 14 + 23 + 17 + 20

/-- The area of the hexagon -/
def hexagon_area (h : CornerCutHexagon) : ℕ :=
  sorry

/-- Theorem stating that the area of the specific hexagon is 1096 -/
theorem specific_hexagon_area : ∃ h : CornerCutHexagon, hexagon_area h = 1096 := by
  sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l3846_384697


namespace NUMINAMATH_CALUDE_expression_simplification_l3846_384663

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 - 2) :
  (m^2 - 4*m + 4) / (m - 1) / ((3 / (m - 1)) - m - 1) = (-3 + 4 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3846_384663


namespace NUMINAMATH_CALUDE_three_planes_division_l3846_384681

theorem three_planes_division (x y : ℕ) : 
  (x = 4 ∧ y = 8) → y - x = 4 := by
  sorry

end NUMINAMATH_CALUDE_three_planes_division_l3846_384681


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l3846_384621

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on a floor -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  let horizontal := (floor.length / tile.length) * (floor.width / tile.width)
  let vertical := (floor.length / tile.width) * (floor.width / tile.length)
  max horizontal vertical

/-- Theorem stating the maximum number of tiles on the given floor -/
theorem max_tiles_on_floor :
  let floor : Dimensions := ⟨100, 150⟩
  let tile : Dimensions := ⟨20, 30⟩
  maxTiles floor tile = 25 := by
  sorry

#eval maxTiles ⟨100, 150⟩ ⟨20, 30⟩

end NUMINAMATH_CALUDE_max_tiles_on_floor_l3846_384621


namespace NUMINAMATH_CALUDE_sum_of_max_min_is_zero_l3846_384662

def f (x : ℝ) : ℝ := x^2 - 2*x - 1

theorem sum_of_max_min_is_zero :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max + min = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_is_zero_l3846_384662


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3846_384665

theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 12 →
  b = 4 →
  b / c = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3846_384665


namespace NUMINAMATH_CALUDE_point_not_above_curve_l3846_384634

theorem point_not_above_curve :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 →
  ¬(b > a * b^3 - b * b^2) := by
  sorry

end NUMINAMATH_CALUDE_point_not_above_curve_l3846_384634


namespace NUMINAMATH_CALUDE_right_triangle_cube_sides_l3846_384613

theorem right_triangle_cube_sides : ∃ (x : ℝ), 
  let a := x^3
  let b := x^3 - x
  let c := x^3 + x
  a^2 + b^2 = c^2 ∧ a = 8 ∧ b = 6 ∧ c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cube_sides_l3846_384613


namespace NUMINAMATH_CALUDE_gcd_1989_1547_l3846_384631

theorem gcd_1989_1547 : Nat.gcd 1989 1547 = 221 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1989_1547_l3846_384631


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3846_384651

/-- 
For a quadratic equation x^2 + x + m = 0 with m ∈ ℝ, 
the condition "m > 1/4" is neither sufficient nor necessary for real roots.
-/
theorem quadratic_real_roots_condition (m : ℝ) : 
  ¬(∀ x : ℝ, x^2 + x + m = 0 → m > 1/4) ∧ 
  ¬(m > 1/4 → ∃ x : ℝ, x^2 + x + m = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3846_384651


namespace NUMINAMATH_CALUDE_total_calories_l3846_384650

/-- The number of calories in a single candy bar -/
def calories_per_bar : ℕ := 3

/-- The number of candy bars -/
def num_bars : ℕ := 5

/-- Theorem: The total calories in 5 candy bars is 15 -/
theorem total_calories : calories_per_bar * num_bars = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_calories_l3846_384650


namespace NUMINAMATH_CALUDE_max_value_of_g_l3846_384653

def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.sqrt 2 ∧
  g x = 25/8 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.sqrt 2 → g y ≤ g x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3846_384653


namespace NUMINAMATH_CALUDE_sin_70_degrees_l3846_384607

theorem sin_70_degrees (a : ℝ) (h : Real.sin (10 * π / 180) = a) : 
  Real.sin (70 * π / 180) = 1 - 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_70_degrees_l3846_384607


namespace NUMINAMATH_CALUDE_geometric_sequence_20th_term_l3846_384698

/-- A geometric sequence is defined by its first term and common ratio -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

/-- Theorem: In a geometric sequence where the 5th term is 5 and the 12th term is 1280, the 20th term is 2621440 -/
theorem geometric_sequence_20th_term 
  (a : ℝ) (r : ℝ) 
  (h1 : geometric_sequence a r 5 = 5)
  (h2 : geometric_sequence a r 12 = 1280) :
  geometric_sequence a r 20 = 2621440 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_20th_term_l3846_384698


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3846_384612

theorem solve_linear_equation (y : ℚ) (h : -3 * y - 9 = 6 * y + 3) : y = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3846_384612
